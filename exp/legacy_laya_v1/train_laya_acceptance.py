"""Single-GPU adaptation of the official Laya RLCD notebook, with isolated evaluation."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time
import numpy as np
import torch
from safetensors.torch import load_file
from legacy_laya_v1.laya_acceptance_common import *
from laya.common import proper_reward


def now():return datetime.now(timezone.utc).isoformat()


def status(stage,**kw):dump(OUT/'status.json',{'stage':stage,'updated_at_utc':now(),**kw})


def freeze_protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((DATA/'dataset.json').read_text())
    for name,h in meta['output_hashes'].items():assert digest(DATA/name)==h
    p={'model':'convaiinnovations/laya-multilingual','base_revision':json.loads((BASE/'download_provenance.json').read_text())['revision'],
       'vendor_commit':subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip(),
       'dataset_manifest_sha256':digest(DATA/'dataset.json'),'train_pairs':20002,'label_epsilon':.01,
       'feedback_semantics':'weak original sentence-metric feedback; semantic reliability NOT validated',
       'gpu':0,'seed':42,'epochs':4,'micro_batch':16,'effective_batch':64,
       'lr_encoder':2.5e-5,'lr_head':1e-4,'weight_decay':.01,'max_len':1024,'head_max_len':256,
       'training':'official Laya model + proper_reward + GRPO logit perturbation (G=4,sigma=.4 to .1) + CE guidance=1; single GPU bf16',
       'sampling':'all train pairs once per epoch; shuffled; 50% before/after swap with label reversal; random option permutation',
       'class_balance':'square-root inverse effective class-frequency loss weights, symmetric Better/Worse after swapping; no heldout resampling',
       'checkpoint_selection':'development precision among top ceil(10% of all dev pairs) eligible pBetter ranks; tie-break mean delta then changed-pair AUC',
       'probability_calibration':'only temperature_calibration; scalar T minimizes NLL in [.1,10]',
       'threshold_calibration':'only threshold_calibration; max coverage satisfying >=10% coverage, >=80% precision(delta>.01), positive mean delta',
       'p_better_grid':[.34,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.98],
       'p_worse_max_grid':[1.,.3,.2,.1,.05],
       'eligibility':'nonidentical, nonempty candidate, normalized length<=1.5*current+8',
       'test_gate':'coverage>=.10 AND precision>=.80 AND mean delta>0 AND source-bootstrap 95% lower mean delta>0; zero acceptance fails',
       'failure_policy':'if verifier gate fails, do not run static/online main experiment',
       'test_access':'only after selected checkpoint, scalar temperatures and operating points are persisted in evaluation_lock.json',
       'controls':['unmodified laya-multilingual','fine-tuned laya-multilingual'],
       'precision_reporting':'Better requires delta>.01; also report any-negative accept count and metric gain',
       'script_sha256':digest(Path(__file__)),'adapter_sha256':digest(ROOT/'laya_acceptance_common.py')}
    path=OUT/'training_protocol.json'
    if path.exists():assert json.loads(path.read_text())==p,'Protocol changed; use a new version'
    else:dump(path,p)
    return p


def rlcd_loss(model,batch,sigma,weights):
    with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=forward(model,batch)
    logits=logits.float();mask=batch['marker_mask'];target=batch['target'];k=mask.sum(-1,keepdim=True).float()
    eps=torch.randn((4,)+logits.shape,device='cuda')*sigma*mask
    eps=(eps-eps.sum(-1,keepdim=True)/k)*mask
    z=logits.detach().unsqueeze(0)+eps
    q=torch.softmax(z.masked_fill(~mask,-1e4),-1)
    with torch.no_grad():
        reward=proper_reward(q,target.unsqueeze(0),batch['qtype'],mask,w_sph=.75,w_rps=1.)
        adv=reward-reward.mean(0,keepdim=True)
        adv=adv/(adv.std()+1e-6)
    logp=-(((z-logits.unsqueeze(0))**2)*mask).sum(-1)/(2*sigma**2)
    loss_rl=-(adv*logp).mean(0)
    loss_ce=-(target*torch.log_softmax(logits.masked_fill(~mask,-1e4),-1)).sum(-1)
    w=torch.tensor([weights[it['semantic_label']] for it in batch['meta']],device='cuda')
    return ((loss_rl+loss_ce)*w).mean(),loss_ce.mean(),reward.mean()


def train(smoke=False):
    p=freeze_protocol()
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    tok=get_tokenizer()
    rows,delta,_=load_fold('train');dev,dev_delta,_=load_fold('development')
    status('encoding')
    tr=EncodedPairs(rows[:64] if smoke else rows,tok,allow_swap=True)
    de=EncodedPairs(dev[:16] if smoke else dev,tok)
    counts=Counter(r['label'] for r in rows)
    f=np.array([(counts['Better']+counts['Worse'])/2,counts['Tie'],(counts['Better']+counts['Worse'])/2])
    weights=1/np.sqrt(f);weights=weights/np.sum(weights*f/f.sum())
    dump(OUT/('smoke_encoding.json' if smoke else 'encoding_audit.json'),{'max_train_tokens':tr.max_len,'max_dev_tokens':de.max_len,
               'truncated':0,'class_weights':weights.tolist(),'parameter_source':p['base_revision']})
    model,cfg=load_model();model.to('cuda')
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.head_checkpointing=True
    # The unused escalation head is not part of this acceptance-only experiment.
    for param in model.act_head.parameters():param.requires_grad_(False)
    enc=[v for name,v in model.named_parameters() if name.startswith('encoder.') and v.requires_grad]
    head=[v for name,v in model.named_parameters() if not name.startswith('encoder.') and v.requires_grad]
    opt=torch.optim.AdamW([{'params':enc,'lr':p['lr_encoder']},{'params':head,'lr':p['lr_head']}],weight_decay=p['weight_decay'])
    total_steps=math.ceil(len(tr.rows)/64)*(1 if smoke else p['epochs'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total_steps,eta_min=1e-6)
    if smoke:
        model.train();before=model.scorer[-1].weight.detach().clone()
        batch=pack([tr.item(i,epoch=0) for i in range(8)],tok)
        loss,ce,reward=rlcd_loss(model,batch,.4,weights)
        assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(norm)
        opt.step()
        assert not torch.equal(before,model.scorer[-1].weight)
        z=infer(model,de)
        result={'loss':float(loss.detach()),'ce':float(ce.detach()),'reward':float(reward),'grad_norm':float(norm),
                'weights_updated':True,'finite_predictions':bool(np.isfinite(z).all()),
                'parameter_count':sum(t.numel() for t in model.parameters()),'cuda_peak_GB':torch.cuda.max_memory_allocated()/1e9}
        dump(OUT/'smoke_result.json',result);print(json.dumps(result),flush=True)
        return
    if (OUT/'best_selection.json').exists():raise RuntimeError('Training already has results; avoid overwriting an experiment')
    base_z=infer(model,de)
    base_metrics=metrics(dev,base_z,dev_delta)
    dump(OUT/'base_development.json',base_metrics)
    np.save(OUT/'base_development_logits.npy',base_z)
    history=[];best_key=None;start=time.time();updates=0
    for epoch in range(p['epochs']):
        model.train();indices=list(range(len(tr.rows)));random.Random(SEED+epoch).shuffle(indices)
        sigma=.4+(.1-.4)*epoch/max(1,p['epochs']-1)
        ce_sum=loss_sum=seen=0.
        status('training',epoch=epoch+1,total_epochs=p['epochs'],updates=updates)
        for start_idx in range(0,len(indices),64):
            group=indices[start_idx:start_idx+64];opt.zero_grad(set_to_none=True)
            for sub in range(0,len(group),16):
                ids=group[sub:sub+16];batch=pack([tr.item(i,epoch=epoch) for i in ids],tok)
                loss,ce,reward=rlcd_loss(model,batch,sigma,weights)
                assert torch.isfinite(loss),'Nonfinite training loss'
                (loss*(len(ids)/len(group))).backward()
                ce_sum+=float(ce.detach())*len(ids);loss_sum+=float(loss.detach())*len(ids);seen+=len(ids)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(grad_norm),'Nonfinite gradients'
            opt.step();scheduler.step();updates+=1
            if updates%25==0:
                progress={'epoch':epoch+1,'total_epochs':p['epochs'],'updates':updates,'total_updates':total_steps,
                          'examples_seen_epoch':int(seen),'train_ce':ce_sum/seen,'train_loss':loss_sum/seen,
                          'elapsed_s':time.time()-start,'cuda_peak_GB':torch.cuda.max_memory_allocated()/1e9}
                status('training',**progress);print(json.dumps(progress),flush=True)
        z=infer(model,de);result=metrics(dev,z,dev_delta);key=selection_key(result)
        entry={'epoch':epoch+1,'ce':ce_sum/seen,'loss':loss_sum/seen,'development':result,'elapsed_s':time.time()-start}
        history.append(entry);dump(OUT/'history.json',history)
        np.save(OUT/f'development_logits_epoch{epoch+1}.npy',z)
        latest=OUT/'checkpoint_latest';save_checkpoint(model,cfg,tok,latest)
        torch.save({'epoch':epoch+1,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'updates':updates},latest/'training_state.pt')
        if best_key is None or key>best_key:
            best_key=key;save_checkpoint(model,cfg,tok,OUT/'best_uncalibrated')
            dump(OUT/'best_selection.json',{'epoch':epoch+1,'selection_key':list(key),'development':result,
                                          'checkpoint_sha256':digest(OUT/'best_uncalibrated/model.safetensors')})
        print('EPOCH_COMPLETE '+json.dumps(entry),flush=True)
    status('training_complete',epochs=p['epochs'],elapsed_s=time.time()-start)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    train(args.smoke)
