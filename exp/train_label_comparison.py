"""Same inputs, initialization, optimizer and budget; only training labels differ."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
from collections import Counter
from datetime import datetime,timezone
import json
import math
import random
import time
import numpy as np
import torch
import laya_acceptance_common as C
from train_laya_acceptance import rlcd_loss
from semantic_label_common import OUT,ROOT,LABELS,dump,read_jsonl


def status(stage,**kw):
    dump(OUT/'training_status.json',{'stage':stage,'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw})


def ready(fold):
    p=OUT/f'{fold}_label_status.json'
    return p.exists() and json.loads(p.read_text()).get('state')=='complete'


def read_fold(fold,allow_test=False):
    if fold=='test':assert allow_test and (OUT/'comparison_evaluation_lock.json').exists()
    assert ready(fold),fold+' semantic labeling is incomplete'
    rows=read_jsonl(OUT/'inputs'/f'{fold}.jsonl')
    lab={r['id']:r for r in read_jsonl(OUT/'labels'/f'{fold}.jsonl')}
    for row in rows:row['semantic_label']=lab[row['id']]['label']
    return rows


def resolved(rows,label='semantic_label'):
    return [{'id':r['id'],'input':r['input'],'label':r[label]} for r in rows if r['semantic_label']!='Uncertain']


def utility(rows):return [dict(Better=1.,Tie=0.,Worse=-1.)[r['label']] for r in rows]


def save(model,cfg,tok,path,arm,temperature=1.):
    C.save_checkpoint(model,cfg,tok,path,temperature)
    conf=json.loads((path/'rl_agent_config.json').read_text())
    conf['model_name']='laya-multilingual-label-comparison-v1-'+arm
    dump(path/'rl_agent_config.json',conf)


def train_arm(arm,train,dev,protocol):
    dest=OUT/arm;dest.mkdir(exist_ok=True)
    if (dest/'selection.json').exists():
        selection=json.loads((dest/'selection.json').read_text())
        prior=json.loads((dest/'training_data.json').read_text())
        assert prior['protocol_sha256']==C.digest(OUT/'pilot_protocol.json')
        assert prior['training_labels_sha256']==C.digest(OUT/'labels/train.jsonl')
        assert C.digest(dest/'best_uncalibrated/model.safetensors')==selection['model_sha256']
        print('REUSE_COMPLETED_ARM',arm,flush=True);return
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    tok=C.get_tokenizer();rows=resolved(train,'bleu_label' if arm=='bleu' else 'semantic_label')
    target_dev=resolved(dev)
    assert len(rows)>=128 and len(target_dev)>=10
    tr=C.EncodedPairs(rows,tok,allow_swap=True);de=C.EncodedPairs(target_dev,tok)
    model,cfg=C.load_model();model.to('cuda')
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.head_checkpointing=True
    for p in model.act_head.parameters():p.requires_grad_(False)
    enc=[p for name,p in model.named_parameters() if name.startswith('encoder.') and p.requires_grad]
    head=[p for name,p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
    opt=torch.optim.AdamW([{'params':enc,'lr':2.5e-5},{'params':head,'lr':1e-4}],weight_decay=.01)
    total=math.ceil(len(rows)/32)*8
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=total,eta_min=1e-6)
    history=[];best=None;best_state=None;updates=0;start=time.time()
    dump(dest/'training_data.json',{'n':len(rows),'ids':[r['id'] for r in rows],
         'label_counts':dict(Counter(r['label'] for r in rows)),'development_n':len(target_dev),
         'total_updates':total,'protocol_sha256':C.digest(OUT/'pilot_protocol.json'),
         'training_labels_sha256':C.digest(OUT/'labels/train.jsonl')})
    for epoch in range(8):
        model.train();idx=list(range(len(rows)));random.Random(42+epoch).shuffle(idx)
        ce_sum=loss_sum=seen=0.;sigma=.4-.3*epoch/7
        status('training',arm=arm,epoch=epoch+1,total_epochs=8,updates=updates,total_updates=total)
        for offset in range(0,len(idx),32):
            group=idx[offset:offset+32];opt.zero_grad(set_to_none=True)
            for start_idx in range(0,len(group),16):
                ids=group[start_idx:start_idx+16]
                batch=C.pack([tr.item(i,epoch=epoch) for i in ids],tok)
                loss,ce,_=rlcd_loss(model,batch,sigma,np.ones(3))
                assert torch.isfinite(loss)
                (loss*(len(ids)/len(group))).backward()
                loss_sum+=float(loss.detach())*len(ids);ce_sum+=float(ce.detach())*len(ids);seen+=len(ids)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm)
            opt.step();scheduler.step();updates+=1
        z=C.infer(model,de);m=C.metrics(target_dev,z,utility(target_dev))
        key=(m['macro_f1'],-m['nll'])
        entry={'epoch':epoch+1,'training_ce':ce_sum/seen,'training_loss':loss_sum/seen,
               'development':m,'selection_key':list(key),'elapsed_s':time.time()-start}
        history.append(entry);dump(dest/'history.json',history);np.save(dest/f'dev_epoch{epoch+1}.npy',z)
        if best is None or key>tuple(best['selection_key']):
            best=entry
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print('EPOCH',arm,epoch+1,'train_ce',entry['training_ce'],'dev_macro_f1',m['macro_f1'],'dev_auc',m['auc_better_changed'],flush=True)
    model.load_state_dict(best_state,strict=True)
    save(model,cfg,tok,dest/'best_uncalibrated',arm)
    dump(dest/'selection.json',{'epoch':best['epoch'],'selection_key':best['selection_key'],
         'development':best['development'],'model_sha256':C.digest(dest/'best_uncalibrated/model.safetensors')})
    del model,opt,best_state;torch.cuda.empty_cache()
    print('ARM_COMPLETE',arm,flush=True)


def run():
    torch.set_num_threads(4)
    protocol=json.loads((OUT/'pilot_protocol.json').read_text())
    for fold,h in protocol['input_hashes'].items():assert C.digest(OUT/'inputs'/f'{fold}.jsonl')==h
    while not (ready('train') and ready('development')):
        status('waiting_for_training_and_development_labels');time.sleep(10)
    train=read_fold('train');dev=read_fold('development')
    kept=[r for r in train if r['semantic_label']!='Uncertain']
    dump(OUT/'matched_training_summary.json',{'original_pairs':len(train),'matched_pairs':len(kept),
         'removed_from_both':len(train)-len(kept),'semantic_counts':dict(Counter(r['semantic_label'] for r in kept)),
         'bleu_counts_on_same_rows':dict(Counter(r['bleu_label'] for r in kept)),
         'retained_temperatures':dict(Counter(r['temperature'] for r in kept)),
         'label_agreement':sum(r['bleu_label']==r['semantic_label'] for r in kept)/len(kept)})
    for arm in ['bleu','semantic']:train_arm(arm,train,dev,protocol)
    a=json.loads((OUT/'bleu/training_data.json').read_text());b=json.loads((OUT/'semantic/training_data.json').read_text())
    assert a['ids']==b['ids'] and a['total_updates']==b['total_updates']
    status('both_arms_trained',matched_pairs=len(kept))


if __name__=='__main__':run()
