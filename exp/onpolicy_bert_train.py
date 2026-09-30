"""Train prospective feedback models; generation must finish first."""
import os
os.environ['BERT_STUDY_GPU']='0'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import json,math,random,gc,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata,spearmanr
from transformers import BertTokenizerFast
import bert_feedback_study as old
from onpolicy_bert_data import OUT,ROOT,normalized,dump
from core.manifest import read_manifest
from core.scoring import Scorer


def assemble():
    scorer=Scorer('wmt19_en_zh');data={};stats={}
    for fold in ['train','validation','test']:
        paths=list((OUT/'collection').glob(f'**/{fold}/optimized-*/full_online.jsonl'))
        assert len(paths)==1,(fold,paths)
        traces=[json.loads(line) for line in paths[0].read_text().splitlines()]
        refs={r.sample_id:r for r in read_manifest(OUT/f'{fold}_manifest.jsonl')}
        assert len(traces)==len(refs) and {r['sample_id'] for r in traces}==set(refs)
        import audit_optimized_flow as auditor
        original_reader=auditor.read_manifest
        try:
            auditor.read_manifest=lambda path:list(refs.values())
            audit=auditor.audit(paths[0],OUT/'library','accumulation')
        finally:
            auditor.read_manifest=original_reader
        dump(OUT/f'{fold}_flow_audit.json',audit)
        unique={};round_count=0;identical=0;empty=0
        for tr in traces:
            ref=refs[tr['sample_id']];before=tr['initial_draft']
            for rd in tr['rounds']:
                if rd['controller_action']!='REFINE':continue
                after=rd['candidate'];round_count+=1
                y0=scorer.primary(ref.reference,before)/100;y1=scorer.primary(ref.reference,after)/100
                assert abs((y1-y0)*100-rd['delta_offline'])<1e-6
                if not after.strip():empty+=1
                else:
                    identical+=int(before==after)
                    key=(ref.source,before,after)
                    unique[key]=[ref.source,before,after,y0,y1]
                if rd['accepted']:before=after
            assert before==tr['final_output']
        data[fold]=list(unique.values())
        delta=np.array([r[4]-r[3] for r in data[fold]])
        stats[fold]={'sources':len(refs),'refine_rounds':round_count,'empty_rounds':empty,'identical_rounds':identical,'unique_pairs':len(unique),'changed_unique_pairs':sum(r[1]!=r[2] for r in data[fold]),'class_counts':np.bincount(labels(data[fold]),minlength=3).tolist(),'mean_abs_gain':float(np.abs(delta).mean())}
    sets={f:{normalized(r[0]) for r in rows} for f,rows in data.items()}
    assert all(not sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    dump(OUT/'data.json',data);dump(OUT/'data_statistics.json',stats)
    print('ASSEMBLED '+json.dumps(stats),flush=True)
    return data

def labels(rows):
    # 0=degrade, 1=tie, 2=improve. Delta is current metric on a [0,1] scale.
    delta=np.array([r[4]-r[3] for r in rows])
    return np.where(delta>.005,2,np.where(delta<-.005,0,1)).astype(np.int64)

def eligible(rows):
    return np.array([bool(r[2].strip()) and r[1]!=r[2] and len(r[2])<=1.02*max(1,len(r[1]))+8 for r in rows])

def metrics(scores,rows,threshold):
    delta=np.array([r[4]-r[3] for r in rows]);changed=np.array([r[1]!=r[2] for r in rows]);positive=delta>1e-6;negative=delta< -1e-6
    mask=changed&(positive|negative);np_=int(positive[mask].sum());nn_=int(negative[mask].sum())
    auc=float((rankdata(scores[mask])[positive[mask]].sum()-np_*(np_+1)/2)/(np_*nn_)) if np_ and nn_ else None
    accept=eligible(rows)&(scores>threshold) if threshold is not None else np.zeros(len(rows),dtype=bool)
    n=int(accept.sum());count=int(changed.sum())
    rho=float(spearmanr(scores[changed],delta[changed]).statistic) if count>1 and np.ptp(scores[changed])>0 and np.ptp(delta[changed])>0 else None
    return {'unique_pairs':len(rows),'changed_pairs':count,'auc_changed_non_tie':auc,'gain_spearman_changed':rho if rho is None or math.isfinite(rho) else None,'threshold':threshold,'accepted':n,'precision':float(positive[accept].mean()) if n else None,'improve':int((accept&positive).sum()),'degrade':int((accept&negative).sum()),'tie':int((accept&~positive&~negative).sum()),'coverage_changed':n/max(count,1),'net_gain_per_changed_pair':float(delta[accept].sum()/max(count,1)),'mean_gain_accepted':float(delta[accept].mean()) if n else None}

def calibrate(scores,rows,arm):
    grid=[0,.005,.01,.02,.03,.05,.075,.1,.15,.2] if arm!='joint' else [.4,.5,.6,.7,.8,.85,.9,.95,.98]
    table=[metrics(scores,rows,t) for t in grid]
    valid=[m for m in table if m['accepted']>=20 and m['precision']>=.8 and m['net_gain_per_changed_pair']>0]
    best=max(valid,key=lambda m:m['accepted']) if valid else None
    return best['threshold'] if best else None,table

class Joint(old.Model):
    def __init__(self,pretrained=True):
        super().__init__(pretrained)
        self.classifier[-1]=nn.Linear(512,3)

def joint_batch(rows,tok,swap_rng=None):
    rows=[list(r) for r in rows]
    if swap_rng is not None:
        for r in rows:
            if swap_rng.random()<.5:r[1],r[2],r[3],r[4]=r[2],r[1],r[4],r[3]
    encoded=tok([r[0] for r in rows],[f'Before: {r[1]}\nAfter: {r[2]}' for r in rows],padding=True,truncation=True,max_length=512,return_tensors='pt')
    return {k:v.cuda() for k,v in encoded.items()},torch.tensor(labels(rows),device='cuda')

@torch.inference_mode()
def predict(model,rows,tok,arm):
    if arm!='joint':
        predictions=old.predict(model,rows,tok)
        return predictions[:,1]-predictions[:,0]
    model.eval();output=[]
    for i in range(0,len(rows),32):
        inputs,_=joint_batch(rows[i:i+32],tok)
        output.extend(model(**inputs).softmax(-1)[:,2].cpu().numpy())
    return np.array(output)

def train(data,arm,tok):
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=(Joint() if arm=='joint' else old.Model()).cuda()
    optim=torch.optim.AdamW([{'params':model.bert.parameters(),'lr':2e-5},{'params':model.classifier.parameters(),'lr':1e-3}],weight_decay=.01)
    batches=math.ceil(len(data['train'])/16);scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda step:max(0,1-step/(5*batches)))
    # Swap augmentation balances improve/degrade in expectation.
    counts=np.bincount(labels(data['train']),minlength=3).astype(float)
    counts[0]=counts[2]=(counts[0]+counts[2])/2
    weight=torch.tensor(np.sqrt(counts.sum()/np.maximum(counts,1)),dtype=torch.float32,device='cuda');weight/=weight.mean()
    best=-1.;history=[]
    for epoch in range(5):
        model.train();loss_sum=0.;started=time.time();rng=random.Random(100+epoch)
        order=np.random.default_rng(42+epoch).permutation(len(data['train']))
        for i in range(0,len(order),16):
            batch=[data['train'][j] for j in order[i:i+16]]
            inputs,target=joint_batch(batch,tok,rng) if arm=='joint' else old.tensor_batch(batch,tok)
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(**inputs).float()
                loss=nn.functional.cross_entropy(pred,target,weight=weight) if arm=='joint' else nn.functional.mse_loss(pred.reshape(-1,2),target)
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();scheduler.step()
            loss_sum+=loss.item()*len(batch)
        scores=predict(model,data['validation'],tok,arm)
        m=metrics(scores,data['validation'],.5 if arm=='joint' else .01)
        threshold,_=calibrate(scores,data['validation'],arm)
        entry={'epoch':epoch+1,'train_loss':loss_sum/len(order),'elapsed_s':time.time()-started,'validation':m,'calibrated_threshold':threshold}
        history.append(entry);dump(OUT/f'{arm}_learning_curve.json',history)
        print(arm+' EPOCH '+json.dumps(entry),flush=True)
        if m['auc_changed_non_tie']>best:
            best=m['auc_changed_non_tie'];torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},OUT/f'{arm}_best.pth');dump(OUT/f'{arm}_selection.json',{'epoch':epoch+1,'validation_auc':best})
    del optim,scheduler,model;gc.collect();torch.cuda.empty_cache()

def evaluate(data,arm,tok):
    checkpoint=old.OLD if arm=='legacy' else ROOT/'runs/bert_feedback_study/absolute_best.pth' if arm=='historical_absolute' else OUT/f'{arm}_best.pth'
    model=Joint(False) if arm=='joint' else old.Model(False)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.cuda()
    validation=predict(model,data['validation'],tok,arm)
    threshold,grid=calibrate(validation,data['validation'],arm)
    # Threshold is frozen before reading test predictions.
    test=predict(model,data['test'],tok,arm)
    np.savez(OUT/f'{arm}_predictions.npz',validation=validation,test=test)
    result={'calibrated_threshold':threshold,'calibration_succeeded':threshold is not None,'validation_search':grid,'test_default':metrics(test,data['test'],.5 if arm=='joint' else .01),'test_calibrated':metrics(test,data['test'],threshold)}
    m=result['test_calibrated'];result['rollout_gate_passed']=m['accepted']>=20 and m['precision']>=.8 and m['net_gain_per_changed_pair']>0
    dump(OUT/f'{arm}_evaluation.json',result)
    print(arm+' FINAL '+json.dumps(result['test_calibrated']),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    torch.set_num_threads(4)
    data=assemble()
    tok=BertTokenizerFast.from_pretrained(old.BASE,local_files_only=True)
    token_audit={}
    for fold,rows in data.items():
        joint_lengths=[len(ids) for ids in tok([r[0] for r in rows],[f'Before: {r[1]}\nAfter: {r[2]}' for r in rows],truncation=False)['input_ids']]
        token_audit[fold]={'pairs':len(rows),'joint_over_512':sum(n>512 for n in joint_lengths),'joint_max_tokens':max(joint_lengths),'joint_median_tokens':float(np.median(joint_lengths))}
    dump(OUT/'tokenization_audit.json',token_audit)
    for arm in ['absolute','joint']:
        if not (OUT/f'{arm}_evaluation.json').exists():train(data,arm,tok);evaluate(data,arm,tok)
    for arm in ['legacy','historical_absolute']:
        if not (OUT/f'{arm}_evaluation.json').exists():evaluate(data,arm,tok)
    print('ONPOLICY STUDY COMPLETE',flush=True)
