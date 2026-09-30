"""Adapters around the pinned official Laya multilingual decision model."""
import os
os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parent.parent
VENDOR=ROOT/'vendor/laya'
sys.path.insert(0,str(VENDOR))
BASE=Path('/mnt/huawei/ymb/model/laya-multilingual')
OUT=Path('/mnt/huawei/ymb/model/laya-multilingual-acceptance-enzh-v1')
DATA=ROOT/'runs/acceptance_data_v2/prepared/weak_feedback_v1'
COLLECTION=ROOT/'runs/acceptance_data_v2'
LABELS=['Better','Tie','Worse']
CRITERIA={'Better':'The candidate is meaningfully better.',
          'Tie':'The candidate is equivalent or differs only trivially.',
          'Worse':'The candidate is meaningfully worse.'}
QUESTION={'t':'choice','ins':'Compared with the current answer, how should the candidate answer be classified for English-to-Chinese translation?', 'crit':CRITERIA}
SEED=42


def read_jsonl(path):
    with path.open() as f:return [json.loads(s) for s in f if s.strip()]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def seed_for(*xs):return int(hashlib.sha256('|'.join(map(str,xs)).encode()).hexdigest()[:8],16)


def state_text(inputs,swap=False):
    a,b=inputs['current'],inputs['candidate']
    if swap:a,b=b,a
    return f"Task: English-to-Chinese Translation\n\nSource:\n{inputs['source']}\n\nCurrent Answer:\n{a}\n\nCandidate Answer:\n{b}"


def semantic_target(label,swap=False):
    result=LABELS.index(label)
    return 2-result if swap else result


def get_tokenizer():
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    _fix_tokenizer_config(str(BASE))
    return AutoTokenizer.from_pretrained(BASE/'tokenizer',local_files_only=True)


def load_model(path=BASE):
    from laya.common import build_model
    from safetensors.torch import load_file
    cfg=json.loads((path/'rl_agent_config.json').read_text())
    assert cfg['encoder']=='jhu-clsp/mmBERT-base', 'Must use laya-multilingual, not English Laya'
    model=build_model(cfg,encoder_dir=str(path/'encoder'),pretrained=False)
    model.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
    model.encoder.config.reference_compile=False
    return model,cfg


class EncodedPairs:
    def __init__(self,rows,tok,allow_swap=False):
        from laya.common import build_sequence
        self.rows=rows;self.tok=tok;self.prefixes={}
        for order in itertools.permutations(range(3)):
            ids,markers=build_sequence(tok,'',QUESTION,1024,256,option_order=list(order),state_ids=[])
            self.prefixes[order]=(ids[:-1],markers)
        self.states=[];self.max_len=0
        # Only the whitelisted input is serialized; labels and audit feedback are separate.
        for r in rows:
            texts=[state_text(r['input'])]
            if allow_swap:texts.append(state_text(r['input'],True))
            ids=[tok(t.replace(tok.mask_token,' '),add_special_tokens=False)['input_ids'] for t in texts]
            self.states.append(ids)
            for st in ids:
                length=max(len(p[0]) for p in self.prefixes.values())+len(st)+1
                assert length<=1024, f'Input truncation would occur: {r["id"]} {length}'
                self.max_len=max(self.max_len,length)

    def item(self,i,epoch=None,force_order=None):
        row=self.rows[i]
        rng=random.Random(seed_for(SEED,row['id'],epoch))
        swap=epoch is not None and rng.random()<.5
        order=list(range(3))
        if epoch is not None:rng.shuffle(order)
        if force_order is not None:order=list(force_order)
        prefix,markers=self.prefixes[tuple(order)]
        ids=prefix+self.states[i][int(swap)]+[self.tok.sep_token_id]
        item={'ids':ids,'markers':markers,'qtype':0,'order':order,'row_index':i,'swapped':swap}
        if 'label' in row:
            semantic=semantic_target(row['label'],swap)
            y=order.index(semantic)
            item.update(label=y,target=[float(j==y) for j in range(3)],semantic_label=semantic)
        return item


def pack(items,tok,device='cuda'):
    from laya.common import collate_items
    batch=collate_items([[i] for i in items],tok.pad_token_id)
    return {k:v.to(device) if hasattr(v,'to') else v for k,v in batch.items()}


def forward(model,batch):
    return model(batch['input_ids'],batch['attention_mask'],batch['marker_pos'],batch['marker_mask'],batch['qtype'])


def probabilities(logits,temperature=1.0):
    import numpy as np
    z=logits/temperature;z=z-z.max(axis=1,keepdims=True)
    q=np.exp(z);return q/q.sum(axis=1,keepdims=True)


def infer(model,encoded,batch_size=32,force_order=None):
    import numpy as np
    import torch
    model.eval();out=np.zeros((len(encoded.rows),3),dtype=np.float32)
    # Sort by length only to reduce padding; restore original row order.
    order=sorted(range(len(encoded.rows)),key=lambda i:len(encoded.states[i][0]))
    with torch.no_grad():
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size]
            items=[encoded.item(i,force_order=force_order) for i in indices]
            batch=pack(items,encoded.tok)
            with torch.autocast('cuda',dtype=torch.bfloat16):z,_=forward(model,batch)
            z=z.float().cpu().numpy()
            for j,(idx,item) in enumerate(zip(indices,items)):
                for pos,semantic in enumerate(item['order']):out[idx,semantic]=z[j,pos]
    assert np.isfinite(out).all()
    return out


def allowed(row):
    a,b=row['input']['current'],row['input']['candidate']
    # Same basic hard length constraint as collection; no absolute quality gate.
    import unicodedata
    def n(t):return len(''.join(unicodedata.normalize('NFKC',t).casefold().split()))
    return a!=b and bool(b.strip()) and n(b)<=1.5*n(a)+8


def accepted_mask(rows,p,threshold):
    import numpy as np
    allowed_mask=np.array([allowed(r) for r in rows])
    if threshold is None:return np.zeros(len(rows),dtype=bool)
    return allowed_mask & (p[:,0]>=threshold['p_better']) & (p[:,2]<=threshold['p_worse_max'])


def acceptance_stats(rows,p,delta,threshold):
    import numpy as np
    mask=accepted_mask(rows,p,threshold);n=int(mask.sum())
    gain=np.asarray(delta)[mask]
    return {'accepted':n,'coverage':n/len(rows),'precision':float(np.mean(gain>.01)) if n else None,
            'mean_delta':float(gain.mean()) if n else None,'net_delta_per_pair':float(gain.sum()/len(rows)),
            'positive':int((gain>.01).sum()),'negative':int((gain<-.01).sum()),'tie':int((abs(gain)<=.01).sum()),
            'any_negative':int((gain<0).sum())}


def metrics(rows,z,delta,temperature=1.0,threshold=None):
    import numpy as np
    from sklearn.metrics import roc_auc_score,f1_score,log_loss
    p=probabilities(z,temperature);y=np.array([LABELS.index(r['label']) for r in rows]);pred=p.argmax(1)
    confidence=p.max(1);correct=(pred==y)
    ece=0.
    buckets=np.minimum((confidence*15).astype(int),14)
    for bucket in range(15):
        m=buckets==bucket
        if m.any():ece+=m.mean()*abs(confidence[m].mean()-correct[m].mean())
    changed=np.array([r['input']['current']!=r['input']['candidate'] for r in rows])
    def auc(mask):
        yy=y[mask]==0
        return float(roc_auc_score(yy,p[mask,0])) if len(set(yy))==2 else None
    result={'n':len(rows),'accuracy':float(correct.mean()),'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'auc_better_all':auc(np.ones(len(rows),bool)),'auc_better_changed':auc(changed),
            'brier':float(((p-np.eye(3)[y])**2).sum(1).mean()),'ece_15':float(ece),
            'nll':float(log_loss(y,p,labels=[0,1,2])),
            'default_acceptance':acceptance_stats(rows,p,delta,{'p_better':.5,'p_worse_max':1.0}),
            'calibrated_acceptance':acceptance_stats(rows,p,delta,threshold)}
    eligible=[i for i,r in enumerate(rows) if allowed(r)]
    eligible.sort(key=lambda i:(-float(p[i,0]),rows[i]['id']))
    top=eligible[:min(len(eligible),math.ceil(.1*len(rows)))]
    result['top10pct']={'n':len(top),'precision':float(np.mean(y[top]==0)) if top else 0.,
                        'mean_delta':float(np.asarray(delta)[top].mean()) if top else 0.}
    return result


def selection_key(result):
    return (result['top10pct']['precision'],result['top10pct']['mean_delta'],result['auc_better_changed'] or 0.)


def save_checkpoint(model,cfg,tok,path,temperature=1.0):
    import torch
    from safetensors.torch import save_file
    path.mkdir(parents=True,exist_ok=True)
    sd={k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()}
    sd['temperature']=torch.tensor([temperature,1.,1.])
    save_file(sd,str(path/'model.safetensors'))
    model.encoder.config.save_pretrained(path/'encoder')
    tok.save_pretrained(path/'tokenizer')
    cfg=dict(cfg);cfg.update(max_len=1024,head_max_len=256,temperature=[temperature,1.,1.],fine_tuned=True,
                            model_name='laya-multilingual-acceptance-enzh-v1',amp_dtype='bf16')
    cfg.pop('temperature_by_options',None)
    dump(path/'rl_agent_config.json',cfg)


def load_fold(fold):
    rows=read_jsonl(DATA/f'{fold}.jsonl')
    traces={r['id']:r for r in read_jsonl(DATA/'audit'/f'{fold}.jsonl')}
    return rows,[traces[r['id']]['delta'] for r in rows],[traces[r['id']]['sample_id'] for r in rows]
