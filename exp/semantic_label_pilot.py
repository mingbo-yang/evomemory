"""Prepare matched inputs and independent semantic labels for a small pilot."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import time
from semantic_label_common import *

FOLDS=['train','development','temperature_calibration','threshold_calibration','test']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


def source_key(row):return ' '.join(row['input']['source'].casefold().split())


def candidates(fold):
    rows=read_jsonl(DATA/f'{fold}.jsonl')
    audit={r['id']:r for r in read_jsonl(DATA/'audit'/f'{fold}.jsonl')}
    result=[]
    for row in rows:
        if row['input']['current']==row['input']['candidate']:continue
        a=audit[row['id']]
        result.append({'id':'p'+row['id'][:12],'original_id':row['id'],'sample_id':a['sample_id'],
                       'input':row['input'],'bleu_label':row['label'],'bleu_delta':a['delta'],
                       'temperature':a['temperature'],'state_index':a['state_index']})
    return result


def pick(rows,n,salt,excluded=()):
    blocked=set(excluded);by_source=defaultdict(list)
    for r in rows:
        if source_key(r) not in blocked:by_source[source_key(r)].append(r)
    def key(s):return hashlib.sha256((salt+'|'+s).encode()).hexdigest()
    sources=sorted(by_source,key=key)[:n]
    assert len(sources)==n
    return [min(by_source[s],key=lambda r:key(r['id'])) for s in sources]


def prepare():
    if (OUT/'pilot_protocol.json').exists():
        print('Frozen pilot already prepared.');return
    reviewed=json.loads((ROOT/'runs/acceptance_data_v2/train_margin_review.json').read_text())
    reviewed_sources={' '.join(r['source'].casefold().split()) for r in reviewed}
    pool=candidates('train')
    test=pick([r for r in pool if r['temperature']==.1],96,'semantic-pilot-test-v1',reviewed_sources)
    train=pick(pool,512,'semantic-pilot-train-v1',reviewed_sources|{source_key(r) for r in test})
    folds={'train':train,'test':test}
    for name in FOLDS[1:-1]:folds[name]=pick(candidates(name),48,'semantic-pilot-'+name+'-v1')
    keys={k:{source_key(r) for r in v} for k,v in folds.items()}
    for a in FOLDS:
        for b in FOLDS:
            if a!=b:assert not keys[a]&keys[b]
    ids=[r['id'] for values in folds.values() for r in values]
    assert len(ids)==len(set(ids))
    stats={}
    for name,rows in folds.items():
        write_jsonl(OUT/'inputs'/f'{name}.jsonl',rows)
        stats[name]={'pairs':len(rows),'sources':len(keys[name]),'temperatures':dict(Counter(r['temperature'] for r in rows))}
    p={'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'purpose':'small paired comparison of BLEU-derived and semantic training labels, not a main experiment',
       'model':'convaiinnovations/laya-multilingual','fresh_initialization_per_arm':True,
       'arms':['bleu','semantic'],'statistics':stats,
       'training_teacher':'GLM-4-9B, separate family from Qwen3-8B generator',
       'evaluation_teacher':'Qwen3.8-27B, distinct from training teacher and original Qwen3-8B generator; not independent human gold',
       'presentation':'blind to reference, BLEU, old label and answer provenance; anonymous A/B, two orders',
       'consensus':'exact mapped agreement only; any disagreement remains Uncertain, including preference versus Tie',
       'training_filter':'semantic Uncertain excluded from BOTH arms; exactly same retained IDs and step budget',
       'evaluation_uncertainty':'report coverage over all sampled rows; metrics on resolved labels; acceptance precision bounds count accepted Uncertain explicitly',
       'sampling':'one changed pair per source; no feedback-based selection; low-temperature heldout test',
       'test_origin':'new pilot-held-out sources from old train pool, excluding previously reviewed 80 cases and all pilot train sources; older 20k checkpoint excluded from main comparison',
       'test_independence':'independent of both fresh pilot training arms; exploratory reuse of original collection, not a new end-to-end benchmark',
       'training':{'seed':42,'epochs':8,'effective_batch':32,'micro_batch':16,'lr_encoder':2.5e-5,'lr_head':1e-4,
                   'loss':'same RLCD+CE as previous run','class_weights':[1.,1.,1.],
                   'augmentation':'same deterministic 50% before/after swap and option shuffle in both arms'},
       'selection':'maximum semantic development macro F1; then lowest semantic development NLL; no test selection',
       'calibration':'scalar temperature on independent semantic probability fold; same semantic threshold fold for both arms',
       'threshold_rule':'maximum coverage with >=10% coverage and >=80% confirmed precision counting unresolved accepted pairs as not confirmed good, and positive worst-case semantic utility',
       'p_better_grid':[.34,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.98],
       'p_worse_max_grid':[1.,.3,.2,.1,.05],
       'test_read_guard':'test semantic labels accessible to evaluation only after both checkpoints, temperatures and thresholds frozen',
       'no_online_rollout':True,'no_human_annotation_claim':True,
       'source_dataset_sha256':sha(DATA/'dataset.json'),
       'input_hashes':{f:sha(OUT/'inputs'/f'{f}.jsonl') for f in FOLDS},
       'prompt_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest()}
    dump(OUT/'pilot_protocol.json',p)
    print(json.dumps(stats,ensure_ascii=False),flush=True)


def save_labels(fold,calls,teacher):
    rows=read_jsonl(OUT/'inputs'/f'{fold}.jsonl');result=[]
    for row in rows:
        first=calls.get((row['id'],False),{'winner':'Uncertain','reason':'missing/parse failure'})
        second=calls.get((row['id'],True),{'winner':'Uncertain','reason':'missing/parse failure'})
        result.append({'id':row['id'],'label':combine(first,second),'forward':first,'reverse':second,'teacher':teacher})
    write_jsonl(OUT/'labels'/f'{fold}.jsonl',result)
    status={'fold':fold,'completed':len(result),'total':len(rows),'state':'complete'}
    if fold!='test':status['label_counts']=dict(Counter(r['label'] for r in result))
    dump(OUT/f'{fold}_label_status.json',status)
    print(json.dumps(status,ensure_ascii=False),flush=True)


def label_glm():
    preflight=json.loads((OUT/'preflight_glm9.json').read_text())
    assert preflight['accuracy']>=.85
    assert all(r['correct'] or r['label']=='Uncertain' for r in preflight['rows'])
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    path='/home/ymb/glm_local/model';tok=AutoTokenizer.from_pretrained(path)
    llm=LLM(model=path,dtype='bfloat16',gpu_memory_utilization=.36,max_model_len=2048,max_num_seqs=64,
            max_num_batched_tokens=4096,enforce_eager=True,enable_prefix_caching=False)
    rows=read_jsonl(OUT/'inputs/train.jsonl');calls={};rawpath=OUT/'train_judge_calls.jsonl'
    if rawpath.exists():
        for r in read_jsonl(rawpath):calls[(r['id'],r['reverse'])]=r['judgment']
    requests=[(r,rev) for r in rows for rev in [False,True] if (r['id'],rev) not in calls]
    for offset in range(0,len(requests),64):
        batch=requests[offset:offset+64];texts=[]
        for row,rev in batch:
            texts.append(tok.apply_chat_template([{'role':'system','content':SYSTEM},
                {'role':'user','content':json.dumps([presented(row,rev)],ensure_ascii=False)}],tokenize=False,add_generation_prompt=True))
        assert max(len(tok.encode(t)) for t in texts)+192<=2048
        output=llm.generate(texts,SamplingParams(temperature=0,max_tokens=192,seed=42),use_tqdm=False)
        with rawpath.open('a') as handle:
            for (row,rev),obj in zip(batch,output):
                text=obj.outputs[0].text
                judgment=parse(text,{row['id']}).get(row['id'],{'winner':'Uncertain','reason':'parse failure'})
                calls[row['id'],rev]=judgment
                record={'id':row['id'],'reverse':rev,'judgment':judgment,'raw':text,
                        'finish_reason':obj.outputs[0].finish_reason,'output_tokens':len(obj.outputs[0].token_ids)}
                handle.write(json.dumps(record,ensure_ascii=False)+'\n')
        dump(OUT/'train_label_status.json',{'state':'running','calls_completed':len(calls),'calls_total':2*len(rows)})
        print('TRAIN_CALLS',len(calls),'/',2*len(rows),flush=True)
    save_labels('train',calls,'GLM-4-9B')


def label_evaluation():
    preflight=json.loads((OUT/'preflight_qwen27.json').read_text())
    assert preflight['accuracy']>=.90, 'Evaluation teacher failed obvious-case preflight'
    for fold in FOLDS[1:]:
        rows=read_jsonl(OUT/'inputs'/f'{fold}.jsonl');rawpath=OUT/f'{fold}_judge_calls.jsonl';calls={}
        if rawpath.exists():
            for r in read_jsonl(rawpath):calls[(r['id'],r['reverse'])]=r['judgment']
        for rev in [False,True]:
            pending=[r for r in rows if (r['id'],rev) not in calls]
            for offset in range(0,len(pending),8):
                batch=pending[offset:offset+8]
                result=request_local(8124,'qwen3.8-27b-local',batch,rev)
                with rawpath.open('a') as handle:
                    for row in batch:
                        judgment=result['parsed'].get(row['id'],{'winner':'Uncertain','reason':'parse failure'})
                        calls[row['id'],rev]=judgment
                        handle.write(json.dumps({'id':row['id'],'reverse':rev,'judgment':judgment},ensure_ascii=False)+'\n')
                with (OUT/f'{fold}_judge_batches.jsonl').open('a') as handle:
                    handle.write(json.dumps({'reverse':rev,'ids':[r['id'] for r in batch],**result},ensure_ascii=False)+'\n')
                dump(OUT/f'{fold}_label_status.json',{'state':'running','calls_completed':len(calls),'calls_total':2*len(rows)})
                print(fold,'CALLS',len(calls),'/',2*len(rows),flush=True)
        save_labels(fold,calls,'Qwen3.8-27B')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','label_train','label_evaluation'])
    stage=parser.parse_args().stage
    {'prepare':prepare,'label_train':label_glm,'label_evaluation':label_evaluation}[stage]()
