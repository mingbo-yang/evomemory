"""Fresh-source Qwen3-8B rollout with frozen Laya and delayed memory feedback."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['VLLM_NO_USAGE_STATS']='1'
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import core
from core import resolve_model_config
from core.manifest import _iter_wmt_train_pairs,read_manifest,SampleRef,sha256_text
from core.experience import load_experiences,save_experiences,render_experience_block
from core.bm25_fields import Experience
from core.scoring import Scorer
from core.optimized_pipeline import INSTRUCTIONS
from acceptance_data import norm,valid_candidate
from semantic_label_common import dump,read_jsonl,OUT as MODEL_OUT
from laya_relaxed_policy import choose_policy,select_candidate
import laya_acceptance_common as C

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/laya_relaxed_flow_v1'
TASK='wmt19_en_zh'


def stable_seed(*parts):
    return int(hashlib.sha256(('relaxed-flow-v1|'+'|'.join(map(str,parts))).encode()).hexdigest()[:8],16)%(2**31)


def status(stage,**kw):
    dump(OUT/'status.json',{'stage':stage,'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw})


def prepare():
    path=OUT/'protocol.json'
    if path.exists():
        p=json.loads(path.read_text())
        for name,h in p['input_hashes'].items():assert C.digest(OUT/name)==h
        assert C.digest(Path(p['policy']['checkpoint'])/'model.safetensors')==p['model_sha256']
        return p
    OUT.mkdir(parents=True,exist_ok=True);policy=choose_policy();blocked=set()
    paths=list((ROOT/'data/manifests').glob('*.jsonl'))+list((ROOT/'runs').rglob('*_manifest.jsonl'))
    for manifest in paths:
        if OUT in manifest.parents:continue
        for line in manifest.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            for key in ['source','reference']:
                if row.get(key):blocked.add(norm(row[key]))
    memory=ROOT/'runs/acceptance_data_v2/initial_memory.jsonl'
    library=load_experiences(memory)
    for e in library:
        blocked.update(norm(t) for t in [e.source_input,e.state_before,e.state_after] if t)
    pool=[];seen=set();excluded=Counter()
    for idx,source,ref in _iter_wmt_train_pairs('en_zh'):
        if idx<40000:continue
        if idx>=80000:break
        keys={norm(source),norm(ref)}
        if '' in keys or keys&blocked or keys&seen:excluded['overlap_or_empty']+=1;continue
        if not 4<=len(source.split())<=180:excluded['source_length']+=1;continue
        seen.update(keys);pool.append((idx,source,ref))
    pool.sort(key=lambda r:hashlib.sha256(('relaxed-fresh-v1|'+r[1]).encode()).hexdigest())
    assert len(pool)>=256
    refs=[SampleRef(f'{TASK}/relaxed_flow_v1/{i:06d}',TASK,'fresh_pilot',idx,source,ref,sha256_text(source),
           'WMT train rows 40000:80000; source/target normalized isolation from previous manifests and initial memory')
          for i,(idx,source,ref) in enumerate(pool[:256])]
    (OUT/'test_manifest.jsonl').write_text(''.join(json.dumps(asdict(r),ensure_ascii=False)+'\n' for r in refs))
    shutil.copy2(memory,OUT/'initial_memory.jsonl')
    shutil.copy2(ROOT/'runs/acceptance_data_v2/retrieval.json',OUT/'retrieval.json')
    dump(OUT/'acceptance_policy.json',policy)
    dump(MODEL_OUT/'semantic/relaxed_acceptance_policy.json',{k:v for k,v in policy.items() if k!='grid'})
    protocol={'created_at_utc':datetime.now(timezone.utc).isoformat(),'task':TASK,'sources':256,
       'generator':'qwen3-8b','generator_path':resolve_model_config('qwen3-8b').path,
       'verifier':'frozen laya-multilingual semantic/best','model_sha256':C.digest(Path(policy['checkpoint'])/'model.safetensors'),
       'policy':{k:v for k,v in policy.items() if k!='grid'},'gpu':0,'gpu_memory_utilization':.34,
       'max_rounds':3,'candidates_per_round':4,'temperature':.1,'top_p':1.,'max_tokens':1024,'max_model_len':4096,
       'batch_size':16,'retrieval_k':4,'alpha':.5,'renderer':'v2, neutral Refined Translation label for all memories',
       'arms':['initial','unfiltered_static','laya_static','conditional_laya_online'],
       'unfiltered_policy':'first basic-valid changed candidate; stop if none; no verifier',
       'laya_policy':'highest pBetter among valid changed candidates passing the frozen threshold; reject then stop',
       'drafts':'one shared freshly generated draft per source, reused exactly by all arms',
       'generation_cache':'identical prompt+system+seed+settings reused across arms; logical token budgets reported, cached wall time not a latency comparison',
       'online_continuation':'only if laya_static has >=1 accepted revision and corpus SacreBLEU exceeds the shared initial; exploratory continuation, CI need not exclude zero',
       'online_admission':'all unique basic-valid changed candidates with delayed legacy proxy feedback delta >1.0 point; acceptance independent; no current-task reference text in prompts or memory',
       'online_timing':'commit after all tasks in a 16-source batch finish; no feedback informs same-task decisions',
       'metrics':'primary corpus SacreBLEU tokenize=zh; secondary chrF++; legacy per-candidate delta is diagnostic/feedback only',
       'success_reporting':'report task-score deltas and source bootstrap CIs plus actual accepted-modification outcomes; no 80% prerequisite',
       'uncertainty':'single seed, small pilot, reference metrics imperfect; no retuning using these fresh outcomes',
       'initial_memory_size':len(library),'pool_size':len(pool),'exclusions':dict(excluded),
       'input_hashes':{n:C.digest(OUT/n) for n in ['test_manifest.jsonl','initial_memory.jsonl','retrieval.json','acceptance_policy.json']}}
    dump(path,protocol);return protocol


class Generator:
    def __init__(self,p):
        from baseline_core.llm import LLMClient
        from baseline_core.tasks import get_adapter
        self.client=LLMClient(resolve_model_config('qwen3-8b'),backend='vllm',gpu='0',
                    gpu_memory_utilization=p['gpu_memory_utilization'],max_model_len=4096,enforce_eager=True)
        self.adapter=get_adapter(TASK);self.p=p;self.calls=0;self.hits=0
        (OUT/'generation_cache').mkdir(exist_ok=True)

    def generate(self,requests):
        from vllm import SamplingParams
        result=[None]*len(requests);pending={}
        for i,r in enumerate(requests):
            payload={'prompt':r['prompt'],'system':self.adapter.system_prompt(),'seed':r['seed'],
                     'temperature':.1,'top_p':1.,'max_tokens':1024,'model':self.p['generator_path']}
            key=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            path=OUT/'generation_cache'/f'{key}.json'
            if path.exists():result[i]=json.loads(path.read_text());self.hits+=1;continue
            text=self.client._chat_text(payload['system'],payload['prompt']);n=self.client.count_tokens(text)
            if n+1024>4096:
                row={'text':'','finish_reason':'context_length_exceeded','input_tokens':n,'output_tokens':0,'seed':r['seed'],'cache_key':key}
                dump(path,row);result[i]=row;continue
            if key not in pending:pending[key]={'indices':[],'text':text,'request':r,'path':path}
            pending[key]['indices'].append(i)
        items=list(pending.items())
        if items:
            params=[SamplingParams(temperature=.1,top_p=1.,max_tokens=1024,seed=item['request']['seed'],
                     stop_token_ids=self.client.stop_token_ids or None) for _,item in items]
            outputs=self.client.model.generate([item['text'] for _,item in items],params,use_tqdm=False)
            assert len(outputs)==len(items);self.calls+=len(items)
            for (key,item),out in zip(items,outputs):
                g=out.outputs[0]
                row={'text':self.adapter.parse_output(g.text),'raw_text':g.text,'finish_reason':g.finish_reason,
                     'input_tokens':len(out.prompt_token_ids),'output_tokens':len(g.token_ids),'seed':item['request']['seed'],'cache_key':key}
                dump(item['path'],row)
                for i in item['indices']:result[i]=row
        assert all(r is not None for r in result)
        return result


class Verifier:
    def __init__(self,p):
        self.tok=C.get_tokenizer();self.model,self.cfg=C.load_model(Path(p['policy']['checkpoint']));self.model.to('cuda')
        self.temperature=self.cfg['temperature'][0];self.cache={};self.scored=0
        empty=C.EncodedPairs([],self.tok);self.overhead=max(len(v[0]) for v in empty.prefixes.values())+1

    def score(self,rows):
        pending={};out={}
        for r in rows:
            key=hashlib.sha256(json.dumps(r['input'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            if key in self.cache:out[r['id']]=self.cache[key];continue
            length=len(self.tok(C.state_text(r['input']).replace(self.tok.mask_token,' '),add_special_tokens=False)['input_ids'])+self.overhead
            if length>1024:out[r['id']]=None;continue
            pending.setdefault(key,{'row':dict(r,id=key),'ids':[]})['ids'].append(r['id'])
        if pending:
            items=list(pending.items());enc=C.EncodedPairs([v['row'] for _,v in items],self.tok)
            p=C.probabilities(C.infer(self.model,enc,batch_size=16),self.temperature);self.scored+=len(items)
            for (key,obj),probs in zip(items,p):
                self.cache[key]=probs.tolist()
                for ident in obj['ids']:out[ident]=self.cache[key]
        return out


def positive_memories(trace,reference,scorer,existing):
    """Called only after task termination; rejected positive candidates also qualify."""
    result=[]
    for rd in trace['rounds']:
        before=rd['before'];old=scorer.primary(reference,before)
        for candidate in rd['candidates']:
            if not candidate['valid'] or not candidate['changed']:continue
            after=candidate['text'];delta=scorer.primary(reference,after)-old
            if delta<=1.:continue
            key=hashlib.sha256(json.dumps([trace['source'],before,after],ensure_ascii=False).encode()).hexdigest()
            if key in existing:continue
            existing.add(key)
            result.append(Experience(exp_id='online-'+key,task=TASK,model='qwen3-8b',source_input=trace['source'],
                state_before=before,state_after=after,intervention_instruction=INSTRUCTIONS[TASK],
                intervention_rationale='Generated candidate with positive delayed benchmark feedback.',
                verdict='better',reason_a='',reason_b='',order_consistent=False,delta_offline=delta,
                provenance='relaxed_flow_v1_delayed_feedback',outcome_label='helped'))
    return result


class Flow:
    def __init__(self,p,generator,verifier):
        from core.hybrid_retrieval import LocalSentenceEncoder,HybridExperienceRetriever
        self.p=p;self.gen=generator;self.verifier=verifier;self.scorer=Scorer(TASK)
        cfg=json.loads((OUT/'retrieval.json').read_text());opts=dict(cfg['retrieval']);opts.pop('method')
        self.encoder=LocalSentenceEncoder(cfg['encoder'],'cpu');self.options=opts;self.retriever_class=HybridExperienceRetriever

    def drafts(self,refs):
        from baseline_core.types import TaskExample
        outputs=[]
        for start in range(0,len(refs),16):
            batch=refs[start:start+16]
            requests=[{'prompt':self.gen.adapter.initial_prompt(TaskExample(index=i,source=r.source,reference='',task=TASK)),
                       'seed':stable_seed(r.sample_id,'draft')} for i,r in enumerate(batch,start)]
            outputs.extend(self.gen.generate(requests));status('drafts',completed=len(outputs),total=len(refs))
        dump(OUT/'shared_drafts.json',outputs);return outputs

    def run(self,name,refs,drafts,online=False):
        from baseline_core.types import TaskExample
        from core.pipeline import build_refine_prompt
        lib=load_experiences(OUT/'initial_memory.jsonl');retriever=self.retriever_class(lib,self.encoder,**self.options)
        existing={hashlib.sha256(json.dumps([e.source_input,e.state_before,e.state_after],ensure_ascii=False).encode()).hexdigest() for e in lib}
        records=[];directory=OUT/name;directory.mkdir(exist_ok=True)
        for start in range(0,len(refs),16):
            batch=refs[start:start+16];path=directory/f'batch_{start:06d}.json'
            if path.exists():
                traces=json.loads(path.read_text())
            else:
                traces=[{'sample_id':r.sample_id,'source':r.source,'initial':d['text'],'final':d['text'],
                         'draft_valid':valid_candidate('',d)[0],'rounds':[],'bank_size_at_start':len(lib)}
                        for r,d in zip(batch,drafts[start:start+16])]
                active=[i for i,t in enumerate(traces) if t['draft_valid']]
                by_id={e.exp_id:e for e in lib}
                for round_index in range(1,4):
                    if not active:break
                    requests=[];rounds={}
                    for i in active:
                        tr=traces[i];before=tr['final'];ret=retriever.retrieve(tr['source'],before,alpha=.5,k=4,exclude_source=tr['source'])
                        block=render_experience_block([by_id[k] for k in ret.exp_ids],count_tokens=self.gen.client.count_tokens,
                             max_units=4,contrastive=True,advice_mode='summary').replace('Refined Translation (Gold):','Refined Translation:')
                        prompt=build_refine_prompt(self.gen.adapter,TaskExample(index=start+i,source=tr['source'],reference='',task=TASK),
                             before,INSTRUCTIONS[TASK],experience_block=block,renderer='v2')
                        rounds[i]={'round':round_index,'before':before,'retrieved_ids':ret.exp_ids,
                             'online_retrieved':sum(k.startswith('online-') for k in ret.exp_ids),'candidates':[],
                             'selected_slot':None,'accepted':False,'bank_size':len(lib)}
                        requests.extend({'prompt':prompt,'seed':stable_seed(tr['sample_id'],round_index,slot)} for slot in range(4))
                    generated=self.gen.generate(requests);score_rows=[]
                    for pos,i in enumerate(active):
                        rd=rounds[i]
                        for slot,g in enumerate(generated[4*pos:4*pos+4]):
                            valid,why=valid_candidate(rd['before'],g);changed=g['text']!=rd['before']
                            ident=f'{traces[i]["sample_id"]}:{round_index}:{slot}'
                            row={**g,'slot':slot,'valid':valid,'validity_reason':why,'changed':changed,'probabilities':None,'id':ident}
                            rd['candidates'].append(row)
                            if name!='unfiltered_static' and valid and changed:
                                score_rows.append({'id':ident,'input':{'source':traces[i]['source'],'current':rd['before'],'candidate':g['text']}})
                    scored=self.verifier.score(score_rows) if score_rows else {};next_active=[]
                    for i in active:
                        rd=rounds[i]
                        for c in rd['candidates']:
                            c['probabilities']=scored.get(c['id'])
                            if name!='unfiltered_static' and c['valid'] and c['changed'] and c['probabilities'] is None:
                                c['verifier_status']='context_overflow'
                        slot=select_candidate(rd['candidates'],self.p['policy'],'unfiltered' if name=='unfiltered_static' else 'laya')
                        if slot is not None:
                            rd['selected_slot']=slot;rd['accepted']=True;traces[i]['final']=rd['candidates'][slot]['text'];next_active.append(i)
                        traces[i]['rounds'].append(rd)
                    active=next_active
                dump(path,traces)
            records.extend(traces)
            if online:
                additions=[]
                for trace,ref in zip(traces,batch):
                    assert trace['sample_id']==ref.sample_id
                    additions.extend(positive_memories(trace,ref.reference,self.scorer,existing))
                lib.extend(additions)
                if additions:retriever=retriever.rebuild(lib)
                dump(directory/f'admission_{start:06d}.json',{'after_completed_sources':start+len(batch),'added':len(additions),
                    'new_ids':[e.exp_id for e in additions],'bank_size':len(lib),'positive_feedback_independent_of_acceptance':True})
            status('rollout',arm=name,completed=len(records),total=len(refs),bank_size=len(lib))
            print('ROLLOUT',name,len(records),'/',len(refs),'bank',len(lib),flush=True)
        if online:save_experiences(lib,directory/'final_memory.jsonl')
        dump(directory/'results.json',records);return records


def main():
    torch.set_num_threads(4);p=prepare();status('loading_generator')
    refs=read_manifest(OUT/'test_manifest.jsonl');gen=Generator(p);verifier=Verifier(p);flow=Flow(p,gen,verifier)
    drafts=flow.drafts(refs)
    static=flow.run('laya_static',refs,drafts)
    # Static task is fully finished before any fresh outcome is computed.
    initial=flow.scorer.score_corpus((r.reference,d['text']) for r,d in zip(refs,drafts))['bleu']
    final=flow.scorer.score_corpus((r.reference,t['final']) for r,t in zip(refs,static))['bleu']
    accepted=sum(rd['accepted'] for tr in static for rd in tr['rounds'])
    continuation=accepted>0 and final>initial
    dump(OUT/'static_continuation.json',{'initial_bleu':initial,'laya_static_bleu':final,'delta':final-initial,
                                      'accepted':accepted,'run_online':continuation,'rule':p['online_continuation']})
    flow.run('unfiltered_static',refs,drafts)
    if continuation:flow.run('laya_online',refs,drafts,online=True)
    dump(OUT/'execution_audit.json',{'actual_uncached_generation_calls':gen.calls,'cache_hits':gen.hits,
                                   'unique_verifier_pairs':verifier.scored,'primary_test_not_used_for_policy_tuning':True})
    status('generation_complete',online_ran=continuation)
    print('GENERATION_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=repr(exc));raise
