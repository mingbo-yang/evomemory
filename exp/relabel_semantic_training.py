"""Replace a failed teacher, without changing sampled inputs or inspecting test labels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['VLLM_NO_USAGE_STATS']='1'
import json
import time
from collections import Counter
from semantic_label_common import *
from semantic_label_pilot import save_labels
from preflight_semantic_teachers import summarize


def run():
    from vllm import LLM,SamplingParams
    start=time.monotonic()
    llm=LLM(model='/mnt/huawei/ymb/model/Qwen3.8-27B',dtype='bfloat16',
            language_model_only=True,max_model_len=2048,max_num_seqs=32,
            max_num_batched_tokens=2048,gpu_memory_utilization=.39,
            cpu_offload_gb=30,kv_cache_memory_bytes=5*1024**3,
            enforce_eager=True,enable_prefix_caching=False,async_scheduling=False,
            generation_config='vllm')
    tok=llm.get_tokenizer()

    def collect(rows,path,status_name):
        calls={}
        if path.exists():
            for r in read_jsonl(path):calls[r['id'],r['reverse']]=r['judgment']
        pending=[(r,rev) for r in rows for rev in [False,True] if (r['id'],rev) not in calls]
        for offset in range(0,len(pending),64):
            batch=pending[offset:offset+64];texts=[]
            for row,rev in batch:
                messages=[{'role':'system','content':SYSTEM},
                          {'role':'user','content':json.dumps([presented(row,rev)],ensure_ascii=False)}]
                texts.append(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
            assert max(len(tok.encode(t)) for t in texts)+160<=2048
            output=llm.generate(texts,SamplingParams(temperature=0,max_tokens=160,seed=42),use_tqdm=False)
            with path.open('a') as handle:
                for (row,rev),obj in zip(batch,output):
                    raw=obj.outputs[0].text
                    judgment=parse(raw,{row['id']}).get(row['id'],{'winner':'Uncertain','reason':'parse failure'})
                    calls[row['id'],rev]=judgment
                    handle.write(json.dumps({'id':row['id'],'reverse':rev,'judgment':judgment,'raw':raw,
                         'finish_reason':obj.outputs[0].finish_reason},ensure_ascii=False)+'\n')
            dump(OUT/status_name,{'state':'running','calls_completed':len(calls),'calls_total':2*len(rows),
                                 'teacher':'Qwen3.8-27B local offline BF16','elapsed_s':time.monotonic()-start})
            print(path.name,len(calls),'/',2*len(rows),'elapsed',time.monotonic()-start,flush=True)
        return calls

    sanity=json.loads((OLD/'investigation/synthetic_cases.json').read_text())
    calls=collect(sanity,OUT/'preflight_qwen27_offline_calls.jsonl','offline_preflight_status.json')
    result=summarize(sanity,[{r['id']:calls[r['id'],rev] for r in sanity} for rev in [False,True]])
    dump(OUT/'preflight_qwen27_offline.json',result)
    assert result['accuracy']>=.9, result
    audit=json.loads((OUT/'blind_training_audit_inputs.json').read_text())
    calls=collect(audit,OUT/'qwen27_training_audit_calls.jsonl','offline_audit_status.json')
    reference={r['id']:r for r in json.loads((OUT/'blind_training_audit_completed.json').read_text())}
    audited=[{'id':r['id'],'reviewer_label':reference[r['id']]['label'],
              'teacher_label':combine(calls[r['id'],False],calls[r['id'],True]),
              'forward':calls[r['id'],False],'reverse':calls[r['id'],True]} for r in audit]
    dump(OUT/'qwen27_training_audit.json',{'reviewer':'Codex AI assisted, not human gold',
          'sampling':'random 24 plus GLM directional/uncertain supplements; not population noise estimate',
          'agreement':sum(r['reviewer_label']==r['teacher_label'] for r in audited)/len(audited),'rows':audited})
    rows=read_jsonl(OUT/'inputs/train.jsonl')
    calls=collect(rows,OUT/'train_qwen27_judge_calls.jsonl','train_label_status.json')
    save_labels('train',calls,'Qwen3.8-27B offline BF16, single pair per request')
    print('RELABEL_COMPLETE',time.monotonic()-start,flush=True)


if __name__=='__main__':run()
