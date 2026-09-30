"""Check independent teachers on fixed obvious errors before choosing a labeler."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import json
import time
from semantic_label_common import *


def summarize(rows,calls):
    results=[]
    for row in rows:
        a=calls[0].get(row['id'],{'winner':'Uncertain','reason':'parse failure'})
        b=calls[1].get(row['id'],{'winner':'Uncertain','reason':'parse failure'})
        label=combine(a,b)
        results.append({'id':row['id'],'kind':row['kind'],'expected':row['label'],
                        'label':label,'correct':label==row['label'],'forward':a,'reverse':b})
    return {'n':len(rows),'accuracy':sum(r['correct'] for r in results)/len(rows),
            'uncertain':sum(r['label']=='Uncertain' for r in results),'rows':results}


def run(teacher):
    rows=json.loads((OLD/'investigation/synthetic_cases.json').read_text())
    OUT.mkdir(exist_ok=True);calls=[{},{}];raw=[];start=time.time()
    if teacher=='glm9':
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        path='/home/ymb/glm_local/model'
        tok=AutoTokenizer.from_pretrained(path)
        llm=LLM(model=path,dtype='bfloat16',gpu_memory_utilization=.36,max_model_len=2048,
                max_num_seqs=64,max_num_batched_tokens=4096,enforce_eager=True,
                enable_prefix_caching=False)
        texts=[];indices=[]
        for rev in [False,True]:
            for row in rows:
                messages=[{'role':'system','content':SYSTEM},
                          {'role':'user','content':json.dumps([presented(row,rev)],ensure_ascii=False)}]
                texts.append(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
                indices.append((int(rev),row))
        outputs=llm.generate(texts,SamplingParams(temperature=0,max_tokens=160,seed=42),use_tqdm=False)
        for (rev,row),output in zip(indices,outputs):
            text=output.outputs[0].text;parsed=parse(text,{row['id']})
            calls[rev].update(parsed);raw.append({'reverse':bool(rev),'id':row['id'],'raw':text})
    else:
        port,model={'qwen27':(8124,'qwen3.8-27b-local'),'glm47':(8123,'glm-4.7-flash-local')}[teacher]
        for rev in [False,True]:
            for offset in range(0,len(rows),8):
                batch=rows[offset:offset+8];result=request_local(port,model,batch,rev)
                calls[int(rev)].update(result['parsed']);raw.append({'reverse':rev,**result})
                print(teacher,rev,offset,len(result['parsed']),flush=True)
    result=summarize(rows,calls);result['elapsed_s']=time.time()-start;result['raw_calls']=raw
    dump(OUT/f'preflight_{teacher}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','raw_calls']}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('teacher',choices=['glm9','qwen27','glm47']);args=ap.parse_args()
    run(args.teacher)
