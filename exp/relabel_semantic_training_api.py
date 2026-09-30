"""Efficient anonymous semantic annotation using the existing local BF16 teacher."""
import json
import time
import requests
from semantic_label_common import *
from semantic_label_pilot import save_labels
from preflight_semantic_teachers import summarize

COMPACT=SYSTEM[:SYSTEM.index('对每个输入 id')]+'''按输入数组顺序输出一个 JSON 对象：{"winners":["A","Tie",...]}。winners 长度必须等于输入条数，每项只能是 A、B、Tie、Uncertain。不要输出理由、id 或其他内容。'''


def collect(rows,path,status_name):
    session=requests.Session();session.trust_env=False;calls={};start=time.monotonic()
    if path.exists():
        for r in read_jsonl(path):calls[r['id'],r['reverse']]=r['judgment']
    for rev in [False,True]:
        pending=[r for r in rows if (r['id'],rev) not in calls]
        for offset in range(0,len(pending),8):
            batch=pending[offset:offset+8];raw=None;parsed=None;last=None
            # Transport/format retries never select among conflicting valid judgments.
            for attempt in range(3):
                try:
                    response=session.post('http://127.0.0.1:8124/v1/chat/completions',json={
                        'model':'qwen3.8-27b-local','messages':[{'role':'system','content':COMPACT},
                          {'role':'user','content':json.dumps([presented(r,rev) for r in batch],ensure_ascii=False)}],
                        'temperature':0,'seed':42,'max_tokens':160,'chat_template_kwargs':{'enable_thinking':False},
                        'response_format':{'type':'json_schema','json_schema':{'name':'anonymous_preferences','strict':True,
                          'schema':{'type':'object','properties':{'winners':{'type':'array','items':{'type':'string','enum':['A','B','Tie','Uncertain']},'minItems':len(batch),'maxItems':len(batch)}},'required':['winners'],'additionalProperties':False}}}},timeout=180)
                    response.raise_for_status();obj=response.json();raw=obj['choices'][0]['message']['content']
                    decoder=json.JSONDecoder()
                    for i,c in enumerate(raw):
                        if c!='{':continue
                        try:parsed=decoder.raw_decode(raw[i:])[0]['winners'];break
                        except (ValueError,KeyError,TypeError):continue
                    assert isinstance(parsed,list) and len(parsed)==len(batch)
                    assert all(x in ['A','B','Tie','Uncertain'] for x in parsed)
                    break
                except Exception as exc:
                    last=repr(exc);parsed=None;time.sleep(2)
            if parsed is None:raise RuntimeError(f'Annotation request failed: {last}; raw={raw}')
            with path.open('a') as handle:
                for row,winner in zip(batch,parsed):
                    judgment={'winner':winner,'reason':'compact format: no rationale requested'}
                    calls[row['id'],rev]=judgment
                    handle.write(json.dumps({'id':row['id'],'reverse':rev,'judgment':judgment},ensure_ascii=False)+'\n')
            with (OUT/'compact_teacher_batches.jsonl').open('a') as handle:
                handle.write(json.dumps({'file':path.name,'reverse':rev,'ids':[r['id'] for r in batch],
                  'raw':raw,'usage':obj.get('usage')},ensure_ascii=False)+'\n')
            dump(OUT/status_name,{'state':'running','calls_completed':len(calls),'calls_total':2*len(rows),'elapsed_s':time.monotonic()-start})
            print(path.name,len(calls),'/',2*len(rows),time.monotonic()-start,flush=True)
    return calls


def run():
    sanity=json.loads((OLD/'investigation/synthetic_cases.json').read_text())
    c=collect(sanity,OUT/'preflight_qwen27_compact_calls.jsonl','compact_preflight_status.json')
    result=summarize(sanity,[{r['id']:c[r['id'],rev] for r in sanity} for rev in [False,True]])
    dump(OUT/'preflight_qwen27_compact.json',result);assert result['accuracy']>=.9,result
    audit=json.loads((OUT/'blind_training_audit_inputs.json').read_text())
    c=collect(audit,OUT/'qwen27_compact_audit_calls.jsonl','compact_audit_status.json')
    reference={r['id']:r for r in json.loads((OUT/'blind_training_audit_completed.json').read_text())}
    result=[{'id':r['id'],'reviewer_label':reference[r['id']]['label'],
             'teacher_label':combine(c[r['id'],False],c[r['id'],True])} for r in audit]
    dump(OUT/'qwen27_training_audit.json',{'reviewer':'Codex AI assisted, not human gold',
         'sampling':'24 random and 8 GLM-label-selected supplements; not population noise estimate',
         'agreement':sum(r['reviewer_label']==r['teacher_label'] for r in result)/len(result),'rows':result})
    critical={r['id']:r['teacher_label'] for r in result}
    assert critical['p26ff21606642']=='Better','Teacher still rewards hallucinated entity'
    assert critical['p3ecc9c8cf095']=='Worse','Teacher misses obvious untranslated word'
    rows=read_jsonl(OUT/'inputs/train.jsonl')
    calls=collect(rows,OUT/'train_qwen27_compact_calls.jsonl','train_label_status.json')
    save_labels('train',calls,'Qwen3.8-27B local BF16, compact 8-pair format')


if __name__=='__main__':run()
