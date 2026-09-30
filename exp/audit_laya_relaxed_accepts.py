"""Reference-blind descriptive audit of every accepted Laya modification."""
from collections import Counter
import hashlib
import json
import time
from pathlib import Path
from legacy_laya_v1.laya_relaxed_flow import OUT
from semantic_label_common import request_local,combine,dump,read_jsonl


def audit(arm):
    traces=json.loads((OUT/arm/'results.json').read_text());rows=[]
    for tr in traces:
        for rd in tr['rounds']:
            if not rd['accepted']:continue
            candidate=rd['candidates'][rd['selected_slot']]['text']
            ident='a'+hashlib.sha256((arm+'|'+tr['sample_id']+'|'+str(rd['round'])).encode()).hexdigest()[:12]
            rows.append({'id':ident,'sample_id':tr['sample_id'],'round':rd['round'],
              'input':{'source':tr['source'],'current':rd['before'],'candidate':candidate}})
    calls={};path=OUT/arm/'accepted_semantic_calls.jsonl'
    if path.exists():
        for c in read_jsonl(path):calls[c['id'],c['reverse']]=c['judgment']
    for rev in [False,True]:
        pending=[r for r in rows if (r['id'],rev) not in calls]
        for start in range(0,len(pending),8):
            batch=pending[start:start+8]
            for attempt in range(3):
                try:result=request_local(8124,'qwen3.8-27b-local',batch,rev);break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2)
            with path.open('a') as f:
                for row in batch:
                    judgment=result['parsed'].get(row['id'],{'winner':'Uncertain','reason':'parse failure'})
                    calls[row['id'],rev]=judgment
                    f.write(json.dumps({'id':row['id'],'reverse':rev,'judgment':judgment},ensure_ascii=False)+'\n')
            print('AUDIT',arm,rev,start+len(batch),'/',len(pending),flush=True)
    labeled=[{**r,'label':combine(calls[r['id'],False],calls[r['id'],True]),
              'forward':calls[r['id'],False],'reverse':calls[r['id'],True]} for r in rows]
    counts=Counter(r['label'] for r in labeled);n=len(rows)
    report={'arm':arm,'teacher':'Qwen3.8-27B local; same teacher family/checkpoint used in classifier supervision; not independent human gold',
       'reference_and_probability_blind':True,'n':n,'counts':dict(counts),
       'confirmed_improvement_lower':counts['Better']/n if n else None,
       'possible_improvement_upper':(counts['Better']+counts['Uncertain'])/n if n else None,
       'net_confirmed_better_minus_worse':counts['Better']-counts['Worse'],'rows':labeled}
    dump(OUT/arm/'accepted_semantic_audit.json',report)


def main():
    while not (OUT/'laya_static/results.json').exists():time.sleep(5)
    audit('laya_static')
    while True:
        state=json.loads((OUT/'status.json').read_text())
        if state.get('stage')=='failed':raise RuntimeError(state)
        if (OUT/'laya_online/results.json').exists():audit('laya_online');break
        cont=OUT/'static_continuation.json'
        if cont.exists() and not json.loads(cont.read_text())['run_online']:break
        time.sleep(5)
    dump(OUT/'semantic_audit_status.json',{'stage':'complete'})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(OUT/'semantic_audit_status.json',{'stage':'failed','error':repr(exc)});raise
