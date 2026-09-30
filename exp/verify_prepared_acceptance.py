"""Independent verification of the exported weak-feedback data, without test scoring."""
import argparse
from collections import Counter
import json
from pathlib import Path
from datetime import datetime, timezone
import acceptance_data as collection
from core.scoring import Scorer
from prepare_acceptance_training import read_jsonl, QUARANTINE


def verify(path):
    info=json.loads((path/'dataset.json').read_text())
    protocol=json.loads((collection.OUT/'label_preparation/weak_feedback_v1_protocol.json').read_text())
    assert info['epsilon']==protocol['epsilon']==.01
    collection.prepare()
    for filename,digest in info['output_hashes'].items():
        assert collection.digest(path/filename)==digest,filename
    scorer=Scorer('wmt19_en_zh')
    summary={}; normalized_sets={}
    for fold in ['train','development','temperature_calibration','threshold_calibration']:
        refs={r.sample_id:r for r in collection.read_manifest(collection.OUT/f'{fold}_manifest.jsonl')}
        items=read_jsonl(path/f'{fold}.jsonl')
        traces={r['id']:r for r in read_jsonl(path/'audit'/f'{fold}.jsonl')}
        assert len(items)==len(traces)
        seen=set();counts=Counter();score_cache={}
        for row in items:
            assert set(row)=={'id','input','label'}
            assert set(row['input'])=={'source','current','candidate'}
            assert row['id'] not in seen;seen.add(row['id'])
            trace=traces[row['id']]; ref=refs[trace['sample_id']]
            assert trace['sample_id'] not in QUARANTINE
            assert ref.source==row['input']['source']
            def score(text):
                key=(trace['sample_id'],text)
                if key not in score_cache:score_cache[key]=scorer.primary(ref.reference,text)/100
                return score_cache[key]
            old,new=score(row['input']['current']),score(row['input']['candidate'])
            assert abs(old-trace['score_before'])<1e-12
            assert abs(new-trace['score_candidate'])<1e-12
            delta=new-old
            expected='Tie'
            if delta > .01:expected='Better'
            if delta < -.01:expected='Worse'
            assert row['label']==expected
            counts[row['label']]+=1
        assert dict(counts)==info['statistics'][fold]['class_counts']
        normalized_sets[fold]={collection.norm(v) for row in items for v in
                              (row['input']['source'],refs[traces[row['id']]['sample_id']].reference)}
        summary[fold]={'pairs':len(items),'labels':dict(counts),'recomputed_feedback_and_labels':'PASS'}
    normalized_sets['test']={collection.norm(v) for r in collection.read_manifest(collection.OUT/'test_manifest.jsonl') for v in (r.source,r.reference)}
    for a in normalized_sets:
        for b in normalized_sets:
            if a!=b:assert not normalized_sets[a]&normalized_sets[b]
    assert not (path/'test.jsonl').exists()
    assert not (collection.OUT/'pairs/test.jsonl').exists()
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'dataset':str(path.resolve()),
            'status':'PASS','semantic_label_quality_validated':False,
            'test_feedback_scored':False,'fold_source_and_reference_isolation':'PASS',
            'feature_whitelist':'PASS','artifact_hashes':'PASS','folds':summary}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();result=verify(args.dataset);collection.dump(args.report,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
