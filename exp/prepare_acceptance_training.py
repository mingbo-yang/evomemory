"""Build a versioned acceptance dataset without changing collection artifacts.

Labels here are deliberately named weak feedback labels. No claim of semantic
annotation is made. The held-out test remains unscored and unexported.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import acceptance_data as collection

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'runs/acceptance_data_v2'
FOLDS = ('train','development','temperature_calibration','threshold_calibration')
QUARANTINE = {
    'wmt19_en_zh/acceptance_v1/train/018316': 'Reference missing source facts and containing different statements; original parquet pair verified.',
    'wmt19_en_zh/acceptance_v1/train/032431': 'Reference omits the complete UK bond-yield sentence; original parquet pair verified.',
    'wmt19_en_zh/acceptance_v1/train/035677': 'Reference adds a complete Cameron/red-card clause absent from source; original parquet pair verified.',
}


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(s) for s in f if s.strip()]


def pair_id(row):
    raw = json.dumps([row['source'],row['before'],row['candidate']],ensure_ascii=False,separators=(',',':'))
    return hashlib.sha256(raw.encode()).hexdigest()


def classify(delta, epsilon):
    if not math.isfinite(delta) or not 0 < epsilon < 1:
        raise ValueError('delta must be finite and epsilon must be in (0,1)')
    return 'Better' if delta > epsilon else 'Worse' if delta < -epsilon else 'Tie'


def convert(row, epsilon=None):
    """Whitelist only inference-available fields into the model input."""
    item = {'id':pair_id(row), 'input':{'source':row['source'], 'current':row['before'], 'candidate':row['candidate']}}
    if epsilon is not None:
        item['label'] = classify(row['delta'],epsilon)
    return item


def swap_training_item(item):
    """Optional training-only augmentation; never apply to calibration/test."""
    result = {'id':item['id']+':swapped', 'input':{'source':item['input']['source'],
              'current':item['input']['candidate'], 'candidate':item['input']['current']}}
    if 'label' in item:
        result['label'] = {'Better':'Worse','Worse':'Better','Tie':'Tie'}[item['label']]
    return result


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def build(output, epsilon=None):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite dataset: {output}')
    collection.prepare()  # original manifest/library integrity
    review_path = DATA/'label_preparation/assistant_review.json'
    review = json.loads(review_path.read_text())
    assert collection.digest(DATA/'train_margin_review.json') == review['review_source_sha256']
    manifests = {fold:{r.sample_id:r for r in collection.read_manifest(DATA/f'{fold}_manifest.jsonl')} for fold in FOLDS}
    cleaned = {}; exclusions=[]
    for fold in FOLDS:
        rows = read_jsonl(DATA/'pairs'/f'{fold}.jsonl')
        kept=[]; seen=set()
        for row in rows:
            ref=manifests[fold][row['sample_id']]
            assert row['source']==ref.source
            assert math.isfinite(row['delta'])
            assert abs(row['delta']-(row['score_candidate']-row['score_before'])) < 1e-12
            key=pair_id(row)
            assert key not in seen, f'Duplicate exported pair {key}'
            seen.add(key)
            why = None
            if not row['valid']:
                why='invalid_candidate:'+row['invalid_reason']
            elif row['sample_id'] in QUARANTINE:
                assert fold=='train'
                why='reference_alignment:'+QUARANTINE[row['sample_id']]
            if why:
                exclusions.append({'id':key,'fold':fold,'sample_id':row['sample_id'],'reason':why})
            else:
                assert row['source'].strip() and row['before'].strip() and row['candidate'].strip()
                if row['before']==row['candidate']:
                    assert row['delta']==0
                kept.append(row)
        cleaned[fold]=kept
    source_keys={fold:{collection.norm(r['source']) for r in rows} for fold,rows in cleaned.items()}
    for a in FOLDS:
        for b in FOLDS:
            if a != b: assert not source_keys[a] & source_keys[b]
    audit=json.loads((DATA/'audit_summary.json').read_text())
    assert audit['cross_fold_overlap']==audit['memory_overlap']==0
    assert not (DATA/'pairs/test.jsonl').exists(), 'Test feedback should remain sealed'
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.staging.',dir=output.parent))
    try:
        statistics={}
        for fold,rows in cleaned.items():
            items=[convert(row,epsilon) for row in rows]
            write_jsonl(staging/f'{fold}.jsonl',items)
            # Provenance/feedback sidecars must never be passed as model features.
            write_jsonl(staging/'audit'/f'{fold}.jsonl',[
                {'id':pair_id(r),'sample_id':r['sample_id'],'state_index':r['state_index'],'slot':r['slot'],
                 'temperature':r['temperature'],'delta':r['delta'],'score_before':r['score_before'],
                 'score_candidate':r['score_candidate'],'identical':r['identical']} for r in rows])
            statistics[fold]={'pairs':len(rows),'sources':len(source_keys[fold]),
                              'changed_pairs':sum(not r['identical'] for r in rows),
                              'class_counts':dict(Counter(i['label'] for i in items)) if epsilon is not None else None}
            # Re-read persisted file and audit schema, order and label-input separation.
            persisted=read_jsonl(staging/f'{fold}.jsonl')
            assert len(persisted)==len(rows)
            for item,row in zip(persisted,rows):
                assert item==convert(row,epsilon)
                assert set(item['input'])=={'source','current','candidate'}
                assert set(item)==({'id','input','label'} if epsilon is not None else {'id','input'})
        write_jsonl(staging/'audit/exclusions.jsonl',exclusions)
        (staging/'test_sealed.json').write_text(json.dumps({
            'status':'sealed_until_model_and_thresholds_frozen',
            'raw_directory':str(DATA/'raw/test'),
            'manifest_sha256':collection.digest(DATA/'test_manifest.jsonl'),
            'sources':400,'unique_valid_pairs_from_generation_audit':793,
            'feedback_computed':False,'labels_exported':False},indent=2))
        metadata={'created_at_utc':datetime.now(timezone.utc).isoformat(),
                  'status':'ready_for_weak_feedback_training' if epsilon is not None else 'cleaned_pending_label_rule',
                  'label_semantics':'existing sentence feedback difference, NOT human or semantic ground truth',
                  'semantic_label_quality_validated':False,
                  'epsilon':epsilon,'feedback_scale':'Scorer.primary/100',
                  'decision_rule':'Better if delta>epsilon; Worse if delta< -epsilon; Tie otherwise',
                  'tie_definition':'within proxy-feedback margin; not a guarantee of semantic equivalence',
                  'input_schema':['source','current','candidate'],
                  'label_order':['Better','Tie','Worse'],
                  'training_augmentation':'none materialized; optional train-only swap helper; never change held-out distributions',
                  'class_weights':'not applied; use training counts only if weighting is later chosen',
                  'statistics':statistics,'exclusions':len(exclusions),'known_reference_issues':QUARANTINE,
                  'review_type':review['reviewer'],'review_hash':collection.digest(review_path),
                  'input_hashes':{f'{f}.jsonl':collection.digest(DATA/'pairs'/f'{f}.jsonl') for f in FOLDS},
                  'collection_protocol_sha256':collection.digest(DATA/'protocol.json'),
                  'exporter_sha256':collection.digest(Path(__file__)),
                  'checks':['original input hashes PASS','pair dedup PASS','numeric delta consistency PASS',
                            'source split isolation PASS','model feature whitelist PASS','test seal PASS',
                            'invalid/quarantined samples removed PASS','persisted round trip PASS'],
                  'output_hashes':{str(p.relative_to(staging)):collection.digest(p) for p in sorted(staging.rglob('*.json*'))}}
        (staging/'dataset.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        (staging/'README.md').write_text(
            '# Acceptance dataset\n\n'+metadata['status']+'\n\n'
            'Read dataset.json for the frozen label rule, class counts and integrity hashes. '
            'Model features are exactly input.source, input.current, input.candidate. '
            'Only the top-level label is a training target. Never concatenate audit/ sidecars into prompts. '
            'The feedback proxy has known lexical and reference-alignment noise; AI-assisted review is not human annotation. '
            'Only known training alignment problems were quarantined, not an exhaustive semantic cleaning of the corpus. '
            'Test remains sealed; no test labels or scores are included. No verifier is trained by this export.\n')
        staging.rename(output)
        return metadata
    except BaseException:
        shutil.rmtree(staging)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epsilon',type=float)
    args=p.parse_args()
    print(json.dumps(build(args.output,args.epsilon),ensure_ascii=False,indent=2))
