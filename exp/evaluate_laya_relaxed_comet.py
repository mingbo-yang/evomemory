"""Independent, post-hoc COMET audit of the frozen 256-source rollout.

This script never changes a checkpoint, threshold, generated answer or
continuation decision. COMET scores are descriptive and must not tune Laya.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/laya_relaxed_flow_v1'
OUT = RUN / 'comet_posthoc_v1'
COMET_DIR = Path('/mnt/huawei/ymb/model/wmt22-comet-da')
ARMS = ('initial', 'laya_static', 'laya_bleu_static', 'unfiltered_static')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def inputs():
    refs = [json.loads(s) for s in (RUN / 'test_manifest.jsonl').read_text().splitlines()]
    assert len(refs) == 256
    drafts = json.loads((RUN / 'shared_drafts.json').read_text())
    assert len(drafts) == len(refs)
    outputs = {'initial': [d['text'] for d in drafts]}
    for arm in ARMS[1:]:
        rows = json.loads((RUN / arm / 'results.json').read_text())
        assert len(rows) == len(refs)
        assert all(r['sample_id'] == x['sample_id'] and r['initial'] == d['text']
                   for r, x, d in zip(rows, refs, drafts))
        outputs[arm] = [r['final'] for r in rows]
    return refs, outputs


def score(model, examples):
    result = model.predict(examples, batch_size=8, gpus=1, num_workers=0)
    scores = result.scores if hasattr(result, 'scores') else result['scores']
    scores = np.asarray(scores, dtype=float)
    assert len(scores) == len(examples) and np.isfinite(scores).all()
    return scores.tolist()


def main():
    from comet import download_model, load_from_checkpoint
    refs, outputs = inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = download_model('Unbabel/wmt22-comet-da',
                                saving_directory=COMET_DIR, local_files_only=True)
    model = load_from_checkpoint(checkpoint)
    existing = OUT / 'raw_scores.json'
    raw = json.loads(existing.read_text()) if existing.exists() else {}
    for arm in ARMS:
        if arm in raw:
            assert len(raw[arm]) == len(refs)
            continue
        examples = [{'src': x['source'], 'mt': y, 'ref': x['reference']}
                    for x, y in zip(refs, outputs[arm])]
        raw[arm] = score(model, examples)
        dump(existing, raw)
        print('COMET', arm, 'mean', np.mean(raw[arm]), flush=True)
    clean = np.array(['\ufffd' not in x['source'] + x['reference'] for x in refs])
    rng = np.random.default_rng(20260930)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'role': 'post-hoc independent metric audit; no model or acceptance tuning',
        'model': 'Unbabel/wmt22-comet-da', 'checkpoint': checkpoint,
        'sources': len(refs), 'clean_reference_sources': int(clean.sum()),
        'gpu': 1, 'metric_semantics': 'reference-based COMET; model metric, not human gold',
        'means': {}, 'paired_vs_initial': {}, 'changed_task_differences': {},
    }
    for subset, mask in [('all', np.ones(len(refs), bool)), ('without_replacement_character', clean)]:
        idx = np.flatnonzero(mask)
        boot = rng.choice(idx, size=(2000, len(idx)), replace=True)
        report['means'][subset] = {arm: float(np.mean(np.asarray(raw[arm])[idx])) for arm in ARMS}
        report['paired_vs_initial'][subset] = {}
        for arm in ARMS[1:]:
            delta = np.asarray(raw[arm]) - np.asarray(raw['initial'])
            report['paired_vs_initial'][subset][arm] = {
                'mean_delta': float(delta[idx].mean()),
                'ci95_source_bootstrap': np.quantile(delta[boot].mean(axis=1), [.025,.975]).tolist(),
            }
    for arm in ARMS[1:]:
        hyp = outputs[arm]
        changed = np.array([a != b for a, b in zip(outputs['initial'], hyp)])
        delta = np.asarray(raw[arm]) - np.asarray(raw['initial'])
        report['changed_task_differences'][arm] = {
            'n': int(changed.sum()), 'positive': int((delta[changed] > 0).sum()),
            'negative': int((delta[changed] < 0).sum()),
            'mean_delta': float(delta[changed].mean()) if changed.any() else None,
        }
    dump(OUT / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
