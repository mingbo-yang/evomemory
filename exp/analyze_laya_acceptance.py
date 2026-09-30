"""Post-freeze diagnostics and the existing BERT policy on the same held-out pairs.

This script never selects a checkpoint, tunes a threshold, or changes a gate.
Its supplementary BERT control uses the existing production policy unchanged.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from laya_acceptance_common import (
    ROOT, OUT, LABELS, read_jsonl, digest, dump, probabilities, allowed,
)


def test_inputs():
    import acceptance_data as collection
    from prepare_acceptance_training import convert
    assert (OUT/'evaluation_report.json').exists(), 'Complete the frozen evaluation first'
    labels = read_jsonl(OUT/'evaluation/test_labels.jsonl')
    inputs = {}
    for raw in collection.unique_rows(collection.iter_raw('test')):
        if raw['valid']:
            row = convert(raw)
            inputs[row['id']] = row
    rows = []
    for item in labels:
        row = inputs[item['id']]
        row['label'] = item['label']
        rows.append(row)
    assert len(rows) == len(inputs) == 793
    return rows, np.array([r['delta'] for r in labels]), [r['sample_id'] for r in labels]


def summarize_acceptance(mask, delta):
    gain = delta[mask]
    return {
        'accepted': int(mask.sum()), 'coverage': float(mask.mean()),
        'precision': float(np.mean(gain > .01)) if len(gain) else None,
        'mean_delta': float(gain.mean()) if len(gain) else None,
        'net_delta_per_pair': float(gain.sum()/len(delta)),
        'positive': int((gain > .01).sum()), 'negative': int((gain < -.01).sum()),
        'tie': int((abs(gain) <= .01).sum()), 'any_negative': int((gain < 0).sum()),
    }


def source_bootstrap_auc(y, scores, sources, mask, replicates=2000):
    names = sorted(set(sources))
    groups = [np.array([i for i, s in enumerate(sources) if s == name and mask[i]], dtype=int)
              for name in names]
    rng = np.random.default_rng(20260926)
    aucs = []
    for _ in range(replicates):
        indices = np.concatenate([groups[j] for j in rng.integers(len(names), size=len(names))])
        if len(np.unique(y[indices])) == 2:
            aucs.append(roc_auc_score(y[indices], scores[indices]))
    return {'unit': 'source cluster', 'replicates': replicates,
            'ci95': np.quantile(aucs, [.025, .975]).tolist()}


def bert_control(rows, delta, sources):
    import torch
    from core.quality_feedback import BertQualityEvaluator, QualityPolicy
    torch.set_num_threads(4)
    config = json.loads((ROOT/'configs/optimized_feedback_v1.json').read_text())
    task = config['tasks']['wmt19_en_zh']
    policy = QualityPolicy(stop_threshold=task['stop_threshold'], **config['policy'])
    evaluator = BertQualityEvaluator(task['checkpoint'], config['bert_base'], device='cuda', batch_size=16)
    pairs = list(dict.fromkeys((row['input']['source'], row['input'][answer])
                              for row in rows for answer in ['current', 'candidate']))
    start = time.perf_counter()
    scores = dict(zip(pairs, evaluator.score_pairs(pairs)))
    elapsed = time.perf_counter() - start
    gains = []; accepted = []; reasons = Counter()
    for row in rows:
        x = row['input']; before = scores[x['source'], x['current']]; after = scores[x['source'], x['candidate']]
        take, reason = policy.accept(x['current'], x['candidate'], before, after)
        gains.append(after-before); accepted.append(take); reasons[reason] += 1
    gains = np.asarray(gains); accepted = np.asarray(accepted)
    changed = np.array([r['input']['current'] != r['input']['candidate'] for r in rows])
    y = delta > .01
    result = {
        'model': 'historical BERT absolute-score regressor; candidate minus current',
        'checkpoint': task['checkpoint'], 'checkpoint_sha256': digest(Path(task['checkpoint'])),
        'policy': asdict(policy),
        'scope': 'supplementary unchanged acceptance policy; no stopping, calibration, retraining or test-driven selection',
        'policy_difference': 'BERT retains its existing raw-length soft/hard gates; Laya uses the frozen normalized hard gate',
        'auc_better_all': float(roc_auc_score(y, gains)),
        'auc_better_changed': float(roc_auc_score(y[changed], gains[changed])),
        'changed_auc_bootstrap': source_bootstrap_auc(y, gains, sources, changed),
        'acceptance': summarize_acceptance(accepted, delta),
        'reasons': dict(reasons), 'unique_source_answer_pairs': len(pairs), 'inference_s': elapsed,
    }
    np.save(OUT/'evaluation/bert_score_gains.npy', gains)
    np.save(OUT/'evaluation/bert_accepted.npy', accepted)
    del evaluator; torch.cuda.empty_cache()
    return result


def run(with_bert=False):
    report = json.loads((OUT/'evaluation_report.json').read_text())
    frozen_hash = digest(OUT/'evaluation_lock.json')
    rows, delta, sources = test_inputs()
    y = np.array([LABELS.index(row['label']) for row in rows])
    changed = np.array([r['input']['current'] != r['input']['candidate'] for r in rows])
    eligible = np.array([allowed(r) for r in rows])
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'post-freeze diagnostics only; no test-driven selection or tuning',
        'evaluation_lock_sha256': frozen_hash, 'analyzer_sha256': digest(Path(__file__)),
        'test_pairs': len(rows), 'changed_pairs': int(changed.sum()), 'eligible_pairs': int(eligible.sum()),
        'class_counts': dict(Counter(row['label'] for row in rows)),
        'changed_class_counts': dict(Counter(rows[i]['label'] for i in np.where(changed)[0])),
        'models': {},
    }
    for name in ['base', 'finetuned']:
        z = np.load(OUT/'evaluation'/f'{name}_test_logits.npy')
        p = probabilities(z, report['calibration'][name]['temperature']); pred = p.argmax(1)
        item = {
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            'confusion_axes': 'rows true, columns predicted; Better, Tie, Worse',
            'classification': classification_report(y, pred, labels=[0, 1, 2], target_names=LABELS,
                                                  output_dict=True, zero_division=0),
            'changed_classification': classification_report(y[changed], pred[changed], labels=[0, 1, 2],
                                                          target_names=LABELS, output_dict=True, zero_division=0),
            'changed_auc_bootstrap': source_bootstrap_auc(y == 0, p[:, 0], sources, changed),
            'identical_tie_accuracy': float(np.mean(pred[~changed] == 1)),
            'changed_predicted_counts': {label: int(np.sum(pred[changed] == i)) for i, label in enumerate(LABELS)},
            'default_acceptance': report['models'][name]['default_acceptance'],
            'calibrated_acceptance': report['models'][name]['calibrated_acceptance'],
        }
        result['models'][name] = item
    if with_bert:
        result['bert_supplementary'] = bert_control(rows, delta, sources)
    assert digest(OUT/'evaluation_lock.json') == frozen_hash
    dump(OUT/'diagnostics.json', result)
    lines = ['# Supplementary diagnostics', '',
             'All diagnostics follow the frozen test evaluation. They do not select a new model or threshold.', '',
             f"Test: {len(rows)} pairs; {changed.sum()} changed; {len(rows)-changed.sum()} identical.",
             f"Changed labels: {result['changed_class_counts']}", '',
             '| Model | Changed Better AUC (95% source CI) | Default accepted | Precision | Mean delta |',
             '|---|---|---:|---:|---:|']
    def number(x): return 'undefined' if x is None else f'{x:.4f}'
    for name in ['base', 'finetuned']:
        m = result['models'][name]; a = m['default_acceptance']; lo, hi = m['changed_auc_bootstrap']['ci95']
        lines.append(f"| {name} | {report['models'][name]['auc_better_changed']:.4f} [{lo:.4f}, {hi:.4f}] | {a['accepted']} | {number(a['precision'])} | {number(a['mean_delta'])} |")
    if with_bert:
        b = result['bert_supplementary']; a = b['acceptance']; lo, hi = b['changed_auc_bootstrap']['ci95']
        lines.append(f"| historical BERT (existing policy) | {b['auc_better_changed']:.4f} [{lo:.4f}, {hi:.4f}] | {a['accepted']} | {number(a['precision'])} | {number(a['mean_delta'])} |")
    lines += ['', 'Laya default acceptance means calibrated pBetter >= 0.5 plus its hard length gate. It is diagnostic, not the selected operating point.',
              'BERT is an acceptance-only supplementary control with the unchanged older soft/hard length gates; this is not a complete pipeline comparison.',
              'Labels represent weak reference-metric feedback, not independently verified translation quality. No classification result establishes full_static/full_online effectiveness.']
    (OUT/'DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--with-bert', action='store_true')
    run(parser.parse_args().with_bert)
