"""Retrospective, fixed-test comparison of every saved Laya epoch.

This diagnostic deliberately evaluates multiple frozen checkpoints on a previously
used test set. It does not change formal development-based checkpoint selection,
construct labels again, or relax evaluate_laya_acceptance's single-checkpoint lock.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import laya_acceptance_common as C


def read(path):
    return json.loads(Path(path).read_text())


def check_record(record, feedback, test_meta):
    path = Path(record['path'])
    for file, key in [('model.safetensors', 'checkpoint_sha256'), ('rl_agent_config.json', 'config_sha256')]:
        if C.digest(path / file) != record[key]:
            raise ValueError(f'Frozen checkpoint changed: {path / file}')
    cfg = C.validate_checkpoint(path)
    if cfg.get('label_feedback') != feedback or feedback != test_meta['label_feedback']:
        raise ValueError('Training/test feedback protocol mismatch')
    if not cfg.get('seen_source_hashes'):
        raise ValueError('Missing training-source provenance')
    if set(cfg['seen_source_hashes']) & set(test_meta['source_hashes']['test']):
        raise ValueError('Test sources overlap training/development sources')
    return path


def aligned_logits(predictions, rows, delta, sample_ids, key=None):
    """Reject reordered, changed or incompletely aligned baseline predictions."""
    if len(predictions) != len(rows):
        raise ValueError('Prediction count mismatch')
    for p, row, gain, sid in zip(predictions, rows, delta, sample_ids):
        if (p['id'], p['sample_id'], p['label']) != (row['id'], sid, row['label']):
            raise ValueError('Prediction identity/label mismatch')
        if p['delta_comet'] != gain:
            raise ValueError('Frozen COMET feedback changed')
    return np.asarray([p['logits'][key] if key else p['logits'] for p in predictions])


def worker(args):
    import torch
    start = time.monotonic()
    protocol = read(args.output / 'evaluation_protocol.json')
    if C.digest(args.test_data / 'dataset.json') != protocol['test_metadata_sha256']:
        raise ValueError('Test metadata changed after freeze')
    record = next(r for r in protocol['checkpoints'] if r['epoch'] == args.epoch)
    path = check_record(record, protocol['label_feedback'], read(args.test_data / 'dataset.json'))
    rows, delta, sample_ids = C.load_fold('test', args.test_data)
    if len(rows) != protocol['test_pairs']:
        raise ValueError('Expected the same 449 changed test pairs')
    tok = C.get_tokenizer(path)
    model, _ = C.load_model(path)
    model.to(args.device)
    encoded = C.EncodedPairs(rows, tok)
    logits = C.infer(model, encoded, batch_size=args.batch_size)
    metrics = C.metrics(rows, logits, delta)
    metrics.update(epoch=args.epoch, checkpoint_sha256=record['checkpoint_sha256'],
                   batch_size=args.batch_size, wall_seconds=time.monotonic() - start,
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
                   retrospective_diagnostic=True, effect_validated=False)
    prefix = args.output / f'epoch_{args.epoch:02d}'
    with prefix.with_suffix('.predictions.jsonl').open('x') as f:
        for row, sid, gain, z in zip(rows, sample_ids, delta, logits):
            f.write(json.dumps({'id': row['id'], 'sample_id': sid, 'label': row['label'],
                                'delta_comet': gain, 'logits': z.tolist()}, ensure_ascii=False) + '\n')
    C.dump(prefix.with_suffix('.json'), metrics)
    print(json.dumps(metrics), flush=True)


def bootstrap_samples(rows, delta, sample_ids, all_logits):
    source_ids = sorted(set(sample_ids))
    source_index = {s: i for i, s in enumerate(source_ids)}
    idx = np.array([source_index[s] for s in sample_ids])
    n = len(source_ids)
    counts = np.bincount(idx, minlength=n)
    draws = np.random.default_rng(20260930).integers(0, n, size=(2000, n))
    denominator = counts[draws].sum(1)
    truth = np.array([r['label'] for r in rows])
    result = {}
    for name, z in all_logits.items():
        correct = np.array(C.decisions(z)) == truth
        gain = np.asarray(delta) * C.accepted_mask(rows, z)
        by_source = {'accuracy': np.bincount(idx, weights=correct, minlength=n),
                     'net_delta_per_pair': np.bincount(idx, weights=gain, minlength=n)}
        result[name] = {k: v[draws].sum(1) / denominator for k, v in by_source.items()}
    return result


def summarize(args, records, rows, delta, sample_ids):
    baseline_predictions = C.read_jsonl(args.baseline / 'frozen_test_predictions.jsonl')
    logits = {'baseline_4ep_best_epoch3': aligned_logits(baseline_predictions, rows, delta, sample_ids,
                                                       'trained_laya_binary')}
    previous_predictions = C.read_jsonl(args.previous / 'test_predictions.jsonl')
    logits['previous_10ep_best_epoch8'] = aligned_logits(previous_predictions, rows, delta, sample_ids)
    metrics = {name: C.metrics(rows, z, delta) for name, z in logits.items()}
    for name, path in [('baseline_4ep_best_epoch3', args.baseline), ('previous_10ep_best_epoch8', args.previous)]:
        saved = read(path / 'test_evaluation.json')
        for key in ['accuracy', 'macro_f1', 'accepted', 'net_delta_per_pair']:
            if metrics[name][key] != saved[key]:
                raise ValueError(f'Baseline metrics changed: {name} {key}')
    for record in records:
        name = f"epoch_{record['epoch']:02d}"
        logits[name] = aligned_logits(C.read_jsonl(args.output / f'{name}.predictions.jsonl'), rows, delta, sample_ids)
        metrics[name] = read(args.output / f'{name}.json')
    samples = bootstrap_samples(rows, delta, sample_ids, logits)
    baseline = 'baseline_4ep_best_epoch3'
    paired = {}
    for name, sample in samples.items():
        metrics[name]['ci95_source_bootstrap'] = {k: np.quantile(v, [.025, .975]).tolist() for k, v in sample.items()}
        if name != baseline:
            paired[name] = {k: {'estimate': metrics[name][k] - metrics[baseline][k],
                                'ci95': np.quantile(v - samples[baseline][k], [.025, .975]).tolist()}
                            for k, v in sample.items()}
    paired_epoch10 = {}
    if 'epoch_10' in samples:
        for name, sample in samples.items():
            if name != 'epoch_10':
                paired_epoch10[name] = {k: {'estimate': metrics[name][k] - metrics['epoch_10'][k],
                                           'ci95': np.quantile(v - samples['epoch_10'][k], [.025, .975]).tolist()}
                                       for k, v in sample.items()}
    training_protocol = read(args.model / 'training_protocol.json')
    warm = training_protocol.get('warm_start')
    inherited_recheck = []
    if warm:
        parent_output = Path(warm['parent_model']) / 'test_all_epochs'
        for epoch in range(1, warm['parent_epoch'] + 1):
            name = f'epoch_{epoch:02d}'
            prior_z = aligned_logits(C.read_jsonl(parent_output / f'{name}.predictions.jsonl'), rows, delta, sample_ids)
            inherited_recheck.append({'epoch': epoch,
                'decision_agreement': float(np.mean(np.array(C.decisions(prior_z)) == np.array(C.decisions(logits[name])))),
                'max_abs_logit_difference': float(np.max(np.abs(prior_z - logits[name])))})
    old_history = read(args.previous / 'history.json')
    new_history = read(args.model / 'history.json')
    previous_epoch = read(args.previous / 'best_selection.json')['epoch']
    original_sha = read(args.previous / 'evaluation_lock.json')['checkpoint_sha256']
    repeated_sha = next(r['checkpoint_sha256'] for r in records if r['epoch'] == previous_epoch)
    keys = ['accuracy', 'macro_f1', 'accepted', 'net_delta_per_pair']
    reproduction = {
        'original_saved_epoch': previous_epoch, 'original_weight_sha256': original_sha,
        'repeated_epoch_weight_sha256': repeated_sha, 'bitwise_weight_match': original_sha == repeated_sha,
        'development_metrics_exact_match': [{
            'epoch': a['epoch'], 'equal': all(a['development'][k] == b['development'][k] for k in keys)
        } for a, b in zip(old_history, new_history)],
        'epoch8_test_decision_agreement': float(np.mean(np.array(C.decisions(logits[f'epoch_{previous_epoch:02d}'])) ==
                                                       np.array(C.decisions(logits['previous_10ep_best_epoch8']))))}
    report = {
        'test_status': 'Retrospective diagnostic on the previously evaluated 449 pairs; not fresh confirmatory testing',
        'test_pairs': len(rows), 'test_sources': len(set(sample_ids)),
        'label_rule': 'delta COMET > 0 Accept; changed text with delta <= 0 Reject; exact no-ops excluded',
        'decision_rule': 'raw two-option argmax; exact logit tie Reject; unchanged validity gate; no tuned threshold',
        'net_delta_definition': 'sum(delta_COMET for accepted valid candidates) / all 449 pairs, native COMET scale',
        'arms': metrics, 'paired_difference_vs_baseline': paired,
        'paired_difference_vs_epoch10': paired_epoch10,
        'training_continuation': warm, 'training_protocol': training_protocol,
        'inherited_epochs_test_recheck': inherited_recheck,
        'formal_development_selected_epoch': read(args.model / 'best_selection.json')['epoch'],
        'posthoc_test_leaders_not_formal_selection': {
            metric: max((f"epoch_{r['epoch']:02d}" for r in records), key=lambda name: metrics[name][metric])
            for metric in ['accuracy', 'macro_f1', 'net_delta_per_pair']},
        'reproduction': reproduction,
        'bootstrap': '2000 paired source-cluster resamples, seed 20260930; individual intervals, no multiplicity correction',
        'effect_validated': False,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    C.dump(args.output / 'comparison.json', report)
    for name, m in metrics.items():
        print(name, 'accuracy', m['accuracy'], 'F1', m['macro_f1'], 'accepted', m['accepted'],
              'precision', m['accept_precision'], 'net_delta', m['net_delta_per_pair'], flush=True)
    print(json.dumps({'reproduction': reproduction, 'leaders': report['posthoc_test_leaders_not_formal_selection']}), flush=True)


def validate_training_lineage(args, lock):
    protocol = read(args.model / 'training_protocol.json')
    for reference in (args.baseline, args.previous):
        reference_protocol = read(reference / 'training_protocol.json')
        for key in ['dataset_sha256', 'base', 'seed', 'micro_batch', 'effective_batch', 'training', 'label_feedback']:
            if protocol[key] != reference_protocol[key]:
                raise ValueError(f'Training protocol differs from reference: {key}')
    if protocol['epochs'] != lock['epochs']:
        raise ValueError('Training horizon differs from checkpoint lock')
    warm = protocol.get('warm_start')
    if warm:
        initial = protocol['initial_epoch']
        if initial != warm['parent_epoch'] or protocol['epochs'] != initial + protocol['additional_epochs']:
            raise ValueError('Invalid continuation epoch numbering')
        parent = Path(warm['parent_model'])
        parent_records = read(parent / 'epoch_checkpoints.json')[:initial]
        inherited = lock['checkpoints'][:initial]
        if len(parent_records) != initial:
            raise ValueError('Incomplete parent checkpoint history')
        for old, new in zip(parent_records, inherited):
            for key in ['epoch', 'path', 'checkpoint_sha256', 'config_sha256']:
                if old[key] != new[key]:
                    raise ValueError('Inherited checkpoint changed')
        if inherited[-1]['checkpoint_sha256'] != warm['checkpoint_sha256']:
            raise ValueError('Continuation does not start from the claimed parent weights')
        if Path(inherited[-1]['path']).resolve() != Path(warm['path']).resolve():
            raise ValueError('Continuation parent path mismatch')
        if not lock.get('test_used_for_selection') or not warm.get('initial_checkpoint_selected_using_test'):
            raise ValueError('Retrospective parent selection must be disclosed')
    elif protocol['epochs'] != read(args.previous / 'training_protocol.json')['epochs']:
        raise ValueError('Repeated experiment must use the original training horizon')
    return protocol


def compare(args):
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError('Positive workers and batch size required')
    lock = read(args.model / 'all_checkpoints_lock.json')
    records = lock['checkpoints']
    if [r['epoch'] for r in records] != list(range(1, lock['epochs'] + 1)):
        raise ValueError('Missing or duplicate epoch checkpoints')
    training_protocol = validate_training_lineage(args, lock)
    test_meta = read(args.test_data / 'dataset.json')
    # Freeze/validate every model before running any test inference. The original
    # test metadata stays bound to the old baseline and is never rewritten.
    for r in records:
        check_record(r, lock['label_feedback'], test_meta)
    if test_meta['frozen_checkpoint_sha256'] != C.digest(args.baseline / 'best/model.safetensors'):
        raise ValueError('Original test-set lock must still match the original baseline')
    for path in (args.baseline, args.previous):
        old_lock = read(path / 'evaluation_lock.json')
        if old_lock['checkpoint_sha256'] != C.digest(path / 'best/model.safetensors'):
            raise ValueError('Reference model changed')
    rows, delta, sample_ids = C.load_fold('test', args.test_data)
    if len(rows) != 449:
        raise ValueError('This audit requires the exact 449-pair comparison set')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'checkpoints': records, 'label_feedback': lock['label_feedback'],
        'test_data': str(args.test_data.resolve()), 'test_metadata_sha256': C.digest(args.test_data / 'dataset.json'),
        'test_pairs': len(rows), 'test_sources': len(set(sample_ids)),
        'all_checkpoints_lock_sha256': C.digest(args.model / 'all_checkpoints_lock.json'),
        'evaluator_sha256': C.digest(Path(__file__)), 'common_code_sha256': C.digest(Path(C.__file__)),
        'batch_size': args.batch_size, 'workers': args.workers,
        'scope': 'user-requested retrospective diagnostic; parent checkpoint may have been selected using earlier test results',
        'training_protocol': training_protocol,
        'frozen_at_utc': datetime.now(timezone.utc).isoformat()}
    C.dump(args.output / 'evaluation_protocol.json', protocol)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    # Launch processes together in bounded groups: ten independent models on one GPU.
    for start in range(0, len(records), args.workers):
        processes = []
        for r in records[start:start + args.workers]:
            log = (args.output / f"epoch_{r['epoch']:02d}.log").open('x')
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--epoch', str(r['epoch']),
                       '--output', str(args.output), '--test-data', str(args.test_data), '--device', args.device,
                       '--batch-size', str(args.batch_size)]
            processes.append((r['epoch'], subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        failures = []
        for epoch, proc, log in processes:
            code = proc.wait()
            log.close()
            if code:
                failures.append((epoch, code))
        if failures:
            raise RuntimeError(f'Evaluation failed (see per-epoch logs): {failures}')
    if C.digest(args.test_data / 'dataset.json') != protocol['test_metadata_sha256']:
        raise ValueError('Test metadata changed during evaluation')
    summarize(args, records, rows, delta, sample_ids)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('/mnt/huawei/ymb/model/laya-multilingual-accept-reject-10ep-allcheckpoints-v1'))
    p.add_argument('--test-data', type=Path, default=C.ROOT / 'runs/acceptance_binary_test_v1')
    p.add_argument('--baseline', type=Path, default=C.OUT)
    p.add_argument('--previous', type=Path, default=Path('/mnt/huawei/ymb/model/laya-multilingual-accept-reject-10ep-v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--epoch', type=int)
    args = p.parse_args()
    worker(args) if args.worker else compare(args)


if __name__ == '__main__':
    main()
