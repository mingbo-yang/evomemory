"""Prepare changed-only binary COMET labels; exact no-ops are retained in audit."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from core.candidate_validation import norm, source_key
import laya_acceptance_common as C
from core.comet_feedback import from_config, validated_scores, validate_epsilon
from legacy_laya_v1.prepare_acceptance_training import QUARANTINE

FOLDS = ('train', 'development', 'temperature_calibration', 'decision_diagnostics')


def read_jsonl(path):
    return C.read_jsonl(path)


def pair_id(row):
    return hashlib.sha256(json.dumps([row['source'], row['before'], row['candidate']],
                                    ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def classify(delta, epsilon):
    epsilon = validate_epsilon(epsilon)
    if not math.isfinite(delta):
        raise ValueError('Feedback delta must be finite')
    return 'Accept' if delta > epsilon else 'Reject'


def convert(row, epsilon=None):
    result = {'id': pair_id(row), 'input': {'source': row['source'], 'current': row['before'], 'candidate': row['candidate']}}
    C.require_changed_pairs([result])
    if epsilon is not None:
        result['label'] = classify(row['delta'], epsilon)
    return result


def map_legacy_label(label):
    """Semantic mapping for audits only; formal training recomputes COMET labels."""
    if label not in {'Better', 'Tie', 'Worse'}:
        raise ValueError('Uncertain/missing legacy labels must not be silently mapped')
    return 'Accept' if label == 'Better' else 'Reject'


def swap_training_item(item, *, delta=None, epsilon=None):
    if delta is None or epsilon is None:
        raise ValueError('A Reject label alone cannot determine the swapped label; raw offline delta and epsilon are required')
    return {'id': item['id'] + ':swapped',
            'input': {'source': item['input']['source'], 'current': item['input']['candidate'], 'candidate': item['input']['current']},
            'label': classify(-delta, epsilon)}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))


def build_dataset(folds, references, feedback, output, epsilon, *, protected_sources=(), frozen_checkpoint=None):
    """Feedback stays in audit sidecars. Never use old delta/labels to select pairs."""
    epsilon = validate_epsilon(epsilon)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite dataset: {output}')
    if not folds or set(folds) - set(FOLDS) - {'test'}:
        raise ValueError('Unknown or empty split specification')
    if feedback.metric != 'comet':
        raise ValueError('Binary labels require explicit COMET feedback')
    feedback_meta = dict(feedback.metadata, epsilon=epsilon)
    frozen_cfg = None
    if 'test' in folds:
        if frozen_checkpoint is None:
            raise ValueError('Test feedback remains sealed until a binary checkpoint is frozen')
        frozen_cfg = C.validate_checkpoint(frozen_checkpoint)
        if frozen_cfg.get('label_feedback') != feedback_meta:
            raise ValueError('Test epsilon/feedback must match the already frozen training protocol')
    protected = {source_key(s) for s in protected_sources}
    # Validate all sources BEFORE any quality scoring or label construction.
    keys = {}
    prepared = {}
    exclusions = []
    for fold, rows in folds.items():
        keys[fold] = set()
        prepared[fold] = []
        seen = set()
        for row in rows:
            sid = row['sample_id']
            ref = references[fold][sid]
            if row['source'] != ref['source']:
                raise ValueError('Source/manifest mismatch')
            key = source_key(row['source'])
            if key in protected:
                raise ValueError('Source overlaps the protected test/initial memory set')
            keys[fold].add(key)
            if not row.get('valid', False):
                exclusions.append({'sample_id': sid, 'fold': fold, 'reason': 'invalid_candidate'})
                continue
            if sid in QUARANTINE:
                exclusions.append({'sample_id': sid, 'fold': fold, 'reason': QUARANTINE[sid]})
                continue
            if not all(isinstance(row[k], str) and row[k].strip() for k in ('source', 'before', 'candidate')):
                raise ValueError('Empty/nontext pair')
            pid = pair_id(row)
            if pid in seen:
                continue
            seen.add(pid)
            prepared[fold].append((row, ref['reference']))
    for a in keys:
        for b in keys:
            if a != b and keys[a] & keys[b]:
                raise ValueError(f'Source leakage between {a} and {b}')
    if frozen_cfg:
        prior = set(frozen_cfg.get('seen_source_hashes', []))
        if not prior:
            raise ValueError('Frozen checkpoint must record its training/development source identities')
        if keys['test'] & prior:
            raise ValueError('Test sources overlap checkpoint development data')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.staging.', dir=output.parent))
    try:
        stats = {}
        for fold, pairs in prepared.items():
            if not pairs:
                raise ValueError('No valid pairs in split ' + fold)
            changed, no_ops = [], []
            for row, ref in pairs:
                inputs = {'source': row['source'], 'current': row['before'], 'candidate': row['candidate']}
                if C.is_noop(inputs):
                    no_ops.append({'id': pair_id(row), 'sample_id': row['sample_id'],
                                   'source_hash': source_key(row['source']), 'input': inputs,
                                   'decision': 'Reject', 'reason': 'current_equals_candidate',
                                   'decision_origin': 'deterministic_rule', 'used_for_model': False})
                else:
                    changed.append((row, ref))
            inputs = [{'source': r['source'], 'current': r['before'], 'candidate': r['candidate'], 'reference': ref} for r, ref in changed]
            scores = validated_scores(inputs, feedback.score_pairs(inputs)) if inputs else []
            items, audit = [], []
            for (row, _), score in zip(changed, scores):
                item = convert(dict(row, delta=score['delta']), epsilon)
                items.append(item)
                audit.append({'id': item['id'], 'sample_id': row['sample_id'], 'source_hash': source_key(row['source']), **score})
            write_jsonl(staging / f'{fold}.jsonl', items)
            write_jsonl(staging / 'audit' / f'{fold}.jsonl', audit)
            write_jsonl(staging / 'audit' / f'{fold}_no_ops.jsonl', no_ops)
            stats[fold] = {'pairs': len(items), 'sources': len({r['source_hash'] for r in audit}),
                           'labels': dict(Counter(r['label'] for r in items)),
                           'valid_unique_pairs': len(pairs), 'no_op_pairs': len(no_ops),
                           'no_op_sources': len({r['source_hash'] for r in no_ops}),
                           'sources_checked_for_isolation': len(keys[fold])}
        write_jsonl(staging / 'audit/exclusions.jsonl', exclusions)
        meta = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'decision_schema': C.SCHEMA,
                'input_schema': list(C.INPUT_FIELDS), 'label_order': C.LABELS, 'label_feedback': feedback_meta,
                'model_population': C.MODEL_POPULATION, 'no_op_rule': C.NO_OP_RULE,
                'label_rule': 'Accept iff delta_COMET > epsilon; Reject otherwise',
                'runtime_rule': 'raw binary classification; epsilon is not a runtime acceptance parameter',
                'augmentation': 'option-order permutation only; no Current/Candidate swap',
                'source_hashes': {f: sorted(v) for f, v in keys.items()}, 'statistics': stats,
                'test': 'labelled after checkpoint freeze' if frozen_cfg else 'sealed; not scored or exported',
                'frozen_checkpoint_sha256': C.digest(Path(frozen_checkpoint) / 'model.safetensors') if frozen_cfg else None,
                'output_hashes': {str(p.relative_to(staging)): C.digest(p) for p in staging.rglob('*.jsonl')}}
        C.dump(staging / 'dataset.json', meta)
        staging.rename(output)
        return meta
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, default=C.COLLECTION)
    p.add_argument('--output', type=Path, default=C.DATA)
    p.add_argument('--epsilon', type=float, help='Offline COMET label tolerance; never used at inference')
    p.add_argument('--config', type=Path, default=C.ROOT / 'configs/laya_binary_v1.json')
    p.add_argument('--folds', nargs='+', choices=list(FOLDS) + ['test'], default=list(FOLDS))
    p.add_argument('--frozen-checkpoint', type=Path, help='Required before labelling test')
    args = p.parse_args()
    raw, refs = {}, {}
    for fold in args.folds:
        old = 'threshold_calibration' if fold == 'decision_diagnostics' else fold
        pairs_path = args.collection / 'pairs' / f'{old}.jsonl'
        if fold == 'test' and not pairs_path.exists():
            from acceptance_data import unique_rows
            records = (json.loads(p.read_text()) for p in sorted((args.collection / 'raw/test').glob('*.json')))
            raw[fold] = list(unique_rows(records))
        else:
            raw[fold] = read_jsonl(pairs_path)
        refs[fold] = {r['sample_id']: r for r in read_jsonl(args.collection / f'{old}_manifest.jsonl')}
    memory = read_jsonl(args.collection / 'initial_memory.jsonl')
    protected = [r['source_input'] for r in memory]
    if 'test' not in args.folds:
        protected += [r['source'] for r in read_jsonl(args.collection / 'test_manifest.jsonl')]
    cfg = json.loads(args.config.read_text())
    epsilon = cfg['label_epsilon'] if args.epsilon is None else args.epsilon
    result = build_dataset(raw, refs, from_config(cfg['feedback']), args.output, epsilon,
                           protected_sources=protected, frozen_checkpoint=args.frozen_checkpoint)
    print(json.dumps(result['statistics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
