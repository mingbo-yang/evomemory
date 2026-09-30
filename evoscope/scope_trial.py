"""Opt-in frozen-bank intervention diagnostic; never an evolution entry point."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from statistics import mean

from .core import Artifacts, Bank, atomic_json, digest, load_manifest
from .environments import AlfWorldEnvironment, WebShopEnvironment, View
from .local_watchdog import rows, stamp
from .models import LocalModel
from .runner import Checkpoint, RestoreError, Runner

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('alfworld', 'webshop')
BRANCHES = ('normal', 'persistent-expose', 'persistent-mask', 'single-step-expose', 'single-step-mask')
COMPARISONS = {
    'persistent': ('persistent-expose', 'persistent-mask'),
    'single-step': ('single-step-expose', 'single-step-mask'),
    'normal-vs-persistent-mask': ('normal', 'persistent-mask'),
    'normal-vs-single-step-mask': ('normal', 'single-step-mask'),
}
REPEATS = 3


def restore_checkpoint(raw):
    value = dict(raw)
    value['views'] = tuple(View(**{**v, 'admissible': tuple(v['admissible'])}) for v in value['views'])
    for name in ('actions', 'seen_ids', 'selected_ids'):
        value[name] = tuple(value[name])
    return Checkpoint(**value)


def freeze_cases(source, dataset):
    """Use every existing valid probe (at most six), never rank by effect sign."""
    base = source / dataset
    checkpoints = {}
    for ep in rows(base / 'learn/episodes.jsonl'):
        for raw in ep['checkpoints']:
            cp = restore_checkpoint(raw)
            checkpoints[cp.id] = raw
    evidence = rows(base / 'probe/evidence.jsonl')
    if not 1 <= len(evidence) <= 6:
        raise ValueError('small diagnostic requires 1..6 valid source probes per dataset; no silent subsampling')
    bank = Bank.load(source / f'{dataset}-m0.json')
    tasks = {t.id: t for t in load_manifest(source / f'{dataset}-manifest.json')}
    cases, used = [], set()
    for row in evidence:
        raw = checkpoints[row['checkpoint_id']]
        cp = restore_checkpoint(raw)
        task, policy = tasks[cp.task_id], bank.get(row['policy_id'])
        key = (policy.id, cp.id)
        if (task.split != 'learn' or row['bank_hash'] != bank.fingerprint
                or row['policy_version'] != policy.version or cp.task_hash != digest(asdict(task))
                or cp.bank_hash != bank.fingerprint or cp.actor_hash != row['model_fingerprint']
                or policy.id not in cp.selected_ids or policy.id in cp.seen_ids or key in used):
            raise ValueError('source evidence is incompatible, contaminated or duplicated')
        used.add(key)
        cases.append({'source_evidence_id': row['id'], 'checkpoint_id': cp.id,
                      'policy_id': policy.id, 'policy_version': policy.version,
                      'checkpoint': raw, 'seed': row['seed'], 'repeats': REPEATS})
    return cases


def prepare(out, source):
    if not json.loads((source / 'final-status.json').read_text()).get('all_requested_completed'):
        raise ValueError('source validation must be complete')
    frozen = {d: freeze_cases(source, d) for d in DATASETS}
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for dataset, cases in frozen.items():
        for suffix in ('manifest', 'm0'):
            name = f'{dataset}-{suffix}.json'
            shutil.copy2(source / name, out / name)
            inputs[name] = sha256((source / name).read_bytes()).hexdigest()
        atomic_json(out / f'{dataset}-cases.json', cases)
        for name in ('learn/episodes.jsonl', 'probe/evidence.jsonl'):
            inputs[f'{dataset}/{name}'] = sha256((source / dataset / name).read_bytes()).hexdigest()
    plan = {'created_utc': stamp(), 'trial': 'probe-scope', 'protocol': 'scope-diagnostic-v1',
            'source': str(source), 'source_hashes': inputs,
            'selection': 'all valid source learn checkpoints; no effect-based selection or replacement',
            'cases': {d: len(c) for d, c in frozen.items()}, 'branches': BRANCHES,
            'repeats': REPEATS, 'planned_rollouts': sum(map(len, frozen.values())) * len(BRANCHES) * REPEATS,
            'branch_order': 'cyclic rotation by case index plus repeat index; same seed across five branches',
            'model': 'qwen3.8-27b-local', 'model_path': '/mnt/huawei/ymb/model/Qwen3.8-27B',
            'gpu': 1, 'max_tokens': 8192, 'temperature': 0, 'local_only': True,
            'bank_updates': False, 'editor_calls': False, 'gate_calls': False, 'held_out_test_used': False,
            'wall_time_limit': None, 'scope_definition': 'single-step changes only the checkpoint action input; every later step returns to natural retrieval with the original when',
            'normal_definition': 'original conditioned bank with natural retrieval at every step; not a persistent forced conditioned target',
            'limitations': 'Retrospective diagnostic on selected clean learn states, not unbiased task-effect estimates. Normal vs persistent differences may include future retrieval/capacity changes. Behavior/cost differences never relabel neutral or change acceptance.'}
    atomic_json(out / 'plan.json', plan)
    snapshot = out / 'source'
    snapshot.mkdir()
    for path in sorted((ROOT / 'evoscope').glob('*.py')):
        shutil.copy2(path, snapshot / path.name)
    shutil.copy2(ROOT / 'evoscope/scripts/serve_qwen_local.sh', snapshot / 'serve_qwen_local.sh')


def compare(left, right):
    """Reward classification stays unchanged; behavior metrics are diagnostic only."""
    if (left.get('error') or right.get('error') or left['score'] is None or right['score'] is None
            or not math.isfinite(left['score']) or not math.isfinite(right['score'])):
        return {'valid': False, 'delta': None, 'neutral_behavior': None}
    a, b = left['actions'], right['actions']
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    delta = left['score'] - right['score']
    return {'valid': True, 'delta': delta, 'same_actions': a == b,
            'first_divergence_after_checkpoint': first, 'step_difference': len(a) - len(b),
            'neutral_behavior': ('same_actions' if a == b else 'different_actions') if delta == 0 else None}


def compare_repeats(repeats, left, right):
    pairs = [compare(r['branches'][left], r['branches'][right]) for r in repeats]
    if not pairs or not all(p['valid'] for p in pairs):
        return {'direction': 'execution_error', 'mean_delta': None, 'pairs': pairs}
    deltas = [p['delta'] for p in pairs]
    direction = ('helpful' if all(d > 0 for d in deltas) else
                 'harmful' if all(d < 0 for d in deltas) else
                 'neutral' if all(d == 0 for d in deltas) else 'unstable')
    return {'direction': direction, 'mean_delta': mean(deltas), 'pairs': pairs,
            'neutral_behavior_counts': dict(Counter(p['neutral_behavior'] for p in pairs if p['neutral_behavior']))}


def run_worker(out, dataset):
    cases = json.loads((out / f'{dataset}-cases.json').read_text())
    tasks = {t.id: t for t in load_manifest(out / f'{dataset}-manifest.json')}
    bank = Bank.load(out / f'{dataset}-m0.json')
    original_hash = bank.fingerprint
    artifacts = Artifacts(out / dataset)
    artifacts.write('initial_bank', bank.to_json())
    artifacts.write('bank', bank.to_json())
    model = LocalModel(artifacts, 'qwen3.8-27b-local', 'http://127.0.0.1:8124/v1',
                       max_tokens=8192, send_seed=True)
    artifacts.write('model_config', model.config)
    max_steps = 30 if dataset == 'alfworld' else 20
    factory = ((lambda: AlfWorldEnvironment(max_steps)) if dataset == 'alfworld' else
               (lambda: WebShopEnvironment(str(ROOT / 'downloads/WebShop-10k'), num_products=None)))
    runner = Runner(factory, model, artifacts, max_steps)
    completed, errors = [], 0
    print('START FROZEN SCOPE DIAGNOSTIC', dataset, 'cases', len(cases), flush=True)
    for index, case in enumerate(cases):
        cp = restore_checkpoint(case['checkpoint'])
        if cp.actor_hash != model.fingerprint or cp.max_steps != max_steps or cp.top_k != runner.top_k:
            raise ValueError('source checkpoint model/runner configuration differs')
        repeats = []
        for repeat in range(REPEATS):
            seed = case['seed'] + repeat * 1000
            shift = (index + repeat) % len(BRANCHES)
            order = BRANCHES[shift:] + BRANCHES[:shift]
            pair = {'repeat': repeat, 'seed': seed, 'order': order, 'branches': {}}
            for branch in order:
                intervention = 'normal' if branch == 'normal' else branch.rsplit('-', 1)[1]
                scope = 'single-step' if branch.startswith('single-step') else 'persistent'
                phase = 'trial-' + branch
                start = len(artifacts.costs)
                try:
                    ep = runner.run(tasks[cp.task_id], bank, phase, seed, cp, case['policy_id'],
                                    intervention, evidence_id=case['source_evidence_id'], intervention_scope=scope)
                    result = {'score': ep.score if math.isfinite(ep.score) else None,
                              'done': ep.done, 'error': ep.error or (None if math.isfinite(ep.score) else 'NonFiniteScore'),
                              'actions': ep.actions[len(cp.actions):],
                              'retrieval_count': ep.retrieval_counts.get(case['policy_id'], 0),
                              'exposure_count': ep.exposure_counts.get(case['policy_id'], 0)}
                except RestoreError as exc:
                    result = {'score': None, 'done': False, 'error': type(exc).__name__, 'actions': []}
                result['costs'] = artifacts.cost_summary(start)
                result['rollout_id'] = f'rollout-{runner.rollout_count:06d}'
                decisions = [r for r in rows(artifacts.root / f'{phase}/decisions.jsonl')
                             if r['rollout_id'] == result['rollout_id']]
                result['intervened_steps'] = sum(r['intervention_applied'] for r in decisions)
                result['target_exposed_outside_retrieval'] = sum(
                    case['policy_id'] not in r['retrieved_ids'] and
                    any(p['id'] == case['policy_id'] for p in r['policies']) for r in decisions)
                if scope == 'single-step' and result['intervened_steps'] > 1:
                    raise AssertionError('single-step intervention leaked into continuation')
                pair['branches'][branch] = result
                errors += bool(result.get('error'))
                artifacts.append('diagnostic/rollouts', {'case': index, 'checkpoint_id': cp.id,
                    'policy_id': case['policy_id'], 'repeat': repeat, 'seed': seed, 'branch': branch, **result})
            repeats.append(pair)
        comparisons = {name: compare_repeats(repeats, *branches) for name, branches in COMPARISONS.items()}
        pattern = []
        for row in repeats:
            n, e, m = (row['branches'][b] for b in ('normal', 'persistent-expose', 'persistent-mask'))
            valid = compare(n, m)['valid'] and compare(e, m)['valid']
            pattern.append(bool(valid and n['score'] == m['score'] and n['actions'] == m['actions']
                                and e['score'] < m['score']))
        record = {'checkpoint_id': cp.id, 'task_id': cp.task_id, 'policy_id': case['policy_id'],
                  'source_evidence_id': case['source_evidence_id'], 'comparisons': comparisons,
                  'normal_matches_mask_and_beats_expose_all_repeats': all(pattern),
                  'unique_action_sequences': {b: len({digest(r['branches'][b]['actions']) for r in repeats
                                                     if not r['branches'][b].get('error')}) for b in BRANCHES}}
        artifacts.append('diagnostic/cases', record)
        completed.append(record)
        artifacts.write('diagnostic/progress', {'completed_cases': len(completed), 'planned_cases': len(cases),
                                               'execution_errors': errors, 'updated_utc': stamp()})
        assert bank.fingerprint == original_hash
        print('CASE COMPLETE', dataset, index, {k: v['direction'] for k, v in comparisons.items()}, flush=True)
    assert Bank.load(out / f'{dataset}-m0.json').fingerprint == original_hash
    result = {'dataset': dataset, 'complete': True, 'protocol': 'scope-diagnostic-v1',
              'planned_cases': len(cases), 'completed_cases': len(completed), 'execution_errors': errors,
              'effects': {name: dict(Counter(c['comparisons'][name]['direction'] for c in completed)) for name in COMPARISONS},
              'neutral_behavior': {name: dict(Counter(p['neutral_behavior'] for c in completed
                  for p in c['comparisons'][name]['pairs'] if p.get('neutral_behavior'))) for name in COMPARISONS},
              'normal_matches_mask_and_beats_expose_cases': sum(c['normal_matches_mask_and_beats_expose_all_repeats'] for c in completed),
              'bank_unchanged': True, 'initial_bank_hash': original_hash, 'accepted_updates': 0,
              'diagnostic_only': True, 'effectiveness_proven': False, 'costs': artifacts.cost_summary()}
    artifacts.write('summary', result)
    model.client.close()
    atomic_json(out / f'{dataset}-summary.json', result)
    print('SCOPE DIAGNOSTIC RESULTS SAVED', dataset, flush=True)
