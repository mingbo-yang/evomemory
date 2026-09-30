"""Local Qwen protocol-v3 validation; finite tasks, no wall-time deadline.

Natural evolution and optional identity/editor diagnostics have separate artifacts.
Diagnostics never deploy a candidate and never count as evidence of task gains.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import httpx

from .core import Artifacts, Bank, atomic_json, load_manifest, validate_tasks
from .engine import EvoScope, Proposal
from .environments import AlfWorldEnvironment, WebShopEnvironment
from .local_watchdog import gpu_snapshot, progress, rows, stamp, stop
from .models import LocalModel, edit_condition
from .runner import Runner

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('alfworld', 'webshop')
MODEL = 'qwen3.8-27b-local'
URL = 'http://127.0.0.1:8124/v1'


def clean_env():
    env = os.environ.copy()
    for key in ('EVOSCOPE_API_KEY', 'EVOSCOPE_CREDENTIALS_FILE', 'OPENAI_API_KEY', 'OPENAI_BASE_URL'):
        env.pop(key, None)
    env.update(TMPDIR='/tmp', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def select_tasks(tasks, count=12):
    """Fixed manifest order, without inspecting outcomes or hidden task fields."""
    learn = [t for t in tasks if t.split == 'learn'][:count]
    if len(learn) != count:
        raise ValueError('insufficient learn tasks')
    selected = learn + [t for t in tasks if t.split == 'gate']
    validate_tasks(selected)
    return selected


def prepare(out, source):
    out.mkdir(parents=True, exist_ok=False)
    bank_hashes = {}
    for dataset in DATASETS:
        tasks = select_tasks(load_manifest(source / f'{dataset}-manifest.json'))
        atomic_json(out / f'{dataset}-manifest.json', [asdict(t) for t in tasks])
        shutil.copy2(source / f'{dataset}-m0.json', out / f'{dataset}-m0.json')
        bank_hashes[dataset] = Bank.load(out / f'{dataset}-m0.json').fingerprint
    atomic_json(out / 'plan.json', {
        'created_utc': stamp(), 'source': str(source), 'model': MODEL, 'gpu': 1,
        'local_only': True, 'initial_bank_hashes': bank_hashes,
        'selection': 'first 12 learn tasks in fixed source order; all reserved gate tasks',
        'datasets': DATASETS, 'held_out_test_used': False, 'wall_time_limit': None,
        'batch_size': 2, 'probe_limits': [6, 3], 'probe_checkpoints_per_round': 1,
        'probe_repeats': 3, 'gate_repeats': 3, 'gate_size': 4, 'gate_local_size': 2,
        'max_candidates_per_policy': 2, 'min_edit_checkpoints': 3, 'min_non_neutral': 1,
        'max_tokens': 8192, 'seed': 42, 'max_steps': {'alfworld': 30, 'webshop': 20},
        'diagnostics': 'If natural editor/gate absent: one editor I/O audit from >=3 real compatible states; one identity dry-run gate on first unused precommitted eligible block. Separate artifacts, no deployment, no fabricated evidence, no outcome-based retries.',
        'limitation': 'Protocol validation only; not a held-out effectiveness comparison.'})
    snapshot = out / 'source'
    snapshot.mkdir()
    for filename in ('protocol_validation.py', 'engine.py', 'gate_plan.py', 'core.py',
                     'runner.py', 'models.py', 'editor_budget.py', 'environments.py'):
        shutil.copy2(ROOT / 'evoscope' / filename, snapshot / filename)
    shutil.copy2(ROOT / 'evoscope/scripts/serve_qwen_local.sh', snapshot / 'serve_qwen_local.sh')


def sampling_audit(records):
    """Check each observed one-slot decision against the declared allocation rule."""
    violations, informative, carried = [], 0, 0
    for row in records:
        available, prior, chosen = row['available'], row['cumulative_before'], row['selected']
        if sum(prior.values()):
            carried += 1
        if all(available.values()) and sum(chosen.values()) == 1:
            informative += 1
            selected = next(k for k, v in chosen.items() if v)
            other = 'failure' if selected == 'success' else 'success'
            if prior[selected] > prior[other]:
                violations.append(row)
    return {'mixed_stratum_decisions': informative, 'rounds_with_prior_counts': carried,
            'violations': violations, 'rule_consistent_on_observed_decisions': not violations}


def run_worker(out, dataset):
    artifacts = Artifacts(out / dataset)
    tasks = load_manifest(out / f'{dataset}-manifest.json')
    bank_path = out / f'{dataset}-m0.json'
    bank = Bank.load(bank_path)
    initial_hash = bank.fingerprint
    max_steps = 30 if dataset == 'alfworld' else 20
    factory = ((lambda: AlfWorldEnvironment(30)) if dataset == 'alfworld' else
               (lambda: WebShopEnvironment(str(ROOT / 'downloads/WebShop-10k'), num_products=None)))
    model = LocalModel(artifacts, MODEL, URL, max_tokens=8192, send_seed=True)
    artifacts.write('model_config', model.config)
    runner = Runner(factory, model, artifacts, max_steps)
    engine = EvoScope(bank, runner, tasks, artifacts)
    engine.probe_limits = (6, 3)
    print('START NATURAL EVOLUTION', dataset, flush=True)
    evolution = engine.evolve(batch_size=2, probe_checkpoints=1, repeats=3,
                              gate_size=4, gate_local_size=2, max_candidates_per_policy=2,
                              gate_repeats=3, seed=42)
    print('NATURAL EVOLUTION COMPLETE', dataset, json.dumps(evolution), flush=True)
    natural_editor_calls = sum(r['kind'] == 'model' and r['phase'] == 'edit' for r in artifacts.costs)
    diagnostics = {}
    eligible = [(p, engine.compatible_evidence(p.id)) for p in engine.bank.policies]
    eligible = [(p, evidence) for p, evidence in eligible if len(evidence) >= 3]
    if not natural_editor_calls and eligible:
        policy, evidence = eligible[0]
        audit = Artifacts(out / f'{dataset}-editor-audit')
        audit.write('context', {'diagnostic_only': True, 'bypasses_signed_threshold_for_io_only': True,
                    'evidence_ids': [e['id'] for e in evidence], 'distinct_states': len(evidence),
                    'effects': [e['public']['direction'] for e in evidence],
                    'natural_evidence_source': str(artifacts.root / 'probe/evidence.jsonl')})
        audit_model = LocalModel(audit, MODEL, URL, max_tokens=8192, send_seed=True)
        try:
            when = edit_condition(audit_model, asdict(policy), [e['public'] for e in evidence], 900042)
            diagnostics['editor'] = {'status': 'completed', 'distinct_states': len(evidence),
                                     'noop': when is None, 'when': when, 'deployed': False}
        except Exception as exc:
            diagnostics['editor'] = {'status': 'error', 'error': type(exc).__name__, 'deployed': False}
        audit.write('result', diagnostics['editor'])
    else:
        diagnostics['editor'] = {'status': 'natural_editor_executed' if natural_editor_calls else 'insufficient_real_states'}
    if not evolution['gate_attempts']:
        eligible_ids = {p.id for p, _ in eligible}
        block = next((b for b in engine.gate_blocks if b.policy_id in eligible_ids
                      and b.id not in engine.attempted_gate_blocks
                      and not {engine.tasks[t].group_id for t in b.task_ids} & engine.used_gate_groups), None)
        if block:
            audit = Artifacts(out / f'{dataset}-gate-audit')
            audit.write('context', {'diagnostic_only': True, 'identity_candidate': True,
                        'block': asdict(block), 'precommitted_plan': str(artifacts.root / 'gate/plan.json')})
            audit_model = LocalModel(audit, MODEL, URL, max_tokens=8192, send_seed=True)
            audit_engine = EvoScope(Bank.load(artifacts.root / 'bank.json'),
                                   Runner(factory, audit_model, audit, max_steps), tasks, audit)
            audit_engine.evidence = dict(engine.evidence)
            audit_engine.gate_blocks = engine.gate_blocks
            audit_engine.used_gate_groups = set(engine.used_gate_groups)
            policy = audit_engine.bank.get(block.policy_id)
            evidence = audit_engine.compatible_evidence(policy.id)
            proposal = Proposal(audit_engine.bank.fingerprint, policy.id, policy.when,
                                tuple(e['id'] for e in evidence))
            diagnostics['gate'] = audit_engine.gate(proposal, list(block.task_ids),
                                                   seed=950042, dry_run=True, repeats=3, block=block)
            assert not diagnostics['gate']['accepted']
            assert audit_engine.bank.fingerprint == engine.bank.fingerprint
        else:
            diagnostics['gate'] = {'status': 'no_unused_precommitted_block_with_enough_real_evidence'}
    else:
        diagnostics['gate'] = {'status': 'natural_gate_executed'}
    assert Bank.load(bank_path).fingerprint == initial_hash
    plan = json.loads((artifacts.root / 'gate/plan.json').read_text())
    summary = {'dataset': dataset, 'complete': True, 'initial_bank_hash': initial_hash,
               'evolution': evolution, 'natural_editor_calls': natural_editor_calls,
               'evidence_effects': dict(Counter(e['public']['direction'] for e in engine.evidence.values())),
               'sampling_audit': sampling_audit(rows(artifacts.root / 'probe/sampling.jsonl')),
               'gate_plan': {'hash': plan['plan_hash'], 'blocks': len(plan['blocks']), 'shortages': plan['shortages']},
               'diagnostics': diagnostics, 'effectiveness_proven': False}
    # This durable receipt precedes native interpreter teardown, which can hang.
    atomic_json(out / f'{dataset}-summary.json', summary)
    print('WORKER RESULTS SAVED', dataset, flush=True)


def worker_complete(out, dataset):
    path = out / f'{dataset}-summary.json'
    return path.exists() and json.loads(path.read_text()).get('complete') is True


def monitor(out, trial="protocol-v3"):
    server, workers, records = None, {}, []
    baseline, start = gpu_snapshot(), time.monotonic()
    atomic_json(out / 'watchdog.json', {'pid': os.getpid(), 'started_utc': stamp()})
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 8124))
        free = int(subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'], text=True))
        if free < 57000:
            raise RuntimeError(f'GPU 1 free VRAM insufficient: {free} MiB')
        env = clean_env()
        env.update(EVOSCOPE_GPU_MEMORY_UTILIZATION='0.66', EVOSCOPE_CPU_OFFLOAD_GB='0',
                   EVOSCOPE_KV_CACHE_BYTES='3221225472')
        with (out / 'server.log').open('w') as log:
            server = subprocess.Popen(['bash', str(ROOT / 'evoscope/scripts/serve_qwen_local.sh')],
                                      cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        records.append({'role': 'server', 'pid': server.pid, 'pgid': server.pid, 'gpu': 1})
        atomic_json(out / 'owned-processes.json', records)
        with httpx.Client(trust_env=False, timeout=2) as client:
            startup = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError('server exited during startup')
                try:
                    if client.get('http://127.0.0.1:8124/health').status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > startup:
                    raise RuntimeError('server startup timeout')
                atomic_json(out / 'status.json', {'updated_utc': stamp(), 'phase': 'server_startup'})
                time.sleep(5)
        for dataset in DATASETS:
            env = clean_env()
            env['CUDA_VISIBLE_DEVICES'] = ''
            with (out / f'{dataset}-worker.log').open('w') as log:
                worker = subprocess.Popen([sys.executable, '-u', '-m', 'evoscope.protocol_validation',
                    '--output', str(out), '--worker-dataset', dataset, '--trial', trial], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            workers[dataset] = worker
            records.append({'role': 'worker', 'dataset': dataset, 'pid': worker.pid, 'pgid': worker.pid})
            atomic_json(out / 'owned-processes.json', records)
        completed = set()
        while len(completed) < len(DATASETS):
            if server.poll() is not None:
                raise RuntimeError('server exited during validation')
            for dataset, worker in workers.items():
                if dataset in completed:
                    continue
                if worker_complete(out, dataset):
                    stop(worker)  # Complete artifacts, even if native teardown never returns.
                    completed.add(dataset)
                elif worker.poll() is not None:
                    raise RuntimeError(f'{dataset} exited without completion receipt: {worker.returncode}')
            atomic_json(out / 'status.json', {'updated_utc': stamp(), 'phase': 'validation',
                'elapsed_seconds': time.monotonic() - start, 'completed': sorted(completed),
                'progress': progress(out)})
            if len(completed) < len(DATASETS):
                time.sleep(10)
    except BaseException as exc:
        atomic_json(out / 'failure.json', {'time': stamp(), 'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        for worker in workers.values():
            stop(worker)
        owned_gpu = []
        for line in gpu_snapshot().get('processes', '').splitlines():
            try:
                pid = int(line.split(',')[0])
                if server is not None and os.getpgid(pid) == server.pid:
                    owned_gpu.append(pid)
            except (ValueError, ProcessLookupError):
                pass
        if server is not None:
            stop(server)
        time.sleep(3)
        after = gpu_snapshot()
        remaining = {int(line.split(',')[0]) for line in after.get('processes', '').splitlines() if line.strip()}
        cleanup = {'time': stamp(), 'owned_gpu_pids': owned_gpu,
                   'owned_gpu_pids_remaining': sorted(set(owned_gpu) & remaining),
                   'verified': 'error' not in after and not set(owned_gpu) & remaining,
                   'before': baseline, 'after': after}
        atomic_json(out / 'cleanup.json', cleanup)
        summary = {d: json.loads((out / f'{d}-summary.json').read_text()) for d in DATASETS if worker_complete(out, d)}
        atomic_json(out / 'final-status.json', {'finished_utc': stamp(), 'trial': trial,
            'all_requested_completed': len(summary) == len(DATASETS), 'gpu_cleanup_verified': cleanup['verified'],
            'elapsed_seconds': time.monotonic() - start, 'datasets': summary})
        report = [f'# Qwen local validation: {trial}', '',
                  f'Completed datasets: {len(summary)}/2. Owned GPU cleanup verified: {cleanup["verified"]}.', '',
                  'Natural evolution and diagnostic-only calls are separate. This is not a held-out effectiveness comparison.', '']
        for dataset, value in summary.items():
            if trial == 'probe-scope':
                report.append(f'- {dataset}: {value["completed_cases"]}/{value["planned_cases"]} cases; '
                              f'{value["execution_errors"]} execution errors; '
                              f'bank unchanged: {value["bank_unchanged"]}; '
                              f'comparisons: {value["effects"]}.')
                continue
            report.append(f'- {dataset}: {value["evolution"]["accepted_edits"]} natural updates; '
                          f'effects {value["evidence_effects"]}; sampling {value["sampling_audit"]}; '
                          f'editor audit {value["diagnostics"]["editor"].get("status")}; '
                          f'gate audit {value["diagnostics"]["gate"].get("status")}.')
        (out / 'REPORT.md').write_text('\n'.join(report) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--source')
    parser.add_argument('--worker-dataset', choices=DATASETS)
    parser.add_argument('--trial', choices=('protocol-v3', 'probe-scope'), default='protocol-v3',
                        help='probe-scope is an isolated frozen-bank diagnostic; default unchanged')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if args.trial == 'probe-scope':
        from .scope_trial import prepare as prepare_trial, run_worker as run_trial_worker
    else:
        prepare_trial, run_trial_worker = prepare, run_worker
    if args.worker_dataset:
        run_trial_worker(out, args.worker_dataset)
    else:
        if not args.source:
            parser.error('--source required for monitor')
        prepare_trial(out, Path(args.source).resolve())
        monitor(out, trial=args.trial)


if __name__ == '__main__':
    main()
