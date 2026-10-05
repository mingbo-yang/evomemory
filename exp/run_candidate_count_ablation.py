"""Isolated 1-vs-4 end-to-end pilot; leaves production model/defaults unchanged.

Uses the real Flow, coupled generation cache, and a persistent COMET worker in
its existing environment. Logical costs include cache hits; elapsed runtime is
not a fair per-arm speed comparison. No feedback is used to select test sources.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path('/mnt/huawei/ymb/model/laya-multilingual-accept-reject-20ep-continued-v1/checkpoints/epoch_20')


def comet_server(args):
    from comet import load_from_checkpoint
    model = load_from_checkpoint(str(args.checkpoint))
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(1)
    conn, _ = server.accept()
    with conn, conn.makefile('r') as reader, conn.makefile('w') as writer:
        for line in reader:
            rows = json.loads(line)
            if rows is None:
                break
            result = model.predict(rows, batch_size=8, gpus=1, num_workers=0, progress_bar=False)
            scores = [float(x) for x in (result.scores if hasattr(result, 'scores') else result['scores'])]
            writer.write(json.dumps(scores, allow_nan=False) + '\n')
            writer.flush()
    server.close()
    args.socket.unlink(missing_ok=True)


class PersistentFeedback:
    metric = 'comet'

    def __init__(self, cfg, output, gpu):
        from core.comet_feedback import from_config
        self.metadata = from_config(cfg).metadata
        self.cache = {}
        self.path = output / 'comet_score_cache.json'
        self.socket_path = Path('/tmp') / ('candidate-comet-' + str(os.getpid()) + '.sock')
        self.log = (output / ('comet_worker_' + str(os.getpid()) + '.log')).open('w')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        self.process = subprocess.Popen([cfg['python'], '-u', str(Path(__file__).resolve()), '--serve',
                     '--socket', str(self.socket_path), '--checkpoint', cfg['checkpoint']],
                     stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self.conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.conn.settimeout(600)
        deadline = time.monotonic() + 180
        while not self.socket_path.exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                self.close()
                raise RuntimeError('COMET worker startup failed; see comet_worker.log')
            time.sleep(.5)
        self.conn.connect(str(self.socket_path))
        self.reader = self.conn.makefile('r')
        self.writer = self.conn.makefile('w')

    def score_pairs(self, rows):
        import fcntl
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                self.cache = json.loads(self.path.read_text())
            return self._score_pairs(rows)

    def _score_pairs(self, rows):
        from core.comet_feedback import validated_scores
        import laya_acceptance_common as C
        pairs, pending = [], {}
        for row in rows:
            keys = []
            for field in ('current', 'candidate'):
                obj = {'src': row['source'], 'mt': row[field], 'ref': row['reference']}
                key = hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                keys.append(key)
                if key not in self.cache:
                    pending[key] = obj
            pairs.append(keys)
        if pending:
            self.writer.write(json.dumps(list(pending.values()), ensure_ascii=False) + '\n')
            self.writer.flush()
            line = self.reader.readline()
            if not line:
                raise RuntimeError('COMET worker exited without feedback')
            scores = json.loads(line)
            if len(scores) != len(pending):
                raise ValueError('COMET score count mismatch')
            self.cache.update(zip(pending, scores))
            C.dump(self.path, self.cache)
        return validated_scores(rows, [{'q_current': self.cache[a], 'q_candidate': self.cache[b]} for a, b in pairs])

    def close(self):
        if hasattr(self, 'writer'):
            try:
                self.writer.write('null\n'); self.writer.flush()
            except (OSError, ValueError):
                pass
        if getattr(self, 'process', None) is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        for name in ('reader', 'writer', 'conn', 'log'):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.close()
        if getattr(self, 'socket_path', None):
            self.socket_path.unlink(missing_ok=True)


def main(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import laya_acceptance_common as C
    from core.comet_feedback import from_config
    from core.manifest import read_manifest
    from laya_relaxed_flow import prepare
    from ablations.laya_candidate_count import choose_policy
    manifest = ROOT / 'data/manifests/wmt19_en_zh__test.jsonl'
    full = read_manifest(manifest)
    baseline_path = ROOT.parent / 'baseline/results_full_main_qwen35_vllm/wmt19_en_zh/qwen3-8b/Direct-Zero.jsonl'
    baseline = C.read_jsonl(baseline_path)
    by_index = {r['index']: r for r in baseline}
    assert len(by_index) == len(baseline) == len(full)
    for r in full:
        assert (r.source, r.reference) == (by_index[r.row_index]['source_text'], by_index[r.row_index]['reference_text'])
    indices = sorted(random.Random(20261004).sample(range(len(full)), args.samples))
    pilot = args.output.with_name(args.output.name + '_manifest.jsonl')
    if pilot.exists():
        raise FileExistsError(pilot)
    pilot.write_text(''.join(json.dumps(asdict(full[i]), ensure_ascii=False) + '\n' for i in indices))
    cfg_path = ROOT / 'configs/laya_binary_v1.json'
    cfg = json.loads(cfg_path.read_text())
    arm_names = ['k1_static', 'k4_static', 'k1_online', 'k4_online']
    prep = SimpleNamespace(config=cfg_path, checkpoint=args.checkpoint, manifest=pilot,
                 initial_memory=C.COLLECTION / 'initial_memory.jsonl', retrieval_config=C.COLLECTION / 'retrieval.json',
                 output=args.output, gpu=args.gpu, arms=['full_static', 'full_online'], candidates_per_round=1)
    protocol, refs = prepare(prep, from_config(cfg['feedback']))
    protocol.update(experiment='candidate-count-ablation-v1', flow_schema='laya-candidate-count-ablation-v1', arms=arm_names, policy=choose_policy(args.checkpoint), samples=len(refs), checkpoint=str(args.checkpoint),
                    candidates_per_round='specified per arm', arm_specs={n: {'candidates_per_round': int(n[1]), 'online': n.endswith('online')} for n in arm_names},
                    selection_seed=20261004, selected_indices=indices, selection_rule='random sample without replacement from baseline manifest; preserve source order',
                    baseline_manifest_sha256=C.digest(manifest), baseline_results_sha256=C.digest(baseline_path),
                    coupling='shared drafts; request cache keyed by model/prompt/seed/sampling; slot 0 identical for identical state/retrieval',
                    evaluation='paired 1-vs-4 final COMET; report bootstrap CI and logical token costs; not equal compute budget',
                    execution_comet_gpu=args.comet_gpu, execution_generator_gpus={'k1': '0', 'k4': '3'},
                    parallel_request_consistency='per-request file locks and shared cache', production_defaults_changed=False)
    sources = ['run_candidate_count_ablation.py', 'laya_relaxed_flow.py', 'laya_relaxed_policy.py',
               'laya_acceptance_common.py', 'core/candidate_validation.py', 'core/hybrid_retrieval.py',
               'core/pipeline.py', 'core/experience.py', 'core/comet_feedback.py',
               'ablations/laya_candidate_count.py', 'core/refinement_instructions.py']
    protocol['code_sha256'] = {p: C.digest(ROOT / p) for p in sources}
    C.dump(args.output / 'protocol.json', protocol)
    pilot.unlink()
    print('FROZEN', json.dumps({'sources': len(refs), 'arms': arm_names, 'checkpoint': str(args.checkpoint)}), flush=True)
    # Concurrent engines, identical test inputs. Workers own separate memory banks.
    processes = []
    import signal
    try:
        for k, gpu in ((1, '0'), (4, '3')):
            log = (args.output / f'worker_k{k}.log').open('w')
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker-k', str(k),
                        '--output', str(args.output), '--checkpoint', str(args.checkpoint), '--gpu', gpu,
                        '--comet-gpu', args.comet_gpu], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
            processes.append((k, process))
            print('WORKER', k, 'gpu', gpu, 'pid', process.pid, flush=True)
        codes = {k: process.wait() for k, process in processes}
        if any(codes.values()):
            raise RuntimeError(f'Worker failures {codes}; inspect worker logs')
        parts = [json.loads((args.output / f'report_k{k}.json').read_text()) for k in (1, 4)]
        report = {'status': 'complete', 'arms': {n: a for part in parts for n, a in part['arms'].items()},
                  'historical_direct_zero': parts[0]['historical_direct_zero'],
                  'execution': {key: sum(part['execution'][key] for part in parts) for key in ('unique_generator_requests', 'cache_hits', 'unique_verifier_pairs')}}
        report['execution']['unique_comet_texts'] = len(json.loads((args.output / 'comet_score_cache.json').read_text()))
        C.dump(args.output / 'report.json', report)
        print('COMPLETE', flush=True)
    finally:
        for _, process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def worker(args):
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import fcntl
    import laya_acceptance_common as C
    from core.manifest import read_manifest
    from laya_relaxed_flow import Generator, Verifier
    from ablations.laya_candidate_count import Flow

    class LockedGenerator(Generator):
        def generate(self, requests):
            # Per-request locks prevent two GPUs sampling different realizations of
            # an identical request. Sorted lock acquisition prevents deadlock.
            keys = set()
            for request in requests:
                payload = {'prompt': request['prompt'], 'system': self.adapter.system_prompt(), 'seed': request['seed'],
                           'temperature': .1, 'top_p': 1., 'max_tokens': 1024, 'model': self.p['generator_path']}
                keys.add(hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
            locks = []
            try:
                for key in sorted(keys):
                    lock = (self.output / 'generation_cache' / (key + '.lock')).open('a')
                    locks.append(lock)
                    fcntl.flock(lock, fcntl.LOCK_EX)
                return super().generate(requests)
            finally:
                for lock in reversed(locks):
                    lock.close()

    protocol = json.loads((args.output / 'protocol.json').read_text())
    protocol = dict(protocol, gpu=args.gpu, candidates_per_round=args.worker_k)
    refs = read_manifest(args.output / 'test_manifest.jsonl')
    cfg = json.loads((ROOT / 'configs/laya_binary_v1.json').read_text())
    feedback = PersistentFeedback(cfg['feedback'], args.output, args.comet_gpu)
    try:
        generator = LockedGenerator(protocol, args.output)
        verifier = Verifier(args.checkpoint)
        flow = Flow(protocol, generator, verifier, feedback, args.output)
        if args.worker_k == 1:
            drafts = flow.drafts(refs)
        else:
            deadline = time.monotonic() + 1800
            while not (args.output / 'shared_drafts.json').exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Shared drafts were not produced by k1 worker')
                time.sleep(1)
            drafts = json.loads((args.output / 'shared_drafts.json').read_text())
        report = {'arms': {}, 'status': 'running'}
        for suffix in ('static', 'online'):
            name = f'k{args.worker_k}_{suffix}'
            print('START', name, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
            started = time.monotonic()
            results = flow.run(name, refs, drafts, online=suffix == 'online')
            pairs = [{'source': ref.source, 'current': draft['text'], 'candidate': row['final'], 'reference': ref.reference}
                     for ref, draft, row in zip(refs, drafts, results)]
            scores = feedback.score_pairs(pairs)
            report['arms'][name] = {'sources': len(results), 'mean_delta_comet': sum(s['delta'] for s in scores) / len(scores),
                                   'final_task_feedback': scores, 'elapsed_including_cache_and_feedback_s': time.monotonic() - started}
            C.dump(args.output / f'report_k{args.worker_k}.json', report)
            print('DONE', name, report['arms'][name]['mean_delta_comet'], flush=True)
        if args.worker_k == 1:
            baseline = C.read_jsonl(ROOT.parent / 'baseline/results_full_main_qwen35_vllm/wmt19_en_zh/qwen3-8b/Direct-Zero.jsonl')
            by_index = {r['index']: r for r in baseline}
            pairs = [{'source': r.source, 'current': d['text'], 'candidate': by_index[r.row_index]['final_output'], 'reference': r.reference}
                     for r, d in zip(refs, drafts)]
            report['historical_direct_zero'] = {'note': 'Same sources, historical outputs and seed; not the shared-draft control',
                                              'scores': feedback.score_pairs(pairs)}
        report['status'] = 'complete'
        report['execution'] = {'unique_generator_requests': generator.calls, 'cache_hits': generator.hits,
                               'unique_verifier_pairs': verifier.scored}
        C.dump(args.output / f'report_k{args.worker_k}.json', report)
        print('COMPLETE', flush=True)
    finally:
        feedback.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serve', action='store_true')
    p.add_argument('--worker-k', type=int, choices=[1, 4])
    p.add_argument('--socket', type=Path)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--output', type=Path, default=ROOT / 'runs/laya_candidate_ablation_v1')
    p.add_argument('--gpu', default='3')
    p.add_argument('--comet-gpu', default='1')
    p.add_argument('--samples', type=int, default=128)
    a = p.parse_args()
    if a.serve:
        comet_server(a)
    elif a.worker_k:
        worker(a)
    else:
        main(a)
