"""Current binary Laya pipeline: direct actions and independent delayed COMET memory admission.

The historical filename remains an entry point; the old three-class experiment
is explicitly preserved under legacy_laya_v1, with separate artifacts.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

from core import resolve_model_config
from core.bm25_fields import Experience
from core.experience import load_experiences, save_experiences, render_experience_block
from core.manifest import read_manifest
from core.optimized_pipeline import INSTRUCTIONS
from core.comet_feedback import from_config, validate_epsilon, validated_scores
from acceptance_data import valid_candidate
import laya_acceptance_common as C
from laya_relaxed_policy import choose_policy, select_candidate
from prepare_acceptance_training import source_key

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/laya_binary_flow_v1'
TASK = 'wmt19_en_zh'


def stable_seed(*parts):
    return C.seed_for('binary-flow-v1', *parts) % (2 ** 31)


def memory_key(source, before, after):
    return hashlib.sha256(json.dumps([source, before, after], ensure_ascii=False).encode()).hexdigest()


class Generator:
    def __init__(self, protocol, output):
        from baseline_core.llm import LLMClient
        from baseline_core.tasks import get_adapter
        self.p = protocol
        self.output = Path(output)
        self.client = LLMClient(resolve_model_config(protocol['generator']), backend='vllm', gpu=protocol['gpu'],
                               gpu_memory_utilization=protocol['gpu_memory_utilization'], max_model_len=4096, enforce_eager=True)
        self.adapter = get_adapter(TASK)
        self.calls = 0
        self.hits = 0
        (self.output / 'generation_cache').mkdir(exist_ok=True)

    def generate(self, requests):
        from vllm import SamplingParams
        result = [None] * len(requests)
        pending = {}
        for i, request in enumerate(requests):
            payload = {'prompt': request['prompt'], 'system': self.adapter.system_prompt(), 'seed': request['seed'],
                       'temperature': .1, 'top_p': 1., 'max_tokens': 1024, 'model': self.p['generator_path']}
            key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            path = self.output / 'generation_cache' / f'{key}.json'
            if path.exists():
                result[i] = json.loads(path.read_text())
                self.hits += 1
                continue
            text = self.client._chat_text(payload['system'], payload['prompt'])
            n = self.client.count_tokens(text)
            if n + 1024 > 4096:
                result[i] = {'text': '', 'finish_reason': 'context_length_exceeded', 'input_tokens': n, 'output_tokens': 0}
                continue
            if key not in pending:
                pending[key] = {'indices': [], 'text': text, 'request': request, 'path': path}
            pending[key]['indices'].append(i)
        items = list(pending.items())
        if items:
            params = [SamplingParams(temperature=.1, top_p=1., max_tokens=1024, seed=item['request']['seed'],
                                     stop_token_ids=self.client.stop_token_ids or None) for _, item in items]
            outputs = self.client.model.generate([item['text'] for _, item in items], params, use_tqdm=False)
            if len(outputs) != len(items):
                raise ValueError('Generator output count mismatch')
            self.calls += len(items)
            for (key, item), output in zip(items, outputs):
                g = output.outputs[0]
                row = {'text': self.adapter.parse_output(g.text), 'raw_text': g.text, 'finish_reason': g.finish_reason,
                       'input_tokens': len(output.prompt_token_ids), 'output_tokens': len(g.token_ids),
                       'seed': item['request']['seed'], 'cache_key': key}
                C.dump(item['path'], row)
                for i in item['indices']:
                    result[i] = row
        return result


class Verifier:
    def __init__(self, checkpoint, device='cuda', diagnostic_temperature=1.0):
        self.cfg = C.validate_checkpoint(checkpoint)
        self.tok = C.get_tokenizer(checkpoint)
        self.model, _ = C.load_model(checkpoint)
        self.model.to(device)
        self.diagnostic_temperature = diagnostic_temperature
        self.cache = {}
        self.scored = 0
        empty = C.EncodedPairs([], self.tok)
        self.overhead = max(len(p[0]) for p in empty.prefixes.values()) + 1

    def predict(self, rows):
        pending, out = {}, {}
        for row in rows:
            inputs = C.model_input(row['input'])
            if C.is_noop(inputs):
                out[row['id']] = {'decision': 'Reject', 'status': 'no_op',
                                  'decision_origin': 'deterministic_rule', 'probabilities': None}
                continue
            key = hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if key in self.cache:
                out[row['id']] = self.cache[key]
                continue
            length = len(self.tok(C.state_text(inputs).replace(self.tok.mask_token, ' '), add_special_tokens=False)['input_ids']) + self.overhead
            if length > 1024:
                out[row['id']] = {'decision': 'Reject', 'status': 'context_overflow', 'probabilities': None}
                continue
            pending.setdefault(key, {'row': {'id': key, 'input': inputs}, 'ids': []})['ids'].append(row['id'])
        if pending:
            items = list(pending.items())
            encoded = C.EncodedPairs([v['row'] for _, v in items], self.tok)
            logits = C.infer(self.model, encoded, batch_size=16)
            classes = C.decisions(logits)
            probabilities = C.probabilities(logits, self.diagnostic_temperature)
            self.scored += len(items)
            for (key, obj), z, decision, p in zip(items, logits, classes, probabilities):
                self.cache[key] = {'decision': decision, 'status': 'classified', 'logits': z.tolist(),
                                   'probabilities': dict(zip(C.LABELS, p.tolist()))}
                for ident in obj['ids']:
                    out[ident] = self.cache[key]
        return out


def feedback_pairs(trace, reference):
    """Enumerate every valid modification, regardless of verifier decision."""
    pairs, locations = [], []
    for rd in trace['rounds']:
        before = rd['before']
        for candidate in rd['candidates']:
            if not candidate.get('valid') or not candidate.get('changed') or candidate['text'] == before:
                continue
            pairs.append({'source': trace['source'], 'current': before, 'candidate': candidate['text'], 'reference': reference})
            locations.append((rd, candidate))
    return pairs, locations


def positive_memories(trace, reference, feedback, existing, epsilon_memory, audit=None, precomputed=None):
    """Task is already complete. Rejected positives qualify independently."""
    epsilon_memory = validate_epsilon(epsilon_memory, 'epsilon_memory')
    if feedback.metric != 'comet':
        raise ValueError('Memory feedback must use COMET in this protocol')
    pairs, locations = feedback_pairs(trace, reference)
    raw_scores = feedback.score_pairs(pairs) if precomputed is None and pairs else (precomputed or [])
    scores = validated_scores(pairs, raw_scores)
    result = []
    for pair, (rd, candidate), score in zip(pairs, locations, scores):
        delta = score['delta']
        key = memory_key(trace['source'], pair['current'], pair['candidate'])
        eligible = delta > epsilon_memory
        added = eligible and key not in existing
        if audit is not None:
            audit.append({'candidate_id': candidate.get('id'), 'round': rd.get('round'), **score,
                          'decision': candidate.get('decision'), 'positive_feedback': eligible,
                          'memory_candidate': added, 'epsilon_memory': epsilon_memory})
        if not added:
            continue
        existing.add(key)
        result.append(Experience(exp_id='online-' + key, task=TASK, model='qwen3-8b', source_input=trace['source'],
                                 state_before=pair['current'], state_after=pair['candidate'],
                                 intervention_instruction=INSTRUCTIONS[TASK],
                                 intervention_rationale='Positive delayed COMET feedback after task completion.',
                                 verdict='better', reason_a='', reason_b='', order_consistent=False,
                                 delta_offline=delta, provenance=C.SCHEMA + '_delayed_comet', outcome_label='helped'))
    return result


class Flow:
    def __init__(self, protocol, generator, verifier, feedback, output, *, encoder=None, retriever_class=None):
        from core.hybrid_retrieval import LocalSentenceEncoder, HybridExperienceRetriever
        self.p, self.gen, self.verifier, self.feedback = protocol, generator, verifier, feedback
        self.output = Path(output)
        cfg = json.loads((self.output / 'retrieval.json').read_text())
        self.options = dict(cfg['retrieval'])
        self.options.pop('method', None)
        self.encoder = encoder if encoder is not None else LocalSentenceEncoder(cfg['encoder'], 'cpu')
        self.retriever_class = retriever_class or HybridExperienceRetriever

    def drafts(self, refs):
        from baseline_core.types import TaskExample
        requests = [{'prompt': self.gen.adapter.initial_prompt(TaskExample(index=i, source=r.source, reference='', task=TASK)),
                     'seed': stable_seed(r.sample_id, 'draft')} for i, r in enumerate(refs)]
        outputs = []
        for start in range(0, len(requests), self.p['batch_size']):
            outputs.extend(self.gen.generate(requests[start:start + self.p['batch_size']]))
        C.dump(self.output / 'shared_drafts.json', outputs)
        return outputs

    def run(self, name, refs, drafts, online=False):
        from baseline_core.types import TaskExample
        from core.pipeline import build_refine_prompt
        if len(refs) != len(drafts):
            raise ValueError('Draft/reference count mismatch')
        lib = load_experiences(self.output / 'initial_memory.jsonl')
        retriever = self.retriever_class(lib, self.encoder, **self.options)
        existing = {memory_key(e.source_input, e.state_before, e.state_after) for e in lib}
        directory = self.output / name
        directory.mkdir(exist_ok=False)
        records = []
        for start in range(0, len(refs), self.p['batch_size']):
            batch = refs[start:start + self.p['batch_size']]
            traces = [{'sample_id': r.sample_id, 'source': r.source, 'initial': d['text'], 'final': d['text'],
                       'draft_valid': valid_candidate('', d)[0], 'rounds': [], 'bank_size_at_start': len(lib)}
                      for r, d in zip(batch, drafts[start:start + len(batch)])]
            active = [i for i, tr in enumerate(traces) if tr['draft_valid']]
            by_id = {e.exp_id: e for e in lib}
            for round_index in range(1, self.p['max_rounds'] + 1):
                if not active:
                    break
                requests, rounds = [], {}
                for i in active:
                    tr = traces[i]
                    before = tr['final']
                    ret = retriever.retrieve(tr['source'], before, alpha=.5, k=4, exclude_source=tr['source'])
                    block = render_experience_block([by_id[k] for k in ret.exp_ids], count_tokens=self.gen.client.count_tokens,
                                                    max_units=4, contrastive=True, advice_mode='summary').replace('Refined Translation (Gold):', 'Refined Translation:')
                    prompt = build_refine_prompt(self.gen.adapter, TaskExample(index=start + i, source=tr['source'], reference='', task=TASK),
                                                 before, INSTRUCTIONS[TASK], experience_block=block, renderer='v2')
                    rounds[i] = {'round': round_index, 'before': before, 'retrieved_ids': ret.exp_ids,
                                 'online_retrieved': sum(k.startswith('online-') for k in ret.exp_ids),
                                 'candidates': [], 'selected_slot': None, 'accepted': False, 'bank_size': len(lib)}
                    requests.extend({'prompt': prompt, 'seed': stable_seed(tr['sample_id'], round_index, slot)}
                                    for slot in range(self.p['candidates_per_round']))
                generated = self.gen.generate(requests)
                if len(generated) != len(requests):
                    raise ValueError('Candidate count mismatch')
                score_rows = []
                k = self.p['candidates_per_round']
                for pos, i in enumerate(active):
                    rd = rounds[i]
                    for slot, g in enumerate(generated[k * pos:k * (pos + 1)]):
                        valid, why = valid_candidate(rd['before'], g)
                        candidate = dict(g, slot=slot, valid=valid, validity_reason=why, changed=g['text'] != rd['before'],
                                         id=f"{traces[i]['sample_id']}:{round_index}:{slot}", decision=None)
                        if not candidate['changed']:
                            candidate.update(decision='Reject', status='no_op',
                                             decision_origin='deterministic_rule', probabilities=None)
                        rd['candidates'].append(candidate)
                        if name != 'unfiltered_static' and valid and candidate['changed']:
                            score_rows.append({'id': candidate['id'], 'input': {'source': traces[i]['source'], 'current': rd['before'], 'candidate': g['text']}})
                scored = self.verifier.predict(score_rows) if score_rows else {}
                next_active = []
                for i in active:
                    rd = rounds[i]
                    for candidate in rd['candidates']:
                        if candidate['id'] in scored:
                            candidate.update(scored[candidate['id']])
                        elif name != 'unfiltered_static' and candidate['valid'] and candidate['changed']:
                            raise ValueError('Missing verifier decision; do not silently accept')
                    selected = select_candidate(rd['candidates'], self.p['policy'], 'unfiltered' if name == 'unfiltered_static' else 'laya')
                    if selected is not None:
                        rd.update(selected_slot=selected, accepted=True)
                        traces[i]['final'] = rd['candidates'][selected]['text']
                        next_active.append(i)
                    traces[i]['rounds'].append(rd)
                active = next_active
            # No feedback is computed until every task in this batch has finished.
            additions, feedback_audit = [], []
            task_pairs = [feedback_pairs(tr, ref.reference)[0] for tr, ref in zip(traces, batch)]
            all_pairs = [pair for pairs in task_pairs for pair in pairs]
            # One COMET worker invocation per completed batch, not per candidate/task.
            all_scores = validated_scores(all_pairs, self.feedback.score_pairs(all_pairs)) if all_pairs else []
            offset = 0
            for tr, ref, pairs in zip(traces, batch, task_pairs):
                additions.extend(positive_memories(tr, ref.reference, self.feedback, existing,
                                                   self.p['epsilon_memory'], audit=feedback_audit,
                                                   precomputed=all_scores[offset:offset + len(pairs)]))
                offset += len(pairs)
            if online:
                lib.extend(additions)
                if additions:
                    retriever = retriever.rebuild(lib)
            C.dump(directory / f'feedback_{start:06d}.json', {'after_completed_sources': start + len(batch),
                   'candidates': feedback_audit, 'admitted': [e.exp_id for e in additions] if online else [],
                   'memory_frozen': not online, 'bank_size': len(lib)})
            C.dump(directory / f'batch_{start:06d}.json', traces)
            records.extend(traces)
        save_experiences(lib, directory / 'final_memory.jsonl')
        C.dump(directory / 'results.json', records)
        return records


def prepare(args, feedback):
    cfg = json.loads(args.config.read_text())
    if cfg.get('decision_schema') != C.SCHEMA:
        raise ValueError('Expected current binary protocol config')
    epsilon_memory = validate_epsilon(cfg['epsilon_memory'], 'epsilon_memory')
    policy = choose_policy(args.checkpoint)
    checkpoint_cfg = C.validate_checkpoint(args.checkpoint)
    trained_feedback = checkpoint_cfg.get('label_feedback', {})
    current_feedback = feedback.metadata
    if (trained_feedback.get('metric') != 'comet'
            or trained_feedback.get('checkpoint_sha256') != current_feedback.get('checkpoint_sha256')
            or trained_feedback.get('epsilon') != validate_epsilon(cfg['label_epsilon'])):
        raise ValueError('Checkpoint must match the configured COMET feedback and offline label definition')
    if cfg.get('labels') != C.LABELS or cfg.get('input_schema') != list(C.INPUT_FIELDS):
        raise ValueError('Binary protocol input/output schema mismatch')
    refs = read_manifest(args.manifest)
    seen = set(checkpoint_cfg.get('seen_source_hashes', []))
    memory = load_experiences(args.initial_memory)
    memory_sources = {source_key(e.source_input) for e in memory}
    keys = [source_key(r.source) for r in refs]
    if (not seen or not refs or any(not r.source.strip() or not r.reference.strip() for r in refs)
            or len(keys) != len(set(keys)) or set(keys) & (seen | memory_sources)):
        raise ValueError('Require unique held-out sources isolated from model-development data and initial memory')
    if args.output.exists():
        raise FileExistsError('Use a fresh binary rollout directory; old results must not be overwritten')
    args.output.mkdir(parents=True)
    shutil.copy2(args.manifest, args.output / 'test_manifest.jsonl')
    shutil.copy2(args.initial_memory, args.output / 'initial_memory.jsonl')
    shutil.copy2(args.retrieval_config, args.output / 'retrieval.json')
    protocol = {'version': C.SCHEMA, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'policy': policy, 'epsilon_memory': epsilon_memory, 'feedback': feedback.metadata,
                'generator': 'qwen3-8b', 'generator_path': resolve_model_config('qwen3-8b').path,
                'gpu': args.gpu, 'gpu_memory_utilization': .34, 'batch_size': 16,
                'max_rounds': 3, 'candidates_per_round': 4, 'model_sha256': C.digest(args.checkpoint / 'model.safetensors'),
                'memory_admission': 'all valid changed candidates with delayed delta_COMET > epsilon_memory; independent of Accept/Reject',
                'feedback_timing': 'after every task in the batch completes', 'arms': args.arms,
                'input_hashes': {n: C.digest(args.output / n) for n in ['test_manifest.jsonl', 'initial_memory.jsonl', 'retrieval.json']}}
    C.dump(args.output / 'protocol.json', protocol)
    return protocol, refs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--initial-memory', type=Path, default=C.COLLECTION / 'initial_memory.jsonl')
    p.add_argument('--retrieval-config', type=Path, default=C.COLLECTION / 'retrieval.json')
    p.add_argument('--checkpoint', type=Path, default=C.OUT / 'best')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/laya_binary_v1.json')
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--gpu', default='0')
    p.add_argument('--arms', nargs='+', choices=['full_static', 'full_online', 'unfiltered_static'], default=['full_static', 'full_online'])
    args = p.parse_args(argv)
    if len(set(args.arms)) != len(args.arms):
        raise ValueError('Duplicate experiment arms')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg = json.loads(args.config.read_text())
    feedback = from_config(cfg['feedback'])
    protocol, refs = prepare(args, feedback)
    generator = Generator(protocol, args.output)
    verifier = Verifier(args.checkpoint)
    flow = Flow(protocol, generator, verifier, feedback, args.output)
    drafts = flow.drafts(refs)
    report = {'version': C.SCHEMA, 'decision_rule': protocol['policy'], 'arms': {}}
    for name in args.arms:
        results = flow.run(name, refs, drafts, online=name == 'full_online')
        pairs = [{'source': r.source, 'current': d['text'], 'candidate': row['final'], 'reference': r.reference}
                 for r, d, row in zip(refs, drafts, results)]
        scores = validated_scores(pairs, feedback.score_pairs(pairs))
        report['arms'][name] = {'sources': len(results), 'mean_delta_comet': sum(s['delta'] for s in scores) / len(scores),
                               'accepted_edits': sum(rd['accepted'] for tr in results for rd in tr['rounds']),
                               'final_task_feedback': scores}
    report['execution_audit'] = {'uncached_generation_calls': generator.calls,
                                 'generation_cache_hits': generator.hits,
                                 'unique_verifier_pairs': verifier.scored,
                                 'test_feedback_used_for_runtime_acceptance': False}
    C.dump(args.output / 'report.json', report)
    print(json.dumps(report['arms'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
