"""Archived 1/4-candidate rollout for explicit candidate-count ablations only.

The formal Flow has one candidate and cannot enter this selection path.
This preserves the completed pilot's multi-candidate dynamics, including its
explicit unfiltered diagnostic variant, without exposing them in production.
"""
import json
from pathlib import Path
from core.experience import load_experiences, save_experiences, render_experience_block
from core.candidate_validation import valid_candidate
from core.refinement_instructions import INSTRUCTIONS
from core.comet_feedback import validated_scores
import laya_acceptance_common as C
from laya_relaxed_flow import TASK, stable_seed, memory_key, feedback_pairs, positive_memories

def choose_policy(checkpoint=C.OUT / 'best'):
    checkpoint = Path(checkpoint)
    C.validate_checkpoint(checkpoint)
    return {'version': C.SCHEMA, 'checkpoint': str(checkpoint.resolve()), 'labels': C.LABELS,
            'selection': 'first valid changed candidate classified Accept in generation order',
            'no_op_rule': C.NO_OP_RULE,
            'decision_rule': 'raw argmax; exact tie Reject', 'probabilities': 'diagnostic only'}


def select_candidate(candidates, policy=None, mode='laya'):
    if mode not in {'laya', 'unfiltered'}:
        raise ValueError('Unknown acceptance mode')
    if policy and any(k in policy for k in ('p_better', 'p_worse_max', 'threshold', 'margin')):
        raise ValueError('Runtime acceptance thresholds are forbidden by the binary protocol')
    for i, candidate in enumerate(candidates):
        if not candidate.get('valid') or not candidate.get('changed'):
            continue
        if mode == 'unfiltered':
            return i
        decision = candidate.get('decision')
        if decision is not None and decision not in C.LABELS:
            raise ValueError('Expected an explicit Accept/Reject decision')
        if decision == 'Accept':
            return i
    return None


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

