"""Paired outcome, logical cost, trajectory and delayed-memory audit for the pilot."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
import laya_acceptance_common as C
from core.experience import load_experiences
from core.candidate_validation import source_key


def interval(values):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261004)
    means = x[rng.integers(len(x), size=(20000, len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def analyze(out):
    report = json.loads((out / 'report.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('Wait for all four arms before outcome analysis')
    p = json.loads((out / 'protocol.json').read_text())
    initial = load_experiences(out / 'initial_memory.jsonl')
    drafts = json.loads((out / 'shared_drafts.json').read_text())
    refs = C.read_jsonl(out / 'test_manifest.jsonl')
    model_cfg = C.validate_checkpoint(Path(p['checkpoint']))
    assert not {source_key(r['source']) for r in refs} & set(model_cfg['seen_source_hashes'])
    assert not {source_key(r['source']) for r in refs} & {source_key(e.source_input) for e in initial}
    assert all(C.digest(Path(__file__).parent / f) == h for f, h in p['code_sha256'].items())
    traces, summary, deltas = {}, {}, {}
    for name, arm in report['arms'].items():
        traces[name] = rows = json.loads((out / name / 'results.json').read_text())
        assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in refs]
        assert [r['initial'] for r in rows] == [r['text'] for r in drafts]
        fb, admitted, temporal_checks = {}, [], 0
        available = {e.exp_id for e in initial}
        for path in sorted((out / name).glob('batch_*.json')):
            batch = json.loads(path.read_text())
            for row in batch:
                for rd in row['rounds']:
                    assert set(rd['retrieved_ids']) <= available
                    temporal_checks += len(rd['retrieved_ids'])
            feedback = json.loads(path.with_name(path.name.replace('batch_', 'feedback_')).read_text())
            fb.update({r['candidate_id']: r for r in feedback['candidates']})
            admitted.extend(feedback['admitted'])
            available.update(feedback['admitted'])
        rounds = [rd for row in rows for rd in row['rounds']]
        candidates = [c for rd in rounds for c in rd['candidates']]
        chosen = [rd['candidates'][rd['selected_slot']] for rd in rounds if rd['accepted']]
        extras = [c for c in chosen if c['slot'] > 0]
        for row in rows:
            current = row['initial']
            stopped = False
            for rd in row['rounds']:
                assert not stopped and rd['before'] == current
                assert len(rd['candidates']) == p['arm_specs'][name]['candidates_per_round']
                for c in rd['candidates']:
                    if not c['changed']:
                        assert c['decision'] == 'Reject' and c['status'] == 'no_op'
                accepting = [i for i, c in enumerate(rd['candidates']) if c['valid'] and c['changed'] and c['decision'] == 'Accept']
                assert rd['selected_slot'] == (accepting[0] if accepting else None)
                if accepting:
                    current = rd['candidates'][accepting[0]]['text']
                else:
                    stopped = True
            assert row['final'] == current
        ds = np.array([r['delta'] for r in arm['final_task_feedback']])
        deltas[name] = ds
        input_tokens = sum(c.get('input_tokens', 0) for c in candidates)
        output_tokens = sum(c.get('output_tokens', 0) for c in candidates)
        draft_tokens = sum(d.get('input_tokens', 0) + d.get('output_tokens', 0) for d in drafts)
        summary[name] = {
            'sources': len(rows), 'mean_final_comet': float(np.mean([r['q_candidate'] for r in arm['final_task_feedback']])),
            'mean_delta_from_draft': float(ds.mean()), 'delta_ci95': interval(ds),
            'sources_improved': int((ds > 0).sum()), 'sources_worsened': int((ds < 0).sum()), 'sources_tied': int((ds == 0).sum()),
            'changed_final_sources': sum(row['final'] != row['initial'] for row in rows),
            'invalid_drafts': sum(not row['draft_valid'] for row in rows), 'rounds': len(rounds),
            'candidates': len(candidates), 'valid_changed': sum(c['valid'] and c['changed'] for c in candidates),
            'noops': sum(not c['changed'] for c in candidates),
            'invalid_reasons': dict(Counter(c['validity_reason'] for c in candidates if not c['valid'])),
            'verifier_statuses': dict(Counter(c.get('status', 'not_classified') for c in candidates)),
            'unique_texts_within_round_total': sum(len({c['text'] for c in rd['candidates']}) for rd in rounds),
            'all_identical_rounds': sum(len({c['text'] for c in rd['candidates']}) == 1 for rd in rounds),
            'selected_edits': len(chosen), 'selected_positive': sum(fb[c['id']]['delta'] > 0 for c in chosen),
            'selected_negative': sum(fb[c['id']]['delta'] < 0 for c in chosen), 'selected_tie': sum(fb[c['id']]['delta'] == 0 for c in chosen),
            'selected_delta_sum': sum(fb[c['id']]['delta'] for c in chosen),
            'selected_extra_slots': len(extras), 'extra_slot_positive': sum(fb[c['id']]['delta'] > 0 for c in extras),
            'extra_slot_negative': sum(fb[c['id']]['delta'] < 0 for c in extras),
            'extra_slot_delta_sum': sum(fb[c['id']]['delta'] for c in extras),
            'positive_rejected_candidate_occurrences': sum(r['delta'] > 0 and r['decision'] == 'Reject' for r in fb.values()),
            'admitted_memories': len(admitted), 'rejected_positive_memories_admitted': sum(r['memory_candidate'] and r['decision'] == 'Reject' for r in fb.values()) if name.endswith('online') else 0,
            'online_retrieval_occurrences': sum(rd['online_retrieved'] for rd in rounds),
            'online_retrieval_sources': sum(any(rd['online_retrieved'] for rd in row['rounds']) for row in rows),
            'refinement_input_tokens': input_tokens, 'refinement_output_tokens': output_tokens,
            'logical_total_tokens_including_drafts': input_tokens + output_tokens + draft_tokens,
            'retrieval_temporal_checks_passed': temporal_checks,
        }
    verifier_seen, verifier_shared, verifier_disagreements = {}, 0, []
    for name, rows in traces.items():
        for row in rows:
            for rd in row['rounds']:
                for c in rd['candidates']:
                    if c.get('status') != 'classified':
                        continue
                    key = (row['source'], rd['before'], c['text'])
                    if key in verifier_seen:
                        verifier_shared += 1
                        if verifier_seen[key] != c['decision']:
                            verifier_disagreements.append({'arm': name, 'candidate_id': c['id'], 'previous': verifier_seen[key], 'decision': c['decision']})
                    verifier_seen[key] = c['decision']
    C.dump(out / 'shared_verifier_audit.json', {'shared_occurrences': verifier_shared, 'disagreements': verifier_disagreements})
    if verifier_disagreements:
        raise ValueError('Identical inputs received different decisions across GPU/batches; resolve before causal interpretation')
    comparisons = {}
    for memory in ('static', 'online'):
        a, b = 'k1_' + memory, 'k4_' + memory
        d = deltas[b] - deltas[a]
        shared, equal = 0, 0
        for ra, rb in zip(traces[a], traces[b]):
            for da, db in zip(ra['rounds'], rb['rounds']):
                if da['round'] == db['round'] and da['before'] == db['before'] and da['retrieved_ids'] == db['retrieved_ids']:
                    shared += 1
                    equal += da['candidates'][0]['text'] == db['candidates'][0]['text']
        assert shared == equal
        comparisons[memory] = {'mean_k4_minus_k1': float(d.mean()), 'paired_source_bootstrap_ci95': interval(d),
                 'sources_k4_better': int((d > 0).sum()), 'sources_k4_worse': int((d < 0).sum()), 'sources_equal': int((d == 0).sum()),
                 'refinement_token_ratio': (summary[b]['refinement_input_tokens'] + summary[b]['refinement_output_tokens']) / (summary[a]['refinement_input_tokens'] + summary[a]['refinement_output_tokens']),
                 'all_token_ratio': summary[b]['logical_total_tokens_including_drafts'] / summary[a]['logical_total_tokens_including_drafts'],
                 'shared_state_slot0_checks': shared, 'shared_state_slot0_equal': equal,
                 'ci_caveat': 'single stream, source-bootstrap descriptive only; online trajectories share evolving memory'}
    result = {'protocol': p, 'initial_mean_comet': float(np.mean([s['q_current'] for s in report['arms']['k1_static']['final_task_feedback']])),
              'historical_direct_zero_mean_comet': float(np.mean([s['q_candidate'] for s in report['historical_direct_zero']['scores']])),
              'summary': summary, 'comparisons': comparisons, 'execution': report['execution'],
              'shared_verifier_occurrences_checked': verifier_shared, 'shared_verifier_disagreements': len(verifier_disagreements),
              'limitations': ['128 sources, one generator, epoch20 only, one ordered stream', 'candidate counts change compute budget; not equal-token comparison', 'shared request caches invalidate raw elapsed-time speed comparisons', 'test results must not be used for further checkpoint/threshold selection']}
    C.dump(out / 'analysis.json', result)
    with (out / 'summary.csv').open('w') as f:
        fields = ['arm'] + [k for k, v in next(iter(summary.values())).items() if isinstance(v, (int, float))]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for name, s in summary.items():
            writer.writerow(dict(s, arm=name))
    with (out / 'paired_sources.csv').open('w') as f:
        fields = ['sample_id'] + list(deltas) + ['static_k4_minus_k1', 'online_k4_minus_k1']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, ref in enumerate(refs):
            writer.writerow(dict(sample_id=ref['sample_id'], **{n: float(d[i]) for n, d in deltas.items()},
                                 static_k4_minus_k1=float(deltas['k4_static'][i] - deltas['k1_static'][i]),
                                 online_k4_minus_k1=float(deltas['k4_online'][i] - deltas['k1_online'][i])))
    print(json.dumps({k: v for k, v in result.items() if k not in ('protocol',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    analyze(p.parse_args().output)
