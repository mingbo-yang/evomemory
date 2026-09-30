#!/usr/bin/env python
"""Read-only audit of optimized online runs and delayed-feedback provenance."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import core
from core.bm25_fields import Experience
from core.manifest import read_manifest, manifest_path
from core.quality_feedback import QualityPolicy
from core.scoring import Scorer, PRIMARY_METRIC


def audit(trace_path, initial_root, split):
    records = [json.loads(line) for line in trace_path.open() if line.strip()]
    assert records, 'empty trace'
    first = records[0]
    task, model = first['task'], first['model']
    refs = {r.sample_id: r for r in read_manifest(manifest_path(task, split))}
    ids = [r['sample_id'] for r in records]
    assert len(ids) == len(set(ids)), 'duplicate sample ids'
    ordered = [r.sample_id for r in read_manifest(manifest_path(task, split))][:len(ids)]
    assert ids == ordered, 'not the requested manifest prefix'
    initial = [Experience.from_dict(json.loads(l)) for l in (initial_root/model/task/'initial.jsonl').open()]
    library = [Experience.from_dict(r) for r in json.loads((trace_path.parent/'online_experience.json').read_text())]
    initial_by_id = {e.exp_id: e for e in initial}
    by_id = {e.exp_id: e for e in library}
    assert len(by_id) == len(library), 'duplicate memory IDs'
    for e in initial:
        assert by_id[e.exp_id] == e, 'initial experience mutated'
    online = {e.exp_id: e for e in library if e.exp_id not in initial_by_id}
    batch = first['config']['optimization']['batch_size']
    policy = QualityPolicy(**first['config']['optimization']['policy'])
    # Derive production times by trace IDs, not by trusting memory metadata.
    birth = {f"{task}/{model}/{r['sample_id']}/r{rd['round_index']}": i//batch
             for i, r in enumerate(records) for rd in r['rounds']}
    counts = Counter()
    expected_memory = set()
    online_hit_samples = set()
    scorer = Scorer(task)
    for i, r in enumerate(records):
        assert r['config_hash'] == first['config_hash'], 'mixed configuration'
        assert r['source_hash'] == refs[r['sample_id']].source_hash, 'wrong input'
        current = r['initial_draft']
        assert current.strip(), 'empty draft'
        before_stop = False
        for rd in r['rounds']:
            assert not before_stop, 'round after STOP'
            counts[rd['controller_action']] += 1
            for exp_id in rd['exp_ids']:
                e = by_id[exp_id]
                assert e.source_input != refs[r['sample_id']].source, 'own-source retrieval'
                if exp_id in online:
                    assert birth[exp_id] < i//batch, 'same-batch or future feedback used'
                    counts['online_retrieval_hits'] += 1
                    online_hit_samples.add(r['sample_id'])
            counts['retrieval_hits'] += len(rd['exp_ids'])
            reason = json.loads(rd['controller_reason'])
            assert reason['evaluator'] == 'bert'
            if rd['controller_action'] == 'STOP':
                assert policy.should_stop(reason['score']), 'invalid BERT STOP'
                assert not rd['accepted'] and not rd['candidate']
                before_stop = True
                continue
            before, after = reason['before'], reason['after']
            accepted, gate = policy.accept(current, rd['candidate'], before, after)
            assert accepted == rd['accepted'] and gate == reason['gate'], 'gate mismatch'
            counts[gate] += 1
            old_metric = scorer.primary(refs[r['sample_id']].reference, current)
            new_metric = scorer.primary(refs[r['sample_id']].reference, rd['candidate'])
            delta = new_metric-old_metric
            assert math.isclose(rd['delta_offline'], delta, abs_tol=1e-8), 'wrong feedback baseline'
            eid = f"{task}/{model}/{r['sample_id']}/r{rd['round_index']}"
            admitted = accepted and delta > 0
            if admitted:
                expected_memory.add(eid)
                e = online[eid]
                assert e.state_before == current and e.state_after == rd['candidate'], 'wrong experience pair'
                assert e.source_input == refs[r['sample_id']].source
                assert e.outcome_label == 'helped' and e.provenance == 'online'
                assert math.isclose(e.delta_offline, delta, abs_tol=1e-8)
            else:
                assert eid not in online, 'ineligible experience admitted'
            if accepted:
                counts['accepted'] += 1
                counts['accepted_improved' if delta > 0 else 'accepted_degraded' if delta < 0 else 'accepted_tied'] += 1
                current = rd['candidate']
            else:
                counts['rejected'] += 1
        assert current == r['final_output'], 'wrong final state'
        if first['config']['optimization'].get('cost_accounting') == 'initial_and_refinement_v2':
            initial_calls = 0 if r['draft_from_cache'] else 1
            refine_calls = sum(d['controller_action'] == 'REFINE' for d in r['rounds'])
            assert r['cost']['n_calls'] == initial_calls + refine_calls, 'missing generation calls'
            assert r['cost'].get('initial_n_calls', 0) == initial_calls, 'wrong initial generation cost'
            assert math.isclose(r['cost']['total_tokens'], r['cost']['input_tokens']+r['cost']['output_tokens']), 'token sum mismatch'
            if initial_calls:
                assert r['cost']['initial_total_tokens'] > 0, 'missing initial tokens'
        assert math.isclose(r['final_metric_offline'], scorer.primary(refs[r['sample_id']].reference, current), abs_tol=1e-8)
    assert set(online) == expected_memory, 'unexpected memory entries'
    counts['online_retrieval_samples'] = len(online_hit_samples)
    pairs = lambda field: [(refs[r['sample_id']].reference, r[field]) for r in records]
    metric = PRIMARY_METRIC[task]
    return dict(task=task, model=model, n=len(records), batch_size=batch,
                config_hash=first['config_hash'], initial_memory=len(initial), new_memory=len(online),
                counts=dict(counts), metric=metric,
                initial_score=scorer.score_corpus(pairs('initial_draft'))[metric],
                final_score=scorer.score_corpus(pairs('final_output'))[metric],
                validation='PASS',
                cost_check=('PASS' if first['config']['optimization'].get('cost_accounting') == 'initial_and_refinement_v2' else 'INITIAL_GENERATION_NOT_COUNTED_IN_THIS_LEGACY_PILOT'),
                trace=str(trace_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--initial-root', type=Path, default=Path('experience/contrastive'))
    ap.add_argument('--split', default='dev')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--expected-samples', type=int, default=32)
    ap.add_argument('--expected-tasks', type=int, default=4)
    args = ap.parse_args()
    files = sorted(args.runs.rglob('full_online.jsonl'))
    assert files, 'no completed runs'
    reports = [audit(p,args.initial_root,args.split) for p in files]
    assert len(reports) == args.expected_tasks, 'missing or duplicate task runs'
    assert len({r['task'] for r in reports}) == args.expected_tasks, 'duplicate task coverage'
    assert all(r['n'] == args.expected_samples for r in reports), 'incomplete sample coverage'
    result = {'scope':'small dev workflow validation, not statistical efficacy evidence', 'runs':reports}
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
