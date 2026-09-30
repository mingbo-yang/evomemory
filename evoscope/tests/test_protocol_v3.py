"""Protocol regressions: state coverage, evidence isolation, precommitted gates."""
from dataclasses import replace
import json
from types import SimpleNamespace as S

import pytest

from evoscope.core import Artifacts, Bank, Policy, Task
from evoscope.engine import EvoScope, sample_checkpoints, spread_probe_schedule
from evoscope.environments import ToyEnvironment, View
from evoscope.gate_plan import prepare_gate_plan
from evoscope.models import ToyModel
from evoscope.runner import Runner


def build(tmp_path, minimum=3, model_type=ToyModel):
    artifacts = Artifacts(tmp_path / "run")
    bank = Bank([Policy("p", "place", "always", "Place the object on the shelf.")])
    tasks = [Task(f"l{i}", f"l{i}", "learn", "Place a clean object on the shelf.",
                  {"object": f"cup{i}", "clean": bool(i % 2)}) for i in range(6)]
    tasks += [Task(f"g{i}", f"g{i}", "gate", "Place a clean object on the shelf.",
                   {"object": f"gate-cup{i}"}) for i in range(8)]
    runner = Runner(ToyEnvironment, model_type(artifacts), artifacts, max_steps=3)
    return EvoScope(bank, runner, tasks, artifacts, min_edit_checkpoints=minimum), runner, artifacts, tasks


def collect(engine, runner, tasks, indices=(0, 1, 2)):
    result = []
    for i in indices:
        ep = runner.run(tasks[i], engine.bank, "learn", i)
        result.append(engine.probe(ep.checkpoints[0], "p", repeats=1))
    return result


def test_single_slot_sampling_balances_across_rounds_and_fills_shortages():
    def ep(t, win):
        return S(error=None, score=int(win), checkpoints=[S(task_id=t, selected_ids=('p',), seen_ids=())])
    counts = {'success': 0, 'failure': 0}
    for _ in range(6):
        _, audit = sample_checkpoints([ep('s', True), ep('f', False)], 'p', 1, counts)
        for k in counts: counts[k] += audit['selected'][k]
    assert counts == {'success': 3, 'failure': 3}
    _, audit = sample_checkpoints([ep('s', True)], 'p', 1, {'success': 8, 'failure': 0})
    assert audit['selected'] == {'success': 1, 'failure': 0}


def test_sampling_counts_attempts_and_resets_only_for_target_version(tmp_path, monkeypatch):
    e, r, _, tasks = build(tmp_path)
    eps = [r.run(t, e.bank, 'learn', i) for i, t in enumerate(tasks[:2])]
    cp, _ = e.sample(eps, e.bank.get('p'), 1)
    def failed(*args, **kwargs):
        from evoscope.runner import RestoreError
        raise RestoreError('fixture failure')
    original = r.run
    monkeypatch.setattr(r, 'run', failed)
    assert e.probe(cp[0], 'p', 1) is None
    assert e.sampling_counts['p', 1] == {'success': 1, 'failure': 0}
    _, audit = e.sample(eps, e.bank.get('p'), 1)
    assert audit['selected']['failure'] == 1
    e.bank = e.bank.revise('p', 'The object is clean.', ())
    monkeypatch.setattr(r, 'run', original)
    eps = [r.run(t, e.bank, 'learn', i) for i, t in enumerate(tasks[:2])]
    _, audit = e.sample(eps, e.bank.get('p'), 1)
    assert audit['cumulative_before'] == {'success': 0, 'failure': 0}


def test_editor_waits_for_distinct_states_retains_neutral_and_never_retries(tmp_path):
    class Recorder(ToyModel):
        edits = 0
        def call(self, system, data, phase, seed):
            if phase == 'edit':
                self.edits += 1
                self.received = data
            return super().call(system, data, phase, seed)
    e, r, _, tasks = build(tmp_path, model_type=Recorder)
    collect(e, r, tasks, (0, 1))
    assert e.propose('p') is None and r.model.edits == 0
    collect(e, r, tasks, (2,))
    proposal = e.propose('p')
    assert proposal and len(proposal.evidence_ids) == 3 and r.model.edits == 1
    assert [x['direction'] for x in r.model.received['evidence']] == ['harmful', 'neutral', 'harmful']
    assert e.propose('p') is None and r.model.edits == 1
    raw = json.dumps(r.model.received)
    assert 'group_id' not in raw and 'task_id' not in raw and 'bank_hash' not in raw


def test_repeated_state_and_repeated_group_do_not_meet_threshold(tmp_path):
    e, r, _, tasks = build(tmp_path)
    collect(e, r, tasks, (0, 0, 0))
    assert len(e.compatible_evidence('p')) == 1 and e.propose('p') is None
    # A different ID and seed describing exactly the same public state is not new evidence.
    tasks[2].payload['object'] = tasks[0].payload['object']
    collect(e, r, tasks, (2,))
    assert len(e.compatible_evidence('p')) == 1
    e.tasks[tasks[4].id] = replace(tasks[4], group_id=tasks[0].group_id)
    collect(e, r, tasks, (4,))
    assert len(e.compatible_evidence('p')) == 1


def test_other_policy_update_and_actor_configuration_invalidate_buffer(tmp_path):
    e, r, _, tasks = build(tmp_path)
    e.bank = Bank([*e.bank.policies, Policy('q', 'astronomy', 'always', 'Compute orbit')])
    original = e.bank
    rows = collect(e, r, tasks)
    assert e.edit_readiness('p')[2] is None
    r.top_k += 1
    assert e.compatible_evidence('p') == []
    r.top_k -= 1
    e.bank = e.bank.revise('q', 'visible planets', ())
    assert e.bank.get('p').version == original.get('p').version
    assert e.compatible_evidence('p') == []
    with pytest.raises(ValueError, match='configuration'):
        e.propose('p', [row['id'] for row in rows])


def test_unstable_only_does_not_trigger_editor_and_errors_are_not_evidence(tmp_path):
    e, r, _, tasks = build(tmp_path)
    rows = collect(e, r, tasks)
    for row in rows: row['public']['direction'] = 'unstable'
    assert e.edit_readiness('p')[2] == 'insufficient_stable_nonzero_evidence'
    assert e.propose('p') is None


def test_plan_uses_public_initial_goal_and_is_independent_of_when(tmp_path):
    a = Artifacts(tmp_path/'a'); b = Artifacts(tmp_path/'b')
    class PublicEnv:
        resets = []
        def reset(self, task):
            self.resets.append(task.id)
            return View('Task: buy size six sandals' if task.id in ('g0','g2') else 'Task: buy a charger', ())
        def close(self): pass
    runner = S(env_factory=PublicEnv)
    bank = Bank([Policy('size', 'size', 'arbitrary', 'click size')])
    tasks = [Task(f'g{i}', f'group{i}', 'gate', 'Generic goal', {'secret': 'size'}) for i in range(6)]
    tasks += [Task('test','test','test','size'), Task('learn','learn','learn','size')]
    plans = prepare_gate_plan(bank, tasks, runner, a, 2, 1, 2, 7)
    changed = prepare_gate_plan(bank.revise('size','a completely different condition',()), tasks, runner, b, 2, 1, 2, 7)
    assert plans == changed and len(plans) == 2
    assert {block.local_ids[0] for block in plans} == {'g0','g2'}
    all_ids = [t for block in plans for t in block.task_ids]
    assert len(all_ids) == len(set(all_ids)) == 4
    assert 'test' not in PublicEnv.resets and 'learn' not in PublicEnv.resets
    raw = (a.root/'gate/plan.json').read_text()
    assert 'secret' not in raw and 'buy size six sandals' in raw


def test_gate_plan_exists_before_editor_and_multistate_evolution_accepts(tmp_path):
    class AuditedModel(ToyModel):
        def call(self, system, data, phase, seed):
            if phase == 'edit':
                assert (self.artifacts.root/'gate/plan.json').exists()
                assert (self.artifacts.root/'gate/block_assignments.jsonl').exists()
                assert len(data['evidence']) >= 3
            return super().call(system, data, phase, seed)
    e, r, a, _ = build(tmp_path, model_type=AuditedModel)
    e.probe_limits = (6, 6)
    initial = e.bank.get('p')
    result = e.evolve(batch_size=2, probe_checkpoints=1, repeats=1, gate_size=2)
    assert result['accepted_edits'] == 1
    assert (e.bank.get('p').key, e.bank.get('p').do) == (initial.key, initial.do)
    receipt = json.loads((a.root/'gate/decisions.jsonl').read_text())
    assert receipt['status'] == 'accepted'
    assert receipt['target_coverage']['local']['old']['exposure_count'] > 0
    assert set(receipt['stratum_mean_deltas']) == {'local','global'}


def test_unexposed_local_gate_cannot_accept_even_positive_reward_noise(tmp_path, monkeypatch):
    e, r, a, tasks = build(tmp_path)
    collect(e, r, tasks); proposal = e.propose('p')
    old = e.bank.fingerprint
    run = r.run
    def unexposed(task, bank, phase, seed):
        ep = run(task, bank, phase, seed)
        return replace(ep, score=int(bank.fingerprint != old), retrieval_counts={}, exposure_counts={})
    monkeypatch.setattr(r, 'run', unexposed)
    receipt = e.gate(proposal, ['g0'])
    assert receipt['mean_delta'] == 1 and not receipt['accepted']
    assert receipt['status'] == 'coverage_insufficient' and e.bank.fingerprint == old
    assert 'g0' in e.used_gate_groups


def test_precommitted_gate_rejects_substituting_tasks(tmp_path):
    e, r, a, tasks = build(tmp_path)
    e.gate_blocks = prepare_gate_plan(e.bank, tasks, r, a, 2, 1, 1, 42)
    collect(e, r, tasks); proposal = e.propose('p');block=e.gate_blocks[0]
    unused=next(t.id for t in tasks if t.split=='gate' and t.id not in block.task_ids)
    with pytest.raises(ValueError, match='precommitted'):
        e.gate(proposal, [block.local_ids[0],unused], block=block)
    assert e.used_gate_groups == set()


def test_new_local_budget_allows_three_states_before_end_of_learning():
    schedule=spread_probe_schedule(25,2,2,(12,6))
    for policy in (0,1):
        rounds=[r for r,n in sorted(schedule.items()) if r%2==policy for _ in range(n)]
        assert len(rounds)==6 and rounds[2]<12


def test_no_relevant_local_groups_is_recorded_before_any_editor_call(tmp_path):
    e, r, a, tasks = build(tmp_path)
    # Ranking sees only visible observations, not task IDs or hidden payloads.
    class UnrelatedInitial:
        def reset(self, task):return View("Your task is: astronomy", ())
        def close(self):pass
    plans = prepare_gate_plan(e.bank, tasks, S(env_factory=UnrelatedInitial), a, 2, 1, 2, 0)
    assert plans == ()
    plan = json.loads((a.root/'gate/plan.json').read_text())
    assert len(plan['shortages']) == 2 and not plan['blocks']
    assert all(c['kind'] == 'reset' for c in a.costs)


def test_cli_exposes_coherent_protocol_defaults(monkeypatch):
    from evoscope import cli
    captured = {}
    monkeypatch.setattr(cli, 'real_run', lambda a: captured.update(vars(a)) or {})
    assert cli.main(['run','--manifest','m','--policies','p','--output','o']) == 0
    assert (captured['min_edit_checkpoints'],captured['min_non_neutral']) == (3,1)
    assert captured['probes_per_policy'] >= 2 * captured['min_edit_checkpoints']
    assert captured['total_probes'] == 24 and captured['gate_size'] == 4
    assert captured['max_candidates_per_policy'] == 2
