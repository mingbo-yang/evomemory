from dataclasses import asdict, replace
import json

import pytest

from evoscope.core import Artifacts, Bank, Policy, Task, atomic_json
from evoscope.environments import View
from evoscope.runner import Runner, RestoreError
from evoscope.scope_trial import compare, compare_repeats, freeze_cases, restore_checkpoint


class FourSteps:
    def reset(self, task):
        self.step_count = 0
        self.late = task.payload.get('late', False)
        self.drop = task.payload.get('drop', False)
        return self.view()

    def view(self):
        if self.step_count == 0:
            text = 'Task: finish\n' + ('plain' if self.late else 'marker')
        else:
            text = ('plain' if self.drop else 'marker') + f' stage {self.step_count}'
        return View(text, ('wait',), 0, self.step_count == 4)

    def step(self, action):
        assert action == 'wait'
        self.step_count += 1
        return self.view()

    def close(self):
        pass


class Recorder:
    fingerprint = 'scope-fixture'
    def __init__(self):
        self.inputs = []

    def call(self, system, data, phase, seed):
        self.inputs.append((system, data, seed))
        return {'action': 'wait'}


def setup(tmp_path, *, late=False, drop=False):
    artifacts = Artifacts(tmp_path / 'run')
    model = Recorder()
    runner = Runner(FourSteps, model, artifacts, 4)
    bank = Bank([Policy(p, 'marker', 'visible condition', 'wait') for p in ('a', 'b', 'c')])
    task = Task('learn', 'learn', 'learn', 'fallback', {'late': late, 'drop': drop})
    ep = runner.run(task, bank, 'learn', 42)
    cp = next(c for c in ep.checkpoints if 'b' in c.selected_ids and 'b' not in c.seen_ids)
    model.inputs.clear()
    return runner, model, bank, task, cp


@pytest.mark.parametrize('branch', ['expose', 'mask'])
def test_default_persistent_matches_explicit_and_changes_all_steps(tmp_path, branch):
    runner, model, bank, task, cp = setup(tmp_path)
    runner.run(task, bank, 'test-default', 9, cp, 'b', branch)
    old = list(model.inputs)
    model.inputs.clear()
    runner.run(task, bank, 'test-explicit', 9, cp, 'b', branch, intervention_scope='persistent')
    assert model.inputs == old
    for _, data, _ in old:
        target = [p for p in data['policies'] if p['id'] == 'b']
        assert (target == [{'id': 'b', 'do': 'wait'}]) if branch == 'expose' else not target


@pytest.mark.parametrize('branch', ['expose', 'mask'])
def test_single_step_changes_only_first_resumed_input_and_preserves_rank(tmp_path, branch):
    runner, model, bank, task, cp = setup(tmp_path, late=True)
    assert len(cp.actions) == 1
    original = bank.fingerprint
    runner.run(task, bank, 'trial', 9, cp, 'b', branch, intervention_scope='single-step')
    first = model.inputs[0][1]['policies']
    rank = cp.selected_ids.index('b')
    if branch == 'expose':
        assert first[rank] == {'id': 'b', 'do': 'wait'}
    else:
        assert 'b' not in [p['id'] for p in first]
    assert [p['id'] for p in first if p['id'] != 'b'] == [p for p in cp.selected_ids if p != 'b']
    for _, data, _ in model.inputs[1:]:
        assert next(p for p in data['policies'] if p['id'] == 'b')['when'] == 'visible condition'
    assert bank.fingerprint == original


def test_single_step_does_not_force_target_after_retrieval_dropout(tmp_path):
    runner, model, bank, task, cp = setup(tmp_path, drop=True)
    runner.run(task, bank, 'trial', 9, cp, 'b', 'expose', intervention_scope='single-step')
    assert model.inputs[0][1]['policies']
    assert all(data['policies'] == [] for _, data, _ in model.inputs[1:])
    model.inputs.clear()
    runner.run(task, bank, 'old', 9, cp, 'b', 'expose')
    assert all(data['policies'] == [{'id': 'b', 'do': 'wait'}] for _, data, _ in model.inputs[1:])


def test_normal_restoration_matches_original_inputs_and_rejects_seen_target(tmp_path):
    runner, model, bank, task, cp = setup(tmp_path)
    runner.run(task, bank, 'baseline', 9)
    baseline = list(model.inputs)
    model.inputs.clear()
    runner.run(task, bank, 'diagnostic', 9, cp, 'b', 'normal')
    assert model.inputs == baseline
    with pytest.raises(RestoreError, match='contaminated'):
        runner.run(task, bank, 'diagnostic', 9, replace(cp, seen_ids=('b',)), 'b', 'normal')
    with pytest.raises(ValueError, match='unknown intervention scope'):
        runner.run(task, bank, 'diagnostic', 9, cp, 'b', 'expose', intervention_scope='invalid')


def test_neutral_behavior_is_not_relabelled_as_helpful_or_steps_reward():
    a = {'score': 1, 'error': None, 'actions': ['a', 'finish']}
    b = {'score': 1, 'error': None, 'actions': ['b', 'wait', 'finish']}
    result = compare(a, b)
    assert result['delta'] == 0 and result['step_difference'] == -1
    assert result['neutral_behavior'] == 'different_actions'
    assert result['first_divergence_after_checkpoint'] == 0
    summary = compare_repeats([{'branches': {'e': a, 'm': b}}] * 3, 'e', 'm')
    assert summary['direction'] == 'neutral'
    bad = {**a, 'error': 'RestoreError'}
    assert not compare(bad, b)['valid']
    assert compare_repeats([{'branches': {'e': bad, 'm': b}}], 'e', 'm')['direction'] == 'execution_error'


def test_serialization_and_case_selection_do_not_read_effect_sign(tmp_path):
    runner, model, bank, task, cp = setup(tmp_path)
    assert restore_checkpoint(json.loads(json.dumps(asdict(cp)))).id == cp.id
    source = tmp_path / 'source'
    folder = source / 'alfworld'
    (folder / 'learn').mkdir(parents=True)
    (folder / 'probe').mkdir()
    atomic_json(source / 'alfworld-manifest.json', [asdict(task)])
    atomic_json(source / 'alfworld-m0.json', bank.to_json())
    (folder / 'learn/episodes.jsonl').write_text(json.dumps({'checkpoints': [asdict(cp)]}) + '\n')
    evidence = {'id': 'e0', 'checkpoint_id': cp.id, 'policy_id': 'b', 'bank_hash': bank.fingerprint,
                'policy_version': bank.get('b').version, 'model_fingerprint': model.fingerprint, 'seed': 42}
    path = folder / 'probe/evidence.jsonl'
    path.write_text(json.dumps({**evidence, 'status': 'harmful'}) + '\n')
    first = freeze_cases(source, 'alfworld')
    path.write_text(json.dumps({**evidence, 'status': 'neutral'}) + '\n')
    assert freeze_cases(source, 'alfworld') == first
    assert first[0]['seed'] == 42
