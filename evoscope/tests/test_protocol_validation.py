import json

from evoscope.core import Task
from evoscope.protocol_validation import sampling_audit, select_tasks, worker_complete


def test_validation_selection_preserves_order_and_excludes_test():
    tasks = [Task(f'learn-{i}', f'learn-group-{i}', 'learn', 'goal', {}, 42) for i in range(14)]
    tasks += [Task('test', 'test-group', 'test', 'goal', {}, 42),
              Task('gate', 'gate-group', 'gate', 'goal', {}, 42)]
    result = select_tasks(tasks)
    assert [t.id for t in result] == [f'learn-{i}' for i in range(12)] + ['gate']


def test_sampling_audit_detects_old_success_first_bug():
    row = {'available': {'success': 1, 'failure': 1},
           'cumulative_before': {'success': 1, 'failure': 0},
           'selected': {'success': 1, 'failure': 0}}
    assert sampling_audit([row])['violations'] == [row]
    row['selected'] = {'success': 0, 'failure': 1}
    audit = sampling_audit([row])
    assert not audit['violations']
    assert audit['mixed_stratum_decisions'] == 1
    assert audit['rounds_with_prior_counts'] == 1


def test_completion_receipt_does_not_depend_on_native_process_exit(tmp_path):
    assert not worker_complete(tmp_path, 'alfworld')
    path = tmp_path / 'alfworld-summary.json'
    path.write_text(json.dumps({'complete': False}))
    assert not worker_complete(tmp_path, 'alfworld')
    path.write_text(json.dumps({'complete': True}))
    assert worker_complete(tmp_path, 'alfworld')
