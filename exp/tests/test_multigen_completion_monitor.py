import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
import pytest
import monitor_multigen_completion as M


def finished(root):
    config = {'protocol_hashes': {}, 'services': []}
    M.write(root / 'campaign_complete.json', {'tasks': {t: {} for t in M.TASKS}})
    for task in M.TASKS:
        d = root / task
        M.write(d / 'protocol.json', {'task': task})
        config['protocol_hashes'][task] = M.sha(d / 'protocol.json')
        M.write(d / 'collection_complete.json', {'task': task, 'test': 'sealed',
                    'target_met': task != 'coedit_gec', 'pool_exhausted': task == 'coedit_gec'})
        for split in ('train', 'dev'):
            p = d / 'exports' / (split + '.jsonl')
            M.write(p, {'example': split})
            M.write(p.with_suffix('.manifest.json'), {'sha256': M.sha(p), 'path': str(p)})
        M.write(d / 'test_integrity_audit.json', {'state': 'sealed',
                   'forbidden_operations_executed': False, 'checks': {'source_generator_records': 2500}})
    return config


def no_gpu(monkeypatch, running=None):
    monkeypatch.setattr(M, 'gpu_snapshot', lambda: {'fake': True})
    monkeypatch.setattr(M, 'coordinator_running', lambda _: running or [])
    monkeypatch.setattr(M, 'service_health', lambda s: {'name': s['name'], 'status': 200})


def test_idle_or_incomplete_campaign_does_not_release(tmp_path, monkeypatch):
    no_gpu(monkeypatch)
    monkeypatch.setattr(M, 'release', lambda *_: pytest.fail('Premature cleanup'))
    r = M.check(tmp_path, {'services': [], 'protocol_hashes': {}}, execute=True)
    assert r['state'] == 'incomplete_coordinator_stopped'
    assert not r['ready']


def test_all_tasks_required_even_if_campaign_marker_exists(tmp_path):
    config = finished(tmp_path)
    (tmp_path / M.TASKS[-1] / 'collection_complete.json').unlink()
    assert not M.completion_gate(tmp_path, config)[0]


def test_complete_or_pool_exhausted_is_allowed_and_no_sealed_test_contents_are_read(tmp_path, monkeypatch):
    config = finished(tmp_path)
    original = M.read
    def audited(path):
        assert 'raw' not in Path(path).parts
        assert Path(path).name not in ('test.jsonl', 'evaluation_lock.json')
        return original(path)
    monkeypatch.setattr(M, 'read', audited)
    assert M.completion_gate(tmp_path, config, verify_hashes=True) == (True, [])


@pytest.mark.parametrize('damage', ['export', 'audit', 'protocol', 'target'])
def test_incomplete_or_corrupt_artifacts_block_release(tmp_path, damage):
    config = finished(tmp_path)
    d = tmp_path / M.TASKS[0]
    if damage == 'export':
        (d / 'exports/train.jsonl').write_text('corrupted\n')
    elif damage == 'audit':
        (d / 'test_integrity_audit.json').unlink()
    elif damage == 'protocol':
        (d / 'protocol.json').write_text('{}')
    else:
        x = M.read(d / 'collection_complete.json')
        x.update(target_met=False, pool_exhausted=False)
        M.write(d / 'collection_complete.json', x)
    assert not M.completion_gate(tmp_path, config, verify_hashes=True)[0]


def test_active_coordinator_blocks_release(tmp_path, monkeypatch):
    config = finished(tmp_path)
    no_gpu(monkeypatch, [123])
    monkeypatch.setattr(M, 'release', lambda *_: pytest.fail('Active coordinator'))
    assert M.check(tmp_path, config, execute=True)['state'] == 'waiting'


def test_locked_campaign_blocks_release(tmp_path, monkeypatch):
    config = finished(tmp_path)
    no_gpu(monkeypatch)
    monkeypatch.setattr(M, 'release', lambda *_: pytest.fail('Campaign lock bypassed'))
    with (tmp_path / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert M.check(tmp_path, config, execute=True)['state'] == 'waiting_for_campaign_lock'


def test_complete_idle_campaign_releases_once(tmp_path, monkeypatch):
    config = finished(tmp_path)
    no_gpu(monkeypatch)
    calls = []
    def release(c, path):
        calls.append(path)
        return {'state': 'released'}
    monkeypatch.setattr(M, 'release', release)
    assert M.check(tmp_path, config, execute=True)['state'] == 'released'
    assert calls == [tmp_path]


def identity(pid, parent=1, start=100):
    return dict(pid=pid, ppid=parent, start_ticks=start, session=10, uid=os.getuid(), argv=['dummy'])


def test_descendants_exclude_unrelated_and_reused_parent():
    root = identity(10)
    table = {10: root, 11: identity(11, 10), 12: identity(12, 11), 50: identity(50)}
    assert {p['pid'] for p in M.descendants(root, table)} == {10, 11, 12}
    assert M.descendants(dict(root, start_ticks=99), table) == []


def test_signal_requires_identity_and_does_not_touch_other_processes():
    children = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']) for _ in range(2)]
    try:
        p = M.process(children[0].pid)
        assert p
        assert not M.safe_signal(dict(p, start_ticks=p['start_ticks'] + 1), signal.SIGTERM)
        assert children[0].poll() is None
        assert M.safe_signal(p, signal.SIGTERM)
        children[0].wait(timeout=5)
        assert children[1].poll() is None
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
