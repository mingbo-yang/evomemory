"""Poll campaign completion; release only explicitly registered vLLM processes.

Independent of frozen collection code. Never reads sealed Test examples/scores.
Uses registered process birth identities; prefers pidfd where the kernel supports it.
On this older host, rechecks identity immediately before a POSIX signal.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

TASKS = ('wmt19_en_zh', 'wmt19_zh_en', 'coedit_gec', 'gigaword')
DEFAULT = Path(__file__).resolve().parent / 'runs/laya_multigen_data_v1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    try:
        if immutable:
            os.link(temp, path)
        else:
            os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def process(pid):
    p = Path('/proc') / str(pid)
    try:
        stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        argv = (p / 'cmdline').read_bytes().split(b'\0')
        argv = [s.decode(errors='replace') for s in argv if s]
        return dict(pid=int(pid), ppid=int(stat[1]), session=int(stat[3]),
                    start_ticks=int(stat[19]), uid=p.stat().st_uid, argv=argv)
    except (OSError, ValueError, IndexError):
        return None


def all_processes():
    return {p['pid']: p for entry in Path('/proc').iterdir()
            if entry.name.isdigit() and (p := process(int(entry.name))) is not None}


def same_process(expected, actual):
    return bool(actual) and all(expected[k] == actual[k]
        for k in ('pid', 'session', 'start_ticks', 'uid', 'argv'))


def descendants(root, table):
    if not same_process(root, table.get(root['pid'])):
        return []
    found = {root['pid']}
    while True:
        added = {p['pid'] for p in table.values()
                 if p['ppid'] in found and p['uid'] == root['uid']}
        if added <= found:
            return [table[pid] for pid in sorted(found)]
        found.update(added)


def option(argv, key):
    return argv[argv.index(key) + 1] if key in argv and argv.index(key) + 1 < len(argv) else None


def arm(output, interval):
    if interval < 10:
        raise ValueError('Polling interval must be at least 10 seconds')
    output = Path(output)
    models = read(output / TASKS[0] / 'protocol.json')['models']
    table, services = all_processes(), []
    for model in models:
        matches = [p for p in table.values()
            if 'vllm.entrypoints.openai.api_server' in p['argv']
            and option(p['argv'], '--served-model-name') == model['name']
            and option(p['argv'], '--model') == model['path']
            and option(p['argv'], '--port') == str(model['port'])]
        if len(matches) != 1 or matches[0]['uid'] != os.getuid():
            raise ValueError('Could not uniquely identify owned service: ' + model['name'])
        root = matches[0]
        if root['session'] != root['pid']:
            raise ValueError('Service does not have an isolated session')
        services.append(dict(name=model['name'], port=model['port'], gpu=model['gpu'],
                             root=root, members=descendants(root, table)))
    if len(services) != 5:
        raise ValueError('Expected exactly five generators')
    config = dict(version=1, created_at=utc(), interval_seconds=interval,
        output=str(output.resolve()), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        monitor_sha256=sha(__file__), services=services,
        protocol_hashes={t: sha(output / t / 'protocol.json') for t in TASKS},
        completion_policy='all_four_tasks_exported_and_test_sealed_then_coordinator_idle',
        authorization='User requested periodic checking and GPU release after all generation tasks complete')
    write(output / 'monitor/config.json', config, immutable=True)
    return config


def completion_gate(output, config, verify_hashes=False):
    """Fail closed; no Test quality, raw contents, or model predictions are read."""
    output = Path(output)
    pending = []
    if not (output / 'campaign_complete.json').exists():
        pending.append('campaign_complete.json missing')
    for task in TASKS:
        d = output / task
        if not (d / 'collection_complete.json').exists():
            pending.append(task + ': collection incomplete')
    if pending:
        return False, pending
    campaign = read(output / 'campaign_complete.json')
    if not set(TASKS) <= set(campaign.get('tasks', {})):
        return False, ['campaign task list incomplete']
    for task in TASKS:
        d = output / task
        if sha(d / 'protocol.json') != config['protocol_hashes'][task]:
            pending.append(task + ': protocol changed; review required')
        done = read(d / 'collection_complete.json')
        if done.get('task') != task or done.get('test') != 'sealed':
            pending.append(task + ': invalid completion marker')
        if not (done.get('target_met') is True or done.get('pool_exhausted') is True):
            pending.append(task + ': target not met and source pool not exhausted')
        for split in ('train', 'dev'):
            exported = d / 'exports' / (split + '.jsonl')
            manifest = d / 'exports' / (split + '.manifest.json')
            if not exported.exists() or not manifest.exists():
                pending.append(task + ': missing ' + split + ' export')
            elif verify_hashes and sha(exported) != read(manifest).get('sha256'):
                pending.append(task + ': corrupt ' + split + ' export')
        audit = d / 'test_integrity_audit.json'
        if not audit.exists():
            pending.append(task + ': Test integrity audit missing')
        else:
            a = read(audit)
            if (a.get('state') != 'sealed' or a.get('forbidden_operations_executed') is not False
                or a.get('checks', {}).get('source_generator_records') != 2500):
                pending.append(task + ': Test integrity audit not complete')
    return not pending, pending


def coordinator_running(output):
    table = all_processes()
    info = Path(output) / 'process.json'
    registration = read(info) if info.exists() else {}
    p = table.get(registration.get('pid'))
    root = str(Path(__file__).resolve().parent)
    registered = (p and p['uid'] == os.getuid() and p['argv'] == registration.get('command')
                  and (registration.get('start_ticks') is None or p['start_ticks'] == registration['start_ticks']))
    active = [p] if registered else []
    for p in table.values():
        if any('comet_feedback_worker.py' in a and root in a for a in p['argv']):
            active.append(p)
    return [p['pid'] for p in active]


def gpu_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        return dict(returncode=result.returncode, output=result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(error=str(e))


def service_health(service):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{service['port']}/health", timeout=5) as response:
            return dict(name=service['name'], status=response.status)
    except Exception as e:
        return dict(name=service['name'], error=str(e))


def safe_signal(identity, sig):
    # New kernels can pin a process; CentOS 4.18 lacks pidfd_open. On the legacy
    # path use exact UID, command, session, and birth-time checks immediately
    # before signalling. Never signal process groups or unregistered identities.
    fd = None
    if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
        try:
            fd = os.pidfd_open(identity['pid'])
        except ProcessLookupError:
            return False
        except OSError as e:
            if e.errno != errno.ENOSYS:
                raise
    try:
        if not same_process(identity, process(identity['pid'])):
            return False
        if fd is not None:
            signal.pidfd_send_signal(fd, sig)
        else:
            os.kill(identity['pid'], sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        if fd is not None:
            os.close(fd)


def release(config, output):
    if config['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('Host rebooted; registered process identities expired')
    table = all_processes()
    identities = {(p['pid'], p['start_ticks']): p for s in config['services'] for p in s['members']}
    for service in config['services']:
        for p in descendants(service['root'], table):
            identities[(p['pid'], p['start_ticks'])] = p
    identities = list(identities.values())
    before = gpu_snapshot()
    actions = []
    def send(items, sig):
        for p in items:
            if safe_signal(p, sig):
                event = dict(at=utc(), pid=p['pid'], start_ticks=p['start_ticks'], signal=sig.name)
                actions.append(event)
                print(json.dumps(event), flush=True)
    def remaining():
        return [p for p in identities if same_process(p, process(p['pid']))]
    def wait(seconds):
        end = time.monotonic() + seconds
        while remaining() and time.monotonic() < end:
            time.sleep(1)
    write(Path(output) / 'monitor/cleanup_started.json', dict(at=utc(), targets=identities, gpu_before=before))
    send([s['root'] for s in config['services']], signal.SIGTERM)
    wait(30)
    send(remaining(), signal.SIGTERM)
    wait(20)
    send(remaining(), signal.SIGKILL)
    wait(5)
    residual = remaining()
    report = dict(at=utc(), state='released' if not residual else 'release_incomplete',
                  actions=actions, remaining_pids=[p['pid'] for p in residual],
                  gpu_before=before, gpu_after=gpu_snapshot())
    write(Path(output) / 'monitor/cleanup_result.json', report)
    if residual:
        raise RuntimeError('Some registered processes survived cleanup')
    return report


def check(output, config, execute=False):
    ready, pending = completion_gate(output, config)
    running = coordinator_running(output)
    report = dict(at=utc(), monitor_pid=os.getpid(), ready=ready, pending=pending,
                  coordinator_or_scorer_pids=running, state='waiting', gpu=gpu_snapshot(), tasks={})
    for task in TASKS:
        d = Path(output) / task
        report['tasks'][task] = {}
        for name in ('progress.json', 'statistics/train.json', 'statistics/dev.json', 'incremental_counts/train.json', 'incremental_counts/dev.json'):
            if (d / name).exists():
                x = read(d / name)
                report['tasks'][task][name] = {k: x[k] for k in
                    ('at', 'stage', 'completed_source_prefix', 'allocated_source_prefix',
                     'raw_pairs', 'unique_valid_changed_pairs', 'unscored_pairs') if k in x}
    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        report['services'] = list(pool.map(service_health, config['services']))
    if not ready and not running:
        report['state'] = 'incomplete_coordinator_stopped'
    if (Path(output) / 'async_failure.json').exists():
        report['failure'] = read(Path(output) / 'async_failure.json')
    if ready and not running and execute:
        with (Path(output) / 'campaign.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                report['state'] = 'waiting_for_campaign_lock'
            else:
                # Recheck under the same lock used by collection and scoring.
                ready, pending = completion_gate(output, config, verify_hashes=True)
                if ready and not coordinator_running(output):
                    report['cleanup'] = release(config, output)
                    report['state'] = 'released'
                else:
                    report.update(state='completion_verification_pending', pending=pending)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('arm', 'check', 'watch'))
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--config', type=Path, help='Versioned monitor configuration')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == 'arm':
        config = arm(output, args.interval)
        print(json.dumps(dict(armed=True, services=[s['name'] for s in config['services']],
                              interval_seconds=config['interval_seconds'])))
        return
    config = read(args.config or (output / 'monitor/config.json'))
    if config['monitor_sha256'] != sha(__file__) or config['output'] != str(output):
        raise ValueError('Monitor configuration or implementation changed after arming')
    if args.command == 'check':
        print(json.dumps(check(output, config), ensure_ascii=False))
        return
    with (output / 'monitor/watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(output / 'monitor/process.json', dict(pid=os.getpid(), started_at=utc(), interval_seconds=config['interval_seconds']))
        while True:
            try:
                report = check(output, config, execute=True)
            except Exception as e:
                report = dict(at=utc(), monitor_pid=os.getpid(), state='monitor_error', error=repr(e))
            report['next_check_seconds'] = config['interval_seconds']
            write(output / 'monitor/status.json', report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if report['state'] == 'released':
                return
            time.sleep(config['interval_seconds'])


if __name__ == '__main__':
    main()
