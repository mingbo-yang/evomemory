"""Start/check/stop only this deployment's local vLLM processes.

One server per model; models sharing a GPU initialize serially to avoid
profiling during another engine's memory allocation. Existing GPU jobs remain
untouched. Model weights and experimental inference code are not modified.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

HERE = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def owned_alive(row):
    return bool(row.get('pid') and process_start(row['pid']) == row.get('process_start'))


def request(url, payload=None, timeout=3):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def ready(row):
    if not owned_alive(row):
        return False
    try:
        request(row['base_url'][:-3] + '/health', timeout=1)
        models = request(row['base_url'] + '/models', timeout=2)
        return row['name'] in {m['id'] for m in models['data']}
    except Exception:
        return False


def start(cfg, run):
    state_path = run / 'state.json'
    if state_path.exists():
        raise FileExistsError('Deployment state exists; inspect status instead of starting duplicates: ' + str(state_path))
    state = {'created_at_utc': now(), 'backend': 'vllm', 'config': cfg, 'models': {}}
    processes = {}
    pending = list(cfg['models'])
    for model in pending:
        with socket.socket() as sock:
            sock.bind((cfg['host'], model['port']))
        assert (Path(model['path']) / 'config.json').is_file()
    deadline = time.monotonic() + 3600
    while pending or any(r['status'] == 'starting' for r in state['models'].values()):
        for model in list(pending):
            dependency = model.get('start_after')
            if dependency:
                prior = state['models'].get(dependency, {})
                if prior.get('status') == 'failed':
                    state['models'][model['name']] = dict(model, status='blocked', reason='co-located model startup failed')
                    pending.remove(model)
                    continue
                if prior.get('status') != 'ready':
                    continue
            command = [cfg['python'], '-u', '-m', 'vllm.entrypoints.openai.api_server',
                       '--host', cfg['host'], '--port', str(model['port']), '--model', model['path'],
                       '--served-model-name', model['name'], '--trust-remote-code', '--dtype', cfg['dtype'],
                       '--tensor-parallel-size', '1', '--max-model-len', str(cfg['max_model_len']),
                       '--gpu-memory-utilization', str(model['gpu_memory_utilization']),
                       '--max-num-seqs', str(cfg['max_num_seqs']), '--max-num-batched-tokens', '4096',
                       '--enforce-eager', '--load-format', 'safetensors', '--safetensors-load-strategy', model['load_strategy'],
                       '--generation-config', 'vllm', '--override-generation-config', '{"temperature":0.1,"top_p":1.0}',
                       '--default-chat-template-kwargs', '{"enable_thinking":false}',
                       '--disable-log-stats', '--uvicorn-log-level', 'warning']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=model['gpu'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       HF_MODULES_CACHE='/mnt/huawei/ymb/.tmp/vllm_hf_modules',
                       VLLM_CACHE_ROOT='/mnt/huawei/ymb/.tmp/vllm_serve_cache')
            log = run / (model['name'] + '.log')
            with log.open('w') as f:
                proc = subprocess.Popen(command, env=env, cwd=HERE.parent, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            processes[model['name']] = proc
            state['models'][model['name']] = dict(model, pid=proc.pid, process_start=process_start(proc.pid),
                   base_url=f"http://{cfg['host']}:{model['port']}/v1", log=str(log), command=command,
                   status='starting', started_at_utc=now())
            pending.remove(model)
            print('START', model['name'], 'GPU', model['gpu'], 'PID', proc.pid, flush=True)
        for name, row in state['models'].items():
            if row['status'] != 'starting':
                continue
            proc = processes[name]
            if proc.poll() is not None:
                row.update(status='failed', exit_code=proc.returncode)
                print('FAILED', name, 'exit', proc.returncode, flush=True)
            elif ready(row):
                row.update(status='ready', ready_at_utc=now())
                print('READY', name, row['base_url'], flush=True)
        state['updated_at_utc'] = now()
        dump(state_path, state)
        if time.monotonic() > deadline:
            raise TimeoutError('Startup time limit reached; inspect model logs. Processes are retained for diagnosis.')
        if pending or any(r['status'] == 'starting' for r in state['models'].values()):
            time.sleep(5)
    print(json.dumps({n: r['status'] for n, r in state['models'].items()}), flush=True)
    return 0 if all(r['status'] == 'ready' for r in state['models'].values()) else 1


def check(state, run, generate=False):
    result = {}
    for name, row in state['models'].items():
        entry = {'gpu': row['gpu'], 'base_url': row.get('base_url'), 'pid': row.get('pid'), 'alive': owned_alive(row)}
        entry['ready'] = ready(row)
        if generate and entry['ready']:
            payload = {'model': name, 'messages': [{'role': 'system', 'content': 'You are a precise translator.'},
                       {'role': 'user', 'content': 'Translate into Chinese. Output only the translation: The weather is nice today.'}],
                       'temperature': .1, 'top_p': 1., 'max_tokens': 64, 'seed': 42, 'n': 1,
                       'chat_template_kwargs': {'enable_thinking': False}}
            try:
                reply = request(row['base_url'] + '/chat/completions', payload, timeout=180)
                text = reply['choices'][0]['message']['content']
                entry.update(smoke_passed=isinstance(text, str) and bool(text.strip()), response=text,
                             finish_reason=reply['choices'][0]['finish_reason'], usage=reply['usage'], returned_model=reply['model'])
                dump(run / (name + '.smoke.json'), {'request': payload, 'response': reply, 'checked_at_utc': now()})
            except Exception as exc:
                entry.update(smoke_passed=False, error=str(exc))
        result[name] = entry
    if generate:
        dump(run / 'validation.json', {'checked_at_utc': now(), 'models': result})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if len(result) == len(state['config']['models']) and all(r['ready'] and (not generate or r.get('smoke_passed')) for r in result.values()) else 1


def stop(state, run):
    for name, row in state['models'].items():
        if owned_alive(row):
            argv = Path(f"/proc/{row['pid']}/cmdline").read_bytes().split(b'\0')
            if b'vllm.entrypoints.openai.api_server' not in argv or name.encode() not in argv:
                raise RuntimeError('Refusing to signal unrecognized process: ' + name)
            os.killpg(row['pid'], signal.SIGTERM)
            row['status'] = 'stop_requested'
            print('STOP', name, row['pid'])
    state['updated_at_utc'] = now()
    dump(run / 'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status', 'test', 'stop'])
    parser.add_argument('--config', type=Path, default=HERE / 'vllm_services.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = Path(config['runtime_dir'])
    runtime.mkdir(parents=True, exist_ok=True)
    if args.action == 'start':
        raise SystemExit(start(config, runtime))
    current = json.loads((runtime / 'state.json').read_text())
    if args.action == 'stop':
        stop(current, runtime)
    else:
        raise SystemExit(check(current, runtime, generate=args.action == 'test'))
