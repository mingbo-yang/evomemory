"""Validate each newly ready vLLM endpoint once, including physical GPU placement."""
import json
import os
from pathlib import Path
import re
import subprocess
import time
from manage_vllm_services import dump, now, ready, request

HERE = Path(__file__).resolve().parent
cfg = json.loads((HERE / 'vllm_services.json').read_text())
run = Path(cfg['runtime_dir'])
expected = {m['name']: m for m in cfg['models']}
result = {'started_at_utc': now(), 'backend': 'vllm', 'models': {}}
deadline = time.monotonic() + 3600


def physical_gpu(row):
    info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    ids = {uuid.strip(): index.strip() for line in info.splitlines() for index, uuid in [line.split(',')]}
    usage = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader'], text=True)
    matches = []
    for line in usage.splitlines():
        uuid, pid, memory = [x.strip() for x in line.split(',')]
        try:
            if os.getpgid(int(pid)) == row['pid']:
                matches.append({'gpu': ids[uuid], 'pid': int(pid), 'memory': memory})
        except ProcessLookupError:
            pass
    return matches


while len(result['models']) < len(expected):
    state = json.loads((run / 'state.json').read_text())
    for name, row in state['models'].items():
        if name in result['models']:
            continue
        if row['status'] in ('failed', 'blocked'):
            result['models'][name] = {'passed': False, 'status': row['status'], 'gpu': row['gpu']}
            continue
        if not ready(row):
            continue
        payload = {'model': name, 'messages': [
                   {'role': 'system', 'content': 'You are a precise translator.'},
                   {'role': 'user', 'content': 'Translate into Chinese. Output only the translation: The weather is nice today.'}],
                   'temperature': .1, 'top_p': 1., 'max_tokens': 64, 'seed': 42, 'n': 1,
                   'chat_template_kwargs': {'enable_thinking': False}}
        entry = {'gpu': row['gpu'], 'base_url': row['base_url'], 'pid': row['pid'], 'checked_at_utc': now()}
        try:
            response = request(row['base_url'] + '/chat/completions', payload, timeout=180)
            choice = response['choices'][0]
            text = choice['message']['content']
            actual_gpu = physical_gpu(row)
            entry.update(response=text, finish_reason=choice['finish_reason'], usage=response['usage'],
                         model=response['model'], gpu_processes=actual_gpu)
            entry['passed'] = bool(isinstance(text, str) and re.search(r'[\u3400-\u9fff]', text)
                     and choice['finish_reason'] == 'stop' and response['model'] == name
                     and actual_gpu and {p['gpu'] for p in actual_gpu} == {row['gpu']})
            dump(run / (name + '.smoke.json'), {'request': payload, 'response': response, 'validation': entry})
        except Exception as exc:
            entry.update(passed=False, error=str(exc))
        result['models'][name] = entry
        print(name, json.dumps(entry, ensure_ascii=False), flush=True)
        dump(run / 'validation_live.json', result)
    if time.monotonic() > deadline:
        raise TimeoutError('Deployment validation did not complete; see validation_live.json and server logs')
    if len(result['models']) < len(expected):
        time.sleep(5)
result.update(completed_at_utc=now(), passed=all(row['passed'] for row in result['models'].values()))
dump(run / 'validation.json', result)
print('COMPLETE', result['passed'], flush=True)
raise SystemExit(0 if result['passed'] else 1)
