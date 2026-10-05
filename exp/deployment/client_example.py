"""Minimal stdlib client for the local vLLM deployment; no extra SDK needed."""
import argparse
import json
from pathlib import Path
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', default='qwen3-8b')
p.add_argument('--prompt', default='Translate into Chinese. Output only the translation: The weather is nice today.')
a = p.parse_args()
cfg = json.loads(Path(__file__).with_name('vllm_services.json').read_text())
models = {m['name']: m for m in cfg['models']}
if a.model not in models:
    p.error('Unknown model; choose one of ' + ', '.join(models))
m = models[a.model]
payload = {'model': a.model, 'messages': [{'role': 'user', 'content': a.prompt}],
           'temperature': .1, 'top_p': 1., 'max_tokens': 1024, 'seed': 42, 'n': 1,
           'chat_template_kwargs': {'enable_thinking': False}}
request = urllib.request.Request(f"http://{cfg['host']}:{m['port']}/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=180) as response:
    result = json.load(response)
print(result['choices'][0]['message']['content'])
