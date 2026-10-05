"""HTTP-only collection. This module never loads references, quality scores, or Laya."""
from __future__ import annotations
import concurrent.futures as futures
import itertools
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from . import common as C

def health(protocol):
    checks = []
    for model in protocol['models']:
        url = 'http://127.0.0.1:%d/v1/models' % model['port']
        response = json.load(urllib.request.urlopen(url, timeout=15))
        found = [x for x in response['data'] if x['id'] == model['name']]
        if len(found) != 1 or found[0]['root'] != model['path'] or found[0]['max_model_len'] != protocol['max_model_len']:
            raise RuntimeError('Deployed model identity mismatch: ' + model['name'])
        checks.append({'model': model['name'], 'root': found[0]['root'], 'max_model_len': found[0]['max_model_len']})
    return checks

def verify_protocol(task_dir, full=False):
    task_dir = Path(task_dir)
    p = C.read(task_dir / 'protocol.json')
    if p['schema'] != C.SCHEMA or p['states'] != 1 or p['candidates'] != 4:
        raise ValueError('Wrong collection protocol')
    for path, expected in p['code_hashes'].items():
        if C.file_hash(path) != expected:
            raise ValueError('Frozen code changed: ' + path)
    for name, expected in p['artifact_hashes'].items():
        if full or name in ('initial_memory.jsonl', 'retrieval.json'):
            if C.file_hash(task_dir / name) != expected:
                raise ValueError('Frozen input changed: ' + name)
    return p

class HTTPGenerator:
    def __init__(self, model, protocol, directory):
        from transformers import AutoTokenizer
        self.model, self.protocol, self.directory = model, protocol, Path(directory)
        self.tokenizer = AutoTokenizer.from_pretrained(model['path'], trust_remote_code=True, local_files_only=True)
        self.url = 'http://127.0.0.1:%d/v1/chat/completions' % model['port']

    def tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, messages, *, source_id, split, phase, slot, seed, temperature):
        payload = {'model': self.model['name'], 'messages': messages, 'temperature': temperature,
                   'top_p': self.protocol['top_p'], 'max_tokens': self.protocol['max_tokens'],
                   'seed': seed, 'stream': False, 'chat_template_kwargs': {'enable_thinking': False}}
        request_hash = C.digest(payload)
        key = source_id.split('/')[-1]
        relative = Path('requests') / split / self.model['name'] / key[:2] / (key + '.' + phase + '.' + str(slot) + '.json')
        path = self.directory / relative
        if path.exists():
            result = C.read(path)
            if result['request_hash'] != request_hash or result['record_hash'] != C.digest({k:v for k,v in result.items() if k != 'record_hash'}):
                raise ValueError('Cached request identity/integrity mismatch')
            return result
        started = C.utc()
        mono = time.monotonic()
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        finish, text, usage, attempts = 'context_length_exceeded', '', {}, []
        if len(ids) + payload['max_tokens'] <= self.protocol['max_model_len']:
            for attempt in range(3):
                t = time.monotonic()
                try:
                    request = urllib.request.Request(self.url, data=C.canonical(payload).encode(),
                                                     headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(request, timeout=600) as response:
                        response = json.load(response)
                    if response.get('model') != self.model['name']:
                        raise ValueError('Response model identity mismatch')
                    choice = response['choices'][0]
                    text, finish = choice['message'].get('content') or '', choice['finish_reason']
                    usage = response.get('usage', {})
                    attempts.append({'attempt': attempt, 'status': 'success', 'seconds': time.monotonic()-t})
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    attempts.append({'attempt': attempt, 'status': 'transport_failure',
                                     'error_type': type(e).__name__, 'seconds': time.monotonic()-t})
                    finish = 'generation_failure'
                    if attempt < 2:
                        time.sleep(2 ** attempt)
        from baseline_core.tasks import get_adapter
        parsed = get_adapter(self.protocol['task']).parse_output(text)
        result = {'request_hash': request_hash, 'request_id': request_hash, 'request': payload,
                  'raw_record_location': str(relative), 'source_id': source_id, 'split': split,
                  'generator': self.model['name'], 'model_version': self.model['identity'],
                  'protocol_version': C.SCHEMA, 'protocol_sha256': C.file_hash(self.directory / 'protocol.json'),
                  'phase': phase, 'slot': slot, 'seed': seed, 'temperature': temperature,
                  'text': parsed, 'raw_text': text, 'finish_reason': finish,
                  'usage': usage, 'input_tokens_preflight': len(ids), 'attempts': attempts,
                  'started_at_utc': started, 'finished_at_utc': C.utc(), 'elapsed_seconds': time.monotonic()-mono}
        result['record_hash'] = C.digest(result)
        C.dump(path, result, immutable=True)
        return result

class FrozenRetrieval:
    def __init__(self, task_dir, generator):
        from core.experience import load_experiences
        from core.hybrid_retrieval import HybridExperienceRetriever, LocalSentenceEncoder
        self.library = load_experiences(Path(task_dir) / 'initial_memory.jsonl')
        cfg = C.read(Path(task_dir) / 'retrieval.json')
        options = dict(cfg['retrieval'])
        options.pop('method')
        self.retriever = HybridExperienceRetriever(self.library, LocalSentenceEncoder(cfg['encoder'], 'cpu'), **options)
        self.by_id = {e.exp_id:e for e in self.library}
        self.generator = generator
        self.lock = threading.Lock()

    def __call__(self, source, current):
        from core.experience import render_experience_block
        with self.lock:
            r = self.retriever.retrieve(source, current, alpha=.5, k=4, exclude_source=source)
        text = render_experience_block([self.by_id[i] for i in r.exp_ids], include_outcome=True,
                    count_tokens=self.generator.tokens, max_units=4, contrastive=True, advice_mode='summary')
        return {'ids': r.exp_ids, 'text': text, 'context_hash': C.digest(text)}

def collect_one(row, task, model, protocol, generator, retrieve):
    """Reference-independent entry point; retrieval is called only AFTER the draft."""
    from baseline_core.tasks import get_adapter
    from baseline_core.types import TaskExample
    from core.pipeline import build_refine_prompt
    from core.refinement_instructions import INSTRUCTIONS
    adapter = get_adapter(task)
    source, sid, split = row['source'], row['source_id'], row['split']
    example = TaskExample(index=row.get('position', 0), source=source, reference='', task=task)
    system = adapter.system_prompt()
    messages = [{'role':'system','content':system}, {'role':'user','content':adapter.initial_prompt(example)}]
    draft = generator.generate(messages, source_id=sid, split=split, phase='initial', slot=0,
               seed=C.seed_for(task, split, sid, model, 'initial'), temperature=.1)
    result = {'dataset':task, 'split':split, 'source_id':sid, 'source':source,
              'cluster_id':row['cluster_id'], 'position':row['position'], 'generator':model,
              'initial':draft, 'candidates':[], 'retrieval':None, 'schema':C.SCHEMA}
    if not C.format_valid(draft)[0]:
        result['status'] = 'initial_invalid'
        return result
    current = draft['text']
    context = retrieve(source, current)
    prompt = build_refine_prompt(adapter, example, current, INSTRUCTIONS[task],
                                 experience_block=context['text'], renderer='v2')
    result['retrieval'] = context
    messages = [{'role':'system','content':system}, {'role':'user','content':prompt}]
    temps = protocol['train_temperatures'] if split == 'train' else protocol['nontrain_temperatures']
    for slot, seed in enumerate(C.candidate_seeds(task, split, sid, model)):
        result['candidates'].append(generator.generate(messages, source_id=sid, split=split,
                 phase='revision', slot=slot, seed=seed, temperature=temps[slot]))
    result['status'] = 'complete'
    return result

class Collector:
    def __init__(self, task_dir, model):
        self.directory = Path(task_dir)
        self.protocol = verify_protocol(self.directory)
        self.task, self.model = self.protocol['task'], model['name']
        self.generator = HTTPGenerator(model, self.protocol, self.directory)
        self.retrieval = FrozenRetrieval(self.directory, self.generator)

    def one(self, row):
        key = row['source_id'].split('/')[-1]
        path = self.directory / 'raw' / row['split'] / self.model / key[:2] / (key + '.json')
        if path.exists():
            value = C.read(path)
            if value['record_hash'] != C.digest({k:v for k,v in value.items() if k != 'record_hash'}):
                raise ValueError('Raw record checksum mismatch')
            return str(path)
        result = collect_one(row, self.task, self.model, self.protocol, self.generator, self.retrieval)
        result['record_hash'] = C.digest(result)
        C.dump(path, result, immutable=True)
        return str(path)

    def run(self, split, start, stop):
        rows = list(itertools.islice(C.jsonl(self.directory / 'manifests' / (split+'.sources.jsonl')), start, stop))
        completion = self.directory / 'completed' / split / self.model / (f'{start:08d}-{stop:08d}.json')
        if completion.exists():
            return C.read(completion)
        concurrency = 8 if self.model in ('qwen3-4b', 'llama3.1-8b') else 16
        with C.timing(self.directory, 'generation', split, self.model, f'{start}:{stop}') as event:
            paths = []
            with futures.ThreadPoolExecutor(concurrency) as pool:
                for path in pool.map(self.one, rows):
                    paths.append(path)
            records = [{'path':p, 'sha256':C.file_hash(p)} for p in paths]
            result = {'sources':len(rows), 'split':split, 'generator':self.model,
                      'start':start, 'stop':stop, 'raw_records':records}
            event['completed_sources'] = len(rows)
            event['requests'] = []
            for path in paths:
                record = C.read(path)
                for g in [record['initial']] + record['candidates']:
                    event['requests'].append({k:g[k] for k in ('request_hash','phase','started_at_utc','finished_at_utc','elapsed_seconds','attempts')})
            C.dump(completion, result, immutable=True)
        return result

def run_shard(collectors, split, start, stop):
    with futures.ThreadPoolExecutor(len(collectors)) as pool:
        pending = [pool.submit(c.run, split, start, stop) for c in collectors]
        return [f.result() for f in pending]

def sealed_audit(task_dir):
    """No labels, no no-op comparisons, no cleaning, and no Laya/metric imports."""
    from collections import Counter
    task_dir = Path(task_dir)
    counters = Counter()
    hashes = []
    for path in sorted((task_dir / 'raw/test').glob('*/*/*.json')):
        row = C.read(path)
        if row['record_hash'] != C.digest({k:v for k,v in row.items() if k != 'record_hash'}):
            raise ValueError('Sealed raw record checksum mismatch')
        counters['source_generator_records'] += 1
        for g in [row['initial']] + row['candidates']:
            ok, reason = C.format_valid(g)
            counters['requests'] += 1
            counters['format_valid' if ok else 'format_invalid'] += 1
            counters['finish_' + g['finish_reason']] += 1
        hashes.append({'path':str(path.relative_to(task_dir)), 'sha256':C.file_hash(path)})
    report = {'state':'sealed', 'checks':dict(counters), 'files':hashes,
              'forbidden_operations_executed':False, 'at':C.utc()}
    C.dump(task_dir / 'test_integrity_audit.json', report)
    return {k:v for k,v in report.items() if k != 'files'}
