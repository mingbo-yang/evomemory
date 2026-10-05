"""Fixed-current collection: pure contracts, sealing, and durable artifacts."""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/laya_multigen_data_v1'
TASKS = ('wmt19_en_zh', 'wmt19_zh_en', 'coedit_gec', 'gigaword')
SCHEMA = 'laya-fixed-current-multigen-v1'
INPUT_VERSION = 'source-current-candidate-only-v2'
INPUT_FIELDS = ('source', 'current', 'candidate')
QUESTION = {'t': 'choice', 'ins': 'Should the candidate replace the current answer?',
            'crit': {'Accept': 'The candidate improves answer quality.',
                     'Reject': 'The candidate does not improve answer quality.'}}
MASTER_SEED = 20261004

def utc():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def dump(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + '\n'
    if immutable and path.exists():
        if path.read_text() != payload:
            raise ValueError('Immutable artifact conflict: ' + str(path))
        return
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if immutable:
            os.link(temp, path)
            temp.unlink()
        else:
            os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()

def jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def norm(text):
    import unicodedata
    return ''.join(unicodedata.normalize('NFKC', str(text)).casefold().split())

def text_key(text):
    return hashlib.sha256(norm(text).encode()).hexdigest()

def gec_source(text):
    return text.split(':', 1)[-1].strip()

def seed_for(task, split, source_id, generator, phase, slot=0, salt=0):
    return int(digest([MASTER_SEED, task, split, source_id, generator, phase, slot, salt])[:16], 16) % (2**31)

def candidate_seeds(task, split, source_id, generator):
    seeds = []
    for slot in range(4):
        salt = 0
        value = seed_for(task, split, source_id, generator, 'revision', slot, salt)
        while value in seeds:
            salt += 1
            value = seed_for(task, split, source_id, generator, 'revision', slot, salt)
        seeds.append(value)
    return seeds

def model_input(row):
    x = {k: row[k] for k in INPUT_FIELDS}
    if any(not isinstance(v, str) for v in x.values()):
        raise ValueError('Laya input must contain three text fields')
    return x

def state_text(row):
    x = model_input(row)
    return f"Source:\n{x['source']}\n\nCurrent:\n{x['current']}\n\nCandidate:\n{x['candidate']}"

class InputSizer:
    def __init__(self):
        import sys
        sys.path.insert(0, str(ROOT / 'vendor/laya'))
        from transformers import AutoTokenizer
        from laya.common import build_sequence
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/mnt/huawei/ymb/model/laya-multilingual/tokenizer', local_files_only=True)
        self.prefixes = [build_sequence(self.tokenizer, '', QUESTION, 1024, 256,
                         option_order=o, state_ids=[])[0][:-1] for o in ([0, 1], [1, 0])]

    def length(self, row):
        text = state_text(row).replace(self.tokenizer.mask_token, ' ')
        return max(map(len, self.prefixes)) + len(self.tokenizer.encode(text, add_special_tokens=False)) + 1

def format_valid(generation):
    """Sealed-Test safe: syntax and successful termination only."""
    if generation.get('finish_reason') != 'stop':
        return False, generation.get('finish_reason', 'generation_failure')
    text = generation.get('text', '')
    if not text.strip():
        return False, 'empty'
    if any(s in text.lower() for s in ('<think>', '</think>', chr(96)*3)):
        return False, 'format'
    return True, 'valid'

def valid_pair(task, current, generation, sizer=None, source=''):
    ok, reason = format_valid(generation)
    if not ok:
        return ok, reason
    candidate = generation['text']
    if task == 'wmt19_en_zh' and not re.search(r'[\u3400-\u9fff]', candidate):
        return False, 'no_chinese'
    if current and len(norm(candidate)) > 1.5 * len(norm(current)) + 8:
        return False, 'length'
    if sizer and sizer.length({'source': source, 'current': current, 'candidate': candidate}) > 1024:
        return False, 'laya_context_overflow'
    return True, 'valid'

def label_scores(before, after):
    if not math.isfinite(before) or not math.isfinite(after):
        raise ValueError('Nonfinite quality score')
    delta = after - before
    return ('Accept' if delta > 0 else 'Reject'), delta

def require_unsealed(task_dir, split):
    """Check BEFORE reading Test references, cleaning, scoring, or inference."""
    if split != 'test':
        return
    task_dir = Path(task_dir)
    path = task_dir / 'evaluation_lock.json'
    if not path.exists():
        raise PermissionError('Test is sealed: scoring, cleaning, export and Laya inference are forbidden')
    lock = read(path)
    validate_evaluation_lock(task_dir, lock)


def validate_evaluation_lock(task_dir, lock):
    task_dir = Path(task_dir)
    required = ('model', 'loss', 'selection_rule', 'evaluation')
    if lock.get('schema') != SCHEMA or any(not lock.get(k) for k in required):
        raise PermissionError('Incomplete evaluation lock')
    if lock.get('protocol_sha256') != file_hash(task_dir / 'protocol.json'):
        raise PermissionError('Evaluation lock does not match protocol')
    if lock['selection_rule'].get('uses_test') is not False:
        raise PermissionError('Checkpoint selection must not use Test')
    if lock['evaluation'].get('bootstrap_unit') != 'source':
        raise PermissionError('Evaluation must cluster by source')
    if not lock['model'].get('checkpoint_weights'):
        raise PermissionError('A selected checkpoint weight file must be frozen')
    weight_file = lock['model']['checkpoint_weights']
    if weight_file not in lock['model'].get('artifacts', {}):
        raise PermissionError('Selected model weights are not hashed')
    if lock['loss'].get('statistics_split') != 'train':
        raise PermissionError('Loss weights must be based only on Train')
    if lock['selection_rule'].get('selection_split') != 'dev':
        raise PermissionError('Checkpoint selection must use Dev')
    if lock['evaluation'].get('bootstrap_draws') != 2000 or not lock['evaluation'].get('comparisons'):
        raise PermissionError('Freeze the bootstrap protocol and comparison objects')
    for key in required:
        entry = lock[key]
        if not entry.get('artifacts'):
            raise PermissionError('Missing frozen artifacts for ' + key)
        for artifact, expected in entry['artifacts'].items():
            if file_hash(artifact) != expected:
                raise PermissionError('Frozen evaluation artifact changed: ' + artifact)

@contextlib.contextmanager
def timing(task_dir, stage, split='', generator='', shard=''):
    start = time.time()
    mono = time.monotonic()
    record = {'stage': stage, 'split': split, 'generator': generator, 'shard': str(shard),
              'started_at_utc': utc(), 'start_unix': start, 'status': 'running'}
    try:
        yield record
        record['status'] = 'complete'
    except BaseException as e:
        record['status'] = 'failed'
        record['error_type'] = type(e).__name__
        raise
    finally:
        record.update(finished_at_utc=utc(), end_unix=time.time(), elapsed_seconds=time.monotonic()-mono)
        dump(Path(task_dir) / 'timing_events' / (uuid.uuid4().hex + '.json'), record, immutable=True)

def union_seconds(intervals):
    end, total = None, 0.0
    for a, b in sorted(intervals):
        total += max(0.0, b - max(a, end if end is not None else a))
        end = max(b, end if end is not None else b)
    return total

def summarize_timing(task_dir):
    task_dir = Path(task_dir)
    events = [read(p) for p in (task_dir / 'timing_events').glob('*.json')]
    groups = {}
    for e in events:
        key = '/'.join((e['split'], e['generator'], e['stage']))
        groups.setdefault(key, []).append(e)
    report = {'updated_at_utc': utc(), 'timezone': 'UTC', 'groups': {},
              'test_scoring': 'sealed_not_executed', 'test_cleaning': 'sealed_not_executed'}
    for key, rows in groups.items():
        intervals = [(e['start_unix'], e['end_unix']) for e in rows]
        active = union_seconds(intervals)
        span = max(b for a, b in intervals) - min(a for a, b in intervals)
        report['groups'][key] = {'started_at_utc': min(e['started_at_utc'] for e in rows),
                'finished_at_utc': max(e['finished_at_utc'] for e in rows),
                'active_seconds': active, 'wall_seconds': span, 'inactive_seconds': max(0, span-active),
                'sessions': len(rows)}
    report['active_wall_seconds'] = union_seconds([(e['start_unix'], e['end_unix']) for e in events])
    generation = [e for e in events if e['stage'] == 'generation']
    report['generation_active_wall_seconds'] = union_seconds([(e['start_unix'], e['end_unix']) for e in generation])
    requests = {r['request_hash']:dict(r,split=e['split'],generator=e['generator']) for e in events for r in e.get('requests',[])}
    report['request_timing_groups'] = {}
    for split in ('train','dev','test'):
        selected = [r for r in requests.values() if r['split']==split]
        intervals = [(datetime.fromisoformat(r['started_at_utc']).timestamp(),datetime.fromisoformat(r['finished_at_utc']).timestamp()) for r in selected]
        elapsed = union_seconds(intervals)
        pairs = sum(r['phase']=='revision' for r in selected)
        item = {'unique_requests':len(selected),'raw_pairs':pairs,'generation_wall_seconds':elapsed,
                'raw_pairs_per_hour':pairs*3600/elapsed if elapsed else None,
                'retry_attempt_seconds':sum(a['seconds'] for r in selected for a in r['attempts'] if a['status']!='success')}
        if split!='test' and (task_dir/'statistics'/(split+'.json')).exists():
            changed=read(task_dir/'statistics'/(split+'.json'))['unique_valid_changed_pairs']
            item['unique_valid_changed_pairs_per_hour']=changed*3600/elapsed if elapsed else None
        report['request_timing_groups'][split]=item
    dump(task_dir / 'timing.json', report)
    return report


def guard_sealed_path(path):
    path = Path(path).absolute()
    for parent in path.parents:
        if (parent / 'test_seal.json').exists():
            relative = path.relative_to(parent)
            if 'test' in relative.parts or any(v.startswith('test.') or v.startswith('test_') for v in relative.parts):
                require_unsealed(parent, 'test')
            return


def freeze_evaluation_lock(task_dir, specification):
    task_dir = Path(task_dir)
    lock = dict(specification, schema=SCHEMA, protocol_sha256=file_hash(task_dir/'protocol.json'))
    validate_evaluation_lock(task_dir, lock)
    dump(task_dir/'evaluation_lock.json', lock, immutable=True)
    return lock
