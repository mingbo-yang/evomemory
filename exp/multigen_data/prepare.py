"""Prepare independent source pools and audited shared memory before any generation."""
from __future__ import annotations
import csv
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from . import common as C

WMT = Path('/mnt/huawei/wwq/project/No-box-translation/ISSTA2025/11.2-ISSTA/data/wmt19-en-zh')
GIGA = Path('/mnt/huawei/ymb/datasets/gigaword_full_english')
TINY = Path('/mnt/huawei/ymb/datasets/datasets--SpeedOfMagic--gigaword_tiny/snapshots/f2876f564f2a11d1781a265b60faea461da72d5b/data')
COEDIT = Path('/mnt/huawei/ymb/.cache/huggingface/datasets/grammarly___coedit')
TEXT_FIELDS = {'source', 'source_input', 'source_text', 'src', 'original', 'reference',
               'reference_text', 'ref', 'tgt', '原文', '完美答案', 'state_before', 'state_after'}

def texts(value):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in TEXT_FIELDS and isinstance(v, str):
                yield v
                if re.match(r'^(Fix |Correct |Remove |Improve ).{0,100}:', v):
                    yield C.gec_source(v)
            elif isinstance(v, (dict, list)):
                yield from texts(v)
    elif isinstance(value, list):
        for v in value:
            yield from texts(v)

def pq_rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=10000):
        yield from batch.to_pylist()

def wmt_rows():
    for i, row in enumerate(pq_rows(WMT / 'train.parquet')):
        t = row['translation']
        yield i, t['en'], t['zh']

def coedit_rows():
    import pyarrow as pa
    import pyarrow.ipc as ipc
    path = next(COEDIT.rglob('coedit-train.arrow'))
    with pa.memory_map(str(path), 'r') as f:
        table = ipc.open_stream(f).read_all()
    i = 0
    for row in table.to_pylist():
        if row['task'] == 'gec':
            yield i, C.gec_source(row['src']), row['tgt']
            i += 1

def giga_rows():
    i = 0
    for path in sorted((GIGA / 'data').glob('train-*.parquet')):
        for row in pq_rows(path):
            yield i, row.get('document', row.get('article')), row['summary']
            i += 1

def inventory():
    paths = set((C.ROOT / 'data').rglob('*.jsonl'))
    paths.update((C.ROOT / 'experience').rglob('*.jsonl'))
    paths.update((C.ROOT / 'runs').rglob('*.jsonl'))
    paths = {p for p in paths if 'laya_multigen_data_' not in str(p) and '/code_before/' not in str(p)}
    paths.update((C.ROOT / 'runs').rglob('*source_split*.json'))
    paths.update((C.ROOT / 'runs').rglob('*data_manifest*.json'))
    for p in Path('/mnt/huawei/ymb/icml/rag').glob('*dataset_analysis*.csv'):
        paths.add(p)
    for p in Path('/mnt/huawei/ymb/icml/bert').rglob('*'):
        if p.suffix in ('.csv', '.jsonl', '.json') and 'model' not in p.parts:
            paths.add(p)
    return sorted(paths)


def history_hashes(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ('source_hash','reference_hash','source_hashes','seen_source_hashes'):
                stack=[v]
                while stack:
                    x=stack.pop()
                    if isinstance(x,dict):stack.extend(x.values())
                    elif isinstance(x,list):stack.extend(x)
                    elif isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x):yield x
            elif isinstance(v,(dict,list)):
                yield from history_hashes(v)
    elif isinstance(value,list):
        for x in value:yield from history_hashes(x)

def isolation_keys(a,b):
    return {C.text_key(a),C.text_key(b),hashlib.sha256(a.encode()).hexdigest(),hashlib.sha256(b.encode()).hexdigest()}

def historical_jsonl(path,recoveries):
    with path.open(encoding='utf-8-sig') as f:
        for number,line in enumerate(f,1):
            if not line.strip():continue
            if chr(0) in line:
                recoveries.append({'line':number,'nul_bytes_removed':line.count(chr(0))})
                line=line.replace(chr(0),'')
            if not line.strip():continue
            try:yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Unrecoverable historical record {path}:{number}') from e

def build_blocklist(out):
    blocked, records = set(), []
    for p in inventory():
        count = 0
        recoveries = []
        if p.suffix == '.csv':
            with p.open(encoding='utf-8-sig') as f:
                iterator = list(csv.DictReader(f))
        elif p.suffix == '.json':
            obj = C.read(p)
            iterator = [obj]
            if 'source_split' in p.name:
                for values in obj.values():
                    if isinstance(values, list):
                        for v in values:
                            if isinstance(v, str):
                                blocked.add(C.text_key(v))
        else:
            iterator = historical_jsonl(p,recoveries)
        for row in iterator:
            blocked.update(history_hashes(row))
            for t in texts(row):
                if t.strip():
                    blocked.add(C.text_key(t))
                    count += 1
        records.append({'path': str(p), 'sha256': C.file_hash(p), 'text_occurrences': count, 'read_only_nul_recovery':recoveries})
    # Protect the complete formal validation pool, not only 1000-row experiment manifests.
    for row in pq_rows(WMT / 'validation.parquet'):
        blocked.update(C.text_key(v) for v in row['translation'].values())
    # Protect every previously available tiny split, including auxiliary memory sources.
    for p in TINY.glob('*.parquet'):
        for row in pq_rows(p):
            blocked.update(C.text_key(row[k]) for k in ('document', 'summary'))
    C.dump(out / 'history_inventory.json', {'files': records, 'blocked_texts': len(blocked)}, immutable=True)
    return blocked

def verify_gigaword(out):
    manifest = C.read(GIGA / 'download_manifest.json')
    existing = out / 'gigaword_provenance.json'
    for name, item in manifest['files'].items():
        if C.file_hash(GIGA / name) != item['sha256']:
            raise ValueError('Gigaword checksum mismatch: ' + name)
    if existing.exists():
        audit = C.read(existing)
        if audit['revision'] != manifest['revision']:
            raise ValueError('Gigaword revision changed')
        return audit
    tiny = {}
    for p in TINY.glob('*.parquet'):
        for row in pq_rows(p):
            key = C.digest([C.norm(row['document']).replace('<unk>', 'unk'),
                            C.norm(row['summary']).replace('<unk>', 'unk')])
            tiny.setdefault(key, set()).add(p.name.split('-')[0])
    found = {}
    counts = {}
    for p in sorted((GIGA / 'data').glob('*.parquet')):
        n = 0
        for row in pq_rows(p):
            n += 1
            key = C.digest([C.norm(row.get('document', row.get('article'))).replace('<unk>', 'unk'),
                            C.norm(row['summary']).replace('<unk>', 'unk')])
            if key in tiny:
                found.setdefault(key, set()).add(p.name.split('-')[0])
        counts[p.name] = n
    report = {'repo': manifest['repo'], 'revision': manifest['revision'], 'rows': counts,
              'tiny_unique_pairs': len(tiny), 'matched_unique_pairs': len(found),
              'split_mapping': dict(Counter((a, b) for k in found for a in tiny[k] for b in found[k]))}
    report['split_mapping'] = {a + '->' + b: n for (a, b), n in report['split_mapping'].items()}
    if len(found) != len(tiny):
        raise ValueError('Full Gigaword provenance mismatch: ' + C.canonical(report))
    C.dump(out / 'gigaword_provenance.json', report, immutable=True)
    return report

def memories(out):
    from core.experience import load_experiences, save_experiences
    formal = set()
    for row in pq_rows(WMT / 'validation.parquet'):
        formal.update(C.text_key(t) for t in row['translation'].values())
    for name in ('coedit_gec', 'gigaword'):
        with Path('/mnt/huawei/ymb/icml/rag', name + '_dataset_analysis_full.csv').open(encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                formal.update(C.text_key(row[k]) for k in ('原文', '完美答案'))
    all_memory_keys = set()
    report = {}
    for task in C.TASKS:
        merged, provenance = {}, {}
        for p in sorted((C.ROOT / 'experience/initial').glob('*/' + task + '/initial.jsonl')):
            for e in load_experiences(p):
                source = C.gec_source(e.source_input) if task == 'coedit_gec' else e.source_input
                keys = {C.text_key(t) for t in (source, e.state_before, e.state_after) if t.strip()}
                all_memory_keys.update(keys)
                if keys & formal:
                    continue
                key = C.digest([task, source, e.state_before, e.state_after, e.intervention_instruction])
                merged.setdefault(key, replace(e, exp_id='shared-' + key[:24], source_input=source, model='shared'))
                provenance.setdefault(key, []).append({'path': str(p), 'exp_id': e.exp_id, 'model': e.model})
        if not merged:
            raise ValueError('No clean shared initial memory for ' + task)
        directory = out / task
        directory.mkdir(parents=True, exist_ok=True)
        save_experiences([merged[k] for k in sorted(merged)], directory / 'initial_memory.jsonl')
        C.dump(directory / 'memory_provenance.json', provenance, immutable=True)
        report[task] = {'units': len(merged), 'formal_overlap': 0,
                       'sha256': C.file_hash(directory / 'initial_memory.jsonl')}
    C.dump(out / 'memory_audit.json', report, immutable=True)
    return all_memory_keys

def source_row(task, index, source, reference):
    return {'task': task, 'source_id': task + '/' + C.text_key(source),
            'source': source, 'reference': reference, 'original_row': index,
            'source_hash': C.text_key(source), 'reference_hash': C.text_key(reference)}

def write_pool(out, task, rows):
    random.Random(C.seed_for(task, 'prepare', '', '', 'pool')).shuffle(rows)
    if len(rows) < 6000:
        raise ValueError(f'{task}: only {len(rows)} isolated sources; need at least 6000')
    d = out / task
    artifacts, counts = {}, {}
    # All remaining training sources are reserved up front, not capped at 5000.
    splits = {'dev': rows[:500], 'test': rows[500:1000], 'train': rows[1000:]}
    for split, samples in splits.items():
        sp = d / 'manifests' / (split + '.sources.jsonl')
        rp = d / 'references' / (split + '.jsonl')
        sp.parent.mkdir(parents=True, exist_ok=True)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with sp.open('x') as sf, rp.open('x') as rf:
            for pos, row in enumerate(samples):
                public = {k: v for k, v in row.items() if k != 'reference'}
                public.update(split=split, position=pos, cluster_id=row['source_hash'])
                sf.write(C.canonical(public) + '\n')
                rf.write(C.canonical({'source_id': row['source_id'], 'reference': row['reference']}) + '\n')
        artifacts[str(sp.relative_to(d))] = C.file_hash(sp)
        artifacts[str(rp.relative_to(d))] = C.file_hash(rp)
        counts[split] = len(samples)
    return artifacts, counts

def prepare(out=C.OUT):
    out = Path(out)
    if (out / 'prepared.json').exists():
        return C.read(out / 'prepared.json')
    if (out / 'history_inventory.json').exists():
        raise RuntimeError('Partial preparation exists; inspect it before restarting into a fresh version')
    out.mkdir(parents=True, exist_ok=True)
    C.dump(out / 'preparation_status.json', {'phase': 'history_inventory', 'started_at_utc': C.utc()})
    blocked = build_blocklist(out)
    blocked.update(memories(out))
    C.dump(out / 'preparation_status.json', {'phase': 'gigaword_provenance', 'at': C.utc()})
    giga_provenance = verify_gigaword(out)
    deployment = C.read(C.ROOT / 'deployment/vllm_services.json')
    from .cli import models_with_identity
    deployment['models'] = models_with_identity(deployment['models'])
    retrieval = C.read(C.ROOT / 'configs/optimized_feedback_v1.json')
    dependencies = list((C.ROOT / 'multigen_data').glob('*.py')) + [
        C.ROOT / 'core/pipeline.py', C.ROOT / 'core/experience.py', C.ROOT / 'core/hybrid_retrieval.py',
        C.ROOT / 'core/bm25_fields.py', C.ROOT / 'core/scoring.py', C.ROOT / 'core/refinement_instructions.py',
        C.ROOT.parent / 'baseline/core/tasks.py']
    code_hashes = {str(p): C.file_hash(p) for p in dependencies}
    from importlib.metadata import version
    versions = {k:version(k) for k in ('vllm','torch','transformers','nltk','rouge-score','sentence-transformers')}
    feedback = dict(C.read(C.ROOT / 'configs/laya_binary_v1.json')['feedback'])
    feedback.update(device='cuda',visible_gpu='0',batch_size=16,checkpoint_sha256=C.file_hash(feedback['checkpoint']))
    source_files = [WMT/'train.parquet',WMT/'validation.parquet',next(COEDIT.rglob('coedit-train.arrow')),GIGA/'download_manifest.json']
    source_artifacts = {str(p):C.file_hash(p) for p in source_files}
    # Drop ALL rows whose EN or ZH side is duplicated. This conservatively excludes
    # ambiguous bilingual connected groups rather than splitting them across tasks.
    C.dump(out / 'preparation_status.json', {'phase': 'wmt_duplicate_groups', 'at': C.utc()})
    frequencies = Counter()
    for i, a, b in wmt_rows():
        frequencies[C.text_key(a)] += 1
        frequencies[C.text_key(b)] += 1
    wmt_pools = {t: [] for t in C.TASKS[:2]}
    rejected = Counter()
    used = set(blocked)
    for i, a, b in wmt_rows():
        keys = {C.text_key(a), C.text_key(b)}
        if i < 40000:  # Additional conservative exclusion of old verifier's entire source reserve.
            rejected['historical_first40000'] += 1
            continue
        if any(frequencies[k] != 1 for k in keys):
            rejected['bilingual_duplicate_group'] += 1
            continue
        if not a.strip() or not b.strip() or isolation_keys(a,b) & used:
            rejected['history_or_empty'] += 1
            continue
        task = C.TASKS[int(C.digest([a, b])[:8], 16) % 2]
        s, r = (a, b) if task == 'wmt19_en_zh' else (b, a)
        wmt_pools[task].append(source_row(task, i, s, r))
        used.update(keys)
    del frequencies
    summary = {}
    for task in C.TASKS:
        C.dump(out / 'preparation_status.json', {'phase': 'pool', 'task': task, 'at': C.utc()})
        if task.startswith('wmt19'):
            rows = wmt_pools.pop(task)
        else:
            rows = []
            iterator = coedit_rows() if task == 'coedit_gec' else giga_rows()
            for i, s, r in iterator:
                keys = {C.text_key(s), C.text_key(r)}
                if not s.strip() or not r.strip() or isolation_keys(s,r) & used:
                    continue
                used.update(keys)
                rows.append(source_row(task, i, s, r))
        d = out / task
        with C.timing(d, 'preparation'):
            artifacts, counts = write_pool(out, task, rows)
            artifacts['initial_memory.jsonl'] = C.file_hash(d / 'initial_memory.jsonl')
            profile = {'encoder': retrieval['encoder'], 'retrieval': retrieval['retrieval']}
            C.dump(d / 'retrieval.json', profile, immutable=True)
            artifacts['retrieval.json'] = C.file_hash(d / 'retrieval.json')
            protocol = {'schema': C.SCHEMA, 'task': task, 'input_fields': list(C.INPUT_FIELDS),
              'input_template_version': C.INPUT_VERSION, 'master_seed': C.MASTER_SEED,
              'current': 'initial_from_source_only_no_retrieval', 'states': 1, 'candidates': 4,
              'train_temperatures': [.1, .1, .7, .7], 'nontrain_temperatures': [.1]*4,
              'initial_temperature': .1, 'top_p': 1.0, 'initial_train_sources': 5000, 'shard_sources': 500,
              'target_unique_valid_changed': 100000, 'pool_counts': counts, 'epsilon': 0.0,
              'feedback': feedback if task.startswith('wmt19')
                          else {'metric': 'nltk_sentence_gleu' if task == 'coedit_gec' else 'rouge1_f1_stemmed'},
              'max_model_len': 4096, 'max_tokens': 128 if task == 'gigaword' else 1024,
              'laya_max_length': 1024, 'length_ratio': 1.5, 'length_allowance': 8,
              'test_policy': 'sealed_until_all_four_evaluation_constraints_frozen',
              'models': deployment['models'], 'artifact_hashes': artifacts, 'code_hashes': code_hashes,
              'dependency_versions':versions,'source_artifacts':source_artifacts,
              'created_at_utc': C.utc(), 'class_weighting': 'deferred_train_only_statistics',
              'global_source_target_overlap': 0}
            C.dump(d / 'protocol.json', protocol, immutable=True)
            C.dump(d / 'test_seal.json', {'state': 'sealed', 'schema': C.SCHEMA,
                         'protocol_sha256': C.file_hash(d / 'protocol.json')}, immutable=True)
            summary[task] = counts
        C.summarize_timing(d)
        del rows
    C.dump(out / 'prepared.json', {'schema': C.SCHEMA, 'tasks': summary, 'wmt_exclusions': dict(rejected),
                'cross_task_and_split_overlap': 0, 'history_overlap': 0, 'at': C.utc()}, immutable=True)
    C.dump(out / 'preparation_status.json', {'phase': 'complete', 'at': C.utc()})
    return summary
