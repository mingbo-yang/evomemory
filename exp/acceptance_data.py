"""Reference-blind, verifier-free two-state acceptance data collection.

Raw trajectories are immutable per source. References are used only in export,
after generation/state selection. No verifier or online memory is instantiated.
"""
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from core.manifest import _iter_wmt_train_pairs, SampleRef, read_manifest, sha256_text
from core.experience import load_experiences, save_experiences, render_experience_block

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs/acceptance_data_v2'
TASK = 'wmt19_en_zh'
SEED = 20260926
FOLDS = {'train': 10000, 'development': 400, 'temperature_calibration': 256,
         'threshold_calibration': 256, 'test': 400}

def norm(s):
    return ''.join(unicodedata.normalize('NFKC', s).casefold().split())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)

def seed_for(*parts):
    return int(hashlib.sha256('|'.join(map(str, (SEED, *parts))).encode()).hexdigest()[:8], 16) % (2**31)

def valid_candidate(before, generation):
    text = generation['text'].strip()
    if generation['finish_reason'] == 'context_length_exceeded':
        return False, 'context_length_exceeded'
    if not text:
        return False, 'empty'
    if generation['finish_reason'] != 'stop':
        return False, 'unfinished'
    if any(x in text.lower() for x in ('<think>', '</think>', '```')):
        return False, 'format'
    if not re.search(r'[\u3400-\u9fff]', text):
        return False, 'no_chinese'
    # Basic safety gate only; never select by feedback or by estimated quality.
    if before and len(norm(text)) > 1.5 * len(norm(before)) + 8:
        return False, 'length'
    return True, 'valid'

def choose_next(sample_id, before, candidates):
    eligible = [i for i, g in enumerate(candidates) if valid_candidate(before, g)[0]]
    return random.Random(seed_for(sample_id, 'state_selection')).choice(eligible) if eligible else None

def candidate_temperature(fold, slot, protocol):
    return protocol.get('train_candidate_temperatures', [.1]*4)[slot] if fold == 'train' else .1

def prepare():
    if (OUT / 'protocol.json').exists():
        p = json.loads((OUT / 'protocol.json').read_text())
        for name, expected in p['artifact_hashes'].items():
            assert digest(OUT / name) == expected, f'Changed frozen input: {name}'
        return p
    OUT.mkdir(parents=True, exist_ok=True)
    blocked = set()
    paths = list((ROOT / 'data/manifests').glob('*.jsonl'))
    paths += list((ROOT / 'runs/bert_onpolicy_20260925').glob('*_manifest.jsonl'))
    for path in paths:
        for r in read_manifest(path):
            blocked.update((norm(r.source), norm(r.reference)))
    history = json.loads((ROOT / 'runs/bert_feedback_study/source_split.json').read_text())
    blocked.update(norm(s) for split in history.values() for s in split)
    original = load_experiences(ROOT / 'experience/contrastive/qwen3-8b/wmt19_en_zh/initial.jsonl')
    main = set()
    for task in ('wmt19_en_zh', 'wmt19_zh_en'):
        for r in read_manifest(ROOT / f'data/manifests/{task}__test.jsonl'):
            main.update((norm(r.source), norm(r.reference)))
    library = [e for e in original if not {norm(e.source_input), norm(e.state_before), norm(e.state_after)} & main]
    library_keys = {norm(t) for e in original for t in (e.source_input, e.state_before, e.state_after)}
    blocked.update(library_keys)
    pool, seen = [], set()
    exclusions = 0
    for index, source, reference in _iter_wmt_train_pairs('en_zh'):
        if index >= 40000:
            break
        keys = {norm(source), norm(reference)}
        if '' in keys or keys & blocked or keys & seen:
            exclusions += 1
            continue
        seen.update(keys)
        pool.append((index, source, reference))
    random.Random(SEED).shuffle(pool)
    assert len(pool) >= sum(FOLDS.values())
    offset = 0
    for fold, n in FOLDS.items():
        records = [SampleRef(sample_id=f'{TASK}/acceptance_v1/{fold}/{i:06d}', task=TASK,
                    split='accumulation', row_index=i, source=s, reference=r, source_hash=sha256_text(s),
                    provenance='WMT train first40000; disjoint source/target; acceptance_v1')
                   for i, s, r in pool[offset:offset+n]]
        offset += n
        (OUT / f'{fold}_manifest.jsonl').write_text(''.join(json.dumps(asdict(r), ensure_ascii=False)+'\n' for r in records))
    save_experiences(library, OUT / 'initial_memory.jsonl')
    profile = json.loads((ROOT / 'configs/optimized_feedback_v1.json').read_text())
    dump(OUT / 'retrieval.json', {'encoder': profile['encoder'], 'retrieval': profile['retrieval']})
    tracked = list(OUT.glob('*_manifest.jsonl')) + [OUT/'initial_memory.jsonl', OUT/'retrieval.json']
    p = {'version': 2, 'task': TASK, 'model': 'qwen3-8b', 'gpu': 0, 'seed': SEED,
         'fold_source_reserves': FOLDS, 'train_target_unique_valid_pairs': 20000,
         'states': 2, 'candidates_per_state': 4, 'temperature': 0.1, 'top_p': 1.0,
         'train_candidate_temperatures': [.1, .1, .7, .7],
         'nontraining_candidate_temperatures': [.1, .1, .1, .1],
         'change_from_v1': 'user approved mixed-temperature train collection; regenerate in separate directory; same source partitions',
         'max_tokens': 1024, 'max_model_len': 4096,
         'selection': 'uniform among basic-valid candidate slots, including identical; no feedback/verifier',
         'memory': 'frozen identical initial library across all folds; no online writes',
         'feedback': 'existing Scorer.primary /100; applied only after full trajectory generation',
         'epsilon': None, 'labels': 'pending training-only delta inspection; no Better/Tie/Worse labels frozen yet',
         'test_policy': 'raw trajectories only; do not score/export test until model and thresholds frozen',
         'dedup': 'exact source,before,candidate within fold; keep all raw occurrences including temperature and slot; identical is valid',
         'context_overflow': 'record flagged empty output without generation; never truncate source silently',
         'normalization': 'isolation uses NFKC+casefold+remove whitespace for source AND reference',
         'pool_size': len(pool), 'excluded_pool_rows': exclusions,
         'initial_memory_before': len(original), 'initial_memory_after': len(library),
         'artifact_hashes': {p.name: digest(p) for p in tracked},
         'code_sha256_at_prepare': digest(Path(__file__))}
    dump(OUT / 'protocol.json', p)
    return p

def iter_raw(fold):
    for path in sorted((OUT / 'raw' / fold).glob('*.json')):
        yield json.loads(path.read_text())

def pair_rows(record):
    for state in record['states']:
        for slot, g in enumerate(state['candidates']):
            ok, reason = valid_candidate(state['before'], g)
            yield {'sample_id': record['sample_id'], 'source': record['source'],
                   'before': state['before'], 'candidate': g['text'], 'state_index': state['index'],
                   'slot': slot, 'temperature': g.get('temperature', .1), 'valid': ok, 'invalid_reason': reason, 'seed': g['seed'],
                   'identical': state['before'] == g['text']}

def unique_rows(records):
    seen = set()
    for record in records:
        for row in pair_rows(record):
            key = (row['source'], row['before'], row['candidate'])
            if key not in seen:
                seen.add(key)
                yield row

def audit_export(folds):
    protocol = json.loads((OUT/'protocol.json').read_text())
    manifests = {fold: read_manifest(OUT/f'{fold}_manifest.jsonl') for fold in FOLDS}
    keys = {fold: {norm(t) for r in refs for t in (r.source, r.reference)} for fold, refs in manifests.items()}
    for a in FOLDS:
        for b in FOLDS:
            if a != b:
                assert not keys[a] & keys[b], (a, b)
    library_keys = {norm(t) for e in load_experiences(OUT/'initial_memory.jsonl') for t in (e.source_input, e.state_before, e.state_after)}
    assert not library_keys & set.union(*keys.values())
    report = {'cross_fold_overlap': 0, 'memory_overlap': 0, 'epsilon': None, 'folds': {}}
    for fold in folds:
        records = list(iter_raw(fold))
        refs = {r.sample_id: r for r in manifests[fold]}
        invalid = Counter()
        raw_count = 0
        for rec in records:
            assert rec['source'] == refs[rec['sample_id']].source
            assert 'reference' not in rec
            assert bool(rec['states']) == valid_candidate('', rec['draft'])[0]
            for state in rec['states']:
                assert all(g.get('temperature', .1) == candidate_temperature(fold, slot, protocol)
                           for slot, g in enumerate(state['candidates']))
                assert all(g['seed'] == seed_for(rec['sample_id'], state['index'], slot)
                           for slot, g in enumerate(state['candidates']))
            if rec['states']:
                first = rec['states'][0]
                assert first['before'] == rec['draft']['text']
                selected = choose_next(rec['sample_id'], first['before'], first['candidates'])
                assert selected == rec['selected_slot']
                assert len(first['candidates']) == 4
                assert len(rec['states']) == (2 if selected is not None else 1)
                if selected is not None:
                    assert rec['states'][1]['before'] == first['candidates'][selected]['text']
                    assert len(rec['states'][1]['candidates']) == 4
            for row in pair_rows(rec):
                raw_count += 1
                if not row['valid']:
                    invalid[row['invalid_reason']] += 1
        rows = list(unique_rows(records))
        valid = [r for r in rows if r['valid']]
        summary = {'sources_completed': len(records), 'raw_pairs': raw_count,
                   'invalid_drafts': dict(Counter(valid_candidate('', r['draft'])[1] for r in records if not valid_candidate('', r['draft'])[0])),
                   'unique_pairs': len(rows), 'unique_valid_pairs': len(valid),
                   'unique_valid_changed_pairs': sum(not r['identical'] for r in valid),
                   'invalid_raw': dict(invalid), 'state_selection_audit': 'PASS'}
        if fold != 'test':
            from core.scoring import Scorer
            scorer = Scorer(TASK)
            bins = Counter()
            for row in rows:
                ref = refs[row['sample_id']].reference
                row['score_before'] = scorer.primary(ref, row['before']) / 100
                row['score_candidate'] = scorer.primary(ref, row['candidate']) / 100
                row['delta'] = row['score_candidate'] - row['score_before']
                if row['valid']:
                    d = abs(row['delta'])
                    bins['<.01' if d < .01 else '.01-.03' if d < .03 else '.03-.05' if d < .05 else '>=.05'] += 1
            dest = OUT/'pairs'/f'{fold}.jsonl'
            dest.parent.mkdir(exist_ok=True)
            dest.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
            # Only training feedback is summarized before model/calibration stages.
            if fold == 'train':
                summary['absolute_delta_bins_valid'] = dict(bins)
                summary['sign_valid'] = dict(Counter('positive' if r['delta']>1e-8 else 'negative' if r['delta']< -1e-8 else 'zero' for r in rows if r['valid']))
                review = []
                for low, high in [(0,.01),(.01,.03),(.03,.05),(.05,2)]:
                    eligible = [r for r in rows if r['valid'] and not r['identical'] and low <= abs(r['delta']) < high]
                    random.Random(SEED).shuffle(eligible)
                    for r in eligible[:20]:
                        review.append({**r, 'reference_for_human_review_only': refs[r['sample_id']].reference,
                                       'human_label': None, 'human_notes': ''})
                dump(OUT/'train_margin_review.json', review)
        report['folds'][fold] = summary
    dump(OUT/'audit_summary.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report

def collect(folds, limit, target):
    import torch
    torch.set_num_threads(4)
    from core import resolve_model_config
    from baseline_core.llm import LLMClient
    from baseline_core.tasks import get_adapter
    from baseline_core.types import TaskExample
    from core.pipeline import build_refine_prompt
    from core.optimized_pipeline import INSTRUCTIONS
    from core.hybrid_retrieval import LocalSentenceEncoder, HybridExperienceRetriever
    from vllm import SamplingParams
    p = prepare()
    client = LLMClient(resolve_model_config('qwen3-8b'), backend='vllm', gpu='0',
                       gpu_memory_utilization=.34, max_model_len=4096, enforce_eager=True)
    profile = json.loads((OUT/'retrieval.json').read_text())
    options = dict(profile['retrieval']); options.pop('method')
    library = load_experiences(OUT/'initial_memory.jsonl')
    retriever = HybridExperienceRetriever(library, LocalSentenceEncoder(profile['encoder'], 'cpu'), **options)
    by_id = {e.exp_id: e for e in library}
    adapter = get_adapter(TASK)

    def generate(requests):
        if not requests:
            return []
        texts = [client._chat_text(adapter.system_prompt(), r['prompt']) for r in requests]
        lengths = [client.count_tokens(t) for t in texts]
        indices = [i for i, n in enumerate(lengths) if n+p['max_tokens'] <= 4096]
        results = [{'text': '', 'raw_text': '', 'finish_reason': 'context_length_exceeded',
                    'seed': r['seed'], 'temperature': r.get('temperature', p['temperature']),
                    'input_tokens': lengths[i], 'output_tokens': 0} for i, r in enumerate(requests)]
        if not indices:
            return results
        selected = [requests[i] for i in indices]
        params = [SamplingParams(temperature=r.get('temperature', p['temperature']), top_p=p['top_p'],
                    max_tokens=p['max_tokens'], seed=r['seed'], stop_token_ids=client.stop_token_ids or None)
                  for r in selected]
        outputs = client.model.generate([texts[i] for i in indices], params, use_tqdm=False)
        assert len(outputs) == len(indices)
        for i, r, o in zip(indices, selected, outputs):
            results[i] = {'text': adapter.parse_output(o.outputs[0].text), 'raw_text': o.outputs[0].text,
                          'finish_reason': o.outputs[0].finish_reason, 'seed': r['seed'],
                          'temperature': r.get('temperature', p['temperature']),
                          'input_tokens': len(o.prompt_token_ids), 'output_tokens': len(o.outputs[0].token_ids)}
        return results

    def refine_batch(records, state_index, fold):
        reqs = []
        for rec in records:
            before = rec['draft']['text'] if state_index == 0 else rec['states'][0]['candidates'][rec['selected_slot']]['text']
            res = retriever.retrieve(rec['source'], before, alpha=.5, k=4, exclude_source=rec['source'])
            block = render_experience_block([by_id[i] for i in res.exp_ids], include_outcome=True,
                    count_tokens=client.count_tokens, max_units=4, contrastive=True, advice_mode='summary')
            example = TaskExample(index=0, source=rec['source'], reference='', task=TASK)
            prompt = build_refine_prompt(adapter, example, before, INSTRUCTIONS[TASK], experience_block=block, renderer='v2')
            rec['states'].append({'index': state_index, 'before': before, 'retrieved_ids': res.exp_ids,
                                  'prompt': prompt, 'candidates': []})
            for slot in range(4):
                reqs.append({'prompt': prompt, 'seed': seed_for(rec['sample_id'], state_index, slot),
                             'temperature': candidate_temperature(fold, slot, p)})
        outputs = generate(reqs)
        for i, rec in enumerate(records):
            rec['states'][-1]['candidates'] = outputs[4*i:4*i+4]

    for fold in folds:
        refs = read_manifest(OUT/f'{fold}_manifest.jsonl')
        if limit:
            refs = refs[:limit]
        dest = OUT/'raw'/fold; dest.mkdir(parents=True, exist_ok=True)
        pending = [r for r in refs if not (dest/f'{r.row_index:06d}.json').exists()]
        unique = sum(r['valid'] for r in unique_rows(iter_raw(fold)))
        for start in range(0, len(pending), 16):
            if fold == 'train' and unique >= target:
                break
            batch = pending[start:start+16]
            # Query references never enter a generation record or state-selection API.
            records = [{'sample_id': r.sample_id, 'source': r.source, 'states': [], 'selected_slot': None} for r in batch]
            requests = [{'prompt': adapter.initial_prompt(TaskExample(index=0, source=r.source, reference='', task=TASK)),
                         'seed': seed_for(r.sample_id, 'draft')} for r in batch]
            for rec, g in zip(records, generate(requests)):
                rec['draft'] = g
            eligible = [r for r in records if valid_candidate('', r['draft'])[0]]
            refine_batch(eligible, 0, fold)
            for rec in eligible:
                rec['selected_slot'] = choose_next(rec['sample_id'], rec['draft']['text'], rec['states'][0]['candidates'])
            refine_batch([r for r in eligible if r['selected_slot'] is not None], 1, fold)
            unique += sum(row['valid'] for row in unique_rows(records))
            for ref, rec in zip(batch, records):
                dump(dest/f'{ref.row_index:06d}.json', rec)
            dump(OUT/'progress.json', {'fold': fold, 'sources_completed': len(list(dest.glob('*.json'))),
                                      'unique_valid_pairs': unique, 'target_train': target})
            print(f'PROGRESS {fold} sources={len(list(dest.glob("*.json")))} unique_valid_pairs={unique}', flush=True)
        audit_export([fold])
    audit_export(folds)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['v1', 'v2'], default='v2')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--audit-only', action='store_true')
    ap.add_argument('--folds', nargs='+', choices=list(FOLDS), default=['train'])
    ap.add_argument('--limit', type=int)
    ap.add_argument('--target', type=int, default=20000)
    args = ap.parse_args()
    OUT = ROOT / f'runs/acceptance_data_{args.dataset}'
    if args.dataset == 'v1' and not (args.audit_only or args.prepare_only):
        ap.error('v1 pilot is preserved; use v2 for new generation')
    prepare()
    if args.audit_only:
        audit_export(args.folds)
    elif not args.prepare_only:
        collect(args.folds, args.limit, args.target)
