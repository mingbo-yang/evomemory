"""Freeze a larger threshold-only fold from unused 0.1-temperature trajectories."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from acceptance_data import norm, valid_candidate
from legacy_laya_v1.laya_acceptance_common import digest, dump, get_tokenizer, read_jsonl, state_text
from semantic_label_common import OUT as MODEL

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/acceptance_data_v2'
OUT = ROOT / 'runs/laya_threshold_calibration_v2'
FOLDS = ('threshold_calibration', 'development')
BLOCKED = ('train', 'development', 'temperature_calibration', 'threshold_calibration', 'test')


def make():
    path = OUT / 'inputs.jsonl'
    if path.exists():
        protocol = json.loads((OUT / 'protocol.json').read_text())
        assert digest(path) == protocol['input_sha256']
        return read_jsonl(path)
    OUT.mkdir(parents=True, exist_ok=True)
    old = {norm(x['input']['source']) for fold in BLOCKED
           for x in read_jsonl(MODEL / 'inputs' / f'{fold}.jsonl')}
    # The old threshold fold was inspected while choosing the previous policy.
    # None of those 48 sources enters this replacement calibration fold.
    blocked = old.copy()
    manifest = ROOT / 'runs/laya_relaxed_flow_v1/test_manifest.jsonl'
    for line in manifest.read_text().splitlines():
        x = json.loads(line)
        blocked.update((norm(x['source']), norm(x['reference'])))
    tok = get_tokenizer()
    from laya.common import build_sequence
    import legacy_laya_v1.laya_acceptance_common as C
    prefix = max(len(build_sequence(tok, '', C.QUESTION, 1024, 256,
                                    option_order=list(order), state_ids=[])[0][:-1])
                 for order in [(0, 1, 2), (0, 2, 1), (1, 0, 2),
                               (1, 2, 0), (2, 0, 1), (2, 1, 0)])
    records = []
    seen = set()
    exclusions = Counter()
    used_sources = set()
    for fold in FOLDS:
        raw = read_jsonl(SOURCE / 'pairs' / f'{fold}.jsonl')
        source_refs = {json.loads(s)['sample_id']: json.loads(s)
                       for s in (SOURCE / f'{fold}_manifest.jsonl').read_text().splitlines()}
        for x in raw:
            src_key = norm(x['source'])
            reference = source_refs[x['sample_id']]['reference']
            if src_key in blocked or norm(reference) in blocked:
                exclusions['previously_used_source_or_reference'] += 1
                continue
            if '\ufffd' in x['source'] + reference:
                exclusions['encoding_damage'] += 1
                continue
            if x['temperature'] != .1 or not x['valid'] or x['identical']:
                exclusions['not_low_temp_valid_changed'] += 1
                continue
            generated = {'text': x['candidate'], 'finish_reason': 'stop'}
            if not valid_candidate(x['before'], generated)[0]:
                exclusions['basic_invalid'] += 1
                continue
            triple = (src_key, norm(x['before']), norm(x['candidate']))
            if triple in seen:
                exclusions['normalized_duplicate'] += 1
                continue
            inputs = {'source': x['source'], 'current': x['before'],
                      'candidate': x['candidate']}
            state_tokens = len(tok(state_text(inputs).replace(tok.mask_token, ' '),
                                   add_special_tokens=False)['input_ids'])
            if prefix + state_tokens + 1 > 1024:
                exclusions['laya_context_overflow'] += 1
                continue
            seen.add(triple)
            used_sources.add(src_key)
            ident = 'cal2-' + hashlib.sha256(json.dumps(triple,ensure_ascii=False).encode()).hexdigest()[:20]
            records.append({'id': ident, 'sample_id': x['sample_id'],
                            'state_index': x['state_index'], 'origin_fold': fold,
                            'input': inputs})
    records.sort(key=lambda x: hashlib.sha256(('calibration-v2|' + x['id']).encode()).hexdigest())
    assert len(records) >= 512, (len(records), dict(exclusions))
    assert len(used_sources) >= 300
    assert all(norm(x['input']['source']) not in blocked for x in records)
    assert len({x['id'] for x in records}) == len(records)
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records))
    protocol = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'role': 'new threshold calibration fold for frozen semantic and BLEU Laya models',
        'inputs': len(records), 'sources': len(used_sources),
        'origin_folds': dict(Counter(x['origin_fold'] for x in records)),
        'source_generation': 'Qwen3-8B, four candidates/state, all candidate slots at temperature 0.1; before is either a shared draft or a randomly chosen basic-valid round-1 candidate',
        'selection': 'all unique valid changed pairs; no quality-feedback or Laya filtering',
        'not_included': 'all previous new-model train, development, temperature calibration, threshold calibration, test, and 256-source rollout records, normalized source and reference matching',
        'reference_blind_inputs': True,
        'exclusions': dict(exclusions),
        'input_sha256': digest(path),
        'source_data_sha256': {f: digest(SOURCE / 'pairs' / f'{f}.jsonl') for f in FOLDS},
    }
    dump(OUT / 'protocol.json', protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
    return records


if __name__ == '__main__':
    make()
