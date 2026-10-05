"""Replay saved real K=1 trajectories through the cleaned formal Flow on CPU.

Recompute retrieval and prompts, but only reuse previously generated outputs,
Laya decisions and COMET feedback. A cache miss fails: this is a semantic
regression check, not new model evaluation or fresh quality evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import laya_acceptance_common as C
from core.comet_feedback import validated_scores
from core.manifest import read_manifest
from laya_relaxed_flow import Flow
from baseline_core.tasks import get_adapter


def replay(source, output):
    from transformers import AutoTokenizer
    protocol = json.loads((source / 'protocol.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    for name in ('initial_memory.jsonl', 'retrieval.json', 'test_manifest.jsonl'):
        shutil.copy2(source / name, output / name)
    tok = AutoTokenizer.from_pretrained(protocol['generator_path'], trust_remote_code=True, local_files_only=True)

    class CachedGenerator:
        adapter = get_adapter('wmt19_en_zh')
        client = SimpleNamespace(count_tokens=lambda text: len(tok.encode(str(text), add_special_tokens=False)))
        calls = 0
        def generate(self, requests):
            result = []
            for request in requests:
                payload = {'prompt': request['prompt'], 'system': self.adapter.system_prompt(), 'seed': request['seed'],
                           'temperature': .1, 'top_p': 1., 'max_tokens': 1024, 'model': protocol['generator_path']}
                key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                path = source / 'generation_cache' / (key + '.json')
                if not path.exists():
                    raise AssertionError('Prompt/seed changed or unseen request: ' + key)
                result.append(json.loads(path.read_text()))
                self.calls += 1
            return result

    class CachedVerifier:
        def __init__(self, records):
            self.decisions = {}
            for row in records:
                for rd in row['rounds']:
                    for c in rd['candidates']:
                        if c['valid'] and c['changed']:
                            key = (row['source'], rd['before'], c['text'])
                            value = {k: c[k] for k in ('decision', 'status', 'logits', 'probabilities') if k in c}
                            assert self.decisions.get(key, value) == value
                            self.decisions[key] = value
        def predict(self, rows):
            return {row['id']: self.decisions[tuple(row['input'][f] for f in C.INPUT_FIELDS)] for row in rows}

    class CachedFeedback:
        metric = 'comet'
        def __init__(self):
            self.cache = json.loads((source / 'comet_score_cache.json').read_text())
        def score_pairs(self, pairs):
            scores = []
            for row in pairs:
                values = {}
                for field in ('current', 'candidate'):
                    obj = {'src': row['source'], 'mt': row[field], 'ref': row['reference']}
                    key = hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    values['q_' + field] = self.cache[key]
                scores.append(values)
            return validated_scores(pairs, scores)

    refs = read_manifest(output / 'test_manifest.jsonl')
    gen, feedback = CachedGenerator(), CachedFeedback()
    results, encoder, drafts = {}, None, None
    for old, name in (('k1_static', 'full_static'), ('k1_online', 'full_online')):
        expected = json.loads((source / old / 'results.json').read_text())
        flow = Flow(dict(protocol, candidates_per_round=1, policy={}), gen, CachedVerifier(expected), feedback,
                    output, encoder=encoder)
        encoder = flow.encoder
        if drafts is None:
            drafts = flow.drafts(refs)
            assert drafts == json.loads((source / 'shared_drafts.json').read_text())
        actual = flow.run(name, refs, drafts, online=name == 'full_online')
        assert actual == expected, f'Trajectory mismatch: {name}'
        assert (output / name / 'final_memory.jsonl').read_bytes() == (source / old / 'final_memory.jsonl').read_bytes()
        for path in (source / old).glob('feedback_*.json'):
            assert json.loads(path.read_text()) == json.loads((output / name / path.name).read_text())
        results[name] = {'sources': len(actual), 'rounds': sum(len(r['rounds']) for r in actual),
                         'trajectories_identical': True, 'final_memory_identical': True, 'delayed_feedback_identical': True}
    result = {'kind': 'CPU cached replay, not fresh model evaluation', 'source': str(source),
              'generator_requests_replayed': gen.calls, 'arms': results}
    C.dump(output / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=C.ROOT / 'runs/laya_candidate_ablation_v1')
    p.add_argument('--output', type=Path, default=C.ROOT / 'runs/laya_single_candidate_cleanup_v1/replay')
    args = p.parse_args()
    replay(args.source, args.output)
