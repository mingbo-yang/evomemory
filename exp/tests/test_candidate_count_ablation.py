"""Behavioral checks for the candidate-count intervention and import isolation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ablations.laya_candidate_count import Flow
from laya_relaxed_flow import stable_seed
from core.experience import save_experiences


class CandidateCountTests(unittest.TestCase):
    def test_runtime_does_not_import_training_jobs_or_select_gpu(self):
        env = dict(os.environ)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        code = "import os,sys; import laya_relaxed_flow; assert 'acceptance_data' not in sys.modules; assert 'prepare_acceptance_training' not in sys.modules; assert 'CUDA_VISIBLE_DEVICES' not in os.environ; assert 'ablations.laya_candidate_count' not in sys.modules; assert 'core.optimized_pipeline' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], env=env, check=True, capture_output=True)

    def test_candidate_budget_is_the_only_intervention(self):
        class Generator:
            adapter = object()
            client = SimpleNamespace(count_tokens=len)
            def __init__(self):
                self.requests = []
            def generate(self, requests):
                self.requests.append(requests)
                return [{'text': ['坏修改', '好修改', '初稿', '持平改写'][i], 'finish_reason': 'stop'} for i, _ in enumerate(requests)]
        class Verifier:
            def predict(self, rows):
                return {r['id']: {'decision': 'Accept' if r['input']['current'] == '初稿' and r['input']['candidate'] == '好修改' else 'Reject'} for r in rows}
        class Feedback:
            metric = 'comet'
            def score_pairs(self, rows):
                score = {'初稿': .5, '坏修改': .4, '好修改': .6, '持平改写': .5}
                return [{'q_current': score[r['current']], 'q_candidate': score[r['candidate']]} for r in rows]
        class Retriever:
            def __init__(self, *a, **kw):
                pass
            def retrieve(self, *a, **kw):
                return SimpleNamespace(exp_ids=[])
        refs = [SimpleNamespace(sample_id='s', source='source', reference='secret')]
        drafts = [{'text': '初稿', 'finish_reason': 'stop'}]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'retrieval.json').write_text(json.dumps({'encoder': '', 'retrieval': {}}))
            save_experiences([], out / 'initial_memory.jsonl')
            generators, records = {}, {}
            for k in (1, 4):
                generators[k] = Generator()
                protocol = dict(batch_size=1, max_rounds=3, candidates_per_round=k, epsilon_memory=0, policy={})
                flow = Flow(protocol, generators[k], Verifier(), Feedback(), out, encoder=object(), retriever_class=Retriever)
                with patch('core.pipeline.build_refine_prompt', side_effect=lambda adapter, ex, before, *a, **kw: ex.source + before):
                    records[k] = flow.run('k' + str(k), refs, drafts)[0]
            self.assertEqual(records[1]['final'], '初稿')
            self.assertEqual(len(records[1]['rounds']), 1)
            self.assertEqual(records[4]['final'], '好修改')
            self.assertEqual(len(records[4]['rounds']), 2)
            self.assertEqual(records[4]['rounds'][1]['before'], '好修改')
            self.assertEqual([len(r) for r in generators[1].requests], [1])
            self.assertEqual([len(r) for r in generators[4].requests], [4, 4])
            self.assertEqual(generators[1].requests[0][0], generators[4].requests[0][0])
            self.assertEqual(generators[1].requests[0][0]['seed'], stable_seed('s', 1, 0))
            self.assertEqual(records[4]['rounds'][0]['selected_slot'], 1)


if __name__ == '__main__':
    unittest.main()
