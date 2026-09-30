"""Optimized decisions, semantic ranking, feedback timing and runner integration."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import core
from core.bm25_fields import Experience
from core.hybrid_retrieval import HybridExperienceRetriever
from core.quality_feedback import QualityPolicy
from core.pipeline import RunConfig
from core.manifest import SampleRef, sha256_text
from core.optimized_pipeline import OptimizedBatchedPipeline
from baseline_core.types import Generation


def exp(key, source='source', state='old'):
    return Experience(key, 'coedit_gec', 'qwen3-8b', source, state, 'correct',
                      'Fix the verb', '', 'better', '', '', True, 1, 'test', 'helped')


class Encoder:
    def encode(self, texts):
        import numpy as np
        vectors = {'query': [1., 0.], 'semantic': [1., 0.], 'unrelated': [0., 1.],
                   'source': [1., 0.], 'old': [1., 0.], 'He go.': [1., 0.]}
        return np.array([vectors.get(t, [0., 1.]) for t in texts])


class Quality:
    def __init__(self):
        self.inputs = []
    def score_pairs(self, pairs):
        self.inputs.extend(pairs)
        return [0.95 if candidate == 'He goes.' else 0.2 for _, candidate in pairs]


class Client:
    backend = 'fake'
    def count_tokens(self, text):
        return len(text.split())
    def generate(self, **kwargs):
        assert kwargs['call_type'] == 'refine', kwargs
        return Generation(text='He goes.', input_tokens=20, output_tokens=3,
                          latency=0.01, seed=kwargs['seed'], call_type='refine', start_time=0., end_time=.01)


def settings():
    return dict(version='test', policy=dict(stop_threshold=0.9, min_gain=0.01,
                soft_length_ratio=1.02, hard_length_ratio=1.5,
                expansion_gain=0.05, length_allowance=8))


def ref(i=0, reference='He goes.'):
    return SampleRef(f'coedit_gec/test/{i:06d}', 'coedit_gec', 'test', i,
                     'He go.', reference, sha256_text('He go.'), 'test')


class OptimizedTest(unittest.TestCase):
    def test_length_gate_allows_supported_expansion_but_blocks_runaway(self):
        p = QualityPolicy(0.9, length_allowance=0)
        self.assertEqual(p.accept('a'*100, 'b'*110, .2, .23)[1], 'expansion_without_strong_gain')
        self.assertTrue(p.accept('a'*100, 'b'*110, .2, .3)[0])
        self.assertEqual(p.accept('a'*100, 'b'*151, .2, .9)[1], 'hard_length_limit')
        self.assertFalse(p.accept('abc', 'abd', .5, .4)[0])
        self.assertFalse(p.accept('abc', '', .2, .9)[0])
        self.assertFalse(p.accept('abc', 'abc', .2, .9)[0])
        with self.assertRaises(ValueError):
            p.accept('a', 'b', float('nan'), .9)

    def test_semantic_match_self_exclusion_and_abstention(self):
        r = HybridExperienceRetriever([exp('own','query','query'), exp('related','semantic','semantic'),
                                       exp('other','unrelated','unrelated')], Encoder(), semantic_weight=1)
        got = r.retrieve('query','query', k=1, exclude_source='query')
        self.assertEqual(got.exp_ids, ['related'])
        empty = HybridExperienceRetriever([exp('other','unrelated','unrelated')], Encoder())
        self.assertEqual(empty.retrieve('query','query').exp_ids, [])
        rebuilt = empty.rebuild([exp('new','semantic','semantic')])
        self.assertIs(rebuilt.encoder, empty.encoder)
        self.assertEqual(rebuilt.retrieve('query','query').exp_ids, ['new'])

    def test_references_do_not_affect_decisions_only_delayed_admission(self):
        for gold, admitted in [('He goes.', 1), ('He go.', 0)]:
            with self.subTest(gold=gold):
                quality = Quality()
                cfg = RunConfig('coedit_gec','qwen3-8b', draft_source='stored', renderer='v2',
                                retrieval_excludes_own_source=True, optimization=settings())
                library = [exp('initial')]
                pipe = OptimizedBatchedPipeline(cfg, HybridExperienceRetriever(library, Encoder()),
                    library, Client(), batch_size=1, online=True, quality_evaluator=quality)
                pipe.drafts = SimpleNamespace(get=lambda i: ('He go.', True))
                seen = []
                traces = pipe.run([ref(reference=gold)], sink=lambda *args: seen.append(len(pipe.library)))
                self.assertEqual(traces[0].final_output, 'He goes.')
                self.assertTrue(traces[0].rounds[0].accepted)
                self.assertEqual(traces[0].rounds[1].controller_action, 'STOP')
                self.assertEqual(seen, [1])  # sink precedes feedback commit
                self.assertEqual(len(pipe.library), 1+admitted)
                self.assertIsInstance(pipe.retriever, HybridExperienceRetriever)
                self.assertEqual(traces[0].cost['n_calls'], 1)
                self.assertEqual(traces[0].cost['quality_pairs'], 3)
                self.assertEqual(quality.inputs, [('He go.','He go.'), ('He go.','He goes.'), ('He go.','He goes.')])

    def test_initial_generation_cost_and_cached_draft_provenance(self):
        class InitialClient(Client):
            def generate(self, **kwargs):
                assert kwargs['call_type'] == 'initial'
                return Generation(text='He goes.', input_tokens=7, output_tokens=3,
                                  latency=.02, seed=kwargs['seed'], call_type='initial',
                                  start_time=0., end_time=.02)
        with tempfile.TemporaryDirectory() as directory:
            for expected_calls in (1, 0):
                cfg = RunConfig('coedit_gec', 'qwen3-8b', draft_source='cached',
                                draft_cache=str(Path(directory)/'drafts.jsonl'),
                                renderer='v2', optimization=settings())
                pipe = OptimizedBatchedPipeline(cfg, None, [], InitialClient(),
                                               quality_evaluator=Quality(), batch_size=1)
                trace = pipe.run([ref()])[0]
                self.assertEqual(trace.cost['n_calls'], expected_calls)
                self.assertEqual(trace.cost['total_tokens'], 10*expected_calls)
                self.assertEqual(trace.draft_from_cache, expected_calls == 0)
                self.assertEqual(trace.rounds[0].controller_action, 'STOP')

    def test_config_identity_and_runner_outputs(self):
        import run_experiment as runner
        cfg = RunConfig('coedit_gec','qwen3-8b')
        opt = RunConfig('coedit_gec','qwen3-8b',optimization=settings())
        self.assertNotEqual(cfg.config_hash, opt.config_hash)
        initial = [exp('initial')]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner,'read_manifest',return_value=[ref()]), \
                 patch.object(runner,'build_library',return_value=initial), \
                 patch('core.optimization_config.build_components', side_effect=lambda *a: (Quality(),HybridExperienceRetriever(initial,Encoder()))), \
                 patch('core.pipeline.StoredDraftSource', return_value=SimpleNamespace(get=lambda i:('He go.', True))):
                for arm in ('full_static','full_online'):
                    result = runner.run_model_task(arm=arm,model='qwen3-8b',task='coedit_gec',seed=42,
                        gpu='0',split='test',tag='',limit=1,max_rounds=3,alpha=.5,k=4,
                        draft_source='stored',draft_cache='',batch_size=1,llm=Client(),out_dir=Path(directory),
                        optimization=settings(),retrieval_excludes_own_source=True)
                    self.assertEqual(result['n'], 1)
                    self.assertEqual(result['cost']['n_calls'], 1)
            files = list(Path(directory).rglob('*.jsonl'))
            self.assertEqual(len(files), 2)
            self.assertEqual(files[0].parent.name, files[1].parent.name) # matched profile tag across arms
            memories = list(Path(directory).rglob('online_experience.json'))
            self.assertEqual(len(memories), 1)
            self.assertEqual(len(json.loads(memories[0].read_text())), 2)


if __name__ == '__main__':
    unittest.main()
