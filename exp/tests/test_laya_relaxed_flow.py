import json
import hashlib
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from laya_relaxed_policy import select_candidate
from laya_relaxed_flow import positive_memories, Flow, Verifier, parse_args, prepare
import laya_acceptance_common as C
from core.experience import save_experiences


class Feedback:
    metric = 'comet'

    def score_pairs(self, pairs):
        assert all(r['current'] != r['candidate'] for r in pairs)
        scores = {'初稿': .5, '好修改': .6, '坏修改': .4, '持平改写': .5, '微小改善': .50001}
        return [{'q_current': scores[r['current']], 'q_candidate': scores[r['candidate']]} for r in pairs]


class RelaxedFlowTests(unittest.TestCase):
    def test_classification_alone_controls_action(self):
        candidates = [{'valid': True, 'changed': True, 'decision': 'Reject', 'probabilities': {'Accept': .99}},
                      {'valid': True, 'changed': True, 'decision': 'Accept', 'probabilities': {'Accept': .01}},
                      {'valid': True, 'changed': True, 'decision': 'Accept', 'probabilities': {'Accept': 1.}}]
        self.assertIsNone(select_candidate([candidates[0]]))
        self.assertEqual(select_candidate([candidates[1]]), 0)
        with self.assertRaises(ValueError):
            select_candidate(candidates)
        with self.assertRaises(ValueError):
            select_candidate([candidates[1]], {'p_better': .275})
        self.assertIsNone(select_candidate([dict(candidates[1], valid=False)]))
        self.assertIsNone(select_candidate([dict(candidates[1], changed=False)]))
        with self.assertRaises(ValueError):
            select_candidate([dict(candidates[1], decision='Better')])

    def test_verifier_rejects_no_ops_without_tokenizing_or_model_calls(self):
        verifier = Verifier.__new__(Verifier)
        verifier.tok = Mock(side_effect=AssertionError('No-op must bypass tokenization'))
        verifier.scored = 0
        inputs = {'source': 'source', 'current': '相同', 'candidate': '相同'}
        key = hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        verifier.cache = {key: {'decision': 'Accept'}}  # Rule wins even over a stale cache.
        with patch.object(C, 'infer', side_effect=AssertionError('No-op must bypass Laya')):
            result = verifier.predict([{'id': 'a', 'input': inputs}, {'id': 'b', 'input': inputs}])
        self.assertEqual(set(result), {'a', 'b'})
        for decision in result.values():
            self.assertEqual(decision['decision'], 'Reject')
            self.assertEqual(decision['status'], 'no_op')
            self.assertIsNone(decision['probabilities'])
        verifier.tok.assert_not_called()
        self.assertEqual(verifier.scored, 0)

    def test_mixed_batch_scores_only_changed_candidates(self):
        verifier = Verifier.__new__(Verifier)
        verifier.tok = Mock(return_value={'input_ids': [1, 2]})
        verifier.tok.mask_token = '[MASK]'
        verifier.overhead = 0
        verifier.cache, verifier.scored = {}, 0
        verifier.model = object()
        verifier.diagnostic_temperature = 1.
        rows = [{'id': str(i), 'input': {'source': 's', 'current': '初稿', 'candidate': text}}
                for i, text in enumerate(['初稿', '改写', '初稿', '改写'])]
        with patch.object(C, 'EncodedPairs', side_effect=lambda rows, tok: SimpleNamespace(rows=rows)):
            with patch.object(C, 'infer', return_value=np.array([[100., -100.]])) as infer:
                result = verifier.predict(rows)
        scored = infer.call_args.args[1].rows
        self.assertEqual([r['input']['candidate'] for r in scored], ['改写'])
        self.assertEqual([result[str(i)]['decision'] for i in range(4)], ['Reject', 'Accept', 'Reject', 'Accept'])
        self.assertEqual(verifier.scored, 1)

    def test_rejected_positive_enters_memory_but_accepted_negative_does_not(self):
        trace = {'source': 'source', 'initial': 'NOT_THE_BEFORE', 'rounds': [{'round': 1, 'before': '初稿', 'candidates': [
            {'text': '好修改', 'valid': True, 'changed': True, 'decision': 'Reject'},
            {'text': '坏修改', 'valid': True, 'changed': True, 'decision': 'Accept'},
            {'text': '持平改写', 'valid': True, 'changed': True, 'decision': 'Accept'},
            {'text': '微小改善', 'valid': True, 'changed': True, 'decision': 'Reject'},
            {'text': '好修改', 'valid': True, 'changed': True, 'decision': 'Reject'}]}]}
        existing = set()
        result = positive_memories(trace, 'SECRET_REFERENCE', Feedback(), existing, 0)
        self.assertEqual([e.state_after for e in result], ['好修改', '微小改善'])
        self.assertTrue(all(e.state_before == '初稿' for e in result))
        self.assertNotIn('SECRET_REFERENCE', str([e.to_dict() for e in result]))
        self.assertFalse(positive_memories(trace, 'SECRET_REFERENCE', Feedback(), existing, 0))

    def test_formal_entry_rejects_multicandidate_bypass_and_missing_checkpoint(self):
        import contextlib
        import io
        base = ['--manifest', 'manifest.jsonl', '--checkpoint', 'epoch_20']
        self.assertEqual(parse_args(base).candidates_per_round, 1)
        for extra in (['--candidates-per-round', '4'], ['--arms', 'unfiltered_static']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(base + extra)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--manifest', 'manifest.jsonl'])
        for value in (4, 0, True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Flow({'candidates_per_round': value}, None, None, None, 'unused')
        f = Flow.__new__(Flow)
        f.p = {'candidates_per_round': 1}
        for name, online in [('unfiltered_static', False), ('full_online', False), ('full_static', True)]:
            with self.subTest(name=name, online=online), self.assertRaises(ValueError):
                f.run(name, [], [], online=online)
        # Config-level multi-candidate injection is rejected before models/data/GPU.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'cfg.json'
            cfg.write_text(json.dumps({'candidates_per_round': 4}))
            with self.assertRaises(ValueError):
                prepare(SimpleNamespace(config=cfg, arms=['full_static'], candidates_per_round=1), None)

    def test_full_flow_delays_feedback_and_only_online_changes_memory(self):
        events = []
        class Generator:
            adapter = object()
            client = SimpleNamespace(count_tokens=lambda text: len(text))
            def generate(self, requests):
                events.append(('generate', [r['prompt'] for r in requests]))
                choices = {'source0|初稿': '好修改', 'source1|初稿': '坏修改',
                           'source1|坏修改': '坏修改', 'source2|初稿': '持平改写'}
                return [{'text': choices[r['prompt']], 'finish_reason': 'stop'} for r in requests]
        class Verifier:
            def predict(self, rows):
                events.append(('classify', [r['id'] for r in rows]))
                for r in rows:
                    assert set(r['input']) == {'source', 'current', 'candidate'}
                    assert r['input']['current'] != r['input']['candidate']
                    assert 'SECRET' not in str(r)
                return {r['id']: {
                    'decision': 'Accept' if r['input']['source'] == 'source1' else 'Reject',
                    'probabilities': {'Accept': .9, 'Reject': .1}} for r in rows}
        class DelayedFeedback(Feedback):
            def score_pairs(self, pairs):
                events.append(('feedback', [r['source'] for r in pairs]))
                return super().score_pairs(pairs)
        class Retriever:
            def __init__(self, library, encoder, **opts):
                self.library = list(library)
            def retrieve(self, source, before, **kwargs):
                return SimpleNamespace(exp_ids=[e.exp_id for e in self.library if e.source_input != source])
            def rebuild(self, library):
                events.append(('rebuild', len(library)))
                return Retriever(library, None)
        protocol = {'batch_size': 2, 'max_rounds': 3, 'candidates_per_round': 1, 'epsilon_memory': 0, 'policy': {}}
        refs = [SimpleNamespace(sample_id=str(i), source='source' + str(i), reference='SECRET' + str(i)) for i in range(3)]
        drafts = [{'text': '初稿', 'finish_reason': 'stop'} for _ in refs]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            save_experiences([], out / 'initial_memory.jsonl')
            (out / 'retrieval.json').write_text(json.dumps({'encoder': 'unused', 'retrieval': {}}))
            flow = Flow(protocol, Generator(), Verifier(), DelayedFeedback(), out, encoder=object(), retriever_class=Retriever)
            with patch('core.pipeline.build_refine_prompt', side_effect=lambda adapter, ex, before, *a, **kw: ex.source + '|' + before):
                static = flow.run('full_static', refs, drafts)
                static_events = list(events)
                online = flow.run('full_online', refs, drafts, online=True)
            self.assertEqual([r['final'] for r in online], ['初稿', '坏修改', '初稿'])
            for records in [static, online]:
                self.assertTrue(all(len(rd['candidates']) == 1 for r in records for rd in r['rounds']))
                candidate = records[1]['rounds'][1]['candidates'][0]
                self.assertEqual(candidate['decision'], 'Reject')
                self.assertEqual(candidate['status'], 'no_op')
                self.assertIsNone(candidate['probabilities'])
                self.assertEqual(records[1]['rounds'][1]['before'], '坏修改')
                self.assertEqual(len(records[0]['rounds']), 1)  # Reject stops this source immediately.
            self.assertEqual(static[2]['bank_size_at_start'], 0)
            self.assertEqual(online[1]['bank_size_at_start'], 0)  # Same batch cannot see feedback.
            self.assertEqual(online[2]['bank_size_at_start'], 1)
            self.assertEqual(online[2]['rounds'][0]['online_retrieved'], 1)
            first_feedback = next(i for i, (event, _) in enumerate(static_events) if event == 'feedback')
            self.assertEqual([v for event, v in static_events[:first_feedback] if event == 'generate'],
                             [['source0|初稿', 'source1|初稿'], ['source1|坏修改']])
            self.assertEqual(sum(e == 'classify' for e, _ in events), 4)
            self.assertEqual(sum(e == 'feedback' for e, _ in events), 4)
            self.assertEqual(sum(e == 'rebuild' for e, _ in events), 1)
            self.assertNotIn('SECRET', (out / 'full_online/final_memory.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
