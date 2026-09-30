import json
import hashlib
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from laya_relaxed_policy import select_candidate
from laya_relaxed_flow import positive_memories, Flow, Verifier
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
        self.assertEqual(select_candidate(candidates), 1)
        with self.assertRaises(ValueError):
            select_candidate(candidates, {'p_better': .275})
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

    def test_full_flow_delays_feedback_and_only_online_changes_memory(self):
        events = []
        class Generator:
            adapter = object()
            client = SimpleNamespace(count_tokens=lambda text: len(text))
            def generate(self, requests):
                events.append('generate')
                return [{'text': text, 'finish_reason': 'stop'} for text in ['好修改', '坏修改', '初稿', '']]
        class Verifier:
            def predict(self, rows):
                events.append('classify')
                for r in rows:
                    assert set(r['input']) == {'source', 'current', 'candidate'}
                    assert r['input']['current'] != r['input']['candidate']
                    assert 'SECRET' not in str(r)
                return {r['id']: {
                    'decision': 'Accept' if r['input']['source'] == 'source1' and r['input']['current'] == '初稿' and r['input']['candidate'] == '坏修改' else 'Reject',
                    'probabilities': {'Accept': .9, 'Reject': .1}} for r in rows}
        class DelayedFeedback(Feedback):
            def score_pairs(self, pairs):
                assert events[-1] == 'classify'
                events.append('feedback')
                return super().score_pairs(pairs)
        class Retriever:
            def __init__(self, library, encoder, **opts):
                self.library = list(library)
            def retrieve(self, source, before, **kwargs):
                return SimpleNamespace(exp_ids=[e.exp_id for e in self.library if e.source_input != source])
            def rebuild(self, library):
                events.append('rebuild')
                return Retriever(library, None)
        protocol = {'batch_size': 1, 'max_rounds': 3, 'candidates_per_round': 4, 'epsilon_memory': 0, 'policy': {}}
        refs = [SimpleNamespace(sample_id=str(i), source='source' + str(i), reference='SECRET' + str(i)) for i in range(2)]
        drafts = [{'text': '初稿', 'finish_reason': 'stop'} for _ in refs]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            save_experiences([], out / 'initial_memory.jsonl')
            (out / 'retrieval.json').write_text(json.dumps({'encoder': 'unused', 'retrieval': {}}))
            flow = Flow(protocol, Generator(), Verifier(), DelayedFeedback(), out, encoder=object(), retriever_class=Retriever)
            static = flow.run('full_static', refs, drafts)
            online = flow.run('full_online', refs, drafts, online=True)
            self.assertEqual([r['final'] for r in online], ['初稿', '坏修改'])
            for records in [static, online]:
                no_ops = [c for tr in records for rd in tr['rounds'] for c in rd['candidates'] if not c['changed']]
                self.assertTrue(no_ops)
                for candidate in no_ops:
                    self.assertEqual(candidate['decision'], 'Reject')
                    self.assertEqual(candidate['status'], 'no_op')
                    self.assertIsNone(candidate['probabilities'])
            self.assertEqual(online[1]['rounds'][1]['before'], '坏修改')
            self.assertEqual(len(online[0]['rounds']), 1)  # Reject stops revision.
            self.assertEqual(static[1]['bank_size_at_start'], 0)
            self.assertEqual(online[1]['bank_size_at_start'], 1)
            self.assertEqual(online[1]['rounds'][0]['online_retrieved'], 1)
            self.assertEqual(events.count('classify'), 6)
            self.assertEqual(events.count('feedback'), 4)
            self.assertEqual(events.count('rebuild'), 2)
            self.assertNotIn('SECRET', (out / 'full_online/final_memory.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
