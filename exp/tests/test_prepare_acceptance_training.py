import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from prepare_acceptance_training import classify, convert, swap_training_item, map_legacy_label, build_dataset, FOLDS
import laya_acceptance_common as C


class Feedback:
    metric = 'comet'
    metadata = {'metric': 'comet', 'checkpoint_sha256': 'test-fixture'}

    def __init__(self):
        self.calls = 0
        self.scored_candidates = []

    def score_pairs(self, rows):
        self.calls += 1
        assert all(r['current'] != r['candidate'] for r in rows)
        self.scored_candidates.extend(r['candidate'] for r in rows)
        quality = {'before': .5, 'good': .6, 'bad': .4, 'equal': .5}
        return [{'q_current': quality[r['current']], 'q_candidate': quality[r['candidate']]} for r in rows]


class DatasetSafety(unittest.TestCase):
    def test_zero_margin_and_legacy_meaning(self):
        for delta, label in [(0., 'Reject'), (-.1, 'Reject'), (.00001, 'Accept'), (.2, 'Accept')]:
            self.assertEqual(classify(delta, 0), label)
        self.assertEqual([map_legacy_label(x) for x in ['Better', 'Tie', 'Worse']], ['Accept', 'Reject', 'Reject'])
        with self.assertRaises(ValueError):
            map_legacy_label('Uncertain')
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                classify(value, 0)

    def test_feedback_and_reference_do_not_enter_features(self):
        row = {'source': 'source', 'before': 'before', 'candidate': 'good', 'delta': .1, 'reference': 'SECRET'}
        a = convert(row, 0)
        b = convert(dict(row, delta=-.1, reference='OTHER'), 0)
        self.assertEqual(a['input'], b['input'])
        self.assertNotEqual(a['label'], b['label'])
        self.assertEqual(set(a), {'id', 'input', 'label'})
        self.assertEqual(set(a['input']), set(C.INPUT_FIELDS))

    def test_reject_cannot_be_naively_inverted(self):
        row = {'id': 'x', 'input': {'source': 's', 'current': 'a', 'candidate': 'b'}, 'label': 'Reject'}
        with self.assertRaises(ValueError):
            swap_training_item(row)
        self.assertEqual(swap_training_item(row, delta=0, epsilon=0)['label'], 'Reject')
        self.assertEqual(swap_training_item(row, delta=-.1, epsilon=0)['label'], 'Accept')

    def test_dataset_recomputes_feedback_and_keeps_audit_separate(self):
        rows = [{'sample_id': 'a', 'source': 's1', 'before': 'before', 'candidate': 'good', 'valid': True, 'delta': -999},
                {'sample_id': 'a', 'source': 's1', 'before': 'before', 'candidate': 'bad', 'valid': True, 'delta': 999}]
        refs = {'train': {'a': {'source': 's1', 'reference': 'SECRET_REFERENCE'}}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data'
            meta = build_dataset({'train': rows}, refs, Feedback(), path, 0)
            data, delta, _ = C.load_fold('train', path)
            self.assertEqual([x['label'] for x in data], ['Accept', 'Reject'])
            self.assertAlmostEqual(delta[0], .1)
            self.assertNotIn('SECRET_REFERENCE', (path / 'train.jsonl').read_text())
            self.assertNotIn('q_candidate', (path / 'train.jsonl').read_text())
            self.assertEqual(meta['label_feedback']['epsilon'], 0)
            with self.assertRaises(FileExistsError):
                build_dataset({'train': rows}, refs, Feedback(), path, 0)
            with (path / 'train.jsonl').open('a') as f:
                f.write('\n')
            with self.assertRaises(ValueError):
                C.load_fold('train', path)

    def test_no_ops_are_audit_only_but_changed_equal_scores_remain_reject_in_every_fold(self):
        for fold in (*FOLDS, 'test'):
            with self.subTest(fold=fold), tempfile.TemporaryDirectory() as temp:
                rows = [{'sample_id': 'a', 'source': 's1', 'before': 'before', 'candidate': text,
                         'valid': True, 'delta': 999} for text in ['good', 'bad', 'equal', 'before', 'before']]
                refs = {fold: {'a': {'source': 's1', 'reference': 'SECRET_REFERENCE'}}}
                f = Feedback()
                path = Path(temp) / 'data'
                checkpoint = Path(temp) / 'checkpoint'
                checkpoint.mkdir()
                (checkpoint / 'model.safetensors').write_bytes(b'frozen-test-fixture')
                cfg = {'label_feedback': dict(f.metadata, epsilon=0), 'seen_source_hashes': ['other_source']}
                with patch.object(C, 'validate_checkpoint', return_value=cfg):
                    meta = build_dataset({fold: rows}, refs, f, path, 0,
                                         frozen_checkpoint=checkpoint if fold == 'test' else None)
                data, delta, _ = C.load_fold(fold, path)
                self.assertEqual([r['input']['candidate'] for r in data], ['good', 'bad', 'equal'])
                self.assertEqual([r['label'] for r in data], ['Accept', 'Reject', 'Reject'])
                self.assertEqual(delta[-1], 0.)  # Changed text with equal COMET stays Reject.
                self.assertEqual(f.scored_candidates, ['good', 'bad', 'equal'])
                self.assertEqual(meta['statistics'][fold]['pairs'], 3)
                self.assertEqual(meta['statistics'][fold]['no_op_pairs'], 1)
                self.assertEqual(meta['statistics'][fold]['valid_unique_pairs'], 4)
                sidecar = path / 'audit' / f'{fold}_no_ops.jsonl'
                no_ops = C.read_jsonl(sidecar)
                self.assertEqual(len(no_ops), 1)
                self.assertEqual(no_ops[0]['input']['current'], no_ops[0]['input']['candidate'])
                self.assertEqual(no_ops[0]['decision'], 'Reject')
                self.assertFalse(no_ops[0]['used_for_model'])
                self.assertNotIn('SECRET_REFERENCE', sidecar.read_text())
                self.assertNotIn('q_candidate', no_ops[0])  # No fabricated quality score.
                self.assertIn(f'audit/{fold}_no_ops.jsonl', meta['output_hashes'])
                self.assertEqual(len(rows), 5)  # Raw collection remains intact.

    def test_only_no_ops_preserve_audit_without_quality_scoring(self):
        row = {'sample_id': 'a', 'source': 's', 'before': 'before', 'candidate': 'before', 'valid': True}
        refs = {'train': {'a': {'source': 's', 'reference': 'ref'}}}
        f = Feedback()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data'
            meta = build_dataset({'train': [row]}, refs, f, path, 0)
            self.assertEqual(f.calls, 0)
            self.assertEqual(meta['statistics']['train']['pairs'], 0)
            self.assertEqual(meta['statistics']['train']['sources'], 0)
            self.assertEqual(meta['statistics']['train']['no_op_pairs'], 1)
            self.assertEqual(len(C.read_jsonl(path / 'audit/train_no_ops.jsonl')), 1)
            with self.assertRaisesRegex(ValueError, 'No changed pairs'):
                C.load_fold('train', path)
            with self.assertRaisesRegex(ValueError, 'No-op'):
                convert(dict(row, delta=0), 0)

    def test_loader_rejects_no_op_contamination_even_with_updated_hash(self):
        rows = [{'sample_id': 'a', 'source': 's', 'before': 'before', 'candidate': 'good', 'valid': True}]
        refs = {'train': {'a': {'source': 's', 'reference': 'ref'}}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data'
            meta = build_dataset({'train': rows}, refs, Feedback(), path, 0)
            item = C.read_jsonl(path / 'train.jsonl')[0]
            item['input']['candidate'] = item['input']['current']
            item['label'] = 'Reject'
            import json
            (path / 'train.jsonl').write_text(json.dumps(item) + '\n')
            meta['output_hashes']['train.jsonl'] = C.digest(path / 'train.jsonl')
            C.dump(path / 'dataset.json', meta)
            with self.assertRaisesRegex(ValueError, 'No-op'):
                C.load_fold('train', path)

    def test_no_op_sources_still_participate_in_split_isolation(self):
        row = {'sample_id': 'a', 'source': 'same', 'before': 'before', 'candidate': 'before', 'valid': True}
        ref = {'a': {'source': 'same', 'reference': 'ref'}}
        f = Feedback()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'Source leakage'):
                build_dataset({'train': [row], 'development': [dict(row, candidate='good')]},
                              {'train': ref, 'development': ref}, f, Path(temp) / 'data', 0)
        self.assertEqual(f.calls, 0)

    def test_split_leakage_detected_before_feedback(self):
        row = {'sample_id': 'a', 'source': 'same', 'before': 'before', 'candidate': 'good', 'valid': True}
        ref = {'a': {'source': 'same', 'reference': 'ref'}}
        f = Feedback()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                build_dataset({'train': [row], 'development': [row]}, {'train': ref, 'development': ref}, f, Path(temp) / 'data', 0)
        self.assertEqual(f.calls, 0)

    def test_test_labels_require_frozen_binary_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                build_dataset({'test': []}, {}, Feedback(), Path(temp) / 'data', 0)


if __name__ == '__main__':
    unittest.main()
