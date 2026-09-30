import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import laya_acceptance_common as C
from train_laya_acceptance import rlcd_loss


class Tokenizer:
    mask_token = '[MASK]'
    mask_token_id = 2
    cls_token_id = 1
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        ids = [4 + ord(c) % 90 for c in text]
        if kwargs.get('truncation'):
            ids = ids[:kwargs['max_length']]
        return {'input_ids': ids}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(100, 4)
        self.scorer = torch.nn.Linear(4, 1)

    def forward(self, ids, attention, markers, marker_mask, qtype):
        # Score the option text following each marker, as a deterministic tiny fixture.
        token = ids.gather(1, markers + 2)
        z = self.scorer(self.embedding(token)).squeeze(-1)
        return z, None


class LayaProtocol(unittest.TestCase):
    def test_input_whitelist(self):
        data = {'source': 'source', 'current': '当前', 'candidate': '候选', 'reference': 'SECRET', 'comet': 999, 'memory': 'PRIVATE'}
        text = C.state_text(data)
        for forbidden in ('SECRET', '999', 'PRIVATE'):
            self.assertNotIn(forbidden, text)
        self.assertEqual(C.LABELS, ['Accept', 'Reject'])

    def test_binary_decision_and_temperature_diagnostics(self):
        z = np.array([[.1, 0.], [0., 0.], [0., .1]])
        self.assertEqual(C.decisions(z), ['Accept', 'Reject', 'Reject'])
        rows = [{'input': {'source': 's', 'current': '当前', 'candidate': '候选'}, 'label': x}
                for x in ['Accept', 'Reject', 'Reject']]
        a = C.metrics(rows, z, [.1, 0., -.1], .1)
        b = C.metrics(rows, z, [.1, 0., -.1], 10.)
        self.assertEqual(a['accepted'], b['accepted'])
        self.assertEqual(a['accuracy'], b['accuracy'])
        self.assertNotEqual(a['diagnostics']['nll'], b['diagnostics']['nll'])
        with self.assertRaises(ValueError):
            C.decisions([[.4, .3, .3]])
        with self.assertRaises(ValueError):
            C.decisions([[float('nan'), 0.]])

    def test_no_op_never_enters_model_encoding_or_main_metrics(self):
        rows = [{'id': 'noop', 'input': {'source': 's', 'current': '相同', 'candidate': '相同'}, 'label': 'Reject'}]
        with self.assertRaisesRegex(ValueError, 'No-op'):
            C.EncodedPairs(rows, Tokenizer())
        with self.assertRaisesRegex(ValueError, 'No-op'):
            C.metrics(rows, [[0., 100.]], [0.])
        self.assertFalse(C.accepted_mask(rows, [[100., 0.]])[0])

    def test_exact_equality_does_not_remove_changed_text(self):
        for candidate in ['答案 ', '答 案', '答案。', '答案！']:
            self.assertFalse(C.is_noop({'current': '答案', 'candidate': candidate}))
        self.assertFalse(C.is_noop({'current': 'Ａ', 'candidate': 'A'}))
        self.assertFalse(C.is_noop({'current': 'A', 'candidate': 'a'}))
        self.assertTrue(C.is_noop({'current': '答案', 'candidate': '答案'}))

    def test_old_checkpoint_fails_before_weights_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / 'rl_agent_config.json').write_text(json.dumps({'encoder': 'jhu-clsp/mmBERT-base', 'labels': ['Better', 'Tie', 'Worse']}))
            with self.assertRaises(ValueError):
                C.validate_checkpoint(p)

    def test_two_options_and_no_answer_swapping(self):
        rows = [{'id': 'r', 'input': {'source': 's', 'current': '当前', 'candidate': '候选'}, 'label': 'Reject'}]
        enc = C.EncodedPairs(rows, Tokenizer())
        for epoch in range(8):
            item = enc.item(0, epoch=epoch)
            self.assertFalse(item['swapped'])
            self.assertEqual(len(item['markers']), 2)
            self.assertEqual(len(item['target']), 2)
            self.assertEqual(item['order'][item['label']], 1)
        with self.assertRaises(ValueError):
            C.EncodedPairs(rows, Tokenizer(), allow_swap=True)

    def test_binary_forward_backward_and_option_order_parity(self):
        rows = [{'id': str(i), 'input': {'source': 's', 'current': '当前', 'candidate': '候选'}, 'label': label}
                for i, label in enumerate(C.LABELS)]
        tok = Tokenizer()
        enc = C.EncodedPairs(rows, tok)
        model = TinyModel()
        a = C.infer(model, enc, force_order=[0, 1])
        b = C.infer(model, enc, force_order=[1, 0])
        np.testing.assert_allclose(a, b)
        batch = C.pack([enc.item(i, epoch=0) for i in range(2)], tok, 'cpu')
        loss, _, _ = rlcd_loss(model, batch, .2, [1., 1.])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.scorer.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
