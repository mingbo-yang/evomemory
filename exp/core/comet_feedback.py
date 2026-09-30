"""Reference-based COMET feedback, isolated from the reference-free Laya model.

Called for offline labelling or after a task/batch has finished. A separate
Python environment avoids mixing COMET and generator dependency versions.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

DEFAULT_CHECKPOINT = Path('/mnt/huawei/ymb/model/wmt22-comet-da/models--Unbabel--wmt22-comet-da/snapshots/2760a223ac957f30acfb18c8aa649b01cf1d75f2/checkpoints/model.ckpt')
DEFAULT_PYTHON = Path('/mnt/huawei/ymb/.tmp/laya_comet_env/bin/python')


def validate_epsilon(value, name='epsilon'):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be explicitly set to a finite nonnegative COMET-score difference')
    return float(value)


def validated_scores(rows, scores):
    if len(rows) != len(scores):
        raise ValueError('Feedback row count mismatch')
    result = []
    for row, pair in zip(rows, scores):
        before, after = float(pair['q_current']), float(pair['q_candidate'])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError('Nonfinite COMET score')
        if row['current'] == row['candidate'] and before != after:
            raise ValueError('Identical answers must share the same feedback score')
        result.append({'q_current': before, 'q_candidate': after, 'delta': after - before})
    return result


class CometFeedback:
    metric = 'comet'

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, python=DEFAULT_PYTHON, device='cpu', batch_size=8):
        # COMET locates hparams.yaml relative to the checkpoint snapshot path.
        self.checkpoint = Path(checkpoint).absolute()
        # Preserve the venv entry path: resolving its symlink bypasses pyvenv.cfg.
        self.python = Path(python).absolute()
        if not self.checkpoint.is_file() or not self.python.is_file():
            raise FileNotFoundError('Local COMET checkpoint and Python environment are required; no online download fallback')
        if device not in {'cpu', 'cuda'} or batch_size < 1:
            raise ValueError('Invalid feedback device/batch size')
        self.device = device
        self.batch_size = batch_size
        self.cache = {}
        self._metadata = None

    @property
    def metadata(self):
        if self._metadata is None:
            h = hashlib.sha256()
            with self.checkpoint.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(block)
            self._metadata = {'metric': 'comet', 'checkpoint': str(self.checkpoint), 'checkpoint_sha256': h.hexdigest(),
                              'scale': 'native COMET score; no BLEU conversion or division by 100'}
        return dict(self._metadata)

    def score_pairs(self, rows):
        """rows have source/current/candidate/reference; never pass these rows to Laya."""
        keys = []
        pending = {}
        for row in rows:
            pair = []
            for field in ('current', 'candidate'):
                obj = {'src': row['source'], 'mt': row[field], 'ref': row['reference']}
                if any(not isinstance(v, str) for v in obj.values()):
                    raise ValueError('COMET expects text fields')
                key = hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                pair.append(key)
                if key not in self.cache:
                    pending[key] = obj
            keys.append(pair)
        if pending:
            with tempfile.TemporaryDirectory(prefix='laya-comet-feedback-') as tmp:
                inp, out = Path(tmp) / 'input.json', Path(tmp) / 'output.json'
                inp.write_text(json.dumps(list(pending.values()), ensure_ascii=False))
                env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
                if self.device == 'cpu':
                    env['CUDA_VISIBLE_DEVICES'] = ''
                command = [str(self.python), str(Path(__file__).with_name('comet_feedback_worker.py')),
                           '--input', str(inp), '--output', str(out), '--checkpoint', str(self.checkpoint),
                           '--device', self.device, '--batch-size', str(self.batch_size)]
                run = subprocess.run(command, env=env, capture_output=True, text=True)
                if run.returncode:
                    raise RuntimeError('COMET feedback failed; no score fallback: ' + run.stderr[-2000:])
                values = json.loads(out.read_text())
                if len(values) != len(pending) or any(not math.isfinite(v) for v in values):
                    raise ValueError('Invalid COMET worker output')
                self.cache.update(zip(pending, map(float, values)))
        return validated_scores(rows, [{'q_current': self.cache[a], 'q_candidate': self.cache[b]} for a, b in keys])


def from_config(config):
    if config.get('metric') != 'comet':
        raise ValueError('The first binary protocol requires COMET; no silent BLEU fallback')
    return CometFeedback(checkpoint=config.get('checkpoint', DEFAULT_CHECKPOINT),
                         python=config.get('python', DEFAULT_PYTHON), device=config.get('device', 'cpu'),
                         batch_size=config.get('batch_size', 8))
