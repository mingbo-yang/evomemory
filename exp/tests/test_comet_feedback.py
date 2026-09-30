import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from core.comet_feedback import CometFeedback, validated_scores, validate_epsilon


def test_feedback_deduplicates_identical_answers_and_reuses_scores():
    with tempfile.TemporaryDirectory() as temp:
        ckpt = Path(temp) / 'model.ckpt'
        ckpt.write_bytes(b'offline test fixture, not model weights')
        feedback = CometFeedback(ckpt, sys.executable)
        rows = [{'source': 's', 'current': 'same', 'candidate': 'same', 'reference': 'feedback-only'},
                {'source': 's', 'current': 'same', 'candidate': 'better', 'reference': 'feedback-only'}]
        def worker(command, **kwargs):
            inputs = json.loads(Path(command[command.index('--input') + 1]).read_text())
            assert len(inputs) == 2
            assert all(x['ref'] == 'feedback-only' for x in inputs)
            assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
            Path(command[command.index('--output') + 1]).write_text(json.dumps([.5 if x['mt'] == 'same' else .6 for x in inputs]))
            return SimpleNamespace(returncode=0, stderr='')
        with patch('core.comet_feedback.subprocess.run', side_effect=worker) as process:
            first = feedback.score_pairs(rows)
            second = feedback.score_pairs(rows)
            assert process.call_count == 1
        assert first == second
        assert first[0]['delta'] == 0
        assert first[1]['delta'] > 0


def test_feedback_failure_never_becomes_a_synthetic_zero_score():
    with tempfile.TemporaryDirectory() as temp:
        ckpt = Path(temp) / 'model.ckpt'
        ckpt.write_bytes(b'fixture')
        feedback = CometFeedback(ckpt, sys.executable)
        with patch('core.comet_feedback.subprocess.run', return_value=SimpleNamespace(returncode=1, stderr='fixture failure')):
            with pytest.raises(RuntimeError, match='no score fallback'):
                feedback.score_pairs([{'source': 's', 'current': 'a', 'candidate': 'b', 'reference': 'r'}])


def test_zero_tolerances_are_valid_but_bad_feedback_is_not():
    assert validate_epsilon(0) == 0
    for x in [None, -1, float('nan'), True]:
        with pytest.raises(ValueError):
            validate_epsilon(x)
    with pytest.raises(ValueError):
        validated_scores([{'current': 'a', 'candidate': 'b'}], [{'q_current': .5, 'q_candidate': float('nan')}])
    with pytest.raises(ValueError):
        validated_scores([{'current': 'a', 'candidate': 'a'}], [{'q_current': .5, 'q_candidate': .6}])


def test_feedback_preserves_virtual_environment_interpreter_symlink(tmp_path):
    blob = tmp_path / 'blob'
    blob.write_bytes(b'fixture')
    checkpoint = tmp_path / 'snapshot/checkpoints/model.ckpt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.symlink_to(blob)
    venv_python = tmp_path / 'venv/bin/python'
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    feedback = CometFeedback(checkpoint, venv_python)
    def worker(command, **kwargs):
        assert command[0] == str(venv_python)
        assert command[command.index('--checkpoint') + 1] == str(checkpoint)
        Path(command[command.index('--output') + 1]).write_text('[0.5, 0.6]')
        return SimpleNamespace(returncode=0, stderr='')
    with patch('core.comet_feedback.subprocess.run', side_effect=worker):
        assert feedback.score_pairs([{'source': 's', 'current': 'a', 'candidate': 'b', 'reference': 'r'}])[0]['delta'] > 0
