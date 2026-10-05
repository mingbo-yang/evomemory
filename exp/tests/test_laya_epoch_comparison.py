"""Prevent accidental source leakage and misaligned retrospective comparisons."""
import json

import numpy as np
import pytest

import compare_laya_epoch_checkpoints as E


def test_manifest_rejects_changed_weights_and_overlapping_sources(tmp_path):
    feedback = {'metric': 'comet', 'epsilon': 0.0}
    cfg = {'encoder': 'jhu-clsp/mmBERT-base', 'decision_schema': E.C.SCHEMA,
           'labels': E.C.LABELS, 'input_schema': list(E.C.INPUT_FIELDS),
           'seen_source_hashes': ['training_source'], 'label_feedback': feedback}
    (tmp_path / 'model.safetensors').write_bytes(b'original')
    (tmp_path / 'rl_agent_config.json').write_text(json.dumps(cfg))
    record = {'path': str(tmp_path), 'checkpoint_sha256': E.C.digest(tmp_path / 'model.safetensors'),
              'config_sha256': E.C.digest(tmp_path / 'rl_agent_config.json')}
    meta = {'label_feedback': feedback, 'source_hashes': {'test': ['test_source']}}
    assert E.check_record(record, feedback, meta) == tmp_path
    with pytest.raises(ValueError, match='overlap'):
        E.check_record(record, feedback, {**meta, 'source_hashes': {'test': ['training_source']}})
    with pytest.raises(ValueError, match='feedback'):
        E.check_record(record, {'metric': 'bleu'}, meta)
    (tmp_path / 'model.safetensors').write_bytes(b'modified')
    with pytest.raises(ValueError, match='changed'):
        E.check_record(record, feedback, meta)


def test_predictions_must_keep_ids_labels_and_exact_feedback():
    rows = [{'id': 'p1', 'label': 'Reject'}]
    predictions = [{'id': 'p1', 'sample_id': 's1', 'label': 'Reject', 'delta_comet': 0.0, 'logits': [0, 1]}]
    np.testing.assert_array_equal(E.aligned_logits(predictions, rows, [0.0], ['s1']), [[0, 1]])
    for field, value in [('id', 'other'), ('label', 'Accept'), ('sample_id', 's2'), ('delta_comet', 1e-10)]:
        with pytest.raises(ValueError):
            E.aligned_logits([{**predictions[0], field: value}], rows, [0.0], ['s1'])
    with pytest.raises(ValueError):
        E.aligned_logits([], rows, [0.0], ['s1'])


def test_paired_bootstrap_zero_for_identical_decisions_and_reject_all():
    rows = [{'id': str(i), 'input': {'source': 's', 'current': '原答案', 'candidate': '新答案'}, 'label': label}
            for i, label in enumerate(['Accept', 'Reject', 'Reject'])]
    z = np.array([[0., 1.]] * 3)
    boot = E.bootstrap_samples(rows, [.2, 0.0, -.1], ['s1', 's2', 's2'], {'a': z, 'b': z.copy()})
    np.testing.assert_array_equal(boot['a']['net_delta_per_pair'], np.zeros(2000))
    for key in boot['a']:
        np.testing.assert_array_equal(boot['a'][key], boot['b'][key])


def test_continuation_lineage_keeps_parent_epochs_and_discloses_test_selection(tmp_path):
    from types import SimpleNamespace
    parent, child, reference = [tmp_path / x for x in ['parent', 'child', 'reference']]
    record = {'epoch': 1, 'path': str(parent / 'checkpoints/epoch_01'), 'checkpoint_sha256': 'parent_hash', 'config_sha256': 'config'}
    common = {'dataset_sha256': 'data', 'base': 'base', 'seed': 42, 'micro_batch': 16,
              'effective_batch': 64, 'training': 'RLCD + CE', 'label_feedback': {'metric': 'comet'}}
    E.C.dump(reference / 'training_protocol.json', {**common, 'epochs': 1})
    E.C.dump(parent / 'epoch_checkpoints.json', [record])
    warm = {'parent_model': str(parent), 'parent_epoch': 1, 'path': record['path'],
            'checkpoint_sha256': 'parent_hash', 'initial_checkpoint_selected_using_test': True}
    E.C.dump(child / 'training_protocol.json', {**common, 'epochs': 2, 'initial_epoch': 1,
                                               'additional_epochs': 1, 'warm_start': warm})
    args = SimpleNamespace(model=child, baseline=reference, previous=reference)
    lock = {'epochs': 2, 'checkpoints': [record, {'epoch': 2}], 'test_used_for_selection': True}
    assert E.validate_training_lineage(args, lock)['initial_epoch'] == 1
    with pytest.raises(ValueError, match='disclosed'):
        E.validate_training_lineage(args, {**lock, 'test_used_for_selection': False})
    with pytest.raises(ValueError, match='Inherited'):
        E.validate_training_lineage(args, {**lock, 'checkpoints': [{**record, 'checkpoint_sha256': 'wrong'}, {'epoch': 2}]})
