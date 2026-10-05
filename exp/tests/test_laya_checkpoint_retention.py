"""Check that later epochs and the best-model alias cannot erase earlier weights."""
import random

import numpy as np
import pytest
import torch

import train_laya_acceptance as T


def test_epoch_checkpoints_are_independent_and_preserve_rng(tmp_path, monkeypatch):
    def fake_save(model, cfg, tok, path, feedback):
        path.mkdir(parents=True)
        (path / 'model.safetensors').write_bytes(model)
        (path / 'rl_agent_config.json').write_text('{}')
    monkeypatch.setattr(T.C, 'save_checkpoint', fake_save)
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    records = []
    for epoch in range(1, 11):
        records.append(T.save_epoch_checkpoint(str(epoch).encode(), {}, None, tmp_path, {},
                                              {'epoch': epoch, 'development': {'macro_f1': epoch / 20}}))
    assert len(list((tmp_path / 'checkpoints').glob('epoch_*/model.safetensors'))) == 10
    for epoch, record in enumerate(records, 1):
        path = tmp_path / 'checkpoints' / f'epoch_{epoch:02d}'
        assert (path / 'model.safetensors').read_bytes() == str(epoch).encode()
        assert T.C.digest(path / 'model.safetensors') == record['checkpoint_sha256']
    with pytest.raises(FileExistsError):
        T.save_epoch_checkpoint(b'overwrite', {}, None, tmp_path, {}, {'epoch': 1, 'development': {}})
    assert (tmp_path / 'checkpoints/epoch_01/model.safetensors').read_bytes() == b'1'
    assert random.getstate() == rng[0]
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
    assert torch.equal(torch.random.get_rng_state(), rng[2])
