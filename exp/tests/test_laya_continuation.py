"""Verify epoch lineage and that persisted Adam state can reproduce a next step."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train_laya_acceptance as T


def test_warm_start_checks_epoch_weight_identity_and_data(tmp_path):
    parent = tmp_path / 'parent'
    data = tmp_path / 'data'
    data.mkdir()
    feedback = {'metric': 'comet', 'epsilon': 0.0}
    meta = {'label_feedback': feedback}
    T.C.dump(data / 'dataset.json', meta)
    records, history = [], []
    for epoch in (1, 2):
        p = parent / 'checkpoints' / f'epoch_{epoch:02d}'
        p.mkdir(parents=True)
        (p / 'model.safetensors').write_bytes(str(epoch).encode())
        T.C.dump(p / 'rl_agent_config.json', {'decision_schema': T.C.SCHEMA, 'labels': T.C.LABELS,
                'input_schema': list(T.C.INPUT_FIELDS), 'encoder': 'jhu-clsp/mmBERT-base', 'label_feedback': feedback})
        record = {'epoch': epoch, 'path': str(p), 'checkpoint_sha256': T.C.digest(p / 'model.safetensors'),
                  'config_sha256': T.C.digest(p / 'rl_agent_config.json'), 'development': {'macro_f1': .5, 'accuracy': .5}}
        T.C.dump(p / 'checkpoint_manifest.json', record)
        records.append(record)
        history.append({'epoch': epoch, 'development': record['development']})
    args = SimpleNamespace(warm_start=parent / 'checkpoints/epoch_02', initial_epoch=2, data=data,
                           base=tmp_path / 'base', seed=42, batch_size=16, effective_batch=64)
    T.C.dump(parent / 'training_protocol.json', {'dataset_sha256': T.C.digest(data / 'dataset.json'),
             'base': str(args.base.resolve()), 'seed': 42, 'micro_batch': 16, 'effective_batch': 64})
    T.C.dump(parent / 'epoch_checkpoints.json', records)
    T.C.dump(parent / 'history.json', history)
    context = T.continuation_context(args, meta)
    assert [r['epoch'] for r in context['records']] == [1, 2]
    assert context['optimizer_state_restored'] is False
    assert context['initial_checkpoint_selected_using_test'] is True
    args.initial_epoch = 1
    with pytest.raises(ValueError, match='epoch'):
        T.continuation_context(args, meta)
    args.initial_epoch = 2
    (args.warm_start / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        T.continuation_context(args, meta)


def test_optimizer_and_rng_state_roundtrip_reproduces_next_update(tmp_path):
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    def step(m, opt, sched):
        opt.zero_grad()
        m(torch.randn(4, 3)).square().mean().backward()
        opt.step()
        sched.step()
    step(model, optimizer, schedule)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    digest = T.save_training_state(tmp_path, optimizer, schedule, 11, 2420, 'modelhash', {'seed': 123})
    assert torch.equal(rng, torch.get_rng_state())
    assert digest == T.C.digest(tmp_path / 'training_state.pt')
    restored = torch.load(tmp_path / 'training_state.pt', map_location='cpu', weights_only=False)
    step(model, optimizer, schedule)
    expected_rng = (random.random(), np.random.rand(), torch.rand(1))
    copy = torch.nn.Linear(3, 2)
    copy.load_state_dict(weights)
    opt2 = torch.optim.AdamW(copy.parameters(), lr=1e-6)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10, eta_min=1e-6)
    opt2.load_state_dict(restored['optimizer'])
    sched2.load_state_dict(restored['scheduler'])
    random.setstate(restored['python_rng'])
    np.random.set_state(restored['numpy_rng'])
    torch.set_rng_state(restored['torch_rng'])
    step(copy, opt2, sched2)
    for a, b in zip(model.parameters(), copy.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert expected_rng[0] == random.random()
    assert expected_rng[1] == np.random.rand()
    torch.testing.assert_close(expected_rng[2], torch.rand(1), rtol=0, atol=0)
    assert restored['epoch'] == 11 and restored['completed_optimizer_steps'] == 2420
    with pytest.raises(FileExistsError):
        T.save_training_state(tmp_path, optimizer, schedule, 12, 2640, 'other', {})
