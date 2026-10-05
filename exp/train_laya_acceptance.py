"""Train Laya multilingual on binary action targets; no threshold search or answer swapping."""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import laya_acceptance_common as C


def rlcd_loss(model, batch, sigma, weights):
    """Retain the existing RLCD + CE objective, now with two option targets."""
    from laya.common import proper_reward
    device = batch['input_ids'].device
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with amp:
        logits, _ = C.forward(model, batch)
    logits = logits.float()
    mask, target = batch['marker_mask'], batch['target']
    if logits.shape[-1] != 2 or target.shape != logits.shape:
        raise ValueError('Binary training requires exactly two logits and targets')
    k = mask.sum(-1, keepdim=True).float()
    noise = torch.randn((4,) + logits.shape, device=device) * sigma * mask
    noise = (noise - noise.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + noise
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        reward = proper_reward(q, target.unsqueeze(0), batch['qtype'], mask, w_sph=.75, w_rps=1.)
        advantage = reward - reward.mean(0, keepdim=True)
        advantage /= advantage.std() + 1e-6
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    rl = -(advantage * logp).mean(0)
    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)
    weight = torch.tensor([weights[it['semantic_label']] for it in batch['meta']], device=device)
    return ((rl + ce) * weight).mean(), ce.mean(), reward.mean()


def save_epoch_checkpoint(model, cfg, tok, output, feedback, record):
    """Retain each completed epoch independently; never overwrite a saved epoch."""
    path = Path(output) / 'checkpoints' / f"epoch_{record['epoch']:02d}"
    if path.exists():
        raise FileExistsError(f'Epoch checkpoint already exists: {path}')
    C.save_checkpoint(model, cfg, tok, path, feedback)
    manifest = {'epoch': record['epoch'], 'path': str(path.resolve()),
                'checkpoint_sha256': C.digest(path / 'model.safetensors'),
                'config_sha256': C.digest(path / 'rl_agent_config.json'),
                'development': record['development'],
                'saved_at_utc': datetime.now(timezone.utc).isoformat()}
    C.dump(path / 'checkpoint_manifest.json', manifest)
    return manifest


def continuation_context(args, meta):
    """Validate the exact parent epoch and retain its immutable ancestry."""
    if args.warm_start is None:
        if args.initial_epoch != 0:
            raise ValueError('initial_epoch requires a warm-start checkpoint')
        return None
    path = args.warm_start.resolve()
    cfg = C.validate_checkpoint(path)
    record = json.loads((path / 'checkpoint_manifest.json').read_text())
    parent = path.parents[1]
    protocol = json.loads((parent / 'training_protocol.json').read_text())
    if args.initial_epoch < 1 or record['epoch'] != args.initial_epoch:
        raise ValueError('Warm-start checkpoint epoch does not match initial_epoch')
    if C.digest(path / 'model.safetensors') != record['checkpoint_sha256'] or C.digest(path / 'rl_agent_config.json') != record['config_sha256']:
        raise ValueError('Warm-start checkpoint changed')
    if cfg['label_feedback'] != meta['label_feedback'] or protocol['dataset_sha256'] != C.digest(args.data / 'dataset.json'):
        raise ValueError('Continuation must use the same data and feedback')
    for key, value in [('base', str(args.base.resolve())), ('seed', args.seed),
                       ('micro_batch', args.batch_size), ('effective_batch', args.effective_batch)]:
        if protocol[key] != value:
            raise ValueError(f'Continuation protocol mismatch: {key}')
    records = [r for r in json.loads((parent / 'epoch_checkpoints.json').read_text()) if r['epoch'] <= args.initial_epoch]
    history = [h for h in json.loads((parent / 'history.json').read_text()) if h['epoch'] <= args.initial_epoch]
    expected = list(range(1, args.initial_epoch + 1))
    if [r['epoch'] for r in records] != expected or [h['epoch'] for h in history] != expected:
        raise ValueError('Incomplete parent history/checkpoints')
    if records[-1]['checkpoint_sha256'] != record['checkpoint_sha256']:
        raise ValueError('Parent index does not match the requested starting weights')
    return {'path': str(path), 'parent_model': str(parent), 'parent_epoch': args.initial_epoch,
            'checkpoint_sha256': record['checkpoint_sha256'], 'config_sha256': record['config_sha256'],
            'optimizer_state_restored': False, 'rng_state_restored': False,
            'initial_checkpoint_selected_using_test': True,
            'mode': 'weights-only continuation with a new AdamW optimizer; not uninterrupted training',
            'records': records, 'history': history}


def save_training_state(path, optimizer, schedule, epoch, completed_steps, checkpoint_sha256, protocol):
    """Save full optimizer/scheduler/RNG state separately from inference weights."""
    state = {'format': 'laya-training-state-v1', 'epoch': epoch, 'completed_optimizer_steps': completed_steps,
             'checkpoint_sha256': checkpoint_sha256, 'training_protocol': protocol,
             'optimizer': optimizer.state_dict(), 'scheduler': schedule.state_dict(),
             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}
    target = Path(path) / 'training_state.pt'
    if target.exists():
        raise FileExistsError(f'Training state already exists: {target}')
    temporary = target.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(target)
    return C.digest(target)


def train(args):
    meta = json.loads((args.data / 'dataset.json').read_text())
    if meta.get('label_feedback', {}).get('metric') != 'comet':
        raise ValueError('Train on the binary COMET-labelled dataset')
    rows, _, _ = C.load_fold('train', args.data)
    dev, delta, _ = C.load_fold('development', args.data)
    if min(args.encoder_lr, args.head_lr, args.min_lr, args.sigma_start, args.sigma_end) <= 0:
        raise ValueError('Positive learning rates and RLCD noise required')
    if args.min_lr > min(args.encoder_lr, args.head_lr):
        raise ValueError('Minimum learning rate exceeds initial rates')
    parent = continuation_context(args, meta)
    total_epochs = args.initial_epoch + args.epochs
    if min(args.epochs, args.batch_size, args.effective_batch) < 1:
        raise ValueError('Positive training sizes required')
    counts = Counter(r['label'] for r in rows)
    if any(counts[label] == 0 for label in C.LABELS):
        raise ValueError('Training data must contain both Accept and Reject')
    if args.output.exists():
        raise FileExistsError('Use a new binary model directory; never overwrite an old experiment')
    args.output.mkdir(parents=True)
    protocol = {'decision_schema': C.SCHEMA, 'labels': C.LABELS, 'label_feedback': meta['label_feedback'],
                'model_population': C.MODEL_POPULATION, 'no_op_rule': C.NO_OP_RULE,
                'dataset_sha256': C.digest(args.data / 'dataset.json'), 'data': str(args.data.resolve()),
                'base': str(args.base.resolve()), 'seed': args.seed, 'epochs': total_epochs,
                'additional_epochs': args.epochs, 'initial_epoch': args.initial_epoch,
                'warm_start': {k: v for k, v in parent.items() if k not in ('records', 'history')} if parent else None,
                'optimizer': 'AdamW; weight_decay=0.01; fresh optimizer for this phase',
                'encoder_lr': args.encoder_lr, 'head_lr': args.head_lr, 'min_lr': args.min_lr,
                'sigma_start': args.sigma_start, 'sigma_end': args.sigma_end,
                'micro_batch': args.batch_size, 'effective_batch': args.effective_batch,
                'training': 'RLCD + CE; two options; random option order; no answer swapping',
                'selection': 'development macro F1, then accuracy; no runtime thresholds',
                'probabilities': 'diagnostic only; no mandatory calibration phase',
                'checkpoint_retention': 'every completed epoch with weights and training_state.pt; inherited epochs remain in their original directories; plus development-selected best',
                'created_at_utc': datetime.now(timezone.utc).isoformat()}
    C.dump(args.output / 'training_protocol.json', protocol)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    initialization_path = args.warm_start if parent else args.base
    tok = C.get_tokenizer(initialization_path)
    tr, de = C.EncodedPairs(rows, tok), C.EncodedPairs(dev, tok)
    model, cfg = C.load_model(initialization_path, initialization=parent is None)
    if tr.template_version != de.template_version:
        raise ValueError('Train and Dev input templates differ')
    cfg['input_template_version'] = tr.template_version
    cfg['seen_source_hashes'] = sorted(set().union(*(set(v) for v in meta['source_hashes'].values())))
    model.to(args.device)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.head_checkpointing = True
    for parameter in model.act_head.parameters():
        parameter.requires_grad_(False)
    encoder = [v for n, v in model.named_parameters() if n.startswith('encoder.') and v.requires_grad]
    head = [v for n, v in model.named_parameters() if not n.startswith('encoder.') and v.requires_grad]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': args.encoder_lr}, {'params': head, 'lr': args.head_lr}], weight_decay=.01)
    frequency = np.array([counts[label] for label in C.LABELS], dtype=float)
    weights = 1 / np.sqrt(frequency)
    weights /= np.sum(weights * frequency / frequency.sum())
    steps = math.ceil(len(rows) / args.effective_batch) * args.epochs
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=args.min_lr)
    best = None
    history = list(parent['history']) if parent else []
    completed_steps = 0
    checkpoints = list(parent['records']) if parent else []
    if parent:
        previous_best = max(checkpoints, key=lambda r: C.selection_key(r['development']))
        best = C.selection_key(previous_best['development'])
        shutil.copytree(Path(previous_best['path']), args.output / 'best',
                        ignore=shutil.ignore_patterns('checkpoint_manifest.json', 'training_state.pt'))
        C.dump(args.output / 'best_selection.json', {'epoch': previous_best['epoch'], 'development': previous_best['development']})
    for local_epoch in range(args.epochs):
        epoch = args.initial_epoch + local_epoch
        totals = torch.zeros(3, device=args.device)
        model.train()
        order = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(order)
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * local_epoch / max(1, args.epochs - 1)
        for start in range(0, len(order), args.effective_batch):
            group = order[start:start + args.effective_batch]
            optimizer.zero_grad(set_to_none=True)
            for sub in range(0, len(group), args.batch_size):
                indices = group[sub:sub + args.batch_size]
                batch = C.pack([tr.item(i, epoch=epoch) for i in indices], tok, args.device)
                loss, ce, reward = rlcd_loss(model, batch, sigma, weights)
                totals += torch.stack((loss.detach(), ce.detach(), reward.detach())) * len(indices)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite binary training loss')
                (loss * len(indices) / len(group)).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite training gradient')
            optimizer.step()
            schedule.step()
            completed_steps += 1
            if completed_steps == 1 or completed_steps % 25 == 0 or start + len(group) == len(order):
                progress = {'epoch': epoch + 1, 'epochs': total_epochs,
                            'phase_epoch': local_epoch + 1, 'phase_epochs': args.epochs,
                            'epoch_pairs_completed': start + len(group), 'epoch_pairs_total': len(order),
                            'optimizer_steps_completed': completed_steps, 'optimizer_steps_total': steps,
                            'last_micro_batch_loss': float(loss.detach().cpu()),
                            'updated_at_utc': datetime.now(timezone.utc).isoformat()}
                C.dump(args.output / 'training_progress.json', progress)
                print(json.dumps({'progress': progress}), flush=True)
        logits = C.infer(model, de, args.batch_size)
        result = C.metrics(dev, logits, delta)
        averaged = (totals / len(rows)).cpu().tolist()
        history.append({'epoch': epoch + 1, 'development': result,
                        'training_objective': dict(zip(['weighted_rlcd_plus_ce', 'ce', 'reward'], averaged)),
                        'sigma': sigma, 'learning_rates': schedule.get_last_lr()})
        C.dump(args.output / 'history.json', history)
        checkpoints.append(save_epoch_checkpoint(model, cfg, tok, args.output, meta['label_feedback'], history[-1]))
        key = C.selection_key(result)
        if best is None or key > best:
            best = key
            C.save_checkpoint(model, cfg, tok, args.output / 'best', meta['label_feedback'])
            C.dump(args.output / 'best_selection.json', history[-1])
        checkpoint = checkpoints[-1]
        checkpoint['training_state_sha256'] = save_training_state(
            checkpoint['path'], optimizer, schedule, epoch + 1,
            math.ceil(len(rows) / args.effective_batch) * args.initial_epoch + completed_steps,
            checkpoint['checkpoint_sha256'], protocol)
        C.dump(Path(checkpoint['path']) / 'checkpoint_manifest.json', checkpoint)
        C.dump(args.output / 'epoch_checkpoints.json', checkpoints)
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    C.dump(args.output / 'all_checkpoints_lock.json', {
        'decision_schema': C.SCHEMA, 'epochs': total_epochs, 'checkpoints': checkpoints,
        'dataset_sha256': protocol['dataset_sha256'], 'label_feedback': meta['label_feedback'],
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'test_used_for_selection': bool(parent),
        'current_phase_test_used_for_selection': False, 'warm_start': protocol['warm_start']})
    C.dump(args.output / 'evaluation_lock.json', {
        'decision_schema': C.SCHEMA, 'checkpoint_sha256': C.digest(args.output / 'best/model.safetensors'),
        'config_sha256': C.digest(args.output / 'best/rl_agent_config.json'),
        'label_feedback': meta['label_feedback'], 'decision_rule': 'raw argmax; exact tie Reject',
        'test_used_for_selection': bool(parent), 'current_phase_test_used_for_selection': False,
        'warm_start': protocol['warm_start'], 'runtime_thresholds': None})
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=C.DATA)
    p.add_argument('--output', type=Path, default=C.OUT)
    p.add_argument('--base', type=Path, default=C.BASE)
    p.add_argument('--warm-start', type=Path, help='Existing binary epoch weights; optimizer is explicitly reset')
    p.add_argument('--initial-epoch', type=int, default=0)
    p.add_argument('--encoder-lr', type=float, default=2.5e-5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--sigma-start', type=float, default=.4)
    p.add_argument('--sigma-end', type=float, default=.1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--effective-batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
