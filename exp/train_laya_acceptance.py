"""Train Laya multilingual on binary action targets; no threshold search or answer swapping."""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

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


def train(args):
    meta = json.loads((args.data / 'dataset.json').read_text())
    if meta.get('label_feedback', {}).get('metric') != 'comet':
        raise ValueError('Train on the binary COMET-labelled dataset')
    rows, _, _ = C.load_fold('train', args.data)
    dev, delta, _ = C.load_fold('development', args.data)
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
                'base': str(args.base.resolve()), 'seed': args.seed, 'epochs': args.epochs,
                'micro_batch': args.batch_size, 'effective_batch': args.effective_batch,
                'training': 'RLCD + CE; two options; random option order; no answer swapping',
                'selection': 'development macro F1, then accuracy; no runtime thresholds',
                'probabilities': 'diagnostic only; no mandatory calibration phase',
                'created_at_utc': datetime.now(timezone.utc).isoformat()}
    C.dump(args.output / 'training_protocol.json', protocol)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    tok = C.get_tokenizer(args.base)
    tr, de = C.EncodedPairs(rows, tok), C.EncodedPairs(dev, tok)
    model, cfg = C.load_model(args.base, initialization=True)
    cfg['seen_source_hashes'] = sorted(set().union(*(set(v) for v in meta['source_hashes'].values())))
    model.to(args.device)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.head_checkpointing = True
    for parameter in model.act_head.parameters():
        parameter.requires_grad_(False)
    encoder = [v for n, v in model.named_parameters() if n.startswith('encoder.') and v.requires_grad]
    head = [v for n, v in model.named_parameters() if not n.startswith('encoder.') and v.requires_grad]
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': 2.5e-5}, {'params': head, 'lr': 1e-4}], weight_decay=.01)
    frequency = np.array([counts[label] for label in C.LABELS], dtype=float)
    weights = 1 / np.sqrt(frequency)
    weights /= np.sum(weights * frequency / frequency.sum())
    steps = math.ceil(len(rows) / args.effective_batch) * args.epochs
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)
    best = None
    history = []
    completed_steps = 0
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(order)
        sigma = .4 - .3 * epoch / max(1, args.epochs - 1)
        for start in range(0, len(order), args.effective_batch):
            group = order[start:start + args.effective_batch]
            optimizer.zero_grad(set_to_none=True)
            for sub in range(0, len(group), args.batch_size):
                indices = group[sub:sub + args.batch_size]
                batch = C.pack([tr.item(i, epoch=epoch) for i in indices], tok, args.device)
                loss, _, _ = rlcd_loss(model, batch, sigma, weights)
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
                progress = {'epoch': epoch + 1, 'epochs': args.epochs,
                            'epoch_pairs_completed': start + len(group), 'epoch_pairs_total': len(order),
                            'optimizer_steps_completed': completed_steps, 'optimizer_steps_total': steps,
                            'last_micro_batch_loss': float(loss.detach().cpu()),
                            'updated_at_utc': datetime.now(timezone.utc).isoformat()}
                C.dump(args.output / 'training_progress.json', progress)
                print(json.dumps({'progress': progress}), flush=True)
        logits = C.infer(model, de, args.batch_size)
        result = C.metrics(dev, logits, delta)
        history.append({'epoch': epoch + 1, 'development': result})
        C.dump(args.output / 'history.json', history)
        key = C.selection_key(result)
        if best is None or key > best:
            best = key
            C.save_checkpoint(model, cfg, tok, args.output / 'best', meta['label_feedback'])
            C.dump(args.output / 'best_selection.json', history[-1])
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    C.dump(args.output / 'evaluation_lock.json', {
        'decision_schema': C.SCHEMA, 'checkpoint_sha256': C.digest(args.output / 'best/model.safetensors'),
        'config_sha256': C.digest(args.output / 'best/rl_agent_config.json'),
        'label_feedback': meta['label_feedback'], 'decision_rule': 'raw argmax; exact tie Reject',
        'test_used_for_selection': False, 'runtime_thresholds': None})
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=C.DATA)
    p.add_argument('--output', type=Path, default=C.OUT)
    p.add_argument('--base', type=Path, default=C.BASE)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--effective-batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
