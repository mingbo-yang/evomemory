"""Formal binary action policy; probability thresholds are not supported."""
from pathlib import Path
import laya_acceptance_common as C


def choose_policy(checkpoint=C.OUT / 'best'):
    checkpoint = Path(checkpoint)
    C.validate_checkpoint(checkpoint)
    return {'version': C.SCHEMA, 'checkpoint': str(checkpoint.resolve()), 'labels': C.LABELS,
            'selection': 'first valid changed candidate classified Accept in generation order',
            'no_op_rule': C.NO_OP_RULE,
            'decision_rule': 'raw argmax; exact tie Reject', 'probabilities': 'diagnostic only'}


def select_candidate(candidates, policy=None, mode='laya'):
    if mode not in {'laya', 'unfiltered'}:
        raise ValueError('Unknown acceptance mode')
    if policy and any(k in policy for k in ('p_better', 'p_worse_max', 'threshold', 'margin')):
        raise ValueError('Runtime acceptance thresholds are forbidden by the binary protocol')
    for i, candidate in enumerate(candidates):
        if not candidate.get('valid') or not candidate.get('changed'):
            continue
        if mode == 'unfiltered':
            return i
        decision = candidate.get('decision')
        if decision is not None and decision not in C.LABELS:
            raise ValueError('Expected an explicit Accept/Reject decision')
        if decision == 'Accept':
            return i
    return None
