"""Single-candidate binary action policy; no ranking, bypass or thresholds."""
from pathlib import Path
import laya_acceptance_common as C


def choose_policy(checkpoint):
    checkpoint = Path(checkpoint)
    C.validate_checkpoint(checkpoint)
    return {'version': C.SCHEMA, 'checkpoint': str(checkpoint.resolve()), 'labels': C.LABELS,
            'selection': 'the sole valid changed candidate replaces current iff classified Accept',
            'no_op_rule': C.NO_OP_RULE,
            'decision_rule': 'raw argmax; exact tie Reject', 'probabilities': 'diagnostic only'}


def select_candidate(candidates, policy=None):
    if len(candidates) != 1:
        raise ValueError('Formal acceptance requires exactly one candidate; use the separate ablation entry point')
    if policy and any(k in policy for k in ('p_better', 'p_worse_max', 'threshold', 'margin')):
        raise ValueError('Runtime acceptance thresholds are forbidden by the binary protocol')
    candidate = candidates[0]
    if not candidate.get('valid') or not candidate.get('changed'):
        return None
    decision = candidate.get('decision')
    if decision not in C.LABELS:
        raise ValueError('Expected an explicit Accept/Reject decision')
    return 0 if decision == 'Accept' else None
