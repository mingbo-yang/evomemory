"""Independently calibrate safe acceptance for the two frozen Laya checkpoints.

This reads only the previously reserved threshold-calibration fold. It does not
inspect or tune against rollout outcomes, and a failed calibration remains a
failed calibration instead of being replaced with a permissive fallback.
"""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np

from laya_acceptance_common import allowed, digest, dump, probabilities, read_jsonl


ROOT = Path(__file__).resolve().parent
MODEL = Path('/mnt/huawei/ymb/model/laya-multilingual-label-comparison-v1')
OUT = ROOT / 'runs/laya_safe_acceptance_v1'
MARGINS = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2)


def select(candidates, margin):
    """Return an eligible candidate slot or None; indices follow Better/Tie/Worse."""
    eligible = []
    for slot, candidate in enumerate(candidates):
        p = candidate.get('probabilities')
        if not candidate.get('valid') or not candidate.get('changed') or p is None:
            continue
        better, tie, worse = p
        if better <= max(tie, worse) or better - worse < margin:
            continue
        eligible.append((better - worse, better, -slot, slot))
    return max(eligible)[-1] if eligible else None


def main():
    lock = json.loads((MODEL / 'comparison_evaluation_lock.json').read_text())
    rows = read_jsonl(MODEL / 'inputs/threshold_calibration.jsonl')
    labels = {r['id']: r['label'] for r in read_jsonl(MODEL / 'labels/threshold_calibration.jsonl')}
    assert len(rows) == len(labels) == 48
    assert digest(MODEL / 'inputs/threshold_calibration.jsonl') == lock['input_hashes']['threshold_calibration']
    assert digest(MODEL / 'labels/threshold_calibration.jsonl') == lock['calibration_label_hashes']['threshold_calibration']

    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'role': 'pre-rollout calibration on the existing threshold fold only',
        'rule': 'basic-valid changed; pBetter > max(pTie,pWorse); pBetter-pWorse >= model-specific margin',
        'selection': 'maximum pBetter-pWorse among eligible candidates',
        'margin_grid': list(MARGINS),
        'minimum_accepted': 5,
        'minimum_confirmed_better_fraction': .5,
        'minimum_worst_case_net': 1,
        'uncertain_treated_as_unconfirmed': True,
        'source_rows': len(rows),
        'checkpoint_hashes': {},
        'fold_hashes': {name: digest(MODEL / name) for name in [
            'inputs/threshold_calibration.jsonl', 'labels/threshold_calibration.jsonl']},
        'models': {},
        'new_rollout_used_for_selection': False,
    }
    for arm in ('semantic', 'bleu'):
        checkpoint = MODEL / arm / 'best'
        actual_hash = digest(checkpoint / 'model.safetensors')
        assert actual_hash == lock['models'][arm]['saved_model_sha256']
        report['checkpoint_hashes'][arm] = actual_hash
        temp = lock['models'][arm]['temperature']
        p = probabilities(np.load(MODEL / f'{arm}_threshold_logits.npy'), temp)
        assert p.shape == (len(rows), 3)
        candidates = []
        for margin in MARGINS:
            selected = [i for i, (row, prob) in enumerate(zip(rows, p))
                        if allowed(row) and prob[0] > max(prob[1], prob[2])
                        and prob[0] - prob[2] >= margin]
            counts = Counter(labels[rows[i]['id']] for i in selected)
            n = len(selected)
            better, worse, uncertain = counts['Better'], counts['Worse'], counts['Uncertain']
            candidates.append({
                'margin': margin,
                'accepted': n,
                'counts': dict(counts),
                'coverage': n / len(rows),
                'confirmed_better_fraction': better / n if n else None,
                'worst_case_net': better - worse - uncertain,
                'eligible_ids': [rows[i]['id'] for i in selected],
            })
        feasible = [x for x in candidates if x['accepted'] >= report['minimum_accepted']
                    and x['confirmed_better_fraction'] >= report['minimum_confirmed_better_fraction']
                    and x['worst_case_net'] >= report['minimum_worst_case_net']]
        chosen = max(feasible, key=lambda x: (x['worst_case_net'], x['accepted'],
                                              x['confirmed_better_fraction'], x['margin'])) if feasible else None
        report['models'][arm] = {
            'checkpoint': str(checkpoint), 'temperature': temp,
            'status': 'CALIBRATED' if chosen else 'NO_FEASIBLE_MARGIN',
            'margin': chosen['margin'] if chosen else None,
            'selected': chosen, 'grid': candidates,
        }
        dump(MODEL / arm / 'safe_acceptance_v1.json', {
            'rule': report['rule'], 'selection': report['selection'],
            'temperature': temp, 'checkpoint_hash': actual_hash,
            'status': report['models'][arm]['status'],
            'margin': report['models'][arm]['margin'],
            'fold_hashes': report['fold_hashes'],
            'calibration': chosen,
        })
    dump(OUT / 'report.json', report)
    print(json.dumps({a: {'status': x['status'], 'grid': x['grid']}
                      for a, x in report['models'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
