"""Evaluate direct binary decisions; optional temperature affects diagnostics only."""
import argparse
import json
from pathlib import Path
import numpy as np
import laya_acceptance_common as C


def fit_temperature(logits, rows):
    C.require_changed_pairs(rows)
    from scipy.optimize import minimize_scalar
    y = np.array([C.semantic_target(r['label']) for r in rows])
    def loss(log_t):
        p = C.probabilities(logits, float(np.exp(log_t)))
        return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean())
    result = minimize_scalar(loss, bounds=(np.log(.1), np.log(10.)), method='bounded')
    if not result.success:
        raise ValueError('Diagnostic calibration failed')
    best = min([result.x, 0., np.log(.1), np.log(10.)], key=loss)
    return {'temperature': float(np.exp(best)), 'nll_before': loss(0.), 'nll_after': loss(best),
            'used_for_decisions': False}


def evaluate(checkpoint, data, fold, output, device='cpu', diagnostic_temperature=1.0):
    cfg = C.validate_checkpoint(checkpoint)
    meta = json.loads((Path(data) / 'dataset.json').read_text())
    if cfg.get('label_feedback') != meta['label_feedback']:
        raise ValueError('Evaluation labels must use the frozen training feedback definition')
    if fold == 'test' and meta.get('frozen_checkpoint_sha256') != C.digest(Path(checkpoint) / 'model.safetensors'):
        raise ValueError('Test labels must have been prepared after freezing this exact checkpoint')
    if Path(output).exists():
        raise FileExistsError('Evaluation output must be new')
    rows, delta, _ = C.load_fold(fold, data)
    tok = C.get_tokenizer(checkpoint)
    model, _ = C.load_model(checkpoint)
    model.to(device)
    logits = C.infer(model, C.EncodedPairs(rows, tok))
    report = C.metrics(rows, logits, delta, diagnostic_temperature)
    report.update(fold=fold, checkpoint_sha256=C.digest(Path(checkpoint) / 'model.safetensors'),
                  no_op_pairs_excluded=meta['statistics'][fold]['no_op_pairs'],
                  effect_validated=False, note='Classification audit, not end-to-end effectiveness evidence')
    C.dump(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=C.OUT / 'best')
    p.add_argument('--data', type=Path, default=C.DATA)
    p.add_argument('--fold', choices=['development', 'temperature_calibration', 'decision_diagnostics', 'test'], default='development')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--diagnostic-temperature', type=float, default=1.0)
    args = p.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.data, args.fold, args.output, args.device,
                              args.diagnostic_temperature), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
