# Laya multilingual acceptance: training and validation

Base: convaiinnovations/laya-multilingual; selected epoch 4; test 793 pairs from 400 sources.

Training labels use the original weak-feedback proxy, epsilon 0.01. This evaluation measures that proxy objective, not independently annotated semantic correctness.

| Model | Better AUC (changed) | Macro F1 | Brier | ECE | Accepted | Coverage | Precision | Mean delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.4793 | 0.2404 | 0.6616 | 0.0856 | 0 | 0.0000 | undefined | undefined |
| finetuned | 0.5518 | 0.5301 | 0.3715 | 0.0700 | 0 | 0.0000 | undefined | undefined |

Verifier gate passed: False. Next stage: verifier_failed_do_not_run_main_experiments.

Temperature fitting and operating-point selection used separate calibration folds. No feasible threshold means acceptance disabled and a failed gate, not successful accuracy. Default pBetter >= 0.5 results, raw scores, source-bootstrap confidence intervals, exact thresholds and option-order diagnostics are retained in evaluation_report.json.

No online memory experiment is justified by classification metrics alone. If this verifier gate passes, a real fixed-memory rollout remains required before full_static/full_online comparison.

Official training source: https://github.com/NandhaKishorM/laya ; base model: https://huggingface.co/convaiinnovations/laya-multilingual .

Checkpoint: /mnt/huawei/ymb/model/laya-multilingual-acceptance-enzh-v1/best
