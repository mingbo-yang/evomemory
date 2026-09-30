# Acceptance training data v1

Status: pilot generation on GPU 0; no verifier training started.

## Frozen collection design

- Qwen3-8B, EN→ZH, existing prompts and hybrid retrieval; no BERT/Laya gate.
- Fixed clean initial memory of 1,982 experiences; no online memory writes.
- One initial draft, four candidates from the initial state, then four candidates from a uniformly random basic-valid first-stage candidate slot. Identical candidates remain eligible. If none qualify, log the source without inventing a second state.
- Sampling remains temperature 0.1, top_p 1.0, output budget 1,024, total context 4,096. Distinct deterministic seed for each source/state/slot. This sampling budget is for data collection only.
- Basic-valid means nonempty, generation finished with stop, no thinking/fenced output, contains Chinese, and length <= 1.5 × previous normalized length + 8. Invalid candidates are retained and flagged. This gate is not a feedback or quality filter.
- Every raw candidate and prompt is retained; exact (source, before, candidate) duplicates are collapsed in the exported table. References are absent from generation records and from the state-selection function.
- Existing sentence feedback proxy is computed after generation and divided by 100. This is reference-derived benchmark feedback, not a human semantic-quality label or corpus BLEU.
- No epsilon or categorical labels are frozen yet. Human review uses training-only delta bins; no test feedback distributions are inspected.

## Data partitioning

Before generation, split by normalized source/reference, keeping every trajectory within its original partition. Exclude previous auxiliary/main manifests, previous 1,536-source BERT study, historical BERT sources, and all initial-memory source/before/after strings. Exact normalized overlap guards do not establish absence of paraphrase or unknown pretraining overlap.

| Partition | Source reserve | Use |
|---|---:|---|
| train | 10,000 | Stop after at least 20,000 unique valid decision pairs, subject to reserve exhaustion |
| development | 400 | Model/checkpoint selection |
| temperature_calibration | 256 | Probability calibration |
| threshold_calibration | 256 | Acceptance threshold |
| test | 400 | Final held-out evaluation; save raw trajectories without scoring/exporting for now |

Reserve is not the number already generated. The first 128 training sources are a pilot and remain in train if the protocol is unchanged. Changing sampling or filtering after the pilot requires a new version; do not silently combine distributions. All raw duplicates and invalid samples remain available for audits; include only valid rows in acceptance training, and report exact-tie frequency rather than hiding it.

## Artifacts and continuation

`runs/acceptance_data_v1/protocol.json` pins manifest, initial-memory and retrieval-config hashes. `raw/<fold>/<row>.json` is written atomically after each complete source trajectory. Resume skips completed source files. `pairs/<fold>.jsonl` contains deduplicated input triples and feedback, without reference text. `train_margin_review.json` contains 20 non-identical training cases per delta bin (where available) with references for human inspection, not model input. `audit_summary.json` checks partition/memory overlap and actual state transitions. `progress.json` is live progress, not proof that the whole job completed.

Commands (from `aaai2027/exp`, environment `qwen35`):

```bash
python acceptance_data.py --folds train --limit 128
python acceptance_data.py --folds train development temperature_calibration threshold_calibration test
python acceptance_data.py --audit-only --folds train development temperature_calibration threshold_calibration test
```

No new model, inference acceptance rule, or memory-admission behavior is deployed by this collector.
