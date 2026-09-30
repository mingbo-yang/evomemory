# Acceptance training data v2

Mixed sampling was explicitly approved by the user after the v1 pilot showed extensive repetition. This version is a separate collection; do not concatenate v1 and v2 as independent examples because their source partitions overlap by design.

## Collection

- Qwen3-8B, EN→ZH, frozen clean initial memory (1,982 entries), existing hybrid retrieval and translation prompts. No BERT, Laya, or online library update is involved in collecting these trajectories.
- Draft temperature 0.1. Two states, four candidates per state. Training candidate temperatures are [0.1, 0.1, 0.7, 0.7]; development, probability calibration, threshold calibration and final test use [0.1, 0.1, 0.1, 0.1]. top_p=1.0, output budget=1,024, context budget=4,096.
- Choose the second state uniformly among basic-valid first-state candidate slots; never use reference feedback, BERT/Laya, or an improvement requirement. Identical candidates remain eligible. Record every sampled slot, including duplicates and invalid output.
- Basic-valid: nonempty, finished with stop, Chinese output without thinking/fences, <=1.5×previous normalized length+8. This is not the final acceptance gate. Context overflow is flagged; source text is not silently truncated.
- Deduplicate exact (source, before, candidate) tuples within each partition for the exported pair table. Keep all raw occurrences and per-slot temperatures separately. Count only valid unique pairs toward the training target; leave epsilon and Better/Tie/Worse labels unset until training-only feedback inspection and human review.
- Compute the existing sentence feedback proxy /100 after complete trajectory generation. Query references do not enter generation or state selection. This proxy is distinct from corpus BLEU and from a human semantic-quality label.

## Data allocation

| Partition | Source reserve | Purpose |
|---|---:|---|
| Training | 10,000 | Stop after at least 20,000 unique valid decision pairs; reserve may not all be used |
| Development | 400 | Checkpoint/model selection |
| Probability calibration | 256 | Temperature scaling |
| Threshold calibration | 256 | Acceptance threshold |
| Test | 400 | Hold out; collect raw trajectories without scoring or exporting feedback |

All trajectories are assigned by source before generation. Source AND reference normalized with NFKC/casefold/whitespace removal are disjoint across partitions. Exclusions cover previous auxiliary/main manifests, historical BERT sources, the previous 1,536-source study, and initial-memory source/before/after strings. Exact normalization does not exclude unknown paraphrase or pretraining overlap.

## Evidence and artifacts

The v1 pilot completed 128 sources / 1,024 raw pairs, but only 250 unique valid pairs, of which 130 changed the current answer. It is a sampling-efficiency diagnostic, not a verifier result.

`runs/acceptance_data_v2/protocol.json` contains frozen input hashes and sampling settings. `raw/<fold>/<row>.json` stores complete trajectories atomically for resume. `pairs/<fold>.jsonl` stores unique triples, validity and delayed numeric feedback; do not feed feedback columns into the verifier. `train_margin_review.json` samples non-identical training pairs in four absolute-delta bins with reference text for human review only. `audit_summary.json` checks partition/memory isolation, before/after state continuity, per-slot seeds/temperature and randomized selection. `progress.json` reports live progress.

Before training: inspect invalid counts, duplicates, exact identities, both feedback signs, and the human margin-review set. Choose epsilon without using the held-out test. Decide training class weighting using training data only; preserve natural calibration/test distributions. Training on mixed-temperature candidates increases diversity but does not prove quality on the low-temperature inference distribution—that remains the job of the untouched development/calibration/test partitions.

```bash
# GPU 0; from aaai2027/exp in the qwen35 environment
python acceptance_data.py --folds train --limit 128
python acceptance_data.py --folds train development temperature_calibration threshold_calibration test
python acceptance_data.py --audit-only --folds train development temperature_calibration threshold_calibration test
```

No verifier training or production inference/memory-admission changes are part of this data collection.

## Completed mixed-temperature pilot

128 sources, 1,024 raw candidates; 382 unique valid pairs, including 260 actual changes. No invalid draft/candidate; zero cross-partition and initial-memory exact normalized overlap; trajectory/seed/temperature/selection audits PASS. Raw feedback signs (not final epsilon labels): 93 positive, 143 negative, 146 zero. The 4 invariant tests passed. Compared with the low-temperature pilot, unique valid pairs increased from 250 to 382 and actual changes from 130 to 260. This validates collection diversity, not verifier quality.

Full collection has been launched on GPU 0; inspect `runs/acceptance_data_v2/job_status.json` and `progress.json` for its actual state. Completion requires the unique-valid training target and all held-out partitions; categorical labels remain pending margin review.
