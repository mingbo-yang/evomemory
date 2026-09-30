# EvoScope reliability protocol v2

## Changes

- Editor inputs use deterministic, outcome-blind evidence compression (`editor-budget-v1`). First try lossless trace deduplication. If over 16,000 UTF-8 bytes, keep each checkpoint's beginning/end and each branch's first divergence/end, clip text using a fixed descending budget, and explicitly mark omissions. All evidence records, expose/mask pair scores, done flags, effect signs, and mean deltas remain. Missing information is unknown, not evidence of absence. Original evidence remains in probe/evidence.jsonl.
- Local editor requests are tokenized by the serving model before generation. Input + unchanged output budget + 512-token margin must fit the server context. An oversized compact payload is rejected explicitly; no pair is silently dropped and no output limit is changed. Budget logs contain original/compact byte lengths, source hash and exact token count. API parse failures preserve the raw response in invalid_outputs.jsonl.
- Expanded evaluation records a failed episode and continues the remaining arms/tasks, without retry or task replacement. Learn errors also do not abort the entire evaluation. Existing probe eligibility excludes errored episodes; gate errors still reject candidates. Error counts and success with errors treated as failures remain reported.
- Bounded evolution assigns probe slots across the full learning sequence before observing outcomes, respecting total/per-policy limits. Unavailable slots are logged and left unused. For 100 learn tasks, batch size 4, two policies and limits (6,3), rounds 0,1,11,12,23,24 receive one slot each. Total budget is not increased. Standalone fixed-M0 probe-only behavior is unchanged.

Core method remains unchanged: fixed key/do, condition-only proposals, independent-group gate with paired repeats, positive gate delta required for acceptance. Compression may omit useful context, so this is a versioned protocol change; v1 and v2 results must be analyzed separately.

## Validation

87 offline tests pass, including signed-pair preservation, deterministic compression, exact token guard, raw malformed output capture, continued comparison after a failed episode, and early/middle/late budget coverage.

Three real local editor-only replays of preserved failed evidence succeeded:

| Model | Original bytes | Compact bytes | Actual input tokens |
|---|---:|---:|---:|
| GLM-4.7-Flash |125047|11911|4283|
| Qwen3.8-27B |78025|3135|1567|
| Qwen3.5-9B |76718|3496|1752|

Output budget stays 8192 and context stays 32768. GLM returned its original condition; Qwen models returned condition candidates. These are editor plumbing checks, not gate acceptance or effectiveness results. No experiment bank was modified by replay. Replay files: results/reliability-v2-editor-replay-20260925/.

## Deployment

Existing model/task processes were not restarted. The newly started GLM expanded run already records v2 in its protocol.json and probe/schedule.json. The active Qwen v1 runs continue; separate unlimited v2 runs are queued after their verified cleanup. See results/reliability-v2-inputs-20260925/runs.json. Local-only model use and owned-server cleanup remain enforced.
