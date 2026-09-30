EvoScope protocol v3 — sampling, evidence and validation repair

The action core, fixed key+do retrieval, persistent expose/mask intervention,
400-character when limit, paired repeats and strictly-positive pooled gate
criterion are unchanged. This version changes experimental protocol; do not pool
its results with v2. Existing runs are not resumed or modified by this change.

**Defaults and entry points**

| Control | `run` CLI | `local_expanded` |
| --- | --- | --- |
| Minimum distinct valid states | 3 | 3 |
| Minimum stable helpful/harmful states | 1 | 1 |
| Total Probe attempts | 24 | 12 |
| Lifetime Probe attempts per policy | 6 | 6 |
| Gate local/global groups per candidate | 2 / 2 | 2 / 2 |
| Maximum editor slots per policy | 2 | 2 |
| Probe / Gate paired repeats | 3 / 3 | 3 / 3 |

CLI controls: `--min-edit-checkpoints`, `--min-non-neutral`, `--total-probes`,
`--probes-per-policy`, `--gate-size`, `--gate-local-size`,
`--max-candidates-per-policy`. Low-level unit fixtures explicitly use a one-state
minimum to test individual operations; real entry points default to three.
`probe-only` remains frozen and retains its original 24/3 budget defaults.

**Candidate-independent Gate**

`gate_plan.py` resets only reserved Gate tasks and reads public initial observations.
It never reads hidden payload fields for ranking, calls an actor, uses rewards,
or sees a proposed when. The fixed initial key+do BM25 score ranks local tasks;
global tasks are sampled from remaining groups using the recorded random seed.
One representative per group is chosen by task ID. A deterministic round-robin
allocation reserves disjoint blocks for policy/attempt slots before learning.
The plan, public observations, scores, seed and plan hash are persisted.

Initial lexical relevance is a proxy, not a promise of later exposure. A plan
shortage prevents that editor slot and is logged. An allocated block is never
replaced based on outcomes. Noop/error editor slots are not recycled; Gate groups
are never reused. WebShop's current 10 groups permit at most two four-group
blocks, possibly fewer if local relevance is insufficient.

Gate receipts record per-pair old/new retrieval and prompt-exposure counts,
local/global totals and mean deltas. Exposure means the policy was present in
the actor input, not proof that the actor applied it. If the entire local block
has zero exposure on both sides, status is `coverage_insufficient`; no update is
accepted even if noise yields a positive score delta. Asymmetric exposure is
allowed because trajectories may diverge. No zero-coverage task is dropped from
the aggregate. Errors remain errors. The original equal-task-weight pooled
mean must be positive; global differences are reported, not a guarantee of no
regressions and not an added non-inferiority test.

**Cross-round sampling**

Counters are keyed by policy ID/version. Choose the less-sampled available
success/failure stratum; ties follow a fixed alternating rule. Fill shortages
from the other stratum. Count when a Probe attempt begins, before its effect is
known; failed attempts consume quota too. Changing another policy does not
reset the target's sampling counts. Updating this policy creates a new counter.
Global and per-policy compute budgets remain lifetime limits, preventing reset
of compute budgets on each version change.

**Multi-state editing**

The append-only evidence archive is the buffer. Editor input selects only the
current policy/version, exact bank hash, model fingerprint, runner settings and
compatible environment protocol. When any policy changes, previous-bank evidence
remains archived but is excluded from future editing. Duplicate public states and
multiple states from the same task group do not inflate the threshold; choose the
first eligible entry without looking at its sign. Neutral and unstable evidence
are retained, but only stable helpful/harmful counts toward the nonzero threshold.
Failed probes have separate records and are not treated as valid states.

An editor slot requires three distinct states and one stable nonzero state by
default. Evidence-too-small is logged and does not call the editor. One state-set
signature can trigger at most one editor call, including noop/invalid output/error.
The editor sees only policy and public learn evidence, never Gate/test material.
Additional evidence can justify a later slot, bounded by the precommitted plan.

**Validation and limits**

The regression suite covers cross-round one-slot balance, shortages, failed
attempt accounting, policy version resets, cross-bank/model isolation, duplicate
state/group rejection, neutral retention, no repeated editing, precommitment,
public-goal planning, immutable when-independent task selection, task-substitution
rejection, positive-score-but-unexposed rejection, and multi-state accepted repair.
Toy accepted repair validates plumbing only, not research effectiveness.

This repair does not redesign the normal/expose/mask estimand, regenerate M0,
change endpoint settings, or fix actor JSON/context-budget issues. Those remain
separate investigations. Real public-observation preflight uses CPU environment
resets only and does not call any model or load GPU weights.

**2026-09-26 verification**

102 offline tests passed; the CLI toy demo completed aggregation, a paired Gate
and one accepted repair. No real-model generation was used. CPU preflight on
the existing manifests allocated four ALFWorld blocks. WebShop allocated two
search-policy blocks, but no size-policy block: none of the ten initial public
queries has a positive score for that policy under the fixed literal BM25.
This is recorded as a planning shortage, not proof that the policy is irrelevant
at every later state. Synonym handling or a different reserved validation split
would be a separate predeclared change; no task split was changed here.

A diagnostic that loaded both native environments in one process completed its
plan files but crashed at JVM teardown. Its native log was retained and that
CPU process was cleaned up. A separate WebShop process exited normally; actual
experiments already isolate datasets in separate workers. All verification
artifacts are in `results/protocol-v3-preflight-20260926/`.

The isolated ALFWorld preflight also saved the same plan hash but remained alive
after completion, reproducing the existing native-worker shutdown problem. Its
CPU-only process was explicitly terminated. Thus real environment reset/planning
was verified, but clean ALFWorld interpreter teardown is not claimed. The ongoing
old experiment retains its existing completed-worker cleanup watcher.
