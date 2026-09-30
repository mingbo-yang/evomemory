# Independent audit — batched refinement pipeline vs the stated design

Auditor: delegated adversarial audit subagent. **Read-only**: nothing under `/mnt/huawei/ymb/aaai2027/exp` was created, modified or deleted. Scratch scripts live in `/tmp` only.

Artifacts pinned for this audit (sha256, first 20 hex):

| file | sha256[:20] |
|---|---|
| `core/batched_pipeline.py` | `ec26248732f16c12495f` |
| `core/pipeline.py` | `95792c8936b320cefbb1` |
| `core/controller.py` | `4608963a404670505481` |
| `core/judge.py` | `be0815759e27f242b167` |
| `core/determinism.py` | `5624861be097a91acdc5` |
| `core/experience.py` | `1ed56a1ce82b1ce84a17` |
| `core/batch_llm.py` | `c9a023c0f3d1ecfa9281` |
| `run_experiment.py` | `d1bd3641776b93b6e037` |
| `tests/test_invariants.py` | `b161214f29c4f5a97494` |

Data snapshot: `/mnt/huawei/ymb/aaai2027/exp/runs/main/full_static/seed42/*/*/full_static.jsonl` copied at **2026-09-12 04:16 EDT** to `/tmp/snap/` (16-hex sha256 of each copy in §5). Files are **live and being rewritten** — see F8.

---

## 1. Task item 1 — unit tests

`/home/ymb/miniconda3/envs/qwen35/bin/python tests/test_invariants.py` → **exit 0, 29/29 checks PASS, `RESULT: PASS (all invariants hold)`**.

Caveats (see F12): the file never imports `core/batched_pipeline.py` or `core/batch_llm.py` (verified by grep), and 3 of the 29 checks are tautological.

---

## 2. FINDINGS

### F1 — CONFIRMED BUG — the batched controller has no repair retry; a malformed reply becomes an immediate STOP

* Contract (authoritative in code): `core/controller.py:9-11` — *"A malformed reply gets exactly **one** repair retry; if that also fails the decision is recorded as a structured failure and the current answer is kept."* Implemented at `core/controller.py:176-221` (repair prompt appended at 178-180, second call with `call_type="controller_repair"` at 181-188, final STOP reason `"structured output failed after one repair retry"` at 214-221).
* Batched: `core/batched_pipeline.py:245-248`

  ```python
  obj = _extract_json(g.text)
  if not obj or "action" not in obj:
      decisions[id(s)] = ("STOP", "", "structured output failed", True)
      continue
  ```
  No second call, no repair prompt, reason string changed.

* **Data proof that this fired in the shipped runs**: over the snapshot's **3 940 rounds / 1 768 samples**, `controller_structured_failure == true` in **13** rounds (llama3.1-8b/en_zh **12**, qwen3-8b/zh_en **1**), and every one is a STOP round. Per-round `n_calls` deltas are **exactly +1 for all 715 STOP rounds and +4 for all 3 225 REFINE rounds** — i.e. the repair call was *never* made (0 anomalies).
* **Contrast (unbatched path really does retry)**: in the pre-batching gate_alpha traces, `n_calls` exceeds the same formula in 3 rounds (`runs/gate_alpha/fixed_rounds/seed42/wmt19_en_zh/glm4-9b/dev/a1.0` ×2, `.../wmt19_zh_en/glm4-9b/dev/a0.0` ×1) with `controller_structured_failure == false` — the retry happened **and succeeded** there.
* Impact: a systematic push toward STOP in every batched arm. The paper's "When" claim (progress.md §1, §3) is built on the STOP/REFINE split and on the fact that the first controller prompt drove STOP to 98.3 %; a path that silently converts parse failures into STOPs moves exactly the quantity under study. It also makes n_calls/cost non-comparable to the unbatched dev-gate, α-selection, smoke and stopping-diagnostic traces.

### F2 — CONFIRMED BUG — the batched judge has no repair retry either

* Unbatched: `core/judge.py:161-196`, comment at `core/judge.py:166` — *"exactly one repair retry, per the plan"* — called twice per round from `core/judge.py:215-222`.
* Batched: `core/batched_pipeline.py:345-349`

  ```python
  obj = _extract_json(g.text) or {}
  vraw = str(obj.get("verdict", "uncertain"))
  ```
  A parse failure therefore maps to `uncertain` (`_map_verdict` → `uncertain`, `_combine` → `uncertain`), which can never be accepted, with no recovery attempt.
* **Data evidence (suggestive, not conclusive)**: in the llama3.1-8b/en_zh trace, **331 / 1 562 REFINE rounds (21.2 %)** have at least one empty `judge_reason_*` and **all 331 are recorded `judge_verdict == "uncertain"`** (272 one-sided, 59 both-sided); qwen3-8b/zh_en has 2/1 521. That is the exact signature of the no-repair fallback, but the raw judge text is not stored in `RoundTrace`, and a *parsed* reply `{"verdict":"uncertain"}` with a missing reason would look identical — so the split between "unparseable (repairable)" and "parsed-but-empty-reason" is **UNVERIFIABLE from the trace**. What is confirmed is that a repair was never attempted (n_calls accounting, F1) and that the unbatched path's repair demonstrably succeeded in 3 pre-batching rounds.
* Minor: `JudgeVerdict.structured_failure` (`core/judge.py:77,234`) is computed but never persisted in either path (`RoundTrace` has no such field), so judge parse health is not auditable from traces.

### F3 — CONFIRMED BUG (latent, silent) — the `full_online` arm loses all online writes under batching, and the default *is* batching

* `run_experiment.py:130` computes `online = bool(spec["online"])`; `:144` sets `snapshot_id="online"`.
* Batched branch `run_experiment.py:157-196` **never references `online`**: no `transitions_to_experiences`, no library growth, no `online_seed{seed}.jsonl` (that save exists only at `run_experiment.py:249-250`, in the unbatched branch after the loop).
* The batched branch is the default: `run_experiment.py:333-334` (`--batch-size` default **32**) and `supervisor.py:108,168` (also 32). So `--arm full_online` through the supervisor runs behaviourally identical to `full_static` while its config/`snapshot_id` claims online accumulation — and the read-only concern that "an item's own experience can never influence its own decisions" is never exercised.
* Not yet triggered: `run_ablation_chain.sh`'s `ARMS` does not include `full_online`. But `run_accumulation.py` (new, 04:08) uses `Pipeline` (unbatched) for the stream and `BatchedPipeline` for the checkpoint evaluations (`run_accumulation.py:222-235` vs `:503`), so the two passes run under different decoding/repair regimes even though its docstring claims *"the round budget and the batch size are all held fixed"* (`run_accumulation.py:23`).

### F4 — DESIGN DIFFERENCE — `latency_s` is not the same quantity in the two paths

* Unbatched: `baseline/core/llm.py:138,168-173` — wall time of that one call.
* Batched: `core/batch_llm.py:106-124` — whole-batch wall time divided by batch size, attributed identically to every request.
* Tokens and `n_calls` have identical definitions (`Generation.total_tokens = input+output`, `baseline/core/types.py:19-21`, so the unbatched `total_tokens` at `pipeline.py:446` and the batched one at `batched_pipeline.py:242` agree). **Only latency is incomparable** — it must not be used for cross-path or "cost per sample" claims. It is also not the baseline's per-call latency.

### F5 — DESIGN DIFFERENCE / latent bug — `BatchedPipeline` supports only `draft_source="stored"`, and silently falls back to an **empty y0**

* `core/batched_pipeline.py:93-98` builds `StoredDraftSource` only for `"stored"`, otherwise `self.drafts = None`; `:176` then does
  `current, from_cache = self.drafts.get(idx) if self.drafts is not None else ("", False)`.
  That is not "generate the draft" — it is **`y0 = ""`**, silently.
* The CLI is protected (`run_experiment.py:149-150` raises for `batch_size>1 and draft_source != "stored"`), and `run_accumulation.py:207-214` passes `"stored"` with the *test* prefix, which is positionally valid. But `BatchedPipeline` itself has no guard, and the consequence today is that **dev runs (which must use `generate`/`cached`) cannot be batched at all** — so batched test results and unbatched dev results are produced by different code paths, including the F1/F2 repair difference.

### F6 — DESIGN DIFFERENCE — frozen prompt/decision logic is duplicated, not shared (single source of truth broken)

* Controller system prompt: `core/batched_pipeline.py:228-233` is a hand-copied f-string instead of `Controller._SYSTEM` (`core/controller.py:36-40`).
* Judge system prompt: `core/batched_pipeline.py:311-316` duplicates `PairwiseJudge._SYSTEM` (`core/judge.py:28-33`).
* Action normalisation: `core/batched_pipeline.py:250-257` duplicates `core/controller.py:224-227`.
* I verified the copies are **byte-identical today** for all 4 tasks (script compared `_SYSTEM.format(...)` against the batched literals; all `True`), and the controller prompt is built through the shared `Controller.build_prompt`. But `frozen.json` freezes the controller prompt; a future edit to `controller.py` will silently not reach the batched path, which is where all main results come from.

### F7 — DESIGN DIFFERENCE / reasoned risk (NOT proven here) — in the batched path the ablation arm feeds back into the sampling conditions

* `core/batched_pipeline.py:162-164` fixes chunks of 32; `:179-183` builds `active` per round from the samples that have not STOPped. Therefore a sample's own STOP decisions change the *number and identity* of the other requests decoded in the same engine call at later rounds.
* Consequence: `Full` and `OutcomeHidden` (and `RandomRetrieve`) will have different active sets after their first divergent STOP, so byte-identical prompts are decoded under different batch compositions. The project's own measurements say vLLM is not bit-deterministic (`reports/progress.md` §B3 / `probe_determinism.py`: same prompt+seed, 5 repeats, 2 distinct outputs), and batch composition is a standard cause of that.
* Two further asymmetries: the last chunk of a 1 000-item task is short (31×32 + 8), so those 8 samples are always decoded in a smaller batch than the rest; and `batch_llm.py:60-73` routes any phase with exactly one request through the single-request baseline path, so **a single run mixes batch sizes**.
* I could not test this without a GPU replay: **UNVERIFIABLE** in this audit. It does not invalidate the invariants as *implemented* (F9), but it means "single-variable" holds at the prompt/seed level, not at the sampling level, and the extra coupling channel is systematic rather than random.

### F8 — CONFIRMED (data integrity / reproducibility, not a code bug in the pipeline itself) — the "completed runs" are neither complete nor stable, and a summary on disk contradicts its own trace

* At the 04:16 snapshot: **qwen3-8b/zh_en 1 000/1 000 (the only complete, self-consistent run)**; llama3.1-8b/en_zh 576/1 000; qwen3-4b/en_zh 64/1 000; qwen3-8b/en_zh 64/1 000 (it had 1 000 lines ~03:56, then 480, then 64 — it is being rewritten); coedit_gec 64/1 000. No ablation arm has produced data yet (`runs/ablation` does not exist; `runs/accumulation` and `runs/stopping` are empty).
* Concretely observed contradiction at ~04:20: `runs/main/full_static/seed42/wmt19_en_zh/qwen3-8b/full_static.summary.json` says `n=1000, duration_s=575.86` (mtime 03:42:06) while the sibling `full_static.jsonl` had **480** lines (mtime 04:06:30) — the summary is from a previous attempt.
* Root cause: each attempt opens the JSONL/CSV with `"w"` (`run_experiment.py:166-168`, `:207-209`), so a supervisor retry **truncates and re-runs from item 0**, and the summary is only written after a task finishes; `supervisor.py:144-150` kills stalled attempts (observed `exit=-9` at 04:15 in `logs/sup_gpu0.log`). There is no attempt id in the output path, no resume, no atomic write.
* Downstream risk: `aggregate.py:55-62` rglobs every `*.jsonl` with **no minimum-n guard and no check that arms/models have equal n**, so a 64-sample prefix and a 1 000-sample prefix are silently pooled into one corpus metric row. Any table produced before all arms finish is not reproducible from the artifacts.
* Also: data already reported as final (e.g. the 1 000-sample qwen3-8b/en_zh summary `mean_initial 37.74 → mean_final 37.71`) no longer matches the trace on disk.

### F9 — VERIFIED OK — the batched path does reuse the shared retrieval / renderer / A-B / seed / y0 machinery

Verified by reading *and* by byte-level comparison where possible:

| property | unbatched | batched | result |
|---|---|---|---|
| retrieval (cache key, random quota profile, `deterministic_random_rank` seed key) | `pipeline.py:312-379` | `batched_pipeline.py:101-140` | identical logic (line-for-line copy) |
| experience block render (`include_outcome`, `count_tokens`, `max_units=k`) | `pipeline.py:421-426` | `batched_pipeline.py:208-213` | identical args/helper |
| controller prompt | `controller.build_prompt` via `pipeline.py:434-442` | `batched_pipeline.py:225-227` | byte-identical (verified) |
| controller sampling | `controller.py:164-171` (T=0, 256 tok) | `batched_pipeline.py:234-236` (T=0.0, 256 tok) | identical |
| refine prompt + sampling | `pipeline.py:485-494` (T=0.1, top_p=1.0) | `batched_pipeline.py:291-296` | identical helper, identical params |
| A/B order + judge seeds | `judge.py:212-222`, `determinism.py:21-40` | `batched_pipeline.py:319-334` (`call_seed(sample_id, t*10+call_idx, "judge")`) | identical function, identical args |
| acceptance rule | `judge.py:80-81,224-227` | `batched_pipeline.py:355-357` | `verdict=="better" and order_consistent` |
| offline fields in prompts | `experience.py:21-33,85-108` whitelist; `RoundTrace` fields never rendered | same helpers | structurally impossible; `metric_offline`/`delta_offline` are computed only *after* the decision (`batched_pipeline.py:358-359`) |
| task prompt reference-freedom | `baseline/core/tasks.py:47-56` — `initial_prompt` uses only `example.source` | same adapter | no reference reaches controller/refine/judge prompts |
| y0 | `StoredDraftSource` positional read, `pipeline.py:194-222` | `batched_pipeline.py:93-96,176` | arm-independent; **0 / 1 768 mismatches** vs the stored `Direct-Zero.csv` rows |

### F10 — VERIFIED OK (data) — chain, retrieval replay, verdict/acceptance consistency, cost accounting

Recomputed independently from the snapshot with the shared `ExperienceRetriever` / `Scorer` (scripts in `/tmp`):

* **Retrieval replay: 0 mismatches in 3 940 rounds.** For every round I reconstructed `current` (y0, then candidate whenever `accepted`), re-ran `retrieve(source, current, alpha=0.5, k=4)` against the current library, and compared with the recorded `exp_ids`. All match ⇒ (i) accepted ⇒ the candidate really became the next `current`; (ii) the recorded experience ids really are the ones the shared helper produces at that state; (iii) no stale retrieval-cache entries are in play despite the cache key containing no library fingerprint (`retrieval_cache.py:36-67`).
* `final_output == reconstructed current` for **1 768 / 1 768** traces; no round follows a STOP; no trace exceeds `max_rounds=3`.
* `accepted == (judge_verdict=="better" and judge_order_consistent)` in **all** rounds; `verdict=="uncertain" ⇒ order_consistent==false`; `verdict=="worse" ⇒ order_consistent==true`.
* `metric_offline`, `delta_offline`, `initial_metric_offline`, `final_metric_offline` all recomputed exactly (0 mismatches), including `delta = metric(candidate) − metric(previous state)` — an independent confirmation of the chain.
* Cost: monotone non-decreasing in every component in every round (0 decreases, including `latency_s`); `final cost == last round's cost` for all traces; `n_calls` delta exactly +1 (STOP) and +4 (REFINE) — 0 exceptions, which is also the F1/F2 evidence.
* `exp_ids`: 3 939 / 3 940 rounds non-empty; all ids ∈ library; `exp_class_counts` equals the verdict composition of the returned ids; 0 duplicates inside a round. The single empty case is legitimate (wmt19_zh_en/test/000597, source `北京倡议`, y0 `Beijing Initiative`: BM25 finds no shared token with the 96-item library) — meaning the Full arm silently degenerates to NoExperience on such items; not a batching artefact (the retriever is shared).
* exp_id sets do vary: 59/64, 61/64, 247/576, 480/999, 53/64 distinct round-0 id sets per file.

### F11 — VERIFIED OK (data) — offline-only fields do not drive decisions; the reference does not leak

* **Acceptance vs the offline metric** (`delta_offline` is the only reference-derived decision-adjacent quantity, and it is stored, not read): AUC of `delta_offline` for accepted vs rejected rounds ≈ **0.5** — qwen3-8b/en_zh 0.507, llama3.1-8b/en_zh 0.499, qwen3-8b/zh_en 0.576, qwen3-4b/en_zh 0.339 (n_accepted = 2). Mean `delta_offline` of accepted rounds is *worse* than rejected in two of the three meaningful files (llama: −3.32 vs −2.75; zh_en: −1.62 vs −2.25). An arm that accepted the offline-best round would give AUC ≈ 1.0. **Accepted rounds are not systematically the offline-best rounds.**
* **Reference in stored text**: no 40-character verbatim substring of the reference appears in any `controller_instruction`, `judge_reason_*` or `candidate` for the three en_zh/gec files. For zh_en, 402/1 521 candidates and 44/1 521 instructions do contain one — but **392/402 and 44/44 respectively are already present in the `current` state**, i.e. they are inherited, not leaked. The origin is the stored draft: **289/1 000 y0 drafts already contain a ≥40-char reference substring and 11/1 000 y0 *are* the reference verbatim.** Any "reference in prompt" test based on substring overlap is therefore non-diagnostic on this dataset; reference-freedom rests on the code whitelist (F9).
* **Renderer on real data**: re-rendering the recorded `exp_ids` with `include_outcome=True/False` gives, in 3 939/3 940 rounds, blocks that are identical after deleting the `Outcome:`/`Judge note 1:`/`Judge note 2:` lines, and the hidden block never contains `Outcome:`; no `delta_offline`/`provenance`/`exp_id`/`reference_text` token ever appears in a rendered block. (The 1 exception is the empty-retrieval round, where both are empty.) The experience token budget never truncated a unit (`truncated_units = 0`), so the tokeniser choice does not affect this result.
* **Library isolation**: for all four libraries, 0 library `source_input`s equal a test source, and no whole test reference (≥40 chars) occurs inside any library field.

### F12 — VERIFIED OK, but the test suite does not test what F1/F2/F3 break, and 3 checks cannot fail

* `tests/test_invariants.py` passes 29/29, but it imports only `bm25_fields`, `determinism`, `experience`, `retrieval_cache` — **no reference to `batched_pipeline`/`batch_llm` anywhere in `tests/`**.
* Tautological checks (they compare a pure function with itself using identical arguments, so no arm difference can ever be detected):
  * `tests/test_invariants.py:88-94` "Full vs OutcomeHidden: identical exp_ids" — calls `retr.retrieve(q_input, q_state, alpha=0.5, k=4)` twice; `retrieve()` has no arm parameter.
  * `tests/test_invariants.py:143-145` "A/B order identical across arms" — `ab_order(model, task, sample_id, rnd)` twice.
  * `tests/test_invariants.py:146-149` "sampling seed identical across arms" — `call_seed(...)` twice.
* The real cross-arm property lives at the call sites (`core/pipeline.py:419-426` vs `core/batched_pipeline.py:206-213`, `:320-331`), which no test exercises. No test covers the repair retry, the online arm, or the draft-source fallback.

### F13 — Observation (shared by both paths, relevant to interpretation) — retrieved experience often collapses to a single auxiliary item

* k=4/5 units are requested, but on llama3.1-8b/en_zh **521/1 621 rounds (32 %)** draw *all* experiences from one auxiliary item (`b0…b4` of the same source, i.e. near-duplicate bootstrap attempts), and coedit_gec/qwen3-8b 19/64. Distinct auxiliary items per round: llama {1: 521, 2: 876, 3: 224}. For k=5 the quota cap `ceil(k/2)=3` does not prevent this.
* The controller therefore sometimes sees one experience repeated rather than a diverse set — material when interpreting "experience helps" claims, and the same in both pipelines (shared retriever), so not attributable to batching.

### F14 — DESIGN DIFFERENCE (dead knob) — `ban_own_experience` is declared but never implemented

* Only occurrence: `core/pipeline.py:64` (`ban_own_experience: bool = True`), serialised into every trace's `config`; no code path filters the item's own experience in either pipeline.
* Inert for the frozen main runs: the library is built from the `initial` auxiliary split (`build_experience.py:63`), the runs use `test`, and I verified 0 source overlap and 0 reference containment (§F11). It becomes live for online/accumulation designs (F3), where the config field currently promises a protection that does not exist.

### F15 — Minor code hygiene

* `run_experiment.py:257-286` `_finish_batched` is dead code (never called).
* `core/judge.py:208-211` is a no-op `if round_index == 0: pass`.
* The batched path bypasses `LLMClient.generate`, so `random.seed(seed)` (`baseline/core/llm.py:135`) is not called there. Irrelevant for vLLM (seeding is via `SamplingParams.seed`), but it is a behavioural difference for the `transformers` backend.
* `BatchedLLM` reports `latency = batch_wall_time / n` even for the 1-request fallback — different definition from the unbatched path (F4).

---

## 3. Summary table

| # | Claim / area | Verdict |
|---|---|---|
| 1 | Full vs OutcomeHidden: byte-identical exp_ids | **VERIFIED OK** by construction (shared `_retrieve`/renderer); hidden block is a pure outcome-line strip on 3 939/3 940 real rounds |
| 2 | RandomRetrieve: same count, same composition, different ids | **VERIFIED OK** on real states: 0 count diffs, 0 composition diffs; identical id *list* in 2/3 940 rounds (one empty-retrieval round, one with a 3-item single-class pool) — "deliberately different" holds in 99.95 % |
| 3 | Same A/B ordering and sampling seeds in all arms | **VERIFIED OK** in code (same pure functions, same args); the unit test that claims to check it is tautological |
| 4 | Offline-only fields never in a prompt | **VERIFIED OK** (whitelist renderer + no prompt builder receives reference/`delta_offline`); corroborated by AUC≈0.5 and by explaining 44/44 instruction overlaps from `current` |
| 5 | y0 byte-identical across arms | **VERIFIED OK** structurally (stored, positional, arm-independent; 0/1 768 mismatches vs CSV). Cross-arm byte identity itself: **UNVERIFIABLE** now — only `full_static` has data |
| — | Controller repair retry | **CONFIRMED BUG** (F1), 13 rounds fired the failure path in shipped data with no retry |
| — | Judge repair retry | **CONFIRMED BUG** (F2), 331/1 562 llama rounds show the no-repair signature |
| — | `full_online` under batching | **CONFIRMED BUG**, latent (F3) |
| — | Batch composition / comparability | **DESIGN DIFFERENCE** + reasoned risk, **UNVERIFIABLE** without GPU (F7) |
| — | latency accounting | **DESIGN DIFFERENCE** (F4) |
| — | Data completeness / summary-vs-trace mismatch | **CONFIRMED** integrity problem (F8) |

## 4. What I could NOT verify

1. **Cross-arm identity on real data for claims 1, 2, 3, 5** — only `full_static` traces exist; `runs/ablation` and `runs/accumulation` are empty.
2. **Whether the unbatched repair would have rescued the 13 controller failures and the llama judge failures** — `RoundTrace` stores neither the raw reply nor a per-call parse-error flag, so the counterfactual is not recoverable from the traces (it would need a re-run with the repair restored).
3. **Whether batch composition changes model outputs here** (F7) — needs a GPU replay of the same prompts/seeds at different batch sizes; the GPUs are busy with the live runs and this audit is read-only.
4. **Any claim about runs that have been overwritten** — e.g. the 1 000-sample qwen3-8b/en_zh run whose `summary.json` survives but whose trace no longer does.
5. **The exact number of "complete" runs**: at snapshot time only qwen3-8b/zh_en (1 000) was complete and self-consistent; the rest are partial and in flux, so all per-file statistics above are labelled with the snapshot's line counts and hashes.
