#!/usr/bin/env python
"""Self-tests for Phase 6 (quality-compute scaling) and the BoN-J arm.

No GPU, no production writes: everything runs against a deterministic stub
engine (same pattern as ``tests/test_resume_equivalence.py``), and the only
files written go to a throwaway directory under ``/tmp``.

What is proven here
-------------------
1. ``n_candidates == 1`` is the *existing* single-candidate path, record for
   record -- both at the pipeline level and end to end through
   ``run_experiment.run_model_task`` (where the BoN-J N=1 record must be
   identical to a ``full_static`` record with the same config).
2. With N > 1 the selected candidate is exactly the one the documented rule
   (round-robin Copeland with head-to-head/lowest-index tie-breaks) must pick,
   re-derived independently from the per-comparison verdicts recorded in the
   trace, and every recorded verdict matches the independent pairwise
   expectation for the stub judge.
3. Candidate seeds are a pure function of (sample_id, round_index,
   candidate_index): identical under a change of batch size and under a
   resume that keeps global manifest indices.
4. Cost is attributed per stage: sampling + judging == total, the finer
   controller/candidate split adds up, and calls add up.
5. The K-budget reuse rule is exact: prefix-truncating a K=3 run to K rounds
   reproduces a real ``--max-rounds K`` run, cost included.
6. The BoN-J resume guard rejects records produced at a different N.

Run:  /home/ymb/miniconda3/envs/qwen35/bin/python tests/test_scaling.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core  # noqa: F401,E402  (registers baseline_core)
from baseline_core.types import Generation  # noqa: E402
from core.bm25_fields import ExperienceRetriever  # noqa: E402
from core.batched_pipeline import BatchedPipeline  # noqa: E402
from core.manifest import manifest_path, read_manifest  # noqa: E402
from core.pipeline import RunConfig  # noqa: E402

import run_experiment as rex  # noqa: E402
import run_scaling as rsc  # noqa: E402
import aggregate as agg  # noqa: E402

FAILURES: list[str] = []
TASK = "wmt19_en_zh"
MODEL = "qwen3-8b"
SEED = 42
N_ITEMS = 12

#: The production arms.  Phase 6 must not have touched their
#: experience_mode/stop_mode/pool/online (a changed *RunConfig* knob would
#: silently change the config hash).
#:
#: Two deliberate v2 edits, both user-directed and both outside RunConfig:
#:   * ``outcome_hidden`` retired -- the contrastive library stores wrong->right
#:     pairs only, so an outcome field would be constant and the arm became a
#:     duplicate of full_static.
#:   * ``positive_only`` pool "positive" -> "full".  Filtering by outcome label
#:     is vacuous on a library where every unit is already positive; the arm now
#:     means "drop the wrong draft", which is the contrastive-modelling
#:     ablation.  ``pool`` is not a RunConfig field, so the hash is unchanged --
#:     asserted below.
FROZEN_ARMS = {
    "full_static":     {"experience_mode": "full",           "stop_mode": "adaptive", "pool": "full",      "online": False},
    "full_online":     {"experience_mode": "full",           "stop_mode": "adaptive", "pool": "full",      "online": True},
    "random_retrieve": {"experience_mode": "random",         "stop_mode": "adaptive", "pool": "full",      "online": False},
    "positive_only":   {"experience_mode": "positive_only",  "stop_mode": "adaptive", "pool": "full",      "online": False},
    "no_experience":   {"experience_mode": "none",           "stop_mode": "adaptive", "pool": "none",      "online": False},
    "fixed_rounds":    {"experience_mode": "full",           "stop_mode": "fixed",    "pool": "full",      "online": False},
    "sr_j_fixed":      {"experience_mode": "none",           "stop_mode": "fixed",    "pool": "none",      "online": False},
    "sr_j_stop":       {"experience_mode": "none",           "stop_mode": "judge",    "pool": "none",      "online": False},
}

#: RunConfig field set -- the config-hash payload.  Adding a field here would
#: change every existing arm's hash and break resume identity.
FROZEN_CONFIG_FIELDS = [
    "task", "model", "experience_mode", "stop_mode", "alpha", "k", "max_rounds",
    "seed", "snapshot_id", "draft_source", "draft_cache", "ban_own_experience",
    # Added with the v2 renderer.  Listing it keeps the field set a deliberate,
    # reviewed decision; it is ALSO in LEGACY_HASH_DEFAULTS, so at its default
    # it never enters the hash payload, which is what keeps every pre-existing
    # run resumable.  That stronger property is asserted below.
    "renderer",
    # Added with the v2 "both" condition (examples shown to the controller as
    # well as the refiner).  Also legacy-defaulted, so v1 and the refiner-only
    # v2 runs keep their identities.
    "controller_sees_examples",
    "advice_mode",
    # Enforces the own-source ban that ``ban_own_experience`` only ever
    # declared.  Legacy-defaulted like the others: every run recorded before it
    # existed handed the refiner its own gold answer, and those rows must never
    # be spliced together with clean ones.
    "retrieval_excludes_own_source",
]


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# deterministic stub engine
# --------------------------------------------------------------------------- #
class JudgeStubLLM:
    """Engine whose output depends only on the call seed -- and, for judge calls,
    on the two answers in the prompt.

    Judge rule (the independent expectation the tests use): the answer carrying
    the numerically larger ``BONJSEED=<n>`` marker wins; equal markers (or no
    marker at all) are a tie.  Candidates carry their own call seed as that
    marker, so candidate i beats candidate j iff ``seed_i > seed_j`` regardless
    of which anonymous A/B slot it was placed in -- which is what makes the
    A/B mapping checkable.
    """

    backend = "stub"

    def __init__(self, constant_candidates: bool = False,
                 echo_current: bool = False,
                 broken_controller_once: bool = False,
                 broken_judge_once: bool = False) -> None:
        self.calls: list[tuple[str, int]] = []
        self.constant_candidates = constant_candidates
        self.echo_current = echo_current
        self.broken_controller_once = broken_controller_once
        self.broken_judge_once = broken_judge_once

    def count_tokens(self, text: str) -> int:
        return max(1, len(str(text)) // 4)

    # -- text builders ------------------------------------------------------
    @staticmethod
    def _echo(prompt: str) -> str:
        """The current answer, verbatim, out of a refine prompt.

        Handles both renderers: v1 labels the slot ``Current answer:`` and closes
        the prompt with ``Revision instruction:``; v2 uses the original method's
        ``Current Draft:`` inside a ``### Current Task`` block.
        """
        if "Current answer:\n" in prompt:
            body = prompt.split("Current answer:\n", 1)[1]
            return body.split("\n\nRevision instruction:\n", 1)[0]
        # v2 keeps the label and its value on one line, as the original method does.
        body = prompt.split("Current Draft: ", 1)[1]
        return body.split("\nInstruction: ", 1)[0]

    def _candidate(self, seed: int, prompt: str) -> str:
        if self.echo_current:
            return self._echo(prompt)
        if self.constant_candidates:
            return "CONSTANT REVISION"
        return f"BONJSEED={seed} revised answer"

    @staticmethod
    def _slots(prompt: str) -> tuple[str, str]:
        a = prompt.split("\nAnswer A:\n", 1)[1]
        a, b = a.split("\nAnswer B:\n", 1)
        b = b.split("\n\nWhich answer", 1)[0]
        return a.strip(), b.strip()

    @staticmethod
    def _marker(text: str) -> int:
        m = re.search(r"BONJSEED=(\d+)", text)
        return int(m.group(1)) if m else -1

    def _judge(self, prompt: str) -> str:
        a, b = self._slots(prompt)
        ma, mb = self._marker(a), self._marker(b)
        if ma == mb:
            verdict = "tie"
        else:
            verdict = "A" if ma > mb else "B"
        return json.dumps({"verdict": verdict, "reason": f"marker A={ma} B={mb}"})

    def _text(self, call_type: str, seed: int, prompt: str) -> str:
        # ``<x>_repair`` is the single retry: it must succeed, exactly as the
        # real engine is expected to when the first reply is malformed.
        if call_type == "judge" and self.broken_judge_once:
            return "I think answer A is better."          # not JSON
        if call_type == "controller" and self.broken_controller_once:
            return "Let me think about it."               # not JSON
        if call_type.startswith("judge"):
            return self._judge(prompt)
        if call_type.startswith("controller"):
            return json.dumps({"action": "REFINE",
                               "instruction": f"tighten the phrasing (s{seed})",
                               "reason": f"a concrete defect s{seed}"})
        if call_type.startswith("refine"):
            return self._candidate(seed, prompt)
        return f"OTHER call_type={call_type} seed={seed}"

    # -- engine API ---------------------------------------------------------
    def generate_batch(self, reqs):
        out = []
        for r in reqs:
            self.calls.append((r.call_type, r.seed))
            out.append(Generation(
                text=self._text(r.call_type, r.seed, r.prompt), input_tokens=3,
                output_tokens=2, latency=0.0, seed=r.seed, call_type=r.call_type,
                start_time=0.0, end_time=0.0, prompt=r.prompt))
        return out

    def generate(self, prompt, system_prompt, seed, call_type, max_tokens,
                 temperature=0.1, top_p=1.0):
        self.calls.append((call_type, seed))
        return Generation(text=self._text(call_type, seed, prompt), input_tokens=3,
                          output_tokens=2, latency=0.0, seed=seed, call_type=call_type,
                          start_time=0.0, end_time=0.0, prompt=prompt)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_cfg(max_rounds: int = 3, task: str = TASK, model: str = MODEL) -> RunConfig:
    spec = rex.ARM_REGISTRY["bon_judge"]
    return RunConfig(
        task=task, model=model,
        experience_mode=spec["experience_mode"], stop_mode=spec["stop_mode"],
        alpha=0.5, k=4, max_rounds=max_rounds, seed=SEED, snapshot_id="initial",
        # "" (not None): exactly what run_experiment.py passes, so config hashes
        # here are the production hashes.
        draft_source="stored", draft_cache="",
        # Mirror run_model_task, which now takes the renderer from the arm spec.
        # Omitting it would silently build a v1 config whose hash matches none of
        # the records the production path writes -- which is exactly the resume
        # guard working as designed, but not what this fixture means to test.
        renderer=spec.get("renderer", "v2"),
    )


def new_pipe(cfg, library, llm, batch_size: int, n_candidates: int):
    cls = rex.BonJBatchedPipeline if n_candidates > 1 else BatchedPipeline
    kwargs = {"n_candidates": n_candidates} if n_candidates > 1 else {}
    return cls(cfg, ExperienceRetriever(library), list(library), llm,
               cache=None, batch_size=batch_size, online=False, **kwargs)


def rec_of(pipe, tr) -> dict:
    """The persisted record, exactly as run_experiment._sink writes it."""
    bonj = pipe if getattr(pipe, "n_candidates", 1) > 1 else None
    return rex._emit_record(tr, bonj)


def dump(pipe, traces) -> dict:
    """sample_id -> record JSON, timing normalised away."""
    out = {}
    for tr in traces:
        rec = rec_of(pipe, tr)
        cost = dict(rec.get("cost") or {})
        cost.pop("latency_s", None)
        rec["cost"] = cost
        rec.pop("config", None)
        out[rec["sample_id"]] = json.dumps(rec, sort_keys=True, ensure_ascii=False)
    return out


def independent_winner(n: int, comparisons: list[dict]) -> int:
    """The documented rule, re-implemented from the trace's own verdicts."""
    wins = {c: 0 for c in range(n)}
    losses = {c: 0 for c in range(n)}
    decisive: dict[tuple[int, int], int | None] = {}
    for row in comparisons:
        i, j, v = row["i"], row["j"], row["verdict"]
        decisive[(i, j)] = None
        if v == "better":
            wins[i] += 1
            losses[j] += 1
            decisive[(i, j)] = i
        elif v == "worse":
            wins[j] += 1
            losses[i] += 1
            decisive[(i, j)] = j
    score = {c: wins[c] - losses[c] for c in range(n)}
    top = max(score.values())
    tied = sorted(c for c in range(n) if score[c] == top)
    if len(tied) == 1:
        return tied[0]
    dominators = [
        c for c in tied
        if all(decisive.get((min(c, d), max(c, d))) == c for d in tied if d != c)
    ]
    return min(dominators or tied)


def independent_verdict(seed_i: int, seed_j: int) -> str:
    """The stub judge's pairwise outcome for two candidate seeds."""
    if seed_i == seed_j:
        return "tie"
    return "better" if seed_i > seed_j else "worse"


def stage_cost_invariants(cost: dict) -> str:
    """Return "" when the stage attribution adds up, else the broken relation."""
    total = cost.get("total_tokens", 0.0)
    sampling = cost.get("sampling_tokens", 0.0)
    judging = cost.get("judging_tokens", 0.0)
    controller = cost.get("controller_tokens", 0.0)
    candidate = cost.get("candidate_tokens", 0.0)
    if abs((sampling + judging) - total) > 1e-6:
        return f"sampling({sampling}) + judging({judging}) != total({total})"
    if abs((controller + candidate) - sampling) > 1e-6:
        return f"controller({controller}) + candidate({candidate}) != sampling({sampling})"
    calls = (cost.get("controller_calls", 0) + cost.get("candidate_calls", 0)
             + cost.get("judge_calls", 0))
    if abs(calls - cost.get("n_calls", 0)) > 1e-6:
        return f"stage calls({calls}) != n_calls({cost.get('n_calls')})"
    if abs((cost.get("input_tokens", 0) + cost.get("output_tokens", 0)) - total) > 1e-6:
        return "input + output != total"
    return ""


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_frozen_surface() -> None:
    print("\n== frozen surface: no existing arm / RunConfig field changed ==")
    check("ARM_REGISTRY holds exactly the production arms plus bon_judge",
          set(rex.ARM_REGISTRY) == set(FROZEN_ARMS) | {"bon_judge"},
          f"got {sorted(rex.ARM_REGISTRY)}")
    check("the retired outcome_hidden arm is not schedulable",
          "outcome_hidden" not in rex.ARM_REGISTRY)
    check("the retired arm is declared, not merely deleted",
          "outcome_hidden" in getattr(rex, "RETIRED_ARMS", ()))
    changed = {
        arm: rex.ARM_REGISTRY.get(arm)
        for arm, spec in FROZEN_ARMS.items()
        if {k: rex.ARM_REGISTRY.get(arm, {}).get(k) for k in spec} != spec
    }
    check("every existing arm keeps its experience_mode/stop_mode/pool/online",
          not changed, f"changed: {changed}")
    check("bon_judge mirrors full_static (only n_candidates is added)",
          {k: rex.ARM_REGISTRY["bon_judge"][k] for k in FROZEN_ARMS["full_static"]}
          == FROZEN_ARMS["full_static"])
    from dataclasses import fields
    got = [f.name for f in fields(RunConfig)]
    check("RunConfig field set is unchanged (config hash payload frozen)",
          got == FROZEN_CONFIG_FIELDS, f"got {got}")
    # The real property behind that check: a field may be ADDED as long as its
    # default is excluded from the hash payload.  Every non-legacy field must
    # therefore actually move the hash, and every legacy-default field must not.
    from core.pipeline import LEGACY_HASH_DEFAULTS
    base = dict(task="wmt19_en_zh", model="qwen3-8b")
    check(
        "legacy-default fields are excluded from the hash payload",
        all(
            RunConfig(**base).config_hash == RunConfig(**base, **{k: v}).config_hash
            for k, v in LEGACY_HASH_DEFAULTS.items()
        ),
    )
    check(
        "a non-default renderer does move the hash",
        RunConfig(**base).config_hash != RunConfig(**base, renderer="v2").config_hash,
    )
    check("n_candidates is not a RunConfig field",
          "n_candidates" not in got)
    check("run_scaling completeness denominator == aggregate.py's",
          all(rsc.expected_n(t) == agg.expected_n(t)
              for t in ("wmt19_en_zh", "wmt19_zh_en", "coedit_gec", "gigaword")))


def test_n1_equivalence(library, refs) -> None:
    print("\n== N=1 is the existing single-candidate path ==")
    cfg = make_cfg(max_rounds=2)
    pipe_plain = new_pipe(cfg, library, JudgeStubLLM(), batch_size=4, n_candidates=1)
    plain = pipe_plain.run(refs)
    pipe_b1 = rex.BonJBatchedPipeline(
        cfg, ExperienceRetriever(library), list(library), JudgeStubLLM(),
        cache=None, batch_size=4, online=False, n_candidates=1)
    bonj1 = pipe_b1.run(refs)
    check("BonJBatchedPipeline(n_candidates=1) == plain BatchedPipeline (record for record)",
          dump(pipe_b1, bonj1) == dump(pipe_plain, plain))
    check("N=1 records carry no bon_judge block / no stage keys",
          all("bon_judge" not in rec_of(pipe_b1, tr) for tr in bonj1)
          and all("sampling_tokens" not in tr.cost for tr in bonj1))
    check("N=1 is not vacuous: it really refined",
          all(len(tr.rounds) == 2 for tr in plain))

    # N>1 must actually differ (otherwise the equality above proves nothing).
    pipe_b4 = new_pipe(cfg, library, JudgeStubLLM(), batch_size=4, n_candidates=4)
    bonj4 = pipe_b4.run(refs)
    check("N=4 is detectably different from N=1 (test is meaningful)",
          dump(pipe_b4, bonj4) != dump(pipe_plain, plain))


def test_selection_rule(library, refs) -> None:
    print("\n== N>1: selection rule + per-comparison verdicts ==")
    cfg = make_cfg(max_rounds=1)
    n = 4
    pipe = new_pipe(cfg, library, JudgeStubLLM(), batch_size=4, n_candidates=n)
    traces = pipe.run(refs)
    n_rounds_checked = 0
    for tr in traces:
        rec = rec_of(pipe, tr)
        block = rec["bon_judge"]
        check(f"{rec['sample_id']}: block records N and the rule",
              block["n_candidates"] == n and "round_robin_copeland" in block["selection_rule"])
        for rd in block["rounds"]:
            n_rounds_checked += 1
            seeds = [c["seed"] for c in rd["candidates"]]
            check(f"{rec['sample_id']} r{rd['round_index']}: one seed per candidate, all distinct",
                  len(seeds) == n and len(set(seeds)) == n)
            check(f"{rec['sample_id']} r{rd['round_index']}: every unordered pair compared once",
                  sorted((row["i"], row["j"]) for row in rd["comparisons"])
                  == [(i, j) for i in range(n) for j in range(i + 1, n)])
            for row in rd["comparisons"]:
                expected = independent_verdict(seeds[row["i"]], seeds[row["j"]])
                check(f"{rec['sample_id']} r{rd['round_index']}: pair "
                      f"({row['i']},{row['j']}) verdict=={expected}",
                      row["verdict"] == expected,
                      f"recorded {row['verdict']}")
                check(f"{rec['sample_id']} r{rd['round_index']}: pair "
                      f"({row['i']},{row['j']}) double-order consistency",
                      row["order_consistent"] == (row["verdict"] in ("better", "worse")))
            want = independent_winner(n, rd["comparisons"])
            check(f"{rec['sample_id']} r{rd['round_index']}: winner == independent rule",
                  rd["winner_index"] == want, f"recorded {rd['winner_index']} want {want}")
            best_seed = max(range(n), key=lambda c: (seeds[c], -c))
            check(f"{rec['sample_id']} r{rd['round_index']}: winner == highest-seed candidate",
                  rd["winner_index"] == best_seed)
            check(f"{rec['sample_id']} r{rd['round_index']}: trace candidate == winning candidate",
                  rec["rounds"][rd["round_index"]]["candidate"]
                  == "BONJSEED=%d revised answer" % seeds[rd["winner_index"]])
    check("selection was exercised on several rounds", n_rounds_checked >= len(refs))


def test_identical_candidates(library, refs) -> None:
    print("\n== identical candidates: ties by definition, no selection calls ==")
    cfg = make_cfg(max_rounds=1)
    pipe = new_pipe(cfg, library, JudgeStubLLM(constant_candidates=True),
                    batch_size=4, n_candidates=4)
    traces = pipe.run(refs)
    for tr in traces:
        rd = rec_of(pipe, tr)["bon_judge"]["rounds"][0]
        check(f"{tr.sample_id}: all pairs identical -> zero selection judge calls",
              all(r["identical"] and r["judge_calls"] == 0 for r in rd["comparisons"]))
        check(f"{tr.sample_id}: Copeland tie -> lowest index wins",
              rd["winner_index"] == 0 and set(rd["copeland_scores"]) == {0})
        check(f"{tr.sample_id}: distinct-from-current winner is still judged",
              rd["identical_to_current"] is False and rd["accept"]["judge_calls"] == 2
              and rd["accept"]["verdict"] == "tie" and not rd["accept"]["accepted"])


def test_identical_to_current(library, refs) -> None:
    print("\n== winner == current: free tie for N>1, delegated 2-call path for N=1 ==")
    cfg = make_cfg(max_rounds=1)
    sub = refs[:4]
    n1 = new_pipe(cfg, library, JudgeStubLLM(echo_current=True), batch_size=4, n_candidates=1)
    t1 = n1.run(sub)
    check("N=1 (delegated, existing path) short-circuits an identical pair to a "
          "FREE tie: the canonical judge rule spends no engine call on a pair that "
          "cannot have a winner.  (This assertion used to expect 2 judge calls -- it "
          "had encoded the old behaviour, and the identical-pair fix changed it.)",
          all(tr.rounds[0].cost["n_calls"] == 2 for tr in t1)   # 1 controller + 1 refine
          and all(tr.rounds[0].judge_verdict == "tie" for tr in t1)
          and all(tr.rounds[0].judge_reason_a.startswith("identical") for tr in t1))

    n4 = new_pipe(cfg, library, JudgeStubLLM(echo_current=True), batch_size=4, n_candidates=4)
    t4 = n4.run(sub)
    for tr in t4:
        rd = rec_of(n4, tr)["bon_judge"]["rounds"][0]
        check(f"{tr.sample_id}: winner == current is a free tie (canonical judge rule)",
              rd["identical_to_current"] is True and rd["accept"]["judge_calls"] == 0
              and rd["accept"]["verdict"] == "tie" and not rd["accept"]["accepted"])
        check(f"{tr.sample_id}: no judge call at all for an all-identical round",
              tr.rounds[0].cost["judge_calls"] == 0
              and tr.rounds[0].cost["judging_tokens"] == 0
              and tr.rounds[0].cost["n_calls"] == 5)   # 1 controller + 4 candidates
        check(f"{tr.sample_id}: trace keeps the current answer (nothing accepted)",
              tr.final_output == tr.initial_draft)


def test_repair_paths(library, refs) -> None:
    print("\n== one repair retry on malformed controller/judge replies (N>1 path) ==")
    cfg = make_cfg(max_rounds=1)
    sub = refs[:3]

    pc = new_pipe(cfg, library, JudgeStubLLM(broken_controller_once=True),
                  batch_size=4, n_candidates=2)
    tc = pc.run(sub)
    check("controller repair kept the round alive (REFINE, no structured failure)",
          all(len(tr.rounds) == 1 and tr.rounds[0].controller_action == "REFINE"
              and not tr.rounds[0].controller_structured_failure for tr in tc))
    check("controller + controller_repair are both accounted",
          all(tr.rounds[0].cost["controller_calls"] == 2 for tr in tc))

    pj = new_pipe(cfg, library, JudgeStubLLM(broken_judge_once=True),
                  batch_size=4, n_candidates=2)
    tj = pj.run(sub)
    for tr in tj:
        rd = rec_of(pj, tr)["bon_judge"]["rounds"][0]
        seeds = [c["seed"] for c in rd["candidates"]]
        check(f"{tr.sample_id}: repaired judge calls yield the correct verdicts",
              all(row["verdict"] == independent_verdict(seeds[row["i"]], seeds[row["j"]])
                  for row in rd["comparisons"]))
        # 1 selection pair (2 calls) + acceptance (2 calls), each repaired once
        check(f"{tr.sample_id}: judge + judge_repair calls are all accounted",
              tr.rounds[0].cost["judge_calls"] == 8
              and tr.rounds[0].cost["n_calls"] == 1 + 2 + 8)
        check(f"{tr.sample_id}: repaired acceptance still accepts the winner",
              rd["accept"]["verdict"] == "better" and tr.rounds[0].accepted)


def test_batch_and_resume_stability(library, refs) -> None:
    print("\n== candidate seeds stable under batch size and resume ==")
    cfg = make_cfg(max_rounds=2)

    def seeds_by_sample(pipe, traces):
        out = {}
        for tr in traces:
            block = rec_of(pipe, tr)["bon_judge"]
            out[tr.sample_id] = json.dumps(
                [(rd["round_index"], rd["candidate_seeds"]) for rd in block["rounds"]])
        return out

    p4 = new_pipe(cfg, library, JudgeStubLLM(), batch_size=4, n_candidates=4)
    b4 = p4.run(refs)
    p8 = new_pipe(cfg, library, JudgeStubLLM(), batch_size=8, n_candidates=4)
    b8 = p8.run(refs)
    check("batch_size 4 == batch_size 8 (record for record)", dump(p4, b4) == dump(p8, b8))
    check("candidate seeds identical across batch sizes",
          seeds_by_sample(p4, b4) == seeds_by_sample(p8, b8))

    cut = N_ITEMS // 2
    q1 = new_pipe(cfg, library, JudgeStubLLM(), batch_size=4, n_candidates=4)
    part1 = q1.run(refs, indices=list(range(0, cut)))
    q2 = new_pipe(cfg, library, JudgeStubLLM(), batch_size=4, n_candidates=4)
    part2 = q2.run(refs, indices=list(range(cut, N_ITEMS)))
    resumed = dump(q1, part1)
    resumed.update(dump(q2, part2))
    check("resumed BoN-J run == from scratch (record for record)", resumed == dump(p4, b4))
    rs = seeds_by_sample(q1, part1)
    rs.update(seeds_by_sample(q2, part2))
    check("candidate seeds identical after resume", rs == seeds_by_sample(p4, b4))


def test_cost_attribution(library, refs) -> None:
    print("\n== cost attribution: sampling + judging == total ==")
    cfg = make_cfg(max_rounds=2)
    pipe = new_pipe(cfg, library, JudgeStubLLM(), batch_size=4, n_candidates=4)
    traces = pipe.run(refs)
    bad = [(tr.sample_id, stage_cost_invariants(tr.cost)) for tr in traces
           if stage_cost_invariants(tr.cost)]
    check("every trace's stage buckets add up", not bad, str(bad[:3]))
    check("every trace spent judging tokens", all(tr.cost["judging_tokens"] > 0 for tr in traces))
    per_round = [r.cost for tr in traces for r in tr.rounds if r.controller_action == "REFINE"]
    check("per-round trace cost also adds up",
          all(not stage_cost_invariants(c) for c in per_round))

    # Judge calls must scale with N: round-robin selection is 2*C(N,2) calls.
    for n, want_selection_calls in ((2, 2), (3, 6), (4, 12)):
        one = new_pipe(make_cfg(max_rounds=1), library, JudgeStubLLM(),
                       batch_size=4, n_candidates=n)
        for tr in one.run(refs[:3]):
            rd = rec_of(one, tr)["bon_judge"]["rounds"][0]
            got = sum(r["judge_calls"] for r in rd["comparisons"])
            check(f"N={n}: selection judge calls == 2*C(N,2)",
                  got == want_selection_calls, f"got {got}")
            check(f"N={n}: acceptance cost 2 judge calls",
                  rd["accept"]["judge_calls"] == 2)
            check(f"N={n}: judging_tokens == selection+acceptance share",
                  tr.rounds[0].cost["judge_calls"] == want_selection_calls + 2)


def test_k_prefix_derivation(library, refs) -> None:
    print("\n== K sweep: prefix truncation == a real K-budget run ==")
    sub = refs[:6]
    full = BatchedPipeline(make_cfg(max_rounds=3), ExperienceRetriever(library),
                           list(library), JudgeStubLLM(), cache=None,
                           batch_size=3, online=False).run(sub)
    check("K=3 run used all 3 rounds", all(len(t.rounds) == 3 for t in full))
    raw3 = [t.to_dict() for t in full]

    for k in (1, 2):
        real = BatchedPipeline(make_cfg(max_rounds=k), ExperienceRetriever(library),
                               list(library), JudgeStubLLM(), cache=None,
                               batch_size=3, online=False).run(sub)
        real_map = {t.sample_id: t.to_dict() for t in real}
        derived_map = {r["sample_id"]: rsc.derive_prefix(r, k) for r in raw3}
        same = True
        detail = ""
        for sid, d in derived_map.items():
            r = real_map[sid]
            for key in ("rounds", "final_output", "final_metric_offline",
                        "initial_draft", "initial_metric_offline", "cost",
                        "config_hash", "sample_id", "task", "model", "seed"):
                if d.get(key) != r.get(key):
                    same = False
                    detail = f"{sid}:{key}: derived={d.get(key)!r} real={r.get(key)!r}"
                    break
            if not same:
                break
        check(f"derived K={k} == real --max-rounds {k} run (rounds/final/cost/hash)",
              same, detail[:300])
        # the derived point's cost must be the K-budget spend, not the K=3 spend
        d0 = derived_map[sub[0].sample_id]
        check(f"K={k}: cost == cumulative cost of the last kept round",
              d0["cost"] == d0["rounds"][k - 1]["cost"])
        check(f"K={k}: config_hash equals a real max_rounds={k} config",
              d0["config_hash"] == make_cfg(max_rounds=k).config_hash)
    # K=3 truncation is the identity on rounds/cost
    d3 = rsc.derive_prefix(raw3[0], 3)
    check("K=3 truncation leaves rounds and cost untouched",
          d3["rounds"] == raw3[0]["rounds"] and d3["cost"] == raw3[0]["cost"])


def test_run_experiment_wiring(library, tmp: Path) -> None:
    print("\n== run_model_task wiring (auto tag, block, resume guard) ==")
    task, model, limit = TASK, MODEL, 3
    common = dict(task=task, model=model, seed=SEED, gpu="", split="test",
                  limit=limit, alpha=0.5, k=4, draft_source="stored",
                  draft_cache="", batch_size=4, out_dir=tmp)

    s4 = rex.run_model_task(arm="bon_judge", tag="", max_rounds=2,
                            llm=JudgeStubLLM(), n_candidates=4, **common)
    p4 = tmp / "bon_judge" / "seed42" / task / model / "n4" / "bon_judge.jsonl"
    check("N=4 landed in the automatic n4 point directory", p4.is_file())
    recs4 = [json.loads(l) for l in p4.read_text().splitlines() if l.strip()]
    check("N=4 wrote one record per limited sample", len(recs4) == limit)
    check("N=4 records carry the bon_judge block",
          all(r.get("bon_judge", {}).get("n_candidates") == 4 for r in recs4))
    check("N=4 records carry the stage attribution",
          all(not stage_cost_invariants(r["cost"]) for r in recs4))
    check("N=4 cost has the finer controller/candidate split",
          all({"controller_tokens", "candidate_tokens", "judging_tokens",
               "sampling_tokens"} <= set(r["cost"]) for r in recs4))
    summ4 = json.loads(p4.with_suffix(".summary.json").read_text())
    check("N=4 summary records N and the selection rule",
          summ4.get("n_candidates") == 4
          and "round_robin_copeland" in (summ4.get("selection_rule") or ""))

    # N=1 must be record-for-record the full_static run of the same config.
    rex.run_model_task(arm="bon_judge", tag="", max_rounds=2,
                       llm=JudgeStubLLM(), n_candidates=1, **common)
    rex.run_model_task(arm="full_static", tag="", max_rounds=2,
                       llm=JudgeStubLLM(), **common)
    p1 = tmp / "bon_judge" / "seed42" / task / model / "bon_judge.jsonl"
    pf = tmp / "full_static" / "seed42" / task / model / "full_static.jsonl"
    check("N=1 wrote into the untagged main-style path", p1.is_file() and pf.is_file())
    r1 = [json.loads(l) for l in p1.read_text().splitlines() if l.strip()]
    rf = [json.loads(l) for l in pf.read_text().splitlines() if l.strip()]
    check("N=1 record == full_static record, record for record", r1 == rf)
    check("N=1 records have no bon_judge block and no stage keys",
          all("bon_judge" not in r and "sampling_tokens" not in r["cost"] for r in r1))

    # resume guard: a record produced at another N may never be reused
    keep = dict(recs4[0])
    foreign = dict(recs4[1])
    foreign["bon_judge"] = dict(foreign["bon_judge"], n_candidates=2)
    d = tmp / "prior"
    d.mkdir(parents=True, exist_ok=True)
    jp = d / "bon_judge.jsonl"
    jp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in (keep, foreign)) + "\n")
    cfg = make_cfg(max_rounds=2)
    prior, ids, dropped = rex._load_prior(jp, cfg.config_hash, task, model, SEED,
                                          require=rex._bonj_record_filter(4))
    check("N=4 resume keeps its own record and drops the N=2 one",
          len(prior) == 1 and prior[0].sample_id == keep["sample_id"] and dropped == 1,
          f"kept {len(prior)} dropped {dropped}")
    prior1, _, dropped1 = rex._load_prior(jp, cfg.config_hash, task, model, SEED,
                                          require=rex._bonj_record_filter(1))
    check("N=1 resume filter rejects every N>1 record",
          len(prior1) == 0 and dropped1 == 2, f"kept {len(prior1)} dropped {dropped1}")


def test_scaling_plan(library, tmp: Path) -> None:
    print("\n== run_scaling plan: reuse / derive / compute split ==")
    # Point paths must match exactly what run_experiment writes with that tag.
    out = tmp / "runs"
    p = rsc.Point(kind="n", value=4, label="n4", arm="bon_judge", config_name="BoN-J",
                  model=MODEL, task=TASK, seed=SEED, n_samples=1000, max_rounds=3,
                  n_candidates=4, path=rsc.point_dir("bon_judge", SEED, TASK, MODEL, "n4",
                                                     None, out) / "bon_judge.jsonl",
                  tag=rsc.point_tag("n4", None))
    cfg = rsc.expected_config("bon_judge", TASK, MODEL, SEED, 0.5, 4, 3)
    check("run_scaling's expected config == the main full_static config hash",
          cfg.config_hash == RunConfig(task=TASK, model=MODEL, experience_mode="full",
                                       stop_mode="adaptive", alpha=0.5, k=4, max_rounds=3,
                                       seed=SEED, snapshot_id="initial",
                                       draft_source="stored", draft_cache="").config_hash)
    calls = rsc.estimate_calls(p)
    check("call estimate counts 2*C(N,2) selection judge calls per round",
          calls == 1000 * 3 * (1 + 4 + 12 + 2), f"got {calls}")
    # point_dir/point_tag must reproduce run_experiment's own run_dir rule:
    # out_dir/arm/seed<seed>/task/model[/split]/<tag>/<arm>.jsonl
    check("point path == run_experiment's run_dir for that tag",
          rsc.point_dir("bon_judge", SEED, TASK, MODEL, "n4", None, out)
          / "bon_judge.jsonl"
          == out / "bon_judge" / f"seed{SEED}" / TASK / MODEL / "n4" / "bon_judge.jsonl")
    check("--limit adds a limit<N>/ component before the point label",
          rsc.point_dir("bon_judge", SEED, TASK, MODEL, "n4", 9, out)
          / "bon_judge.jsonl"
          == out / "bon_judge" / f"seed{SEED}" / TASK / MODEL / "limit9" / "n4"
          / "bon_judge.jsonl"
          and rsc.point_tag("n4", 9) == "limit9/n4")
    check("K-point estimate is 4 calls/round",
          rsc.estimate_calls(rsc.Point(kind="k", value=5, label="k5", arm="full_static",
                                       config_name="Full", model=MODEL, task=TASK,
                                       seed=SEED, n_samples=1000, max_rounds=5,
                                       n_candidates=1, path=Path("/dev/null"),
                                       tag="k5")) == 1000 * 5 * 4)
    check("expected_n follows aggregate.py for every task",
          rsc.expected_n("gigaword") == 100 and rsc.expected_n("coedit_gec") == 1000)

    # completeness: a parseable full file counts, a short or broken one does not
    good = tmp / "good.jsonl"
    good.write_text("\n".join(json.dumps({"sample_id": f"s{i}"}) for i in range(5)) + "\n")
    short = tmp / "short.jsonl"
    short.write_text("\n".join(json.dumps({"sample_id": f"s{i}"}) for i in range(4)) + "\n")
    broken = tmp / "broken.jsonl"
    broken.write_text(json.dumps({"sample_id": "s0"}) + "\n{\"sample_id\": \"s1\"")
    check("complete file detected", rsc.completeness(good, 5)[0])
    check("short file rejected", not rsc.completeness(short, 5)[0])
    check("truncated tail file rejected (as aggregate.py skips it)",
          not rsc.completeness(broken, 1)[0])


def test_dry_run_is_gpu_free(tmp: Path) -> None:
    print("\n== --dry-run: full plan, no GPU import, no writes ==")
    out_dir = tmp / "dry_out"
    code = (
        "import sys; sys.argv=['run_scaling.py','--dry-run','--models','qwen3-8b',"
        "'--tasks','wmt19_en_zh','--out-dir',%r];"
        "import run_scaling; rc=run_scaling.main();"
        "print('RC', rc); print('TORCH', 'torch' in sys.modules);"
        "print('VLLM', 'vllm' in sys.modules);"
        "print('BASELINE_LLM', 'baseline_core.llm' in sys.modules)"
    ) % str(out_dir)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(ROOT))
    check("dry-run exits 0", proc.returncode == 0 and "RC 0" in proc.stdout,
          proc.stderr[-400:])
    check("dry-run imports neither torch, vllm nor the LLM client",
          "TORCH False" in proc.stdout and "VLLM False" in proc.stdout
          and "BASELINE_LLM False" in proc.stdout, proc.stdout[-300:])
    check("dry-run wrote nothing", not out_dir.exists())
    check("dry-run prints the full plan with the reuse/derive/compute split",
          "Phase 6 plan summary" in proc.stdout and "samples to compute" in proc.stdout)
    check("dry-run lists all 20 Phase 6 points for one model x one task "
          "(4 configs x {k1,k2,k3,k5} plus {n1,n2,n4,n8})",
          proc.stdout.count("\n   k") + proc.stdout.count("\n   n") == 20,
          str(proc.stdout.count("\n   k") + proc.stdout.count("\n   n")))


def main() -> int:
    refs_all = read_manifest(manifest_path(TASK, "test"))
    refs = refs_all[:N_ITEMS]
    library = rex.build_library(MODEL, TASK, "full")
    print(f"manifest={len(refs_all)} items, using {len(refs)}; "
          f"experience library={len(library)} units; task={TASK} model={MODEL}")
    check("library is non-empty (retrieval path exercised)", bool(library))

    test_frozen_surface()
    test_n1_equivalence(library, refs)
    test_selection_rule(library, refs)
    test_identical_candidates(library, refs)
    test_identical_to_current(library, refs)
    test_repair_paths(library, refs)
    test_batch_and_resume_stability(library, refs)
    test_cost_attribution(library, refs)
    test_k_prefix_derivation(library, refs)

    tmp = Path(tempfile.mkdtemp(prefix="test_scaling_"))
    try:
        test_run_experiment_wiring(library, tmp)
        test_scaling_plan(library, tmp)
        test_dry_run_is_gpu_free(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all Phase 6 / BoN-J checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
