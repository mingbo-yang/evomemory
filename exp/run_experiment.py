#!/usr/bin/env python
"""Main experiment / ablation runner.

Each arm is a *named configuration* of the two orthogonal knobs, so that the
only difference between an ablation and ``Full`` is the single column it is
meant to isolate (plan v5 sections 4.1 and 6):

===============  ================  ==========  ==================
arm              experience_mode   stop_mode   pool
===============  ================  ==========  ==================
full_static      full              adaptive    full
full_online      full              adaptive    full (+ online writes)
random_retrieve  random            adaptive    full
outcome_hidden   outcome_hidden    adaptive    full
positive_only    positive_only     adaptive    positives only
no_experience    none              adaptive    -
fixed_rounds     full              fixed       full
sr_j_fixed       none              fixed       -
sr_j_stop        none              judge       -
bon_judge        full              adaptive    full (+ best-of-N, same judge)
===============  ================  ==========  ==================

``bon_judge`` is the Phase 6 best-of-N arm (BoN-J).  Its retrieval/stopping
configuration is identical to ``full_static`` -- the only extra knob is
``--n-candidates N``: the refinement stage draws N candidate revisions of the
same state and picks the winner with the *same* pairwise judge used everywhere
else (see ``BONJ_SELECTION_RULE``).  ``N == 1`` delegates to the unmodified
single-candidate path, so a BoN-J N=1 run is record-for-record identical to
``full_static``; the Phase 6 driver reuses the existing ``full_static`` runs for
that point instead of recomputing it.

Usage
-----
    python run_experiment.py --arm full_static --gpu 2 \
        --models glm4-9b --tasks wmt19_en_zh --seed 42 --limit 20
    python run_experiment.py --arm bon_judge --n-candidates 4 --gpu 2 \
        --models qwen3-8b --tasks wmt19_en_zh --tag n4 --out-dir runs/scaling
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    EXP_ROOT,
    MODELS,
    TASKS,
    TEST_SAMPLES,
    ensure_gpu_whitelist,
    gpu_mem_util,
    MAX_MODEL_LEN,
    resolve_model_config,
)
from core.batch_llm import GenRequest  # noqa: E402
from core.batched_pipeline import BatchedPipeline  # noqa: E402
from core.bm25_fields import Experience, ExperienceRetriever  # noqa: E402
from core.determinism import call_seed, randomised_ab  # noqa: E402
from core.experience import (  # noqa: E402
    experience_dir,
    filter_positive_only,
    load_experiences,
    render_experience_block,
    save_experiences,
)
from core.judge import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    _SYSTEM as _JUDGE_SYSTEM,
    _TEMPLATE as _JUDGE_TEMPLATE,
    _combine,
    _extract_json,
    _map_verdict,
    accept_revision,
)
from core.manifest import manifest_path, read_manifest, sha256_text  # noqa: E402
from core.pipeline import (  # noqa: E402
    DEFAULT_REFINE_INSTRUCTION,
    Pipeline,
    RoundTrace,
    RunConfig,
    SampleTrace,
    build_refine_prompt,
    transitions_to_experiences,
)
from core.retrieval_cache import (RetrievalCache, cache_path,
                                  library_fingerprint)  # noqa: E402

from baseline_core.types import TaskExample  # noqa: E402

ARM_REGISTRY: Dict[str, dict] = {
    # Every arm runs the v2 renderer: the retrieved unit is the original
    # method's four-line contrastive example, and it conditions the *refiner*
    # directly (the controller decides only whether/where to revise).
    "full_static":     {"experience_mode": "full",           "stop_mode": "adaptive", "pool": "full",      "online": False},
    "full_online":     {"experience_mode": "full",           "stop_mode": "adaptive", "pool": "full",      "online": True},
    "random_retrieve": {"experience_mode": "random",         "stop_mode": "adaptive", "pool": "full",      "online": False},
    # Contrastive-modelling ablation: identical retrieved ids, but the wrong
    # draft is dropped so only the gold answer is shown.  See
    # RunConfig.render_contrastive -- the switch is derived from the mode, not a
    # separate field, so this arm cannot silently drift from the main method.
    "positive_only":   {"experience_mode": "positive_only",  "stop_mode": "adaptive", "pool": "full",      "online": False},
    # Architecture A/B: does the CONTROLLER also need to see the examples?
    # Identical to full_static except that the retrieved block is shown to the
    # controller as well as the refiner.  Added because moving the examples out
    # of the controller (the v2 default) was measured on dev to raise refinement
    # frequency, and the user asked for the two options to be compared rather
    # than decided by argument.
    "full_both":       {"experience_mode": "full",           "stop_mode": "adaptive", "pool": "full",      "online": False,
                        "controller_sees_examples": True},
    "no_experience":   {"experience_mode": "none",           "stop_mode": "adaptive", "pool": "none",      "online": False},
    "fixed_rounds":    {"experience_mode": "full",           "stop_mode": "fixed",    "pool": "full",      "online": False},
    "sr_j_fixed":      {"experience_mode": "none",           "stop_mode": "fixed",    "pool": "none",      "online": False},
    "sr_j_stop":       {"experience_mode": "none",           "stop_mode": "judge",    "pool": "none",      "online": False},
    # Phase 6 BoN-J.  Same knobs as full_static; n_candidates is a *pipeline*
    # argument, NOT a RunConfig field, so the config hash (and therefore every
    # existing arm's resume identity) is untouched.
    "bon_judge":       {"experience_mode": "full",           "stop_mode": "adaptive", "pool": "full",      "online": False,
                        "n_candidates": 1},
}

#: ``outcome_hidden`` is retired.  The contrastive library stores wrong->right
#: pairs only, so every unit would read ``helped`` and there is no outcome field
#: left to hide; the arm had become a no-op duplicate of ``full_static``.  The
#: string stays a valid ``experience_mode`` (see core.pipeline.EXPERIENCE_MODES)
#: purely so that rows already on disk still parse.  The guard below makes the
#: retirement enforceable rather than a comment.
RETIRED_ARMS = ("outcome_hidden",)
assert not (set(RETIRED_ARMS) & set(ARM_REGISTRY)), "retired arm is still schedulable"


def build_library(model: str, task: str, pool: str) -> List[Experience]:
    if pool == "none":
        return []
    exps = load_experiences(experience_dir(model, task) / "initial.jsonl")
    if not exps:
        raise FileNotFoundError(
            f"no initial experience for {model}/{task}; run build_experience.py first"
        )
    if pool == "positive":
        exps = filter_positive_only(exps)
    return exps


# --------------------------------------------------------------------------- #
# BoN-J: best-of-N candidate revisions selected by the SAME pairwise judge
# --------------------------------------------------------------------------- #

#: The one selection rule BoN-J uses, stated once here and implemented once in
#: :func:`bonj_select_winner`.  Every trace records the per-comparison verdicts
#: (``rec["bon_judge"]``), so the rule can be re-applied to a finished run
#: without trusting this function.
BONJ_SELECTION_RULE = (
    "round_robin_copeland: every unordered pair of the N candidates is judged "
    "once with the same double-order randomised-A/B pairwise judge; a candidate "
    "scores +1 for each opponent it strictly beats and -1 for each opponent that "
    "strictly beats it; the highest score wins, ties are broken by the "
    "head-to-head dominator among the tied candidates and then by the lowest "
    "candidate index"
)

#: Same wording as ``core.judge.PairwiseJudge`` uses when the two texts are
#: byte-identical: that case is a tie by definition and costs no engine call.
_IDENTICAL_PAIR_REASON = "identical: candidate equals current verbatim"


def bonj_copeland_scores(
    n_candidates: int, decisive: Dict[Tuple[int, int], Optional[int]]
) -> List[int]:
    """Copeland scores for a complete round-robin over ``n_candidates``.

    ``decisive[(i, j)]`` (with ``i < j``) holds the index of the candidate the
    judge strictly preferred, or ``None`` when the comparison was a tie /
    uncertain.  A win is ``+1`` for the winner and ``-1`` for the loser.
    """
    n = int(n_candidates)
    if n < 1:
        raise ValueError("n_candidates must be >= 1")
    score = [0] * n
    for (i, j), winner in decisive.items():
        if winner is None:
            continue
        if not (0 <= i < j < n):
            raise ValueError(f"pair ({i}, {j}) outside 0..{n - 1}")
        if winner not in (i, j):
            raise ValueError(f"pair ({i}, {j}) has winner {winner}")
        loser = j if winner == i else i
        score[winner] += 1
        score[loser] -= 1
    return score


def bonj_select_winner(
    n_candidates: int, decisive: Dict[Tuple[int, int], Optional[int]]
) -> int:
    """Apply :data:`BONJ_SELECTION_RULE` and return the winning candidate index.

    Tie-breaks, in order: the head-to-head dominator among the tied candidates
    (the one that beat every other tied candidate in their own comparison, when
    such a candidate exists), then the lowest candidate index.
    """
    n = int(n_candidates)
    score = bonj_copeland_scores(n, decisive)
    top = max(score)
    tied = [c for c in range(n) if score[c] == top]
    if len(tied) == 1:
        return tied[0]
    dominators = [
        c for c in tied
        if all(
            decisive.get((min(c, d), max(c, d))) == c
            for d in tied if d != c
        )
    ]
    return min(dominators or tied)


class _BonJRecord:
    """A ``SampleTrace`` whose persisted record also carries the BoN-J detail.

    ``SampleTrace.to_dict`` is ``asdict``-based and therefore cannot hold extra
    keys; this thin wrapper delegates every attribute and only extends the
    serialised record.  It is used for ``n_candidates > 1`` runs *only*, so the
    record shape of every existing arm is untouched.
    """

    __slots__ = ("_trace", "_block")

    def __init__(self, trace: SampleTrace, block: dict) -> None:
        self._trace = trace
        self._block = block

    def __getattr__(self, name: str):
        return getattr(self._trace, name)

    def to_dict(self) -> dict:
        rec = self._trace.to_dict()
        rec["bon_judge"] = self._block
        return rec


def _emit_record(trace, bonj_pipe: Optional["BonJBatchedPipeline"]) -> dict:
    """Trace -> persisted record.  ``bonj_pipe`` is non-None for BoN-J N>1 only.

    With ``bonj_pipe is None`` this is exactly ``trace.to_dict()``, i.e. the
    record every existing arm has always written.
    """
    if bonj_pipe is None:
        return trace.to_dict()
    return _BonJRecord(trace, bonj_pipe.bonj_trace_block(trace.sample_id)).to_dict()


def _bonj_record_filter(n_candidates: int):
    """Resume guard: a BoN-J record is reusable only at the same N.

    ``n_candidates`` is a pipeline argument, not a ``RunConfig`` field (the
    config hash must stay frozen for every existing arm), so without this filter
    a resumed N=2 run could silently absorb records produced at N=4.  Records
    without a ``bon_judge`` block are the N=1 shape and are accepted only for
    N=1 (that is exactly what ``n_candidates == 1`` produces).
    """
    def _ok(rec: dict) -> bool:
        block = rec.get("bon_judge")
        if block is None:
            return int(n_candidates) == 1
        return int(block.get("n_candidates", 0)) == int(n_candidates)

    return _ok


class BonJBatchedPipeline(BatchedPipeline):
    """Batched lock-step pipeline with a best-of-N refinement stage (BoN-J).

    Everything except the round body is inherited from
    ``core.batched_pipeline.BatchedPipeline`` (``run``/``_run_chunk``/draft
    resolution/resume/``indices``), so each item keeps its global manifest
    position, its stored-draft lookup and its seeds -- a resumed or
    differently-batched run reproduces the same records.

    * ``n_candidates == 1`` delegates every round to ``super()._round``: the
      single-candidate path is literally the existing code rather than a
      re-implementation claiming to match it.
    * ``n_candidates > 1`` draws N revisions of the *same* state in one batched
      engine call.  Candidate ``c`` of ``(sample_id, round t)`` is seeded with
      ``call_seed(sample_id, t, "refine")`` for ``c == 0`` (exactly the seed the
      single-candidate arm uses) and
      ``call_seed(sample_id, t, f"refine_c{c}")`` for ``c > 0``, so the candidate
      set is a pure function of (sample_id, round_index, candidate_index) --
      independent of the batch size, of resume, and of every other sample
      decoded in the same engine call.  The winner is chosen by
      ``BONJ_SELECTION_RULE`` and only the winner goes through the standard
      double-order acceptance test against the current answer.

    Cost is attributed per stage (``sampling_tokens`` = controller + candidate
    generation, ``judging_tokens`` = every judge call including repairs, plus the
    finer ``controller_tokens``/``candidate_tokens`` split) so the Phase 6 token
    axis can separate sampling compute from judging compute.
    """

    _STAGE_KEYS = (
        "controller_tokens", "candidate_tokens", "judging_tokens",
        "sampling_tokens", "controller_calls", "candidate_calls", "judge_calls",
    )

    def __init__(self, *args, n_candidates: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_candidates = max(1, int(n_candidates))
        #: sample_id -> {round_index: detail dict} (REFINE rounds only)
        self.bonj_detail: Dict[str, Dict[int, dict]] = {}

    # -- detail for the persisted record ------------------------------------
    def bonj_trace_block(self, sample_id: str) -> dict:
        per_round = self.bonj_detail.get(sample_id) or {}
        return {
            "n_candidates": self.n_candidates,
            "selection_rule": BONJ_SELECTION_RULE,
            "rounds": [per_round[t] for t in sorted(per_round)],
        }

    # -- cost accounting ----------------------------------------------------
    def _init_stage_cost(self, s) -> None:
        for key in self._STAGE_KEYS:
            s.cost.setdefault(key, 0.0)

    def _add_gen(self, s, gen, stage: str) -> None:
        """Account one generation exactly as the parent does, plus its stage.

        The five keys the parent maintains (input/output/total tokens, latency,
        ``n_calls``) are updated identically, so every total stays comparable
        with every other arm; only the stage buckets are additional.
        """
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            s.cost[key] += getattr(gen, key)
        s.cost["latency_s"] += gen.latency
        s.cost["n_calls"] += 1
        if stage == "controller":
            tkey, ckey = "controller_tokens", "controller_calls"
        elif stage == "candidate":
            tkey, ckey = "candidate_tokens", "candidate_calls"
        elif stage == "judging":
            tkey, ckey = "judging_tokens", "judge_calls"
        else:  # pragma: no cover - programming error
            raise ValueError(f"unknown cost stage {stage!r}")
        s.cost[tkey] += float(gen.total_tokens)
        s.cost[ckey] += 1.0
        if stage != "judging":
            # "sampling" is every non-judge generation (controller + candidates)
            s.cost["sampling_tokens"] = (
                s.cost["controller_tokens"] + s.cost["candidate_tokens"]
            )

    # -- judge plumbing (same template / A-B helper / double-order protocol) --
    def _pair_requests(self, s, task_input: str, ab_round: int, candidate_text: str,
                       current_text: str, seeds: Sequence[int]):
        """The two judge requests of one comparison (arrangement + swap).

        ``candidate_text`` plays the judge's "candidate" and ``current_text`` its
        "current"; the anonymous A/B arrangement comes from the shared
        ``randomised_ab`` helper.  The derived ``ab_round`` is what makes each
        round-robin pair get its own deterministic arrangement instead of every
        pair reusing one global order.
        """
        slot_a, slot_b, cand_is_a = randomised_ab(
            self.cfg.model, self.cfg.task, s.ref.sample_id, ab_round,
            current_text, candidate_text,
        )
        system_prompt = _JUDGE_SYSTEM.format(task_name=self.task_cfg.display_name)
        out = []
        for call_idx, (sa, sb, cia) in enumerate(
            ((slot_a, slot_b, cand_is_a), (slot_b, slot_a, not cand_is_a))
        ):
            out.append(
                (
                    GenRequest(
                        prompt=_JUDGE_TEMPLATE.format(
                            task_input=task_input, slot_a=sa, slot_b=sb
                        ),
                        system_prompt=system_prompt,
                        seed=int(seeds[call_idx]),
                        call_type="judge",
                        max_tokens=JUDGE_MAX_TOKENS,
                        temperature=0.0,
                    ),
                    bool(cia),
                )
            )
        return out

    def _run_judge_batch(self, reqs: List[GenRequest], owners: List) -> List[Tuple[str, str]]:
        """Run judge requests with the one repair retry, accounting cost per owner.

        Returns ``(raw_verdict, reason)`` per request; mapping to
        candidate/current is the caller's job because only the caller knows each
        call's slot assignment.
        """
        gens = self.llm.generate_batch(reqs)
        objs = [_extract_json(g.text) for g in gens]
        broken = [k for k, o in enumerate(objs) if not o or "verdict" not in o]
        if broken:
            rreqs = [
                GenRequest(
                    prompt=reqs[k].prompt + "\n\nYour previous reply was not valid JSON. "
                                            "Reply with ONLY the JSON object.",
                    system_prompt=reqs[k].system_prompt,
                    seed=reqs[k].seed,
                    call_type="judge_repair",
                    max_tokens=JUDGE_MAX_TOKENS,
                    temperature=0.0,
                )
                for k in broken
            ]
            rgens = self.llm.generate_batch(rreqs)
            for k, g2 in zip(broken, rgens):
                self._add_gen(owners[k], g2, "judging")
                o2 = _extract_json(g2.text)
                if o2 and "verdict" in o2:
                    objs[k] = o2
        out: List[Tuple[str, str]] = []
        for k, g in enumerate(gens):
            self._add_gen(owners[k], g, "judging")
            obj = objs[k] or {}
            out.append((str(obj.get("verdict", "uncertain")), str(obj.get("reason", ""))))
        return out

    # -- round override -----------------------------------------------------
    def _round(self, adapter, active: List, t: int) -> None:
        if self.n_candidates == 1:
            # Delegation, not a copy: N=1 IS the existing single-candidate path.
            return super()._round(adapter, active, t)
        return self._round_bonj(adapter, active, t)

    def _round_bonj(self, adapter, active: List, t: int) -> None:
        """One lock-step round with N candidate revisions per active sample.

        Mirrors ``core.batched_pipeline.BatchedPipeline._round`` phase by phase
        (retrieval, controller + repair, refine, judge + repair, accounting), and
        differs only in that the refine phase issues N requests per sample and
        that the judge phase first runs the round-robin selection and then the
        standard winner-vs-current acceptance test.
        """
        n = self.n_candidates
        for s in active:
            self._init_stage_cost(s)

        # ---- phase 1: retrieval (CPU, no engine call) ----------------------
        blocks: Dict[int, str] = {}
        retr: Dict[int, tuple] = {}
        for s in active:
            exp_ids, scores, si, ss, cc = self._retrieve(s.ref, s.current, t)
            retr[id(s)] = (exp_ids, scores, si, ss, cc)
            blocks[id(s)] = render_experience_block(
                [self._by_id(e) for e in exp_ids],
                include_outcome=self.cfg.include_outcome,
                count_tokens=self.llm.count_tokens,
                max_units=self.cfg.k,
                contrastive=self.cfg.render_contrastive,
                advice_mode=self.cfg.advice_mode,
            )

        # ---- phase 2: controller (or forced REFINE) ------------------------
        decisions: Dict[int, tuple] = {}
        if self.cfg.stop_mode == "judge":
            for s in active:
                decisions[id(s)] = ("REFINE", "", "", False)
        else:
            reqs = []
            for s in active:
                reqs.append(
                    GenRequest(
                        prompt=self.controller.build_prompt(
                            s.ref.source,
                            s.current,
                            # v2 moves the examples to the refiner prompt.
                            blocks[id(s)]
                            if (self.cfg.renderer != "v2" or self.cfg.controller_sees_examples)
                            else "",
                            self.cfg.stop_mode,
                            renderer=self.cfg.renderer,
                        ),
                        system_prompt=(
                            "You are the controller of an iterative "
                            f"{self.task_cfg.display_name} system. You decide whether another "
                            "revision round is worthwhile, and if so you write the instruction "
                            "for it. You do not write the answer yourself. Reply with JSON only."
                        ),
                        seed=call_seed(s.ref.sample_id, t, "controller"),
                        call_type="controller", max_tokens=256, temperature=0.0,
                    )
                )
            gens = self.llm.generate_batch(reqs)
            parsed = []
            for s, g, rq in zip(active, gens, reqs):
                self._add_gen(s, g, "controller")
                parsed.append((s, rq, g, _extract_json(g.text)))

            # exactly one repair retry on a malformed reply, as in the parent
            broken = [x for x in parsed if not x[3] or "action" not in x[3]]
            if broken:
                rreqs = [
                    GenRequest(
                        prompt=x[1].prompt + "\n\nYour previous reply was not valid JSON. "
                                             "Reply with ONLY the JSON object.",
                        system_prompt=x[1].system_prompt,
                        seed=x[1].seed,
                        call_type="controller_repair",
                        max_tokens=256,
                        temperature=0.0,
                    )
                    for x in broken
                ]
                rgens = self.llm.generate_batch(rreqs)
                fixed = {}
                for x, g2 in zip(broken, rgens):
                    self._add_gen(x[0], g2, "controller")
                    fixed[id(x[0])] = _extract_json(g2.text)

            for s, rq, g, obj in parsed:
                if not obj or "action" not in obj:
                    obj = fixed.get(id(s))
                if not obj or "action" not in obj:
                    decisions[id(s)] = ("STOP", "", "structured output failed", True)
                    continue
                action = str(obj.get("action", "")).strip().upper()
                if self.cfg.stop_mode == "fixed":
                    action = "REFINE"
                if action not in ("REFINE", "STOP"):
                    action = "REFINE" if str(obj.get("instruction", "")).strip() else "STOP"
                decisions[id(s)] = (
                    action, str(obj.get("instruction", "")).strip(),
                    str(obj.get("reason", "")).strip(), False,
                )

        # ---- phase 3: STOP bookkeeping (identical to the parent) -----------
        refiners = []
        for s in active:
            action, instruction, reason, sfail = decisions[id(s)]
            if action != "REFINE":
                exp_ids, scores, si, ss, cc = retr[id(s)]
                s.rounds.append(RoundTrace(
                    round_index=t, exp_ids=exp_ids, exp_scores=scores, exp_sim_input=si,
                    exp_sim_state=ss, exp_class_counts=cc, controller_action="STOP",
                    controller_instruction="", controller_reason=reason,
                    controller_structured_failure=sfail, candidate="", judge_verdict="",
                    judge_order_consistent=False, judge_reason_a="", judge_reason_b="",
                    accepted=False, metric_offline=None, delta_offline=None,
                    cost=dict(s.cost),
                ))
                s.active = False
            else:
                refiners.append((s, instruction, reason, sfail))

        if not refiners:
            return

        # ---- phase 4: N candidate revisions per sample, ONE batched call ---
        rreqs: List[GenRequest] = []
        owners: List[tuple] = []
        for s, instruction, reason, sfail in refiners:
            instr = instruction or DEFAULT_REFINE_INSTRUCTION
            example = TaskExample(index=s.index, source=s.ref.source,
                                  reference=s.ref.reference, task=self.cfg.task)
            prompt = build_refine_prompt(
                adapter, example, s.current, instr,
                # v2: the aligned contrastive examples condition the revision
                # directly, as in the original method.
                experience_block=blocks.get(id(s), ""),
                renderer=self.cfg.renderer,
            )
            for c in range(n):
                # c == 0 keeps the single-candidate arm's exact seed; c > 0 gets
                # its own pure function of (sample, round, candidate).
                seed = (
                    call_seed(s.ref.sample_id, t, "refine") if c == 0
                    else call_seed(s.ref.sample_id, t, f"refine_c{c}")
                )
                rreqs.append(
                    GenRequest(
                        prompt=prompt, system_prompt=adapter.system_prompt(),
                        seed=seed, call_type="refine", max_tokens=self.budget.max_tokens,
                        temperature=self.budget.temperature, top_p=self.budget.top_p,
                    )
                )
                owners.append((s, c, seed))
        rgens = self.llm.generate_batch(rreqs)
        cands: Dict[int, List[str]] = {id(s): [] for s, *_ in refiners}
        cand_seeds: Dict[int, List[int]] = {id(s): [] for s, *_ in refiners}
        for (s, c, seed), g in zip(owners, rgens):
            self._add_gen(s, g, "candidate")
            cands[id(s)].append(adapter.parse_output(g.text))
            cand_seeds[id(s)].append(int(seed))

        # ---- phase 5: round-robin selection over the same judge ------------
        sel_reqs: List[GenRequest] = []
        sel_meta: List[tuple] = []          # (state, i, j, call_idx, candidate_is_a)
        for s, _i, _r, _f in refiners:
            cs = cands[id(s)]
            pair_index = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if cs[i] == cs[j]:
                        pair_index += 1      # tie by definition, no engine call
                        continue
                    seeds = tuple(
                        call_seed(s.ref.sample_id, t, f"judge_select_{i}_{j}_{k}")
                        for k in (0, 1)
                    )
                    for call_idx, (req, cia) in enumerate(self._pair_requests(
                        s, s.ref.source, (t + 1) * 1000 + pair_index, cs[i], cs[j], seeds
                    )):
                        sel_reqs.append(req)
                        sel_meta.append((s, i, j, call_idx, cia))
                    pair_index += 1
        sel_results = (
            self._run_judge_batch(sel_reqs, [m[0] for m in sel_meta])
            if sel_reqs else []
        )
        pair_calls: Dict[int, Dict[Tuple[int, int], List[Optional[Tuple[str, str]]]]] = {
            id(s): {} for s, *_ in refiners
        }
        for (s, i, j, call_idx, _cia), (vraw, reason) in zip(sel_meta, sel_results):
            pair_calls[id(s)].setdefault((i, j), [None, None])[call_idx] = (
                _map_verdict(vraw, _cia), reason
            )

        winners: Dict[int, int] = {}
        comparisons: Dict[int, List[dict]] = {}
        copeland: Dict[int, List[int]] = {}
        for s, _i, _r, _f in refiners:
            cs = cands[id(s)]
            decisive: Dict[Tuple[int, int], Optional[int]] = {}
            rows: List[dict] = []
            for i in range(n):
                for j in range(i + 1, n):
                    if cs[i] == cs[j]:
                        rows.append({
                            "i": i, "j": j, "verdict": "tie", "order_consistent": False,
                            "reason_a": _IDENTICAL_PAIR_REASON,
                            "reason_b": _IDENTICAL_PAIR_REASON,
                            "identical": True, "judge_calls": 0,
                        })
                        decisive[(i, j)] = None
                        continue
                    c0 = pair_calls[id(s)][(i, j)][0]
                    c1 = pair_calls[id(s)][(i, j)][1]
                    assert c0 is not None and c1 is not None
                    v = _combine(c0[0], c1[0])
                    rows.append({
                        "i": i, "j": j, "verdict": v,
                        "order_consistent": c0[0] == c1[0] and c0[0] in ("candidate", "current"),
                        "reason_a": c0[1], "reason_b": c1[1],
                        "identical": False, "judge_calls": 2,
                    })
                    decisive[(i, j)] = i if v == "better" else (j if v == "worse" else None)
            winners[id(s)] = bonj_select_winner(n, decisive)
            comparisons[id(s)] = rows
            copeland[id(s)] = bonj_copeland_scores(n, decisive)

        # ---- phase 6: standard acceptance test, winner vs current ----------
        acc_reqs: List[GenRequest] = []
        acc_meta: List[tuple] = []          # (state, call_idx, candidate_is_a)
        for s, _i, _r, _f in refiners:
            winner_text = cands[id(s)][winners[id(s)]]
            if winner_text == s.current:
                continue                    # identical pair: tie, no engine call
            seeds = tuple(call_seed(s.ref.sample_id, t * 10 + k, "judge") for k in (0, 1))
            for call_idx, (req, cia) in enumerate(self._pair_requests(
                s, s.ref.source, t, winner_text, s.current, seeds
            )):
                acc_reqs.append(req)
                acc_meta.append((s, call_idx, cia))
        acc_results = (
            self._run_judge_batch(acc_reqs, [m[0] for m in acc_meta])
            if acc_reqs else []
        )
        acc_calls: Dict[int, List[Optional[Tuple[str, str]]]] = {id(s): [None, None] for s, *_ in refiners}
        for (s, call_idx, cia), (vraw, reason) in zip(acc_meta, acc_results):
            acc_calls[id(s)][call_idx] = (_map_verdict(vraw, cia), reason)

        # ---- phase 7: combine and advance ----------------------------------
        for (s, instruction, reason, sfail) in refiners:
            winner = winners[id(s)]
            cand = cands[id(s)][winner]
            calls = acc_calls[id(s)]
            if calls[0] is None or calls[1] is None:
                v1, r1 = "tie", _IDENTICAL_PAIR_REASON
                v2, r2 = "tie", _IDENTICAL_PAIR_REASON
                identical_to_current = True
            else:
                v1, r1 = calls[0]
                v2, r2 = calls[1]
                identical_to_current = False
            verdict = _combine(v1, v2)
            order_consistent = v1 == v2 and v1 in ("candidate", "current")
            # Same shared rule as every other path -- do NOT re-implement it.
            accepted = accept_revision(verdict, order_consistent, s.current, cand,
                                       self.accept_max_len_ratio)
            prev = self.scorer.primary(s.ref.reference, s.current)
            cand_m = self.scorer.primary(s.ref.reference, cand)
            exp_ids, scores, si, ss, cc = retr[id(s)]
            s.rounds.append(RoundTrace(
                round_index=t, exp_ids=exp_ids, exp_scores=scores, exp_sim_input=si,
                exp_sim_state=ss, exp_class_counts=cc, controller_action="REFINE",
                controller_instruction=instruction, controller_reason=reason,
                controller_structured_failure=sfail, candidate=cand,
                judge_verdict=verdict, judge_order_consistent=order_consistent,
                judge_reason_a=r1, judge_reason_b=r2, accepted=accepted,
                metric_offline=cand_m, delta_offline=cand_m - prev, cost=dict(s.cost),
            ))
            self.bonj_detail.setdefault(s.ref.sample_id, {})[t] = {
                "round_index": t,
                "n_candidates": n,
                "candidate_seeds": list(cand_seeds[id(s)]),
                "candidates": [
                    {
                        "index": c,
                        "seed": int(cand_seeds[id(s)][c]),
                        "sha256": sha256_text(cands[id(s)][c])[:16],
                        "chars": len(cands[id(s)][c]),
                    }
                    for c in range(n)
                ],
                "comparisons": comparisons[id(s)],
                "copeland_scores": copeland[id(s)],
                "winner_index": winner,
                "winner_sha256": sha256_text(cand)[:16],
                "identical_to_current": identical_to_current,
                "accept": {
                    "verdict": verdict,
                    "order_consistent": order_consistent,
                    "accepted": accepted,
                    "reason_a": r1,
                    "reason_b": r2,
                    "judge_calls": 0 if identical_to_current else 2,
                },
            }
            if accepted:
                s.current = cand
            elif self.cfg.stop_mode == "judge":
                s.active = False


def run_model_task(
    *,
    arm: str,
    model: str,
    task: str,
    seed: int,
    gpu: str,
    split: str,
    tag: str,
    limit: Optional[int],
    max_rounds: int,
    alpha: float,
    k: int,
    draft_source: str,
    draft_cache: str,
    batch_size: int,
    llm,
    out_dir: Path,
    n_candidates: int = 1,
    accept_max_len_ratio: Optional[float] = None,
    renderer: Optional[str] = None,
    controller_sees_examples: bool = False,
    advice_mode: Optional[str] = None,
    retrieval_excludes_own_source: bool = False,
    optimization: Optional[dict] = None,
) -> dict:
    spec = ARM_REGISTRY[arm]
    quality_evaluator = None
    component_setup_s = 0.0
    if optimization:
        if arm not in ("full_static", "full_online", "fixed_rounds", "no_experience", "sr_j_fixed", "sr_j_stop", "positive_only", "random_retrieve"):
            raise ValueError("arm is not defined for the BERT optimized protocol")
        if n_candidates != 1 or accept_max_len_ratio is not None:
            raise ValueError("optimized policy owns the length gate and supports one candidate")
        if not retrieval_excludes_own_source:
            raise ValueError("optimized runs require own-source exclusion")
        if renderer not in (None, "v2") or controller_sees_examples:
            raise ValueError("optimized runs use v2 and the independent BERT controller")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
    n_candidates = int(n_candidates)
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1 (got {n_candidates})")
    if n_candidates > 1 and arm != "bon_judge":
        raise ValueError(
            f"--n-candidates > 1 is implemented for arm 'bon_judge' only "
            f"(got arm={arm!r}); computing another arm with N>1 would silently "
            f"change its single-variable meaning"
        )
    if n_candidates > 1 and not (batch_size and batch_size > 1):
        raise ValueError(
            "--n-candidates > 1 is implemented on the batched lock-step path only; "
            f"pass --batch-size > 1 (got batch_size={batch_size})"
        )
    # Stored Direct-Zero drafts exist only for the *test* prefix, and the
    # lookup is positional.  Using them for another split would silently pair a
    # test item's draft with an auxiliary item's reference (measured: every
    # initial metric collapsed to 0.00).  Fail loudly instead.
    if split != "test" and draft_source == "stored":
        raise ValueError(
            f"draft_source='stored' is only valid for split='test' "
            f"(got split={split!r}); pass --draft-source generate"
        )
    refs = read_manifest(manifest_path(task, split))
    if limit is not None:
        refs = refs[:limit]

    library = build_library(model, task, spec["pool"])
    online = bool(spec["online"])
    retriever = ExperienceRetriever(library) if library else None
    if optimization:
        from core.optimization_config import build_components
        setup_start = time.perf_counter()
        quality_evaluator, retriever = build_components(optimization, library)
        component_setup_s = time.perf_counter() - setup_start

    # The memo must be keyed by retrieval SEMANTICS as well as by the library:
    # two runs over the same library that retrieve different ids (here, with and
    # without the own-source exclusion) must never share a memo file, or the
    # cached ids silently reinstate whatever the other run did.
    _cache_variant = "ownsrc-excluded" if retrieval_excludes_own_source else "ownsrc-kept"
    cache = (RetrievalCache(cache_path(model, task, "initial",
                                       library_fingerprint(library),
                                       variant=_cache_variant))
             if library else None)

    cfg = RunConfig(
        task=task,
        model=model,
        experience_mode=spec["experience_mode"],
        stop_mode=spec["stop_mode"],
        alpha=alpha,
        k=k,
        max_rounds=max_rounds,
        seed=seed,
        snapshot_id="initial" if not online else "online",
        draft_source=draft_source,
        draft_cache=draft_cache,
        renderer=renderer or spec.get("renderer", "v2"),
        controller_sees_examples=(controller_sees_examples
                                  or bool(spec.get("controller_sees_examples", False))),
        advice_mode=advice_mode or spec.get("advice_mode", "summary"),
        retrieval_excludes_own_source=retrieval_excludes_own_source,
        optimization=optimization or {},
    )
    if optimization:
        cfg.optimization = {**cfg.optimization, "library_fingerprint": library_fingerprint(library),
                            "batch_size": batch_size,
                            "cost_accounting": "initial_and_refinement_v2"}
        import hashlib
        profile_id = hashlib.sha256(json.dumps(cfg.optimization, sort_keys=True).encode()).hexdigest()[:16]
        tag = "/".join(filter(None, (tag, "optimized-" + profile_id)))
        cache = None

    # This used to reject every non-`stored` draft source under batching, which
    # forced all dev / auxiliary runs to batch_size=1 and made them 5.6x slower.
    # `BatchedPipeline._run_chunk` resolves missing drafts in ONE batched call
    # (its `need` branch) and the `cached` path writes back through the same
    # cache, so batching is supported for all three sources; the only genuinely
    # unsafe case is `stored` off the test split, which the positional-draft
    # check above already refuses.  Batching does change the *sampled text* (vLLM
    # is not batch-invariant, which the project already documents) but not the
    # per-call seeds, so this is the same regime every test run already uses.
    if batch_size and batch_size > 1 and draft_source == "cached" and not draft_cache:
        raise ValueError("draft_source='cached' requires --draft-cache")
    # BoN-J has one point per N and the config hash is (by design) identical
    # across N -- n_candidates is a pipeline argument, not a RunConfig field, so
    # every existing arm's config hash stays frozen.  Without a tag, two N values
    # would share one output directory and a resume could mix them, so the point
    # tag is derived from N when the caller does not supply one.
    if arm == "bon_judge" and n_candidates > 1 and not tag:
        tag = f"n{n_candidates}"
        print(f"    [bon_judge] n_candidates={n_candidates} -> output tag {tag!r} "
              f"(pass --tag to override)", flush=True)
    run_dir = out_dir / arm / f"seed{seed}" / task / model
    if split != "test":
        run_dir = run_dir / split
    if tag:
        run_dir = run_dir / tag

    if optimization or (batch_size and batch_size > 1):
        from core.batched_pipeline import BatchedPipeline

        run_dir.mkdir(parents=True, exist_ok=True)
        stem = run_dir / arm
        jpath, cpath = stem.with_suffix(".jsonl"), stem.with_suffix(".csv")
        fields = ["task", "model", "arm", "seed", "sample_id", "n_rounds",
                  "initial_metric_offline", "final_metric_offline",
                  "total_tokens", "latency_s", "n_calls"]

        # ---- resume ---------------------------------------------------------
        # These files used to be opened "w", so every restart -- an engine
        # death, a supervisor retry, an operator restart -- silently discarded
        # every sample already computed.  Keep the persisted prefix instead and
        # run only the missing manifest positions, preserving each item's
        # ORIGINAL index so its seeds, stored-draft lookup and TaskExample.index
        # are identical to what a from-scratch run would have produced.
        prior, prior_ids, dropped = _load_prior(
            jpath, cfg.config_hash, task, model, seed,
            # n_candidates is not part of the config hash, so BoN-J needs its own
            # guard: a record produced at another N must never be reused here.
            require=_bonj_record_filter(n_candidates) if arm == "bon_judge" else None,
        )
        if online and prior:
            # The online arm builds its library from every sample it has already
            # seen, in order.  Replaying only the tail would hand it a library it
            # could never have had, so a partial online run must start over.
            print(f"    [resume] {arm}/{task}/{model}: DISCARDING {len(prior)} "
                  f"records -- the online arm cannot resume", flush=True)
            dropped += len(prior)
            prior, prior_ids = [], set()
        pending = [i for i, ref in enumerate(refs) if ref.sample_id not in prior_ids]
        if dropped:
            print(f"    [resume] {arm}/{task}/{model}: dropped {dropped} "
                  f"unusable/stale/duplicate records", flush=True)
        print(f"    [resume] {arm}/{task}/{model}: {len(prior)} already done, "
              f"{len(pending)} to run, {len(refs)} total", flush=True)

        with jpath.open("w", encoding="utf-8") as jf, cpath.open(
            "w", encoding="utf-8", newline=""
        ) as cf:
            w = csv.DictWriter(cf, fieldnames=fields)
            w.writeheader()
            # Re-emit the kept prefix first, so the file on disk is never worse
            # than before this attempt even if the engine dies immediately.
            for tr in prior:
                rec = tr.to_dict()
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                w.writerow(_csv_row(rec, task, model, arm, seed))
            jf.flush()
            cf.flush()

            def _sink(chunk_traces, done, total):
                for tr in chunk_traces:
                    rec = _emit_record(tr, bonj_pipe)
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    w.writerow(_csv_row(rec, task, model, arm, seed))
                jf.flush()
                cf.flush()

            # n_candidates == 1 uses the unmodified single-candidate pipeline;
            # N > 1 swaps in the BoN-J round, which inherits everything else
            # (indices/resume/drafts/retrieval) unchanged.
            if optimization:
                from core.optimized_pipeline import OptimizedBatchedPipeline
                pipe = OptimizedBatchedPipeline(cfg, retriever, library, llm, cache=None,
                                               batch_size=batch_size, online=online,
                                               quality_evaluator=quality_evaluator)
                bonj_pipe = None
            elif n_candidates > 1:
                pipe = BonJBatchedPipeline(cfg, retriever, library, llm, cache=cache,
                                           batch_size=batch_size, online=online,
                                           n_candidates=n_candidates,
                                           accept_max_len_ratio=accept_max_len_ratio)
                bonj_pipe = pipe
            else:
                pipe = BatchedPipeline(cfg, retriever, library, llm, cache=cache,
                                       batch_size=batch_size, online=online,
                                       accept_max_len_ratio=accept_max_len_ratio)
                bonj_pipe = None
            t0 = time.time()
            traces = pipe.run(refs, sink=_sink, indices=pending) if pending else []
            if bonj_pipe is not None:
                traces = [_BonJRecord(tr, bonj_pipe.bonj_trace_block(tr.sample_id))
                          for tr in traces]
        merged = prior + traces
        if pending:
            _persist_merged(jpath, cpath, merged, refs, fields, task, model, arm, seed)
        # The online arm's whole point is the accumulated library, and the
        # batched path (the default) previously kept it only in memory, so a
        # supervised `full_online` run left no artefact at all.  Persist it with
        # exactly the same call the unbatched path uses.
        if online and pipe.library:
            lib_path = (run_dir / "online_experience.json" if optimization else
                        experience_dir(model, task) / f"online_seed{seed}.jsonl")
            if optimization:
                lib_path.write_text(json.dumps([e.to_dict() for e in pipe.library], ensure_ascii=False), encoding="utf-8")
            else:
                save_experiences(pipe.library, lib_path)
            print(f"    [online] saved {len(pipe.library)} accumulated experiences "
                  f"-> {lib_path}", flush=True)
        summary = summarize(merged, arm, task, model, seed, time.time() - t0)
        summary["batched"] = True
        summary["batch_size"] = batch_size
        summary["resumed_from"] = len(prior)
        summary["accept_max_len_ratio"] = accept_max_len_ratio
        if optimization:
            summary["optimization"] = cfg.optimization
            summary["component_setup_s"] = component_setup_s
        if arm == "bon_judge":
            summary["n_candidates"] = n_candidates
            summary["selection_rule"] = BONJ_SELECTION_RULE if n_candidates > 1 else None
        stem.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    pipe = Pipeline(cfg, retriever, library, llm, cache=cache,
                    accept_max_len_ratio=accept_max_len_ratio)

    run_dir.mkdir(parents=True, exist_ok=True)
    stem = run_dir / f"{arm}"
    jsonl_path = stem.with_suffix(".jsonl")
    csv_path = stem.with_suffix(".csv")
    summary_path = stem.with_suffix(".summary.json")

    traces: List[SampleTrace] = []
    started = time.time()
    with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=[
                "task", "model", "arm", "seed", "sample_id", "n_rounds",
                "initial_metric_offline", "final_metric_offline",
                "total_tokens", "latency_s", "n_calls",
            ],
        )
        writer.writeheader()
        for i, ref in enumerate(refs):
            tr = pipe.run_sample(ref, i)
            traces.append(tr)
            jf.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
            jf.flush()
            writer.writerow(
                {
                    "task": task, "model": model, "arm": arm, "seed": seed,
                    "sample_id": tr.sample_id, "n_rounds": len(tr.rounds),
                    "initial_metric_offline": tr.initial_metric_offline,
                    "final_metric_offline": tr.final_metric_offline,
                    "total_tokens": tr.cost["total_tokens"],
                    "latency_s": tr.cost["latency_s"],
                    "n_calls": tr.cost["n_calls"],
                }
            )
            # Online arm: experiences are committed only after the sample ends,
            # so an item can never influence its own decisions.
            if online:
                new = transitions_to_experiences(
                    tr, model, source_input=ref.source, provenance="online",
                    # Evolution stores wrong->right pairs only.
                    only_improving=True,
                )
                if new:
                    library.extend(new)
                    retriever = ExperienceRetriever(library)
                    pipe.retriever = retriever
                    pipe.library = library
            if (i + 1) % 25 == 0:
                print(f"    [{arm} {task}/{model}] {i + 1}/{len(refs)}", flush=True)

    if online and library:
        save_experiences(library, experience_dir(model, task) / f"online_seed{seed}.jsonl")

    summary = summarize(traces, arm, task, model, seed, time.time() - started)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _finish_batched(traces, arm, task, model, seed, run_dir, started):
    """Persist a batched run in exactly the same triple as the unbatched path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = run_dir / arm
    jsonl_path = stem.with_suffix(".jsonl")
    csv_path = stem.with_suffix(".csv")
    summary_path = stem.with_suffix(".summary.json")
    with jsonl_path.open("w", encoding="utf-8") as jf, csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as cf:
        w = csv.DictWriter(cf, fieldnames=[
            "task", "model", "arm", "seed", "sample_id", "n_rounds",
            "initial_metric_offline", "final_metric_offline",
            "total_tokens", "latency_s", "n_calls",
        ])
        w.writeheader()
        for tr in traces:
            jf.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
            w.writerow({
                "task": task, "model": model, "arm": arm, "seed": seed,
                "sample_id": tr.sample_id, "n_rounds": len(tr.rounds),
                "initial_metric_offline": tr.initial_metric_offline,
                "final_metric_offline": tr.final_metric_offline,
                "total_tokens": tr.cost["total_tokens"],
                "latency_s": tr.cost["latency_s"], "n_calls": tr.cost["n_calls"],
            })
    summary = summarize(traces, arm, task, model, seed, time.time() - started)
    summary["batched"] = True
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class _PriorRecord:
    """A persisted trace rehydrated just enough for ``summarize``/re-persist.

    Resume must not require the original dataclasses to round-trip perfectly:
    it only needs the fields the summary and the CSV use, and the raw dict so
    the JSONL can be rewritten byte-for-byte.
    """

    __slots__ = ("_rec", "sample_id", "rounds", "cost",
                 "initial_metric_offline", "final_metric_offline")

    def __init__(self, rec: dict) -> None:
        self._rec = rec
        self.sample_id = rec.get("sample_id")
        self.rounds = [SimpleNamespace(**r) for r in (rec.get("rounds") or [])]
        self.cost = rec.get("cost") or {}
        self.initial_metric_offline = rec.get("initial_metric_offline")
        self.final_metric_offline = rec.get("final_metric_offline")

    def to_dict(self) -> dict:
        return self._rec


def _csv_row(rec: dict, task: str, model: str, arm: str, seed: int) -> dict:
    cost = rec.get("cost") or {}
    return {
        "task": task, "model": model, "arm": arm, "seed": seed,
        "sample_id": rec.get("sample_id"),
        "n_rounds": len(rec.get("rounds") or []),
        "initial_metric_offline": rec.get("initial_metric_offline"),
        "final_metric_offline": rec.get("final_metric_offline"),
        "total_tokens": cost.get("total_tokens"),
        "latency_s": cost.get("latency_s"),
        "n_calls": cost.get("n_calls"),
    }


def _load_prior(jpath: Path, config_hash: str, task: str, model: str, seed: int,
                require=None):
    """Load records a previous attempt already persisted for this exact run.

    A record is kept only when it provably belongs here: identical config hash
    (so a row produced under a different prompt, alpha or budget can never be
    resurrected), identical task/model/seed, a parseable line, and no earlier
    duplicate of the same sample id.  Everything else is counted and dropped
    loudly -- silently mixing configs is exactly the failure this guards.

    ``require`` is an optional extra predicate on the raw record, used by BoN-J
    to reject records produced at a different ``n_candidates`` (which is not part
    of the config hash).  ``None`` -- the default, and what every existing caller
    passes -- leaves the behaviour unchanged.
    """
    prior: List[_PriorRecord] = []
    seen = set()
    dropped = 0
    if not jpath.exists():
        return prior, seen, dropped
    for line in jpath.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1                      # truncated tail of an interrupted write
            continue
        if (rec.get("config_hash") != config_hash
                or rec.get("task") != task
                or rec.get("model") != model
                or rec.get("seed") != seed):
            dropped += 1
            continue
        if require is not None and not require(rec):
            dropped += 1
            continue
        sid = rec.get("sample_id")
        if sid is None or sid in seen:
            dropped += 1
            continue
        seen.add(sid)
        prior.append(_PriorRecord(rec))
    return prior, seen, dropped


def _persist_merged(jpath: Path, cpath: Path, merged: List, refs, fields,
                    task: str, model: str, arm: str, seed: int) -> None:
    """Rewrite JSONL+CSV in manifest order, atomically.

    Called only once a run has finished, so a crash during the rewrite cannot
    lose work: the temp files are fsynced and then ``os.replace``d over the
    originals.
    """
    order = {ref.sample_id: i for i, ref in enumerate(refs)}
    ordered = sorted(merged, key=lambda t: order.get(t.sample_id, len(order)))
    tmp_j = jpath.with_suffix(".jsonl.tmp")
    tmp_c = cpath.with_suffix(".csv.tmp")
    with tmp_j.open("w", encoding="utf-8") as jf, tmp_c.open(
        "w", encoding="utf-8", newline=""
    ) as cf:
        w = csv.DictWriter(cf, fieldnames=fields)
        w.writeheader()
        for tr in ordered:
            rec = tr.to_dict()
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.writerow(_csv_row(rec, task, model, arm, seed))
        jf.flush()
        cf.flush()
        os.fsync(jf.fileno())
        os.fsync(cf.fileno())
    os.replace(tmp_j, jpath)
    os.replace(tmp_c, cpath)


def summarize(
    traces: List[SampleTrace], arm: str, task: str, model: str, seed: int, duration: float
) -> dict:
    n = len(traces)
    if not n:
        return {"arm": arm, "task": task, "model": model, "seed": seed, "n": 0}
    verdicts: Counter = Counter()
    actions: Counter = Counter()
    n_refine = 0
    n_accepted = 0
    totals = defaultdict(float)
    for t in traces:
        for r in t.rounds:
            if r.controller_action == "REFINE":
                n_refine += 1
                verdicts[r.judge_verdict] += 1
                if r.accepted:
                    n_accepted += 1
            actions[r.controller_action] += 1
        for kk, vv in t.cost.items():
            totals[kk] += vv
    return {
        "arm": arm, "task": task, "model": model, "seed": seed,
        "n": n,
        "mean_initial_metric": sum(t.initial_metric_offline or 0.0 for t in traces) / n,
        "mean_final_metric": sum(t.final_metric_offline or 0.0 for t in traces) / n,
        "mean_rounds": sum(len(t.rounds) for t in traces) / n,
        "refine_attempts": n_refine,
        "accepted": n_accepted,
        "accept_rate": (n_accepted / n_refine) if n_refine else 0.0,
        "judge_verdicts": dict(verdicts),
        "controller_actions": dict(actions),
        "cost": {k: v / n for k, v in totals.items()},
        "duration_s": duration,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_REGISTRY))
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--models", default="glm4-9b")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="samples advanced in lock-step; 1 = the original sequential path")
    ap.add_argument("--tag", default="", help="extra output subdir; REQUIRED when sweeping a parameter")
    ap.add_argument("--split", default="test",
                    choices=["test", "dev", "initial", "accumulation"],
                    help="dev is auxiliary data used only for alpha selection / gates")
    ap.add_argument("--limit", type=int, default=None, help="debug only: first N items")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--draft-source", default="stored", choices=["stored", "generate", "cached"])
    ap.add_argument("--draft-cache", default="", help="JSONL draft cache; required for --draft-source cached")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--experience-root", default=None,
                    help="override the experience-library root (used to run with a "
                         "rebuilt library without disturbing the existing one)")
    ap.add_argument("--accept-max-len-ratio", type=float, default=None,
                    help="acceptance-gate length guard: reject a revision longer than "
                         "ratio x the current answer even if the judge prefers it. "
                         "Default None = the historical gate, bit-for-bit. Chosen on "
                         "dev and frozen; never tuned on test.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--retrieval-excludes-own-source", action="store_true",
                    help="drop the query item's own experience unit. MANDATORY for a "
                         "valid test-split claim: the library is built from assets that "
                         "contain the test items, so without this the refiner is shown "
                         "the gold answer to the sentence it is translating.")
    ap.add_argument("--advice-mode", default=None, choices=["summary", "full"],
                    help="v2 only: 'summary' = the original method's last paragraph; "
                         "'full' = the whole critique (the v1 renderer's behaviour)")
    ap.add_argument("--controller-sees-examples", action="store_true",
                    help="v2 only: also show the retrieved examples to the controller "
                         "(default: the refiner alone sees them)")
    ap.add_argument("--renderer", default=None, choices=["v1", "v2"],
                    help="experience-block renderer generation. Default: the arm's own "
                         "setting, which is v2 for every schedulable arm. Pass v1 "
                         "explicitly only to resume a run recorded before v2 landed "
                         "(v1 is excluded from config_hash, so its identity is preserved).")
    ap.add_argument("--n-candidates", type=int, default=1,
                    help="BoN-J (--arm bon_judge) best-of-N: draw N candidate revisions "
                         "per round and pick the winner with the same pairwise judge. "
                         "1 = the existing single-candidate path (default). "
                         "N > 1 requires --batch-size > 1 and gets an automatic "
                         "'n<N>' output tag when --tag is empty.")
    ap.add_argument("--optimization-config", default=None,
                    help="opt-in BERT/hybrid/delayed-feedback profile; gets isolated output identity")
    args = ap.parse_args()
    optimizations = {}
    if args.optimization_config:
        from core.optimization_config import resolve_optimization
        selected = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
        optimizations = {t.strip(): resolve_optimization(args.optimization_config, t.strip()) for t in selected}
        if (args.arm not in ("full_static", "full_online", "fixed_rounds", "no_experience", "sr_j_fixed", "sr_j_stop", "positive_only", "random_retrieve")
                or args.n_candidates != 1 or args.accept_max_len_ratio is not None
                or not args.retrieval_excludes_own_source or args.batch_size < 1
                or args.renderer not in (None, "v2") or args.controller_sees_examples):
            ap.error("incompatible optimized protocol options; require own-source exclusion, v2, one candidate, and profile-owned length policy")


    if args.n_candidates < 1:
        print(f"REFUSING: --n-candidates must be >= 1 (got {args.n_candidates})")
        return 2
    if args.n_candidates > 1 and args.arm != "bon_judge":
        print(f"REFUSING: --n-candidates > 1 is implemented for --arm bon_judge only "
              f"(got --arm {args.arm})")
        return 2
    if args.n_candidates > 1 and args.batch_size <= 1:
        print("REFUSING: --n-candidates > 1 is implemented on the batched lock-step "
              "path only; pass --batch-size > 1")
        return 2

    gpus = [g.strip() for g in args.gpu.split(",") if g.strip()]
    from core import ALLOWED_GPUS, MAX_GPUS
    illegal = [g for g in gpus if g not in ALLOWED_GPUS]
    if illegal or len(gpus) > MAX_GPUS:
        print(f"REFUSING: {gpus} violates the GPU budget (any {MAX_GPUS} of {ALLOWED_GPUS})")
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
    ensure_gpu_whitelist()

    models = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    tasks = list(TASKS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    out_dir = Path(args.out_dir) if args.out_dir else EXP_ROOT / "runs" / "main"

    # ---- validate BEFORE loading a model ---------------------------------
    # This check also lives in run_model_task, but by then the vLLM engine is
    # already up.  Raising there killed run_experiment.py while leaving the
    # EngineCore child alive and holding ~37 GiB with ppid=1 -- an orphan that
    # then made every later load on that card fail (observed three times, once
    # as a self-sustaining retry loop).  Refusing early costs one line and
    # cannot orphan an engine, because none has been created yet.
    # NOTE: this must stay in step with the identical check inside
    # run_model_task.  It previously refused every non-`stored` source under
    # batching, while run_model_task had already been relaxed to allow
    # generate/cached -- so this early gate kept rejecting aux runs (which must
    # generate their own drafts) with exit code 2, and the four high-powered
    # validation jobs burned their retries before anyone noticed.  Two copies of
    # one rule diverged; the fix is that both now test the SAME condition.
    if args.batch_size and args.batch_size > 1 and args.draft_source == "cached" \
            and not args.draft_cache:
        print("REFUSING: --draft-source cached needs --draft-cache "
              f"(got batch_size={args.batch_size}).", flush=True)
        return 2

    if args.experience_root:
        from core.experience import set_experience_root
        set_experience_root(args.experience_root)
        print(f"[exp] experience root -> {args.experience_root}", flush=True)

    from baseline_core.llm import LLMClient

    import torch

    results = []
    for model in models:
        # A checkpoint costs minutes to load from the NFS-backed model mounts
        # (llama3.1 needed >3 min for a single shard under I/O contention), so
        # each model is loaded exactly once and reused for every task.
        print(f"[LOAD] model={model}", flush=True)
        llm = LLMClient(
            resolve_model_config(model),
            backend="vllm",
            gpu=gpus[0],
            gpu_memory_utilization=gpu_mem_util(model, args.gpu_memory_utilization),
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
        )
        for task in tasks:
            print(f"[RUN] arm={args.arm} model={model} task={task} seed={args.seed}", flush=True)
            s = run_model_task(
                arm=args.arm, model=model, task=task, seed=args.seed, gpu=gpus[0],
                split=args.split, tag=args.tag, limit=args.limit, max_rounds=args.max_rounds, alpha=args.alpha, k=args.k,
                draft_source=args.draft_source, draft_cache=args.draft_cache,
                batch_size=args.batch_size,
                llm=llm, out_dir=out_dir,
                n_candidates=args.n_candidates,
                accept_max_len_ratio=args.accept_max_len_ratio,
                renderer=args.renderer,
                controller_sees_examples=args.controller_sees_examples,
                advice_mode=args.advice_mode,
                retrieval_excludes_own_source=args.retrieval_excludes_own_source,
                optimization=optimizations.get(task),
            )
            results.append(s)
            print(
                f"    final={s.get('mean_final_metric', float('nan')):.2f} "
                f"init={s.get('mean_initial_metric', float('nan')):.2f} "
                f"rounds={s.get('mean_rounds', 0):.2f} "
                f"accept={s.get('accept_rate', 0):.2%}",
                flush=True,
            )
        del llm
        torch.cuda.empty_cache()
    print(json.dumps(results, ensure_ascii=False, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
