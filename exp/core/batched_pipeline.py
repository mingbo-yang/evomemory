"""Round-lockstep batched execution of the refinement loop.

The unbatched pipeline processes one sample at a time, which forces batch size 1
at the engine.  Samples are independent of each other (only the rounds *within*
a sample are sequential), so this module advances many samples in lock-step and
issues exactly one batched engine call per phase:

    round t:  [controller x active]  ->  [refine x refiners]  ->  [judge x 2N]

Prompt text, retrieved experience ids, A/B ordering and per-request seeds are all
computed by the same helpers as the unbatched path, so the single-variable
invariants carry over unchanged.  What differs is only the sampling batch, which
is the same nondeterminism source that is already measured and documented.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from baseline_core.config import TASK_CONFIGS
from baseline_core.types import InferenceBudget, TaskExample

from .batch_llm import BatchedLLM, GenRequest
from .bm25_fields import Experience, ExperienceRetriever, deterministic_random_rank
from .controller import Controller
from .determinism import call_seed, randomised_ab
from .experience import render_experience_block
from .judge import (PairwiseJudge, _combine, _extract_json, _map_verdict,
                    accept_revision, JUDGE_MAX_TOKENS)
from .manifest import SampleRef, sha256_text
from .pipeline import (
    DEFAULT_REFINE_INSTRUCTION,
    RunConfig,
    RoundTrace,
    SampleTrace,
    build_refine_prompt,
    transitions_to_experiences,
)
from .retrieval_cache import RetrievalCache, cache_key, state_key
from .scoring import Scorer

_SYS_JUDGE_CACHE: Dict[str, str] = {}


class _State:
    __slots__ = ("ref", "index", "current", "initial", "rounds", "cost", "active",
                 "from_cache", "pending")

    def __init__(self, ref: SampleRef, index: int, current: str, from_cache: bool):
        self.ref = ref
        self.index = index
        self.current = current
        self.initial = current
        self.rounds: List[RoundTrace] = []
        self.cost = {
            "input_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0,
            "latency_s": 0.0, "n_calls": 0.0,
        }
        self.active = True
        self.from_cache = from_cache
        self.pending = None


class BatchedPipeline:

    def _rebuild_retriever(self):
        return ExperienceRetriever(self.library)

    def _hit_is_valid(self, hit) -> bool:
        """A memo is only usable if every id it names still exists.

        Defence in depth for the library-swap bug: even with the fingerprint in
        the cache path, a hand-edited or partially-written memo must never be
        able to crash a run -- an unusable hit is simply recomputed.
        """
        if self.retriever is None:
            return not hit.get("exp_ids")
        known = getattr(self.retriever, "by_id", None)
        if known is None:
            return True
        return all(e in known for e in hit.get("exp_ids") or [])

    def __init__(
        self,
        config: RunConfig,
        retriever: Optional[ExperienceRetriever],
        library: Sequence[Experience],
        client,
        cache: Optional[RetrievalCache] = None,
        batch_size: int = 32,
        online: bool = False,
        accept_max_len_ratio: Optional[float] = None,
    ):
        self.cfg = config
        self.library = list(library)
        self.retriever = retriever
        # Acceptance-gate length guard.  A pipeline argument, NOT a RunConfig
        # field, so enabling it cannot perturb any existing config hash.
        self.accept_max_len_ratio = accept_max_len_ratio
        self.llm = BatchedLLM(client)
        self.cache = cache
        self.batch_size = max(1, batch_size)
        # F3: without this the batched path silently never wrote experience, so
        # the "full_online" arm degenerated into "full_static".
        self.online = bool(online)
        self.task_cfg = TASK_CONFIGS[config.task]
        self.scorer = Scorer(config.task)
        self.judge = PairwiseJudge(config.task, self.task_cfg.display_name)
        self.controller = Controller(config.task, self.task_cfg.display_name)
        self.budget = InferenceBudget(
            max_rounds=config.max_rounds,
            max_tokens=self.task_cfg.max_tokens,
            feedback_max_tokens=512,
            temperature=0.1,
            top_p=1.0,
        )
        # Draft sources.  "stored" (test) replays the baseline Direct-Zero text;
        # "cached" (dev) replays a frozen draft file; "generate" draws new ones.
        # The dev path MUST be usable here too, otherwise dev runs (alpha
        # selection, gates) go through different code than the test runs and the
        # two are not protocol-equivalent.
        self.draft_source = config.draft_source
        if config.draft_source == "stored":
            from .pipeline import StoredDraftSource

            self.drafts = StoredDraftSource(config.task, config.model)
            self.draft_cache = None
        elif config.draft_source == "cached":
            if not config.draft_cache:
                raise ValueError("draft_source='cached' requires draft_cache")
            from .pipeline import CachedDraftSource

            self.drafts = None
            self.draft_cache = CachedDraftSource(Path(config.draft_cache))
        else:
            self.drafts = None
            self.draft_cache = None

    # -- retrieval (identical logic to the unbatched path) -------------------
    def _retrieve(self, ref: SampleRef, current: str, round_index: int):
        if not self.cfg.use_experience or self.retriever is None:
            return [], [], [], [], {}
        in_hash = sha256_text(ref.source)
        st_hash = sha256_text(current)
        profile = None
        if self.cfg.experience_mode == "random":
            sim_res = self.retriever.retrieve(
                ref.source, current, alpha=self.cfg.alpha, k=self.cfg.k,
                exclude_source=ref.source if self.cfg.retrieval_excludes_own_source else None,
            )
            profile = dict(sim_res.per_class_counts)
        key = cache_key(
            model=self.cfg.model, task=self.cfg.task, snapshot_id=self.cfg.snapshot_id,
            round_index=round_index, mode=self.cfg.experience_mode, alpha=self.cfg.alpha,
            k=self.cfg.k, quota_profile=profile,
            rng_seed_key=state_key(ref.source, current) if self.cfg.use_random_ranking else None,
            input_hash=in_hash, state_hash=st_hash,
            own_source_excluded=self.cfg.retrieval_excludes_own_source,
        )
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None and self._hit_is_valid(hit):
                return (hit["exp_ids"], hit["scores"], hit["sim_input"],
                        hit["sim_state"], hit["per_class_counts"])
        rng = None
        if self.cfg.use_random_ranking:
            rng = deterministic_random_rank(
                [e.exp_id for e in self.library], seed_key=state_key(ref.source, current)
            )
        res = self.retriever.retrieve(
            ref.source, current, alpha=self.cfg.alpha, k=self.cfg.k,
            quota_profile=profile, rng_rank=rng,
            # The original method's own-item skip; see ExperienceRetriever.retrieve.
            exclude_source=ref.source if self.cfg.retrieval_excludes_own_source else None,
        )
        if self.cache is not None:
            self.cache.put(key, {
                "exp_ids": res.exp_ids, "scores": res.scores,
                "sim_input": res.sim_input, "sim_state": res.sim_state,
                "per_class_counts": {str(k): int(v) for k, v in res.per_class_counts.items()},
            })
        return (res.exp_ids, res.scores, res.sim_input, res.sim_state,
                {str(k): int(v) for k, v in res.per_class_counts.items()})

    def _by_id(self, exp_id: str) -> Experience:
        for e in self.library:
            if e.exp_id == exp_id:
                return e
        raise KeyError(exp_id)

    # -- main ---------------------------------------------------------------
    def run(self, refs: Sequence[SampleRef], sink=None,
            indices: Sequence[int] = None) -> List[SampleTrace]:
        """Advance all samples in lock-step, emitting each chunk as it completes.

        ``sink`` is called with the traces of every finished chunk (and the
        running total) so the caller can flush to disk incrementally.  Without
        it a task's whole progress would only be persisted at the very end,
        which loses monitoring and crash resilience.

        ``indices`` restricts the work to those *global* positions of ``refs``
        (used by the resume path).  The global index of every processed item is
        preserved exactly, so a resumed run reproduces the same seeds, the same
        stored-draft lookups and the same ``TaskExample.index`` values it would
        have had in a from-scratch run.  ``indices=None`` means "all of them"
        and is byte-identical to a full run.
        """
        from baseline_core.tasks import get_adapter

        adapter = get_adapter(self.cfg.task)
        out: List[SampleTrace] = []
        n = len(refs)
        if indices is None:
            work = list(range(n))
        else:
            work = list(indices)
            for j in work:
                if not (0 <= j < n):
                    raise IndexError(f"resume index {j} outside refs[0:{n}]")
            if len(set(work)) != len(work):
                raise ValueError("resume indices contain duplicates")
        # Chunk on the ORIGINAL index grid (not on the filtered list) so that the
        # batch boundaries -- and therefore every lock-step round -- are the same
        # as in a full run.
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            sel = [j for j in work if start <= j < stop]
            if not sel:
                continue
            chunk = [refs[j] for j in sel]
            states = self._run_chunk(adapter, chunk, start, indices=sel)
            out.extend(states)
            if sink is not None:
                sink(states, len(out), len(work))
            if self.online:
                # Commit this chunk's transitions only AFTER every sample in it
                # has finished, so an item's own experience can never inform its
                # own decisions.  Rebuild the retriever so later chunks see them.
                added = 0
                for tr, ref in zip(states, chunk):
                    new_exp = transitions_to_experiences(
                        tr, self.cfg.model, source_input=ref.source, provenance="online",
                        # Evolution stores wrong->right pairs only, never a
                        # rejected or no-op revision.
                        only_improving=True,
                    )
                    if new_exp:
                        self.library.extend(new_exp)
                        added += len(new_exp)
                if added:
                    self.retriever = self._rebuild_retriever()
            print(f"    [batched {self.cfg.task}/{self.cfg.model}] "
                  f"{stop}/{n}" + ("" if indices is None else " (resume)"), flush=True)
        return out

    def _run_chunk(self, adapter, chunk: Sequence[SampleRef], offset: int,
                   indices: Sequence[int] = None) -> List[SampleTrace]:
        """Run one lock-step chunk.

        ``offset + i`` is the *global* index of item ``i``: it drives the stored
        draft lookup, the ``TaskExample.index`` and every seeded call.  When a
        run is resumed, ``indices`` supplies those global indices explicitly so
        that a subset of the manifest keeps the exact index it would have had in
        a full run.  Passing ``indices=None`` (or ``range(offset, offset+n)``)
        is therefore byte-identical to the pre-resume behaviour.
        """
        if indices is None:
            indices = list(range(offset, offset + len(chunk)))
        if len(indices) != len(chunk):
            raise ValueError(f"indices/chunk length mismatch: {len(indices)} != {len(chunk)}")
        states: List[_State] = []
        # Resolve every draft for this chunk in ONE batched call when the text is
        # not already available (cached-but-missing, or generate-from-scratch).
        need: List[int] = []
        drafts: Dict[int, str] = {}
        draft_cached: Dict[int, bool] = {}
        initial_generations = {}
        for i, ref in enumerate(chunk):
            idx = indices[i]
            if self.drafts is not None:
                text, cached = self.drafts.get(idx)
                drafts[i] = text
                draft_cached[i] = cached
                continue
            hit = self.draft_cache.get(ref.sample_id) if self.draft_cache is not None else None
            if hit is not None:
                drafts[i] = hit
                draft_cached[i] = True
            else:
                need.append(i)
        if need:
            reqs = []
            for i in need:
                ref = chunk[i]
                ex = TaskExample(index=indices[i], source=ref.source,
                                 reference=ref.reference, task=self.cfg.task)
                reqs.append(
                    GenRequest(
                        prompt=adapter.initial_prompt(ex),
                        system_prompt=adapter.system_prompt(),
                        seed=call_seed(ref.sample_id, 0, "initial"),
                        call_type="initial", max_tokens=self.budget.max_tokens,
                        temperature=self.budget.temperature, top_p=self.budget.top_p,
                    )
                )
            gens = self.llm.generate_batch(reqs)
            for i, g in zip(need, gens):
                initial_generations[i] = g
                drafts[i] = adapter.parse_output(g.text)
                if self.draft_cache is not None:
                    self.draft_cache.put(chunk[i].sample_id, drafts[i])
        for i, ref in enumerate(chunk):
            state = _State(ref, indices[i], drafts[i],
                           draft_cached.get(i, False) if self.cfg.optimization else self.drafts is not None)
            # Legacy runs retain their historical accounting; optimized runs
            # explicitly count draft generation as well as refinement.
            if self.cfg.optimization and i in initial_generations:
                g = initial_generations[i]
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    state.cost[key] += getattr(g, key)
                state.cost["latency_s"] += g.latency
                state.cost["n_calls"] += 1
                state.cost["initial_n_calls"] = 1
                state.cost["initial_total_tokens"] = g.total_tokens
            states.append(state)

        for t in range(self.cfg.max_rounds):
            active = [s for s in states if s.active]
            if not active:
                break
            self._round(adapter, active, t)

        traces: List[SampleTrace] = []
        for s in states:
            traces.append(
                SampleTrace(
                    sample_id=s.ref.sample_id, task=self.cfg.task, model=self.cfg.model,
                    config_hash=self.cfg.config_hash, seed=self.cfg.seed,
                    source_hash=s.ref.source_hash, initial_draft=s.initial,
                    final_output=s.current, rounds=s.rounds,
                    initial_metric_offline=self.scorer.primary(s.ref.reference, s.initial),
                    final_metric_offline=self.scorer.primary(s.ref.reference, s.current),
                    draft_from_cache=s.from_cache, cost=s.cost, config=asdict(self.cfg),
                )
            )
        return traces

    # -- one lock-step round over the active samples ------------------------
    def _round(self, adapter, active: List[_State], t: int) -> None:
        # ---- phase 1: retrieval (CPU, no engine call) ---------------------
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

        # ---- phase 2: controller (or forced REFINE) -----------------------
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
                            # v2 moves the examples to the refiner prompt, so the
                            # controller must not be handed a block it is no
                            # longer meant to read (nor told to consult it).
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
                for k in ("input_tokens", "output_tokens", "total_tokens"):
                    s.cost[k] += getattr(g, k)
                # NOTE: Generation exposes `latency`, not `latency_s`
                s.cost["latency_s"] += g.latency
                s.cost["n_calls"] += 1
                parsed.append((s, rq, g, _extract_json(g.text)))

            # Mirror the unbatched controller exactly: a malformed reply gets
            # EXACTLY ONE repair retry, issued here as a second batched call.
            # Without this the batched path fell straight through to STOP with
            # structured_failure=True, which biased every batched arm toward
            # stopping -- the very quantity the "When" claim measures.
            broken = [t for t in parsed if not t[3] or "action" not in t[3]]
            if broken:
                rreqs = []
                for s, rq, g, _obj in broken:
                    rreqs.append(
                        GenRequest(
                            prompt=rq.prompt + "\n\nYour previous reply was not valid JSON. "
                                                "Reply with ONLY the JSON object.",
                            system_prompt=rq.system_prompt,
                            seed=rq.seed,
                            call_type="controller_repair",
                            max_tokens=256,
                            temperature=0.0,
                        )
                    )
                rgens = self.llm.generate_batch(rreqs)
                fixed = {}
                for (s, rq, g, _obj), g2 in zip(broken, rgens):
                    for k in ("input_tokens", "output_tokens", "total_tokens"):
                        s.cost[k] += getattr(g2, k)
                    s.cost["latency_s"] += g2.latency
                    s.cost["n_calls"] += 1
                    fixed[id(s)] = _extract_json(g2.text)

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

        # ---- phase 3: refine batch ---------------------------------------
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

        rreqs = []
        for s, instruction, reason, sfail in refiners:
            instr = instruction or DEFAULT_REFINE_INSTRUCTION
            example = TaskExample(index=s.index, source=s.ref.source,
                                  reference=s.ref.reference, task=self.cfg.task)
            rreqs.append(
                GenRequest(
                    prompt=build_refine_prompt(
                        adapter, example, s.current, instr,
                        # v2: the aligned contrastive examples condition the
                        # revision directly, as in the original method.
                        experience_block=blocks.get(id(s), ""),
                        renderer=self.cfg.renderer,
                    ),
                    system_prompt=adapter.system_prompt(),
                    seed=call_seed(s.ref.sample_id, t, "refine"),
                    call_type="refine", max_tokens=self.budget.max_tokens,
                    temperature=self.budget.temperature, top_p=self.budget.top_p,
                )
            )
        rgens = self.llm.generate_batch(rreqs)
        cands = []
        for (s, instruction, reason, sfail), g in zip(refiners, rgens):
            s.cost["input_tokens"] += g.input_tokens
            s.cost["output_tokens"] += g.output_tokens
            s.cost["total_tokens"] += g.total_tokens
            s.cost["latency_s"] += g.latency
            s.cost["n_calls"] += 1
            cands.append(adapter.parse_output(g.text))

        # ---- phase 4: judge batch (both orders, per-request seeds) --------
        jreqs = []
        meta = []
        sys_j = (
            f"You are a careful evaluator of {self.task_cfg.display_name} outputs. "
            "You compare two candidate answers to the same task and decide which is better. "
            "You never see a reference answer; judge only from the task input and the answers. "
            "Reply with JSON only."
        )
        from .judge import _TEMPLATE

        for (s, _i, _r, _f), cand in zip(refiners, cands):
            if s.current == cand:
                # Byte-identical pair.  The two A/B arrangements are then the
                # SAME prompt, so a model that consistently answers "A" gets
                # mapped through two contradictory slot assignments and the
                # round was recorded "uncertain" rather than "tie" (measured on
                # the Phase 4 traces: 84 such rounds, 81 with byte-identical
                # reasons).  The unbatched judge short-circuits this in
                # PairwiseJudge.judge; the batched path builds its requests
                # inline and so bypassed that -- meaning the fix was inert on
                # the path that actually produces the data.  Mirror it here and
                # spend no engine call on a pair that cannot have a winner.
                continue
            slot_a, slot_b, cand_is_a = randomised_ab(
                self.cfg.model, self.cfg.task, s.ref.sample_id, t, s.current, cand
            )
            for call_idx, (sa, sb, cia) in enumerate(
                ((slot_a, slot_b, cand_is_a), (slot_b, slot_a, not cand_is_a))
            ):
                jreqs.append(
                    GenRequest(
                        prompt=_TEMPLATE.format(task_input=s.ref.source, slot_a=sa, slot_b=sb),
                        system_prompt=sys_j,
                        seed=call_seed(s.ref.sample_id, t * 10 + call_idx, "judge"),
                        call_type="judge", max_tokens=JUDGE_MAX_TOKENS, temperature=0.0,
                    )
                )
                meta.append((s, cand, cia, call_idx))
        jgens = self.llm.generate_batch(jreqs)

        # Mirror the unbatched judge exactly: a malformed reply gets EXACTLY ONE
        # repair retry.  Without it an unparseable judge call could never be
        # accepted, biasing every batched arm toward rejecting refinements
        # (observed: llama3.1-8b/en_zh had 331/1562 rounds with an empty judge
        # reason, all recorded "uncertain").
        jobjs = [_extract_json(g.text) for g in jgens]
        broken_j = [i for i, o in enumerate(jobjs) if not o or "verdict" not in o]
        if broken_j:
            jrreqs = [
                GenRequest(
                    prompt=jreqs[i].prompt + "\n\nYour previous reply was not valid JSON. "
                                              "Reply with ONLY the JSON object.",
                    system_prompt=jreqs[i].system_prompt,
                    seed=jreqs[i].seed,
                    call_type="judge_repair",
                    max_tokens=JUDGE_MAX_TOKENS,
                    temperature=0.0,
                )
                for i in broken_j
            ]
            jrgens = self.llm.generate_batch(jrreqs)
            for i, g2 in zip(broken_j, jrgens):
                s_of = meta[i][0]
                for k in ("input_tokens", "output_tokens", "total_tokens"):
                    s_of.cost[k] += getattr(g2, k)
                s_of.cost["latency_s"] += g2.latency
                s_of.cost["n_calls"] += 1
                o2 = _extract_json(g2.text)
                if o2 and "verdict" in o2:
                    jobjs[i] = o2

        # ---- phase 5: combine and advance ---------------------------------
        per_pair: Dict[int, List[tuple]] = {}
        for (s, cand, cia, call_idx), g, obj in zip(meta, jgens, jobjs):
            s.cost["input_tokens"] += g.input_tokens
            s.cost["output_tokens"] += g.output_tokens
            s.cost["total_tokens"] += g.total_tokens
            s.cost["latency_s"] += g.latency
            s.cost["n_calls"] += 1
            obj = obj or {}
            vraw = str(obj.get("verdict", "uncertain"))
            per_pair.setdefault(id(s), []).append(
                (_map_verdict(vraw, cia), str(obj.get("reason", "")))
            )

        for (s, instruction, reason, sfail), cand in zip(refiners, cands):
            if s.current == cand:
                # Sibling of the short-circuit above: synthesise the same
                # verdict the unbatched judge returns, with no engine calls.
                v1 = v2 = "tie"
                r1 = r2 = "identical: candidate equals current verbatim"
                verdict = "tie"
                order_consistent = False
                accepted = False
            else:
                calls = per_pair.get(id(s), [("uncertain", ""), ("uncertain", "")])
                v1, r1 = calls[0]
                v2, r2 = calls[1]
                verdict = _combine(v1, v2)
                order_consistent = v1 == v2 and v1 in ("candidate", "current")
                accepted = accept_revision(verdict, order_consistent, s.current,
                                           cand, self.accept_max_len_ratio)
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
            if accepted:
                s.current = cand
            elif self.cfg.stop_mode == "judge":
                s.active = False
