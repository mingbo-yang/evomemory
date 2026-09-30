"""The experience-driven refinement loop.

One sample, one config (plan v5 sections 3-7):

1. ``y0`` -- the initial draft.  For seed 42 this is the *stored* Direct-Zero
   output, reused verbatim, so every arm and every baseline shares a
   byte-identical starting point and no extra call is spent.
2. retrieve up to four experiences at the current state (memoised),
3. render them (the only difference between ``full`` and ``outcome_hidden``),
4. controller decides REFINE vs STOP (or the judge does, in ``judge`` stop mode),
5. the refiner produces a candidate,
6. a double-order, reference-free judge decides whether to accept it,
7. the transition is recorded as an experience candidate for later tasks.

Nothing is written to the experience library until the sample has finished, so
an item's own experience can never influence its own later decisions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from baseline_core.config import TASK_CONFIGS
from baseline_core.types import InferenceBudget, TaskExample

from . import MODELS, TEST_SAMPLES, baseline_results_dir
from .bm25_fields import Experience, ExperienceRetriever, deterministic_random_rank
from .controller import Controller
from .determinism import call_seed
from .experience import render_experience_block
from .judge import PairwiseJudge, accept_revision
from .manifest import SampleRef, sha256_text
from .retrieval_cache import RetrievalCache, cache_key, cache_path, state_key
from .scoring import PRIMARY_METRIC, Scorer

FEEDBACK_MAX_TOKENS = 512
MAX_ROUNDS_DEFAULT = 3

#: Used when the controller returns REFINE but writes no instruction.  It is a
#: module constant (not an inline literal) so that the counterfactual runs in
#: ``run_stopping_diagnostics.py`` refine from a STOP state with exactly the
#: instruction the pipeline would have used.
DEFAULT_REFINE_INSTRUCTION = (
    "Fix any remaining omissions, mistranslations, awkward wording, "
    "or unsupported content. Change as little as possible."
)

EXPERIENCE_MODES = ("full", "random", "none", "outcome_hidden", "positive_only")
STOP_MODES = ("adaptive", "judge", "fixed")
#: Experience-block renderer generations.  ``v1`` is kept schedulable only so
#: that pre-existing runs remain resumable; all new arms use ``v2``.
RENDERERS = ("v1", "v2")

#: A config field equal to its entry here is omitted from ``config_hash``.
#: This is what lets a new knob be added without invalidating the identity of
#: every run already on disk.
LEGACY_HASH_DEFAULTS: Dict[str, object] = {
    "renderer": "v1",
    "controller_sees_examples": False,
    "advice_mode": "summary",
    "retrieval_excludes_own_source": False,
    "optimization": {},
}


# --------------------------------------------------------------------------- #


@dataclass
class RunConfig:
    task: str
    model: str
    experience_mode: str = "full"
    stop_mode: str = "adaptive"
    alpha: float = 0.5
    k: int = 4
    max_rounds: int = MAX_ROUNDS_DEFAULT
    seed: int = 42
    snapshot_id: str = "initial"
    draft_source: str = "stored"      # stored | generate | cached
    draft_cache: str = ""             # JSONL cache used when draft_source="cached"
    ban_own_experience: bool = True
    #: Renderer generation for the experience block.  ``v1`` is the historical
    #: ad-hoc schema (``[Experience i] / Task input / ... / Outcome``); ``v2`` is
    #: the original method's four-line contrastive example block.  Named so that
    #: a v1 run can still be resumed byte-identically after v2 lands.
    renderer: str = "v1"
    #: v2 only: also show the retrieved examples to the controller.  Default
    #: False = the examples condition the refiner alone.  Giving them to both is
    #: the ``both`` condition, added because removing them from the controller
    #: entirely measurably increased over-refinement on dev.
    controller_sees_examples: bool = False
    #: v2 only: how the retrieved unit's advice is rendered.  ``summary`` is the
    #: original method's last paragraph (a generalised takeaway); ``full`` keeps
    #: the whole critique.  The knob exists because the v1 renderer fed the
    #: controller the entire ~2,400-char critique, and a 360-char takeaway may
    #: not carry enough to guide a concrete fix through a two-stage pipeline.
    advice_mode: str = "summary"
    #: Whether retrieval drops the query item's OWN unit.  ``ban_own_experience``
    #: above was declared True but never implemented, so every run recorded
    #: before this field existed handed the refiner the gold answer to the very
    #: sentence it was translating (98.3% of rows).  A separate field is needed
    #: rather than just fixing ``ban_own_experience`` in place: that field is
    #: already True in every recorded config, so repairing it would change
    #: behaviour without changing the hash, and a resumed run would splice
    #: contaminated and clean rows into one file.
    retrieval_excludes_own_source: bool = False

    # Opt-in optimized experiment identity; legacy hashes omit the empty mapping.
    optimization: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.experience_mode not in EXPERIENCE_MODES:
            raise ValueError(f"experience_mode must be one of {EXPERIENCE_MODES}")
        if self.stop_mode not in STOP_MODES:
            raise ValueError(f"stop_mode must be one of {STOP_MODES}")
        if self.renderer not in RENDERERS:
            raise ValueError(f"renderer must be one of {RENDERERS}")

    @property
    def config_hash(self) -> str:
        # Fields sitting at their legacy default are omitted so that adding the
        # ``renderer`` knob does not move the identity of the ~50 production
        # runs already on disk (tests/test_config_hash_legacy.py pins them).
        # A knob contributes to the hash only once it is set away from default,
        # which is what makes a v2 run unmistakably distinct from a v1 run.
        payload = json.dumps(
            {
                k: v
                for k, v in asdict(self).items()
                if not (k in LEGACY_HASH_DEFAULTS and v == LEGACY_HASH_DEFAULTS[k])
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def render_contrastive(self) -> bool:
        """``PositiveOnly`` drops the wrong draft and keeps the gold answer.

        The contrast lives *inside* every unit of the contrastive library, so
        this single flag is the whole difference between the main method and its
        contrastive-modelling ablation; the retrieved ids are identical.
        """
        return self.experience_mode != "positive_only"

    @property
    def include_outcome(self) -> bool:
        return self.experience_mode != "outcome_hidden"

    @property
    def use_experience(self) -> bool:
        return self.experience_mode != "none"

    @property
    def use_random_ranking(self) -> bool:
        return self.experience_mode == "random"


@dataclass
class RoundTrace:
    round_index: int
    exp_ids: List[str]
    exp_scores: List[float]
    exp_sim_input: List[float]
    exp_sim_state: List[float]
    exp_class_counts: Dict[str, int]
    controller_action: str
    controller_instruction: str
    controller_reason: str
    controller_structured_failure: bool
    candidate: str
    judge_verdict: str
    judge_order_consistent: bool
    judge_reason_a: str
    judge_reason_b: str
    accepted: bool
    metric_offline: Optional[float]
    delta_offline: Optional[float]
    cost: Dict[str, float] = field(default_factory=dict)


@dataclass
class SampleTrace:
    sample_id: str
    task: str
    model: str
    config_hash: str
    seed: int
    source_hash: str
    initial_draft: str
    final_output: str
    rounds: List[RoundTrace]
    initial_metric_offline: Optional[float]
    final_metric_offline: Optional[float]
    draft_from_cache: bool
    cost: Dict[str, float]
    config: Dict[str, object]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "task": self.task,
            "model": self.model,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "source_hash": self.source_hash,
            "initial_draft": self.initial_draft,
            "final_output": self.final_output,
            "initial_metric_offline": self.initial_metric_offline,
            "final_metric_offline": self.final_metric_offline,
            "draft_from_cache": self.draft_from_cache,
            "cost": self.cost,
            "config": self.config,
            "rounds": [asdict(r) for r in self.rounds],
        }


# --------------------------------------------------------------------------- #
# draft reuse
# --------------------------------------------------------------------------- #


class CachedDraftSource:
    """Reads (or lazily creates) a frozen draft per sample.

    Any sweep over a retrieval parameter must hold the initial draft fixed.
    vLLM is not bit-deterministic at temperature 0.1, so regenerating drafts per
    sweep point silently changes the starting condition: measured corpus BLEU on
    the same 16 dev items was 32.51 in one pass and 38.21 in the next, purely
    from redrawn drafts.  Caching them removes that confound -- the same role
    the stored Direct-Zero file plays for the test split.

    The cache is a JSONL of {"sample_id": ..., "draft": ...}.  Concurrent
    writers are avoided by generating the cache in a separate pass first.
    """

    def __init__(self, path: Path):
        self.path = path
        self._mem: Dict[str, str] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._mem[rec["sample_id"]] = rec["draft"]

    def get(self, sample_id: str) -> Optional[str]:
        return self._mem.get(sample_id)

    def put(self, sample_id: str, draft: str) -> None:
        if sample_id in self._mem:
            return
        self._mem[sample_id] = draft
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"sample_id": sample_id, "draft": draft}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._mem)


class StoredDraftSource:
    """Reads the initial draft verbatim from the existing Direct-Zero run."""

    def __init__(self, task: str, model: str):
        self.task = task
        self.model = model
        self.rows: Optional[List[str]] = None

    def _load(self) -> List[str]:
        if self.rows is not None:
            return self.rows
        import csv

        path = baseline_results_dir() / self.task / self.model / "Direct-Zero.csv"
        out: List[str] = []
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                out.append(r.get("final_output", ""))
        self.rows = out
        return out

    def get(self, index: int) -> Tuple[str, bool]:
        rows = self._load()
        if index >= len(rows):
            raise IndexError(
                f"stored Direct-Zero has {len(rows)} rows, need index {index} "
                f"for {self.task}/{self.model}"
            )
        return rows[index], True


# --------------------------------------------------------------------------- #
# refinement prompt
# --------------------------------------------------------------------------- #


#: The four focus dimensions of the original method's ``### Current Task`` block.
#: The two translation tasks reproduce the original wording verbatim; the other
#: two keep the identical shape with task-appropriate questions, because the
#: original list is translation-specific ("mistranslations", "sound more native")
#: and would be meaningless for grammar correction or summarisation.
CURRENT_TASK_DIMENSIONS: Dict[str, str] = {
    "wmt19_en_zh": (
        "1. **Nuance & Completeness**: Is there any missing information or slight deviation in meaning?\n"
        "2. **Terminology Accuracy**: Are there severe mistranslations of specific terms or entities?\n"
        "3. **Tone & Register**: Is the translation too casual? Should it be more academic, formal, or journalistic?\n"
        "4. **Sentence Structure**: How to improve the fluency or logical flow to sound more native?"
    ),
    "wmt19_zh_en": (
        "1. **Nuance & Completeness**: Is there any missing information or slight deviation in meaning?\n"
        "2. **Terminology Accuracy**: Are there severe mistranslations of specific terms or entities?\n"
        "3. **Tone & Register**: Is the translation too casual? Should it be more academic, formal, or journalistic?\n"
        "4. **Sentence Structure**: How to improve the fluency or logical flow to sound more native?"
    ),
    "coedit_gec": (
        "1. **Grammatical Accuracy**: Are there errors in agreement, tense, articles, prepositions or punctuation?\n"
        "2. **Minimality**: Is every change strictly required, with already-correct content left untouched?\n"
        "3. **Meaning Preservation**: Does the correction keep the original meaning fully intact?\n"
        "4. **Fluency**: Does the corrected sentence read naturally?"
    ),
    "gigaword": (
        "1. **Factual Consistency**: Is every claim in the summary supported by the source text?\n"
        "2. **Salience**: Does the summary cover the most important information?\n"
        "3. **Conciseness**: Is there redundancy that can be removed?\n"
        "4. **Fluency**: Does the summary read as one well-formed sentence?"
    ),
}


def build_refine_prompt(
    adapter,
    example: TaskExample,
    current: str,
    instruction: str,
    experience_block: str = "",
    renderer: str = "v1",
) -> str:
    """Build the revision prompt.

    ``renderer="v2"`` reproduces the original method's structure: the retrieved
    contrastive examples sit immediately above a ``### Current Task`` block and
    therefore condition generation directly.  The controller's instruction is
    folded in as the specific defect to fix, with the original four focus
    dimensions retained as the general frame.

    ``renderer="v1"`` is the historical prompt, kept byte-identical so that runs
    already on disk remain resumable.
    """
    if renderer == "v2":
        dimensions = CURRENT_TASK_DIMENSIONS.get(example.task, CURRENT_TASK_DIMENSIONS["wmt19_en_zh"])
        lead = instruction.strip() or "Improve the Current Draft, using the examples above."
        current_task = (
            "### Current Task\n"
            f"Source: {example.source}\n"
            f"Current Draft: {current}\n"
            f"Instruction: {lead}\n"
            "Focus on the following dimensions:\n"
            f"{dimensions}\n\n"
            "Output only the revised final text."
        )
        block = experience_block.strip()
        return f"{block}\n\n{current_task}" if block else current_task

    constraint = {
        "wmt19_en_zh": "Keep the source meaning; fix omissions, mistranslations and unnatural wording; add nothing unsupported.",
        "wmt19_zh_en": "Keep the source meaning; fix omissions, mistranslations and unnatural wording; add nothing unsupported.",
        "coedit_gec": "Make only the corrections required; preserve meaning and do not rewrite correct content.",
        "gigaword": "Improve factual consistency and salience; remove redundancy; introduce no unsupported facts.",
    }[example.task]
    return (
        f"Original task:\n{adapter.initial_prompt(example)}\n\n"
        f"Current answer:\n{current}\n\n"
        f"Revision instruction:\n{instruction}\n\n"
        f"{constraint}\n"
        "Output only the revised final text."
    )


# --------------------------------------------------------------------------- #
# cost accounting
# --------------------------------------------------------------------------- #


def _new_cost() -> Dict[str, float]:
    return {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "total_tokens": 0.0,
        "latency_s": 0.0,
        "n_calls": 0.0,
    }


def _add_cost(cost: Dict[str, float], gen) -> None:
    cost["input_tokens"] += float(gen.input_tokens)
    cost["output_tokens"] += float(gen.output_tokens)
    cost["total_tokens"] += float(gen.total_tokens)
    cost["latency_s"] += float(gen.latency)
    cost["n_calls"] += 1.0


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #


class Pipeline:
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
        llm,
        cache: Optional[RetrievalCache] = None,
        accept_max_len_ratio: Optional[float] = None,
    ):
        self.cfg = config
        self.library = list(library)
        self.retriever = retriever
        # Acceptance-gate length guard; a pipeline argument, not a RunConfig
        # field, so it cannot perturb any existing config hash.
        self.accept_max_len_ratio = accept_max_len_ratio
        self.llm = llm
        self.cache = cache
        self.task_cfg = TASK_CONFIGS[config.task]
        self.scorer = Scorer(config.task)
        self.judge = PairwiseJudge(config.task, self.task_cfg.display_name)
        self.controller = Controller(config.task, self.task_cfg.display_name)
        if config.draft_source == "stored":
            self.drafts = StoredDraftSource(config.task, config.model)
            self.draft_cache = None
        elif config.draft_source == "cached":
            if not config.draft_cache:
                raise ValueError("draft_source='cached' requires draft_cache")
            self.drafts = None
            self.draft_cache = CachedDraftSource(Path(config.draft_cache))
        else:
            self.drafts = None
            self.draft_cache = None
        self.budget = InferenceBudget(
            max_rounds=config.max_rounds,
            max_tokens=self.task_cfg.max_tokens,
            feedback_max_tokens=FEEDBACK_MAX_TOKENS,
            temperature=0.1,
            top_p=1.0,
        )

    # -- helpers ------------------------------------------------------------
    def _retrieve(self, ref: SampleRef, current: str, round_index: int):
        if not self.cfg.use_experience or self.retriever is None:
            return [], [], [], [], {}
        in_hash = sha256_text(ref.source)
        st_hash = sha256_text(current)
        profile = None
        if self.cfg.experience_mode == "random":
            # pin the composition to what Full would have produced: run the
            # similarity ranking first, then reuse its per-class counts
            sim_res = self.retriever.retrieve(
                ref.source, current, alpha=self.cfg.alpha, k=self.cfg.k,
                exclude_source=ref.source if self.cfg.retrieval_excludes_own_source else None,
            )
            profile = dict(sim_res.per_class_counts)
        mode = self.cfg.experience_mode
        key = cache_key(
            model=self.cfg.model,
            task=self.cfg.task,
            snapshot_id=self.cfg.snapshot_id,
            round_index=round_index,
            mode=mode,
            alpha=self.cfg.alpha,
            k=self.cfg.k,
            quota_profile=profile,
            rng_seed_key=state_key(ref.source, current) if self.cfg.use_random_ranking else None,
            input_hash=in_hash,
            state_hash=st_hash,
            own_source_excluded=self.cfg.retrieval_excludes_own_source,
        )
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None and self._hit_is_valid(hit):
                return (
                    hit["exp_ids"],
                    hit["scores"],
                    hit["sim_input"],
                    hit["sim_state"],
                    hit["per_class_counts"],
                )
        rng = None
        if self.cfg.use_random_ranking:
            rng = deterministic_random_rank(
                [e.exp_id for e in self.library], seed_key=state_key(ref.source, current)
            )
        res = self.retriever.retrieve(
            ref.source,
            current,
            alpha=self.cfg.alpha,
            k=self.cfg.k,
            quota_profile=profile,
            rng_rank=rng,
            # The original method's own-item skip; see ExperienceRetriever.retrieve.
            exclude_source=ref.source if self.cfg.retrieval_excludes_own_source else None,
        )
        if self.cache is not None:
            self.cache.put(
                key,
                {
                    "exp_ids": res.exp_ids,
                    "scores": res.scores,
                    "sim_input": res.sim_input,
                    "sim_state": res.sim_state,
                    "per_class_counts": {str(k): int(v) for k, v in res.per_class_counts.items()},
                },
            )
        return (
            res.exp_ids,
            res.scores,
            res.sim_input,
            res.sim_state,
            {str(k): int(v) for k, v in res.per_class_counts.items()},
        )

    # -- one sample ---------------------------------------------------------
    def run_sample(self, ref: SampleRef, index: int) -> SampleTrace:
        from baseline_core.tasks import get_adapter

        adapter = get_adapter(self.cfg.task)
        example = TaskExample(
            index=index, source=ref.source, reference=ref.reference, task=self.cfg.task
        )

        cost = _new_cost()
        # ---- step 1: initial draft ----------------------------------------
        if self.drafts is not None:
            current, from_cache = self.drafts.get(index)
        elif self.draft_cache is not None and self.draft_cache.get(ref.sample_id) is not None:
            current = self.draft_cache.get(ref.sample_id)  # type: ignore[assignment]
            from_cache = True
        else:
            gen = self.llm.generate(
                prompt=adapter.initial_prompt(example),
                system_prompt=adapter.system_prompt(),
                seed=call_seed(ref.sample_id, 0, "initial"),
                call_type="initial",
                max_tokens=self.budget.max_tokens,
                temperature=self.budget.temperature,
                top_p=self.budget.top_p,
            )
            current = adapter.parse_output(gen.text)
            _add_cost(cost, gen)
            from_cache = False
            if self.draft_cache is not None:
                self.draft_cache.put(ref.sample_id, current)

        initial_draft = current
        initial_metric = self.scorer.primary(ref.reference, current)
        rounds: List[RoundTrace] = []

        # ---- steps 2-6: refinement rounds ---------------------------------
        for t in range(self.cfg.max_rounds):
            exp_ids, exp_scores, sim_i, sim_s, class_counts = self._retrieve(ref, current, t)
            exps = [self._by_id(e) for e in exp_ids]
            block = render_experience_block(
                exps,
                include_outcome=self.cfg.include_outcome,
                count_tokens=self.llm.count_tokens,
                max_units=self.cfg.k,
                contrastive=self.cfg.render_contrastive,
                advice_mode=self.cfg.advice_mode,
            )

            # --- controller (or judge-signalled stop) ---
            if self.cfg.stop_mode == "judge":
                # no controller at all: always attempt one more revision; the
                # judge's acceptance is the only stopping signal.
                action, instruction, reason, sfail = "REFINE", "", "", False
            else:
                dec = self.controller.decide(
                    self.llm,
                    ref.sample_id,
                    t,
                    ref.source,
                    current,
                    # v2 moves the examples to the refiner prompt, so the
                    # controller is not handed evidence it must no longer read.
                    block
                    if (self.cfg.renderer != "v2" or self.cfg.controller_sees_examples)
                    else "",
                    stop_mode=self.cfg.stop_mode,
                    renderer=self.cfg.renderer,
                )
                for c in dec.calls:
                    cost["input_tokens"] += c.input_tokens
                    cost["output_tokens"] += c.output_tokens
                    cost["total_tokens"] += c.input_tokens + c.output_tokens
                    cost["latency_s"] += c.latency_s
                    cost["n_calls"] += 1
                action = dec.action
                instruction = dec.instruction
                reason = dec.reason
                sfail = dec.structured_failure
                if action == "STOP":
                    rounds.append(
                        RoundTrace(
                            round_index=t,
                            exp_ids=exp_ids,
                            exp_scores=exp_scores,
                            exp_sim_input=sim_i,
                            exp_sim_state=sim_s,
                            exp_class_counts=class_counts,
                            controller_action="STOP",
                            controller_instruction="",
                            controller_reason=reason,
                            controller_structured_failure=sfail,
                            candidate="",
                            judge_verdict="",
                            judge_order_consistent=False,
                            judge_reason_a="",
                            judge_reason_b="",
                            accepted=False,
                            metric_offline=None,
                            delta_offline=None,
                            cost=dict(cost),
                        )
                    )
                    break

            # --- refine ---
            if not instruction:
                instruction = DEFAULT_REFINE_INSTRUCTION
            prompt = build_refine_prompt(
                adapter, example, current, instruction,
                experience_block=block,
                renderer=self.cfg.renderer,
            )
            rgen = self.llm.generate(
                prompt=prompt,
                system_prompt=adapter.system_prompt(),
                seed=call_seed(ref.sample_id, t, "refine"),
                call_type="refine",
                max_tokens=self.budget.max_tokens,
                temperature=self.budget.temperature,
                top_p=self.budget.top_p,
            )
            _add_cost(cost, rgen)
            candidate = adapter.parse_output(rgen.text)

            # --- judge ---
            verdict = self.judge.judge(
                self.llm,
                self.cfg.model,
                ref.sample_id,
                t,
                ref.source,
                current,
                candidate,
            )
            for jc in verdict.calls:
                cost["input_tokens"] += jc.input_tokens
                cost["output_tokens"] += jc.output_tokens
                cost["total_tokens"] += jc.input_tokens + jc.output_tokens
                cost["latency_s"] += jc.latency_s
                cost["n_calls"] += 1

            cand_metric = self.scorer.primary(ref.reference, candidate)
            prev_metric = self.scorer.primary(ref.reference, current)
            accepted = accept_revision(verdict.verdict, verdict.order_consistent,
                                       current, candidate, self.accept_max_len_ratio)
            rounds.append(
                RoundTrace(
                    round_index=t,
                    exp_ids=exp_ids,
                    exp_scores=exp_scores,
                    exp_sim_input=sim_i,
                    exp_sim_state=sim_s,
                    exp_class_counts=class_counts,
                    controller_action="REFINE",
                    controller_instruction=instruction,
                    controller_reason=reason,
                    controller_structured_failure=sfail,
                    candidate=candidate,
                    judge_verdict=verdict.verdict,
                    judge_order_consistent=verdict.order_consistent,
                    judge_reason_a=verdict.reason_a,
                    judge_reason_b=verdict.reason_b,
                    accepted=accepted,
                    metric_offline=cand_metric,
                    delta_offline=cand_metric - prev_metric,
                    cost=dict(cost),
                )
            )
            if accepted:
                current = candidate
            elif self.cfg.stop_mode == "judge":
                # judge-signalled stopping: a rejected candidate ends the loop
                break

        final_metric = self.scorer.primary(ref.reference, current)
        return SampleTrace(
            sample_id=ref.sample_id,
            task=self.cfg.task,
            model=self.cfg.model,
            config_hash=self.cfg.config_hash,
            seed=self.cfg.seed,
            source_hash=ref.source_hash,
            initial_draft=initial_draft,
            final_output=current,
            rounds=rounds,
            initial_metric_offline=initial_metric,
            final_metric_offline=final_metric,
            draft_from_cache=from_cache,
            cost=cost,
            config=asdict(self.cfg),
        )

    def _by_id(self, exp_id: str) -> Experience:
        for e in self.library:
            if e.exp_id == exp_id:
                return e
        raise KeyError(exp_id)


# --------------------------------------------------------------------------- #
# transitions -> experience candidates
# --------------------------------------------------------------------------- #


def transitions_to_experiences(
    trace: SampleTrace,
    model: str,
    source_input: str,
    provenance: str = "self-generated",
    only_improving: bool = False,
) -> List[Experience]:
    """Convert transitions into experience records.

    Written only *after* the sample has completed, so an item's own experience
    can never feed back into its own decisions.  ``source_input`` must be the
    task input the transition belongs to; it is required because the retrieved
    experience is matched on both the input and the state.

    ``only_improving=True`` is the online-evolution contract: the library stores
    **wrong -> right** pairs only.  A round qualifies when it was accepted by the
    gate *and* the offline metric actually rose, so ``state_before`` is genuinely
    the worse answer and ``state_after`` genuinely the better one.  Recording
    every REFINE transition instead -- the historical behaviour -- fills the
    library with revisions that were rejected or that changed nothing, which is
    precisely the pathology that made the v1 library teach the model the
    micro-edits already measured to fail.
    """
    if not source_input:
        raise ValueError("source_input is required to build a usable experience")
    # Replay accepted state independently of admission to the experience store.
    # Rejected candidates must never become the next round's state_before.
    out: List[Experience] = []
    current = trace.initial_draft or trace.final_output
    for r in trace.rounds:
        if r.controller_action != "REFINE" or not r.candidate:
            continue
        improved = r.delta_offline is not None and r.delta_offline > 0
        if only_improving and not (getattr(r, "accepted", False) and improved):
            if r.accepted:
                current = r.candidate
            continue
        out.append(
            Experience(
                exp_id=f"{trace.task}/{model}/{trace.sample_id}/r{r.round_index}",
                task=trace.task,
                model=model,
                source_input=source_input,
                state_before=current,
                state_after=r.candidate,
                intervention_instruction=r.controller_instruction,
                intervention_rationale=r.controller_reason,
                verdict=(
                    "better"
                    if r.judge_verdict == "better"
                    else "worse"
                    if r.judge_verdict == "worse"
                    else "tie"
                    if r.judge_verdict == "tie"
                    else "uncertain"
                ),
                reason_a=r.judge_reason_a,
                reason_b=r.judge_reason_b,
                order_consistent=r.judge_order_consistent,
                delta_offline=r.delta_offline,
                provenance=provenance,
                # A kept unit is by construction a measured improvement, so the
                # label is not an inference -- it is the admission criterion.
                outcome_label="helped" if only_improving else "",
            )
        )
        if r.accepted:
            current = r.candidate
    return out
