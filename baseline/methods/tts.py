from __future__ import annotations

import json
import math
from typing import Callable, Dict, List, Optional, Tuple

from core.llm import LLMClient
from core.metrics import language_aware_tokens
from core.tasks import TaskAdapter, parse_json_object
from core.types import Generation, InferenceBudget, RunResult, TaskExample
from core.utility import UtilityPredictor


def _gen(
    llm: LLMClient,
    adapter: TaskAdapter,
    prompt: str,
    seed: int,
    call_type: str,
    max_tokens: int,
    budget: InferenceBudget,
    temperature: Optional[float] = None,
) -> Generation:
    return llm.generate(
        prompt=prompt,
        system_prompt=adapter.system_prompt(),
        seed=seed,
        call_type=call_type,
        max_tokens=max_tokens,
        temperature=budget.temperature if temperature is None else temperature,
        top_p=budget.top_p,
    )


def _candidate_from_generation(adapter: TaskAdapter, gen: Generation) -> str:
    return adapter.parse_output(gen.text)


def _scores(utility: UtilityPredictor, example: TaskExample, candidates: List[str]) -> List[float]:
    return utility.score_batch([example] * len(candidates), candidates)


def _best_idx(scores: List[float]) -> int:
    return max(range(len(scores)), key=lambda i: float(scores[i])) if scores else 0


def direct_zero(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    gen = _gen(llm, adapter, adapter.initial_prompt(example), seed, "initial", budget.max_tokens, budget)
    y0 = _candidate_from_generation(adapter, gen)
    return RunResult(y0, [y0], [gen], [], 0, {"sequential_call_indices": [0]})


def bon_u_4(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    if utility is None:
        raise RuntimeError("BoN-U-4 requires a utility predictor.")
    generations: List[Generation] = []
    candidates: List[str] = []
    for i in range(budget.n_candidates):
        gen = _gen(llm, adapter, adapter.initial_prompt(example), seed + i, "parallel_candidate", budget.max_tokens, budget)
        generations.append(gen)
        candidates.append(_candidate_from_generation(adapter, gen))
    scores = _scores(utility, example, candidates)
    best = _best_idx(scores)
    max_lat = max((g.latency for g in generations), default=0.0)
    max_tok = max((g.total_tokens for g in generations), default=0)
    return RunResult(
        candidates[best],
        candidates,
        generations,
        scores,
        0,
        {
            "n": budget.n_candidates,
            "best_idx": best,
            "sequential_call_indices": [best],
            "sequential_latency_override": max_lat,
            "sequential_tokens_override": max_tok,
        },
    )


def self_refine_fixed(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    generations: List[Generation] = []
    candidates: List[str] = []
    feedbacks: List[str] = []
    gen = _gen(llm, adapter, adapter.initial_prompt(example), seed, "initial", budget.max_tokens, budget)
    generations.append(gen)
    current = _candidate_from_generation(adapter, gen)
    candidates.append(current)
    for t in range(budget.max_rounds):
        fb = _gen(llm, adapter, adapter.feedback_prompt(example, current), seed + 100 + t, "feedback", budget.feedback_max_tokens, budget)
        generations.append(fb)
        feedbacks.append(fb.text)
        prompt = adapter.feedback_refine_prompt(example, current, fb.text, history=list(zip(candidates, feedbacks)))
        rg = _gen(llm, adapter, prompt, seed + 200 + t, "refine", budget.max_tokens, budget)
        generations.append(rg)
        current = _candidate_from_generation(adapter, rg)
        candidates.append(current)
    return RunResult(current, candidates, generations, [], budget.max_rounds, {"feedbacks": feedbacks, "sequential_call_indices": list(range(len(generations)))})


def self_refine_u(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    if utility is None:
        raise RuntimeError("SelfRefine-U requires a utility predictor.")
    generations: List[Generation] = []
    candidates: List[str] = []
    feedbacks: List[str] = []
    gen = _gen(llm, adapter, adapter.initial_prompt(example), seed, "initial", budget.max_tokens, budget)
    generations.append(gen)
    current = _candidate_from_generation(adapter, gen)
    candidates.append(current)
    scores = [utility.score(example, current)]
    if scores[0] >= utility.threshold:
        return RunResult(current, candidates, generations, scores, 0, {"stopped_by": "threshold", "sequential_call_indices": [0]})
    stop_round = budget.max_rounds
    for t in range(budget.max_rounds):
        fb = _gen(llm, adapter, adapter.feedback_prompt(example, current), seed + 100 + t, "feedback", budget.feedback_max_tokens, budget)
        generations.append(fb)
        feedbacks.append(fb.text)
        rg = _gen(llm, adapter, adapter.feedback_refine_prompt(example, current, fb.text, history=list(zip(candidates, feedbacks))), seed + 200 + t, "refine", budget.max_tokens, budget)
        generations.append(rg)
        current = _candidate_from_generation(adapter, rg)
        candidates.append(current)
        scores.append(utility.score(example, current))
        if scores[-1] >= utility.threshold:
            stop_round = t + 1
            break
    best = _best_idx(scores)
    return RunResult(candidates[best], candidates, generations, scores, stop_round, {"best_idx": best, "feedbacks": feedbacks, "sequential_call_indices": list(range(len(generations)))})


def sr_fixed(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    generations: List[Generation] = []
    candidates: List[str] = []
    gen = _gen(llm, adapter, adapter.initial_prompt(example), seed, "initial", budget.max_tokens, budget)
    generations.append(gen)
    current = _candidate_from_generation(adapter, gen)
    candidates.append(current)
    for t in range(budget.max_rounds):
        rg = _gen(llm, adapter, adapter.direct_refine_prompt(example, current), seed + 300 + t, "direct_refine", budget.max_tokens, budget)
        generations.append(rg)
        current = _candidate_from_generation(adapter, rg)
        candidates.append(current)
    return RunResult(current, candidates, generations, [], budget.max_rounds, {"sequential_call_indices": list(range(len(generations)))})


def sr_u(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    if utility is None:
        raise RuntimeError("SR-U requires a utility predictor.")
    generations: List[Generation] = []
    candidates: List[str] = []
    gen = _gen(llm, adapter, adapter.initial_prompt(example), seed, "initial", budget.max_tokens, budget)
    generations.append(gen)
    current = _candidate_from_generation(adapter, gen)
    candidates.append(current)
    scores = [utility.score(example, current)]
    if scores[0] >= utility.threshold:
        return RunResult(current, candidates, generations, scores, 0, {"stopped_by": "threshold", "sequential_call_indices": [0]})
    stop_round = budget.max_rounds
    for t in range(budget.max_rounds):
        rg = _gen(llm, adapter, adapter.direct_refine_prompt(example, current), seed + 300 + t, "direct_refine", budget.max_tokens, budget)
        generations.append(rg)
        current = _candidate_from_generation(adapter, rg)
        candidates.append(current)
        scores.append(utility.score(example, current))
        if scores[-1] >= utility.threshold:
            stop_round = t + 1
            break
    best = _best_idx(scores)
    return RunResult(candidates[best], candidates, generations, scores, stop_round, {"best_idx": best, "sequential_call_indices": list(range(len(generations)))})


def pdr_2_1(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    generations: List[Generation] = []
    drafts: List[str] = []
    for i in range(2):
        gen = _gen(llm, adapter, adapter.initial_prompt(example), seed + i, "parallel_draft", budget.max_tokens, budget)
        generations.append(gen)
        drafts.append(_candidate_from_generation(adapter, gen))
    wg = _gen(llm, adapter, adapter.pdr_workspace_prompt(example, drafts), seed + 20, "distill", budget.workspace_max_tokens, budget, temperature=0.0)
    generations.append(wg)
    fg = _gen(llm, adapter, adapter.pdr_final_prompt(example, wg.text), seed + 21, "pdr_final", budget.max_tokens, budget)
    generations.append(fg)
    final = _candidate_from_generation(adapter, fg)
    max_draft_lat = max(g.latency for g in generations[:2])
    max_draft_tok = max(g.total_tokens for g in generations[:2])
    return RunResult(
        final,
        drafts + [final],
        generations,
        [],
        1,
        {
            "workspace": wg.text,
            "sequential_latency_override": max_draft_lat + wg.latency + fg.latency,
            "sequential_tokens_override": max_draft_tok + wg.total_tokens + fg.total_tokens,
        },
    )


def checklist_refine(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    generations: List[Generation] = []
    cg = _gen(llm, adapter, adapter.checklist_prompt(example), seed + 10, "checklist_generation", budget.feedback_max_tokens, budget, temperature=0.0)
    generations.append(cg)
    checklist = parse_json_object(cg.text).get("items") or [
        {"id": "c1", "question": "Does the output preserve the input meaning and avoid unsupported content?"},
        {"id": "c2", "question": "Is the output fluent and concise for the task?"},
    ]
    gen = _gen(llm, adapter, adapter.initial_prompt(example), seed, "initial", budget.max_tokens, budget)
    generations.append(gen)
    current = _candidate_from_generation(adapter, gen)
    candidates = [current]
    failed_history: List[List[dict]] = []
    stop_round = budget.max_rounds
    for t in range(budget.max_rounds):
        ev = _gen(llm, adapter, adapter.checklist_eval_prompt(example, current, checklist), seed + 400 + t, "checklist_eval", budget.feedback_max_tokens, budget, temperature=0.0)
        generations.append(ev)
        verdicts = parse_json_object(ev.text).get("verdicts") or []
        failed = [v for v in verdicts if str(v.get("verdict", "")).upper() != "YES"]
        failed_history.append(failed)
        if not failed:
            stop_round = t
            break
        rg = _gen(llm, adapter, adapter.checklist_refine_prompt(example, current, failed), seed + 500 + t, "checklist_refine", budget.max_tokens, budget)
        generations.append(rg)
        current = _candidate_from_generation(adapter, rg)
        candidates.append(current)
    return RunResult(current, candidates, generations, [], stop_round, {"checklist": checklist, "failed_history": failed_history, "sequential_call_indices": list(range(len(generations)))})


def _jaccard_ngram(tokens_a: List[str], tokens_b: List[str], n: int) -> float:
    a = {tuple(tokens_a[i : i + n]) for i in range(max(0, len(tokens_a) - n + 1))}
    b = {tuple(tokens_b[i : i + n]) for i in range(max(0, len(tokens_b) - n + 1))}
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _modex_select(candidates: List[str], tau: float = 0.8) -> int:
    import numpy as np

    if len(candidates) <= 1:
        return 0
    toks = [language_aware_tokens(c) for c in candidates]
    A = np.zeros((len(candidates), len(candidates)), dtype=float)
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i != j:
                A[i, j] = sum(_jaccard_ngram(toks[i], toks[j], n) for n in (1, 2, 3))
    active = np.arange(len(candidates))
    while len(active) > 2:
        sub = A[np.ix_(active, active)]
        D = np.diag(sub.sum(axis=1))
        L = D - sub
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1] if vecs.shape[1] > 1 else vecs[:, 0]
        left = np.where(fiedler >= 0)[0]
        right = np.where(fiedler < 0)[0]
        if len(left) == 0 or len(right) == 0:
            break
        cut = sub[np.ix_(left, right)].sum()
        vol = min(sub[left].sum(), sub[right].sum())
        conductance = cut / max(vol, 1e-12)
        if conductance >= tau:
            break
        if len(left) > len(right):
            keep = left
        elif len(right) > len(left):
            keep = right
        else:
            keep = left if sub[np.ix_(left, left)].sum() >= sub[np.ix_(right, right)].sum() else right
        active = active[keep]
    final = A[np.ix_(active, active)]
    local = int(np.argmax(final.sum(axis=1)))
    return int(active[local])


def modex_4(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    generations: List[Generation] = []
    candidates: List[str] = []
    for i in range(budget.n_candidates):
        gen = _gen(llm, adapter, adapter.initial_prompt(example), seed + i, "parallel_candidate", budget.max_tokens, budget)
        generations.append(gen)
        candidates.append(_candidate_from_generation(adapter, gen))
    best = _modex_select(candidates)
    max_lat = max((g.latency for g in generations), default=0.0)
    max_tok = max((g.total_tokens for g in generations), default=0)
    return RunResult(candidates[best], candidates, generations, [], 0, {"best_idx": best, "sequential_latency_override": max_lat, "sequential_tokens_override": max_tok})


def tear_native(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    if example.task not in {"wmt19_en_zh", "wmt19_zh_en"}:
        return RunResult("", [], [], [], 0, {"skipped": True, "reason": "TEaR is translation-only"})
    generations: List[Generation] = []
    gen = _gen(llm, adapter, adapter.initial_prompt(example), seed, "initial", budget.max_tokens, budget)
    generations.append(gen)
    current = _candidate_from_generation(adapter, gen)
    candidates = [current]
    mqm_history: List[str] = []
    stop_round = budget.max_rounds
    for t in range(budget.max_rounds):
        eg = _gen(llm, adapter, adapter.tear_estimate_prompt(example, current), seed + 600 + t, "estimate", budget.feedback_max_tokens, budget, temperature=0.0)
        generations.append(eg)
        mqm_history.append(eg.text)
        obj = parse_json_object(eg.text)
        if obj.get("no_error") is True or str(obj.get("no_error", "")).lower() == "true":
            stop_round = t
            break
        rg = _gen(llm, adapter, adapter.tear_refine_prompt(example, current, eg.text), seed + 700 + t, "refine", budget.max_tokens, budget)
        generations.append(rg)
        current = _candidate_from_generation(adapter, rg)
        candidates.append(current)
    return RunResult(current, candidates, generations, [], stop_round, {"mqm_history": mqm_history, "sequential_call_indices": list(range(len(generations)))})


def tear_u(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    if utility is None:
        raise RuntimeError("TEaR-U requires a utility predictor.")
    res = tear_native(example, adapter, llm, budget, seed, utility)
    if res.metadata.get("skipped"):
        return res
    scores = _scores(utility, example, res.candidates)
    best = _best_idx(scores)
    res.final_output = res.candidates[best]
    res.utility_scores = scores
    res.metadata["best_idx"] = best
    return res


def adacompute_sr_u(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, utility: Optional[UtilityPredictor]) -> RunResult:
    if utility is None:
        raise RuntimeError("AdaCompute-SR-U requires a utility predictor.")
    gen = _gen(llm, adapter, adapter.initial_prompt(example), seed, "initial", budget.max_tokens, budget)
    current = _candidate_from_generation(adapter, gen)
    u0 = utility.score(example, current)
    src_len = llm.count_tokens(example.source)
    out_len = llm.count_tokens(current)
    ratio = out_len / max(src_len, 1)
    if u0 >= utility.threshold:
        rounds = 0
    elif u0 < utility.threshold * 0.55:
        rounds = 3
    elif ratio < 0.25 or ratio > 2.5:
        rounds = 2
    else:
        rounds = 1
    new_budget = InferenceBudget(
        max_rounds=rounds,
        n_candidates=budget.n_candidates,
        max_tokens=budget.max_tokens,
        feedback_max_tokens=budget.feedback_max_tokens,
        workspace_max_tokens=budget.workspace_max_tokens,
        temperature=budget.temperature,
        top_p=budget.top_p,
    )
    if rounds == 0:
        return RunResult(current, [current], [gen], [u0], 0, {"training_free_adapted": True, "allocated_rounds": 0, "features": {"u0": u0, "src_len": src_len, "out_len": out_len, "ratio": ratio}, "sequential_call_indices": [0]})
    tail = sr_fixed_from_initial(example, adapter, llm, new_budget, seed, gen, current)
    scores = _scores(utility, example, tail.candidates)
    best = _best_idx(scores)
    tail.final_output = tail.candidates[best]
    tail.utility_scores = scores
    tail.metadata.update({"training_free_adapted": True, "allocated_rounds": rounds, "best_idx": best, "features": {"u0": u0, "src_len": src_len, "out_len": out_len, "ratio": ratio}})
    return tail


def sr_fixed_from_initial(example: TaskExample, adapter: TaskAdapter, llm: LLMClient, budget: InferenceBudget, seed: int, initial_gen: Generation, initial_candidate: str) -> RunResult:
    generations = [initial_gen]
    candidates = [initial_candidate]
    current = initial_candidate
    for t in range(budget.max_rounds):
        rg = _gen(llm, adapter, adapter.direct_refine_prompt(example, current), seed + 300 + t, "direct_refine", budget.max_tokens, budget)
        generations.append(rg)
        current = _candidate_from_generation(adapter, rg)
        candidates.append(current)
    return RunResult(current, candidates, generations, [], budget.max_rounds, {"sequential_call_indices": list(range(len(generations)))})


METHOD_REGISTRY: Dict[str, Callable[[TaskExample, TaskAdapter, LLMClient, InferenceBudget, int, Optional[UtilityPredictor]], RunResult]] = {
    "Direct-Zero": direct_zero,
    "BoN-U-4": bon_u_4,
    "SelfRefine-Fixed": self_refine_fixed,
    "SelfRefine-U": self_refine_u,
    "SR-Fixed": sr_fixed,
    "SR-U": sr_u,
    "PDR-2-1": pdr_2_1,
    "ChecklistRefine": checklist_refine,
    "ModeX-4": modex_4,
    "TEaR-Native": tear_native,
    "TEaR-U": tear_u,
    "AdaCompute-SR-U": adacompute_sr_u,
}


METHODS_REQUIRING_UTILITY = {"BoN-U-4", "SelfRefine-U", "SR-U", "TEaR-U", "AdaCompute-SR-U"}
