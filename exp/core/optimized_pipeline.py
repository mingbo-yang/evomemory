"""BERT-controlled refinement with hybrid retrieval and delayed feedback.

BERT receives source/output only. Reference scores are evaluated for reporting
and (after a whole batch finishes) for feedback-based admission to memory.
"""
from __future__ import annotations

import json
import time
from .batched_pipeline import BatchedPipeline
from .batch_llm import GenRequest
from .pipeline import RoundTrace, build_refine_prompt
from .determinism import call_seed
from .experience import render_experience_block
from .quality_feedback import QualityPolicy
from baseline_core.types import TaskExample

INSTRUCTIONS = {
    "wmt19_en_zh": "Correct mistranslations, omissions, names and numbers. Preserve meaning and avoid unnecessary expansion.",
    "wmt19_zh_en": "Correct mistranslations, omissions, names and numbers. Preserve meaning and avoid unnecessary expansion.",
    "coedit_gec": "Correct grammatical errors with minimal edits. Preserve the author's meaning and avoid paraphrasing correct text.",
    "gigaword": "Improve factual accuracy and coverage of the main point. Keep the summary concise and do not add unsupported details.",
}


class OptimizedBatchedPipeline(BatchedPipeline):
    def __init__(self, *args, quality_evaluator, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_evaluator = quality_evaluator
        self.policy = QualityPolicy(**self.cfg.optimization["policy"])
        # Library changes invalidate ranking even if all cached IDs still exist.
        # No persisted retrieval memo is reused in this pipeline.
        self.cache = None

    def _rebuild_retriever(self):
        if hasattr(self.retriever, "rebuild"):
            return self.retriever.rebuild(self.library)
        if self.retriever is None and self.cfg.optimization.get("retrieval", {}).get("method") == "hybrid":
            from .hybrid_retrieval import LocalSentenceEncoder, HybridExperienceRetriever
            settings = dict(self.cfg.optimization["retrieval"])
            settings.pop("method")
            encoder = LocalSentenceEncoder(self.cfg.optimization["encoder"], self.cfg.optimization["device"])
            return HybridExperienceRetriever(self.library, encoder, **settings)
        return super()._rebuild_retriever()

    def _score(self, states, answers):
        start = time.perf_counter()
        scores = self.quality_evaluator.score_pairs([(s.ref.source, a) for s, a in zip(states, answers)])
        if len(scores) != len(states):
            raise ValueError("BERT score count does not match batch")
        elapsed = (time.perf_counter()-start) / max(1, len(states))
        for s in states:
            s.cost["quality_pairs"] = s.cost.get("quality_pairs", 0) + 1
            s.cost["quality_latency_s"] = s.cost.get("quality_latency_s", 0) + elapsed
            s.cost["latency_s"] += elapsed
        return scores

    def _trace(self, s, t, retrieval, action, reason, candidate="", accepted=False,
               verdict="", metric=None, delta=None):
        ids, scores, si, ss, counts = retrieval
        s.rounds.append(RoundTrace(
            round_index=t, exp_ids=ids, exp_scores=scores, exp_sim_input=si,
            exp_sim_state=ss, exp_class_counts=counts, controller_action=action,
            controller_instruction=INSTRUCTIONS[self.cfg.task] if action == "REFINE" else "",
            controller_reason=reason, controller_structured_failure=False,
            candidate=candidate, judge_verdict=verdict,
            # Single deterministic evaluator; no double-order judge is invoked.
            judge_order_consistent=False, judge_reason_a=reason, judge_reason_b="",
            accepted=accepted, metric_offline=metric, delta_offline=delta,
            cost=dict(s.cost),
        ))

    def _round(self, adapter, active, t):
        before = self._score(active, [s.current for s in active])
        pending, requests, retrievals, previous = [], [], [], []
        for s, score in zip(active, before):
            if self.cfg.stop_mode != "fixed" and self.policy.should_stop(score):
                self._trace(s, t, ([], [], [], [], {}), "STOP",
                            json.dumps({"evaluator": "bert", "score": score,
                                        "reason": "quality_threshold"}))
                s.active = False
                continue
            start = time.perf_counter()
            retrieval = self._retrieve(s.ref, s.current, t)
            elapsed = time.perf_counter()-start
            s.cost["retrieval_latency_s"] = s.cost.get("retrieval_latency_s", 0) + elapsed
            s.cost["latency_s"] += elapsed
            block = render_experience_block(
                [self._by_id(e) for e in retrieval[0]],
                include_outcome=self.cfg.include_outcome, count_tokens=self.llm.count_tokens,
                max_units=self.cfg.k, contrastive=self.cfg.render_contrastive,
                advice_mode=self.cfg.advice_mode,
            )
            example = TaskExample(index=s.index, source=s.ref.source,
                                  reference=s.ref.reference, task=self.cfg.task)
            requests.append(GenRequest(
                prompt=build_refine_prompt(adapter, example, s.current, INSTRUCTIONS[self.cfg.task],
                                           experience_block=block, renderer=self.cfg.renderer),
                system_prompt=adapter.system_prompt(), seed=call_seed(s.ref.sample_id, t, "refine"),
                call_type="refine", max_tokens=self.budget.max_tokens,
                temperature=self.budget.temperature, top_p=self.budget.top_p,
            ))
            pending.append(s)
            previous.append(score)
            retrievals.append(retrieval)
        if not pending:
            return
        generations = self.llm.generate_batch(requests)
        if len(generations) != len(pending):
            raise ValueError("generation count does not match refinement batch")
        candidates = [adapter.parse_output(g.text) for g in generations]
        after = self._score(pending, candidates)
        for s, g, candidate, old_score, new_score, retrieval in zip(
                pending, generations, candidates, previous, after, retrievals):
            for k in ("input_tokens", "output_tokens", "total_tokens"):
                s.cost[k] += getattr(g, k)
            s.cost["latency_s"] += g.latency
            s.cost["n_calls"] += 1
            accepted, why = self.policy.accept(s.current, candidate, old_score, new_score)
            reason = json.dumps({"evaluator": "bert", "before": old_score, "after": new_score,
                                 "gain": new_score-old_score, "gate": why})
            # Delayed task feedback is never consulted in the BERT decision above.
            old_metric = self.scorer.primary(s.ref.reference, s.current)
            metric = self.scorer.primary(s.ref.reference, candidate)
            verdict = "better" if new_score > old_score else "worse" if new_score < old_score else "tie"
            self._trace(s, t, retrieval, "REFINE", reason, candidate, accepted,
                        verdict, metric, metric-old_metric)
            if accepted:
                s.current = candidate
            elif self.cfg.stop_mode == "judge":
                s.active = False
