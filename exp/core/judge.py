"""Reference-free pairwise judge with double-order evaluation.

Protocol (plan v5 section 7):

* The two answers are placed in anonymous slots A/B; the *order* is derived
  deterministically from ``(model, task, sample_id, round)`` so every ablation
  arm sees the identical arrangement.
* The pair is evaluated twice with the slots swapped.
* A candidate is accepted **only if both evaluations clearly prefer it**.  A
  single preference plus a tie is recorded as a tie, and two contradictory
  preferences are recorded as ``uncertain``.

The judge never sees the reference answer, and the same model instance that
produced the task output is reused (no second model family is introduced).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .determinism import ab_order, call_seed, randomised_ab

JUDGE_MAX_TOKENS = 256

_SYSTEM = (
    "You are a careful evaluator of {task_name} outputs. "
    "You compare two candidate answers to the same task and decide which is better. "
    "You never see a reference answer; judge only from the task input and the answers. "
    "Reply with JSON only."
)

_TEMPLATE = """Task input:
{task_input}

Answer A:
{slot_a}

Answer B:
{slot_b}

Which answer better satisfies the task? Consider faithfulness to the input, completeness,
fluency, and absence of unsupported content. Do not reward length by itself.

Return JSON with exactly these keys:
{{"verdict": "A" | "B" | "tie" | "uncertain", "reason": "<one short sentence>"}}
"""


@dataclass
class JudgeCall:
    round_index: int
    candidate_is_a: bool
    raw: str
    verdict_raw: str
    verdict_mapped: str
    reason: str
    reprompted: bool
    input_tokens: int
    output_tokens: int
    latency_s: float
    parse_error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JudgeVerdict:
    verdict: str          # better | worse | tie | uncertain
    order_consistent: bool
    reason_a: str
    reason_b: str
    calls: List[JudgeCall] = field(default_factory=list)
    structured_failure: bool = False

    @property
    def accepted(self) -> bool:
        return self.verdict == "better" and self.order_consistent

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "order_consistent": self.order_consistent,
            "reason_a": self.reason_a,
            "reason_b": self.reason_b,
            "structured_failure": self.structured_failure,
            "calls": [c.to_dict() for c in self.calls],
        }


def _extract_json(text: str) -> Optional[dict]:
    s = str(text).strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def accept_revision(verdict: str, order_consistent: bool, current: str,
                    candidate: str, max_len_ratio: Optional[float] = None) -> bool:
    """THE acceptance rule.  Both pipelines call this and nothing else.

    It lives here, in one place, because the identical-pair fix was once applied
    only to ``PairwiseJudge.judge`` -- the *unbatched* path -- while production
    runs ``BatchedPipeline._round``, so the "fix" was inert where it mattered.
    A rule that is duplicated is a rule that will diverge.

    ``max_len_ratio`` is the acceptance gate's length guard.  Measured on the
    Phase 4 traces: the judge is the same model that wrote the revision, and its
    "better" verdict is largely a length preference -- for glm4-9b/zh_en, 84.7%
    of revisions it called better were worse by the offline metric, at a mean
    length ratio of 1.57, while for qwen3-8b (mean ratio 1.05) only 33.8% were
    wrong.  Every cell's entire loss was attributable to this gate.  When
    ``max_len_ratio`` is None the guard is off and the historical behaviour is
    reproduced exactly; the value is a pipeline argument, never a ``RunConfig``
    field, so enabling it cannot change any existing config hash.
    """
    if not (verdict == "better" and order_consistent):
        return False
    if max_len_ratio is None:
        return True
    return len(candidate) <= max_len_ratio * max(1, len(current))


def _map_verdict(verdict_raw: str, candidate_is_a: bool) -> str:
    v = str(verdict_raw).strip().upper()
    if v in ("A", "B"):
        chose_a = v == "A"
        chose_candidate = chose_a == candidate_is_a
        return "candidate" if chose_candidate else "current"
    if v == "TIE":
        return "tie"
    return "uncertain"


def _combine(v1: str, v2: str) -> str:
    if "uncertain" in (v1, v2):
        return "uncertain"
    if v1 == v2:
        return {"candidate": "better", "current": "worse", "tie": "tie"}[v1]
    if {v1, v2} == {"candidate", "current"}:
        return "uncertain"
    # one preference + one tie -> not strong enough to accept
    return "tie"


class PairwiseJudge:
    def __init__(self, task: str, task_name: str):
        self.task = task
        self.task_name = task_name

    def _one_call(
        self,
        llm,
        sample_id: str,
        round_index: int,
        slot: int,
        candidate_is_a: bool,
        task_input: str,
        slot_a: str,
        slot_b: str,
    ) -> JudgeCall:
        prompt = _TEMPLATE.format(
            task_input=task_input, slot_a=slot_a, slot_b=slot_b
        )
        seed = call_seed(sample_id, round_index * 10 + slot, "judge")
        gen = llm.generate(
            prompt=prompt,
            system_prompt=_SYSTEM.format(task_name=self.task_name),
            seed=seed,
            call_type="judge",
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=0.0,
        )
        obj = _extract_json(gen.text)
        reprompted = False
        parse_error = None
        if obj is None or "verdict" not in obj:
            parse_error = "structured_output_failed"
            # exactly one repair retry, per the plan
            repair = (
                prompt
                + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
            )
            gen2 = llm.generate(
                prompt=repair,
                system_prompt=_SYSTEM.format(task_name=self.task_name),
                seed=seed,
                call_type="judge_repair",
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=0.0,
            )
            reprompted = True
            obj = _extract_json(gen2.text)
            gen = gen2
        verdict_raw = str((obj or {}).get("verdict", "uncertain"))
        reason = str((obj or {}).get("reason", ""))
        return JudgeCall(
            round_index=round_index,
            candidate_is_a=candidate_is_a,
            raw=gen.text,
            verdict_raw=verdict_raw,
            verdict_mapped=_map_verdict(verdict_raw, candidate_is_a),
            reason=reason,
            reprompted=reprompted,
            input_tokens=gen.input_tokens,
            output_tokens=gen.output_tokens,
            latency_s=gen.latency,
            parse_error=parse_error,
        )

    def judge(
        self,
        llm,
        model: str,
        sample_id: str,
        round_index: int,
        task_input: str,
        current: str,
        candidate: str,
    ) -> JudgeVerdict:
        if round_index == 0:
            # Round 0 is the initial draft: no ablation has diverged yet, so the
            # arrangement must be identical across arms.
            pass
        if current == candidate:
            # Byte-identical pair: the two A/B arrangements are the *same*
            # prompt, so a consistent slot preference (the model answering "A"
            # both times) was being mapped to contradictory labels and recorded
            # as ``uncertain``.  Measured on the Phase 4 traces: 84 such rounds,
            # 81 of them with reason_a == reason_b verbatim.  Nothing can be
            # preferred when the two texts are equal, so return ``tie`` without
            # spending two engine calls.  This is a pure function of the two
            # texts, so it can be applied to already-written traces as well.
            return JudgeVerdict(
                verdict="tie",
                order_consistent=False,
                reason_a="identical: candidate equals current verbatim",
                reason_b="identical: candidate equals current verbatim",
                calls=[],
                structured_failure=False,
            )
        slot_a, slot_b, candidate_is_a = randomised_ab(
            model, self.task, sample_id, round_index, current, candidate
        )
        call1 = self._one_call(
            llm, sample_id, round_index, 0, candidate_is_a, task_input, slot_a, slot_b
        )
        # swapped arrangement
        slot_a2, slot_b2 = slot_b, slot_a
        call2 = self._one_call(
            llm, sample_id, round_index, 1, not candidate_is_a, task_input, slot_a2, slot_b2
        )
        verdict = _combine(call1.verdict_mapped, call2.verdict_mapped)
        order_consistent = (
            call1.verdict_mapped == call2.verdict_mapped
            and call1.verdict_mapped in ("candidate", "current")
        )
        return JudgeVerdict(
            verdict=verdict,
            order_consistent=order_consistent,
            reason_a=call1.reason,
            reason_b=call2.reason,
            calls=[call1, call2],
            structured_failure=bool(call1.parse_error and call2.parse_error),
        )
