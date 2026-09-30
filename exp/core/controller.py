"""Experience-conditioned controller: decide REFINE vs STOP, and how to refine.

Protocol (plan v5 sections 3 and 7):

* Input: the task input, the current answer, and up to four retrieved
  experiences.
* Output: strict JSON ``{"action": "REFINE"|"STOP", "instruction": ..., "reason": ...}``
  decoded at ``temperature=0`` with a 256-token cap.
* A malformed reply gets exactly **one** repair retry; if that also fails the
  decision is recorded as a structured failure and the current answer is kept.
  A decision is never fabricated.

Arm variations are confined to two orthogonal knobs:

``experience_mode``
    ``full`` / ``random`` / ``positive_only`` populate the block;
    ``outcome_hidden`` renders the *same* experiences with the outcome lines
    omitted; ``none`` renders nothing at all.

``stop_mode``
    ``adaptive`` lets the controller choose STOP; ``fixed`` removes the option
    from the schema and forces REFINE for the full budget.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .determinism import call_seed

CONTROLLER_MAX_TOKENS = 256

_SYSTEM = (
    "You are the controller of an iterative {task_name} system. "
    "You decide whether another revision round is worthwhile, and if so you write "
    "the instruction for it. You do not write the answer yourself. Reply with JSON only."
)

_BASE = """Task input:
{task_input}

Current answer:
{current}
{experience_block}
Decide whether asking for another revision of the current answer is worth the compute.

Guidance:
- First try to state one concrete, specific defect in the current answer.
- If you cannot name such a defect, choose STOP.
- If you can name one, choose REFINE only when fixing it is likely to be worth another
  revision; otherwise choose STOP.
- The experiences above are evidence about what kinds of revision helped or hurt in
  similar situations. They are not the only consideration.

Do not presume that revision is either helpful or harmful in general: decide from the
defect you can actually name and from the evidence above.

{output_schema}"""

_SCHEMA_ADAPTIVE = (
    'Return JSON with exactly these keys:\n'
    '{{"action": "REFINE" | "STOP", "instruction": "<what to change; empty if STOP>", '
    '"reason": "<one short sentence>"}}'
)

_SCHEMA_FIXED = (
    'Return JSON with exactly these keys:\n'
    '{{"action": "REFINE", "instruction": "<what to change>", '
    '"reason": "<one short sentence>"}}\n'
    'Exactly one more revision round will be performed, so always produce an instruction.'
)

_EXPERIENCE_HEADER = "\nRelevant past revision experiences:\n{block}\n"

#: v2 controller prompt.  Under the v2 renderer the retrieved examples are moved
#: *out* of the controller and into the refiner prompt, matching the original
#: method's mechanism: the demonstration conditions generation directly instead
#: of being paraphrased into an instruction first.  Measured on the Phase-4
#: traces, that paraphrase was the bottleneck -- 41.2% of the instructions the
#: controller produced were near-copies of a retrieved example's advice.  With
#: no block to read, every "the experiences above" clause must go, or the
#: controller is told to consult evidence it was never given.
_BASE_V2 = """Task input:
{task_input}

Current answer:
{current}

Decide whether asking for another revision of the current answer is worth the compute.

Guidance:
- First try to state one concrete, specific defect in the current answer.
- If you cannot name such a defect, choose STOP.
- If you can name one, choose REFINE only when fixing it is likely to be worth another
  revision; otherwise choose STOP.

Do not presume that revision is either helpful or harmful in general: decide from the
defect you can actually name.

{output_schema}"""

#: v2 controller prompt for the ``both`` condition: the examples are shown to the
#: controller *as well as* the refiner.  Moving them out of the controller
#: entirely was measured to backfire on dev -- qwen3-8b/en_zh went from 52 to 96
#: REFINE rounds and from 3 to 11 accepted revisions, and its corpus gain fell
#: from -0.036 to -1.179 -- i.e. without the evidence the controller stops
#: discriminating and simply refines more, and the extra accepted revisions hurt.
_BASE_V2_WITH_EXP = """Task input:
{task_input}

Current answer:
{current}
{experience_block}
Decide whether asking for another revision of the current answer is worth the compute.

Guidance:
- First try to state one concrete, specific defect in the current answer.
- If you cannot name such a defect, choose STOP.
- If you can name one, choose REFINE only when fixing it is likely to be worth another
  revision; otherwise choose STOP.
- The experiences above are evidence about which kinds of revision actually corrected a
  wrong answer in similar situations.  They are not the only consideration.

Do not presume that revision is either helpful or harmful in general: decide from the
defect you can actually name and from the evidence above.

{output_schema}"""


@dataclass
class ControllerCall:
    raw: str
    reprompted: bool
    parse_error: Optional[str]
    input_tokens: int
    output_tokens: int
    latency_s: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ControllerDecision:
    action: str              # REFINE | STOP
    instruction: str
    reason: str
    structured_failure: bool = False
    calls: List[ControllerCall] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "instruction": self.instruction,
            "reason": self.reason,
            "structured_failure": self.structured_failure,
            "calls": [c.to_dict() for c in self.calls],
        }


def _unescape_braces(schema: str) -> str:
    """Collapse the doubled braces that only a template format would collapse.

    ``_SCHEMA_ADAPTIVE`` / ``_SCHEMA_FIXED`` are written with ``{{`` and ``}}``
    so they can be embedded in a template *literal*.  They are in fact passed as
    a substituted value, so the doubling never collapses and the model is asked
    for ``{{"action": ...}}``.  v2 corrects this; v1 keeps the historical text.
    """
    return schema.replace("{{", "{").replace("}}", "}")


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


class Controller:
    def __init__(self, task: str, task_name: str):
        self.task = task
        self.task_name = task_name

    def build_prompt(
        self,
        task_input: str,
        current: str,
        experience_block: str,
        stop_mode: str,
        renderer: str = "v1",
    ) -> str:
        exp = ""
        if experience_block.strip():
            exp = _EXPERIENCE_HEADER.format(block=experience_block.strip())
        schema = _SCHEMA_FIXED if stop_mode == "fixed" else _SCHEMA_ADAPTIVE
        if renderer == "v2":
            # The prompt must match what the controller was actually handed: a
            # block present means it may consult it, absent means no clause may
            # refer to evidence it never saw.
            if experience_block.strip():
                return _BASE_V2_WITH_EXP.format(
                    task_input=task_input,
                    current=current,
                    experience_block=_EXPERIENCE_HEADER.format(block=experience_block.strip()),
                    output_schema=_unescape_braces(schema),
                )
            return _BASE_V2.format(
                task_input=task_input,
                current=current,
                output_schema=_unescape_braces(schema),
            )
        return _BASE.format(
            task_input=task_input,
            current=current,
            experience_block=exp,
            output_schema=schema,
        )

    def decide(
        self,
        llm,
        sample_id: str,
        round_index: int,
        task_input: str,
        current: str,
        experience_block: str,
        stop_mode: str = "adaptive",
        temperature: float = 0.0,
        seed_salt: str = "",
        renderer: str = "v1",
    ) -> ControllerDecision:
        prompt = self.build_prompt(task_input, current, experience_block, stop_mode, renderer)
        seed = call_seed(sample_id, round_index, f"controller{seed_salt}")
        gen = llm.generate(
            prompt=prompt,
            system_prompt=_SYSTEM.format(task_name=self.task_name),
            seed=seed,
            call_type="controller",
            max_tokens=CONTROLLER_MAX_TOKENS,
            temperature=temperature,
        )
        calls: List[ControllerCall] = []
        obj = _extract_json(gen.text)
        parse_error = None
        reprompted = False
        if obj is None or "action" not in obj:
            parse_error = "structured_output_failed"
            repair = (
                prompt + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
            )
            gen2 = llm.generate(
                prompt=repair,
                system_prompt=_SYSTEM.format(task_name=self.task_name),
                seed=seed,
                call_type="controller_repair",
                max_tokens=CONTROLLER_MAX_TOKENS,
                temperature=0.0,
            )
            reprompted = True
            obj = _extract_json(gen2.text)
            calls.append(
                ControllerCall(
                    raw=gen.text,
                    reprompted=False,
                    parse_error=parse_error,
                    input_tokens=gen.input_tokens,
                    output_tokens=gen.output_tokens,
                    latency_s=gen.latency,
                )
            )
            gen = gen2

        calls.append(
            ControllerCall(
                raw=gen.text,
                reprompted=reprompted,
                parse_error=None if obj else "structured_output_failed",
                input_tokens=gen.input_tokens,
                output_tokens=gen.output_tokens,
                latency_s=gen.latency,
            )
        )

        if not obj or "action" not in obj:
            return ControllerDecision(
                action="STOP",
                instruction="",
                reason="structured output failed after one repair retry",
                structured_failure=True,
                calls=calls,
            )

        action = str(obj.get("action", "")).strip().upper()
        if stop_mode == "fixed":
            action = "REFINE"
        if action not in ("REFINE", "STOP"):
            action = "REFINE" if str(obj.get("instruction", "")).strip() else "STOP"
        return ControllerDecision(
            action=action,
            instruction=str(obj.get("instruction", "")).strip(),
            reason=str(obj.get("reason", "")).strip(),
            structured_failure=False,
            calls=calls,
        )
