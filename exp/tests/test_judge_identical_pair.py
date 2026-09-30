#!/usr/bin/env python
"""Regression test: a byte-identical A/B pair must be judged ``tie``.

Why this exists
---------------
When the candidate revision is byte-identical to the current answer, the two
A/B arrangements are the *same prompt*.  A model that consistently answers "A"
then gets mapped through two contradictory slot assignments, so the pair was
recorded ``uncertain`` instead of ``tie``.  Measured on the real Phase 4 traces:
84 such rounds, 81 of them with ``reason_a == reason_b`` verbatim.

The first fix only patched ``PairwiseJudge.judge`` (the *unbatched* path).  The
production path is ``BatchedPipeline._round``, which builds its judge requests
inline and calls ``_combine`` itself -- so the fix was **inert on the path that
actually produces the data**.  This test pins BOTH paths, because a fix that
only covers the path nobody runs is worse than no fix: it looks correct.

It also asserts that an identical pair costs **zero** judge calls.  Besides
being semantically empty work, those calls were inflating the recorded cost of
every affected round.

GPU-free: a stub engine echoes the current answer verbatim.

Run:  /home/ymb/miniconda3/envs/qwen35/bin/python tests/test_judge_identical_pair.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core  # noqa: F401  (registers baseline_core)
from baseline_core.tasks import get_adapter  # noqa: E402
from baseline_core.types import Generation  # noqa: E402
from core.batched_pipeline import BatchedPipeline  # noqa: E402
from core.bm25_fields import ExperienceRetriever  # noqa: E402
from core.judge import PairwiseJudge  # noqa: E402
from core.manifest import manifest_path, read_manifest  # noqa: E402
from core.pipeline import RunConfig  # noqa: E402

import run_experiment as rex  # noqa: E402

FAILURES: list[str] = []
TASK, MODEL = "wmt19_en_zh", "qwen3-8b"
IDENTICAL_REASON = "identical: candidate equals current verbatim"


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class NoEngine:
    """Fails loudly if an engine is touched -- used for the unbatched path."""

    backend = "stub"

    def generate(self, *a, **k):
        raise AssertionError("an identical pair must not call the engine")

    def count_tokens(self, text: str) -> int:
        return 1


class EchoRefinerLLM:
    """Controller always REFINEs; the refiner echoes the current answer verbatim.

    Everything the model sees is real -- the refine prompt, the controller
    schema, the judge template, the scorer -- only the generated text is chosen
    to create the degenerate case on purpose.
    """

    backend = "stub"

    def __init__(self) -> None:
        self.refine_calls = 0
        self.judge_calls = 0

    def count_tokens(self, text: str) -> int:
        return max(1, len(str(text)) // 4)

    def generate(self, prompt, system_prompt, seed, call_type, max_tokens,
                 temperature=0.1, top_p=1.0):
        if call_type.startswith("controller"):
            text = json.dumps({"action": "REFINE",
                               "instruction": "polish the wording",
                               "reason": "one concrete defect"})
        elif call_type.startswith("judge"):
            self.judge_calls += 1
            text = json.dumps({"verdict": "A", "reason": "looks fine"})
        else:
            self.refine_calls += 1
            # The exact text the pipeline put in the "Current answer:" slot.
            text = prompt.split("Current answer:\n", 1)[1].split(
                "\n\nRevision instruction:", 1)[0]
        return Generation(text=text, input_tokens=1, output_tokens=1, latency=0.0,
                          seed=seed, call_type=call_type, start_time=0.0,
                          end_time=0.0, prompt=prompt)


def main() -> int:
    # ---- 1. unbatched path -------------------------------------------------
    v = PairwiseJudge(TASK, "translation").judge(
        NoEngine(), MODEL, "s1", 0, "input", "同一个答案", "同一个答案")
    check("unbatched: identical pair -> tie", v.verdict == "tie", v.verdict)
    check("unbatched: identical pair makes no engine call", len(v.calls) == 0,
          f"{len(v.calls)} calls")

    # a genuinely different pair must still go to the engine
    class OneJudge:
        backend = "stub"
        def __init__(self): self.n = 0
        def count_tokens(self, t): return 1
        def generate(self, prompt, system_prompt, seed, call_type, max_tokens,
                     temperature=0.1, top_p=1.0):
            self.n += 1
            slot_a = prompt.split("Answer A:", 1)[1].split("Answer B:", 1)[0]
            return Generation(text=json.dumps({"verdict": "A" if "REVISED" in slot_a else "B",
                                               "reason": "clearer"}),
                              input_tokens=1, output_tokens=1, latency=0.0, seed=seed,
                              call_type=call_type, start_time=0.0, end_time=0.0, prompt=prompt)
    oj = OneJudge()
    v2 = PairwiseJudge(TASK, "translation").judge(
        oj, MODEL, "s1", 0, "input", "OLD", "REVISED new text")
    check("unbatched: a DIFFERENT pair still judges normally (not short-circuited)",
          oj.n == 2 and v2.verdict in ("better", "worse", "tie", "uncertain"),
          f"calls={oj.n} verdict={v2.verdict}")

    # ---- 2. batched path (the one production actually uses) ----------------
    refs = read_manifest(manifest_path(TASK, "test"))[:3]
    lib = rex.build_library(MODEL, TASK, "full")
    cfg = RunConfig(task=TASK, model=MODEL, experience_mode="full",
                    stop_mode="adaptive", alpha=0.5, k=4, max_rounds=3, seed=42,
                    snapshot_id="initial", draft_source="stored", draft_cache="")
    llm = EchoRefinerLLM()
    traces = BatchedPipeline(cfg, ExperienceRetriever(lib), list(lib), llm,
                             cache=None, batch_size=3, online=False).run(refs)

    rounds = [r for tr in traces for r in tr.rounds]
    verdicts = collections.Counter(r.judge_verdict for r in rounds)
    check("batched: the run actually produced revision rounds", len(rounds) > 0)
    check("batched: every round really is an identical pair "
          "(so the case under test is present, not vacuous)",
          all(r.candidate == tr.initial_draft for tr in traces for r in tr.rounds),
          "stub did not reproduce the degenerate case")
    check("batched: NO round is recorded 'uncertain'",
          verdicts.get("uncertain", 0) == 0, dict(verdicts))
    check("batched: every identical pair is recorded 'tie'",
          verdicts.get("tie", 0) == len(rounds), dict(verdicts))
    check("batched: identical pairs cost ZERO judge calls",
          llm.judge_calls == 0, f"{llm.judge_calls} judge calls for {len(rounds)} rounds")
    check("batched: the short-circuit is visible in the reasons",
          all(r.judge_reason_a == IDENTICAL_REASON and r.judge_reason_b == IDENTICAL_REASON
              for r in rounds))
    check("batched: a tie is never accepted", not any(r.accepted for r in rounds))

    # ---- 3. the two paths must agree ---------------------------------------
    check("the two judge paths agree on the verdict for the same input",
          v.verdict == "tie" and verdicts.get("tie", 0) == len(rounds))

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print("all identical-pair judge checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
