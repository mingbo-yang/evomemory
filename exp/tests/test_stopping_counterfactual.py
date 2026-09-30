#!/usr/bin/env python
"""Phase 7 self-test: the counterfactual path of run_stopping_diagnostics.py.

The ``--counterfactual`` mode needs a GPU, so this test drives the *same*
function (``counterfactual_one``) with a stub engine: prompt construction,
experience re-rendering, controller instruction, judge and scorer are the real
ones, only the generated text is fake.  It therefore checks the wiring that a
GPU run would exercise -- the refine seed must equal the seed the pipeline would
have used for that round, the judge must be the double-order judge, the metric
must be the shared Scorer's -- without touching CUDA.

Run:  /home/ymb/miniconda3/envs/qwen35/bin/python tests/test_stopping_counterfactual.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core  # noqa: F401  (registers baseline_core)
from baseline_core.types import Generation  # noqa: E402
from core.determinism import call_seed  # noqa: E402
from core.scoring import Scorer  # noqa: E402

import run_stopping_diagnostics as rsd  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class FakeLLM:
    """Stub engine: controller -> REFINE + instruction, judge -> prefers REVISED."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def count_tokens(self, text: str) -> int:
        return max(1, len(str(text)) // 4)

    def generate(self, prompt, system_prompt, seed, call_type, max_tokens,
                 temperature=0.1, top_p=1.0):
        self.calls.append({"call_type": call_type, "seed": seed, "prompt": prompt})
        if call_type.startswith("controller"):
            text = json.dumps({"action": "REFINE",
                               "instruction": "Fix the mistranslated technical term.",
                               "reason": "one term is wrong"})
        elif call_type.startswith("judge"):
            slot_a = prompt.split("Answer A:", 1)[1].split("Answer B:", 1)[0]
            text = json.dumps({"verdict": "A" if "REVISED" in slot_a else "B",
                               "reason": "more faithful"})
        else:
            text = "REVISED " + prompt.split("Current answer:", 1)[1].split("\n", 1)[0]
        return Generation(text=text.strip(), input_tokens=10, output_tokens=5,
                          latency=0.01, seed=seed, call_type=call_type,
                          start_time=0.0, end_time=0.01, prompt=prompt)


def find_state(want_round: int | None, prefer_model: str | None = None):
    """Any trace under runs/ that has a usable STOP state (skips deleted runs)."""
    for path in sorted((ROOT / "runs").rglob("*.jsonl")):
        recs, bad = rsd.load_traces(path)
        if not recs or not recs[0].get("config"):
            continue
        rec = {"path": path, "tag": path.parent.name, "arm": path.stem,
               "task": recs[0]["task"], "model": recs[0]["model"],
               "seed": recs[0].get("seed"), "config": recs[0]["config"],
               "records": recs, "bad_lines": bad}
        states = [s for bucket in rsd.collect_stop_states([rec], None).values() for s in bucket]
        if want_round is not None:
            states = [s for s in states if int(s["round"]["round_index"]) == want_round]
        states = [s for s in states if s["round"].get("exp_ids")]
        if prefer_model is not None:
            states = [s for s in states if s["run"]["model"] == prefer_model]
        if states:
            return rec, states[0]
    return None, None


def run_case(want_round: int | None, prefer_model: str | None = None) -> bool:
    rec, state = find_state(want_round, prefer_model)
    if rec is None:
        print(f"[SKIP] no trace under runs/ with a STOP state at round={want_round}")
        return False
    lib = rsd.library_for_run(rec, {})
    check(f"library resolves for {rec['model']}/{rec['task']}", bool(lib))
    if not lib:
        return False
    llm = FakeLLM()
    pipe = rsd._build_pipeline(rec["config"], lib, llm)
    row = rsd.counterfactual_one(llm, pipe, rec, state)

    tid, t = row["sample_id"], row["round_index"]
    refine_calls = [c for c in llm.calls if c["call_type"] == "refine"]
    check(f"{tid} r{t}: exactly one refine call", len(refine_calls) == 1)
    check(f"{tid} r{t}: refine seed == call_seed(sample, round, 'refine')",
          bool(refine_calls) and refine_calls[0]["seed"] == call_seed(tid, t, "refine"))
    check(f"{tid} r{t}: judge is the double-order judge (verdict better)",
          row["judge_verdict"] == "better" and row["would_accept"] == 1, str(row))
    check(f"{tid} r{t}: premature flag is consistent with the delta",
          row["premature"] == int(row["cf_delta"] > 0.0), str(row))
    check(f"{tid} r{t}: instruction comes from the controller",
          row["instruction"] == "Fix the mistranslated technical term.")
    check(f"{tid} r{t}: no experience id was missing from the library",
          row["n_missing_experiences"] == 0)
    check(f"{tid} r{t}: experiences were re-rendered", row["n_experiences"] > 0)

    # state reconstruction must equal replaying the trace's acceptance flags
    trace = state["trace"]
    expect = trace["initial_draft"]
    for rd in trace["rounds"]:
        if int(rd["round_index"]) >= t:
            break
        if rd["controller_action"] == "REFINE" and rd["accepted"] and rd["candidate"]:
            expect = rd["candidate"]
    check(f"{tid} r{t}: reconstructed state matches the trace",
          rsd.reconstruct_current(trace, t) == expect)

    # the metric must be the shared scorer applied to (reference, text)
    ref = rsd.refs_for(row["task"], row["split"])[tid].reference
    check(f"{tid} r{t}: current metric is Scorer.primary(reference, current)",
          abs(Scorer(row["task"]).primary(ref, expect) - row["current_metric"]) < 1e-6)
    return True


def main() -> int:
    ran = 0
    ran += int(run_case(0))
    ran += int(run_case(1))
    if not ran:
        print("SKIP: no usable STOP states under runs/")
        return 0
    print()
    if FAILURES:
        print(f"FAILURES: {FAILURES}")
        return 1
    print("OK: counterfactual path verified with a stub engine (no GPU used)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
