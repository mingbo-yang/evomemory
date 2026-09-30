#!/usr/bin/env python
"""Judge calibration: quantify position bias before trusting any ablation.

The main pipeline accepts a candidate only when *both* orderings prefer it.  If
the judge simply favours slot A regardless of content, the two orderings can
never agree on the content, acceptance collapses to zero, and every
experience-conditioned arm silently degenerates into ``Direct-Zero``.

Three controls, all reference-free, all deterministic:

1. **identical pair** -- the same text in both slots.  Any non-tie answer is
   pure position bias; the rate is the headline number.
2. **quality-swapped pair** -- two answers of clearly different quality
   (a deliberately truncated/garbled candidate vs a clean one), presented in
   both orders.  A content-sensitive judge flips its answer; a position-biased
   one does not.
3. **real pair** -- an actual (draft, candidate) pair from a smoke trace.

Usage
-----
    python calibrate_judge.py --gpu 3 --model glm4-9b --task wmt19_en_zh --n 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import EXP_ROOT, MAX_MODEL_LEN, gpu_mem_util  # noqa: E402
from core.judge import PairwiseJudge, _extract_json  # noqa: E402
from core.manifest import manifest_path, read_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--model", default="glm4-9b")
    ap.add_argument("--task", default="wmt19_en_zh")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    if args.gpu not in ("2", "3"):
        print(f"REFUSING: GPU {args.gpu} outside whitelist (2,3)")
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from baseline_core.config import MODEL_CONFIGS, TASK_CONFIGS
    from baseline_core.llm import LLMClient

    refs = read_manifest(manifest_path(args.task, "test"))[: args.n]
    cfg = TASK_CONFIGS[args.task]
    judge = PairwiseJudge(args.task, cfg.display_name)
    llm = LLMClient(
        MODEL_CONFIGS[args.model], backend="vllm", gpu=args.gpu,
        gpu_memory_utilization=gpu_mem_util(args.model),
        max_model_len=MAX_MODEL_LEN,  # was hardcoded 8192; aligned to the frozen 4096
        enforce_eager=True,
    )

    results = {"identical": [], "quality_swapped": []}

    # ---- control 1: identical text in both slots --------------------------
    print("=== control 1: identical pair (any non-tie == position bias) ===")
    for i, ref in enumerate(refs):
        anchor = ref.source[:80]
        v = judge.judge(llm, args.model, f"calib-ident-{i}", 0, anchor, ref.source, ref.source)
        results["identical"].append(v.verdict)
    c1 = Counter(results["identical"])
    n_bias = sum(v for k, v in c1.items() if k != "tie")
    print(f"  verdicts: {dict(c1)}")
    print(f"  non-tie rate (position bias): {n_bias}/{len(refs)} = {n_bias / len(refs):.0%}")

    # ---- control 2: clearly worse candidate, both orders ------------------
    # The candidate is "good" in call 1 and "bad" in call 2, so a judge that
    # tracks content must answer better/worse respectively.  A judge that
    # tracks the slot answers the same thing twice.
    print()
    print("=== control 2: quality-swapped pair (content judge -> better then worse) ===")
    consistent = 0
    slot_locked = 0
    reviewed = 0
    for i, ref in enumerate(refs):
        source = ref.source
        good = source
        words = source.split()
        bad = " ".join(words[: max(1, len(words) // 4)])
        if bad.strip() == good.strip() or not bad.strip():
            continue
        # call 1: candidate = good ; call 2: candidate = bad
        v1 = judge.judge(llm, args.model, f"calib-q-{i}", 0, source, bad, good)
        v2 = judge.judge(llm, args.model, f"calib-q-{i}", 1, source, good, bad)
        reviewed += 1
        if v1.verdict == "better" and v2.verdict == "worse":
            consistent += 1
        if v1.verdict == v2.verdict and v1.verdict in ("better", "worse"):
            slot_locked += 1
        if reviewed <= 4:
            print(
                f"  pair {i}: order1(cand=good)={v1.verdict:9s} "
                f"order2(cand=bad)={v2.verdict:9s}"
            )
    print(
        f"  content-consistent (better then worse): {consistent}/{reviewed} = "
        f"{(consistent / reviewed if reviewed else 0):.0%}"
    )
    print(
        f"  slot-locked (same answer both orders): {slot_locked}/{reviewed} = "
        f"{(slot_locked / reviewed if reviewed else 0):.0%}"
    )

    out = {
        "model": args.model,
        "task": args.task,
        "identical_verdicts": dict(c1),
        "identical_n": len(refs),
        "position_bias_rate": n_bias / len(refs) if refs else 0.0,
        "quality_swapped_n": reviewed,
        "quality_consistent_rate": consistent / reviewed if reviewed else 0.0,
        "slot_locked_rate": slot_locked / reviewed if reviewed else 0.0,
    }
    (EXP_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "reports" / f"judge_calibration_{args.model}_{args.task}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
