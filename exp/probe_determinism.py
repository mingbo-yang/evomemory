#!/usr/bin/env python
"""Is the reproduction-gate divergence inference nondeterminism, or a setup error?

The gate found 58/60 stored Direct-Zero outputs reproduced exactly, with
``qwen3-4b`` / ``wmt19_zh_en`` at 4/5.  Before deciding whether ``y0`` reuse is
still sound, we need to know *why*.  Two candidate explanations:

* **setup error** -- the pinned environment differs from the one that produced
  the baseline (wrong checkpoint, wrong chat template, wrong decoding params).
  A setup error is systematic: the same prompt would give the same *wrong*
  answer every time.
* **inference nondeterminism** -- vLLM's kernels and batch scheduling are not
  bit-reproducible, so at ``temperature=0.1`` a ~1e-6 logit perturbation can
  flip a sampled token.  Nondeterminism shows up as *unstable* repeats.

The probe distinguishes them by generating the SAME prompt with the SAME seed
several times inside one process and counting distinct outputs.

Usage
-----
    python probe_determinism.py --gpu 3 --model qwen3-4b --task wmt19_zh_en --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import EXP_ROOT, baseline_results_dir, gpu_mem_util  # noqa: E402
from core.manifest import manifest_path, read_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--task", default="wmt19_zh_en")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--items", type=int, default=5)
    args = ap.parse_args()

    if args.gpu not in ("2", "3"):
        print(f"REFUSING: GPU {args.gpu} outside whitelist (2,3)")
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from baseline_core.config import MODEL_CONFIGS, TASK_CONFIGS
    from baseline_core.llm import LLMClient
    from baseline_core.tasks import get_adapter
    from baseline_core.types import TaskExample

    refs = read_manifest(manifest_path(args.task, "test"))[: args.items]
    with (baseline_results_dir() / args.task / args.model / "Direct-Zero.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        stored = list(csv.DictReader(f))

    adapter = get_adapter(args.task)
    cfg = TASK_CONFIGS[args.task]
    llm = LLMClient(
        MODEL_CONFIGS[args.model], backend="vllm", gpu=args.gpu,
        gpu_memory_utilization=gpu_mem_util(args.model),
        max_model_len=4096, enforce_eager=True,
    )

    per_item = []
    total_distinct = 0
    total_stored_hit = 0
    for i, ref in enumerate(refs):
        ex = TaskExample(index=i, source=ref.source, reference=ref.reference, task=args.task)
        prompt = adapter.initial_prompt(ex)
        seed = 42 + i * 9973  # the baseline seed formula
        outs: List[str] = []
        for _ in range(args.repeats):
            g = llm.generate(
                prompt=prompt, system_prompt=adapter.system_prompt(), seed=seed,
                call_type="probe", max_tokens=cfg.max_tokens,
                temperature=0.1, top_p=1.0,
            )
            outs.append(adapter.parse_output(g.text).strip())
        distinct = len(set(outs))
        expected = stored[i]["final_output"].strip() if i < len(stored) else ""
        hit = expected in set(outs)
        total_distinct += distinct
        total_stored_hit += int(hit)
        per_item.append(
            {
                "index": i,
                "distinct_outputs": distinct,
                "repeats": args.repeats,
                "stored_reproduced_in_repeats": hit,
                "stored_matches_prompt_run": outs[0] == expected,
            }
        )
        print(
            f"  item {i}: distinct={distinct}/{args.repeats} "
            f"stored_reproduced={hit} first_run_matches_stored={outs[0] == expected}",
            flush=True,
        )

    n = len(per_item)
    stable = sum(1 for p in per_item if p["distinct_outputs"] == 1)
    out = {
        "model": args.model,
        "task": args.task,
        "items": n,
        "repeats": args.repeats,
        "items_with_single_distinct_output": stable,
        "items_stable_rate": stable / n if n else 0.0,
        "mean_distinct_outputs": total_distinct / n if n else 0.0,
        "stored_output_reproduced_in_any_repeat": total_stored_hit,
        "stored_reproduced_rate": total_stored_hit / n if n else 0.0,
        "per_item": per_item,
        "interpretation": (
            "same seed repeated in-process must be deterministic; any variation "
            "is vLLM kernel/batching nondeterminism rather than a setup error"
        ),
    }
    (EXP_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    path = EXP_ROOT / "reports" / f"determinism_probe_{args.model}_{args.task}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(json.dumps({k: v for k, v in out.items() if k != "per_item"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
