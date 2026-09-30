#!/usr/bin/env python
"""Phase 0b: Direct-Zero reproduction gate.

Plan v5 section 2.6.  Before any stored artefact may be reused as ``y0``, we
must show that the pinned environment reproduces the stored Direct-Zero outputs
on this machine.  Five samples per model-task are regenerated and compared
byte-for-byte against the stored CSV.

Outcome
-------
* all identical -> ``draft_source = "stored"`` is enabled (the default).
* any divergence -> the gate fails loudly and every run must regenerate its own
  draft (``--draft-source generate``), so the comparison stays honest.

Usage
-----
    /home/ymb/miniconda3/envs/qwen35/bin/python repro_gate.py --gpu 2 [--samples 5]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    EXP_ROOT,
    MODELS,
    TASKS,
    TEST_SAMPLES,
    baseline_results_dir,
    ensure_gpu_whitelist,
    gpu_mem_util,
)
from core.determinism import sample_seed  # noqa: E402
from core.manifest import read_manifest, manifest_path  # noqa: E402


def load_stored(task: str, model: str) -> List[dict]:
    path = baseline_results_dir() / task / model / "Direct-Zero.csv"
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--tasks", default=",".join(TASKS))
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpu.split(",") if g.strip()]
    from core import ALLOWED_GPUS, MAX_GPUS
    illegal = [g for g in gpus if g not in ALLOWED_GPUS]
    if illegal or len(gpus) > MAX_GPUS:
        print(f"REFUSING: {gpus} violates the GPU budget (any {MAX_GPUS} of {ALLOWED_GPUS})")
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
    ensure_gpu_whitelist()

    from baseline_core.config import MODEL_CONFIGS, TASK_CONFIGS
    from baseline_core.llm import LLMClient
    from baseline_core.tasks import get_adapter
    from baseline_core.types import TaskExample

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    report: Dict[str, dict] = {}
    n_match = n_total = 0
    import torch

    # model-outer / task-inner: each checkpoint is loaded exactly once, which
    # matters a lot for the 32B model.
    for model in models:
        llm = None
        for task in tasks:
            try:
                stored = load_stored(task, model)
            except FileNotFoundError:
                report[f"{task}/{model}"] = {"status": "no_stored_output"}
                continue
            if llm is None:
                llm = LLMClient(
                    MODEL_CONFIGS[model],
                    backend="vllm",
                    gpu=gpus[0],
                    gpu_memory_utilization=gpu_mem_util(model),
                    max_model_len=4096,
                    enforce_eager=True,
                )
            refs = read_manifest(manifest_path(task, "test"))[: args.samples]
            adapter = get_adapter(task)
            task_cfg = TASK_CONFIGS[task]
            matches = []
            for i, ref in enumerate(refs):
                ex = TaskExample(index=i, source=ref.source, reference=ref.reference, task=task)
                gen = llm.generate(
                    prompt=adapter.initial_prompt(ex),
                    system_prompt=adapter.system_prompt(),
                    seed=42 + i * 9973,  # exactly the baseline seed formula
                    call_type="initial",
                    max_tokens=task_cfg.max_tokens,
                    temperature=0.1,
                    top_p=1.0,
                )
                produced = adapter.parse_output(gen.text)
                expected = (stored[i]["final_output"] if i < len(stored) else "").strip()
                matches.append(produced.strip() == expected)
            n = sum(matches)
            n_match += n
            n_total += len(matches)
            report[f"{task}/{model}"] = {
                "status": "ok" if n == len(matches) else "mismatch",
                "matched": n,
                "total": len(matches),
            }
            print(f"[GATE] {task:14s} {model:12s} {n}/{len(matches)} identical", flush=True)
        if llm is not None:
            del llm
            torch.cuda.empty_cache()

    verdict = "PASS" if n_total and n_match == n_total else "FAIL"
    reports_dir = EXP_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "repro_gate.json"
    # The driver runs one process per model, so merge instead of overwriting.
    merged: Dict[str, dict] = {"per_model_task": {}}
    if report_path.exists():
        try:
            merged = json.loads(report_path.read_text(encoding="utf-8"))
            merged.setdefault("per_model_task", {})
        except Exception:
            pass
    merged["per_model_task"].update(report)
    all_hits = [v for v in merged["per_model_task"].values()]
    n_match_all = sum(v.get("matched", 0) for v in all_hits)
    n_total_all = sum(v.get("total", 0) for v in all_hits)
    merged["matched"] = n_match_all
    merged["total"] = n_total_all
    merged["verdict"] = "PASS" if n_total_all and n_match_all == n_total_all else "FAIL"
    report_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print(
        f"REPRO GATE (this process): {verdict} ({n_match}/{n_total} identical)"
    )
    print(
        f"REPRO GATE (cumulative)  : {merged['verdict']} "
        f"({n_match_all}/{n_total_all} identical)"
    )
    if verdict == "PASS":
        print("=> draft_source='stored' authorised for these model-tasks.")
    else:
        print("=> y0 reuse NOT authorised for these model-tasks.")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
