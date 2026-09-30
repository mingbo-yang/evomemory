#!/usr/bin/env python
"""Phase 3: build the shared initial experience library (+ Phase 1f precheck).

Protocol (plan v5 sections 3, 4.4 and 6)

For every auxiliary input in the ``initial`` manifest:

1. generate one initial draft,
2. draw **two different** refinement interventions from the same draft
   (the same controller prompt sampled at temperature 0.7 with two seeds, so
   the library naturally covers a range of interventions),
3. produce a candidate for each,
4. score both with the same double-order, reference-free judge,
5. store the transition verbatim -- no manual improvement/degradation labels.

The build deliberately runs with ``experience_mode="none"``: there is no library
yet, and every experiment variant must start from the *same* initial library, so
the bootstrap must not depend on retrieval.

The positive-pool precheck then counts ``better`` transitions per model-task.
If any falls below ``--min-positive`` the script stops and reports the required
global increase of interventions-per-input K, rather than letting an individual
arm be tuned after the fact.

Usage
-----
    python build_experience.py --gpu 2 --models glm4-9b,qwen3-8b
    python build_experience.py --gpu 2 --models all --tasks all --interventions 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import EXP_ROOT, MAX_MODEL_LEN, MODELS, TASKS, gpu_mem_util  # noqa: E402, resolve_model_config
from core.bm25_fields import Experience  # noqa: E402
from core.experience import experience_dir, save_experiences  # noqa: E402
from core.manifest import manifest_path, read_manifest  # noqa: E402
from core.pipeline import build_refine_prompt  # noqa: E402

CONTROLLER_BUILD_TEMPERATURE = 0.7


def build_one_model_task(llm, model: str, task: str, n_interventions: int) -> List[Experience]:
    from baseline_core.config import TASK_CONFIGS
    from baseline_core.tasks import get_adapter
    from baseline_core.types import TaskExample

    from core.controller import Controller
    from core.determinism import call_seed
    from core.judge import PairwiseJudge
    from core.scoring import Scorer

    refs = read_manifest(manifest_path(task, "initial"))
    adapter = get_adapter(task)
    task_cfg = TASK_CONFIGS[task]
    controller = Controller(task, task_cfg.display_name)
    judge = PairwiseJudge(task, task_cfg.display_name)
    # The bootstrap runs on the AUXILIARY (initial) split, whose references are
    # available, so the true gold-metric effect of every intervention can be
    # measured and stored.  Rendering the unreliable judge verdict as the
    # outcome instead left the library's only outcome field mostly noise.
    scorer = Scorer(task)

    out: List[Experience] = []
    skipped = {"empty_draft": 0, "empty_instruction": 0, "empty_candidate": 0}
    for i, ref in enumerate(refs):
        ex = TaskExample(index=i, source=ref.source, reference=ref.reference, task=task)
        draft_gen = llm.generate(
            prompt=adapter.initial_prompt(ex),
            system_prompt=adapter.system_prompt(),
            seed=call_seed(ref.sample_id, 0, "initial"),
            call_type="initial",
            max_tokens=task_cfg.max_tokens,
            temperature=0.1,
            top_p=1.0,
        )
        draft = adapter.parse_output(draft_gen.text)
        if not draft.strip():
            skipped["empty_draft"] += 1
            continue

        for k in range(n_interventions):
            # The library must contain *transitions*, so the ability to STOP is
            # removed during bootstrap: with the adaptive schema the controller
            # declines to intervene on most auxiliary inputs and the observation
            # is silently lost (measured yield was 20/64 = 31%).
            dec = controller.decide(
                llm,
                ref.sample_id,
                0,
                ref.source,
                draft,
                "",
                stop_mode="fixed",
                temperature=CONTROLLER_BUILD_TEMPERATURE,
                seed_salt=f"build{k}",
            )
            instruction = dec.instruction.strip()
            if not instruction:
                skipped["empty_instruction"] += 1
                continue
            rgen = llm.generate(
                prompt=build_refine_prompt(adapter, ex, draft, instruction),
                system_prompt=adapter.system_prompt(),
                seed=call_seed(ref.sample_id, k, "refine_build"),
                call_type="refine",
                max_tokens=task_cfg.max_tokens,
                temperature=0.1,
                top_p=1.0,
            )
            candidate = adapter.parse_output(rgen.text)
            if not candidate.strip():
                skipped["empty_candidate"] += 1
                continue
            verdict = judge.judge(
                llm, model, ref.sample_id, k, ref.source, draft, candidate
            )
            delta = scorer.primary(ref.reference, candidate) - scorer.primary(ref.reference, draft)
            label = "helped" if delta > 1e-9 else ("hurt" if delta < -1e-9 else "unchanged")
            out.append(
                Experience(
                    exp_id=f"{task}/{model}/{ref.sample_id}/b{k}",
                    task=task,
                    model=model,
                    source_input=ref.source,
                    state_before=draft,
                    state_after=candidate,
                    intervention_instruction=instruction,
                    intervention_rationale=dec.reason,
                    verdict=verdict.verdict,
                    reason_a=verdict.reason_a,
                    reason_b=verdict.reason_b,
                    order_consistent=verdict.order_consistent,
                    delta_offline=delta,
                    provenance="initial-bootstrap",
                    outcome_label=label,
                )
            )
        if (i + 1) % 8 == 0:
            print(f"    {task}/{model}: {i + 1}/{len(refs)} inputs done", flush=True)
    print(f"    skips: {skipped}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--models", default="glm4-9b,qwen3-8b")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--interventions", type=int, default=2)
    ap.add_argument("--k-map", default="coedit_gec=8",
                    help="per-task override of interventions-per-input, e.g. coedit_gec=8")
    ap.add_argument("--min-positive", type=int, default=8)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpu.split(",") if g.strip()]
    from core import ALLOWED_GPUS, MAX_GPUS
    illegal = [g for g in gpus if g not in ALLOWED_GPUS]
    if illegal or len(gpus) > MAX_GPUS:
        print(f"REFUSING: {gpus} violates the GPU budget (any {MAX_GPUS} of {ALLOWED_GPUS})")
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]

    # Per-task K, optionally narrowed to one model with "model:task=K".
    # Section 4.4 requires K to be fixed once, globally, before any arm runs;
    # K may differ between model-tasks because the library is built once per
    # model-task and every arm then reads that identical library.
    k_map = {}
    for part in (args.k_map or "").split(","):
        if "=" in part:
            key, v = part.split("=", 1)
            k_map[key.strip()] = int(v)

    def k_for(model: str, task: str, default: int) -> int:
        return k_map.get(f"{model}:{task}", k_map.get(task, default))
    models = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    tasks = list(TASKS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]

    from baseline_core.llm import LLMClient

    import torch

    report: Dict[str, dict] = {}
    short = []
    for model in models:
        llm = LLMClient(
            resolve_model_config(model),
            backend="vllm",
            gpu=gpus[0],
            gpu_memory_utilization=gpu_mem_util(model, args.gpu_memory_utilization),
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
        )
        for task in tasks:
            print(f"[BUILD] {model} / {task}", flush=True)
            k_eff = k_for(model, task, args.interventions)
            exps = build_one_model_task(llm, model, task, k_eff)
            out_path = experience_dir(model, task) / "initial.jsonl"
            save_experiences(exps, out_path)
            counts = Counter(e.verdict for e in exps)
            n_pos = counts.get("better", 0)
            report[f"{model}/{task}"] = {
                "n": len(exps),
                "k": k_eff,
                "verdicts": dict(counts),
                "positive": n_pos,
                "positive_ok": n_pos >= args.min_positive,
            }
            print(
                f"    -> {len(exps)} experiences {dict(counts)} "
                f"positive={n_pos} {'OK' if n_pos >= args.min_positive else 'LOW'}",
                flush=True,
            )
            if n_pos < args.min_positive:
                short.append(f"{model}/{task} (positive={n_pos}, need>={args.min_positive})")
        del llm
        torch.cuda.empty_cache()

    # Merge rather than overwrite: the library is often rebuilt for a single
    # task (e.g. raising K for GEC), and the report must stay cumulative.
    reports_dir = EXP_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "initial_experience.json"
    merged = {}
    if report_path.exists():
        try:
            merged = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    merged.update(report)
    report_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    if short:
        print("POSITIVE POOL PRECHECK: FAIL")
        for s in short:
            print(f"  - {s}")
        print()
        print(
            "Per plan section 4.4, K (interventions per auxiliary input) must be raised "
            "GLOBALLY and the library rebuilt; K must never be tuned per arm."
        )
        return 1
    print("POSITIVE POOL PRECHECK: PASS (every model-task has >= "
          f"{args.min_positive} positive transitions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
