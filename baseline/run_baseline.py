#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import METHODS_FIRST_AND_SECOND_BATCH, MODEL_CONFIGS, TASK_CONFIGS
from core.data import load_examples
from core.llm import LLMClient
from core.metrics import task_metric
from core.tasks import get_adapter
from core.types import InferenceBudget, ModelConfig, RunResult, TaskExample
from core.utility import UtilityPredictor
from methods.tts import METHOD_REGISTRY, METHODS_REQUIRING_UTILITY


CSV_FIELDS = [
    "task",
    "model",
    "method",
    "index",
    "source_text",
    "reference_text",
    "final_output",
    "final_metric",
    "total_tokens",
    "sequential_tokens",
    "total_latency_s",
    "sequential_latency_s",
    "stop_round",
    "num_llm_calls",
    "sample_policy",
    "backend",
]


def _expand(value: str, choices: Iterable[str]) -> List[str]:
    if value == "all":
        return list(choices)
    out = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in out if x not in choices]
    if unknown:
        raise ValueError(f"Unknown value(s): {unknown}; choices={list(choices)}")
    return out


def _output_paths(output_dir: Path, task: str, model: str, method: str):
    run_dir = output_dir / task / model
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_method = method.replace("/", "_")
    return run_dir / f"{safe_method}.csv", run_dir / f"{safe_method}.jsonl", run_dir / f"{safe_method}.summary.json"


def _count_csv_rows(path: Path) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return None


def _count_jsonl_rows(path: Path) -> Optional[int]:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return None


def _expected_rows(task: str, max_samples: Optional[int]) -> int:
    default_rows = TASK_CONFIGS[task].default_samples
    return default_rows if max_samples is None else min(default_rows, max_samples)


def _method_complete(output_dir: Path, task: str, model: str, method: str, max_samples: Optional[int]) -> bool:
    csv_path, jsonl_path, summary_path = _output_paths(output_dir, task, model, method)
    if not (csv_path.exists() and jsonl_path.exists() and summary_path.exists()):
        return False
    expected = _expected_rows(task, max_samples)
    csv_rows = _count_csv_rows(csv_path)
    jsonl_rows = _count_jsonl_rows(jsonl_path)
    return csv_rows is not None and jsonl_rows is not None and csv_rows >= expected and jsonl_rows >= expected


def _row(task: str, model: str, method: str, ex: TaskExample, res: RunResult, sample_policy: str, backend: str) -> Dict[str, object]:
    return {
        "task": task,
        "model": model,
        "method": method,
        "index": ex.index,
        "source_text": ex.source,
        "reference_text": ex.reference,
        "final_output": res.final_output,
        "final_metric": task_metric(task, ex.reference, res.final_output) if res.final_output else 0.0,
        "total_tokens": res.total_tokens,
        "sequential_tokens": res.sequential_tokens,
        "total_latency_s": res.total_latency,
        "sequential_latency_s": res.sequential_latency,
        "stop_round": res.stop_round,
        "num_llm_calls": len(res.generations),
        "sample_policy": sample_policy,
        "backend": backend,
    }


def _summary(rows: List[Dict[str, object]], meta: Dict[str, object]) -> Dict[str, object]:
    def vals(name: str) -> List[float]:
        return [float(r[name]) for r in rows]

    if not rows:
        return {**meta, "rows": 0}
    total_latency = vals("total_latency_s")
    seq_latency = vals("sequential_latency_s")
    total_tokens = vals("total_tokens")
    metrics = vals("final_metric")
    return {
        **meta,
        "rows": len(rows),
        "avg_final_metric": sum(metrics) / len(metrics),
        "avg_total_latency_s": sum(total_latency) / len(total_latency),
        "avg_sequential_latency_s": sum(seq_latency) / len(seq_latency),
        "p50_total_latency_s": statistics.median(total_latency),
        "p95_total_latency_s": sorted(total_latency)[max(0, int(0.95 * len(total_latency)) - 1)],
        "avg_total_tokens": sum(total_tokens) / len(total_tokens),
        "avg_num_llm_calls": sum(float(r["num_llm_calls"]) for r in rows) / len(rows),
    }


def run_one(
    task: str,
    model_key: str,
    method: str,
    args,
    llm: Optional[LLMClient] = None,
    utility_cache: Optional[Dict[str, UtilityPredictor]] = None,
) -> None:
    adapter = get_adapter(task)
    task_cfg = TASK_CONFIGS[task]
    if args.dry_run:
        planned_n = args.max_samples if args.max_samples is not None else task_cfg.default_samples
        csv_path, jsonl_path, _ = _output_paths(Path(args.output_dir), task, model_key, method)
        print(f"[DRY] task={task} model={model_key} method={method} planned_examples={planned_n} output={csv_path}")
        return

    if _method_complete(Path(args.output_dir), task, model_key, method, args.max_samples):
        csv_path, _, summary_path = _output_paths(Path(args.output_dir), task, model_key, method)
        rows = _count_csv_rows(csv_path)
        print(
            f"[SKIP] task={task} model={model_key} method={method} "
            f"reason=already_complete rows={rows} summary={summary_path}"
        )
        return

    examples, sample_policy = load_examples(task, args.max_samples)
    max_tokens = args.max_tokens if args.max_tokens is not None else task_cfg.max_tokens
    budget = InferenceBudget(
        max_rounds=args.max_rounds,
        n_candidates=args.n_candidates,
        max_tokens=max_tokens,
        feedback_max_tokens=args.feedback_max_tokens,
        workspace_max_tokens=args.workspace_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    csv_path, jsonl_path, summary_path = _output_paths(Path(args.output_dir), task, model_key, method)
    print(f"[RUN] task={task} model={model_key} method={method} examples={len(examples)} output={csv_path}")

    model_cfg = MODEL_CONFIGS[model_key]
    if args.tensor_parallel_size:
        model_cfg = ModelConfig(
            key=model_cfg.key,
            path=model_cfg.path,
            param_count=model_cfg.param_count,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=model_cfg.max_model_len,
        )
    if llm is None:
        llm = LLMClient(
            model_cfg,
            backend=args.backend,
            gpu=args.gpu,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            enforce_eager=args.enforce_eager,
        )
    if utility_cache is None:
        utility_cache = {}
    utility = None
    if method in METHODS_REQUIRING_UTILITY:
        if task not in utility_cache:
            utility_cache[task] = UtilityPredictor(task)
        utility = utility_cache[task]
    fn = METHOD_REGISTRY[method]

    rows: List[Dict[str, object]] = []
    started = time.time()
    with csv_path.open("w", encoding="utf-8", newline="") as csv_f, jsonl_path.open("w", encoding="utf-8") as jsonl_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i, ex in enumerate(examples):
            try:
                res = fn(ex, adapter, llm, budget, args.seed + i * 9973, utility)
            except Exception as e:
                res = RunResult("", [], [], [], -1, {"error": f"{type(e).__name__}: {e}"})
            row = _row(task, model_key, method, ex, res, sample_policy, args.backend)
            writer.writerow(row)
            rows.append(row)
            trace = {
                "task": task,
                "model": model_key,
                "method": method,
                "index": ex.index,
                "source_text": ex.source,
                "reference_text": ex.reference,
                "sample_policy": sample_policy,
                "backend": args.backend,
                **res.to_trace_dict(),
            }
            jsonl_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
            if args.progress_every and (i + 1) % args.progress_every == 0:
                print(f"[PROGRESS] {i + 1}/{len(examples)}")

    summary = _summary(
        rows,
        {
            "task": task,
            "model": model_key,
            "method": method,
            "backend": args.backend,
            "sample_policy": sample_policy,
            "duration_s": time.time() - started,
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
        },
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] summary={summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AAAI2027 TTS baselines.")
    parser.add_argument("--task", default="all", help="Task key or comma list; use all for every task.")
    parser.add_argument("--model", default="all", help="Model key or comma list; use all for every model.")
    parser.add_argument("--method", default="all", help="Method name or comma list; use all for implemented methods.")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 0,1.")
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--feedback-max-tokens", type=int, default=512)
    parser.add_argument("--workspace-max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    tasks = _expand(args.task, TASK_CONFIGS.keys())
    models = _expand(args.model, MODEL_CONFIGS.keys())
    methods = _expand(args.method, METHOD_REGISTRY.keys())

    for task in tasks:
        for model in models:
            shared_llm = None
            utility_cache: Dict[str, UtilityPredictor] = {}
            if not args.dry_run and len(methods) > 1:
                model_cfg = MODEL_CONFIGS[model]
                if args.tensor_parallel_size:
                    model_cfg = ModelConfig(
                        key=model_cfg.key,
                        path=model_cfg.path,
                        param_count=model_cfg.param_count,
                        tensor_parallel_size=args.tensor_parallel_size,
                        max_model_len=model_cfg.max_model_len,
                    )
                shared_llm = LLMClient(
                    model_cfg,
                    backend=args.backend,
                    gpu=args.gpu,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.vllm_max_model_len,
                    enforce_eager=args.enforce_eager,
                )
            for method in methods:
                if method.startswith("TEaR") and task not in {"wmt19_en_zh", "wmt19_zh_en"}:
                    print(f"[SKIP] task={task} method={method} reason=TEaR translation-only")
                    continue
                run_one(task, model, method, args, llm=shared_llm, utility_cache=utility_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
