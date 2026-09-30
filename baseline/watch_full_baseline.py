#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import MODEL_CONFIGS, TASK_CONFIGS
from run_full_baseline_scheduler import job_complete, methods_for_task


ERROR_PATTERNS = (
    "Traceback",
    "RuntimeError",
    "ERROR",
    "OutOfMemory",
    "CUDA out of memory",
    "Engine core initialization failed",
    "Exception",
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {message}\n")


def ps_rows() -> List[Tuple[int, int, str]]:
    out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    rows: List[Tuple[int, int, str]] = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def parse_arg(args: List[str], name: str) -> Optional[str]:
    for i, value in enumerate(args):
        if value == name and i + 1 < len(args):
            return args[i + 1]
    return None


def active_scheduler_pids(output_dir: Path) -> List[int]:
    marker = str(output_dir)
    scheduler_script = str(ROOT / "run_full_baseline_scheduler.py")
    pids = []
    for pid, _, cmd in ps_rows():
        if marker not in cmd:
            continue
        try:
            args = shlex.split(cmd)
        except Exception:
            args = cmd.split()
        if scheduler_script in args:
            pids.append(pid)
    return pids


def disallowed_baseline_pids(output_dir: Path, allowed_gpus: Set[str]) -> List[Tuple[int, Optional[str], str]]:
    marker = str(output_dir)
    baseline_script = str(ROOT / "run_baseline.py")
    bad = []
    for pid, _, cmd in ps_rows():
        if marker not in cmd:
            continue
        try:
            args = shlex.split(cmd)
        except Exception:
            args = cmd.split()
        if baseline_script not in args:
            continue
        gpu = parse_arg(args, "--gpu")
        if gpu not in allowed_gpus:
            bad.append((pid, gpu, cmd))
    return bad


def descendants(parent_pid: int) -> List[int]:
    rows = ps_rows()
    children: Dict[int, List[int]] = {}
    for pid, ppid, _ in rows:
        children.setdefault(ppid, []).append(pid)
    stack = list(children.get(parent_pid, []))
    found = []
    while stack:
        pid = stack.pop()
        found.append(pid)
        stack.extend(children.get(pid, []))
    return found


def terminate_tree(pid: int) -> None:
    targets = descendants(pid) + [pid]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for target in targets:
            try:
                os.kill(target, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(3)


def recent_errors(log_dir: Path, limit: int = 20) -> List[str]:
    hits: List[str] = []
    if not log_dir.exists():
        return hits
    for path in sorted(log_dir.glob("*.log")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for lineno, line in enumerate(lines, start=1):
            if any(pattern in line for pattern in ERROR_PATTERNS):
                hits.append(f"{path.name}:{lineno}:{line[:500]}")
    return hits[-limit:]


def validate_csv(csv_path: Path) -> Tuple[int, int, int, int]:
    rows = 0
    empty = 0
    bad_metric = 0
    bad_cost = 0
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        for row in reader:
            rows += 1
            if not row.get("final_output", "").strip():
                empty += 1
            try:
                metric = float(row.get("final_metric", "nan"))
                if metric != metric:
                    bad_metric += 1
            except Exception:
                bad_metric += 1
            try:
                tokens = float(row.get("total_tokens", "0"))
                latency = float(row.get("total_latency_s", "0"))
                if tokens <= 0 or latency <= 0:
                    bad_cost += 1
            except Exception:
                bad_cost += 1
        if not {"final_output", "final_metric", "total_tokens", "total_latency_s"} <= fields:
            bad_cost = max(bad_cost, rows or 1)
    return rows, empty, bad_metric, bad_cost


def expected_rows_for(task: str, max_samples: Optional[int]) -> int:
    default_rows = TASK_CONFIGS[task].default_samples
    return default_rows if max_samples is None else min(default_rows, max_samples)


def output_status(output_dir: Path, max_samples: Optional[int]) -> Dict[str, object]:
    completed_methods = 0
    bad_completed = []
    missing_methods = 0
    for model in MODEL_CONFIGS:
        for task, task_config in TASK_CONFIGS.items():
            for method in methods_for_task(task):
                csv_path = output_dir / task / model / f"{method}.csv"
                jsonl_path = output_dir / task / model / f"{method}.jsonl"
                summary_path = output_dir / task / model / f"{method}.summary.json"
                if not (csv_path.exists() and jsonl_path.exists() and summary_path.exists()):
                    missing_methods += 1
                    continue
                rows, empty, bad_metric, bad_cost = validate_csv(csv_path)
                try:
                    jsonl_rows = sum(1 for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip())
                except Exception:
                    jsonl_rows = -1
                completed_methods += 1
                expected = expected_rows_for(task, max_samples)
                if rows < expected or jsonl_rows < expected or empty or bad_metric or bad_cost:
                    bad_completed.append(
                        {
                            "path": str(csv_path.relative_to(output_dir)),
                            "rows": rows,
                            "jsonl_rows": jsonl_rows,
                            "expected": expected,
                            "empty": empty,
                            "bad_metric": bad_metric,
                            "bad_cost": bad_cost,
                        }
                    )
    return {
        "completed_methods": completed_methods,
        "missing_methods": missing_methods,
        "bad_completed": bad_completed[:50],
        "bad_completed_count": len(bad_completed),
    }


def all_jobs_complete(output_dir: Path, max_samples: Optional[int]) -> bool:
    for model in MODEL_CONFIGS:
        for task in TASK_CONFIGS:
            class JobLike:
                pass
            job = JobLike()
            job.model = model
            job.task = task
            if not job_complete(output_dir, job, max_samples):
                return False
    return True


def launch_scheduler(args: argparse.Namespace, log_path: Path) -> int:
    cmd = [
        args.python,
        str(ROOT / "run_full_baseline_scheduler.py"),
        "--output-dir",
        str(args.output_dir),
        "--gpus",
        args.gpus,
        "--poll-interval",
        str(args.scheduler_poll_interval),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    f = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, text=True)
    return proc.pid


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch and guard the full AAAI2027 baseline run.")
    parser.add_argument("--python", default="/home/ymb/miniconda3/envs/qwen35/bin/python")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_full_main_qwen35_vllm")
    parser.add_argument("--gpus", default="0,1,2,5")
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--scheduler-poll-interval", type=int, default=30)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    watch_log = args.output_dir / "watchdog.log"
    scheduler_log = args.output_dir / "scheduler.log"
    state_path = args.output_dir / "watchdog_state.json"
    allowed_gpus = {g.strip() for g in args.gpus.split(",") if g.strip()}

    append_log(
        watch_log,
        f"watchdog start output_dir={args.output_dir} gpus={sorted(allowed_gpus)} max_samples={args.max_samples}",
    )
    while True:
        bad_gpu_jobs = disallowed_baseline_pids(args.output_dir, allowed_gpus)
        for pid, gpu, cmd in bad_gpu_jobs:
            append_log(watch_log, f"terminate disallowed gpu job pid={pid} gpu={gpu} cmd={cmd}")
            terminate_tree(pid)

        scheduler_pids = active_scheduler_pids(args.output_dir)
        complete = all_jobs_complete(args.output_dir, args.max_samples)
        restarted_pid = None
        if not scheduler_pids and not complete:
            restarted_pid = launch_scheduler(args, scheduler_log)
            append_log(watch_log, f"restart scheduler pid={restarted_pid}")

        status = output_status(args.output_dir, args.max_samples)
        errors = recent_errors(log_dir)
        state = {
            "updated_at": time.time(),
            "updated_at_text": now(),
            "allowed_gpus": sorted(allowed_gpus),
            "scheduler_pids": scheduler_pids,
            "restarted_pid": restarted_pid,
            "complete": complete,
            "bad_gpu_jobs": [{"pid": pid, "gpu": gpu, "cmd": cmd} for pid, gpu, cmd in bad_gpu_jobs],
            "recent_errors": errors,
            **status,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        append_log(
            watch_log,
            "status "
            f"complete={complete} schedulers={scheduler_pids or ([restarted_pid] if restarted_pid else [])} "
            f"completed_methods={status['completed_methods']} bad_completed={status['bad_completed_count']} "
            f"errors={len(errors)} bad_gpu_jobs={len(bad_gpu_jobs)}",
        )

        if complete:
            append_log(watch_log, "all jobs complete; watchdog exit")
            return 0
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
