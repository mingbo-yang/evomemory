#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import MODEL_CONFIGS, TASK_CONFIGS
from methods.tts import METHOD_REGISTRY


BASE_METHODS = [
    "Direct-Zero",
    "BoN-U-4",
    "SelfRefine-Fixed",
    "SelfRefine-U",
    "SR-Fixed",
    "SR-U",
    "PDR-2-1",
    "ChecklistRefine",
    "ModeX-4",
    "AdaCompute-SR-U",
]
TRANSLATION_EXTRA_METHODS = ["TEaR-Native", "TEaR-U"]


@dataclass
class Job:
    model: str
    task: str
    attempt: int = 0


@dataclass
class JobRecord:
    model: str
    task: str
    gpu: str
    attempt: int
    status: str
    returncode: Optional[int]
    started_at: float
    ended_at: Optional[float]
    elapsed_s: Optional[float]
    log: str


def methods_for_task(task: str) -> List[str]:
    methods = [m for m in BASE_METHODS if m in METHOD_REGISTRY]
    if task in {"wmt19_en_zh", "wmt19_zh_en"}:
        methods.extend(m for m in TRANSLATION_EXTRA_METHODS if m in METHOD_REGISTRY)
    return methods


def read_csv_rows(path: Path) -> Optional[int]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return None


def expected_rows_for(task: str, max_samples: Optional[int]) -> int:
    default_rows = TASK_CONFIGS[task].default_samples
    return default_rows if max_samples is None else min(default_rows, max_samples)


def job_complete(output_dir: Path, job: Job, max_samples: Optional[int] = None) -> bool:
    expected_rows = expected_rows_for(job.task, max_samples)
    for method in methods_for_task(job.task):
        csv_path = output_dir / job.task / job.model / f"{method}.csv"
        jsonl_path = output_dir / job.task / job.model / f"{method}.jsonl"
        summary_path = output_dir / job.task / job.model / f"{method}.summary.json"
        if not (csv_path.exists() and jsonl_path.exists() and summary_path.exists()):
            return False
        csv_rows = read_csv_rows(csv_path)
        if csv_rows is None or csv_rows < expected_rows:
            return False
        try:
            jsonl_rows = sum(1 for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip())
        except Exception:
            return False
        if jsonl_rows < expected_rows:
            return False
    return True


def gpu_snapshot() -> Dict[str, Tuple[int, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    snapshot: Dict[str, Tuple[int, int]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            snapshot[parts[0]] = (int(parts[1]), int(parts[2]))
    return snapshot


def active_run_baseline_jobs(output_dir: Path) -> Dict[Tuple[str, str], int]:
    cmd = ["ps", "-eo", "pid=,args="]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:
        return {}
    active: Dict[Tuple[str, str], int] = {}
    output_dir_text = str(output_dir)
    for line in out.splitlines():
        line = line.strip()
        if "run_baseline.py" not in line or output_dir_text not in line:
            continue
        try:
            pid_text, raw_args = line.split(None, 1)
            parts = shlex.split(raw_args)
        except Exception:
            continue
        task = None
        model = None
        for i, part in enumerate(parts):
            if part == "--task" and i + 1 < len(parts):
                task = parts[i + 1]
            elif part == "--model" and i + 1 < len(parts):
                model = parts[i + 1]
        if task and model:
            active[(model, task)] = int(pid_text)
    return active


def resolve_requested_gpus(value: str) -> List[str]:
    if value.strip().lower() != "all":
        return [g.strip() for g in value.split(",") if g.strip()]
    return sorted(gpu_snapshot(), key=lambda x: int(x))


def write_state(path: Path, queue: List[Job], running: Dict[str, Tuple[Job, subprocess.Popen, JobRecord]], records: List[JobRecord]) -> None:
    state = {
        "updated_at": time.time(),
        "queued": [asdict(j) for j in queue],
        "running": [
            {
                **asdict(record),
                "pid": proc.pid,
            }
            for _, proc, record in running.values()
        ],
        "records": [asdict(r) for r in records],
    }
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def launch_job(args, job: Job, gpu: str, log_dir: Path) -> Tuple[subprocess.Popen, JobRecord]:
    log_path = log_dir / f"{job.model}__{job.task}__attempt{job.attempt}.log"
    cmd = [
        args.python,
        str(ROOT / "run_baseline.py"),
        "--task",
        job.task,
        "--model",
        job.model,
        "--method",
        "all",
        "--backend",
        "vllm",
        "--gpu",
        gpu,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization.get(job.model, args.default_gpu_memory_utilization)),
        "--vllm-max-model-len",
        str(args.vllm_max_model_len),
        "--enforce-eager",
        "--progress-every",
        str(args.progress_every),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n")
        f.flush()
    log_f = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, text=True)
    proc._codex_log_file = log_f  # type: ignore[attr-defined]
    record = JobRecord(
        model=job.model,
        task=job.task,
        gpu=gpu,
        attempt=job.attempt,
        status="running",
        returncode=None,
        started_at=started,
        ended_at=None,
        elapsed_s=None,
        log=str(log_path),
    )
    return proc, record


def parse_gpu_memory_utilization(values: Iterable[str]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for value in values:
        if not value:
            continue
        model, raw = value.split("=", 1)
        parsed[model.strip()] = float(raw)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full AAAI2027 baseline matrix with opportunistic GPU scheduling.")
    parser.add_argument("--python", default="/home/ymb/miniconda3/envs/qwen35/bin/python")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_full_main_qwen35_vllm")
    parser.add_argument("--gpus", default="all", help="Comma-separated GPU ids, or all to scan every visible GPU.")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--idle-memory-mib", type=int, default=1000)
    parser.add_argument("--vllm-max-model-len", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--default-gpu-memory-utilization", type=float, default=0.40)
    parser.add_argument("--gpu-memory-utilization", action="append", default=[
        "glm4-9b=0.45",
        "llama3.1-8b=0.40",
        "qwen3-4b=0.30",
        "qwen3-8b=0.40",
        "qwen3-32b=0.86",
    ])
    args = parser.parse_args()
    args.gpu_memory_utilization = parse_gpu_memory_utilization(args.gpu_memory_utilization)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "scheduler_state.json"

    requested_gpus = resolve_requested_gpus(args.gpus)
    jobs = [Job(model=model, task=task) for model in MODEL_CONFIGS for task in TASK_CONFIGS]
    queue = [job for job in jobs if not job_complete(args.output_dir, job, args.max_samples)]
    records: List[JobRecord] = []
    running: Dict[str, Tuple[Job, subprocess.Popen, JobRecord]] = {}

    print(f"[SCHEDULER] output_dir={args.output_dir}", flush=True)
    print(f"[SCHEDULER] gpus={requested_gpus} queued={len(queue)} already_complete={len(jobs)-len(queue)}", flush=True)
    write_state(state_path, queue, running, records)

    while queue or running:
        finished_gpus = []
        for gpu, (job, proc, record) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log_f = getattr(proc, "_codex_log_file", None)
            if log_f is not None:
                log_f.close()
            ended = time.time()
            record.returncode = rc
            record.ended_at = ended
            record.elapsed_s = ended - record.started_at
            record.status = "ok" if rc == 0 and job_complete(args.output_dir, job, args.max_samples) else "failed"
            records.append(record)
            finished_gpus.append(gpu)
            del running[gpu]
            print(
                f"[END] gpu={gpu} model={job.model} task={job.task} rc={rc} "
                f"status={record.status} elapsed={record.elapsed_s:.1f}s",
                flush=True,
            )
            if record.status != "ok" and job.attempt < args.max_retries:
                retry = Job(job.model, job.task, job.attempt + 1)
                queue.append(retry)
                print(f"[RETRY] model={retry.model} task={retry.task} attempt={retry.attempt}", flush=True)

        snapshot = gpu_snapshot()
        busy_gpus = set(running)
        for gpu in requested_gpus:
            if gpu in busy_gpus or not queue:
                continue
            used_mib, util = snapshot.get(gpu, (10**9, 100))
            if used_mib > args.idle_memory_mib:
                continue
            next_job = None
            active_external = active_run_baseline_jobs(args.output_dir)
            checked = 0
            while queue and checked < len(queue):
                candidate = queue.pop(0)
                checked += 1
                if job_complete(args.output_dir, candidate, args.max_samples):
                    records.append(
                        JobRecord(
                            model=candidate.model,
                            task=candidate.task,
                            gpu=gpu,
                            attempt=candidate.attempt,
                            status="already_complete",
                            returncode=0,
                            started_at=time.time(),
                            ended_at=time.time(),
                            elapsed_s=0.0,
                            log="",
                        )
                    )
                    continue
                if (candidate.model, candidate.task) in active_external:
                    queue.append(candidate)
                    continue
                next_job = candidate
                break
            if next_job is None:
                continue
            proc, record = launch_job(args, next_job, gpu, log_dir)
            running[gpu] = (next_job, proc, record)
            print(
                f"[START] gpu={gpu} model={next_job.model} task={next_job.task} "
                f"attempt={next_job.attempt} pid={proc.pid} log={record.log}",
                flush=True,
            )

        write_state(state_path, queue, running, records)
        if queue or running:
            print(
                f"[HEARTBEAT] queued={len(queue)} running={len(running)} "
                f"records={len(records)} snapshot={snapshot}",
                flush=True,
            )
            time.sleep(args.poll_interval)

    failed = [r for r in records if r.status == "failed"]
    write_state(state_path, queue, running, records)
    print(f"[SUMMARY] total_records={len(records)} failed={len(failed)}", flush=True)
    if failed:
        print(json.dumps([asdict(r) for r in failed], ensure_ascii=False, indent=2), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
