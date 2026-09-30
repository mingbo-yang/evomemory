#!/usr/bin/env python
"""No-idle GPU dispatcher for the aaai2027 experiment queue.

What it does
------------
Polls the four experiment cards (0-3) every 30 s.  A card is considered BUSY
if some live process's ``/proc/<pid>/cmdline`` names ``supervisor.py`` or
``run_experiment.py`` together with ``--gpu <N>``.  Whenever a card is FREE and
a pending job exists, the highest-priority pending job is handed to a brand new
detached ``supervisor.py`` on that card, so no card sits idle between jobs.

SAFETY CONTRACT -- do not weaken this file
------------------------------------------
1. This program NEVER signals an existing process.  There is no ``os.kill``,
   no ``signal`` import, no ``pkill``/``pgrep``, and no
   ``Popen.terminate()/kill()`` anywhere in this file, and none may ever be
   added.  Other people's jobs on this shared box are only *observed*, through
   ``/proc``.  Even a job the dispatcher itself started is never killed by the
   dispatcher: if it must be stopped, a human does it by exact PID.
2. The only way a card becomes "ours" is by launching a *new* detached
   process: ``subprocess.Popen(..., start_new_session=True)`` (== setsid), so
   it survives a dispatcher restart or a terminal closing.
3. Busy detection never matches the dispatcher itself: our own pid and any
   cmdline mentioning ``dispatcher.py`` are skipped.  Patterns are never
   passed to ``pkill -f``/``pgrep -f``.
4. Before every launch the target card is re-scanned; a card that is busy is
   never used, and a card is used at most once per tick.

Queue file (``configs/queue.json``, created only if absent)
----------------------------------------------------------
    {
      "jobs": [
        {"id": "main-gpu0", "gpu": 0, "arm": "full_static",
         "models": "qwen3-8b,qwen3-4b", "out_dir": "runs/main",
         "attempts_left": 2, "status": "pending",
         "pid": null, "started_at": null}
      ]
    }

* ``gpu: null``  -> any free card.  An explicit ``gpu`` is a *preference*:
  it wins when that card is free, otherwise the job may be placed elsewhere
  (logged as ``pin-override``).  Add ``"pin_hard": true`` to forbid that.
* optional per-job overrides: ``seed`` (42), ``alpha`` (0.5), ``tasks``
  ("all"), ``batch_size`` (32), ``stuck_min`` (12), ``max_attempts`` (4),
  ``split`` ("test"), ``pin_hard`` (false).
* ``status`` is maintained by the dispatcher: pending|running|done|failed.
  ``attempts_left`` counts *retries after the first try*; each failure with
  ``attempts_left > 0`` decrements it, sets the job back to ``pending`` and
  records ``not_before`` = now + 60 s (backoff).
* To force a retry by hand, set ``status`` back to ``pending`` for a
  ``failed`` job; the next tick picks that up.

Completion rule (mirrors ``run_experiment.py`` output layout)
-------------------------------------------------------------
``<out_dir>/<arm>/seed<seed>/<task>/<model>/<arm>.jsonl`` must hold at least
``n_test`` lines (read from ``data/manifests/<task>__test.jsonl``, fallback
1000, gigaword 100) AND a sibling ``<arm>.summary.json`` must exist and be
non-empty.  All (task, model) pairs of the job must satisfy that.  For
``split != "test"`` the split component is inserted, exactly as
``run_experiment.py`` does.

Usage
-----
    # one read-only decision tick
    python dispatcher.py --dry-run --once
    # the real thing (detached, survives this shell)
    nohup setsid python dispatcher.py --interval 30 >> logs/dispatcher_console.log 2>&1 &
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
PROC = Path("/proc")

# Interpreter and launcher used for every job (production values).
PY = "/home/ymb/miniconda3/envs/qwen35/bin/python"
SUPERVISOR = ROOT / "supervisor.py"

DEFAULT_QUEUE = ROOT / "configs" / "queue.json"
DEFAULT_LOG = ROOT / "logs" / "dispatcher.log"
LOCK_PATH = ROOT / "logs" / "dispatcher.lock"

DEFAULT_CARDS: Tuple[int, ...] = (0, 1, 2, 3)
DEFAULT_INTERVAL = 30.0
RETRY_BACKOFF_S = 60

# A cmdline is only interesting if it names one of these programs.
JOB_MARKERS = ("supervisor.py", "run_experiment.py")
# Never count the dispatcher (or a sibling dispatcher) as a GPU job.
SELF_MARKER = "dispatcher.py"

TASKS: Tuple[str, ...] = ("wmt19_en_zh", "wmt19_zh_en", "coedit_gec", "gigaword")
FALLBACK_TASK_N = {"wmt19_en_zh": 1000, "wmt19_zh_en": 1000,
                   "coedit_gec": 1000, "gigaword": 100}
DEFAULT_TASK_N = 1000

VALID_STATUS = ("pending", "running", "done", "failed")

QUEUE_TEMPLATE: Dict[str, Any] = {
    "_readme": [
        "Queue for dispatcher.py.  'jobs' is EMPTY on purpose: add real jobs here.",
        "status is maintained by the dispatcher (pending|running|done|failed).",
        "gpu:null means 'any free card'; an explicit gpu is a preference, add "
        "\"pin_hard\": true to make it a hard pin.",
        "attempts_left counts retries after the first try (0 = no retry).",
        "Optional per-job overrides: seed=42, alpha=0.5, tasks='all', "
        "batch_size=32, stuck_min=12, max_attempts=4, split='test'.",
    ],
    "_example_job": {
        "id": "EXAMPLE-do-not-run",
        "gpu": 0,
        "arm": "full_static",
        "models": "qwen3-8b,qwen3-4b",
        "out_dir": "runs/main",
        "attempts_left": 2,
        "status": "pending",
        "pid": None,
        "started_at": None,
    },
    "jobs": [],
}


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
class Logger:
    """Timestamped append-only log, mirrored to stdout so ``tee`` works.

    The file is opened/closed per line: cheap at this rate, survives a restart,
    and never holds a stale file handle across a rotation.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def set_path(self, path: Path) -> None:
        self.path = Path(path)

    def log(self, msg: str) -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:  # never let logging kill the dispatcher
            print(f"dispatcher: cannot append to {self.path}: {exc}",
                  file=sys.stderr, flush=True)
        print(line, flush=True)


LOG = Logger(DEFAULT_LOG)


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def parse_ts(value: Any) -> Optional[datetime]:
    """Accept ISO strings (with or without space/T) and epoch numbers."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip().replace("Z", "")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------- #
# /proc observation (read-only: this is the ONLY way we look at other jobs)
# --------------------------------------------------------------------------- #
def read_cmdline(pid: int) -> Optional[List[str]]:
    """argv of pid, or None when the pid is gone/zombie/has no cmdline."""
    try:
        data = (PROC / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    if not data:
        return None  # kernel thread, or a zombie: nothing is running there
    return [part.decode("utf-8", "replace") for part in data.split(b"\0") if part]


def proc_state(pid: int) -> str:
    try:
        stat = (PROC / str(pid) / "stat").read_text(errors="replace")
    except OSError:
        return ""
    # comm may contain spaces/parentheses: cut at the LAST ')'.
    tail = stat.rsplit(")", 1)[-1].split()
    return tail[0] if tail else ""


def iter_pids() -> List[int]:
    try:
        entries = os.listdir(PROC)
    except OSError:
        return []
    return [int(e) for e in entries if e.isdigit()]


def argv_gpus(argv: Sequence[str], joined: str) -> set:
    """Every card number mentioned as ``--gpu <N>`` / ``--gpu=<N>``.

    Handles both a normal argv and a wrapper whose single argument embeds the
    whole command line (``bash -c "python supervisor.py --gpu 0 ..."``).
    """
    raw = set()
    for i, arg in enumerate(argv):
        if arg == "--gpu" and i + 1 < len(argv):
            raw.add(argv[i + 1])
        elif arg.startswith("--gpu="):
            raw.add(arg.split("=", 1)[1])
    for match in re.finditer(r"--gpu[=\s]+([^\s\"']+)", joined):
        raw.add(match.group(1))
    gpus = set()
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                gpus.add(part)
    return gpus


def arg_value(joined: str, name: str) -> Optional[str]:
    """Value of ``name`` in a flattened cmdline (``--name v`` / ``--name=v``)."""
    match = re.search(re.escape(name) + r"[=\s]+([^\s\"']+)", joined)
    return match.group(1) if match else None


def scan_processes(cards: Sequence[int]) -> Tuple[Dict[int, str], set]:
    """One read-only /proc pass.

    Returns ``(busy, active)``:
      * ``busy``   card -> short reason, for every card currently occupied.
      * ``active`` signatures ``(arm, out_dir, model, split)`` of job work that
        some live process is already doing -- used so the dispatcher never
        starts a second supervisor for the same (arm, model, out_dir) and
        corrupts the same output files.  Signatures are collected for every
        matching process, whatever card it is on.
    """
    busy: Dict[int, str] = {}
    active: set = set()
    self_pid = os.getpid()
    wanted = {str(c) for c in cards}
    for pid in iter_pids():
        if pid == self_pid:
            continue
        argv = read_cmdline(pid)
        if not argv:
            continue
        joined = " ".join(argv)
        if SELF_MARKER in joined:
            continue  # never mistake the dispatcher for a job
        if not any(marker in joined for marker in JOB_MARKERS):
            continue
        if proc_state(pid) in ("Z", "X"):
            continue
        marker = "supervisor.py" if "supervisor.py" in joined else "run_experiment.py"
        arm = arg_value(joined, "--arm")
        out_dir = arg_value(joined, "--out-dir")
        models = arg_value(joined, "--models")
        if arm and out_dir:
            split = arg_value(joined, "--split") or "test"
            out_key = _norm_path(out_dir)
            if models:
                for model in str(models).split(","):
                    model = model.strip()
                    if model:
                        active.add((arm, out_key, model, split))
            else:
                active.add((arm, out_key, "*", split))
        gpus = argv_gpus(argv, joined)
        if not gpus:
            continue
        for card in sorted(wanted & gpus):
            card_i = int(card)
            if card_i not in busy:
                busy[card_i] = f"pid={pid} {marker} --gpu {card_i}"
    return busy, active


def scan_cards(cards: Sequence[int]) -> Dict[int, str]:
    """card -> short reason string, for every card that is currently BUSY."""
    return scan_processes(cards)[0]


#: Room a fresh vLLM load needs on top of its configured fraction, for the CUDA
#: context and the allocator's own overhead.
LOAD_MARGIN_MIB = 1024

#: Fallback only -- see :func:`gpu_memory_fractions`.
_FALLBACK_FRACTIONS = {"glm4-9b": 0.45, "llama3.1-8b": 0.40,
                       "qwen3-4b": 0.30, "qwen3-8b": 0.40, "qwen3-32b": 0.86}


def gpu_memory_fractions() -> Dict[str, float]:
    """``core.GPU_MEMORY_UTILIZATION`` without importing ``core``.

    The dispatcher deliberately never imports the experiment package: that pulls
    in torch and vLLM, which a scheduler has no business loading.  The table is
    therefore read out of ``core/__init__.py`` with ``ast`` so there is still
    exactly ONE source of truth, and it cannot silently drift from what
    ``run_experiment.py`` actually passes to vLLM.
    """
    try:
        tree = ast.parse((ROOT / "core" / "__init__.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "GPU_MEMORY_UTILIZATION" in targets:
                value = ast.literal_eval(node.value)
                if isinstance(value, dict) and value:
                    return {str(k): float(v) for k, v in value.items()}
    except Exception:
        pass
    return dict(_FALLBACK_FRACTIONS)


def gpu_memory_state() -> Dict[int, Tuple[int, int]]:
    """card -> (free MiB, total MiB).  Empty dict if nvidia-smi is unusable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return {}
    state: Dict[int, Tuple[int, int]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            state[int(parts[0])] = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    return state


def memory_shortfall(job: Dict[str, Any], free_mib: int, total_mib: int) -> Optional[str]:
    """Why this job cannot load on a card with ``free_mib`` free, or None.

    Card choice used to be purely liveness-based: the dispatcher saw a card with
    no *job* on it and launched, even when another user's process had taken the
    memory.  Every such launch then died ~60 s later with ``ValueError: Free
    memory on device ... less than desired GPU memory utilization``, burning a
    retry and leaving a dead engine behind.  Requiring the footprint up front
    turns a guaranteed crash into a short wait.
    """
    fractions = gpu_memory_fractions()
    for model in job_models(job):
        frac = fractions.get(model)
        if frac is None:
            continue
        need = int(frac * total_mib) + LOAD_MARGIN_MIB
        if free_mib < need:
            return f"{model} needs {need} MiB free, card has {free_mib}"
    return None


def recorded_pid_is_job(job: Dict[str, Any]) -> bool:
    """Restart safety: is this job's recorded pid still OUR supervisor?

    Checks the live cmdline, not just pid existence, so a recycled pid is not
    mistaken for a running job.  Read-only, never signals anything.
    """
    pid = job.get("pid")
    if not pid:
        return False
    argv = read_cmdline(int(pid))
    if not argv:
        return False
    joined = " ".join(argv)
    if "supervisor.py" not in joined:
        return False
    for flag, field in (("--out-dir", "out_dir"), ("--arm", "arm"),
                        ("--models", "models")):
        got = arg_value(joined, flag)
        want = job.get(field)
        if got is None or want in (None, ""):
            continue
        if not _same_value(str(got), str(want)):
            return False
    return True


def _same_value(got: str, want: str) -> bool:
    g, w = got.rstrip("/"), want.rstrip("/")
    if g == w:
        return True
    # relative vs absolute spelling of the same directory
    return g.endswith("/" + w) or w.endswith("/" + g)


def _norm_path(value: str) -> str:
    return os.path.normpath(str(value).strip()).replace("\\", "/")


def job_conflict(job: Dict[str, Any], active: set) -> Optional[str]:
    """Reason string when a live process is already computing this job."""
    arm = str(job.get("arm"))
    out_dir = _norm_path(job.get("out_dir") or "runs/main")
    split = str(job.get("split") or "test")
    models = set(job_models(job))
    for a, o, m, s in active:
        if a != arm or s != split or not _same_value(o, out_dir):
            continue
        if m == "*" or m in models:
            return f"live process is already running arm={a} model={m} out_dir={o}"
    return None


# --------------------------------------------------------------------------- #
# output completeness
# --------------------------------------------------------------------------- #
_MANIFEST_N: Dict[str, int] = {}


def required_n(task: str, split: str = "test") -> int:
    """Expected number of samples for a task ON THIS SPLIT.

    This used to hardcode the test manifest, so a *successful* dev run (32 rows)
    was read as "32/1000, incomplete" and re-dispatched -- each retry burning a
    full model load and eventually marking a succeeded job failed.  The split is
    part of the cache key too, or the first split looked up would win for both.
    """
    key = f"{split}/{task}"
    if key in _MANIFEST_N:
        return _MANIFEST_N[key]
    path = ROOT / "data" / "manifests" / f"{task}__{split}.jsonl"
    n: Optional[int] = None
    try:
        if path.is_file():
            n = sum(1 for _ in path.open("rb"))
    except OSError as exc:
        LOG.log(f"[warn] cannot read manifest {path}: {exc}")
    if not n or n <= 0:
        n = FALLBACK_TASK_N.get(task, DEFAULT_TASK_N)
        LOG.log(f"[warn] no usable manifest for task={task} split={split}; fallback n={n}")
    _MANIFEST_N[key] = n
    return n


def count_lines_capped(path: Path, cap: int) -> Tuple[int, bool]:
    """Line count, stopping early once ``cap`` lines are seen."""
    n = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                return n, False
            n += chunk.count(b"\n")
            if n >= cap:
                return n, True


def job_models(job: Dict[str, Any]) -> List[str]:
    models = job.get("models") or ""
    if isinstance(models, (list, tuple)):
        items = [str(m) for m in models]
    else:
        items = str(models).split(",")
    return [m.strip() for m in items if m.strip()]


def job_tasks(job: Dict[str, Any]) -> List[str]:
    tasks = job.get("tasks") or "all"
    if isinstance(tasks, (list, tuple)):
        items = [str(t) for t in tasks]
    else:
        items = str(tasks).split(",")
    items = [t.strip() for t in items if t.strip()]
    if not items or items == ["all"]:
        return list(TASKS)
    return items


def pair_paths(job: Dict[str, Any], task: str, model: str) -> Tuple[Path, Path]:
    out_dir = Path(str(job.get("out_dir") or "runs/main"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    run_dir = out_dir / str(job["arm"]) / f"seed{job.get('seed', 42)}" / task / model
    split = str(job.get("split") or "test")
    if split != "test":
        run_dir = run_dir / split
    if job.get("tag"):
        run_dir = run_dir / str(job["tag"])
    # Same with_suffix() logic as run_experiment.py, so paths match exactly.
    stem = run_dir / str(job["arm"])
    return stem.with_suffix(".jsonl"), stem.with_suffix(".summary.json")


def pair_problem(job: Dict[str, Any], task: str, model: str, need: int) -> Optional[str]:
    """None when this (task, model) pair is complete, else a short reason."""
    jsonl, summary = pair_paths(job, task, model)
    if not jsonl.is_file():
        return "no jsonl"
    try:
        n, capped = count_lines_capped(jsonl, need)
    except OSError as exc:
        return f"jsonl unreadable ({exc.__class__.__name__})"
    if n < need:
        return f"jsonl {n}/{need}"
    if not summary.is_file():
        return "no summary.json"
    try:
        if summary.stat().st_size == 0:
            return "empty summary.json"
    except OSError:
        return "summary.json unreadable"
    return None


def job_status_of_outputs(job: Dict[str, Any]) -> Tuple[bool, str]:
    expect = job.get("expect_files")
    if expect:
        missing = []
        for rel in expect:
            f = ROOT / str(rel)
            if not f.is_file() or f.stat().st_size == 0:
                missing.append(str(rel))
        if not missing:
            return True, f"{len(expect)}/{len(expect)} expected files present"
        return False, f"{len(expect) - len(missing)}/{len(expect)} expected files; missing {missing[:3]}"
    models = job_models(job)
    if not models:
        return False, "no models listed"
    total, ok, problems = 0, 0, []
    for task in job_tasks(job):
        need = required_n(task, str(job.get("split") or "test"))
        for model in models:
            total += 1
            problem = pair_problem(job, task, model, need)
            if problem is None:
                ok += 1
            else:
                problems.append(f"{task}/{model}: {problem}")
    if ok == total:
        return True, f"{ok}/{total} output pairs complete"
    shown = "; ".join(problems[:6])
    if len(problems) > 6:
        shown += f"; (+{len(problems) - 6} more)"
    return False, f"{ok}/{total} output pairs complete; {shown}"


# --------------------------------------------------------------------------- #
# queue file
# --------------------------------------------------------------------------- #
def normalize_job(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Fill in defaults; unknown keys are preserved verbatim."""
    job = dict(raw)
    job["id"] = str(job.get("id") or f"job{index}")
    gpu = job.get("gpu", None)
    job["gpu"] = None if gpu in (None, "", "null", "any") else int(gpu)
    job["arm"] = str(job.get("arm") or "full_static")
    models = job.get("models") or ""
    if isinstance(models, (list, tuple)):
        models = ",".join(str(m) for m in models)
    job["models"] = str(models)
    job["out_dir"] = str(job.get("out_dir") or "runs/main")
    try:
        job["attempts_left"] = max(0, int(job.get("attempts_left", 0)))
    except (TypeError, ValueError):
        job["attempts_left"] = 0
    status = str(job.get("status") or "pending").lower()
    job["status"] = status if status in VALID_STATUS else "pending"
    pid = job.get("pid", None)
    job["pid"] = None if pid in (None, "", 0) else int(pid)
    job["started_at"] = job.get("started_at") or None
    job["seed"] = int(job.get("seed", 42))
    job["alpha"] = float(job.get("alpha", 0.5))
    job["batch_size"] = int(job.get("batch_size", 32))
    job["stuck_min"] = int(job.get("stuck_min", 12))
    job["max_attempts"] = int(job.get("max_attempts", 4))
    job["tasks"] = job.get("tasks") or "all"
    job["split"] = str(job.get("split") or "test")
    job["pin_hard"] = bool(job.get("pin_hard", False))
    job["not_before"] = job.get("not_before") or None
    return job


def compose(raw: Any, jobs: List[Dict[str, Any]]) -> Any:
    """Keep the on-disk top-level shape (object with 'jobs', or a bare list)."""
    if isinstance(raw, list):
        return jobs
    obj = dict(raw) if isinstance(raw, dict) else {}
    obj["jobs"] = jobs
    return obj


def load_queue(path: Path, create: bool) -> Tuple[Any, List[Dict[str, Any]]]:
    """Read the queue; create it from the template only when it is missing."""
    if not path.exists():
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = json.loads(json.dumps(QUEUE_TEMPLATE))
            save_queue(path, raw, [])
            LOG.log(f"[queue] created empty queue template at {path}")
            return raw, []
        LOG.log(f"[warn] queue {path} does not exist (dry-run: not creating it)")
        return json.loads(json.dumps(QUEUE_TEMPLATE)), []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Refuse to guess: never clobber a file we could not parse.
        raise SystemExit(f"dispatcher: cannot parse queue {path}: {exc}")
    if isinstance(raw, dict):
        items = raw.get("jobs") or []
    elif isinstance(raw, list):
        items = raw
    else:
        raise SystemExit(f"dispatcher: queue {path} must be a list or an object")
    jobs: List[Dict[str, Any]] = []
    seen = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            LOG.log(f"[warn] queue entry #{index} is not an object; skipped")
            continue
        job = normalize_job(item, index)
        if job["id"] in seen:
            LOG.log(f"[warn] duplicate job id={job['id']}; entry #{index} skipped")
            continue
        seen.add(job["id"])
        jobs.append(job)
    return raw, jobs


def save_queue(path: Path, raw: Any, jobs: List[Dict[str, Any]]) -> None:
    """Atomic write: temp file in the same directory, fsync, os.replace."""
    payload = json.dumps(compose(raw, jobs), indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def merge_external_edits(disk_jobs: List[Dict[str, Any]],
                         prev_jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adopt operator edits without losing the state of running jobs.

    * unknown id on disk            -> new work, adopted as-is
    * known id, disk == our state   -> our state
    * known id, edited while not running -> operator wins (e.g. failed->pending)
    * known id, edited while running     -> our bookkeeping wins (logged)
    * id we know but that vanished       -> dropped unless it is running
    """
    prev = {job["id"]: job for job in prev_jobs}
    out: List[Dict[str, Any]] = []
    for disk_job in disk_jobs:
        old = prev.pop(disk_job["id"], None)
        if old is None:
            LOG.log(f"[queue] new job id={disk_job['id']} status={disk_job['status']} "
                    f"gpu={disk_job['gpu']} arm={disk_job['arm']}")
            out.append(disk_job)
        elif old == disk_job:
            out.append(disk_job)
        elif old.get("status") == "running":
            merged = dict(disk_job)
            for key in ("status", "pid", "started_at", "attempts_left",
                        "not_before", "gpu_used"):
                merged[key] = old.get(key)
            LOG.log(f"[queue] external edit of RUNNING job id={disk_job['id']} "
                    f"ignored (dispatcher owns its bookkeeping)")
            out.append(merged)
        else:
            LOG.log(f"[queue] operator edit adopted for id={disk_job['id']}: "
                    f"{old.get('status')} -> {disk_job['status']}")
            out.append(disk_job)
    for old in prev.values():
        if old.get("status") == "running":
            LOG.log(f"[warn] running job id={old['id']} was removed from the queue "
                    f"file; still observing pid={old.get('pid')}, re-adding it")
            out.append(old)
        else:
            LOG.log(f"[queue] job id={old['id']} removed from the queue file")
    return out


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def eligible(job: Dict[str, Any], now: datetime) -> bool:
    if job.get("status") != "pending":
        return False
    not_before = parse_ts(job.get("not_before"))
    return not_before is None or not_before <= now


def choose_job(jobs: Sequence[Dict[str, Any]], card: int, now: datetime,
               blocked: Optional[set] = None) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Pick the job for one free card.  Returns (job, pin_overridden).

    ``blocked`` holds job ids that must not be dispatched in this tick
    (jobs just finished/retried, and ids already claimed by another card in
    the same tick) -- this is what keeps one job from being started twice
    when several cards are free.

    Priority: (1) pins pointing at this card, (2) ``gpu: null`` jobs,
    (3) jobs pinned to a card that is not this one -- still better than leaving
    the card idle, which is what this dispatcher exists to prevent.  Ties are
    broken by queue order.  ``pin_hard`` jobs are never placed on another card.
    """
    blocked = blocked or set()
    pending = [j for j in jobs if eligible(j, now) and j["id"] not in blocked]
    # Honour an explicit `priority` (lower = sooner).  The field existed in the
    # queue but was never read, so dispatch order silently depended on the file
    # having been written pre-sorted -- a trap for anyone who reorders it.
    # Python's sort is stable, so the queue's own order still breaks ties.
    pending.sort(key=lambda j: j.get("priority", 5))
    for job in pending:
        if job.get("gpu") == card:
            return job, False
    for job in pending:
        if job.get("gpu") is None and not job.get("pin_hard"):
            return job, False
    for job in pending:
        if job.get("gpu") is not None and job.get("gpu") != card and not job.get("pin_hard"):
            return job, True
    return None, False


def build_cmd(job: Dict[str, Any], card: int) -> List[str]:
    # A job may carry an explicit command instead of the supervisor template
    # (the re-judge experiment, Phase 7 counterfactual, Phase 8 accumulation).
    # `{gpu}` is substituted with the assigned card.
    if job.get("cmd"):
        return [str(x).replace("{gpu}", str(card)) for x in job["cmd"]]
    cmd = [PY, str(SUPERVISOR), "--gpu", str(card),
           "--models", ",".join(job_models(job)),
           "--arm", str(job["arm"]),
           "--out-dir", str(job["out_dir"]),
           "--batch-size", str(job.get("batch_size", 32)),
           "--stuck-min", str(job.get("stuck_min", 12)),
           "--max-attempts", str(job.get("max_attempts", 4)),
           "--seed", str(job.get("seed", 42)),
           "--alpha", str(job.get("alpha", 0.5))]
    tasks = job.get("tasks") or "all"
    if isinstance(tasks, (list, tuple)):
        tasks = ",".join(str(t) for t in tasks)
    if str(tasks) != "all":
        cmd += ["--tasks", str(tasks)]
    # Dev-split jobs exist to re-create evidence with no surviving artefact.
    # run_experiment.py refuses --draft-source stored off the test split (the
    # stored drafts are positional and belong to the test prefix), so a dev run
    # must generate its own drafts.
    if job.get("tag"):
        cmd += ["--tag", str(job["tag"])]
    if int(job.get("n_candidates") or 1) > 1:
        cmd += ["--n-candidates", str(int(job["n_candidates"]))]
    if job.get("experience_root"):
        cmd += ["--experience-root", str(job["experience_root"])]
    if job.get("accept_max_len_ratio") is not None:
        cmd += ["--accept-max-len-ratio", str(job["accept_max_len_ratio"])]
    if job.get("renderer"):
        cmd += ["--renderer", str(job["renderer"])]
    if job.get("controller_sees_examples"):
        cmd += ["--controller-sees-examples"]
    if job.get("advice_mode"):
        cmd += ["--advice-mode", str(job["advice_mode"])]
    if job.get("retrieval_excludes_own_source"):
        cmd += ["--retrieval-excludes-own-source"]
    if job.get("draft_cache"):
        # Pins y0 for a dev/aux job.  Without it those splits regenerate their
        # drafts once per process, so two runs of the same cell do not share a
        # starting point and every paired comparison between them is confounded
        # (quantified in scores/y0_identity.csv).
        cmd += ["--draft-cache", str(job["draft_cache"])]
    split = str(job.get("split") or "test")
    if split != "test":
        cmd += ["--split", split,
                "--draft-source", str(job.get("draft_source") or "generate")]
    return cmd


def launch_job(job: Dict[str, Any], card: int) -> Tuple[int, List[str], Path]:
    """Start a NEW detached supervisor.  Never touches an existing process."""
    cmd = build_cmd(job, card)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(job["id"]))[:60]
    log_path = ROOT / "logs" / f"dispatcher_job_{safe_id}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(card))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as fh:
        fh.write(f"\n===== dispatcher launch {now_iso()} job={job['id']} "
                 f"gpu={card} =====\n".encode("utf-8"))
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # setsid: survives the dispatcher
            close_fds=True,
        )
    return proc.pid, cmd, log_path


def preflight() -> List[str]:
    problems = []
    if not Path(PY).is_file():
        problems.append(f"interpreter missing: {PY}")
    if not SUPERVISOR.is_file():
        problems.append(f"launcher missing: {SUPERVISOR}")
    return problems


def rel_display(path: Any) -> str:
    """Path relative to the experiment root when possible, else as given."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# one tick
# --------------------------------------------------------------------------- #
_LAST_WRITTEN: List[Dict[str, Any]] = []


def tick(cards: Sequence[int], queue_path: Path, dry_run: bool = False,
         explain: bool = False) -> Dict[str, int]:
    """One poll: observe -> completion/retry -> dispatch -> persist -> heartbeat."""
    global _LAST_WRITTEN
    now = datetime.now()
    raw, disk_jobs = load_queue(queue_path, create=not dry_run)
    jobs = disk_jobs if dry_run else merge_external_edits(disk_jobs, _LAST_WRITTEN)
    dirty = False

    # ---- 1. state machine: completion, death, retry ----------------------- #
    # job id -> (pid still alive?, pid as recorded when we looked)
    running_alive: Dict[str, Tuple[bool, Optional[int]]] = {}
    # Job ids that must NOT be dispatched during this tick: everything that
    # finished, failed or was requeued now, plus ids claimed by a card below.
    blocked: set = set()
    for job in jobs:
        status = job.get("status")
        if status not in ("running", "pending"):
            continue
        complete, detail = job_status_of_outputs(job)
        alive = recorded_pid_is_job(job) if status == "running" else False
        if status == "running":
            running_alive[job["id"]] = (alive, job.get("pid"))
        if complete:
            blocked.add(job["id"])
            if status == "pending":
                LOG.log(f"[done] job={job['id']} outputs already complete "
                        f"({detail}); not launching it")
            else:
                LOG.log(f"[done] job={job['id']} pid={job.get('pid')} ({detail})")
            if dry_run:
                LOG.log(f"[dry] would mark job={job['id']} done")
            else:
                job.update(status="done", pid=None, finished_at=now_iso())
                dirty = True
        elif status == "running" and alive:
            if explain:
                LOG.log(f"[explain] job={job['id']} running pid={job['pid']} "
                        f"gpu={job.get('gpu_used')} :: {detail}")
        elif status == "running":
            blocked.add(job["id"])
            pid = job.get("pid")
            if job.get("attempts_left", 0) > 0:
                left = int(job["attempts_left"]) - 1
                not_before = (now + timedelta(seconds=RETRY_BACKOFF_S)).replace(
                    microsecond=0).isoformat(sep=" ")
                LOG.log(f"[retry] job={job['id']} pid={pid} is gone and incomplete "
                        f"({detail}); attempts_left {job['attempts_left']} -> {left}, "
                        f"backoff {RETRY_BACKOFF_S}s until {not_before}")
                if dry_run:
                    LOG.log(f"[dry] would requeue job={job['id']}")
                else:
                    job.update(status="pending", pid=None, attempts_left=left,
                               not_before=not_before, last_error=detail,
                               gpu_used=None)
                    dirty = True
            else:
                LOG.log(f"[fail] job={job['id']} pid={pid} exited incomplete "
                        f"({detail}); attempts_left=0 -> failed")
                if dry_run:
                    LOG.log(f"[dry] would mark job={job['id']} failed")
                else:
                    job.update(status="failed", pid=None, finished_at=now_iso(),
                               last_error=detail)
                    dirty = True
        elif explain:
            LOG.log(f"[explain] job={job['id']} waiting on backoff/turn :: {detail}")

    # ---- 2. observe the cards -------------------------------------------- #
    busy, active = scan_processes(cards)
    # Defence in depth: a card that one of OUR running jobs still holds is
    # busy even if a scan missed the process (restart safety, requirement 7).
    for job in jobs:
        held = running_alive.get(job["id"])
        if not held or not held[0]:
            continue
        card = job.get("gpu_used")
        if card is None or card not in cards or card in busy:
            continue
        busy[int(card)] = (f"own job={job['id']} pid={held[1]} "
                           f"(recorded pid alive)")
    free = [c for c in cards if c not in busy]

    # Never start a second supervisor for work that a live process (ours, the
    # operator's, or another driver's) is already doing: both would write the
    # same <arm>.jsonl and corrupt the result.  Those jobs stay pending and are
    # picked up once the other run finishes.
    conflicts = {}
    if active:
        for job in jobs:
            if job.get("status") != "pending" or job["id"] in blocked:
                continue
            if not eligible(job, now):
                continue
            reason = job_conflict(job, active)
            if reason:
                conflicts[job["id"]] = reason
    if conflicts:
        blocked.update(conflicts)
        LOG.log("[skip] already being computed by a live process (not "
                "duplicating): " + "; ".join(
                    f"{jid} <- {why}" for jid, why in conflicts.items()))

    dispatched: List[str] = []
    reserved: Dict[int, str] = {}  # card -> label for the post-tick map
    would_run: set = set()  # dry-run only: jobs that would have been launched

    # ---- 3. dispatch: at most one job per free card ---------------------- #
    for card in list(free):
        job, overridden = choose_job(jobs, card, now, blocked)
        if job is None:
            continue
        free.remove(card)  # this card is spent for the rest of the tick
        blocked.add(job["id"])  # and this job is spent too: never start it twice
        if recorded_pid_is_job(job):
            LOG.log(f"[skip] job={job['id']} still has live pid={job.get('pid')} "
                    f"from an earlier launch; not starting a duplicate")
            continue
        if dry_run:
            LOG.log(f"[dry] would dispatch job={job['id']} gpu={card} "
                    f"arm={job['arm']} models={','.join(job_models(job))} "
                    f"out_dir={job['out_dir']} cmd={' '.join(build_cmd(job, card))}")
            dispatched.append(f"{job['id']}->gpu{card}(dry)")
            reserved[card] = f"reserved(job={job['id']}, dry-run)"
            would_run.add(job["id"])
            continue
        fresh, fresh_active = scan_processes([card])
        if card in fresh:
            LOG.log(f"[skip] card {card} became busy ({fresh[card]}) before launch; "
                    f"job={job['id']} stays pending")
            continue
        reason = job_conflict(job, fresh_active)
        if reason:
            LOG.log(f"[skip] job={job['id']} would duplicate live work ({reason}); "
                    f"stays pending")
            continue
        # Memory gate: a free card is not necessarily a card that can host a
        # fresh vLLM load.  This is a *wait*, not a failure -- the job keeps its
        # retries and stays pending until some engine exits and frees VRAM.
        mem = gpu_memory_state().get(card)
        if mem is not None:
            short = memory_shortfall(job, mem[0], mem[1])
            if short:
                LOG.log(f"[skip] job={job['id']} card {card} lacks memory "
                        f"({short}); stays pending")
                free.add(card)  # this card was not spent after all
                blocked.discard(job["id"])
                continue
        try:
            pid, cmd, log_path = launch_job(job, card)
        except Exception as exc:  # a broken launch must not block the other cards
            LOG.log(f"[err] launch failed for job={job['id']} gpu={card}: "
                    f"{exc.__class__.__name__}: {exc}; job stays pending")
            continue
        job.update(status="running", pid=pid, started_at=now_iso(),
                   gpu_used=card, log=rel_display(log_path))
        dirty = True
        pin_note = ""
        if overridden:
            pin_note = f" [pin-override: job asked for gpu {job.get('gpu')}]"
        LOG.log(f"[dispatch] job={job['id']} gpu={card} pid={pid} "
                f"arm={job['arm']} models={','.join(job_models(job))} "
                f"out_dir={job['out_dir']} log={job['log']}{pin_note} "
                f"cmd={' '.join(cmd)}")
        dispatched.append(f"{job['id']}->gpu{card}(pid{pid})")
        reserved[card] = f"pid={pid} job={job['id']} (just launched)"

    # ---- 4. persist ------------------------------------------------------- #
    if dirty and not dry_run:
        save_queue(queue_path, raw, jobs)
    if not dry_run:
        _LAST_WRITTEN = [dict(job) for job in jobs]

    # ---- 5. heartbeat ----------------------------------------------------- #
    counts = {s: 0 for s in VALID_STATUS}
    for job in jobs:
        counts[job.get("status", "pending")] = counts.get(job.get("status"), 0) + 1
    shown = dict(busy)
    shown.update(reserved)
    card_map = " ".join(
        f"{c}={'BUSY(' + shown[c] + ')' if c in shown else 'FREE'}" for c in cards
    )
    free_now = [c for c in cards if c not in shown]
    waiting = [j["id"] for j in jobs
               if j.get("status") == "pending" and j["id"] not in would_run]
    line = (f"[hb] {card_map} | free={','.join(map(str, free_now)) or '-'} | "
            f"jobs: pending={counts.get('pending', 0)} running={counts.get('running', 0)} "
            f"done={counts.get('done', 0)} failed={counts.get('failed', 0)}")
    if dispatched:
        line += f" | dispatched={','.join(dispatched)}"
    if waiting:
        line += f" | waiting={','.join(waiting[:8])}"
    LOG.log(line)
    return counts


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def parse_cards(text: str) -> Tuple[int, ...]:
    cards = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        cards.append(int(part))
    if not cards:
        raise SystemExit("dispatcher: --gpus must list at least one card")
    return tuple(cards)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="No-idle GPU job dispatcher (observes /proc, only launches "
                    "new detached supervisors; never signals any process).")
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and log only: launch nothing, write nothing")
    ap.add_argument("--once", action="store_true", help="run a single tick and exit")
    ap.add_argument("--queue", default=str(DEFAULT_QUEUE),
                    help=f"queue JSON path (default {DEFAULT_QUEUE})")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                    help=f"seconds between ticks (default {DEFAULT_INTERVAL:g})")
    ap.add_argument("--log", default=str(DEFAULT_LOG),
                    help=f"log file (default {DEFAULT_LOG})")
    ap.add_argument("--gpus", default=",".join(str(c) for c in DEFAULT_CARDS),
                    help="card set to dispatch on (default 0,1,2,3)")
    ap.add_argument("--explain", action="store_true",
                    help="also log per-job output progress each tick")
    ap.add_argument("--no-lock", action="store_true",
                    help="do not take the single-instance lock")
    return ap


def acquire_lock(path: Path) -> Optional[Any]:
    """Advisory single-instance lock (released automatically if we die)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
    except OSError as exc:
        LOG.log(f"[warn] cannot open lock {path}: {exc}; continuing unlocked")
        return None
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        LOG.log(f"[warn] another dispatcher already holds {path}; exiting")
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()} {now_iso()}\n")
    handle.flush()
    return handle


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    LOG.set_path(Path(args.log))
    cards = parse_cards(args.gpus)
    queue_path = Path(args.queue)
    if not queue_path.is_absolute():
        queue_path = ROOT / queue_path

    LOG.log(f"[start] dispatcher pid={os.getpid()} cards={','.join(map(str, cards))} "
            f"queue={queue_path} interval={args.interval:g}s dry_run={args.dry_run} "
            f"argv={' '.join(sys.argv)}")

    problems = preflight()
    if problems:
        for problem in problems:
            LOG.log(f"[warn] preflight: {problem}")
        if not args.dry_run:
            LOG.log("[fatal] refusing to dispatch with a broken preflight")
            return 2

    if args.interval <= 0:
        LOG.log("[fatal] --interval must be > 0")
        return 2

    lock = None if args.no_lock else acquire_lock(LOCK_PATH)
    if not args.no_lock and lock is None:
        return 3

    rc = 0
    try:
        while True:
            try:
                tick(cards, queue_path, dry_run=args.dry_run, explain=args.explain)
            except SystemExit:
                raise
            except Exception as exc:  # a bad tick must never kill the loop
                LOG.log(f"[err] tick failed: {exc.__class__.__name__}: {exc}")
                rc = 1
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        LOG.log("[stop] interrupted; children were started detached and keep running")
    finally:
        if lock is not None:
            try:
                lock.close()
            except OSError:
                pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
