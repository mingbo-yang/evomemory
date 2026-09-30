#!/usr/bin/env python
"""Detached per-GPU supervisor: run a model chain with hang detection and no idle gaps.

Why this exists
---------------
Two failures on this shared box made shell drivers inadequate:

1. **Load hangs.**  vLLM engine startup intermittently hangs after the weights are
   resident (main thread in ``D``, no further reads, GPU idle) and the process
   never exits, so a plain retry loop waits forever.
2. **Terminal restarts.**  A shell driver dies with its terminal, leaving the
   python job alive but nothing to start the next one -- the card then sits idle
   and gets taken by another tenant.

The supervisor is launched with ``setsid`` so it is fully detached, watches a
child's own CPU+I/O progress (card utilisation is useless here -- the box is
shared), kills a stalled attempt, and starts the next model the moment the
previous one finishes.

Usage
-----
    setsid nohup python supervisor.py --gpu 0 --models qwen3-8b,llama3.1-8b \
        --logdir logs --stuck-min 12 > logs/supervisor_gpu0.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
PY = "/home/ymb/miniconda3/envs/qwen35/bin/python"


def proc_tree(root: int) -> List[int]:
    out = [root]
    try:
        kids = subprocess.run(
            ["pgrep", "-P", str(root)], capture_output=True, text=True
        ).stdout.split()
        out += [int(k) for k in kids]
    except Exception:
        pass
    return out


def progress_metric(root: int, log: Optional[Path] = None) -> int:
    """A monotone progress signal: process I/O + CPU ticks + log growth.

    CPU ticks alone are NOT sufficient.  A model load was observed livelocked:
    the engine burned CPU continuously while the log and GPU memory stayed frozen
    for 14 minutes, so a CPU-based watchdog never fired.  The log file only grows
    when a shard actually completes, so its size is the real progress signal
    during loading; the per-sample JSONL growth is the signal during sampling.
    """
    total = 0
    if log is not None:
        try:
            total += log.stat().st_size * 1000  # weight log growth heavily
        except Exception:
            pass
        run_dir = ROOT / "runs" / "main" / "full_static"
        try:
            for j in run_dir.rglob("*.jsonl"):
                total += j.stat().st_size
        except Exception:
            pass
    for p in proc_tree(root):
        try:
            with open(f"/proc/{p}/io") as f:
                for line in f:
                    if line.startswith(("read_bytes:", "write_bytes:")):
                        total += int(line.split()[1])
        except Exception:
            pass
        try:
            with open(f"/proc/{p}/stat") as f:
                parts = f.read().split()
            total += int(parts[13]) + int(parts[14])
        except Exception:
            pass
    return total


def kill_tree(pid: int) -> None:
    for p in proc_tree(pid):
        try:
            os.kill(p, signal.SIGKILL)
        except Exception:
            pass


def run_model(gpu: str, model: str, logdir: Path, stuck_min: int, seed: int,
              tasks: str, alpha: float, arm: str = "full_static",
              out_dir: str = "runs/main", batch_size: int = 32,
              split: str = "test", draft_source: str = "stored",
              draft_cache: str = "",
              accept_max_len_ratio: float = None,
              experience_root: str = None, n_candidates: int = 1,
              tag: str = "", renderer: str = "",
              controller_sees_examples: bool = False,
              advice_mode: str = "",
              retrieval_excludes_own_source: bool = False) -> bool:
    # The log name must distinguish the split and the tag: a dev run and a test
    # run of the same arm/model/seed used to write the SAME file and overwrite
    # each other, which is how a dev crash stayed invisible.
    _suffix = "" if split == "test" else f"_{split}"
    log = logdir / f"{arm}_{model}_s{seed}{_suffix}.log"
    cmd = [
        PY, str(ROOT / "run_experiment.py"),
        "--arm", arm, "--gpu", gpu, "--models", model,
        "--tasks", tasks, "--seed", str(seed), "--alpha", str(alpha),
        "--max-rounds", "3", "--draft-source", draft_source,
        "--batch-size", str(batch_size),
        "--out-dir", out_dir,
    ]
    if experience_root:
        cmd += ["--experience-root", str(experience_root)]
    if tag:
        # Without this a job whose output the dispatcher expects under
        # <model>/<tag>/ writes to <model>/ instead, so it is NEVER seen as
        # complete and is relaunched forever (bon_judge N=4 spent a card on this).
        cmd += ["--tag", str(tag)]
    if int(n_candidates) > 1:
        # BoN-J: a PIPELINE argument, never a RunConfig field, so the config
        # hash of every existing arm stays frozen.
        cmd += ["--n-candidates", str(int(n_candidates))]
    if accept_max_len_ratio is not None:
        cmd += ["--accept-max-len-ratio", str(accept_max_len_ratio)]
    if renderer:
        # v2 is the arm default; pass it only when overriding (e.g. resuming v1).
        cmd += ["--renderer", str(renderer)]
    if controller_sees_examples:
        cmd += ["--controller-sees-examples"]
    if advice_mode:
        cmd += ["--advice-mode", str(advice_mode)]
    if retrieval_excludes_own_source:
        cmd += ["--retrieval-excludes-own-source"]
    if draft_cache:
        # A FROZEN draft file.  Off the test split the initial draft is generated,
        # which is not bit-reproducible, so two runs of the same dev cell do not
        # start from the same y0 and every paired comparison between them is
        # confounded (this really happened: see scores/y0_identity.csv).  Passing
        # a cache pins y0 for every arm and every tau.
        cmd += ["--draft-cache", str(draft_cache)]
    if split != "test":
        # DEV EVIDENCE RE-RUN.  run_experiment.py refuses --draft-source stored
        # off the test split (the stored drafts are positional and belong to the
        # test prefix), so a dev run must generate its own drafts.
        cmd += ["--split", split]
    # NOTE: PYTORCH_ALLOC_CONF=expandable_segments:True was removed here.  It was
    # added to work around vLLM's memory profiler assert, but the model-load hangs
    # started at exactly that point: every successful glm4-9b load (reproduction
    # gate, smoke) ran without it, and every hung one ran with it.  The allocator
    # override is the prime suspect, so it is gone.
    # NCCL/CUDA-IPC hardening.  Stalled loads show the engine's main thread in
    # state D with wchan=0 (a stuck driver call) while pt_nccl_* threads sit
    # idle.  This host has five network interfaces (192.168.3.7 plus four
    # container bridges), which is a classic setup for NCCL picking the wrong
    # one; P2P/IB are pointless at world_size=1 anyway.
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        NCCL_P2P_DISABLE="1",
        NCCL_IB_DISABLE="1",
        NCCL_SHM_DISABLE="0",
        VLLM_DISABLE_CUSTOM_ALL_REDUCE="1",
    )
    print(f"[SUP] gpu={gpu} arm={arm} start model={model} at {time.strftime('%H:%M:%S')}", flush=True)
    with log.open("a") as fh:
        fh.write(f"\n===== supervisor start {time.strftime('%F %T')} model={model} =====\n")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(ROOT), env=env, start_new_session=True)
    last, stuck = -1, 0
    while proc.poll() is None:
        time.sleep(60)
        cur = progress_metric(proc.pid, log)
        if cur == last:
            stuck += 1
        else:
            stuck = 0
        last = cur
        if stuck >= stuck_min:
            print(f"[SUP] gpu={gpu} model={model} STALLED {stuck}m -> kill+retry", flush=True)
            fh_note = f"\n!!! supervisor: no progress for {stuck}m, killing\n"
            with log.open("a") as fh:
                fh.write(fh_note)
            kill_tree(proc.pid)
            time.sleep(10)
            return False
    rc = proc.returncode
    print(f"[SUP] gpu={gpu} model={model} exit={rc} at {time.strftime('%H:%M:%S')}", flush=True)
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--stuck-min", type=int, default=12)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--arm", default="full_static")
    ap.add_argument("--out-dir", default="runs/main")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--split", default="test",
                    choices=["test", "dev", "initial", "accumulation"],
                    help="auxiliary splits (dev/initial/accumulation) are not test data")
    ap.add_argument("--draft-source", default="stored",
                    choices=["stored", "generate", "cached"],
                    help="stored drafts are test-only; 'cached' replays a frozen "
                         "--draft-cache file and is the ONLY way to make a dev/aux "
                         "run start from a reproducible y0")
    ap.add_argument("--draft-cache", default="",
                    help="JSONL draft cache; required for --draft-source cached")
    ap.add_argument("--accept-max-len-ratio", type=float, default=None,
                    help="acceptance-gate length guard; default None = the historical gate")
    ap.add_argument("--experience-root", default=None,
                    help="override the experience-library root (rebuilt libraries)")
    ap.add_argument("--n-candidates", type=int, default=1,
                    help="BoN-J best-of-N; >1 requires --arm bon_judge and --batch-size >1")
    ap.add_argument("--tag", default="", help="extra output subdir (must match the queue's tag)")
    ap.add_argument("--controller-sees-examples", action="store_true",
                    help="v2 only: also show the retrieved examples to the controller")
    ap.add_argument("--retrieval-excludes-own-source", action="store_true",
                    help="drop the query item's own experience unit (required for validity)")
    ap.add_argument("--advice-mode", default="", choices=["", "summary", "full"],
                    help="v2 only: advice truncation; empty = the arm's setting")
    ap.add_argument("--renderer", default="", choices=["", "v1", "v2"],
                    help="experience-block renderer; empty = the arm's own setting")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logdir = ROOT / args.logdir
    logdir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    for model in models:
        ok = False
        for attempt in range(1, args.max_attempts + 1):
            # Keyword arguments beyond the positional core.  This used to be a
            # long positional list, and inserting ``draft_cache`` into the
            # signature while appending it to the end of the call silently
            # shifted every later parameter -- ``n_candidates`` received "" and
            # died on int(""), which killed all four priority-0 E2 jobs within
            # 30s of launch.  Keywords make that class of error impossible.
            if run_model(args.gpu, model, logdir, args.stuck_min, args.seed,
                         args.tasks, args.alpha, args.arm, args.out_dir, args.batch_size,
                         args.split, args.draft_source,
                         draft_cache=args.draft_cache,
                         accept_max_len_ratio=args.accept_max_len_ratio,
                         experience_root=args.experience_root,
                         n_candidates=args.n_candidates,
                         tag=args.tag,
                         renderer=args.renderer,
                         controller_sees_examples=args.controller_sees_examples,
                         advice_mode=args.advice_mode,
                         retrieval_excludes_own_source=args.retrieval_excludes_own_source):
                ok = True
                break
            print(f"[SUP] gpu={args.gpu} model={model} attempt {attempt} failed; "
                  f"retrying immediately", flush=True)
            time.sleep(15)  # release the device, then go straight on
        if not ok:
            print(f"[SUP] gpu={args.gpu} model={model} GAVE UP after "
                  f"{args.max_attempts} attempts", flush=True)
    print(f"[SUP] gpu={args.gpu} arm={args.arm} chain complete at {time.strftime('%F %T')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
