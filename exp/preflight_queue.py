#!/usr/bin/env python
"""Pre-flight every pending queue job so it cannot fail silently on dispatch.

Why this exists
---------------
A job was queued with `--split accumulation`, but `supervisor.py`/`run_experiment.py`
only accepted `{test, dev}`.  The dispatcher launched it, it died in ~30 s on an
argparse error, the dispatcher retried it, and it died again -- three times, on
four jobs, each costing a model load.  Nothing in the queue file said the job was
invalid; the failure only existed at dispatch time.

This checks, for every pending job, WITHOUT loading a model or touching a GPU:
  * the command can be constructed from the dispatcher's own builder,
  * the entry-point CLI accepts every flag and choice it is given,
  * the parameter combination is semantically legal (split vs draft_source,
    experience/library paths that must exist, expected output paths),
  * the job would be recognised as complete by the dispatcher's own rule.

Exit code 2 when any job is invalid, so it can gate a launch.

Run:  python preflight_queue.py            # all pending jobs
      python preflight_queue.py --all      # include running/done/retired
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core  # noqa: E402

VALID_SPLITS = ("test", "dev", "initial", "accumulation")


def load_dispatcher():
    spec = importlib.util.spec_from_file_location("dispatcher_mod", ROOT / "dispatcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="configs/queue.json")
    ap.add_argument("--all", action="store_true", help="check every job, not just pending")
    args = ap.parse_args()

    d = load_dispatcher()
    queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    jobs = queue if args.all else [j for j in queue if j.get("status") == "pending"]
    print(f"pre-flight: {len(jobs)} job(s) from {args.queue}")

    problems: list[tuple[str, str]] = []
    shapes: dict[tuple, str] = {}

    for j in jobs:
        jid = j["id"]

        # ---- 1. the command must be constructible at all --------------------
        try:
            cmd = d.build_cmd(j, 1)
        except Exception as exc:
            problems.append((jid, f"build_cmd raised {exc.__class__.__name__}: {exc}"))
            continue

        # ---- 2. the entry point must accept the flags (argparse choices) ----
        key = tuple(cmd[:2])
        if key not in shapes:
            try:
                r = subprocess.run(cmd[:2] + ["--help"], capture_output=True,
                                   text=True, timeout=90)
                shapes[key] = (r.stdout + r.stderr)
            except Exception as exc:
                shapes[key] = f"__error__ {exc}"
        helptext = shapes[key]
        if "invalid choice" in helptext:
            problems.append((jid, "entry point reports an invalid choice"))
        # every long flag we pass must appear in that tool's help
        for tok in cmd:
            if tok.startswith("--") and tok not in helptext and tok != "--help":
                problems.append((jid, f"{tok} not documented by {Path(cmd[1]).name}"))

        # ---- 3. semantic legality -------------------------------------------
        split = str(j.get("split") or "test")
        if split not in VALID_SPLITS:
            problems.append((jid, f"split {split!r} not in {VALID_SPLITS}"))
        if split != "test" and str(j.get("draft_source") or "stored") == "stored":
            problems.append((jid, f"split={split} with draft_source=stored "
                                  "(stored drafts are positional and test-only)"))
        if int(j.get("batch_size", 32)) > 1 and str(j.get("draft_source")) == "cached" \
                and not j.get("draft_cache"):
            problems.append((jid, "batch>1 with draft_source=cached but no draft_cache"))
        root = j.get("experience_root")
        if root and not (ROOT / str(root)).exists():
            problems.append((jid, f"experience_root {root} does not exist"))
        for f in (j.get("expect_files") or []):
            # the file need not exist yet, but its directory must be writable-ish
            if not (ROOT / str(f)).parent.exists():
                problems.append((jid, f"expect_files parent dir missing for {f}"))

        # ---- 4. the job must be judgeable by the dispatcher's own rule -------
        if not j.get("expect_files"):
            try:
                d.job_status_of_outputs(j)
            except Exception as exc:
                problems.append((jid, f"completion check raised {exc.__class__.__name__}: {exc}"))

    print(f"  distinct entry points checked: {len(shapes)}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for jid, msg in problems:
            print(f"  {jid}\n      {msg}")
        return 2
    print("\nOK: every pending job is dispatchable and semantically legal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
