#!/usr/bin/env python
"""Phase 1a: build all dataset manifests and verify test/aux isolation.

Usage:
    /home/ymb/miniconda3/envs/qwen35/bin/python build_manifests.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import TASKS  # noqa: E402
from core.manifest import build_all, check_isolation  # noqa: E402


def main() -> int:
    print("=" * 78)
    print("Phase 1a: building manifests")
    print("=" * 78)
    summary = build_all(verbose=True)

    print()
    print("=" * 78)
    print("Isolation check (test ∩ aux must be empty, aux splits disjoint)")
    print("=" * 78)
    ok = True
    for task in TASKS:
        rep = check_isolation(task)
        status = "OK " if rep["ok"] else "FAIL"
        ok = ok and rep["ok"]
        print(
            f"[{status}] {task:14s} test={rep['test_n']:5d} "
            f"initial={rep.get('initial_n')} dev={rep.get('dev_n')} "
            f"accum={rep.get('accumulation_n')}"
        )
        for v in rep["violations"]:
            print(f"        VIOLATION: {v}")

    out = ROOT / "data" / "manifests" / "_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"summary -> {out}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
