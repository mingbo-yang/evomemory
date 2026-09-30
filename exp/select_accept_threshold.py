#!/usr/bin/env python
"""Select the acceptance-gate length threshold ON DEV, then it gets frozen.

Why dev and not test
--------------------
The gate's defect was *found* by looking at the test traces: every cell's entire
loss was attributable to the acceptance gate, whose "better" verdict is largely
a length preference (reports/judge_bottleneck_diagnosis.md).  Picking the
threshold from those same test traces would be test-set selection.  So the
threshold is chosen here, on the 32-item dev split, frozen into
``configs/frozen.json``, and only then are the test arms re-run.

Pre-registered selection rule (fixed before looking at any dev number)
---------------------------------------------------------------------
1. Candidate thresholds: a fixed grid, including ``None`` (the historical gate,
   which performs no length check at all).
2. Score of a threshold = the **sum over every (model, task) dev cell** of
   ``corpus(dev, gate=tau) - corpus(dev, draft)``, i.e. how much the gate gains
   over simply not refining.  Summing across cells prevents one model from
   dominating.
3. The winner is the highest score.  **Exact ties go to the LARGER tau**, i.e.
   the more permissive gate -- the same "ties favour the more conservative
   choice" convention already used for alpha, and it keeps the rule as close to
   the pre-registered one as the evidence allows.
4. If every threshold scores at or below zero, no threshold is adopted and the
   historical gate stands; that outcome is reported as-is.

The simulation replays the stored rounds: a revision is accepted iff the judge
called it better in both A/B orders AND ``len(candidate) <= tau * len(current)``.
That is exactly what ``core.judge.accept_revision`` does at run time, so the
offline choice and the online rule cannot drift apart.

GPU-free; reads only.

Run:  /home/ymb/miniconda3/envs/qwen35/bin/python select_accept_threshold.py
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import os
import sys

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core  # noqa: E402
from core.judge import accept_revision  # noqa: E402
from core.manifest import manifest_path, read_manifest  # noqa: E402
from core.scoring import Scorer  # noqa: E402

#: The fixed grid.  ``None`` is the shipped, unguarded gate.
TAU_GRID = [None, 1.00, 1.01, 1.02, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]
PRIMARY = {"wmt19_en_zh": "bleu", "wmt19_zh_en": "bleu",
           "coedit_gec": "gleu", "gigaword": "rouge1"}


def load(path: Path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dev_cells(runs_root: Path, split: str) -> dict:
    """(model, task) -> records, for a completed dev run of one arm."""
    cells = {}
    for p in sorted(glob.glob(str(runs_root / "seed*" / "*" / "*" / split / "*.jsonl"))):
        path = Path(p)
        try:
            recs = load(path)
        except Exception:
            continue
        if not recs:
            continue
        task, model = recs[0]["task"], recs[0]["model"]
        cells[(model, task)] = (path, recs)
    return cells


def replay(recs, refs, scorer, task, tau):
    """Return (draft pairs, gated pairs) for one cell under threshold ``tau``."""
    draft, gated = [], []
    n_rounds = n_accepted = 0
    for r in recs:
        ref = refs.get(r["sample_id"])
        if ref is None:
            continue
        draft.append((ref, r["initial_draft"]))
        cur = r["initial_draft"]
        for rd in r["rounds"]:
            if rd.get("controller_action") != "REFINE":
                continue
            n_rounds += 1
            cand = rd.get("candidate", "")
            # The SAME predicate the pipelines call, so offline selection and
            # online behaviour cannot diverge.
            if accept_revision(rd.get("judge_verdict", "uncertain"),
                               bool(rd.get("judge_order_consistent")), cur, cand, tau):
                cur = cand
                n_accepted += 1
        gated.append((ref, cur))
    metric = PRIMARY[task]
    return (scorer.score_corpus(draft)[metric] if draft else float("nan"),
            scorer.score_corpus(gated)[metric] if gated else float("nan"),
            n_rounds, n_accepted)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/dev_gate_fix", help="dev traces root")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--arm", default="full_static")
    ap.add_argument("--freeze", action="store_true",
                    help="write the chosen threshold into configs/frozen.json")
    ap.add_argument("--json-out", default="scores/accept_threshold_selection.json")
    args = ap.parse_args()

    root = Path(args.runs) / args.arm
    cells = dev_cells(root, args.split)
    if not cells:
        print(f"no completed dev traces under {root}; nothing to select from")
        return 2

    scorers, refcache = {}, {}
    for (model, task) in cells:
        scorers[task] = Scorer(task)
        if task not in refcache:
            refcache[task] = {r.sample_id: r.reference
                              for r in read_manifest(manifest_path(task, args.split))}

    print(f"dev cells found: {len(cells)}")
    for (model, task), (path, recs) in sorted(cells.items()):
        print(f"  {model:14s} {task:13s} n={len(recs):3d}  {path}")

    # ---- replay every threshold over every cell ---------------------------
    table = {}
    for tau in TAU_GRID:
        per_cell, total = {}, 0.0
        for (model, task), (_p, recs) in sorted(cells.items()):
            d, g, nr, na = replay(recs, refcache[task], scorers[task], task, tau)
            per_cell[f"{model}/{task}"] = {"draft": d, "gated": g, "gain": g - d,
                                           "rounds": nr, "accepted": na}
            total += g - d
        table["none" if tau is None else f"{tau:.2f}"] = {"tau": tau, "total": total,
                                                          "cells": per_cell}

    print()
    print(f"{'阈值 tau':>10s}{'跨格总增益':>12s}" + "".join(
        f"{k.split('/')[0][:6]+'/'+k.split('/')[1][:5]:>14s}" for k in
        sorted(next(iter(table.values()))["cells"])))
    print("-" * (22 + 14 * len(next(iter(table.values()))["cells"])))
    for name, row in table.items():
        cells_sorted = sorted(row["cells"])
        print(f"{name:>10s}{row['total']:+12.3f}" + "".join(
            f"{row['cells'][k]['gain']:+14.3f}" for k in cells_sorted))

    # ---- apply the pre-registered rule ------------------------------------
    best = max(table.items(), key=lambda kv: (round(kv[1]["total"], 6),
                                              -1 if kv[1]["tau"] is None else kv[1]["tau"]))
    # ties -> LARGER tau (None sorts last, i.e. treated as the most conservative)
    top = round(best[1]["total"], 6)
    tied = [k for k, v in table.items() if round(v["total"], 6) == top]
    if len(tied) > 1:
        tied.sort(key=lambda k: (-1 if table[k]["tau"] is None else table[k]["tau"]))
        best = (tied[-1], table[tied[-1]])
        print(f"\nexact tie between {tied} -> larger tau per the pre-registered rule")

    chosen = best[1]["tau"]
    print(f"\nCHOSEN tau = {chosen}   (cross-cell gain {best[1]['total']:+.3f})")

    report = {
        "rule": ("maximise the sum over (model, task) dev cells of "
                 "corpus(gate=tau) - corpus(draft); exact ties -> larger tau"),
        "grid": ["none" if t is None else t for t in TAU_GRID],
        "cells": sorted(f"{m}/{t}" for (m, t) in cells),
        "table": {k: {"tau": v["tau"], "total": round(v["total"], 6),
                      "cells": {c: {kk: (None if vv != vv else round(vv, 6))
                                    for kk, vv in cv.items()}
                                for c, cv in v["cells"].items()}}
                  for k, v in table.items()},
        "chosen_tau": chosen,
        "chosen_total_gain": round(best[1]["total"], 6),
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")

    if args.freeze:
        fp = Path("configs/frozen.json")
        c = json.loads(fp.read_text(encoding="utf-8"))
        c["accept_gate"] = {
            "frozen_at": "2026-09-12",
            "chosen_on": "dev split only",
            "chosen_tau": chosen,
            "rule": report["rule"],
            "evidence": str(out),
            "why": ("the acceptance gate was the sole cause of every cell's loss "
                    "(reports/judge_bottleneck_diagnosis.md): the judge is the same "
                    "model that wrote the revision and prefers its longer rewrites. "
                    "The threshold is selected on dev and frozen BEFORE any test-set "
                    "re-run; the unguarded gate is kept as a reported condition."),
            "default_when_unset": None,
            "note": ("None reproduces the historical gate bit-for-bit. The value is a "
                     "pipeline argument, never a RunConfig field, so enabling it does "
                     "not change any existing config hash."),
        }
        fp.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> frozen into {fp} as accept_gate.chosen_tau = {chosen}")
    else:
        print("(not frozen; re-run with --freeze once the dev cells are complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
