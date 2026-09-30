#!/usr/bin/env python
"""Relabel the experience libraries with the GOLD-metric outcome, offline.

Why this exists
---------------
The rendered outcome of an experience unit is what the whole method is supposed
to learn from.  Today that outcome is the double-order **judge verdict** -- and
the judge is the same model that wrote the revision, so its verdict is largely a
length preference (measured precision 21.5%).  Three independent checks showed
the consequence: retrieving "better" experiences predicted *worse* subsequent
revisions (17.8% vs 20.7% improvement), retrieval similarity was flat across
quartiles (16.4%..18.9%), and experience did not raise the revision quality at
all while costing 11x the compute of a one-round baseline.

The bootstrap runs on the AUXILIARY (``initial``) split, whose references are
available, so the true gold-metric effect of every stored intervention can be
measured.  This rewrites each unit with:

* ``delta_offline``  -- the real gold-metric delta (kept offline-only), and
* ``outcome_label``  -- one of ``helped|hurt|unchanged``, which the prompt
  renderer prefers over the judge verdict when present.

**The unit set, the retrieval fields and every prompt stay byte-identical** --
only the outcome label changes.  That makes "label source" a clean
single-variable change rather than a new library.

Nothing is overwritten in place: the relabelled libraries are written under a
new root, so the runs already on disk remain backed by the library they used.

GPU-free.  Reads and writes files only.

Run:
  python relabel_experience.py --out-root experience/initial_gold          # all
  python relabel_experience.py --out-root experience/initial_gold --models glm4-9b qwen3-8b
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core  # noqa: E402
from core.bm25_fields import Experience  # noqa: E402
from core.manifest import manifest_path, read_manifest  # noqa: E402
from core.scoring import Scorer  # noqa: E402

EPS = 1e-9


def relabel(path: Path, refs: dict, scorer: Scorer) -> tuple:
    """Return (relabelled units, stats) for one library file."""
    units, stats = [], Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = Experience.from_dict(json.loads(line))
            # exp_id == "<task>/<model>/<task>/<split>/<index>/b<k>", so the
            # sample_id is everything between the model and the "b<k>" suffix.
            parts = e.exp_id.split("/")
            sid = "/".join(parts[2:-1]) if len(parts) >= 5 else ""
            ref = refs.get(sid)
            if ref is None:
                stats["no_reference"] += 1
                units.append(e)          # keep the unit, leave the label empty
                continue
            before = scorer.primary(ref, e.state_before)
            after = scorer.primary(ref, e.state_after)
            delta = (after or 0.0) - (before or 0.0)
            label = "helped" if delta > EPS else ("hurt" if delta < -EPS else "unchanged")
            # Experience is a frozen dataclass
            e = dataclasses.replace(e, delta_offline=delta, outcome_label=label)
            stats[f"gold_{label}"] += 1
            stats[f"judge_{e.verdict}"] += 1
            # how often the two sources agree -- this is the headline number
            if (label == "helped") == (e.verdict == "better" and e.order_consistent):
                stats["agree"] += 1
            else:
                stats["disagree"] += 1
            units.append(e)
    return units, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="experience/initial")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    src, out = Path(args.src_root), Path(args.out_root)
    files = sorted(src.rglob("initial.jsonl"))
    scorers, refcache = {}, {}
    grand = Counter()
    n_files = 0
    for f in files:
        model, task = f.parent.parent.name, f.parent.name
        if args.models and model not in args.models:
            continue
        scorers.setdefault(task, Scorer(task))
        if task not in refcache:
            refcache[task] = {r.sample_id: r.reference
                              for r in read_manifest(manifest_path(task, "initial"))}
        units, stats = relabel(f, refcache[task], scorers[task])
        dest = out / model / task / "initial.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as fh:
            for e in units:
                fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
        n_files += 1
        grand.update(stats)
        agree = stats["agree"]; dis = stats["disagree"]
        print(f"  {model:14s} {task:13s} n={len(units):3d} "
              f"gold(helped/hurt/unchanged)={stats['gold_helped']}/{stats['gold_hurt']}/{stats['gold_unchanged']} "
              f"| judge vs gold agree={agree} disagree={dis}"
              + (f" ({agree/(agree+dis):.0%})" if agree + dis else ""))

    a, d = grand["agree"], grand["disagree"]
    print(f"\n{n_files} libraries -> {out}")
    print(f"gold labels : helped={grand['gold_helped']} hurt={grand['gold_hurt']} "
          f"unchanged={grand['gold_unchanged']}")
    print(f"judge labels: better={grand['judge_better']} worse={grand['judge_worse']} "
          f"tie={grand['judge_tie']} uncertain={grand['judge_uncertain']}")
    if a + d:
        print(f"\n** judge agrees with the gold outcome on only {a}/{a+d} = {a/(a+d):.1%} of units **")
        print("That is the share of the library whose rendered outcome was correct.")
    if grand["no_reference"]:
        print(f"({grand['no_reference']} units had no reference and kept an empty label)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
