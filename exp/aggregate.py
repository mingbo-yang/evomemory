#!/usr/bin/env python
"""Phase 9: aggregate every run into paper tables, CIs and plots.

Reads the per-sample traces under ``runs/`` and recomputes the headline metric
with the single shared scorer, so no table can mix metric definitions.

Corpus metrics (SacreBLEU for translation) are computed over the whole run,
while per-sample values are used as the pairing unit for the bootstrap -- every
arm shares the same test items and the same ``y0``, so paired resampling is the
right test.

Usage
-----
    python aggregate.py                       # everything under runs/
    python aggregate.py --runs runs/main runs/ablation
    python aggregate.py --out-prefix main
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import EXP_ROOT  # noqa: E402
from core.scoring import PRIMARY_METRIC, Scorer  # noqa: E402
from core.stats import holm_correct, paired_bootstrap  # noqa: E402


def load_traces(path: Path) -> List[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def expected_n(task: str) -> int:
    """Full test size for a task, matching baseline.TEST_SAMPLES."""
    from core import TEST_SAMPLES

    return TEST_SAMPLES.get(task, 1000)


def discover(run_dirs: List[Path], include_partial: bool = False) -> List[dict]:
    """Yield one record per (run_dir, arm, seed, task, model) trace file."""
    found = []
    for base in run_dirs:
        if not base.exists():
            continue
        for jsonl in sorted(base.rglob("*.jsonl")):
            try:
                recs = load_traces(jsonl)
            except Exception:
                continue
            if not recs:
                continue
            # Completeness guard.  A supervisor retry opens the JSONL with "w"
            # and restarts from item 0, so a run can be truncated mid-flight.
            # Pooling a 64-sample prefix with a 1000-sample one would silently
            # corrupt any corpus-level metric, so partial runs are skipped
            # unless the caller explicitly opts in.
            if not include_partial:
                n_here = len(recs)
                if n_here < expected_n(recs[0]["task"]):
                    print(f"  [skip partial] {jsonl} has {n_here} samples "
                          f"(expected {expected_n(recs[0]['task'])})")
                    continue
            cfg = recs[0].get("config", {})
            # a sweep writes into .../<split>/<tag>/<arm>.jsonl; recover the tag
            # so different alpha / budget points never collapse into one row
            # A sweep writes into .../<split>/<tag>/<arm>.jsonl, so the parent dir
            # is the tag -- but for an ordinary run the parent dir is just the
            # model name, which is NOT a tag.  Reporting the model name as a tag
            # made every (arm, model) look like a distinct sweep point and broke
            # the arm lookup below.
            parent = jsonl.parent.name
            tag = "" if parent in ("test", "dev") or parent == recs[0]["model"] else parent
            found.append(
                {
                    "path": jsonl,
                    "tag": tag,
                    "arm": jsonl.stem,
                    "task": recs[0]["task"],
                    "model": recs[0]["model"],
                    "seed": recs[0].get("seed"),
                    "split": "dev" if "/dev/" in str(jsonl) else "test",
                    "config": cfg,
                    "records": recs,
                }
            )
    return found


def metrics_for(rec: dict) -> Tuple[Dict[str, float], List[float], List[float]]:
    """Return (corpus metrics, per-sample initial, per-sample final)."""
    task = rec["task"]
    scorer = Scorer(task)
    pairs_init = [(r["config"].get("_ref", ""), r["initial_draft"]) for r in rec["records"]]
    # references are not stored in the trace; recover them from the manifest
    from core.manifest import manifest_path, read_manifest

    split = rec["split"]
    refs = {r.sample_id: r.reference for r in read_manifest(manifest_path(task, split))}
    pairs_init = [(refs.get(r["sample_id"], ""), r["initial_draft"]) for r in rec["records"]]
    pairs_final = [(refs.get(r["sample_id"], ""), r["final_output"]) for r in rec["records"]]
    corpus_init = scorer.score_corpus(pairs_init)
    corpus_final = scorer.score_corpus(pairs_final)
    per_init = [r.get("initial_metric_offline") or 0.0 for r in rec["records"]]
    per_final = [r.get("final_metric_offline") or 0.0 for r in rec["records"]]
    merged = {f"init_{k}": v for k, v in corpus_init.items()}
    merged.update({f"final_{k}": v for k, v in corpus_final.items()})
    return merged, per_init, per_final


def _write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out-prefix", default="all")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--compare", nargs="*", default=None,
                    help="explicit pairs 'ARM_A:ARM_B' (repeatable); default is the "
                         "paper's named comparison set")
    ap.add_argument("--include-partial", action="store_true",
                    help="include truncated runs (off by default: a retry can leave a prefix)")
    args = ap.parse_args()

    bases = [Path(p) for p in args.runs] if args.runs else [EXP_ROOT / "runs"]
    recs = discover(bases, args.include_partial)
    if not recs:
        print("no runs found")
        return 1

    rows = []
    per_sample: Dict[Tuple[str, str, str, str], Tuple[List[float], List[float]]] = {}
    # Pairing by POSITION is only valid while every arm's file is in manifest
    # order, and nothing enforced that.  A resumed run, a run whose prefix was
    # rewritten, or any reordering would silently pair sample i of one arm with
    # sample j of another and produce a meaningless bootstrap that still looks
    # fine.  Keep an explicit sample_id -> value map and pair on the id.
    by_id: Dict[Tuple[str, str, str, str], Dict[str, Tuple[float, float, str]]] = {}
    for rec in recs:
        m, pi, pf = metrics_for(rec)
        key = (rec["arm"], rec["tag"], rec["task"], rec["model"], str(rec["seed"]))
        per_sample[key] = (pi, pf)
        by_id[key] = {
            r["sample_id"]: (pi[i], pf[i], r.get("initial_draft", ""))
            for i, r in enumerate(rec["records"])
        }
        primary = PRIMARY_METRIC[rec["task"]]
        rounds = [len(r["rounds"]) for r in rec["records"]]
        n_ref = sum(
            1
            for r in rec["records"]
            for x in r["rounds"]
            if x.get("controller_action") == "REFINE"
        )
        n_acc = sum(
            1
            for r in rec["records"]
            for x in r["rounds"]
            if x.get("accepted")
        )
        toks = sum(r["cost"]["total_tokens"] for r in rec["records"])
        calls = sum(r["cost"]["n_calls"] for r in rec["records"])
        rows.append(
            {
                "arm": rec["arm"],
                "tag": rec["tag"],
                "task": rec["task"],
                "model": rec["model"],
                "seed": rec["seed"],
                "split": rec["split"],
                "n": len(rec["records"]),
                "primary_metric": primary,
                "corpus_init": round(m.get(f"init_{primary}", float("nan")), 4),
                "corpus_final": round(m.get(f"final_{primary}", float("nan")), 4),
                "corpus_delta": round(
                    m.get(f"final_{primary}", 0.0) - m.get(f"init_{primary}", 0.0), 4
                ),
                "mean_rounds": round(sum(rounds) / len(rounds), 3) if rounds else 0.0,
                "refine_attempts": n_ref,
                "accepted": n_acc,
                "accept_rate": round(n_acc / n_ref, 4) if n_ref else 0.0,
                "tokens_per_sample": round(toks / len(rec["records"]), 1),
                "calls_per_sample": round(calls / len(rec["records"]), 2),
            }
        )

    scores_dir = EXP_ROOT / "scores"
    _write_csv(rows, scores_dir / f"{args.out_prefix}_runs.csv")

    # ---- paired bootstrap on the comparisons the paper actually reports ----
    # The old code picked "the control" by dict-overwrite, so which arm a run was
    # compared against depended on iteration order -- and once sr_j_fixed and
    # sr_j_stop exist there are several legitimate controls.  State every
    # comparison explicitly instead.
    def _key(arm, task, model, seed):
        """The sample-id map for one arm at one (task, model, seed).

        Matches on the tag-less key first; a sweep point (tag != "") is only
        accepted when it is the single candidate, so two sweep points can never
        be silently averaged into one comparison.
        """
        hits = [k for k in by_id
                if k[0] == arm and k[2] == task and k[3] == model and k[4] == str(seed)]
        if not hits:
            return None
        plain = [k for k in hits if k[1] == ""]
        if plain:
            return plain[0]
        return hits[0] if len(hits) == 1 else None

    DEFAULT_COMPARISONS = [
        ("full_static", "no_experience", "does experience matter at all"),
        ("full_static", "sr_j_fixed", "vs self-refine, forced rounds"),
        ("full_static", "sr_j_stop", "vs self-refine, judge-stopped"),
        ("full_static", "outcome_hidden", "does the outcome field carry the value"),
        ("full_static", "random_retrieve", "does the retrieval ranking matter"),
        ("full_static", "positive_only", "does the negative pool matter"),
        ("full_static", "full_online", "does online accumulation help"),
        ("full_static", "fixed_rounds", "does adaptive stopping beat always-refining"),
        ("no_experience", "sr_j_fixed", "experience vs forced refinement, both without stopping"),
    ]
    pairs = []
    if args.compare:
        for spec in args.compare:
            a, _, b = spec.partition(":")
            pairs.append((a.strip(), b.strip(), "user-specified"))
    else:
        pairs = DEFAULT_COMPARISONS

    y0_mismatches = []
    boots = []
    pvals = []
    for arm_a, arm_b, why in pairs:
        for (task, model, seed) in sorted({(k[2], k[3], k[4]) for k in by_id}):
            ka, kb = _key(arm_a, task, model, seed), _key(arm_b, task, model, seed)
            if ka is None or kb is None:
                continue
            da, db = by_id[ka], by_id[kb]
            common = sorted(set(da) & set(db))
            if not common:
                continue
            # The whole comparison rests on both arms starting from the SAME y0.
            # Nothing verified that on real data before; check it here so a
            # broken pairing can never be reported as a result.
            bad = [s for s in common if da[s][2] != db[s][2]]
            if bad:
                y0_mismatches.append(
                    {"arm_a": arm_a, "arm_b": arm_b, "task": task, "model": model,
                     "seed": seed, "n_mismatch": len(bad), "n_common": len(common),
                     "example": bad[0]}
                )
            va = [da[s][1] for s in common]
            vb = [db[s][1] for s in common]
            res = paired_bootstrap(va, vb, n_resamples=args.bootstrap)
            boots.append({
                "arm_a": arm_a, "arm_b": arm_b, "why": why,
                "task": task, "model": model, "seed": seed,
                "n_paired": res.n, "n_a_only": len(set(da) - set(db)),
                "n_b_only": len(set(db) - set(da)),
                "mean_a": round(res.mean_a, 4), "mean_b": round(res.mean_b, 4),
                "diff_a_minus_b": round(res.diff, 4),
                "ci_low": round(res.ci_low, 4), "ci_high": round(res.ci_high, 4),
                "p_raw": round(res.p_two_sided, 5), "y0_identical": not bad,
                "_p": res.p_two_sided,
            })
            pvals.append(res.p_two_sided)
    if boots:
        adjusted = holm_correct(pvals)
        for b, a in zip(boots, adjusted):
            b["p_holm"] = round(a, 5)
            b["ci_excludes_0"] = int(b["ci_low"] > 0 or b["ci_high"] < 0)
        for b in boots:
            b.pop("_p", None)
        _write_csv(boots, scores_dir / f"{args.out_prefix}_bootstrap.csv")
    if y0_mismatches:
        _write_csv(y0_mismatches, scores_dir / f"{args.out_prefix}_y0_mismatches.csv")
        print(f"!! {len(y0_mismatches)} (arm,task,model) groups have a y0 that differs "
              f"between arms -- paired comparison is INVALID for those -> "
              f"scores/{args.out_prefix}_y0_mismatches.csv")

    # ---- console table -----------------------------------------------------
    print(f"{'arm':14s}{'tag':6s}{'task':14s}{'model':11s}{'init':>8s}{'final':>8s}{'delta':>8s}"
          f"{'rounds':>8s}{'accept':>8s}{'tok/s':>9s}")
    for r in rows:
        print(
            f"{r['arm']:14s}{r['tag']:6s}{r['task']:14s}{r['model']:11s}"
            f"{r['corpus_init']:8.2f}{r['corpus_final']:8.2f}{r['corpus_delta']:+8.2f}"
            f"{r['mean_rounds']:8.2f}{r['accept_rate']:8.1%}{r['tokens_per_sample']:9.0f}"
        )
    print()
    print(f"{len(rows)} runs -> scores/{args.out_prefix}_runs.csv")
    if boots:
        print(f"{len(boots)} paired bootstraps -> scores/{args.out_prefix}_bootstrap.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
