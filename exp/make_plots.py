#!/usr/bin/env python
"""Phase 9 plotting layer: publication figures from the JSONL traces alone.

Figures (all written to ``plots/`` by default)
---------------------------------------------
1. ``fig1_quality_init_vs_final``   -- per task x model, initial vs final quality
   with paired-bootstrap 95% CIs (``core.stats.paired_bootstrap``).
2. ``fig2_full_minus_baseline``     -- Full minus the strongest non-experience
   arm per task x model, with 95% CIs and an explicit "CI excludes 0" marker;
   Holm-adjusted p-values come from ``core.stats.holm_correct``.
3. ``fig3_quality_vs_compute``      -- final quality against mean total tokens
   per sample, one line per (arm, model), every point labelled with its n.
4. ``fig4_unnecessary_refinement``  -- unnecessary-refinement rate per model x
   task (REFINE rounds whose candidate was *worse* under the offline metric than
   the answer it replaced), using exactly the convention of
   ``run_stopping_diagnostics.py``: ``worse / n_refine_scored``.

Rules that keep the numbers honest
----------------------------------
* Statistics are *imported*, never reimplemented: the bootstrap and the Holm
  correction come from ``core/stats.py``; the arm attribution, the completeness
  rule and the baseline auto-selection come from ``export_human_eval.py``, which
  in turn uses ``core.TEST_SAMPLES`` exactly like ``aggregate.py``.
* A run is COMPLETE only when its trace holds at least the full test size
  (1000 samples, 100 for gigaword).  Partial runs are either excluded
  (``--complete-only``) or drawn with a "†" marker and reported explicitly --
  they are never silently pooled with complete runs.
* Bar heights for figure 1 are the *corpus* metric recomputed with the shared
  ``core.scoring.Scorer`` when that is possible (the paper's headline number);
  the paired-bootstrap CI is always computed on the per-sample offline metric,
  which is the pairing unit the plan uses.  When the scorer's dependencies are
  missing the figure falls back to mean per-sample metric bars and says so in
  the axis label, the console report and the manifest.
* Every figure is reproducible from the traces: the optional corpus cache
  (``plots/_corpus_metrics.json``) is only a speed-up, is written by this script
  from the traces plus manifests, and is invalidated whenever a trace changes
  size or mtime.

matplotlib is imported lazily: if it is missing the script prints the exact
situation and exits 2 without touching anything (``--no-figures
--refresh-corpus`` deliberately runs without matplotlib).  In this checkout
matplotlib is *not* installed in ``/home/ymb/miniconda3/envs/qwen35/bin/python``
(the environment that has sacrebleu/nltk/rouge_score) but it is available in
``/home/ymb/miniconda3/envs/llama4/bin/python``, which is the interpreter used
for the figures below.

Usage
-----
    python make_plots.py                       # all four figures -> plots/
    python make_plots.py --complete-only       # drop every partial run
    python make_plots.py --baseline-arm sr_j_stop
    python make_plots.py --refresh-corpus --no-figures   # needs sacrebleu only
"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import EXP_ROOT, TASKS  # noqa: E402
from core.scoring import PRIMARY_METRIC  # noqa: E402  (module import is stdlib-only)
from core.stats import holm_correct, paired_bootstrap  # noqa: E402

# shared with Phase 10 -- one completeness rule, one arm attribution
from export_human_eval import (  # noqa: E402
    DEFAULT_RUN_DIRS,
    BaselineChoice,
    RunInfo,
    choose_baseline_arm,
    discover_runs,
    group_runs,
    is_non_experience,
    print_baseline_evidence,
)

CORPUS_CACHE_DEFAULT = "plots/_corpus_metrics.json"
FIGURE_FORMATS = ("png", "pdf")
BASELINE_MARKER = "†"  # partial-run marker in tick labels

TASK_SHORT = {
    "wmt19_en_zh": "en-zh",
    "wmt19_zh_en": "zh-en",
    "coedit_gec": "gec",
    "gigaword": "giga",
}

MODEL_ORDER = ("glm4-9b", "llama3.1-8b", "qwen3-4b", "qwen3-8b", "qwen3-32b")
ARM_COLORS = {
    "full_static": "#1f77b4",
    "no_experience": "#d62728",
    "sr_j_stop": "#2ca02c",
    "sr_j_fixed": "#9467bd",
    "outcome_hidden": "#ff7f0e",
    "random_retrieve": "#8c564b",
}
_MODEL_MARKERS = ("o", "s", "^", "D", "v", "P")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _seed_for(label: str, base: int) -> int:
    """Stable per-cell bootstrap seed (independent of iteration order)."""
    return (zlib.crc32(label.encode("utf-8")) ^ (base * 2654435761)) % (2 ** 31 - 1)


def mean_ci(values: Sequence[float], *, n_resamples: int, seed: int) -> Tuple[float, float, float]:
    """Mean and 95% CI of the mean, via ``core.stats.paired_bootstrap`` vs zero."""
    vals = [float(v) for v in values]
    res = paired_bootstrap(vals, [0.0] * len(vals), n_resamples=n_resamples, seed=seed)
    return res.mean_a, res.ci_low, res.ci_high


def paired_diff_ci(a: Sequence[float], b: Sequence[float], *, n_resamples: int, seed: int):
    """Paired bootstrap of ``mean(a - b)`` using ``core.stats.paired_bootstrap``."""
    return paired_bootstrap(
        [float(x) for x in a], [float(y) for y in b], n_resamples=n_resamples, seed=seed
    )


def metric_values(run: RunInfo, which: str) -> List[float]:
    key = "initial_metric_offline" if which == "init" else "final_metric_offline"
    return [float(r[key]) for r in run.records if r.get(key) is not None]


def tokens_per_sample(run: RunInfo) -> Optional[float]:
    toks = []
    for rec in run.records:
        cost = rec.get("cost") or {}
        if cost.get("total_tokens") is None:
            return None
        toks.append(float(cost["total_tokens"]))
    return sum(toks) / len(toks) if toks else None


def label_cell(task: str, model: str, run: Optional[RunInfo]) -> str:
    if run is None:
        return f"{TASK_SHORT.get(task, task)}\n{model}\n(no data)"
    tag = BASELINE_MARKER if not run.complete else ""
    return f"{TASK_SHORT.get(task, task)}\n{model}{tag}"


def _by_sample(run: RunInfo, key: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for rec in run.records:
        val = rec.get(key)
        if val is not None:
            out[str(rec.get("sample_id", ""))] = float(val)
    return out


def paired_values(
    full: RunInfo, other: RunInfo, key: str
) -> Tuple[List[float], List[float], List[str]]:
    """Per-sample values of both arms on their shared sample ids (sorted)."""
    a, b = _by_sample(full, key), _by_sample(other, key)
    shared = sorted(set(a) & set(b))
    return [a[s] for s in shared], [b[s] for s in shared], shared


# --------------------------------------------------------------------------- #
# corpus metrics (optional enrichment, cached)
# --------------------------------------------------------------------------- #


def load_corpus_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # a half-written cache must never crash a figure
        print(f"[WARN] ignoring unreadable corpus cache {path}: {exc}")
        return {}
    return payload.get("runs", {}) if isinstance(payload, dict) else {}


def scorer_available() -> Tuple[bool, str]:
    try:
        from core.scoring import Scorer  # noqa: F401

        return True, "core.scoring.Scorer importable"
    except Exception as exc:  # pragma: no cover - depends on the interpreter
        return False, f"core.scoring.Scorer unavailable: {type(exc).__name__}: {exc}"


def compute_corpus(run: RunInfo) -> dict:
    """Corpus primary metric of one run, recomputed with the shared Scorer."""
    from core.manifest import manifest_path, read_manifest
    from core.scoring import Scorer

    mpath = manifest_path(run.task, run.split)
    if not mpath.exists():
        return {"error": f"manifest missing: {mpath}"}
    refs = {r.sample_id: r.reference for r in read_manifest(mpath)}
    scorer = Scorer(run.task)
    primary = PRIMARY_METRIC[run.task]
    pairs_init = [(refs.get(str(r.get("sample_id", "")), ""), r.get("initial_draft") or "")
                  for r in run.records]
    pairs_final = [(refs.get(str(r.get("sample_id", "")), ""), r.get("final_output") or "")
                   for r in run.records]
    try:
        ci = scorer.score_corpus(pairs_init)
        cf = scorer.score_corpus(pairs_final)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "corpus_init": float(ci[primary]),
        "corpus_final": float(cf[primary]),
        "primary": primary,
        "n": run.n,
        "mtime": run.info.mtime,
        "path": run.rel,
        "arm": run.arm,
        "task": run.task,
        "model": run.model,
    }


def corpus_for(
    run: RunInfo, cache: dict, *, allow_compute: bool, log=print
) -> Optional[dict]:
    """Cached corpus metrics for a run, invalidated on size/mtime change."""
    entry = cache.get(run.rel)
    if entry and not entry.get("error") and entry.get("n") == run.n and \
            abs(float(entry.get("mtime", -1)) - run.info.mtime) < 1e-6:
        return entry
    if entry and (entry.get("n") != run.n):
        log(f"[corpus] cache stale for {run.rel} (n {entry.get('n')} -> {run.n})")
    if not allow_compute:
        return None
    fresh = compute_corpus(run)
    if fresh.get("error"):
        log(f"[corpus] {run.rel}: {fresh['error']}")
        return None
    cache[run.rel] = fresh
    return fresh


def write_corpus_cache(path: Path, cache: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_meta": meta, "runs": cache}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #


def write_rows(rows: Sequence[dict], path: Path, fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def save_figure(fig, out_dir: Path, stem: str, formats: Sequence[str]) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    return paths


def align_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Left-align a table (header + rows) to the widest cell per column."""
    ncol = len(header)
    widths = [len(str(h)) for h in header]
    for row in rows:
        for i, cell in enumerate(list(row)[:ncol]):
            widths[i] = max(widths[i], len(str(cell)))
    out = ["  " + "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)).rstrip()]
    for row in rows:
        cells = list(row)[:ncol]
        out.append("  " + "  ".join(str(c).ljust(widths[i])
                                    for i, c in enumerate(cells)).rstrip())
    return out


def report_figure(name: str, files: Sequence[Path], header: Sequence[str],
                  rows: Sequence[Sequence[str]], log=print) -> None:
    log("")
    log(f"=== {name} ===")
    for p in files:
        log(f"  file: {p}")
    if rows:
        for line in align_table(header, rows):
            log(line)
    else:
        log("  (no numbers)")


def _fmt(v: Optional[float], nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


# --------------------------------------------------------------------------- #
# figure 1: initial vs final quality
# --------------------------------------------------------------------------- #


def figure_quality(args, runs: Sequence[RunInfo], corpus: Dict[str, dict], plt, log=print) -> dict:
    main_runs = [r for r in runs if r.arm == args.main_arm]
    models = [m for m in MODEL_ORDER if any(r.model == m for r in main_runs)]
    models += sorted({r.model for r in main_runs} - set(models))
    tasks = list(args.tasks)
    cells: Dict[Tuple[str, str], RunInfo] = {(r.task, r.model): r for r in main_runs}

    plotted: List[dict] = []
    for task in tasks:
        for model in models:
            run = cells.get((task, model))
            if run is None or (args.complete_only and not run.complete):
                continue
            init_vals, final_vals = metric_values(run, "init"), metric_values(run, "final")
            if not init_vals or not final_vals:
                continue
            label = f"{task}/{model}"
            res = paired_diff_ci(final_vals, init_vals,
                                 n_resamples=args.bootstrap, seed=_seed_for(label, args.seed))
            ci = corpus.get(run.rel) if run.rel in corpus else None
            plotted.append(
                {
                    "task": task, "model": model, "label": label, "run": run,
                    "n": len(final_vals), "status": run.status,
                    "init_mean": sum(init_vals) / len(init_vals),
                    "final_mean": sum(final_vals) / len(final_vals),
                    "init_ci": mean_ci(init_vals, n_resamples=args.bootstrap,
                                       seed=_seed_for(label + "|init", args.seed)),
                    "final_ci": mean_ci(final_vals, n_resamples=args.bootstrap,
                                        seed=_seed_for(label + "|final", args.seed)),
                    "paired": res,
                    "corpus_init": ci.get("corpus_init") if ci else None,
                    "corpus_final": ci.get("corpus_final") if ci else None,
                    "primary": (ci or {}).get("primary", PRIMARY_METRIC[task]),
                }
            )
    if not plotted:
        log(f"[fig1] SKIPPED: no {args.main_arm} run with usable metrics "
            f"(tasks={tasks}, models={models})")
        return {"name": "fig1_quality_init_vs_final", "files": [], "rows": [], "status": "skipped"}

    # Holm correction over the whole family of init->final tests in this figure
    pvals = [c["paired"].p_two_sided for c in plotted]
    adjusted = holm_correct(pvals)
    for cell, p_adj in zip(plotted, adjusted):
        cell["p_holm"] = p_adj

    ncols = max(1, len(models))
    nrows = max(1, len(tasks))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols + 1.6, 2.5 * nrows + 1.2),
                             squeeze=False)
    for i, task in enumerate(tasks):
        for j, model in enumerate(models):
            ax = axes[i][j]
            cell = next((c for c in plotted if c["task"] == task and c["model"] == model), None)
            if cell is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", color="#999999",
                        fontsize=10, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{TASK_SHORT.get(task, task)} / {model}", fontsize=9)
                continue
            have_corpus = cell["corpus_init"] is not None and cell["corpus_final"] is not None
            if have_corpus:
                heights = [cell["corpus_init"], cell["corpus_final"]]
                bar_ci = None
            else:
                heights = [cell["init_mean"], cell["final_mean"]]
                bar_ci = [
                    [heights[0] - cell["init_ci"][1], heights[1] - cell["final_ci"][1]],
                    [cell["init_ci"][2] - heights[0], cell["final_ci"][2] - heights[1]],
                ]
            xs = [0, 1]
            bars = ax.bar(xs, heights, width=0.55, color=["#9ecae1", "#3182bd"],
                          yerr=bar_ci, capsize=3,
                          error_kw={"elinewidth": 1.0, "ecolor": "#333333"} if bar_ci else None)
            partial = not cell["run"].complete
            for b in bars:
                if partial:
                    b.set_hatch("///")
                    b.set_edgecolor("#555555")
            top = max(heights + [0.0])
            span = max(top, 1.0)
            ax.set_ylim(0, top + 0.58 * span)
            for x, h in zip(xs, heights):
                ax.text(x, h + 0.02 * span, f"{h:.2f}", ha="center", fontsize=8)
            res = cell["paired"]
            star = "*" if (res.ci_low > 0 or res.ci_high < 0) else "n.s."
            ax.text(0.5, top + 0.38 * span,
                    f"Δ(per-sample)={res.diff:+.2f} [{res.ci_low:+.2f},{res.ci_high:+.2f}] {star}\n"
                    f"Holm p={cell['p_holm']:.3g}",
                    ha="center", va="bottom", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.22", fc="#f5f5f5", ec="#bbbbbb", lw=0.6))
            title = f"{TASK_SHORT.get(task, task)} / {model}"
            if partial:
                title += f"{BASELINE_MARKER} (partial n={cell['n']}/{cell['run'].expected_n})"
            else:
                title += f" (n={cell['n']})"
            ax.set_title(title, fontsize=9)
            ax.set_xticks(xs)
            ax.set_xticklabels(["initial", "final"], fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
            if j == 0:
                ax.set_ylabel(PRIMARY_METRIC[task].upper(), fontsize=8)
    mode = ("bars = corpus metric (shared Scorer); box = paired-bootstrap 95% CI of the "
            "per-sample init->final difference (2000 resamples, Holm-corrected)")
    if any(c["corpus_init"] is None for c in plotted):
        mode += ("; cells without a corpus value fall back to mean per-sample metric bars "
                 "with bootstrap CI whiskers")
    if any(not c["run"].complete for c in plotted):
        mode += f"; {BASELINE_MARKER} marks a PARTIAL run (n < full test size)"
    fig.suptitle(f"Initial vs final quality -- arm {args.main_arm}\n{mode}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    files = save_figure(fig, Path(args.out_dir), "fig1_quality_init_vs_final", args.formats)
    plt.close(fig)

    rows = []
    for c in sorted(plotted, key=lambda x: (TASKS.index(x["task"]) if x["task"] in TASKS else 99,
                                            MODEL_ORDER.index(x["model"])
                                            if x["model"] in MODEL_ORDER else 99)):
        rows.append([
            f"{TASK_SHORT.get(c['task'], c['task']):7s}",
            f"{c['model']:12s}",
            f"n={c['n']} {c['status']}",
            _fmt(c['corpus_init'] if c['corpus_init'] is not None else c['init_mean']),
            _fmt(c['corpus_final'] if c['corpus_final'] is not None else c['final_mean']),
            f"{c['paired'].diff:+.3f}",
            f"[{c['paired'].ci_low:+.3f},{c['paired'].ci_high:+.3f}]",
            f"{c['p_holm']:.3g}",
            "corpus" if c["corpus_init"] is not None else "per-sample",
        ])
    header = ["task", "model", "n/status", "initial", "final",
              "paired delta", "95% CI", "Holm p", "metric source"]
    all_rows = rows
    report_figure("fig1_quality_init_vs_final", files, header, all_rows, log=log)
    csv_path = Path(args.out_dir) / "fig1_quality_init_vs_final.csv"
    write_rows(
        [
            {
                "task": c["task"], "model": c["model"], "arm": args.main_arm,
                "status": c["status"], "n": c["n"], "expected_n": c["run"].expected_n,
                "run": c["run"].rel,
                "init_mean_persample": round(c["init_mean"], 4),
                "final_mean_persample": round(c["final_mean"], 4),
                "init_ci_low": round(c["init_ci"][1], 4), "init_ci_high": round(c["init_ci"][2], 4),
                "final_ci_low": round(c["final_ci"][1], 4), "final_ci_high": round(c["final_ci"][2], 4),
                "paired_delta": round(c["paired"].diff, 4),
                "paired_ci_low": round(c["paired"].ci_low, 4),
                "paired_ci_high": round(c["paired"].ci_high, 4),
                "p_two_sided": round(c["paired"].p_two_sided, 5),
                "p_holm": round(c["p_holm"], 5),
                "corpus_init": "" if c["corpus_init"] is None else round(c["corpus_init"], 4),
                "corpus_final": "" if c["corpus_final"] is None else round(c["corpus_final"], 4),
                "primary_metric": c["primary"],
            }
            for c in plotted
        ],
        csv_path,
        ["task", "model", "arm", "status", "n", "expected_n", "run", "init_mean_persample",
         "final_mean_persample", "init_ci_low", "init_ci_high", "final_ci_low", "final_ci_high",
         "paired_delta", "paired_ci_low", "paired_ci_high", "p_two_sided", "p_holm",
         "corpus_init", "corpus_final", "primary_metric"],
    )
    return {"name": "fig1_quality_init_vs_final",
            "files": [str(p) for p in files + [csv_path]],
            "header": header, "rows": rows, "status": "ok",
            "cells": [
                {"task": c["task"], "model": c["model"], "n": c["n"], "status": c["status"],
                 "paired_delta": round(c["paired"].diff, 4),
                 "paired_ci": [round(c["paired"].ci_low, 4), round(c["paired"].ci_high, 4)],
                 "p_holm": round(c["p_holm"], 5),
                 "corpus_init": c["corpus_init"], "corpus_final": c["corpus_final"]}
                for c in plotted
            ]}


# --------------------------------------------------------------------------- #
# figure 2: Full minus strongest non-experience arm
# --------------------------------------------------------------------------- #


def figure_delta(args, runs: Sequence[RunInfo], baseline_arm: str, plt, log=print) -> dict:
    by_cell: Dict[Tuple[str, str], Dict[str, RunInfo]] = defaultdict(dict)
    for run in runs:
        if run.arm in (args.main_arm, baseline_arm):
            if args.complete_only and not run.complete:
                continue
            by_cell[(run.task, run.model)][run.arm] = run

    cells = []
    for (task, model) in sorted(by_cell, key=lambda k: (TASKS.index(k[0]) if k[0] in TASKS else 99,
                                                        MODEL_ORDER.index(k[1])
                                                        if k[1] in MODEL_ORDER else 99)):
        cell = by_cell[(task, model)]
        if args.main_arm not in cell or baseline_arm not in cell:
            continue
        a, b, shared = paired_values(cell[args.main_arm], cell[baseline_arm],
                                     "final_metric_offline")
        if not shared:
            continue
        label = f"{task}/{model}"
        res = paired_diff_ci(a, b, n_resamples=args.bootstrap, seed=_seed_for(label, args.seed))
        cells.append(
            {"task": task, "model": model, "label": label, "n": len(shared),
             "mean_full": res.mean_a, "mean_base": res.mean_b, "res": res,
             "full": cell[args.main_arm], "base": cell[baseline_arm],
             "status": ("partial" if not (cell[args.main_arm].complete
                                          and cell[baseline_arm].complete) else "complete")}
        )
    if not cells:
        log(f"[fig2] SKIPPED: no (task, model) cell has both {args.main_arm} and "
            f"{baseline_arm} run with data")
        return {"name": "fig2_full_minus_baseline", "files": [], "rows": [], "status": "skipped"}

    adjusted = holm_correct([c["res"].p_two_sided for c in cells])
    for c, p_adj in zip(cells, adjusted):
        c["p_holm"] = p_adj

    fig, ax = plt.subplots(figsize=(1.05 * len(cells) + 3.4, 4.4))
    xs = list(range(len(cells)))
    for x, c in zip(xs, cells):
        res = c["res"]
        sig = res.ci_low > 0 or res.ci_high < 0
        ax.bar(
            x, res.diff, width=0.6,
            color=("#2ca02c" if res.diff > 0 else "#d62728") if sig else "#bdbdbd",
            edgecolor="#111111" if sig else "#777777",
            linewidth=1.6 if sig else 0.8,
            hatch="" if sig else "//",
        )
        ax.errorbar(x, res.diff, yerr=[[res.diff - res.ci_low], [res.ci_high - res.diff]],
                    fmt="none", ecolor="#111111", elinewidth=1.3, capsize=4)
        ax.annotate(f"{res.diff:+.2f}\n[{res.ci_low:+.2f},{res.ci_high:+.2f}]",
                    (x, res.ci_high if res.ci_high > 0 else res.ci_low),
                    textcoords="offset points", xytext=(0, 8 if res.ci_high > 0 else -20),
                    ha="center", fontsize=7.5)
        if sig:
            ax.annotate("*", (x, res.diff), textcoords="offset points",
                        xytext=(0, 2 if res.diff > 0 else -12), ha="center", fontsize=16)
    ax.axhline(0.0, color="#333333", lw=0.9)
    lo = min(0.0, min(c["res"].ci_low for c in cells))
    hi = max(0.0, max(c["res"].ci_high for c in cells))
    pad = 0.30 * max(hi - lo, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{TASK_SHORT.get(c['task'], c['task'])}\n{c['model']}"
                        + (f"\n{BASELINE_MARKER}n={c['n']}" if c["status"] == "partial"
                           else f"\nn={c['n']}")
                        for c in cells], fontsize=8)
    ax.set_ylabel(f"paired Δ final per-sample metric\n({args.main_arm} - {baseline_arm})", fontsize=9)
    ax.set_title(f"{args.main_arm} minus {baseline_arm} (strongest non-experience arm)\n"
                 "solid + * = paired-bootstrap 95% CI excludes 0; hatched grey = includes 0; "
                 "Holm-corrected across cells", fontsize=9.5)
    fig.tight_layout()
    files = save_figure(fig, Path(args.out_dir), "fig2_full_minus_baseline", args.formats)
    plt.close(fig)

    rows = [[
        f"{TASK_SHORT.get(c['task'], c['task']):7s}",
        f"{c['model']:12s}",
        f"n={c['n']} {c['status']}",
        f"{c['mean_full']:.3f}",
        f"{c['mean_base']:.3f}",
        f"{c['res'].diff:+.3f}",
        f"[{c['res'].ci_low:+.3f},{c['res'].ci_high:+.3f}]",
        f"{'excludes0' if (c['res'].ci_low > 0 or c['res'].ci_high < 0) else 'includes0':9s}",
        f"{c['p_holm']:.3g}",
    ] for c in cells]
    header = ["task", "model", "n/status", "full", "baseline", "delta",
              "95% CI", "CI vs 0", "Holm p"]
    report_figure("fig2_full_minus_baseline", files, header, rows, log=log)
    csv_path = Path(args.out_dir) / "fig2_full_minus_baseline.csv"
    write_rows(
        [
            {
                "task": c["task"], "model": c["model"], "n": c["n"], "status": c["status"],
                "full_arm": args.main_arm, "baseline_arm": baseline_arm,
                "full_run": c["full"].rel, "baseline_run": c["base"].rel,
                "full_n": c["full"].n, "baseline_n": c["base"].n,
                "mean_final_full": round(c["mean_full"], 4),
                "mean_final_baseline": round(c["mean_base"], 4),
                "delta": round(c["res"].diff, 4),
                "ci_low": round(c["res"].ci_low, 4), "ci_high": round(c["res"].ci_high, 4),
                "ci_excludes_zero": int(c["res"].ci_low > 0 or c["res"].ci_high < 0),
                "p_two_sided": round(c["res"].p_two_sided, 5),
                "p_holm": round(c["p_holm"], 5),
            }
            for c in cells
        ],
        csv_path,
        ["task", "model", "n", "status", "full_arm", "baseline_arm", "full_run", "baseline_run",
         "full_n", "baseline_n", "mean_final_full", "mean_final_baseline", "delta", "ci_low",
         "ci_high", "ci_excludes_zero", "p_two_sided", "p_holm"],
    )
    return {"name": "fig2_full_minus_baseline",
            "files": [str(p) for p in files + [csv_path]],
            "header": header, "rows": rows, "status": "ok", "baseline_arm": baseline_arm,
            "cells": [
                {"task": c["task"], "model": c["model"], "n": c["n"], "status": c["status"],
                 "delta": round(c["res"].diff, 4),
                 "ci": [round(c["res"].ci_low, 4), round(c["res"].ci_high, 4)],
                 "ci_excludes_zero": bool(c["res"].ci_low > 0 or c["res"].ci_high < 0),
                 "p_holm": round(c["p_holm"], 5)}
                for c in cells
            ]}


# --------------------------------------------------------------------------- #
# figure 3: quality vs compute
# --------------------------------------------------------------------------- #


def figure_quality_vs_compute(args, runs: Sequence[RunInfo], corpus: Dict[str, dict],
                              plt, log=print) -> dict:
    """Final quality against mean total tokens per sample.

    One panel per task (so a panel never mixes BLEU / GLEU / ROUGE on one axis),
    one line per arm, one marker per model.  The y value is the *mean per-sample*
    final offline metric, uniformly for every point: it is the only metric every
    trace carries, and mixing corpus values with per-sample fallbacks inside one
    panel would not be comparable.  Corpus values per run live in the fig1 CSV.
    """
    points: List[dict] = []
    for run in runs:
        if args.complete_only and not run.complete:
            continue
        tps = tokens_per_sample(run)
        final_vals = metric_values(run, "final")
        if tps is None or not final_vals:
            if tps is None:
                log(f"[fig3] {run.rel}: cost.total_tokens missing in some record; point skipped")
            continue
        points.append(
            {"arm": run.arm, "task": run.task, "model": run.model, "run": run,
             "tokens": tps, "y": sum(final_vals) / len(final_vals), "n": run.n,
             "status": run.status, "src": "per-sample"}
        )
    if not points:
        log("[fig3] SKIPPED: no run with both cost.total_tokens and a final metric")
        return {"name": "fig3_quality_vs_compute", "files": [], "rows": [], "status": "skipped"}

    tasks = [t for t in args.tasks if any(p["task"] == t for p in points)]
    arms = sorted({p["arm"] for p in points}, key=lambda a: (0 if a == args.main_arm else 1, a))
    models = sorted({p["model"] for p in points},
                    key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    markers = {m: _MODEL_MARKERS[i % len(_MODEL_MARKERS)] for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, max(1, len(tasks)), figsize=(3.3 * max(1, len(tasks)) + 1.2, 4.6),
                             squeeze=False)
    for ax, task in zip(list(axes[0]), tasks):
        for arm in arms:
            pts = sorted([p for p in points if p["arm"] == arm and p["task"] == task],
                         key=lambda p: p["tokens"])
            if not pts:
                continue
            color = ARM_COLORS.get(arm, "#7f7f7f")
            ax.plot([p["tokens"] for p in pts], [p["y"] for p in pts], color=color,
                    lw=1.5, alpha=0.85, label=arm,
                    ls="--" if is_non_experience(pts[0]["run"]) else "-")
            for p in pts:
                ax.plot([p["tokens"]], [p["y"]], marker=markers[p["model"]], ms=7.5, color=color,
                        mfc=color if p["status"] == "complete" else "white", mew=1.4, ls="none")
                ax.annotate(f"{p['model']}\nn={p['n']}" + (BASELINE_MARKER if p["status"] != "complete" else ""),
                            (p["tokens"], p["y"]), textcoords="offset points",
                            xytext=(5, -10), fontsize=6.5, color="#333333")
        ax.set_title(f"{task}  ({PRIMARY_METRIC[task]})", fontsize=9)
        ax.set_xlabel("mean total tokens per sample", fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=7.5)
    axes[0][0].set_ylabel("mean per-sample final metric", fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, fontsize=8, title="arm", loc="upper right",
                   bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Quality vs compute -- one line per arm, one panel per task\n"
                 "filled marker = complete run, open marker = PARTIAL run "
                 f"({BASELINE_MARKER}); dashed line = non-experience arm; "
                 "y = mean per-sample final offline metric (sentence-BLEU proxy for wmt19)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.88, 0.90))
    files = save_figure(fig, Path(args.out_dir), "fig3_quality_vs_compute", args.formats)
    plt.close(fig)

    rows = [[
        p["arm"], p["model"], TASK_SHORT.get(p["task"], p["task"]),
        f"n={p['n']} {p['status']}",
        f"{p['tokens']:.1f}", f"{p['y']:.3f}",
    ] for p in sorted(points, key=lambda p: (p["task"], p["arm"], p["model"]))]
    header = ["arm", "model", "task", "n/status", "tokens/sample", "final(per-sample)"]
    report_figure("fig3_quality_vs_compute", files, header, rows, log=log)
    csv_path = Path(args.out_dir) / "fig3_quality_vs_compute.csv"
    write_rows(
        [
            {"arm": p["arm"], "model": p["model"], "task": p["task"], "status": p["status"],
             "n": p["n"], "expected_n": p["run"].expected_n, "run": p["run"].rel,
             "tokens_per_sample": round(p["tokens"], 2),
             "final_mean_persample": round(p["y"], 4),
             "corpus_final": round(float(corpus[p["run"].rel]["corpus_final"]), 4)
             if corpus.get(p["run"].rel, {}).get("corpus_final") is not None else "",
             "metric_source": "per-sample (uniform)"}
            for p in points
        ],
        csv_path,
        ["arm", "model", "task", "status", "n", "expected_n", "run", "tokens_per_sample",
         "final_mean_persample", "corpus_final", "metric_source"],
    )
    return {"name": "fig3_quality_vs_compute", "files": [str(p) for p in files + [csv_path]],
            "header": header, "rows": rows, "status": "ok",
            "points": [
                {"arm": p["arm"], "model": p["model"], "task": p["task"], "n": p["n"],
                 "status": p["status"], "tokens_per_sample": round(p["tokens"], 2),
                 "final_mean_persample": round(p["y"], 4)}
                for p in points
            ]}


# --------------------------------------------------------------------------- #
# figure 4: unnecessary refinement rate
# --------------------------------------------------------------------------- #


def unnecessary_refinement(run: RunInfo) -> dict:
    """worse / n_refine_scored -- identical convention to run_stopping_diagnostics."""
    n_refine = worse = same = better = 0
    for rec in run.records:
        for rd in rec.get("rounds", []):
            if str(rd.get("controller_action", "")).upper() != "REFINE":
                continue
            n_refine += 1
            delta = rd.get("delta_offline")
            if delta is None:
                continue
            d = float(delta)
            if d < 0:
                worse += 1
            elif d > 0:
                better += 1
            else:
                same += 1
    scored = worse + same + better
    return {"n_refine": n_refine, "scored": scored, "worse": worse, "same": same,
            "better": better, "rate": (worse / scored) if scored else None}


def figure_unnecessary_refinement(args, runs: Sequence[RunInfo], plt, log=print) -> dict:
    by_arm: Dict[str, List[RunInfo]] = defaultdict(list)
    for run in runs:
        if args.arms and run.arm not in args.arms:
            continue
        if run.task not in args.tasks:
            continue
        if args.complete_only and not run.complete:
            continue
        by_arm[run.arm].append(run)
    by_arm = {a: rs for a, rs in by_arm.items() if any(unnecessary_refinement(r)["scored"] for r in rs)}
    if not by_arm:
        log("[fig4] SKIPPED: no run has a scored REFINE round (rate undefined)")
        return {"name": "fig4_unnecessary_refinement", "files": [], "rows": [], "status": "skipped"}

    arms = sorted(by_arm, key=lambda a: (0 if a == args.main_arm else 1, a))
    models = sorted({r.model for rs in by_arm.values() for r in rs},
                    key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    tasks = [t for t in args.tasks]

    fig, axes = plt.subplots(len(arms), 1, figsize=(9.0, 3.0 * len(arms) + 1.0), squeeze=False)
    rows: List[List[str]] = []
    csv_rows: List[dict] = []
    for ax, arm in zip([a[0] for a in axes], arms):
        xs = list(range(len(tasks)))
        width = 0.8 / max(1, len(models))
        for k, model in enumerate(models):
            offs = [x - 0.4 + width * (k + 0.5) for x in xs]
            for x, task in zip(offs, tasks):
                run = next((r for r in by_arm[arm] if r.task == task and r.model == model), None)
                stats = unnecessary_refinement(run) if run is not None else None
                rate = stats["rate"] if stats else None
                val = 0.0 if rate is None else 100.0 * rate
                bar = ax.bar([x], [val], width=width * 0.92,
                             color=ARM_COLORS.get(arm, "#7f7f7f"),
                             alpha=min(1.0, 0.5 + 0.14 * k),
                             edgecolor="#333333", lw=0.5)
                if run is None or stats is None or rate is None:
                    ax.text(x, 2.0, "no data", ha="center", fontsize=6, rotation=90,
                            color="#888888")
                    continue
                if not run.complete:
                    bar[0].set_hatch("///")
                ax.text(x, min(val + 1.5, 118.0),
                        f"{100 * rate:.1f}%\n{stats['worse']}/{stats['scored']}",
                        ha="center", fontsize=6.5)
                rows.append([
                    arm, TASK_SHORT.get(task, task), model,
                    f"n={run.n} {run.status}",
                    f"{stats['worse']}/{stats['scored']}",
                    f"{100 * rate:.1f}%",
                ])
                csv_rows.append(
                    {"arm": arm, "task": task, "model": model, "status": run.status,
                     "n_samples": run.n, "expected_n": run.expected_n, "run": run.rel,
                     "n_refine": stats["n_refine"], "n_refine_scored": stats["scored"],
                     "worse": stats["worse"], "same": stats["same"], "better": stats["better"],
                     "unnecessary_rate": round(rate, 6),
                     "useless_rate": "" if not stats["scored"] else
                     round((stats["worse"] + stats["same"]) / stats["scored"], 6)}
                )
        ax.set_xticks(xs)
        ax.set_xticklabels([TASK_SHORT.get(t, t) for t in tasks], fontsize=9)
        ax.set_xlim(-0.5, len(tasks) - 0.5)
        ax.set_ylabel("unnecessary\nrefinement %", fontsize=8)
        ax.set_ylim(0, 130)
        ax.set_title(f"arm: {arm}", fontsize=9.5)
        handles = [plt.Rectangle((0, 0), 1, 1, fc=ARM_COLORS.get(arm, "#7f7f7f"),
                                 alpha=min(1.0, 0.5 + 0.14 * k), ec="#333333", lw=0.5)
                   for k in range(len(models))]
        ax.legend(handles, models, fontsize=7, title="model", ncol=min(len(models), 4))
    fig.suptitle("Unnecessary refinement rate = REFINE rounds whose candidate scored WORSE "
                 "than the answer it replaced\n(offline per-sample metric; convention of "
                 f"run_stopping_diagnostics.py; {BASELINE_MARKER} hatch = PARTIAL run)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    files = save_figure(fig, Path(args.out_dir), "fig4_unnecessary_refinement", args.formats)
    plt.close(fig)

    header = ["arm", "task", "model", "n/status", "worse/scored", "unnecessary rate"]
    report_figure("fig4_unnecessary_refinement", files, header, rows, log=log)
    csv_path = Path(args.out_dir) / "fig4_unnecessary_refinement.csv"
    write_rows(csv_rows, csv_path,
               ["arm", "task", "model", "status", "n_samples", "expected_n", "run", "n_refine",
                "n_refine_scored", "worse", "same", "better", "unnecessary_rate", "useless_rate"])
    return {"name": "fig4_unnecessary_refinement", "files": [str(p) for p in files + [csv_path]],
            "header": header, "rows": rows, "status": "ok", "cells": csv_rows}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="make_plots.py",
        description="Phase 9: publication figures from the run traces (CPU only, no GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Statistics are imported from core/stats.py (paired bootstrap + Holm); the "
               "completeness rule and baseline choice are shared with export_human_eval.py.",
    )
    ap.add_argument("--runs", nargs="*", default=list(DEFAULT_RUN_DIRS),
                    help="Run directories to scan.")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), help="Tasks to plot.")
    ap.add_argument("--models", nargs="*", default=None, help="Models to plot (default: all found).")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="Restrict figure 4 to these arms (default: all arms found).")
    ap.add_argument("--main-arm", default="full_static", help="The Full/experience arm.")
    ap.add_argument("--baseline-arm", default="auto",
                    help="Strongest non-experience arm, or 'auto' (ranked from the traces).")
    ap.add_argument("--out-dir", default=str(EXP_ROOT / "plots"), help="Figure output directory.")
    ap.add_argument("--formats", nargs="*", default=list(FIGURE_FORMATS),
                    help="Figure formats to write.")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Paired-bootstrap resamples.")
    ap.add_argument("--seed", type=int, default=42, help="Bootstrap seed base.")
    ap.add_argument("--complete-only", action="store_true",
                    help="Drop partial runs instead of labelling them.")
    ap.add_argument("--split", default="test", help="Trace split to use.")
    ap.add_argument("--allow-tagged", action="store_true",
                    help="Include traces under sweep tags (dev/gate).")
    ap.add_argument("--drop-last-line", action="store_true",
                    help="Also drop the final (terminated) line of every trace file.")
    ap.add_argument("--corpus-cache", default=str(EXP_ROOT / CORPUS_CACHE_DEFAULT),
                    help="JSON cache of corpus metrics (optional speed-up, trace-derived).")
    ap.add_argument("--refresh-corpus", action="store_true",
                    help="Recompute the corpus cache from the traces and exit (or plot).")
    ap.add_argument("--no-figures", action="store_true",
                    help="Do everything except import matplotlib / draw (for --refresh-corpus).")
    ap.add_argument("--no-corpus", action="store_true",
                    help="Never recompute corpus metrics; use only the cache (per-sample fallback).")
    return ap.parse_args(argv)


def check_matplotlib() -> Tuple[object, str]:
    if importlib.util.find_spec("matplotlib") is None:
        return None, f"matplotlib is not importable from {sys.executable}"
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend, never opens a window
        import matplotlib.pyplot as plt

        return plt, f"matplotlib {matplotlib.__version__} (Agg) from {sys.executable}"
    except Exception as exc:  # pragma: no cover
        return None, f"matplotlib import failed: {type(exc).__name__}: {exc}"


def print_inventory(runs: Sequence[RunInfo], log=print) -> None:
    log("")
    log(f"[data] {len(runs)} trace file(s) in scope "
        f"(split={sorted({r.split for r in runs})}, "
        f"tagged={'yes' if any(r.tag for r in runs) else 'no'})")
    log(f"  {'arm':16s}{'task':14s}{'model':12s}{'n':>6s}{'expected':>9s}  status   file")
    for run in sorted(runs, key=lambda r: (r.arm, r.task, r.model, r.rel)):
        log(f"  {run.arm:16s}{run.task:14s}{run.model:12s}{run.n:>6d}{run.expected_n:>9d}"
            f"  {run.status:8s} {run.rel}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    plt, mpl_msg = (None, "not needed (--no-figures)") if args.no_figures else check_matplotlib()
    if plt is None and not args.no_figures:
        print(f"[FATAL] {mpl_msg}")
        print("  matplotlib is required for the figures.  It is NOT installed in")
        print("  /home/ymb/miniconda3/envs/qwen35/bin/python (that env has sacrebleu/nltk/"
              "rouge_score instead);")
        print("  it IS available in /home/ymb/miniconda3/envs/llama4/bin/python.  Run:")
        print("    /home/ymb/miniconda3/envs/qwen35/bin/python make_plots.py "
              "--refresh-corpus --no-figures   # corpus cache (needs sacrebleu)")
        print("    /home/ymb/miniconda3/envs/llama4/bin/python make_plots.py   # figures")
        print("  Nothing is installed automatically.")
        return 2

    runs_all = discover_runs(
        args.runs, models=args.models, tasks=args.tasks, splits=[args.split],
        allow_tagged=args.allow_tagged, force_drop_last=args.drop_last_line,
    )
    runs = sorted(group_runs(runs_all).values(), key=lambda r: r.rel)
    if not runs:
        print(f"[FATAL] no usable traces under {args.runs} "
              f"(tasks={args.tasks}, models={args.models}, split={args.split})")
        return 2
    print_inventory(runs)

    # ---- corpus metrics ---------------------------------------------------
    cache_path = Path(args.corpus_cache)
    cache = load_corpus_cache(cache_path)
    ok, why = scorer_available()
    print(f"\n[corpus] shared Scorer: {why}")
    print(f"[corpus] cache: {cache_path} ({len(cache)} entr(ies) loaded)")
    allow_compute = ok and not args.no_corpus
    corpus: Dict[str, dict] = {}
    for run in runs:
        entry = corpus_for(run, cache, allow_compute=(allow_compute or args.refresh_corpus))
        if entry:
            corpus[run.rel] = entry
    missing = [r.rel for r in runs if r.rel not in corpus]
    if missing:
        print(f"[corpus] unavailable for {len(missing)} run(s) -> per-sample fallback: {missing}")
    if args.refresh_corpus:
        write_corpus_cache(
            cache_path, cache,
            {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
             "generator": "make_plots.py --refresh-corpus",
             "interpreter": sys.executable, "scorer": why,
             "note": "derived from runs/**/*.jsonl + data/manifests via core.scoring.Scorer"},
        )
        print(f"[corpus] wrote {cache_path} ({len(cache)} run entries)")
        if args.no_figures:
            print("[done] corpus cache refreshed; no figure drawn (--no-figures)")
            return 0

    # ---- baseline choice (same rule as the human-eval export) -------------
    baseline_arm = args.baseline_arm
    choice = BaselineChoice(None, [], "explicit")
    if baseline_arm in (None, "", "auto"):
        choice = choose_baseline_arm(runs, args.main_arm, args.tasks,
                                     args.models or sorted({r.model for r in runs}))
        print_baseline_evidence(choice)
        baseline_arm = choice.arm
        if baseline_arm is None:
            print("[fig2] no non-experience arm with a COMPLETE run -> figure 2 will be skipped")
    else:
        print(f"[baseline] explicit --baseline-arm {baseline_arm!r}")
    if args.models is None:
        args.models = sorted({r.model for r in runs},
                             key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)

    print(f"\n[figures] interpreter: {sys.executable}")
    print(f"[figures] {mpl_msg}")
    results = []
    results.append(figure_quality(args, runs, corpus, plt))
    if baseline_arm:
        results.append(figure_delta(args, runs, baseline_arm, plt))
    else:
        results.append({"name": "fig2_full_minus_baseline", "files": [], "rows": [],
                        "status": "skipped (no baseline arm)"})
    results.append(figure_quality_vs_compute(args, runs, corpus, plt))
    results.append(figure_unnecessary_refinement(args, runs, plt))

    # ---- manifest ---------------------------------------------------------
    manifest = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "interpreter": sys.executable,
        "matplotlib": mpl_msg,
        "scorer": why,
        "corpus_cache": str(cache_path),
        "baseline_arm": baseline_arm,
        "baseline_selection": choice.reason,
        "main_arm": args.main_arm,
        "bootstrap_resamples": args.bootstrap,
        "seed": args.seed,
        "complete_only": bool(args.complete_only),
        "runs": [
            {"path": r.rel, "arm": r.arm, "task": r.task, "model": r.model, "n": r.n,
             "expected_n": r.expected_n, "status": r.status,
             "corpus_metric": corpus.get(r.rel, {}).get("primary", ""),
             "corpus_init": corpus.get(r.rel, {}).get("corpus_init", ""),
             "corpus_final": corpus.get(r.rel, {}).get("corpus_final", "")}
            for r in runs
        ],
        "figures": results,
    }
    manifest_path = Path(args.out_dir) / "figures_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    print("")
    print("=== summary ===")
    for res in results:
        state = res["status"]
        files = ", ".join(res["files"]) if res["files"] else "-"
        print(f"  {res['name']:32s} {state:28s} {files}")
        rows = list(res.get("rows", []))
        if rows:
            hdr = res.get("header") or [f"col{i}" for i in range(len(rows[0]))]
            for line in align_table(hdr, rows):
                print("      " + line.strip())
    print(f"  {'manifest':32s} {'ok':28s} {manifest_path}")
    n_skip = sum(1 for r in results if r["status"] != "ok")
    if n_skip:
        print(f"[done] {len(results) - n_skip}/{len(results)} figures produced; "
              f"{n_skip} skipped because the data does not exist yet (stated above)")
    else:
        print(f"[done] {len(results)}/{len(results)} figures produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
