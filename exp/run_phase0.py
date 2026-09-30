#!/usr/bin/env python
"""Phase 0: environment assertion, motivation table, unified baseline rescoring.

All three are GPU-free except for the (optional) reproduction gate in
``repro_gate.py``.

Outputs
-------
* ``reports/phase0_env.json``            -- pinned versions + GPU whitelist check
* ``scores/motivation_steps.csv``        -- per-step improve/degrade/tie rates
* ``scores/baseline_rescored.csv``       -- all 220 baseline runs, one scorer
* ``scores/baseline_rescored_summary.csv``
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    EXP_ROOT,
    MODELS,
    PINNED_VERSIONS,
    TASKS,
    TEST_SAMPLES,
    baseline_results_dir,
)
from core.scoring import PRIMARY_METRIC, Scorer  # noqa: E402

EPS = 1e-4

# --------------------------------------------------------------------------- #
# 0a. environment
# --------------------------------------------------------------------------- #


def env_report() -> dict:
    rep = {"versions": {}, "pinned_ok": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for mod, want in PINNED_VERSIONS.items():
        try:
            m = importlib.import_module(mod)
            got = getattr(m, "__version__", "?")
        except Exception as e:  # pragma: no cover
            got = f"IMPORT_ERROR:{type(e).__name__}"
        rep["versions"][mod] = got
        if got != want:
            rep["pinned_ok"] = False
    try:
        import torch

        rep["cuda_available"] = bool(torch.cuda.is_available())
        rep["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as e:  # pragma: no cover
        rep["cuda_available"] = False
        rep["cuda_error"] = f"{type(e).__name__}: {e}"
    return rep


# --------------------------------------------------------------------------- #
# 0c. motivation table
# --------------------------------------------------------------------------- #


def motivation_table() -> List[dict]:
    """Per-step refinement outcome rates from the existing baseline traces."""
    from baseline_core.metrics import task_metric  # baseline scorer, comparable to stored traces
    from baseline_core.tasks import get_adapter

    adapters: Dict[str, object] = {}

    def parse(task: str, text: str) -> str:
        if task not in adapters:
            adapters[task] = get_adapter(task)
        return adapters[task].parse_output(text)

    root = baseline_results_dir()
    rows: List[dict] = []
    for task in TASKS:
        n = TEST_SAMPLES[task]
        for method in ("SR-Fixed", "SelfRefine-Fixed"):
            imp = deg = tie = 0
            net = 0.0
            tot = 0
            for model in MODELS:
                path = root / task / model / f"{method}.jsonl"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as f:
                    for k, line in enumerate(f):
                        if k >= n:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        cands = [parse(task, c) for c in d.get("candidates", [])]
                        if len(cands) < 2:
                            continue
                        mets = [task_metric(task, d["reference_text"], c) for c in cands]
                        for i in range(len(mets) - 1):
                            delta = mets[i + 1] - mets[i]
                            tot += 1
                            net += delta
                            if delta > EPS:
                                imp += 1
                            elif delta < -EPS:
                                deg += 1
                            else:
                                tie += 1
            if tot:
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "steps": tot,
                        "improve_rate": imp / tot,
                        "degrade_rate": deg / tot,
                        "tie_rate": tie / tot,
                        "net_delta_per_step": net / tot,
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# 0c-bis. oracle-stop upper bound
# --------------------------------------------------------------------------- #
def oracle_stop_bound() -> Tuple[List[dict], List[dict]]:
    """How much quality a perfect stop/select rule could have recovered.

    The motivation table shows what an *average* refinement step does.  This
    computes the complementary bound: for every baseline refinement chain, what
    would the score have been had an oracle stopped at the best step instead of
    running the chain to its end?  The gap to the chain's final answer is the
    headroom that better *stopping* -- rather than better *revising* -- could
    recover, which is exactly the quantity this method targets.

    Uses the baseline's own scorer on the baseline's own traces, so the bound is
    stated in the same units as the motivation table.  No new model, no new
    metric, GPU-free.
    """
    from baseline_core.metrics import task_metric
    from baseline_core.tasks import get_adapter

    adapters: Dict[str, object] = {}

    def parse(task: str, text: str) -> str:
        if task not in adapters:
            adapters[task] = get_adapter(task)
        return adapters[task].parse_output(text)

    root = baseline_results_dir()
    per_sample: List[dict] = []
    summary: List[dict] = []
    for task in TASKS:
        n = TEST_SAMPLES[task]
        for method in ("SR-Fixed", "SelfRefine-Fixed"):
            for model in MODELS:
                path = root / task / model / f"{method}.jsonl"
                if not path.exists():
                    continue
                init_v, final_v, best_v, steps_used, wasted = [], [], [], [], 0
                with path.open(encoding="utf-8") as f:
                    for k, line in enumerate(f):
                        if k >= n:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        cands = [parse(task, c) for c in d.get("candidates", [])]
                        if len(cands) < 2:
                            continue
                        mets = [task_metric(task, d["reference_text"], c) for c in cands]
                        first, last, best = mets[0], mets[-1], max(mets)
                        best_step = mets.index(best)
                        init_v.append(first)
                        final_v.append(last)
                        best_v.append(best)
                        steps_used.append(len(mets) - 1)
                        # steps taken after the best one are pure waste to an oracle
                        wasted += len(mets) - 1 - best_step
                        per_sample.append({
                            "task": task, "method": method, "model": model,
                            "sample_index": k, "n_steps": len(mets) - 1,
                            "initial": round(first, 4), "final": round(last, 4),
                            "oracle_best": round(best, 4), "best_step": best_step,
                        })
                if not init_v:
                    continue
                m = len(init_v)
                summary.append({
                    "task": task, "method": method, "model": model, "n": m,
                    "mean_initial": sum(init_v) / m,
                    "mean_final": sum(final_v) / m,
                    "mean_oracle_best": sum(best_v) / m,
                    # headroom a perfect stop rule leaves on the table today
                    "oracle_gain_over_final": (sum(best_v) - sum(final_v)) / m,
                    # how far the chain can wander (motivation for stopping at all)
                    "oracle_gain_over_initial": (sum(best_v) - sum(init_v)) / m,
                    "mean_steps": sum(steps_used) / m,
                    "mean_wasted_steps_after_best": wasted / m,
                })
    return per_sample, summary


def print_oracle_bound(summary: List[dict]) -> None:
    print(f"{'task':14s}{'method':18s}{'model':12s}{'n':>6s}"
          f"{'init':>9s}{'final':>9s}{'oracle':>9s}{'gain/final':>12s}{'waste steps':>13s}")
    for r in summary:
        print(f"{r['task']:14s}{r['method']:18s}{r['model']:12s}{r['n']:6d}"
              f"{r['mean_initial']:9.2f}{r['mean_final']:9.2f}{r['mean_oracle_best']:9.2f}"
              f"{r['oracle_gain_over_final']:+12.3f}{r['mean_wasted_steps_after_best']:13.2f}")
    by_task: Dict[str, List[float]] = defaultdict(list)
    for r in summary:
        by_task[r["task"]].append(r["oracle_gain_over_final"])
    print()
    for task, vals in sorted(by_task.items()):
        print(f"  {task:14s} 平均 oracle 增益 = {sum(vals) / len(vals):+.3f}")


# --------------------------------------------------------------------------- #
# 0d. unified rescoring of every stored baseline run
# --------------------------------------------------------------------------- #


def rescore_baselines() -> Tuple[List[dict], List[dict]]:
    root = baseline_results_dir()
    per_run: List[dict] = []
    for task in TASKS:
        n = TEST_SAMPLES[task]
        scorer = Scorer(task)
        primary = PRIMARY_METRIC[task]
        for model in MODELS:
            model_dir = root / task / model
            if not model_dir.is_dir():
                continue
            for csv_path in sorted(model_dir.glob("*.csv")):
                method = csv_path.stem
                with csv_path.open(encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))[:n]
                if not rows:
                    continue
                pairs = [(r.get("reference_text", ""), r.get("final_output", "")) for r in rows]
                metrics = scorer.score_corpus(pairs)
                costs = {
                    "avg_total_tokens": _mean(rows, "total_tokens"),
                    "avg_sequential_tokens": _mean(rows, "sequential_tokens"),
                    "avg_total_latency_s": _mean(rows, "total_latency_s"),
                    "avg_sequential_latency_s": _mean(rows, "sequential_latency_s"),
                    "avg_num_llm_calls": _mean(rows, "num_llm_calls"),
                }
                per_run.append(
                    {
                        "task": task,
                        "model": model,
                        "method": method,
                        "n": len(rows),
                        "primary_metric": primary,
                        "primary_value": metrics.get(primary, float("nan")),
                        **{k: round(v, 6) for k, v in metrics.items()},
                        **{k: round(v, 4) for k, v in costs.items()},
                    }
                )
    return per_run, _summarize(per_run)


def _mean(rows: List[dict], key: str) -> float:
    vals = []
    for r in rows:
        v = r.get(key)
        if v in (None, "", "None"):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    return sum(vals) / len(vals) if vals else float("nan")


def _summarize(per_run: List[dict]) -> List[dict]:
    grp: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in per_run:
        grp[(r["task"], r["method"])].append(r)
    out = []
    for (task, method), rs in sorted(grp.items()):
        out.append(
            {
                "task": task,
                "method": method,
                "n_models": len(rs),
                "primary_metric": rs[0]["primary_metric"],
                "mean_primary": sum(r["primary_value"] for r in rs) / len(rs),
                "mean_sequential_latency_s": sum(r["avg_sequential_latency_s"] for r in rs) / len(rs),
                "mean_total_tokens": sum(r["avg_total_tokens"] for r in rs) / len(rs),
                "mean_num_llm_calls": sum(r["avg_num_llm_calls"] for r in rs) / len(rs),
            }
        )
    return out


def _write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Different tasks emit different metric keys (bleu / gleu / rouge1...), so
    # the header must be the union of every row's keys, in first-seen order.
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    print("=" * 78)
    print("Phase 0a: environment")
    print("=" * 78)
    env = env_report()
    print(json.dumps(env, indent=2))
    (EXP_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (EXP_ROOT / "reports" / "phase0_env.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    print("=" * 78)
    print("Phase 0c: per-step refinement outcome rates (existing traces)")
    print("=" * 78)
    mot = motivation_table()
    _write_csv(mot, EXP_ROOT / "scores" / "motivation_steps.csv")
    print(f"{'task':14s}{'method':18s}{'steps':>8s}{'improve':>9s}{'degrade':>9s}{'tie':>8s}{'net dQ':>10s}")
    for r in mot:
        print(
            f"{r['task']:14s}{r['method']:18s}{r['steps']:8d}"
            f"{r['improve_rate']:9.1%}{r['degrade_rate']:9.1%}{r['tie_rate']:8.1%}"
            f"{r['net_delta_per_step']:+10.4f}"
        )

    print()
    print("=" * 78)
    print("Phase 0c-bis: oracle-stop upper bound (best step vs chain end)")
    print("=" * 78)
    os_per_sample, os_summary = oracle_stop_bound()
    _write_csv(os_summary, EXP_ROOT / "scores" / "oracle_stop_bound.csv")
    _write_csv(os_per_sample, EXP_ROOT / "scores" / "oracle_stop_bound_per_sample.csv")
    print_oracle_bound(os_summary)
    print(f"\n{len(os_summary)} chains -> scores/oracle_stop_bound.csv")

    print()
    print("=" * 78)
    print("Phase 0d: unified rescoring of stored baselines")
    print("=" * 78)
    per_run, summary = rescore_baselines()
    _write_csv(per_run, EXP_ROOT / "scores" / "baseline_rescored.csv")
    _write_csv(summary, EXP_ROOT / "scores" / "baseline_rescored_summary.csv")
    print(f"{len(per_run)} runs rescored -> scores/baseline_rescored.csv")
    print()
    print(f"{'task':14s}{'method':18s}{'models':>7s}{'metric':>9s}{'value':>9s}{'calls':>7s}{'lat_s':>8s}")
    for r in summary:
        print(
            f"{r['task']:14s}{r['method']:18s}{r['n_models']:7d}"
            f"{r['primary_metric']:>9s}{r['mean_primary']:9.2f}"
            f"{r['mean_num_llm_calls']:7.2f}{r['mean_sequential_latency_s']:8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
