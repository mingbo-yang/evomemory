#!/usr/bin/env python
"""Phase 7: stopping diagnostics -- did the controller stop at the right time?

Two directions are measured, both against the traces under ``runs/**/*.jsonl``:

1. **Unnecessary refinement rate** -- the GPU-free default.  Over every REFINE
   decision in a trace, the share whose candidate came out *worse* than the
   answer it replaced under the offline reference metric.  The trace already
   stores ``delta_offline`` (candidate metric minus current metric) per round, so
   this needs no new generation at all.  ``--recompute`` re-derives the same
   numbers with the shared ``Scorer`` and reports how far they disagree with the
   stored values; no new metric is introduced.
2. **Premature stop rate** -- ``--counterfactual``, generation, OFF by default.
   A STOP decision leaves no candidate behind, so nothing can be concluded from
   the trace alone.  This mode forces *one extra refinement* at sampled STOP
   states and scores it:

   * the experience block is re-rendered from the ``exp_ids`` the controller
     actually saw at that round (no re-retrieval, so the state is exact),
   * the instruction is produced by the same ``Controller`` in ``stop_mode=
     "fixed"`` -- the mode that removes STOP from the schema,
   * the candidate is generated through ``build_refine_prompt`` with the seed
     the pipeline would have used for that round,
     ``call_seed(sample_id, round, "refine")``,
   * the transition is judged by the same ``PairwiseJudge`` (same deterministic
     A/B order and seeds) and scored by the same ``Scorer``.

   A STOP state counts as *premature* when that extra round would have improved
   the offline reference metric.  Whether the candidate would also have been
   accepted by the reference-free judge is reported separately, because the two
   can disagree.

The counterfactual is capped by ``--max-stop-states`` (default 100) and sampled
round-robin across (task, model, arm) groups so no single run dominates.  It is
the only part of this script that needs a GPU; the default mode imports neither
torch nor vLLM.

Usage
-----
    # GPU-free diagnostics over the main traces
    python run_stopping_diagnostics.py --runs runs/main

    # everything under runs/ (the default), including dev/gate traces
    python run_stopping_diagnostics.py

    # force one extra refinement at up to 100 STOP states (needs a free GPU)
    python run_stopping_diagnostics.py --runs runs/main --counterfactual \
        --max-stop-states 100 --gpu 3

    # show the plan (including how many STOP states a cap would cover) and exit
    python run_stopping_diagnostics.py --runs runs/main --counterfactual --dry-run

Outputs
-------
    scores/stopping_diagnostics.csv                per-run / per-arm / per-task-model rows
    scores/stopping_diagnostics_counterfactual.csv one row per counterfactual STOP state
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import EXP_ROOT, MODELS, TASKS, ensure_gpu_whitelist  # noqa: E402
from core.experience import experience_dir, load_experiences  # noqa: E402
from core.manifest import SampleRef, manifest_path, read_manifest  # noqa: E402
from core.scoring import Scorer  # noqa: E402
from core.stats import paired_bootstrap  # noqa: E402

SCORES_DIR = EXP_ROOT / "scores"

#: Columns printed in the readable tables.
_TABLE_COLS = ("n_refine", "worse", "same", "better", "unnec", "useless", "accept", "mean_delta")

_CSV_FIELDS = [
    "scope", "arm", "tag", "task", "model", "seed", "split", "source",
    "n_samples", "n_rounds", "n_refine", "n_refine_scored", "n_stop", "n_other",
    "n_delta_missing", "worse", "same", "better", "identical_text",
    "unnecessary_rate", "useless_rate", "useful_rate", "accept_rate",
    "judge_worse", "judge_better", "judge_tie", "judge_uncertain",
    "useful_but_rejected", "worse_but_accepted", "mean_delta",
    "metric_mismatch_max",
]

_CF_FIELDS = [
    "task", "model", "arm", "tag", "split", "sample_id", "round_index", "source",
    "current_metric", "cf_metric", "cf_delta", "premature", "judge_verdict",
    "judge_order_consistent", "would_accept", "empty_candidate",
    "n_experiences", "n_missing_experiences", "instruction",
    "input_tokens", "output_tokens", "n_calls",
]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_traces(path: Path) -> Tuple[List[dict], int]:
    """Read a trace JSONL, skipping records that are not sample traces.

    ``bad`` counts unparseable lines: the production runners append to these
    files while they run, so a partially flushed last line is normal and must not
    abort a diagnostic pass.
    """
    out: List[dict] = []
    bad = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                bad += 1
                continue
            if isinstance(rec, dict) and isinstance(rec.get("rounds"), list):
                out.append(rec)
            else:
                bad += 1
    return out, bad


def discover(
    run_dirs: Sequence[Path],
    models: Sequence[str],
    tasks: Sequence[str],
    arms: Sequence[str],
) -> List[dict]:
    """One record per trace file, in a stable order."""
    found: List[dict] = []
    for base in run_dirs:
        if not base.exists():
            print(f"[WARN] run dir does not exist: {base}")
            continue
        for jsonl in sorted(base.rglob("*.jsonl")):
            recs, bad = load_traces(jsonl)
            if not recs:
                if bad:
                    print(f"[WARN] no usable records in {jsonl} ({bad} bad lines)")
                continue
            task, model = recs[0].get("task", ""), recs[0].get("model", "")
            if task not in tasks or model not in models:
                continue
            arm = jsonl.stem
            if arms and arm not in arms:
                continue
            # a sweep writes into .../<split>/<tag>/<arm>.jsonl; recover the tag
            parent = jsonl.parent.name
            tag = "" if parent in ("test", "dev", "accumulation") else parent
            found.append(
                {
                    "path": jsonl,
                    "tag": tag,
                    "arm": arm,
                    "task": task,
                    "model": model,
                    "seed": recs[0].get("seed"),
                    "config": recs[0].get("config", {}) or {},
                    "records": recs,
                    "bad_lines": bad,
                }
            )
    return found


def split_of(rec: dict) -> str:
    """The manifest split a trace belongs to (sample ids are ``task/split/idx``)."""
    for tr in rec["records"]:
        parts = str(tr.get("sample_id", "")).split("/")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    parent = rec["path"].parent.name
    return parent if parent in ("test", "dev", "accumulation") else "test"


_REF_CACHE: Dict[Tuple[str, str], Dict[str, SampleRef]] = {}


def refs_for(task: str, split: str) -> Dict[str, SampleRef]:
    """sample_id -> manifest record (source/reference/row index) for that split."""
    key = (task, split)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    path = manifest_path(task, split)
    refs: Dict[str, SampleRef] = {}
    if path.exists():
        refs = {r.sample_id: r for r in read_manifest(path)}
    else:
        print(f"[WARN] no manifest for {task}/{split} ({path}); references unavailable")
    _REF_CACHE[key] = refs
    return refs


def reconstruct_current(trace: dict, round_index: int) -> str:
    """The answer the controller was looking at in ``round_index``.

    The pipeline only advances the state on an accepted candidate, so replaying
    the acceptance flags reconstructs the exact state of that round.
    """
    current = trace.get("initial_draft") or ""
    for rd in trace.get("rounds", []):
        if int(rd.get("round_index", 0)) >= round_index:
            break
        if rd.get("controller_action") == "REFINE" and rd.get("accepted") and rd.get("candidate"):
            current = rd["candidate"]
    return current


# --------------------------------------------------------------------------- #
# direction 1: unnecessary refinement (GPU-free)
# --------------------------------------------------------------------------- #


def _classify(delta: float) -> str:
    if delta < 0.0:
        return "worse"
    if delta > 0.0:
        return "better"
    return "same"


def analyze_run(rec: dict, limit: Optional[int], recompute: bool) -> dict:
    """Counters for one trace file (one arm x tag x task x model x seed)."""
    task = rec["task"]
    scorer = Scorer(task) if recompute else None
    split = split_of(rec)
    refs = refs_for(task, split) if recompute else {}

    c: Counter = Counter()
    delta_sum = 0.0
    mismatch_max = 0.0
    samples = rec["records"][:limit] if limit else rec["records"]
    for tr in samples:
        sid = tr.get("sample_id", "")
        sample_ref = refs.get(sid)
        reference = sample_ref.reference if sample_ref is not None else None
        current = tr.get("initial_draft") or ""
        for rd in tr.get("rounds", []):
            c["n_rounds"] += 1
            action = str(rd.get("controller_action", "")).upper()
            if action == "STOP":
                c["n_stop"] += 1
                continue
            if action != "REFINE":
                c["n_other"] += 1
                continue
            c["n_refine"] += 1
            cand = rd.get("candidate") or ""
            delta = rd.get("delta_offline")
            if recompute and reference is not None:
                if rd.get("metric_offline") is not None:
                    mismatch_max = max(
                        mismatch_max,
                        abs(scorer.primary(reference, cand) - float(rd["metric_offline"])),
                    )
                delta = scorer.primary(reference, cand) - scorer.primary(reference, current)
            if delta is None:
                c["n_delta_missing"] += 1
            else:
                delta = float(delta)
                c["n_refine_scored"] += 1
                c[_classify(delta)] += 1
                delta_sum += delta
            if cand and cand == current:
                c["identical_text"] += 1
            verdict = str(rd.get("judge_verdict") or "none")
            c[f"judge_{verdict}"] += 1
            if rd.get("accepted"):
                c["accepted"] += 1
            if delta is not None and float(delta) > 0.0 and not rd.get("accepted"):
                c["useful_but_rejected"] += 1
            if delta is not None and float(delta) < 0.0 and rd.get("accepted"):
                c["worse_but_accepted"] += 1
            if rd.get("accepted") and cand:
                current = cand

    c["n_samples"] = len(samples)
    c["delta_sum"] = delta_sum  # type: ignore[assignment]
    return {
        "scope": "run",
        "arm": rec["arm"],
        "tag": rec["tag"],
        "task": task,
        "model": rec["model"],
        "seed": rec["seed"],
        "split": split,
        "source": str(rec["path"]),
        "counters": c,
        "metric_mismatch_max": mismatch_max,
    }


def merge_rows(rows: Sequence[dict], scope: str, **overrides) -> dict:
    """Pool counters over rows (rates are recomputed, never averaged)."""
    c: Counter = Counter()
    mismatch = 0.0
    base: Dict[str, object] = {}
    for i, r in enumerate(rows):
        for k, v in r["counters"].items():
            c[k] += v
        if i == 0:
            base = {k: r[k] for k in ("arm", "tag", "task", "model", "seed", "split", "source")}
        mismatch = max(mismatch, float(r.get("metric_mismatch_max", 0.0)))
    out = dict(base)
    out.update(overrides)
    out["scope"] = scope
    out["counters"] = c
    out["metric_mismatch_max"] = mismatch
    return out


def row_to_csv(r: dict) -> dict:
    c: Counter = r["counters"]
    scored = c.get("n_refine_scored", 0)
    n_ref = c.get("n_refine", 0)
    row = {k: "" for k in _CSV_FIELDS}
    row.update(
        {
            "scope": r["scope"], "arm": r["arm"], "tag": r["tag"], "task": r["task"],
            "model": r["model"], "seed": r["seed"], "split": r["split"],
            "source": r.get("source", ""),
            "n_samples": c.get("n_samples", 0), "n_rounds": c.get("n_rounds", 0),
            "n_refine": n_ref, "n_refine_scored": scored, "n_stop": c.get("n_stop", 0),
            "n_other": c.get("n_other", 0), "n_delta_missing": c.get("n_delta_missing", 0),
            "worse": c.get("worse", 0), "same": c.get("same", 0), "better": c.get("better", 0),
            "identical_text": c.get("identical_text", 0),
            "unnecessary_rate": round(c.get("worse", 0) / scored, 6) if scored else "",
            "useless_rate": round((c.get("worse", 0) + c.get("same", 0)) / scored, 6) if scored else "",
            "useful_rate": round(c.get("better", 0) / scored, 6) if scored else "",
            "accept_rate": round(c.get("accepted", 0) / n_ref, 6) if n_ref else "",
            "judge_worse": c.get("judge_worse", 0), "judge_better": c.get("judge_better", 0),
            "judge_tie": c.get("judge_tie", 0),
            "judge_uncertain": c.get("judge_uncertain", 0),
            "useful_but_rejected": c.get("useful_but_rejected", 0),
            "worse_but_accepted": c.get("worse_but_accepted", 0),
            "mean_delta": round(c.get("delta_sum", 0.0) / scored, 6) if scored else "",
            # 0 is meaningful here (the stored deltas are reproduced exactly), so
            # it is written rather than left blank
            "metric_mismatch_max": round(float(r.get("metric_mismatch_max", 0.0)), 9),
        }
    )
    return row


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)


def print_table(rows: List[dict], title: str) -> None:
    print()
    print(title)
    header = (
        f"{'scope':10s}{'task':14s}{'model':12s}{'arm':16s}{'tag':7s}"
        f"{'n_ref':>7s}{'worse':>7s}{'same':>6s}{'better':>7s}"
        f"{'unnec%':>8s}{'useless%':>9s}{'accept%':>9s}{'meanD':>9s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        c: Counter = r["counters"]
        scored = c.get("n_refine_scored", 0)
        n_ref = c.get("n_refine", 0)

        def pct(num: int, den: int = scored) -> str:
            return f"{100.0 * num / den:7.1f}%" if den else "      -"

        print(
            f"{r['scope']:10s}{r['task']:14s}{r['model']:12s}{r['arm'][:15]:16s}"
            f"{str(r['tag'])[:6]:7s}{n_ref:7d}{c.get('worse', 0):7d}{c.get('same', 0):6d}"
            f"{c.get('better', 0):7d}{pct(c.get('worse', 0)):>8s}"
            f"{pct(c.get('worse', 0) + c.get('same', 0)):>9s}"
            f"{pct(c.get('accepted', 0), n_ref):>9s}"
            f"{(c.get('delta_sum', 0.0) / scored if scored else 0.0):9.3f}"
        )


# --------------------------------------------------------------------------- #
# direction 2: premature stop (counterfactual, generation)
# --------------------------------------------------------------------------- #


def collect_stop_states(runs: Sequence[dict], limit: Optional[int]) -> Dict[Tuple[str, str, str], List[dict]]:
    """Every STOP decision in the traces, grouped by (task, model, arm)."""
    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for rec in runs:
        samples = rec["records"][:limit] if limit else rec["records"]
        for tr in samples:
            for rd in tr.get("rounds", []):
                if str(rd.get("controller_action", "")).upper() != "STOP":
                    continue
                groups[(rec["task"], rec["model"], rec["arm"])].append(
                    {"run": rec, "trace": tr, "round": rd}
                )
    return groups


def sample_stop_states(
    groups: Dict[Tuple[str, str, str], List[dict]], cap: int, seed: int
) -> List[dict]:
    """Round-robin over groups so a cap covers every run rather than the first one.

    Each group's bucket is shuffled with a seed derived from ``--sampling-seed``
    and the group key, so raising the cap is a stable superset and different
    seeds draw different states.
    """
    keys = sorted(groups)
    buckets: Dict[Tuple[str, str, str], List[dict]] = {}
    for k in keys:
        bucket = list(groups[k])
        random.Random(f"{seed}|{k}").shuffle(bucket)
        buckets[k] = bucket
    chosen: List[dict] = []
    i = 0
    while len(chosen) < cap:
        progressed = False
        for k in keys:
            bucket = buckets[k]
            if i < len(bucket):
                chosen.append(bucket[i])
                progressed = True
                if len(chosen) >= cap:
                    break
        if not progressed:
            break
        i += 1
    # stable execution order: deterministic and grouped by model for LLM reuse
    chosen.sort(
        key=lambda s: (
            s["run"]["model"], s["run"]["task"], s["run"]["arm"],
            str(s["trace"].get("sample_id")), int(s["round"].get("round_index", 0)),
        )
    )
    return chosen


def library_for_run(rec: dict, cache: Dict[tuple, List]) -> List:
    """The experience library that run actually used.

    ``snapshot_id="initial"`` is the shared bootstrap library; an online arm
    saved its grown library next to it, and that file is preferred when present
    so the counterfactual renders exactly the experiences the run could see.
    """
    cfg = rec["config"]
    key = (cfg.get("model"), cfg.get("task"), cfg.get("snapshot_id"), cfg.get("seed"))
    if key in cache:
        return cache[key]
    root = experience_dir(str(cfg.get("model")), str(cfg.get("task")))
    snap = str(cfg.get("snapshot_id") or "initial")
    candidates: List[Path] = []
    if snap != "initial":
        candidates += [
            root / f"{snap}_seed{cfg.get('seed')}.jsonl",
            root / f"{snap}.jsonl",
        ]
    candidates.append(root / "initial.jsonl")
    exps: List = []
    for p in candidates:
        if p.exists():
            exps = load_experiences(p)
            break
    cache[key] = exps
    return exps


def _build_pipeline(cfg_dict: dict, library: List, llm):
    """A Pipeline used only as a bundle of the *same* helpers the run used.

    Nothing here retrieves or generates on construction; ``_retrieve`` is never
    called because the counterfactual re-renders the stored ``exp_ids``.
    """
    from core.pipeline import Pipeline, RunConfig

    cfg_kwargs = dict(cfg_dict)
    if cfg_kwargs.get("draft_source") == "cached" and not cfg_kwargs.get("draft_cache"):
        cfg_kwargs["draft_source"] = "stored"  # drafts are irrelevant here
    cfg = RunConfig(**cfg_kwargs)
    return Pipeline(cfg, retriever=None, library=library, llm=llm, cache=None)


def counterfactual_one(llm, pipe, rec: dict, state: dict) -> dict:
    """Force one extra refinement at a STOP state and score it."""
    from baseline_core.tasks import get_adapter
    from baseline_core.types import TaskExample

    from core.determinism import call_seed
    from core.experience import render_experience_block
    from core.pipeline import DEFAULT_REFINE_INSTRUCTION, build_refine_prompt

    cfg = pipe.cfg
    trace, rd = state["trace"], state["round"]
    t = int(rd.get("round_index", 0))
    split = split_of(rec)
    sample_id = str(trace.get("sample_id", ""))
    sample_ref = refs_for(cfg.task, split).get(sample_id)
    if sample_ref is None:
        raise KeyError(f"no manifest reference for {sample_id} ({cfg.task}/{split})")

    current = reconstruct_current(trace, t)
    by_id = {e.exp_id: e for e in pipe.library}
    stored_ids = [str(x) for x in rd.get("exp_ids", [])]
    missing = [x for x in stored_ids if x not in by_id]
    block = render_experience_block(
        [by_id[x] for x in stored_ids if x in by_id],
        include_outcome=cfg.include_outcome,
        count_tokens=llm.count_tokens,
        max_units=cfg.k,
    )

    in_tok = out_tok = 0
    n_calls = 0

    # 1. instruction: the controller with STOP removed from the schema, seeded
    #    separately from the real decision so this is a genuine counterfactual.
    dec = pipe.controller.decide(
        llm, sample_id, t, sample_ref.source, current, block,
        stop_mode="fixed", temperature=0.0, seed_salt="counterfactual",
    )
    for call in dec.calls:
        in_tok += int(call.input_tokens)
        out_tok += int(call.output_tokens)
        n_calls += 1
    instruction = dec.instruction.strip() or DEFAULT_REFINE_INSTRUCTION

    # 2. refine: identical prompt construction and identical seed to the round
    #    the pipeline *would* have run had the controller said REFINE.
    adapter = get_adapter(cfg.task)
    example = TaskExample(
        index=sample_ref.row_index, source=sample_ref.source,
        reference=sample_ref.reference, task=cfg.task,
    )
    rgen = llm.generate(
        prompt=build_refine_prompt(adapter, example, current, instruction),
        system_prompt=adapter.system_prompt(),
        seed=call_seed(sample_id, t, "refine"),
        call_type="refine",
        max_tokens=pipe.budget.max_tokens,
        temperature=pipe.budget.temperature,
        top_p=pipe.budget.top_p,
    )
    candidate = adapter.parse_output(rgen.text)
    in_tok += int(rgen.input_tokens)
    out_tok += int(rgen.output_tokens)
    n_calls += 1

    # 3. judge: same deterministic A/B ordering and seeds as the pipeline.
    verdict = pipe.judge.judge(
        llm, cfg.model, sample_id, t, sample_ref.source, current, candidate
    )
    for call in verdict.calls:
        in_tok += int(call.input_tokens)
        out_tok += int(call.output_tokens)
        n_calls += 1

    prev_m = pipe.scorer.primary(sample_ref.reference, current)
    cand_m = pipe.scorer.primary(sample_ref.reference, candidate)
    delta = cand_m - prev_m
    return {
        "task": cfg.task, "model": cfg.model, "arm": rec["arm"], "tag": rec["tag"],
        "split": split, "sample_id": sample_id, "round_index": t,
        "source": str(rec["path"]),
        "current_metric": round(prev_m, 6), "cf_metric": round(cand_m, 6),
        "cf_delta": round(delta, 6), "premature": int(delta > 0.0),
        "judge_verdict": verdict.verdict,
        "judge_order_consistent": int(bool(verdict.order_consistent)),
        "would_accept": int(bool(verdict.accepted)),
        "empty_candidate": int(not candidate.strip()),
        "n_experiences": len(stored_ids), "n_missing_experiences": len(missing),
        "instruction": instruction[:200].replace("\n", " "),
        "input_tokens": in_tok, "output_tokens": out_tok, "n_calls": n_calls,
    }


def write_cf_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CF_FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)


def print_cf_table(rows: List[dict]) -> None:
    print()
    print("Premature-stop counterfactual (forced one extra refinement at each STOP state)")
    header = (
        f"{'task':14s}{'model':12s}{'states':>8s}{'premature':>10s}{'rate':>8s}"
        f"{'judge_acc':>10s}{'meanD':>9s}{'95% CI':>20s}{'tok/state':>11s}"
    )
    print(header)
    print("-" * len(header))
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["model"])].append(r)
    for (task, model), rs in sorted(groups.items()):
        deltas = [r["cf_delta"] for r in rs]
        n = len(rs)
        prem = sum(r["premature"] for r in rs)
        acc = sum(r["would_accept"] for r in rs)
        ci = paired_bootstrap(deltas, [0.0] * n, n_resamples=2000, seed=0)
        toks = sum(r["input_tokens"] + r["output_tokens"] for r in rs) / n
        print(
            f"{task:14s}{model:12s}{n:8d}{prem:10d}{prem / n:8.1%}{acc / n:10.1%}"
            f"{sum(deltas) / n:9.3f}"
            f"{f'[{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]':>20s}{toks:11.0f}"
        )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def positive_int(text: str) -> int:
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 7 stopping diagnostics: unnecessary refinement + premature stop"
    )
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run directories to scan recursively (default: runs/)")
    ap.add_argument("--models", default="all")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--arms", default="all", help="comma list of arm names to keep")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug only: use at most N samples per trace file")
    ap.add_argument("--out-dir", default=None,
                    help="directory for the CSV output (default: scores/)")
    ap.add_argument("--recompute", action="store_true",
                    help="re-derive metric_offline/delta with the shared Scorer and "
                         "report the disagreement with the stored values")
    ap.add_argument("--counterfactual", action="store_true",
                    help="ALSO run generation: force one extra refinement at STOP states")
    ap.add_argument("--max-stop-states", type=positive_int, default=100,
                    help="cap on counterfactual STOP states (default: 100)")
    ap.add_argument("--sampling-seed", type=int, default=0,
                    help="seed for drawing which STOP states the cap covers")
    ap.add_argument("--gpu", default="2", help="card used by --counterfactual only")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (and STOP-state availability) without generating")
    args = ap.parse_args()

    models = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    tasks = list(TASKS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    arms = [] if args.arms == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]
    run_dirs = [Path(p) for p in args.runs] if args.runs else [EXP_ROOT / "runs"]
    out_dir = Path(args.out_dir) if args.out_dir else SCORES_DIR

    runs = discover(run_dirs, models, tasks, arms)
    bad = sum(r["bad_lines"] for r in runs)
    print(
        f"scanned {len(run_dirs)} dir(s): {len(runs)} trace file(s), "
        f"{sum(len(r['records']) for r in runs)} sample traces"
        + (f", skipped {bad} unparseable line(s)" if bad else "")
    )
    if not runs:
        print("no traces found; nothing to diagnose")
        return 1

    # ---- direction 1: unnecessary refinement (never touches a GPU) ---------
    run_rows = [analyze_run(rec, args.limit, args.recompute) for rec in runs]
    arm_rows = [
        merge_rows(group, "arm", tag="*", seed="*", source="*")
        for _key, group in sorted(
            _group_by(run_rows, ("task", "model", "arm")).items()
        )
    ]
    tm_rows = [
        merge_rows(group, "task_model", arm="*", tag="*", seed="*", source="*")
        for _key, group in sorted(_group_by(run_rows, ("task", "model")).items())
    ]
    all_rows = run_rows + arm_rows + tm_rows
    csv_rows = [row_to_csv(r) for r in all_rows]
    write_csv(csv_rows, out_dir / "stopping_diagnostics.csv")

    print_table(run_rows, "Per trace file (all REFINE decisions)")
    print_table(tm_rows, "Pooled per task x model (the numbers to report)")
    totals = merge_rows(run_rows, "all", arm="*", tag="*", seed="*", source="*")
    print_table([totals], "Pooled over every trace file")

    if args.recompute:
        mm = max((r["metric_mismatch_max"] for r in run_rows), default=0.0)
        print()
        print(f"--recompute: max |recomputed metric_offline - stored| = {mm:.3e} "
              f"(0 means the stored deltas are reproduced exactly)")
    print()
    print(f"-> {out_dir / 'stopping_diagnostics.csv'} ({len(csv_rows)} rows)")

    # ---- direction 2: premature stop (generation, opt-in) ------------------
    groups = collect_stop_states(runs, args.limit)
    n_states = sum(len(v) for v in groups.values())
    print()
    print(f"STOP states available in these traces: {n_states}")
    for (task, model, arm), v in sorted(groups.items()):
        print(f"    {task:14s}{model:12s}{arm:14s}{len(v):6d}")
    cap = min(args.max_stop_states, n_states)
    print(f"counterfactual would sample {cap} state(s) "
          f"(cap --max-stop-states={args.max_stop_states})")

    if not args.counterfactual:
        print("counterfactual mode is OFF (pass --counterfactual to run generation)")
        return 0

    # GPU budget check: the helper in core/__init__.py is the single authority.
    import os

    gpus = [g.strip() for g in args.gpu.split(",") if g.strip()]
    if not gpus:
        print("REFUSING: --gpu must name at least one card")
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    try:
        ensure_gpu_whitelist()
    except RuntimeError as exc:
        print(f"REFUSING: {exc}")
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]

    if n_states == 0:
        print("no STOP states to counterfact; nothing to generate")
        return 0
    if args.dry_run:
        est = cap * 4
        print()
        print("[dry-run] would load one model at a time and make ~"
              f"{est}-{est + cap} calls total (~4-5 calls per state: 1 controller "
              "+ 1 refine + 2 judge, plus the occasional repair retry)")
        print("[dry-run] no model was loaded and no GPU was touched")
        return 0

    states = sample_stop_states(groups, cap, args.sampling_seed)
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for s in states:
        by_model[s["run"]["model"]].append(s)

    from baseline_core.config import MODEL_CONFIGS
    from baseline_core.llm import LLMClient

    import torch

    from core import MAX_MODEL_LEN, gpu_mem_util

    lib_cache: Dict[tuple, List] = {}
    pipe_cache: Dict[tuple, object] = {}
    cf_rows: List[dict] = []
    skipped: Counter = Counter()
    for model in sorted(by_model):
        print(f"[LOAD] model={model} ({len(by_model[model])} STOP state(s))", flush=True)
        llm = LLMClient(
            MODEL_CONFIGS[model],
            backend="vllm",
            gpu=gpus[0],
            gpu_memory_utilization=gpu_mem_util(model, args.gpu_memory_utilization),
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
        )
        for state in by_model[model]:
            rec, trace = state["run"], state["trace"]
            lib = library_for_run(rec, lib_cache)
            if not lib:
                skipped["no_library"] += 1
                continue
            # the helper bundle is per (config, library): two runs can share a
            # config but read different snapshot libraries
            lib_fp = hashlib.sha256(
                "|".join(e.exp_id for e in lib).encode("utf-8")
            ).hexdigest()[:12]
            pkey = (rec["model"], rec["task"],
                    json.dumps(rec["config"], sort_keys=True), lib_fp)
            if pkey not in pipe_cache:
                pipe_cache[pkey] = _build_pipeline(rec["config"], lib, llm)
            try:
                row = counterfactual_one(llm, pipe_cache[pkey], rec, state)
            except KeyError as exc:
                skipped[f"unresolved:{exc}"] += 1
                continue
            cf_rows.append(row)
            print(
                f"    {row['task']:14s}{row['model']:12s}{row['sample_id']:28s}"
                f"r{row['round_index']} d={row['cf_delta']:+.4f} "
                f"premature={row['premature']} judge={row['judge_verdict']}",
                flush=True,
            )
        del llm
        torch.cuda.empty_cache()

    if skipped:
        print(f"[WARN] skipped states: {dict(skipped)}")
    if not cf_rows:
        print("no counterfactual states completed")
        return 1

    cf_path = out_dir / "stopping_diagnostics_counterfactual.csv"
    write_cf_csv(cf_rows, cf_path)
    print_cf_table(cf_rows)
    print()
    print(f"-> {cf_path} ({len(cf_rows)} states)")
    return 0


def _group_by(rows: Sequence[dict], keys: Sequence[str]) -> Dict[tuple, List[dict]]:
    out: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
