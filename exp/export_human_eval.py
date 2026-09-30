#!/usr/bin/env python
"""Phase 10: build a blinded pairwise human-evaluation package.

The package compares the main experience arm (``full_static``) against the
*strongest non-experience arm available in the traces* (``--arms FULL,BASELINE``;
the baseline may be ``auto``, see below).  It is written to ``human_eval/`` and
consists of exactly three files:

* ``human_eval_pairs.csv``  -- what the annotator opens.  One row per pair:
  ``pair_id, task, model, input, system_a, system_b, winner, confidence,
  notes``.  No arm name (nor anything derived from one) appears anywhere in it.
* ``answer_key.DO_NOT_OPEN.json`` -- the answer key, mapping ``pair_id`` to which
  arm is ``A`` and which is ``B``.  Annotators must not open this file.
* ``README.md``             -- protocol, including that ties are allowed.

Design rules (all enforced by the code, not by convention):

* a pair is always the **same sample id** under the two arms -- sample ids are
  never mixed inside a pair;
* presentation order (``A`` = full or ``A`` = baseline) is drawn from a seed
  derived deterministically from ``(seed, sample_id)``, so the package is
  reproducible and the arm is not inferable from the row order;
* ``pair_id`` is a sequential ``P0001``-style label assigned after a seeded
  shuffle -- it encodes neither the arm nor the cell;
* traces may be mid-write while production runs are still growing: a final line
  without a trailing newline is dropped, unparseable lines are skipped, and both
  events are reported (never a silent truncation);
* completeness follows ``aggregate.py``: a run is COMPLETE only when the JSONL
  holds at least ``core.TEST_SAMPLES[task]`` records (1000 for the three big
  tasks, 100 for gigaword); anything else is PARTIAL and is labelled as such in
  the console report, the README and the answer key.  ``--complete-only``
  restricts the package to complete runs.

Baseline choice (``auto``, the documented default)
--------------------------------------------------
A *non-experience* arm is one whose records carry ``config.experience_mode ==
"none"`` (falling back to a documented name list when the config is absent).
``auto`` ranks every non-experience arm that has at least one complete run by its
mean **rank** of the per-sample final offline metric inside each (task, model)
cell where it and ``full_static`` are both present -- rank 1 is the best arm in
that cell -- and breaks ties with the fixed priority
``sr_j_stop > sr_j_fixed > no_experience > direct_zero``.  A coverage tie-break
then applies: if a candidate is complete in strictly more required cells and the
per-cell paired-bootstrap 95% CI of the quality difference (``core.stats``,
paired by sample id) includes 0 in every shared cell, the two arms are not
distinguishable on the available data and the wider-coverage arm is chosen --
otherwise the stronger arm is kept.  The evidence table and the selection reason
are printed, so the choice is auditable, and ``--baseline-arm`` overrides it.  If the requested arm (or, with ``auto``,
any candidate) has no complete run at all, the script fails loudly with exit
code 2 and the package is not written.

This module also holds the shared trace reader used by ``make_plots.py``
(``load_traces`` / ``discover_runs`` / ``choose_baseline_arm``), so both Phase 9
and Phase 10 key off exactly the same completeness rule and arm attribution.

Usage
-----
    python export_human_eval.py --dry-run
    python export_human_eval.py                       # -> human_eval/
    python export_human_eval.py --limit 200 --seed 42
    python export_human_eval.py --arms full_static,no_experience
    python export_human_eval.py --baseline-arm sr_j_stop --complete-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import EXP_ROOT, TASKS, TEST_SAMPLES  # noqa: E402  (stdlib-only import)
from core.manifest import manifest_path, read_manifest  # noqa: E402
from core.scoring import PRIMARY_METRIC  # noqa: E402  (import is lazy-heavy: stdlib only)
from core.stats import paired_bootstrap  # noqa: E402

# --------------------------------------------------------------------------- #
# constants / policy
# --------------------------------------------------------------------------- #

FULL_ARM_DEFAULT = "full_static"

#: models the human evaluation must cover
EVAL_MODELS = ("glm4-9b", "qwen3-8b")

#: the four production tasks (order is the paper's order)
EVAL_TASKS: Tuple[str, ...] = tuple(TASKS)

#: config.experience_mode values that mean "no experience is used"
NON_EXPERIENCE_MODES = frozenset({"none", "no_experience", "off", "false"})

#: name-level fallback for traces whose config does not carry experience_mode
NON_EXPERIENCE_NAMES = frozenset(
    {"no_experience", "sr_j_stop", "sr_j_fixed", "direct_zero", "initial_only"}
)

#: tie-break order when two candidate baselines score equally strong
BASELINE_PRIORITY: Tuple[str, ...] = ("sr_j_stop", "sr_j_fixed", "no_experience", "direct_zero")

#: direct parent directories that are *not* sweep tags
_UNTAGGED_PARENTS = frozenset({"test", "dev", "accumulation"})

DEFAULT_RUN_DIRS = ("runs/main", "runs/ablation")

ANNOTATION_COLUMNS = (
    "pair_id",
    "task",
    "model",
    "input",
    "system_a",
    "system_b",
    "winner",
    "confidence",
    "notes",
)

KEY_FILE_NAME = "answer_key.DO_NOT_OPEN.json"
PAIRS_FILE_NAME = "human_eval_pairs.csv"
README_FILE_NAME = "README.md"


# --------------------------------------------------------------------------- #
# trace reading (never crashes on a truncated / mid-write line)
# --------------------------------------------------------------------------- #


@dataclass
class ReadInfo:
    raw_lines: int = 0
    kept: int = 0
    bad_lines: int = 0
    dropped_tail: bool = False
    truncated_tail: bool = False
    mtime: float = 0.0


def load_traces(path: Path, *, force_drop_last: bool = False) -> Tuple[List[dict], ReadInfo]:
    """Read a JSONL trace, tolerating a file that is still being appended to.

    * a trailing line without a final newline is *dropped*: the writer may be
      halfway through ``json.dumps`` (this is "drop the last line of a growing
      file"); ``force_drop_last`` additionally drops the last terminated line;
    * any line that does not parse (or that is not a trace record) is skipped
      and counted -- a truncated line never raises;
    * every kept record is a dict with a ``rounds`` list, like the pipeline
      writes.
    """
    info = ReadInfo()
    try:
        info.mtime = path.stat().st_mtime
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return [], info
    if not text:
        return [], info
    lines = text.split("\n")
    if lines and lines[-1] == "":
        # normal case: the file ends with a newline, so every line is terminated
        lines = lines[:-1]
        if force_drop_last and lines:
            info.dropped_tail = True
            lines = lines[:-1]
    else:
        # unterminated final line -- almost certainly mid-write
        info.truncated_tail = True
        if lines:
            lines = lines[:-1]
    info.raw_lines = len([x for x in lines if x.strip()])
    out: List[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            info.bad_lines += 1
            continue
        if not isinstance(rec, dict) or not isinstance(rec.get("rounds"), list):
            info.bad_lines += 1
            continue
        out.append(rec)
    info.kept = len(out)
    return out, info


@dataclass
class RunInfo:
    """One trace file: one arm x tag x task x model x seed."""

    path: Path
    arm: str
    tag: str
    task: str
    model: str
    split: str
    seed: Optional[int]
    config: dict
    records: List[dict]
    info: ReadInfo

    @property
    def rel(self) -> str:
        try:
            return str(self.path.relative_to(EXP_ROOT))
        except ValueError:
            return str(self.path)

    @property
    def n(self) -> int:
        return len(self.records)

    @property
    def expected_n(self) -> int:
        return TEST_SAMPLES.get(self.task, 1000)

    @property
    def complete(self) -> bool:
        return self.n >= self.expected_n

    @property
    def status(self) -> str:
        return "complete" if self.complete else "partial"

    @property
    def key(self) -> Tuple[str, str, str, str, str, str]:
        return (self.arm, self.tag, self.task, self.model, self.split, str(self.seed))

    def by_sample_id(self) -> Dict[str, dict]:
        """sample_id -> record.  Duplicate ids keep the first occurrence."""
        out: Dict[str, dict] = {}
        for rec in self.records:
            sid = str(rec.get("sample_id", "")).strip()
            if sid and sid not in out:
                out[sid] = rec
        return out

    def sample_ids(self) -> List[str]:
        return sorted(self.by_sample_id())


def _split_of(records: Sequence[dict], path: Path) -> str:
    for rec in records:
        parts = str(rec.get("sample_id", "")).split("/")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    parent = path.parent.name
    return parent if parent in _UNTAGGED_PARENTS else "test"


def _tag_of(path: Path, model: str) -> str:
    parent = path.parent.name
    if parent in _UNTAGGED_PARENTS or parent == model:
        return ""
    return parent


def discover_runs(
    bases: Sequence[Path],
    *,
    models: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    arms: Optional[Sequence[str]] = None,
    splits: Sequence[str] = ("test",),
    allow_tagged: bool = False,
    force_drop_last: bool = False,
    verbose: bool = True,
    log=print,
) -> List[RunInfo]:
    """One :class:`RunInfo` per usable trace file, in a stable order.

    ``arm`` comes from the record's own ``arm`` field when present and otherwise
    from the file stem (the convention ``aggregate.py`` uses); ``task``/``model``
    come from the records.  Files whose records disagree with each other are
    skipped with a warning rather than silently pooled.
    """
    want_models = set(models) if models else None
    want_tasks = set(tasks) if tasks else None
    want_arms = set(arms) if arms else None
    want_splits = set(splits) if splits else None

    found: List[RunInfo] = []
    for base in bases:
        base = Path(base)
        if not base.is_absolute():
            base = EXP_ROOT / base
        if not base.exists():
            log(f"[WARN] run dir does not exist: {base}")
            continue
        for jsonl in sorted(base.rglob("*.jsonl")):
            recs, info = load_traces(jsonl, force_drop_last=force_drop_last)
            if not recs:
                if info.bad_lines or info.truncated_tail:
                    log(f"[WARN] no usable records in {jsonl} "
                        f"({info.bad_lines} bad line(s), truncated_tail={info.truncated_tail})")
                continue
            task = str(recs[0].get("task", ""))
            model = str(recs[0].get("model", ""))
            if not task or not model:
                log(f"[WARN] skipping {jsonl}: records carry no task/model")
                continue
            arm = str(recs[0].get("arm") or jsonl.stem)
            bad_attr = [
                k for k in ("task", "model")
                if any(str(r.get(k, "")) != (task if k == "task" else model) for r in recs)
            ]
            if bad_attr:
                log(f"[WARN] skipping {jsonl}: records disagree on {bad_attr}")
                continue
            split = _split_of(recs, jsonl)
            tag = _tag_of(jsonl, model)
            if want_tasks is not None and task not in want_tasks:
                continue
            if want_models is not None and model not in want_models:
                continue
            if want_arms is not None and arm not in want_arms:
                continue
            if want_splits is not None and split not in want_splits:
                continue
            if tag and not allow_tagged:
                if verbose:
                    log(f"[skip tagged] {jsonl} (tag={tag!r}; pass --allow-tagged to include)")
                continue
            if info.truncated_tail and verbose:
                log(f"[WARN] {jsonl}: dropped an unterminated final line (file is mid-write)")
            if info.bad_lines and verbose:
                log(f"[WARN] {jsonl}: skipped {info.bad_lines} unparseable line(s)")
            found.append(
                RunInfo(
                    path=jsonl, arm=arm, tag=tag, task=task, model=model, split=split,
                    seed=recs[0].get("seed"), config=recs[0].get("config", {}) or {},
                    records=recs, info=info,
                )
            )
    return found


def group_runs(runs: Iterable[RunInfo], *, log=print) -> Dict[Tuple[str, ...], RunInfo]:
    """Collapse duplicate traces for the same (arm, tag, task, model, split, seed).

    The record with the most samples wins (a half-written duplicate must never
    shadow a fuller one); ties go to the lexicographically first path for
    determinism.
    """
    grouped: Dict[Tuple[str, ...], RunInfo] = {}
    for run in sorted(runs, key=lambda r: r.rel):
        prev = grouped.get(run.key)
        if prev is None:
            grouped[run.key] = run
            continue
        keep, drop = (run, prev) if run.n > prev.n else (prev, run)
        log(f"[WARN] duplicate trace for {run.key}: keeping {keep.rel} "
            f"(n={keep.n}) over {drop.rel} (n={drop.n})")
        grouped[run.key] = keep
    return grouped


def is_non_experience(run: RunInfo) -> bool:
    """True when the run's records say no retrieved experience was used."""
    mode = str(run.config.get("experience_mode", "")).strip().lower()
    if mode:
        return mode in NON_EXPERIENCE_MODES
    return run.arm in NON_EXPERIENCE_NAMES


# --------------------------------------------------------------------------- #
# baseline selection
# --------------------------------------------------------------------------- #


@dataclass
class BaselineChoice:
    arm: Optional[str]
    evidence: List[dict] = field(default_factory=list)
    reason: str = ""


def _rule() -> str:
    return (
        "auto: among non-experience arms with >=1 COMPLETE run, rank by mean rank of the "
        "per-sample final metric inside each (task, model) cell where the arm and the full "
        "arm both have data (rank 1 = strongest); ties break by "
        + " > ".join(BASELINE_PRIORITY)
        + ".  Coverage tie-break: a candidate that is complete in strictly more required "
          "cells replaces the top-ranked one when the per-cell paired-bootstrap 95% CI of "
          "the quality difference includes 0 in every shared cell (i.e. the two arms are "
          "not distinguishable on the data we have)"
    )


def _complete_cells(
    runs: Sequence[RunInfo], arm: str, tasks: Sequence[str], models: Sequence[str]
) -> List[Tuple[str, str]]:
    grid = {(t, m) for t in tasks for m in models}
    return sorted({(r.task, r.model) for r in runs
                   if r.arm == arm and r.complete and (r.task, r.model) in grid})


def _indistinguishable(
    runs: Sequence[RunInfo],
    arm_a: str,
    arm_b: str,
    cells: Sequence[Tuple[str, str]],
    *,
    n_resamples: int = 2000,
) -> Tuple[bool, str]:
    """True when no shared cell shows a significant paired difference (a vs b).

    Uses ``core.stats.paired_bootstrap`` (the same machinery as the paper), on the
    per-sample final offline metric, paired by sample id.
    """
    details = []
    shared = 0
    for (task, model) in cells:
        a = next((r for r in runs if r.arm == arm_a and r.task == task and r.model == model), None)
        b = next((r for r in runs if r.arm == arm_b and r.task == task and r.model == model), None)
        if a is None or b is None:
            continue
        av, bv = a.by_sample_id(), b.by_sample_id()
        ids = sorted(set(av) & set(bv))
        xa = [float(av[i]["final_metric_offline"]) for i in ids
              if av[i].get("final_metric_offline") is not None
              and bv[i].get("final_metric_offline") is not None]
        xb = [float(bv[i]["final_metric_offline"]) for i in ids
              if av[i].get("final_metric_offline") is not None
              and bv[i].get("final_metric_offline") is not None]
        if not ids:
            continue
        shared += 1
        res = paired_bootstrap(xa, xb, n_resamples=n_resamples, seed=0)
        sig = not (res.ci_low <= 0.0 <= res.ci_high)
        details.append(
            f"{task}/{model} n={len(ids)} diff={res.diff:+.3f} "
            f"CI[{res.ci_low:+.3f},{res.ci_high:+.3f}]{' *' if sig else ''}"
        )
        if sig:
            return False, "; ".join(details)
    return (shared > 0), ("; ".join(details) if details else "no shared complete cell")


def choose_baseline_arm(
    runs: Sequence[RunInfo],
    full_arm: str,
    tasks: Sequence[str],
    models: Sequence[str],
    *,
    coverage_tiebreak: bool = True,
    log=print,
) -> BaselineChoice:
    """Pick the strongest non-experience arm available (see module docstring).

    ``runs`` must be the de-duplicated runs (see :func:`group_runs`).  Only runs
    in the required (task, model) grid are considered.
    """
    grid = {(t, m) for t in tasks for m in models}
    by_cell: Dict[Tuple[str, str], Dict[str, RunInfo]] = {}
    for run in runs:
        if (run.task, run.model) in grid:
            by_cell.setdefault((run.task, run.model), {})[run.arm] = run

    candidates = sorted({r.arm for r in runs if is_non_experience(r) and r.arm != full_arm})
    complete_candidates = sorted(
        a for a in candidates
        if any(r.arm == a and r.complete for r in runs)
    )
    coverage = {a: len(_complete_cells(runs, a, tasks, models)) for a in complete_candidates}

    evidence: List[dict] = []
    rank_sum: Dict[str, float] = {a: 0.0 for a in complete_candidates}
    rank_cnt: Dict[str, int] = {a: 0 for a in complete_candidates}
    for (task, model) in sorted(grid):
        cell = by_cell.get((task, model), {})
        present = [a for a in complete_candidates if a in cell and cell[a].complete]
        if not present or full_arm not in cell:
            continue
        # only cells where we can compare at least two arms are informative
        if len(present) < 2:
            for a in present:
                evidence.append(
                    {"task": task, "model": model, "arm": a,
                     "mean_final": _mean_final(cell[a]), "rank": "",
                     "n": cell[a].n, "coverage": coverage[a],
                     "note": "only comparable candidate in this cell"}
                )
            continue
        scored = sorted(
            ((a, _mean_final(cell[a])) for a in present if _mean_final(cell[a]) is not None),
            key=lambda x: (-x[1], BASELINE_PRIORITY.index(x[0]) if x[0] in BASELINE_PRIORITY else 99, x[0]),
        )
        for i, (a, val) in enumerate(scored, start=1):
            rank_sum[a] += i
            rank_cnt[a] += 1
            evidence.append(
                {"task": task, "model": model, "arm": a,
                 "mean_final": round(val, 4), "rank": i, "n": cell[a].n,
                 "coverage": coverage[a], "note": ""}
            )

    if not complete_candidates:
        return BaselineChoice(None, evidence, "no non-experience arm has a COMPLETE run")

    def sort_key(a: str) -> Tuple[float, int, str]:
        avg = rank_sum[a] / rank_cnt[a] if rank_cnt[a] else float("inf")
        prio = BASELINE_PRIORITY.index(a) if a in BASELINE_PRIORITY else len(BASELINE_PRIORITY)
        return (avg, prio, a)

    ranked = sorted(complete_candidates, key=sort_key)
    best = ranked[0]
    reason = (
        f"auto-selected {best!r} from {complete_candidates} "
        f"(mean rank {rank_sum[best] / rank_cnt[best]:.2f} over {rank_cnt[best]} comparable "
        f"cell(s); complete cells: {coverage[best]})"
        if rank_cnt[best] else
        f"auto-selected {best!r} by priority order (no cell had two comparable candidates; "
        f"complete cells: {coverage[best]})"
    )
    if coverage_tiebreak:
        for cand in ranked[1:]:
            if coverage[cand] <= coverage[best]:
                continue
            same, detail = _indistinguishable(runs, best, cand, sorted(
                set(_complete_cells(runs, best, tasks, models))
                & set(_complete_cells(runs, cand, tasks, models))
            ))
            if same:
                reason += (
                    f"; coverage tie-break: {cand!r} covers {coverage[cand]} complete cells vs "
                    f"{coverage[best]} and is not distinguishable from it ({detail}) -> chose {cand!r}"
                )
                best = cand
                break
            reason += (f"; kept {best!r} over wider-coverage {cand!r}: significantly different "
                       f"({detail})")
    for row in evidence:
        if row["arm"] == best and row["rank"] != "":
            row["note"] = "selected"
    return BaselineChoice(best, evidence, reason)


def _mean_final(run: RunInfo) -> Optional[float]:
    vals = [float(r["final_metric_offline"]) for r in run.records
            if r.get("final_metric_offline") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def print_baseline_evidence(choice: BaselineChoice, log=print) -> None:
    log(f"[baseline] {choice.reason}")
    if not choice.evidence:
        return
    log(f"[baseline] evidence ({_rule()}):")
    log(f"  {'task':14s}{'model':12s}{'arm':16s}{'mean_final':>11s}{'rank':>6s}"
        f"{'n':>6s}{'cells':>7s}  note")
    for row in choice.evidence:
        mf = row["mean_final"]
        log(f"  {row['task']:14s}{row['model']:12s}{row['arm']:16s}"
            f"{(f'{mf:.4f}' if mf is not None else 'n/a'):>11s}"
            f"{str(row['rank']):>6s}{row['n']:>6d}{row.get('coverage', 0):>7d}  {row['note']}")


# --------------------------------------------------------------------------- #
# pair construction
# --------------------------------------------------------------------------- #


def presentation_order(seed: int, sample_id: str, full_arm: str, baseline_arm: str) -> Tuple[str, str]:
    """Deterministic A/B assignment for one sample id.

    ``sha256("seed|sample_id")`` decides whether the full arm is shown as A;
    the same sample id always gets the same order, and the order does not depend
    on which cell the pair landed in or on any other pair.
    """
    digest = hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()
    full_is_a = int(digest[:16], 16) % 2 == 0
    return (full_arm, baseline_arm) if full_is_a else (baseline_arm, full_arm)


def _spread_indices(n_avail: int, k: int) -> List[int]:
    """k indices spread evenly across ``range(n_avail)`` (deterministic)."""
    if k <= 0:
        return []
    if k >= n_avail:
        return list(range(n_avail))
    if k == 1:
        return [n_avail // 2]
    return sorted({round(i * (n_avail - 1) / (k - 1)) for i in range(k)})


def allocate(avail: Dict[str, int], budget: int, order: Sequence[str]) -> Dict[str, int]:
    """Even-as-possible allocation of ``budget`` pairs across the available cells.

    Every cell in ``order`` with data gets a share; shares never exceed what the
    cell can supply, and leftovers are redistributed deterministically.
    """
    active = [c for c in order if avail.get(c, 0) > 0]
    alloc: Dict[str, int] = {c: 0 for c in order}
    if not active or budget <= 0:
        return alloc
    base = budget // len(active)
    remainder = budget - base * len(active)
    for i, c in enumerate(active):
        want = base + (1 if i < remainder else 0)
        alloc[c] = min(want, avail[c])
    leftover = budget - sum(alloc.values())
    while leftover > 0:
        spare = sorted(
            ((avail[c] - alloc[c], c) for c in active if avail[c] > alloc[c]),
            key=lambda x: (-x[0], x[1]),
        )
        if not spare:
            break
        for _, c in spare:
            if leftover <= 0:
                break
            alloc[c] += 1
            leftover -= 1
    return alloc


def _cell_order(tasks: Sequence[str], models: Sequence[str]) -> List[Tuple[str, str]]:
    """Deterministic cell order: rounds over models so no model is starved."""
    return [(t, m) for m in models for t in tasks]


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #


def _load_sources(task: str, split: str) -> Dict[str, str]:
    path = manifest_path(task, split)
    if not path.exists():
        raise FileNotFoundError(f"manifest missing for task {task!r} split {split!r}: {path}")
    return {r.sample_id: r.source for r in read_manifest(path)}


def build_pairs(args, runs: Sequence[RunInfo], full_arm: str, baseline_arm: str,
                log=print) -> Tuple[List[dict], Dict[str, dict], dict]:
    """Return (rows, answer_key, coverage)."""
    by_cell: Dict[Tuple[str, str], Dict[str, RunInfo]] = {}
    for run in runs:
        if run.arm in (full_arm, baseline_arm):
            by_cell.setdefault((run.task, run.model), {})[run.arm] = run

    cells = _cell_order(args.tasks, args.models)
    avail: Dict[str, int] = {}
    pair_ids_by_cell: Dict[str, List[str]] = {}
    cell_runs: Dict[str, Tuple[Optional[RunInfo], Optional[RunInfo]]] = {}
    for key in cells:
        cell = by_cell.get(key, {})
        full, base = cell.get(full_arm), cell.get(baseline_arm)
        cell_runs[key] = (full, base)
        if full is None or base is None:
            avail[key] = 0
            continue
        if args.complete_only and not (full.complete and base.complete):
            avail[key] = 0
            continue
        shared = sorted(set(full.by_sample_id()) & set(base.by_sample_id()))
        pair_ids_by_cell[key] = shared
        avail[key] = len(shared)

    alloc = allocate(avail, args.limit, cells)

    coverage: Dict[str, dict] = {}
    rows: List[dict] = []
    answer_key: Dict[str, dict] = {}
    source_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    for key in cells:
        task, model = key
        full, base = cell_runs[key]
        ids = pair_ids_by_cell.get(key, [])
        chosen = [ids[i] for i in _spread_indices(len(ids), alloc[key])]
        entry = {
            "task": task,
            "model": model,
            "available_pairs": len(ids),
            "requested": alloc[key],
            "realised": len(chosen),
            "full_run": full.rel if full else None,
            "baseline_run": base.rel if base else None,
            "full_n": full.n if full else 0,
            "baseline_n": base.n if base else 0,
            "full_status": full.status if full else "missing",
            "baseline_status": base.status if base else "missing",
            "note": "",
        }
        if not ids:
            if full is None and base is None:
                entry["note"] = "neither arm has a run for this cell"
            elif full is None:
                entry["note"] = f"{full_arm} has no run for this cell"
            elif base is None:
                entry["note"] = f"{baseline_arm} has no run for this cell"
            elif args.complete_only:
                entry["note"] = "excluded by --complete-only (partial run)"
            else:
                entry["note"] = "no shared sample id between the two arms"
        coverage[f"{task}/{model}"] = entry
        if not chosen:
            continue
        mkey = (task, full.split if full else "test")
        if mkey not in source_cache:
            source_cache[mkey] = _load_sources(*mkey)
        sources = source_cache[mkey]
        full_map, base_map = full.by_sample_id(), base.by_sample_id()
        for sid in chosen:
            order = presentation_order(args.seed, sid, full_arm, baseline_arm)
            a_arm, b_arm = order
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "sample_id": sid,
                    "input": sources.get(sid, ""),
                    "system_a": full_map[sid].get("final_output") if a_arm == full_arm else base_map[sid].get("final_output"),
                    "system_b": full_map[sid].get("final_output") if b_arm == full_arm else base_map[sid].get("final_output"),
                    "a_arm": a_arm,
                    "b_arm": b_arm,
                    "full_n": full.n,
                    "baseline_n": base.n,
                    "full_status": full.status,
                    "baseline_status": base.status,
                    "full_rel": full.rel,
                    "baseline_rel": base.rel,
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    out_rows: List[dict] = []
    for i, row in enumerate(rows, start=1):
        pid = f"P{i:04d}"
        out_rows.append(
            {
                "pair_id": pid,
                "task": row["task"],
                "model": row["model"],
                "input": row["input"],
                "system_a": row["system_a"],
                "system_b": row["system_b"],
                "winner": "",
                "confidence": "",
                "notes": "",
            }
        )
        answer_key[pid] = {
            "A": row["a_arm"],
            "B": row["b_arm"],
            "task": row["task"],
            "model": row["model"],
            "sample_id": row["sample_id"],
            "full_arm": full_arm,
            "baseline_arm": baseline_arm,
            "full_run": row["full_rel"],
            "baseline_run": row["baseline_rel"],
            "full_status": row["full_status"],
            "baseline_status": row["baseline_status"],
            "full_n": row["full_n"],
            "baseline_n": row["baseline_n"],
        }
    return out_rows, answer_key, coverage


def write_package(out_dir: Path, rows: List[dict], answer_key: Dict[str, dict], meta: dict) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / PAIRS_FILE_NAME
    key_path = out_dir / KEY_FILE_NAME
    readme_path = out_dir / README_FILE_NAME

    with pairs_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ANNOTATION_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ANNOTATION_COLUMNS})

    payload = {
        "protocol": "blinded pairwise human evaluation (AAAI2027 experience-refinement experiment)",
        "warning": "DO NOT GIVE THIS FILE TO ANNOTATORS -- it unblinds the package",
        "generated_at": meta.get("generated_at"),
        "seed": meta.get("seed"),
        "full_arm": meta.get("full_arm"),
        "baseline_arm": meta.get("baseline_arm"),
        "arms_selected_by": meta.get("arms_selected_by"),
        "n_pairs": len(rows),
        "runs": meta.get("runs", []),
        "coverage": meta.get("coverage", {}),
        "partial_runs_included": meta.get("partial_runs_included", []),
        "pairs": answer_key,
    }
    tmp = key_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(key_path)

    readme_path.write_text(_readme_text(rows, meta), encoding="utf-8")
    return {"pairs": pairs_path, "key": key_path, "readme": readme_path}


def _readme_text(rows: List[dict], meta: dict) -> str:
    coverage = meta.get("coverage", {})
    lines = [
        "# Blinded pairwise human evaluation",
        "",
        f"Generated: {meta.get('generated_at')}  |  seed: {meta.get('seed')}  |  "
        f"pairs: {len(rows)}",
        "",
        "## What you are annotating",
        "",
        f"`{PAIRS_FILE_NAME}` contains {len(rows)} pairs.  Each row is one task input",
        "with two anonymised system outputs, **System A** and **System B**, in a",
        "randomised order.  The systems are two different configurations of the same",
        "refinement pipeline; the row does not say which is which and neither does",
        "the file name or the pair id.",
        "",
        "For every row, fill in:",
        "",
        "* `winner` -- `A`, `B`, or `tie`.  **Ties are allowed and are a real answer**:",
        "  use `tie` whenever the two outputs are equally good (including when they are",
        "  identical, or when the difference is too small to defend).  Do not force a",
        "  choice.",
        "* `confidence` -- how sure you are: `1` = guess, `2` = fairly sure,",
        "  `3` = very sure.  A confident `tie` is fine.",
        "* `notes` -- optional: what decided it (fluency, adequacy, meaning error,",
        "  hallucination, truncation, ...).",
        "",
        "Judge each output against the task input only.  Do not try to guess which",
        "system produced which output, and do not open the answer key.",
        "",
        "## Files",
        "",
        f"* `{PAIRS_FILE_NAME}` -- open this one; it is the annotation sheet.",
        f"* `{KEY_FILE_NAME}` -- **do not open while annotating**; it maps each pair id",
        "  to the arms behind System A / System B and is only used after annotations",
        "  are frozen, to compute the win/tie rates.",
        "* `README.md` -- this file.",
        "",
        "## How the package was built",
        "",
        f"* Full arm: `{meta.get('full_arm')}`; comparison arm: `{meta.get('baseline_arm')}`",
        f"  (selection: {meta.get('arms_selected_by_short') or meta.get('arms_selected_by')}).",
        "* Every pair is the **same sample id** under both arms; sample ids are never",
        "  mixed within a pair.",
        "* The A/B order is drawn from `sha256(seed|sample_id)`, so it is reproducible",
        "  and independent of the pair's position in the file.",
        "* Per-cell coverage (realised pairs out of the pairs the traces could supply):",
        "",
        "| task | model | realised | available | full run n | baseline run n | note |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for cell, entry in coverage.items():
        lines.append(
            f"| {entry['task']} | {entry['model']} | {entry['realised']} | "
            f"{entry['available_pairs']} | {entry['full_n']} ({entry['full_status']}) | "
            f"{entry['baseline_n']} ({entry['baseline_status']}) | {entry['note']} |"
        )
    partial = meta.get("partial_runs_included", [])
    lines += [
        "",
        "## Caveats recorded at generation time",
        "",
        f"* Runs are COMPLETE only when the trace holds at least the full test size",
        f"  (`{TEST_SAMPLES}`); anything shorter is PARTIAL.  "
        + (f"Partial runs used here: {', '.join(partial)}."
           if partial else "No partial run was used."),
        "* A pair can only be shown when both arms produced a final output for that",
        "  sample id; the `available` column above is that intersection.",
        "",
        "## Aggregating the results",
        "",
        f"After annotation, join `{PAIRS_FILE_NAME}` with `{KEY_FILE_NAME}` on",
        "`pair_id`, count wins per arm, and report win/tie/loss rates with ties kept",
        "as a third category (do not silently drop them).",
        "",
    ]
    return "\n".join(lines)


def verify_blinding(pairs_path: Path, arm_names: Sequence[str], log=print) -> int:
    """Count occurrences of every arm name in the annotation file (proof of blinding)."""
    text = pairs_path.read_text(encoding="utf-8")
    hits = 0
    for arm in arm_names:
        c = text.count(arm)
        hits += c
        log(f"[blind] occurrences of {arm!r} in {pairs_path.name}: {c}")
    if hits:
        log(f"[blind] WARNING: {hits} arm-name occurrence(s) found in the annotation file")
    else:
        log(f"[blind] OK: none of {list(arm_names)} appears in {pairs_path.name}")
    return hits


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="export_human_eval.py",
        description="Phase 10: blinded pairwise human-eval export "
                    "(Full vs the strongest non-experience arm).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Blinding: the annotation CSV never contains an arm name; the mapping lives "
               f"in {KEY_FILE_NAME}, which annotators must not open.",
    )
    ap.add_argument("--arms", default=f"{FULL_ARM_DEFAULT},auto",
                    help="FULL,BASELINE.  BASELINE may be 'auto' (see --baseline-arm).")
    ap.add_argument("--baseline-arm", default=None,
                    help="Baseline arm name or 'auto'; overrides the second entry of --arms.")
    ap.add_argument("--runs", nargs="*", default=list(DEFAULT_RUN_DIRS),
                    help="Run directories to scan (default: the production trees).")
    ap.add_argument("--tasks", nargs="*", default=list(EVAL_TASKS), help="Tasks to cover.")
    ap.add_argument("--models", nargs="*", default=list(EVAL_MODELS), help="Models to cover.")
    ap.add_argument("--limit", type=int, default=200, help="Maximum number of pairs (<=200 for the paper).")
    ap.add_argument("--seed", type=int, default=42, help="Seed for A/B randomisation and pair order.")
    ap.add_argument("--out-dir", default=str(EXP_ROOT / "human_eval"), help="Output directory.")
    ap.add_argument("--split", default="test", help="Manifest/trace split to use.")
    ap.add_argument("--allow-tagged", action="store_true",
                    help="Also use traces under sweep tags (e.g. gate_alpha a0.5/rand).")
    ap.add_argument("--complete-only", action="store_true",
                    help="Only pair runs that reached the full test size.")
    ap.add_argument("--strict-strength", action="store_true",
                    help="Disable the coverage tie-break: always take the top-ranked arm.")
    ap.add_argument("--drop-last-line", action="store_true",
                    help="Also drop the final (terminated) line of every trace file.")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan; write nothing.")
    return ap.parse_args(argv)


def resolve_arms(args) -> Tuple[str, Optional[str]]:
    parts = [p.strip() for p in str(args.arms).replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise SystemExit(f"--arms must be FULL,BASELINE (got {args.arms!r})")
    full, base = parts
    if args.baseline_arm:
        base = args.baseline_arm.strip()
    base = None if base.lower() in ("auto", "") else base
    return full, base


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.tasks = list(args.tasks)
    args.models = list(args.models)
    full_arm, baseline_arg = resolve_arms(args)

    if args.limit > 200:
        print(f"[WARN] --limit {args.limit} exceeds the paper's cap of 200 pairs")
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    runs = discover_runs(
        args.runs, models=args.models, tasks=args.tasks, splits=[args.split],
        allow_tagged=args.allow_tagged, force_drop_last=args.drop_last_line,
    )
    grouped = group_runs(runs)
    runs = sorted(grouped.values(), key=lambda r: r.rel)
    if not runs:
        print(f"[FATAL] no usable traces under {args.runs} for tasks={args.tasks} models={args.models}")
        return 2

    print(f"[data] {len(runs)} trace file(s) usable; arms present: "
          f"{sorted({r.arm for r in runs})}")

    selected_by = "explicit --baseline-arm/--arms"
    if baseline_arg is None:
        choice = choose_baseline_arm(runs, full_arm, args.tasks, args.models,
                                     coverage_tiebreak=not args.strict_strength)
        print_baseline_evidence(choice)
        baseline_arm = choice.arm
        selected_by = choice.reason
        if baseline_arm is None:
            print(f"[FATAL] no non-experience arm has a COMPLETE run under {args.runs}; "
                  "cannot build the Full-vs-baseline package.  Wait for the ablation arm "
                  "(sr_j_stop / sr_j_fixed / no_experience) to finish, or pass "
                  "--complete-only off / --allow-tagged / a different --runs scope.")
            return 2
    else:
        baseline_arm = baseline_arg
        have = [r for r in runs if r.arm == baseline_arm]
        if not have:
            print(f"[FATAL] --baseline-arm {baseline_arm!r} has NO run at all under {args.runs}. "
                  f"Arms seen: {sorted({r.arm for r in runs})}")
            return 2
        complete = [r for r in have if r.complete]
        if not complete:
            print(f"[FATAL] --baseline-arm {baseline_arm!r} has no COMPLETE run "
                  f"(sizes: {[(r.task, r.n, r.expected_n) for r in have]}). "
                  "--complete-only cannot help here; wait for the run to finish.")
            return 2
        print(f"[baseline] using explicitly requested arm {baseline_arm!r}; "
              f"complete runs: {[(r.task, r.model, r.n) for r in complete]}")

    full_runs = [r for r in runs if r.arm == full_arm]
    if not full_runs:
        print(f"[FATAL] full arm {full_arm!r} has no run under {args.runs}; "
              f"arms seen: {sorted({r.arm for r in runs})}")
        return 2

    try:
        rows, answer_key, coverage = build_pairs(args, runs, full_arm, baseline_arm)
    except FileNotFoundError as exc:
        print(f"[FATAL] {exc}")
        return 2

    # ---- report -----------------------------------------------------------
    print()
    print(f"[pairs] full={full_arm!r} baseline={baseline_arm!r} "
          f"(selection: {selected_by})")
    print(f"  {'cell':28s}{'realised':>9s}{'available':>10s}{'full n':>9s}{'base n':>9s}  status")
    for cell, entry in coverage.items():
        print(f"  {cell:28s}{entry['realised']:>9d}{entry['available_pairs']:>10d}"
              f"{entry['full_n']:>9d}{entry['baseline_n']:>9d}  "
              f"full={entry['full_status']}, baseline={entry['baseline_status']}"
              + (f" -- {entry['note']}" if entry["note"] else ""))
    empty = [c for c, e in coverage.items() if e["realised"] == 0]
    if empty:
        print(f"[coverage] {len(empty)} required cell(s) produced no pair: {empty}")
    print(f"[pairs] total realised pairs: {len(rows)} (limit {args.limit})")
    # only runs that actually contributed a pair count as "used"
    used_cells = {c: e for c, e in coverage.items() if e["realised"] > 0}
    partial_used = sorted(
        {e["full_run"] for e in used_cells.values()
         if e["full_status"] == "partial" and e["full_run"]}
        | {e["baseline_run"] for e in used_cells.values()
           if e["baseline_status"] == "partial" and e["baseline_run"]}
    )
    if partial_used:
        print(f"[coverage] PARTIAL runs used (labelled in the answer key/README): {partial_used}")
    n_full_a = sum(1 for v in answer_key.values() if v["A"] == full_arm)
    print(f"[blind] A/B randomisation: full arm shown as A in {n_full_a}/{len(rows)} pairs, "
          f"as B in {len(rows) - n_full_a}/{len(rows)}")

    if not rows:
        print("[FATAL] no pair could be built; nothing written.")
        return 2

    if args.dry_run:
        print("\n[dry-run] no file written.  Files that would be created:")
        out_dir = Path(args.out_dir)
        print(f"  {out_dir / PAIRS_FILE_NAME}")
        print(f"  {out_dir / KEY_FILE_NAME}")
        print(f"  {out_dir / README_FILE_NAME}")
        print(f"[dry-run] first pair would be: pair_id=P0001 task={rows[0]['task']} "
              f"model={rows[0]['model']} input={rows[0]['input'][:60]!r} "
              f"A={answer_key['P0001']['A']!r} B={answer_key['P0001']['B']!r}")
        return 0

    import datetime

    meta = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "full_arm": full_arm,
        "baseline_arm": baseline_arm,
        "arms_selected_by": selected_by,
        "arms_selected_by_short": (
            f"auto-selected {baseline_arm!r} as the strongest non-experience arm available "
            "-- per-cell means, ranks and the coverage tie-break are recorded under "
            "'arms_selected_by' in the answer key"
            if baseline_arg is None else selected_by),
        "coverage": coverage,
        "partial_runs_included": partial_used,
        "runs": [
            {"path": r.rel, "arm": r.arm, "task": r.task, "model": r.model, "n": r.n,
             "expected_n": r.expected_n, "status": r.status}
            for r in runs
        ],
    }
    paths = write_package(Path(args.out_dir), rows, answer_key, meta)
    print("\n[written]")
    for name, p in paths.items():
        print(f"  {name:6s} {p}")
    verify_blinding(paths["pairs"], [full_arm, baseline_arm])
    print(f"[done] {len(rows)} blinded pairs; answer key kept separately in {paths['key'].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
