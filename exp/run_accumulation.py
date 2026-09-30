#!/usr/bin/env python
"""Phase 8: experience accumulation -- does a library grown online help later items?

The experiment walks a fixed auxiliary stream (the manifest split
``accumulation``, 128 items per model-task) exactly once, in a fixed order, and
periodically measures what the library has bought so far:

* processing an item runs the *normal* adaptive refinement loop and then commits
  that item's transitions to the library (the same online update the
  ``full_online`` arm performs), so the library only ever contains experience
  from items already seen;
* after 0 / 32 / 64 / 96 / 128 processed items the library is snapshotted and
  **evaluated on a fixed evaluation set** -- the task's TEST manifest, capped by
  ``--eval-limit`` (default 200).  The evaluation never writes to the library:
  it only reads a snapshot and produces traces.

Three conditions -- the three accumulation methods of the plan -- are compared at
every checkpoint (``--conditions`` selects a subset; all three run by default):

``dynamic``       **Full-Dynamic**: the library as accumulated so far, i.e. the
                  bootstrap library plus *every* newly observed unit;
``frozen``        **Frozen**: the initial bootstrap library only (no accumulation);
``no_negative``   **NoNegativeExperience**: Full-Dynamic except that a newly
                  observed unit whose *offline outcome was negative* -- the
                  refinement made quality worse, ``delta_offline < 0`` -- is
                  never admitted to the library.

The conditions share ONE stream run per order: the stream's own refinement loop
retrieves from the Full-Dynamic library, so the units the stream *observes* are
identical for all conditions, and the ``no_negative`` library is then exactly
the Full-Dynamic library minus the negative units -- the same units in the same
construction order, retrieved by the same retriever, so the single-variable
"negative experience" contrast holds.  The filter is applied at commit time to
the newly observed units only: the bootstrap library is the shared prior of all
three conditions and is deliberately not re-filtered.  ``is_negative_outcome``
is the one definition of "negative" (``delta_offline < 0``; for a unit whose
delta was never recorded -- every unit of the shipped bootstrap libraries, whose
``delta_offline`` is null -- the judge's ``worse`` verdict is the fallback).

Because the evaluation set, the initial drafts (stored Direct-Zero outputs), the
round budget and the batch size are all held fixed, the difference between the
conditions is *quality at fixed compute*.  The same traces also give the second
curve: mean refinement rounds / tokens per sample as experience grows.

``--orders N`` repeats the stream in N different orders (order 0 is the manifest
order, orders 1..N-1 are seeded shuffles; default 3).  An order's permutation is
a pure function of the base seed and the order index (``order_seed``), the orders
are distinct by construction, and they are drawn once at plan time, so the plan
``--dry-run`` prints is exactly the order the run uses and a re-run reproduces
every permutation byte for byte.  Spread across orders is summarised with
``core.stats.order_variance``; the frozen condition does not depend on the order
at all, so by default it is measured once per model-task and reused
(``--no-reuse-frozen`` re-measures it for every order/checkpoint).

No resume, by design: the accumulating library is *order-dependent* (each stream
item is refined against the library its predecessors built), so a partial run can
never be resumed -- replaying only the tail would hand the condition a library it
could never have had.  Every artefact is written from scratch, so a crashed or
interrupted run simply starts over.  Compare ``run_experiment.py``'s
``if online and prior:`` guard, which discards a persisted prefix for the same
reason; nothing here may weaken that rule.

Usage
-----
    # validate the setup and print the plan; loads no model, touches no GPU
    python run_accumulation.py --dry-run

    # the real run (needs a free GPU)
    python run_accumulation.py --gpu 3 --models qwen3-8b --tasks wmt19_en_zh \
        --orders 3 --eval-limit 200

    # quick debug pass: 32 stream items, 20 eval items, one order, two methods
    python run_accumulation.py --gpu 3 --models qwen3-8b --tasks wmt19_en_zh \
        --limit 32 --eval-limit 20 --orders 1 --conditions dynamic,frozen

Outputs
-------
    runs/accumulation/<model>/<task>/order<k>/library_c<NNN>.jsonl  Full-Dynamic snapshots
    runs/accumulation/<model>/<task>/order<k>/stream.{jsonl,csv}    stream traces (one per order)
    runs/accumulation/<model>/<task>/order<k>/eval_c<NNN>_<cond>.jsonl  eval traces
    runs/accumulation/<model>/<task>/order<k>/no_negative/library_c<NNN>.jsonl
    runs/accumulation/<model>/<task>/order<k>/no_negative/eval_c<NNN>_no_negative.jsonl
    runs/accumulation/<model>/<task>/frozen/eval_c0000_frozen.jsonl
    scores/accumulation_curve.csv                one row per (order, checkpoint, condition)
    scores/accumulation_order_variance.json      spread across orders (Full-Dynamic)
    scores/accumulation_order_variance_no_negative.json    spread (NoNegativeExperience)

``dynamic`` keeps the historical artefact layout (files directly under
``order<k>/``); each additional accumulating condition gets its own subdirectory
there, because its snapshot has different contents and must not collide with the
Full-Dynamic one -- and because every path a pre-existing run wrote stays valid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregate import _write_csv  # noqa: E402  (aggregate.py's CSV convention)
from core import EXP_ROOT, MODELS, TASKS, ensure_gpu_whitelist  # noqa: E402
from core.bm25_fields import ExperienceRetriever  # noqa: E402
from core.experience import experience_dir, load_experiences, save_experiences  # noqa: E402
from core.manifest import SampleRef, check_isolation, manifest_path, read_manifest  # noqa: E402
from core.pipeline import Pipeline, RunConfig, SampleTrace, transitions_to_experiences  # noqa: E402
from core.scoring import PRIMARY_METRIC, Scorer  # noqa: E402
from core.stats import order_variance, paired_bootstrap  # noqa: E402

SCORES_DIR = EXP_ROOT / "scores"


def order_variance_path(condition: str) -> Path:
    """Order-variance file of one accumulating condition.

    Full-Dynamic keeps the historical name (``accumulation_order_variance.json``)
    so the artefact of a pre-existing run is reproduced unchanged; every further
    condition gets its own file keyed by the condition label.
    """
    if condition == "dynamic":
        return SCORES_DIR / "accumulation_order_variance.json"
    return SCORES_DIR / f"accumulation_order_variance_{condition}.json"


_TRACE_CSV_FIELDS = [
    "order", "checkpoint", "condition", "task", "model", "sample_id", "n_rounds",
    "initial_metric_offline", "final_metric_offline", "total_tokens", "latency_s",
    "n_calls",
]

#: The three accumulation methods of the Phase 8 plan.  The values are the
#: labels written into every artefact (the ``condition`` column / file suffix):
#: ``dynamic`` is what the plan calls Full-Dynamic, and it is kept under that
#: name because every curve/order-variance artefact predating this change is
#: keyed by it.
CONDITIONS = ("dynamic", "frozen", "no_negative")

#: Plan name of each condition, for the human-readable plan/dry-run output.
PLAN_NAMES = {
    "dynamic": "Full-Dynamic",
    "frozen": "Frozen",
    "no_negative": "NoNegativeExperience",
}

#: Conditions whose library grows with the stream (frozen never changes).
ACCUMULATING_CONDITIONS = ("dynamic", "no_negative")

#: What a newly observed unit must satisfy to enter each accumulating library.
ADMISSION_RULE = {
    "dynamic": "every observed unit",
    "no_negative": "every observed unit with delta_offline >= 0 (no worse-than-before refinement)",
}

DEFAULT_CONDITIONS = ",".join(CONDITIONS)

#: How many distinct permutations to try before giving up on an order.  A
#: collision between two index-addressed shuffles of a >=10-item stream is
#: already a ~1/3.6M event, so this is only ever reached by a pathologically
#: short stream, where a distinct order is impossible anyway.
_MAX_ORDER_ATTEMPTS = 64


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def resolve(names: str, all_values: Sequence[str]) -> List[str]:
    if names == "all":
        return list(all_values)
    return [x.strip() for x in names.split(",") if x.strip()]


def checkpoints_for(n_stream: int, chunk: int) -> List[int]:
    """0, chunk, 2*chunk, ... plus the exact end of the stream."""
    cps = list(range(0, n_stream + 1, chunk))
    if not cps or cps[-1] != n_stream:
        cps.append(n_stream)
    return cps


def order_seed(base_seed: int, order_index: int, attempt: int = 0) -> int:
    """RNG seed of one stream order: a pure function of the order index.

    ``hashlib`` rather than ``base_seed + order_index`` so that neighbouring
    orders are not seeded with neighbouring Mersenne-Twister states, and so the
    derivation is explicit and platform-independent.  Same inputs -> same
    permutation, forever: a re-run of the same ``--orders``/``--seed`` on the
    same stream reproduces every order exactly.
    """
    payload = f"accumulation-order|{int(base_seed)}|{int(order_index)}|{int(attempt)}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def ordered_stream(
    stream: Sequence[SampleRef], order_index: int, seed: int,
    taken: Sequence[Sequence[str]] = (),
) -> Tuple[List[SampleRef], str, str]:
    """The stream in one order.

    Order 0 is the manifest order ("natural"); order k >= 1 is the seeded
    shuffle produced by ``order_seed(seed, k)``.  ``taken`` holds the
    sample-id sequences of the orders already drawn for this stream; a candidate
    that repeats one of them is re-drawn with the next attempt counter, so the
    ``--orders N`` orders are *distinct by construction* while staying a
    deterministic function of the order index.  Streams shorter than two items
    have only one order each and are not shuffled.
    """
    items = list(stream)
    if order_index == 0:
        return items, "natural", ""
    if len(items) < 2:
        # nothing to permute; distinctness is mathematically impossible
        return items, f"shuffle{order_index}", ""
    for attempt in range(_MAX_ORDER_ATTEMPTS):
        idx = list(range(len(items)))
        random.Random(order_seed(seed, order_index, attempt)).shuffle(idx)
        ids = [items[i].sample_id for i in idx]
        if all(ids != list(prev) for prev in taken):
            digest = hashlib.sha256(
                ",".join(str(i) for i in idx).encode("utf-8")
            ).hexdigest()[:16]
            return [items[i] for i in idx], f"shuffle{order_index}", digest
    raise RuntimeError(
        f"cannot draw a distinct stream order {order_index}: {len(taken)} order(s) "
        f"already use every arrangement of these {len(items)} item(s)"
    )


def plan_orders(
    stream: Sequence[SampleRef], n_orders: int, seed: int
) -> List[dict]:
    """All ``n_orders`` orders of one stream, drawn once, in index order.

    Returns ``[{"index", "label", "digest", "stream"}, ...]``.  Drawing the
    orders at plan time is what lets ``--dry-run`` print exactly the
    permutations the real run will use (``plan["orders"][k]``) instead of
    re-deriving them later.
    """
    drawn: List[dict] = []
    taken: List[List[str]] = []
    for k in range(n_orders):
        items, label, digest = ordered_stream(stream, k, seed, taken=taken)
        taken.append([r.sample_id for r in items])
        drawn.append({"index": k, "label": label, "digest": digest, "stream": items})
    return drawn


def plan_model_task(model: str, task: str, args) -> dict:
    """Everything a run needs, validated.  No model, no GPU, no generation."""
    stream_path = manifest_path(task, "accumulation")
    test_path = manifest_path(task, "test")
    lib_path = experience_dir(model, task) / "initial.jsonl"
    missing = [str(p) for p in (stream_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing manifest(s): {missing}")
    stream = read_manifest(stream_path)
    if args.limit is not None:
        stream = stream[: args.limit]
    eval_refs = read_manifest(test_path)[: args.eval_limit]
    library = load_experiences(lib_path)
    return {
        "model": model,
        "task": task,
        "stream_all": read_manifest(stream_path),
        "stream": stream,
        "orders": plan_orders(stream, args.orders, args.seed),
        "eval_refs": eval_refs,
        "library": library,
        "library_path": lib_path,
        "library_exists": lib_path.exists(),
        "n_initial": len(library),
        "checkpoints": checkpoints_for(len(stream), args.chunk_size),
    }


# --------------------------------------------------------------------------- #
# accumulation conditions
# --------------------------------------------------------------------------- #


def parse_conditions(names: str) -> List[str]:
    """``--conditions`` -> canonical condition list (raises ValueError).

    Always returned in ``CONDITIONS`` order so artefacts and curve rows do not
    depend on the order the flag was written in.
    """
    wanted = [x.strip() for x in names.split(",") if x.strip()]
    if not wanted:
        raise ValueError("--conditions must name at least one condition")
    unknown = [c for c in wanted if c not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"unknown condition(s) {unknown}; known: {list(CONDITIONS)}"
        )
    if len(set(wanted)) != len(wanted):
        raise ValueError(f"duplicate condition(s) in {names!r}")
    return [c for c in CONDITIONS if c in wanted]


def is_negative_outcome(exp: Experience) -> bool:
    """Did this refinement make quality worse?

    The **offline outcome** is the authority: ``delta_offline`` is the gold-metric
    delta the pipeline records for the transition, and a negative delta is
    exactly "the refinement made quality worse".  For a unit without a recorded
    delta (all units of the shipped bootstrap libraries have
    ``delta_offline = null``, and the format allows it) the judge's ``worse``
    verdict is the fallback, so the filter stays well defined for any library.
    Ties (``delta_offline == 0``) and ``uncertain`` units are *not* negative --
    NoNegativeExperience drops negatives, it is not the ``positive_only`` arm.
    """
    if exp.delta_offline is not None:
        return float(exp.delta_offline) < 0.0
    return exp.verdict == "worse"


def filter_for_condition(condition: str, units: Sequence[Experience]) -> List[Experience]:
    """The admission rule of one accumulating condition, applied to new units.

    This is the *only* difference between the accumulating conditions: the
    stream run, the commit points, the ordering and the retriever construction
    are shared, so each condition's library is a subsequence of Full-Dynamic's.

    Note the deliberate difference from ``core.experience.filter_positive_only``
    (the ablation arm the v5 plan calls ``PositiveOnly``, renamed from
    ``NoNegativeExperience``): that arm keeps only judge-accepted units
    (``verdict == "better" and order_consistent``) and therefore drops ties and
    uncertain units as well.  The accumulation method asked for here drops only
    the units whose *offline outcome* was negative; ties/uncertain units stay, so
    Full-Dynamic minus this filter is exactly Full-Dynamic minus the negative
    experiences.
    """
    if condition == "dynamic":
        return list(units)
    if condition == "no_negative":
        return [e for e in units if not is_negative_outcome(e)]
    raise ValueError(f"condition {condition!r} does not accumulate experience")


def condition_dir(base: Path, condition: str) -> Path:
    """Artefact directory of one condition, relative to ``order<k>/``.

    ``dynamic`` keeps the historical layout (files directly under
    ``order<k>/``) so every path written before this change stays valid; a
    further accumulating condition gets its own subdirectory because its
    snapshot contents differ and must not overwrite the Full-Dynamic one.
    """
    if condition not in ACCUMULATING_CONDITIONS or condition == "dynamic":
        return base
    return base / condition


# --------------------------------------------------------------------------- #
# traces
# --------------------------------------------------------------------------- #


class TraceWriter:
    """JSONL + CSV pair, flushed as chunks complete (same shape as run_experiment)."""

    def __init__(self, stem: Path, meta: Dict[str, object]):
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.jf = stem.with_suffix(".jsonl").open("w", encoding="utf-8")
        self.cf = stem.with_suffix(".csv").open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.cf, fieldnames=_TRACE_CSV_FIELDS, restval="")
        self.writer.writeheader()
        self.jsonl_path = stem.with_suffix(".jsonl")

    def sink(self, traces: Sequence[SampleTrace], _done: int, _total: int) -> None:
        for tr in traces:
            self.jf.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
            self.writer.writerow({
                **self.meta,
                "sample_id": tr.sample_id, "n_rounds": len(tr.rounds),
                "initial_metric_offline": tr.initial_metric_offline,
                "final_metric_offline": tr.final_metric_offline,
                "total_tokens": tr.cost["total_tokens"],
                "latency_s": tr.cost["latency_s"], "n_calls": tr.cost["n_calls"],
            })
        self.jf.flush()
        self.cf.flush()

    def close(self) -> None:
        self.jf.close()
        self.cf.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# the two passes
# --------------------------------------------------------------------------- #


def eval_config(
    model: str, task: str, library: Sequence, args, snapshot_id: str
) -> RunConfig:
    """The evaluation configuration.

    ``draft_source="stored"`` is required (and valid) for the test split: every
    checkpoint then starts from the byte-identical Direct-Zero draft, which is
    what makes the checkpoints and the frozen/dynamic conditions paired.
    """
    return RunConfig(
        task=task, model=model,
        experience_mode="full" if library else "none",
        stop_mode="adaptive",
        alpha=args.alpha, k=args.k, max_rounds=args.max_rounds,
        seed=args.seed, snapshot_id=snapshot_id,
        draft_source="stored",
    )


def run_eval(
    llm, model: str, task: str, library: Sequence, eval_refs: Sequence[SampleRef],
    args, out_dir: Path, *, order_label: str, checkpoint: int, condition: str,
) -> List[SampleTrace]:
    """Evaluate on the fixed eval set.  Reads a library, never writes one."""
    from core.batched_pipeline import BatchedPipeline

    snapshot_id = f"acc_{order_label}_c{checkpoint:04d}_{condition}"
    cfg = eval_config(model, task, library, args, snapshot_id)
    retriever = ExperienceRetriever(library) if library else None
    pipe = BatchedPipeline(
        cfg, retriever, list(library), llm, cache=None, batch_size=args.batch_size
    )
    stem = condition_dir(
        out_dir / model / task / order_label, condition
    ) / f"eval_c{checkpoint:04d}_{condition}"
    with TraceWriter(stem, {
        "order": order_label, "checkpoint": checkpoint, "condition": condition,
        "task": task, "model": model,
    }) as tw:
        traces = pipe.run(eval_refs, sink=tw.sink)
    return traces


def summarize_eval(
    traces: Sequence[SampleTrace], eval_refs: Sequence[SampleRef], task: str,
    *, model: str, order_label: str, checkpoint: int, condition: str,
    n_library: int, n_initial: int, duration: float,
    frozen: Optional[Sequence[SampleTrace]], frozen_reused: bool, bootstrap: int,
) -> dict:
    scorer = Scorer(task)
    refs = {r.sample_id: r.reference for r in eval_refs}
    pairs = [(refs.get(t.sample_id, ""), t.final_output) for t in traces]
    corpus = scorer.score_corpus(pairs)
    primary = PRIMARY_METRIC[task]
    finals = [t.final_metric_offline or 0.0 for t in traces]
    n = max(1, len(traces))
    n_refine = sum(1 for t in traces for r in t.rounds if r.controller_action == "REFINE")
    n_accepted = sum(1 for t in traces for r in t.rounds if r.accepted)
    row = {
        "model": model, "task": task, "order": order_label, "checkpoint": checkpoint,
        "processed_items": checkpoint, "condition": condition,
        "n_library": n_library, "library_growth": n_library - n_initial,
        "n_eval": len(traces),
        "primary_metric": primary,
        "corpus_metric": round(corpus.get(primary, float("nan")), 4),
        "mean_sample_metric": round(sum(finals) / n, 4),
        "mean_rounds": round(sum(len(t.rounds) for t in traces) / n, 4),
        "tokens_per_sample": round(sum(t.cost["total_tokens"] for t in traces) / n, 1),
        "calls_per_sample": round(sum(t.cost["n_calls"] for t in traces) / n, 2),
        "refine_attempts": n_refine,
        "accept_rate": round(n_accepted / n_refine, 4) if n_refine else 0.0,
        "frozen_reused": int(frozen_reused),
        "duration_s": round(duration, 1),
    }
    if frozen is not None and len(frozen) == len(traces):
        f_finals = [t.final_metric_offline or 0.0 for t in frozen]
        res = paired_bootstrap(finals, f_finals, n_resamples=bootstrap)
        row.update({
            "frozen_mean_sample_metric": round(res.mean_b, 4),
            "delta_vs_frozen": round(res.diff, 4),
            "ci_low": round(res.ci_low, 4), "ci_high": round(res.ci_high, 4),
            "p_two_sided": round(res.p_two_sided, 5),
        })
    return row


def process_stream_item(
    llm, pipe: Pipeline, ref: SampleRef, index: int, model: str, library: List,
    retriever, provenance: str = "accumulation",
    extra_libraries: Optional[Dict[str, List[Experience]]] = None,
) -> Tuple[SampleTrace, int]:
    """One stream item, then commit its transitions (online update).

    ``library`` is the Full-Dynamic library **and** the library the stream's own
    refinement loop retrieves from: the stream is the data-collection process and
    is identical for every condition, so the units it observes cannot depend on
    which condition is being measured.  ``extra_libraries`` mirrors the very same
    newly observed units into the other accumulating conditions through their
    admission rule (``filter_for_condition``), preserving the observation order,
    so every such library is a subsequence of the Full-Dynamic one.  With
    ``extra_libraries=None`` the behaviour is exactly the pre-existing one.
    """
    trace = pipe.run_sample(ref, index)
    # NOTE: this is the Phase-8 *admission-rule comparison*, not the v2 online
    # arm.  Its whole design is that one stream keeps EVERY observed transition
    # and the other conditions are subsequences of it selected by
    # ``filter_for_condition``; applying only_improving here would collapse
    # ``dynamic`` and ``no_negative`` into the same library and make the study
    # vacuous.  The v2 evolution rule ("store wrong->right pairs only") is
    # enforced in the full_online arm (BatchedPipeline / run_experiment).
    # Re-specifying this study for v2 semantics is deferred and tracked in
    # reports/progress.md; it is not part of the E2 campaign.
    new = transitions_to_experiences(trace, model, source_input=ref.source,
                                     provenance=provenance)
    if new:
        library.extend(new)
        for cond, lib in (extra_libraries or {}).items():
            kept = filter_for_condition(cond, new)
            if kept:
                lib.extend(kept)
        retriever = ExperienceRetriever(library)
        pipe.retriever = retriever
        pipe.library = library
    return trace, len(new)


# --------------------------------------------------------------------------- #
# dry run / reporting
# --------------------------------------------------------------------------- #


def print_plan(plans: Sequence[dict], args, conditions: Sequence[str]) -> None:
    header = (
        f"{'model':12s}{'task':14s}{'stream':>7s}{'eval':>6s}{'lib0':>6s}"
        f"{'chunk':>6s}{'checkpoints':>28s}{'orders':>7s}{'methods':>8s}"
    )
    print(header)
    print("-" * len(header))
    for p in plans:
        cps = ",".join(str(c) for c in p["checkpoints"])
        stream_n = (
            f"{len(p['stream'])}"
            if len(p["stream"]) == len(p["stream_all"])
            else f"{len(p['stream'])}/{len(p['stream_all'])}"
        )
        print(
            f"{p['model']:12s}{p['task']:14s}{stream_n:>7s}"
            f"{len(p['eval_refs']):6d}{p['n_initial']:6d}{args.chunk_size:6d}"
            f"{cps:>28s}{args.orders:7d}{len(conditions):8d}"
        )


def print_methods(conditions: Sequence[str]) -> None:
    """The three accumulation methods of the plan, with their admission rule."""
    print(f"accumulation methods ({len(conditions)} selected by --conditions):")
    for cond in conditions:
        rule = (
            "bootstrap library only (no accumulation)"
            if cond == "frozen"
            else ADMISSION_RULE[cond]
        )
        print(f"  {PLAN_NAMES[cond]:21s} label={cond:12s} library = {rule}")


def print_orders(plans: Sequence[dict], args) -> None:
    """The drawn stream orders, with the permutation digest per (task, order)."""
    print(f"stream orders (--orders {args.orders}, base seed {args.seed}; "
          "digest = sha256 of the item permutation, '' = manifest order):")
    seen: set = set()
    for p in plans:
        key = (p["task"], len(p["stream"]))
        if key in seen:
            continue
        seen.add(key)
        drawn = " ".join(
            f"order{o['index']}={o['label']}#{o['digest'] or 'identity'}"
            for o in p["orders"]
        )
        print(f"  {p['task']:14s} n={len(p['stream']):4d} {drawn}")
    print("  (drawn once at plan time from order_seed(seed, index): a re-run "
          "reproduces every permutation exactly)")


def print_schedule(plans: Sequence[dict], args, conditions: Sequence[str],
                   out_dir: Path, orders: int) -> None:
    """When the library is snapshotted, and where each condition's copy goes."""
    print(f"snapshot schedule: after every {args.chunk_size} processed stream items "
          f"+ the exact end of the stream")
    cps = ",".join(str(c) for c in plans[0]["checkpoints"])
    print(f"  checkpoints (processed items, n_stream={len(plans[0]['stream'])}): {cps}")
    print(f"  every checkpoint x {orders} order(s) x {len(conditions)} condition(s):")
    for cond in conditions:
        if cond == "frozen":
            print(f"    {PLAN_NAMES[cond]:21s} -> no snapshot "
                  "(the bootstrap library never changes)")
            continue
        rel = condition_dir(Path("order<k>"), cond) / "library_c<NNNN>.jsonl"
        print(f"    {PLAN_NAMES[cond]:21s} -> {out_dir}/<model>/<task>/{rel}")


def estimate_calls(plans: Sequence[dict], args,
                   conditions: Sequence[str]) -> Tuple[int, int]:
    """(optimistic, pessimistic) LLM-call estimate for the whole run.

    Stream items pay a draft plus, per round, 1 controller + 1 refine + 2 judge;
    eval items reuse the stored draft.  Optimistic = every adaptive loop stops
    after one round, pessimistic = every loop uses the full round budget.  Every
    accumulating condition selected by ``--conditions`` is evaluated at every
    checkpoint; the frozen baseline is measured once per model-task (unless
    ``--no-reuse-frozen``) and reused by all of them.
    """
    n_acc = sum(1 for c in conditions if c in ACCUMULATING_CONDITIONS)
    lo = hi = 0
    for p in plans:
        n_cp = len(p["checkpoints"])
        stream = args.orders * len(p["stream"])
        evals = n_acc * args.orders * n_cp
        if "frozen" in conditions:
            evals += 1 if args.reuse_frozen else args.orders * n_cp
        lo += stream * (1 + 4) + evals * len(p["eval_refs"]) * 4
        hi += stream * (1 + 4 * args.max_rounds) + evals * len(p["eval_refs"]) * 4 * args.max_rounds
    return lo, hi


def print_curve(rows: Sequence[dict]) -> None:
    header = (
        f"{'model':12s}{'task':14s}{'order':10s}{'cond':8s}{'cp':>5s}{'lib':>6s}"
        f"{'metric':>9s}{'delta':>9s}{'rounds':>8s}{'tok/s':>9s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        delta = r.get("delta_vs_frozen")
        print(
            f"{r['model']:12s}{r['task']:14s}{str(r['order']):10s}{r['condition']:8s}"
            f"{r['checkpoint']:5d}{r['n_library']:6d}{r['mean_sample_metric']:9.2f}"
            f"{(f'{delta:+.2f}' if delta is not None else '-'):>9s}"
            f"{r['mean_rounds']:8.2f}{r['tokens_per_sample']:9.0f}"
        )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 8: does experience accumulated online improve later items?"
    )
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--models", default="all")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug only: use only the first N items of the stream")
    ap.add_argument("--eval-limit", type=int, default=200,
                    help="fixed evaluation subset of the TEST manifest (default 200)")
    ap.add_argument("--chunk-size", type=int, default=32,
                    help="stream items processed between two checkpoints")
    ap.add_argument("--orders", type=int, default=3,
                    help="stream orders (default 3, the plan's replication): order 0 "
                         "is the manifest order, orders 1..N-1 are distinct seeded "
                         "shuffles whose seed is a pure function of the order index")
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS,
                    help="comma-separated accumulation methods to evaluate at every "
                         "checkpoint (default all three): dynamic=Full-Dynamic, "
                         "frozen=Frozen, no_negative=NoNegativeExperience")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--stream-draft-cache", default="",
                    help="optional JSONL cache so stream drafts are generated once "
                         "and replayed across orders")
    ap.add_argument("--no-reuse-frozen", dest="reuse_frozen", action="store_false",
                    help="re-measure the frozen condition at every checkpoint/order "
                         "instead of once per model-task")
    ap.add_argument("--no-save-snapshots", dest="save_snapshots", action="store_false",
                    help="do not write the per-checkpoint library snapshots")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out-dir", default=None,
                    help="where traces and snapshots go (default: runs/accumulation)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the setup and print the plan; loads no model")
    ap.set_defaults(reuse_frozen=True, save_snapshots=True)
    args = ap.parse_args()

    if args.orders < 1:
        print("REFUSING: --orders must be >= 1")
        return 2
    if args.eval_limit < 2:
        print("REFUSING: --eval-limit must be >= 2 (a paired test needs pairs)")
        return 2
    if args.chunk_size < 1:
        print("REFUSING: --chunk-size must be >= 1")
        return 2
    try:
        conditions = parse_conditions(args.conditions)
    except ValueError as exc:
        print(f"REFUSING: {exc}")
        return 2
    accumulating = [c for c in ACCUMULATING_CONDITIONS if c in conditions]
    if not accumulating:
        print("REFUSING: --conditions selects no accumulating condition; "
              f"pick at least one of {list(ACCUMULATING_CONDITIONS)}")
        return 2

    # ---- GPU budget check: core/__init__.py is the single authority --------
    # Pure string validation (no CUDA involvement), so it also runs under
    # --dry-run; the budget is core.ALLOWED_GPUS / core.MAX_GPUS.
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

    models = resolve(args.models, MODELS)
    tasks = resolve(args.tasks, TASKS)
    out_dir = Path(args.out_dir) if args.out_dir else EXP_ROOT / "runs" / "accumulation"

    # ---- setup validation (never needs a GPU) ------------------------------
    plans: List[dict] = []
    problems: List[str] = []
    for model in models:
        for task in tasks:
            try:
                plans.append(plan_model_task(model, task, args))
            except FileNotFoundError as exc:
                problems.append(f"{model}/{task}: {exc}")
            except RuntimeError as exc:  # --orders > distinct stream orders
                problems.append(f"{model}/{task}: {exc}")
    if problems:
        print("SETUP PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1

    for task in sorted({p["task"] for p in plans}):
        iso = check_isolation(task)
        status = "ok" if iso["ok"] else f"VIOLATIONS {iso['violations']}"
        print(f"[isolation] {task:14s} test vs aux(splits): {status}")

    empty_libs = [f"{p['model']}/{p['task']}" for p in plans if p["n_initial"] == 0]
    print()
    print_plan(plans, args, conditions)
    print()
    print_methods(conditions)
    print()
    print_orders(plans, args)
    print()
    print_schedule(plans, args, conditions, out_dir, args.orders)
    print()
    if empty_libs:
        print(f"[WARN] empty initial library for: {', '.join(empty_libs)} "
              "(the frozen condition would be experience-free)")
    lo, hi = estimate_calls(plans, args, conditions)
    print(f"planned orders={args.orders} chunk={args.chunk_size} "
          f"eval-limit={args.eval_limit} max-rounds={args.max_rounds} batch={args.batch_size}")
    print(f"conditions={','.join(conditions)} (accumulating: {','.join(accumulating)})")
    print(f"estimated LLM calls: {lo:,} (all loops stop after 1 round) .. {hi:,} "
          "(all loops use the full round budget)")
    print(f"outputs: {out_dir}/<model>/<task>/order<k>/... and "
          f"{SCORES_DIR}/accumulation_curve.csv")

    if args.dry_run:
        print()
        print("[dry-run] setup validated; no model was loaded and no GPU was touched")
        return 0

    from baseline_core.config import MODEL_CONFIGS
    from baseline_core.llm import LLMClient

    import torch

    from core import MAX_MODEL_LEN, gpu_mem_util

    rows: List[dict] = []
    # per_order_metrics[condition][model/task/c####][order_label] -> per-sample finals
    per_order_metrics: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    by_model: Dict[str, List[dict]] = {}
    for p in plans:
        by_model.setdefault(p["model"], []).append(p)

    for model in sorted(by_model):
        print(f"[LOAD] model={model}", flush=True)
        llm = LLMClient(
            MODEL_CONFIGS[model], backend="vllm", gpu=gpus[0],
            gpu_memory_utilization=gpu_mem_util(model, args.gpu_memory_utilization),
            max_model_len=MAX_MODEL_LEN, enforce_eager=True,
        )
        for plan in by_model[model]:
            task = plan["task"]
            # the frozen condition does not depend on the stream order at all,
            # so one measurement per model-task is reused unless asked otherwise
            frozen_cached: Optional[List[SampleTrace]] = None
            for order in plan["orders"]:
                order_label, perm_hash = order["label"], order["digest"]
                stream = order["stream"]
                # ---- one library per accumulating condition -----------------
                # All of them start from the same bootstrap library and are fed
                # by the SAME stream run below; only the admission rule of
                # filter_for_condition() differs, so each non-Full-Dynamic
                # library is a subsequence of the Full-Dynamic one.  The stream's
                # own retrieval always uses the Full-Dynamic library, even when
                # --conditions does not select that method for evaluation: the
                # observed units must not depend on which methods are measured.
                stream_library: List[Experience] = list(plan["library"])
                libraries: Dict[str, List[Experience]] = {
                    cond: (stream_library if cond == "dynamic" else list(plan["library"]))
                    for cond in accumulating
                }
                library = stream_library
                extra_libraries = {
                    cond: lib for cond, lib in libraries.items() if cond != "dynamic"
                }
                retriever = ExperienceRetriever(library) if library else None
                print(f"[STREAM] {model}/{task} order={order_label} "
                      f"n={len(stream)} perm={perm_hash or 'identity'} "
                      f"lib0={len(library)}", flush=True)

                stream_cfg = RunConfig(
                    task=task, model=model, experience_mode="full", stop_mode="adaptive",
                    alpha=args.alpha, k=args.k, max_rounds=args.max_rounds,
                    seed=args.seed, snapshot_id=f"acc_stream_{order_label}",
                    draft_source="cached" if args.stream_draft_cache else "generate",
                    draft_cache=args.stream_draft_cache,
                )
                pipe = Pipeline(stream_cfg, retriever, library, llm, cache=None)
                made = 0
                cps = plan["checkpoints"]
                # ---- no resume, by design ----------------------------------
                # Unlike run_experiment.py's batched path this loop has no
                # resume/dedup: TraceWriter opens every file "w" and the library
                # is rebuilt from plan["library"] from scratch, so an interrupted
                # run starts over.  That is REQUIRED, not an omission -- the
                # accumulating library is order-dependent (each stream item is
                # refined against the library its predecessors built), so
                # replaying only the missing tail would hand the condition a
                # library it could never have had.  run_experiment.py's
                # `if online and prior:` guard discards a persisted prefix for
                # the same reason; nothing here may weaken that rule.
                with TraceWriter(out_dir / model / task / order_label / "stream",
                                 {"order": order_label, "checkpoint": "", "condition": "stream",
                                  "task": task, "model": model}) as stream_writer:
                    for ci, c in enumerate(cps):
                        # ---- the frozen baseline ---------------------------
                        frozen_fresh = False
                        frozen_now: Optional[List[SampleTrace]] = None
                        if "frozen" in conditions:
                            if args.reuse_frozen and frozen_cached is not None:
                                frozen_now = frozen_cached
                            else:
                                t0 = time.time()
                                frozen_now = run_eval(
                                    llm, model, task, plan["library"], plan["eval_refs"],
                                    args, out_dir, order_label="frozen", checkpoint=0,
                                    condition="frozen",
                                )
                                frozen_fresh = True
                                if args.reuse_frozen:
                                    frozen_cached = frozen_now
                                print(f"    [frozen] {model}/{task} measured "
                                      f"({time.time() - t0:.0f}s)", flush=True)

                        # ---- snapshot + evaluate every accumulation method -
                        for cond in accumulating:
                            snapshot = list(libraries[cond])
                            if args.save_snapshots:
                                save_experiences(
                                    snapshot,
                                    condition_dir(
                                        out_dir / model / task / order_label, cond
                                    ) / f"library_c{c:04d}.jsonl",
                                )
                            t0 = time.time()
                            traces = run_eval(
                                llm, model, task, snapshot, plan["eval_refs"], args,
                                out_dir, order_label=order_label, checkpoint=c,
                                condition=cond,
                            )
                            seconds = time.time() - t0
                            key = f"{model}/{task}/c{c:04d}"
                            per_order_metrics.setdefault(cond, {}).setdefault(
                                key, {}
                            )[order_label] = [
                                t.final_metric_offline or 0.0 for t in traces
                            ]
                            rows.append(summarize_eval(
                                traces, plan["eval_refs"], task, model=model,
                                order_label=order_label, checkpoint=c, condition=cond,
                                n_library=len(snapshot), n_initial=plan["n_initial"],
                                duration=seconds, frozen=frozen_now,
                                frozen_reused=not frozen_fresh, bootstrap=args.bootstrap,
                            ))
                            print(f"    [{cond} {order_label} c{c:04d}] "
                                  f"library={len(snapshot)} "
                                  f"({len(snapshot) - plan['n_initial']:+d} accumulated, "
                                  f"{seconds:.0f}s)", flush=True)

                        # ---- the frozen row (once per checkpoint) ----------
                        if frozen_now is not None:
                            rows.append(summarize_eval(
                                frozen_now, plan["eval_refs"], task, model=model,
                                order_label=order_label, checkpoint=c, condition="frozen",
                                n_library=plan["n_initial"], n_initial=plan["n_initial"],
                                duration=0.0, frozen=None,
                                frozen_reused=not frozen_fresh, bootstrap=args.bootstrap,
                            ))

                        # ---- then process the next chunk of the stream -----
                        nxt = cps[ci + 1] if ci + 1 < len(cps) else c
                        for i in range(c, nxt):
                            tr, n_new = process_stream_item(
                                llm, pipe, stream[i], i, model, library, retriever,
                                extra_libraries=extra_libraries,
                            )
                            made += n_new
                            stream_writer.sink([tr], i + 1, len(stream))
                        if nxt > c:
                            kept = ", ".join(
                                f"{cond}={len(lib)}" for cond, lib in libraries.items()
                            )
                            print(f"    [stream {order_label}] {nxt}/{len(stream)} "
                                  f"items, stream_library={len(stream_library)} "
                                  f"(+{made} observed; {kept})", flush=True)
                del pipe
        del llm
        torch.cuda.empty_cache()

    _write_csv(rows, SCORES_DIR / "accumulation_curve.csv")
    print_curve(rows)

    # ---- order variance ----------------------------------------------------
    # One file per accumulating condition.  Full-Dynamic keeps the historical
    # name, path and key format (model/task/c####), so the artefact written by a
    # pre-existing run stays exactly as it was; NoNegativeExperience writes its
    # own file instead of adding keys to that one.
    ov_by_cond: Dict[str, Dict[str, dict]] = {}
    for cond in accumulating:
        ov = {
            key: order_variance(per_order)
            for key, per_order in sorted(per_order_metrics.get(cond, {}).items())
            if len(per_order) > 1
        }
        if ov:
            ov_by_cond[cond] = ov
    if ov_by_cond:
        for cond, ov in ov_by_cond.items():
            path = order_variance_path(cond)
            path.write_text(
                json.dumps(ov, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print()
            print(f"Order variance of the {PLAN_NAMES[cond]} condition "
                  "(mean over eval samples):")
            print(f"{'run/order/checkpoint':44s}{'mean':>9s}{'std':>9s}{'n':>4s}")
            for key, stats in ov.items():
                print(f"{key:44s}{stats['mean']:9.3f}{stats['std']:9.3f}{stats['n_orders']:4d}")
            print(f"-> {path}")
    else:
        print()
        print("order variance needs --orders >= 2; skipped")
    print(f"-> {SCORES_DIR / 'accumulation_curve.csv'} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
