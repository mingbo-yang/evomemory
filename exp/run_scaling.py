#!/usr/bin/env python
"""Phase 6 driver: quality-compute scaling curves (round budget K and BoN-J N).

What Phase 6 measures
---------------------
Every curve is x = **total tokens** (``cost.total_tokens``, already carried by
each trace) and y = the shared primary metric.  Two sweeps:

* **round budget** ``K in {1, 2, 3, 5}`` for the four configurations the plan
  names -- ``full_static`` (Full), ``fixed_rounds`` (FixedRounds),
  ``no_experience`` (NoExperience), ``sr_j_fixed`` (SR-J-Fixed);
* **best-of-N with the same judge** (``bon_judge``, "BoN-J")
  ``N in {1, 2, 4, 8}``: N candidate revisions of the same state per round,
  winner picked by the round-robin Copeland rule over the *same* pairwise judge
  (see ``run_experiment.BONJ_SELECTION_RULE``), then the standard winner-vs-
  current acceptance test.

Reuse before recompute
----------------------
The existing Phase 4/5 runs already are the K=3 point of all four
configurations.  For a given arm, ``max_rounds`` only truncates the refinement
loop, and the loop is a pure function of the round index (same seeds, same
prompts, same state sequence -- the seeds depend on ``(sample_id, t)`` and
nothing else), so:

* ``K <= 3`` is **derived** from the completed K=3 anchor by prefix truncation
  (``K == 3`` is a byte copy of the anchor);
* ``K == 5`` must be computed;
* BoN-J ``N == 1`` **is** the single-candidate path (the pipeline delegates to
  the unmodified ``BatchedPipeline`` round), so the point is a byte copy of the
  completed ``full_static`` anchor and is not recomputed;
* BoN-J ``N > 1`` must be computed.

Completeness is detected the same way ``aggregate.py`` does it: a run file is
usable only when **every** line parses and it holds at least
``TEST_SAMPLES[task]`` records (a supervisor retry can leave a prefix, and
pooling a prefix with a complete run silently corrupts corpus metrics).

Output layout mirrors ``runs/main``::

    runs/scaling/<arm>/seed<seed>/<task>/<model>/<point>/<arm>.jsonl
    runs/scaling/<arm>/seed<seed>/<task>/<model>/<point>/<arm>.csv
    runs/scaling/<arm>/seed<seed>/<task>/<model>/<point>/<arm>.summary.json

with ``<point>`` in ``{k1, k2, k3, k5}`` for the K sweep and
``{n1, n2, n4, n8}`` for the N sweep (``--limit`` debug runs get an extra
``limit<N>/`` component so they can never be mistaken for a real point).
Provenance for every point is written to ``runs/scaling/PLAN.json``.

Usage
-----
    # read-only plan: what is reused, what is derived, what must be computed
    python run_scaling.py --dry-run

    # materialise the derived/copied points (CPU only) and run the rest on GPU 0
    python run_scaling.py --gpu 0 --models glm4-9b,qwen3-8b --tasks all

    # a single sweep, one task, for a quick look
    python run_scaling.py --gpu 1 --only n --tasks wmt19_en_zh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    EXP_ROOT,
    MAX_MODEL_LEN,
    MODELS,
    TASKS,
    TEST_SAMPLES,
    ensure_gpu_whitelist,
    gpu_mem_util,
    resolve_model_config,
)
from core.manifest import manifest_path, read_manifest  # noqa: E402
from core.pipeline import RunConfig  # noqa: E402

import run_experiment as rex  # noqa: E402

SCALING_DIR = EXP_ROOT / "runs" / "scaling"
#: Where a completed Phase 4/5 run of an arm may live (exact layouts only --
#: never an rglob, so a dev/alpha-sweep tag can never be mistaken for the main
#: run).
ANCHOR_BASES = (EXP_ROOT / "runs" / "main", EXP_ROOT / "runs" / "ablation")

#: Every existing main/ablation run used ``--max-rounds 3``.
ANCHOR_ROUNDS = 3

#: Phase 6 configurations: plan name -> arm (the K sweep).
CONFIGS: Tuple[Tuple[str, str], ...] = (
    ("Full", "full_static"),
    ("FixedRounds", "fixed_rounds"),
    ("NoExperience", "no_experience"),
    ("SR-J-Fixed", "sr_j_fixed"),
)
K_POINTS: Tuple[int, ...] = (1, 2, 3, 5)
N_POINTS: Tuple[int, ...] = (1, 2, 4, 8)
BONJ_ARM = "bon_judge"

#: The plan's 2x2 models (frozen.json phase5_ablation_substitution): the quality-
#: compute curve is reported for these two.
DEFAULT_MODELS = "glm4-9b,qwen3-8b"

#: Byte-identical to the CSV schema run_experiment.py / aggregate.py use.
FIELDS = ["task", "model", "arm", "seed", "sample_id", "n_rounds",
          "initial_metric_offline", "final_metric_offline",
          "total_tokens", "latency_s", "n_calls"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def expected_n(task: str) -> int:
    """Completeness denominator -- exactly ``aggregate.py``'s ``expected_n``."""
    return int(TEST_SAMPLES.get(task, 1000))


def rel(path: Path) -> str:
    """Path relative to the experiment root when possible, else absolute."""
    try:
        return str(Path(path).relative_to(EXP_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_records(path: Path) -> List[dict]:
    """Parse a run JSONL in full.

    Raises ``ValueError`` on any unparsable non-empty line: ``aggregate.py``
    skips such a file entirely (a killed attempt can leave a truncated tail), so
    it cannot be treated as complete here either.
    """
    out: List[dict] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {i} is not valid JSON ({exc})") from exc
    return out


def completeness(path: Path, need: int) -> Tuple[bool, int, str]:
    """``(complete, n_records, why)`` using aggregate.py's rule."""
    if not path.is_file():
        return False, 0, "no file"
    try:
        recs = load_records(path)
    except (OSError, ValueError) as exc:
        return False, 0, f"unreadable ({exc})"
    if len(recs) < need:
        return False, len(recs), f"{len(recs)}/{need} records"
    return True, len(recs), f"{len(recs)}/{need} records"


def point_dir(arm: str, seed: int, task: str, model: str, label: str,
              limit: Optional[int], out_dir: Path) -> Path:
    base = out_dir / arm / f"seed{seed}" / task / model
    if limit is not None:
        base = base / f"limit{limit}"
    return base / label


def point_tag(label: str, limit: Optional[int]) -> str:
    """The ``--tag`` value that makes run_experiment write into ``point_dir``."""
    return label if limit is None else f"limit{limit}/{label}"


def expected_config(arm: str, task: str, model: str, seed: int, alpha: float,
                    k: int, max_rounds: int) -> RunConfig:
    """The exact RunConfig of the corresponding main run (frozen alpha/k/etc.)."""
    spec = rex.ARM_REGISTRY[arm]
    return RunConfig(
        task=task, model=model,
        experience_mode=spec["experience_mode"], stop_mode=spec["stop_mode"],
        alpha=alpha, k=k, max_rounds=max_rounds, seed=seed,
        snapshot_id="online" if spec["online"] else "initial",
        draft_source="stored", draft_cache="",
    )


def anchor_candidates(arm: str, seed: int, task: str, model: str,
                      out_dir: Path) -> List[Path]:
    """Exact expected locations of a completed K=3 anchor, in preference order."""
    out = [out_dir / arm / f"seed{seed}" / task / model / f"k{ANCHOR_ROUNDS}" / f"{arm}.jsonl"]
    for base in ANCHOR_BASES:
        out.append(base / arm / f"seed{seed}" / task / model / f"{arm}.jsonl")
    return out


@dataclass
class Anchor:
    path: Path
    records: List[dict]


def find_anchor(arm: str, seed: int, task: str, model: str, need: int,
                alpha: float, k: int, out_dir: Path) -> Tuple[Optional[Anchor], str]:
    """First *complete* K=3 run of this arm with exactly the main config.

    The ``config_hash`` equality check is what makes the reuse safe: a run made
    with another alpha, budget or stop mode can never be silently truncated into
    a Phase 6 point.
    """
    expected = expected_config(arm, task, model, seed, alpha, k, ANCHOR_ROUNDS)
    problems: List[Tuple[Path, str]] = []
    for path in anchor_candidates(arm, seed, task, model, out_dir):
        ok, n, why = completeness(path, need)
        if ok:
            try:
                recs = load_records(path)
            except ValueError as exc:
                problems.append((path, str(exc)))
                continue
            hashes = {r.get("config_hash") for r in recs}
            if hashes != {expected.config_hash}:
                problems.append((path, "exists but its config_hash is not the main "
                                       f"{arm} config"))
                continue
            return Anchor(path=path, records=recs), ""
        problems.append((path, why))
    # Prefer a path that exists but is unusable over a plain "no file": that is
    # the one an operator is actually waiting for (a run still in flight).
    for path, why in problems:
        if why != "no file":
            return None, f"{rel(path)}: {why}"
    first = anchor_candidates(arm, seed, task, model, out_dir)[0]
    return None, f"no run at {rel(first)}"


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
@dataclass
class Point:
    kind: str                 # "k" | "n"
    value: int
    label: str                # "k1" ... "n8"
    arm: str
    config_name: str          # paper name (Full / BoN-J / ...)
    model: str
    task: str
    seed: int
    n_samples: int
    max_rounds: int
    n_candidates: int
    path: Path
    tag: str
    status: str = "compute"   # complete | copy | derive | compute | blocked
    planned: str = "compute"  # status at plan time (stays after materialisation)
    source: Optional[Path] = None
    note: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["path"] = rel(self.path)
        d["source"] = rel(self.source) if self.source else None
        return d


def estimate_calls(p: Point) -> int:
    """Worst-case engine calls (the controller never stops before the budget).

    K points: controller + refine + 2 judge calls per round.
    BoN-J N: controller + N refine + 2*C(N,2) selection judge + 2 acceptance
    judge calls per round (identical candidate pairs cost nothing, so this is an
    upper bound).
    """
    if p.kind == "k":
        per_round = 4
    else:
        per_round = 1 + p.n_candidates + p.n_candidates * (p.n_candidates - 1) + 2
    return int(p.n_samples) * int(p.max_rounds) * per_round


def build_plan(models: Sequence[str], tasks: Sequence[str], seed: int,
               limit: Optional[int], alpha: float, k: int, only: str,
               out_dir: Path) -> List[Point]:
    plan: List[Point] = []
    for model in models:
        for task in tasks:
            need = min(expected_n(task), limit) if limit else expected_n(task)
            anchors: Dict[str, Tuple[Optional[Anchor], str]] = {}

            def anchor_for(arm: str):
                if arm not in anchors:
                    anchors[arm] = find_anchor(arm, seed, task, model, need, alpha, k, out_dir)
                return anchors[arm]

            if only in ("all", "k"):
                for config_name, arm in CONFIGS:
                    anchor, why = anchor_for(arm)
                    for K in K_POINTS:
                        pdir = point_dir(arm, seed, task, model, f"k{K}", limit, out_dir)
                        p = Point(
                            kind="k", value=K, label=f"k{K}", arm=arm,
                            config_name=config_name, model=model, task=task, seed=seed,
                            n_samples=need, max_rounds=K, n_candidates=1,
                            path=pdir / f"{arm}.jsonl", tag=point_tag(f"k{K}", limit),
                        )
                        ok, n, msg = completeness(p.path, need)
                        if ok:
                            p.status, p.note = "complete", msg
                        elif K <= ANCHOR_ROUNDS:
                            if anchor is None:
                                p.status = "blocked"
                                p.note = (f"no complete max_rounds={ANCHOR_ROUNDS} "
                                          f"{arm} anchor: {why}")
                            elif K == ANCHOR_ROUNDS:
                                p.status, p.source = "copy", anchor.path
                                p.note = "byte copy of the completed K=3 anchor"
                            else:
                                p.status, p.source = "derive", anchor.path
                                p.note = f"prefix truncation of {anchor.path.name} to K={K}"
                        else:
                            p.status = "compute"
                            p.note = f"new point: --max-rounds {K}"
                        p.planned = p.status
                        plan.append(p)

            if only in ("all", "n"):
                for N in N_POINTS:
                    pdir = point_dir(BONJ_ARM, seed, task, model, f"n{N}", limit, out_dir)
                    p = Point(
                        kind="n", value=N, label=f"n{N}", arm=BONJ_ARM,
                        config_name="BoN-J", model=model, task=task, seed=seed,
                        n_samples=need, max_rounds=ANCHOR_ROUNDS, n_candidates=N,
                        path=pdir / f"{BONJ_ARM}.jsonl", tag=point_tag(f"n{N}", limit),
                    )
                    ok, n, msg = completeness(p.path, need)
                    if ok:
                        p.status, p.note = "complete", msg
                    elif N == 1:
                        anchor, why = anchor_for("full_static")
                        if anchor is None:
                            p.status = "blocked"
                            p.note = f"no complete full_static anchor: {why}"
                        else:
                            p.status, p.source = "copy", anchor.path
                            p.note = ("N=1 delegates to the single-candidate path, so this "
                                      "is the full_static run itself (byte copy)")
                    else:
                        p.status = "compute"
                        p.note = (f"new point: N={N} candidates, "
                                  f"{N * (N - 1)} selection judge calls/round")
                    p.planned = p.status
                    plan.append(p)
    return plan


# --------------------------------------------------------------------------- #
# record transforms (CPU only)
# --------------------------------------------------------------------------- #
def derive_prefix(rec: dict, k: int) -> dict:
    """The first ``k`` rounds of a completed run, as its own K-budget record.

    Valid because the refinement loop is a pure function of the round index:
    seeds are ``call_seed(sample_id, t, ...)``, prompts are built from the state
    after round ``t-1``, and that state is identical in the source run and in a
    run whose budget is ``k``.  The truncated point therefore carries the source
    run's own prefix -- computed once, re-scored as the state after ``k`` rounds.
    """
    rounds = list(rec.get("rounds") or [])
    keep = rounds[:k]
    out = dict(rec)
    out["rounds"] = keep

    current = rec.get("initial_draft")
    final_metric = rec.get("initial_metric_offline")
    for r in keep:
        if r.get("controller_action") == "REFINE" and r.get("candidate") and r.get("accepted"):
            current = r["candidate"]
            if r.get("metric_offline") is not None:
                final_metric = r["metric_offline"]
    out["final_output"] = current
    out["final_metric_offline"] = final_metric

    if keep:
        # per-round cost is cumulative, so the last kept round is the K-budget spend
        out["cost"] = dict(keep[-1].get("cost") or {})

    cfg = dict(rec.get("config") or {})
    cfg["max_rounds"] = int(k)
    out["config"] = cfg
    # recomputed from the same field set RunConfig hashes -> byte-identical to a
    # real --max-rounds k run's hash, so a derived point can be resumed and can
    # never be mistaken for a different budget.
    out["config_hash"] = RunConfig(**cfg).config_hash
    return out


def write_point(arm: str, task: str, model: str, seed: int, recs: List[dict],
                summary_extra: dict, path: Path) -> dict:
    """Write JSONL+CSV+summary in exactly the layout/blessed writers of main."""
    refs = read_manifest(manifest_path(task, "test"))
    path.parent.mkdir(parents=True, exist_ok=True)
    cpath = path.with_suffix(".csv")
    wrapped = [rex._PriorRecord(r) for r in recs]
    rex._persist_merged(path, cpath, wrapped, refs, FIELDS, task, model, arm, seed)
    summary = rex.summarize(wrapped, arm, task, model, seed, 0.0)
    summary.update(summary_extra)
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def materialize(p: Point) -> None:
    """Produce a copy/derived point from its completed source run (no GPU)."""
    assert p.source is not None
    src_sha = sha256_file(p.source)
    # A completed run is persisted in manifest order, so its first n_samples
    # records ARE what a --limit n_samples run over the same prefix produces
    # (global manifest indices -- hence seeds and stored-draft lookups -- are
    # preserved by the resume machinery).
    src_recs = load_records(p.source)[:p.n_samples]
    src_rel = rel(p.source)

    if p.status == "copy":
        recs = src_recs
        extra = {
            "point": p.label, "reused": True, "derived": False,
            "reused_from": src_rel, "source_sha256": src_sha,
            "max_rounds": p.max_rounds, "n_candidates": p.n_candidates,
            "n_candidates_note": (
                "N=1 delegates to the unmodified single-candidate round, so this "
                "point is the full_static run itself (byte copy)"
            ) if p.kind == "n" else "byte copy of the completed K=3 anchor",
        }
    else:
        recs = [derive_prefix(r, p.value) for r in src_recs]
        extra = {
            "point": p.label, "reused": False, "derived": True,
            "derived_from": src_rel, "source_sha256": src_sha,
            "derivation": "prefix truncation of a max_rounds=3 run",
            "max_rounds": p.value, "n_candidates": 1,
        }

    summary = write_point(p.arm, p.task, p.model, p.seed, recs, extra, p.path)

    same = sha256_file(p.path) == src_sha
    if p.status == "copy" and not same:
        print(f"    [WARN] {p.label}: copy is not byte-identical to {src_rel} "
              f"(content still reused verbatim)")
    p.status = "complete"
    p.note = (f"{len(recs)} records" + (" (byte-identical copy)" if same else ""))
    tok = summary.get("cost", {}).get("total_tokens")
    print(f"    [{p.status}] {p.label:>3s} {p.config_name:<12s} {p.model:<10s} {p.task:<13s}"
          f" n={summary.get('n')} mean_rounds={summary.get('mean_rounds', 0):.2f}"
          f" tokens/sample={tok if tok is None else round(tok, 1)}  <- {p.note}", flush=True)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def print_plan(plan: List[Point]) -> None:
    order = {"blocked": 0, "compute": 1, "derive": 2, "copy": 3, "complete": 4}
    print(f"{'point':>5s}  {'config':<12s} {'arm':<14s} {'model':<10s} {'task':<13s}"
          f" {'samples':>7s}  {'status':<8s} note")
    for p in sorted(plan, key=lambda q: (q.model, q.task, q.kind, q.value)):
        print(f"{p.label:>5s}  {p.config_name:<12s} {p.arm:<14s} {p.model:<10s} {p.task:<13s}"
              f" {p.n_samples:>7d}  {p.status.upper():<8s} {p.note}")

    counts: Dict[str, int] = {}
    for p in plan:
        counts[p.status] = counts.get(p.status, 0) + 1
    print()
    print("Phase 6 plan summary")
    print(f"  points          : {len(plan)} " +
          " ".join(f"{k}={counts.get(k, 0)}" for k in
                   ("complete", "copy", "derive", "compute", "blocked")))
    todo = [p for p in plan if p.status == "compute"]
    samples = sum(p.n_samples for p in todo)
    calls = sum(estimate_calls(p) for p in todo)
    print(f"  samples to compute : {samples}  ({len(todo)} points)")
    print(f"  worst-case engine calls to compute : {calls}")
    for p in todo:
        per_round = 4 if p.kind == "k" else 1 + p.n_candidates + p.n_candidates * (p.n_candidates - 1) + 2
        print(f"    {p.label:>3s} {p.arm:<14s} {p.model:<10s} {p.task:<13s}"
              f" {p.n_samples:>5d} samples x {p.max_rounds} rounds x {per_round} calls"
              f" = {estimate_calls(p)}")
    if counts.get("blocked"):
        print()
        print("  WARNING: blocked points -- their Phase 4/5 anchor run is missing or "
              "incomplete; re-run this driver once the anchor finishes:")
        for p in plan:
            if p.status == "blocked":
                print(f"    {p.label:>3s} {p.arm:<14s} {p.model:<10s} {p.task:<13s} {p.note}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6 quality-compute scaling driver")
    ap.add_argument("--gpu", default="0", help="card id(s); same whitelist as run_experiment.py")
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help=f"comma list, or 'all'; default {DEFAULT_MODELS} (the plan's 2x2)")
    ap.add_argument("--tasks", default="all", help="comma list, or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug only: first N items; writes under limit<N>/ so it can "
                         "never be mistaken for a real point")
    ap.add_argument("--alpha", type=float, default=0.5, help="frozen at 0.5 (configs/frozen.json)")
    ap.add_argument("--k", type=int, default=4, help="retrieval k, frozen at 4")
    ap.add_argument("--only", default="all", choices=["all", "k", "n"],
                    help="restrict to the round-budget sweep, the BoN-J sweep, or both")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the full plan and exit; touches no GPU and writes nothing")
    ap.add_argument("--reuse-only", action="store_true",
                    help="materialise the reused/derived points (CPU only) and stop; "
                         "never loads a model")
    ap.add_argument("--out-dir", default=str(SCALING_DIR))
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    args = ap.parse_args()

    models = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = list(TASKS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    gpus = [g.strip() for g in args.gpu.split(",") if g.strip()]
    from core import ALLOWED_GPUS, MAX_GPUS

    illegal = [g for g in gpus if g not in ALLOWED_GPUS]
    if illegal or not gpus or len(gpus) > MAX_GPUS:
        print(f"REFUSING: {gpus} violates the GPU budget (any {MAX_GPUS} of {ALLOWED_GPUS})")
        return 2

    print(f"Phase 6 scaling: models={models} tasks={tasks} seed={args.seed} "
          f"alpha={args.alpha} k={args.k} batch_size={args.batch_size} "
          f"limit={args.limit} only={args.only}")
    print(f"scaling out_dir : {out_dir}")
    print(f"anchors read from: {[rel(b) for b in ANCHOR_BASES]}")
    print()
    plan = build_plan(models, tasks, args.seed, args.limit, args.alpha, args.k,
                      args.only, out_dir)
    print_plan(plan)

    if args.dry_run:
        print("dry run: nothing written, no GPU touched")
        return 0

    # ---- CPU stage: reuse / derive every point that does not need an engine --
    todo = [p for p in plan if p.status in ("copy", "derive")]
    if todo:
        print(f"materialising {len(todo)} reused/derived point(s) (CPU only)")
        for p in todo:
            materialize(p)
        print()

    # ---- GPU stage: only the genuinely new points ---------------------------
    pending = [p for p in plan if p.status == "compute"]
    if args.reuse_only:
        print(f"--reuse-only: {len(pending)} point(s) left uncomputed "
              f"({sum(p.n_samples for p in pending)} samples)")
        _write_plan(args, plan, out_dir)
        return 0
    if not pending:
        print("nothing to compute: every Phase 6 point is already on disk")
        if not args.dry_run:
            _write_plan(args, plan, out_dir)
        return 0

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
    ensure_gpu_whitelist()
    from baseline_core.llm import LLMClient

    import torch

    for model in models:
        todo_model = [p for p in pending if p.model == model]
        if not todo_model:
            continue
        print(f"[LOAD] model={model} ({len(todo_model)} point(s) to compute)", flush=True)
        llm = LLMClient(
            resolve_model_config(model),
            backend="vllm",
            gpu=gpus[0],
            gpu_memory_utilization=gpu_mem_util(model, args.gpu_memory_utilization),
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
        )
        for p in todo_model:
            print(f"[RUN] {p.label} {p.config_name} arm={p.arm} model={model} task={p.task} "
                  f"max_rounds={p.max_rounds} n_candidates={p.n_candidates}", flush=True)
            t0 = time.time()
            s = rex.run_model_task(
                arm=p.arm, model=model, task=p.task, seed=args.seed, gpu=gpus[0],
                split="test", tag=p.tag, limit=args.limit, max_rounds=p.max_rounds,
                alpha=args.alpha, k=args.k, draft_source="stored", draft_cache="",
                batch_size=args.batch_size, llm=llm, out_dir=out_dir,
                n_candidates=p.n_candidates,
            )
            ok, n, msg = completeness(p.path, p.n_samples)
            p.status = "complete" if ok else "blocked"
            p.note = msg if ok else f"run finished but file is {msg}"
            print(f"    final={s.get('mean_final_metric', float('nan')):.2f} "
                  f"init={s.get('mean_initial_metric', float('nan')):.2f} "
                  f"rounds={s.get('mean_rounds', 0):.2f} "
                  f"accept={s.get('accept_rate', 0):.2%} "
                  f"tokens/sample={s.get('cost', {}).get('total_tokens', float('nan')):.0f} "
                  f"({time.time() - t0:.0f}s, {msg})", flush=True)
        del llm
        torch.cuda.empty_cache()

    _write_plan(args, plan, out_dir)

    print()
    print("Phase 6 status")
    for p in sorted(plan, key=lambda q: (q.model, q.task, q.kind, q.value)):
        print(f"  {p.status.upper():<8s} {p.label:>3s} {p.arm:<14s} {p.model:<10s} "
              f"{p.task:<13s} {p.note}")
    bad = [p for p in plan if p.status == "blocked"]
    if bad:
        print()
        print(f"WARNING: {len(bad)} point(s) are blocked; re-run this driver to pick "
              f"them up once their source run completes")
    return 0


def _write_plan(args, plan: List[Point], out_dir: Path) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": {k: v for k, v in vars(args).items()},
        "out_dir": str(out_dir),
        "anchor_bases": [rel(b) for b in ANCHOR_BASES],
        "anchor_rounds": ANCHOR_ROUNDS,
        "selection_rule": rex.BONJ_SELECTION_RULE,
        "points": [p.as_dict() for p in plan],
        "counts": {
            s: sum(1 for p in plan if p.status == s)
            for s in ("complete", "copy", "derive", "compute", "blocked")
        },
        "planned_counts": {
            s: sum(1 for p in plan if p.planned == s)
            for s in ("complete", "copy", "derive", "compute", "blocked")
        },
        "samples_to_compute": sum(p.n_samples for p in plan if p.status == "compute"),
    }
    path = out_dir / "PLAN.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"plan written -> {rel(path)}")


if __name__ == "__main__":
    raise SystemExit(main())
