#!/usr/bin/env python
"""Re-judge stored (current, candidate) pairs with a different judge, offline.

Why
---
The acceptance gate is the sole cause of the method's loss, and the length guard
that fixes the *harm* captures only ~3% of the available oracle space
(reports/judge_bottleneck_diagnosis.md, progress.md Round 13).  The reason is
that the judge is **the same model that wrote the revision**: it prefers its own
longer rewrites.  A length heuristic can only veto; it cannot tell a genuinely
better revision from a fluent expansion.

This tool attacks the cause directly.  Every REFINE round in a stored trace
already contains the exact (current, candidate) pair and the true offline delta,
so alternative judges can be evaluated **without re-running a single pipeline**:
generate new verdicts for the stored pairs, then replay the acceptance rule
offline and read off the corpus metric.

The pipeline's own prompt, A/B randomisation and per-call seeds are reused
(`core.judge._TEMPLATE`, `randomised_ab`, `call_seed`), so a re-judged round is
directly comparable with the stored one -- only the judge differs.

GPU: yes, but only judge calls (short prompts).  It never runs the refiner.

Usage
-----
  python rejudge_pairs.py --judge-model qwen3-8b --variant base \
      --runs runs/dev_gate_fix --out scores/rejudge_dev_qwen3-8b_base.jsonl
  python rejudge_pairs.py --judge-model qwen3-8b --variant verify \
      --runs runs/main --limit-per-cell 200 --out scores/rejudge_test.jsonl

Then evaluate with `--evaluate`, which replays acceptance and prints the corpus
metric under each rule.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core  # noqa: E402
from core.determinism import call_seed, randomised_ab  # noqa: E402
from core.judge import (JUDGE_MAX_TOKENS, _SYSTEM, _TEMPLATE, _combine,  # noqa: E402
                        _extract_json, _map_verdict, accept_revision)
from core.manifest import manifest_path, read_manifest  # noqa: E402
from core.scoring import Scorer  # noqa: E402

PRIMARY = {"wmt19_en_zh": "bleu", "wmt19_zh_en": "bleu",
           "coedit_gec": "gleu", "gigaword": "rouge1"}

#: Judge variants.  ``base`` is the shipped prompt; ``verify`` asks the two
#: objective questions that a verbosity preference cannot answer "yes" to
#: cheaply, instead of the open-ended "which is better".
VARIANTS = {
    "base": _SYSTEM,
    "verify": (
        "You are a strict verifier of {task_name} outputs. You never see a "
        "reference answer. You are given a task input and two answers, A and B. "
        "Answer ONLY the two questions asked. Being longer, more detailed or "
        "more fluent is NOT evidence of being better; added content that the "
        "input does not support is a defect. Reply with JSON only."
    ),
}

VERIFY_TEMPLATE = """Task input:
{task_input}

Answer A:
{slot_a}

Answer B:
{slot_b}

Q1: Does either answer state facts, numbers, entities or details that the task input does NOT support? If so, which one?
Q2: Does either answer OMIT information that the task input requires? If so, which one?

Return JSON with exactly these keys:
{{"unsupported": "A" | "B" | "both" | "neither", "omits": "A" | "B" | "both" | "neither"}}

A revision is better ONLY if it introduces no unsupported content and omits nothing
that the other answer contains.
"""


def load(path: Path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def collect_pairs(runs_root: Path, split: str, limit_per_cell: int):
    """Every REFINE round with its (current, candidate) pair and true delta."""
    items = []
    pat = str(runs_root / "*" / "seed*" / "*" / "*" / ("" if split == "test" else split) / "*.jsonl")
    for p in sorted(glob.glob(pat)):
        path = Path(p)
        try:
            recs = load(path)
        except Exception:
            continue
        if not recs:
            continue
        task, model = recs[0]["task"], recs[0]["model"]
        n = 0
        for r in recs:
            cur = r["initial_draft"]
            for rd in r["rounds"]:
                if rd.get("controller_action") != "REFINE":
                    continue
                items.append({
                    "file": str(path), "task": task, "model": model,
                    "sample_id": r["sample_id"], "round_index": rd["round_index"],
                    "task_input": None,  # filled from the manifest below
                    "current": cur, "candidate": rd.get("candidate", ""),
                    "delta_offline": rd.get("delta_offline"),
                    "stored_verdict": rd.get("judge_verdict"),
                    "stored_order_consistent": rd.get("judge_order_consistent"),
                })
                if rd.get("accepted"):
                    cur = rd["candidate"]
            n += 1
            if limit_per_cell and n >= limit_per_cell:
                break
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--variant", default="base", choices=sorted(VARIANTS))
    ap.add_argument("--runs", default="runs/dev_gate_fix")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--limit-per-cell", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evaluate", action="store_true",
                    help="replay acceptance with the new verdicts and report corpus metrics")
    args = ap.parse_args()

    if args.evaluate:
        return evaluate(Path(args.out), args.runs, args.split)

    items = collect_pairs(Path(args.runs), args.split, args.limit_per_cell)
    if not items:
        print(f"no REFINE rounds under {args.runs} (split={args.split})")
        return 2
    print(f"collected {len(items)} stored REFINE rounds from {args.runs}")

    need_input = {(it["task"], it["sample_id"]) for it in items}
    inputs = {}
    for task in {t for t, _ in need_input}:
        for r in read_manifest(manifest_path(task, args.split)):
            inputs[(task, r.sample_id)] = r.source

    # ---- build the judge calls, exactly as the pipeline would -------------
    from baseline_core.llm import LLMClient
    from core import gpu_mem_util, resolve_model_config
    from core.batch_llm import BatchedLLM, GenRequest

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    llm = BatchedLLM(LLMClient(resolve_model_config(args.judge_model), backend="vllm",
                               gpu=args.gpu,
                               gpu_memory_utilization=gpu_mem_util(args.judge_model),
                               max_model_len=core.MAX_MODEL_LEN, enforce_eager=True))

    template = VERIFY_TEMPLATE if args.variant == "verify" else _TEMPLATE
    sysmsg = VARIANTS[args.variant]
    reqs, meta = [], []
    for it in items:
        src = inputs.get((it["task"], it["sample_id"]), "")
        it["task_input"] = src
        # same A/B randomisation and seeds as the pipeline used
        sa, sb, cia = randomised_ab(it["model"], it["task"], it["sample_id"],
                                    int(it["round_index"]), it["current"], it["candidate"])
        for call_idx, (x, y, c) in enumerate(((sa, sb, cia), (sb, sa, not cia))):
            reqs.append(GenRequest(
                prompt=template.format(task_input=src, slot_a=x, slot_b=y),
                system_prompt=sysmsg.format(task_name=it["task"]),
                seed=call_seed(it["sample_id"], int(it["round_index"]) * 10 + call_idx, "judge"),
                call_type="judge", max_tokens=JUDGE_MAX_TOKENS, temperature=0.0))
            meta.append((it, c, call_idx))

    print(f"judging {len(reqs)} calls with {args.judge_model} (variant={args.variant}) ...")
    gens = llm.generate_batch(reqs)

    per_round = defaultdict(dict)
    for (it, c, call_idx), g in zip(meta, gens):
        obj = _extract_json(g.text) or {}
        key = (it["file"], it["sample_id"], it["round_index"])
        if args.variant == "verify":
            # map the two objective answers onto a preference
            uns, om = str(obj.get("unsupported", "")).strip().upper(), str(obj.get("omits", "")).strip().upper()
            bad_self = (uns == ("A" if c else "B")) or (om == ("A" if c else "B"))
            v = "current" if bad_self else ("candidate" if (uns in ("A", "B") or om in ("A", "B")) else "tie")
        else:
            v = _map_verdict(str(obj.get("verdict", "uncertain")), c)
        per_round[key][call_idx] = v

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_better = 0
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            key = (it["file"], it["sample_id"], it["round_index"])
            v1 = per_round[key].get(0, "uncertain")
            v2 = per_round[key].get(1, "uncertain")
            combined = _combine(v1, v2)
            oc = v1 == v2 and v1 in ("candidate", "current")
            if combined == "better":
                n_better += 1
            f.write(json.dumps({**it, "new_v1": v1, "new_v2": v2,
                                "new_verdict": combined,
                                "new_order_consistent": oc}, ensure_ascii=False) + "\n")
    print(f"-> {out}   (new judge says 'better' on {n_better}/{len(items)} = {n_better/max(1,len(items)):.1%} of rounds)")

    # how well does the new judge agree with the TRUE offline delta?
    tp = fp = tn = fn = 0
    for it in items:
        key = (it["file"], it["sample_id"], it["round_index"])
        v1 = per_round[key].get(0, "uncertain"); v2 = per_round[key].get(1, "uncertain")
        acc = accept_revision(_combine(v1, v2), v1 == v2 and v1 in ("candidate", "current"),
                              it["current"], it["candidate"], None)
        truth = (it["delta_offline"] or 0.0) > 0
        if acc and truth: tp += 1
        elif acc and not truth: fp += 1
        elif not acc and not truth: tn += 1
        else: fn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"vs the true offline delta: accepted-and-truly-better={tp}  "
          f"accepted-but-worse={fp}  rejected-but-better={fn}")
    print(f"  precision={prec:.1%}  recall={rec:.1%}   "
          f"(a reliable judge should raise BOTH; the length guard raises precision by destroying recall)")
    return 0


def evaluate(path: Path, runs_root: str, split: str) -> int:
    """Replay acceptance with stored vs re-judged verdicts and score the corpus."""
    recs_by_file = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        d = json.loads(line)
        recs_by_file[d["file"]].append(d)
    print(f"{len(recs_by_file)} trace files re-judged")

    scorers, refcache = {}, {}
    print(f"\n{'cell':28s}{'draft':>8s}{'stored':>9s}{'rejudge':>9s}{'rejudge+tau':>12s}{'oracle':>9s}")
    print("-" * 78)
    tot = defaultdict(float)
    for f, rounds in sorted(recs_by_file.items()):
        recs = load(Path(f))
        task = recs[0]["task"]
        scorers.setdefault(task, Scorer(task))
        if task not in refcache:
            refcache[task] = {r.sample_id: r.reference
                              for r in read_manifest(manifest_path(task, split))}
        sc, refs, prim = scorers[task], refcache[task], PRIMARY[task]
        newv = {(d["sample_id"], d["round_index"]): d for d in rounds}

        D, S, R, RT, O = [], [], [], [], []
        for r in recs:
            ref = refs.get(r["sample_id"])
            if ref is None:
                continue
            D.append((ref, r["initial_draft"]))
            cur_s = cur_r = cur_rt = r["initial_draft"]
            best, bm = cur_s, r["initial_metric_offline"] or 0.0
            for rd in r["rounds"]:
                if rd.get("controller_action") != "REFINE":
                    continue
                cand = rd.get("candidate", "")
                if rd["metric_offline"] > bm:
                    best, bm = cand, rd["metric_offline"]
                if accept_revision(rd.get("judge_verdict", "uncertain"),
                                   bool(rd.get("judge_order_consistent")), cur_s, cand, None):
                    cur_s = cand
                nv = newv.get((r["sample_id"], rd["round_index"]))
                if nv:
                    if accept_revision(nv["new_verdict"], nv["new_order_consistent"], cur_r, cand, None):
                        cur_r = cand
                    if accept_revision(nv["new_verdict"], nv["new_order_consistent"], cur_rt, cand, 1.02):
                        cur_rt = cand
            S.append((ref, cur_s)); R.append((ref, cur_r)); RT.append((ref, cur_rt)); O.append((ref, best))
        m = lambda X: sc.score_corpus(X)[prim]
        d, s, rr, rt, o = m(D), m(S), m(R), m(RT), m(O)
        cell = f"{task}/{recs[0]['model']}"
        print(f"{cell:28s}{d:8.2f}{s:9.2f}{rr:9.2f}{rt:12.2f}{o:9.2f}")
        tot["d"] += d; tot["s"] += s; tot["r"] += rr; tot["rt"] += rt; tot["o"] += o
    print("-" * 78)
    print(f"{'SUM':28s}{tot['d']:8.2f}{tot['s']:9.2f}{tot['r']:9.2f}{tot['rt']:12.2f}{tot['o']:9.2f}")
    print(f"{'gain vs draft':28s}{'':8s}{tot['s']-tot['d']:+9.2f}{tot['r']-tot['d']:+9.2f}"
          f"{tot['rt']-tot['d']:+12.2f}{tot['o']-tot['d']:+9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
