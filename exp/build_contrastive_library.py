#!/usr/bin/env python
"""Build the experience library from the ORIGINAL contrastive assets.

What was wrong before
---------------------
``build_experience.py`` generated a *fresh* draft from the initial prompt, let
the controller invent a "defect", refined, and recorded whatever happened.  Such
a unit has **no known-wrong answer and no known-correct answer**, so it is not a
contrastive example at all -- it is a log of an arbitrary edit attempt.  Measured
consequence: 75-84% of the resulting units were hurt/unchanged, retrieval
returned mostly those, the model imitated their instruction text (41% near
copies), and providing "experience" produced no quality gain at 1.9x the cost.

The intended design is contrastive example modelling: every unit is a real
**wrong -> right** pair with the correction that produced it.

Source assets (read-only)
-------------------------
``原文 | 不完美答案 | 完美答案 | 修改建议`` -- the imperfect answer is a genuine
model error, the perfect answer is the gold, and the advice is a written
explanation of the fix.

Mapping onto the Experience schema (identical to the original format)
--------------------------------------------------------------------
    source_input             <- 原文
    state_before             <- 不完美答案   (the real error)
    state_after              <- 完美答案     (the gold correction)
    intervention_instruction <- 修改建议     (how to get from wrong to right)
    outcome_label            <- "helped"     (true by construction)
    verdict                  <- "better"     (kept consistent with the label)

The library is NOT frozen: this only seeds it.  Online accumulation appends more
units, and those must obey the same wrong->right rule.

Run:
  python build_contrastive_library.py --out-root experience/contrastive
  python build_contrastive_library.py --out-root experience/contrastive --limit 5000
"""

from __future__ import annotations

import argparse
import csv
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

#: task -> (csv path, column names for source / imperfect / perfect / advice)
SOURCES = {
    "wmt19_en_zh": (
        "/mnt/huawei/wwq/model/aaa_experiment/code/wmt19_dataset_analysis_top10.csv",
        "原文", "不完美答案", "完美答案", "修改建议"),
    "wmt19_zh_en": (
        "/mnt/huawei/wwq/model/aaa_experiment/code/wmt19_zh_en_dataset_analysis_top10_3.csv",
        "原文", "不完美答案", "完美答案", "修改建议"),
    "coedit_gec": (
        "/mnt/huawei/ymb/icml/rag/coedit_gec_dataset_analysis_full.csv",
        "原文", "不完美答案", "完美答案", "修改建议"),
    "gigaword": (
        "/mnt/huawei/ymb/icml/rag/gigaword_dataset_analysis_full.csv",
        "原文", "不完美答案", "完美答案", "修改建议"),
    "gigaword_expand": (
        "/mnt/huawei/ymb/icml/rag/gigaword_expand_dataset_analysis_full.csv",
        "原摘要", "模型扩写", "参考扩写", "修改建议"),
}


def load_task(task: str, model: str, limit: int | None) -> tuple[list, Counter]:
    path, k_src, k_imp, k_per, k_adv = SOURCES[task]
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"contrastive asset missing for {task}: {p}")
    units, stats = [], Counter()
    with p.open(encoding="utf-8-sig", errors="replace") as f:
        rd = csv.DictReader(f)
        for i, row in enumerate(rd):
            if limit is not None and i >= limit:
                break
            src = (row.get(k_src) or "").strip()
            imp = (row.get(k_imp) or "").strip()
            per = (row.get(k_per) or "").strip()
            adv = (row.get(k_adv) or "").strip()
            if not src or not imp or not per:
                stats["skipped_empty"] += 1
                continue
            if imp == per:
                # no correction happened -> not a contrastive example
                stats["skipped_no_contrast"] += 1
                continue
            units.append(Experience(
                exp_id=f"{task}/{model}/c{i:07d}",
                task=task, model=model,
                source_input=src,
                state_before=imp,          # the real error
                state_after=per,           # the gold correction
                intervention_instruction=adv,
                intervention_rationale="",
                verdict="better",
                reason_a="gold-metric outcome: helped",
                reason_b="gold-metric outcome: helped",
                order_consistent=True,
                delta_offline=None,
                provenance="contrastive-asset",
                outcome_label="helped",
            ))
            stats["kept"] += 1
    return units, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--models", nargs="*", default=["glm4-9b", "llama3.1-8b", "qwen3-8b", "qwen3-4b"])
    ap.add_argument("--tasks", nargs="*", default=["wmt19_en_zh", "wmt19_zh_en", "coedit_gec", "gigaword"])
    ap.add_argument("--limit", type=int, default=None, help="cap rows read per task")
    args = ap.parse_args()

    out = Path(args.out_root)
    grand = Counter()
    for task in args.tasks:
        units, stats = load_task(task, "TEMPLATE", args.limit)
        for m in args.models:
            import dataclasses
            mu = [dataclasses.replace(u, exp_id=u.exp_id.replace("/TEMPLATE/", f"/{m}/"), model=m)
                  for u in units]
            dest = out / m / task / "initial.jsonl"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w", encoding="utf-8") as fh:
                for u in mu:
                    fh.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")
        grand.update(stats)
        print(f"  {task:14s} kept={stats['kept']:7d}  skipped(empty)={stats['skipped_empty']:5d}  "
              f"skipped(no contrast)={stats['skipped_no_contrast']:5d}  -> {out}/<model>/{task}/initial.jsonl")
    print(f"\nlibraries written under {out} for models: {', '.join(args.models)}")
    print(f"total units per model: {grand['kept']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
