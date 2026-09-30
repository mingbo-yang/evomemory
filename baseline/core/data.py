from __future__ import annotations

import csv
import os
from typing import List, Optional, Tuple

from .config import COEDIT_GEC_CSV, GIGAWORD_CSV, TASK_CONFIGS, WMT19_PARQUET
from .types import TaskExample


def _limit_examples(examples: List[TaskExample], max_samples: Optional[int]) -> List[TaskExample]:
    if max_samples is None:
        return examples
    return examples[: max(0, max_samples)]


def _load_wmt(task: str, max_samples: Optional[int]) -> Tuple[List[TaskExample], str]:
    n = max_samples if max_samples is not None else TASK_CONFIGS[task].default_samples
    examples: List[TaskExample] = []
    path = WMT19_PARQUET["validation"]

    try:
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files={"validation": path})["validation"]
        upper = min(n, len(ds))
        for i in range(upper):
            tr = ds[i]["translation"]
            if task == "wmt19_en_zh":
                src, ref = tr["en"], tr["zh"]
            else:
                src, ref = tr["zh"], tr["en"]
            examples.append(TaskExample(i, str(src), str(ref), task, {"dataset_path": path}))
    except Exception:
        import pandas as pd

        df = pd.read_parquet(path)
        upper = min(n, len(df))
        for i in range(upper):
            tr = df.iloc[i]["translation"]
            if task == "wmt19_en_zh":
                src, ref = tr["en"], tr["zh"]
            else:
                src, ref = tr["zh"], tr["en"]
            examples.append(TaskExample(i, str(src), str(ref), task, {"dataset_path": path}))

    return examples, "match_icml_3000" if max_samples is None else f"debug_{len(examples)}"


def _read_csv_dicts(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _load_csv_task(task: str, path: str, source_cols: List[str], ref_cols: List[str], max_samples: Optional[int]) -> Tuple[List[TaskExample], str]:
    rows = _read_csv_dicts(path)
    n = max_samples if max_samples is not None else TASK_CONFIGS[task].default_samples
    out: List[TaskExample] = []
    for i, row in enumerate(rows[:n]):
        src = next((row[c] for c in source_cols if c in row and row[c] is not None), "")
        ref = next((row[c] for c in ref_cols if c in row and row[c] is not None), "")
        out.append(TaskExample(i, str(src), str(ref), task, {"dataset_path": path}))
    return out, TASK_CONFIGS[task].sample_policy if max_samples is None else f"debug_{len(out)}"


def load_examples(task: str, max_samples: Optional[int] = None) -> Tuple[List[TaskExample], str]:
    if task in {"wmt19_en_zh", "wmt19_zh_en"}:
        return _load_wmt(task, max_samples)
    if task == "coedit_gec":
        return _load_csv_task(task, COEDIT_GEC_CSV, ["原文", "en", "source"], ["完美答案", "zh", "target"], max_samples)
    if task == "gigaword":
        return _load_csv_task(task, GIGAWORD_CSV, ["原文", "document", "source"], ["完美答案", "summary", "target"], max_samples)
    raise ValueError(f"Unknown task: {task}")

