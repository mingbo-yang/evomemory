"""Dataset manifests and leak-proof test/auxiliary isolation.

Design notes (v5 plan, sections 1 and 6):

* The test set must reproduce the *existing baseline* exactly: first-N prefix in
  original order, original row numbers, duplicates preserved.  We therefore load
  it through ``baseline.core.data.load_examples`` rather than re-deriving it.
* Auxiliary data must be disjoint from the test set **by input hash**.  This is a
  *filter*, not an assertion: ``wmt19`` train.parquet genuinely shares 1 English
  and 2 Chinese strings with validation.parquet, so a pure "assert zero overlap"
  check would fail.  We drop colliding rows instead.
* Every sample carries a stable id plus the sha256 of its source text, so the
  ablation arms can be checked for leakage and for identity of the initial draft.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import AUX_SIZES, EXP_ROOT, TASKS, TEST_SAMPLES, aux_sizes

# --- Paths -----------------------------------------------------------------

WMT19_TRAIN = Path(
    "/mnt/huawei/wwq/project/No-box-translation/ISSTA2025/11.2-ISSTA/data/wmt19-en-zh/train.parquet"
)
COEDIT_GEC_CSV = Path("/mnt/huawei/ymb/icml/rag/coedit_gec_dataset_analysis_full.csv")
GIGAWORD_TEST_CSV = Path("/mnt/huawei/ymb/icml/rag/gigaword_dataset_analysis_full.csv")
GIGAWORD_TINY_DIR = Path(
    "/mnt/huawei/ymb/datasets/datasets--SpeedOfMagic--gigaword_tiny/snapshots/"
    "f2876f564f2a11d1781a265b60faea461da72d5b/data"
)

# icml built its GEC test set as the first 3000 rows of the coedit `gec` subset.
# Anything at or beyond this index is clean auxiliary material.
COEDIT_GEC_TEST_ROWS = 3000


@dataclass(frozen=True)
class SampleRef:
    sample_id: str
    task: str
    split: str
    row_index: int
    source: str
    reference: str
    source_hash: str
    provenance: str

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _make(task: str, split: str, row_index: int, source: str, reference: str, provenance: str) -> SampleRef:
    h = sha256_text(source)
    return SampleRef(
        sample_id=f"{task}/{split}/{row_index:06d}",
        task=task,
        split=split,
        row_index=row_index,
        source=source,
        reference=reference,
        source_hash=h,
        provenance=provenance,
    )


# --- Test manifests (must match baseline exactly) ---------------------------


def build_test_manifest(task: str) -> List[SampleRef]:
    """Load the first-N test prefix exactly as the baseline runner does."""
    from baseline_core.data import load_examples  # baseline, read-only

    n = TEST_SAMPLES[task]
    examples, policy = load_examples(task, n)
    prov = f"baseline.core.data.load_examples(task={task}, max_samples={n}); policy={policy}"
    return [
        _make(task, "test", ex.index, str(ex.source), str(ex.reference), prov)
        for ex in examples
    ]


# --- Auxiliary manifests ----------------------------------------------------


def _iter_wmt_train_pairs(direction: str) -> Iterable[Tuple[int, str, str]]:
    """Lazily yield (row_index, source, reference) from the 2M-row train split.

    Streaming in row-group batches with early exit keeps this cheap: only a few
    hundred clean rows are needed and the file is 212 MB.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(WMT19_TRAIN)
    idx = 0
    for batch in pf.iter_batches(batch_size=20000, columns=["translation"]):
        for tr in batch.column("translation").to_pylist():
            en, zh = str(tr["en"]), str(tr["zh"])
            yield (idx, en, zh) if direction == "en_zh" else (idx, zh, en)
            idx += 1


def _load_coedit_gec() -> List[Tuple[int, str, str]]:
    """Return (row_index, src, tgt) for the whole coedit `gec` subset, in icml order.

    ``icml/rag/data_gec.py`` filters ``task == "gec"`` and then takes
    ``select(range(3000))``; row indices here refer to that *filtered* order so
    that indices >= 3000 are guaranteed to be outside the icml test set.
    """
    import os

    os.environ.setdefault("HF_HOME", "/mnt/huawei/ymb/.cache/huggingface")
    from datasets import load_dataset

    ds = load_dataset("grammarly/coedit", split="train")
    ds = ds.filter(lambda x: x.get("task", "") == "gec")
    return [(i, str(ds[i]["src"]), str(ds[i]["tgt"])) for i in range(len(ds))]


def _load_gigaword_splits(splits: Sequence[str]) -> List[Tuple[int, str, str, str]]:
    """Return (global_index, source, reference, split_name) from gigaword_tiny.

    ``gigaword_tiny`` ships three *disjoint* 100-row splits.  The aaai2027 test
    CSV was built from the ``train`` split (91/100 document overlap), so
    ``validation`` and ``test`` are clean auxiliary material.
    """
    import pandas as pd

    out: List[Tuple[int, str, str, str]] = []
    idx = 0
    for split in splits:
        df = pd.read_parquet(GIGAWORD_TINY_DIR / f"{split}-00000-of-00001.parquet")
        for i in range(len(df)):
            out.append((idx, str(df.iloc[i]["document"]), str(df.iloc[i]["summary"]), split))
            idx += 1
    return out


def _test_source_hashes(task: str) -> set:
    return {r.source_hash for r in build_test_manifest(task)}


def build_aux_splits(task: str) -> Dict[str, List[SampleRef]]:
    """Build initial/dev/accumulation splits, filtered against the test set."""
    sizes = aux_sizes(task)
    needed = sum(sizes.values())
    forbidden = _test_source_hashes(task)

    if task in ("wmt19_en_zh", "wmt19_zh_en"):
        direction = "en_zh" if task == "wmt19_en_zh" else "zh_en"
        # Filter against BOTH directions' test inputs: wmt19 validates both
        # directions on the same parallel pairs, so either side leaking is a leak.
        forbidden |= _test_source_hashes(
            "wmt19_zh_en" if task == "wmt19_en_zh" else "wmt19_en_zh"
        )
        raw = _iter_wmt_train_pairs(direction)
        provenance = f"wmt19 train.parquet (direction={direction}), filtered against test hashes"
    elif task == "coedit_gec":
        raw = _load_coedit_gec()
        raw = raw[COEDIT_GEC_TEST_ROWS:]  # drop the icml test prefix
        provenance = f"coedit gec rows >= {COEDIT_GEC_TEST_ROWS}"
    elif task == "gigaword":
        rows = _load_gigaword_splits(["validation", "test"])
        raw = [(i, s, r) for (i, s, r, _sp) in rows]
        provenance = "gigaword_tiny validation+test splits"
    else:
        raise ValueError(f"unknown task {task!r}")

    kept: List[SampleRef] = []
    dropped_collision = 0
    dropped_duplicate = 0
    seen_hashes = set()
    for row_index, source, reference in raw:
        h = sha256_text(source)
        if h in forbidden:
            dropped_collision += 1
            continue
        if h in seen_hashes:
            dropped_duplicate += 1
            continue
        seen_hashes.add(h)
        kept.append(_make(task, "aux", row_index, source, reference, provenance))
        if len(kept) >= needed:
            break

    if len(kept) < needed:
        raise RuntimeError(
            f"{task}: auxiliary pool exhausted. needed={needed} got={len(kept)} "
            f"(dropped_collision={dropped_collision}, dropped_duplicate={dropped_duplicate}). "
            "Refusing to silently shrink the auxiliary set."
        )

    out: Dict[str, List[SampleRef]] = {}
    cursor = 0
    for split in ("initial", "dev", "accumulation"):
        n = sizes[split]
        chunk = kept[cursor : cursor + n]
        cursor += n
        out[split] = [
            SampleRef(**{**s.to_dict(), "sample_id": f"{task}/{split}/{i:06d}", "split": split})
            for i, s in enumerate(chunk)
        ]
    return out


# --- Persistence ------------------------------------------------------------


def manifest_path(task: str, split: str) -> Path:
    return EXP_ROOT / "data" / "manifests" / f"{task}__{split}.jsonl"


def write_manifest(refs: Iterable[SampleRef], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in refs:
        payload.append(json.dumps(r.to_dict(), ensure_ascii=False))
    text = "\n".join(payload) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> List[SampleRef]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(SampleRef(**json.loads(line)))
    return out


def build_all(verbose: bool = True) -> dict:
    """Build and persist manifests for every task/split.  Returns a summary."""
    summary = {}
    for task in TASKS:
        test = build_test_manifest(task)
        h_test = write_manifest(test, manifest_path(task, "test"))
        splits = build_aux_splits(task)
        hashes = {}
        for split, refs in splits.items():
            hashes[split] = write_manifest(refs, manifest_path(task, split))
        summary[task] = {
            "test_n": len(test),
            "test_manifest_sha256": h_test,
            "aux_n": {k: len(v) for k, v in splits.items()},
            "aux_manifest_sha256": hashes,
        }
        if verbose:
            print(
                f"[MANIFEST] {task:14s} test={len(test):5d} "
                + " ".join(f"{k}={len(v):4d}" for k, v in splits.items())
            )
    return summary


# --- Isolation --------------------------------------------------------------


def check_isolation(task: str) -> dict:
    """Verify test ∩ aux == ∅ by source hash for one task."""
    test = read_manifest(manifest_path(task, "test"))
    report = {"task": task, "test_n": len(test), "violations": []}
    test_hashes = {r.source_hash for r in test}
    for split in ("initial", "dev", "accumulation"):
        refs = read_manifest(manifest_path(task, split))
        overlap = test_hashes & {r.source_hash for r in refs}
        if overlap:
            report["violations"].append({"split": split, "n_overlap": len(overlap)})
        report[f"{split}_n"] = len(refs)
    # cross-split disjointness
    seen: Dict[str, str] = {}
    for split in ("initial", "dev", "accumulation"):
        for r in read_manifest(manifest_path(task, split)):
            if r.source_hash in seen:
                report["violations"].append(
                    {"split": split, "duplicate_of": seen[r.source_hash], "hash": r.source_hash}
                )
            seen[r.source_hash] = split
    report["ok"] = not report["violations"]
    return report
