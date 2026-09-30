#!/usr/bin/env python
"""Phase 8 self-test: the accumulation loop of run_accumulation.py.

``run_accumulation.py`` needs a GPU, so this drives the real ``main()`` with a
stub engine (``baseline_core.llm.LLMClient`` replaced, ``torch`` replaced by a
no-op module) against the real manifests and the real initial library, writing
only under /tmp.  It checks the things a GPU run must get right:

* checkpoints at 0 / chunk / ... / end, with a growing library;
* the evaluation reads the *snapshot* library (retrieval scores change as the
  library grows) and never writes to the experience store;
* the frozen baseline is measured once and reused;
* order variance is reported when --orders >= 2.

Run:  /home/ymb/miniconda3/envs/qwen35/bin/python tests/test_accumulation_loop.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import types
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import core  # noqa: F401  (registers baseline_core)
from baseline_core.types import Generation  # noqa: E402

# stub torch so nothing in this test can reach CUDA
fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
sys.modules["torch"] = fake_torch

import run_accumulation as ra  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class FakeClient:
    backend = "fake"

    def __init__(self, *a, **kw):
        self.round = 0

    def count_tokens(self, text: str) -> int:
        return max(1, len(str(text)) // 4)

    def generate(self, prompt, system_prompt, seed, call_type, max_tokens,
                 temperature=0.1, top_p=1.0):
        if call_type == "controller":
            self.round += 1
            action = "REFINE" if self.round % 4 in (1, 2) else "STOP"
            text = json.dumps({"action": action,
                               "instruction": "Tighten the wording." if action == "REFINE" else "",
                               "reason": "stub"})
        elif call_type.startswith("judge"):
            text = json.dumps({"verdict": "A", "reason": "stub prefers A"})
        elif call_type == "initial":
            text = "STUB DRAFT for " + prompt[:40]
        else:
            text = "STUB REVISION of " + prompt[:40]
        gen = Generation(text=text, input_tokens=10, output_tokens=5, latency=0.01,
                         seed=seed, call_type=call_type, start_time=0.0, end_time=0.01,
                         prompt=prompt)
        # harmless alias: a 04:13 revision of core/batched_pipeline.py read
        # getattr(g, "latency_s") although Generation calls the field `latency`
        # (that revision was fixed again; the alias simply keeps this stub
        # compatible with both spellings)
        gen.latency_s = gen.latency
        return gen


def lib_fingerprint() -> dict:
    """Hash the experience LIBRARY store (initial + snapshots).

    experience/indexes/ is excluded: it is a retrieval memo cache that live
    production runs append to concurrently, and this script passes cache=None
    everywhere so it never writes there.
    """
    out = {}
    for sub in ("initial", "snapshots"):
        for p in sorted((EXP / "experience" / sub).rglob("*")):
            if p.is_file():
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def main() -> int:
    import baseline_core.llm as bllm

    bllm.LLMClient = FakeClient

    out_dir = Path("/tmp/acc_selftest/runs")
    scores_dir = Path("/tmp/acc_selftest/scores")
    for d in (out_dir, scores_dir):
        if d.exists():
            shutil.rmtree(d)
    ra.SCORES_DIR = scores_dir

    before = lib_fingerprint()
    sys.argv = [
        "run_accumulation.py", "--gpu", "0", "--models", "qwen3-8b",
        "--tasks", "wmt19_en_zh", "--limit", "8", "--eval-limit", "6",
        "--chunk-size", "4", "--orders", "2", "--out-dir", str(out_dir),
    ]
    rc = ra.main()
    check("main() returns 0", rc == 0, f"rc={rc}")
    after = lib_fingerprint()
    check("the experience store was not modified by the evaluation", before == after)

    rows = list(csv.DictReader((scores_dir / "accumulation_curve.csv").open()))
    key = {(r["order"], r["checkpoint"], r["condition"]) for r in rows}
    check("curve has natural/shuffle1 x checkpoints 0,4,8 x dynamic",
          {("natural", "0", "dynamic"), ("natural", "8", "dynamic"),
           ("shuffle1", "8", "dynamic")} <= key, str(sorted(key)))
    check("the frozen condition is present", ("natural", "0", "frozen") in key)

    snap_dir = out_dir / "qwen3-8b" / "wmt19_en_zh" / "natural"
    snaps = sorted(snap_dir.glob("library_c*.jsonl"))
    check("one snapshot per checkpoint", len(snaps) == 3, str(snaps))
    sizes = [sum(1 for line in p.open() if line.strip()) for p in snaps]
    check(f"library grows across checkpoints {sizes}", sizes[0] < sizes[-1] and sizes == sorted(sizes))
    check("only the dynamic traces land in the order dir",
          len(list(snap_dir.glob("eval_c*.jsonl"))) == 3)
    check("the frozen traces are measured once",
          len(list((out_dir / "qwen3-8b" / "wmt19_en_zh" / "frozen").glob("eval_c*.jsonl"))) == 1)

    def scores_by_sample(path):
        recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        return {t["sample_id"]: [r["exp_scores"] for r in t["rounds"]] for t in recs}

    s0 = scores_by_sample(snap_dir / "eval_c0000_dynamic.jsonl")
    s8 = scores_by_sample(snap_dir / "eval_c0008_dynamic.jsonl")
    changed = [k for k in s0 if s0[k] != s8[k]]
    check("the c=8 evaluation sees the grown library (retrieval scores change)",
          bool(changed), f"{len(changed)}/{len(s0)} samples changed")
    fresh = [r for r in rows if r["condition"] == "frozen" and r["frozen_reused"] == "0"]
    reused = [r for r in rows if r["condition"] == "frozen" and r["frozen_reused"] == "1"]
    check("the frozen baseline is measured exactly once and reused afterwards",
          len(fresh) == 1 and len(reused) == 5
          and (fresh[0]["order"], fresh[0]["checkpoint"]) == ("natural", "0"),
          f"{len(fresh)} fresh / {len(reused)} reused")

    ov_path = scores_dir / "accumulation_order_variance.json"
    check("order variance is reported for --orders 2", ov_path.exists())
    if ov_path.exists():
        ov = json.loads(ov_path.read_text())
        check("order variance covers every checkpoint",
              set(ov) == {"qwen3-8b/wmt19_en_zh/c0000", "qwen3-8b/wmt19_en_zh/c0004",
                          "qwen3-8b/wmt19_en_zh/c0008"}, str(list(ov)))
        check("order variance sees both orders",
              all(v["n_orders"] == 2 for v in ov.values()))

    print()
    if FAILURES:
        print(f"FAILURES: {FAILURES}")
        return 1
    print("OK: accumulation loop verified with a stub engine (no GPU used)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
