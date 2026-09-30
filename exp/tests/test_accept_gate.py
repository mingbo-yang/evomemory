#!/usr/bin/env python
"""Acceptance-gate tests: the length guard, and that OFF is bit-identical.

Background
----------
Every cell's entire loss in the Phase 4 main experiment was attributable to the
acceptance gate, and none to the controller or the refiner (see
``reports/judge_bottleneck_diagnosis.md``).  The gate is "the judge prefers the
revision in both A/B orders", but the judge is the same model that wrote the
revision, so its preference is largely a *length* preference: for glm4-9b/zh_en,
84.7% of the revisions it called better were worse by the offline metric at a
mean length ratio of 1.57, versus 33.8% for qwen3-8b at a mean ratio of 1.05.

The fix is a length guard on acceptance, chosen on dev and frozen.  Two things
must hold and are checked here:

1. **With the guard off (``None``) behaviour is the historical behaviour** --
   any drift here would silently invalidate the runs already on disk.
2. **The rule lives in ONE place** and both pipelines call it.  The
   byte-identical-pair fix was previously applied only to the unbatched judge
   while production runs the batched path, so the "fix" was inert.  A guard
   that is duplicated is a guard that will diverge.

GPU-free.

Run:  /home/ymb/miniconda3/envs/qwen35/bin/python tests/test_accept_gate.py
"""

from __future__ import annotations

import glob
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core  # noqa: F401  (registers baseline_core)
from core.batched_pipeline import BatchedPipeline  # noqa: E402
from core.judge import accept_revision  # noqa: E402
from core.pipeline import Pipeline, RunConfig  # noqa: E402

import run_experiment as rex  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    R = accept_revision

    # ---- 1. the rule's semantics ------------------------------------------
    check("guard OFF: a preferred revision is accepted (historical behaviour)",
          R("better", True, "abc", "abcdefgh", None) is True)
    check("guard OFF: even a much longer revision is accepted -- the old gate "
          "did no length checking at all",
          R("better", True, "abc", "a" * 50, None) is True)
    check("guard ON: a 1.5x longer revision is rejected even though preferred",
          R("better", True, "a" * 10, "a" * 15, 1.05) is False)
    check("guard ON: same length accepted", R("better", True, "a" * 10, "a" * 10, 1.05) is True)
    check("guard ON: shorter accepted", R("better", True, "a" * 10, "a" * 9, 1.05) is True)
    check("guard ON: boundary is inclusive (exactly ratio -> accepted)",
          R("better", True, "a" * 20, "a" * 21, 1.05) is True)
    check("just past the boundary -> rejected",
          R("better", True, "a" * 20, "a" * 22, 1.05) is False)
    check("a non-preferred revision is rejected regardless of length",
          R("worse", True, "a" * 10, "a" * 10, 1.05) is False)
    check("an order-inconsistent verdict is rejected regardless of length",
          R("better", False, "a" * 10, "a" * 10, 1.05) is False)
    check("empty current answer does not divide by zero",
          isinstance(R("better", True, "", "x", 1.05), bool))

    # ---- 2. config hashes must be untouched --------------------------------
    # The guard is a *pipeline argument*, never a RunConfig field.  If it had
    # been added to RunConfig, every config hash would change and a supervisor
    # restart would silently discard the finished prefix of a running job.
    bad = ok = 0
    for f in sorted(glob.glob("runs/main/**/*.jsonl", recursive=True)
                    + glob.glob("runs/ablation/**/*.jsonl", recursive=True)):
        try:
            rec = json.loads(open(f).readline())
        except Exception:
            continue
        arm = f.split("/")[-1].replace(".jsonl", "")
        if arm not in rex.ARM_REGISTRY:
            continue
        sp = rex.ARM_REGISTRY[arm]
        cfg = RunConfig(task=rec["task"], model=rec["model"],
                        experience_mode=sp["experience_mode"], stop_mode=sp["stop_mode"],
                        alpha=0.5, k=4, max_rounds=3, seed=42,
                        snapshot_id="online" if sp["online"] else "initial",
                        draft_source="stored", draft_cache="")
        if cfg.config_hash == rec["config_hash"]:
            ok += 1
        else:
            bad += 1
            print(f"    mismatch: {f}")
    check(f"every production run's config hash is unchanged ({ok} runs)", bad == 0,
          f"{bad} changed -- a restart would discard their finished prefixes")

    # ---- 3. one rule, both paths ------------------------------------------
    bt = inspect.getsource(BatchedPipeline._round)
    ub = inspect.getsource(Pipeline.run_sample)
    check("batched pipeline uses the shared rule", "accept_revision(" in bt)
    check("unbatched pipeline uses the shared rule", "accept_revision(" in ub)
    # The BoN-J round is a THIRD implementation site and had its own copy of
    # the rule; grep the whole tree so a future one cannot hide either.
    dup_sites = [str(q) for q in Path(".").rglob("*.py")
                 if q.name != "test_accept_gate.py"
                 and ("accepted = verdict == \"better\" and order_consistent" in q.read_text(errors="replace")
                      or "accepted = verdict.accepted" in q.read_text(errors="replace"))]
    check("exactly ONE acceptance rule exists in the tree (no duplicates)", not dup_sites,
          f"duplicated in {dup_sites}")

    # ---- 4. the guard is reachable from the CLI, the supervisor, the queue --
    rx = Path("run_experiment.py").read_text(encoding="utf-8")
    sv = Path("supervisor.py").read_text(encoding="utf-8")
    dq = Path("dispatcher.py").read_text(encoding="utf-8")
    check("run_experiment.py exposes --accept-max-len-ratio",
          "--accept-max-len-ratio" in rx)
    check("run_experiment.py passes it to EVERY pipeline construction site (batched, BoN-J, unbatched)",
          rx.count("accept_max_len_ratio=accept_max_len_ratio)") == 3 and
          rx.count("accept_max_len_ratio=args.accept_max_len_ratio") == 1,
          f"found {rx.count('accept_max_len_ratio=args.accept_max_len_ratio')}")
    check("supervisor.py forwards the flag", "--accept-max-len-ratio" in sv)
    check("dispatcher.py forwards the flag", "accept_max_len_ratio" in dq)

    test_draft_source_rule_is_not_duplicated()

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print("all acceptance-gate checks passed")
    return 0


def test_draft_source_rule_is_not_duplicated() -> None:
    """The batching/draft-source rule exists in TWO places and must agree.

    It once did not: the pre-load gate in main() still refused every non-`stored`
    source under batching while run_model_task had already been relaxed, so aux
    runs (which must generate their own drafts) died with exit code 2 and burned
    their retries.  Assert both sites test the same condition.
    """
    import re
    src = Path("run_experiment.py").read_text(encoding="utf-8")
    early = re.search(r"if args\.batch_size and args\.batch_size > 1 and "
                      r"args\.draft_source == \"cached\"[^\n]*", src)
    late = re.search(r"if batch_size and batch_size > 1 and "
                     r"draft_source == \"cached\"[^\n]*", src)
    check("the pre-load gate allows generate/cached under batching",
          early is not None, "main() no longer has the relaxed condition")
    check("run_model_task tests the SAME condition as the pre-load gate",
          late is not None)
    check("neither site still refuses every non-stored source under batching",
          'args.draft_source != "stored"' not in src
          and 'draft_source != "stored"' not in src)


if __name__ == "__main__":
    raise SystemExit(main())
