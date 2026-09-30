#!/usr/bin/env python
"""Self-test for crash-resume: a resumed run must equal a from-scratch run.

Why this test exists
--------------------
Run outputs used to be opened ``"w"``, so every restart -- an engine death, a
supervisor retry, an operator restart -- silently discarded every sample already
computed.  Resume now keeps the persisted prefix and runs only the missing
manifest positions.  That is only safe if a resumed run reproduces *exactly*
what a from-scratch run would have produced for those positions.

The failure mode this guards against is subtle: the stored drafts are looked up
**positionally** and every engine call is seeded from the item's index, so a
resume that renumbered its work would quietly pair sample *i* with sample *j*'s
draft and seed.  Nothing would crash; the numbers would just be wrong.

How the test makes that detectable
----------------------------------
The stub engine's output is a pure function of the **call seed**, and the seed
is derived from the item's index in the manifest.  So if resume renumbers
anything, the generated text changes and the comparison fails.  The controller,
judge, retriever, scorer, prompt renderer and draft cache are the real ones.

No GPU is used and nothing under ``runs/`` is written.

Run:  /home/ymb/miniconda3/envs/qwen35/bin/python tests/test_resume_equivalence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core  # noqa: F401  (registers baseline_core)
from baseline_core.types import Generation  # noqa: E402
from core.bm25_fields import ExperienceRetriever  # noqa: E402
from core.manifest import manifest_path, read_manifest  # noqa: E402
from core.pipeline import RunConfig  # noqa: E402

import run_experiment as rex  # noqa: E402

FAILURES: list[str] = []
N_ITEMS = 24
BATCH = 8
TASK = "wmt19_en_zh"
MODEL = "qwen3-8b"


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


class SeedStubLLM:
    """Deterministic engine whose output depends only on the call seed.

    The seed encodes the caller's manifest index, which is what makes an
    index-shifting resume visible instead of silent.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    # ``BatchedLLM`` wraps whatever client it is given and only takes its own
    # batched path when the backend is vLLM; anything else falls through to the
    # per-request ``generate`` below, which keeps this test GPU-free while still
    # exercising the real indexing/seeding/draft-lookup logic.
    backend = "stub"

    def count_tokens(self, text: str) -> int:
        return max(1, len(str(text)) // 4)

    @staticmethod
    def _text(call_type: str, seed: int) -> str:
        if call_type.startswith("controller"):
            return json.dumps({"action": "REFINE",
                               "instruction": f"tighten the phrasing (s{seed})",
                               "reason": f"a concrete defect s{seed}"})
        if call_type.startswith("judge"):
            return json.dumps({"verdict": "B", "reason": f"prefers revised s{seed}"})
        return f"CANDIDATE seed={seed}"

    def generate_batch(self, reqs):
        out = []
        for r in reqs:
            self.calls.append((r.call_type, r.seed))
            out.append(Generation(
                text=self._text(r.call_type, r.seed), input_tokens=3, output_tokens=2,
                latency=0.0, seed=r.seed, call_type=r.call_type,
                start_time=0.0, end_time=0.0, prompt=r.prompt))
        return out

    def generate(self, prompt, system_prompt, seed, call_type, max_tokens,
                 temperature=0.1, top_p=1.0):
        self.calls.append((call_type, seed))
        return Generation(text=self._text(call_type, seed), input_tokens=3,
                          output_tokens=2, latency=0.0, seed=seed, call_type=call_type,
                          start_time=0.0, end_time=0.0, prompt=prompt)


def make_cfg() -> RunConfig:
    return RunConfig(
        task=TASK, model=MODEL, experience_mode="full", stop_mode="adaptive",
        alpha=0.5, k=4, max_rounds=3, seed=42, snapshot_id="initial",
        draft_source="stored", draft_cache=None,
    )


def new_pipe(cfg, library, llm):
    from core.batched_pipeline import BatchedPipeline
    return BatchedPipeline(cfg, ExperienceRetriever(library), list(library), llm,
                           cache=None, batch_size=BATCH, online=False)


def canon(traces) -> dict:
    """Sample id -> record, with the timing fields normalised away."""
    out = {}
    for tr in traces:
        rec = tr.to_dict()
        cost = dict(rec.get("cost") or {})
        cost.pop("latency_s", None)
        rec["cost"] = cost
        rec.pop("config", None)
        out[rec["sample_id"]] = json.dumps(rec, sort_keys=True, ensure_ascii=False)
    return out


def main() -> int:
    refs = read_manifest(manifest_path(TASK, "test"))[:N_ITEMS]
    print(f"manifest items: {len(refs)}  batch={BATCH}  task={TASK}  model={MODEL}")

    library = rex.build_library(MODEL, TASK, "full")
    print(f"experience library: {len(library)} units")
    check("library is non-empty (retrieval path is exercised)", bool(library))
    cfg = make_cfg()

    # ---- reference: one from-scratch run over everything --------------------
    base = new_pipe(cfg, library, SeedStubLLM()).run(refs)
    base_map = canon(base)
    check("baseline produced every item", len(base_map) == N_ITEMS,
          f"{len(base_map)} != {N_ITEMS}")
    check("baseline items are distinct", len({t.sample_id for t in base}) == N_ITEMS)

    # ---- equivalence of the refactor itself ---------------------------------
    # indices=range(n) must be byte-identical to passing no indices at all.
    allidx = new_pipe(cfg, library, SeedStubLLM()).run(refs, indices=list(range(N_ITEMS)))
    check("indices=range(n) is identical to a full run", canon(allidx) == base_map,
          "the explicit-index path changed behaviour")

    # ---- the real scenario: crash after 2 chunks, then resume ---------------
    part1 = new_pipe(cfg, library, SeedStubLLM()).run(refs, indices=list(range(0, 16)))
    part2 = new_pipe(cfg, library, SeedStubLLM()).run(refs, indices=list(range(16, N_ITEMS)))
    resumed = canon(part1)
    resumed.update(canon(part2))
    check("resumed run covers every item", len(resumed) == N_ITEMS,
          f"{len(resumed)} != {N_ITEMS}")
    check("RESUMED == FROM SCRATCH (record for record)", resumed == base_map,
          "a resumed run diverged from a full run")

    # ---- the dangerous variant: resume renumbers its work -------------------
    # Passing a *filtered* ref list (the naive implementation) must NOT be
    # equivalent -- otherwise this test proves nothing.
    naive = new_pipe(cfg, library, SeedStubLLM()).run(refs[16:])
    naive_map = canon(naive)
    check("naive renumbered resume is detectably different (test is meaningful)",
          naive_map != {k: v for k, v in base_map.items() if k in naive_map},
          "the stub cannot see an index shift, so equivalence is vacuous")

    # ---- gap case: a hole in the middle, not just a tail --------------------
    hole = [i for i in range(N_ITEMS) if i not in (5, 6, 7, 13)]
    a = new_pipe(cfg, library, SeedStubLLM()).run(refs, indices=hole)
    b = new_pipe(cfg, library, SeedStubLLM()).run(refs, indices=[5, 6, 7, 13])
    merged = canon(a)
    merged.update(canon(b))
    check("resume across an interior hole is also exact", merged == base_map)

    # ---- run_experiment-level guards ---------------------------------------
    tmp = Path("/tmp/resume_equiv")
    tmp.mkdir(exist_ok=True)
    good = base[0].to_dict()
    jpath = tmp / "full_static.jsonl"
    lines = [json.dumps(good, ensure_ascii=False)]
    # a duplicate id, a truncated final line, a foreign task, a stale config
    lines.append(json.dumps(good, ensure_ascii=False))
    lines.append(json.dumps(base[1].to_dict(), ensure_ascii=False)[:80])
    foreign = dict(base[2].to_dict()); foreign["task"] = "wmt19_zh_en"
    lines.append(json.dumps(foreign, ensure_ascii=False))
    stale = dict(base[3].to_dict()); stale["config_hash"] = "0000000000000000"
    lines.append(json.dumps(stale, ensure_ascii=False))
    jpath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    prior, ids, dropped = rex._load_prior(jpath, cfg.config_hash, TASK, MODEL, 42)
    check("only the one valid record is kept", len(prior) == 1, f"kept {len(prior)}")
    check("the kept record is the right one", prior[0].sample_id == good["sample_id"])
    check("duplicate / truncated / foreign / stale are all dropped", dropped == 4,
          f"dropped {dropped}, expected 4")
    check("a stale config_hash can never be resurrected", stale["sample_id"] not in ids)

    # ---- _persist_merged restores manifest order after an interior hole -----
    j2, c2 = tmp / "m.jsonl", tmp / "m.csv"
    fields = ["task", "model", "arm", "seed", "sample_id", "n_rounds",
              "initial_metric_offline", "final_metric_offline",
              "total_tokens", "latency_s", "n_calls"]
    rex._persist_merged(j2, c2, list(reversed(base)), refs, fields, TASK, MODEL, "full_static", 42)
    got = [json.loads(l)["sample_id"] for l in j2.read_text(encoding="utf-8").splitlines()]
    check("merged output is rewritten in manifest order",
          got == [r.sample_id for r in refs], "order differs from the manifest")
    check("csv row count matches", len(c2.read_text().strip().splitlines()) == N_ITEMS + 1)

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {FAILURES}")
        return 1
    print("all resume-equivalence checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
