#!/usr/bin/env python
"""Phase 8 gap closure: the NoNegativeExperience condition and 3 stream orders.

Two gaps in ``run_accumulation.py`` are verified here, GPU-free (the engine is a
stub and ``torch`` is replaced by a no-op module; the real manifests and the real
initial libraries are read, everything written goes to /tmp):

(a) **the third accumulation method.**  On a crafted history that contains a
    *negative* experience (``delta_offline < 0``) the three conditions have
    genuinely different libraries: Full-Dynamic keeps every observed unit,
    NoNegativeExperience keeps the same units minus the negative ones (a strict
    subsequence, same construction and retrieval order) and Frozen stays at the
    bootstrap library.  On a history with no negative outcome,
    NoNegativeExperience's library is identical to Full-Dynamic's -- which is the
    single-variable property the condition is supposed to have.  Both directions
    are checked through the real commit path (``process_stream_item`` on a
    crafted trace) and end-to-end through ``main()`` with a stub engine (a
    refiner that degrades the answer vs one that echoes it, so no negative
    transition exists).

(b) **three distinct, reproducible orders.**  ``--orders 3`` yields three
    different permutations of the same stream, order 0 being the manifest order;
    the permutation of each order is a pure function of ``(seed, order index)``,
    so a re-run reproduces it exactly (checked at the helper level *and* by
    running ``main()`` twice and comparing the per-order stream traces).

(c) **the existing dynamic-vs-frozen behaviour is unchanged.**  The dynamic and
    frozen curve rows and artefact bytes are identical whether or not the third
    condition is selected; every historical artefact path/format is still written
    (``order<k>/library_c####.jsonl``, ``frozen/eval_c0000_frozen.jsonl``,
    ``scores/accumulation_order_variance.json`` keyed ``model/task/c####``); and
    ``tests/test_accumulation_loop.py`` -- the pre-existing, unmodified test of
    that behaviour -- still passes.

(d) **all conditions share the evaluation setup.**  Instrumented
    ``BatchedPipeline`` instantiations show every condition/checkpoint evaluating
    the same sample ids, in the same chunk grid, with the same batch size, seed,
    round budget, alpha, k, stop mode and draft source; and at checkpoint 0
    (where the Full-Dynamic and NoNegativeExperience libraries are equal by
    construction) the eval traces and the full sequence of ``(call_type, seed)``
    engine calls are identical.  The stream itself is confirmed to run
    Full-Dynamic / adaptive generation for every condition, with the same budget.

Run:  /home/ymb/miniconda3/envs/qwen35/bin/python tests/test_accumulation_gaps.py
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import core  # noqa: F401  (registers baseline_core)
from baseline_core.types import Generation  # noqa: E402

# stub torch before anything can import it for real: nothing in this test may
# reach CUDA (all four cards are busy with production work)
fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
sys.modules["torch"] = fake_torch

import core.batched_pipeline as bp  # noqa: E402
import run_accumulation as ra  # noqa: E402
from core.bm25_fields import Experience  # noqa: E402
from core.manifest import read_manifest  # noqa: E402
from core.pipeline import RoundTrace, SampleTrace  # noqa: E402

FAILURES: list[str] = []
TMP = Path("/tmp/acc_gaps")

TASK = "wmt19_en_zh"
MODEL = "qwen3-8b"


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}"
          + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# stub engine
# --------------------------------------------------------------------------- #


class StubClient:
    """Deterministic GPU-free engine, with two refiner behaviours.

    ``REFINE_MODE = "degrade_once"``: the first round of an item replaces the
    (reference-text) draft with a mark that scores ~0 instead of ~100, so that
    transition has a negative ``delta_offline`` -- the negative experience
    NoNegativeExperience must drop -- while the following rounds echo the
    degraded text back (delta 0).  A run therefore observes *both* kinds of unit
    and the three conditions' libraries are all different.  The rule is
    content-based, so it needs no hidden counter and is identical in every
    process.

    ``REFINE_MODE = "echo"``: every revision is the current answer verbatim, so
    every transition has ``delta_offline == 0`` and no negative experience exists
    at all.
    """

    backend = "stub"
    REFINE_MODE = "degrade_once"
    CALLS: list = []  # (call_type, seed), for the seed-equality checks

    def __init__(self, *a, **kw):
        pass

    def count_tokens(self, text: str) -> int:
        return max(1, len(str(text)) // 4)

    @staticmethod
    def _current_answer(prompt: str) -> str:
        marker = "Current answer:\n"
        start = prompt.index(marker) + len(marker)
        return prompt[start:prompt.index("\n\nRevision instruction:", start)]

    @staticmethod
    def _judge_preference(prompt: str) -> str:
        """Prefer the slot that holds the degraded revision.

        That is what makes the judge ACCEPT a revision which lowers the offline
        metric -- exactly the negative-experience phenomenon the plan studies
        (the real pipeline produces some: see reports/progress.md).  The choice
        is a pure function of the anonymous A/B slots, so the double-order judge
        is consistent and the revision is accepted.
        """
        slot_a = prompt.split("Answer A:\n", 1)[1].split("\n\nAnswer B:", 1)[0]
        slot_b = prompt.split("Answer B:\n", 1)[1].split("\n\nWhich answer", 1)[0]
        if "STUB REVISION" in slot_b and "STUB REVISION" not in slot_a:
            return "B"
        return "A"

    def generate(self, prompt, system_prompt, seed, call_type, max_tokens,
                 temperature=0.1, top_p=1.0):
        StubClient.CALLS.append((call_type, seed))
        if call_type == "controller":
            text = json.dumps({"action": "REFINE", "instruction": "Tighten the wording.",
                               "reason": "stub"})
        elif call_type.startswith("judge"):
            text = json.dumps({"verdict": self._judge_preference(prompt),
                               "reason": "stub prefers the marked revision"})
        elif call_type == "initial":
            # no quotes / adapter markers, so parse_output() is idempotent
            text = "STUB DRAFT answer text"
        else:  # refine
            if StubClient.REFINE_MODE == "echo":
                text = self._current_answer(prompt)
            else:  # degrade_once
                cur = self._current_answer(prompt)
                text = (cur if cur.startswith("STUB REVISION")
                        else "STUB REVISION degraded " + prompt[:32].replace("\n", " "))
        gen = Generation(text=text, input_tokens=10, output_tokens=5, latency=0.01,
                         seed=seed, call_type=call_type, start_time=0.0, end_time=0.01,
                         prompt=prompt)
        gen.latency_s = gen.latency
        return gen


def run_main(argv: list, scores_dir: Path, capture: bool = False):
    """Drive the real main() with the stub engine.  Returns rc (or (rc, stdout))."""
    import baseline_core.llm as bllm

    bllm.LLMClient = StubClient
    ra.SCORES_DIR = scores_dir
    old_argv = sys.argv
    sys.argv = ["run_accumulation.py", *argv]
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ra.main()
    finally:
        sys.argv = old_argv
    text = buf.getvalue()
    if rc != 0:
        print(text[-3000:])
    return (rc, text) if capture else rc


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def base_argv(out_dir: Path, **over) -> list:
    argv = [
        "--gpu", "0", "--models", MODEL, "--tasks", TASK,
        "--limit", "8", "--eval-limit", "6", "--chunk-size", "4",
        "--orders", "1", "--out-dir", str(out_dir),
    ]
    for key, val in over.items():
        argv += [f"--{key.replace('_', '-')}", str(val)]
    return argv


def lib_ids(path: Path) -> list:
    return [json.loads(l)["exp_id"] for l in path.read_text().splitlines() if l.strip()]


def lib_records(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def initial_path() -> Path:
    return ra.experience_dir(MODEL, TASK) / "initial.jsonl"


def manifest_ids(n: int) -> list:
    return [r.sample_id for r in read_manifest(ra.manifest_path(TASK, "accumulation"))][:n]


def stream_draft_cache(n: int) -> Path:
    """Freeze the stream drafts to the manifest reference text.

    The reference scores 100 on the task metric, so the "degrade_once" refiner
    provably lowers it (~0) and the observed transition has a negative
    ``delta_offline`` -- deterministically, independent of tokeniser quirks.
    """
    path = TMP / "stream_drafts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ref in read_manifest(ra.manifest_path(TASK, "accumulation"))[:n]:
            f.write(json.dumps({"sample_id": ref.sample_id, "draft": ref.reference},
                               ensure_ascii=False) + "\n")
    return path


# --------------------------------------------------------------------------- #
# (a) the third condition
# --------------------------------------------------------------------------- #


def crafted_trace(sample_id: str, rounds: list) -> SampleTrace:
    """A trace with explicitly chosen per-round offline deltas."""
    rts = []
    for i, (delta, verdict) in enumerate(rounds):
        rts.append(RoundTrace(
            round_index=i, exp_ids=[], exp_scores=[], exp_sim_input=[], exp_sim_state=[],
            exp_class_counts={}, controller_action="REFINE", controller_instruction="fix it",
            controller_reason="crafted", controller_structured_failure=False,
            candidate=f"candidate {sample_id} r{i}", judge_verdict=verdict,
            judge_order_consistent=True, judge_reason_a="", judge_reason_b="",
            accepted=verdict == "better", metric_offline=None, delta_offline=delta, cost={},
        ))
    return SampleTrace(
        sample_id=sample_id, task=TASK, model=MODEL, config_hash="crafted", seed=42,
        source_hash="crafted", initial_draft="crafted draft", final_output="crafted final",
        rounds=rts, initial_metric_offline=0.0, final_metric_offline=0.0,
        draft_from_cache=False, cost={}, config={},
    )


class _FakePipe:
    """Stands in for core.pipeline.Pipeline in the commit-path unit test."""

    def __init__(self, trace: SampleTrace):
        self.trace = trace
        self.library = None
        self.retriever = None

    def run_sample(self, _ref, _index) -> SampleTrace:
        return self.trace


class _Ref:
    def __init__(self, source: str):
        self.source = source


def grow(history: list) -> tuple:
    """Run a crafted history through the REAL commit path.

    ``history`` is a list of ``(sample_id, [(delta, verdict), ...])``.  Returns
    the (frozen, dynamic, no_negative) libraries after the whole history.
    """
    bootstrap = [Experience(
        exp_id="INIT/0", task=TASK, model=MODEL, source_input="bootstrap", state_before="a",
        state_after="b", intervention_instruction="", intervention_rationale="",
        verdict="tie", reason_a="", reason_b="", order_consistent=False,
        delta_offline=None, provenance="initial",
    )]
    libs = {"dynamic": list(bootstrap), "no_negative": list(bootstrap)}
    for i, (sid, rounds) in enumerate(history):
        pipe = _FakePipe(crafted_trace(sid, rounds))
        ra.process_stream_item(
            None, pipe, _Ref(f"source of {sid}"), i, MODEL, libs["dynamic"], None,
            extra_libraries={"no_negative": libs["no_negative"]},
        )
    return list(bootstrap), libs["dynamic"], libs["no_negative"]


def test_crafted_history() -> None:
    print("\n--- (a) crafted history through the real commit path ---")
    # a history WITH a negative experience: s2/r0 made quality worse
    history = [
        ("s1", [(2.0, "better")]),
        ("s2", [(-1.5, "worse"), (0.5, "better")]),
        ("s3", [(0.0, "tie")]),
        ("s4", [(None, "uncertain")]),   # no recorded delta -> not provably negative
    ]
    frozen, dyn, noneg = grow(history)
    dyn_ids = [e.exp_id for e in dyn]
    nn_ids = [e.exp_id for e in noneg]
    check("Full-Dynamic keeps every observed unit, in observation order",
          dyn_ids == ["INIT/0", f"{TASK}/{MODEL}/s1/r0", f"{TASK}/{MODEL}/s2/r0",
                      f"{TASK}/{MODEL}/s2/r1", f"{TASK}/{MODEL}/s3/r0",
                      f"{TASK}/{MODEL}/s4/r0"], str(dyn_ids))
    check("NoNegativeExperience drops exactly the negative-outcome unit",
          nn_ids == ["INIT/0", f"{TASK}/{MODEL}/s1/r0", f"{TASK}/{MODEL}/s2/r1",
                     f"{TASK}/{MODEL}/s3/r0", f"{TASK}/{MODEL}/s4/r0"], str(nn_ids))
    check("NoNegativeExperience is a strict subsequence of Full-Dynamic "
          "(same construction and ordering)",
          nn_ids == [e for e in dyn_ids if e in set(nn_ids)] and len(nn_ids) < len(dyn_ids))
    check("only the negative-delta unit was dropped, nothing else",
          all(e.delta_offline is None or e.delta_offline >= 0 for e in noneg)
          and [e.exp_id for e in dyn if e.delta_offline is not None
               and e.delta_offline < 0] == [f"{TASK}/{MODEL}/s2/r0"])
    check("Frozen is untouched by the history", [e.exp_id for e in frozen] == ["INIT/0"])

    # the same crafted history WITHOUT a negative round: the two methods must
    # produce byte-identical libraries (what makes the contrast single variable)
    clean = [("s1", [(2.0, "better")]), ("s2", [(0.5, "better")]),
             ("s3", [(0.0, "tie")]), ("s4", [(None, "uncertain")])]
    frozen2, dyn2, nn2 = grow(clean)
    check("with no negative outcome the NoNegativeExperience library is identical "
          "to Full-Dynamic",
          [e.to_dict() for e in dyn2] == [e.to_dict() for e in nn2])
    check("... and the identical libraries are still genuinely grown (not vacuous)",
          len(dyn2) == 5 and [e.exp_id for e in frozen2] == ["INIT/0"])

    # predicate, incl. the fallback for units whose delta was never recorded
    def mk(delta, verdict):
        return Experience(
            exp_id="x", task=TASK, model=MODEL, source_input="", state_before="",
            state_after="", intervention_instruction="", intervention_rationale="",
            verdict=verdict, reason_a="", reason_b="", order_consistent=False,
            delta_offline=delta, provenance="")

    check("is_negative_outcome: delta<0 -> negative; 0 and >0 -> not",
          ra.is_negative_outcome(mk(-0.001, "worse"))
          and not ra.is_negative_outcome(mk(0.0, "worse"))
          and not ra.is_negative_outcome(mk(1.0, "uncertain")))
    check("is_negative_outcome: no recorded delta falls back to the judge verdict",
          ra.is_negative_outcome(mk(None, "worse"))
          and not ra.is_negative_outcome(mk(None, "tie"))
          and not ra.is_negative_outcome(mk(None, "better")))
    check("filter_for_condition(no_negative) drops only negatives",
          [e.exp_id for e in ra.filter_for_condition(
              "no_negative", [mk(-1.0, "worse"), mk(0.0, "tie"), mk(1.0, "better")])]
          == ["x", "x"])
    check("filter_for_condition(dynamic) is the identity",
          len(ra.filter_for_condition("dynamic", [mk(-1.0, "worse"), mk(0.0, "tie")])) == 2)
    check("condition_dir: dynamic keeps the historical layout, others get a subdir",
          ra.condition_dir(Path("order0"), "dynamic") == Path("order0")
          and ra.condition_dir(Path("order0"), "no_negative")
          == Path("order0/no_negative"))
    check("--conditions parses to the canonical order",
          ra.parse_conditions("no_negative,frozen,dynamic")
          == ["dynamic", "frozen", "no_negative"]
          and ra.parse_conditions(ra.DEFAULT_CONDITIONS) == list(ra.CONDITIONS))
    check("the default selects all three plan methods",
          ra.parse_conditions(ra.DEFAULT_CONDITIONS)
          == ["dynamic", "frozen", "no_negative"])
    for bad in ("", "dynamic,nope", "dynamic,dynamic"):
        try:
            ra.parse_conditions(bad)
            check(f"parse_conditions rejects {bad!r}", False)
        except ValueError:
            check(f"parse_conditions rejects {bad!r}", True)


def test_end_to_end_conditions() -> None:
    print("\n--- (a) end-to-end: three methods, both regimes ---")
    init_ids = lib_ids(initial_path())
    init_text = initial_path().read_text()
    drafts = stream_draft_cache(8)   # every stream item starts from its reference

    # ---- regime 1: the stream observes a negative experience --------------
    StubClient.REFINE_MODE = "degrade_once"
    out = fresh_dir(TMP / "degrade" / "runs")
    scores = fresh_dir(TMP / "degrade" / "scores")
    rc = run_main(base_argv(out, stream_draft_cache=drafts), scores)
    check("degrade-regime run returns 0", rc == 0, f"rc={rc}")
    order_dir = out / MODEL / TASK / "natural"
    dyn_last = lib_records(order_dir / "library_c0008.jsonl")
    nn_last = lib_records(order_dir / "no_negative" / "library_c0008.jsonl")
    check("Full-Dynamic snapshot owns negative-outcome units",
          any(r["delta_offline"] is not None and r["delta_offline"] < 0 for r in dyn_last))
    check("NoNegativeExperience snapshot owns none",
          all(r["delta_offline"] is None or r["delta_offline"] >= 0 for r in nn_last))
    check("the three conditions' libraries are genuinely different",
          len({tuple(r["exp_id"] for r in dyn_last),
               tuple(r["exp_id"] for r in nn_last),
               tuple(init_ids)}) == 3,
          f"dyn={len(dyn_last)} noneg={len(nn_last)} frozen={len(init_ids)}")
    nn_ids = [r["exp_id"] for r in nn_last]
    check("end-to-end, NoNegativeExperience is a strict subsequence of Full-Dynamic",
          nn_ids == [r["exp_id"] for r in dyn_last if r["exp_id"] in set(nn_ids)]
          and len(nn_ids) < len(dyn_last))
    check("Frozen == the bootstrap initial.jsonl (written snapshot c=0 is byte-identical)",
          (order_dir / "library_c0000.jsonl").read_text() == init_text)
    check("checkpoint 0 snapshots are identical for both accumulating methods",
          lib_ids(order_dir / "library_c0000.jsonl")
          == lib_ids(order_dir / "no_negative" / "library_c0000.jsonl"))
    check("library growth: frozen < no_negative < Full-Dynamic",
          len(init_ids) < len(nn_last) < len(dyn_last),
          f"{len(init_ids)} / {len(nn_last)} / {len(dyn_last)}")
    rows = list(csv.DictReader((scores / "accumulation_curve.csv").open()))
    check("the frozen curve rows measure the bootstrap library only",
          all(int(r["n_library"]) == len(init_ids) and int(r["library_growth"]) == 0
              for r in rows if r["condition"] == "frozen"))
    check("the no_negative curve rows never exceed the dynamic library size",
          all(int(a["n_library"]) <= int(b["n_library"])
              for a in rows if a["condition"] == "no_negative"
              for b in rows if b["condition"] == "dynamic"
              and a["checkpoint"] == b["checkpoint"]))

    # ---- regime 2: the stream observes no negative experience -------------
    StubClient.REFINE_MODE = "echo"
    out2 = fresh_dir(TMP / "echo" / "runs")
    scores2 = fresh_dir(TMP / "echo" / "scores")
    rc = run_main(base_argv(out2, stream_draft_cache=drafts), scores2)
    check("echo-regime run returns 0", rc == 0, f"rc={rc}")
    order2 = out2 / MODEL / TASK / "natural"
    for cp in (0, 4, 8):
        d = (order2 / f"library_c{cp:04d}.jsonl").read_text()
        n = (order2 / "no_negative" / f"library_c{cp:04d}.jsonl").read_text()
        check(f"no negative experience -> libraries identical at c={cp}", d == n)
    echo_last = lib_records(order2 / "library_c0008.jsonl")
    check("echo-regime library still grew (test is not vacuous)",
          len(echo_last) > len(init_ids), f"{len(echo_last)} vs {len(init_ids)}")
    check("echo-regime deltas are all zero (the regime really has no negative)",
          all(r["delta_offline"] == 0.0 for r in echo_last[len(init_ids):]))
    check("Frozen is the bootstrap library in the echo regime too",
          (order2 / "library_c0000.jsonl").read_text() == init_text)


# --------------------------------------------------------------------------- #
# (b) three distinct, reproducible orders
# --------------------------------------------------------------------------- #


def test_orders() -> None:
    print("\n--- (b) --orders 3: distinct and reproducible ---")
    stream = read_manifest(ra.manifest_path(TASK, "accumulation"))[:32]
    mids = [r.sample_id for r in stream]

    check("order_seed is a pure function of (seed, order index, attempt)",
          ra.order_seed(42, 1) == ra.order_seed(42, 1)
          and ra.order_seed(42, 1) != ra.order_seed(42, 2)
          and ra.order_seed(42, 1) != ra.order_seed(42, 1, 1)
          and ra.order_seed(42, 1) != ra.order_seed(43, 1))
    orders = ra.plan_orders(stream, 3, 42)
    ids = [[r.sample_id for r in o["stream"]] for o in orders]
    check("order 0 is the manifest order", ids[0] == mids)
    check("the three orders are distinct permutations of the stream",
          len({tuple(x) for x in ids}) == 3
          and all(sorted(x) == sorted(mids) for x in ids))
    check("order labels are natural/shuffle1/shuffle2",
          [o["label"] for o in orders] == ["natural", "shuffle1", "shuffle2"])
    again = ra.plan_orders(stream, 3, 42)
    check("a re-run reproduces every permutation and digest exactly",
          [[r.sample_id for r in o["stream"]] for o in again] == ids
          and [o["digest"] for o in again] == [o["digest"] for o in orders])
    check("a different base seed changes the shuffled orders",
          [[r.sample_id for r in o["stream"]] for o in ra.plan_orders(stream, 3, 7)][1:]
          != ids[1:])
    forced_a = ra.ordered_stream(stream, 1, 42, taken=[ids[1]])
    forced_b = ra.ordered_stream(stream, 1, 42, taken=[ids[1]])
    check("a permutation already taken is re-drawn (distinctness is enforced)",
          [r.sample_id for r in forced_a[0]] != ids[1]
          and [r.sample_id for r in forced_a[0]] == [r.sample_id for r in forced_b[0]]
          and forced_a[2] == forced_b[2] != orders[1]["digest"])

    # end-to-end: two runs of --orders 3 produce the same three orders
    def stream_orders(out: Path) -> list:
        return [
            [json.loads(l)["sample_id"]
             for l in (out / MODEL / TASK / lbl / "stream.jsonl").read_text().splitlines()
             if l.strip()]
            for lbl in ("natural", "shuffle1", "shuffle2")
        ]

    def curve_rows(scores: Path) -> list:
        rows = []
        for r in csv.DictReader((scores / "accumulation_curve.csv").open()):
            r.pop("duration_s", None)   # wall-clock, not part of the plan
            rows.append(r)
        return rows

    runs = []
    for tag in ("runA", "runB"):
        out = fresh_dir(TMP / "orders" / tag / "runs")
        scores = fresh_dir(TMP / "orders" / tag / "scores")
        argv = base_argv(out, limit=12, eval_limit=4, chunk_size=6, orders=3,
                         conditions="dynamic,frozen")
        runs.append((run_main(argv, scores), out, scores))
    check("both --orders 3 runs return 0", all(rc == 0 for rc, _, _ in runs))
    seq_a, seq_b = stream_orders(runs[0][1]), stream_orders(runs[1][1])
    check("the three stream orders are distinct end-to-end",
          len({tuple(s) for s in seq_a}) == 3)
    check("the three stream orders are permutations of the used stream",
          all(sorted(s) == sorted(seq_a[0]) for s in seq_a) and len(seq_a[0]) == 12)
    check("order 0 processed the manifest order", seq_a[0] == manifest_ids(12))
    check("a re-run reproduces all three orders exactly", seq_a == seq_b)
    check("a re-run reproduces the curve exactly (minus wall-clock)",
          curve_rows(runs[0][2]) == curve_rows(runs[1][2]))


# --------------------------------------------------------------------------- #
# (c) the existing dynamic-vs-frozen behaviour
# --------------------------------------------------------------------------- #


def artifact_bytes(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_dynamic_frozen_unchanged() -> None:
    print("\n--- (c) dynamic-vs-frozen behaviour is unchanged ---")
    runs = {}
    for tag, conditions in (("legacy", "dynamic,frozen"),
                            ("three", "dynamic,frozen,no_negative")):
        out = fresh_dir(TMP / "compat" / tag / "runs")
        scores = fresh_dir(TMP / "compat" / tag / "scores")
        argv = base_argv(out, limit=8, eval_limit=4, chunk_size=4, orders=2,
                         conditions=conditions)
        runs[tag] = (run_main(argv, scores), out, scores)
    check("both compat runs return 0", all(rc == 0 for rc, _, _ in runs.values()))

    def curve(scores: Path) -> list:
        rows = []
        for r in csv.DictReader((scores / "accumulation_curve.csv").open()):
            r.pop("duration_s", None)
            rows.append(r)
        return rows

    legacy_rows = [r for r in curve(runs["legacy"][2])
                   if r["condition"] in ("dynamic", "frozen")]
    three_rows = [r for r in curve(runs["three"][2])
                  if r["condition"] in ("dynamic", "frozen")]
    check("the dynamic/frozen curve rows are identical with and without the "
          "third condition", legacy_rows == three_rows,
          f"{len(legacy_rows)} vs {len(three_rows)} rows")
    check("the third condition adds only its own rows",
          {r["condition"] for r in curve(runs["three"][2])}
          == {"dynamic", "frozen", "no_negative"}
          and {r["condition"] for r in curve(runs["legacy"][2])}
          == {"dynamic", "frozen"})

    legacy_bytes = artifact_bytes(runs["legacy"][1])
    three_bytes = artifact_bytes(runs["three"][1])
    shared = {k: v for k, v in legacy_bytes.items() if "/no_negative/" not in k}
    check("every legacy artefact is byte-identical when the third condition runs",
          all(three_bytes.get(k) == v for k, v in shared.items()),
          str([k for k, v in shared.items() if three_bytes.get(k) != v][:4]))
    check("the third condition adds artefacts and changes nothing else",
          set(three_bytes) - set(legacy_bytes)
          == {k for k in three_bytes if "/no_negative/" in k}
          and set(legacy_bytes) <= set(three_bytes))

    # historical layout and frozen-reuse semantics
    order_dir = Path(runs["three"][1]) / MODEL / TASK / "natural"
    check("historical snapshot path order<k>/library_c####.jsonl still used",
          sorted(p.name for p in order_dir.glob("library_c*.jsonl"))
          == ["library_c0000.jsonl", "library_c0004.jsonl", "library_c0008.jsonl"])
    check("NoNegativeExperience snapshots live in their own subdirectory",
          sorted(p.name for p in (order_dir / "no_negative").glob("library_c*.jsonl"))
          == ["library_c0000.jsonl", "library_c0004.jsonl", "library_c0008.jsonl"])
    check("only Full-Dynamic eval traces stay directly in the order dir",
          sorted(p.name for p in order_dir.glob("eval_c*.jsonl"))
          == ["eval_c0000_dynamic.jsonl", "eval_c0004_dynamic.jsonl",
              "eval_c0008_dynamic.jsonl"])
    check("frozen traces keep their historical path "
          "(frozen/eval_c0000_frozen.jsonl, measured once)",
          len(list((Path(runs["three"][1]) / MODEL / TASK / "frozen")
                   .glob("eval_c*.jsonl"))) == 1)
    rows = curve(runs["three"][2])
    fresh = [r for r in rows if r["condition"] == "frozen" and r["frozen_reused"] == "0"]
    reused = [r for r in rows if r["condition"] == "frozen" and r["frozen_reused"] == "1"]
    check("frozen is measured once per model-task and reused by every condition",
          len(fresh) == 1 and len(reused) == 5
          and (fresh[0]["order"], fresh[0]["checkpoint"]) == ("natural", "0"))
    ov = json.loads((runs["three"][2] / "accumulation_order_variance.json").read_text())
    check("accumulation_order_variance.json keeps its historical keys/format",
          set(ov) == {f"{MODEL}/{TASK}/c0000", f"{MODEL}/{TASK}/c0004",
                      f"{MODEL}/{TASK}/c0008"}
          and all(v["n_orders"] == 2 for v in ov.values()), str(sorted(ov)))
    ov_nn = json.loads(
        (runs["three"][2] / "accumulation_order_variance_no_negative.json").read_text())
    check("NoNegativeExperience spread is reported in its own file",
          set(ov_nn) == set(ov) and all(v["n_orders"] == 2 for v in ov_nn.values()))
    check("the legacy run writes only the historical order-variance file",
          {p.name for p in Path(runs["legacy"][2]).glob("*.json")}
          == {"accumulation_order_variance.json"})

    # the pre-existing, unmodified test of this behaviour is the strongest oracle
    prior = subprocess.run(
        [sys.executable, str(EXP / "tests" / "test_accumulation_loop.py")],
        capture_output=True, text=True, cwd=str(EXP), timeout=1800,
    )
    check("tests/test_accumulation_loop.py (pre-existing, unmodified) still passes",
          prior.returncode == 0,
          (prior.stdout[-800:] + prior.stderr[-400:]) if prior.returncode else "")


def test_selection_independence() -> None:
    """Which methods are *evaluated* must not change what the stream *observes*."""
    print("\n--- selecting a subset of methods is single-variable ---")
    StubClient.REFINE_MODE = "degrade_once"
    out = fresh_dir(TMP / "subset" / "runs")
    scores = fresh_dir(TMP / "subset" / "scores")
    argv = base_argv(out, limit=8, eval_limit=4, chunk_size=4, orders=2,
                     conditions="no_negative,frozen",
                     stream_draft_cache=stream_draft_cache(8))
    rc = run_main(argv, scores)
    check("a run without Full-Dynamic selected returns 0", rc == 0, f"rc={rc}")
    order_dir = out / MODEL / TASK / "natural"
    check("only the selected method's artefacts are written",
          not list(order_dir.glob("library_c*.jsonl"))
          and not list(order_dir.glob("eval_c*.jsonl"))
          and bool(list((order_dir / "no_negative").glob("library_c*.jsonl"))))
    ref = (TMP / "degrade" / "runs" / MODEL / TASK / "natural" / "no_negative"
           / "library_c0008.jsonl")
    check("the NoNegativeExperience library is identical whether or not "
          "Full-Dynamic is evaluated (the stream is shared)",
          (order_dir / "no_negative" / "library_c0008.jsonl").read_bytes()
          == ref.read_bytes())
    check("order variance is reported for the selected method's own file",
          (scores / "accumulation_order_variance_no_negative.json").exists()
          and not (scores / "accumulation_order_variance.json").exists())


# --------------------------------------------------------------------------- #
# (d) shared evaluation setup
# --------------------------------------------------------------------------- #


def test_shared_eval_setup() -> None:
    print("\n--- (d) shared evaluation points, chunking and seeds ---")
    probes: list = []
    stream_cfgs: list = []
    orig_init, orig_run = bp.BatchedPipeline.__init__, bp.BatchedPipeline.run
    orig_run_sample = ra.Pipeline.run_sample

    def patched_init(self, config, retriever, library, client, cache=None,
                     batch_size=32, online=False):
        orig_init(self, config, retriever, library, client, cache=cache,
                  batch_size=batch_size, online=online)
        self._gap_probe = {
            "snapshot_id": config.snapshot_id, "seed": config.seed,
            "max_rounds": config.max_rounds, "k": config.k, "alpha": config.alpha,
            "stop_mode": config.stop_mode, "draft_source": config.draft_source,
            "experience_mode": config.experience_mode,
            "batch_size": self.batch_size, "n_library": len(list(library)),
        }
        probes.append(self._gap_probe)

    def patched_run(self, refs, sink=None, indices=None):
        pr = self._gap_probe
        pr["n_refs"] = len(refs)
        pr["sample_ids"] = [r.sample_id for r in refs]
        pr["chunk_grid"] = [(s, min(s + self.batch_size, len(refs)))
                            for s in range(0, len(refs), self.batch_size)]
        StubClient.CALLS = []
        try:
            return orig_run(self, refs, sink=sink, indices=indices)
        finally:
            pr["calls"] = list(StubClient.CALLS)

    def patched_run_sample(self, ref, index):
        stream_cfgs.append({
            "experience_mode": self.cfg.experience_mode,
            "stop_mode": self.cfg.stop_mode, "draft_source": self.cfg.draft_source,
            "max_rounds": self.cfg.max_rounds, "seed": self.cfg.seed,
            "alpha": self.cfg.alpha, "k": self.cfg.k,
            "n_library": len(self.library),
        })
        return orig_run_sample(self, ref, index)

    bp.BatchedPipeline.__init__ = patched_init
    bp.BatchedPipeline.run = patched_run
    ra.Pipeline.run_sample = patched_run_sample
    try:
        StubClient.REFINE_MODE = "degrade_once"
        out = fresh_dir(TMP / "shared" / "runs")
        scores = fresh_dir(TMP / "shared" / "scores")
        argv = base_argv(out, limit=6, eval_limit=4, chunk_size=3, orders=1,
                         conditions="dynamic,frozen,no_negative", batch_size=3)
        rc = run_main(argv, scores)
    finally:
        bp.BatchedPipeline.__init__ = orig_init
        bp.BatchedPipeline.run = orig_run
        ra.Pipeline.run_sample = orig_run_sample
    check("instrumented run returns 0", rc == 0, f"rc={rc}")

    # snapshot_id == acc_<order_label>_c<NNNN>_<condition> (condition labels may
    # contain "_", so anchor on the known set instead of splitting blindly)
    sid_re = re.compile(
        r"^acc_(?P<label>.+)_c(?P<cp>\d{4})_(?P<cond>dynamic|frozen|no_negative)$")
    by_label = {}
    for pr in probes:
        m = sid_re.match(pr["snapshot_id"])
        assert m, pr["snapshot_id"]
        by_label[(m["label"], int(m["cp"]), m["cond"])] = pr
    cps = {("natural", 0), ("natural", 3), ("natural", 6)}
    check("every accumulating condition was evaluated at every checkpoint",
          {(o, c, cond) for (o, c, cond) in by_label if cond != "frozen"} == {
              (o, c, cond) for (o, c) in cps for cond in ("dynamic", "no_negative")},
          str(sorted(by_label)))
    check("the frozen baseline was measured exactly once and reused "
          "(order_label 'frozen', checkpoint 0)",
          sorted(k for k in by_label if k[2] == "frozen") == [("frozen", 0, "frozen")],
          str(sorted(by_label)))

    shared_fields = ("n_refs", "sample_ids", "chunk_grid", "batch_size", "seed",
                     "max_rounds", "k", "alpha", "stop_mode", "draft_source")
    for (label, cp) in sorted(cps):
        grp = {cond: by_label[(label, cp, cond)]
               for cond in ("dynamic", "no_negative")}
        check(f"c={cp}: eval points / chunk grid / batch size / seeds identical "
              "across the accumulating conditions",
              all(len({json.dumps(grp[c][f], sort_keys=True) for c in grp}) == 1
                  for f in shared_fields),
              str({c: {f: grp[c][f] for f in shared_fields} for c in grp}))
        check(f"c={cp}: the eval batch size is --batch-size (3)",
              all(grp[c]["batch_size"] == 3 for c in grp)
              and all(grp[c]["chunk_grid"] == [(0, 3), (3, 4)] for c in grp))
        check(f"c={cp}: eval uses stored drafts, adaptive stop, full experience",
              all(grp[c]["draft_source"] == "stored"
                  and grp[c]["stop_mode"] == "adaptive"
                  and grp[c]["experience_mode"] == "full" for c in grp))
    frozen_pr = by_label[("frozen", 0, "frozen")]
    dyn0 = by_label[("natural", 0, "dynamic")]
    check("the reused frozen measurement shares the eval points / chunking / seeds "
          "of the accumulating conditions",
          all(json.dumps(frozen_pr[f], sort_keys=True) == json.dumps(dyn0[f], sort_keys=True)
              for f in shared_fields),
          str({f: (frozen_pr[f], dyn0[f]) for f in shared_fields
               if frozen_pr[f] != dyn0[f]}))
    check("frozen evaluates the bootstrap library, the others the snapshots",
          frozen_pr["n_library"] == dyn0["n_library"])

    # checkpoint 0: equal libraries -> identical traces AND identical seeds
    d0, n0 = dyn0, by_label[("natural", 0, "no_negative")]
    check("checkpoint 0 sees equal-sized libraries for both accumulating methods",
          d0["n_library"] == n0["n_library"])

    def eval_bodies(order_dir: Path, cond: str, cp: int) -> list:
        path = ra.condition_dir(order_dir, cond) / f"eval_c{cp:04d}_{cond}.jsonl"
        out_bodies = []
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                # config/config_hash are derived from the RunConfig, whose only
                # per-condition field is the snapshot_id label -- drop them so
                # the comparison covers drafts, retrieval, rounds and costs only
                rec.pop("config", None)
                rec.pop("config_hash", None)
                out_bodies.append(rec)
        return out_bodies

    order_dir = out / MODEL / TASK / "natural"
    check("checkpoint 0: Full-Dynamic and NoNegativeExperience eval traces are "
          "identical (same drafts, retrieval, rounds and seeds)",
          eval_bodies(order_dir, "dynamic", 0) == eval_bodies(order_dir, "no_negative", 0))
    check("checkpoint 0: the two methods made the identical (call_type, seed) "
          "engine calls", d0["calls"] == n0["calls"] and len(d0["calls"]) > 0,
          f"{len(d0['calls'])} vs {len(n0['calls'])} calls")
    check("the reused frozen measurement evals the same sample ids",
          frozen_pr["sample_ids"] == d0["sample_ids"])

    check("the stream runs Full-Dynamic/adaptive for every condition",
          bool(stream_cfgs) and all(c["experience_mode"] == "full"
                                    and c["stop_mode"] == "adaptive"
                                    for c in stream_cfgs))
    check("the stream uses freshly generated drafts, not the stored eval ones",
          all(c["draft_source"] == "generate" for c in stream_cfgs))
    check("the stream shares the eval round budget / alpha / k / seed",
          all(c["max_rounds"] == 3 and c["alpha"] == 0.5 and c["k"] == 4
              and c["seed"] == 42 for c in stream_cfgs))
    check("the stream's own retrieval library is the Full-Dynamic one (it grows)",
          stream_cfgs[-1]["n_library"] > stream_cfgs[0]["n_library"])


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_cli_defaults() -> None:
    print("\n--- CLI defaults ---")
    out = fresh_dir(TMP / "cli" / "runs")
    scores = fresh_dir(TMP / "cli" / "scores")
    rc, text = run_main(
        ["--dry-run", "--gpu", "0", "--models", MODEL, "--tasks", TASK,
         "--out-dir", str(out)], scores, capture=True)
    check("--dry-run returns 0", rc == 0, f"rc={rc}")
    check("--dry-run plans the plan's 3 accumulation orders by default",
          "planned orders=3 chunk=32" in text, text[-1500:])
    check("--dry-run prints a 3-method plan with the snapshot schedule",
          all(s in text for s in ("accumulation methods (3 selected",
                                  "Full-Dynamic", "Frozen", "NoNegativeExperience",
                                  "snapshot schedule", "checkpoints (processed items",
                                  "0,32,64,96,128")), text[-2500:])
    check("--dry-run shows the per-order permutation digests",
          "order0=natural#identity" in text and "order1=shuffle1#" in text
          and "order2=shuffle2#" in text, text[-2500:])
    check("--dry-run touches no GPU and loads no model",
          "[dry-run] setup validated" in text)
    proc = subprocess.run(
        [sys.executable, str(EXP / "run_accumulation.py"), "--help"],
        capture_output=True, text=True, cwd=str(EXP), timeout=300,
    )
    check("--help lists --orders (default 3) and --conditions (three methods)",
          "--orders" in proc.stdout and "default 3" in proc.stdout
          and "--conditions" in proc.stdout and "NoNegativeExperience" in proc.stdout,
          proc.stdout)
    rc2, text2 = run_main(
        ["--dry-run", "--gpu", "0", "--models", MODEL, "--tasks", TASK,
         "--orders", "2", "--conditions", "dynamic,frozen", "--out-dir", str(out)],
        scores, capture=True)
    check("--orders and --conditions stay overridable",
          rc2 == 0 and "planned orders=2" in text2
          and "accumulation methods (2 selected" in text2
          and "NoNegativeExperience" not in text2, text2[-1200:])


def main() -> int:
    print("Phase 8 gap tests: NoNegativeExperience + 3 accumulation orders "
          "(stub engine, no GPU)")
    fresh_dir(TMP)
    test_crafted_history()
    test_end_to_end_conditions()
    test_orders()
    test_dynamic_frozen_unchanged()
    test_selection_independence()
    test_shared_eval_setup()
    test_cli_defaults()
    print()
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        return 1
    print("OK: both Phase 8 gaps verified, GPU-free (stub engine; torch stubbed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
