#!/usr/bin/env python
"""Self-test for Phase 10 (``export_human_eval.py``) -- GPU-free, stdlib only.

The export reads traces that production runs are still appending to, so the
tests drive the real ``main()`` against a *synthetic* run tree in a temporary
directory (real manifests are used read-only for the task inputs, which is what
the real export does too).  Nothing outside the temp directory is written, no
model is loaded and no GPU is touched.

What is asserted
----------------
* the annotation file contains no arm name anywhere (proof of blinding), keeps
  the empty ``winner``/``confidence``/``notes`` columns, and the answer key is a
  separate file that maps every pair id to its two arms;
* a pair is always the same sample id under both arms, and the two anonymised
  outputs really are those two arms' ``final_output`` values;
* the A/B order is a deterministic function of ``(seed, sample_id)``: rerunning
  reproduces the package byte for byte, and a different seed reshuffles the
  order without changing which samples are compared;
* ``--limit`` is respected, the allocation is balanced across task x model as
  far as availability allows, and the realised counts are reported;
* partial runs are labelled (not silently mixed) and ``--complete-only``
  excludes them;
* a truncated final line and unparseable lines never crash the reader and never
  silently inflate a run's sample count;
* a requested baseline arm with no complete run fails loudly (exit 2) and
  nothing is written;
* the ``auto`` baseline picks the strongest non-experience arm from the data.

Run:  /home/ymb/miniconda3/envs/qwen35/bin/python tests/test_human_eval_export.py
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import export_human_eval as ex  # noqa: E402

# Tasks whose manifests exist and are read-only inputs of the export.
TASK_A = "wmt19_en_zh"   # expected_n = 1000 (runs below are deliberately partial)
TASK_B = "gigaword"      # expected_n = 100
MODELS = ("glm4-9b", "qwen3-8b")


def _round(i: int, delta: float, action: str = "REFINE") -> dict:
    return {
        "round_index": i,
        "exp_ids": [],
        "controller_action": action,
        "controller_instruction": "x",
        "controller_reason": "y",
        "candidate": f"cand{i}",
        "judge_verdict": "tie",
        "accepted": False,
        "metric_offline": 10.0 + delta,
        "delta_offline": delta,
        "cost": {"input_tokens": 1.0, "output_tokens": 1.0, "total_tokens": 2.0,
                 "latency_s": 0.1, "n_calls": 1.0},
    }


class ExportTestBase(unittest.TestCase):
    """Builds a synthetic runs/ tree that mirrors the production layout."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="human_eval_test_")
        self.root = Path(self.tmp.name)
        self.runs = self.root / "runs"
        self.out = self.root / "human_eval"
        self.counter = 0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # -- fixture helpers ----------------------------------------------------
    def _next_tag(self) -> int:
        self.counter += 1
        return self.counter

    def write_run(
        self,
        arm: str,
        task: str,
        model: str,
        n: int,
        *,
        experience_mode: str,
        base_metric: float = 10.0,
        sample_offset: int = 0,
        tree: str = "main",
        extra_lines: str = "",
        arm_field: str | None = None,
        jitter: float = 0.0,
    ) -> Path:
        """Write one trace file; returns its path.

        Outputs are deliberately arm-neutral strings (``out-1-...``) so the
        blinding test is meaningful: if the exporter leaked an arm name into the
        CSV, these fixtures could not hide it.
        """
        tag = self._next_tag()
        if tree == "main":
            path = self.runs / "main" / arm / "seed42" / task / model / f"{arm}.jsonl"
        else:
            path = self.runs / "ablation" / arm / "seed42" / task / model / f"{arm}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for i in range(n):
            idx = sample_offset + i
            sid = f"{task}/test/{idx:06d}"
            # arm-specific noise, so that two arms' paired differences have a
            # realistic spread (a constant offset would make every CI exclude 0)
            noise = 0.0
            if jitter:
                h = ((tag * 1000003 + idx) * 2654435761) % 4294967296
                noise = jitter * (h / 4294967296.0 - 0.5)
            rec = {
                "sample_id": sid,
                "task": task,
                "model": model,
                "config_hash": f"hash{tag}",
                "seed": 42,
                "source_hash": "deadbeef",
                "initial_draft": f"init-{sid}",
                "final_output": f"out-{tag}-{sid}",
                "initial_metric_offline": base_metric,
                "final_metric_offline": base_metric + (2.0 if experience_mode == "full" else 0.0)
                + noise,
                "draft_from_cache": True,
                "cost": {"input_tokens": 100.0, "output_tokens": 20.0, "total_tokens": 120.0,
                         "latency_s": 0.5, "n_calls": 3.0},
                "config": {"task": task, "model": model, "experience_mode": experience_mode,
                           "stop_mode": "adaptive", "alpha": 0.5, "k": 4, "max_rounds": 3,
                           "seed": 42, "snapshot_id": "initial", "draft_source": "stored"},
                "rounds": [_round(0, -1.0 if i % 3 == 0 else 1.0),
                           _round(1, 0.0, action="STOP")],
            }
            if arm_field:
                rec["arm"] = arm_field
            lines.append(json.dumps(rec, ensure_ascii=False))
        text = "\n".join(lines) + "\n" + extra_lines
        path.write_text(text, encoding="utf-8")
        return path

    def run_export(self, *argv: str) -> tuple[int, str]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = ex.main(list(argv))
        return code, buf.getvalue()

    def base_argv(self, *extra: str) -> list[str]:
        return ["--runs", str(self.runs / "main"), str(self.runs / "ablation"),
                "--out-dir", str(self.out), *extra]

    def read_pairs(self) -> tuple[list[dict], list[str]]:
        import csv as _csv

        with (self.out / ex.PAIRS_FILE_NAME).open(encoding="utf-8", newline="") as fh:
            reader = _csv.DictReader(fh)
            return list(reader), list(reader.fieldnames or [])

    def read_key(self) -> dict:
        return json.loads((self.out / ex.KEY_FILE_NAME).read_text(encoding="utf-8"))

    def seed_full_and_baseline(self, n: int = 12) -> None:
        """Both models on TASK_B (complete at n=100) plus partial TASK_A runs."""
        for model in MODELS:
            self.write_run("full_static", TASK_B, model, 100, experience_mode="full",
                           base_metric=20.0)
            self.write_run("no_experience", TASK_B, model, 100, experience_mode="none",
                           base_metric=18.0, tree="ablation")
            # TASK_A: 1000 expected, so these are PARTIAL
            self.write_run("full_static", TASK_A, model, 20, experience_mode="full",
                           base_metric=30.0)
            self.write_run("no_experience", TASK_A, model, 20, experience_mode="none",
                           base_metric=29.0, tree="ablation")


class TestPackageStructure(ExportTestBase):
    def test_blinding_columns_and_separate_key(self):
        self.seed_full_and_baseline()
        code, out = self.run_export(*self.base_argv("--limit", "12"))
        self.assertEqual(code, 0, out)

        rows, fields = self.read_pairs()
        self.assertEqual(fields, list(ex.ANNOTATION_COLUMNS))
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertEqual(row["winner"], "")
            self.assertEqual(row["confidence"], "")
            self.assertEqual(row["notes"], "")
            self.assertTrue(row["input"], "task input must be exported")
            self.assertTrue(row["system_a"] and row["system_b"])

        # --- proof of blinding: no arm name anywhere in the annotation file
        raw = (self.out / ex.PAIRS_FILE_NAME).read_text(encoding="utf-8")
        for arm in ("full_static", "no_experience"):
            self.assertNotIn(arm, raw, f"arm name {arm!r} leaked into the annotation CSV")
        # ...nor any arm-identifying column name
        self.assertNotIn("arm", fields)

        # --- the key is separate, complete and consistent
        key = self.read_key()
        self.assertEqual(key["n_pairs"], len(rows))
        self.assertEqual(set(key["pairs"]), {r["pair_id"] for r in rows})
        for pid, entry in key["pairs"].items():
            self.assertIn(entry["A"], ("full_static", "no_experience"))
            self.assertIn(entry["B"], ("full_static", "no_experience"))
            self.assertNotEqual(entry["A"], entry["B"])
        self.assertEqual(ex.verify_blinding(self.out / ex.PAIRS_FILE_NAME,
                                            ["full_static", "no_experience"]), 0)

    def test_readme_documents_ties(self):
        self.seed_full_and_baseline()
        code, out = self.run_export(*self.base_argv("--limit", "6"))
        self.assertEqual(code, 0, out)
        readme = (self.out / ex.README_FILE_NAME).read_text(encoding="utf-8")
        self.assertIn("tie", readme.lower())
        self.assertIn("Ties are allowed", readme)
        self.assertIn(ex.KEY_FILE_NAME, readme)
        self.assertIn("same sample id", readme)

    def test_pairing_is_same_sample_id_and_real_outputs(self):
        self.seed_full_and_baseline()
        code, out = self.run_export(*self.base_argv("--limit", "16"))
        self.assertEqual(code, 0, out)
        rows, _ = self.read_pairs()
        key = self.read_key()

        # rebuild sample_id -> final_output for each arm from the synthetic traces
        truth: dict[tuple[str, str, str, str], str] = {}
        for run in ex.discover_runs([self.runs / "main", self.runs / "ablation"],
                                    models=list(MODELS)):
            for rec in run.records:
                truth[(run.arm, run.task, run.model, rec["sample_id"])] = rec["final_output"]

        for row in rows:
            entry = key["pairs"][row["pair_id"]]
            sid = entry["sample_id"]
            a_out = truth[(entry["A"], entry["task"], entry["model"], sid)]
            b_out = truth[(entry["B"], entry["task"], entry["model"], sid)]
            self.assertNotEqual(a_out, b_out)
            self.assertIn(row["system_a"], (a_out, b_out))
            self.assertIn(row["system_b"], (a_out, b_out))
            self.assertNotEqual(row["system_a"], row["system_b"])
            # the two systems in one row must be two different arms of that sample
            self.assertEqual(entry["task"], row["task"])
            self.assertEqual(entry["model"], row["model"])
            # inputs come from the real manifest and match the row's sample
            self.assertTrue(row["input"])

    def test_ab_order_is_deterministic_and_seed_sensitive(self):
        self.seed_full_and_baseline()
        code, out = self.run_export(*self.base_argv("--limit", "16", "--seed", "7"))
        self.assertEqual(code, 0, out)
        rows1, _ = self.read_pairs()
        key1 = self.read_key()
        csv1 = (self.out / ex.PAIRS_FILE_NAME).read_bytes()

        code, out = self.run_export(*self.base_argv("--limit", "16", "--seed", "7"))
        self.assertEqual(code, 0, out)
        csv2 = (self.out / ex.PAIRS_FILE_NAME).read_bytes()
        key2 = self.read_key()
        self.assertEqual(csv1, csv2, "same seed must reproduce the package byte for byte")
        self.assertEqual(key1["pairs"], key2["pairs"])

        code, out = self.run_export(*self.base_argv("--limit", "16", "--seed", "8"))
        self.assertEqual(code, 0, out)
        rows3, _ = self.read_pairs()
        key3 = self.read_key()
        self.assertEqual({r["pair_id"] for r in rows1}, {r["pair_id"] for r in rows3})
        # same samples are compared ...
        self.assertEqual(
            sorted(v["sample_id"] for v in key1["pairs"].values()),
            sorted(v["sample_id"] for v in key3["pairs"].values()),
        )
        # ... but the A/B order changes for at least one pair (a 16-pair package
        # would have to be pathologically unlucky for that not to happen)
        flipped = sum(1 for pid, v in key1["pairs"].items()
                      if key3["pairs"][pid]["A"] != v["A"])
        self.assertGreater(flipped, 0, "changing the seed must reshuffle A/B")
        self.assertTrue(all(rows3[i]["pair_id"] == f"P{i + 1:04d}" for i in range(len(rows3))))

    def test_presentation_order_function(self):
        orders = [ex.presentation_order(42, f"task/test/{i:06d}", "full", "base")[0]
                  for i in range(200)]
        self.assertEqual(len(set(orders)), 2, "both A/B orders must occur")
        again = [ex.presentation_order(42, f"task/test/{i:06d}", "full", "base")[0]
                 for i in range(200)]
        self.assertEqual(orders, again, "presentation order must be a pure function")


class TestAllocationAndCoverage(ExportTestBase):
    def test_limit_is_respected_and_balanced(self):
        self.seed_full_and_baseline()
        code, out = self.run_export(*self.base_argv("--limit", "8"))
        self.assertEqual(code, 0, out)
        rows, _ = self.read_pairs()
        self.assertEqual(len(rows), 8)
        key = self.read_key()
        per_cell: dict[str, int] = {}
        for entry in key["pairs"].values():
            cell = f"{entry['task']}/{entry['model']}"
            per_cell[cell] = per_cell.get(cell, 0) + 1
        self.assertEqual(len(per_cell), 4, f"all four cells should get pairs: {per_cell}")
        self.assertEqual(sorted(per_cell.values()), [2, 2, 2, 2], per_cell)
        # 50 samples available per cell for TASK_B, 20 for TASK_A -> no cell saturated
        for cell, entry in key["coverage"].items():
            self.assertEqual(entry["realised"], per_cell.get(cell, 0))

    def test_allocation_redistributes_leftovers(self):
        avail = {"a": 100, "b": 100, "c": 1, "d": 0}
        alloc = ex.allocate(avail, 10, ["a", "b", "c", "d"])
        self.assertEqual(sum(alloc.values()), 10)
        self.assertEqual(alloc["d"], 0)
        self.assertEqual(alloc["c"], 1, "no cell may exceed its availability")
        self.assertLessEqual(abs(alloc["a"] - alloc["b"]), 1)
        self.assertGreaterEqual(min(alloc["a"], alloc["b"]), 4,
                                "the leftover must be redistributed to cells with spare data")

        # tiny budget: every cell with data gets a chance before any gets two
        alloc = ex.allocate({"a": 5, "b": 5, "c": 5}, 2, ["a", "b", "c"])
        self.assertEqual(sum(alloc.values()), 2)
        self.assertEqual(len([v for v in alloc.values() if v > 0]), 2)

    def test_disjoint_sample_ids_are_never_mixed(self):
        self.write_run("full_static", TASK_B, "qwen3-8b", 100, experience_mode="full")
        self.write_run("no_experience", TASK_B, "qwen3-8b", 100, experience_mode="none",
                       sample_offset=500, tree="ablation")
        code, out = self.run_export(*self.base_argv())
        self.assertEqual(code, 2, out)
        self.assertIn("no pair could be built", out)
        self.assertFalse((self.out / ex.PAIRS_FILE_NAME).exists())

    def test_partial_labelled_and_complete_only(self):
        self.seed_full_and_baseline()
        code, out = self.run_export(*self.base_argv("--limit", "200"))
        self.assertEqual(code, 0, out)
        key = self.read_key()
        self.assertTrue(key["partial_runs_included"], "partial runs must be reported")
        self.assertTrue(any("wmt19_en_zh" in p for p in key["partial_runs_included"]))
        statuses = {k: v["full_status"] for k, v in key["coverage"].items()}
        self.assertEqual(statuses["wmt19_en_zh/qwen3-8b"], "partial")
        self.assertEqual(statuses["gigaword/qwen3-8b"], "complete")

        code, out = self.run_export(*self.base_argv("--limit", "200", "--complete-only"))
        self.assertEqual(code, 0, out)
        rows, _ = self.read_pairs()
        key = self.read_key()
        for entry in key["pairs"].values():
            self.assertEqual(entry["task"], TASK_B)
        self.assertEqual(len(rows), 200)
        self.assertFalse((self.out / ex.README_FILE_NAME).read_text().count("PARTIAL runs used"))

    def test_no_usable_pairs_fails_loudly(self):
        # only partial runs, and --complete-only: nothing can be paired
        self.write_run("full_static", TASK_A, "glm4-9b", 20, experience_mode="full")
        self.write_run("no_experience", TASK_A, "glm4-9b", 20, experience_mode="none",
                       tree="ablation")
        code, out = self.run_export(*self.base_argv("--models", "glm4-9b", "--complete-only"))
        self.assertEqual(code, 2, out)
        self.assertIn("[FATAL]", out)
        self.assertFalse((self.out / ex.PAIRS_FILE_NAME).exists())


class TestFailLoudly(ExportTestBase):
    def test_missing_baseline_arm(self):
        self.write_run("full_static", TASK_B, "qwen3-8b", 100, experience_mode="full")
        code, out = self.run_export(*self.base_argv("--baseline-arm", "sr_j_stop"))
        self.assertEqual(code, 2, out)
        self.assertIn("has NO run at all", out)
        self.assertFalse((self.out / ex.PAIRS_FILE_NAME).exists())

    def test_partial_baseline_arm(self):
        self.write_run("full_static", TASK_B, "qwen3-8b", 100, experience_mode="full")
        self.write_run("sr_j_stop", TASK_B, "qwen3-8b", 7, experience_mode="none",
                       tree="ablation")
        code, out = self.run_export(*self.base_argv("--baseline-arm", "sr_j_stop"))
        self.assertEqual(code, 2, out)
        self.assertIn("no COMPLETE run", out)

    def test_auto_requires_a_complete_non_experience_arm(self):
        self.write_run("full_static", TASK_B, "qwen3-8b", 100, experience_mode="full")
        self.write_run("no_experience", TASK_B, "qwen3-8b", 40, experience_mode="none",
                       tree="ablation")
        code, out = self.run_export(*self.base_argv())
        self.assertEqual(code, 2, out)
        self.assertIn("no non-experience arm has a COMPLETE run", out)

    def test_no_traces_at_all(self):
        self.runs.mkdir(parents=True, exist_ok=True)
        code, out = self.run_export(*self.base_argv())
        self.assertEqual(code, 2, out)
        self.assertIn("no usable traces", out)


class TestBaselineSelection(ExportTestBase):
    def test_auto_picks_the_strongest_non_experience_arm(self):
        # sr_j_fixed is weaker than no_experience in this cell, so the *data*
        # (not the priority list) must decide
        self.write_run("full_static", TASK_B, "qwen3-8b", 100, experience_mode="full",
                       base_metric=20.0)
        self.write_run("sr_j_fixed", TASK_B, "qwen3-8b", 100, experience_mode="none",
                       base_metric=10.0, tree="ablation")
        self.write_run("no_experience", TASK_B, "qwen3-8b", 100, experience_mode="none",
                       base_metric=15.0, tree="ablation")
        runs = sorted(ex.group_runs(ex.discover_runs(
            [self.runs / "main", self.runs / "ablation"])).values(), key=lambda r: r.rel)
        choice = ex.choose_baseline_arm(runs, "full_static", [TASK_B], ["qwen3-8b"])
        self.assertEqual(choice.arm, "no_experience", choice.reason)
        ranks = {row["arm"]: row["rank"] for row in choice.evidence}
        self.assertEqual(ranks["no_experience"], 1)
        self.assertEqual(ranks["sr_j_fixed"], 2)
        self.assertIn("selected", [row["note"] for row in choice.evidence
                                   if row["arm"] == "no_experience"])

    def test_coverage_tiebreak_prefers_the_wider_arm_when_indistinguishable(self):
        # sr_j_fixed is a hair stronger but complete in ONE cell; no_experience is
        # complete in TWO cells and statistically indistinguishable -> pick the
        # wider one (and say so in the reason)
        for task, n in ((TASK_B, 100), (TASK_A, 1000)):
            self.write_run("full_static", task, "qwen3-8b", n, experience_mode="full",
                           base_metric=20.0, jitter=4.0)
            if task == TASK_B:  # the stronger candidate is complete in ONE cell only
                self.write_run("sr_j_fixed", task, "qwen3-8b", n, experience_mode="none",
                               base_metric=20.005, tree="ablation", jitter=4.0)
            self.write_run("no_experience", task, "qwen3-8b", n, experience_mode="none",
                           base_metric=20.0, tree="ablation", jitter=4.0)
        runs = sorted(ex.group_runs(ex.discover_runs(
            [self.runs / "main", self.runs / "ablation"])).values(), key=lambda r: r.rel)
        choice = ex.choose_baseline_arm(runs, "full_static", [TASK_A, TASK_B], ["qwen3-8b"])
        self.assertEqual(choice.arm, "no_experience", choice.reason)
        self.assertIn("coverage tie-break", choice.reason)
        strict = ex.choose_baseline_arm(runs, "full_static", [TASK_A, TASK_B], ["qwen3-8b"],
                                        coverage_tiebreak=False)
        self.assertEqual(strict.arm, "sr_j_fixed", strict.reason)

    def test_experience_arms_are_not_candidates(self):
        self.write_run("full_static", TASK_B, "qwen3-8b", 100, experience_mode="full")
        self.write_run("outcome_hidden", TASK_B, "qwen3-8b", 100,
                       experience_mode="outcome_hidden", tree="ablation")
        runs = sorted(ex.group_runs(ex.discover_runs(
            [self.runs / "main", self.runs / "ablation"])).values(), key=lambda r: r.rel)
        self.assertFalse(any(ex.is_non_experience(r) for r in runs))
        choice = ex.choose_baseline_arm(runs, "full_static", [TASK_B], ["qwen3-8b"])
        self.assertIsNone(choice.arm)

    def test_arm_field_wins_over_file_stem(self):
        path = self.write_run("weird_stem", TASK_B, "qwen3-8b", 5, experience_mode="none",
                              tree="ablation", arm_field="no_experience")
        runs = ex.discover_runs([self.runs / "ablation"], models=["qwen3-8b"])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].arm, "no_experience")
        self.assertTrue(ex.is_non_experience(runs[0]))
        self.assertTrue(path.exists())


class TestTraceReading(ExportTestBase):
    def test_truncated_and_bad_lines_never_crash(self):
        path = self.write_run("full_static", TASK_B, "qwen3-8b", 5, experience_mode="full")
        with path.open("a", encoding="utf-8") as fh:
            fh.write("this is not json\n")           # terminated bad line -> skipped
            fh.write('{"sample_id": "gigaword/test/999999", "task"')  # mid-write tail
        recs, info = ex.load_traces(path)
        self.assertEqual(len(recs), 5, "the truncated tail and the bad line must be dropped")
        self.assertEqual(info.bad_lines, 1)
        self.assertTrue(info.truncated_tail)
        self.assertEqual(info.kept, 5)

    def test_drop_last_line_flag(self):
        path = self.write_run("full_static", TASK_B, "qwen3-8b", 5, experience_mode="full")
        recs, info = ex.load_traces(path, force_drop_last=True)
        self.assertEqual(len(recs), 4)
        self.assertTrue(info.dropped_tail)

    def test_empty_and_missing_files(self):
        missing = self.root / "nope.jsonl"
        recs, info = ex.load_traces(missing)
        self.assertEqual(recs, [])
        self.assertEqual(info.kept, 0)
        empty = self.root / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        recs, info = ex.load_traces(empty)
        self.assertEqual(recs, [])

    def test_completeness_rule_matches_aggregate(self):
        from core import TEST_SAMPLES

        p1 = self.write_run("full_static", TASK_B, "qwen3-8b", 100, experience_mode="full")
        p2 = self.write_run("full_static", TASK_A, "qwen3-8b", 999, experience_mode="full")
        runs = {r.task: r for r in ex.discover_runs([self.runs / "main"],
                                                    models=["qwen3-8b"])}
        self.assertTrue(runs[TASK_B].complete)
        self.assertEqual(runs[TASK_B].expected_n, TEST_SAMPLES[TASK_B])
        self.assertFalse(runs[TASK_A].complete, "999 < 1000 must count as partial")
        self.assertEqual(runs[TASK_A].expected_n, TEST_SAMPLES[TASK_A])
        self.assertTrue(p1.exists() and p2.exists())

    def test_tagged_traces_are_skipped_by_default(self):
        self.write_run("no_experience", TASK_B, "qwen3-8b", 5, experience_mode="none",
                       tree="ablation")
        tagged = (self.runs / "ablation" / "no_experience" / "seed42" / TASK_B
                  / "qwen3-8b" / "a0.5" / "no_experience.jsonl")
        tagged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.runs / "ablation" / "no_experience" / "seed42" / TASK_B
                    / "qwen3-8b" / "no_experience.jsonl", tagged)
        runs = ex.discover_runs([self.runs / "ablation"], models=["qwen3-8b"])
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].tag, "")
        runs = ex.discover_runs([self.runs / "ablation"], models=["qwen3-8b"],
                                allow_tagged=True)
        self.assertEqual(len(runs), 2)
        self.assertEqual({r.tag for r in runs}, {"", "a0.5"})

    def test_duplicate_traces_keep_the_fullest(self):
        short = self.write_run("full_static", TASK_B, "qwen3-8b", 5, experience_mode="full")
        # a second scan root holding the same (arm, tag, task, model, split, seed) key
        dup = (self.runs / "main_dup" / "full_static" / "seed42" / TASK_B / "qwen3-8b"
               / "full_static.jsonl")
        dup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(short, dup)
        with dup.open("a", encoding="utf-8") as fh:
            for i in range(5, 9):
                rec = json.loads(short.read_text(encoding="utf-8").splitlines()[0])
                rec["sample_id"] = f"{TASK_B}/test/{i:06d}"
                fh.write(json.dumps(rec) + "\n")
        runs = ex.discover_runs([self.runs / "main", self.runs / "main_dup"],
                                models=["qwen3-8b"])
        self.assertEqual(len(runs), 2)
        grouped = ex.group_runs(runs)
        self.assertEqual(len(grouped), 1)
        kept = list(grouped.values())[0]
        self.assertEqual(kept.n, 9, "the fuller duplicate must win")


class TestCliBehaviour(ExportTestBase):
    def test_dry_run_writes_nothing(self):
        self.seed_full_and_baseline()
        code, out = self.run_export(*self.base_argv("--limit", "10", "--dry-run"))
        self.assertEqual(code, 0, out)
        self.assertIn("[dry-run] no file written", out)
        self.assertFalse(self.out.exists(), "--dry-run must not create the output directory")

    def test_arms_flag_and_parser(self):
        self.seed_full_and_baseline()
        code, out = self.run_export(*self.base_argv("--arms", "full_static,no_experience",
                                                    "--limit", "4"))
        self.assertEqual(code, 0, out)
        key = self.read_key()
        self.assertEqual(key["baseline_arm"], "no_experience")
        with self.assertRaises(SystemExit) as cm:
            self.run_export(*self.base_argv("--arms", "full_static"))
        self.assertIn("--arms must be FULL,BASELINE", str(cm.exception))

    def test_limit_zero_or_negative_is_rejected(self):
        self.seed_full_and_baseline()
        with self.assertRaises(SystemExit):
            self.run_export(*self.base_argv("--limit", "0"))

    def test_partial_baseline_waiting_message_for_auto(self):
        self.write_run("full_static", TASK_B, "qwen3-8b", 100, experience_mode="full")
        code, out = self.run_export(*self.base_argv())
        self.assertEqual(code, 2)
        self.assertIn("sr_j_stop / sr_j_fixed / no_experience", out)


class TestPlotMetricConventions(ExportTestBase):
    """The Phase 9 unnecessary-refinement definition must match Phase 7's."""

    def test_unnecessary_refinement_matches_stopping_diagnostics(self):
        import make_plots as mp

        path = self.write_run("full_static", TASK_B, "qwen3-8b", 6, experience_mode="full")
        run = ex.discover_runs([self.runs / "main"], models=["qwen3-8b"])[0]
        stats = mp.unnecessary_refinement(run)
        # each sample contributes exactly one REFINE round (the second is STOP):
        # delta -1.0 for i % 3 == 0 else +1.0 over i in 0..5
        self.assertEqual(stats["n_refine"], 6)
        self.assertEqual(stats["scored"], 6)
        self.assertEqual(stats["worse"], 2)
        self.assertEqual(stats["better"], 4)
        self.assertEqual(stats["same"], 0)
        self.assertAlmostEqual(stats["rate"], 2 / 6)
        self.assertTrue(path.exists())

    def test_stop_rounds_are_not_counted(self):
        import make_plots as mp

        self.write_run("full_static", TASK_B, "qwen3-8b", 3, experience_mode="full")
        run = ex.discover_runs([self.runs / "main"], models=["qwen3-8b"])[0]
        for rec in run.records:
            rec["rounds"] = [_round(0, 0.0, action="STOP")]
        stats = mp.unnecessary_refinement(run)
        self.assertEqual(stats["n_refine"], 0)
        self.assertIsNone(stats["rate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
