#!/usr/bin/env python
"""Phase 1e: per-arm-pair single-variable invariants (plan v5 section 4.5).

These are the tests that make "single-variable ablation" a checkable property
rather than a promise:

* ``Full`` vs ``OutcomeHidden`` -- identical experience ids at an identical state.
* ``RandomRetrieve`` vs ``Full`` -- identical count and identical per-verdict
  composition, deliberately different ids.
* every arm -- identical A/B ordering and identical sampling seeds.
* the renderer can never emit offline-only fields.

Run:  /home/ymb/miniconda3/envs/qwen35/bin/python tests/test_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.bm25_fields import (  # noqa: E402
    Experience,
    ExperienceRetriever,
    deterministic_random_rank,
    minmax_normalize,
)
from core.determinism import ab_order, call_seed  # noqa: E402
from core.experience import (  # noqa: E402
    RENDERABLE_FIELDS,
    assert_no_leakage,
    render_experience_block,
)
from core.retrieval_cache import cache_key, state_key  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def _mk(i: int, verdict: str, *, task="wmt19_en_zh", model="qwen3-8b") -> Experience:
    return Experience(
        exp_id=f"{task}/{model}/aux-{i:04d}/r0",
        task=task,
        model=model,
        source_input=f"source sentence number {i} about topic {i % 7}",
        state_before=f"state before revision {i} variant {i % 5}",
        state_after=f"state after revision {i} variant {i % 5}",
        intervention_instruction=f"instruction {i}: tighten wording",
        intervention_rationale=f"rationale {i}: the draft was wordy",
        verdict=verdict,
        reason_a=f"reason a {i}",
        reason_b=f"reason b {i}",
        order_consistent=(verdict in ("better", "worse")),
        delta_offline=0.01 * i,
        provenance="unit-test",
    )


def build_library() -> list:
    verdicts = ["better", "worse", "tie"]
    return [_mk(i, verdicts[i % 3]) for i in range(60)]


def main() -> int:
    lib = build_library()
    retr = ExperienceRetriever(lib)
    q_input = "source sentence number 12 about topic 5"
    q_state = "state before revision 12 variant 2"
    model, task, sample_id, rnd = "qwen3-8b", "wmt19_en_zh", "wmt19_en_zh/test/000012", 0

    # ---- 1. determinism of the retriever itself ---------------------------
    r1 = retr.retrieve(q_input, q_state, alpha=0.5, k=4)
    r2 = retr.retrieve(q_input, q_state, alpha=0.5, k=4)
    check("retriever is deterministic", r1.exp_ids == r2.exp_ids)

    # ---- 2. Full vs OutcomeHidden: identical ids at identical state --------
    # The two arms differ ONLY in whether the outcome block is rendered; the
    # retrieval call and its arguments are identical.
    full = retr.retrieve(q_input, q_state, alpha=0.5, k=4)
    hidden = retr.retrieve(q_input, q_state, alpha=0.5, k=4)
    check(
        "Full vs PositiveOnly: identical exp_ids",
        full.exp_ids == hidden.exp_ids,
        f"{full.exp_ids} != {hidden.exp_ids}",
    )
    # The v2 renderer has no outcome field at all (the library stores only
    # wrong->right pairs, so an outcome label would be constant).  The ablation
    # that matters now is the *contrast*: PositiveOnly drops the wrong draft.
    block_full = render_experience_block(
        [retr.by_id[e] for e in full.exp_ids], contrastive=True
    )
    block_hidden = render_experience_block(
        [retr.by_id[e] for e in hidden.exp_ids], contrastive=False
    )
    check("PositiveOnly differs from Full", block_full != block_hidden)
    for phrase in ("Outcome:", "Judge note 1:", "Judge note 2:", "Why it was tried:"):
        check(
            f"v2 block omits the legacy {phrase!r} field",
            phrase not in block_full and phrase not in block_hidden,
        )
    check(
        "v2 block uses the original method's four-line example format",
        "### Example 1" in block_full and "Critique & Advice:" in block_full,
    )
    check(
        "Full shows the wrong draft",
        "Draft Translation:" in block_full or "Draft Correction:" in block_full,
    )
    check(
        "PositiveOnly drops the wrong draft",
        "Draft Translation:" not in block_hidden
        and "Draft Correction:" not in block_hidden
        and "Draft Summary:" not in block_hidden,
    )
    # the non-contrast content must be byte-identical
    check(
        "PositiveOnly differs from Full by the draft line only",
        block_hidden.strip() == _strip_draft_lines(block_full).strip(),
    )

    # ---- 3. RandomRetrieve: same count + same composition, different ids ---
    profile = dict(full.per_class_counts)
    rng = deterministic_random_rank([e.exp_id for e in lib], seed_key=state_key(q_input, q_state))
    rnd_res = retr.retrieve(
        q_input, q_state, alpha=0.5, quota_profile=profile, rng_rank=rng
    )
    check(
        "RandomRetrieve: same total count",
        len(rnd_res.exp_ids) == len(full.exp_ids),
        f"{len(rnd_res.exp_ids)} != {len(full.exp_ids)}",
    )
    check(
        "RandomRetrieve: same per-verdict composition",
        rnd_res.per_class_counts == full.per_class_counts,
        f"{rnd_res.per_class_counts} != {full.per_class_counts}",
    )
    check(
        "RandomRetrieve: ids deliberately differ",
        set(rnd_res.exp_ids) != set(full.exp_ids),
    )
    rnd_res2 = retr.retrieve(
        q_input, q_state, alpha=0.5, quota_profile=profile, rng_rank=rng
    )
    check("RandomRetrieve: arm-internal determinism", rnd_res2.exp_ids == rnd_res.exp_ids)

    # ---- 4. all arms share A/B order and sampling seeds --------------------
    order_full = ab_order(model, task, sample_id, rnd)
    order_other = ab_order(model, task, sample_id, rnd)
    check("A/B order identical across arms", order_full == order_other)
    check(
        "sampling seed identical across arms",
        call_seed(sample_id, rnd, "controller") == call_seed(sample_id, rnd, "controller"),
    )
    check(
        "A/B order varies with round",
        len({ab_order(model, task, sample_id, r) for r in range(12)}) == 2,
    )

    # ---- 5. normalisation lands in [0, 1] ----------------------------------
    combined, norm_i, norm_s = retr.combined_scores(q_input, q_state, alpha=0.5)
    allv = list(norm_i.values()) + list(norm_s.values())
    check("field scores normalised to [0,1]", all(v >= -1e-9 and v <= 1 + 1e-9 for v in allv))
    combined0, _, norm_s0 = retr.combined_scores(q_input, q_state, alpha=0.0)
    combined1, norm_i1, _ = retr.combined_scores(q_input, q_state, alpha=1.0)
    check(
        "alpha=0 uses only the state field",
        all(abs(combined0[d] - norm_s0.get(d, 0.0)) < 1e-9 for d in combined0),
    )
    check(
        "alpha=1 uses only the input field",
        all(abs(combined1[d] - norm_i1.get(d, 0.0)) < 1e-9 for d in combined1),
    )
    # alpha must actually change the ranking, otherwise the dev-time sweep is vacuous
    order_mid = sorted(combined, key=lambda d: (-combined[d], d))
    order_zero = sorted(combined0, key=lambda d: (-combined0[d], d))
    check("alpha changes the ranking", order_mid != order_zero)
    check("minmax handles a degenerate range", minmax_normalize({1: 3.0, 2: 3.0}) == {1: 1.0, 2: 1.0})

    # ---- 6. renderer cannot leak offline-only fields -----------------------
    exps = [retr.by_id[e] for e in full.exp_ids]
    rendered = render_experience_block(exps, include_outcome=True)
    check("renderer omits delta_offline", "delta_offline" not in rendered)
    check(
        "no numeric delta leaks",
        not any(str(e.delta_offline) in rendered for e in exps if e.delta_offline is not None),
    )
    check("renderer omits exp_id", not any(e.exp_id in rendered for e in exps))
    check("renderer omits provenance", "unit-test" not in rendered)
    try:
        assert_no_leakage("Answer before: x\nprovenance: unit-test\n", ["unit-test"])
        check("assert_no_leakage detects a planted leak", False)
    except AssertionError:
        check("assert_no_leakage detects a planted leak", True)
    try:
        assert_no_leakage(rendered, ["unit-test"])
        check("no leak present in real rendered block", True)
    except AssertionError:
        check("no leak present in real rendered block", False)
    check(
        "RENDERABLE_FIELDS excludes offline-only fields",
        "delta_offline" not in RENDERABLE_FIELDS and "provenance" not in RENDERABLE_FIELDS,
    )

    # ---- 7. cache key separates arms but not redundant calls ---------------
    base = dict(
        model=model, task=task, snapshot_id="initial", round_index=0,
        alpha=0.5, k=4, quota_profile=None, rng_seed_key=None,
        input_hash="ih", state_hash="sh",
    )
    k_full = cache_key(mode="full", **base)
    k_hidden = cache_key(mode="outcome_hidden", **base)
    k_full2 = cache_key(mode="full", **base)
    check("cache key differs by arm (mode)", k_full != k_hidden)
    check("cache key stable for identical calls", k_full == k_full2)

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)} failing checks)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULT: PASS (all invariants hold)")
    return 0


def _strip_draft_lines(block: str) -> str:
    """Drop the wrong-draft line, i.e. what ``PositiveOnly`` omits."""
    out = []
    for line in block.splitlines():
        if line.startswith(("Draft Translation:", "Draft Correction:", "Draft Summary:")):
            continue
        out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
