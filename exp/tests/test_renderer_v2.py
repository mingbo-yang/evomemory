"""Gate: the v2 experience block must match the ORIGINAL method byte for byte.

The user's requirement is "the experience format must be the same as the
original".  A test that compares my renderer against itself would prove nothing,
so this file **re-transcribes** the original implementation verbatim from

    /mnt/huawei/ymb/icml/exp/generation/en_zh/llm_re_cosine_qwen8b.py

(``extract_summary_advice`` and the ``construct_icl_prompt`` example loop) and
compares it against ``core.experience.render_experience`` on real asset rows.

Three properties are pinned:

1. Field order is Source -> Draft -> **Critique & Advice** -> Refined (Gold).
   Note the advice sits *before* the gold answer in the prompt, even though the
   asset CSV orders its columns ``原文|不完美答案|完美答案|修改建议`` (advice last).
2. The advice is the LAST paragraph only, not the full ~2.4k-char critique.
3. No outcome / judge-note field is rendered: the contrastive library stores
   wrong->right pairs only, so every unit is positive by construction.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.bm25_fields import Experience  # noqa: E402
from core.experience import render_experience, summary_advice  # noqa: E402

ASSET = Path("/mnt/huawei/wwq/model/aaa_experiment/code/wmt19_dataset_analysis_top10.csv")
N_ROWS = 200

_failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        _failures.append(name)


def _mk(source, before, after, advice, task="wmt19_en_zh", helped=False):
    """Build a library unit the way build_contrastive_library.py does."""
    return Experience(
        exp_id="t", task=task, model="m",
        source_input=source, state_before=before, state_after=after,
        intervention_instruction=advice, intervention_rationale="",
        verdict="better" if helped else "tie", reason_a="", reason_b="",
        order_consistent=True, delta_offline=None,
        provenance="contrastive-asset",
        outcome_label="helped" if helped else "",
    )


# --- verbatim transcription of the ORIGINAL implementation ------------------ #
def original_extract_summary_advice(advice_text):
    """Copied from llm_re_cosine_qwen8b.py, unmodified."""
    if not isinstance(advice_text, str):
        return ""
    paragraphs = advice_text.strip().split('\n\n')
    if paragraphs:
        return paragraphs[-1].strip()
    return advice_text


def original_example_block(row, i):
    """Copied from construct_icl_prompt's example loop, unmodified."""
    refined_advice = original_extract_summary_advice(row['修改建议'])
    return (
        f"### Example {i+1}\n"
        f"Source: {row['原文']}\n"
        f"Draft Translation: {row['不完美答案']}\n"
        f"Critique & Advice: {refined_advice}\n"
        f"Refined Translation (Gold): {row['完美答案']}\n\n"
    )
# --------------------------------------------------------------------------- #


def main() -> int:
    check("original asset is readable", ASSET.is_file(), str(ASSET))
    if not ASSET.is_file():
        return 1

    with ASSET.open(encoding="utf-8", newline="") as f:
        rows = []
        for i, row in enumerate(csv.DictReader(f)):
            if i >= N_ROWS:
                break
            rows.append(row)
    check("read asset rows", len(rows) == N_ROWS, f"got {len(rows)}")

    # ---- 1. byte-identical against the original -----------------------------
    n_match = 0
    mismatches = []
    for i, row in enumerate(rows):
        exp = _mk(row["原文"], row["不完美答案"], row["完美答案"], row["修改建议"], helped=True)
        mine = render_experience(exp, index=i + 1, contrastive=True)
        # the original appends a trailing blank line after each example
        theirs = original_example_block(row, i).rstrip("\n")
        if mine == theirs:
            n_match += 1
        elif len(mismatches) < 2:
            mismatches.append((i, mine[:200], theirs[:200]))
    check(
        f"v2 example block is byte-identical to the original on {N_ROWS} real rows",
        n_match == len(rows),
        f"{n_match}/{len(rows)}; first mismatch: {mismatches[:1]}",
    )

    # ---- 2. field order, including the advice-before-gold subtlety ----------
    sample = rows[0]
    exp = _mk(sample["原文"], sample["不完美答案"], sample["完美答案"], sample["修改建议"])
    block = render_experience(exp, index=1, contrastive=True)
    order = [
        block.index("Source:"),
        block.index("Draft Translation:"),
        block.index("Critique & Advice:"),
        block.index("Refined Translation (Gold):"),
    ]
    check("field order is Source -> Draft -> Advice -> Refined (Gold)",
          order == sorted(order), f"offsets {order}")
    check("advice precedes the gold answer (as in the original prompt)",
          block.index("Critique & Advice:") < block.index("Refined Translation (Gold):"))

    # ---- 3. the advice is the LAST paragraph only --------------------------
    full_len = sum(len(r["修改建议"]) for r in rows) / len(rows)
    kept_len = sum(len(summary_advice(r["修改建议"])) for r in rows) / len(rows)
    check("advice is materially truncated (last paragraph only)",
          kept_len < 0.35 * full_len, f"kept {kept_len:.0f} of {full_len:.0f} chars")
    check("summary_advice reproduces the original on all rows",
          all(summary_advice(r["修改建议"]) == original_extract_summary_advice(r["修改建议"])
              for r in rows))
    check("summary_advice is idempotent",
          all(summary_advice(summary_advice(r["修改建议"])) == summary_advice(r["修改建议"])
              for r in rows))

    # ---- 4. no outcome field survives into the prompt -----------------------
    for phrase in ("Outcome:", "Judge note", "HELPED", "HURT", "Why it was tried"):
        check(f"v2 block contains no {phrase!r}", phrase not in block)

    # ---- 5. PositiveOnly differs by exactly the draft line ------------------
    pos = render_experience(exp, index=1, contrastive=False)
    check("PositiveOnly drops the draft line",
          "Draft Translation:" not in pos and "Source:" in pos
          and "Critique & Advice:" in pos and "Refined Translation (Gold):" in pos)
    check("PositiveOnly differs from full by the draft line only",
          len(block.splitlines()) == len(pos.splitlines()) + 1)
    check("PositiveOnly is the original minus the draft -> matches the original's "
          "own fields for the retained lines",
          original_extract_summary_advice(sample["修改建议"]) in pos)

    # ---- 6. non-translation tasks keep the same shape ----------------------
    for task, draft_label, final_label in (
        ("coedit_gec", "Draft Correction", "Corrected Text (Gold)"),
        ("gigaword", "Draft Summary", "Reference Summary (Gold)"),
    ):
        e = _mk("S", "BAD", "GOOD", "a\n\nb", task=task)
        b = render_experience(e, index=2, contrastive=True)
        check(
            f"{task} uses the same four-line shape with task-appropriate labels",
            b.startswith(f"### Example 2\nSource: S\n{draft_label}: BAD\n")
            and f"Critique & Advice: b\n{final_label}: GOOD" in b,
            b,
        )

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print("all v2 renderer-alignment checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
