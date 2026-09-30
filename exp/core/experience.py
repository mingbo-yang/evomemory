"""Experience store, snapshots, and the leakage-guarded prompt renderer.

The renderer is deliberately a **whitelist**: the prompt is assembled field by
field and never by serialising the whole experience object.  ``delta_offline``
(the gold-metric delta kept for offline analysis) and any reference text are
therefore structurally incapable of reaching a controller / judge / refiner
prompt, and a unit test asserts this on real rendered snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from . import EXP_ROOT
from .bm25_fields import Experience

#: Fields that may ever be shown to a model.  Anything not listed here is
#: evaluation-only and must not be rendered.
RENDERABLE_FIELDS = (
    "source_input",
    "state_before",
    "intervention_instruction",
    "intervention_rationale",
    "state_after",
    "verdict",
    "reason_a",
    "reason_b",
    # A categorical label derived from the GOLD metric of an *auxiliary*
    # experience (never of the query item).  Adding it is a deliberate method
    # decision: the verdict-only outcome was measured to be mostly noise.
    "outcome_label",
)

#: Fields that exist only for offline evaluation / auditing.
OFFLINE_ONLY_FIELDS = ("delta_offline", "provenance", "exp_id", "model")


#: Optional override of the experience-library root.  Rebuilding the library
#: with trustworthy outcome labels must not disturb the libraries that back the
#: runs already on disk, so the new one lives beside them and runs opt in.
_EXPERIENCE_ROOT: Optional[Path] = None


def set_experience_root(path) -> None:
    global _EXPERIENCE_ROOT
    _EXPERIENCE_ROOT = Path(path) if path else None


def experience_dir(model: str, task: str) -> Path:
    base = _EXPERIENCE_ROOT if _EXPERIENCE_ROOT is not None else EXP_ROOT / "experience" / "initial"
    return base / model / task


def snapshot_dir(model: str, task: str) -> Path:
    return EXP_ROOT / "experience" / "snapshots" / model / task


def load_experiences(path: Path) -> List[Experience]:
    out: List[Experience] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Experience.from_dict(json.loads(line)))
    return out


def save_experiences(exps: Iterable[Experience], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in exps:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


# --- Filtering (ablation manipulations) ------------------------------------


def filter_positive_only(exps: Sequence[Experience]) -> List[Experience]:
    """``PositiveOnly`` arm: keep only transitions the double-order judge accepted."""
    return [e for e in exps if e.verdict == "better" and e.order_consistent]


def filter_gold_positive(exps: Sequence[Experience]) -> List[Experience]:
    """Keep only units whose GOLD-metric outcome was positive.

    This is NOT the same as ``filter_positive_only``, which keeps units the
    double-order judge called "better" -- and that judge is the same model that
    wrote the revision, agreeing with the gold outcome only 68.6% of the time.

    Why this filter exists: measured on the Phase-4 traces, 79% of refinement
    rounds retrieved 3 or 4 examples whose gold outcome was hurt/unchanged, only
    0.5% retrieved an all-positive set, and 41% of the instructions the model then
    produced were near-copies of a retrieved example's instruction.  The library
    was therefore teaching the model, by demonstration, the very micro-edits that
    had been measured to fail.  Keeping only gold-positive units turns the block
    from counter-examples into demonstrations of what actually worked.
    """
    return [e for e in exps if getattr(e, "outcome_label", "") == "helped"]


def class_profile(result) -> Dict[str, int]:
    return dict(result.per_class_counts)


# --- Rendering --------------------------------------------------------------

#: Per-task wording for the four-line contrastive example block.  The two
#: translation tasks reproduce the original method's labels verbatim; the other
#: two keep the identical shape with task-appropriate nouns.  ``Critique &
#: Advice`` is the same label on all four tasks, as in the original.
EXAMPLE_LABELS: Dict[str, Dict[str, str]] = {
    "wmt19_en_zh": {"draft": "Draft Translation", "final": "Refined Translation (Gold)"},
    "wmt19_zh_en": {"draft": "Draft Translation", "final": "Refined Translation (Gold)"},
    "coedit_gec": {"draft": "Draft Correction", "final": "Corrected Text (Gold)"},
    "gigaword": {"draft": "Draft Summary", "final": "Reference Summary (Gold)"},
}

_DEFAULT_LABELS = {"draft": "Draft Translation", "final": "Refined Translation (Gold)"}


def summary_advice(text: str) -> str:
    """Reproduce the original method's ``extract_summary_advice``.

    The asset's advice column is a long critique (measured mean 2,398 chars on
    the contrastive library) whose **last paragraph** is the generalised
    takeaway (mean 360 chars, i.e. 15%).  The original pipeline injected only
    that paragraph, and the reason is methodological rather than budgetary: what
    transfers to a *new* sentence is the generalised lesson, not the critique of
    the one sentence it was written about.  Reproducing this exactly is part of
    aligning with the original method, not an optimisation.
    """
    if not isinstance(text, str):
        return ""
    paragraphs = [p for p in text.strip().split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs[-1].strip()
    return text.strip()


def render_experience(
    exp: Experience,
    include_outcome: bool = True,
    index: int = 1,
    contrastive: bool = True,
    advice_mode: str = "summary",
) -> str:
    """Render ONE experience unit as the original method's four-line example.

    The library stores only genuine wrong->right pairs, so shipping an outcome
    field would be pointless: it would read ``helped`` on every unit.  The
    contrast is *inside* the unit instead -- the wrong draft sits directly above
    the gold correction::

        ### Example 1
        Source: <task input>
        Draft Translation: <the real error>
        Critique & Advice: <generalised takeaway>
        Refined Translation (Gold): <the gold correction>

    ``contrastive=False`` is the ``PositiveOnly`` ablation.  It drops the draft
    line and keeps source / advice / gold answer; that single line is the whole
    difference between the two arms.  The retrieved ids are identical in both,
    so the ablation isolates the contrast and nothing else.

    ``include_outcome`` is retained for call-site compatibility only; the v2
    renderer has no outcome field to include or omit.
    """
    labels = EXAMPLE_LABELS.get(exp.task, _DEFAULT_LABELS)
    lines = [f"### Example {index}"]
    lines.append(f"Source: {exp.source_input}")
    if contrastive:
        lines.append(f"{labels['draft']}: {exp.state_before}")
    # "summary" reproduces the original method (last paragraph only).  "full"
    # keeps the whole critique and exists because the two-stage controller ->
    # refiner architecture may not carry a 360-char generalised takeaway far
    # enough to guide a concrete fix, whereas the v1 renderer fed the controller
    # the entire ~2,400-char critique.
    raw = exp.intervention_instruction
    advice = summary_advice(raw) if advice_mode == "summary" else (raw or "").strip()
    lines.append(f"Critique & Advice: {advice or '(no advice given)'}")
    lines.append(f"{labels['final']}: {exp.state_after}")
    return "\n".join(lines)


def render_experience_block(
    exps: Sequence[Experience],
    include_outcome: bool = True,
    token_budget: int = 2048,
    count_tokens: Optional[Callable[[str], int]] = None,
    max_units: int = 4,
    contrastive: bool = True,
    advice_mode: str = "summary",
) -> str:
    """Pack whole experience units up to ``token_budget``.

    Truncation is always at unit boundaries -- a half experience is worse than
    one fewer experience.  After the v2 alignment the four examples measure
    ~2,776 chars (~1,633 tokens) on the contrastive library, so ``max_units``
    binds well before the token budget; the budget stays as a guard for tasks
    with longer text.  ``contrastive=False`` renders the ``PositiveOnly``
    ablation (draft line omitted).
    """
    if count_tokens is None:
        count_tokens = _approx_tokens

    chosen: List[str] = []
    used = 0
    for i, e in enumerate(exps[:max_units], start=1):
        block = render_experience(
            e, include_outcome=include_outcome, index=i, contrastive=contrastive,
            advice_mode=advice_mode,
        )
        n = count_tokens(block)
        if chosen and used + n > token_budget:
            break
        if not chosen and n > token_budget:
            break
        chosen.append(block)
        used += n
    return "\n\n".join(chosen)


def _approx_tokens(text: str) -> int:
    """Fallback counter (only used when no tokenizer is supplied)."""
    import re

    cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    non_cjk_words = len(re.findall(r"[A-Za-z0-9]+", text))
    return cjk + int(non_cjk_words * 1.3) + 8


# --- Leakage guard ----------------------------------------------------------


def assert_no_leakage(rendered: str, forbidden: Sequence[str]) -> None:
    """Raise if any forbidden string survives into a rendered prompt."""
    for token in forbidden:
        if token and token in rendered:
            raise AssertionError(
                f"leakage: forbidden content reached a prompt: {token[:80]!r}"
            )
