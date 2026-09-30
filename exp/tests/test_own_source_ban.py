"""Gate: retrieval must never hand a sample the gold answer to its own sentence.

Why this exists
---------------
``RunConfig.ban_own_experience`` defaulted to True from the start but was never
implemented: nothing filtered the retrieved units, so the query item's own unit
was ranked #1 (identical source scores cosine 1.0) for 98.3% of rows.  That unit's
``state_after`` IS the item's reference, and the v2 renderer prints it as
``Refined Translation (Gold): ...`` -- so the refiner was shown the answer to the
sentence it was translating.

This is not a design choice to be debated: the original method skips exactly this
case (``if similarities[i][idx] > 0.99 and candidate_text == query_texts[i]:
continue``), and its retrieval DB and test file are literally the same csv.

The bug cost a full campaign's worth of invalid results, so it gets a test that
fails loudly if the exclusion is ever dropped again.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.bm25_fields import ExperienceRetriever  # noqa: E402
from core.experience import experience_dir, load_experiences, set_experience_root  # noqa: E402
from core.manifest import manifest_path, read_manifest  # noqa: E402

N = 60
_failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        _failures.append(name)


def main() -> int:
    set_experience_root("experience/contrastive")
    lib = load_experiences(experience_dir("qwen3-8b", "wmt19_en_zh") / "initial.jsonl")
    check("contrastive library loads", len(lib) > 100, f"n={len(lib)}")
    by_id = {e.exp_id: e for e in lib}
    retr = ExperienceRetriever(lib)

    test = list(read_manifest(manifest_path("wmt19_en_zh", "test")))[:N]

    # ---- the library really does contain the test items (that is WHY the ban
    # ---- is necessary; if this ever stops being true the test is vacuous)
    overlap = sum(1 for t in test if any(e.source_input == t.source for e in lib))
    check(
        "the library overlaps the test set (so the ban is load-bearing)",
        overlap > 0.5 * len(test),
        f"{overlap}/{len(test)}",
    )

    # ---- without the ban the own unit is retrieved; with it, never ----------
    own_without = own_with = 0
    short = 0
    for t in test:
        a = retr.retrieve(t.source, "", alpha=0.5, k=4)
        b = retr.retrieve(t.source, "", alpha=0.5, k=4, exclude_source=t.source)
        if any(by_id[i].source_input == t.source for i in a.exp_ids):
            own_without += 1
        if any(by_id[i].source_input == t.source for i in b.exp_ids):
            own_with += 1
        if len(b.exp_ids) < 4:
            short += 1

    check(
        "WITHOUT the ban the query's own unit is retrieved (the original defect)",
        own_without > 0.8 * len(test),
        f"{own_without}/{len(test)}",
    )
    check(
        "WITH the ban the query's own unit is never retrieved",
        own_with == 0,
        f"{own_with}/{len(test)}",
    )
    check(
        "the ban still fills the full budget of k=4",
        short == 0,
        f"{short} rows returned fewer than 4",
    )

    # ---- the excluded unit is exactly the one whose text is the answer ------
    t = test[0]
    own = [e for e in lib if e.source_input == t.source]
    check("a test item's own unit exists in the library", len(own) >= 1)
    if own:
        check(
            "that unit's state_after is byte-identical to the test reference",
            own[0].state_after == t.reference,
            "the leak this test guards against",
        )

    # ---- the retrieval MEMO must be keyed by retrieval semantics too --------
    # The first corrected campaign was silently invalid because cache_path()
    # ignored this knob: the fixed run read memos written by the contaminated
    # one and kept returning the query item's own unit (413/416 violations),
    # reinstating the very leak the flag prevents.
    from core.retrieval_cache import cache_path
    kept = cache_path("qwen3-8b", "wmt19_en_zh", "initial", "LIB", variant="ownsrc-kept")
    excl = cache_path("qwen3-8b", "wmt19_en_zh", "initial", "LIB", variant="ownsrc-excluded")
    check("the retrieval memo file differs between the two retrieval semantics",
          kept != excl, f"{kept.name} == {excl.name}")
    check("the memo file still changes with the library",
          cache_path("qwen3-8b", "wmt19_en_zh", "initial", "OTHER", variant="ownsrc-kept") != kept)

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print("own-source ban verified: the refiner can never be shown its own gold answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
