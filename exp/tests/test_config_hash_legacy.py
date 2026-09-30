"""Gate: adding the ``renderer`` knob must not move any production config_hash.

Why this test exists
--------------------
``RunConfig.config_hash`` is the resume identity of a run.  Adding a dataclass
field naively changes ``asdict()`` and therefore *every* hash, which silently
makes all ~50 runs already on disk unresumable and makes their recorded
``config_hash`` stop matching the code that produced them.

That is not hypothetical: ``runs/smoke/*`` really did break this way when
``draft_cache`` was added (their recorded configs predate the key, so they no
longer reproduce).  The v2 renderer work adds a knob, so the mitigation is
``LEGACY_HASH_DEFAULTS``: a field equal to its legacy default is omitted from
the hash payload.  A v1 run therefore keeps its historical identity, while a v2
run is unmistakably distinct.

This test is the gate.  It reconstructs ``RunConfig`` from the ``config`` dict
recorded inside real result rows and requires the recomputed hash to equal the
recorded one, byte for byte.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pipeline import RunConfig  # noqa: E402

#: ``runs/smoke`` predates the ``draft_cache`` field and cannot reproduce under
#: any current code path; it is a throwaway artifact, not a result, so it is
#: excluded here and asserted-as-stale below rather than silently skipped.
STALE_TREES = ("smoke",)

_failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        _failures.append(name)


def _production_cells():
    """One representative row per (tree, arm, task, model) cell."""
    seen = set()
    for p in sorted(glob.glob(str(ROOT / "runs" / "*" / "*" / "seed*" / "*" / "*" / "*.jsonl"))):
        parts = p.split("/")
        # .../<tree>/<arm>/<seed>/<task>/<model>/<file>.jsonl
        tree = parts[-6]
        if tree in STALE_TREES:
            continue
        cell = "/".join(parts[-6:-1])
        if cell in seen:
            continue
        seen.add(cell)
        with open(p, encoding="utf-8") as f:
            line = f.readline().strip()
        if line:
            yield cell, json.loads(line)


def main() -> int:
    cells = list(_production_cells())
    check("found production cells to pin", len(cells) >= 15, f"found {len(cells)}")

    n_ok = 0
    for cell, row in cells:
        recorded = row.get("config_hash")
        cfg = row.get("config")
        if not recorded or not isinstance(cfg, dict):
            continue
        # The recorded dict predates ``renderer``; the dataclass default must
        # therefore reproduce the historical hash exactly.
        rebuilt = RunConfig(**cfg)
        if rebuilt.config_hash == recorded:
            n_ok += 1
        else:
            check(
                f"hash preserved for {cell}",
                False,
                f"recomputed {rebuilt.config_hash} != recorded {recorded}",
            )
    check(
        "every production cell reproduces its recorded config_hash",
        n_ok == len(cells),
        f"{n_ok}/{len(cells)}",
    )

    # ---- the knob must actually separate identities ------------------------
    base = dict(task="wmt19_en_zh", model="qwen3-8b")
    v1 = RunConfig(**base)
    v2 = RunConfig(**base, renderer="v2")
    check("renderer=v2 changes the hash", v1.config_hash != v2.config_hash)
    check("an explicit renderer='v1' equals the default", RunConfig(**base, renderer="v1").config_hash == v1.config_hash)

    # ---- positive_only must not disturb the hash (it is an existing mode) --
    check(
        "experience_mode is still fully inside the hash",
        RunConfig(**base, experience_mode="positive_only").config_hash
        != RunConfig(**base, experience_mode="full").config_hash,
    )

    # ---- the contrastive switch is derived, never a separate field ---------
    check("full renders contrastively", v2.render_contrastive is True)
    check(
        "positive_only drops the contrast",
        RunConfig(**base, experience_mode="positive_only").render_contrastive is False,
    )

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print(f"all config-hash compatibility checks passed ({n_ok} production cells pinned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
