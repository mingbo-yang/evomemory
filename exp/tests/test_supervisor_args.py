"""Gate: supervisor's run_model call must not bind parameters positionally.

``run_model`` grew from 16 to 20 parameters.  The call site kept a positional
list while a new parameter (``draft_cache``) was inserted in the MIDDLE of the
signature, so every later argument shifted by one: ``n_candidates`` received the
empty string and the process died on ``int("")``.  All four priority-0 E2 jobs
were killed within 30 seconds of launch by this, and the dispatcher silently
moved on to lower-priority work -- the campaign looked alive while the primary
comparison was not running at all.

This test parses the source rather than importing it, so it needs no GPU, and it
fails the moment anyone reintroduces a long positional call.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
#: Positional parameters that are stable, call-site-first, and unlikely to move.
SAFE_POSITIONAL = 12

_failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        _failures.append(name)


def main() -> int:
    tree = ast.parse((ROOT / "supervisor.py").read_text(encoding="utf-8"))

    params = None
    call = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_model":
            params = [a.arg for a in node.args.args]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_model":
            call = node

    check("run_model is defined", params is not None)
    check("run_model is called", call is not None)
    if params is None or call is None:
        return 1

    check(
        f"run_model uses at most {SAFE_POSITIONAL} positional arguments",
        len(call.args) <= SAFE_POSITIONAL,
        f"got {len(call.args)} positional for params {params}",
    )

    passed_kw = {k.arg for k in call.keywords}
    missing = [p for p in params[SAFE_POSITIONAL:] if p not in passed_kw]
    check(
        "every parameter after the positional core is passed by keyword",
        not missing,
        f"missing: {missing}",
    )
    # The particular regression: n_candidates must never be positional.
    check("n_candidates is passed by keyword",
          "n_candidates" in passed_kw, "n_candidates bound positionally")

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print(f"supervisor argument binding is keyword-safe ({len(params)} params)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
