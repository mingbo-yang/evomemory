"""Persistent retrieval memoisation.

Purpose (v5 plan section 4.2).  Retrieval depends on ``state_before``, which
depends on the previous round's decision, which the ablation may have changed.
Caching on ``(input_hash, state_hash)`` gives two things:

1. **Arm-internal determinism** -- revisiting the same state returns the same
   ids, so a run is reproducible even if the library grew meanwhile.
2. **A testable cross-arm invariant** -- for ``Full`` vs ``OutcomeHidden`` the
   trajectories coincide until the first divergence, so at every shared state
   the two arms must receive byte-identical id lists.  ``RandomRetrieve``
   deliberately differs, and is covered by a *different* assertion (same count,
   same per-class composition, different ids).

The cache key includes every parameter that can change the result, including
the arm-specific ``mode`` and the random-ranking seed, so it can never
accidentally paper over a genuine difference between arms.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from . import EXP_ROOT
from .manifest import sha256_text


def state_key(input_text: str, state_before: str) -> str:
    """Stable identity of a control point, independent of arm."""
    return sha256_text(f"{sha256_text(input_text)}::{sha256_text(state_before)}")


def cache_key(
    *,
    model: str,
    task: str,
    snapshot_id: str,
    round_index: int,
    mode: str,
    alpha: float,
    k: int,
    quota_profile: Optional[Dict[str, int]],
    rng_seed_key: Optional[str],
    input_hash: str,
    state_hash: str,
    own_source_excluded: bool = False,
) -> str:
    # ``own_source_excluded`` is part of the key, not merely of the memo path:
    # two runs over one library that retrieve different ids must never share a
    # memo entry.  Leaving it out once already let the corrected campaign replay
    # the contaminated campaign's retrievals (413/416 rows still got the query
    # item's own unit) -- the flag looked enforced while the cache bypassed it.
    payload = json.dumps(
        {
            "model": model,
            "own_source_excluded": bool(own_source_excluded),
            "task": task,
            "snapshot_id": snapshot_id,
            "round": round_index,
            "mode": mode,
            "alpha": round(float(alpha), 6),
            "k": k,
            "quota_profile": quota_profile,
            "rng_seed_key": rng_seed_key,
            "input_hash": input_hash,
            "state_hash": state_hash,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RetrievalCache:
    """JSONL-backed memo of retrieval results, keyed by the full parameter set."""

    def __init__(self, path: Path):
        self.path = path
        self._mem: Dict[str, dict] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec

    def get(self, key: str) -> Optional[dict]:
        return self._mem.get(key)

    def put(self, key: str, value: dict) -> None:
        if key in self._mem:
            return
        rec = {"key": key, **value}
        self._mem[key] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._mem)


def cache_path(model: str, task: str, snapshot_id: str, library_key: str = "",
               variant: str = "") -> Path:
    """Memo file for one (model, task, snapshot_id, LIBRARY, VARIANT) combination.

    ``library_key`` identifies the experience library the memo belongs to.  It
    was missing, so a library swap reused memos whose ``exp_ids`` referred to the
    *previous* library's units; every cache hit then raised ``KeyError`` where the
    caller looks the id up, killing the engine ~18 s after load.  That is exactly
    what happened when the contrastive library replaced the generated one:
    ``KeyError: 'wmt19_en_zh/qwen3-4b/wmt19_en_zh/initial/000009/b1'``.  Relabelling
    (initial -> initial_gold) escaped only because it kept the old id scheme.

    ``variant`` distinguishes retrieval *semantics* over the SAME library.  It was
    missing when ``retrieval_excludes_own_source`` landed, so the corrected run
    read memos written by the contaminated one and went on retrieving the query
    item's own unit -- silently reinstating the exact leak the flag exists to
    prevent.  Any knob that changes WHICH ids come back belongs here.
    """
    tag = ""
    if library_key:
        tag = "." + hashlib.sha256(library_key.encode("utf-8")).hexdigest()[:12]
    if variant:
        tag += "." + hashlib.sha256(variant.encode("utf-8")).hexdigest()[:8]
    return (
        EXP_ROOT
        / "experience"
        / "indexes"
        / model
        / task
        / f"{snapshot_id}{tag}.retrieval_cache.jsonl"
    )


def library_fingerprint(library) -> str:
    """Stable identity of a library: unit count + hash of the ordered exp_ids.

    Cheap, and it changes whenever the unit set or its ids change -- which is the
    only thing a retrieval memo depends on.
    """
    ids = [str(getattr(e, "exp_id", "")) for e in library]
    h = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]
    return f"n{len(ids)}-{h}"
