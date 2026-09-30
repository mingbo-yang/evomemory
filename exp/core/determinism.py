"""Deterministic derivation of every per-call random choice.

Plan v5 section 4.3.  If each arm drew its own A/B ordering and its own sampling
seeds from a shared global RNG, the ordering would silently become a *second*
variable alongside whatever the ablation actually manipulates.  Every random
choice is therefore a pure function of the sample identity and the round, so
that all arms see byte-identical inputs whenever they reach the same state.
"""

from __future__ import annotations

import hashlib
from typing import Tuple


def _digest(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def ab_order(model: str, task: str, sample_id: str, round_index: int) -> bool:
    """Whether the *candidate* occupies slot A.  Identical across arms."""
    return bool(_digest("ab", model, task, sample_id, round_index) & 1)


def call_seed(sample_id: str, round_index: int, call_type: str) -> int:
    """Stable seed for one LLM call.  Identical across arms."""
    return _digest("seed", sample_id, round_index, call_type) % (2**31 - 1)


def sample_seed(sample_id: str, base_seed: int) -> int:
    """Stable per-sample seed for the initial draft."""
    return (base_seed + _digest("draft", sample_id) % 9973) % (2**31 - 1)


def randomised_ab(model: str, task: str, sample_id: str, round_index: int,
                  current: str, candidate: str) -> Tuple[str, str, bool]:
    """Return ``(slot_a_text, slot_b_text, candidate_is_a)`` for the judge."""
    candidate_is_a = ab_order(model, task, sample_id, round_index)
    return (candidate, current, True) if candidate_is_a else (current, candidate, False)
