"""Paired bootstrap and order-variance reporting.

Plan v5 sections 8 and 9:

* Main results use 2000 paired bootstrap resamples of the *per-sample* metric
  difference, which is the right unit because every arm shares the same ``y0``
  and the same test items.
* Experience accumulation additionally reports the spread across auxiliary
  stream orders.  A sample-level bootstrap cannot see order sensitivity, so the
  mean/std/individual results across orders are reported separately.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_two_sided: float

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "diff": self.diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_two_sided": self.p_two_sided,
        }


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedResult:
    """Bootstrap the mean of ``a - b`` over paired samples.

    ``p_two_sided`` is the share of resampled means on the opposite side of zero
    from the observed difference, doubled -- i.e. a bootstrap analogue of a
    two-sided paired test, not a parametric p-value.
    """
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be the same length: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return PairedResult(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    diffs = [x - y for x, y in zip(a, b)]
    obs = sum(diffs) / n
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[max(0, int((alpha / 2) * n_resamples) - 1)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    if obs >= 0:
        p = 2.0 * (sum(1 for m in means if m <= 0) / n_resamples)
    else:
        p = 2.0 * (sum(1 for m in means if m >= 0) / n_resamples)
    return PairedResult(
        n=n,
        mean_a=sum(a) / n,
        mean_b=sum(b) / n,
        diff=obs,
        ci_low=lo,
        ci_high=hi,
        p_two_sided=min(1.0, p),
    )


def order_variance(per_order: Dict[str, Sequence[float]]) -> dict:
    """Summarise an accumulation curve across auxiliary-stream orders."""
    means = {k: (sum(v) / len(v) if v else float("nan")) for k, v in per_order.items()}
    vals = [m for m in means.values() if m == m]
    return {
        "per_order_mean": means,
        "mean": sum(vals) / len(vals) if vals else float("nan"),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "n_orders": len(vals),
    }


def holm_correct(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjustment, for the family of ablation comparisons."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running = max(running, adj)
        out[idx] = min(1.0, running)
    return out
