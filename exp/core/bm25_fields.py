"""Lexical experience retrieval: field-wise BM25 + per-query normalisation.

Design (v5 plan section 3, points R1-R4, R6, R7):

* **No encoder model.**  Retrieval is BM25 over two *fields* -- the task input
  and the current answer (``state_before``).
* **Field-wise tokenisation.**  The two fields are not the same language: for
  ``zh_en`` the input is Chinese while the state is English, and vice versa for
  ``en_zh``.  CJK text is tokenised as character 1-2 grams, everything else as
  lower-cased word tokens.
* **Per-query min-max normalisation before combining.**  Raw BM25 scores are
  unbounded and not comparable across fields or languages, so
  ``alpha*sim(input) + (1-alpha)*sim(state)`` on raw scores would make ``alpha``
  meaningless.  Each field's scores are min-max scaled *within the query's
  candidate pool* first.
* **Outcome-stratified quota.**  Greedy by combined score, but at most
  ``ceil(k/2)`` experiences may come from any single verdict class.  Without it
  top-k collapses onto the majority class (on GEC ~90% of steps are ties), the
  controller learns nothing, and the ``PositiveOnly`` ablation stops being a
  single-variable comparison.
* **Determinism.**  Scores, tie-breaks and the resulting id list are fully
  determined by (query, library).  This is what makes the Full vs
  ``OutcomeHidden`` invariant testable.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD = re.compile(r"[a-z0-9]+")


def tokenize_field(text: str) -> List[str]:
    """Language-aware tokenisation.

    CJK -> character uni- and bi-grams (whitespace splitting would make a whole
    Chinese sentence a single token).  Otherwise -> lower-cased word tokens.
    """
    s = str(text)
    if _CJK.search(s):
        chars = [c for c in s if not c.isspace()]
        unigrams = chars
        bigrams = ["".join(chars[i : i + 2]) for i in range(len(chars) - 1)]
        return unigrams + bigrams
    return _WORD.findall(s.lower())


@dataclass(frozen=True)
class Experience:
    exp_id: str
    task: str
    model: str
    source_input: str
    state_before: str
    state_after: str
    intervention_instruction: str
    intervention_rationale: str
    verdict: str  # better | worse | tie | uncertain
    reason_a: str
    reason_b: str
    order_consistent: bool
    delta_offline: Optional[float]
    provenance: str
    #: Trustworthy outcome label derived from the gold metric at bootstrap time.
    #: The rendered outcome used to be the double-order judge verdict, but the
    #: judge is the same model that wrote the revision and its verdict is largely
    #: a length preference (measured precision 21.5%).  A library whose only
    #: outcome field is that verdict carries mostly noise, which is why
    #: retrieving "better" experiences predicted *worse* subsequent revisions
    #: (17.8% vs 20.7%).  ``outcome_label`` is one of helped|hurt|unchanged.
    outcome_label: str = ""

    def to_dict(self) -> dict:
        return {
            "exp_id": self.exp_id,
            "task": self.task,
            "model": self.model,
            "source_input": self.source_input,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "intervention": {
                "instruction": self.intervention_instruction,
                "rationale": self.intervention_rationale,
            },
            "outcome": {
                "verdict": self.verdict,
                "reason_a": self.reason_a,
                "reason_b": self.reason_b,
                "order_consistent": self.order_consistent,
            },
            "delta_offline": self.delta_offline,
            "provenance": self.provenance,
            # Written ONLY when set, so a library built before this field existed
            # (or one that deliberately does not use it) serialises byte-for-byte
            # as before.  tests/test_accumulation_gaps.py asserts that identity.
            **({"outcome_label": self.outcome_label} if self.outcome_label else {}),
        }

    @staticmethod
    def from_dict(d: dict) -> "Experience":
        iv = d.get("intervention", {}) or {}
        oc = d.get("outcome", {}) or {}
        return Experience(
            exp_id=d["exp_id"],
            task=d.get("task", ""),
            model=d.get("model", ""),
            source_input=d.get("source_input", ""),
            state_before=d.get("state_before", ""),
            state_after=d.get("state_after", ""),
            intervention_instruction=iv.get("instruction", ""),
            intervention_rationale=iv.get("rationale", ""),
            verdict=oc.get("verdict", "uncertain"),
            reason_a=oc.get("reason_a", ""),
            reason_b=oc.get("reason_b", ""),
            order_consistent=bool(oc.get("order_consistent", False)),
            delta_offline=d.get("delta_offline"),
            provenance=d.get("provenance", ""),
            outcome_label=d.get("outcome_label", ""),
        )


class BM25Field:
    """Single-field BM25 over a fixed document collection."""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = [tokenize_field(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        self.tf: List[Counter] = [Counter(t) for t in self.doc_tokens]
        self.inverted: Dict[str, List[int]] = defaultdict(list)
        for i, tf in enumerate(self.tf):
            for tok in tf:
                self.inverted[tok].append(i)
        n = len(self.doc_tokens)
        self.idf: Dict[str, float] = {}
        for tok, posting in self.inverted.items():
            df = len(posting)
            self.idf[tok] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def raw_scores(self, query: str) -> Dict[int, float]:
        """Unbounded BM25 scores.  Only ever combined after normalisation."""
        q_tokens = set(tokenize_field(query))
        if not q_tokens:
            return {}
        candidates = set()
        for tok in q_tokens:
            candidates.update(self.inverted.get(tok, ()))
        out: Dict[int, float] = {}
        for doc_id in candidates:
            tf = self.tf[doc_id]
            dl = self.doc_len[doc_id]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / max(self.avg_len, 1e-9))
            s = 0.0
            for tok in q_tokens:
                f = tf.get(tok, 0)
                if not f:
                    continue
                s += self.idf.get(tok, 0.0) * f * (self.k1 + 1) / (f + denom_norm)
            out[doc_id] = s
        return out


def minmax_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    """Scale a query's raw field scores into [0, 1].

    A degenerate range (all equal, or a single candidate) maps every candidate
    to 1.0, i.e. the field contributes no ranking information rather than
    contributing noise.
    """
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-12:
        return {d: 1.0 for d in scores}
    return {d: (v - lo) / (hi - lo) for d, v in scores.items()}


@dataclass
class RetrievalResult:
    exp_ids: List[str]
    scores: List[float]
    per_class_counts: Dict[str, int]
    sim_input: List[float]
    sim_state: List[float]


class ExperienceRetriever:
    """Two-field BM25 retriever with an outcome-stratified quota."""

    MAX_K = 4

    def __init__(self, experiences: Sequence[Experience]):
        self.experiences = list(experiences)
        self.by_id = {e.exp_id: e for e in self.experiences}
        self.idx_input = BM25Field([e.source_input for e in self.experiences])
        self.idx_state = BM25Field([e.state_before for e in self.experiences])
        self.verdicts = [e.verdict for e in self.experiences]

    # -- scoring ------------------------------------------------------------
    def combined_scores(
        self, query_input: str, query_state: str, alpha: float
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        raw_i = self.idx_input.raw_scores(query_input)
        raw_s = self.idx_state.raw_scores(query_state)
        norm_i = minmax_normalize(raw_i)
        norm_s = minmax_normalize(raw_s)
        all_ids = set(norm_i) | set(norm_s)
        combined = {
            d: alpha * norm_i.get(d, 0.0) + (1.0 - alpha) * norm_s.get(d, 0.0)
            for d in all_ids
        }
        return combined, norm_i, norm_s

    # -- selection ----------------------------------------------------------
    def retrieve(
        self,
        query_input: str,
        query_state: str,
        alpha: float = 0.5,
        k: int = MAX_K,
        quota: bool = True,
        quota_profile: Optional[Dict[str, int]] = None,
        rng_rank: Optional[Dict[int, float]] = None,
        exclude_source: Optional[str] = None,
    ) -> RetrievalResult:
        """Return up to ``k`` experiences.

        ``quota_profile`` pins the exact per-verdict composition (used by the
        ``RandomRetrieve`` arm so that it changes *only* the ranking, not the
        positive/negative mix).  When given, ``k`` is taken from its sum.
        ``rng_rank`` replaces similarity ranking with a deterministic random
        ranking while keeping the same quota rule.

        ``exclude_source`` drops any unit whose ``source_input`` equals the
        query's own source, reproducing the original method verbatim::

            if similarities[i][idx] > 0.99 and candidate_text == query_texts[i]:
                continue

        It is required for validity because the original's retrieval DB and its
        test file are THE SAME csv (``RETRIEVAL_DB_PATH == TEST_DATA_PATH`` in
        llm_re_cosine_qwen8b.py), so the pool legitimately contains the query
        item -- and each unit's ``state_after`` is that item's gold answer.
        Without the skip, the refiner is handed
        ``Refined Translation (Gold): <the answer to the sentence it is
        translating>``, measured to occur for 98.3% of rows, which inflates BLEU
        by rewarding verbatim gold n-grams.
        """
        combined, norm_i, norm_s = self.combined_scores(query_input, query_state, alpha)

        if rng_rank is not None:
            rank_of = lambda d: (rng_rank.get(d, 0.0), self.experiences[d].exp_id)
            ordered = sorted(combined.keys(), key=lambda d: rank_of(d), reverse=True)
        else:
            ordered = sorted(
                combined.keys(),
                key=lambda d: (combined[d], self.experiences[d].exp_id),
                reverse=True,
            )

        if exclude_source is not None:
            ordered = [
                d for d in ordered
                if self.experiences[d].source_input != exclude_source
            ]

        if quota_profile is not None:
            k = sum(quota_profile.values())
            chosen = self._select_with_profile(ordered, quota_profile)
        elif quota:
            chosen = self._select_with_quota(ordered, k)
        else:
            chosen = ordered[:k]

        return RetrievalResult(
            exp_ids=[self.experiences[d].exp_id for d in chosen],
            scores=[round(combined[d], 8) for d in chosen],
            per_class_counts=dict(Counter(self.verdicts[d] for d in chosen)),
            sim_input=[round(norm_i.get(d, 0.0), 8) for d in chosen],
            sim_state=[round(norm_s.get(d, 0.0), 8) for d in chosen],
        )

    @staticmethod
    def _cap_for(k: int) -> int:
        return max(1, math.ceil(k / 2))

    def _select_with_quota(self, ordered: List[int], k: int) -> List[int]:
        cap = self._cap_for(k)
        counts: Counter = Counter()
        chosen: List[int] = []
        for d in ordered:
            v = self.verdicts[d]
            if counts[v] >= cap:
                continue
            chosen.append(d)
            counts[v] += 1
            if len(chosen) >= k:
                break
        # If the quota starved us (library has < k experiences outside the cap),
        # top up in rank order so we never silently return fewer than possible.
        if len(chosen) < k:
            taken = set(chosen)
            for d in ordered:
                if d not in taken:
                    chosen.append(d)
                    taken.add(d)
                    if len(chosen) >= k:
                        break
        return chosen

    def _select_with_profile(
        self, ordered: List[int], profile: Dict[str, int]
    ) -> List[int]:
        """Fill exactly ``profile[verdict]`` slots per class, in rank order."""
        remaining = dict(profile)
        chosen: List[int] = []
        for d in ordered:
            v = self.verdicts[d]
            if remaining.get(v, 0) > 0:
                chosen.append(d)
                remaining[v] -= 1
        # classes that could not be filled (pool too small) are simply short;
        # that shortfall is recorded by the caller via per_class_counts.
        return chosen


def deterministic_random_rank(exp_ids: Sequence[str], seed_key: str) -> Dict[int, float]:
    """Stable pseudo-random rank keyed by (seed_key, exp_id).

    Deterministic so the arm stays reproducible and so the same state always
    yields the same random ranking, while being uncorrelated with similarity.
    """
    import hashlib

    out: Dict[int, float] = {}
    for i, eid in enumerate(exp_ids):
        h = hashlib.sha256(f"{seed_key}|{eid}".encode("utf-8")).digest()
        out[i] = int.from_bytes(h[:8], "big") / float(1 << 64)
    return out
