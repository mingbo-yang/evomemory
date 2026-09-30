"""Unified scoring for every artefact in the experiment.

The plan (section 7) requires that the initial draft, intermediate candidates,
final outputs and *all re-scored baselines* go through one scorer instance, so
that no comparison mixes metric definitions.

Metrics
-------
* ``wmt19_en_zh`` -> SacreBLEU ``tokenize=zh``   (corpus BLEU)
* ``wmt19_zh_en`` -> SacreBLEU ``tokenize=13a``  (corpus BLEU)
* ``coedit_gec``  -> NLTK ``sentence_gleu``
* ``gigaword``    -> ROUGE-1/2/L with stemming

Note that this deliberately *replaces* the hand-written word/char BLEU in
``baseline/core/metrics.py``; re-scored numbers will therefore differ from the
existing ``*.summary.json`` files, and every table must be regenerated.

All metrics are reported on a 0-100 scale.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_WS = re.compile(r"\s+")

# --- Metric implementations -------------------------------------------------


def _sacrebleu_corpus(hyps: Sequence[str], refs: Sequence[str], tokenize: str) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_bleu(list(hyps), [list(refs)], tokenize=tokenize).score)


def _nltk_gleu(ref: str, hyp: str) -> float:
    from nltk.translate.gleu_score import sentence_gleu

    if not ref.strip() or not hyp.strip():
        return 0.0
    return float(sentence_gleu([ref.split()], hyp.split()))


def _rouge(ref: str, hyp: str) -> Dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = _get_rouge()
    if not ref.strip() or not hyp.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = scorer.score(ref, hyp)
    return {k: float(v.fmeasure) for k, v in scores.items()}


_ROUGE_SINGLETON = None


def _get_rouge():
    global _ROUGE_SINGLETON
    if _ROUGE_SINGLETON is None:
        from rouge_score import rouge_scorer

        _ROUGE_SINGLETON = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    return _ROUGE_SINGLETON


# --- Public API -------------------------------------------------------------

PRIMARY_METRIC = {
    "wmt19_en_zh": "bleu",
    "wmt19_zh_en": "bleu",
    "coedit_gec": "gleu",
    "gigaword": "rouge1",
}


class Scorer:
    """One scorer instance, shared by drafts / candidates / baselines."""

    def __init__(self, task: str):
        if task not in PRIMARY_METRIC:
            raise ValueError(f"unknown task {task!r}")
        self.task = task

    # -- per-example --------------------------------------------------------
    def score_one(self, reference: str, candidate: str) -> Dict[str, float]:
        if self.task == "coedit_gec":
            return {"gleu": 100.0 * _nltk_gleu(reference, candidate)}
        if self.task == "gigaword":
            return {k: 100.0 * v for k, v in _rouge(reference, candidate).items()}
        # translation handled corpora-wise; keep a cheap per-item proxy so that
        # per-step deltas (improve/degrade diagnostics) stay well defined.
        return {"bleu": 100.0 * _sentence_bleu_proxy(reference, candidate, self.task)}

    def primary(self, reference: str, candidate: str) -> float:
        return self.score_one(reference, candidate)[PRIMARY_METRIC[self.task]]

    # -- corpus -------------------------------------------------------------
    def score_corpus(self, pairs: Iterable[Tuple[str, str]]) -> Dict[str, float]:
        """pairs: iterable of (reference, candidate).  Returns corpus metrics."""
        refs: List[str] = []
        hyps: List[str] = []
        for ref, hyp in pairs:
            refs.append(ref)
            hyps.append(hyp)
        if self.task in ("wmt19_en_zh", "wmt19_zh_en"):
            tokenize = "zh" if self.task == "wmt19_en_zh" else "13a"
            return {"bleu": _sacrebleu_corpus(hyps, refs, tokenize)}
        if self.task == "coedit_gec":
            vals = [_nltk_gleu(r, h) for r, h in zip(refs, hyps)]
            return {"gleu": 100.0 * (sum(vals) / len(vals) if vals else 0.0)}
        # gigaword
        agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for r, h in zip(refs, hyps):
            for k, v in _rouge(r, h).items():
                agg[k] += v
        n = max(1, len(refs))
        return {k: 100.0 * v / n for k, v in agg.items()}


def _sentence_bleu_proxy(reference: str, candidate: str, task: str) -> float:
    """Sentence-level BLEU used only for *per-step* diagnostics.

    Corpus BLEU (SacreBLEU) is the reported number; this proxy is for
    improve/unchanged/degraded bookkeeping where a stable per-item value is
    needed.  It mirrors SacreBLEU's tokenisation choice per direction.
    """
    if task == "wmt19_en_zh":
        ref_tokens = _char_tokens(reference)
        cand_tokens = _char_tokens(candidate)
    else:
        ref_tokens = _13a_tokens(reference)
        cand_tokens = _13a_tokens(candidate)
    if not cand_tokens or not ref_tokens:
        return 0.0

    def ngrams(toks, n):
        return [tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))]

    from collections import Counter

    log_sum = 0.0
    for n in range(1, 5):
        cn, rn = ngrams(cand_tokens, n), ngrams(ref_tokens, n)
        if not cn:
            continue
        rc = Counter(rn)
        matches = sum(min(c, rc.get(g, 0)) for g, c in Counter(cn).items())
        p = matches / len(cn)
        if p <= 0:
            p = 1e-9
        log_sum += 0.25 * math.log(p)
    bp = math.exp(min(0.0, 1.0 - len(ref_tokens) / max(1, len(cand_tokens))))
    return float(bp * math.exp(log_sum))


def _char_tokens(text: str) -> List[str]:
    return list(_WS.sub("", str(text)))


def _13a_tokens(text: str) -> List[str]:
    return _WS.sub(" ", str(text)).strip().split()
