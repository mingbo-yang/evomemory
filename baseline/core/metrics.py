from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Sequence, Tuple


def _char_tokens(text: str) -> List[str]:
    return list("".join(str(text).split()))


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE)


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def _clipped_matches(candidate: Sequence[Tuple[str, ...]], reference: Sequence[Tuple[str, ...]]) -> int:
    cand_counts = Counter(candidate)
    ref_counts = Counter(reference)
    return sum(min(count, ref_counts.get(ng, 0)) for ng, count in cand_counts.items())


def bleu_score(reference: str, candidate: str, max_n: int = 4, char_level: bool = True) -> float:
    cand_tokens = _char_tokens(candidate) if char_level else _word_tokens(candidate)
    ref_tokens = _char_tokens(reference) if char_level else _word_tokens(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    bp = math.exp(min(0.0, 1.0 - len(ref_tokens) / max(1, len(cand_tokens))))
    log_sum = 0.0
    for n in range(1, max_n + 1):
        cand_ng = _ngrams(cand_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        total = max(1, len(cand_ng))
        precision = _clipped_matches(cand_ng, ref_ng) / total
        if precision <= 0:
            precision = 1e-9
        log_sum += (1.0 / max_n) * math.log(precision)
    return float(bp * math.exp(log_sum))


def rouge1_f1(reference: str, candidate: str) -> float:
    ref = _word_tokens(reference)
    cand = _word_tokens(candidate)
    if not ref or not cand:
        return 0.0
    ref_counts = Counter(ref)
    cand_counts = Counter(cand)
    overlap = sum(min(c, ref_counts.get(tok, 0)) for tok, c in cand_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand)
    recall = overlap / len(ref)
    return float((2 * precision * recall) / max(1e-12, precision + recall))


def gleu_like(reference: str, candidate: str) -> float:
    """Small dependency-free GLEU-style score for GEC smoke tests and summaries.

    The original ICML scripts use task-specific metric code. This approximation is
    deterministic and keeps this baseline runner self-contained; paper tables can
    recompute final metrics offline with the exact metric implementation.
    """
    ref = _word_tokens(reference)
    cand = _word_tokens(candidate)
    if not ref or not cand:
        return 0.0
    vals: List[float] = []
    for n in range(1, 5):
        ref_ng = _ngrams(ref, n)
        cand_ng = _ngrams(cand, n)
        if not ref_ng or not cand_ng:
            continue
        matches = _clipped_matches(cand_ng, ref_ng)
        precision = matches / len(cand_ng)
        recall = matches / len(ref_ng)
        vals.append(min(precision, recall))
    return float(sum(vals) / len(vals)) if vals else 0.0


def task_metric(task: str, reference: str, candidate: str) -> float:
    if task in {"wmt19_en_zh", "wmt19_zh_en"}:
        return bleu_score(reference, candidate, char_level=(task == "wmt19_en_zh"))
    if task == "coedit_gec":
        return gleu_like(reference, candidate)
    if task == "gigaword":
        return rouge1_f1(reference, candidate)
    return 0.0


def language_aware_tokens(text: str) -> List[str]:
    s = str(text)
    if re.search(r"[\u4e00-\u9fff]", s):
        return _char_tokens(s)
    return _word_tokens(s)

