"""Batched generation on top of the read-only baseline LLMClient.

Why
---
``baseline/core/llm.py`` calls ``self.model.generate([text_input], params)`` --
one prompt per call.  Every experiment therefore ran at **batch size 1**, which
leaves vLLM latency-bound: GPU utilisation oscillates and the KV cache we paid
for sits almost entirely unused.

Samples in this experiment are mutually independent (only the *rounds within* a
sample are sequential), so the scheduler can run many samples in lock-step and
issue one batched call per phase.  That is what this module provides.

Per-request seeds are preserved: vLLM accepts a *list* of ``SamplingParams``, one
per prompt, so each request keeps the deterministic seed derived by
``core.determinism.call_seed``.  Prompt construction is byte-identical to the
unbatched path, so the single-variable invariants (A/B ordering, retrieved
experience ids, prompt text) are unaffected.  Only the sampling *batch* differs,
which is the same source of nondeterminism already measured and documented.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from baseline_core.types import Generation


@dataclass
class GenRequest:
    prompt: str
    system_prompt: str
    seed: int
    call_type: str
    max_tokens: int
    temperature: float = 0.1
    top_p: float = 1.0


class BatchedLLM:
    """Thin wrapper exposing ``generate_batch`` over a baseline ``LLMClient``."""

    def __init__(self, client):
        self.client = client

    # -- pass-throughs ------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return self.client.count_tokens(text)

    @property
    def backend(self) -> str:
        return self.client.backend

    # -- batched ------------------------------------------------------------
    def generate_batch(self, requests: Sequence[GenRequest]) -> List[Generation]:
        if not requests:
            return []
        if len(requests) == 1:
            # keep the single-request path byte-identical to the baseline
            r = requests[0]
            return [
                self.client.generate(
                    prompt=r.prompt,
                    system_prompt=r.system_prompt,
                    seed=r.seed,
                    call_type=r.call_type,
                    max_tokens=r.max_tokens,
                    temperature=r.temperature,
                    top_p=r.top_p,
                )
            ]

        if self.client.backend == "vllm":
            return self._vllm_batch(requests)
        return [self._one(requests, i) for i in range(len(requests))]

    def _one(self, requests: Sequence[GenRequest], i: int) -> Generation:
        r = requests[i]
        return self.client.generate(
            prompt=r.prompt,
            system_prompt=r.system_prompt,
            seed=r.seed,
            call_type=r.call_type,
            max_tokens=r.max_tokens,
            temperature=r.temperature,
            top_p=r.top_p,
        )

    def _vllm_batch(self, requests: Sequence[GenRequest]) -> List[Generation]:
        from vllm import SamplingParams

        texts = [self.client._chat_text(r.system_prompt, r.prompt) for r in requests]
        in_tokens = [self.client.count_tokens(t) for t in texts]
        params = [
            SamplingParams(
                temperature=r.temperature,
                top_p=r.top_p,
                max_tokens=r.max_tokens,
                seed=r.seed,
                stop_token_ids=self.client.stop_token_ids or None,
            )
            for r in requests
        ]
        start = time.time()
        outs = self.client.model.generate(texts, params)
        end = time.time()
        per_call = (end - start) / max(1, len(requests))

        results: List[Generation] = []
        for i, (r, out) in enumerate(zip(requests, outs)):
            o = out.outputs[0] if out.outputs else None
            raw = o.text if o is not None else ""
            out_tok = len(getattr(o, "token_ids", []) or []) if o is not None else 0
            results.append(
                Generation(
                    text=str(raw).strip(),
                    input_tokens=int(in_tokens[i]),
                    output_tokens=int(out_tok),
                    # vLLM reports only whole-batch timing; attribute an equal
                    # share to each request so per-sample cost stays meaningful.
                    latency=float(per_call),
                    seed=int(r.seed),
                    call_type=r.call_type,
                    start_time=float(start),
                    end_time=float(end),
                    prompt=r.prompt,
                )
            )
        return results
