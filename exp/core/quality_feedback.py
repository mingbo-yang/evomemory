"""Reference-free BERT decisions; delayed task feedback remains separate.

Architecture and pair tokenization match icml's DeepBertClassifier. No random
weight fallback, sigmoid, score clamping, or test-reference input is permitted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import time


@dataclass(frozen=True)
class QualityPolicy:
    stop_threshold: float
    min_gain: float = 0.01
    soft_length_ratio: float = 1.02
    hard_length_ratio: float = 1.5
    expansion_gain: float = 0.05
    length_allowance: int = 8

    def __post_init__(self):
        values = (self.stop_threshold, self.min_gain, self.soft_length_ratio,
                  self.hard_length_ratio, self.expansion_gain)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("quality policy must be finite")
        if not (0 <= self.min_gain <= self.expansion_gain):
            raise ValueError("require 0 <= min_gain <= expansion_gain")
        if not (1 <= self.soft_length_ratio <= self.hard_length_ratio):
            raise ValueError("require 1 <= soft_length_ratio <= hard_length_ratio")
        if self.length_allowance < 0:
            raise ValueError("length_allowance must be nonnegative")

    def should_stop(self, score):
        if not math.isfinite(score):
            raise ValueError("nonfinite BERT quality score")
        return score > self.stop_threshold  # exact threshold convention in icml

    def accept(self, current, candidate, before, after):
        if not all(math.isfinite(v) for v in (before, after)):
            raise ValueError("nonfinite BERT quality score")
        if not candidate.strip() or candidate == current:
            return False, "empty_or_identical"
        gain = after - before
        if gain <= self.min_gain:
            return False, "insufficient_quality_gain"
        n = max(1, len(current))
        if len(candidate) > self.hard_length_ratio * n + self.length_allowance:
            return False, "hard_length_limit"
        if (len(candidate) > self.soft_length_ratio * n + self.length_allowance
                and gain <= self.expansion_gain):
            return False, "expansion_without_strong_gain"
        return True, "quality_gain"


class BertQualityEvaluator:
    def __init__(self, checkpoint, base_model, device="cpu", batch_size=16):
        import torch
        from torch import nn
        from transformers import BertConfig, BertModel, BertTokenizer

        if device == "cpu":
            torch.set_num_threads(min(torch.get_num_threads(), 4))
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        self.tokenizer = BertTokenizer.from_pretrained(base_model, local_files_only=True)
        config = BertConfig.from_pretrained(base_model, local_files_only=True)
        # Checkpoints contain the fine-tuned encoder AND the regression head.
        # Loading base weights first would be redundant.
        class DeepBertClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel(config)
                layers = [nn.Linear(config.hidden_size, 512), nn.ReLU(), nn.Dropout(0.3)]
                for _ in range(3):
                    layers.extend([nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3)])
                layers.append(nn.Linear(512, 1))
                self.classifier = nn.Sequential(*layers)

            def forward(self, input_ids, attention_mask, token_type_ids):
                return self.classifier(self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                                  token_type_ids=token_type_ids).pooler_output)

        self.model = DeepBertClassifier()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.elapsed_s = 0.0
        self.pairs_scored = 0

    def score_pairs(self, pairs):
        import torch
        if not pairs:
            return []
        start = time.perf_counter()
        scores = []
        with torch.inference_mode():
            for offset in range(0, len(pairs), self.batch_size):
                encoded = self.tokenizer.batch_encode_plus(
                    [(str(a), str(b)) for a, b in pairs[offset:offset+self.batch_size]],
                    add_special_tokens=True, max_length=512, padding="max_length",
                    truncation=True, return_attention_mask=True,
                    return_token_type_ids=True, return_tensors="pt",
                )
                scores.extend(self.model(**{k: v.to(self.device) for k, v in encoded.items()})
                              .squeeze(1).cpu().tolist())
        if not all(math.isfinite(v) for v in scores):
            raise ValueError("BERT returned nonfinite scores")
        self.elapsed_s += time.perf_counter() - start
        self.pairs_scored += len(pairs)
        return scores
