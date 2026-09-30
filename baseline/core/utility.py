from __future__ import annotations

import os
from typing import List

from .config import UTILITY_THRESHOLDS
from .metrics import task_metric
from .types import TaskExample


UTILITY_PATHS = {
    "wmt19_en_zh": "/mnt/huawei/wwq/model/aaa_experiment/control_n/model/layers_3_best.pth",
    "wmt19_zh_en": "/mnt/huawei/wwq/model/aaa_experiment/control_n/model/layers_3_best.pth",
    "coedit_gec": "/mnt/huawei/ymb/icml/bert/gec/model/model/layers_3_best.pth",
    "gigaword": "/mnt/huawei/ymb/icml/bert/gigaword/model/model/layers_3_best.pth",
}

BERT_LOCAL_PATH = "/home/ymb/.cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots/3f076fdb1ab68d5b2880cb87a0886f315b8146f8"


class UtilityPredictor:
    def __init__(self, task: str, device: str = "cuda") -> None:
        self.task = task
        self.threshold = UTILITY_THRESHOLDS.get(task, 0.6)
        self.device_name = device
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            import torch.nn as nn
            from transformers import BertModel, BertTokenizer
        except Exception as e:
            raise RuntimeError(
                "Utility predictor requires torch and transformers. Use a Python environment matching the ICML experiments."
            ) from e

        class DeepBertClassifier(nn.Module):
            def __init__(self, bert_model_name: str, layers_num: int, hidden_dim: int = 512):
                super().__init__()
                self.bert = BertModel.from_pretrained(bert_model_name, local_files_only=True)
                layers = [nn.Linear(self.bert.config.hidden_size, hidden_dim), nn.ReLU(), nn.Dropout(0.3)]
                for _ in range(layers_num):
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3)])
                layers.append(nn.Linear(hidden_dim, 1))
                self.classifier = nn.Sequential(*layers)

            def forward(self, input_ids, attention_mask, token_type_ids):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                return self.classifier(outputs.pooler_output)

        self.torch = torch
        bert_source = BERT_LOCAL_PATH if os.path.isdir(BERT_LOCAL_PATH) else "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(bert_source, local_files_only=True)
        self.device = torch.device(self.device_name if torch.cuda.is_available() and self.device_name == "cuda" else "cpu")
        self.model = DeepBertClassifier(bert_source, 3).to(self.device)
        path = UTILITY_PATHS.get(self.task)
        if path and os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Utility predictor weights not found for {self.task}: {path}")
        self.model.eval()
        self._loaded = True

    def score_batch(self, examples: List[TaskExample], outputs: List[str]) -> List[float]:
        self._load()
        pairs = [(str(ex.source), str(out)) for ex, out in zip(examples, outputs)]
        enc = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            vals = self.model(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]).squeeze(1)
        return [float(x) for x in vals.detach().cpu().tolist()]

    def score(self, example: TaskExample, output: str) -> float:
        return self.score_batch([example], [output])[0]


class ReferenceMetricScorer:
    """Oracle/reference scorer for diagnostics only, never used for U baselines."""

    def __init__(self, task: str):
        self.task = task

    def score(self, example: TaskExample, output: str) -> float:
        return task_metric(self.task, example.reference, output)
