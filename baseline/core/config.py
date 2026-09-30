from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .types import ModelConfig, TaskConfig


BASELINE_ROOT = Path("/mnt/huawei/ymb/aaai2027/baseline")
ICML_ROOT = Path("/mnt/huawei/ymb/icml")

WMT19_PARQUET = {
    "train": "/mnt/huawei/wwq/project/No-box-translation/ISSTA2025/11.2-ISSTA/data/wmt19-en-zh/train.parquet",
    "validation": "/mnt/huawei/wwq/project/No-box-translation/ISSTA2025/11.2-ISSTA/data/wmt19-en-zh/validation.parquet",
}

COEDIT_GEC_CSV = str(ICML_ROOT / "rag" / "coedit_gec_dataset_analysis_full.csv")
GIGAWORD_CSV = str(ICML_ROOT / "rag" / "gigaword_dataset_analysis_full.csv")

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "glm4-9b": ModelConfig(
        "glm4-9b",
        "/mnt/huawei/ymb/model/glm-4-9b/model",
        9e9,
    ),
    "llama3.1-8b": ModelConfig("llama3.1-8b", "/mnt/huawei/wwq/model/llama3.1/model", 8e9),
    "qwen3-4b": ModelConfig("qwen3-4b", "/mnt/huawei/wwq/model/Qwen3-4B/model", 4e9),
    "qwen3-8b": ModelConfig("qwen3-8b", "/mnt/huawei/wwq/model/Qwen3-8B/model", 8e9),
    "qwen3-32b": ModelConfig("qwen3-32b", "/mnt/huawei/wwq/model/Qwen3-32B/model", 32e9),
}

TASK_CONFIGS: Dict[str, TaskConfig] = {
    "wmt19_en_zh": TaskConfig("wmt19_en_zh", "Translation(English-to-Chinese): WMT19 En-Zh", 3000, 1024, "match_icml_3000", "bleu"),
    "wmt19_zh_en": TaskConfig("wmt19_zh_en", "Translation(Chinese-to-English): WMT19 Zh-En", 3000, 1024, "match_icml_3000", "bleu"),
    "coedit_gec": TaskConfig("coedit_gec", "Grammatical Error Correction: CoEdit GEC", 3000, 1024, "match_icml_3000", "gleu"),
    "gigaword": TaskConfig("gigaword", "Summarization: Gigaword", 100, 128, "match_icml_100", "rouge1"),
}

METHODS_FIRST_AND_SECOND_BATCH: List[str] = [
    "Direct-Zero",
    "BoN-U-4",
    "SelfRefine-Fixed",
    "SelfRefine-U",
    "SR-Fixed",
    "SR-U",
    "PDR-2-1",
    "ChecklistRefine",
    "ModeX-4",
    "TEaR-Native",
    "TEaR-U",
    "AdaCompute-SR-U",
]

UTILITY_THRESHOLDS = {
    "wmt19_en_zh": 0.6,
    "wmt19_zh_en": 0.6,
    "coedit_gec": 0.6,
    "gigaword": 0.5,
}
