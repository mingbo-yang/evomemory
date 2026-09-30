from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    latency: float
    seed: int
    call_type: str
    start_time: float
    end_time: float
    prompt: str = ""

    @property
    def total_tokens(self) -> int:
        return int(self.input_tokens) + int(self.output_tokens)

    def to_dict(self, include_prompt: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        if not include_prompt:
            data.pop("prompt", None)
        return data


@dataclass
class RunResult:
    final_output: str
    candidates: List[str]
    generations: List[Generation]
    utility_scores: List[float]
    stop_round: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(g.total_tokens for g in self.generations)

    @property
    def total_latency(self) -> float:
        return sum(g.latency for g in self.generations)

    @property
    def sequential_tokens(self) -> int:
        if "sequential_tokens_override" in self.metadata:
            return int(self.metadata["sequential_tokens_override"])
        ids = self.metadata.get("sequential_call_indices")
        if not ids:
            return self.total_tokens
        return sum(self.generations[i].total_tokens for i in ids if 0 <= i < len(self.generations))

    @property
    def sequential_latency(self) -> float:
        if "sequential_latency_override" in self.metadata:
            return float(self.metadata["sequential_latency_override"])
        ids = self.metadata.get("sequential_call_indices")
        if not ids:
            return self.total_latency
        return sum(self.generations[i].latency for i in ids if 0 <= i < len(self.generations))

    def to_trace_dict(self) -> Dict[str, Any]:
        return {
            "final_output": self.final_output,
            "candidates": self.candidates,
            "generations": [g.to_dict(include_prompt=True) for g in self.generations],
            "utility_scores": self.utility_scores,
            "stop_round": self.stop_round,
            "metadata": self.metadata,
            "total_tokens": self.total_tokens,
            "sequential_tokens": self.sequential_tokens,
            "total_latency_s": self.total_latency,
            "sequential_latency_s": self.sequential_latency,
            "num_llm_calls": len(self.generations),
        }


@dataclass
class TaskExample:
    index: int
    source: str
    reference: str
    task: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceBudget:
    max_rounds: int = 3
    n_candidates: int = 4
    max_tokens: int = 1024
    feedback_max_tokens: int = 512
    workspace_max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    key: str
    path: str
    param_count: float
    tensor_parallel_size: int = 1
    max_model_len: int = 32768


@dataclass(frozen=True)
class TaskConfig:
    key: str
    display_name: str
    default_samples: int
    max_tokens: int
    sample_policy: str
    metric_name: str
