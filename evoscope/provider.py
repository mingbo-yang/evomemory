"""Read-only memory-provider bridge; online ingestion never bypasses the gate.

When EvolveLab is importable, return its native response/item classes. Otherwise
use lightweight compatible records, allowing independent installation/testing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .core import Bank


@dataclass
class MemoryItem:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    score: float | None = None


@dataclass
class MemoryResponse:
    memories: list
    memory_type: object
    total_count: int
    request_id: str | None = None


class EvoScopeProvider:
    """Matches initialize/provide_memory/take_in_memory without modifying upstream.

    Pass the host's existing MemoryType value when integrating with EvolveLab;
    it has no built-in EVOSCOPE enum. Register this provider in the host yourself.
    """

    def __init__(self, bank_path: str, memory_type="evoscope", config: dict | None = None):
        self.bank_path = bank_path
        self.memory_type = memory_type
        self.config = config or {}
        self.bank = None

    def initialize(self) -> bool:
        self.bank = Bank.load(self.bank_path)
        return True

    def get_memory_type(self):
        return self.memory_type

    def get_config(self):
        return dict(self.config)

    def refresh(self) -> bool:
        """Atomically load a new bank at a task boundary; keep a task's snapshot."""
        candidate = Bank.load(self.bank_path)
        changed = self.bank is None or candidate.fingerprint != self.bank.fingerprint
        if changed:
            self.bank = candidate
        return changed

    def provide_memory(self, request):
        if self.bank is None:
            raise RuntimeError("initialize the provider first")
        status = getattr(request, "status", None)
        if getattr(status, "value", status) == "begin":
            self.refresh()
        item_cls, response_cls = MemoryItem, MemoryResponse
        import importlib.util
        if importlib.util.find_spec("EvolveLab") is not None:
            from EvolveLab.memory_types import MemoryItem as item_cls, MemoryResponse as response_cls
        selected = self.bank.retrieve(request.query + " " + (request.context or ""),
                                      int(self.config.get("top_k", 3)))
        memories = [item_cls(id=p.id, content=f"When: {p.when}\nDo: {p.do}",
                            metadata={"version": p.version, "bank_hash": self.bank.fingerprint})
                    for p in selected]
        return response_cls(memories=memories, memory_type=self.memory_type,
                            total_count=len(memories))

    def take_in_memory(self, trajectory_data) -> tuple[bool, str]:
        # Accepting trajectories here would bypass split validation, replay and gating.
        return False, "Use the EvoScope runner to collect learn trajectories and gate condition edits."
