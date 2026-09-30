"""Immutable policies, fixed retrieval, manifests and append-only run artifacts."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import re
import tempfile
from typing import Any


CONDITION_MAX_CHARS = 400


def validate_condition(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > CONDITION_MAX_CHARS:
        raise ValueError(f"condition must contain 1..{CONDITION_MAX_CHARS} characters")
    return value.strip()


def condition_stats(bank) -> dict:
    lengths = [len(p.when) for p in bank.policies]
    return {"unit": "characters", "limit": CONDITION_MAX_CHARS,
            "mean": sum(lengths) / len(lengths) if lengths else 0,
            "max": max(lengths, default=0)}


def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class Policy:
    id: str
    key: str
    when: str
    do: str
    version: int = 1
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("id", "key", "when", "do"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"policy {name} must be a nonempty string")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("policy version must be a positive integer")
        object.__setattr__(self, "when", validate_condition(self.when))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))


@dataclass(frozen=True)
class Bank:
    """A value object: edits return a fresh bank, never change the deployed one."""

    policies: tuple[Policy, ...]

    def __init__(self, policies: list[Policy] | tuple[Policy, ...]):
        object.__setattr__(self, "policies", tuple(policies))
        if len({p.id for p in self.policies}) != len(self.policies):
            raise ValueError("duplicate policy id")

    @property
    def fingerprint(self) -> str:
        return digest(self.to_json())

    def to_json(self) -> list[dict]:
        return [asdict(p) for p in self.policies]

    @classmethod
    def load(cls, path: str | Path) -> Bank:
        return cls([Policy(**p) for p in json.loads(Path(path).read_text(encoding="utf-8"))])

    def get(self, policy_id: str) -> Policy:
        return next(p for p in self.policies if p.id == policy_id)

    def revise(self, policy_id: str, when: str, evidence_ids: tuple[str, ...]) -> Bank:
        old = self.get(policy_id)
        when = validate_condition(when)
        new = replace(old, when=when.strip(), version=old.version + 1,
                      evidence_ids=old.evidence_ids + tuple(evidence_ids))
        return Bank([new if p.id == policy_id else p for p in self.policies])

    def retrieve(self, query: str, k: int = 3) -> tuple[Policy, ...]:
        """BM25 over immutable key+do only; deterministic ID tie breaking."""
        if k < 1:
            raise ValueError("k must be positive")
        scores = self.retrieval_scores(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.policies[i].id))
        return tuple(self.policies[i] for i in order if scores[i] > 0)[:k]

    def retrieval_scores(self, query: str) -> tuple[float, ...]:
        """Expose the fixed key+do BM25 scores for candidate-independent planning."""
        tokenize = lambda s: re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", s.lower())
        docs = [Counter(tokenize(p.key + " " + p.do)) for p in self.policies]
        n = len(docs)
        if not n:
            return ()
        avg = sum(sum(d.values()) for d in docs) / n or 1
        scores = [0.0] * n
        for term in set(tokenize(query)):
            df = sum(term in d for d in docs)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, d in enumerate(docs):
                tf = d[term]
                scores[i] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * sum(d.values()) / avg))
        return tuple(scores)


@dataclass(frozen=True)
class Task:
    id: str
    group_id: str
    split: str
    goal: str
    payload: dict = field(default_factory=dict)
    seed: int = 0


def load_manifest(path: str | Path) -> list[Task]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = [Task(**row) for row in rows]
    validate_tasks(tasks)
    return tasks


def validate_tasks(tasks: list[Task]) -> None:
    ids: set[str] = set()
    groups: dict[str, str] = {}
    games: dict[str, str] = {}
    for task in tasks:
        if task.split not in {"bootstrap", "learn", "gate", "test"}:
            raise ValueError(f"unknown split: {task.split}")
        if not task.id or not task.group_id or not task.goal or task.id in ids:
            raise ValueError("task id/group/goal empty, or duplicate task id")
        ids.add(task.id)
        if groups.setdefault(task.group_id, task.split) != task.split:
            raise ValueError(f"group crosses splits: {task.group_id}")
        if "goal_index" in task.payload:
            index = task.payload["goal_index"]
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError("invalid WebShop goal index")
            key = "webshop:" + str(index)
            if key in games:
                raise ValueError("duplicate WebShop goal")
            games[key] = task.id
        if "gamefile" in task.payload:
            path = Path(task.payload["gamefile"]).expanduser().resolve()
            # Same file or byte-identical copies cannot masquerade as new tasks.
            keys = [str(path)]
            if path.is_file():
                keys.append(sha256(path.read_bytes()).hexdigest())
            for key in keys:
                if key in games:
                    raise ValueError(f"duplicate ALFWorld game: {task.id}, {games[key]}")
                games[key] = task.id


class Artifacts:
    """Exclusive run directory; no accidental overwriting of earlier experiments."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.costs: list[dict] = []
        self.cost_context: dict = {}

    def append(self, name: str, value: dict) -> None:
        path = self.root / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def write(self, name: str, value: Any) -> None:
        atomic_json(self.root / f"{name}.json", value)

    def cost(self, phase: str, kind: str, **values: Any) -> None:
        row = {"phase": phase, "kind": kind, **self.cost_context, **values}
        self.costs.append(row)
        self.append("costs", row)

    def cost_summary(self, start: int = 0) -> dict:
        result: dict[str, dict] = {}
        for row in self.costs[start:]:
            bucket = result.setdefault(row["phase"], {"model_calls": 0, "failed_calls": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "unknown_usage_calls": 0,
                "environment_steps": 0, "replay_steps": 0})
            if row["kind"] == "model":
                bucket["model_calls"] += 1
                bucket["failed_calls"] += int(not row.get("ok", True))
                bucket["unknown_usage_calls"] += int(row.get("prompt_tokens") is None)
                for key in ("prompt_tokens", "completion_tokens"):
                    bucket[key] += row.get(key) or 0
            elif row["kind"] == "step":
                bucket["environment_steps"] += 1
                bucket["replay_steps"] += int(row.get("replay", False))
        return result
