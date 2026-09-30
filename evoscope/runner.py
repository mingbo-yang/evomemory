"""Frozen-bank rollouts, clean checkpoints, exact observable replay checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Callable

from .core import Artifacts, Bank, Task, digest
from .environments import Environment, View
from .models import Model, actor_call


def resolve_goal(fallback: str, initial_observation: str) -> str:
    """Extract public ALFWorld task text without an additional model call.

    Preserve the initial observation on unknown formats rather than silently
    dropping task semantics. No hidden task file or ground truth is accessed.
    """
    match = re.search(r"(?:your task is to|your task is|task goal|task)\s*:\s*([^\n]+)",
                      initial_observation, flags=re.IGNORECASE)
    return match.group(1).strip() if match else fallback + "\n" + initial_observation


class RestoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class Checkpoint:
    task_id: str
    task_hash: str
    bank_hash: str
    actor_hash: str
    actions: tuple[str, ...]
    views: tuple[View, ...]
    seen_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    max_steps: int
    top_k: int
    resolved_goal: str
    environment_protocol: str = "strict-admissible-v1"

    @property
    def id(self) -> str:
        return digest(asdict(self))[:20]


@dataclass
class Episode:
    task_id: str
    phase: str
    bank_hash: str
    score: float
    done: bool
    actions: list[str]
    views: list[View]
    checkpoints: list[Checkpoint]
    error: str | None = None
    retrieval_counts: dict[str, int] = field(default_factory=dict)
    exposure_counts: dict[str, int] = field(default_factory=dict)

    def public_history(self) -> list[dict]:
        return history(self.views, self.actions)


def history(views: list[View] | tuple[View, ...], actions) -> list[dict]:
    # Observations and executed actions only: no memory text, hidden state or thoughts.
    return [{"observation": view.observation, "admissible": list(view.admissible),
             **({"action": actions[i - 1]} if i else {})}
            for i, view in enumerate(views)]


class Runner:
    def __init__(self, env_factory: Callable[[], Environment], model: Model,
                 artifacts: Artifacts, max_steps: int = 50, top_k: int = 3):
        if max_steps < 1 or top_k < 1:
            raise ValueError("positive max_steps and top_k required")
        self.env_factory = env_factory
        self.model = model
        self.artifacts = artifacts
        self.max_steps = max_steps
        self.top_k = top_k
        self.rollout_count = 0

    def run(self, task: Task, bank: Bank, phase: str, seed: int,
            checkpoint: Checkpoint | None = None, target: str | None = None,
            intervention: str | None = None, evidence_id: str | None = None,
            *, intervention_scope: str = "persistent") -> Episode:
        if intervention_scope not in {"persistent", "single-step"}:
            raise ValueError("unknown intervention scope")
        if intervention_scope != "persistent" and intervention is None:
            raise ValueError("single-step scope requires an explicit intervention")
        if intervention not in {None, "expose", "mask", "normal"}:
            raise ValueError("unknown intervention")
        if (target is None) != (intervention is None):
            raise ValueError("target and intervention must be supplied together")
        if intervention and checkpoint is None:
            raise ValueError("interventions require a clean checkpoint")
        if checkpoint:
            if (checkpoint.task_id != task.id or checkpoint.task_hash != digest(asdict(task))
                    or checkpoint.bank_hash != bank.fingerprint
                    or checkpoint.actor_hash != self.model.fingerprint
                    or checkpoint.max_steps != self.max_steps or checkpoint.top_k != self.top_k):
                raise RestoreError("checkpoint configuration mismatch")
            if target in checkpoint.seen_ids or target not in checkpoint.selected_ids:
                raise RestoreError("target is contaminated or was not retrieved")
        env = self.env_factory()
        protocol = getattr(env, "action_protocol", "strict-admissible-v1")
        self.rollout_count += 1
        rollout_id = f"rollout-{self.rollout_count:06d}"
        previous_context = self.artifacts.cost_context
        self.artifacts.cost_context = {"rollout_id": rollout_id, "evidence_id": evidence_id,
                                       "branch": intervention, "intervention_scope": intervention_scope}
        actions: list[str] = []
        views: list[View] = []
        checkpoints: list[Checkpoint] = []
        seen = set(checkpoint.seen_ids if checkpoint else ())
        error = None
        retrieval_counts, exposure_counts = {}, {}
        try:
            if checkpoint and checkpoint.environment_protocol != protocol:
                raise RestoreError("environment action protocol mismatch")
            views.append(env.reset(task))
            goal = resolve_goal(task.goal, views[0].observation)
            if checkpoint and goal != checkpoint.resolved_goal:
                raise RestoreError("resolved goal mismatch")
            if checkpoint:
                if views[0] != checkpoint.views[0]:
                    raise RestoreError("initial observation mismatch")
                for i, action in enumerate(checkpoint.actions):
                    self.artifacts.cost(phase, "step", replay=True)
                    views.append(env.step(action))
                    actions.append(action)
                    if views[-1] != checkpoint.views[i + 1]:
                        raise RestoreError(f"replay mismatch at action {i}")
            while len(actions) < self.max_steps and not views[-1].done:
                selected = bank.retrieve(goal + " " + views[-1].observation, self.top_k)
                if checkpoint and len(actions) == len(checkpoint.actions):
                    if tuple(p.id for p in selected) != checkpoint.selected_ids:
                        raise RestoreError("retrieval mismatch")
                if phase == "learn":
                    checkpoints.append(Checkpoint(task.id, digest(asdict(task)), bank.fingerprint, self.model.fingerprint,
                        tuple(actions), tuple(views), tuple(sorted(seen)),
                        tuple(p.id for p in selected), self.max_steps, self.top_k, goal, protocol))
                # Preserve the initial retrieval rank throughout the intervention.
                # On dropout reserve one slot only if capacity would be exceeded.
                intervention_applied = (intervention in {"expose", "mask"} and
                    (intervention_scope == "persistent" or len(actions) == len(checkpoint.actions)))
                if intervention_applied:
                    others = [p for p in selected if p.id != target][:self.top_k - 1]
                    memory = [{"id": p.id, "when": p.when, "do": p.do} for p in others]
                    rank = checkpoint.selected_ids.index(target)
                    if intervention == "expose":
                        memory.insert(min(rank, len(memory)), {"id": target, "do": bank.get(target).do})
                else:
                    memory = [{"id": p.id, "when": p.when, "do": p.do} for p in selected]
                for policy in selected:
                    retrieval_counts[policy.id] = retrieval_counts.get(policy.id, 0) + 1
                for policy in memory:
                    exposure_counts[policy["id"]] = exposure_counts.get(policy["id"], 0) + 1
                seen.update(p["id"] for p in memory)
                decision_log = "gate/steps" if phase == "gate" else f"{phase}/decisions"
                self.artifacts.append(decision_log, {
                    "task_id": task.id, "bank_hash": bank.fingerprint, "step": len(actions),
                    "rollout_id": rollout_id, "evidence_id": evidence_id, "resolved_goal": goal,
                    "target": target, "intervention": intervention,
                    "intervention_scope": intervention_scope,
                    "intervention_applied": intervention_applied,
                    "retrieved_ids": [p.id for p in selected],
                    "policies": memory, "seed": seed + len(actions)})
                action = actor_call(self.model, goal, history(views, actions), memory,
                                    phase, seed + len(actions),
                                    validate_actions=protocol == "strict-admissible-v1")
                self.artifacts.append(f"{phase}/action_audit", {
                    "rollout_id": rollout_id, "step": len(actions), "action": action,
                    "environment_protocol": protocol,
                    "listed_exactly": action in views[-1].admissible,
                    "note": "unlisted is diagnostic only; native environment decides the outcome"})
                self.artifacts.cost(phase, "step", replay=False)
                views.append(env.step(action))
                actions.append(action)
        except RestoreError:
            raise
        except Exception as exc:
            # Exception messages may include server credentials; record type only.
            error = type(exc).__name__
        finally:
            try:
                env.close()
            finally:
                self.artifacts.cost_context = previous_context
        episode = Episode(task.id, phase, bank.fingerprint, views[-1].score if views else 0.0,
                          views[-1].done if views else False, actions, views, checkpoints, error, retrieval_counts, exposure_counts)
        self.artifacts.append(f"{phase}/episodes", {**asdict(episode), "rollout_id": rollout_id,
                              "evidence_id": evidence_id, "intervention": intervention,
                              "intervention_scope": intervention_scope})
        return episode
