"""Preallocate disjoint validation blocks from frozen public initial observations."""
from dataclasses import asdict, dataclass
import random

from .core import digest
from .runner import resolve_goal


@dataclass(frozen=True)
class GateBlock:
    id: str
    policy_id: str
    attempt: int
    local_ids: tuple[str, ...]
    global_ids: tuple[str, ...]

    @property
    def task_ids(self):
        return self.local_ids + self.global_ids


def prepare_gate_plan(bank, tasks, runner, artifacts, gate_size, local_size, attempts, seed):
    """No actor/editor calls, rewards, candidates or hidden task payloads in ranking.

    Initial observations are only a relevance proxy; rollout coverage is audited
    separately. Reserve groups globally, including unused/noop proposal slots.
    """
    if not 1 <= local_size <= gate_size or attempts < 1:
        raise ValueError("invalid local/global gate sizes or candidate budget")
    representatives = {}
    for task in sorted(tasks, key=lambda t: t.id):
        if task.split == "gate":
            representatives.setdefault(task.group_id, task)
    public = []
    for task in representatives.values():
        env = runner.env_factory()
        try:
            view = env.reset(task)
            goal = resolve_goal(task.goal, view.observation)
            query = goal + " " + view.observation
            public.append({"task_id": task.id, "group_id": task.group_id,
                           "goal": goal, "initial_observation": view.observation,
                           "scores": dict(zip((p.id for p in bank.policies), bank.retrieval_scores(query)))})
            artifacts.cost("gate_plan", "reset", task_id=task.id, ok=True)
        except Exception as exc:
            artifacts.cost("gate_plan", "reset", task_id=task.id, ok=False)
            artifacts.append("gate/planning_errors", {"task_id": task.id, "error": type(exc).__name__})
        finally:
            env.close()
    used, blocks, shortages = set(), [], []
    rng = random.Random(seed)
    for attempt in range(attempts):
        for policy in bank.policies:
            available = [r for r in public if r["group_id"] not in used]
            local = sorted((r for r in available if r["scores"][policy.id] > 0),
                           key=lambda r: (-r["scores"][policy.id], r["task_id"]))[:local_size]
            local_groups = {r["group_id"] for r in local}
            remaining = sorted((r for r in available if r["group_id"] not in local_groups),
                               key=lambda r: r["task_id"])
            rng.shuffle(remaining)
            global_rows = remaining[:gate_size-local_size]
            if len(local) < local_size or len(global_rows) < gate_size-local_size:
                shortages.append({"policy_id": policy.id, "attempt": attempt,
                                  "reason": "insufficient relevant or unused validation groups"})
                continue
            block = GateBlock(f"{policy.id}-{attempt}", policy.id, attempt,
                              tuple(r["task_id"] for r in local), tuple(r["task_id"] for r in global_rows))
            blocks.append(block)
            used.update(r["group_id"] for r in local + global_rows)
    value = {"version": "candidate-independent-gate-v1", "initial_bank_hash": bank.fingerprint,
             "seed": seed, "local_size": local_size, "global_size": gate_size-local_size,
             "max_candidates_per_policy": attempts, "public_tasks": public,
             "blocks": [asdict(b) for b in blocks], "shortages": shortages,
             "rule": "round-robin by attempt/policy; local positive fixed key+do BM25; random global; no group reuse"}
    artifacts.write("gate/plan", {**value, "plan_hash": digest(value)})
    return tuple(blocks)
