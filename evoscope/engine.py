"""One condition proposal per round, paired probing and independent bank gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import mean

from .core import Artifacts, Bank, Task, validate_tasks, condition_stats, digest
from .gate_plan import prepare_gate_plan
from .models import edit_condition
from .runner import Checkpoint, RestoreError, Runner, history


def spread_probe_schedule(rounds, policy_count, checkpoints, limits):
    """Allocate fixed slots across each policy's full sequence before any outcomes."""
    if not limits or not policy_count:return None
    total,per_policy=limits
    pools={p:[r for r in range(rounds) if r%policy_count==p for _ in range(checkpoints)] for p in range(policy_count)}
    quotas={p:0 for p in pools}
    for _ in range(total):
        eligible=[p for p in pools if quotas[p]<min(per_policy,len(pools[p]))]
        if not eligible:break
        p=min(eligible,key=lambda p:(quotas[p],p));quotas[p]+=1
    schedule={}
    for p,pool in pools.items():
        k=quotas[p]
        indices=[len(pool)//2] if k==1 else [i*(len(pool)-1)//(k-1) for i in range(k)] if k else []
        for idx in indices:
            r=pool[idx];schedule[r]=schedule.get(r,0)+1
    return schedule


def sample_checkpoints(episodes, target: str, limit: int, accumulated=None):
    """Balance across rounds, filling shortages without using memory effect labels."""
    pools = {"success": [], "failure": []}
    for ep in episodes:
        if ep.error:
            continue
        clean = [cp for cp in ep.checkpoints
                 if target in cp.selected_ids and target not in cp.seen_ids]
        if clean:
            pools["success" if ep.score >= 1 else "failure"].append(clean[0])
    available = {k: len(v) for k, v in pools.items()}
    prior = dict(accumulated or {"success": 0, "failure": 0})
    selected, counts = [], {"success": 0, "failure": 0}
    while len(selected) < limit and any(pools.values()):
        totals = {k: prior[k] + counts[k] for k in counts}
        tie_first = "success" if (sum(totals.values()) // 2) % 2 == 0 else "failure"
        name = min((k for k in pools if pools[k]), key=lambda k: (totals[k], k != tie_first))
        selected.append(pools[name].pop(0))
        counts[name] += 1
    return selected, {"available": available, "selected": counts,
                      "cumulative_before": prior,
                      "rule": "least-sampled stratum per policy/version; alternate ties; fill shortages"}


def checkpoint_coverage(episode, policy_id: str) -> dict:
    retrieved = [cp for cp in episode.checkpoints if policy_id in cp.selected_ids]
    clean = [cp for cp in retrieved if policy_id not in cp.seen_ids]
    return {"task_id": episode.task_id, "policy_id": policy_id,
            "bank_hash": episode.bank_hash,
            "first_retrieved_step": len(retrieved[0].actions) if retrieved else None,
            "retrieved_checkpoint_count": len(retrieved),
            "clean_checkpoint_count": len(clean),
            "seen_excluded_count": len(retrieved) - len(clean),
            "probe_eligible_clean_count": 0 if episode.error else len(clean),
            "episode_error": episode.error, "episode_success": episode.score >= 1 and not episode.error}


@dataclass(frozen=True)
class Proposal:
    parent_hash: str
    policy_id: str
    when: str
    evidence_ids: tuple[str, ...]


class EvoScope:
    def __init__(self, bank: Bank, runner: Runner, tasks: list[Task], artifacts: Artifacts,
                 *, min_edit_checkpoints: int = 3, min_non_neutral: int = 1):
        validate_tasks(tasks)
        if min_edit_checkpoints < 1 or min_non_neutral < 1 or min_non_neutral > min_edit_checkpoints:
            raise ValueError("invalid multi-state evidence thresholds")
        self.min_edit_checkpoints = min_edit_checkpoints
        self.min_non_neutral = min_non_neutral
        self.sampling_counts = {}
        self.sample_strata = {}
        self.edited_evidence_sets = set()
        self.gate_blocks = None
        self.attempted_gate_blocks = set()
        self.bank = bank
        self.runner = runner
        self.tasks = {t.id: t for t in tasks}
        self.artifacts = artifacts
        self.used_gate_groups: set[str] = set()
        self.evidence: dict[str, dict] = {}
        self.probe_attempts: list[dict] = []
        self.probe_only_active = False
        self.probe_limits = None
        self.artifacts.write("manifest", [asdict(t) for t in tasks])
        self.artifacts.write("initial_bank", bank.to_json())
        self.artifacts.write("bank", bank.to_json())

    def record_coverage(self, episodes) -> None:
        for episode in episodes:
            for policy in self.bank.policies:
                self.artifacts.append("probe/coverage", checkpoint_coverage(episode, policy.id))

    def sample(self, episodes, policy, limit):
        key = (policy.id, policy.version)
        chosen, audit = sample_checkpoints(episodes, policy.id, limit, self.sampling_counts.get(key))
        outcomes = {e.task_id: "success" if e.score >= 1 else "failure" for e in episodes}
        for cp in chosen:
            self.sample_strata[(cp.id, policy.id)] = outcomes[cp.task_id]
        return chosen, {**audit, "policy_version": policy.version}

    def compatible_evidence(self, policy_id, evidence_ids=None):
        policy = self.bank.get(policy_id)
        rows = list(self.evidence.values()) if evidence_ids is None else [self.evidence[e] for e in evidence_ids]
        valid = []
        for row in rows:
            compatible = (row["policy_id"] == policy_id and row["policy_version"] == policy.version
                          and row["bank_hash"] == self.bank.fingerprint
                          and row["model_fingerprint"] == self.runner.model.fingerprint
                          and row["runner_config"] == {"max_steps": self.runner.max_steps, "top_k": self.runner.top_k})
            if not compatible:
                if evidence_ids is not None:
                    raise ValueError("evidence target/version/configuration mismatch")
                continue
            valid.append(row)
        # One state per independent task group; repeats/duplicate public states
        # never inflate the minimum distinct-state count. Keep all effects.
        groups, states, selected = set(), set(), []
        protocol = valid[-1]["environment_protocol"] if valid else None
        for row in valid:
            group = self.tasks[row["task_id"]].group_id
            state = row["state_hash"]
            if row["environment_protocol"] != protocol or group in groups or state in states:
                continue
            groups.add(group); states.add(state); selected.append(row)
        return selected

    def edit_readiness(self, policy_id, evidence_ids=None):
        rows = self.compatible_evidence(policy_id, evidence_ids)
        signed = sum(r["public"]["direction"] in {"helpful", "harmful"} for r in rows)
        signature = digest({"policy_id": policy_id, "bank_hash": self.bank.fingerprint,
                            "states": sorted(r["state_hash"] for r in rows)})
        reason = ("insufficient_distinct_states" if len(rows) < self.min_edit_checkpoints else
                  "insufficient_stable_nonzero_evidence" if signed < self.min_non_neutral else
                  "evidence_already_used" if signature in self.edited_evidence_sets else None)
        return rows, signature, reason

    def probe_budget_available(self, policy_id: str | None = None) -> bool:
        if self.probe_limits is None:
            return True
        total, per_policy = self.probe_limits
        if len(self.probe_attempts) >= total:
            return False
        counts = {p.id: sum(a["policy_id"] == p.id for a in self.probe_attempts)
                  for p in self.bank.policies}
        return (counts[policy_id] < per_policy if policy_id is not None else
                any(count < per_policy for count in counts.values()))

    def probe(self, checkpoint: Checkpoint, policy_id: str,
              repeats: int = 3, seed: int = 1000) -> dict | None:
        task = self.tasks[checkpoint.task_id]
        if task.split != "learn":
            raise ValueError("only learn checkpoints may enter probe evidence")
        if repeats < 1:
            raise ValueError("positive repeat count required")
        if not self.probe_budget_available(policy_id):
            self.artifacts.append("probe/budget_skips", {"policy_id": policy_id,
                "checkpoint_id": checkpoint.id, "reason": "probe budget exhausted"})
            return None
        evidence_id = f"evidence-{len(self.probe_attempts):06d}"
        attempt = {"id": evidence_id, "checkpoint_id": checkpoint.id, "policy_id": policy_id,
                   "policy_version": self.bank.get(policy_id).version,
                   "bank_hash": self.bank.fingerprint, "model_fingerprint": self.runner.model.fingerprint,
                   "seed": seed, "environment_seed": task.seed, "repeats": repeats,
                   "seed_sent_to_api": bool(getattr(self.runner.model, "send_seed", False))}
        stratum = self.sample_strata.get((checkpoint.id, policy_id))
        if stratum:
            counts = self.sampling_counts.setdefault((policy_id, self.bank.get(policy_id).version),
                                                     {"success": 0, "failure": 0})
            counts[stratum] += 1  # Count attempts, including failed probes, before effects exist.
            attempt.update(sample_stratum=stratum, cumulative_sampling=dict(counts))
        self.probe_attempts.append(attempt)
        cost_start = len(self.artifacts.costs)
        pairs, executions = [], []
        failure_kind = "unrecoverable"
        try:
            for i in range(repeats):
                outcomes = {}
                for branch in (("expose", "mask") if i % 2 == 0 else ("mask", "expose")):
                    branch_start = len(self.artifacts.costs)
                    previous_rollout = self.runner.rollout_count
                    try:
                        episode = self.runner.run(task, self.bank, "probe", seed + i * 1000,
                                                  checkpoint, policy_id, branch, evidence_id=evidence_id)
                    finally:
                        executions.append({"repeat": i, "branch": branch, "seed": seed + i * 1000,
                            "rollout_id": (f"rollout-{self.runner.rollout_count:06d}"
                                           if self.runner.rollout_count != previous_rollout else None),
                            "costs": self.artifacts.cost_summary(branch_start)})
                    if episode.error or not math.isfinite(episode.score):
                        failure_kind = "execution_error"
                        raise RestoreError("invalid probe rollout")
                    outcomes[branch] = {"score": episode.score, "done": episode.done,
                                       "continuation": episode.public_history()[len(checkpoint.views):]}
                pairs.append(outcomes)
        except RestoreError as exc:
            attempt.update(status=failure_kind, reason=str(exc), executions=executions, completed_pairs=pairs,
                           costs=self.artifacts.cost_summary(cost_start))
            self.artifacts.append("probe/skips", attempt)
            self.artifacts.append("probe/attempts", attempt)
            return None
        deltas = [p["expose"]["score"] - p["mask"]["score"] for p in pairs]
        direction = ("helpful" if all(d > 0 for d in deltas) else
                     "harmful" if all(d < 0 for d in deltas) else
                     "neutral" if all(d == 0 for d in deltas) else "unstable")
        attempt.update(status=direction, executions=executions, costs=self.artifacts.cost_summary(cost_start))
        self.artifacts.append("probe/attempts", attempt)
        # Editor-facing public evidence deliberately excludes task id/payload/group.
        public = {"goal": checkpoint.resolved_goal, "history": history(checkpoint.views, checkpoint.actions),
                  "pairs": pairs, "mean_delta": mean(deltas), "direction": direction}
        row = {**attempt, "id": evidence_id, "checkpoint_id": checkpoint.id,
               "policy_id": policy_id, "bank_hash": self.bank.fingerprint,
               "task_id": task.id, "public": public,
               "state_hash": digest({"goal": public["goal"], "history": public["history"]}),
               "runner_config": {"max_steps": self.runner.max_steps, "top_k": self.runner.top_k},
               "environment_protocol": checkpoint.environment_protocol}
        self.evidence[evidence_id] = row
        self.artifacts.append("probe/evidence", row)
        return row

    def propose(self, policy_id: str, evidence_ids: list[str] | None = None, seed: int = 0) -> Proposal | None:
        if self.probe_only_active:
            raise RuntimeError("editing is forbidden in probe-only mode")
        rows, signature, reason = self.edit_readiness(policy_id, evidence_ids)
        self.artifacts.append("edit/eligibility", {"policy_id": policy_id,
            "policy_version": self.bank.get(policy_id).version, "bank_hash": self.bank.fingerprint,
            "distinct_states": len(rows), "effects": [r["public"]["direction"] for r in rows],
            "evidence_ids": [r["id"] for r in rows], "reason": reason})
        if reason:
            return None
        self.edited_evidence_sets.add(signature)  # No retry on noop, invalid output or transport failure.
        evidence_ids = [r["id"] for r in rows]
        policy = self.bank.get(policy_id)
        try:
            when = edit_condition(self.runner.model, asdict(policy), [r["public"] for r in rows], seed)
        except Exception as exc:
            self.artifacts.append("edit/rejections", {"policy_id": policy_id,
                                                       "reason": type(exc).__name__})
            return None
        if when is None or when == policy.when:
            return None
        proposal = Proposal(self.bank.fingerprint, policy_id, when, tuple(evidence_ids))
        self.artifacts.append("edit/proposals", asdict(proposal))
        return proposal

    def gate(self, proposal: Proposal, task_ids: list[str], seed: int = 20000,
             dry_run: bool = False, repeats: int = 3, block=None) -> dict:
        if self.probe_only_active:
            raise RuntimeError("gating is forbidden in probe-only mode")
        if repeats < 1:
            raise ValueError("positive gate repeat count required")
        if proposal.parent_hash != self.bank.fingerprint:
            raise ValueError("stale proposal")
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("gate requires distinct tasks")
        tasks = [self.tasks[t] for t in task_ids]
        groups = {t.group_id for t in tasks}
        if len(groups) != len(tasks):
            raise ValueError("gate requires one task per group")
        if any(t.split != "gate" for t in tasks) or groups & self.used_gate_groups:
            raise ValueError("gate tasks must be fresh reserved groups")
        if not proposal.evidence_ids:
            raise ValueError("proposal requires learn probe evidence")
        for evidence_id in proposal.evidence_ids:
            row = self.evidence[evidence_id]
            if row["bank_hash"] != proposal.parent_hash or row["policy_id"] != proposal.policy_id:
                raise ValueError("proposal evidence mismatch")
        if self.gate_blocks is not None:
            if (block not in self.gate_blocks or block.policy_id != proposal.policy_id
                    or tuple(task_ids) != block.task_ids):
                raise ValueError("gate tasks must match their precommitted block")
        local_ids = set(block.local_ids if block else task_ids)
        candidate = self.bank.revise(proposal.policy_id, proposal.when, proposal.evidence_ids)
        parent = self.bank.fingerprint
        # Consume BEFORE executing: failed runs cannot silently reuse validation tasks.
        self.used_gate_groups.update(groups)
        self.artifacts.write("gate/consumed_groups", sorted(self.used_gate_groups))
        results = []
        for i, task in enumerate(tasks):
            pairs = []
            for repeat_index in range(repeats):
                pair_seed = seed + (i * repeats + repeat_index) * (self.runner.max_steps + 1)
                pair = {"repeat": repeat_index, "seed": pair_seed}
                order = (("old", self.bank), ("new", candidate))
                if (i + repeat_index) % 2:
                    order = tuple(reversed(order))
                pair["order"] = [label for label, _ in order]
                for label, bank in order:
                    episode = self.runner.run(task, bank, "gate", pair_seed)
                    finite = math.isfinite(episode.score)
                    pair[label] = {"score": episode.score if finite else None,
                        "error": episode.error or (None if finite else "NonFiniteScore"),
                        "rollout_id": f"rollout-{self.runner.rollout_count:06d}",
                        "target_retrieval_count": episode.retrieval_counts.get(proposal.policy_id, 0),
                        "target_exposure_count": episode.exposure_counts.get(proposal.policy_id, 0)}
                pairs.append(pair)
            row = {"task_id": task.id, "group_id": task.group_id, "pairs": pairs,
                   "stratum": "local" if task.id in local_ids else "global"}
            for label in ("old", "new"):
                errors = [pair[label]["error"] for pair in pairs if pair[label]["error"]]
                row[label] = {"score": None if errors else mean(pair[label]["score"] for pair in pairs),
                              "error": errors[0] if errors else None}
            row["mean_delta"] = (None if row["old"]["error"] or row["new"]["error"] else
                                 mean(pair["new"]["score"] - pair["old"]["score"] for pair in pairs))
            results.append(row)
        valid = all(row["mean_delta"] is not None for row in results)
        # Each independent task/group has equal weight, regardless of repeats.
        delta = mean(row["mean_delta"] for row in results) if valid else None
        coverage = {stratum: {label: {
            "retrieval_count": sum(pair[label]["target_retrieval_count"] for row in results
                                   if row["stratum"] == stratum for pair in row["pairs"]),
            "exposure_count": sum(pair[label]["target_exposure_count"] for row in results
                                  if row["stratum"] == stratum for pair in row["pairs"])}
            for label in ("old", "new")} for stratum in ("local", "global")}
        informative = any(coverage["local"][side]["exposure_count"] > 0 for side in ("old", "new"))
        stratum_deltas = {s: (mean(r["mean_delta"] for r in results if r["stratum"] == s)
                             if valid and any(r["stratum"] == s for r in results) else None)
                          for s in ("local", "global")}
        would_accept = valid and informative and delta > 0 and self.bank.fingerprint == parent
        accepted = would_accept and not dry_run
        receipt = {"parent_hash": parent, "candidate_hash": candidate.fingerprint,
                   "accepted": accepted, "would_accept": would_accept, "dry_run": dry_run,
                   "mean_delta": delta, "results": results, "paired_repeats": repeats,
                   "gate_block_id": block.id if block else None, "target_coverage": coverage,
                   "stratum_mean_deltas": stratum_deltas,
                   "status": "execution_error" if not valid else "coverage_insufficient" if not informative else
                             "accepted" if accepted else "dry_run" if would_accept else "no_positive_gain",
                   "aggregation": "mean of within-task paired mean differences",
                   "proposal": asdict(proposal)}
        self.artifacts.append("gate/decisions", receipt)
        if accepted:
            self.bank = candidate
            self.artifacts.write("bank", self.bank.to_json())
        return receipt

    def probe_only(self, batch_size: int = 8, probe_checkpoints: int = 4,
                   repeats: int = 3, seed: int = 0, total_probes: int = 24,
                   probes_per_policy: int = 3) -> dict:
        if min(batch_size, probe_checkpoints, repeats) < 1:
            raise ValueError("positive batch sizes/repeats required")
        if min(total_probes, probes_per_policy) < 0:
            raise ValueError("probe budgets must be nonnegative")
        if self.evidence or self.used_gate_groups or self.probe_attempts:
            raise ValueError("probe-only requires a fresh engine and fixed M0")
        self.probe_only_active = True
        self.probe_limits = (total_probes, probes_per_policy)
        frozen = self.bank.fingerprint
        learn = [t for t in self.tasks.values() if t.split == "learn"]
        first = []
        for round_index, offset in enumerate(range(0, len(learn), batch_size)):
            if not self.probe_budget_available():
                break
            episodes = [self.runner.run(t, self.bank, "learn", seed + offset + i)
                        for i, t in enumerate(learn[offset:offset + batch_size])]
            first.extend(episodes)
            self.record_coverage(episodes)
            queues, sampling, selected = {}, {}, {}
            for policy in self.bank.policies:
                remaining = probes_per_policy - sum(a["policy_id"] == policy.id for a in self.probe_attempts)
                queues[policy.id], sampling[policy.id] = self.sample(
                    episodes, policy, min(probe_checkpoints, max(0, remaining)))
                selected[policy.id] = []
            # Fixed round-robin allocation; never prioritize by observed effect sign.
            for slot in range(probe_checkpoints):
                for policy in self.bank.policies:
                    queue = queues[policy.id]
                    if slot < len(queue) and self.probe_budget_available(policy.id):
                        cp = queue[slot]
                        selected[policy.id].append(cp)
                        self.probe(cp, policy.id, repeats,
                                   seed + 100000 + len(self.probe_attempts) * repeats * 1000)
            outcomes = {ep.task_id: ep.score >= 1 for ep in episodes}
            for policy in self.bank.policies:
                coverage = sampling[policy.id]
                coverage["selected"] = {
                    "success": sum(outcomes[cp.task_id] for cp in selected[policy.id]),
                    "failure": sum(not outcomes[cp.task_id] for cp in selected[policy.id])}
                self.artifacts.append("probe/sampling", {"round": round_index,
                    "policy_id": policy.id, **coverage})
            if self.bank.fingerprint != frozen:
                raise RuntimeError("M0 changed during probe-only")
        statuses = ("helpful", "harmful", "neutral", "unstable", "unrecoverable", "execution_error")
        counts = {s: sum(a["status"] == s for a in self.probe_attempts) for s in statuses}
        per_policy = {p.id: {s: sum(a["policy_id"] == p.id and a["status"] == s
                                  for a in self.probe_attempts) for s in statuses}
                      for p in self.bank.policies}
        result = {"mode": "probe-only", "initial_bank_hash": frozen,
                  "final_bank_hash": self.bank.fingerprint, "first_run": summarize(first),
                  "attempts": len(self.probe_attempts), "effects": counts,
                  "probe_budget": {"unit": "policy-task-checkpoint attempt; failures included",
                      "total_limit": total_probes, "per_policy_limit": probes_per_policy,
                      "used": len(self.probe_attempts),
                      "used_per_policy": {p.id: sum(a["policy_id"] == p.id for a in self.probe_attempts)
                                          for p in self.bank.policies},
                      "paired_repeats": repeats},
                  "learn_tasks_unvisited": len(learn) - len(first),
                  "stop_reason": "task_list_exhausted" if len(first) == len(learn) else "probe_budget_exhausted",
                  "per_policy": per_policy,
                  "sampling_counts": [{"policy_id": p, "policy_version": v, **c}
                                      for (p, v), c in self.sampling_counts.items()],
                  "policies_with_both_signs": [p for p, c in per_policy.items()
                                               if c["helpful"] and c["harmful"]],
                  "condition_lengths": condition_stats(self.bank),
                  "costs": self.artifacts.cost_summary()}
        self.artifacts.write("summary", result)
        return result

    def evolve(self, batch_size: int = 8, probe_checkpoints: int = 4,
               repeats: int = 3, gate_size: int = 4, seed: int = 0,
               gate_repeats: int = 3, gate_local_size: int | None = None,
               max_candidates_per_policy: int = 2) -> dict:
        if self.probe_only_active:
            raise RuntimeError("evolution is forbidden in probe-only mode")
        if min(batch_size, probe_checkpoints, repeats, gate_size, gate_repeats) < 1:
            raise ValueError("all batch sizes/repeats must be positive")
        if self.probe_limits is not None and min(self.probe_limits) < 0:
            raise ValueError("probe budgets must be nonnegative")
        learn = [t for t in self.tasks.values() if t.split == "learn"]
        gate = [t for t in self.tasks.values() if t.split == "gate"]
        first = []
        receipts = []
        if self.gate_blocks is not None:
            raise ValueError("evolve requires a fresh engine")
        local_size = max(1, gate_size // 2) if gate_local_size is None else gate_local_size
        self.gate_blocks = prepare_gate_plan(self.bank, gate, self.runner, self.artifacts,
                                             gate_size, local_size, max_candidates_per_policy, seed)
        self.artifacts.write("evolution_protocol", {"version": "evoscope-protocol-v3",
            "min_edit_checkpoints": self.min_edit_checkpoints, "min_non_neutral": self.min_non_neutral,
            "probe_limits": self.probe_limits, "gate_size": gate_size, "gate_local_size": local_size,
            "max_candidates_per_policy": max_candidates_per_policy,
            "sampling_unit": "policy_id,policy_version", "evidence_context": "bank_hash,model,runner,environment"})
        schedule=spread_probe_schedule(math.ceil(len(learn)/batch_size),len(self.bank.policies),probe_checkpoints,self.probe_limits)
        self.artifacts.write('probe/schedule',{'version':'evenly-spaced-rounds-v1','round_slots':schedule,
            'limits':self.probe_limits,'unavailable_slots':'record and leave unused; never redistribute based on effect'})
        for round_index, offset in enumerate(range(0, len(learn), batch_size)):
            episodes = [self.runner.run(t, self.bank, "learn", seed + offset + i)
                        for i, t in enumerate(learn[offset:offset + batch_size])]
            first.extend(episodes)
            self.record_coverage(episodes)
            if not self.bank.policies:
                continue
            target = self.bank.policies[round_index % len(self.bank.policies)].id
            slots=probe_checkpoints if schedule is None else schedule.get(round_index,0)
            clean, coverage = self.sample(episodes, self.bank.get(target), slots)
            self.artifacts.append("probe/sampling", {"round": round_index, "policy_id": target, "scheduled_slots":slots, **coverage})
            rows = [self.probe(cp, target, repeats, seed + 100000 + round_index * 10000 + i * 100)
                    for i, cp in enumerate(clean)]
            _, _, reason = self.edit_readiness(target)
            if reason:
                self.propose(target, seed=seed + round_index)  # Logs why collection continues; no editor call.
                continue
            block = next((b for b in self.gate_blocks if b.policy_id == target
                          and b.id not in self.attempted_gate_blocks), None)
            if block is None:
                self.artifacts.append("gate/skips", {"round": round_index, "policy_id": target,
                    "reason": "precommitted gate budget or relevant task coverage exhausted"})
                continue
            self.attempted_gate_blocks.add(block.id)
            self.artifacts.append("gate/block_assignments", {"round": round_index,
                "policy_id": target, "block_id": block.id, "before_editor": True})
            proposal = self.propose(target, seed=seed + round_index)
            if proposal:
                receipts.append(self.gate(proposal, list(block.task_ids),
                                          seed + 200000 + round_index * 10000,
                                          repeats=gate_repeats, block=block))
        summary = {"first_run": summarize(first), "gate_attempts": len(receipts), "gate_repeats": gate_repeats,
                   "protocol": "evoscope-protocol-v3",
                   "planned_gate_blocks": len(self.gate_blocks),
                   "editor_slots_used": len(self.attempted_gate_blocks),
                   "sampling_counts": [{"policy_id": p, "policy_version": v, **c}
                                       for (p, v), c in self.sampling_counts.items()],
                   "uninformative_gates": sum(r["status"] == "coverage_insufficient" for r in receipts),
                   "accepted_edits": sum(r["accepted"] for r in receipts),
                   "final_bank_hash": self.bank.fingerprint,
                   "condition_lengths": condition_stats(self.bank),
                   "costs": self.artifacts.cost_summary()}
        self.artifacts.write("summary", summary)
        return summary


def summarize(episodes) -> dict:
    valid = [e for e in episodes if not e.error]
    return {"episodes": len(episodes), "valid_episodes": len(valid),
            "errors": len(episodes) - len(valid),
            "mean_score": mean(e.score for e in valid) if valid else None,
            "success_rate": mean(float(e.score >= 1) for e in valid) if valid else None,
            "success_rate_with_errors_as_failures":
                sum(e.score >= 1 for e in valid) / len(episodes) if episodes else None}
