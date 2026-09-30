from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from evoscope.core import Artifacts, Bank, Policy, Task, atomic_json
from evoscope.engine import EvoScope, sample_checkpoints
from evoscope.environments import ToyEnvironment, View
from evoscope.models import ToyModel, edit_condition
from evoscope.provider import EvoScopeProvider
from evoscope.runner import Runner, resolve_goal


def build(tmp_path, model_type=ToyModel):
    artifacts = Artifacts(tmp_path / "run")
    bank = Bank([Policy("p", "place", "always", "Place the object on the shelf.")])
    tasks = [Task(f"l{i}", f"l{i}", "learn", "Place a clean object on the shelf.",
                  {"clean": bool(i % 2)}) for i in range(4)]
    tasks += [Task("g1", "duplicate", "gate", "Place a clean object on the shelf."),
              Task("g2", "duplicate", "gate", "Place a clean object on the shelf.")]
    runner = Runner(ToyEnvironment, model_type(artifacts), artifacts, max_steps=3)
    return EvoScope(bank, runner, tasks, artifacts, min_edit_checkpoints=1), runner, artifacts, tasks


def test_zero_score_never_enters_context_or_seen_ids(tmp_path):
    artifacts = Artifacts(tmp_path / "run")
    bank = Bank([Policy("unrelated", "astronomy", "always", "Compute orbital velocity")])
    runner = Runner(ToyEnvironment, ToyModel(artifacts), artifacts, max_steps=3)
    episode = runner.run(Task("t", "t", "learn", "Place cup"), bank, "learn", 0)
    assert all(not cp.selected_ids and not cp.seen_ids for cp in episode.checkpoints)
    assert bank.retrieve("") == ()


def test_public_goal_persists_after_observation_changes(tmp_path):
    class GoalEnv:
        def reset(self, task):
            self.n = 0
            return View("Room.\nYour task is to: heat apple and place it.\nObjects nearby.", ("go",))
        def step(self, action):
            self.n += 1
            return View("Empty hallway.", ("go",), 0, self.n == 2)
        def close(self):
            pass
    class Actor:
        fingerprint = "test"
        def call(self, system, data, phase, seed):
            assert "heat apple" in data["goal"]
            return {"action": "go"}
    artifacts = Artifacts(tmp_path / "run")
    runner = Runner(GoalEnv, Actor(), artifacts, max_steps=2)
    bank = Bank([Policy("heat", "heat apple", "always", "Use heater")])
    ep = runner.run(Task("t", "t", "learn", "Follow instructions"), bank, "learn", 0)
    assert ep.error is None
    assert len(ep.checkpoints) == 2
    assert all(cp.selected_ids == ("heat",) for cp in ep.checkpoints)
    assert resolve_goal("generic", "unrecognized public task text") == "generic\nunrecognized public task text"


def test_probe_only_freezes_bank_and_never_edits_or_gates(tmp_path, monkeypatch):
    engine, _, artifacts, _ = build(tmp_path)
    original = engine.bank.fingerprint
    def forbidden(*args, **kwargs):
        pytest.fail("probe-only called evolution")
    monkeypatch.setattr(engine, "propose", forbidden)
    monkeypatch.setattr(engine, "gate", forbidden)
    result = engine.probe_only(batch_size=4, probe_checkpoints=4, repeats=2, probes_per_policy=4)
    assert result["initial_bank_hash"] == result["final_bank_hash"] == original
    assert result["effects"] == {"helpful": 0, "harmful": 2, "neutral": 2,
                                 "unstable": 0, "unrecoverable": 0, "execution_error": 0}
    assert not (artifacts.root / "edit").exists() and not (artifacts.root / "gate").exists()
    with pytest.raises(RuntimeError, match="forbidden"):
        engine.evolve()


def test_sampling_balances_and_fills_shortages():
    def episode(name, success):
        cp = SimpleNamespace(selected_ids=("p",), seen_ids=(), task_id=name)
        return SimpleNamespace(error=None, score=int(success), checkpoints=[cp, cp])
    episodes = [episode("f1", False), episode("f2", False), episode("f3", False), episode("s", True)]
    selected, coverage = sample_checkpoints(episodes, "p", 2)
    assert {cp.task_id for cp in selected} == {"s", "f1"}
    assert coverage["available"] == {"success": 1, "failure": 3}
    selected, coverage = sample_checkpoints(episodes, "p", 4)
    assert len({cp.task_id for cp in selected}) == 4
    assert coverage["selected"] == {"success": 1, "failure": 3}


def test_gate_rejects_duplicate_groups_before_consuming(tmp_path):
    engine, runner, _, tasks = build(tmp_path)
    cp = runner.run(tasks[0], engine.bank, "learn", 0).checkpoints[0]
    evidence = engine.probe(cp, "p", repeats=1)
    proposal = engine.propose("p", [evidence["id"]])
    with pytest.raises(ValueError, match="one task per group"):
        engine.gate(proposal, ["g1", "g2"])
    assert engine.used_gate_groups == set()


def test_evolution_gate_budget_counts_groups(tmp_path):
    engine, _, artifacts, _ = build(tmp_path)
    summary = engine.evolve(batch_size=4, probe_checkpoints=4, repeats=1, gate_size=2)
    assert summary["gate_attempts"] == 0
    assert (artifacts.root / "gate/skips.jsonl").exists()


def test_condition_limit_applies_to_initial_load_edit_and_revision(tmp_path):
    bank = Bank([Policy("p", "key", "x" * 400, "action")])
    with pytest.raises(ValueError):
        Policy("p", "key", "x" * 401, "action")
    with pytest.raises(ValueError):
        bank.revise("p", "x" * 401, ())
    with pytest.raises(ValueError):
        edit_condition(SimpleNamespace(call=lambda *args: {"when": "x" * 401}),
                       bank.to_json()[0], [], 0)
    path = tmp_path / "bad.json"
    atomic_json(path, [{"id": "p", "key": "key", "when": "x" * 401, "do": "act"}])
    with pytest.raises(ValueError):
        Bank.load(path)


def test_evidence_costs_join_exactly_to_ledger(tmp_path):
    engine, runner, artifacts, tasks = build(tmp_path)
    cp = runner.run(tasks[0], engine.bank, "learn", 0).checkpoints[0]
    row = engine.probe(cp, "p", repeats=2, seed=17)
    assert row["policy_version"] == 1 and row["model_fingerprint"] == runner.model.fingerprint
    assert row["seed"] == 17 and len(row["executions"]) == 4
    ledger = [r for r in artifacts.costs if r.get("evidence_id") == row["id"]]
    assert row["costs"]["probe"]["model_calls"] == sum(r["kind"] == "model" for r in ledger)
    assert row["costs"]["probe"]["environment_steps"] == sum(r["kind"] == "step" for r in ledger)
    assert all(r.get("rollout_id") and r.get("branch") for r in ledger)
    assert "model_fingerprint" not in row["public"]


def test_provider_refreshes_only_at_task_boundary(tmp_path):
    path = tmp_path / "bank.json"
    bank = Bank([Policy("p", "place", "old condition", "place")])
    atomic_json(path, bank.to_json())
    provider = EvoScopeProvider(str(path))
    provider.initialize()
    new = bank.revise("p", "new condition", ())
    atomic_json(path, new.to_json())
    request = SimpleNamespace(query="place", context="", status="in")
    assert "old condition" in provider.provide_memory(request).memories[0].content
    request.status = "begin"
    assert "new condition" in provider.provide_memory(request).memories[0].content


def test_expose_preserves_initial_target_rank(tmp_path):
    engine, runner, artifacts, tasks = build(tmp_path)
    engine.bank = Bank([Policy("a", "clean", "always", "Wait"), engine.bank.get("p"),
                        Policy("z", "shelf", "always", "Wait")])
    cp = runner.run(tasks[0], engine.bank, "learn", 0).checkpoints[0]
    rank = cp.selected_ids.index("p")
    engine.probe(cp, "p", repeats=1)
    rows = [json.loads(l) for l in (artifacts.root / "probe/decisions.jsonl").read_text().splitlines()]
    expose = next(r for r in rows if r["step"] == 0 and r["intervention"] == "expose")
    assert [p["id"] for p in expose["policies"]] == list(cp.selected_ids)
    assert expose["policies"][rank] == {"id": "p", "do": engine.bank.get("p").do}


def test_unstable_and_unrecoverable_are_separate(tmp_path, monkeypatch):
    engine, runner, _, tasks = build(tmp_path)
    cp = runner.run(tasks[0], engine.bank, "learn", 0).checkpoints[0]
    original = runner.run
    # Execution order is expose/mask then mask/expose: opposite effect signs.
    scores = iter([1, 0, 1, 0])
    def varied(*args, **kwargs):
        return replace(original(*args, **kwargs), score=next(scores))
    monkeypatch.setattr(runner, "run", varied)
    row = engine.probe(cp, "p", repeats=2)
    assert row["public"]["direction"] == "unstable"
    assert engine.propose("p", [row["id"]]) is None
    engine.probe(replace(cp, actor_hash="wrong"), "p", repeats=1)
    assert engine.probe_attempts[-1]["status"] == "unrecoverable"
