from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from evoscope.core import Artifacts, Bank, Policy, Task, atomic_json, validate_tasks
from evoscope.engine import EvoScope
from evoscope.environments import ToyEnvironment, View
from evoscope.models import EDITOR_PROMPT, ToyModel, edit_condition
from evoscope.provider import EvoScopeProvider
from evoscope.runner import RestoreError, Runner


def setup(tmp_path, when="always", env=ToyEnvironment, model_cls=ToyModel):
    artifacts = Artifacts(tmp_path / "run")
    model = model_cls(artifacts)
    runner = Runner(env, model, artifacts, max_steps=3)
    bank = Bank([Policy("p", "place clean shelf", when, "Place the object on the shelf.")])
    tasks = [Task(s, s, s, "Place a clean object on the shelf.", {"clean": False})
             for s in ("learn", "gate", "test")]
    engine = EvoScope(bank, runner, tasks, artifacts, min_edit_checkpoints=1)
    return engine, runner, artifacts, tasks


def proposal(engine, runner, tasks):
    episode = runner.run(tasks[0], engine.bank, "learn", 0)
    row = engine.probe(episode.checkpoints[0], "p", repeats=3)
    return engine.propose("p", [row["id"]])


def test_end_to_end_accepted_repair_preserves_action_core(tmp_path):
    engine, runner, artifacts, tasks = setup(tmp_path)
    initial = engine.bank.get("p")
    summary = engine.evolve(batch_size=1, probe_checkpoints=1, repeats=3, gate_size=1)
    updated = engine.bank.get("p")
    assert summary["first_run"]["success_rate"] == 0
    assert summary["accepted_edits"] == 1
    assert updated.key == initial.key and updated.do == initial.do
    assert updated.version == 2 and updated.evidence_ids
    assert runner.run(tasks[2], engine.bank, "test", 0).score == 1
    assert Bank.load(artifacts.root / "bank.json").fingerprint == engine.bank.fingerprint
    assert all("test" not in r["task_id"] for r in engine.evidence.values())


def test_tie_rejected_and_bank_unchanged(tmp_path):
    engine, runner, artifacts, tasks = setup(tmp_path)
    candidate = proposal(engine, runner, tasks)
    original = engine.bank.fingerprint
    receipt = engine.gate(replace(candidate, when="always "), ["gate"])
    assert receipt["mean_delta"] == 0 and not receipt["accepted"]
    assert engine.bank.fingerprint == original
    assert Bank.load(artifacts.root / "bank.json").fingerprint == original


def test_gate_tasks_cannot_be_reused_or_replaced_with_test(tmp_path):
    engine, runner, _, tasks = setup(tmp_path)
    p = proposal(engine, runner, tasks)
    with pytest.raises(ValueError, match="fresh"):
        engine.gate(p, ["test"])
    engine.gate(replace(p, when="always "), ["gate"])
    with pytest.raises(ValueError, match="fresh"):
        engine.gate(p, ["gate"])


def test_stale_proposals_rejected(tmp_path):
    engine, runner, _, tasks = setup(tmp_path)
    p = proposal(engine, runner, tasks)
    engine.gate(p, ["gate"])
    with pytest.raises(ValueError, match="stale"):
        engine.gate(p, ["gate"])


def test_probe_does_not_mutate_bank_or_backfill(tmp_path):
    engine, runner, artifacts, tasks = setup(tmp_path)
    engine.bank = Bank([engine.bank.get("p"), Policy("q", "clean", "always", "Wait."),
                        Policy("r", "clean", "always", "Wait."), Policy("s", "clean", "always", "Wait.")])
    cp = runner.run(tasks[0], engine.bank, "learn", 0).checkpoints[0]
    original = engine.bank.fingerprint
    assert engine.probe(cp, "p", repeats=1)
    rows = [json.loads(s) for s in (artifacts.root / "probe/decisions.jsonl").read_text().splitlines()]
    expose = next(r for r in rows if r["intervention"] == "expose")
    mask = next(r for r in rows if r["intervention"] == "mask")
    assert {p["id"] for p in expose["policies"]} - {p["id"] for p in mask["policies"]} == {"p"}
    assert len(mask["policies"]) == 2
    assert "when" not in next(p for p in expose["policies"] if p["id"] == "p")
    assert engine.bank.fingerprint == original


def test_contaminated_or_drifted_checkpoint_rejected(tmp_path):
    engine, runner, _, tasks = setup(tmp_path)
    cp = runner.run(tasks[0], engine.bank, "learn", 0).checkpoints[0]
    for corrupt in (replace(cp, seen_ids=("p",)), replace(cp, actor_hash="wrong"),
                    replace(cp, bank_hash="wrong"), replace(cp, top_k=5),
                    replace(cp, views=(View("wrong", ()),))):
        with pytest.raises(RestoreError):
            runner.run(tasks[0], engine.bank, "probe", 0, corrupt, "p", "mask")


def test_later_checkpoint_replays_and_charges_prefix(tmp_path):
    engine, runner, artifacts, tasks = setup(tmp_path, when="The object is visibly clean.")
    episode = runner.run(tasks[0], engine.bank, "learn", 0)
    cp = replace(episode.checkpoints[1], seen_ids=())
    result = runner.run(tasks[0], engine.bank, "probe", 7, cp, "p", "expose")
    assert result.views[:len(cp.views)] == list(cp.views)
    assert result.actions[:len(cp.actions)] == list(cp.actions)
    assert artifacts.cost_summary()["probe"]["replay_steps"] == 1
    assert artifacts.cost_summary()["probe"]["environment_steps"] == 2


def test_probe_skips_restore_failure_without_evidence(tmp_path):
    engine, runner, artifacts, tasks = setup(tmp_path)
    cp = runner.run(tasks[0], engine.bank, "learn", 0).checkpoints[0]
    assert engine.probe(replace(cp, views=(View("wrong", ()),)), "p") is None
    assert engine.evidence == {}
    assert (artifacts.root / "probe/skips.jsonl").exists()


def test_retrieval_independent_of_condition_and_deterministic():
    bank = Bank([Policy("z", "clean", "always", "wash"), Policy("a", "place", "never", "put")])
    revised = bank.revise("z", "place place place", ())
    assert [p.id for p in bank.retrieve("place")] == [p.id for p in revised.retrieve("place")]
    assert [p.id for p in bank.retrieve("absent")] == []


@pytest.mark.parametrize("bad", [{"when": "clean", "do": "changed"}, {"when": ""},
                                {"noop": False}, {"when": 4}, {"when": "x" * 2001}])
def test_editor_schema_cannot_rewrite_core(bad):
    model = SimpleNamespace(call=lambda *args: bad)
    with pytest.raises(ValueError):
        edit_condition(model, {"key": "key", "when": "always", "do": "act"}, [], 0)


def test_editor_never_receives_private_ids_or_payload(tmp_path):
    class RecordingModel(ToyModel):
        def call(self, system, data, phase, seed):
            if system == EDITOR_PROMPT:
                self.editor_data = data
            return super().call(system, data, phase, seed)
    engine, runner, _, tasks = setup(tmp_path, model_cls=RecordingModel)
    tasks[0].payload["secret_gold"] = "HIDDEN_ANSWER"
    assert proposal(engine, runner, tasks)
    raw = json.dumps(runner.model.editor_data)
    for private in ("HIDDEN_ANSWER", "secret_gold", "task_id", "group_id", "bank_hash", "checkpoint_id"):
        assert private not in raw


def test_group_split_isolation_and_duplicate_ids():
    a = Task("a", "same-family", "learn", "goal")
    with pytest.raises(ValueError, match="crosses splits"):
        validate_tasks([a, Task("b", "same-family", "test", "goal")])
    with pytest.raises(ValueError, match="duplicate"):
        validate_tasks([a, a])


def test_duplicate_game_copies_detected(tmp_path):
    one, two = tmp_path / "a", tmp_path / "b"
    one.write_text("identical game")
    two.write_text("identical game")
    with pytest.raises(ValueError, match="duplicate ALFWorld"):
        validate_tasks([Task("a", "a", "learn", "g", {"gamefile": str(one)}),
                        Task("b", "b", "gate", "g", {"gamefile": str(two)})])


def test_test_checkpoints_cannot_be_probed(tmp_path):
    engine, runner, _, tasks = setup(tmp_path)
    cp = runner.run(tasks[0], engine.bank, "learn", 0).checkpoints[0]
    with pytest.raises(ValueError, match="only learn"):
        engine.probe(replace(cp, task_id="test"), "p")


def test_run_directory_exclusive(tmp_path):
    Artifacts(tmp_path / "run")
    with pytest.raises(FileExistsError):
        Artifacts(tmp_path / "run")


def test_provider_read_only_interface(tmp_path):
    bank = Bank([Policy("p", "place", "clean", "place")])
    path = tmp_path / "bank.json"
    atomic_json(path, bank.to_json())
    provider = EvoScopeProvider(str(path))
    assert provider.initialize()
    response = provider.provide_memory(SimpleNamespace(query="place", context=""))
    assert response.total_count == 1
    assert response.memories[0].content == "When: clean\nDo: place"
    assert provider.take_in_memory(None)[0] is False
    assert Bank.load(path).fingerprint == bank.fingerprint


def test_gate_execution_errors_reject_and_consume(tmp_path):
    class FailureModel(ToyModel):
        def call(self, system, data, phase, seed):
            if phase == "gate":
                raise RuntimeError("simulated endpoint failure")
            return super().call(system, data, phase, seed)
    engine, runner, _, tasks = setup(tmp_path, model_cls=FailureModel)
    p = proposal(engine, runner, tasks)
    original = engine.bank.fingerprint
    receipt = engine.gate(p, ["gate"])
    assert not receipt["accepted"] and receipt["mean_delta"] is None
    assert engine.bank.fingerprint == original and "gate" in engine.used_gate_groups


def test_uncertain_probe_not_forced_into_edit(tmp_path):
    engine, runner, _, tasks = setup(tmp_path)
    tasks[0].payload["clean"] = True
    cp = runner.run(tasks[0], engine.bank, "learn", 0).checkpoints[0]
    row = engine.probe(cp, "p", repeats=3)
    assert row["public"]["direction"] == "neutral"
    assert engine.propose("p", [row["id"]]) is None


def test_missing_gate_budget_does_not_install_proposal(tmp_path):
    engine, _, artifacts, _ = setup(tmp_path)
    original = engine.bank.fingerprint
    summary = engine.evolve(batch_size=1, probe_checkpoints=1, repeats=1, gate_size=2)
    assert summary["accepted_edits"] == 0 and engine.bank.fingerprint == original
    assert (artifacts.root / "gate/skips.jsonl").exists()


def test_dry_run_gate_never_commits_even_with_positive_gain(tmp_path):
    engine, runner, artifacts, tasks = setup(tmp_path)
    candidate = proposal(engine, runner, tasks)
    original = engine.bank.fingerprint
    receipt = engine.gate(candidate, ["gate"], dry_run=True)
    assert receipt["would_accept"] and receipt["mean_delta"] > 0
    assert not receipt["accepted"] and receipt["dry_run"]
    assert engine.bank.fingerprint == original
    assert Bank.load(artifacts.root / "bank.json").fingerprint == original
