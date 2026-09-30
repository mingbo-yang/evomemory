from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from evoscope.cli import main, real_run
from evoscope.core import Artifacts, Bank, Policy, Task, atomic_json
from evoscope.engine import EvoScope
from evoscope.environments import ToyEnvironment
from evoscope.models import APIModel, DEFAULT_MAX_TOKENS, ToyModel
from evoscope.package_source import package_source
from evoscope.runner import RestoreError, Runner


def setup(tmp_path):
    artifacts = Artifacts(tmp_path / "run")
    bank = Bank([Policy("p", "place", "always", "Place the object on the shelf.")])
    tasks = [Task(f"l{i}", f"l{i}", "learn", "Place a clean object on the shelf.") for i in range(6)]
    tasks.append(Task("g", "g", "gate", "Place a clean object on the shelf."))
    runner = Runner(ToyEnvironment, ToyModel(artifacts), artifacts, max_steps=3)
    return EvoScope(bank, runner, tasks, artifacts, min_edit_checkpoints=1), runner, artifacts, tasks


def make_proposal(engine, runner, tasks):
    episode = runner.run(tasks[0], engine.bank, "learn", 0)
    evidence = engine.probe(episode.checkpoints[0], "p", repeats=1)
    return engine.propose("p", [evidence["id"]])


def test_gate_uses_all_three_pairs_not_lucky_first_pair(tmp_path, monkeypatch):
    engine, runner, _, tasks = setup(tmp_path)
    proposal = make_proposal(engine, runner, tasks)
    parent = engine.bank.fingerprint
    original_run = runner.run
    scores = {"old": iter([0, 1, 1]), "new": iter([1, 0, 0])}
    calls = []
    def varied(task, bank, phase, seed):
        label = "old" if bank.fingerprint == parent else "new"
        calls.append((label, seed))
        return replace(original_run(task, bank, phase, seed), score=next(scores[label]))
    monkeypatch.setattr(runner, "run", varied)
    receipt = engine.gate(proposal, ["g"], seed=10)
    assert receipt["paired_repeats"] == 3
    assert receipt["mean_delta"] == pytest.approx(-1 / 3)
    assert not receipt["accepted"] and engine.bank.fingerprint == parent
    assert calls == [("old", 10), ("new", 10), ("new", 14), ("old", 14), ("old", 18), ("new", 18)]
    assert len(receipt["results"][0]["pairs"]) == 3


def test_one_failed_repeat_invalidates_entire_gate(tmp_path, monkeypatch):
    engine, runner, _, tasks = setup(tmp_path)
    proposal = make_proposal(engine, runner, tasks)
    original = runner.run
    count = 0
    def fail_once(*args, **kwargs):
        nonlocal count
        count += 1
        episode = original(*args, **kwargs)
        return replace(episode, error="EndpointFailure") if count == 6 else episode
    monkeypatch.setattr(runner, "run", fail_once)
    receipt = engine.gate(proposal, ["g"])
    assert not receipt["accepted"] and receipt["mean_delta"] is None
    assert "g" in engine.used_gate_groups


def test_probe_total_budget_spans_batches_and_stops_collection(tmp_path):
    engine, _, _, _ = setup(tmp_path)
    result = engine.probe_only(batch_size=1, repeats=1, total_probes=2, probes_per_policy=3)
    assert result["attempts"] == 2
    assert result["first_run"]["episodes"] == 2
    assert result["learn_tasks_unvisited"] == 4
    assert result["probe_budget"]["used_per_policy"] == {"p": 2}
    assert result["stop_reason"] == "probe_budget_exhausted"


def test_probe_per_policy_budget_cannot_be_reset_each_batch(tmp_path):
    engine, _, _, _ = setup(tmp_path)
    result = engine.probe_only(batch_size=1, repeats=1, total_probes=24, probes_per_policy=3)
    assert result["attempts"] == 3 and result["first_run"]["episodes"] == 3


def test_failed_probes_consume_budget(tmp_path, monkeypatch):
    engine, runner, _, _ = setup(tmp_path)
    original = runner.run
    def fail_probe(task, bank, phase, *args, **kwargs):
        if phase == "probe":
            raise RestoreError("injected replay failure")
        return original(task, bank, phase, *args, **kwargs)
    monkeypatch.setattr(runner, "run", fail_probe)
    result = engine.probe_only(batch_size=1, repeats=3, total_probes=2)
    assert result["attempts"] == result["effects"]["unrecoverable"] == 2
    assert not engine.evidence


def test_zero_probe_budget_performs_no_model_calls(tmp_path):
    engine, _, artifacts, _ = setup(tmp_path)
    result = engine.probe_only(total_probes=0)
    assert result["attempts"] == 0 and not artifacts.costs


def test_probe_budget_is_allocated_round_robin(tmp_path):
    engine, _, _, _ = setup(tmp_path)
    engine.bank = Bank([Policy(p, "place", "always", "Place the object on the shelf.") for p in ("a", "b")])
    result = engine.probe_only(batch_size=4, repeats=1, total_probes=2, probes_per_policy=3)
    assert result["probe_budget"]["used_per_policy"] == {"a": 1, "b": 1}


def test_coverage_includes_unretrieved_and_seen_exclusions(tmp_path):
    engine, runner, artifacts, tasks = setup(tmp_path)
    engine.bank = Bank([Policy("p", "place", "The object is visibly clean.", "Place the object."),
                        Policy("absent", "astronomy", "always", "Compute orbital velocity")])
    episode = runner.run(tasks[0], engine.bank, "learn", 0)
    engine.record_coverage([episode])
    rows = [json.loads(line) for line in (artifacts.root / "probe/coverage.jsonl").read_text().splitlines()]
    rows = {row["policy_id"]: row for row in rows}
    assert rows["p"]["first_retrieved_step"] == 0
    assert rows["p"]["clean_checkpoint_count"] == 1
    assert rows["p"]["seen_excluded_count"] == 1
    assert rows["absent"]["first_retrieved_step"] is None
    assert rows["absent"]["clean_checkpoint_count"] == 0


def test_run_requires_pre_generated_bank_before_any_api_call():
    with pytest.raises(SystemExit) as exc:
        main(["run", "--manifest", "not-read.json", "--output", "not-created"])
    assert exc.value.code == 2


def test_m0_digest_mismatch_stops_before_client_creation(tmp_path, monkeypatch):
    import evoscope.cli as cli
    tasks = [Task("l", "l", "learn", "goal", {"gamefile": str(tmp_path / "game")})]
    (tmp_path / "game").write_text("fixture")
    bank_path = tmp_path / "bank.json"
    atomic_json(bank_path, Bank([Policy("p", "key", "when", "do")]).to_json())
    monkeypatch.setattr(cli, "load_manifest", lambda _: tasks)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(cli, "APIModel", lambda *a, **k: pytest.fail("API constructed"))
    args = SimpleNamespace(command="run", manifest="manifest", policies=str(bank_path),
                           expected_bank_hash="wrong", output=str(tmp_path / "output"))
    with pytest.raises(ValueError, match="hash mismatch"):
        real_run(args)
    assert not (tmp_path / "output").exists()


def test_cli_and_client_share_formal_output_budget(monkeypatch, capsys):
    import evoscope.cli as cli
    captured = {}
    monkeypatch.setattr(cli, "real_run", lambda args: captured.update(vars(args)) or {})
    assert main(["run", "--manifest", "m", "--policies", "p", "--output", "o"]) == 0
    assert captured["max_tokens"] == DEFAULT_MAX_TOKENS == 8192
    assert captured["gate_repeats"] == 3
    import inspect
    assert inspect.signature(APIModel).parameters["max_tokens"].default == 8192


def test_source_zip_excludes_credentials_results_and_symlinks(tmp_path):
    root = tmp_path / "project"
    package = root / "evoscope"
    (package / ".local").mkdir(parents=True)
    (package / ".local/api.json").write_text('{"api_key":"private-fixture"}')
    (package / "results").mkdir()
    (package / "results/private.py").write_text("private result")
    (package / ".env").write_text("secret fixture")
    (package / "core.py").write_text("pass\n")
    (package / "leak.py").symlink_to(package / ".local/api.json")
    output = tmp_path / "source.zip"
    package_source(root, output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"evomemory/evoscope/core.py", "evomemory/PACKAGE_MANIFEST.json"}
        assert b"private-fixture" not in b"".join(archive.read(n) for n in archive.namelist())
    assert (package / ".local/api.json").exists()


def test_gate_steps_and_acceptance_receipts_have_separate_logs(tmp_path):
    engine, runner, artifacts, tasks = setup(tmp_path)
    proposal = make_proposal(engine, runner, tasks)
    engine.gate(proposal, ["g"])
    decisions = [json.loads(s) for s in (artifacts.root / "gate/decisions.jsonl").read_text().splitlines()]
    steps = [json.loads(s) for s in (artifacts.root / "gate/steps.jsonl").read_text().splitlines()]
    assert len(decisions) == 1 and decisions[0]["paired_repeats"] == 3
    assert steps and all("step" in row and "accepted" not in row for row in steps)
