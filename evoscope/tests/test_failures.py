import json

from evoscope.core import Artifacts, Bank, Policy, Task
from evoscope.engine import EvoScope
from evoscope.environments import ToyEnvironment
from evoscope.models import ToyModel
from evoscope.runner import Runner


def test_editor_endpoint_failure_keeps_first_results_and_bank(tmp_path):
    class FailingEditor(ToyModel):
        def call(self, system, data, phase, seed):
            if phase == "edit":
                self.artifacts.cost("edit", "model", ok=False,
                                    prompt_tokens=None, completion_tokens=None)
                raise RuntimeError("do not persist private endpoint details")
            return super().call(system, data, phase, seed)
    artifacts = Artifacts(tmp_path / "run")
    bank = Bank([Policy("p", "place", "always", "Place the object on the shelf.")])
    tasks = [Task(s, s, s, "Place a clean object on the shelf.") for s in ("learn", "gate")]
    runner = Runner(ToyEnvironment, FailingEditor(artifacts), artifacts, max_steps=3)
    engine = EvoScope(bank, runner, tasks, artifacts, min_edit_checkpoints=1)
    result = engine.evolve(batch_size=1, repeats=1, gate_size=1)
    assert result["first_run"]["episodes"] == 1
    assert result["gate_attempts"] == 0 and engine.bank.fingerprint == bank.fingerprint
    assert result["costs"]["edit"]["failed_calls"] == 1
    assert "private endpoint" not in (artifacts.root / "edit/rejections.jsonl").read_text()


def test_paired_rollouts_share_initial_seed_and_alternate_order(tmp_path):
    artifacts = Artifacts(tmp_path / "run")
    bank = Bank([Policy("p", "place", "always", "Place the object on the shelf.")])
    task = Task("learn", "learn", "learn", "Place a clean object on the shelf.")
    runner = Runner(ToyEnvironment, ToyModel(artifacts), artifacts, max_steps=3)
    engine = EvoScope(bank, runner, [task], artifacts)
    checkpoint = runner.run(task, bank, "learn", 0).checkpoints[0]
    engine.probe(checkpoint, "p", repeats=3, seed=13)
    rows = [json.loads(line) for line in (artifacts.root / "probe/decisions.jsonl").read_text().splitlines()]
    starts = [(row["intervention"], row["seed"]) for row in rows if row["step"] == 0]
    assert starts == [("expose", 13), ("mask", 13), ("mask", 1013),
                      ("expose", 1013), ("expose", 2013), ("mask", 2013)]
