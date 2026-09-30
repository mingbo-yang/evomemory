import json

from evoscope.cli import bootstrap
from evoscope.core import Artifacts, Task
from evoscope.environments import ToyEnvironment
from evoscope.models import ToyModel
from evoscope.runner import Runner


def test_bootstrap_records_natural_source_and_decomposition_prompt(tmp_path):
    class Extractor(ToyModel):
        def call(self, system, data, phase, seed):
            if phase == "bootstrap_extract":
                assert "unconditional action core" in system
                assert "400 characters" in system
                assert "manufacture errors" in system
                assert len(data["trajectories"]) == 1
                return {"policies": [{"key": "place", "when": "The object is clean.",
                                      "do": "Place the object."}]}
            return super().call(system, data, phase, seed)
    artifacts = Artifacts(tmp_path / "run")
    runner = Runner(ToyEnvironment, Extractor(artifacts), artifacts, max_steps=3)
    tasks = [Task(s, s, s, "Place a clean object on the shelf.", {"clean": True})
             for s in ("bootstrap", "learn", "gate", "test")]
    bank = bootstrap(runner, tasks, 8, 11)
    provenance = json.loads((artifacts.root / "bootstrap/provenance.json").read_text())
    assert provenance["task_ids"] == ["bootstrap"]
    assert provenance["bank_hash"] == bank.fingerprint
    assert provenance["condition_lengths"]["max"] <= 400
    assert not (artifacts.root / "test").exists()
