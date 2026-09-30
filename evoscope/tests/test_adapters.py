"""Contract tests use local fake endpoints; no paid API or benchmark calls."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from evoscope.core import Artifacts, Bank, Policy, Task
from evoscope.environments import AlfWorldEnvironment
from evoscope.models import APIModel


def test_bank_and_policy_are_immutable():
    policy = Policy("p", "key", "when", "do")
    bank = Bank([policy])
    with pytest.raises(FrozenInstanceError):
        bank.policies = ()
    with pytest.raises(FrozenInstanceError):
        policy.do = "different"


@pytest.mark.parametrize("mode", ["success", "invalid_json", "transport_error", "truncated"])
def test_api_usage_and_errors_are_accounted_without_secrets(tmp_path, monkeypatch, mode):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        if mode == "transport_error":
            raise RuntimeError("DO_NOT_LOG_API_KEY")
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="length" if mode == "truncated" else "stop",
                message=SimpleNamespace(
                content='{"action":"place"}' if mode in {"success", "truncated"} else 'bad json'))],
            usage=SimpleNamespace(prompt_tokens=23, completion_tokens=7),
            model="pinned-returned-model", system_fingerprint="backend-fingerprint")
    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    monkeypatch.setenv("EVOSCOPE_API_KEY", "DO_NOT_LOG_API_KEY")
    artifacts = Artifacts(tmp_path / "api")
    model = APIModel(artifacts, send_seed=True)
    if mode == "success":
        assert model.call("system", {"input": "data"}, "probe", 55) == {"action": "place"}
    else:
        with pytest.raises((ValueError, RuntimeError)):
            model.call("system", {}, "probe", 55)
    assert calls[0]["seed"] == 55
    costs = artifacts.cost_summary()["probe"]
    assert costs["model_calls"] == 1
    assert costs["failed_calls"] == int(mode != "success")
    assert costs["unknown_usage_calls"] == int(mode == "transport_error")
    assert costs["prompt_tokens"] == (0 if mode == "transport_error" else 23)
    assert "DO_NOT_LOG_API_KEY" not in (artifacts.root / "costs.jsonl").read_text()
    assert "DO_NOT_LOG_API_KEY" not in json.dumps(model.config)


def test_alfworld_adapter_requests_no_expert_and_resets_same_game(tmp_path, monkeypatch):
    registrations = []
    instances = []
    class FakeEnv:
        def seed(self, seed):
            self.seed_value = seed
        def reset(self):
            return ["Public observation"], {"admissible_commands": [["look", "take"]], "won": [False]}
        def step(self, actions):
            assert actions == ["take"]
            return ["Done"], [99], [True], {"admissible_commands": [[]], "won": [True]}
        def close(self):
            self.closed = True
    def register(games, infos, **kwargs):
        registrations.append((games, infos, kwargs))
        return "registered-game"
    def make(_):
        env = FakeEnv()
        instances.append(env)
        return env
    tw = ModuleType("textworld")
    gym = ModuleType("textworld.gym")
    gym.register_games, gym.make = register, make
    tw.gym = gym
    tw.EnvInfos = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "textworld", tw)
    monkeypatch.setitem(sys.modules, "textworld.gym", gym)
    for name in ("alfworld", "alfworld.agents", "alfworld.agents.environment"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    adapter = ModuleType("alfworld.agents.environment.alfred_tw_env")
    adapter.AlfredDemangler = lambda **kwargs: kwargs
    adapter.AlfredInfos = object
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    monkeypatch.setattr(AlfWorldEnvironment, "_registrations", {})
    game = tmp_path / "game.tw-pddl"
    game.write_text("fake fixture")
    task = Task("game", "family", "learn", "public goal", {"gamefile": str(game)}, seed=73)
    env = AlfWorldEnvironment(max_steps=50)
    first = env.reset(task)
    end = env.step("take")
    assert end.score == 1 and end.done  # use won, not arbitrary backend score 99
    assert env.reset(task) == first
    assert instances[0].closed and instances[1].seed_value == 73
    assert len(registrations) == 1
    assert registrations[0][1] == {"won": True, "admissible_commands": True, "extras": ["gamefile"]}
    assert registrations[0][2]["wrappers"][0] == {"shuffle": False}
    assert registrations[0][2]["asynchronous"] is False
    env.close()
    assert instances[1].closed


@pytest.mark.parametrize("fail_close", [False, True])
def test_alfworld_closes_planner_once_through_wrappers(monkeypatch, fail_close):
    released = []
    monkeypatch.setitem(sys.modules, "fast_downward", SimpleNamespace(close_lib=released.append))
    lib = object()
    planner = SimpleNamespace(downward_lib=lib)
    wrapper = SimpleNamespace(_wrapped_env=planner)
    batch = SimpleNamespace(envs=[wrapper, wrapper])
    def close():
        batch.envs.clear()
        if fail_close:
            raise RuntimeError("close failed")
    adapter = AlfWorldEnvironment()
    adapter.env = SimpleNamespace(batch_env=batch, close=close)
    if fail_close:
        with pytest.raises(RuntimeError):
            adapter.close()
    else:
        adapter.close()
    adapter.close()
    assert released == [lib]
    assert adapter.env is None and planner.downward_lib is None
