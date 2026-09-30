import json
import sys
from types import SimpleNamespace

import pytest

from evoscope.core import Artifacts
from evoscope.models import APIModel, DEFAULT_API_BASE, DEFAULT_MODEL, edit_condition


def test_editor_compatible_with_json_mode_endpoint(tmp_path, monkeypatch):
    def create(**kwargs):
        assert kwargs["response_format"] == {"type": "json_object"}
        assert "json" in kwargs["messages"][0]["content"].lower()
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content='{"noop": true}'))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4), model="test")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kwargs:
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))))
    monkeypatch.setenv("EVOSCOPE_API_KEY", "fake-test-only")
    model = APIModel(Artifacts(tmp_path / "run"))
    assert edit_condition(model, {"key": "place", "when": "clean", "do": "place"}, [], 0) is None


def test_gateway_ignores_old_provider_environment(tmp_path, monkeypatch):
    credentials = tmp_path / "credentials.json"
    credentials.write_text(json.dumps({"api_key": "fake-local-test-key"}))
    monkeypatch.delenv("EVOSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("EVOSCOPE_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unrelated.invalid/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-unrelated-key")
    captured = {}
    def factory(**kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    model = APIModel(Artifacts(tmp_path / "run"))
    assert captured["base_url"] == DEFAULT_API_BASE
    assert captured["api_key"] == "fake-local-test-key"
    assert model.model == DEFAULT_MODEL
    assert "fake-local-test-key" not in json.dumps(model.config)
    with pytest.raises(ValueError, match="user-designated gateway"):
        APIModel(Artifacts(tmp_path / "rejected"), base_url="https://unrelated.invalid/v1")
