"""Configuration precedence and final-wire regressions for model parameters."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from sharedagents_agent.agent import AgentConfig
from sharedagents_command.commands import normalize_effort
from sharedagents_gateway.worker import forward_sampling_fields
from sharedagents_gateway.worker_request import _build_worker_request
from sharedagents_model import effort_prefs
from sharedagents_model.effort_discovery import clear_effort_discovery_cache, discover_effort_capabilities
from sharedagents_model.model_parameters import (
    ModelParametersError,
    clear_model_parameters_cache,
)
from sharedagents_model.models import derive_effort_capability
from sharedagents_model.sampling import configured_sampling
from sharedagents_session.session import Session


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.delenv("SHAREDAGENTS_AGENTS_HOME", raising=False)
    clear_model_parameters_cache()
    clear_effort_discovery_cache()
    yield tmp_path
    clear_model_parameters_cache()
    clear_effort_discovery_cache()


def write(home, entry):
    (home / "model_parameters.json").write_text(json.dumps({"version": 1, "models": [entry]}))
    clear_model_parameters_cache()


def entry(**changes):
    return {
        "provider": "vllm",
        "model": "example-model",
        "aliases": ["example-alias"],
        "effort": {"values": ["low", "custom_Depth"], "wire": "reasoning_effort"},
        **changes,
    }


def wire(model="example-model", effort="", *, backend="vllm", row=None, sampling=None, mode="auto"):
    row = row or {"id": model, "backend": backend, "baseUrl": "http://example.com/v1"}
    cfg = AgentConfig(name="test", models=[row])
    sess = Session(
        hash="config-test",
        title="Test",
        created_at="2026-11-15",
        system_prompt="Test",
        model=model,
        endpoint=row["baseUrl"],
        context_window=128000,
        system_mode=mode,
        agent_name="test",
        effort=effort,
    )
    return _build_worker_request(
        SimpleNamespace(cfg=cfg), sess, [{"role": "user", "content": "Hello"}], include_tools=False, sampling=sampling
    )


def test_unknown_local_model_never_passes_effort_through():
    payload = wire(effort="high")
    assert "reasoning_effort" not in forward_sampling_fields(payload)
    assert "extra" not in payload
    assert derive_effort_capability({"backend": "vllm", "id": "unknown"})["values"] == []


def test_served_aliases_have_identical_optional_payloads(home):
    declaration = entry(
        model="example3.8-model",
        aliases=["example38-model", "vendor/Example3.8-Model", "example38"],
        effort={"values": ["low", "medium", "xhigh"], "wire": "reasoning_effort"},
    )
    write(home, declaration)
    bodies = [wire(name, "medium") for name in [declaration["model"], *declaration["aliases"]]]
    for body in bodies:
        assert body.pop("model")
        assert body["reasoning_effort"] == "medium"
        assert "extra" not in body
    assert all(body == bodies[0] for body in bodies)


def test_unfamiliar_token_round_trips_commands_preferences_and_wire(home):
    write(home, entry())
    assert normalize_effort("custom_Depth") == "custom_Depth"
    row = {"backend": "vllm", "id": "example-model"}
    effort_prefs.set_row_pref(row, "custom_Depth")
    assert effort_prefs.get_row_pref(row) == "custom_Depth"
    assert derive_effort_capability(row)["values"] == ["low", "custom_Depth"]
    assert wire(effort="custom_Depth")["reasoning_effort"] == "custom_Depth"


def test_no_default_means_no_implicit_effort(home):
    write(home, entry())
    assert "reasoning_effort" not in wire()
    assert AgentConfig(name="test").effort == ""
    assert effort_prefs.selected_level_for({"backend": "vllm", "id": "example-model"}) == ""


def test_stale_selection_is_omitted_even_with_configured_default(home):
    configured = entry()
    configured["effort"]["default"] = "low"
    write(home, configured)
    assert wire()["reasoning_effort"] == "low"
    assert "reasoning_effort" not in wire(effort="high")


@pytest.mark.parametrize("disabled", [None, {"values": []}])
def test_disabled_entry_suppresses_shipped_and_provider_levels(home, disabled):
    write(home, entry(model="qwen3.8", effort=disabled))
    row = {"id": "qwen3.8", "backend": "vllm", "reasoning": {"supported_efforts": ["high"]}}
    cap = derive_effort_capability(row)
    assert cap["values"] == []
    found = discover_effort_capabilities([row], None, fetch=lambda *a: pytest.fail("must not probe"))
    assert found[0]["values"] == []
    assert "reasoning_effort" not in wire("qwen3.8", "high")
    assert "extra" not in wire("qwen3.8")


def test_whole_entry_replacement_removes_packaged_defaults(home):
    write(
        home,
        entry(model="qwen3.8", effort={"values": ["new"], "wire": "reasoning_effort"}, parameters={"temperature": 0}),
    )
    payload = wire("qwen3.8")
    assert payload["temperature"] == 0
    assert "extra" not in payload and "reasoning_effort" not in payload
    assert derive_effort_capability({"id": "qwen3.8", "backend": "vllm"})["values"] == ["new"]


def test_explicit_false_zero_null_and_mode_overrides(home):
    write(
        home,
        entry(
            parameters={"temperature": 0.4, "top_p": None, "chat_template_kwargs.enable_thinking": False},
            modes={"plan": {"parameters": {"temperature": 0}}},
        ),
    )
    payload = wire(mode="plan")
    assert payload["temperature"] == 0
    assert "top_p" not in payload
    assert payload["extra"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "temperature" not in wire(sampling={"temperature": None})


def test_budget_mapping_uses_only_one_declared_destination(home):
    write(
        home,
        entry(
            effort={
                "values": ["quick", "deep"],
                "default": "quick",
                "wire": "chat_template_kwargs.thinking_budget",
                "wire_values": {"quick": 0, "deep": 4096},
            }
        ),
    )
    payload = wire(effort="deep")
    assert payload["extra"] == {"chat_template_kwargs": {"thinking_budget": 4096}}
    assert "reasoning_effort" not in payload
    assert wire()["extra"]["chat_template_kwargs"]["thinking_budget"] == 0


def test_provider_scoping_and_no_substring_match(home):
    write(home, entry())
    assert "reasoning_effort" not in wire("prefix-example-model", "low")
    assert "reasoning_effort" not in wire(effort="low", backend="sglang")


def test_advertisement_is_capability_without_implicit_request_default():
    row = {
        "backend": "openai",
        "id": "other",
        "baseUrl": "https://example.com/v1",
        "reasoning": {"supported_efforts": ["novel", "low"], "default_effort": "novel"},
    }
    assert derive_effort_capability(row)["default"] == "novel"
    assert "reasoning_effort" not in wire("other", row=row)
    assert wire("other", "novel", row=row)["reasoning_effort"] == "novel"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(parameters={"messages": []}),
        lambda d: d.update(parameters={"temperature": float("nan")}),
        lambda d: d.update(aliases=["example-model"]),
        lambda d: d["effort"].update(default="missing"),
        lambda d: d["effort"].update(wire="headers.Authorization"),
        lambda d: d["effort"].update(wire="chat_template_kwargs.thinking_budget"),
        lambda d: d.update(provider="bedrock"),
    ],
)
def test_invalid_configuration_is_not_silently_replaced(home, mutation):
    configured = deepcopy(entry())
    mutation(configured)
    write(home, configured)
    with pytest.raises(ModelParametersError, match="model_parameters.json"):
        wire()


def test_malformed_file_is_reported(home):
    (home / "model_parameters.json").write_text("{ broken")
    with pytest.raises(ModelParametersError, match="cannot read valid JSON"):
        wire()


def test_side_sampling_can_omit_file_values(home):
    write(home, entry(parameters={"temperature": 0.5}))
    row = {"id": "example-model", "backend": "vllm"}
    assert configured_sampling({}, row, purpose="title") == {"temperature": 0.5}
    assert configured_sampling({"side_sampling": {"title": {"temperature": None}}}, row, purpose="title") == {}


@pytest.mark.asyncio
async def test_native_engine_accepts_configured_custom_tokens(home):
    from sharedagents_agent.agents.local_llm import LocalLLMEngine

    write(home, entry())
    session = SimpleNamespace(effort="", save=lambda: None)
    engine = object.__new__(LocalLLMEngine)
    engine._sessions = {"test": session}
    await engine.set_config_option("test", "effort", "custom_Depth")
    assert session.effort == "custom_Depth"


def test_sampling_disabled_omits_even_configured_fields(home, monkeypatch):
    write(home, entry(parameters={"temperature": 0.6, "chat_template_kwargs.enable_thinking": True}))
    monkeypatch.setenv("SHAREDAGENTS_DISABLE_MODEL_SAMPLING", "1")
    payload = wire(effort="low")
    assert "temperature" not in payload
    assert "reasoning_effort" not in payload
    assert "extra" not in payload


def test_mode_default_agrees_between_picker_and_wire(home):
    write(home, entry(modes={"plan": {"default": "low"}}))
    row = {"id": "example-model", "backend": "vllm"}
    assert effort_prefs.selected_level_for(row, mode="plan") == "low"
    assert wire(mode="plan")["reasoning_effort"] == "low"
    assert effort_prefs.selected_level_for(row, mode="auto") == ""
    assert "reasoning_effort" not in wire(mode="auto")


def test_explicit_loaded_legacy_null_suppresses_file_temperature(home):
    write(home, entry(parameters={"temperature": 0.5}))
    agent = home / "agents" / "test"
    agent.mkdir(parents=True)
    (agent / "config.json").write_text(json.dumps({"name": "test", "temperature": None}))
    cfg = AgentConfig.load("test", ambient_discovery=False)
    assert configured_sampling(cfg, {"id": "example-model", "backend": "vllm"}) == {}


def test_endpoint_auth_fallback_does_not_borrow_another_models_policy(home):
    write(home, entry())
    row = {
        "id": "example-model",
        "wire_id": "example-model",
        "alias": "example-alias",
        "backend": "vllm",
        "baseUrl": "http://example.com/v1",
        "effort": {"values": ["low"]},
        "sampling": {"temperature": 0.3},
    }
    payload = wire("different-model", "low", row=row)
    assert "reasoning_effort" not in payload
    assert "temperature" not in payload


def test_explicit_turn_null_can_suppress_a_configured_budget(home):
    write(
        home,
        entry(
            effort={
                "values": ["quick"],
                "default": "quick",
                "wire": "chat_template_kwargs.thinking_budget",
                "wire_values": {"quick": 2048},
            }
        ),
    )
    assert "extra" not in wire(sampling={"thinking_budget": None})


@pytest.mark.parametrize("selected", ["", "custom_Depth", "stale"])
def test_actual_worker_serializes_only_effective_parameters(home, monkeypatch, capsys, selected):
    import io
    import sys

    import httpx

    from sharedagents_gateway import worker

    write(home, entry(parameters={"temperature": 0, "chat_template_kwargs.enable_thinking": False}))
    captured = []
    real_client = httpx.Client

    def handle(request):
        captured.append(json.loads(request.content))
        event = {"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop", "index": 0}]}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode(),
        )

    monkeypatch.setattr(worker, "_install_term_handler", lambda: None)
    monkeypatch.setattr(
        worker.httpx, "Client", lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(handle), **kw)
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(wire(effort=selected))))
    assert worker.main() == 0
    capsys.readouterr()
    assert len(captured) == 1
    body = captured[0]
    assert body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert ("reasoning_effort" in body) == (selected == "custom_Depth")
    if selected == "custom_Depth":
        assert body["reasoning_effort"] == selected
