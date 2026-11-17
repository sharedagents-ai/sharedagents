"""Switching a live session's agent or model through ``SessionRuntime``."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from sharedagents_agent.agent import AgentConfig
from sharedagents_gateway.session_runtime import SessionRuntime, SwitchRequest
from sharedagents_session.session_reader import tail_jsonl
from sharedagents_session.session_writer import get_session_writer, release_session_writer


@pytest.fixture(autouse=True)
def _scaffolded_agents(tmp_path, monkeypatch):
    """AgentConfig.load reads the on-disk scaffold — pin it to a tmp home."""
    home = tmp_path / "home"
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(home))
    monkeypatch.delenv("SHAREDAGENTS_AGENTS_HOME", raising=False)
    from sharedagents_agent.agents.default_scaffolds import ensure_builtin_profile_agents

    ensure_builtin_profile_agents()
    # The real deploy carries a discovered model catalogue; seed the rows the
    # switch tests resolve (endpoint/backend/context_window come from here).
    import json

    plan_cfg = home / "agents" / "plan" / "config.json"
    cfg = json.loads(plan_cfg.read_text())
    cfg["models"] = [
        {
            "alias": "deepseek-flash",
            "id": "deepseek-flash",
            "wire_id": "deepseek-flash",
            "model_id": "deepseek-flash",
            "baseUrl": "https://api.deepseek.com/v1",
            "backend": "openai",
            "provider": "deepseek",
            "contextWindow": 1000000,
        },
        {
            "alias": "deepseek-v4.1-flash",
            "id": "deepseek-flash",
            "wire_id": "deepseek-flash",
            "model_id": "deepseek-flash",
            "baseUrl": "https://api.deepseek.com/v1",
            "backend": "openai",
            "provider": "deepseek",
            "contextWindow": 1000000,
        },
    ]
    plan_cfg.write_text(json.dumps(cfg))
    # An external-CLI agent for the cross-agent switch test.
    claude = home / "agents" / "claude"
    claude.mkdir(parents=True, exist_ok=True)
    (claude / "config.json").write_text(
        json.dumps(
            {
                "name": "claude",
                "engine": "claude_cli",
                "upstream_model": "claude-opus-5-5",
                "upstream_max_context": 1000000,
            }
        )
    )


@pytest.mark.asyncio
async def test_switch_model_on_native_agent_resolves_catalogue_endpoint_and_backend(tmp_path: Path):
    sid = "test-session-switch-wire"
    w = get_session_writer(tmp_path, sid, engine="plan")
    try:
        w.session_started(
            header={"agent_name": "plan", "model": "qwen3.8-27b-fp8", "endpoint": "http://127.0.0.1:8000/v1"}
        )
    finally:
        release_session_writer(w)

    cfg = AgentConfig.load("plan")
    agent_obj = SimpleNamespace(cfg=cfg)
    backend = SimpleNamespace(agent=agent_obj, _active_sessions={}, load_session=None, set_mode=None)
    manager = SimpleNamespace(get=lambda name: backend)

    rt = SessionRuntime(sessions_dir=tmp_path, agent_names=["plan"], manager=manager)
    res = await rt.switch(SwitchRequest(session_id=sid, target_agent="plan", target_model="deepseek-flash"))
    assert res["ok"] is True
    assert res["model"] == "deepseek-flash"

    events = [e for e, _ in tail_jsonl(tmp_path / f"{sid}.jsonl", 0)]
    switch_ev = next(e for e in events if e.get("type") == "agent_switched")
    assert switch_ev["to_agent"] == "plan"
    assert switch_ev["model"] == "deepseek-flash"
    assert switch_ev["endpoint"] == "https://api.deepseek.com/v1"
    assert switch_ev["backend"] == "openai"

    patch_ev = next(e for e in events if e.get("type") == "header_update")
    assert patch_ev["patch"]["endpoint"] == "https://api.deepseek.com/v1"
    assert patch_ev["patch"]["backend"] == "openai"
    assert patch_ev["patch"]["context_window"] == 1000000


@pytest.mark.asyncio
async def test_switch_model_by_alias_resolves_wire_id_and_endpoint(tmp_path: Path):
    sid = "test-session-switch-alias"
    w = get_session_writer(tmp_path, sid, engine="plan")
    try:
        w.session_started(
            header={"agent_name": "plan", "model": "qwen3.8-27b-fp8", "endpoint": "http://127.0.0.1:8000/v1"}
        )
    finally:
        release_session_writer(w)

    cfg = AgentConfig.load("plan")
    agent_obj = SimpleNamespace(cfg=cfg)
    backend = SimpleNamespace(agent=agent_obj, _active_sessions={}, load_session=None, set_mode=None)
    manager = SimpleNamespace(get=lambda name: backend)

    rt = SessionRuntime(sessions_dir=tmp_path, agent_names=["plan"], manager=manager)
    res = await rt.switch(SwitchRequest(session_id=sid, target_agent="plan", target_model="deepseek-v4.1-flash"))
    assert res["ok"] is True
    assert res["model"] == "deepseek-flash"

    events = [e for e, _ in tail_jsonl(tmp_path / f"{sid}.jsonl", 0)]
    switch_ev = next(e for e in events if e.get("type") == "agent_switched")
    assert switch_ev["model"] == "deepseek-flash"
    assert switch_ev["endpoint"] == "https://api.deepseek.com/v1"
    assert switch_ev["backend"] == "openai"


@pytest.mark.asyncio
async def test_switch_to_external_agent_sets_external_engine_backend(tmp_path: Path, monkeypatch):
    # The claude CLI need not be installed: a missing CLI makes every mode unsupported.
    monkeypatch.setattr("sharedagents_agent.engine_modes._engine_availability", lambda engine, cfg: (True, "", ""))
    sid = "test-session-switch-ext"
    w = get_session_writer(tmp_path, sid, engine="plan")
    try:
        w.session_started(
            header={"agent_name": "plan", "model": "qwen3.8-27b-fp8", "endpoint": "http://127.0.0.1:8000/v1"}
        )
    finally:
        release_session_writer(w)

    cfg = AgentConfig.load("claude")
    agent_obj = SimpleNamespace(cfg=cfg)
    backend = SimpleNamespace(agent=agent_obj, _active_sessions={}, load_session=None, set_mode=None)
    manager = SimpleNamespace(get=lambda name: backend)

    rt = SessionRuntime(sessions_dir=tmp_path, agent_names=["plan", "claude"], manager=manager)
    res = await rt.switch(SwitchRequest(session_id=sid, target_agent="claude"))
    assert res["ok"] is True

    events = [e for e, _ in tail_jsonl(tmp_path / f"{sid}.jsonl", 0)]
    switch_ev = next(e for e in events if e.get("type") == "agent_switched")
    assert switch_ev["to_agent"] == "claude"
    assert switch_ev["backend"] == "claude_cli"
    assert "endpoint" not in switch_ev or switch_ev["endpoint"] == ""
