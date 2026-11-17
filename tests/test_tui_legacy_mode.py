"""Regression: the TUI's legacy ``/mode`` fallback must import and toggle."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from sharedagents_tui.agent_client import AgentClient


def _client(agent: str) -> AgentClient:
    c = AgentClient(agent_name=agent)
    c._ensure_connected = AsyncMock(return_value=AsyncMock())  # type: ignore[method-assign]
    c.switch_agent = AsyncMock()  # type: ignore[method-assign]
    return c


def test_legacy_mode_toggles_build_to_plan() -> None:
    c = _client("build")
    res = asyncio.run(c._session_command_legacy("sid1", "/mode"))
    assert res["ok"] is True
    assert res["state_patch"] == {"agent": "plan"}
    c.switch_agent.assert_awaited_once_with("sid1", "plan", frontend="tui")


def test_legacy_mode_explicit_build_from_plan() -> None:
    c = _client("plan")
    res = asyncio.run(c._session_command_legacy("sid1", "/mode build"))
    assert res["ok"] is True
    assert res["data"] == {"agent": "build"}


def test_legacy_mode_unknown_arg_rejected() -> None:
    c = _client("plan")
    res = asyncio.run(c._session_command_legacy("sid1", "/mode bogus"))
    assert res["ok"] is False
    c.switch_agent.assert_not_awaited()
