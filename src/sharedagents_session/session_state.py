"""Derived session state — read straight from the single ``<sid>.jsonl`` ledger.

There is exactly ONE class for a session (``SessionWriter`` appends every event
to ``<sessions_dir>/<sid>.jsonl`` in sequence; ``session_reader`` tails it).
This module derives the *current* state (active agent, turn count, agent
history) by scanning that one file, with no secondary store, lanes, or sidecar
files.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA_TYPE_EVENTS = (
    "agent_switched",
    "turn_started",
    "turn_ended",
)

_DEFAULT_AGENT = "jarvis"


def _iter_events(sessions_dir: Any, sid: str) -> Iterator[dict[str, Any]]:
    path = Path(sessions_dir) / f"{sid}.jsonl"
    if not path.exists():
        return
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with fh:
        for line in fh:
            line = line.rstrip("\n")
            sp = line.find(" ")
            if sp != 8:  # 8-hex CRC prefix + space
                continue
            try:
                ev = json.loads(line[sp + 1 :])
            except (ValueError, TypeError):
                continue
            if isinstance(ev, dict):
                yield ev


def active_agent(sessions_dir: Any, sid: str, *, default: str = _DEFAULT_AGENT) -> str:
    """The session's current agent: last ``agent_switched.to_agent`` or the
    last ``turn_started.agent_name`` in the file."""
    agent = default
    for ev in _iter_events(sessions_dir, sid):
        kind = ev.get("type")
        if kind == "agent_switched":
            to_agent = str(ev.get("to_agent") or "").strip()
            if to_agent:
                agent = to_agent
        elif kind == "turn_started":
            name = str(ev.get("agent_name") or "").strip()
            if name:
                agent = name
    return agent


def turn_count(sessions_dir: Any, sid: str) -> int:
    """Number of turns seen in the file (started OR ended events)."""
    return sum(1 for ev in _iter_events(sessions_dir, sid) if ev.get("type") == "turn_started")


def revision(sessions_dir: Any, sid: str) -> int:
    """Logical revision = number of completed turns (turn_ended rows)."""
    return sum(1 for ev in _iter_events(sessions_dir, sid) if ev.get("type") == "turn_ended")


def agent_history(sessions_dir: Any, sid: str) -> list[str]:
    """Ordered distinct agents that have run in this session (switch order)."""
    agents: list[str] = []
    seen: set[str] = set()
    for ev in _iter_events(sessions_dir, sid):
        kind = ev.get("type")
        name = ""
        if kind == "agent_switched":
            name = str(ev.get("to_agent") or "").strip()
        elif kind == "turn_started":
            name = str(ev.get("agent_name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            agents.append(name)
    return agents


def is_first_owner_turn_after_switch(
    sessions_dir: Any,
    sid: str,
    agent_name: str,
) -> bool:
    """True when ``agent_name`` is the last ``agent_switched.to_agent`` and
    that agent has not yet started an owner turn after the switch.

    ``turn_started`` is written on the first in-progress flush, after
    transcript injection, so this is True for the first owner prompt
    after a badge flip.
    """
    target = str(agent_name or "").strip()
    if not target:
        return False
    last_switch_to = ""
    owner_started_after_switch = False
    for ev in _iter_events(sessions_dir, sid):
        kind = ev.get("type")
        if kind == "agent_switched":
            to_agent = str(ev.get("to_agent") or "").strip()
            if to_agent:
                last_switch_to = to_agent
                owner_started_after_switch = False
        elif kind == "turn_started":
            name = str(ev.get("agent_name") or "").strip()
            if last_switch_to and name == last_switch_to:
                owner_started_after_switch = True
    return bool(last_switch_to) and last_switch_to == target and not owner_started_after_switch


def switched_from_agent(sessions_dir: Any, sid: str, agent_name: str) -> str:
    """The ``from_agent`` of the last ``agent_switched`` that targeted
    ``agent_name``, or ``""`` when the last switch went elsewhere (or none).
    """
    target = str(agent_name or "").strip()
    if not target:
        return ""
    last_to = last_from = ""
    for ev in _iter_events(sessions_dir, sid):
        if ev.get("type") == "agent_switched":
            last_to = str(ev.get("to_agent") or "").strip()
            last_from = str(ev.get("from_agent") or "").strip()
    return last_from if last_to == target else ""


def engine_had_owner_turn(sessions_dir: Any, sid: str, agent_name: str) -> bool:
    """True when this agent already started at least one owner turn on SID."""
    target = str(agent_name or "").strip()
    if not target:
        return False
    for ev in _iter_events(sessions_dir, sid):
        if ev.get("type") != "turn_started":
            continue
        if str(ev.get("agent_name") or "").strip() == target:
            return True
    return False


def permission_mode(sessions_dir: Any, sid: str, *, default: str = "auto") -> str:
    """The effective permission mode of the last turn_started."""
    mode = default
    for ev in _iter_events(sessions_dir, sid):
        if ev.get("type") == "turn_started":
            pm = str(ev.get("permission_mode") or "").strip()
            if pm:
                mode = pm
        elif ev.get("type") == "header_update":
            pm = str((ev.get("patch") or {}).get("system_mode") or "").strip()
            if pm:
                mode = pm
    return mode
