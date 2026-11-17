"""What an admin can grant in the Users tab: agents, models, tools, skills.

Read from the same sources the runtime uses — the on-disk agent roster, the
cached model catalogues (never a live provider probe), the built-in tool
registry plus configured MCP servers (as ``mcp__<server>__*`` groups), and the
skill search chain. Each section fails soft to an empty list. Cached briefly:
the Users tab re-reads it on every open.
"""

from __future__ import annotations

import time
from typing import Any

_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_TTL_S = 30.0


def _agent_names() -> list[str]:
    from sharedagents_agent.agent import list_agents

    return sorted(str(n) for n in list_agents())


def _agent_engine(name: str) -> str:
    from sharedagents_agent.agent import AgentConfig

    return str(getattr(AgentConfig.load(name), "engine", "") or "")


def _cached_models() -> dict[str, list[dict]]:
    from sharedagents_agent.discovery.cache_lock import read_all_cached_models

    return read_all_cached_models() or {}


def _configured_models() -> list[dict]:
    from sharedagents_core.kernel.config import load_global_config

    rows = (load_global_config() or {}).get("models") or []
    return [r for r in rows if isinstance(r, dict)]


def _builtin_tools() -> list[tuple[str, str]]:
    from sharedagents_agent.tools.registry import BUILTIN_TOOLS, ToolRegistry

    rows = []
    for schema in ToolRegistry(BUILTIN_TOOLS).openai_schemas():
        fn = schema.get("function") or {}
        name = str(fn.get("name") or "").strip()
        if name:
            desc = " ".join(str(fn.get("description") or "").split())
            rows.append((name, desc[:160]))
    return rows


def _mcp_server_names() -> list[str]:
    from sharedagents_mcp.mcp_client import load_mcp_servers

    return sorted(load_mcp_servers() or {})


def _skill_rows() -> list[tuple[str, str]]:
    from sharedagents_skill.skills import discover_skills

    return [(s.name, s.short_description or "") for s in discover_skills()]


def _soft(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001 — one broken source must not hide the rest
        return default


def _agents() -> list[dict]:
    from sharedagents_agent.external_registry import EXTERNAL_ENGINE_VALUES
    from sharedagents_web.agent_visibility import is_non_conversational_agent

    rows = []
    for name in _soft(_agent_names, []):
        engine = _soft(lambda n=name: _agent_engine(n), "")
        rows.append(
            {
                "name": name,
                "engine": engine or "local_llm",
                # Brings its own tools and skills: usable only with All of both.
                "external": engine in EXTERNAL_ENGINE_VALUES,
                # Delegation-only utilities still matter as delegation targets.
                "conversational": not is_non_conversational_agent(name),
            }
        )
    return rows


def _models() -> list[dict]:
    """Cached picker catalogues plus the deployments configured in
    ``config.json`` (grantable even before anyone opened a picker)."""
    seen: dict[str, dict] = {}
    known: set[str] = set()
    sources = list(_soft(_cached_models, {}).values())
    sources.append(_soft(_configured_models, []))
    for rows in sources:
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id") or row.get("alias") or "").strip()
            alias = str(row.get("alias") or "").strip()
            if not model_id or model_id in known or (alias and alias in known):
                continue
            known.update(n for n in (model_id, alias) if n)
            seen[model_id] = {
                "id": model_id,
                "alias": alias,
                "display": str(row.get("display") or row.get("name") or model_id),
                "provider": str(row.get("provider") or row.get("backend") or ""),
            }
    return sorted(seen.values(), key=lambda r: (r["provider"], r["display"].lower(), r["id"]))


# How the Users tab groups built-in tools, and which ones act on this host
# as the operator's account ("host"), read its files ("read") or spend
# money / schedule work ("spend"). Unlisted tools are conversation helpers.
_TOOL_GROUPS: dict[str, tuple[str, str]] = {
    "bash": ("Shell", "host"),
    "BashOutput": ("Shell", "host"),
    "KillShell": ("Shell", "host"),
    "write": ("Files", "host"),
    "edit": ("Files", "host"),
    "apply_patch": ("Files", "host"),
    "read": ("Files", "read"),
    "glob": ("Files", "read"),
    "grep": ("Files", "read"),
    "list": ("Files", "read"),
    "lsp": ("Files", "read"),
    "AgentAsync": ("Delegation", ""),
    "AgentSync": ("Delegation", ""),
    "acp_call": ("Delegation", "host"),
    "cron_create": ("Scheduling", "spend"),
    "cron_delete": ("Scheduling", ""),
    "cron_list": ("Scheduling", ""),
    "schedule_wakeup": ("Scheduling", "spend"),
    "goal": ("Scheduling", "spend"),
    "budget": ("Scheduling", "spend"),
    "webfetch": ("Web", ""),
    "websearch": ("Web", ""),
    "x_tweets": ("Web", ""),
    "meeting_search": ("Memory", "private"),  # the operator's meeting transcripts
}


def _tools() -> list[dict]:
    rows = []
    for name, desc in _soft(_builtin_tools, []):
        group, risk = _TOOL_GROUPS.get(name, ("Conversation", ""))
        rows.append({"name": name, "description": desc, "group": group, "risk": risk})
    rows.extend(
        {
            "name": f"mcp__{server}__*",
            "description": f"Every tool of the {server} MCP server",
            "group": "MCP",
            "risk": "",
        }
        for server in _soft(_mcp_server_names, [])
    )
    return rows


def _skills() -> list[dict]:
    return [{"name": n, "description": d} for n, d in _soft(_skill_rows, [])]


def build_catalog() -> dict:
    from sharedagents_core.folder_policy import approved_roots

    return {
        "agents": _agents(),
        "models": _models(),
        "tools": _tools(),
        "skills": _skills(),
        "folder_roots": [{"path": str(p), "label": p.name or str(p)} for p in approved_roots()],
    }


def catalog() -> dict:
    now = time.monotonic()
    if _CACHE["value"] is None or now - _CACHE["at"] > _TTL_S:
        _CACHE.update(at=now, value=build_catalog())
    return _CACHE["value"]
