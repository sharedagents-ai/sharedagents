"""On-demand MCP tool schemas for native agents.

An agent that reaches MCP servers only through implicit access (a non-strict
allowlist, or full access) used to ship every MCP tool schema on every round,
even when most servers were unused. Here those servers are *deferred*: the
prompt lists each one with its tool names, and the schemas join ``tools[]``
once the agent loads the server with ``load_tools`` — or simply calls one of
its tools, which the permission gate allows exactly as before.

Only the WIRE presentation changes. The permission set (``compose_tool_allowlist``)
and the dispatch gate are untouched, so a deferred tool called by name still
runs. Always on, never deferred:

* every built-in tool (no discovery hop for everyday tools);
* MCP tools granted explicitly in ``tools_allowlist`` (``mcp__x__*`` or exact);
* session-scoped (ACP) MCP tools;
* the builtin MCP servers (memory, web, delegate) — small and heavily used;
* a server listed in the agent's ``mcp_always_loaded``;
* a server whose ``mcpServers`` row sets ``"defer": false``.

The global config key ``mcpDeferral: false`` turns the whole feature off.

"Loaded" is derived from successful ``load_tools`` results or direct MCP
calls, including the session's durable tool-call records. Compaction can
drop their wire messages without unloading or reordering the schemas. No
separate state file is needed, and children use only their own records.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

LOAD_TOOLS = "load_tools"

# Builtin MCP servers (sharedagents_mcp.mcp_client._BUILTIN_MCP_MODULES) are always on.
_ALWAYS_ON_SERVERS = frozenset({"memory", "web", "delegate"})


def mcp_server_of(name: str) -> str | None:
    """``mcp__<server>__<tool>`` → ``<server>`` (split on the LAST ``__``,
    mirroring ``mcp_client.parse_mcp_namespaced``); ``None`` otherwise."""
    if not isinstance(name, str) or not name.startswith("mcp__"):
        return None
    rest = name[len("mcp__") :]
    return rest.rsplit("__", 1)[0] if "__" in rest else None


def _global_cfg() -> dict[str, Any]:
    try:
        from sharedagents_core.kernel.config import load_global_config

        return load_global_config()
    except Exception:  # noqa: BLE001
        return {}


def _server_rows(gcfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = gcfg.get("mcpServers") or gcfg.get("mcp_servers") or {}
    return raw if isinstance(raw, dict) else {}


def deferred_servers(
    cfg: object,
    allowed: Iterable[str],
    *,
    session_mcp_names: Iterable[str] = (),
    global_cfg: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """``{server: [tool, …]}`` for the MCP tools in *allowed* that this agent
    sees only on demand. Empty when deferral is off or nothing qualifies.

    *allowed* is the agent's final tool set (after ``disabled_tools`` and any
    Web-user grants), so a server the user may not use is never offered.
    """
    gcfg = _global_cfg() if global_cfg is None else global_cfg
    if gcfg.get("mcpDeferral") is False or LOAD_TOOLS in (getattr(cfg, "disabled_tools", None) or ()):
        # Without the loader, keep the permitted MCP schemas available eagerly.
        return {}
    allowed = set(allowed)
    rows = _server_rows(gcfg)
    wl = getattr(cfg, "tools_allowlist", None)
    explicit: set[str] = set()
    if isinstance(wl, list) and wl:
        from sharedagents_agent.agent_profiles import expand_tool_allowlist

        mcp_allowed = [n for n in allowed if n.startswith("mcp__")]
        explicit = expand_tool_allowlist(
            [t for t in wl if isinstance(t, str)],
            mcp_allowed,
        )
    session_mcp = set(session_mcp_names or ())
    pinned = getattr(cfg, "mcp_always_loaded", None)
    pinned = {str(x) for x in pinned} if isinstance(pinned, list) else set()
    out: dict[str, list[str]] = {}
    for name in allowed:
        server = mcp_server_of(name)
        if server is None or name in explicit or name in session_mcp:
            continue
        if server in _ALWAYS_ON_SERVERS or server in pinned:
            continue
        row = rows.get(server)
        if isinstance(row, dict) and row.get("defer") is False:
            continue
        out.setdefault(server, []).append(name)
    return {s: sorted(v) for s, v in sorted(out.items())}


def loaded_servers(
    messages: Iterable[Mapping[str, Any]],
    *,
    turns: Iterable[Mapping[str, Any]] = (),
) -> list[str]:
    """Recover first-load order from this session's records, then live messages.

    Durable records survive compaction and include hook-rewritten arguments.
    An unsuccessful or denied loader call never loads even a partial subset.
    """
    out: dict[str, None] = {}

    for turn in turns or ():
        if not isinstance(turn, Mapping):
            continue
        results = {
            str(r.get("call_id") or r.get("id") or ""): r
            for r in turn.get("tool_results") or ()
            if isinstance(r, Mapping)
        }
        for call in turn.get("tool_calls") or ():
            if not isinstance(call, Mapping):
                continue
            name = call.get("name")
            server = mcp_server_of(name)
            if server:
                out.setdefault(server)
            elif name == LOAD_TOOLS:
                result = results.get(str(call.get("call_id") or call.get("id") or ""))
                if result and result.get("ok") is True:
                    for server in requested_servers(call.get("arguments")):
                        out.setdefault(server)

    messages = list(messages or ())
    successful: dict[str, list[str]] = {}
    for msg in messages:
        if not isinstance(msg, Mapping) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        try:
            # PostToolUse may append feedback after the framework's JSON.
            result, _ = json.JSONDecoder().raw_decode(content)
        except ValueError:
            continue
        if isinstance(result, dict) and isinstance(result.get("loaded_servers"), list):
            successful[str(msg.get("tool_call_id") or "")] = [s for s in result["loaded_servers"] if isinstance(s, str)]

    for msg in messages:
        if not isinstance(msg, Mapping) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or ():
            if not isinstance(tc, Mapping):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") if isinstance(fn, Mapping) else None
            if not isinstance(name, str):
                continue
            server = mcp_server_of(name)
            if server:
                out.setdefault(server)
            elif name == LOAD_TOOLS:
                for s in successful.get(str(tc.get("id") or ""), ()):
                    out.setdefault(s)
    return list(out)


def requested_servers(arguments: Any) -> list[str]:
    """Server names from ``load_tools`` arguments (object or JSON string;
    ``server`` may be one name or a list)."""
    args = arguments
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (TypeError, ValueError):
            return []
    if not isinstance(args, Mapping):
        return []
    raw = args.get("server")
    if raw is None:
        raw = args.get("servers")
    vals = raw if isinstance(raw, list) else [raw]
    return [str(v).strip() for v in vals if isinstance(v, str) and v.strip()]


def load_tools_schema(deferred: Mapping[str, list[str]]) -> dict[str, Any]:
    """The ``load_tools`` function schema for this agent's deferred servers."""
    names = sorted(deferred)
    return {
        "type": "function",
        "function": {
            "name": LOAD_TOOLS,
            "description": (
                "Load the tools of an MCP server listed under 'MCP servers "
                "on demand' in your instructions. They are callable from "
                "your next step and stay loaded for this session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "enum": names,
                        "description": "MCP server name.",
                    },
                },
                "required": ["server"],
                "additionalProperties": False,
            },
        },
    }


def present_tools(
    schemas: list[dict[str, Any]],
    deferred: Mapping[str, list[str]],
    loaded: Iterable[str],
) -> list[dict[str, Any]]:
    """The wire order: non-MCP tools, ``load_tools``, always-on MCP tools,
    then each loaded deferred server in first-load order. Loading a server
    only appends, so the cached prompt prefix survives it."""
    if not deferred:
        return schemas
    by_name = {t["function"]["name"]: t for t in schemas}
    on_demand = {n for tools in deferred.values() for n in tools}
    base = [t for t in schemas if t["function"]["name"] not in on_demand]
    at = next(
        (i for i, t in enumerate(base) if str(t["function"]["name"]).startswith("mcp__")),
        len(base),
    )
    out = base[:at] + [load_tools_schema(deferred)] + base[at:]
    for server in dict.fromkeys(loaded):
        out.extend(by_name[n] for n in deferred.get(server, ()) if n in by_name)
    return out


def catalog_prompt_lines(deferred: Mapping[str, list[str]]) -> list[str]:
    """The ``## Tooling`` subsection naming each deferred server's tools."""
    if not deferred:
        return []
    rows = _server_rows(_global_cfg())
    lines = [
        "### MCP servers on demand",
        "These servers' tools are not in your tool list yet. Call "
        f"``{LOAD_TOOLS}`` with the server name once; its tools are then "
        "callable as ``mcp__<server>__<tool>`` for the rest of the session.",
    ]
    for server, tools in deferred.items():
        row = rows.get(server)
        desc = str(row.get("description") or "").strip() if isinstance(row, dict) else ""
        short = ", ".join(t.rsplit("__", 1)[-1] for t in tools)
        lines.append(f"- {server}" + (f" — {desc}" if desc else "") + f": {short}")
    return lines


def load_result_text(
    requested: list[str],
    deferred: Mapping[str, list[str]],
) -> tuple[bool, str]:
    """``(ok, text)`` for a ``load_tools`` call."""
    if not requested:
        return False, "load_tools needs a server name."
    unknown = [s for s in requested if s not in deferred]
    if unknown:
        avail = ", ".join(sorted(deferred)) or "none"
        return False, (f"Unknown server: {', '.join(unknown)}. Servers on demand: {avail}.")
    return True, json.dumps(
        {
            "loaded_servers": requested,
            "tools": {s: deferred[s] for s in requested},
            "message": "Call them directly from your next step.",
        }
    )
