"""Model Context Protocol client.

sharedagents connects out to MCP servers (stdio + HTTP/SSE transports) so the
model can call tools provided by, e.g., the Brave-search MCP server, a
filesystem server, or a custom company tool. Servers are configured once
in `~/.sharedagents/config.json` under `mcp_servers`; the model picks one by
its short name when it issues a tool call.

Connection model — one-shot per call
----------------------------------

The MCP SDK's `ClientSession` is an async context manager that owns the
lifetime of a transport (a stdio subprocess or an HTTP/SSE stream). The
straightforward way to support every tool runner cleanly is to spin up a
fresh session per call:

    async with mcp_session(server_name) as sess:
        result = await sess.call_tool(tool_name, arguments=...)

Per-call sessions trade a small startup cost (cheap for stdio, ~50 ms;
HTTP is just a TCP open) for two big simplifications:

1. **Process model fits.** Each sharedagents tool call already runs in its
   own subprocess (`tool_runner.py`). A persistent MCP connection would
   need a long-lived broker the runner can't easily reach.
2. **Reconnect is implicit.** A failed call returns an error; the next
   call opens a fresh connection. No background reconnect loop, no
   stale-socket detection.

If a server proves to need persistent state across calls (e.g. a
session-ful query language) we'll graduate it to a long-lived broker
inside the agent process; today, no configured server needs that.

Configuration shape
-------------------

stdio:
    {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {"BRAVE_API_KEY": "..."},
      "transport": "stdio"
    }

http (Streamable HTTP):
    {
      "url": "http://localhost:3100",
      "transport": "http",
      "headers": {"Authorization": "Bearer ..."}
    }

sse (legacy SSE):
    {
      "url": "http://localhost:3101/sse",
      "transport": "sse",
      "headers": {"Authorization": "Bearer ..."}
    }

Public API
----------
- ``MCPServerConfig.from_dict(name, raw)`` — parse one config row.
- ``load_mcp_servers(global_cfg)`` — read all configured servers.
- ``mcp_session(name)`` async context manager.
- ``call_tool(name, tool_name, arguments)`` — one-shot helper.
- ``list_tools(name)`` — one-shot helper.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Any

_log = logging.getLogger("sharedagents.mcp")


# ---- config -----------------------------------------------------------


def _acp_kv(value: Any) -> dict[str, str]:
    """Normalize an ACP env/headers field to ``{str: str}``.

    The ACP wire schema is a list of ``{name, value}`` objects, but some
    clients send a plain ``{key: value}`` object — accept both. Anything
    else (None, malformed) yields an empty dict."""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, list):
        return {str(e.get("name")): str(e.get("value", "")) for e in value if isinstance(e, dict) and e.get("name")}
    return {}


@dataclass(frozen=True)
class MCPServerConfig:
    """One configured MCP server, with the transport-specific fields it
    needs to spawn or connect to.

    Validation lives here (not at config-load time) so a malformed row
    doesn't take down the agent at boot — the error only surfaces when
    the model actually tries to use that server.
    """

    name: str
    transport: str  # "stdio" | "http" | "sse"
    # stdio
    command: str = ""
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    # http / sse
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    # Optional human-readable description, surfaced to the model.
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> MCPServerConfig:
        if not isinstance(raw, dict):
            raise ValueError(f"mcp_servers[{name!r}]: not a dict")
        # Default to stdio when `command` is set, http when `url` is set,
        # so existing configs without a `transport` field keep working.
        transport = str(raw.get("transport") or "").strip().lower()
        if not transport:
            if raw.get("command"):
                transport = "stdio"
            elif raw.get("url"):
                transport = "http"
            else:
                raise ValueError(f"mcp_servers[{name!r}]: missing transport (and neither command nor url is set)")
        if transport == "stdio":
            cmd = str(raw.get("command") or "").strip()
            if not cmd:
                raise ValueError(f"mcp_servers[{name!r}]: stdio transport requires `command`")
            args = tuple(str(a) for a in (raw.get("args") or []))
            env_raw = raw.get("env") or {}
            env = {str(k): str(v) for k, v in env_raw.items()} if isinstance(env_raw, dict) else {}
            return cls(
                name=name,
                transport="stdio",
                command=cmd,
                args=args,
                env=env,
                cwd=str(raw.get("cwd")) if raw.get("cwd") else None,
                description=str(raw.get("description") or ""),
            )
        if transport in ("http", "sse"):
            url = str(raw.get("url") or "").strip()
            if not url:
                raise ValueError(f"mcp_servers[{name!r}]: {transport} transport requires `url`")
            headers_raw = raw.get("headers") or {}
            headers = {str(k): str(v) for k, v in headers_raw.items()} if isinstance(headers_raw, dict) else {}
            return cls(
                name=name,
                transport=transport,
                url=url,
                headers=headers,
                description=str(raw.get("description") or ""),
            )
        raise ValueError(f"mcp_servers[{name!r}]: unknown transport {transport!r} (expected stdio | http | sse)")

    @classmethod
    def from_acp(cls, raw: dict[str, Any]) -> MCPServerConfig:
        """Parse one ACP ``mcpServers`` entry (the ``session/new`` /
        ``session/load`` schema) into an :class:`MCPServerConfig`.

        ACP's shape differs from sharedagents's own config: the server name
        is a field (not the dict key), and ``env`` / ``headers`` are
        lists of ``{name, value}`` objects rather than maps. stdio is
        ``{name, command, args[], env[]}``; http/sse is
        ``{type, name, url, headers[]}``. An absent ``type`` means
        stdio (the ACP baseline transport every agent MUST support).
        """
        if not isinstance(raw, dict):
            raise ValueError("acp mcpServers entry: not an object")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError("acp mcpServers entry: missing name")
        transport = str(raw.get("type") or "").strip().lower() or "stdio"
        if transport in ("http", "sse"):
            url = str(raw.get("url") or "").strip()
            if not url:
                raise ValueError(f"acp mcpServers[{name!r}]: {transport} requires url")
            return cls(
                name=name,
                transport=transport,
                url=url,
                headers=_acp_kv(raw.get("headers")),
            )
        if transport != "stdio":
            raise ValueError(f"acp mcpServers[{name!r}]: unknown type {transport!r}")
        command = str(raw.get("command") or "").strip()
        if not command:
            raise ValueError(f"acp mcpServers[{name!r}]: stdio requires command")
        args = tuple(str(a) for a in (raw.get("args") or []))
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=args,
            env=_acp_kv(raw.get("env")),
        )


def load_mcp_servers(global_cfg: dict[str, Any] | None = None) -> dict[str, MCPServerConfig]:
    """Parse the MCP server map out of the global config dict.

    Accepts both the standard ``mcpServers`` (camelCase, what Claude
    Desktop / Claude Code / the MCP Python SDK use) and the legacy
    ``mcp_servers`` (snake_case, what early sharedagents configs used).
    The standard key wins when both are present.

    Bad rows are dropped with a logged warning rather than raising —
    one mistyped server config shouldn't keep the model from using the
    other configured ones. Pass `global_cfg=None` to load from disk.
    """
    if global_cfg is None:
        try:
            from sharedagents_agent.agent import load_global_config

            global_cfg = load_global_config()
        except Exception:  # noqa: BLE001
            global_cfg = {}
    cfg = global_cfg or {}
    # Standard key (mcpServers) wins; legacy key (mcp_servers) is the
    # fallback. Both empty → no servers from the user config.
    raw = cfg.get("mcpServers") or cfg.get("mcp_servers") or {}
    if not isinstance(raw, dict):
        raw = {}
    # Claude Code's canonical user-level MCP file at
    # ``~/.claude/.mcp.json`` (and the unprefixed ``mcp.json``) — read
    # it as another source so an operator who configured an MCP via
    # Claude Code sees the same server in sharedagents. User's
    # sharedagents config still wins on collisions.
    cc_raw: dict[str, Any] = {}
    try:
        import json as _json
        from pathlib import Path as _Path

        for _cand in (
            _Path.home() / ".claude" / ".mcp.json",
            _Path.home() / ".claude" / "mcp.json",
        ):
            if _cand.is_file():
                _doc = _json.loads(_cand.read_text(encoding="utf-8"))
                if isinstance(_doc, dict):
                    cc_raw = _doc.get("mcpServers") or _doc.get("mcp_servers") or {}
                    if not isinstance(cc_raw, dict):
                        cc_raw = {}
                break
    except (OSError, ValueError) as e:
        _log.debug("~/.claude mcp.json read skipped: %s", e)
    # Merge installed plugins' .mcp.json blocks + Claude Code mcp.json
    # on top — user config wins on name collisions; plugins / Claude
    # Code fill in anything not set.
    try:
        from sharedagents_agent.plugins import plugin_mcp_servers as _plug_mcp

        plug_raw = _plug_mcp() or {}
    except Exception as e:  # noqa: BLE001
        _log.debug("plugin mcpServers merge skipped: %s", e)
        plug_raw = {}
    # sharedagents's own built-in MCP servers — ALWAYS available (they re-expose the
    # former web / memory / delegate builtins).
    # Lowest precedence: an operator ``mcpServers`` entry of the same name wins,
    # so a custom `web` server (or a disabling override) still takes effect.
    builtin_mcp: dict[str, Any] = {
        "memory": {"command": sys.executable, "transport": "stdio", "args": ["-m", "sharedagents_mcp.memory.server"]},
        "web": {"command": sys.executable, "transport": "stdio", "args": ["-m", "sharedagents_mcp.web_mcp_server"]},
        "delegate": {
            "command": sys.executable,
            "transport": "stdio",
            "args": ["-m", "sharedagents_mcp.delegate_mcp_server"],
        },
    }
    merged: dict[str, Any] = dict(builtin_mcp)
    if isinstance(plug_raw, dict):
        merged.update(plug_raw)
    if cc_raw:
        merged.update(cc_raw)
    merged.update(raw)
    raw = merged
    out: dict[str, MCPServerConfig] = {}
    for name, row in raw.items():
        try:
            out[name] = MCPServerConfig.from_dict(name, row)
        except (ValueError, TypeError) as e:
            _log.warning("MCP server %r ignored: %s", name, e)
    return out


def list_mcp_servers_with_source(
    global_cfg: dict[str, Any] | None = None,
) -> list[tuple[MCPServerConfig, str]]:
    """Like :func:`load_mcp_servers` but also tags each server with the
    discovery source — ``"config"`` (~/.sharedagents/config.json), ``"plugin"``
    (a bundle's .mcp.json), or ``"claude"`` (~/.claude/.mcp.json).

    Used by the ``/mcp`` slash command to render where each server came
    from, mirroring Claude Code's ``/mcp`` introspection. Source
    precedence matches :func:`load_mcp_servers`: a server present in
    multiple stores reports the highest-priority one (config > claude
    > plugin) — the same merge order the loader uses.
    """
    if global_cfg is None:
        try:
            from sharedagents_agent.agent import load_global_config

            global_cfg = load_global_config()
        except Exception:  # noqa: BLE001
            global_cfg = {}
    cfg = global_cfg or {}
    config_names = set((cfg.get("mcpServers") or cfg.get("mcp_servers") or {}).keys())
    claude_names: set[str] = set()
    try:
        import json as _json
        from pathlib import Path as _Path

        for _cand in (
            _Path.home() / ".claude" / ".mcp.json",
            _Path.home() / ".claude" / "mcp.json",
        ):
            if _cand.is_file():
                _doc = _json.loads(_cand.read_text(encoding="utf-8"))
                if isinstance(_doc, dict):
                    cc_raw = _doc.get("mcpServers") or _doc.get("mcp_servers") or {}
                    if isinstance(cc_raw, dict):
                        claude_names = set(cc_raw.keys())
                break
    except (OSError, ValueError):
        pass
    merged = load_mcp_servers(cfg)
    out: list[tuple[MCPServerConfig, str]] = []
    for name, server in sorted(merged.items()):
        if name in config_names:
            src = "config"
        elif name in claude_names:
            src = "claude"
        elif _BUILTIN_MCP_MODULES.get(name, "\0") in tuple(server.args or ()):
            # An always-on builtin (web/memory/delegate) still backed by OUR
            # module — label it "builtin", not the "plugin" fall-through. An
            # operator/plugin override of the same name has different args and
            # correctly reports its own source.
            src = "builtin"
        else:
            src = "plugin"
        out.append((server, src))
    return out


# ---- standard-pattern tool injection ------------------------------------
#
# Per the MCP spec lifecycle (https://modelcontextprotocol.io/specification/
# 2025-11-25/basic/lifecycle): client spawns server → initialize →
# notifications/initialized → tools/list. The discovered tools are
# injected DIRECTLY into the model's tool registry as
# ``mcp__<server>__<tool>`` (Claude Code's namespacing convention) so
# the model picks them via standard tool-use, not via meta-tools.
#
# We cache the discovery at the process level — re-running tools/list
# every turn would block the event loop on subprocess spawn. Cache
# is invalidated when:
#   - the agent restarts (process boot)
#   - load_global_config() returns a config whose hash differs
#   - call_mcp_namespaced raises an error (server died, retry on next list)


_MCP_NAMESPACE_SEP = "__"  # Claude Code: mcp__<server>__<tool>


def mcp_tool_namespaced(server: str, tool: str) -> str:
    """``mcp__<server>__<tool>`` — the convention Claude Code uses."""
    return f"mcp{_MCP_NAMESPACE_SEP}{server}{_MCP_NAMESPACE_SEP}{tool}"


def parse_mcp_namespaced(name: str) -> tuple[str, str] | None:
    """Inverse of ``mcp_tool_namespaced``. Returns ``(server, tool)`` or
    ``None`` for a non-MCP name. Splits on the **LAST** ``__`` in the
    remainder, so the tool is the final segment and the server is everything
    between the ``mcp__`` prefix and that separator. This is what makes a
    server whose name itself contains ``__`` (e.g. ``a__b``) parse as that
    whole server rather than being partial-prefix captured by another
    server's scoped grant (``mcp__a__*`` must NOT claim ``mcp__a__b__tool``).
    """
    prefix = f"mcp{_MCP_NAMESPACE_SEP}"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    sep = _MCP_NAMESPACE_SEP
    if sep not in rest:
        return None
    server, tool = rest.rsplit(sep, 1)
    return server, tool


# Process-wide cache. Each entry: (server_cfg, raw_tool_name, description, input_schema).
_DISCOVERED_TOOLS: list[tuple[MCPServerConfig, str, str, dict[str, Any]]] = []
_DISCOVERED_TOOLS_LOADED: bool = False
# Fingerprint of the mcpServers config the current discovery cache
# was built from. A change here (config edited/added/removed) invalidates the
# cache on the next discover_all_tools call, so no agent is served a tool set
# that no longer matches the live configuration.
_DISCOVERED_CONFIG_HASH: str | None = None
_DISCOVERED_TOOLS_LOCK: asyncio.Lock | None = None

# Parallel side-table: namespaced tool name (``mcp__<server>__<tool>``) → the
# server's declared annotations dict (``{readOnlyHint, destructiveHint, …}``).
# Kept alongside (not inside) ``_DISCOVERED_TOOLS`` so the tuple shape and its
# unpack sites stay unchanged; ``registry.discovered_mcp_tools`` reads this to
# pick a truthful criticality instead of flattening every MCP tool to
# NETWORK/mutating. Repopulated on each ``discover_all_tools`` run.
_DISCOVERED_ANNOTATIONS: dict[str, dict[str, Any]] = {}

# Hard per-server cap for boot-time discovery. ``list_tools`` already bounds
# ``initialize()``/``tools/list`` internally, but NOT the ``stdio_client``
# spawn/teardown that brackets them — a plugin mapped to bare ``echo`` (exits
# before the handshake) makes that teardown block until the OS reaps the
# child. Wrapping the whole per-server probe in ``wait_for`` caps that in the
# cooperative case; the categorical fix (running the warm on a worker thread)
# lives in the gateway. Env-overridable; non-positive/garbage → 20s.
try:
    _DISCOVERY_PER_SERVER_TIMEOUT_S: float = float(os.environ.get("SHAREDAGENTS_MCP_DISCOVERY_TIMEOUT_S", "20") or 20)
except (TypeError, ValueError):
    _DISCOVERY_PER_SERVER_TIMEOUT_S = 20.0
if not (_DISCOVERY_PER_SERVER_TIMEOUT_S > 0):
    _DISCOVERY_PER_SERVER_TIMEOUT_S = 20.0

# ---- per-session ACP MCP server overlay --------------------------------
#
# ACP clients pass ``mcpServers`` per session in ``session/new`` /
# ``session/load``. Per the ACP spec the agent MUST connect to these on
# session init. They are an OVERLAY on top of the operator's global
# config servers, scoped to and torn down with the session — keyed by
# ACP ``session_id`` so the in-memory gateway's concurrent sessions
# don't cross-contaminate. ``_ACP_SESSION_SERVERS`` holds the configs;
# ``_ACP_SESSION_DISCOVERED`` caches their discovered tools in the same
# ``(server, tool, desc, schema)`` tuple shape ``_DISCOVERED_TOOLS`` uses.
_ACP_SESSION_SERVERS: dict[str, dict[str, MCPServerConfig]] = {}
_ACP_SESSION_DISCOVERED: dict[str, list[tuple[MCPServerConfig, str, str, dict[str, Any]]]] = {}


# The always-on builtin MCP server names — the GUARANTEED floor of the reserved
# set (used even if ``load_mcp_servers`` can't be read). See
# ``_reserved_session_server_names`` for why session servers are name-reserved.
_RESERVED_MCP_SERVERS: frozenset[str] = frozenset({"memory", "web", "delegate"})


def _reserved_session_server_names() -> frozenset[str]:
    """Names an untrusted ACP session server MUST NOT use: the always-on
    builtins PLUS every currently-configured GLOBAL server (operator config /
    plugin / Claude ``.mcp.json``).

    Annotations are cached process-wide by ``mcp__<server>__<tool>`` (name only)
    and populated from GLOBAL discovery, and the runner overlays session servers
    ON TOP of the global ones at call time (``{**load_mcp_servers(), **acp_...}``
    in ``tools/runners``). So a session server shadowing ANY global name would
    inherit that global server's ``readOnlyHint`` (gate → read_only) while its
    OWN, possibly mutating, handler executes — a too-permissive gate bypass.
    Reserving every global name closes that. Operator/global config is trusted
    (its own annotations are discovered); only the session/env channel is
    reserved. Best-effort: on any read error we still return the builtin floor.
    """
    names = set(_RESERVED_MCP_SERVERS)
    try:
        names |= set(load_mcp_servers().keys())
    except Exception:  # noqa: BLE001
        pass
    return frozenset(names)


def register_acp_session_servers(session_id: str, servers: list[MCPServerConfig]) -> None:
    """Register a session's ACP-supplied MCP servers. No-op on an empty
    ``session_id`` (a session that never got a real id can't be torn
    down deterministically, so we don't leak entries for it).

    A server whose name collides with any reserved name
    (:func:`_reserved_session_server_names` — the builtins + every configured
    global server) is DROPPED, so an ACP client can't shadow a global server to
    inherit its criticality annotations.
    """
    if not session_id:
        return
    reserved = _reserved_session_server_names()
    kept: dict[str, MCPServerConfig] = {}
    for s in servers:
        if s.name in reserved:
            _log.warning(
                "acp mcp: refusing session server %r — collides with a configured global MCP server name",
                s.name,
            )
            continue
        kept[s.name] = s
    _ACP_SESSION_SERVERS[session_id] = kept


def acp_session_servers(session_id: str) -> dict[str, MCPServerConfig]:
    """Return a copy of the ACP MCP servers registered for ``session_id``
    (empty dict if none)."""
    return dict(_ACP_SESSION_SERVERS.get(session_id, {}))


def unregister_acp_session(session_id: str) -> None:
    """Drop a session's ACP MCP servers + their discovered-tool cache.
    Called from the backend's ``close_session``.
    """
    _ACP_SESSION_SERVERS.pop(session_id, None)
    _ACP_SESSION_DISCOVERED.pop(session_id, None)


async def discover_acp_session_tools(
    session_id: str,
) -> list[tuple[MCPServerConfig, str, str, dict[str, Any]]]:
    """Discover tools for a session's ACP-supplied MCP servers and cache
    them in ``_ACP_SESSION_DISCOVERED[session_id]``.

    Best-effort, mirroring :func:`discover_all_tools`: a server that
    fails to spawn or whose ``tools/list`` errors is logged and skipped
    so one bad ACP server can't block the whole session.

    Servers are probed CONCURRENTLY (``asyncio.gather``) so N servers
    cost ~one server's latency, not the sum — this can run on the
    session-setup path, so serial probing would let a slow server delay
    every other server's tools.
    """
    servers = list(_ACP_SESSION_SERVERS.get(session_id, {}).items())
    if not servers:
        _ACP_SESSION_DISCOVERED[session_id] = []
        return []

    async def _one(name: str, srv: MCPServerConfig):
        # Bound EACH ACP server's probe with the same per-server cap as
        # global discovery. Without the outer wait_for, a single hung ACP server
        # would block the whole ``asyncio.gather`` below (all N coroutines) and
        # wedge session discovery (the second-process-freeze class).
        try:
            ok, tools, err = await asyncio.wait_for(list_tools(srv), timeout=_DISCOVERY_PER_SERVER_TIMEOUT_S)
        except asyncio.TimeoutError:
            _log.warning(
                "acp mcp discover %r: timed out after %gs",
                name,
                _DISCOVERY_PER_SERVER_TIMEOUT_S,
            )
            return []
        except Exception as e:  # noqa: BLE001
            _log.warning("acp mcp discover %r: spawn/init failed: %s", name, e)
            return []
        if not ok:
            _log.warning("acp mcp discover %r: tools/list failed: %s", name, err)
            return []
        rows: list[tuple[MCPServerConfig, str, str, dict[str, Any]]] = []
        for t in tools or []:
            tname = str(t.get("name") or "").strip()
            if not tname:
                continue
            schema = t.get("inputSchema") or {"type": "object", "properties": {}}
            desc = str(t.get("description") or "").strip()
            rows.append((srv, tname, desc, schema))
            # A per-session ACP tool never gets a criticality DISCOUNT —
            # its ``readOnlyHint`` is deliberately NOT recorded, so _mcp_kind_for
            # finds no read hint → NETWORK → mutating (too-strict is safe; a
            # readOnlyHint is honored only for the fixed, non-colliding always-on
            # global servers). But a ``destructiveHint`` is a floor ESCALATION,
            # not a discount, and since the most-cautious
            # classification wins it must still resolve WRITE — so we record ONLY that
            # hint. Recording only the escalation is contamination-safe: a stale
            # or cross-session destructive hint can only make the gate stricter
            # than READ, never more permissive.
            ann = t.get("annotations")
            if isinstance(ann, dict) and ann.get("destructiveHint"):
                _DISCOVERED_ANNOTATIONS[mcp_tool_namespaced(srv.name, tname)] = {"destructiveHint": True}
        return rows

    results = await asyncio.gather(*(_one(name, srv) for name, srv in servers), return_exceptions=True)
    out: list[tuple[MCPServerConfig, str, str, dict[str, Any]]] = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
    _ACP_SESSION_DISCOVERED[session_id] = out
    return out


# Env channel: a session's ACP MCP servers must reach the tool-runner
# SUBPROCESS (``_default_run_tool`` spawns one process per tool call),
# which can't see the parent's in-memory ``_ACP_SESSION_SERVERS``. The
# parent serializes this session's servers into this env var; the runner
# overlays them onto ``load_mcp_servers()``. Env is per-subprocess so it
# stays correctly scoped to one session even under concurrent turns.
ACP_MCP_SERVERS_ENV = "SHAREDAGENTS_ACP_MCP_SERVERS"


def encode_acp_servers_for_env(servers: dict[str, MCPServerConfig]) -> str:
    """JSON-encode a ``{name: MCPServerConfig}`` map for the env channel."""
    return json.dumps({name: asdict(cfg) for name, cfg in servers.items()})


def acp_servers_from_env(
    environ: dict[str, str] | None = None,
) -> dict[str, MCPServerConfig]:
    """Decode ACP MCP servers from :data:`ACP_MCP_SERVERS_ENV`. Returns
    an empty dict when absent, empty, or malformed (the runner must
    degrade to config-only servers, never crash).
    """
    env = os.environ if environ is None else environ
    raw = env.get(ACP_MCP_SERVERS_ENV) or ""
    if not raw.strip():
        return {}
    try:
        doc = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(doc, dict):
        return {}
    out: dict[str, MCPServerConfig] = {}
    reserved = _reserved_session_server_names()
    for name, d in doc.items():
        if not isinstance(d, dict):
            continue
        if str(name) in reserved:
            # Defense-in-depth: even a hand-crafted env can't shadow a global
            # server name (register_acp_session_servers already drops these
            # upstream, but this channel is decoded in the tool-runner subprocess).
            continue
        try:
            out[str(name)] = MCPServerConfig(
                name=str(d.get("name") or name),
                transport=str(d.get("transport") or "stdio"),
                command=str(d.get("command") or ""),
                args=tuple(d.get("args") or ()),
                env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
                cwd=d.get("cwd"),
                url=str(d.get("url") or ""),
                headers={str(k): str(v) for k, v in (d.get("headers") or {}).items()},
                description=str(d.get("description") or ""),
            )
        except (TypeError, ValueError):
            continue
    return out


# The always-on builtin MCP servers and the module that owns each one's static
# ``TOOL_DEFS`` — used to seed the discovery cache synchronously (no spawn).
_BUILTIN_MCP_MODULES: dict[str, str] = {
    "web": "sharedagents_mcp.web_mcp_server",
    "memory": "sharedagents_mcp.memory.server",
    "delegate": "sharedagents_mcp.delegate_mcp_server",
}


def _append_builtin_tool_defs(
    out_list: list[tuple[MCPServerConfig, str, str, dict[str, Any]]],
    ann_dict: dict[str, dict[str, Any]],
    have: set[tuple[str, str]],
    cfgs: dict[str, MCPServerConfig],
) -> int:
    """Append the ALWAYS-ON builtin tools (from their static ``TOOL_DEFS``) that
    aren't already in ``have`` to ``out_list`` / ``ann_dict``. Shared by
    ``seed_builtin_mcp_tools`` (mutates the live globals at startup) and
    ``discover_all_tools`` (fills LOCAL pre-publish state). Only a server still
    backed by OUR module is seeded (an operator override is left to discovery)."""
    import importlib

    n = 0
    for name, mod_path in _BUILTIN_MCP_MODULES.items():
        srv = cfgs.get(name)
        if srv is None or mod_path not in tuple(srv.args or ()):
            continue  # disabled, or operator-overridden → let discovery handle it
        try:
            defs = getattr(importlib.import_module(mod_path), "TOOL_DEFS", [])
        except Exception:  # noqa: BLE001
            continue
        for d in defs:
            tname = str(d.get("name") or "")
            if not tname or (srv.name, tname) in have:
                continue
            schema = d.get("inputSchema") or {"type": "object", "properties": {}}
            out_list.append((srv, tname, str(d.get("description") or ""), schema))
            ann = d.get("annotations")
            if isinstance(ann, dict) and ann:
                ann_dict[mcp_tool_namespaced(srv.name, tname)] = ann
            have.add((srv.name, tname))
            n += 1
    return n


def seed_builtin_mcp_tools() -> int:
    """Synchronously seed the discovery cache with the ALWAYS-ON builtin
    web/memory/delegate tools from their static ``TOOL_DEFS`` — NO subprocess
    spawn, so it's safe on the startup hot path (the off-thread discovery warm
    exists precisely to keep spawns off it).

    Without this, ``discover_all_tools`` only runs in the background warm, so a
    worker request built in the startup window sees an EMPTY cache and the
    always-on MCP tools are absent from ``tools[]`` AND strict-allowlist
    expansion until the warm lands. Called ONCE at gateway
    startup (single-threaded) so its in-place mutation of the globals is safe;
    the warm folds the same backfill into its atomic publish. Idempotent.
    """
    cfgs = load_mcp_servers()
    have = {(s.name, t) for s, t, _, _ in _DISCOVERED_TOOLS}
    return _append_builtin_tool_defs(_DISCOVERED_TOOLS, _DISCOVERED_ANNOTATIONS, have, cfgs)


def _mcp_config_hash(servers: dict[str, MCPServerConfig]) -> str:
    """Stable fingerprint of the loaded mcpServers config.

    Covers each server's identity and spawn/transport shape so any config
    change (added/removed server, changed command/args/url/transport)
    yields a different hash and forces re-discovery.
    """
    import hashlib

    payload = sorted(
        (
            name,
            getattr(c, "transport", "") or "",
            getattr(c, "command", "") or "",
            tuple(getattr(c, "args", ()) or ()),
            getattr(c, "url", "") or "",
        )
        for name, c in servers.items()
    )
    return hashlib.sha256(json.dumps(payload, default=str, sort_keys=True).encode("utf-8")).hexdigest()


async def discover_all_tools(
    *,
    force_refresh: bool = False,
) -> list[tuple[MCPServerConfig, str, str, dict[str, Any]]]:
    """Spawn every configured MCP server, run tools/list against each,
    and return ``(server, tool_name, description, input_schema)`` tuples
    for every discovered tool.

    Subsequent calls reuse the cached result unless ``force_refresh``
    is set or the cache is empty. A failed server is logged and skipped
    (other servers' tools still surface).

    Discovery progress is mirrored to stdout (the daemon's gateway.log
    sink) via ``print(...)`` because the agent's logging is configured
    by ``sharedagents_telegram.bot``'s FileHandler — the rest of the runtime
    surfaces info via ``print()`` to stdout, and we follow that pattern
    so MCP discovery lands alongside ``acp listening on ...`` etc.
    """
    global _DISCOVERED_TOOLS, _DISCOVERED_TOOLS_LOADED, _DISCOVERED_TOOLS_LOCK
    global _DISCOVERED_ANNOTATIONS, _DISCOVERED_CONFIG_HASH
    if _DISCOVERED_TOOLS_LOCK is None:
        _DISCOVERED_TOOLS_LOCK = asyncio.Lock()
    async with _DISCOVERED_TOOLS_LOCK:
        # Load the live config up front and fingerprint it, so a CONFIG CHANGE
        # (not just force_refresh / restart / explicit invalidation) invalidates
        # the cache — no agent is ever offered a tool set that no longer matches
        # the live mcpServers configuration.
        servers = load_mcp_servers()
        try:
            cfg_hash: str | None = _mcp_config_hash(servers)
        except Exception:  # noqa: BLE001 — never let hashing wedge discovery
            cfg_hash = None
        if _DISCOVERED_TOOLS_LOADED and not force_refresh and cfg_hash == _DISCOVERED_CONFIG_HASH:
            return list(_DISCOVERED_TOOLS)
        # Sentinel — written progressively so a discovery that hangs
        # or crashes still leaves a forensic trail.
        sentinel: Any | None = None
        sentinel_lines: list[str] = [f"started at: {time.time()}"]
        try:
            from sharedagents_agent.agent import sharedagents_root

            sentinel = sharedagents_root() / "mcp_discovery.last"
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("\n".join(sentinel_lines) + "\n")
        except Exception:  # noqa: BLE001
            sentinel = None
        sentinel_lines.append(f"servers: {sorted(servers)}")
        if sentinel is not None:
            try:
                sentinel.write_text("\n".join(sentinel_lines) + "\n")
            except Exception:  # noqa: BLE001
                pass
        if not servers:
            print("[mcp] no mcpServers configured — skipping discovery", file=sys.stderr, flush=True)
            _DISCOVERED_TOOLS = []
            _DISCOVERED_TOOLS_LOADED = True
            _DISCOVERED_CONFIG_HASH = cfg_hash
            return []
        print(
            f"[mcp] discovering tools across {len(servers)} server(s): {sorted(servers.keys())}",
            file=sys.stderr,
            flush=True,
        )

        # Probe every server CONCURRENTLY, each bracketed by a hard
        # ``wait_for`` cap — so one broken/slow server neither serializes in
        # front of the healthy ones nor stalls the warm past the cap. Row
        # order follows ``servers`` insertion order (gather preserves the
        # coroutine order) so discovery output stays deterministic.
        async def _probe_one(
            name: str,
            srv: MCPServerConfig,
        ) -> tuple[
            str,
            list[tuple[MCPServerConfig, str, str, dict[str, Any]]],
            dict[str, dict[str, Any]],
            str,
        ]:
            try:
                ok, tools, err = await asyncio.wait_for(list_tools(srv), timeout=_DISCOVERY_PER_SERVER_TIMEOUT_S)
            except asyncio.TimeoutError:
                return name, [], {}, (f"{name}: discovery timed out after {_DISCOVERY_PER_SERVER_TIMEOUT_S:g}s")
            except Exception as e:  # noqa: BLE001
                return name, [], {}, f"{name}: spawn/init failed: {type(e).__name__}: {e}"
            if not ok:
                return name, [], {}, f"{name}: tools/list failed: {err}"
            rows: list[tuple[MCPServerConfig, str, str, dict[str, Any]]] = []
            ann_local: dict[str, dict[str, Any]] = {}
            for t in tools or []:
                tname = str(t.get("name") or "").strip()
                if not tname:
                    continue
                schema = t.get("inputSchema") or {"type": "object", "properties": {}}
                desc = str(t.get("description") or "").strip()
                rows.append((srv, tname, desc, schema))
                ann = t.get("annotations")
                if isinstance(ann, dict) and ann:
                    # keyed by the namespaced name registry.discovered_mcp_tools uses
                    ann_local[mcp_tool_namespaced(srv.name, tname)] = ann
            tnames = ", ".join(str(t.get("name") or "?") for t in (tools or []))
            return name, rows, ann_local, f"{name}: ok ({len(tools or [])} tool(s): {tnames})"

        # Build the fresh discovery locally and SWAP it in atomically at the end.
        # Do NOT clear the live cache upfront — that would leave a window (the
        # whole async probe) where the seeded builtin tools/annotations are gone
        # and their readOnlyHint mis-gates as mutating. The
        # live cache keeps serving the startup seed until the swap below.
        probed = await asyncio.gather(
            *(_probe_one(name, srv) for name, srv in servers.items()),
            return_exceptions=True,
        )
        out: list[tuple[MCPServerConfig, str, str, dict[str, Any]]] = []
        new_ann: dict[str, dict[str, Any]] = {}
        for r in probed:
            if isinstance(r, BaseException):
                # A per-server probe should never raise (it catches its own
                # errors), but gather with return_exceptions keeps one rogue
                # failure from sinking the rest.
                msg = f"probe crashed: {type(r).__name__}: {r}"
                sentinel_lines.append(msg)
                print(f"[mcp] discover {msg}", file=sys.stderr, flush=True)
                _log.warning("mcp discover: %s", msg)
                continue
            name, rows, ann_local, status = r
            out.extend(rows)
            new_ann.update(ann_local)
            sentinel_lines.append(status)
            print(f"[mcp] discover {status}", file=sys.stderr, flush=True)
            if rows:
                _log.info("mcp discover %r: %d tool(s)", name, len(rows))
            else:
                _log.warning("mcp discover %r: %s", name, status)
            if sentinel is not None:
                try:
                    sentinel.write_text("\n".join(sentinel_lines) + "\n")
                except Exception:
                    pass  # noqa: BLE001
        # Backfill any ALWAYS-ON builtin whose probe FAILED (timeout/spawn error)
        # into the LOCAL state from its static TOOL_DEFS — a valid builtin tool
        # must never vanish just because its subprocess didn't answer this round.
        # Idempotent: skips builtins the probe returned.
        try:
            _bf = _append_builtin_tool_defs(out, new_ann, {(s.name, t) for s, t, _, _ in out}, servers)
            if _bf:
                _log.info("mcp discover: backfilled %d builtin tool(s) from TOOL_DEFS", _bf)
        except Exception:  # noqa: BLE001
            pass
        # ATOMIC publish for lock-free readers on OTHER threads (the warm runs
        # off-thread while sync readers read these globals lock-free).
        # Rebind each global in ONE store (a name rebind is atomic;
        # the old ``clear()`` + ``update()`` was two ops with an empty window).
        # Publish ANNOTATIONS first so a reader that sees a freshly-published tool
        # already finds its readOnlyHint (never mis-gates a read-only tool).
        _DISCOVERED_ANNOTATIONS = new_ann
        _DISCOVERED_TOOLS = out
        _DISCOVERED_CONFIG_HASH = cfg_hash
        _DISCOVERED_TOOLS_LOADED = True
        sentinel_lines.append(f"DONE: {len(out)} total tool(s)")
        if sentinel is not None:
            try:
                sentinel.write_text("\n".join(sentinel_lines) + "\n")
            except Exception:
                pass  # noqa: BLE001
        return list(out)


def invalidate_discovery_cache() -> None:
    """Force the next ``discover_all_tools`` call to re-spawn servers."""
    global _DISCOVERED_TOOLS, _DISCOVERED_TOOLS_LOADED, _DISCOVERED_CONFIG_HASH
    _DISCOVERED_TOOLS = []
    _DISCOVERED_TOOLS_LOADED = False
    _DISCOVERED_CONFIG_HASH = None


# ---- session helpers --------------------------------------------------


def _mcp_attr(obj: Any, camel: str, default: Any = None) -> Any:
    """Read an MCP SDK model field by its wire (camelCase) name.

    mcp 1.x exposes fields as camelCase attributes (``isError``,
    ``inputSchema``, ``readOnlyHint``); mcp 2.x renamed them to
    snake_case (``is_error``, ``input_schema``, ``read_only_hint``).
    """
    val = getattr(obj, camel, None)
    if val is None:
        snake = "".join(f"_{c.lower()}" if c.isupper() else c for c in camel)
        val = getattr(obj, snake, None)
    return default if val is None else val


@contextlib.asynccontextmanager
async def _streamable_http_streams(
    url: str,
    headers: dict[str, str] | None,
) -> AsyncIterator[tuple[Any, Any]]:
    """Open a Streamable-HTTP transport and yield ``(read, write)``.

    The MCP SDK renamed ``streamablehttp_client`` (1.x: takes ``headers=``
    and yields a 3-tuple) to ``streamable_http_client`` (2.x: takes a
    pre-configured ``http_client`` and yields a 2-tuple). Support both.
    """
    from mcp.client import streamable_http as _sh

    legacy = getattr(_sh, "streamablehttp_client", None)
    if legacy is not None:
        async with legacy(url, headers=headers) as (read, write, _close):
            yield read, write
        return
    async with (
        _sh.create_mcp_http_client(headers=headers) as http_client,
        _sh.streamable_http_client(url, http_client=http_client) as (read, write),
    ):
        yield read, write


@contextlib.asynccontextmanager
async def mcp_session(
    server: MCPServerConfig,
    *,
    timeout_s: float = 30.0,
) -> AsyncIterator[Any]:
    """Open an `mcp.ClientSession` against `server` and yield it
    initialized + ready to use.

    Imports inside the function body so callers without the optional
    `mcp` package don't pay the import cost at module load.
    Raises `RuntimeError` if the SDK isn't installed.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as e:
        raise RuntimeError("mcp package not installed. Install with: pip install mcp") from e

    if server.transport == "stdio":
        params = StdioServerParameters(
            command=server.command,
            args=list(server.args),
            # The SDK merges this with the parent env; pass exactly the
            # operator-specified overrides.
            env={**os.environ, **server.env},
            cwd=server.cwd,
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as sess:
            await asyncio.wait_for(sess.initialize(), timeout=timeout_s)
            yield sess
        return

    if server.transport == "http":
        async with (
            _streamable_http_streams(server.url, server.headers or None) as (read, write),
            ClientSession(read, write) as sess,
        ):
            await asyncio.wait_for(sess.initialize(), timeout=timeout_s)
            yield sess
        return

    if server.transport == "sse":
        from mcp.client.sse import sse_client

        async with sse_client(server.url, headers=server.headers or None) as (read, write):
            async with ClientSession(read, write) as sess:
                await asyncio.wait_for(sess.initialize(), timeout=timeout_s)
                yield sess
        return

    raise RuntimeError(f"unsupported MCP transport: {server.transport!r}")


# ---- one-shot helpers (the runner's hot path) -------------------------


def _flatten_content(items: Any) -> str:
    """Render an MCP `CallToolResult.content` list (a list of
    `TextContent`/`ImageContent`/...) into a plain string.

    The sharedagents tool layer hands a string back to the model — images
    and other binary blobs are cited by their content-type and length,
    not embedded.
    """
    if items is None:
        return ""
    if isinstance(items, str):
        return items
    if not isinstance(items, list):
        try:
            return json.dumps(items, default=str, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(items)
    parts: list[str] = []
    for it in items:
        if it is None:
            continue
        # MCP SDK emits pydantic models; fall back to dict access.
        if hasattr(it, "type"):
            t = getattr(it, "type", None)
        elif isinstance(it, dict):
            t = it.get("type")
        else:
            t = None
        if t == "text":
            parts.append(getattr(it, "text", None) or (it.get("text") if isinstance(it, dict) else "") or "")
        elif t == "image":
            mime = _mcp_attr(it, "mimeType") or (it.get("mimeType") if isinstance(it, dict) else "") or "image/*"
            data = getattr(it, "data", None) or (it.get("data") if isinstance(it, dict) else "") or ""
            parts.append(f"[image: {mime}, {len(data)} bytes (base64)]")
        elif t == "resource":
            uri = getattr(it, "uri", None) or (it.get("uri") if isinstance(it, dict) else "") or "?"
            parts.append(f"[resource: {uri}]")
        else:
            try:
                parts.append(json.dumps(it, default=str, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                parts.append(str(it))
    return "\n".join(p for p in parts if p)


async def call_tool(
    server: MCPServerConfig,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    timeout_s: float = 60.0,
) -> tuple[bool, str]:
    """One-shot: open a session, call the tool, close. Returns
    `(ok, body_string)`. `ok=False` carries either an MCP-reported
    `isError` flag or a transport / SDK exception message.
    """
    try:
        async with mcp_session(server) as sess:
            result = await asyncio.wait_for(
                sess.call_tool(tool_name, arguments=arguments or {}),
                timeout=timeout_s,
            )
    except asyncio.TimeoutError:
        return False, f"mcp_call timeout after {timeout_s}s"
    except RuntimeError as e:
        return False, f"mcp_call failed: {e}"
    except Exception as e:  # noqa: BLE001
        return False, f"mcp_call {type(e).__name__}: {e}"
    is_error = bool(_mcp_attr(result, "isError", False))
    body = _flatten_content(getattr(result, "content", None))
    return (not is_error, body)


async def list_tools(
    server: MCPServerConfig,
    *,
    timeout_s: float = 15.0,
) -> tuple[bool, list[dict[str, Any]], str]:
    """One-shot: open a session, list tools, close. Returns
    `(ok, tool_dicts, error_string)`. Each tool dict has `name`,
    `description`, `inputSchema`. On failure: ``(False, [], reason)``.

    The error string lets the caller surface what went wrong to the
    model — the previous shape ``(False, [])`` only logged the reason
    locally, so the model just saw "failed against {server!r}" with no
    way to fix it (was it auth? DNS? a missing dependency? a
    SDK-version mismatch?). Now the reason rides back with the result.
    """
    try:
        async with mcp_session(server) as sess:
            result = await asyncio.wait_for(sess.list_tools(), timeout=timeout_s)
    except asyncio.TimeoutError:
        msg = f"timeout after {timeout_s}s"
        _log.warning("mcp list_tools timeout against %s", server.name)
        return False, [], msg
    except RuntimeError as e:
        # Raised from `mcp_session` for unsupported transport / missing
        # SDK. Surface verbatim — these are fixable misconfigurations.
        return False, [], str(e)
    except Exception as e:  # noqa: BLE001 — SDK / transport / protocol errors
        msg = f"{type(e).__name__}: {e}"
        _log.warning("mcp list_tools failed against %s: %s", server.name, msg)
        return False, [], msg
    out: list[dict[str, Any]] = []
    for t in getattr(result, "tools", []) or []:
        # Carry the standard MCP tool annotations through so discovery can pick
        # a truthful criticality (readOnlyHint→READ, destructiveHint→WRITE)
        # instead of flattening every MCP tool to NETWORK/mutating.
        ann = getattr(t, "annotations", None)
        ann_dict: dict[str, Any] = {}
        if ann is not None:
            for _k in ("readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint", "title"):
                _v = _mcp_attr(ann, _k)
                if _v is not None:
                    ann_dict[_k] = _v
        out.append(
            {
                "name": getattr(t, "name", ""),
                "description": getattr(t, "description", "") or "",
                "inputSchema": _mcp_attr(t, "inputSchema") or {},
                "annotations": ann_dict,
            }
        )
    return True, out, ""


__all__ = [
    "MCPServerConfig",
    "call_tool",
    "list_tools",
    "load_mcp_servers",
    "list_mcp_servers_with_source",
    "mcp_session",
]
