"""Extracted tool-runner family: integrations (de-monolith of runners.py).

Behavior-preserving split — code moved verbatim.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sharedagents_core.cwdutil import safe_getcwd

from ._utils import (
    ToolResult,
    _err,
    _truncate,
)

# ---- MCP / ACP tool runners ----------------------------------------
#
# The MCP runners go through `sharedagents.mcp_client`, which opens a
# one-shot session per call (see that module's docstring for the
# rationale). The ACP runner goes through `sharedagents_acp.client` — it
# picks the right transport from the configured `acp_agents` row
# (stdio / ws / http), runs the standard ACP handshake, drives one
# prompt turn, and closes the session.
#
# All three are NETWORK-kind tools, so the existing permission gate
# treats them like web_fetch/delegate_claude — `auto` runs them
# silently, `ask` prompts.


async def run_mcp_list_tools_async(args: dict[str, Any]) -> ToolResult:
    server_name = args.get("server_name")
    if not isinstance(server_name, str) or not server_name:
        return _err("server_name required")
    from sharedagents_mcp import mcp_client as _mcp

    servers = _mcp.load_mcp_servers()
    server = servers.get(server_name)
    if server is None:
        configured = ", ".join(sorted(servers)) or "(none configured)"
        return _err(
            f"unknown MCP server: {server_name!r}. "
            f"Configured: {configured}. Edit `mcp_servers` in "
            "~/.sharedagents/config.json to add one."
        )
    ok, tools, err = await _mcp.list_tools(server)
    if not ok:
        return _err(f"mcp_list_tools failed against {server_name!r}: {err or 'unknown error'}")
    if not tools:
        return ToolResult(
            content=f"# MCP server {server_name!r}\n\n(server reports no tools)",
            ok=True,
        )
    lines = [f"# MCP server {server_name!r}", ""]
    for t in tools:
        lines.append(f"## {t.get('name', '?')}")
        desc = (t.get("description") or "").strip()
        if desc:
            lines.append("")
            lines.append(desc)
        schema = t.get("inputSchema") or {}
        if schema:
            lines.append("")
            lines.append("Input schema:")
            lines.append("```json")
            try:
                lines.append(json.dumps(schema, ensure_ascii=False, indent=2))
            except (TypeError, ValueError):
                lines.append(str(schema))
            lines.append("```")
        lines.append("")
    body = "\n".join(lines).rstrip() + "\n"
    body, truncated = _truncate(body)
    return ToolResult(content=body, ok=True, truncated=truncated)


async def run_mcp_call_async(args: dict[str, Any]) -> ToolResult:
    server_name = args.get("server_name")
    tool_name = args.get("tool_name")
    arguments = args.get("arguments")
    if not isinstance(server_name, str) or not server_name:
        return _err("server_name required")
    if not isinstance(tool_name, str) or not tool_name:
        return _err("tool_name required")
    # Local models routinely double-encode the nested ``arguments`` object as a
    # JSON *string* (e.g. ``arguments='{"url": "..."}'``) instead of a real
    # object. Coerce that common shape rather than dead-ending the model in a
    # retry loop on an opaque "must be an object" error. ``None`` / empty ->
    # ``{}`` so a zero-arg tool call still works.
    if arguments is None:
        arguments = {}
    elif isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            arguments = {}
        else:
            try:
                arguments = json.loads(stripped)
            except (ValueError, TypeError):
                return _err(
                    "arguments must be a JSON object; got an unparseable "
                    "string. Pass it as an object, e.g. "
                    '{"url": "..."}, not a quoted JSON string.'
                )
    if not isinstance(arguments, dict):
        return _err(
            "arguments must be a JSON object (got "
            f"{type(arguments).__name__}). Pass it as an object, e.g. "
            '{"url": "..."}.'
        )
    from sharedagents_mcp import mcp_client as _mcp

    servers = _mcp.load_mcp_servers()
    # Overlay this session's ACP-supplied MCP servers (passed by the ACP client
    # in session/new, serialized into env by the parent process) — ADD-ONLY, so
    # a session server can never OVERRIDE a global one (same reserved-namespace
    # rule as the live namespaced dispatch path). This ``mcp_call`` meta-tool is
    # removed from BUILTIN_TOOLS post-#729 and blocked at the gate, so this path
    # is effectively dead; kept consistent for defense-in-depth.
    servers = {
        **servers,
        **{k: v for k, v in _mcp.acp_servers_from_env().items() if k not in servers},
    }
    server = servers.get(server_name)
    if server is None:
        configured = ", ".join(sorted(servers)) or "(none configured)"
        return _err(
            f"unknown MCP server: {server_name!r}. "
            f"Configured: {configured}. Edit `mcp_servers` in "
            "~/.sharedagents/config.json to add one."
        )
    ok, body = await _mcp.call_tool(server, tool_name, arguments)
    body, truncated = _truncate(body or "(empty result)")
    header = f"[mcp {server_name}/{tool_name}]"
    full = f"{header}\n\n{body}"
    return ToolResult(content=full, ok=ok, truncated=truncated)


def _load_acp_agents_cfg() -> dict[str, dict[str, Any]]:
    """Read the ``acp_agents`` map out of the global config.

    Returns ``{name: row_dict}`` for every entry; bad rows (non-dict)
    are dropped silently. We deliberately do NOT validate row shape
    here — the per-row dispatcher in ``_drive_acp_session_async``
    inspects ``url`` / ``command`` to pick the transport, and an
    unrecognised shape is reported as a runtime error there. Kept as
    a module-level helper so tests can monkeypatch it.
    """
    try:
        from sharedagents_agent.agent import load_global_config

        global_cfg = load_global_config() or {}
    except Exception:  # noqa: BLE001
        global_cfg = {}
    raw = global_cfg.get("acp_agents") or {}
    if not isinstance(raw, dict):
        return {}
    return {name: row for name, row in raw.items() if isinstance(row, dict)}


async def _drive_acp_session_async(
    cfg: dict[str, Any],
    prompt: str,
    *,
    mode: str,
    cwd: str | None,
    on_delta: Callable[[str], None] | None,
) -> tuple[bool, str, dict[str, Any]]:
    """One-shot drive of a remote ACP agent over the appropriate transport.

    Picks the transport from ``cfg``:

    - ``cfg["command"]`` (list[str]) → stdio (``spawn_stdio_agent``).
    - ``cfg["url"]`` starts with ``ws://`` / ``wss://`` → ``connect_ws``.
    - ``cfg["url"]`` starts with ``http://`` / ``https://`` →
      ``connect_http``.

    Runs the standard handshake (``initialize`` → ``session/new`` →
    ``session/prompt``), assembles agent-message-chunk text into the
    return body, forwards each chunk through ``on_delta``, and closes
    the session + transport on the way out. Returns
    ``(ok, body, meta)`` so the caller renders the same ``[acp …]``
    header shape as before.
    """
    import acp.schema as _acp_schema

    from sharedagents_acp.client import (
        connect_http,
        connect_ws,
        spawn_stdio_agent,
    )

    text_buf: list[str] = []
    meta: dict[str, Any] = {"state": "unknown", "error": None}

    # --- Pre-flight checklist gate + secret egress SINK guard ---
    # for the raw acp_call escape hatch.
    #
    # acp_call is a distinct single-shot delegate path that NEVER funnels through
    # the delegate_shims chokepoint, so the shared pre-flight gate never sees it
    # unless we invoke it HERE. The remote ACP server fronts a frontier model
    # (generic ACP is in the expensive engine set), so acp_call is a model-facing
    # external delegate and MUST pass BOTH gates every other delegate path passes:
    #   (1) the delegation checklist gate, and
    #   (2) the secret screen.
    #
    # (1) Checklist gate. acp_call's tool schema
    # exposes only agent_name/prompt/mode/cwd — it has no goal/constraints/
    # acceptance_criteria/budget_usd fields (widening the schema is a separate
    # concern and would break every call). So, exactly like the
    # operator-slash path (/claude etc.), we AUTO-FILL the checklist from the
    # operator-supplied prompt via ``operator_slash_checklist`` and thread it into
    # the args we hand ``evaluate_preflight``. Net effect: an acp_call now
    # satisfies the checklist gate with an auto-derived checklist rather than being exempt,
    # and it runs on the resolved ACP engine kind (external, not local) so the
    # gate actually engages. Fail CLOSED — on an incomplete/blocked checklist we
    # return the caller's error tuple and open ZERO sessions.
    #
    # (2) Secret SINK guard. ``evaluate_preflight`` already
    # screens the prompt via its args-screen, but the pre-prompt METADATA still
    # rides the wire and must be screened too: cwd rides the stdio spawn
    # working-dir and session/new, and `mode` rides session/set_mode — all BEFORE
    # session/prompt. Screen every outbound value through the SAME shared secret
    # guard before ANY of it leaves the process. Hard-block — no
    # override, fail CLOSED: on a secret we return the error tuple the caller
    # surfaces and never open the session (do NOT redact-and-send).
    from sharedagents_agent import delegation_gate as _dgate
    from sharedagents_agent.operator_delegation import operator_slash_checklist as _op_checklist

    # ACP is in the expensive/external engine set (delegation_gate._EXPENSIVE_
    # ENGINE_KINDS) — resolve to that kind, NOT a local kind, so the checklist
    # gate engages. Auto-derive the four checklist fields from the prompt, then
    # merge the prompt itself so the gate's secret screen also sees it.
    _preflight_args = {**_op_checklist(prompt or ""), "prompt": prompt}
    _pf = _dgate.evaluate_preflight("acp", _preflight_args)
    if not _pf.ok:
        try:
            from sharedagents_agent.tools._log import log_event as _log_event

            _log_event(
                "DELEGATE_PREFLIGHT_BLOCKED",
                engine="acp",
                tool="acp_call",
                missing_fields=list(_pf.missing_fields),
                secret_findings=list(_pf.secret_findings),
            )
        except Exception:  # noqa: BLE001
            pass
        return False, "", {"state": "error", "error": _pf.reason}
    try:
        for _egress_val in (prompt, cwd, mode):
            if isinstance(_egress_val, str) and _egress_val:
                _dgate.guard_egress_payload("acp", _egress_val)
    except _dgate.SecretEgressBlocked as _seb:
        try:
            from sharedagents_agent.tools._log import log_event as _log_event

            _log_event(
                "DELEGATE_SECRET_EGRESS_BLOCKED",
                engine="acp",
                tool="acp_call",
                findings=list(_seb.findings),
            )
        except Exception:  # noqa: BLE001
            pass
        return False, "", {"state": "error", "error": _seb.reason}

    # Pick the transport based on the row's shape.
    command = cfg.get("command")
    url = cfg.get("url")
    bearer = cfg.get("token") or None

    client = None
    aux: Any = None  # process | aiohttp.ClientSession | None
    transport_kind: str
    try:
        if isinstance(command, list) and command:
            transport_kind = "stdio"
            # The acp_call escape hatch spawns a
            # THIRD-PARTY external ACP server (this row is in the expensive
            # engine set — see the gate above), so it must NOT inherit the
            # parent os.environ. Reduce to the least-privilege allowlist and pass it
            # verbatim. acp_call carries no per-agent api_keys, so no overrides.
            from sharedagents_agent.delegate_env import build_third_party_spawn_env

            client, aux = await spawn_stdio_agent(
                [str(x) for x in command],
                cwd=cwd,
                env=build_third_party_spawn_env(),
                inherit_parent_env=False,
            )
        elif isinstance(url, str) and url.startswith(("ws://", "wss://")):
            transport_kind = "ws"
            headers = {"Authorization": f"Bearer {bearer}"} if bearer else None
            client, aux = await connect_ws(url, headers=headers)
        elif isinstance(url, str) and url.startswith(("http://", "https://")):
            transport_kind = "http"
            client = await connect_http(url, bearer_token=bearer)
        else:
            return (
                False,
                "",
                {
                    "state": "error",
                    "error": (
                        "ACP agent row needs `command` (list[str]) for "
                        "stdio, or `url` (ws://, wss://, http://, https://) "
                        "for network transports"
                    ),
                },
            )
    except Exception as e:  # noqa: BLE001
        return (
            False,
            "",
            {
                "state": "error",
                "error": f"ACP transport setup failed: {type(e).__name__}: {e}",
            },
        )

    try:
        try:
            await client.initialize()
        except Exception as e:  # noqa: BLE001
            return (
                False,
                "",
                {
                    "state": "error",
                    "error": f"initialize failed: {type(e).__name__}: {e}",
                },
            )

        try:
            session_id = await client.session_new(cwd or safe_getcwd())
        except Exception as e:  # noqa: BLE001
            return (
                False,
                "",
                {
                    "state": "error",
                    "error": f"session/new failed: {type(e).__name__}: {e}",
                },
            )

        # `mode` is sharedagents's permission concept (plan/ask/auto). The
        # agent may or may not advertise a matching session mode; we
        # try to set it but tolerate failure (capability gate or
        # unknown id) — the prompt still goes through.
        try:
            await client.session_set_mode(session_id, mode)
        except Exception:  # noqa: BLE001
            pass

        text_block = _acp_schema.TextContentBlock(text=prompt, type="text")
        prompt_blocks = [text_block.model_dump(by_alias=True)]
        stop_reason: str | None = None
        try:
            async for item in client.session_prompt(session_id, prompt_blocks):
                if isinstance(item, _acp_schema.SessionNotification):
                    update = item.update
                    chunk = ""
                    # AgentMessageChunk / AgentThoughtChunk both carry
                    # a `content` ContentBlock; surface text only —
                    # images/audio/resources are ignored for the
                    # assembled text body but still streamed by
                    # transport.
                    content = getattr(update, "content", None)
                    if content is not None and getattr(content, "type", None) == "text":
                        chunk = getattr(content, "text", "") or ""
                    if chunk:
                        text_buf.append(chunk)
                        if on_delta is not None:
                            try:
                                on_delta(chunk)
                            except Exception:  # noqa: BLE001
                                pass
                elif isinstance(item, _acp_schema.PromptResponse):
                    stop_reason = item.stop_reason
        except Exception as e:  # noqa: BLE001
            meta["state"] = "error"
            meta["error"] = f"session/prompt failed: {type(e).__name__}: {e}"
            return False, "".join(text_buf), meta

        # Map StopReason → legacy meta.state for the [acp …] header.
        # `end_turn` is the success path; everything else surfaces as
        # the state string and is treated as not-ok at the caller.
        if stop_reason == "end_turn":
            meta["state"] = "done"
            meta["error"] = None
            ok = True
        else:
            meta["state"] = stop_reason or "incomplete"
            meta["error"] = None if stop_reason is None else f"stop_reason={stop_reason}"
            ok = False

        try:
            await client.session_close(session_id)
        except Exception:  # noqa: BLE001 — best-effort
            pass
        return ok, "".join(text_buf), meta
    finally:
        # Orderly client teardown. The transport's close path is the
        # one wired into the killability story (subprocess stdin EOF /
        # WS close frame / HTTP DELETE).
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass
        # Each transport family has its own auxiliary handle:
        # - stdio: asyncio.subprocess.Process — wait briefly for clean
        #   exit, then SIGKILL if it's still running. Mirrors the
        #   killability rule without leaving zombies.
        # - ws: aiohttp.ClientSession — must close for connector cleanup.
        # - http: no auxiliary; client.close() does it all.
        if transport_kind == "stdio" and aux is not None:
            proc = aux
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        elif transport_kind == "ws" and aux is not None:
            try:
                await aux.close()
            except Exception:  # noqa: BLE001
                pass


async def run_acp_call_async(
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
) -> ToolResult:
    agent_name = args.get("agent_name")
    prompt = args.get("prompt")
    mode = args.get("mode") or os.environ.get("SHAREDAGENTS_PERMISSION_MODE", "auto")
    cwd = args.get("cwd")
    if not isinstance(agent_name, str) or not agent_name:
        return _err("agent_name required")
    if not isinstance(prompt, str) or not prompt.strip():
        return _err("prompt required")
    if mode not in ("plan", "ask", "auto"):
        mode = "auto"
    # An ACP agent brings its own tools and skills and runs outside the folder
    # sandbox: a Web user's session may call it only with the agent granted
    # and All tools, skills and folders, exactly like
    # delegating to an external engine. The remote picks its own model, which
    # sharedagents cannot see, so it also needs All models.
    _sessions_dir = os.environ.get("SHAREDAGENTS_SESSIONS_DIR", "")
    _sid = os.environ.get("SHAREDAGENTS_SESSION_ID", "")
    if _sessions_dir and _sid:
        from sharedagents_core.access_grants import delegation_refusal
        from sharedagents_session.access_binding import session_access_grants

        _grants = session_access_grants(_sessions_dir, _sid)
        _refusal = delegation_refusal(_grants, agent_name, external=True)
        if _refusal:
            return _err(_refusal)
        if _grants is not None and not _grants.is_all("models"):
            return _err(
                f"[access] ACP agent {agent_name!r} runs a model sharedagents cannot check; it needs All models granted"
            )
    # Resolve through the ``runners`` re-export so a test patching
    # ``runners._load_acp_agents_cfg`` is honoured after the de-monolith split.
    import sharedagents_agent.tools.runners as _runners

    _load_acp_agents_cfg = getattr(_runners, "_load_acp_agents_cfg", globals()["_load_acp_agents_cfg"])
    agents = _load_acp_agents_cfg()
    cfg = agents.get(agent_name)
    if cfg is None:
        configured = ", ".join(sorted(agents)) or "(none configured)"
        return _err(
            f"unknown ACP agent: {agent_name!r}. "
            f"Configured: {configured}. Edit `acp_agents` in "
            "~/.sharedagents/config.json to add one."
        )
    ok, body, meta = await _drive_acp_session_async(
        cfg,
        prompt,
        mode=mode,
        cwd=str(cwd) if isinstance(cwd, str) and cwd else None,
        on_delta=on_delta,
    )
    if not body:
        body = "(remote returned empty body)"
    header_bits = [f"agent={agent_name}", f"mode={mode}"]
    if meta.get("state"):
        header_bits.append(f"state={meta['state']}")
    if meta.get("error"):
        header_bits.append("error")
    header = "[acp " + " · ".join(header_bits) + "]"
    full = f"{header}\n\n{body}"
    if meta.get("error") and not ok:
        full += f"\n\nerror: {meta['error']}"
    full, truncated = _truncate(full)
    return ToolResult(content=full, ok=ok, truncated=truncated)


def _lsp_root(args: dict[str, Any]) -> str:
    # Tool runners inherit the orchestrator's process cwd (normally
    # ~/.sharedagents), while the session's actual repository is carried in this
    # trusted env field.  Keep the legacy explicit root override for internal
    # callers, then use the session workspace before falling back to cwd.
    root = args.get("root") or os.environ.get("SHAREDAGENTS_WORKSPACE_CWD")
    return str(Path(root).expanduser()) if root else safe_getcwd()


# opencode operation -> (lsp_client function name, arg-shape). One
# enum-dispatched ``lsp`` tool instead of one-per-op, to keep the per-round
# wire schema under budget (see registry.py's note). Op
# vocabulary + param names mirror opencode's ``lsp`` tool (tool/lsp.ts);
# sharedagents's own lsp_client.py backs each op. Arg-shapes:
#   "pos"   — needs filePath + line + character (1-based)
#   "file"  — needs only filePath (documentSymbol)
#   "query" — uses the ``query`` string; filePath selects the server
_LSP_OPS: dict[str, tuple[str, str]] = {
    "goToDefinition": ("lsp_definition", "pos"),
    "findReferences": ("lsp_references", "pos"),
    "hover": ("lsp_hover", "pos"),
    "documentSymbol": ("lsp_symbols", "file"),
    "workspaceSymbol": ("lsp_workspace_symbols", "query"),
    "goToImplementation": ("lsp_implementation", "pos"),
    "prepareCallHierarchy": ("lsp_prepare_call_hierarchy", "pos"),
    "incomingCalls": ("lsp_incoming_calls", "pos"),
    "outgoingCalls": ("lsp_outgoing_calls", "pos"),
}


def _run_lsp(args: dict[str, Any], fn_name: str, shape: str) -> ToolResult:
    """Shared body for the read-only LSP tools. Degrades gracefully (ok=False
    with a message) when no language server is installed — never a hard error.

    ``shape`` is one of ``pos`` / ``file`` / ``query`` (see ``_LSP_OPS``).
    ``filePath`` is the opencode param name; ``file``/``path`` are accepted as
    legacy aliases.
    """
    from sharedagents_agent import lsp_client as _lsp

    file = args.get("filePath") or args.get("file") or args.get("path")
    if not file:
        return _err("filePath required")
    fpath = Path(file).expanduser()
    if not fpath.is_absolute():
        fpath = Path(_lsp_root(args)) / fpath
    if not fpath.exists():
        return _err(f"file not found: {file}")
    fn = getattr(_lsp, fn_name)
    try:
        if shape == "pos":
            line = args.get("line")
            character = args.get("character") or args.get("col")
            if line is None or character is None:
                return _err("line and character (1-based) required")
            out = fn(_lsp_root(args), str(fpath), int(line), int(character))
        elif shape == "query":
            # workspaceSymbol: filePath selects the server; ``query`` (empty
            # string requests all symbols) is the actual request argument.
            out = fn(_lsp_root(args), str(fpath), str(args.get("query") or ""))
        else:  # "file"
            out = fn(_lsp_root(args), str(fpath))
    except _lsp.LspUnavailable as e:
        # The requested operation was NOT performed — surface as a handled
        # failure (ok=False) so the TUI / tool-failure supervision sees it, with
        # an actionable message.
        return ToolResult(content=f"LSP unavailable: {e}", ok=False)
    except Exception as e:  # noqa: BLE001
        return _err(f"lsp {fn_name}: {e}")
    body, truncated = _truncate(out)
    return ToolResult(content=body, ok=True, truncated=truncated)


def run_lsp(args: dict[str, Any]) -> ToolResult:
    raw = args.get("operation")
    # Tool args are model-generated and NOT schema-validated at dispatch, so a
    # non-string here (e.g. 3) must not reach .strip() and crash the runner.
    op = raw.strip() if isinstance(raw, str) else ""
    entry = _LSP_OPS.get(op)
    if entry is None:
        return _err(f"operation must be one of {', '.join(sorted(_LSP_OPS))} (got {raw!r})")
    fn_name, shape = entry
    return _run_lsp(args, fn_name, shape)


__all__ = [
    "run_mcp_list_tools_async",
    "run_mcp_call_async",
    "_load_acp_agents_cfg",
    "_drive_acp_session_async",
    "run_acp_call_async",
    "_lsp_root",
    "_LSP_OPS",
    "_run_lsp",
    "run_lsp",
]
