"""Minimal JSON-RPC 2.0 / stdio scaffold shared by sharedagents's in-repo MCP
servers (``memory``, ``web``, ``delegate``).

The first in-repo server (``youtube_mcp_server``) hand-rolled the whole
protocol loop inline (~1k lines). These newer servers share this scaffold so
each one is just its ``TOOL_DEFS`` + ``TOOL_DISPATCH`` — the transport,
framing, and the ``initialize``/``tools/list``/``tools/call``/``ping`` methods
live here once.

Protocol: MCP ``2024-11-05`` over line-delimited JSON on stdin/stdout, matching
what ``sharedagents_mcp.mcp_client`` speaks. A tool handler is
``fn(arguments: dict) -> str`` and MAY raise; a raise becomes an
``isError: true`` result (never crashes the loop). A handler wrapping a runner
that returns a ``ToolResult`` should pass it through ``result_text`` so an
``ok=False`` failure ALSO becomes ``isError`` (not a success carrying an error
string). Declare a tool read-only by
putting ``"annotations": {"readOnlyHint": true}`` in its ``TOOL_DEFS`` entry —
``sharedagents_agent.tools.registry.discovered_mcp_tools`` reads that to pick READ
criticality instead of the NETWORK/mutating default.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

PROTOCOL_VERSION = "2024-11-05"

ToolHandler = Callable[[dict], str] | Callable[[dict], Awaitable[str]]


class ToolError(RuntimeError):
    """A handler raises this to return an MCP ``isError`` result with a clean
    message (no exception-type prefix). ``result_text`` raises it for a runner
    ``ToolResult`` whose ``ok`` is False."""


def result_text(res: Any) -> str:
    """Unwrap a runner ``ToolResult`` for an MCP handler. Returns ``.content``,
    but RAISES :class:`ToolError` when ``.ok`` is False so a runner FAILURE
    (missing Brave key, refused / path-traversal-blocked memory read, missing
    cancel ``job_id``, …) surfaces as ``isError: true`` — NOT as a successful
    call carrying an error string the model would trust. A plain value with no
    ``ok`` attribute is treated as success.
    """
    ok = getattr(res, "ok", True)
    content = getattr(res, "content", res)
    if not ok:
        raise ToolError(str(content))
    return str(content)


def _frame(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _ok(mid: Any, result: dict) -> None:
    _frame({"jsonrpc": "2.0", "id": mid, "result": result})


def _error(mid: Any, code: int, message: str) -> None:
    _frame({"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}})


def run_stdio_server(
    server_name: str,
    server_version: str,
    tool_defs: Sequence[Mapping[str, Any]],
    tool_dispatch: Mapping[str, ToolHandler],
) -> None:
    """Serve ``tool_dispatch`` over stdio until stdin closes (fail-safe).

    ``tool_defs`` is the list returned by ``tools/list`` (each a
    ``{name, description, inputSchema, annotations?}`` dict); ``tool_dispatch``
    maps tool name → handler. Names in the two MUST agree.
    """
    tool_defs = list(tool_defs)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(req, dict):
            continue
        mid = req.get("id")
        method = req.get("method")
        params = req.get("params") if isinstance(req.get("params"), dict) else {}
        # A notification (no id) never gets a response.
        is_notification = mid is None
        if method == "initialize":
            if not is_notification:
                _ok(
                    mid,
                    {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": server_name, "version": server_version},
                    },
                )
        elif method == "tools/list":
            if not is_notification:
                _ok(mid, {"tools": tool_defs})
        elif method == "tools/call":
            if is_notification:
                continue
            name = str(params.get("name") or "")
            args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            fn = tool_dispatch.get(name)
            if fn is None:
                _error(mid, -32601, f"unknown tool: {name!r}")
                continue
            try:
                result = fn(args)
                if isinstance(result, Awaitable):
                    import asyncio

                    result = asyncio.run(result)
                text = result
                is_error = False
            except ToolError as exc:
                # Handler-signalled failure (ok=False runner result) — clean
                # message, no exception-type prefix.
                text = str(exc)
                is_error = True
            except Exception as exc:  # noqa: BLE001 — a bad call must not kill the loop
                text = f"{type(exc).__name__}: {exc}"
                is_error = True
            _ok(
                mid,
                {
                    "content": [{"type": "text", "text": str(text)}],
                    "isError": is_error,
                },
            )
        elif method == "ping":
            if not is_notification:
                _ok(mid, {})
        elif method in ("notifications/initialized", "initialized"):
            continue  # client handshake notification — no reply
        else:
            if not is_notification:
                _error(mid, -32601, f"unknown method: {method!r}")
