"""Stdio ACP entry point for ``sharedagents-agent``.

Wires :class:`sharedagents_gateway.acp_bridge.CoreBackend` to
:func:`sharedagents_acp.server.serve_stdio`. Invoked by the CLI's
``sharedagents_gateway acp-stdio`` subcommand and by ACP clients that spawn
``python -m sharedagents_gateway.acp_stdio_main`` as a child process.

The advertised :class:`AgentCapabilities`:

* ``loadSession: true`` — sessions persist to
  ``cfg.sessions_dir/<hash>.json`` so reattach is real.
* ``sessionCapabilities = {list: {}, close: {}}`` — list + close are
  supported; resume is intentionally absent.
* ``mcpCapabilities = {http: false, sse: false}`` — only stdio MCP is
  supported. The dispatcher rejects http/sse entries at the wire
  boundary so the bridge never sees them.
* ``promptCapabilities = {image: true, audio: false,
  embeddedContext: false}`` — text, resource_link, and image content
  blocks (the bridge fans image blocks out to the OpenAI multi-part
  ``image_url`` payload for vision-capable upstreams).
"""

from __future__ import annotations

import sys
from typing import Any

from sharedagents_acp.server import serve_stdio
from sharedagents_agent.agent import AgentConfig

# sharedagents's mode set today is just the permission-mode triple. The
# dispatcher passes the full block straight through to the wire.
_DEFAULT_MODES: list[dict[str, Any]] = [
    {"id": "auto", "description": "Auto-approve safe tools; prompt on destructive actions."},
    {"id": "ask", "description": "Prompt for confirmation on every tool call."},
    {"id": "plan", "description": "Plan-only mode — propose tool calls without executing."},
]


def _build_capabilities(cfg: AgentConfig) -> dict[str, Any]:
    """Build the static :class:`AgentCapabilities` payload.

    All fields here are session-fixed; including them in the dispatcher
    constructor means they sit in the prefix of every ``initialize``
    response and stay byte-identical across requests — keeps the
    upstream prefix cache happy.
    """
    # promptCapabilities.image MUST match what the server actually
    # accepts. Image content blocks are now translated to the OpenAI
    # multi-part ``image_url`` payload by ``_blocks_to_user_content``
    # in ``acp_bridge`` — vision-capable upstreams (vLLM / openrouter
    # / kimi) consume them natively, non-vision models ignore the
    # ``image_url`` parts. The dispatcher's
    # ``_SUPPORTED_CONTENT_BLOCK_TYPES`` includes ``"image"`` to match
    # this advertisement. Audio + embeddedContext stay false until
    # those paths are wired.
    return {
        "loadSession": True,
        "promptCapabilities": {
            "image": True,
            "audio": False,
            "embeddedContext": False,
        },
        "mcpCapabilities": {
            "http": False,
            "sse": False,
        },
        "sessionCapabilities": {
            "list": {},
            "close": {},
        },
    }


def _build_agent_info(cfg: AgentConfig) -> dict[str, Any]:
    return {"name": f"sharedagents-{cfg.name}", "version": "2.0.0"}


async def serve_acp_stdio(config: AgentConfig) -> int:
    """Serve ACP over stdio until EOF; return a process exit code.

    Constructs an :class:`AgentACPBackend` bound to ``config`` —
    the same unified backend the gateway mounts in-process — and
    hands it to :func:`sharedagents_acp.server.serve_stdio`. The backend's
    engine kind decides whether prompt/cancel route to the in-process
    CoreBackend wrapper (local_llm) or directly through Agent.run_turn
    → CLI engine (claude_cli / codex_cli / gemini_cli).

    Blocks until the read loop exits (peer closed stdin / EOF).
    Returns 0 on clean shutdown, 1 on unrecoverable transport error.
    """
    from sharedagents_agent.agents.factory import load_agent
    from sharedagents_agent.agents.gateway_shim import AgentACPBackend

    agent = load_agent(config.name)
    backend = AgentACPBackend(agent)

    async def _wire_server_handle(server: Any) -> None:
        # Hand the AcpServer back to the backend so the Decision.ASK
        # branch can issue outbound ``session/request_permission``
        # requests over the wire. For local-LLM agents this lands on
        # the inner CoreBackend; for CLI engines it's stored but
        # currently unused (CLI permission flag handles ASK).
        backend.server_handle = server

    try:
        await serve_stdio(
            backend,
            agent_capabilities=_build_capabilities(config),
            agent_info=_build_agent_info(config),
            auth_methods=[],
            on_server_ready=_wire_server_handle,
        )
    except Exception as exc:  # noqa: BLE001
        # Print to stderr so the parent can correlate; never to stdout
        # (that's the wire).
        print(f"sharedagents_gateway.acp_stdio: fatal: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


__all__ = [
    "serve_acp_stdio",
    "_build_capabilities",
    "_build_agent_info",
]
