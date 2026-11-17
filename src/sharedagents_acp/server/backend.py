"""Backend Protocol for ``sharedagents_acp.server``.

The dispatcher in :mod:`sharedagents_acp.server.server` is *transport-agnostic*
and *backend-agnostic*. It owns:

* JSON-RPC framing / parsing (delegated to ``sharedagents_acp.protocol``).
* Method-name → handler routing.
* Spec-level validation that lives at the wire boundary
  (ContentBlock variant gating, MCP transport gating, batch rejection).

It does NOT own:

* Where sessions live (in-memory dict? sqlite? sharedagents_core's tracker?).
* How prompts run (subprocess pool? thread? remote LLM?).
* What ``stopReason`` to return — the backend decides.

That separation is what this Protocol enforces. A production deployment
wires the dispatcher to ``sharedagents_core``'s session/turn machinery via an
adapter that satisfies :class:`AgentBackend`. Tests wire it to a fake.

Killability rule (CLAUDE.md):
    Every operation is instantly killable with ESC.

The Protocol is shaped to make that achievable without the dispatcher
having to know anything about subprocesses:

* :meth:`AgentBackend.prompt` returns an ``AsyncIterator`` of session
  updates. The dispatcher streams them out as ``session/update``
  notifications until the iterator yields its terminal sentinel
  (a ``PromptResponse``-shaped dict containing ``stopReason``), at which
  point the dispatcher writes the JSON-RPC response and returns control
  to the read-loop. The backend may yield ``cancelled`` as the
  ``stopReason`` if a concurrent :meth:`cancel` arrives — it MUST do
  exactly that per spec, and the dispatcher relays it byte-faithfully.

* :meth:`AgentBackend.cancel` is fire-and-forget (no response by spec —
  ``session/cancel`` is a notification). The implementation is expected
  to SIGTERM the worker subprocess that's currently running the prompt
  for the given ``session_id``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol


class AgentBackend(Protocol):
    """Backend interface the dispatcher calls into for every agent-side ACP method.

    All methods are async except :meth:`prompt` (which is a *factory*
    returning an async iterator — the dispatcher iterates it, the
    backend produces values).

    Parameters and return values are intentionally typed as raw
    JSON-shaped Python objects (``dict[str, Any]`` / ``str`` / ``None``)
    rather than SDK Pydantic models. The dispatcher converts wire JSON
    ↔ Python at its boundary; the backend stays SDK-version-independent
    so swapping ``acp`` SDK versions is a one-file change in the
    dispatcher and not a breaking change to backend implementations.

    Implementations MUST be safe to call concurrently across distinct
    ``session_id`` values. Calls scoped to the same ``session_id`` are
    serialized by the dispatcher, with one exception: :meth:`cancel`
    can fire concurrently with an in-flight :meth:`prompt`, and the
    backend is responsible for unblocking the prompt iterator by
    yielding its terminal sentinel with ``stopReason = "cancelled"``.
    """

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Mapping[str, Any] | None,
        client_info: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        """Notify the backend an ``initialize`` round-trip is happening.

        The dispatcher already advertises agent capabilities / agent
        info / auth methods (configured at construction); this hook
        exists so backends can record the negotiated peer version /
        capabilities for their own use. Return value is ignored — the
        InitializeResponse is built from the dispatcher's static config.
        """
        ...

    async def authenticate(self, method_id: str) -> None:
        """Run an authentication method. Default stance: no auth methods
        advertised, so ``authenticate`` is rejected at the dispatcher
        with ``-32601`` before this hook fires. Implementations that
        DO advertise auth methods override the dispatcher's handler."""
        ...

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Create a new session. Returns a JSON-shaped dict matching
        ``NewSessionResponse``: ``{"sessionId": ..., "modes": ...,
        "models": ..., "configOptions": [...]}``. The dispatcher
        validates the response against the SDK schema before sending."""
        ...

    async def load_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Reattach to an existing session. Returns a JSON-shaped dict
        matching ``LoadSessionResponse``."""
        ...

    async def list_sessions(
        self,
        cursor: str | None,
        cwd: str | None,
    ) -> Mapping[str, Any]:
        """List sessions (cursor-paginated). Returns a JSON-shaped
        dict matching ``ListSessionsResponse``: ``{"sessions": [...],
        "nextCursor": ...}``."""
        ...

    async def close_session(self, session_id: str) -> None:
        """Close a session and free its resources."""
        ...

    async def set_mode(self, session_id: str, mode_id: str) -> None:
        """Switch the active mode for a session."""
        ...

    async def set_config_option(
        self,
        session_id: str,
        config_id: str,
        value: Any,
    ) -> None:
        """Update a session-scoped config option."""
        ...

    async def inject_user_message(
        self,
        session_id: str,
        text: str,
    ) -> Mapping[str, Any]:
        """Push a mid-turn user message to the engine's inject buffer.

        Optional capability. Backends that support it return
        ``{"ok": True, "queued": True}`` once queued. The engine drains
        the buffer at every round boundary so a steering prompt issued
        while a multi-round plan is in flight lands at the next
        worker spawn — no waiting for the turn to terminate.

        Default raises ``NotImplementedError`` so the dispatcher
        surfaces ``METHOD_NOT_FOUND`` for ABC-bare backends.
        """
        raise NotImplementedError("session/inject_user_message not supported")

    async def run_command(
        self,
        session_id: str,
        text: str,
        *,
        frontend: str = "api",
        confirmed: bool = False,
    ) -> Mapping[str, Any]:
        """Execute a slash command via the shared server-side runtime.

        Optional capability. Returns a JSON-shaped ``CommandResponse`` dict
        (see :mod:`sharedagents_command.command_runtime`). This is the SINGLE
        command implementation every frontend renders — frontends carry
        zero command logic. Default raises ``NotImplementedError`` so the
        dispatcher surfaces ``METHOD_NOT_FOUND`` for ABC-bare backends.
        """
        raise NotImplementedError("session/command not supported")

    def prompt(
        self,
        session_id: str,
        message_id: str | None,
        prompt: list[Mapping[str, Any]],
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Run one prompt turn. Yields a stream of JSON-shaped dicts:

        * Intermediate values: ``SessionUpdate``-shaped dicts the
          dispatcher relays as ``session/update`` notifications.
          Each dict MUST be ready to drop into a ``SessionNotification``
          envelope's ``update`` field (i.e. the ``sessionUpdate``
          discriminator is set, etc.).

        * Terminal value: a ``PromptResponse``-shaped dict containing
          ``stopReason``. The dispatcher reads the iterator until it
          gets a value that is NOT a ``SessionUpdate`` (no
          ``sessionUpdate`` key), treats that as the terminal
          ``PromptResponse``, and writes it as the JSON-RPC response.

        Cancellation: when a concurrent :meth:`cancel` fires for the
        same session, the iterator MUST eventually yield a terminal
        value with ``"stopReason": "cancelled"``. Per spec, this is
        the only way to confirm a cancellation reached the agent.
        """
        ...

    async def cancel(self, session_id: str, *, force: bool = False) -> None:
        """Cancel an in-flight prompt for ``session_id``. Returns
        immediately; the actual confirmation flows back via the
        :meth:`prompt` iterator yielding ``stopReason = "cancelled"``.

        ``force=False`` is a graceful stop (let the in-flight tool /
        round finish, then end the turn); ``force=True`` is a hard kill
        (SIGKILL + killpg the worker and every runner / subagent now)."""
        ...


__all__ = ["AgentBackend"]
