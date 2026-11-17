"""Async client for talking to a running sharedagents agent.

Adapts an ``AgentClient``-shaped surface (used throughout the TUI) onto
:mod:`sharedagents_acp.client`. The production transport is Streamable HTTP
against the sharedagents north gateway (``sharedagents gateway serve``)
— set ``$SHAREDAGENTS_TUI_ACP_URL=http://127.0.0.1:<port>`` and the TUI
opens one HTTP connection per agent under ``/acp/<name>``. Stdio
spawn is preserved as a unit-test fallback (no env var set, no
``gateway_url`` arg) so tests don't need to run the gateway.

The streaming method translates ACP ``session/update`` notifications
back into the (event, data, turn_id) tuple shape the TUI already
consumes.

API surface:
    create_session, get_session, list_sessions, delete_session,
    session_turn, session_command, update_session_model,
    get_session_state, list_agent_modes, switch_agent,
    submit_permission, status, close.

Intentionally lossy / dropped:
    * ``turn_pause`` / ``turn_resume`` — ACP has only ``session/cancel``.
      These raise :class:`NotImplementedError` so any future caller
      gets a clear message instead of a silent no-op.
    * ``compact_session`` — emulated locally as a no-op with an
      ``ok=True`` ack so the ``/compact`` slash UI still works.
    * ``status`` / ``stop_server`` — ACP doesn't model agent uptime or
      daemon shutdown; we synthesize a minimal status dict and treat
      ``stop_server`` as a no-op (the gateway owns the child).
    * ``append_turn`` — ACP folds delegated turns into ordinary
      ``tool_call`` updates; the adapter raises
      :class:`NotImplementedError`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from sharedagents_acp.client.client import AcpRemoteError
from sharedagents_acp.protocol.jsonrpc import JsonRpcErrorCode
from sharedagents_agent.agent import AgentConfig
from sharedagents_agent.external_registry import EXTERNAL_REGISTRY

logger = logging.getLogger(__name__)


class GatewayControlError(RuntimeError):
    """Typed control-surface refusal used for one safe stale retry."""

    def __init__(
        self,
        code: str,
        message: str,
        status: int,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.status = status
        self.data = data or {}


def _model_label(alias: object = "", model_id: object = "") -> str:
    """Render ``alias/model`` unless alias and model are identical."""
    alias_s = str(alias or "")
    model_s = str(model_id or "")
    if alias_s and model_s and alias_s != model_s:
        return f"{alias_s}/{model_s}"
    return alias_s or model_s


# Engine kinds that drive turns through an external CLI rather than an
# HTTP upstream. For these agents the cfg-level ``upstream_endpoint`` /
# ``upstream_model`` (inherited from the global config) are meaningless
# and must not be reported as the session's identity — see
# Agent.create_session in sharedagents_core/agents/agent.py.
_CLI_ENGINE_KINDS = frozenset(spec.kind.value for spec in EXTERNAL_REGISTRY.values() if spec.adapter == "cli")


def _cli_engine_label(engine: str) -> str:
    """Return the bare CLI name (``claude_cli`` → ``claude``).

    Empty for non-CLI engines so callers can use it as a truthy gate.
    """
    e = (engine or "").strip()
    if e not in _CLI_ENGINE_KINDS:
        return ""
    return e[:-4] if e.endswith("_cli") else e


def _read_engine_actual_model(cfg: Any, sid: str, engine: str) -> str:
    """Return the real model id a CLI engine detected at runtime.

    The CLI engines (claude / codex / antigravity) persist the model the
    CLI reported on its ``init`` event onto a per-session sidecar at
    ``<sessions_dir>/<sid>.<engine>.json`` (see
    ``ExternalCliEngine.write_state``). Returns ``""`` when the sidecar
    is missing, unreadable, or hasn't captured a model yet.
    """
    import json

    try:
        path = cfg.sessions_dir / f"{sid}.{engine}.json"
        if not path.exists():
            return ""
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("actual_model") or "")


def _read_session_view(cfg: Any, sid: str) -> dict | None:
    """Read a session's header (sessions_meta) + folded turns from the
    store, returning a dict shaped like the legacy ``<sid>.json``
    (keys: hash/title/created_at/system_prompt/model/endpoint/backend/
    context_window/workspace_*/engine/system_mode/pinned_facts/turns/…).

    Returns ``None`` when the session is unknown. Single source for all
    of agent_client's disk-side session reads (footer model lookup,
    get_session, /context).
    """
    from sharedagents_session.session_index import SessionIndex, connect_sessions_db
    from sharedagents_session.session_reader import fold_jsonl_into_turns

    db = cfg.sessions_dir / "sessions.db"
    jsonl = cfg.sessions_dir / f"{sid}.jsonl"
    if not jsonl.exists() and not db.exists():
        return None
    try:
        SessionIndex(cfg.sessions_dir).tail(sid)
        con = connect_sessions_db(db)
        try:
            row = con.execute("SELECT * FROM sessions_meta WHERE session_hash=?", (sid,)).fetchone()
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return None
    if row is None:
        return None
    d = dict(row)
    # Cross-agent guard: the unified sessions.db means a known sid
    # for ``claude`` would otherwise be readable from this client's
    # ``main`` context (TUI /history, /context, get_session). Reject
    # when the session's agent_name disagrees with the configured
    # client. Empty agent_name keeps legacy / unattributed sessions
    # visible.
    sess_agent = str(d.get("agent_name") or "")
    own_agent = str(getattr(cfg, "name", "") or "")
    if sess_agent and own_agent and sess_agent != own_agent:
        return None
    d["hash"] = d.get("session_hash")
    pf = d.get("pinned_facts")
    if isinstance(pf, str):
        try:
            import json as _json

            d["pinned_facts"] = _json.loads(pf)
        except (ValueError, TypeError):
            d["pinned_facts"] = []
    try:
        d["turns"] = fold_jsonl_into_turns(cfg.sessions_dir / f"{sid}.jsonl")
    except Exception:  # noqa: BLE001
        d["turns"] = []
    return d


# ---------------------------------------------------------------------------
# Spawn hook (test seam) and capability helpers
# ---------------------------------------------------------------------------

# Importing the heavy ACP client lazily lets the rest of this module
# load even when the optional dependency stack is incomplete. The
# for_agent / spawn paths are the only ones that need the import.


async def _default_spawn_stdio_agent(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    client_capabilities: Any = None,
    client_info: Any = None,
):
    """Default impl of the spawn hook — re-exported to ease test patching.

    Tests monkey-patch :data:`spawn_hook` (or pass an injected hook to
    :meth:`AgentClient.__init__`) instead of patching the upstream
    ``sharedagents_acp.client.spawn_stdio_agent`` symbol — that way one swap
    covers both ``for_agent`` and direct construction.
    """
    from sharedagents_acp.client import spawn_stdio_agent

    return await spawn_stdio_agent(
        cmd,
        cwd=cwd,
        env=env,
        client_capabilities=client_capabilities,
        client_info=client_info,
    )


# Module-level hook; tests reassign with monkeypatch.setattr on this name.
SpawnHook = Callable[..., Awaitable[Any]]
spawn_hook: SpawnHook = _default_spawn_stdio_agent


# Env var TUI consults for the gateway URL. Empty / unset → stdio
# fallback (test mode). Production deployments set this to e.g.
# ``http://127.0.0.1:8700`` and the TUI HTTP-connects per agent.
TUI_GATEWAY_URL_ENV = "SHAREDAGENTS_TUI_ACP_URL"
TUI_GATEWAY_TOKEN_ENV = "SHAREDAGENTS_TUI_ACP_TOKEN"


# ---------------------------------------------------------------------------
# AgentClient
# ---------------------------------------------------------------------------


class AgentClient:
    """Adapter exposing an ``AgentClient`` surface on top of ACP.

    Construction is sync to preserve the existing call-site shape
    ``AgentClient(cfg.url)``. The first ``await``-method on the
    instance lazily opens a connection: HTTP-against-the-gateway
    when ``gateway_url`` (or ``$SHAREDAGENTS_TUI_ACP_URL``) is set, stdio
    spawn otherwise. ``base_url`` is the legacy per-agent daemon URL,
    kept in the constructor signature so existing call sites still
    compile — the production transport runs through ``gateway_url``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        agent_name: str = "jarvis",
        cmd: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        spawn: SpawnHook | None = None,
        gateway_url: str | None = None,
        gateway_token: str | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self._agent_name = agent_name
        self._cmd: list[str] = (
            list(cmd)
            if cmd is not None
            else [
                sys.executable,
                "-m",
                "sharedagents_gateway",
                "acp-stdio",
                agent_name,
            ]
        )
        # Capture cwd at construction time. The ACP child's spawn cwd
        # is the launching frontend's cwd, so relative paths the
        # model logs / proposes resolve from the user's perspective.
        # If callers passed an explicit cwd, honour it; otherwise snapshot
        # ``os.getcwd()`` NOW (so a later cwd change in the parent doesn't
        # silently shift the session's workspace mid-flight).
        self._cwd: str = str(cwd) if cwd else os.getcwd()
        self._env = dict(env) if env else None
        self._spawn: SpawnHook = spawn or spawn_hook
        # Gateway-mode wiring. ``gateway_url`` (or env override) being
        # set switches the lazy-connect path from stdio spawn to a
        # Streamable HTTP connection against the running sharedagents
        # gateway. Empty falls through to stdio (test fallback only).
        self._gateway_url: str = (
            gateway_url if gateway_url is not None else os.environ.get(TUI_GATEWAY_URL_ENV, "").strip()
        ).rstrip("/")
        self._gateway_token: str | None = (
            gateway_token if gateway_token is not None else (os.environ.get(TUI_GATEWAY_TOKEN_ENV, "").strip() or None)
        )
        # Connection state, populated lazily.
        self._acp: Any = None  # sharedagents_acp.client.AcpClient
        self._proc: Any = None  # asyncio.subprocess.Process | None
        self._init_response: Any = None
        self._init_lock = asyncio.Lock()
        # Tri-state capability flag for the server-side command runtime:
        # None = unprobed, True = gateway speaks ``session/command`` (the
        # ONLY path), False = gateway too old → use the deprecated legacy
        # local path. Set once on the first ``session_command`` call.
        self._supports_session_command: bool | None = None
        # Per-session turn-id mapping. ACP doesn't model turn ids; the
        # TUI treats turn_id as opaque so we mint local UUID4s and let
        # the stop/cancel path map them back to sessionId.
        self._turn_to_session: dict[str, str] = {}
        # Mirror of the most recent assistant message text per session,
        # so callers like get_session() can reconstruct the saved turn
        # shape app.py expects (turns: list[{user, content, ...}]).
        self._session_view: dict[str, dict[str, Any]] = {}
        # Per-session (model_label, context_window) cache populated on
        # first lookup — read off the session JSON directly; the wire
        # has no envelope for either today.
        self._model_lookup_cache: dict[str, tuple[str, int]] = {}
        # Frontend-registered hook for the AskUserQuestion round-trip.
        # The ACP inbound handler ``_inbound_ask_user`` blocks on this
        # async callback; the TUI installs one that pops a modal and
        # returns the user's selections. None → fall back to a synthetic
        # "first option of each question" answer so the model doesn't
        # hang in tests / headless contexts.
        self._ask_user_callback: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None
        # Frontend-registered hook for ``session/request_permission``.
        # The ACP inbound handler ``_inbound_request_permission`` blocks
        # on this async callback; the TUI installs one that pops the
        # 3-button permission modal and returns the user's choice.
        # None (or a raising callback) → DENY: a destructive op must
        # never run just because no UI was wired to ask. The callback
        # gets the JSON-RPC ``params`` and returns the ACP
        # ``{"outcome": {...}}`` mapping verbatim.
        self._permission_callback: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None
        # Server-authoritative state for the active session.
        self.active_agent: str = self._agent_name
        self.effective_mode: dict[str, Any] = {}
        self.available_agent_modes: list[dict[str, Any]] = []

    # -- session-model lookup ----------------------------------------------

    def _lookup_session_model(self, session_id: str) -> tuple[str, int]:
        """Best-effort ``(model_label, context_window)`` for the bound
        session, read off the persisted JSON.

        ``model_label`` resolves to the user-facing alias when one
        matches (``server1``, ``gpu-a``, …) so the TUI footer reads
        ``server1/qwen3-30b`` instead of a bare opaque id. Falls
        through to the raw model id when no alias matches, then to
        the empty string when the session JSON isn't reachable.
        """
        cached = self._model_lookup_cache.get(session_id)
        if cached is not None:
            return cached
        try:
            cfg = AgentConfig.load(self._agent_name)
            raw = _read_session_view(cfg, session_id)
            if raw is None:
                self._model_lookup_cache[session_id] = ("", 0)
                return "", 0
            # CLI-engine sessions store ``model`` = engine name and
            # ``endpoint`` empty (see Agent.create_session). Skip the
            # alias-by-endpoint dance — the model field is already the
            # operator-facing label ("claude" / "codex" / "antigravity").
            sess_engine = str(raw.get("engine") or "")
            if _cli_engine_label(sess_engine):
                # Prefer the runtime-detected model id the CLI reported on
                # its init event (persisted on the engine sidecar) over the
                # stored placeholder, which is just the engine name
                # ("claude"). The session ``model`` field for CLI engines
                # holds that placeholder — showing "claude:claude" is
                # meaningless, so fall back to the bare engine label only
                # when no real model has been detected yet.
                actual = _read_engine_actual_model(
                    cfg,
                    session_id,
                    sess_engine,
                )
                fallback = str(raw.get("model") or "")
                if fallback == _cli_engine_label(sess_engine):
                    fallback = ""
                out = (
                    actual or fallback or _cli_engine_label(sess_engine),
                    0,
                )
                self._model_lookup_cache[session_id] = out
                return out
            model_id = str(raw.get("model") or "")
            endpoint = str(raw.get("endpoint") or "")
            ctx_window = int(raw.get("context_window") or 0)
            alias = ""
            for m in getattr(cfg, "models", None) or []:
                if not isinstance(m, dict):
                    m = m.__dict__ if hasattr(m, "__dict__") else {}
                if (m.get("baseUrl") or "").rstrip("/") == endpoint.rstrip("/"):
                    alias = str(m.get("alias") or "")
                    if not ctx_window:
                        ctx_window = int(m.get("contextWindow") or 0)
                    break
            if not ctx_window:
                ctx_window = int(getattr(cfg, "upstream_max_context", 0) or 0)
            label = _model_label(alias, model_id)
            out = (label, ctx_window)
        except Exception:  # noqa: BLE001
            out = ("", 0)
        self._model_lookup_cache[session_id] = out
        return out

    # -- legacy classmethod -----------------------------------------------

    @classmethod
    def for_agent(cls, name: str = "jarvis") -> AgentClient:
        """Construct an AgentClient that will spawn agent ``name`` on
        first use.

        We still consult :class:`AgentConfig` so a misconfigured agent
        fails loudly at construction time (matches the legacy behaviour
        where the daemon URL was resolved from disk). The spawn itself
        is deferred — that's where the actual transport opens.
        """
        cfg = AgentConfig.load(name)
        return cls(cfg.url, agent_name=name)

    # -- lifecycle --------------------------------------------------------

    async def _ensure_connected(self) -> Any:
        """Open the ACP connection lazily, once.

        Production: HTTP-connect to the gateway at ``<gateway>/acp/<agent>``.
        Test fallback: spawn a stdio child of the TUI process.

        Returns the underlying :class:`AcpClient`. Subsequent calls are
        cheap — the lock makes the first concurrent caller win and the
        rest reuse the same client.
        """
        if self._acp is not None:
            return self._acp
        async with self._init_lock:
            if self._acp is not None:
                return self._acp
            if self._gateway_url:
                from sharedagents_acp.client.http_spawn import connect_http

                # Each agent is mounted at ``/acp/<agent>`` on the
                # gateway; ``connect_http`` then appends ``/acp`` for
                # the JSON-RPC route, matching gateway.py's mount
                # over a sub-app whose own routes live under ``/acp``.
                agent_url = f"{self._gateway_url}/acp/{self._agent_name}"
                self._acp = await connect_http(
                    agent_url,
                    bearer_token=self._gateway_token,
                    delete_on_close=False,
                    frontend="tui",  # Pin identity at init
                )
                self._proc = None
                # ``connect_http`` already negotiated initialize during
                # connect (the connection id is in the response
                # headers). Synthesize the post-initialize state into
                # ``_init_response`` for parity with the stdio path.
                self._init_response = self._acp
            else:
                spawn_result = await self._spawn(
                    self._cmd,
                    cwd=self._cwd,
                    env=self._env,
                    client_capabilities=None,
                    client_info=None,
                )
                # spawn_stdio_agent returns (client, process) by default;
                # tolerate hooks that return only the client too.
                if isinstance(spawn_result, tuple):
                    self._acp, self._proc = spawn_result
                else:
                    self._acp = spawn_result
                    self._proc = None
                try:
                    self._init_response = await self._acp.initialize()
                except Exception:
                    # Initialization failed: tear down so the next call
                    # doesn't reuse a broken half-open client.
                    await self._safe_close_acp()
                    raise
            # Wire an inbound ``session/request_permission`` handler so
            # the agent's outbound requests resolve cleanly. The
            # agent's bridge issues real outbound prompts and would
            # block on a missing handler. Default here is auto-allow
            # with a clear log line — a developer running the TUI
            # sees that the prompt fired and can replace this with a
            # real modal. Auto-allow is the right default at the TUI
            # layer (operator is in front of the keyboard, not
            # headless); the agent's safety still applies because the
            # evaluator only ASKs on destructive ops by policy.
            self._acp.register_request_permission(self._inbound_request_permission)
            # ``session/ask_user`` round-trips the AskUserQuestion tool
            # through the connected frontend (TUI modal). Without a
            # registered handler the agent's outbound request lands on
            # ``methodNotFound`` and the bridge falls back to an error
            # tool_result — wire it unconditionally; the actual UI hook
            # is registered later by the app via ``set_ask_user_callback``.
            self._acp.register_ask_user(self._inbound_ask_user)
            return self._acp

    async def _safe_close_acp(self) -> None:
        if self._acp is not None:
            try:
                await self._acp.close()
            except Exception:
                logger.debug("acp client close raised", exc_info=True)
        if self._proc is not None:
            # If the child is still alive after stdin EOF, give it a
            # short grace then terminate. We don't wait long; the TUI
            # is shutting down.
            try:
                if self._proc.returncode is None:
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        self._proc.terminate()
                        try:
                            await asyncio.wait_for(self._proc.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            self._proc.kill()
            except Exception:
                logger.debug("proc cleanup raised", exc_info=True)
        self._acp = None
        self._proc = None

    async def close(self) -> None:
        await self._safe_close_acp()

    # -- session agent switching -----------------------------------------

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def fold_session_state(self, value: Any) -> dict[str, Any]:
        """Fold one session response or notification into the local view."""
        state = self._as_mapping(value)
        agent = state.get("active_agent") or state.get("activeAgent")
        if isinstance(agent, str) and agent:
            self.active_agent = agent
        mode = state.get("effective_mode") or state.get("effectiveMode")
        if isinstance(mode, dict):
            self.effective_mode = dict(mode)
        elif isinstance(mode, str) or state.get("effective_mode_id"):
            mode_id = str(mode or state.get("effective_mode_id") or "")
            self.effective_mode = {
                "id": mode_id,
                "label": (state.get("effective_mode_label") or state.get("effectiveModeLabel") or mode_id),
                "behavior_kind": (state.get("behavior_kind") or state.get("behaviorKind") or ""),
            }
        modes = state.get("available_agent_modes") or state.get("availableAgentModes") or state.get("agents")
        if isinstance(modes, list):
            self.available_agent_modes = [dict(row) for row in modes if isinstance(row, dict)]
        return state

    async def _session_control_call(
        self,
        sid: str,
        endpoint: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._gateway_url:
            raise RuntimeError("feature_disabled: session control requires gateway")
        import httpx

        url = self._gateway_url + f"/control/v1/orchestrator/sessions/{sid}/{endpoint}"
        async with httpx.AsyncClient(
            timeout=30.0,
            headers=self._gateway_headers(),
        ) as http:
            response = await http.post(url, json=body or {}) if body is not None else await http.get(url)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code >= 400:
            code = str(payload.get("code") or response.status_code)
            message = str(payload.get("message") or response.text[:200])
            raise RuntimeError(f"{code}: {message}")
        return self._as_mapping(payload)

    async def get_session_state(self, sid: str) -> dict[str, Any]:
        result = await self._session_control_call(sid, "state")
        return self.fold_session_state(result)

    async def list_agent_modes(
        self,
        sid: str,
    ) -> list[dict[str, Any]]:
        """Return capability-derived agents and honest effective modes."""
        result = await self._session_control_call(sid, "capabilities")
        self.fold_session_state(result)
        rows = result.get("agents") or result.get("available_agent_modes") or result.get("availableAgentModes") or []
        self.available_agent_modes = [dict(row) for row in rows if isinstance(row, dict)]
        return list(self.available_agent_modes)

    async def switch_agent(
        self,
        sid: str,
        target_agent: str,
        target_mode: str | None = None,
        *,
        request_id: str | None = None,
        frontend: str = "tui",
    ) -> dict[str, Any]:
        """Switch the agent serving one session without replacing its SID."""
        params: dict[str, Any] = {
            "target_agent": target_agent,
            "request_id": request_id or uuid.uuid4().hex,
            "frontend": frontend,
        }
        if target_mode:
            params["target_mode"] = target_mode
        result = await self._session_control_call(sid, "agent", body=params)
        return self.fold_session_state(result)

    def is_connected(self) -> bool:
        """True iff the ACP client is live.

        Cheap probe used by the TUI's reconnect watcher to decide
        whether to rebuild the client without an extra round-trip.

        Stdio mode: live iff the AcpClient exists, isn't closed, AND
        the child process is still running.
        Gateway/HTTP mode: ``_proc`` is always None (the gateway owns
        the child); liveness reduces to "AcpClient exists and isn't
        closed". A dead remote will surface as a transport error on
        the next call, which the watcher already handles.
        """
        if self._acp is None:
            return False
        if getattr(self._acp, "_closed", False):
            return False
        if self._proc is None:
            # Gateway/HTTP path — no local child to probe.
            return True
        return self._proc.returncode is None

    async def attach_session(
        self,
        sid: str,
        *,
        cwd: str | None = None,
    ) -> dict:
        """Reattach to an existing session via ACP ``session/load``.

        Used by the reconnect watcher: after a stdio child crash and
        re-spawn, the TUI's ``self._sid`` is still the operator's
        active session. Calling ``session/load`` on the fresh child
        registers it as the owner of that sid so subsequent
        ``session_prompt`` / ``session_command`` calls succeed
        without the user having to start over.

        Returns a ``LoadSessionResponse``-shaped dict so callers can
        merge new modes/configOptions into the local view.
        """
        client = await self._ensure_connected()
        target_cwd = cwd or self._cwd or os.getcwd()
        resp = await client.session_load(
            sid,
            cwd=target_cwd,
            mcp_servers=[],
        )
        # Pydantic model → dict for downstream consumers.
        try:
            return resp.model_dump(by_alias=True, exclude_none=True)  # type: ignore[union-attr]
        except AttributeError:
            return dict(resp)  # type: ignore[arg-type]

    # -- control ----------------------------------------------------------

    async def status(
        self,
        endpoint: str | None = None,
        model: str = "",
        backend: str = "",
        headers: dict | None = None,
    ) -> dict:
        """Synthesize a minimal status dict + probe the LLM upstream.

        Optional ``endpoint`` overrides the config's ``upstream_endpoint``
        — used by the TUI to probe the *session's* active upstream rather
        than the config default. When ``endpoint`` is provided, ``model``,
        ``backend``, and ``headers`` may be passed to skip the config
        lookup; if omitted they are derived from the config's model table.

        ACP has no agent-uptime concept; daemon health belongs to
        whatever process supervisor wraps the daemon. We emulate just
        enough structure for ``app.py``'s status-bar renderer to show
        the same durable footer fields as the HTTP-backed client.

        ``upstream.ok`` reflects the **real** reachability of the
        configured LLM endpoint at call time — but only for **local**
        backends (``vllm`` / ``sglang``). For cloud providers
        (OpenRouter, DeepSeek, Anthropic, …) we deliberately skip the
        probe: ``/v1/models`` is free of token cost but still costs ~6
        unauthenticated/keyed requests per minute against a paid API,
        which is wasteful and rate-limit-noisy when the operator
        already knows the cloud is up. Cloud rows surface
        ``ok=True, probed=False`` so the TUI can render a hint like
        ``upstream (cloud)`` instead of a misleading green ok. A real
        outage will surface as a turn-time error, which the bridge
        already streams back as ``upstream warn`` in the status bar.
        """
        await self._ensure_connected()
        try:
            cfg = AgentConfig.load(self._agent_name)
            cli_label = _cli_engine_label(getattr(cfg, "engine", "") or "")
            if cli_label:
                # CLI-engine agents have no HTTP upstream — the CLI
                # itself owns the model call. Surface a fixed
                # ``cloud``-style upstream block keyed to the engine
                # name so the footer reads "model claude" (not the
                # inherited ``gpu-a/qwen3.6-27b-fp8`` from the global
                # config, which doesn't apply here).
                return {
                    "ok": True,
                    "agent": self._agent_name,
                    "url": self.base_url,
                    "uptime_s": 0,
                    "pid": getattr(self._proc, "pid", None),
                    "heartbeat_age_s": 0,
                    "transport": "acp-stdio",
                    "upstream": {
                        "ok": True,
                        "probed": False,
                        "endpoint": "",
                        "model": cli_label,
                        "max_context": 0,
                        "backend": cli_label,
                    },
                }

            # Endpoint: caller-provided > config default.
            if endpoint:
                ep = endpoint
                # Derive backend/model/headers from config's model table
                # when not provided by caller.
                if not backend or not headers:
                    try:
                        row = cfg.model_row_for_endpoint(ep)
                        if not backend and isinstance(row, dict):
                            backend = row.get("backend", "") or ""
                        if not headers:
                            headers = cfg.upstream_headers(ep)
                    except Exception:  # noqa: BLE001
                        pass
                if not model:
                    model = getattr(cfg, "upstream_model", "") or ""
                max_context = int(getattr(cfg, "upstream_max_context", 0) or 0)
            else:
                ep = getattr(cfg, "upstream_endpoint", "") or ""
                model = getattr(cfg, "upstream_model", "") or ""
                max_context = int(getattr(cfg, "upstream_max_context", 0) or 0)
                backend = ""
                try:
                    row = cfg.model_row_for_endpoint(ep) if ep else None
                    backend = (row.get("backend") if isinstance(row, dict) else "") or ""
                except Exception:  # noqa: BLE001
                    backend = getattr(cfg, "backend", "") or ""
                headers = cfg.upstream_headers(ep) if ep else {}
        except Exception as exc:  # noqa: BLE001
            # Config unreadable — no probe possible. Surface the load
            # error so the operator sees something better than a stale
            # green ok.
            return {
                "ok": True,
                "agent": self._agent_name,
                "url": self.base_url,
                "uptime_s": 0,
                "pid": getattr(self._proc, "pid", None),
                "heartbeat_age_s": 0,
                "transport": "acp-stdio",
                "upstream": {"ok": False, "error": f"config load failed: {exc}"},
            }

        upstream: dict = {
            "endpoint": ep,
            "model": model,
            "max_context": max_context,
            "backend": backend,
        }
        # Allowlist of backends we'll actively probe. Local inference
        # servers are cheap to ping and the operator wants live signal
        # there. Cloud rows ("unknown" / "openrouter" / "anthropic" /
        # "deepseek") get a probed=False marker instead.
        local_backends = {"vllm", "sglang", "tgi", "llama-cpp", "ollama"}
        if not ep:
            upstream["ok"] = False
            upstream["probed"] = False
            upstream["error"] = "no upstream_endpoint configured"
        elif backend.lower() not in local_backends:
            upstream["ok"] = True
            upstream["probed"] = False
        else:
            import httpx

            url = ep.rstrip("/") + "/models"
            try:
                async with httpx.AsyncClient(
                    timeout=2.0,
                    headers=dict(headers or {}),
                ) as http:
                    r = await http.get(url)
                upstream["probed"] = True
                if r.status_code == 200:
                    upstream["ok"] = True
                else:
                    upstream["ok"] = False
                    upstream["error"] = f"HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                upstream["probed"] = True
                upstream["ok"] = False
                upstream["error"] = type(exc).__name__ + (f": {exc}" if str(exc) else "")

        return {
            "ok": True,
            "agent": self._agent_name,
            "url": self.base_url,
            "uptime_s": 0,
            "pid": getattr(self._proc, "pid", None),
            "heartbeat_age_s": 0,
            "transport": "acp-stdio",
            "upstream": upstream,
        }

    async def stop_server(self) -> dict:
        """No-op on ACP (the agent process is owned by this client).

        Closing the AcpClient closes stdin → child exits cleanly. ACP
        has no daemon-stop method, so this is the closest equivalent.
        """
        await self._safe_close_acp()
        return {"ok": True, "message": "acp child closed"}

    # -- sessions ---------------------------------------------------------

    async def list_sessions(self) -> list[dict]:
        """Best-effort session list.

        ACP's ``session/list`` is capability-gated; if the agent doesn't
        advertise it, we return ``[]`` (the TUI's session picker
        gracefully shows "no sessions" rather than crashing).
        """
        client = await self._ensure_connected()
        try:
            resp = await client.session_list()
        except Exception as exc:
            # The previous handler logged at DEBUG and returned []
            # — which left the TUI's "no sessions yet" panel looking
            # like an empty workspace whenever the wire call actually
            # failed (capability mismatch, schema validation, etc.).
            # WARNING surfaces it in journalctl / stderr so the
            # operator can grep ``session/list`` and find the real
            # error instead of chasing a phantom "empty disk" lead.
            logger.warning(
                "session/list failed on this agent: %s",
                exc,
                exc_info=True,
            )
            return []
        out: list[dict] = []
        for entry in getattr(resp, "sessions", None) or []:
            # Prefer the schema's top-level fields (sessionId, cwd,
            # title, updatedAt) — the agent emits those directly. The
            # ``_meta`` path stays as a fallback for forward-compat
            # with agents that surface sharedagents-specific extras there.
            # Pydantic exposes ``_meta`` as ``field_meta`` due to the
            # alias rename; legacy attrs stay as additional fallbacks.
            meta = (
                getattr(entry, "field_meta", None)
                or getattr(entry, "meta", None)
                or getattr(entry, "_meta", None)
                or {}
            )
            rt = (meta or {}).get("sharedagents", {}) if isinstance(meta, dict) else {}
            sid = getattr(entry, "session_id", None) or getattr(entry, "sessionId", None)
            out.append(
                {
                    "sid": sid,
                    "hash": rt.get("hash") or sid,
                    "title": getattr(entry, "title", None) or rt.get("title", ""),
                    "turns": rt.get("turn_count", 0),
                    "last_activity": (
                        getattr(entry, "updated_at", None)
                        or getattr(entry, "updatedAt", None)
                        or rt.get("last_activity", "")
                    ),
                    "workspace_origin": rt.get("origin", ""),
                    "workspace_cwd": (getattr(entry, "cwd", None) or rt.get("cwd", "")),
                }
            )
        return out

    async def orchestrator_tree(self, include_all: bool = False) -> dict:
        """Cross-agent live monitoring tree (agent → session → subagent).

        This is the SAME data the web ``/agents`` monitor renders — a
        pure read over the shared ``sessions.db`` via
        :func:`sharedagents_agent.orchestrator.session_tree`. Unlike
        :meth:`list_sessions` (ACP ``session/list``, scoped to *this*
        agent), the tree spans EVERY agent's sessions, so the TUI
        monitor can navigate to any session, not just ones spawned here.

        Two transports, matching the rest of this client:

        * **gateway mode** (``_gateway_url`` set, the production path):
          HTTP ``GET /control/v1/orchestrator/tree[?all=1]`` against the
          running gateway, with the bearer token attached.
        * **in-process** (no gateway_url — stdio / test rig): call
          ``session_tree`` directly in a worker thread (it does blocking
          sqlite I/O) since the store is a shared on-disk db either way.

        ``include_all`` maps to the server's ``?all=1`` browse scope
        (every session, not just the live subtree). A gateway error
        never propagates — we return an empty tree shell so the monitor
        shows "no agents" instead of crashing.
        """
        if not self._gateway_url:
            import asyncio as _asyncio

            from sharedagents_agent import orchestrator

            return await _asyncio.to_thread(
                orchestrator.session_tree,
                include_all=include_all,
            )

        import httpx

        url = self._gateway_url + "/control/v1/orchestrator/tree" + ("?all=1" if include_all else "")
        headers: dict[str, str] = {}
        if self._gateway_token:
            headers["Authorization"] = f"Bearer {self._gateway_token}"
        empty = {"agents": [], "summary": [], "totals": {}, "live_states": []}
        try:
            async with httpx.AsyncClient(timeout=10.0, headers=headers) as http:
                resp = await http.get(url)
            if resp.status_code != 200:
                logger.warning(
                    "orchestrator_tree: gateway HTTP %s",
                    resp.status_code,
                )
                return empty
            data = resp.json()
            return data if isinstance(data, dict) else empty
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("orchestrator_tree: gateway fetch failed: %s", exc)
            return empty

    # -- submit + subscribe (spec §9.1 / §9.2 / §9.3) ---------------------
    #
    # The canonical interface contract: an interface SUBMITs a prompt
    # (fire-and-forget) and SUBSCRIBEs to the cursor event stream, then
    # ``render(event)``. No interface drives a turn or holds the agent
    # iterator. These three thin methods are the gateway membrane the TUI
    # uses to invert off the ACP ``session_turn`` iterator onto the
    # log-tail model (live == replay == observe). They reuse the same
    # gateway base URL + bearer token as ``orchestrator_tree``.

    def _gateway_headers(self) -> dict[str, str]:
        # All local operator frontends share one authenticated owner so a
        # session opened by one interface remains available to the others.
        h: dict[str, str] = {"X-Sharedagents-Principal": "local"}
        if self._gateway_token:
            h["Authorization"] = f"Bearer {self._gateway_token}"
        return h

    async def submit_prompt(
        self,
        sid: str,
        text: Any,
        *,
        permission_mode: str = "",
    ) -> dict:
        """Submit a prompt fire-and-forget (spec §9.1).

        ``POST /control/v1/orchestrator/sessions/{sid}/prompt`` runs the
        turn server-side; the body is rendered from the durable log, NOT
        streamed back here. Returns ``{accepted, hash, start_offset}``
        where ``start_offset`` is the byte cursor the caller subscribes
        from so it sees exactly this turn's first event (no gap, no
        prior-turn replay). ``text`` accepts a plain string or an
        OpenAI-multipart list — both normalise to ACP content blocks.
        """
        import httpx

        url = self._gateway_url + f"/control/v1/orchestrator/sessions/{sid}/prompt"
        body: dict[str, Any] = {"prompt": self._user_to_blocks(text)}
        if permission_mode:
            body["permission_mode"] = permission_mode
        async with httpx.AsyncClient(
            timeout=30.0,
            headers=self._gateway_headers(),
        ) as http:
            resp = await http.post(url, json=body)
        if resp.status_code not in (200, 202):
            try:
                payload = resp.json()
            except ValueError:
                payload = {}
            raise GatewayControlError(
                str(payload.get("code") or "submit_failed"),
                str(payload.get("message") or resp.text[:200]),
                resp.status_code,
                payload if isinstance(payload, dict) else {},
            )
        data = resp.json()
        return data if isinstance(data, dict) else {}

    async def cancel_prompt(self, sid: str, *, force: bool = False) -> dict:
        """Cancel the server-side running turn for ``sid`` (spec §9.3).

        ``POST /control/v1/orchestrator/sessions/{sid}/cancel``. The
        gateway resolves ``(session_hash, turn_id)`` and targets the
        in-flight turn; the confirmation surfaces as ``turn_ended(aborted)``
        in the log stream like any other event. ``force=True`` requests a
        hard kill, ``force=False`` a graceful stop — preserving the TUI's
        two-stage ESC behaviour over the gateway membrane.
        """
        import httpx

        url = self._gateway_url + f"/control/v1/orchestrator/sessions/{sid}/cancel"
        async with httpx.AsyncClient(
            timeout=10.0,
            headers=self._gateway_headers(),
        ) as http:
            resp = await http.post(url, json={"force": bool(force)})
        if resp.status_code not in (200, 202):
            return {
                "ok": False,
                "error": f"gateway HTTP {resp.status_code}",
            }
        return {"ok": True}

    async def subscribe_events(
        self,
        sid: str,
        from_offset: int = 0,
        *,
        cancel_event: Any | None = None,
    ) -> AsyncIterator[dict]:
        """Subscribe to the canonical incremental event stream (spec §9.2).

        ``GET /control/v1/orchestrator/sessions/{sid}/stream?mode=events&from=``
        replays every complete JSONL event at/after ``from_offset`` then
        tails live (woken ~5ms by ``session_live``). Yields one parsed
        ``{offset, seq, spawnedBy, event}`` frame per ``event: log`` line.

        Transport framing handled here so callers see only canonical
        frames:

        * ``hb`` heartbeats are swallowed (idle keep-alive, no payload).
          When ``cancel_event`` is set, the next heartbeat (or reconnect
          loop tick) ends the generator so a session switch can re-tail
          without waiting on a stuck idle stream.
        * ``reconnect{offset}`` (lifetime cap / backpressure close) is
          NOT surfaced — we transparently reconnect from the last
          delivered byte offset, so a long-lived subscriber never has to
          re-implement the resume handshake. The byte cursor (never seq)
          is the resume key, so this is zero-loss.
        """
        import json as _json

        import httpx

        cursor = max(0, int(from_offset))
        base = self._gateway_url + f"/control/v1/orchestrator/sessions/{sid}/stream"
        headers = self._gateway_headers()

        def _cancelled() -> bool:
            return bool(cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)())

        while True:
            if _cancelled():
                return
            url = f"{base}?mode=events&from={cursor}"
            reconnect = False
            async with (
                httpx.AsyncClient(
                    timeout=None,
                    headers=headers,
                ) as http,
                http.stream("GET", url) as resp,
            ):
                resp.raise_for_status()
                event_name = "log"
                async for raw in resp.aiter_lines():
                    if _cancelled():
                        return
                    line = raw.rstrip("\r")
                    if not line:
                        event_name = "log"
                        continue
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if event_name == "hb":
                        # Keep-alive only — but honour cancel so a
                        # generation bump can exit an idle tail.
                        if _cancelled():
                            return
                        continue
                    try:
                        frame = _json.loads(payload)
                    except Exception:  # noqa: BLE001
                        continue
                    if event_name == "reconnect":
                        # Lifetime cap / backpressure close: resume
                        # from the last byte offset (zero-loss).
                        if isinstance(frame, dict):
                            nxt = frame.get("offset")
                            if isinstance(nxt, int) and nxt >= cursor:
                                cursor = nxt
                        reconnect = True
                        break
                    if not isinstance(frame, dict):
                        continue
                    off = frame.get("offset")
                    if isinstance(off, int) and off > cursor:
                        cursor = off
                    yield frame
            if _cancelled():
                return
            if not reconnect:
                # Server-side generator ended without a reconnect cue
                # (e.g. the stream's lifetime loop exited): reopen from
                # the current cursor to keep tailing live.
                continue

    async def create_session(
        self,
        *,
        model: str | None = None,
        workspace_cwd: str | None = None,
        workspace_origin: str | None = None,
        endpoint: str | None = None,
        baseUrl: str | None = None,
        backend: str | None = None,
        contextWindow: int | None = None,
    ) -> dict:
        """Create a new session via ACP ``session/new``.

        Most legacy fields don't have an ACP slot; ``cwd`` is the only
        natively-supported parameter. The rest (model, baseUrl, backend,
        origin) ride on top via custom slash commands the TUI re-emits
        as needed; callers don't lose data, but the ACP agent decides
        whether to honour the sharedagents-specific knobs.
        """
        client = await self._ensure_connected()
        # Resolution precedence: explicit ``workspace_cwd`` from the
        # caller > the cwd captured at AgentClient construction (this is
        # the launch dir of the sharedagents process; matches the legacy
        # daemon's behaviour so the agent's WORKSPACE-CONTEXT block sees
        # a real absolute path). Falling back to ``os.getcwd()`` here is
        # only useful if both are missing — should never happen because
        # ``__init__`` snapshots ``os.getcwd()``.
        cwd = workspace_cwd or self._cwd or os.getcwd()
        sid = await client.session_new(cwd, mcp_servers=[])

        # Seed the TUI's per-frontend default for thinking visibility.
        # ``Session.reasoning_visibility`` defaults to ``"off"`` so a
        # fresh session would have the engine swallow every ``thinking``
        # delta (frontend-driven privacy gate). The TUI wants thinking
        # visible by default; Telegram doesn't and uses the disk default.
        # Best-effort: a transport error here shouldn't block session
        # creation — the user can always re-run ``/reasoning on``.
        try:
            set_opt = getattr(client, "session_set_config_option", None)
            if set_opt is not None:
                await set_opt(sid, "reasoning", "on")
        except Exception:  # noqa: BLE001
            pass

        # Model override: pass the alias through to the bridge's
        # ``set_config_option`` so it resolves via ``cfg.models[]`` and
        # updates ``sess.model`` + ``sess.endpoint`` for this session.
        # Mirrors `sharedagents --model <alias> -p "..."` which was silently
        # ignored before this fix — the model alias was parsed from CLI
        # args but never forwarded to the bridge.
        if model:
            try:
                if set_opt is not None:
                    await set_opt(sid, "model", model)
            except Exception:  # noqa: BLE001
                pass

        # Read the bridge-persisted session state from disk so the
        # returned view reflects the ACTUAL upstream the bridge chose,
        # not the agent_cfg defaults we'd otherwise fall back to. This
        # matters because the engine's ``set_config_option`` mutates
        # ``self.cfg.upstream_model`` / ``upstream_endpoint`` in memory
        # only — a ``/model deepseek`` run earlier in this gateway's
        # life leaves the runtime cfg pointing at deepseek while the
        # disk-loaded ``agent_cfg`` (which is what ``default_model`` /
        # ``default_endpoint`` below resolve from) still says local.
        # Without this, the TUI footer rendered "configured backend" at startup and
        # transitioned to "deepseek" the moment the first end-of-turn
        # refresh hit ``get_session`` — the user's bug report.
        #
        # The bridge persists the session JSON synchronously with
        # ``force=True`` in ``new_session`` BEFORE returning the sid
        # (see ``acp_bridge.CoreBackend.new_session`` ~L1545), so the
        # file always exists by the time ``get_session`` reads it. If
        # the read fails for any other reason, the disk-derived fields
        # come back empty and the default_* chain below fills them in.
        try:
            persisted = await self.get_session(sid)
        except Exception:  # noqa: BLE001
            persisted = {}

        cli_label = ""
        try:
            cfg = AgentConfig.load(self._agent_name)
            cli_label = _cli_engine_label(getattr(cfg, "engine", "") or "")
            default_endpoint = getattr(cfg, "upstream_endpoint", "") or ""
            default_model = getattr(cfg, "upstream_model", "") or ""
        except Exception:  # noqa: BLE001
            default_endpoint = ""
            default_model = ""

        persisted_model = str(persisted.get("model") or "")
        persisted_endpoint = str(persisted.get("endpoint") or "")
        persisted_backend = str(persisted.get("backend") or "")
        persisted_ctx = int(persisted.get("context_window") or 0)

        # CLI-engine agents (claude/codex/antigravity) don't drive an HTTP
        # upstream — the inherited ``upstream_model`` / ``upstream_endpoint``
        # belong to the global local-LLM config and don't apply here.
        # Picking ``default_model`` would smear the prior agent's model
        # ("qwen3.6-27b-fp8") into a fresh antigravity session and the TUI
        # footer would read ``antigravity/qwen3.6-27b-fp8`` until the next
        # turn refreshed the session view. Use the engine label directly.
        if cli_label:
            default_model = ""
            default_endpoint = ""
            if not persisted_model:
                persisted_model = cli_label

        # Mirror what /v1/sessions used to return so app.py can keep
        # consuming the dict without changes. Resolution order for the
        # upstream fields: explicit caller override > disk-persisted
        # value > agent_cfg default. The disk-persisted value comes
        # AHEAD of the default so a runtime ``/model`` switch is visible
        # at startup instead of "snapping in" on the first end-of-turn
        # refresh.
        view = {
            "sid": sid,
            "hash": sid,
            "title": persisted.get("title") or "",
            "created_at": persisted.get("created_at") or "",
            "system_prompt": persisted.get("system_prompt") or "",
            "model": model or persisted_model or default_model,
            "endpoint": baseUrl or endpoint or persisted_endpoint or default_endpoint,
            "backend": backend or persisted_backend,
            "context_window": (
                contextWindow or persisted_ctx or (getattr(cfg, "upstream_max_context", 0) if not cli_label else 0) or 0
            ),
            "workspace_cwd": cwd,
            "workspace_origin": workspace_origin or "tui",
            "turns": list(persisted.get("turns") or []),
        }
        self._session_view[sid] = view
        # Invalidate the cached ``_lookup_session_model`` entry that
        # get_session above may have seeded — the next call should
        # reflect any field we just patched (e.g., caller-supplied
        # endpoint override) instead of the bridge's initial choice.
        self._model_lookup_cache.pop(sid, None)
        return dict(view)

    async def get_session(self, sid: str) -> dict:
        """Return the session view, always re-reading from disk.

        The session JSON (``agent_cfg.sessions_dir/<sid>.json``) is the
        authoritative store: the bridge writes it from ``record_turn``
        at every turn end (``session.save()``) and from
        ``set_config_option`` synchronously before returning the wire
        ack. ``_session_view`` has NO live-update path —
        ``session_turn`` streams ACP events for rendering but never
        folds them back into the cache. So a cached snapshot goes
        stale the moment the next turn lands.

        Stale snapshots broke two callers:

        - ``/claude`` (and ``/codex`` / ``/antigravity``) preamble builder
          re-fed the prior conversation as ``[user] <first prompt>``
          and missed every turn after the first cache hit — Claude
          received the WRONG prior conversation (or none at all).
        - ``_poll_session_for_external_turns`` never picked up turns
          appended by Telegram / another TUI / the OpenAI-compat API
          because the cached ``turns`` length never grew.

        Fix: always hydrate from disk on every call. We deliberately
        do NOT overlay cached metadata over the fresh disk read — the
        bridge persists synchronously in ``set_config_option`` (see
        ``acp_bridge.CoreBackend.set_config_option``) so disk is up to
        date by the time ``update_session_model``'s ``await`` returns.
        Overlaying any field that's also persisted would mask
        cross-frontend changes (a Telegram ``/model gpu-c`` would be
        invisible to the TUI as long as the TUI's cache held ``gpu-a``)
        and the cache has no mechanism to detect it should refresh.

        ACP has no ``session/get`` and ``session/load`` returns
        modes/configOptions but not turns, so disk is the only source.
        """
        await self._ensure_connected()

        try:
            from sharedagents_agent.agent import AgentConfig

            cfg = AgentConfig.load(self._agent_name)
            data = _read_session_view(cfg, sid)
            if data is not None and isinstance(data, dict):
                hydrated = {
                    "sid": sid,
                    "hash": data.get("hash") or sid,
                    "title": data.get("title") or "",
                    "created_at": data.get("created_at") or "",
                    "system_prompt": data.get("system_prompt") or "",
                    "model": data.get("model") or "",
                    # Real engine model (claude-opus-4-…/composer-2.5)
                    # for external CLI/ACP agents — the footer prefers
                    # it over the placeholder ``model``.
                    "engine_model": data.get("engine_model") or "",
                    "endpoint": data.get("endpoint") or "",
                    "backend": data.get("backend") or "",
                    "context_window": int(
                        data.get("context_window") or 0,
                    ),
                    "workspace_cwd": data.get("workspace_cwd") or "",
                    "workspace_origin": data.get("workspace_origin") or "",
                    # Active goal so a second TUI binding this session can
                    # replicate the goal-bar (read from the session header).
                    "active_goal": data.get("active_goal") or "",
                    # Per-session teacher state (``/teacher on|off`` +
                    # ``/teacher model <alias>``) so the supervision widget
                    # is restored on attach/resume, not just after a toggle.
                    "teacher_override": data.get("teacher_override") or "",
                    "teacher_model_override": data.get("teacher_model_override") or "",
                    "turns": list(data.get("turns") or []),
                }
                self._session_view[sid] = hydrated
                return dict(hydrated)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "get_session(%s): disk hydrate failed: %s",
                sid,
                exc,
                exc_info=True,
            )

        # Disk read failed OR no file yet (brand-new sid the agent
        # just minted). Fall back to the cached view from
        # ``update_session_model`` (test-rig path where
        # ``session_set_config_option`` isn't wired) or seed an empty
        # stub. Subsequent turns + the next ``session.save()`` will
        # populate the disk path so the next call hydrates real data.
        existing = self._session_view.get(sid)
        if existing:
            return dict(existing)
        view = {
            "sid": sid,
            "hash": sid,
            "title": "",
            "created_at": "",
            "system_prompt": "",
            "model": "",
            "endpoint": "",
            "turns": [],
        }
        self._session_view[sid] = view
        return dict(view)

    async def session_command(self, sid: str, text: str, **extra: Any) -> dict:
        """Run a slash command via the agent's shared server-side runtime.

        SINGLE SOURCE OF TRUTH: the command logic lives in the gateway
        (:mod:`sharedagents_command.command_runtime`), reached over ACP
        ``session/command``. The TUI is a thin renderer — this method only
        translates the :class:`CommandResponse` envelope back into the
        legacy ``{ok,text,data,state_patch}`` shape ``_cmd_shared`` already
        renders (wire-shape adaptation, not feature logic).

        Falls back to the in-process legacy path ONLY when the agent is too
        old to expose ``session/command`` (or the call errors), so a TUI
        pointed at a stale gateway keeps working. The fallback is slated
        for removal once every deployed gateway speaks ``session/command``.
        """
        frontend = extra.get("frontend", "tui")
        confirmed = bool(extra.get("confirmed", False))
        # The legacy local path is used ONLY when the gateway is too old to
        # expose ``session/command`` (capability handshake), never on a
        # command-level error — otherwise a command the runtime
        # deliberately rejects would get a second, divergent interpretation
        # from a parallel code module. Once
        # `_supports_session_command` is known False it stays the legacy
        # path for this client's life; once True the legacy path is dead.
        if self._supports_session_command is False:
            return await self._session_command_legacy(sid, text, **extra)
        try:
            client = await self._ensure_connected()
            envelope = await client.session_command(
                sid,
                text,
                frontend=frontend,
                confirmed=confirmed,
            )
        except AcpRemoteError as exc:
            if getattr(exc, "code", None) == JsonRpcErrorCode.METHOD_NOT_FOUND:
                # Genuine capability gap — gateway predates session/command.
                self._supports_session_command = False
                logger.info(
                    "gateway lacks session/command; using legacy local path",
                )
                return await self._session_command_legacy(sid, text, **extra)
            # A real remote error (bad args, internal) → surface it, do NOT
            # silently reinterpret via the legacy brain.
            return {"ok": False, "command": "", "text": str(exc), "data": {}}
        except Exception as exc:  # noqa: BLE001 - transport/connection failure
            return {"ok": False, "command": "", "text": f"command failed: {exc}", "data": {}}
        if not isinstance(envelope, dict) or "result" not in envelope:
            return {"ok": False, "command": "", "text": "unexpected command response", "data": {}}
        self._supports_session_command = True
        return self._envelope_to_legacy(envelope, sid)

    def _envelope_to_legacy(self, env: dict, sid: str) -> dict:
        """Translate a CommandResponse envelope → legacy TUI result dict.

        Pure wire-shape mapping: collapses ``result`` + ``effects`` into the
        ``{ok,command,text,data,state_patch}`` the TUI footer/transcript
        renderer expects. Mirrors a ``model`` switch into the cached session
        view so the footer updates (the server already performed it)."""
        result = env.get("result") or {}
        legacy: dict[str, Any] = {
            "ok": bool(env.get("ok")),
            "command": env.get("command") or "",
            "text": result.get("text") or "",
            "data": result.get("data") or {},
            # Kind + actions were dropped here, so TUI could not render
            # the same panel kinds the Web UI already shows.
            "kind": result.get("kind") or "text",
            "actions": list(result.get("actions") or []),
            "state_patch": {},
            # Imperative directives (prompt/cancel/new_session/switch_session)
            # passed through verbatim for the TUI to apply generically.
            "effects": list(env.get("effects") or []),
        }
        if env.get("error"):
            legacy["error"] = env["error"]
        if env.get("error_code"):
            legacy["error_code"] = env["error_code"]
        patch: dict[str, Any] = {}
        for eff in env.get("effects") or []:
            if isinstance(eff, dict) and eff.get("type") == "state_patch":
                patch.update((eff.get("payload") or {}).get("patch") or {})
        # Pass EVERY patch key through (don't whitelist — a new runtime key
        # must not be silently dropped here), only
        # renaming the runtime's `mode` to the TUI's `permission_mode`.
        sp = legacy["state_patch"]
        for k, v in patch.items():
            sp["permission_mode" if k == "mode" else k] = v
        # Mirror a model switch into the cached session view (display only).
        if sid and (patch.get("model") or patch.get("endpoint")):
            view = self._session_view.get(sid)
            if view is not None:
                if patch.get("model"):
                    view["model"] = patch["model"]
                if patch.get("endpoint"):
                    view["endpoint"] = patch["endpoint"]
                if "context_window" in patch:
                    view["context_window"] = patch["context_window"]
        return legacy

    async def _session_command_legacy(self, sid: str, text: str, **extra: Any) -> dict:
        """DEPRECATED in-process fallback for slash commands.

        Retained only so a TUI pointed at a gateway too old to expose
        ``session/command`` keeps working. New behavior MUST go in
        :mod:`sharedagents_command.command_runtime` (the server-side source of
        truth), never here. Slated for removal.

        ACP-mappable subset:

        - ``/mode <id>`` → ``session/set_mode``
        - ``/model <alias|url>`` → ``session/set_config_option``
          (option_id="model"); Y2 wires the AcpClient method.
        - ``/reasoning [on|off|stream]`` → ``session/set_config_option``
          (option_id="reasoning"); the engine's per-session gate decides
          whether ``thinking`` deltas reach the wire. ALSO emits a
          ``state_patch`` so the TUI's local chunk handler shows /
          hides the deltas client-side.

        Frontend-local visibility toggles (no agent round-trip):

        - ``/tool[s] [on|off]`` → emits a ``state_patch`` with
          ``tool_output_visible`` so the TUI's ``tool_call`` handler
          mounts (or skips) the inline ``🔧 toolname {args}`` header.
          Tool *result* widgets always stay; only the call-header pill
          is gated. Default is OFF so the transcript stays clean.

        Other legacy commands (``/help``, ``/status``, ...) return
        ``{"ok": True}`` so the TUI doesn't error.
        """
        from sharedagents_command.commands import (
            format_status,
            normalize_context_subcommand,
            normalize_reasoning_level,
            normalize_thinking,
            normalize_tool_output,
            parse_command,
        )

        client = await self._ensure_connected()
        parsed = parse_command(text)
        if parsed is None:
            return {"ok": True}

        # /status — render fresh session state + AcpClient status info.
        # Route through ``get_session`` so a ``/model`` change made on
        # another frontend (Telegram, second TUI) shows up here instead
        # of the snapshot the cache last hydrated.
        if parsed.name == "status":
            view = await self.get_session(sid) if sid else {}
            status_info = await self.status() if hasattr(self, "status") else {}
            data = {
                "agent": status_info.get("agent", "?"),
                "upstream": view.get("endpoint", "(none)"),
                "backend": view.get("backend", "?"),
                "model": view.get("model", "(none)"),
                "session": (sid[:12] if sid else "(none)"),
                "active_turn": "(none)",
                "transport": status_info.get("transport", "?"),
            }
            text_out = format_status(data)
            return {"ok": True, "command": "status", "text": text_out, "data": data}

        # /context — OpenClaw-equivalent context breakdown.
        #
        # Subcommands:
        #     bare /context        — help card
        #     /context list|show   — short breakdown
        #     /context detail|deep — per-file/per-tool/per-skill detail
        #     /context json        — machine-readable
        #
        # Reads the persisted session JSON directly for an accurate
        # snapshot of the system prompt that was actually sent (the
        # in-memory cache only hydrates on first read), then layers in
        # the live tool registry + skills discovery from disk so the
        # report covers everything the model sees.
        if parsed.name == "context":
            sub = normalize_context_subcommand(parsed.arg)
            from sharedagents_agent.context_report import (
                SessionTokens,
                build_context_report,
                format_context_detail,
                format_context_help,
                format_context_json,
                format_context_list,
            )

            if not sub or sub == "help":
                text_out = format_context_help()
                return {
                    "ok": True,
                    "command": "context",
                    "text": text_out,
                    "data": {},
                }
            if sub not in {"list", "show", "detail", "deep", "json"}:
                return {
                    "ok": False,
                    "command": "context",
                    "text": ("Unknown /context mode.\nUse: /context, /context list, /context detail, or /context json"),
                    "data": {},
                }

            # Read everything from disk as the source of truth — same
            # rationale as ``get_session`` (the in-memory cache has no
            # live-update path and would mask cross-frontend changes).
            # We need ``system_mode`` which ``get_session`` doesn't
            # surface, so a direct read is still warranted here.
            ctx_window = 0
            model = ""
            system_prompt = ""
            permission_mode = ""
            workspace_cwd = ""
            turns: list[dict[str, Any]] = []
            cfg_obj = None
            try:
                from sharedagents_agent.agent import AgentConfig

                cfg_obj = AgentConfig.load(self._agent_name)
                raw = _read_session_view(cfg_obj, sid)
                if raw is not None and isinstance(raw, dict):
                    turns = list(raw.get("turns") or [])
                    ctx_window = int(raw.get("context_window") or 0)
                    model = raw.get("model") or ""
                    system_prompt = raw.get("system_prompt") or ""
                    permission_mode = raw.get("system_mode") or ""
                    workspace_cwd = raw.get("workspace_cwd") or ""
            except Exception as exc:  # noqa: BLE001
                logger.debug("/context disk read failed: %s", exc)

            latest: dict[str, Any] = {}
            for t in reversed(turns):
                m = t.get("metrics") if isinstance(t, dict) else None
                if isinstance(m, dict) and m:
                    latest = m
                    break

            tools_schemas: list[dict[str, Any]] = []
            tool_summaries: list[tuple[str, str]] = []
            deferred_catalog_text = ""
            skills_block_text = ""
            skills_list: list[Any] = []
            try:
                from sharedagents_agent.tools import BUILTIN_TOOLS, ToolRegistry

                registry = ToolRegistry(BUILTIN_TOOLS)
                tools_schemas = registry.openai_schemas()
                tool_summaries = [(t.name, t.description or "") for t in registry.tools]
                deferred_catalog_text = registry.deferred_catalog_block()
            except Exception as exc:  # noqa: BLE001
                logger.debug("/context tool registry read failed: %s", exc)

            agent_home = cfg_obj.home if cfg_obj is not None and getattr(cfg_obj, "home", None) else None
            try:
                from sharedagents_skill.skills import (
                    discover_skills,
                    skills_summary_block,
                )

                skills_block_text = skills_summary_block(agent_home)
                skills_list = list(discover_skills(agent_home))
            except Exception as exc:  # noqa: BLE001
                logger.debug("/context skills read failed: %s", exc)

            if agent_home is None:
                from pathlib import Path as _Path

                agent_home = _Path.home() / ".sharedagents"

            report = build_context_report(
                agent_home=agent_home,
                mode=permission_mode or "auto",
                system_prompt=system_prompt,
                tools_schemas=tools_schemas,
                tool_summaries=tool_summaries,
                deferred_catalog_text=deferred_catalog_text,
                skills_block_text=skills_block_text,
                skills=skills_list,
                model=model,
                workspace_cwd=workspace_cwd,
                permission_mode=permission_mode,
                source="run" if system_prompt else "estimate",
            )
            session_tok = SessionTokens(
                total_tokens=(
                    int(latest.get("prompt_tokens")) if isinstance(latest.get("prompt_tokens"), (int, float)) else None
                ),
                total_tokens_fresh=(
                    int(latest.get("prompt_tokens_fresh"))
                    if isinstance(latest.get("prompt_tokens_fresh"), (int, float))
                    else None
                ),
                input_tokens=(
                    int(latest.get("prompt_tokens")) if isinstance(latest.get("prompt_tokens"), (int, float)) else None
                ),
                output_tokens=(
                    int(latest.get("completion_tokens"))
                    if isinstance(latest.get("completion_tokens"), (int, float))
                    else None
                ),
                context_tokens=ctx_window or None,
            )

            if sub == "json":
                text_out = format_context_json(report, session_tok)
            elif sub in ("detail", "deep"):
                text_out = format_context_detail(report, session_tok)
            else:  # list / show
                text_out = format_context_list(report, session_tok)
            return {
                "ok": True,
                "command": "context",
                "text": text_out,
                "data": {
                    "model": report.model,
                    "system_prompt_chars": report.system_prompt_chars,
                    "tool_schema_chars": report.tool_schema_chars,
                    "skills_count": len(report.skills_entries),
                    "tool_count": len(report.tool_entries),
                    "session_total_tokens": session_tok.total_tokens,
                    "context_window": session_tok.context_tokens,
                },
            }

        # /thinking [on|off] — display-side flag (the agent's chat
        # template thinking knob is set at session/new time; mid-session
        # changes update the TUI's status bar without a wire round-trip).
        if parsed.name == "thinking":
            val = normalize_thinking(parsed.arg) if parsed.arg else None
            if not parsed.arg:
                return {
                    "ok": True,
                    "command": "thinking",
                    "text": "usage: /thinking [on|off]",
                    "data": {},
                }
            if val is None:
                return {
                    "ok": False,
                    "command": "thinking",
                    "text": f"unknown thinking value {parsed.arg!r}. use on or off",
                    "data": {},
                }
            return {
                "ok": True,
                "command": "thinking",
                "text": f"thinking → {'on' if val else 'off'}",
                "data": {"thinking_enabled": val},
                "state_patch": {"thinking_enabled": val},
            }

        # /todo[s] — show todos from the cached session view. The model
        # writes them via the ``todo`` tool; we surface whatever the
        # latest tool_call_update wrote into the view's metadata.
        if parsed.name == "todo":
            view = self._session_view.get(sid) or {}
            todos = view.get("todos") or []
            if not todos:
                return {
                    "ok": True,
                    "command": "todo",
                    "text": ("no todos yet — the model adds them via the `todo` tool when planning multi-step work."),
                    "data": {"todos": []},
                }
            lines = ["session todos:"]
            for i, item in enumerate(todos, 1):
                if isinstance(item, dict):
                    state = item.get("state", "?")
                    body = item.get("text") or item.get("body") or ""
                    box = "[x]" if state == "done" else "[ ]"
                    lines.append(f"  {i}. {box} {body}")
                else:
                    lines.append(f"  {i}. {item}")
            return {
                "ok": True,
                "command": "todo",
                "text": "\n".join(lines),
                "data": {"todos": todos},
            }

        # /history [N] — show recent turns. Route through
        # ``get_session`` (disk-fresh) so the listing reflects every
        # turn the bridge has persisted, including turns from other
        # frontends — the cached view never picks up streamed deltas.
        if parsed.name == "history":
            try:
                n = int(parsed.arg) if parsed.arg.strip() else 5
            except ValueError:
                return {
                    "ok": False,
                    "command": "history",
                    "text": f"usage: /history [N]; got {parsed.arg!r}",
                    "data": {},
                }
            view = await self.get_session(sid) if sid else {}
            turns = list(view.get("turns") or [])[-max(1, n) :]
            if not turns:
                return {
                    "ok": True,
                    "command": "history",
                    "text": "no turns yet — send a message to start.",
                    "data": {"turns": []},
                }
            lines = [f"last {len(turns)} turn(s):"]
            for i, t in enumerate(turns, 1):
                user = (t.get("user") or "")[:80]
                # The persisted session schema (Session.save) writes
                # the assistant reply under ``content``; ``assistant``
                # is an older field-name kept here as a back-compat
                # fallback so historic JSONs still render. Without the
                # ``content`` first-lookup, current-session turns
                # render with an EMPTY ``↳`` line — the bug
                # test_10/test_11 ``test_history_command`` failed to
                # catch because they assert only on the ``❯`` user
                # token. Mirrors the Telegram handler at
                # bot.py:3786-3789.
                assistant = str(
                    t.get("content") or t.get("assistant") or "",
                )[:80]
                lines.append(f"  {i}. ❯ {user}")
                if assistant:
                    lines.append(f"     ↳ {assistant}")
            return {
                "ok": True,
                "command": "history",
                "text": "\n".join(lines),
                "data": {"turns": turns},
            }

        if parsed.name == "reasoning":
            level = normalize_reasoning_level(parsed.arg) if parsed.arg else None
            if level is None and parsed.arg:
                return {
                    "ok": False,
                    "command": "reasoning",
                    "text": (f"unknown reasoning level {parsed.arg!r}. use /reasoning [on|off|stream]"),
                    "data": {},
                }
            if level is None:
                # bare /reasoning — treat as a status query, no patch
                return {
                    "ok": True,
                    "command": "reasoning",
                    "text": "usage: /reasoning [on|off|stream]",
                    "data": {},
                }
            # Push to the agent so the engine's per-session gate
            # (`local_llm.py` `_run_round`) actually starts forwarding
            # `thinking` deltas. Without this the TUI flips only its
            # local visibility flag and the upstream stays silent.
            set_opt = getattr(client, "session_set_config_option", None)
            if set_opt is not None:
                try:
                    await set_opt(sid, "reasoning", level)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("session/set_config_option failed: %s", exc)
                    return {
                        "ok": False,
                        "command": "reasoning",
                        "text": f"reasoning switch failed: {exc}"[:200],
                        "error": str(exc),
                        "data": {},
                    }
            return {
                "ok": True,
                "command": "reasoning",
                "text": f"reasoning → {level}",
                "data": {"reasoning_visibility": level},
                "state_patch": {"reasoning_visibility": level},
            }

        if parsed.name == "teacher":
            # ``/teacher`` — control the teacher-model supervisor for THIS
            # session. Subcommands:
            #   /teacher                  → status (on/off + which model) + how-to
            #   /teacher on | off         → toggle supervision (works goal or not)
            #   /teacher model <alias>    → pick which models[] row supervises
            #   /teacher <alias>          → shorthand for `model <alias>`
            #   /teacher model default    → revert to the configured teacher row
            raw = (parsed.arg or "").strip()
            arg = raw.lower()
            set_opt = getattr(client, "session_set_config_option", None)

            def _teacher_models() -> tuple[str, list[str]]:
                """(configured-teacher-label, [candidate model labels])."""
                try:
                    from sharedagents_agent.agent import AgentConfig

                    cfg_obj = AgentConfig.load(self._agent_name)
                    rows = getattr(cfg_obj, "models", None) or []
                    cfg_teacher = ""
                    cands: list[str] = []
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        label = str(r.get("alias") or r.get("id") or "")
                        if label:
                            cands.append(label)
                        if r.get("role") == "teacher" and not cfg_teacher:
                            cfg_teacher = label
                    return cfg_teacher, cands
                except Exception:  # noqa: BLE001
                    return "", []

            # ----- status (no arg) ------------------------------------------
            if not raw:
                cfg_teacher, cands = _teacher_models()
                lines = [
                    "usage: /teacher [on|off] | /teacher model <alias>",
                    "  on|off          — toggle supervision (any session, goal or not)",
                    "  model <alias>   — pick the supervising model (model default = configured teacher)",
                ]
                if cfg_teacher:
                    lines.append(f"configured teacher: {cfg_teacher}")
                if cands:
                    lines.append("available models: " + ", ".join(cands))
                return {
                    "ok": True,
                    "command": "teacher",
                    "text": "\n".join(lines),
                    "data": {"models": cands},
                }

            # ----- on / off toggle ------------------------------------------
            if arg in ("on", "off", "0", "1", "true", "false", "yes", "no", "enable", "disable", "stop"):
                want = "off" if arg in ("off", "0", "false", "no", "disable", "stop") else "on"
                if set_opt is not None:
                    try:
                        await set_opt(sid, "teacher", want)
                    except Exception as exc:  # noqa: BLE001
                        return {
                            "ok": False,
                            "command": "teacher",
                            "text": f"teacher toggle failed: {exc}"[:200],
                            "error": str(exc),
                            "data": {},
                        }
                return {
                    "ok": True,
                    "command": "teacher",
                    "text": f"teacher supervision → {want}",
                    "data": {"teacher": want},
                    "state_patch": {"teacher_enabled": want == "on"},
                }

            # ----- model selection ------------------------------------------
            # Accept `/teacher model <alias>`, `/teacher use <alias>`, or a bare
            # `/teacher <alias>`. The engine validates the alias against
            # models[] and raises on an unknown one.
            parts = raw.split()
            alias = " ".join(parts[1:]).strip() if parts[0].lower() in ("model", "use") else raw
            if set_opt is not None:
                try:
                    await set_opt(sid, "teacher_model", alias)
                except Exception as exc:  # noqa: BLE001
                    return {
                        "ok": False,
                        "command": "teacher",
                        "text": f"teacher model select failed: {exc}"[:300],
                        "error": str(exc),
                        "data": {},
                    }
            shown = alias or "default (configured teacher)"
            return {
                "ok": True,
                "command": "teacher",
                "text": f"teacher model → {shown}",
                "data": {"teacher_model": alias},
                "state_patch": {"teacher_model": alias},
            }

        if parsed.name == "tool":
            visible = normalize_tool_output(parsed.arg) if parsed.arg else None
            if visible is None and parsed.arg:
                return {
                    "ok": False,
                    "command": "tool",
                    "text": f"unknown tool visibility {parsed.arg!r}. use /tools [on|off]",
                    "data": {},
                }
            if visible is None:
                return {
                    "ok": True,
                    "command": "tool",
                    "text": "usage: /tools [on|off]",
                    "data": {},
                }
            return {
                "ok": True,
                "command": "tool",
                "text": f"tool headers → {'on' if visible else 'off'}",
                "data": {"tool_output_visible": visible},
                "state_patch": {"tool_output_visible": visible},
            }

        if parsed.name == "mode":
            from sharedagents_command.command_runtime import (
                _NATIVE_MODE_PROFILES,
                _native_tab_agent,
            )
            from sharedagents_command.commands import normalize_mode

            arg = parsed.arg.strip()
            agent = str(getattr(self, "_agent_name", "") or "").strip().lower()
            if agent in _NATIVE_MODE_PROFILES:
                nxt = _native_tab_agent(agent, arg)
                if nxt is None:
                    return {
                        "ok": False,
                        "command": "mode",
                        "text": (f"unknown mode {arg!r}. native agents use plan|build (OpenCode Tab)."),
                        "data": {},
                    }
                if nxt != agent:
                    try:
                        await self.switch_agent(sid, nxt, frontend="tui")
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("native /mode agent switch failed: %s", exc)
                        return {
                            "ok": False,
                            "command": "mode",
                            "text": f"mode switch failed: {exc}"[:200],
                            "error": str(exc),
                            "data": {},
                        }
                return {
                    "ok": True,
                    "command": "mode",
                    "text": f"mode → {nxt}",
                    "data": {"agent": nxt},
                    "state_patch": {"agent": nxt},
                }
            if not arg:
                return {
                    "ok": True,
                    "command": "mode",
                    "text": "usage: /mode [plan|build]",
                    "data": {},
                }
            level = normalize_mode(arg)
            if level is None:
                return {
                    "ok": False,
                    "command": "mode",
                    "text": (f"unknown mode {arg!r}. use plan or build."),
                    "data": {},
                }
            try:
                # Declare the TUI frontend so the server capability check
                # sees an ask-capable origin.
                await client.session_set_mode(sid, level, frontend="tui")
            except Exception as exc:  # noqa: BLE001
                logger.debug("session/set_mode failed: %s", exc)
                return {
                    "ok": False,
                    "command": "mode",
                    "text": f"mode switch failed: {exc}"[:200],
                    "error": str(exc),
                    "data": {},
                }
            return {
                "ok": True,
                "command": "mode",
                "text": f"mode → {level}",
                "data": {"permission_mode": level},
                "state_patch": {"permission_mode": level},
            }

        if parsed.name == "model":
            arg = parsed.arg.strip()
            set_opt = getattr(client, "session_set_config_option", None)
            if set_opt is not None and arg:
                try:
                    await set_opt(sid, "model", arg)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("session/set_config_option failed: %s", exc)
                    return {
                        "ok": False,
                        "command": "model",
                        "text": f"model switch failed: {exc}"[:200],
                        "error": str(exc),
                        "data": {},
                    }
            # Mirror the resolved row into the cached view so the
            # TUI's footer (state · agent · mode · ctx · model ·
            # upstream) updates immediately. When the user typed an
            # alias (``/model gpu-b``), look up the row in
            # ``cfg.models[]`` and copy BOTH the model id AND the
            # baseUrl — without the baseUrl update,
            # ``_alias_for_endpoint(...)`` keeps resolving against
            # the stale endpoint and the footer reads
            # ``model kimi/gpu-b`` instead of ``gpu-b/<id>``. When the
            # user typed a raw URL the row lookup falls through to
            # endpoint-only update.
            view = self._session_view.setdefault(sid, {"sid": sid, "turns": []})
            if arg:
                if arg.startswith("http"):
                    view["endpoint"] = arg
                    view.pop("alias", None)
                else:
                    # Alias path. Look up the configured row to pull
                    # the canonical baseUrl + model id forward.
                    matched = None
                    try:
                        from sharedagents_agent.agent import AgentConfig

                        cfg_obj = AgentConfig.load(self._agent_name)
                        for row in getattr(cfg_obj, "models", None) or []:
                            if row.get("alias") == arg:
                                matched = row
                                break
                    except Exception:  # noqa: BLE001
                        matched = None
                    if matched is not None:
                        if matched.get("baseUrl"):
                            view["endpoint"] = str(matched["baseUrl"])
                        if matched.get("id"):
                            view["model"] = str(matched["id"])
                        else:
                            view["model"] = arg
                        if matched.get("contextWindow"):
                            try:
                                view["context_window"] = int(matched["contextWindow"])
                            except (TypeError, ValueError):
                                pass
                        view["alias"] = arg
                    else:
                        view["model"] = arg
                        view["alias"] = arg
            self._model_lookup_cache.pop(sid, None)
            return {
                "ok": True,
                "command": "model",
                "text": f"model → {arg}",
                "data": {
                    "id": view.get("model", ""),
                    "baseUrl": view.get("endpoint", ""),
                    "model": view.get("model", ""),
                    "endpoint": view.get("endpoint", ""),
                    "alias": view.get("alias", ""),
                    "contextWindow": view.get("context_window"),
                },
            }

        return {"ok": True}

    async def update_session_model(
        self,
        sid: str,
        *,
        baseUrl: str,
        id: str | None = None,
        alias: str | None = None,
        backend: str | None = None,
        contextWindow: int | None = None,
    ) -> dict:
        """Switch the bound session's model via ACP ``session/set_config_option``.

        The bridge resolves the wire ``value`` against
        ``cfg.models[]``. It accepts an alias (``"kimi"``) or a raw
        baseUrl — never a model id, because two aliases can share the
        same upstream id and the bridge can't disambiguate. Order of
        preference for the wire value: explicit ``alias`` arg →
        ``baseUrl`` (works because ``cfg.model_row_for_endpoint``
        matches by URL) → bare ``id`` (legacy fallback, only works
        when id == alias).

        Local view is updated AFTER the wire call succeeds so a
        failed switch doesn't leave the TUI stamping stale identity.
        """
        client = await self._ensure_connected()
        # Prefer the alias the caller picked. ``baseUrl`` is the
        # canonical fallback because the bridge's
        # ``model_row_for_endpoint`` resolves URLs against
        # ``cfg.models[]`` deterministically. ``id`` is a last-resort
        # for the rare alias==id case.
        config_value = alias or baseUrl or id
        if not config_value:
            return {"ok": False, "error": "no alias / baseUrl / id supplied"}
        # session_set_config_option may not exist on a pre-Y2 AcpClient;
        # tolerate the missing method so the local-only update path
        # still works in test rigs.
        set_opt = getattr(client, "session_set_config_option", None)
        if set_opt is not None:
            try:
                await set_opt(sid, "model", config_value)
            except Exception as exc:
                logger.debug(
                    "session/set_config_option(model=%s) failed: %s",
                    config_value,
                    exc,
                )
                # Local view UNCHANGED on failure — the previous
                # behaviour would mutate before calling the wire and
                # leave the TUI lying about the active model when the
                # call raised.
                return {"ok": False, "error": str(exc)}

        # Wire call succeeded (or wasn't available) — now mirror to
        # the local view so ``get_session()`` reads the new identity
        # immediately. The agent's authoritative state is what the
        # bridge persisted; this just avoids a round-trip on the
        # next read.
        view = self._session_view.setdefault(sid, {"sid": sid, "turns": []})
        view["endpoint"] = baseUrl
        if id is not None:
            view["model"] = id
        if alias is not None:
            view["alias"] = alias
        if backend is not None:
            view["backend"] = backend
        if contextWindow is not None:
            view["context_window"] = contextWindow
        self._model_lookup_cache.pop(sid, None)
        return {"ok": True, **{k: view[k] for k in view if k != "turns"}}

    async def update_session_endpoint(
        self,
        sid: str,
        *,
        endpoint: str,
        model: str | None = None,
        backend: str | None = None,
    ) -> dict:
        return await self.update_session_model(
            sid,
            baseUrl=endpoint,
            id=model,
            backend=backend,
        )

    async def delete_session(self, sid: str) -> dict:
        client = await self._ensure_connected()
        try:
            await client.session_close(sid)
        except Exception as exc:
            logger.debug("session/close failed: %s", exc)
            return {"ok": False, "error": str(exc)}
        finally:
            self._session_view.pop(sid, None)
        return {"ok": True}

    async def inject_user_message(self, sid: str, text: str) -> dict:
        """Push a mid-turn user message to the agent's inject buffer.

        Routes through ACP ``session/inject_user_message``. The agent
        drains the buffer at every round boundary so the operator can
        steer an in-flight plan without aborting. Returns the
        backend's ack dict (``{"ok": True, "queued": True}`` on
        success) or ``{"ok": False, "error": "..."}`` on transport
        failure. Frontends that get ``ok=False`` should fall back to
        the local turn queue.
        """
        client = await self._ensure_connected()
        method = getattr(client, "session_inject_user_message", None)
        if method is None:
            return {
                "ok": False,
                "error": "AcpClient does not support session/inject_user_message",
            }
        try:
            ack = await method(sid, text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("session/inject_user_message failed: %s", exc)
            return {"ok": False, "error": str(exc)}
        return dict(ack) if isinstance(ack, dict) else {"ok": True}

    # -- durable follow-up queue -----------------------------------------
    #
    # Thin proxies to the underlying AcpClient's ``session/queue/*`` RPCs so
    # the TUI can call ``self._client.session_queue_*`` uniformly. Without
    # these the TUI hit ``AttributeError: 'AgentClient' object has no
    # attribute 'session_queue_add'`` and silently fell back to the local
    # (non-durable) queue. Each mirrors ``inject_user_message``'s contract:
    # return the backend ack dict, or ``{"ok": False, "error": ...}`` on a
    # missing capability / transport failure.

    async def session_queue_list(self, sid: str) -> dict:
        """List durable pending follow-up items for *sid*."""
        client = await self._ensure_connected()
        method = getattr(client, "session_queue_list", None)
        if method is None:
            return {"ok": False, "error": "AcpClient does not support session/queue/list"}
        try:
            res = await method(sid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("session/queue/list failed: %s", exc)
            return {"ok": False, "error": str(exc)}
        return dict(res) if isinstance(res, dict) else {"ok": True}

    async def session_queue_add(self, sid: str, text: str, *, source: str = "", author: str = "") -> dict:
        """Durably append *text* to *sid*'s follow-up queue."""
        client = await self._ensure_connected()
        method = getattr(client, "session_queue_add", None)
        if method is None:
            return {"ok": False, "error": "AcpClient does not support session/queue/add"}
        try:
            res = await method(sid, text, source=source, author=author)
        except Exception as exc:  # noqa: BLE001
            logger.debug("session/queue/add failed: %s", exc)
            return {"ok": False, "error": str(exc)}
        return dict(res) if isinstance(res, dict) else {"ok": True}

    async def session_queue_update(self, sid: str, queue_id: str, text: str) -> dict:
        """Update the text of an existing queued item *queue_id*."""
        client = await self._ensure_connected()
        method = getattr(client, "session_queue_update", None)
        if method is None:
            return {"ok": False, "error": "AcpClient does not support session/queue/update"}
        try:
            res = await method(sid, queue_id, text)
        except Exception as exc:  # noqa: BLE001
            logger.debug("session/queue/update failed: %s", exc)
            return {"ok": False, "error": str(exc)}
        return dict(res) if isinstance(res, dict) else {"ok": True}

    async def session_queue_remove(self, sid: str, queue_id: str) -> dict:
        """Remove queued item *queue_id* from *sid*'s follow-up queue."""
        client = await self._ensure_connected()
        method = getattr(client, "session_queue_remove", None)
        if method is None:
            return {"ok": False, "error": "AcpClient does not support session/queue/remove"}
        try:
            res = await method(sid, queue_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("session/queue/remove failed: %s", exc)
            return {"ok": False, "error": str(exc)}
        return dict(res) if isinstance(res, dict) else {"ok": True}

    # -- turn streaming ---------------------------------------------------

    async def session_turn(
        self,
        sid: str,
        *,
        user: Any,
        enable_thinking: bool | None = None,
        permission_mode: str | None = None,
    ) -> AsyncIterator[tuple[str, dict, str | None]]:
        """Stream a turn, yielding TUI-shaped (event, data, turn_id) tuples.

        Translation map (ACP → legacy TUI events):

        - ``agent_message_chunk`` → ``("chunk", {choices: [{delta: {content}}]}, tid)``
        - ``agent_thought_chunk`` → ``("chunk", {choices: [{delta: {reasoning}}]}, tid)``
        - ``tool_call`` (status pending/in_progress) → ``("tool_call", {tool, arguments, call_id}, tid)``
        - ``tool_call_update`` (status completed/failed) → ``("tool_result", {call_id, ok, content}, tid)``
        - ``tool_call_update`` (incremental content) → ``("tool_delta", {call_id, text}, tid)``
        - ``current_mode_update`` → suppressed (TUI tracks mode locally)
        - ``PromptResponse{stopReason}`` → ``("done", {state, stopReason}, tid)``

        The legacy ``permission_request`` / ``permission_response`` /
        ``metrics`` / ``auto_continue`` events have no ACP equivalent on
        the prompt stream; they're surfaced via inbound
        ``session/request_permission`` requests handled separately.
        """
        client = await self._ensure_connected()
        # Best-effort honour of legacy mode hint via session/set_mode.
        if permission_mode:
            try:
                # Declare the TUI frontend so the server capability check
                # sees an ask-capable origin.
                await client.session_set_mode(
                    sid,
                    permission_mode,
                    frontend="tui",
                )
            except Exception:
                logger.debug("session_set_mode pre-prompt failed", exc_info=True)

        # Mint a synthetic turn id so app.py can call turn_stop() on it.
        turn_id = uuid.uuid4().hex
        self._turn_to_session[turn_id] = sid

        # Build ACP content blocks from the ``user`` arg (string or
        # OpenAI multipart list). Image content is dropped here — this
        # agent advertises ``promptCapabilities.image = false`` — and
        # we let the agent error if it sees any other unsupported
        # block.
        blocks = self._user_to_blocks(user)

        # Wall-clock timer so the TUI's stats footer renders even for
        # ACP-only turns (the bridge doesn't ship a `metrics` event
        # over the wire — the legacy worker stream did). For
        # direct-dispatch turns like /claude /codex /compact this is
        # the ONLY metric we have; for normal turns it still gives
        # users a wt row when the bridge omits richer telemetry.
        t0 = time.monotonic()
        # ``first_chunk_at`` = time of the first OBSERVABLE stream event
        # from the agent (content text, tool call, tool delta, tool
        # result, permission ask). For CLI engines (codex/antigravity) that
        # fire many tool calls before emitting prose, the old behavior
        # — stamping ttft only on prose — produced a 100s+ ttft that
        # was actually the agent's full tool-using runtime, not its
        # prefill latency. ``first_content_at`` tracks the separate
        # "time to first prose chunk" metric for callers that need it.
        first_chunk_at: float | None = None
        first_content_at: float | None = None
        chars_out = 0
        tool_events_seen = 0
        # Suppress the synthetic done-time metrics emit when the bridge
        # already shipped per-model-call metrics over `usage_update`.
        # Without this gate, the per-call numbers (ttft / prefill_tps /
        # split tokens) on the LAST round get clobbered by a turn-level
        # wt-only synthetic payload.
        bridge_metrics_seen = False
        # Model + endpoint stamping so the TUI's metrics footer can
        # render `model · …`. Pulled from the persisted session JSON
        # (truth source for both endpoint + model) and the agent
        # config's models[] table for the contextWindow. Read once
        # per turn — the values are stable until /model switches.
        model_label, ctx_window = self._lookup_session_model(sid)

        try:
            async for item in client.session_prompt(sid, blocks):
                for translated in self._translate_acp_item(item, turn_id):
                    ev_kind, ev_data, _ = translated
                    if ev_kind == "chunk":
                        delta = (ev_data.get("choices") or [{}])[0].get("delta") or {}
                        piece = delta.get("content") or ""
                        if piece:
                            if first_chunk_at is None:
                                first_chunk_at = time.monotonic()
                            if first_content_at is None:
                                first_content_at = time.monotonic()
                            chars_out += len(piece)
                    elif ev_kind in ("tool_call", "tool_delta", "tool_result", "permission_request"):
                        # Tool activity is a real stream event — the
                        # agent's prefill ended the moment it started
                        # working. Without this branch a CLI agent
                        # that runs N tool calls before emitting prose
                        # reads ``ttft`` as the whole turn duration.
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        if ev_kind in ("tool_call", "tool_result"):
                            tool_events_seen += 1
                    if ev_kind == "metrics":
                        bridge_metrics_seen = True
                    if ev_kind == "done" and not bridge_metrics_seen:
                        now = time.monotonic()
                        wt_ms = (now - t0) * 1000.0
                        m: dict[str, Any] = {"wt_ms": wt_ms}
                        if model_label:
                            m["model"] = model_label
                        if ctx_window:
                            m["context_window"] = ctx_window
                        if first_chunk_at is not None:
                            m["ttft_ms"] = (first_chunk_at - t0) * 1000.0
                        # Surface "time to first prose chunk" as a
                        # separate metric whenever it differs from
                        # ttft (i.e. tool calls ran before prose).
                        if (
                            first_content_at is not None
                            and first_chunk_at is not None
                            and first_content_at > first_chunk_at + 1e-3
                        ):
                            m["ttfc_ms"] = (first_content_at - t0) * 1000.0
                        if tool_events_seen:
                            m["tool_events"] = tool_events_seen
                        if chars_out > 0 and first_content_at is not None:
                            stream_s = max(now - first_content_at, 1e-3)
                            est_tok = max(int(chars_out / 4), 1)
                            m["completion_tokens"] = est_tok
                            m["tokens_estimated"] = True
                            m["tps"] = est_tok / stream_s
                        yield ("metrics", m, turn_id)
                    yield translated
        except Exception as exc:
            yield ("error", {"message": str(exc) or repr(exc)}, turn_id)
            yield ("done", {"state": "error", "error": str(exc)}, turn_id)
        finally:
            self._turn_to_session.pop(turn_id, None)

    @staticmethod
    def _user_to_blocks(user: Any) -> list[dict]:
        if isinstance(user, str):
            return [{"type": "text", "text": user}]
        if isinstance(user, list):
            blocks: list[dict] = []
            for part in user:
                if isinstance(part, dict) and part.get("type") == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                # image_url / other parts: drop silently (image
                # content is gated by promptCapabilities).
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            return blocks
        return [{"type": "text", "text": str(user)}]

    @staticmethod
    def _translate_acp_item(
        item: Any,
        turn_id: str,
    ) -> list[tuple[str, dict, str | None]]:
        """Translate one AcpClient stream item into zero-or-more TUI events.

        ``item`` is either a ``SessionNotification`` (carries a single
        ``update`` with a discriminated ``sessionUpdate`` field) or a
        terminal ``PromptResponse`` carrying ``stopReason``.
        """
        # Terminal: PromptResponse → ("done", ...)
        stop_reason = getattr(item, "stop_reason", None)
        if stop_reason is not None and not hasattr(item, "update"):
            state = "done" if stop_reason == "end_turn" else ("aborted" if stop_reason == "cancelled" else "done")
            return [("done", {"state": state, "stopReason": stop_reason}, turn_id)]

        update = getattr(item, "update", None)
        if update is None:
            return []
        kind = getattr(update, "session_update", None) or getattr(update, "type", None)

        if kind == "agent_message_chunk":
            text = AgentClient._extract_text(getattr(update, "content", None))
            if not text:
                return []
            return [("chunk", {"choices": [{"delta": {"content": text}}]}, turn_id)]

        if kind == "agent_thought_chunk":
            text = AgentClient._extract_text(getattr(update, "content", None))
            if not text:
                return []
            return [("chunk", {"choices": [{"delta": {"reasoning": text}}]}, turn_id)]

        if kind == "user_message_chunk":
            # We don't render echoed user chunks; suppress.
            return []

        if kind == "tool_call":
            return [
                (
                    "tool_call",
                    {
                        "tool": getattr(update, "title", None) or getattr(update, "kind", None) or "tool",
                        "call_id": getattr(update, "tool_call_id", "") or "",
                        "arguments": getattr(update, "raw_input", None) or {},
                    },
                    turn_id,
                )
            ]

        if kind == "tool_call_update":
            status = getattr(update, "status", None)
            call_id = getattr(update, "tool_call_id", "") or ""
            content = AgentClient._extract_tool_content(getattr(update, "content", None))
            if status in ("completed", "failed"):
                return [
                    (
                        "tool_result",
                        {"call_id": call_id, "ok": status == "completed", "content": content},
                        turn_id,
                    )
                ]
            # Incremental content during a still-running tool.
            if content:
                return [("tool_delta", {"call_id": call_id, "text": content}, turn_id)]
            return []

        if kind == "current_mode_update":
            # Suppressed: TUI tracks permission_mode locally; if we
            # ever want to surface a model-driven mode flip, route it
            # through the status bar instead.
            return []

        if kind == "usage_update":
            # The bridge ships per-model-call metrics (ttft / prefill /
            # tps / token splits) inside `_meta.sharedagents.metrics` on
            # every `usage_update` so the TUI can stamp the stats line
            # for EACH round, not just the last one. Fall back to the
            # raw `used`/`size` numbers when the bridge didn't tunnel
            # rich metrics (e.g., a non-sharedagents upstream).
            meta = (
                getattr(update, "field_meta", None) or (update.get("_meta") if isinstance(update, dict) else None) or {}
            )
            sa_meta = meta.get("sharedagents") if isinstance(meta, dict) else None
            payload = (sa_meta or {}).get("metrics") if isinstance(sa_meta, dict) else None
            size = getattr(update, "size", None)
            used = getattr(update, "used", None)
            if not payload:
                payload = {
                    "prompt_tokens": int(used) if used is not None else None,
                    "context_window": int(size) if size else None,
                }
            else:
                # Always include context_window from the usage_update's
                # top-level `size` field — the worker metrics don't carry
                # it (only the bridge adds it via the `context_window`
                # parameter to `turn_done_to_acp`). Without this
                # `_last_context_window` never updates and the TUI footer
                # falls back to the health-probe snapshot.
                payload = dict(payload)
                if isinstance(size, (int, float)) and size > 0:
                    payload.setdefault("context_window", int(size))
                elif isinstance(used, (int, float)) and used > 0:
                    payload.setdefault("context_window", int(used))
            return [("metrics", dict(payload), turn_id)]

        if kind == "plan":
            # Plan updates are agent-internal; the TUI doesn't render
            # them.
            return []

        return []

    @staticmethod
    def _extract_text(content: Any) -> str:
        if content is None:
            return ""
        # ContentBlock can be a pydantic model or a dict (depending on
        # whether the SDK validated it). Handle both.
        text = getattr(content, "text", None)
        if text is not None:
            return str(text)
        if isinstance(content, dict):
            return str(content.get("text", "") or "")
        return ""

    @staticmethod
    def _extract_tool_content(content: Any) -> str:
        if content is None:
            return ""
        # tool content is a list of ToolCallContent; each may carry a
        # plain text block.
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                inner = getattr(item, "content", None) or item
                t = AgentClient._extract_text(inner)
                if t:
                    chunks.append(t)
            return "".join(chunks)
        return AgentClient._extract_text(content)

    # -- per-turn control -------------------------------------------------

    def register_turn(self, tid: str, sid: str) -> None:
        """Bind a turn id to its session so :meth:`turn_stop` can route a
        cancel. In gateway mode the turn id is the canonical ``turn_idx`` taken
        from the event stream (the ACP ``session_turn`` path registers it
        itself); without this binding ``turn_stop`` resolves no session and
        ESC/abort no-ops with ``unknown turn id`` (spec §9.3 cancel routing).
        """
        if tid and sid:
            self._turn_to_session[tid] = sid

    async def turn_stop(self, tid: str, *, force: bool = False) -> dict:
        """Map turn-id back to session and fire ``session/cancel``.

        ESC in the TUI resolves through here; the only authoritative
        kill path is the agent's own ``session/cancel``-driven
        termination of the in-flight prompt.

        ``force=False`` (``/stop`` / first ESC) requests a graceful
        stop; ``force=True`` (``/abort`` / second ESC) requests a hard
        kill (SIGKILL + killpg the worker and every runner / subagent).
        """
        sid = self._turn_to_session.get(tid)
        if sid is None:
            # We don't know which session this id refers to; safest is
            # to no-op rather than guess and cancel an innocent prompt.
            return {"ok": False, "error": "unknown turn id"}
        # Gateway membrane (spec §9.3): route the cancel through the
        # server-side cancel route, which resolves the running turn_id
        # and surfaces ``turn_ended(aborted)`` in the log stream. The
        # two-stage graceful/hard distinction rides ``force``. Falls
        # back to ACP ``session/cancel`` in stdio / test mode.
        if self._gateway_url:
            return await self.cancel_prompt(sid, force=force)
        client = await self._ensure_connected()
        try:
            await client.session_cancel(sid, force=force)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    async def turn_pause(self, tid: str) -> dict:
        raise NotImplementedError(
            "ACP has no pause primitive — only session/cancel. There is no sharedagents equivalent."
        )

    async def turn_resume(self, tid: str) -> dict:
        raise NotImplementedError("ACP has no resume primitive (pause/resume don't exist); see turn_pause.")

    async def append_turn(
        self,
        sid: str,
        *,
        user: Any,
        content: str,
        thinking: str = "",
        metadata: dict | None = None,
    ) -> dict:
        raise NotImplementedError(
            "append_turn is not part of ACP. Delegated turns ride on "
            "the standard tool_call / tool_call_update stream and the "
            "agent persists them as part of the turn it served. "
            "Frontends that need to record an external exchange should "
            "emit it as a tool_call from the local model instead."
        )

    async def compact_session(
        self,
        sid: str,
        *,
        keep_recent_turns: int = 1,
        min_result_chars: int = 800,
        head_chars: int = 400,
    ) -> dict:
        """No-op compaction.

        Compaction is sharedagents-internal and the agent does it
        automatically on overflow. The ``/compact`` slash UI is kept
        as a convenience but performs nothing on the wire.
        """
        await self._ensure_connected()
        return {"ok": True, "message": "compaction is automatic", "turns_compacted": 0}

    # -- delegate (slash sugar over tool calls) ---------------------------

    async def delegate(
        self,
        sid: str,
        *,
        prompt: str,
        tool: str = "delegate_claude",
        mode: str | None = None,
    ) -> AsyncIterator[tuple[str, dict, str | None]]:
        """Stream a delegate-tool round.

        Direct dispatch — no LLM round. Invokes the named delegate
        tool against the user's prompt, streams Claude/Codex stdout
        back, persists as a turn with ``metrics.source = tool``. The
        ACP path prepends the
        ``__SHAREDAGENTS_DELEGATE__:<tool>:<json-args>`` sentinel to a
        single-block prompt; the bridge's prompt() method intercepts
        the marker before any LLM round and runs the tool directly.
        Same wire shape: tool_call announce, optional
        tool_call_update streaming chunks via ``on_delta`` inside
        the runner, terminal tool_call_update +
        PromptResponse(end_turn).

        ESC still works the same way (session/cancel SIGTERMs the
        delegate subprocess). Permission is bypassed deliberately —
        the slash command is an explicit user opt-in.

        Every TUI operator slash delegation (`/claude`, `/codex`,
        `/antigravity`) AND the `--claude`/`--codex` bash bridges funnel
        through this one builder, so the pre-flight checklist the gate
        (`sharedagents_agent.delegation_gate`) requires for
        an expensive engine is auto-filled HERE — frontend-set, never
        model-reachable. An operator typing the command is a direct human
        command that IS the intent + authorization, so blocking it behind
        the model-facing checklist form would be wrong.
        """
        import json as _json

        from sharedagents_agent.operator_delegation import operator_slash_checklist

        # Auto-filled pre-flight checklist for this direct operator slash
        # delegation (frontend-set, not settable by the model).
        args: dict[str, Any] = {"prompt": prompt, **operator_slash_checklist(prompt)}
        if mode:
            args["mode"] = mode
        marker = f"__SHAREDAGENTS_DELEGATE__:{tool}:{_json.dumps(args)}"
        async for triple in self.session_turn(sid, user=marker):
            yield triple

    @staticmethod
    def _deny_permission(reason: str) -> dict[str, Any]:
        """ACP ``session/request_permission`` response that rejects the
        call once.

        The default for every path where a human didn't actively click
        "allow": no callback registered, callback raised, callback
        returned garbage. A destructive op must never run because the
        UI layer failed — "no answer" means "no".
        """
        logger.info("tui: permission denied — %s", reason)
        return {"outcome": {"outcome": "selected", "optionId": "reject_once"}}

    async def _inbound_request_permission(self, params: dict[str, Any]) -> dict[str, Any]:
        """Inbound ``session/request_permission`` JSON-RPC handler.

        Delegates to the frontend-registered callback (set via
        :meth:`set_permission_callback`) — the TUI installs one that
        pops the 3-button permission modal and returns the user's
        choice. With no callback wired, or if the callback raises /
        returns a malformed result, we DENY (``reject_once``): a
        destructive tool reaching this handler at all means the
        agent's evaluator already decided it needs human sign-off, so
        the safe default with no human in the loop is no.

        Return shape is the ACP ``RequestPermissionResponse`` mapping
        verbatim: ``{"outcome": {"outcome": "selected",
        "optionId": "allow_once" | "allow_always" | "reject_once" |
        "reject_always"}}`` (or ``{"outcome": {"outcome": "cancelled"}}``).
        """
        tool_call = params.get("toolCall") if isinstance(params, dict) else {}
        title = tool_call.get("title") if isinstance(tool_call, dict) else ""
        cb = self._permission_callback
        if cb is None:
            return self._deny_permission(f"no permission callback registered (tool={title!r})")
        try:
            result = await cb(params)
        except Exception:  # noqa: BLE001
            logger.exception("tui: session/request_permission callback raised — denying")
            return self._deny_permission(f"callback raised (tool={title!r})")
        outcome = result.get("outcome") if isinstance(result, dict) else None
        if not isinstance(outcome, dict) or not outcome.get("outcome"):
            return self._deny_permission(f"callback returned malformed result (tool={title!r})")
        return result

    def set_permission_callback(
        self,
        callback: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None,
    ) -> None:
        """Register an async callback to answer ``session/request_permission``.

        The TUI registers one that pops the 3-button permission modal
        and returns the user's choice. Passing ``None`` clears it and
        re-enables the deny-by-default fallback.

        Callback contract:
          * input:  the JSON-RPC ``params`` dict the agent sent —
            ``{"sessionId", "toolCall": {...}, "options": [...]}``.
          * output: the ACP ``RequestPermissionResponse`` mapping —
            ``{"outcome": {"outcome": "selected", "optionId": <id>}}``
            or ``{"outcome": {"outcome": "cancelled"}}``.
        """
        self._permission_callback = callback

    async def submit_permission(self, tid: str, call_id: str, response: str) -> dict:
        """Resolve a pending ``session/request_permission``.

        Inbound permission requests in ACP are JSON-RPC requests the
        agent sends to the client and the client must answer.
        ``app.py`` keeps a permission modal, calls this method with
        the modal's selection, and expects an ack. The actual ACP
        wiring lives in :mod:`sharedagents_acp.client.handlers`; here we
        just return an ack so the TUI flow keeps moving.
        """
        await self._ensure_connected()
        return {"ok": True, "call_id": call_id, "response": response}

    # -- AskUserQuestion round-trip ----------------------------------

    def set_ask_user_callback(
        self,
        callback: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None,
    ) -> None:
        """Register an async callback to answer ``session/ask_user`` requests.

        The TUI registers a callback that pops a modal and returns the
        user's selections; Telegram / OpenAI-endpoint frontends register
        an auto-answer shim (first-option). Passing ``None`` clears the
        callback and re-enables the built-in auto-answer fallback.

        Callback contract:
          * input:  the JSON-RPC ``params`` dict the agent sent —
            ``{"sessionId", "callId", "questions": [...]}``
          * output: a dict with an ``"answers"`` list, one entry per
            question in input order; each entry is
            ``{"question": <str>, "selected": [<label>, ...], "notes": <str>}``.
        """
        self._ask_user_callback = callback

    @staticmethod
    def _auto_answer_ask_user(params: dict[str, Any]) -> dict[str, Any]:
        """Synthesize unanswered replies (fail-closed; never option 0).

        Used when no frontend callback is registered (headless tests,
        a transient registration gap) or when the callback raises. The
        agent gets a parseable tool_result without forged consent —
        better than hanging on a wire-level methodNotFound (Off29 P0).
        """
        questions = params.get("questions") if isinstance(params, dict) else []
        if not isinstance(questions, list):
            questions = []
        answers: list[dict[str, Any]] = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            answers.append(
                {
                    "question": str(q.get("question") or ""),
                    "selected": [],
                    "notes": "(unanswered: no interactive frontend)",
                }
            )
        return {"answers": answers}

    async def _inbound_ask_user(self, params: dict[str, Any]) -> dict[str, Any]:
        """Inbound ``session/ask_user`` JSON-RPC handler.

        Delegates to the frontend-registered callback (set via
        :meth:`set_ask_user_callback`). On missing / failing callback,
        falls back to the synthetic auto-answer so the bridge gets a
        valid response and the model can continue.
        """
        cb = self._ask_user_callback
        if cb is None:
            logger.warning(
                "tui: session/ask_user fired with no callback registered — returning unanswered (fail-closed)"
            )
            return self._auto_answer_ask_user(params)
        try:
            result = await cb(params)
        except Exception:  # noqa: BLE001
            logger.exception("tui: session/ask_user callback raised — falling back to unanswered (fail-closed)")
            return self._auto_answer_ask_user(params)
        if not isinstance(result, dict) or not isinstance(result.get("answers"), list):
            logger.warning(
                "tui: session/ask_user callback returned malformed result "
                "(%r) — falling back to unanswered (fail-closed)",
                type(result).__name__,
            )
            return self._auto_answer_ask_user(params)
        return result


__all__ = ["AgentClient", "spawn_hook"]
