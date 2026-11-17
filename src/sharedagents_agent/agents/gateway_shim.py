"""``AgentACPBackend`` — single, ACP-compliant backend for every agent.

The unified ACP surface the gateway mounts at ``/acp/<agent>`` for
EVERY agent regardless of engine kind. Single in-process backend
per agent; no child subprocess for the backend itself.

For ``local_llm`` engines: prompt routes through
:class:`CoreBackend` (which holds the same :class:`Agent` +
:class:`LocalLLMEngine` instance this backend was built with, so
session-lock / promoted-tools / mode-overrides state is single-
sourced) → :meth:`Agent.run_turn` → :meth:`LocalLLMEngine.run_turn`.
CoreBackend still owns session lifecycle (new_session, load_session,
list_sessions, close_session, cancel) and outbound permission/ask
routing.

For CLI engines (``claude_cli`` / ``codex_cli`` / ``antigravity_cli``):
prompt routes through :meth:`Agent.run_turn` → :meth:`Engine.run_turn`
(:class:`ExternalCliEngine`) which translates the per-CLI
``run_delegate_*_async`` runner's callbacks into the uniform
:data:`AgentTurnEvent` stream. The Agent layer composes the
``--append-system-prompt`` priming + ``--resume`` sha-change policy.

Session lifecycle (new_session, list_sessions, etc.) routes through
CoreBackend for local-LLM and through :class:`Agent` directly for
CLI agents; both paths share the on-disk session format under
``~/.sharedagents/agents/<name>/sessions/``.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from sharedagents_goal.goal_service import (
    run_goal_action as _run_goal_action,
    stop_goals_for_session as _stop_goals_for_session,
)
from sharedagents_loop.loop_service import (
    run_loop_action as _run_loop_action,
    stop_loops_for_session as _stop_loops_for_session,
)

from .agent import Agent
from .engine import (
    EngineKind,
    EngineSessionState,
    TextDelta,
    ToolCallStarted,
    TurnDone,
    TurnError,
    TurnRequest,
)

_LOG = logging.getLogger("sharedagents.agents.gateway_shim")


class AgentACPBackend:
    """Unified ACP backend wrapping one :class:`Agent`.

    Same instance for the lifetime of an agent registration. For
    local-LLM agents we hold a :class:`CoreBackend` that owns
    session lifecycle + outbound channel routing + cancel cascade;
    the actual multi-round turn loop lives in
    :class:`LocalLLMEngine.run_turn` (CoreBackend's ``prompt`` is a
    4-line proxy that forwards through ``Agent.run_turn``).
    Forwards prompt directly to :meth:`Agent.run_turn` for
    CLI-engine agents.
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self._engine_kind = agent.engine.kind

        # CoreBackend hosts the session-lifecycle / outbound-channel
        # / cancel-routing surface for local-LLM agents and proxies
        # ``prompt`` through ``Agent.run_turn``. We pass our
        # factory-built Agent in so CoreBackend reuses it instead of
        # constructing a second Agent + LocalLLMEngine pair —
        # per-session state on the engine (mode_overrides,
        # promoted_tools, session_locks, _cancelled,
        # _pending_user_messages, _active_turns, _session_to_turn)
        # is single-sourced. CLI-engine agents skip this layer
        # entirely; ``Agent.run_turn`` is called directly from
        # :meth:`prompt`.
        self._core: Any | None = None
        if self._engine_kind is EngineKind.LOCAL_LLM:
            from sharedagents_gateway.acp_bridge import CoreBackend

            self._core = CoreBackend(agent.cfg, agent=agent)

        # Per-session active turn map — populated by the new CLI-engine
        # prompt path so :meth:`cancel` can route through
        # :meth:`Agent.abort`. For local-LLM agents, cancel forwards to
        # CoreBackend.cancel which has its own per-session map.
        self._cli_active_turns: dict[str, str] = {}  # sid → turn_id
        # CLI-engine outbound channel state. local-LLM agents use
        # ``self._core``'s equivalents; CLI agents have no inner
        # backend so the binding lives on AgentACPBackend itself.
        # Without these, the spec ``current_mode_update`` notification
        # and any future mode=ask gating fall through to ``None`` and
        # silently drop — exactly the regression antigravity round-7 B3
        # flagged.
        self._cli_server_handle: Any | None = None
        self._cli_session_servers: dict[str, Any] = {}
        from sharedagents_loop.loop_supervisor import schedule_loop_recovery

        schedule_loop_recovery(agent)

    # ---- config hot-reload -----------------------------------------

    def reload_config(self) -> Any:
        """Re-read this agent's on-disk config and apply hot-reloadable
        fields to the live ``cfg`` in place (SIGHUP path).

        ``self.agent.cfg`` is the SAME object the mounted ACP sub-app and —
        for local-LLM agents — the inner ``CoreBackend`` / engine hold, so an
        in-place mutation is picked up on the next turn without re-mounting or
        rebuilding the engine. Structural changes (engine kind, profile, bind)
        are reported but not applied; see
        :meth:`AgentConfig.reload_from_disk`.

        Per-agent ``secrets.env`` is re-read too so a key rotation lands on
        SIGHUP alongside the shared-secrets refresh. Returns the
        :class:`ConfigReloadReport`.
        """
        report = self.agent.cfg.reload_from_disk()
        # Refresh per-agent secrets (process-memory only; best-effort so a
        # bad secrets.env never tanks the reload).
        try:
            from sharedagents_agent.agent import _load_agent_secrets, sharedagents_root

            self.agent.cfg.api_keys = _load_agent_secrets(
                sharedagents_root(),
                self.agent.cfg.name,
            )
        except Exception:  # noqa: BLE001
            pass
        return report

    def has_active_turn(self) -> bool:
        """True if any session on this agent has a turn in flight.

        The mtime config-watcher consults this to DEFER a reload for a busy
        agent (retried next tick) so a poll-triggered cfg mutation never lands
        mid-round. The SIGHUP path deliberately does NOT consult it — an
        operator ``kill -HUP`` is an explicit, one-shot request whose timing
        the operator owns.

        Union of the three in-flight-turn maps (any non-empty ⇒ busy):
          * CLI-engine agents — ``self._cli_active_turns`` (sid → turn_id).
          * local-LLM agents — the engine's ``_active_turns`` (the
            authoritative map ``abort()`` routes through) and CoreBackend's
            ``_new_backend_active_turns`` mirror.
        Defensive ``getattr`` throughout: a missing map reads as idle rather
        than raising inside the watch loop.
        """
        if getattr(self, "_cli_active_turns", None):
            return True
        eng = getattr(self.agent, "engine", None)
        if getattr(eng, "_active_turns", None):
            return True
        core = self._core
        return bool(core is not None and getattr(core, "_new_backend_active_turns", None))

    def active_turn_id(self, session_id: str) -> str:
        """Return the live turn for one physical session, if any."""
        turn_id = str(getattr(self, "_cli_active_turns", {}).get(session_id) or "")
        if turn_id:
            return turn_id
        core = self._core
        if core is not None:
            getter = getattr(core, "active_turn_id", None)
            if callable(getter):
                return str(getter(session_id) or "")
        eng = getattr(self.agent, "engine", None)
        session_map = getattr(eng, "_session_to_turn", None)
        if isinstance(session_map, dict):
            return str(session_map.get(session_id) or "")
        return ""

    def is_session_loaded(self, session_id: str) -> bool:
        """Confirm an in-process attachment without lazy durable loading."""
        if self._core is not None:
            checker = getattr(self._core, "is_session_loaded", None)
            return bool(checker(session_id)) if callable(checker) else False
        # CLI Agent.get_session() is a durable lazy loader, not proof of an
        # attached native session. Conservatively require session/load.
        return False

    async def verify_native_readiness(
        self,
        session_id: str,
        mode_id: str,
    ) -> Mapping[str, Any]:
        """Establish/resume the native engine and verify mode before commit."""
        if self._engine_kind is EngineKind.LOCAL_LLM:
            session = self.agent.get_session(session_id)
            return {
                "ready": bool(session is not None and str(getattr(session, "system_mode", "") or "") == mode_id),
                "kind": "local",
                "mode": mode_id,
            }
        # Cursor ACP: session/load + set_mode already proved attachability.
        # A probe turn that must reply exactly ``READY`` is flaky — Cursor
        # often elaborates or touches tools, which refused switches with
        # ``native readiness handshake failed for cursor`` (Jul 30).
        # Other ACP engines (OpenCode / generic) keep the structured
        # receipt handshake — their readiness tests depend on it.
        if self._engine_kind is EngineKind.CURSOR_ACP:
            session = self.agent.get_session(session_id)
            return {
                "ready": session is not None,
                "kind": self._engine_kind.value,
                "mode": mode_id,
                "session_id": session_id,
                "probe": "attach_only",
            }
        engine = self.agent.engine
        current_state = engine.read_state(self.agent.sessions_dir, session_id)
        state = replace(current_state)
        prior_engine_session_id = current_state.engine_session_id
        session = self.agent.get_session(session_id)
        cwd = str(getattr(session, "workspace_cwd", "") or "")
        workspace_origin = str(getattr(session, "workspace_origin", "") or "")
        system_prompt = self.agent.compose_system_prompt(
            mode=mode_id,
            workspace_cwd=cwd,
            workspace_origin=workspace_origin,
        )
        from .session_log import compute_system_prompt_hash

        system_prompt_sha256 = compute_system_prompt_hash(system_prompt)
        request = TurnRequest(
            prompt=(
                "Acknowledge this mode check briefly without using tools."
                if self._engine_kind is EngineKind.CLAUDE_CLI
                else ("SharedAgents internal native-session readiness handshake. Do not use tools; reply only READY.")
            ),
            mode=mode_id,
            cwd=cwd,
            system_prompt=system_prompt,
            system_prompt_sha256=system_prompt_sha256,
            extra={
                "api_keys": dict(getattr(self.agent.cfg, "api_keys", {}) or {}),
                # Mirror the normal turn path (turn_execution.py): carry the
                # ACTUAL agent's merged upstream_model AND whether it was
                # EXPLICITLY set per-agent, so a readiness handshake resolves
                # the SAME effective default the real turns bind
                # (external_cli.py reads ``extra["upstream_model"]`` /
                # ``extra["upstream_model_explicit"]``). Without these a
                # readiness-shaped request resolved "" and bound the native
                # session to the CLI's default while real turns used the
                # custom per-agent model — and resumes deliberately omit
                # ``--model``, so the wrong binding persisted.
                "upstream_model": str(getattr(self.agent.cfg, "upstream_model", "") or ""),
                "upstream_model_explicit": bool(
                    getattr(
                        self.agent.cfg,
                        "_upstream_model_is_per_agent",
                        False,
                    )
                ),
                "_native_readiness_probe": True,
            },
        )
        saw_error = False
        saw_tool = False
        terminal = ""
        content: list[str] = []
        # Readiness is a transaction candidate. Engines may create a replacement
        # native id, but must not publish it before validation completes.
        with tempfile.TemporaryDirectory(
            prefix=".native-readiness-",
            dir=self.agent.sessions_dir,
        ) as staging:
            staging_dir = Path(staging)
            async for event in engine.run_turn(
                request,
                state,
                sessions_dir=staging_dir,
                turn_id=f"readiness-{session_id}",
            ):
                if isinstance(event, TurnError):
                    saw_error = True
                elif isinstance(event, ToolCallStarted):
                    saw_tool = True
                elif isinstance(event, TextDelta) and event.channel == "content":
                    content.append(event.text)
                elif isinstance(event, TurnDone):
                    terminal = event.state
        verified = state
        receipt = request.extra.get("_native_acceptance_receipt")
        structured_native_mode = bool(
            isinstance(receipt, Mapping)
            and (
                (receipt.get("kind") == "cli_native_mode_terminal" and receipt.get("native_value"))
                or (self._engine_kind is EngineKind.OPENCODE_ACP and receipt.get("kind") == "opencode_success_terminal")
            )
        )
        binding_replaced = bool(prior_engine_session_id and verified.engine_session_id != prior_engine_session_id)
        effective_prompt_sha256 = verified.priming_system_prompt_sha256
        opaque_cli_resume = bool(
            prior_engine_session_id
            and not binding_replaced
            and self._engine_kind
            in {
                EngineKind.CLAUDE_CLI,
                EngineKind.CODEX_CLI,
                EngineKind.ANTIGRAVITY_CLI,
            }
        )
        expected_effective_prompt_sha256 = (
            current_state.priming_system_prompt_sha256 if opaque_cli_resume else system_prompt_sha256
        )
        response_text = "".join(content).strip()
        if not response_text:
            response_text = str((request.extra.get("_engine_audit") or {}).get("content") or "").strip()
        if (
            saw_error
            or saw_tool
            or bool(request.extra.get("_native_tool_observed"))
            or terminal != "ok"
            or (not structured_native_mode and response_text != "READY")
            or not verified.engine_session_id
            or effective_prompt_sha256 != expected_effective_prompt_sha256
            or request.extra.get("_native_mode_evidenced") != mode_id
            or not isinstance(receipt, Mapping)
            or receipt.get("mode") != mode_id
            or receipt.get("effective_prompt_sha256") != effective_prompt_sha256
            or receipt.get("session_id") != verified.engine_session_id
            or receipt.get("kind")
            not in {
                "acp_prompt_response",
                "cli_success_terminal",
                "cli_native_mode_terminal",
                "opencode_success_terminal",
            }
        ):
            raise RuntimeError(f"native readiness handshake failed for {self.agent.cfg.name}")
        candidate_state = {
            field: (getattr(verified, field).value if field == "engine" else getattr(verified, field))
            for field in EngineSessionState.__dataclass_fields__
        }
        return {
            "ready": True,
            "kind": self._engine_kind.value,
            "mode": mode_id,
            "session_id": session_id,
            "engine_session_id": verified.engine_session_id,
            "prior_engine_session_id": prior_engine_session_id,
            "binding_replaced": binding_replaced,
            # This is the prompt snapshot actually bound to the native
            # session. For a resumed CLI mode transition it intentionally
            # remains the prior hash; native per-invocation flags carry the
            # mode without pretending a user message replaced the system
            # channel.
            "effective_prompt_sha256": effective_prompt_sha256,
            "requested_prompt_sha256": system_prompt_sha256,
            "mode_evidence": mode_id,
            "_candidate_state": candidate_state,
            "handshake": "completed",
        }

    def promote_native_readiness(
        self,
        session_id: str,
        readiness: Mapping[str, Any],
    ) -> None:
        """Atomically publish a validated readiness candidate."""
        if self._engine_kind is EngineKind.LOCAL_LLM:
            return
        candidate = readiness.get("_candidate_state")
        if not isinstance(candidate, Mapping):
            raise RuntimeError("native readiness candidate is missing")
        if str(candidate.get("sharedagents_session_id") or "") != session_id:
            raise RuntimeError("native readiness candidate session mismatch")
        engine = self.agent.engine
        current = engine.read_state(self.agent.sessions_dir, session_id)
        expected_prior = str(readiness.get("prior_engine_session_id") or "")
        if current.engine_session_id != expected_prior:
            raise RuntimeError("native readiness binding changed before commit")
        promoted = EngineSessionState(
            sharedagents_session_id=session_id,
            engine=EngineKind(str(candidate["engine"])),
            engine_session_id=str(candidate.get("engine_session_id") or ""),
            priming_status=str(candidate.get("priming_status") or "never"),
            priming_system_prompt_sha256=str(candidate.get("priming_system_prompt_sha256") or ""),
            last_used_at=str(candidate.get("last_used_at") or ""),
            last_error=str(candidate.get("last_error") or ""),
            spawn_cwd=str(candidate.get("spawn_cwd") or ""),
            actual_model=str(candidate.get("actual_model") or ""),
        )
        engine.write_state(self.agent.sessions_dir, promoted)

    def attachment_capabilities(self) -> Mapping[str, Any]:
        """Declared handoff access forms for the active adapter."""
        custom = getattr(self.agent.engine, "attachment_capabilities", None)
        if callable(custom):
            value = custom()
            if isinstance(value, Mapping):
                return value
        return {
            "media_types": [
                "image/*",
                "audio/*",
                "application/pdf",
                "application/json",
                "text/*",
            ],
            "storage_scopes": [
                "local_session",
                "artifact_store",
            ],
            "requires_locator": True,
        }

    def validate_attachment_reference(
        self,
        reference: Mapping[str, Any],
        *,
        workspace_cwd: str = "",
    ) -> Mapping[str, Any]:
        """Validate a real local artifact inside an authorized root.

        Only ``local_session`` and ``artifact_store`` are implemented by the
        default adapter. Remote/token/data locators require an engine-specific
        resolver and are never advertised here.
        """
        custom = getattr(self.agent.engine, "validate_attachment_reference", None)
        if callable(custom):
            value = custom(reference)
            if isinstance(value, Mapping):
                return value
        scope = str(reference.get("storage_scope") or "")
        if scope not in {"local_session", "artifact_store"}:
            return {
                "ok": False,
                "code": "attachment_unsupported",
                "message": "adapter has no authorized resolver for this scope",
            }
        from sharedagents_agent.artifact_links import enrich_local_artifact_reference

        enriched = enrich_local_artifact_reference(
            dict(reference),
            workspace_cwd=workspace_cwd,
            sessions_dir=Path(self.agent.sessions_dir),
            agent_home=Path(getattr(self.agent.cfg, "home", "") or Path(self.agent.sessions_dir).parent),
        )
        if enriched is None:
            return {
                "ok": False,
                "code": "attachment_unavailable",
                "message": "attachment locator is missing or unauthorized",
            }
        return {"ok": True, "reference": enriched}

    # ---- AcpServer hookup ------------------------------------------

    @property
    def server_handle(self) -> Any:
        """Outbound ACP channel — used by mode=ask permission gating
        and ``current_mode_update`` notifications. For local-LLM
        agents forwarded to the inner :class:`CoreBackend`; for
        CLI-engine agents stored on this instance directly."""
        if self._core is not None:
            return getattr(self._core, "server_handle", None)
        return self._cli_server_handle

    @server_handle.setter
    def server_handle(self, value: Any) -> None:
        if self._core is not None:
            self._core.server_handle = value
        else:
            self._cli_server_handle = value

    def bind_session_server(self, session_id: str, server: Any) -> None:
        """Bind ``server`` as the outbound channel for ``session_id``.

        Called by :class:`AcpServer` on ``session/new`` /
        ``session/load`` so permission / ask_user / current_mode_update
        requests for the session route to the connection that owns it.
        The gateway shares one backend across every connection to
        ``/acp/<agent>``; per-session binding lets a multi-client
        deployment (TUI + Telegram) reach the right one.
        """
        if self._core is not None:
            self._core.bind_session_server(session_id, server)
        elif session_id:
            self._cli_session_servers[session_id] = server

    def unbind_session_server(self, session_id: str) -> None:
        """Drop the per-session outbound binding."""
        if self._core is not None:
            self._core.unbind_session_server(session_id)
        else:
            self._cli_session_servers.pop(session_id, None)

    def _outbound_for(self, session_id: str) -> Any:
        """Resolve the outbound ACP channel for ``session_id``.

        Local-LLM path delegates to ``CoreBackend._outbound_for``;
        CLI path uses the new ``_cli_session_servers`` map with a
        fallback to the bare ``server_handle`` for stub setups.
        """
        if self._core is not None:
            return self._core._outbound_for(session_id)
        srv = self._cli_session_servers.get(session_id)
        if srv is not None:
            return srv
        return self._cli_server_handle

    # ---- handshake -------------------------------------------------
    #
    # All method signatures here MUST match what
    # ``sharedagents_acp.server.server.AcpServer`` calls (see
    # ``acp/server/server.py`` for the call sites). The AcpServer is
    # protocol-aware; this backend is the agent-side hookpoint.

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Mapping[str, Any] | None,
        client_info: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        if self._core is not None:
            return await self._core.initialize(
                protocol_version,
                client_capabilities,
                client_info,
            )
        # CLI-engine path: AcpServer's static capabilities suffice;
        # nothing to record on the backend.
        return None

    async def authenticate(self, method_id: str) -> None:
        if self._core is not None:
            await self._core.authenticate(method_id)
        # CLI engines: no-op; bearer auth is gateway-level.

    # ---- session lifecycle -----------------------------------------

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Any],
    ) -> Mapping[str, Any]:
        if self._core is not None:
            return await self._core.new_session(cwd, mcp_servers)
        # CLI-engine path: build a Session via the Agent layer. The
        # response shape MUST match what ``CoreBackend.new_session``
        # returns so local-LLM and CLI agents look identical on the
        # wire — spec-shaped ``modes`` ({availableModes, currentModeId})
        # and ``configOptions`` (list). Snake-case ``config_options`` /
        # ``available_modes`` / ``active_mode`` are non-spec and would
        # break strict ACP clients.
        session = self.agent.create_session(
            workspace_cwd=cwd or "",
            # Named operator surface (not "") so the memory privacy gate
            # resolves a real origin instead of the
            # fail-open empty string.
            workspace_origin="gateway",
        )
        active_mode = session.system_mode or getattr(self.agent.cfg, "permission_mode", "") or "auto"
        return {
            "sessionId": session.hash,
            "modes": self._modes_state(active_mode),
            "configOptions": [],
        }

    async def load_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[Any],
    ) -> Mapping[str, Any]:
        if self._core is not None:
            return await self._core.load_session(
                session_id,
                cwd,
                mcp_servers,
            )
        # CLI-engine path: mirror ``CoreBackend.load_session`` — return
        # one mapping (NOT a stream). The ACP spec's LoadSessionResponse
        # is a single mapping; frontends re-render past turns by reading
        # the on-disk session JSON directly, not by streaming sessionUpdate
        # events back through ``session/load``.
        session = self.agent.get_session(session_id)
        if session is None:
            raise RuntimeError(f"session not found: {session_id!r}")
        if cwd:
            try:
                session.workspace_cwd = cwd
            except Exception:  # noqa: BLE001
                pass
        active_mode = session.system_mode or getattr(self.agent.cfg, "permission_mode", "") or "auto"
        return {
            "modes": self._modes_state(active_mode),
            "configOptions": [],
        }

    async def list_sessions(
        self,
        cursor: str | None,
        cwd: str | None,
    ) -> Mapping[str, Any]:
        if self._core is not None:
            return await self._core.list_sessions(cursor, cwd)
        # CLI-engine agents (claude / codex / antigravity) historically fell
        # through to ``self.agent.list_sessions()`` which returns rows
        # in sharedagents-internal shape (``hash`` / ``turn_count`` /
        # ``created_at``). The ACP SDK validates session/list responses
        # against ``SessionInfo`` (``sessionId`` / ``cwd`` / ``title`` /
        # ``updatedAt``), so the rows were rejected on the wire and the
        # TUI / Telegram pickers silently received []. Use the shared
        # helper that BOTH backend paths now share for an ACP-shape
        # response — same sidecar filtering, same empty-turn drop,
        # same 20-cap.
        from sharedagents_gateway.acp_bridge import list_sessions_acp

        return list_sessions_acp(
            self.agent.cfg.sessions_dir,
            self.agent.cfg.home,
        )

    async def set_session_title(
        self,
        session_id: str,
        title: str,
        *,
        lock: bool = False,
    ) -> dict[str, Any]:
        """Persist ``title`` as this session's title in the session log.

        Used by the web title-sync path so a conversation's title in the web
        UI is the SAME title the session log / monitor tree show. Loads the
        session (live in-memory copy if attached, else folded from disk),
        sets ``title``, and ``save()``s — which emits a ``header_update``
        event that the projector folds into ``sessions_meta.title``. A
        placeholder ``untitled-…`` title is refused (the projector ignores
        those, and they would clobber a real turn-derived title). Same
        code path for local-LLM and CLI agents — both reach the on-disk
        session via :meth:`Agent.get_session`.

        ``lock=True`` additionally pins the title (``title_locked``): the
        title agent never regenerates it again (turn 1, the turn-10 refresh,
        the boot reconciler). Used by the Tasks board when the title is the
        operator-typed task name; ordinary title sync keeps ``lock=False``
        so the automatic lifecycle is untouched.
        """
        title = (title or "").strip()
        if not title or title.startswith("untitled-"):
            return {"ok": False, "reason": "empty or placeholder title"}
        # get_session may hydrate a separate object. Update both live caches
        # before any await so an in-flight title call observes the pin too.
        core_sessions = getattr(getattr(self, "_core", None), "_active_sessions", {})
        engine_sessions = getattr(getattr(self.agent, "engine", None), "_sessions", {})
        cached = [s for s in (core_sessions.get(session_id), engine_sessions.get(session_id)) if s is not None]
        sess = cached[0] if cached else await asyncio.to_thread(self.agent.get_session, session_id)
        if sess is None:
            raise RuntimeError(f"session not found: {session_id!r}")
        try:
            for current in [sess, *cached]:
                current.title = title
                if lock:
                    current.title_locked = True
            await asyncio.to_thread(
                sess.save,
                root=self.agent.sessions_dir,
                force=False,
                force_fields=("title", "title_locked") if lock else ("title",),
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"save failed: {exc}"}
        return {"ok": True, "title": title, "locked": bool(lock)}

    async def close_session(self, session_id: str) -> None:
        if self._core is not None:
            await self._core.close_session(session_id)
            return
        self._cli_active_turns.pop(session_id, None)
        self.agent.close_session(session_id)

    # ---- live config / mode ----------------------------------------

    async def set_mode(self, session_id: str, mode_id: str) -> None:
        from .permissions_integration import (
            VALID_PERMISSION_MODES,
            coerce_permission_mode,
        )

        requested_mode_id = str(mode_id or "").strip()
        mode_id = coerce_permission_mode(mode_id)
        from sharedagents_agent.acp_mode_dict import native_id_for_mode
        from sharedagents_agent.engine_modes import EngineModeRegistry

        caps = EngineModeRegistry().discover(self.agent.name, self.agent.cfg)
        # Native aliases (Cursor ``agent``) and permission-layer ids
        # (``auto`` after coerce) must both resolve. Cursor's catalogue
        # is agent|plan|ask — ``caps.mode("auto")`` is None.
        native = native_id_for_mode(caps.engine_kind, mode_id)
        capability = caps.mode(requested_mode_id) or caps.mode(mode_id) or caps.mode(native)
        if capability is None or not capability.supported:
            allowed = ", ".join(mode.requested_mode_id for mode in caps.modes if mode.supported)
            raise ValueError(f"mode {mode_id!r} is not available for {self.agent.name}; expected {allowed}")
        if self._core is not None:
            await self._core.set_mode(session_id, mode_id)
            return
        # Use the public canonical mode set rather than a local
        # hardcoded literal — keeps the two ``set_mode`` entry points
        # (bridge + shim) in lock-step about which mode_id values are
        # acceptable. ValueError on bogus mode_id mirrors the bridge's
        # behavior.
        sess = self.agent.get_session(session_id)
        previous_session_mode = getattr(sess, "system_mode", "") if sess is not None else ""
        previous_effective_mode = (
            previous_session_mode
            if previous_session_mode in VALID_PERMISSION_MODES
            else getattr(self.agent.cfg, "permission_mode", "auto")
        )
        if previous_effective_mode not in VALID_PERMISSION_MODES:
            previous_effective_mode = "auto"
        # CLI engines apply the mode to the NEXT --resume call's
        # permission flag. Let ValueError propagate so the caller sees
        # the same error semantics as the bridge path; only swallow
        # the OSError / unexpected-exception classes that the engine
        # might raise for transport / runtime issues.
        try:
            self.agent.engine.set_mode(session_id, mode_id)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"engine rejected mode {mode_id!r}: {exc}") from exc
        # Persist on the session record so ``load_session`` round-
        # trips return the right ``currentModeId`` (antigravity round-6
        # finding B3 — CLI-engine set_mode was dropping the persist
        # entirely).
        if sess is not None:
            try:
                sess.system_mode = mode_id
                sess.save(root=self.agent.sessions_dir, force=False)
            except Exception as exc:  # noqa: BLE001
                # Persist BEFORE notifying. A failed persist
                # must NOT be announced or remain effective in memory. Restore
                # both the session field and engine override to the last
                # persisted mode before surfacing the failure.
                sess.system_mode = previous_session_mode
                try:
                    self.agent.engine.set_mode(session_id, previous_effective_mode)
                except Exception:  # noqa: BLE001
                    _LOG.exception(
                        "set_mode: failed to roll back engine mode for %s",
                        session_id,
                    )
                raise RuntimeError(f"failed to persist mode {mode_id!r}: {exc}") from exc
        # Emit the spec-required ``current_mode_update`` notification
        # so connected frontends can refresh their mode UI without
        # re-issuing ``session/load``. Use the per-session outbound
        # resolver so a multi-client (TUI + Telegram) deployment
        # reaches the connection that owns the session, not whichever
        # transport happened to set ``server_handle`` last.
        srv = self._outbound_for(session_id)
        notify = getattr(srv, "notify_session_update", None)
        if notify is None:
            return

        async def _emit() -> None:
            try:
                await notify(
                    session_id=session_id,
                    update={
                        "sessionUpdate": "current_mode_update",
                        "currentModeId": mode_id,
                        # Carry the byte-identical mode list so
                        # a frontend sees the same mode UI whether a CLI engine
                        # or the local-LLM CoreBackend drives the session.
                        "availableModes": self._modes_state(mode_id)["availableModes"],
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        try:
            import asyncio

            asyncio.create_task(_emit())
        except RuntimeError:
            pass

    async def set_config_option(
        self,
        session_id: str,
        option_id: str,
        value: Any,
    ) -> None:
        if self._core is not None:
            await self._core.set_config_option(session_id, option_id, value)
            return
        try:
            self.agent.engine.set_config_option(session_id, option_id, value)
        except ValueError:
            raise
        except Exception:  # noqa: BLE001
            pass

        if option_id == "model":
            sess = self.agent.get_session(session_id)
            if sess is None:
                raise RuntimeError(f"session not found: {session_id!r}")
            engine = self.agent.engine
            kind = getattr(engine, "kind", "")
            kind_value = str(getattr(kind, "value", kind) or "")
            from sharedagents_agent.external_registry import is_external_engine_kind

            if is_external_engine_kind(kind_value):
                getter = getattr(engine, "session_model", None)
                selected = ""
                if callable(getter):
                    try:
                        selected = str(getter(session_id) or "").strip()
                    except Exception:  # noqa: BLE001
                        selected = ""
                if selected:
                    sess.model = selected
                else:
                    # RESET / clear (``/model default|reset|auto|none``):
                    # the engine's override is gone, so resolve the stamp
                    # through the SAME generic effective-default/empty path
                    # the fresh-spawn and session-create paths use —
                    # NEVER ``display_model()`` (which returns the bare
                    # engine label when nothing is known: the codex probe
                    # reproduced a durable session model of "codex" here)
                    # and never ``kind_value``.
                    # For CLI kinds this yields the validated default
                    # (codex: per-agent value / codex config.toml model, or
                    # ""; claude: the operator-explicit
                    # per-agent value / ANTHROPIC_MODEL / settings.json, or
                    # "" — the Claude CLI applies its own default); for ACP
                    # kinds "" (the ACP server owns the model).
                    from sharedagents_agent.discovery._common import (
                        external_cli_effective_default,
                    )

                    sess.model = external_cli_effective_default(
                        kind_value,
                        configured=(
                            str(
                                getattr(
                                    self.agent.cfg,
                                    "upstream_model",
                                    "",
                                )
                                or ""
                            )
                        ),
                        configured_explicit=bool(
                            getattr(
                                self.agent.cfg,
                                "_upstream_model_is_per_agent",
                                False,
                            )
                        ),
                    )
                sess.endpoint = ""
                sess.backend = kind_value
                try:
                    ctx_window = getattr(engine, "context_window", None)
                    sess.context_window = int(ctx_window(sess.model) or 0) if callable(ctx_window) else 0
                except (TypeError, ValueError):
                    sess.context_window = 0
                try:
                    sess.save(root=self.agent.sessions_dir, force=False)
                except Exception:  # noqa: BLE001
                    pass

    async def steer_now(self, session_id: str) -> dict[str, Any]:
        """Tell the live turn on ``session_id`` to take the queued text now.

        Routed like :meth:`cancel`: an external-engine turn (every ACP agent)
        is found in ``_cli_active_turns`` and steered through its engine's
        :meth:`Engine.steer_now`; a local-LLM turn goes to the core bridge.
        ``steered`` is ``False`` when no turn runs here or the engine cannot
        steer in place.
        """
        turn_id = self._cli_active_turns.get(session_id)
        if turn_id:
            steer = getattr(getattr(self.agent, "engine", None), "steer_now", None)
            if not callable(steer):
                return {"ok": True, "steered": False, "reason": "unsupported"}
            try:
                steered = bool(steer(turn_id))
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "steered": False, "reason": f"steer failed: {exc}"}
            return {
                "ok": True,
                "steered": steered,
                "reason": "" if steered else "unsupported",
            }
        core = self._core
        steer = getattr(core, "steer_now", None) if core is not None else None
        if callable(steer):
            return dict(await steer(session_id))
        return {"ok": True, "steered": False, "reason": "no_live_turn"}

    async def inject_user_message(
        self,
        session_id: str,
        text: str,
    ) -> dict[str, Any]:
        """Mid-turn user-message inject — forward to the core bridge.

        The CoreBackend owns the live Agent + LocalLLMEngine pair and
        knows how to push the message into the engine's per-sid
        inject buffer. CLI-only agents (no ``_core``) don't yet
        support steering an in-flight CLI subprocess, so we surface a
        clear "not supported" ack and let the frontend fall back to
        the local turn queue.
        """
        if self._core is not None:
            return dict(await self._core.inject_user_message(session_id, text))
        # CLI engine path: external CLI subprocess owns the turn loop
        # and can't accept mid-flight steering today. Future work
        # could wrap ``claude --resume`` / ``codex exec resume`` for
        # this — out of scope for the local-LLM path.
        return {
            "ok": False,
            "reason": "CLI engines do not support mid-turn inject",
        }

    async def queue_add(
        self,
        session_id: str,
        text: str,
        *,
        source: str = "",
        author: str = "",
        actor: dict[str, Any] | None = None,
        access: Any = None,
    ) -> dict[str, Any]:
        """Enqueue a follow-up message (durable). Returns {ok, queue_id}."""
        try:
            qid = self.agent.queue_add(
                session_id,
                text,
                source=source,
                author=author,
                actor=actor,
                access=access,
            )
        except ValueError:
            return {"ok": False, "error": "empty"}
        return {"ok": True, "queue_id": qid}

    async def queue_list(self, session_id: str) -> dict[str, Any]:
        """List pending follow-up items. Returns {ok, pending:[...]}."""
        return {"ok": True, "pending": self.agent.queue_list(session_id)}

    async def queue_update(
        self,
        session_id: str,
        queue_id: str,
        text: str,
        *,
        access: Any = None,
    ) -> dict[str, Any]:
        """Edit a PENDING queued message. {ok} or {ok:False, error:'not_pending'}."""
        ok = self.agent.queue_update(session_id, queue_id, text, access=access)
        return {"ok": True} if ok else {"ok": False, "error": "not_pending"}

    async def queue_remove(
        self,
        session_id: str,
        queue_id: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Delete a PENDING queued message. {ok} or {ok:False, error:'not_pending'}."""
        kwargs: dict[str, str] = {}
        if reason:
            kwargs["reason"] = reason
        ok = self.agent.queue_remove(session_id, queue_id, **kwargs)
        return {"ok": True} if ok else {"ok": False, "error": "not_pending"}

    async def _run_goal_command(
        self,
        session_id: str,
        arg: str,
    ) -> dict[str, Any]:
        """`/goal` for CLI-engine agents — register durably in ``goals.db``.

        Mirrors :meth:`CoreBackend._run_goal_command` so a goal SET from a
        standard CLI-engine chat (Telegram / web command bar / TUI) lands in
        the registry and is supervised on the live gateway agent, exactly like
        the local-LLM path. Before this existed the CLI command context omitted
        ``goal_command_fn``, so ``/goal`` returned "goals are not available on
        this backend" on every CLI-engine surface.
        """
        sess = self.agent.get_session(session_id) if session_id else None
        if sess is None:
            return {"ok": False, "message": f"session not found: {session_id!r}"}
        attached_sess = sess
        attach = getattr(self.agent.engine, "attach_session", None)
        if callable(attach):
            try:
                attach(session_id, sess)
            except Exception:  # noqa: BLE001
                pass
        from sharedagents_goal.goal_service import execute_goal_command

        result = execute_goal_command(self.agent, sess, arg)
        refreshed = self.agent.get_session(session_id)
        if refreshed is not None:
            sess = refreshed
        # Capture the AUTHORITATIVE post-command projection BEFORE the
        # generic engine goal setter. The engine's
        # ``set_config_option("goal", …)`` → ``_set_goal_state(…,
        # reset_progress=True)`` clears ``goal_terminal`` and
        # ``goal_turn_count`` on ITS cached Session (attach_session) —
        # which, for a session without a sessions_meta row yet, is the very
        # object this backend returns (get_session falls back to the engine
        # cache). The service already wrote the authoritative projection;
        # the setter must not clobber it, and the returned dict must be the
        # service's state, not the setter's leftovers (pause projection
        # clobber in real CLI/ACP adapter paths).
        projection = {
            "active_goal": str(getattr(sess, "active_goal", "") or ""),
            "goal_id": str(getattr(sess, "goal_id", "") or ""),
            "goal_terminal": str(getattr(sess, "goal_terminal", "") or ""),
            "goal_turn_count": int(getattr(sess, "goal_turn_count", 0) or 0),
        }
        if result.ok:
            engine_set_opt = getattr(self.agent.engine, "set_config_option", None)
            if callable(engine_set_opt):
                try:
                    res = engine_set_opt(
                        session_id,
                        "goal",
                        projection["active_goal"],
                    )
                    if hasattr(res, "__await__"):
                        await res
                except Exception:  # noqa: BLE001
                    pass
            # The setter owns ``attached_sess`` while the service/refresh path
            # can own a distinct freshly hydrated ``sess``. Restore both
            # identity domains after binding so the engine cache and the
            # persisted projection agree.
            for target in (attached_sess, sess):
                for key, value in projection.items():
                    try:
                        setattr(target, key, value)
                    except Exception:  # noqa: BLE001
                        pass
            try:
                sess.save(root=self.agent.sessions_dir, force=False)
            except Exception:  # noqa: BLE001
                pass
        return {
            "ok": result.ok,
            "message": result.message,
            "goal": result.goal.to_dict() if result.goal else None,
            "conflicts": result.conflicts,
            # Post-command projection from the REFRESHED session — the
            # authoritative snapshot for the /goal state_patch. This backend
            # resolves a FRESH Session per get_session call, so the command
            # runtime's ctx.live_session can be a stale pre-command object;
            # the returned projection is what the widget must render
            # (stale CLI-engine state patches).
            **projection,
        }

    def goal_action(self, goal_id: str, action: str) -> dict[str, Any]:
        """Run a stop-style goal action (``pause`` / ``archive``) on the LIVE
        agent that owns the supervisor.

        The web Pause/Archive buttons proxy here via the gateway control
        endpoint. Running on the live agent is what makes the supervisor's
        re-drive task actually cancel — calling ``pause_goal`` on an ephemeral
        web-process agent only writes the DB status and no-ops the task cancel.
        """
        if self._core is not None:
            return self._core.goal_action(goal_id, action)
        return _run_goal_action(self.agent, goal_id, action)

    def supervise_goal(self, goal_id: str, *, source: str = "gateway") -> dict[str, Any]:
        if self._core is not None:
            return self._core.supervise_goal(goal_id, source=source)
        gid = str(goal_id or "").strip()
        if not gid:
            return {"ok": False, "reason": "empty goal_id"}
        try:
            self.agent.goal_supervisor.ensure_started(gid, source=source)
            active = bool(self.agent.goal_supervisor.is_active(gid))
            return {
                "ok": active,
                "goal_id": gid,
                "active": active,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "goal_id": gid, "reason": str(exc)}

    async def _run_loop_command(
        self,
        session_id: str,
        arg: str,
    ) -> dict[str, Any]:
        """`/loop` for CLI-engine agents — register durably in ``loops.db``.

        Mirrors :meth:`CoreBackend._run_loop_command` so a loop created from
        a standard CLI-engine chat (Telegram / web command bar / TUI) lands
        in the registry and is supervised on the live gateway agent, exactly
        like the local-LLM path. Before this existed the CLI command context
        omitted ``loop_command_fn``, so ``/loop`` fell through to the legacy
        file-command expansion and never touched the registry.
        """
        sess = self.agent.get_session(session_id) if session_id else None
        if sess is None:
            return {"ok": False, "message": f"session not found: {session_id!r}"}
        attach = getattr(self.agent.engine, "attach_session", None)
        if callable(attach):
            try:
                attach(session_id, sess)
            except Exception:  # noqa: BLE001
                pass
        from sharedagents_loop.loop_service import execute_loop_command

        result = await execute_loop_command(self.agent, sess, arg or "")
        return dict(result)

    def loop_action(self, loop_id: str, action: str) -> dict[str, Any]:
        """Run a loop action (``pause`` / ``resume`` / ``archive``) on the
        LIVE agent that owns the supervisor.

        The web Pause/Resume/Archive buttons proxy here via the gateway
        control endpoint. Running on the live agent is what makes the
        supervisor's drive task actually cancel — calling ``pause_loop`` on
        an ephemeral web-process agent only writes the DB status and no-ops
        the task cancel.
        """
        if self._core is not None:
            return self._core.loop_action(loop_id, action)
        return _run_loop_action(self.agent, loop_id, action)

    async def supervise_loop(self, loop_id: str, *, source: str = "gateway") -> dict[str, Any]:
        """Ensure the loop supervisor is started (exactly one live drive task
        per loop) on the agent that owns it; reports active state.

        The kick is awaited so the report reflects the task just started:
        ``LoopSupervisor.supervise_loop`` runs inline (claim lease, create
        drive task, return) without yielding, so ``is_active`` is true by
        the time we report — closing the false-409 window that an
        ``ensure_future`` + immediate read left open. Unknown or not-running
        loops still fail fast: the kick never creates a task for them.
        """
        if self._core is not None:
            return await self._core.supervise_loop(loop_id, source=source)
        key = str(loop_id or "").strip()
        if not key:
            return {"ok": False, "reason": "empty loop_id"}
        try:
            sup = getattr(self.agent, "loop_supervisor", None)
            if sup is None:
                return {
                    "ok": False,
                    "loop_id": key,
                    "reason": "loop supervisor unavailable",
                }
            result = await sup.supervise_loop(key, source=source)
            active = sup.is_active(key)
            kicked = bool(result.get("ok")) if isinstance(result, dict) else False
            return {"ok": kicked and active, "loop_id": key, "active": active}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "loop_id": key, "reason": str(exc)}

    async def run_command(
        self,
        session_id: str,
        text: str,
        *,
        frontend: str = "api",
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Run a slash command via the shared server-side runtime.

        Local-LLM agents forward to :meth:`CoreBackend.run_command` (full
        live-session access). CLI agents (no ``_core``) have no in-process
        Session, but global/read-only commands (``/help``, ``/skills``,
        ``/models``, ``/mcp``, ``/commands``, ``/plugin``,
        ``/context``, ``/status`` from disk …) still work off ``cfg`` +
        the core modules, so we build a session-less context. Mutating
        session commands degrade to ``unsupported`` rather than crashing.
        """
        if self._core is not None:
            return dict(
                await self._core.run_command(
                    session_id,
                    text,
                    frontend=frontend,
                    confirmed=confirmed,
                )
            )
        from sharedagents_command.command_runtime import CommandContext, run_command as _run_command

        def _monitors() -> list[dict[str, Any]]:
            try:
                from sharedagents_agent.tools.monitor import _registry

                return list(_registry().list())
            except Exception:  # noqa: BLE001
                return []

        async def _list_sessions() -> list[dict[str, Any]]:
            try:
                res = await self.list_sessions(None, None)
            except Exception:  # noqa: BLE001
                return []
            return list(res.get("sessions") or []) if isinstance(res, dict) else []

        live_sess = self.agent.get_session(session_id) if session_id else None
        ctx = CommandContext(
            cfg=self.agent.cfg,
            agent_name=getattr(self.agent.cfg, "name", "") or "",
            session_id=session_id or "",
            frontend=frontend,
            confirmed=confirmed,  # irreversible-command re-challenge
            live_session=live_sess,
            active_turn_id=self._cli_active_turns.get(session_id) if session_id else None,
            # CLI agents DO support these (AgentACPBackend implements them),
            # so /mode, /reasoning, /model etc. keep parity even without a
            # local-LLM CoreBackend.
            set_mode_fn=self.set_mode,
            set_config_option_fn=self.set_config_option,
            list_sessions_fn=_list_sessions,
            cancel_fn=self.cancel,
            monitors_fn=_monitors,
            inject_fn=self.inject_user_message,
            goal_command_fn=lambda arg: self._run_goal_command(session_id, arg),
            loop_command_fn=lambda arg: self._run_loop_command(session_id, arg),
        )
        resp = await _run_command(ctx, text)
        return resp.to_dict()

    def _modes_state(self, active_mode: str) -> dict[str, Any]:
        """Spec-shaped ``SessionModeState`` payload.

        Matches the ACP spec (``schema.json:SessionModeState``) and
        ``CoreBackend._modes_state`` byte-for-byte so frontends see the
        same mode list regardless of which engine drives the agent.
        """
        return {
            "availableModes": [
                {
                    "id": "auto",
                    "name": "auto",
                    "description": "Auto-approve safe tools; prompt on destructive actions.",
                },
                {"id": "ask", "name": "ask", "description": "Prompt for confirmation on every tool call."},
                {"id": "plan", "name": "plan", "description": "Plan-only mode — propose tool calls without executing."},
            ],
            "currentModeId": active_mode,
        }

    # ---- turn dispatch ---------------------------------------------

    async def cancel(self, session_id: str, *, force: bool = False) -> None:
        from sharedagents_agent.tools._log import log_event

        had_active = bool(self._core is not None or self._cli_active_turns.get(session_id))
        log_event(
            "EXTERNAL_CANCEL",
            sid=session_id[:12],
            had_active_turn=had_active,
            force=bool(force),
            mode="hard" if force else "graceful",
            via="gateway_shim",
        )
        if self._core is not None:
            await self._core.cancel(session_id, force=force)
            return
        # An operator stop (``/abort`` OR graceful ``/stop``) on a goal-driven
        # session must halt the goal's outer loop, not just the in-flight turn:
        # without this the supervisor sees the goal still ``running`` and
        # immediately re-drives the next turn ("it just kept running"). Pause
        # the goal (resumable) BEFORE aborting the turn so the supervisor's
        # status re-check folds the loop out instead of starting turn N+1.
        _stop_goals_for_session(self.agent, session_id, reason="aborted")
        # Same contract for loops: an operator stop ends a
        # session's running loops + cancels their pending wakeups — never
        # raises, mirrors the goals path above.
        _stop_loops_for_session(self.agent, session_id, reason="aborted")
        # Codex/Claude/Antigravity cancel used to
        # miss the between-attempt retry sleep (no cancel_retry_burst,
        # stale turn_id). Operator Stop/steer must end that burst even
        # when no CLI driver is registered.
        try:
            cancel_burst = getattr(self.agent, "cancel_retry_burst", None)
            if callable(cancel_burst):
                cancel_burst(session_id)
        except Exception:  # noqa: BLE001
            pass
        # Record the Stop BEFORE any engine is killed: a runner that sees its
        # engine die checks this record to tell a user Stop from an OOM kill.
        # It also gates every automatic turn start (late child results,
        # memory flush, retries) until the user prompts again.
        from sharedagents_agent.agents import session_stop

        stopped_below = session_stop.begin_stop(
            self.agent,
            self.agent.sessions_dir,
            session_id,
            force=force,
        )
        turn_id = self._cli_active_turns.get(session_id)
        if turn_id:
            # Graceful (force=False) must be SIGNAL-FREE.
            # Prefer the engine's cooperative stop hook (mirrors the
            # CoreBackend local-LLM path): the in-flight round/operation
            # finishes and the turn ends with NO signal sent. Only a HARD
            # cancel (force=True) enters the SIGTERM/SIGKILL abort path.
            engine = getattr(self.agent, "engine", None)
            graceful = getattr(engine, "request_graceful_stop", None) if not force else None
            if graceful is not None:
                try:
                    res = graceful(turn_id)
                    if hasattr(res, "__await__"):
                        await res
                except Exception:  # noqa: BLE001
                    pass
            elif force:
                # Hard kill: SIGKILL + killpg the worker + subagents.
                try:
                    await self.agent.abort(
                        turn_id,
                        force=force,
                        session_id=session_id,
                    )
                except TypeError:
                    try:
                        await self.agent.abort(turn_id)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
            # else: graceful cancel on an engine WITHOUT a cooperative hook
            # is a signal-free no-op — the turn finishes naturally; the
            # operator escalates via force=True / a double-press
            # for a hard signal. Never enter the abort/signal path here.
        # Hard abort must reach detached background children even when this
        # session has no mapped in-flight turn (idle jarvis with async work).
        # Silence deliveries so reconcile cannot revive the aborted parent.
        if force:
            cancelled: list[str] = []
            try:
                from sharedagents_agent.agents import async_jobs

                cancelled = async_jobs.cancel_subtree_for_abort(session_id)
            except Exception:  # noqa: BLE001 — must not fail the operator's abort
                pass
            # FIX: the killed runners cannot
            # emit terminal events themselves, and unlike every other
            # hard-abort path (CoreBackend._cancel_session's child cascade,
            # _handle_subagent_abort, the Agent-tool abort) this CLI-engine
            # path never folded the killed sessions' rows out of the live
            # set — an aborted codex subagent stayed "streaming" in the
            # Agent Monitor forever ("stale · streaming 16m"). Mirror the
            # sibling paths: finalize each cancelled job's child row
            # (turn_ended aborted + mark_session_terminal), then fold the
            # target session itself when nothing was claimed.
            try:
                from sharedagents_agent.tools.impl.subagents import (
                    _finalize_cancelled_job_sessions,
                )

                _finalize_cancelled_job_sessions(cancelled)
            except Exception:  # noqa: BLE001 — must not fail the operator's abort
                pass
            # Recursive cascade to every descendant in sessions_meta: kill
            # each engine subtree and fold each still-busy row out of the
            # live set. Finished descendants are left as they are.
            stopped_below = session_stop.stop_descendants(
                self.agent.sessions_dir,
                session_id,
                stopped_below,
                agent=self.agent,
            )
            # A parent parked ``waiting`` on the subagents just stopped has no
            # turn of its own to close: settle it ``idle`` BEFORE the abort
            # write below, which would otherwise fold it to ``error``.
            try:
                from sharedagents_session.session_index import connect_sessions_db, mark_session_terminal

                s_dir = self.agent.sessions_dir
                db = s_dir / "sessions.db"
                if db.exists():
                    con = connect_sessions_db(db)
                    try:
                        p_row = con.execute(
                            "SELECT session_state FROM sessions_meta WHERE session_hash=?",
                            (session_id,),
                        ).fetchone()
                    finally:
                        con.close()
                    if p_row and p_row["session_state"] == "waiting":
                        mark_session_terminal(s_dir, session_id, state="idle")
            except Exception:  # noqa: BLE001
                pass
            try:
                from sharedagents_gateway.control_endpoint import (
                    _abort_child_session_write,
                )

                _abort_child_session_write(session_id)
            except Exception:  # noqa: BLE001 — must not fail the operator's abort
                pass
            # Purge the follow-up queues so stopped items never auto-drain or
            # get recovered into new turns.
            session_stop.purge_queues(self.agent, [session_id, *stopped_below])

    async def prompt(
        self,
        session_id: str,
        message_id: str | None,
        prompt: list[Mapping[str, Any]],
    ) -> AsyncIterator[Mapping[str, Any]]:
        """ACP ``session/prompt`` — drive one turn.

        For local-LLM agents: forward through :class:`CoreBackend`,
        whose ``prompt`` is a 4-line proxy that calls
        :meth:`Agent.run_turn` and translates the AgentTurnEvent
        stream into ACP sessionUpdate dicts via
        :mod:`.acp_translation`.

        For CLI-engine agents: call :meth:`Agent.run_turn` directly
        and translate the AgentTurnEvent stream here.
        """
        from sharedagents_loop.loop_supervisor import schedule_loop_recovery

        schedule_loop_recovery(self.agent)
        from sharedagents_gateway.acp_bridge import _prompt_source

        source = _prompt_source(prompt)
        if self._core is not None:
            core_kwargs = {"source": source} if source != "user" else {}
            async for ev in self._core.prompt(
                session_id,
                message_id,
                prompt,
                **core_kwargs,
            ):
                yield ev
            return

        # CLI-engine path.
        from sharedagents_gateway.acp_bridge import (
            _blocks_to_user_content,
            _flatten_prompt,
            _parse_direct_delegate_marker,
        )

        from . import acp_translation

        # Direct-dispatch sentinel: ``/claude``, ``/codex``,
        # ``/antigravity``, ``/delegate <agent>`` from any frontend.
        # User-facing semantics are "run this prompt against the
        # named agent and return the result" — independent of which
        # agent the user is currently on. Honor the sentinel here so
        # ``/claude`` works whether the active agent is local-LLM
        # (main, scraper, …) or a CLI agent (codex, antigravity,
        # claude itself). The actual cross-agent call rides on the
        # legacy delegate-shim, which under the new architecture
        # drives the named child agent via ``Agent.run_turn``.
        direct = _parse_direct_delegate_marker(prompt)
        if direct is not None:
            # A restricted (Web user) submit never skips the model — except
            # for /compact. Decided on this submit's own stamp: the session's
            # binding is still the previous turn's.
            from sharedagents_core.access_grants import stamp_from_blocks
            from sharedagents_session.access_binding import direct_dispatch_allowed

            if not direct_dispatch_allowed(
                self.agent.sessions_dir,
                session_id,
                direct,
                incoming=stamp_from_blocks(prompt),
            ):
                direct = None
        if direct is not None and str(direct.get("tool") or "") == "compact_context":
            # Native local-LLM handles this sentinel in-process. CLI/ACP
            # engines have no compact_context runner — rewrite to the
            # engine's own compact command (``/compress`` on Hermes). An
            # engine without one still gets ``/compact``: the turn loop
            # answers it itself instead of handing it to the model.
            # Block 0's ``_meta`` (the submitter's grants and name) rides along.
            self._squeeze_sharedagents_session_best_effort(session_id)
            from sharedagents_agent.external_registry import compact_command_for

            compact: dict[str, Any] = {
                "type": "text",
                "text": compact_command_for(getattr(getattr(self.agent, "engine", None), "kind", None)) or "/compact",
            }
            first_meta = prompt[0].get("_meta") if prompt else None
            if isinstance(first_meta, Mapping):
                compact["_meta"] = dict(first_meta)
            prompt = [compact]
            direct = None
        if direct is not None:
            async for msg in self._dispatch_direct_delegate(
                session_id,
                direct,
            ):
                yield msg
            return

        text = _flatten_prompt(prompt)
        parts: list[dict[str, Any]] = []
        if any((b.get("type") or "") != "text" for b in prompt):
            user_content = _blocks_to_user_content(prompt)
            if isinstance(user_content, list):
                parts = [dict(p) for p in user_content]

        run_kwargs: dict[str, Any] = {
            "prompt": text,
            "prompt_parts": parts or None,
        }
        if source != "user":
            run_kwargs["source"] = source
        from sharedagents_session.actors import actor_from_blocks

        _actor = actor_from_blocks(prompt)
        if _actor is not None:
            run_kwargs["actors"] = [_actor]
        from sharedagents_core.access_grants import stamp_from_blocks

        _access = stamp_from_blocks(prompt)
        if _access is not None:
            run_kwargs["access"] = _access
        from sharedagents_core.attachment_context import files_from_blocks

        _files = files_from_blocks(prompt)
        if _files:
            run_kwargs["attachments"] = _files
        events = self.agent.run_turn(session_id, **run_kwargs)

        def _on_turn_started(ev: Any) -> None:
            try:
                self._cli_active_turns[session_id] = ev.turn_id
            except Exception:  # noqa: BLE001
                pass

        try:
            async for msg in acp_translation.agent_events_to_acp(
                events,
                context_window=int(getattr(self.agent.cfg, "upstream_max_context", 0) or 0),
                on_turn_started=_on_turn_started,
            ):
                yield msg
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "agent_events_to_acp failed for sid=%s: %s",
                session_id[:12],
                exc,
            )
            yield {
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": f"[backend error: {exc}]",
                },
            }
            yield {"stopReason": "end_turn"}
        finally:
            self._cli_active_turns.pop(session_id, None)

    def _squeeze_sharedagents_session_best_effort(self, session_id: str) -> None:
        """Trim persisted sharedagents tool bodies before forwarding ``/compact``.

        External engines own their own context window; this only keeps the
        sharedagents JSONL/session mirror lean. Failures must not block delivering
        ``/compact`` to the bound engine.
        """
        try:
            getter = getattr(self.agent, "get_session", None)
            session = getter(session_id) if callable(getter) else None
            if session is None:
                return
            from sharedagents_agent.compaction import compact_session_turns_in_place

            compact_session_turns_in_place(session)
            save = getattr(session, "save", None)
            if callable(save):
                save()
        except Exception:  # noqa: BLE001
            pass

    async def _dispatch_direct_delegate(
        self,
        session_id: str,
        direct: Mapping[str, Any],
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Run a direct-dispatch ``/claude`` / ``/codex`` / ``/antigravity``
        / ``/delegate <agent>`` sentinel and stream ACP events back.

        Works for every active agent — local-LLM AND CLI-engine —
        because the work is performed by a *child* Agent
        (``factory.load_agent(<short>)``) regardless of who the
        parent is. The active agent's session is left untouched;
        the child's reply rides back as one ``tool_call`` /
        ``tool_call_update`` pair so the frontend renders it inline.
        """
        from sharedagents_gateway.acp_bridge import _text_tool_content

        from . import delegate_shims

        tool = str(direct.get("tool") or "")
        args = direct.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}

        shim = delegate_shims.delegate_shim_for_tool(tool)
        if shim is None:
            yield {
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": f"[direct-dispatch: unknown tool {tool!r}]",
                },
            }
            yield {"stopReason": "end_turn"}
            return

        # --- Unconditional owner-visible pre-egress flag ---
        # When sensitive-but-sendable data (positions / PII) is about to leave
        # the local boundary to a THIRD-PARTY delegate, surface an owner-facing
        # message BEFORE the child turn is driven — rendered here in the
        # production ACP adapter, independent of ``/tools`` gating, so the owner
        # always sees it (visibility, not a block; the turn still proceeds and
        # the shared chokepoint durably archives the same flag). Emitted before
        # the driver task so it is on the wire strictly ahead of the call.
        from sharedagents_agent import delegation_gate

        _sensitive = delegation_gate.sensitive_egress_classes(args)
        _third_party = tool in (
            "delegate_claude",
            "delegate_codex",
            "delegate_antigravity",
        )
        if _sensitive and not _third_party and tool == "delegate_agent":
            # A sibling-agent delegate is only third-party when its resolved
            # engine leaves the local boundary; resolve best-effort and fail
            # toward NOT over-flagging a local sibling.
            try:
                from . import factory

                _target = str(
                    args.get("name")
                    or args.get("agent_id")
                    or args.get("agent_name")
                    or args.get("agent")
                    or args.get("target")
                    or ""
                ).strip()
                if _target:
                    _third_party = delegation_gate.is_expensive(
                        factory.load_agent(_target).engine.kind,
                    )
            except Exception:  # noqa: BLE001
                _third_party = False
        if _sensitive and _third_party:
            yield {
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": delegation_gate.sensitive_egress_notice_text(
                        _sensitive,
                        tool.removeprefix("delegate_") if tool != "delegate_agent" else _target,
                    ),
                },
            }

        # Map the delegate tool to a friendly title for the inline
        # ``tool_call`` widget the frontends render.
        title_map = {
            "delegate_claude": "↳ via claude",
            "delegate_codex": "↳ via codex",
            "delegate_antigravity": "↳ via antigravity",
            "delegate_agent": f"↳ via {args.get('name') or 'agent'}",
        }
        call_id = f"direct-{session_id[:8]}-{tool}"
        yield {
            "sessionUpdate": "tool_call",
            "toolCallId": call_id,
            "title": title_map.get(tool, tool),
            "kind": "other",
            "rawInput": dict(args),
        }

        # Bridge ``on_delta`` chunks → ``tool_call_update`` events
        # so the user sees the child's reply stream live.
        ev_queue: asyncio.Queue[Mapping[str, Any] | None] = asyncio.Queue()
        inner_call_ids: dict[str, str] = {}

        def _inner_call_id(raw: Any) -> str:
            raw_id = str(raw or "")
            if raw_id in inner_call_ids:
                return inner_call_ids[raw_id]
            suffix = raw_id or str(len(inner_call_ids) + 1)
            mapped = f"{call_id}.inner.{suffix}"
            inner_call_ids[raw_id] = mapped
            return mapped

        def _on_delta(chunk: str) -> None:
            try:
                if chunk:
                    ev_queue.put_nowait(
                        {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": call_id,
                            "status": "in_progress",
                            "content": _text_tool_content(chunk),
                        }
                    )
            except Exception:  # noqa: BLE001
                pass

        def _on_event(payload: dict) -> None:
            try:
                kind = (payload or {}).get("kind") or ""
                if kind == "tool_use":
                    ev_queue.put_nowait(
                        {
                            "sessionUpdate": "tool_call",
                            "toolCallId": _inner_call_id(payload.get("id")),
                            "title": str(payload.get("name") or "?"),
                            "kind": "other",
                            "status": "in_progress",
                            "rawInput": dict(payload.get("input") or {}),
                        }
                    )
                elif kind == "tool_result":
                    ev_queue.put_nowait(
                        {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": _inner_call_id(payload.get("tool_use_id")),
                            "status": ("completed" if bool(payload.get("ok", True)) else "failed"),
                            "content": _text_tool_content(str(payload.get("text") or "")),
                        }
                    )
            except Exception:  # noqa: BLE001
                pass

        async def _drive() -> Any:
            try:
                return await shim(
                    args,
                    on_delta=_on_delta,
                    on_event=_on_event,
                )
            finally:
                await ev_queue.put(None)  # sentinel

        driver = asyncio.create_task(_drive())
        self._cli_active_turns[session_id] = call_id

        try:
            while True:
                ev = await ev_queue.get()
                if ev is None:
                    break
                yield ev
            try:
                result = await driver
            except asyncio.CancelledError:
                yield {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": "failed",
                    "content": _text_tool_content("[cancelled]"),
                }
                yield {"stopReason": "cancelled"}
                return
            except Exception as exc:  # noqa: BLE001
                yield {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": "failed",
                    "content": _text_tool_content(f"[shim error: {exc}]"),
                }
                yield {"stopReason": "end_turn"}
                return

            body = str(getattr(result, "content", "") or "")
            ok = bool(getattr(result, "ok", True))
            yield {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": "completed" if ok else "failed",
                "content": _text_tool_content(body),
            }
            yield {"stopReason": "end_turn"}
        finally:
            if not driver.done():
                driver.cancel()
                try:
                    await driver
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            self._cli_active_turns.pop(session_id, None)
