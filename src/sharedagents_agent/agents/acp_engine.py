"""``AcpEngine`` — drive ANY ACP server as an ACP client.

ACP (the Agent Client Protocol) is a *generic* protocol, so this engine
is too: it spawns an arbitrary ACP server (configured per-agent) and
drives it over newline-delimited JSON-RPC stdio via
:mod:`sharedagents_acp.client`. Cursor's first-party ``cursor-agent acp`` is
just one configuration; Gemini, Zed agents, or any other ACP server work
the same way.

Per-agent config (``~/.sharedagents/agents/<name>/config.json``)::

    {
      "engine": "acp",
      "acp_command": ["cursor-agent", "acp"],   # the ACP server argv
      "acp_name": "cursor",                       # short label (logs/footer)
      "acp_auth_method": "cursor_login",          # authenticate methodId (optional)
      "acp_model": "composer-2.5"                 # idle footer label (optional)
    }

Unlike the ``*_CLI`` engines (claude/codex/antigravity) which shell out
to each tool's native print mode, this one talks ACP. The turn flow:
spawn → initialize → (authenticate) → session/new|load → session/prompt,
translating ``session/update`` notifications into :data:`AgentTurnEvent`.
The model is read from the ACP ``session/new`` response
(``models.currentModelId``) so the TUI footer reflects the real upstream
model for any agent — falling back to the configured ``acp_model``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from sharedagents_model.usage_pricing import metrics_from_provider_usage

from .engine import (
    AgentTurnEvent,
    ArtifactReference,
    Engine,
    EngineKind,
    EngineSessionState,
    HandoffAccepted,
    QueueMessageInjected,
    TextDelta,
    ToolCallResult,
    ToolCallStarted,
    TurnDone,
    TurnError,
    TurnRequest,
    TurnStarted,
    UserMessageInjected,
)

logger = logging.getLogger(__name__)

# JSON-RPC error code an ACP server returns when the client must
# authenticate first. Mirrors jsonrpc.JsonRpcErrorCode.AUTH_REQUIRED.
_AUTH_REQUIRED = -32000

# Bounded grace window for a graceful-first ACP abort —
# how long ``abort(force=False)`` waits for the cleanly-requested stop to reap
# the driven process before escalating to a hard kill.
ACP_GRACEFUL_ABORT_S: float = float(os.environ.get("SHAREDAGENTS_ACP_GRACEFUL_ABORT_S", "5") or "5")


async def _next_prompt_item(prompt_iter: Any) -> tuple[bool, Any]:
    """Read one prompt-stream item as ``(True, item)``; ``(False, None)`` at end.

    Run as a task so the turn can wait on the stream and on a Steer at the
    same time without cancelling the read (a cancelled read closes the
    stream before the peer has answered).
    """
    try:
        return True, await prompt_iter.__anext__()
    except StopAsyncIteration:
        return False, None


async def _drop_pending_read(task: asyncio.Task[Any] | None) -> None:
    """Cancel an outstanding stream read and wait until it has stopped.

    Waiting matters: the stream cannot be closed while a read is still
    running on it. A cancel aimed at the caller itself still propagates.
    """
    if task is None:
        return
    if not task.done():
        task.cancel()
    # Child cancellation is a discarded result; cancellation of this caller
    # still raises from gather, including on Python 3.10. Also retrieve any
    # exception from a read that finished before cleanup started.
    await asyncio.gather(task, return_exceptions=True)


async def _finish_prompt_for_followup(
    prompt_iter: Any,
    *,
    timeout_s: float,
    pending: asyncio.Task[Any] | None = None,
) -> None:
    """Wait until the prompt we just cancelled actually ends.

    A queued follow-up is the next ``session/prompt`` on the same ACP
    session. Closing the iterator and sending ``session/cancel`` in the
    same breath frees the client slot before the peer has answered the
    cancel, so the notification lands on the follow-up. The peer then
    returns ``stopReason=cancelled`` at once, the queue item is already
    marked injected, and the operator's text never runs.

    ``pending`` is a read already in flight (a Steer arrived while the
    turn was waiting on the stream); its item is consumed first.
    """

    async def _drain() -> None:
        if pending is not None:
            has_item, item = await pending
            if not has_item or getattr(item, "update", None) is None:
                return
        async for item in prompt_iter:
            if getattr(item, "update", None) is None:
                return

    try:
        await asyncio.wait_for(_drain(), timeout_s)
    except (asyncio.TimeoutError, StopAsyncIteration):
        pass
    except Exception:  # noqa: BLE001 — the follow-up must still be sent
        pass
    await _drop_pending_read(pending)
    try:
        await prompt_iter.aclose()
    except Exception:  # noqa: BLE001
        pass


# Recognized ACP ``sessionUpdate`` kinds ``_translate_update`` translates; any
# kind NOT here is an unknown/evolving wire event that must leave a trace
# rather than vanish.
_KNOWN_ACP_UPDATE_KINDS = frozenset({"agent_message_chunk", "agent_thought_chunk", "tool_call", "tool_call_update"})

SpawnFn = Callable[..., Awaitable[tuple[Any, Any]]]


async def _default_spawn(cmd: list[str], *, cwd: str | None, env: dict[str, str] | None):
    from sharedagents_acp.client import spawn_stdio_agent
    from sharedagents_agent.delegate_env import build_third_party_spawn_env

    # An AcpEngine ALWAYS drives a THIRD-PARTY
    # external ACP server (cursor / opencode / generic ACP) — never a local
    # engine and never sharedagents's own agent (that is the TUI AgentClient path).
    # So it never inherits the full parent os.environ: reduce to the env
    # allowlist and overlay the delegate's OWN per-agent creds (``env``, the
    # ACP analog of the CLI ``_spawn_env_overrides``), then pass the result
    # VERBATIM with inherit_parent_env=False.
    child_env = build_third_party_spawn_env(env or None)
    # Advertise the FULL client-callback surface (fs
    # read+write, terminal) so the engine can register handlers for each —
    # registering a handler for an un-advertised capability raises.
    from sharedagents_acp.client.callbacks import build_client_capabilities

    return await spawn_stdio_agent(
        cmd,
        cwd=cwd,
        env=child_env,
        inherit_parent_env=False,
        client_capabilities=build_client_capabilities(),
        client_info={"name": "sharedagents", "version": "1.0.0"},
        return_process=True,
    )


@dataclass(slots=True)
class _LiveTurn:
    client: Any
    proc: Any
    session_id: str = ""
    lease_id: str | None = ""
    stderr_tail: bytearray = field(default_factory=bytearray)
    stderr_task: asyncio.Task[None] | None = None
    # Set by :meth:`AcpEngine.steer_now`: deliver the queued follow-up now
    # instead of at the next completed tool call.
    steer: asyncio.Event = field(default_factory=asyncio.Event)
    # The stream read in flight, cancelled if the turn ends while waiting.
    pending_read: asyncio.Task[Any] | None = None


def _read_acp_config(cfg: Any) -> dict[str, Any]:
    """Read the ``acp_*`` settings from the agent's ``config.json``.

    Kept out of :class:`AgentConfig` (which is a fixed dataclass that
    drops unknown keys) so adding a new ACP agent needs no schema change
    — the engine reads its own config from the agent home directory.
    """
    out: dict[str, Any] = {
        "command": [],
        "name": "",
        "auth_method": "",
        "model": "",
        "context_window": 0,
    }
    home = getattr(cfg, "home", None)
    if home is None and cfg is not None:
        try:
            from sharedagents_agent.agent import agents_root

            home = agents_root() / getattr(cfg, "name", "")
        except Exception:  # noqa: BLE001
            home = None
    if home is None:
        return out
    try:
        data = json.loads((Path(home) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    if not isinstance(data, dict):
        return out
    cmd = data.get("acp_command")
    if isinstance(cmd, list):
        out["command"] = [str(x) for x in cmd]
    elif isinstance(cmd, str) and cmd.strip():
        out["command"] = cmd.split()
    out["name"] = str(data.get("acp_name") or "")
    out["auth_method"] = str(data.get("acp_auth_method") or "")
    out["model"] = str(data.get("acp_model") or "")
    try:
        out["context_window"] = int(data.get("acp_context_window") or 0)
    except (TypeError, ValueError):
        out["context_window"] = 0
    return out


def _parse_model_context(model_id: str) -> int:
    """Extract a context window from an ACP model id or display label that encodes it.

    Cursor encodes it for some models, e.g. ``gpt-5.5[context=272k]`` →
    272000, ``claude-opus-4-7[...context=300k...]`` → 300000,
    ``claude-opus-4-8[context=1m]`` → 1000000. Also parses explicit
    ``1M`` / ``1m`` markers or known model families (e.g. Gemini → 1M).
    Returns 0 when absent/unrecognized (e.g. ``composer-2.5[fast=true]``
    has no context field)."""
    import re

    if not model_id:
        return 0
    m = re.search(r"context=(\d+)\s*([kmKM]?)", model_id)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit == "m":
            return n * 1_000_000
        if unit == "k":
            return n * 1000
        return n
    if re.search(r"\b1[mM]\b", model_id):
        return 1_000_000
    if re.search(r"\b300[kK]\b", model_id):
        return 300_000
    if re.search(r"\b272[kK]\b", model_id):
        return 272_000
    if re.search(r"\b200[kK]\b", model_id):
        return 200_000
    if re.search(r"\b128[kK]\b", model_id):
        return 128_000
    if "gemini" in model_id.lower():
        return 1_000_000
    return 0


class AcpEngine(Engine):
    """Run a turn by driving a configured ACP server as an ACP client."""

    kind = EngineKind.ACP
    install_url = "https://agentclientprotocol.com"

    def __init__(
        self,
        cfg: Any = None,
        *,
        engine_kind: EngineKind | None = None,
        command: list[str] | None = None,
        name: str | None = None,
        auth_method: str | None = None,
        model: str | None = None,
        spawn_fn: SpawnFn | None = None,
    ) -> None:
        acp = _read_acp_config(cfg)
        selected_kind = engine_kind or self.kind
        self.kind = selected_kind
        spec = None
        if selected_kind is not EngineKind.ACP:
            from sharedagents_agent.external_registry import spec_for

            spec = spec_for(selected_kind)
        default_command = list(spec.default_acp_command or ()) if spec else []
        self._command: list[str] = list(command or acp["command"] or default_command)
        self._name: str = name or acp["name"] or (spec.short_name if spec else "acp")
        self._auth_method: str = auth_method or acp["auth_method"] or (spec.default_auth if spec else "") or ""
        self._model: str = model or acp["model"] or (spec.default_model if spec else "") or ""
        self._config_window: int = int(acp["context_window"] or 0)
        self._default_window: int = int((spec.default_context_window if spec else 0) or 0)
        self._mode_map: dict[str, str] = dict(spec.mode_map or {}) if spec else {}
        self._spawn: SpawnFn = spawn_fn or _default_spawn
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._active: dict[str, _LiveTurn] = {}
        # Most recent model the ACP server reported (session/new). Cached
        # so the footer reflects the real model after the first turn.
        self._cached_model: str = ""
        # Context window parsed from the model id Cursor reports
        # (``...[context=272k]``); 0 when the model doesn't encode it.
        self._cached_context_window: int = 0
        # Per-session running char count for a context-size ESTIMATE.
        # ACP servers (Cursor included) don't report token usage on the
        # wire, and the conversation history lives server-side where the
        # client can't see it — so we accumulate the chars we send + the
        # chars we receive each turn and convert to an approximate token
        # count for the TUI's ``ctx`` meter (flagged estimated).
        self._ctx_chars: dict[str, int] = {}
        self._discovered_modes: list[dict[str, str]] = []
        self._cfg = cfg

    @property
    def name(self) -> str:
        return self._name

    # ---- agent-specific hooks (overridden by subclasses) ------------

    def _acp_mode(self, mode: str) -> str:
        """Pass the picked native id through to ACP (catalogue identity)."""
        from sharedagents_agent.acp_mode_dict import native_id_for_mode

        wire = native_id_for_mode(
            self.kind.value,
            mode,
            discovered=self._discovered_modes,
        )
        if wire:
            return wire
        return self._mode_map.get(mode, mode)

    def _clean_model_label(self, label: str) -> str:
        """Normalize the model label for display. Base is a no-op;
        subclasses strip vendor-specific encodings (e.g. Cursor's
        ``composer-2.5[fast=true]`` suffix)."""
        if self.kind is EngineKind.CURSOR_ACP:
            cleaned = re.sub(r"\[[^\]]*\]\s*$", "", label).strip()
            return cleaned or label
        return label

    def display_model(self) -> str:
        # Prefer the model the ACP server actually reported; fall back to
        # the configured label, then the engine name. Generic — any ACP
        # agent that reports ``models.currentModelId`` gets a live label.
        if self._cached_model or self._model:
            return self._cached_model or self._model
        if self.kind is EngineKind.CURSOR_ACP:
            try:
                from sharedagents_agent.discovery.acp_list import (
                    discover_cursor_current_model,
                )

                discovered = discover_cursor_current_model(command=self._command)
                if discovered:
                    self._cached_model = self._clean_model_label(discovered)
                    return self._cached_model
            except Exception:  # noqa: BLE001
                pass
        return self._name

    def context_window(self, model: str | None = None) -> int:
        """Best-known context window (denominator of the ctx meter).

        Prefers the window parsed from the model id Cursor reports
        (``...[context=272k]``), then the configured ``acp_context_window``.
        0 means "unknown" — the agent layer falls back to the global
        default. Subclasses set a sensible per-agent default (Cursor's
        composer doesn't advertise a window, so it returns 200000).

        ``model`` is accepted for call-site parity with
        :meth:`ExternalCliEngine.context_window`; ACP engines already
        resolve their window from their own model state, so it is unused.
        """
        if self._cached_context_window:
            return self._cached_context_window
        if self._config_window:
            return self._config_window
        if self.kind is EngineKind.CURSOR_ACP:
            model_name = (self._cached_model or self._model or "").lower()
            if "composer" in model_name:
                return 200_000
        return self._default_window

    def available_models(self) -> list[dict[str, Any]] | None:
        """Discover models through the external-agent registry."""
        if self.kind is EngineKind.ACP:
            return None
        from sharedagents_agent.discovery import discover_models

        return discover_models(self.kind, command=self._command)

    # ---- per-session model selection (mirrors external_cli) ---------
    #
    # The ``/models`` picker resolves a row by alias/id and calls
    # ``set_config_option(sid, "model", <id>)``. We stash the wire model
    # id per sharedagents session; :meth:`run_turn` forwards it to the ACP
    # server via ``session/set_config_option`` before the prompt.

    def _session_models_map(self) -> dict[str, str]:
        m = getattr(self, "_session_models", None)
        if m is None:
            m = {}
            self._session_models = m  # type: ignore[attr-defined]
        return m

    def session_model(self, sharedagents_sid: str) -> str:
        """Wire model id selected for ``sharedagents_sid``, or ``""`` (ACP default)."""
        return self._session_models_map().get(sharedagents_sid, "")

    def _resolve_model_value(self, value: str) -> str:
        v = (value or "").strip()
        if not v:
            return ""
        try:
            rows = self.available_models() or []
        except Exception:  # noqa: BLE001
            rows = []
        vlow = v.lower()
        for row in rows:
            if str(row.get("alias", "")).lower() == vlow:
                return str(row.get("id") or v)
        for row in rows:
            if str(row.get("id", "")).lower() == vlow:
                return str(row.get("id") or v)
        for row in rows:
            if str(row.get("display", "")).lower() == vlow:
                return str(row.get("id") or v)
        return v  # pass-through — the ACP server is the authority on raw ids

    def _resolve_live_model_value(self, client: Any, value: str) -> str:
        """Resolve a requested id against this concrete ACP session's catalogue.

        Out-of-band discovery is only picker intent. The live ``session/new``
        response is authoritative because an ACP server may use different wire
        ids or expose a session-specific subset.
        """
        if self.kind is EngineKind.KIMI_ACP:
            value = {
                "kimi-k2-thinking-turbo": "moonshot-ai/kimi-k3",
                "kimi-k2-thinking": "moonshot-ai/kimi-k3",
                "kimi-k2": "moonshot-ai/kimi-k3",
                "kimi-k3": "moonshot-ai/kimi-k3",
            }.get((value or "").strip().lower(), value)
        models = getattr(client, "last_session_models", None)
        rows = models.get("availableModels") if isinstance(models, dict) else None
        if not isinstance(rows, list):
            return value
        wanted = (value or "").strip().lower()
        for row in rows:
            if not isinstance(row, dict):
                continue
            wire_id = str(row.get("modelId") or "")
            name = str(row.get("name") or "")
            if wanted in {wire_id.lower(), name.lower()}:
                return wire_id or value
            # Catalog ids are often ``provider/model``; picker/config may
            # store the bare model token (``kimi-k3``).
            if wire_id.lower().endswith("/" + wanted) or name.lower().endswith("/" + wanted):
                return wire_id or value
        if self.kind is EngineKind.CURSOR_ACP:
            requested_label = value
            try:
                for row in self.available_models() or []:
                    if str(row.get("alias") or "").lower() == wanted:
                        requested_label = str(row.get("display") or value)
                        break
            except Exception:  # noqa: BLE001
                pass

            def family(label: str) -> str:
                text = re.sub(r"\[[^\]]*\]", "", label.lower())
                text = re.sub(
                    r"\b(cursor|claude|1m|thinking|extra|xhigh|max|high|medium|low|fast)\b",
                    " ",
                    text,
                )
                return re.sub(r"[^a-z0-9]+", "", text)

            family_name = family(requested_label)
            matches = [
                str(row.get("modelId") or "")
                for row in rows
                if isinstance(row, dict)
                and family_name
                and family(str(row.get("name") or row.get("modelId") or "")) == family_name
            ]
            if len(matches) == 1 and matches[0]:
                return matches[0]
        return value

    async def _apply_session_model(
        self,
        client: Any,
        session_id: str,
        wire_model: str,
    ) -> Any:
        """Push the wire model via the method this ACP server actually honors.

        Hermes (and others that advertise a ``models`` block on
        ``session/new``) switch with ``session/set_model``. Kimi-style
        servers expose a ``model`` config option and honor
        ``session/set_config_option``. Hermes' set-config handler stores
        the value and does not change the session model.
        """
        models = getattr(client, "last_session_models", None)
        if isinstance(models, dict) and "availableModels" in models:
            return await client.session_set_model(session_id, wire_model)
        return await client.session_set_config_option(
            session_id,
            "model",
            wire_model,
        )

    @staticmethod
    def _acknowledged_model(result: Any, requested: str) -> str:
        """Return the model id confirmed by a set-config response.

        Servers that return the schema-shaped ``configOptions`` state must echo
        the requested model as ``currentValue``. Older peers returning an empty
        success object still provide a correlated JSON-RPC acknowledgement.
        """
        if not isinstance(result, Mapping):
            return requested
        options = result.get("configOptions")
        if not isinstance(options, list):
            return requested
        for option in options:
            if not isinstance(option, Mapping) or option.get("id") != "model":
                continue
            current = str(option.get("currentValue") or "")
            if not current:
                raise RuntimeError("ACP model acknowledgement omitted currentValue")
            if current != requested:
                raise RuntimeError(
                    f"ACP model acknowledgement mismatch: requested {requested!r}, effective {current!r}"
                )
            return current
        raise RuntimeError("ACP model acknowledgement omitted the model option")

    def set_config_option(self, sharedagents_sid: str, option_id: str, value: Any) -> None:
        if option_id != "model":
            return
        raw = str(value or "").strip()
        store = self._session_models_map()
        if not raw or raw.lower() in ("auto", "default", "reset", "none"):
            store.pop(sharedagents_sid, None)
            return
        store[sharedagents_sid] = self._resolve_model_value(raw)

    # ---- session sidecar I/O ----------------------------------------

    def state_path(self, sessions_dir: Path, sharedagents_sid: str) -> Path:
        return Path(sessions_dir) / f"{sharedagents_sid}.{self.kind.value}.json"

    def read_state(self, sessions_dir: Path, sharedagents_sid: str) -> EngineSessionState:
        state = EngineSessionState(sharedagents_session_id=sharedagents_sid, engine=self.kind)
        path = self.state_path(Path(sessions_dir), sharedagents_sid)
        if not path.exists():
            # Migrate the pre-generic ``cursor_acp`` sidecar name.
            legacy = Path(sessions_dir) / f"{sharedagents_sid}.cursor_acp.json"
            path = legacy if legacy.exists() else path
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return state
        if not isinstance(data, dict):
            return state
        eid = data.get("engine_session_id") or data.get("cursor_session_id") or ""
        state.engine_session_id = str(eid or "")
        state.priming_status = str(data.get("priming_status") or ("completed" if eid else "never"))
        state.priming_system_prompt_sha256 = str(data.get("priming_system_prompt_sha256") or "")
        state.last_used_at = str(data.get("last_used_at") or "")
        state.last_error = str(data.get("last_error") or "")
        state.spawn_cwd = str(data.get("spawn_cwd") or "")
        # Restore the running char-occupancy estimate across process
        # restart. Without this, the next turn's prompt_tokens collapses
        # to just that turn's sent+recv (operator saw 15889 → 580 after
        # a deploy restart; WebUI footer then read 580/200K).
        try:
            cc = int(data.get("ctx_chars") or 0)
        except (TypeError, ValueError):
            cc = 0
        if cc > int(self._ctx_chars.get(sharedagents_sid, 0) or 0):
            self._ctx_chars[sharedagents_sid] = cc
        return state

    def write_state(self, sessions_dir: Path, state: EngineSessionState) -> None:
        path = self.state_path(Path(sessions_dir), state.sharedagents_session_id)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "sharedagents_session_id": state.sharedagents_session_id,
                    "engine": state.engine.value,
                    "engine_session_id": state.engine_session_id,
                    "priming_status": state.priming_status,
                    "priming_system_prompt_sha256": state.priming_system_prompt_sha256,
                    "last_used_at": state.last_used_at,
                    "last_error": state.last_error,
                    "spawn_cwd": state.spawn_cwd,
                    "ctx_chars": int(self._ctx_chars.get(state.sharedagents_session_id, 0) or 0),
                },
                ensure_ascii=False,
            )
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, payload.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)
        except OSError:
            logger.debug("acp write_state failed for %s", path, exc_info=True)
            try:
                tmp.unlink()
            except OSError:
                pass

    # ---- introspection ----------------------------------------------

    def _resolved_cli(self) -> str | None:
        """Resolve the ACP server binary (``acp_command[0]``).

        Delegates to :func:`sharedagents_agent.tools.runners._which_cli`, the
        single source of truth for "is this CLI launchable": it tries the
        inherited ``PATH`` first, then the well-known user bin dirs
        (``~/.local/bin``, ``~/.opencode/bin``, ``~/.npm-global/bin``, …).
        This matters because a detached daemon (systemd / nohup) frequently
        starts with a minimal ``PATH`` that omits the dir an ACP CLI
        (``opencode``, ``cursor-agent``) was installed into — so a bare
        ``shutil.which`` would spuriously report it "not on PATH" even
        though the operator installed it. An absolute ``acp_command[0]``
        passes through unchanged."""
        if not self._command:
            return None
        from sharedagents_agent.tools.runners import _which_cli

        return _which_cli(self._command[0])

    def is_available(self) -> tuple[bool, str]:
        if not self._command:
            return (False, "no acp_command configured for this ACP agent")
        if self._resolved_cli() is None:
            return (
                False,
                f"`{self._command[0]}` (acp_command) not found on PATH or "
                f"the well-known CLI bin dirs (~/.local/bin, ~/.opencode/bin, "
                f"…). See {self.install_url}.",
            )
        return (True, "")

    # ---- system-prompt policy ---------------------------------------

    def system_prompt_policy(self, state: EngineSessionState, *, current_system_prompt_sha256: str) -> str:
        if state.priming_status in {"never", "in_progress", "failed"}:
            return "first_turn"
        if current_system_prompt_sha256 and state.priming_system_prompt_sha256 != current_system_prompt_sha256:
            return "sha_changed"
        return "never"

    # ---- steer ------------------------------------------------------

    def steer_now(self, turn_id: str) -> bool:
        """Deliver the queued follow-up to the running turn now.

        The turn normally takes queued text at the next completed tool
        call. Steer does not wait for that: the turn drains the queue,
        cancels the in-flight ``session/prompt``, reads it to its
        PromptResponse, and sends the drained text as the next prompt of
        the same turn. Same path for every ACP agent. Returns ``False``
        when ``turn_id`` is not running here.
        """
        live = self._active.get(turn_id)
        if live is None:
            return False
        live.steer.set()
        return True

    # ---- abort ------------------------------------------------------

    async def abort(self, turn_id: str, *, force: bool = False) -> None:
        """Stop the in-flight ACP turn.

                ``force=False`` (first press) is a genuine GRACEFUL-FIRST tier
        : request a clean ``session/cancel`` and wait up to
                :data:`ACP_GRACEFUL_ABORT_S` for the driven process to exit before
                escalating to a hard kill — never kill in the same call that requests
                the clean stop. ``force=True`` (second press) hard-kills immediately
                after a best-effort cancel.
        """
        live = self._active.get(turn_id)
        if live is None:
            return

        import signal as _signal

        from sharedagents_agent import sandbox_broker as _sb

        def _hard_kill() -> None:
            # Route through the capability layer first (spec §10.3,
            # downward-only); ``_kill_proc`` stays the guaranteed mechanism so
            # the kill never regresses if the lease is absent or the group kill
            # doesn't apply.
            _sb.capability_kill_proc(
                live.lease_id,
                sig=(_signal.SIGKILL if force else _signal.SIGTERM),
            )
            _kill_proc(live.proc, force=force)

        if force:
            if live.session_id:
                try:
                    await live.client.session_cancel(live.session_id, force=True)
                except Exception:  # noqa: BLE001
                    logger.debug("acp session_cancel failed", exc_info=True)
            _hard_kill()
            return

        # --- graceful-first tier -------------------------------------------
        if live.session_id:
            try:
                await live.client.session_cancel(live.session_id, force=False)
            except Exception:  # noqa: BLE001
                logger.debug("acp session_cancel failed", exc_info=True)
        # If the clean stop already reaped the process, do NOT hard-kill.
        if getattr(live.proc, "returncode", None) is not None:
            return
        # Otherwise wait a bounded grace window for a clean exit; only escalate
        # to a hard kill if the peer is still alive when it elapses.
        waiter = getattr(live.proc, "wait", None)
        if callable(waiter):
            try:
                await asyncio.wait_for(waiter(), ACP_GRACEFUL_ABORT_S)
                return
            except asyncio.TimeoutError:
                pass
            except Exception:  # noqa: BLE001
                pass
        _hard_kill()

    # ---- turn dispatch ----------------------------------------------

    async def run_turn(
        self,
        request: TurnRequest,
        state: EngineSessionState,
        *,
        sessions_dir: Path,
        turn_id: str,
        env: dict[str, str] | None = None,
    ) -> AsyncIterator[AgentTurnEvent]:
        yield TurnStarted(
            turn_id=turn_id,
            sharedagents_session_id=state.sharedagents_session_id,
            engine_session_id=state.engine_session_id,
            resumed=bool(state.engine_session_id),
        )

        if not self._command:
            yield TurnError(message="no acp_command configured", classification="config")
            yield TurnDone(state="error", engine_session_id=state.engine_session_id)
            return

        manual_budget_exhausted = bool(
            request.extra.get("_budget_exhausted_manual") if isinstance(request.extra, dict) else False
        )
        manual_budget_notice = (
            str(request.extra.get("_manual_budget_notice") or "").strip() if isinstance(request.extra, dict) else ""
        )
        effective_request_mode = "plan" if manual_budget_exhausted else request.mode

        # A resume must reuse the cwd that MINTED the
        # engine session id, never a later/different requested cwd — the ACP
        # server keys its on-disk session/load by cwd, so a mismatch silently
        # misses the saved session and drops the prior conversation. Whenever a
        # resumable session id + its recorded spawn cwd exist, that recorded
        # cwd wins over ``request.cwd``.
        cwd = state.spawn_cwd if state.engine_session_id and state.spawn_cwd else request.cwd or state.spawn_cwd or None
        if not cwd or not os.path.isabs(str(cwd)):
            # Web/chat turns may omit cwd, or pass a relative one. Pin the
            # actual directory for both callbacks and session/new, and persist
            # it for later session/load. Applies to every ACP kind: the wire
            # schema requires an ABSOLUTE cwd (opencode rejects "." with
            # -32602), and the spawn inherits this same directory when cwd is
            # None — resolving keeps the two identical and dead-cwd-safe.
            from sharedagents_core.cwdutil import safe_getcwd

            base = safe_getcwd()
            cwd = os.path.normpath(os.path.join(base, str(cwd))) if cwd else base

        # --- Screen pre-prompt METADATA before it egresses ---
        # `cwd` reaches the third-party ACP server via the spawn working-dir
        # (self._spawn below) AND session/new (self._open_session), and `mode`
        # via session/set_mode — all BEFORE the composed-prompt sink guard
        # further down. A secret-shaped cwd/mode would therefore reach the ACP
        # client unscreened (the prompt guard runs too late to cover metadata).
        # Screen them here through the SAME shared guard, ordered ahead of the
        # spawn / session-config / set-mode calls. Hard-block per
        # No override, fail CLOSED (abort the turn, never
        # spawn). No-op for local engines; the base64 data: URI exemption
        # is inherited via screen_payload/scrub_text.
        from sharedagents_agent import delegation_gate as _dgate_meta

        try:
            if cwd:
                _dgate_meta.guard_egress_payload(self.kind, str(cwd))
            _meta_mode = self._acp_mode(effective_request_mode) if effective_request_mode else ""
            if _meta_mode:
                _dgate_meta.guard_egress_payload(self.kind, str(_meta_mode))
        except _dgate_meta.SecretEgressBlocked as _seb_meta:
            try:
                from sharedagents_agent.tools._log import log_event as _log_event

                _log_event(
                    "DELEGATE_SECRET_EGRESS_BLOCKED",
                    engine=self.kind.value,
                    sharedagents_session_id=state.sharedagents_session_id,
                    findings=list(_seb_meta.findings),
                    turn_id=turn_id,
                    phase="pre_prompt_metadata",
                )
            except Exception:  # noqa: BLE001
                pass
            yield TurnError(
                message=_seb_meta.reason,
                classification="secret_egress_blocked",
            )
            yield TurnDone(state="error", engine_session_id=state.engine_session_id)
            return

        cmd = [self._resolved_cli() or self._command[0], *self._command[1:]]
        # Fold the per-agent provider credentials into
        # the spawn env overrides — the ACP analog of the CLI path's
        # ``_spawn_env_overrides`` (external_cli marshals ``request.extra
        # ["api_keys"]`` there). Since the third-party spawn drops the parent
        # os.environ down to the allowlist, a delegate that authenticates via a
        # provider env var (e.g. opencode reading a key) must receive its OWN
        # per-agent key here or delegation would break. Caller-supplied ``env``
        # wins any collision. These become the ``overrides`` in _default_spawn's
        # build_third_party_spawn_env — always included, never the infra env.
        spawn_env: dict[str, str] = {}
        try:
            _ak = request.extra.get("api_keys") if isinstance(request.extra, dict) else None
            if isinstance(_ak, dict):
                spawn_env.update({str(k): str(v) for k, v in _ak.items()})
        except Exception:  # noqa: BLE001
            pass
        if env:
            spawn_env.update(env)
        try:
            client, proc = await self._spawn(cmd, cwd=cwd, env=(spawn_env or None))
        except FileNotFoundError:
            yield TurnError(
                message=(
                    f"`{self._command[0]}` not found on PATH or the well-known "
                    f"CLI bin dirs (~/.local/bin, ~/.opencode/bin, …) "
                    f"({self.install_url})"
                ),
                classification="spawn_failed",
            )
            yield TurnDone(state="error", engine_session_id=state.engine_session_id)
            return

        # Lease the spawned ACP server agent (§10.4) so its cancel is in the
        # capability model. pgid is left unset (the spawn's group leadership is
        # not guaranteed) → the capability kill validates lineage then defers
        # to _kill_proc, never mis-targeting a process group. This TOOL lease
        # is uncounted and does not make ``session_live(sid)`` true — the
        # durable engine pidfile below is what the stale-live sweep trusts.
        from sharedagents_agent import sandbox_broker as _sb

        _acp_lease = _sb.lease_process(pgid=None, kind="acp")
        # Durable pidfile — same backstop CLI engines already write. Without
        # it, ``finalize_stale_live_sessions`` folds a still-running Cursor
        # ACP turn after 300s of no JSONL (long bash / headed browser) with
        # ``stale_live_reconcile``, the UI paints ABORTED, and follow-ups
        # queue behind the leftover ACP process until its 900s idle timeout.
        #
        # ``pgid=None`` is mandatory: ACP spawn does not ``start_new_session``,
        # so the child shares the gateway PGID. Auto-pgid + later
        # ``kill_session_engine`` would ``killpg`` the gateway.
        # Descendants + the exact pid are still reaped.
        try:
            from . import engine_pidfile

            _acp_pid = getattr(proc, "pid", None)
            if _acp_pid is not None and state.sharedagents_session_id:
                engine_pidfile.record(
                    sessions_dir,
                    str(state.sharedagents_session_id),
                    int(_acp_pid),
                    cli=self._name or "acp",
                    pgid=None,
                )
        except Exception:  # noqa: BLE001 — sidecar is best-effort
            pass
        live = _LiveTurn(
            client=client,
            proc=proc,
            session_id=state.engine_session_id,
            lease_id=_acp_lease,
        )
        stderr = getattr(proc, "stderr", None)
        if stderr is not None and callable(getattr(stderr, "read", None)):
            live.stderr_task = asyncio.create_task(
                _drain_stderr_tail(stderr, live.stderr_tail),
                name=f"acp-stderr-{turn_id}",
            )
        self._active[turn_id] = live
        # Register the FULL advertised CLIENT_METHODS surface
        # — fs read/write + the five terminal/* callbacks + request_permission —
        # so an external agent's fs/terminal callbacks are handled according to
        # the workspace policy instead of hitting a bare methodNotFound. Each
        # register_* is capability-gated by the caps advertised at spawn.
        from sharedagents_acp.client.callbacks import (
            make_fs_read_handler,
            make_fs_write_handler,
            make_terminal_handlers,
        )
        from sharedagents_agent.delegate_env import build_third_party_spawn_env

        # Each register_* is capability-gated: the AcpClient raises ValueError
        # if the matching capability was NOT advertised at spawn. Register only
        # what was advertised — a ValueError means this client
        # opted out of that surface, so skip it rather than fail the turn.
        def _try_register(reg_name: str, arg: Any) -> None:
            reg = getattr(client, reg_name, None)
            if not callable(reg):
                return
            try:
                reg(arg)
            except ValueError:
                pass

        # Only Devin's allow-all mode opts out of workspace confinement. Ask,
        # plan (including the manual budget gate), and other engines retain it.
        confine_workspace = not (
            self.kind is EngineKind.DEVIN_ACP
            and str(effective_request_mode or "").strip().lower() in {"auto", "bypass"}
        )
        _try_register(
            "register_fs_read_text_file",
            make_fs_read_handler(cwd, confine=confine_workspace),
        )
        _try_register(
            "register_fs_write_text_file",
            make_fs_write_handler(cwd, confine=confine_workspace),
        )
        _try_register(
            "register_terminal",
            make_terminal_handlers(
                cwd,
                build_third_party_spawn_env(spawn_env or None),
                confine=confine_workspace,
            ),
        )
        client.register_request_permission(_make_permission_handler(effective_request_mode))

        try:
            init = await client.initialize()
            policy = self.system_prompt_policy(state, current_system_prompt_sha256=request.system_prompt_sha256)
            prior_eid = state.engine_session_id
            prior_status = state.priming_status
            session_id = await self._open_session(client, init, state, policy, cwd)
            live.session_id = session_id
            self._capture_model(client)
            self._capture_modes(client)

            state.engine_session_id = session_id
            # Only mark "in_progress" when a NEW engine session was opened
            # (it still needs priming). A successful LOAD of an already-primed
            # session keeps "completed" on disk so a deploy-restart / kill
            # mid-turn doesn't poison the sidecar — otherwise the next turn
            # reads "in_progress", opens a fresh engine session, and silently
            # drops conversation context.
            resumed = bool(prior_eid) and session_id == prior_eid and prior_status == "completed"
            if not resumed:
                state.priming_status = "in_progress"
            state.spawn_cwd = cwd or ""
            self.write_state(sessions_dir, state)

            if effective_request_mode:
                try:
                    await self._apply_mode(
                        client,
                        session_id,
                        effective_request_mode,
                    )
                    request.extra["_native_mode_evidenced"] = effective_request_mode
                except Exception:  # noqa: BLE001
                    # A rejected/unsupported permission-mode
                    # set is a security-relevant silent failure — the operator
                    # believes plan/ask is enforced when it is not. Surface it
                    # ABOVE debug level (before the prompt streams) instead of
                    # swallowing it at debug.
                    logger.warning(
                        "acp mode application (%s) rejected by the remote ACP "
                        "server (%s) — plan/ask enforcement may NOT be active "
                        "for this turn; the client-side permission handler "
                        "still applies.",
                        effective_request_mode,
                        self._name,
                        exc_info=True,
                    )
                    if request.extra.get("_native_readiness_probe") or self._mode_failure_is_terminal():
                        raise

            # Resolve the per-session model but do NOT send it to the ACP
            # server yet: ``model`` is model-controllable (``Agent(model=...)``
            # / the ``/models`` picker) and ``session_set_config_option`` ships
            # it to the third-party ACP server OUTSIDE the prompt payload — the
            # exact argv/model leak Codex flagged on the CLI path, present here
            # too. It must clear the SAME egress screen as
            # the prompt before it leaves the process, so the actual
            # set_config_option call is deferred to AFTER the guard below.
            #
            # Custom agents pin a default via ``acp_model`` in config.json.
            # ``auto`` / ``default`` / empty means "leave the ACP server's
            # own default" (e.g. Cursor Auto) — do not call set_config.
            selected_model = self.session_model(state.sharedagents_session_id)
            if not selected_model:
                pinned = (self._model or "").strip()
                if pinned and pinned.lower() not in (
                    "auto",
                    "default",
                    "reset",
                    "none",
                ):
                    selected_model = pinned

            # Label guard (owner round-2 scope change): the engine's display
            # name ("cursor" / "opencode" / "kimi" / "acp" — what
            # ``display_model()`` degrades to before the ACP server reports
            # a model) is NEVER a wire model id. A label can arrive here via
            # session-stamp inheritance; sending it would be a set-config
            # option the server must reject. Fail CLOSED with the same
            # classification the CLI gate uses, BEFORE the wire.
            from sharedagents_agent.discovery._common import CLI_ENGINE_LABELS

            if selected_model and selected_model.lower() in CLI_ENGINE_LABELS:
                yield TurnError(
                    message=(
                        f"{self._name} delegate: model '{selected_model}' is"
                        " the engine label, never a wire model id. It was"
                        " inherited from a session stamp; remove the override"
                        " or set acp_model to a real model id and retry."
                    ),
                    classification="model_not_allowed",
                )
                yield TurnDone(
                    state="error",
                    engine_session_id=state.engine_session_id,
                )
                return

            blocks = _build_prompt_blocks(request, policy)
            if manual_budget_notice:
                # On a resumed ACP session there is no system channel to
                # replace. Add the framework notice to the prompt exactly once
                # so the model still sees the spend/cap figures this turn.
                block_text = "\n".join(str(b.get("text") or "") for b in blocks if isinstance(b, dict))
                if manual_budget_notice not in block_text:
                    blocks.insert(
                        0,
                        {
                            "type": "text",
                            "text": (
                                "<sharedagents-manual-budget-notice>\n"
                                f"{manual_budget_notice}\n"
                                "</sharedagents-manual-budget-notice>\n\n"
                            ),
                        },
                    )
            # Operator requirement: the ACP engines (cursor / opencode /
            # kimi) treat the FIRST reply to a session/prompt as the END of
            # the turn — sharedagents does not poll the session afterwards, so an
            # acknowledgment-only "started" reply (or a hand-off to the
            # agent's own background subprocess / thread / cloud agent) is a
            # LOST result. Ask the agent in the prompt itself to respond
            # only with the final result. Appended here — the single shared
            # choke point every external-ACP turn passes through (fresh
            # prime AND resume) — so no ACP engine can be dispatched without
            # it. Native local_llm agents never pass through AcpEngine, so
            # their prompts stay byte-identical. The appended text rides the
            # composed payload through the secret egress screen below.
            blocks = _outbound_prompt_blocks(request, blocks)
            if not priming_includes_system_prompt(blocks, request, policy):
                raise RuntimeError("priming omitted the agent system prompt; prompt was not sent")
            # --- Egress SINK guard (hard block, no override) -----------------
            # The pre-flight delegation gate screens only the CALL ARGS upstream
            # in delegate_shims. The payload streamed to the ACP server is
            # composed HERE: on priming turns _build_prompt_blocks folds the
            # system prompt + preamble (child AGENTS.md / MEMORY / pinned facts /
            # todos / skills / memory-recall) into a leading <system> block
            # alongside the user text — PLUS the model-controllable ``model``
            # override, which otherwise reaches the ACP server as a
            # session-config field outside the prompt. Screen that fully-
            # composed string at the true third-party egress — the same shared
            # helper the CLI path uses — so a secret in ANY context source, and
            # in the model field, is caught, not just the top-level args
            # (closes the "wrong-place scrub" gap AND the model-field gap).
            # Hard-block: abort the turn BEFORE either
            # session_set_config_option(model) or session_prompt sends anything
            # to the server; do not redact-and-send. No-op for local engines;
            # the base64 data: URI exemption is inherited.
            from sharedagents_agent import delegation_gate as _dgate

            try:
                _dgate.guard_egress_payload(
                    self.kind,
                    "\n\n".join(
                        [b.get("text", "") for b in blocks if isinstance(b, dict)]
                        + ([selected_model] if selected_model else [])
                    ),
                )
            except _dgate.SecretEgressBlocked as _seb:
                try:
                    from sharedagents_agent.tools._log import log_event as _log_event

                    _log_event(
                        "DELEGATE_SECRET_EGRESS_BLOCKED",
                        engine=self.kind.value,
                        sharedagents_session_id=state.sharedagents_session_id,
                        findings=list(_seb.findings),
                        turn_id=turn_id,
                    )
                except Exception:  # noqa: BLE001
                    pass
                yield TurnError(
                    message=_seb.reason,
                    classification="secret_egress_blocked",
                )
                yield TurnDone(
                    state="error",
                    engine_session_id=state.engine_session_id,
                )
                return

            # Guard passed → NOW it is safe to push the (screened) model to the
            # ACP server, immediately before the prompt streams. Deferred from
            # above so a secret-bearing model never leaves the process
            if selected_model:
                try:
                    wire_model = self._resolve_live_model_value(
                        client,
                        selected_model,
                    )
                    ack = await self._apply_session_model(
                        client,
                        session_id,
                        wire_model,
                    )
                    effective_model = self._acknowledged_model(ack, wire_model)
                except Exception as exc:  # noqa: BLE001
                    # An explicit selection is a hard precondition for this
                    # physical turn. Never prompt on the previous/default model
                    # after rejection or transport loss.
                    message = (
                        f"{self._name} could not apply requested model {selected_model!r}; prompt was not sent: {exc}"
                    )
                    diagnostic = _child_failure_diagnostic(live)
                    if diagnostic:
                        message = f"{message} ({diagnostic})"
                    yield TurnError(
                        message=message,
                        classification="model_switch_failed",
                    )
                    yield TurnDone(
                        state="error",
                        engine_session_id=state.engine_session_id,
                    )
                    return
                # Only an acknowledged wire id may become effective-model
                # metadata. Requested picker intent is not execution evidence.
                self._cached_model = self._clean_model_label(effective_model)
                parsed_window = _parse_model_context(effective_model)
                if not parsed_window:
                    models = getattr(client, "last_session_models", None)
                    if isinstance(models, dict):
                        for m in models.get("availableModels") or []:
                            if isinstance(m, dict) and m.get("modelId") == effective_model:
                                nm = m.get("name")
                                if isinstance(nm, str) and nm:
                                    parsed_window = _parse_model_context(nm)
                                break
                if parsed_window:
                    self._cached_context_window = parsed_window
                request.extra["_native_model_evidenced"] = effective_model

            from sharedagents_agent.continuation import (
                MAX_IDLE_EMPTY_CONTINUES,
                TurnEventTracker,
                apply_nudge_hooks,
                idle_empty_continue_notice,
            )
            from sharedagents_agent.hooks import fire_hook_event

            prompt_blocks = blocks
            sent_chars = sum(len(b.get("text", "")) for b in prompt_blocks if isinstance(b, dict))
            recv_chars = 0
            real_used: int | None = None
            real_size: int | None = None
            stop_reason = "end_turn"
            # Bound the prompt stream by ACP first-output / idle timeouts.
            # usage_update leaves the startup window (peer is alive) but
            # must NOT reset last_output_at — a usage heartbeat must not
            # defeat idle. Visible content / tools reset the idle clock.
            turn_start = time.monotonic()
            first_token_at: float | None = None
            last_token_at: float | None = None
            delta_count = 0

            # Telegram watchdog is 540s (AGY print 480 + 60). Reset the
            # idle clock only on events that translate to real,
            # user-visible output.
            from sharedagents_agent.tools.runners import (
                _acp_first_token_timeout_s,
                _delegate_idle_timeout_s,
            )

            first_token_timeout_s = _acp_first_token_timeout_s()
            idle_timeout_s = _delegate_idle_timeout_s()
            handoff_accepted = False
            started_tool_call_ids: set[str] = set()
            started_tool_args: dict[str, dict[str, Any]] = {}
            started_tool_names: dict[str, str] = {}
            # Cursor/OpenCode Task*/TodoWrite run inside the ACP server.
            # Mirror them into `<sid>.todos.json` so /todo and the web
            # footer survive turn_ended the same way claude_cli does.
            _task_mirror = None
            sharedagents_sid = str(state.sharedagents_session_id or "")
            if sharedagents_sid:
                try:
                    from sharedagents_agent.tools.harness_task_mirror import (
                        HarnessTaskMirror,
                    )
                    from sharedagents_agent.tools.runners import todos_path_for_session

                    _task_mirror = HarnessTaskMirror(
                        tasks_path=sessions_dir / f"{sharedagents_sid}.tasks.json",
                        todos_path=todos_path_for_session(
                            sessions_dir,
                            sharedagents_sid,
                        ),
                    )
                except Exception:  # noqa: BLE001
                    _task_mirror = None
            nudge_count = 0
            initial_stall: tuple[str, int] | None = None
            tool_rounds_so_far = 0
            turn_visible_chars = 0
            nudge_tracker = TurnEventTracker()
            saved_followup_blocks: list[dict[str, Any]] | None = None
            expect_followup = False
            followup_cancel_retries = 0
            while True:
                this_is_followup = expect_followup
                expect_followup = False
                got_first_output = False
                last_output_at = time.monotonic()
                round_content_parts: list[str] = []
                round_thinking_parts: list[str] = []
                round_had_tools = False
                nudge_tracker.begin_round()
                queued_followup_blocks: list[dict[str, Any]] | None = None
                stalled_nudge_blocks: list[dict[str, Any]] | None = None
                _prompt_iter = client.session_prompt(
                    session_id,
                    prompt_blocks,
                ).__aiter__()
                # One stream read in flight at a time; it survives a Steer
                # wake-up so no item is lost or read twice.
                _next_read: asyncio.Task[Any] | None = None
                while True:
                    _now = time.monotonic()
                    _limit = first_token_timeout_s if not got_first_output else idle_timeout_s
                    _bound = _limit - (_now - last_output_at)
                    try:
                        if _bound <= 0:
                            raise asyncio.TimeoutError
                        if _next_read is None:
                            _next_read = asyncio.ensure_future(
                                _next_prompt_item(_prompt_iter),
                            )
                            live.pending_read = _next_read
                        _steer_wait = asyncio.ensure_future(live.steer.wait())
                        try:
                            _done, _ = await asyncio.wait(
                                {_next_read, _steer_wait},
                                timeout=_bound,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                        finally:
                            _steer_wait.cancel()
                        if live.steer.is_set():
                            # Steer: take the queue now, not at the next
                            # completed tool call.
                            live.steer.clear()
                            q_events, q_blocks = _queue_followup_plan(
                                request,
                                round_idx=tool_rounds_so_far,
                            )
                            if not q_blocks:
                                continue  # already delivered; keep reading
                            for q_ev in q_events:
                                yield q_ev
                            queued_followup_blocks = q_blocks
                            if live.session_id:
                                try:
                                    await client.session_cancel(
                                        live.session_id,
                                        force=True,
                                    )
                                except Exception:  # noqa: BLE001
                                    pass
                            _read, _next_read = _next_read, None
                            await _finish_prompt_for_followup(
                                _prompt_iter,
                                timeout_s=ACP_GRACEFUL_ABORT_S,
                                pending=_read,
                            )
                            break
                        elif _next_read in _done:
                            _read, _next_read = _next_read, None
                            _has_item, item = _read.result()
                            if not _has_item:
                                raise StopAsyncIteration
                        else:
                            raise asyncio.TimeoutError
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        _read, _next_read = _next_read, None
                        await _drop_pending_read(_read)
                        stage = (
                            "idle"
                            if got_first_output
                            else "continuation"
                            if nudge_count or tool_rounds_so_far
                            else "startup"
                        )
                        secs = int(_limit)
                        if initial_stall is None:
                            initial_stall = (stage, secs)
                        try:
                            await _prompt_iter.aclose()
                        except Exception:  # noqa: BLE001
                            pass
                        if live.session_id:
                            try:
                                await client.session_cancel(live.session_id, force=True)
                            except Exception:  # noqa: BLE001
                                pass
                        stall_plan = await apply_nudge_hooks(
                            nudge_tracker,
                            hook_engine=request.extra.get("hook_engine"),
                            hook_ctx=request.extra.get("hook_ctx") or {},
                            finish_reason="stop",
                            nudge_count=nudge_count,
                            max_nudges=MAX_IDLE_EMPTY_CONTINUES,
                        )
                        if nudge_tracker.has_work and stall_plan.is_idle_resume:
                            body, reason_code = (
                                stall_plan.body,
                                stall_plan.decision.reason_code,
                            )
                            nudge_count += 1
                            yield TextDelta(
                                text=idle_empty_continue_notice(
                                    reason_code,
                                    attempt=nudge_count,
                                ),
                                channel="content",
                            )
                            stalled_nudge_blocks = [{"type": "text", "text": body}]
                            sent_chars += len(body)
                            break
                        recovery_detail = (
                            f"; initial {initial_stall[0]} timeout after "
                            f"{initial_stall[1]}s before {nudge_count} recovery attempts"
                            if nudge_count
                            else ""
                        )
                        yield TurnError(
                            message=(
                                f"{self._name} produced no output for {secs}s "
                                f"({stage} timeout{recovery_detail}); tune via "
                                "SHAREDAGENTS_ACP_FIRST_TOKEN_TIMEOUT_S / "
                                "SHAREDAGENTS_DELEGATE_IDLE_TIMEOUT_S."
                            ),
                            classification="delegate_stall",
                        )
                        yield TurnDone(state="error", engine_session_id=state.engine_session_id)
                        return
                    # A valid update/response proves the ACP peer accepted the
                    # prompt. Transport/auth/protocol failures before this point
                    # deliberately leave the handoff pending for retry.
                    update = getattr(item, "update", None)
                    if update is not None:
                        d = _as_dict(update)
                        if d.get("sessionUpdate") == "usage_update":
                            # Protocol-only: do not bump last_output_at (a
                            # periodic usage heartbeat must not defeat idle).
                            # It DOES prove the peer accepted the prompt, so
                            # leave the first-output window.
                            got_first_output = True
                            # REAL context usage the agent reported (ACP
                            # session-usage RFD: used/size). Cursor doesn't
                            # send this yet, but we honour it the moment it does.
                            u, s = d.get("used"), d.get("size")
                            if isinstance(u, (int, float)):
                                real_used = int(u)
                            if isinstance(s, (int, float)):
                                real_size = int(s)
                            token_usage = d.get("usage") or d.get("tokenUsage")
                            if not isinstance(token_usage, dict):
                                # A few ACP servers flatten token fields onto
                                # the update object instead of nesting them.
                                token_usage = d
                            checkpoint = request.extra.get("_usage_checkpoint")
                            if callable(checkpoint):
                                usage_metrics = metrics_from_provider_usage(
                                    token_usage,
                                    model=self._cached_model or self._model or self._name,
                                    provider=(self._name or "").lower(),
                                )
                                if usage_metrics:
                                    checkpoint(usage_metrics)
                            continue
                        events = _translate_update(update)
                        # ACP agents can repeat in-progress tool updates as input,
                        # progress, and execution state evolve. Every frontend
                        # needs one mounted widget per call id; later updates and
                        # results target that existing widget.
                        deduped_events: list[AgentTurnEvent] = []
                        for event in events:
                            if isinstance(event, ToolCallStarted):
                                prev_args = started_tool_args.get(event.call_id)
                                if event.call_id in started_tool_call_ids and not event.arguments:
                                    continue
                                if prev_args is not None and event.arguments == prev_args:
                                    continue
                                started_tool_call_ids.add(event.call_id)
                                started_tool_args[event.call_id] = event.arguments or {}
                                if event.name:
                                    started_tool_names[event.call_id] = event.name
                                round_had_tools = True
                                nudge_tracker.on_tool(
                                    event.name or started_tool_names.get(event.call_id, ""),
                                    arguments=event.arguments or {},
                                )
                            elif isinstance(event, ToolCallResult):
                                # Some ACP servers suppress the initial
                                # ``tool_call`` when it has no useful
                                # arguments, then emit only the completion
                                # update.  A result still proves this round
                                # performed tool work and must make an empty
                                # final response eligible for continuation.
                                round_had_tools = True
                                name = started_tool_names.get(event.call_id, "")
                                start, event, args = _recover_plan_tool_events(
                                    call_id=event.call_id,
                                    name=name,
                                    ok=event.ok,
                                    content=str(event.content or ""),
                                    prev_args=started_tool_args.get(event.call_id) or {},
                                    session_hint=str(state.engine_session_id or ""),
                                )
                                if start is not None:
                                    started_tool_call_ids.add(start.call_id)
                                    started_tool_args[start.call_id] = start.arguments or {}
                                    if start.name:
                                        started_tool_names[start.call_id] = start.name
                                    nudge_tracker.on_tool(
                                        start.name or name,
                                        arguments=start.arguments or {},
                                    )
                                    deduped_events.append(start)
                                nudge_tracker.on_tool(
                                    name or started_tool_names.get(event.call_id, ""),
                                    ok=event.ok,
                                    arguments=args,
                                    result=str(event.content or ""),
                                )
                            deduped_events.append(event)
                        events = deduped_events
                        if events and request.handoff is not None and not handoff_accepted:
                            handoff_accepted = True
                            request.extra["_handoff_consumed"] = True
                            yield HandoffAccepted(
                                handoff_id=request.handoff.handoff_id,
                                through_seq=request.handoff.through_seq,
                            )
                        _auth_fail = next(
                            (
                                ev.text
                                for ev in events
                                if isinstance(ev, TextDelta)
                                # Only inspect user-visible
                                # content. Thinking streams routinely discuss
                                # third-party auth (MongoDB, HTTP 401 tests,
                                # Card API ACL) and must not kill the turn.
                                and ev.channel == "content"
                                and _detect_provider_auth_failure(ev.text)
                            ),
                            None,
                        )
                        if _auth_fail is not None:
                            # The driven ACP server reported (in
                            # its own prose stream) that ITS downstream provider
                            # rejected the request for an auth reason. Classify the
                            # turn as an ERROR instead of streaming that prose
                            # through as normal content indistinguishable from any
                            # other output.
                            try:
                                await _prompt_iter.aclose()
                            except Exception:  # noqa: BLE001
                                pass
                            if live.session_id:
                                try:
                                    await client.session_cancel(live.session_id, force=True)
                                except Exception:  # noqa: BLE001
                                    pass
                            yield TurnError(
                                message=(f"{self._name} upstream provider auth failure: {_auth_fail.strip()[:200]}"),
                                classification="auth",
                            )
                            yield TurnDone(
                                state="error",
                                engine_session_id=state.engine_session_id,
                            )
                            return
                        if events:
                            got_first_output = True
                            last_output_at = time.monotonic()
                        for ev in events:
                            if _task_mirror is not None:
                                if isinstance(ev, ToolCallStarted):
                                    _task_mirror.on_tool_call(
                                        ev.name,
                                        ev.call_id,
                                        ev.arguments,
                                    )
                                elif isinstance(ev, ToolCallResult):
                                    _task_mirror.on_tool_result(
                                        ev.call_id,
                                        ev.content,
                                        ev.ok,
                                    )
                            if isinstance(ev, TextDelta) and ev.channel in (
                                "content",
                                "thinking",
                            ):
                                recv_chars += len(ev.text)
                                if ev.channel == "content":
                                    round_content_parts.append(ev.text)
                                    nudge_tracker.on_content(ev.text)
                                else:
                                    round_thinking_parts.append(ev.text)
                                    nudge_tracker.on_thinking(ev.text)
                                if first_token_at is None:
                                    first_token_at = time.monotonic()
                                last_token_at = time.monotonic()
                                delta_count += 1
                            yield ev
                        # ACP has no intra-prompt round boundary: one
                        # session/prompt runs until PromptResponse. Local
                        # LLM drains the durable follow-up queue between
                        # rounds; without a mid-prompt hook here, WebUI
                        # type-ahead sits until the whole Cursor turn ends.
                        # A completed tool is the closest analog — cancel
                        # the in-flight prompt and send the drained text
                        # as the next session/prompt (same session).
                        if any(isinstance(ev, ToolCallResult) for ev in events):
                            q_events, q_blocks = _queue_followup_plan(
                                request,
                                round_idx=tool_rounds_so_far,
                            )
                            if q_blocks:
                                for q_ev in q_events:
                                    yield q_ev
                                queued_followup_blocks = q_blocks
                                # Cancel first, then read this prompt to
                                # its PromptResponse before opening the
                                # follow-up. See _finish_prompt_for_followup.
                                if live.session_id:
                                    try:
                                        await client.session_cancel(
                                            live.session_id,
                                            force=True,
                                        )
                                    except Exception:  # noqa: BLE001
                                        pass
                                await _finish_prompt_for_followup(
                                    _prompt_iter,
                                    timeout_s=ACP_GRACEFUL_ABORT_S,
                                )
                                break
                    else:
                        if request.handoff is not None and not handoff_accepted:
                            handoff_accepted = True
                            request.extra["_handoff_consumed"] = True
                            yield HandoffAccepted(
                                handoff_id=request.handoff.handoff_id,
                                through_seq=request.handoff.through_seq,
                            )
                        stop_reason = getattr(item, "stop_reason", None) or "end_turn"
                        # PromptResponse.usage (RFD): real per-turn tokens.
                        usage = getattr(item, "usage", None)
                        if usage is not None:
                            ud = _as_dict(usage)
                            checkpoint = request.extra.get("_usage_checkpoint")
                            if callable(checkpoint):
                                usage_metrics = metrics_from_provider_usage(
                                    ud,
                                    model=self._cached_model or self._model or self._name,
                                    provider=(self._name or "").lower(),
                                )
                                if usage_metrics:
                                    checkpoint(usage_metrics)
                            if real_used is None:
                                tot = ud.get("input_tokens") or ud.get("total_tokens")
                                if isinstance(tot, (int, float)):
                                    real_used = int(tot)

                round_content_chars = len("".join(round_content_parts).strip())
                if round_had_tools:
                    tool_rounds_so_far += 1
                turn_visible_chars += round_content_chars
                if stalled_nudge_blocks:
                    prompt_blocks = stalled_nudge_blocks
                    continue
                if queued_followup_blocks:
                    prompt_blocks = queued_followup_blocks
                    saved_followup_blocks = queued_followup_blocks
                    expect_followup = True
                    followup_cancel_retries = 0
                    sent_chars += sum(len(b.get("text", "")) for b in prompt_blocks if isinstance(b, dict))
                    continue
                # Cursor writes ``*.plan.md`` a beat after the empty
                # Create Plan completion. Recover once more before the
                # idle classifier so the 📐 card and delivery flag land.
                for cid, name in list(started_tool_names.items()):
                    prev = started_tool_args.get(cid) or {}
                    start, result, args = _recover_plan_tool_events(
                        call_id=cid,
                        name=name,
                        ok=True,
                        content="",
                        prev_args=prev,
                        session_hint=str(state.engine_session_id or session_id or ""),
                    )
                    if start is None:
                        continue
                    started_tool_args[cid] = args
                    yield start
                    yield result
                    nudge_tracker.on_tool(
                        name,
                        ok=True,
                        arguments=args,
                        result=result.content,
                    )
                sr = str(stop_reason or "end_turn").lower()
                cancelled_stop = sr in {
                    "cancelled",
                    "canceled",
                    "abort",
                    "aborted",
                }
                finish_reason = "length" if sr in {"max_tokens", "length"} else "stop"
                # A follow-up that comes back cancelled with no output was
                # hit by the previous prompt's cancel. Send it once more
                # now that the slot is clear. A second cancel is the
                # operator stopping the turn.
                followup_round_empty = not round_content_parts and not round_thinking_parts and not round_had_tools
                if (
                    this_is_followup
                    and cancelled_stop
                    and followup_round_empty
                    and saved_followup_blocks
                    and followup_cancel_retries < 1
                ):
                    followup_cancel_retries += 1
                    prompt_blocks = saved_followup_blocks
                    expect_followup = True
                    continue
                _hook_engine = request.extra.get("hook_engine")
                _hook_ctx = request.extra.get("hook_ctx") or {}
                for _t in list(nudge_tracker.tools_this_round):
                    await fire_hook_event(
                        _hook_engine,
                        _hook_ctx,
                        "PreToolUse",
                        {"tool_name": _t.get("name") or "", "tool_input": _t.get("arguments") or {}},
                    )
                    await fire_hook_event(
                        _hook_engine,
                        _hook_ctx,
                        "PostToolUse",
                        {
                            "tool_name": _t.get("name") or "",
                            "tool_input": _t.get("arguments") or {},
                            "tool_response": _t.get("result") or "",
                        },
                    )
                plan = await apply_nudge_hooks(
                    nudge_tracker,
                    hook_engine=_hook_engine,
                    hook_ctx=_hook_ctx,
                    finish_reason=finish_reason,
                    cancelled=cancelled_stop,
                    nudge_count=nudge_count,
                    max_nudges=MAX_IDLE_EMPTY_CONTINUES,
                )
                nudge_tracker.commit_round()
                if not plan.is_idle_resume:
                    q_events, q_blocks = _queue_followup_plan(
                        request,
                        round_idx=tool_rounds_so_far,
                    )
                    if q_blocks:
                        for q_ev in q_events:
                            yield q_ev
                        prompt_blocks = q_blocks
                        saved_followup_blocks = q_blocks
                        expect_followup = True
                        followup_cancel_retries = 0
                        sent_chars += sum(len(b.get("text", "")) for b in prompt_blocks if isinstance(b, dict))
                        continue
                    break
                body, reason_code = plan.body, plan.decision.reason_code
                nudge_count += 1
                for sm in plan.system_messages:
                    yield TextDelta(text=f"[hook] {sm}\n", channel="content")
                yield TextDelta(
                    text=idle_empty_continue_notice(
                        reason_code,
                        attempt=nudge_count,
                    ),
                    channel="content",
                )
                prompt_blocks = [{"type": "text", "text": body}]
                sent_chars += len(body)

            turn_end = time.monotonic()
            state.priming_status = "completed"
            state.priming_system_prompt_sha256 = request.system_prompt_sha256
            state.last_used_at = _now_iso()
            state.last_error = ""
            request.extra["_native_acceptance_receipt"] = {
                "kind": self._acceptance_receipt_kind(),
                "mode": effective_request_mode,
                "effective_prompt_sha256": (request.system_prompt_sha256),
                "session_id": session_id,
            }
            metrics = self._turn_metrics(
                state.sharedagents_session_id,
                sent_chars,
                recv_chars,
                stop_reason,
                real_used=real_used,
                real_size=real_size,
                turn_start=turn_start,
                turn_end=turn_end,
                first_token_at=first_token_at,
                last_token_at=last_token_at,
                delta_count=delta_count,
                sessions_dir=sessions_dir,
            )
            # Persist ctx_chars AFTER the estimate updates so a process
            # restart resumes the running occupancy instead of 0.
            self.write_state(sessions_dir, state)
            yield TurnDone(
                state="ok",
                engine_session_id=session_id,
                metrics=metrics,
            )
        except asyncio.CancelledError:
            # The subprocess must be dead BEFORE the
            # aborted terminal is observed, and CancelledError must not
            # re-raise after that yield.
            if live.session_id:
                try:
                    await client.session_cancel(live.session_id, force=True)
                except Exception:  # noqa: BLE001
                    pass
            await _drop_pending_read(live.pending_read)
            try:
                await _reap_proc(proc)
            except Exception:  # noqa: BLE001
                pass
            yield TurnDone(state="aborted", engine_session_id=state.engine_session_id)
            return
        except Exception as exc:  # noqa: BLE001 - turn boundary
            if state.engine_session_id:
                state.priming_status = "failed"
                state.last_error = f"{exc.__class__.__name__}: {exc}"
                self.write_state(sessions_dir, state)
            # Stamp a normalized, shared failure-class
            # label (matching ExternalCliEngine's generic post-spawn catch-all)
            # rather than the raw Python exception class name, so the
            # ~/.sharedagents/errors pipeline and alerting can filter on a stable
            # taxonomy value. Spawn-time / auth / protocol failures are already
            # classified above; anything reaching this turn-boundary catch is a
            # generic engine exception.
            message = str(exc)
            diagnostic = _child_failure_diagnostic(live)
            if diagnostic and _is_transport_closed(exc):
                message = f"{message} ({diagnostic})"
            if "not logged in in this runtime" not in message.lower() and (
                _is_auth_required(exc) or "not authenticated" in message.lower()
            ):
                message = _unauthenticated_engine_message(self._name, exc)
            yield TurnError(message=message, classification="engine_exception")
            yield TurnDone(state="error", engine_session_id=state.engine_session_id)
        finally:
            self._active.pop(turn_id, None)
            await _drop_pending_read(live.pending_read)
            from sharedagents_agent import sandbox_broker as _sb

            _sb.release_process(_acp_lease)
            try:
                from . import engine_pidfile

                if state.sharedagents_session_id:
                    engine_pidfile.clear(sessions_dir, str(state.sharedagents_session_id))
            except Exception:  # noqa: BLE001
                pass
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
            await _reap_proc(proc)
            if live.stderr_task is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(live.stderr_task),
                        timeout=0.25,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    live.stderr_task.cancel()

    # ---- helpers ----------------------------------------------------

    async def _apply_mode(
        self,
        client: Any,
        session_id: str,
        mode: str,
    ) -> None:
        """Apply one canonical mode through the provider's ACP control.

        Most agents implement ``session/set_mode``. A provider that exposes
        behavior through another acknowledged ACP field overrides this hook
        while retaining the common turn loop.
        """
        if self.kind is EngineKind.OPENCODE_ACP:
            native_mode = self.native_mode_agent(mode)
            result = await client.session_set_config_option(
                session_id,
                "mode",
                native_mode,
            )
            self._acknowledged_config_value(result, "mode", native_mode)
            return
        if self.kind is EngineKind.DEVIN_ACP:
            # Devin advertises mode as a select config option. ``session/set_mode``
            # returns an empty success and leaves ``currentValue`` unchanged.
            native_mode = self._acp_mode(mode)
            result = await client.session_set_config_option(
                session_id,
                "mode",
                native_mode,
            )
            self._acknowledged_config_value(
                result,
                "mode",
                native_mode,
                agent_name="Devin",
            )
            return
        await client.session_set_mode(session_id, self._acp_mode(mode))

    @staticmethod
    def native_mode_agent(mode: str) -> str:
        if mode == "plan":
            return "plan"
        if mode in ("auto", "ask", "build", "allow_all", "allow-all"):
            return "build"
        raise ValueError(f"unsupported OpenCode mode: {mode}")

    @staticmethod
    def _acknowledged_config_value(
        result: Any,
        option_id: str,
        requested: str,
        agent_name: str = "OpenCode",
    ) -> str:
        if not isinstance(result, Mapping):
            raise RuntimeError(f"{agent_name} mode acknowledgement is missing")
        options = result.get("configOptions")
        if not isinstance(options, list):
            raise RuntimeError(f"{agent_name} mode acknowledgement omitted configOptions")
        for option in options:
            if not isinstance(option, Mapping) or option.get("id") != option_id:
                continue
            current = str(option.get("currentValue") or "")
            if not current:
                raise RuntimeError(f"{agent_name} {option_id} acknowledgement omitted currentValue")
            if current != requested:
                raise RuntimeError(
                    f"{agent_name} {option_id} acknowledgement mismatch: requested {requested!r}, effective {current!r}"
                )
            return current
        raise RuntimeError(f"{agent_name} acknowledgement omitted {option_id!r}")

    def _acceptance_receipt_kind(self) -> str:
        """Stable receipt discriminator used by readiness verification."""
        if self.kind is EngineKind.OPENCODE_ACP:
            return "opencode_success_terminal"
        return "acp_prompt_response"

    def _mode_failure_is_terminal(self) -> bool:
        """Whether a rejected mode transition prevents prompt dispatch."""
        return self.kind in {EngineKind.OPENCODE_ACP, EngineKind.DEVIN_ACP}

    def _baseline_ctx_chars(self, sid: str, sessions_dir: Path | None) -> int:
        """Running char occupancy to add this turn's sent+recv onto.

        Prefers the in-memory / sidecar total. After a process restart the
        sidecar may be missing ``ctx_chars`` (older sessions); fall back to
        the jsonl peak ``prompt_tokens`` so a collapsed newest turn (580
        after a restart that had been 15k) does not become the new baseline.
        ``4`` matches ``context_report._CHARS_PER_TOKEN``.
        """
        n = int(self._ctx_chars.get(sid, 0) or 0)
        if n > 0:
            return n
        if sessions_dir is None:
            return 0
        jsonl = Path(sessions_dir) / f"{sid}.jsonl"
        try:
            from sharedagents_session.session_reader import peak_turn_context_meter

            peak_tok, _cw = peak_turn_context_meter(jsonl)
        except Exception:  # noqa: BLE001 — estimate restore is best-effort
            peak_tok = 0
        if peak_tok > 0:
            n = int(peak_tok) * 4
            self._ctx_chars[sid] = n
        return n

    def _turn_metrics(
        self,
        sid: str,
        sent_chars: int,
        recv_chars: int,
        stop_reason: str,
        *,
        real_used: int | None = None,
        real_size: int | None = None,
        turn_start: float | None = None,
        turn_end: float | None = None,
        first_token_at: float | None = None,
        last_token_at: float | None = None,
        delta_count: int = 0,
        sessions_dir: Path | None = None,
    ) -> dict[str, Any]:
        """TurnDone metrics with the best-available context size.

        Prefers REAL usage the ACP server reported this turn via a
        ``usage_update`` notification or ``PromptResponse.usage`` (per the
        ACP session-usage RFD). When the agent reports nothing (Cursor
        today), fall back to an ESTIMATE: the running total of chars
        sent + received this session converted to tokens
        (``tokens_estimated=True`` → the TUI shows ``ctx`` with ``≈``).
        The estimate is a lower bound — it can't see the agent's
        server-side system prompt / codebase index / command list.

        The running char total is persisted on the ACP sidecar
        (``ctx_chars``) so a gateway/orchestrator restart does not
        reset occupancy to this-turn-only (the 580/200K footer)."""
        from sharedagents_agent.context_report import (
            estimate_tokens_for_model,
            estimate_tokens_from_chars,
        )

        if real_used is not None:
            used = int(real_used)
            estimated = False
        else:
            baseline = self._baseline_ctx_chars(sid, sessions_dir)
            total = baseline + max(sent_chars, 0) + max(recv_chars, 0)
            self._ctx_chars[sid] = total
            used = estimate_tokens_from_chars(total)
            estimated = True
        window = int(real_size or self.context_window() or 0)

        # Use model-aware estimate for completion tokens.
        model_label = self._cached_model or self._model or ""
        ct = estimate_tokens_for_model(recv_chars, model_label)

        # Keep this engine's REAL/preferred values (cumulative ctx ``used``,
        # the reported ``window``, the model) and let the shared estimator fill
        # the rest (reasoning/result split, total, prefill, gen tok/s) so
        # Cursor/opencode render the same full row as every other agent — and
        # no longer a blank footer on a single-blob priming turn.
        ttft_s = (
            (first_token_at - turn_start)
            if (first_token_at is not None and turn_start is not None and delta_count >= 2)
            else None
        )
        gen_window_s = (
            (last_token_at - first_token_at)
            if (first_token_at is not None and last_token_at is not None and delta_count >= 3)
            else None
        )
        turn_s = (turn_end - turn_start) if (turn_start is not None and turn_end is not None) else None
        from sharedagents_agent.metrics_format import complete_metrics

        raw: dict[str, Any] = {
            "stop_reason": stop_reason,
            "model": model_label,
            "prompt_tokens": used,
            "completion_tokens": ct,
            "tokens_estimated": estimated,
            # Tag the provider (the ACP agent label, e.g. "cursor") so the
            # pricing table resolves this delegate's cost to that provider's
            # rate instead of the generic default. ``used`` is the real
            # PromptResponse.usage count when the server reported it.
            "provider": (self._name or "").lower(),
        }
        if window:
            raw["context_window"] = window
        return complete_metrics(
            raw,
            thinking_chars=0,
            content_chars=max(0, recv_chars),
            model=model_label,
            context_window=window or None,
            turn_s=turn_s,
            ttft_s=ttft_s,
            gen_window_s=gen_window_s,
        )

    def _capture_modes(self, client: Any) -> None:
        """Remember native modes advertised on session/new or session/load."""
        from sharedagents_agent.acp_mode_dict import extract_native_modes

        payload = getattr(client, "last_session_new", None)
        if not isinstance(payload, Mapping):
            payload = {
                "modes": getattr(client, "last_session_modes", None),
                "configOptions": getattr(client, "last_session_config_options", None),
            }
        rows = extract_native_modes(payload)
        if rows:
            self._discovered_modes = rows
            if self._cfg is not None:
                try:
                    self._cfg.discovered_acp_modes = rows
                except (AttributeError, TypeError):
                    pass

    def _capture_model(self, client: Any) -> None:
        """Cache the model the ACP server reported in session/new/load.

        Reads ``models.currentModelId`` and resolves its display name via
        ``availableModels`` so the footer shows e.g. ``composer-2.5``."""
        models = getattr(client, "last_session_models", None)
        if not isinstance(models, dict):
            if self.kind is EngineKind.OPENCODE_ACP:
                options = getattr(client, "last_session_config_options", None)
                if isinstance(options, list):
                    for option in options:
                        if isinstance(option, Mapping) and option.get("id") == "model":
                            current = str(option.get("currentValue") or "").strip()
                            if current:
                                self._cached_model = self._clean_model_label(current)
                            return
            return
        current = models.get("currentModelId")
        if not isinstance(current, str) or not current:
            return
        # Cursor encodes the window in some model ids (gpt/claude/grok);
        # composer omits it (stays 0 → subclass default applies).
        win = _parse_model_context(current)
        label = current
        for m in models.get("availableModels") or []:
            if isinstance(m, dict) and m.get("modelId") == current:
                nm = m.get("name")
                if isinstance(nm, str) and nm:
                    label = nm
                if not win:
                    win = _parse_model_context(nm or "")
                break
        if win:
            self._cached_context_window = win
        self._cached_model = self._clean_model_label(label)

    async def _open_session(
        self,
        client: Any,
        init: Any,
        state: EngineSessionState,
        policy: str,
        cwd: str | None,
    ) -> str:
        # The ACP wire schema requires an ABSOLUTE cwd — never degrade to
        # "." (opencode rejects it with -32602 invalid_params).
        if cwd:
            load_cwd = cwd
        else:
            from sharedagents_core.cwdutil import safe_getcwd

            load_cwd = safe_getcwd()
        # A FAILED prime persists the resumable
        # engine_session_id precisely so the next call attempts session/load
        # rather than starting cold. ``policy == "never"`` alone excluded the
        # failed case (priming_status="failed" -> policy "first_turn"), so a
        # partial first prime silently opened a brand-new session and dropped
        # the resumable id every retry. Treat a recorded id + a failed prime as
        # loadable too.
        can_load = (
            bool(state.engine_session_id)
            and _supports_load(init)
            and (policy in {"never", "sha_changed"} or state.priming_status == "failed")
        )
        if can_load:
            try:
                await client.session_load(state.engine_session_id, cwd=load_cwd)
                return state.engine_session_id
            except Exception as exc:  # noqa: BLE001
                # Authenticate and retry the LOAD before abandoning the
                # saved session — otherwise an auth-required resume falls
                # through to session/new and silently drops conversation
                # continuity on every turn after the first.
                if _is_auth_required(exc) and self._auth_method:
                    try:
                        await client.authenticate(self._auth_method)
                        await client.session_load(state.engine_session_id, cwd=load_cwd)
                        return state.engine_session_id
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "acp session_load(%s) failed after re-auth; opening a NEW "
                            "engine session — prior conversation context is dropped",
                            state.engine_session_id,
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        "acp session_load(%s) failed; opening a NEW engine session — "
                        "prior conversation context is dropped",
                        state.engine_session_id,
                        exc_info=True,
                    )
        return await self._new_session_with_auth(client, load_cwd)

    async def _new_session_with_auth(self, client: Any, cwd: str) -> str:
        try:
            return await client.session_new(cwd)
        except Exception as exc:  # noqa: BLE001
            if _is_auth_required(exc) and self._auth_method:
                try:
                    await client.authenticate(self._auth_method)
                    return await client.session_new(cwd)
                except Exception as auth_exc:  # noqa: BLE001
                    raise RuntimeError(_unauthenticated_engine_message(self._name, auth_exc)) from auth_exc
            if _is_auth_required(exc):
                raise RuntimeError(_unauthenticated_engine_message(self._name, exc)) from exc
            raise


# =====================================================================
# Module-level helpers (pure → unit-testable without a live client).
# =====================================================================


def _openai_image_url_to_acp(part: Mapping[str, Any]) -> dict[str, Any] | None:
    """Translate an OpenAI ``image_url`` part into an ACP ``image`` block.

    Web / Telegram / TUI land images on engines as OpenAI vision parts
    (``prompt_parts``). ACP ``session/prompt`` wants
    ``{type: "image", data, mimeType}`` — the same shape Telegram already
    sends on the north gateway. Returns ``None`` when the part has no
    usable payload.
    """
    image_url = part.get("image_url")
    url = ""
    if isinstance(image_url, dict):
        url = str(image_url.get("url") or "")
    elif isinstance(image_url, str):
        url = image_url
    if not url:
        return None
    mime = "image/png"
    data = ""
    if url.startswith("data:"):
        # data:[<mime>][;base64],<payload>
        try:
            header, payload = url.split(",", 1)
        except ValueError:
            return None
        meta = header[5:]  # strip "data:"
        if ";" in meta:
            mime = meta.split(";", 1)[0] or mime
        elif meta:
            mime = meta
        data = payload.strip()
    else:
        # Non-data URLs are not inlined into ACP image blocks; the local
        # CLI engines materialize those to disk instead.
        return None
    if not data:
        return None
    if not mime.startswith("image/"):
        mime = "image/png"
    return {"type": "image", "data": data, "mimeType": mime}


def _acp_foreground_instruction() -> str:
    """Operator requirement appended to every external-ACP prompt.

    ACP v1 treats the reply to a ``session/prompt`` as the END of the
    turn — sharedagents never polls the session afterwards, so an
    acknowledgment-only "started" reply (or a hand-off to the agent's
    own background subprocess / thread / cloud agent) is a LOST result.
    The prompt itself must ask the agent to respond with the final
    result only, once the work is complete, in the same reply.
    """
    return (
        "\n\nIMPORTANT (operator requirement): run at least one task in the "
        "FOREGROUND of this session. Do NOT hand your work off to your own "
        "background subprocess, thread, background task, cloud/background "
        "agent, or scheduled feature. Do NOT reply with only an "
        "acknowledgment or a 'started' message — this session will not be "
        "polled for later results. Respond with the final result only after "
        "the work is complete, in this same response."
    )


def _outbound_prompt_blocks(
    request: TurnRequest,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The payload sent to the agent: ``blocks`` plus the foreground note.

    An engine command (``/compact``) goes out bare: anything added around
    it makes it an unknown command or turns it into plain chat.
    """
    if request.extra.get("_engine_command"):
        return [{"type": "text", "text": request.prompt}]
    return _append_acp_foreground_instruction(blocks)


def _append_acp_foreground_instruction(
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append the operator instruction to the ACP prompt payload.

    The instruction rides the FIRST text block — on priming turns that is
    the block holding the ``<system>`` fold followed by the user text, so
    the requirement reads after the system content, never inside it. An
    image-only payload has no text block; the instruction then becomes its
    own leading text block so the agent still sees it.
    """
    instruction = _acp_foreground_instruction()
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            block["text"] = f"{block.get('text', '')}{instruction}"
            return blocks
    blocks.insert(0, {"type": "text", "text": instruction.strip()})
    return blocks


# Match local_llm/turn_loop.py mid-turn steer wrapper so ACP follow-ups
# that arrived as source="steer" get the same interrupt semantics.
_ACP_STEER_WRAP = (
    "⚠ MID-TURN USER STEER — the user has "
    "interrupted with a new instruction. Stop "
    "expanding on the prior plan. Treat this "
    "as the operative request; any prior tool "
    "work counts only if it directly serves "
    "this new directive.\n\n"
)


def _drain_followup_queue(
    request: TurnRequest,
) -> list[tuple[str, str, str]]:
    """Invoke ``request.extra['queue_drain']``; never raise into the turn."""
    qdrain = request.extra.get("queue_drain") if request.extra else None
    if not callable(qdrain):
        return []
    try:
        items = qdrain() or []
    except Exception:  # noqa: BLE001
        return []
    out: list[tuple[str, str, str]] = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            continue
        qid = str(item[0] or "")
        text = str(item[1] or "")
        src = str(item[2] or "") if len(item) >= 3 else ""
        if qid and text.strip():
            out.append((qid, text, src))
    return out


def _followup_prompt_blocks(
    items: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Join drained follow-ups into one ACP text block (steer-wrapped)."""
    parts: list[str] = []
    for _qid, text, src in items:
        if str(src) == "steer":
            parts.append(f"{_ACP_STEER_WRAP}{text}")
        else:
            parts.append(text)
    joined = "\n\n".join(p for p in parts if p.strip())
    if not joined.strip():
        return []
    return [{"type": "text", "text": joined}]


def _queue_followup_plan(
    request: TurnRequest,
    *,
    round_idx: int,
) -> tuple[list[AgentTurnEvent], list[dict[str, Any]]]:
    """Drain the durable follow-up queue into live events + next prompt.

    Returns ``([], [])`` when nothing is pending. Callers MUST send the
    returned prompt blocks (or the drained items vanish from the widget
    without reaching the model).
    """
    items = _drain_followup_queue(request)
    blocks = _followup_prompt_blocks(items)
    if not blocks:
        return [], []
    events: list[AgentTurnEvent] = []
    for qid, text, src in items:
        if str(src) == "steer":
            events.append(UserMessageInjected(text=text, round_idx=round_idx))
        else:
            events.append(
                QueueMessageInjected(
                    text=text,
                    queue_id=qid,
                    round_idx=round_idx,
                )
            )
    return events, blocks


def priming_includes_system_prompt(
    blocks: list[dict[str, Any]],
    request: TurnRequest,
    policy: str,
) -> bool:
    """True when this turn is allowed to be recorded as primed.

    Resume turns are already primed. A priming turn with no system prompt
    has nothing to send. Any other priming turn counts only when the
    composed prompt is present in the text that was actually sent.
    """
    if policy not in {"first_turn", "sha_changed"}:
        return True
    expected = (request.system_prompt or "").strip()
    if not expected:
        return True
    return any(isinstance(block, dict) and expected in str(block.get("text") or "") for block in blocks)


def _build_prompt_blocks(request: TurnRequest, policy: str) -> list[dict[str, Any]]:
    """Compose the ACP prompt content blocks for this turn.

    ACP has no separate system-prompt channel, so on priming turns
    (``first_turn`` / ``sha_changed``) we fold the composed system prompt
    AND the below-cache-boundary ``preamble`` blocks (pinned facts,
    compaction summary, todos, memory recall, skills) into a leading
    ``<system>`` block — the same content CLI engines bake into their
    first-message priming (see ``external_cli`` preamble folding).
    An attached image does not skip that fold: the picture is a separate
    content block, and the agent's AGENTS.md still has to be on the turn.

    Multimodal: when ``request.prompt_parts`` carries OpenAI ``image_url``
    parts (web paste / Telegram photo), emit ACP ``image`` content blocks
    alongside the text. Dropping them left Cursor/OpenCode ACP turns
    text-only even though the web UI staged the screenshot.
    """
    text = request.prompt or ""
    image_blocks: list[dict[str, Any]] = []
    if request.prompt_parts:
        # prompt_parts wins for multimodal (TurnRequest contract). Collect
        # text parts into ``text`` when the string prompt is empty, and
        # always collect image_url → ACP image blocks.
        part_texts: list[str] = []
        for part in request.prompt_parts:
            if not isinstance(part, Mapping):
                continue
            ptype = str(part.get("type") or "")
            if ptype == "text":
                piece = str(part.get("text") or "")
                if piece:
                    part_texts.append(piece)
            elif ptype == "image_url":
                acp_img = _openai_image_url_to_acp(part)
                if acp_img is not None:
                    image_blocks.append(acp_img)
        if not text and part_texts:
            text = "\n".join(part_texts)
    handoff = getattr(request, "handoff", None)
    if handoff is not None and getattr(handoff, "rendered", ""):
        text = f"<sharedagents-canonical-handoff>\n{handoff.rendered}\n</sharedagents-canonical-handoff>\n\n{text}"
    if policy in {"first_turn", "sha_changed"}:
        sys_parts: list[str] = []
        if request.system_prompt:
            sys_parts.append(request.system_prompt)
        for block in request.preamble or []:
            content = str(block.get("content") or "") if isinstance(block, dict) else ""
            if content:
                sys_parts.append(content)
        if sys_parts:
            sys_text = "\n\n".join(sys_parts)
            text = f"<system>\n{sys_text}\n</system>\n\n---\n\n{text}"
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    blocks.extend(image_blocks)
    # Empty prompt with no images still needs a text block so ACP servers
    # that reject empty content arrays keep working (resume/prime edges).
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


# Keys Cursor/ACP inject into rawInput that are NOT real tool arguments.
# A payload of only these must be treated as empty so we wait for the
# ``in_progress`` update (same as OpenCode's empty-rawInput two-phase).
_ACP_TOOL_META_KEYS = frozenset({"_toolName", "toolName", "criticality"})

# ACP ``kind`` → short display name when ``title`` is a dumped command body.
_ACP_KIND_LABELS = {
    "execute": "Shell",
    "shell": "Shell",
    "bash": "Shell",
    "read": "Read",
    "edit": "Edit",
    "write": "Write",
    "search": "Search",
    "grep": "Grep",
    "delete": "Delete",
    "move": "Move",
    "think": "Think",
    "todo": "Todos",
    "other": "tool",
}


def _path_from_acp_content(content: Any) -> str:
    """Path off a completed ``content`` block when ``rawInput`` stayed empty.

    Cursor Read/Edit often put ``path`` on a ``diff`` / resource item and
    never fill ``rawInput``. Only sets a path; never clears one.
    """
    blocks = content if isinstance(content, list) else [content]
    for block in blocks:
        if not isinstance(block, dict):
            continue
        path = block.get("path")
        if path:
            return str(path)
        inner = block.get("content")
        if isinstance(inner, dict) and inner.get("path"):
            return str(inner["path"])
    return ""


def _path_from_acp_tool_title(title: str) -> str:
    """Pull a filesystem path off titles like ``Read File /tmp/foo.js``.

    Cursor sometimes puts the path only in ``title`` on the first
    ``tool_call`` (empty ``rawInput`` / ``locations``). Do not treat a
    bare verb (``Read File``, ``grep foo``) as a path.
    """
    text = (title or "").strip()
    if not text:
        return ""
    lower = text.lower()
    rest = ""
    for prefix in ("read file ", "read ", "grep ", "search "):
        if lower.startswith(prefix):
            rest = text[len(prefix) :].strip()
            break
    if not rest:
        return ""
    if rest.startswith("file://"):
        rest = rest[7:]
    if rest.startswith(("~", "/", ".")) or "/" in rest:
        return rest
    return ""


def _acp_tool_call_arguments(d: Mapping[str, Any]) -> dict[str, Any]:
    """Extract tool args from an ACP ``tool_call`` sessionUpdate.

    Cursor and other ACP servers may send ``rawInput`` as a dict, a JSON
    string, or omit it entirely while populating ``locations`` with file
    paths. Normalise all shapes so Telegram/TUI headers can show a useful
    one-liner (``grep: pattern``, ``Read File: path/to/foo.py``).

    Cursor often ships ``rawInput: {"_toolName": "updateTodos"}`` (or
    similar metadata-only dicts) on the initial ``tool_call``; treat those
    as empty so :func:`_emit_tool_call_started` waits for real args on the
    first ``in_progress`` update — otherwise the TUI shows
    ``🔧 Update TODOs {"_toolName": "updateTodos"}``.
    """
    raw_input = d.get("rawInput")
    if raw_input is None:
        raw_input = d.get("raw_input")
    args: dict[str, Any] = {}
    if isinstance(raw_input, dict):
        args = {k: v for k, v in raw_input.items() if k not in _ACP_TOOL_META_KEYS}
    elif isinstance(raw_input, str) and raw_input.strip():
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, dict):
                args = {k: v for k, v in parsed.items() if k not in _ACP_TOOL_META_KEYS}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    if not args:
        locs = d.get("locations")
        if isinstance(locs, list):
            for loc in locs:
                if not isinstance(loc, dict):
                    continue
                path = loc.get("path")
                if not path and isinstance(loc.get("uri"), str):
                    uri = loc["uri"]
                    path = uri[7:] if uri.startswith("file://") else uri
                if path:
                    args.setdefault("path", path)
                    break
    # Cursor Read uses ``target_file``; OpenCode uses ``filePath``.
    if not args.get("path"):
        for key in ("target_file", "targetFile", "filePath", "file_path"):
            if args.get(key):
                args["path"] = args[key]
                break
    if not args.get("path"):
        titled = _path_from_acp_tool_title(str(d.get("title") or ""))
        if titled:
            args["path"] = titled
    if not args.get("path"):
        from_content = _path_from_acp_content(d.get("content"))
        if from_content:
            args["path"] = from_content
    if not args.get("todos") and not args.get("items"):
        extra = _acp_todos_from_update(d)
        if extra:
            args = dict(args)
            args.update(extra)
    return args


def _acp_todos_from_update(d: Mapping[str, Any]) -> dict[str, Any]:
    """Pull a Cursor/Claude todo list off fields ``rawInput`` missed.

    Cursor's first ``Update TODOs`` frame is metadata-only; later updates
    sometimes put the list on the update object or inside ``content``.
    """
    for key in ("todos", "items"):
        val = d.get(key)
        if isinstance(val, list) and val:
            return {key: val}
    for blob in (d.get("content"), d.get("rawOutput"), d.get("raw_output")):
        parsed = _parse_todo_payload(blob)
        if parsed:
            return parsed
    return {}


def _parse_todo_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        for key in ("todos", "items"):
            val = value.get(key)
            if isinstance(val, list) and val:
                out = {key: val}
                if value.get("merge") is not None:
                    out["merge"] = value.get("merge")
                return out
        return {}
    if isinstance(value, str) and value.strip().startswith(("{", "[")):
        try:
            return _parse_todo_payload(json.loads(value))
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(it, dict) and it.get("content") for it in value):
            return {"todos": list(value)}
        for item in value:
            parsed = _parse_todo_payload(item.get("text") if isinstance(item, dict) else item)
            if parsed:
                return parsed
            if isinstance(item, dict):
                nested = item.get("content")
                parsed = _parse_todo_payload(nested)
                if parsed:
                    return parsed
    return {}


def _acp_tool_call_title(d: Mapping[str, Any]) -> str:
    """Short stable tool label for TUI/Telegram headers.

    Cursor sometimes puts the entire shell command (markdown-fenced) in
    ``title``, which then renders as a multi-line fake "user" block in
    the TUI. Prefer ``kind`` / a short title; fall back to ``tool``.
    """
    kind = str(d.get("kind") or "").strip().lower()
    title = str(d.get("title") or "").strip()
    if kind in _ACP_KIND_LABELS and (
        not title or len(title) > 60 or "\n" in title or (title.startswith("`") and title.endswith("`"))
    ):
        return _ACP_KIND_LABELS[kind]
    if title and len(title) <= 60 and "\n" not in title:
        # Strip a single pair of wrapping backticks (Cursor shell titles).
        if title.startswith("`") and title.endswith("`") and title.count("`") == 2:
            inner = title[1:-1]
            if kind in _ACP_KIND_LABELS:
                return _ACP_KIND_LABELS[kind]
            if len(inner) > 40:
                return "Shell"
            return inner
        return title
    if kind in _ACP_KIND_LABELS:
        return _ACP_KIND_LABELS[kind]
    return kind or "tool"


def _safe_resource_links(value: Any) -> list[ArtifactReference]:
    """Extract explicit ACP resource links without retaining secret-bearing URI parts."""
    values = value if isinstance(value, list) else [value]
    refs: list[ArtifactReference] = []
    for block in values:
        if not isinstance(block, Mapping) or block.get("type") != "resource_link":
            continue
        raw_uri = block.get("uri")
        if not isinstance(raw_uri, str) or not raw_uri or len(raw_uri) > 8192:
            continue
        if any(ord(char) < 32 for char in raw_uri):
            continue
        try:
            parsed = urlsplit(raw_uri)
        except ValueError:
            continue
        scheme = parsed.scheme.lower()
        is_absolute_path = not scheme and Path(raw_uri).is_absolute()
        if scheme not in {"file", "http", "https"} and not is_absolute_path:
            # In particular, never retain data: or opaque provider-native URIs.
            continue
        if is_absolute_path:
            uri = ""
            local_path = raw_uri
        elif scheme in {"http", "https"}:
            if not parsed.netloc:
                continue
            # Strip userinfo, query, and fragment: all commonly carry tokens.
            netloc = parsed.netloc.rsplit("@", 1)[-1]
            uri = urlunsplit((scheme, netloc, parsed.path, "", ""))
        else:
            # File links are explicit artifact paths. Queries/fragments are not
            # meaningful filesystem identity and can carry secret material.
            uri = urlunsplit((scheme, parsed.netloc, parsed.path, "", ""))
        digest = str(block.get("sha256") or "").lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            digest = ""
        try:
            size = max(0, int(block.get("sizeBytes", block.get("size", 0)) or 0))
        except (TypeError, ValueError):
            size = 0
        refs.append(
            ArtifactReference(
                uri=uri,
                path=(local_path if is_absolute_path else ""),
                display_name=str(block.get("name") or "")[:512],
                media_type=str(block.get("mimeType") or block.get("media_type") or "")[:255],
                size_bytes=size,
                sha256=digest,
                storage_scope=("remote" if scheme in {"http", "https"} else "local_session"),
                locator=(local_path if is_absolute_path else uri),
            )
        )
    return refs


def _emit_tool_call_started(
    out: list[AgentTurnEvent],
    d: Mapping[str, Any],
) -> None:
    """Emit :class:`ToolCallStarted` as soon as the call has an id or title.

    OpenCode/Cursor often send an empty ``rawInput`` on the initial
    ``tool_call`` and fill args on the first ``in_progress`` update.
    Waiting for those args delayed the WebUI ``tool_call`` event, so the
    first reasoning block never closed and later thoughts merged into it.
    Emit the title immediately; a later start with the same ``call_id``
    updates arguments.

    do not wait for output/args).
    """
    call_id = str(d.get("toolCallId") or "")
    name = _acp_tool_call_title(d)
    if not call_id and not name:
        return
    args = _acp_tool_call_arguments(d)
    from sharedagents_agent.tools.widget_adapt import adapt_todo_args, widget_kind

    if widget_kind(name, acp_kind=str(d.get("kind") or "")) == "todo":
        adapted = adapt_todo_args(args)
        if adapted:
            args = adapted
    out.append(
        ToolCallStarted(
            call_id=call_id,
            name=name,
            arguments=args,
        )
    )


def _translate_update(update: Any) -> list[AgentTurnEvent]:
    d = _as_dict(update)
    kind = d.get("sessionUpdate")
    out: list[AgentTurnEvent] = []
    if kind == "agent_message_chunk":
        text = _text_of(d.get("content"))
        if text:
            out.append(TextDelta(text=text, channel="content"))
        out.extend(_safe_resource_links(d.get("content")))
    elif kind == "agent_thought_chunk":
        text = _text_of(d.get("content"))
        if text:
            out.append(TextDelta(text=text, channel="thinking"))
    elif kind == "tool_call":
        _emit_tool_call_started(out, d)
    elif kind == "tool_call_update":
        from sharedagents_agent.tools.widget_adapt import widget_kind

        call_id = str(d.get("toolCallId") or "")
        status = d.get("status")
        if status in ("completed", "failed"):
            # Completion-only servers never sent a useful start. Mount the
            # card before the result when a title/args exist so the WebUI
            # does not drop the body.
            if d.get("title") or d.get("kind") or d.get("rawInput") or d.get("raw_input"):
                _emit_tool_call_started(out, d)
            content = _acp_plan_result_text(d)
            if (
                content
                and widget_kind(
                    _acp_tool_call_title(d),
                    acp_kind=str(d.get("kind") or ""),
                )
                == "plan"
            ):
                _emit_tool_call_started(
                    out,
                    {**d, "rawInput": {**_acp_tool_call_arguments(d), "plan": content}},
                )
            out.append(
                ToolCallResult(
                    call_id=call_id,
                    ok=(status == "completed"),
                    content=content,
                )
            )
            out.extend(_safe_resource_links(d.get("content")))
        else:
            # First ``in_progress`` update often carries the real args.
            _emit_tool_call_started(out, d)
            text = _acp_tool_output_text(d)
            if text:
                out.append(TextDelta(text=text, channel="tool_delta", call_id=call_id))
    if not out and kind not in _KNOWN_ACP_UPDATE_KINDS:
        # An evolving/unknown ACP wire-event kind must
        # leave a diagnostic trace, never vanish as a silent empty translation.
        logger.debug("acp: unrecognized session update kind %r (dropped)", kind)
    return out


def _detect_provider_auth_failure(text: str) -> bool:
    """Heuristic: does this ACP-server prose report a provider auth failure?

    The driven ACP server may report that ITS OWN downstream provider rejected
    a request for an auth reason inside a normal ``agent_message_chunk`` — the
    only place today's narrow JSON-RPC ``-32000`` / session-load substring
    checks never inspect. Conservative on purpose (this is a
    LOW/MEDIUM-confidence heuristic surface): require an unambiguous auth phrase.

    Deliberately narrow: bare ``unauthorized`` / ``authentication failed`` in
    agent prose usually refers to the task (MongoDB, HTTP routes under review,
    pytest fixtures) rather than the Cursor/OpenAI upstream login.
    """
    t = (text or "").lower()
    if not t:
        return False
    if "invalid api key" in t or "invalid_api_key" in t:
        return True
    # OpenAI's canonical 401 body is literally "Incorrect API key provided" —
    # require the "key" qualifier so a bare "incorrect" in prose can't trip it.
    if "incorrect api key" in t:
        return True
    if "401" in t and "unauthorized" in t:
        return True
    if "login required" in t and any(
        marker in t for marker in ("cursor", "provider", "upstream", "openai", "anthropic")
    ):
        return True
    if "authentication failed" in t or "authentication error" in t:
        if _looks_like_task_auth_discussion(t):
            return False
        return any(
            marker in t
            for marker in (
                "provider",
                "upstream",
                "http",
                "cursor",
                "openai",
                "anthropic",
                "api key",
                "login required",
                "not logged in",
                "sign in",
            )
        )
    if "unauthorized" in t:
        return any(
            marker in t
            for marker in (
                "provider",
                "http 401",
                "401 unauthorized",
                "api key",
                "not authorized to access",
                "login",
                "sign in",
                "cursor",
                "openai",
                "anthropic",
                "upstream",
            )
        )
    return False


def _looks_like_task_auth_discussion(text: str) -> bool:
    """Agent prose about auth in the task domain, not the ACP provider login."""
    return any(
        marker in text
        for marker in (
            "mongodb",
            "mongoservererror",
            "mongosh",
            "shard",
            "replica set",
            "pytest",
            "test_",
            "unauthorized test",
            "card api",
            "require_auth",
            "flask",
            "shared data",
        )
    )


def _make_permission_handler(mode: str) -> Callable[[dict], Awaitable[dict]]:
    """Map sharedagents's permission mode onto ACP ``session/request_permission``.

    ``auto`` and Devin's native ``bypass`` allow; ``plan`` rejects (plan mode
    is also set on the session). Other modes wait for an operator bridge;
    the turn's idle timeout fails closed if no decision arrives.
    """

    async def handler(params: dict) -> dict:
        options = params.get("options") or []
        m = (mode or "").strip().lower()
        if m == "plan":
            option_id = _pick_option(
                options,
                ("reject_once", "reject_always"),
                fallback=False,
            )
        elif m in {"auto", "bypass"}:
            option_id = _pick_option(
                options,
                ("allow_always", "allow_once"),
                fallback=False,
            )
        else:
            # ``ask`` must BLOCK for an explicit operator
            # decision and never silently auto-allow. No client-side operator
            # bridge delivers that decision yet, so the permission request
            # blocks here — run_turn's idle timeout then fails the turn SAFE
            # rather than granting an unattended permission.
            await asyncio.Event().wait()
            return {"outcome": {"outcome": "cancelled"}}
        if option_id is None:
            return {"outcome": {"outcome": "cancelled"}}
        return {"outcome": {"outcome": "selected", "optionId": option_id}}

    return handler


def _pick_option(
    options: list,
    kinds: tuple[str, ...],
    *,
    fallback: bool = True,
) -> str | None:
    for kind in kinds:
        for opt in options:
            if isinstance(opt, dict) and opt.get("kind") == kind:
                return opt.get("optionId")
    if fallback:
        for opt in options:
            if isinstance(opt, dict) and opt.get("optionId"):
                return opt.get("optionId")
    return None


def _acp_tool_output_text(d: Mapping[str, Any]) -> str:
    """Best-effort tool body from ACP ``content`` or ``rawOutput``.

    Cursor Read/grep often leave ``content`` as a ``diff`` / resource block
    (or put the body only in ``rawOutput``). ``_text_of`` used to return
    empty, so the WebUI painted SUCCESS cards with "No output returned."
    """
    text = _text_of(d.get("content"))
    if text.strip():
        return text
    raw = d.get("rawOutput")
    if raw is None:
        raw = d.get("raw_output")
    if isinstance(raw, str) and raw.strip():
        return raw
    if isinstance(raw, dict):
        for key in ("text", "content", "output", "stdout", "result"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        try:
            dumped = json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError):
            dumped = ""
        if dumped and dumped not in ("{}", "null", "[]"):
            return dumped
    return text


def _recover_plan_tool_events(
    *,
    call_id: str,
    name: str,
    ok: bool,
    content: str,
    prev_args: Mapping[str, Any] | None,
    session_hint: str = "",
) -> tuple[ToolCallStarted | None, ToolCallResult, dict[str, Any]]:
    """Fill empty Create Plan args/result from the Cursor ``*.plan.md`` file."""
    from sharedagents_agent.tools.widget_adapt import (
        extract_plan_markdown,
        is_plan_tool_name,
        recover_cursor_plan_markdown,
    )

    args = dict(prev_args or {})
    text = (content or "").strip() or extract_plan_markdown(args)
    if not text and is_plan_tool_name(name):
        text = recover_cursor_plan_markdown(session_hint=session_hint)
    started = None
    if text and is_plan_tool_name(name) and not extract_plan_markdown(args):
        args = {**args, "plan": text}
        started = ToolCallStarted(call_id=call_id, name=name, arguments=args)
    return started, ToolCallResult(call_id=call_id, ok=ok, content=text or content or ""), args


def _acp_plan_result_text(d: Mapping[str, Any], *, known_name: str = "", session_hint: str = "") -> str:
    """Plan-card body: ACP text first, then the Cursor ``*.plan.md`` file."""
    from sharedagents_agent.tools.widget_adapt import (
        extract_plan_markdown,
        is_plan_tool_name,
        recover_cursor_plan_markdown,
        widget_kind,
    )

    args = _acp_tool_call_arguments(d)
    text = extract_plan_markdown(args) or _acp_tool_output_text(d)
    name = _acp_tool_call_title(d) or known_name
    if text.strip():
        return text
    if widget_kind(name, acp_kind=str(d.get("kind") or "")) != "plan" and not is_plan_tool_name(name):
        return text
    return recover_cursor_plan_markdown(session_hint=session_hint)


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        kind = str(content.get("type") or "")
        if kind == "text":
            return str(content.get("text") or "")
        if kind == "diff":
            return str(content.get("newText") or content.get("new_text") or "")
        if kind in {"resource", "resource_link"}:
            # Never surface the URI as chat text — malformed / secret-bearing
            # links stay in ArtifactReference only.
            return str(content.get("text") or "")
        inner = content.get("content")
        if inner is not None:
            return _text_of(inner)
        for key in ("text", "newText", "new_text", "output"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    if isinstance(content, (list, tuple)):
        return "".join(_text_of(c) for c in content)
    dump = getattr(content, "model_dump", None)
    if callable(dump):
        try:
            return _text_of(dump(by_alias=True, exclude_none=True))
        except Exception:  # noqa: BLE001
            pass
    return ""


def _as_dict(update: Any) -> dict[str, Any]:
    if isinstance(update, dict):
        return _normalize_acp_keys(update)
    dump = getattr(update, "model_dump", None)
    if callable(dump):
        try:
            return _normalize_acp_keys(dump(by_alias=True, exclude_none=True))
        except Exception:  # noqa: BLE001
            try:
                return _normalize_acp_keys(dump(exclude_none=True))
            except Exception:  # noqa: BLE001
                pass
    return {}


def _normalize_acp_keys(d: Mapping[str, Any]) -> dict[str, Any]:
    """Accept camelCase wire keys or snake_case model_dump keys."""
    if not isinstance(d, dict):
        return {}
    out = dict(d)
    for camel, snake in (
        ("rawInput", "raw_input"),
        ("rawOutput", "raw_output"),
        ("toolCallId", "tool_call_id"),
        ("sessionUpdate", "session_update"),
        ("newText", "new_text"),
    ):
        if camel not in out and snake in out:
            out[camel] = out[snake]
    return out


def _supports_load(init: Any) -> bool:
    caps = getattr(init, "agent_capabilities", None)
    if caps is None and isinstance(init, dict):
        caps = init.get("agentCapabilities") or init.get("agent_capabilities")
    val = getattr(caps, "load_session", None)
    if val is None and isinstance(caps, dict):
        val = caps.get("loadSession") or caps.get("load_session")
    return bool(val)


def _unauthenticated_engine_message(engine_name: str, exc: Exception) -> str:
    """Human-readable failure when an ACP server is not logged in.

    Cursor (and some other ACP CLIs) reject ``session/new`` with JSON-RPC
    ``-32602 Invalid params`` when unauthenticated. The Web UI was showing
    only that opaque code; the cursor-agent log says
    ``Session creation rejected: not authenticated``.
    """
    return (
        f"{engine_name} ACP is not logged in in this runtime. "
        f"The server rejected the session ({exc}). "
        "This is not a bad image payload — unauthenticated Cursor/OpenCode "
        "returns JSON-RPC -32602 Invalid params. "
        "Log in inside the same runtime that hosts the orchestrator "
        "(e.g. `docker exec -it <web-container> cursor-agent login`) "
        "or mount the host Cursor credentials into the container."
    )


def _is_auth_required(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code == _AUTH_REQUIRED:
        return True
    blob = " ".join(str(part).lower() for part in (exc, getattr(exc, "message", ""), getattr(exc, "data", "")) if part)
    return ("auth" in blob and "requir" in blob) or "not authenticated" in blob or "unauthenticated" in blob


async def _drain_stderr_tail(reader: Any, tail: bytearray) -> None:
    """Drain child stderr continuously while retaining only a bounded tail."""
    try:
        while True:
            chunk = await reader.read(1024)
            if not chunk:
                return
            tail.extend(chunk)
            if len(tail) > 8192:
                del tail[:-8192]
    except (asyncio.CancelledError, Exception):
        return


def _is_transport_closed(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    return "transportclosed" in name or "transport closed" in str(exc).lower()


def _child_failure_diagnostic(live: _LiveTurn) -> str:
    """Bound and redact subprocess diagnostics safe for terminal errors."""
    proc = live.proc
    code = getattr(proc, "returncode", None)
    parts = [f"child exit code: {code if code is not None else 'unknown'}"]
    if live.stderr_tail:
        from sharedagents_core.secrets_scrubber import scrub_text

        raw = bytes(live.stderr_tail).decode("utf-8", "replace")
        parts.append(f"child stderr tail: {scrub_text(raw)[-2000:]}")
    else:
        parts.append("child stderr tail: unavailable")
    return "; ".join(parts)


def _kill_proc(proc: Any, *, force: bool) -> None:
    if proc is None:
        return
    try:
        if force and hasattr(proc, "kill"):
            proc.kill()
        elif hasattr(proc, "terminate"):
            proc.terminate()
    except (ProcessLookupError, OSError):
        pass


async def _reap_proc(proc: Any) -> None:
    if proc is None:
        return
    _kill_proc(proc, force=False)
    waiter = getattr(proc, "wait", None)
    if not callable(waiter):
        return
    try:
        await asyncio.wait_for(waiter(), timeout=2.0)
    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
        _kill_proc(proc, force=True)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = ["AcpEngine"]
