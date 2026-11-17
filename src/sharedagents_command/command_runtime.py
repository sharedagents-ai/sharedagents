"""Server-side slash-command runtime — the single source of truth.

ARCHITECTURE LAW (do not violate): *zero* feature/command logic lives on
any frontend (TUI, Telegram, web). Every slash command's behavior is
computed here, inside the agent/gateway, and returned as a
:class:`CommandResponse`. Frontends are thin renderers: they map the
generic ``result.kind`` to a widget and apply ``effects``/``refresh`` —
they never know what ``/skills`` or ``/kanban`` *mean*.

Transports are shells over :func:`run_command`:

* ACP JSON-RPC method ``session/command`` (TUI + Telegram).
* Gateway REST ``POST /control/v1/{agent}/sessions/{sid}/command`` (web).

Both MUST return byte-identical :meth:`CommandResponse.to_dict` output for
the same input — there is one implementation (this module) and many
transport adapters.

The render contract (``kind`` taxonomy, ``effects``, ``actions``) was
designed so a generic renderer can display any command without
command-specific branches. ``kind`` is purely *what to render*; imperative
consequences (state patches, prompt submission, session switch) ride in
``effects``.

Blocking work (sqlite, disk scans, plugin/skill/mcp discovery, context
report) is offloaded with :func:`asyncio.to_thread` so the gateway event
loop never stalls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from sharedagents_agent.external_registry import is_external_engine_kind
from sharedagents_command.commands import (
    engine_model_rows_for_engine,
    format_help,
    format_model_list,
    format_status,
    local_model_rows,
    model_switch_text,
    normalize_context_subcommand,
    normalize_effort,
    normalize_mode,
    normalize_reasoning_level,
    normalize_thinking,
    normalize_tool_output,
    parse_command,
    resolve_model_row,
    resolve_session_model,
)
from sharedagents_model.models import (
    derive_effort_capability,
    effort_capability_confirmed,
)

SCHEMA_VERSION = 1

# Declared per-frontend capability boundaries (a spec, not a hidden gap).
# ``ask_mode`` needs an outbound permission / question channel. Web has
# permission cards and ``POST /api/sessions/<sid>/answer``, so it is ask-capable
# (same as TUI). Unknown frontends still fail closed. plan/auto work everywhere.
FRONTEND_CAPABILITIES: dict[str, dict[str, bool]] = {
    "tui": {"ask_mode": True, "interactive_modals": True},
    "telegram": {"ask_mode": True, "interactive_modals": False},
    "web": {"ask_mode": True, "interactive_modals": True},
    # WhatsApp answers permission prompts by numbered reply (sharedagents_whatsapp
    # interactions), so it has an outbound permission channel; no modals.
    "whatsapp": {"ask_mode": True, "interactive_modals": False},
}


def enforce_mode_capability(frontend: str, mode: str) -> str | None:
    """Single frontend-aware capability choke-point for mode-setting.

    EVERY mode-setting entry point routes through here so the ask-mode
    capability boundary cannot be bypassed on any one wire surface:

    * ``_cmd_mode`` (the ``session/command`` slash path),
    * the gateway REST ``permission_mode`` body field
      (``control_endpoint._handle_prompt_submit``), and
    * the raw ACP JSON-RPC ``session/set_mode`` method
      (``sharedagents_acp.server.server._handle_session_set_mode``).

    ``ask`` mode needs an interactive outbound permission channel to prompt the
    user per tool call. Eligibility is driven by ``FRONTEND_CAPABILITIES`` and
    FAILS CLOSED: an unknown/undeclared frontend has no ``ask_mode`` capability,
    so ``ask`` is refused rather than silently downgraded. Web, TUI,
    Telegram and WhatsApp advertise ``ask_mode``. ``plan``/``auto`` work on every frontend
    and are never refused here.

    Returns a human-readable refusal reason when the (frontend, mode) pair is
    not permitted, or ``None`` when the mode change is allowed. Every
    mode-change path must go through this check.
    """
    if mode == "ask" and not FRONTEND_CAPABILITIES.get(
        frontend,
        {},
    ).get("ask_mode", False):
        return "ask-mode isn't available on this frontend (no interactive permission channel). Use plan or build."
    return None


def resolve_effective_frontend(
    transport_frontend: str,
    requested_frontend: str | None,
) -> str:
    """Resolve the frontend used for a capability check WITHOUT ever widening.

    ``transport_frontend`` is the AUTHENTICATED transport identity — the seam
    that cannot be spoofed by request content. The web/gateway HTTP transport is
    authoritatively ``"web"`` (ask-capable via permission cards + answer POST);
    an ACP connection's identity is pinned at ``initialize`` (see ``AcpServer``).

    ``requested_frontend`` is a caller-supplied hint (a request-body / per-request
    ``frontend`` field). It may
    only NARROW capability — select a frontend no more capable than the transport
    — never WIDEN it. A ``web`` caller can therefore never claim ``tui`` to unlock
    ask-mode: any request that would grant a capability the transport lacks is
    ignored and the transport identity stands (fails closed).
    """
    if not requested_frontend or not isinstance(requested_frontend, str):
        return transport_frontend
    if requested_frontend == transport_frontend:
        return transport_frontend
    t_caps = FRONTEND_CAPABILITIES.get(transport_frontend, {})
    r_caps = FRONTEND_CAPABILITIES.get(requested_frontend, {})
    # Reject any request that would WIDEN capability beyond the transport: if the
    # requested frontend grants a capability the transport does not, the request
    # is a widening attempt — discard it and keep the authoritative transport.
    for _cap in ("ask_mode", "interactive_modals"):
        if r_caps.get(_cap, False) and not t_caps.get(_cap, False):
            return transport_frontend
    return requested_frontend


# Render kinds — frontends implement ONE renderer per kind, never per command.
KIND_TEXT = "text"
KIND_KEYVALS = "keyvals"
KIND_LIST = "list"
KIND_TABLE = "table"
KIND_CHECKLIST = "checklist"
KIND_DETAIL = "detail"
KIND_ERROR = "error"

# Effect types — imperative directives the frontend applies after rendering.
EFFECT_STATE_PATCH = "state_patch"  # apply scoped display/session deltas
EFFECT_PROMPT = "prompt"  # submit this text as a normal chat turn
EFFECT_SWITCH_SESSION = "switch_session"  # rebind the UI to another session
EFFECT_NEW_SESSION = "new_session"  # start a fresh session
EFFECT_CANCEL = "cancel"  # turn was cancelled
EFFECT_SWITCH_AGENT = "switch_agent"  # rebind the UI to another agent

# /goal actions that MUTATE the durable goal or its session projection.
# Successful runs of these emit a state_patch so the originating page's goal
# widget reflects the new state immediately (clear → hidden, pause → bound
# paused + Resume, resume/restart/archive/topup/budget → updated snapshot).
# Read-only list/status/show stay patch-free.
_GOAL_MUTATING_ACTIONS = frozenset(
    {
        "set",
        "clear",
        "pause",
        "resume",
        "restart",
        "archive",
        "topup",
        "budget",
        "met",
    }
)

# Allowed state_patch keys (typed/scoped — frontends apply to local view only).
# This is the AUTHORITATIVE allowlist: every state_patch effect is built through
# ``_state_patch_effect`` (below), which drops any key not listed here. It must
# be the union of every key a handler legitimately ships — the session/display
# toggles, the teacher block (``_teacher_state``), AND the goal/cost tracker
# (``_goal_state`` → ``_cmd_goal``). It mirrors the frontend's own enforced
# allowlist (``sharedagents_web/static/app.js`` ``STATE_PATCH_KEYS``); keep the two in
# sync.
STATE_PATCH_KEYS = frozenset(
    {
        # session / display toggles
        "mode",
        "reasoning_visibility",
        "thinking_enabled",
        "tool_output_visible",
        "tool_output",  # three-state tool-output level (off|on|stream) for display
        "model",
        "endpoint",
        "context_window",
        "effort",
        "effort_selected",
        "speed",
        # teacher-supervision block (_teacher_state)
        "teacher_enabled",
        "teacher_model",
        "teacher_active",
        "teacher_override",
        # goal-progress + cost tracker (_goal_state / cost_snapshot_metrics)
        "active_goal",
        "goal_id",
        "goal_terminal",
        "goal_turn_count",
        "max_goal_turns",
        "session_cost_usd",
        "goal_cost_usd",
        "goal_max_cost_usd",
        "goal_wait",
    }
)


# ---------------------------------------------------------------------------
# Wire contract (ed envelope)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Action:
    """A generic, user-triggerable command the frontend renders as a control.

    The frontend submits ``command`` back through ``run_command`` — it never
    needs to know what the command does.
    """

    id: str
    label: str
    command: str
    style: str = "secondary"  # primary | secondary | danger
    confirm: bool = False
    disabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "command": self.command,
            "style": self.style,
            "confirm": self.confirm,
            "disabled": self.disabled,
        }


@dataclass(slots=True)
class Effect:
    """An imperative consequence the frontend applies (not render data)."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "payload": self.payload}


@dataclass(slots=True)
class RenderResult:
    """What to render for one command response, nothing about what to do.

    ``kind`` is a small, closed taxonomy; a generic frontend renderer
    displays any command from ``kind`` alone, never branching on which
    command produced it.
    """

    kind: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    actions: list[Action] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "data": self.data,
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass(slots=True)
class CommandResponse:
    """The wire envelope every transport returns for one command.

    Carries an explicit ``schema_version`` so a future incompatible change
    to this shape can be detected by clients, even before any current
    consumer inspects the value.
    """

    ok: bool
    command: str
    result: RenderResult
    schema_version: int = SCHEMA_VERSION
    effects: list[Effect] = field(default_factory=list)
    refresh: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "command": self.command,
            "result": self.result.to_dict(),
            "effects": [e.to_dict() for e in self.effects],
            "refresh": list(self.refresh),
            "warnings": list(self.warnings),
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        return payload


# -- response factories ------------------------------------------------------


def _resp(
    command: str,
    kind: str,
    *,
    text: str = "",
    data: dict | None = None,
    actions: list[Action] | None = None,
    effects: list[Effect] | None = None,
    refresh: list[str] | None = None,
    warnings: list[str] | None = None,
    ok: bool = True,
) -> CommandResponse:
    return CommandResponse(
        ok=ok,
        command=command,
        result=RenderResult(kind=kind, text=text, data=data or {}, actions=actions or []),
        effects=effects or [],
        refresh=refresh or [],
        warnings=warnings or [],
    )


def _err(command: str, message: str, *, code: str = "bad_request") -> CommandResponse:
    return CommandResponse(
        ok=False,
        command=command,
        result=RenderResult(kind=KIND_ERROR, text=message),
        error=message,
        error_code=code,
    )


def _text(command: str, text: str) -> CommandResponse:
    return _resp(command, KIND_TEXT, text=text)


def _state_patch_effect(patch: dict[str, Any]) -> Effect:
    """Build the one ``state_patch`` Effect, ENFORCING ``STATE_PATCH_KEYS``.

    The single construction site for ``EFFECT_STATE_PATCH``. It filters the
    patch to the declared allowlist so ``STATE_PATCH_KEYS`` is authoritative
    rather than decorative: any key not in the set is dropped (silently, like
    the frontend's own ``applyStatePatch``), so a handler can never ship an
    out-of-band key into a frontend's local view. Legitimate keys — every key
    ``_teacher_state``/``_goal_state`` emit — are in the allowlist and pass
    through unchanged, so behavior is preserved for real patches.
    """
    safe = {k: v for k, v in (patch or {}).items() if k in STATE_PATCH_KEYS}
    return Effect(EFFECT_STATE_PATCH, {"patch": safe})


# ---------------------------------------------------------------------------
# Command context — the backend builds this from live session state.
# ---------------------------------------------------------------------------


@dataclass
class CommandContext:
    """Everything :func:`run_command` needs, provided by the agent backend.

    ``live_session`` is the in-memory :class:`sharedagents_session.session.Session`
    (or ``None`` for a read-only/no-session call, in which case disk is the
    fallback). The optional async callables are bound to the backend so
    mutating commands (mode/model/reasoning/cancel/list) act on real state.
    """

    cfg: Any
    agent_name: str
    session_id: str = ""
    frontend: str = "api"
    live_session: Any = None
    active_turn_id: str | None = None
    # The frontend's explicit confirmation for an irreversible command
    # (the slash-command parallel of the session-delete re-challenge). When
    # False, an irreversible command from the ``web`` frontend is re-challenged
    # rather than executed; a non-web frontend is unaffected.
    confirmed: bool = False

    set_mode_fn: Callable[[str, str], Awaitable[None]] | None = None
    set_config_option_fn: Callable[[str, str, Any], Awaitable[None]] | None = None
    list_sessions_fn: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None
    cancel_fn: Callable[..., Awaitable[None]] | None = None
    monitors_fn: Callable[[], list[dict[str, Any]]] | None = None
    inject_fn: Callable[[str, str], Awaitable[Any]] | None = None
    # Durable goal control — bound to the backend's execute_goal_command
    # adapter (registers in goals.db + syncs the engine). Takes the raw
    # ``/goal`` argument string, returns a dict with at least ``ok`` and
    # ``message`` (and optionally ``goal`` / ``conflicts``).
    goal_command_fn: Callable[[str], Awaitable[dict[str, Any]]] | None = None
    # Durable loop control — bound to the backend's execute_loop_command
    # adapter (registers in loops.db + kicks the loop supervisor). Takes the
    # raw ``/loop`` argument string, returns a dict with at least ``ok`` and
    # ``message`` (and optionally ``loop`` / ``note`` / ``loops``).
    loop_command_fn: Callable[[str], Awaitable[dict[str, Any]]] | None = None

    @property
    def agent_home(self) -> Path | None:
        return getattr(self.cfg, "home", None)

    @property
    def sessions_dir(self) -> Path | None:
        return getattr(self.cfg, "sessions_dir", None)


# ---------------------------------------------------------------------------
# Session-view helper — prefer live in-memory session, fall back to disk.
# ---------------------------------------------------------------------------


def _sess_attr(ctx: CommandContext, name: str, default: Any = None) -> Any:
    s = ctx.live_session
    if s is not None:
        return getattr(s, name, default)
    return default


def _load_disk_view(ctx: CommandContext) -> dict[str, Any]:
    """Fold the persisted session JSON (header + turns) for ctx.session_id."""
    if not ctx.session_id or ctx.sessions_dir is None:
        return {}
    try:
        from sharedagents_session.session_reader import load_session_dict

        raw = load_session_dict(ctx.sessions_dir, ctx.session_id, expected_agent=ctx.agent_name)
        return raw or {}
    except Exception:
        return {}


def _effective_engine(ctx: CommandContext) -> str:
    """Engine whose model catalogue this agent owns.

    The catalogue is a property of the agent's configured engine, not a
    leftover session header. A conversation founded on OpenCode keeps
    ``engine: opencode_acp`` on the founding session id after ``/agent
    jarvis`` or ``/agent cursor``; using that header here listed OpenCode
    models under those agents.
    """
    return (getattr(ctx.cfg, "engine", "") or "").strip()


# ---------------------------------------------------------------------------
# Individual command handlers (async; blocking work offloaded).
# ---------------------------------------------------------------------------


async def _cmd_help(ctx: CommandContext, arg: str) -> CommandResponse:
    return _text("help", format_help())


async def _cmd_status(ctx: CommandContext, arg: str) -> CommandResponse:
    s = ctx.live_session
    if s is not None:
        view = {
            "model": getattr(s, "model", "") or "(none)",
            "endpoint": getattr(s, "endpoint", "") or "(none)",
            "backend": getattr(s, "backend", "") or "?",
            "mode": getattr(s, "system_mode", "") or "auto",
            "reasoning": getattr(s, "reasoning_visibility", "off"),
            # Empty session override → "default" so frontends apply the
            # *model-specific* capability default (e.g. DeepSeek's own
            # default thinking depth). The agent/global effort still applies
            # at the request seam when the override is cleared; do not
            # surface it here as a fake override.
            "effort": getattr(s, "effort", "") or "default",
            "tool": "on" if getattr(s, "tool_output_visible", False) else "off",
            "hash": getattr(s, "hash", "") or ctx.session_id,
        }
    else:
        d = await asyncio.to_thread(_load_disk_view, ctx)
        view = {
            "model": d.get("model") or "(none)",
            "endpoint": d.get("endpoint") or "(none)",
            "backend": d.get("backend") or "?",
            "mode": d.get("system_mode") or "auto",
            "reasoning": "off",
            "effort": d.get("effort") or "default",
            "tool": "off",
            "hash": d.get("hash") or ctx.session_id,
        }
    data = {
        "agent": ctx.agent_name,
        "upstream": view["endpoint"],
        "backend": view["backend"],
        "model": view["model"],
        "mode": view["mode"],
        "reasoning": view["reasoning"],
        "effort": view["effort"],
        "tool": view["tool"],
        "session": (ctx.session_id[:12] if ctx.session_id else "(none)"),
        "active_turn": ctx.active_turn_id or "(none)",
    }
    row = _active_effort_model_row(ctx)
    if row is not None:
        from sharedagents_model import effort_prefs

        data["effort_selected"] = effort_prefs.selected_level_for(
            row,
            getattr(ctx.cfg, "effort", ""),
            engine_provider=effort_prefs.engine_provider(ctx.cfg, ctx.agent_name),
            session_model=view["model"],
            session_effort=view["effort"],
        )
    # Fold the EFFECTIVE teacher-supervision state into /status so every
    # frontend can render a single status snapshot that includes the
    # teacher indicators (teacher_active / teacher_enabled / teacher_model)
    # without a second round-trip to /teacher. Best-effort — a config that
    # can't compute teacher state must not break the status command.
    try:
        data.update(_teacher_state(ctx))
    except Exception:  # noqa: BLE001
        pass
    # Fold the goal-progress snapshot in too so a single /status carries the
    # active goal text AND the outer-loop tracker (terminal state + turns
    # consumed / budget). Same single-source contract as the teacher block:
    # every frontend renders straight from these keys, none re-derives them.
    try:
        data.update(_goal_state(ctx))
    except Exception:  # noqa: BLE001
        pass
    pairs = [[k, str(v)] for k, v in data.items()]
    return _resp("status", KIND_KEYVALS, text=format_status(data), data={"pairs": pairs, **data})


async def _cmd_context(ctx: CommandContext, arg: str) -> CommandResponse:
    sub = normalize_context_subcommand(arg)
    from sharedagents_agent.context_report import (
        SessionTokens,
        build_context_report,
        format_context_detail,
        format_context_help,
        format_context_json,
        format_context_list,
    )

    if not sub or sub == "help":
        return _text("context", format_context_help())
    if sub not in {"list", "show", "detail", "deep", "json"}:
        return _err("context", "Unknown /context mode. Use: /context, /context list, /context detail, or /context json")

    def _build() -> tuple[str, dict[str, Any]]:
        d = _load_disk_view(ctx)
        turns = list(d.get("turns") or [])
        ctx_window = int(d.get("context_window") or 0)
        model = d.get("model") or _sess_attr(ctx, "model", "") or ""
        system_prompt = d.get("system_prompt") or _sess_attr(ctx, "system_prompt", "") or ""
        permission_mode = d.get("system_mode") or _sess_attr(ctx, "system_mode", "") or ""
        workspace_cwd = d.get("workspace_cwd") or _sess_attr(ctx, "workspace_cwd", "") or ""
        latest: dict[str, Any] = {}
        for t in reversed(turns):
            m = t.get("metrics") if isinstance(t, dict) else None
            if isinstance(m, dict) and m:
                latest = m
                break
        tools_schemas: list[dict[str, Any]] = []
        tool_summaries: list[tuple[str, str]] = []
        deferred_catalog_text = ""
        try:
            from sharedagents_agent.tools import BUILTIN_TOOLS, ToolRegistry

            registry = ToolRegistry(BUILTIN_TOOLS)
            tools_schemas = registry.openai_schemas()
            tool_summaries = [(t.name, t.description or "") for t in registry.tools]
            deferred_catalog_text = registry.deferred_catalog_block()
        except Exception:
            pass
        agent_home = ctx.agent_home
        skills_block_text = ""
        skills_list: list[Any] = []
        try:
            from sharedagents_skill.skills import discover_skills, skills_summary_block

            _sk_allow = getattr(ctx.cfg, "skill_allowlist", None)
            _sk_strict = bool(getattr(ctx.cfg, "skill_allowlist_strict", False))
            skills_block_text = skills_summary_block(agent_home, allowlist=_sk_allow, allowlist_strict=_sk_strict)
            skills_list = list(discover_skills(agent_home, allowlist=_sk_allow, allowlist_strict=_sk_strict))
        except Exception:
            pass
        if agent_home is None:
            from sharedagents_agent.agent import sharedagents_root

            agent_home = sharedagents_root()
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

        def _as_int(v: Any) -> int | None:
            return int(v) if isinstance(v, (int, float)) else None

        session_tok = SessionTokens(
            total_tokens=_as_int(latest.get("prompt_tokens")),
            total_tokens_fresh=_as_int(latest.get("prompt_tokens_fresh")),
            input_tokens=_as_int(latest.get("prompt_tokens")),
            output_tokens=_as_int(latest.get("completion_tokens")),
            context_tokens=ctx_window or None,
        )
        if sub == "json":
            text_out = format_context_json(report, session_tok)
        elif sub in ("detail", "deep"):
            text_out = format_context_detail(report, session_tok)
        else:
            text_out = format_context_list(report, session_tok)
        data = {
            "model": report.model,
            "system_prompt_chars": report.system_prompt_chars,
            "tool_schema_chars": report.tool_schema_chars,
            "skills_count": len(report.skills_entries),
            "tool_count": len(report.tool_entries),
            "session_total_tokens": session_tok.total_tokens,
            "context_window": session_tok.context_tokens,
        }
        return text_out, data

    text_out, data = await asyncio.to_thread(_build)
    return _resp("context", KIND_DETAIL, text=text_out, data={"title": f"context ({sub})", "body": text_out, **data})


async def _cmd_todo(ctx: CommandContext, arg: str) -> CommandResponse:
    def _load() -> list[dict[str, Any]]:
        if ctx.sessions_dir is None or not ctx.session_id:
            return []
        try:
            from sharedagents_agent.tools import load_todos, todos_path_for_session

            return load_todos(todos_path_for_session(ctx.sessions_dir, ctx.session_id))
        except Exception:
            return []

    todos = await asyncio.to_thread(_load)
    items = []
    for t in todos:
        if not isinstance(t, dict):
            continue
        items.append(
            {
                "content": t.get("content") or t.get("text") or t.get("body") or "",
                "status": t.get("status") or "pending",
                "activeForm": t.get("activeForm") or "",
            }
        )
    if items:
        lines = []
        glyph = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}
        for it in items:
            lines.append(f"{glyph.get(it['status'], '[ ]')} {it['content']}")
        text_out = "todos:\n" + "\n".join(lines)
    else:
        text_out = "no todos for this session"
    return _resp("todo", KIND_CHECKLIST, text=text_out, data={"items": items})


async def _cmd_skills(ctx: CommandContext, arg: str) -> CommandResponse:
    name = (arg or "").strip()

    def _all() -> list[Any]:
        from sharedagents_skill.skills import discover_skills

        return list(discover_skills(ctx.agent_home))

    if name:

        def _one() -> Any:
            from sharedagents_skill.skills import find_skill

            return find_skill(name, ctx.agent_home)

        sk = await asyncio.to_thread(_one)
        if sk is None:
            return _err("skill", f"skill not found: {name}", code="not_found")
        body = getattr(sk, "body", "") or ""
        return _resp(
            "skill",
            KIND_DETAIL,
            text=f"# {sk.name}\n{sk.description}\n\n{body}",
            data={
                "title": sk.name,
                "subtitle": sk.description,
                "body": body,
                "fields": [["source", str(getattr(sk, "source", ""))], ["version", getattr(sk, "version", "")]],
            },
        )
    skills = await asyncio.to_thread(_all)
    items = [
        {
            "id": s.name,
            "title": s.name,
            "subtitle": s.description,
            "actions": [Action(id=f"open-{s.name}", label="open", command=f"/skill {s.name}").to_dict()],
        }
        for s in skills
    ]
    text_out = (
        "installed skills:\n" + "\n".join(f"  {s.name} — {s.description}" for s in skills)
        if skills
        else "no skills installed"
    )
    return _resp("skills", KIND_LIST, text=text_out, data={"title": "skills", "items": items})


async def _cmd_mcp(ctx: CommandContext, arg: str) -> CommandResponse:
    flt = (arg or "").strip().lower()

    def _list() -> list[tuple[Any, str]]:
        try:
            from sharedagents_mcp.mcp_client import list_mcp_servers_with_source
        except Exception:
            return []
        try:
            return list(list_mcp_servers_with_source())
        except Exception:
            return []

    rows = await asyncio.to_thread(_list)
    items = []
    for srv, source in rows:
        nm = getattr(srv, "name", "") or "?"
        if flt and flt not in nm.lower():
            continue
        transport = getattr(srv, "transport", "") or "?"
        desc = getattr(srv, "description", "") or ""
        target = getattr(srv, "url", "") or getattr(srv, "command", "") or ""
        items.append(
            {
                "id": nm,
                "title": nm,
                "subtitle": f"{transport} · {source}",
                "body": desc or target,
                "badges": [transport, source],
            }
        )
    text_out = (
        "mcp servers:\n" + "\n".join(f"  {i['title']} ({i['subtitle']})" for i in items) if items else "no mcp servers"
    )
    return _resp("mcp", KIND_LIST, text=text_out, data={"title": "mcp servers", "items": items})


async def _cmd_commands(ctx: CommandContext, arg: str) -> CommandResponse:
    flt = (arg or "").strip().lower()

    def _list() -> list[Any]:
        from sharedagents_command.file_commands import discover_commands

        return list(discover_commands(ctx.agent_home))

    cmds = await asyncio.to_thread(_list)
    items = []
    for c in cmds:
        if flt and flt not in c.name.lower():
            continue
        items.append(
            {
                "id": c.name,
                "title": f"/{c.name}",
                "subtitle": c.description,
                "body": getattr(c, "argument_hint", ""),
            }
        )
    text_out = (
        "file commands:\n" + "\n".join(f"  /{c.name} — {c.description}" for c in cmds) if cmds else "no file commands"
    )
    return _resp("commands", KIND_LIST, text=text_out, data={"title": "file commands", "items": items})


async def _cmd_plugin(ctx: CommandContext, arg: str) -> CommandResponse:
    parts = (arg or "").split()
    sub = parts[0].lower() if parts else "list"
    rest = " ".join(parts[1:]).strip()
    import sharedagents_agent.plugins as plugins

    if sub in ("", "list", "ls"):
        rows = await asyncio.to_thread(plugins.list_plugins)
        items = [
            {
                "id": p.name,
                "title": p.name,
                "subtitle": ("enabled" if p.enabled else "disabled"),
                "body": p.description,
                "badges": [p.version or "", p.source or ""],
                "actions": [
                    Action(id=f"dis-{p.name}", label="disable", command=f"/plugin disable {p.name}").to_dict()
                    if p.enabled
                    else Action(id=f"en-{p.name}", label="enable", command=f"/plugin enable {p.name}").to_dict(),
                    Action(
                        id=f"rm-{p.name}",
                        label="remove",
                        style="danger",
                        confirm=True,
                        command=f"/plugin remove {p.name}",
                    ).to_dict(),
                ],
            }
            for p in rows
        ]
        text_out = (
            "plugins:\n" + "\n".join(f"  {'●' if p.enabled else '○'} {p.name} — {p.description}" for p in rows)
            if rows
            else "no plugins installed"
        )
        return _resp("plugin", KIND_LIST, text=text_out, data={"title": "plugins", "items": items})

    if sub == "search":
        if not rest:
            return _err("plugin", "usage: /plugin search <query>")
        rows = await asyncio.to_thread(lambda: plugins.search(rest))
        items = [
            {
                "id": e.name,
                "title": e.name,
                "subtitle": e.source,
                "body": e.description,
                "badges": list(getattr(e, "tags", ()) or ()),
                "actions": [
                    Action(id=f"inst-{e.name}", label="install", command=f"/plugin install {e.name}").to_dict()
                ],
            }
            for e in rows
        ]
        return _resp(
            "plugin",
            KIND_LIST,
            text="\n".join(f"  {e.name} — {e.description}" for e in rows) or "no matches",
            data={"title": f"marketplace: {rest}", "items": items},
        )

    if sub in ("install", "enable", "disable", "remove"):
        if not rest:
            return _err("plugin", f"usage: /plugin {sub} <name>")
        try:
            if sub == "install":
                p = await asyncio.to_thread(lambda: plugins.install(rest))
                msg = f"installed plugin {p.name}"
            elif sub == "enable":
                p = await asyncio.to_thread(lambda: plugins.enable(rest))
                msg = f"enabled plugin {p.name}"
            elif sub == "disable":
                p = await asyncio.to_thread(lambda: plugins.disable(rest))
                msg = f"disabled plugin {p.name}"
            else:
                path = await asyncio.to_thread(lambda: plugins.remove(rest))
                msg = f"removed plugin {rest} ({path})"
        except Exception as exc:  # noqa: BLE001
            return _err("plugin", f"{sub} failed: {exc}", code="plugin_error")
        return _resp("plugin", KIND_TEXT, text=msg, refresh=["plugins"])

    return _err("plugin", f"unknown /plugin subcommand: {sub}")


async def _cmd_models(ctx: CommandContext, arg: str) -> CommandResponse:
    # Show only the options usable by THIS agent's engine. For external
    # CLI/ACP engines the local ``cfg.models`` rows don't apply, so the
    # engine's own menu (or an empty list → active-model-only) wins.
    engine_rows = engine_model_rows_for_engine(_effective_engine(ctx), ctx.cfg)
    rows = local_model_rows(ctx.cfg) if engine_rows is None else engine_rows
    rows = [dict(row) for row in rows if isinstance(row, dict)]
    # Chat-enter picker is sessionless web `/models`. Do not HTTP-probe
    # provider catalogues here — discover_effort_capabilities defaults to a
    # 4s timeout per unique endpoint, which is the 3s "command" stall in
    # DevTools. Bound `/models` (typed in a session) still discovers.
    slim_web_picker = ctx.frontend == "web" and not (ctx.session_id or "").strip()
    if rows:
        from sharedagents_model.effort_discovery import discover_effort_capabilities

        discovered = await asyncio.to_thread(
            discover_effort_capabilities,
            rows,
            ctx.cfg,
            probe=not slim_web_picker,
        )
        for row, capability in zip(rows, discovered):
            if capability is not None:
                row["effort"] = {
                    **(row.get("effort") if isinstance(row.get("effort"), dict) else {}),
                    **capability,
                }
    cur_ep = _sess_attr(ctx, "endpoint", "") or ""
    cur_model = (
        resolve_session_model(
            _sess_attr(ctx, "engine_model", "") or _sess_attr(ctx, "model", "") or "",
            ctx.cfg,
        )
        if is_external_engine_kind(_effective_engine(ctx))
        else (_sess_attr(ctx, "engine_model", "") or _sess_attr(ctx, "model", "") or "")
    )
    from sharedagents_model import effort_prefs

    prefs = effort_prefs.load_prefs()
    pref_provider = effort_prefs.engine_provider(ctx.cfg, ctx.agent_name)
    items = []
    default_key = str(getattr(ctx.cfg, "upstream_alias", "") or "").strip()
    if not default_key:
        default_key = str(getattr(ctx.cfg, "upstream_model", "") or "").strip()
    if not default_key:
        for m in rows:
            if isinstance(m, dict) and m.get("role") == "primary":
                default_key = str(m.get("alias") or m.get("id") or "").strip()
                if default_key:
                    break
    for m in rows:
        # Keep the picker actionable for legacy rows without an alias: the
        # configured wire id is also accepted by the command resolver.
        alias = m.get("alias") or m.get("id") or "?"
        model_id = str(m.get("id") or "")
        display = str(m.get("display") or alias)
        active = bool(model_id and model_id == str(cur_model))
        items.append(
            {
                "id": alias,
                "title": display,
                # Explicit selection metadata keeps the operator-facing alias
                # separate from the engine wire id. The WebUI must never recover
                # either field (or provider identity) by parsing presentation text.
                "alias": alias,
                "wire_id": model_id,
                "provider": str(m.get("provider") or m.get("backend") or "configured"),
                "default": bool(default_key and (alias == default_key or model_id == default_key)),
                # The level remembered for this provider/model ("" = none).
                "effort_pref": prefs.get(effort_prefs.row_key(m, pref_provider), ""),
                "effort_selected": effort_prefs.selected_level_for(
                    m,
                    getattr(ctx.cfg, "effort", ""),
                    engine_provider=pref_provider,
                    session_model=_sess_attr(ctx, "model", "") or "",
                    session_effort=_sess_attr(ctx, "effort", "") or "",
                    mode=_sess_attr(ctx, "system_mode", "") or getattr(ctx.cfg, "permission_mode", "auto"),
                ),
                "selection_capabilities": {
                    "reasoning": _picker_effort_capability(ctx, m),
                    "speed": m.get("speed") if isinstance(m.get("speed"), dict) else None,
                },
                "subtitle": f"{m.get('id') or '(missing id)'} · {m.get('backend') or '?'}",
                "body": (m.get("baseUrl") or "").rstrip("/"),
                "badges": [f"{int(m.get('contextWindow') or 0) // 1024}k" if m.get("contextWindow") else "?"],
                "active": active,
                "actions": [Action(id=f"use-{alias}", label="switch", command=f"/model {alias}").to_dict()],
            }
        )
    # Web picker only reads items[]. Skip the TUI transcript duplicate
    # (`text`) and per-row baseUrl (`body`). Typed `/models` in a bound
    # session still gets the full list for display.
    if slim_web_picker:
        for it in items:
            it.pop("body", None)
        return _resp("models", KIND_LIST, text="", data={"title": "models", "items": items})
    return _resp(
        "models",
        KIND_LIST,
        text=format_model_list(rows, current_endpoint=cur_ep, current_model=cur_model),
        data={"title": "models", "items": items},
    )


def _resolve_engine_model_row(
    rows: list[dict[str, Any]],
    target: str,
) -> dict[str, Any] | None:
    t = (target or "").strip().lower()
    if not t:
        return None
    for key in ("alias", "id", "display"):
        for row in rows:
            if str(row.get(key) or "").strip().lower() == t:
                return dict(row)
    return None


def _picker_effort_capability(
    ctx: CommandContext,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Show only a declared/advertised vocabulary that the wire can apply."""
    capability = derive_effort_capability(
        row,
        mode=_sess_attr(ctx, "system_mode", "") or getattr(ctx.cfg, "permission_mode", "auto"),
    )
    if is_external_engine_kind(_effective_engine(ctx)):
        return {**capability, "applies": True}
    applies = bool(capability.get("values")) and effort_capability_confirmed(capability)
    return {**capability, "applies": applies}


async def _effort_capability_for_row(
    ctx: CommandContext,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the row's live API capability, falling back visibly."""
    from sharedagents_model.effort_discovery import discover_effort_capability

    resolved = dict(row)
    capability = await asyncio.to_thread(
        discover_effort_capability,
        resolved,
        ctx.cfg,
    )
    if capability is not None:
        resolved["effort"] = {
            **(resolved.get("effort") if isinstance(resolved.get("effort"), dict) else {}),
            **capability,
        }
    return derive_effort_capability(
        resolved,
        mode=_sess_attr(ctx, "system_mode", "") or getattr(ctx.cfg, "permission_mode", "auto"),
    )


def _active_effort_model_row(ctx: CommandContext) -> dict[str, Any] | None:
    """Resolve the active model row for model-specific effort validation."""
    current_model = _sess_attr(ctx, "model", "") or ""
    engine_rows = engine_model_rows_for_engine(_effective_engine(ctx), ctx.cfg)
    if engine_rows is None:
        row = resolve_model_row(ctx.cfg, current_model) if current_model else None
        if row is None:
            try:
                row = dict(ctx.cfg._primary_model_row())
            except Exception:  # noqa: BLE001
                row = None
        return row
    return _resolve_engine_model_row(engine_rows, current_model)


def _same_model(row: Mapping[str, Any], model: str) -> bool:
    """True when ``model`` (a session's model) names ``row``."""
    want = (model or "").strip().lower()
    if not want:
        return False
    return any(
        str(row.get(key) or "").strip().lower() == want for key in ("id", "alias", "wire_id", "model_id", "display")
    )


_LOCAL_MODEL_BACKENDS = frozenset(
    {
        "vllm",
        "sglang",
        "local",
        "openai",
        "openai-compat",
        "bedrock",
    }
)


def _is_known_local_engine_model(target: str, cfg: Any) -> bool:
    """True when ``target`` names a local/vLLM catalogue row (or the ``local`` alias).

    Used to refuse applying a local-engine model onto an external ACP/CLI
    session at ``/model`` time, before the header is poisoned and the live
    ACP ``session_set_config_option`` fails mid-turn.
    """
    t = (target or "").strip().lower()
    if not t or t.startswith("http://") or t.startswith("https://"):
        return False
    if t == "local":
        return True
    rows = list(getattr(cfg, "models", None) or [])
    # Foreign engines no longer inherit global ``models[]`` on load.
    # Still refuse those ids here so ``/model openrouter/auto`` cannot
    # pass through as a raw ACP id on cursor/opencode/claude.
    engine = str(getattr(cfg, "engine", "") or "").strip()
    if engine and engine != "local_llm":
        try:
            from sharedagents_agent.agent import load_global_config

            glob = load_global_config()
            rows = rows + list(glob.get("models") or glob.get("upstream_endpoints") or [])
        except Exception:  # noqa: BLE001
            pass
    for row in rows:
        if not isinstance(row, dict):
            continue
        backend = str(row.get("backend") or "").strip().lower()
        if backend and backend not in _LOCAL_MODEL_BACKENDS:
            continue
        for key in ("alias", "id", "display"):
            if str(row.get(key) or "").strip().lower() == t:
                return True
    return False


async def _cmd_model(ctx: CommandContext, arg: str) -> CommandResponse:
    """Apply a model selection and optional effort/speed tuple atomically."""
    tokens = (arg or "").strip().split()
    target = tokens[0] if tokens else ""
    if not target:
        return _err("model", "usage: /model <alias|url>")
    options: dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            return _err("model", f"invalid model option {token!r}; use key=value", code="invalid_option")
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = unquote(value.strip())
        if key == "speed":
            value = value.lower()
        if key not in {"effort", "speed"} or key in options:
            return _err("model", f"unknown or duplicate model option {key!r}", code="invalid_option")
        options[key] = value
    engine_name = _effective_engine(ctx)
    engine_rows = engine_model_rows_for_engine(engine_name, ctx.cfg)
    is_default = target.lower() in ("default", "auto", "reset")
    if engine_rows is None:
        if is_default:
            row = ctx.cfg._primary_model_row() if hasattr(ctx.cfg, "_primary_model_row") else None
        else:
            row = resolve_model_row(ctx.cfg, target)
        if not row:
            return _err("model", f"{target} is not a model of {ctx.agent_name or 'this agent'}", code="not_found")
    else:
        if is_default:
            row = next((dict(r) for r in engine_rows if r.get("default")), None)
            if row is None:
                row = _resolve_engine_model_row(engine_rows, "default")
            if row is None:
                return _err("model", "default model is unavailable for this engine", code="not_found")
        else:
            row = _resolve_engine_model_row(engine_rows, target)
        if row is None:
            # External CLIs own their model-id space for true upstream ids.
            # Refuse known-foreign local/vLLM aliases so they cannot poison
            # an ACP session header and fail later at turn start.
            if _is_known_local_engine_model(target, ctx.cfg):
                return _err(
                    "model",
                    f"{target!r} is a local-engine model and cannot be applied to {engine_name or 'this'} engine",
                    code="not_found",
                )
            # Discovery covers the normal picker path, but a typed raw id must
            # still reach the CLI as-typed — INCLUDING when the engine's menu
            # is empty (generic ACP), whose id space is entirely upstream-owned
            row = {"alias": target, "id": target, "display": target}
    if ctx.set_config_option_fn is None:
        return _err("model", "model switch not available in this context", code="unsupported")

    # Validate the complete transaction before its first mutation.  The
    # WebUI uses this extension to submit model + effort + speed as one
    # acknowledged command.
    from sharedagents_model import effort_prefs

    pref_provider = effort_prefs.engine_provider(ctx.cfg, ctx.agent_name)
    prior_model = _sess_attr(ctx, "model", "") or ""
    prior_effort = _sess_attr(ctx, "effort", "") or ""
    effort_stored = None
    effort_patch = None
    effort_note = ""
    # The level to remember for this model after the switch is applied:
    # ``None`` leaves the store alone, ``"default"`` clears the entry.
    effort_remember = None
    if "effort" in options:
        requested = normalize_effort(options["effort"])
        capability = await _effort_capability_for_row(ctx, row)
        if requested is None or requested != "default" and requested not in capability["values"]:
            valid = "|".join(capability["values"])
            return _err(
                "model",
                f"invalid effort {options['effort']!r}; use {valid}|default",
                code="invalid_effort",
            )
        # A picked level is stored concretely, including the model's own
        # advertised default: ``""`` means "use the AGENT's effort", which
        # can be a different level. Only ``default`` stores ``""``.
        effort_stored = "" if requested == "default" else requested
        effort_patch = requested
        effort_remember = requested
    else:
        # No explicit level: load the level remembered for this model. A
        # session override set for another model does not follow the switch.
        remembered = effort_prefs.get_row_pref(row, pref_provider)
        if remembered:
            capability = await _effort_capability_for_row(ctx, row)
            if remembered in capability["values"]:
                effort_stored = remembered
                effort_patch = remembered
                effort_note = " (remembered for this model)"
        if effort_stored is None and prior_effort and not _same_model(row, prior_model):
            effort_stored = ""
            effort_patch = "default"

    speed_stored = None
    speed_patch = None
    if "speed" in options:
        speed_cap = row.get("speed") if isinstance(row.get("speed"), dict) else None
        speed_values = [str(v).strip().lower() for v in (speed_cap or {}).get("values", [])]
        speed_wire = str((speed_cap or {}).get("wire") or "").strip()
        requested_speed = options["speed"]
        if not speed_wire:
            return _err("model", "speed is not supported by this model", code="unsupported")
        if requested_speed not in speed_values:
            return _err(
                "model",
                f"invalid speed {requested_speed!r}; use {'|'.join(speed_values)}",
                code="invalid_speed",
            )
        speed_stored = "" if requested_speed in ("", "default", "auto") else requested_speed
        speed_patch = requested_speed
    # Local backends resolve configured aliases, while external engines own
    # canonical wire ids. Never hand an external CLI a display alias (for
    # example ``sonnet-5`` instead of ``claude-sonnet-5``): discovery caches
    # can differ across the web and gateway processes, and aliases are not a
    # stable transport contract. Local raw ids are still translated back to
    # their configured alias so resolve_model cannot fall back to primary.
    switch_target = target
    if engine_rows is not None:
        switch_target = str(row.get("id") or target).strip()
    elif not target.lower().startswith("http"):
        alias = str(row.get("alias") or (row.get("id") if is_default else "") or "").strip()
        if alias:
            switch_target = alias
    prior_speed = _sess_attr(ctx, "speed", "") or ""
    applied: list[tuple[str, Any]] = []
    try:
        # Record each attempted mutation before calling the adapter. An
        # adapter may update its live session and then fail while persisting
        # or acknowledging the change; that still needs the same rollback
        # protection as a failure raised before mutation.
        applied.append(("model", prior_model))
        await ctx.set_config_option_fn(ctx.session_id, "model", switch_target)
        if effort_stored is not None:
            applied.append(("effort", prior_effort))
            await ctx.set_config_option_fn(ctx.session_id, "effort", effort_stored)
        if speed_stored is not None:
            applied.append(("speed", prior_speed))
            await ctx.set_config_option_fn(ctx.session_id, "speed", speed_stored)
    except Exception as exc:  # noqa: BLE001
        for key, old_value in reversed(applied):
            try:
                await ctx.set_config_option_fn(ctx.session_id, key, old_value)
            except Exception:  # noqa: BLE001
                pass
        return _err("model", f"apply failed: {exc}", code="apply_failed")
    if effort_remember is not None:
        await asyncio.to_thread(
            effort_prefs.set_row_pref,
            row,
            effort_remember,
            pref_provider,
        )
    patch = {
        "model": row.get("id") or target,
        "endpoint": row.get("baseUrl") or row.get("endpoint") or "",
        "context_window": int(row.get("contextWindow") or 0),
    }
    if effort_patch is not None:
        patch["effort"] = effort_patch
    if speed_patch is not None:
        patch["speed"] = speed_patch
    patch["effort_selected"] = effort_prefs.selected_level_for(
        row,
        getattr(ctx.cfg, "effort", ""),
        engine_provider=pref_provider,
        session_model=str(row.get("id") or target),
        session_effort=effort_stored if effort_stored is not None else prior_effort,
    )
    text = model_switch_text(row)
    if effort_patch is not None:
        text += f"\neffort → {effort_patch}{effort_note}"
    if speed_patch is not None:
        text += f"\nspeed → {speed_patch}"
    return _resp("model", KIND_TEXT, text=text, effects=[_state_patch_effect(patch)], refresh=["session"])


_NATIVE_CONVERSATIONAL = frozenset({"jarvis", "build", "plan", "research"})
_NATIVE_NON_CONVERSATIONAL = frozenset(
    {
        "explore",
        "general",
        "docs",
        "compaction",
        "title",
        "summary",
        # review is auto-only (read-only verdict gate). This set pins the TUI
        # ``/mode`` response ONLY — unlike sharedagents_web.agent_visibility's
        # NON_CONVERSATIONAL_AGENTS, it does not gate any picker, so adding here
        # does not hide the agent. Do not "sync" it with that Web list.
        "review",
    }
)


# Native agents that the legacy TUI ``/mode`` path toggles between as
# agents (OpenCode Tab) via :func:`_native_tab_agent`.
_NATIVE_MODE_PROFILES: frozenset[str] = frozenset({"plan", "build"})


def _native_tab_agent(current: str, requested: str) -> str | None:
    """Legacy helper: Plan/Build remain agents, not permission modes."""
    cur = (current or "").strip().lower()
    want = (requested or "").strip().lower()
    if want in {"plan", "architect"}:
        return "plan"
    if want in {"build"}:
        return "build"
    if want in {"", "tab", "cycle"}:
        return "plan" if cur == "build" else "build"
    return None


async def _cmd_mode(ctx: CommandContext, arg: str) -> CommandResponse:
    from sharedagents_agent.engine_modes import EngineModeRegistry

    agent = str(ctx.agent_name or "").strip().lower()
    engine = str(getattr(ctx.cfg, "engine", "") or "").strip().lower()
    if agent in _NATIVE_CONVERSATIONAL and engine in {"", "local_llm"}:
        target = (arg or "").strip().lower() or "auto"
        if target in {"allow-all", "allowall", "allow all", "allow_all"}:
            target = "auto"
        if target in {"auto", "ask"}:
            mode = normalize_mode(target)
            if ctx.set_mode_fn is not None and mode:
                await ctx.set_mode_fn(ctx.session_id, mode)
            return _resp(
                "mode",
                KIND_TEXT,
                text=f"mode → {mode or target}",
                effects=[_state_patch_effect({"mode": mode or target})],
                refresh=["session"],
            )
        return _err("mode", "native conversational agents use auto|ask")
    if agent in _NATIVE_NON_CONVERSATIONAL and engine in {"", "local_llm"}:
        return _resp("mode", KIND_TEXT, text="mode → auto")

    cur = _sess_attr(ctx, "system_mode", "") or "auto"
    caps = EngineModeRegistry().discover(ctx.agent_name, ctx.cfg)
    allowed = [mode.requested_mode_id for mode in caps.modes if mode.supported]
    if not allowed:
        allowed = ["auto"]
    hint = "|".join(allowed)
    target = (arg or "").strip()
    if not target:
        try:
            nxt = allowed[(allowed.index(cur) + 1) % len(allowed)]
        except ValueError:
            nxt = allowed[0]
        target = nxt
    mode = normalize_mode(target)
    if mode is None or mode not in allowed:
        return _err("mode", f"unknown mode: {target}. use {hint}")
    # Route through the single frontend-aware capability choke-point so this
    # path enforces IDENTICALLY to the REST ``permission_mode`` body field and
    # the raw ACP ``session/set_mode`` method — no wire surface can bypass the
    # ask-mode boundary. Fails closed
    # on unknown frontends; plan/auto work everywhere.
    _refusal = enforce_mode_capability(ctx.frontend, mode)
    if _refusal is not None:
        return _err("mode", _refusal, code="unsupported")
    if ctx.set_mode_fn is not None:
        try:
            await ctx.set_mode_fn(ctx.session_id, mode)
        except Exception as exc:  # noqa: BLE001
            return _err("mode", f"set mode failed: {exc}", code="set_failed")
    return _resp(
        "mode", KIND_TEXT, text=f"mode → {mode}", effects=[_state_patch_effect({"mode": mode})], refresh=["session"]
    )


async def _cmd_reasoning(ctx: CommandContext, arg: str) -> CommandResponse:
    if not arg:
        cur = _sess_attr(ctx, "reasoning_visibility", "off")
        return _resp("reasoning", KIND_TEXT, text=f"reasoning: {cur}")
    level = normalize_reasoning_level(arg)
    if level is None:
        return _err("reasoning", f"unknown reasoning value {arg!r}. use on|off|stream")
    if ctx.set_config_option_fn is not None:
        try:
            await ctx.set_config_option_fn(ctx.session_id, "reasoning", level)
        except Exception:
            pass
    return _resp(
        "reasoning",
        KIND_TEXT,
        text=f"reasoning → {level}",
        effects=[_state_patch_effect({"reasoning_visibility": level})],
    )


async def _cmd_effort(ctx: CommandContext, arg: str) -> CommandResponse:
    """Show or set the session's reasoning-depth override."""
    current = _sess_attr(ctx, "effort", "")
    if not current and ctx.live_session is None:
        disk = await asyncio.to_thread(_load_disk_view, ctx)
        current = disk.get("effort") or ""
    current = current or getattr(ctx.cfg, "effort", "") or "default"
    if not arg:
        return _resp("effort", KIND_TEXT, text=f"effort: {current}")
    level = normalize_effort(arg)
    if level is None:
        return _err(
            "effort",
            f"unknown effort value {arg!r}. use a level advertised by /models or default",
        )
    if level != "default":
        row = _active_effort_model_row(ctx)
        capability = await _effort_capability_for_row(ctx, row) if row is not None else {"values": []}
        if level not in capability["values"]:
            return _err(
                "effort",
                f"effort {level!r} is not supported by the active model; use {'|'.join(capability['values'])}|default",
                code="invalid_effort",
            )
    stored = "" if level == "default" else level
    if ctx.set_config_option_fn is not None:
        try:
            await ctx.set_config_option_fn(ctx.session_id, "effort", stored)
        except Exception as exc:  # noqa: BLE001
            return _err("effort", f"set effort failed: {exc}", code="set_failed")
    # Remember the level for the active model (``default`` forgets it).
    remember_row = row if level != "default" else _active_effort_model_row(ctx)
    if remember_row is not None:
        from sharedagents_model import effort_prefs

        await asyncio.to_thread(
            effort_prefs.set_row_pref,
            remember_row,
            level,
            effort_prefs.engine_provider(ctx.cfg, ctx.agent_name),
        )
    return _resp(
        "effort",
        KIND_TEXT,
        text=f"effort → {level}",
        effects=[
            _state_patch_effect(
                {
                    "effort": level,
                    "effort_selected": effort_prefs.selected_level_for(
                        remember_row,
                        getattr(ctx.cfg, "effort", ""),
                        engine_provider=effort_prefs.engine_provider(ctx.cfg, ctx.agent_name),
                        session_model=str(remember_row.get("id") or ""),
                        session_effort=stored,
                    )
                    if remember_row is not None
                    else "",
                }
            )
        ],
        refresh=["session"],
    )


async def _cmd_thinking(ctx: CommandContext, arg: str) -> CommandResponse:
    if not arg:
        return _resp("thinking", KIND_TEXT, text="usage: /thinking [on|off]")
    val = normalize_thinking(arg)
    if val is None:
        return _err("thinking", f"unknown thinking value {arg!r}. use on or off")
    if ctx.set_config_option_fn is not None:
        try:
            await ctx.set_config_option_fn(ctx.session_id, "thinking", val)
        except Exception:
            pass
    return _resp(
        "thinking",
        KIND_TEXT,
        text=f"thinking → {'on' if val else 'off'}",
        effects=[_state_patch_effect({"thinking_enabled": val})],
    )


async def _cmd_tool(ctx: CommandContext, arg: str) -> CommandResponse:
    if not arg:
        cur = "on" if _sess_attr(ctx, "tool_output_visible", False) else "off"
        return _resp("tool", KIND_TEXT, text=f"tool output: {cur}")
    level = normalize_tool_output(arg)
    if level is None:
        return _err("tool", f"unknown tool value {arg!r}. use on|off")
    visible = level != "off"
    if ctx.set_config_option_fn is not None:
        try:
            await ctx.set_config_option_fn(ctx.session_id, "tool_output", level)
        except Exception:
            pass
    # Carry BOTH the legacy boolean (session field / other consumers) AND the
    # authoritative three-state level so the frontend mirrors on/off/stream
    # from the effect channel, never from parsing result.text.
    return _resp(
        "tool",
        KIND_TEXT,
        text=f"tool output → {level}",
        effects=[_state_patch_effect({"tool_output_visible": visible, "tool_output": level})],
    )


def _goal_state(
    ctx: CommandContext,
    *,
    goal_id_override: str | None = None,
) -> dict[str, Any]:
    """Assemble the goal-progress snapshot for this session.

    ONE source the goal widget (TUI bar, web tracker, Telegram /status)
    reads, so no frontend has to know how goal progress is computed.
    Returns:
      • ``active_goal``     — str, the completion-condition text ("" if none)
      • ``goal_terminal``   — str, "" | "met" | "blocked" | "paused"
      • ``goal_turn_count`` — int, re-drive turns consumed by the outer loop
      • ``max_goal_turns``  — int, the configured re-drive budget
      • ``goal_wait``       — str, why the goal stopped ("" while it runs)

    Live session wins; falls back to the persisted header on disk so a
    resumed/observed session still reports its progress. Best-effort —
    never raises into /status.
    """
    s = ctx.live_session
    disk: dict[str, Any] = {}
    if s is None:
        disk = _load_disk_view(ctx)

    def _field(name: str, default: Any) -> Any:
        if s is not None:
            return getattr(s, name, default)
        v = disk.get(name)
        return default if v is None else v

    def _safe_int(value: Any, default: int) -> int:
        """Coerce one persisted field to int WITHOUT poisoning the rest of the
        snapshot. A malformed ``goal_turn_count`` on disk (e.g. ``"abc"``)
        used to raise inside the dict literal, which the caller caught by
        dropping the ENTIRE goal block (active_goal/terminal/budget all gone).
        Each numeric field now falls back independently."""
        try:
            return int(value or 0)
        except Exception:  # noqa: BLE001
            return default

    # ``max_goal_turns`` is surfaced here as a DISPLAY-ONLY progress label — it
    # never terminates the goal loop (the loop is cost-terminated; see
    # goal_supervisor._drive_loop). An opt-in turn cap was rejected —
    # owner override 2026-07-11 "only cost limit no turn limit".
    max_turns = _safe_int(getattr(ctx.cfg, "max_goal_turns", 0), 0)
    # A post-command snapshot may have an authoritative goal identity that is
    # different from the pre-command live Session. ``None`` means no override;
    # the empty string is intentional (clear/archive detached the goal).
    goal_id = str(_field("goal_id", "") if goal_id_override is None else goal_id_override)
    out = {
        "active_goal": str(_field("active_goal", "") or ""),
        "goal_id": goal_id,
        "goal_terminal": str(_field("goal_terminal", "") or ""),
        "goal_turn_count": _safe_int(_field("goal_turn_count", 0), 0),
        "max_goal_turns": max_turns if max_turns > 0 else None,  # None → ∞ in UI
        # Why the goal stopped ("not achievable: …", "Goal paused after …"),
        # "" while it runs. Rendered verbatim.
        "goal_wait": "",
    }
    if goal_id:
        try:
            from sharedagents_goal.goal_registry import get_goal_registry

            _g = get_goal_registry().get(goal_id)
            if _g is not None:
                out["goal_wait"] = _g.wait_label
        except Exception:  # noqa: BLE001
            pass
    # Realtime cost: cumulative session + goal spend (and the goal's budget, or
    # None == ∞) so the web goal widget + session footer update each /status and
    # session-open without a frontend computing anything.
    try:
        from sharedagents_goal.goal_service import cost_snapshot_metrics

        root_sid = str(_field("cost_root_session_id", "") or getattr(ctx, "session_id", "") or _field("hash", ""))
        out.update(cost_snapshot_metrics(root_sid, goal_id))
    except Exception:  # noqa: BLE001
        pass
    return out


def _teacher_state(
    ctx: CommandContext, *, override: str | None = None, model_override: str | None = None
) -> dict[str, Any]:
    """Compute the EFFECTIVE teacher state for this session — the single
    source the indicators (TUI widget, Telegram /status, web badge) read.

    ``override`` / ``model_override`` let a just-applied toggle/model change
    be reflected without re-reading the not-yet-persisted session. The
    returned patch carries:
      • ``teacher_active`` — bool, the EFFECTIVE gate the indicators show/hide
        on (explicit ``/teacher on`` only when a ``/goal`` is active; otherwise
        ``require_goal == false`` may enable policy-wide supervision). Web +
        Telegram render straight from this.
      • ``teacher_override`` — str ("on"/"off"/""), the PERSISTED per-session
        toggle. Distinct from ``teacher_active`` on purpose: an active ``/goal``
        alone keeps override "" and ``teacher_active`` false — the goal Stop
        hook owns continuation. Explicit ``/teacher on`` sets override "on".
      • ``teacher_enabled`` — bool, back-compat echo == ``teacher_active``.
      • ``teacher_model``   — the effective supervising model label.
    """
    from sharedagents_agent.teacher_supervisor import (
        effective_teacher_model,
        is_teacher_active,
    )

    try:
        pol = (ctx.cfg.teacher_policy() if hasattr(ctx.cfg, "teacher_policy") else {}) or {}
    except Exception:  # noqa: BLE001
        pol = {}
    try:
        row = (ctx.cfg.teacher_model() if hasattr(ctx.cfg, "teacher_model") else None) or {}
    except Exception:  # noqa: BLE001
        row = {}
    ov = override if override is not None else str(_sess_attr(ctx, "teacher_override", "") or "")
    mov = model_override if model_override is not None else str(_sess_attr(ctx, "teacher_model_override", "") or "")
    goal = str(_sess_attr(ctx, "active_goal", "") or "")
    active = is_teacher_active(pol, ov, goal)
    model = effective_teacher_model(row, mov)
    return {"teacher_active": active, "teacher_enabled": active, "teacher_override": ov, "teacher_model": model}


async def _cmd_teacher(ctx: CommandContext, arg: str) -> CommandResponse:
    parts = (arg or "").split()
    if not parts:
        st = _teacher_state(ctx)
        label = (
            f"on ({st['teacher_model']})"
            if st["teacher_active"] and st["teacher_model"]
            else "on"
            if st["teacher_active"]
            else "off"
        )
        return _resp("teacher", KIND_TEXT, text=f"teacher: {label}", effects=[_state_patch_effect(st)])
    head = parts[0].lower()
    if head == "model":
        alias = " ".join(parts[1:]).strip()
        if not alias:
            return _err("teacher", "usage: /teacher model <alias>")
        if ctx.set_config_option_fn is not None:
            try:
                await ctx.set_config_option_fn(ctx.session_id, "teacher_model", alias)
            except Exception as exc:  # noqa: BLE001
                return _err("teacher", f"set teacher model failed: {exc}")
        _norm = "" if alias.lower() in ("default", "auto", "reset", "clear", "none") else alias
        st = _teacher_state(ctx, model_override=_norm)
        return _resp(
            "teacher",
            KIND_TEXT,
            text=f"teacher model → {st['teacher_model'] or 'default'}",
            effects=[_state_patch_effect(st)],
        )
    val = normalize_thinking(head)  # reuse on/off normalizer
    if val is None:
        return _err("teacher", f"unknown teacher value {head!r}. use on|off|model <alias>")
    if ctx.set_config_option_fn is not None:
        try:
            await ctx.set_config_option_fn(ctx.session_id, "teacher", "on" if val else "off")
        except Exception as exc:  # noqa: BLE001
            return _err("teacher", f"set teacher failed: {exc}")
    st = _teacher_state(ctx, override=("on" if val else "off"))
    label = (
        f"on ({st['teacher_model']})"
        if st["teacher_active"] and st["teacher_model"]
        else "on"
        if st["teacher_active"]
        else "off"
    )
    return _resp("teacher", KIND_TEXT, text=f"teacher → {label}", effects=[_state_patch_effect(st)])


async def _cmd_history(ctx: CommandContext, arg: str) -> CommandResponse:
    try:
        n = int((arg or "").strip()) if (arg or "").strip() else 10
    except ValueError:
        n = 10
    n = max(1, min(100, n))

    def _load() -> list[dict[str, Any]]:
        d = _load_disk_view(ctx)
        return list(d.get("turns") or [])

    turns = await asyncio.to_thread(_load)
    tail = turns[-n:]
    rows = []
    for t in tail:
        if not isinstance(t, dict):
            continue
        user = t.get("user") or ""
        asst = t.get("content") or t.get("assistant") or ""
        rows.append(
            {
                "user": (user[:200] if isinstance(user, str) else str(user)),
                "assistant": (asst[:200] if isinstance(asst, str) else str(asst)),
            }
        )
    text_out = "\n\n".join(f"▸ {r['user']}\n  {r['assistant']}" for r in rows) or "no turns yet"
    return _resp(
        "history",
        KIND_TABLE,
        text=text_out,
        data={"title": f"last {len(rows)} turns", "columns": ["user", "assistant"], "rows": rows},
    )


async def _cmd_sessions(ctx: CommandContext, arg: str) -> CommandResponse:
    target = (arg or "").strip()
    sessions: list[dict[str, Any]] = []
    if ctx.list_sessions_fn is not None:
        try:
            sessions = await ctx.list_sessions_fn() or []
        except Exception:
            sessions = []
    if not sessions:
        sessions = await asyncio.to_thread(_list_sessions_from_disk, ctx)

    # Switch directive when the arg selects one.
    if target:
        chosen = None
        if target.isdigit():
            idx = int(target)
            if 0 <= idx < len(sessions):
                chosen = sessions[idx]
        else:
            for s in sessions:
                sid = str(s.get("sessionId") or s.get("id") or s.get("hash") or "")
                if sid.startswith(target):
                    chosen = s
                    break
        if chosen is None:
            return _err("sessions", f"no session matching {target!r}", code="not_found")
        sid = str(chosen.get("sessionId") or chosen.get("id") or chosen.get("hash") or "")
        return _resp(
            "sessions",
            KIND_TEXT,
            text=f"switching to {sid[:12]}",
            effects=[Effect(EFFECT_SWITCH_SESSION, {"session_id": sid})],
        )

    items = []
    for i, s in enumerate(sessions):
        sid = str(s.get("sessionId") or s.get("id") or s.get("hash") or "")
        title = s.get("title") or "(untitled)"
        items.append(
            {
                "id": sid,
                "title": f"[{i}] {title}",
                "subtitle": sid[:12],
                "body": s.get("created_at") or s.get("createdAt") or "",
                "actions": [Action(id=f"sw-{sid}", label="open", command=f"/sessions {i}").to_dict()],
            }
        )
    text_out = (
        "sessions:\n" + "\n".join(f"  [{i}] {it['subtitle']} {it['title']}" for i, it in enumerate(items))
        if items
        else "no saved sessions"
    )
    return _resp("sessions", KIND_LIST, text=text_out, data={"title": "sessions", "items": items})


def _list_sessions_from_disk(ctx: CommandContext) -> list[dict[str, Any]]:
    sd = ctx.sessions_dir
    if sd is None:
        return []
    db = Path(sd) / "sessions.db"
    if not db.exists():
        return []
    try:
        from sharedagents_session.session_index import connect_sessions_db

        con = connect_sessions_db(db)
        try:
            cur = con.execute(
                "SELECT session_hash, title, created_at FROM sessions_meta ORDER BY created_at DESC LIMIT 100"
            )
            return [{"hash": r[0], "title": r[1], "created_at": r[2]} for r in cur.fetchall()]
        finally:
            con.close()
    except Exception:
        return []


async def _cmd_refresh(ctx: CommandContext, arg: str) -> CommandResponse:
    def _count() -> int:
        d = _load_disk_view(ctx)
        return len(d.get("turns") or [])

    n = await asyncio.to_thread(_count)
    return _resp("refresh", KIND_TEXT, text=f"session has {n} turns", refresh=["session", "history"])


async def _cmd_compact(ctx: CommandContext, arg: str) -> CommandResponse:
    """Per-engine ``/compact`` dispatch.

    Every engine emits the direct-dispatch compaction sentinel. Native
    local-LLM handles it in-process (zero upstream tokens, so it works at a
    full window). CLI/ACP engines rewrite the sentinel to ``/compact`` in
    ``AgentACPBackend.prompt`` and deliver that command to the bound engine.
    """
    # Compaction is an IN-PROCESS op, NOT a model turn — for the local-LLM
    # engine. The old behaviour submitted a synthetic prompt asking the model
    # to call the ``compact_context`` tool, i.e. a normal LLM round. That
    # DEADLOCKED when the running context was already at/over the window: the
    # round had to fit to emit the tool call, but it couldn't — which is the
    # very overflow the operator was trying to clear (qwen3.6-27b @ 65536 →
    # raw HTTP 400). Emit the direct-dispatch sentinel instead so the engine
    # squeezes the running message list with ZERO upstream tokens; it works
    # even at a completely full window. ``compact_context`` is whitelisted for
    # direct dispatch in local_llm's ``_run_direct_tool_call``.
    #
    # External CLI/ACP engines (claude/codex/antigravity/cursor/opencode/kimi)
    # used to get a prose "call compact_context" prompt because they had no
    # dispatch path — that prompt never reached the engine as a compact
    # command, and ACP engines that are not ``*_cli`` already received the
    # sentinel only to hit ``unknown tool 'compact_context'``. The sentinel
    # is now universal: ``AgentACPBackend.prompt`` rewrites it to ``/compact``
    # and runs it on the bound engine.
    return _resp(
        "compact",
        KIND_TEXT,
        text="compacting context…",
        effects=[Effect(EFFECT_PROMPT, {"text": "__SHAREDAGENTS_DELEGATE__:compact_context:{}"})],
    )


def _monitor_rows(ctx: CommandContext) -> list[dict[str, Any]]:
    """Authoritative monitor list for the session — read from the on-disk
    sidecar (cross-process truth), same source the TUI used to read."""
    sd = ctx.sessions_dir
    if sd is None or not ctx.session_id:
        return []
    try:
        from sharedagents_agent.tools.monitor import live_monitors_for_session

        return list(live_monitors_for_session(Path(sd), ctx.session_id))
    except Exception:
        return []


async def _cmd_monitors(ctx: CommandContext, arg: str) -> CommandResponse:
    parts = (arg or "").split()
    sub = parts[0].lower() if parts and parts[0].lower() in ("list", "tail", "kill") else ""
    target = parts[1] if sub in ("tail", "kill") and len(parts) > 1 else ""

    if sub == "tail":
        if not target:
            return _err("monitors", "usage: /monitors tail <id>")

        def _tail() -> str:
            for m in _monitor_rows(ctx):
                if str(m.get("monitor_id")) == target:
                    lp = m.get("log_path") or ""
                    if not lp:
                        return "(no log)"
                    try:
                        from sharedagents_agent.tools.monitor import read_log_lines

                        size = Path(lp).stat().st_size
                        start = max(0, size - 16 * 1024)
                        lines, _ = read_log_lines(Path(lp), start_byte=start, max_lines=200)
                        return "\n".join(t for t, _err in lines) or "(empty)"
                    except Exception as exc:  # noqa: BLE001
                        return f"(tail failed: {exc})"
            return f"(no monitor {target})"

        body = await asyncio.to_thread(_tail)
        return _resp("monitors", KIND_DETAIL, text=body, data={"title": f"monitor {target}", "body": body})

    if sub == "kill":
        if not target:
            return _err("monitors", "usage: /monitors kill <id>")

        def _kill() -> str:
            import os
            import signal as _sig

            for m in _monitor_rows(ctx):
                if str(m.get("monitor_id")) == target:
                    pid = m.get("pid")
                    if not pid:
                        return "(no pid)"
                    try:
                        os.killpg(int(pid), _sig.SIGTERM)
                        return f"sent SIGTERM to {target} (pid {pid})"
                    except ProcessLookupError:
                        return f"{target} already exited"
                    except Exception as exc:  # noqa: BLE001
                        return f"kill failed: {exc}"
            return f"(no monitor {target})"

        msg = await asyncio.to_thread(_kill)
        return _resp("monitors", KIND_TEXT, text=msg, refresh=["monitors"])

    rows = await asyncio.to_thread(_monitor_rows, ctx)
    items = []
    for m in rows:
        mid = str(m.get("monitor_id") or m.get("id") or "?")
        alive = m.get("alive")
        items.append(
            {
                "id": mid,
                "title": mid,
                "subtitle": ("running" if alive else "exited"),
                "body": str(m.get("command") or ""),
                "badges": [f"pid {m.get('pid')}" if m.get("pid") else ""],
                "actions": [
                    Action(id=f"tail-{mid}", label="tail", command=f"/monitors tail {mid}").to_dict(),
                    Action(
                        id=f"kill-{mid}", label="kill", style="danger", confirm=True, command=f"/monitors kill {mid}"
                    ).to_dict(),
                ],
            }
        )
    text_out = (
        "background shells:\n" + "\n".join(f"  {i['title']} {i['subtitle']}" for i in items)
        if items
        else "no background shells"
    )
    # `rows` carries the raw monitor records so a rich frontend (the TUI
    # monitor panel) renders fully from the server response — no local
    # sidecar reads.
    return _resp("monitors", KIND_LIST, text=text_out, data={"title": "monitors", "items": items, "rows": rows})


async def _cmd_subagents(ctx: CommandContext, arg: str) -> CommandResponse:
    def _list() -> list[dict[str, Any]]:
        sd = ctx.sessions_dir
        if sd is None or not ctx.session_id:
            return []
        from sharedagents_session.session_index import list_child_sessions

        return list_child_sessions(sd, ctx.session_id)

    rows = await asyncio.to_thread(_list)
    items = [{"id": r["id"], "title": r["title"], "subtitle": r["subtitle"]} for r in rows]
    text_out = (
        "subagent sessions:\n" + "\n".join(f"  {i['subtitle']} {i['title']}" for i in items)
        if items
        else "no subagent sessions"
    )
    return _resp("subagents", KIND_LIST, text=text_out, data={"title": "subagents", "items": items})


async def _cmd_abort(ctx: CommandContext, arg: str) -> CommandResponse:
    # ``/abort`` is the INSTANT hard kill: SIGKILL the current agent AND
    # every subagent under it, on this tick — matching the TUI's typed
    # ``/abort`` and Telegram's ``/abort`` (both already force=True). The
    # cooperative "wind down" is ``/stop`` / first-ESC; an explicit
    # ``/abort soft`` still reaches the graceful path through this generic
    # command surface for callers that want it.
    a = (arg or "").strip().lower()
    force = a not in ("soft", "--soft", "graceful", "--graceful", "stop")
    if ctx.cancel_fn is not None:
        try:
            await ctx.cancel_fn(ctx.session_id, force=force)
        except Exception as exc:  # noqa: BLE001
            return _err("abort", f"cancel failed: {exc}", code="cancel_failed")
    return _resp(
        "abort",
        KIND_TEXT,
        text=("hard kill issued" if force else "graceful stop issued"),
        effects=[Effect(EFFECT_CANCEL, {"force": force})],
    )


async def _cmd_new(ctx: CommandContext, arg: str) -> CommandResponse:
    return _resp("new", KIND_TEXT, text="starting a new session", effects=[Effect(EFFECT_NEW_SESSION, {})])


async def _cmd_agents(ctx: CommandContext, arg: str) -> CommandResponse:
    def _list() -> list[str]:
        try:
            from sharedagents_agent.agent import list_agents

            return list(list_agents())
        except Exception:
            return []

    names = await asyncio.to_thread(_list)
    items = [
        {
            "id": n,
            "title": n,
            "subtitle": ("current" if n == ctx.agent_name else ""),
            "actions": [Action(id=f"use-{n}", label="switch", command=f"/agent {n}").to_dict()],
        }
        for n in names
    ]
    text_out = (
        "agents:\n" + "\n".join(f"  {'●' if n == ctx.agent_name else ' '} {n}" for n in names) if names else "no agents"
    )
    return _resp("agents", KIND_LIST, text=text_out, data={"title": "agents", "items": items})


async def _cmd_agent(ctx: CommandContext, arg: str) -> CommandResponse:
    name = (arg or "").strip()
    if not name:
        return _resp("agent", KIND_TEXT, text=f"current agent: {ctx.agent_name}")
    return _resp(
        "agent", KIND_TEXT, text=f"switching to agent {name}", effects=[Effect(EFFECT_SWITCH_AGENT, {"agent": name})]
    )


def _tasks_path(ctx: CommandContext) -> Path | None:
    if ctx.sessions_dir is None or not ctx.session_id:
        return None
    return Path(ctx.sessions_dir) / f"{ctx.session_id}.tasks.json"


def _load_harness_tasks(path: Path) -> dict[str, Any]:
    import json

    if not path.exists():
        return {"next_id": 1, "tasks": {}}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"next_id": 1, "tasks": {}}
    if not isinstance(d, dict):
        return {"next_id": 1, "tasks": {}}
    d.setdefault("tasks", {})
    d.setdefault("next_id", max((int(k) for k in d["tasks"] if str(k).isdigit()), default=0) + 1)
    return d


def _save_harness_tasks(path: Path, state: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


async def _cmd_task(ctx: CommandContext, arg: str) -> CommandResponse:
    """Operator-side harness task list (``<sid>.tasks.json`` — the SAME store
    the model's TaskList/TaskCreate tools use; distinct from the kanban board).
    Verbs: list (default), add <subject>, in|start <N>, done <N>, del <N>.
    """
    path = _tasks_path(ctx)
    if path is None:
        # /task mutates/reads live session-scoped state (<sid>.tasks.json); with
        # no session it degrades to a distinct, non-silent ``unsupported`` error
        # rather than crashing or no-oping.
        return _err("task", "no active session", code="unsupported")
    parts = (arg or "").split(None, 1)
    verb = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    def _items(state: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for tid, t in sorted(state["tasks"].items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
            if not isinstance(t, dict):
                continue
            out.append({"content": f"#{tid} {t.get('subject') or ''}", "status": t.get("status") or "pending"})
        return out

    def _do() -> tuple[bool, str, list[dict[str, Any]]]:
        state = _load_harness_tasks(path)
        if verb in ("", "list", "ls"):
            return True, "tasks", _items(state)
        if verb == "add":
            if not rest:
                return False, "usage: /task add <subject>", []
            tid = str(state["next_id"])
            state["next_id"] = int(tid) + 1
            state["tasks"][tid] = {
                "id": int(tid),
                "subject": rest,
                "status": "pending",
                "owner": "operator",
                "metadata": {"source": "operator"},
            }
            _save_harness_tasks(path, state)
            return True, f"added #{tid}: {rest}", _items(state)
        if verb in ("in", "in-progress", "start", "claim", "done", "complete", "finish", "del", "delete", "rm"):
            tid = rest.lstrip("#").strip()
            if not tid.isdigit() or tid not in state["tasks"]:
                return False, f"unknown task #{tid}", []
            if verb in ("del", "delete", "rm"):
                state["tasks"].pop(tid, None)
                _save_harness_tasks(path, state)
                return True, f"deleted #{tid}", _items(state)
            new_status = "in_progress" if verb in ("in", "in-progress", "start", "claim") else "completed"
            state["tasks"][tid]["status"] = new_status
            _save_harness_tasks(path, state)
            return True, f"#{tid} → {new_status}", _items(state)
        return False, f"unknown /task verb: {verb}", []

    ok, msg, items = await asyncio.to_thread(_do)
    if not ok:
        return _err("task", msg)
    glyph = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]", "deleted": "[-]"}
    listing = "\n".join(f"{glyph.get(it['status'], '[ ]')} {it['content']}" for it in items) or "(no tasks)"
    # Put the rendered list in `text` so text-only frontends (TUI transcript,
    # Telegram) show it; structured `items` for the web checklist renderer.
    text_out = listing if verb in ("", "list", "ls") else f"{msg}\n{listing}"
    return _resp("task", KIND_CHECKLIST, text=text_out, data={"items": items}, refresh=["task"])


async def _cmd_mouse(ctx: CommandContext, arg: str) -> CommandResponse:
    # Mouse tracking is a terminal-emulator concern (TUI only). Recognized
    # everywhere so it's never "unknown", but it has no server-side effect.
    return _resp("mouse", KIND_TEXT, text="mouse tracking is a terminal-only setting (TUI); no effect on this frontend")


async def _cmd_steer(ctx: CommandContext, arg: str) -> CommandResponse:
    text = (arg or "").strip()
    if not text:
        return _err("steer", "usage: /steer <text>")
    if ctx.inject_fn is None:
        # No live inject channel (CLI agent / session-less): fall back to
        # submitting the steer text as a normal prompt.
        return _resp("steer", KIND_TEXT, text="steering…", effects=[Effect(EFFECT_PROMPT, {"text": text})])
    try:
        res = await ctx.inject_fn(ctx.session_id, text)
    except Exception as exc:  # noqa: BLE001
        return _err("steer", f"steer failed: {exc}", code="steer_failed")
    queued = bool(res.get("queued")) if isinstance(res, dict) else False
    if not queued:
        # Engine couldn't queue (no active turn) — submit as a prompt.
        return _resp("steer", KIND_TEXT, text="steering…", effects=[Effect(EFFECT_PROMPT, {"text": text})])
    return _resp("steer", KIND_TEXT, text="steer queued for the running turn")


async def _cmd_restart(ctx: CommandContext, arg: str) -> CommandResponse:
    hard = "--hard" in (arg or "")

    def _restart() -> dict[str, Any]:
        from sharedagents_core.restart import restart_agent_detached

        return restart_agent_detached(ctx.agent_name, hard=hard)

    res = await asyncio.to_thread(_restart)
    if not res.get("ok"):
        return _err("restart", res.get("error") or "restart failed", code="restart_failed")
    return _resp("restart", KIND_TEXT, text=f"restart scheduled ({res.get('mode')})", data=res)


async def _cmd_title(ctx: CommandContext, arg: str) -> CommandResponse:
    """``/title <text>`` — set the current session's title.

    Mirrors ``gateway_shim.set_session_title``: prefer the live in-memory
    session (``save()`` emits a ``header_update`` the projector folds into
    ``sessions_meta.title``); fall back to a direct ``sessions_meta`` UPDATE
    when no session object is attached — which is the path a remote
    ``session/command`` ``/title`` (issued by
    ``remote_backend.set_session_title``) lands on. A placeholder
    ``untitled-…`` title is refused so it can't clobber a real one.

    An operator ``/title`` rename is an explicit intent, so it also PINS the
    title (``title_locked``): the title agent — including the unconditional
    turn-10 refresh — never overwrites it again.
    """
    title = (arg or "").strip()
    if not title:
        return _err("title", "usage: /title <new title>")
    if title.startswith("untitled-"):
        return _err("title", "refusing to set a placeholder title")
    live = ctx.live_session
    if live is not None:
        try:
            live.title = title
            live.title_locked = True
            # ``Session.save`` serializes + writes the session header to
            # ``sessions.db``; keep that blocking I/O off the gateway's event
            # loop. Pass the context's sessions_dir
            # EXPLICITLY: ``Session.save``'s default root is the import-time
            # module constant ``SESSIONS_DIR`` (``~/.sharedagents/sessions``), which
            # is computed from ``Path.home()`` and ignores ``$SHAREDAGENTS_HOME``.
            # Under a dev / alternate home the default therefore wrote to the
            # WRONG (prod) sessions tree, failing with "unable to open database
            # file" (dir absent) or "attempt to write a readonly database" (prod
            # db locked). Every other gateway caller already passes
            # ``cfg.sessions_dir`` (heartbeat, acp_bridge) — match them so
            # /title lands in the same tree turns are written to, and the writer
            # creates that dir on demand.
            sdir = ctx.sessions_dir
            if sdir is not None:
                await asyncio.to_thread(live.save, sdir, force_fields=("title", "title_locked"))
            else:
                await asyncio.to_thread(live.save, force_fields=("title", "title_locked"))
        except Exception as exc:  # noqa: BLE001
            return _err("title", f"title save failed: {exc}")
        return _resp("title", KIND_TEXT, text=f"title set to: {title}")
    if ctx.sessions_dir is not None and ctx.session_id:

        def _write_title() -> bool:
            from sharedagents_session.session_index import connect_sessions_db

            db = ctx.sessions_dir / "sessions.db"
            if not db.exists():
                return False
            con = connect_sessions_db(db)
            try:
                con.execute(
                    "UPDATE sessions_meta SET title=?, title_locked=1 WHERE session_hash=?",
                    (title, ctx.session_id),
                )
                con.commit()
            finally:
                con.close()
            return True

        try:
            # sqlite open/UPDATE/commit is blocking — offload it
            wrote = await asyncio.to_thread(_write_title)
        except Exception as exc:  # noqa: BLE001
            return _err("title", f"title save failed: {exc}")
        if wrote:
            return _resp("title", KIND_TEXT, text=f"title set to: {title}")
    return _err("title", "no session available to set title")


# ---------------------------------------------------------------------------
# Dispatch table + entrypoint
# ---------------------------------------------------------------------------


def _post_command_goal_snapshot(
    ctx: CommandContext,
    res: dict[str, Any],
) -> dict[str, Any]:
    """Authoritative projection for the state_patch after a goal mutation.

    Shared by ``_cmd_goal`` and ``_cmd_budget`` (the /budget aliases mutate
    the goal budget and must patch the widget exactly like /goal budget).
    The backend resolves its OWN session object
    (``gateway_shim.get_session`` / ``acp_bridge._session_for`` load a FRESH
    :class:`Session` per call) and mutates THAT — ``ctx.live_session`` can
    be a stale pre-command object, so reading it after a successful command
    would emit a patch with the PRE-command goal state and the widget would
    not move (stale CLI-engine state patches). The
    backend already reports the post-command projection in its result dict;
    its goal id selects the matching cost metrics before the remaining fields
    are overlaid onto the ``_goal_state`` snapshot. Fields the backend did not
    report fall back to the snapshot.
    """
    authoritative_goal_id = str(res.get("goal_id") or "") if "goal_id" in res else None
    snapshot = _goal_state(ctx, goal_id_override=authoritative_goal_id)
    for key in ("goal_id", "active_goal", "goal_terminal", "goal_turn_count"):
        if key in res:
            snapshot[key] = res[key]
    return snapshot


async def _cmd_goal(ctx: CommandContext, arg: str) -> CommandResponse:
    """`/goal` — durable goal control, identical across every frontend.

    Routes through the backend's ``goal_command_fn`` (→
    ``goal_service.execute_goal_command`` → ``goals.db``) so a goal SET from
    Telegram, the web command bar, or the TUI is registered the same way, and
    ``/goal list`` renders the SAME registry view everywhere. On a
    fresh set the web client submits the condition through the same
    ``/prompt`` API as a typed message — this handler only registers
    the goal and patches the widget.

    Before this handler existed, ``/goal`` had no entry in ``_HANDLERS`` and
    fell through to the file-command path, which merely expanded the
    ``goal.md`` prompt template and never touched the registry — so a goal
    set from Telegram was invisible to ``goals.db`` and the web portfolio.
    """
    if ctx.goal_command_fn is None:
        return _err("goal", "goals are not available on this backend", code="unsupported")
    from sharedagents_goal.goal_service import parse_goal_subcommand

    action, payload, _force = parse_goal_subcommand(arg)
    try:
        res = await ctx.goal_command_fn(arg) or {}
    except Exception as exc:  # noqa: BLE001
        return _err("goal", f"{type(exc).__name__}: {exc}", code="internal_error")
    message = str(res.get("message") or "")
    if not res.get("ok", False):
        return _err(
            "goal", message or "goal command failed", code="goal_conflict" if res.get("conflicts") else "bad_request"
        )

    if action == "set":
        # Widget only. The chat turn is the same /prompt the composer
        # already uses — no EFFECT_PROMPT special path.
        return _resp(
            "goal",
            KIND_TEXT,
            text=message or "goal set",
            data={"goal": res.get("goal")},
            effects=[_state_patch_effect(_post_command_goal_snapshot(ctx, res))],
        )
    if action in _GOAL_MUTATING_ACTIONS:
        # Every other SUCCESSFUL mutating action (clear/pause/resume/restart/
        # archive/topup/budget) emits the same lightweight state_patch so the
        # originating page's goal widget reflects the new state immediately —
        # /goal clear hides the widget (explicit empty goal fields), /goal
        # pause shows the bound paused goal with its Resume action. Read-only
        # list/status stay patch-free (no over-patching on a query).
        return _resp(
            "goal",
            KIND_TEXT,
            text=message or "(goal)",
            data={"goal": res.get("goal")},
            effects=[_state_patch_effect(_post_command_goal_snapshot(ctx, res))],
        )
    return _resp("goal", KIND_TEXT, text=message or "(no goals)", data={"goal": res.get("goal")})


async def _cmd_budget(ctx: CommandContext, arg: str) -> CommandResponse:
    """`/budget` — view or set this session tree's USD cost budget.

      /budget                 — show this session's spend vs budget
      /budget <usd>           — set this session's budget (e.g. /budget 5)
      /budget add <usd>       — top up this session's budget (+ resume a paused goal)
      /budget <goal_id> <usd> — set a specific goal's budget (also stamps the session)

    Routes through the shared ``goal_command_fn`` (→ ``execute_goal_command``)
    so it behaves identically across TUI / Telegram / web, same as ``/goal``.
    """
    if ctx.goal_command_fn is None:
        return _err("budget", "budgets are not available on this backend", code="unsupported")
    a = (arg or "").strip()
    # Translate the /budget surface into a /goal subcommand.
    if not a:
        sub = "status"
    elif a.split()[0].lower() in ("add", "+", "topup"):
        rest = a.split(None, 1)[1] if len(a.split(None, 1)) > 1 else ""
        sub = f"budget add {rest}".strip()
    else:
        sub = f"budget {a}"
    # Only the MUTATING aliases patch the widget: /budget <usd>,
    # /budget add|topup|+ <usd> (and the explicit /budget <goal_id> <usd>
    # form). A bare /budget (status view) is read-only — no state_patch, so
    # a query never re-renders the widget (standalone /budget aliases used
    # to mutate the budget with no patch).
    mutating = bool(sub) and sub != "status"
    try:
        res = await ctx.goal_command_fn(sub) or {}
    except Exception as exc:  # noqa: BLE001
        return _err("budget", f"{type(exc).__name__}: {exc}", code="internal_error")
    message = str(res.get("message") or "")
    if not res.get("ok", False):
        return _err("budget", message or "budget command failed")
    effects: list[Effect] = []
    if mutating:
        effects = [_state_patch_effect(_post_command_goal_snapshot(ctx, res))]
    return _resp("budget", KIND_TEXT, text=message or "(budget)", data={"goal": res.get("goal")}, effects=effects)


async def _cmd_loop(ctx: CommandContext, arg: str) -> CommandResponse:
    """`/loop` — scheduled-work control, identical across every frontend.

    Routes through the backend's ``loop_command_fn`` (→
    ``loop_service.execute_loop_command`` → loops.db + the live loop
    supervisor) so a loop created from Telegram, the web command bar, or
    the TUI is registered the same way and actually fires on schedule —
    the durable counterpart of the legacy file-command expansion, which
    only ever re-read ``loop.md`` and never touched the registry.

    Unlike ``/goal set``, creating a loop deliberately does NOT inject a
    turn: the supervisor fires the loop's prompt on its own cadence.
    """
    if ctx.loop_command_fn is None:
        return _err("loop", "loops are not available on this backend", code="unsupported")
    try:
        res = await ctx.loop_command_fn(arg or "") or {}
    except Exception as exc:  # noqa: BLE001
        return _err("loop", f"{type(exc).__name__}: {exc}", code="internal_error")
    message = str(res.get("message") or "")
    if not res.get("ok", False):
        return _err("loop", message or "loop command failed", code="bad_request")
    data: dict[str, Any] = {}
    if res.get("loop") is not None:
        data["loop"] = res["loop"]
    if res.get("note"):
        data["note"] = res["note"]
    if res.get("loops"):
        data["loops"] = res["loops"]
    return _resp("loop", KIND_TEXT, text=message or "loop scheduled", data=data)


async def _cmd_memory(ctx: CommandContext, arg: str) -> CommandResponse:
    """/memory — memory is always on; opt-in gating was removed."""
    _ = arg
    this_chat = ctx.session_id or ""
    return _resp(
        "memory",
        KIND_TEXT,
        text="memory: always on (origin/opt-in allowlists removed)",
        data={"chat_id": this_chat, "opted_in": True},
    )


_HANDLERS: dict[str, Callable[[CommandContext, str], Awaitable[CommandResponse]]] = {
    "help": _cmd_help,
    "status": _cmd_status,
    "context": _cmd_context,
    "todo": _cmd_todo,
    "skills": _cmd_skills,
    "mcp": _cmd_mcp,
    "commands": _cmd_commands,
    "plugin": _cmd_plugin,
    "models": _cmd_models,
    "model": _cmd_model,
    "mode": _cmd_mode,
    "reasoning": _cmd_reasoning,
    "effort": _cmd_effort,
    "thinking": _cmd_thinking,
    "tool": _cmd_tool,
    "teacher": _cmd_teacher,
    "goal": _cmd_goal,
    "budget": _cmd_budget,
    "loop": _cmd_loop,  # durable scheduled work (loops.db + supervisor)
    "history": _cmd_history,
    "sessions": _cmd_sessions,
    "refresh": _cmd_refresh,
    "compact": _cmd_compact,
    "monitors": _cmd_monitors,
    "subagents": _cmd_subagents,
    "abort": _cmd_abort,
    "new": _cmd_new,
    "restart": _cmd_restart,
    "agents": _cmd_agents,
    "agent": _cmd_agent,
    "steer": _cmd_steer,
    "mouse": _cmd_mouse,
    "task": _cmd_task,  # operator harness-task store (NOT the kanban board)
    "tasks": _cmd_task,
    "title": _cmd_title,  # set session title (also the remote-backend path)
    "memory": _cmd_memory,
}

# Commands that mutate state — callers may serialize these per-session.
MUTATING_COMMANDS = frozenset(
    {
        "model",
        "mode",
        "reasoning",
        "effort",
        "thinking",
        "tool",
        "teacher",
        "goal",
        "budget",
        "plugin",
        "abort",
        "new",
        "restart",
        "compact",
        "agent",
        "title",
        "memory",
        "loop",
    }
)

# Mutating commands that are UNSAFE while a turn is in flight (they'd
# corrupt the running model/context). Rejected with a clear error when an
# active turn exists — mirrors the TUI's "finish or /abort first" gate.
# (cancel/abort/mode/reasoning/effort/thinking/tool are intentionally allowed
# mid-turn: they're designed to steer or stop a live turn.)
TURN_UNSAFE_COMMANDS = frozenset({"model", "compact"})

# Irreversible commands — running one discards state that
# CANNOT be recovered, so the web frontend re-challenges with an explicit
# confirmation before executing (the slash-command parallel of the
# session-delete re-challenge in ``sharedagents_web``). This frozenset is the ONE
# home for the "always irreversible" set; two further commands are irreversible
# only for a specific destructive subcommand (see ``command_is_irreversible``):
#   /task del|delete|rm <N> — permanently drops a task record from the store
#   /monitors kill <id>     — SIGTERMs an in-flight background-shell process
# Every other command, and every recoverable subcommand of task/monitors
# (list/add/in/done/tail…), executes one-shot with no added step.
IRREVERSIBLE_COMMANDS = frozenset({"compact", "restart", "abort"})


def command_is_irreversible(name: str, arg: str) -> bool:
    """True iff ``/<name> <arg>`` irreversibly discards state.

    THE single classification the confirm re-challenge keys on; the
    web transport imports this rather than re-deriving the set, so the
    irreversible list has one authoritative home (beside ``MUTATING_COMMANDS``).
    ``name`` is the ``parse_command``-normalized command (``/shells``→``monitors``,
    ``/stop``→``abort``, ``/tasks``→``tasks``); ``arg`` is the raw remainder.
    """
    n = (name or "").strip().lower()
    if n in IRREVERSIBLE_COMMANDS:
        return True
    parts = (arg or "").split()
    verb = parts[0].lower() if parts else ""
    if n in ("task", "tasks"):
        return verb in ("del", "delete", "rm")
    if n in ("monitors", "shells"):
        return verb == "kill"
    return False


def _irreversible_dialog(name: str, arg: str) -> tuple[str, str]:
    """(object, what-is-irreversible) copy for the re-challenge dialog.

    Names the specific object affected and states what is irreversible about
    the action.
    """
    n = (name or "").strip().lower()
    parts = (arg or "").split()
    if n in ("task", "tasks"):
        tid = (parts[1].lstrip("#") if len(parts) > 1 else "").strip()
        obj = f"task #{tid}" if tid else "this task"
        return obj, f"Deleting {obj} permanently removes it from the task list."
    if n in ("monitors", "shells"):
        mid = parts[1] if len(parts) > 1 else ""
        obj = f"monitor {mid}" if mid else "this background shell"
        return obj, f"Killing {obj} terminates the in-flight process."
    labels = {
        "compact": ("the conversation context", "Compacting rewrites the conversation history in place."),
        "restart": ("this session", "Restarting discards the running worker and its in-flight turn."),
        "abort": ("the running turn", "Aborting hard-kills the worker, its tools, and all subagents."),
    }
    return labels.get(n, (f"/{n}", f"Running /{n} is irreversible."))


def _confirm_required(name: str, arg: str) -> CommandResponse:
    """Re-challenge envelope for an unconfirmed irreversible command.

    Mirrors the web session-delete re-challenge payload (``confirmation_required``
    + ``object`` + ``prompt`` + ``irreversible``) so a frontend renders ONE
    confirm dialog for both surfaces, then re-submits the command with the
    confirmation set. ``ok=False`` + ``error_code`` so a non-web transport that
    never sends the flag simply surfaces a (non-fatal) error envelope.
    """
    obj, what = _irreversible_dialog(name, arg)
    return CommandResponse(
        ok=False,
        command=name,
        result=RenderResult(
            kind=KIND_ERROR,
            text=f"{what} This cannot be undone — confirm to proceed.",
            data={
                "confirmation_required": True,
                "irreversible": True,
                "object": obj,
                "prompt": f"{what} This cannot be undone.",
            },
        ),
        error="confirmation required",
        error_code="confirmation_required",
    )


def list_commands() -> list[str]:
    """Built-in command names the runtime can dispatch (for capability discovery).

    Returns the built-in dispatch-table (``_HANDLERS``) keys only — the
    pre-invocation discovery surface; agent-alias and file-command names are
    dispatchable too but are not enumerated here.
    """
    return sorted(_HANDLERS)


async def run_command(ctx: CommandContext, text: str) -> CommandResponse:
    """Parse and execute a slash command. THE single implementation.

    Returns a :class:`CommandResponse`; never raises for normal command
    errors (those become ``ok=False`` responses). Unexpected exceptions are
    caught at the wire boundary by the transport adapter.
    """
    parsed = parse_command(text)
    if parsed is None:
        return _err("", "not a slash command", code="not_a_command")
    handler = _HANDLERS.get(parsed.name)
    if handler is None:
        # Delegate bridge: `/claude <prompt>`, `/codex <prompt>`, or any
        # sibling agent name. With an arg → run the prompt on THAT agent
        # (prompt effect carrying the target agent). Bare → switch to it.
        agents = await asyncio.to_thread(_known_agent_names)
        if parsed.name in agents:
            if parsed.arg:
                return _resp(
                    parsed.name,
                    KIND_TEXT,
                    text=f"delegating to {parsed.name}…",
                    effects=[Effect(EFFECT_PROMPT, {"text": parsed.arg, "agent": parsed.name})],
                )
            return _resp(
                parsed.name,
                KIND_TEXT,
                text=f"switching to agent {parsed.name}",
                effects=[Effect(EFFECT_SWITCH_AGENT, {"agent": parsed.name})],
            )
        # Unknown /name → maybe a file command (commands/*.md). Expand it
        # server-side and return a prompt effect so the frontend submits
        # the expanded text as a normal turn — file-command logic stays in
        # the backend; the frontend just runs the prompt it's handed.
        fc = await asyncio.to_thread(_expand_file_command, ctx, parsed.name, parsed.arg)
        if fc is not None:
            return _resp(
                parsed.name,
                KIND_TEXT,
                text=f"running file command /{parsed.name}",
                effects=[Effect(EFFECT_PROMPT, {"text": fc})],
            )
        return _err(parsed.name, f"unknown command: /{parsed.name}", code="unknown_command")
    # Refuse turn-unsafe mutations while a turn is running (parity with the
    # TUI, which blocks /model etc. mid-turn). cancel/mode/reasoning are safe.
    if parsed.name in TURN_UNSAFE_COMMANDS and ctx.active_turn_id:
        return _err(
            parsed.name,
            f"/{parsed.name} is unavailable while a turn is running — finish it or /abort first",
            code="turn_active",
        )
    # An irreversible slash command from the web frontend is
    # re-challenged (parity with the session-delete re-challenge) unless the
    # request carried an explicit confirmation. Enforced server-side in THE
    # single command implementation, so a direct API call bypassing the dialog
    # is re-challenged too; scoped to ``web`` so the TUI/Telegram surfaces
    # (their own confirm affordances) are unaffected.
    if ctx.frontend == "web" and not ctx.confirmed and command_is_irreversible(parsed.name, parsed.arg):
        return _confirm_required(parsed.name, parsed.arg)
    try:
        return await handler(ctx, parsed.arg)
    except Exception as exc:  # noqa: BLE001 - defensive; transport also guards
        return _err(parsed.name, f"{type(exc).__name__}: {exc}", code="internal_error")


def _known_agent_names() -> set[str]:
    try:
        from sharedagents_agent.agent import list_agents

        return set(list_agents())
    except Exception:
        return set()


def _expand_file_command(ctx: CommandContext, name: str, arg: str) -> str | None:
    """Resolve a file command (commands/*.md) and expand its body. Returns
    the expanded prompt text, or None when no such file command exists."""
    try:
        from sharedagents_command.file_commands import expand_command, find_command

        cmd = find_command(name, ctx.agent_home)
        if cmd is None:
            return None
        return expand_command(cmd, arg or "")
    except Exception:
        return None


__all__ = [
    "SCHEMA_VERSION",
    "Action",
    "Effect",
    "RenderResult",
    "CommandResponse",
    "CommandContext",
    "run_command",
    "list_commands",
    "MUTATING_COMMANDS",
    "TURN_UNSAFE_COMMANDS",
    "IRREVERSIBLE_COMMANDS",
    "command_is_irreversible",
    "STATE_PATCH_KEYS",
    "FRONTEND_CAPABILITIES",
    "enforce_mode_capability",
    "resolve_effective_frontend",
]
