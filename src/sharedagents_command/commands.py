"""Shared slash-command parsing + formatting for sharedagents frontends.

The runtime owns command semantics; frontends should only adapt the
returned result to their UI surface (TUI widgets, Telegram messages,
plain JSON, etc.). Keep this module mostly pure so Telegram, the TUI,
and API endpoint do not drift in validation or wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sharedagents_agent.external_registry import (
    EXTERNAL_ENGINE_VALUES,
    is_external_engine_kind,
)

REASONING_VALID = {"off", "on", "stream"}
MODE_VALID = {
    "auto",
    "ask",
    "plan",
    "build",
    "agent",
    "default",
    "dont_ask",
    "accept_edits",
    "allow_all",
}
_MODE_ALIASES = {
    "allow-all": "auto",
    "allowall": "auto",
    "allow all": "auto",
    "yolo": "auto",
    "code": "auto",
    "architect": "plan",
    "dontask": "dont_ask",
    "don't ask": "dont_ask",
    "acceptedits": "accept_edits",
}
TOOL_OUTPUT_VALID = {"on", "off"}


@dataclass(slots=True)
class ParsedCommand:
    raw: str
    name: str
    arg: str = ""


@dataclass(slots=True)
class CommandResult:
    ok: bool
    command: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    state_patch: dict[str, Any] = field(default_factory=dict)
    ui: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "command": self.command,
            "text": self.text,
            "data": self.data,
            "state_patch": self.state_patch,
            "ui": self.ui,
        }
        if self.error:
            payload["error"] = self.error
        return payload


def parse_command(text: str) -> ParsedCommand | None:
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return None
    head, _, arg = raw.partition(" ")
    # Telegram may send /cmd@BotName in groups.
    cmd = head.split("@", 1)[0].lower()
    aliases = {
        "/models": "models",
        "/model": "model",
        "/reason": "reasoning",
        "/reasoning": "reasoning",
        "/effort": "effort",
        "/thinking": "thinking",
        "/teacher": "teacher",
        "/on": "reasoning",
        "/off": "reasoning",
        "/stream": "reasoning",
        "/mode": "mode",
        "/status": "status",
        "/help": "help",
        "/new": "new",
        "/sessions": "sessions",
        "/history": "history",
        "/refresh": "refresh",
        "/abort": "abort",
        "/stop": "abort",
        "/compact": "compact",
        "/todo": "todo",
        "/todos": "todo",
        "/restart": "restart",
        "/tool": "tool",
        "/tools": "tool",
        "/context": "context",
        "/ctx": "context",
        "/skills": "skills",
        "/skill": "skills",
        "/commands": "commands",
        "/plugin": "plugin",
        "/plugins": "plugin",
        "/mcp": "mcp",
        "/mcps": "mcp",
        "/subagents": "subagents",
        "/monitors": "monitors",
        "/shells": "monitors",
    }
    name = aliases.get(cmd, cmd[1:])
    if cmd in ("/on", "/off", "/stream") and not arg.strip():
        arg = cmd[1:]
    return ParsedCommand(raw=raw, name=name, arg=arg.strip())


def normalize_reasoning_level(raw: str) -> str | None:
    key = (raw or "").strip().lower().replace("_", "-")
    if key in ("off", "false", "no", "0", "hide", "hidden", "disable", "disabled"):
        return "off"
    if key in ("on", "true", "yes", "1", "show", "visible", "enable", "enabled"):
        return "on"
    if key in ("stream", "streaming", "draft", "live"):
        return "stream"
    return None


def normalize_mode(raw: str) -> str | None:
    key = (raw or "").strip().lower()
    key = _MODE_ALIASES.get(key, key)
    return key if key in MODE_VALID else None


def normalize_effort(raw: str) -> str | None:
    """Preserve provider tokens; the resolved model capability validates them."""
    key = str(raw or "").strip()
    if key.lower() in ("", "default", "auto", "reset", "clear"):
        return "default"
    return key


def normalize_thinking(raw: str) -> bool | None:
    key = (raw or "").strip().lower().replace("_", "-")
    if key in ("on", "true", "yes", "1", "enable", "enabled"):
        return True
    if key in ("off", "false", "no", "0", "disable", "disabled"):
        return False
    return None


def normalize_tool_output(raw: str) -> str | None:
    """Three-state tool-output visibility:

    * ``"off"``    — hide everything tool-related (no header, no body)
    * ``"on"``     — show the tool-call header (``🔧 <title>…``) but
                     NOT the raw body the tool returned. The model's
                     prose answer (which may quote results) still
                     lands; only the raw tool payload is suppressed.
    * ``"stream"`` — show BOTH the call header AND the raw body
                     streaming as the tool produces it.

    Returns the canonical token or ``None`` on unknown input. Direct-
    dispatch tools (``delegate_claude`` / ``delegate_codex`` /
    ``compact_context``) IGNORE this gate — their body IS the
    answer; suppressing it would leave the chat empty.
    """
    key = (raw or "").strip().lower().replace("_", "-")
    if key in ("off", "false", "no", "0", "hide", "hidden", "disable", "disabled"):
        return "off"
    if key in (
        "on",
        "true",
        "yes",
        "1",
        "show",
        "visible",
        "enable",
        "enabled",
        "header",
        "headers",
    ):
        return "on"
    if key in ("stream", "streaming", "live", "verbose", "all"):
        return "stream"
    return None


def format_help() -> str:
    return (
        "sharedagents — quick reference\n\n"
        "Slash commands:\n"
        "  /help                 — this card\n"
        "  /new                  — abort + start a new session\n"
        "  /stop                 — graceful stop (finish current step, then end)\n"
        "  /abort                — hard kill now (worker + tools + subagents)\n"
        "  /status               — session + mode + active turn\n"
        "  /models               — list configured models\n"
        "  /model <alias|url>    — switch this session model\n"
        "  /mode [plan|build]    — Plan/Build (OpenCode Tab); native switches agent\n"
        "  /effort <level|default> — model-specific reasoning depth\n"
        "  /reasoning [on|off|stream] — reasoning visibility (default stream)\n"
        "  /on, /off, /stream    — shortcuts for /reasoning <level>\n"
        "  /thinking [on|off]    — enable/disable model thinking\n"
        "  /tool [on|off]        — tool-output visibility (default off)\n"
        "  /teacher [on|off]     — toggle teacher-model supervision (any session, goal or not)\n"
        "  /teacher model <alias> — pick the supervising model (model default = configured teacher)\n"
        "  /goal <condition>     — set a durable goal (registered in goals.db; work until met)\n"
        "  /goal [list|status]   — list every goal (any interface) / show this session's goal\n"
        "  /goal pause|resume|restart|met|delete <id> — control a durable goal\n"
        "  /budget [<usd>]       — show / set this session's USD cost budget\n"
        "  /budget add <usd>     — top up this session's budget and resume\n"
        "  /compact              — compact saved session context\n"
        "  /skills               — list installed skills\n"
        "  /skill <name>         — show one skill's SKILL.md body\n"
        "  /commands             — list file-based slash commands\n"
        "  /<command> [args]     — run a file command (commands/*.md)\n"
        "  /plugin list                       — list installed plugins\n"
        "  /plugin search <query>             — search the plugin marketplace (skills/hooks/mcp/...)\n"
        "  /plugin install <name|url|path>    — install a plugin (skills+commands+hooks+agents+mcp)\n"
        "  /plugin enable|disable|remove <n>  — toggle / uninstall a plugin\n"
        "  /mcp [filter]                      — list MCP servers (builtin web/memory/delegate + config + ~/.claude/.mcp.json + plugins)\n"
        "  /subagents [N]        — list / enter delegated subagent sessions\n"
        "  /monitors, /shells    — live panel of background shells the model spawned\n"
        "  /todo                 — show this session's todo checklist\n"
        "  /context [list|detail|json] — context breakdown (chars + tokens)\n"
        "  /sessions [i|hash]    — list/switch saved sessions\n"
        "  /history [N]          — show recent turns\n"
        "  /refresh              — show/catch up current session state\n"
        "  /restart [--hard]     — restart the gateway; --hard also bounces the Telegram bot (systemd)\n"
        "  /claude <prompt>      — delegate to Claude Code where available\n\n"
        "Modes: plan / build (OpenCode). Native = switch agent. External = engine mode."
    )


def format_status(data: dict[str, Any]) -> str:
    keys = (
        "agent",
        "upstream",
        "backend",
        "model",
        "mode",
        "reasoning",
        "effort",
        "thinking",
        "tool",
        "session",
        "active_turn",
    )
    return "\n".join(f"{k}: {data.get(k, '?')}" for k in keys if k in data)


CONTEXT_SUBCOMMANDS = {"", "help", "list", "show", "detail", "deep", "json"}


def normalize_context_subcommand(arg: str) -> str:
    """Normalize ``/context <sub>`` → canonical subcommand.

    Mirrors OpenClaw's ``parseContextArgs`` behavior. Returns ``""`` for
    the bare ``/context`` (help mode), ``"list"`` / ``"detail"`` /
    ``"json"`` for the breakdown modes, or the raw token if unknown so
    the handler can render an "Unknown /context mode" error.
    """
    raw = (arg or "").strip().lower()
    if not raw:
        return ""
    head = raw.split()[0]
    return head


# Engines that drive an external model transport (a CLI subprocess or a
# remote ACP server). Their model is chosen out-of-band — the CLI/ACP
# server owns it — so an agent's local ``models[]`` rows (the vLLM /
# DeepSeek upstreams) DON'T apply. The registry owns this set; the generic
# ``acp`` adapter is included by ``EXTERNAL_ENGINE_VALUES``.
EXTERNAL_MODEL_ENGINES = EXTERNAL_ENGINE_VALUES


def engine_model_rows_for_engine(
    engine: str,
    agent_cfg: Any = None,
) -> list[dict[str, Any]] | None:
    """Per-engine model menu for an engine name — the single source of
    truth every frontend's ``/models`` consults so they don't drift.

    Returns:

    * ``None`` — ``local_llm`` (or unknown / empty): the caller uses the
      agent's own ``cfg.models`` rows, which are already per-agent.
    * a **non-empty** list — the engine owns a switchable menu returned by its
      registry-registered discovery hook. Show exactly these.
    * ``[]`` — the engine owns the model out-of-band
      (generic ACP): the local ``cfg.models`` rows do NOT apply,
      so the caller shows ONLY the session's active model, never the local
      upstream list.
    """
    engine = (engine or "").strip().lower()
    if not is_external_engine_kind(engine):
        return None
    # A user-configured generic ACP server has no registry discovery hook;
    # its model id space is intentionally opaque and the typed /model value is
    # forwarded to that server.
    if engine == "acp":
        return []
    try:
        from sharedagents_agent.discovery.cache_lock import discover_agent_models

        agent_name = getattr(agent_cfg, "name", "") or engine.split("_")[0]
        cached_or_discovered = discover_agent_models(agent_name)
        if cached_or_discovered:
            return cached_or_discovered

        from sharedagents_agent.agents.factory import _engine_for, resolve_engine_kind

        kind = resolve_engine_kind(engine)
        # Route through the engine capability, not through a second command
        # layer that has to know which vendor accepts a custom command. Every
        # registered external row therefore gets the same /model semantics.
        engine_instance = _engine_for(kind, agent_cfg)
        rows = engine_instance.available_models()
        clean_rows = [dict(row) for row in (rows or []) if isinstance(row, dict)]
        from sharedagents_agent.discovery.cache_lock import stamp_configured_default

        return stamp_configured_default(agent_name, clean_rows, agent_cfg=agent_cfg)
    except Exception:  # noqa: BLE001
        return []


def engine_model_rows(agent_cfg: Any) -> list[dict[str, Any]] | None:
    """:func:`engine_model_rows_for_engine` keyed off an agent config's
    ``engine`` attribute. Convenience for callers that hold the config."""
    return engine_model_rows_for_engine(
        getattr(agent_cfg, "engine", "") or "",
        agent_cfg,
    )


def cold_external_model_for(agent_cfg: Any) -> str:
    """Best-effort concrete model id for an external engine before any turn.

    Used by Telegram ``/new``, the TUI footer, and other cold-start surfaces
    so they show ``composer-2.5`` / ``gpt-5.5-codex`` instead of the bare
    engine name (``cursor`` / ``codex``). Returns ``""`` when unknown.
    """
    engine = (getattr(agent_cfg, "engine", "") or "").strip().lower()
    if not is_external_engine_kind(engine):
        return ""
    try:
        from sharedagents_agent.agents.factory import _engine_for, resolve_engine_kind

        kind = resolve_engine_kind(engine)
        eng = _engine_for(kind, agent_cfg)
        display = getattr(eng, "display_model", None)
        if callable(display):
            label = str(display() or "").strip()
            # Skip placeholder engine names — only return a concrete id.
            if label and not is_engine_placeholder_model(
                label,
                engine,
                str(getattr(agent_cfg, "name", "") or ""),
            ):
                return label
    except Exception:  # noqa: BLE001
        pass
    # Native CLI engines usually do not expose a live current-model label
    # until their first turn. Use the same registry-backed catalogue as
    # /models and prefer an explicitly marked default before the first row.
    rows = engine_model_rows_for_engine(engine, agent_cfg) or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not row.get("default"):
            continue
        candidate = str(row.get("id") or row.get("display") or row.get("alias") or "").strip()
        if candidate and not is_engine_placeholder_model(candidate, engine, str(getattr(agent_cfg, "name", "") or "")):
            return candidate
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate = str(row.get("id") or row.get("display") or row.get("alias") or "").strip()
        if candidate and not is_engine_placeholder_model(candidate, engine, str(getattr(agent_cfg, "name", "") or "")):
            return candidate
    return ""


def is_engine_placeholder_model(
    model: str,
    engine: str,
    agent_name: str = "",
) -> bool:
    """True when ``model`` is just the engine/agent label, not a concrete id."""
    m = (model or "").strip()
    if not m:
        return False
    placeholders = {
        engine,
        engine.removesuffix("_cli") if engine.endswith("_cli") else "",
        engine.removesuffix("_acp") if engine.endswith("_acp") else "",
        (agent_name or "").strip(),
    }
    return m.lower() in {p.lower() for p in placeholders if p}


def resolve_session_model(cur_model: str, agent_cfg: Any) -> str:
    """Return the best display model for a session header value.

    Stale placeholders (``cursor``, ``claude``, ``codex``) are replaced by
    :func:`cold_external_model_for` so ``/new`` and ``/models`` show the
    discovered upstream id even when an older session header never recorded
    ``engine_model``.
    """
    engine = (getattr(agent_cfg, "engine", "") or "").strip()
    name = str(getattr(agent_cfg, "name", "") or "")
    cur = (cur_model or "").strip()
    if cur and not is_engine_placeholder_model(cur, engine, name):
        return cur
    discovered = cold_external_model_for(agent_cfg)
    return discovered or cur


def format_model_list(
    rows: list[dict[str, Any]],
    *,
    current_endpoint: str = "",
    current_model: str = "",
) -> str:
    lines = ["configured models:"]
    current_norm = (current_endpoint or "").rstrip("/")
    current_id = (current_model or "").strip()
    ep_known = bool(current_norm)
    id_known = bool(current_id)
    # How many configured rows share the active endpoint — used to keep
    # the endpoint-only fallback from marking EVERY sibling on a shared
    # mux endpoint as "current" when the caller gave us no id to pick by.
    ep_match_count = sum(1 for m in rows if (m.get("baseUrl") or "").rstrip("/") == current_norm) if ep_known else 0
    for m in rows:
        # Older configs may omit the optional short alias while still
        # providing the required wire id. The wire id is a valid exact
        # selector, so expose it instead of rendering an unusable ``?``.
        alias = m.get("alias") or m.get("id") or "?"
        # ``/v1/models`` auto-discovery has been removed — every row
        # now requires an explicit ``id``. Surface ``(missing id)``
        # for any row that still has an empty id so the operator
        # spots the misconfig before they try to /model it.
        model_id = m.get("id") or "(missing id)"
        backend = m.get("backend") or "?"
        url = (m.get("baseUrl") or "").rstrip("/")
        ctx = int(m.get("contextWindow") or 0)
        ctx_label = f"{ctx // 1024}k" if ctx else "?"
        # Mark the ACTIVE row. Match on endpoint AND model id when both
        # are known: a single endpoint (e.g. the vLLM mux router on
        # :8000) can front several model ids, so endpoint-only matching
        # would mis-mark a sibling alias as "current". Degrade to
        # whichever key is known when the caller supplies only one.
        row_id = str(m.get("id") or "")
        if ep_known and id_known:
            active = (bool(url) and url == current_norm) and row_id == current_id
        elif ep_known:
            # Endpoint-only: mark it ONLY when the endpoint uniquely names
            # a row. On a shared mux endpoint (>1 row) we can't tell which
            # sibling is active without an id, so mark none rather than
            # falsely flag all of them.
            active = bool(url) and url == current_norm and ep_match_count == 1
        elif id_known:
            active = row_id == current_id
        else:
            active = False
        mark = "● " if active else "  "
        caps = m.get("capabilities") or []
        cap_label = ""
        if isinstance(caps, list) and caps:
            cap_label = " · " + ",".join(str(c).strip() for c in caps if str(c).strip())
        lines.append(f"{mark}{alias}: {model_id} · {backend} · ctx={ctx_label}{cap_label}")
        lines.append(f"    {url or '(no baseUrl)'}")
    lines.append("")
    lines.append("switch: /model <alias|url>  (list anytime with /models)")
    if current_model:
        lines.append(f"current: {current_model}")
    return "\n".join(lines)


def local_model_rows(cfg: Any) -> list[dict[str, Any]]:
    """Runtime ``cfg.models`` for local_llm, including Codex OAuth rows.

    Codex rows are folded into ``cfg.models`` once at config load / reload,
    not on every picker read. Codex CLI models stay on the ``codex`` agent;
    these rows let jarvis / build / plan use the same subscription as a
    worker transport without ``codex exec``.
    """
    try:
        from sharedagents_agent.discovery.openai_codex import selectable_local_models

        rows = selectable_local_models(cfg)
    except Exception:  # noqa: BLE001
        rows = [dict(r) for r in (getattr(cfg, "models", None) or []) if isinstance(r, dict)]
    # The web refresh may run in a separate process from the long-lived
    # gateway. Merge rows written by that refresh into the gateway's local
    # catalogue so a newly advertised provider id is selectable immediately.
    try:
        from sharedagents_agent.discovery.cache_lock import native_models_with_cache

        return native_models_with_cache(
            str(getattr(cfg, "name", "") or "jarvis"),
            rows,
        )
    except Exception:  # noqa: BLE001
        return rows


def resolve_model_row(cfg: Any, arg: str) -> dict[str, Any] | None:
    """Resolve `/model <arg>` without silently falling back on typos.

    `AgentConfig.resolve_model()` intentionally falls back to the primary
    model for compatibility. Commands need stricter behavior so a typo —
    e.g. `/model gpu-aa` when the alias is `gpu-a` — reports an error
    instead of silently switching to the default.
    """
    target = (arg or "").strip()
    if not target:
        return None
    if target.startswith("http"):
        return cfg.resolve_model(target)
    rows = local_model_rows(cfg)
    if not rows:
        primary = cfg._primary_model_row()
        if target == (primary.get("alias") or ""):
            return primary
        return None
    # Match by alias first (the canonical key), then fall back to the real
    # engine model id and the display name. The web picker carries the real
    # model id around (e.g. "deepseek-v4-flash" after a session load / state
    # patch) and applies it via `/model` on the first turn, so a raw id must
    # resolve here too. All matches are exact — a typo still reports an error
    # instead of silently switching to the default.
    for key in ("alias", "id", "display"):
        for row in rows:
            if target == (row.get(key) or ""):
                return dict(row)
    return None


def model_switch_text(result: dict[str, Any]) -> str:
    model_id = result.get("id") or result.get("model") or "(missing id)"
    url = result.get("baseUrl") or result.get("endpoint") or "?"
    ctx = int(result.get("contextWindow") or 0)
    ctx_label = f" · ctx={ctx // 1024}k" if ctx else ""
    msg = f"model → {model_id}\n{url}{ctx_label}"
    warn = result.get("upstream_error")
    if warn:
        msg += f"\nupstream warn: {str(warn)[:120]}"
    return msg
