"""Single source of truth for sharedagents on-disk layout.

All path math (sessions dir, sessions.db, logs dir, agent config dir)
lives here. Resolution order is documented inline and is honored by
every reader and writer in the sharedagents runtime.

The unified layout (operator goal 2026-05-26):

    ~/.sharedagents/
        config.json
        sessions/                  ← ONE folder, all agents share it
            sessions.db            ← ONE sqlite, one row per session
            <sid>.jsonl            ← session_started header tags agent
            .<sid>.lock
            <sid>.monitors.json
            <sid>.<engine>.json
        logs/                      ← ONE folder, one file per service
            gateway.log
            heartbeat.log
            tui.log
            telegram.log
        agents/<name>/             ← agent-LOCAL CONFIG only (no data)
            config.json
            hooks.json
            gateway.pid
            SOUL.md / AGENTS.md / TOOLS.md
            secrets.env
            session_reviews/

``SHAREDAGENTS_AGENT_NAME`` is the one process-scoped variable every log /
self-improver call uses to attribute a write to the calling agent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_LOG = logging.getLogger("sharedagents.kernel.layout")

#: Overrides already refused by the sandbox-escape guard below — each
#: unique (override, fallback) pair is logged ONCE so a polling monitor
#: (which resolves sessions_root() every refresh) doesn't spam stderr.
_REFUSED_SESSIONS_OVERRIDES: set[tuple[str, str]] = set()


def sharedagents_root() -> Path:
    """Top of the sharedagents runtime tree.

    Resolution order (most specific wins):

    1. ``$SHAREDAGENTS_AGENTS_HOME`` (kept for backward compat) — exact
       agents directory; the root is its parent. Used by tests that
       point at a tmp dir's ``agents/`` subdir.
    2. ``$SHAREDAGENTS_HOME`` — workspace root. Set this to redirect the
       whole runtime tree to e.g. ``~/.sharedagents-test``.
    3. ``~/.sharedagents``.
    """
    base = os.environ.get("SHAREDAGENTS_AGENTS_HOME")
    if base:
        return Path(base).expanduser().parent
    home = os.environ.get("SHAREDAGENTS_HOME")
    if home:
        return Path(home).expanduser()
    return Path.home() / ".sharedagents"


def agents_root() -> Path:
    """Where per-agent CONFIG dirs live (NOT data).

    Honors ``$SHAREDAGENTS_AGENTS_HOME`` (exact path) and ``$SHAREDAGENTS_HOME``
    (workspace root; agents live under ``<root>/agents``).
    """
    base = os.environ.get("SHAREDAGENTS_AGENTS_HOME")
    if base:
        return Path(base).expanduser()
    return sharedagents_root() / "agents"


def agent_config_dir(name: str) -> Path:
    """The agent-local CONFIG dir for ``name``.

    Holds ``config.json``, ``hooks.json``, ``gateway.pid``, ``SOUL.md``,
    ``secrets.env``, ``session_reviews/``. NOT sessions, NOT logs —
    those live in the unified locations.
    """
    return agents_root() / name


def _is_sandbox_escape_override(override_path: Path, root: Path) -> bool:
    """True when an explicit sessions override must be refused.

    Refusal is deliberately NARROW. The gateway injects the OPERATOR's
    live sessions dir into every tool-call subprocess
    (``sharedagents_gateway/tool_execution.py:361``); an agent's bash tool
    running pytest therefore inherited ``SHAREDAGENTS_SESSIONS_DIR=~/.sharedagents/
    sessions`` while the checkout's conftest redirected ``SHAREDAGENTS_HOME``
    to a tmp/test home — and ``sessions_root()`` preferred the override,
    silently writing test sessions into the LIVE store (the phantom
    ``rt``/``reload-test`` rows). So: an override that resolves to the
    DEFAULT live store (``~/.sharedagents/sessions``) while the effective home
    is redirected elsewhere is refused. Every other override — sandbox
    homes, ``<home>/alt-sessions``, arbitrary tmp dirs — stays honored
    (that is the documented test/sandboxing knob).
    """
    try:
        override_r = override_path.resolve()
        live_r = (Path.home() / ".sharedagents" / "sessions").resolve()
        root_r = root.resolve()
    except OSError:
        return False
    if override_r != live_r:
        return False
    return root_r != Path.home().resolve() / ".sharedagents"


def sessions_root() -> Path:
    """The ONE sessions folder shared by every agent.

    Override via ``$SHAREDAGENTS_SESSIONS_DIR`` for tests / sandboxing.
    Falls back to ``<sharedagents_root()>/sessions``.

    Sandbox-escape guard: when the effective runtime home is redirected
    away from the default (``SHAREDAGENTS_HOME``/``SHAREDAGENTS_AGENTS_HOME`` set to
    a test/tmp home), an ``SHAREDAGENTS_SESSIONS_DIR`` that resolves to the
    operator's live ``~/.sharedagents/sessions`` is REFUSED — the sandboxed
    home's sessions dir is returned instead, so a test run can never
    write the live store.
    """
    root = sharedagents_root()
    override = os.environ.get("SHAREDAGENTS_SESSIONS_DIR")
    if override:
        override_path = Path(override).expanduser()
        if _is_sandbox_escape_override(override_path, root):
            fallback = root / "sessions"
            key = (str(override_path), str(fallback))
            if key not in _REFUSED_SESSIONS_OVERRIDES:
                _REFUSED_SESSIONS_OVERRIDES.add(key)
                _LOG.warning(
                    "sessions_root: refusing SHAREDAGENTS_SESSIONS_DIR=%s — it "
                    "resolves to the operator's live store while the runtime "
                    "home is %s; falling back to %s (sandbox-escape guard, "
                    "see sharedagents_gateway/tool_execution.py:361)",
                    override_path,
                    root,
                    fallback,
                )
            return fallback
        return override_path
    return root / "sessions"


def sessions_db_path() -> Path:
    """The ONE sessions.db shared by every agent."""
    return sessions_root() / "sessions.db"


def goals_root() -> Path:
    """Durable goals storage under the sharedagents runtime tree."""
    return sharedagents_root() / "goals"


def goals_db_path() -> Path:
    """SQLite registry for durable goals (outlive sessions)."""
    return goals_root() / "goals.db"


def loops_root() -> Path:
    """Durable scheduled-loop storage under the sharedagents runtime tree."""
    return sharedagents_root() / "loops"


def loops_db_path() -> Path:
    """SQLite registry for scheduled loops + cron jobs (outlive sessions)."""
    return loops_root() / "loops.db"


def tasks_root() -> Path:
    """Durable task-board storage under the sharedagents runtime tree."""
    return sharedagents_root() / "tasks"


def tasks_db_path() -> Path:
    """SQLite registry of web-tracked sessions (the Tasks tab)."""
    return tasks_root() / "tasks.db"


def usage_root() -> Path:
    """Durable token-cost accounting storage under the sharedagents tree."""
    return sharedagents_root() / "usage"


def usage_ledger_db_path() -> Path:
    """SQLite ledger of per-turn token cost (authoritative spend record)."""
    return usage_root() / "ledger.db"


def pricing_path() -> Path:
    """Default per-model pricing table location (USD per 1M tokens)."""
    return sharedagents_root() / "pricing.json"


def pricing_auto_path() -> Path:
    """Machine-managed pricing overlay (USD per 1M tokens).

    Written by ``sharedagents_model.pricing_sync`` from the model catalogue, never
    by hand: it prices models the bundled asset and the operator's file do not
    cover. Resolves BELOW ``pricing_path()`` and above the bundled asset, so an
    operator edit always wins and deleting this one file rolls every
    auto-populated rate back."""
    return sharedagents_root() / "pricing_auto.json"


def voice_models_root() -> Path:
    """Operator-downloaded wake-word/voice ONNX models.

    The shared feature + wake-word models are fetched on demand
    (``sharedagents web fetch-voice-models``): they are CC BY-NC-SA 4.0 and
    cannot ship inside the MIT-licensed package. MIT-licensed models that do
    ship (``silero_vad.onnx``) live in the package's ``static/models/``; this
    directory holds the rest. The web server resolves ``/static/models/<f>``
    here first, then falls back to the bundled files.
    """
    return sharedagents_root() / "voice-models"


def logs_root() -> Path:
    """The ONE logs folder shared by every agent.

    Override via ``$SHAREDAGENTS_LOGS_DIR`` for tests / sandboxing. Falls
    back to ``<sharedagents_root()>/logs``.
    """
    override = os.environ.get("SHAREDAGENTS_LOGS_DIR")
    if override:
        return Path(override).expanduser()
    return sharedagents_root() / "logs"


def gateway_logfile() -> Path:
    # The Windows launcher scopes all backend writers (including tool events)
    # to their role. Existing CLI/systemd environments retain gateway.log.
    role = os.environ.get("SHAREDAGENTS_COMPONENT_ROLE")
    return logs_root() / ("orchestrator.log" if role == "orchestrator" else "gateway.log")


def heartbeat_logfile() -> Path:
    return logs_root() / "heartbeat.log"


def tui_logfile() -> Path:
    return logs_root() / "tui.log"


def telegram_logfile() -> Path:
    return logs_root() / "telegram.log"


def current_agent_name() -> str:
    """Best-effort agent name for the current process.

    Set by the runtime when it spawns a gateway / heartbeat / tui /
    self-improver. Empty string when the process is not running under
    an agent context (e.g. one-shot CLI calls).
    """
    return (os.environ.get("SHAREDAGENTS_AGENT_NAME") or "").strip()


__all__ = [
    "sharedagents_root",
    "agents_root",
    "agent_config_dir",
    "sessions_root",
    "sessions_db_path",
    "goals_root",
    "goals_db_path",
    "loops_root",
    "loops_db_path",
    "voice_models_root",
    "logs_root",
    "gateway_logfile",
    "heartbeat_logfile",
    "tui_logfile",
    "telegram_logfile",
    "current_agent_name",
]
