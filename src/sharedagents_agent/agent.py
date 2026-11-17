"""Agent definition + on-disk layout.

An *agent* is a long-running sharedagents service that exposes an
OpenAI-compatible HTTP API extended with session CRUD and per-turn
pause/stop/resume. The default install ships only the ``main`` agent;
adding more agents is a matter of dropping a config under
``~/.sharedagents/agents/<name>/``.

On-disk layout per agent:

    ~/.sharedagents/agents/<name>/
        config.json           # AgentConfig persisted (port, upstream, defaults)
        AGENTS.md             # identity, user facts, local notes (operator-editable)
        gateway.pid           # written while the runtime is running
        logs/                 # rotating-free log files (gateway, heartbeat, telegram)
            gateway.log       # stdout+stderr of the runtime
            heartbeat.log     # background heartbeat scheduler
            telegram.log      # telegram bot subprocess (when enabled)
        sessions/<sid>.jsonl  # per-conversation append-only event log
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_core.config_io import config_write_lock, write_config_text

# The light global-config accessors and the shipped memory-synth template moved
# down into the kernel leaf so the memory engine imports them without depending
# on this heavy module. Re-exported here by identity so existing callers of
# ``from sharedagents_agent.agent import load_global_config`` (etc.) are unaffected.
from sharedagents_core.kernel.config import (  # noqa: E402,F401
    _MEMORY_SYNTH_TEMPLATE,
    global_config_path,
    load_global_config,
    memory_synth_template_text,
)


def sharedagents_root() -> Path:
    """Top of the sharedagents runtime tree (parent of `agents/`).

    Resolution order (most specific wins):

    1. ``$SHAREDAGENTS_AGENTS_HOME`` — exact agents directory; root is its
       parent. Used by the test suite to point at a tmpdir's
       ``agents/`` subdir.
    2. ``$SHAREDAGENTS_HOME`` — workspace root. Set this to redirect the
       whole runtime tree (config, agents, errors, sessions, memory)
       to e.g. ``~/.sharedagents-test`` for a sandboxed test run.
    3. ``~/.sharedagents``.
    """
    base = os.environ.get("SHAREDAGENTS_AGENTS_HOME")
    if base:
        # Mirror kernel `layout.sharedagents_root()` exactly: expand `~` so a
        # `~`-prefixed SHAREDAGENTS_AGENTS_HOME resolves to the identical path
        # both resolvers produce.
        return Path(base).expanduser().parent
    home = os.environ.get("SHAREDAGENTS_HOME")
    if home:
        return Path(home).expanduser()
    return Path.home() / ".sharedagents"


def agents_root() -> Path:
    """Where all per-agent dirs live (`<root>/agents/`).

    Honors ``$SHAREDAGENTS_AGENTS_HOME`` (exact path) and ``$SHAREDAGENTS_HOME``
    (workspace root, agents live under ``<root>/agents``). See
    :func:`sharedagents_root` for full resolution order.
    """
    base = os.environ.get("SHAREDAGENTS_AGENTS_HOME")
    if base:
        return Path(base).expanduser()
    return sharedagents_root() / "agents"


def _parse_dotenv(text: str) -> dict[str, str]:
    """Minimal dotenv parser used for ``~/.sharedagents/secrets.env`` and
    ``~/.sharedagents/agents/<name>/secrets.env``.

    Accepts:
      * Comments (``# …``) and blank lines (skipped).
      * Optional ``export `` prefix (stripped — lets the same file
        double as a shell-sourceable env file).
      * ``KEY=value`` pairs; surrounding single OR double quotes on
        ``value`` are stripped if matched.
      * Empty values (``EMPTY=`` → ``""``).
      * Lines without ``=`` are skipped.

    Deliberately does NOT support multi-line values, variable
    expansion, or escape sequences — secrets.env is for API keys,
    not shell scripts. Mirrors ``tests/conftest.py:_parse_dotenv`` so
    the runtime and the test harness agree on the wire shape.
    """
    out: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if k:
            out[k] = v
    return out


def _warn_if_world_readable(path: Path) -> None:
    """Log a WARNING if ``path`` is readable by group OR other.

    Doesn't refuse to load (behaviour change risk for first-time
    setups), just surfaces. The operator's response is a one-line
    fix: ``chmod 600 <path>``.
    """
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return
    if mode & 0o077:
        import logging

        logging.getLogger("sharedagents.agent").warning(
            "secrets file %s has loose permissions (mode=%o); group or other can read it. Run: chmod 600 %s",
            path,
            mode,
            path,
        )


def _load_agent_secrets(home: Path, name: str) -> dict[str, str]:
    """Resolve the API-key map for one agent.

    Layers (highest precedence first):
      1. ``<home>/agents/<name>/secrets.env`` — per-agent overrides.
      2. ``<home>/secrets.env`` — shared fallback for agents that
         haven't been minted distinct per-provider keys yet.

    Missing files are NOT an error — operators may run with all keys
    in process env (systemd ``EnvironmentFile`` / shell rc); the
    secrets.env layer just lets them attribute usage per-agent. Bad
    files (unreadable, malformed) raise so the gateway boot loudly
    fails instead of silently shipping requests with no Authorization.

    File-mode audit: ``_warn_if_world_readable`` logs a WARNING if
    the secrets file is group/other readable so the operator can
    catch a misconfigured editor / restored backup before keys leak.
    """
    out: dict[str, str] = {}
    # Shared first so per-agent overrides win on the second merge.
    shared = home / "secrets.env"
    if shared.is_file():
        _warn_if_world_readable(shared)
        out.update(_parse_dotenv(shared.read_text(encoding="utf-8")))
    per_agent = home / "agents" / name / "secrets.env"
    if per_agent.is_file():
        _warn_if_world_readable(per_agent)
        out.update(_parse_dotenv(per_agent.read_text(encoding="utf-8")))
    return out


# Fields that live in the GLOBAL config (`~/.sharedagents/config.json`), shared
# across every agent. Per-agent files only carry what actually differs
# between agents (name, port, description, system).
_GLOBAL_FIELDS: frozenset[str] = frozenset(
    {
        "upstream_endpoint",
        "upstream_model",
        "upstream_alias",
        # `models` is the canonical OpenClaw-style list of model
        # deployments. Shared across local_llm agents (jarvis/build/plan).
        # ``AgentConfig.load`` strips these from foreign engines unless the
        # per-agent file or builtin defaults own them — OpenRouter ids in
        # the global catalogue are also OpenCode catalogue ids, so inheriting
        # them made Web/Telegram associate OpenCode models with jarvis/cursor.
        # The legacy alias `upstream_endpoints` is still accepted by
        # `AgentConfig.load()` for back-compat with hand-edited configs from
        # before the rename.
        "models",
        "upstream_endpoints",  # legacy alias — see `_normalize_models()`
        "backend",
        "temperature",
        "sampling",
        "side_sampling",
        "effort",
        "enable_thinking",
        "host",
        "brave_api_key",
        "upstream_max_context",
        "compact_at_frac",
        "compact_min_turns",
        "compact_prune_on_load",
        # OpenCode-style small/utility model for hidden title + summary agents.
        # Empty → resolve via role/alias heuristics (see ``small_model_row``).
        "small_model",
        # NOTE: ``tools_allowlist`` is intentionally NOT here (same reasoning as
        # ``disabled_tools`` below). Listing it as a global field made
        # ``AgentConfig.save()`` drop it from the per-agent config, so a normal
        # load+save (CLI/TUI config edit, wizard) silently lost the allowlist —
        # and for a ``tools_allowlist_strict`` agent like ``jarvis`` that quietly
        # re-grants every core tool, breaking the delegate-only boundary (codex
        # 2026-06-13). It now persists per-agent; ``tools_allowlist_strict`` is
        # already per-agent, so the pair stays in sync. A hand-set global default
        # is still honoured via the ``{**glob, **per_agent}`` merge in ``load()``.
        # NOTE: ``disabled_tools`` is intentionally NOT here. It's a per-agent
        # tool blocklist (e.g. disable ``UserQuestion`` only for a Zed-facing
        # ACP agent). Listing it as a global field made ``AgentConfig.save()``
        # drop it from the per-agent config, so a normal save silently lost the
        # blocklist; it now persists per-agent. A hand-set global default is
        # still honoured via the full-dict merge in ``AgentConfig.load()``.
        # ---- MCP + ACP (see mcp_client / acp_client / acp_server) ----
        # Shared across agents because servers/peers/auth are deployment-
        # wide concerns, not per-agent quirks. Per-agent override is
        # possible via the existing `merged = {**glob, **per_agent}` path
        # in `AgentConfig.load()` if a future use case needs it.
        #
        # `mcpServers` (camelCase) is the canonical key — the MCP-ecosystem
        # standard and what `mcp_client.load_mcp_servers()` reads first; the
        # wizard writes it. `mcp_servers` (snake_case) is the legacy key,
        # kept here so hand-edited pre-camelCase configs still round-trip.
        # Both must be in this set or `save_global_config` silently drops
        # whichever the wizard wrote — the bug that left the wizard's
        # default MCP servers out of every fresh config.json.
        "mcpServers",
        "mcp_servers",
        "mcpDeferral",
        "acp_agents",
        "acp_server",
        # Per-skill enable/disable map: { "<skill-name>": {"enabled": bool} }.
        # The wizard seeds it with every discovered skill enabled; absent
        # entries default to enabled, so a skill not listed here still works
        # — only an explicit `enabled: false` turns one off. Read by
        # `skills.discover_skills()`.
        "skills",
        # Extra skill search roots (list of paths), prepended to the
        # discovery chain just below `SHAREDAGENTS_SKILLS_DIRS`. The persistent
        # way to register an external skill bundle (e.g. an
        # everything-claude-code checkout) so the gateway, TUI, and CLI
        # all see it. Read by `skills.skill_search_paths()`.
        "skills_dirs",
        # Extra file-command search roots (list of paths) — the same idea
        # as `skills_dirs` for Markdown slash commands (`commands/*.md`).
        # Read by `file_commands.command_search_paths()`.
        "commands_dirs",
        # Self-hosted inference services (primary/backup). Global by default;
        # an agent's per-agent config.json MAY carry its own block to override
        # via the `merged = {**glob, **per_agent}` path in `load()`.
        "stt",
        "tts",
        "embedding",
        "reranking",
        # Supervision + self-repair policy blocks belong in the global
        # config only. Persisting them per-agent (via load-merge-save
        # round-trips) bloats every ``agents/*/config.json`` with identical
        # copies of the global ``teacher`` / ``self_improver`` blocks.
        "teacher",
        "self_improver",
        # Gateway/orchestrator/telegram deployment blocks + hot-reload cadence,
        # external-model registry, and memory-embedding config are all read from
        # ``load_global_config()`` by live runtime modules; each must be in the
        # allowlist or ``save_global_config()`` silently drops it on the next save
        # The same data-loss class as the tools_allowlist and
        # mcpServers incidents above).
        "gateway",
        "orchestrator",
        "telegram",
        "config_watch_interval_s",
        "external_models",
        "memory_embedding",
    }
)


# Per-agent dataclass fields that MUST NOT be persisted into
# ``~/.sharedagents/agents/<name>/config.json`` by ``AgentConfig.save()``.
# These are loaded into memory at ``AgentConfig.load()`` from secure
# external sources and would leak if round-tripped through the
# (default mode 644, not gitignored by name) JSON file.
_NON_PERSISTED_FIELDS: frozenset[str] = frozenset(
    {
        "api_keys",
    }
)


def _operator_model_values(per_agent: dict[str, Any]) -> dict[str, Any]:
    """The model route + parameters the agent's own config file holds.

    A lean profile save re-writes exactly these (see ``AgentConfig.save``);
    inherited global values are absent here, so they are never frozen into a
    profile file.
    """
    from sharedagents_agent.agents.default_scaffolds import OPERATOR_MODEL_FIELDS

    return {
        k: copy.deepcopy(per_agent[k]) for k in OPERATOR_MODEL_FIELDS if k in per_agent and per_agent[k] != "_UNSET"
    }


# Local-LLM engines keep the global catalogue. Everything else is a
# foreign engine whose model identity lives in that engine's own menu.
_LOCAL_LLM_ENGINE_NAMES = frozenset({"", "local_llm"})


def _drop_inherited_local_route(
    merged: dict[str, Any],
    *,
    per_agent: dict[str, Any],
    builtin_defaults: dict[str, Any],
) -> None:
    """Strip the global local-LLM route from a foreign-engine merge.

    Mutates ``merged`` in place. Jarvis/build/plan (``local_llm``) keep
    the shared ``models[]`` / ``upstream_model``. Cursor, OpenCode,
    Claude, Codex, Kimi, Hermes, Antigravity keep only the route fields
    they (or their builtin defaults) explicitly own — so a global
    ``openrouter/auto`` cannot become those agents' identity.

    Codex has no builtin model pin: its default is its
    own per-agent ``upstream_model``, else codex's ``config.toml`` model.
    """
    engine = str(merged.get("engine") or "").strip() or "local_llm"
    if engine in _LOCAL_LLM_ENGINE_NAMES:
        return
    owned = set(per_agent) | set(builtin_defaults)
    if "models" not in owned and "upstream_endpoints" not in owned:
        merged["models"] = []
        merged.pop("upstream_endpoints", None)
    if "upstream_model" not in owned:
        merged["upstream_model"] = ""
    if "upstream_alias" not in owned:
        merged["upstream_alias"] = ""
    if "upstream_endpoint" not in owned:
        merged["upstream_endpoint"] = ""
    if "backend" not in owned:
        merged["backend"] = ""
    if "upstream_max_context" not in owned:
        merged.pop("upstream_max_context", None)


def derive_upstream_alias(endpoint: str) -> str:
    """Extract a friendly host alias from an upstream URL.

    `http://host:8001/v1` → `host`. `http://gpu-a:8001/v1`
    → `gpu-a`. Used as the wizard's default for
    `upstream_alias` and as the runtime fallback when no alias is
    configured. Returns "" on a malformed / empty URL.
    """
    if not endpoint:
        return ""
    try:
        from urllib.parse import urlparse

        return urlparse(endpoint).hostname or ""
    except Exception:  # noqa: BLE001
        return ""


# Per-agent baked-in defaults for built-in sibling agents (``coder``,
# ``scraper``, …). Applied in :meth:`AgentConfig.load` between the
# global config and the per-agent file so the per-agent file STILL
# wins on explicit overrides, but a fresh install of a known sibling
# agent gets a sensible upstream without forcing the operator to
# scaffold-and-edit ``~/.sharedagents/agents/<name>/config.json`` by hand.
#
# ``coder`` is intentionally NOT pinned here (see the leaf-implementer note
# on the dict below): it runs on the local model like ``main``. The
# corresponding ``models[]`` row (with ``apiKeyEnv`` etc.) for any pinned
# alias lives in the global config; this map only pins which alias is the
# default for the named agent.
_BUILTIN_AGENT_DEFAULTS: dict[str, dict[str, Any]] = {
    # `coder` intentionally has NO built-in upstream pin: it is a LEAF implementer
    # that runs on the local model (inherits the global `~/.sharedagents/config.json`
    # upstream — local Qwen, not a remote API). Its focused tool/skill surface is
    # supplied by the CODER AgentProfile (agent_profiles.py) + the per-agent
    # config.json, so there is nothing to pin here.
    #
    # The self-improver's own brain is the cheap deepseek-v4-flash triage model
    # (alias ``dpsk``). It is a standard orchestrator agent; its autonomous
    # triage→fix pipeline is driven by ``sharedagents self-improver`` (orchestrator
    # scope), NOT by the gateway. The claude‖codex fix loop is delegated.
    "self-improver": {
        "upstream_alias": "dpsk",
        "upstream_endpoint": "https://api.deepseek.com/v1",
        "upstream_model": "deepseek-v4-flash",
        "upstream_max_context": 1_000_000,
        # Triage→fix orchestrator: scope skills to delegation/review (it delegates the
        # claude‖codex fix loop). Tools left unrestricted — it's a full orchestrator.
        "skill_allowlist": ["delegate-codex"],
        "skill_allowlist_strict": True,
    },
    # NO builtin upstream pin for codex (hardcoded model fallbacks removed
    # 2026-09-23): a pinned id goes stale as the backend's list changes. The
    # codex default is the operator's own choice — this agent's per-agent
    # ``upstream_model``, else codex's ``config.toml`` model — and never the
    # merged global/local-LLM upstream (see
    # ``discovery._common._codex_effective_default``). The key stays so
    # ``codex`` remains a known agent name.
    "codex": {},
    # NO builtin upstream pin for claude: a
    # pinned ``upstream_model`` merged into every claude profile and forced
    # ``--model claude-sonnet-5`` on EVERY fresh spawn, shadowing the
    # operator's own selection (``~/.claude/settings.json`` model /
    # ``ANTHROPIC_MODEL`` env — the merge ran before the default chain ever
    # consulted them). With no explicit sharedagents model selection the Claude
    # CLI applies its own precedence; sharedagents passes no ``--model``
    # (``external_cli_effective_default`` → ``""`` for ``claude_cli``).
    "claude": {},
}


def save_global_config(data: dict[str, Any]) -> Path:
    """Persist shared settings to `~/.sharedagents/config.json`.

    Only fields in `_GLOBAL_FIELDS` are written; unknown keys are
    silently dropped so callers can't leak per-agent fields here.

    Defensive snapshot: before overwriting an existing file, copy it
    to ``config.json.bak.<UTC-timestamp>``. The wizard (and other
    paths through this function) sometimes shrink ``models[]`` to a
    single row; without an automatic undo lane the prior list is
    gone the instant the new file lands. Retention prunes the
    snapshots back to the newest 10 so the directory doesn't grow
    without bound on operators who reconfigure often.
    """
    clean = {k: v for k, v in data.items() if k in _GLOBAL_FIELDS}
    p = global_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with config_write_lock(p):
        if p.exists():
            try:
                # Microsecond precision so two saves inside the same wall-
                # clock second produce distinct snapshot names — without
                # this, rapid `save_global_config` calls (tests, scripted
                # reconfigures) collide on a single ``.bak.<ts>`` and
                # retention has nothing to prune.
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                bak = p.with_name(f"{p.name}.bak.{ts}")
                bak.write_bytes(p.read_bytes())
            except OSError:
                # Snapshot is best-effort; never block a successful save
                # because the backup write happened to fail.
                pass
            else:
                _prune_global_config_backups(p, keep=10)
        p.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def _prune_global_config_backups(p: Path, *, keep: int) -> None:
    """Trim ``config.json.bak.*`` snapshots to the newest ``keep`` files."""
    try:
        snapshots = sorted(
            p.parent.glob(f"{p.name}.bak.*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in snapshots[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass


def _env_float(name: str, default: float) -> float:
    """Float from env ``name``, falling back to ``default`` when unset/blank/bad.

    Lets operators override cost-budget defaults without editing a config:
    e.g. ``SHAREDAGENTS_MAX_GOAL_COST_USD=100 SHAREDAGENTS_MAX_PROMPT_COST_USD=25``.
    A per-agent ``config.json`` value still wins (it's an explicit field);
    this only changes the DEFAULT applied when the field is unset.
    """
    raw = os.environ.get(name, "")
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


@dataclass
class ConfigReloadReport:
    """Result of :meth:`AgentConfig.reload_from_disk`.

    ``applied`` — hot-reloadable field names mutated on the live cfg.
    ``restart_only`` — structural field names that CHANGED on disk but were
    deliberately not applied (a restart is required to pick them up).
    ``unchanged`` — True when nothing on disk differed from the live cfg.
    """

    applied: list[str] = field(default_factory=list)
    restart_only: list[str] = field(default_factory=list)
    unchanged: bool = True


@dataclass
class AgentConfig:
    """Settings for one sharedagents agent.

    The on-disk split:

    - **Global** (`~/.sharedagents/config.json`): `upstream_endpoint`,
      `upstream_model`, `backend`, `temperature`, `enable_thinking`,
      `host` — shared across every agent so you don't restate the
      LLM endpoint per-agent.
    - **Per-agent** (`~/.sharedagents/agents/<name>/config.json`):
      `name`, `description`, `port`, `system` — what actually differs
      between agents.

    The dataclass keeps a flat surface (callers do `cfg.upstream_endpoint`
    just like before); `load()` merges global + per-agent and `save()`
    writes only per-agent fields. Per-agent files MAY also carry a
    global field as a per-agent override — handy for spinning up an agent
    that points at a different upstream than the default.
    """

    name: str
    description: str = ""
    host: str = "127.0.0.1"
    # Legacy per-agent port — vestigial under the single-process
    # gateway (every agent is mounted at ``/acp/<name>`` on the ONE
    # gateway port). Kept for the retired per-agent-daemon paths that
    # still read ``cfg.port``; defaulted to the gateway's :8100 so a
    # stray reader can never disagree with the live listener.
    port: int = 8100
    # No default endpoint — the first-run wizard makes the user pick one.
    # Hardcoding a host name would leak the author's env into a
    # public default and mean a fresh install silently points at a server
    # that doesn't exist on most machines.
    upstream_endpoint: str = ""
    upstream_model: str = ""  # required — empty rejects at session/new (no auto-discover)
    # Friendly host alias displayed in the TUI status bar (e.g. "host"
    # so the bar shows `model=host/qwen3.6-27b-fp8` instead of the IP).
    # Empty → fall back to the URL's hostname at render time.
    upstream_alias: str = ""
    # Configured model deployments — the OpenClaw-style list of models
    # sharedagents can talk to. Each row:
    #   {
    #     "alias": "gpu-a",                          # short user-facing handle
    #     "baseUrl": "http://gpu-a:8001/v1", # OpenAI-compat endpoint
    #     "id": "qwen3.6-27b-fp8",                  # model id sent in chat requests
    #     "backend": "vllm",                        # sharedagents-specific: vllm|sglang|unknown
    #     "contextWindow": 262144,                  # max input + output tokens
    #     "maxTokens": 8192,                        # optional model output limit; sends min(this, cap) (0 = discover)
    #     "apiKeyEnv": "OPENROUTER_API_KEY",       # optional Bearer token env var
    #     "headers": {"HTTP-Referer": "..."}       # optional extra upstream headers
    #   }
    # The legacy field name was `upstream_endpoints` with per-row keys
    # `endpoint` / `model` / `max_context`; `AgentConfig.load()`
    # transparently migrates loaded data to the new shape via
    # `_normalize_models()` so existing configs keep working.
    models: list[dict] = field(default_factory=list)
    # Teacher-model supervision policy.
    # The teacher *endpoint* is a normal `models[]` row tagged
    # `role == "teacher"`; this block holds the supervision policy. Empty
    # / `enabled=false` means teacher supervision is off.
    teacher: dict = field(default_factory=dict)
    # Legacy ``goal`` config block (the removed /goal evaluator's policy).
    # Still parsed so older config.json files load; nothing reads it.
    goal: dict = field(default_factory=dict)
    # Outer /goal turn budget surfaced on /status (global config may override).
    # Retained for display/back-compat; the loop terminator is now the goal
    # COST budget below, not this turn cap.
    max_goal_turns: int = 0  # 0 == unbounded; cost is the only limit on goal work
    # Outer /goal COST budget (USD). The loop re-drives until the goal's
    # accumulated spend (usage-ledger SUM) reaches this, or the goal is
    # met/blocked. Default $25.00 (the budget is never
    # unbounded; 0/unset floors to the default in goal_service.resolve_budget_usd).
    # Override: $SHAREDAGENTS_MAX_GOAL_COST_USD.
    max_goal_cost_usd: float = field(default_factory=lambda: _env_float("SHAREDAGENTS_MAX_GOAL_COST_USD", 25.0))
    # Per-session COST cap (USD). 0 == unbounded. New auto-turns are refused
    # once the session's accumulated spend crosses this.
    max_session_cost_usd: float = 0.0
    # Per-PROMPT COST cap (USD) for non-goal interactive turns. 0 == unbounded.
    # A single user prompt's spend is bounded here; goal turns use the goal
    # budget, never this. Default $10.00 (override: $SHAREDAGENTS_MAX_PROMPT_COST_USD).
    max_prompt_cost_usd: float = field(default_factory=lambda: _env_float("SHAREDAGENTS_MAX_PROMPT_COST_USD", 10.0))
    # Override path for the per-model pricing table; "" == ~/.sharedagents/pricing.json.
    pricing_path: str = ""
    goal_outer_loop_enabled: bool = True
    goal_outer_loop_delay_s: float = 2.0
    goal_outer_loop_error_delay_s: float = 10.0
    goal_outer_loop_max_errors: int = 3
    goal_outer_loop_manual_interrupt_policy: str = "yield"
    # Claude Code /goal no-progress guard: this many consecutive goal turns
    # without a tool call pause the goal until the user's next message.
    goal_no_progress_turns: int = 3
    # Legacy knobs of earlier goal loops. Still parsed so older config.json
    # files load; nothing reads them.
    goal_outer_loop_alert_streak: int = 4
    goal_outer_loop_stall_retry_s: float = 300.0
    goal_outer_loop_max_ineffective: int = 3
    goal_outer_loop_no_progress_idle_s: float = 120.0
    goal_outer_loop_reserve_last_turn: bool = True
    goal_outer_loop_compact_directive: bool = True
    self_improver: Any = field(default_factory=dict)  # bool (simple on/off) or dict
    # Self-hosted inference services (STT/ASR, TTS, embedding, reranking) — each a
    # `{"primary": {...}, "backup": {...}}` block resolved by
    # `sharedagents_core.services.resolve_service`. Configured globally; an agent MAY
    # carry its OWN block here to override the global primary/backup for that
    # service (shallow replace — the agent gets its own primary AND fallback).
    # A flat `{"base_url", "model"}` is accepted and treated as the primary.
    stt: dict = field(default_factory=dict)
    tts: dict = field(default_factory=dict)
    embedding: dict = field(default_factory=dict)
    reranking: dict = field(default_factory=dict)
    # Backend kind: "vllm", "sglang", or "" (unknown / auto-detect).
    # Detected from the upstream's `/v1/models[].owned_by` field at wizard
    # time. Surfaced in `/v1/control/status` so frontends can show it; the
    # streaming router is already alias-aware (`delta.reasoning` vs
    # `delta.reasoning_content`), so this is mostly informational today.
    backend: str = ""
    # The agent's "system" identity is composed from the injected context
    # files (SOUL/USER/TOOLS/AGENTS/MEMORY) under the agent's home dir.
    # `system` is opt-in: when set, it's appended as a final # SYSTEM
    # section after the injected files. Leave empty in the common case.
    system: str = ""
    temperature: float | None = None
    sampling: dict = field(default_factory=dict)
    side_sampling: dict = field(default_factory=dict)
    # Agent-level reasoning-depth default. An attached session override wins.
    # Empty means no implicit request level; model configuration may opt into
    # a default. An explicitly configured agent value is still honoured.
    effort: str = ""
    enable_thinking: bool = True
    # Brave Search API key for the `mcp__web__search` tool. Stored in the global
    # config; never logged. Pass `BRAVE_API_KEY` via env to override per
    # invocation. Empty string disables the tool's network call (the
    # runner returns a friendly error pointing the user at the wizard).
    brave_api_key: str = ""
    # Permission mode for tool calls: "plan" / "ask" / "auto".
    # See `tools.permissions.PermissionMode`. Stored as a string here so the
    # config file stays JSON-friendly; the runtime maps it to the enum.
    # Auto is the safe default — destructive ops always re-prompt; the user
    # can switch to plan or ask via Shift+Tab in the TUI.
    permission_mode: str = "auto"
    # OpenCode-style profile id (``explore``, ``plan``, ``build``, …).
    # When set (or when ``name`` matches a known profile), ``save()``
    # persists the canonical lean profile ``config.json`` scaffold.
    agent_profile: str = ""
    # Upstream model context window in tokens (input + output). Used by
    # the runtime's compaction guard: when an upcoming round's estimated
    # prompt + max_tokens would push past `compact_at_frac * this`, we
    # squeeze old tool results before sending. 96000 is qwen3.6-27b-fp8's
    # default; override via wizard / config for other models.
    upstream_max_context: int = 96000
    # Proactive compaction trigger — occupancy at 75% of the live context
    # window is always on (see ``_COMPACT_OCCUPANCY_FRAC``). This field is
    # an additional output-aware gate: when
    # `prompt_tokens + max_tokens >= compact_at_frac × upstream_max_context`,
    # the runtime also compacts before the next round. Default `1.0` means
    # this extra gate only fires at the wall; the 75% occupancy rule still
    # runs. Set lower (e.g. 0.85) to compact even earlier.
    compact_at_frac: float = 1.0
    # Minimum session turns that must elapse between proactive compacts
    # (OpenCode ``min_messages``). 0 disables. Overflow, 75% occupancy,
    # and force bypass.
    compact_min_turns: int = 3
    # OpenCode ``compaction.prune`` (default OFF there too). When True, old
    # tool results are erased every time a session is loaded from disk, not
    # only when compacting. That rewrites earlier prompt messages, which
    # invalidates the upstream prefix cache (vLLM APC) from that point, so the
    # next call re-prefills the history: slower time-to-first-token. Leave it
    # off; tool results are pruned and summarized when the session compacts.
    compact_prune_on_load: bool = False
    # OpenCode ``small_model`` — alias or wire id for hidden title/summary
    # (and similar side) LLM calls. Empty → ``small_model_row()`` heuristics
    # (role ``small`` / aliases dpsk|haiku|fast|small), then primary.
    small_model: str = ""
    # Per-agent tool allowlist. When set, restricts the catalog the
    # bridge sends to the upstream and dispatches at runtime to only
    # these names — used for lean sub-agents (the `coder`, `scraper`)
    # that don't need skills, memory, MCP, delegate_*, or user-facing
    # tools like UserQuestion. ``None`` (default) = no restriction;
    # the agent sees the full ``registry.core_tools()`` plus promoted.
    tools_allowlist: list[str] | None = None
    # When True AND ``tools_allowlist`` is set, the allowlist is AUTHORITATIVE:
    # the usual core-tool union is SKIPPED, so the agent sees EXACTLY the listed
    # tools (used by delegate-only agents like ``jarvis``). Default False
    # preserves every existing agent's behavior. Per-agent (not in the global
    # serialized-fields list, same as ``disabled_tools``).
    tools_allowlist_strict: bool = False
    # Per-agent SKILL permission — narrows the skill catalog the model sees (helps
    # small models focus). ``None`` = no restriction. Semantics mirror tools_allowlist
    # (see agent_profiles.AgentProfile.skill_allowlist). Per-agent, not globally serialized.
    skill_allowlist: list[str] | None = None
    skill_allowlist_strict: bool = False
    # Per-agent tool blocklist. When set, these tools are excluded from
    # the catalog the bridge sends to the upstream and from runtime
    # dispatch — used to disable tools that are incompatible with the
    # connected ACP client (e.g. ``UserQuestion`` / ``AskUserQuestion``
    # hang in Zed because it doesn't implement ``session/ask_user``).
    # ``None`` (default) = no restriction. Evaluated AFTER
    # ``tools_allowlist`` so both can be combined: allowlist first,
    # then remove any disabled names.
    disabled_tools: list[str] | None = None
    # MCP servers whose tools ship on every round for this agent even when it
    # reaches them only through implicit access. Other implicitly
    # reachable servers are offered on demand (``load_tools``). ``None`` = the
    # defaults (builtin memory/web/delegate). Per-agent, not globally serialized.
    mcp_always_loaded: list[str] | None = None
    # Per-agent list of disabled HARDCODED system-prompt section
    # identifiers (the Agent Builder's enable/disable cards).
    # Each id is a stable slug of the section title (see
    # ``prompt_section_slug``); an empty/absent list means every section
    # is enabled. Only hardcoded sections are filterable — agents_md /
    # system / dynamic composition is untouched. Per-agent (not in the
    # global serialized-fields list, same as ``disabled_tools``).
    disabled_prompt_sections: list[str] = field(default_factory=list)
    # Per-agent delegation-mode default for first-party ``Agent`` calls
    # ``"background"`` pins every
    # Agent dispatch to background (jarvis scaffold default), ``"foreground"``
    # pins them inline (build/plan scaffold default), ``"default"`` explicitly
    # opts back into the unpoliced legacy semantics. The runtime resolver
    # reads the raw on-disk value from this agent's config.json (via
    # ``SHAREDAGENTS_AGENT_HOME``), so this field is the typed mirror used by
    # scaffolding, save-round-trip and the Agent Builder. Absent ⇒ the applied
    # profile's default seeds it (apply_profile_to_config); a global config
    # value is NEVER inherited (see load()).
    delegation_mode: str | None = None
    # Auto-complete matching TodoWrite items when a subagent
    # (Agent / delegate_* tool) returns success. Small models
    # routinely forget to call TodoWrite to flip their own plan
    # items; the harness auto-completes the todo whose content
    # text matches the spawned subagent. Set ``False`` to opt
    # out per-agent (e.g. for ``dashboard_dev`` where the human
    # is running ad-hoc agents and the plan is aspirational).
    auto_complete_todos: bool = True
    # Master kill-switch for ALL loop/nudge/supervision "protections"
    # (text-loop tiers, strict+broad tool-call-repetition, cross-round
    # xround-semantic, premature-end nudge, teacher supervision, and the
    # loop-recovery/force-terminate cascade). Purely SUBTRACTIVE: True
    # disables everything for THIS agent; it never force-ENABLES a
    # protection that is otherwise off. Defaults False (protections on).
    # There is also a per-model-row ``disableProtections`` flag (on a
    # ``models[]`` entry in config.json) resolved per inference-endpoint —
    # EITHER source being true turns protections off for the turn. Meant
    # for big/hosted models (Opus via API) that don't need small-local-
    # model babysitting; the per-row flag lets the SAME base model be
    # guarded on one quant/endpoint and unguarded on another.
    disable_protections: bool = False
    # Sustained-zero-progress recovery thresholds (local_llm turn engine).
    # A per-turn ``zero_progress_streak`` counts rounds that make no progress
    # (no new tool target, no mutation, no new tool-result content, no new
    # visible content; a still-running background-job poll is a neutral
    # carve-out). At ``progress_nudge_streak`` the engine injects a plain
    # re-grounding user message; at ``progress_hardstop_streak`` it injects a
    # "finalize now" message, allows one more round, then ENDS the turn
    # gracefully with whatever answer exists (terminal_state "ok" — never an
    # error). This REPLACES the old surface-repetition distill→force-terminate
    # loop response, which an FP audit measured at ~0-15% precision. Env
    # overrides: SHAREDAGENTS_PROGRESS_NUDGE_STREAK / SHAREDAGENTS_PROGRESS_HARDSTOP_STREAK.
    # 0 → fall back to the module/env default (3 / 8).
    progress_nudge_streak: int = 3
    progress_hardstop_streak: int = 8
    # --- Runtime routing / engine knobs (previously read via getattr but
    # NOT declared, so on-disk values were silently dropped by load()'s
    # dataclass-field filter). Declaring
    # them here makes them actually loadable/configurable. Defaults preserve
    # the prior hardcoded getattr fallbacks exactly. ---
    #
    # Deterministic DIRECT-routing guard for orchestrators. None → decide by
    # name (main/jarvis default on); True/False → explicit override. Consumed
    # by LocalLLMEngine._route_guard_target.
    route_guard_enabled: bool | None = None
    # Same-round transient-error retry cap (small-model/muxed-endpoint hiccups).
    # None → module default (_MAX_TRANSIENT_RETRIES).
    max_transient_retries: int | None = None
    # Global output-token cap for this agent (OpenCode's OUTPUT_TOKEN_MAX):
    # requests send min(model output limit, cap). None → $SHAREDAGENTS_OUTPUT_TOKEN_MAX
    # → 32000. See sharedagents_model.models.resolve_max_tokens.
    upstream_max_completion_tokens: int | None = None
    # LLM-driven context compaction. True (default) keeps it on.
    llm_compaction: bool = True
    # Auto-continue a turn the model ended prematurely. True (default) keeps it on.
    auto_complete_continue: bool = True
    # Max distill-based recovery attempts per turn. None → 3.
    distill_max_recoveries: int | None = None
    # Stop-hook continuation ceiling. None → module default
    # (_MAX_STOP_CONTINUATIONS=0, OpenCode parity). 0 → no auto-complete
    # continuations. (The old ``value or 3`` read made 0 impossible.)
    max_stop_continuations: int | None = None
    # NOTE: Telegram-bot configuration used to live here
    # (``telegram_bot_token`` + ``telegram_allowed_chat_ids``). The
    # bot is now a standalone service with its own config at
    # ``~/.sharedagents/telegram/config.json``; the legacy fields are no
    # longer read. Old per-agent ``config.json`` files that still
    # carry them are tolerated (``AgentConfig.load`` ignores unknown
    # keys) and the values can be migrated with
    # ``sharedagents telegram migrate``.
    # Heartbeat scheduler. Periodically fires a turn into a DEDICATED
    # heartbeat session (separate from any user session) so the
    # prefix cache stays warm tick-to-tick. The agent reads
    # `HEARTBEAT.md` and decides whether to do background work (check
    # email, calendar, summarize other sessions…) or reply
    # `HEARTBEAT_OK` and stay quiet. Mirrors OpenClaw's heartbeat
    # behavior — see `heartbeat.py:HEARTBEAT_PROMPT`. State (current
    # heartbeat session id, per-session last-analyzed turn idx) lives
    # in `<home>/heartbeat-state.json` so the model can ask "what's
    # new since last tick?" without redoing analysis.
    heartbeat_enabled: bool = False
    # 30 minutes — matches OpenClaw's "every ~30 min is fine" framing
    # in the canonical AGENTS.md template.
    heartbeat_interval_s: int = 1800
    # Memory search (OpenClaw-style hybrid retrieval over MEMORY.md +
    # memory/YYYY-MM-DD.md). When enabled, the `memory_search` and
    # `memory_get` tools become available and the heartbeat tick lazy-
    # indexes any drifted files. Disable per-agent by setting False.
    memory_search_enabled: bool = True
    # Pre-compaction memory flush. When the active turn's est. tokens
    # cross `ctx − reserve_floor − soft_threshold`, the runtime pauses,
    # runs a silent agentic turn that lets the model save context to
    # MEMORY.md / memory/YYYY-MM-DD.md, then continues with the
    # compaction. After compaction, `force_post_compaction_index` runs
    # `index_all()` so the saved notes are searchable next turn.
    # Same defaults OpenClaw 2026.4.26 ships.
    memory_flush_enabled: bool = True
    memory_flush_reserve_tokens_floor: int = 20000
    memory_flush_soft_threshold_tokens: int = 4000
    # End-of-session flush trigger. The flush is DEBOUNCED: it runs once the
    # session has been idle (no new turn) for this many seconds — i.e. it has
    # effectively ended — and fires at most ONCE per session. Decoupled from
    # compaction so it can never loop. ``memory_flush_min_turns`` skips trivial
    # sessions that did less than this many turns.
    memory_flush_idle_seconds: int = 900
    memory_flush_min_turns: int = 3
    # Optional model alias to use for the flush turn (cheap model).
    # Empty → reuse the session's primary model.
    memory_flush_model: str = ""
    # Synthesizer model for `remember target=memory` integrations.
    # Empty → reuse `memory_flush_model`, then primary upstream.
    remember_synthesizer_model: str = ""
    # User-defined pre/post tool hooks. Path is relative to the
    # agent home unless absolute. Empty / missing file → no hooks
    # loaded; the engine still constructs and matches return [].
    # See `sharedagents.hooks` for the rule schema. Edit-and-restart
    # semantics: hooks load once at gateway startup, so changes to
    # the file require `sharedagents agent restart <name>` to apply.
    hooks_file: str = "hooks.json"
    # Active memory provider name (see `memory_manager.PROVIDER_REGISTRY`).
    # "" (the default) = no provider → file-only memory (built-in
    # `remember` / MEMORY.md + dailies). The registry ships
    # empty; an external provider (Honcho, Mem0, …) registers by
    # adding to it. "none" also disables provider dispatch explicitly.
    memory_provider: str = ""
    # Recall composition mode for the active provider:
    #   - "context": auto-inject `[RECALLED CONTEXT]` blocks before the
    #     API call (provider.prefetch). No model-callable tools beyond
    #     the built-ins.
    #   - "tools":   provider exposes only the explicit tools (model
    #     decides when to call). No auto-inject.
    #   - "hybrid":  both. Default — auto-inject + tools available.
    # The caller (agent_runtime hot path) reads this to decide whether
    # to invoke `memory_manager.prefetch_for_query`.
    memory_recall_mode: str = "hybrid"
    # Skill auto-invoke: DISABLED. OpenCode-style skill behavior is the
    # inline ``## Skills`` catalog in the system prompt + the ``Skill``
    # tool to load a full body. Embedding auto-injection was removed as
    # prompt slop; leave the hook off unless an operator explicitly opts
    # in via config. See `sharedagents.skill_index` for the unused index path.
    skill_auto_invoke: dict = field(default_factory=dict)
    # Turn-dispatch engine kind. Selects which `sharedagents_agent.agents`
    # Engine implementation owns turn execution for this agent.
    # Values: "local_llm" (default — current upstream HTTP path),
    # "claude_cli" / "codex_cli" / "antigravity_cli" (external CLI agents
    # like `dashboard_dev` with Claude as the engine — see
    # `sharedagents_agent.agents` package docstring).
    # Empty string is normalized to "local_llm" so configs from
    # before this field existed keep their behavior. Unknown values
    # raise at `factory.load_agent` time, not here, so the dataclass
    # stays JSON-friendly.
    engine: str = "local_llm"
    # Per-agent API key map, loaded from ``~/.sharedagents/agents/<name>/secrets.env``
    # at ``AgentConfig.load()`` time and merged with the shared fallback at
    # ``~/.sharedagents/secrets.env``. Lets the operator mint a DIFFERENT key per
    # agent on each provider's dashboard so cost / rate-limit attribution
    # works (OpenRouter spend per agent, OpenAI Project usage per agent,
    # …). ``upstream_headers()`` checks this dict BEFORE ``os.environ`` so
    # systemd ``EnvironmentFile`` keys act as the global default and
    # per-agent ``secrets.env`` wins for the agents that override. Never
    # persisted into the on-disk config; lives only in process memory.
    api_keys: dict[str, str] = field(default_factory=dict)
    # Claude CLI auto-memory (the ``MEMORY.md`` index Claude Code loads from
    # its config home into every fresh session). True keeps today's
    # behavior. False spawns claude_cli turns with
    # ``CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`` so the operator's global notes
    # cannot pull an unrelated project into an sharedagents session (seen after an
    # agent switch). Only that channel is closed: CLAUDE.md, settings and
    # credentials still come from the same home.
    cli_auto_memory: bool = True

    @property
    def home(self) -> Path:
        return agents_root() / self.name

    @property
    def config_path(self) -> Path:
        return self.home / "config.json"

    @property
    def sessions_dir(self) -> Path:
        """Unified sessions folder shared by every agent.

        Per-agent on-disk session directories were retired in the
        2026-05-26 layout refactor — all agents read+write the ONE
        ``~/.sharedagents/sessions/`` folder (override via
        ``$SHAREDAGENTS_SESSIONS_DIR``). Per-session attribution comes from
        ``sessions_meta.agent_name``, not from the path.
        """
        from sharedagents_core.layout import sessions_root

        return sessions_root()

    @property
    def pidfile(self) -> Path:
        return self.home / "gateway.pid"

    @property
    def logs_dir(self) -> Path:
        """Unified logs folder shared by every agent.

        Holds the ONE ``gateway.log`` / ``heartbeat.log`` /
        ``tui.log`` / ``telegram.log`` — per-agent attribution lives
        in each JSONL line's ``agent`` field, not in the path.
        """
        from sharedagents_core.layout import logs_root

        return logs_root()

    @property
    def logfile(self) -> Path:
        from sharedagents_core.layout import gateway_logfile

        return gateway_logfile()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def hooks_path(self) -> Path:
        """Resolved path to the agent's `hooks.json` file.

        Relative paths are joined to `home`; absolute paths are used
        verbatim. The file may not exist — `HookEngine.load()` treats
        a missing file as an empty engine.
        """
        raw = self.hooks_file or "hooks.json"
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (self.home / p)

    def resolve_model(self, alias_or_url: str) -> dict:
        """Resolve an alias or raw URL to a configured model row.

        Returns the full row `{alias, baseUrl, id, backend,
        contextWindow, maxTokens}` so callers can pick whichever fields
        they need (URL for the worker, contextWindow for compaction
        guard, etc.). When `alias_or_url` is a raw URL not in the list,
        a synthetic row is returned with sensible defaults so unknown
        URLs still work.

        Falls back to a synthetic row built from `upstream_endpoint`
        when neither alias nor URL match — preserves the pre-`models`
        single-primary behaviour.
        """
        if not alias_or_url:
            return self._primary_model_row()
        if alias_or_url.startswith("http"):
            url_norm = alias_or_url.rstrip("/")
            for m in self.models or []:
                if (m.get("baseUrl") or "").rstrip("/") == url_norm:
                    return dict(m)
            # Synthetic row — unknown raw URL.
            return {
                "alias": derive_upstream_alias(alias_or_url),
                "baseUrl": url_norm,
                "id": "",
                "backend": "",
                "contextWindow": self.upstream_max_context or 0,
                "maxTokens": 0,
            }
        for m in self.models or []:
            if m.get("alias") == alias_or_url:
                return dict(m)
        return self._primary_model_row()

    def model_row_for_endpoint(self, endpoint: str, model: str = "") -> dict:
        """Resolve the ``models[]`` row for an endpoint, tiebreaking on ``id``.

        When several rows share a ``baseUrl``, a matching wire ``id`` wins over
        the first endpoint-only match so the correct row's ``apiKeyEnv`` /
        static ``headers`` are used.
        """
        url_norm = (endpoint or "").rstrip("/")
        # When several rows share a ``baseUrl`` (e.g. ``local``+``int4`` on the
        # local vLLM, or ``dpsk``+``dpsk-pro`` on DeepSeek), an endpoint-only
        # match returns whichever is listed FIRST — silently picking the wrong
        # row's ``apiKeyEnv`` / static ``headers`` when two same-endpoint rows
        # differ (distinct sub-account keys, an extra provider header). Prefer
        # the row whose wire ``id`` ALSO matches when the caller knows the
        # active model; fall back to the first endpoint match otherwise.
        mdl_norm = (model or "").strip()
        if mdl_norm:
            for m in self.models or []:
                if (m.get("baseUrl") or "").rstrip("/") == url_norm and str(
                    m.get("id") or "",
                ) == mdl_norm:
                    return dict(m)
        for m in self.models or []:
            if (m.get("baseUrl") or "").rstrip("/") == url_norm:
                return dict(m)
        if url_norm == (self.upstream_endpoint or "").rstrip("/"):
            return self._primary_model_row()
        return {"baseUrl": url_norm}

    def alias_for_model_id(self, model_id: str) -> str:
        """Resolve an engine-reported wire model id to its configured alias.

        ``engine_model`` (the real model an engine reported running —
        ``qwen3.6-27b-fp8``, ``claude-opus-4-…``) is display metadata; the
        operator-facing name is the ``models[]`` row's ``alias``. Strict by
        design: only a row that actually maps this id (or an already-alias
        value, making the helper idempotent) yields a label. Unknown ids —
        e.g. an external CLI model the config does not list — return ``""``
        so the caller falls back to the raw id instead of fabricating the
        primary row's alias (a wrong label is worse than none).
        """
        mdl = str(model_id or "").strip()
        if not mdl:
            return ""
        for m in self.models or []:
            if str(m.get("id") or "") == mdl:
                return str(m.get("alias") or "") or mdl
            if str(m.get("alias") or "") == mdl:
                return mdl
        return ""

    def upstream_headers(
        self,
        endpoint: str = "",
        model: str = "",
    ) -> dict[str, str]:
        """Headers for an OpenAI-compatible upstream.

        Model rows may define:
          - apiKeyEnv: name of an environment variable. If set and present,
            sends `Authorization: Bearer <value>`.
          - headers: static extra headers (OpenRouter uses HTTP-Referer and
            X-Title; local vLLM rows leave this empty).

        Secrets are resolved at runtime from env and never persisted into
        session JSON.

        (The bundled key was retired; the no-key path now warns
        toward `sharedagents configure`.)
        """
        row = self.model_row_for_endpoint(
            endpoint or self.upstream_endpoint,
            model,
        )
        headers: dict[str, str] = {}
        raw_headers = row.get("headers") or {}
        if isinstance(raw_headers, dict):
            headers.update({str(k): str(v) for k, v in raw_headers.items() if v is not None})
        api_key_env = str(row.get("apiKeyEnv") or row.get("api_key_env") or "").strip()
        if api_key_env:
            # Resolution order: per-agent api_keys (loaded from
            # ``agents/<name>/secrets.env`` + shared ``secrets.env``
            # fallback) → process env. The per-agent dict wins so an
            # operator running ``main`` and ``coder`` against the same
            # provider can mint distinct OpenRouter sub-account keys for
            # each, get per-agent cost attribution on the provider's
            # dashboard, and keep distinct rate-limit budgets.
            # IMPORTANT: membership check (not ``or``-coalesce) so an
            # operator can DELIBERATELY BLOCK env fallback by writing
            # ``OPENROUTER_API_KEY=`` into ``agents/<x>/secrets.env``
            # (use case: "this agent must NEVER hit OpenRouter even
            # though the shared key would work"). Pre-fix the empty
            # string was falsy and the fallback fired silently — the
            # opt-out workflow was defeated and per-agent attribution
            # quietly merged back into the shared key. See
            # test_b2_empty_api_keys_value_blocks_env_fallback.
            api_key = self.api_keys[api_key_env] if api_key_env in self.api_keys else os.environ.get(api_key_env, "")
            # No bundled-key fallback: the shared demo key was removed.
            # A row that declares
            # an apiKeyEnv now resolves it only from per-agent secrets or the
            # process env; if empty, the request goes out unauthenticated and
            # the caller is steered to `sharedagents configure`.
            if api_key:
                headers.setdefault("Authorization", f"Bearer {api_key}")
            else:
                # The row configured an apiKeyEnv but it's not set in the
                # gateway's process env. Without this warning the request goes
                # out with NO Authorization header and the upstream returns 401
                # with no sharedagents-side hint. Common cause: the var is in the
                # operator's shell but not in ``/etc/environment``, so the
                # systemd-managed daemon doesn't inherit it.
                import logging

                logging.getLogger("sharedagents.agent").warning(
                    "upstream auth: row %r declares apiKeyEnv=%s but it's "
                    "unset/empty in the gateway environ — request will go out "
                    "without Authorization (upstream 401 likely). Fix: run "
                    "`sharedagents configure`, or persist the var / restart with it inline.",
                    row.get("alias") or "?",
                    api_key_env,
                )
        # Allow ``${VAR}`` expansion inside the ``headers`` map so a row
        # whose upstream wants a non-Bearer header (e.g. bedrock-access-
        # gateway's ``x-api-key``) can still source the value from the
        # gateway's env or — critically — from the per-agent api_keys
        # map. The same precedence as the Bearer path: per-agent dict
        # wins, env is the fallback. Without this, header-based
        # providers silently bypass per-agent attribution.
        if any(isinstance(v, str) and "${" in v for v in headers.values()):
            import re as _re

            def _expand(m: _re.Match[str]) -> str:
                name = m.group(1)
                if name in self.api_keys:
                    return self.api_keys[name]
                return os.environ.get(name, "")

            for k, v in list(headers.items()):
                if isinstance(v, str):
                    headers[k] = _re.sub(
                        r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
                        _expand,
                        v,
                    )
        return headers

    def audit_api_keys(self) -> dict[str, str]:
        """Map each ``apiKeyEnv`` referenced by ``models[]`` AND every
        ``${VAR}`` reference inside ``headers`` to its resolution source:

          * ``"agent-local"``       — value from per-agent secrets.env
          * ``"agent-local-empty"`` — operator explicitly blanked it
                                      (blocks env fallback by design)
          * ``"env"``               — value from process env
          * ``"missing"``           — neither place has it; request
                                      will 401 on first call

        Used by the gateway boot to print ONE line per agent so the
        operator can see at a glance which provider keys will work
        and which will 401 on first call. Catches the kimi-style bug
        ("OPENROUTER_API_KEY only in mcp_servers.youtube.env, gateway
        never saw it") on restart instead of on first vision request.

        Model rows without auth credentials are skipped — they don't
        need a key, no point cluttering the report with them.
        """
        import re as _re

        out: dict[str, str] = {}

        def _status(env_name: str) -> str:
            if env_name in self.api_keys:
                val = self.api_keys[env_name]
                # Empty string is a DELIBERATE block — distinguish from
                # "value present" so the boot log tells the truth.
                return "agent-local" if val else "agent-local-empty"
            if os.environ.get(env_name):
                return "env"
            return "missing"

        var_re = _re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
        for row in self.models or []:
            if not isinstance(row, dict):
                continue
            env_name = str(
                row.get("apiKeyEnv") or row.get("api_key_env") or "",
            ).strip()
            if env_name and env_name not in out:
                out[env_name] = _status(env_name)
            # Also pick up ${VAR} refs in static headers so header-
            # based providers (bedrock, azure) appear in the audit.
            for hv in (row.get("headers") or {}).values():
                if not isinstance(hv, str):
                    continue
                for m in var_re.finditer(hv):
                    name = m.group(1)
                    if name not in out:
                        out[name] = _status(name)
        return out

    def _sync_codex_oauth_models(self, *, force: bool = False) -> None:
        """Fold Codex OAuth picker rows into runtime ``self.models`` once.

        Called from :meth:`load` and :meth:`reload_models` (``force=True``
        on reload so a user-requested refresh rediscovers). Does not
        persist to ``config.json``.
        """
        if str(getattr(self, "engine", "") or "local_llm").strip() not in (
            "",
            "local_llm",
        ):
            return
        try:
            from sharedagents_agent.discovery.openai_codex import (
                sync_codex_oauth_models_into_cfg,
            )

            sync_codex_oauth_models_into_cfg(self, force=force)
        except Exception:  # noqa: BLE001
            pass

    def _sync_bedrock_models(self, *, force: bool = False) -> None:
        try:
            from sharedagents_agent.discovery.bedrock import sync_bedrock_models_into_cfg

            sync_bedrock_models_into_cfg(self, force=force)
        except Exception as exc:  # Optional discovery must not block agent loading.
            import logging

            logging.getLogger("sharedagents.agent").warning(
                "Bedrock discovery unavailable (%s)",
                type(exc).__name__,
            )

    def reload_models(self) -> list[dict]:
        """Re-read this agent's model rows from disk (the same global +
        per-agent merge as :meth:`load`) and refresh ``self.models`` in place.

        Lets a long-lived gateway pick up models added AFTER it booted — a
        background local-GPU install completing, a wizard re-run, or a
        hand-edited ``config.json`` — without a full restart. Best-effort: any
        read/parse error leaves the current ``self.models`` untouched.
        """
        try:
            glob = load_global_config()
            per_agent: dict[str, Any] = {}
            path = agents_root() / self.name / "config.json"
            if path.exists():
                per_agent = json.loads(path.read_text(encoding="utf-8"))
            self._effort_is_per_agent = "effort" in per_agent
            self._operator_model_values = _operator_model_values(per_agent)
            builtin_defaults = _BUILTIN_AGENT_DEFAULTS.get(self.name, {})
            merged = {**glob, **builtin_defaults, **per_agent}
            _drop_inherited_local_route(
                merged,
                per_agent=per_agent,
                builtin_defaults=builtin_defaults,
            )
        except Exception:  # noqa: BLE001 - never let a reload tank a switch
            # Read/parse failure is best-effort: leave ``self.models`` intact.
            return self.models or []
        # A duplicate-alias collision is a HARD error at every load/reload
        # chokepoint, not a best-effort skip — so
        # normalization runs OUTSIDE the guard above and its ``ValueError``
        # propagates exactly as ``load()`` lets it.
        self.models = _normalize_models(merged.get("models") or merged.get("upstream_endpoints") or [])
        engine = str(merged.get("engine") or "").strip() or "local_llm"
        if engine in _LOCAL_LLM_ENGINE_NAMES:
            from sharedagents_model.provider_presets import ensure_xai_grok_models

            self.models = ensure_xai_grok_models(
                self.models,
                api_keys=getattr(self, "api_keys", None),
            )
        if isinstance(merged.get("teacher"), dict):
            self.teacher = merged["teacher"]
        if isinstance(merged.get("goal"), dict):
            self.goal = merged["goal"]
        if isinstance(merged.get("self_improver"), (dict, bool)):
            self.self_improver = merged["self_improver"]
        # Disk models[] first, then Codex OAuth rows when that provider is
        # configured — runtime cfg.models is the single catalogue. Reload
        # replaces from disk, so this must re-run after every refresh.
        self._sync_codex_oauth_models(force=True)
        self._sync_bedrock_models()
        return self.models or []

    # Structural / lifecycle fields that CANNOT be safely mutated on a
    # live agent — changing them needs a new Engine instance, a re-mount,
    # or a scheduler restart. ``reload_from_disk`` reports (but never
    # applies) these so the operator knows a restart is still required.
    #   engine          → _engine_for() runs at load; needs a fresh Engine
    #   agent_profile   → apply_profile_to_config re-pins permission_mode (the
    #                     tool/skill allowlist is operator-authoritative, not re-pinned)
    #   name/host/port  → identity + server bind, fixed at boot
    #   heartbeat_*     → the scheduler is start/stopped explicitly at boot
    #   hooks_file      → HookEngine.load() caches hooks at first use
    #   memory_provider → selects a memory backend activated once at load
    # NB: self_improver is also restart-only, but it's a _MODEL_REFRESH_FIELD
    # (reload_models refreshes it), so it's enforced in reload_from_disk's
    # post-reload_models block (snapshot + restore) rather than listed here.
    _RESTART_ONLY_FIELDS = frozenset(
        {
            "name",
            "host",
            "port",
            "engine",
            "agent_profile",
            "heartbeat_enabled",
            "heartbeat_interval_s",
            "hooks_file",
            "memory_provider",
        }
    )
    # Handled by reload_models() (models[] + teacher/goal/self_improver),
    # so the generic scalar loop skips them to keep ONE model-refresh path.
    # (self_improver is snapshot-restored to restart-only in reload_from_disk.)
    _MODEL_REFRESH_FIELDS = frozenset(
        {
            "models",
            "teacher",
            "goal",
            "self_improver",
        }
    )

    def reload_from_disk(self) -> ConfigReloadReport:
        """Re-read this agent's config (global + built-in + per-agent merge,
        same order as :meth:`load`) and apply the *hot-reloadable* fields onto
        this live instance in place.

        This is the SIGHUP hot-reload path: the gateway holds one long-lived
        ``cfg`` object per agent (the mounted ACP sub-app + LocalLLM engine
        reference it). The engine re-reads cfg across the turn/round loop
        (sampling, tool allowlists, budget, compaction), so a live mutation is
        picked up on the next read — normally the next turn. NOTE: it is NOT a
        hard turn-boundary guarantee — a SIGHUP delivered mid-turn can land
        between rounds of a multi-round turn, exactly the same live-mutation
        window the ``/model`` picker / :meth:`reload_models` already carry.
        Send ``kill -HUP`` while the agent is idle for a clean boundary.

        Structural fields (``_RESTART_ONLY_FIELDS``) are reported but NOT
        mutated — swapping them on a live agent would need a new Engine /
        re-mount. MCP server lists are not dataclass fields (never touched
        here); changing them also requires a restart to re-warm discovery.
        Model rows + teacher/goal/self_improver route through
        :meth:`reload_models` (the single model-refresh path).

        For profile agents (``explore`` / ``plan`` / ``coder`` / …) the live
        profile is RE-APPLIED after the scalar merge — exactly as :meth:`load`
        does — so a SIGHUP can never strip the profile-pinned
        ``permission_mode`` back to a raw un-profiled value. The tool/skill
        allowlist is OPERATOR-authoritative: a
        reload RESPECTS the operator's on-disk allowlist rather than re-pinning
        the profile default (only ``permission_mode`` is re-pinned).

        Semantics match :meth:`load`: only keys PRESENT in the merged config
        are applied, so removing a key from ``config.json`` leaves the live
        value untouched rather than snapping it back to the dataclass default
        (documented no-revert-on-removal). Best-effort: any read/parse error
        leaves this instance untouched and returns an empty report.

        Returns a :class:`ConfigReloadReport` (applied / restart_only field
        names + an ``unchanged`` flag).
        """
        applied: list[str] = []
        restart_only: list[str] = []
        try:
            glob = load_global_config()
            per_agent: dict[str, Any] = {}
            path = agents_root() / self.name / "config.json"
            if path.exists():
                per_agent = json.loads(path.read_text(encoding="utf-8"))
            builtin_defaults = _BUILTIN_AGENT_DEFAULTS.get(self.name, {})
            merged = {**glob, **builtin_defaults, **per_agent}
            _drop_inherited_local_route(
                merged,
                per_agent=per_agent,
                builtin_defaults=builtin_defaults,
            )
            merged.pop("upstream_endpoints", None)
            _coerce_config_knobs(merged)
            # Refresh the operator-explicit ownership flags so the claude
            # default chain keeps seeing the CURRENT per-agent file state
            # after a hot reload (mirrors ``_effort_is_per_agent``).
            self._upstream_model_is_per_agent = "upstream_model" in per_agent
        except Exception as exc:  # noqa: BLE001 - never let a reload tank the gateway
            # A SIGHUP hot-reload crash is an agent-runtime crash boundary:
            # capture it structurally (SessionError machinery) instead of
            # silently swallowing it, so the failure is visible on the same
            # path the TUI already uses. Best-effort — the
            # reload must still degrade to "untouched, unchanged report".
            try:
                from sharedagents_core import errors as _errors

                _errors.log_error(
                    f"agent:{self.name}:reload",
                    exc,
                    summary=f"config hot-reload failed for agent {self.name!r}",
                    agent_name=self.name,
                )
            except Exception:  # noqa: BLE001
                pass
            return ConfigReloadReport(applied=[], restart_only=[], unchanged=True)

        for f in self.__dataclass_fields__.values():
            fname = f.name
            if fname == "api_keys":
                # Secrets are process-memory only, never persisted here;
                # refreshed separately (see AgentACPBackend.reload_config).
                continue
            if fname in self._MODEL_REFRESH_FIELDS:
                continue
            if fname not in merged:
                continue
            new_val = merged[fname]
            if new_val == getattr(self, fname, None):
                continue
            if fname in self._RESTART_ONLY_FIELDS:
                restart_only.append(fname)
                continue
            setattr(self, fname, new_val)
            applied.append(fname)

        # Model rows + teacher/goal via the blessed reload_models() path.
        # reload_models() ALSO refreshes self_improver, but the SI supervisor
        # is only started / reconciled at gateway boot — mutating the live
        # value on SIGHUP would half-apply (value changes, supervisor doesn't),
        # so self_improver is RESTART-ONLY: snapshot it, let reload_models run,
        # then restore the old value and report the change as restart-only
        # rather than applying it. teacher/goal ARE read live (per-turn / per
        # goal), so they apply hot — but reload_models tracks only models, so we
        # snapshot + compare each here for an accurate ConfigReloadReport.
        try:
            models_before = list(self.models or [])
            teacher_before = self.teacher
            goal_before = self.goal
            si_before = self.self_improver
            self.reload_models()
            if list(self.models or []) != models_before:
                applied.append("models")
            if self.teacher != teacher_before:
                applied.append("teacher")
            if self.goal != goal_before:
                applied.append("goal")
            if self.self_improver != si_before:
                self.self_improver = si_before  # restore — never hot-apply
                restart_only.append("self_improver")
        except Exception:  # noqa: BLE001
            pass

        # Re-pin the profile-owned ``permission_mode`` exactly as load() does at the
        # end of its merge — the raw scalar apply above may have overwritten it with
        # the un-profiled config value. The tool/skill allowlist is OPERATOR-authoritative
        # apply_profile_to_config below is given the
        # operator's config keys, so a hot reload RESPECTS the operator's allowlist and
        # only ``permission_mode`` is re-pinned. agent_profile itself is restart-only, so
        # we re-apply the LIVE profile (resolved from self), never switch it.
        #
        # For a PROFILE-TAGGED agent the profile wins over live hot-reload of
        # ``permission_mode``: the
        # profile authoritatively OWNS ``permission_mode`` — the governance
        # floor holds — so ``apply_profile_to_config``'s re-pin to
        # ``profile.mode`` is authoritative and an on-disk ``permission_mode``
        # edit to a profile agent does NOT override it. Live hot-reload of
        # ``permission_mode`` is thereby carved out for profile agents. We do NOT
        # silently revert, though: when the merged config
        # carries an EXPLICIT ``permission_mode`` that DIVERGES from the
        # profile's pinned value, emit a WARNING so the operator learns their
        # edit was overridden by the profile floor (edit the profile, not the
        # agent config). (``_BUILTIN_AGENT_DEFAULTS`` never sets
        # permission_mode, so presence in ``merged`` means a global- or
        # per-agent-config edit.)
        try:
            from sharedagents_agent.agent_profiles import (
                apply_profile_to_config,
                resolve_profile_for_config,
            )

            prof = resolve_profile_for_config(self)
            if prof is not None:
                explicit_mode = merged.get("permission_mode")
                # Operator owns the allowlist: a hot
                # config reload must not overwrite it with the profile default
                # either. ``per_agent`` (re-read above) marks the operator's keys.
                apply_profile_to_config(
                    self,
                    prof,
                    operator_keys=set(per_agent.keys()),
                )
                # Profile floor wins. WARN — never silently revert —
                # when an on-disk edit diverges from the profile-owned value.
                if explicit_mode is not None and explicit_mode != self.permission_mode:
                    import logging

                    logging.getLogger("sharedagents.agent").warning(
                        "agent %r: on-disk permission_mode=%r diverges from "
                        "profile %r's pinned %r; profile floor wins. "
                        "Edit the profile, not the agent config, to change it.",
                        self.name,
                        explicit_mode,
                        prof.name,
                        self.permission_mode,
                    )
        except Exception:  # noqa: BLE001
            pass

        return ConfigReloadReport(
            applied=sorted(applied),
            restart_only=sorted(restart_only),
            unchanged=(not applied and not restart_only),
        )

    # ----- teacher-model supervision -------------------------------------
    def teacher_model(self) -> dict | None:
        """The `models[]` row tagged ``role == "teacher"``, or ``None``."""
        for m in self.models or []:
            if isinstance(m, dict) and m.get("role") == "teacher":
                return dict(m)
        return None

    def fallback_model(self) -> dict | None:
        """The `models[]` row tagged ``role == "fallback"``, or ``None``.

        The LLM upstream's BACKUP: when the primary endpoint is unreachable
        (connection refused / DNS / all-attempts-failed), the engine fails the
        round over to this row's endpoint+model. Per-agent overridable via the
        agent's own `models[]` (which merges over global). Mirrors the
        primary/backup pattern used for STT/embedding/reranking services.
        """
        for m in self.models or []:
            if isinstance(m, dict) and m.get("role") == "fallback":
                return dict(m)
        return None

    def teacher_policy(self) -> dict:
        """The teacher supervision policy block, merged over defaults.

        Returns a dict with ``enabled`` reflecting BOTH the policy flag and
        the presence of a configured teacher endpoint — supervision can
        only run if there is a teacher model to call. ``enabled`` is the
        MASTER switch, not per-session activation: the engine still requires
        an explicit ``/teacher on`` unless the policy sets
        ``require_goal: false``.
        """
        pol = dict(self.teacher or {})
        defaults = {
            "enabled": False,
            "triggers": [
                "Stop",
                "PrematureTurnEnd",
                "ReasoningLoopDetected",
                "StopFailure",
                "SessionError",
                "PostToolUseFailure",
            ],
            "idle_seconds": 0,
            "periodic_minutes": 5,
            "max_interventions": 6,
            "min_reviews_per_session": 1,
            "stall_limit": 2,
            # Proactive mid-turn firing thresholds (operator directive
            # 2026-06-01). The teacher fires to HELP the student the moment
            # either trips — it no longer waits for a turn boundary:
            #   * tool_failure_threshold — N CONSECUTIVE errored tool calls in
            #     a turn fire the teacher (also forces a Tier-2 review so it
            #     sees the error bodies).
            #   * no_result_round_limit — N consecutive rounds in a turn-
            #     segment that deliver no result fire the teacher.
            "tool_failure_threshold": 5,
            "no_result_round_limit": 20,
            "min_confidence": 0.6,
            "timeout_seconds": 45,
            "deep_tier": "auto",
            "escalation": "auto",
            "escalation_targets": ["teacher"],
            "restore_model_after_escalation": True,
            "fallback_endpoint": None,
            "on_unreachable": "warn",
        }
        merged = {**defaults, **pol}
        merged["enabled"] = bool(pol.get("enabled")) and self.teacher_model() is not None
        return merged

    def small_model_row(self) -> dict:
        """OpenCode-style small/utility model for title + summary side calls.

        Resolution order (first hit wins):
          1. Explicit ``cfg.small_model`` alias or wire ``id`` in ``models[]``
          2. ``models[]`` row with ``role == "small"``
          3. Common fast aliases: ``dpsk`` / ``haiku`` / ``fast`` / ``small``
          4. Primary model (same fallback OpenCode uses when no small model)

        Always returns a row so side callers never get a hard miss — an
        invalid ``small_model`` falls through rather than raising.

        Traceability: OpenCode Provider.getSmallModel + Agent title/summary
        """
        needle = str(getattr(self, "small_model", "") or "").strip()
        if needle:
            if needle.startswith("http"):
                return self.resolve_model(needle)
            for m in self.models or []:
                if not isinstance(m, dict):
                    continue
                alias = str(m.get("alias") or "").strip()
                mid = str(m.get("id") or "").strip()
                if needle in (alias, mid):
                    return dict(m)
            # Unresolved configured id — keep trying heuristics, then primary.
        for m in self.models or []:
            if isinstance(m, dict) and str(m.get("role") or "").lower() == "small":
                return dict(m)
        for alias in ("dpsk", "haiku", "fast", "small"):
            for m in self.models or []:
                if not isinstance(m, dict):
                    continue
                if str(m.get("alias") or "").lower() == alias:
                    return dict(m)
                if str(m.get("id") or "").lower() == alias:
                    return dict(m)
        return self._primary_model_row()

    # ----- self-improver (orchestrator-scoped, config-driven) ------------
    def self_improver_policy(self) -> dict:
        """The self-improver policy block, merged over defaults — the CONFIG by
        which the orchestrator (south) decides whether/how to run the self-
        improver in-process. It is NOT a separate service and the north gateway
        has no control surface over it; flip ``enabled`` in config to turn it on.

        Accepts a SIMPLE on/off form — ``"self_improver": true`` (or ``false``)
        — which is the primary way to use it; everything else has safe defaults.
        An optional dict form lets power users set ``mode``/``push``.

        ``mode``: ``triage`` (open/dedup forge issues only — default) or
        ``fix`` (also run the claude‖codex agreement-loop fix pipeline).
        """
        raw = self.self_improver
        pol = {"enabled": raw} if isinstance(raw, bool) else dict(raw or {})
        defaults = {
            "enabled": False,  # opt-in via config; safe default
            "mode": "triage",  # "triage" | "fix"
            "model_alias": "dpsk",  # deepseek-v4-flash
            "triggers": ["session_saved"],  # in-process bus event to react to
            "debounce_seconds": 5,
            "max_queue": 64,  # bounded backpressure
            "push": False,  # never push without explicit opt-in
            "max_rounds": 2,
        }
        merged = {**defaults, **pol}
        merged["enabled"] = bool(pol.get("enabled"))
        merged["mode"] = "fix" if str(pol.get("mode", "")).lower() == "fix" else "triage"
        return merged

    def _primary_model_row(self) -> dict:
        """The fallback row when no alias is given. Uses the global
        `upstream_endpoint` / `upstream_model` / `upstream_alias` fields
        if `models[]` is empty (single-primary configs). Otherwise
        returns the first row of `models[]`.
        """
        if self.models:
            return dict(self.models[0])
        return {
            "alias": self.upstream_alias or derive_upstream_alias(self.upstream_endpoint),
            "baseUrl": self.upstream_endpoint,
            "id": self.upstream_model,
            "backend": self.backend,
            "contextWindow": self.upstream_max_context or 0,
            "maxTokens": 0,
        }

    # ----- Back-compat shims used by `app.py` / `agent_cli.py` while -----
    # ----- the rename rolls out. New code should call `resolve_model()`. -----

    def resolve_endpoint(self, alias_or_url: str) -> str:
        return (self.resolve_model(alias_or_url).get("baseUrl") or "").rstrip("/")

    def resolve_alias(self, alias_or_url: str) -> str:
        return self.resolve_model(alias_or_url).get("alias") or ""

    def save(self) -> Path:
        """Write only the per-agent fields. Global fields are left alone.

        To persist `upstream_endpoint` etc., call `save_global_config()`
        separately — typically the wizard does both.

        IMPORTANT: ``api_keys`` is in-memory only. It's loaded from
        ``secrets.env`` at ``AgentConfig.load()`` time and must NEVER
        round-trip into the on-disk config (which is 644 by default
        and not gitignored by name). The save would otherwise leak
        every key the very first time ``heartbeat`` or the wizard
        called ``save()``.
        """
        self.home.mkdir(parents=True, exist_ok=True)
        try:
            from sharedagents_agent.agent_profiles import resolve_profile

            prof_key = str(getattr(self, "agent_profile", "") or "").strip()
            prof = resolve_profile(prof_key) if prof_key else None
        except Exception:  # noqa: BLE001
            prof = None
        if prof is not None:  # profile agents persist LEAN (the main carve-out was retired)
            from sharedagents_agent.agents.default_scaffolds import _profile_config_dict

            per_agent = dict(_profile_config_dict(prof))
            # Heartbeat is an OPERATOR-mutable operational field, not profile
            # identity — carry the LIVE value so a lean save preserves the wizard's
            # advanced choice and `sharedagents agent heartbeat enable/disable <agent>`
            # instead of resetting to the profile scaffold baseline. (The dataclass
            # default is False, so this is safe for every profile agent; memory
            # stays profile-derived above because its dataclass default is True.)
            per_agent["heartbeat_enabled"] = bool(self.heartbeat_enabled)
            per_agent["heartbeat_interval_s"] = int(self.heartbeat_interval_s)
            # A binding agent (the orchestrator jarvis, whose profile scaffolds a
            # real port) carries its operator-set port/host so a lean save preserves
            # them — restoring the parity the retired `main` had via its full-surface
            # save (Codex retire-main review). Non-binding phase agents (profile port
            # 0, gateway-mounted under /acp/<name>) stay at 0 — deliberately not a
            # misleading listener port.
            if per_agent.get("port"):
                per_agent["port"] = int(self.port)
                per_agent["host"] = str(self.host)
            # Operator-authoritative allowlist: when a
            # loaded config owns the tool/skill allowlist on disk, a lean save must
            # PRESERVE the operator's value (including a null == unrestricted choice)
            # rather than re-deriving from the profile. ``_operator_allowlist_keys`` is
            # set by apply_profile_to_config from the operator's raw config keys; a
            # freshly-CONSTRUCTED config (wizard first-run, tests) has no such hint, so
            # it still seeds from the profile (the scaffold default).
            _op_allow = getattr(self, "_operator_allowlist_keys", None) or set()
            if "tools_allowlist" in _op_allow:
                per_agent["tools_allowlist"] = list(self.tools_allowlist) if self.tools_allowlist is not None else None
                per_agent["tools_allowlist_strict"] = bool(self.tools_allowlist_strict)
            if "skill_allowlist" in _op_allow:
                per_agent["skill_allowlist"] = list(self.skill_allowlist) if self.skill_allowlist is not None else None
                per_agent["skill_allowlist_strict"] = bool(self.skill_allowlist_strict)
            if self.description and self.description.strip() and (self.description != prof.description):
                per_agent["description"] = self.description
            if self.system and str(self.system).strip():
                per_agent["system"] = str(self.system).strip()
            # Operator-toggled hardcoded prompt sections (Agent Builder
            # enable/disable cards) — carry the LIVE list so a lean
            # save preserves the operator's choices instead of re-enabling
            # every section from the profile scaffold baseline.
            if self.disabled_prompt_sections:
                per_agent["disabled_prompt_sections"] = list(self.disabled_prompt_sections)
            if self.mcp_always_loaded is not None:
                per_agent["mcp_always_loaded"] = list(self.mcp_always_loaded)
            # Preserve an operator-owned per-agent delegation mode (Agent
            # Builder select / direct config edit) — a lean profile save must
            # not reset it to the profile scaffold default. A null/cleared value pops the key so the profile
            # default applies again.
            if getattr(self, "_delegation_mode_is_per_agent", False):
                if self.delegation_mode:
                    per_agent["delegation_mode"] = self.delegation_mode
                else:
                    per_agent.pop("delegation_mode", None)
            # Operator-owned model route + parameters (Agent Builder model,
            # WebUI "Set default"): only the keys this agent's own file held,
            # so an inherited global is never frozen into a profile file. Live
            # values win (an in-process change persists), except ``models``,
            # whose live rows may be enriched at runtime: the file's rows stay.
            for _k, _raw in (getattr(self, "_operator_model_values", None) or {}).items():
                if _k == "effort":
                    continue  # keeps its own ownership rule just below
                _live = _raw if _k == "models" else getattr(self, _k, _raw)
                per_agent[_k] = copy.deepcopy(_live)
            # Preserve an operator-owned per-agent override, but do not copy an
            # inherited global effort into every profile file on an unrelated
            # save (that would freeze those agents against later global edits).
            _effort_owner = getattr(self, "_effort_is_per_agent", None)
            if self.effort and (_effort_owner is True or _effort_owner is None):
                per_agent["effort"] = self.effort
        else:
            per_agent = {
                f.name: getattr(self, f.name)
                for f in self.__dataclass_fields__.values()
                if f.name not in _GLOBAL_FIELDS and f.name not in _NON_PERSISTED_FIELDS
            }
            # Never persist a null delegation_mode for plain agents — the
            # resolver treats absence as unpoliced; a literal null is noise.
            if per_agent.get("delegation_mode") is None:
                per_agent.pop("delegation_mode", None)
            if per_agent.get("mcp_always_loaded") is None:
                per_agent.pop("mcp_always_loaded", None)
            _effort_owner = getattr(self, "_effort_is_per_agent", None)
            if self.effort and (_effort_owner is True or _effort_owner is None):
                per_agent["effort"] = self.effort
        write_config_text(self.config_path, json.dumps(per_agent, indent=2, ensure_ascii=False) + "\n")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        try:
            from sharedagents_gateway.heartbeat import sync_heartbeat_files

            sync_heartbeat_files(self.home, enabled=bool(self.heartbeat_enabled))
        except Exception:  # noqa: BLE001
            pass
        return self.config_path

    @classmethod
    def load(cls, name: str, *, ambient_discovery: bool = True) -> AgentConfig:
        """Load agent ``name`` from disk (global + built-in + per-agent merge).

        Merges in a fixed three-layer order — global config, built-in agent
        defaults, then the per-agent file (which always wins). Raises
        ``FileNotFoundError`` for an unscaffolded agent rather than
        materializing one; layers ``secrets.env`` into ``api_keys`` and syncs
        ``HEARTBEAT.md`` to ``heartbeat_enabled``.

        ``ambient_discovery=False`` skips the Codex OAuth and Bedrock catalogue
        folds. Both can make network calls when their short-lived caches
        expire; the picker summary must not, so it reads the configured rows
        only and asks a provider for its models when that provider is opened.
        """
        from sharedagents_core.ids import bare_id

        name = bare_id(name)
        path = agents_root() / name / "config.json"
        if not path.exists():
            raise FileNotFoundError(f"agent {name!r} not configured: {path} missing")
        per_agent = json.loads(path.read_text(encoding="utf-8"))
        glob = load_global_config()
        # Merge order:
        #   1. global (``~/.sharedagents/config.json``) — the deployment-wide
        #      defaults (single upstream, MCP servers, skills dirs, …).
        #   2. built-in agent defaults (``_BUILTIN_AGENT_DEFAULTS``) —
        #      per-name pins for shipped sibling agents like ``coder``,
        #      so a fresh install gets the right upstream without
        #      requiring a hand-edited per-agent ``config.json``.
        #   3. per-agent file (``~/.sharedagents/agents/<name>/config.json``)
        #      — the operator's explicit overrides. Always wins.
        builtin_defaults = _BUILTIN_AGENT_DEFAULTS.get(name, {})
        merged: dict[str, Any] = {**glob, **builtin_defaults, **per_agent}
        _drop_inherited_local_route(
            merged,
            per_agent=per_agent,
            builtin_defaults=builtin_defaults,
        )
        # ``delegation_mode`` is strictly PER-AGENT: never inherit a global
        # value (the runtime resolver reads only the per-agent file; a global
        # value would silently diverge and apply to every agent). Only the
        # per-agent file may carry it; otherwise the profile default seeds it.
        if "delegation_mode" in merged and "delegation_mode" not in per_agent:
            merged.pop("delegation_mode", None)
        # Back-compat: accept the legacy `upstream_endpoints` key when
        # `models` isn't set, and translate per-row legacy keys
        # (`endpoint`/`model`/`max_context`) to the OpenClaw-aligned
        # ones (`baseUrl`/`id`/`contextWindow`).
        merged["models"] = _normalize_models(merged.get("models") or merged.get("upstream_endpoints") or [])
        engine = str(merged.get("engine") or "").strip() or "local_llm"
        # Keep legacy alias out of the dataclass kwargs so we don't get
        # an "unexpected keyword" error; it lives only on disk.
        merged.pop("upstream_endpoints", None)
        _coerce_config_knobs(merged)
        kwargs: dict[str, Any] = {}
        for f in cls.__dataclass_fields__.values():
            if f.name in merged:
                kwargs[f.name] = merged[f.name]
        if "name" not in kwargs:
            kwargs["name"] = name
        from sharedagents_model.model_parameters import resolve_model_parameters

        resolve_model_parameters({})  # Validate the snapshot at config initialization.
        cfg = cls(**kwargs)
        # ``effort`` may come from the shared config or the agent file. Track
        # ownership so save() can round-trip a real per-agent override without
        # accidentally materialising an inherited global default locally.
        cfg._effort_is_per_agent = "effort" in per_agent
        cfg._temperature_is_explicit = "temperature" in per_agent or "temperature" in glob
        cfg._delegation_mode_is_per_agent = "delegation_mode" in per_agent
        cfg._operator_model_values = _operator_model_values(per_agent)
        # Same ownership tracking for ``upstream_model``: the claude CLI
        # default chain may only be seeded by an OPERATOR-EXPLICIT per-agent
        # value — the merged global ``upstream_model`` describes the local-LLM
        # deployment and must never ride to the Claude CLI as ``--model``.
        cfg._upstream_model_is_per_agent = "upstream_model" in per_agent
        # Layer the per-agent + shared secrets.env on top so
        # ``upstream_headers()`` can resolve keys without reading
        # ``os.environ`` for every call site. ``api_keys`` is never
        # persisted into the on-disk config; it lives only in process
        # memory.
        try:
            cfg.api_keys = _load_agent_secrets(sharedagents_root(), name)
        except Exception as exc:  # noqa: BLE001
            # Bad/unreadable secrets.env shouldn't tank agent loading —
            # log so the operator notices, but proceed with an empty
            # map (process env still works as a fallback).
            import logging

            logging.getLogger("sharedagents.agent").warning(
                "agent %r: failed to load secrets.env (%s); falling back to os.environ only",
                name,
                exc,
            )
            cfg.api_keys = {}
        # Direct-xAI Grok rows are appended ONLY when this system actually
        # resolves an xAI key — a deployment that runs on one other provider
        # alone (AWS Bedrock, local vLLM, …) must not have xAI forced into its
        # catalogue. Ordered AFTER the secrets.env layer so an agent-local
        # ``XAI_API_KEY`` counts, and an explicitly blanked one still blocks
        # the env fallback exactly as ``upstream_headers()`` does.
        # (the injection used to be unconditional.)
        if engine in _LOCAL_LLM_ENGINE_NAMES:
            from sharedagents_model.provider_presets import ensure_xai_grok_models

            cfg.models = ensure_xai_grok_models(cfg.models, api_keys=cfg.api_keys)
        # OpenCode-style profile agents (``explore``, ``plan``, …): when
        # the directory name or ``agent_profile`` field matches a profile,
        # swap in the agent-home ``AGENTS.md`` as the profile system
        # prompt (replaces stacked SOUL/AGENTS/MODE files at compose
        # time) and enforce the profile's permission_mode. The tool/skill
        # allowlist is operator-authoritative:
        # the profile only SEEDS it at install; load respects the operator's
        # on-disk allowlist (via operator_keys below).
        try:
            from sharedagents_agent.agent_profiles import apply_profile_to_config, resolve_profile

            prof_key = str(merged.get("agent_profile") or "").strip()
            prof = resolve_profile(prof_key) if prof_key else None
            if prof is not None:
                # The operator's per-agent config.json is authoritative for the
                # tool/skill allowlist — the profile only seeds what the operator
                # did not set. ``per_agent`` is the raw
                # operator file, so its keys mark what the operator owns.
                apply_profile_to_config(
                    cfg,
                    prof,
                    operator_keys=set(per_agent.keys()),
                )
                if not str(getattr(cfg, "agent_profile", "") or "").strip():
                    cfg.agent_profile = prof.name
                # The retired jarvis profile scaffold stored port 0, which sends
                # its heartbeat scheduler to 127.0.0.1:0 after upgrade.  Adopt
                # the current non-zero profile baseline only for that exact
                # managed zero-port state.  Operator-selected non-zero ports and
                # gateway-mounted phase agents (whose baseline is 0) are left
                # untouched; a later save persists the healed jarvis value.
                from sharedagents_agent.agents.default_scaffolds import (
                    _config_is_managed_profile,
                    _profile_config_dict,
                )

                if (
                    prof.name == "jarvis"
                    and _config_is_managed_profile(per_agent, prof.name)
                    and per_agent.get("port") == 0
                ):
                    profile_port = int(_profile_config_dict(prof).get("port") or 0)
                    if profile_port:
                        cfg.port = profile_port
        except Exception:  # noqa: BLE001
            pass
        try:
            from sharedagents_gateway.heartbeat import sync_heartbeat_files

            sync_heartbeat_files(cfg.home, enabled=bool(cfg.heartbeat_enabled))
        except Exception:  # noqa: BLE001
            pass
        if ambient_discovery:
            cfg._sync_codex_oauth_models()
            cfg._sync_bedrock_models()
        return cfg


_KNOB_BOOL_FIELDS = ("route_guard_enabled", "llm_compaction", "auto_complete_continue", "cli_auto_memory")
_KNOB_INT_FIELDS = (
    "max_transient_retries",
    "upstream_max_completion_tokens",
    "distill_max_recoveries",
    "max_stop_continuations",
)
_TRUE_STRS = {"true", "1", "yes", "on"}
_FALSE_STRS = {"false", "0", "no", "off"}


def _coerce_config_knobs(merged: dict[str, Any]) -> None:
    """Validate/coerce the runtime routing knobs IN PLACE before they reach the
    dataclass, so a stringy JSON value can't silently misbehave (``"false"``
    staying truthy) or crash mid-turn (``"bad"`` for an int). Invalid values are
    DROPPED from ``merged`` — on load that means the field falls back to its
    default; on hot-reload it means the live (last-known-good) value is kept.
    """
    for f in _KNOB_BOOL_FIELDS:
        if f not in merged:
            continue
        v = merged[f]
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int,)):
            merged[f] = bool(v)
        elif isinstance(v, str) and v.strip().lower() in _TRUE_STRS:
            merged[f] = True
        elif isinstance(v, str) and v.strip().lower() in _FALSE_STRS:
            merged[f] = False
        else:
            del merged[f]
    for f in _KNOB_INT_FIELDS:
        if f not in merged:
            continue
        v = merged[f]
        if v is None:
            continue
        if isinstance(v, bool):  # bool is an int subclass — reject to avoid True→1
            del merged[f]
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            del merged[f]
            continue
        if iv < 0:
            del merged[f]
        else:
            merged[f] = iv


def _normalize_models(rows: list[dict]) -> list[dict]:
    """Translate per-row legacy keys to OpenClaw-aligned ones.

    Accepted aliases (legacy → canonical):
        endpoint        → baseUrl
        model           → id
        max_context     → contextWindow
        max_tokens      → maxTokens
    Canonical keys are written as-is when present; legacy keys are
    only consulted when the canonical form is missing. Unknown keys
    pass through untouched so future fields stay intact across loads.
    """
    out: list[dict] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        norm = dict(row)
        if "baseUrl" not in norm and "endpoint" in row:
            norm["baseUrl"] = row["endpoint"]
        if "id" not in norm and "model" in row:
            norm["id"] = row["model"]
        if "contextWindow" not in norm and "max_context" in row:
            norm["contextWindow"] = row["max_context"]
        if "maxTokens" not in norm and "max_tokens" in row:
            norm["maxTokens"] = row["max_tokens"]
        # Strip legacy keys from the in-memory shape so the rest of the
        # codebase only sees the canonical names; the on-disk file is
        # rewritten with canonical keys on the next save.
        for legacy in ("endpoint", "model", "max_context", "max_tokens"):
            norm.pop(legacy, None)
        out.append(norm)

    # Aliases are the model-switch key: the TUI ``/model`` picker and the ACP
    # bridge (``_resolve_model_row`` / engine ``_find``) resolve a switch by
    # alias and return the FIRST matching row. Two rows sharing an alias make
    # the later one unselectable — the switch silently binds the first while
    # the UI reports the one the operator picked. Reject duplicates loudly at
    # this single load/reload chokepoint instead of shipping that footgun.
    # (Blank/missing aliases are never selectable-by-alias, so they don't
    # collide and are exempt.)
    seen: dict[str, int] = {}
    for i, row in enumerate(out):
        alias = str(row.get("alias") or "")
        if not alias:
            continue
        if alias in seen:
            raise ValueError(
                f"duplicate model alias {alias!r} in models[] "
                f"(rows {seen[alias]} and {i}); aliases must be unique because "
                f"the /model picker and bridge resolve by alias and bind only "
                f"the first match. Rename one in ~/.sharedagents/config.json."
            )
        seen[alias] = i
    return out


def has_capability(model_row: dict, capability: str) -> bool:
    """True if `model_row["capabilities"]` contains `capability`.

    Empty / missing `capabilities` is treated as text-only by convention.
    Comparison is case-insensitive and tolerates whitespace.
    """
    if not isinstance(model_row, dict):
        return False
    caps = model_row.get("capabilities") or []
    if not isinstance(caps, list):
        return False
    needle = (capability or "").strip().lower()
    if not needle:
        return False
    return any(str(c).strip().lower() == needle for c in caps)


def find_model_with_capability(cfg: AgentConfig, capability: str) -> dict | None:
    """Return the first configured model row carrying `capability`.

    Scans `cfg.models[]` in declared order, then falls back to the
    synthetic primary row (so a single-primary config that declares
    `capabilities` on the global record still matches). Returns None
    when no row qualifies — caller decides what to do (proceed with
    the text-only model, log a warning, etc.).
    """
    for row in cfg.models or []:
        if has_capability(row, capability):
            return dict(row)
    primary = cfg._primary_model_row()
    if has_capability(primary, capability):
        return dict(primary)
    return None


def list_agents() -> list[str]:
    """Names of all configured agents (alphabetical)."""
    root = agents_root()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "config.json").exists())


# Default workspace templates — verbatim copies of the canonical defaults
# Operator-editable context templates scaffolded into every new agent's
# home directory on first run.

_AGENTS_TEMPLATE = """\
# AGENTS.md — Your Workspace

This folder is home: your config, memory, and notes live here.

## Startup

Use the runtime-provided context (this file and recent memory). Don't re-read
these files unless needed — just start working.

## ⛔ Protected `main` & Local Checkout Sync (owner ruling 2026-09-16)

Remote `main` is protected and can ONLY be modified via a Pull Request merge; never push directly to `origin/main`. The local `main` checkout may ONLY be synchronized to remote `main` (e.g. `git pull --ff-only origin main` or `git reset --hard origin/main`); never make code edits, staging, commits, or local merges of feature branches on `main`. All work must be developed in an isolated task branch worktree (`.worktrees/<branch>`) and merged via PR. Instructions to "merge to main" mean ensuring the PR is merged on the remote forge, then fast-forward syncing local `main` — never running `git merge` locally on the `main` checkout.

## Guidelines

- **Always give a real reply.** A user message gets a user reply — never
  end a turn with only a tool marker or internal note.
- **Save durable facts as they land.** A confirmed decision, user preference,
  or fix found -> call `remember` this turn without announcing it.
- **Be resourceful before asking.** Read files, search code, and verify before asking.
  Return answers, not questions.
- **Execution bias.** Concise when needed, thorough when it matters. Prefer
  verifying against live state over speculation.

## Memory

- `memory/YYYY-MM-DD.md` — raw daily log (append-only)
- `MEMORY.md` — curated long-term memory. Main session ONLY.

Write memory ONLY via `remember(action, target, content)` — `action`:
add|replace|remove, `target`: memory|daily. `Write`/`Edit`/`Bash` against
MEMORY.md or `memory/*.md` are refused by the runtime. Use `memory_search`
before answering questions about past work.

## Safety

- Never exfiltrate private data.
- Ask before destructive operations or actions that leave the host; prefer
  non-destructive alternatives. When in doubt, confirm first.
"""

# MEMORY.md — curated long-term memory. Canonical OpenClaw doesn't ship
# a MEMORY.md template (the runtime treats it as user/agent-curated
# state), but sharedagents seeds a minimal stub so the file is present from
# day one and `scaffold_context_files` will re-create it if the model
# (or the user) deletes it. The agent itself fills the body over time
# via `Read` / `Write` / `Edit` tool calls — same
# pattern OpenClaw documents in `AGENTS.md` (`### 🧠 MEMORY.md - Your
# Long-Term Memory`).
_MEMORY_TEMPLATE = """\
# MEMORY.md - Curated Long-Term Memory

_Your distilled memory. Decisions, lessons, opinions, preferences,
project context. Anything worth carrying across sessions._

_Daily raw logs live at `memory/YYYY-MM-DD.md`; periodically (during
heartbeats or close-out) review them and lift the worth-keeping bits
into this file with a `## <Topic> (Updated YYYY-MM-DD)` heading._

_Edit this file freely via tool calls. It is part of your context,
not an append-only log — pruning outdated entries is fine and
encouraged._
"""

_TEMPLATES = {
    "AGENTS.md": _AGENTS_TEMPLATE,
    "MEMORY.md": _MEMORY_TEMPLATE,
}

# The host-safety contract is shared by every tool-call prompt. Tool schemas no
# longer expose a ``criticality`` property, so the model must not be told
# to emit one; the runtime derives and enforces severity host-side.
_CRITICALITY_CONTRACT = """\
The runtime derives tool criticality from the registered tool kind and, for
shell commands, a host-side safety classifier. Do not add a `criticality`
argument unless the individual tool schema explicitly declares one. Follow the
schema exactly; unknown arguments are rejected. The host confirms destructive
operations, including irreversible or high-blast-radius actions, before they
run and rejects forbidden operations outright.
"""

# DEFAULT tool-call schema (scaffolded as TOOLCALL.md). Used by every
# model that doesn't ship a model-specific schema — the wire format is
# left to the upstream's own chat template / tool-call parser; sharedagents
# only layers on the criticality contract.
_TOOLCALL_TEMPLATE = "# TOOL CALLS\n\n" + _CRITICALITY_CONTRACT

# Qwen3-family native tool-call format. Lifted VERBATIM from the format
# block in the model's own shipped `chat_template.jinja` — the model was
# trained to emit calls as an inner `<function=...>` block (one
# `<parameter=NAME>` per argument) nested inside `<tool_call>` tags, and
# reads tool results back from `<tool_response>` tags. Reinforcing the
# trained shape in the system prompt curbs the documented intermittent
# drift that otherwise leaks a half-formed call into the visible answer.
#
# Deliberately GENERATION-NEUTRAL. The Qwen3 family (3.5 / 3.6 / 3.8 …)
# shares this wire format, so naming one generation guarantees staleness
# the moment the served model is upgraded: until 2026-09-16 this section
# opened with "You are a Qwen3.6 model" while the live replica was
# already serving Qwen3.8.
#
# Also PARSER-NEUTRAL. The vLLM tool-call parser is the host's choice and
# differs per build — `qwen3_coder` on the curated fp8 recipe,
# `qwen3_xml` for AWQ / generic Qwen3 (see
# `sharedagents_model.local_gpu._parsers_for_model`). The text used to assert
# `--tool-call-parser qwen3_coder` as fact; the replica actually serving
# this agent runs `qwen3_xml`. Describe the SHAPE, not a flag the host
# may not be running.
_QWEN3_TOOLCALL_FORMAT = """\
You were trained on ONE tool-call format — emit calls in exactly that
shape so the server's Qwen3 XML tool-call parser reads them cleanly.
Drifting from it (a Hermes-style `<tool_call>{"name": ...}` JSON
blob, or a stray `</function_invocation>` close tag) makes the call
leak into your visible reply, forcing the runtime to salvage it.

When you call a function, reply in EXACTLY this form — an inner
`<function=...></function>` block nested inside `<tool_call>` tags,
one `<parameter=NAME>` block per argument, each value on its own
line(s):

```
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>
```

Rules (from your training):

- Function calls MUST follow this format: an inner `<function=...>`
  block nested within `<tool_call></tool_call>` tags. No JSON object,
  no echo of the `<tools>` list, no suffix text.
- Required parameters MUST be specified.
- You may write optional reasoning in natural language BEFORE the
  function call, but NOT after it — the call is the last thing you emit.
- Tool results come back wrapped in `<tool_response>...</tool_response>`.
- One `<tool_call>` block per function; for parallel calls emit several
  back-to-back.
"""

# Qwen3 tool-call schema (scaffolded as TOOLCALL-QWEN3.md). Self-contained:
# the model family's native wire format FIRST, then the shared criticality
# contract. The context loader swaps this in for the default whenever the
# active model's family is `qwen3`. The filename is generation-neutral on
# purpose — the wire format is a property of the family, not of 3.6.
_TOOLCALL_QWEN3_TEMPLATE = "# TOOL CALLS\n\n" + _QWEN3_TOOLCALL_FORMAT + "\n\n" + _CRITICALITY_CONTRACT

_MODE_PLAN_TEMPLATE = """\
# MODE — PLAN

This is plan mode. Only `read_only` tool calls run. Anything classified
as `mutating` or `destructive` is denied before execution.

Read-only is broad: `Read`, `Glob`, `Grep`, `Skill`, web search,
AND read-only shell — including read-only PIPELINES and chains
(`cat f | grep x`, `ls && cat y`, `grep -E "a|b" file`). A shell call
is denied only if a segment writes/deletes (`> file`, `tee`, `rm`,
`mkdir`, …) or can't be proven read-only. So inspect freely; you don't
need additional authority to investigate.

Use this mode to design, compare, research, or analyze. Be exhaustive:

- Surface tradeoffs explicitly.
- Ground claims via tool calls (read files, search the web, list the
  current state); don't speculate when you can verify.
- Cite sources from your tool results — name the URL, the file path,
  the command output you're drawing from.
- Don't compress the final answer because you've already done the
  thinking. The user wants the analysis, not just the conclusion.
- Aim for thoroughness over brevity when the question warrants it.

For any non-trivial build / change / refactor, gather enough context to
make the plan concrete. Read the relevant files, inspect current state,
and load a skill only when a named or clearly matching runbook would
change the plan. Keep the plan grounded in evidence, not ceremony.

When the plan is ready, present it and call `ExitPlanMode(plan="…")` for
operator readiness review. This read-only tool leaves the current mode unchanged
and never starts Build.
Implementation requires a separate operator request, approval, and authority.
Plan first, propose second, then stop.
"""

_MODE_ASK_TEMPLATE = """\
# MODE — ASK

Permission prompts fire only for truly destructive host actions
(drop database, shutdown, service stop, force-delete unmerged branches,
container teardown). Routine edits, tests and merged-branch cleanup do
not prompt.

Use the `question` tool only for architecture or intent forks that
change the design. Do not interview for routine work. Filesystem wipes
of `/` or `$HOME` are denied by the host — do not ask about them.
"""

_MODE_AUTO_TEMPLATE = """\
# MODE — AUTO

Allow-all applies to tool permissions, not user questions. No permission
cards: configuration already approved tools. Filesystem wipes of `/` or
`$HOME` are denied by the host before they run.

Use the `question` tool to clarify intent and improve alignment when a
missing choice changes the work. Do not call `question` for tool
permission. Routine work proceeds; ask one focused question at a time
with a recommended default.

Use tools liberally. Verify before answering. When the task is complex,
think before you act. Cite the file paths, command outputs, and URLs
that informed your decisions so the user can audit your work.
"""

# The ONE persona section. Deliberately model-AGNOSTIC: it makes no
# claim about parameter count, architecture, training process, context
# window, or which weights are serving the session.
#
# Replaced a table of 14 per-model personas that asserted
# exactly those things ("You are Qwen3.6 — a dense 27-billion-parameter
# model …", "You are DeepSeek V4 Pro — 1.6 trillion parameters …").
# That design failed three ways and is deliberately not coming back:
#
#   1. STRUCTURALLY STALE. The text was keyed to a model GENERATION, so
#      upgrading the served model silently left a false identity in the
#      prompt — the live replica had moved to Qwen3.8 while the section
#      still said "You are Qwen3.6".
#   2. MATCHED NOTHING IN PRACTICE. Resolution used ``str.startswith``
#      against the full model id, but real ids are namespaced
#      (``qwen/qwen3.8-max``, ``anthropic/claude-opus-5``,
#      ``google/gemini-3.7-flash``), so 32 of the 40 configured models
#      fell through to the generic persona and never matched the family
#      entry written for them.
#   3. UNVERIFIABLE AND OFTEN FALSE. The runtime cannot know the
#      architecture of a model reached through an OpenAI-compatible
#      endpoint, and the "signature strengths" bullets ("Never skip
#      thinking", "commit", "don't over-reason") acted as behavioural
#      overrides competing with the runtime's own mode contract.
#
# The reference implementation studied here (opencode v1.18.30) ships
# NO model-identity prose in any of its 10 per-family prompts: no
# parameter counts, no architecture, no training details, no context
# window. Behaviour that genuinely differs per model already lives
# where it belongs — in ``sharedagents_model.models`` sampling profiles.
# Keep this section short and true for every model the operator may
# route to.
_PERSONA_TEMPLATE = """\
# about this agent

You are an sharedagents agent — a capable coding assistant. sharedagents routes
this session to whichever model the operator has configured, and that
choice can change between sessions: do not assume, claim, or reason
about which weights you are running on.

Focus on technical accuracy and clear communication. Think before
acting on complex tasks, and prefer verifying over asserting — tool
results, not priors, are the ground truth for this workspace. The
operator-owned `AGENTS.md` carries this agent's working rules; follow
it over general habit.
"""


# ``_MEMORY_SYNTH_TEMPLATE`` (the operator-editable memory-synthesizer prompt)
# moved to ``sharedagents_core.kernel.config`` and is re-exported at the top of this
# module; it is still available here as ``_MEMORY_SYNTH_TEMPLATE`` for
# ``_SHAREDAGENTS_TEMPLATES`` below.

# Runtime-shipped /goal command body (not written to agent home).
# ``file_commands`` injects this at the ``<agent_home>/commands``
# precedence slot as the primary source, with the shipped goal
# template as the fallback below the shared dirs.
_GOAL_TEMPLATE = """\
---
description: Set a goal — keep working until the condition is met.
argument-hint: "<condition>"
---

# /goal — keep working until the condition is met

You are `{{name}}`. Work on this until it is true:

> $ARGUMENTS

After every turn a separate evaluator reads this conversation and checks
whether the condition is met. If it is not, you are sent straight back to
work with the evaluator's reason. The goal ends when the condition is met.

- Work autonomously. Do not stop to ask for confirmation; make reasonable
  decisions yourself and keep going.
- The evaluator judges from the conversation only, so show the evidence in
  it: run the test and show its output, read back the file, query the
  endpoint. A claim without evidence counts as not met.
- If the condition genuinely cannot be achieved in this session, explain
  exactly why and what you tried; the evaluator decides.
- Do not call `ScheduleWakeup` or any scheduling tool — continuation is
  automatic.
"""


_WORKSPACE_TEMPLATE = """\
# WORKSPACE

You are an sharedagents agent named `{{name}}`. Your workspace ("agent home")
is `{{home}}`. Operator context lives in `AGENTS.md`; memory files
(`memory/YYYY-MM-DD.md`, `MEMORY.md`) live here too. Write to this
directory; do not invent paths like `~/.claw/` or `~/.openclaw/`.

The user's home is `{{user_home}}`. Sessions persist to the unified
`~/.sharedagents/sessions/<id>.jsonl` (one folder for every agent —
append-only event log; the runtime writes them, you don't). Per-
agent attribution lives in each session's ``agent_name`` header,
not in the path.
"""


_SHAREDAGENTS_TEMPLATES = {
    "WORKSPACE.md": _WORKSPACE_TEMPLATE,
    "GOAL.md": _GOAL_TEMPLATE,
    "TOOLCALL.md": _TOOLCALL_TEMPLATE,
    "TOOLCALL-QWEN3.md": _TOOLCALL_QWEN3_TEMPLATE,
    "MODE-PLAN.md": _MODE_PLAN_TEMPLATE,
    "MODE-ASK.md": _MODE_ASK_TEMPLATE,
    "MODE-AUTO.md": _MODE_AUTO_TEMPLATE,
    "MEMORY-SYNTH.md": _MEMORY_SYNTH_TEMPLATE,
    "PERSONA.md": _PERSONA_TEMPLATE,
}


def hardcoded_context_body(
    filename: str,
    *,
    agent_name: str = "",
    agent_home: Path | None = None,
) -> str | None:
    """Return a runtime-shipped context section (not read from agent home)."""
    body = _SHAREDAGENTS_TEMPLATES.get(filename)
    if body is None:
        return None
    if agent_name:
        body = body.replace("{{name}}", agent_name)
    if filename == "WORKSPACE.md" and agent_home is not None:
        user_home = str(Path.home())
        return body.replace("{{home}}", str(agent_home)).replace("{{user_home}}", user_home)
    return body


def hardcoded_goal_markdown(*, agent_name: str = "") -> str:
    """Full ``/goal`` command markdown (frontmatter + body), runtime-shipped."""
    body = hardcoded_context_body("GOAL.md", agent_name=agent_name)
    return body or _GOAL_TEMPLATE.replace("{{name}}", agent_name or "agent")


def personalize_agents_md(cfg: AgentConfig, name: str) -> bool:
    """Pre-fill ``AGENTS.md`` Name / What-to-call-them lines after wizard setup."""
    name = (name or "").strip()
    if not name:
        return False
    p = cfg.home / "AGENTS.md"
    if not p.exists():
        return False
    body = p.read_text(encoding="utf-8")
    new = re.sub(
        r"^- \*\*Name:\*\*[^\n]*$",
        f"- **Name:** {name}",
        body,
        count=1,
        flags=re.MULTILINE,
    )
    new = re.sub(
        r"^- \*\*What to call them:\*\*[^\n]*$",
        f"- **What to call them:** {name}",
        new,
        count=1,
        flags=re.MULTILINE,
    )
    if new == body:
        block = f"\n## About your human\n\n- **Name:** {name}\n- **What to call them:** {name}\n"
        if "## About your human" not in body:
            new = body.rstrip() + block + "\n"
        else:
            return False
    p.write_text(new, encoding="utf-8")
    return True


def personalize_user_md(cfg: AgentConfig, name: str) -> bool:
    """Backward-compatible alias for :func:`personalize_agents_md`."""
    return personalize_agents_md(cfg, name)


def scaffold_context_files(cfg: AgentConfig, *, overwrite: bool = False) -> list[Path]:
    """Drop operator-editable templates into the agent's home.

    Scaffolds ``AGENTS.md`` and ``MEMORY.md`` only. ``HEARTBEAT.md`` is
    created when ``heartbeat_enabled`` is turned on (see
    ``heartbeat.sync_heartbeat_files``). WORKSPACE, MODE-*, TOOLCALL*,
    GOAL (``/goal``), and MEMORY-SYNTH are runtime-hardcoded — not
    written here.
    """
    written: list[Path] = []
    cfg.home.mkdir(parents=True, exist_ok=True)
    for fname, body in _TEMPLATES.items():
        path = cfg.home / fname
        if path.exists() and not overwrite:
            continue
        path.write_text(body, encoding="utf-8")
        written.append(path)

    return written


# Files that belong at the agent-home root.
_AGENT_ROOT_FILES: frozenset[str] = frozenset(
    {
        "config.json",
        "AGENTS.md",
        "MEMORY.md",
        "HEARTBEAT.md",
        "hooks.json",
        "heartbeat-state.json",
        "secrets.env",
        "gateway.pid",
    }
)

# Legacy per-file context folded into ``AGENTS.md`` by
# :func:`merge_agent_context_files`.
_MERGE_BEFORE_AGENTS: tuple[str, ...] = ("SOUL.md", "USER.md")
_MERGE_AFTER_AGENTS: tuple[str, ...] = ("TOOLS.md",)

# Runtime-hardcoded sections that must not live at the agent root.
_HARDCODED_CONTEXT_FILES: tuple[str, ...] = (
    "GOAL.md",
    "TOOLCALL.md",
    "TOOLCALL-QWEN3.md",
    "WORKSPACE.md",
    "MODE-PLAN.md",
    "MODE-ASK.md",
    "MODE-AUTO.md",
    "MEMORY-SYNTH.md",
)

# Superseded names that are still swept out of an agent root if an older
# install left one behind. They are NOT injected any more — they are only
# archived so a stale copy cannot shadow the current name:
#   * ``TOOLCALL-QWEN3.6.md`` — generation-pinned twin of
#     ``TOOLCALL-QWEN3.md``.
#   * every ``PERSONA-*.md`` — the retired per-model persona table.
_RETIRED_CONTEXT_FILES: tuple[str, ...] = (
    "TOOLCALL-QWEN3.6.md",
    "PERSONA-QWEN3.6.md",
    "PERSONA-DEEPSEEK-V4-FLASH.md",
    "PERSONA-DEEPSEEK-V4-PRO.md",
    "PERSONA-GEMMA4.md",
    "PERSONA-CLAUDE.md",
    "PERSONA-GPT4.md",
    "PERSONA-GEMINI.md",
    "PERSONA-KIMI.md",
    "PERSONA-META.md",
    "PERSONA-BEAST.md",
    "PERSONA-CODEX.md",
    "PERSONA-COPILOT.md",
    "PERSONA-TRINITY.md",
)

_LEGACY_CONTEXT_FILES: tuple[str, ...] = (
    *_MERGE_BEFORE_AGENTS,
    *_MERGE_AFTER_AGENTS,
    *_HARDCODED_CONTEXT_FILES,
    *_RETIRED_CONTEXT_FILES,
)


def merge_agent_context_files(home: Path) -> list[str]:
    """Fold legacy context into ``AGENTS.md``; archive stray root files.

    Idempotent: skips blocks already present in ``AGENTS.md``. Returns
    relative paths of files changed this call.
    """
    import shutil

    home = Path(home)
    if not home.is_dir():
        return []
    changed: list[str] = []
    archive_ctx = home / "archive" / "context"
    agents_path = home / "AGENTS.md"
    agents_body = agents_path.read_text(encoding="utf-8").strip() if agents_path.is_file() else ""

    def _read_legacy(fname: str) -> str:
        for src in (home / fname, archive_ctx / fname):
            if src.is_file():
                return src.read_text(encoding="utf-8").strip()
        return ""

    prefix_parts: list[str] = []
    for fname in _MERGE_BEFORE_AGENTS:
        block = _read_legacy(fname)
        if block and block not in agents_body:
            prefix_parts.append(block)

    suffix_parts: list[str] = []
    for fname in _MERGE_AFTER_AGENTS:
        block = _read_legacy(fname)
        if block and block not in agents_body:
            suffix_parts.append(block)

    if prefix_parts or suffix_parts:
        merged_sections = prefix_parts + ([agents_body] if agents_body else []) + suffix_parts
        merged = "\n\n".join(section for section in merged_sections if section)
        if merged != agents_body:
            agents_path.parent.mkdir(parents=True, exist_ok=True)
            agents_path.write_text(merged + "\n", encoding="utf-8")
            changed.append("AGENTS.md")

    notes = home / "notes"
    for fname in _LEGACY_CONTEXT_FILES:
        src = home / fname
        if not src.is_file():
            continue
        dest = archive_ctx / fname
        if dest.exists():
            src.unlink()
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        changed.append(f"archive/context/{fname}")

    for entry in list(home.iterdir()):
        if not entry.is_file():
            continue
        if entry.name in _AGENT_ROOT_FILES or entry.name in _LEGACY_CONTEXT_FILES:
            continue
        dest = notes / entry.name
        if dest.exists():
            entry.unlink()
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry), str(dest))
        changed.append(f"notes/{entry.name}")

    return changed


def merge_all_agent_context_files() -> dict[str, list[str]]:
    """Run :func:`merge_agent_context_files` for every configured agent."""
    from sharedagents_core.layout import agents_root

    out: dict[str, list[str]] = {}
    root = agents_root()
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / "config.json").is_file():
            continue
        rel = merge_agent_context_files(child)
        if rel:
            out[child.name] = rel
    return out


# `ensure_main_agent()` was removed 2026-07-21 when the built-in
# `main` orchestrator was retired. jarvis is the sole orchestrator and is
# scaffolded by `ensure_voice_agent()`; there is no `main` agent to materialize.
