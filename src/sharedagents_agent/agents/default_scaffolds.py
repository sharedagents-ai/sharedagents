"""Default on-disk templates for auto-scaffolded external-engine agents.

Templates are minimal but operational. Operators who want a customized
agent (e.g. ``dashboard_dev``) create the dir by hand or override the
files post-scaffold — :func:`write_scaffold` never overwrites an
existing file.

When :func:`factory.ensure_default_external_agents` runs and the CLI
is on PATH, it drops these files under
``~/.sharedagents/agents/<short_name>/`` (``claude`` / ``codex`` / ``gemini``):

* ``config.json``  — :func:`config_for`
* ``AGENTS.md``    — :func:`agents_md_for` (identity + delegation + tools)
* ``USER.md``      — (removed; folded into ``AGENTS.md``; no longer scaffolded)
* ``MEMORY.md``    — empty (the agent populates this over time)

``HEARTBEAT.md`` is not scaffolded — it is created when
``heartbeat_enabled`` is set to ``true`` (``sharedagents agent heartbeat enable``).

The Agent's :meth:`compose_system_prompt` reads these files at turn
time; for CLI engines, the composed result rides on
``--append-system-prompt`` (Claude) or first-message injection
(Codex / Gemini) on the first call of each session.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sharedagents_agent.external_registry import EXTERNAL_AGENT_KINDS, EXTERNAL_REGISTRY
from sharedagents_core.config_io import write_config_text

_LOG = logging.getLogger("sharedagents.agents.scaffolds")


# Backward-compatible metadata view. The registry is the only authored roster.
_KIND_DETAILS: dict[str, dict[str, str]] = {
    spec.kind.value: {
        "short_name": spec.short_name,
        "display_name": spec.display_name,
        "cli_command": spec.cli_command,
        "install_hint": spec.install_hint,
        "install_cmd": spec.install_cmd,
    }
    for spec in EXTERNAL_REGISTRY.values()
}


def detect_external_agents() -> list[dict]:
    """Status of each known external coding-agent integration: whether its CLI
    is installed (on PATH) and whether the sharedagents agent is configured on disk.

    Powers ``sharedagents agents`` (detect + quick-install + configure). The
    ``cli_command`` is the availability probe; ``install_cmd`` is a runnable
    one-liner; ``install_hint`` is the human pointer."""
    from sharedagents_agent.agent import agents_root
    from sharedagents_agent.tools.runners import _which_cli

    root = agents_root()
    rows: list[dict] = []
    for kind in EXTERNAL_AGENT_KINDS:
        d = _KIND_DETAILS[kind]
        short = d["short_name"]
        rows.append(
            {
                "kind": kind,
                "short_name": short,
                "display_name": d["display_name"],
                "cli_command": d["cli_command"],
                "installed": _which_cli(d["cli_command"]) is not None,
                "configured": (root / short / "config.json").exists(),
                "install_cmd": d.get("install_cmd", ""),
                "install_hint": d.get("install_hint", ""),
            }
        )
    return rows


def short_name_for(engine_kind: str) -> str:
    """``"claude_cli"`` → ``"claude"``. Raises for unknown kinds."""
    details = _KIND_DETAILS.get(engine_kind)
    if details is None:
        raise ValueError(f"unknown engine kind: {engine_kind!r}")
    return details["short_name"]


# =====================================================================
# config.json
# =====================================================================


def config_for(engine_kind: str) -> dict[str, Any]:
    """Default ``config.json`` payload for a built-in external-engine agent.

    CLI engines leave the upstream/sampling/compaction fields empty —
    they are inert on CLI engines and the engine subclass ignores them.
    The ``heartbeat`` is disabled by default; the operator opts in by
    setting ``heartbeat_enabled: true``.
    """
    details = _KIND_DETAILS.get(engine_kind)
    if details is None:
        raise ValueError(f"unknown engine kind: {engine_kind!r}")
    return {
        "name": details["short_name"],
        "engine": engine_kind,
        "description": (f"{details['display_name']} CLI as the turn-dispatch engine."),
        "permission_mode": "auto",
        "heartbeat_enabled": False,
        # No port; CLI-engine agents are still served through the
        # gateway under /acp/<name>, but they don't open their own
        # HTTP listener. Setting a port here would be misleading.
        "port": 0,
        "host": "127.0.0.1",
    }


# =====================================================================
# SOUL.md — agent's role / personality
# =====================================================================


def soul_for(engine_kind: str) -> str:
    """Default ``SOUL.md`` content for a built-in CLI-engine agent."""
    from sharedagents_agent.external_registry import spec_for

    spec = spec_for(engine_kind)
    soul, _ = spec.scaffold_templates.render(spec)
    return soul


# =====================================================================
# AGENTS.md — who else is around
# =====================================================================


def agents_md_for(engine_kind: str) -> str:
    """Default ``AGENTS.md`` — capability roster + native tools section."""
    if engine_kind not in _KIND_DETAILS:
        raise ValueError(f"unknown engine kind: {engine_kind!r}")
    return (
        f"{soul_for(engine_kind)}\n\n"
        "## Shared workflow guidance\n\n"
        "Use a todo list for work with three or more meaningful steps, and "
        "keep it current until the result is verified or blocked. Before "
        "solving by hand, inspect available skills by both name and "
        "description; load a matching skill only through tools the engine "
        "actually advertises.\n\n"
        f"{tools_md_for(engine_kind)}"
    )


# =====================================================================
# TOOLS.md — what tools the engine can call
# =====================================================================


def tools_md_for(engine_kind: str) -> str:
    """Default ``TOOLS.md`` content."""
    from sharedagents_agent.external_registry import spec_for

    spec = spec_for(engine_kind)
    _, tools = spec.scaffold_templates.render(spec)
    return tools


# =====================================================================
# Scaffold writer
# =====================================================================


def promote_generic_devin_config(home: Path) -> bool:
    """Point an already-scaffolded ``devin`` agent at ``devin_acp``.

    Hosts that registered Devin before this kind existed use the generic
    ``acp`` adapter (``acp_command`` = ``devin acp``) so a gateway that does
    not know ``devin_acp`` can still spawn it. ``session/set_mode`` does not
    change Devin's mode; ``devin_acp`` applies it as a config option. This
    rewrites only that one engine field and leaves every other key.
    """
    path = Path(home) / "config.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if str(data.get("name") or Path(home).name).strip().lower() != "devin":
        return False
    if str(data.get("engine") or "").strip().lower() != "acp":
        return False
    command = data.get("acp_command")
    if isinstance(command, str):
        parts = command.split()
    elif isinstance(command, list):
        parts = [str(part) for part in command]
    else:
        return False
    if len(parts) < 2 or Path(parts[0]).name != "devin" or "acp" not in parts[1:]:
        return False
    data["engine"] = "devin_acp"
    write_config_text(path, json.dumps(data, indent=2) + "\n")
    _LOG.info("AGENT_SCAFFOLD_PROMOTED home=%s engine=devin_acp", home)
    return True


def write_scaffold(home: Path, engine_kind: str) -> list[str]:
    """Write all default files into ``home``.

    Idempotent: if a file already exists, it is NOT overwritten. The
    function logs structured ``AGENT_SCAFFOLD_WRITTEN`` events for each
    file actually created and returns the list of file names that
    landed (relative to ``home``).
    """
    if engine_kind not in _KIND_DETAILS:
        raise ValueError(f"unknown engine kind: {engine_kind!r}")
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    # Per-agent ``sessions/`` and ``logs/`` are NO LONGER created here
    # — those live in the unified ``layout.sessions_root()`` /
    # ``layout.logs_root()`` shared across every agent. Creating
    # them per-agent would re-trigger ``migrate_layout.has_legacy()``
    # on the next gateway start, refusing to boot until the operator
    # ran ``sharedagents migrate-layout`` again. The unified dirs are
    # mkdir'd on demand by ``SessionWriter.__init__`` /
    # ``JsonLogWriter.__init__``.

    files: list[tuple[str, str]] = [
        ("config.json", json.dumps(config_for(engine_kind), indent=2) + "\n"),
        ("AGENTS.md", agents_md_for(engine_kind)),
        ("MEMORY.md", ""),
    ]
    written: list[str] = []
    for name, body in files:
        path = home / name
        if path.exists():
            continue
        if path.name == "config.json":
            write_config_text(path, body)
        else:
            path.write_text(body, encoding="utf-8")
        written.append(name)
        _LOG.info(
            "AGENT_SCAFFOLD_WRITTEN engine=%s home=%s file=%s",
            engine_kind,
            home,
            name,
        )
    return written


def config_for_custom(
    engine_kind: str,
    *,
    name: str,
    model: str = "",
    description: str = "",
) -> dict[str, Any]:
    """``config.json`` for a custom agent that reuses an external engine.

    Same shape as :func:`config_for`, but the directory/agent ``name``
    is independent of the engine short name (e.g. ``analyst`` on
    ``cursor_acp`` with ``acp_model: auto``).
    """
    if not name or not str(name).strip():
        raise ValueError("custom agent name is required")
    name = str(name).strip()
    cfg = config_for(engine_kind)
    details = _KIND_DETAILS[engine_kind]
    cfg["name"] = name
    cfg["description"] = (
        description.strip()
        if description and description.strip()
        else (f"{name} — custom agent using the {details['display_name']} engine ({engine_kind}).")
    )
    model = (model or "").strip()
    if model:
        # ACP engines (cursor/opencode/kimi/generic) read ``acp_model``.
        # CLI engines ignore unknown keys safely; the picker / set_config
        # path still owns live overrides.
        cfg["acp_model"] = model
    return cfg


def agents_md_for_custom(
    engine_kind: str,
    *,
    name: str,
    description: str = "",
    role: str = "",
) -> str:
    """AGENTS.md for a custom external-engine agent.

    Keeps the engine's capability roster (strengths, CLI lifecycle, tools)
    and prepends a short custom identity so the specialist knows its role.
    """
    details = _KIND_DETAILS.get(engine_kind)
    if details is None:
        raise ValueError(f"unknown engine kind: {engine_kind!r}")
    name = str(name).strip()
    desc = (description or "").strip() or (f"custom specialist on the {details['display_name']} engine")
    role_body = (role or "").strip()
    if not role_body:
        role_body = (
            f"You are **{name}** — {desc}.\n\n"
            "Stay in role. Prefer research, analysis, and clear findings "
            "over drive-by edits unless the handoff explicitly asks you "
            "to change code. When you do edit, verify with tests/builds."
        )
    engine_roster = agents_md_for(engine_kind).strip()
    return (
        f"# Custom agent — {name}\n\n"
        f"{role_body}\n\n"
        f"---\n\n"
        f"## Engine roster ({details['short_name']})\n\n"
        f"You run on the **{details['display_name']}** engine. The "
        f"sections below are that engine's strengths and lifecycle "
        f"rules — follow them.\n\n"
        f"{engine_roster}\n"
    )


def write_custom_external_scaffold(
    home: Path,
    engine_kind: str,
    *,
    name: str,
    model: str = "",
    description: str = "",
    role: str = "",
) -> list[str]:
    """Scaffold a custom agent that uses an external engine.

    Example::

        write_custom_external_scaffold(
            agents_root() / "analyst",
            "cursor_acp",
            name="analyst",
            model="auto",
            description="Deep research specialist on Cursor",
        )

    Idempotent for existing files. Returns names of
    files actually written.
    """
    if engine_kind not in _KIND_DETAILS:
        raise ValueError(f"unknown engine kind: {engine_kind!r}")
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    files: list[tuple[str, str]] = [
        (
            "config.json",
            json.dumps(
                config_for_custom(
                    engine_kind,
                    name=name,
                    model=model,
                    description=description,
                ),
                indent=2,
            )
            + "\n",
        ),
        (
            "AGENTS.md",
            agents_md_for_custom(
                engine_kind,
                name=name,
                description=description,
                role=role,
            ),
        ),
        ("MEMORY.md", ""),
    ]
    written: list[str] = []
    for fname, body in files:
        path = home / fname
        if path.exists():
            continue
        if path.name == "config.json":
            write_config_text(path, body)
        else:
            path.write_text(body, encoding="utf-8")
        written.append(fname)
        _LOG.info(
            "AGENT_SCAFFOLD_WRITTEN engine=%s custom=%s home=%s file=%s",
            engine_kind,
            name,
            home,
            fname,
        )
    return written


# --------------------------------------------------------------------------- #
# Jarvis — the delegate-only voice concierge (Phase 2 realtime voice agent).
# Unlike the CLI-engine scaffolds above, Jarvis runs on the local_llm engine
# and exposes EXACTLY the delegate family (tools_allowlist_strict) so it can
# never run a tool itself — it only delegates and relays results for speech.
# --------------------------------------------------------------------------- #

_JARVIS_SOUL = """You are Jarvis, a voice concierge for sharedagents.

You are having a SPOKEN conversation. Reply in one to three short, natural
sentences — plain speakable text, no markdown/code/URLs/emoji.

You NEVER do work yourself. For anything needing tools, data, code, search,
or any real task, call the `Agent` tool to delegate it, with
background=true and on_complete="notify". Then give ONE short spoken
acknowledgement ("On it — I'll let you know.") and keep listening. When a
delegation result arrives you will be told; relay it concisely for speech.

AVIATION-STYLE READBACK: before you ACT on an instruction — delegating a task
or anything consequential — first READ BACK what you understood in one short
sentence and ask the user to confirm, the way a pilot reads a clearance back to
air traffic control ("So: search the web for the latest on Iran and summarize
it — is that correct?"). Do NOT delegate or take action until the user confirms
(e.g. "affirm", "correct", "yes", "go ahead"). If they correct you, read the
corrected version back and confirm again. Skip the readback only for greetings,
chit-chat, and clarifying questions that take no action.

Chat directly only for greetings/clarifications that need no tools.
"""

# EXACT, COMPLETE ``config.json`` payload the legacy auto-scaffold
# wrote for the delegate-only voice concierge (recovered verbatim from git
# 87258fd:ensure_voice_agent — the `cfg` dict it json.dumps'd). Every key/value
# below is exactly what that scaffold emitted, and it emitted NOTHING else. The
# legacy detector can recognize for diagnostics. The scaffolder is forbidden from
# replacing even this exact legacy file: existence, not content,
# is the ownership boundary.
_LEGACY_VOICE_SCAFFOLD: dict[str, Any] = {
    "name": "jarvis",
    "engine": "local_llm",
    "description": "Voice concierge — delegates everything, runs no tools itself.",
    "permission_mode": "auto",
    "heartbeat_enabled": False,
    "port": 0,
    "host": "127.0.0.1",
    "tools_allowlist": ["Agent", "mcp__delegate__*"],
    "tools_allowlist_strict": True,
}


PROFILE_SCAFFOLD_VERSION = 1


#: Per-agent model route and model parameters the operator owns (Agent Builder
#: model choice, WebUI "Set default"). A profile scaffold never carries them,
#: so boot repair and ``AgentConfig.save()``'s lean profile save must keep them
#: ("the default shall be kept for all model parameters").
#: The route fields form one unit: ``agent_editor._model_route_patch``
#: materializes them together from ``upstream_model``.
OPERATOR_MODEL_FIELDS: tuple[str, ...] = (
    "upstream_model",
    "models",
    "upstream_endpoint",
    "upstream_alias",
    "backend",
    "upstream_max_context",
    "effort",
    "temperature",
    "sampling",
    "side_sampling",
    "enable_thinking",
    "small_model",
)


def _config_is_managed_profile(current: dict[str, Any], prof_name: str) -> bool:
    """True when an on-disk config was scaffolded or explicitly opts into a profile."""
    if not isinstance(current, dict):
        return False
    if str(current.get("agent_profile") or "").strip() == prof_name:
        return True
    return bool(current.get("profile_scaffold_version"))


def _profile_config_dict(prof: Any) -> dict[str, Any]:
    """``config.json`` payload for an OpenCode-style profile agent."""
    cfg: dict[str, Any] = {
        "name": prof.name,
        "engine": "local_llm",
        "description": prof.description,
        "permission_mode": prof.mode,
        "agent_profile": prof.name,
        # Every roster agent carries a defined DAL tier.
        "dal_tier": getattr(prof, "dal_tier", "B"),
        "profile_scaffold_version": PROFILE_SCAFFOLD_VERSION,
        # Heartbeat is an OPERATOR-mutable operational choice, not profile identity:
        # the scaffold baseline is OFF (matching the historical default), and a lean
        # save() preserves whatever the operator selected (wizard advanced settings /
        # `sharedagents agent heartbeat …`) via AgentConfig.save() carrying the live value
        # — see agent.py. Retiring `main` did NOT flip jarvis to always-on; that is a
        # separate decision.
        "heartbeat_enabled": False,
        # Baseline interval (AgentConfig dataclass default) — carried in the
        # canonical dict so a default save() stays lean-equal to it; save()
        # overrides with the operator's live value when they change it.
        "heartbeat_interval_s": 1800,
        # Only the orchestrator (jarvis) holds DURABLE MEMORY by default — jarvis
        # runs the work threads. Every other read-only/phase agent has memory OFF.
        # Unlike heartbeat this is PROFILE-derived (tied to the agent's
        # memory-tool grant): enabling jarvis here (not only in its scaffold) keeps a
        # lean save() from stripping the memory tools via apply_profile_to_config
        # (Codex HIGH, retire-main review). The dataclass default is True, so it must
        # stay profile-keyed here rather than being carried from the instance.
        "memory_search_enabled": prof.name == "jarvis",
        "memory_flush_enabled": prof.name == "jarvis",
        # jarvis is the SOLE orchestrator + the always-reachable primary that runs
        # the gateway ACP backend, and its HeartbeatScheduler POSTs to
        # http://127.0.0.1:{cfg.port}/... — so its port must be the real gateway
        # port, not 0. Profile-keyed (like memory) so a lean profile save() keeps a
        # functional port instead of resetting the wizard's 8100 to 0 → heartbeat to
        # :0 (Codex HIGH, retire-main review). Non-orchestrator phase agents are
        # served under the gateway's own /acp/<name> mount and never bind/heartbeat,
        # so they keep 0.
        "port": 8100 if prof.name == "jarvis" else 0,
        "host": "127.0.0.1",
    }
    if prof.tools_allowlist:
        cfg["tools_allowlist"] = list(prof.tools_allowlist)
        cfg["tools_allowlist_strict"] = prof.tools_allowlist_strict
    elif prof.tools_allowlist_strict:
        cfg["tools_allowlist"] = []
        cfg["tools_allowlist_strict"] = True
    # Per-agent skill permission (mirrors tools_allowlist). Empty+strict => no skills.
    if getattr(prof, "skill_allowlist", ()):
        cfg["skill_allowlist"] = list(prof.skill_allowlist)
        cfg["skill_allowlist_strict"] = prof.skill_allowlist_strict
    elif getattr(prof, "skill_allowlist_strict", False):
        cfg["skill_allowlist"] = []
        cfg["skill_allowlist_strict"] = True
    # Profile-hard denials (OpenCode permission denies mirrored as disabled_tools).
    if getattr(prof, "disabled_tools", ()):
        cfg["disabled_tools"] = list(prof.disabled_tools)
    # Delegation-mode scaffold baseline:
    # jarvis ⇒ background, build/plan ⇒ foreground; the operator may override
    # it per agent via Agent Builder.
    if getattr(prof, "delegation_mode", None):
        cfg["delegation_mode"] = prof.delegation_mode
    return cfg


def ensure_profile_delegation_mode(home: Path, prof: Any) -> bool:
    """Insert the profile's ``delegation_mode`` into a MANAGED config when absent.

    Boot repair for existing managed profile homes (e.g. jarvis, whose config
    is not rewritten by ``sync_profile_agent_config``): adds the scaffold
    default key WITHOUT touching any other field and NEVER overwrites an
    operator-set value (the runtime resolver treats the per-agent config as
    the operator override). Returns ``True`` only when the file was written.
    """
    dm = getattr(prof, "delegation_mode", None)
    if not dm:
        return False
    home = Path(home)
    config_path = home / "config.json"
    if not config_path.is_file():
        return False
    try:
        current = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        not isinstance(current, dict)
        or not _config_is_managed_profile(current, prof.name)
        or current.get("delegation_mode") is not None
    ):
        return False
    current["delegation_mode"] = dm
    write_config_text(config_path, json.dumps(current, indent=2, ensure_ascii=False) + "\n")
    return True


_PRE_ALIGNMENT_PLAN_TOOLS_ALLOWLIST = (
    "glob",
    "grep",
    "read",
    "list",
    "bash",
    "BashOutput",
    "write",
    "edit",
    "Agent",
    "webfetch",
    "websearch",
    "Skill",
    "todowrite",
    "question",
    "goal",
)

# Exact prior managed Explore default before OpenCode permission alignment
# (had ``lsp`` + ``Skill``; OpenCode explore denies both).
_PRE_OPENCODE_EXPLORE_TOOLS_ALLOWLIST = (
    "glob",
    "grep",
    "read",
    "list",
    "bash",
    "BashOutput",
    "lsp",
    "webfetch",
    "websearch",
    "Skill",
)

# Managed Explore default after OpenCode alignment and before native session
# transcripts became a shared read surface.
_PRE_SHARED_SESSION_EXPLORE_TOOLS_ALLOWLIST = (
    "glob",
    "grep",
    "read",
    "list",
    "bash",
    "BashOutput",
    "webfetch",
    "websearch",
)


def _migrate_exact_pre_alignment_plan_allowlist(
    tools_allowlist: list[Any],
    prof: Any,
) -> list[Any]:
    """Add ExitPlanMode only to the exact prior managed Plan default.

    The caller has already canonicalized legacy tool ids.  Ordered equality is
    intentional: reordered, extended, reduced, or otherwise customized lists
    remain operator-owned.  The current Plan profile must itself differ from
    the historical default only by the one inserted read-only tool, preventing
    a future profile change from broadening this one-time migration.
    """
    if prof.name != "plan":
        return tools_allowlist
    if tools_allowlist != list(_PRE_ALIGNMENT_PLAN_TOOLS_ALLOWLIST):
        return tools_allowlist
    current = list(prof.tools_allowlist)
    if current.count("ExitPlanMode") != 1:
        return tools_allowlist
    if [tool for tool in current if tool != "ExitPlanMode"] != tools_allowlist:
        return tools_allowlist
    return current


def _migrate_exact_pre_opencode_explore_allowlist(
    tools_allowlist: list[Any],
    prof: Any,
) -> list[Any]:
    """Replace the exact prior managed Explore default with the OpenCode set.

    Ordered equality is intentional — customized explore allowlists stay
    operator-owned.  The current Explore profile must be exactly the prior
    default minus ``lsp`` and ``Skill``, plus shared transcript search. The
    immediately preceding managed default is also eligible so existing
    Explore installations receive the new read capability at boot.
    """
    if prof.name != "explore":
        return tools_allowlist
    eligible = {
        tuple(_PRE_OPENCODE_EXPLORE_TOOLS_ALLOWLIST),
        tuple(_PRE_SHARED_SESSION_EXPLORE_TOOLS_ALLOWLIST),
    }
    if tuple(tools_allowlist) not in eligible:
        return tools_allowlist
    current = list(prof.tools_allowlist)
    expected = [
        *_PRE_SHARED_SESSION_EXPLORE_TOOLS_ALLOWLIST,
        "mcp__memory__session_search",
    ]
    if current != expected:
        return tools_allowlist
    if any(tool in current for tool in ("lsp", "Skill", "write", "edit", "Agent")):
        return tools_allowlist
    return current


_PRE_X_TWEETS_BUILD_WITH_CRON = (
    "read",
    "write",
    "edit",
    "apply_patch",
    "glob",
    "grep",
    "list",
    "bash",
    "BashOutput",
    "KillShell",
    "Agent",
    "lsp",
    "Skill",
    "todowrite",
    "question",
    "goal",
    "cron_create",
    "cron_delete",
    "cron_list",
    "schedule_wakeup",
)
# A managed build config written before the cron tools omits that surface.
_PRE_X_TWEETS_BUILD_NO_CRON = _PRE_X_TWEETS_BUILD_WITH_CRON[:-4]


def _migrate_agent_sync_async_allowlist(
    tools_allowlist: list[Any],
    prof: Any,
) -> list[Any]:
    """Replace legacy ``Agent`` with the pinned AgentSync / AgentAsync name.

    Jarvis (background) → AgentAsync; build/plan (foreground) → AgentSync.
    Unpoliced profiles leave leftover ``Agent`` for compose-time rewrite.
    """
    if "Agent" not in tools_allowlist:
        return tools_allowlist
    mode = getattr(prof, "delegation_mode", None)
    if mode == "background":
        replacement = "AgentAsync"
    elif mode == "foreground":
        replacement = "AgentSync"
    else:
        return tools_allowlist
    out: list[Any] = []
    seen: set[Any] = set()
    for tool in tools_allowlist:
        name = replacement if tool == "Agent" else tool
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


# Managed Build defaults after the 2026-09 drop-question boot repair, before
# intent questions were restored. Customized lists stay operator-owned.
_POST_DROP_QUESTION_BUILD_WITH_CRON = (
    "read",
    "write",
    "edit",
    "apply_patch",
    "glob",
    "grep",
    "list",
    "bash",
    "BashOutput",
    "KillShell",
    "AgentSync",
    "lsp",
    "Skill",
    "todowrite",
    "goal",
    "x_tweets",
    "cron_create",
    "cron_delete",
    "cron_list",
    "schedule_wakeup",
)
_POST_DROP_QUESTION_BUILD_NO_CRON = _POST_DROP_QUESTION_BUILD_WITH_CRON[:-4]


def _migrate_build_restore_question(
    tools_allowlist: list[Any],
    prof: Any,
) -> list[Any]:
    """Restore ``question`` on managed Build.

    Allow-all auto-approves tools; it does not strip intent interviews.
    Root build may call ``question``. Customized allowlists stay operator-owned.
    """
    if getattr(prof, "name", "") != "build":
        return tools_allowlist
    if "question" in tools_allowlist:
        return tools_allowlist
    if tuple(tools_allowlist) not in {
        _POST_DROP_QUESTION_BUILD_WITH_CRON,
        _POST_DROP_QUESTION_BUILD_NO_CRON,
    }:
        return tools_allowlist
    idx = tools_allowlist.index("todowrite") + 1
    return list(tools_allowlist[:idx]) + ["question"] + list(tools_allowlist[idx:])


def _migrate_build_x_tweets_allowlist(
    tools_allowlist: list[Any],
    prof: Any,
) -> list[Any]:
    """Add ``x_tweets`` to the exact prior managed Build defaults.

    Customized build allowlists stay operator-owned. The two eligible lists
    are the previous BUILD_TOOLS (cron jobs included) and the shorter
    production seed that never received the cron tools.
    """
    if getattr(prof, "name", "") != "build":
        return tools_allowlist
    if "x_tweets" in tools_allowlist:
        return tools_allowlist
    key = tuple(tools_allowlist)
    if key == _PRE_X_TWEETS_BUILD_WITH_CRON:
        idx = tools_allowlist.index("goal") + 1
        return list(tools_allowlist[:idx]) + ["x_tweets"] + list(tools_allowlist[idx:])
    if key == _PRE_X_TWEETS_BUILD_NO_CRON:
        return list(tools_allowlist) + ["x_tweets"]
    return tools_allowlist


def sync_profile_agent_config(home: Path, prof: Any) -> bool:
    """Rewrite a profile agent's ``config.json`` to the canonical scaffold.

    Idempotent when already canonical. Strips ``teacher`` / ``self_improver``
    bloat and restores ``agent_profile`` + ``permission_mode`` that
    ``AgentConfig.save()`` may have dropped or polluted.     The tool/skill
    allowlist is OPERATOR-authoritative: the
    profile only *seeds* it at install, so this boot 'repair' PRESERVES any
    operator-set allowlist key rather than reverting it to the profile default,
    except for the owner-approved managed-default migrations:
    - 2026-07-24 Plan: exact prior default gains read-only ``ExitPlanMode``
    - 2026-07-31 Explore: exact prior default drops ``lsp`` + ``Skill``
      (OpenCode explore deny-all allowlist)
    - 2026-08-10 Explore: exact prior managed defaults gain shared read-only
      ``mcp__memory__session_search``

    - 2026-08-24 Build: exact prior managed defaults gain ``x_tweets``
      (Grok x_search download)
    - 2026-09-07: ``Agent`` → ``AgentAsync`` (jarvis) / ``AgentSync``
      (build/plan); no ``background`` argument
    - 2026-09-14 Build: exact post-drop managed defaults regain
      ``question`` (allow-all is permissions, not interviews)

    The per-agent MODEL route is OPERATOR-authoritative on the same footing:
    the scaffold ships no model key at all, so an Agent Builder
    model choice lives only in this file and must survive boot, together with
    the ``_MODEL_ROUTE_FIELDS`` derived from it. The model PARAMETERS (effort,
    temperature, sampling, thinking, small model) follow the same rule:
    a WebUI "Set default" lives only here too. See the preserve
    loop below.

    Only touches configs that explicitly opt in via ``agent_profile`` or
    ``profile_scaffold_version`` — never a custom agent whose directory
    name happens to collide with a bundled profile id.
    """
    home = Path(home)
    config_path = home / "config.json"
    canonical = _profile_config_dict(prof)
    try:
        current = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        current = {}
    if not isinstance(current, dict):
        current = {}
    if not _config_is_managed_profile(current, prof.name):
        return False
    # Operator-authoritative allowlist: the profile
    # SEEDS the tool/skill allowlist at install, but the operator owns it thereafter.
    # Preserve any operator-set allowlist key so this boot 'repair' never reverts an
    # operator edit — the repair still fixes structural drift + strips teacher/SI
    # bloat, but leaves the allowlist the operator chose (incl. a null == unrestricted
    # choice) intact. permission_mode stays profile-owned.
    #
    # The per-agent MODEL route joins that list (native `review`
    # agent). The scaffold deliberately carries NO model key — a profile must not
    # pin a model, so every deploy resolves its own from the global config — which
    # means the only way an agent gets a distinct model is the Agent Builder Model
    # dropdown writing ``upstream_model`` here. Because the canonical dict has no
    # model key at all, the ``set(current) - set(canonical)`` check below used to
    # flag every such config and then rewrite it verbatim from ``canonical``,
    # silently deleting the operator's choice at the next restart. Preserving the
    # whole route, not just the id, is deliberate: ``_MODEL_ROUTE_FIELDS`` are
    # materialized atomically as one consequence of selecting ``upstream_model``
    # (``agent_editor._model_route_patch``), so keeping the id while the provider
    # endpoint/alias/backend reverted to global would call the global endpoint
    # with a foreign model id.
    try:
        from sharedagents_agent.agent_editor import UNSET as _UNSET
    except Exception:  # noqa: BLE001 — sentinel literal is the same string
        _UNSET = "_UNSET"
    for _k in (
        "tools_allowlist",
        "tools_allowlist_strict",
        "skill_allowlist",
        "skill_allowlist_strict",
        "disabled_tools",
        # The per-agent delegation-mode pin is OPERATOR-authoritative (owner
        # ruling 2026-08-24): boot repair SEEDS it when absent (via the
        # canonical scaffold) but must never revert an operator-set value
        # (Agent Builder → Delegation mode).
        "delegation_mode",
        # --- operator-chosen model route + model parameters ---
        *OPERATOR_MODEL_FIELDS,
    ):
        if _k not in current:
            continue
        # ``_UNSET`` means "inherit global" and is normally POPPED by
        # ``agent_editor.patch_config`` before it ever reaches disk. If one is
        # found here (hand-edited config), honour the removal rather than
        # persisting the literal — ``AgentConfig.load`` would otherwise send
        # the string "_UNSET" to the wire as a model id.
        if current[_k] == _UNSET:
            continue
        canonical[_k] = current[_k]
    # Upgrade migration: a pre-lean-agent-deck config's tools_allowlist holds the
    # OLD TitleCase builtin ids (Read/Write/Bash/TodoWrite/...), which the renamed
    # registry no longer exposes — so an upgraded profile would silently lose those
    # tools. Translate legacy ids to canonical while preserving the operator's
    # chosen SET (order/dedup-preserving).
    _tal = canonical.get("tools_allowlist")
    if isinstance(_tal, list):
        from sharedagents_agent.tools.registry import canonicalize_tool_id

        _seen: set = set()
        _mig: list = []
        for _t in _tal:
            _c = canonicalize_tool_id(_t) if isinstance(_t, str) else _t
            if _c not in _seen:
                _seen.add(_c)
                _mig.append(_c)
        _mig = _migrate_exact_pre_alignment_plan_allowlist(_mig, prof)
        _mig = _migrate_exact_pre_opencode_explore_allowlist(_mig, prof)
        _mig = _migrate_build_x_tweets_allowlist(_mig, prof)
        _mig = _migrate_agent_sync_async_allowlist(_mig, prof)
        canonical["tools_allowlist"] = _migrate_build_restore_question(_mig, prof)
    # OpenCode hard denials (e.g. general denies todowrite) always union into
    # managed configs — operator can widen the blocklist, never drop the profile
    # floor.
    _profile_disabled = tuple(getattr(prof, "disabled_tools", ()) or ())
    if _profile_disabled:
        _existing_disabled = canonical.get("disabled_tools") or []
        if not isinstance(_existing_disabled, list):
            _existing_disabled = []
        from sharedagents_agent.tools.registry import canonicalize_tool_id

        _merged = {
            canonicalize_tool_id(t) if isinstance(t, str) else t
            for t in list(_existing_disabled) + list(_profile_disabled)
        }
        canonical["disabled_tools"] = sorted(_merged)
    needs_write = any(current.get(k) != v for k, v in canonical.items())
    needs_write = needs_write or "teacher" in current or "self_improver" in current
    needs_write = needs_write or set(current) - set(canonical)
    if not needs_write and config_path.is_file():
        return False
    home.mkdir(parents=True, exist_ok=True)
    write_config_text(config_path, json.dumps(canonical, indent=2, ensure_ascii=False) + "\n")
    return True


def ensure_builtin_profile_agents() -> list[str]:
    """Idempotently scaffold built-in profile agents on disk.

    Writes public profile homes while hidden runtime agents remain bundled.
    (config + ``AGENTS.md`` from shipped ``profile_agents/`` templates). The
    ``jarvis`` orchestrator is handled separately by :func:`ensure_voice_agent`.
    Never clobbers operator edits. Returns names of agents whose ``config.json``
    was newly created.
    """
    from sharedagents_agent.agent import AgentConfig, scaffold_context_files
    from sharedagents_agent.agent_profiles import list_profiles
    from sharedagents_core.layout import agents_root

    root = agents_root()
    created: list[str] = []
    repaired: list[str] = []

    for prof in list_profiles():
        if prof.name in ("jarvis",):
            # ``jarvis`` (the sole orchestrator identity) is scaffolded by
            # :func:`ensure_voice_agent` as the delegate-only concierge instance;
            # skipping here avoids a double-scaffold / "unmanaged config" warning.
            # Both dirs are still produced by the fresh
            # install — see :func:`ensure_all_roster_agents`.)
            continue
        # Hidden runtime profiles (compaction / title / summary / teacher) load bundled
        # templates directly — no public ``config.json`` (avoids gateway mount).
        if prof.opencode_mode == "hidden":
            continue
        home = root / prof.name
        config_path = home / "config.json"
        managed = False
        if config_path.is_file():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            managed = _config_is_managed_profile(existing, prof.name)
            if not managed:
                _LOG.warning(
                    "AGENT_SCAFFOLD_SKIP profile=%s home=%s (existing config is not a managed profile scaffold)",
                    prof.name,
                    home,
                )
        if not config_path.exists():
            home.mkdir(parents=True, exist_ok=True)
            write_config_text(config_path, json.dumps(_profile_config_dict(prof), indent=2) + "\n")
            created.append(prof.name)
            managed = True
            _LOG.info(
                "AGENT_SCAFFOLD_CREATED profile=%s home=%s",
                prof.name,
                home,
            )
        elif managed and sync_profile_agent_config(home, prof):
            repaired.append(prof.name)
            _LOG.info(
                "AGENT_CONFIG_REPAIRED profile=%s home=%s",
                prof.name,
                home,
            )

        if not managed:
            continue

        # The predefined config carries the per-agent delegation env default:
        # every managed profile home ships an explicit
        # SHAREDAGENTS_EXPERIMENTAL_BACKGROUND_SUBAGENTS (true for background-pinned
        # profiles — jarvis, scaffolded by ensure_voice_agent — false for the
        # foreground/unpoliced rest), so shipped homes are self-describing.
        # An operator assignment is never replaced.
        from sharedagents_agent.background_subagents import ensure_default_in_env_file

        if ensure_default_in_env_file(
            home,
            enabled=(prof.delegation_mode == "background"),
        ):
            _LOG.info(
                "AGENT_SCAFFOLD_WRITTEN profile=%s home=%s file=.env",
                prof.name,
                home,
            )

        from sharedagents_agent.agent_profiles import sync_profile_agents_md

        if sync_profile_agents_md(home, prof):
            _LOG.info(
                "AGENT_AGENTS_MD_REPAIRED profile=%s home=%s",
                prof.name,
                home,
            )
        else:
            from sharedagents_agent.agent_profiles import scaffold_profile_agents_md

            scaffold_profile_agents_md(home, prof)

        # Hidden profile agents (compaction / title / summary / teacher) are runtime-
        # invoked only — no MEMORY.md or orchestrator context scaffolding.
        if prof.opencode_mode != "hidden":
            try:
                cfg = AgentConfig.load(prof.name)
                scaffold_context_files(cfg, overwrite=False)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "AGENT_SCAFFOLD_CONTEXT profile=%s failed: %s",
                    prof.name,
                    exc,
                )

    if repaired:
        _LOG.info(
            "repaired OpenCode profile agent configs: %s",
            ", ".join(repaired),
        )

    return created


def ensure_voice_agent() -> list[str]:
    """Idempotently scaffold the ``jarvis`` ORCHESTRATOR agent.

    jarvis is the SDLC orchestrator identity and the
    user-facing DEFAULT agent that supersedes the internal ``main`` fallback.
    It is scaffolded with
    ``agent_profile="jarvis"`` so :func:`AgentConfig.load` resolves the JARVIS
    profile (the delegate-first orchestrator grant + durable memory; never edits
    files directly) and the picker (``opencode_mode=="primary"``) shows it.

    Writes ``~/.sharedagents/agents/jarvis/`` (config + the shipped orchestrator
    ``AGENTS.md`` template). A recognized stale managed template is upgraded;
    unmanaged or operator-edited files survive. Returns the list of file names
    actually written.

    DEFERRED/flagged: this supersedes the previous delegate-only *voice
    concierge* that used the jarvis name. The realtime-voice path still drives an
    agent named ``jarvis`` (it just resolves the orchestrator now); the tight
    "1-3 spoken sentences" persona (the historical ``_JARVIS_SOUL`` template,
    kept above for reference) should move to a dedicated concierge agent
    if still wanted.
    """
    from sharedagents_agent.agent_profiles import (
        JARVIS,
        scaffold_profile_agents_md,
        sync_profile_agents_md,
    )
    from sharedagents_core.layout import agents_root

    home = agents_root() / "jarvis"
    home.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _orchestrator_config() -> dict[str, Any]:
        # The orchestrator's DURABLE MEMORY + always-on HEARTBEAT are now intrinsic
        # to the JARVIS entry in _profile_config_dict, so the fresh-install scaffold
        # and a later lean profile save() agree (no divergence to clobber — Codex
        # HIGH, retire-main review). Kept as a named seam for the scaffold call.
        return _profile_config_dict(JARVIS)

    config_path = home / "config.json"
    managed = False
    if not config_path.exists():  # fresh install → scaffold orchestrator
        write_config_text(config_path, json.dumps(_orchestrator_config(), indent=2) + "\n")
        managed = True
        written.append("config.json")
        _LOG.info(
            "AGENT_SCAFFOLD_WRITTEN engine=local_llm home=%s file=config.json",
            home,
        )
    else:
        try:
            current = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        managed = _config_is_managed_profile(current, JARVIS.name)

    agents_md = home / "AGENTS.md"
    if not agents_md.exists():
        # Shipped orchestrator template (profile_agents/jarvis/AGENTS.md).
        scaffold_profile_agents_md(home, JARVIS)
        written.append("AGENTS.md")
        _LOG.info(
            "AGENT_SCAFFOLD_WRITTEN engine=local_llm home=%s file=AGENTS.md",
            home,
        )
    elif managed and sync_profile_agents_md(home, JARVIS):
        written.append("AGENTS.md")
        _LOG.info(
            "AGENT_AGENTS_MD_REPAIRED profile=jarvis home=%s",
            home,
        )
    # Jarvis is the non-blocking orchestrator.  Seed its per-agent experimental
    # default without replacing an operator's valid true/false choice or any
    # unrelated .env entries. Build and Plan receive no such seed and therefore
    # remain foreground by default.
    from sharedagents_agent.background_subagents import ensure_default_in_env_file

    if ensure_default_in_env_file(home, enabled=True):
        written.append(".env")
        _LOG.info(
            "AGENT_SCAFFOLD_WRITTEN engine=local_llm home=%s file=.env",
            home,
        )
    # Existing MANAGED jarvis homes (legacy) lack the delegation_mode
    # key. Insert the profile default without rewriting anything else and
    # never clobber an operator-set value (Agent Builder → Delegation mode) —
    # the runtime resolver treats the per-agent config as the operator
    # override. ``sync_profile_agent_config`` skips jarvis, so this is the
    # only boot path that migrates its config.
    if ensure_profile_delegation_mode(home, JARVIS):
        written.append("config.json")
        _LOG.info(
            "AGENT_CONFIG_REPAIRED profile=jarvis home=%s file=config.json",
            home,
        )
    return written


def ensure_all_roster_agents() -> list[str]:
    """Scaffold the lean roster so a fresh install ships the live local agents.

    A fresh install ships jarvis, build, plan, explore, and
    general — each with a ``config.json`` + ``AGENTS.md``. Hidden utilities
    (compaction/title/summary/teacher) load bundled templates without public homes.
    This is the single entrypoint that composes the scaffolders in the same
    order the gateway boot uses:

      * :func:`ensure_voice_agent` — ``jarvis`` (the sole orchestrator).
      * :func:`ensure_builtin_profile_agents` — build, plan, review, explore, general.

    Idempotent; never clobbers operator edits. Returns the union of newly
    created agent names.
    """
    created: list[str] = []
    if ensure_voice_agent():
        created.append("jarvis")
    created.extend(ensure_builtin_profile_agents())
    return created
