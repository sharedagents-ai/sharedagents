"""System-prompt composition extracted from ``turn_runtime``.

Builds the per-session system prompt (injection-defense header, stable
cached prefix, dynamic MEMORY/HEARTBEAT sections) and the engine/profile
gates that decide which scaffolding an agent is shown. Every symbol is
re-exported from ``sharedagents_agent.turn_runtime``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from sharedagents_agent.context import (
    load_system_prompt_parts,
    strip_legacy_artifact_prompt_section,
)

if TYPE_CHECKING:
    from sharedagents_agent.agent import AgentConfig


def _engine_brings_own_tools(cfg: AgentConfig) -> bool:
    """True when the agent's engine ships its own native tool surface.

    CLI / ACP engines (``claude`` / ``codex`` / ``antigravity`` / ``acp``
    families, and their concrete ``engine_kind`` values) drive an
    external agent binary that brings its OWN built-in tools — Claude
    Code's Read/Write/Edit/Bash/Glob/Grep/Skill/Task/TodoWrite/
    WebSearch/WebFetch/ToolSearch, codex's, etc. sharedagents's local
    tool-registry scaffolding (the ``## Tooling`` deferred-ToolSearch /
    TodoWrite contract, the skills summary, the
    MCP/ACP capability list, the deferred-tool catalog) describes tools
    those binaries DON'T have. Injecting it makes the external model
    burn calls on ``No matching deferred tools`` and — with extended
    thinking on — the bogus ``ToolSearch`` dance corrupts the thinking-
    block sequence, yielding ``API Error: 400 ... thinking blocks cannot
    be modified`` which the CLI surfaces as ``error_during_execution``
    (root cause of sharedagents session <session-id>).

    Only the in-process ``local`` engine consumes sharedagents's registry,
    so only it gets the scaffolding. CLI/ACP engines receive the
    engine-neutral persona (.md) + Execution Bias and use their own
    native tools.
    """
    fam = (getattr(cfg, "engine", "") or "").strip().lower()
    kind = (getattr(cfg, "engine_kind", "") or "").strip().lower()
    if fam in {"claude", "codex", "antigravity", "acp", "cursor"}:
        return True
    from sharedagents_agent.context import EXTERNAL_ENGINE_KINDS

    # cfg.engine holds the EngineKind VALUE for external agents
    # ("claude_cli", "opencode_acp", …); ``engine_kind`` is a belt-and-
    # braces legacy attribute that nothing populates today. Either way
    # the shared constant is the single source of truth.
    return bool(fam in EXTERNAL_ENGINE_KINDS or kind in EXTERNAL_ENGINE_KINDS)


def _profile_system_prompt(cfg: AgentConfig) -> str | None:
    """Live ``AGENTS.md`` body for OpenCode profile agents (or None)."""
    try:
        from sharedagents_agent.agent_profiles import profile_system_prompt_for_config

        return profile_system_prompt_for_config(cfg)
    except Exception:  # noqa: BLE001
        cached = getattr(cfg, "_agent_profile_system", None)
        return str(cached).strip() or None if cached else None


def _model_family_and_id(cfg: AgentConfig) -> tuple[str, str]:
    """Resolve the active model's (family, id) for prompt composition.

    Shared by the runtime composer and the Agent Builder mirror so the
    TOOLCALL / persona resolution can never drift between them. The
    system prompt is sticky per session, so resolving from cfg's primary
    model id is the right (and only) signal at compose time.
    """
    model_family = ""
    try:
        from sharedagents_model import models as _models

        model_id = (getattr(cfg, "upstream_model", "") or "").strip()
        if not model_id:
            try:
                model_id = str((cfg._primary_model_row() or {}).get("id") or "")
            except Exception:  # noqa: BLE001
                model_id = ""
        if model_id:
            model_family = _models.for_name(model_id).family
    except Exception:  # noqa: BLE001
        model_family = ""
        model_id = ""
    return model_family, model_id


def _on_demand_mcp_servers(cfg: AgentConfig) -> dict[str, list[str]]:
    """MCP servers this agent loads on demand — computed from the same
    tool set the wire starts from: the permission set (full access = every
    discovered MCP tool), narrowed to the Web user's turn grants."""
    try:
        from sharedagents_agent.agent_profiles import compose_tool_allowlist
        from sharedagents_agent.mcp_deferral import deferred_servers
        from sharedagents_agent.tools.registry import discovered_mcp_tools_sync
        from sharedagents_core.access_grants import filter_tool_names
        from sharedagents_session.access_binding import turn_grants

        allowed = compose_tool_allowlist(cfg)
        if allowed is None:
            allowed = {t.name for t in discovered_mcp_tools_sync()}
        _turn = turn_grants()
        if _turn is not None:
            allowed = filter_tool_names(allowed, _turn)
        return deferred_servers(cfg, set(allowed))
    except Exception:  # noqa: BLE001 - the prompt never fails over the catalog
        return {}


def head_prompt_parts(cfg: AgentConfig) -> list[dict]:
    """The hardcoded stable-head sections, in exact compose order.

    Returns ``{"source", "title", "text"}`` parts (source is always
    ``"hardcoded"``): Prompt-injection defense → Tooling / Tool Call
    Style / Execution Bias / Safety (or just Execution Bias + Safety for
    engines that bring their own tools) → Chat images and files.

    Shared by ``_compose_system_prompt`` (which joins them) and the
    Agent Builder's ``render_system_prompt`` mirror (which annotates
    them) so the two can never drift on the head.
    """
    from sharedagents_agent.agent_profiles import (
        compose_tool_allowlist,
        resolve_profile_for_config,
    )
    from sharedagents_agent.system_prompt import (
        build_tooling_prompt_section,
        execution_bias_prompt_section,
        output_artifacts_prompt_section,
        safety_prompt_section,
        tool_call_style_prompt_section,
    )
    from sharedagents_agent.tools import BUILTIN_TOOLS, ToolRegistry
    from sharedagents_agent.tools.registry import (
        discovered_mcp_tools_sync,
        rewrite_agent_tools_in_allowlist,
    )
    from sharedagents_core.injection_guard import injection_defense_prompt_section

    own_tools = _engine_brings_own_tools(cfg)
    registry = ToolRegistry(BUILTIN_TOOLS + discovered_mcp_tools_sync())
    prof = resolve_profile_for_config(cfg)
    profile_name = (
        prof.name if prof is not None else str(getattr(cfg, "agent_profile", "") or getattr(cfg, "name", "") or "")
    )
    compose_allowed = compose_tool_allowlist(cfg)
    if compose_allowed is None:
        compose_allowed = rewrite_agent_tools_in_allowlist(
            {t.name for t in BUILTIN_TOOLS},
            getattr(cfg, "delegation_mode", None),
        )
    else:
        compose_allowed = rewrite_agent_tools_in_allowlist(
            set(compose_allowed),
            getattr(cfg, "delegation_mode", None),
        )
    # A Web user's turn advertises only the tools granted to them — the same
    # narrowing the wire and the dispatch gate apply.
    from sharedagents_core.access_grants import filter_tool_names
    from sharedagents_session.access_binding import turn_grants

    _turn = turn_grants()
    if _turn is not None:
        compose_allowed = filter_tool_names(compose_allowed, _turn)
    deferred = _on_demand_mcp_servers(cfg)

    parts: list[dict] = [
        {"source": "hardcoded", "title": "Prompt-injection defense", "text": injection_defense_prompt_section()},
    ]
    if own_tools:
        parts.append(
            {"source": "hardcoded", "title": "Execution Bias", "text": execution_bias_prompt_section(profile_name)}
        )
        parts.append({"source": "hardcoded", "title": "Safety", "text": safety_prompt_section()})
    else:
        parts.append(
            {
                "source": "hardcoded",
                "title": "Tooling (permission-filtered)",
                "text": build_tooling_prompt_section(registry, allowed_names=compose_allowed, deferred=deferred),
            }
        )
        if _turn is not None and _turn.folders is not None:
            parts.append(
                {
                    "source": "hardcoded",
                    "title": "Filesystem (this user)",
                    "text": _filesystem_prompt_section(_turn.folders),
                }
            )
        parts.append({"source": "hardcoded", "title": "Tool Call Style", "text": tool_call_style_prompt_section()})
        parts.append(
            {"source": "hardcoded", "title": "Execution Bias", "text": execution_bias_prompt_section(profile_name)}
        )
        parts.append({"source": "hardcoded", "title": "Safety", "text": safety_prompt_section()})
    parts.append({"source": "hardcoded", "title": "Chat images and files", "text": output_artifacts_prompt_section()})
    return parts


def _filesystem_prompt_section(folders) -> str:
    """What a folder-confined turn's shell and file tools can see."""
    if not folders:
        return (
            "Your shell and file tools run in a sandbox with no folders of this "
            "machine: they see only a private scratch directory that is discarded "
            "after each call. Say so when a task needs files you cannot reach."
        )
    lines = [f"- `{f.path}` ({'read and write' if f.mode == 'rw' else 'read only'})" for f in folders]
    return (
        "Your shell and file tools run in a sandbox that shows only these folders "
        "of this machine; everything else does not exist for them. Work inside "
        "them and use absolute paths:\n" + "\n".join(lines)
    )


def _narrow_to_turn_grants(allow, strict):
    """Intersect an agent's skill allowlist with the running turn's grants.

    The skills catalogue is composed without a session id, so it reads the
    grants the turn bound (``access_binding.turn_grants``).
    Same encoding as ``apply_skill_allowlist``: ``None`` or ``[]`` non-strict
    = all, a list = only those, ``[]`` strict = none.
    """
    from sharedagents_session.access_binding import turn_grants

    grants = turn_grants()
    if grants is None or grants.is_all("skills"):
        return allow, strict
    granted = set(grants.names("skills") or ())
    if allow is None or (not allow and not strict):
        narrowed = granted
    else:
        narrowed = {str(n).strip() for n in allow if str(n).strip()} & granted
    return sorted(narrowed), True


def tail_prompt_parts(cfg: AgentConfig) -> list[dict]:
    """The hardcoded tail blocks (Skills, MCP/ACP) appended AFTER the .md
    stack in the stable prefix — shared by the runtime composer and the
    Agent Builder mirror.

    NOTE: like the composer this has the side effect of binding the
    active skill allowlist ContextVar for the current turn.
    """
    from sharedagents_agent.agent_profiles import resolve_profile_for_config
    from sharedagents_agent.system_prompt import mcp_acp_capabilities_block
    from sharedagents_skill import skills as _skills_mod

    own_tools = _engine_brings_own_tools(cfg)
    prof = resolve_profile_for_config(cfg)
    profile_name = (
        prof.name if prof is not None else str(getattr(cfg, "agent_profile", "") or getattr(cfg, "name", "") or "")
    )
    # FAIL CLOSED: a cfg that can't be read denies all skills rather than
    # falling back to unrestricted (a permission boundary must not open
    # on error).
    try:
        _sk_allow = getattr(cfg, "skill_allowlist", None)
        _sk_strict = bool(getattr(cfg, "skill_allowlist_strict", False))
    except Exception:  # noqa: BLE001
        _sk_allow, _sk_strict = [], True  # deny-all
    _sk_allow, _sk_strict = _narrow_to_turn_grants(_sk_allow, _sk_strict)
    _skills_mod.set_active_skill_allowlist(_sk_allow, _sk_strict)
    # Only advertise skills to an agent that can actually load one. A thin
    # orchestrator without the ``Skill`` tool gets no Skills block, since one that
    # says "Skill to load" is a dead instruction.
    _wl = getattr(cfg, "tools_allowlist", None)
    _has_skill_tool = (_wl is None) or ("Skill" in _wl)
    from sharedagents_session.access_binding import turn_grants as _turn_grants

    _tg = _turn_grants()
    if _tg is not None and not _tg.allows("tools", "Skill"):
        _has_skill_tool = False  # this user was not granted the Skill tool
    skills_block = (
        ""
        if (own_tools or not _has_skill_tool)
        else (_skills_mod.skills_summary_block(cfg.home, allowlist=_sk_allow, allowlist_strict=_sk_strict))
    )
    parts: list[dict] = []
    if skills_block:
        parts.append({"source": "hardcoded", "title": "Skills (discovered)", "text": skills_block})
    # Orchestrator agents cannot call MCP tools — omit the capability
    # block so the model is not nudged toward ``mcp_call`` it cannot invoke.
    skip_mcp = profile_name in ("jarvis",)
    # A Web user without any MCP/ACP tool grant is not told about them.
    if (
        _tg is not None
        and not _tg.is_all("tools")
        and not any(n.startswith("mcp__") or n == "acp_call" for n in (_tg.names("tools") or ()))
    ):
        skip_mcp = True
    mcp_acp_block = (
        "" if own_tools or skip_mcp else mcp_acp_capabilities_block(on_demand=set(_on_demand_mcp_servers(cfg)))
    )
    if mcp_acp_block:
        parts.append({"source": "hardcoded", "title": "External tools (MCP / ACP)", "text": mcp_acp_block})
    return parts


def dynamic_prompt_parts(
    cfg: AgentConfig,
    *,
    workspace_cwd: str = "",
    workspace_origin: str = "",
    skip_memory: bool,
) -> list[dict]:
    """The below-cache-boundary sections (WORKSPACE-CONTEXT, MEMORY.md,
    HEARTBEAT.md) — shared by the runtime composer and the Agent Builder
    mirror (which passes empty cwd/origin).
    """
    parts: list[dict] = []
    # ``source folder`` is the code/scripts root, injected verbatim from the
    # ``SOURCE_FOLDER`` env var when the environment sets one.
    # ``workspace_origin`` is deliberately NOT rendered (no useful signal;
    # small models misread a bare ``frontend: acp`` as a codebase).
    _source_folder = os.environ.get("SOURCE_FOLDER", "").strip()
    if workspace_cwd or _source_folder:
        bits = ["# WORKSPACE-CONTEXT"]
        if workspace_cwd:
            bits.append(f"cwd: {workspace_cwd}")
        if _source_folder:
            bits.append(f"source folder: {_source_folder}")
        parts.append({"source": "dynamic", "title": "WORKSPACE-CONTEXT", "text": "\n".join(bits)})
    # MEMORY.md goes BELOW the cache boundary in every session — same
    # OpenClaw pattern. Skipped when ``memory_search_enabled`` is false
    # (profile scaffolds set this per agent).
    if not skip_memory:
        memory_path = cfg.home / "MEMORY.md"
        if memory_path.exists():
            try:
                mem_body = memory_path.read_text(encoding="utf-8").strip()
            except OSError:
                mem_body = ""
            mem_body = strip_legacy_artifact_prompt_section(mem_body)
            if mem_body:
                parts.append({"source": "dynamic", "title": "MEMORY.md", "text": f"# MEMORY.md\n\n{mem_body}"})
    # HEARTBEAT.md is injected ONLY into the dedicated heartbeat session
    # (workspace_origin="heartbeat") — mirrors OpenClaw's
    # `runKind == "heartbeat"` filter.
    if workspace_origin == "heartbeat":
        hb_path = cfg.home / "HEARTBEAT.md"
        if hb_path.exists():
            try:
                hb_body = hb_path.read_text(encoding="utf-8").strip()
            except OSError:
                hb_body = ""
            if hb_body:
                parts.append({"source": "dynamic", "title": "HEARTBEAT.md", "text": f"# HEARTBEAT.md\n\n{hb_body}"})
    return parts


def filter_disabled_prompt_parts(parts: list[dict], disabled: set[str]) -> list[dict]:
    """Drop hardcoded sections whose id is in *disabled*.

    Only ``source == "hardcoded"`` parts are filterable — agents_md /
    system / dynamic composition is never touched.
    """
    if not disabled:
        return parts
    return [p for p in parts if p.get("source") != "hardcoded" or p.get("id") not in disabled]


def assign_prompt_part_ids(parts: list[dict]) -> None:
    """Give every part a stable ``id`` (in place).

    The id is a slug of the section title (``prompt_section_slug``); when
    two sections produce the same slug, the first keeps the bare slug and
    later duplicates get ``-2``, ``-3``, … in render order.
    """
    from sharedagents_agent.system_prompt import prompt_section_slug

    seen: dict[str, int] = {}
    for p in parts:
        base = prompt_section_slug(str(p.get("title") or "")) or "section"
        n = seen.get(base, 0) + 1
        seen[base] = n
        p["id"] = base if n == 1 else f"{base}-{n}"


def _compose_system_prompt(
    cfg: AgentConfig,
    mode: str,
    *,
    workspace_cwd: str = "",
    workspace_origin: str = "",
    skip_memory: bool = False,
) -> str:
    """Assemble the full system prompt — workspace .md files + skills
    + per-session workspace context — with the sharedagents cache boundary
    marker between the stable head and the dynamic tail.

    This is the single source of truth for what goes into a session's
    system message. It's called:

    - Once at session creation (`create_session`), persisted onto the
      session record's `system_prompt` field.
    - Once per turn from `_build_messages` IFF the cached prompt is
      missing or the active mode changed since it was cached.
    - Once per compaction event (`_reload_system_prompt_in_messages`)
      so a fresh, up-to-date prompt lands at messages[0] when the
      conversation context is reset.

    Layout:
        <WORKSPACE.md>...<MODE-*.md><skills>
        <SHAREDAGENTS_CACHE_BOUNDARY>
        <# WORKSPACE-CONTEXT>

    Why split: vLLM APC / SGLang RadixAttention reuse KV cache by
    block-aligned token prefix. Putting the per-session-specific bits
    (cwd, origin) AFTER the boundary means two sessions in different
    cwds share the long stable head — only the last few lines diverge.
    The boundary itself is an HTML comment the model ignores; sharedagents's
    own code (`system_prompt.split_system_prompt_cache_boundary`) uses
    it to insert dynamic additions in a cache-safe spot when needed
    (heartbeat content, time-of-day, …).

    Byte-stable for a given (.md files on disk, mode, workspace_cwd,
    workspace_origin, disabled_prompt_sections) tuple — the prefix
    cache holds across turns of a session as long as none of those
    inputs changes.

    Hardcoded sections named in ``cfg.disabled_prompt_sections`` (the
    Agent Builder's enable/disable cards) are omitted wherever
    they appear — head, md stack (WORKSPACE/PERSONA/TOOLCALL/MODE), or
    tail; every other part of the composition is byte-identical to the
    pre-filter output.
    """
    from sharedagents_agent.system_prompt import compose_with_cache_boundary

    sections = composed_prompt_sections(
        cfg,
        mode,
        workspace_cwd=workspace_cwd,
        workspace_origin=workspace_origin,
        skip_memory=skip_memory,
    )
    return compose_with_cache_boundary(
        [p["text"] for p in sections if p.get("source") != "dynamic"],
        [p["text"] for p in sections if p.get("source") == "dynamic"],
    )


def composed_prompt_sections(
    cfg: AgentConfig,
    mode: str,
    *,
    workspace_cwd: str = "",
    workspace_origin: str = "",
    skip_memory: bool = False,
    record_breakdown: bool = True,
) -> list[dict]:
    """The sections ``_compose_system_prompt`` joins, in join order.

    Each entry is the builder dict (``title``, ``text``, ``source``).
    Empty sections are already dropped, matching the joined string.
    ``record_breakdown=False`` keeps a context-sheet read from writing
    the runtime ``/context`` report.
    """
    from sharedagents_agent.subagent_policy import profile_skips_memory

    if not skip_memory:
        skip_memory = profile_skips_memory(cfg)
    disabled = {str(s).strip() for s in (getattr(cfg, "disabled_prompt_sections", None) or []) if str(s).strip()}
    model_family, model_id = _model_family_and_id(cfg)
    head_parts = head_prompt_parts(cfg)
    md_parts = load_system_prompt_parts(
        cfg.system,
        root=cfg.home,
        mode=mode,
        model_family=model_family,
        model_id=model_id,
        profile_system=_profile_system_prompt(cfg),
        agent_name=cfg.name,
        engine_kind=str(getattr(cfg, "engine", "") or ""),
        record_breakdown=record_breakdown,
    )
    tail_parts = tail_prompt_parts(cfg)
    dynamic_parts = dynamic_prompt_parts(
        cfg, workspace_cwd=workspace_cwd, workspace_origin=workspace_origin, skip_memory=skip_memory
    )
    assign_prompt_part_ids(head_parts + md_parts + tail_parts + dynamic_parts)
    live: list[dict] = []
    for p in filter_disabled_prompt_parts(head_parts, disabled):
        if p["text"] and str(p["text"]).strip():
            live.append(p)
    # The md stack carries its OWN hardcoded sections (WORKSPACE.md /
    # PERSONA-*.md / TOOLCALL*.md / MODE-*.md) — filter them too, or a
    # disabled md section would leak back into the composed prompt.
    for p in filter_disabled_prompt_parts(md_parts, disabled):
        if p["text"] and str(p["text"]).strip():
            live.append(p)
    for p in filter_disabled_prompt_parts(tail_parts, disabled):
        if p["text"] and str(p["text"]).strip():
            live.append(p)
    for p in dynamic_parts:
        if p["text"] and str(p["text"]).strip():
            live.append(p)
    return live
