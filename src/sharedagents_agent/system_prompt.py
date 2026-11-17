"""System-prompt cache-boundary helpers (mirrors OpenClaw verbatim).

The system prompt is split into a STABLE prefix (cached aggressively
across turns and across sessions) and a DYNAMIC suffix (allowed to
change tick-to-tick: heartbeat content, time-of-day, ephemeral workspace
context). The split is marked by a single HTML-comment line that the
model ignores but sharedagents's own code uses to insert dynamic additions
in a cache-safe spot.

Same pattern as `dist/system-prompt-cache-boundary-CeY_ZCdW.js` and
`dist/prompt-cache-stability-D3r6Fd59.js` in the OpenClaw runtime —
function for function. The marker string is the only intentional
divergence: we use `SHAREDAGENTS_CACHE_BOUNDARY` so a prompt that round-trips
through both runtimes can be told apart and so logs make it clear
which agent owned the boundary.

Why this matters for vLLM APC / SGLang RadixAttention:
- The KV cache is hashed on a *block-aligned token prefix*. A single
  byte change near the start of the prompt invalidates the entire
  cache from there on, including the conversation history that follows.
- Putting "frequently-changing-but-still-per-session-stable" content
  (workspace cwd, frontend tag, …) BELOW the boundary keeps the long
  shared prefix byte-stable across sessions in different cwds — so the
  long head of every system prompt across your whole session library
  shares the same KV blocks on the upstream.
- Putting truly dynamic content (HEARTBEAT.md edits, current time,
  per-tick prompt additions) AFTER the boundary means *only the
  trailing few hundred tokens* re-prefill on a tick, not the whole
  10k-token system prompt.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sharedagents_agent.tools.registry import ToolRegistry


# Single canonical marker string. Length = 33 chars including the
# leading and trailing `\n`. The OpenClaw bundle uses a magic `34` in
# `splitSystemPromptCacheBoundary` for `boundaryIndex + 34`; that's
# the length of THEIR marker (`OPENCLAW_CACHE_BOUNDARY` is one char
# shorter than `SHAREDAGENTS_CACHE_BOUNDARY`). We compute the length here
# instead of hard-coding it so the marker can be changed without
# silently corrupting the split.
SYSTEM_PROMPT_CACHE_BOUNDARY = "\n<!-- SHAREDAGENTS_CACHE_BOUNDARY -->\n"
_BOUNDARY_LEN = len(SYSTEM_PROMPT_CACHE_BOUNDARY)


# Workflow guidance for the `todo` tool. Lives ABOVE the cache boundary
# because it never changes between turns — adding it here doesn't cost
# us prefix-cache stability across a session. Mirrors OpenCode's
# TodoWrite usage hint: short, action-oriented, no examples (the tool's
# own JSON schema is the authoritative spec).
# NOTE: TODO_TOOL_PROMPT_SECTION used to live here. Removed
# 2026-05-06 — operators wanted a single source of truth for the
# planning rule, so the imperative "if 2+ steps you MUST call todo"
# contract now lives in the per-agent ``AGENTS.md`` workspace file
# instead of being runtime-injected. ``_compose_system_prompt``
# composes the .md files; whatever the operator writes there is what
# the model sees.


# The prompt-pair parsers moved down into the kernel leaf so the memory
# synthesizer can import them without pulling in this module's heavy rendering
# helpers (which lazily import the tool registry + agent). Re-exported here by
# identity so ``from sharedagents_agent.system_prompt import load_prompt_pair`` and
# ``parse_prompt_pair_text`` keep working unchanged.
from sharedagents_core.kernel.prompt_pair import (  # noqa: E402,F401
    load_prompt_pair,
    parse_prompt_pair_text,
)

# CRLF normalization + trailing-whitespace cleanup. Matches OpenClaw's
# `normalizeStructuredPromptSection` byte for byte: the inference
# backend's tokenizer treats `\r\n` and `\n` differently, and a stray
# trailing space on a line that the model otherwise produces clean
# would break the cache hash on every round.
_CRLF_RE = re.compile(r"\r\n?")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def normalize_structured_prompt_section(text: str) -> str:
    """Mirror of `normalizeStructuredPromptSection`. Idempotent.

    1. `\\r\\n` / lone `\\r` → `\\n`.
    2. Strip trailing tabs/spaces from every line.
    3. Strip leading + trailing whitespace from the whole string.
    """
    if not isinstance(text, str):
        return ""
    out = _CRLF_RE.sub("\n", text)
    out = _TRAILING_WS_RE.sub("", out)
    return out.strip()


def strip_system_prompt_cache_boundary(text: str) -> str:
    """Replace every cache-boundary marker with a single `\\n`. Used
    for downstream consumers (logging, debug dumps, SSE replies) that
    want the system-prompt body without the sharedagents-internal split
    marker leaking into UIs."""
    return text.replace(SYSTEM_PROMPT_CACHE_BOUNDARY, "\n")


def split_system_prompt_cache_boundary(text: str) -> dict | None:
    """Split the prompt at the FIRST cache-boundary marker.

    Returns `{"stable_prefix": str, "dynamic_suffix": str}` (both
    leading/trailing whitespace stripped) or `None` if no marker is
    present. Mirrors OpenClaw's `splitSystemPromptCacheBoundary`.
    """
    idx = text.find(SYSTEM_PROMPT_CACHE_BOUNDARY)
    if idx == -1:
        return None
    return {
        "stable_prefix": text[:idx].rstrip(),
        "dynamic_suffix": text[idx + _BOUNDARY_LEN :].lstrip(),
    }


def prepend_system_prompt_addition_after_cache_boundary(
    *,
    system_prompt: str,
    system_prompt_addition: str | None,
) -> str:
    """Insert `system_prompt_addition` immediately after the cache
    boundary, ahead of any existing dynamic suffix. Mirrors OpenClaw's
    `prependSystemPromptAdditionAfterCacheBoundary`.

    - If the addition is empty, returns `system_prompt` unchanged.
    - If the system prompt has no boundary, the addition is prepended
      to the WHOLE prompt with `\\n\\n` separator (matches OpenClaw —
      a missing boundary means "treat this as legacy / fully dynamic").
    - If the dynamic suffix is empty, the addition becomes the suffix.
    - Otherwise: `<stable>\\n<boundary>\\n<addition>\\n\\n<dynamic>`.
    """
    addition = (
        normalize_structured_prompt_section(system_prompt_addition) if isinstance(system_prompt_addition, str) else ""
    )
    if not addition:
        return system_prompt
    split = split_system_prompt_cache_boundary(system_prompt)
    if split is None:
        return f"{addition}\n\n{system_prompt}"
    if not split["dynamic_suffix"]:
        return f"{split['stable_prefix']}{SYSTEM_PROMPT_CACHE_BOUNDARY}{addition}"
    dynamic = normalize_structured_prompt_section(split["dynamic_suffix"])
    return f"{split['stable_prefix']}{SYSTEM_PROMPT_CACHE_BOUNDARY}{addition}\n\n{dynamic}"


# Tooling / Tool Call Style / Execution Bias / Safety — mirrors OpenClaw
# 2026.6.8 ``buildAgentSystemPrompt`` stable-prefix layout. Tool-contract
# guidance lives WITH the tool registry, not in operator-editable .md files.
# AGENTS.md still owns behavioural style on top.


def build_tooling_prompt_section(
    registry: ToolRegistry,
    *,
    allowed_names: set[str] | None = None,
    deferred: dict[str, list[str]] | None = None,
) -> str:
    """Render the ``## Tooling`` section from the live tool registry.

    A flat, permission-filtered tool list. Every tool the agent
    may use is present with its full schema on the wire — there is no deferred /
    ``ToolSearch`` subsection. When *allowed_names* is set (all built-in profiles
    are strict), only those tools are listed — aligned with the worker
    ``tools[]`` payload.
    """
    from sharedagents_agent.tools.registry import ToolRegistry as _TR

    if not isinstance(registry, _TR):
        registry = _TR(registry.tools if hasattr(registry, "tools") else ())

    # Flat list of every allowed tool — the permission set IS the tool set —
    # except MCP servers offered on demand, which get one catalog line each.
    listed = allowed_names
    if deferred:
        on_demand = {n for tools in deferred.values() for n in tools}
        base = set(allowed_names) if allowed_names is not None else {t.name for t in registry.tools}
        listed = base - on_demand
    tool_lines = registry.core_tool_prompt_lines(allowed_names=listed) + registry.deferred_schemas_prompt_lines(
        allowed_names=listed
    )
    if deferred:
        from sharedagents_agent.mcp_deferral import LOAD_TOOLS, catalog_prompt_lines

        tool_lines.append(f"- {LOAD_TOOLS}: Load the tools of an MCP server listed below")
    tool_names = set(allowed_names) if allowed_names is not None else {t.name for t in registry.tools}
    has_todo = "todowrite" in tool_names
    has_agent = bool(tool_names & {"AgentSync", "AgentAsync"})

    lines: list[str] = [
        "## Tooling",
        "Available tools are policy-filtered. Names are case-sensitive; call exactly as listed.",
    ]
    if tool_lines:
        lines.extend(tool_lines)
    if deferred:
        lines.extend(catalog_prompt_lines(deferred))
    lines.append("AGENTS.md is usage guidance, not availability.")
    if "bash" in tool_names:
        if "Monitor" in tool_names:
            lines.extend(
                [
                    "For long waits, avoid rapid poll loops: use ``bash`` with "
                    "``run_in_background=true`` and poll with ``BashOutput``, or "
                    "``Monitor`` for push-based events.",
                ]
            )
        else:
            lines.extend(
                [
                    "For long waits, avoid rapid poll loops: use ``bash`` with "
                    "``run_in_background=true`` and poll with ``BashOutput``.",
                ]
            )
    elif has_agent:
        lines.append("Delegate only when a child can own a bounded unit under its effective profile.")
    if has_agent:
        lines.append(
            "Select parallel execution only for mutually independent, "
            "non-duplicative work with disjoint files and topics. Keep "
            "dependent or overlapping work, and repeated work against one "
            "child, sequential."
        )
        if "AgentAsync" in tool_names and "AgentSync" not in tool_names:
            lines.append(
                "Delegate with ``AgentAsync`` only. Every call detaches "
                "immediately and returns a handle; results are pushed later. "
                "There is no ``background`` argument. Do not poll."
            )
        elif "AgentSync" in tool_names and "AgentAsync" not in tool_names:
            lines.append(
                "Delegate with ``AgentSync`` only. Every call waits inline "
                "for the child or the whole ``runs`` batch. There is no "
                "``background`` argument."
            )
        else:
            lines.append(
                "Use ``AgentSync`` to wait inline or ``AgentAsync`` to detach. There is no ``background`` argument."
            )
        if "AgentAsync" in tool_names:
            lines.extend(
                [
                    "Before every fresh ``AgentAsync`` dispatch, call "
                    '``AgentAsync(action="list")`` and inspect the current '
                    "lineage. Use that list so you do not duplicate live "
                    "children or wait on dead ones.",
                    "Never ask the user a question (``question`` tool or a "
                    "question in a reply) while any listed child is still live. "
                    "Wait until every subagent has finished so you have the "
                    "full context.",
                ]
            )
        lines.append(
            "A child result is private input to its caller. Check relevance, "
            "integrate useful evidence, and present it yourself; never imply "
            "that the child reported directly to the user."
        )
    if has_todo:
        lines.extend(
            [
                "",
                "For work with three or more meaningful steps, use ``todowrite``: "
                "keep exactly one ``in_progress`` item, and mark an item complete "
                "only after supporting evidence exists.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def tooling_prompt_section() -> str:
    """Legacy entrypoint — builds Tooling from the module-frozen registry."""
    from sharedagents_agent.tools.registry import BUILTIN_TOOLS, ToolRegistry

    return build_tooling_prompt_section(ToolRegistry(BUILTIN_TOOLS))


def tool_call_style_prompt_section() -> str:
    """Render ``## Tool Call Style`` — OpenClaw 2026.6.8 ``tool_call_style``."""
    return "\n".join(
        [
            "## Tool Call Style",
            "Routine low-risk calls: no narration.",
            "Narrate only for complex, sensitive/destructive, or explicitly requested steps.",
            "First-class tool exists: use it; do not ask user to run equivalent CLI/slash command.",
            "Tool descriptions are the source of truth for arguments and return "
            "shape. Re-read the schema if a call fails — the fix is almost always "
            "in the tool's own description.",
            "Delegated subagents (via ``AgentSync`` / ``AgentAsync``) run a DIFFERENT "
            "toolset — describe tasks in plain terms; do not instruct them to use "
            "tools they do not have.",
            "",
        ]
    )


def execution_bias_prompt_section(profile_name: str | None = None) -> str:
    """Render ``## Execution Bias`` — a lightweight, OVERRIDABLE default.

    The orchestrators (``main``/``jarvis``) define their own execution bias in
    their editable ``AGENTS.md`` (thin-orchestrator behavior: delegation map, ask-only-on-
    ambiguity, minimal-faithful delegation, faithful relay) — the operator owns and
    tunes it there, so nothing is hardcoded for them here (return ``""``). Other
    profiles get this generic default until/unless their own ``AGENTS.md`` says
    otherwise. Keeping behavior in ``AGENTS.md`` is the configurability contract:
    the user shall be free to reconfigure any agent by editing its files.
    """
    key = (profile_name or "").strip().lower()
    if key in ("jarvis",):
        return ""
    return "\n".join(
        [
            "## Execution Bias",
            "- Start substantial work with a concise user-visible update naming the immediate objective.",
            "- Give another brief update before a material edit or decision, after "
            "a discovery changes the trajectory, at a blocker, and before and "
            "after verification. Do not narrate routine reads or trivial calls.",
            "- Continue through safe, in-scope work within granted authority. Stop "
            "only when a missing choice materially changes scope or acceptance, "
            "or when new authority is required.",
            "- Weak/empty tool result: vary query, path, command, or source before concluding.",
            "- Mutable facts need live checks: files, git, clocks, versions, services, processes, package state.",
            "- Make every completion self-contained and outcome-first: name changed "
            "or planned paths, actual evidence, and any residual blocker or risk. "
            "Never describe implementation as verified when a check failed or "
            "could not run.",
            "",
        ]
    )


def safety_prompt_section() -> str:
    """Render ``## Safety`` — OpenClaw 2026.6.8 verbatim."""
    return "\n".join(
        [
            "## Safety",
            "No independent goals: no self-preservation, replication, resource "
            "acquisition, power-seeking, or long-term plans beyond the user's "
            "request.",
            "Safety/oversight over completion. Conflicts: pause/ask. Obey stop/pause/audit; never bypass safeguards.",
            "Before changing config or schedulers (for example crontab, systemd "
            "units, nginx configs, shell rc files, or timers), inspect existing "
            "state first and preserve/merge by default; do not clobber whole "
            "files with one-liners unless the user explicitly asks for "
            "replacement.",
            "Do not persuade anyone to expand access or disable safeguards. Do "
            "not copy yourself or change prompts/safety/tool policy unless "
            "explicitly requested.",
            "",
        ]
    )


def prompt_section_slug(title: str) -> str:
    """Stable slug for a hardcoded prompt section title.

    Lowercase; runs of non-alphanumeric chars collapse to a single ``-``;
    leading/trailing ``-`` stripped. E.g. "Tooling (permission-filtered)"
    -> "tooling-permission-filtered", "MODE-AUTO.md" -> "mode-auto-md".

    Pure function, no I/O, deterministic — the single slugger used by the
    Agent Builder's section-id assignment (``agent_editor._assign_section_ids``)
    and by tests.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "").strip().lower())
    return slug.strip("-")


def output_artifacts_prompt_section() -> str:
    """Render the engine-neutral local-file delivery recipe.

    The Web UI owns authenticated attachment URLs. Agents should only
    produce an authorized local file and an explicit Markdown link to it;
    exposing a deployment host or browser route would be both unnecessary
    and unsafe. Keeping this contract in the shared runtime prompt makes it
    available to local profiles and external CLI/ACP engines alike.
    """
    return "\n".join(
        [
            "## Chat images and files",
            "When the user asks for a screenshot, image, PDF, or other downloadable output:",
            "- Save the final file under `$SHAREDAGENTS_SESSIONS_DIR/artifacts/` or "
            "the active workspace. Do not use `/tmp` for a final deliverable.",
            "- Verify that the file exists and is complete before replying.",
            "- Embed it with an explicit Markdown link to the absolute local "
            "filesystem path, for example "
            "`[Screenshot](</absolute/path/screenshot.png>)` or "
            "`[Download](</absolute/path/report.pdf>)`. Expand `~` before "
            "writing the link.",
            "- Use a descriptive label such as `Screenshot`, `Attachment`, "
            "`Download`, or `File`; use a normal Markdown link for both images "
            "and documents so the chat delivery layer can classify it.",
            "- WebUI attachment paths use `$SHAREDAGENTS_WEB_BASE_URL/files/<session-id>/<sha256>/<filename>`.",
            "- The sharedagents chat delivery layer converts approved local links into "
            "safe attachment cards; keep filesystem paths and attachment "
            "metadata out of public URLs.",
            "",
        ]
    )


# MCP / ACP capability hint. Lives in the stable section so it's
# cache-friendly. Surfaces the configured server / agent names along
# with a one-line description so the model picks the right name on the
# first try, instead of guessing and getting "unknown MCP server".
def mcp_acp_capabilities_block(
    global_cfg: dict | None = None,
    *,
    on_demand: set[str] | None = None,
) -> str:
    """Render an `# EXTERNAL TOOLS — MCP / ACP` section.

    Returns "" when nothing is configured. Reads the live config so a
    config edit is picked up next time the prompt composes.
    """
    if global_cfg is None:
        try:
            from sharedagents_agent.agent import load_global_config

            global_cfg = load_global_config()
        except Exception:  # noqa: BLE001
            global_cfg = {}
    # Canonical key is `mcpServers` (camelCase, what the wizard writes);
    # `mcp_servers` (snake_case) is the legacy fallback. Mirrors
    # `mcp_client.load_mcp_servers()` so the system-prompt MCP block and
    # the runtime see the same servers.
    _gcfg = global_cfg or {}
    mcp_raw = _gcfg.get("mcpServers") or _gcfg.get("mcp_servers") or {}
    acp_raw = _gcfg.get("acp_agents") or {}
    if not isinstance(mcp_raw, dict) or not isinstance(acp_raw, dict):
        return ""
    if not mcp_raw and not acp_raw:
        return ""
    lines: list[str] = ["# EXTERNAL TOOLS — MCP / ACP"]
    if mcp_raw:
        lines.append("")
        if on_demand:
            lines.append(
                "MCP server tools are called as `mcp__<server>__<tool>`. "
                "Servers marked (on demand) join your tool list after one "
                "`load_tools` call; the others are already in it. Servers "
                "configured:"
            )
        else:
            lines.append(
                "Each configured MCP server's tools are already in your tool "
                "list as `mcp__<server>__<tool>` (full schemas, directly "
                "callable — no discovery or wrapper call). Servers configured:"
            )
        lines.append("")
        for name, row in sorted(mcp_raw.items()):
            if not isinstance(row, dict):
                continue
            transport = row.get("transport") or ("stdio" if row.get("command") else "http")
            desc = str(row.get("description") or "").strip()
            tail = f" — {desc}" if desc else ""
            mark = " (on demand)" if on_demand and name in on_demand else ""
            lines.append(f"- `{name}` ({transport}){mark}{tail}")
    if acp_raw:
        lines.append("")
        lines.append(
            "ACP agents are reachable via `acp_call` — delegate a "
            "self-contained prompt to a remote agent that has its own "
            "tools and may run autonomously:"
        )
        lines.append("")
        for name, row in sorted(acp_raw.items()):
            if not isinstance(row, dict):
                continue
            desc = str(row.get("description") or "").strip()
            caps_raw = row.get("capabilities") or []
            caps = ", ".join(str(c) for c in caps_raw) if isinstance(caps_raw, list) else ""
            tail_bits = []
            if desc:
                tail_bits.append(desc)
            if caps:
                tail_bits.append(f"capabilities: {caps}")
            tail = " — " + " · ".join(tail_bits) if tail_bits else ""
            lines.append(f"- `{name}`{tail}")
    return "\n".join(lines)


def compose_with_cache_boundary(
    stable_sections: list[str],
    dynamic_sections: list[str] | None = None,
) -> str:
    """Convenience builder: glue `stable_sections` with `\\n\\n`,
    append the boundary, then glue `dynamic_sections` with `\\n\\n`.

    Empty/whitespace-only sections drop out so the output never has
    a hanging boundary with nothing dynamic on the other side.

    Cacheable-prefix stability: the bytes BEFORE the
    cache boundary (the *stable prefix*) are a deterministic function of
    ``stable_sections`` alone. Two turns with identical ``stable_sections``
    yield a byte-identical stable prefix regardless of the dynamic suffix,
    so an upstream KV / prefix cache stays warm across turns and is not
    gratuitously invalidated. Scoped to the cacheable prefix (KV-cache
    warmth), NOT whole-prompt byte-fidelity.
    """
    stable = "\n\n".join(s for s in stable_sections if (s and s.strip()))
    dynamic_sections = dynamic_sections or []
    dynamic = "\n\n".join(s for s in dynamic_sections if (s and s.strip()))
    if not dynamic:
        return stable
    if not stable:
        # No stable head; emit just the dynamic body. A boundary with
        # nothing on the stable side would be a footgun for downstream
        # `split_*` callers expecting a non-empty `stable_prefix`.
        return dynamic
    return f"{stable}{SYSTEM_PROMPT_CACHE_BOUNDARY}{dynamic}"
