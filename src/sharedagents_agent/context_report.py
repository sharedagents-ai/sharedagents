"""Context-window breakdown report — sharedagents port of OpenClaw's
``/context`` command.

OpenClaw exposes ``/context``, ``/context list``, ``/context detail``,
and ``/context json``. This module produces the same data structure
(``ContextReport``) and renders the same four output flavors so
operators can read prompt bloat the same way on either runtime.

The report measures:

- **System prompt** total chars + estimated tokens, split into a stable
  prefix (cached across turns) and a "Project Context" tail (the
  dynamic suffix below the cache boundary — workspace cwd + MEMORY.md).
- **Injected workspace files**: per-file char counts for WORKSPACE.md /
  SOUL.md / USER.md / TOOLS.md / TOOLCALL.md / AGENTS.md / MODE-*.md /
  MEMORY.md, with status (OK / TRUNCATED / MISSING).
- **Skills**: total prompt-text size + per-skill block size.
- **Tools**: ``tool list`` text in the system prompt (deferred-catalog
  block) + JSON schema size for ``tools[]`` (the wire payload that
  doesn't surface as text but still costs context tokens).
- **Session tokens**: cached prompt-token total from the latest turn.

The token estimator is the same 1 token ≈ 4 chars heuristic OpenClaw
uses (``estimateTokensFromChars``) so a side-by-side comparison
matches numerically when the same content is loaded.

The separately maintained, more accurate per-model token estimator is
never substituted into this path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# OpenClaw's heuristic — keep IDENTICAL so operators reading the same
# system prompt under both runtimes see the same numbers.
_CHARS_PER_TOKEN = 4


def estimate_tokens_from_chars(chars: int) -> int:
    if not chars or chars <= 0:
        return 0
    return chars // _CHARS_PER_TOKEN


def _format_int(n: int) -> str:
    return f"{int(n):,}"


# ---- per-model token estimator (engine-metrics only) -----------------
#
# NOT the /context parity path — that must remain estimate_tokens_from_chars
# with the flat 4-char ratio. This helper is used by the engine layers
# (ExternalCliEngine, AcpEngine) to produce better completion-token
# estimates for the metrics footer. The ≈ suffix on the rendered row
# already signals "fuzzy estimate" to the operator.

_MODEL_CHARS_PER_TOKEN: dict[str, float] = {
    # Ordered longest-first so partial matches prefer specific prefixes.
    "claude": 3.5,
    "opus": 3.5,
    "sonnet": 3.5,
    "haiku": 3.5,
    "gemini": 4.0,
    "gpt": 4.0,
    "cursor": 4.0,
    "composer": 4.0,
}


def estimate_tokens_for_model(chars: int, model: str = "") -> int:
    """Estimate tokens from character count using a per-model ratio.

    Engine-metrics only — the /context parity path must NOT call this.
    Returns 0 for non-positive char counts (same contract as
    ``estimate_tokens_from_chars``).
    """
    if not chars or chars <= 0:
        return 0
    ratio = 4.0  # default fallback
    model_lower = (model or "").lower()
    for key, r in _MODEL_CHARS_PER_TOKEN.items():
        if key in model_lower:
            ratio = r
            break
    return max(1, int(chars / ratio))


def _format_chars_and_tokens(chars: int) -> str:
    return f"{_format_int(chars)} chars (~{_format_int(estimate_tokens_from_chars(chars))} tok)"


@dataclass
class WorkspaceFileEntry:
    name: str
    raw_chars: int = 0
    injected_chars: int = 0
    missing: bool = False
    truncated: bool = False


@dataclass
class SkillEntry:
    name: str
    block_chars: int = 0


@dataclass
class ToolEntry:
    name: str
    schema_chars: int = 0
    summary_chars: int = 0
    properties_count: int | None = None


@dataclass
class ContextReport:
    source: str = "run"  # "run" if read from a live session, "estimate" otherwise
    model: str = ""
    workspace_dir: str = ""
    bootstrap_max_chars: int = 0
    bootstrap_total_max_chars: int = 0
    sandbox_mode: str = ""
    sandbox_sandboxed: bool = False
    system_prompt_chars: int = 0
    project_context_chars: int = 0  # the dynamic suffix (workspace ctx + MEMORY.md)
    injected_workspace_files: list[WorkspaceFileEntry] = field(default_factory=list)
    skills_prompt_chars: int = 0
    skills_entries: list[SkillEntry] = field(default_factory=list)
    tool_list_chars: int = 0  # the deferred-catalog block in the system prompt
    tool_schema_chars: int = 0  # full JSON schemas sent in tools[]
    tool_entries: list[ToolEntry] = field(default_factory=list)


@dataclass
class SessionTokens:
    total_tokens: int | None = None
    total_tokens_fresh: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    context_tokens: int | None = None  # the model's max context window


# Workspace-context tail tags — used to extract the "Project Context"
# slice (everything below the cache boundary). Mirrors what
# `_compose_system_prompt` writes.
_PROJECT_CONTEXT_HEADERS = ("# WORKSPACE-CONTEXT", "# MEMORY.md", "# HEARTBEAT.md")


def _extract_project_context_chars(system_prompt: str) -> int:
    """Return char count of the dynamic suffix (below the cache boundary).

    Best-effort: we look for the cache boundary marker first; if it's
    missing (legacy session, mock prompt), we fall back to scanning
    for the WORKSPACE-CONTEXT / MEMORY.md headers and take everything
    from the earliest header forward.
    """
    from sharedagents_agent.system_prompt import split_system_prompt_cache_boundary

    split = split_system_prompt_cache_boundary(system_prompt)
    if split is not None:
        return len(split["dynamic_suffix"])
    earliest = -1
    for header in _PROJECT_CONTEXT_HEADERS:
        idx = system_prompt.find(header)
        if idx == -1:
            continue
        if earliest == -1 or idx < earliest:
            earliest = idx
    if earliest == -1:
        return 0
    return len(system_prompt) - earliest


def _read_md_file(path: Path) -> tuple[int, int, bool, bool]:
    """Read ``path`` and return ``(raw_chars, injected_chars, missing, truncated)``.

    ``injected_chars`` matches what ``load_system_prompt`` writes into
    the prompt: clamped at ``CONTEXT_FILE_MAX_BYTES`` with a
    truncation marker appended. We don't know the per-file budget the
    composer used — pass it in via the report builder.
    """
    from sharedagents_agent.context import CONTEXT_FILE_MAX_BYTES

    if not path.is_file():
        return (0, 0, True, False)
    try:
        blob = path.read_bytes()
    except OSError:
        return (0, 0, True, False)
    raw = len(blob)
    if raw > CONTEXT_FILE_MAX_BYTES:
        injected = CONTEXT_FILE_MAX_BYTES + 120  # truncation marker length is ~120
        return (raw, injected, False, True)
    return (raw, raw, False, False)


def build_context_report(
    *,
    agent_home: Path,
    mode: str,
    system_prompt: str,
    tools_schemas: list[dict[str, Any]],
    tool_summaries: list[tuple[str, str]] | None = None,
    deferred_catalog_text: str = "",
    skills_block_text: str = "",
    skills: list[Any] | None = None,
    model: str = "",
    workspace_cwd: str = "",
    permission_mode: str = "",
    source: str = "run",
) -> ContextReport:
    """Assemble a ``ContextReport`` from the running session's state.

    Caller is responsible for passing in the canonical system prompt
    (the cached string the model actually saw) and the live tool
    schemas + deferred-catalog text. The builder doesn't recompose the
    prompt — it only measures what's already there.
    """
    from sharedagents_agent.agent import hardcoded_context_body
    from sharedagents_agent.context import (
        CONTEXT_FILE_MAX_BYTES,
        CONTEXT_FILES,
        CONTEXT_TOTAL_MAX_BYTES,
        DEFAULT_TOOLCALL_FILE,
        HARDCODED_CONTEXT_FILES,
        MEMORY_FILE,
        MODE_FILES,
        toolcall_filename_for_family,
    )

    # Per-file sizes AS COMPOSED into ``system_prompt`` (recorded by
    # ``load_system_prompt``), so /context reports what the model actually saw
    # rather than a fresh, post-edit disk read.
    try:
        from sharedagents_agent.context import composition_breakdown as _comp_bd

        _composed = _comp_bd(system_prompt)
    except Exception:  # noqa: BLE001
        _composed = None

    files: list[WorkspaceFileEntry] = []
    for fname in HARDCODED_CONTEXT_FILES:
        body = (
            hardcoded_context_body(
                fname,
                agent_name=agent_home.name,
                agent_home=agent_home,
            )
            or ""
        )
        files.append(
            WorkspaceFileEntry(
                name=fname,
                raw_chars=len(body),
                injected_chars=len(body),
                missing=False,
                truncated=False,
            )
        )
    toolcall_file = DEFAULT_TOOLCALL_FILE
    model_family = ""
    if model:
        try:
            from sharedagents_model import models as _models

            model_family = _models.for_name(model).family
        except Exception:  # noqa: BLE001
            model_family = ""
    toolcall_file = toolcall_filename_for_family(model_family, agent_home)
    tc_body = hardcoded_context_body(toolcall_file) or ""
    files.append(
        WorkspaceFileEntry(
            name=toolcall_file,
            raw_chars=len(tc_body),
            injected_chars=len(tc_body),
            missing=False,
            truncated=False,
        )
    )
    for fname in CONTEXT_FILES:
        if _composed is not None and fname in _composed:
            raw, inj, missing, trunc = _composed[fname]
        else:
            raw, inj, missing, trunc = _read_md_file(agent_home / fname)
        files.append(
            WorkspaceFileEntry(
                name=fname,
                raw_chars=raw,
                injected_chars=inj,
                missing=missing,
                truncated=trunc,
            )
        )
    mode_file = MODE_FILES.get(mode or "")
    if mode_file:
        body = hardcoded_context_body(mode_file) or ""
        files.append(
            WorkspaceFileEntry(
                name=mode_file,
                raw_chars=len(body),
                injected_chars=len(body),
                missing=False,
                truncated=False,
            )
        )
    raw, inj, missing, trunc = _read_md_file(agent_home / MEMORY_FILE)
    if not missing:
        files.append(
            WorkspaceFileEntry(
                name=MEMORY_FILE,
                raw_chars=raw,
                injected_chars=inj,
                missing=missing,
                truncated=trunc,
            )
        )

    skill_entries: list[SkillEntry] = []
    for s in skills or []:
        body = getattr(s, "body", "") or ""
        desc = getattr(s, "description", "") or ""
        skill_entries.append(
            SkillEntry(
                name=getattr(s, "name", "?"),
                block_chars=len(body) + len(desc),
            )
        )

    tool_entries: list[ToolEntry] = []
    schema_chars_total = 0
    summary_lookup: dict[str, str] = dict(tool_summaries or [])
    for ts in tools_schemas:
        try:
            name = (ts.get("function") or {}).get("name", "?")
            params = (ts.get("function") or {}).get("parameters") or {}
            props = params.get("properties") if isinstance(params, dict) else None
            props_count = len(props) if isinstance(props, dict) else None
            schema_chars = len(json.dumps(ts, separators=(",", ":")))
        except Exception:  # noqa: BLE001
            name = "?"
            props_count = None
            schema_chars = 0
        schema_chars_total += schema_chars
        summary = summary_lookup.get(name, "")
        tool_entries.append(
            ToolEntry(
                name=name,
                schema_chars=schema_chars,
                summary_chars=len(summary),
                properties_count=props_count,
            )
        )

    return ContextReport(
        source=source,
        model=model,
        workspace_dir=str(agent_home),
        bootstrap_max_chars=CONTEXT_FILE_MAX_BYTES,
        bootstrap_total_max_chars=CONTEXT_TOTAL_MAX_BYTES,
        sandbox_mode=permission_mode or "",
        sandbox_sandboxed=bool(permission_mode and permission_mode != "auto"),
        system_prompt_chars=len(system_prompt),
        project_context_chars=_extract_project_context_chars(system_prompt),
        injected_workspace_files=files,
        skills_prompt_chars=len(skills_block_text),
        skills_entries=skill_entries,
        tool_list_chars=len(deferred_catalog_text),
        tool_schema_chars=schema_chars_total,
        tool_entries=tool_entries,
    )


# --- rendering --------------------------------------------------------------


def _format_name_list(names: list[str], cap: int) -> str:
    if len(names) <= cap:
        return ", ".join(names)
    return ", ".join(names[:cap]) + f", … (+{len(names) - cap} more)"


def _format_top(entries: list[tuple[str, int]], cap: int) -> tuple[list[str], int]:
    sorted_e = sorted(entries, key=lambda kv: kv[1], reverse=True)
    top = sorted_e[:cap]
    omitted = max(0, len(sorted_e) - len(top))
    lines = [f"- {name}: {_format_chars_and_tokens(value)}" for name, value in top]
    return lines, omitted


def format_context_help() -> str:
    return "\n".join(
        [
            "🧠 /context",
            "",
            "What counts as context (high-level), plus a breakdown mode.",
            "",
            "Try:",
            "- /context list   (short breakdown)",
            "- /context detail (per-file + per-tool + per-skill + system prompt size)",
            "- /context json   (same, machine-readable)",
            "",
            "Inline shortcut = a command token inside a normal message (e.g. "
            "“hey /status”). It runs immediately (allowlisted senders only) "
            "and is stripped before the model sees the remaining text.",
        ]
    )


def _shared_lines(report: ContextReport, session: SessionTokens) -> list[str]:
    file_lines: list[str] = []
    for f in report.injected_workspace_files:
        status = "MISSING" if f.missing else "TRUNCATED" if f.truncated else "OK"
        if f.missing:
            raw = "0"
            injected = "0"
        else:
            raw = _format_chars_and_tokens(f.raw_chars)
            injected = _format_chars_and_tokens(f.injected_chars)
        file_lines.append(f"- {f.name}: {status} | raw {raw} | injected {injected}")
    sandbox_line = f"Sandbox: mode={report.sandbox_mode or 'unknown'} sandboxed={str(report.sandbox_sandboxed).lower()}"
    skill_names = [s.name for s in report.skills_entries]
    skills_line = (
        f"Skills list (system prompt text): "
        f"{_format_chars_and_tokens(report.skills_prompt_chars)} "
        f"({len(skill_names)} skills)"
    )
    skills_names_line = f"Skills: {_format_name_list(skill_names, 20)}" if skill_names else "Skills: (none)"
    system_prompt_line = (
        f"System prompt ({report.source}): "
        f"{_format_chars_and_tokens(report.system_prompt_chars)} "
        f"(Project Context "
        f"{_format_chars_and_tokens(report.project_context_chars)})"
    )
    return [
        f"Workspace: {report.workspace_dir}",
        f"Bootstrap max/file: {_format_int(report.bootstrap_max_chars)} chars",
        f"Bootstrap max/total: {_format_int(report.bootstrap_total_max_chars)} chars",
        sandbox_line,
        system_prompt_line,
        "",
        "Injected workspace files:",
        *file_lines,
        "",
        skills_line,
        skills_names_line,
    ]


def _totals_line(report: ContextReport, session: SessionTokens) -> str:
    cached = session.total_tokens
    ctx_label = _format_int(session.context_tokens) if session.context_tokens is not None else "?"
    if cached is None:
        return f"Session tokens (cached): unknown / ctx={ctx_label}"
    return f"Session tokens (cached): {_format_int(cached)} total / ctx={ctx_label}"


def _tools_lines(report: ContextReport) -> tuple[str, str, str]:
    tool_list_line = f"Tool list (system prompt text): {_format_chars_and_tokens(report.tool_list_chars)}"
    tool_schema_line = (
        f"Tool schemas (JSON): "
        f"{_format_chars_and_tokens(report.tool_schema_chars)} "
        f"(counts toward context; not shown as text)"
    )
    tool_names = [t.name for t in report.tool_entries]
    tools_names_line = f"Tools: {_format_name_list(tool_names, 30)}" if tool_names else "Tools: (none)"
    return tool_list_line, tool_schema_line, tools_names_line


_INLINE_SHORTCUT_NOTE = (
    "Inline shortcut: a command token inside normal text (e.g. “hey /status”) "
    "that runs immediately (allowlisted senders only) and is stripped before the "
    "model sees the remaining message."
)


def format_context_list(report: ContextReport, session: SessionTokens) -> str:
    tool_list_line, tool_schema_line, tools_names_line = _tools_lines(report)
    return "\n".join(
        [
            "🧠 Context breakdown",
            *_shared_lines(report, session),
            tool_list_line,
            tool_schema_line,
            tools_names_line,
            "",
            _totals_line(report, session),
            "",
            _INLINE_SHORTCUT_NOTE,
        ]
    )


def format_context_detail(report: ContextReport, session: SessionTokens) -> str:
    tool_list_line, tool_schema_line, tools_names_line = _tools_lines(report)
    per_skill, per_skill_omit = _format_top(
        [(s.name, s.block_chars) for s in report.skills_entries],
        30,
    )
    per_tool_schema, per_tool_omit = _format_top(
        [(t.name, t.schema_chars) for t in report.tool_entries],
        30,
    )
    per_tool_summary, per_tool_summary_omit = _format_top(
        [(t.name, t.summary_chars) for t in report.tool_entries],
        30,
    )
    tool_props = [(t.name, t.properties_count or 0) for t in report.tool_entries if t.properties_count is not None]
    tool_props.sort(key=lambda kv: kv[1], reverse=True)
    tool_props_lines = [f"- {n}: {c} params" for n, c in tool_props[:30]]

    cached = session.total_tokens
    tracked_chars = report.system_prompt_chars + report.tool_schema_chars
    tracked_line = f"Tracked prompt estimate: {_format_chars_and_tokens(tracked_chars)}"
    if cached is not None:
        actual_line = f"Actual context usage (cached): {_format_int(cached)} tok"
        overhead = cached - estimate_tokens_from_chars(tracked_chars)
        overhead_line = (
            f"Untracked provider/runtime overhead: ~{_format_int(overhead)} tok"
            if overhead > 0
            else "Untracked provider/runtime overhead: not observed in cached usage"
        )
    else:
        actual_line = "Actual context usage (cached): unavailable"
        overhead_line = ""

    parts: list[str] = ["🧠 Context breakdown (detailed)", *_shared_lines(report, session)]
    if per_skill:
        parts.append("Top skills (prompt entry size):")
        parts.extend(per_skill)
    if per_skill_omit:
        parts.append(f"… (+{per_skill_omit} more skills)")
    parts.append("")
    parts.extend([tool_list_line, tool_schema_line, tools_names_line])
    parts.append("Top tools (schema size):")
    parts.extend(per_tool_schema)
    if per_tool_omit:
        parts.append(f"… (+{per_tool_omit} more tools)")
    parts.append("")
    parts.append("Top tools (summary text size):")
    parts.extend(per_tool_summary)
    if per_tool_summary_omit:
        parts.append(f"… (+{per_tool_summary_omit} more tools)")
    if tool_props_lines:
        parts.append("")
        parts.append("Tools (param count):")
        parts.extend(tool_props_lines)
    parts.append("")
    parts.append(tracked_line)
    parts.append(actual_line)
    if overhead_line:
        parts.append(overhead_line)
    parts.append("")
    parts.append(_totals_line(report, session))
    parts.append("")
    parts.append(_INLINE_SHORTCUT_NOTE)
    return "\n".join(parts)


def format_context_json(report: ContextReport, session: SessionTokens) -> str:
    payload = {
        "report": {
            "source": report.source,
            "model": report.model,
            "workspaceDir": report.workspace_dir,
            "bootstrapMaxChars": report.bootstrap_max_chars,
            "bootstrapTotalMaxChars": report.bootstrap_total_max_chars,
            "sandbox": {
                "mode": report.sandbox_mode,
                "sandboxed": report.sandbox_sandboxed,
            },
            "systemPrompt": {
                "chars": report.system_prompt_chars,
                # Fixed 4-char/token parity estimator (NEVER the per-model one)
                # so all four /context flavors report identical numbers
                "estimatedTokens": estimate_tokens_from_chars(report.system_prompt_chars),
                "projectContextChars": report.project_context_chars,
            },
            "injectedWorkspaceFiles": [
                {
                    "name": f.name,
                    "rawChars": f.raw_chars,
                    "injectedChars": f.injected_chars,
                    "missing": f.missing,
                    "truncated": f.truncated,
                }
                for f in report.injected_workspace_files
            ],
            "skills": {
                "promptChars": report.skills_prompt_chars,
                "entries": [{"name": s.name, "blockChars": s.block_chars} for s in report.skills_entries],
            },
            "tools": {
                "listChars": report.tool_list_chars,
                "schemaChars": report.tool_schema_chars,
                "entries": [
                    {
                        "name": t.name,
                        "schemaChars": t.schema_chars,
                        "summaryChars": t.summary_chars,
                        "propertiesCount": t.properties_count,
                    }
                    for t in report.tool_entries
                ],
            },
        },
        "session": {
            "totalTokens": session.total_tokens,
            "totalTokensFresh": session.total_tokens_fresh,
            "inputTokens": session.input_tokens,
            "outputTokens": session.output_tokens,
            "contextTokens": session.context_tokens,
        },
    }
    return json.dumps(payload, indent=2)
