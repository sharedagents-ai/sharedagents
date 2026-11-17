"""OpenClaw-style context injection.

At session start, prepend hardcoded sharedagents sections (WORKSPACE, MODE-*)
plus the operator's ``AGENTS.md`` and tool-call schema to the user's
configured ``system`` prompt.

Callers always pass an explicit ``root`` (the agent's home dir at
``~/.sharedagents/agents/<name>/``).

Missing ``AGENTS.md`` is skipped silently. With no injectable content,
the function returns the user's system prompt unchanged.

MEMORY.md is not loaded into the prompt (prefix-cache stability).

The system prompt is sticky per session: built once at session creation,
reused every turn, re-read from disk only after compaction.
"""

from __future__ import annotations

import hashlib as _hashlib
from collections import OrderedDict as _OrderedDict
from pathlib import Path

from sharedagents_agent.external_registry import EXTERNAL_ENGINE_VALUES

# Defaults are absolute (~/.sharedagents/...) so we never leak a `.sharedagents/`
# into the project tree if a caller forgets to pass an explicit root.
CONTEXT_DIR = Path.home() / ".sharedagents"


# --- Composition breakdown cache -------------------------
# ``load_system_prompt`` records, per composed prompt, the (raw, injected,
# missing, truncated) char counts of every operator workspace file it folded
# in. ``/context`` (``build_context_report``) looks the record up by the cached
# prompt text so it reports the sizes AS COMPOSED, never a fresh post-edit disk
# read that would silently desync the report from what the model saw.
_COMPOSITION_BREAKDOWNS: _OrderedDict[str, dict[str, tuple[int, int, bool, bool]]] = _OrderedDict()
_COMPOSITION_BREAKDOWN_MAX = 8


def _composition_key(text: str) -> str:
    return _hashlib.sha1((text or "").encode("utf-8", "replace")).hexdigest()


def _record_composition_breakdown(
    text: str,
    breakdown: dict[str, tuple[int, int, bool, bool]],
) -> None:
    if not breakdown:
        return
    key = _composition_key(text)
    _COMPOSITION_BREAKDOWNS[key] = dict(breakdown)
    _COMPOSITION_BREAKDOWNS.move_to_end(key)
    while len(_COMPOSITION_BREAKDOWNS) > _COMPOSITION_BREAKDOWN_MAX:
        _COMPOSITION_BREAKDOWNS.popitem(last=False)


def composition_breakdown(
    text: str,
) -> dict[str, tuple[int, int, bool, bool]] | None:
    """Per-file ``(raw, injected, missing, truncated)`` sizes recorded when
    ``text`` was composed by :func:`load_system_prompt`. ``None`` if the prompt
    was not composed by this process (caller falls back to a disk read)."""
    return _COMPOSITION_BREAKDOWNS.get(_composition_key(text))


# Operator-editable identity + local notes — the only agent-home file
# in the stable prefix. TOOLCALL schemas are runtime-hardcoded.
CONTEXT_FILES: tuple[str, ...] = ("AGENTS.md",)

# Shipped at runtime from ``sharedagents_agent.agent`` templates — not read
# from the agent home directory.
HARDCODED_CONTEXT_FILES: tuple[str, ...] = ("WORKSPACE.md",)

# Mode files: only ONE is loaded per turn, picked by the active permission
# mode. Shipped from runtime templates, not agent-home files.
MODE_FILES: dict[str, str] = {
    "plan": "MODE-PLAN.md",
    "ask": "MODE-ASK.md",
    "auto": "MODE-AUTO.md",
}

DEFAULT_TOOLCALL_FILE = "TOOLCALL.md"
# Family-keyed TOOLCALL schema. The filename is GENERATION-NEUTRAL: the
# wire format is a property of the model family, not of one release, so
# naming a version here would go stale on the next upgrade (it did — the
# qwen3 entry pointed at ``TOOLCALL-QWEN3.6.md`` while the fleet had
# already moved to Qwen3.8).
TOOLCALL_FILES: dict[str, str] = {
    "qwen3": "TOOLCALL-QWEN3.md",
}

# There is exactly ONE persona file, and it is model-AGNOSTIC. The
# former ``PERSONA_FILES`` / ``PERSONA_NAME_PREFIXES`` tables resolved a
# per-model section by family or by id prefix; both are gone. They
# injected hardcoded weight/architecture claims ("You are Qwen3.6 — a
# dense 27-billion-parameter model …") that were (a) keyed to a model
# generation and so silently stale after any upgrade, and (b) matched by
# ``str.startswith`` against the full model id, which never fired for
# the namespaced ids in real use (``qwen/qwen3.8-max``,
# ``anthropic/claude-opus-5``, ``google/gemini-3.7-flash`` …) — 32 of
# the 40 configured models fell through to the generic text while 4 were
# handed a persona for a model the runtime was not serving.
DEFAULT_PERSONA_FILE = "PERSONA.md"

# Engine kinds whose turns run in an EXTERNAL agent binary / ACP server.
# Built-in values come from the external registry; generic ``acp`` is included
# there as the user-configurable adapter. These agents receive NO
# model-specific persona and NO
# family-specific TOOLCALL schema: their system channel is owned by the
# external product, and a persona pinned to the merged default local
# model (``qwen3.6-27b-fp8``) leaks a wrong identity into it.
# External agents get no model-specific persona at all.
EXTERNAL_ENGINE_KINDS: frozenset[str] = EXTERNAL_ENGINE_VALUES

MEMORY_FILE = "MEMORY.md"

CONTEXT_FILE_MAX_BYTES = 1 * 1024 * 1024
CONTEXT_TOTAL_MAX_BYTES = 10 * 1024 * 1024

_LEGACY_ARTIFACT_SECTION_HEADINGS = frozenset(
    {
        "## SharedAgents Web UI output artifacts",
        "## SharedAgents Web UI screenshots and files",
    }
)


def strip_legacy_artifact_prompt_section(text: str) -> str:
    """Remove the retired deployment-specific artifact section.

    Older external-agent scaffolds and durable agent memory embedded browser
    delivery details. Those files are operator-owned and are deliberately not
    overwritten during upgrades, so remove only the known managed sections at
    prompt composition time. The shared runtime recipe supplies the safe,
    host-neutral replacement.
    """
    if not isinstance(text, str) or not any(heading in text for heading in _LEGACY_ARTIFACT_SECTION_HEADINGS):
        return text
    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.strip() in _LEGACY_ARTIFACT_SECTION_HEADINGS:
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept).strip()


def toolcall_filename_for_family(
    model_family: str | None,
    root: Path,
    *,
    engine_kind: str | None = None,
) -> str:
    """Resolve which TOOLCALL file to load for a model family."""
    if (engine_kind or "").strip().lower() in EXTERNAL_ENGINE_KINDS:
        # External engines bypass the family lookup entirely: their tool
        # surface is native, so only the engine-neutral schema applies.
        return DEFAULT_TOOLCALL_FILE
    fam = (model_family or "").strip().lower()
    if fam:
        override = TOOLCALL_FILES.get(fam)
        if override:
            from sharedagents_agent.agent import hardcoded_context_body

            if hardcoded_context_body(override):
                return override
            if (root / override).exists():
                return override
    return DEFAULT_TOOLCALL_FILE


def persona_filename_for_model(
    model_family: str | None,
    model_id: str | None,
    *,
    engine_kind: str | None = None,
) -> str:
    """Resolve which PERSONA file to load for a model.

    Priority:
    1. External engine kinds (``EXTERNAL_ENGINE_KINDS``) → ``""`` —
       no persona section at all.
    2. The single model-agnostic ``PERSONA.md``.
    3. ``""`` when that template is unavailable.

    ``model_family`` / ``model_id`` are accepted and deliberately UNUSED
    so the three callers (the runtime composer, the Agent Builder's
    render mirror, and the ``/context`` report) do not have to change,
    and so a future behaviour-only, family-keyed addendum has a seam to
    land on without another signature change.

    There are no per-model personas on purpose. See the module-level
    comment above ``DEFAULT_PERSONA_FILE`` for why the table was removed.
    """
    del model_family, model_id  # kept for signature compatibility
    if (engine_kind or "").strip().lower() in EXTERNAL_ENGINE_KINDS:
        return ""
    from sharedagents_agent.agent import hardcoded_context_body

    if hardcoded_context_body(DEFAULT_PERSONA_FILE):
        return DEFAULT_PERSONA_FILE
    return ""


def load_system_prompt_parts(
    user_system: str = "",
    *,
    root: Path | None = None,
    mode: str | None = None,
    model_family: str | None = None,
    model_id: str | None = None,
    profile_system: str | None = None,
    agent_name: str | None = None,
    engine_kind: str | None = None,
    record_breakdown: bool = True,
) -> list[dict]:
    """Same file walk as :func:`load_system_prompt`, but returns the
    attributed parts instead of the joined string.

    Each part is ``{"source", "title", "text"}`` where ``source`` is
    ``"hardcoded"`` (WORKSPACE.md / PERSONA-*.md / TOOLCALL*.md /
    MODE-*.md), ``"agents_md"`` (AGENTS.md or the profile AGENTS.md
    body), or ``"system"`` (the ``# SYSTEM`` config-system fallback).

    ``record_breakdown=False`` skips
    :func:`_record_composition_breakdown` — the Agent Builder render
    path uses it so page views do not pollute the runtime ``/context``
    report.

    The join order, caps, truncation markers, and profile branch are
    identical to the legacy joined-string implementation, so
    ``"\\n\\n".join(p["text"] for p in parts)`` reproduces the old
    output byte-for-byte.
    """
    if root is None:
        root = CONTEXT_DIR
    name = (agent_name or root.name or "").strip()

    total_consumed = 0
    # Per-file composed sizes, keyed by filename, recorded for /context so it
    # can report what was injected without re-reading disk.
    _breakdown: dict[str, tuple[int, int, bool, bool]] = {}

    def _read(path: Path) -> str | None:
        nonlocal total_consumed
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                blob = f.read(CONTEXT_FILE_MAX_BYTES + 1)
        except OSError:
            return None
        truncated = len(blob) > CONTEXT_FILE_MAX_BYTES
        if truncated:
            blob = blob[:CONTEXT_FILE_MAX_BYTES]
        if total_consumed + len(blob) > CONTEXT_TOTAL_MAX_BYTES:
            # A file that EXISTS but is dropped because the
            # aggregate context budget is already exhausted must never vanish
            # silently — leave an explicit omission marker naming the file, the
            # same operator-visible signal the per-file cap emits on truncation.
            return (
                f"[... {path.name} omitted by sharedagents context loader: "
                f"aggregate context budget "
                f"CONTEXT_TOTAL_MAX_BYTES={CONTEXT_TOTAL_MAX_BYTES} exhausted]"
            )
        total_consumed += len(blob)
        body = blob.decode("utf-8", errors="replace").strip()
        if path.name == "AGENTS.md":
            body = strip_legacy_artifact_prompt_section(body)
        if truncated:
            body += (
                f"\n\n[... truncated by sharedagents context loader: file "
                f"exceeded CONTEXT_FILE_MAX_BYTES={CONTEXT_FILE_MAX_BYTES}]"
            )
        result = body or None
        if result is not None:
            _breakdown[path.name] = (len(blob), len(result), False, truncated)
        return result

    def _hardcoded(filename: str) -> str | None:
        from sharedagents_agent.agent import hardcoded_context_body

        body = hardcoded_context_body(
            filename,
            agent_name=name,
            agent_home=root,
        )
        return body.strip() or None if body else None

    def _toolcall_file() -> str | None:
        """Filename _toolcall_body() would load (None when its content
        resolves to nothing) — used to attribute the part's title."""
        fam = (model_family or "").strip().lower()
        if (engine_kind or "").strip().lower() not in EXTERNAL_ENGINE_KINDS:
            family_file = TOOLCALL_FILES.get(fam) if fam else None
            if family_file:
                return family_file if _hardcoded(family_file) is not None else None
        # External engine kinds skip the family lookup: generic schema only.
        return DEFAULT_TOOLCALL_FILE

    def _toolcall_title() -> str:
        return _toolcall_file() or DEFAULT_TOOLCALL_FILE

    def _toolcall_body() -> str | None:
        fam = (model_family or "").strip().lower()
        if (engine_kind or "").strip().lower() not in EXTERNAL_ENGINE_KINDS:
            family_file = TOOLCALL_FILES.get(fam) if fam else None
            if family_file:
                hardcoded = _hardcoded(family_file)
                if hardcoded is not None:
                    return hardcoded
        # External engine kinds skip the family lookup: generic schema only.
        return _hardcoded(DEFAULT_TOOLCALL_FILE)

    def _persona_body() -> str | None:
        pf = persona_filename_for_model(
            model_family,
            model_id,
            engine_kind=engine_kind,
        )
        if not pf:
            return None
        return _hardcoded(pf)

    if profile_system and profile_system.strip():
        clean_profile = strip_legacy_artifact_prompt_section(
            profile_system,
        ).strip()
        parts: list[dict] = (
            [{"source": "agents_md", "title": "AGENTS.md (profile)", "text": clean_profile}] if clean_profile else []
        )
        if mode and mode in MODE_FILES:
            body = _hardcoded(MODE_FILES[mode])
            if body is not None:
                parts.append({"source": "hardcoded", "title": MODE_FILES[mode], "text": body})
        if user_system.strip():
            parts.append(
                {"source": "system", "title": "# SYSTEM (config system)", "text": f"# SYSTEM\n{user_system.strip()}"}
            )
        return parts

    parts = []

    ws = _hardcoded("WORKSPACE.md")
    if ws is not None:
        parts.append({"source": "hardcoded", "title": "WORKSPACE.md", "text": ws})

    pers = _persona_body()
    if pers is not None:
        parts.append(
            {
                "source": "hardcoded",
                "title": persona_filename_for_model(model_family, model_id, engine_kind=engine_kind),
                "text": pers,
            }
        )

    for fname in CONTEXT_FILES:
        body = _read(root / fname)
        if body is not None:
            parts.append({"source": "agents_md", "title": fname, "text": body})

    tc = _toolcall_body()
    if tc is not None:
        parts.append({"source": "hardcoded", "title": _toolcall_title(), "text": tc})

    if mode and mode in MODE_FILES:
        body = _hardcoded(MODE_FILES[mode])
        if body is not None:
            parts.append({"source": "hardcoded", "title": MODE_FILES[mode], "text": body})

    if not parts:
        return [{"source": "system", "title": "# SYSTEM (config system)", "text": user_system}] if user_system else []

    if user_system.strip():
        parts.append(
            {"source": "system", "title": "# SYSTEM (config system)", "text": f"# SYSTEM\n{user_system.strip()}"}
        )

    if record_breakdown:
        _record_composition_breakdown("\n\n".join(p["text"] for p in parts), _breakdown)
    return parts


def load_system_prompt(
    user_system: str = "",
    *,
    root: Path | None = None,
    mode: str | None = None,
    model_family: str | None = None,
    model_id: str | None = None,
    profile_system: str | None = None,
    agent_name: str | None = None,
    engine_kind: str | None = None,
) -> str:
    """Concatenate workspace context into one system prompt.

    Section order (stable prefix):

        WORKSPACE (hardcoded) → PERSONA (hardcoded, model-specific; OMITTED
        for external engine kinds) → AGENTS.md (operator) → TOOLCALL
        → MODE-<active> (hardcoded) → optional ``# SYSTEM``

    OpenCode-style profile agents replace the stack with the agent-home
    ``AGENTS.md`` (+ hardcoded MODE only).

    Thin wrapper over :func:`load_system_prompt_parts` — the join is
    byte-identical to the pre-parts implementation.
    """
    return "\n\n".join(
        p["text"]
        for p in load_system_prompt_parts(
            user_system,
            root=root,
            mode=mode,
            model_family=model_family,
            model_id=model_id,
            profile_system=profile_system,
            agent_name=agent_name,
            engine_kind=engine_kind,
        )
    )


def memory_path(*, root: Path | None = None) -> Path:
    if root is None:
        root = CONTEXT_DIR
    return root / MEMORY_FILE
