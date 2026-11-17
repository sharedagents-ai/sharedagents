"""Skills — AgentSkills-compatible runbooks for the agent.

A *skill* is a directory containing `SKILL.md` (Markdown with YAML
frontmatter). Compatible with OpenClaw's skill format ([docs](https://docs.openclaw.ai/tools/skills),
[clawhub spec](https://github.com/openclaw/clawhub/blob/main/docs/skill-format.md))
so existing OpenClaw skills work in sharedagents without conversion.

## Layout

```
<skill-name>/
    SKILL.md              # required — frontmatter + body
    scripts/              # optional — executable code
    references/           # optional — extra docs
    assets/               # optional — files used in output
```

`SKILL.md` frontmatter (only `name` + `description` are required;
`description` is what the model sees when deciding whether to invoke
the skill, so write it concretely):

```yaml
---
name: brave-search
description: Search the web via Brave Search API. Returns ranked results.
version: 1.0.0
metadata:
  openclaw:
    requires:
      env: [BRAVE_API_KEY]
---
```

## Search-path precedence (high → low, first match wins)

1. **Env override**: `SHAREDAGENTS_SKILLS_DIRS=path1:path2:…` (colon
   separated, prepended at highest precedence — a true override)
2. **Per-agent**: `<agent_home>/skills/<name>/SKILL.md`
3. **Shared sharedagents**: `~/.sharedagents/skills/<name>/SKILL.md`
4. **Config bundle dirs**: `skills_dirs` in `~/.sharedagents/config.json`
   (a list of paths) — the persistent way to register an external
   skill bundle (e.g. an everything-claude-code checkout under
   `~/.sharedagents/ecc/skills`) so the gateway, TUI, and CLI all see it.
   Deliberately LOW precedence: a bulk bundle fills gaps but never
   silently shadows the operator's own curated skills above it. This
   is also the slot for an `~/.openclaw/skills` dir if the operator
   runs OpenClaw too — that path is environment-specific, so it lives
   in the operator's config, NOT hardcoded in sharedagents.
5. **Built-in default**: `<sharedagents package>/builtin_skills/<name>/SKILL.md`
   (skills shipped with the sharedagents code itself — present on every
   install without the user setting anything up; user-installed
   skills at higher-priority paths can override these)

## API

- `discover_skills(cfg)` — list all unique skills across the chain, with
  the highest-precedence source winning on name collisions.
- `find_skill(name, cfg)` — one skill by name, or `None`.
- `parse_skill_md(path)` — read + parse one file into a `Skill`.
- `skills_summary_block(cfg)` — `# SKILLS` Markdown block injected into
  the system prompt so the model can `skill_get(name)` on demand.
"""

from __future__ import annotations

import contextvars as _contextvars
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Skill:
    """One AgentSkills-compatible skill loaded from disk."""

    name: str
    description: str
    body: str  # the rest of SKILL.md after the frontmatter
    source: Path  # absolute path to the SKILL.md file
    version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    requires_env: tuple[str, ...] = ()
    requires_bins: tuple[str, ...] = ()
    always: bool = False  # always-active hint from frontmatter

    @property
    def directory(self) -> Path:
        return self.source.parent

    @property
    def short_description(self) -> str:
        """First sentence (or first 120 chars) of the description."""
        d = (self.description or "").strip()
        cut = re.split(r"(?<=[.!?])\s+", d, maxsplit=1)
        return (cut[0] if cut else d)[:120]


# Two-layer YAML parsing: (1) extract the frontmatter block bounded by
# `---` lines; (2) parse a *very limited* dialect — flat scalars, lists
# of scalars, one nested `metadata.openclaw.requires.*` map. Avoiding a
# pyyaml dep keeps `skills` importable in the tool_runner subprocess
# without bloating its boot-time imports.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


_BLOCK_SCALAR_RE = re.compile(r"^([>|])([+-]?)\s*$")


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter_dict, body_after_frontmatter).

    Supports flat ``key: value``, list-style ``- item``, properly
    INDENTED nested maps (``metadata:\\n  openclaw:\\n    requires:\\n
    env: [A, B]``), and YAML block scalars (``key: >`` folded /
    ``key: |`` literal, with optional ``-``/``+`` chomping). NOT a
    full YAML parser — falls back to leaving unparseable values as
    strings. Dotted-path keys (``metadata.openclaw.always: true``)
    are stored verbatim — callers reading nested fields should write
    them as proper indented maps.

    The block-scalar support is load-bearing for cross-ecosystem
    skill bundles: Claude-Code skills (e.g. everything-claude-code)
    routinely write a multi-line `description: >-` block, and without
    folding it the description would parse as the literal string
    ``">-"`` — breaking both the `# SKILLS` prompt block and the
    embedding-backed auto-invoke index.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group(1)
    body = text[m.end() :]
    data: dict[str, Any] = {}
    current_key: str | None = None
    indent_stack: list[tuple[int, dict[str, Any]]] = [(0, data)]
    # Indexed walk (not `for line in ...`) so a block scalar can
    # consume the lines that belong to it before the loop continues.
    lines = raw.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # List item
        m_list = re.match(r"^(\s*)-\s+(.+)$", line)
        if m_list and current_key is not None:
            val = _coerce(m_list.group(2).strip())
            # Last container at or below this indent
            container = indent_stack[-1][1]
            existing = container.get(current_key)
            if isinstance(existing, list):
                existing.append(val)
            else:
                container[current_key] = [val]
            continue
        # Key: value
        m_kv = re.match(r"^(\s*)([A-Za-z_][\w.-]*)\s*:\s*(.*)$", line)
        if not m_kv:
            continue
        indent = len(m_kv.group(1))
        key = m_kv.group(2)
        rest = m_kv.group(3).strip()
        # Pop indent stack to current level
        while indent_stack and indent_stack[-1][0] > indent:
            indent_stack.pop()
        container = indent_stack[-1][1]
        m_block = _BLOCK_SCALAR_RE.match(rest)
        if m_block:
            # Block scalar: consume every following line indented
            # deeper than the key (blank lines included — they are
            # paragraph breaks / preserved newlines). A line at or
            # below the key's indent ends the block.
            style, chomp = m_block.group(1), m_block.group(2)
            block_lines: list[str] = []
            while i < n:
                bl = lines[i]
                if bl.strip() == "":
                    block_lines.append("")
                    i += 1
                    continue
                if (len(bl) - len(bl.lstrip())) <= indent:
                    break
                block_lines.append(bl)
                i += 1
            container[key] = _join_block(block_lines, style, chomp)
            current_key = key
        elif rest == "":
            # Nested map opens here — make a new dict bound to this key.
            new_map: dict[str, Any] = {}
            container[key] = new_map
            indent_stack.append((indent + 2, new_map))
            current_key = None
        elif rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1]
            items = [_coerce(x.strip()) for x in inner.split(",") if x.strip()]
            container[key] = items
            current_key = key
        else:
            container[key] = _coerce(rest)
            current_key = key
    return data, body


def _join_block(block_lines: list[str], style: str, chomp: str) -> str:
    """Render YAML block-scalar lines into a single string.

    `style` is ``">"`` (folded — single newlines fold to spaces,
    blank lines are paragraph breaks) or ``"|"`` (literal — newlines
    preserved). `chomp` is ``""`` (clip), ``"-"`` (strip), or ``"+"``
    (keep) — for the flat-string form we keep in the frontmatter
    dict, clip and strip are equivalent (no trailing newline) and
    `+` keeps exactly one.
    """
    # Strip the block's common leading indent off every line.
    non_empty = [ln for ln in block_lines if ln.strip()]
    common = min((len(ln) - len(ln.lstrip()) for ln in non_empty), default=0)
    stripped = [ln[common:] if len(ln) >= common else ln.lstrip() for ln in block_lines]
    # Trailing blank lines are re-applied (or not) by chomping.
    while stripped and stripped[-1] == "":
        stripped.pop()
    if style == "|":
        out = "\n".join(stripped)
    else:
        # Folded: run consecutive non-blank lines together with a
        # space; a blank line becomes a hard newline (paragraph break).
        chunks: list[str] = []
        buf: list[str] = []
        for ln in stripped:
            if ln == "":
                chunks.append(" ".join(buf))
                buf = []
            else:
                buf.append(ln)
        chunks.append(" ".join(buf))
        out = "\n".join(c for c in chunks).strip("\n")
    return out + "\n" if chomp == "+" else out.rstrip("\n")


def _coerce(s: str) -> Any:
    """Best-effort YAML scalar → Python value."""
    if s == "" or s is None:
        return ""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~"):
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


def parse_skill_md(path: Path) -> Skill | None:
    """Parse a single `SKILL.md` (or `skill.md`) into a `Skill`.

    Returns None if the file is missing or has no `name` in frontmatter.
    """
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    fm, body = _parse_frontmatter(text)
    name = str(fm.get("name") or "").strip()
    if not name:
        # Spec says name is required — skip silently rather than 500-ing
        # the agent's session start.
        return None
    description = str(fm.get("description") or "").strip()
    version = str(fm.get("version") or "")

    md = fm.get("metadata") or {}
    if not isinstance(md, dict):
        md = {}
    oc = md.get("openclaw") or md.get("clawdbot") or md.get("clawdis") or {}
    if not isinstance(oc, dict):
        oc = {}
    requires = oc.get("requires") or {}
    if not isinstance(requires, dict):
        requires = {}

    def _tuple(x: Any) -> tuple[str, ...]:
        if isinstance(x, list):
            return tuple(str(v) for v in x)
        if isinstance(x, str) and x:
            return (x,)
        return ()

    # ``always`` may be set at the frontmatter top level OR nested under
    # ``metadata.openclaw.always`` (legacy OpenClaw convention). Accept
    # either so a CC-style skill that just sets ``always: true`` at the
    # top of its frontmatter is honoured.
    always = bool(fm.get("always", False) or oc.get("always", False))
    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        source=path,
        version=version,
        metadata=md,
        requires_env=_tuple(requires.get("env")),
        requires_bins=_tuple(requires.get("bins")),
        always=always,
    )


def _config_skills_dirs() -> list[Path]:
    """Resolve the global config's `skills_dirs` list ([] on any failure).

    Lazy import of `load_global_config` mirrors `_load_skills_config`
    — keeps `skills` cheap to import in the tool-runner subprocess and
    dodges import-order coupling with `agent`. A bare string is
    accepted as a one-element list; non-string entries are skipped.
    """
    try:
        from sharedagents_agent.agent import load_global_config

        cfg = load_global_config() or {}
    except Exception:  # noqa: BLE001
        return []
    raw = cfg.get("skills_dirs")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[Path] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(Path(item.strip()).expanduser())
    return out


def skill_search_paths(agent_home: Path | None = None) -> list[Path]:
    """Resolve the search chain in precedence order (high → low).

    See module docstring for the rules. Honors `SHAREDAGENTS_SKILLS_DIRS`
    (`:`-separated) and the global config's `skills_dirs` list at the
    top of the chain if set.
    """
    chain: list[Path] = []
    extra = os.environ.get("SHAREDAGENTS_SKILLS_DIRS")
    if extra:
        for p in extra.split(":"):
            p = p.strip()
            if p:
                chain.append(Path(p).expanduser())
    if agent_home is not None:
        chain.append(Path(agent_home) / "skills")
    chain.append(Path.home() / ".sharedagents" / "skills")
    # Claude Code's canonical user-level skill dir. A skill installed
    # by ``claude /plugin install ...`` or dropped by the operator
    # into ``~/.claude/skills/<name>/SKILL.md`` shows up in sharedagents
    # AND in Claude Code with zero duplication — the user's goal of
    # one Claude-Code-compatible format. Lower precedence than
    # ``~/.sharedagents/skills/`` so an sharedagents-specific override wins, but
    # higher than plugin-bundle contributions and config bundles.
    chain.append(Path.home() / ".claude" / "skills")
    # Installed plugins (~/.sharedagents/plugins/<name>/skills) — each
    # enabled plugin contributes its own skill dir automatically. The
    # filesystem is the source of truth (no plugins.json registry);
    # disabled plugins live at <name>.disabled and don't contribute.
    try:
        from sharedagents_agent.plugins import plugin_skill_dirs

        chain.extend(plugin_skill_dirs())
    except Exception:  # noqa: BLE001
        pass
    # Config-declared bundle dirs (`skills_dirs`) — additive, low
    # precedence ON PURPOSE: a bulk bundle (e.g. everything-claude-code)
    # fills gaps but never silently shadows the operator's own curated
    # skills in the dirs above. Use `SHAREDAGENTS_SKILLS_DIRS` for a true
    # override. This is also where an operator points sharedagents at an
    # `~/.openclaw/skills` dir if they happen to run OpenClaw too —
    # that path is environment-specific and is NOT hardcoded here.
    chain.extend(_config_skills_dirs())
    # Bundled defaults — ship with the sharedagents package, lowest
    # precedence so a user copy in any of the above paths overrides.
    # Resolved relative to this module so it works from a wheel or an
    # editable install equally.
    chain.append(Path(__file__).resolve().parent / "builtin_skills")
    # De-dupe while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in chain:
        rp = p.resolve(strict=False)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _load_skills_config() -> dict[str, Any]:
    """Read the global config's ``skills`` map ({} on any failure).

    Lazy import of ``load_global_config`` keeps ``skills`` cheap to
    import in the tool-runner subprocess (see the frontmatter-parser
    note above) and dodges any import-order coupling with ``agent``.
    """
    try:
        from sharedagents_agent.agent import load_global_config

        cfg = load_global_config() or {}
    except Exception:  # noqa: BLE001
        return {}
    raw = cfg.get("skills")
    return raw if isinstance(raw, dict) else {}


def _skill_enabled(name: str, skills_cfg: dict[str, Any]) -> bool:
    """A skill is enabled unless the global config explicitly says no.

    Config entry may be a ``{"enabled": bool}`` dict or a bare bool;
    anything else — or no entry at all — counts as enabled. The
    default-on rule means a skill the wizard never enumerated (a fresh
    builtin, a hand-dropped ``~/.sharedagents/skills/<name>``) still works
    without a config edit; only an explicit ``enabled: false`` is a
    kill switch.
    """
    entry = skills_cfg.get(name)
    if entry is None:
        return True
    if isinstance(entry, bool):
        return entry
    if isinstance(entry, dict):
        val = entry.get("enabled", True)
        return val if isinstance(val, bool) else True
    return True


# Task-local active skill permission. Set at turn-compose (turn_runtime) from the agent's
# cfg, so the CONTEXT-FREE skill runners (run_skill_search / run_skill_get, which call
# discover_skills()/find_skill(name) with no agent context) also honour the allowlist —
# i.e. SkillSearch shows only the agent's allowed skills, not just the prompt catalog.
# Holds ``(allowlist, strict)`` or ``None``. ContextVar => isolated per asyncio task, so
# concurrent agents in the gateway don't leak allowlists into each other.
_ACTIVE_SKILL_ALLOWLIST: _contextvars.ContextVar[tuple | None] = _contextvars.ContextVar(
    "sharedagents_active_skill_allowlist", default=None
)

# Sentinel for find_skill: distinguishes "caller passed nothing" (→ use active permission)
# from "caller explicitly passed None" (→ unrestricted lookup).
_UNSET = object()

# Retired model-facing skills. Never advertise these names, even if a
# leftover SKILL.md is dropped back into a skills directory.
_RETIRED_SKILL_NAMES = frozenset({"sdlc", "sdlc-trace"})


def set_active_skill_allowlist(allowlist: list[str] | None, strict: bool) -> None:
    """Bind the active agent's skill permission for this task (called at turn compose)."""
    _ACTIVE_SKILL_ALLOWLIST.set((list(allowlist) if allowlist is not None else None, bool(strict)))


def get_active_skill_allowlist() -> tuple[list[str] | None, bool]:
    """The task-local active skill permission (``(allowlist, strict)``); ``(None, False)`` if unset.

    The context-free skill runners read this so ``SkillSearch`` / ``Skill`` show and load only
    the agent's allowed skills. Other callers pass ``allowlist=`` explicitly (no implicit fallback,
    so no-arg ``discover_skills()`` stays "all" — back-compat).
    """
    active = _ACTIVE_SKILL_ALLOWLIST.get()
    return (active[0], active[1]) if active is not None else (None, False)


def apply_skill_allowlist(
    skills: list[Skill],
    allowlist: list[str] | None,
    strict: bool,
) -> list[Skill]:
    """Per-agent skill permission (mirrors ``tools_allowlist`` semantics).

    - ``allowlist is None``        → no restriction (all skills). THE DEFAULT:
      every built-in profile (build/plan/explore/general/jarvis) resolves to
      None, so skills are permissive/discoverable by default.
    - ``allowlist`` non-empty      → ONLY skills whose name is listed (operator opt-in).
    - empty list + ``strict``      → NO skills (an operator who deliberately locks
      an agent to zero skills; no built-in profile ships this way).
    - empty list + not ``strict``  → all skills.
    """
    if allowlist is None:
        return skills
    names = {str(n).strip() for n in allowlist if str(n).strip()}
    if names:
        return [s for s in skills if s.name in names]
    return [] if strict else skills


def discover_skills(
    agent_home: Path | None = None,
    *,
    include_disabled: bool = False,
    allowlist: list[str] | None = None,
    allowlist_strict: bool = False,
) -> list[Skill]:
    """Walk the search chain and return all skills, name-deduplicated.

    First match per name wins (highest precedence in the chain). Skills
    with malformed / missing frontmatter are silently skipped. Result is
    sorted by name for stable system-prompt rendering (cache-friendly).

    Skills the global config's ``skills`` map marks ``enabled: false``
    are filtered out (so they vanish from the system prompt, `skill_get`,
    and tool dispatch alike) unless ``include_disabled=True`` — which
    ``sharedagents skills list`` passes so an operator can still see what is
    available to re-enable.
    """
    by_name: dict[str, Skill] = {}
    for root in skill_search_paths(agent_home):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if child.name in _RETIRED_SKILL_NAMES:
                continue
            for fname in ("SKILL.md", "skill.md"):
                skill_path = child / fname
                if skill_path.is_file():
                    skill = parse_skill_md(skill_path)
                    if skill is None:
                        continue
                    if skill.name in _RETIRED_SKILL_NAMES:
                        continue
                    if skill.name in by_name:
                        # Lower-precedence; skip.
                        continue
                    by_name[skill.name] = skill
                    break
    if not include_disabled:
        skills_cfg = _load_skills_config()
        if skills_cfg:
            by_name = {n: s for n, s in by_name.items() if _skill_enabled(n, skills_cfg)}
    out = sorted(by_name.values(), key=lambda s: s.name)
    return apply_skill_allowlist(out, allowlist, allowlist_strict)


def find_skill(name: str, agent_home: Path | None = None, *, allowlist=_UNSET, allowlist_strict=_UNSET) -> Skill | None:
    """Look up one skill by name across the search chain (ENFORCING the allowlist).

    ``find_skill`` is a load/permission boundary, so when the caller does NOT pass an
    explicit allowlist it falls back to the task-local ACTIVE permission. This closes the
    bypass where a runtime path (``Skill`` / ``SkillSearch(select:…)`` / command lookup)
    loads a skill OUTSIDE the agent's allowlist by naming it directly. Pass ``allowlist=None``
    explicitly to opt out (unrestricted lookup). (NB: ``discover_skills``/``skills_summary_block``
    keep ``None``-default = all for back-compat; only this lookup auto-enforces.)
    """
    if not name:
        return None
    if allowlist is _UNSET:
        allowlist, active_strict = get_active_skill_allowlist()
        if allowlist_strict is _UNSET:
            allowlist_strict = active_strict
    if allowlist_strict is _UNSET:
        allowlist_strict = False
    for s in discover_skills(agent_home, allowlist=allowlist, allowlist_strict=allowlist_strict):
        if s.name == name:
            return s
    return None


def skills_summary_block(
    agent_home: Path | None = None, *, allowlist: list[str] | None = None, allowlist_strict: bool = False
) -> str:
    """Render the ``## Skills`` block injected into the system prompt.

    Lists every skill's name + short description inline so the model can
    pick the right one in one step (no extra tool call). The model loads
    a full body via the ``Skill`` tool when needed.

    Returns ``""`` when no skills are installed (or the agent's allowlist is empty+strict).
    """
    skills = discover_skills(agent_home, allowlist=allowlist, allowlist_strict=allowlist_strict)
    if not skills:
        return ""
    lines = [
        "## Skills",
        "Skills provide specialized instructions and workflows for specific tasks.",
        "Use the ``Skill`` tool to load a skill when a task matches its description.",
        "",
    ]
    always = [s for s in skills if getattr(s, "always", False)]
    on_demand = [s for s in skills if not getattr(s, "always", False)]
    for label, group in (("Always-active", always), ("Available", on_demand)):
        if not group:
            continue
        lines.append(f"### {label} ({len(group)}) {'— loaded every turn' if label == 'Always-active' else ''}")
        lines.append("")
        for s in group:
            desc = s.short_description or s.description
            lines.append(f"- **{s.name}**: {desc}" if desc else f"- **{s.name}**")
        lines.append("")
    return "\n".join(lines)


def import_openclaw_skill(name: str, *, target_root: Path | None = None) -> Path:
    """Snapshot a skill from `~/.openclaw/skills/<name>/` into sharedagents's
    shared dir (`~/.sharedagents/skills/<name>/`).

    Returns the destination directory. Raises FileNotFoundError if the
    source doesn't exist; FileExistsError if the destination already does.
    `target_root` overrides the destination root (used by tests).

    This is a deliberate copy, not a symlink — once you import it,
    sharedagents's edits don't bleed back into OpenClaw's copy and vice versa.
    Re-run after upstream updates if you want to pull changes.
    """
    import shutil

    src = Path.home() / ".openclaw" / "skills" / name
    if not src.is_dir():
        raise FileNotFoundError(f"OpenClaw skill not found: {src}")
    if target_root is None:
        target_root = Path.home() / ".sharedagents" / "skills"
    dst = target_root / name
    if dst.exists():
        raise FileExistsError(f"already imported: {dst} (delete it or pick a different target)")
    target_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst


def create_skill(
    name: str,
    target_root: Path | None = None,
    *,
    description: str = "",
    create_venv: bool = False,
) -> Path:
    """Scaffold a new AgentSkills-compatible skill directory.

    Creates:
      <target_root>/<name>/
      ├── SKILL.md
      ├── scripts/
      │   ├── run.py
      │   └── requirements.txt
      ├── references/
      ├── tests/
      │   └── test_<name>.py
      └── .venv/ (if create_venv is True)
    """
    if not name or not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
        raise ValueError(f"invalid skill name: {name!r}. Skill names must be kebab-case (e.g. 'my-tool').")
    if target_root is None:
        target_root = Path.home() / ".sharedagents" / "skills"
    dest = target_root / name
    if dest.exists():
        raise FileExistsError(f"skill directory already exists: {dest}")

    dest.mkdir(parents=True, exist_ok=False)
    scripts_dir = dest / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    references_dir = dest / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = dest / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    desc_text = (
        description.strip()
        or f"Use when performing tasks with {name}. Trigger whenever the user mentions {name} workflows."
    )
    title_text = " ".join(part.capitalize() for part in name.split("-"))

    skill_md_content = f"""---
name: {name}
description: {desc_text}
metadata:
  sharedagents:
    always: false
---

# {title_text}

## Overview
What this skill does and why it exists.

## When to Use
Clear trigger conditions for when this skill should be invoked.

## Quick Reference
Summary of key commands or workflows:
```bash
~/.sharedagents/skills/{name}/.venv/bin/python ~/.sharedagents/skills/{name}/scripts/run.py
```

## Implementation
Detailed step-by-step instructions.

## Common Mistakes
Failure modes and gotchas to avoid.
"""
    (dest / "SKILL.md").write_text(skill_md_content, encoding="utf-8")
    (scripts_dir / "requirements.txt").write_text("# Python dependencies for this skill\n", encoding="utf-8")
    run_script = f"""#!/usr/bin/env python3
\"\"\"Main entry point for {name} skill.\"\"\"
import sys


def main() -> int:
    print(f"Running {name} skill script")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
    run_path = scripts_dir / "run.py"
    run_path.write_text(run_script, encoding="utf-8")
    try:
        run_path.chmod(0o755)
    except OSError:
        pass

    clean_name = name.replace("-", "_")
    test_content = f"""\"\"\"Tests for {name} skill.\"\"\"
from pathlib import Path


def test_{clean_name}_structure():
    skill_dir = Path(__file__).resolve().parent.parent
    assert (skill_dir / "SKILL.md").is_file()
    assert (skill_dir / "scripts" / "run.py").is_file()
"""
    (tests_dir / f"test_{clean_name}.py").write_text(test_content, encoding="utf-8")

    if create_venv:
        import venv

        venv.create(dest / ".venv", with_pip=True)

    return dest


def check_skill(
    skill_or_path: Any,
    agent_home: Path | None = None,
) -> list[str]:
    """Validate and lint a skill. Returns a list of error/warning strings."""
    import shutil

    warnings: list[str] = []

    # Resolve skill
    if isinstance(skill_or_path, Path) or (
        isinstance(skill_or_path, str) and ("/" in skill_or_path or "\\" in skill_or_path)
    ):
        p = Path(skill_or_path).resolve()
        if p.is_dir():
            skill_dir = p
            skill_file = p / "SKILL.md" if (p / "SKILL.md").is_file() else p / "skill.md"
        else:
            skill_file = p
            skill_dir = p.parent
        if not skill_file.is_file():
            return [f"missing SKILL.md in {skill_dir}"]
        skill = parse_skill_md(skill_file)
        if skill is None:
            return [f"failed to parse YAML frontmatter in {skill_file}"]
    else:
        name = str(skill_or_path)
        skill = find_skill(name, agent_home=agent_home, allowlist=None)
        if skill is None:
            return [f"skill not found: {name!r}"]
        skill_dir = skill.directory
        skill_file = skill.source

    # 1. Name check
    if not skill.name:
        warnings.append("frontmatter missing 'name'")
    elif skill.name != skill_dir.name and skill_dir.name not in ("builtin_skills", "skills"):
        warnings.append(f"frontmatter name {skill.name!r} does not match directory name {skill_dir.name!r}")

    # 2. Description check & token budget (<150 words recommended)
    if not skill.description:
        warnings.append("frontmatter missing 'description'")
    else:
        words = skill.description.split()
        if len(words) > 150:
            warnings.append(f"description is long ({len(words)} words; recommended <150 words for discoverability)")

    # 3. Required env vars
    for ev in skill.requires_env:
        if ev not in os.environ:
            warnings.append(f"missing required environment variable: {ev}")

    # 4. Required binaries
    for b in skill.requires_bins:
        if shutil.which(b) is None:
            warnings.append(f"missing required binary in PATH: {b}")

    # 5. Scripts directory check
    scripts_dir = skill_dir / "scripts"
    if scripts_dir.is_dir():
        for script in sorted(scripts_dir.iterdir()):
            if script.is_file() and script.suffix == ".py":
                try:
                    mode = script.stat().st_mode
                    if not (mode & 0o111):
                        warnings.append(f"script is not executable: scripts/{script.name}")
                    content = script.read_text(encoding="utf-8", errors="ignore")
                    if not content.startswith("#!"):
                        warnings.append(f"script missing shebang (#!/usr/bin/env python3): scripts/{script.name}")
                except OSError:
                    pass

    return warnings
