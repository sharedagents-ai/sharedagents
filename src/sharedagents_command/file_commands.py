"""File-based slash commands — Claude Code–compatible.

A *command* is a Markdown file (``<name>.md``) whose body is a
"quick executable prompt": invoking ``/<name> [args]`` expands the
file's body (with ``$ARGUMENTS`` / ``$1``…``$9`` substitution) and
feeds it to the agent as a normal user turn. This is exactly Claude
Code's ``commands/`` mechanism — a ``commands/*.md`` written for
Claude Code (or shipped by a plugin like everything-claude-code)
works here unchanged.

## Layout

```
<commands-dir>/
    refactor-clean.md      # /refactor-clean
    tdd.md                 # /tdd
    security-scan.md       # /security-scan
```

Optional YAML frontmatter (only ``description`` is conventional):

```yaml
---
description: Safely identify and remove dead code with verification.
argument-hint: "[path]"
---
# the rest of the file is the prompt body
```

## Search-path precedence (high → low, first match wins)

1. ``SHAREDAGENTS_COMMANDS_DIRS`` (``:``-separated env override)
2. ``<agent_home>/commands/`` — plus the runtime-shipped ``/goal`` command
   (agent name baked in), folded in at this same precedence. Only ``/goal``
   is injected here; other top-level files (``SOUL.md`` …) never become
   commands.
3. ``~/.sharedagents/commands/``
4. ``~/.claude/commands/`` — Claude Code's canonical user-level
   command dir. Mirrors the ``~/.claude/skills/`` entry in
   ``skills.py``: a command authored once in CC's layout works in
   sharedagents too, with zero duplication. Below the sharedagents-specific
   dirs above so an operator override still wins.
5. Installed plugins (``~/.sharedagents/plugins/<name>/commands/``).
6. ``commands_dirs`` in ``~/.sharedagents/config.json`` — the persistent
   way to register an external command bundle (e.g. an
   everything-claude-code checkout's ``commands/``). Deliberately low
   precedence, mirroring ``skills_dirs``: a bundle fills gaps but
   never silently shadows the operator's own commands.

## Argument substitution

* ``$ARGUMENTS`` — the full argument string.
* ``$1`` … ``$9`` / ``${1}`` … — whitespace-split positional args.

Unmatched placeholders are left verbatim (a body that legitimately
contains ``$1`` shell syntax still works when the command takes no
args).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# Runtime-shipped per-agent ``/goal`` command (not read from disk).
# Injected at the ``<agent_home>/commands`` precedence slot.


@dataclass(frozen=True)
class FileCommand:
    """One Markdown slash command loaded from disk."""

    name: str
    description: str
    body: str  # the prompt body (post-frontmatter)
    source: Path
    argument_hint: str = ""

    @property
    def short_description(self) -> str:
        d = (self.description or "").strip()
        cut = re.split(r"(?<=[.!?])\s+", d, maxsplit=1)
        return (cut[0] if cut else d)[:120]


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a command file into (flat frontmatter dict, body).

    Command frontmatter is simple — flat ``key: value`` scalars
    (``description``, ``argument-hint``, …). No nested maps / block
    scalars are used by Claude Code commands, so a flat parser is
    enough; unparseable lines are skipped rather than raising.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw, body = m.group(1), text[m.end() :]
    data: dict[str, str] = {}
    for line in raw.splitlines():
        mk = re.match(r"^\s*([A-Za-z_][\w.-]*)\s*:\s*(.*)$", line)
        if not mk:
            continue
        val = mk.group(2).strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        data[mk.group(1)] = val
    return data, body


def parse_command_md(path: Path) -> FileCommand | None:
    """Parse one ``<name>.md`` into a :class:`FileCommand` (None on error)."""
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    fm, body = _parse_frontmatter(text)
    name = path.stem.strip().lower()
    if not name:
        return None
    return FileCommand(
        name=name,
        description=str(fm.get("description") or "").strip(),
        body=body.strip(),
        source=path,
        argument_hint=str(fm.get("argument-hint") or fm.get("argument_hint") or ""),
    )


def _config_commands_dirs() -> list[Path]:
    """Read the global config's ``commands_dirs`` list ([] on failure)."""
    try:
        from sharedagents_agent.agent import load_global_config

        cfg = load_global_config() or {}
    except Exception:  # noqa: BLE001
        return []
    raw = cfg.get("commands_dirs")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[Path] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(Path(item.strip()).expanduser())
    return out


def command_search_paths(agent_home: Path | None = None) -> list[Path]:
    """Resolve the command search chain (high → low precedence)."""
    chain: list[Path] = []
    extra = os.environ.get("SHAREDAGENTS_COMMANDS_DIRS")
    if extra:
        for p in extra.split(":"):
            p = p.strip()
            if p:
                chain.append(Path(p).expanduser())
    if agent_home is not None:
        chain.append(Path(agent_home) / "commands")
    chain.append(Path.home() / ".sharedagents" / "commands")
    # Claude Code's canonical user-level command dir. A ``commands/*.md``
    # the operator authored for Claude Code (or that a CC plugin dropped
    # into ``~/.claude/commands/``) shows up in sharedagents too without
    # duplication — same logic that ``skills.py`` applies to
    # ``~/.claude/skills/``. Lower precedence than the sharedagents-specific
    # dirs above so an explicit operator override still wins.
    chain.append(Path.home() / ".claude" / "commands")
    # Installed plugins (~/.sharedagents/plugins/<name>/commands).
    try:
        from sharedagents_agent.plugins import plugin_command_dirs

        chain.extend(plugin_command_dirs())
    except Exception:  # noqa: BLE001
        pass
    chain.extend(_config_commands_dirs())
    # De-dupe, preserve order.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in chain:
        rp = p.resolve(strict=False)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def _hardcoded_root_commands(agent_home: Path | None) -> list[FileCommand]:
    """Runtime-shipped ``/goal`` for the agent named by ``agent_home``."""
    if agent_home is None:
        return []
    from sharedagents_agent.agent import hardcoded_goal_markdown

    agent_name = Path(agent_home).name
    text = hardcoded_goal_markdown(agent_name=agent_name)
    fm, body = _parse_frontmatter(text)
    return [
        FileCommand(
            name="goal",
            description=str(fm.get("description") or "").strip(),
            body=body.strip(),
            source=Path(agent_home) / ".sharedagents-hardcoded" / "GOAL.md",
            argument_hint=str(fm.get("argument-hint") or fm.get("argument_hint") or "").strip(),
        )
    ]


def discover_commands(
    agent_home: Path | None = None,
) -> list[FileCommand]:
    """Every file command across the search chain, name-deduplicated.

    First match per name wins (highest precedence). Result is sorted
    by name for stable rendering.

    The bundled hardcoded commands (e.g. ``GOAL.md``) are a LOWEST-precedence
    fallback: an operator command of the same name at ANY search tier wins and
    is never silently shadowed by the bundle. The bundle only
    supplies a command the operator has not provided anywhere.
    """
    by_name: dict[str, FileCommand] = {}
    root_cmds = _hardcoded_root_commands(agent_home)
    for root in command_search_paths(agent_home):
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.suffix.lower() != ".md" or not child.is_file():
                continue
            cmd = parse_command_md(child)
            if cmd is None or cmd.name in by_name:
                continue
            by_name[cmd.name] = cmd
    # Fold the bundled commands in LAST, as a pure fallback,
    # so no bundled command can shadow an operator command of the same name at
    # any tier. (Previously injected at the <agent_home>/commands slot, where a
    # bundled GOAL.md silently shadowed an operator goal.md at or below it.)
    for cmd in root_cmds:
        by_name.setdefault(cmd.name, cmd)
    return sorted(by_name.values(), key=lambda c: c.name)


def find_command(
    name: str,
    agent_home: Path | None = None,
) -> FileCommand | None:
    """Look up one file command by name (case-insensitive)."""
    if not name:
        return None
    want = name.strip().lower().lstrip("/")
    for c in discover_commands(agent_home):
        if c.name == want:
            return c
    return None


_ARG_TOKEN_RE = re.compile(r"\$\{?(\d{1,2}|ARGUMENTS)\}?")


def expand_command(cmd: FileCommand, args: str = "") -> str:
    """Expand a command's body into the prompt text to send the agent.

    ``$ARGUMENTS`` → the full argument string; ``$1``…``$9`` /
    ``${1}`` → whitespace-split positional args. Placeholders with no
    matching argument are left verbatim so a body that legitimately
    contains shell ``$1`` syntax survives a no-arg invocation.
    """
    args = (args or "").strip()
    positional = args.split()

    def _sub(m: re.Match[str]) -> str:
        tok = m.group(1)
        if tok == "ARGUMENTS":
            return args
        idx = int(tok)
        if 1 <= idx <= len(positional):
            return positional[idx - 1]
        return m.group(0)  # leave verbatim

    body = _ARG_TOKEN_RE.sub(_sub, cmd.body)
    # If the body never referenced $ARGUMENTS but the operator passed
    # args, append them so a plain prompt-style command still receives
    # the input (Claude Code's behaviour for arg-less command bodies).
    if args and "$ARGUMENTS" not in cmd.body and not _ARG_TOKEN_RE.search(cmd.body):
        body = f"{body}\n\n{args}"
    return body
