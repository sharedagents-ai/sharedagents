"""Hermes-style capture of successful delegation prompts as orchestrator skills.

Patterns live under ``<main_agent_home>/skills/delegation-<slug>/SKILL.md``.
Upsert deduplicates by subagent_type + title similarity so the skill corpus
does not bloat with near-identical entries.

Upsert first attempts to merge into an existing similar pattern
(same subagent_type, word-similarity above a fixed threshold)
by overwriting it in place; only when no similar pattern
exists is a new skill file created, and creation is
refused once a fixed total-pattern cap is reached.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MAX_DELEGATION_PATTERNS = 40
PATTERN_TAG = "delegation-pattern"
_SIMILARITY_THRESHOLD = 0.72


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _word_similarity(a: str, b: str) -> float:
    wa = set(_normalize(a).split())
    wb = set(_normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _slugify(text: str, *, max_len: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", _normalize(text))
    s = s.strip("-")
    return s[:max_len].rstrip("-") or "pattern"


def _skills_root(agent_home: Path) -> Path:
    return Path(agent_home) / "skills"


def _iter_delegation_skills(agent_home: Path) -> list[Path]:
    root = _skills_root(agent_home)
    if not root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if skill_md.is_file() and child.name.startswith("delegation-"):
            out.append(skill_md)
    return out


def _parse_frontmatter_name(text: str) -> str:
    m = re.match(r"\A---\s*\n.*?^name:\s*(.+?)\s*$", text, re.MULTILINE | re.DOTALL)
    return m.group(1).strip().strip('"').strip("'") if m else ""


def _parse_subagent_type(text: str) -> str:
    m = re.search(
        r"subagent_type:\s*(.+?)\s*$",
        text,
        re.MULTILINE,
    )
    return m.group(1).strip().strip('"').strip("'") if m else ""


def find_similar_pattern(
    agent_home: Path,
    *,
    subagent_type: str,
    title: str,
) -> Path | None:
    st = _normalize(subagent_type)
    for path in _iter_delegation_skills(agent_home):
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _parse_subagent_type(body) != st:
            continue
        existing_title = _parse_frontmatter_name(body) or path.parent.name
        if _word_similarity(title, existing_title) >= _SIMILARITY_THRESHOLD:
            return path
        if _word_similarity(title, body[:400]) >= _SIMILARITY_THRESHOLD:
            return path
    return None


def upsert_delegation_pattern(
    agent_home: Path,
    *,
    subagent_type: str,
    title: str,
    prompt_template: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Create or merge a delegation-pattern skill. Returns action metadata."""
    agent_home = Path(agent_home)
    subagent_type = (subagent_type or "").strip().lower()
    title = (title or "").strip()
    prompt_template = (prompt_template or "").strip()
    if not subagent_type or not title or not prompt_template:
        return {
            "ok": False,
            "reason": "subagent_type, title, and prompt_template are required",
        }

    existing = find_similar_pattern(
        agent_home,
        subagent_type=subagent_type,
        title=title,
    )
    extra_tags = [t.strip() for t in (tags or []) if t and t.strip()]
    tag_line = ", ".join(dict.fromkeys([PATTERN_TAG, subagent_type, *extra_tags]))
    skill_name = f"delegation-{_slugify(f'{subagent_type}-{title}')}"
    body = f"""---
name: {skill_name}
description: "Reusable {subagent_type} delegation prompt — {title[:120]}"
metadata:
  sharedagents:
    tags: [{tag_line}]
    subagent_type: {subagent_type}
---

# Delegation pattern — {title}

**Subagent:** ``Agent(subagent_type="{subagent_type}", ...)``

## Prompt template

```
{prompt_template}
```

## When to reuse

- Same objective class as the captured success (adjust paths/hostnames inline).
- Load via ``Skill`` before delegating a similar task — do not duplicate
  near-identical prompts across turns.

## Anti-bloat

One pattern per objective class. Merge edits into this file instead of creating
a sibling skill unless the workflow genuinely diverges.
"""

    if existing is not None:
        existing.write_text(body, encoding="utf-8")
        return {
            "ok": True,
            "action": "updated",
            "path": str(existing),
            "skill_name": _parse_frontmatter_name(body) or skill_name,
        }

    skills = _iter_delegation_skills(agent_home)
    if len(skills) >= MAX_DELEGATION_PATTERNS:
        return {
            "ok": False,
            "reason": (
                f"delegation pattern cap reached ({MAX_DELEGATION_PATTERNS}) — "
                "merge into an existing pattern or prune ~/.sharedagents/agents/jarvis/skills/"
            ),
        }

    dest_dir = _skills_root(agent_home) / skill_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "SKILL.md"
    dest.write_text(body, encoding="utf-8")
    return {
        "ok": True,
        "action": "created",
        "path": str(dest),
        "skill_name": skill_name,
    }
