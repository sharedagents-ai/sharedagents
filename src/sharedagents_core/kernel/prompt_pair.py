"""Prompt-pair parsing — the leaf half of ``system_prompt``.

A "prompt pair" is one markdown file carrying a ``<!-- system -->`` section and
a ``<!-- user -->`` section (used by the memory synthesizer's editable
``MEMORY-SYNTH.md`` and the shipped default template). Parsing them is pure
string work with no sharedagents dependency, so it lives here in the kernel leaf —
the memory engine imports ``load_prompt_pair`` / ``parse_prompt_pair_text``
downward from here without dragging in ``system_prompt`` (which lazily imports
the tool registry and agent module for its *rendering* helpers).

``sharedagents_agent.system_prompt`` re-exports these two functions unchanged, so its
own callers are unaffected; only the physical home moved.
"""

from __future__ import annotations

import re
from pathlib import Path


def load_prompt_pair(path: Path) -> dict[str, str]:
    """Load a (system, user) prompt pair from one .md file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    return parse_prompt_pair_text(text)


def parse_prompt_pair_text(text: str) -> dict[str, str]:
    """Parse ``<!-- system -->`` / ``<!-- user -->`` sections from markdown."""
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []

    def _flush() -> None:
        if current is not None:
            sections[current] = "".join(buf).strip("\n")
        buf.clear()

    marker_re = re.compile(r"^\s*<!--\s*(\w+)\s*-->\s*$")
    for line in text.splitlines(keepends=True):
        m = marker_re.match(line)
        if m:
            _flush()
            current = m.group(1).lower()
            continue
        if current is not None:
            buf.append(line)
    _flush()
    return sections


__all__ = ["load_prompt_pair", "parse_prompt_pair_text"]
