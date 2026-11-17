"""Light global-config accessors + the shipped memory-synth template.

These three accessors — ``global_config_path``, ``load_global_config`` and
``memory_synth_template_text`` — are tiny (read ``~/.sharedagents/config.json``,
return a module constant) but previously lived in the heavy ``agent`` module
(2000+ lines). The memory engine borrowed them lazily, which pinned memory to
``sharedagents_agent.agent``. They are leaf primitives, so they live here in the
kernel; ``sharedagents_agent.agent`` re-exports them unchanged for its own callers,
and the memory engine imports them downward from here — removing the engine's
only reason to touch ``agent.py``.

``_sharedagents_root`` mirrors ``sharedagents_agent.agent.sharedagents_root`` byte-for-byte so
``global_config_path`` resolves the config path identically to before the move.
(The layout module has a near-identical ``sharedagents_root``; the copy here keeps
this a leaf and the re-export behaviourally identical to agent.py's original.)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _sharedagents_root() -> Path:
    """Top of the sharedagents runtime tree (mirror of ``agent.sharedagents_root``).

    Resolution order: ``$SHAREDAGENTS_AGENTS_HOME`` (root = its parent) →
    ``$SHAREDAGENTS_HOME`` → ``~/.sharedagents``.
    """
    base = os.environ.get("SHAREDAGENTS_AGENTS_HOME")
    if base:
        return Path(base).parent
    home = os.environ.get("SHAREDAGENTS_HOME")
    if home:
        return Path(home).expanduser()
    return Path.home() / ".sharedagents"


def global_config_path() -> Path:
    """Where shared (across-all-agents) settings live."""
    return _sharedagents_root() / "config.json"


def load_global_config() -> dict[str, Any]:
    """Read `~/.sharedagents/config.json`; return {} if absent or unreadable."""
    p = global_config_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# Memory synthesizer prompt — operator-editable. Loaded fresh on every
# run by ``sharedagents_mcp.memory.engine.memory_synth.synth_memory``. The user section's
# ``{memory}`` / ``{dailies}`` / ``{n_dailies}`` slots are filled at
# call time. The system section steers the model toward emitting four
# canonical ``<!-- sharedagents:section ... -->`` blocks; the synth refuses
# to write MEMORY.md when those markers are absent (see
# memory_synth.py: ``synth output missing section markers; refusing to
# write``).
_MEMORY_SYNTH_TEMPLATE = """\
<!-- system -->
You curate the agent's long-term memory by distilling raw daily logs
into a stable, compact MEMORY.md.

OUTPUT FORMAT — STRICT. Reply with ONLY the new MEMORY.md body, using
exactly these four section markers in this order:

    <!-- sharedagents:section project-state -->
    Concrete state of the project / codebase / open work — what's
    real today. Bullet points. Drop entries that are no longer true.

    <!-- sharedagents:section decisions -->
    Decisions and chosen approaches with a one-line "why". Outcomes
    of explicit tradeoffs the user / agent locked in.

    <!-- sharedagents:section context -->
    Stable user / domain / preference context that's worth carrying
    across sessions. Names, roles, working style, recurring pet
    peeves. Light on details that rot fast.

    <!-- sharedagents:section lessons -->
    Lessons learned the hard way — bugs that recurred, gotchas, rules
    of thumb the agent should remember. Each entry one line.

Rules:
- One line per bullet, no walls of text.
- Use ``- `` bullets. No nested bullets.
- Empty section is allowed: write
  ``_(empty — perishable context belongs in dailies, not here)_``.
- Do NOT include any prose outside the four marker blocks. No
  preamble, no ``# MEMORY.md`` header, no Markdown code fences.
- Keep durable principles; drop one-off task chatter that already
  served its purpose.
- Pruning is encouraged. If MEMORY.md has stale lines that the
  dailies contradict, drop them.

If a daily mentions a fact already present in MEMORY.md, do not
duplicate — keep the existing line as-is unless the daily updates it.
<!-- user -->
Current MEMORY.md
=================
{memory}

Recent dailies ({n_dailies} most recent, newest first)
======================================================
{dailies}

Rewrite MEMORY.md per the rules above. Reply with ONLY the new body.
"""


def memory_synth_template_text() -> str:
    """Shipped MEMORY-SYNTH prompt (used when agent home has no override)."""
    return _MEMORY_SYNTH_TEMPLATE


__all__ = [
    "global_config_path",
    "load_global_config",
    "memory_synth_template_text",
    "_MEMORY_SYNTH_TEMPLATE",
]
