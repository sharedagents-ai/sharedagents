"""Plan-step extraction.

Pulled out of ``sharedagents_tui.app`` so the parser can be unit-tested
without importing Textual/rich. Used by the TUI's bottom-of-screen
plan-checklist widget (``#plan-checklist``) to render a tickable
list of todos extracted from the bridge's ``plan_pass`` tool body.

Accepts:

* Numbered lists (``1.`` / ``1)``).
* Bulleted lists (``-`` / ``*`` / ``•``).
* JSON-shaped ``{"subtasks": [...]}`` / ``{"steps": [...]}`` /
  ``{"plan": [...]}`` / ``{"todos": [...]}``.
* Plain-text fallback — each non-blank line, capped at 12.

Returns a flat list of trimmed step strings; an empty list when the
body has no recognizable plan structure.
"""

from __future__ import annotations

import json
import re

_BULLET_RE = re.compile(r"^(?:[-*•]|\d+[\.)])\s+(.*)$")
_PLAN_LIST_KEYS = ("subtasks", "steps", "plan", "todos")
_MAX_STEPS = 12


def extract_plan_steps(body: str) -> list[str]:
    body = (body or "").strip()
    if not body:
        return []

    # JSON path — the plan classifier sometimes emits structured.
    try:
        j = json.loads(body)
    except (ValueError, TypeError):
        j = None
    if isinstance(j, dict):
        for key in _PLAN_LIST_KEYS:
            v = j.get(key)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]

    # Bullet path.
    bullets: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _BULLET_RE.match(line)
        if m:
            bullets.append(m.group(1).strip())
    if bullets:
        return bullets[:_MAX_STEPS]

    # Plain-text fallback — one step per non-blank line, capped.
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return lines[:_MAX_STEPS]


__all__ = ["extract_plan_steps"]
