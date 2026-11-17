"""Dev tab forge rail: cap Issues and Open PRs at three visible rows."""

from __future__ import annotations

FORGE_PANEL_LIMIT = 3


def cap_forge_panel(items, *, limit: int = FORGE_PANEL_LIMIT) -> tuple[list, int]:
    """Keep the Dev side rail to three issues and three PRs."""
    rows = list(items or [])
    try:
        cap = max(0, int(limit))
    except (TypeError, ValueError):
        cap = FORGE_PANEL_LIMIT
    return rows[:cap], len(rows)


def forge_limit_arg(raw) -> int:
    if raw is None or str(raw).strip() == "":
        return FORGE_PANEL_LIMIT
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return FORGE_PANEL_LIMIT


def fill_forge_panel(primary, extra, *, limit: int = FORGE_PANEL_LIMIT) -> tuple[list, int]:
    """Prefer open rows, then recent closed ones; total is the deduplicated union."""
    seen: set = set()
    rows: list = []
    for row in list(primary or []) + list(extra or []):
        if not isinstance(row, dict):
            continue
        key = row.get("number")
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    try:
        cap = max(0, int(limit))
    except (TypeError, ValueError):
        cap = FORGE_PANEL_LIMIT
    return rows[:cap], len(rows)
