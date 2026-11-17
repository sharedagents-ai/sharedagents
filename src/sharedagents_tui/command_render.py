"""TUI-only render helpers for shared command-runtime envelopes.

The Web UI renders one panel per ``kind`` (keyvals / list / table /
checklist / detail / text / error). The TUI used to drop ``kind`` on
the ACP envelope and squash every result into a 200-char footer line.
These helpers format the same envelope into a transcript card. They
do not change command_runtime or any Web UI path.
"""

from __future__ import annotations

from typing import Any


def confirm_copy_for(text: str) -> tuple[str, str] | None:
    """Return ``(title, prompt)`` when *text* is an irreversible slash.

    Uses the shared classifier so TUI confirm copy stays aligned with
    the Web UI re-challenge. ``None`` means execute immediately.
    ``/stop`` is a TUI-local graceful alias and is never passed here.
    """
    try:
        from sharedagents_command.command_runtime import (
            _irreversible_dialog,
            command_is_irreversible,
        )
        from sharedagents_command.commands import parse_command
    except Exception:  # noqa: BLE001
        return None
    parsed = parse_command(text)
    if parsed is None:
        return None
    if not command_is_irreversible(parsed.name, parsed.arg):
        return None
    obj, what = _irreversible_dialog(parsed.name, parsed.arg)
    return (f"Confirm /{parsed.name}", f"{what} This cannot be undone.")


def _as_pairs(data: dict) -> list[tuple[str, str]]:
    raw = data.get("pairs") or []
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict):
            key = item.get("key") or item.get("name") or item.get("id") or ""
            val = item.get("value") or item.get("text") or ""
            out.append((str(key), str(val)))
    return out


def _item_title(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)
    return str(item.get("title") or item.get("id") or item.get("name") or "")


def format_kind_body(kind: str, text: str, data: dict | None = None) -> str:
    """Format a command result into a monospace transcript body.

    Prefers structured ``data`` (same fields the Web UI panel reads).
    Falls back to ``text`` when the payload has no rows.
    """
    data = data if isinstance(data, dict) else {}
    kind = (kind or "text").strip().lower() or "text"
    fallback = (text or "").rstrip()

    if kind == "keyvals":
        pairs = _as_pairs(data)
        if pairs:
            width = max(len(k) for k, _ in pairs)
            width = min(width, 24)
            lines = [f"{k.ljust(width)}  {v}" for k, v in pairs]
            return "\n".join(lines)
        return fallback

    if kind == "list":
        items = data.get("items") or []
        if items:
            lines: list[str] = []
            for item in items:
                if not isinstance(item, dict):
                    lines.append(f"• {item}")
                    continue
                title = _item_title(item)
                sub = str(item.get("subtitle") or "")
                body = str(item.get("body") or "")
                badges = [str(b) for b in (item.get("badges") or []) if b]
                head = title
                if sub:
                    head = f"{title}  —  {sub}" if title else sub
                if badges:
                    head = f"{head}  [{', '.join(badges)}]" if head else f"[{', '.join(badges)}]"
                lines.append(f"• {head}" if head else "•")
                if body:
                    lines.append(f"    {body}")
            return "\n".join(lines) if lines else "empty"
        return fallback or "empty"

    if kind == "table":
        cols = [str(c) for c in (data.get("columns") or [])]
        rows = data.get("rows") or []
        if cols and rows:
            widths = [len(c) for c in cols]
            norm_rows: list[list[str]] = []
            for row in rows:
                if isinstance(row, dict):
                    cells = [str(row.get(c, "")) for c in cols]
                elif isinstance(row, (list, tuple)):
                    cells = [str(x) for x in row]
                    if len(cells) < len(cols):
                        cells.extend([""] * (len(cols) - len(cells)))
                    cells = cells[: len(cols)]
                else:
                    cells = [str(row)] + [""] * (len(cols) - 1)
                norm_rows.append(cells)
                for i, cell in enumerate(cells):
                    widths[i] = min(max(widths[i], len(cell)), 28)
            header = "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
            rule = "  ".join("-" * widths[i] for i in range(len(cols)))
            body_lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in norm_rows]
            return "\n".join([header, rule, *body_lines])
        return fallback

    if kind == "checklist":
        items = data.get("items") or []
        if items:
            lines = []
            for item in items:
                if isinstance(item, dict):
                    done = bool(
                        item.get("done")
                        or item.get("completed")
                        or str(item.get("status") or "").lower() in {"done", "completed", "complete"}
                    )
                    title = _item_title(item) or str(item.get("text") or "")
                else:
                    done = False
                    title = str(item)
                mark = "x" if done else " "
                lines.append(f"[{mark}] {title}")
            return "\n".join(lines)
        return fallback or "empty"

    return fallback or ("(no output)" if kind != "error" else "error")


def card_title(command: str, kind: str, data: dict | None = None) -> str:
    """Header line for a command-result card (Web UI panel title analogue)."""
    data = data if isinstance(data, dict) else {}
    title = str(data.get("title") or "").strip()
    if title:
        return title
    cmd = (command or "").strip().lstrip("/") or kind or "result"
    return f"/{cmd}" if not cmd.startswith("/") else cmd
