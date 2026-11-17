"""Minimal 5-field cron evaluator (no external dependency).

Supports the standard 5-field syntax ``minute hour dom month dow`` with
``*``, ``*/N`` steps, ``N-M`` ranges, ``N-M/S`` stepped ranges, and
``,`` lists on every field. ``0`` and ``7`` both mean Sunday (POSIX
convention). A 6th (seconds) field is rejected — scheduled work in this
project has 1-minute granularity.

Used by the loop/cron job machinery (``loop_registry``) to compute the
next fire time for ``CronCreate``-style jobs and cron-backed loops.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_MAX_SCAN_YEARS = 5


def _parse_field(token: str, lo: int, hi: int) -> list[int]:
    """Expand one cron field token into the sorted set of matching values."""
    token = token.strip()
    if not token:
        raise ValueError("cron fields and list members must not be empty")
    if token == "*":
        return list(range(lo, hi + 1))
    # ``*/N`` — every Nth value.
    if token.startswith("*/"):
        step = _int_token(token[2:], lo, hi)
        if step < 1:
            raise ValueError(f"step must be >= 1: {token!r}")
        return list(range(lo, hi + 1, step))
    # Comma lists.
    if "," in token:
        out: list[int] = []
        parts = token.split(",")
        if any(not part.strip() for part in parts):
            raise ValueError(f"empty cron list member: {token!r}")
        for part in parts:
            out.extend(_parse_field(part, lo, hi))
        return sorted(set(out))
    # ``N-M/S`` stepped range (or ``N-M``).
    if "-" in token:
        rest = token
        step = 1
        if "/" in rest:
            rest, _, step_tok = rest.partition("/")
            step = _int_token(step_tok, lo, hi)
        a_str, _, b_str = rest.partition("-")
        a = _int_token(a_str, lo, hi)
        b = _int_token(b_str, lo, hi)
        if a > b:
            raise ValueError(f"range start exceeds end: {token!r}")
        if step < 1:
            raise ValueError(f"step must be >= 1: {token!r}")
        return list(range(a, b + 1, step))
    return [_int_token(token, lo, hi)]


def _int_token(token: str, lo: int, hi: int) -> int:
    if not token.strip().isdigit():
        raise ValueError(f"invalid cron value: {token!r}")
    v = int(token)
    if not (lo <= v <= hi):
        raise ValueError(f"cron value {v} out of range [{lo}, {hi}] (token {token!r})")
    return v


def parse_cron(expr: str) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """Parse a 5-field cron expression into per-field value sets.

    Raises :class:`ValueError` on malformed expressions (bad token
    count, out-of-range values, unsupported syntax).
    """
    fields = [f.strip() for f in str(expr or "").split()]
    if len(fields) != 5:
        raise ValueError(
            f"cron expression must have 5 fields (minute hour day month weekday), got {len(fields)}: {expr!r}"
        )
    minutes = _parse_field(fields[0], 0, 59)
    hours = _parse_field(fields[1], 0, 23)
    doms = _parse_field(fields[2], 1, 31)
    months = _parse_field(fields[3], 1, 12)
    dows = _parse_field(fields[4], 0, 7)
    if 7 in dows:
        dows = sorted((set(dows) - {7}) | {0})
    return minutes, hours, doms, months, dows


def _matches(
    dt: _dt.datetime,
    minutes,
    hours,
    doms,
    months,
    dows,
    *,
    dom_wildcard: bool,
    dow_wildcard: bool,
) -> bool:
    if dt.minute not in minutes or dt.hour not in hours or dt.month not in months:
        return False
    dom_match = dt.day in doms
    # Python numbers Monday as zero; POSIX cron numbers Sunday as zero.
    dow_match = ((dt.weekday() + 1) % 7) in dows
    if dom_wildcard or dow_wildcard:
        # If either day field starts with ``*``, both fields must match. The
        # wildcard side is naturally true, so the restricted side governs.
        return dom_match and dow_match
    # Vixie/POSIX cron: two restricted day fields are alternatives.
    return dom_match or dow_match


def _local_timezone() -> _dt.tzinfo:
    """Resolve the host's named local zone so DST transitions are retained."""
    configured = os.environ.get("TZ", "").strip()
    if configured:
        try:
            return ZoneInfo(configured)
        except ZoneInfoNotFoundError:
            pass
    try:
        resolved = str(Path("/etc/localtime").resolve())
        marker = "/zoneinfo/"
        if marker in resolved:
            return ZoneInfo(resolved.split(marker, 1)[1])
    except (OSError, ZoneInfoNotFoundError):
        pass
    try:
        configured = Path("/etc/timezone").read_text(encoding="utf-8").strip()
        if configured:
            return ZoneInfo(configured)
    except (OSError, ZoneInfoNotFoundError):
        pass
    return _dt.datetime.now().astimezone().tzinfo or _dt.timezone.utc


def cron_next(
    expr: str,
    after: _dt.datetime,
    *,
    timezone: _dt.tzinfo | None = None,
) -> _dt.datetime | None:
    """Return the next datetime matching ``expr`` strictly after ``after``.

    Returns ``None`` when no match exists within a bounded scan window
    (5 years) — callers treat that as an unreachable schedule. Cron fields
    are evaluated in the host's local wall-clock zone by default. The scan
    itself advances real UTC instants, which correctly skips nonexistent
    spring-forward times and visits both fall-back folds.
    """
    minutes, hours, doms, months, dows = parse_cron(expr)
    fields = str(expr or "").split()
    dom_wildcard = fields[2].startswith("*")
    dow_wildcard = fields[4].startswith("*")
    schedule_tz = timezone or _local_timezone()
    if after.tzinfo is None:
        after = after.replace(tzinfo=schedule_tz)
    return_tz = after.tzinfo or schedule_tz
    cursor = after.astimezone(_dt.timezone.utc).replace(second=0, microsecond=0) + _dt.timedelta(minutes=1)
    horizon = cursor + _dt.timedelta(days=365 * _MAX_SCAN_YEARS)
    while cursor <= horizon:
        local = cursor.astimezone(schedule_tz)
        if _matches(
            local,
            minutes,
            hours,
            doms,
            months,
            dows,
            dom_wildcard=dom_wildcard,
            dow_wildcard=dow_wildcard,
        ):
            return cursor.astimezone(return_tz)
        cursor += _dt.timedelta(minutes=1)
    return None
