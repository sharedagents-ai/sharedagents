"""Xiaomi MiMo Token Plan usage, estimated from sharedagents's own token ledger.

Xiaomi publishes NO API for plan usage: the only official view is the console's
Token Plan progress bar (https://mimo.mi.com/docs/en-US/tokenplan/Token%20Plan/
subscription). So the card is computed locally — every MiMo turn sharedagents makes
is already in the usage ledger (``turn_costs``) with its token buckets, and the
plan meters *credits* at a published per-token rate.

Source of the rates (official FAQ "Usage and Quota"):
https://mimo.mi.com/static/docs/quick-start/faq/token-plan/Usage&Quota.md

* credits per token differ by model and by bucket (cache hit / cache miss /
  output);
* usage between 00:00 and 08:00 Beijing time (UTC 16:00-24:00) burns at 0.8x;
* TTS is free and consumes no credits.

This is an ESTIMATE of sharedagents's own consumption: calls made by other tools
(Claude Code, MiMoCode, the console playground) with the same plan key are not
in this ledger, and per-turn timestamps stand in for per-request ones when
applying the off-peak coefficient.

The plan itself is operator-declared in ``~/.sharedagents/config.json``::

    "subscriptions": {"mimo": {
        "plan": "Standard", "credits": 11000000000,
        "period_end": "2026-11-02T23:59:59Z", "auto_renew": true}}
"""

from __future__ import annotations

import calendar
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: model id -> (cache-hit, cache-miss, output) credits PER TOKEN.
CREDIT_RATES: dict[str, tuple[float, float, float]] = {
    "mimo-v2.6-pro": (2.5, 300.0, 600.0),
    "mimo-v2.5-pro": (2.5, 300.0, 600.0),
    "mimo-v2.6-flash": (2.0, 100.0, 200.0),
    "mimo-v2.5": (2.0, 100.0, 200.0),
}
#: A model missing from the table is charged at the dearest published rate, so
#: the estimate errs toward "more used" rather than silently under-counting.
_FALLBACK_RATES = CREDIT_RATES["mimo-v2.6-pro"]
OFF_PEAK_COEFFICIENT = 0.8
#: UTC hours [16, 24) == Beijing 00:00-08:00.
_OFF_PEAK_UTC_HOURS = range(16, 24)
#: Ledger ``provider`` value for rows served by the Xiaomi plan host. OpenRouter
#: rows for the same models carry ``openrouter`` and are NOT plan usage.
LEDGER_PROVIDER = "xiaomi"
#: Env var holding the Token Plan key. A ``models[]`` row served with it is a
#: plan model; rows on the pay-per-use key or on OpenRouter are not.
PLAN_KEY_ENV = "XIAOMI_SUB_KEY"


class MimoPlanNotConfigured(ValueError):
    """``subscriptions.mimo`` is missing or incomplete in config.json."""


def credits_for_turn(
    model: str,
    *,
    fresh_input: int,
    cached_read: int,
    thinking: int,
    output: int,
    when: datetime,
) -> float:
    """Credits one ledger turn burned.

    Ledger semantics (``usage_pricing.normalize_usage``): ``input_tokens`` is the
    uncached remainder (cache MISS), ``cached_read_tokens`` the cache HITS, and
    ``thinking`` + ``output`` are disjoint generated tokens — both bill at the
    output rate.
    """
    hit, miss, out = CREDIT_RATES.get(model.strip().lower(), _FALLBACK_RATES)
    credits = fresh_input * miss + cached_read * hit + (thinking + output) * out
    if when.astimezone(timezone.utc).hour in _OFF_PEAK_UTC_HOURS:
        credits *= OFF_PEAK_COEFFICIENT
    return credits


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _shift_months(stamp: datetime, months: int) -> datetime:
    index = stamp.year * 12 + (stamp.month - 1) + months
    year, month = divmod(index, 12)
    month += 1
    day = min(stamp.day, calendar.monthrange(year, month)[1])
    return stamp.replace(year=year, month=month, day=day)


def plan_period(cfg: dict[str, Any], now: datetime) -> tuple[datetime, datetime]:
    """``(start, end)`` of the billing period containing ``now``.

    ``period_end`` is the plan's own expiry. The start is the same calendar day
    one month earlier at 00:00 UTC (a purchase day opens at midnight UTC and the
    plan runs to 23:59:59 UTC a month later). With ``auto_renew`` a lapsed
    period rolls forward a month at a time — credits reset on renewal.
    """
    end = _parse_utc(cfg.get("period_end"))
    if end is None:
        raise MimoPlanNotConfigured("subscriptions.mimo.period_end is missing or not an ISO timestamp")
    if cfg.get("period_start"):
        start = _parse_utc(cfg["period_start"])
        if start is None:
            raise MimoPlanNotConfigured("subscriptions.mimo.period_start is not an ISO timestamp")
    else:
        start = _shift_months(end, -1).replace(hour=0, minute=0, second=0, microsecond=0)
    if cfg.get("auto_renew", True):
        span = 0
        while end < now and span < 120:
            start, end = _shift_months(start, 1), _shift_months(end, 1)
            span += 1
    return start, end


def plan_config() -> dict[str, Any]:
    from sharedagents_core.kernel.config import load_global_config

    subs = load_global_config().get("subscriptions")
    cfg = subs.get("mimo") if isinstance(subs, dict) else None
    if not isinstance(cfg, dict):
        raise MimoPlanNotConfigured(
            "MiMo plan not declared — add subscriptions.mimo {plan, credits, period_end} to ~/.sharedagents/config.json"
        )
    try:
        credits = float(cfg.get("credits"))
    except (TypeError, ValueError):
        credits = 0.0
    if credits <= 0:
        raise MimoPlanNotConfigured("subscriptions.mimo.credits must be a positive number")
    return cfg


def plan_models() -> dict[str, str]:
    """``{model id: base URL}`` for every configured model the plan serves."""
    from sharedagents_core.kernel.config import load_global_config

    rows = load_global_config().get("models")
    return {
        str(row["id"]): str(row.get("baseUrl") or "").rstrip("/")
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and row.get("id") and row.get("apiKeyEnv") == PLAN_KEY_ENV
    }


def estimate_usage(
    cfg: dict[str, Any],
    *,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Credits sharedagents has burned this period, summed from ``turn_costs``."""
    now = now or datetime.now(timezone.utc)
    start, end = plan_period(cfg, now)
    if db_path is None:
        from sharedagents_core.layout import usage_ledger_db_path

        db_path = Path(usage_ledger_db_path())
    if not Path(db_path).exists():
        raise FileNotFoundError(f"usage ledger not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    try:
        rows = conn.execute(
            "SELECT model, input_tokens, cached_read_tokens, thinking_tokens, "
            "output_tokens, created_at FROM turn_costs "
            "WHERE lower(provider) = ? AND created_at >= ?",
            (LEDGER_PROVIDER, start.date().isoformat()),
        ).fetchall()
    finally:
        conn.close()
    used = 0.0
    turns = 0
    for model, fresh, cached, thinking, output, created in rows:
        when = _parse_utc(created)
        if when is None or not (start <= when <= end):
            continue
        used += credits_for_turn(
            str(model or ""),
            fresh_input=int(fresh or 0),
            cached_read=int(cached or 0),
            thinking=int(thinking or 0),
            output=int(output or 0),
            when=when,
        )
        turns += 1
    return {
        "used": round(used),
        "limit": float(cfg["credits"]),
        "start": start,
        "end": end,
        "turns": turns,
    }


def iso_z(stamp: datetime) -> str:
    return stamp.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "CREDIT_RATES",
    "LEDGER_PROVIDER",
    "MimoPlanNotConfigured",
    "PLAN_KEY_ENV",
    "credits_for_turn",
    "estimate_usage",
    "iso_z",
    "plan_config",
    "plan_models",
    "plan_period",
]
