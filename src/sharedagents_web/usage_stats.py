"""Read-only usage aggregates for the WebUI Usage dashboard.

The per-turn usage ledger is the accounting source of truth.  This module
projects it into a compact dashboard snapshot without copying usage into a
second store or scanning session JSONL files on every page refresh.
"""

from __future__ import annotations

import ipaddress
import re
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

from sharedagents_core.layout import usage_ledger_db_path
from sharedagents_model.usage_ledger import (
    _stale_native_cursor_grok,
    deflate_session_outputs,
    resolve_generation_model,
    resolve_generation_provider,
)
from sharedagents_web.router_status import load_mux_topology, model_category

# Chart/API "tokens" are generated output only. Prompt, cache, and thinking
# inflate Anthropic (and other long-context) bars without showing new text.
_TOKEN_TOTAL_SQL = """
    output_tokens
"""

_MODEL_NAME_SQL = """
    COALESCE(NULLIF(model, ''), NULLIF(model_alias, ''), 'Unknown')
"""

_PROVIDER_SQL = "usage_provider(provider, model, model_alias, agent_name)"


def _project_provider(provider: Any, model: Any, model_alias: Any, agent_name: Any) -> str:
    """Apply legacy turn attribution repairs without mutating the ledger."""
    if _stale_native_cursor_grok(
        provider=provider,
        engine="",
        model=model,
        model_alias=model_alias,
        agent_name=agent_name,
    ):
        provider = ""
    return resolve_generation_provider(
        provider=provider,
        model=model,
        model_alias=model_alias,
        agent_name=agent_name,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _range(days: int, now: datetime) -> tuple[date, date]:
    end = now.date()
    return end - timedelta(days=days - 1), end


def _empty_snapshot(days: int, now: datetime) -> dict[str, Any]:
    start, end = _range(days, now)
    return {
        "generated_at": now.isoformat(),
        "range": {
            "days": days,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": "UTC",
        },
        "totals": {
            "tokens": 0,
            "requests": 0,
            "sessions": 0,
            "active_days": 0,
            "active_streak": 0,
            "models": 0,
            "cost_pico": 0,
            "cost_usd": 0.0,
            "estimated_requests": 0,
        },
        "activity": [],
        "token_mix": {
            "input": 0,
            "cached_read": 0,
            "cached_write": 0,
            "thinking": 0,
            "output": 0,
            "cache_hit_rate": 0.0,
        },
        "models": [],
        "pace": {
            "last_hour_requests": 0,
            "last_24h_requests": 0,
            "previous_24h_requests": 0,
            "day_change_pct": 0.0,
            "last_7d_requests": 0,
            "previous_7d_requests": 0,
            "week_change_pct": 0.0,
            "average_tokens_per_request": 0.0,
        },
        "breakdown": {"hourly": [], "daily": []},
        "baseline": {"dated_hourly": {}, "dated_daily": {}},
    }


def _active_streak(raw_days: list[str], today: date) -> int:
    """Consecutive active UTC days ending today (or yesterday).

    Allowing yesterday as the anchor keeps an evening dashboard from showing a
    broken streak merely because no request has run yet today.  Older activity
    is historical rather than an active streak.
    """

    parsed: set[date] = set()
    for value in raw_days:
        try:
            parsed.add(date.fromisoformat(value))
        except (TypeError, ValueError):
            continue
    anchor = today if today in parsed else today - timedelta(days=1)
    if anchor not in parsed:
        return 0
    streak = 0
    while anchor in parsed:
        streak += 1
        anchor -= timedelta(days=1)
    return streak


def _change_percent(current: int, previous: int) -> float | None:
    if previous <= 0:
        return 0.0 if current <= 0 else None
    return round(((current - previous) / previous) * 100.0, 1)


def _connect_read_only(path: Path) -> sqlite3.Connection:
    # ``as_uri`` correctly quotes spaces and other URI-significant path bytes.
    con = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro",
        uri=True,
        timeout=5.0,
    )
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=5000")
    return con


def usage_snapshot(
    *,
    days: int = 365,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Return the all-time totals and recent activity used by ``/usage``.

    ``days`` controls only the activity heatmap.  The hero, KPI strip, and
    token mix remain all-time processed tokens.  Model ranking uses output
    tokens only so cache-heavy cloud prompts cannot bury local models.
    """

    days = min(730, max(28, int(days)))
    current = now or _utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    snapshot = _empty_snapshot(days, current)

    path = Path(db_path or usage_ledger_db_path())
    if not path.is_file():
        return snapshot
    start_date, end_date = _range(days, current)
    range_start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    chart_start = current - timedelta(days=365)
    # Oldest displayed daily bucket (-6d) needs twelve preceding weeks (-84d).
    history_date = current.date() - timedelta(days=90)
    history_start = datetime.combine(history_date, time.min, tzinfo=timezone.utc)
    range_end = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)
    last_hour = current - timedelta(hours=1)
    last_24h = current - timedelta(hours=24)
    previous_24h = current - timedelta(hours=48)
    last_7d = current - timedelta(days=7)
    previous_7d = current - timedelta(days=14)
    deflated_hourly: dict[tuple[str, str, str], int] = {}
    deflated_daily: dict[tuple[str, str, str], int] = {}

    try:
        with _connect_read_only(path) as con:
            con.create_function("usage_provider", 4, _project_provider)
            totals = con.execute(
                f"""
                SELECT
                    COUNT(*) AS requests,
                    COUNT(DISTINCT COALESCE(
                        NULLIF(root_session_id, ''), session_id
                    )) AS sessions,
                    COUNT(DISTINCT date(created_at)) AS active_days,
                    COUNT(DISTINCT CASE
                        WHEN COALESCE(
                            NULLIF(model, ''), NULLIF(model_alias, ''), ''
                        ) != ''
                        THEN {_MODEL_NAME_SQL}
                    END) AS models,
                    COALESCE(SUM({_TOKEN_TOTAL_SQL}), 0) AS tokens,
                    COALESCE(SUM(cost_pico), 0) AS cost_pico,
                    COALESCE(SUM(cost_estimated), 0) AS estimated_requests,
                    COALESCE(SUM(input_tokens), 0) AS input_tokens,
                    COALESCE(SUM(cached_read_tokens), 0) AS cached_read_tokens,
                    COALESCE(SUM(cached_write_tokens), 0) AS cached_write_tokens,
                    COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens
                FROM turn_costs
                """
            ).fetchone()

            active_day_rows = con.execute(
                """
                SELECT DISTINCT date(created_at) AS day
                FROM turn_costs
                WHERE date(created_at) IS NOT NULL
                ORDER BY day
                """
            ).fetchall()

            activity_rows = con.execute(
                f"""
                SELECT
                    date(created_at) AS day,
                    COUNT(*) AS requests,
                    COALESCE(SUM({_TOKEN_TOTAL_SQL}), 0) AS tokens,
                    COALESCE(SUM(cost_pico), 0) AS cost_pico
                FROM turn_costs
                WHERE julianday(created_at) >= julianday(?)
                  AND julianday(created_at) < julianday(?)
                GROUP BY date(created_at)
                ORDER BY day
                """,
                (range_start.isoformat(), range_end.isoformat()),
            ).fetchall()

            peer_hour_rows = con.execute(
                """
                SELECT strftime('%Y-%m-%dT%H:00:00Z', created_at) AS bucket,
                       COUNT(*) AS requests, COALESCE(SUM(cost_pico), 0) AS cost_pico
                FROM turn_costs
                WHERE julianday(created_at) >= julianday(?)
                  AND julianday(created_at) < julianday(?)
                GROUP BY 1
                """,
                (
                    history_start.isoformat(),
                    datetime.combine(current.date(), time.min, tzinfo=timezone.utc).isoformat(),
                ),
            ).fetchall()

            model_rows = con.execute(
                f"""
                SELECT
                    {_MODEL_NAME_SQL} AS name,
                    COUNT(*) AS requests,
                    COALESCE(SUM(output_tokens), 0) AS tokens,
                    COALESCE(SUM(cost_pico), 0) AS cost_pico
                FROM turn_costs
                GROUP BY {_MODEL_NAME_SQL}
                ORDER BY tokens DESC, requests DESC, name COLLATE NOCASE
                """
            ).fetchall()

            breakdown_hourly = con.execute(
                f"""
                SELECT
                    strftime('%Y-%m-%dT%H:00:00Z', created_at) AS bucket,
                    {_PROVIDER_SQL} AS provider,
                    {_MODEL_NAME_SQL} AS model,
                    COUNT(*) AS requests,
                    COALESCE(SUM({_TOKEN_TOTAL_SQL}), 0) AS tokens,
                    COALESCE(SUM(cost_pico), 0) AS cost_pico
                FROM turn_costs
                WHERE julianday(created_at) >= julianday(?)
                  AND julianday(created_at) <= julianday(?)
                GROUP BY 1, 2, 3
                ORDER BY 1, 2, 3
                """,
                ((current - timedelta(hours=48)).isoformat(), current.isoformat()),
            ).fetchall()

            breakdown_daily = con.execute(
                f"""
                SELECT
                    date(created_at) AS bucket,
                    {_PROVIDER_SQL} AS provider,
                    {_MODEL_NAME_SQL} AS model,
                    COUNT(*) AS requests,
                    COALESCE(SUM({_TOKEN_TOTAL_SQL}), 0) AS tokens,
                    COALESCE(SUM(cost_pico), 0) AS cost_pico
                FROM turn_costs
                WHERE julianday(created_at) >= julianday(?)
                  AND julianday(created_at) <= julianday(?)
                GROUP BY 1, 2, 3
                ORDER BY 1, 2, 3
                """,
                (chart_start.isoformat(), current.isoformat()),
            ).fetchall()

            turn_rows = con.execute(
                f"""
                SELECT
                    session_id,
                    strftime('%Y-%m-%dT%H:00:00Z', created_at) AS hour_bucket,
                    date(created_at) AS day_bucket,
                    {_PROVIDER_SQL} AS provider,
                    {_MODEL_NAME_SQL} AS model,
                    output_tokens
                FROM turn_costs
                WHERE julianday(created_at) <= julianday(?)
                ORDER BY session_id, julianday(created_at), turn_id
                """,
                (current.isoformat(),),
            ).fetchall()
            # Deflate once with recorded session context, then slice buckets.
            # Re-deflating a range slice can misclassify cumulative sessions.
            peer_tokens, daily_tokens = _deflated_token_lookups(turn_rows)
            deflated_hourly = {
                key: value
                for key, value in peer_tokens.items()
                if key[0] >= (current - timedelta(hours=48)).strftime("%Y-%m-%dT%H:00:00Z")
            }
            deflated_daily = {
                key: value for key, value in daily_tokens.items() if key[0] >= chart_start.date().isoformat()
            }
            snapshot["baseline"] = _dated_baseline(
                peer_hour_rows,
                turn_rows,
                first_hour=min((str(row["hour_bucket"]) for row in turn_rows if row["hour_bucket"]), default=None),
                current=current,
            )

            pace = con.execute(
                """
                SELECT
                    SUM(CASE WHEN julianday(created_at) >= julianday(?)
                        THEN 1 ELSE 0 END) AS last_hour,
                    SUM(CASE WHEN julianday(created_at) >= julianday(?)
                        THEN 1 ELSE 0 END) AS last_24h,
                    SUM(CASE WHEN julianday(created_at) >= julianday(?)
                                  AND julianday(created_at) < julianday(?)
                        THEN 1 ELSE 0 END) AS previous_24h,
                    SUM(CASE WHEN julianday(created_at) >= julianday(?)
                        THEN 1 ELSE 0 END) AS last_7d,
                    SUM(CASE WHEN julianday(created_at) >= julianday(?)
                                  AND julianday(created_at) < julianday(?)
                        THEN 1 ELSE 0 END) AS previous_7d
                FROM turn_costs
                """,
                (
                    last_hour.isoformat(),
                    last_24h.isoformat(),
                    previous_24h.isoformat(),
                    last_24h.isoformat(),
                    last_7d.isoformat(),
                    previous_7d.isoformat(),
                    last_7d.isoformat(),
                ),
            ).fetchone()
    except sqlite3.OperationalError as exc:
        # A fresh install may have an empty sqlite file before the ledger has
        # created its table.  Treat only that state as empty; lock/I/O failures
        # remain visible to the route instead of masquerading as zero usage.
        if "no such table: turn_costs" in str(exc).lower():
            return snapshot
        raise

    tokens = int(totals["tokens"] or 0)
    requests = int(totals["requests"] or 0)
    cost_pico = int(totals["cost_pico"] or 0)
    input_tokens = int(totals["input_tokens"] or 0)
    cached_read = int(totals["cached_read_tokens"] or 0)
    cache_denominator = input_tokens + cached_read

    snapshot["totals"] = {
        "tokens": tokens,
        "requests": requests,
        "sessions": int(totals["sessions"] or 0),
        "active_days": int(totals["active_days"] or 0),
        "active_streak": _active_streak([str(row["day"]) for row in active_day_rows], current.date()),
        "models": int(totals["models"] or 0),
        "cost_pico": cost_pico,
        "cost_usd": round(cost_pico / 1_000_000_000_000, 6),
        "estimated_requests": int(totals["estimated_requests"] or 0),
    }
    snapshot["activity"] = [
        {
            "date": str(row["day"]),
            "tokens": int(row["tokens"] or 0),
            "requests": int(row["requests"] or 0),
            "cost_pico": int(row["cost_pico"] or 0),
        }
        for row in activity_rows
        if row["day"]
    ]
    snapshot["token_mix"] = {
        "input": input_tokens,
        "cached_read": cached_read,
        "cached_write": int(totals["cached_write_tokens"] or 0),
        "thinking": int(totals["thinking_tokens"] or 0),
        "output": int(totals["output_tokens"] or 0),
        "cache_hit_rate": round((cached_read / cache_denominator) * 100.0, 1) if cache_denominator else 0.0,
    }
    snapshot["models"] = _merge_model_totals(
        [
            {
                "name": str(row["name"] or "Unknown"),
                "requests": int(row["requests"] or 0),
                "tokens": int(row["tokens"] or 0),
                "cost_pico": int(row["cost_pico"] or 0),
            }
            for row in model_rows
        ]
    )

    last_hour_count = int(pace["last_hour"] or 0)
    last_24h_count = int(pace["last_24h"] or 0)
    previous_24h_count = int(pace["previous_24h"] or 0)
    last_7d_count = int(pace["last_7d"] or 0)
    previous_7d_count = int(pace["previous_7d"] or 0)
    snapshot["pace"] = {
        "last_hour_requests": last_hour_count,
        "last_24h_requests": last_24h_count,
        "previous_24h_requests": previous_24h_count,
        "day_change_pct": _change_percent(last_24h_count, previous_24h_count),
        "last_7d_requests": last_7d_count,
        "previous_7d_requests": previous_7d_count,
        "week_change_pct": _change_percent(last_7d_count, previous_7d_count),
        "average_tokens_per_request": round(tokens / requests, 1) if requests else 0.0,
    }
    snapshot["breakdown"] = {
        "hourly": _apply_deflated_tokens(
            _merge_canonical_breakdown([_breakdown_row(row) for row in breakdown_hourly if row["bucket"]]),
            token_lookup=deflated_hourly,
        ),
        "daily": _apply_deflated_tokens(
            _merge_canonical_breakdown([_breakdown_row(row) for row in breakdown_daily if row["bucket"]]),
            token_lookup=deflated_daily,
        ),
    }
    _annotate_model_categories(snapshot)
    return snapshot


def _annotate_model_categories(snapshot: dict[str, Any]) -> None:
    """Supply Models history grouping without browser-side name heuristics.

    Ledger buckets have no kind. Prefer the configured pool kind (including
    hidden pools), with the shared name fallback for retired pools.
    """
    pools = load_mux_topology().get("pools") or {}
    for rows in snapshot.get("breakdown", {}).values():
        for row in rows:
            provider = str(row.get("provider") or "local-vllm").strip().lower()
            local = _is_local_provider(provider)
            model = str(row.get("model") or "")
            kind = (pools.get(model) or {}).get("kind") if local else "llm"
            row["category"] = model_category(kind or "", model)
            row["local"] = local


def _is_local_provider(provider: str) -> bool:
    if re.search(
        r"openrouter|anthropic|openai|google|gemini|grok|xai|deepseek|mistral|"
        r"together|fireworks|bedrock|azure|claude|cursor|codex",
        provider,
    ):
        return False
    return bool(
        provider in {"", "local-vllm", "vllm", "local", "mux", "router"}
        or "local" in provider
        or "vllm" in provider
        or re.search(r":\d+$", provider)
        or _names_private_ip(provider)
    )


def _names_private_ip(provider: str) -> bool:
    """Whether ``provider`` contains a loopback or private-range IPv4 address."""
    for candidate in re.findall(r"\d{1,3}(?:\.\d{1,3}){3}", provider):
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.is_loopback or address.is_private:
            return True
    return False


def _dated_baseline(
    rows: list[sqlite3.Row],
    turn_rows: list[sqlite3.Row],
    *,
    first_hour: str | None,
    current: datetime,
) -> dict[str, Any]:
    """Exact preceding weekly peers relative to each displayed UTC bucket."""
    metrics = ("tokens", "requests", "cost_pico")
    zero = dict.fromkeys(metrics, 0)
    hours = {
        str(row["bucket"]): {"tokens": 0, "requests": int(row["requests"]), "cost_pico": int(row["cost_pico"])}
        for row in rows
        if row["bucket"]
    }
    days: dict[str, dict[str, int]] = {}
    for bucket, values in hours.items():
        total = days.setdefault(bucket[:10], dict(zero))
        for metric in metrics:
            total[metric] += values[metric]
    result: dict[str, Any] = {"dated_hourly": {}, "dated_daily": {}}
    # The cumulative-output heuristic must not inspect the target bucket's
    # own date or later turns. Cache one causal lookup per displayed UTC date.
    causal_tokens: dict[str, tuple[dict[str, int], dict[str, int]]] = {}
    end_hour = current.replace(minute=0, second=0, microsecond=0)
    for key, count, weeks, end, step, lookup in (
        ("dated_hourly", 24, 4, end_hour, timedelta(hours=1), hours),
        ("dated_daily", 7, 12, end_hour.replace(hour=0), timedelta(days=1), days),
    ):
        for offset in range(count - 1, -1, -1):
            bucket = end - offset * step
            hourly = key == "dated_hourly"

            def label(value: datetime) -> str:
                return value.strftime("%Y-%m-%dT%H:00:00Z" if hourly else "%Y-%m-%d")

            bucket_date = bucket.date().isoformat()
            if bucket_date not in causal_tokens:
                hourly_tokens, daily_tokens = _deflated_token_lookups(
                    [row for row in turn_rows if row["day_bucket"] < bucket_date]
                )
                total_hourly: dict[str, int] = {}
                total_daily: dict[str, int] = {}
                for grouped, total in ((hourly_tokens, total_hourly), (daily_tokens, total_daily)):
                    for (stamp, _provider, _model), tokens in grouped.items():
                        total[stamp] = total.get(stamp, 0) + tokens
                causal_tokens[bucket_date] = total_hourly, total_daily
            tokens_by_bucket = causal_tokens[bucket_date][0 if hourly else 1]
            history_start = first_hour if hourly else first_hour[:10] if first_hour else None
            peers = [
                {**lookup.get(label(peer), zero), "tokens": tokens_by_bucket.get(label(peer), 0)}
                for k in range(1, weeks + 1)
                if history_start and label(peer := bucket - timedelta(weeks=k)) >= history_start
            ]
            result[key][label(bucket)] = {
                "samples": len(peers),
                **{metric: median([peer[metric] for peer in peers]) if peers else None for metric in metrics},
            }
    return result


def _merge_canonical_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse alias and wire-id groups onto the full model name."""
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        model = resolve_generation_model(
            model=row.get("model"),
            provider=row.get("provider"),
        )
        key = (str(row["bucket"]), str(row["provider"]), model)
        if key not in merged:
            merged[key] = {**dict(row), "model": model}
            order.append(key)
            continue
        acc = merged[key]
        acc["requests"] += int(row.get("requests") or 0)
        acc["tokens"] += int(row.get("tokens") or 0)
        acc["cost_pico"] += int(row.get("cost_pico") or 0)
        acc["cost_usd"] = round(acc["cost_pico"] / 1_000_000_000_000, 6)
    return [merged[key] for key in order]


def _merge_model_totals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse alias and wire-id totals onto the full model name."""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        name = resolve_generation_model(model=row.get("name")) or "Unknown"
        if name not in merged:
            merged[name] = {
                "name": name,
                "requests": 0,
                "tokens": 0,
                "cost_pico": 0,
            }
            order.append(name)
        acc = merged[name]
        acc["requests"] += int(row.get("requests") or 0)
        acc["tokens"] += int(row.get("tokens") or 0)
        acc["cost_pico"] += int(row.get("cost_pico") or 0)
    return [merged[name] for name in order]


def _deflated_token_lookups(
    rows: list[sqlite3.Row],
) -> tuple[dict[tuple[str, str, str], int], dict[tuple[str, str, str], int]]:
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(row["session_id"] or ""), []).append(row)
    hourly: dict[tuple[str, str, str], int] = {}
    daily: dict[tuple[str, str, str], int] = {}
    for session_rows in grouped.values():
        deltas = deflate_session_outputs([int(row["output_tokens"] or 0) for row in session_rows])
        for row, tokens in zip(session_rows, deltas):
            provider = resolve_generation_provider(
                provider=row["provider"],
                model=row["model"],
            )
            model = resolve_generation_model(
                model=row["model"],
                provider=row["provider"],
            )
            hour = str(row["hour_bucket"] or "")
            day = str(row["day_bucket"] or "")
            if hour:
                key = (hour, provider, model)
                hourly[key] = hourly.get(key, 0) + tokens
            if day:
                key = (day, provider, model)
                daily[key] = daily.get(key, 0) + tokens
    return hourly, daily


def _apply_deflated_tokens(
    rows: list[dict[str, Any]],
    *,
    token_lookup: dict[tuple[str, str, str], int],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row["bucket"]), str(row["provider"]), str(row["model"]))
        seen.add(key)
        if key in token_lookup:
            row["tokens"] = int(token_lookup[key])
    for key, tokens in token_lookup.items():
        if key in seen or tokens <= 0:
            continue
        rows.append(
            {
                "bucket": key[0],
                "provider": key[1],
                "model": key[2],
                "requests": 0,
                "tokens": int(tokens),
                "cost_pico": 0,
                "cost_usd": 0.0,
            }
        )
    return rows


def _breakdown_row(row: sqlite3.Row) -> dict[str, Any]:
    cost_pico = int(row["cost_pico"] or 0)
    return {
        "bucket": str(row["bucket"]),
        "provider": resolve_generation_provider(
            provider=row["provider"],
            model=row["model"],
        ),
        "model": resolve_generation_model(
            model=row["model"],
            provider=row["provider"],
        ),
        "requests": int(row["requests"] or 0),
        "tokens": int(row["tokens"] or 0),
        "cost_pico": cost_pico,
        "cost_usd": round(cost_pico / 1_000_000_000_000, 6),
    }
