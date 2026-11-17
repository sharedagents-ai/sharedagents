#!/usr/bin/env python3
"""Re-price stored turn costs for one provider over a time window.

Rewrites ``turn_costs.cost_pico`` (and, by default, the per-round
``ai_generations.cost_usd`` rows) for turns whose recorded cost was computed
with a stale/incorrect pricing table (for example the 2026-09-17 DeepSeek
``deepseek-flash`` fall-through that billed every round at the generic default
row). Tokens are never touched: each row is re-priced from its own token
buckets with the CURRENT pricing table — including any peak/off-peak schedule,
evaluated at the row's ``created_at``.

Turn rows are written back through ``UsageLedger.record(..., authoritative=True)``
so the ledger's ``cost_accumulators`` (goal/session exact + charged pico) are
corrected with the same tested arithmetic the live path uses.

Dry-run by default; ``--apply`` writes.

Examples
--------
    python scripts/reprice_turn_costs.py --provider deepseek \
        --since 2026-09-14 --dry-run
    python scripts/reprice_turn_costs.py --provider deepseek \
        --since 2026-09-14 --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from sharedagents_core.layout import usage_ledger_db_path  # noqa: E402
from sharedagents_model.usage_ledger import UsageLedger  # noqa: E402
from sharedagents_model.usage_pricing import (  # noqa: E402
    CanonicalUsage,
    CostResult,
    PricingTable,
    normalize_usage,
    price,
)

_COLUMNS = (
    "session_id, turn_id, root_session_id, ancestor_goal_id, agent_name, "
    "model, model_alias, provider, input_tokens, cached_read_tokens, "
    "cached_write_tokens, thinking_tokens, output_tokens, cost_pico, "
    "cost_estimated, created_at"
)

_GENERATION_COLUMNS = (
    "generation_id, model, model_alias, provider, input_tokens, cached_tokens, "
    "fresh_tokens, output_tokens, thinking_tokens, content_tokens, "
    "cost_usd, created_at"
)


def _parse_when(value: str, *, end_of_day: bool = False) -> str:
    """Normalise a CLI date/datetime into a UTC ISO string for SQL comparison."""
    raw = (value or "").strip().replace("Z", "+00:00")
    if not raw:
        raise argparse.ArgumentTypeError("empty timestamp")
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        try:
            ts = datetime.strptime(raw, "%Y-%m-%d")
        except ValueError as exc:  # pragma: no cover - argparse surfaces it
            raise argparse.ArgumentTypeError(f"bad timestamp {value!r}") from exc
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if end_of_day and ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        ts = ts.replace(hour=23, minute=59, second=59, microsecond=999999)
    return ts.astimezone(timezone.utc).isoformat()


def _row_usage(row: sqlite3.Row) -> CanonicalUsage:
    return CanonicalUsage(
        input=int(row["input_tokens"] or 0),
        cached_read=int(row["cached_read_tokens"] or 0),
        cached_write=int(row["cached_write_tokens"] or 0),
        thinking=int(row["thinking_tokens"] or 0),
        output=int(row["output_tokens"] or 0),
        model=str(row["model"] or ""),
        model_alias=str(row["model_alias"] or ""),
        provider=str(row["provider"] or ""),
        estimated=bool(row["cost_estimated"]),
    )


def _row_when(row: sqlite3.Row) -> datetime:
    raw = str(row["created_at"] or "").replace("Z", "+00:00")
    ts = datetime.fromisoformat(raw)
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _pico_int(exact_pico: Decimal) -> int:
    """Quantize an exact pico-USD sum to the ledger's integer column."""
    return int(exact_pico.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _scope_key(row: sqlite3.Row) -> str:
    """Mirror UsageLedger.record's scope key for a stored turn row."""
    goal = str(row["ancestor_goal_id"] or "")
    if goal:
        return f"goal:{goal}"
    return f"session:{row['root_session_id'] or row['session_id']}"


def _ensure_accumulator(con: sqlite3.Connection, scope_key: str) -> bool:
    """Create the scope's ``cost_accumulators`` row when it is missing.

    ``UsageLedger.record`` corrects an EXISTING accumulator on a re-finalize,
    but never inserts one — a scope whose rows were written by an older code
    path would otherwise keep no exact/charged total. The rebuilt total is the
    sum of the rows' integer allocations, which keeps the ledger invariant
    ``SUM(row cost_pico) == charged_pico`` exactly.
    """
    exists = con.execute("SELECT 1 FROM cost_accumulators WHERE scope_key=?", (scope_key,)).fetchone()
    if exists is not None:
        return False
    if scope_key.startswith("goal:"):
        where, param = "ancestor_goal_id = ?", scope_key[len("goal:") :]
    else:
        where = "COALESCE(NULLIF(root_session_id, ''), session_id) = ?"
        param = scope_key[len("session:") :]
    total = con.execute(f"SELECT COALESCE(SUM(cost_pico), 0) FROM turn_costs WHERE {where}", (param,)).fetchone()[0]
    con.execute(
        "INSERT INTO cost_accumulators (scope_key, exact_pico, charged_pico) VALUES (?, ?, ?)",
        (scope_key, str(int(total or 0)), int(total or 0)),
    )
    con.commit()
    return True


def _is_cost_override(row: sqlite3.Row) -> bool:
    """True if row is an authoritative cost override (0 tokens with non-zero cost)."""
    keys = row.keys() if hasattr(row, "keys") else []
    if "cost_pico" in keys:
        toks = (
            int(row["input_tokens"] or 0)
            + int(row["cached_read_tokens"] or 0)
            + int(row["cached_write_tokens"] or 0)
            + int(row["thinking_tokens"] or 0)
            + int(row["output_tokens"] or 0)
        )
        return toks == 0 and int(row["cost_pico"] or 0) > 0
    if "cost_usd" in keys:
        toks = (
            int(row["input_tokens"] or 0)
            + int(row["cached_tokens"] or 0)
            + int(row["fresh_tokens"] or 0)
            + int(row["thinking_tokens"] or 0)
            + int(row["output_tokens"] or 0)
            + int(row["content_tokens"] or 0)
        )
        return toks == 0 and float(row["cost_usd"] or 0.0) > 0.0
    return False


def _is_zero_cost_opus(row: sqlite3.Row) -> bool:
    """True if row is an opus row recorded with $0 cost."""
    model = str(row["model"] or "").lower()
    if "opus" not in model:
        return False
    keys = row.keys() if hasattr(row, "keys") else []
    if "cost_pico" in keys:
        return int(row["cost_pico"] or 0) == 0
    if "cost_usd" in keys:
        return float(row["cost_usd"] or 0.0) == 0.0
    return False


def _select_rows(con: sqlite3.Connection, args: argparse.Namespace) -> list[sqlite3.Row]:
    where = ["created_at >= ?"]
    params: list[object] = [_parse_when(args.since)]
    if args.provider:
        where.append("provider = ?")
        params.append(args.provider)
    if args.until:
        where.append("created_at <= ?")
        params.append(_parse_when(args.until, end_of_day=True))
    if args.model_prefix:
        where.append("model LIKE ?")
        params.append(f"{args.model_prefix}%")
    if getattr(args, "audit_targets", False):
        where.append(
            "(model = 'claude-opus-5[1m]' OR (model = 'qwen3.8-27b-fp8' AND (provider = '' OR provider IS NULL)))"
        )
    sql = f"SELECT {_COLUMNS} FROM turn_costs WHERE {' AND '.join(where)} ORDER BY created_at"
    return list(con.execute(sql, params))


def _generation_own_usage(row: sqlite3.Row) -> CanonicalUsage:
    """Round metrics from an ``ai_generations`` row's own token columns."""
    return normalize_usage(
        {
            "prompt_tokens": int(row["input_tokens"] or 0),
            "prompt_tokens_cached": int(row["cached_tokens"] or 0),
            "prompt_tokens_fresh": int(row["fresh_tokens"] or 0),
            "reasoning_tokens": int(row["thinking_tokens"] or 0),
            "completion_tokens": int(row["output_tokens"] or 0),
            "content_tokens": int(row["content_tokens"] or 0),
            "model": str(row["model"] or ""),
            "model_alias": str(row["model_alias"] or ""),
            "provider": str(row["provider"] or ""),
        }
    )


_GENERATION_TOKEN_COLUMNS = (
    "input_tokens",
    "cached_tokens",
    "fresh_tokens",
    "output_tokens",
    "thinking_tokens",
    "content_tokens",
)


def _generation_tokens_empty(row: sqlite3.Row) -> bool:
    """True when every token bucket on the generation row is zero."""
    return all(int(row[col] or 0) == 0 for col in _GENERATION_TOKEN_COLUMNS)


def _generation_usage(
    row: sqlite3.Row,
    con: sqlite3.Connection | None = None,
    counts: dict[tuple[str, str], int] | None = None,
) -> CanonicalUsage:
    """Rebuild round metrics from an ``ai_generations`` row (see record_generation).

    A turn owns ONE ``turn_costs`` row holding the WHOLE turn's buckets, while
    ``ai_generations`` holds one row per round. Pricing every sibling round from
    the turn row therefore bills the turn's tokens once per round — the turn row
    is only a fallback for a lone round that recorded no tokens of its own.
    """
    own = _generation_own_usage(row)
    if con is None:
        return own
    try:
        if _turn_generation_count(con, row, counts) != 1:
            return own
        tc = con.execute(
            "SELECT * FROM turn_costs WHERE session_id=? AND turn_id=?",
            (row["session_id"], row["turn_id"]),
        ).fetchone()
        if tc is None:
            return own
        if _generation_tokens_empty(row):
            return _row_usage(tc)
        # ``ai_generations`` has no cache-write column, so a lone round keeps
        # the bucket from its turn row; every other bucket stays the round's own.
        if _round_read_aliases_turn_write(row, tc):
            # The row's ``cached_tokens`` IS the turn's write bucket, not a read.
            return replace(
                own,
                cached_read=int(tc["cached_read_tokens"] or 0),
                cached_write=int(tc["cached_write_tokens"] or 0),
            )
        cached_write = _lone_round_cache_write(row, tc)
        if cached_write == own.cached_write:
            return own
        return replace(own, cached_write=cached_write)
    except Exception:
        pass
    return own


def _turn_generation_count(
    con: sqlite3.Connection,
    row: sqlite3.Row,
    counts: dict[tuple[str, str], int] | None = None,
) -> int:
    """How many ``ai_generations`` rows the row's turn has, window-independent."""
    key = (str(row["session_id"]), str(row["turn_id"]))
    if counts is not None and key in counts:
        return int(counts[key])
    return int(
        con.execute(
            "SELECT COUNT(*) FROM ai_generations WHERE session_id=? AND turn_id=?",
            key,
        ).fetchone()[0]
        or 0
    )


def _round_read_aliases_turn_write(row: sqlite3.Row, tc: sqlite3.Row) -> bool:
    """True when a lone round's ``cached_tokens`` holds the turn's cache-WRITE bucket.

    The ``finish_reason='ledger'`` backfill rows are rebuilt from their turn row
    as ``input = fresh + write`` with ``cached = write``; their turns record
    ``cached_read = 0``. Billing that alias as a cache READ while ALSO taking
    ``cached_write`` from the turn row charges the same tokens twice — once at
    the write rate, once at the read rate.

    Two identities pin the alias down, both taken from the turn row rather than
    from ``finish_reason`` (a reason, not a structure):

    * the round's ``cached_tokens`` equals the turn's write bucket, and
    * the turn records NO cache read at all, so a read of exactly that size
      cannot be genuine.

    The second half is what keeps a live row safe: a lone round whose
    ``cached_tokens`` is a genuine read keeps it even when the number coincides
    with the turn's write bucket. (``input - cached - fresh == 0`` alone cannot
    discriminate — it holds for 38,357 of the 38,395 joined live rows and for
    629 rows that carry a real read.)
    """
    turn_write = int(tc["cached_write_tokens"] or 0)
    if turn_write <= 0 or int(tc["cached_read_tokens"] or 0) != 0:
        return False
    return int(row["cached_tokens"] or 0) == turn_write


def _lone_round_cache_write(row: sqlite3.Row, tc: sqlite3.Row) -> int:
    """Cache-write tokens for a lone round, whose own table has no such column.

    The turn row is the source: across the live ledger it equals the round's own
    ``input - cached - fresh`` on every row that records a cache write, and on
    the ``finish_reason='ledger'`` backfill rows — where that difference is 0
    and the tokens would otherwise be dropped entirely — it is the only source
    left. The rebuilt difference is only a fallback for a turn row that records
    no cache write at all.
    """
    turn_write = int(tc["cached_write_tokens"] or 0)
    if turn_write > 0:
        return turn_write
    rebuilt = int(row["input_tokens"] or 0) - int(row["cached_tokens"] or 0) - int(row["fresh_tokens"] or 0)
    return rebuilt if rebuilt > 0 else 0


def _price_generation_row(
    row: sqlite3.Row,
    con: sqlite3.Connection,
    counts: dict[tuple[str, str], int] | None,
    table: PricingTable,
    old_table: PricingTable | None,
) -> CostResult | None:
    """Price one round row with the current table, or None when it cannot be priced.

    ``counts`` is the (optional) sibling-count cache; ``None`` makes every count
    read come straight from ``con`` — which is what the write path needs.
    """
    if _is_cost_override(row) or _is_zero_cost_opus(row):
        return None
    usage = _generation_usage(row, con=con, counts=counts)
    entry = table.resolve(usage)
    if entry is table.default:
        return None
    if old_table is not None:
        old_entry = old_table.resolve(usage)
        if old_entry is not old_table.default:
            return None
    return price(usage, entry, when=_row_when(row))


def _write_generation_row(
    con: sqlite3.Connection,
    row: sqlite3.Row,
    table: PricingTable,
    old_table: PricingTable | None,
    stored: float,
    *,
    force: bool,
) -> tuple[CostResult | None, bool]:
    """Re-price one round row under the write lock and store it when it changed.

    The whole-table sibling count is a SNAPSHOT taken outside any transaction: a
    second connection can commit another round of the same turn while this run
    walks the table. Acting on that snapshot would let a lone-round decision
    charge one round the WHOLE turn (or lend it the turn's cache-write bucket)
    after the turn stopped being a lone round — the original inflation defect, at
    runtime. ``BEGIN IMMEDIATE`` takes the write lock first and the count is
    re-read INSIDE it, so no sibling can land between the count and the
    ``UPDATE``: every value written is priced under a count that still holds at
    commit time. The lock is held for one row, not for the run; a ledger busy
    beyond the connection timeout aborts the run loudly instead of overcharging.

    Returns ``(cost, wrote)``: ``cost`` is the freshly computed value (None when
    the row can no longer be priced) and ``wrote`` says whether it was stored.
    """
    con.execute("BEGIN IMMEDIATE")
    try:
        fresh = _price_generation_row(row, con, None, table, old_table)
        if fresh is None or (abs(fresh.total_usd - stored) < 1e-9 and not force):
            con.rollback()
            return fresh, False
        con.execute(
            "UPDATE ai_generations SET cost_usd=? WHERE generation_id=?",
            (fresh.total_usd, str(row["generation_id"])),
        )
        con.commit()
        return fresh, True
    except BaseException:
        con.rollback()
        raise


def reprice_generations(
    *,
    db_path: Path,
    table: PricingTable,
    old_table: PricingTable | None,
    args: argparse.Namespace,
    apply: bool,
) -> dict[str, float]:
    """Re-price per-round ``ai_generations.cost_usd`` rows over the same window."""
    where = ["created_at >= ?"]
    params: list[object] = [_parse_when(args.since)]
    if args.provider:
        where.append("provider = ?")
        params.append(args.provider)
    if args.until:
        where.append("created_at <= ?")
        params.append(_parse_when(args.until, end_of_day=True))
    if args.model_prefix:
        where.append("model LIKE ?")
        params.append(f"{args.model_prefix}%")
    if getattr(args, "audit_targets", False):
        where.append(
            "(model = 'claude-opus-5[1m]' OR (model = 'qwen3.8-27b-fp8' AND (provider = '' OR provider IS NULL)))"
        )
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        rows = list(
            con.execute(
                f"SELECT {_GENERATION_COLUMNS}, session_id, turn_id FROM ai_generations WHERE {' AND '.join(where)}",
                params,
            )
        )
        # Round counts for the WHOLE table, so a window-scoped sibling that is
        # filtered out here still marks its turn as multi-round.
        counts = {
            (str(r["session_id"]), str(r["turn_id"])): int(r["n"])
            for r in con.execute(
                "SELECT session_id, turn_id, COUNT(*) AS n FROM ai_generations GROUP BY session_id, turn_id"
            )
        }
        stats = {
            "rows": float(len(rows)),
            "repriced": 0.0,
            "skipped": 0.0,
            "before": 0.0,
            "after": 0.0,
            "restaled": 0.0,
        }
        for row in rows:
            stored = float(row["cost_usd"] or 0.0)
            cost = _price_generation_row(row, con, counts, table, old_table)
            wrote = False
            if cost is not None and (abs(cost.total_usd - stored) >= 1e-9 or args.force):
                if apply:
                    pre_scan = cost
                    cost, wrote = _write_generation_row(
                        con,
                        row,
                        table,
                        old_table,
                        stored,
                        force=args.force,
                    )
                    if cost is not None and abs(cost.total_usd - pre_scan.total_usd) >= 1e-9:
                        stats["restaled"] += 1
                else:
                    wrote = True
            if cost is None:
                stats["skipped"] += 1
                continue
            stats["before"] += stored
            stats["after"] += cost.total_usd
            if wrote:
                stats["repriced"] += 1
        if apply:
            con.commit()
        return stats
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="", help="provider column value, e.g. deepseek (default: all providers)")
    ap.add_argument(
        "--since", default="1970-01-01", help="window start (UTC date or ISO datetime, default: 1970-01-01)"
    )
    ap.add_argument("--until", default="", help="optional window end (inclusive)")
    ap.add_argument("--model-prefix", default="", help="optional model LIKE filter")
    ap.add_argument(
        "--audit-targets",
        action="store_true",
        help="reprice only: (a) claude-opus-5[1m] rows with doubled fable rates, and (b) empty-provider qwen3.8-27b-fp8 rows",
    )
    ap.add_argument("--db", default="", help="ledger.db path (default: sharedagents usage ledger)")
    ap.add_argument("--pricing", default="", help="pricing.json path (default: sharedagents pricing path)")
    ap.add_argument(
        "--old-pricing",
        default="",
        help="optional old pricing.json path to reprice only rows previously resolving to default",
    )
    ap.add_argument("--apply", action="store_true", help="write corrections (default: dry run)")
    ap.add_argument("--force", action="store_true", help="rewrite rows even when the cost is unchanged")
    ap.add_argument(
        "--skip-generations",
        action="store_true",
        help="do not touch the per-round ai_generations.cost_usd rows",
    )
    args = ap.parse_args(argv)

    db_path = Path(args.db).expanduser() if args.db else usage_ledger_db_path()
    # An explicit --pricing/--old-pricing names ONE table: load it standalone, or
    # the current bundled asset would be merged underneath a historical copy and
    # that copy would appear to contain rows added after it was taken. With no
    # flag, the effective (layered) table is the right one -- it is what bills.
    table = PricingTable.load(Path(args.pricing).expanduser(), standalone=True) if args.pricing else PricingTable.load()
    old_table = PricingTable.load(Path(args.old_pricing).expanduser(), standalone=True) if args.old_pricing else None
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = _select_rows(con, args)
    con.close()

    ledger = UsageLedger(db_path) if args.apply else None
    before_total = Decimal(0)
    after_total = Decimal(0)
    touched = unchanged = skipped_default = 0
    per_day: dict[str, list[Decimal]] = {}
    samples: list[str] = []
    scopes: set[str] = set()

    for row in rows:
        if _is_cost_override(row) or _is_zero_cost_opus(row):
            unchanged += 1
            continue
        usage = _row_usage(row)
        entry = table.resolve(usage)
        if entry is table.default:
            skipped_default += 1
            continue
        if old_table is not None:
            old_entry = old_table.resolve(usage)
            if old_entry is not old_table.default:
                skipped_default += 1
                continue
        when = _row_when(row)
        cost = price(usage, entry, when=when)
        exact = cost.exact_total_pico or Decimal(0)
        old_pico = int(row["cost_pico"] or 0)
        new_pico = _pico_int(exact)
        day = str(row["created_at"])[:10]
        before_total += Decimal(old_pico)
        after_total += exact
        per_day.setdefault(day, [Decimal(0), Decimal(0), Decimal(0)])
        per_day[day][0] += Decimal(old_pico)
        per_day[day][1] += exact
        per_day[day][2] += 1

        if new_pico == old_pico and not args.force:
            unchanged += 1
            continue
        touched += 1
        scopes.add(_scope_key(row))
        if len(samples) < 6:
            samples.append(
                f"  {row['created_at'][:19]} {str(row['model']):18s} "
                f"${old_pico / 1e12:.6f} -> ${float(exact) / 1e12:.6f}"
            )
        if ledger is not None:
            ledger.record(
                session_id=str(row["session_id"]),
                turn_id=str(row["turn_id"]),
                root_session_id=str(row["root_session_id"] or ""),
                ancestor_goal_id=str(row["ancestor_goal_id"] or "") or None,
                agent_name=str(row["agent_name"] or ""),
                cost=CostResult(
                    total_pico=new_pico,
                    exact_total_pico=exact,
                    estimated=usage.estimated,
                    model=str(row["model"] or ""),
                    model_alias=str(row["model_alias"] or ""),
                    provider=str(row["provider"] or ""),
                    authoritative=True,
                ),
                usage=usage,
                return_allocated_pico=True,
            )

    created_accumulators = 0
    if args.apply and scopes:
        fix_con = sqlite3.connect(str(db_path))
        try:
            for scope_key in sorted(scopes):
                if _ensure_accumulator(fix_con, scope_key):
                    created_accumulators += 1
        finally:
            fix_con.close()

    gen_stats = None
    if not args.skip_generations:
        gen_stats = reprice_generations(
            db_path=db_path,
            table=table,
            old_table=old_table,
            args=args,
            apply=args.apply,
        )

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"[{mode}] provider={args.provider} since={args.since} until={args.until or 'now'}")
    print(f"  db={db_path}")
    print(
        f"  rows matched={len(rows)}  repriced={touched}  unchanged={unchanged}  "
        f"skipped(default-row resolution)={skipped_default}"
    )
    if args.apply and created_accumulators:
        print(f"  rebuilt missing cost_accumulators rows={created_accumulators}")
    if gen_stats is not None:
        print(
            f"  generations: matched={int(gen_stats['rows'])} "
            f"repriced={int(gen_stats['repriced'])} "
            f"skipped(default-row resolution)={int(gen_stats['skipped'])}  "
            f"total before=${gen_stats['before']:.4f} after=${gen_stats['after']:.4f}"
        )
        if gen_stats.get("restaled"):
            print(
                f"  generations: {int(gen_stats['restaled'])} row(s) re-decided under "
                f"the write lock — a sibling round landed after the count snapshot"
            )
    print(f"  total before=${float(before_total) / 1e12:.4f}  after=${float(after_total) / 1e12:.4f}")
    for day in sorted(per_day):
        old, new, count = per_day[day]
        print(f"    {day}: {int(count):3d} rows  ${float(old) / 1e12:9.4f} -> ${float(new) / 1e12:8.4f}")
    if samples:
        print("  samples:")
        print("\n".join(samples))
    if not args.apply:
        print("  (dry run — pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
