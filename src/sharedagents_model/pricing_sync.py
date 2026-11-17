"""Populate the machine-managed pricing overlay from the model catalogue.

Discovery knows which models an engine can run, but nothing priced the ids it
found. A model the bundled asset does not cover therefore fell through to its
provider's fallback row: the reported case billed ``claude-sonnet-5-5`` at the
anthropic fallback ($5/$25) instead of its list rate ($2/$10) — a silent 2.5x
overcharge on every turn — while ``claude-opus-5-5`` and
``claude-sonnet-4-5-20250929`` were overcharged and ``claude-fable-5-1``
undercharged by 2x.

This module reconciles the ids discovery has seen against the models.dev
catalogue (:mod:`sharedagents_model.catalogue`, whose slim copy now carries each
model's list rate) and writes what is missing into ``pricing_auto.json`` — the
layer BELOW the operator's file and ABOVE the bundled asset.

What it will not do
-------------------
* Overwrite a row the operator's file defines. They win, always.
* Rewrite a row the bundled asset defines. Those are hand-audited and carry
  schedules the catalogue does not publish (DeepSeek peak windows); a
  disagreement is REPORTED as drift for a repo audit instead.
  An explicit ``prefer_catalogue`` policy opts into migrating supported
  manual/bundled model rows; unavailable models and fallback policy stay manual.
* Put a top-level ``provider`` tag on a created row. :meth:`PricingTable.resolve`
  treats a row whose top-level provider equals the turn's provider as that
  provider's fallback, so a tagged row would silently price every OTHER
  unpriced model of that provider at this model's rates. Auto rows key on the
  exact model id (plus the engine's alias) and record the upstream provider in
  their provenance.
* Write a ``default`` block: the non-zero floor is policy, not a supplier rate.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_model import catalogue
from sharedagents_model.usage_pricing import (
    LAYER_BUNDLED,
    LAYER_OPERATOR,
    MAX_RATE_USD,
    _default_pricing_path,
    _pricing_auto_path_for,
    _pricing_layers,
    _read_pricing_dict,
    _write_pricing_dict,
)

_LOG = logging.getLogger("sharedagents.model.pricing_sync")

#: Set to 0/false/no/off (or empty) to stop the sync writing anything. It still
#: produces its report, so `--dry-run` stays useful for diagnosis.
AUTOSYNC_ENV = "SHAREDAGENTS_PRICING_AUTOSYNC"
#: How often the discovery path may re-reconcile one overlay.
DEFAULT_MIN_INTERVAL_S = 6 * 3600
#: After a failure, wait this long before trying again.
FAILURE_BACKOFF_S = 300.0
#: One upstream step this large is likelier a typo than a real repricing: apply
#: it, but surface it (history is immutable — already-recorded turns keep the
#: rate they were billed at, so the exposure is forward-only).
REVIEW_FACTOR = 5.0
#: Provenance key on a row this module owns.
AUTO_META_KEY = "auto"
#: The rate fields compared when deciding whether a row is already right.
_RATE_FIELDS = (
    "input_per_mtok",
    "cached_read_per_mtok",
    "cached_write_per_mtok",
    "output_per_mtok",
    "thinking_per_mtok",
)

#: Overlay path -> monotonic time of the last attempt (discovery debounce).
_LAST_SYNC: dict[str, float] = {}
_LAST_DISCOVERY: dict[str, tuple[tuple[str, ...], ...]] = {}


def autosync_enabled() -> bool:
    """Whether the sync may WRITE (the report is always available)."""
    raw = os.environ.get(AUTOSYNC_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Catalogue lookups (pure functions of a slim providers dict, so the tests can
# inject a body instead of reaching the network)
# ---------------------------------------------------------------------------


def _id_variants(pid: str, model_id: str) -> list[str]:
    """Catalogue spellings of a discovered id, most specific first.

    Engine rows are provider-qualified (``amazon-bedrock/anthropic.claude-x``,
    ``vllm/qwen3.6-27b``) while the catalogue lists the bare model id under
    that provider."""
    out = [model_id]
    for prefix in (f"{pid}/",):
        if model_id.startswith(prefix):
            out.append(model_id[len(prefix) :])
    if "/" in model_id:
        out.append(model_id.rsplit("/", 1)[1])
    return list(dict.fromkeys(out))


def _upstream_costs(
    providers: Mapping[str, Any],
    row: Mapping[str, Any],
    model_id: str,
    provider_hint: str = "",
) -> list[tuple[str, str, dict[str, float]]]:
    """``(provider_id, catalogue_model_id, cost)`` for each candidate pricing it.

    ``provider_hint`` is the provider a row we WROTE came from: an auto row
    carries no provider tag on purpose (it must not act as that provider's
    fallback), so re-resolving one for a refresh needs the provenance to supply
    it."""
    if not providers:
        return []
    pids = catalogue.providers_for_row(row, dict(providers))
    if not pids and provider_hint and provider_hint in providers:
        pids = [provider_hint]
    if not pids and not (row.get("baseUrl") or row.get("endpoint")):
        prefix = model_id.split("/", 1)[0]
        if prefix in providers:
            pids = [prefix]
    found: list[tuple[str, str, dict[str, float]]] = []
    for pid in pids:
        models = (providers.get(pid) or {}).get("models")
        if not isinstance(models, dict):
            continue
        for candidate_id in _id_variants(pid, model_id):
            entry = models.get(candidate_id)
            cost = entry.get("cost") if isinstance(entry, dict) else None
            if isinstance(cost, dict):
                found.append((pid, candidate_id, {k: float(v) for k, v in cost.items()}))
                break
    return found


def _choose_cost(
    found: Sequence[tuple[str, str, dict[str, float]]],
) -> tuple[dict[str, float] | None, str | None, str | None, bool]:
    """Pick one rate when several catalogue providers list the model.

    Returns ``(cost, provider_id, catalogue_model_id, disagreed)``. When the
    candidates disagree the highest input+output total wins -- never silently
    undercharge -- and the run is flagged for review.
    """
    if not found:
        return None, None, None, False

    def total(item: tuple[str, str, dict[str, float]]) -> float:
        cost = item[2]
        return cost.get("input", 0.0) + cost.get("output", 0.0)

    best = max(found, key=total)
    disagreed = len({(c.get("input"), c.get("output")) for _p, _m, c in found}) > 1
    return best[2], best[0], best[1], disagreed


def _guard(cost: Mapping[str, Any]) -> str | None:
    """Why this upstream rate must not be written, or None when it is usable."""
    inp = cost.get("input")
    outp = cost.get("output")
    if inp is None or outp is None:
        return "catalogue carries no input/output rate"
    for name, value in (("input", inp), ("output", outp)):
        if value <= 0:
            # A zero is indistinguishable from a placeholder, and a zero bucket
            # makes a /goal budget unenforceable for that model. Reported; the
            # operator can add the row by hand if the model really is free.
            return f"zero {name} rate (would silently make that bucket free)"
        if value > float(MAX_RATE_USD):
            return f"{name} rate above the {MAX_RATE_USD}/Mtok ceiling"
    for name in ("cached_read", "cached_write"):
        value = cost.get(name)
        if value is not None and not 0 <= value <= float(MAX_RATE_USD):
            return f"{name} rate outside the plausible range"
    if outp < inp:
        return f"output rate {outp} below input rate {inp}"
    cached = cost.get("cached_read")
    if cached is not None and cached > inp:
        return f"cache-read rate {cached} above the input rate {inp}"
    return None


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _priced_ids(rows: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """``{model id or alias: row}`` for the live rows of one layer."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("disabled"):
            continue
        match = value if isinstance(value := row.get("match"), dict) else {}
        for key in (match.get("id"), match.get("alias")):
            if key:
                out.setdefault(str(key), row)
    return out


def _rates(row: Mapping[str, Any]) -> dict[str, float | None]:
    return {name: row.get(name) for name in _RATE_FIELDS}


def _same_rates(row: Mapping[str, Any], cost: Mapping[str, Any]) -> bool:
    """Whether an existing row already bills exactly what upstream publishes."""
    for name, key in (("input_per_mtok", "input"), ("output_per_mtok", "output")):
        if row.get(name) != cost.get(key):
            return False
    for name, key in (("cached_read_per_mtok", "cached_read"), ("cached_write_per_mtok", "cached_write")):
        if key in cost and row.get(name) != cost.get(key):
            return False
        if key not in cost and row.get(name) is not None:
            return False
    # thinking is derived from output when the catalogue does not publish one.
    return row.get("thinking_per_mtok") in (cost.get("output"), None)


def _creates_review(previous: Mapping[str, Any], cost: Mapping[str, Any]) -> bool:
    """Whether refreshing a row is a step large enough to surface for review."""
    for name, key in (("input_per_mtok", "input"), ("output_per_mtok", "output")):
        old, new = previous.get(name), cost.get(key)
        if old in (None, 0) or new is None:
            continue
        if max(float(old), float(new)) / min(float(old), float(new)) >= REVIEW_FACTOR:
            return True
    return False


def _build_row(
    *,
    model_id: str,
    alias: str,
    cost: Mapping[str, Any],
    provider_id: str,
    catalogue_model_id: str,
    source: str,
    revision: int | None,
    fetched_at: str,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One auto-overlay row: exact-id match, rates, and provenance.

    Deliberately carries NO top-level ``provider`` and no ``match.provider``:
    either would make this row that provider's fallback for every model it does
    not itself price (see the module docstring)."""
    row: dict[str, Any] = {"match": {"id": model_id}}
    if alias and alias != model_id:
        row["match"]["alias"] = alias
    row["input_per_mtok"] = float(cost["input"])
    if cost.get("cached_read") is not None:
        row["cached_read_per_mtok"] = float(cost["cached_read"])
    if cost.get("cached_write") is not None:
        row["cached_write_per_mtok"] = float(cost["cached_write"])
    row["output_per_mtok"] = float(cost["output"])
    # thinking follows output unless the existing row hinted at its own value
    # (mirrors usage_pricing._apply_rates).
    old_out = previous.get("output_per_mtok") if previous else None
    old_think = previous.get("thinking_per_mtok") if previous else None
    if old_think is None or old_think == old_out:
        row["thinking_per_mtok"] = float(cost["output"])
    elif previous is not None:
        row["thinking_per_mtok"] = old_think
    if previous:
        for key in ("peak_multiplier", "peak_hours_utc", "peak_weekdays_only"):
            if key in previous:
                row[key] = previous[key]
    row["source_url"] = source
    meta: dict[str, Any] = {
        "upstream_model_id": catalogue_model_id,
        "upstream_provider": provider_id,
        "fetched_at": fetched_at,
    }
    if revision is not None:
        meta["catalogue_revision"] = revision
    row[AUTO_META_KEY] = meta
    return row


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class SyncReport:
    """What one reconcile pass saw, decided, and wrote."""

    created: dict[str, dict[str, Any]] = field(default_factory=dict)
    refreshed: dict[str, dict[str, Any]] = field(default_factory=dict)
    retired: list[str] = field(default_factory=list)
    operator_owned: list[str] = field(default_factory=list)
    bundled_owned: list[str] = field(default_factory=list)
    bundled_drift: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    auto_path: str = ""
    applied: bool = False
    dry_run: bool = False
    disabled: bool = False
    stale: bool = False
    review_required: bool = False
    catalogue_source: str = ""
    catalogue_revision: int | None = None
    candidates_seen: int = 0
    migrated: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[str]:
        return sorted(set(self.created) | set(self.refreshed) | set(self.retired))

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_path": self.auto_path,
            "applied": self.applied,
            "dry_run": self.dry_run,
            "disabled": self.disabled,
            "stale": self.stale,
            "review_required": self.review_required,
            "catalogue": {
                "source": self.catalogue_source,
                "revision": self.catalogue_revision,
            },
            "candidates_seen": self.candidates_seen,
            "counts": {
                "created": len(self.created),
                "refreshed": len(self.refreshed),
                "retired": len(self.retired),
                "operator_owned": len(self.operator_owned),
                "bundled_owned": len(self.bundled_owned),
                "bundled_drift": len(self.bundled_drift),
                "rejected": len(self.rejected),
                "unresolved": len(self.unresolved),
                "migrated": len(self.migrated),
            },
            "created": self.created,
            "refreshed": self.refreshed,
            "retired": self.retired,
            "operator_owned": self.operator_owned,
            "bundled_owned": self.bundled_owned,
            "bundled_drift": self.bundled_drift,
            "rejected": self.rejected,
            "unresolved": self.unresolved,
            "notes": self.notes,
            "migrated": self.migrated,
        }

    def summary(self) -> str:
        counts = self.to_dict()["counts"]
        head = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        state = "dry-run" if self.dry_run else ("applied" if self.applied else "no write")
        extra = " [review required]" if self.review_required else ""
        return f"pricing sync ({state}): {head or 'nothing to do'}{extra}"


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------


def _discovered_rows() -> list[tuple[str, Mapping[str, Any]]]:
    """``(agent, row)`` for every model discovery has cached.

    Imported lazily: ``sharedagents_agent`` imports this package's catalogue, so a
    module-level import here would close a cycle."""
    from sharedagents_agent.discovery.cache_lock import read_all_cached_models

    out: list[tuple[str, Mapping[str, Any]]] = []
    for agent, rows in (read_all_cached_models() or {}).items():
        for row in rows or []:
            if isinstance(row, Mapping) and str(row.get("id") or "").strip():
                out.append((agent, row))
    return out


def _iso_now(now: datetime | None) -> str:
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sync_pricing(
    *,
    path: str | Path | None = None,
    dry_run: bool = False,
    catalogue_body: Mapping[str, Any] | None = None,
    discovered_rows: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
    refresh_catalogue: bool = True,
    only: Sequence[str] | None = None,
    now: datetime | None = None,
    prefer_catalogue: bool = False,
) -> SyncReport:
    """Reconcile discovery + catalogue under a per-overlay writer lock.

    Dry runs and disabled autosync remain read-only. Concurrent discovery,
    CLI and web writers must read the last committed overlay before deciding
    what to keep, otherwise one run can discard another run's new models.
    ``only`` filters creation; existing auto rows still refresh and retire.
    ``prefer_catalogue`` persists an opt-in to migrate supported manual model
    rows and refresh catalogue rates even when the bundled asset prices them.
    """
    from sharedagents_core.oscompat import file_lock

    def reconcile() -> SyncReport:
        return _sync_pricing_unlocked(
            path=path,
            dry_run=dry_run,
            catalogue_body=catalogue_body,
            discovered_rows=discovered_rows,
            refresh_catalogue=refresh_catalogue,
            only=only,
            now=now,
            prefer_catalogue=prefer_catalogue,
        )

    if dry_run or not autosync_enabled():
        return reconcile()
    user_path = Path(path).expanduser() if path else _default_pricing_path()
    auto_path = _pricing_auto_path_for(user_path)
    auto_path.parent.mkdir(parents=True, exist_ok=True)
    with auto_path.with_suffix(".lock").open("a+b") as lock, file_lock(lock):
        return reconcile()


def _sync_pricing_unlocked(
    *,
    path: str | Path | None = None,
    dry_run: bool = False,
    catalogue_body: Mapping[str, Any] | None = None,
    discovered_rows: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
    refresh_catalogue: bool = True,
    only: Sequence[str] | None = None,
    now: datetime | None = None,
    prefer_catalogue: bool = False,
) -> SyncReport:
    """Reconcile the auto overlay against discovery + the model catalogue.

    Writes nothing when ``dry_run`` is set, when the autosync switch is off, or
    when the catalogue cannot be read. Every decision lands in the report either
    way.

    ``catalogue_body`` / ``discovered_rows`` inject fixtures instead of reading
    the network and the shared discovery cache (the tests use them).

    ``only`` restricts which discovered ids may be CREATED (a prefix match), for
    a bounded, reviewable run. Refresh and retire of rows we already own are
    never filtered: leaving a known-wrong rate in place would be worse than the
    run being broad.
    """
    user_path = Path(path).expanduser() if path else _default_pricing_path()
    auto_path = _pricing_auto_path_for(user_path)
    report = SyncReport(auto_path=str(auto_path), dry_run=dry_run, disabled=not autosync_enabled())

    if report.disabled:
        report.notes.append(f"{AUTOSYNC_ENV} is off — reporting only, nothing will be written")

    # Refuse to touch an overlay we cannot parse: rewriting it would discard
    # whatever it holds, and we cannot tell what that is.
    auto_data: dict[str, Any] = {"version": 1, "entries": []}
    if auto_path.exists():
        loaded, err = _read_pricing_dict(auto_path)
        if loaded is None:
            report.stale = True
            report.notes.append(f"auto overlay {auto_path} is unreadable ({err}); left untouched")
            return report
        auto_data = loaded

    layers = _pricing_layers(user_path)
    bundled_rows = next((d.get("entries") or [] for n, d in layers if n == LAYER_BUNDLED), [])
    operator_rows = next((d.get("entries") or [] for n, d in layers if n == LAYER_OPERATOR), [])
    operator_data = next((d for n, d in layers if n == LAYER_OPERATOR), {})
    catalogue_first = prefer_catalogue or operator_data.get("prefer_catalogue") is True
    auto_rows = [r for r in (auto_data.get("entries") or []) if isinstance(r, dict)]

    # Catalogue: injected fixture, else the local slim copy (refreshed first if
    # stale). An empty map means we cannot price anything -- report, never guess.
    providers: Mapping[str, Any] = {}
    revision: int | None = None
    source = ""
    if catalogue_body is not None:
        slim = catalogue._slim(dict(catalogue_body)) or {}
        providers = slim.get("providers") or {}
        source = "injected"
    else:
        if refresh_catalogue and catalogue._source_url():
            try:
                catalogue.refresh()
            except Exception as exc:  # noqa: BLE001 -- the sync must still report
                report.notes.append(f"catalogue refresh failed: {exc}")
        providers = catalogue.slim_providers()
        revision, source = catalogue.catalogue_revision()
    report.catalogue_source = source
    report.catalogue_revision = revision
    if not providers:
        report.stale = True
        report.notes.append(
            "no model catalogue available (disabled source, stale copy, or no "
            "per-model costs) — nothing priced from upstream this run"
        )

    bundled_by_id = _priced_ids(bundled_rows)
    operator_by_id = _priced_ids(operator_rows)
    auto_by_id = {str((row.get("match") or {}).get("id") or ""): row for row in auto_rows}
    # Explicit opt-in: migrate only exact model rows with guarded catalogue
    # rates. Provider fallbacks, unavailable models and tombstones stay manual.
    eligible: dict[str, dict[str, Any]] = {}
    if catalogue_first:
        for row in [*bundled_rows, *operator_rows]:
            if not isinstance(row, dict):
                continue
            match = row.get("match") or {}
            mid = str(match.get("id") or "").strip()
            if not mid:
                continue
            eligible.pop(mid, None)
            if row.get("disabled") or (only and not any(mid.startswith(prefix) for prefix in only)):
                continue
            cost, _pid, _cid, _dis = _choose_cost(_upstream_costs(providers, row, mid))
            if cost is not None and _guard(cost) is None:
                eligible[mid] = row
        retained_operator = [
            row
            for row in operator_rows
            if not (
                isinstance(row, dict)
                and not row.get("disabled")
                and str((row.get("match") or {}).get("id") or "").strip() in eligible
            )
        ]
        report.migrated = [str(row["match"]["id"]) for row in operator_rows if row not in retained_operator]
        operator_by_id = _priced_ids(retained_operator)
    else:
        retained_operator = operator_rows
    fetched_at = _iso_now(now)

    # Taken names: never invent an alias another row already answers to.
    # resolve() matches aliases BEFORE ids, so an alias equal to a different
    # model's ID would hijack that model's turns -- both spellings count.
    taken_aliases: set[str] = set()
    for _row in [*bundled_rows, *operator_rows, *auto_rows]:
        if not isinstance(_row, dict):
            continue
        _match = value if isinstance(value := _row.get("match"), dict) else {}
        for _key in (_match.get("id"), _match.get("alias")):
            if _key:
                taken_aliases.add(str(_key).strip())
    taken_aliases.discard("")

    # ── 1. The auto rows we own: refresh from upstream, retire if a higher
    #       layer has since taken the model over.
    kept_auto: list[dict[str, Any]] = []
    for row in auto_rows:
        match = value if isinstance(value := row.get("match"), dict) else {}
        model_id = str(match.get("id") or "").strip()
        alias = str(match.get("alias") or "").strip()
        if alias in eligible and alias != model_id:
            # An exact migrated id must not be hijacked by an older alias.
            row = dict(row)
            match = dict(match)
            match.pop("alias", None)
            row["match"] = match
            report.refreshed[model_id] = {"from": _rates(row), "alias_removed": alias}
        meta = value if isinstance(value := row.get(AUTO_META_KEY), dict) else {}
        hint = str(meta.get("upstream_provider") or "")
        if model_id and (model_id in operator_by_id or (model_id in bundled_by_id and not catalogue_first)):
            # A hand-authored row now covers it: retire ours rather than let a
            # stale machine rate shadow it.
            report.retired.append(model_id)
            continue
        if not model_id or not providers:
            kept_auto.append(row)
            continue
        found = _upstream_costs(providers, row, model_id, hint)
        cost, pid, catalogue_id, disagreed = _choose_cost(found)
        if cost is None:
            report.unresolved.append(model_id)
            kept_auto.append(row)
            continue
        if disagreed:
            report.review_required = True
            report.notes.append(f"{model_id}: catalogue providers disagree on rates")
        previous = eligible.get(model_id, row)
        target_alias = str((previous.get("match") or {}).get("alias") or "").strip()
        if target_alias in eligible and target_alias != model_id:
            target_alias = ""
        schedule_keys = ("peak_multiplier", "peak_hours_utc", "peak_weekdays_only")
        metadata_same = target_alias == str(match.get("alias") or "").strip() and all(
            row.get(k) == previous.get(k) for k in schedule_keys
        )
        if _same_rates(row, cost) and metadata_same:
            kept_auto.append(row)
            continue
        reason = _guard(cost)
        if reason is not None:
            report.rejected[model_id] = reason
            kept_auto.append(row)  # keep the last-good rate rather than drop it
            continue
        if _creates_review(row, cost):
            report.review_required = True
        alias = target_alias
        kept_auto.append(
            _build_row(
                model_id=model_id,
                alias=alias,
                cost=cost,
                provider_id=pid or "",
                catalogue_model_id=catalogue_id or model_id,
                source=source,
                revision=revision,
                fetched_at=fetched_at,
                previous=previous,
            )
        )
        report.refreshed[model_id] = {
            "from": _rates(row),
            "to": {k: cost.get(k) for k in ("input", "output", "cached_read", "cached_write")},
        }

    # ── 2. Models discovery has seen that nothing prices yet.
    seen: dict[str, Mapping[str, Any]] = {}
    discovery = list(discovered_rows if discovered_rows is not None else _discovered_rows())
    if catalogue_first:
        discovery.extend(("pricing", dict(row, id=mid)) for mid, row in eligible.items())
    for _agent, row in discovery:
        model_id = str(row.get("id") or "").strip()
        if model_id and model_id not in seen:
            seen[model_id] = row
    # Migration must use the source row whose provider passed the guard, not
    # a discovery row for a different endpoint with the same bare model id.
    for mid, row in eligible.items():
        seen[mid] = dict(row, id=mid)
    if only:
        seen = {mid: row for mid, row in seen.items() if any(mid.startswith(prefix) for prefix in only)}
        report.notes.append(f"scoped to prefixes {', '.join(only)}")
    report.candidates_seen = len(seen)
    # Reserve every discovered id before generating aliases, including ids
    # outside a scoped run that may be priced in the next reconcile.
    taken_aliases.update(str(row.get("id") or "").strip() for _agent, row in discovery)

    for model_id, row in seen.items():
        if model_id in operator_by_id:
            report.operator_owned.append(model_id)
            continue
        if model_id in bundled_by_id and not catalogue_first:
            lower = bundled_by_id[model_id]
            report.bundled_owned.append(model_id)
            found = _upstream_costs(providers, row, model_id)
            cost, _pid, _cid, _dis = _choose_cost(found)
            if cost is not None and not _same_rates(lower, cost):
                # Not ours to change: the repo audit owns these rows (they carry
                # schedules the catalogue does not publish). Report the drift.
                report.bundled_drift[model_id] = {
                    "table": _rates(lower),
                    "upstream": {k: cost.get(k) for k in ("input", "output", "cached_read", "cached_write")},
                }
            continue
        if model_id in auto_by_id:
            continue  # handled in pass 1
        found = _upstream_costs(providers, row, model_id)
        cost, pid, catalogue_id, disagreed = _choose_cost(found)
        if cost is None:
            report.unresolved.append(model_id)
            continue
        reason = _guard(cost)
        if reason is not None:
            report.rejected[model_id] = reason
            continue
        if disagreed:
            report.review_required = True
            report.notes.append(f"{model_id}: catalogue providers disagree on rates")
        previous = eligible.get(model_id)
        if previous and _creates_review(previous, cost):
            report.review_required = True
            report.notes.append(f"{model_id}: migration changes a rate by {REVIEW_FACTOR:g}x or more")
        alias = (
            str((previous.get("match") or {}).get("alias") or "").strip()
            if previous
            else str(row.get("alias") or "").strip()
        )
        if alias and alias in taken_aliases and not previous:
            alias = ""  # never shadow another row's alias; the id still resolves
        if alias:
            taken_aliases.add(alias)
        kept_auto.append(
            _build_row(
                model_id=model_id,
                alias=alias,
                cost=cost,
                provider_id=pid or "",
                catalogue_model_id=catalogue_id or model_id,
                source=source,
                revision=revision,
                fetched_at=fetched_at,
                previous=previous,
            )
        )
        report.created[model_id] = {
            "input": cost.get("input"),
            "output": cost.get("output"),
            "cached_read": cost.get("cached_read"),
            "cached_write": cost.get("cached_write"),
            "upstream_provider": pid,
            "alias": alias or None,
        }

    # ── 3. Write only when something actually changed.
    changed = bool(report.created or report.refreshed or report.retired)
    if changed and not dry_run and not report.disabled:
        out = dict(auto_data)
        out["version"] = int(auto_data.get("version") or 1)
        out["entries"] = kept_auto
        out.setdefault("source_url", source)
        _write_pricing_dict(auto_path, out)
        report.applied = True
    policy_changed = prefer_catalogue and operator_data.get("prefer_catalogue") is not True
    if (report.migrated or policy_changed) and not dry_run and not report.disabled and providers:
        # Write the overlay first: a failed migration must leave manual rates
        # authoritative, never expose a missing replacement.
        out_operator = dict(operator_data)
        out_operator["prefer_catalogue"] = True
        out_operator["entries"] = retained_operator
        current_operator, error = _read_pricing_dict(user_path) if user_path.exists() else ({}, None)
        if error is not None or current_operator != operator_data:
            raise RuntimeError("operator pricing changed during sync; retry the migration")
        _write_pricing_dict(user_path, out_operator)
        report.applied = True
    return report


def sync_after_discovery(
    *,
    path: str | Path | None = None,
    discovered_rows: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
) -> SyncReport | None:
    """Debounced, never-raising reconcile for the discovery hot path.

    Returns None when the sync is off, still inside its interval, or when it
    failed: discovery must never fail because pricing could not be refreshed."""
    if not autosync_enabled():
        return None
    key: str | None = None
    try:
        key = str(_pricing_auto_path_for(Path(path).expanduser() if path else None))
        now = time.monotonic()
        rows = [*_discovered_rows(), *(discovered_rows or [])]
        signature = tuple(
            sorted(
                {
                    tuple(str(row.get(field) or "") for field in ("id", "alias", "provider", "baseUrl", "endpoint"))
                    for _agent, row in rows
                }
            )
        )
        if now - _LAST_SYNC.get(key, float("-inf")) < min_interval_s and signature == _LAST_DISCOVERY.get(key):
            return None
        _LAST_DISCOVERY[key] = signature
        report = sync_pricing(path=path, discovered_rows=rows)
    except Exception as exc:  # noqa: BLE001 -- never break discovery
        _LOG.warning("model pricing sync failed: %s", exc)
        if key is not None:
            _LAST_SYNC[key] = time.monotonic() + min_interval_s - FAILURE_BACKOFF_S
        return None
    _LAST_SYNC[key] = now
    if report.changed:
        _LOG.info(
            "%s created=%d refreshed=%d retired=%d",
            report.summary(),
            len(report.created),
            len(report.refreshed),
            len(report.retired),
        )
    elif report.stale:
        _LOG.info("model pricing sync: %s", "; ".join(report.notes) or "catalogue unavailable")
    return report
