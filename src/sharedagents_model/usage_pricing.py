"""Per-model token cost accounting — normalizer + pricer.

Mirrors the converged OpenClaw / Hermes pattern (``agent/usage_pricing.py``):
a **normalizer** turns an engine's metrics dict into a provider-agnostic
:class:`CanonicalUsage`, and a **pricer** turns that into a :class:`CostResult`
using a per-model :class:`PricingTable`.

Money is carried as integer **pico-USD** (1e-12 USD). Costs are computed with
``Decimal`` (``tokens * rate_per_mtok / 1_000_000`` USD) then quantized to
pico-USD with ``ROUND_HALF_UP`` — exact for any realistic ≤6-decimal ``$/Mtok``
rate, so tiny per-turn costs never truncate to zero over a long-running goal.

Token buckets (Hermes canonical shape):
  input         — fresh prefill / prompt tokens
  cached_read   — prompt tokens served from cache
  cached_write  — prompt tokens written to cache (best-effort; usually absent)
  thinking      — reasoning tokens
  output        — generated result tokens (excludes thinking)

Rate fallbacks when a provider doesn't separate a bucket:
  cached_read / cached_write → input ;  thinking → output.

Peak/off-peak schedules: some suppliers (DeepSeek) publish an off-peak list
rate and bill a fixed multiple of it inside named UTC windows on weekdays.
A pricing entry may carry ``peak_multiplier`` + ``peak_hours_utc`` (half-open
``[start, end)`` hour ranges) + ``peak_weekdays_only``; the pricer then needs
the wall-clock ``when`` of the turn to pick the right column.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("sharedagents.usage_pricing")

PICO_PER_USD = 1_000_000_000_000  # 1e12
_MTOK = Decimal(1_000_000)
_ONE = Decimal(1)

# Ordered token buckets shared by the normalizer, pricer, and ledger.
BUCKETS = ("input", "cached_read", "cached_write", "thinking", "output")


def usd_to_pico(usd: float | Decimal | int) -> int:
    """USD → integer pico-USD (ROUND_HALF_UP)."""
    d = usd if isinstance(usd, Decimal) else Decimal(str(usd))
    return int((d * PICO_PER_USD).quantize(_ONE, rounding=ROUND_HALF_UP))


def pico_to_usd(pico: int) -> float:
    """Integer pico-USD → USD float (for display only)."""
    return float(Decimal(int(pico)) / PICO_PER_USD)


# ---------------------------------------------------------------------------
# Canonical usage
# ---------------------------------------------------------------------------


@dataclass
class CanonicalUsage:
    input: int = 0
    cached_read: int = 0
    cached_write: int = 0
    thinking: int = 0
    output: int = 0
    model: str = ""
    model_alias: str = ""
    provider: str = ""
    estimated: bool = False

    def total_tokens(self) -> int:
        return self.input + self.cached_read + self.cached_write + self.thinking + self.output


def _int(value: Any) -> int:
    try:
        n = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def normalize_usage(metrics: dict[str, Any] | None) -> CanonicalUsage:
    """Map an engine metrics dict into provider-agnostic token buckets.

    Reuses the existing sharedagents metrics keys (``metrics_format``):
    ``prompt_tokens`` / ``prompt_tokens_cached`` / ``prompt_tokens_fresh`` /
    ``reasoning_tokens`` / ``completion_tokens`` / ``content_tokens`` /
    ``model`` / ``model_alias`` / ``provider`` / ``tokens_estimated``.

    Output is the *non-thinking* generated tokens so reasoning is never priced
    twice (once as ``thinking`` and again inside ``output``).

    ``prompt_tokens_billed*`` take precedence over ``prompt_tokens*`` when
    present: CLI engines (claude_cli) narrow the DISPLAYED ``prompt_tokens`` to
    the last API call's context occupancy (so the ctx-meter isn't nonsense on a
    long agentic turn) but stash the turn's CUMULATIVE totals under the
    ``*_billed*`` keys so billing still charges every internal cache read/write.
    """
    m = metrics or {}
    # Prefer the cumulative billed totals (CLI engines set these); fall back to
    # the plain display fields for every other engine, where they coincide.
    # The input bucket is the uncached prompt remainder. Cache creation/write
    # tokens are their own canonical bucket and must not be counted twice.
    if m.get("prompt_tokens_billed") is not None:
        prompt = _int(m.get("prompt_tokens_billed"))
        cached = _int(m.get("prompt_tokens_billed_cached"))
        cache_write = _int(m.get("prompt_tokens_billed_cache_write"))
        input_tokens = max(prompt - cached - cache_write, 0)
    else:
        prompt = _int(m.get("prompt_tokens"))
        cached = _int(m.get("prompt_tokens_cached"))
        cache_write = _int(m.get("prompt_tokens_cache_write"))
        fresh = m.get("prompt_tokens_fresh")
        input_tokens = _int(fresh) if fresh is not None else max(prompt - cached, 0)

    thinking = _int(m.get("reasoning_tokens"))
    content = m.get("content_tokens")
    if content is not None:
        output = _int(content)
    else:
        comp = _int(m.get("completion_tokens"))
        output = max(comp - thinking, 0)

    return CanonicalUsage(
        input=input_tokens,
        cached_read=cached,
        cached_write=cache_write,
        thinking=thinking,
        output=output,
        model=str(m.get("model") or ""),
        model_alias=str(m.get("model_alias") or ""),
        provider=str(m.get("provider") or ""),
        estimated=bool(m.get("tokens_estimated")),
    )


def metrics_from_provider_usage(
    usage: dict[str, Any] | None,
    *,
    model: str = "",
    model_alias: str = "",
    provider: str = "",
) -> dict[str, Any]:
    """Convert common provider usage payloads into sharedagents metrics.

    Engines receive several wire shapes in practice: Anthropic uses
    ``input_tokens`` plus cache read/write fields, OpenAI-compatible APIs use
    ``prompt_tokens``/``completion_tokens``, and DeepSeek exposes prompt-cache
    hit/miss fields.  The ledger must see the same canonical fields regardless
    of which engine produced the turn.

    The returned prompt fields describe *one provider call*. Callers that are
    aggregating a multi-call turn should pass the individual dictionaries to
    :func:`aggregate_billed_metrics` rather than summing this result manually.
    An empty result means the payload did not contain a usable token split.
    """
    if not isinstance(usage, dict) or not usage:
        return {}

    def first_int(*keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if value is None:
                continue
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number >= 0:
                return number
        return 0

    # OpenAI-compatible chat APIs nest prefix-cache hits under
    # ``prompt_tokens_details.cached_tokens``; Responses APIs use the same
    # shape under ``input_tokens_details``. Prefer that split when present,
    # then fall back to Anthropic/DeepSeek top-level names.
    prompt_details = usage.get("prompt_tokens_details")
    input_details = usage.get("input_tokens_details")
    nested_details = (
        prompt_details if isinstance(prompt_details, dict) else input_details if isinstance(input_details, dict) else {}
    )
    nested_cached = 0
    nested_cache_present = False
    for key in ("cached_tokens", "cache_read_input_tokens"):
        if key not in nested_details:
            continue
        nested_cache_present = True
        try:
            nested_cached = max(0, int(nested_details.get(key) or 0))
        except (TypeError, ValueError):
            nested_cached = 0
        break
    cached = (
        nested_cached
        if nested_cache_present
        else first_int(
            "cache_read_input_tokens",
            "prompt_cache_hit_tokens",
            "cached_read_tokens",
            "prompt_tokens_cached",
        )
    )
    cache_write = first_int(
        "cache_creation_input_tokens",
        "prompt_cache_write_tokens",
        "cached_write_tokens",
        "prompt_tokens_cache_write",
    )
    prompt_total = first_int("prompt_tokens", "total_prompt_tokens")
    input_tokens = first_int("input_tokens")
    fresh = first_int("prompt_cache_miss_tokens", "prompt_tokens_fresh")
    explicit_fresh = any(usage.get(key) is not None for key in ("prompt_cache_miss_tokens", "prompt_tokens_fresh"))
    # ``input_tokens`` is Anthropic's uncached bucket, but OpenAI Responses
    # uses it as the total input count when ``input_tokens_details`` is
    # present. The nested cache shape disambiguates those protocols.
    if prompt_total > 0:
        if not explicit_fresh:
            fresh = max(prompt_total - cached - cache_write, 0)
    elif input_tokens > 0:
        if nested_cache_present:
            prompt_total = input_tokens
            fresh = max(input_tokens - cached - cache_write, 0)
        else:
            fresh = input_tokens
            prompt_total = fresh + cached + cache_write
    else:
        prompt_total = fresh + cached + cache_write

    completion = first_int("completion_tokens", "output_tokens")
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = usage.get("output_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}
    thinking = first_int(
        "reasoning_tokens",
        "thinking_tokens",
        "completion_tokens_reasoning",
    )
    if thinking <= 0:
        for key in ("reasoning_tokens", "thinking_tokens"):
            if key not in completion_details:
                continue
            try:
                thinking = max(0, int(completion_details.get(key) or 0))
            except (TypeError, ValueError):
                thinking = 0
            break
    content = first_int("content_tokens", "completion_tokens_content")
    if content <= 0 and completion > 0:
        content = max(completion - thinking, 0)

    if prompt_total <= 0 and completion <= 0 and thinking <= 0 and content <= 0:
        return {}

    result: dict[str, Any] = {
        "prompt_tokens": prompt_total,
        "prompt_tokens_fresh": fresh,
        "prompt_tokens_cached": cached,
        "prompt_tokens_cache_write": cache_write,
        # Mark the provider-call total explicitly. This avoids ambiguity for
        # cache-creation tokens: Anthropic's prompt total includes writes,
        # while some local metrics expose writes as a separate bucket.
        "prompt_tokens_billed": prompt_total,
        "prompt_tokens_billed_cached": cached,
        "prompt_tokens_billed_cache_write": cache_write,
        "completion_tokens": completion,
        "reasoning_tokens": thinking,
        "content_tokens": content,
        "total_tokens": prompt_total + completion,
        "tokens_estimated": False,
    }
    if model:
        result["model"] = model
    if model_alias:
        result["model_alias"] = model_alias
    if provider:
        result["provider"] = provider
    return result


def aggregate_billed_metrics(
    rounds: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Build a cumulative ledger snapshot from per-call metrics.

    ``prompt_tokens`` remains the latest context occupancy for display, while
    ``prompt_tokens_billed*`` and generated-token fields are cumulative. This
    lets a live checkpoint and the terminal finalizer write the same stable
    turn row without confusing the context meter with billed prefill.
    """
    valid = [dict(r) for r in rounds if isinstance(r, dict)]
    if not valid:
        return {}
    latest = dict(valid[-1])
    billed = 0
    cached = 0
    cache_write = 0
    cost_overrides: list[Decimal] = []
    for rm in valid:
        if rm.get("cost_usd_override") is not None:
            try:
                cost_overrides.append(Decimal(str(rm["cost_usd_override"])))
            except (TypeError, ValueError, ArithmeticError):
                pass
        if rm.get("prompt_tokens_billed") is not None:
            billed += _int(rm.get("prompt_tokens_billed"))
            cached += _int(rm.get("prompt_tokens_billed_cached"))
            cache_write += _int(rm.get("prompt_tokens_billed_cache_write"))
            continue
        prompt = _int(rm.get("prompt_tokens"))
        round_cached = _int(rm.get("prompt_tokens_cached"))
        round_write = _int(rm.get("prompt_tokens_cache_write"))
        if prompt <= 0:
            prompt = _int(rm.get("prompt_tokens_fresh")) + round_cached
        billed += prompt
        cached += round_cached
        cache_write += round_write

    latest.update(
        {
            "prompt_tokens_billed": billed,
            "prompt_tokens_billed_cached": cached,
            "prompt_tokens_billed_cache_write": cache_write,
        }
    )
    if cost_overrides:
        latest["cost_usd_override"] = float(sum(cost_overrides, Decimal(0)))
    for key in ("completion_tokens", "reasoning_tokens", "content_tokens"):
        total = sum(_int(rm.get(key)) for rm in valid)
        if total > 0:
            latest[key] = total
    latest["total_tokens"] = _int(latest.get("prompt_tokens")) + _int(latest.get("completion_tokens"))
    return latest


# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------


def _bundled_default_pricing_path() -> Path | None:
    """Resolve the bundled default pricing asset (ships inside the package)."""
    path = Path(__file__).resolve().parent / "data" / "default_pricing.json"
    return path if path.exists() else None


def _rate(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001
        return None


@dataclass
class PricingEntry:
    provider: str = ""
    match: dict[str, str] = field(default_factory=dict)
    input_per_mtok: Decimal = Decimal(0)
    cached_read_per_mtok: Decimal | None = None
    cached_write_per_mtok: Decimal | None = None
    thinking_per_mtok: Decimal | None = None
    output_per_mtok: Decimal = Decimal(0)
    # Optional time-of-day schedule (DeepSeek-style peak/off-peak): the base
    # rates above are the OFF-PEAK list; inside a peak window every bucket is
    # billed at ``base * peak_multiplier``. ``peak_hours_utc`` holds half-open
    # ``[start_hour, end_hour)`` ranges; ``peak_weekdays_only`` skips Sat/Sun.
    # Absent fields mean "flat rate all day" — the historical behaviour.
    peak_multiplier: Decimal | None = None
    peak_hours_utc: tuple[tuple[int, int], ...] = ()
    peak_weekdays_only: bool = False
    #: Which layer this row came from (``bundled`` | ``auto`` | ``operator``),
    #: set by :meth:`PricingTable.load`. Presentation/audit only — resolution
    #: never consults it.
    layer: str = ""
    #: Catalogue provenance (also retained when an automatic row is overridden
    #: by the operator), or None for entirely hand-written rows.
    provenance: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PricingEntry:
        return cls(
            provider=str(d.get("provider") or ""),
            match={k: str(v) for k, v in (d.get("match") or {}).items()},
            input_per_mtok=_rate(d.get("input_per_mtok")) or Decimal(0),
            cached_read_per_mtok=_rate(d.get("cached_read_per_mtok")),
            cached_write_per_mtok=_rate(d.get("cached_write_per_mtok")),
            thinking_per_mtok=_rate(d.get("thinking_per_mtok")),
            output_per_mtok=_rate(d.get("output_per_mtok")) or Decimal(0),
            peak_multiplier=_rate(d.get("peak_multiplier")),
            peak_hours_utc=_peak_windows(d.get("peak_hours_utc")),
            peak_weekdays_only=bool(d.get("peak_weekdays_only")),
        )

    def peak_multiplier_for(self, when: datetime | None) -> Decimal | None:
        """Multiplier to apply at ``when``, or None off-peak / unscheduled.

        ``when`` is interpreted as UTC (naive timestamps are taken as UTC,
        matching the ledger's ``created_at``). A multiplier is returned only
        when the entry declares a schedule and ``when`` falls inside one of
        its windows; weekend timestamps are off-peak when
        ``peak_weekdays_only`` is set.
        """
        if when is None or self.peak_multiplier is None or not self.peak_hours_utc:
            return None
        ts = when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)
        if self.peak_weekdays_only and ts.weekday() >= 5:
            return None
        hour = ts.hour + ts.minute / 60.0
        for start, end in self.peak_hours_utc:
            if start <= hour < end:
                return self.peak_multiplier
        return None

    def base_rate_for(self, bucket: str) -> Decimal:
        """Off-peak (list) rate for a bucket, applying bucket fallbacks."""
        if bucket == "input":
            return self.input_per_mtok
        if bucket == "output":
            return self.output_per_mtok
        if bucket == "thinking":
            return self.thinking_per_mtok if self.thinking_per_mtok is not None else self.output_per_mtok
        if bucket == "cached_read":
            return self.cached_read_per_mtok if self.cached_read_per_mtok is not None else self.input_per_mtok
        if bucket == "cached_write":
            return self.cached_write_per_mtok if self.cached_write_per_mtok is not None else self.input_per_mtok
        return Decimal(0)

    def rate_for(self, bucket: str, *, when: datetime | None = None) -> Decimal:
        """Effective rate for a bucket at ``when``, applying fallbacks + peak.

        ``when=None`` (no known turn time) prices at the off-peak base rate.
        """
        rate = self.base_rate_for(bucket)
        multiplier = self.peak_multiplier_for(when)
        if multiplier is not None:
            return rate * multiplier
        return rate


def _peak_windows(value: Any) -> tuple[tuple[int, int], ...]:
    """Parse ``[[start, end], …]`` UTC hour windows, ignoring malformed rows."""
    windows: list[tuple[int, int]] = []
    if isinstance(value, (list, tuple)):
        for raw in value:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                continue
            try:
                start, end = int(raw[0]), int(raw[1])
            except (TypeError, ValueError):
                continue
            if 0 <= start < end <= 24:
                windows.append((start, end))
    return tuple(windows)


#: The three pricing layers, lowest precedence first. ``load`` merges them per
#: row key so a machine-written rate can fill a gap without ever overriding a
#: rate the operator wrote.
LAYER_BUNDLED = "bundled"
LAYER_AUTO = "auto"
LAYER_OPERATOR = "operator"

#: File name of the machine-managed overlay, beside the operator's file.
AUTO_PRICING_FILENAME = "pricing_auto.json"


def _default_pricing_path() -> Path:
    from sharedagents_core.layout import pricing_path

    return pricing_path()


def _pricing_auto_path_for(user_path: Path | None) -> Path:
    """The auto overlay's path for a given operator file.

    A sibling -- not a fixed root path -- so an ``AgentConfig.pricing_path``
    override gets its own overlay instead of sharing the default one.
    """
    base = user_path if user_path is not None else _default_pricing_path()
    return base.with_name(AUTO_PRICING_FILENAME)


def _read_pricing_dict(path: Path) -> tuple[dict[str, Any] | None, Exception | None]:
    """Parse a pricing file, returning ``(data, error)`` and never raising."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- unreadable or invalid JSON
        return None, exc
    if not isinstance(loaded, dict):
        return None, ValueError("not a JSON object")
    return loaded, None


def _layer_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    """A row's identity across layers.

    A row that names a model id is identified by THAT alone, ignoring its
    top-level ``provider`` tag: the tag is not how the pricer looks a model up
    (:meth:`PricingTable.resolve` matches ``match.id`` whatever the provider
    says), so two rows differing only by that tag are the same row and the
    higher layer must replace the lower one. Keying them apart let the bundled
    row survive underneath an operator's override and win on iteration order --
    which silently broke "the operator always wins".

    Without a model id, the explicit ``match.provider`` is the identity; failing
    that, the alias -- ``_match_key`` collapses every alias-only row onto one
    key, so a merge would keep only the last.
    """
    match = entry.get("match")
    match = match if isinstance(match, dict) else {}
    model_id = str(match.get("id") or "").strip()
    if model_id:
        return ("", model_id, "")
    provider = str(entry.get("provider") or "").strip()
    match_provider = str(match.get("provider") or "").strip()
    if match_provider:
        return (provider, "", match_provider)
    return (provider, "", str(match.get("alias") or "").strip())


def _merge_pricing_layers(
    layers: Sequence[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    """Merge pricing layers low -> high precedence, per row key.

    A higher layer's row replaces a lower layer's row with the same key
    wholesale: an operator edit is authoritative, never a field-wise patch of a
    machine-written row.

    A row marked ``"disabled": true`` in a higher layer drops that key from the
    merge entirely, so a deliberately removed row cannot reappear from a lower
    layer. Resolution then falls through to the provider tag or the default
    floor -- exactly what deleting that row did when one file was the whole
    table.

    Returns the surviving ``(layer_name, row)`` pairs, HIGHEST precedence
    first (the order ``resolve`` must see them in).
    """
    winners: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}
    order: list[tuple[str, str, str]] = []
    for name, data in layers:
        for entry in data.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            key = _layer_key(entry)
            if key == ("", "", ""):
                continue  # unaddressable: it could never resolve anyway
            if key not in winners:
                order.append(key)
            winners[key] = (name, entry)
    out: list[tuple[str, dict[str, Any]]] = []
    for key in order:
        name, entry = winners[key]
        if entry.get("disabled"):
            continue
        out.append((name, entry))
    # Highest layer FIRST: `resolve` returns the first match, so ordering by
    # precedence makes "a higher layer wins" hold for the provider-wide passes
    # too, not just for exact ids. Stable, so each layer keeps its file order
    # (and a table with no higher layer at all is left exactly as it was).
    rank = {name: index for index, (name, _data) in enumerate(layers)}
    out.sort(key=lambda item: -rank.get(item[0], 0))
    return out


def _merge_pricing_defaults(layers: Sequence[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Field-wise merge of the layers' ``default`` blocks (low -> high).

    Field-wise, unlike a row: a user file carrying a partial ``default`` must
    not blank the bundled non-zero floor, which is what keeps an unmatched
    model accruing cost instead of silently defeating a /goal budget.
    """
    out: dict[str, Any] = {}
    for _name, data in layers:
        block = data.get("default")
        if isinstance(block, dict):
            out.update({k: v for k, v in block.items() if v is not None})
    return out


def _pricing_layers(user_path: Path | None, *, strict: bool = False) -> list[tuple[str, dict[str, Any]]]:
    """The readable pricing layers for ``user_path``, lowest precedence first.

    Bundled asset, then the auto overlay, then the operator's file. A layer
    that is absent, unreadable, or not an object is skipped with a warning
    rather than aborting the merge: a corrupt machine-written overlay must
    never zero or block the operator's rates. The operator's file is the one
    exception -- ``load(strict=True)`` re-raises for it, so the hot-reload path
    can keep the last-good table during a transient bad edit.
    """
    layers: list[tuple[str, dict[str, Any]]] = []

    bundled = _bundled_default_pricing_path()
    if bundled is not None:
        data, err = _read_pricing_dict(bundled)
        if data is not None:
            layers.append((LAYER_BUNDLED, data))
        elif err is not None:
            _LOG.warning("bundled pricing table %s unreadable: %s", bundled, err)

    auto_path = _pricing_auto_path_for(user_path)
    if auto_path.exists():
        data, err = _read_pricing_dict(auto_path)
        if data is not None:
            layers.append((LAYER_AUTO, data))
        else:
            _LOG.warning("auto pricing overlay %s unreadable: %s", auto_path, err)

    if user_path is not None and user_path.exists():
        data, err = _read_pricing_dict(user_path)
        if data is not None:
            layers.append((LAYER_OPERATOR, data))
        elif err is not None:
            _LOG.warning("pricing table %s unreadable: %s", user_path, err)
            if strict:
                raise err
    return layers


@dataclass
class PricingTable:
    entries: list[PricingEntry] = field(default_factory=list)
    default: PricingEntry = field(default_factory=PricingEntry)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PricingTable:
        entries = [PricingEntry.from_dict(e) for e in (data.get("entries") or [])]
        default = PricingEntry.from_dict(data.get("default") or {})
        return cls(entries=entries, default=default)

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        *,
        strict: bool = False,
        standalone: bool = False,
    ) -> PricingTable:
        """Load the pricing table by merging three layers, low precedence first:

        1. the bundled asset (``sharedagents_model/data/default_pricing.json``);
        2. the machine-managed overlay (:data:`AUTO_PRICING_FILENAME`, beside
           the operator's file) -- the rates ``sharedagents_model.pricing_sync``
           derived from the model catalogue for models nothing else covers;
        3. the explicit ``path`` / ``AgentConfig.pricing_path`` / the default
           ``~/.sharedagents/pricing.json`` -- the operator's own file.

        Merging is per row key, so an operator's row always wins while a model
        the bundled asset does not cover can still be priced from the overlay.
        A row marked ``"disabled": true`` in a higher layer suppresses the same
        key from every lower layer.

        The bundled layer's NON-ZERO default survives unless a higher layer
        overrides that exact field, so a missing or partial user file never
        silently zeroes every cost (which would make a /goal budget
        unenforced). Only if no layer is readable at all do we return the
        all-zero table.

        ``strict`` re-raises when the USER file exists but fails to parse,
        instead of falling back to the lower layers. The hot-reload path uses
        this to keep the last-good table during a transient bad edit.

        ``standalone`` reads ONE file as the whole table (the first readable of
        ``path`` then the bundled asset), with no merge. A tool comparing one
        table against another needs the file as written: merging the CURRENT
        bundled asset underneath an old copy would put rows into it that were
        added after the copy was taken.
        """
        user_path = Path(path).expanduser() if path else _default_pricing_path()
        if standalone:
            candidates = [user_path]
            bundled = _bundled_default_pricing_path()
            if bundled is not None:
                candidates.append(bundled)
            for candidate in candidates:
                if not candidate.exists():
                    continue
                data, err = _read_pricing_dict(candidate)
                if data is not None:
                    return cls.from_dict(data)
                _LOG.warning("pricing table %s unreadable: %s", candidate, err)
                if strict and candidate == user_path and err is not None:
                    raise err
            _LOG.warning("no readable pricing table — costs will be zero")
            return cls()

        layers = _pricing_layers(user_path, strict=strict)
        if not layers:
            _LOG.warning("no readable pricing table — costs will be zero")
            return cls()

        entries: list[PricingEntry] = []
        for name, row in _merge_pricing_layers(layers):
            entry = PricingEntry.from_dict(row)
            entry.layer = name
            provenance = row.get("auto")
            if isinstance(provenance, dict):
                entry.provenance = dict(provenance)
            entries.append(entry)
        return cls(
            entries=entries,
            default=PricingEntry.from_dict(_merge_pricing_defaults(layers)),
        )

    def resolve(self, usage: CanonicalUsage) -> PricingEntry:
        """Match precedence: model_alias → alias → wire id (with suffix tolerance)
        → explicit match.provider → top-level provider tag → default.

        Explicit match.provider entries always beat top-level provider tags so
        table iteration order cannot change the resolution outcome.
        """
        # Pass 1a: model_alias on match.alias
        if usage.model_alias:
            for e in self.entries:
                if e.match.get("alias") == usage.model_alias:
                    return e
        # Pass 1b: model on match.alias
        if usage.model:
            for e in self.entries:
                if e.match.get("alias") == usage.model:
                    return e
        # Pass 1c: exact model id, falling back to suffix-tolerant match for [1m] etc.
        if usage.model:
            for e in self.entries:
                if e.match.get("id") == usage.model:
                    return e
            if "[" in usage.model and usage.model.endswith("]"):
                base_model = usage.model.split("[", 1)[0]
                for e in self.entries:
                    if e.match.get("id") == base_model:
                        return e
        # Pass 1d: explicit match.provider
        if usage.provider:
            for e in self.entries:
                if e.match.get("provider") == usage.provider:
                    return e
        # Pass 2: top-level provider tag
        if usage.provider:
            for e in self.entries:
                if e.provider == usage.provider:
                    return e
        # Pass 3: default floor
        return self.default


def _rate_json(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)


def serialize_pricing_entry(entry: PricingEntry, *, kind: str = "entry") -> dict[str, Any]:
    """Public row for the Usage Prices tab (USD per 1M tokens)."""
    match = dict(entry.match)
    label = match.get("id") or match.get("provider") or entry.provider or "*"
    row: dict[str, Any] = {
        "kind": kind,
        "provider": entry.provider or "",
        "match": match,
        "label": label,
        "input_per_mtok": float(entry.input_per_mtok),
        "cached_read_per_mtok": _rate_json(entry.cached_read_per_mtok),
        "cached_write_per_mtok": _rate_json(entry.cached_write_per_mtok),
        "output_per_mtok": float(entry.output_per_mtok),
        "cache_read_effective_per_mtok": float(entry.rate_for("cached_read")),
        "cache_write_effective_per_mtok": float(entry.rate_for("cached_write")),
    }
    # Which layer resolved this row, and (for the auto overlay) what it came
    # from -- the Prices tab shows both so an operator can tell a rate they
    # wrote from one the sync derived.
    if entry.layer:
        row["layer"] = entry.layer
    if entry.provenance:
        row["provenance"] = dict(entry.provenance)
    # Peak/off-peak schedule: surfaced so the UI can show
    # that the listed rates are the off-peak base (and what peak bills).
    if entry.peak_multiplier is not None and entry.peak_hours_utc:
        row["peak_multiplier"] = float(entry.peak_multiplier)
        row["peak_hours_utc"] = [[start, end] for start, end in entry.peak_hours_utc]
        row["peak_weekdays_only"] = bool(entry.peak_weekdays_only)
    return row


def _is_alias_only(entry: PricingEntry) -> bool:
    return bool(entry.match.get("alias")) and not entry.match.get("id") and not entry.match.get("provider")


def serialize_pricing_table(table: PricingTable) -> dict[str, Any]:
    """JSON snapshot of registered cache / input / output rates.

    Alias-only rows are lookup keys, not priced models — omit them.
    """
    return {
        "unit": "usd_per_mtok",
        "default": serialize_pricing_entry(table.default, kind="default"),
        "entries": [serialize_pricing_entry(e) for e in table.entries if not _is_alias_only(e)],
    }


MAX_RATE_USD = Decimal("100000")


class PricingSaveError(ValueError):
    """Invalid pricing payload — do not write the table."""


def _parse_optional_rate(value: Any, field_name: str = "rate") -> Decimal | None:
    if value is None or value == "":
        return None
    rate = _rate(value)
    if rate is None or not rate.is_finite() or rate < 0:
        raise PricingSaveError(f"invalid rate for {field_name}: {value!r} (must be a non-negative number)")
    if rate > MAX_RATE_USD:
        raise PricingSaveError(f"rate for {field_name} exceeds maximum allowable rate of ${MAX_RATE_USD}")
    return rate


def _parse_required_rate(value: Any, field: str) -> Decimal:
    rate = _parse_optional_rate(value, field)
    if rate is None:
        raise PricingSaveError(f"{field} is required")
    return rate


def _match_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    match = value if isinstance(value := entry.get("match"), dict) else {}
    return (
        str(entry.get("provider") or ""),
        str(match.get("id") or ""),
        str(match.get("provider") or ""),
    )


def _apply_rates(dest: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    out = dict(dest)
    new_input = float(_parse_required_rate(src.get("input_per_mtok"), "input_per_mtok"))
    new_output = float(_parse_required_rate(src.get("output_per_mtok"), "output_per_mtok"))
    out["input_per_mtok"] = new_input
    out["output_per_mtok"] = new_output
    cached = _parse_optional_rate(src.get("cached_read_per_mtok"), "cached_read_per_mtok")
    cached_w = _parse_optional_rate(src.get("cached_write_per_mtok"), "cached_write_per_mtok")
    if cached is None:
        out.pop("cached_read_per_mtok", None)
    else:
        out["cached_read_per_mtok"] = float(cached)
    if cached_w is None:
        out.pop("cached_write_per_mtok", None)
    else:
        out["cached_write_per_mtok"] = float(cached_w)
    old_out = _parse_optional_rate(dest.get("output_per_mtok"), "output_per_mtok")
    old_think = _parse_optional_rate(dest.get("thinking_per_mtok"), "thinking_per_mtok")
    if old_think is None or (old_out is not None and old_think == old_out):
        out["thinking_per_mtok"] = new_output
    return out


def _own_pricing_dict(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    """The operator's file ALONE -- never a lower layer's rows.

    Writes must distinguish rows this file defines from rows a lower layer
    supplies: ``delete_pricing_entry`` has to know whether to remove a row or
    to tombstone it, and cannot ask a merged view.
    """
    dest = Path(path).expanduser() if path else _default_pricing_path()
    existing: dict[str, Any] = {}
    if dest.exists():
        try:
            loaded = json.loads(dest.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise PricingSaveError(f"existing pricing file is unreadable: {exc}") from exc
        if not isinstance(loaded, dict):
            raise PricingSaveError("existing pricing file is not an object")
        existing = loaded
    return existing, dest


def _write_pricing_dict(dest: Path, data: dict[str, Any]) -> None:
    """Write a pricing file atomically (a reader never sees a half-written one)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, dest)


def _pricing_snapshot(path: str | Path | None, dest: Path) -> dict[str, Any]:
    """The serialized EFFECTIVE table after a write (all layers merged)."""
    from sharedagents_model.usage_ledger import get_pricing_table

    table = get_pricing_table(None if path is None else str(dest), reload=True)
    snap = serialize_pricing_table(table)
    snap["source"] = str(dest)
    return snap


def _entry_matches(entry: dict[str, Any], target_provider: str, target_id: str) -> bool:
    """Whether ``entry`` is the row an operator asked to delete."""
    eprov = str(entry.get("provider") or "").strip()
    ematch = value if isinstance(value := entry.get("match"), dict) else {}
    eid = str(ematch.get("id") or ematch.get("provider") or eprov).strip()
    ealias = str(ematch.get("alias") or "").strip()

    if target_provider and target_id:
        return eprov == target_provider and (
            eid == target_id or ematch.get("id") == target_id or ematch.get("provider") == target_id
        )
    if target_id:
        return (
            eid == target_id
            or ematch.get("id") == target_id
            or ealias == target_id
            or (eprov == target_id and not ematch.get("id"))
        )
    return False


def _tombstone_for(path: str | Path | None, target_provider: str, target_id: str) -> dict[str, Any] | None:
    """A ``disabled`` row suppressing the matching lower-layer row, or None.

    Copies the lower layer's own ``provider``/``match`` verbatim, so the
    tombstone lands on the same :func:`_layer_key` as the row it suppresses.
    """
    user_path = Path(path).expanduser() if path else _default_pricing_path()
    # Merged ROWS, not layers: find the row a lower layer actually resolves.
    for name, row in _merge_pricing_layers(_pricing_layers(user_path)):
        if name == LAYER_OPERATOR:
            continue
        if not _entry_matches(row, target_provider, target_id):
            continue
        match = row.get("match")
        return {
            "provider": str(row.get("provider") or ""),
            "match": ({str(k): str(v) for k, v in match.items()} if isinstance(match, dict) else {}),
            "disabled": True,
        }
    return None


def load_pricing_file_dict(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    """Read the target pricing.json (or bundled defaults), returning (dict, resolved_path)."""
    existing, dest = _own_pricing_dict(path)
    if existing or dest.exists():
        return existing, dest
    bundled = _bundled_default_pricing_path()
    if bundled is not None and bundled.exists():
        existing = json.loads(bundled.read_text(encoding="utf-8"))
    return existing, dest


def upsert_pricing_entry(
    entry_payload: dict[str, Any],
    *,
    replace_key: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Insert or update a single pricing entry and return the serialized table."""
    if not isinstance(entry_payload, dict):
        raise PricingSaveError("entry payload must be an object")

    provider = str(entry_payload.get("provider") or "").strip()
    match_dict = entry_payload.get("match")
    if not isinstance(match_dict, dict):
        match_dict = {}
    model_id = str(entry_payload.get("id") or match_dict.get("id") or "").strip()
    alias = str(entry_payload.get("alias") or match_dict.get("alias") or "").strip()
    match_prov = str(match_dict.get("provider") or "").strip()

    if not provider and not match_prov:
        if model_id:
            provider = "custom"
        else:
            raise PricingSaveError("provider is required")

    final_match: dict[str, str] = {}
    if model_id:
        final_match["id"] = model_id
    if alias:
        final_match["alias"] = alias
    if match_prov:
        final_match["provider"] = match_prov
    elif not model_id and provider:
        final_match["provider"] = provider

    if not final_match:
        raise PricingSaveError("at least one match identifier (model id or provider) is required")

    inp = _parse_required_rate(entry_payload.get("input_per_mtok"), "input_per_mtok")
    outp = _parse_required_rate(entry_payload.get("output_per_mtok"), "output_per_mtok")
    cached_read = _parse_optional_rate(entry_payload.get("cached_read_per_mtok"), "cached_read_per_mtok")
    cached_write = _parse_optional_rate(entry_payload.get("cached_write_per_mtok"), "cached_write_per_mtok")
    thinking = _parse_optional_rate(entry_payload.get("thinking_per_mtok"), "thinking_per_mtok")

    # The operator's OWN file, never the bundled asset's contents: writing a
    # copy of all 48 bundled rows into the operator layer would make them
    # operator-owned, so every later bundled fix would be shadowed and never
    # refreshed. The lower layers already supply whatever this file omits.
    existing, dest = _own_pricing_dict(path)
    out = dict(existing)
    out["version"] = int(existing.get("version") or 1)
    entries = list(existing.get("entries") or [])

    new_entry: dict[str, Any] = {
        "provider": provider,
        "match": final_match,
        "input_per_mtok": float(inp),
        "output_per_mtok": float(outp),
    }
    if cached_read is not None:
        new_entry["cached_read_per_mtok"] = float(cached_read)
    if cached_write is not None:
        new_entry["cached_write_per_mtok"] = float(cached_write)
    if thinking is not None:
        new_entry["thinking_per_mtok"] = float(thinking)

    target_key = (provider, final_match.get("id", ""), final_match.get("provider", ""))

    found_idx = -1
    for idx, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        ekey = _match_key(e)
        if ekey == target_key:
            found_idx = idx
            break
        if replace_key:
            rep = replace_key.strip()
            if rep == ekey[1] or rep == f"{ekey[0]}:{ekey[1]}" or rep == ekey[0]:
                found_idx = idx
                break

    if found_idx >= 0:
        old_e = dict(entries[found_idx])
        # Re-pricing a row re-enables it: the operator now supplies it, so a
        # tombstone a delete left behind must not keep suppressing it.
        old_e.pop("disabled", None)
        old_e.update(new_entry)
        entries[found_idx] = old_e
    else:
        entries.append(new_entry)

    out["entries"] = entries
    _write_pricing_dict(dest, out)
    return _pricing_snapshot(path, dest)


def update_pricing_rates(
    payload: dict[str, Any],
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Change exactly one effective row's rates, preserving its match and schedule.

    Editing an automatic/bundled row creates an operator override for that row
    only. Identifiers are selected by ``key`` and cannot be rewritten by rates.
    The default row is edited independently from every model entry.
    """
    if not isinstance(payload, dict):
        raise PricingSaveError("payload must be an object")
    key, rates = payload.get("key"), payload.get("rates")
    if not isinstance(key, dict) or not isinstance(rates, dict):
        raise PricingSaveError("one row key and rates object are required")
    allowed = {"input_per_mtok", "output_per_mtok", "cached_read_per_mtok", "cached_write_per_mtok"}
    if set(rates) - allowed:
        raise PricingSaveError("only input, output and cache rates can be edited")
    existing, dest = _own_pricing_dict(path)
    layers = _pricing_layers(dest)
    out = dict(existing)
    out["version"] = int(existing.get("version") or 1)
    if key.get("kind") == "default":
        out["default"] = _apply_rates(_merge_pricing_defaults(layers), rates)
    elif key.get("kind") == "entry":
        target = _layer_key(key)
        if target == ("", "", ""):
            raise PricingSaveError("a model or provider match is required")
        base = next((row for _layer, row in _merge_pricing_layers(layers) if _layer_key(row) == target), None)
        if base is None:
            raise PricingSaveError("pricing entry not found; refresh the table")
        edited = _apply_rates(base, rates)
        entries = list(existing.get("entries") or [])
        index = next((i for i, row in enumerate(entries) if isinstance(row, dict) and _layer_key(row) == target), None)
        if index is None:
            entries.append(edited)
        else:
            entries[index] = edited
        out["entries"] = entries
    else:
        raise PricingSaveError("row kind must be entry or default")
    _write_pricing_dict(dest, out)
    return _pricing_snapshot(path, dest)


def delete_pricing_entry(
    entry_key: str | dict[str, Any],
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Delete a single pricing entry by model id, provider, or key tuple.

    A row the OPERATOR's own file defines is removed from it. A row that only a
    lower layer supplies (the bundled asset or the auto overlay) is suppressed
    with a ``"disabled": true`` tombstone instead: the row leaves the effective
    table without the operator's file having to shadow the whole table, and
    re-pricing that row clears the tombstone again.
    """
    if entry_key == "default" or entry_key == "*":
        raise PricingSaveError("default pricing row cannot be deleted")

    existing, dest = _own_pricing_dict(path)
    entries = list(existing.get("entries") or [])

    target_provider = ""
    target_id = ""
    if isinstance(entry_key, dict):
        target_provider = str(entry_key.get("provider") or "").strip()
        m = value if isinstance(value := entry_key.get("match"), dict) else {}
        target_id = str(entry_key.get("id") or m.get("id") or m.get("provider") or "").strip()
    elif isinstance(entry_key, str):
        parts = entry_key.split(":", 1)
        if len(parts) == 2:
            target_provider, target_id = parts[0].strip(), parts[1].strip()
        else:
            target_id = entry_key.strip()

    removed = False
    new_entries = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        # A tombstone is not a priced row: deleting is never how an operator
        # un-suppresses one (re-pricing the row clears it instead), so a repeat
        # delete cannot silently restore the lower layer's rates.
        if not removed and not e.get("disabled") and _entry_matches(e, target_provider, target_id):
            removed = True
            continue
        new_entries.append(e)

    if not removed:
        # Not one of the operator's own rows: suppress it in a lower layer.
        tombstone = _tombstone_for(path, target_provider, target_id)
        if tombstone is None:
            raise PricingSaveError(f"pricing entry {entry_key!r} not found")
        new_entries.append(tombstone)

    out = dict(existing)
    out["version"] = int(existing.get("version") or 1)
    out["entries"] = new_entries
    _write_pricing_dict(dest, out)
    return _pricing_snapshot(path, dest)


def save_pricing_table(
    payload: dict[str, Any],
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Merge edited rates onto the user pricing file and reload the table."""
    # Own rows only -- see upsert_pricing_entry: a materialised copy of the
    # bundled table would pin every one of those rows in a higher layer.
    existing, dest = _own_pricing_dict(path)

    if not isinstance(payload, dict):
        raise PricingSaveError("payload must be an object")
    default_src = payload.get("default")
    if not isinstance(default_src, dict):
        raise PricingSaveError("default row is required")
    entries_src = payload.get("entries")
    if not isinstance(entries_src, list):
        raise PricingSaveError("entries must be a list")

    out = dict(existing)
    out["version"] = int(existing.get("version") or 1)
    base_default = existing.get("default")
    out["default"] = _apply_rates(
        dict(base_default) if isinstance(base_default, dict) else {},
        default_src,
    )
    by_key = {_match_key(entry): dict(entry) for entry in (existing.get("entries") or []) if isinstance(entry, dict)}
    effective = {_match_key(row): dict(row) for _layer, row in _merge_pricing_layers(_pricing_layers(dest))}
    updated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for src in entries_src:
        if not isinstance(src, dict):
            raise PricingSaveError("each entry must be an object")
        key = _match_key(src)
        if key in seen:
            raise PricingSaveError("duplicate pricing entry")
        seen.add(key)
        base = (
            by_key.get(key)
            or effective.get(key)
            or {
                "provider": str(src.get("provider") or ""),
                "match": {str(k): str(v) for k, v in (src.get("match") or {}).items()}
                if isinstance(src.get("match"), dict)
                else {},
            }
        )
        row = _apply_rates(base, src)
        # The browser sends every visible row. Keep unchanged lower-layer
        # rows there so automatic refreshes and bundled fixes still take effect.
        if key not in by_key and row == base:
            continue
        row.pop("disabled", None)  # an edit re-enables a tombstoned row
        updated.append(row)
    for key, base in by_key.items():
        if key in seen:
            continue
        match = value if isinstance(value := base.get("match"), dict) else {}
        if match.get("alias") and not match.get("id") and not match.get("provider"):
            continue
        updated.append(base)
    out["entries"] = updated
    _write_pricing_dict(dest, out)
    return _pricing_snapshot(path, dest)


# ---------------------------------------------------------------------------
# Pricer
# ---------------------------------------------------------------------------


@dataclass
class CostResult:
    total_pico: int = 0
    by_bucket_pico: dict[str, int] = field(default_factory=dict)
    estimated: bool = False
    model: str = ""
    model_alias: str = ""
    provider: str = ""
    # Appended to preserve the public positional order of the pre-existing
    # fields. The ledger carries this precision across turn rows before
    # ROUND_HALF_UP quantization so repeated sub-pico charges accumulate.
    exact_total_pico: Decimal | None = None
    # A provider/turn finalizer supplied an explicit authoritative cost (for
    # example a CLI's reported total). This permits a same-token checkpoint to
    # replace an interim token-priced estimate without weakening ordinary
    # duplicate idempotency.
    authoritative: bool = False

    @property
    def total_usd(self) -> float:
        return pico_to_usd(self.total_pico)

    def to_metrics(self) -> dict[str, Any]:
        return {
            "cost_pico": self.total_pico,
            "cost_usd": self.total_usd,
            "cost_estimated": self.estimated,
        }


def price(
    usage: CanonicalUsage,
    entry: PricingEntry,
    *,
    when: datetime | None = None,
) -> CostResult:
    """Price canonical usage with one entry. Decimal math → pico-USD.

    ``when`` is the turn's wall-clock time (UTC) for entries that declare a
    peak schedule; None prices at the off-peak base rate.
    """
    by_bucket: dict[str, int] = {}
    total = Decimal(0)
    exact_total_pico = Decimal(0)
    for bucket in BUCKETS:
        tokens = getattr(usage, bucket, 0)
        if not tokens:
            continue
        rate = entry.rate_for(bucket, when=when)
        if rate <= 0:
            by_bucket[bucket] = 0
            continue
        cost_usd = (Decimal(int(tokens)) * rate) / _MTOK
        exact_total_pico += cost_usd * PICO_PER_USD
        pico = usd_to_pico(cost_usd)
        by_bucket[bucket] = pico
        total += Decimal(pico)
    return CostResult(
        total_pico=int(total),
        by_bucket_pico=by_bucket,
        exact_total_pico=exact_total_pico,
        estimated=usage.estimated,
        model=usage.model,
        model_alias=usage.model_alias,
        provider=usage.provider,
    )


def price_metrics(
    metrics: dict[str, Any] | None,
    table: PricingTable,
    *,
    when: datetime | None = None,
) -> CostResult:
    """Convenience: normalize a metrics dict, resolve its entry, price it."""
    usage = normalize_usage(metrics)
    entry = table.resolve(usage)
    return price(usage, entry, when=when)
