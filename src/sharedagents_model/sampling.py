"""Explicit sampling configuration shared by turns and bounded side calls."""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

SAMPLING_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    }
)


def normalize(d: Any) -> dict[str, Any]:
    """Coerce valid numeric values; retain null tombstones for layer merging.

    Unknown keys, booleans, non-finite numbers and invalid ranges are ignored.
    Integer samplers reject fractional values rather than silently truncating.
    """
    if not isinstance(d, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in d.items():
        if key not in SAMPLING_KEYS or isinstance(value, bool):
            continue
        if value is None:
            out[key] = None
            continue
        try:
            number = float(value)
            if not math.isfinite(number):
                continue
            if key in ("top_k", "seed"):
                integer = Decimal(str(value))
                if integer != integer.to_integral_value() or (key == "top_k" and integer < 0):
                    continue
                value = int(integer)
            else:
                value = number
                if key in ("top_p", "min_p") and not 0 <= value <= 1:
                    continue
                if key in ("temperature", "repetition_penalty") and value < 0:
                    continue
                if key in ("presence_penalty", "frequency_penalty") and not -2 <= value <= 2:
                    continue
        except (ValueError, TypeError, OverflowError, InvalidOperation):
            continue
        out[key] = value
    return out


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def sampling_row(cfg: Any, endpoint: str, model_id: str) -> Mapping[str, Any]:
    """Find the exact deployment for side callers that only have its route."""
    resolver = getattr(cfg, "model_row_for_endpoint", None)
    if callable(resolver):
        row = resolver(endpoint, model_id)
        if row:
            return row
    for row in _get(cfg, "models", []) or []:
        if (
            isinstance(row, Mapping)
            and row.get("id") == model_id
            and str(row.get("baseUrl") or "").rstrip("/") == endpoint.rstrip("/")
        ):
            return row
    return {}


def configured_sampling(
    cfg: Any,
    row: Any = None,
    model_id: str = "",
    *,
    purpose: str | None = None,
    override: Any = None,
    mode: str = "auto",
) -> dict[str, Any]:
    """Resolve file defaults < legacy < main/purpose < deployment < turn.

    Side calls inherit the legacy temperature alias, but never global sampling.
    Explicit nulls remove inherited values. Missing values stay omitted.
    """
    from .model_parameters import resolve_model_parameters

    metadata = row if isinstance(row, Mapping) else {}
    declaration = resolve_model_parameters(metadata, "" if metadata.get("id") else model_id, mode)
    out = normalize((declaration or {}).get("parameters", {}))
    temperature = _get(cfg, "temperature")
    if (
        temperature is not None
        or isinstance(cfg, Mapping)
        and "temperature" in cfg
        or getattr(cfg, "_temperature_is_explicit", False)
    ):
        out.update(normalize({"temperature": temperature}))
    if purpose is None:
        layer = _get(cfg, "sampling")
    else:
        side = _get(cfg, "side_sampling")
        layer = side.get(purpose) if isinstance(side, Mapping) else None
    out.update(normalize(layer))
    out.update(normalize(_get(row, "sampling")))
    out.update(normalize(override))
    return {k: v for k, v in out.items() if v is not None}


def apply_sampling(payload: dict[str, Any], sampling: Any) -> None:
    """Replace sampling fields without disturbing tokens, tools or templates."""
    for key in SAMPLING_KEYS:
        payload.pop(key, None)
    payload.update({k: v for k, v in normalize(sampling).items() if v is not None})
