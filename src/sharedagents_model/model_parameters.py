"""Validated, data-driven optional model parameters.

Snapshots are loaded once per runtime home. Call clear_model_parameters_cache at
an application's normal reload boundary; request assembly never probes upstream.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class ModelParametersError(ValueError):
    """An operator or shipped declaration cannot be used safely."""


_SAMPLING = frozenset(
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
_TEMPLATE = frozenset(
    {
        "chat_template_kwargs.enable_thinking",
        "chat_template_kwargs.thinking_budget",
    }
)
_EXTRA = frozenset(
    {
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_sequence_breakers",
        "repetition_detection",
    }
)
_WIRES = frozenset({"reasoning_effort", "chat_template_kwargs.thinking_budget"})
_RESERVED = frozenset({"default", "auto", "reset", "clear"})


def _error(where: str, key: str) -> None:
    raise ModelParametersError(f"{where}: invalid {key}")


def _keys(value: Any, allowed: set[str] | frozenset[str], where: str) -> None:
    if not isinstance(value, dict):
        _error(where, "object")
    unknown = set(value) - allowed
    if unknown:
        _error(where, "unknown key " + sorted(unknown)[0])


def validate_effort(value: Any, where: str = "effort", *, legacy: bool = False) -> dict[str, Any]:
    """Validate a declaration; legacy row lists use the standard wire by default."""
    if value is None:
        return {"values": []}
    _keys(value, {"values", "default", "wire", "wire_values", "source", "applies"}, where)
    values = value.get("values")
    if (
        not isinstance(values, list)
        or any(not isinstance(v, str) or not v or v != v.strip() or v.lower() in _RESERVED for v in values)
        or len(set(values)) != len(values)
    ):
        _error(where, "values")
    default = value.get("default")
    if default is not None and default not in values:
        _error(where, "default")
    wire = value.get("wire", "reasoning_effort" if legacy and values else None)
    if values and wire not in _WIRES:
        _error(where, "wire")
    mapping = value.get("wire_values")
    if not values:
        if default is not None or mapping is not None:
            _error(where, "disabled effort default/wire_values")
    elif mapping is not None:
        if not isinstance(mapping, dict) or set(mapping) != set(values):
            _error(where, "wire_values")
        for mapped in mapping.values():
            if wire == "chat_template_kwargs.thinking_budget":
                if type(mapped) is not int or mapped < 0:
                    _error(where, "wire_values budget")
            elif not isinstance(mapped, str) or not mapped.strip():
                _error(where, "wire_values token")
    elif values and wire == "chat_template_kwargs.thinking_budget":
        _error(where, "wire_values required for budget")
    return {
        k: deepcopy(v)
        for k, v in {
            "values": values,
            "default": default,
            "wire": wire,
            "wire_values": mapping,
        }.items()
        if v is not None
    }


def _parameters(value: Any, where: str) -> None:
    from .sampling import normalize

    _keys(value, _SAMPLING | _TEMPLATE | _EXTRA, where)
    for key, item in value.items():
        if item is None:
            continue
        valid = True
        if key in _SAMPLING:
            valid = key in normalize({key: item})
        elif key.endswith("enable_thinking"):
            valid = type(item) is bool
        elif key.endswith("thinking_budget") or key == "dry_allowed_length":
            valid = type(item) is int and item >= 0
        elif key == "dry_sequence_breakers":
            valid = isinstance(item, list) and all(isinstance(v, str) for v in item)
        elif key == "repetition_detection":
            valid = (
                isinstance(item, dict)
                and set(item)
                == {
                    "max_pattern_size",
                    "min_pattern_size",
                    "min_count",
                }
                and all(type(v) is int and v > 0 for v in item.values())
            )
            if valid:
                valid = item["min_pattern_size"] <= item["max_pattern_size"]
        else:
            import math

            valid = type(item) in (int, float) and math.isfinite(item) and item >= 0
        if not valid:
            _error(where, key)


def validate_document(raw: Any, where: str) -> list[dict[str, Any]]:
    _keys(raw, {"version", "models"}, where)
    if type(raw.get("version")) is not int or raw["version"] != 1 or not isinstance(raw.get("models"), list):
        _error(where, "version/models")
    selectors: set[tuple[str, str]] = set()
    out = []
    for index, original in enumerate(raw["models"]):
        loc = f"{where}: models[{index}]"
        _keys(original, {"provider", "model", "aliases", "effort", "parameters", "modes"}, loc)
        entry = deepcopy(original)
        for key in ("provider", "model"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                _error(loc, key)
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
            _error(loc, "aliases")
        for model in [entry["model"], *aliases]:
            selector = (entry["provider"].strip().lower(), model.strip().lower())
            if selector in selectors:
                _error(loc, "duplicate model/alias selector")
            selectors.add(selector)
        entry["effort"] = validate_effort(entry.get("effort"), loc + ".effort")
        _parameters(entry.get("parameters", {}), loc + ".parameters")
        modes = entry.get("modes", {})
        _keys(modes, {"plan", "ask", "auto"}, loc + ".modes")
        for mode, override in modes.items():
            _keys(override, {"parameters", "default"}, loc + ".modes." + mode)
            _parameters(override.get("parameters", {}), loc + ".modes." + mode)
            if override.get("default") is not None and override["default"] not in entry["effort"]["values"]:
                _error(loc, "mode default")
        if entry["provider"].strip().lower() == "bedrock" and entry["effort"]["values"]:
            _error(loc, "Bedrock effort is unsupported")
        wire = entry["effort"].get("wire")
        for params in [entry.get("parameters", {}), *(m.get("parameters", {}) for m in modes.values())]:
            if wire and wire in params:
                _error(loc, "effort/parameters wire collision")
        if (
            wire in _TEMPLATE
            or any(k in _TEMPLATE | _EXTRA for k in entry.get("parameters", {}))
            or any(k in _TEMPLATE | _EXTRA for m in modes.values() for k in m.get("parameters", {}))
        ):
            if entry["provider"].lower() not in ("vllm", "sglang"):
                _error(loc, "local-only parameters require vllm/sglang")
        out.append(entry)
    return out


def _read(resource: Any, where: str) -> list[dict[str, Any]]:
    try:
        raw = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelParametersError(f"{where}: cannot read valid JSON ({type(exc).__name__})") from exc
    return validate_document(raw, where)


@lru_cache(maxsize=16)
def _snapshot(user_path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shipped = _read(
        files("sharedagents_model").joinpath("data/model_parameters.defaults.json"), "model_parameters.defaults.json"
    )
    user = Path(user_path)
    try:
        user.stat()
    except FileNotFoundError:
        overrides = []
    except OSError as exc:
        raise ModelParametersError(f"{user}: cannot read ({type(exc).__name__})") from exc
    else:
        overrides = _read(user, str(user))
    return overrides, shipped


def clear_model_parameters_cache() -> None:
    _snapshot.cache_clear()


def provider_identity(row: Mapping[str, Any]) -> str:
    """Use explicit provider/backend identity or an exact known provider host."""
    explicit = str(row.get("provider") or "").strip().lower()
    if explicit:
        return explicit
    backend = str(row.get("backend") or "").strip().lower()
    if backend in ("vllm", "sglang", "bedrock"):
        return backend
    host = urlparse(str(row.get("baseUrl") or row.get("base_url") or row.get("endpoint") or "")).hostname
    # Provider identity is transport metadata, never model policy.
    from .provider_presets import PROVIDER_PRESETS

    for preset in PROVIDER_PRESETS:
        if host and urlparse(preset.base_url).hostname == host:
            return preset.key
    return backend or str(row.get("owned_by") or "").strip().lower()


def resolve_model_parameters(row: Mapping[str, Any], model_id: str = "", mode: str = "auto") -> dict[str, Any] | None:
    from sharedagents_core.kernel.layout import sharedagents_root

    layers = _snapshot(str(sharedagents_root() / "model_parameters.json"))
    provider = provider_identity(row)
    # A supplied served id is authoritative. Configured row aliases are only
    # considered when they are explicitly named by a declaration.
    target = (
        str(
            model_id
            or row.get("wire_id")
            or row.get("model_id")
            or row.get("id")
            or row.get("model")
            or row.get("alias")
            or ""
        )
        .strip()
        .lower()
    )
    for entries, source in zip(layers, ("configured", "shipped")):
        for entry in entries:
            if entry["provider"].strip().lower() != provider:
                continue
            if target not in {v.strip().lower() for v in [entry["model"], *entry.get("aliases", [])]}:
                continue
            result = deepcopy(entry)
            result["source"] = source
            override = result.get("modes", {}).get(mode, {})
            result["parameters"] = {**result.get("parameters", {}), **override.get("parameters", {})}
            if "default" in override:
                result["effort"]["default"] = override["default"]
            return result
    return None


def effort_capability(declaration: Mapping[str, Any]) -> dict[str, Any]:
    return {**deepcopy(declaration["effort"]), "source": declaration["source"]}
