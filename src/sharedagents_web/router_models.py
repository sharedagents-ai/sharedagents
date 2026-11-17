"""CRUD for the operator vLLM router model list (Usage Router tab).

Reads and writes ``$SHAREDAGENTS_VLLM_CONFIG`` or ``$SHAREDAGENTS_HOME/vllm-router.yaml``.
Never writes the packaged empty example. Secrets (``api_key``) are rejected.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sharedagents_vllm.config import (
    ConfigError,
    default_config_path,
    packaged_example_path,
    parse_config,
)
from sharedagents_web.router_status import model_category

POOL_KINDS = ("llm", "embedding", "reranker", "asr", "tts")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

EMPTY_TEMPLATE: dict[str, Any] = {
    "bind": "127.0.0.1",
    "port": 8000,
    "health_check_seconds": 10,
    "connect_timeout_seconds": 5,
    "read_timeout_seconds": 120,
    "first_token_timeout_seconds": 300,
    "failure_cooldown_seconds": 120,
    "session_ttl_seconds": 900,
    "backpressure": {"reject_status": 503, "retry_after_seconds": 5},
    "external_providers": {},
    "pools": {},
}


class RouterModelsError(ValueError):
    """Invalid model payload or refused write."""


def operator_yaml_path() -> Path:
    return Path(default_config_path()).expanduser()


def _yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RouterModelsError("PyYAML is required to edit router models") from exc
    return yaml


def _empty_raw() -> dict[str, Any]:
    return {
        "bind": EMPTY_TEMPLATE["bind"],
        "port": EMPTY_TEMPLATE["port"],
        "health_check_seconds": EMPTY_TEMPLATE["health_check_seconds"],
        "connect_timeout_seconds": EMPTY_TEMPLATE["connect_timeout_seconds"],
        "read_timeout_seconds": EMPTY_TEMPLATE["read_timeout_seconds"],
        "first_token_timeout_seconds": EMPTY_TEMPLATE["first_token_timeout_seconds"],
        "failure_cooldown_seconds": EMPTY_TEMPLATE["failure_cooldown_seconds"],
        "session_ttl_seconds": EMPTY_TEMPLATE["session_ttl_seconds"],
        "backpressure": dict(EMPTY_TEMPLATE["backpressure"]),
        "external_providers": {},
        "pools": {},
    }


def load_operator_raw(path: Path | None = None) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser() if path is not None else operator_yaml_path()
    if not resolved.is_file():
        return _empty_raw(), resolved
    yaml = _yaml()
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RouterModelsError(f"cannot read router YAML: {exc}") from exc
    if raw is None:
        return _empty_raw(), resolved
    if not isinstance(raw, dict):
        raise RouterModelsError("router YAML root must be a mapping")
    if not isinstance(raw.get("pools"), dict):
        raw["pools"] = {}
    if not isinstance(raw.get("external_providers"), dict):
        raw.setdefault("external_providers", {})
    return raw, resolved


def serialize_pool(pool_id: str, pdata: dict[str, Any]) -> dict[str, Any]:
    replicas = []
    for row in pdata.get("replicas") or []:
        if not isinstance(row, dict):
            continue
        replicas.append(
            {
                "name": str(row.get("name") or "").strip(),
                "url": str(row.get("url") or "").rstrip("/"),
                "max_active": int(row.get("max_active") or 0),
                "upstream_model": str(row.get("upstream_model") or "").strip() or None,
            }
        )
    fallbacks = []
    for hop in pdata.get("fallbacks") or []:
        if not isinstance(hop, dict):
            continue
        pool = str(hop.get("pool") or "").strip()
        external = str(hop.get("external") or "").strip()
        if pool:
            fallbacks.append({"kind": "pool", "id": pool})
        elif external:
            fallbacks.append({"kind": "external", "id": external})
    try:
        cap = int(pdata.get("max_active_requests") or 0) or None
    except (TypeError, ValueError):
        cap = None
    return {
        "id": str(pool_id),
        "kind": str(pdata.get("kind") or "llm").strip() or "llm",
        "category": model_category(str(pdata.get("kind") or ""), str(pool_id)),
        "max_active_requests": cap,
        "upstream_model": str(pdata.get("upstream_model") or "").strip() or None,
        "replicas": replicas,
        "fallbacks": fallbacks,
    }


def list_router_models(path: Path | None = None) -> dict[str, Any]:
    raw, resolved = load_operator_raw(path)
    models = [
        serialize_pool(str(pid), pdata) for pid, pdata in (raw.get("pools") or {}).items() if isinstance(pdata, dict)
    ]
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "models": models,
    }


def _assert_writable(path: Path) -> None:
    packaged = packaged_example_path().resolve()
    try:
        if path.expanduser().resolve() == packaged:
            raise RouterModelsError("refusing to write the packaged empty template")
    except OSError as exc:
        raise RouterModelsError(f"invalid router YAML path: {exc}") from exc


def _dump_and_validate(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        parse_config(raw)
    except ConfigError as exc:
        raise RouterModelsError(str(exc)) from exc
    return raw


def write_operator_raw(raw: dict[str, Any], path: Path) -> None:
    _assert_writable(path)
    _dump_and_validate(raw)
    yaml = _yaml()
    text = yaml.safe_dump(
        raw,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _validate_model_id(model_id: str) -> str:
    name = str(model_id or "").strip()
    if not MODEL_ID_RE.match(name):
        raise RouterModelsError(
            "model id must start with a letter or digit and use only letters, digits, dot, underscore, colon, or hyphen"
        )
    return name


def _validate_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RouterModelsError("replica url must be http(s)://host[:port][/path]")
    return cleaned


def pool_from_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise RouterModelsError("model payload must be an object")
    if "api_key" in payload:
        raise RouterModelsError("api_key is forbidden; use api_key_env on providers")
    model_id = _validate_model_id(str(payload.get("id") or ""))
    kind = str(payload.get("kind") or "llm").strip().lower()
    if kind not in POOL_KINDS:
        raise RouterModelsError(f"kind must be one of: {', '.join(POOL_KINDS)}")
    try:
        cap = int(payload.get("max_active_requests") or 0)
    except (TypeError, ValueError) as exc:
        raise RouterModelsError("max_active_requests must be an integer") from exc
    if cap < 0:
        raise RouterModelsError("max_active_requests cannot be negative")
    replicas = []
    for row in payload.get("replicas") or []:
        if not isinstance(row, dict):
            raise RouterModelsError("each replica must be an object")
        name = str(row.get("name") or "").strip()
        if not name:
            raise RouterModelsError("replica name is required")
        url = _validate_url(str(row.get("url") or ""))
        try:
            max_active = int(row.get("max_active") or 0)
        except (TypeError, ValueError) as exc:
            raise RouterModelsError("replica max_active must be an integer") from exc
        if max_active < 0:
            raise RouterModelsError("replica max_active cannot be negative")
        replicas.append({"name": name, "url": url, "max_active": max_active})
    if not replicas:
        raise RouterModelsError("add at least one replica (name + url)")
    fallbacks = []
    for hop in payload.get("fallbacks") or []:
        if not isinstance(hop, dict):
            raise RouterModelsError("each fallback must be an object")
        kind_hop = str(hop.get("kind") or "").strip().lower()
        hid = str(hop.get("id") or hop.get("pool") or hop.get("external") or "").strip()
        if kind_hop == "pool" or hop.get("pool"):
            if not hid:
                raise RouterModelsError("fallback pool id is required")
            fallbacks.append({"pool": hid})
        elif kind_hop == "external" or hop.get("external"):
            if not hid:
                raise RouterModelsError("fallback external id is required")
            fallbacks.append({"external": hid})
        else:
            raise RouterModelsError("fallback kind must be pool or external")
    pdata: dict[str, Any] = {"kind": kind, "replicas": replicas}
    if cap:
        pdata["max_active_requests"] = cap
    if fallbacks:
        pdata["fallbacks"] = fallbacks
    return model_id, pdata


def upsert_router_model(
    payload: dict[str, Any],
    *,
    replace_id: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    model_id, pdata = pool_from_payload(payload)
    raw, resolved = load_operator_raw(path)
    pools = raw.setdefault("pools", {})
    if not isinstance(pools, dict):
        raise RouterModelsError("pools must be a mapping")
    previous = str(replace_id or "").strip()
    if previous and previous != model_id:
        if previous not in pools:
            raise RouterModelsError(f"model {previous!r} is not configured")
        if model_id in pools:
            raise RouterModelsError(f"model {model_id!r} already exists")
    elif model_id in pools and not previous:
        raise RouterModelsError(f"model {model_id!r} already exists")
    elif previous and previous not in pools and previous != model_id:
        raise RouterModelsError(f"model {previous!r} is not configured")
    existing = {}
    source_id = previous if previous and previous in pools else (model_id if model_id in pools else "")
    if source_id and isinstance(pools.get(source_id), dict):
        existing = dict(pools[source_id])
    merged = dict(existing)
    merged["kind"] = pdata["kind"]
    # The form edits name/url/max_active only; keep YAML-only replica keys
    # (``upstream_model``) for a replica that is still there by name.
    kept = {str(r.get("name") or "").strip(): r for r in existing.get("replicas") or [] if isinstance(r, dict)}
    merged["replicas"] = [
        {**{k: v for k, v in kept.get(r["name"], {}).items() if k not in ("name", "url", "max_active")}, **r}
        for r in pdata["replicas"]
    ]
    if "max_active_requests" in pdata:
        merged["max_active_requests"] = pdata["max_active_requests"]
    elif "max_active_requests" in merged:
        del merged["max_active_requests"]
    if "fallbacks" in pdata:
        merged["fallbacks"] = pdata["fallbacks"]
    else:
        merged.pop("fallbacks", None)
    if previous and previous != model_id:
        del pools[previous]
    pools[model_id] = merged
    pdata = merged
    write_operator_raw(raw, resolved)
    listed = list_router_models(resolved)
    listed["saved"] = serialize_pool(model_id, pdata)
    return listed


def delete_router_model(model_id: str, path: Path | None = None) -> dict[str, Any]:
    raw, resolved = load_operator_raw(path)
    pools = raw.get("pools") or {}
    # An existing id is deletable as written, even one the add form would now
    # refuse (``nvidia/Qwen3.8-Flash-Next-FP4-Local`` has a slash).
    name = str(model_id or "")
    if name not in pools:
        name = _validate_model_id(model_id)
    if name not in pools:
        raise RouterModelsError(f"model {name!r} is not configured")
    del pools[name]
    raw["pools"] = pools
    write_operator_raw(raw, resolved)
    listed = list_router_models(resolved)
    listed["deleted"] = name
    return listed
