"""Agent & Model picker data: a cheap summary, then one provider at a time.

Opening the picker must never wait on a model provider. :func:`agent_summary`
names an agent's default model row and its provider tabs from configuration
and saved catalogues only — no network call of any kind. Each tab carries a
status decided here: ``ready``, or an error (``no_key`` when the provider's
API key is not set in this process's environment).

A provider's models are fetched by :func:`provider_models` when its tab is
opened, and only that provider is asked. Results are cached per provider
(``PROVIDER_TTL_S``) in the shared discovery cache, merged into the native
catalogue so a refresh of one provider never drops another provider's rows.
A failed refresh keeps the saved rows and reports ``unreachable``.

Frontends render what these functions return; they decide nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from sharedagents_agent.discovery import cache_lock as _cl

_LOG = logging.getLogger("sharedagents.discovery.picker")

#: A provider opened again within this window is served from the saved rows.
PROVIDER_TTL_S = 600.0
#: Cache entry holding ``{agent/provider key: epoch seconds}`` of the last refresh.
REFRESHED_AT_KEY = "__native_refreshed_at__"
PROVIDER_ROWS_KEY = "__provider_rows__"


def _cache_key(agent: str, provider: str) -> str:
    return json.dumps([agent, tab_key(provider)], separators=(",", ":"))


def public_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Presentation fields only; connection settings and credentials stay here."""
    fields = {
        "id",
        "alias",
        "wire_id",
        "model_id",
        "display",
        "title",
        "label",
        "subtitle",
        "description",
        "tab",
        "provider",
        "backend",
        "default",
        "active",
        "contextWindow",
        "maxTokens",
        "max_tokens",
        "effort",
        "reasoning",
        "speed",
        "selection_capabilities",
        "capabilities",
        "badges",
        "effort_pref",
        "effort_selected",
    }
    return [{k: v for k, v in row.items() if k in fields} for row in rows]


_LOCAL_ONLY_PROVIDERS = frozenset({"vllm", "openai-compat", "configured", "unknown"})
_GENERIC_BACKENDS = frozenset({"openai", "unknown", "configured"})

#: The tab an external engine's catalogue sits under, by registry short name.
#: OpenCode and RPC engines (Pi) list real upstream providers per row instead.
_ENGINE_TABS = {
    "antigravity": "antigravity",
    "claude": "anthropic",
    "codex": "codex",
    "cursor": "cursor",
    "opencode": "opencode",
    "kimi": "kimi",
    "hermes": "hermes",
    "devin": "devin",
}


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def tab_key(name: Any) -> str:
    """Case- and space-insensitive identity of a provider tab."""
    return re.sub(r"\s+", "", str(name or "").strip().lower())


# ── Which tab a row belongs to ───────────────────────────────────────────


def _native_tab(row: dict[str, Any]) -> str:
    explicit = str(row.get("provider") or "").strip()
    if explicit and explicit.lower() not in _LOCAL_ONLY_PROVIDERS:
        return explicit
    backend = str(row.get("backend") or "").strip()
    if backend and backend.lower() not in _GENERIC_BACKENDS:
        return backend
    return backend or "vllm"


def _heuristic_tab(row: dict[str, Any], fallback: str) -> str:
    explicit = str(row.get("provider") or "").strip()
    if explicit and explicit.lower() not in _LOCAL_ONLY_PROVIDERS:
        return explicit
    mid = str(row.get("id") or row.get("alias") or "").strip().lower()
    url = str(row.get("baseUrl") or row.get("endpoint") or "").lower()
    if "cerebras.ai" in url or mid.startswith("cerebras"):
        return "Cerebras"
    if mid.startswith("gemini"):
        return "antigravity"
    if mid.startswith("claude"):
        return "anthropic"
    if mid.startswith("codex:") or re.match(r"^(gpt-5\.[4-6]|gpt-6-|codex-)", mid):
        return "codex"
    if re.match(r"^(gpt-|o1|o3|o4)", mid):
        return "openai"
    return str(row.get("backend") or "").strip() or fallback or "vllm"


def _engine_tab_for(spec: Any, agent: str) -> str:
    short = str(getattr(spec, "short_name", "") or agent).strip().lower()
    return _ENGINE_TABS.get(short) or str(getattr(spec, "provider", "") or "").strip() or agent


def _row_tab(row: dict[str, Any], agent: str, spec: Any) -> str:
    if spec is None:
        return _native_tab(row)
    short = str(getattr(spec, "short_name", "") or agent).strip().lower()
    if getattr(spec, "adapter", "") == "rpc" or short not in _ENGINE_TABS:
        return _heuristic_tab(row, _engine_tab_for(spec, agent))
    if short == "opencode":
        explicit = str(row.get("provider") or "").strip()
        if explicit and explicit.lower() not in _LOCAL_ONLY_PROVIDERS:
            return explicit
    return _ENGINE_TABS[short]


def stamp_tabs(agent: str, rows: list[dict[str, Any]], spec: Any = None) -> list[dict[str, Any]]:
    """Copy ``rows`` with ``tab`` set to the provider tab each one sits under.

    Spellings that differ only in case (``OpenRouter`` / ``openrouter``) share
    one tab, labelled by the first row that names it.
    """
    labels: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        name = _row_tab(row, agent, spec)
        row["tab"] = labels.setdefault(tab_key(name), name)
        out.append(row)
    return out


def engine_spec(agent: str) -> Any:
    """The external engine spec of ``agent``, or ``None`` for a native agent."""
    try:
        from sharedagents_agent.external_registry import spec_for_short_name

        return spec_for_short_name(agent)
    except ValueError:
        return None
    except Exception:  # noqa: BLE001 — unknown engine stays native, like discovery
        return None


def _load_cfg(agent: str) -> Any:
    """``AgentConfig.load`` without the ambient (networked) catalogue folds."""
    from sharedagents_agent.agent import AgentConfig

    return AgentConfig.load(agent, ambient_discovery=False)


# ── No-network credential checks ─────────────────────────────────────────


def _key_value(cfg: Any, env_name: str) -> str:
    """Same precedence as ``AgentConfig.upstream_headers``: per-agent
    secrets (an explicit blank blocks the env fallback), then the env."""
    keys = getattr(cfg, "api_keys", None) or {}
    if env_name in keys:
        return str(keys[env_name] or "")
    return os.environ.get(env_name, "")


def _codex_login_present() -> bool:
    try:
        from sharedagents_agent.discovery.openai_codex import (
            _read_codex_oauth_token,
            _read_codex_refresh_token,
        )

        access, _account = _read_codex_oauth_token()
        return bool(access or _read_codex_refresh_token())
    except Exception:  # noqa: BLE001
        return False


def _bedrock_credentials_present() -> bool:
    try:
        from sharedagents_agent.discovery.bedrock import bedrock_credentials_present

        return bedrock_credentials_present()
    except Exception:  # noqa: BLE001
        return False


def _is_codex_row(row: dict[str, Any]) -> bool:
    try:
        from sharedagents_agent.discovery.openai_codex import CODEX_OAUTH_AUTH
    except Exception:  # noqa: BLE001
        CODEX_OAUTH_AUTH = "codex-oauth"
    return row.get("auth") == CODEX_OAUTH_AUTH


def _is_bedrock_row(row: dict[str, Any]) -> bool:
    try:
        from sharedagents_model.bedrock_config import is_bedrock_row

        return bool(is_bedrock_row(row))
    except Exception:  # noqa: BLE001
        return str(row.get("backend") or "").lower() == "bedrock"


def _provider_kind(rows: list[dict[str, Any]]) -> str:
    """``codex`` / ``bedrock`` catalogues come from their own SDK calls."""
    if rows and all(_is_codex_row(r) for r in rows):
        return "codex"
    if rows and all(_is_bedrock_row(r) for r in rows):
        return "bedrock"
    return "http"


def _native_tab_status(cfg: Any, label: str, rows: list[dict[str, Any]]) -> dict[str, str] | None:
    """``None`` when the provider can be asked, else a ``no_key`` error.

    Decided from configuration and the environment only.
    """
    kind = _provider_kind(rows)
    if kind == "codex":
        if _codex_login_present():
            return None
        return _error("no_key", f"{label}: no ChatGPT login was found. Run `codex login`.")
    if kind == "bedrock":
        env_rows = [r for r in rows if str(r.get("apiKeyEnv") or r.get("api_key_env") or "").strip()]
        if any(_key_value(cfg, str(r.get("apiKeyEnv") or r.get("api_key_env")).strip()) for r in env_rows):
            return None
        if len(env_rows) < len(rows) and _bedrock_credentials_present():
            return None
        if env_rows:
            env_name = str(env_rows[0].get("apiKeyEnv") or env_rows[0].get("api_key_env"))
            return _error("no_key", f"{label}: API key {env_name} is not set")
        return _error("no_key", f"{label}: AWS credentials are not configured")
    missing = ""
    for row in rows:
        env_name = str(row.get("apiKeyEnv") or row.get("api_key_env") or "").strip()
        headers = row.get("headers") or {}
        if any(str(k).lower() == "authorization" and v for k, v in headers.items()):
            return None
        if not env_name:
            return None  # an endpoint that needs no key (a local server)
        if _key_value(cfg, env_name):
            return None
        missing = missing or env_name
    if missing:
        return _error("no_key", f"{label}: API key {missing} is not set")
    return None


# ── Rows known without asking a provider ─────────────────────────────────


def _native_known_rows(cfg: Any, cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Configured rows plus saved discovery rows — what the picker can show
    before any provider is asked."""
    rows = list(getattr(cfg, "models", None) or [])
    cached_rows = cache.get(_cl.NATIVE_CACHE_KEY)
    scoped = cache.get(PROVIDER_ROWS_KEY) or {}
    if isinstance(scoped, dict):
        for key, saved in scoped.items():
            try:
                owner, provider = json.loads(key)
            except (ValueError, TypeError):
                continue
            if owner == cfg.name and isinstance(saved, list):
                cached_rows = [r for r in (cached_rows or []) if tab_key(_native_tab(r)) != provider] + saved
    known = _cl.native_models_with_cache(
        str(getattr(cfg, "name", "") or ""),
        rows,
        codex_available=_codex_login_present(),
        cached_rows=cached_rows if isinstance(cached_rows, list) else [],
    )
    if str(getattr(cfg, "engine", "") or "local_llm").strip() in ("", "local_llm"):
        try:
            from sharedagents_agent.discovery.bedrock import cached_bedrock_models

            seen = {(r.get("baseUrl"), r.get("id")) for r in known}
            for row in cached_bedrock_models():
                if (row.get("baseUrl"), row.get("id")) not in seen:
                    known.append(dict(row))
                    seen.add((row.get("baseUrl"), row.get("id")))
        except Exception:  # noqa: BLE001
            pass
    return _cl._native_picker_models(known)


def _native_tabs(cfg: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        label = str(row.get("tab") or "")
        key = tab_key(label)
        labels.setdefault(key, label)
        by_key.setdefault(key, []).append(row)
    engine_native = str(getattr(cfg, "engine", "") or "local_llm").strip() in ("", "local_llm")
    # A provider whose rows were never saved still gets its tab: the models
    # load when it is opened.
    if engine_native and "codex" not in by_key and _codex_login_present():
        labels["codex"] = "Codex"
        by_key["codex"] = [{"auth": "codex-oauth"}]
    if engine_native and "bedrock" not in by_key and _bedrock_credentials_present():
        labels["bedrock"] = "bedrock"
        by_key["bedrock"] = [{"backend": "bedrock", "baseUrl": "bedrock://"}]
    tabs = []
    configured: dict[str, list[dict[str, Any]]] = {}
    for row in stamp_tabs(cfg.name, _cl._native_picker_models(cfg.models or [])):
        configured.setdefault(tab_key(row["tab"]), []).append(row)
    for key, label in labels.items():
        error = _native_tab_status(cfg, label, configured.get(key) or by_key[key])
        tabs.append(_tab_entry(label, error))
    return tabs


def _tab_entry(label: str, error: dict[str, str] | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": label, "label": label, "status": "error" if error else "ready"}
    if error:
        entry["error"] = error
    return entry


def _default_target(cfg: Any) -> str:
    for attr in ("upstream_model", "upstream_alias"):
        value = getattr(cfg, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _mark_default(rows: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    if not target or not any(_cl._row_names_model(r, target) for r in rows):
        return rows
    out = []
    for row in rows:
        row = dict(row)
        if _cl._row_names_model(row, target):
            row["default"] = True
        else:
            row.pop("default", None)
        out.append(row)
    return out


def _native_summary(agent: str, cfg: Any, cache: dict[str, Any]) -> dict[str, Any]:
    rows = stamp_tabs(agent, _native_known_rows(cfg, cache))
    target = _default_target(cfg)
    default = next((r for r in rows if target and _cl._row_names_model(r, target)), None)
    if default is None and not target and rows:
        default = rows[0]
    if default is None and target:
        # The saved model is not in any catalogue this process holds. Name it
        # anyway: the gateway validates it when a turn starts.
        default = {"id": target, "alias": target, "display": target, "tab": ""}
    if default is not None:
        default = dict(default, default=True)
    try:
        effective = (
            str(getattr(cfg, "upstream_model", "") or "").strip()
            or str((cfg._primary_model_row() or {}).get("id") or "").strip()
        )
    except Exception:  # noqa: BLE001
        effective = target
    return {
        "name": agent,
        "external": False,
        "default": default,
        "defaults": {"model": effective, "effort": getattr(cfg, "effort", None)},
        "providers": _native_tabs(cfg, rows),
        "_rows": rows,
    }


def _external_summary(agent: str, spec: Any, cfg: Any, cache: dict[str, Any]) -> dict[str, Any]:
    expected = (
        ""
        if getattr(spec, "adapter", "") == "rpc"
        else str(getattr(spec, "provider", "") or getattr(spec, "short_name", "") or agent).strip()
    )
    cached = cache.get(agent)
    rows = _cl._ensure_model_provider(cached if isinstance(cached, list) else [], expected)
    rows = _cl._stamp_configured_default(agent, rows, agent_cfg=cfg) if cfg is not None else rows
    rows = stamp_tabs(agent, rows, spec)
    pinned = ""
    if cfg is not None and getattr(cfg, "_upstream_model_is_per_agent", False):
        pinned = str(getattr(cfg, "upstream_model", "") or "").strip()
    default = None
    if pinned:
        default = next((r for r in rows if _cl._row_names_model(r, pinned)), None) or {
            "id": pinned,
            "alias": pinned,
            "display": pinned,
            "tab": _engine_tab_for(spec, agent),
        }
    elif rows:
        default = next((r for r in rows if r.get("default")), None) or rows[0]
    if default is not None:
        default = dict(default, default=True)
    labels: dict[str, str] = {}
    for row in rows:
        labels.setdefault(tab_key(row["tab"]), row["tab"])
    if not labels:
        tab = _engine_tab_for(spec, agent)
        labels[tab_key(tab)] = tab
    return {
        "name": agent,
        "external": True,
        "default": default,
        "defaults": {
            "model": str((default or {}).get("id") or pinned or ""),
            "effort": getattr(cfg, "effort", None) if cfg is not None else None,
        },
        "providers": [_tab_entry(label, None) for label in labels.values()],
        "_rows": rows,
    }


def _summary_for(agent: str, cache: dict[str, Any]) -> dict[str, Any] | None:
    spec = engine_spec(agent)
    try:
        cfg = _load_cfg(agent)
    except FileNotFoundError:
        cfg = None
    except Exception as exc:  # noqa: BLE001
        _LOG.info("picker summary: %s failed to load (%s)", agent, type(exc).__name__)
        cfg = None
    if spec is None:
        if cfg is None:
            return None
        return _native_summary(agent, cfg, cache)
    if cfg is None and not isinstance(cache.get(agent), list):
        return None
    return _external_summary(agent, spec, cfg, cache)


RowFilter = Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]


def _apply_row_filter(summary: dict[str, Any], row_filter: RowFilter | None) -> dict[str, Any]:
    """Restrict a summary to the rows an account may pick.

    Only tabs that still hold a granted row are offered; a default the
    account may not use gives way to its first granted row.
    """
    if row_filter is None:
        return summary
    agent = summary["name"]
    allowed = row_filter(agent, summary["_rows"])
    keys = {tab_key(r.get("tab")) for r in allowed}
    summary["_rows"] = allowed
    summary["providers"] = [t for t in summary["providers"] if tab_key(t["id"]) in keys]
    default = summary.get("default")
    if default is not None and not row_filter(agent, [default]):
        default = next((r for r in allowed if r.get("default")), None) or (allowed[0] if allowed else None)
        summary["default"] = dict(default, default=True) if default else None
    summary["defaults"] = dict(summary["defaults"], model=str((summary.get("default") or {}).get("id") or ""))
    return summary


def _public(summary: dict[str, Any]) -> dict[str, Any]:
    result = {k: v for k, v in summary.items() if not k.startswith("_")}
    if result.get("default"):
        result["default"] = public_model_rows([result["default"]])[0]
    return result


def saved_models(agent: str) -> list[dict[str, Any]]:
    """Configured and saved rows only, including on an external cache miss."""
    summary = _summary_for(agent, _cl.read_cache_file())
    return public_model_rows(summary["_rows"]) if summary else []


def agent_summary(
    agent: str,
    *,
    cache: dict[str, Any] | None = None,
    row_filter: RowFilter | None = None,
) -> dict[str, Any] | None:
    """Default model row + provider tabs of one agent. Never asks a provider.

    ``None`` for an agent that is not configured. ``row_filter(agent, rows)``
    narrows the rows (and so the tabs) to what an account may pick.
    """
    data = cache if cache is not None else _cl.read_cache_file()
    summary = _summary_for(agent, data)
    return _public(_apply_row_filter(summary, row_filter)) if summary else None


def picker_summary(
    agents: list[str],
    *,
    row_filter: RowFilter | None = None,
) -> list[dict[str, Any]]:
    """:func:`agent_summary` for each agent, reading the saved catalogues once."""
    data = _cl.read_cache_file()
    out = []
    for agent in agents:
        summary = agent_summary(agent, cache=data, row_filter=row_filter)
        if summary is not None:
            out.append(summary)
    return out


# ── One provider, asked when its tab is opened ───────────────────────────


def _refreshed_at(cache: dict[str, Any], key: str) -> float:
    stamps = cache.get(REFRESHED_AT_KEY)
    try:
        return float((stamps or {}).get(key) or 0.0) if isinstance(stamps, dict) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _is_stale(cache: dict[str, Any], key: str) -> bool:
    return time.time() - _refreshed_at(cache, key) >= PROVIDER_TTL_S


def _fetch_error(label: str, exc: Exception) -> dict[str, str]:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return _error("unreachable", f"{label}: the provider answered HTTP {status}")
    return _error("unreachable", f"{label}: the provider could not be reached ({type(exc).__name__})")


def _fetch_http_tab(
    cfg: Any,
    label: str,
    configured: list[dict[str, Any]],
    tab_rows: list[dict[str, Any]],
    reserved_aliases: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    """Ask each endpoint of one provider for its models (nothing else).

    Returns the provider's merged rows and an error when every endpoint
    failed.
    """
    from sharedagents_model import providers

    endpoints: dict[str, dict[str, Any]] = {}
    for row in tab_rows:
        endpoint = _cl._native_endpoint(row)
        if (
            endpoint
            and endpoint not in endpoints
            and not _is_bedrock_row(row)
            and _native_tab_status(cfg, label, [row]) is None
        ):
            endpoints[endpoint] = row
    merged = [dict(r) for r in configured]
    by_endpoint_id = {
        (_cl._native_endpoint(r), _cl._live_model_id(r)): i
        for i, r in enumerate(merged)
        if _cl._native_endpoint(r) and _cl._live_model_id(r)
    }
    # Aliases of every other provider stay taken: a new wire id never
    # shadows a model another tab already lists.
    used_aliases = {str(r.get("alias") or "").strip() for r in merged} | set(reserved_aliases)
    listed = {_cl._native_provider_id(r) for r in merged if _cl._live_model_id(r)}
    error: dict[str, str] | None = None
    answered = 0
    listed_models = 0
    for endpoint, template in endpoints.items():
        try:
            headers_fn = getattr(cfg, "upstream_headers", None)
            headers = headers_fn(endpoint, _cl._live_model_id(template)) if callable(headers_fn) else {}
            _provider, live = providers.detect_from_endpoint_sync(endpoint, timeout=5.0, headers=headers)
        except Exception as exc:  # noqa: BLE001
            _LOG.info("picker: %s model list failed: %s", label, type(exc).__name__)
            error = error or _fetch_error(label, exc)
            continue
        answered += 1
        listed_models += sum(
            1
            for raw in (live if isinstance(live, list) else [])
            if isinstance(raw, dict) and _cl._live_model_id(raw) and _cl._row_is_llm(raw)
        )
        _cl.merge_endpoint_live_rows(
            merged,
            endpoint,
            template,
            live,
            by_endpoint_id=by_endpoint_id,
            used_aliases=used_aliases,
            listed=listed,
        )
    if error is not None:
        return [], error
    if not endpoints:
        return [], _error("no_models", f"{label}: no endpoint is configured for this provider")
    if not answered:
        return [], error
    if not listed_models:
        # It answered, with nothing: its configured rows stay listed under
        # the message, unsaved.
        return merged, _error("no_models", f"{label}: the provider lists no models")
    return merged, None


def _save_tab_rows(agent: str, key: str, rows: list[dict[str, Any]], base: list[dict[str, Any]]) -> None:
    """Replace one provider's rows in the shared native catalogue.

    Rows of every other provider are kept, so no refresh of one tab can
    shrink another. ``base`` seeds the catalogue when nothing is saved yet.
    """
    with _cl.ModelDiscoveryLock(timeout_s=10.0, strict=True):
        data = _cl.read_cache_file()
        saved = data.get(_cl.NATIVE_CACHE_KEY)
        current = saved if isinstance(saved, list) and saved else base
        try:
            from sharedagents_agent.discovery.bedrock import DISCOVERED_BY
        except Exception:  # noqa: BLE001
            DISCOVERED_BY = "bedrock-environment"
        # Ambient Bedrock rows stay out of the shared native catalogue.
        current = [r for r in current if isinstance(r, dict) and r.get("discoveredBy") != DISCOVERED_BY]
        kept = [r for r in stamp_tabs(agent, _cl._native_picker_models(current)) if tab_key(r.get("tab")) != key]
        fresh = [{k: v for k, v in r.items() if k != "tab"} for r in rows]
        kept = [{k: v for k, v in r.items() if k != "tab"} for r in kept]
        stamps = data.get(REFRESHED_AT_KEY)
        stamps = dict(stamps) if isinstance(stamps, dict) else {}
        scope = _cache_key(agent, key)
        stamps[scope] = time.time()
        scoped = data.get(PROVIDER_ROWS_KEY)
        scoped = dict(scoped) if isinstance(scoped, dict) else {}
        scoped[scope] = fresh
        _cl.update_cache_entries(
            {
                _cl.NATIVE_CACHE_KEY: kept + [r for r in fresh if r.get("discoveredBy") != DISCOVERED_BY],
                REFRESHED_AT_KEY: stamps,
                PROVIDER_ROWS_KEY: scoped,
            }
        )


def _after_refresh(cfg: Any, agent: str, rows: list[dict[str, Any]]) -> None:
    """Pricing reconcile and output-limit catalogue: off the click's path."""

    def run() -> None:
        try:
            from sharedagents_model import catalogue

            catalogue.refresh()
        except Exception as exc:  # noqa: BLE001
            _LOG.info("model catalogue refresh skipped (%s)", type(exc).__name__)
        try:
            from sharedagents_model.pricing_sync import sync_after_discovery

            sync_after_discovery(
                path=getattr(cfg, "pricing_path", None),
                discovered_rows=[(agent, row) for row in rows],
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.info("model pricing sync skipped (%s)", type(exc).__name__)

    threading.Thread(target=run, name="picker-after-refresh", daemon=True).start()


def _result(
    agent: str,
    label: str,
    rows: list[dict[str, Any]],
    tabs: list[dict[str, Any]],
    *,
    error: dict[str, str] | None = None,
    degraded: bool = False,
    loaded: list[str] | None = None,
    refreshed: bool = False,
) -> dict[str, Any]:
    return {
        "agent": agent,
        "provider": label,
        "models": rows,
        "providers": tabs,
        "loaded": loaded if loaded is not None else [label],
        "error": error,
        "degraded": bool(degraded),
        "refreshed": bool(refreshed),
    }


def _native_provider_models(
    agent: str,
    cfg: Any,
    provider: str,
    *,
    force: bool,
    allow_live: bool,
) -> dict[str, Any]:
    key = tab_key(provider)
    scope = _cache_key(agent, key)
    cache = _cl.read_cache_file()
    summary = _native_summary(agent, cfg, cache)
    tabs = summary["providers"]
    tab = next((t for t in tabs if tab_key(t["id"]) == key), None)
    if tab is None:
        return _result(
            agent, provider, [], tabs, error=_error("no_models", f"{provider}: not a provider of {agent}"), loaded=[]
        )
    label = tab["id"]
    if tab.get("error"):
        return _result(agent, label, [], tabs, error=tab["error"])
    known = [r for r in summary["_rows"] if tab_key(r.get("tab")) == key]
    saved = (cache.get(PROVIDER_ROWS_KEY) or {}).get(scope)
    if isinstance(saved, list) and _provider_kind(saved) == "bedrock":
        known = stamp_tabs(agent, saved)
        for row in known:
            row["tab"] = label
    target = _default_target(cfg)
    kind = (
        _provider_kind(known) if known else ("codex" if key == "codex" else ("bedrock" if key == "bedrock" else "http"))
    )
    live = allow_live and (force or _is_stale(cache, scope))
    rows = known
    error: dict[str, str] | None = None
    refreshed = False

    if live and kind == "http":
        configured = stamp_tabs(agent, _cl._native_picker_models(getattr(cfg, "models", None) or []))
        configured_tab = [r for r in configured if tab_key(r.get("tab")) == key]
        reserved = {str(r.get("alias") or "").strip() for r in summary["_rows"] if tab_key(r.get("tab")) != key}
        merged, error = _fetch_http_tab(cfg, label, configured_tab, known, reserved)
        if error is None:
            rows = [r for r in stamp_tabs(agent, merged) if tab_key(r.get("tab")) == key]
            _save_tab_rows(agent, key, rows, summary["_rows"])
            refreshed = True
            _after_refresh(cfg, agent, rows)
    elif live and kind == "codex":
        try:
            from sharedagents_agent.discovery.openai_codex import discover_codex_oauth_transport_models

            fresh = discover_codex_oauth_transport_models(force=force)
        except Exception as exc:  # noqa: BLE001
            fresh = []
            _LOG.info("picker: Codex model list failed: %s", type(exc).__name__)
        if fresh:
            rows = list(stamp_tabs(agent, _cl._native_picker_models(fresh)))
            for r in rows:
                r["tab"] = label
            _save_tab_rows(agent, key, rows, summary["_rows"])
            refreshed = True
        else:
            error = _error("unreachable", f"{label}: the model list could not be read")
    elif live and kind == "bedrock":
        try:
            from sharedagents_agent.discovery.bedrock import sync_bedrock_models_into_cfg

            scratch = _load_cfg(agent)
            all_rows = sync_bedrock_models_into_cfg(scratch, force=force)
            fresh = [r for r in stamp_tabs(agent, _cl._native_picker_models(all_rows)) if tab_key(r.get("tab")) == key]
        except Exception as exc:  # noqa: BLE001
            fresh = []
            _LOG.info("picker: Bedrock model list failed: %s", type(exc).__name__)
        if fresh:
            rows = fresh
            _save_tab_rows(agent, key, rows, summary["_rows"])
            refreshed = True
        else:
            error = _error("unreachable", f"{label}: the AWS model catalogue could not be read")

    if error is None and not rows:
        error = _error("no_models", f"{label}: the provider lists no models")
    rows = _mark_default(rows, target)
    return _result(agent, label, rows, tabs, error=error, degraded=error is not None, refreshed=refreshed)


def _external_provider_models(
    agent: str,
    spec: Any,
    provider: str,
    *,
    force: bool,
    allow_live: bool,
) -> dict[str, Any]:
    """An external engine's one catalogue — every tab it lists loads at once."""
    if allow_live:
        rows, degraded = _cl.discover_agent_models_status(agent, force=force)
    else:
        summary = _summary_for(agent, _cl.read_cache_file())
        rows = summary["_rows"] if summary else []
        degraded = False
    rows = stamp_tabs(agent, rows if isinstance(rows, list) else [], spec)
    labels: dict[str, str] = {}
    for row in rows:
        labels.setdefault(tab_key(row["tab"]), row["tab"])
    requested = labels.get(tab_key(provider)) or provider or _engine_tab_for(spec, agent)
    labels.setdefault(tab_key(requested), requested)
    name = str(getattr(spec, "short_name", "") or agent)
    error: dict[str, str] | None = None
    if degraded:
        error = _error(
            "unreachable",
            f"{name}: the model list could not be loaded" + ("; showing the built-in list" if rows else ""),
        )
    elif not rows:
        error = _error("no_models", f"{name}: no models are advertised")
    tabs = [
        _tab_entry(label, error if tab_key(label) == tab_key(requested) and not rows else None)
        for label in labels.values()
    ]
    return _result(
        agent,
        requested,
        rows,
        tabs,
        error=error,
        degraded=degraded,
        loaded=list(labels.values()),
        refreshed=bool(force and allow_live),
    )


def provider_models(
    agent: str,
    provider: str,
    *,
    force: bool = False,
    allow_live: bool = True,
    row_filter: RowFilter | None = None,
) -> dict[str, Any]:
    """Rows of one provider tab of ``agent``, asking only that provider.

    ``allow_live=False`` serves configured and saved rows only (used for
    accounts that may not trigger a provider request). ``force`` skips the
    per-provider cache. The result names an ``error`` (``no_key`` /
    ``no_models`` / ``unreachable``) when the tab cannot list models.
    """
    if row_filter is not None:
        summary = agent_summary(agent, row_filter=row_filter) or {}
        tabs = summary.get("providers") or []
        if not any(tab_key(t["id"]) == tab_key(provider) for t in tabs):
            return _result(
                agent, provider, [], tabs, loaded=[], error=_error("no_models", "No granted models for this provider")
            )
    spec = engine_spec(agent)
    if spec is not None:
        result = _external_provider_models(agent, spec, provider, force=force, allow_live=allow_live)
    else:
        try:
            cfg = _load_cfg(agent)
        except FileNotFoundError:
            return _result(
                agent, provider, [], [], error=_error("no_models", f"agent {agent!r} is not configured"), loaded=[]
            )
        scope = _cache_key(agent, provider)
        lock_name = hashlib.sha256(scope.encode()).hexdigest()
        with _cl.ModelDiscoveryLock(timeout_s=15.0, name=f"provider-{lock_name}", strict=True):
            result = _native_provider_models(agent, cfg, provider, force=force, allow_live=allow_live)
    if row_filter is not None:
        rows = row_filter(agent, result["models"])
        keys = {tab_key(r.get("tab")) for r in rows}
        summary = agent_summary(agent, row_filter=row_filter) or {}
        keys |= {tab_key(t["id"]) for t in summary.get("providers") or []}
        result["providers"] = [t for t in result["providers"] if tab_key(t["id"]) in keys]
        result["loaded"] = [label for label in result["loaded"] if tab_key(label) in keys]
        if result["models"] and not rows and not result["error"]:
            result["error"] = _error(
                "no_models", f"{result['provider']}: no model of this provider is granted to your account"
            )
        result["models"] = rows
    result["models"] = public_model_rows(result["models"])
    return result


__all__ = [
    "PROVIDER_TTL_S",
    "REFRESHED_AT_KEY",
    "agent_summary",
    "engine_spec",
    "picker_summary",
    "provider_models",
    "stamp_tabs",
    "tab_key",
]
