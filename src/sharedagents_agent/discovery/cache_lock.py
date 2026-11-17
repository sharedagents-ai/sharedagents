"""Model cache with file locking to ensure non-parallel discovery across processes."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat
from sharedagents_core.layout import sharedagents_root

_LOG = logging.getLogger("sharedagents.discovery.cache_lock")

# Native agents (jarvis / build / plan / …) all share the ONE global
# ``config.json`` model catalogue. The discovery cache must therefore be keyed
# once for the whole native family, not per agent name — otherwise a force
# refresh of a single native agent (jarvis) enriches only that agent while its
# siblings (build) keep showing the stale configured list.
NATIVE_CACHE_KEY = "__native__"

# Provider catalogues sometimes publish an unversioned wire id. DeepSeek's
# ``GET /v1/models`` returns ``deepseek-flash``; the
# operator-facing alias must still carry the version. The wire id stays in
# ``id`` so requests and row resolution keep working.
_DISCOVERED_ALIAS_OVERRIDES = {
    "deepseek-flash": "deepseek-v4.1-flash",
}


def _paths() -> tuple[Path, Path]:
    root = sharedagents_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / "discovered_models.json", root / "discovered_models.lock"


class ModelDiscoveryLock:
    """Exclusive file lock ensuring no parallel discovery runs across processes."""

    def __init__(self, timeout_s: float = 30.0, *, name: str = "", strict: bool = False) -> None:
        self.timeout_s = timeout_s
        self.strict = strict
        # ``name`` scopes the lock to one provider so a slow provider never
        # holds up a click on another one; the default guards the whole file.
        self.name = "".join(ch for ch in str(name or "") if ch.isalnum() or ch in "-_.")
        self._fp: Any | None = None
        self._acquired: bool = False

    def __enter__(self) -> ModelDiscoveryLock:
        try:
            _, lock_path = _paths()
            if self.name:
                lock_path = lock_path.with_name(f"discovered_models.{self.name}.lock")
            self._fp = open(lock_path, "a+")
            deadline = time.monotonic() + self.timeout_s
            while True:
                try:
                    oscompat.file_lock(self._fp, nonblocking=True)
                    self._acquired = True
                    return self
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        if self.strict:
                            raise TimeoutError("Model discovery lock timed out") from None
                        _LOG.warning(
                            "Timed out after %.1fs waiting for model discovery lock; continuing",
                            self.timeout_s,
                        )
                        return self
                    time.sleep(0.1)
        except Exception as exc:  # noqa: BLE001
            if self.strict:
                if self._fp:
                    self._fp.close()
                    self._fp = None
                raise
            _LOG.warning("Failed to open or acquire model discovery lock: %s", type(exc).__name__)
            return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._fp:
            if self._acquired:
                try:
                    oscompat.file_unlock(self._fp)
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._fp.close()
            except Exception:  # noqa: BLE001
                pass
            self._fp = None
            self._acquired = False


def get_cached_models(agent_or_kind: str) -> list[dict[str, Any]]:
    """Read discovered models for an agent from the shared cache file."""
    try:
        cache_path, _ = _paths()
        if not cache_path.is_file():
            return []
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        models = data.get(agent_or_kind)
        return list(models) if isinstance(models, list) else []
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("failed to read cached models for %s: %s", agent_or_kind, type(exc).__name__)
        return []


def read_cache_file() -> dict[str, Any]:
    """The whole shared discovery cache (``{}`` when missing or unreadable)."""
    try:
        cache_path, _ = _paths()
        if not cache_path.is_file():
            return {}
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("failed to read the model discovery cache: %s", type(exc).__name__)
        return {}


def update_cache_entries(entries: dict[str, Any]) -> None:
    """Replace several top-level cache entries in one atomic write.

    Callers serialise through :class:`ModelDiscoveryLock`; entries not named
    here are kept as they are.
    """
    temp_path: Path | None = None
    try:
        cache_path, _ = _paths()
        data = read_cache_file()
        data.update(entries)
        temp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_path.replace(cache_path)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("failed to update the model discovery cache: %s", type(exc).__name__)
        if temp_path and temp_path.is_file():
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


def save_cached_models(agent_or_kind: str, models: list[dict[str, Any]]) -> None:
    """Save discovered models for an agent to the shared cache file."""
    temp_path: Path | None = None
    try:
        cache_path, _ = _paths()
        data: dict[str, Any] = {}
        if cache_path.is_file():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:  # noqa: BLE001
                data = {}
        if agent_or_kind == NATIVE_CACHE_KEY:
            # Self-heal: purge legacy per-native-agent entries superseded by
            # the single shared catalogue.
            for legacy in _native_agent_names():
                data.pop(legacy, None)
        data[agent_or_kind] = models
        temp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_path.replace(cache_path)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("failed to save cached models for %s: %s", agent_or_kind, type(exc).__name__)
        if temp_path and temp_path.is_file():
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


def _native_agent_names() -> list[str]:
    """Configured agents that are NOT external engines (jarvis / build / …)."""
    try:
        from sharedagents_agent.agent import list_agents
        from sharedagents_agent.external_registry import spec_for_short_name

        names: list[str] = []
        for name in list_agents():
            try:
                spec_for_short_name(name)
            except ValueError:
                names.append(name)
            except Exception:  # noqa: BLE001
                names.append(name)
        return names
    except Exception:  # noqa: BLE001
        return []


def read_all_cached_models() -> dict[str, list[dict[str, Any]]]:
    """Read discovered models, expanded per agent.

    External engines keep their own per-agent cache entry. Native agents share
    a single :data:`NATIVE_CACHE_KEY` catalogue that is fanned out to every
    configured native agent here, so one refresh hydrates them all. Legacy
    per-native-agent entries written by an older build are ignored.
    """
    try:
        cache_path, _ = _paths()
        if not cache_path.is_file():
            return {}
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
    except Exception:  # noqa: BLE001
        return {}

    native_names = set(_native_agent_names())
    out: dict[str, list[dict[str, Any]]] = {}
    native_rows = data.get(NATIVE_CACHE_KEY)
    for key, rows in data.items():
        if key == NATIVE_CACHE_KEY:
            continue
        if key in native_names:
            continue  # legacy stale per-native-agent entry
        if isinstance(rows, list):
            out[key] = rows
    if isinstance(native_rows, list):
        for name in native_names:
            out[name] = native_rows
    return out


def _native_picker_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy configured rows and stamp a real provider (never the agent name).

    Local vLLM entries often have ``backend: vllm`` and no ``provider``.
    The Web picker used to fall back to jarvis/build/plan.
    """
    out: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        if not str(row.get("provider") or "").strip():
            backend = str(row.get("backend") or "").strip()
            key = backend.lower()
            url = str(row.get("baseUrl") or row.get("endpoint") or "")
            if key and key not in {"openai", "unknown", "configured"}:
                row["provider"] = backend
            elif any(token in url for token in ("127.0.0.1", "localhost", ":8000")):
                row["provider"] = backend or "vllm"
            else:
                row["provider"] = backend or "vllm"
        out.append(row)
    return out


def _native_endpoint(row: dict[str, Any]) -> str:
    return str(row.get("baseUrl") or row.get("endpoint") or "").strip().rstrip("/")


def _native_provider_id(row: dict[str, Any]) -> tuple[str, str]:
    """One picker row per provider and wire id.

    The local vLLM router lists every pool it fronts, so a model configured
    directly against its own server comes back from the router endpoint too.
    The row already listed (the operator's own) wins.
    """
    return str(row.get("provider") or "").strip().lower(), _live_model_id(row)


def native_models_with_cache(
    agent_name: str,
    rows: list[dict[str, Any]],
    *,
    codex_available: bool | None = None,
    cached_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge a forced native refresh into the current live config rows.

    The web process and the long-lived gateway do not share an ``AgentConfig``
    object. Forced discovery therefore persists its result in the shared cache;
    command/model-switch paths in the gateway use this helper to see the same
    newly advertised rows without replacing the operator's current config.

    The catalogue is global to the native family (:data:`NATIVE_CACHE_KEY`),
    so a refresh performed under one native agent is visible to every sibling.
    Provider-click rows are scoped per agent, including at the gateway's
    model validation boundary, so another account's refresh cannot replace them.

    ``codex_available`` answers the Codex-login question up front (the
    picker summary checks the login file; the default asks for request
    headers, which may refresh an expired token). ``cached_rows`` passes an
    already-read native cache entry.
    """
    current = _native_picker_models(rows)
    from sharedagents_agent.discovery.bedrock import DISCOVERED_BY

    endpoints = {_native_endpoint(row) for row in current if _native_endpoint(row)}
    try:
        if not isinstance(cached_rows, list):
            data = read_cache_file()
            cached_rows = data.get(NATIVE_CACHE_KEY) or []
            from sharedagents_agent.discovery.picker import PROVIDER_ROWS_KEY, _native_tab, tab_key

            scoped = data.get(PROVIDER_ROWS_KEY) or {}
            for key, saved in scoped.items():
                try:
                    owner, provider = json.loads(key)
                except (ValueError, TypeError):
                    continue
                if owner == agent_name and isinstance(saved, list):
                    cached_rows = [r for r in cached_rows if tab_key(_native_tab(r)) != provider] + saved
        cached = _native_picker_models(cached_rows)
    except Exception:  # noqa: BLE001
        return current
    seen_aliases = {str(row.get("alias") or "").strip() for row in current}
    seen_endpoint_ids = {
        (_native_endpoint(row), _live_model_id(row)) for row in current if _native_endpoint(row) and _live_model_id(row)
    }
    seen_provider_ids = {_native_provider_id(row) for row in current if _live_model_id(row)}
    codex_oauth_available: bool | None = codex_available
    for row in cached:
        endpoint = _native_endpoint(row)
        model_id = _live_model_id(row)
        # Config load/reload owns ambient discovery. The shared disk cache is
        # not bound to an AWS account or catalogue generation and can retain
        # deleted profiles even when the current region/auth still match.
        if row.get("discoveredBy") == DISCOVERED_BY:
            continue
        if not model_id:
            continue
        if not _row_is_llm(row):
            # A non-LLM pool (asr/decision/…) that was once discovered is
            # not a chat model: it never re-enters the picker.
            continue
        if endpoint not in endpoints:
            # A failed live Codex catalogue fetch can leave cfg.models without
            # its OAuth rows. Keep previously discovered rows selectable only
            # while this process can still use a Codex login for this endpoint.
            if row.get("auth") != "codex-oauth":
                continue
            from sharedagents_agent.discovery.openai_codex import (
                CODEX_OAUTH_AUTH,
                CODEX_RESPONSES_API,
                CODEX_TRANSPORT_ALIAS_PREFIX,
                codex_oauth_request_headers,
                codex_responses_base_url,
            )

            is_codex_row = (
                row.get("auth") == CODEX_OAUTH_AUTH
                and row.get("api") == CODEX_RESPONSES_API
                and str(row.get("alias") or "").startswith(CODEX_TRANSPORT_ALIAS_PREFIX)
                and endpoint == codex_responses_base_url().rstrip("/")
            )
            if not is_codex_row:
                continue
            if codex_oauth_available is None:
                try:
                    codex_oauth_available = bool(codex_oauth_request_headers())
                except Exception:  # noqa: BLE001 — retain non-Codex rows
                    codex_oauth_available = False
            if not codex_oauth_available:
                continue
        if (endpoint, model_id) in seen_endpoint_ids:
            continue
        if _native_provider_id(row) in seen_provider_ids:
            continue
        alias = str(row.get("alias") or model_id).strip() or model_id
        if alias in seen_aliases:
            continue
        current.append(dict(row))
        seen_aliases.add(alias)
        seen_endpoint_ids.add((endpoint, model_id))
        seen_provider_ids.add(_native_provider_id(row))
    return current


def _live_model_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("model") or row.get("name") or "").strip()


#: Pool kinds a local vLLM router advertises that are NOT chat models.
#: Mirrors ``sharedagents_web.router_status`` (asr/tts/embedding/reranker) plus
#: its ``decision`` scorer pools. The LLM picker offers only ``llm`` pools;
#: endpoints that carry no kind (raw vLLM / SGLang servers, older routers)
#: stay assumed-LLM for back-compat.
_NON_LLM_POOL_KINDS = frozenset(
    {
        "decision",
        "embedding",
        "embed",
        "reranker",
        "asr",
        "tts",
        "whisper",
        "speech",
    }
)


def _row_is_llm(raw: dict[str, Any]) -> bool:
    """Whether a provider-advertised row is a chat model.

    The local vLLM router lists every pool it fronts, including
    non-LLM pools (asr/tts/embedding/reranker and the ``decision``
    scorers). Those must not surface in the LLM picker. Rows without a
    ``kind`` (raw vLLM/SGLang servers, older routers) remain LLM by
    assumption.
    """
    kind = str(raw.get("kind") or "").strip().lower()
    return kind not in _NON_LLM_POOL_KINDS


def _live_model_display(row: dict[str, Any], model_id: str) -> str:
    for key in ("display", "display_name", "displayName", "title", "name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    if model_id in {"deepseek-flash", "deepseek-v4.1-flash"}:
        return "DeepSeek Flash 4.1"
    return model_id


def _live_output_limit(row: dict[str, Any]) -> int:
    """The output limit the provider itself reports for a live model row
    (OpenRouter's ``top_provider.max_completion_tokens``, or a flat field)."""
    top = row.get("top_provider")
    for value in (
        top.get("max_completion_tokens") if isinstance(top, dict) else None,
        row.get("max_completion_tokens"),
        row.get("max_output_tokens"),
        row.get("output_token_limit"),
    ):
        if isinstance(value, bool):
            continue
        try:
            number = int(value or 0)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return 0


def _live_context_window(row: dict[str, Any]) -> int:
    for key in ("contextWindow", "context_window", "context_length", "max_model_len", "max_context_length"):
        try:
            value = int(row.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def merge_endpoint_live_rows(
    configured: list[dict[str, Any]],
    endpoint: str,
    template: dict[str, Any],
    live_rows: Any,
    *,
    by_endpoint_id: dict[tuple[str, str], int],
    used_aliases: set[str],
    listed: set[tuple[str, str]],
) -> None:
    """Fold one endpoint's live ``GET /models`` rows into ``configured``.

    Configured rows gain the provider's display name, context window and
    output limit; new wire ids are appended with the endpoint's connection
    settings (``template``). ``by_endpoint_id`` / ``used_aliases`` /
    ``listed`` are updated in place so several endpoints can be folded in
    turn. Shared by the full native refresh and the per-provider picker
    request.
    """
    if not isinstance(live_rows, list):
        return
    for raw in live_rows:
        if not isinstance(raw, dict):
            continue
        model_id = _live_model_id(raw)
        if not model_id:
            continue
        key = (endpoint, model_id)
        existing_index = by_endpoint_id.get(key)
        display = _live_model_display(raw, model_id)
        context_window = _live_context_window(raw)
        output_limit = _live_output_limit(raw)
        if existing_index is not None:
            existing = dict(configured[existing_index])
            if display and not existing.get("display"):
                existing["display"] = display
            if context_window and not existing.get("contextWindow"):
                existing["contextWindow"] = context_window
            # The provider's own limit, kept apart from the operator's
            # ``maxTokens`` (which outranks it) and refreshed each time.
            if output_limit:
                existing["outputLimit"] = output_limit
            configured[existing_index] = existing
            continue

        if not _row_is_llm(raw):
            # A non-LLM pool (asr/decision/…) the router fronts is not a
            # chat model: configured rows stay the only picker rows, so
            # it is never appended here.
            continue

        provider = str(template.get("provider") or raw.get("provider") or raw.get("owned_by") or "openai").strip()
        if (provider.lower(), model_id) in listed:
            continue
        alias = _DISCOVERED_ALIAS_OVERRIDES.get(model_id, model_id)
        if alias in used_aliases:
            suffix = 2
            while f"{alias}-{suffix}" in used_aliases:
                suffix += 1
            alias = f"{alias}-{suffix}"
        used_aliases.add(alias)
        new_row: dict[str, Any] = {
            "alias": alias,
            "id": model_id,
            "display": display,
            "baseUrl": endpoint,
            "backend": str(template.get("backend") or "openai"),
            "provider": provider,
        }
        kind = str(raw.get("kind") or "").strip().lower()
        if kind:
            # Self-describing cache rows: the merge path keeps or drops
            # them by pool kind without re-probing.
            new_row["kind"] = kind
        if context_window:
            new_row["contextWindow"] = context_window
        elif template.get("contextWindow"):
            new_row["contextWindow"] = template["contextWindow"]
        for field in ("apiKeyEnv", "api_key_env", "headers"):
            if field in template:
                new_row[field] = template[field]
        # Not the template's ``maxTokens``: that is one model's output
        # limit, not a connection setting. Each model gets its own, as
        # the provider reports it.
        if output_limit:
            new_row["outputLimit"] = output_limit
        configured.append(new_row)
        by_endpoint_id[key] = len(configured) - 1
        listed.add((provider.lower(), model_id))


def _native_refresh_models(cfg: Any) -> list[dict[str, Any]]:
    """Merge live ``GET /models`` rows into a native agent catalogue.

    The configured rows remain authoritative for aliases, credentials, headers,
    and deployment roles. A provider can add new wire ids at any time; those
    rows are appended with the endpoint's existing connection settings and use
    the wire id as their initial picker alias. This is deliberately a refresh
    operation rather than a boot-time HTTP probe. Ambient Bedrock discovery
    also runs at config load with a short-lived cache.
    """
    configured = _native_picker_models(getattr(cfg, "models", []) or [])
    from sharedagents_model.bedrock_config import BEDROCK_BASE_SCHEME, is_bedrock_row

    by_endpoint_id = {
        (_native_endpoint(row), _live_model_id(row)): index
        for index, row in enumerate(configured)
        if _native_endpoint(row) and _live_model_id(row)
    }
    groups: dict[str, dict[str, Any]] = {}
    for row in configured:
        # Native Bedrock rows are not an HTTP catalogue: never group/probe
        # ``bedrock://REGION`` via ``GET /models``. The AWS SDK catalogue is
        # refreshed separately below.
        if is_bedrock_row(row):
            continue
        endpoint = _native_endpoint(row)
        if endpoint and endpoint not in groups:
            groups[endpoint] = row

    try:
        from sharedagents_model import providers
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("native model provider probe unavailable (%s)", type(exc).__name__)
        return configured

    used_aliases = {str(row.get("alias") or "").strip() for row in configured}
    listed = {_native_provider_id(row) for row in configured if _live_model_id(row)}

    def fetch(item):
        endpoint, template = item
        if endpoint.lower().startswith(f"{BEDROCK_BASE_SCHEME}://") or template.get("auth") == "codex-oauth":
            return endpoint, template, []
        try:
            headers_fn = getattr(cfg, "upstream_headers", None)
            headers = headers_fn(endpoint, _live_model_id(template)) if callable(headers_fn) else {}
            _provider, live_rows = providers.detect_from_endpoint_sync(endpoint, timeout=5.0, headers=headers)
            return endpoint, template, live_rows
        except Exception as exc:  # noqa: BLE001
            _LOG.info("native model discovery failed (%s)", type(exc).__name__)
            return endpoint, template, []

    # Legacy whole-agent refresh remains available to non-chat selectors.
    # Bound latency by the slowest batch instead of summing provider timeouts.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        for endpoint, template, live_rows in pool.map(fetch, groups.items()):
            merge_endpoint_live_rows(
                configured,
                endpoint,
                template,
                live_rows,
                by_endpoint_id=by_endpoint_id,
                used_aliases=used_aliases,
                listed=listed,
            )

    try:
        cfg.models = configured
    except Exception:  # noqa: BLE001
        pass
    # Output limits for providers whose /models does not report them (the
    # models.dev catalogue OpenCode uses); a no-op while the copy is fresh.
    try:
        from sharedagents_model import catalogue

        catalogue.refresh()
    except Exception as exc:  # noqa: BLE001
        _LOG.info("model catalogue refresh skipped: %s", type(exc).__name__)
    from sharedagents_agent.discovery.bedrock import sync_bedrock_models_into_cfg

    return sync_bedrock_models_into_cfg(cfg, force=True)


_LOCAL_ONLY_PROVIDERS = frozenset({"vllm", "openai-compat", "configured", "unknown"})


def _ensure_model_provider(rows: list[dict[str, Any]], default_provider: str) -> list[dict[str, Any]]:
    """Stamp each row with the engine provider.

    vLLM is only the provider of local models. Cached external-engine rows
    historically arrived with no provider, or with ``vllm`` / ``google``,
    which made the Web picker show a vLLM rail on Antigravity and siblings.
    """
    default = str(default_provider or "").strip()
    out: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        current = str(row.get("provider") or "").strip()
        key = current.lower()
        if default and (not current or key in _LOCAL_ONLY_PROVIDERS):
            row["provider"] = default
        elif default == "antigravity" and key == "google":
            row["provider"] = "antigravity"
        out.append(row)
    return out


def discover_agent_models(
    agent_name: str,
    *,
    force: bool = False,
    timeout_s: float = 30.0,
) -> list[dict[str, Any]]:
    """Discover models for an agent under file-lock.

    External agents persist only a live discovery result; a failed refresh
    keeps the saved list (or serves the provider's own fallback unsaved).
    Native agents return their configured model list.
    On ANY failure, safely returns [] so the whole system never collapses.
    """
    return discover_agent_models_status(
        agent_name,
        force=force,
        timeout_s=timeout_s,
    )[0]


def discover_agent_models_status(
    agent_name: str,
    *,
    force: bool = False,
    timeout_s: float = 30.0,
) -> tuple[list[dict[str, Any]], bool]:
    """:func:`discover_agent_models` plus whether the rows are ``degraded``.

    Degraded means discovery failed. Rows may be the last successful list,
    the provider's static fallback or empty; the WebUI must preserve a good
    list and show the refresh error rather than discard a selected model.
    """
    try:
        from sharedagents_agent.discovery import discover_models
        from sharedagents_agent.external_registry import spec_for_short_name

        try:
            spec = spec_for_short_name(agent_name)
        except ValueError:
            spec = None
        if not spec:
            # Native agents share the configured OpenAI-compatible deployments.
            # A forced picker refresh also probes each deployment so a newly
            # advertised provider model can be selected without restarting the
            # long-lived gateway.
            try:
                from sharedagents_agent.agent import AgentConfig

                cfg = AgentConfig.load(agent_name)
                if force:
                    # A picker refresh must retry Codex too. Preserve usable
                    # saved OAuth rows if that live request fails, before the
                    # refreshed native catalogue replaces the shared cache.
                    cfg._sync_codex_oauth_models(force=True)
                    rows = _native_refresh_models(cfg)
                else:
                    rows = getattr(cfg, "models", []) or []
                rows = native_models_with_cache(agent_name, rows)
                rows = _native_picker_models(rows)
                default_target = getattr(cfg, "upstream_model", None) or getattr(cfg, "upstream_alias", None)
                if isinstance(default_target, str) and default_target.strip():
                    target_str = default_target.strip()
                    for r in rows:
                        if (
                            str(r.get("id") or "").strip() == target_str
                            or str(r.get("alias") or "").strip() == target_str
                        ):
                            r["default"] = True
                if force:
                    try:
                        # One shared entry for the whole native family so a
                        # refresh under any native agent hydrates every sibling.
                        save_cached_models(NATIVE_CACHE_KEY, rows)
                    except Exception as exc:  # noqa: BLE001
                        _LOG.warning(
                            "failed to cache native models for %s: %s",
                            agent_name,
                            type(exc).__name__,
                        )
                    # Reconcile AFTER the full native/Bedrock refresh, using
                    # the fresh rows even when writing the discovery cache failed.
                    try:
                        from sharedagents_model.pricing_sync import sync_after_discovery

                        sync_after_discovery(
                            path=getattr(cfg, "pricing_path", None),
                            discovered_rows=[(agent_name, row) for row in rows],
                        )
                    except Exception as exc:  # noqa: BLE001
                        _LOG.info("model pricing sync skipped: %s", type(exc).__name__)
                return rows, False
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("native model discovery failed for %s: %s", agent_name, type(exc).__name__)
                return [], True

        # RPC catalogues carry actual provider identities (including a custom
        # local provider). Do not relabel those as the Pi engine itself.
        expected_provider = "" if spec.adapter == "rpc" else str(spec.provider or spec.short_name or agent_name).strip()

        # External agent: return cache if not forced
        if not force:
            try:
                cached = get_cached_models(agent_name)
                if cached:
                    return _stamp_configured_default(
                        agent_name, _ensure_model_provider(cached, expected_provider)
                    ), False
            except Exception:  # noqa: BLE001
                pass

        # Acquire lock and execute discovery
        try:
            with ModelDiscoveryLock(timeout_s=timeout_s):
                if not force:
                    try:
                        cached = get_cached_models(agent_name)
                        if cached:
                            return _stamp_configured_default(
                                agent_name, _ensure_model_provider(cached, expected_provider)
                            ), False
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    rows = discover_models(spec.kind, force=True)
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("discovery failed for %s: %s", agent_name, type(exc).__name__)
                    rows = []
                rows = _ensure_model_provider(rows if isinstance(rows, list) else [], expected_provider)
                # A failed refresh (nothing, or only the static fallback) is
                # never persisted: it would replace a good live list until the
                # next forced refresh. Keep serving the saved list instead.
                from ._common import served_fallback

                if not rows or served_fallback(str(spec.provider or "")):
                    prior = get_cached_models(agent_name)
                    if prior:
                        return _stamp_configured_default(
                            agent_name, _ensure_model_provider(prior, expected_provider)
                        ), True
                    return _stamp_configured_default(agent_name, rows), True
                try:
                    save_cached_models(agent_name, rows)
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("failed to cache models for %s: %s", agent_name, type(exc).__name__)
                # The one funnel every external-engine discovery goes through,
                # so one hook covers the CLI/TUI and the WebUI.
                # Debounced and never raising.
                try:
                    from sharedagents_model.pricing_sync import sync_after_discovery

                    sync_after_discovery()
                except Exception as exc:  # noqa: BLE001
                    _LOG.info("model pricing sync skipped: %s", type(exc).__name__)
                return _stamp_configured_default(agent_name, rows), False
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("locked discovery failed for %s: %s", agent_name, type(exc).__name__)
            saved = _stamp_configured_default(
                agent_name, _ensure_model_provider(get_cached_models(agent_name), expected_provider)
            )
            return saved, True
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("unexpected error in discover_agent_models(%s): %s", agent_name, type(exc).__name__)
        return [], True


def _row_names_model(row: dict[str, Any], target: str) -> bool:
    return any(str(row.get(key) or "").strip() == target for key in ("id", "alias", "wire_id", "model_id"))


def _stamp_configured_default(
    agent_name: str,
    rows: list[dict[str, Any]],
    agent_cfg: Any = None,
) -> list[dict[str, Any]]:
    """Mark the row for the agent's saved upstream_model / upstream_alias as the default.

    Rows are only touched when the saved model is in the catalogue: a saved model
    the catalogue does not list leaves the catalogue's own default flags alone.
    """
    if not isinstance(rows, list):
        return rows
    try:
        cfg = agent_cfg
        if cfg is None and agent_name:
            from sharedagents_agent.agent import AgentConfig

            cfg = AgentConfig.load(agent_name, ambient_discovery=False)
        default_target = getattr(cfg, "upstream_model", None) or getattr(cfg, "upstream_alias", None)
        target = default_target.strip() if isinstance(default_target, str) else ""
        if not target or not any(isinstance(r, dict) and _row_names_model(r, target) for r in rows):
            return rows
        stamped: list[Any] = []
        for r in rows:
            if not isinstance(r, dict):
                stamped.append(r)
                continue
            r_copy = dict(r)
            if _row_names_model(r_copy, target):
                r_copy["default"] = True
            else:
                r_copy.pop("default", None)
            stamped.append(r_copy)
        return stamped
    except Exception:  # noqa: BLE001
        return rows


stamp_configured_default = _stamp_configured_default


# ── Model names for access grants ─────────────────────────
#
# A model grant names a catalogue model; a session or a delegation may refer
# to the same model by id, alias, wire id or real id. Built from the CACHED
# catalogues only — never a live provider probe (this sits on submit and
# delegation paths) — and refreshed every 30 s.

_ALIAS_INDEX: dict[str, Any] = {"at": 0.0, "index": {}}
_ALIAS_INDEX_TTL_S = 30.0
_MODEL_NAME_KEYS = ("id", "alias", "wireId", "realId")


def model_alias_index() -> dict[str, set[str]]:
    """Every name a cached catalogue model answers to → all of its names."""
    import time as _time

    now = _time.monotonic()
    if now - float(_ALIAS_INDEX["at"]) < _ALIAS_INDEX_TTL_S:
        return _ALIAS_INDEX["index"]
    index: dict[str, set[str]] = {}
    sources: list[list] = []
    try:
        sources.extend((read_all_cached_models() or {}).values())
    except Exception:  # noqa: BLE001
        pass
    try:  # configured deployments are grantable before any catalogue is cached
        from sharedagents_core.kernel.config import load_global_config

        sources.append((load_global_config() or {}).get("models") or [])
    except Exception:  # noqa: BLE001
        pass
    for rows in sources:
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            names = {str(row.get(k) or "").strip() for k in _MODEL_NAME_KEYS} - {""}
            for name in names:
                index.setdefault(name, set()).update(names)
    _ALIAS_INDEX.update(at=now, index=index)
    return index


def expand_model_names(*names: str) -> set[str]:
    """``names`` plus every other name the cached catalogues know them by."""
    wanted = {str(n or "").strip() for n in names} - {""}
    index = model_alias_index()
    out = set(wanted)
    for name in wanted:
        out |= index.get(name, set())
    return out
