"""Provider-model effort capability discovery.

The model picker calls each configured OpenAI-compatible catalogue and keeps
only the effort vocabulary/default for the requested model. Responses are
cached briefly so opening the picker does not repeatedly hit upstream APIs.
No credentials or raw catalogue payloads are retained in the cache.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .model_parameters import effort_capability, resolve_model_parameters
from .models import derive_effort_capability, effort_capability_from_metadata

_TTL_S = 300.0
_ERROR_TTL_S = 30.0
#: A provider's advertised effort vocabulary is stable, but the model-list
#: cache above is deliberately short-lived. The WIRE reads this module with
#: ``probe=False``, so without a longer-lived memory the 5-minute TTL expiring
#: silently downgraded an advertising provider to "unconfirmed" and dropped
#: ``reasoning_effort`` from the request — the operator's explicit pick became
#: a no-op mid-session. Keep what we last learned well past the freshness TTL.
_LAST_KNOWN_TTL_S = 7 * 24 * 3600.0
_cache: dict[tuple[str, str], tuple[float, dict[str, dict[str, Any]]]] = {}
_last_known: dict[tuple[str, str], tuple[float, dict[str, dict[str, Any]]]] = {}
_cache_lock = threading.Lock()


def clear_effort_discovery_cache() -> None:
    """Clear the short-lived catalogue cache (test/reload seam).

    Clears the long-lived last-known vocabulary too, so a test starts from a
    genuinely unknown provider.
    """
    with _cache_lock:
        _cache.clear()
        _last_known.clear()


def _default_fetch(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
) -> Mapping[str, Any]:
    import httpx

    response = httpx.get(url, headers=dict(headers), timeout=timeout)
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, Mapping) else {}


def _entries(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = payload.get("data") or payload.get("models") or []
    return [item for item in raw if isinstance(item, Mapping)]


def _entry_id(entry: Mapping[str, Any]) -> str:
    return str(entry.get("id") or entry.get("slug") or entry.get("name") or "").strip()


def _entry_capability(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    return effort_capability_from_metadata(entry)


def _catalogue_capabilities(
    endpoint: str,
    headers: Mapping[str, str],
    *,
    timeout: float,
    fetch: Callable[[str, Mapping[str, str], float], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    payload = fetch(f"{endpoint.rstrip('/')}/models", headers, timeout)
    out: dict[str, dict[str, Any]] = {}
    for entry in _entries(payload):
        model_id = _entry_id(entry)
        capability = _entry_capability(entry)
        if model_id and capability is not None:
            out[model_id] = capability
    return out


def discover_effort_capabilities(
    rows: Sequence[Mapping[str, Any]],
    cfg: Any,
    *,
    timeout: float = 4.0,
    force: bool = False,
    probe: bool = True,
    fetch: Callable[[str, Mapping[str, str], float], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any] | None]:
    """Return API-derived effort capabilities aligned with ``rows``.

    Explicit row capabilities are authoritative and require no call. Remaining
    rows are grouped by endpoint/auth-variable name so one ``GET /models``
    serves every model on that provider.

    ``probe=False`` reads the process cache only — used by the sessionless
    WebUI picker so opening the sheet never waits on upstream HTTP.
    """
    result: list[dict[str, Any] | None] = [None] * len(rows)
    groups: dict[tuple[str, str], list[tuple[int, Mapping[str, Any]]]] = {}
    from .bedrock_config import is_bedrock_row
    from .models import bedrock_effort_capability

    for index, row in enumerate(rows):
        # Native Bedrock is not an OpenAI-compat ``/models`` catalogue: probing
        # ``bedrock://REGION/models`` is a guaranteed failure and its v1 has no
        # reasoning depth at all. Report the explicit unsupported capability
        # instead of a cache miss or a live-looking empty list.
        if is_bedrock_row(row):
            result[index] = bedrock_effort_capability()
            continue
        declaration = resolve_model_parameters(row)
        if declaration is not None:
            result[index] = effort_capability(declaration)
            continue
        override = row.get("effort")
        if "effort" in row and (override is None or isinstance(override, Mapping) and "values" in override):
            result[index] = derive_effort_capability(row)
            continue
        direct = effort_capability_from_metadata(row)
        if direct is not None:
            result[index] = direct
            continue
        endpoint = str(row.get("baseUrl") or row.get("base_url") or row.get("endpoint") or "").strip().rstrip("/")
        if not endpoint:
            continue
        # Codex ChatGPT OAuth Responses is not an OpenAI-compat ``/models``
        # catalogue. Probing it without the OAuth bearer stalls the Web UI
        # picker (~4s) and 401s. Effort, when known, already came from
        # ChatGPT catalogue metadata on the row.
        api = str(row.get("api") or "").strip().lower()
        auth = str(row.get("auth") or "").strip().lower()
        ep = endpoint.lower()
        if (
            api in ("codex-responses", "openai-chatgpt-responses")
            or auth == "codex-oauth"
            or ep.endswith("/backend-api/codex")
            or ep.endswith("/backend-api/codex/v1")
        ):
            result[index] = None
            continue
        key_env = str(row.get("apiKeyEnv") or row.get("api_key_env") or "").strip()
        groups.setdefault((endpoint, key_env), []).append((index, row))

    now = time.monotonic()
    fetcher = fetch or _default_fetch
    for cache_key, members in groups.items():
        endpoint, _key_env = cache_key
        capabilities: dict[str, dict[str, Any]] | None = None
        if not force and fetch is None:
            with _cache_lock:
                hit = _cache.get(cache_key)
                if hit and hit[0] > now:
                    capabilities = {k: dict(v) for k, v in hit[1].items()}
                else:
                    # Freshness miss. A provider's advertised vocabulary
                    # outlives the model-list TTL, and the wire reads this
                    # cache-only, so serve what we last learned rather than
                    # silently downgrading the row to "unconfirmed" (which
                    # drops ``reasoning_effort`` from the request).
                    remembered = _last_known.get(cache_key)
                    if remembered and remembered[0] > now and remembered[1]:
                        capabilities = {k: dict(v) for k, v in remembered[1].items()}
        if capabilities is None:
            if not probe and fetch is None:
                capabilities = {}
            else:
                first_row = members[0][1]
                model_id = _entry_id(first_row)
                try:
                    headers = cfg.upstream_headers(endpoint, model_id)
                except Exception:  # noqa: BLE001
                    headers = {}
                try:
                    capabilities = _catalogue_capabilities(
                        endpoint,
                        headers,
                        timeout=timeout,
                        fetch=fetcher,
                    )
                    ttl = _TTL_S
                except Exception:  # noqa: BLE001
                    capabilities = {}
                    ttl = _ERROR_TTL_S
                if fetch is None:
                    stamp = time.monotonic()
                    with _cache_lock:
                        _cache[cache_key] = (
                            stamp + ttl,
                            {k: dict(v) for k, v in capabilities.items()},
                        )
                        if capabilities:
                            # Keep it for the wire's cache-only read, well past
                            # the freshness TTL above.
                            _last_known[cache_key] = (
                                stamp + _LAST_KNOWN_TTL_S,
                                {k: dict(v) for k, v in capabilities.items()},
                            )
        for index, row in members:
            result[index] = capabilities.get(_entry_id(row))
    return result


def discover_effort_capability(
    row: Mapping[str, Any],
    cfg: Any,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Single-row convenience wrapper used by model/effort validation."""
    return discover_effort_capabilities([row], cfg, **kwargs)[0]
