"""Provider-specific model discovery implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._common import (
    _LOG,
    OPENAI_MODELS_URL,
    _cache_get,
    _cache_put,
    _cache_put_ttl,
    _codex_cache_context_windows,
    _codex_gate_lookup,
    _http_get_json,
    _is_codex_model_id,
    _row_with_effort,
)
from .openai_codex import (
    _codex_chatgpt_rows,
    _codex_model_window,
    _codex_version_lt,
    _openai_api_key,
)


def _merge_codex_cli_cache(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union live rows with the installed codex's own model list.

    Rollout catalogues can briefly disagree between the saved list and the
    backend endpoint.  The spawn gate trusts that list, so the picker should
    not erase a model it has on the next refresh. Rows added here take their
    display name, window and effort levels from the model's own backend
    entry — no per-model special cases.
    """
    try:
        from .codex_catalogue import load_codex_catalogue

        catalogue = load_codex_catalogue()
        entries, installed = catalogue.models, catalogue.version
    except Exception:  # noqa: BLE001
        entries, installed = {}, ""
    wins = _codex_cache_context_windows()
    merged = []
    for row in rows:
        r = dict(row)
        if not r.get("contextWindow"):
            # Backfill from the saved list when the live catalogue omits the
            # per-model window (older backend responses), or the context
            # meter falls back to the spec's 200K default.
            win = wins.get(str(r.get("id") or "").strip().lower())
            if win:
                r["contextWindow"] = win
        merged.append(r)
    seen = {str(row.get("id") or "").strip().lower() for row in merged}
    for key, entry in entries.items():
        model_id = str(entry.get("slug") or key)
        # Same filters as the live picker rows: helpers the backend marks
        # hidden (codex-auto-review) stay valid for the gate but are not
        # offered, nor is a model that needs a newer codex than the installed one.
        if key in seen or str(entry.get("visibility") or "list") != "list":
            continue
        min_cv = str(entry.get("minimal_client_version") or "").strip()
        if min_cv and installed and _codex_version_lt(installed, min_cv):
            continue
        row: dict[str, Any] = {
            "alias": model_id,
            "id": model_id,
            "display": str(entry.get("display_name") or model_id),
            "provider": "openai",
        }
        win = wins.get(key)
        if win:
            # List-only rows (newer than the live catalogue) still need a
            # per-model window, or the context meter falls back to the
            # spec's 200K default.
            row["contextWindow"] = win
        merged.append(_row_with_effort(row, entry))
        seen.add(key)
    return merged


def discover_openai_models(
    *,
    force: bool = False,
    fetch: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
    api_key: str | None = ...,  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Return the OpenAI/Codex model catalogue as picker rows.

    Resolution order (first non-empty wins, all memoised for :data:`_TTL_S`):

    1. Platform ``GET /v1/models`` (filtered to chat/codex ids) when an
       ``OPENAI_API_KEY`` resolves — either passed explicitly, from the env, or
       from ``~/.codex/auth.json``'s ``OPENAI_API_KEY`` (API-key login).
    2. The ChatGPT backend codex catalogue (``/backend-api/codex/models``)
       using the ``codex`` CLI's own ChatGPT OAuth token — the live source when
       codex is logged in via ChatGPT (no platform key). The OAuth token's
       audience is ``api.openai.com/v1`` but it lacks the ``api.model.read``
       scope the platform endpoint demands, so the ChatGPT backend is the only
       reachable live source for that login mode.
    3. The config fallback (``~/.sharedagents/config.json`` → ``external_models.openai``).

    Always non-empty. ``force=True`` bypasses the cache for an explicit refresh.
    """
    key_cache = "openai"
    try:
        if not force:
            cached = _cache_get(key_cache)
            if cached is not None:
                return cached

        auto = api_key is ...
        api = _openai_api_key() if auto else (api_key or "")
        rows: list[dict[str, Any]] = []
        if api:
            try:
                data = (fetch or _http_get_json)(
                    OPENAI_MODELS_URL,
                    {"Authorization": f"Bearer {api}"},
                )
                if data.get("data"):
                    # The platform rows are filtered through the installed
                    # codex's own list: make sure it is known first (asks codex
                    # once on a fresh install — an API-key login has no ChatGPT
                    # list to fetch). Only when there are rows to filter.
                    _codex_gate_lookup()
                for m in data.get("data") or []:
                    mid = str(m.get("id") or "").strip()
                    if mid and _is_codex_model_id(mid):
                        row: dict[str, Any] = {
                            "alias": mid,
                            "id": mid,
                            "display": mid,
                            "provider": "openai",
                        }
                        win = _codex_model_window(m)
                        if win > 0:
                            row["contextWindow"] = win
                        rows.append(_row_with_effort(row, m))
            except Exception as exc:  # noqa: BLE001
                _LOG.info("openai model discovery failed: %s", exc)
                rows = []
        if not rows and auto:
            # No platform key (auto-resolve) and platform path empty: try the
            # codex CLI's ChatGPT OAuth token against the ChatGPT backend catalogue.
            try:
                oauth_rows, attempted = _codex_chatgpt_rows(fetch or _http_get_json)
                rows = oauth_rows
                if not rows and attempted:
                    rows = _merge_codex_cli_cache(rows)
                    _cache_put_ttl(key_cache, rows, 60.0)
                    return [dict(r) for r in rows]
            except Exception as exc:  # noqa: BLE001
                _LOG.info("codex chatgpt discovery failed: %s", exc)
                rows = []
        rows = _merge_codex_cli_cache(rows)
        _cache_put(key_cache, rows)
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("discover_openai_models failed: %s", exc)
        return []


__all__ = ["discover_openai_models"]
