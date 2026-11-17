"""Provider-specific model discovery implementation."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._common import (
    _LOG,
    ANTHROPIC_MODELS_URL,
    CLAUDE_OAUTH_CLIENT_ID,
    CLAUDE_OAUTH_TOKEN_URL,
    _cache_get,
    _cache_put,
    _cache_put_ttl,
    _fallback_models,
    _http_get_json,
    _row_with_effort,
    _set_fallback_served,
)


def _claude_config_dir() -> Path:
    """Claude Code config dir, including its supported environment override."""
    env = (os.environ.get("CLAUDE_CONFIG_DIR") or "").strip()
    return Path(env) if env else (Path.home() / ".claude")


def _claude_credentials_path() -> Path:
    return _claude_config_dir() / ".credentials.json"


def _read_claude_oauth_record() -> dict[str, Any]:
    """The Claude Code OAuth record, or an empty dict when unavailable."""
    try:
        data = json.loads(_claude_credentials_path().read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth") or {}
        return dict(oauth) if isinstance(oauth, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _read_claude_oauth_token() -> str:
    """Reuse the token the ``claude`` CLI already stored, so discovery needs no
    separate key. Returns ``""`` when absent / unreadable.
    """
    return str(_read_claude_oauth_record().get("accessToken") or "")


def _claude_token_expired(record: dict[str, Any], skew_s: float = 60.0) -> bool:
    """Whether Claude Code's millisecond ``expiresAt`` has elapsed."""
    try:
        expires_at = float(record.get("expiresAt"))
    except (TypeError, ValueError):
        return False
    # Claude Code stores milliseconds. Accept seconds too so an alternate
    # credential writer cannot accidentally make a valid token look ancient.
    if expires_at > 100_000_000_000:
        expires_at /= 1000.0
    return time.time() >= (expires_at - skew_s)


def _http_post_json(
    url: str,
    payload: dict[str, str],
    timeout: float = 8.0,
) -> dict[str, Any]:
    """JSON POST returning JSON, used by Claude Code's OAuth refresh flow."""
    import httpx

    resp = httpx.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _claude_auth_writable() -> bool:
    """True only when a refreshed Claude token can be durably persisted."""
    try:
        path = _claude_credentials_path()
        return os.access(path if path.is_file() else path.parent, os.W_OK)
    except Exception:  # noqa: BLE001
        return False


def _persist_claude_tokens(
    access: str,
    refresh: str,
    expires_in: float,
) -> bool:
    """Atomically update only Claude Code's OAuth tokens in its credential file."""
    temp_name = ""
    try:
        path = _claude_credentials_path()
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        oauth = data.get("claudeAiOauth") or {}
        oauth["accessToken"] = access
        oauth["refreshToken"] = refresh
        oauth["expiresAt"] = int((time.time() + float(expires_in)) * 1000)
        data["claudeAiOauth"] = oauth
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp:
            temp_name = temp.name
            json.dump(data, temp, indent=2)
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)
        return True
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("claude token persist failed: %s", exc)
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        return False


def _refresh_claude_oauth_token(
    record: dict[str, Any],
    *,
    post: Callable[..., dict[str, Any]] | None = None,
) -> str | None:
    """Refresh and persist Claude Code's OAuth access token."""
    refresh = str(record.get("refreshToken") or "")
    if not refresh or not _claude_auth_writable():
        return None
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh,
        "client_id": (os.environ.get("CLAUDE_CODE_OAUTH_CLIENT_ID") or CLAUDE_OAUTH_CLIENT_ID),
    }
    try:
        data = (post or _http_post_json)(CLAUDE_OAUTH_TOKEN_URL, payload)
    except Exception as exc:  # noqa: BLE001
        _LOG.info("claude oauth refresh failed: %s", exc)
        return None
    access = str(data.get("access_token") or "")
    new_refresh = str(data.get("refresh_token") or refresh)
    try:
        expires_in = float(data.get("expires_in") or 3600)
    except (TypeError, ValueError):
        expires_in = 3600.0
    if not access or not _persist_claude_tokens(access, new_refresh, expires_in):
        return None
    return access


def _anthropic_headers_with_state() -> tuple[dict[str, str] | None, bool]:
    """Resolve Anthropic auth and report whether OAuth refresh failed."""
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if key:
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}, False
    record = _read_claude_oauth_record()
    tok = str(record.get("accessToken") or "")
    needs_refresh = (not tok and bool(record.get("refreshToken"))) or (bool(tok) and _claude_token_expired(record))
    if needs_refresh:
        tok = _refresh_claude_oauth_token(record) or ""
        if not tok:
            return None, True
    if tok:
        return {
            "Authorization": f"Bearer {tok}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "oauth-2025-04-20",
        }, False
    return None, False


def _anthropic_headers() -> dict[str, str] | None:
    return _anthropic_headers_with_state()[0]


def _anthropic_alias(model_id: str) -> str:
    """``claude-opus-4-8`` -> ``opus-4-8`` (unique, readable). Falls back to the
    full id."""
    s = model_id
    if s.startswith("claude-"):
        s = s[len("claude-") :]
    return s or model_id


def _parse_anthropic_page(page: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in page.get("data") or []:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        row: dict[str, Any] = {
            "alias": _anthropic_alias(mid),
            "id": mid,
            "display": str(m.get("display_name") or mid),
            "provider": "anthropic",
        }
        try:
            win = int(m.get("max_input_tokens") or 0)
        except (TypeError, ValueError):
            win = 0
        if win > 0:
            # The session context meter and the picker badge read
            # ``contextWindow``; without it every Claude model fell back to
            # the claude engine's flat 200K default (opus-5-5 is 1M).
            row["contextWindow"] = win
        rows.append(_row_with_effort(row, m))
    return rows


def _list_anthropic(
    headers: dict[str, str],
    fetch: Callable[[str, dict[str, str]], dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    url = f"{ANTHROPIC_MODELS_URL}?limit=100"
    pages = 0
    while True:
        page = fetch(url, headers)
        out.extend(_parse_anthropic_page(page))
        if not page.get("has_more"):
            break
        last = str(page.get("last_id") or "")
        if not last:
            break
        url = f"{ANTHROPIC_MODELS_URL}?limit=100&after_id={last}"
        pages += 1
        if pages > 20:  # defensive: never loop forever on a misbehaving page
            break
    return out


def discover_anthropic_models(
    *,
    force: bool = False,
    fetch: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
    headers: dict[str, str] | None = ...,  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Return the Anthropic model catalogue as picker rows.

    Live ``GET /v1/models`` when a credential resolves, else the static
    fallback (``external_models.anthropic`` ∪ the shipped catalogue). Always
    non-empty. A live list is memoised for :data:`_TTL_S`; the fallback only
    for 60 s, so one slow or failed fetch does not hide the live list (and
    any model the fallback lacks) for an hour.

    ``headers`` defaults to the auto-resolved credential; pass ``None``
    explicitly to force the no-auth path (tests), or a dict to inject one.
    ``fetch`` overrides the HTTP getter (tests).
    """
    try:
        key = "anthropic"
        if not force:
            cached = _cache_get(key)
            if cached is not None:
                return cached

        hdrs = _anthropic_headers_with_state()[0] if headers is ... else headers
        rows: list[dict[str, Any]] = []
        if hdrs:
            try:
                rows = _list_anthropic(hdrs, fetch or _http_get_json)
            except Exception as exc:  # noqa: BLE001
                _LOG.info("anthropic model discovery failed: %s", exc)
                rows = []
        if rows:
            _set_fallback_served(key, False)
            _cache_put(key, rows)
        else:
            # No credential, a failed OAuth refresh or a failed fetch. An
            # empty list made the claude engine pass a picker alias
            # (``opus-5-5``) to ``claude --model`` verbatim, which the CLI
            # rejects as an unknown model.
            rows = _fallback_models(key)
            _set_fallback_served(key, True)
            _cache_put_ttl(key, rows, 60.0)
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("discover_anthropic_models failed: %s", exc)
        _set_fallback_served("anthropic", True)
        return _fallback_models("anthropic")


__all__ = [
    "_claude_config_dir",
    "_claude_credentials_path",
    "_read_claude_oauth_record",
    "_read_claude_oauth_token",
    "_claude_token_expired",
    "_http_post_json",
    "_claude_auth_writable",
    "_persist_claude_tokens",
    "_refresh_claude_oauth_token",
    "_anthropic_headers_with_state",
    "_anthropic_headers",
    "_anthropic_alias",
    "_parse_anthropic_page",
    "_list_anthropic",
    "discover_anthropic_models",
]
