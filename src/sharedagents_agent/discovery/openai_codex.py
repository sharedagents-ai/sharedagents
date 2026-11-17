"""Provider-specific model discovery implementation."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._common import (
    _LOG,
    CODEX_OAUTH_AUTH,
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    CODEX_RESPONSES_API,
    CODEX_RESPONSES_CLIENT_VERSION_FALLBACK,
    CODEX_TRANSPORT_ALIAS_PREFIX,
    _cache_get,
    _cache_put,
    _cache_put_ttl,
    _http_get_json,
    _is_codex_model_id,
    _row_with_effort,
)


def _openai_api_key() -> str:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if key:
        return key
    try:
        data = json.loads((_codex_home() / "auth.json").read_text(encoding="utf-8"))
        return str(data.get("OPENAI_API_KEY") or "")
    except Exception:  # noqa: BLE001
        return ""


def _codex_home() -> Path:
    """Codex CLI home dir: ``$CODEX_HOME`` when set, else ``~/.codex``. Mirrors
    the codex CLI's own resolution so discovery reads the SAME credential store
    the CLI uses (not a hard-coded ``~/.codex``).
    """
    env = (os.environ.get("CODEX_HOME") or "").strip()
    return Path(env) if env else (Path.home() / ".codex")


def _codex_parse_toml_subset(text: str) -> dict[str, Any]:
    """Minimal TOML subset parser for codex ``config.toml``: top-level
    ``key = "value"``, ``[section]`` and ``[a.b]`` nested tables, string / bool
    / int values, and ``#`` comments. Used so discovery reads the SAME config
    the codex CLI does on Python 3.10 (which has no ``tomllib``) WITHOUT adding
    a dependency. Good enough for the credential-store / base-url / profile
    keys we need; not a general TOML parser.
    """
    root: dict[str, Any] = {}
    cur: dict[str, Any] = root
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            cur = root
            for part in line[1:-1].strip().split("."):
                part = part.strip().strip('"').strip("'")
                cur = cur.setdefault(part, {})
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip().strip('"').strip("'")
            v = v.strip()
            if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
                v = v[1:-1]
            elif v.lower() in ("true", "false"):
                v = v.lower() == "true"
            else:
                try:
                    v = int(v)
                except ValueError:
                    pass  # leave raw string (arrays etc. not needed here)
            cur[k] = v
    return root


def _codex_config(home: Path) -> dict[str, Any]:
    """Parsed codex ``config.toml`` as a dict (``tomllib`` on 3.11+, else the
    subset parser on 3.10). ``{}`` on any read/parse failure — callers fall back
    to defaults, never crash.
    """
    try:
        text = (home / "config.toml").read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return {}
    try:
        import tomllib  # Python 3.11+

        return tomllib.loads(text)
    except ImportError:
        return _codex_parse_toml_subset(text)
    except Exception:  # noqa: BLE001
        return _codex_parse_toml_subset(text)


def _codex_config_profile(data: dict[str, Any]) -> dict[str, Any]:
    """The effective ``[profiles.<profile>]`` table when codex's top-level
    ``profile = "name"`` is set, else ``{}``. Codex lets a profile override
    base-url and other settings, so discovery must apply the SAME profile the
    CLI runs under.
    """
    name = str(data.get("profile") or "").strip()
    if not name:
        return {}
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        return {}
    prof = profiles.get(name)
    return prof if isinstance(prof, dict) else {}


def _codex_credentials_store(home: Path) -> str:
    """Resolve codex's credential-store mode from ``$CODEX_HOME/config.toml``.

    Codex's real key is the top-level ``cli_auth_credentials_store`` (values
    ``file`` | ``keyring`` | ``auto``); older builds used
    ``auth_credentials_store`` or ``[auth] credentials_store``. We return a
    normalised mode:

    * ``"keyring"`` — codex relocates the credential to the OS keyring and
      REMOVES ``auth.json``. sharedagents discovery does NOT reverse-engineer the
      keyring record format (service ``"Codex Auth"``, hashed account, a
      serialized whole-record blob) — that's fragile and codex-owned. So the
      caller treats keyring as "no live token here" and falls back (short-TTL),
      rather than reading a removed ``auth.json`` or a wrong raw entry.
    * ``"file"`` — the default (incl. ``auto``: codex falls back to file when
      keyring is unavailable, and ``auth.json`` is typically present).
    """
    data = _codex_config(home)
    raw = ""
    for k in ("cli_auth_credentials_store", "auth_credentials_store"):
        raw = str(data.get(k) or "").strip()
        if raw:
            break
    if not raw:
        auth = data.get("auth") or {}
        raw = str(auth.get("credentials_store") or "").strip()
    mode = raw.lower()
    if mode == "keyring":
        return "keyring"
    return "file"  # file | auto | unknown -> file


def _codex_chatgpt_base_url() -> str:
    """Codex's effective ChatGPT base URL: the ``chatgpt_base_url`` from the
    active ``[profiles.<profile>]`` table (when ``profile = "name"`` is set),
    falling back to the top-level value, defaulting to ``https://chatgpt.com``.
    For proxy / staging / alternate ChatGPT environments. Trailing slash
    stripped. The OAuth token's audience is the same environment the CLI
    targets, so discovery must hit the SAME base — production otherwise 401s /
    falls back.
    """
    data = _codex_config(_codex_home())
    base = ""
    prof = _codex_config_profile(data)
    if prof:
        base = str(prof.get("chatgpt_base_url") or "").strip()
    if not base:
        base = str(data.get("chatgpt_base_url") or "").strip()
    return base.rstrip("/") if base else "https://chatgpt.com"


def _codex_chatgpt_models_url(base: str) -> str:
    """Compose the catalogue URL from a resolved ChatGPT base, appending only
    the codex-specific suffix appropriate for that base: codex's documented
    ``chatgpt_base_url`` may already include ``/backend-api`` (e.g. a staging
    base of ``https://chatgpt-staging.com/backend-api``), in which case we
    append just ``/codex/models`` — never a doubled ``/backend-api``.
    """
    b = base.rstrip("/")
    if b.endswith("/backend-api"):
        return f"{b}/codex/models"
    return f"{b}/backend-api/codex/models"


def codex_responses_base_url() -> str:
    """Base URL the local-LLM worker POSTs ``/responses`` against.

    Same ChatGPT environment as catalogue discovery (honors
    ``chatgpt_base_url``), with the Codex suffix and no trailing slash.
    """
    base = _codex_chatgpt_base_url()
    if base.endswith("/backend-api"):
        return f"{base}/codex"
    return f"{base}/backend-api/codex"


def is_codex_responses_endpoint(url: str) -> bool:
    """True for the ChatGPT Codex Responses base URL."""
    u = (url or "").rstrip("/").lower()
    if not u:
        return False
    return u.endswith("/backend-api/codex") or u.endswith("/backend-api/codex/v1")


def _codex_fedramp() -> bool:
    """True when the codex ID token's ``chatgpt_account_is_fedramp`` claim is
    set — codex then routes backend requests via FedRAMP and adds
    ``X-OpenAI-Fedramp: true``; discovery mirrors that so the catalogue request
    isn't mis-routed/rejected. ``False`` on any miss (file store only; under the
    keyring store discovery skips anyway).
    """
    try:
        data = json.loads((_codex_home() / "auth.json").read_text(encoding="utf-8"))
        id_token = str((data.get("tokens") or {}).get("id_token") or "")
        return bool(_codex_jwt_claim(id_token, "chatgpt_account_is_fedramp"))
    except Exception:  # noqa: BLE001
        return False


def _read_codex_oauth_token() -> tuple[str, str]:
    """Reuse the ``codex`` CLI's ChatGPT OAuth token so codex model discovery
    needs no separate platform ``OPENAI_API_KEY``.

    Resolves the token from ``$CODEX_HOME/auth.json`` for the ``file`` store
    (codex's default, incl. ``auto``). For the ``keyring`` store codex removes
    ``auth.json`` and owns the record format, so we return ``("", account_id)``
    — discovery falls back (short-TTL) rather than reading a removed file or
    guessing the keyring blob. Returns ``("", "")`` when absent / unreadable
    (e.g. codex logged in via API key, not ChatGPT).
    """
    home = _codex_home()
    if _codex_credentials_store(home) == "keyring":
        _LOG.debug(
            "codex credential store is keyring; discovery falls back "
            "(set cli_auth_credentials_store=file or OPENAI_API_KEY)"
        )
        return "", ""
    try:
        data = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return "", ""
    tok = data.get("tokens") or {}
    return (
        str(tok.get("access_token") or ""),
        str(tok.get("account_id") or data.get("account_id") or ""),
    )


def _read_codex_refresh_token() -> str:
    """The codex OAuth ``refresh_token`` from ``$CODEX_HOME/auth.json`` (it
    stays in the file even under the ``keyring`` store, which only relocates
    the access token). ``""`` when absent."""
    try:
        data = json.loads((_codex_home() / "auth.json").read_text(encoding="utf-8"))
        return str((data.get("tokens") or {}).get("refresh_token") or "")
    except Exception:  # noqa: BLE001
        return ""


def _codex_jwt_claim(token: str, name: str) -> Any:
    """A single claim from a JWT's payload, or ``None`` when the token isn't a
    decodable JWT or the claim is absent."""
    if not token or token.count(".") < 2:
        return None
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims.get(name)
    except Exception:  # noqa: BLE001
        return None


def _codex_jwt_exp(token: str) -> float | None:
    """``exp`` claim (unix seconds) of a JWT, or ``None`` when the token isn't
    a decodable JWT or the claim is non-numeric (a corrupt / externally supplied
    token) — never raises, so a malformed stored token degrades to the fallback
    instead of crashing the model picker."""
    exp = _codex_jwt_claim(token, "exp")
    if exp is None:
        return None
    try:
        return float(exp)
    except (TypeError, ValueError):
        return None


def _codex_token_expired(token: str, skew_s: float = 60.0) -> bool:
    """True when the JWT ``exp`` is past (with a small clock-skew buffer). False
    when the token isn't a JWT (let the backend 401 decide) or still valid."""
    exp = _codex_jwt_exp(token)
    if exp is None:
        return False
    return time.time() >= (exp - skew_s)


def _http_post_form(
    url: str,
    form: dict[str, str],
    timeout: float = 8.0,
) -> dict[str, Any]:
    """URL-encoded POST returning JSON. Used for the OAuth token refresh."""
    import httpx

    resp = httpx.post(url, data=form, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _codex_auth_writable() -> bool:
    """True when the codex ``file`` credential store is durably writable — i.e.
    we can persist a refreshed (rotated) token back to ``auth.json``. We refuse
    to refresh otherwise: a refresh rotates the refresh_token server-side, and
    if we can't save the new one the stored login is invalidated and the user
    must re-``codex login``. Keyring store isn't supported for refresh, so it's
    treated as not writable (refresh is skipped, discovery falls back).
    """
    try:
        if _codex_credentials_store(_codex_home()) == "keyring":
            return False
        p = _codex_home() / "auth.json"
        if p.is_file():
            return os.access(p, os.W_OK)
        return os.access(p.parent, os.W_OK)
    except Exception:  # noqa: BLE001
        return False


def _persist_codex_tokens(
    access: str,
    refresh: str,
    id_token: str,
    account_id: str,
) -> bool:
    """Write the refreshed tokens back to ``$CODEX_HOME/auth.json`` so the
    codex CLI itself picks them up. Returns ``True`` on durable success,
    ``False`` on any failure (never raises) — the caller treats ``False`` as a
    refresh FAILURE so it doesn't report success with an unpersisted (and
    server-side rotated) token that would invalidate the stored login.
    """
    try:
        home = _codex_home()
        p = home / "auth.json"
        data = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        tok = data.get("tokens") or {}
        tok["access_token"] = access
        tok["refresh_token"] = refresh
        if id_token:
            tok["id_token"] = id_token
        if account_id:
            tok["account_id"] = account_id
        data["tokens"] = tok
        data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("codex token persist failed: %s", exc)
        return False


def _refresh_codex_oauth_token(
    refresh_token: str,
    account_id: str,
    *,
    post: Callable[..., dict[str, Any]] | None = None,
) -> tuple[str, str, str] | None:
    """Exchange the codex OAuth ``refresh_token`` for a fresh access token.

    Returns ``(access_token, refresh_token, id_token)`` on success — the new
    tokens are durably persisted to ``auth.json`` FIRST (a refresh rotates the
    refresh_token server-side, so we never report success unless we can save
    the new one; otherwise the stored login would be invalidated). Returns
    ``None`` on any failure (incl. a non-writable store or persist failure) so
    the caller degrades to the config fallback. Mirrors the codex CLI's own
    auth manager so discovery recovers an expired login instead of caching a
    stale list for an hour.
    """
    if not refresh_token:
        return None
    # Pre-flight: refuse to rotate a token we can't durably persist.
    if not _codex_auth_writable():
        _LOG.info("codex oauth refresh skipped: credential store not writable")
        return None
    # Honor codex's staging / alternate-authority overrides so a login made
    # against a non-production authority refreshes against the SAME authority
    # (else the production endpoint rejects the refresh token and discovery
    # stops working once the access token expires, even though codex itself
    # can refresh). Mirrors the codex CLI's env handling.
    client_id = os.environ.get("CODEX_APP_SERVER_LOGIN_CLIENT_ID") or CODEX_OAUTH_CLIENT_ID
    token_url = os.environ.get("CODEX_REFRESH_TOKEN_URL_OVERRIDE") or CODEX_OAUTH_TOKEN_URL
    form = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "scope": "openid profile email offline_access",
    }
    try:
        data = (post or _http_post_form)(token_url, form)
    except Exception as exc:  # noqa: BLE001
        _LOG.info("codex oauth refresh failed: %s", exc)
        return None
    access = str(data.get("access_token") or "")
    if not access:
        return None
    new_rt = str(data.get("refresh_token") or refresh_token)
    new_id = str(data.get("id_token") or "")
    if not _persist_codex_tokens(access, new_rt, new_id, account_id):
        # Refresh succeeded server-side (token rotated) but we couldn't save it.
        # Report failure so we fall back; the stored refresh_token is now stale
        # and the user will need to re-login — but we don't silently use an
        # unpersisted token and we don't cache a fake success.
        _LOG.warning(
            "codex refresh succeeded but persist failed; stored login may be invalid — re-run `codex login` if needed"
        )
        return None
    return access, new_rt, new_id


def _codex_cli_version() -> str:
    """Version of the installed ``codex`` CLI (e.g. ``0.155.1``), exactly as
    codex reports it and sends as ``client_version``. Re-read after an
    upgrade: a process-lifetime memo made long-running sharedagents services ask
    the backend for the OLD version's model list
    (:func:`.codex_catalogue.installed_codex_version`).
    """
    from .codex_catalogue import installed_codex_version

    return installed_codex_version()


def _codex_version_lt(a: str, b: str) -> bool:
    """``True`` if semver ``a`` < ``b``; tolerant of unequal segment counts."""

    def parts(s: str) -> list[int]:
        out: list[int] = []
        for p in (s or "").split("."):
            digits = "".join(ch for ch in p if ch.isdigit())
            out.append(int(digits) if digits else 0)
        return out

    return parts(a) < parts(b)


def _codex_model_window(m: dict[str, Any]) -> int:
    """Max context window (tokens) the ChatGPT backend publishes for ``m``.

    ``max_context_window`` (the absolute max) wins over ``context_window``
    (the model's default window) — the picker badge and the session context
    meter both want the MAX (e.g. gpt-5.6-sol: 272K default / 872K max).
    0 when the entry publishes neither.
    """
    try:
        cw = int(m.get("context_window") or 0)
    except (TypeError, ValueError):
        cw = 0
    try:
        mcw = int(m.get("max_context_window") or 0)
    except (TypeError, ValueError):
        mcw = 0
    return max(cw, mcw)


def _list_codex_chatgpt_models(
    fetch: Callable[[str, dict[str, str]], dict[str, Any]],
    access_token: str,
    account_id: str,
    client_version: str,
    *,
    cli_gate: bool = True,
) -> list[dict[str, Any]]:
    """Live codex catalogue from the ChatGPT backend, reusing the ``codex``
    CLI's OAuth token. Returns picker rows; empty on any failure.

    ``visibility == "list"`` keeps user-pickable models (drops hidden helpers
    like ``codex-auto-review``); models whose ``minimal_client_version`` exceeds
    the installed CLI are dropped so the picker never offers a model the local
    codex can't drive.

    Rows are ALSO filtered through
    ``_is_codex_model_id`` — the SAME source the spawn gate uses — so the
    picker can never offer a slug that ``validate_codex_model`` would refuse.
    A list fetched as the INSTALLED codex's version is saved first
    (:func:`.codex_catalogue.remember_live_models`), so the gate in every
    sharedagents process agrees with what the picker just showed, even while the
    shared ``models_cache.json`` holds another codex version's list.
    """
    if not access_token:
        return []
    cv = (client_version or _codex_cli_version() or "").strip()
    if not cv:
        return []
    headers = {
        "Authorization": f"Bearer {access_token}",
        "ChatGPT-Account-Id": account_id or "",
        "OpenAI-Beta": "responses-2024-06-01",
        "Accept": "application/json",
    }
    if _codex_fedramp():
        # Mirror the codex CLI: route the backend request via FedRAMP so the
        # catalogue isn't mis-routed/rejected for FedRAMP accounts.
        headers["X-OpenAI-Fedramp"] = "true"
    url = f"{_codex_chatgpt_models_url(_codex_chatgpt_base_url())}?client_version={urllib.parse.quote(cv, safe='')}"
    data = fetch(url, headers)
    try:
        from .codex_catalogue import remember_live_models

        remember_live_models(cv, data)
    except Exception as exc:  # noqa: BLE001
        _LOG.info("could not save the codex model list: %s", exc)
    rows: list[dict[str, Any]] = []
    for m in data.get("models") or []:
        if str(m.get("visibility") or "") != "list":
            continue
        slug = str(m.get("slug") or "").strip()
        if not slug:
            continue
        min_cv = str(m.get("minimal_client_version") or "").strip()
        if cli_gate and min_cv and _codex_version_lt(cv, min_cv):
            continue
        # CLI spawn gate: a row the gate would refuse must never be offered
        # by the Codex *agent* picker. Local-LLM Responses transport skips
        # this — it does not launch ``codex exec``.
        if cli_gate and not _is_codex_model_id(slug):
            continue
        row: dict[str, Any] = {
            "alias": slug,
            "id": slug,
            "display": str(m.get("display_name") or slug),
            "provider": "openai",
        }
        win = _codex_model_window(m)
        if win > 0:
            # The session context meter and the picker badge read
            # ``contextWindow``; without it every codex model fell back to
            # the spec's 200K default regardless of its real window.
            row["contextWindow"] = win
        rows.append(_row_with_effort(row, m))
    return rows


def _codex_chatgpt_rows(
    fetch: Callable[[str, dict[str, str]], dict[str, Any]],
    *,
    refresh_post: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Live codex catalogue via the ChatGPT backend, with OAuth refresh.

    Returns ``(rows, attempted)``: ``rows`` is the live list (possibly empty on
    failure), and ``attempted`` is True when an OAuth credential was present so
    the caller knows to short-TTL-cache the fallback (re-try soon) rather than
    serve a stale list for the full hour.

    Refreshes the access token proactively (decoded JWT ``exp`` past) and
    reactively (backend 401) using the stored refresh_token, mirroring the
    codex CLI's own auth manager. Best-effort: any refresh failure degrades to
    an empty list, never raises.
    """
    access, account = _read_codex_oauth_token()
    if not access:
        return [], False

    def _maybe_refresh() -> str:
        rt = _read_codex_refresh_token()
        refreshed = _refresh_codex_oauth_token(rt, account, post=refresh_post)
        return refreshed[0] if refreshed else ""

    if _codex_token_expired(access):
        access = _maybe_refresh()
        if not access:
            return [], True

    try:
        return _list_codex_chatgpt_models(fetch, access, account, ""), True
    except Exception as exc:  # noqa: BLE001
        # Reactive 401: the backend rejected a token that looked valid locally
        # (clock skew / server-side revocation). Refresh once and retry.
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
        if status == 401:
            access = _maybe_refresh()
            if not access:
                return [], True
            try:
                return _list_codex_chatgpt_models(fetch, access, account, ""), True
            except Exception:  # noqa: BLE001
                return [], True
        _LOG.info("codex chatgpt model discovery failed: %s", exc)
        return [], True


def _codex_oauth_access(
    *,
    refresh_post: Callable[..., dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Return ``(access_token, account_id)``, refreshing when the JWT is past.

    Empty access token means no usable ChatGPT login. Never raises.
    """
    access, account = _read_codex_oauth_token()
    if not access:
        return "", account
    if not _codex_token_expired(access):
        return access, account
    rt = _read_codex_refresh_token()
    refreshed = _refresh_codex_oauth_token(rt, account, post=refresh_post)
    if not refreshed:
        return "", account
    return refreshed[0], account


def codex_oauth_request_headers(
    *,
    refresh_post: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Bearer + ChatGPT account headers for the Codex Responses backend.

    Empty dict when no ChatGPT OAuth login is available.
    """
    access, account = _codex_oauth_access(refresh_post=refresh_post)
    if not access:
        return {}
    headers = {
        "Authorization": f"Bearer {access}",
        "OpenAI-Beta": "responses-2024-06-01",
        "originator": "sharedagents",
        "User-Agent": "sharedagents",
    }
    if account:
        headers["ChatGPT-Account-Id"] = account
    if _codex_fedramp():
        headers["X-OpenAI-Fedramp"] = "true"
    return headers


def discover_codex_oauth_transport_models(
    *,
    force: bool = False,
    fetch: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
    refresh_post: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """ChatGPT-OAuth Codex models as local-LLM deployment rows.

    Unlike :func:`discover_openai_models` (Codex *CLI* picker), this list is
    for jarvis / build / plan: each row points at the Responses backend and
    does not require ``codex exec``. Empty when no OAuth login is present.
    Cached under a distinct key so the CLI catalogue is unaffected.
    """
    key_cache = "codex-oauth-transport"
    if not force:
        cached = _cache_get(key_cache)
        if cached is not None:
            return [dict(r) for r in cached]
    access, account = _codex_oauth_access(refresh_post=refresh_post)
    if not access:
        # A login can appear after this process starts. Do not hide it for the
        # normal one-hour positive-cache lifetime.
        _cache_put_ttl(key_cache, [], 60.0)
        return []
    cv = (_codex_cli_version() or "").strip() or CODEX_RESPONSES_CLIENT_VERSION_FALLBACK
    try:
        raw = _list_codex_chatgpt_models(
            fetch or _http_get_json,
            access,
            account,
            cv,
            cli_gate=False,
        )
    except Exception as exc:  # noqa: BLE001
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
        if status == 401:
            rt = _read_codex_refresh_token()
            refreshed = _refresh_codex_oauth_token(
                rt,
                account,
                post=refresh_post,
            )
            if refreshed:
                try:
                    raw = _list_codex_chatgpt_models(
                        fetch or _http_get_json,
                        refreshed[0],
                        account,
                        cv,
                        cli_gate=False,
                    )
                except Exception:  # noqa: BLE001
                    raw = []
            else:
                raw = []
        else:
            _LOG.info("codex oauth transport discovery failed: %s", exc)
            raw = []
    base = codex_responses_base_url()
    rows: list[dict[str, Any]] = []
    for r in raw:
        slug = str(r.get("id") or "").strip()
        if not slug:
            continue
        display = str(r.get("display") or slug)
        if "(Codex)" not in display:
            display = f"{display} (Codex)"
        row = dict(r)
        row.update(
            {
                "alias": f"{CODEX_TRANSPORT_ALIAS_PREFIX}{slug}",
                "id": slug,
                "display": display,
                "baseUrl": base,
                "backend": "openai",
                "api": CODEX_RESPONSES_API,
                "auth": CODEX_OAUTH_AUTH,
                "provider": "Codex",
            }
        )
        rows.append(row)
    # A transient catalogue error also produces no rows. Retry promptly so a
    # single failed fetch cannot remove Codex from the picker for an hour.
    if rows:
        _cache_put(key_cache, rows)
    else:
        _cache_put_ttl(key_cache, [], 60.0)
    return [dict(r) for r in rows]


def merge_codex_oauth_transport_rows(
    rows: list[dict[str, Any]],
    *,
    extra: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Append Codex OAuth Responses rows, skipping alias collisions."""
    out = [dict(r) for r in rows if isinstance(r, dict)]
    if extra is None:
        extra = discover_codex_oauth_transport_models()
    seen_alias = {str(r.get("alias") or "") for r in out if r.get("alias")}
    for r in extra or []:
        if not isinstance(r, dict):
            continue
        alias = str(r.get("alias") or "")
        if alias and alias in seen_alias:
            continue
        out.append(dict(r))
        if alias:
            seen_alias.add(alias)
    return out


def sync_codex_oauth_models_into_cfg(
    cfg: Any,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Fold Codex OAuth transport rows into ``cfg.models`` once.

    Runs at config load / ``reload_models`` (``force=True``) / explicit
    refresh — not on every picker open, inherit, or worker lookup.
    Subsequent calls are a no-op until ``force=True``. Empty discovery
    (no login) still marks the cfg synced so we do not re-hit ChatGPT.

    Does **not** rewrite ``~/.sharedagents/config.json``.
    """
    if cfg is None:
        return []
    existing = list(getattr(cfg, "models", None) or [])
    engine = str(getattr(cfg, "engine", "") or "local_llm").strip()
    if engine not in ("", "local_llm"):
        return existing
    if not force and getattr(cfg, "_codex_oauth_models_synced", False):
        return existing
    try:
        extra = discover_codex_oauth_transport_models(force=force)
    except Exception:  # noqa: BLE001
        extra = []
    try:
        cfg._codex_oauth_models_synced = True
    except Exception:  # noqa: BLE001
        pass
    if not extra:
        return existing
    merged = merge_codex_oauth_transport_rows(existing, extra=extra)
    try:
        cfg.models = merged
    except Exception:  # noqa: BLE001
        return merged
    return merged


def selectable_local_models(cfg: Any) -> list[dict[str, Any]]:
    """Return runtime ``cfg.models`` (Codex rows already folded at load)."""
    return [dict(r) for r in (getattr(cfg, "models", None) or []) if isinstance(r, dict)]


__all__ = [
    "_openai_api_key",
    "_codex_home",
    "_codex_parse_toml_subset",
    "_codex_config",
    "_codex_config_profile",
    "_codex_credentials_store",
    "_codex_chatgpt_base_url",
    "_codex_chatgpt_models_url",
    "codex_responses_base_url",
    "is_codex_responses_endpoint",
    "_codex_fedramp",
    "_read_codex_oauth_token",
    "_read_codex_refresh_token",
    "_codex_jwt_claim",
    "_codex_jwt_exp",
    "_codex_token_expired",
    "_http_post_form",
    "_codex_auth_writable",
    "_persist_codex_tokens",
    "_refresh_codex_oauth_token",
    "_codex_cli_version",
    "_codex_version_lt",
    "_list_codex_chatgpt_models",
    "_codex_model_window",
    "_codex_chatgpt_rows",
    "_codex_oauth_access",
    "codex_oauth_request_headers",
    "discover_codex_oauth_transport_models",
    "merge_codex_oauth_transport_rows",
    "sync_codex_oauth_models_into_cfg",
    "selectable_local_models",
]
