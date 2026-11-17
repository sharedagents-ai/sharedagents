"""
sharedagents-web — lightweight web chat server for sharedagents agents.

Serves:
  /              — the chat UI (index.html)
  /api/models    — list available agents (proxied from gateway)
  /api/chat      — chat endpoint with SSE streaming
  /api/sessions  — list/create sessions
  /api/verify    — verify dashboard JWT token

Architecture:
  - Proxies chat requests to the sharedagents gateway's OpenAI-compatible
    endpoint at http://127.0.0.1:8100/openai/v1/
  - Manages conversation sessions in-memory (maps to gateway sessions)
  - Auth via dashboard JWT cookie (forwarded to /verify-token)
"""

import gzip
import json
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

# When installed as part of the `sharedagents` wheel, the sibling packages
# (sharedagents_core, sharedagents_telegram, …) are already importable — this block
# is a no-op. It only matters for an editable/source checkout where this
# file lives at ``<repo>/src/sharedagents_web/server.py`` and something launches
# it directly without ``src`` on the path: fall back to inserting the
# repo's ``src`` dir, but ONLY if it actually contains the packages (so we
# never shadow an installed copy with an unrelated site-packages dir).
_REPO_SRC = Path(__file__).resolve().parents[1]
if (_REPO_SRC / "sharedagents_core").is_dir():
    _repo_src_s = str(_REPO_SRC)
    if _repo_src_s not in sys.path:
        sys.path.insert(0, _repo_src_s)

# Legacy file-path launches must import exactly one canonical server module.
# Delegate before app/route initialization, preserving argv and secret setup;
# setting __package__ alone would still import a second server via blueprints.
if __name__ == "__main__" and not __package__:
    from sharedagents_web.cli import main as _package_main

    _package_main()
    raise SystemExit(0)

import requests  # noqa: E402
from flask import Flask, Response, g, jsonify, redirect, request, send_from_directory  # noqa: E402

from sharedagents_core import oscompat  # noqa: E402
from sharedagents_core.layout import sessions_root, sharedagents_root  # noqa: E402
from sharedagents_telegram.forum_topics import (  # noqa: E402
    all_canonical_names as all_canonical_names,
    load_forum_topics,
    load_session_topic_bindings,
    resolve_session_topic,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GATEWAY_URL = os.environ.get("SHAREDAGENTS_GATEWAY_URL", "http://127.0.0.1:8100")
# White-label: the product name shown in every page's <title> and the main
# heading. Set ``SHAREDAGENTS_WEB_TITLE`` to rebrand the UI without touching code
# (e.g. SHAREDAGENTS_WEB_TITLE="Acme Assistant"). Defaults to "sharedagents". Read once
# at import so a process restart picks up the new value; injected into every
# template's context via the ``site_title`` context processor below.
SITE_TITLE = os.environ.get("SHAREDAGENTS_WEB_TITLE") or "sharedagents"
# Default agent for new/unattributed web sessions. `jarvis` is the user-facing
# SDLC orchestrator — it superseded `main` as the DEFAULT
# on 2026-07-09. `main` remains a MOUNTED gateway fallback (so pre-existing
# main-bound sessions still route) and is now ALSO shown in the picker like
# every other configured agent; hence the fallbacks below read old
# rows' stored agent and only default NEW work to jarvis. Referenced instead of
# a scattered "main" literal so the default has a
# single source of truth.
DEFAULT_AGENT = "jarvis"
# Read timeout for the proxied live-conversation SSE. Must exceed the gateway
# stream's bounded lifetime (_SESSION_STREAM_MAX_SECONDS=600s) + heartbeat
# margin, else requests would tear down a healthy idle stream mid-heartbeat.
_SESSION_STREAM_READ_TIMEOUT = 720
# /api/chat decoupled-turn bounds. Worker sits just over Telegram's 540s
# no-progress watchdog (Antigravity print 480 + 60s buffer). The client
# SSE generator outlives the worker's terminal/sentinel yet cannot spin
# forever if those are lost.
_CHAT_WORKER_MAX_SECONDS = 550
# Generator backstop: worker 550s + one final 120s upstream read.
_CHAT_STREAM_MAX_SECONDS = 670
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:5000")
# Path to the SQLite database backing the web session store.
# Replaces the old per-owner file-per-session layout at SESSION_DIR/<owner>/<sid>.json.
# Single WAL-mode database, indexed by (owner, updated_at) for fast per-user listing
# and LRU eviction. Legacy files at ~/.sharedagents/sessions/<owner>/*.json are migrated
# automatically on first init — see SessionStore._migrate_from_files().
# Derive the web-state dir from the runtime-tree root (SHAREDAGENTS_HOME-aware via
# ``sharedagents_root()``) rather than a hardcoded ``~/.sharedagents`` literal — otherwise
# an isolated instance (e.g. the uxsim dev web, SHAREDAGENTS_HOME=~/.sharedagents-usersim)
# silently shares the PROD web-state DB and can clobber prod's active-gateway /
# auth rows. Explicit SHAREDAGENTS_WEB_DB still overrides.
SESSION_DB_PATH = Path(
    os.environ.get("SHAREDAGENTS_WEB_DB") or (sharedagents_root() / "web" / "sessions.db")
).expanduser()

# Per-session artifact store: images/PDFs an agent emits during a turn are
# written here (under <ARTIFACTS_DIR>/<session_hash>/<name>) and served
# same-origin via GET /files/<sid>/<name>. This lets the web UI render agent
# OUTPUT images inline and link/preview PDFs in its own UI — no Telegram detour.
ARTIFACTS_DIR = Path(
    os.environ.get("SHAREDAGENTS_WEB_ARTIFACTS_DIR") or (sharedagents_root() / "web" / "artifacts")
).expanduser()
# Extensions the /files route is allowed to serve, mapped to their MIME type.
# Raster images/PDF may render inline after signature checks; all other types
# are forced to opaque attachment downloads. Executable/script types stay out.
_ARTIFACT_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".zip": "application/zip",
    ".7z": "application/x-7z-compressed",
    ".tar": "application/x-tar",
    ".gz": "application/gzip",
    ".tgz": "application/gzip",
    ".parquet": "application/octet-stream",
    ".feather": "application/octet-stream",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}

# The gateway session-log dir (where the gateway writes/streams <hash>.jsonl).
# The web server reads JSONL sizes here to compute the live-stream tail watermark.
GW_SESSIONS_DIR = Path(os.environ.get("SHAREDAGENTS_GW_SESSIONS_DIR") or sessions_root()).expanduser()

# Legacy paths scanned during one-shot migration (never mutated — read-only).
# These point at places where the old per-user file store may have lived.
_SESSIONS_DIR_DEPRECATED = Path(
    os.environ.get("SHAREDAGENTS_SESSION_DIR") or (sharedagents_root() / "web" / "legacy_sessions")
).expanduser()
# Also scan the gateway sessions dir (backward compat for files already there).
# Only directories with `.json` files (not `.jsonl`/`.lock`) are ingested.
_GW_SESSIONS_DEPRECATED = Path(os.environ.get("SHAREDAGENTS_GW_SESSIONS_DIR") or sessions_root()).expanduser()
# Trailing COMPLETED turns on the initial history page. Older turns load
# on scroll-up via ?before=. A turn normally projects to two chat messages;
# ``_page_history`` (web-only store) therefore uses 2× this many messages.
# Keep in lockstep with the gateway's authoritative default.
HISTORY_PAGE_DEFAULT = 2
HISTORY_PAGE_MESSAGES = HISTORY_PAGE_DEFAULT * 2


def _jsonl_eof(sid):
    """Byte size of the gateway session log = the live-stream tail watermark.

    Opening the SSE from this offset means nothing historical replays (we're at
    EOF); the browser renders the REST page and only tails NEW events. 0 when the
    log is absent (plain web session) → caller falls back to a full replay."""
    if not sid:
        return 0
    try:
        return (GW_SESSIONS_DIR / f"{sid}.jsonl").stat().st_size
    except OSError:
        return 0


def _gw_session_log_exists(sid):
    """True if a gateway session log exists — a cheap O(1) stat.

    The existence test for a gateway-native session (one opened by hash from
    the TUI / agent monitor that the web store hasn't adopted yet). Used on the
    SSE-open hot path INSTEAD of ``gw_session_history`` (which folds the entire
    ``<hash>.jsonl`` just to produce a truthy/None) so switching to a large or
    active session doesn't pay a whole-file fold just to decide it can stream."""
    if not sid:
        return False
    try:
        return (GW_SESSIONS_DIR / f"{sid}.jsonl").exists()
    except OSError:
        return False


def _page_history(messages, before, limit):
    """Slice the folded message list to the page ending at ``before`` (exclusive).

    Returns ``(page, before_cursor, total)``. ``before`` is a message index;
    ``None`` → the latest page. ``before_cursor`` is the index to request for the
    next OLDER page (``None`` at beginning-of-history). This keeps the browser to
    O(page) even though the gateway still folds the whole log server-side (the
    reverse-reader optimization is deferred — the felt cost was the browser)."""
    msgs = messages or []
    total = len(msgs)
    lim = limit if (isinstance(limit, int) and limit > 0) else HISTORY_PAGE_MESSAGES
    end = total if before is None else max(0, min(before, total))
    start = max(0, end - lim)
    return msgs[start:end], (start if start > 0 else None), total


# Public base path the app is served under (e.g. "/sharedagents" when reverse-proxied
# at example.com/sharedagents). Empty = served at root (local/dev). The reverse
# proxy STRIPS this prefix, so Flask routes stay at root; the app only needs to
# EMIT base-prefixed URLs so the browser's asset/API requests come back through
# the prefixed location. Injected into the page as window.SHAREDAGENTS_BASE.
#
# ``SHAREDAGENTS_WEB_BASE_URL`` is the optional canonical public origin + path. It
# lets the browser receive an absolute URL when the Web UI is behind a proxy
# (for example ``https://example.com/sharedagentsweb``) without asking an
# agent/model to guess or hand-write a reverse-proxy URL. When present, its
# path also supplies BASE_PATH unless SHAREDAGENTS_WEB_BASE_PATH is set explicitly.
def _normalise_public_base_url(raw: str) -> str:
    value = str(raw or "").strip().rstrip("/")
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


_CONFIGURED_PUBLIC_BASE_URL = _normalise_public_base_url(os.environ.get("SHAREDAGENTS_WEB_BASE_URL", ""))
_configured_base_path = os.environ.get("SHAREDAGENTS_WEB_BASE_PATH", "").strip().strip("/")
if not _configured_base_path and _CONFIGURED_PUBLIC_BASE_URL:
    _configured_base_path = urlsplit(_CONFIGURED_PUBLIC_BASE_URL).path.strip("/")
BASE_PATH = f"/{_configured_base_path}" if _configured_base_path else ""
PUBLIC_BASE_URL = _CONFIGURED_PUBLIC_BASE_URL
if PUBLIC_BASE_URL and not urlsplit(PUBLIC_BASE_URL).path and BASE_PATH:
    PUBLIC_BASE_URL = f"{PUBLIC_BASE_URL}{BASE_PATH}"


def public_web_url(path: str) -> str:
    """Return a browser-facing URL through the configured Web base."""
    relative = "/" + str(path or "").lstrip("/")
    return f"{PUBLIC_BASE_URL}{relative}" if PUBLIC_BASE_URL else f"{BASE_PATH}{relative}"


# Fail-fast on missing SECRET_KEY — never fall back to a default.
_SECRET_KEY = os.environ.get("SHAREDAGENTS_WEB_SECRET")
if not _SECRET_KEY:
    raise RuntimeError(
        "SHAREDAGENTS_WEB_SECRET env var is required and was not set. "
        "Export a random value, e.g.: export SHAREDAGENTS_WEB_SECRET=$(openssl rand -hex 32)"
    )

# Absolute asset paths so the app serves correctly whether launched from an
# installed wheel (any cwd) or an editable checkout. Flask would otherwise
# resolve these relative to the app's import location, which is fine here but
# brittle if the module is ever imported oddly.
_PKG_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(_PKG_DIR / "templates"),
    static_folder=str(_PKG_DIR / "static"),
    static_url_path="/static",
)
app.config["SECRET_KEY"] = _SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB max request body (base64 images + document uploads)


@app.context_processor
def _inject_site_title():  # noqa: D401
    """Inject the white-label ``site_title`` into every template's context so
    each page's <title> / main heading renders the operator's chosen product
    name (``SHAREDAGENTS_WEB_TITLE``) instead of a hardcoded "sharedagents" literal.
    A context processor is the single chokepoint: no per-route
    ``render_template`` call needs to pass it explicitly.

    Also injects ``is_admin`` for any current or future template surface that
    opts into the reusable admin whitelist.  Agent Builder itself is visible
    to every authenticated user.
    """
    email = getattr(g, "user_email", None)
    admin = email is not None and is_admin_user(email)
    viewer_email = str(email or "").strip().lower()
    return {
        # Signed-in viewer for the chat's optimistic author and header badge.
        # Empty on pages rendered without an identity.
        "current_user_email": viewer_email,
        "current_user_name": auth_store.display_name(viewer_email) if viewer_email else "",
        "site_title": SITE_TITLE,
        "is_admin": admin,
        "public_base": PUBLIC_BASE_URL,
        "disable_sse": os.environ.get("SHAREDAGENTS_WEB_DISABLE_SSE", "").strip().lower() in {"1", "true", "yes", "on"},
    }


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

# ── SessionStore (extracted → session_store.py, Task 2 de-monolith) ──
# Re-exported so ``sharedagents_web.server.SessionStore`` and every existing import
# keep resolving. The extracted class references this module's constants,
# the ``_iso`` helper, and the per-session gw-lock registry lazily via
# ``server.<name>`` at call time, so importing it here (mid-module) is
# cycle-free.
from sharedagents_web.session_store import SessionStore  # noqa: E402  (re-export)

store = SessionStore()

# Native auth store + cross-worker web state live in the SAME ~/.sharedagents/web
# SQLite database as sessions (no new infrastructure — no-external-dep
# posture, even internally). Native auth is the PRIMARY access path;
# the web-state store carries the live-switchable gateway registry and
# device presence for seamless cross-device handoff.
from sharedagents_web.native_auth import COOKIE_NAME as _AUTH_COOKIE, NativeAuthStore  # noqa: E402
from sharedagents_web.session_sharing import SharingStore  # noqa: E402
from sharedagents_web.webstate import WebState  # noqa: E402

auth_store = NativeAuthStore(SESSION_DB_PATH)
web_state = WebState(SESSION_DB_PATH, seed_gateway=GATEWAY_URL)
# Per-conversation visibility (private / public to everyone or picked people).
sharing_store = SharingStore(SESSION_DB_PATH)


def current_gateway() -> str:
    """The active gateway URL, resolved per-request from the web-state store.

    Replaces the import-time ``GATEWAY_URL`` constant everywhere a live request
    dials the gateway, so an operator can switch gateways at runtime
    without restarting the serving process. Falls back to the env-seeded
    ``GATEWAY_URL`` if the store has no row yet.
    """
    try:
        return web_state.get_active_gateway(GATEWAY_URL) or GATEWAY_URL
    except Exception:
        return GATEWAY_URL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_gateway_auth_token() -> str:
    """Bearer token the web presents to the gateway's control/OpenAI surface.

       The gateway REQUIRES this whenever it is bound outside loopback
    — which it MUST be so other LAN clients (e.g. Open WebUI
       at ``<host>:8100/openai/v1``) can reach it. The web is a first-party client
       and must present the SAME token, or every control call 401s and the
       monitor/chat break. Read from env first, then the ``gateway`` block of the
       global config (the exact token the gateway itself loads). Empty ⇒ send no
       header, which is correct for an open, loopback-only gateway.
    """
    tok = os.environ.get("SHAREDAGENTS_GATEWAY_AUTH_TOKEN", "").strip()
    if tok:
        return tok
    try:
        from sharedagents_core.kernel.config import load_global_config

        gw = (load_global_config() or {}).get("gateway") or {}
        return str(gw.get("auth_token") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


_GATEWAY_AUTH_TOKEN = _load_gateway_auth_token()
_GATEWAY_UNAVAILABLE_RETRY_DELAYS_S = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0)


def _gateway_auth_header() -> dict:
    """``{Authorization: Bearer <token>}`` when a gateway token is configured,
    else ``{}``. Sending a token to an open gateway is harmless (it accepts
    any), so this is safe to attach unconditionally when a token exists."""
    return {"Authorization": f"Bearer {_GATEWAY_AUTH_TOKEN}"} if _GATEWAY_AUTH_TOKEN else {}


def _sent_then_dropped(exc) -> bool:
    """True when a requests error happened after the request was written."""
    import urllib3

    if isinstance(exc, (requests.ConnectTimeout, requests.exceptions.ProxyError, requests.exceptions.SSLError)):
        return False
    if isinstance(exc, requests.exceptions.ChunkedEncodingError):
        return True
    cause = exc.args[0] if exc.args else None
    return isinstance(cause, urllib3.exceptions.ProtocolError)


def gateway_request(method, path, **kwargs):
    """Make a request to the sharedagents gateway (live-resolved target)."""
    url = f"{current_gateway()}{path}"
    # Read-only gateway calls are commonly issued by the WebUI immediately
    # after its process starts.  A systemd ``After=`` ordering does not wait
    # for the gateway socket, so default GETs to the local gateway to the
    # same bounded readiness retry used by the approvals panel.  Mutating
    # calls remain fail-fast unless a caller opts in explicitly.
    report_read_timeout = kwargs.pop("report_read_timeout", False)
    retry_unavailable = bool(kwargs.pop("retry_unavailable", method.upper() == "GET"))
    kwargs.setdefault("timeout", 120)
    if _GATEWAY_AUTH_TOKEN:
        headers = dict(kwargs.get("headers") or {})
        headers.setdefault("Authorization", f"Bearer {_GATEWAY_AUTH_TOKEN}")
        kwargs["headers"] = headers
    retry_delays = (0.0, *_GATEWAY_UNAVAILABLE_RETRY_DELAYS_S)
    for attempt, delay in enumerate(retry_delays):
        if delay:
            time.sleep(delay)
        try:
            resp = requests.request(method, url, **kwargs)
        except requests.ReadTimeout:
            if report_read_timeout:
                raise
            if not retry_unavailable or attempt == len(retry_delays) - 1:
                return None
            continue
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
            # A connection dropped after the request was sent (not a refused
            # connect) leaves the outcome as unknown as a read timeout.
            if report_read_timeout and _sent_then_dropped(exc):
                raise requests.ReadTimeout(str(exc)) from exc
            if not retry_unavailable or attempt == len(retry_delays) - 1:
                return None
            continue
        except requests.RequestException:
            if not retry_unavailable or attempt == len(retry_delays) - 1:
                return None
            continue
        if retry_unavailable and resp.status_code in {502, 503, 504} and attempt < len(retry_delays) - 1:
            continue
        return resp
    return None


# Per-web-session lock so two concurrent /api/chat or /api/command calls
# can't each create a separate gateway session (race → chat and commands
# landing on different gateway sessions). Keyed by web session id.
_gw_session_locks = {}
_gw_locks_guard = threading.Lock()


def ensure_gw_session(sdata):
    """Return the web session's durable SID unchanged."""
    return str((sdata or {}).get("id") or "") or None


def _iso(ts):
    """Coerce a gateway timestamp (epoch-ms int, or already-ISO string) into a
    sortable ISO string. The web session list sorts conversations by
    ``updated_at`` lexicographically, so gateway-originated rows (epoch-ms)
    must be rendered into the same ISO shape the web store uses."""
    if ts is None or ts == "":
        return ""
    if isinstance(ts, str):
        return ts
    try:
        return datetime.fromtimestamp(float(ts) / 1000.0, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def gw_session_history(
    hash_id,
    *,
    before=None,
    limit=HISTORY_PAGE_DEFAULT,
):
    """Fetch one gateway session's header + folded chat history by hash.

    Cross-agent read over the gateway's session log (sessions.db + JSONL).
    Returns the parsed dict ``{hash, agent, title, model, engine_model,
    messages:[{role,content}], updated_at}`` or ``None`` if the gateway is
    unreachable / the session is unknown. This is what lets the web open a
    conversation started in the TUI (or a subagent session double-clicked in
    the agent monitor) and render its full recent history.
    """
    if not hash_id:
        return None
    params = {"limit": max(1, int(limit or HISTORY_PAGE_DEFAULT))}
    if before is not None:
        params["before"] = int(before)
    resp = gateway_request(
        "GET",
        f"/control/v1/orchestrator/sessions/{hash_id}",
        params=params,
        timeout=15,
    )
    if resp is None or resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if not isinstance(data, dict) or not data.get("ok"):
        return None
    return data


def gw_push_title(hash_id, title):
    """Push a conversation title into the gateway session log (best-effort).

    Keeps the web UI's title and the session log / monitor-tree title in
    lock-step. Cross-agent route — the gateway resolves the owning agent. Never
    raises into the caller; a failed sync just leaves the log's own
    turn-derived title in place."""
    if not hash_id or not title or title == "New Chat":
        return False
    resp = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{hash_id}/title",
        json={"title": title},
        timeout=10,
    )
    if resp is None or resp.status_code != 200:
        return False
    # Require the gateway to confirm the title was actually written (ok:true) —
    # a 200 with ok:false (refused / save failed) must NOT be marked synced, or
    # the web would stop retrying a title the log never recorded.
    try:
        return bool(resp.json().get("ok"))
    except ValueError:
        return False


def gw_root_sessions(limit=50):
    """List the gateway's top-level sessions (across all agents), newest first.

    Prefers the cheap ``?roots=1`` index read. Falls back to flattening the
    full monitor tree when the gateway is an older build without that route.
    Subagent sessions stay out of the sidebar (monitor + deep-link only).
    """
    resp = gateway_request(
        "GET",
        "/control/v1/orchestrator/sessions",
        params={"roots": "1", "limit": int(limit)},
        timeout=2,
        retry_unavailable=False,
    )
    if resp is not None and resp.status_code == 200:
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        rows = (payload or {}).get("sessions") if isinstance(payload, dict) else None
        if isinstance(rows, list):
            out = []
            for root in rows:
                if not isinstance(root, dict):
                    continue
                out.append(
                    {
                        "hash": root.get("hash", ""),
                        "title": root.get("title") or "",
                        "session_summary": root.get("session_summary") or "",
                        "agent": root.get("agent") or DEFAULT_AGENT,
                        "model": root.get("model") or "",
                        "topic": root.get("topic") or "general",
                        "updated_at": _iso(root.get("last_turn_ts")),
                        "last_turn_ts": root.get("last_turn_ts") or 0,
                        "live": bool(root.get("live")),
                        "state": root.get("state") or "idle",
                    }
                )
            out.sort(key=lambda r: r.get("last_turn_ts", 0), reverse=True)
            return out[:limit]
    return []


def _native_token_from_request() -> str:
    """Return the native device token from cookie or ``Authorization: Bearer``.

    Corporate proxies sometimes strip ``Cookie`` while still forwarding
    ``Authorization``.  SSE and polling stay on ordinary HTTP so a Bearer
    header is enough — the token is never put in the query string.
    """
    raw = (request.cookies.get(_AUTH_COOKIE) or "").strip()
    if raw:
        return raw
    auth = (request.headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        if token:
            return token
    return ""


def verify_dashboard_auth():
    """Verify auth by forwarding cookies/Bearer to dashboard /verify-token.

    Denies access when:
      - No Cookie or Authorization header is present
      - Dashboard is unreachable
      - Dashboard returns non-200

    This is sharedagents external auth, an optional path: it is enabled
    only when SHAREDAGENTS_EXTERNAL_AUTH=1 and is never the source of truth. It is
    reached from ``require_auth`` only after native auth declines.

    Returns (True, user_info) or (False, error_message).
    """
    cookie_header = request.headers.get("Cookie", "")
    authorization = (request.headers.get("Authorization") or "").strip()
    if not cookie_header and not authorization:
        return False, "No authentication cookie"

    # Authorize the actual public mount, not the legacy /sharedagents/ alias.
    headers = {"X-Original-URI": (BASE_PATH or "/sharedagents") + "/"}
    if cookie_header:
        headers["Cookie"] = cookie_header
    if authorization:
        headers["Authorization"] = authorization

    try:
        resp = requests.get(
            f"{DASHBOARD_URL}/verify-token",
            headers=headers,
            timeout=5,
        )
    except requests.RequestException:
        # Dashboard unreachable → deny (fail-closed)
        return False, "Dashboard unreachable"

    if resp.status_code != 200:
        return False, f"Auth failed: {resp.status_code}"

    email = resp.headers.get("X-User-Email")
    if not email:
        return False, "No user identity from dashboard"
    return True, {
        "email": email,
        "name": resp.headers.get("X-User-Name", "User"),
    }


def verify_native_auth():
    """Verify the native sharedagents device token against the LOCAL SQLite store.

    Accepts the same token ``/api/verify`` checks, from (in order):
    the ``sharedagents_session`` cookie, then ``Authorization: Bearer``.

    This is the PRIMARY auth path and the source of truth for access.
    It touches nothing but local SQLite — no dashboard, no network — so valid
    native credentials authenticate even with every external dependency
    unreachable. Fail-closed: an absent/invalid/expired/revoked token
    returns (False, ...).

    Returns (True, {"email": ...}) or (False, error_message).
    """
    raw = _native_token_from_request()
    if not raw:
        return False, "No native session"
    email = auth_store.verify_token(raw)
    if not email:
        return False, "Invalid or expired native session"
    return True, {"email": email, "name": email}


# How ``require_auth`` accepted this request (``g.auth_via``): a long-lived
# stream re-checks the login through the same mechanism.
AUTH_VIA_NATIVE = "native"
AUTH_VIA_DASHBOARD = "dashboard"
AUTH_VIA_DEBUG = "debug"


def _external_auth_enabled() -> bool:
    """Whether sharedagents external auth (a dashboard-verified login) is enabled (default OFF).

    Native auth is primary; external auth is at most an optional
    alternative. Ships disabled by default; set SHAREDAGENTS_EXTERNAL_AUTH=1 to also accept a login the
    dashboard at ``DASHBOARD_URL`` verifies.
    """
    return os.environ.get("SHAREDAGENTS_EXTERNAL_AUTH") == "1"


def external_login_redirect():
    """Configured SSO login for proxied installs; native-only installs stay local.

    Keep the browser's mount and query in the return target. Never return to
    the native login route itself, which would create a sign-in loop.
    """
    login_url = os.environ.get("SHAREDAGENTS_EXTERNAL_LOGIN_URL", "").strip()
    if not _external_auth_enabled() or not login_url:
        return None
    from urllib.parse import urlencode

    target = (BASE_PATH or "") + ("/" if request.path == "/login" else request.full_path.rstrip("?"))
    separator = "&" if "?" in login_url else "?"
    return redirect(login_url + separator + urlencode({"next": target}))


def _waitress_proxy_kwargs() -> dict:
    """Waitress drops X-Forwarded-For by default. Keep it when trusted proxies
    are listed; ``routes.auth._client_ip`` only reads it from those peers."""
    if os.environ.get("SHAREDAGENTS_WEB_TRUSTED_PROXIES", "").strip():
        return {"clear_untrusted_proxy_headers": False}
    return {}


def _loopback_debug_bypass_ok() -> bool:
    """Loopback-only debug bypass.

    SHAREDAGENTS_WEB_DEBUG historically skipped auth entirely — a separate
    unauthenticated break-glass path. It now survives ONLY when the server is
    bound to loopback and NOT behind a trusted proxy, so it can never expose an
    unauthenticated surface on a LAN/proxied bind. Fail-closed otherwise.
    """
    if os.environ.get("SHAREDAGENTS_WEB_DEBUG") != "1":
        return False
    if os.environ.get("SHAREDAGENTS_WEB_TRUST_PROXY") == "1":
        return False
    host = os.environ.get("SHAREDAGENTS_WEB_HOST", "127.0.0.1")
    return host in ("127.0.0.1", "localhost", "::1", "[::1]")


def require_auth(f):
    """Flask decorator that blocks unauthenticated requests on API routes.

    Native-first: try the local native session; only if that declines
    AND external auth is enabled do we fall back to dashboard delegation.
    Stores the authenticated user's email on flask.g so downstream routes can
    scope data (sessions, etc.) to the current user — preventing IDOR.

    Then runs the access gate: a non-admin account reaches only the endpoints
    ``access.USER_ENDPOINTS`` opens, and only conversations they can reach.
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        if _loopback_debug_bypass_ok():
            g.user_email = "debug@localhost"
            g.auth_via = AUTH_VIA_DEBUG
            return f(*args, **kwargs)
        authenticated, info = verify_native_auth()
        via = AUTH_VIA_NATIVE
        if not authenticated and _external_auth_enabled():
            authenticated, info = verify_dashboard_auth()
            via = AUTH_VIA_DASHBOARD
        if not authenticated:
            return jsonify({"error": info}), 401
        email = info.get("email")
        if not email:
            return jsonify({"error": "No user identity"}), 401
        g.user_email = email
        g.auth_via = via
        from sharedagents_web.access import gate_request

        denied = gate_request(email)
        if denied is not None:
            return denied
        return f(*args, **kwargs)

    return decorated


def require_auth_page(f):
    """Like ``require_auth`` but redirects an unauthenticated BROWSER to /login
    (spec §4.3). Same fail-closed decision — it never serves the page without a
    resolved identity — only the *response* differs (302 vs 401 JSON).
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        if _loopback_debug_bypass_ok():
            g.user_email = "debug@localhost"
            g.auth_via = AUTH_VIA_DEBUG
            return f(*args, **kwargs)
        authenticated, info = verify_native_auth()
        via = AUTH_VIA_NATIVE
        if not authenticated and _external_auth_enabled():
            authenticated, info = verify_dashboard_auth()
            via = AUTH_VIA_DASHBOARD
        email = info.get("email") if authenticated else None
        if not email:
            external_redirect = external_login_redirect()
            if external_redirect is not None:
                return external_redirect
            return redirect((BASE_PATH or "") + "/login")
        g.user_email = email
        g.auth_via = via
        from sharedagents_web.access import gate_request

        denied = gate_request(email)
        if denied is not None:
            return denied
        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# Admins — role on the account, SHAREDAGENTS_WEB_ADMINS, the
# local operator, the loopback debug identity, else the oldest account.
# ---------------------------------------------------------------------------


def is_admin_user(email: str) -> bool:
    """Whether ``email`` is an admin (see ``access.admin_source``)."""
    from sharedagents_web.access import is_admin

    return is_admin(email)


def require_admin(f):
    """Defense in depth for admin routes (already admin-only via the gate,
    which denies any endpoint not opened to users). Apply after
    ``require_auth``."""

    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_admin_user(getattr(g, "user_email", "") or ""):
            return jsonify({"error": "admin only"}), 403
        return f(*args, **kwargs)

    return decorated


# ---------------------------------------------------------------------------
# CSP — defense-in-depth: block all off-origin loads at the HTTP layer
# ---------------------------------------------------------------------------

_GZIP_TYPES = frozenset(
    (
        "text/html",
        "text/css",
        "text/plain",
        "text/javascript",
        "application/javascript",
        "application/json",
        "application/manifest+json",
        "image/svg+xml",
    )
)


@app.after_request
def _gzip_response(response):
    """Gzip compressible responses for mobile / limited-network clients.

    The uncompressed JS/CSS/JSON shipped ~3x heavier than needed (app.js 266KB,
    style.css 93KB) — a real problem on a phone over a slow link. We gzip by
    CONTENT-TYPE (covers Flask static files, whose responses are
    ``direct_passthrough``) and explicitly SKIP streaming SSE
    (``text/event-stream``) — gzipping a live stream would buffer it. Never
    breaks a response on error."""
    try:
        if "gzip" not in (request.headers.get("Accept-Encoding", "").lower()):
            return response
        ctype = (response.mimetype or "").lower()
        if ctype not in _GZIP_TYPES:
            return response
        if response.headers.get("Content-Encoding"):
            return response
        # Reading get_data() on a passthrough (static file) response materializes
        # the bytes; safe for files, and we already excluded the SSE streams by
        # content-type above.
        response.direct_passthrough = False
        data = response.get_data()
        if len(data) < 860:  # below ~1 TCP segment, not worth the CPU/overhead
            return response
        comp = gzip.compress(data, compresslevel=6)
        response.set_data(comp)
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(comp))
        existing_vary = response.headers.get("Vary", "")
        if "accept-encoding" not in existing_vary.lower():
            response.headers["Vary"] = f"{existing_vary}, Accept-Encoding".lstrip(", ")
    except Exception:  # noqa: BLE001 — compression must never break a response
        pass
    return response


@app.after_request
def set_csp(response):
    """Strict Content-Security-Policy on every response.

    Allows same-origin ('self') for all resource types so the app's own
    vendored JS/CSS still loads.  The dashboard origin is allowed as a
    frame-ancestor so it can iframe this app.  Everything external is
    explicitly blocked — even if something slips past the sanitizer,
    the browser will refuse to fetch it.

    frame-ancestors source: prefer the explicit SHAREDAGENTS_WEB_FRAME_ANCESTORS
    allow-list (e.g. "'self' https://dashboard.example.com") — the browser-facing
    dashboard origin that iframes this app is NOT the same as DASHBOARD_URL
    (the server-side auth endpoint, typically http://127.0.0.1:20100). Behind a
    reverse proxy these differ, so deriving frame-ancestors solely from
    DASHBOARD_URL blocks the real embed. Fall back to the DASHBOARD_URL origin
    only when the allow-list is unset.
    """
    frame_ancestors = os.environ.get("SHAREDAGENTS_WEB_FRAME_ANCESTORS", "").strip()
    if not frame_ancestors:
        # Extract the origin from DASHBOARD_URL (e.g. http://127.0.0.1:5000)
        dash_url = DASHBOARD_URL.rstrip("/")
        # origin = scheme + "://" + netloc
        if "://" in dash_url:
            scheme, rest = dash_url.split("://", 1)
            netloc = rest.split("/")[0]
            frame_ancestors = f"{scheme}://{netloc}"
        else:
            frame_ancestors = dash_url

    # The inline realtime-voice button opens a WebSocket to the realtime server.
    # Same-origin (a reverse-proxy path like "/sharedagentsrt/") is already covered by
    # 'self'; a cross-origin REALTIME_URL (e.g. http://host:8123/ in dev) must be
    # whitelisted explicitly for both the http (clientconfig) and ws origins.
    connect_src = "'self'"
    if REALTIME_URL and "://" in REALTIME_URL:
        from urllib.parse import urlsplit

        parts = urlsplit(REALTIME_URL)
        if parts.scheme and parts.netloc:
            ws_scheme = "wss" if parts.scheme == "https" else "ws"
            connect_src = f"'self' {parts.scheme}://{parts.netloc} {ws_scheme}://{parts.netloc}"

    # Inline scripts are forbidden globally; a route that must serve one (the
    # login page) sets g.csp_nonce, and its <script nonce=...> is allowed via
    # the per-request nonce appended here.
    script_src = "'self' blob: 'wasm-unsafe-eval'"
    _nonce = getattr(g, "csp_nonce", None)
    if _nonce:
        script_src += f" 'nonce-{_nonce}'"

    policy = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; "
        # blob: is required for the realtime-voice AudioWorklet (PCM tap loaded
        # from a Blob URL). 'wasm-unsafe-eval' is required for the in-browser
        # "Hey Jarvis" wake word: onnxruntime-web instantiates WebAssembly, which
        # strict CSP blocks without it (the engine would then silently fall back
        # to continuous streaming and the wake word would never fire).
        f"script-src {script_src}; "
        "worker-src 'self' blob:; "
        "font-src 'self'; "
        f"connect-src {connect_src}; "
        "object-src 'none'; "
        "base-uri 'none'; "
        f"frame-ancestors {frame_ancestors};"
    )
    response.headers["Content-Security-Policy"] = policy
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/static/models/<path:filename>")
def voice_model(filename: str):
    """Voice/wake-word ONNX models: operator-downloaded copies win over bundled.

    The openWakeWord feature + wake-word models are CC BY-NC-SA 4.0 and cannot
    ship inside the MIT-licensed package, so ``sharedagents web
    fetch-voice-models`` drops them under ``<home>/voice-models/``. MIT-licensed
    models that do ship (``silero_vad.onnx``) fall through to the bundled
    ``static/models/`` directory. Same public/no-auth posture as the rest of
    the static tree; ``send_from_directory`` rejects traversal either way.
    """
    from sharedagents_core.layout import voice_models_root

    downloaded = voice_models_root() / filename
    if downloaded.is_file():
        return send_from_directory(str(voice_models_root()), filename)
    return send_from_directory(os.path.join(app.static_folder, "models"), filename)


@app.route("/favicon.ico")
def favicon():
    """Browsers request /favicon.ico unconditionally (even with a <link> tag,
    in headless Chrome); serve the SVG so it doesn't 404. Public (no auth) so
    the tab icon shows on the login page too."""
    return send_from_directory(app.static_folder, "favicon.svg", mimetype="image/svg+xml")


# ---------------------------------------------------------------------------
# PWA — installable web-app manifest + service worker
# ---------------------------------------------------------------------------
# Installability needs a linked manifest AND a registered service worker. The
# SW is deliberately NON-CACHING (network-only passthrough): it exists to make
# the app installable + push-capable, not to serve an offline shell. Because it
# caches no application asset, a deploy can never be masked by a stale SW cache
# The SW layer is trivially cache-coherent). Both are public
# (no auth) like /favicon.ico: installability metadata is not sensitive, and the
# SW must register before login so the app is installable from the login page.

_SERVICE_WORKER_JS = """\
// sharedagents service worker. Network-only passthrough: NO app
// asset is ever cached, so a deployment is never served stale from here.
// Present only to satisfy PWA installability + future push.
self.addEventListener('install', function () { self.skipWaiting(); });
self.addEventListener('activate', function (event) {
  event.waitUntil(self.clients.claim());
});
self.addEventListener('fetch', function (event) {
  // Only intercept top-level navigations (installability wants a fetch
  // handler); proxy straight to the network. Everything else -- SSE streams,
  // /api calls, static assets -- is left entirely to the browser so we never
  // interfere with EventSource or range requests.
  if (event.request.mode === 'navigate') {
    event.respondWith(fetch(event.request));
  }
});
// Notifications are displayed through THIS registration
// (registration.showNotification), never the page-side `new Notification()`
// constructor -- Chrome on Android does not implement that constructor and
// throws `Illegal constructor`. Because display happens here, the click is
// delivered here too: without this handler, tapping the alert does nothing.
self.addEventListener('notificationclick', function (event) {
  event.notification.close();
  var data = event.notification.data || {};
  var target = data.url || self.registration.scope;
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then(function (list) {
        // Focus an sharedagents tab if one is already open; only then open a new one.
        for (var i = 0; i < list.length; i++) {
          var c = list[i];
          if (c.url.indexOf(self.registration.scope) === 0 && 'focus' in c) {
            return c.focus();
          }
        }
        if (self.clients.openWindow) return self.clients.openWindow(target);
        return undefined;
      })
  );
});
"""


@app.route("/manifest.webmanifest")
def web_app_manifest():
    """Installable-PWA manifest. Public, like /favicon.ico.

    Colors/name track the app's own design tokens (style.css :root --bg-primary)
    rather than a bolted-on palette, so the installed app chrome matches the UI.
    """
    scope = (BASE_PATH + "/") if BASE_PATH else "/"
    manifest = {
        "name": SITE_TITLE,
        "short_name": SITE_TITLE,
        "description": f"{SITE_TITLE} operator console",
        "start_url": scope,
        "scope": scope,
        "display": "standalone",
        "background_color": "#09090e",
        "theme_color": "#09090e",
        "icons": [
            {
                "src": f"{BASE_PATH}/static/favicon.svg",
                "sizes": "any",
                "type": "image/svg+xml",
                "purpose": "any maskable",
            },
            # Android needs a raster icon: the notification tray does not
            # render SVG reliably and falls back to a blank glyph.
            {
                "src": f"{BASE_PATH}/static/icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any",
            },
        ],
    }
    resp = jsonify(manifest)
    resp.mimetype = "application/manifest+json"
    return resp


@app.route("/sw.js")
def service_worker():
    """Serve the (non-caching) service worker. Public, and
    ``no-cache`` so a new SW is always re-checked on navigation.
    """
    resp = Response(_SERVICE_WORKER_JS, mimetype="text/javascript")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.template_global()
def asset_rev(path: str) -> int:
    """Cache-busting version (file mtime) for a static asset URL.

    Browsers/edges kept serving pre-deploy app.js/style.css against fresh
    HTML (observed on Android Chrome 2026-06-12: page reloads with zero
    asset revalidation requests). Versioned URLs make every deploy a new
    URL, so a stale cache entry for the old URL can never be reused.
    """
    try:
        return int(os.stat(os.path.join(app.static_folder, path)).st_mtime)
    except OSError:
        return 0


# Realtime voice runs in the standalone sharedagents_realtime server (loopback
# :8123 by default; not WS-proxyable through Flask). The Voice nav button
# points here. Leave unset to let the browser derive <current-host>:8123;
# set SHAREDAGENTS_REALTIME_URL once the realtime server is exposed behind a proxy.
REALTIME_URL = os.environ.get("SHAREDAGENTS_REALTIME_URL", "").strip()


# ── Document → markdown (chat attachment pipeline) ──────────────────────────
# The web venv deliberately does NOT bundle markitdown. We shell out to the
# MAIN sharedagents venv's `markitdown` binary — the same install behind
# markitdown-mcp and the agent Read tool — so the web app stays lean and there
# is one converter for the whole system. Override the path with
# SHAREDAGENTS_MARKITDOWN_BIN; falls back to `markitdown` on PATH.


# ---------------------------------------------------------------------------
# Native authentication routes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Live gateway switching
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Presence + seamless cross-device handoff
# ---------------------------------------------------------------------------


# Upper bound on how many conversations the Topics page scans+groups. This is
# a real cap (the gateway tree scan is bounded too): with more than this many
# sessions, topics whose only conversations are older than the newest
# TOPICS_SCAN_LIMIT fall off the page. /api/topics reports ``capped`` so the UI
# can say so rather than imply a complete inventory.
TOPICS_SCAN_LIMIT = 500

# Web-owned topic store — lets the UI create topics WITHOUT Telegram. A tiny
# JSON sidecar of ``sessions.db`` (same SHAREDAGENTS_WEB_DB dir), independent of the
# Telegram forum registry. Merged into /api/topics as empty buckets; a name
# that later appears in Telegram or gains a bound session is deduped away.
WEB_TOPICS_PATH = SESSION_DB_PATH.parent / "web_topics.json"

# Reserved catch-all — every ungrouped conversation lands in "general", so it is
# not a user-creatable topic (matching how the sidebar hides the lone bucket).
_RESERVED_TOPIC_NAMES = {"general"}
_MAX_TOPIC_LEN = 120


def load_web_topics(*, path=None):
    """Return the web-local topic names (list of strings), newest-first.

    Missing / malformed store → ``[]``. Never raises into the request path.
    """
    p = path or WEB_TOPICS_PATH
    try:
        if not p.is_file():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    names = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(names, list):
        return []
    out = []
    seen = set()
    for n in names:
        s = str(n).strip()
        if not s or s.casefold() in seen:
            continue
        seen.add(s.casefold())
        out.append(s)
    return out


def add_web_topic(name, *, path=None):
    """Persist a new web-local topic name.

    Returns ``(topic, created)`` where ``created`` is False when the name
    already existed (idempotent). Raises ``ValueError`` for an empty or
    reserved name. Atomic write; newest name first.
    """
    topic = str(name or "").strip()[:_MAX_TOPIC_LEN].strip()
    if not topic:
        raise ValueError("topic name is empty")
    if topic.casefold() in _RESERVED_TOPIC_NAMES:
        raise ValueError(f"'{topic}' is reserved")
    p = path or WEB_TOPICS_PATH
    existing = load_web_topics(path=p)
    if any(topic.casefold() == e.casefold() for e in existing):
        return topic, False
    payload = {"topics": [topic] + existing}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, p)
    return topic, True


def _resolve_session_topics(sessions, registry=None):
    """Normalize each row's ``topic`` via the Telethon forum-topic registry.

    Returns ``(sessions, registry)``.
    """
    reg = registry if registry is not None else load_forum_topics()
    bindings = load_session_topic_bindings(reg)
    out = []
    for s in sessions:
        row = dict(s)
        sid = row.get("id") or ""
        row["topic"] = resolve_session_topic(
            sid,
            row.get("topic") or "general",
            reg,
            bindings,
        )
        out.append(row)
    return out, reg


def _load_meta_headers(hashes):
    """Batch-read title/model from ``sessions_meta`` for sidebar overlay.

    Subagent rows live in the web store with the adopt-time placeholder
    (``New Chat - HHMMSS``) and an empty model; they are not in
    ``gw_root_sessions``, so the list used to keep showing the stub. One
    IN-query overlays the generated title and the live model the monitor
    already reads from the same table.
    """
    ids = [str(h) for h in (hashes or []) if h]
    if not ids:
        return {}
    db = GW_SESSIONS_DIR / "sessions.db"
    if not db.is_file():
        return {}
    try:
        from sharedagents_session.session_index import connect_sessions_db

        con = connect_sessions_db(db)
        try:
            ph = ",".join("?" * len(ids))
            rows = con.execute(
                f"SELECT session_hash, title, model, agent_name, parent_session_id "
                f"FROM sessions_meta WHERE session_hash IN ({ph})",
                ids,
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return {}
    out = {}
    for row in rows:
        if isinstance(row, sqlite3.Row):
            hid = str(row["session_hash"] or "")
            out[hid] = {
                "title": str(row["title"] or ""),
                "model": str(row["model"] or ""),
                "agent": str(row["agent_name"] or ""),
                "parent_session_id": str(row["parent_session_id"] or ""),
            }
        else:
            hid = str(row[0] or "")
            out[hid] = {
                "title": str(row[1] or ""),
                "model": str(row[2] or ""),
                "agent": str(row[3] or ""),
                "parent_session_id": str(row[4] or "") if len(row) > 4 else "",
            }
    return out


def resolve_authoritative_session_title(
    session_hash: str,
    store_title: str,
) -> str:
    """Prefer gateway ``sessions_meta`` / JSONL title over web-store stubs.

    The web store keeps ``New Chat`` until something explicitly writes it;
    the gateway title agent publishes to ``sessions_meta`` and the session
    log. SSE ``event: state`` and GET session must not clobber a generated
    title with the store placeholder.
    """
    from sharedagents_session.session_title import is_default_title

    title = str(store_title or "").strip() or "New Chat"
    hid = str(session_hash or "").strip()
    if not hid or not is_default_title(title):
        return title
    meta = _load_meta_headers([hid]).get(hid) or {}
    meta_title = str(meta.get("title") or "").strip()
    if meta_title and not is_default_title(meta_title):
        return meta_title
    try:
        from sharedagents_session.session_reader import load_session_dict

        d = load_session_dict(GW_SESSIONS_DIR, hid, fold=False) or {}
        hist_title = str(d.get("title") or "").strip()
        if hist_title and not is_default_title(hist_title):
            return hist_title
    except Exception:  # noqa: BLE001 — best-effort overlay
        pass
    return title


def overlay_session_list_from_meta(
    sessions,
    meta_by_hash=None,
    *,
    drop_subagents=True,
):
    """Prefer generated ``sessions_meta`` title/model over web-store stubs.

    Adopted subagent rows are not roots (``gw_root_sessions`` already
    excludes them) but viewing a child from the Agent Monitor used to
    ``adopt_gateway`` them into the web store, so Conversations filled up
    with ``New Chat - HHMMSS`` clones. Drop those here; they stay
    openable from the monitor / deep-link.
    """
    from sharedagents_session.session_title import is_default_title

    rows = list(sessions or [])
    if meta_by_hash is None:
        hashes = []
        for s in rows:
            hid = s.get("id") or ""
            if hid:
                hashes.append(hid)
        meta_by_hash = _load_meta_headers(hashes)
    kept = []
    for s in rows:
        hid = str(s.get("id") or "")
        meta = meta_by_hash.get(hid) or {}
        parent = str(meta.get("parent_session_id") or s.get("parent_session_id") or "")
        if drop_subagents and parent:
            continue
        meta_title = str(meta.get("title") or "").strip()
        if meta_title and not is_default_title(meta_title):
            s["title"] = meta_title
        meta_model = str(meta.get("model") or "").strip()
        if meta_model:
            s["model"] = meta_model
        kept.append(s)
    return kept


def _merged_sessions(
    user_email,
    agent=None,
    limit=200,
    *,
    gateway_roots=None,
):
    """Web-store + gateway-native sessions, deduped, each carrying a topic.

    The same merge ``/api/sessions`` does, factored out so the Topics page can
    group conversations without re-implementing the dedup. Returns up to
    ``limit`` rows (newest-activity first via the callers' sort); a missing
    ``topic`` is defaulted to ``"general"`` so every row is groupable.
    """
    # Index gateway roots by hash so a bound/adopted web row can INHERIT its
    # gateway session's topic (the topic lives on the gateway session — e.g.
    # set from Telegram — while the web store predates the field), and so we
    # only append the gateway rows that no web row already represents.
    from sharedagents_web import access

    # A non-admin sees their own conversations plus the ones shared with them;
    # gateway roots then only feed live badges.
    viewer = str(user_email or "").strip().lower()
    scoped = not access.is_admin(viewer)
    shared_ids = access.visible_shared_ids(viewer) if scoped else set()

    def _visible(row):
        return access.row_owner(row) == viewer or row.get("id") in shared_ids

    gw_by_hash = {}
    if (not scoped) or shared_ids:
        roots = gw_root_sessions(limit=limit) if gateway_roots is None else list(gateway_roots)[:limit]
        for gw in roots:
            h = gw.get("hash")
            # Single-user: a session bound under a different
            # address belongs to the same operator, so binding owner does not
            # filter the listing. Comparing it to user_email is what made 99
            # real gateway-native conversations answer "Session not found"
            # after a second credential claimed the binding.
            if h:
                gw_by_hash[h] = gw

    sessions = []
    seen = set()
    activity_keys = (
        "activity",
        "version",
        "phase",
        "attention",
        "lifecycle",
        "last_outcome",
        "own_live",
        "tree_busy",
        "stale",
        "active_descendant_count",
        "open_job_count",
        "phase_changed_at",
        "heartbeat_at",
        "outcome_at",
        "status",
        "display",
        "capabilities",
    )

    def _is_placeholder_title(t: str) -> bool:
        t = (t or "").strip()
        # ``New Chat - HHMMSS`` is intentionally *not* a placeholder here:
        # it is the user-visible pending-title value until the detached title
        # agent publishes its generated replacement.
        return (
            not t
            or t.startswith("untitled-")
            or t in ("New Chat", "Untitled", "Session")
            or t.startswith("New session")
        )

    for s in store.list_sessions(
        user_email,
        agent,
        limit,
        only=_visible if scoped else None,
    ):
        s = dict(s)  # copy — never mutate the store's row objects
        gw = gw_by_hash.get(s.get("id"))
        # A web/adopted row bound to a gateway session with no topic of its own
        # inherits the gateway topic (so a Telegram-topic'd conversation viewed
        # on the web groups under the real topic, not "general").
        if gw and not s.get("topic"):
            s["topic"] = gw.get("topic") or "general"
        if not s.get("topic"):
            s["topic"] = "general"
        # Prefer the gateway/OpenCode title agent result over the web store's
        # first-line heuristic when the gateway has a real title.
        if gw:
            gw_title = (gw.get("title") or "").strip()
            if gw_title and not _is_placeholder_title(gw_title):
                s["title"] = gw_title
            gw_summary = (gw.get("session_summary") or "").strip()
            if gw_summary:
                s["session_summary"] = gw_summary
        # A web/adopted row is "running" only while its bound gateway session
        # has a live turn; a plain web row with no gateway binding is never
        # live (a web turn always runs through the gateway, so a live web turn
        # HAS a bound gw session in ``gw_by_hash``).
        s["live"] = bool(gw.get("live")) if gw else False
        # Durable session_state from the orchestrator tree (idle/streaming/
        # waiting/…). Lets the sidebar show Waiting vs Live correctly.
        s["state"] = (gw.get("state") or "idle") if gw else (s.get("state") or "idle")
        if gw:
            for key in activity_keys:
                if key in gw:
                    s[key] = gw[key]
        sessions.append(s)
        seen.add(s["id"])
        if s.get("id"):
            seen.add(s["id"])

    for h, gw in gw_by_hash.items():
        if h in seen:
            continue
        if scoped and h not in shared_ids:
            continue
        if agent and gw.get("agent") != agent:
            continue
        item = {
            "id": h,
            "title": gw.get("title") or "Session",
            "session_summary": gw.get("session_summary") or "",
            "agent": gw.get("agent", DEFAULT_AGENT),
            "model": gw.get("model", ""),
            "engine_model": "",
            "topic": gw.get("topic") or "general",
            "origin": "gateway",
            "updated_at": gw.get("updated_at", ""),
            "message_count": 0,
            "live": bool(gw.get("live")),
            "state": gw.get("state") or "idle",
        }
        for key in activity_keys:
            if key in gw:
                item[key] = gw[key]
        sessions.append(item)
        seen.add(h)

    # Shared gateway-native conversations older than the newest roots page
    # are still the member's: list them from the session index.
    if scoped:
        missing = sorted(h for h in shared_ids - seen if h)
        meta = _load_meta_headers(missing) if missing else {}
        for h in missing:
            m = meta.get(h) or {}
            if m.get("parent_session_id") or store.get(h) is not None:
                continue  # a sub-agent, or a Web row just past this page
            if agent and m.get("agent") and m.get("agent") != agent:
                continue
            sessions.append(
                {
                    "id": h,
                    "title": m.get("title") or "Shared conversation",
                    "session_summary": "",
                    "agent": m.get("agent") or DEFAULT_AGENT,
                    "model": m.get("model") or "",
                    "engine_model": "",
                    "topic": "general",
                    "origin": "gateway",
                    "updated_at": "",
                    "message_count": 0,
                    "live": False,
                    "state": "idle",
                }
            )

    # Enforce the cap on the MERGED result (newest-first) so ``limit`` is a true
    # total, not a per-source bound — otherwise the count could reach 2*limit
    # and the ``capped`` flag would under-report.
    sessions = overlay_session_list_from_meta(sessions)
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    capped = sessions[:limit]
    _stamp_sharing(capped, viewer)
    return _resolve_session_topics(capped)[0]


def _stamp_sharing(rows, viewer):
    """Mark each sidebar row private/public and, when someone else owns it,
    whose it is. One bulk read of the sharing table."""
    from sharedagents_web import access

    shares = sharing_store.get_many(r.get("id") for r in rows)
    admin = access.is_admin(viewer)
    owners: dict[str, str] = {}
    for row in rows:
        sharing = shares.get(str(row.get("id") or ""))
        row["visibility"] = sharing.visibility if sharing else "private"
        owner = access.row_owner(row) if "owner" in row else ""
        # Only the owner or an admin may delete / re-share it.
        row["can_manage"] = admin or (bool(owner) and owner == viewer)
        if owner and owner != viewer:
            if owner not in owners:
                owners[owner] = auth_store.display_name(owner)
            row["owner_email"] = owner
            row["owner_name"] = owners[owner]


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Agent monitor — live "agent → active sessions → subagents" tree
# ---------------------------------------------------------------------------
#
# Backend authority is the gateway's ``/control/v1/orchestrator/tree`` route
# (one read over the shared sessions.db). The web layer carries ZERO
# aggregation logic — it proxies the snapshot for the one-shot fetch and, for
# the live view, pushes the gateway snapshot over SSE on a tight cadence. We
# resend a full snapshot only when it actually changed (cheap hash compare),
# emitting a named ``hb`` heartbeat event otherwise so the connection (and any
# proxy) stays warm without churning the DOM, and so the client can treat the
# heartbeat as a liveness signal. The client diffs by session hash, so an
# unchanged tree costs one tiny frame on the wire and zero re-render.

# Hard cap on a single SSE connection's lifetime. EventSource auto-reconnects,
# so this just bounds the per-connection Flask worker thread (defends against
# leaked streams from abandoned tabs behind a tunnel).
_TREE_STREAM_MAX_SECONDS = 600
_TREE_STREAM_INTERVAL = 1.5
_TREE_DEFAULT_LIMIT = 200
_TREE_CACHE_TTL_S = 1.0
_tree_cache_lock = threading.Lock()
_tree_cache: dict[tuple, tuple[float, object]] = {}


def _parse_tree_limit(raw: str | None) -> int:
    if not raw:
        return _TREE_DEFAULT_LIMIT
    try:
        return max(1, min(int(raw), 5000))
    except (TypeError, ValueError):
        return _TREE_DEFAULT_LIMIT


def _tree_cache_scope(headers: dict | None):
    """Return the cache scope for gateway authorization headers.

    Unauthenticated/local calls may share the short-lived snapshot.  A
    principal-scoped request must be partitioned by the principal, otherwise
    one user's conversation tree could be served to another user during the
    cache TTL.  Unknown headers are deliberately not cached: they may carry
    opaque authorization material whose semantics we must not guess.
    """
    if not headers:
        return "shared"
    principal = next(
        (value for key, value in headers.items() if str(key).lower() == "x-sharedagents-principal"),
        None,
    )
    if principal:
        return ("principal", str(principal))
    return None


def _fetch_tree(
    include_all: bool = False,
    *,
    limit: int | None = None,
    refresh: bool = False,
    include_approvals: bool = False,
    session_id: str = "",
    headers: dict | None = None,
    retry_unavailable: bool | None = None,
):
    """Fetch the orchestrator tree snapshot from the gateway. None on failure."""
    params: list[str] = []
    if include_all:
        params.append("all=1")
    if limit is not None:
        params.append(f"limit={int(limit)}")
    if refresh:
        params.append("refresh=1")
    if include_approvals:
        params.append("approvals=1")
    if session_id:
        params.append("session=" + quote(str(session_id), safe=""))
    path = "/control/v1/orchestrator/tree"
    if params:
        path += "?" + "&".join(params)
    cache_scope = _tree_cache_scope(headers)
    cache_key = (
        bool(include_all),
        int(limit) if limit is not None else 0,
        bool(refresh),
        bool(include_approvals),
        str(session_id or ""),
        cache_scope,
    )
    now = time.monotonic()
    if not refresh and cache_scope is not None:
        with _tree_cache_lock:
            hit = _tree_cache.get(cache_key)
            if hit and (now - hit[0]) < _TREE_CACHE_TTL_S:
                return hit[1]
    request_kwargs = {"timeout": 10}
    if headers:
        request_kwargs["headers"] = headers
    if retry_unavailable is not None:
        request_kwargs["retry_unavailable"] = retry_unavailable
    resp = gateway_request("GET", path, **request_kwargs)
    tree = None
    if resp is not None and resp.status_code == 200:
        try:
            tree = resp.json()
        except ValueError:
            tree = None
    if tree is not None and cache_scope is not None:
        with _tree_cache_lock:
            _tree_cache[cache_key] = (now, tree)
    return tree


# ---------------------------------------------------------------------------
# Goals dashboard — durable goals portfolio (outlive sessions)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Blueprint registration (Task 2 de-monolith)
# ---------------------------------------------------------------------------


def register_blueprints(app):
    """Register every route blueprint onto ``app`` (idempotent).

    The route groups formerly defined here as import-time ``@app.route``
    handlers now live in ``sharedagents_web.routes.*`` as Flask Blueprints. This
    imports that package (which pulls each blueprint's ``from
    sharedagents_web.server import <names>`` against the now-complete ``server``
    module) and registers them. Guarded so a second ``create_app()`` call can
    never double-register (Flask raises on a duplicate blueprint name).
    """
    if getattr(app, "_sharedagents_blueprints_registered", False):
        return
    from sharedagents_web.routes import ALL_BLUEPRINTS

    for bp in ALL_BLUEPRINTS:
        app.register_blueprint(bp)
    app._sharedagents_blueprints_registered = True


def create_app():
    """App factory — return the configured module-level ``app`` with all route
    blueprints registered.

    ``app`` is still built (and its middleware/static routes attached) at module
    import scope, so ``from sharedagents_web.server import app`` keeps working exactly
    as before. This factory is the single blueprint-registration entry point and
    is invoked once at module import so the module-scope ``app`` is fully wired.
    """
    register_blueprints(app)
    # Do not start background threads here.  ``create_app`` runs in Gunicorn's
    # master before it forks workers.  If a fork catches the SDLC supervisor
    # while it owns (for example) ``dev_sdlc._store_lock``, the child inherits
    # a locked mutex whose owner thread no longer exists.  Every SDLC read in
    # that worker then hangs forever.  ``_start_postfork_services`` below is
    # wired to Gunicorn's post_worker_init hook instead.
    return app


_sdlc_supervisor_lock_fd = None


def _claim_sdlc_supervisor_lock() -> bool:
    """Exactly one gunicorn worker may run the SDLC sweep.

    ``start_sdlc_supervisor`` is process-local; two workers both calling it
    race the same blocked-run claim.  Hold an exclusive flock for the
    worker lifetime so the loser skips boot recovery.
    """
    global _sdlc_supervisor_lock_fd
    if oscompat.IS_WINDOWS:
        return False  # Dev SDLC remains POSIX-only, including direct callers.

    path = SESSION_DB_PATH.parent / ".sdlc_supervisor.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    if not oscompat.try_file_lock(fd):
        os.close(fd)
        return False
    _sdlc_supervisor_lock_fd = fd
    return True


def _reopen_prefork_sqlite() -> None:
    """Give each Gunicorn worker its own SQLite connections.

    ``store``, ``auth_store``, and ``web_state`` are constructed at import
    time in the master.  An inherited connection + ``BEGIN IMMEDIATE`` is
    the two-worker deadlock: the child waits on a reserved lock whose
    owner thread never existed in this process.  Must run even when SDLC
    recovery is skipped.
    """
    store.reopen_after_fork()
    auth_store.reopen_after_fork()
    web_state.reopen_after_fork()
    sharing_store.reopen_after_fork()


def _start_postfork_services(_worker=None) -> bool:
    """Start worker-local background services after Gunicorn has forked.

    Gunicorn calls this once in each worker.  The request hook in
    ``routes.dev`` remains the fallback for other WSGI servers and for a fresh
    install that has no enabled owner at worker boot.
    """
    if oscompat.IS_WINDOWS:
        return False
    _reopen_prefork_sqlite()
    from sharedagents_web.subscription_limits import ensure_pace_sampler
    from sharedagents_web.task_recap import ensure_sweeper

    ensure_sweeper()
    ensure_pace_sampler()
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Console entry point (``sharedagents-web`` / ``python -m sharedagents_web``).

    Runs Flask under Gunicorn on POSIX or threaded Waitress on Windows.
    The packaged console script and ``__main__`` share this entry point.
    """
    # CLI credential bootstrap: the ONLY way to create the
    # first native operator account, local and offline. Also serves as password
    # recovery (re-run with an existing email). No web-exposed registration
    # route exists, keeping the "no secondary bypass path" property auditable.
    argv = sys.argv[1:]
    if argv and argv[0] == "--create-user":
        import getpass

        rest = argv[1:]
        email = None
        password = None
        display_name = None
        role = None
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == "--role":
                # Admin | user. Without it a NEW account is
                # an admin only while no admin exists; a reset keeps the role.
                if i + 1 >= len(rest) or rest[i + 1] not in ("admin", "user"):
                    print("error: --role must be admin or user", file=sys.stderr)
                    raise SystemExit(2)
                role = rest[i + 1]
                i += 2
                continue
            if tok == "--password":
                if i + 1 >= len(rest):
                    print("error: --password requires a value", file=sys.stderr)
                    raise SystemExit(2)
                password = rest[i + 1]
                i += 2
                continue
            if tok == "--display-name":
                if i + 1 >= len(rest):
                    print("error: --display-name requires a value", file=sys.stderr)
                    raise SystemExit(2)
                display_name = rest[i + 1]
                i += 2
                continue
            if tok.startswith("-"):
                print(f"error: unknown flag {tok}", file=sys.stderr)
                raise SystemExit(2)
            if email is None:
                email = tok.strip().lower()
            i += 1
        if not email:
            email = input("email: ").strip().lower()
        if not email:
            print("error: email required", file=sys.stderr)
            raise SystemExit(2)
        if not password:
            password = (os.environ.get("SHAREDAGENTS_WEB_BOOTSTRAP_PASSWORD") or "").strip()
        if password:
            pw1 = password
        else:
            pw1 = getpass.getpass("password: ")
            pw2 = getpass.getpass("confirm password: ")
            if not pw1 or pw1 != pw2:
                print("error: passwords empty or do not match", file=sys.stderr)
                raise SystemExit(2)
        if not pw1:
            print("error: password required", file=sys.stderr)
            raise SystemExit(2)
        replace = auth_store.user_exists(email)
        if role is None and auth_store.account(email) is None:
            from sharedagents_web.native_auth import resolve_admin_source

            no_admin = not any(
                resolve_admin_source(auth_store, a["email"]) for a in auth_store.list_accounts() if not a["disabled"]
            )
            role = "admin" if no_admin else "user"
        auth_store.create_user(
            email,
            pw1,
            replace=True,
            display_name=display_name,
            role=role,
        )
        acct = auth_store.account(email) or {}
        print(
            f"{'updated' if replace else 'created'} native web user: {email} "
            f"(display name: {auth_store.display_name(email)}, role: {acct.get('role', '?')})"
        )
        raise SystemExit(0)

    if argv and argv[0] == "--set-display-name":
        if len(argv) != 3:
            print("usage: sharedagents-web --set-display-name EMAIL NAME", file=sys.stderr)
            raise SystemExit(2)
        if not auth_store.set_display_name(argv[1], argv[2]):
            print(f"error: no native web user {argv[1].strip().lower()}", file=sys.stderr)
            raise SystemExit(1)
        print(f"display name for {argv[1].strip().lower()}: {auth_store.display_name(argv[1])}")
        raise SystemExit(0)

    port = int(os.environ.get("SHAREDAGENTS_WEB_PORT", 8110))
    host = os.environ.get("SHAREDAGENTS_WEB_HOST", "127.0.0.1")

    # Localhost-only by default. A non-loopback bind is allowed ONLY when
    # explicitly opted in, via either:
    #   - SHAREDAGENTS_WEB_DEBUG=1     → local-network testing WITHOUT auth, or
    #   - SHAREDAGENTS_WEB_TRUST_PROXY=1 → behind a trusted reverse proxy that does
    #     auth (e.g. nginx auth_request); auth here stays ENFORCED. This lets
    #     the proxy reach the app over the LAN without disabling auth (which is
    #     what DEBUG would do).
    _allow_nonloopback = (
        os.environ.get("SHAREDAGENTS_WEB_DEBUG") == "1" or os.environ.get("SHAREDAGENTS_WEB_TRUST_PROXY") == "1"
    )
    if not _allow_nonloopback and host not in ("127.0.0.1", "localhost", "::1", "[::1]"):
        raise RuntimeError(
            f"SHAREDAGENTS_WEB_HOST must be a loopback address; got '{host}'. "
            "Set SHAREDAGENTS_WEB_TRUST_PROXY=1 (auth stays on, for a trusted reverse "
            "proxy) or SHAREDAGENTS_WEB_DEBUG=1 (no auth) to bind a non-loopback host."
        )

    _thread_count = int(os.environ.get("SHAREDAGENTS_WEB_THREADS", "32"))
    if oscompat.IS_WINDOWS:
        from waitress import serve

        from sharedagents_web.subscription_limits import ensure_pace_sampler
        from sharedagents_web.task_recap import ensure_sweeper

        ensure_sweeper()
        ensure_pace_sampler()
        print(f"[sharedagents-web] starting Waitress on http://{host}:{port}/ (1 process, {_thread_count} thread(s))")
        # No prefork hook or SDLC recovery on Windows. This inactivity limit
        # accommodates long SSE responses; it is not a worker-kill timeout.
        serve(
            app,
            host=host.strip("[]"),
            port=port,
            threads=_thread_count,
            channel_timeout=_SESSION_STREAM_READ_TIMEOUT + 30,
            **_waitress_proxy_kwargs(),
        )
        return

    from gunicorn.app.base import BaseApplication

    class _GunicornApp(BaseApplication):
        """Embedded gUnicorn that replaces the Flask dev server.

        GThread worker class with bounded threads — handles SSE streaming
        without unbounded thread growth (the Werkzeug dev server's default).
        """

        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                if key in self.cfg.settings and value is not None:
                    self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    # gUnicorn with gthread worker class — bounded threads per process.
    # Threads > 1 is required because SSE endpoints hold a worker thread for the
    # stream's lifetime (up to 720s). Default 32 (was 8): each session switch
    # opens an SSE relay that only frees its thread when it next tries to yield
    # to the now-closed browser — at the gateway's heartbeat. Rapid session
    # switching parked all 8 threads, so a subsequent plain GET (e.g. clicking
    # the parent "main" session) queued in the backlog for many seconds — the
    # "parent loads very slowly after many switches" bug. 32 raises the ceiling
    # 4x; a per-stream disconnect/heartbeat fix is the deeper change.
    # Default two workers so WAL SessionStore / web_state get a real
    # multi-process workout. Session get/list/delete read SQLite, not the
    # process-local cache. Residual: SDLC recovery still starts per worker.
    _worker_count = int(os.environ.get("SHAREDAGENTS_WEB_WORKERS", "2"))
    print(
        f"[sharedagents-web] starting gUnicorn on http://{host}:{port}/ "
        f"({_worker_count} worker(s), {_thread_count} thread(s))"
    )
    _GunicornApp(
        app,
        {
            "bind": f"{host}:{port}",
            "workers": _worker_count,
            "threads": _thread_count,
            "worker_class": "gthread",
            # Starting this from create_app() is pre-fork and can strand Python
            # locks in the worker.  Every autonomous service starts worker-local.
            "post_worker_init": _start_postfork_services,
            "timeout": _SESSION_STREAM_READ_TIMEOUT + 30,  # 750s — exceed max stream
            "keepalive": 60,
            "accesslog": "-",
            "errorlog": "-",
            "loglevel": "info",
        },
    ).run()


# Wire the route blueprints onto the module-scope ``app`` at import time so
# ``from sharedagents_web.server import app`` yields a fully-assembled application
# (import-time route registration preserved, exactly as the old ``@app.route``
# decorators did). Runs after every helper/const is defined, so each blueprint's
# ``from sharedagents_web.server import <names>`` resolves cleanly.
create_app()

# ---------------------------------------------------------------------------
# Import-surface compatibility (de-monolith): the route handlers below were
# defined in server.py before Task 2 split them into blueprints. Re-export them
# so any historical ``from sharedagents_web.server import <handler>`` keeps resolving.
# Placed after create_app() so every route module is already imported (no cycle).
# ---------------------------------------------------------------------------
from .routes.agents import (  # noqa: E402,F401
    agents_monitor,
    api_agents_tree,
    api_agents_tree_stream,
)
from .routes.artifacts import (  # noqa: E402,F401
    _artifact_gateway_session,
    _artifact_session_dir,
    _safe_artifact_path,
    api_session_artifact,
    serve_artifact,
)
from .routes.auth import (  # noqa: E402,F401
    _cookie_is_secure,
    api_login,
    api_logout,
    api_verify,
    login_page,
)
from .routes.chat import (  # noqa: E402,F401
    _markitdown_bin,
    api_chat,
    api_extract_document,
    api_health,
    index,
)
from .routes.commands import (  # noqa: E402,F401
    api_command,
    api_commands,
)
from .routes.goals import (  # noqa: E402,F401
    _api_goal_action,
    _reconcile_goal_views,
    _web_goal_action,
    _web_goal_detail,
    _web_load_agent,
    _web_supervise_goal,
    api_budget,
    api_goals_archive,
    api_goals_create,
    api_goals_events,
    api_goals_get,
    api_goals_list,
    api_goals_pause,
    api_goals_restart,
    api_goals_resume,
    api_goals_session_tail,
    api_goals_set_budget,
    goals_page,
)
from .routes.presence import (  # noqa: E402,F401
    api_continue_here,
    api_presence,
    api_session_handoff,
)
from .routes.sessions import (  # noqa: E402,F401
    _delete_confirmed,
    _resolve_gw_hash,
    api_approvals,
    api_create_session,
    api_delete_session,
    api_get_session,
    api_list_sessions,
    api_models,
    api_session_answer,
    api_session_cancel,
    api_session_live,
    api_session_permission,
    api_session_prompt,
    api_session_queue_add,
    api_session_queue_delete,
    api_session_queue_list,
    api_session_queue_update,
    api_session_stream,
)
from .routes.settings import (  # noqa: E402,F401
    api_settings_gateway,
)
from .routes.topics import (  # noqa: E402,F401
    api_create_topic,
    api_list_topics,
    topics_page,
)
from .routes.voice import (  # noqa: E402,F401
    api_transcribe,
)

if __name__ == "__main__":
    main()
