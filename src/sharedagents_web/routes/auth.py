"""sharedagents_web.routes.auth — Native authentication routes (login page, login/logout/verify).

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

import html
import json
import math
import os
import secrets
import threading
import time

from flask import Blueprint, Response, g, jsonify, make_response, request

from sharedagents_web import server as _server  # call-time SITE_TITLE read (monkeypatch-friendly)
from sharedagents_web.server import _AUTH_COOKIE, BASE_PATH, auth_store, require_auth

bp = Blueprint("auth", __name__)


@bp.route("/api/verify")
@require_auth
def api_verify():
    """Verify user authentication — @require_auth already verified, just return user info."""
    return jsonify({"authenticated": True, "email": g.user_email})


_LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__SITE_TITLE__ — sign in</title>
<style>
  body{font-family:system-ui,sans-serif;background:#0e1116;color:#e6e6e6;
       display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}
  form{background:#171b22;padding:2rem;border-radius:12px;width:min(92vw,340px);
       box-shadow:0 8px 30px rgba(0,0,0,.4)}
  h1{font-size:1.1rem;margin:0 0 1rem}
  input{width:100%;box-sizing:border-box;padding:.6rem;margin:.35rem 0;
        background:#0e1116;border:1px solid #2a2f38;border-radius:6px;color:#e6e6e6}
  button{width:100%;padding:.6rem;margin-top:.6rem;background:#2f6feb;border:0;
         border-radius:6px;color:#fff;font-weight:600;cursor:pointer}
  .err{color:#f4736f;font-size:.85rem;min-height:1.1rem;margin-top:.4rem}
</style></head><body>
<form id="f">
  <h1>Sign in to __SITE_TITLE__</h1>
  <input id="email" type="email" placeholder="email" autocomplete="username" required>
  <input id="password" type="password" placeholder="password" autocomplete="current-password" required>
  <button type="submit">Sign in</button>
  <div class="err" id="err"></div>
</form>
<script nonce="__NONCE__">
async function doLogin(e){
  e.preventDefault();
  const err=document.getElementById('err'); err.textContent='';
  const r=await fetch((window.SHAREDAGENTS_BASE||'')+'/api/login',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({email:email.value,password:password.value})});
  if(r.ok){ location.href=(window.SHAREDAGENTS_BASE||'')+'/'; }
  else { const d=await r.json().catch(()=>({})); err.textContent=d.error||'Sign in failed'; }
  return false;
}
// Attach the handler programmatically: the global CSP forbids inline event
// handler attributes (onsubmit="..."), which a nonce cannot whitelist.
document.getElementById('f').addEventListener('submit', doLogin);
window.SHAREDAGENTS_BASE=__BASE_PATH__;
</script></body></html>"""


def _cookie_is_secure() -> bool:
    """Set the Secure flag when the connection is TLS (directly or via proxy).

    On plain-http loopback dev a Secure cookie would never be sent back, so it is
    scoped to actual TLS — the production deployment is always behind HTTPS.
    """
    if request.is_secure:
        return True
    return request.headers.get("X-Forwarded-Proto", "").lower() == "https"


@bp.route("/login")
def login_page():
    """Unauthenticated native login form.

    Browser routes that fail auth land here; API/SSE routes still return 401 JSON
    (handled by ``require_auth``). This page has NO registration path — first
    credential creation is CLI-only (``sharedagents-web --create-user``), keeping the
    "no secondary bypass path" property easy to audit.
    """
    external_redirect = _server.external_login_redirect()
    if external_redirect is not None:
        return external_redirect
    # Escape before substituting: unlike the Jinja templates (which autoescape
    # ``site_title``), this page is a raw string, so a product name containing
    # "<", "&" or a closing tag would corrupt or inject markup into the
    # UNAUTHENTICATED login page. quote=True also protects attribute contexts.
    # BASE_PATH is substituted as a JSON string literal (json.dumps) so it is
    # always a valid JS string, including when it is "" (no prefix).
    # This page is the only one with an inline <script>; the global CSP
    # forbids inline scripts, so issue a per-request nonce (read by the
    # _csp_after_request hook via g.csp_nonce) to allow exactly this one.
    nonce = secrets.token_urlsafe(16)
    g.csp_nonce = nonce
    body = (
        _LOGIN_PAGE.replace(
            "__SITE_TITLE__",
            html.escape(_server.SITE_TITLE, quote=True),
        )
        .replace("__BASE_PATH__", json.dumps(BASE_PATH))
        .replace("__NONCE__", nonce)
    )
    return Response(body, mimetype="text/html")


# Login throttle: after LOGIN_FREE_FAILURES misses, a client IP or an account
# is blocked for LOGIN_BLOCK_BASE_S, doubling per further miss up to
# LOGIN_BLOCK_MAX_S. A key is forgotten LOGIN_BLOCK_MAX_S after its last miss
# and an account is cleared by its successful login. At most _LOGIN_MAX_KEYS
# keys are kept (oldest evicted). Process-local: each web worker keeps its own
# counts.
LOGIN_FREE_FAILURES = 5
LOGIN_BLOCK_BASE_S = 30.0
LOGIN_BLOCK_MAX_S = 900.0
_LOGIN_MAX_KEYS = 10_000
_clock = time.monotonic
_login_lock = threading.Lock()
_login_failures: dict[
    tuple[str, str], tuple[int, float, float, int]
] = {}  # key -> (misses, last_miss, blocked_until, in_flight)


def _client_ip() -> str:
    """The peer address, or the ``X-Forwarded-For`` last hop when the peer is
    listed in ``SHAREDAGENTS_WEB_TRUSTED_PROXIES`` (comma-separated IPs).

    ``SHAREDAGENTS_WEB_TRUST_PROXY`` alone does not trust the header: the
    Docker image sets it just to bind a published port, which forwards
    client headers untouched.
    """
    peer = request.remote_addr or ""
    trusted = {p.strip() for p in os.environ.get("SHAREDAGENTS_WEB_TRUSTED_PROXIES", "").split(",") if p.strip()}
    if peer in trusted:
        hop = request.headers.get("X-Forwarded-For", "").split(",")[-1].strip()
        if hop:
            return hop
    return peer


def _login_begin(keys) -> int:
    """Seconds to wait, or 0 after reserving an in-flight slot on each key.

    Check and reservation share one locked step, and a key admits only as
    many concurrent attempts as it has misses left (one once blocked
    before), so a burst cannot outrun the threshold.
    """
    now = _clock()
    with _login_lock:
        entries = []
        for k in keys:
            misses, last, until, busy = _login_failures.get(k, (0, 0.0, 0.0, 0))
            if until > now:
                return math.ceil(until - now)
            if now - last > LOGIN_BLOCK_MAX_S:
                misses = 0
            if busy >= max(1, LOGIN_FREE_FAILURES - misses):
                return 1
            entries.append((k, (misses, last, until, busy + 1)))
        if len(_login_failures) > _LOGIN_MAX_KEYS - len(keys):
            for k in [k for k, v in _login_failures.items() if now - v[1] > LOGIN_BLOCK_MAX_S and not v[3]]:
                del _login_failures[k]
            idle = sorted((v[1], k) for k, v in _login_failures.items() if not v[3])
            while len(_login_failures) > _LOGIN_MAX_KEYS - len(keys):
                if not idle:
                    return 1  # table full of in-flight checks: never drop a reservation
                del _login_failures[idle.pop(0)[1]]
        _login_failures.update(entries)
    return 0


def _login_end(keys, ok: bool) -> None:
    """Release the slots; a miss counts on every key, a success clears the account.

    A key evicted at capacity meanwhile is not recreated, so the cap holds.
    """
    now = _clock()
    with _login_lock:
        for kind, key in keys:
            if (kind, key) not in _login_failures:
                continue
            misses, last, until, busy = _login_failures[(kind, key)]
            if ok:
                if kind == "account":  # forget its misses, keep other in-flight checks
                    misses, until = 0, 0.0
                if misses == 0 and busy <= 1:
                    _login_failures.pop((kind, key), None)
                    continue
            else:
                misses += 1
                last = now
                if misses >= LOGIN_FREE_FAILURES:
                    until = now + min(LOGIN_BLOCK_BASE_S * 2 ** (misses - LOGIN_FREE_FAILURES), LOGIN_BLOCK_MAX_S)
            _login_failures[(kind, key)] = (misses, last, until, max(busy - 1, 0))


@bp.route("/api/login", methods=["POST"])
def api_login():
    """Validate native credentials and issue a revocable device token.

    Fail-closed: bad credentials return 401 with no token. On success an opaque
    256-bit token is set as an HttpOnly cookie; only its SHA-256 hash is stored.
    Repeated misses per client IP or per account get 429 with Retry-After.
    """
    data = request.get_json(silent=True) or {}
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")
    if not email or not password:
        return jsonify({"error": "email and password required"}), 400
    keys = (("ip", _client_ip()[:64]), ("account", email[:254]))
    wait = _login_begin(keys)
    if wait:
        resp = make_response(jsonify({"error": "Too many attempts, try again later"}), 429)
        resp.headers["Retry-After"] = str(wait)
        return resp
    ok = False
    try:
        ok = auth_store.verify_credentials(email, password)
    finally:
        _login_end(keys, ok)
    if not ok:
        return jsonify({"error": "Invalid credentials"}), 401
    device_label = str(data.get("device_label") or request.headers.get("User-Agent", ""))[:120]
    raw = auth_store.issue_token(email, device_label=device_label)
    resp = make_response(jsonify({"authenticated": True, "email": email}))
    resp.set_cookie(
        _AUTH_COOKIE,
        raw,
        max_age=90 * 86400,
        httponly=True,
        secure=_cookie_is_secure(),
        samesite="Lax",
        path=(BASE_PATH or "/"),
    )
    return resp


@bp.route("/api/logout", methods=["POST"])
def api_logout():
    """Revoke the current native session token and clear the cookie."""
    raw = request.cookies.get(_AUTH_COOKIE)
    if raw:
        auth_store.revoke_token(raw)
    resp = make_response(jsonify({"ok": True}))
    resp.set_cookie(_AUTH_COOKIE, "", expires=0, path=(BASE_PATH or "/"))
    return resp
