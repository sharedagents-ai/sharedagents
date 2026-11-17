"""aiohttp server: ``GET /v1/realtime`` WebSocket + the voice test page.

Auth mirrors the web UI (``web/server.py``): the dashboard JWT cookie is
forwarded to the dashboard ``/verify-token`` endpoint. Because the page is
served same-origin, the browser sends that cookie on BOTH the page request
and the WS handshake, so the same login that gates ``/sharedagentsweb`` gates
voice — no separate token.

``SHAREDAGENTS_REALTIME_DEBUG=1`` skips auth (loopback/tunnel dev only).

A realtime session persists across a connection loss: a reconnecting client
presents its server-issued resume TOKEN in the ``Sec-WebSocket-Protocol``
handshake header so it rebinds to its existing conversation instead of a fresh
session.

The token rides in a handshake HEADER (a subprotocol offer), NOT the URL query
string: a bearer capability in the URL leaks via aiohttp's access
log (which records the request line/query), browser history, and ``Referer``.
``Sec-WebSocket-Protocol`` is absent from the default access-log format
(``%a %t "%r" %s %b "%{Referer}i" "%{User-Agent}i"``) and from history/Referer,
so the token no longer touches any logged surface.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import httpx
from aiohttp import WSMsgType, web

from .cascade import Cascade
from .config import RealtimeConfig, load_config
from .meeting import MeetingSession, MeetingTranscriptStore
from .session import RealtimeSession

_LOG = logging.getLogger("sharedagents.realtime.server")

# WebSocket subprotocol negotiation. The client
# always offers the base protocol so the server has one to echo back, and — only
# when reconnecting — a second offer that carries the resume TOKEN in its own
# token characters (``sharedagents.resume.<token>``). The server reads the token from
# the offered list (a handshake header, never logged) and echoes ONLY the base
# protocol, so the token is never reflected back either. token_urlsafe emits
# ``[A-Za-z0-9_-]`` and the ``rt_`` prefix — all valid RFC 6455 subprotocol token
# chars — so the token needs no encoding to ride here.
_REALTIME_SUBPROTOCOL = "sharedagents.realtime.v1"
_RESUME_SUBPROTOCOL_PREFIX = "sharedagents.resume."


def _resume_token_from_request(request: web.Request) -> str | None:
    """Extract the resume token from the ``Sec-WebSocket-Protocol`` offer, or
    ``None`` if the client offered no resume subprotocol. Kept OUT of the URL
    query so the bearer token never lands in the access log / history / Referer.
    """
    raw = request.headers.get("Sec-WebSocket-Protocol", "")
    for proto in (p.strip() for p in raw.split(",")):
        if proto.startswith(_RESUME_SUBPROTOCOL_PREFIX):
            token = proto[len(_RESUME_SUBPROTOCOL_PREFIX) :]
            return token or None
    return None


# ── Resume-token log redaction ───────────────
#
# The resume token is a BEARER capability. Commit 3de456e moved it out of the WS
# URL into a ``Sec-WebSocket-Protocol`` subprotocol offer so it no longer touches
# the access log / browser history / Referer. One residual server-side surface
# remains: when a MALFORMED / stale / attacker client offers ONLY
# ``sharedagents.resume.<token>`` and OMITS the base ``sharedagents.realtime.v1``, aiohttp's
# handshake negotiator (``aiohttp.websocket`` logger, aiohttp/web_ws.py) finds no
# overlap and emits a WARNING whose args include the raw offered-protocol list —
# token and all — BEFORE our handler ever runs. Legit clients offer the base
# first and match before the resume offer is examined, so they never hit this
# path; but the token must not reach a log line regardless of offer shape.
#
# This filter is the definitive, offer-shape-independent close: it scrubs every
# resume-token form out of a record's fully-rendered message and clears the
# args, so nothing downstream (any handler, any formatter, up the whole logger
# tree) can reintroduce the token. token_urlsafe emits ``[A-Za-z0-9_-]`` and both
# the ``sharedagents.resume.`` offer prefix and the bare ``rt_`` token prefix use that
# same alphabet, so one character class covers both shapes.
_RESUME_LEAK_RE = re.compile(r"sharedagents\.resume\.[A-Za-z0-9_\-]+")
_RAW_TOKEN_RE = re.compile(r"\brt_[A-Za-z0-9_\-]{24,}")

# aiohttp loggers that can emit the handshake / no-overlap warning. The CONFIRMED
# emitter is ``aiohttp.websocket`` (ws_logger.warning in web_ws.py:_handshake);
# the others are attached defensively so a future aiohttp that routes the same
# warning through a different logger is still covered.
_REDACTOR_LOGGERS = (
    "aiohttp.websocket",  # confirmed no-overlap warning emitter
    "aiohttp.web",
    "aiohttp.server",
    "aiohttp.access",
    # Belt-and-suspenders: our OWN realtime loggers. The `session.config` event
    # payload carries `resume_token` by design (the client needs it to
    # reconnect); if any code path logs a raw event payload through one of these
    # loggers, the redactor scrubs the token there too. Python logging filters
    # do NOT propagate to descendant loggers, so the concrete child logger names
    # are listed here (not the `sharedagents.realtime` parent). The rt_<token> form is
    # matched by _RAW_TOKEN_RE, so a real gateway session id (different shape) is
    # not over-redacted.
    "sharedagents.realtime.session",
    "sharedagents.realtime.server",
    "sharedagents.realtime.cascade",
)


def _scrub_resume_token(text: str) -> str:
    """Redact every resume-token form out of ``text`` (idempotent, never raises
    for a str input)."""
    text = _RESUME_LEAK_RE.sub("sharedagents.resume.<redacted>", text)
    text = _RAW_TOKEN_RE.sub("rt_<redacted>", text)
    return text


class _ResumeTokenRedactor(logging.Filter):
    """Scrub any resume-token substring out of a log record before it is emitted.

    Renders the record's message with its args applied, redacts the token, and —
    only if something was actually redacted — replaces ``record.msg`` with the
    scrubbed text and clears ``record.args`` so re-formatting downstream can't
    reintroduce the token (an empty ``args`` makes ``LogRecord.getMessage`` skip
    ``%`` interpolation entirely). Defensive: any failure drops the args rather
    than risk leaking the token. Always returns True — it redacts, never filters
    the record out."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — un-renderable args: drop them, keep the record
            record.args = ()
            try:
                record.msg = _scrub_resume_token(str(record.msg))
            except Exception:  # noqa: BLE001
                record.msg = "<redacted log record>"
            return True
        scrubbed = _scrub_resume_token(msg)
        if scrubbed != msg:
            record.msg = scrubbed
            record.args = ()
        return True


def _install_resume_redaction() -> None:
    """Attach the resume-token redactor to every aiohttp logger that could emit
    the handshake/no-overlap warning. Idempotent — never attaches a second
    copy."""
    redactor = _ResumeTokenRedactor()
    for name in _REDACTOR_LOGGERS:
        lg = logging.getLogger(name)
        if not any(isinstance(f, _ResumeTokenRedactor) for f in lg.filters):
            lg.addFilter(redactor)


class ResumeRegistry:
    """Server-issued resume tokens → gateway session ids (authz).

    A resume handle on the WS handshake is otherwise IDENTITY-BLIND: this is a
    personal single-owner system whose realtime socket carries no per-session
    owner (the gateway attributes a session to its *agent*, not the browser
    user), so ``session_exists()==200`` alone lets any authenticated client
    rebind voice to a conversation it merely knows/guesses the sid of.
    Instead, when a voice session binds a gateway conversation we mint an
    unguessable token and hand it ONLY to that client (in ``session.config``);
    on reconnect the client echoes the TOKEN, and only a token this server
    issued resolves back to a gateway sid. A raw/guessed gateway sid resolves to
    ``None`` → fresh session (fail closed). Bounded by TTL + max size so tokens
    from long-dead sessions can't accumulate.
    """

    def __init__(self, ttl_s: float = 24 * 3600, max_entries: int = 4096) -> None:
        self._ttl = ttl_s
        self._max = max_entries
        # token -> (gateway_sid, issued_monotonic-ish wall time)
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()

    def issue(self, sid: str) -> str:
        """Mint a fresh unguessable token bound to ``sid`` and remember it."""
        self._prune()
        token = "rt_" + secrets.token_urlsafe(32)
        self._store[token] = (sid, time.time())
        self._store.move_to_end(token)
        while len(self._store) > self._max:
            self._store.popitem(last=False)  # evict oldest
        return token

    def resolve(self, token: str | None) -> str | None:
        """Return the gateway sid a token was issued for, or ``None`` if the
        token is absent, unknown, or expired. ``None`` means: do NOT rebind."""
        if not token:
            return None
        entry = self._store.get(token)
        if entry is None:
            return None
        sid, ts = entry
        if time.time() - ts > self._ttl:
            self._store.pop(token, None)
            return None
        return sid

    def _prune(self) -> None:
        now = time.time()
        for tok in [t for t, (_s, ts) in self._store.items() if now - ts > self._ttl]:
            self._store.pop(tok, None)


def _sharedagents_web_static() -> Path | None:
    # Installed wheel / editable checkout: assets ride inside the
    # sharedagents_web package (src/sharedagents_web/static).
    try:
        from sharedagents_web import STATIC_DIR

        return STATIC_DIR
    except Exception:  # noqa: BLE001 — package optional / import-time issue
        return None


_STATIC_CANDIDATES = tuple(
    p
    for p in (
        _sharedagents_web_static(),
        # Legacy source-tree fallback (pre-packaging repos): repo/web/static.
        Path(__file__).resolve().parents[2] / "web" / "static",
    )
    if p is not None
)


def _static_dir() -> Path | None:
    for p in _STATIC_CANDIDATES:
        if (p / "realtime.html").exists():
            return p
    return None


def _dashboard_url() -> str:
    return (
        os.environ.get("SHAREDAGENTS_DASHBOARD_URL") or os.environ.get("DASHBOARD_URL") or "http://127.0.0.1:20100"
    ).rstrip("/")


def _native_token_ok(request: web.Request) -> bool:
    """Verify the SAME native login cookie the web UI issues, against the SAME
    local SQLite store (one credential, both transports), and
    admit admins only — a web *user* account never reaches voice.

    Local-only, no network — so voice authenticates even with the dashboard
    unreachable, exactly like the web transport.
    """
    try:
        from sharedagents_web.native_auth import COOKIE_NAME

        raw = request.cookies.get(COOKIE_NAME)
        if not raw:
            return False
        store = _auth_store()
        email = store.verify_token(raw)
        if not email:
            return False
        return _is_admin(store, email)
    except Exception:
        return False


_NATIVE_AUTH_STORE = None


def _auth_store():
    """The web's native account store (same SQLite file as the web UI)."""
    global _NATIVE_AUTH_STORE
    if _NATIVE_AUTH_STORE is None:
        from pathlib import Path

        from sharedagents_web.native_auth import NativeAuthStore

        db_path = os.environ.get("SHAREDAGENTS_WEB_DB", "~/.sharedagents/web/sessions.db")
        _NATIVE_AUTH_STORE = NativeAuthStore(Path(db_path).expanduser())
    return _NATIVE_AUTH_STORE


def _is_admin(store, email: str) -> bool:
    """Voice drives agents directly, outside the web's per-user grants and
    conversation sharing: admins only, whatever authenticated them.
    """
    from sharedagents_web.native_auth import resolve_admin_source

    return resolve_admin_source(store, email) is not None


async def _authorized(request: web.Request) -> bool:
    """Native-first auth: verify the local native login cookie; only if
    that declines AND external auth is enabled fall back to dashboard
    delegation. Fail-closed on a missing cookie or an unreachable dashboard."""
    if os.environ.get("SHAREDAGENTS_REALTIME_DEBUG") == "1":
        return True
    # Primary: native token against the local store — no external dependency.
    if _native_token_ok(request):
        return True
    # External auth (dashboard delegation) is optional and OFF by default.
    if os.environ.get("SHAREDAGENTS_EXTERNAL_AUTH") != "1":
        return False
    cookie = request.headers.get("Cookie", "")
    if not cookie:
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{_dashboard_url()}/verify-token",
                headers={"Cookie": cookie, "X-Original-URI": "/sharedagentsrt/"},
            )
    except httpx.HTTPError:
        return False  # dashboard unreachable → deny
    email = str(r.headers.get("X-User-Email") or "").strip().lower()
    if r.status_code != 200 or not email:
        return False
    try:
        return _is_admin(_auth_store(), email)
    except Exception:  # noqa: BLE001 — fail closed
        return False


class _InteractiveSession(Protocol):
    """The interactive-session surface the router drives (start/close/route)."""

    id: str

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def handle_event(self, event: dict, /) -> None: ...


class _MeetingSession(Protocol):
    """The passive meeting-session surface the router drives (route/close)."""

    @property
    def finished(self) -> bool: ...

    async def handle_event(self, event: dict, /) -> None: ...
    async def close(self) -> None: ...


def _factory_accepts(factory: Callable[..., object], param: str) -> bool:
    """True when ``factory`` can take ``param`` as a keyword argument.

    That is, ``param`` is declared positional-or-keyword or keyword-only, or the
    signature has ``**kwargs``. A positional-only ``param`` does not count:
    passing it by keyword raises ``TypeError``. Lets the router pass
    ``asr_language`` only to factories that accept it, so legacy
    ``(send, cascade, *, store)`` meeting factories keep working.
    """
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    for p in sig.parameters.values():
        if p.name == param and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            return True
        if p.kind is p.VAR_KEYWORD:  # **kwargs accepts any keyword
            return True
    return False


class RealtimeConnectionRouter:
    """Own exactly one interactive or meeting session for a WebSocket.

    Factories are injectable so the structural exclusion can be verified
    without constructing gateway/network clients.
    """

    def __init__(
        self,
        send,
        cascade,
        cfg,
        registry,
        meeting_store,
        *,
        resume_session: str | None = None,
        interactive_factory: Callable[..., _InteractiveSession] = RealtimeSession,
        meeting_factory: Callable[..., _MeetingSession] = MeetingSession,
    ) -> None:
        self._send = send
        self._cascade = cascade
        self._cfg = cfg
        self._registry = registry
        self._meeting_store = meeting_store
        self._interactive_factory = interactive_factory
        self._meeting_factory = meeting_factory
        self.interactive: _InteractiveSession | None = self._new_interactive(resume_session)
        self.meeting: _MeetingSession | None = None

    def _new_interactive(self, resume_session: str | None = None):
        return self._interactive_factory(
            self._send,
            self._cascade,
            self._cfg,
            resume_session=resume_session,
            resume_registry=self._registry,
        )

    def _create_meeting(self) -> _MeetingSession:
        """Build the meeting session, preserving the legacy factory contract.

        The default :class:`MeetingSession` accepts ``asr_language`` (so the
        configured language reaches its noise guard), but factories written
        against the older ``(send, cascade, *, store)`` contract do not. Pass
        ``asr_language`` only when the factory can take it as a keyword (see
        :func:`_factory_accepts`) so legacy factories keep working unchanged.
        """
        kwargs: dict[str, object] = {"store": self._meeting_store}
        if _factory_accepts(self._meeting_factory, "asr_language"):
            kwargs["asr_language"] = self._cfg.asr_language
        return self._meeting_factory(self._send, self._cascade, **kwargs)

    async def start(self) -> None:
        await self.interactive.start()

    async def handle_event(self, event: dict) -> None:
        if event.get("type") == "meeting.start" and self.meeting is None:
            if self.interactive is not None:
                await self.interactive.close()
                self.interactive = None
            self.meeting = self._create_meeting()
            await self.meeting.handle_event(event)
        elif self.meeting is not None:
            await self.meeting.handle_event(event)
        elif self.interactive is not None:
            await self.interactive.handle_event(event)
        await self._restore_interactive_if_finished()

    async def _restore_interactive_if_finished(self) -> None:
        if self.meeting is None or not self.meeting.finished:
            return
        self.meeting = None
        self.interactive = self._new_interactive()
        await self.interactive.start()

    async def close(self) -> None:
        if self.meeting is not None:
            await self.meeting.close()
            self.meeting = None
        if self.interactive is not None:
            await self.interactive.close()
            self.interactive = None


async def _ws_handler(request: web.Request) -> web.StreamResponse:
    """WebSocket entry point for one realtime voice session.

    A reconnecting client presents its server-issued resume TOKEN in the
    ``Sec-WebSocket-Protocol`` handshake header (subprotocol offer) so the new
    session rebinds to its prior conversation (auto-reconnect + resend of
    buffered audio on the client side); a fresh connect omits it.
    """
    if not await _authorized(request):
        raise web.HTTPUnauthorized(text="dashboard login required")
    # Echo ONLY the base subprotocol: a legit client always offers it FIRST, so
    # aiohttp matches it and returns before it ever examines the resume-token
    # offer — the token is never handed to the negotiator's warning path, and is
    # never reflected back in the response ``Sec-WebSocket-Protocol`` header.
    #
    # We deliberately KEEP ``protocols=(_REALTIME_SUBPROTOCOL,)`` rather than drop
    # negotiation: dropping it (``protocols=()``) would make EVERY resume offer —
    # including legit reconnects — fail the overlap check and fire aiohttp's
    # token-bearing no-overlap warning, i.e. strictly worse. With the base
    # accepted, only a MALFORMED / base-missing offer reaches that warning, and
    # the _ResumeTokenRedactor filter (installed in make_app) is the guarantee
    # that even then the token never lands on a log line. The warning fires INSIDE
    # ws.prepare(), before this handler regains control, so a pre-emit logging
    # filter is the only place the leak can be closed comprehensively.
    ws = web.WebSocketResponse(heartbeat=30, protocols=(_REALTIME_SUBPROTOCOL,))
    await ws.prepare(request)

    cfg: RealtimeConfig = request.app["cfg_factory"]()
    cascade = Cascade(cfg)
    send_lock = asyncio.Lock()

    async def send(event: dict) -> None:
        async with send_lock:
            if not ws.closed:
                await ws.send_str(json.dumps(event, ensure_ascii=False))

    # The ``Sec-WebSocket-Protocol`` offer carries
    # a server-ISSUED resume TOKEN, not a raw gateway sid (and not a URL query,
    # which would leak it to the access log). Resolve it against the registry: a
    # token we minted for this browser maps back to its gateway sid (rebind);
    # anything else — a guessed/replayed/foreign sid, or no token — resolves to
    # None and we fall back to a FRESH session. This is what proves the caller
    # originated the conversation before we rebind voice to it (session_exists
    # alone does NOT — it only checks existence, not entitlement).
    registry: ResumeRegistry = request.app["resume_registry"]
    resume_token = _resume_token_from_request(request)
    resume = registry.resolve(resume_token)
    router = RealtimeConnectionRouter(
        send,
        cascade,
        cfg,
        registry,
        request.app["meeting_store"],
        resume_session=resume,
    )
    _LOG.info(
        "session %s open (peer=%s, resume=%s)", router.interactive.id, request.remote, "token→sid" if resume else "-"
    )
    await router.start()
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    ev = json.loads(msg.data)
                except ValueError:
                    await send({"type": "error", "error": {"message": "invalid JSON frame"}})
                    continue
                await router.handle_event(ev)
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        await router.close()
        await cascade.aclose()
        _LOG.info("realtime socket closed")
    return ws


async def _page(request: web.Request) -> web.Response:
    # Gate the page too — the dashboard cookie rides on every same-origin
    # request (incl. the relative <script src> fetch), so this is uniform with
    # how /sharedagentsweb gates its shell. JS itself carries no secrets, but a
    # consistent 401 keeps unauthenticated browsers out entirely.
    if not await _authorized(request):
        raise web.HTTPUnauthorized(text="dashboard login required")
    static = _static_dir()
    if static is None:
        return web.Response(status=404, text="realtime.html not found — run from a repo checkout")
    # Serve only the known page scripts by name; anything else → the shell.
    leaf = request.path.rsplit("/", 1)[-1]
    name = leaf if leaf in {"realtime.js", "wakeword.js"} else "realtime.html"
    ctype = "application/javascript" if name.endswith(".js") else "text/html"
    body = (static / name).read_bytes()
    if name == "realtime.html":
        # Cache-bust the JS by its mtime so a browser that cached an older
        # realtime.js (back when a CDN sent max-age) is forced to refetch the
        # current one — a stale client JS otherwise survives the no-store fix.
        rev = int((static / "realtime.js").stat().st_mtime)
        body = body.replace(b'src="realtime.js"', f'src="realtime.js?v={rev}"'.encode())
        wrev = int((static / "wakeword.js").stat().st_mtime)
        body = body.replace(b'src="wakeword.js"', f'src="wakeword.js?v={wrev}"'.encode())
        # White-label: substitute the operator's chosen product name
        # (``SHAREDAGENTS_WEB_TITLE``, same env var the Flask web UI uses) for the
        # ``__SITE_TITLE__`` placeholder in the page's <title> and <h1>. The
        # realtime page is served as a static byte body (no Jinja), so the
        # substitution is a literal byte replace — mirroring the cache-bust
        # pattern above. Defaults to "sharedagents" when the env var is unset.
        _site_title = os.environ.get("SHAREDAGENTS_WEB_TITLE") or "sharedagents"
        # HTML-escape: the placeholder appears in element text (<title>) AND in
        # an attribute value (meta[name=sharedagents-site-title] content="..."), so an
        # unescaped quote or angle bracket would break out of either context.
        import html as _html

        body = body.replace(
            b"__SITE_TITLE__",
            _html.escape(_site_title, quote=True).encode(),
        )
    # no-store: these responses sit behind dashboard auth, so a CDN (Cloudflare)
    # must NOT cache an authenticated 200 and then replay it to anonymous
    # clients. Without this header CF applies a default TTL to .js by extension.
    return web.Response(body=body, content_type=ctype, headers={"Cache-Control": "no-store"})


async def _client_config(request: web.Request) -> web.Response:
    """Advertise client-side capture settings to the browser. The browser uses
    this to tune the mandatory local wake gate. A disabled/unavailable engine
    fails closed to explicit push-to-talk. Auth-gated like the page."""
    if not await _authorized(request):
        raise web.HTTPUnauthorized(text="dashboard login required")
    cfg: RealtimeConfig = request.app["cfg_factory"]()
    return web.json_response(
        {
            "wake": {
                "enabled": bool(cfg.client_wake),
                "keyword": cfg.wake_keyword,
                "phrase": cfg.wake_phrase,
                "supports_literal_word": bool(cfg.wake_supports_literal_word),
                "threshold": cfg.wake_threshold,
                "preroll_ms": cfg.wake_preroll_ms,
                "cooldown_ms": cfg.wake_cooldown_ms,
                "vad_gate": bool(cfg.wake_vad_gate),
            },
            "conversation_end_word": cfg.conversation_end_word if cfg.conversation_end_word else None,
        },
        headers={"Cache-Control": "no-store"},
    )


async def _health(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "service": "sharedagents-realtime"}, headers={"Cache-Control": "no-store"})


async def _meeting_list(request: web.Request) -> web.Response:
    """List stored meetings for explicit, later retrieval (never injection)."""
    if not await _authorized(request):
        raise web.HTTPUnauthorized(text="dashboard login required")
    store: MeetingTranscriptStore = request.app["meeting_store"]
    return web.json_response({"meetings": store.list_meetings()}, headers={"Cache-Control": "no-store"})


async def _meeting_get(request: web.Request) -> web.Response:
    """Retrieve chunk content only in response to this authenticated request."""
    if not await _authorized(request):
        raise web.HTTPUnauthorized(text="dashboard login required")
    store: MeetingTranscriptStore = request.app["meeting_store"]
    try:
        payload = store.retrieve(request.match_info["meeting_id"])
    except KeyError:
        raise web.HTTPNotFound(text="meeting not found") from None
    return web.json_response(payload, headers={"Cache-Control": "no-store"})


def make_app(cfg_factory=load_config) -> web.Application:
    # Install the resume-token log redactor
    # before any WS handshake can run, so aiohttp's no-overlap handshake warning
    # can never leak the token to a log line (see _ResumeTokenRedactor).
    _install_resume_redaction()
    app = web.Application()
    app["cfg_factory"] = cfg_factory
    # Shared across WS connections so a reconnect
    # (a NEW socket / RealtimeSession) can resolve a token minted by the ORIGINAL
    # connection. App-scoped, so it lives for the server's lifetime.
    app["resume_registry"] = ResumeRegistry()
    app["meeting_store"] = MeetingTranscriptStore()
    app.router.add_get("/v1/realtime", _ws_handler)
    app.router.add_get("/v1/realtime/clientconfig", _client_config)
    app.router.add_get("/v1/realtime/meetings", _meeting_list)
    app.router.add_get("/v1/realtime/meetings/{meeting_id}", _meeting_get)
    app.router.add_get("/", _page)
    app.router.add_get("/realtime.js", _page)
    app.router.add_get("/wakeword.js", _page)
    app.router.add_get("/health", _health)
    # Wake-word assets (ONNX models + onnxruntime-web WASM). These are public OSS
    # artifacts carrying no secrets, so they are served un-gated and cacheable —
    # large and immutable, we WANT a CDN/browser to cache them (unlike the
    # auth-gated page/JS which are no-store).
    static = _static_dir()
    if static is not None:
        if (static / "models").is_dir():
            app.router.add_static("/models/", static / "models")
        if (static / "vendor").is_dir():
            app.router.add_static("/vendor/", static / "vendor")
    return app


def main(host: str = "127.0.0.1", port: int = 8123) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    # Agent mode (cfg.agent set) gets its LLM turn from the gateway agent, so a
    # local chat stage is not required — only STT/TTS. Raw-cascade mode needs
    # all three.
    required = ("stt", "tts") if cfg.agent else ("stt", "tts", "chat")
    for name in ("stt", "tts", "chat"):
        stage = getattr(cfg, name)
        _LOG.info("stage %-4s → %s (%s)", name, stage.base_url or "UNCONFIGURED", stage.model or "-")
        if name in required and not stage.configured:
            raise SystemExit(f"stage {name!r} is unconfigured — see specs/realtime-voice.md")
    if cfg.agent:
        _LOG.info("agent mode → driving gateway agent %r at %s", cfg.agent, cfg.gateway_url)
    web.run_app(make_app(), host=host, port=port)
