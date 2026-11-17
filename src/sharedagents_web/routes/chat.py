"""sharedagents_web.routes.chat — Chat UI shell, chat SSE, health, durable uploads.

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

import json
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Blueprint, Response, g, jsonify, render_template, request, stream_with_context

import sharedagents_web.server as _server  # DASHBOARD_URL read live (test-patchable)
from sharedagents_core.attachments import sniff_image_mime
from sharedagents_core.ids import ID_PREFIXES, bare_id
from sharedagents_web.principal import (
    canonical_gateway_principal,
    may_adopt_gateway_session,
)
from sharedagents_web.server import (
    _CHAT_STREAM_MAX_SECONDS,
    _CHAT_WORKER_MAX_SECONDS,
    BASE_PATH,
    DEFAULT_AGENT,
    REALTIME_URL,
    SessionStore,
    current_gateway,
    ensure_gw_session,
    gateway_request,
    gw_session_history,
    require_auth,
    require_auth_page,
    store,
)
from sharedagents_web.uploads import save_upload, upload_directory

bp = Blueprint("chat", __name__)


_MARKITDOWN_BIN_DEFAULT = str(
    # chat.py lives at <checkout>/src/sharedagents_web/routes/chat.py.  The
    # project's runtime venv is at <checkout>/.venv, not <checkout>/src/.venv.
    Path(__file__).resolve().parents[3] / ".venv" / "bin" / "markitdown"
)


_MARKITDOWN_TIMEOUT = int(os.environ.get("SHAREDAGENTS_MARKITDOWN_TIMEOUT", "120"))


_DOC_MAX_UPLOAD = 25 * 1024 * 1024  # 25 MB raw upload ceiling


_DOC_MAX_CHARS = 400_000  # cap extracted text prepended to a prompt


def _markitdown_bin():
    """Find the converter in the checkout, active virtualenv, or PATH."""
    candidates = [
        os.environ.get("SHAREDAGENTS_MARKITDOWN_BIN"),
        _MARKITDOWN_BIN_DEFAULT,
        str(Path(sys.executable).with_name("markitdown")),
    ]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("markitdown")


def _convert_with_markitdown_library(path: str):
    """Use the installed library when its console-script is unavailable."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        return None, None
    try:
        result = MarkItDown().convert_local(path)
        text = (result.markdown if result else "").strip()
    except Exception as exc:  # noqa: BLE001
        return "", f"conversion failed: {exc}"
    if not text:
        return "", "no text extracted — the file may be empty or a scanned image"
    return text, None


def _extract_upload(path: Path) -> tuple[str, str | None]:
    """Best-effort preview. The upload remains saved on every parser failure."""
    try:
        if path.suffix.lower() in {".txt", ".md", ".markdown"}:
            data = path.read_bytes()
            if b"\x00" in data[:8192]:
                return "", "no text preview available"
            return data.decode("utf-8", "replace").lstrip("\ufeff").strip(), None
        binpath = _markitdown_bin()
        if binpath:
            proc = subprocess.run(
                [binpath, str(path)],
                capture_output=True,
                timeout=_MARKITDOWN_TIMEOUT,
            )
            if proc.returncode != 0:
                return "", "no text preview available"
            text = (proc.stdout or b"").decode("utf-8", "replace").strip()
        else:
            text, error = _convert_with_markitdown_library(str(path))
            if error or text is None:
                return "", "no text preview available"
    except subprocess.TimeoutExpired:
        return "", "text preview timed out"
    except Exception:  # noqa: BLE001 — parsing must never reject a saved file
        return "", "no text preview available"
    return text, None if text else "no text preview available"


@bp.route("/api/attachments", methods=["POST"])
@bp.route("/api/extract", methods=["POST"])
@require_auth
def api_extract_document():
    """Save any multipart ``file``; extraction requires an explicit request.

    ``destination_dir`` selects a directory; omitted means Downloads (or
    SHAREDAGENTS_DOWNLOADS_DIR). Regular users have isolated subfolders. Both the
    original path reaches the agent without loading file contents into its
    prompt. ``extract_text=true`` or the legacy /api/extract endpoint opts in
    to best-effort conversion for clients that explicitly need it.
    """
    f = request.files.get("file")
    if f is None or not (f.filename or "").strip():
        return jsonify({"error": "no file uploaded"}), 400
    data = f.read(_DOC_MAX_UPLOAD + 1)
    if len(data) > _DOC_MAX_UPLOAD:
        return jsonify({"error": "file too large (max 25 MB)"}), 413
    from sharedagents_web.access import viewer_is_admin

    try:
        directory = upload_directory(
            g.user_email,
            admin=viewer_is_admin(),
            destination=request.form.get("destination_dir", "").strip(),
        )
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 403
    except (OSError, ValueError, RuntimeError):
        return jsonify({"error": "invalid destination folder"}), 400
    try:
        path = save_upload(data, f.filename, directory)
    except OSError:
        return jsonify({"error": "could not save upload to destination folder"}), 500

    image_mime = sniff_image_mime(data)
    text, preview_error = "", None
    extract_text = request.form.get(
        "extract_text",
        "true" if request.path == "/api/extract" else "false",
    ).strip().lower() in {"true", "1", "yes"}
    if extract_text and data and not image_mime:
        text, preview_error = _extract_upload(path)
    truncated = len(text) > _DOC_MAX_CHARS
    if truncated:
        text = text[:_DOC_MAX_CHARS].rstrip() + "\n\n…[truncated]"
    return jsonify(
        {
            "name": path.name,
            "path": str(path),
            "size": len(data),
            "mime": image_mime or mimetypes.guess_type(path.name)[0] or "",
            "text": text,
            "chars": len(text),
            "truncated": truncated,
            "image_mime": image_mime,
            "preview_error": preview_error,
        }
    )


@bp.route("/")
@require_auth_page
def index():
    """Serve the chat UI — requires auth so the shell isn't served anonymously.

    The rendered shell carries the persistent active-agent surfaces and the
    selector control; ``/api/sessions`` persists the selected agent.
    """
    # Pass the ACTUAL configured endpoints so the Settings sheet reflects THIS
    # instance (e.g. the dev web on :8200), not a hardcoded prod default.
    return render_template(
        "index.html",
        base=BASE_PATH,
        realtime_url=REALTIME_URL,
        gateway_url=current_gateway(),
        dashboard_url=_server.DASHBOARD_URL,
    )


@bp.route("/api/chat", methods=["POST"])
@require_auth
def api_chat():
    """Chat endpoint with SSE streaming.

    Request body:
    {
        "session_id": "<sid>",
        "message": "<user text>",
        "model": "<model_id or agent name>",
        "agent": "<agent name>"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    session_id = data.get("session_id")
    message = data.get("message", "").strip()
    model = data.get("model", "")
    agent = bare_id(data.get("agent") or DEFAULT_AGENT) or DEFAULT_AGENT

    if not message:
        return jsonify({"error": "Empty message"}), 400

    # Byte-level guard: reject oversized user messages
    if len(message.encode("utf-8")) > SessionStore.MAX_MESSAGE_BYTES:
        return jsonify({"error": f"Message too large (max {SessionStore.MAX_MESSAGE_BYTES // 1024} KB)"}), 413

    # Get or create session (scoped to current user)
    # Only create a new session when NO session_id is supplied.
    # If a session_id is given but not found in the web store, it may be a
    # gateway-native session (started in the TUI, or opened from the agent
    # monitor) the user is now CONTINUING — adopt that same SID so
    # the rest of this turn attaches to and extends that same gateway Session.
    if session_id:
        sdata = store.get(session_id, g.user_email)
        if not sdata:
            from sharedagents_web import access

            if may_adopt_gateway_session(g.user_email):
                bound_owner = store.gateway_binding_owner(session_id)
                if bound_owner is not None and bound_owner != g.user_email:
                    return jsonify({"error": "Session not found"}), 403
                hist = gw_session_history(session_id)
                if hist is None:
                    return jsonify({"error": "Session not found"}), 403
                try:
                    sdata = store.adopt_gateway(
                        g.user_email,
                        session_id,
                        hist,
                        gateway_principal=canonical_gateway_principal(g.user_email),
                    )
                except PermissionError:
                    return jsonify({"error": "Session not found"}), 403
            elif access.session_access(g.user_email, session_id) >= access.CHAT:
                hist = gw_session_history(session_id)
                if hist is None:
                    return jsonify({"error": "Session not found"}), 403
                sdata = {
                    "id": session_id,
                    "agent": bare_id(hist.get("agent") or "") or DEFAULT_AGENT,
                    "model": hist.get("model") or "",
                    "messages": hist.get("messages") or [],
                }
            else:
                return jsonify({"error": "Session not found"}), 403
    else:
        from sharedagents_web.agent_visibility import conversational_agent_or_default

        agent = conversational_agent_or_default(agent, DEFAULT_AGENT)
        sdata = store.create(g.user_email, agent, model)
        session_id = sdata["id"]

    # Ensure agent is set
    if not sdata.get("agent"):
        sdata["agent"] = agent
    if not sdata.get("model"):
        sdata["model"] = model

    # Resolve model id
    model_id = model or f"sharedagents/{agent}"
    model_id = f"sharedagents/{bare_id(model_id)}"

    # Add user message to session
    store.add_message(session_id, "user", message)

    # Bind to a persistent gateway session so chat shares one Session
    # with this web session's slash commands (Option C). When bound, the
    # gateway session OWNS the conversation history — we send only the new
    # user message (no replay → no duplicated transcript). Falls back to
    # the stateless full-history replay if the gateway can't issue a sid.
    had_session = _server._gw_session_log_exists(session_id)
    target_sid = ensure_gw_session(sdata)
    # Send the task's operator title with the turn. The gateway pins it after
    # loading the parent and BEFORE prompting, including the first turn.
    from sharedagents_web.task_board import get_task_board_registry

    task_row = get_task_board_registry().find_by_session(session_id) or {}
    task_title = str(task_row.get("label") or "").strip() if not task_row.get("archived_at") else ""
    if target_sid:
        chat_headers = {"X-Sharedagents-Session-Id": target_sid, "X-Sharedagents-Frontend": "web"}
        # First bind of a web session that already has prior turns: seed the
        # new gateway session with the full transcript so /context, /history,
        # and the model see the conversation so far. After seeding, only the
        # new user message is sent each turn (gateway session owns history).
        if not had_session and len(sdata.get("messages", [])) > 1:
            messages = []
            for msg in sdata.get("messages", []):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})
        else:
            messages = [{"role": "user", "content": message}]
    else:
        messages = []
        for msg in sdata.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        chat_headers = None

    # ── Decouple the upstream turn from the browser connection (F2) ──────
    # The gateway cancels a turn when ITS inbound HTTP request closes. If the
    # SSE generator drove that request directly, a browser navigating away
    # mid-turn would tear the generator down, drop the web→gateway connection,
    # and the gateway would cancel the turn — stranding the transcript with a
    # user message and NO assistant reply (the "navigate away kills the turn"
    # bug). So a BACKGROUND worker owns the upstream request: it consumes the
    # gateway stream to completion and persists the assistant message
    # regardless of the browser, while the client-facing generator merely
    # drains a queue the worker feeds. If the browser leaves, the worker keeps
    # reading the gateway (turn completes + is saved) and just stops feeding
    # the now-dead queue; the saved turn shows up on the next load / via the
    # live session stream.
    q: queue.Queue = queue.Queue(maxsize=2048)
    _SENTINEL = object()

    def emit_delta(ev):
        # Best-effort: if the client buffer is full, DROP this delta rather than
        # stalling the worker. The terminal ``message_end`` carries the FULL
        # content and the client re-renders from it, so a dropped delta
        # self-heals — and the worker never blocks on a slow/absent consumer.
        try:
            q.put_nowait(ev)
        except queue.Full:
            pass

    def emit_terminal(ev):
        # Terminal / error frames MUST land for a connected client (they carry
        # the full content and clear the client's streaming state). If the queue
        # is full it's full of now-stale deltas — message_end re-renders from
        # full content, so they're redundant — so EVICT from the front to make
        # room rather than dropping the terminal frame (codex R4). Non-blocking
        # (only the worker produces, and it's done producing deltas by now), so
        # it can't spin: a present consumer drains it, an absent one just leaves
        # it queued for the generator's deadline to reap.
        for _ in range(q.maxsize + 1):
            try:
                q.put_nowait(ev)
                return
            except queue.Full:
                try:
                    q.get_nowait()  # drop an old, now-superseded delta
                except queue.Empty:
                    pass

    def worker():
        assistant_content = []
        # The gateway echoes the engine's REAL model (claude-opus-4-…,
        # gpt-5.5-codex, composer-2.5) on the terminal chunk for session-bound
        # turns. Capture it so the UI shows the model the agent actually used.
        engine_model = ""
        # Operator-facing label for ``engine_model`` (the models[] alias the
        # gateway resolved from its config; "" when none exists).
        engine_model_alias = ""
        started = time.monotonic()
        resp = None
        watchdog = None
        try:
            resp = gateway_request(
                "POST",
                "/openai/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": messages,
                    "stream": True,
                    **({"task_title": task_title} if task_title else {}),
                },
                headers=chat_headers,
                stream=True,
                # (connect, read) — the READ timeout makes the worker STRICTLY
                # bounded: a wedged upstream that trickles bytes without yielding
                # a newline-delimited SSE line can't sit inside iter_lines()
                # forever (a 300s total timeout wouldn't catch that). 120s
                # between tokens is far beyond any healthy gap, so it won't abort
                # a legitimately slow turn; combined with the per-line wall-clock
                # deadline below, total worker life is finite.
                timeout=(10, 120),
            )

            # HARD wall-clock bound: the per-line deadline check below only runs
            # *between* yielded lines, and the read timeout resets on every byte —
            # so a peer that trickles bytes WITHOUT a newline could keep
            # iter_lines() blocked indefinitely. A watchdog that closes the
            # response at the deadline forces iter_lines() to raise, so the worker
            # (and the gateway turn) are STRICTLY bounded regardless of trickle
            # pattern; the except/finally below then finalize normally.
            if resp is not None:

                def _trip():
                    try:
                        resp.close()
                    except Exception:
                        pass

                watchdog = threading.Timer(_CHAT_WORKER_MAX_SECONDS, _trip)
                watchdog.daemon = True
                watchdog.start()

            if not resp:
                emit_terminal({"type": "error", "error": "Gateway unreachable"})
                emit_terminal({"type": "message_end", "content": "Gateway unreachable"})
                return
            if resp.status_code != 200:
                error_msg = f"Gateway error: {resp.status_code}"
                emit_terminal({"type": "error", "error": error_msg})
                emit_terminal({"type": "message_end", "content": error_msg})
                return

            for line in resp.iter_lines(decode_unicode=True):
                # Absolute worker deadline — a trickling or wedged upstream that
                # never sends [DONE] must not keep this thread (or the gateway
                # turn) alive forever. Stop, persist what we have, finalize.
                if time.monotonic() - started > _CHAT_WORKER_MAX_SECONDS:
                    break
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        # The terminal chunk carries the engine's real model;
                        # ignore the ``sharedagents/<agent>`` id echoed on content
                        # chunks (that's the agent, not the model).
                        cm = str(chunk.get("model") or "").strip()
                        if cm and cm != model_id and not cm.startswith(ID_PREFIXES):
                            engine_model = cm
                        # The gateway resolves the operator-facing alias for
                        # that wire id from the agent config;
                        # empty when the config maps no row — the UI then
                        # falls back to the raw id.
                        if engine_model:
                            engine_model_alias = str(chunk.get("model_alias") or "").strip() or engine_model_alias
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content", "")
                        if text:
                            assistant_content.append(text)
                            # Cap accumulated content to prevent RAM exhaustion.
                            if len("".join(assistant_content).encode("utf-8")) > SessionStore.MAX_STREAM_BYTES:
                                emit_terminal({"type": "error", "error": "Response size limit exceeded"})
                                emit_terminal({"type": "message_end", "content": ""})
                                return
                            emit_delta({"type": "delta", "content": text})
                    except json.JSONDecodeError:
                        pass

            # Finalize — PERSIST FIRST (independent of the browser), then emit,
            # so the assistant reply is saved even if the client already left.
            full_content = "".join(assistant_content)
            if full_content:
                store.add_message(session_id, "assistant", full_content)

            # Persist + surface the engine's real model in a DEDICATED field
            # (kept separate from the picker's ``model`` selection). AUTHORITATIVE
            # per-turn sync: the gateway's terminal frame always reflects the
            # session's current ``engine_model`` — a concrete id, or "" when the
            # engine cleared it — so mirror BOTH (incl. the clear). Only on the
            # NORMAL completion path; error / early returns leave it untouched.
            end_event = {"type": "message_end", "content": full_content}
            if sdata.get("engine_model", "") != engine_model:
                try:
                    # Re-read inside the write txn. A get()+_save of the
                    # request-start snapshot would clobber add_message
                    # from this worker or a sibling gunicorn worker.
                    store.update_fields(
                        session_id,
                        engine_model=engine_model,
                        engine_model_alias=engine_model_alias,
                    )
                except Exception:
                    pass
            end_event["model"] = engine_model
            end_event["model_alias"] = engine_model_alias
            emit_terminal(end_event)

        except Exception:
            emit_terminal({"type": "error", "error": "An unexpected error occurred"})
            emit_terminal({"type": "message_end", "content": ""})
        finally:
            if watchdog is not None:
                watchdog.cancel()  # turn finished within the deadline — disarm
            # Always signal end-of-stream to the generator — evict a stale delta
            # if needed so the sentinel can't be dropped on a full queue (it must
            # follow the message_end we just inserted). The generator's absolute
            # deadline is a backstop, so it can never spin even if no consumer.
            emit_terminal(_SENTINEL)

    threading.Thread(target=worker, name="sharedagents-web-chat", daemon=True).start()

    def generate():
        """SSE generator — relays the worker's queue to the client. Independent
        of the upstream turn, so the browser can disconnect without cancelling
        it. Bounded by an absolute deadline so it can NEVER spin forever even if
        a terminal frame / sentinel is somehow lost."""
        yield f"data: {json.dumps({'type': 'session_update', 'session_id': session_id})}\n\n"
        yield f"data: {json.dumps({'type': 'message_start', 'session_id': session_id, 'role': 'assistant'})}\n\n"
        started = time.monotonic()
        while True:
            if time.monotonic() - started > _CHAT_STREAM_MAX_SECONDS:
                break
            try:
                ev = q.get(timeout=15)
            except queue.Empty:
                # Heartbeat comment (ignored by the client parser) — also lets a
                # dead client socket surface as a write error so we promptly stop
                # holding this request thread when the browser has gone.
                yield ": ping\n\n"
                continue
            if ev is _SENTINEL:
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@bp.route("/api/health")
@require_auth
def api_health():
    """Health check — requires auth to prevent leaking GATEWAY_URL."""
    resp = gateway_request("GET", "/")
    healthy = resp is not None and resp.status_code == 200
    return jsonify({"healthy": healthy, "gateway": current_gateway()})
