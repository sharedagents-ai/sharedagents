"""sharedagents_web.routes.artifacts — Session artifact store routes (agent-emitted image/PDF media).

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

import hashlib
import os

from flask import Blueprint, g, jsonify, request, send_from_directory

from sharedagents_web.principal import may_adopt_gateway_session
from sharedagents_web.server import (
    _ARTIFACT_MIME,
    ARTIFACTS_DIR,
    _gw_session_log_exists,
    public_web_url,
    require_auth,
    store,
)

bp = Blueprint("artifacts", __name__)


_ARTIFACTS_ROOT = ARTIFACTS_DIR.resolve()


def _artifact_session_dir(sid: str):
    """Resolved absolute artifacts dir for one session hash, or None if ``sid``
    is not a single well-formed component that stays under ARTIFACTS_DIR.

    ``sid`` must be one path segment (no separator, ``.``/``..``, or NUL). We
    then ``resolve()`` and require the result to sit under the resolved
    ARTIFACTS_DIR — so a crafted sid AND a symlinked ``<ARTIFACTS_DIR>/<sid>``
    pointing outside the root both fail closed (the session dir itself, not
    just <sid>/<name>, is proven in-jail).
    """
    if not sid or sid in (".", "..") or "/" in sid or "\\" in sid or "\x00" in sid:
        return None
    base = (_ARTIFACTS_ROOT / sid).resolve()
    if base != _ARTIFACTS_ROOT and not str(base).startswith(str(_ARTIFACTS_ROOT) + os.sep):
        return None
    return base


def _safe_artifact_path(sid: str, name: str):
    """Resolve <ARTIFACTS_DIR>/<sid>/<name> and prove it stays inside the
    session's own artifacts dir. Returns the resolved Path or None if the
    request escapes the jail, targets a disallowed extension, or is malformed.

    Defense in depth: (1) the extension whitelist rejects anything not in
    _ARTIFACT_MIME, (2) resolve() collapses ``..``/symlinks and we require the
    result to sit under the session dir, so ``/files/<sid>/../../etc/passwd``
    and absolute-path smuggling both fail closed.
    """
    if not sid or not name:
        return None
    ext = os.path.splitext(name)[1].lower()
    if ext not in _ARTIFACT_MIME:
        return None
    base = _artifact_session_dir(sid)
    if base is None:
        return None
    try:
        target = (base / name).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    # Containment: target must be base itself's child (string-prefix on the
    # resolved paths, with a separator to avoid ``/a/bc`` matching ``/a/b``).
    if target != base and not str(target).startswith(str(base) + os.sep):
        return None
    return target


def _safe_response_media_type(target) -> tuple[str, bool]:
    """Return a signature-checked type and whether inline rendering is safe."""
    declared = _ARTIFACT_MIME.get(
        os.path.splitext(target.name)[1].lower(),
        "application/octet-stream",
    )
    try:
        with target.open("rb") as handle:
            head = handle.read(16)
    except OSError:
        return "application/octet-stream", False
    valid_inline = (
        (declared == "image/png" and head.startswith(b"\x89PNG\r\n\x1a\n"))
        or (declared == "image/jpeg" and head.startswith(b"\xff\xd8\xff"))
        or (declared == "image/gif" and head[:6] in {b"GIF87a", b"GIF89a"})
        or (declared == "image/webp" and head.startswith(b"RIFF") and head[8:12] == b"WEBP")
        or (declared == "application/pdf" and head.startswith(b"%PDF-"))
    )
    return (
        (declared, True)
        if valid_inline
        else (
            "application/octet-stream",
            False,
        )
    )


def _content_address_matches(target, name: str) -> bool:
    """Verify digest-addressed assets; legacy flat asset names remain valid."""
    first = str(name or "").split("/", 1)[0].lower()
    if not (len(first) == 64 and all(char in "0123456789abcdef" for char in first)):
        return True
    try:
        hasher = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest() == first
    except OSError:
        return False


@bp.route("/files/<sid>/<path:name>")
@require_auth
def serve_artifact(sid, name):
    """Serve an agent-emitted artifact (image/PDF) for a session, same-origin.

    Access control mirrors every other session route: the caller must own the
    session (``store.get(sid, g.user_email)``), so one user can never read
    another's artifacts. Served inline (Content-Disposition: inline) so images
    render in <img> and PDFs open in the browser's native viewer. The strict
    extension whitelist + path-traversal jail live in _safe_artifact_path.
    """
    # Ownership check — 404 (not 403) so we don't leak which sids exist.
    storage_sid = _artifact_gateway_session(sid)
    if not storage_sid:
        return jsonify({"error": "not found"}), 404
    target = _safe_artifact_path(storage_sid, name)
    # Backward compatibility for files written before assets were keyed by the
    # logical gateway conversation rather than the Web UUID.
    if (target is None or not target.is_file()) and storage_sid != sid:
        target = _safe_artifact_path(sid, name)
    if target is None or not target.is_file() or not _content_address_matches(target, name):
        return jsonify({"error": "not found"}), 404
    media_type, inline_safe = _safe_response_media_type(target)
    force_download = request.args.get("download") in {"1", "true", "yes"}
    resp = send_from_directory(
        str(target.parent),
        target.name,
        mimetype=media_type,
        as_attachment=(force_download or not inline_safe),
    )
    # Inline so the browser renders it in-page rather than downloading. Encode
    # the filename per RFC 5987 so names with quotes/control chars/`%` can't
    # break out of the header.
    from urllib.parse import quote as _urlquote

    disposition = "attachment" if force_download or not inline_safe else "inline"
    resp.headers["Content-Disposition"] = f"{disposition}; filename*=UTF-8''" + _urlquote(target.name, safe="")
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


@bp.route("/api/sessions/<sid>/artifacts", methods=["POST"])
@require_auth
def api_session_artifact(sid):
    """Producer channel: attach an image/PDF to a session's artifact store.

    Accepts EITHER a multipart ``file`` upload OR a JSON body
    ``{"name": "chart.png", "data_base64": "..."}``. Writes it under the
    session's artifacts dir and returns ``{"url": "...", "name": ...}`` under
    the configured Web base — the exact same-origin URL an agent then drops
    into its markdown so it renders inline.

    When ``content_addressed`` is true (JSON body or form field), bytes are
    stored under ``<gateway_key>/<sha256>/<safe_name>`` — the same layout
    agent outputs use — so GET ``/files/<web-sid>/<sha256>/<name>`` verifies
    the digest. User-attached WebUI images use this path.

    Used by the agent/orchestrator to surface a turn's output image or PDF in
    the web UI directly, replacing the ``sendPhoto``-to-Telegram detour.
    """
    storage_sid = _artifact_gateway_session(sid)
    if not storage_sid:
        return jsonify({"error": "not found"}), 404

    raw = None
    name = None
    content_addressed = False
    upload = request.files.get("file") if request.files else None
    if upload is not None:
        name = upload.filename or ""
        raw = upload.read()
        content_addressed = str(request.form.get("content_addressed") or "").lower() in {"1", "true", "yes"}
    else:
        body = request.get_json(silent=True) or {}
        name = body.get("name") or ""
        b64 = body.get("data_base64") or ""
        content_addressed = bool(body.get("content_addressed"))
        if b64:
            import base64 as _b64

            # Tolerate a data-URI prefix (data:image/png;base64,....).
            if "," in b64 and b64.lstrip().startswith("data:"):
                b64 = b64.split(",", 1)[1]
            try:
                raw = _b64.b64decode(b64, validate=False)
            except (ValueError, TypeError):
                return jsonify({"error": "bad base64"}), 400

    if not name or raw is None:
        return jsonify({"error": "missing file or name"}), 400

    # Sanitize the name to a single safe component and enforce the whitelist.
    name = os.path.basename(name).replace("\x00", "")
    ext = os.path.splitext(name)[1].lower()
    if ext not in _ARTIFACT_MIME:
        return jsonify({"error": f"unsupported type: {ext or '(none)'}"}), 400

    sess_dir = _artifact_session_dir(storage_sid)
    if sess_dir is None:
        return jsonify({"error": "invalid name"}), 400

    digest = hashlib.sha256(raw).hexdigest()
    relative_name = f"{digest}/{name}" if content_addressed else name
    if _safe_artifact_path(storage_sid, relative_name) is None:
        return jsonify({"error": "invalid name"}), 400

    try:
        if content_addressed:
            target_dir = sess_dir / digest
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / name
            # Atomic replace so a concurrent reader never sees a partial file.
            tmp = target_dir / f".staging-{os.getpid()}-{name}"
            try:
                tmp.write_bytes(raw)
                os.replace(tmp, target)
            except Exception:
                try:
                    if tmp.is_file():
                        tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        else:
            sess_dir.mkdir(parents=True, exist_ok=True)
            (sess_dir / name).write_bytes(raw)
    except OSError as e:
        return jsonify({"error": f"write failed: {e}"}), 500

    # URL-quote the name so a filename with `#`, `?`, `%`, or spaces yields a
    # valid URL that round-trips back to the same file via the /files route.
    from urllib.parse import quote as _urlquote

    if content_addressed:
        url = public_web_url(f"/files/{sid}/{digest}/{_urlquote(name, safe='')}")
    else:
        url = public_web_url(f"/files/{sid}/{_urlquote(name, safe='')}")
    payload = {"url": url, "name": name, "bytes": len(raw)}
    if content_addressed:
        payload["sha256"] = digest
    return jsonify(payload), 201


def _artifact_gateway_session(sid: str) -> str:
    """Resolve an authorized Web SID."""
    sdata = store.get(sid, g.user_email)
    if sdata is not None:
        return str(sid or "")

    # Monitor-only delegated children use the same read authorization as the
    # history/SSE routes. Import lazily to avoid a blueprint import cycle.
    from sharedagents_web.routes.sessions import (
        _configured_native_view_allowed,
        _owned_gateway_ancestor,
    )

    authorized = (
        may_adopt_gateway_session(g.user_email)
        or _configured_native_view_allowed(sid, g.user_email)
        or bool(_owned_gateway_ancestor(sid, g.user_email))
    )
    if not authorized:
        from sharedagents_web import access

        authorized = access.session_access(g.user_email, sid) >= access.VIEW
    return sid if authorized and _gw_session_log_exists(sid) else ""
