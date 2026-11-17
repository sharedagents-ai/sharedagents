"""Send the files a turn recorded to a Telegram chat.

Agents deliver a file by linking it in their answer; the runtime records it as
an ``artifact_reference`` (authorized roots, sha256, durable copy) and the
gateway serves the bytes at ``GET …/sessions/{sid}/artifacts/{sha256}``. This
module fetches each recorded file and sends it — PNG/JPEG/WebP as a photo when
it fits Telegram's photo limit, everything else (PDF, Markdown, a GIF, a
spreadsheet…) as a document. A file that is gone, too large, or refused by
Telegram gets a one-line notice instead of silently disappearing.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from typing import Any

import httpx

from sharedagents_core.attachments import INLINE_IMAGE_MIMES, sniff_image_mime

# Bot API upload limits: sendPhoto 10 MB; sendDocument 50 MB on the cloud API.
# A self-hosted ``--local`` server accepts 2000 MB, but the file is held in the
# bot's memory while it uploads, so cap it where the inbound document path does.
PHOTO_MAX_BYTES = 10 * 1024 * 1024
DOCUMENT_MAX_BYTES = 50 * 1024 * 1024
LOCAL_DOCUMENT_MAX_BYTES = 200 * 1024 * 1024
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_FETCH_TIMEOUT = httpx.Timeout(60.0)


class _Unavailable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _display_name(event: dict, digest: str) -> str:
    base = os.path.basename(str(event.get("display_name") or "").replace("\\", "/"))
    base = "".join(ch for ch in base if ch.isprintable()).strip()[:200]
    return base if base and base not in {".", ".."} else f"attachment-{digest[:8]}"


async def _fetch(
    http: httpx.AsyncClient,
    url: str,
    max_bytes: int,
) -> tuple[bytes, str]:
    async with http.stream("GET", url) as response:
        if response.status_code == 404:
            raise _Unavailable("it is no longer available")
        if response.status_code >= 400:
            raise _Unavailable(f"the gateway answered HTTP {response.status_code}")
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > max_bytes:
                raise _Unavailable("too large")
            body.extend(chunk)
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return bytes(body), content_type


async def deliver_artifacts(
    api: Any,
    *,
    base: str,
    headers: dict[str, str],
    chat_id: int,
    thread_id: int | None,
    sid: str,
    artifacts: Iterable[dict],
    log: logging.Logger | None = None,
) -> int:
    """Send each recorded file of one turn to ``chat_id``; returns how many went out."""
    log = log or logging.getLogger("sharedagents.telegram")
    doc_cap = LOCAL_DOCUMENT_MAX_BYTES if getattr(api, "local_mode", False) else DOCUMENT_MAX_BYTES
    sent = 0
    seen: set[str] = set()

    async def notice(text: str) -> None:
        try:
            await api.send_message(chat_id, text, message_thread_id=thread_id)
        except Exception:  # noqa: BLE001 — a notice must never break the turn
            log.warning("attachment notice failed chat=%s", chat_id)

    async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT, headers=headers) as http:
        for event in artifacts:
            digest = str(event.get("sha256") or "").lower()
            if not _DIGEST_RE.match(digest) or digest in seen:
                continue
            seen.add(digest)
            name = _display_name(event, digest)
            too_large = (
                f"📎 {name} is too large to send on Telegram "
                f"(limit {doc_cap // (1024 * 1024)} MB) — open it in the sharedagents web UI."
            )
            try:
                declared = int(event.get("size_bytes") or 0)
            except (TypeError, ValueError):
                declared = 0
            if declared > doc_cap:
                await notice(too_large)
                continue
            url = f"{base.rstrip('/')}/control/v1/orchestrator/sessions/{sid}/artifacts/{digest}"
            try:
                data, content_type = await _fetch(http, url, doc_cap)
            except _Unavailable as exc:
                await notice(too_large if exc.reason == "too large" else f"📎 Couldn't attach {name} — {exc.reason}.")
                continue
            except httpx.HTTPError as exc:
                log.warning("artifact fetch failed chat=%s file=%s: %s", chat_id, name, exc)
                await notice(f"📎 Couldn't attach {name} — the gateway is unreachable.")
                continue
            image_mime = sniff_image_mime(data)
            message_id = None
            if image_mime in INLINE_IMAGE_MIMES and len(data) <= PHOTO_MAX_BYTES:
                message_id = await api.send_photo(
                    chat_id,
                    data,
                    filename=name,
                    message_thread_id=thread_id,
                )
            if message_id is None:
                mime = (
                    str(event.get("media_type") or "").strip()
                    or image_mime
                    or content_type
                    or "application/octet-stream"
                )
                message_id = await api.send_document(
                    chat_id,
                    data,
                    filename=name,
                    mime=mime,
                    message_thread_id=thread_id,
                )
            if message_id is None:
                await notice(f"📎 Telegram refused {name}.")
                continue
            sent += 1
            log.info("TX-FILE chat=%s file=%s bytes=%d", chat_id, name, len(data))
    return sent
