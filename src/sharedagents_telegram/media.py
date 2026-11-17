"""Telegram media download + document conversion (extracted from ``bot.py``).

Self-contained media helpers used by :class:`~sharedagents_telegram.bot.AcpTelegramBot`:

* :func:`_make_default_audio_downloader` — pulls a file off the Telegram
  bot-file CDN (getFile → download), returning ``(bytes, filename)``.
* :func:`_convert_office_bytes_to_markdown` — best-effort office-doc → markdown
  via ``markitdown`` (``_OFFICE_EXTENSIONS`` gate).

Depends on NOTHING in ``bot.py`` (one-way import), so ``bot.py`` re-exports these
names for callers/tests importing them from ``sharedagents_telegram.bot``. The
optional ``markitdown`` dependency stays LAZY-imported inside the converter.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:  # annotations only (PEP 563 strings at runtime)
    from sharedagents_telegram.api import _TelegramAPI


def _make_default_audio_downloader(api: _TelegramAPI):
    """Return an async ``download_audio(file_id) -> (bytes, filename)`` that
    pulls the file off the Telegram bot-file CDN.

    Used for both voice input (run through STT) and photos (forwarded
    as ACP ImageContent). Telegram's two-step flow: ``getFile`` returns
    a ``file_path``; the file itself lives at
    ``<api_base>/file/bot<token>/<file_path>``.
    """

    async def _dl(file_id: str):
        meta = await api.call("getFile", file_id=file_id)
        if not isinstance(meta, dict) or not meta.get("ok"):
            # Surface the reason instead of failing silently — the most
            # common one is "file is too big" (cloud Bot API's 20 MB
            # getFile cap), which the caller's size-guard normally
            # catches first, but log it here for anything that slips
            # through (e.g. file_size absent from the message).
            api._log.warning(
                "getFile failed: %s",
                meta.get("description") if isinstance(meta, dict) else meta,
            )
            return None
        result = meta.get("result")
        if not isinstance(result, dict):
            return None
        file_path = result.get("file_path")
        if not file_path:
            return None
        if getattr(api, "local_mode", False):
            # Self-hosted server. In --local mode getFile returns an ABSOLUTE
            # on-disk path; but a misconfigured container (e.g. the aiogram
            # image where `command: ["--local"]` is a silent no-op → cloud
            # mode) returns a RELATIVE path like 'voice/file_123.oga'. Resolve
            # a relative path under the bot-api data root as
            # ``<root>/<token>/<file_path>`` — the volume is bind-mounted at the
            # same path and the file is readable by the bot user (granted via
            # ACL). Read it directly (off the event loop — files can be large),
            # confined to the data root (``resolve()`` defeats symlink / ".."
            # traversal). If the disk read isn't possible, FALL THROUGH to the
            # HTTP download below instead of failing outright — that path works
            # for any server that serves ``/file/`` (so voice survives whether
            # the server returns absolute paths, relative paths, or runs in
            # cloud mode). The earlier code rejected every relative path with
            # "local getFile path not readable" → "couldn't download voice".
            from pathlib import Path as _P

            _root = _P(os.environ.get("SHAREDAGENTS_TELEGRAM_FILE_ROOT", "/var/lib/telegram-bot-api"))
            p = _P(file_path)
            if not p.is_absolute():
                p = _root / api._token / file_path
            if p.is_file():
                try:
                    p.resolve().relative_to(_root.resolve())
                except ValueError:
                    api._log.warning("local getFile path outside data root: %r", file_path)
                    return None
                try:
                    data = await asyncio.to_thread(p.read_bytes)
                    return data, p.name
                except OSError as exc:
                    api._log.warning("local file read failed (%s) — trying HTTP", exc)
            else:
                api._log.info("local getFile %r not a readable on-disk file (%s) — trying HTTP download", file_path, p)
            # fall through to the HTTP download (serves cloud-mode / relative)
        url = f"{api._api_base}/file/bot{api._token}/{file_path}"
        try:
            r = await api._http.get(url)
        except httpx.HTTPError as exc:
            api._log.warning("file download transport error: %s", exc)
            return None
        if r.status_code != 200 or not r.content:
            api._log.warning(
                "file download HTTP %s len=%d",
                r.status_code,
                len(r.content or b""),
            )
            return None
        # Path's basename gives a stable filename (e.g. voice/file_123.oga).
        from pathlib import Path as _P

        return r.content, _P(file_path).name

    return _dl


_OFFICE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".docx",
        ".doc",  # Word
        ".pptx",
        ".ppt",  # PowerPoint
        ".xlsx",
        ".xls",  # Excel
        ".pdf",  # PDF
        ".md",  # Markdown
    }
)


def _convert_office_bytes_to_markdown(
    data: bytes,
    filename: str,
) -> str | None:
    """Convert an office document's bytes to markdown via markitdown.

    Returns the markdown string, or ``None`` on any failure (markitdown
    not installed, conversion error, empty result).
    """
    try:
        from markitdown import MarkItDown as _MarkItDown
    except ImportError:
        return None
    ext = Path(filename).suffix.lower()
    if ext not in _OFFICE_EXTENSIONS:
        return None
    import io as _io

    try:
        result = _MarkItDown().convert_stream(
            _io.BytesIO(data),
            file_extension=ext,
        )
        md = result.markdown if result else ""
    except Exception:  # noqa: BLE001
        return None
    if not md:
        return None
    # Strip to a reasonable size — Telegram messages cap at 4096 chars
    # per message, and the ACP streaming path handles the rest, but the
    # converted content itself should stay manageable.
    if len(md) > 10 * 1024 * 1024:  # 10 MB hard cap
        md = md[:10_000_000] + "\n\n…[truncated at 10 MB]"
    return md
