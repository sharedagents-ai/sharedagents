"""Channels-tab WhatsApp pairing QR (JSON + data-URL for the Web UI)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from sharedagents_whatsapp.qr import QrShow, peek_qr, show_qr

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def status() -> dict[str, Any]:
    """Current link state plus a QR image only when one is already on disk."""
    return payload(peek_qr())


def generate(phone: str | None = None) -> dict[str, Any]:
    """Restart the linker and return a fresh QR (or linked status)."""
    return payload(show_qr(phone=phone or None, force=True))


def payload(result: QrShow) -> dict[str, Any]:
    detail = result.detail or ""
    if result.image_path is not None:
        detail = "Scan now in WhatsApp → Linked devices. Click Refresh QR if it expires (about 20 seconds)."
    return {
        "linked": bool(result.linked),
        "account": result.account or "",
        "detail": detail,
        "started_linker": bool(result.started_linker),
        "image": _data_url(result.image_path),
    }


def _data_url(path: Path | None) -> str | None:
    if path is None:
        return None
    raw = Path(path).read_bytes()
    if not raw.startswith(_PNG_MAGIC):
        return None
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
