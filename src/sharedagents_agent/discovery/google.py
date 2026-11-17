"""Provider-specific model discovery implementation."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

from ._common import (
    _LOG,
    GEMINI_MODELS_URL,
    _cache_get,
    _cache_put,
    _http_get_json,
    _row_with_effort,
)


def _gemini_api_key() -> str:
    """Resolve the Gemini key from the env. The Google AI Studio SDKs accept
    either ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``; honour both."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        key = (os.environ.get(var) or "").strip()
        if key:
            return key
    return ""


def _gemini_alias(model_id: str) -> str:
    """``gemini-3.5-flash`` -> ``3.5-flash`` (short, readable). Falls back to
    the full id."""
    s = model_id
    if s.startswith("gemini-"):
        s = s[len("gemini-") :]
    return s or model_id


def _is_gemini_chat_model_id(model_id: str) -> bool:
    """Keep chat-capable Gemini ids; drop embeddings, tts, image/imagen,
    live/realtime, aqa, etc. that ``antigravity --model`` can't drive as a
    conversational engine."""
    mid = model_id.lower()
    if not mid.startswith("gemini"):
        return False
    bad = ("embedding", "embed", "aqa", "tts", "imagen", "image", "vision", "live")
    return not any(b in mid for b in bad)


def _parse_gemini_models(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse a Gemini ``GET /v1beta/models`` page into picker rows.

    The catalogue array lives under ``models`` (real API) — accept ``data``
    too for symmetry with the other providers. Each entry's ``name`` carries a
    ``models/`` prefix that we strip to get the wire id.
    """
    rows: list[dict[str, Any]] = []
    for m in data.get("models") or data.get("data") or []:
        name = str(m.get("name") or "").strip()
        if name.startswith("models/"):
            name = name[len("models/") :]
        if not name or not _is_gemini_chat_model_id(name):
            continue
        limit = int(m.get("inputTokenLimit") or 1_000_000)
        rows.append(
            _row_with_effort(
                {
                    "alias": _gemini_alias(name),
                    "id": name,
                    "display": str(m.get("displayName") or name),
                    "contextWindow": limit,
                    "provider": "antigravity",
                },
                m,
            )
        )
    return rows


def _discover_antigravity_cli_models() -> list[dict[str, Any]]:
    """Discover available models directly from the local antigravity CLI."""
    exe = shutil.which("antigravity")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [exe, "models"],
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        if proc.returncode != 0:
            return []
        rows: list[dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            clean = re.sub(
                r"^[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏\s\.]*Fetching available models\.\.\.",
                "",
                line,
            ).strip()
            if not clean:
                continue
            parts = clean.split(None, 1)
            if not parts:
                continue
            mid = parts[0]
            display = parts[1] if len(parts) > 1 else mid
            alias = _gemini_alias(mid)
            ctx = 1_000_000 if "gemini" in mid else 200_000
            row = {
                "alias": alias,
                "id": mid,
                "display": display,
                "contextWindow": ctx,
                "provider": "antigravity",
            }
            if "-high" in mid or "(High)" in display:
                row["effort"] = "high"
            elif "-medium" in mid or "(Medium)" in display:
                row["effort"] = "medium"
            elif "-low" in mid or "(Low)" in display:
                row["effort"] = "low"
            rows.append(_row_with_effort(row, {}))
        return rows
    except Exception as exc:  # noqa: BLE001
        _LOG.info("antigravity models CLI discovery failed: %s", exc)
        return []


def discover_gemini_models(
    *,
    force: bool = False,
    fetch: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
    api_key: str | None = ...,  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Return the Gemini (antigravity) model catalogue as picker rows.

    Live ``GET /v1beta/models?key=…`` (filtered to chat-capable ids) when a
    ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` resolves, else ``antigravity models``
    CLI discovery, degrading to the config fallback (``external_models.antigravity``).
    Always non-empty when the config section is present. Memoised for :data:`_TTL_S`.

    ``api_key`` defaults to the auto-resolved env key; pass ``""`` to force the
    no-auth path in tests, or a string to inject one.
    ``fetch`` overrides the HTTP getter (tests).
    """
    try:
        key_cache = "gemini"
        if not force:
            cached = _cache_get(key_cache)
            if cached is not None:
                return cached

        api = _gemini_api_key() if api_key is ... else (api_key or "")
        rows: list[dict[str, Any]] = []
        if api:
            try:
                data = (fetch or _http_get_json)(
                    f"{GEMINI_MODELS_URL}?key={api}",
                    {},
                )
                rows = _parse_gemini_models(data)
            except Exception as exc:  # noqa: BLE001
                _LOG.info("gemini model discovery failed: %s", exc)
                rows = []
        if not rows and fetch is None and api_key is ...:
            try:
                rows = _discover_antigravity_cli_models()
            except Exception as exc:  # noqa: BLE001
                _LOG.info("antigravity cli discovery failed: %s", exc)
                rows = []

        _cache_put(key_cache, rows)
        return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("discover_gemini_models failed: %s", exc)
        return []


__all__ = [
    "_gemini_api_key",
    "_gemini_alias",
    "_is_gemini_chat_model_id",
    "_parse_gemini_models",
    "discover_gemini_models",
]
