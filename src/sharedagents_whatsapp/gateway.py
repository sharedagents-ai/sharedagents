"""Loopback client for the sharedagents gateway control API.

The WhatsApp channel drives agent sessions exactly like web/TUI/Telegram do:
mint a session (``POST /control/v1/{agent}/sessions``), submit prompts to the
canonical fire-and-forget route (``POST …/sessions/{sid}/prompt``), answer
parked questions/permissions, run server-side slash commands, tail the
session's canonical event log (``GET …/sessions/{sid}/stream?mode=events``),
and fetch the files a turn recorded (``GET …/sessions/{sid}/artifacts/{sha256}``).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import httpx

WORKSPACE_ORIGIN = "whatsapp"
FRONTEND = "whatsapp"
# The events stream defaults every visibility pref to "stream" when omitted,
# which would push reasoning, tool bodies and sub-agent internals into WhatsApp.
# ``reasoning=off&tools=on`` keeps text_delta, tool_call (ask detection),
# tool_result_meta, permission_* and turn_* only.
STREAM_QUERY = "mode=events&reasoning=off&tools=on"

_POST_TIMEOUT = httpx.Timeout(30.0)
# The gateway emits an ``hb`` frame every 15 s while a session is idle, so a
# 60 s read timeout detects a stalled stream without cutting a quiet one.
_STREAM_TIMEOUT = httpx.Timeout(30.0, read=60.0)
_FETCH_TIMEOUT = httpx.Timeout(60.0)
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_DISPOSITION_NAME_RE = re.compile(r"filename\*=UTF-8''([^;]+)", re.I)


class GatewayError(RuntimeError):
    """A gateway control call failed; ``code`` is the gateway's error code."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"{code or status}: {message}")
        self.status = status
        self.code = code
        self.message = message


class SessionNotFound(GatewayError):
    """The session id is unknown to the gateway (deleted or never existed)."""


class ArtifactUnavailable(GatewayError):
    """A recorded file cannot be delivered: gone (404) or over the cap (413).

    Permanent for that file, so callers report it once instead of retrying.
    """


@dataclass(frozen=True)
class Artifact:
    """A turn's recorded file. ``content_type`` is the gateway's verified type:
    a signature-checked PNG/JPEG/GIF/WebP or PDF, else ``application/octet-stream``."""

    data: bytes
    content_type: str
    filename: str


@dataclass(frozen=True)
class SubmitResult:
    """``queued`` = folded into the session's follow-up queue (no offset, a ``queue_id``)."""

    queued: bool
    start_offset: int | None
    queue_id: str = ""


@dataclass(frozen=True)
class StreamItem:
    """Canonical ``event``, idle ``heartbeat``, or a ``reconnect`` boundary."""

    kind: str
    offset: int
    event: dict


def _raise_for(response: httpx.Response, *, session_scoped: bool) -> dict:
    try:
        payload: Any = response.json()
    except ValueError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if response.status_code < 400:
        return payload
    raw = payload.get("result")
    result: dict = raw if isinstance(raw, dict) else {}
    code = str(payload.get("error_code") or payload.get("code") or "")
    message = str(
        payload.get("error")
        or payload.get("message")
        or result.get("text")
        or response.text[:200]
        or f"HTTP {response.status_code}"
    )
    if response.status_code == 404 and session_scoped and code != "unknown_agent":
        raise SessionNotFound(404, code or "not_found", message)
    raise GatewayError(response.status_code, code, message)


class GatewayClient:
    """Async client for the gateway's ``/control/v1`` routes."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.AsyncClient(
            timeout=_POST_TIMEOUT,
            headers=headers,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, path: str, body: dict, *, session_scoped: bool = True) -> dict:
        response = await self._http.post(f"{self._base}{path}", json=body)
        return _raise_for(response, session_scoped=session_scoped)

    async def create_session(self, agent: str) -> str:
        """Mint a gateway session for ``agent``, tagged ``workspace_origin=whatsapp``."""
        data = await self._post(
            f"/control/v1/{agent}/sessions",
            {"workspace_origin": WORKSPACE_ORIGIN},
            session_scoped=False,
        )
        sid = data.get("sessionId") or data.get("session_id")
        if not sid:
            raise GatewayError(500, "agent_error", f"no sessionId for agent {agent!r}")
        return str(sid)

    async def session_exists(self, sid: str) -> bool:
        """True/False on a definite answer; raises when the gateway can't say.

        An unreachable gateway is not a missing session — the caller must keep
        the binding and retry rather than start a new conversation.
        """
        response = await self._http.get(
            f"{self._base}/control/v1/orchestrator/sessions/{sid}/state",
        )
        if response.status_code == 404:
            return False
        _raise_for(response, session_scoped=False)
        return True

    async def submit(
        self,
        sid: str,
        blocks: list[dict],
        *,
        actor: dict[str, str] | None = None,
    ) -> SubmitResult:
        """Fire-and-forget prompt submit; the observer delivers the answer.

        ``actor`` is presentation provenance for the session transcript, never
        an authorization claim.  The gateway validates it before carrying it
        in the model-invisible prompt metadata.
        """
        body: dict[str, Any] = {"prompt": blocks}
        if actor:
            body["actor"] = actor
        data = await self._post(
            f"/control/v1/orchestrator/sessions/{sid}/prompt",
            body,
        )
        offset = data.get("start_offset")
        return SubmitResult(
            queued=bool(data.get("queued")),
            start_offset=offset if isinstance(offset, int) and not isinstance(offset, bool) else None,
            queue_id=str(data.get("queue_id") or ""),
        )

    async def remove_queued(self, sid: str, queue_id: str) -> bool:
        """Withdraw a still-pending follow-up; False when it was already consumed."""
        response = await self._http.delete(
            f"{self._base}/control/v1/orchestrator/sessions/{sid}/queue/{queue_id}",
        )
        return bool(_raise_for(response, session_scoped=True).get("ok"))

    async def cancel(self, sid: str, *, force: bool = False) -> None:
        await self._post(
            f"/control/v1/orchestrator/sessions/{sid}/cancel",
            {"force": bool(force)},
        )

    async def answer(self, sid: str, call_id: str, answers: list[dict]) -> None:
        await self._post(
            f"/control/v1/orchestrator/sessions/{sid}/answer",
            {"callId": call_id, "answers": answers},
        )

    async def permission(self, sid: str, request_id: str, option_id: str | None) -> None:
        """Resolve a parked permission; ``option_id=None`` cancels (denies) it."""
        outcome = {"outcome": "selected", "optionId": option_id} if option_id else {"outcome": "cancelled"}
        await self._post(
            f"/control/v1/orchestrator/sessions/{sid}/permission",
            {"requestId": request_id, "outcome": outcome},
        )

    async def fetch_artifact(self, sid: str, digest: str, *, max_bytes: int) -> Artifact:
        """Download one file the session recorded, reading at most ``max_bytes``."""
        if not _DIGEST_RE.match(digest or ""):
            raise ArtifactUnavailable(404, "not_found", "malformed artifact digest")
        url = f"{self._base}/control/v1/orchestrator/sessions/{sid}/artifacts/{digest}"
        async with self._http.stream("GET", url, timeout=_FETCH_TIMEOUT) as response:
            if response.status_code >= 400:
                await response.aread()
                if response.status_code == 404:
                    raise ArtifactUnavailable(404, "not_found", "artifact not found")
                _raise_for(response, session_scoped=False)
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise ArtifactUnavailable(
                        413,
                        "too_large",
                        f"artifact is larger than {max_bytes} bytes",
                    )
                chunks.append(chunk)
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            named = _DISPOSITION_NAME_RE.search(response.headers.get("content-disposition", ""))
        filename = unquote(named.group(1).strip()) if named else ""
        return Artifact(b"".join(chunks), content_type, filename)

    async def command(self, agent: str, text: str, sid: str | None) -> dict:
        """Run one server-side slash command; returns the CommandResponse envelope."""
        body: dict[str, Any] = {"text": text, "frontend": FRONTEND}
        if sid:
            body["sessionId"] = sid
        return await self._post(f"/control/v1/{agent}/command", body, session_scoped=False)

    async def stream(self, sid: str, from_offset: int) -> AsyncIterator[StreamItem]:
        """Yield the session's canonical events from ``from_offset``.

        Follows ``event: reconnect`` frames (lifetime cap / backpressure) by
        resubscribing from the resume offset they carry; ends when the gateway
        closes the stream without one. ``hb`` frames surface as ``heartbeat``
        items — the gateway only emits them once the backlog is drained.
        """
        cursor = max(0, int(from_offset))
        while True:
            url = f"{self._base}/control/v1/orchestrator/sessions/{sid}/stream?{STREAM_QUERY}&from={cursor}"
            reconnect_to: int | None = None
            async with self._http.stream("GET", url, timeout=_STREAM_TIMEOUT) as response:
                if response.status_code >= 400:
                    await response.aread()
                    _raise_for(response, session_scoped=True)
                frame_type = "log"
                async for raw in response.aiter_lines():
                    line = raw.rstrip("\r")
                    if not line:
                        frame_type = "log"
                        continue
                    if line.startswith("event:"):
                        frame_type = line[len("event:") :].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    if frame_type == "hb":
                        yield StreamItem("heartbeat", cursor, {})
                        continue
                    try:
                        frame = json.loads(line[len("data:") :].strip())
                    except ValueError:
                        continue
                    if not isinstance(frame, dict):
                        continue
                    offset = frame.get("offset")
                    if frame_type == "reconnect":
                        reconnect_to = max(cursor, offset) if isinstance(offset, int) else cursor
                        break
                    if isinstance(offset, int) and offset > cursor:
                        cursor = offset
                    event = frame.get("event")
                    if isinstance(event, dict) and event.get("type"):
                        yield StreamItem("event", cursor, event)
            if reconnect_to is None:
                return
            cursor = reconnect_to
            yield StreamItem("reconnect", cursor, {})
