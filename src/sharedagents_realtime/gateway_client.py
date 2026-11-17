"""Loopback client for driving a gateway agent session from the realtime
voice server: submit a prompt, consume the canonical event stream.

The realtime server (``sharedagents_realtime``) normally talks straight to the
cascade (STT/LLM/TTS REST triples). In *agent mode* it instead drives a real
gateway agent session (e.g. ``jarvis``): create/resume the session, submit the
user's transcript as a prompt, and consume the session's canonical event log
(``text_delta`` / ``turn_ended`` / ``subagent_result``) over loopback.

A reconnecting client may resume the SAME gateway session; :meth:`session_exists`
is the resume-validation probe that decides rebind vs. fresh fallback.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

# POSTs (submit) must always be bounded so a wedged gateway can't hang a turn.
# The long-lived SSE GET overrides ``read`` to None per-request below.
_POST_TIMEOUT = httpx.Timeout(30.0)
_STREAM_TIMEOUT = httpx.Timeout(30.0, read=None)


class GatewayClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8100", token: str | None = None) -> None:
        self._base = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        # Default timeout is bounded (protects ``submit_prompt``); the event
        # stream passes its own unbounded-read timeout on the GET.
        self._client = httpx.AsyncClient(timeout=_POST_TIMEOUT, headers=headers)
        self._sessions: dict[str, str] = {}  # agent -> cached session id

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ensure_session(self, agent: str) -> str:
        """Create (or return the cached) gateway session id for ``agent``.

        Sessions are minted via ``POST /control/v1/{agent}/sessions`` (the same
        route the web app uses); the response carries ``sessionId``. The session
        is attributed to ``agent`` server-side, which the prompt-submit route
        relies on to resolve the owning backend — so this MUST run before the
        first :meth:`submit_prompt`. Cached per-agent for the client's lifetime.
        """
        cached = self._sessions.get(agent)
        if cached:
            return cached
        r = await self._client.post(
            f"{self._base}/control/v1/{agent}/sessions",
            json={"workspace_origin": "realtime"},
        )
        r.raise_for_status()
        data = r.json()
        sid = (data.get("sessionId") or data.get("session_id")) if isinstance(data, dict) else None
        if not sid:
            raise RuntimeError(f"gateway returned no sessionId for agent {agent!r}")
        self._sessions[agent] = sid
        return sid

    async def session_exists(self, sid: str) -> bool:
        """True iff the gateway still holds session ``sid`` — the resume probe.

        Reads the session header (``GET /control/v1/orchestrator/sessions/{sid}``,
        the same route the web app uses to open a session's history): 200 = the
        conversation is live and can be rebound; 404 = gone/expired. Any
        transport error is treated as *absent* so a reconnecting voice client
        falls back to a fresh session cleanly rather than wedging on a dead id.
        """
        try:
            r = await self._client.get(f"{self._base}/control/v1/orchestrator/sessions/{sid}")
        except Exception:  # noqa: BLE001 — unreachable gateway → treat as gone
            return False
        return r.status_code == 200

    async def submit_prompt(
        self,
        sid: str,
        text: str,
        *,
        voice_mode: bool = True,
    ) -> int:
        """POST a prompt; return ``start_offset`` (where this turn's events
        begin in the session log). Fire-and-forget — the turn runs server-side.

        ``voice_mode=True`` (default for realtime) asks the gateway to apply
        TTS-brevity guidance for **this turn only** — typed Jarvis on the
        same session is unaffected.
        """
        body: dict = {"prompt": [{"type": "text", "text": text}]}
        if voice_mode:
            body["voice_mode"] = True
        r = await self._client.post(
            f"{self._base}/control/v1/orchestrator/sessions/{sid}/prompt",
            json=body,
        )
        r.raise_for_status()
        return int(r.json().get("start_offset") or 0)

    async def cancel_turn(self, sid: str, force: bool = False) -> None:
        """Best-effort: cancel the server-side running turn for ``sid`` (e.g. on
        barge-in). Submit is fire-and-forget, so without this the gateway keeps
        generating. Swallows errors — cancellation must not raise into the
        caller's own teardown."""
        try:
            await self._client.post(
                f"{self._base}/control/v1/orchestrator/sessions/{sid}/cancel",
                json={"force": force},
            )
        except Exception:  # noqa: BLE001 — best-effort
            pass

    async def stream_events(self, sid: str, from_offset: int) -> AsyncIterator[dict]:
        """Yield the inner ``event`` dict of each canonical SSE ``log`` frame
        from ``from_offset``.

        Transparently follows the gateway's ``event: reconnect`` frames (emitted
        at the stream-lifetime cap or under backpressure) by resubscribing from
        the resume ``offset`` they carry — so a long turn that outlives one
        stream window still delivers every ``text_delta`` / ``turn_ended`` /
        ``subagent_result`` event. Stops when the caller breaks (e.g. on
        ``turn_ended``) or the gateway closes the stream without a reconnect.
        """
        cur = from_offset
        while True:
            url = f"{self._base}/control/v1/orchestrator/sessions/{sid}/stream?mode=events&from={cur}"
            reconnect_to: int | None = None
            async with self._client.stream("GET", url, timeout=_STREAM_TIMEOUT) as resp:
                resp.raise_for_status()
                etype = "log"  # the SSE ``event:`` type of the frame in progress
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        etype = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    try:
                        frame = json.loads(line[5:].strip())
                    except ValueError:
                        etype = "log"
                        continue
                    if etype == "reconnect":
                        # Resume cursor for the next window; never rewind (a
                        # reconnect frame may omit/under-report the offset).
                        reconnect_to = max(cur, int(frame.get("offset") or 0))
                        break
                    if etype == "log":
                        # ``offset`` is the authoritative byte cursor PAST this
                        # frame — advance it so a reconnect resumes after the
                        # last delivered event (no replay / repeated speech).
                        off = frame.get("offset")
                        if isinstance(off, int) and off > cur:
                            cur = off
                        ev = frame.get("event")
                        if isinstance(ev, dict) and ev.get("type"):
                            yield ev
                    etype = "log"  # reset before the next frame's event: line
            if reconnect_to is None:
                return
            cur = reconnect_to
