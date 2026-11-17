"""OpenAI-compatible HTTP API on the sharedagents gateway.

Mounted at ``/openai/v1`` (point clients here as their OpenAI base URL,
e.g. Open WebUI's "OpenAI Connection" panel).

Routes (matches every endpoint Open WebUI's OpenAI router touches):

    GET  /openai/v1/models             list agents as OpenAI Model objects
    POST /openai/v1/chat/completions   chat (stream + non-stream); the
                                       agent runs its full per-turn tool
                                       chain internally and surfaces the
                                       final assistant text to the caller
    POST /openai/v1/embeddings         text embeddings via the memory
                                       engine's configured embedder
    POST /openai/v1/audio/transcriptions  STT — mirrors the telegram bot's
                                       OpenRouter ``gpt-4o-transcribe``
                                       pipeline (see bot.py:591). Same
                                       ``OPENROUTER_API_KEY`` env var.
    POST /openai/v1/audio/speech       TTS — proxies to the unified ``tts``
                                       service block's OpenAI-compatible
                                       ``/v1/audio/speech`` endpoint.
    POST /openai/v1/responses          404 stub — only fires when a model
                                       is flagged as GPT-5-style; we never
                                       flag, so this is effectively dead
    GET  /openai/v1/slots              [] stub — llama.cpp probe Open WebUI
                                       fires harmlessly; satisfying it
                                       avoids log noise

Conversation model: stateless. Every ``/chat/completions`` request
spawns a fresh ACP session, runs ONE prompt turn that carries the
full history (serialised as text — the agent's bridge sees it as a
single user message containing prior context + the new question),
then closes the session. Matches Open WebUI's wire behaviour: it
sends the entire ``messages`` array on every request.

Tool calling: agents use their OWN internal tool registry (web_fetch,
memory_search, MCP, …) transparently — the caller doesn't supply
``tools`` and doesn't see ``tool_calls`` in the response. The agent
runs as many tool rounds as needed and the caller only sees the
final assistant text. Open WebUI's "Native Function Calling" mode
sends a ``tools`` field — we ignore it (consciously) for MVP.

Auth: reuses the gateway's bearer token (same ``Authorization:
Bearer …`` header that authorises ``/acp/<name>``). Empty tokens
are accepted to match Open WebUI's "no key" connection mode against
local backends.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Mapping
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from sharedagents_core.ids import bare_id

_LOG = logging.getLogger("sharedagents.openai")

# Agents are exposed to OpenAI-compatible clients as ``sharedagents/<agent>``
# so they're visually distinct from upstream model ids in tools like
# Open WebUI. Bare ``<agent>`` is also accepted on inbound requests for
# backwards compatibility with clients that have the un-prefixed name
# saved.
_MODEL_PREFIX = "sharedagents/"


def _public_model_id(agent: str) -> str:
    return f"{_MODEL_PREFIX}{agent}"


def _default_session_cwd() -> str:
    """Working directory for ephemeral ``/chat/completions`` sessions.

    Historically hardcoded to ``"/"``. That is fine for a normally-deployed
    gateway (the agent's own tool guards bound its writes), but a CONTAINED
    dev instance — e.g. the 24/7 usersim soak gateway wrapped in a
    read-only-root ``bwrap`` where only a scratch WORK_ROOT is writable —
    must start driven sessions INSIDE that writable scratch, or the agent's
    first relative-path write lands on the read-only ``/`` and fails.

    ``SHAREDAGENTS_DEFAULT_SESSION_CWD`` lets that instance pin the ephemeral cwd
    to its WORK_ROOT WITHOUT changing shipped default behaviour: unset (the
    normal case) this returns ``"/"`` exactly as before. Only a non-empty,
    existing directory is honoured — a stale/missing path silently falls
    back to ``"/"`` so a misconfigured env can never wedge session creation.
    """
    override = os.environ.get("SHAREDAGENTS_DEFAULT_SESSION_CWD", "").strip()
    if override and os.path.isdir(override):
        return override
    return "/"


def _resolve_agent_name(ctx: OpenAIContext, model_field: str) -> str | None:
    """Map an inbound ``model`` field to a known agent name, or None.

    Accepts both the prefixed form (``sharedagents/main``) and the bare form
    (``main``) so old configs keep working.
    """
    if not model_field:
        return None
    bare = bare_id(model_field)
    return bare if ctx.is_known_agent(bare) else None


# ---------------------------------------------------------------------------
# config / context types
# ---------------------------------------------------------------------------


class OpenAIContext:
    """Bundle of dependencies the route handlers need.

    Wired in :func:`make_openai_app` and stuffed onto ``request.scope``
    under the ``sharedagents_openai_ctx`` key so Starlette routes can pull
    it out without each handler taking the same five params.
    """

    def __init__(
        self,
        *,
        agent_names: list[str],
        manager: Any,
        bearer_token: str | None,
        listed_agent_names: list[str] | None = None,
    ) -> None:
        # ``agent_names`` — every RESOLVABLE agent (the gateway mount set); used
        # to route /chat/completions & /models/{id}. ``listed_agent_names`` —
        # the curated PICKER subset rendered by /v1/models (defaults to
        # ``agent_names`` for back-compat). Keeping them distinct lets the
        # gateway hide the internal ``main`` fallback from the picker while it
        # stays addressable.
        self.agent_names = list(agent_names)
        self.listed_agent_names = list(listed_agent_names) if listed_agent_names is not None else list(agent_names)
        self.manager = manager
        self.bearer_token = (bearer_token or "").strip() or None

    def is_known_agent(self, name: str) -> bool:
        return name in self.agent_names


def _ctx(request: Request) -> OpenAIContext:
    ctx = request.scope.get("sharedagents_openai_ctx")
    if not isinstance(ctx, OpenAIContext):
        # Should never happen — make_openai_app wires this up.
        raise RuntimeError("OpenAIContext missing from request scope")
    return ctx


# ---------------------------------------------------------------------------
# auth middleware
# ---------------------------------------------------------------------------


class _BearerAuth(BaseHTTPMiddleware):
    """Validate ``Authorization: Bearer <token>`` against the gateway
    token. If the gateway has no token configured, accept any request.

    Open WebUI sends an empty bearer (``Bearer ``) when the connection's
    API Key field is blank — we accept that against an unconfigured
    gateway too, so local development "just works".
    """

    def __init__(self, app: Any, *, expected_token: str | None) -> None:
        super().__init__(app)
        self._expected = (expected_token or "").strip() or None

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if self._expected is None:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse(
                _openai_error("missing bearer token", "invalid_request_error", 401),
                status_code=401,
            )
        provided = auth.split(" ", 1)[1].strip()
        if provided != self._expected:
            return JSONResponse(
                _openai_error("invalid bearer token", "invalid_request_error", 401),
                status_code=401,
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# error / response helpers
# ---------------------------------------------------------------------------


def _openai_error(message: str, type_: str, code: int | str) -> dict[str, Any]:
    """OpenAI's canonical error envelope."""
    return {
        "error": {
            "message": message,
            "type": type_,
            "param": None,
            "code": code,
        }
    }


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# message → ACP prompt translation
# ---------------------------------------------------------------------------


def _summarise_image_url(url: str) -> str:
    """Compact text marker for an image-url part.

    Open WebUI ships pasted images as ``data:image/png;base64,<HUGE>``
    URIs inside an ``image_url`` content part. Rendering the URL
    verbatim dumped ~200 KB of base64 into the user prompt for a
    typical screenshot and blew past every reasonable wire-layer cap.

    Strategy:
      * ``data:`` URI → strip body, keep the MIME + approximate byte
        size (computed from the base64 length × 0.75) so the model
        knows an image was attached.
      * HTTP/HTTPS / file URL → keep verbatim; short and the model
        may follow the link via a tool.

    This is the HISTORY-side renderer. The LIVE user message's images
    bypass this and reach the agent as proper ACP image blocks via
    :func:`build_prompt_blocks_from_messages`.
    """
    if not isinstance(url, str) or not url:
        return "[image attached]"
    if url.startswith("data:"):
        header, _, body = url.partition(",")
        mime = header.removeprefix("data:").split(";", 1)[0] or "image/?"
        approx_bytes = (len(body) * 3) // 4
        return f"[image attached: {mime}, ~{approx_bytes} bytes (stripped)]"
    return f"[image url={url}]"


def _flatten_message_content(content: Any) -> str:
    """OpenAI ``messages[i].content`` may be a string OR a list of
    content parts (text/image/etc). Reduce to a single text string.

    Image / audio parts are summarised inline via
    :func:`_summarise_image_url` so the agent knows an image was
    attached without paying the cost of carrying its bytes through.
    For the LAST (live) user message, images are extracted upstream
    as ACP image blocks (:func:`build_prompt_blocks_from_messages`)
    and the agent's `_resolve_images_for_model` handles vision
    routing — this flatten path runs only on historical context.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if not isinstance(p, dict):
                parts.append(str(p))
                continue
            t = p.get("type")
            if t == "text":
                parts.append(str(p.get("text", "")))
            elif t == "image_url":
                u = (p.get("image_url") or {}).get("url", "")
                parts.append(_summarise_image_url(u))
            elif t == "input_audio":
                parts.append("[audio attachment]")
            else:
                parts.append(f"[{t} part]")
        return "\n".join(s for s in parts if s)
    return str(content)


# Fallback chars-per-token ratio, matched to ``turn_runtime._FALLBACK_CHARS_PER_TOKEN``
# so the two trimmers (this one and the agent-side compact_messages) agree on
# rough sizes. 3.0 is conservative — dense JSON/code tokenises ~3.5–4.0,
# english prose ~4.0+; HTML / minified text dips to ~2.5.
_CHARS_PER_TOKEN_FLOOR = 3.0

# Hard cap on wire-layer input tokens, applied to every Open WebUI
# /chat/completions request before serialising. 64k chosen by operator
# directive: smaller prompts → faster TTFT and decode, and
# the cap is well below every modern long-context model's window so the
# upstream context-length 400 class can't trigger at all. No cfg
# lookup, no iteration, no upstream-error parsing — one clamp. A
# cfg.upstream_max_context driven budget silently skips the trim for the
# many agents that leave that field `null`. A hardcoded constant removes
# the configuration dependency.
OPENAI_ENDPOINT_INPUT_TOKEN_CAP = 64_000
OPENAI_ENDPOINT_INPUT_CHAR_CAP = int(
    OPENAI_ENDPOINT_INPUT_TOKEN_CAP * _CHARS_PER_TOKEN_FLOOR,
)


def _message_chars(msg: Mapping[str, Any]) -> int:
    """Char count of a single OpenAI ``messages[i]`` entry.

    Handles both the string AND multipart-list content shapes (vision
    requests: a list of ``{type: "text"/"image_url", …}`` parts).
    Image-url base64 bodies count at their full length so a 200 KB
    image isn't undercounted.
    """
    content = msg.get("content")
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for p in content:
            if isinstance(p, str):
                total += len(p)
            elif isinstance(p, Mapping):
                t = p.get("text")
                if isinstance(t, str):
                    total += len(t)
                u = p.get("image_url")
                if isinstance(u, Mapping):
                    uu = u.get("url")
                    if isinstance(uu, str):
                        total += len(uu)
                elif isinstance(u, str):
                    total += len(u)
        return total
    return len(str(content))


def _truncate_message_content(content: Any, max_len: int) -> Any:
    """Return ``content`` truncated so its ``_message_chars`` value <= ``max_len``.

    Handles the string AND multipart-list content shapes. For a list, text is
    kept oldest-first until the char budget is exhausted (a final text part is
    truncated to the remaining budget); non-text parts that don't fit whole
    (e.g. an oversized base64 image) are dropped rather than corrupted.
    """
    if max_len <= 0:
        return ""
    if content is None:
        return content
    if isinstance(content, str):
        return content if len(content) <= max_len else content[:max_len]
    if isinstance(content, list):
        out: list[Any] = []
        remaining = max_len
        for part in content:
            if remaining <= 0:
                break
            if isinstance(part, str):
                take = part[:remaining]
                out.append(take)
                remaining -= len(take)
            elif isinstance(part, Mapping):
                plen = _message_chars({"content": [part]})
                if plen <= remaining:
                    out.append(part)
                    remaining -= plen
                else:
                    t = part.get("text")
                    if isinstance(t, str):
                        newpart = dict(part)
                        newpart["text"] = t[:remaining]
                        out.append(newpart)
                        remaining = 0
                    # oversized non-text part (e.g. image_url): drop it whole
        return out
    return str(content)[:max_len]


def trim_messages_to_char_budget(
    messages: list[Mapping[str, Any]],
    max_chars: int,
) -> tuple[list[Mapping[str, Any]], int]:
    """Drop oldest non-protected messages until the total content fits.

    Why: Open WebUI sends the FULL chat history on every
    ``/chat/completions`` request. ``_serialise_history_as_prompt`` then
    collapses the array into a single user-prompt text — if the result
    exceeds the agent's upstream context window, vLLM returns a
    `maximum context length is N tokens` 400 that the agent-side
    reactive squeeze can't recover from because everything is already
    one protected user message. Trimming the wire-format ``messages``
    list BEFORE collapsing is the only point where dropping older
    turns is still meaningful.

    Protected from dropping:
      * All leading ``role=system`` messages (caller-supplied system
        context — semantically setup, not history).
      * The LAST message in the array (the live turn — must survive
        every stage, mirroring `compact_messages`'s ``keep_tail``
        invariant).

    Drop strategy:
      * Walks the droppable head oldest-first.
      * When the oldest droppable is a ``user`` immediately followed
        by an ``assistant``, drops them as a PAIR so the trimmed
        history never starts with a dangling assistant reply (which
        makes no semantic sense without its preceding user prompt and
        confuses some chat templates).
      * Continues until the running total ≤ ``max_chars`` OR no
        droppable candidates remain. Caller decides what to do if even
        the protected pair overflows — typically falls through to the
        agent-side ``max_tokens_override`` reactive squeeze.

    Returns ``(trimmed_messages, dropped_count)``. ``dropped_count``
    lets the caller surface a one-line breadcrumb in the prompt text
    (``"[N earlier messages truncated to fit context]"``).
    """
    if not messages:
        return messages, 0

    # Contiguous leading system block.
    sys_end = 0
    for m in messages:
        if (m.get("role") or "").lower() == "system":
            sys_end += 1
        else:
            break
    head_sys = list(messages[:sys_end])
    rest = list(messages[sys_end:])

    if not rest:
        return list(messages), 0

    # Protected tail: the LAST message in the original array.
    protected_tail = [rest[-1]]
    droppable = rest[:-1]

    def total_chars() -> int:
        return sum(_message_chars(m) for m in (head_sys + droppable + protected_tail))

    if total_chars() <= max_chars:
        return list(messages), 0

    dropped = 0
    while droppable and total_chars() > max_chars:
        # Drop the oldest. If it leaves an assistant at the new head,
        # drop that too so the trimmed history doesn't open with a
        # dangling reply.
        droppable.pop(0)
        dropped += 1
        if droppable and (droppable[0].get("role") or "").lower() == "assistant":
            droppable.pop(0)
            dropped += 1

    retained = head_sys + droppable + protected_tail
    # The input cap is HARD and cannot be bypassed. The drop
    # loop above never removes the protected system block or the live-turn
    # tail, so a single oversized protected message would otherwise sail past
    # the cap. When the retained set still exceeds the budget, truncate message
    # CONTENT (oldest-first) so the total never exceeds ``max_chars``.
    if sum(_message_chars(m) for m in retained) > max_chars:
        clamped: list[Mapping[str, Any]] = []
        remaining = max_chars
        for m in retained:
            c = _message_chars(m)
            if c <= remaining:
                clamped.append(m)
                remaining -= c
            else:
                nm = dict(m)
                nm["content"] = _truncate_message_content(m.get("content"), remaining)
                clamped.append(nm)
                remaining = 0
        retained = clamped
    return retained, dropped


def _serialise_history_as_prompt(messages: list[Mapping[str, Any]]) -> str:
    """Bake the OpenAI ``messages`` array into a single agent prompt.

    Stateless backend, OpenWebUI sends full history every turn. Rather
    than spawn N synthetic prompt rounds (one per prior turn) — which
    would burn N model calls AND poison the agent's session.turns —
    we collapse history into ONE structured text block prefixed with
    role tags. The agent's first turn then handles the whole thing.

    This is lossy versus injecting native ACP turn records, but it
    keeps the OpenAI integration to a single file with no acp_bridge
    surface changes. Acceptable because:

      * Open WebUI's title / tags / follow-up generation calls send
        only 2-3 messages — short histories anyway.
      * Long chats reach the agent with one big prompt that the model
        sees as well-structured prior context — close enough to native
        multi-turn for most reasoning tasks.

    Phase 2 (later, separate feature): add a ``new_session_with_turns``
    backend method that pre-populates ``sess.turns`` so KV cache prefix
    can be reused across requests with the same history prefix.
    """
    if not messages:
        return ""

    # Pull leading system message(s) into a header — they semantically
    # live "before" the conversation and the agent's bridge already
    # has its own system prompt; treat the caller's system as guidance
    # appended to the existing system, not replacing it.
    sys_lines: list[str] = []
    convo: list[Mapping[str, Any]] = []
    seen_non_system = False
    for m in messages:
        role = (m.get("role") or "").lower()
        if role == "system" and not seen_non_system:
            txt = _flatten_message_content(m.get("content"))
            if txt:
                sys_lines.append(txt)
        else:
            seen_non_system = True
            convo.append(m)

    if len(convo) == 1 and (convo[0].get("role") or "").lower() == "user" and not sys_lines:
        # Single-shot prompt — return it as-is, no history framing.
        return _flatten_message_content(convo[0].get("content"))

    out: list[str] = []
    if sys_lines:
        out.append("[Caller-supplied system context]")
        out.extend(sys_lines)
        out.append("")

    if len(convo) > 1:
        out.append("[Prior conversation]")
        for m in convo[:-1]:
            role = (m.get("role") or "").upper()
            txt = _flatten_message_content(m.get("content"))
            if txt:
                out.append(f"{role}: {txt}")
        out.append("")

    last = convo[-1] if convo else None
    if last is not None:
        last_role = (last.get("role") or "").upper()
        last_txt = _flatten_message_content(last.get("content"))
        if last_role == "USER":
            out.append(f"[Current message from user]\n{last_txt}")
        else:
            # Trailing assistant / tool message — happens with Open WebUI
            # tool-call retries. Treat as continuation context.
            out.append(f"[Conversation continues — last {last_role.lower()}]\n{last_txt}")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Live-image extraction → ACP prompt blocks
# ---------------------------------------------------------------------------


def _extract_image_blocks_from_content(content: Any) -> list[dict[str, Any]]:
    """Pull ``image_url`` parts out of an OpenAI content list and
    translate to ACP image blocks (``{type:"image", data, mimeType}``).

    Only data: URIs are converted — that's where Open WebUI pastes
    user-supplied images. HTTP/HTTPS URLs would need a fetch step
    we don't want to do server-side, so they pass through to the
    flatten path as a text reference instead.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(content, list):
        return out
    for p in content:
        if not isinstance(p, Mapping):
            continue
        if p.get("type") != "image_url":
            continue
        url = (p.get("image_url") or {}).get("url", "")
        if not isinstance(url, str) or not url.startswith("data:"):
            continue
        header, _, body = url.partition(",")
        mime = header.removeprefix("data:").split(";", 1)[0] or "image/jpeg"
        if not body:
            continue
        out.append(
            {
                "type": "image",
                "data": body,
                "mimeType": mime,
            }
        )
    return out


def build_prompt_blocks_from_messages(
    messages: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert an OpenAI ``messages[]`` array into ACP prompt blocks.

    Composition:
      * One ``text`` block carrying the serialised history + live
        user prompt (image parts in history become text placeholders
        via :func:`_summarise_image_url`).
      * Zero or more ``image`` blocks extracted from the LAST user
        message's content list. The agent's
        :func:`acp_bridge._blocks_to_user_content` then assembles
        them into OpenAI ``image_url`` parts for the worker, and
        :func:`acp_bridge._resolve_images_for_model` either forwards
        to a vision-capable upstream or transcribes for a text-only
        one.

    Why only the LAST message? Historical images are typically not
    re-shipped to vision models (most local long-context models
    handle vision as a single-turn signal); sending every historical
    image would explode prompt size and likely 400 on text-only
    upstreams. The agent's vision pipeline focuses on the live turn.
    """
    if not messages:
        return [{"type": "text", "text": ""}]
    last_idx = len(messages) - 1
    last = messages[last_idx] if (messages[last_idx].get("role") or "").lower() == "user" else None
    image_blocks: list[dict[str, Any]] = []
    if last is not None:
        image_blocks = _extract_image_blocks_from_content(last.get("content"))
    # History serialisation: the live user message's text parts STILL
    # need to land in the text block, but its image parts must NOT
    # be re-rendered (we already extracted them). Substitute the live
    # user's content with text-only parts before serialising.
    if last is not None and image_blocks:
        text_only_parts = []
        for p in last.get("content") or []:
            if isinstance(p, Mapping) and p.get("type") == "image_url":
                continue
            text_only_parts.append(p)
        messages = list(messages)
        messages[last_idx] = {**last, "content": text_only_parts or ""}
    prompt_text = _serialise_history_as_prompt(list(messages))
    blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    blocks.extend(image_blocks)
    return blocks


# ---------------------------------------------------------------------------
# ACP backend driver — runs ONE turn against an agent
# ---------------------------------------------------------------------------


async def _run_one_turn(
    *,
    manager: Any,
    agent: str,
    prompt_blocks: list[Mapping[str, Any]],
    session_id: str | None = None,
    sampling: dict[str, Any] | None = None,
    pinned_title: str = "",
):
    """Drive ONE prompt turn against ``agent`` via the in-process
    :class:`AgentACPBackend` registry.

    The backend is in-process; no per-agent stdio child subprocess.
    Output shape matches the ACP wire (``sessionUpdate`` dicts +
    terminal ``PromptResponse``) so downstream consumers can rely on
    the same schema as a remote ACP call.

    ``prompt_blocks`` is an ACP ContentBlock array (text + optional
    image blocks). Built by :func:`build_prompt_blocks_from_messages`
    so live-message images reach the agent intact and route through
    :func:`acp_bridge._resolve_images_for_model`.

    Session binding (opt-in, ``X-Sharedagents-Session-Id`` header): when
    ``session_id`` is provided, reattach to that PERSISTENT session
    (load, not new) and do NOT close it afterward — the gateway session
    owns the conversation history, so a session-bound frontend (the web
    app) sends only the NEW user message each turn and the engine
    accumulates history exactly like the TUI/Telegram. With no
    ``session_id`` the legacy ephemeral behavior is unchanged: a fresh
    session is created and closed per request (Open WebUI stays
    stateless).
    """
    backend = manager.get(agent)
    bound = bool(session_id)
    if bound:
        try:
            await backend.load_session(session_id, "/", [])
        except Exception as exc:  # noqa: BLE001
            # Session not materialized yet (or load unsupported). This
            # shouldn't happen — the web creates the session via
            # /control/v1/{agent}/sessions before its first bound chat — so
            # log LOUDLY rather than silently hiding a binding bug, then
            # fall back to an ephemeral session so the turn still runs.
            _LOG.warning(
                "bound session %r load failed (%s); falling back to ephemeral",
                session_id,
                exc,
            )
            new = await backend.new_session(_default_session_cwd(), [])
            session_id = new.get("sessionId") or new.get("session_id")
            bound = False
    else:
        new = await backend.new_session(_default_session_cwd(), [])
        # ACP wire shape uses ``sessionId`` (camelCase); legacy
        # CoreBackend.new_session also returns ``sessionId``.
        session_id = new.get("sessionId") or new.get("session_id")
    if bound and pinned_title:
        result = await backend.set_session_title(session_id, pinned_title, lock=True)
        if not isinstance(result, Mapping) or not result.get("ok"):
            raise RuntimeError("task title pin failed")
    prompt_args = (session_id, None, list(prompt_blocks))
    if sampling:
        try:
            prompt_stream = backend.prompt(*prompt_args, sampling=dict(sampling))
        except TypeError:
            # Remote/third-party ACP backends implement the standard prompt
            # signature and have no sampling extension. Keep them compatible;
            # CoreBackend accepts the structured per-turn kwarg directly.
            prompt_stream = backend.prompt(*prompt_args)
    else:
        prompt_stream = backend.prompt(*prompt_args)
    try:
        async for ev in prompt_stream:
            # Enrich the terminal PromptResponse (the one non-sessionUpdate
            # event) with the engine's ACTUAL model for SESSION-BOUND turns
            # (the sharedagents web app). External CLI/ACP engines fold their
            # real id (claude-opus-4-…/gpt-5.5-codex/composer-2.5) into the
            # session header at turn end, so the bound session now carries
            # it — surface it as ``_engine_model`` so the completion can
            # echo the model the agent actually used. Left untouched for
            # ephemeral / Open WebUI turns (``bound`` is False) so the
            # stateless OpenAI contract keeps echoing the requested id.
            if bound and not _is_session_update(ev):
                try:
                    # Read the engine model from the PERSISTED header, not an
                    # in-memory session object: ``Agent.run_turn`` mutates a
                    # session built fresh from the row each turn (and folds
                    # ``engine_model`` into the header via a synchronous
                    # ``header_update`` BEFORE this terminal frame is yielded),
                    # which is NOT the bridge's cached ``_session_for`` object.
                    # ``agent.get_session`` tails the JSONL → reflects that
                    # just-written value (concrete id, or "" when cleared).
                    _agent_obj = getattr(backend, "agent", None)
                    _sess = _agent_obj.get_session(session_id) if _agent_obj is not None else None
                    _mdl = str(getattr(_sess, "engine_model", "") or "") if _sess is not None else ""
                    if _mdl:
                        ev = {**ev, "_engine_model": _mdl}
                        # The operator-facing label for the wire id. The
                        # config owns the alias mapping, so resolve it here
                        # (the web has no config of its own); unknown ids
                        # (external CLI models) get "" and the UI falls back
                        # to the raw id.
                        try:
                            _cfg = getattr(_agent_obj, "cfg", None)
                            if _cfg is not None:
                                _alias = str(_cfg.alias_for_model_id(_mdl) or "")
                                if _alias:
                                    ev = {
                                        **ev,
                                        "_engine_model_alias": _alias,
                                    }
                        except Exception:  # noqa: BLE001
                            pass
                except Exception:  # noqa: BLE001
                    pass
            yield ev
    finally:
        if not bound:
            try:
                await backend.close_session(session_id)
            except Exception:  # noqa: BLE001
                pass


def _is_session_update(payload: Mapping[str, Any]) -> bool:
    return isinstance(payload, dict) and "sessionUpdate" in payload


def _extract_text_chunk(update: Mapping[str, Any]) -> str:
    """Pull text content out of a ``session/update`` payload that
    represents a model output chunk. Returns empty when the update is
    something else (tool-call start, plan, etc.) — the caller skips it.
    """
    t = update.get("sessionUpdate")
    if t == "agent_message_chunk":
        c = update.get("content") or {}
        if isinstance(c, dict) and c.get("type") == "text":
            return str(c.get("text") or "")
    # ACP also emits ``user_message_chunk`` for echoed user content
    # and various tool_call events; don't surface those as completion
    # text — they'd corrupt the openai-compat response.
    return ""


def format_session_updates_for_openai(
    update: Mapping[str, Any] | None,
    state: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    """Project one ACP update onto OpenAI answer-content deltas.

    Only text from ``agent_message_chunk`` is answer content. Structured
    thought, tool, plan, user, usage, unknown, and flush events produce no
    chunks. ``state`` is retained solely for source compatibility with callers
    of the former stateful presentation wrapper.
    """
    next_state = dict(state or {})
    if update is None:
        return [], next_state
    text = _extract_text_chunk(update)
    return ([text] if text else []), next_state


def _terminal_finish_reason(terminal: Mapping[str, Any]) -> str:
    """Map ACP stop reasons into OpenAI ``finish_reason`` values."""
    raw = (terminal.get("stopReason") or "").lower()
    if raw == "end_turn":
        return "stop"
    if raw == "max_turn_requests":
        return "length"
    if raw == "max_tokens":
        return "length"
    if raw == "cancelled":
        return "stop"  # Open WebUI doesn't model 'cancelled'; closest is stop
    if raw == "refusal":
        return "content_filter"
    return "stop"


# ---------------------------------------------------------------------------
# /chat/completions
# ---------------------------------------------------------------------------


async def _handle_chat_completions(request: Request) -> Response:
    ctx = _ctx(request)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(
            _openai_error("invalid JSON body", "invalid_request_error", 400),
            status_code=400,
        )

    model_field = (body.get("model") or "").strip()
    messages = body.get("messages") or []
    stream = bool(body.get("stream"))
    sampling = dict(body["sampling"]) if isinstance(body.get("sampling"), Mapping) else None

    # Opt-in session binding (sharedagents extension): a session-aware
    # frontend (the web app) sends ``X-Sharedagents-Session-Id`` so chat runs
    # on a PERSISTENT gateway session — the same one its /control/v1
    # commands target — giving /context, /history, /todo, /mode, /model
    # etc. true parity. Absent the header, the endpoint stays stateless
    # (Open WebUI and other OpenAI clients are unaffected). The session
    # owns history, so a bound frontend sends only the new user message.
    bound_session_id = (request.headers.get("x-sharedagents-session-id") or "").strip() or None
    pinned_title = str(body.get("task_title") or "").strip()[:120] if bound_session_id else ""

    if not model_field:
        return JSONResponse(
            _openai_error("missing 'model' field", "invalid_request_error", 400),
            status_code=400,
        )
    model = _resolve_agent_name(ctx, model_field)
    if model is None:
        known = ", ".join(_public_model_id(n) for n in ctx.agent_names) or "(none)"
        return JSONResponse(
            _openai_error(
                f"unknown agent {model_field!r} (known: {known})",
                "invalid_request_error",
                "model_not_found",
            ),
            status_code=404,
        )
    if not isinstance(messages, list) or not messages:
        return JSONResponse(
            _openai_error("'messages' must be a non-empty list", "invalid_request_error", 400),
            status_code=400,
        )

    # Hard-cap wire-layer input at 64k tokens for faster responses.
    # Open WebUI ships the FULL chat array on every request; long
    # conversations easily blow past any local-model context window.
    # A cfg-driven budget silently skips the trim for agents that leave
    # ``cfg.upstream_max_context`` `null`. A hardcoded cap removes the
    # configuration dependency, keeps TTFT low, and stays well below
    # every modern long-context window so the upstream 400 class
    # can't trigger. ``OPENAI_ENDPOINT_INPUT_CHAR_CAP`` = 64k tokens
    # × 3 chars/token = 192k chars.
    trimmed, dropped = trim_messages_to_char_budget(
        messages,
        OPENAI_ENDPOINT_INPUT_CHAR_CAP,
    )
    if dropped > 0:
        _LOG.info(
            "openai-endpoint trimmed %d earlier messages to last %dk tokens (agent=%s)",
            dropped,
            OPENAI_ENDPOINT_INPUT_TOKEN_CAP // 1000,
            model,
        )
        messages = list(trimmed)
        # Surface a one-line breadcrumb in the prompt so the model
        # knows context was cut (otherwise it might cite missing
        # context with confidence).
        messages.insert(
            len(messages) - 1,
            {
                "role": "system",
                "content": (
                    f"[note: {dropped} earlier message(s) were "
                    f"truncated to keep input under "
                    f"{OPENAI_ENDPOINT_INPUT_TOKEN_CAP // 1000}k tokens "
                    "for faster response]"
                ),
            },
        )

    # Build ACP prompt blocks: serialised history text + any image
    # blocks extracted from the LAST user message. Images bypass
    # the text flattener and reach the agent intact so the vision
    # pipeline (acp_bridge._resolve_images_for_model) can route them
    # to a vision-capable model or transcribe for text-only ones.
    prompt_blocks = build_prompt_blocks_from_messages(messages)
    text_parts_total = sum(len(str(b.get("text", ""))) for b in prompt_blocks if b.get("type") == "text")
    image_parts_total = sum(1 for b in prompt_blocks if b.get("type") == "image")
    if text_parts_total == 0 and image_parts_total == 0:
        return JSONResponse(
            _openai_error("messages contained no usable content", "invalid_request_error", 400),
            status_code=400,
        )

    completion_id = _completion_id()
    created = int(time.time())

    if stream:
        return StreamingResponse(
            _stream_chat_completion(
                manager=ctx.manager,
                agent=model,
                prompt_blocks=prompt_blocks,
                completion_id=completion_id,
                created=created,
                session_id=bound_session_id,
                sampling=sampling,
                **({"pinned_title": pinned_title} if pinned_title else {}),
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: collect only genuine answer chunks plus the final stop
    # reason. Activity remains on the canonical session stream and is never
    # promoted into the OpenAI answer projection.
    full_text_parts: list[str] = []
    terminal: Mapping[str, Any] = {}
    # Capture the last usage_update for real token metrics (not 0/0/0).
    last_usage_metrics: dict[str, Any] = {}
    try:
        async for ev in _run_one_turn(
            manager=ctx.manager,
            agent=model,
            prompt_blocks=prompt_blocks,
            session_id=bound_session_id,
            sampling=sampling,
            **({"pinned_title": pinned_title} if pinned_title else {}),
        ):
            if _is_session_update(ev):
                # Preserve the engine's real token counts from ACP usage_update
                # events — format_session_updates_for_openai extracts only the
                # text/content parts; the metrics ride alongside on the same
                # sessionUpdate but were silently dropped before.
                if ev.get("sessionUpdate") == "usage_update":
                    _meta = (ev.get("_meta") or {}).get("sharedagents", {}).get("metrics", {})
                    if _meta:
                        last_usage_metrics = dict(_meta)
                pieces, _ = format_session_updates_for_openai(ev, None)
                for p in pieces:
                    full_text_parts.append(p)
            else:
                terminal = ev  # terminal PromptResponse
                break
    except Exception as e:  # noqa: BLE001
        _LOG.exception("chat/completions failed for agent %s", model)
        return JSONResponse(
            _openai_error(f"agent error: {e}", "internal_server_error", 500),
            status_code=500,
        )

    content = "".join(full_text_parts)
    # Echo the engine's real model for session-bound (web) turns; fall
    # back to the public agent id for stateless / Open WebUI requests.
    _engine_model = str(terminal.get("_engine_model") or "").strip()
    _engine_model_alias = str(terminal.get("_engine_model_alias") or "").strip()
    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": _engine_model or _public_model_id(model),
            "model_alias": _engine_model_alias,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": _terminal_finish_reason(terminal),
                }
            ],
            "usage": {
                "prompt_tokens": int(last_usage_metrics.get("prompt_tokens") or 0),
                "completion_tokens": int(last_usage_metrics.get("completion_tokens") or 0),
                "total_tokens": int(last_usage_metrics.get("total_tokens") or 0),
            },
        }
    )


def _sse_chunk(payload: Mapping[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


async def _stream_chat_completion(
    *,
    manager: Any,
    agent: str,
    prompt_blocks: list[Mapping[str, Any]],
    completion_id: str,
    created: int,
    session_id: str | None = None,
    sampling: dict[str, Any] | None = None,
    pinned_title: str = "",
):
    """Async generator that produces strict OpenAI SSE chunks from an
    ACP prompt stream.

    Frame contract (Open WebUI is strict about this):
        data: {chunk-json}\\n\\n
        data: {chunk-json}\\n\\n
        ...
        data: [DONE]\\n\\n
    """
    public_model = _public_model_id(agent)
    # Initial chunk: delta with role only — matches OpenAI's first
    # SSE frame when a new completion starts.
    yield _sse_chunk(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": public_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
    )

    finish_reason = "stop"
    engine_model = ""
    engine_model_alias = ""
    try:
        async for ev in _run_one_turn(
            manager=manager,
            agent=agent,
            prompt_blocks=prompt_blocks,
            session_id=session_id,
            sampling=sampling,
            **({"pinned_title": pinned_title} if pinned_title else {}),
        ):
            if _is_session_update(ev):
                pieces, _ = format_session_updates_for_openai(ev, None)
                for text in pieces:
                    if not text:
                        continue
                    yield _sse_chunk(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": public_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            else:
                finish_reason = _terminal_finish_reason(ev)
                engine_model = str(ev.get("_engine_model") or "").strip()
                engine_model_alias = str(ev.get("_engine_model_alias") or "").strip()
                break
    except Exception as e:  # noqa: BLE001
        _LOG.exception("streaming chat/completions failed for agent %s", agent)
        # Emit the error as a final SSE event with finish_reason=error
        # AND end the stream cleanly so Open WebUI doesn't stall.
        yield _sse_chunk(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": public_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"\n\n[agent error: {e}]"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        yield b"data: [DONE]\n\n"
        return

    # Terminal chunk: empty delta + finish_reason. For session-bound
    # (web) turns we echo the engine's REAL model here — claude-opus-4-…,
    # gpt-5.5-codex, composer-2.5 — so the web can capture the model the
    # agent actually used. Content chunks keep ``public_model`` (the agent
    # id) so a streaming client reading the first frame is unaffected.
    yield _sse_chunk(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": engine_model or public_model,
            "model_alias": engine_model_alias,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        }
    )
    yield b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------


async def _handle_models(request: Request) -> Response:
    ctx = _ctx(request)
    created = int(time.time())
    # Discover the picker roster LIVE per request (never a boot-time snapshot) so
    # an agent added to / removed from ~/.sharedagents/agents shows up / disappears
    # immediately, with no gateway restart. Falls back to the boot list only if
    # live discovery is unavailable (e.g. a north/south split where this process
    # can't read the agents folder).
    try:
        from sharedagents_gateway.gateway import _discover_agents

        names = _discover_agents()
    except Exception:  # noqa: BLE001
        names = list(ctx.listed_agent_names)
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": _public_model_id(name),
                    "object": "model",
                    "created": created,
                    "owned_by": "sharedagents",
                }
                for name in names
            ],
        }
    )


async def _handle_model_get(request: Request) -> Response:
    ctx = _ctx(request)
    raw = request.path_params.get("model_id", "")
    name = _resolve_agent_name(ctx, raw)
    if name is None:
        return JSONResponse(
            _openai_error(f"unknown agent {raw!r}", "invalid_request_error", "model_not_found"),
            status_code=404,
        )
    return JSONResponse(
        {
            "id": _public_model_id(name),
            "object": "model",
            "created": int(time.time()),
            "owned_by": "sharedagents",
        }
    )


# ---------------------------------------------------------------------------
# /embeddings — uses the memory engine's resolved embedder
# ---------------------------------------------------------------------------


async def _handle_embeddings(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(
            _openai_error("invalid JSON body", "invalid_request_error", 400),
            status_code=400,
        )

    inputs = body.get("input")
    model = (body.get("model") or "").strip() or "sharedagents-default"

    if isinstance(inputs, str):
        texts = [inputs]
    elif isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
        texts = inputs
    else:
        return JSONResponse(
            _openai_error(
                "'input' must be a string or list of strings",
                "invalid_request_error",
                400,
            ),
            status_code=400,
        )
    if not texts:
        return JSONResponse(
            _openai_error("'input' cannot be empty", "invalid_request_error", 400),
            status_code=400,
        )

    # Resolve the embedder over the memory WIRE (WireEmbedder
    # proxy), never the in-process engine. None → no embedder configured
    # server-side → 501, same contract as before.
    from sharedagents_mcp.memory.client import get_memory_client

    embedder = await get_memory_client().embedder()
    if embedder is None:
        return JSONResponse(
            _openai_error(
                "no embedder configured (set memory.embedder in sharedagents config)",
                "service_unavailable",
                501,
            ),
            status_code=501,
        )

    try:
        vectors = await embedder.embed_batch(texts)
    except Exception as e:  # noqa: BLE001
        _LOG.exception("embedding failed")
        return JSONResponse(
            _openai_error(f"embedding failed: {e}", "internal_server_error", 500),
            status_code=500,
        )

    # A wire timeout/error yields [] (fail-safe) — but returning HTTP 200 with no
    # embeddings for nonempty input is a malformed success. Surface a bounded
    # explicit error instead.
    if texts and len(vectors) != len(texts):
        return JSONResponse(
            _openai_error(
                "embedding unavailable: the memory wire returned no vectors (timeout or embedder error)",
                "service_unavailable",
                503,
            ),
            status_code=503,
        )

    # Estimate token count loosely — we don't have a tokenizer-bound
    # number; len(text) // 4 is the standard cheap heuristic.
    prompt_tokens = sum(len(t) // 4 for t in texts)
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": vec,
                }
                for i, vec in enumerate(vectors)
            ],
            "model": model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }
    )


# ---------------------------------------------------------------------------
# /audio/transcriptions — STT (mirrors telegram bot's OpenRouter pattern)
# ---------------------------------------------------------------------------
#
# Wire-equivalent to ``sharedagents_telegram.bot._make_default_openrouter_stt``:
# accept audio, base64-encode + post to OpenRouter's whisper-compatible
# ``gpt-4o-transcribe`` endpoint, return the transcript. Open WebUI hits
# this when its STT engine is set to "OpenAI" against the sharedagents
# connection. Logic is duplicated rather than extracted from bot.py so a
# refactor of the OpenAI endpoint can never break the production
# telegram bot's voice path.

# STT endpoints (primary/backup) come from the unified sharedagents config "stt"
# block via sharedagents_core.services — no hardcoded provider URL/model here.
_STT_FORMAT_BY_MIME = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}
_STT_FORMAT_BY_EXT = {".mp3": "mp3", ".wav": "wav"}
_STT_TRANSCODE_FROM_MIME = frozenset(
    {
        "audio/ogg",
        "audio/oga",
        "audio/opus",
        "audio/mp4",
        "audio/x-m4a",
        "audio/m4a",
        "audio/aac",
        "video/mp4",
        "video/quicktime",
        "video/webm",
    }
)
_STT_TRANSCODE_FROM_EXT = frozenset(
    {
        ".ogg",
        ".oga",
        ".opus",
        ".m4a",
        ".aac",
        ".mp4",
        ".mov",
        ".webm",
    }
)


def _stt_format_for(mime: str, filename: str) -> str | None:
    if mime:
        f = _STT_FORMAT_BY_MIME.get(mime.lower().split(";")[0].strip())
        if f:
            return f
    if filename:
        from pathlib import Path as _P

        f = _STT_FORMAT_BY_EXT.get(_P(filename).suffix.lower())
        if f:
            return f
    return None


def _stt_needs_transcode(mime: str, filename: str) -> bool:
    if mime and mime.lower().split(";")[0].strip() in _STT_TRANSCODE_FROM_MIME:
        return True
    if filename:
        from pathlib import Path as _P

        if _P(filename).suffix.lower() in _STT_TRANSCODE_FROM_EXT:
            return True
    return False


def _resolve_ffmpeg() -> str | None:
    import shutil as _shutil

    p = _shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def _transcode_to_mp3(audio_bytes: bytes) -> bytes | None:
    import subprocess as _sub

    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return None
    try:
        proc = _sub.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "4",
                "-f",
                "mp3",
                "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
            timeout=60,
        )
    except (_sub.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout


async def _handle_audio_transcriptions(request: Request) -> Response:
    """OpenAI-compatible STT. Accepts ``multipart/form-data`` with a
    ``file`` field (the audio) plus optional ``model`` / ``language`` /
    ``prompt`` / ``response_format`` fields. Returns ``{"text": "…"}``
    by default; ``response_format=text`` returns plain text.
    """
    from sharedagents_core.services import resolve_service

    svc = resolve_service("stt")
    _mismatch = svc.model_mismatch()
    if _mismatch:
        _LOG.warning("%s", _mismatch)
    if not svc.configured:
        return JSONResponse(
            _openai_error(
                'STT not configured: add an "stt" block (primary/backup) to ~/.sharedagents/config.json',
                "service_unavailable",
                501,
            ),
            status_code=501,
        )

    try:
        form = await request.form()
    except Exception:  # noqa: BLE001
        return JSONResponse(
            _openai_error("expected multipart/form-data", "invalid_request_error", 400),
            status_code=400,
        )
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse(
            _openai_error("missing 'file' upload field", "invalid_request_error", 400),
            status_code=400,
        )
    audio_bytes = await upload.read()  # type: ignore[attr-defined]
    filename = getattr(upload, "filename", "") or ""
    mime = getattr(upload, "content_type", "") or ""
    response_format = (form.get("response_format") or "json").lower()

    if not audio_bytes:
        return JSONResponse(
            _openai_error("audio file is empty", "invalid_request_error", 400),
            status_code=400,
        )

    # Self-hosted Qwen3-ASR (vLLM) accepts mp3/wav/ogg directly; m4a/webm/mp4 etc.
    # are transcoded to mp3 first. The wire format is OpenAI-compatible MULTIPART
    # (file + model), NOT OpenRouter's base64-JSON.
    send_name = filename or "audio"
    if _stt_format_for(mime, filename) is None and _stt_needs_transcode(mime, filename):
        transcoded = _transcode_to_mp3(audio_bytes)
        if transcoded:
            audio_bytes = transcoded
            send_name = "audio.mp3"
        else:
            return JSONResponse(
                _openai_error(
                    f"audio format {mime or filename!r} needs transcode but ffmpeg is unavailable",
                    "invalid_request_error",
                    400,
                ),
                status_code=400,
            )
        # else: unknown format — let the STT server decide (it may still accept it).

    # Try the PRIMARY then BACKUP endpoint (async failover over svc.candidates()).
    import httpx

    errors: list[str] = []
    text: str | None = None
    for ep in svc.candidates():
        files = {"file": (send_name, audio_bytes, mime or "application/octet-stream")}
        data = {"model": ep.model} if ep.model else {}
        headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                r = await client.post(ep.url("audio/transcriptions"), files=files, data=data, headers=headers)
            if r.status_code != 200:
                errors.append(f"{ep.label}({ep.base_url}): HTTP {r.status_code} {r.text[:200]}")
                continue
            text = (r.json().get("text") or "").strip()
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f"{ep.label}({ep.base_url}): {e}")
            continue
    if text is None:
        _LOG.warning("STT failed on all endpoints: %s", " | ".join(errors))
        return JSONResponse(
            _openai_error("STT failed — " + " | ".join(errors), "internal_server_error", 502),
            status_code=502,
        )

    if response_format == "text":
        return PlainTextResponse(text)
    # OpenAI's default JSON shape for whisper.
    return JSONResponse({"text": text})


# ---------------------------------------------------------------------------
# /audio/speech — TTS via unified service config
# ---------------------------------------------------------------------------
#
# Default model: ``openai/gpt-4o-mini-tts-2025-12-15``. OpenRouter
# requires the **dated** slug; the un-dated ``openai/gpt-4o-mini-tts``
# returns HTTP 400 "Model … does not exist".

_TTS_DEFAULT_MODEL = "openai/gpt-4o-mini-tts-2025-12-15"


def _tts_speech_url(ep: Any) -> str:
    base = ep.base_url.rstrip("/")
    if base.endswith("/audio/speech"):
        return base
    return ep.url("audio/speech")


async def _handle_audio_speech(request: Request) -> Response:
    from sharedagents_core.services import resolve_service

    svc = resolve_service("tts")
    if not svc.configured:
        return JSONResponse(
            _openai_error(
                'TTS unavailable: add a "tts" block (primary/backup) to '
                "~/.sharedagents/config.json to enable /audio/speech",
                "service_unavailable",
                501,
            ),
            status_code=501,
        )

    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(
            _openai_error("invalid JSON body", "invalid_request_error", 400),
            status_code=400,
        )
    if not body.get("input") or not body.get("model") or not body.get("voice"):
        return JSONResponse(
            _openai_error(
                "missing required field(s): input, model, voice "
                "(note: OpenRouter requires DATED model slugs like "
                "'openai/gpt-4o-mini-tts-2025-12-15')",
                "invalid_request_error",
                400,
            ),
            status_code=400,
        )

    import httpx

    errors: list[str] = []
    r = None
    for ep in svc.candidates():
        req_body = dict(body)
        if not req_body.get("model"):
            req_body["model"] = ep.model or _TTS_DEFAULT_MODEL
        headers = {"Content-Type": "application/json"}
        if ep.api_key:
            headers["Authorization"] = f"Bearer {ep.api_key}"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
                r = await client.post(
                    _tts_speech_url(ep),
                    content=json.dumps(req_body),
                    headers=headers,
                )
        except Exception as e:  # noqa: BLE001
            errors.append(f"{ep.label}({ep.base_url}): {e}")
            continue
        if r.status_code == 200:
            break
        errors.append(f"{ep.label}({ep.base_url}): HTTP {r.status_code} {r.text[:200]}")
        r = None
    if r is None:
        _LOG.warning("TTS failed on all endpoints: %s", " | ".join(errors))
        return JSONResponse(
            _openai_error(
                "TTS failed — " + " | ".join(errors),
                "internal_server_error",
                502,
            ),
            status_code=502,
        )
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "audio/mpeg"),
    )


# ---------------------------------------------------------------------------
# /responses — GPT-5-style API (stub; not implemented)
# ---------------------------------------------------------------------------


async def _handle_responses(request: Request) -> Response:
    return JSONResponse(
        _openai_error(
            "the /responses (GPT-5-style) API is not implemented; use /chat/completions instead",
            "invalid_request_error",
            "not_found",
        ),
        status_code=404,
    )


# ---------------------------------------------------------------------------
# /slots — llama.cpp probe stub
# ---------------------------------------------------------------------------


async def _handle_slots(request: Request) -> Response:
    # Open WebUI hits this opportunistically against llama.cpp backends;
    # returning [] (HTTP 200) keeps its detection silent.
    return JSONResponse([])


# ---------------------------------------------------------------------------
# scope-context middleware — attaches OpenAIContext to every request
# ---------------------------------------------------------------------------


class _ContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any, *, ctx: OpenAIContext) -> None:
        super().__init__(app)
        self._ctx = ctx

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request.scope["sharedagents_openai_ctx"] = self._ctx
        return await call_next(request)


# ---------------------------------------------------------------------------
# entry point — build the Starlette mini-app
# ---------------------------------------------------------------------------


def make_openai_app(
    *,
    agent_names: list[str],
    manager: Any,
    bearer_token: str | None,
    listed_agent_names: list[str] | None = None,
) -> Starlette:
    """Build the Starlette mini-app to mount under ``/openai/v1``.

    ``manager`` is the gateway's :class:`AgentRegistry` — used to
    acquire the per-agent in-process :class:`AgentACPBackend` for
    chat/completions. ``agent_names`` is every RESOLVABLE agent (the mount
    set), validated against the request's ``model`` field. ``listed_agent_names``
    is the curated PICKER subset rendered by ``/models`` (defaults to
    ``agent_names``); it lets the gateway hide the internal ``main`` fallback from
    the picker while it stays addressable. ``bearer_token`` enforces auth
    (None → open).
    """
    ctx = OpenAIContext(
        agent_names=agent_names,
        manager=manager,
        bearer_token=bearer_token,
        listed_agent_names=listed_agent_names,
    )

    routes = [
        Route("/models", _handle_models, methods=["GET"]),
        # ``:path`` so ``sharedagents/<agent>`` (containing a slash) matches.
        Route("/models/{model_id:path}", _handle_model_get, methods=["GET"]),
        Route("/chat/completions", _handle_chat_completions, methods=["POST"]),
        Route("/embeddings", _handle_embeddings, methods=["POST"]),
        Route("/audio/transcriptions", _handle_audio_transcriptions, methods=["POST"]),
        Route("/audio/speech", _handle_audio_speech, methods=["POST"]),
        Route("/responses", _handle_responses, methods=["POST"]),
        Route("/slots", _handle_slots, methods=["GET"]),
    ]

    middleware = [
        Middleware(_ContextMiddleware, ctx=ctx),
        Middleware(_BearerAuth, expected_token=bearer_token),
    ]

    return Starlette(routes=routes, middleware=middleware)


__all__ = ["make_openai_app", "OpenAIContext"]
