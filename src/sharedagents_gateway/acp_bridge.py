"""ACP backend bridge — adapts sharedagents_core to ``AgentBackend``.

This module satisfies the :class:`sharedagents_acp.server.AgentBackend`
Protocol for local-LLM agents. The multi-round turn loop runs in
:class:`sharedagents_agent.agents.local_llm.LocalLLMEngine`; what
:class:`CoreBackend` owns is:

* **Session lifecycle** — ``new_session`` / ``load_session`` /
  ``list_sessions`` / ``close_session`` / ``inject_user_message``,
  delegating to :class:`Session` + the on-disk JSON store.
* **Mode / config plumbing** — ``set_mode`` / ``set_config_option``
  forward into the live :class:`LocalLLMEngine` so per-session
  state on the engine (``_mode_overrides``, ``_promoted_tools``,
  ``cfg.reasoning_visibility`` / ``cfg.temperature``) takes effect
  on the next round.
* **Cancel routing** — ``cancel(sid)`` resolves the live ``turn_id``
  via ``_new_backend_active_turns`` and calls
  :meth:`Engine.abort(turn_id)`. Two presses inside
  ``_HARD_KILL_WINDOW_S`` escalate to SIGKILL via the engine's
  ``TurnState.stop(force=True)`` path.
* **Outbound channel binding** — ``bind_session_server`` /
  ``unbind_session_server`` / ``_outbound_for`` so
  ``session/request_permission`` and ``session/ask_user`` reach
  the connection that actually owns the session (the gateway
  shares one CoreBackend across many HTTP/WS connections).
* **Prompt** — ``prompt(sid, mid, blocks)`` is a 4-line proxy that
  resolves the session, allocates a turn id, and yields the
  :class:`AgentTurnEvent` → ACP sessionUpdate translation of
  :meth:`Agent.run_turn`. There is no second prompt code path.

Module-level helpers (``_default_run_tool``, ``_default_spawn_worker``,
``_build_worker_request``, ``_build_parallel_groups``,
``_ask_user_outbound``, etc.) are
the reusable building blocks the engine and the AcpServer call into
on the live path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sys
import time as _time
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from sharedagents_agent.agent import (
    AgentConfig,
    find_model_with_capability,
    has_capability,
)
from sharedagents_agent.turn_runtime import (
    TurnState,
    _compose_system_prompt,
    _new_turn_id,
    _now,
)
from sharedagents_core import oscompat
from sharedagents_model.bedrock_config import is_bedrock_row  # noqa: E402
from sharedagents_model.sampling import apply_sampling, configured_sampling
from sharedagents_session.session import Session

from .ask_user import (  # noqa: E402,F401  re-export
    _WEB_ASK_CHANNELS,
    _ask_user_auto_answers,
    _ask_user_outbound,
    _format_ask_answers_for_model,
    _normalize_ask_questions,
    _web_ask_channel_for,
    register_web_ask_channel,
    unregister_web_ask_channel,
)

# --- de-monolith re-exports ---------------------------------------------
#
# The module-level helpers below were extracted into cohesive sibling
# modules (behavior-preserving split of a 3.8k-line file). They are
# re-imported here so ``sharedagents_gateway.acp_bridge`` remains their public
# import path — notably ``sharedagents_agent.agents.local_llm`` imports several
# of them LAZILY from ``sharedagents_gateway.acp_bridge`` on the live turn path,
# and the gateway/control_endpoint/tests import others by that path too.
# Do NOT remove these; they are the stable API surface of this module.
from .outbound import _resolve_outbound_channel  # noqa: E402,F401  re-export
from .round_build import (  # noqa: E402,F401  re-export
    DIRECT_DELEGATE_PREFIX,
    _build_assistant_round_msg,
    _parse_direct_delegate_marker,
    _text_tool_content,
)
from .session_helpers import (  # noqa: E402,F401  re-export
    _loads_json_dict,
    _loads_json_list,
    _turn_has_terminal_event,
    list_sessions_acp,
)
from .tool_execution import (  # noqa: E402,F401  re-export
    _PARALLEL_UNSAFE_TOOLS,
    _SUBAGENT_TOOL_NAMES,
    _build_parallel_groups,
    _call_is_parallel_eligible,
    _default_run_tool,
    _describe_runner_exit,
    _inject_subagent_attribution,
    _state_file_key,
)
from .worker_request import (  # noqa: E402,F401  re-export
    _build_worker_request,
    _default_spawn_worker,
    bedrock_oneshot_request,
    worker_env_for_model,
)

_LOG = __import__("logging").getLogger("sharedagents-agent.bridge")


# --- types ---------------------------------------------------------------

# Async iterator of worker JSONL events. Module-level helpers
# (``_default_spawn_worker``, ``_subprocess_event_stream``) produce
# these; :class:`sharedagents_agent.agents.local_llm.LocalLLMEngine` consumes
# them.
WorkerEventStream = AsyncIterator[Mapping[str, Any]]


# --- helpers -------------------------------------------------------------


def _content_block_to_text(block: Mapping[str, Any]) -> str:
    """Pull the textual payload from an ACP ContentBlock.

    The dispatcher has already rejected unsupported types (this agent
    advertises text + resource_link only). ``resource_link`` blocks
    have no inline text, so we collapse them to their ``uri`` — the
    model can fetch them via tools later.
    """
    t = block.get("type")
    if t == "text":
        return str(block.get("text") or "")
    if t == "resource_link":
        # Best-effort plain-text rendering; the wire boundary already
        # validated structure.
        uri = str(block.get("uri") or "")
        name = str(block.get("name") or uri)
        return f"[resource: {name}]({uri})" if uri else name
    return ""


def _flatten_prompt(blocks: list[Mapping[str, Any]]) -> str:
    """Join ContentBlocks into a single user-message string."""
    return "".join(_content_block_to_text(b) for b in blocks)


def _prompt_source(blocks: list[Mapping[str, Any]]) -> str:
    """Read an internal turn-source marker without changing prompt text.

    The control-plane background recovery path carries this marker as an
    extra ContentBlock field. ACP validation permits extra fields and the
    flatteners ignore them, so it survives a split gateway hop without
    becoming model-visible text.
    """
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        source = str(block.get("_sharedagents_source") or "").strip()
        if source:
            return source
    return "user"


def _blocks_to_user_content(
    blocks: list[Mapping[str, Any]],
) -> Any:
    """Translate ACP ContentBlock[] into the upstream worker's user-content.

    When the prompt is text-only (the common case), returns a single
    string — keeps the worker request small and matches the
    string-content pattern OpenAI-compat upstreams default to.

    When ANY image block is present, returns the OpenAI multi-part
    list shape:
      ``[{"type": "text", "text": "..."},
         {"type": "image_url",
          "image_url": {"url": "data:<mime>;base64,<b64>"}}]``

    vLLM, openrouter, and every OpenAI-compat vision-capable upstream
    accept this shape natively. The vision model decides whether to
    look at the image; non-vision models will ignore the image_url
    parts.
    """
    has_image = any(isinstance(b, Mapping) and b.get("type") == "image" for b in blocks)
    if not has_image:
        return _flatten_prompt(blocks)
    parts: list[dict[str, Any]] = []
    text_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            joined = "".join(text_buf)
            if joined:
                parts.append({"type": "text", "text": joined})
            text_buf.clear()

    for blk in blocks:
        if not isinstance(blk, Mapping):
            continue
        t = blk.get("type")
        if t == "image":
            flush_text()
            data = str(blk.get("data") or "")
            mime = str(blk.get("mimeType") or "image/jpeg")
            if not mime.startswith("image/"):
                mime = "image/jpeg"
            if data:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{data}"},
                    }
                )
        elif t == "text":
            text_buf.append(str(blk.get("text") or ""))
        elif t == "resource_link":
            uri = str(blk.get("uri") or "")
            name = str(blk.get("name") or uri)
            text_buf.append(f"[resource: {name}]({uri})" if uri else name)
    flush_text()
    return parts


# --- worker IO -----------------------------------------------------------


async def _subprocess_event_stream(
    proc: asyncio.subprocess.Process,
) -> WorkerEventStream:
    """Yield JSONL events parsed from ``proc.stdout`` line by line.

    Async generator so the bridge can interleave cancel-checks
    between events. EOF on stdout terminates the generator cleanly;
    the proc is awaited by the caller.
    """
    if proc.stdout is None:
        raise RuntimeError("worker subprocess stdout pipe is None")
    while True:
        line = await proc.stdout.readline()
        if not line:
            return
        try:
            yield json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            # Defensive: skip a corrupt line rather than fail the turn.
            continue


# --- vision-capability bridge --------------------------------------------
#
# Telegram (and the TUI) let a user attach an image to ANY session,
# regardless of which model is selected. The bridge translates that
# image into the OpenAI ``image_url`` multi-part shape — which a
# text-only upstream (e.g. ``deepseek-v4-flash``) rejects outright:
#
#   HTTP 400: unknown variant `image_url`, expected `text`
#
# Worse, the offending message stays in the session transcript, so the
# 400 repeats on every subsequent turn until the session is dropped.
#
# The fix: before spawning the worker, if the selected model is NOT
# vision-capable (its ``cfg.models[]`` row lacks the ``vision``
# capability), route the image through the first configured
# vision-capable model for a one-shot transcription, then splice that
# text back in place of the ``image_url`` part. The selected model
# then sees a faithful text description and the turn runs normally.
# When no vision model is configured (or the transcription fails) the
# image is replaced with a short placeholder so the turn still runs
# instead of crashing.

_IMAGE_TRANSCRIBE_PROMPT = (
    "You are an image-to-text bridge. The image(s) below were sent to a "
    "text-only model that cannot see them. Transcribe everything visible "
    "in faithful, literal detail: all text verbatim, UI layout, charts "
    "and their numbers, code, error messages, diagrams, people and "
    "objects. Do not answer questions about the image or add commentary "
    "— only describe what is present so the text model can reason about it."
)

# Bounded cache of vision transcriptions keyed by (model id + image
# URLs) hash. A session whose transcript already carries an image
# replays it every turn; without this we'd re-bill the vision model
# each round.
_TRANSCRIPTION_CACHE: dict[str, str] = {}
_TRANSCRIPTION_CACHE_MAX = 64


def _message_image_parts(content: Any) -> list[Mapping[str, Any]]:
    """Return the ``image_url`` parts in a chat message's content.

    Empty for the common case — a plain-string content or a multi-part
    list with no images.
    """
    if not isinstance(content, list):
        return []
    return [p for p in content if isinstance(p, Mapping) and p.get("type") == "image_url"]


async def _run_oneshot_worker(
    request: dict[str, Any],
    *,
    timeout_s: float = 90.0,
    env_override: Mapping[str, str] | None = None,
    want_metrics: bool = False,
) -> Any:
    """Spawn a one-shot ``sharedagents_gateway.worker`` and return its content.

    Used for out-of-band model calls (image transcription, compaction
    summaries, loop distillation) that are NOT the user's turn — no
    tools, no streaming to the client, just the assembled ``content``
    text. Raises :class:`RuntimeError` on a worker error, timeout, or
    empty output.

    ``env_override`` (native Bedrock rows) replaces the inherited
    parent environment with the per-row child env built by
    :func:`worker_env_for_model`, so a row's credential mode is isolated
    per subprocess instead of leaking through ``os.environ``.

    Returns ``(text, metrics)`` when ``want_metrics`` is set, else just
    the text (the transcription path's contract).
    """
    package_parent = str(Path(__file__).resolve().parent.parent)
    env = dict(env_override if env_override is not None else os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_parent + (os.pathsep + existing if existing else "")
    env["PYTHONUNBUFFERED"] = "1"
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "sharedagents_gateway.worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        limit=16 * 1024 * 1024,
        # Own session / process group. A force-cancel SIGKILLs the whole
        # process group via ``os.killpg`` (TurnState.stop(force=True)); if
        # this subprocess shared the gateway's group, a hard ESC /
        # ``/abort`` would kill the GATEWAY itself (observed:
        # the gateway died and systemd restarted it). ``start_new_session``
        # makes it a session leader so killpg reaches only it and its
        # descendants (the bash/sleep/CLI it spawns).
        start_new_session=not oscompat.IS_WINDOWS,
        **oscompat.detach_kwargs(),
    )
    if proc.stdin is None:
        raise RuntimeError("worker stdin pipe is None")
    proc.stdin.write(json.dumps(request).encode("utf-8"))
    await proc.stdin.drain()
    proc.stdin.close()

    content: list[str] = []
    err: str | None = None
    metrics: dict[str, Any] | None = None

    async def _drain() -> None:
        nonlocal err, metrics
        # The worker exits after emitting ``done``/``aborted``, so the
        # event stream ends at stdout EOF — no explicit break needed.
        async for ev in _subprocess_event_stream(proc):
            kind = ev.get("kind")
            if kind == "content":
                content.append(str(ev.get("text") or ""))
            elif kind == "metrics":
                metrics = dict(ev)
            elif kind == "error":
                err = str(ev.get("message") or "worker error")

    try:
        await asyncio.wait_for(_drain(), timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        raise RuntimeError(f"transcription worker timed out after {timeout_s:.0f}s") from None
    finally:
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001
            pass

    if err:
        raise RuntimeError(err)
    text = "".join(content).strip()
    if not text:
        raise RuntimeError("vision model returned an empty transcription")
    if want_metrics:
        return text, metrics
    return text


async def _transcribe_image_parts(
    cfg: AgentConfig,
    vision_row: Mapping[str, Any],
    image_parts: Sequence[Mapping[str, Any]],
    caption: str,
) -> str:
    """Transcribe ``image_parts`` via the configured vision model.

    Results are cached on the (model id + image URL) hash so a
    replayed transcript image is only billed once per process.
    """
    endpoint = str(vision_row.get("baseUrl") or "").rstrip("/")
    model_id = str(vision_row.get("id") or "").strip()
    if not endpoint or not model_id:
        raise RuntimeError(f"vision model row missing baseUrl/id: {dict(vision_row)!r}")

    urls = "\0".join(str((p.get("image_url") or {}).get("url") or "") for p in image_parts)
    key = hashlib.sha256(f"{model_id}\0{urls}".encode()).hexdigest()
    cached = _TRANSCRIPTION_CACHE.get(key)
    if cached is not None:
        return cached

    prompt = _IMAGE_TRANSCRIBE_PROMPT
    if caption:
        prompt += f"\n\nThe user's accompanying message: {caption}"
    user_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    user_parts.extend(dict(p) for p in image_parts)

    resolved_sampling = configured_sampling(cfg, vision_row, model_id, purpose="vision")
    oneshot_messages = [{"role": "user", "content": user_parts}]
    if is_bedrock_row(vision_row):
        # Native Bedrock vision row: the Converse adapter converts the data
        # URL into a native image block, and the per-row child env carries
        # the credential (bearer token only for an explicit API-key row,
        # ambient bearer removed for an IAM row). ``bedrock://`` is never an
        # HTTP endpoint — no headers, no /chat/completions.
        request = bedrock_oneshot_request(
            vision_row,
            oneshot_messages,
            sampling=resolved_sampling,
        )
        env = worker_env_for_model(cfg, vision_row, os.environ)
    else:
        try:
            headers = cfg.upstream_headers(endpoint)
        except Exception:  # noqa: BLE001
            headers = {}

        request = {
            "endpoint": endpoint,
            "model": model_id,
            "messages": oneshot_messages,
            "enable_thinking": False,
        }
        apply_sampling(request, resolved_sampling)
        if headers:
            request["headers"] = dict(headers)
        backend_kind = str(vision_row.get("backend") or "")
        if backend_kind:
            request["backend"] = backend_kind
        env = None

    text = await _run_oneshot_worker(request, env_override=env)

    if len(_TRANSCRIPTION_CACHE) >= _TRANSCRIPTION_CACHE_MAX:
        _TRANSCRIPTION_CACHE.pop(next(iter(_TRANSCRIPTION_CACHE)), None)
    _TRANSCRIPTION_CACHE[key] = text
    _LOG.info(
        "transcribed %d image part(s) via %s (%d chars)",
        len(image_parts),
        vision_row.get("alias") or model_id,
        len(text),
    )
    return text


async def _resolve_images_for_model(
    backend: CoreBackend,
    sess: Session,
    messages: list[Mapping[str, Any]],
    *,
    endpoint_override: str | None = None,
    model_override: str | None = None,
) -> list[dict[str, Any]]:
    """Make ``image_url`` content safe for the session's selected model.

    * Selected model is vision-capable → images pass through untouched
      (the upstream reads them natively).
    * Selected model is text-only → every ``image_url`` part is
      replaced with a transcription from the first vision-capable model
      in ``cfg.models[]``.
    * No vision model configured, or the transcription fails → the
      image is replaced with a short placeholder.

    ``endpoint_override`` / ``model_override`` (when given) resolve the
    "selected model" from the engine's per-round LLM-fallback failover
    instead of the durable session fields.

    Never raises: a failure here must not be worse than the text-only
    400 it exists to prevent.
    """
    msgs: list[dict[str, Any]] = [dict(m) if isinstance(m, Mapping) else m for m in messages]
    img_indices = [i for i, m in enumerate(msgs) if isinstance(m, Mapping) and _message_image_parts(m.get("content"))]
    if not img_indices:
        return msgs

    eff_endpoint = str(endpoint_override or sess.endpoint or "")
    eff_model = str(model_override or getattr(sess, "model", "") or "")
    cfg = backend.cfg
    try:
        current_row = cfg.model_row_for_endpoint(
            eff_endpoint,
            eff_model,
        )
    except Exception:  # noqa: BLE001
        current_row = {}
    if has_capability(current_row, "vision"):
        # Selected model reads images natively — leave them alone.
        return msgs

    try:
        vision_row = find_model_with_capability(cfg, "vision")
    except Exception:  # noqa: BLE001
        vision_row = None

    for i in img_indices:
        m = msgs[i]
        content = m.get("content")
        image_parts = _message_image_parts(content)
        # Every text part on this message — the user's caption plus any
        # ``[Turn N]`` marker the message builder prepended.
        caption = "".join(
            str(p.get("text") or "") for p in content if isinstance(p, Mapping) and p.get("type") == "text"
        ).strip()

        if vision_row is None:
            _LOG.warning(
                "prompt carries %d image part(s) but no vision-capable "
                "model is configured (selected model=%s); replacing with "
                "a placeholder so the turn still runs",
                len(image_parts),
                eff_model,
            )
            inserted_text = (
                f"\n\n[{len(image_parts)} image(s) attached — no "
                "vision-capable model is configured to read them. Add a "
                'model with "capabilities": ["vision"] to '
                "~/.sharedagents/config.json, or ask the user to describe it.]"
            )
        else:
            label = str(vision_row.get("alias") or vision_row.get("id") or "vision model")
            try:
                transcription = await _transcribe_image_parts(
                    cfg,
                    vision_row,
                    image_parts,
                    caption,
                )
                inserted_text = (
                    f"\n\n[Transcription of {len(image_parts)} attached "
                    f"image(s), read by {label} — the selected model is "
                    f"text-only:]\n{transcription}"
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "vision transcription via %s failed: %s; replacing image with placeholder",
                    label,
                    exc,
                )
                inserted_text = (
                    f"\n\n[{len(image_parts)} image(s) attached but could not be transcribed by {label}: {exc}]"
                )

        # Rebuild content: keep text parts in order, splice the
        # transcription where the first image was, drop the image_url
        # parts (the selected model can't read them).
        new_parts: list[Any] = []
        inserted = False
        for p in content:
            if isinstance(p, Mapping) and p.get("type") == "image_url":
                if not inserted:
                    new_parts.append({"type": "text", "text": inserted_text})
                    inserted = True
                continue
            new_parts.append(p)
        if not inserted:
            new_parts.append({"type": "text", "text": inserted_text})

        # Collapse to a plain string when nothing but text remains —
        # matches the string-content convention text-only upstreams want.
        if all(isinstance(p, Mapping) and p.get("type") == "text" for p in new_parts):
            m["content"] = "".join(str(p.get("text") or "") for p in new_parts)
        else:
            m["content"] = new_parts
        msgs[i] = m

    return msgs


# --- bridge --------------------------------------------------------------


# Two-press hard-kill window (fix D). When the second ``cancel(sid)``
# arrives within this many seconds of the first, escalate the active
# subprocesses' termination from ``SIGTERM`` to ``SIGKILL`` + ``killpg``
# on the process group. The default ``SIGTERM`` flow is enough for any
# cooperative runner — its asyncio signal handler cancels the dispatch
# task which kills child process groups itself — but a wedged python
# interpreter that swallowed SIGTERM (or a C extension not yielding
# the GIL) wouldn't see it, leaving the operator with no second-line
# kill. One second matches the "natural retry latency" of a frustrated
# operator mashing ESC; tune via ``SHAREDAGENTS_HARD_KILL_WINDOW_S``.
try:
    _HARD_KILL_WINDOW_S: float = float(os.environ.get("SHAREDAGENTS_HARD_KILL_WINDOW_S", "1.0") or 1.0)
except (TypeError, ValueError):
    _HARD_KILL_WINDOW_S = 1.0

# How long ``session/new`` / ``session/load`` will wait for ACP-supplied
# MCP discovery before returning (discovery keeps warming in the
# background past this). Bounds session-setup latency so a slow/hung
# client-supplied MCP server can't stall the response; small enough that
# a stock client won't time the session out, large enough that fast local
# stdio servers are advertised on turn 1.
try:
    _ACP_MCP_TURN1_BUDGET_S: float = float(os.environ.get("SHAREDAGENTS_ACP_MCP_TURN1_BUDGET_S", "3.0") or 3.0)
except (TypeError, ValueError):
    _ACP_MCP_TURN1_BUDGET_S = 3.0


class CoreBackend:
    """Adapter from sharedagents_core's session/turn primitives to
    :class:`sharedagents_acp.server.AgentBackend`.

    Owns session lifecycle (new_session / load_session / list_sessions /
    close_session / inject_user_message / set_mode / set_config_option),
    cancel routing into the live :class:`LocalLLMEngine` via
    :meth:`Engine.abort(turn_id)`, and outbound channel binding
    (``bind_session_server`` / ``_outbound_for``) so permission +
    ask_user requests reach the connection that owns the session.

    :meth:`prompt` is a 4-line proxy that forwards through
    :meth:`Agent.run_turn` and translates the AgentTurnEvent stream
    to ACP sessionUpdate dicts via :mod:`.acp_translation`. The
    multi-round turn loop / compaction / loop detection / tool
    dispatch / hook lifecycle all live in
    :class:`sharedagents_agent.agents.local_llm.LocalLLMEngine`.

    State (keyed by ACP session id):

    * ``_active_sessions``: the in-memory :class:`Session` for sessions
      that have been created or loaded this run. Persistence is to
      ``cfg.sessions_dir/<hash>.json`` via :meth:`Session.save`.
    * ``_new_backend_active_turns``: ``sid → turn_id`` for the live
      in-flight turn so :meth:`cancel` can route through
      :meth:`Engine.abort(turn_id)`.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        server_handle: Any | None = None,
        agent: Any | None = None,
    ) -> None:
        self.cfg = config
        # ``agent`` is the pre-built :class:`Agent` (with its
        # LocalLLMEngine) constructed by :class:`AgentACPBackend`.
        # When provided, ``_get_new_agent`` returns it instead of
        # building a duplicate pair — single source of truth for the
        # session lock / promoted-tools / mode-overrides state the
        # engine owns. Falls back to lazy construction for direct
        # callers (tests, stdio entry point) that don't have an
        # Agent on hand.
        self._preset_agent = agent
        # Optional handle to the AcpServer hosting this backend.
        # ``server_handle`` is the SINGLE-connection fallback (stdio
        # path, tests). Under the gateway, ONE CoreBackend is shared
        # by every HTTP/WS connection to ``/acp/<agent>`` — a single
        # handle would race between e.g. a TUI and a Telegram client.
        # ``_session_servers`` is the per-session override: each
        # AcpServer binds itself here for the sessions opened on its
        # connection (see ``bind_session_server``), and
        # ``_outbound_for`` resolves the right channel per session.
        # Permission / ask_user requests therefore always reach the
        # connection that actually owns the session.
        self.server_handle = server_handle
        self._session_servers: dict[str, Any] = {}
        self._active_sessions: dict[str, Session] = {}
        # The live Agent + LocalLLMEngine; built on first use (or
        # passed in via ``agent=``). All turn-loop state lives on
        # the engine — promoted_tools, mode_overrides, cancelled,
        # pending_user_messages, active_turns, session_to_turn.
        self._new_agent: Any | None = None
        # sid → turn_id for the live in-flight turn so :meth:`cancel`
        # can route through ``engine.abort(turn_id)``. Populated by
        # the ``on_turn_started`` callback in ``agent_events_to_acp``.
        self._new_backend_active_turns: dict[str, str] = {}
        # sid → count of prompts currently in-flight OR queued behind the
        # running turn. Concurrent same-session prompts already FIFO-
        # serialize on the engine's per-session lock (agent.py
        # ``engine.session_lock``); this bounds how many can pile up so a
        # runaway client (or a buggy loop firing session/prompt) can't
        # queue unboundedly. Past the cap, ``prompt`` terminates
        # immediately with ``stopReason: refusal``.
        self._prompt_depth: dict[str, int] = {}
        # sid → background ACP-MCP discovery task (warms the per-session
        # tool cache off the session-setup path). Cancelled on
        # close_session so a dropped session doesn't leave it running.
        self._acp_discovery_tasks: dict[str, Any] = {}
        # sid → monotonic timestamp of the most recent :meth:`cancel`
        # call (fix D). A second cancel arriving within
        # ``_HARD_KILL_WINDOW_S`` of the first escalates ``TurnState.stop``
        # to SIGKILL + ``killpg`` so a wedged python tool runner (one
        # that swallowed the first SIGTERM) doesn't outlive the operator
        # double-tapping ESC. Cleared lazily — entries linger until the
        # next cancel rebinds them, which is fine since the timestamp
        # comparison is per-session and only an explicit cancel reads it.
        self._last_cancel_at: dict[str, float] = {}

    # -- outbound-channel routing -----------------------------------------

    def bind_session_server(self, session_id: str, server: Any) -> None:
        """Register ``server`` as the outbound ACP channel for ``session_id``.

        Called by :class:`AcpServer` on ``session/new`` / ``session/load``.
        Lets a backend shared across many connections (the gateway
        mounts one CoreBackend per agent) route ``session/request_permission``
        and ``session/ask_user`` to the connection that owns the
        session rather than to whichever connection happened to attach
        last. See ``_outbound_for``.
        """
        if session_id:
            self._session_servers[session_id] = server

    def unbind_session_server(self, session_id: str) -> None:
        """Drop the per-session outbound binding for ``session_id``.

        Called by :class:`AcpServer` on ``session/close`` and on
        connection teardown. Idempotent — unbinding an unknown session
        is a no-op.
        """
        self._session_servers.pop(session_id, None)

    def _outbound_for(self, session_id: str) -> Any | None:
        """Resolve the outbound ACP channel for ``session_id``.

        Prefers the per-session binding written by
        :meth:`bind_session_server` (so a backend shared across many
        connections routes ``session/request_permission`` and
        ``session/ask_user`` to the connection that owns the session,
        not to whichever connection happened to attach last). Falls
        back to the bare ``server_handle`` for single-connection
        callers / stub backends in tests. ``None`` when neither is
        bound.
        """
        srv = self._session_servers.get(session_id)
        if srv is not None:
            return srv
        return getattr(self, "server_handle", None)

    # -- internal helpers --------------------------------------------------

    def _session_for(self, session_id: str) -> Session | None:
        sess = self._active_sessions.get(session_id)
        if sess is not None:
            return sess
        # Try to lazy-load from sessions.db so a fresh backend
        # after a restart can answer ``session/load`` for a hash it
        # didn't create. Header from sessions_meta, turns folded from
        # the JSONL event log.
        from sharedagents_session.session_index import SessionIndex, connect_sessions_db
        from sharedagents_session.session_reader import (
            fold_jsonl_into_turns,
            session_log_path,
        )

        db = self.cfg.sessions_dir / "sessions.db"
        jsonl = self.cfg.sessions_dir / f"{session_id}.jsonl"
        log = session_log_path(self.cfg.sessions_dir, session_id)
        if log is None and not jsonl.exists() and not db.exists():
            return None
        row = None
        try:
            if db.exists() or jsonl.exists():
                SessionIndex(self.cfg.sessions_dir).tail(session_id)
                con = connect_sessions_db(db)
                try:
                    row = con.execute(
                        "SELECT * FROM sessions_meta WHERE session_hash=?",
                        (session_id,),
                    ).fetchone()
                finally:
                    con.close()
        except Exception:  # noqa: BLE001
            row = None
        if row is None:
            return None
        d = dict(row)
        # Scope to this backend's agent: under the unified sessions.db
        # every agent shares the table, so a known sid would otherwise
        # let agent A attach / resume agent B's session via ACP
        # ``load_session`` / prompt. Empty agent_name is treated as a
        # match so legacy / unattributed sessions stay accessible to
        # the configured agent that finds them (preserves the pre-
        # unified edge).
        sess_agent = str(d.get("agent_name") or "")
        own_agent = str(getattr(self.cfg, "name", "") or "")
        if sess_agent and own_agent and sess_agent != own_agent:
            from sharedagents_session.session_state import active_agent

            if (
                active_agent(
                    self.cfg.sessions_dir,
                    session_id,
                    default=sess_agent,
                )
                != own_agent
            ):
                return None
            try:
                heal = connect_sessions_db(db)
                try:
                    heal.execute(
                        "UPDATE sessions_meta SET agent_name=? WHERE session_hash=?",
                        (own_agent, session_id),
                    )
                    heal.commit()
                finally:
                    heal.close()
            except Exception:  # noqa: BLE001 — heal is best-effort
                pass
            d["agent_name"] = own_agent
        from sharedagents_session.session import build_new_chat_title

        sess = Session(
            hash=d.get("session_hash") or session_id,
            title=d.get("title") or build_new_chat_title(d.get("created_at")),
            created_at=d.get("created_at") or _now(),
            system_prompt=d.get("system_prompt") or "",
            model=d.get("model") or self.cfg.upstream_model,
            endpoint=d.get("endpoint") or self.cfg.upstream_endpoint,
            turns=[],
            system_mode=d.get("system_mode") or "",
            workspace_cwd=d.get("workspace_cwd") or "",
            workspace_origin=d.get("workspace_origin") or "",
            backend=d.get("backend") or "",
            context_window=int(d.get("context_window") or 0),
            compaction_count=int(d.get("compaction_count") or 0),
            memory_flush_compaction_count=int(d.get("memory_flush_compaction_count") or -1),
            last_memory_flush_at=d.get("last_memory_flush_at") or "",
            last_prompt_tokens=int(d.get("last_prompt_tokens") or 0),
            pinned_facts=_loads_json_list(d.get("pinned_facts")),
            compaction_summary=d.get("compaction_summary") or "",
            compaction_last_stats=_loads_json_dict(d.get("compaction_last_stats")),
            session_summary=d.get("session_summary") or "",
            reasoning_visibility=d.get("reasoning_visibility") or "stream",
            tool_output_visible=bool(d.get("tool_output_visible") or 0),
            title_refinements=int(d.get("title_refinements") or 0),
            title_locked=bool(d.get("title_locked") or False),
        )
        sess.teacher_override = d.get("teacher_override") or ""
        sess.teacher_model_override = d.get("teacher_model_override") or ""
        # The session's effort/speed overrides: without them the next header
        # save (set_mode, set_config_option, a turn) wrote them back as "".
        sess.effort = d.get("effort") or ""
        sess.speed = d.get("speed") or ""
        # TIER 2c — ONE shared hydration path with the Agent loader: restores
        # _storage_root + goal_id/goal_turn_count/goal_terminal/cost fields
        # and reconciles disagreements from the durable goal registry.
        # Previously ONLY active_goal was restored here — the
        # split-brain that let a ``goal`` tool action see an empty projection
        # and reset the durable turn counter / freeze the goal cost ledger.
        try:
            from sharedagents_session.session_hydration import (
                hydrate_session_goal_fields,
            )

            hydrate_session_goal_fields(
                sess,
                d,
                agent_name=own_agent,
                sessions_dir=self.cfg.sessions_dir,
            )
        except Exception:  # noqa: BLE001 — hydration must never break a load
            pass
        try:
            sess.turns = fold_jsonl_into_turns(self.cfg.sessions_dir / f"{session_id}.jsonl")
            # OpenCode ``compaction.prune`` parity: OFF by default. Pruning on
            # resume rewrites earlier prompt messages and costs the upstream
            # prefix cache; tool results are pruned at compaction instead.
            # Opt in with ``compact_prune_on_load``. Best-effort.
            try:
                from sharedagents_agent.compaction import prune_session_on_load

                prune_session_on_load(
                    sess,
                    enabled=bool(getattr(self.cfg, "compact_prune_on_load", False)),
                )
            except Exception:  # noqa: BLE001 — resume must not fail on a prune error
                pass
        except Exception:  # noqa: BLE001
            sess.turns = []
        self._active_sessions[session_id] = sess
        return sess

    def is_session_loaded(self, session_id: str) -> bool:
        """Whether this backend already owns an attached in-memory session.

        This deliberately does not call ``_session_for``: that helper may
        reconstruct a durable session after restart; callers use this method
        only when they need proof that session/load ran in this process.
        """
        return session_id in self._active_sessions

    # -- AgentBackend Protocol --------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Mapping[str, Any] | None,
        client_info: Mapping[str, Any] | None,
    ) -> Mapping[str, Any] | None:
        """Record the negotiated peer info — the dispatcher's static
        ``initialize`` response is built from advertised capabilities,
        so the return value here is informational only."""
        self._peer_protocol_version = protocol_version
        self._peer_client_capabilities = client_capabilities
        self._peer_client_info = client_info
        return None

    async def authenticate(self, method_id: str) -> None:
        """No auth methods advertised today — the dispatcher refuses
        ``authenticate`` with METHOD_NOT_FOUND before reaching this
        hook. Implemented as a no-op for Protocol completeness."""
        return None

    async def _register_acp_mcp(
        self,
        session_id: str,
        mcp_servers: list[Mapping[str, Any]],
        cwd: str = "",
    ) -> None:
        """Connect the ACP client's per-session MCP servers (spec: the
        agent connects to ``mcpServers`` on session init).

        The dispatcher already rejected http/sse entries unless the
        matching capability is advertised; stdio entries are accepted.
        We parse each into an :class:`MCPServerConfig`, stamp the session
        workspace ``cwd`` onto stdio servers that didn't set their own,
        and register them under ``session_id`` so the per-turn tool
        registry (``acp_session_mcp_tools_sync``) and the runner env
        channel see them.

        Discovery (which SPAWNS each server to list its tools, needed
        only to ADVERTISE schemas to the model) runs in the BACKGROUND:
        ``session/new`` / ``session/load`` are dispatched synchronously
        on the read loop, so awaiting a slow/hung client-supplied MCP
        server here would stall session creation and make a stock client
        (Zed/acpx) declare the agent dead. We wait a short, NON-cancelling
        budget so fast local servers are advertised on turn 1; a slow
        server keeps warming in the background and its tools appear on a
        later turn (the sync registry reader tolerates a partial cache).
        Best-effort throughout: a bad entry or a server that won't start
        is logged and skipped, never fatal."""
        if not session_id or not mcp_servers:
            return
        import asyncio as _asyncio
        from dataclasses import replace as _dc_replace

        from sharedagents_mcp import mcp_client as _mcp

        parsed: list[_mcp.MCPServerConfig] = []
        for entry in mcp_servers:
            try:
                cfg = _mcp.MCPServerConfig.from_acp(dict(entry))
                # stdio servers inherit the session workspace cwd unless
                # they set their own — so a client-supplied filesystem /
                # relative-path server operates against the session's
                # workspace, not the gateway process cwd.
                if cwd and cfg.transport == "stdio" and not cfg.cwd:
                    cfg = _dc_replace(cfg, cwd=cwd)
                parsed.append(cfg)
            except (ValueError, TypeError) as e:
                self._log_event("ACP_MCP_BAD_ENTRY", error=str(e)[:200])
        if not parsed:
            return
        # Register synchronously so the INVOKE path (env channel) sees
        # the servers immediately, then warm discovery off the hot path.
        _mcp.register_acp_session_servers(session_id, parsed)

        async def _warm() -> None:
            try:
                tools = await _mcp.discover_acp_session_tools(session_id)
                self._log_event(
                    "ACP_MCP_CONNECTED",
                    sid=session_id[:12],
                    servers=len(parsed),
                    tools=len(tools),
                )
            except Exception as e:  # noqa: BLE001
                self._log_event("ACP_MCP_DISCOVERY_FAILED", error=str(e)[:200])

        # Cancel any discovery still warming from a prior register on this
        # sid (e.g. session/new then session/load on reattach) so we don't
        # orphan a redundant probe that could clobber the cache late.
        _prev = self._acp_discovery_tasks.get(session_id)
        if _prev is not None and not _prev.done():
            _prev.cancel()
        task = _asyncio.ensure_future(_warm())
        self._acp_discovery_tasks[session_id] = task

        def _done(t: Any, sid: str = session_id) -> None:
            if self._acp_discovery_tasks.get(sid) is t:
                self._acp_discovery_tasks.pop(sid, None)

        task.add_done_callback(_done)
        # Bounded, NON-cancelling wait: asyncio.wait returns on timeout
        # WITHOUT cancelling the task, so a slow server keeps warming in
        # the background while session setup proceeds.
        try:
            await _asyncio.wait({task}, timeout=_ACP_MCP_TURN1_BUDGET_S)
        except Exception:  # noqa: BLE001
            pass

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        # The dispatcher already rejected http/sse mcp entries against
        # our advertised capabilities; stdio entries (no ``type`` field
        # or ``type=="stdio"``) are accepted unconditionally.
        # Require an explicit upstream model id. The previous
        # ``/v1/models`` auto-discovery was a footgun: the picked id
        # depended on whichever model the upstream happened to expose
        # first at session-start time, so two sessions launched minutes
        # apart could silently bind to different models — and the id
        # is baked into ``Session`` at creation, never re-resolved per
        # turn, so the operator only noticed when they got off-key
        # answers from the wrong checkpoint. Force the operator to
        # name the model explicitly via ``cfg.upstream_model`` (or via
        # the active row in ``cfg.models[]`` for alias switches).
        endpoint = self.cfg.upstream_endpoint
        model = (self.cfg.upstream_model or "").strip()
        # Refuse endpoint-INDEPENDENTLY: a local-LLM session
        # must never start without an explicit ``upstream_model``, even when the
        # endpoint is also blank. A blank endpoint used to skip this guard,
        # letting a session start unbound and later resolve to whatever model
        # the default upstream happened to expose — the exact silent divergence
        # this check prevents.
        if not model:
            raise RuntimeError(
                f"upstream {endpoint!r} has no `upstream_model` configured. "
                f"Set an explicit model id in ~/.sharedagents/config.json — "
                f"`/v1/models` auto-discovery has been removed."
            )
        # Compose the full system prompt: cfg.system + SOUL/USER/TOOLS/
        # AGENTS context files + skill index + MCP/ACP advertisement.
        # Build once at session creation and persist on the Session
        # record so every subsequent turn replays a byte-identical
        # prefix — vLLM APC / SGLang RadixAttention only reuses KV
        # cache if the prefix matches block-aligned bytes.
        # Re-composition would invalidate the cache mid-session.
        active_mode = self.cfg.permission_mode or "auto"
        try:
            composed_system = _compose_system_prompt(
                self.cfg,
                active_mode,
                workspace_cwd=cwd,
                workspace_origin="acp",
            )
        except Exception:  # noqa: BLE001
            # Best-effort: a missing/unreadable workspace file must not
            # block session creation. Fall back to the bare cfg.system
            # so the session is still usable.
            composed_system = self.cfg.system or ""
        sess = Session.new(
            system_prompt=composed_system,
            system_mode=active_mode,
            model=model,
            endpoint=endpoint,
            sessions_dir=self.cfg.sessions_dir,
            workspace_cwd=cwd,
            workspace_origin="acp",
            # Bind the session's tokenizer/protocol family + context
            # window from config so the ``sessions_meta`` row records a
            # real backend instead of a blank (matches what
            # ``Agent.create_session`` does for the local-LLM engine).
            backend=getattr(self.cfg, "backend", "") or "",
            context_window=int(getattr(self.cfg, "upstream_max_context", 0) or 0),
        )
        # Stamp attribution (agent_name / engine / system_prompt_sha256)
        # AND emit the initial ``session_started`` event — the SAME
        # canonical sink ``Agent.create_session`` uses. The previous
        # ``sess.save(force=True)`` here wrote the ``sessions_meta`` row
        # with empty ``agent_name`` + ``backend`` because the bridge
        # never set them, and ``Session.save``'s ``header_update`` path
        # excludes ``agent_name`` so the blank could never self-heal.
        # Result: every local-LLM session (``orig='acp'`` — the live
        # TUI / gateway path) showed a blank agent + backend in
        # ``session/list`` and the TUI session picker (operator report
        # 2026-05-20). ``record_session_creation`` inserts the row too,
        # so the restart / first-prompt sid resolution the old
        # ``force=True`` save guaranteed still holds.
        engine_kind = getattr(self.cfg, "engine", "") or "local_llm"
        try:
            from sharedagents_agent.agents.session_log import record_session_creation

            record_session_creation(
                sess,
                agent_name=getattr(self.cfg, "name", "") or "",
                engine=engine_kind,
                system_prompt=composed_system,
                sessions_dir=self.cfg.sessions_dir,
            )
        except Exception as exc:  # noqa: BLE001
            # Best-effort fallback to the legacy force-save so the row
            # still exists for ``session/list`` + ``Agent.get_session``
            # (sid resolution on the first prompt) even if attribution
            # stamping somehow fails. Loud in logs: the common refusal is
            # an unknown agent_name (record_session_creation validates
            # the identity) — the fallback row then carries a BLANK
            # agent_name, never a phantom one.
            logging.getLogger("sharedagents.acp_bridge").warning(
                "new_session: record_session_creation failed (agent_name=%r): %s — falling back to unattributed save",
                getattr(self.cfg, "name", "") or "",
                exc,
            )
            try:
                sess.save(self.cfg.sessions_dir, force=True)
            except Exception:  # noqa: BLE001
                pass
        # A new session starts with the effort remembered for its model.
        try:
            from sharedagents_agent.agents.session_lifecycle import initial_session_effort

            remembered = initial_session_effort(
                self.cfg,
                getattr(self.cfg, "name", "") or "",
                model,
                None,
                is_external=False,
            )
            if remembered:
                sess.effort = remembered
                sess.save_header_fields(["effort"], root=self.cfg.sessions_dir)
        except Exception as exc:  # noqa: BLE001 — a preference never blocks a session
            logging.getLogger("sharedagents.acp_bridge").warning(
                "new_session: remembered effort not applied (%s)",
                type(exc).__name__,
            )
        self._active_sessions[sess.hash] = sess
        await self._register_acp_mcp(sess.hash, list(mcp_servers or []), cwd=cwd)
        return {
            "sessionId": sess.hash,
            "modes": self._modes_state(active_mode),
            # ``models`` is a non-spec extension we kept for legacy
            # frontends; rename to a non-conflicting key prefixed with
            # ``_`` so spec-strict ACP clients don't see it as a typed
            # field. Frontends that read it from sharedagents still get the
            # data via ``configOptions``.
            "configOptions": self._config_options_state(),
            "_meta": {"sharedagents": {"siblings": self._sibling_registry()}},
        }

    def _modes_state(self, active_mode: str) -> dict[str, Any]:
        """Build a spec-shaped ``SessionModeState`` payload.

        Spec ref: ``schema.json:SessionModeState`` —
        ``{availableModes: SessionMode[], currentModeId: string}``.
        Without this, ACP clients can't drive ``session/set_mode``
        because they have no list of valid mode ids to choose from.
        """
        return {
            "availableModes": [
                {
                    "id": "auto",
                    "name": "auto",
                    "description": "Auto-approve safe tools; prompt on destructive actions.",
                },
                {"id": "ask", "name": "ask", "description": "Prompt for confirmation on every tool call."},
                {"id": "plan", "name": "plan", "description": "Plan-only mode — propose tool calls without executing."},
            ],
            "currentModeId": active_mode,
        }

    def _sibling_registry(self) -> list[dict[str, Any]]:
        """Names + capability hints of every sharedagents-native sibling
        agent the gateway hosts. Frontends consume this to auto-wire
        ``/<name> <task>`` slash commands without hardcoding.

        Excludes the calling agent's OWN name (you can't delegate to
        yourself) and excludes external-LLM proxies (claude / codex /
        gemini) since those have bespoke per-vendor handlers, not the
        uniform ``delegate_agent(name, task)`` interface.

        Reads from ``load_global_config()`` directly because
        ``acp_agents`` isn't a dataclass field on ``AgentConfig`` (the
        merged-config path drops global-only keys that weren't in the
        per-agent allowlist). Same source the ``delegate_agent`` runner
        uses, so the registry the frontend sees can never disagree with
        what the runner can dispatch.
        """
        from sharedagents_agent.agent import load_global_config

        glob = load_global_config()
        agents = glob.get("acp_agents") or {}
        my_name = getattr(self.cfg, "name", "") or ""
        out: list[dict[str, Any]] = []
        for name, row in agents.items():
            if not isinstance(row, Mapping):
                continue
            if name == my_name:
                continue
            out.append(
                {
                    "name": name,
                    "capabilities": list(row.get("capabilities") or []),
                    "timeout_s": row.get("timeout_s"),
                }
            )
        return out

    def _config_options_state(self) -> list[dict[str, Any]]:
        """Spec-shaped ``SessionConfigOption`` list for the wire.

        The ACP SDK's ``SessionConfigOptionSelect`` schema requires
        ``{id, name, type:"select", currentValue, options:[{id,
        label}]}`` — building a payload that doesn't validate breaks
        every ``session/new`` response (pydantic rejects, the SDK
        raises mid-prompt and the bot surfaces "agent error").
        Today sharedagents does NOT expose ``model`` as a wire-level config
        option (we use the dedicated ``session/set_config_option``
        path with a free-form alias). Until the schema-shaped option
        list is wired, keep the field empty so the response validates.
        """
        return []

    async def load_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        sess = self._session_for(session_id)
        if sess is None:
            raise RuntimeError(f"session not found: {session_id!r}")
        # Refresh workspace context — a frontend reattaching from a
        # different cwd should see its updated workspace string in the
        # next composed system prompt. Persist immediately so a
        # gateway restart between reattach and the first prompt
        # doesn't resurrect the stale on-disk cwd.
        if cwd:
            sess.workspace_cwd = cwd
            try:
                sess.save(self.cfg.sessions_dir)
            except Exception:  # noqa: BLE001
                pass
        # ACP clients re-send mcpServers on reattach; (re)connect them.
        await self._register_acp_mcp(session_id, list(mcp_servers or []), cwd=cwd)
        return {
            "modes": self._modes_state(
                sess.system_mode or self.cfg.permission_mode or "auto",
            ),
            "configOptions": self._config_options_state(),
            "_meta": {"sharedagents": {"siblings": self._sibling_registry()}},
        }

    async def list_sessions(
        self,
        cursor: str | None,
        cwd: str | None,
    ) -> Mapping[str, Any]:
        return list_sessions_acp(self.cfg.sessions_dir, self.cfg.home)

    async def close_session(self, session_id: str) -> None:
        # Hard-cancel any in-flight turn first so we don't leak a worker
        # or a runner/subagent subprocess — the session is going away,
        # so there's nothing to gracefully wind down.
        await self.cancel(session_id, force=True)
        # Cancel any background ACP-MCP discovery still warming for this
        # session, then drop its server overlay + discovered-tool cache
        # so a long-lived gateway process doesn't accumulate them.
        task = self._acp_discovery_tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
        try:
            from sharedagents_mcp import mcp_client as _mcp

            _mcp.unregister_acp_session(session_id)
        except Exception:  # noqa: BLE001
            pass
        sess = self._active_sessions.pop(session_id, None)
        if sess is not None:
            try:
                sess.save(self.cfg.sessions_dir)
            except Exception:  # noqa: BLE001
                pass

    async def steer_now(self, session_id: str) -> Mapping[str, Any]:
        """Ask this backend's live turn on ``session_id`` to take the queue now.

        Routed like cancel: ``session_id → turn_id`` from the active-turn map,
        then :meth:`Engine.steer_now`. ``steered`` is ``False`` when this
        backend runs no turn for the session or its engine cannot steer.
        """
        turn_id = self._new_backend_active_turns.get(session_id)
        agent = self._new_agent
        if not turn_id or agent is None:
            return {"ok": True, "steered": False, "reason": "no_live_turn"}
        steer = getattr(agent.engine, "steer_now", None)
        if not callable(steer):
            return {"ok": True, "steered": False, "reason": "unsupported"}
        try:
            steered = bool(steer(turn_id))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "steered": False, "reason": f"steer failed: {exc}"}
        return {
            "ok": True,
            "steered": steered,
            "reason": "" if steered else "unsupported",
        }

    async def inject_user_message(
        self,
        session_id: str,
        text: str,
    ) -> Mapping[str, Any]:
        """Push a mid-turn user prompt into the engine's inject buffer.

        Frontend calls this when the operator submits a fresh prompt
        while a multi-round plan is still streaming. The engine drains
        the buffer at every round boundary, appending each entry to
        the live message list as a ``user`` role message before the
        next worker spawn — so the model sees the steering message on
        its very next round.

        Returns ``{"ok": True, "queued": True}`` on success;
        ``{"ok": False, "reason": "..."}`` if the engine doesn't
        support injection or the message is empty. Does NOT require
        an active turn; if no turn is running the message lands at
        the start of the next ``prompt`` call for this sid.
        """
        sess = self._session_for(session_id)
        if sess is None:
            raise RuntimeError(f"session not found: {session_id!r}")
        text = (text or "").strip()
        if not text:
            return {"ok": False, "reason": "empty text"}
        agent = self._get_new_agent()
        ok = False
        try:
            ok = bool(agent.inject_user_message(session_id, text))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": f"inject failed: {exc}"}
        return {"ok": ok, "queued": ok}

    def supervise_goal(self, goal_id: str, *, source: str = "gateway") -> dict[str, Any]:
        """Start or recover outer-loop supervision on this backend's event loop."""
        gid = str(goal_id or "").strip()
        if not gid:
            return {"ok": False, "reason": "empty goal_id"}
        agent = self._get_new_agent()
        try:
            agent.goal_supervisor.ensure_started(gid, source=source)
            active = bool(agent.goal_supervisor.is_active(gid))
            # Re-project the goal onto the gateway's CACHED live session. A goal
            # created from the Goals page is written to disk by a throwaway
            # web-process agent, so this backend's `_active_sessions[sid]` copy
            # (what `/status` → `_session_for` reads to feed the chat goal widget)
            # stays stale with active_goal="". Mutating the cached session here —
            # supervise_goal is called right after create — makes the widget show
            # for goals-page goals too. Best-effort; never fail the supervise.
            try:
                from sharedagents_goal.goal_registry import (
                    get_goal_registry,
                    sync_session_goal_projection,
                )

                goal = get_goal_registry().get(gid)
                if goal is not None and goal.root_session_id:
                    cached = self._active_sessions.get(goal.root_session_id)
                    if cached is not None:
                        sync_session_goal_projection(cached, goal)
            except Exception:  # noqa: BLE001
                pass
            return {"ok": active, "goal_id": gid, "active": active}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "goal_id": gid, "reason": str(exc)}

    def goal_action(self, goal_id: str, action: str) -> dict[str, Any]:
        """Run a stop-style goal action (``pause`` / ``archive``) on the live
        agent so the supervisor's re-drive task is actually cancelled.

        Backs the web Pause/Archive buttons via the gateway control endpoint.
        """
        from sharedagents_goal.goal_service import run_goal_action

        try:
            agent = self._get_new_agent()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"agent unavailable: {exc}"}
        return run_goal_action(agent, goal_id, action)

    def goal_budget(self, session_id: str, action: str, usd: float, goal_id: str = "") -> dict[str, Any]:
        """Raise or set a goal's or prompt's cost budget on the LIVE agent.

        Called when the user explicitly authorizes more spend in chat. Uses the
        in-process agent so ``topup`` → ``resume_goal`` → ``ensure_started``
        reaches the live supervisor and the paused goal actually resumes.
        ``action=set`` sets the absolute session-tree budget (0 = $25 default);
        ``action=topup`` adds ``usd`` and resumes a goal paused for
        ``cost_budget_exhausted``. A session always has a budget; a goal is
        optional and is kept in sync when one is bound.
        """
        from sharedagents_goal.goal_service import (
            set_session_budget,
            topup_session_budget,
        )

        try:
            agent = self._get_new_agent()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"agent unavailable: {exc}"}
        session = self._session_for(session_id)
        try:
            if action == "set":
                res = set_session_budget(agent, session, float(usd))
            elif action == "topup":
                res = topup_session_budget(agent, session, float(usd))
            else:
                return {"ok": False, "message": "action must be 'set' or 'topup'"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"budget failed: {exc}"}
        return {"ok": bool(res.ok), "message": res.message}

    def goal(
        self,
        session_id: str,
        action: str,
        text: str = "",
        goal_id: str = "",
        args: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The model-facing ``goal`` tool, on the LIVE agent.

        ``set`` creates a goal with literal objective text; ``status`` reads it.
        As in Claude Code ``/goal``, the model has no way to end, pause or
        rewrite a goal: the goal evaluator judges the condition after every
        turn, and only the owner pauses or clears it.
        """
        from sharedagents_goal import goal_service as _gs

        try:
            agent = self._get_new_agent()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"agent unavailable: {exc}"}
        session = self._session_for(session_id)
        act = (action or "set").strip().lower()
        try:
            if act == "set":
                cond = (text or "").strip()
                if not cond:
                    return {"ok": False, "message": "goal: `text` (the objective) is required for action=set"}
                if str(getattr(session, "parent_session_id", "") or "").strip():
                    # A delegated child works for its parent's goal; it never
                    # starts a goal loop of its own.
                    return {
                        "ok": False,
                        "message": (
                            "goal: a delegated agent cannot set a goal; do the task you were given and report back."
                        ),
                    }
                live = _gs.live_goal_for_session(session)
                if live is not None:
                    # A set objective is frozen against the model: replacing
                    # it (even with a paraphrase) is the owner's /goal.
                    return {
                        "ok": False,
                        "message": (
                            f"goal {live.goal_id} is already set for this session "
                            "and only the owner can change its objective. Keep "
                            "working on it; it ends when the condition is met."
                        ),
                    }
                # Objective text must never enter the owner's command parser:
                # e.g. "checks cmd: ..." would otherwise accept model-written
                # commands as owner-authored checks on a previously met goal.
                condition, budget_usd = _gs.split_budget(cond)
                res = _gs.create_goal_for_session(
                    agent,
                    session,
                    condition_text=condition,
                    max_cost_usd=budget_usd,
                    replace_existing=False,
                )
            elif act in ("status", "list"):
                res = _gs.execute_goal_command(agent, session, "status", goal_id=goal_id)
            elif act in ("pause", "resume", "restart", "archive", "delete", "clear", "met", "complete", "blocked"):
                return {
                    "ok": False,
                    "message": (
                        f"goal: {act} is not available to you. Keep working; after "
                        "every turn the goal condition is checked against this "
                        "conversation, and the goal ends when it is met."
                    ),
                }
            else:
                return {"ok": False, "message": f"goal: unknown action {act!r} (use set or status)"}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"goal failed: {exc}"}
        if bool(getattr(res, "ok", False)) and act == "set":
            # The goal tool is synchronous (it executes inline inside the live
            # tool runner), so it cannot await LocalLLMEngine.set_config_option.
            # Call the SAME synchronous goal-state hook that that TUI option
            # path delegates to. This replaces any previous _active_goals
            # binding before the current turn's terminal evaluator runs; a
            # stale prior objective must never pause the newly-created row.
            bind_goal = getattr(
                getattr(agent, "engine", None),
                "_set_goal_state",
                None,
            )
            if callable(bind_goal):
                try:
                    active = str(
                        getattr(getattr(res, "goal", None), "condition_text", "")
                        or getattr(session, "active_goal", "")
                        or ""
                    )
                    bind_goal(session_id, active, reset_progress=True)
                except Exception:  # noqa: BLE001 — registry write already committed
                    pass
        return {"ok": bool(getattr(res, "ok", False)), "message": str(getattr(res, "message", "") or "")}

    # ---- Loop/cron tool backends ---------------------
    # Each receives the raw tool JSON args and delegates to the shared
    # ``sharedagents_loop.loop_tools`` runners, which dispatch to ``loop_service``
    # — the same backends the control endpoints and web API will use. The
    # kill-switch (``SHAREDAGENTS_DISABLE_CRON=1``) is enforced in those backends.

    def cron_create(self, session_id: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """``cron_create`` tool: schedule a cron job on the LIVE agent."""
        from sharedagents_loop import loop_tools

        try:
            agent = self._get_new_agent()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"agent unavailable: {exc}"}
        return loop_tools.run_cron_create(
            dict(args),
            agent=agent,
            session_id=session_id,
        )

    def cron_delete(self, session_id: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """``cron_delete`` tool: cancel a scheduled job by ID."""
        from sharedagents_loop import loop_tools

        try:
            agent = self._get_new_agent()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"agent unavailable: {exc}"}
        return loop_tools.run_cron_delete(
            dict(args),
            agent=agent,
            session_id=session_id,
        )

    def cron_list(self, session_id: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """``cron_list`` tool: this session's scheduled jobs."""
        from sharedagents_loop import loop_tools

        try:
            agent = self._get_new_agent()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"agent unavailable: {exc}"}
        return loop_tools.run_cron_list(
            dict(args),
            agent=agent,
            session_id=session_id,
        )

    def schedule_wakeup(self, session_id: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """``schedule_wakeup`` tool: pace/stop the session's running loop."""
        from sharedagents_loop import loop_tools

        try:
            agent = self._get_new_agent()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"agent unavailable: {exc}"}
        return loop_tools.run_schedule_wakeup(
            dict(args),
            agent=agent,
            session_id=session_id,
        )

    async def _run_loop_command(
        self,
        session_id: str,
        arg: str,
    ) -> dict[str, Any]:
        """Execute a ``/loop`` subcommand against the durable loop registry.

        THE single loop-mutation path for the shared command runtime.
        Registers / looks up loops in ``loops.db`` via
        :func:`loop_service.execute_loop_command` and kicks the live
        supervisor. Both the TUI's command context and the CLI-engine shim
        funnel through the same backend seam, so a loop created from ANY
        frontend (TUI, Telegram, web) lands in the registry identically.

        Returns a dict (``ok`` / ``message`` / ``loop`` / ``note`` /
        ``loops``). Never raises for a normal command failure — the caller
        decides how to surface it.
        """
        sess = self._session_for(session_id) if session_id else None
        if sess is None:
            return {"ok": False, "message": f"session not found: {session_id!r}"}
        try:
            agent = self._get_new_agent()
        except Exception:  # noqa: BLE001
            agent = None
        if agent is None:
            return {"ok": False, "message": "agent unavailable"}
        # Attach BEFORE the engine reads the session (a /loop can fire before
        # the first prompt) so the supervisor's session binding lands on the
        # right engine instance.
        attach = getattr(agent.engine, "attach_session", None)
        if callable(attach):
            try:
                attach(session_id, sess)
            except Exception:  # noqa: BLE001
                pass
        from sharedagents_loop.loop_service import execute_loop_command

        return await execute_loop_command(agent, sess, arg or "")

    def loop_action(self, loop_id: str, action: str) -> dict[str, Any]:
        """Run a loop action (``pause`` / ``resume`` / ``archive``) on the
        live agent so the supervisor's drive task is actually cancelled.

        Backs the web Pause/Resume/Archive buttons via the gateway control
        endpoint (mirror of :meth:`goal_action`).
        """
        from sharedagents_loop.loop_service import run_loop_action

        try:
            agent = self._get_new_agent()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"agent unavailable: {exc}"}
        return run_loop_action(agent, loop_id, action)

    async def supervise_loop(self, loop_id: str, *, source: str = "gateway") -> dict[str, Any]:
        """Start or recover loop supervision on this backend's event loop.

        Mirrors :meth:`supervise_goal` for the loop supervisor: idempotent
        kick of exactly one live drive task per loop on the agent that owns
        it, plus the ``is_active`` report the control endpoint returns.

        The kick is awaited so the report reflects the task just started
        (the supervisor coroutine runs inline and creates the drive task
        before returning) — closing the false-409 window that an
        ``ensure_future`` + immediate read left open. Unknown or not-running
        loops still fail fast: the kick never creates a task for them.
        """
        key = str(loop_id or "").strip()
        if not key:
            return {"ok": False, "reason": "empty loop_id"}
        try:
            agent = self._get_new_agent()
            sup = getattr(agent, "loop_supervisor", None)
            if sup is None:
                return {
                    "ok": False,
                    "loop_id": key,
                    "reason": "loop supervisor unavailable",
                }
            result = await sup.supervise_loop(key, source=source)
            active = sup.is_active(key)
            kicked = bool(result.get("ok")) if isinstance(result, dict) else False
            return {"ok": kicked and active, "loop_id": key, "active": active}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "loop_id": key, "reason": str(exc)}

    async def run_command(
        self,
        session_id: str,
        text: str,
        *,
        frontend: str = "api",
        confirmed: bool = False,
    ) -> Mapping[str, Any]:
        """Execute a slash command via the shared server-side runtime.

        THE single command implementation lives in
        :mod:`sharedagents_command.command_runtime`; this method only adapts the
        live backend state into a :class:`CommandContext` and returns the
        :class:`CommandResponse` as a JSON-shaped dict. The ACP
        ``session/command`` method and the gateway REST control endpoint
        both call here, so every frontend (TUI, Telegram, web) renders the
        same output — zero command logic on any frontend.
        """
        from sharedagents_command.command_runtime import CommandContext, run_command as _run_command

        sess = self._session_for(session_id) if session_id else None

        async def _list_sessions() -> list[dict[str, Any]]:
            try:
                res = await self.list_sessions(None, None)
            except Exception:  # noqa: BLE001
                return []
            if isinstance(res, Mapping):
                return list(res.get("sessions") or [])
            return []

        def _monitors() -> list[dict[str, Any]]:
            try:
                from sharedagents_agent.tools.monitor import _registry

                return list(_registry().list())
            except Exception:  # noqa: BLE001
                return []

        ctx = CommandContext(
            cfg=self.cfg,
            agent_name=getattr(self.cfg, "name", "") or "",
            session_id=session_id or "",
            frontend=frontend,
            live_session=sess,
            active_turn_id=(self._new_backend_active_turns.get(session_id) if session_id else None),
            confirmed=confirmed,  # irreversible-command re-challenge
            set_mode_fn=self.set_mode,
            set_config_option_fn=self.set_config_option,
            list_sessions_fn=_list_sessions,
            cancel_fn=self.cancel,
            monitors_fn=_monitors,
            inject_fn=self.inject_user_message,
            goal_command_fn=lambda arg: self._run_goal_command(
                session_id or "",
                arg,
            ),
            loop_command_fn=lambda arg: self._run_loop_command(
                session_id or "",
                arg,
            ),
        )
        resp = await _run_command(ctx, text)
        return resp.to_dict()

    async def set_mode(self, session_id: str, mode_id: str) -> None:
        # Validate UP FRONT, before mutating any state. A late
        # ValueError from ``engine.set_mode`` would otherwise leave
        # ``sess.system_mode`` mutated to the bogus value while disk +
        # engine stay at the old mode — a small split-brain window.
        # Mirrors the entry-time check in ``gateway_shim.set_mode``.
        from sharedagents_agent.agents.permissions_integration import (
            coerce_permission_mode,
        )

        # Native ACP aliases (Cursor ``agent``, OpenCode ``build``, …)
        # collapse onto the permission set. Persist the permission id so
        # ``_effective_mode`` and local-LLM ``engine.set_mode`` stay in
        # {plan, ask, auto}; the engine maps ``auto`` back to the native
        # wire id on the next turn (``_acp_mode``).
        mode_id = coerce_permission_mode(mode_id)
        sess = self._session_for(session_id)
        if sess is None:
            raise RuntimeError(f"session not found: {session_id!r}")
        sess.system_mode = mode_id

        # Propagate to the engine's per-session ``_mode_overrides``
        # map FIRST (before the disk fsync below) so the live turn
        # loop's rounds-cap + permission policy see the new mode with
        # zero latency. Eagerly build the engine if the client called
        # ``set_mode`` before its first prompt — otherwise the engine
        # state is set on the wrong instance and the mode silently
        # no-ops for the rest of the session.
        try:
            agent = self._get_new_agent()
        except Exception:  # noqa: BLE001
            agent = None
        if agent is not None:
            engine_set_mode = getattr(agent.engine, "set_mode", None)
            if callable(engine_set_mode):
                try:
                    res = engine_set_mode(session_id, mode_id)
                    if hasattr(res, "__await__"):
                        await res
                except ValueError:
                    raise
                except Exception:  # noqa: BLE001
                    pass

        # Persist LAST. Without this, a gateway restart between
        # ``/mode plan`` and the next prompt loses the override: the
        # in-memory mutation dies and the on-disk JSON still says the
        # old mode. Mirrors the ``sess.save(...)`` pattern used by the
        # ``set_config_option`` paths below (L1887/L1900). Persistence
        # failure must be LOUD — silently swallowing it would resurrect
        # the very split-brain this fix closes (in-memory says plan,
        # disk says auto → next restart elevates back to auto).
        try:
            sess.save(self.cfg.sessions_dir)
        except OSError as exc:
            # Persist BEFORE notifying. A failed persist must
            # NOT be announced — announcing a mode we couldn't write to disk
            # resurrects the split-brain (client believes ``plan``, disk still
            # says ``auto``, next restart elevates back). Log and return WITHOUT
            # emitting ``current_mode_update``.
            _LOG.warning(
                "set_mode: failed to persist session %s system_mode=%s: %s",
                session_id,
                mode_id,
                exc,
            )
            return
        # Spec compliance (ACP 0.12.x): emit ``current_mode_update`` so
        # the client can refresh its mode UI. Fire-and-forget — we
        # MUST NOT block the set_mode response on a transport write,
        # because the same transport is also writing the response to
        # the in-flight set_mode request and a circular await on
        # ``write_lock`` would deadlock the whole connection.
        # Use the per-session outbound resolver so a TUI + Telegram
        # multi-connection setup notifies the connection that owns
        # this session, not whichever happened to attach last.
        srv = self._outbound_for(session_id) if hasattr(self, "_outbound_for") else getattr(self, "server_handle", None)
        notify = getattr(srv, "notify_session_update", None)
        if notify is None:
            return

        async def _emit() -> None:
            try:
                await notify(
                    session_id=session_id,
                    update={
                        "sessionUpdate": "current_mode_update",
                        "currentModeId": mode_id,
                        # Carry the canonical mode list so the
                        # frontend renders an identical mode UI regardless of
                        # which backend/engine drives the session.
                        "availableModes": self._modes_state(mode_id)["availableModes"],
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        try:
            asyncio.create_task(_emit())
        except RuntimeError:
            # No running loop (in-process tests); skip the notification.
            pass

    async def _run_goal_command(
        self,
        session_id: str,
        arg: str,
    ) -> dict[str, Any]:
        """Execute a ``/goal`` subcommand against the durable registry.

        THE single goal-mutation path. Registers / looks up goals in
        ``goals.db`` via :func:`goal_service.execute_goal_command` and syncs
        the live engine's per-session goal projection. Both the TUI's
        ``set_config_option("goal", …)`` path and the shared command
        runtime's ``/goal`` handler funnel through here, so a goal set from
        ANY frontend (TUI, Telegram, web) lands in the registry identically.

        Returns a dict (``ok`` / ``message`` / ``goal`` / ``conflicts`` /
        ``active_goal`` / ``goal_id`` / ``goal_terminal`` /
        ``goal_turn_count`` — the last three are the post-command projection
        for the /goal state_patch). Never raises for a normal command failure
        (conflict, unknown goal id) — the caller decides how to surface it.
        """
        sess = self._session_for(session_id) if session_id else None
        if sess is None:
            return {"ok": False, "message": f"session not found: {session_id!r}"}
        attached_sess = sess
        try:
            agent = self._get_new_agent()
        except Exception:  # noqa: BLE001
            agent = None
        if agent is None:
            return {"ok": False, "message": "agent unavailable"}
        # Attach BEFORE the engine reads the session (a /goal can fire before
        # the first prompt) so the per-session goal projection lands on the
        # right engine instance.
        attach = getattr(agent.engine, "attach_session", None)
        if callable(attach):
            try:
                attach(session_id, sess)
            except Exception:  # noqa: BLE001
                pass
        from sharedagents_goal.goal_service import execute_goal_command

        result = execute_goal_command(agent, sess, arg)
        refreshed = agent.get_session(session_id)
        if refreshed is not None:
            sess = refreshed
        # Capture the AUTHORITATIVE post-command projection BEFORE the
        # generic engine goal setter. The engine's
        # ``set_config_option("goal", …)`` → ``_set_goal_state(…,
        # reset_progress=True)`` clears ``goal_terminal`` and
        # ``goal_turn_count`` on ITS cached Session (attach_session) —
        # which, for a session without a sessions_meta row yet, is the very
        # object this backend returns (get_session falls back to the engine
        # cache). The service already wrote the authoritative projection;
        # the setter must not clobber it, and the returned dict must be the
        # service's state, not the setter's leftovers (pause projection
        # clobber in real CLI/ACP adapter paths).
        projection = {
            "active_goal": str(getattr(sess, "active_goal", "") or ""),
            "goal_id": str(getattr(sess, "goal_id", "") or ""),
            "goal_terminal": str(getattr(sess, "goal_terminal", "") or ""),
            "goal_turn_count": int(getattr(sess, "goal_turn_count", 0) or 0),
        }
        if result.ok:
            engine_set_opt = getattr(agent.engine, "set_config_option", None)
            if callable(engine_set_opt):
                try:
                    res = engine_set_opt(
                        session_id,
                        "goal",
                        projection["active_goal"],
                    )
                    if hasattr(res, "__await__"):
                        await res
                except Exception:  # noqa: BLE001
                    pass
            # The setter owns ``attached_sess`` while the service/refresh path
            # can own a distinct freshly hydrated ``sess``. Restore both
            # identity domains after binding so the engine cache and the
            # persisted projection agree.
            for target in (attached_sess, sess):
                for key, value in projection.items():
                    try:
                        setattr(target, key, value)
                    except Exception:  # noqa: BLE001
                        pass
            try:
                sess.save()
            except Exception:  # noqa: BLE001
                pass
        return {
            "ok": result.ok,
            "message": result.message,
            "goal": result.goal.to_dict() if result.goal else None,
            "conflicts": result.conflicts,
            # Post-command projection from the REFRESHED session — the
            # authoritative snapshot for the /goal state_patch. The command
            # runtime's ctx.live_session may be a stale pre-command object
            # (a lazy-loaded session is a separate instance), so the patch
            # must use THIS projection.
            **projection,
        }

    async def set_config_option(self, session_id: str, config_id: str, value: Any) -> None:
        """Apply a session-scoped config option.

        Today the only registered option is ``model`` — the value is an
        alias from ``cfg.models[]``. We resolve the row, mutate the
        bound :class:`Session` so subsequent worker-spawn rounds talk to
        the new endpoint/model, and persist via :meth:`Session.save` so
        a frontend reattach (or session reload after a restart) sees
        the same model.

        Unknown aliases raise :class:`ValueError`; the dispatcher's
        generic-exception handler turns the raise into a JSON-RPC
        error response so the caller sees a clean failure rather than
        a silent no-op.
        """
        sess = self._session_for(session_id)
        if sess is None:
            raise RuntimeError(f"session not found: {session_id!r}")

        # Propagate every option to the live engine so per-session
        # ``_promoted_tools`` / ``reasoning_visibility`` / ``temperature``
        # / etc. take effect on the next round. Eagerly build the
        # engine if the client called ``set_config_option`` before
        # its first prompt — the previous gating on
        # ``self._new_agent is not None`` silently no-op'd
        # pre-prompt callers and dropped the option on the floor.
        try:
            agent = self._get_new_agent()
        except Exception:  # noqa: BLE001
            agent = None

        if config_id in ("goal", "goal_replace", "goal_advance"):
            # Delegate to the shared registry path so the TUI's
            # ``set_config_option("goal", …)`` and the command runtime's
            # ``/goal`` handler register goals identically. A normal command
            # failure (conflict, bad id) surfaces as ValueError, which the
            # dispatcher turns into a clean JSON-RPC error for the caller.
            value_s = "" if value is None else str(value)
            res = await self._run_goal_command(session_id, value_s)
            if not res.get("ok", False):
                raise ValueError(res.get("message") or "goal command failed")
            return

        if agent is not None:
            # Ensure the session is attached before the engine reads it.
            # ``set_config_option`` may fire BEFORE the first prompt
            # (TUI seeds ``reasoning=on`` right after ``session_new``);
            # without this, ``LocalLLMEngine.set_config_option`` for
            # the ``reasoning`` option would raise — its writer mutates
            # ``self._sessions[session_id].reasoning_visibility`` and
            # the per-session dict is otherwise populated only by the
            # first-prompt path (~L2102).
            attach = getattr(agent.engine, "attach_session", None)
            if callable(attach):
                try:
                    attach(session_id, sess)
                except Exception:  # noqa: BLE001
                    pass
            engine_set_opt = getattr(
                agent.engine,
                "set_config_option",
                None,
            )
            if callable(engine_set_opt):
                try:
                    res = engine_set_opt(session_id, config_id, value)
                    if hasattr(res, "__await__"):
                        await res
                except ValueError:
                    raise
                except Exception:  # noqa: BLE001
                    pass

        # Effort/speed are session state even for external CLI/ACP engines
        # whose transport owns the actual wire.  LocalLLMEngine already
        # validates and stores these options; repeating the assignment here
        # keeps the CoreBackend session authoritative across every engine.
        if config_id == "effort":
            from sharedagents_command.commands import normalize_effort

            value_s = normalize_effort(str(value or ""))
            value_s = "" if value_s == "default" else value_s
            sess.effort = value_s
            try:
                sess.save()
            except Exception:  # noqa: BLE001
                pass
        elif config_id == "speed":
            value_s = "" if value is None else str(value).strip().lower()
            if value_s in ("", "default", "auto", "reset", "clear", "none"):
                value_s = ""
            elif value_s not in {"fast", "standard"}:
                raise ValueError(
                    f"speed: invalid value {value!r} — use default|fast|standard|auto",
                )
            sess.speed = value_s
            try:
                sess.save()
            except Exception:  # noqa: BLE001
                pass

        if config_id == "model":
            value_s = "" if value is None else str(value)
            engine = getattr(agent, "engine", None) if agent is not None else None
            kind = getattr(engine, "kind", "")
            kind_value = str(getattr(kind, "value", kind) or "")
            from sharedagents_agent.external_registry import is_external_engine_kind

            if is_external_engine_kind(kind_value):
                selected = ""
                getter = getattr(engine, "session_model", None)
                if callable(getter):
                    try:
                        selected = str(getter(session_id) or "").strip()
                    except Exception:  # noqa: BLE001
                        selected = ""
                if selected:
                    sess.model = selected
                    sess.endpoint = ""
                    sess.backend = kind_value
                    try:
                        ctx_window = getattr(engine, "context_window", None)
                        sess.context_window = int(ctx_window(selected) or 0) if callable(ctx_window) else 0
                    except (TypeError, ValueError):
                        sess.context_window = 0
                else:
                    # RESET / clear (``/model default|reset|auto|none``):
                    # the engine's override is gone, so resolve the stamp
                    # through the SAME generic effective-default/empty path
                    # the fresh-spawn and session-create paths use —
                    # NEVER ``display_model()`` (which returns the bare
                    # engine label when nothing is known: the codex probe
                    # reproduced a durable session model of "codex" here)
                    # and never ``kind_value``.
                    # For CLI kinds this yields the validated default
                    # (codex: gpt-5.6-sol; claude: the operator-explicit
                    # per-agent value / ANTHROPIC_MODEL / settings.json, or
                    # "" — the Claude CLI applies its own default); for ACP
                    # kinds "" (the ACP server owns the model).
                    from sharedagents_agent.discovery._common import (
                        external_cli_effective_default,
                    )

                    sess.model = external_cli_effective_default(
                        kind_value,
                        configured=(str(getattr(agent.cfg, "upstream_model", "") or "") if agent is not None else None),
                        configured_explicit=bool(
                            getattr(
                                agent.cfg,
                                "_upstream_model_is_per_agent",
                                False,
                            )
                            if agent is not None
                            else False
                        ),
                    )
                    sess.endpoint = ""
                    sess.backend = kind_value
                    sess.context_window = 0
                # Persistence failure must be LOUD and uniform with every
                # other session write: a poisoned writer /
                # durable-write failure propagates so the dispatcher turns it
                # into a JSON-RPC error, rather than silently dropping the
                # /model switch while turn writes fail loudly.
                sess.save(self.cfg.sessions_dir)
                return

            row = self._resolve_model_row(value_s)
            if row is None and value_s.startswith(("http://", "https://")):
                # Manual ``/model https://…`` / picker-without-alias
                # path: resolve the row by URL through the regular
                # endpoint resolver. ``model_row_for_endpoint``
                # returns a synthetic row when the URL doesn't match
                # any configured entry — that's intentional, lets an
                # operator switch to an ad-hoc upstream.
                try:
                    candidate = self.cfg.model_row_for_endpoint(value_s)
                except Exception:  # noqa: BLE001
                    candidate = None
                if candidate and candidate.get("baseUrl"):
                    row = candidate
            if row is None:
                raise ValueError(f"unknown model alias: {value_s}")
            row_id = str(row.get("id") or "").strip()
            base_url = str(row.get("baseUrl") or "")
            if not row_id:
                # Auto-discovery removed: every model row must name an
                # explicit id. Reject the switch with a clear pointer
                # to the config so the operator knows where to fix it.
                raise ValueError(
                    f"model alias {value_s!r} has no `id` configured "
                    f"({base_url or '<no baseUrl>'}). Set the model id "
                    f"explicitly in ~/.sharedagents/config.json — auto-discover "
                    f"via /v1/models has been removed."
                )
            sess.model = row_id
            sess.endpoint = base_url
            sess.backend = str(row.get("backend") or "")
            try:
                sess.context_window = int(row.get("contextWindow") or 0)
            except (TypeError, ValueError):
                sess.context_window = 0
            # Persistence failure must be LOUD and uniform with every other
            # session write: propagate rather than swallow,
            # so a /model persist failure is observable on the same channel as
            # a turn write instead of resurrecting the in-memory/on-disk
            # split-brain the swallow used to hide.
            sess.save(self.cfg.sessions_dir)
            return

        if config_id == "reasoning":
            # Engine mutated ``sess.reasoning_visibility``. Persist so a
            # session reattach (TUI restart, gateway restart) sees the same
            # preference instead of falling back to the disk default. A
            # persistence failure propagates LOUDLY, uniform with the /model
            # path and every turn write — no silent swallow.
            sess.save(self.cfg.sessions_dir)
            return

        # Unknown config id — record on the session so subsequent
        # ``set_mode`` / ``load_session`` round-trips can observe the
        # value, but don't act on anything else.
        if not hasattr(sess, "_acp_config"):
            sess._acp_config = {}  # type: ignore[attr-defined]
        sess._acp_config[config_id] = value  # type: ignore[attr-defined]

    def _resolve_model_row(self, alias: str) -> Mapping[str, Any] | None:
        """Look up a model row in ``cfg.models`` by alias or wire id.

        Returns ``None`` when neither selector matches a configured row. Rows
        from older configs may omit ``alias``; their explicit wire ``id`` is
        still a valid exact selector. We deliberately do NOT fall through to
        :meth:`AgentConfig.resolve_model` here — that helper synthesizes
        a primary row when nothing matches, which would silently mask
        an unknown-alias request.
        """
        if not alias:
            return None
        rows = [dict(row) for row in (getattr(self.cfg, "models", None) or []) if isinstance(row, Mapping)]
        try:
            from sharedagents_agent.discovery.cache_lock import native_models_with_cache

            rows = native_models_with_cache(
                str(getattr(self.cfg, "name", "") or "jarvis"),
                rows,
            )
        except Exception:  # noqa: BLE001
            pass
        for row in rows:
            if str(row.get("alias") or "") == alias or str(row.get("id") or "") == alias:
                return dict(row)
        return None

    def active_turn_id(self, session_id: str) -> str | None:
        """The live in-flight ``turn_id`` for ``session_id``, or ``None``.

        Public read over ``_new_backend_active_turns`` (the same map cancel
        routes through). The gateway's submit route consults this so a turn
        started outside submit — a TUI/Telegram ACP turn mid-migration — also
        blocks a concurrent submit (spec §9.1 single-flight).
        """
        return self._new_backend_active_turns.get(session_id)

    # Max prompts (running + queued) allowed per session before new
    # ones are refused. 1 running + 8 waiting on the per-session lock.
    _MAX_INFLIGHT_PROMPTS = 9

    def _admit_prompt(self, session_id: str) -> bool:
        """Reserve a per-session prompt slot. Returns False (without
        reserving) when the session already has ``_MAX_INFLIGHT_PROMPTS``
        prompts running/queued — the caller must then refuse the prompt.
        Pair every True with exactly one :meth:`_release_prompt`.
        """
        n = self._prompt_depth.get(session_id, 0)
        if n >= self._MAX_INFLIGHT_PROMPTS:
            return False
        self._prompt_depth[session_id] = n + 1
        return True

    def _release_prompt(self, session_id: str) -> None:
        """Release a slot reserved by :meth:`_admit_prompt`. Drops the
        per-session counter entirely once it hits zero so idle sessions
        don't accumulate keys."""
        n = self._prompt_depth.get(session_id, 0) - 1
        if n <= 0:
            self._prompt_depth.pop(session_id, None)
        else:
            self._prompt_depth[session_id] = n

    async def cancel(self, session_id: str, *, force: bool = False) -> None:
        """Stop the in-flight turn for ``session_id``.

        Two operator intents, threaded explicitly from the frontend
        (``force`` rides on the ``session/cancel`` notification):

        * **graceful** (``force=False`` — ``/stop`` / first ESC): route
          through :meth:`Engine.request_graceful_stop`. Kills NOTHING —
          the in-flight tool / streaming round finishes, then the round
          loop ends the turn before the next round / next tool group.
        * **hard** (``force=True`` — ``/abort`` / second ESC): route
          through :meth:`Engine.abort(turn_id, force=True)` → SIGKILL +
          ``killpg`` on the worker AND every registered tool runner /
          subagent subprocess, and flip the engine's hard ``_cancelled``
          flag so the round/dispatch loops break immediately.

        A double-cancel arriving inside ``_HARD_KILL_WINDOW_S`` still
        escalates to a hard kill even if the frontend didn't set
        ``force`` — a safety net for clients that mash cancel without
        the two-stage UI. Tracked even when no turn is active so a
        cancel-after-turn-end double-press doesn't reset the window for
        the next turn.
        """
        now_mono = _time.monotonic()
        last_mono = self._last_cancel_at.get(session_id)
        within_window = bool(last_mono is not None and (now_mono - last_mono) <= _HARD_KILL_WINDOW_S)
        self._last_cancel_at[session_id] = now_mono
        force_kill = bool(force) or within_window

        # An operator stop (``/abort`` OR graceful ``/stop``) on a goal-driven
        # session must halt the goal's outer loop, not just the in-flight turn:
        # otherwise the supervisor sees the goal still ``running`` and re-drives
        # turn N+1 ("it just kept running"). Pause the goal (resumable) first so
        # the supervisor's status re-check folds the loop out. Never raises.
        if self._new_agent is not None:
            try:
                from sharedagents_goal.goal_service import stop_goals_for_session

                stop_goals_for_session(self._new_agent, session_id, reason="aborted")
            except Exception:  # noqa: BLE001
                pass
            # Same contract for loops: an operator stop
            # ends the session's loops + cancels pending wakeups. Never raises.
            try:
                from sharedagents_loop.loop_service import stop_loops_for_session

                stop_loops_for_session(self._new_agent, session_id, reason="aborted")
            except Exception:  # noqa: BLE001
                pass
            # Stop must end the between-attempt retry
            # sleep even when the mapped turn_id is already stale (inner
            # attempt finished, outer burst still holding conversation_busy).
            try:
                cancel_burst = getattr(
                    self._new_agent,
                    "cancel_retry_burst",
                    None,
                )
                if callable(cancel_burst):
                    cancel_burst(session_id)
            except Exception:  # noqa: BLE001
                pass

        # Record the Stop BEFORE any engine is killed (same contract as the
        # CLI-engine path): it tells a runner that sees its engine die a user
        # Stop from an OOM kill, and gates every automatic turn start (late
        # child results, memory flush, retries) until the user prompts again.
        from sharedagents_agent.agents import session_stop

        stopped_below = session_stop.begin_stop(
            self._new_agent,
            self.cfg.sessions_dir,
            session_id,
            force=force_kill,
        )

        new_turn_id = self._new_backend_active_turns.get(session_id)
        had_active_turn = bool(new_turn_id)
        if new_turn_id and self._new_agent is not None:
            engine = self._new_agent.engine
            if force_kill:
                # Hard kill: SIGKILL + killpg the whole turn process
                # group (worker + every in-flight runner / subagent).
                try:
                    await engine.abort(new_turn_id, force=True)
                except TypeError:
                    try:
                        await engine.abort(new_turn_id)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
            else:
                # Graceful: cooperative stop, no signals. Fall back to a
                # plain (SIGTERM) abort only if the engine predates the
                # graceful-stop API, so a ``/stop`` never silently no-ops.
                graceful = getattr(engine, "request_graceful_stop", None)
                if graceful is not None:
                    try:
                        await graceful(new_turn_id)
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    try:
                        await engine.abort(new_turn_id)
                    except Exception:  # noqa: BLE001
                        pass
        # Cascade the kill DOWN the session tree. A subagent runs in its
        # OWN child session (linked durably via
        # ``sessions_meta.parent_session_id``); aborting the parent above
        # only touches the parent's own ``turn_id`` and leaves each live
        # child registered/streaming — its subprocess is reaped by the
        # runner's ``_kill_tree`` on the same tick, but its session log
        # would otherwise hang in a "live" state (zombie on the dashboard)
        # until the next gateway restart's ``reset_stale_sessions`` sweep.
        # Only on a HARD abort: a graceful ``/stop`` must let an in-flight
        # subagent finish its current round, not tear its children down.
        if force_kill:
            # The active-turn map can be EMPTY for a session whose turn died
            # abnormally — its ``prompt()`` finally already popped the entry,
            # yet no terminal ``turn_ended`` ever reached the log (hung
            # delegate, torn SSE), so the session stays stuck in a live
            # ``session_state`` and the UI shows "still working" FOREVER while
            # every Stop press logs ``had_active_turn=false`` and does nothing
            # (the 2026-06-16 web lock). A hard abort must NEVER be a silent
            # no-op: when no turn is mapped, finalize a live-but-unmapped
            # session directly so it folds out of the live set and the
            # operator is never locked.
            if not had_active_turn:
                # An engine subprocess may still be live even with no mapped
                # turn — lingering after its result, or ORPHANED across an
                # orchestrator restart (start_new_session=True → it survives
                # the parent and the in-memory maps are gone). SIGKILL it
                # before finalizing: prefer the engine's by-session abort
                # (in-memory + durable), else fall back to the standalone
                # durable killer, which needs NO in-memory state and so works
                # even post-restart.
                eng = getattr(self._new_agent, "engine", None)
                abort_session = getattr(eng, "abort_session", None)
                reaped = False
                if abort_session is not None:
                    try:
                        await abort_session(
                            session_id,
                            sessions_dir=self.cfg.sessions_dir,
                            force=True,
                        )
                        reaped = True
                    except Exception:  # noqa: BLE001
                        pass
                if not reaped:
                    try:
                        from sharedagents_agent.agents import engine_pidfile

                        engine_pidfile.kill_session_engine(
                            self.cfg.sessions_dir,
                            session_id,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                await self._finalize_unmapped_session(session_id)
            await self._cancel_child_sessions(session_id, force_kill=True)
            # Background (detached) delegations are NOT in-gateway turns, so the
            # cascade above doesn't reach them — reap the whole background
            # subtree (this session + every descendant) by killpg-ing the
            # detached runners. Walks DOWN only, so a sub-agent abort never
            # touches the orchestrator. Silence deliveries so the live
            # reconciler cannot revive the aborted parent with a retry prompt
            # (/abort stays dead).
            try:
                from sharedagents_agent.agents import async_jobs

                async_jobs.cancel_subtree_for_abort(session_id)
            except Exception:  # noqa: BLE001 — must not fail the operator's abort
                pass
            # Every descendant, not just the live rows walked above: an idle
            # mid-level child can still hold busy grandchildren.
            stopped_below = session_stop.stop_descendants(
                self.cfg.sessions_dir,
                session_id,
                stopped_below,
                agent=self._new_agent,
            )
            # Waiting parents have no live_turn_idx — finalize never flips them.
            # After the subtree is killed, clear ``waiting`` so Stop/monitor
            # don't stick on a dead tree.
            await self._clear_waiting_state(session_id)
            # Purge the follow-up queues so stopped items never auto-drain or
            # get recovered into new turns (parity with the CLI-engine path).
            if self._new_agent is not None:
                session_stop.purge_queues(
                    self._new_agent,
                    [session_id, *stopped_below],
                )
        # External cancel (ESC in TUI, ``/stop`` / ``/abort``, ACP
        # session/cancel from a remote client). ``mode`` distinguishes
        # the operator's graceful "wind down" from the hard "kill now"
        # so a post-mortem can tell which press finally landed.
        self._log_event(
            "EXTERNAL_CANCEL",
            sid=session_id[:12],
            had_active_turn=had_active_turn,
            force=force_kill,
            mode="hard" if force_kill else "graceful",
        )

    async def _finalize_unmapped_session(self, session_id: str) -> None:
        """Append ``turn_ended(aborted)`` to a LIVE session that has NO turn
        registered in ``_new_backend_active_turns``.

        This is the backstop that makes a hard abort un-fail-able. When a turn
        dies without a clean terminal event (a hung delegate, an engine death,
        a torn gateway SSE), the in-memory active-turn entry is already gone
        but the session's ``session_state`` is still live — so the UI never
        clears and the operator's Stop button is a no-op. Reads the live turn
        index straight from ``sessions_meta`` and, ONLY if the session is still
        in a live state, writes the close that folds it out of the live set —
        mirroring the per-child logic in :meth:`_cancel_child_sessions`.

        Never raises: a bookkeeping failure here must not fail the abort.
        """
        try:
            from sharedagents_session.session_activity import (
                TREE_BUSY_STATES,
            )
            from sharedagents_session.session_index import (
                connect_sessions_db,
                mark_session_terminal,
            )
            from sharedagents_session.session_writer import (
                close_session_writer,
                get_session_writer,
            )

            sd = self.cfg.sessions_dir
            db = sd / "sessions.db"
            if not db.exists():
                return
            _LIVE_STATES = tuple(TREE_BUSY_STATES)
            con = connect_sessions_db(db)
            try:
                row = con.execute(
                    "SELECT session_state, live_turn_idx, last_turn_idx FROM sessions_meta WHERE session_hash=?",
                    (session_id,),
                ).fetchone()
            finally:
                con.close()
            if row is None:
                return
            if row["session_state"] not in _LIVE_STATES:
                return
            # ``waiting`` has no live turn — nothing to finalize here
            # (the hard-cancel path clears it via ``_clear_waiting_state``).
            if row["session_state"] == "waiting":
                return
            live_ti = row["live_turn_idx"]
            if live_ti is None:
                # FIX (2026-08-11 'waiting' wedge): a live row with NULL
                # live_turn_idx (trailing delta after a prior turn ended)
                # used to return here and stay live forever. Fall back to
                # ``last_turn_idx`` — the in-flight turn that never closed —
                # mirroring ``reset_stale_sessions``.
                live_ti = row["last_turn_idx"]
            if live_ti is not None and not _turn_has_terminal_event(sd, session_id, int(live_ti)):
                w = get_session_writer(sd, session_id, engine="")
                w.turn_ended(
                    turn_idx=int(live_ti),
                    aborted=True,
                    metrics={"abort_reason": "force_cancel_no_active_turn"},
                )
                close_session_writer(sd, session_id)
                self._log_event(
                    "ORPHANED_TURN_FINALIZED",
                    sid=session_id[:12],
                    turn_idx=int(live_ti),
                )
            # Unconditional terminal backstop: the row must leave the live
            # set even when no turn event could be emitted (NULL live+last
            # idx, or the event already existed on the log).
            mark_session_terminal(sd, session_id, state="error")
        except Exception:  # noqa: BLE001 — never let finalize fail the abort
            pass

    async def _clear_waiting_state(self, session_id: str) -> None:
        """Flip ``waiting`` → ``idle`` after a hard tree cancel.

        ``waiting`` means the parent's own turn already ended; force-cancel
        therefore has no ``live_turn_idx`` to abort-close, and without this
        write the parent stays ``waiting`` forever once children/jobs are gone
        (nothing else re-folds the parent row). Never raises.
        """
        try:
            import time as _time

            from sharedagents_session.session_index import connect_sessions_db

            sd = self.cfg.sessions_dir
            db = sd / "sessions.db"
            if not db.exists():
                return
            now_ms = int(_time.time() * 1000)
            con = connect_sessions_db(db)
            try:
                con.execute(
                    "UPDATE sessions_meta SET session_state='idle', "
                    "session_phase='idle', session_attention='none', "
                    "session_state_updated_at=? "
                    "WHERE session_hash=? AND session_state='waiting'",
                    (now_ms, session_id),
                )
                con.commit()
            finally:
                con.close()
        except Exception:  # noqa: BLE001
            pass

    async def _cancel_child_sessions(
        self,
        parent_session_id: str,
        *,
        force_kill: bool = True,
        _depth: int = 0,
    ) -> None:
        """Recursively terminate all live child (subagent) sessions.

        Called after cancelling the parent's turn on a HARD abort. For
        each live child session (found via
        ``sessions_meta.parent_session_id``):

        1. Abort the child's subprocess if it has a live turn registered
           in ``_new_backend_active_turns`` (the engine's active-turn
           map) — reuses the same ``engine.abort`` path as the parent.
        2. Append ``turn_ended(aborted)`` to the child's JSONL so its
           session state folds out of the live set immediately, instead
           of waiting for the next restart's ``reset_stale_sessions``.
        3. Recurse into grandchildren (``_depth`` guard bounds malformed
           parent-cycles).

        Never raises: a bookkeeping failure here must not fail the
        operator's abort of the parent.
        """
        if _depth >= 10:
            return  # depth guard against malformed parent cycles
        try:
            import time as _time

            from sharedagents_agent.orchestrator import LIVE_FRESHNESS_SECONDS
            from sharedagents_session.session_index import (
                EventProjector,
                connect_sessions_db,
                mark_session_terminal,
            )
            from sharedagents_session.session_writer import (
                close_session_writer,
                get_session_writer,
            )

            sd = self.cfg.sessions_dir
            db = sd / "sessions.db"
            if not db.exists():
                return
            # Include ``waiting`` so a mid-level parent that already ended
            # its turn but still holds live grandchildren is still walked
            # during hard abort cascade.
            from sharedagents_session.session_activity import TREE_BUSY_STATES

            _LIVE_STATES = tuple(TREE_BUSY_STATES)
            _TURN_STATES = set(TREE_BUSY_STATES) - {"waiting"}
            child_con = connect_sessions_db(db)
            try:
                live_phs = ",".join("?" for _ in _LIVE_STATES)
                rows = child_con.execute(
                    "SELECT session_hash, session_state, live_turn_idx, "
                    "last_turn_idx FROM sessions_meta "
                    "WHERE parent_session_id=? "
                    f"  AND session_state IN ({live_phs})",
                    (parent_session_id, *_LIVE_STATES),
                ).fetchall()
            finally:
                child_con.close()
            for child_row in rows:
                child_hash = child_row["session_hash"]
                child_state = child_row["session_state"]
                child_ti = child_row["live_turn_idx"]
                if not child_hash:
                    continue
                # 1. Abort the child's subprocess if a live turn is mapped.
                child_turn_id = self._new_backend_active_turns.get(child_hash)
                if child_turn_id and self._new_agent is not None:
                    try:
                        await self._new_agent.engine.abort(
                            child_turn_id,
                            force=force_kill,
                        )
                    except Exception:  # noqa: BLE001
                        pass
                # 1b. SIGKILL the child's engine process subtree from the
                # DURABLE registry. A subagent (local_llm) turn runs in a
                # ``tool_runner`` SUBPROCESS that is NOT in this gateway's
                # in-memory ``_new_backend_active_turns`` map, so step 1 is a
                # no-op for it — the only thing that reaped its OS process was
                # the top-level ``_kill_tree`` /proc walk on the aborted turn's
                # runner. That walk MISSES a grandchild whose own launcher
                # already died (reparented to init), which is exactly how a
                # nested ``BatchAgent`` subtree kept running after ``/abort``.
                # Each subagent records its pid in ``<sid>.engine_pid.json``;
                # kill that pid + its whole ``/proc`` subtree here so recursive
                # cancellation walks the parent→child registry instead of
                # depending on live process ancestry. Downward-only (we only
                # reached this child via ``parent_session_id``), so the
                # gateway / orchestrator is never in the kill set. Best-effort.
                if force_kill:
                    try:
                        from sharedagents_agent.agents import engine_pidfile

                        engine_pidfile.kill_session_engine(sd, str(child_hash))
                    except Exception:  # noqa: BLE001
                        pass
                # 2. Close the child's session log (folds it out of live).
                # FIX (2026-08-11 'waiting' wedge): the terminal write was
                # gated on ``child_ti is not None``, so a mid-tool-call child
                # whose live_turn_idx was NULL never folded — a ghost
                # ``streaming``/``prompting`` row that held its parent
                # ``waiting`` forever. Fall back to ``last_turn_idx`` (the
                # in-flight turn that never closed, mirroring
                # ``reset_stale_sessions``) and ALWAYS fold the row terminal
                # afterwards. A ``waiting`` child has no live turn of its own
                # (its turn ended cleanly) — it is released after the
                # recursion once its killed subtree shows no fresh work.
                if child_state in _TURN_STATES:
                    if child_ti is None:
                        child_ti = child_row["last_turn_idx"]
                    if child_ti is not None:
                        try:
                            w = get_session_writer(sd, str(child_hash), engine="")
                            w.turn_ended(
                                turn_idx=int(child_ti),
                                aborted=True,
                                metrics={"abort_reason": "parent_cancelled"},
                            )
                            close_session_writer(sd, str(child_hash))
                        except Exception:  # noqa: BLE001
                            pass
                    # Unconditional terminal backstop: the row must leave the
                    # live set even when no turn event could be emitted.
                    mark_session_terminal(sd, str(child_hash), state="error")
                # 3. Drop the child from the active-turn map (defensive).
                self._new_backend_active_turns.pop(str(child_hash), None)
                # 4. Recurse into grandchildren.
                await self._cancel_child_sessions(
                    str(child_hash),
                    force_kill=force_kill,
                    _depth=_depth + 1,
                )
                # 5. A ``waiting`` intermediate whose whole subtree was just
                # hard-killed must not stay ``waiting``. Descendant terminal
                # events normally cascade via ``_maybe_clear_parent_waiting``;
                # this backstop covers descendants that had no turn event at
                # all (NULL live+last idx). Fresh grandchildren (recursion
                # failed to reap them) keep the intermediate waiting.
                if child_state == "waiting":
                    try:
                        con = connect_sessions_db(db)
                        try:
                            now_ms = int(_time.time() * 1000)
                            still_busy = EventProjector()._parent_still_has_work(
                                con,
                                str(child_hash),
                                freshness_ms=now_ms - int(LIVE_FRESHNESS_SECONDS) * 1000,
                            )
                        finally:
                            con.close()
                        if not still_busy:
                            mark_session_terminal(sd, str(child_hash), state="idle")
                    except Exception:  # noqa: BLE001 — backstop must not fail
                        pass
        except Exception:  # noqa: BLE001 — never let the cascade fail the abort
            pass

    # -- prompt orchestration ----------------------------------------------

    def _log_event(self, event: str, **fields: Any) -> None:
        """Append one structured event line to the unified gateway log.

        Used for silent control-flow milestones — loop-detector hits,
        external cancels, runner aborts, permission denials, max-round
        exhaustion, worker errors. Without this, the only signal that
        something cancelled a turn is ``aborted=true`` in the session
        JSON and the user-facing ``stopReason=cancelled`` — neither
        names a cause, which is how the "I didn't cancel anything"
        2026-05-07 ticket happened.

        Routed through :func:`sharedagents_agent.tools._log.log_event` so the
        line lands in the SAME JSONL stream as every other gateway
        event — with origin tags (agent / service / session_id) plus
        the cross-process flock that the legacy raw ``open(...,"a")``
        path didn't have. (The previous
        plaintext ``EVENT {json}\\n`` line shape corrupted the unified
        JSONL contract.)

        Never raises: the logfile is best-effort observability and a
        failed write must not abort the turn it's describing.
        """
        from sharedagents_agent.tools._log import log_event as _gw_log_event

        try:
            _gw_log_event(
                event,
                agent=str(getattr(self.cfg, "name", "") or ""),
                **fields,
            )
        except Exception:  # noqa: BLE001
            # Fall back to stderr so journalctl still captures the
            # signal even if the JSONL writer is unwell.
            try:
                print(f"{event} {fields!r}", file=sys.stderr, flush=True)
            except Exception:  # noqa: BLE001
                pass
        # Also surface via the python logger so anything that wires
        # up a handler (tests, dev runs) sees the event without having
        # to tail gateway.log. ``warning`` (not info) so abort/cancel
        # signals stand out from the per-round noise.
        try:
            _LOG.warning("%s %s", event, fields)
        except Exception:  # noqa: BLE001
            pass

    def _get_new_agent(self) -> Any:
        """Return the live Agent that drives this backend's turns.

        Single-source-of-truth contract: when the gateway pre-built
        the Agent and passed it via ``agent=`` to our constructor,
        return it unchanged so :class:`AgentACPBackend` and
        :class:`CoreBackend` share ONE :class:`LocalLLMEngine`
        instance (per-session locks, promoted-tools, mode overrides
        all single-sourced). Stand-alone callers (tests, the stdio
        entry point) that didn't pre-build an Agent get a lazily-
        constructed one wired to the same ``_default_run_tool``
        dispatcher.

        The ``backend=self`` closure is preserved in both branches:
        the engine's :meth:`_invoke_runner` calls
        ``runner(session_id, call, turn, **kwargs)`` — narrower than
        the standalone ``_default_run_tool`` signature that expects
        ``backend`` first — so the wrapper has to inject ``self``.
        Without this binding every tool call 404s with
        ``_default_run_tool() missing 1 required positional argument:
        'turn'`` (observed: every TUI tool call failed).
        """
        if self._new_agent is not None:
            self._schedule_goal_recovery(self._new_agent)
            self._schedule_loop_recovery(self._new_agent)
            return self._new_agent

        backend = self  # captured by the closure below

        async def _tool_runner(
            session_id: str,
            call: Mapping[str, Any],
            turn: TurnState,
            **kwargs: Any,
        ) -> Any:
            return await _default_run_tool(backend, session_id, call, turn, **kwargs)

        if self._preset_agent is not None:
            # Re-wire the preset engine's tool_runner so tool calls
            # go through THIS CoreBackend (the one bound to this
            # connection) instead of any backend wired at factory
            # time (factory.load_agent has no CoreBackend handle).
            try:
                self._preset_agent.engine._tool_runner = _tool_runner
            except Exception:  # noqa: BLE001
                pass
            # Install our per-session outbound resolver so
            # ``Agent.resolve_permission`` can surface mode=ask
            # prompts to the connection that opened the session.
            # Without this the new path's PermissionRequested →
            # ask_user(outbound=None) flow silently auto-denies on
            # every destructive tool call.
            self._preset_agent._outbound_resolver = self._outbound_for
            self._new_agent = self._preset_agent
            self._schedule_goal_recovery(self._new_agent)
            self._schedule_loop_recovery(self._new_agent)
            return self._new_agent

        from sharedagents_agent.agents import factory

        # Direct bridge callers already own the authoritative in-memory
        # config, so route it through the single engine factory entry point
        # rather than constructing LocalLLMEngine here.
        self._new_agent = factory.load_agent(
            self.cfg.name,
            config=self.cfg,
        )
        self._new_agent.engine._tool_runner = _tool_runner
        self._new_agent._outbound_resolver = self._outbound_for
        self._schedule_goal_recovery(self._new_agent)
        self._schedule_loop_recovery(self._new_agent)
        return self._new_agent

    def _schedule_loop_recovery(self, agent: Any) -> None:
        """One-shot, fail-safe recovery for durable scheduled work."""
        from sharedagents_loop.loop_supervisor import schedule_loop_recovery

        schedule_loop_recovery(agent)

    def _schedule_goal_recovery(self, agent: Any) -> None:
        """One-shot per constructed agent: resume outer-loop supervision for
        running goals, idempotently, without ever raising into construction.

        A scheduling attempt made with NO running event loop
        cannot drive recovery — but it must NOT permanently suppress it. Clear
        the one-shot flag in that case so a later (re)construction on a live
        loop retries. A genuine per-agent failure that DID reach the loop (e.g.
        the supervisor cannot be constructed) stays flagged and is swallowed —
        retrying it every construction would never succeed and must never block
        startup.
        """
        if getattr(agent, "_goals_recovery_scheduled", False):
            return
        agent._goals_recovery_scheduled = True
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop yet: recovery is deferred, not abandoned. Reset the
            # one-shot flag so the next construction on a live loop retries.
            agent._goals_recovery_scheduled = False
            return
        try:
            loop.create_task(
                agent.goal_supervisor.recover_running_goals(),
                name=f"goal_recovery_{agent.name}",
            )
        except Exception:  # noqa: BLE001 — a schedule failure must never block construction
            pass

    async def _prompt_via_new_backend(
        self,
        session_id: str,
        prompt: list[Mapping[str, Any]],
        turn_id: str,
        sampling: dict[str, Any] | None = None,
        source: str = "user",
        actors: list[dict[str, Any]] | None = None,
        access: Any = None,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Run one turn through ``Agent.run_turn``.

        Routes through the cached Agent (see :meth:`_get_new_agent`)
        and translates the ``AgentTurnEvent`` stream to ACP
        sessionUpdate dicts via :mod:`sharedagents_agent.agents.acp_translation`.
        The Agent's :class:`~sharedagents_agent.agents.local_llm.LocalLLMEngine`
        owns the full multi-round loop — worker spawn, tool dispatch,
        permission policy, the Claude Code hook lifecycle, compaction,
        layered loop detection, distill / nudge / force-terminate
        recovery, and the three premature-turn-end variants.
        """
        from sharedagents_agent.agents import acp_translation

        agent = self._get_new_agent()

        # First-call-per-sid: attach the live session so the engine's
        # per-session state (cancelled, promoted_tools, mode override)
        # gets populated from the loaded Session record.
        if session_id not in agent.engine._cancelled:
            sess = self._session_for(session_id)
            if sess is not None:
                try:
                    agent.engine.attach_session(session_id, sess)
                except Exception:  # noqa: BLE001
                    pass

        # Flatten the ACP prompt blocks into (text, parts).
        text = _flatten_prompt(prompt)
        parts: list[dict[str, Any]] = []
        if any((b.get("type") or "") != "text" for b in prompt):
            user_content = _blocks_to_user_content(prompt)
            if isinstance(user_content, list):
                parts = [dict(p) for p in user_content]

        # Open the turn-root execution lease (§10): tool / subagent leases
        # issued during this turn nest UNDER it (real session/turn lineage,
        # not a bare SUPERVISOR parent), and the capability kill cascade is
        # scoped to this turn's subtree. contextvars propagate into the
        # engine's round loop (same task) where tools spawn.
        from sharedagents_agent import sandbox_broker as _sb

        _turn_tok, _turn_root = _sb.begin_turn(session_id, str(turn_id))

        run_kwargs: dict[str, Any] = {
            "prompt": text,
            "prompt_parts": parts or None,
            "sampling": sampling,
        }
        if source != "user":
            run_kwargs["source"] = source
        if actors:
            run_kwargs["actors"] = [dict(a) for a in actors]
        if access is not None:
            run_kwargs["access"] = access
        from sharedagents_core.attachment_context import files_from_blocks

        _files = files_from_blocks(prompt)
        if _files:
            run_kwargs["attachments"] = _files
        events = agent.run_turn(session_id, **run_kwargs)

        def _on_turn_started(ev: Any) -> None:
            # Wire the gateway's sid → turn_id map for ACP cancel
            # routing (phase-3a final #2).
            try:
                self._new_backend_active_turns[session_id] = ev.turn_id
            except Exception:  # noqa: BLE001
                pass

        try:
            async for msg in acp_translation.agent_events_to_acp(
                events,
                context_window=int(getattr(self.cfg, "upstream_max_context", 0) or 0),
                on_turn_started=_on_turn_started,
            ):
                yield msg
        except Exception as exc:  # noqa: BLE001
            self._log_event(
                "NEW_BACKEND_FAILED",
                sid=session_id[:12],
                turn_id=turn_id,
                error=str(exc)[:300],
            )
            # Route the bridge turn-crash boundary through structured
            # capture so the SessionError-hook machinery fires uniformly
            # across surfaces, not only in the TUI.
            # Best-effort — a capture failure must never mask the crash.
            try:
                from sharedagents_core import errors as _errors

                _errors.log_error(
                    "gateway:acp_bridge",
                    exc,
                    summary="ACP bridge turn failed",
                    context={"turn_id": turn_id},
                    session_id=session_id,
                )
            except Exception:  # noqa: BLE001
                pass
            yield {
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": f"[new backend failed: {exc}]",
                },
            }
            yield {"stopReason": "end_turn"}
        finally:
            # Drop the active-turn mapping so a subsequent cancel
            # doesn't fire against a finished turn_id.
            self._new_backend_active_turns.pop(session_id, None)
            # Close the turn-root lease (revokes the whole turn subtree).
            _sb.end_turn(_turn_tok, _turn_root)

    async def prompt(
        self,
        session_id: str,
        message_id: str | None,
        prompt: list[Mapping[str, Any]],
        *,
        sampling: dict[str, Any] | None = None,
        source: str = "user",
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Run one prompt turn against the bound session.

        sharedagents has ONE turn architecture: the Agent + Engine path.
        This thin wrapper resolves the session, allocates a turn id,
        and routes through :meth:`_prompt_via_new_backend` ->
        ``Agent.run_turn`` -> ``LocalLLMEngine.run_turn``, which owns
        the entire multi-round loop — worker spawn, tool dispatch,
        permission policy, the Claude Code hook lifecycle
        (UserPromptSubmit / PreToolUse / PostToolUse / Stop /
        PreCompact), proactive + reactive compaction, layered loop
        detection, distill / nudge / force-terminate recovery, and
        the three premature-turn-end variants (silent EOS,
        mid-syntax cutoff, colon cliffhanger). Yields
        ``AgentTurnEvent``s that ``acp_translation`` turns into ACP
        sessionUpdate dicts.
        """
        sess = self._session_for(session_id)
        if sess is None:
            raise RuntimeError(f"session not found: {session_id!r}")
        # Bound the per-session prompt queue. Concurrent same-session
        # prompts FIFO-serialize on the engine's per-session lock; this
        # only refuses the overflow past _MAX_INFLIGHT_PROMPTS so a
        # runaway client can't pile up unbounded queued turns. ``refusal``
        # is a standard ACP stop reason → a clean PromptResponse.
        if not self._admit_prompt(session_id):
            self._log_event(
                "PROMPT_QUEUE_FULL",
                sid=session_id[:12],
                max_inflight=self._MAX_INFLIGHT_PROMPTS,
            )
            yield {
                "stopReason": "refusal",
                "_meta": {
                    "sharedagents": {
                        "reason": "queue_full",
                        "maxInflight": self._MAX_INFLIGHT_PROMPTS,
                    }
                },
            }
            return
        try:
            turn_id = _new_turn_id()
            if not isinstance(sampling, dict):
                sampling = None
            effective_source = source if source != "user" else _prompt_source(prompt)
            from sharedagents_core.access_grants import stamp_from_blocks
            from sharedagents_session.actors import actor_from_blocks

            _actor = actor_from_blocks(prompt)
            async for ev in self._prompt_via_new_backend(
                session_id,
                prompt,
                turn_id,
                sampling=sampling,
                source=effective_source,
                actors=[_actor] if _actor is not None else None,
                access=stamp_from_blocks(prompt),
            ):
                yield ev
        finally:
            self._release_prompt(session_id)
