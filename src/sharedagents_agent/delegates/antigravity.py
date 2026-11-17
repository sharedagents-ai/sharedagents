"""Antigravity CLI delegate runner."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import signal
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sharedagents_agent import sandbox_broker
from sharedagents_core.cwdutil import safe_getcwd

from .base import (
    _ANTIGRAVITY_MODE_MAP,
    _REASONING_BLOCK_TYPES,
    _REASONING_DELTA_TYPES,
    CliAnswerSink,
    ReasoningTextRelay,
    _delegate_state_path,
    _format_cli_tool_use,
    _load_tool_utils,
    _log_delegate_error,
    _read_delegate_state,
    _reasoning_signal_chars,
    _reasoning_text_chars,
    _reasoning_tokens_from_usage,
    _resolve_delegate_cwd,
    _update_delegate_state,
)

if TYPE_CHECKING:
    # Bound into this module at call time by ``_load_tool_utils(globals())``;
    # a top-level import would close the tools <-> delegates import cycle.
    from sharedagents_agent.tools.impl._utils import (
        ToolResult,
        _antigravity_idle_timeout_s,
        _antigravity_print_timeout_argv,
        _antigravity_print_timeout_s,
        _build_delegate_spawn_env,
        _delegate_first_token_timeout_s,
        _err,
        _feed_stdin,
        _kill_group,
        _kill_tree,
        _resolve_cli_executable,
        _signal_name,
        _truncate,
    )
from .base import _CLI_KNOB_OFF, _env_cli_choice


def _antigravity_state_path(
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> Path | None:
    """Resolve the per-sharedagents-session state file for ``delegate_antigravity``.

    Mirrors :func:`_claude_state_path` — engine-pipeline callers pass
    ``sa_sid`` / ``sa_sdir`` explicitly so concurrent CLI turns can't
    race on ``os.environ``; legacy direct callers fall back to env.
    """
    return _delegate_state_path("antigravity", sa_sid, sa_sdir)


def _read_antigravity_session_id(
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> str | None:
    sid = _read_delegate_state("antigravity", sa_sid, sa_sdir).get("antigravity_session_id")
    return sid if isinstance(sid, str) and sid else None


def _write_antigravity_session_id(
    antigravity_sid: str,
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> None:
    _update_delegate_state(
        "antigravity",
        {"antigravity_session_id": antigravity_sid},
        sa_sid,
        sa_sdir,
    )


def _write_antigravity_actual_model(
    model: str,
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> None:
    """Persist the best-effort native model in the delegate sidecar."""
    _update_delegate_state(
        "antigravity",
        {"actual_model": model},
        sa_sid,
        sa_sdir,
    )


# Token buckets of the CLI's conversation-lifetime ``result.usage`` total.
_AGY_TOTAL_FIELDS = (
    "input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
    "thinking_tokens",
)


def _read_antigravity_usage_total(
    conversation_id: str,
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> dict[str, int] | None:
    """Last ``result.usage`` total recorded for ``conversation_id``, if known."""
    total = _read_delegate_state("antigravity", sa_sid, sa_sdir).get("usage_total")
    if not isinstance(total, dict) or total.get("conversation_id") != conversation_id:
        return None
    out: dict[str, int] = {}
    for key in _AGY_TOTAL_FIELDS:
        try:
            out[key] = max(0, int(total.get(key) or 0))
        except (TypeError, ValueError):
            return None
    return out


def _write_antigravity_usage_total(
    conversation_id: str,
    total: dict[str, int],
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> None:
    _update_delegate_state(
        "antigravity",
        {
            "usage_total": {
                "conversation_id": conversation_id,
                **{key: int(total.get(key) or 0) for key in _AGY_TOTAL_FIELDS},
            },
        },
        sa_sid,
        sa_sdir,
    )


def _antigravity_usage_growth(
    total: dict[str, int],
    previous: dict[str, int],
) -> dict[str, int] | None:
    """``total - previous`` per bucket, in the normalized usage shape."""
    grown = {key: max(0, int(total.get(key) or 0) - int(previous.get(key) or 0)) for key in _AGY_TOTAL_FIELDS}
    if not any(grown.values()):
        return None
    return {
        "input_tokens": grown["input_tokens"],
        "cache_read_input_tokens": grown["cache_read_input_tokens"],
        "cache_creation_input_tokens": 0,
        "output_tokens": grown["output_tokens"],
        "thinking_tokens": grown["thinking_tokens"],
        "reasoning_tokens": grown["thinking_tokens"],
    }


def _antigravity_error_text(payload: dict[str, Any]) -> str:
    """Best-effort message from a result or error-step payload."""
    for key in ("error", "message", "text", "content"):
        val = payload.get(key)
        if isinstance(val, dict):
            val = val.get("message") or val.get("type")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _agy_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise ValueError("truncated protobuf varint")


def _agy_fields(data: bytes):
    offset = 0
    while offset < len(data):
        tag, offset = _agy_varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if number <= 0:
            raise ValueError("invalid protobuf field")
        if wire == 0:
            value, offset = _agy_varint(data, offset)
        elif wire == 1:
            value = data[offset : offset + 8]
            offset += 8
        elif wire == 2:
            length, offset = _agy_varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        elif wire == 5:
            value = data[offset : offset + 4]
            offset += 4
        else:
            raise ValueError("unsupported protobuf wire type")
        if offset > len(data):
            raise ValueError("truncated protobuf field")
        yield number, wire, value


def _antigravity_conversation_db(conversation_id: str | None) -> Path | None:
    """Path of the CLI's conversation DB, or ``None`` for an unusable id."""
    if not isinstance(conversation_id, str) or not conversation_id:
        return None
    # UUID-shaped names prevent path traversal and match the CLI's DB layout.
    import re

    if not re.fullmatch(r"[0-9a-fA-F-]{36}", conversation_id):
        return None
    root = Path(os.environ.get("ANTIGRAVITY_HOME") or Path.home() / ".gemini" / "antigravity-cli").expanduser()
    return root / "conversations" / f"{conversation_id}.db"


def _antigravity_conversation_model(conversation_id: str | None) -> str:
    """Read the last native model from an Antigravity conversation DB.

    ``gen_metadata.data`` is a protobuf envelope whose field 1 contains the
    generation message; field 19 of that message is the backend model name.
    Reads are best-effort because the CLI may create the DB asynchronously.
    """
    path = _antigravity_conversation_db(conversation_id)
    if path is None:
        return ""
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0) as con:
            rows = con.execute("select data from gen_metadata order by idx desc")
            for (data,) in rows:
                try:
                    nested = next(value for number, wire, value in _agy_fields(data) if number == 1 and wire == 2)
                    for number, wire, value in _agy_fields(nested):
                        if number != 19 or wire != 2:
                            continue
                        model = value.decode("utf-8").strip()
                        if model:
                            return model
                except (UnicodeDecodeError, ValueError, TypeError):
                    continue
    except (OSError, sqlite3.Error, ValueError):
        return ""
    return ""


# A finished answer in the CLI's ``steps`` table, as stored by CLI 1.4.12
# (identical across 500 local conversation DBs): ``step_type`` 15,
# ``status`` 3 (DONE), ``step_format`` 0. Its payload restates the type in
# field 1, carries the step's own ``{2: idx, 4: conversation id}`` in field
# 5.20, and the planner response in field 20: text in 1 (mirrored in 8), tool
# calls in 7. A user prompt is ``step_type`` 14.
_AGY_STEP_USER_INPUT = 14
_AGY_STEP_RESPONSE = 15
_AGY_STEP_DONE = 3
_AGY_STEP_FORMAT = 0
# A shorter overlap could match a common opening phrase by chance.
_AGY_TAIL_MIN_OVERLAP = 24
# A stream that lost more than this lost a whole answer, not a tail.
_AGY_TAIL_MAX_CHARS = 4096


def _antigravity_tail_reconcile_enabled() -> bool:
    """``SHAREDAGENTS_ANTIGRAVITY_TAIL_RECONCILE=off`` disables tail recovery."""
    return (
        _env_cli_choice(
            "SHAREDAGENTS_ANTIGRAVITY_TAIL_RECONCILE",
            "auto",
            frozenset({"auto", "on"}),
            _CLI_KNOB_OFF,
        )
        is not None
    )


def _antigravity_conversation_last_idx(conversation_id: str | None) -> int | None:
    """Highest step idx already in the conversation DB (-1 when empty).

    Taken before a resumed prompt starts, so tail recovery only considers
    steps this prompt produced. ``None`` (no readable DB) disables recovery.
    """
    path = _antigravity_conversation_db(conversation_id)
    if path is None or not path.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
        try:
            (top,) = con.execute("select max(idx) from steps").fetchone()
        finally:
            con.close()
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return None
    return -1 if top is None else int(top)


def _agy_message(data: bytes) -> dict[int, list[tuple[int, Any]]]:
    out: dict[int, list[tuple[int, Any]]] = {}
    for number, wire, value in _agy_fields(data):
        out.setdefault(number, []).append((wire, value))
    return out


def _agy_one(msg: dict[int, list[tuple[int, Any]]], number: int, wire: int) -> Any:
    """The single value of field ``number``; anything else is unknown schema."""
    values = msg.get(number) or []
    if len(values) != 1 or values[0][0] != wire:
        raise ValueError(f"unexpected protobuf field {number}")
    return values[0][1]


def _antigravity_turn_answer(
    conversation_id: str | None,
    after_idx: int,
) -> tuple[int, str] | None:
    """The final answer step this prompt produced, as ``(idx, text)``.

    Only a DONE, text-only response that is the conversation's newest step,
    names this conversation and itself, and follows exactly one user prompt
    after ``after_idx`` qualifies. Anything else — an older answer, another
    turn, a tool-call step, an unknown payload shape — returns ``None``.
    """
    path = _antigravity_conversation_db(conversation_id)
    if path is None or not path.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0)
        try:
            # One read transaction: the count and the row describe one state.
            con.execute("begin")
            row = con.execute(
                "select idx, step_type, status, step_format, step_payload "
                "from steps where idx > ? order by idx desc limit 1",
                (after_idx,),
            ).fetchone()
            if row is None:
                return None
            idx, step_type, status, step_format, payload = row
            (prompts,) = con.execute(
                "select count(*) from steps where idx > ? and idx < ? and step_type = ?",
                (after_idx, idx, _AGY_STEP_USER_INPUT),
            ).fetchone()
        finally:
            con.close()
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return None
    if (
        step_type != _AGY_STEP_RESPONSE
        or status != _AGY_STEP_DONE
        or step_format != _AGY_STEP_FORMAT
        or prompts != 1
        or not isinstance(payload, bytes)
    ):
        return None
    try:
        top = _agy_message(payload)
        if _agy_one(top, 1, 0) != _AGY_STEP_RESPONSE:
            return None
        meta = _agy_message(_agy_one(_agy_message(_agy_one(top, 5, 2)), 20, 2))
        if _agy_one(meta, 2, 0) != idx or _agy_one(meta, 4, 2).decode("utf-8") != conversation_id:
            return None
        answer = _agy_message(_agy_one(top, 20, 2))
        if 7 in answer:
            return None
        text = _agy_one(answer, 1, 2).decode("utf-8")
        if 8 in answer and _agy_one(answer, 8, 2).decode("utf-8") != text:
            return None
    except (UnicodeDecodeError, ValueError, TypeError):
        return None
    return (idx, text) if text.strip() else None


async def _antigravity_conversation_last_message(
    conversation_id: str | None,
    after_idx: int | None,
    retries: int = 5,
    retry_delay_s: float = 0.05,
) -> str:
    """This prompt's final answer as the CLI stored it, or ``""``.

    The CLI may finish writing the DB just after the stream, so a missing
    answer is retried a bounded number of times. A found answer is read
    once more and must be unchanged; a moving DB is not trusted.
    Best-effort: it never raises, so it can never fail the turn.
    """
    if after_idx is None:
        return ""
    path = _antigravity_conversation_db(conversation_id)
    try:
        for attempt in range(max(1, retries)):
            found = _antigravity_turn_answer(conversation_id, after_idx)
            if found is not None:
                again = _antigravity_turn_answer(conversation_id, after_idx)
                return found[1] if again == found else ""
            if path is None or not path.is_file() or attempt + 1 >= retries:
                return ""
            await asyncio.sleep(retry_delay_s)
    except Exception:  # noqa: BLE001
        return ""
    return ""


def _antigravity_reconcile_tail(body_text: str, db_last: str) -> str:
    """The part of ``db_last`` the stream never delivered, or ``""``.

    CLI 1.4.12 can drop the tail of the in-flight answer from stream-json
    (and from ``result.response``) when the print-mode wall interrupts the
    run, while its conversation DB keeps the whole message. Recovery appends
    only the missing suffix, and only when the streamed body ends with a
    prefix of at least ``_AGY_TAIL_MIN_OVERLAP`` characters of the stored
    answer, so a complete or unrelated body is never restated.
    """
    db_last = db_last.rstrip()
    if not body_text or not db_last or body_text.endswith(db_last):
        return ""
    total = len(db_last)
    first = max(1, total - len(body_text))
    last = min(total - _AGY_TAIL_MIN_OVERLAP, _AGY_TAIL_MAX_CHARS)
    for missing in range(first, last + 1):
        if body_text.endswith(db_last[: total - missing]):
            return db_last[total - missing :]
    return ""


def _antigravity_looks_cut(text: str) -> bool:
    """A body ending mid-word; used only to decide what to log."""
    return bool(text) and text[-1].isalnum()


def _format_antigravity_tool_use(name: str, params: Any) -> str:
    """Render an in-band marker for a Antigravity ``tool_use`` event.

    Mirrors :func:`_format_claude_tool_use` so the surrounding TUI /
    Telegram renderers don't have to special-case Antigravity's payload —
    same ``🔧 <name> <json>`` shape, same compact JSON args, same
    160-char clamp. Stable, no timestamps, no ids — cache-friendly
    when the result is replayed via ``_build_messages``.
    """
    return _format_cli_tool_use(name, params)


# CLI 1.1.x (verified 1.1.21) stream-json envelope. Older fixtures still
# emit ``{"type": ...}``; live ``antigravity`` now emits ``{"event": ...}``.
_ANTIGRAVITY_STREAM_TYPES = (
    frozenset(
        {
            "init",
            "message",
            "tool_use",
            "tool_result",
            "error",
            "result",
            "step_update",
        }
    )
    | _REASONING_BLOCK_TYPES
    | _REASONING_DELTA_TYPES
)


def _antigravity_event_type(event: dict[str, Any]) -> str:
    """Return the stream-json label for one Antigravity NDJSON line.

    Prefer ``type`` when present so CLI ≤1.0 fixtures keep working; fall
    back to the 1.1.x ``event`` field.
    """
    raw = event.get("type")
    if isinstance(raw, str) and raw:
        return raw
    raw = event.get("event")
    return raw if isinstance(raw, str) else ""


def _antigravity_session_id(event: dict[str, Any]) -> str | None:
    """Pull the resume conversation id from a 1.0 or 1.1.x envelope."""
    for key in ("session_id", "conversation_id"):
        val = event.get(key)
        if isinstance(val, str) and val:
            return val
    for nested_key in ("init", "result", "step_update"):
        inner = event.get(nested_key)
        if not isinstance(inner, dict):
            continue
        for key in ("session_id", "conversation_id"):
            val = inner.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def _antigravity_result_payload(event: dict[str, Any]) -> dict[str, Any]:
    inner = event.get("result")
    return inner if isinstance(inner, dict) else event


def _antigravity_step_payload(event: dict[str, Any]) -> dict[str, Any]:
    inner = event.get("step_update")
    return inner if isinstance(inner, dict) else event


_THINKING_TEXT_KEYS = (
    "text",
    "thinking",
    "reasoning",
    "thought",
    "summary",
    "delta",
    "text_delta",
    "thought_delta",
    "thinking_delta",
)
_THINKING_STEP_TYPES = frozenset({"thought", "thinking", "reasoning"})


def _antigravity_thinking_text(payload: Any) -> str:
    """Return visible thinking text from one Antigravity event or step."""
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""
    parts: list[str] = []
    for key in _THINKING_TEXT_KEYS:
        val = payload.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item:
                    parts.append(item)
                elif isinstance(item, dict):
                    nested = (
                        item.get("text") or item.get("thinking") or item.get("reasoning") or item.get("thought") or ""
                    )
                    if isinstance(nested, str) and nested:
                        parts.append(nested)
    return "".join(parts)


def _antigravity_usage_from_event(event: dict[str, Any]) -> dict[str, int] | None:
    """Normalize Antigravity/Gemini usage payload into canonical token counters.

    The CLI (1.2.x stream-json) reports ``input_tokens`` as the UNCACHED input
    and ``cache_read_tokens`` beside it — cache reads routinely exceed the
    input. Gemini API payloads instead count cached tokens inside
    ``prompt_token_count``, so only that shape subtracts them.
    """
    raw = event.get("usage") if isinstance(event.get("usage"), dict) else event
    if not isinstance(raw, dict):
        return None
    inp = raw.get("input_tokens") or raw.get("prompt_tokens") or raw.get("prompt_token_count") or 0
    cli_cache_shape = raw.get("cache_read_tokens") is not None
    cached = (
        raw.get("cache_read_tokens")
        if cli_cache_shape
        else (raw.get("cached_content_token_count") or raw.get("cache_read_input_tokens") or raw.get("cached_tokens"))
    ) or 0
    out = raw.get("output_tokens") or raw.get("completion_tokens") or raw.get("candidates_token_count") or 0
    think = _reasoning_tokens_from_usage(raw)
    try:
        inp, cached, out = int(inp), int(cached), int(out)
    except (TypeError, ValueError):
        return None
    if inp <= 0 and out <= 0 and think <= 0 and cached <= 0:
        return None
    return {
        "input_tokens": max(0, inp if cli_cache_shape else inp - cached),
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
        "output_tokens": out,
        "thinking_tokens": think,
        "reasoning_tokens": think,
    }


def _antigravity_tool_output(step: dict[str, Any]) -> tuple[str, bool]:
    """Return ``(output_text, is_error)`` from a 1.1.x tool ``step_update``."""
    info = step.get("tool_info") if isinstance(step.get("tool_info"), dict) else {}
    err = info.get("error") if info.get("error") is not None else step.get("error")
    if err:
        msg = str(err.get("message") or err.get("type") or err) if isinstance(err, dict) else str(err)
        return msg, True
    out = info.get("output")
    if isinstance(out, str):
        return out, False
    if out is None:
        return "", False
    return str(out), False


def _format_antigravity_tool_result(event: dict) -> str:
    """Render an in-band marker for a Antigravity ``tool_result`` event.

    Pairs with :func:`_format_antigravity_tool_use`. Antigravity's stream-json
    ``tool_result`` carries ``status`` (``success`` / ``error``) and
    ``output`` (the textual result). We surface a 1-line summary plus
    char count, identical to the Claude variant.
    """
    is_err = str(event.get("status") or "").lower() not in ("success", "ok")
    raw = event.get("output")
    if isinstance(raw, list):
        parts: list[str] = []
        for b in raw:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text") or "")
            elif isinstance(b, str):
                parts.append(b)
        text = "".join(parts)
    elif isinstance(raw, str):
        text = raw
    else:
        text = "" if raw is None else str(raw)
    n = len(text)
    head = next((line for line in text.splitlines() if line.strip()), "")
    summary = head.strip()[:140]
    arrow = "✗" if is_err else "↳"
    if summary:
        return f"{arrow} {summary} ({n} chars)\n"
    return f"{arrow} ({n} chars)\n"


async def run_delegate_antigravity_async(
    args: dict[str, Any],
    *,
    on_delta: Callable[[str], None] | None = None,
    on_event: Callable[[dict], None] | None = None,
    on_spawn: Callable[[Any], None] | None = None,
) -> ToolResult:
    """Spawn ``antigravity -p '' ...`` and return its reply as the tool result.

    .. deprecated:: refactor-phase-4
       Legacy delegate-as-tool entry point. The new
       :class:`sharedagents_agent.agents.antigravity_engine.AntigravityCliEngine`
       reuses this via :meth:`delegate_runner`; the function stays
       until Phase 7's legacy sweep.


    Mirrors :func:`run_delegate_claude_async` against the Google
    Antigravity CLI (https://github.com/google-antigravity/gemini-cli). Streams
    ``--output-format stream-json`` (newline-delimited JSON events).
    CLI ≤1.0 used ``type``: ``init`` / ``message`` / ``tool_use`` /
    ``tool_result`` / ``result`` / ``error``. CLI 1.1.x (verified
    1.1.21) uses ``event``: ``init`` / ``step_update`` / ``result``,
    with assistant text on ``step_update.text_delta`` and tools as
    ``step_type=tool`` ACTIVE/DONE. Unrecognized 1.1.x envelopes are
    swallowed — never dumped as chat text. Forwards each text delta
    plus a compact in-band marker for every tool use the way the
    Claude runner does — the slash-command bridge (``/antigravity``)
    and the autonomous-tool path (``tool_delta`` SSE) consume the
    same ``on_delta`` callback, so live progress shows up everywhere
    without per-frontend code.

    Per-sharedagents-session state: reads ``antigravity_session_id`` from
    ``<sessions_dir>/<sid>.delegate_antigravity.json`` and passes
    ``--resume <id>`` so a follow-up call continues the same
    Antigravity conversation. The new id is written back on success.

    Killability: a SIGTERM landing on this coroutine (via the
    parent ``tool_runner``'s signal handler) propagates through
    CancelledError; the ``finally`` block kills the antigravity
    subprocess so it doesn't keep running orphaned.

    Auth: requires ``ANTIGRAVITY_API_KEY`` (or the documented
    ``GOOGLE_GENAI_USE_VERTEXAI`` / ``GOOGLE_GENAI_USE_GCA``
    alternatives) in the runner's environment — operators put it
    in ``<agent>/.env`` the same way they put ``OPENROUTER_API_KEY``.
    """
    _load_tool_utils(globals())
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _err("prompt required")

    # Engine-pipeline session identity + mode — see :func:`run_delegate_claude_async`
    # for rationale. Concurrent CLI turns on the same agent must not
    # race on process-wide ``os.environ``.
    sa_sid_arg = args.get("_sharedagents_session_id")
    sa_sid: str | None = sa_sid_arg if isinstance(sa_sid_arg, str) and sa_sid_arg else None
    sa_sdir_arg = args.get("_sharedagents_sessions_dir")
    sa_sdir: str | None = sa_sdir_arg if isinstance(sa_sdir_arg, str) and sa_sdir_arg else None
    sa_perm_arg = args.get("_sharedagents_permission_mode")
    sa_perm: str = sa_perm_arg if isinstance(sa_perm_arg, str) and sa_perm_arg else ""

    def _log_error(reason: str, **fields: Any) -> None:
        _log_delegate_error("antigravity", reason, sharedagents_session_id=sa_sid, **fields)

    mode_raw = args.get("mode")
    if isinstance(mode_raw, str) and mode_raw in _ANTIGRAVITY_MODE_MAP:
        antigravity_mode = _ANTIGRAVITY_MODE_MAP[mode_raw]
    else:
        sa_mode = sa_perm or os.environ.get("SHAREDAGENTS_PERMISSION_MODE", "")
        antigravity_mode = _ANTIGRAVITY_MODE_MAP.get(sa_mode, "default")

    spawn_cwd, cwd_err = _resolve_delegate_cwd(args)
    if cwd_err is not None:
        _log_error(
            "bad_cwd",
            prompt=prompt,
            cwd=safe_getcwd(),
            mode=antigravity_mode,
            message=cwd_err,
        )
        return _err(cwd_err)

    force_fresh = bool(args.get("_sharedagents_antigravity_no_resume"))
    # Resume-id override (new Agent/Engine pipeline). When the caller
    # is :class:`ExternalCliEngine`, the engine manages its own
    # sidecar and passes the resume id explicitly via
    # ``_force_resume_id``.
    forced_resume = args.get("_force_resume_id")
    if isinstance(forced_resume, str) and forced_resume:
        resume_id = None if force_fresh else forced_resume
    else:
        resume_id = None if force_fresh else _read_antigravity_session_id(sa_sid, sa_sdir)

    # System-prompt priming for the new Agent/Engine pipeline. Antigravity
    # CLI exposes a ``--system-prompt`` flag when available; on older
    # builds we fall back to a leading-user-message injection. Same
    # rationale as the codex/claude branches.
    appended_sys = args.get("_append_system_prompt")
    appended_sys_text = appended_sys.strip() if isinstance(appended_sys, str) and appended_sys.strip() else ""

    # Optional model override. Antigravity CLI defaults to
    # ``gemini-2.0-flash-exp`` (or whatever the user picked in
    # ``~/.antigravity/settings.json``); accept a tool arg so the local
    # model can dispatch to a stronger Antigravity model when it explicitly
    # wants to (``delegate_antigravity(prompt=..., model="gemini-2.5-pro")``).
    #
    # MODEL GATE (owner round-2 scope change): direct callers bypass the
    # engine gate, so the SAME fail-fast gate applies here — the literal
    # engine label "antigravity" is never a wire model id and is refused
    # BEFORE the spawn. When an operator catalogue
    # (``external_models.antigravity``) exists the gate fails closed on
    # ids outside it; without one the label invariant still holds and the
    # CLI remains the authority (documented limitation). No override on a
    # fresh call resolves the engine's effective default (validated agent
    # ``upstream_model``, else ``""`` → no flag → the CLI's own default:
    # antigravity has no safe sharedagents pin, its own settings file governs).
    from sharedagents_agent.discovery._common import (
        external_cli_effective_default,
        external_cli_model_refusal,
        validate_external_cli_model,
    )

    model_override = args.get("model")
    if isinstance(model_override, str) and model_override.strip():
        _canonical_model = validate_external_cli_model(
            "antigravity_cli",
            model_override,
        )
        if _canonical_model is None:
            return ToolResult(
                content=external_cli_model_refusal(
                    model_override,
                    "antigravity_cli",
                    source="explicit",
                ),
                ok=False,
            )
        model_override = _canonical_model
    elif not resume_id:
        model_override = external_cli_effective_default("antigravity_cli")
    if not isinstance(model_override, str) or not model_override.strip():
        model_override = None

    # Prompt is piped on stdin (antigravity ``-p`` "Appended to input on
    # stdin (if any)" per --help). Empty ``-p ""`` is the
    # non-interactive trigger. Same trick we use for claude — argv
    # would hit Linux ARG_MAX on long context dumps.
    import uuid

    sid_slug = sa_sid if sa_sid else str(uuid.uuid4())
    log_path = Path(f"/tmp/agy_log_{sid_slug}.log")

    cmd = [
        _resolve_cli_executable("antigravity"),
        "-p",
        "",
        # CLI default is ``text`` (one dump at process exit). Without
        # stream-json the runner never sees ``tool_use`` / ``message``
        # deltas, so the web UI shows only "prefill / still working"
        # until the final answer. Mirrors the claude argv.
        "--output-format",
        "stream-json",
        "--log-file",
        str(log_path),
        *_antigravity_print_timeout_argv(),
    ]
    print_timeout_s = _antigravity_print_timeout_s()
    from sharedagents_agent.continuation import antigravity_time_budget_notice

    budget_note = antigravity_time_budget_notice(print_timeout_s)
    if antigravity_mode == "yolo":
        cmd.append("--dangerously-skip-permissions")
    elif antigravity_mode == "plan":
        cmd += ["--mode", "plan", "--sandbox"]
    elif antigravity_mode == "accept-edits":
        cmd += ["--mode", "accept-edits"]

    if model_override:
        # --model selects this invocation's model, including when resuming
        # a conversation. Omitting it here silently discards picker changes.
        cmd += ["--model", model_override]
    if resume_id:
        cmd += ["--conversation", resume_id]

    if appended_sys_text:
        # Inject as a leading-user-message in the stdin stream so it
        # lands in the bound Antigravity session's state on the first call
        # and rides on every ``--resume`` after. (The antigravity CLI's
        # native --system-prompt flag isn't universally present
        # across versions; the leading-user-message form works
        # against every antigravity build.)
        prompt = f"<system>\n{appended_sys_text}\n</system>\n\n---\n\n{prompt}"
    # Every spawn, including resume / timeout-continue, carries the
    # print-mode wall so the model chunks work instead of starting a
    # job it cannot finish in this turn.
    if budget_note not in prompt:
        prompt = f"{budget_note}\n\n{prompt}"

    # antigravity (1.1.x) requires the prompt as the ``-p`` ARGUMENT and ignores
    # stdin entirely — a piped prompt with ``-p ""`` fails "empty prompt", and a
    # non-empty ``-p`` value with content on stdin drops the stdin. Put the
    # finalized prompt on argv (verified against antigravity 1.1.4). Long context
    # dumps risk ARG_MAX, but the CLI exposes no stdin/file input path.
    try:
        cmd[cmd.index("-p") + 1] = prompt
    except ValueError:  # pragma: no cover - cmd always carries -p
        cmd += ["-p", prompt]

    # Tail recovery may only trust DB steps this prompt writes, so note where
    # the resumed conversation ends now. A new conversation has no history.
    tail_after_idx: int | None = None
    if _antigravity_tail_reconcile_enabled():
        tail_after_idx = _antigravity_conversation_last_idx(resume_id) if resume_id else -1

    # State variables (initialize before log watcher to prevent race conditions on session ID assignment)
    final_session_id: str | None = resume_id
    text_buf: list[str] = []
    answer_sink = CliAnswerSink()
    reasoning_chars = 0
    # One trace per thinking step: deltas stream, the finished step restates
    # only what the deltas have not already produced.
    thinking_relay = ReasoningTextRelay()
    session_id_emitted = False
    is_error = False
    error_event: dict[str, Any] = {}
    seen_tool_uses: set[str] = set()
    seen_tool_results: set[str] = set()
    # The final ``result`` describes the whole CONVERSATION, not this run: its
    # status/error replay any earlier error step on every resume (verified on
    # CLI 1.2.9) and its usage is the lifetime total. So this run
    # is judged by its own stream — an ``error_message`` step with no answer
    # after it — and billed from its per-call ``step_update`` usage.
    result_error: dict[str, Any] | None = None
    # ``result.finished_reason`` of this run ("print_timeout" on a wall cut).
    result_finished_reason = ""
    step_usage_seen = False
    event_seq = 0
    last_error_seq: int | None = None
    last_answer_seq: int | None = None
    run_error_text = ""
    legacy_stream = False

    def _note_reasoning(value: Any) -> None:
        """Record the size of reasoning; thinking text is streamed separately."""
        nonlocal reasoning_chars
        if isinstance(value, int) and value > 0:
            reasoning_chars += value

    def _emit_thinking(
        text: str,
        *,
        identity: str = "",
        consolidated: bool = False,
    ) -> None:
        """Forward Antigravity thinking onto the official thinking channel.

        A step streams ``thought_delta`` fragments and then restates the whole
        thought when it finishes; the relay forwards each trace once.
        """
        if not isinstance(text, str) or not text:
            return
        _note_reasoning(len(text))
        key = identity or "step"
        kept = thinking_relay.consolidated(key, text) if consolidated else thinking_relay.delta(key, text)
        if not kept or on_event is None:
            return
        try:
            on_event({"kind": "thinking", "text": kept})
        except Exception:  # noqa: BLE001
            pass

    def _emit_step_thinking(step: dict[str, Any]) -> None:
        """Relay one thinking step: deltas first, restatement tail-only."""
        identity = f"step-{step.get('step_index', '')}"
        for key in ("thought_delta", "thinking_delta", "text_delta", "delta"):
            if key in step:
                frag = _antigravity_thinking_text({key: step[key]})
                if frag:
                    _emit_thinking(frag, identity=identity)
        for key in ("thinking", "thought", "reasoning", "summary", "text"):
            if key in step:
                full = _antigravity_thinking_text({key: step[key]})
                if full:
                    _emit_thinking(
                        full,
                        identity=identity,
                        consolidated=True,
                    )
                    break

    # Set up background log watcher for session ID discovery
    watcher_stop = asyncio.Event()
    watcher_task = None

    def _on_session_id(sid: str):
        nonlocal final_session_id, session_id_emitted
        final_session_id = sid
        if not session_id_emitted and on_event is not None:
            try:
                on_event({"kind": "session_id", "id": sid})
            except Exception:  # noqa: BLE001
                pass
            session_id_emitted = True

    async def _watch_log(path: Path, stop_evt: asyncio.Event):
        try:
            while not stop_evt.is_set():
                if path.exists():
                    break
                await asyncio.sleep(0.05)
            if not path.exists():
                return

            import re

            conv_re = re.compile(r"Created conversation ([a-f0-9\-]{36})|Print mode: conversation=([a-f0-9\-]{36})")
            with open(path, encoding="utf-8", errors="replace") as f:
                while not stop_evt.is_set():
                    line = f.readline()
                    if not line:
                        await asyncio.sleep(0.05)
                        continue
                    match = conv_re.search(line)
                    if match:
                        sid = match.group(1) or match.group(2)
                        if sid:
                            _on_session_id(sid)
                            break
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            pass

    watcher_task = asyncio.create_task(_watch_log(log_path, watcher_stop))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
            start_new_session=True,
            cwd=spawn_cwd,
            env=_build_delegate_spawn_env(args),
        )
        # Lease the spawned *_cli agent's process group (§10.4): the broker-
        # aware _kill_group then capability-validates its cancel; released in
        # this runner's ``finally``.
        _lease_id = sandbox_broker.lease_process(pgid=proc.pid, kind="cli")
        if on_spawn is not None:
            try:
                on_spawn(proc)
            except Exception:  # noqa: BLE001
                pass
    except OSError as spawn_exc:
        # FileNotFoundError → CLI not installed. E2BIG (errno 7, "Argument
        # list too long") → the prompt-on-argv exceeded ARG_MAX; antigravity
        # exposes no stdin/file input path, so this prompt cannot be delegated
        # to it. Both must clean up the already-created watcher task (else it
        # leaks) rather than propagate an unhandled OSError.
        watcher_stop.set()
        if watcher_task is not None:
            try:
                watcher_task.cancel()
            except Exception:  # noqa: BLE001
                pass
        try:
            if log_path.exists():
                log_path.unlink()
        except Exception:  # noqa: BLE001
            pass

        if isinstance(spawn_exc, FileNotFoundError):
            _log_error(
                "spawn_failed",
                prompt=prompt,
                cwd=spawn_cwd,
                mode=antigravity_mode,
                resume_id=resume_id,
                message="`antigravity` CLI not on PATH",
            )
            return _err("`antigravity` CLI not on PATH. Install: `npm install -g @google/antigravity-cli`")
        if getattr(spawn_exc, "errno", None) == errno.E2BIG:
            _log_error(
                "spawn_failed",
                prompt=prompt,
                cwd=spawn_cwd,
                mode=antigravity_mode,
                resume_id=resume_id,
                message="prompt exceeds ARG_MAX (antigravity has no stdin/file input)",
            )
            return _err(
                f"prompt too large for the antigravity CLI "
                f"({len(prompt.encode('utf-8', 'ignore'))} bytes exceeds the OS "
                "argument-length limit, and antigravity accepts the prompt only "
                "on argv). Route this turn to a delegate that reads stdin "
                "(claude/codex) or shorten the context."
            )
        # Any other spawn OSError: clean up (done above) and surface it.
        _log_error(
            "spawn_failed",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=antigravity_mode,
            resume_id=resume_id,
            message=f"antigravity spawn failed: {spawn_exc!r}",
        )
        return _err(f"antigravity spawn failed: {spawn_exc}")

    # antigravity takes the prompt on argv (above) and ignores stdin; feed it
    # empty so the process does not block waiting on input.
    stdin_task = asyncio.create_task(_feed_stdin(proc, b""))

    def _emit_answer(kept: str) -> None:
        nonlocal last_answer_seq
        if not kept:
            return
        if kept.strip():
            last_answer_seq = event_seq
        text_buf.append(kept)
        if on_delta is not None:
            try:
                on_delta(kept)
            except Exception:  # noqa: BLE001
                pass

    def _push(chunk: str) -> None:
        _emit_answer(answer_sink.push_delta(chunk))

    def _push_consolidated(text: str) -> None:
        nonlocal last_answer_seq
        # A new answer step can repeat text from before a recovered error.
        # Display deduplication must not hide that post-error answer from the
        # run-status check. Result replays use _push_final instead.
        if text.strip():
            last_answer_seq = event_seq
        _emit_answer(answer_sink.push_consolidated(text))

    def _push_final(text: str) -> None:
        nonlocal last_answer_seq
        # The result can replay a partial answer from before the failure.
        answer_seq = last_answer_seq
        _emit_answer(answer_sink.push_final(text))
        last_answer_seq = answer_seq

    def _push_marker(chunk: str) -> None:
        """Tool marker push. Keeps the inline ``🔧 …`` / ``↳ …``
        text in the persisted body but skips ``on_delta`` when
        ``on_event`` is wired — the structured event already drives
        a native widget on the frontend, so we don't want the inline
        text to render alongside it.
        """
        if not chunk:
            return
        text_buf.append(chunk)
        if on_event is None and on_delta is not None:
            try:
                on_delta(chunk)
            except Exception:  # noqa: BLE001
                pass

    # Helper async generator to stream stdout chunks (or buffered JSON protocol lines) with timeouts.
    async def read_stream(stdout: asyncio.StreamReader):
        buffer = bytearray()
        is_json_mode = False
        first_timeout = _delegate_first_token_timeout_s()
        idle_timeout = _antigravity_idle_timeout_s()

        try:
            chunk = await asyncio.wait_for(stdout.read(4096), timeout=first_timeout)
        except asyncio.TimeoutError:
            raise

        if not chunk:
            return

        if chunk.lstrip().startswith(b"{"):
            is_json_mode = True

        if is_json_mode:
            buffer.extend(chunk)
            while b"\n" in buffer:
                line_part, _, rest = buffer.partition(b"\n")
                yield "json_line", line_part + b"\n"
                buffer = rest
        else:
            yield "plain_text", chunk.decode("utf-8", "replace")

        while True:
            try:
                chunk = await asyncio.wait_for(stdout.read(4096), timeout=idle_timeout)
            except asyncio.TimeoutError:
                raise

            if not chunk:
                if is_json_mode and buffer:
                    yield "json_line", bytes(buffer)
                break

            if is_json_mode:
                buffer.extend(chunk)
                while b"\n" in buffer:
                    line_part, _, rest = buffer.partition(b"\n")
                    yield "json_line", line_part + b"\n"
                    buffer = rest
            else:
                yield "plain_text", chunk.decode("utf-8", "replace")

    try:
        if proc.stdout is None:
            raise RuntimeError("subprocess stdout pipe is None — spawn config bug")
        first_token_timeout_s = _delegate_first_token_timeout_s()
        idle_timeout_s = _antigravity_idle_timeout_s()
        got_first_event = False
        reader = read_stream(proc.stdout)
        while True:
            try:
                item_type, payload = await reader.__anext__()
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                stage = "startup" if not got_first_event else "idle"
                secs = int(first_token_timeout_s if not got_first_event else idle_timeout_s)
                # antigravity's finally does an UNBOUNDED proc.wait(); kill
                # the group first so it returns instead of hanging on a
                # stalled child.
                _kill_group(proc, signal.SIGTERM)
                _kill_group(proc, signal.SIGKILL)
                _log_error(
                    f"{stage}_timeout",
                    prompt=prompt,
                    cwd=spawn_cwd,
                    mode=antigravity_mode,
                    resume_id=resume_id,
                    remote_session_id=final_session_id,
                    returncode=proc.returncode,
                )
                return _err(
                    f"antigravity produced no output for {secs}s ({stage} "
                    "timeout) — the CLI appears stalled. Aborted instead of "
                    "hanging; tune via SHAREDAGENTS_DELEGATE_FIRST_TOKEN_TIMEOUT_S "
                    "/ SHAREDAGENTS_DELEGATE_IDLE_TIMEOUT_S."
                )

            got_first_event = True

            if item_type == "plain_text":
                _push(payload)
                continue

            # Otherwise, item_type is "json_line"
            try:
                decoded_line = payload.decode("utf-8", "replace")
                event = json.loads(decoded_line)
                etype = _antigravity_event_type(event) if isinstance(event, dict) else ""
                if not isinstance(event, dict) or etype not in _ANTIGRAVITY_STREAM_TYPES:
                    # 1.1.x protocol envelopes we do not yet map must not
                    # leak as raw NDJSON into the chat transcript.
                    if isinstance(event, dict) and isinstance(event.get("event"), str):
                        sid = _antigravity_session_id(event)
                        if sid:
                            _on_session_id(sid)
                        continue
                    _push(decoded_line)
                    continue
            except json.JSONDecodeError:
                _push(decoded_line)
                continue
            event_seq += 1
            if "type" in event and "event" not in event:
                legacy_stream = True
            # ``init`` carries the session / conversation id we'll persist
            # for ``--conversation`` on the next call.
            if etype == "init":
                sid = _antigravity_session_id(event)
                if sid:
                    _on_session_id(sid)
                continue
            if etype in _REASONING_BLOCK_TYPES or etype in _REASONING_DELTA_TYPES:
                thought = _antigravity_thinking_text(event)
                if thought:
                    _emit_thinking(
                        thought,
                        identity=str(event.get("id") or "message"),
                        consolidated=etype in _REASONING_BLOCK_TYPES,
                    )
                else:
                    _note_reasoning(_reasoning_signal_chars(event))
                continue
            if etype == "message":
                # Assistant message chunks. ``delta=true`` is a streaming
                # piece; the consolidated final message arrives with no
                # delta (or delta=false). We push deltas as-is for live
                # render, and skip the consolidated final to avoid
                # duplicating the body.
                role = event.get("role") or ""
                if role and role != "assistant":
                    continue
                for key in ("thinking", "reasoning", "thought"):
                    if key in event:
                        thought = _antigravity_thinking_text(event.get(key))
                        if thought:
                            # ``delta=true`` chunks stream; the consolidated
                            # final message restates them.
                            _emit_thinking(
                                thought,
                                identity=str(event.get("id") or "message"),
                                consolidated=not event.get("delta"),
                            )
                        else:
                            _note_reasoning(max(1, _reasoning_text_chars(event.get(key))))
                content = event.get("content") or ""
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            if isinstance(block, str) and event.get("delta"):
                                _push(block)
                            continue
                        block_type = block.get("type")
                        if block_type in _REASONING_BLOCK_TYPES or (block_type in _REASONING_DELTA_TYPES):
                            thought = _antigravity_thinking_text(block)
                            if thought:
                                _emit_thinking(
                                    thought,
                                    identity=str(event.get("id") or "message"),
                                    consolidated=(block_type in _REASONING_BLOCK_TYPES),
                                )
                            else:
                                _note_reasoning(_reasoning_signal_chars(block))
                            continue
                        if block_type in {"text", "output_text"}:
                            text = block.get("text") or block.get("content") or ""
                            if not isinstance(text, str) or not text:
                                continue
                            if event.get("delta"):
                                _push(text)
                            else:
                                _push_consolidated(text)
                if not isinstance(content, str) or not content:
                    continue
                if event.get("delta"):
                    _push(content)
                else:
                    # Non-delta message: only push the suffix not yet
                    # streamed (mirror the Claude consolidated path so
                    # silent-stream Antigravity configs still show output).
                    _push_consolidated(content)
                continue
            if etype == "step_update":
                # CLI 1.1.x: assistant text + tools arrive as nested
                # ``step_update`` rows (user_input / agent_response / tool).
                step = _antigravity_step_payload(event)
                sid = _antigravity_session_id(event)
                if sid:
                    _on_session_id(sid)
                step_type = str(step.get("step_type") or "")
                state = str(step.get("state") or "").upper()
                usage = step.get("usage") if isinstance(step.get("usage"), dict) else {}
                u_norm = _antigravity_usage_from_event(usage)
                if u_norm:
                    step_usage_seen = True
                    conversation = sid or final_session_id or resume_id or ""
                    if conversation:
                        # Checkpoint billed steps even if cancellation or a
                        # timeout prevents the lifetime result from arriving.
                        previous = (
                            _read_antigravity_usage_total(
                                conversation,
                                sa_sid,
                                sa_sdir,
                            )
                            or {}
                        )
                        _write_antigravity_usage_total(
                            conversation,
                            {key: previous.get(key, 0) + u_norm.get(key, 0) for key in _AGY_TOTAL_FIELDS},
                            sa_sid,
                            sa_sdir,
                        )
                    think = u_norm.get("thinking_tokens") or 0
                    if think > 0:
                        _note_reasoning(think)
                    if on_event is not None:
                        try:
                            on_event(
                                {
                                    "kind": "usage",
                                    "usage": u_norm,
                                    # This one provider call's prompt, for the ctx
                                    # meter; ``usage`` sums into the billed total.
                                    "context_usage": {
                                        "input_tokens": u_norm["input_tokens"],
                                        "cache_read_input_tokens": u_norm["cache_read_input_tokens"],
                                        "cache_creation_input_tokens": 0,
                                    },
                                }
                            )
                        except Exception:  # noqa: BLE001
                            pass
                else:
                    think = usage.get("thinking_tokens")
                    if isinstance(think, int) and think > 0:
                        _note_reasoning(think)
                if step_type == "error_message" or (state == "ERROR" and step_type == "agent_response"):
                    # An error raised by THIS run (the CLI streams it as its own
                    # step); the run recovered only if an answer follows it.
                    last_error_seq = event_seq
                    run_error_text = _antigravity_error_text(step) or run_error_text
                    continue
                if step_type in ("", "user_input"):
                    continue
                if step_type in _THINKING_STEP_TYPES:
                    _emit_step_thinking(step)
                    continue
                if any(
                    k in step
                    for k in (
                        "thinking",
                        "reasoning",
                        "thought",
                        "thought_delta",
                        "thinking_delta",
                    )
                ):
                    _emit_step_thinking(step)
                if step_type == "agent_response":
                    delta = step.get("text_delta")
                    full = step.get("text") or step.get("content") or ""
                    if isinstance(delta, str) and delta:
                        _push(delta)
                    elif isinstance(full, str) and full:
                        _push_consolidated(full)
                    continue
                if step_type != "tool":
                    continue
                tu_id = str(step.get("tool_id") or (f"agy-{step['step_index']}" if "step_index" in step else ""))
                info = step.get("tool_info") if isinstance(step.get("tool_info"), dict) else {}
                name = step.get("tool_name") or info.get("name") or "?"
                params = info.get("parameters") or step.get("parameters") or {}
                if state == "ACTIVE":
                    if tu_id and tu_id in seen_tool_uses:
                        continue
                    if tu_id:
                        seen_tool_uses.add(tu_id)
                    _push_marker(_format_antigravity_tool_use(name, params))
                    if on_event is not None:
                        try:
                            on_event(
                                {
                                    "kind": "tool_use",
                                    "id": tu_id,
                                    "name": name,
                                    "input": params if isinstance(params, dict) else {"_raw": params},
                                }
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    continue
                if state in ("DONE", "ERROR"):
                    if tu_id and tu_id in seen_tool_results:
                        continue
                    if tu_id:
                        seen_tool_results.add(tu_id)
                    out_text, is_err = _antigravity_tool_output(step)
                    if state == "ERROR":
                        is_err = True
                    synth = {
                        "status": "error" if is_err else "success",
                        "output": out_text,
                    }
                    _push_marker(_format_antigravity_tool_result(synth))
                    if on_event is not None:
                        try:
                            on_event(
                                {
                                    "kind": "tool_result",
                                    "tool_use_id": tu_id,
                                    "ok": not is_err,
                                    "text": out_text,
                                }
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    continue
                continue
            if etype == "tool_use":
                tu_id = event.get("tool_id") or ""
                if tu_id and tu_id in seen_tool_uses:
                    continue
                if tu_id:
                    seen_tool_uses.add(tu_id)
                name = event.get("tool_name") or "?"
                params = event.get("parameters") or {}
                _push_marker(_format_antigravity_tool_use(name, params))
                if on_event is not None:
                    try:
                        on_event(
                            {
                                "kind": "tool_use",
                                "id": tu_id,
                                "name": name,
                                "input": params if isinstance(params, dict) else {"_raw": params},
                            }
                        )
                    except Exception:  # noqa: BLE001
                        pass
                continue
            if etype == "tool_result":
                tu_id = event.get("tool_id") or ""
                if tu_id and tu_id in seen_tool_results:
                    continue
                if tu_id:
                    seen_tool_results.add(tu_id)
                _push_marker(_format_antigravity_tool_result(event))
                if on_event is not None:
                    out_field = event.get("output") or event.get("content") or event.get("text") or ""
                    if isinstance(out_field, list):
                        parts: list[str] = []
                        for b in out_field:
                            if isinstance(b, dict) and b.get("type") == "text":
                                parts.append(str(b.get("text") or ""))
                            elif isinstance(b, str):
                                parts.append(b)
                        out_text = "".join(parts)
                    elif isinstance(out_field, str):
                        out_text = out_field
                    else:
                        out_text = str(out_field or "")
                    is_err = bool(event.get("is_error") or event.get("error"))
                    try:
                        on_event(
                            {
                                "kind": "tool_result",
                                "tool_use_id": tu_id,
                                "ok": not is_err,
                                "text": out_text,
                            }
                        )
                    except Exception:  # noqa: BLE001
                        pass
                continue
            if etype == "error":
                # Warnings may recover, but an unanswered error belongs to
                # this invocation, unlike the conversation's final status.
                # Legacy (``type``) streams are judged by their own
                # per-invocation result; their error events are documented
                # non-fatal warnings.
                if not legacy_stream:
                    last_error_seq = event_seq
                    run_error_text = _antigravity_error_text(event) or run_error_text
                _log_error(
                    "stream_error",
                    prompt=prompt,
                    cwd=spawn_cwd,
                    mode=antigravity_mode,
                    resume_id=resume_id,
                    remote_session_id=final_session_id,
                    event=event,
                )
                continue
            if etype == "result":
                payload = _antigravity_result_payload(event)
                sid = _antigravity_session_id(event)
                if sid:
                    _on_session_id(sid)
                res_usage = payload.get("usage") or event.get("usage")
                total = _antigravity_usage_from_event(res_usage) if isinstance(res_usage, dict) else None
                conversation = sid or final_session_id or ""
                if total and not step_usage_seen:
                    # No per-call usage streamed: bill only what the lifetime
                    # total grew by since the last recorded one. A new
                    # conversation starts from zero; an unknown baseline bills
                    # the total as reported rather than nothing.
                    # Legacy result usage is per invocation, just like its
                    # status. Only modern envelopes carry lifetime totals.
                    previous = (
                        None
                        if legacy_stream
                        else _read_antigravity_usage_total(
                            conversation,
                            sa_sid,
                            sa_sdir,
                        )
                    )
                    bill = total if previous is None else _antigravity_usage_growth(total, previous)
                    if bill and on_event is not None:
                        try:
                            on_event({"kind": "usage", "usage": bill})
                        except Exception:  # noqa: BLE001
                            pass
                if total and conversation and not legacy_stream:
                    _write_antigravity_usage_total(
                        conversation,
                        total,
                        sa_sid,
                        sa_sdir,
                    )
                status_val = str(payload.get("status") or "").lower()
                result_finished_reason = str(payload.get("finished_reason") or "")
                resp = payload.get("response")
                if isinstance(resp, str) and resp:
                    _push_final(resp)
                if status_val and status_val not in ("success", "ok"):
                    result_error = payload
                continue
    except asyncio.CancelledError:
        _log_error(
            "cancelled",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=antigravity_mode,
            resume_id=resume_id,
            remote_session_id=final_session_id,
            returncode=proc.returncode,
        )
        # Hard cancel → SIGKILL the WHOLE tree so antigravity's spawned
        # tools / subagents (incl. ones in their own session) die too.
        _kill_tree(proc, signal.SIGKILL)
        raise
    finally:
        watcher_stop.set()
        if watcher_task is not None:
            try:
                await watcher_task
            except Exception:  # noqa: BLE001
                pass
        try:
            if log_path.exists():
                log_path.unlink()
        except Exception:  # noqa: BLE001
            pass

        sandbox_broker.release_process(locals().get("_lease_id"))
        if "stdin_task" in locals() and not stdin_task.done():
            stdin_task.cancel()
        try:
            if "stdin_task" in locals():
                await stdin_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        try:
            if "proc" in locals():
                await proc.wait()
        except Exception:  # noqa: BLE001
            pass

    tail_reconciled = False

    async def _reconcile_stream_tail() -> None:
        """Append the answer tail the stream lost but the CLI's DB kept."""
        nonlocal tail_reconciled
        tail_reconciled = True
        if tail_after_idx is None or (resume_id and final_session_id != resume_id):
            return
        # The CLI can drop the end of its last answer from the stream while its
        # conversation DB keeps it, so the reply shown ends mid-word.
        streamed = "".join(text_buf)
        db_last = await _antigravity_conversation_last_message(
            final_session_id,
            tail_after_idx,
        )
        new_tail = _antigravity_reconcile_tail(streamed.rstrip(), db_last)
        if new_tail and streamed != streamed.rstrip():
            # The stream already delivered the whitespace before the cut.
            new_tail = new_tail.lstrip()
        if new_tail:
            _emit_answer(new_tail)
            _log_error(
                "stream_tail_recovered",
                prompt=prompt,
                cwd=spawn_cwd,
                mode=antigravity_mode,
                resume_id=resume_id,
                remote_session_id=final_session_id,
                returncode=proc.returncode,
                finished_reason=result_finished_reason or None,
                streamed_chars=len(streamed),
                recovered_chars=len(new_tail),
            )
        elif _antigravity_looks_cut(streamed.rstrip()) and not (
            db_last.strip() and streamed.rstrip().endswith(db_last.rstrip())
        ):
            _log_error(
                "stream_tail_lost",
                prompt=prompt,
                cwd=spawn_cwd,
                mode=antigravity_mode,
                resume_id=resume_id,
                remote_session_id=final_session_id,
                returncode=proc.returncode,
                finished_reason=result_finished_reason or None,
                streamed_chars=len(streamed),
                db_chars=len(db_last),
            )

    unrecovered_run_error = last_error_seq is not None and (last_answer_seq is None or last_answer_seq < last_error_seq)
    # A print-mode wall cut is a truncation, not a failure: the model was
    # interrupted mid-answer and the CLI had already written what it composed
    # to its conversation DB; only stdout lost the tail. CLI 1.5.0 reports the
    # cut as ``status=ERROR`` with exit 3 (1.4.12: SUCCESS, exit 0); both name
    # it with ``finished_reason=print_timeout``, the trigger here. The exit
    # code only corroborates. Recover the tail before the error return below,
    # which stays as is so the engine still continues the turn; an error step
    # in this run's stream is a genuine failure and gets no recovery.
    if result_finished_reason == "print_timeout" and proc.returncode in (0, 3) and not unrecovered_run_error:
        await _reconcile_stream_tail()
    if unrecovered_run_error:
        is_error = True
    if result_error is not None:
        if proc.returncode == 0 and not unrecovered_run_error and last_answer_seq is not None and not legacy_stream:
            # The status is the conversation's: a stale error step from an
            # earlier run, or one this run recovered from and answered after.
            # Failing here re-ran finished turns five times.
            _log_error(
                "result_error_ignored",
                prompt=prompt,
                cwd=spawn_cwd,
                mode=antigravity_mode,
                resume_id=resume_id,
                remote_session_id=final_session_id,
                returncode=proc.returncode,
                message=_antigravity_error_text(result_error),
                event=result_error,
                run_error=run_error_text or None,
            )
        else:
            is_error = True
            error_event = result_error

    if proc.returncode != 0 and not is_error:
        stderr_bytes = b""
        if proc.stderr is not None:
            try:
                stderr_bytes = await proc.stderr.read()
            except Exception:  # noqa: BLE001
                pass
        err_msg = stderr_bytes.decode("utf-8", "replace")[:400] or _signal_name(proc.returncode)
        _log_error(
            "nonzero_exit",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=antigravity_mode,
            resume_id=resume_id,
            remote_session_id=final_session_id,
            returncode=proc.returncode,
            stderr=stderr_bytes.decode("utf-8", "replace"),
        )
        return _err(f"antigravity exited {proc.returncode}: {err_msg}")

    if is_error:
        # Prefer the error this run streamed over the result's text, which
        # may be an older one replayed from the conversation.
        err_detail = (
            run_error_text or _antigravity_error_text(error_event) or str(error_event.get("status") or "") or "unknown"
        )
        _log_error(
            "reported_error",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=antigravity_mode,
            resume_id=resume_id,
            remote_session_id=final_session_id,
            returncode=proc.returncode,
            message=err_detail,
            event=error_event,
        )
        return _err(f"antigravity reported error: {err_detail}")

    if not tail_reconciled:
        await _reconcile_stream_tail()

    body_text = "".join(text_buf).strip()
    if not body_text and reasoning_chars <= 0:
        _log_error(
            "empty_response",
            prompt=prompt,
            cwd=spawn_cwd,
            mode=antigravity_mode,
            resume_id=resume_id,
            remote_session_id=final_session_id,
            returncode=proc.returncode,
        )
        return _err("antigravity returned empty response")

    if reasoning_chars > 0 and on_event is not None:
        try:
            on_event({"kind": "reasoning", "chars": reasoning_chars})
        except Exception:  # noqa: BLE001
            pass

    # Antigravity's stream envelope does not include a model field.  Its
    # conversation DB is written just after the stream, so retry once before
    # giving up; a missing/pruned DB must never turn a successful delegate into
    # an error.  ExternalCliEngine consumes this event and persists its own
    # ``<sid>.antigravity_cli.json`` sidecar; the delegate sidecar is updated
    # as well for legacy callers and diagnostics.
    actual_model = ""
    for attempt in range(2):
        actual_model = _antigravity_conversation_model(final_session_id)
        if actual_model:
            break
        if attempt == 0:
            await asyncio.sleep(0.05)
    if actual_model:
        _write_antigravity_actual_model(actual_model, sa_sid, sa_sdir)
        if on_event is not None:
            try:
                on_event({"kind": "model", "id": actual_model})
            except Exception:  # noqa: BLE001
                pass

    if final_session_id:
        _write_antigravity_session_id(final_session_id, sa_sid, sa_sdir)

    header_bits = []
    if final_session_id:
        header_bits.append(f"session={final_session_id[:12]}")
    header_bits.append(f"mode={antigravity_mode}")
    if resume_id:
        header_bits.append("resumed")
    header = f"[antigravity {' · '.join(header_bits)}]"

    # No "send only NEW context" footer: continuity is owned by the
    # session-id link (``--resume <id>``), not a per-result note. The
    # contract for the orchestrator lives once in the delegate-strategy
    # skill, not stapled onto every tool result (where it also leaked
    # into user-facing output and lied after a stale-resume fell back
    # to a fresh session).
    body = f"{header}\n\n{body_text}" if body_text else header
    body, truncated = _truncate(body)
    return ToolResult(content=body, ok=True, truncated=truncated)
