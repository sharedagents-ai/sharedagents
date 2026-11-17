"""Foundational module-level helpers + constants for the Telegram bot.

Extracted from ``bot.py`` (de-monolith). Pure/utility layer the bot, its mixins
and the render/api modules all sit on top of: on-disk state I/O (session map,
offset, inflight turns, visibility), tool/turn watchdog + supervision,
incident/error assembly, text chunking + parsers, plus the small constants those
functions read. Everything moved verbatim.

This module is DELIBERATELY self-contained — it imports only stdlib + the two
cross-package symbols it needs (``TelegramConfig``, ``AcpRemoteError``) from
their real homes, and NOTHING from ``bot``. That one-way layering lets ``bot.py``
re-export every name here early, with no import cycle. ``bot.py`` re-exports all
of these so callers/tests importing them from ``sharedagents_telegram.bot`` keep
working.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sharedagents_acp.client.client import AcpRemoteError
from sharedagents_telegram.config import TelegramConfig

# Substrings the gateway uses when a session id is no longer attached
# to a live child (idle-reaped, or the gateway restarted between turns).
# Matched case-insensitively against ``AcpRemoteError.message`` so the
# bot can recover by re-loading the session before re-issuing the
# prompt — see ``_reopen_session_after_loss``. The full message looks
# like ``backend error: LookupError: no live child owns session …``.
_RECOVERABLE_SESSION_ERROR_PHRASES = (
    "no live child owns session",
    "idle-reaped",
    "session not found",
    "unknown session",
)


def _is_recoverable_session_error(exc: BaseException) -> bool:
    """Return True for errors that just mean "the session id you sent
    is not live on the gateway anymore" — recoverable by re-loading or
    re-creating the session, not by surfacing the failure.

    Three error families count as recoverable:
      * :class:`AcpRemoteError` whose ``message`` matches one of the
        gateway's "session not attached" phrases — the gateway is
        alive but our sid is stale.
      * :class:`AcpTransportClosed` — the transport itself died
        (SSE stream torn down by an OOM-kill, network blip, etc.).
        Reopening the client + session puts us back in business.
      * The canonical submit/subscribe HTTP path (spec §Project 1) raises a
        plain :class:`LookupError`/``RuntimeError`` whose text carries the
        same "session not found" phrase (a 404 from the submit route means
        the gateway has no log for our sid) — recover the same way.
    """
    from sharedagents_acp.client.client import AcpTransportClosed

    if isinstance(exc, AcpTransportClosed):
        return True
    if isinstance(exc, AcpRemoteError):
        msg = (getattr(exc, "message", "") or "").lower()
        return any(phrase in msg for phrase in _RECOVERABLE_SESSION_ERROR_PHRASES)
    # Canonical HTTP submit path: classify by the exception text so a 404 →
    # ``LookupError("session not found: …")`` reopens + retries.
    if isinstance(exc, (LookupError, RuntimeError)):
        msg = str(exc).lower()
        return any(phrase in msg for phrase in _RECOVERABLE_SESSION_ERROR_PHRASES)
    return False


def _format_relative_age(iso_ts: str) -> str:
    """Render an ISO 8601 timestamp as a compact relative age
    (``5m``, ``2h``, ``3d``). Empty / unparseable input returns ``?``
    so the picker line still renders without a hole.
    """
    if not iso_ts:
        return "?"
    try:
        dt = datetime.fromisoformat(iso_ts)
    except (TypeError, ValueError):
        return "?"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta_s = max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))
    if delta_s < 60:
        return f"{delta_s}s"
    if delta_s < 3600:
        return f"{delta_s // 60}m"
    if delta_s < 86400:
        return f"{delta_s // 3600}h"
    return f"{delta_s // 86400}d"


# A turn that hits Telegram's per-chat edit 429 this many times degraded the
# live UX — the streamed message froze on "💭 thinking…" for retry_after
# (up to 34s) stretches even though the turn completed. Such a turn SUCCEEDS, so
# it never becomes an error session and the self-improver was BLIND to it
# ("why did the self-improver not get that?"). At/above this
# count we surface a synthetic incident so the SI can triage + fix the edit
# cadence. The incident fingerprint is STABLE per kind, so repeated throttled
# turns dedup to ONE issue (no flood); a recurrence after a merged fix re-opens it.
THROTTLE_INCIDENT_429_THRESHOLD = 2

# Persisted marker for Telegram reply bubbles that were mid-turn when the bot
# stopped. Startup recovery reads the session log and sends the final folded
# assistant text as a new message, so a service restart does not strand the
# chat on "thinking…".
INFLIGHT_TURNS_FILE = "inflight_turns.json"

# Hard ceiling on a single Telegram message we render. Telegram caps text
# at 4096; we leave headroom for the ``\n\n[…]`` truncation marker.
TELEGRAM_MAX_TEXT = 4000

# Marker appended when a pinned block's text is capped to a single message.
_REASONING_TRUNCATED_MARKER = "\n\n… _(reasoning truncated — full trace in web/logs)_"


def _cap_to_single_message(text: str, *, prefix: str = "") -> str:
    """Cap ``text`` so ``prefix + text`` fits in ONE Telegram message.

    A pinned block that sits ABOVE later messages (the consolidated top-💭
    reasoning block, finalized only at ``turn_ended`` after the 🔧/💬 messages
    of the turn have already been sent) must never overflow: Telegram orders
    by message id and is append-only, so overflow rolls into BRAND-NEW messages
    that land BELOW the answer — the "reasoning after the answer" ordering bug.
    Keeping such a block to a single message lets ``finalize`` edit it in place,
    preserving order. The full (untruncated) reasoning still lives in the
    session log and the web UI; this only bounds the Telegram bubble.
    """
    if not text:
        return text
    budget = TELEGRAM_MAX_TEXT - len(prefix)
    if len(text) <= budget:
        return text
    head_budget = budget - len(_REASONING_TRUNCATED_MARKER)
    if head_budget <= 0:
        # Degenerate prefix — hard-truncate to the raw budget.
        return text[: max(0, budget)]
    return text[:head_budget].rstrip() + _REASONING_TRUNCATED_MARKER


ACP_TRUSTED_CHATS_ENV = "SHAREDAGENTS_TELEGRAM_TRUSTED_CHATS"


def _truncate_for_telegram(text: str) -> str:
    """Last-line-of-defense truncation for one-shot ``sendMessage`` calls.

    The streaming flow rolls into multiple messages via
    :class:`_RollingMessage` instead — this function only ever fires for
    ad-hoc text (slash-command replies, error markers) that exceeds
    ``TELEGRAM_MAX_TEXT`` despite all callers being short.
    """
    if len(text) <= TELEGRAM_MAX_TEXT:
        return text
    return text[: TELEGRAM_MAX_TEXT - 6] + "\n\n[…]"


def _split_at(text: str, max_len: int) -> int:
    """Pick a clean break index ≤ ``max_len`` for splitting a long reply.

    Prefers (in order): the last newline, then the last space, both only
    when they fall in the right half of ``max_len`` so we don't fragment
    a paragraph at the very start. Falls back to a hard cut at ``max_len``
    when the chunk is one giant unbroken token (a base64 blob, etc.).
    """
    if len(text) <= max_len:
        return len(text)
    half = max_len // 2
    cut = text.rfind("\n", 0, max_len)
    if cut >= half:
        return cut
    cut = text.rfind(" ", 0, max_len)
    if cut >= half:
        return cut
    return max_len


def _chunk_for_telegram(text: str, max_len: int = TELEGRAM_MAX_TEXT) -> list[str]:
    """Split ``text`` into Telegram-sized chunks at clean boundaries.

    Replaces the old truncate-with-``[…]`` behaviour for one-shot
    ``sendMessage`` calls: instead of dropping the tail, the whole reply
    rolls across as many messages as it takes, breaking at newline >
    space > hard-cut via :func:`_split_at`. Mirrors
    :meth:`_RollingMessage._drain_overflow` so streamed and one-shot
    replies split identically. The full body is preserved verbatim — no
    ellipsis marker is ever appended.
    """
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    frozen = 0
    while len(text) - frozen > max_len:
        pending = text[frozen:]
        cut = _split_at(pending, max_len)
        head = pending[:cut].rstrip()
        if not head:
            # The whole window before the break was whitespace (or one
            # unbroken token): hard-cut so we always make progress and
            # never lose a byte.
            head = pending[:max_len]
            frozen += max_len
        else:
            frozen += len(head)
        chunks.append(head)
    tail = text[frozen:]
    if tail.strip() or not chunks:
        chunks.append(tail)
    return chunks


def _parse_chat_key(key: str) -> tuple[int | None, int | None]:
    """Inverse of :meth:`TelegramConfig.chat_key`.

    ``"-1001234567890:4"`` → ``(-1001234567890, 4)``;
    ``"123456789"`` → ``(123456789, None)``;
    ``"garbage"`` → ``(None, None)`` so the caller can mark the entry
    stale instead of crashing on a malformed map.
    """
    if not key:
        return None, None
    if ":" in key:
        chat_part, _, thread_part = key.partition(":")
        try:
            return int(chat_part), int(thread_part)
        except ValueError:
            return None, None
    try:
        return int(key), None
    except ValueError:
        return None, None


def _acp_sessions_path(tg_cfg: TelegramConfig) -> Path:
    return tg_cfg.sessions_path


def _offset_path(tg_cfg: TelegramConfig) -> Path:
    return tg_cfg.offset_path


def _inflight_turns_path(tg_cfg: TelegramConfig) -> Path:
    return tg_cfg.home / INFLIGHT_TURNS_FILE


def _load_inflight_turns(tg_cfg: TelegramConfig) -> dict[str, dict[str, Any]]:
    p = _inflight_turns_path(tg_cfg)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, val in raw.items():
        if not isinstance(key, str) or not isinstance(val, dict):
            continue
        try:
            chat_id = int(val.get("chat_id"))
        except (TypeError, ValueError):
            continue
        thread_raw = val.get("thread_id")
        thread_id: int | None
        if thread_raw is None:
            thread_id = None
        else:
            try:
                thread_id = int(thread_raw)
            except (TypeError, ValueError):
                continue
        sid = str(val.get("sid") or "")
        agent = str(val.get("agent") or "")
        if not sid or not agent:
            continue
        # NOTE: the record's ``agent`` is preserved verbatim — it is the session's
        # on-disk owner (attribution lives in the SHARED sessions dir, keyed by
        # agent_name). Healing a retired-managed-`main` owner happens at RECOVERY
        # time (_recover_inflight_turns_on_startup): the load TARGET heals to jarvis
        # so a live config resolves the shared sessions_dir, while the session READ
        # keeps expected_agent == this original owner so the completed turn is found.
        out[key] = {
            "chat_id": chat_id,
            "thread_id": thread_id,
            "sid": sid,
            "agent": agent,
            "placeholder_id": val.get("placeholder_id"),
            "last_text": str(val.get("last_text") or ""),
            "started_at": float(val.get("started_at") or 0.0),
        }
    return out


def _save_inflight_turns(
    tg_cfg: TelegramConfig,
    mapping: dict[str, dict[str, Any]],
) -> None:
    p = _inflight_turns_path(tg_cfg)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if mapping:
            p.write_text(
                json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif p.exists():
            p.unlink()
    except OSError:
        logging.getLogger("sharedagents.telegram").exception(
            "%s save failed",
            p.name,
        )


def _load_acp_session_map(
    tg_cfg: TelegramConfig,
) -> dict[str, dict[str, str]]:
    """Load ``{chat_key: {agent_name: session_id}}`` from disk.

    Two shapes are accepted:

    1. **Nested** (current) — ``{chat_key: {agent: sid}}``.
    2. **Flat** (legacy from Phase A) — ``{chat_key: sid_string}``.
       Each entry is hoisted into ``{tg_cfg.default_agent: sid}``
       so old installs migrate transparently.

    Malformed entries are dropped silently — better to lose one
    binding than crash the bot's boot path.
    """
    p = _acp_sessions_path(tg_cfg)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    default_agent = tg_cfg.default_agent or "jarvis"
    out: dict[str, dict[str, str]] = {}
    for chat_key, val in raw.items():
        if not isinstance(chat_key, str):
            continue
        if isinstance(val, str) and val:
            # Legacy flat record — assume the default agent.
            out[chat_key] = {default_agent: val}
            continue
        if not isinstance(val, dict):
            continue
        agent_map: dict[str, str] = {}
        for agent_name, sid in val.items():
            if not isinstance(agent_name, str) or not isinstance(sid, str):
                continue
            if not agent_name or not sid:
                continue
            agent_map[agent_name] = sid
        if agent_map:
            out[chat_key] = agent_map
    return out


def _load_visibility_map(
    tg_cfg: TelegramConfig,
) -> dict[str, dict[str, Any]]:
    """Load ``{chat_key: {reasoning_visibility, tool_output}}`` from
    disk. Missing file or malformed JSON ⇒ empty dict (every chat
    falls through to the off/off defaults). Bad per-chat entries are
    dropped silently — better to reset one chat's display state than
    to brick the bot's startup."""
    p = tg_cfg.visibility_path
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for chat_key, val in raw.items():
        if not isinstance(chat_key, str) or not isinstance(val, dict):
            continue
        entry: dict[str, Any] = {}
        # ``reasoning_visibility`` is a string token (off/on/stream);
        # tolerate the legacy bool ``tool_output_visible`` shape too
        # so an old visibility.json from before the 3-state /tools
        # rework still loads.
        rv = val.get("reasoning_visibility")
        if isinstance(rv, str) and rv:
            entry["reasoning_visibility"] = rv
        to = val.get("tool_output")
        if isinstance(to, str) and to:
            entry["tool_output"] = to
        elif "tool_output_visible" in val:
            entry["tool_output"] = "on" if val.get("tool_output_visible") else "off"
        if entry:
            out[chat_key] = entry
    return out


def _save_visibility_map(
    tg_cfg: TelegramConfig,
    mapping: dict[str, dict[str, Any]],
) -> None:
    """Persist the per-(chat,thread) visibility map. Atomic-ish via
    plain ``write_text``; matches the chat→session map's ergonomics."""
    p = tg_cfg.visibility_path
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logging.getLogger("sharedagents.telegram").exception(
            "%s save failed",
            p.name,
        )


def _save_acp_session_map(
    tg_cfg: TelegramConfig,
    mapping: dict[str, dict[str, str]],
) -> None:
    """Persist the nested chat→agent→session map.

    Atomic-ish via plain ``write_text`` — the file is small and torn
    writes rare; the worst case is one lost mapping which the next
    save overwrites.
    """
    p = _acp_sessions_path(tg_cfg)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logging.getLogger("sharedagents.telegram").exception(
            "%s save failed",
            p.name,
        )


def _load_update_offset(tg_cfg: TelegramConfig) -> int:
    p = _offset_path(tg_cfg)
    if not p.exists():
        return 0
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if isinstance(raw, dict):
        v = raw.get("update_offset")
        if isinstance(v, int):
            return v
    return 0


def _save_update_offset(tg_cfg: TelegramConfig, offset: int) -> None:
    p = _offset_path(tg_cfg)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"update_offset": int(offset)}) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logging.getLogger("sharedagents.telegram").exception(
            "%s save failed",
            p.name,
        )


def _trusted_chats_from_env_or_cfg(tg_cfg: TelegramConfig) -> set[int]:
    """Resolve the trusted-chats set from the bot config + env var.

    ``TelegramConfig.trusted_chat_ids`` is the canonical source. The
    legacy env var ``SHAREDAGENTS_TELEGRAM_TRUSTED_CHATS`` is still merged
    in as a comma-separated list of int chat ids so existing test
    setups that export it keep working without writing config.
    """
    chats: set[int] = {int(x) for x in (tg_cfg.trusted_chat_ids or [])}
    raw = os.environ.get(ACP_TRUSTED_CHATS_ENV, "").strip()
    if raw:
        for piece in raw.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                chats.add(int(piece))
            except ValueError:
                pass
    return chats


# Must sit ABOVE the slowest engine idle (Antigravity print-timeout 480s)
# so the engine reports a stall first. 480 + TELEGRAM_IDLE_BUFFER_S (60).
_DEFAULT_TURN_IDLE_TIMEOUT_S: float = 540.0  # ws/stdio fallback only (no gateway)

_DEFAULT_GATEWAY_HEALTH_POLL_S: float = 15.0  # orchestrator liveness poll interval


def _parse_tool_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _operator_wait_pending(progress: dict) -> bool:
    """True when the turn is parked on a question or permission card."""
    sid = progress.get("probe_sid") or progress.get("sid")
    sessions_dir = progress.get("sessions_dir")
    if not sid or not sessions_dir:
        return False
    try:
        from sharedagents_session.ask_pending import session_pending_interaction

        return bool(session_pending_interaction(sessions_dir, str(sid)))
    except Exception:  # noqa: BLE001
        return False


def _effective_idle_timeout(progress: dict, default: float) -> float:
    """Turn watchdog budget — extends while a blocking shell tool runs."""
    for key in ("tool_idle_timeout", "idle_timeout"):
        v = progress.get(key)
        if v is not None:
            return float(v)
    return default


def _recompute_tool_idle_timeout(progress: dict) -> None:
    budgets = progress.get("tool_idle_budgets") or {}
    progress["tool_idle_timeout"] = max(budgets.values()) if budgets else None


def _apply_tool_watchdog(
    progress: dict | None,
    tool_name: str,
    args: dict[str, Any],
    call_id: str = "",
) -> None:
    if progress is None:
        return
    from sharedagents_agent.tools.runners import telegram_idle_budget_s

    budget = telegram_idle_budget_s(tool_name, args)
    if budget is None or not call_id:
        return
    budgets = progress.setdefault("tool_idle_budgets", {})
    budgets[call_id] = budget
    _recompute_tool_idle_timeout(progress)


def _clear_tool_watchdog(progress: dict | None, call_id: str = "") -> None:
    if progress is None or not call_id:
        return
    budgets = progress.get("tool_idle_budgets")
    if not isinstance(budgets, dict):
        return
    budgets.pop(call_id, None)
    _recompute_tool_idle_timeout(progress)


async def _await_with_idle_watchdog(task, progress: dict, idle_timeout: float):
    """Await ``task``, cancelling it if it makes NO progress for
    ``idle_timeout`` seconds. The caller bumps ``progress['last']`` (monotonic)
    on every streamed event; a wedged delegate that emits nothing trips the
    watchdog, which sets ``progress['stalled'] = True`` and cancels ``task`` (so
    the awaiting ``await task`` raises ``CancelledError`` — the caller inspects
    ``progress['stalled']`` to tell a watchdog cancel from a real user abort).
    ``idle_timeout <= 0`` disables the watchdog."""

    async def _wd() -> None:
        tick = max(5.0, min(idle_timeout / 4.0, 30.0))
        while not task.done():
            await asyncio.sleep(tick)
            if idle_timeout <= 0 or task.done():
                continue
            if _operator_wait_pending(progress):
                progress["last"] = time.monotonic()
                continue
            if time.monotonic() - progress["last"] > _effective_idle_timeout(
                progress,
                idle_timeout,
            ):
                progress["stalled"] = True
                task.cancel()
                return

    wd = asyncio.ensure_future(_wd())
    try:
        return await task
    finally:
        wd.cancel()


def _probe_session_turn_state(
    sid: str,
    sessions_dir,
    since_offset: int = 0,
) -> dict[str, Any]:
    """Read the session JSONL and report orchestrator turn liveness.

    Returns a dict with:
    * ``active`` — a ``turn_started`` is open with no matching ``turn_ended``
    * ``error`` / ``error_classification`` / ``aborted`` / ``final_content`` —
      from the last ``turn_ended`` when the session is no longer active
    * ``found`` — whether the log file exists
    * ``artifacts`` — that terminal turn's ``artifact_reference`` records,
      reduced to what a re-fetch through the gateway needs (never the path)

    ``since_offset`` scopes the scan to OUR turn's window: records at a byte
    offset below it belong to PRIOR turns (a message queued behind an earlier
    turn in the same session, e.g. a /goal session) and must be ignored — else
    the caller recovers a stale prior reply as if it were ours. A ``turn_ended``
    that closes a turn whose ``turn_started`` was BEFORE the window (a
    concurrent prior turn ending after we submitted) is likewise not ours, so
    it is only counted when we have an open turn from within the window.
    """
    from pathlib import Path

    from sharedagents_session.session_reader import tail_jsonl

    path = Path(sessions_dir) / f"{sid}.jsonl"
    if not path.is_file():
        return {"found": False, "active": False}
    open_turns = 0
    last_ended: dict[str, Any] | None = None
    turn_files: list[dict[str, Any]] = []
    last_files: list[dict[str, Any]] = []
    for ev, off in tail_jsonl(path, 0):
        # ``tail_jsonl`` yields the line's END offset (byte after its \n). A
        # record is pre-submit iff its end is at or before the floor (the log
        # size captured at submit), so skip with ``<=`` — otherwise a
        # concurrent prior turn whose ``turn_started`` is the LAST pre-submit
        # line (end == floor) leaks in and its later ``turn_ended`` is
        # misattributed to us.
        if since_offset and isinstance(off, int) and off <= since_offset:
            continue
        etype = ev.get("type")
        if etype == "turn_started":
            open_turns += 1
            turn_files = []
        elif etype == "artifact_reference" and open_turns > 0:
            # Only what a recovering caller needs to fetch it back through the
            # gateway — never the host path/locator.
            turn_files.append(
                {
                    key: ev[key]
                    for key in ("type", "turn_idx", "sha256", "display_name", "media_type", "size_bytes")
                    if key in ev
                }
            )
        elif etype == "turn_ended" and open_turns > 0:
            # Closes one of OUR in-window turns.
            open_turns -= 1
            last_ended = ev
            last_files, turn_files = turn_files, []
            # else: closes a turn that started before our window — a prior /
            # concurrent turn. Not ours; ignore its terminal record.
    active = open_turns > 0
    out: dict[str, Any] = {"found": True, "active": active}
    if last_ended is not None and not active:
        err = last_ended.get("error")
        if err:
            out["error"] = str(err)
            out["error_classification"] = str(
                last_ended.get("error_classification") or "",
            )
        out["aborted"] = bool(last_ended.get("aborted"))
        fc = last_ended.get("final_content")
        if fc is not None:
            out["final_content"] = str(fc)
        out["artifacts"] = last_files
    return out


def _gateway_health_poll_s() -> float:
    """Poll interval for gateway-integrated turn supervision
    (``telegram.gateway_health_poll_s`` in ``~/.sharedagents/config.json``)."""
    try:
        from sharedagents_agent.agent import load_global_config

        tg = (load_global_config() or {}).get("telegram") or {}
        v = tg.get("gateway_health_poll_s")
        if v is not None:
            return max(1.0, float(v))
    except Exception:  # noqa: BLE001
        pass
    return _DEFAULT_GATEWAY_HEALTH_POLL_S


async def _await_with_turn_supervision(
    task,
    progress: dict,
    idle_timeout: float,
    *,
    sessions_dir=None,
    use_gateway_health: bool = False,
):
    """Await a turn dispatch task with orchestrator-aware supervision.

    When ``use_gateway_health`` is true (HTTP gateway deployments), poll the
    session log every :func:`_gateway_health_poll_s` seconds instead of using a
    dumb no-output timer:

    * **Agent still running** (``active`` in JSONL) → keep waiting; silence on
      the Telegram stream is fine.
    * **Turn ended with error** → cancel the dispatch within one poll (~15s),
      surface the real engine error.
    * **Turn ended OK but stream stuck** → recover ``final_content`` from the log.

    The legacy idle watchdog (``turn_idle_timeout_s``) applies only to the
    ws/stdio fallback path where no gateway session log is available.
    """
    if not use_gateway_health or sessions_dir is None:
        return await _await_with_idle_watchdog(task, progress, idle_timeout)

    poll_s = _gateway_health_poll_s()

    async def _health() -> None:
        while not task.done():
            await asyncio.sleep(poll_s)
            if task.done():
                return
            sid = progress.get("probe_sid") or progress.get("sid")
            if not sid:
                continue
            try:
                state = await asyncio.to_thread(
                    _probe_session_turn_state,
                    sid,
                    sessions_dir,
                    progress.get("min_offset") or 0,
                )
            except Exception:  # noqa: BLE001
                continue
            if not state.get("found"):
                continue
            if state.get("active") or _operator_wait_pending(progress):
                # Orchestrator says the turn is still live, or the log is
                # parked on a question/permission — do not treat quiet
                # human wait as a stall.
                progress["last"] = time.monotonic()
                continue
            # NOTE: ``_probe_session_turn_state`` is scoped to OUR turn's log
            # window via ``progress['min_offset']`` (set at submit). While our
            # turn is still QUEUED behind a prior turn (the SOUTH orchestrator
            # is single-flight; a /goal session may queue us behind an auto
            # turn), the probe reports no in-window terminal record, so we keep
            # waiting rather than recover a PRIOR turn's reply — which would
            # cancel our dispatch and deliver a stale duplicate (2026-07-04
            # bug). The dispatch task's own ``queue_timeout_s`` bounds the wait.
            err = state.get("error")
            if err:
                progress["gateway_error"] = err
                progress["gateway_error_class"] = state.get("error_classification") or ""
                progress["supervision_cancel"] = True
                task.cancel()
                return
            fc = state.get("final_content")
            if fc is not None:
                progress["gateway_recover"] = fc
                # Propagate the recovered turn's REAL terminal flag so the
                # supervision branch in _stream_turn_for_chat classifies the
                # reaction correctly: an aborted recovery (partial content)
                # must clear the ✍️ reaction, not mark it 👍. (codex pre-
                # deploy review round 2, HIGH finding)
                progress["gateway_recover_aborted"] = bool(state.get("aborted"))
                progress["recovered_artifacts"] = list(state.get("artifacts") or [])
                progress["supervision_cancel"] = True
                task.cancel()
                return

    health_task = asyncio.ensure_future(_health())
    try:
        return await task
    finally:
        health_task.cancel()


def _turn_idle_timeout_s() -> float:
    """No-progress timeout for ws/stdio turns without gateway health polling
    (``telegram.turn_idle_timeout_s`` in ``~/.sharedagents/config.json``; 0 disables).
    Ignored on the HTTP gateway path — :func:`_await_with_turn_supervision`
    polls the orchestrator instead."""
    try:
        from sharedagents_agent.agent import load_global_config

        tg = (load_global_config() or {}).get("telegram") or {}
        v = tg.get("turn_idle_timeout_s")
        if v is not None:
            return float(v)
    except Exception:  # noqa: BLE001
        pass
    return _DEFAULT_TURN_IDLE_TIMEOUT_S


_DEFAULT_QUEUE_TIMEOUT_S: float = 300.0  # drop a message queued > 5 min


def _queue_timeout_s() -> float:
    """Seconds a message may sit queued behind a running turn before it is
    dropped instead of answered (``telegram.queue_timeout_s`` in
    ``~/.sharedagents/config.json``; 0 disables — wait forever). Mirrors
    :func:`_turn_idle_timeout_s`: unified config, no scattered env vars."""
    try:
        from sharedagents_agent.agent import load_global_config

        tg = (load_global_config() or {}).get("telegram") or {}
        v = tg.get("queue_timeout_s")
        if v is not None:
            return float(v)
    except Exception:  # noqa: BLE001
        pass
    return _DEFAULT_QUEUE_TIMEOUT_S


def _is_transient_connectivity_error(exc: BaseException) -> bool:
    """A connectivity / restart artifact, NOT a product bug — the orchestrator
    being briefly unreachable (a 503 while it restarts, incl. the self-improver's
    OWN auto-restarts) or a torn connection. These self-heal, so they must NOT be
    surfaced to the self-improver (else every restart churns the fixer)."""
    s = str(exc).lower()
    return any(
        p in s
        for p in (
            "unreachable",
            "connecterror",
            "connectionerror",
            "connection refused",
            "all connection attempts failed",
            "connection reset",
            "timed out",
            "timeout",
            "temporarily unavailable",
            # A mid-stream orchestrator restart (e.g. a deploy / SI auto-restart)
            # tears the chunked SSE body; httpx raises RemoteProtocolError /
            # "peer closed connection …" / "incomplete chunked read" rather than a
            # clean ConnectError. Same class of self-healing blip — must NOT churn
            # the self-improver nor read as a product bug.
            "peer closed connection",
            "incomplete chunked read",
            "remoteprotocolerror",
            "server disconnected",
            "connection closed",
        )
    )


def _record_bot_incident(kind: str, detail: str = "") -> None:
    """Log a bot-side failure that happens OUTSIDE an agent turn.

    Such failures have no session log, so they are logged at WARNING to keep
    them visible. Best-effort — never raises.
    """
    try:
        logging.getLogger("sharedagents.telegram").warning(
            "bot incident [%s]: %s",
            kind,
            detail,
        )
    except Exception:  # noqa: BLE001 — never let incident logging break a turn
        pass


def _maybe_record_agent_error(exc: BaseException) -> bool:
    """A NON-recoverable telegram turn failure happens at the bot boundary, so
    the self-improver is otherwise BLIND to it (no agent error session). Surface
    it as an incident UNLESS it is a transient connectivity/restart artifact.
    Fingerprint is per exception TYPE so distinct error classes get distinct
    issues but repeats dedup. Returns True if an incident was recorded."""
    if _is_transient_connectivity_error(exc):
        return False
    _bot._record_bot_incident(
        f"telegram_agent_error_{type(exc).__name__.lower()}",
        (
            f"telegram turn failed non-recoverably with "
            f"{type(exc).__name__}: {str(exc)[:300]} — surfaced to the "
            f"self-improver (the bot-boundary failure has no agent error "
            f"session). Transient connectivity/restart errors are filtered out."
        ),
    )
    return True


def _assemble_error_reply(
    visible_parts: list[str],
    pending_thoughts: list[str],
    exc: BaseException,
) -> str:
    """Final reply for a non-recoverable turn failure that PRESERVES any
    text already streamed instead of wiping it.

    The streamed content is live in the rolling Telegram message, but the
    turn's final paint repaints the message with this function's return
    value — so returning a bare ``⚠️ agent error`` pointer erased whatever
    the user had already read. Here we keep the
    streamed buffer (plus any ``on``-mode thoughts buffered but not yet
    flushed) and append a footer carrying the CONCRETE error — type +
    truncated message — so the operator sees what failed without digging
    through telegram.log. With nothing streamed yet, the footer stands
    alone (still the real error, not a generic pointer).
    """
    parts = list(visible_parts)
    if pending_thoughts:
        if parts:
            parts.append("\n\n")
        parts.append("💭 ")
        parts.append("".join(pending_thoughts))
    streamed = "".join(parts).strip()
    label = type(exc).__name__
    detail = " ".join(str(exc).split())
    if len(detail) > 300:
        detail = detail[:300] + "…"
    footer = (
        f"⚠️ agent error ({label}): {detail} — see telegram.log"
        if detail
        else f"⚠️ agent error ({label}) — see telegram.log"
    )
    if streamed:
        return f"{streamed}\n\n{footer}"
    return footer


def _maybe_record_throttle_incident(
    *,
    n_edits_429: int,
    n_rollovers: int,
    cur_throttle: float,
    elapsed: float,
) -> bool:
    """If a turn hit Telegram's per-chat edit 429 at/above
    ``THROTTLE_INCIDENT_429_THRESHOLD``, surface a synthetic incident so the
    self-improver can triage + fix the streaming edit cadence. Returns True when
    an incident was recorded. Such a turn SUCCEEDS (no error session), so without
    this hook the SI is blind to the freeze users actually experience (operator
    2026-06-04). Best-effort; never raises.
    """
    if n_edits_429 < THROTTLE_INCIDENT_429_THRESHOLD:
        return False
    _bot._record_bot_incident(
        "telegram_edit_throttled",
        (
            # Stable lead (keeps the triage summary / ledger symptom stable so
            # repeated throttled turns dedup by fingerprint).
            "Telegram per-chat edit rate-limit (HTTP 429) repeatedly throttled a "
            "streaming turn's live message edits, freezing the message on the "
            "placeholder for retry_after stretches (up to ~34s) even though the "
            "turn completed — users read this as the agent being stuck. A "
            "self-hosted telegram-bot-api in --local mode does NOT lift Telegram's "
            "per-chat message edit rate limit, so client-side cadence must respect "
            "it. FIX in src/sharedagents_telegram/bot.py: make the streaming edit "
            "cadence 429-resilient — e.g. raise the EDIT_THROTTLE_S floor under "
            "sustained 429, coalesce stream deltas so fewer edits are issued per "
            "second, honor retry_after with a longer embargo, and/or stop editing "
            "the placeholder once cur_throttle nears EDIT_THROTTLE_MAX_S and let "
            "the final force-paint carry the result. The adaptive throttle "
            "(_RollingMessage edit path) backs off but not aggressively enough to "
            "avoid a visible multi-second freeze. "
            # Per-turn metrics LAST (context for the fixer; intentionally not in
            # the lead so they don't perturb the triage symptom).
            f"[metrics: edit_429={n_edits_429}, rollovers={n_rollovers}, "
            f"final_throttle={cur_throttle:.1f}s, elapsed={elapsed:.0f}s]"
        ),
    )
    return True


def _capabilities_for_chat(chat_id: int, *, trusted_chats: set[int]):
    """Return :class:`acp.schema.ClientCapabilities` for one chat.

    Public-internet inputs default to **text-only**: no fs read, no fs
    write, no terminal. A chat in ``trusted_chats`` gets
    ``fs.readTextFile=True`` so the operator can ask the agent to look at
    local files; we deliberately do NOT enable ``fs.writeTextFile`` or
    ``terminal`` even for trusted chats — those need a more deliberate
    uplift than the comma-separated env var policy.
    """
    # Lazy import — keep module-import cost low. ``acp.schema`` pulls in
    # pydantic + the SDK's full dataclass tree.
    import acp.schema as _acp

    fs = _acp.FileSystemCapabilities(
        read_text_file=chat_id in trusted_chats,
        write_text_file=False,
    )
    return _acp.ClientCapabilities(fs=fs, terminal=False)


# ---------------------------------------------------------------------------
# ``_record_bot_incident`` is the ONE helper here that a test monkeypatches on
# the ``bot`` module (test_telegram_agent_error_detect). Its internal callers
# above therefore reach it through the ``_bot`` module alias (``_bot._record_
# bot_incident``) so the patch is honoured — exactly the module-global liveness
# semantics they had when everything lived in ``bot.py``. This is a plain module
# import (bound to the partially-initialised module during load, resolved at call
# time), so it stays cycle-safe with bot.py's early ``from .bot_helpers import``
# re-export regardless of which module is imported first. Do NOT change to a
# ``from bot import _record_bot_incident`` copy — that would silently ignore the
# patch.
# ---------------------------------------------------------------------------
import sharedagents_telegram.bot as _bot  # noqa: E402
