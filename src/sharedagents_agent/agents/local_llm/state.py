"""Shared module-level symbols for the ``local_llm`` package.

Holds the constants, helper functions, exceptions and ``_RoundState``
that were module-global in the original ``local_llm.py``.  Every mixin
module does ``from .state import *`` so each relocated method sees the
exact same global namespace it had in the monolith — the invariant that
keeps the split behaviour-preserving.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sharedagents_agent.continuation import (
    IDLE_EMPTY_CONTINUE_AFTER_TOOLS,
    IDLE_EMPTY_CONTINUE_BLANK,
    IDLE_EMPTY_CONTINUE_REASONING,
    IDLE_EMPTY_CONTINUE_STARVED,
)
from sharedagents_core.injection_guard import wrap_untrusted

# Per-turn hook-outcome collector. Set by ``run_turn`` at the start
# of every turn (before any hook fires) and read by ``_fire_hook``
# to append a stripped HookOutcome dict per event. Async-safe — a
# ``ContextVar`` is local to each asyncio task, so concurrent turns
# on the same engine instance write to disjoint logs. A ``None``
# value means "no run_turn frame on the stack" — _fire_hook then
# fires without logging (e.g. heartbeat-time or test-time calls).
_HOOK_OUTCOMES_LOG: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "sharedagents_hook_outcomes_log", default=None
)


_LOG = logging.getLogger("sharedagents.agents.local_llm")

from ..engine import (  # noqa: E402
    AgentTurnEvent,
    Engine,
    EngineKind,
    EngineSessionState,
    QueueMessageInjected,
    RoundComplete,
    TextDelta,
    TurnDone,
    TurnError,
    TurnRequest,
    TurnStarted,
    UserMessageInjected,
)


def _now_iso() -> str:
    """Wall-clock ISO timestamp, local TZ. Matches the legacy
    ``acp_bridge._now()`` format so saved-turn timestamps are
    comparable across backends."""
    import datetime as _dt

    return _dt.datetime.now().astimezone().isoformat()


def _namespace_inner_call_id(
    outer_call_id: str,
    raw: Any,
    cache: dict[str, str],
) -> str:
    """Namespace a delegate/subagent's *inner* tool-call id under the
    parent's *outer* call id.

    When a delegate tool (``Agent`` / ``BatchAgent`` / ``delegate_*``)
    drives a child agent turn, the child emits its own ``tool_use`` /
    ``tool_result`` blocks carrying the child engine's RAW call ids
    (``toolu_*`` for Claude, ``chatcmpl-tool-*`` / ``call_*`` for
    OpenAI-compatible servers). Those inner events are surfaced into the
    PARENT turn's event stream (so the TUI mounts native widgets and the
    session log records the delegate's activity).

    If the raw child id is recorded into the parent session **verbatim**,
    it masquerades as one of the parent model's own tool calls: the parent
    log ends up with foreign tool ids the parent model never emitted, the
    delegate's own outer call is left dangling, and the parent turn can no
    longer be reconstructed cleanly. Namespacing every inner id under the
    outer call id (``<outer>.inner.<raw>``) keeps inner events traceable
    to their delegate call and prevents the collision.

    ``cache`` is a per-call dict so the same raw id maps stably to the
    same namespaced id across its ``tool_use`` and ``tool_result`` pair.
    """
    raw_id = str(raw or "")
    if raw_id in cache:
        return cache[raw_id]
    suffix = raw_id or str(len(cache) + 1)
    mapped = f"{outer_call_id}.inner.{suffix}"
    cache[raw_id] = mapped
    return mapped


def _classify(msg: str) -> str:
    """Forward to ``turn_runtime._classify_upstream_error``.

    Module-level shim so we don't have to import inside the hot path.
    """
    try:
        from sharedagents_agent.turn_runtime import _classify_upstream_error

        return _classify_upstream_error(msg or "")
    except Exception:  # noqa: BLE001
        return ""


async def _terminate_and_reap(
    proc: asyncio.subprocess.Process | None,
    stderr_task: asyncio.Task[None] | None,
) -> None:
    """Background cleanup task for a worker subprocess.

    ``terminate()`` schedules SIGTERM
    but doesn't reap the process; if no one awaits ``proc.wait()`` it
    becomes a zombie. This helper drains the stderr task + reaps the
    proc. Scheduled from ``_schedule_background_reap``; self-removed
    from the engine's reaper set via ``add_done_callback``.

    Best-effort: every exception is swallowed so a stuck worker
    doesn't bring down the engine.
    """
    if stderr_task is not None and not stderr_task.done():
        stderr_task.cancel()
        try:
            await stderr_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    if proc is not None:
        try:
            if proc.returncode is None:
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Worker didn't exit in 5s; escalate.
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    await proc.wait()
                except Exception:  # noqa: BLE001
                    pass
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001
            pass


async def _drain_stderr(
    stream: asyncio.StreamReader,
    sharedagents_sid: str,
    turn_id: str,
) -> None:
    """Continuously read worker stderr and surface it via the logger.

    ``stderr=asyncio.subprocess.PIPE``
    is set in the legacy spawn helper; if no one drains it, a noisy
    worker can fill its PIPE and block on write, hanging the turn.

    This coroutine reads stderr line-by-line and emits each as a
    structured log line. Cancellation (the caller calls
    ``stderr_task.cancel()`` when run_turn exits) is handled silently.
    """
    try:
        while True:
            line = await stream.readline()
            if not line:
                return
            try:
                text = line.decode("utf-8", "replace").rstrip()
            except Exception:  # noqa: BLE001
                continue
            if text:
                _LOG.info(
                    "WORKER_STDERR sid=%s turn=%s line=%s",
                    sharedagents_sid[:12],
                    turn_id[:12],
                    text[:500],
                )
    except asyncio.CancelledError:
        return
    except Exception:  # noqa: BLE001
        # Best-effort; don't propagate.
        return


# =====================================================================
# Constants the local-LLM path uses today. Promoted here (rather than
# re-imported from acp_bridge / turn_runtime / worker) so the new
# module is self-contained for review. Values stay in lockstep with
# the legacy constants — diff-check during step-2 implementation.
# =====================================================================

# Maximum consecutive rounds that produce only read-only *research* tool calls
# (Grep/Glob/Read / Bash inspection / WebSearch / WebFetch) before the engine
# intervenes. Agent(list) / BashOutput polls are waits, not research — they
# do not advance this streak.
# At SOFT_LIMIT the engine injects a synthesize nudge; at HARD_LIMIT it
# requests a graceful finalize. Defaults used to be 8/12, which fired on
# native jarvis/explore/general turns (Agent list polling, short evidence
# sweeps) and burned extra local-LLM rounds. 0 disables either threshold.
_MAX_READONLY_SOFT_LIMIT: int = int(os.environ.get("SHAREDAGENTS_READONLY_SOFT_LIMIT", "32") or "32")
_MAX_READONLY_HARD_LIMIT: int = int(os.environ.get("SHAREDAGENTS_READONLY_HARD_LIMIT", "48") or "48")

# OpenCode max-steps handoff — injected on the final round with tools disabled.
MAX_STEPS_HANDOFF_USER: str = (
    "[sharedagents max-steps] Step budget exhausted. Tools are disabled for this "
    "final round. Respond with a concise text summary of: (1) work completed "
    "with evidence (paths, commands, outcomes), (2) remaining tasks, "
    "(3) any blockers. Do not call tools."
)
BUDGET_HANDOFF_USER: str = (
    "[sharedagents budget] Cost budget exhausted. Tools are disabled for this final "
    "round. Respond with a concise summary of: (1) work completed with evidence "
    "(paths, commands, outcomes), (2) remaining tasks, (3) blockers. The user "
    "can authorize more budget to continue. Do not call tools."
)
_DOOM_LOOP_REPEAT_THRESHOLD: int = 3


class _NoSummaryCompaction(Exception):
    """Internal sentinel: LLM compaction found nothing useful to summarize, so
    the caller should fall back to mechanical compaction WITHOUT arming the
    summarizer-failure cooldown (this is not an endpoint failure)."""


class _CompactionAborted(Exception):
    """Internal sentinel: a hard abort (``_cancelled[sid]``) landed while the
    engine-side compaction summariser HTTP call was in flight. That call is
    in-process — NOT the worker subprocess a force-abort SIGKILLs — so without
    an explicit cancellation race a ``/abort`` during a large compaction could
    not land until the multi-hundred-K-token summary finished, stranding the
    operator for minutes. The caller abandons the summary
    and lets the round loop's cancel checkpoint end the turn."""


# Tool names that are purely read-only search/inspection — no state mutation.
# Must match the lowercase names the agent uses (Grep, Glob, Read, etc.).
_READONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "grep",
        "glob",
        "read",
        "websearch",
        "webfetch",
        "list",
    }
)
# Permission-read-only calls that are orchestration waits, not evidence
# gathering. A round of only these must not advance the read-only research
# budget — native jarvis/explore parents poll Agent(list) while children run.
_READONLY_WAIT_TOOLS: frozenset[str] = frozenset({"bashoutput", "monitor"})
_AGENT_WAIT_ACTIONS: frozenset[str] = frozenset({"list"})
# Control actions still need a follow-up model round (lineage / steer / abort).
_AGENT_CONTROL_ACTIONS: frozenset[str] = frozenset({"list", "steer", "abort"})
# Side-effect bookkeeping that may ride along with a background dispatch.
_BOOKKEEPING_TOOLS: frozenset[str] = frozenset({"todowrite", "todo_write"})


def _is_readonly_wait_call(call: dict[str, Any]) -> bool:
    """True for Agent(list) / BashOutput-family polls (not research reads)."""
    if not isinstance(call, dict):
        return False
    name = str(call.get("name") or "").strip().lower()
    if name in _READONLY_WAIT_TOOLS:
        return True
    if name != "agent":
        return False
    args = call.get("arguments")
    if not isinstance(args, dict):
        return False
    return str(args.get("action") or "").strip().lower() in _AGENT_WAIT_ACTIONS


def _readonly_budget_class(
    calls: list[dict[str, Any]],
    *,
    all_read_only: bool,
) -> str:
    """Classify a round for the read-only research budget.

    ``wait`` — every call is an Agent(list) / BashOutput-family poll; leave
    the streak unchanged. ``research`` — every call is read-only evidence
    gathering. ``other`` — empty, mutating, or mixed: reset the streak.
    """
    if not calls:
        return "other"
    if all_read_only and all(_is_readonly_wait_call(c) for c in calls):
        return "wait"
    if all_read_only:
        return "research"
    return "other"


def _tool_flag_true(value: Any) -> bool:
    """Truthy tool-arg flag (bool, 1, or common string spellings)."""
    if value is True:
        return True
    if isinstance(value, (int, float)) and value == 1:
        return True
    return bool(isinstance(value, str) and value.strip().lower() in ("true", "1", "yes", "on"))


def _is_background_job_handle(value: Any) -> bool:
    """True when a result row was stamped as a detached-job handle.

    Provenance is the runner's ``background_handle`` flag (copied onto
    ``bg_handle``), never inferred from child text. A lookalike JSON body
    must not end the parent turn.
    """
    return isinstance(value, dict) and value.get("bg_handle") is True


def _call_is_background_agent(name: str, args: dict[str, Any]) -> bool:
    """True for Agent/delegate_* calls that only start a detached job."""
    n = str(name or "").strip().lower()
    if n != "agent" and not n.startswith("delegate_"):
        return False
    if str(args.get("action") or "").strip().lower() in _AGENT_CONTROL_ACTIONS:
        return False
    if _tool_flag_true(args.get("background")) or _tool_flag_true(
        args.get("return_background"),
    ):
        return True
    runs = args.get("runs")
    if isinstance(runs, list) and runs:
        flags = []
        for item in runs:
            if not isinstance(item, dict):
                return False
            flags.append(_tool_flag_true(item.get("background")) or _tool_flag_true(item.get("return_background")))
        return bool(flags) and all(flags)
    return False


def _round_is_background_delegate_wait(
    calls: list[dict[str, Any]],
    results: list[dict[str, Any]] | None = None,
) -> bool:
    """True when this round only started background Agent jobs (+ TodoWrite).

    After such a round the parent must emit ``turn_ended`` (session_state
    ``waiting``) instead of another full prefill just to recap "job started".
    Observed in a jarvis session: ``Agent(
    background=true)`` at 11:58:35 then a ~2 min 341k-token recap round while
    the WebUI stayed streaming / Stop.

    Execution that actually returned a job handle is treated as a wait even
    when the model argued ``background=false`` — that mismatch used to feed
    the handle back into another Agent round and loop.
    """
    ok_by_id: dict[str, Any] = {}
    handle_by_id: dict[str, bool] = {}
    for row in results or []:
        if not isinstance(row, dict):
            continue
        rid = row.get("id") or row.get("call_id")
        if rid is None:
            continue
        rid = str(rid)
        if row.get("denied"):
            ok_by_id[rid] = False
        elif "ok" in row:
            ok_by_id[rid] = row.get("ok")
        handle_by_id[rid] = handle_by_id.get(rid, False) or (_is_background_job_handle(row))
    started = 0
    for call in calls or []:
        if not isinstance(call, dict):
            return False
        name = str(call.get("name") or "")
        key = name.strip().lower()
        args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        if key in _BOOKKEEPING_TOOLS:
            continue
        cid = str(call.get("id") or "")
        from_args = _call_is_background_agent(name, args)
        from_handle = bool(cid and handle_by_id.get(cid))
        if not from_args and not from_handle:
            return False
        if cid and ok_by_id.get(cid) is False:
            return False
        started += 1
    return started > 0


def _local_worker_idle_timeout_s() -> float:
    """Seconds to wait for the local worker's next JSONL event.

    This sits below Telegram's default 300s no-progress watchdog so a
    wedged orchestrator worker is reported by the engine itself first.
    """
    try:
        return float(os.environ.get("SHAREDAGENTS_LOCAL_WORKER_IDLE_TIMEOUT_S", "240") or "240")
    except ValueError:
        return 240.0


async def _next_worker_event(
    event_iter: AsyncIterator[dict[str, Any]],
    timeout_s: float,
) -> dict[str, Any]:
    if timeout_s <= 0:
        return await event_iter.__anext__()
    return await asyncio.wait_for(event_iter.__anext__(), timeout=timeout_s)


# Engine-side loop-detector knobs. The stride and ENABLED flag both
# defer to the same env vars that turn_runtime.py uses, so an operator
# setting SHAREDAGENTS_LOOP_DETECT=0 disables every detection path. The
# historical LOOP_NGRAM_LEN / LOOP_REPEAT_THRESHOLD / LOOP_WINDOW_CHARS
# mirrors were silently dead: _detect_loops_layered swallows them via
# **_kwargs and the real values live on turn_runtime.LOOP_* (env-tunable).
try:
    _stride_raw = os.environ.get("SHAREDAGENTS_LOOP_CHECK_STRIDE", "300") or "300"
    LOOP_CHECK_STRIDE: int = int(_stride_raw)
except ValueError:
    LOOP_CHECK_STRIDE = 300
LOOP_DETECT_ENABLED: bool = os.environ.get(
    "SHAREDAGENTS_LOOP_DETECT",
    "1",
) not in ("0", "false", "False")

# Mirror of worker.py defaults.
WORKER_READ_TIMEOUT_S: float = 120.0  # per-chunk SSE read ceiling
PREFILL_FLOOR_MS: float = 50.0  # min plausible prefill duration
LOOP_CHARS_PER_TOKEN_MIN: float = 0.5  # token-loop ratio guard floor — Qwen3.6 model row overrides to 0.8
LOOP_RATIO_MIN_TOKENS: int = 500  # min completion tokens before the ratio check fires

# NOTE: the proactive gate is now OpenCode's ``isOverflow`` (billed total >=
# ``context - max_output``, see ``local_llm/compaction.py``). The three
# settings below (``compact_at_frac``, occupancy fraction, min-turns) no longer
# move it; they remain only for config/env compatibility.
# Agent-level compact_at_frac fallback when the config field is missing.
# The always-on occupancy trigger is ``_COMPACT_OCCUPANCY_FRAC`` (75% of
# the live context window), independent of this value.
DEFAULT_COMPACT_AT_FRAC: float = 0.75

# OpenCode-style min turns between proactive compacts (thrash guard).
# Overflow / 75% occupancy / force=True always bypass. 0 disables.
_COMPACT_MIN_TURNS_BETWEEN: int = int(os.environ.get("SHAREDAGENTS_COMPACT_MIN_TURNS", "3") or "3")

# Always-on occupancy trigger: compact (hidden ``compaction`` agent, not
# the PR-blurb ``summary`` agent) once prompt tokens reach this fraction
# of the live context window. 75% of 131072 ≈ 98k; 75% of 262144 = 196608.
# Override via env; do not add per-window absolute caps.
_COMPACT_OCCUPANCY_FRAC: float = float(os.environ.get("SHAREDAGENTS_COMPACT_OCCUPANCY_FRAC", "0.75") or "0.75")

# Loop guard: a PROACTIVE squeeze that frees fewer than this many characters
# made no useful progress. Below it, ``_compaction_target`` disarms further
# proactive compaction for the session until the working set grows again, so
# compaction can never loop on content it can't shrink (~5k tokens of slack).
_MIN_COMPACTION_CHARS_FREED: int = 16_000

# How many times a blocking ``Stop`` hook may re-enter the round loop
# within one turn before the engine ends it anyway. OpenCode parity:
# DEFAULT 0 — natural stops end the turn; the engine waits for the next
# explicit turn (no AUTO_COMPLETE / Stop-hook re-drive). Override via
# ``cfg.max_stop_continuations`` when an operator wants re-entry.
# Round-cap handoffs NEVER continue regardless of this knob.
_MAX_STOP_CONTINUATIONS: int = 0

# GOAL BLOCKED sentinel — the model explicitly reporting it cannot make
# further progress on the active goal. Compiled once at module load rather
# than per Stop-hook iteration.
# It ends the current turn (no further in-turn nudges); it does NOT change goal
# state — the goal evaluator reads it as evidence (Claude Code semantics).
_GOAL_BLOCKED_RE = re.compile(r"GOAL\s+BLOCKED\s*:", re.IGNORECASE)

# Per-turn cap on the reactive max-tokens fallback (the
# ``COMPACTION_REACTIVE_MAX_TOKENS_FALLBACK`` path in
# ``_reactive_compact_on_overflow``). When ``compact_messages`` can't
# squeeze the live prompt any further, the recovery shrinks
# ``max_tokens`` so input + output fits and signals ONE retry. If that
# retry STILL overflows (e.g. the prompt keeps accreting per round), the
# fallback would otherwise re-fire every round — each time shrinking
# ``max_tokens`` a little more and re-spawning the worker — storming the
# upstream for the whole round budget (observed: 81 fallbacks
# in ~60s on one session, ``new_max_tokens`` marching 16384→…→<256). Bound
# it so the runtime surfaces the overflow error after at most this many
# fallback retries instead of looping.
_MAX_REACTIVE_MAXTOK_FALLBACKS: int = 2
# Cap consecutive overflow recoveries, including paid LLM summaries, so a
# still-oversized prompt cannot thrash the upstream. A successful round
# resets the consecutive cap for long tasks. Two retries gives the
# heal-session-context_window + capped-output-budget path room to land
# after a stale 256k→65k model switch (Jul 30 plan/build overflows).
_MAX_REACTIVE_COMPACT_RETRIES: int = 2

# Per-turn cap on TRANSIENT upstream-error retries — the thinking-inactivity
# ReadTimeout (worker.py 30s no-token watchdog) and mid-stream connection
# drops (vLLM closing the socket, "peer closed connection", connection
# reset). These are transient: a small model that goes silent for 30s or an
# endpoint that drops one stream usually recovers on a fresh round. We
# re-send the SAME round (no compaction — this is not a context problem) up
# to this many times with a short backoff before surfacing the error, so a
# complex local-model task survives the inevitable hiccups instead of dying
# on the first stall. A persistently-wedged endpoint still surfaces after
# the cap. Overridable via ``cfg.max_transient_retries``.
_MAX_TRANSIENT_RETRIES: int = 2


def _is_transient_upstream(err: str) -> bool:
    """True for upstream errors that a plain round-retry can recover:
    the thinking-inactivity ReadTimeout and mid-stream connection drops.
    Deliberately NARROW — context-overflow / 4xx / auth errors are NOT
    transient and must not be retried blindly.
    """
    e = (err or "").lower()
    needles = (
        "readtimeout",
        "read timeout",
        "timed out",
        "timeout",
        "peer closed connection",
        "incomplete chunked",
        "connection reset",
        "connection aborted",
        "remoteprotocolerror",
        "server disconnected",
        "connectionerror",
        "connection error",
    )
    return any(n in e for n in needles)


def _is_endpoint_down(err: str) -> bool:
    """True when the PRIMARY upstream looks unreachable (not a mid-stream
    hiccup): connection refused, DNS failure, all-connection-attempts-failed,
    name resolution, or a local vLLM router 502/503 (all replicas down or
    at capacity). These warrant failing over to the LLM fallback model
    (``role == 'fallback'``) rather than re-sending to the same dead endpoint.

    HTTP 502/503 from the local router is exhaustion, not a transient
    hiccup. ``llm_fallback_used`` still fires only once per turn so a
    cloud backup that also 503s does not ping-pong.
    """
    e = (err or "").lower()
    needles = (
        "connection refused",
        "connectionrefused",
        "errno 111",
        "all connection attempts failed",
        "connecterror",
        "name or service not known",
        "nodename nor servname",
        "temporary failure in name resolution",
        "no route to host",
        "network is unreachable",
        "failed to establish a new connection",
        "http 502",
        "http 503",
        "upstream_unavailable",
        "capacity_exceeded",
        "rate_limit_exceeded",
        "at capacity",
        "all local replicas",
    )
    return any(n in e for n in needles)


# Continuation-nudge budget. Small models — particularly qwen3.6-class —
# sometimes drop EOS after a tool chain or a reasoning-only block without
# writing any user-visible summary, so the operator sees a silent idle
# end. We re-prompt the model up to ``MAX_NUDGES_PER_TURN`` times asking
# it to either finish the task or summarize. Hard-capped so a model that
# won't comply can't loop the agent. Overridable per-session via
# ``cfg.max_nudges_per_turn`` so an operator can extend (or tighten) the
# cap for a specific agent / mode.
#: LIFETIME nudges per turn. The consecutive-idle bound
#: (``MAX_IDLE_EMPTY_CONTINUES``, 3) is what stops a loop; this is the hard
#: ceiling that stops an alternating productive/empty pattern. Raised 3 -> 6
#: because three starved rounds consumed the entire budget before the model
#: recovered, so a later relapse in the same turn got nothing.
MAX_NUDGES_PER_TURN: int = 6
# Final-round content shorter than this (after .strip()) is treated as
# "the model said nothing" by phantom-round suppression. Behavioral
# silent-nudge gates in ``classify_turn_end`` no longer use this floor —
# they require truly empty content (``produced_chars == 0``) so a
# complete terse answer like "pong" / "ok" / "done." is never re-prompted
# (Telegram topic 1842 PING→PONG duplicate). Kept at 10 for the phantom-
# round heuristic only (a 1-token empty stop after the turn already has
# a real answer).
NUDGE_MIN_TEXT_CHARS: int = 10
NUDGE_PROMPT_BODY: str = IDLE_EMPTY_CONTINUE_AFTER_TOOLS
# Continuation-nudge body for the mid-sentence-cutoff variant. Distinct
# wording from ``NUDGE_PROMPT_BODY`` (silent-EOS-after-tools) — that one
# tells the model "you said nothing, write a final answer"; this one
# tells the model "you wrote prose then paused for a tool, your prose
# ended mid-sentence at <tail>, RESUME from there."
CUTOFF_NUDGE_PROMPT_BODY: str = (
    "[sharedagents continuation] Your previous response was cut mid-sentence "
    "at: …{tail}\n\nResume writing from that exact point — finish the "
    "sentence first, then continue the section you were in. Do not "
    "restart the report or re-summarize what you've already written."
)
# Continuation-nudge body for the colon-cliffhanger variant. The model
# emitted prose that ended on a colon (which usually precedes a tool
# call) but produced no tool_calls. Witnessed with
# a vLLM Qwen3.6-27B FP8 MTP route + qwen3_coder tool-call parser: the round
# stops at "the routine stores `tif`:" with ``finish_reason=stop`` and
# the operator sees a dangling colon as the "final answer".
COLON_CLIFFHANGER_NUDGE_PROMPT_BODY: str = (
    "[sharedagents continuation] Your previous response ended on a colon "
    "promising more content or a tool call, but nothing followed: …{tail}"
    "\n\nEither emit the tool call you were about to issue, or finish "
    "the sentence and write the user-facing answer. Do not restart or "
    "re-summarize."
)
STREAM_TRUNCATED_NUDGE_PROMPT_BODY: str = (
    "[sharedagents continuation] Token budget ran out (finish_reason=length) "
    "while you were still writing — the operator saw a cut-off answer. "
    "The last visible text ended at: …{tail}\n\nResume from that exact "
    "point: finish the current sentence, then complete the section. Do "
    "not restart or re-summarize what you already wrote."
)
# Continuation-nudge body for a tool call cut off by the output cap. There is
# no sentence to resume: the call itself never ran. ``{tail}`` names the
# call(s) and how many argument characters were written before the cut, so the
# model can size smaller calls. Observed: told to "resume from that
# exact point", the model re-sent the same `write` five times.
TOOL_CALL_TRUNCATED_NUDGE_PROMPT_BODY: str = (
    "[sharedagents continuation] The output-token limit (finish_reason=length) "
    "cut off your response while you were still writing tool-call "
    "arguments: {tail}. Those calls never ran, so nothing they would have "
    "done has happened. Re-sending the same call may be cut off again: the "
    "limit covers your whole response, reasoning included. Split the work "
    "into smaller calls, one per response — for a large file, write the "
    "first part, then add the rest with further edit calls — keeping each "
    "call well under the size that was cut off. Do not restart the task or "
    "repeat completed steps."
)
TRANSPORT_ERROR_NUDGE_PROMPT_BODY: str = (
    "[sharedagents continuation] The upstream stream dropped with a transport "
    "error before a terminal finish — the operator saw a cut-off answer. "
    "The last visible text ended at: …{tail}\n\nResume from that exact "
    "point: finish the current sentence, then complete the section. Do "
    "not restart or re-summarize what you already wrote."
)
NO_TERMINAL_NUDGE_PROMPT_BODY: str = (
    "[sharedagents continuation] The stream ended without a terminal "
    "finish_reason — the operator saw a cut-off answer. The last visible "
    "text ended at: …{tail}\n\nResume from that exact point: finish the "
    "current sentence, then complete the section. Do not restart or "
    "re-summarize what you already wrote."
)
BACKGROUND_DELEGATE_EMPTY_NUDGE_PROMPT_BODY: str = (
    "[sharedagents continuation] You dispatched background Agent job(s) and "
    "stopped without writing anything user-visible. The operator only "
    "sees content tokens — a bare job handle is not an answer. Write "
    "one short status line now: who you spawned, what they are doing, "
    "and that you are waiting. Do not start a new chain of thought and "
    "do not recap the whole task."
)
# Continuation-nudge body for the reasoning-only variant. The model produced
# reasoning tokens but ZERO user-visible content and no tool call on the first
# round (``finish_reason=stop``) — the operator watched it think and got
# nothing back (session <session-id>: reasoning_tokens 200-470,
# content_chars=2). Same "silent" nudge machinery as NUDGE_PROMPT_BODY but
# worded for the no-tools-yet case: the model never started a tool chain, it
# just thought and stopped.
REASONING_ONLY_NUDGE_PROMPT_BODY: str = IDLE_EMPTY_CONTINUE_REASONING
# Continuation-nudge body for THINKING-STARVATION truncation: the FINAL round
# exhausted its output budget (``finish_reason=length``) while still reasoning
# and was killed before writing ANY visible content. Unlike
# STREAM_TRUNCATED_NUDGE_PROMPT_BODY there is no partial answer to resume from
# — the model never reached the point of writing. Observed (treasury
# rollover): final round 38,610 reasoning chars, content_chars=0; the turn
# ended "complete" with no nudge and the operator had to re-prompt by hand.
THINKING_STARVED_NUDGE_PROMPT_BODY: str = IDLE_EMPTY_CONTINUE_STARVED
# Continuation-nudge body for a completely blank terminal round — no content,
# no reasoning, no tool call (e.g. an immediate EOS, or a dangling
# ``finish_reason="tool_calls"`` label whose call JSON never parsed). Operator
# requirement 2026-08-14: no prompt may finish without an output, so this
# shape resumes instead of ending the turn silently.
EMPTY_RESPONSE_NUDGE_PROMPT_BODY: str = IDLE_EMPTY_CONTINUE_BLANK
# One-shot "write the answer now" injection fired when a graceful finalize
# (read-only research budget / sustained-zero-progress) is about to end a
# turn that has produced ZERO user-visible output. The finalize paths
# previously broke the round loop silently, so the operator watched tools
# run and then got nothing. One extra tool-less answer round per turn, then
# the finalize proceeds regardless of the outcome.
FINAL_ANSWER_NUDGE_BODY: str = (
    "[sharedagents continuation] This turn is ending but you have produced NO "
    "user-visible output yet — the user has seen nothing at all. Do not "
    "call any more tools. Using only what you already learned this turn, "
    "write the answer to the user now: state what you found or did, and "
    "what remains. This is the final round."
)


def _detect_stream_truncation(emitted: str, full: str) -> str | None:
    """Return missing suffix when ``full`` extends ``emitted`` (stream gap)."""
    from sharedagents_agent.delivery import detect_stream_truncation as _dst

    return _dst(emitted, full)


def _is_phantom_round_metrics(
    metrics: dict[str, Any] | None,
    turn_content_parts: list[str],
) -> bool:
    """Suppress empty 1-token ``stop`` rounds after the turn already has an answer."""
    if not metrics:
        return False
    try:
        ct = int(metrics.get("completion_tokens") or 0)
        cc = int(metrics.get("content_chars") or 0)
    except (TypeError, ValueError):
        return False
    fr = str(metrics.get("finish_reason") or "")
    turn_chars = len("".join(turn_content_parts).strip())
    return fr == "stop" and ct <= 1 and cc == 0 and turn_chars >= NUDGE_MIN_TEXT_CHARS


def _format_runtime_warning(event: str, **fields: Any) -> str:
    """Format a runtime-warning line for the TUI's visible message stream.

    Surfaces loop / repetition / premature-finish events that the
    engine catches and recovers from. Without this, the recovery
    machinery (loop detector → distill → force-terminate, nudge
    detectors) is silent on the TUI: turns either appear to die or
    appear to continue normally, and the operator has to grep
    ``gateway.log`` to see what actually fired. Each line is one
    notification, prefixed with ``⚠`` for hard interventions
    (loops/force-terminate) and ``↻`` for soft recoveries
    (distill/nudges). Newlines top-and-tail isolate the warning from
    surrounding model output in the rendered bubble.

    ``event`` is a short kebab-case label (``loop-detected``,
    ``loop-distill``, ``loop-force-terminate``, ``nudge``). ``fields``
    are stringified ``k=v`` pairs; values longer than 80 chars are
    truncated.
    """
    soft = event in (
        "loop-distill",
        "nudge",
        "progress-nudge",
        "read-only-synthesize",
    )
    glyph = "↻" if soft else "⚠"
    parts: list[str] = [f"{glyph} runtime: {event}"]
    for k, v in fields.items():
        if v is None or v == "":
            continue
        sv = str(v)
        if len(sv) > 80:
            sv = sv[:77] + "…"
        parts.append(f"{k}={sv}")
    body = " · ".join(parts)
    return f"\n\n{body}\n\n"


# Per-turn cap on how many characters of a prior turn we embed into a
# direct-dispatch delegate preamble. Generous — the only thing this
# guards against is one pathological turn (e.g. a multi-MB tool dump)
# blowing the delegate's prompt window.
_DIRECT_PREAMBLE_PER_TURN_CHARS = 200_000

# Total preamble budget per delegate engine. Sized to leave headroom under
# each CLI's model context window (claude ~200K tok; codex ~272K input;
# antigravity/gemini ~1M) for the agent's OWN system prompt, tools, the
# user's request, and the response — at ~4 chars/token. Without a total cap
# a long orchestrator session could hand the CLI a preamble bigger than the
# model context window → a "prompt is too long" / "input context is too
# large" failure. Override globally via SHAREDAGENTS_DELEGATE_MAX_PREAMBLE_CHARS.
_DIRECT_PREAMBLE_TOTAL_CHARS = {
    "delegate_claude": 300_000,  # ~75K tok, well under the 200K window
    "delegate_codex": 400_000,  # ~100K tok, under ~272K input
    "delegate_antigravity": 800_000,  # ~200K tok, under ~1M
}


def _direct_preamble_total_budget(tool_name: str) -> int:
    """Total char budget for the assembled delegate preamble."""
    raw = os.environ.get("SHAREDAGENTS_DELEGATE_MAX_PREAMBLE_CHARS")
    if raw:
        try:
            return max(2_000, int(raw))
        except ValueError:
            pass
    return _DIRECT_PREAMBLE_TOTAL_CHARS.get(tool_name, 300_000)


def _collapse_leading_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge consecutive leading ``system`` messages into a single one.

    sharedagents injects below-cache-boundary preamble blocks (skill
    auto-invoke, memory recall, pinned facts, todos, compaction summary)
    as their own ``role="system"`` messages right after the main system
    prompt. Tolerant backends (DeepSeek, OpenAI) accept several system
    messages; some strict chat templates — notably a local qwen3.x vLLM
    server — reject a second one with ``HTTP 400: "System
    message must be at the beginning."`` and the whole turn dies before
    the model ever runs.

    Collapsing the contiguous leading run into one system message is
    semantically identical (same content, same order) and makes the
    preamble survive on strict backends. No-op in the common case (a
    single system prompt, no active preamble), so caching is unaffected
    unless a preamble block is actually present.
    """
    if len(messages) < 2 or not isinstance(messages[0], dict):
        return messages
    if messages[0].get("role") != "system":
        return messages
    k = 1
    while k < len(messages) and isinstance(messages[k], dict) and messages[k].get("role") == "system":
        k += 1
    if k < 2:
        return messages
    merged = "\n\n".join(str(m.get("content") or "") for m in messages[:k])
    return [{"role": "system", "content": merged}, *messages[k:]]


def _delegate_resume_available(session: Any, tool_name: str) -> bool:
    """True iff a resumable child CLI session exists for this parent+kind.

    The preamble's first-vs-resume decision MUST track the CLI's ACTUAL
    ``--resume`` state, not merely the parent session's turn history. If a
    prior same-kind delegate turn exists but the child never captured an
    ``engine_session_id`` (a failed / aborted first call, or a stale resume
    that cold-started), the CLI starts fresh — and the preamble must then
    forward ALL prior turns, not just the post-delegate slice, or the
    delegate loses every pre-delegate turn. Best-effort: any error → treat
    as not resumable (forward everything; safe, at worst slightly redundant
    with the CLI's own state).
    """
    short = {
        "delegate_claude": "claude",
        "delegate_codex": "codex",
        "delegate_antigravity": "antigravity",
    }.get(tool_name)
    if not short:
        return False
    parent_sid = str(getattr(session, "hash", "") or "")
    if not parent_sid:
        return False
    try:
        from .. import factory

        child = factory.load_agent(short)
        sidecar = child.delegate_sidecar_path(parent_sid, child.engine.kind)
        if not sidecar.exists():
            return False
        import json as _json

        data = _json.loads(sidecar.read_text(encoding="utf-8"))
        if str(data.get("engine_session_id") or ""):
            return True
        # Older sidecars stored only the parent→child mapping; consult the
        # child engine's own state file for the resume id.
        child_sid = str(data.get("sharedagents_child_session_id") or data.get("sharedagents_session_id") or "")
        if not child_sid:
            return False
        state = child.engine.read_state(child.sessions_dir, child_sid)
        return bool(str(getattr(state, "engine_session_id", "") or ""))
    except Exception:  # noqa: BLE001
        return False


def _build_direct_delegate_preamble(session: Any, tool_name: str, *, resumable: bool = True) -> str:
    """Render prior sharedagents turns as an ``<sharedagents-context>`` preamble
    for a slash-command delegate (/claude /codex /gemini).

    The local model is NOT in the loop for slash-command delegation, so
    the delegate would otherwise receive only the bare user text with
    zero conversation context (the TUI builds this preamble client-side,
    but Telegram and other frontends did not — so `/claude is this
    correct` reached Claude with nothing to resolve "this" against).

    Forward exactly the turns the delegate hasn't seen: on the first
    call of its kind, every prior turn; on a resume, only the turns
    since the last same-kind delegate call (the CLI's own ``--resume``
    already carries everything up to that point). Cross-kind delegate
    turns ARE forwarded — the other CLI has no shared resume state.
    Mirrors :func:`sharedagents_tui.app._build_delegate_preamble`.
    """
    try:
        turns = list(getattr(session, "turns", []) or [])
    except Exception:  # noqa: BLE001
        return ""
    marker = f"__SHAREDAGENTS_DELEGATE__:{tool_name}:"
    # Find the last COMPLETED same-kind delegate turn. CRITICAL: skip
    # in-progress turns. The Agent flushes the CURRENT delegate turn
    # (user=sentinel) into ``session.turns`` on the ``ToolCallStarted``
    # event — which the direct-dispatch path emits BEFORE this preamble is
    # built — so a naive scan matches the current turn as ``last_same``,
    # leaves ``unsynced`` empty, and forwards ZERO context. The delegate
    # then never learns what the user has been doing with the local model
    # (and ``is_first_call`` wrongly flips False). Only prior, completed
    # delegate turns may anchor ``last_same``.
    last_same = -1
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or t.get("in_progress"):
            continue
        if str(t.get("user") or "").startswith(marker):
            last_same = i
    # The "resume" shortcut (forward only turns SINCE the last delegate,
    # trusting the CLI's --resume to carry the rest) is only valid when the
    # CLI actually HAS a resumable session. Otherwise the CLI cold-starts
    # and we must forward ALL prior turns. See _delegate_resume_available.
    if not resumable:
        last_same = -1
    is_first_call = last_same < 0
    unsynced = turns[last_same + 1 :]

    # One block per unsynced turn; trimmed to the total budget below.
    blocks: list[str] = []
    cap = _DIRECT_PREAMBLE_PER_TURN_CHARS
    for t in unsynced:
        if not isinstance(t, dict) or t.get("in_progress"):
            continue
        from sharedagents_core.attachment_context import model_user

        u = str(model_user(t.get("user"), t.get("attachments")) or "").strip()
        # Drop the raw delegate sentinel from any prior delegate turn —
        # it's noise; the assistant reply content is what matters.
        if u.startswith("__SHAREDAGENTS_DELEGATE__:"):
            u = ""
        c = str(t.get("content") or "").strip()
        if not u and not c:
            continue
        parts: list[str] = []
        if u:
            parts.append(f"[user] {u[:cap]}" + (" …[truncated]" if len(u) > cap else ""))
        if c:
            parts.append(f"[assistant] {c[:cap]}" + (" …[truncated]" if len(c) > cap else ""))
        blocks.append("\n".join(parts))
    if not blocks:
        return ""

    # Respect the delegate CLI's context window: keep the most RECENT
    # turns, drop oldest until under budget, and mark the elision so the
    # delegate knows history was trimmed rather than silently losing it.
    budget = _direct_preamble_total_budget(tool_name)
    kept: list[str] = []
    total = 0
    for blk in reversed(blocks):  # newest first
        blk_len = len(blk) + 2  # +2 for the "\n\n" join
        if kept and total + blk_len > budget:
            break
        kept.append(blk)
        total += blk_len
    dropped = len(blocks) - len(kept)
    kept.reverse()  # restore chronological order
    body = "\n\n".join(kept)
    if dropped:
        body = (
            f"[… {dropped} older turn(s) omitted to fit the delegate "
            f"context budget; the most recent context follows …]\n\n" + body
        )

    if is_first_call:
        header = (
            "You were invoked from inside an sharedagents session. The user has "
            "been talking to a local model (sharedagents) before this. Use the "
            'conversation below as context — when the user says "this" or '
            '"that", they mean the topic of the prior exchange.'
        )
    else:
        header = (
            "Additional sharedagents turns since your last invocation (the user "
            "kept talking to the local model in between). Use them as context."
        )
    return (
        f"<sharedagents-context>\n{header}\n\n"
        "=== sharedagents conversation history ===\n" + body.rstrip() + "\n=== end of conversation history ===\n"
        "</sharedagents-context>\n\n"
        "The user's request follows.\n\n"
    )


class _RoundState:
    """Mutable state accumulated during one worker round.

    Reference: the local variables inside ``CoreBackend.prompt``'s
    inner loop (``round_thinking`` / ``round_content`` /
    ``round_tool_calls`` / ``loop_detected*`` / etc., acp_bridge.py
    L3047–3060).

    Lives only inside :meth:`LocalLLMEngine._consume_worker_events`
    and the dispatch helpers it hands off to. Dropped at end-of-round.
    """

    def __init__(self) -> None:
        self.thinking_buf: list[str] = []
        self.content_buf: list[str] = []
        self.pending_tool_calls: list[dict[str, Any]] = []
        # A Bedrock worker's opaque / signed reasoning for this round's tool
        # calls (``tool_calls`` event ``reasoning_replay``): carried onto the
        # assistant round message so the next round hands it back verbatim.
        self.reasoning_replay: dict[str, Any] | None = None
        self.metrics: dict[str, Any] | None = None
        self.loop_detected: bool = False
        self.loop_tier: str | None = None
        self.loop_evidence: str = ""
        # Full diagnostic payload from a worker-side ``tool_args_loop``
        # event (tier, evidence, args_chars, checked_at, tool_name,
        # tool_index, channel). Persisted into ``session.round_metrics``
        # at round end so an aborted round is still triageable from the
        # session JSON. ``None`` when no in-flight detector fired.
        self.loop_event: dict[str, Any] | None = None
        self.thinking_loop_check_chars: int = 0
        self.content_loop_check_chars: int = 0
        # Per-channel ambiguous-chrome sustain state. Maps each
        # channel ("thinking" / "content") to a dict of
        # ``evidence.strip() -> first-seen monotonic time``. Consumed
        # by ``turn_runtime._should_fire_with_sustain`` so a brief
        # ``--------`` / ``========`` / ``****`` run that finishes in
        # under :data:`turn_runtime.LOOP_AMBIGUOUS_SUSTAIN_SECONDS`
        # does NOT abort the round — only sustained collapse does.
        # Unambiguous heads (``!`` / ``?`` / alphanumeric / paraphrase
        # rotation) skip the gate inside the helper.
        #
        # Scope: per-round, dropped when the ``_RoundState`` is
        # reconstructed at the next round boundary. That's correct
        # for two reasons:
        #   1. Every documented Qwen / Gemma token-collapse failure
        #      (vllm#27364 / #37554 / #39407, sglang#19603 / #23687,
        #      deepmind/gemma#622, OpenReview lPwsHTEPQ9) runs to
        #      ``max_tokens`` *within a single round* — a model that
        #      collapsed mid-round will not magically recover at the
        #      next round boundary.
        #   2. Round-end is a meaningful "the model successfully
        #      finished a unit of work" checkpoint; carrying sustain
        #      state across rounds would penalise a model that
        #      legitimately emitted a horizontal rule in round N for
        #      30 s of slow human reading time before round N+1.
        # The post-round ratio-loop guard (``_check_ratio_loop``) is
        # an orthogonal backstop that catches any cross-round
        # pathology this scope misses.
        self.loop_sustain: dict[str, dict[str, float]] = {}
        self.json_decode_failures: int = 0
        self.terminated_naturally: bool = False


__all__ = [
    "AgentTurnEvent",
    "Any",
    "AsyncIterator",
    "Awaitable",
    "BUDGET_HANDOFF_USER",
    "COLON_CLIFFHANGER_NUDGE_PROMPT_BODY",
    "CUTOFF_NUDGE_PROMPT_BODY",
    "Callable",
    "DEFAULT_COMPACT_AT_FRAC",
    "EMPTY_RESPONSE_NUDGE_PROMPT_BODY",
    "Engine",
    "EngineKind",
    "EngineSessionState",
    "FINAL_ANSWER_NUDGE_BODY",
    "LOOP_CHARS_PER_TOKEN_MIN",
    "LOOP_CHECK_STRIDE",
    "LOOP_DETECT_ENABLED",
    "LOOP_RATIO_MIN_TOKENS",
    "MAX_NUDGES_PER_TURN",
    "MAX_STEPS_HANDOFF_USER",
    "NUDGE_MIN_TEXT_CHARS",
    "NUDGE_PROMPT_BODY",
    "Optional",
    "PREFILL_FLOOR_MS",
    "Path",
    "QueueMessageInjected",
    "REASONING_ONLY_NUDGE_PROMPT_BODY",
    "RoundComplete",
    "STREAM_TRUNCATED_NUDGE_PROMPT_BODY",
    "TOOL_CALL_TRUNCATED_NUDGE_PROMPT_BODY",
    "THINKING_STARVED_NUDGE_PROMPT_BODY",
    "TRANSPORT_ERROR_NUDGE_PROMPT_BODY",
    "NO_TERMINAL_NUDGE_PROMPT_BODY",
    "BACKGROUND_DELEGATE_EMPTY_NUDGE_PROMPT_BODY",
    "TextDelta",
    "TurnDone",
    "TurnError",
    "TurnRequest",
    "TurnStarted",
    "UserMessageInjected",
    "WORKER_READ_TIMEOUT_S",
    "_CompactionAborted",
    "_DIRECT_PREAMBLE_PER_TURN_CHARS",
    "_DIRECT_PREAMBLE_TOTAL_CHARS",
    "_DOOM_LOOP_REPEAT_THRESHOLD",
    "_GOAL_BLOCKED_RE",
    "_HOOK_OUTCOMES_LOG",
    "_LOG",
    "_MAX_REACTIVE_MAXTOK_FALLBACKS",
    "_MAX_REACTIVE_COMPACT_RETRIES",
    "_MAX_READONLY_HARD_LIMIT",
    "_MAX_READONLY_SOFT_LIMIT",
    "_MAX_STOP_CONTINUATIONS",
    "_MAX_TRANSIENT_RETRIES",
    "_COMPACT_MIN_TURNS_BETWEEN",
    "_COMPACT_OCCUPANCY_FRAC",
    "_MIN_COMPACTION_CHARS_FREED",
    "_NoSummaryCompaction",
    "_READONLY_TOOL_NAMES",
    "_READONLY_WAIT_TOOLS",
    "_AGENT_WAIT_ACTIONS",
    "_AGENT_CONTROL_ACTIONS",
    "_BOOKKEEPING_TOOLS",
    "_is_readonly_wait_call",
    "_readonly_budget_class",
    "_tool_flag_true",
    "_is_background_job_handle",
    "_call_is_background_agent",
    "_round_is_background_delegate_wait",
    "_RoundState",
    "_build_direct_delegate_preamble",
    "_classify",
    "_collapse_leading_system_messages",
    "_delegate_resume_available",
    "_detect_stream_truncation",
    "_direct_preamble_total_budget",
    "_drain_stderr",
    "_format_runtime_warning",
    "_is_endpoint_down",
    "_is_phantom_round_metrics",
    "_is_transient_upstream",
    "_local_worker_idle_timeout_s",
    "_namespace_inner_call_id",
    "_next_worker_event",
    "_now_iso",
    "_stride_raw",
    "_terminate_and_reap",
    "asyncio",
    "contextvars",
    "datetime",
    "hashlib",
    "json",
    "logging",
    "os",
    "re",
    "sys",
    "time",
    "timezone",
    "wrap_untrusted",
]
