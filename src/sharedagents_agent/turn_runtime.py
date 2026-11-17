"""Turn lifecycle + prompt-build pipeline + session-helper utilities.

This module owns the **session-lifecycle** layer: top-level helpers
(prompt-compose, compaction, plan-pass, distillation, loop-detector)
and `TurnState`. The daemon / wire layer lives in
`sharedagents_gateway.runtime`. Direction is one-way: `sharedagents_gateway.runtime`
imports from this module, never the reverse.
"""

from __future__ import annotations

import asyncio
import re
import signal
import time

from sharedagents_core import oscompat

from .compaction import (  # noqa: F401  (re-export — keep importable from turn_runtime)
    _COMPACT_MAX_TRANSCRIPT_CHARS,
    _COMPACT_SAFETY_BUFFER,
    _COMPACT_THINKING_BUDGET,
    _COMPACTION_SUMMARY_MAX_CHARS,
    _DEFAULT_COMPACT_SYSTEM,
    _DEFAULT_DISTILL_SYSTEM,
    _DEFAULT_DISTILL_USER,
    _DISTILL_SUCCESS_MARKERS,
    _DISTILL_THINKING_BUDGET,
    _FALLBACK_CHARS_PER_TOKEN,
    _IMAGE_PART_TOKEN_COST,
    _OC_CLEARED_PLACEHOLDER,
    _OC_PRUNE_MINIMUM,
    _OC_PRUNE_PROTECT,
    _OC_PRUNE_PROTECTED_TOOLS,
    _OC_PRUNE_TAIL_TURNS,
    _OC_TOOL_OUTPUT_MAX_CHARS,
    _PINNED_FACTS_MAX_CHARS,
    _PRUNE_MINIMUM_TOKENS,
    _PRUNE_NEVER_TOOLS,
    _PRUNE_PROTECT_TOKENS,
    _PRUNE_TAIL_TURNS,
    _PRUNED_MARKER_PREFIX,
    _SUMMARY_TEMPLATE,
    _approx_tokens,
    _approx_tokens_image_aware,
    _build_compact_user_prompt,
    _content_has_image,
    _distill_partial_reasoning,
    _format_compact_summary,
    _format_compaction_summary,
    _format_distill_for_recovery,
    _hash_prefix,
    _hash_tools_payload,
    _inject_compaction_summary,
    _inject_current_todos,
    _inject_pinned_facts,
    _manual_compact_target,
    _merge_compaction_summary,
    _msg_chars,
    _oc_estimate_tokens,
    _parse_distill_json,
    _reclassify_unearned_success,
    _render_messages_as_transcript,
    _split_for_summary,
    _split_for_summary_oc,
    _stage0_prune_tool_outputs,
    _strip_compact_fences,
    _strip_images_from_user_content,
    _summarize_context_span,
    _tool_name_for_call,
    _user_field_chars,
    _wrap_compact_summary,
    compact_messages,
    compact_session_data_in_place,
    compact_session_turns_in_place,
    compaction_agent_home,
    load_compaction_prompts,
    prune_session_tool_outputs,
    scaled_prune_minimum,
    scaled_prune_protect,
)
from .loop_detection import (  # noqa: F401  (re-export — keep importable from turn_runtime)
    _BG_POLL_TOOLS,
    _EVIDENCE_METADATA_CHAR,
    _MD_TABLE_SEP_RE,
    _RULE_CHARS,
    _TOOL_CALL_LOOP_BROAD_WINDOW,
    _TOOL_CALL_LOOP_WINDOW,
    _TREE_GLYPHS,
    _XROUND_FENCE_RE,
    _XROUND_INLINE_CODE_RE,
    _XROUND_NONPROSE_LINE_RE,
    _XROUND_PATH_KEYS,
    _XROUND_PROGRESS_TOOLS,
    _XROUND_STOPWORDS,
    _XROUND_WORD_RE,
    LOOP_AMBIGUOUS_CHARS,
    LOOP_AMBIGUOUS_SUSTAIN_SECONDS,
    LOOP_CHECK_STRIDE,
    LOOP_DETECT_ENABLED,
    LOOP_ENTROPY_BITS,
    LOOP_ENTROPY_WINDOW,
    LOOP_MIN_CONTEXT_CHARS,
    LOOP_NGRAM_LEN,
    LOOP_REASONING_MIN_PERIODICITY,
    LOOP_REPEAT_THRESHOLD,
    LOOP_STRUCTURED_DISTINCT_LINES,
    LOOP_STRUCTURED_DISTINCT_SHAPES,
    LOOP_STRUCTURED_WINDOW,
    LOOP_TIGHT_MIN_CONTEXT_CHARS,
    LOOP_TIGHT_NGRAM_LEN,
    LOOP_TIGHT_REPEAT_THRESHOLD,
    LOOP_TIGHT_WINDOW_CHARS,
    LOOP_TOOL_ARGS_MIN_COVERAGE,
    LOOP_WINDOW_CHARS,
    PROGRESS_HARDSTOP_STREAK,
    PROGRESS_KEPT_TEXT_TIERS,
    PROGRESS_NUDGE_STREAK,
    TOOL_ARGS_ABORT_TIERS,
    XROUND_CONSECUTIVE_STALL,
    XROUND_LOOP_ACTION,
    XROUND_LOOP_DETECT_ENABLED,
    XROUND_MIN_CUM_REASONING_CHARS,
    XROUND_MIN_ROUND_IDX,
    XROUND_MIN_ROUND_WORDS,
    XROUND_NOVELTY_THRESHOLD,
    XROUND_SHINGLE_N,
    XROUND_TELEMETRY,
    ProgressTracker,
    TurnLoopTracker,
    _broad_sig,
    _call_args,
    _call_name,
    _calls_are_bg_poll_only,
    _detect_bigram_entropy_collapse,
    _detect_entropy_collapse,
    _detect_loops_layered,
    _detect_reasoning_loop,
    _detect_tight_loop,
    _detect_tool_call_loop,
    _detect_tool_call_loop_broad,
    _env_float,
    _env_int,
    _is_ambiguous_chrome_run,
    _is_rule_run,
    _is_unary_collapse_evidence,
    _loop_coverage,
    _loop_hit_survives_channel,
    _ngram_periodicity,
    _progress_target_key,
    _reasoning_prose_shingles,
    _reasoning_prose_text,
    _rotate_seed,
    _round_is_bg_poll_wait,
    _round_progressed,
    _shingle_novelty,
    _should_fire_with_sustain,
    _suffix_has_markdown_table,
    _suffix_is_identical_line_degeneracy,
    _suffix_is_structured_content,
    _suffix_is_structured_text,
    _tool_target_key,
    progress_finalize_body,
    progress_nudge_body,
    tool_args_loop_should_abort,
)
from .prompt_build import (  # noqa: F401  (re-export — keep importable from turn_runtime)
    _compose_system_prompt,
    _engine_brings_own_tools,
    _profile_system_prompt,
)
from .session_helpers import (  # noqa: F401  (re-export — keep importable from turn_runtime)
    _has_image,
    _new_turn_id,
    _now,
    _publish_turn_completed,
    _spawn_detached_restart,
)

# Common phrasings of an OpenAI-compatible context-overflow 400. We
# extract the three numbers when present so the reactive squeeze can
# size its trim budget off what the upstream actually reported instead
# of falling back to a (possibly stale or wrong) cfg.upstream_max_context
# — the failure that surfaced to the user as the raw `HTTP 400: ... your
# prompt contains at least 245761 input tokens, for a total of at least
# 262145 tokens` payload.
_OVERFLOW_MAX_RE = re.compile(
    r"maximum\s+context\s+length\s+is\s+(\d+)",
    re.IGNORECASE,
)
_OVERFLOW_REQUESTED_OUT_RE = re.compile(
    r"requested\s+(\d+)\s+(?:output|completion)\s+tokens",
    re.IGNORECASE,
)
_OVERFLOW_INPUT_RE = re.compile(
    r"prompt\s+contains\s+(?:at\s+least\s+)?(\d+)\s+(?:input\s+)?tokens",
    re.IGNORECASE,
)
_OVERFLOW_INPUT_LEGACY_RE = re.compile(
    r"messages?\s+resulted\s+in\s+(\d+)\s+tokens",
    re.IGNORECASE,
)


def parse_overflow_numbers(msg: str) -> dict:
    """Extract the model_max / requested_max_tokens / input_tokens from
    an OpenAI-compatible context-overflow 400 body.

    Returns a dict containing whatever keys could be parsed:
        {"model_max": int, "requested_max_tokens": int, "input_tokens": int}
    Missing keys are simply absent (caller uses ``.get``). On a
    non-overflow / empty / unrecognised string, returns ``{}``.

    Recognises:
      * vLLM verbose: "maximum context length is N tokens. However, you
        requested M output tokens and your prompt contains at least K
        input tokens, for a total of at least N+1 tokens."
      * OpenAI legacy: "maximum context length is N tokens. However,
        your messages resulted in K tokens."
    """
    out: dict[str, int] = {}
    if not msg:
        return out
    m = _OVERFLOW_MAX_RE.search(msg)
    if m:
        out["model_max"] = int(m.group(1))
    m = _OVERFLOW_REQUESTED_OUT_RE.search(msg)
    if m:
        out["requested_max_tokens"] = int(m.group(1))
    m = _OVERFLOW_INPUT_RE.search(msg)
    if m:
        out["input_tokens"] = int(m.group(1))
    elif "input_tokens" not in out:
        m = _OVERFLOW_INPUT_LEGACY_RE.search(msg)
        if m:
            out["input_tokens"] = int(m.group(1))
    return out


def _classify_upstream_error(msg: str) -> str:
    """Best-effort one-line summary for `errors.log_error(..., summary=)`.

    Keeps the structured error file's filename slug (and the Markdown
    tagline) human-readable so a glance at `~/.sharedagents/errors/` tells
    the next agent what class of failure this was without opening the
    file. Falls back to the raw HTTP code when nothing matches.
    """
    m = (msg or "").lower()
    if "missing endpoint/model" in m:
        # The worker refused before any request: the agent has no model
        # configured. Retrying the identical request cannot recover.
        return "config"
    if "maximum context length" in m or "context window" in m or "context_length_exceeded" in m:
        return "upstream context-length overflow"
    if '"code":429' in m or "rate limit" in m or "429" in m[:30]:
        return "upstream rate-limited"
    # Permanent client refusals — retrying the same request cannot recover.
    # xAI grok-4.6 rejects an unsupported reasoning slider as
    # ``HTTP 400: {"code":"invalid-argument","error":"Invalid reasoning effort."}``.
    if "invalid reasoning effort" in m or "invalid-argument" in m or "invalid argument" in m:
        return "bad_request"
    if "timeout" in m or "timed out" in m:
        return "upstream timeout"
    if msg.startswith("HTTP "):
        # `HTTP 400: {...}` → bad_request for deterministic 4xx;
        # transient 408/429/5xx stay as ``upstream HTTP NNN``.
        code = msg.split(":", 1)[0]
        try:
            status = int(code.split()[1])
        except (IndexError, ValueError):
            status = 0
        if status in {400, 401, 403, 404, 422}:
            return "bad_request"
        if status == 402:
            # Payment required / credits exhausted — the next attempt gets
            # the same answer until the operator tops the account up.
            return "quota_exhausted"
        return f"upstream {code}"
    return "upstream error"


class TurnState:
    """Lifecycle of one in-flight turn."""

    def __init__(self, turn_id: str, session_hash: str, proc: asyncio.subprocess.Process) -> None:
        self.id = turn_id
        self.session_hash = session_hash
        self.proc = proc
        # Per-tool-call subprocesses keyed by tool_call_id. The bridge
        # dispatch loop now runs read-only / state-file-distinct calls
        # concurrently, so a single ``turn.proc`` slot is no longer
        # enough — ESC must SIGTERM ALL active runners. ``self.proc``
        # still points at "the most recent process" (worker initially,
        # then a runner under the legacy single-call path) for back
        # compat with the cancel + drain helpers; ``runner_procs``
        # carries the multi-runner case.
        self.runner_procs: dict[str, asyncio.subprocess.Process] = {}
        self.state = "streaming"  # streaming | paused | done | aborted | error
        self.started_at = _now()
        # Monotonic timestamp for staleness checks (the ISO `started_at`
        # is wall-clock and not safe across NTP jumps). The heartbeat
        # scheduler uses this to evict TurnState entries that have
        # leaked past the heartbeat interval.
        self.started_mono = time.monotonic()
        self.ended_at: str | None = None
        self.error: str | None = None
        self.thinking_buf: list[str] = []
        self.content_buf: list[str] = []
        self.metrics: dict | None = None
        # Tool calls the model emitted in this round (if any). The
        # session_turn loop dispatches these and starts another round.
        self.pending_tool_calls: list[dict] = []
        self._listeners: set[asyncio.Queue] = set()
        # Set True by the streaming loop detector
        # (`_detect_reasoning_loop`) when the model's CoT shows
        # degenerative repetition. The worker is SIGTERM'd as soon as
        # this is set; `session_turn` routes the round to the distill
        # recovery path the same way it would for a `thinking_only_stop`.
        # Stays False on healthy turns.
        self.loop_detected: bool = False
        # Mirrors the per-turn ``post_force_terminate`` flag in the
        # bridge prompt loop (acp_bridge.py). When True, the runner
        # spawner (``_default_run_tool``) clamps EXEC-kind tools'
        # ``timeout`` / ``wall_timeout`` args to a short ceiling so a
        # token-collapsed model that slips past the bridge's deny gate
        # can't dispatch a 30-minute bash. Belt-and-suspenders for the
        # bridge-level block.
        self.post_force_terminate: bool = False
        # Last `len(thinking_buf joined)` at which we ran the loop
        # detector. Throttles detection to once per ~LOOP_CHECK_STRIDE
        # chars rather than once per token; the detector itself is O(n)
        # in buffer size so without throttling it's quadratic over a
        # streaming round.
        self._loop_last_check_chars: int = 0

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._listeners.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._listeners.discard(q)

    async def emit(self, event: str, data: dict) -> None:
        for q in list(self._listeners):
            await q.put((event, data))

    def stop(self, *, force: bool = False) -> None:
        """SIGTERM all in-flight subprocesses for this turn.

        With ``force=True`` (hard cancel — ``/abort`` / second ESC),
        escalate to ``SIGKILL`` and tear down each runner's ENTIRE
        descendant tree.

        Each runner (worker / tool_runner) is spawned
        ``start_new_session=True`` so it sits in its OWN process group —
        critical, because otherwise ``killpg`` would hit the gateway's
        group and kill the gateway itself. But a
        runner's own children may start FURTHER sessions (e.g.
        ``run_bash_async`` gives the bash its own session so a tool
        timeout can ``killpg`` it), so a single ``killpg(runner)`` would
        miss the bash and orphan a long ``sleep``. The force path delegates
        to :func:`sharedagents_agent.tools.runners._kill_tree`, which walks the
        descendant tree to reach every grandchild regardless of session
        boundaries, never touches the runner's ANCESTORS (the gateway), and
        only ``killpg``s a group the proc itself leads — so the gateway
        always survives. That is the SAME instant whole-tree primitive the
        delegate-CLI cancel handlers use, so ``/abort`` kills the current
        agent and every subagent the same way no matter which path runs.
        """
        procs: list[asyncio.subprocess.Process] = []
        if self.proc is not None:
            procs.append(self.proc)
        procs.extend(self.runner_procs.values())
        if force:
            # Lazy import: tools.runners imports lazily from core, so a
            # module-level import here would risk a load cycle.
            from .tools.runners import _kill_tree

            for proc in procs:
                _kill_tree(proc, oscompat.FORCE_KILL)
            return
        for proc in procs:
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

    def register_runner(self, call_id: str, proc: asyncio.subprocess.Process) -> None:
        """Track an in-flight tool-runner subprocess by tool_call_id."""
        self.runner_procs[call_id] = proc

    def unregister_runner(self, call_id: str) -> None:
        """Drop a runner once its subprocess has exited / been reaped."""
        self.runner_procs.pop(call_id, None)
