"""Shared delegate runner primitives (Phase 1 split).

Extracted from ``tools/impl/delegate.py`` — spawn helpers, cwd resolve,
logging, usage normalization, reasoning signal counters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sharedagents_agent.artifact_links import local_artifact_references
from sharedagents_core.cwdutil import safe_getcwd

# ---- delegation: external-agent bridges ----


def _load_tool_utils(namespace: dict[str, Any]) -> None:
    """Populate a delegate module with shared tool-runner primitives lazily.

    Importing ``tools.impl._utils`` while this module is initializing first
    executes ``tools.__init__``.  That imports ``tools.runners``, whose legacy
    re-export reaches back into this partially initialized module.  Defer the
    import until a runner is called so direct imports of the split modules do
    not form that cycle.
    """
    from sharedagents_agent.tools.impl import _utils

    for name in _utils.__all__:
        namespace.setdefault(name, getattr(_utils, name))


def log_event(event: str, **fields: Any) -> None:
    """Lazy compatibility wrapper for the shared structured logger."""
    from sharedagents_agent.tools._log import log_event as _log_event

    _log_event(event, **fields)


def _delegate_state_path(
    engine: str,
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> Path | None:
    """Resolve one CLI delegate's per-sharedagents-session state sidecar."""
    sid = sa_sid or os.environ.get("SHAREDAGENTS_SESSION_ID")
    sdir = sa_sdir or os.environ.get("SHAREDAGENTS_SESSIONS_DIR")
    if not sid or not sdir:
        return None
    return Path(sdir) / f"{sid}.delegate_{engine}.json"


def _read_delegate_state(
    engine: str,
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> dict[str, Any]:
    """Read a CLI sidecar; missing or corrupt state is an empty mapping."""
    path = _delegate_state_path(engine, sa_sid, sa_sdir)
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_delegate_state(
    engine: str,
    state: dict[str, Any],
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> None:
    """Best-effort write of one CLI delegate sidecar."""
    path = _delegate_state_path(engine, sa_sid, sa_sdir)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _update_delegate_state(
    engine: str,
    updates: dict[str, Any],
    sa_sid: str | None = None,
    sa_sdir: str | None = None,
) -> None:
    """Merge fields into one CLI delegate sidecar without dropping state."""
    data = _read_delegate_state(engine, sa_sid, sa_sdir)
    data.update(updates)
    _write_delegate_state(engine, data, sa_sid, sa_sdir)


def _format_cli_tool_use(name: str, arguments: Any) -> str:
    """Render the shared compact marker for a CLI agent's inner tool call."""
    try:
        args_json = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        args_json = str(arguments)
    if len(args_json) > 160:
        args_json = args_json[:160] + "…"
    return f"\n\n🔧 {name} {args_json}\n"


# Map sharedagents permission modes → claude --permission-mode values.
# Same table as `app._claude_permission_mode`; duplicated here so the
# tool runner doesn't have to import the TUI module.
#
# `auto` → `bypassPermissions` (not `acceptEdits`): in headless `-p`
# mode there's no human to answer a permission prompt, so anything that
# isn't a plain edit (Bash, Write to a new path, etc.) deadlocks under
# `acceptEdits`. `bypassPermissions` is the canonical full-bypass mode
# and is what the user opts into by cycling sharedagents to `auto`.
_CLAUDE_MODE_MAP: dict[str, str] = {
    "plan": "plan",
    "ask": "default",
    "auto": "bypassPermissions",
}

# sharedagents permission mode → antigravity native id. The current CLI
# (``agy`` / ``antigravity``) has no ``--approval-mode``. Allow-all is
# ``--dangerously-skip-permissions``; plan is ``--mode plan --sandbox``;
# ask is the CLI default (no extra flags). ``--mode`` only accepts
# ``accept-edits`` and ``plan``.
_ANTIGRAVITY_MODE_MAP: dict[str, str] = {
    "plan": "plan",
    "ask": "default",
    "auto": "yolo",
}

# Highest portable reasoning-depth level. ``max``/``xhigh`` are retained as
# explicit overrides for engines/models that advertise them, but ``high`` is
# the only top-tier value accepted by every wired engine family.
_CLI_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _codex_local_artifact_references(
    text: str,
    spawn_cwd: str | None,
) -> list[dict[str, Any]]:
    """Backward-compatible Codex name for the shared output-link parser."""
    return local_artifact_references(text, spawn_cwd)


def _effective_cli_effort(args: dict[str, Any]) -> str:
    effort = str(args.get("effort") or "max").strip().lower()
    return effort if effort in _CLI_EFFORT_LEVELS else "max"


def _claude_effort_flags(args: dict[str, Any]) -> list[str]:
    """Claude Code's native per-session reasoning-depth flag."""
    return ["--effort", _effective_cli_effort(args)]


def _codex_effort_flags(args: dict[str, Any]) -> list[str]:
    """Codex CLI config override for the model reasoning effort."""
    effort = _effective_cli_effort(args)
    return ["-c", f'model_reasoning_effort="{effort}"']


_CLAUDE_THINKING_DISPLAYS = frozenset({"summarized", "omitted", "highlights"})
_CODEX_REASONING_SUMMARIES = frozenset({"auto", "concise", "detailed", "none"})
_CLI_KNOB_OFF = frozenset({"off", "0", "false", "no"})


def _env_cli_choice(
    name: str,
    default: str,
    choices: frozenset[str],
    off: frozenset[str],
) -> str | None:
    """Read a CLI-flag env knob: a valid choice, ``None`` when switched off,
    else ``default`` (an unset, empty or unknown value never reaches argv —
    the CLIs reject an invalid choice and the whole turn would fail)."""
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in off:
        return None
    return raw if raw in choices else default


def _claude_thinking_display_flags() -> list[str]:
    """Ask Claude Code for readable thinking text.

    In ``-p`` stream-json mode Claude Code leaves ``thinking.display`` unset,
    and the API then returns thinking blocks with empty text (a signature
    only), so the reasoning card has nothing to show. ``--thinking-display``
    (hidden from ``--help``; summarized|omitted|highlights) makes the text
    stream as ``thinking_delta``. Env ``SHAREDAGENTS_CLAUDE_THINKING_DISPLAY``
    overrides; ``off``/``none`` omits the flag for a CLI build that rejects it.
    """
    display = _env_cli_choice(
        "SHAREDAGENTS_CLAUDE_THINKING_DISPLAY",
        "summarized",
        _CLAUDE_THINKING_DISPLAYS,
        _CLI_KNOB_OFF | {"none"},
    )
    return ["--thinking-display", display] if display else []


def _codex_reasoning_summary_flags() -> list[str]:
    """Ask Codex for reasoning summaries; without one it emits no reasoning item.

    Env ``SHAREDAGENTS_CODEX_REASONING_SUMMARY`` (auto|concise|detailed|none)
    overrides; ``off`` omits the override so ``~/.codex/config.toml`` decides
    (e.g. an API-key org that is not verified for reasoning summaries).
    """
    summary = _env_cli_choice(
        "SHAREDAGENTS_CODEX_REASONING_SUMMARY",
        "detailed",
        _CODEX_REASONING_SUMMARIES,
        _CLI_KNOB_OFF,
    )
    return ["-c", f'model_reasoning_summary="{summary}"'] if summary else []


def _looks_like_context_overflow(text: str) -> bool:
    """Best-effort classifier for model context-window failures."""
    s = (text or "").lower()
    if not s:
        return False
    return (
        "context_length" in s
        or "context length" in s
        or "maximum context" in s
        or "input context is too large" in s
        or "prompt is too long" in s
        or "input_tokens" in s
    )


def _resolve_delegate_cwd(
    args: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Validate and resolve the optional `cwd` arg for delegate_* runners.

    Returns `(cwd, None)` on success — `cwd` is an absolute path string
    suitable for `Popen(..., cwd=...)`, or `None` when the caller didn't
    pass one (in which case the spawned subprocess inherits sharedagents's
    launch directory, the historical default we don't change).
    On failure returns `(None, error_msg)`; the runner should surface
    that as a tool-result error and skip spawning — a bad cwd is almost
    always a typo, so we fail loud rather than silently fall back.

    Relative paths are resolved against `safe_getcwd()` (sharedagents's launch
    cwd at the moment the runner subprocess invokes this), per the
    contract: "interpret as the model gives it, don't try to be clever".
    """
    raw = args.get("cwd")
    if raw is None:
        return None, None
    if not isinstance(raw, str) or not raw.strip():
        return None, "cwd must be a non-empty string when provided"
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path(safe_getcwd()) / p
    try:
        resolved = p.resolve(strict=False)
    except OSError as e:
        return None, f"cwd resolve failed: {e}"
    if not resolved.exists():
        return None, f"cwd does not exist: {resolved}"
    if not resolved.is_dir():
        return None, f"cwd is not a directory: {resolved}"
    return str(resolved), None


def _log_delegate_error(
    delegate: str,
    reason: str,
    *,
    prompt: str | None = None,
    cwd: str | None = None,
    mode: str | None = None,
    resume_id: str | None = None,
    remote_session_id: str | None = None,
    returncode: int | None = None,
    stderr: str | None = None,
    message: str | None = None,
    event: dict[str, Any] | None = None,
    sharedagents_session_id: str | None = None,
    **extra: Any,
) -> None:
    """Best-effort structured logging for delegate_* failures.

    Tool results intentionally stay compact because they are fed back
    into the model. The gateway log is where operators need the richer
    failure context: whether this was a stale resume, a subprocess
    launch problem, an API/context error surfaced by the delegate CLI,
    or a local validation failure.

    ``sharedagents_session_id`` is the runner's explicit session identity. Engine
    turns never export ``SHAREDAGENTS_SESSION_ID`` (concurrent turns must not race
    on ``os.environ``), so without it their log lines carry a null session.
    """
    fields: dict[str, Any] = {
        "sharedagents_session_id": sharedagents_session_id or os.environ.get("SHAREDAGENTS_SESSION_ID"),
        "agent_name": os.environ.get("SHAREDAGENTS_AGENT_NAME"),
        "reason": reason,
        "mode": mode,
        "cwd": cwd or safe_getcwd(),
        "resume_id": resume_id,
        "resumed": bool(resume_id),
        "remote_session_id": remote_session_id,
        "returncode": returncode,
        "prompt_chars": len(prompt) if isinstance(prompt, str) else None,
    }
    if stderr:
        fields["stderr_preview"] = stderr[:1000]
    if message:
        fields["message_preview"] = message[:1000]
    if event:
        fields.update(
            {
                "subtype": event.get("subtype"),
                "api_error_status": event.get("api_error_status"),
                "terminal_reason": event.get("terminal_reason"),
                "result_preview": str(event.get("result") or "")[:1000],
                "usage": event.get("usage"),
                "model_usage": event.get("modelUsage"),
            }
        )
    fields.update(extra)
    from sharedagents_agent.tools._log import log_event

    log_event(f"DELEGATE_{delegate.upper()}_ERROR", **fields)


def _codex_usage_from_event(event: dict[str, Any]) -> dict[str, int] | None:
    """Normalize Codex ``turn.completed.usage`` for the engine metrics path.

    Codex/OpenAI reports ``input_tokens`` as the total prompt count and
    ``cached_input_tokens`` as a subset. The external-engine canonical shape
    expects fresh input and cache-read input as additive fields, so subtract
    the cached subset exactly once.
    """
    if not isinstance(event, dict) or event.get("type") != "turn.completed":
        return None
    raw = event.get("usage")
    if not isinstance(raw, dict):
        return None

    def _count(name: str) -> int:
        try:
            return max(0, int(raw.get(name) or 0))
        except (TypeError, ValueError):
            return 0

    total_in = _count("input_tokens")
    cached = _count("cached_input_tokens")
    output = _count("output_tokens")
    if total_in <= 0 and output <= 0:
        return None
    cached = min(cached, total_in)
    return {
        "input_tokens": total_in - cached,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
        "output_tokens": output,
        # Reasoning rides inside ``output_tokens``; surfacing it separately
        # lets the engine split the reasoning/content halves of the total
        # (a 0 is honest — this provider can genuinely report none).
        "reasoning_tokens": _count("reasoning_output_tokens"),
    }


def _codex_raw_cumulative(event: dict[str, Any]) -> dict[str, int] | None:
    """Extract RAW cumulative token counts from a ``turn.completed`` event.

    Codex's ``turn.completed`` usage is CUMULATIVE across the whole thread
    (every API call in the thread so far), NOT a per-invocation delta
    (openai/codex#17539). Returns ``{"input_tokens": total_in,
    "cached_input_tokens": cached, "output_tokens": output}`` — the raw
    cumulative basis the delta computation needs (cached kept as a subset
    of total, NOT yet split out) — or ``None`` when unusable.
    """
    if not isinstance(event, dict) or event.get("type") != "turn.completed":
        return None
    raw = event.get("usage")
    if not isinstance(raw, dict):
        return None

    def _count(name: str) -> int:
        try:
            return max(0, int(raw.get(name) or 0))
        except (TypeError, ValueError):
            return 0

    total_in = _count("input_tokens")
    cached = _count("cached_input_tokens")
    output = _count("output_tokens")
    if total_in <= 0 and output <= 0:
        return None
    return {
        "input_tokens": total_in,
        "cached_input_tokens": min(cached, total_in),
        "output_tokens": output,
        "reasoning_output_tokens": _count("reasoning_output_tokens"),
    }


def _codex_usage_delta(
    raw_cum: dict[str, int],
    last_cum: dict[str, int] | None,
) -> dict[str, int] | None:
    """Per-invocation DELTA of cumulative codex usage, normalized to the
    4-field shape the engine bills (see :func:`_codex_usage_from_event`).

    ``delta = max(0, current_cumulative - last_seen_cumulative)`` per
    field. Returns ``None`` when no new tokens were produced (the
    invocation is not billable). Detects a thread reset (cumulative went
    DOWN — the thread was reset/GC'd) by treating ``last_cum`` as ``None``
    and billing fresh.
    """
    if last_cum is not None:
        try:
            if int(raw_cum["input_tokens"]) < int(last_cum.get("input_tokens") or 0):
                last_cum = None  # thread was reset/GC'd — bill fresh
        except (TypeError, ValueError):
            pass

    def _last(name: str) -> int:
        try:
            return max(0, int((last_cum or {}).get(name) or 0))
        except (TypeError, ValueError):
            return 0

    total_in = max(0, int(raw_cum["input_tokens"]) - _last("input_tokens"))
    cached = max(0, int(raw_cum["cached_input_tokens"]) - _last("cached_input_tokens"))
    output = max(0, int(raw_cum["output_tokens"]) - _last("output_tokens"))
    reasoning = max(
        0,
        int(raw_cum.get("reasoning_output_tokens") or 0) - _last("reasoning_output_tokens"),
    )
    if total_in <= 0 and output <= 0:
        return None
    cached = min(cached, total_in)
    return {
        "input_tokens": total_in - cached,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
        "output_tokens": output,
        "reasoning_tokens": min(reasoning, output),
    }


# Reasoning is deliberately tracked as a count only.  The CLI engines must
# be able to distinguish a successful reasoning-only stop from a genuinely
# silent/error turn, but private chain-of-thought must never be copied into a
# ToolResult body or an ``on_delta`` callback.  These are the provider event
# labels currently seen by the three external CLI adapters.
_REASONING_BLOCK_TYPES = frozenset(
    {
        "agent_reasoning",
        "reasoning",
        "thinking",
        "redacted_thinking",
        "thought",
    }
)
_REASONING_DELTA_TYPES = frozenset(
    {
        "agent_reasoning_delta",
        "reasoning_delta",
        "thinking_delta",
    }
)


def _reasoning_text_chars(value: Any) -> int:
    """Count private reasoning text without returning any of its contents."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        total = 0
        for item in value:
            if isinstance(item, dict):
                total += _reasoning_text_chars(item.get("text") or item.get("thinking") or item.get("reasoning"))
            elif isinstance(item, str):
                total += len(item)
        return total
    return 0


def _reasoning_signal_chars(
    value: Any,
    *,
    type_key: str = "type",
) -> int:
    """Return a private reasoning count for one known provider signal.

    A redacted/no-text reasoning event still counts as one signal character;
    the continuation classifier needs a non-zero signal, not the secret text.
    Unknown event shapes return zero so ordinary provider metadata cannot
    accidentally turn a genuinely empty turn into a continuation.
    """
    if not isinstance(value, dict):
        return 0
    label = value.get(type_key)
    if label not in _REASONING_BLOCK_TYPES and label not in _REASONING_DELTA_TYPES:
        return 0
    count = 0
    for key in ("text", "thinking", "reasoning", "summary", "content", "delta"):
        count += _reasoning_text_chars(value.get(key))
    return max(1, count)


# Providers report the reasoning VOLUME under several names and at different
# nesting depths: OpenAI flat ``reasoning_tokens`` / ``reasoning_output_tokens``,
# Gemini ``thinking_tokens``, Claude's
# ``output_tokens_details.thinking_tokens``. One extractor keeps the three CLI
# adapters from each keeping a private copy of that table.
_REASONING_TOKEN_KEYS = (
    "reasoning_tokens",
    "thinking_tokens",
    "reasoning_output_tokens",
)
_REASONING_TOKEN_CONTAINERS = (
    "output_tokens_details",  # Claude Code
    "candidates_tokens_details",  # Gemini / Antigravity
    "completion_tokens_details",  # OpenAI chat-completions
)


def _reasoning_tokens_from_usage(usage: Any) -> int:
    """Return the provider-reported reasoning-token count, else ``0``.

    Flat keys are read first, then each family's detail sub-object, so a
    caller never needs to know which provider produced the payload.
    """
    if not isinstance(usage, dict):
        return 0
    payloads: list[Any] = [usage]
    for container in _REASONING_TOKEN_CONTAINERS:
        nested = usage.get(container)
        if isinstance(nested, dict):
            payloads.append(nested)
    for payload in payloads:
        for key in _REASONING_TOKEN_KEYS:
            try:
                value = int(payload.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return 0


class ReasoningTextRelay:
    """Emit one reasoning trace once, across both provider event shapes.

    A streaming provider reports the same trace twice: incrementally as
    deltas (``thinking_delta``, ``item.updated``, ``thought_delta``) and again
    as the consolidated block that restates it (the full ``assistant``
    message, ``item.completed``, a finished ``thought`` step). Forwarding both
    prints the trace twice *and* persists it twice, so each block is emitted
    once: deltas stream as they arrive, and a restatement only forwards the
    tail its deltas have not produced yet.

    Blocks are keyed by whatever identity the provider offers (an item id, a
    step index, a running message slot). A provider that streams no deltas
    falls through to the consolidated text unchanged; :meth:`forget` drops a
    key whose numbering restarts (a new message).
    """

    def __init__(self) -> None:
        self._streamed: dict[str, str] = {}

    def delta(self, identity: str, text: str) -> str:
        """Record a streamed fragment; returns the text to emit."""
        if not text:
            return ""
        self._streamed[identity] = self._streamed.get(identity, "") + text
        return text

    def consolidated(self, identity: str, text: str) -> str:
        """Return only the part of a restated block not already streamed."""
        if not text:
            return ""
        streamed = self._streamed.get(identity, "")
        if not streamed:
            self._streamed[identity] = text
            return text
        if text.startswith(streamed):
            self._streamed[identity] = text
            return text[len(streamed) :]
        if text in streamed:
            return ""
        # Unrelated text reusing the key — a genuinely new block.
        self._streamed[identity] = streamed + text
        return text

    def forget(self, identity: str) -> None:
        """Drop one block's buffer — its numbering is restarting."""
        self._streamed.pop(identity, None)


def _claude_error_is_abort(event: dict[str, Any]) -> bool:
    """Return True for Claude result errors caused by user cancellation."""
    terminal_reason = str(event.get("terminal_reason") or "").lower()
    if terminal_reason in {
        "aborted",
        "aborted_streaming",
        "aborted_tools",
        "cancelled",
        "canceled",
        "user_abort",
        "interrupted",
    }:
        return True
    return terminal_reason.startswith("aborted_")


def _is_numeric_zero(value: Any) -> bool:
    """True only when ``value`` is a number equal to 0 (absent → False)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == 0


def _claude_result_is_task_notification(
    event: dict[str, Any],
    *,
    saw_task_notification: bool = False,
) -> bool:
    """True for a `result` that closes a background-task DRAIN, not our turn.

    On ``--resume``, the Claude CLI reconciles background tasks the
    previous (one-shot ``claude -p``) run left unreaped by enqueueing a
    ``<task-notification>`` and answering it as its own pseudo-turn
    BEFORE reading our prompt. That pseudo-turn emits a complete
    ``result`` event which is NOT the answer to the prompt we sent.

    Observed against CLI 2.1.220 ``--output-format stream-json``, the
    drain result is tagged first-class::

        {"type": "result", "subtype": "success", "result": "",
         "origin": {"kind": "task-notification"},
         "num_turns": 0, "duration_api_ms": 0, "total_cost_usd": 0}

    ``origin.kind`` is the authoritative discriminator and the only one
    this relies on when present. Other origin kinds exist, so an
    explicit non-task ``origin.kind`` is treated as NOT a drain rather
    than falling through to the heuristic.

    The heuristic below is a narrow compatibility fallback used ONLY
    when the ``origin`` key is absent entirely. It is deliberately biased
    toward surfacing (a missed skip merely reproduces the old bug,
    whereas a wrong skip could swallow a real answer), so it fires only
    when ALL hold: this stream already carried a ``system``/
    ``task_notification`` event, the subtype is exactly ``success``,
    ``is_error`` is falsy, ``num_turns`` is present and exactly
    numeric 0, ``result`` is a present empty string, and both
    ``duration_api_ms`` and ``total_cost_usd`` are present and zero. Requiring ``subtype == "success"`` keeps a
    malformed ``error_during_execution`` (which the CLI is known to
    emit with inconsistent ``is_error``) on the structured error path.
    """
    if "origin" in event:
        # The CLI told us where this result came from — believe it, and
        # never fall through to the heuristic. A present-but-malformed
        # origin (None, {}, "", a scalar) is NOT an assertion that this
        # is a drain, so it fails closed.
        origin = event.get("origin")
        return isinstance(origin, dict) and str(origin.get("kind") or "").strip().lower() == "task-notification"
    # Compatibility fallback: `origin` key absent entirely.
    if not saw_task_notification:
        return False
    if event.get("local_command"):
        # The CLI's own answer to a built-in command (``/compact``).
        return False
    if str(event.get("subtype") or "").strip().lower() != "success":
        return False
    if event.get("is_error"):
        return False
    result_text = event.get("result")
    if not isinstance(result_text, str) or result_text.strip():
        return False
    if not _is_numeric_zero(event.get("duration_api_ms")):
        return False
    if not _is_numeric_zero(event.get("total_cost_usd")):
        return False
    # Strict zero — no lossy int() coercion (0.5 and True must not pass).
    return _is_numeric_zero(event.get("num_turns"))


def _claude_result_context_window(event: dict[str, Any], model: str) -> int:
    """Window the Claude CLI ran ``model`` with, from a `result` event.

    ``result.modelUsage`` is keyed by the exact ``system/init`` model id
    (``claude-opus-5-5[1m]`` included) and each entry carries the
    ``contextWindow`` the CLI enforced. That can be smaller than the Models
    API's ``max_input_tokens``: CLI 2.1.280 runs claude-sonnet-4-5 at 200K
    (API: 1M), and ``CLAUDE_CODE_DISABLE_1M_CONTEXT=1`` caps 1M models at
    200K. Keying on the init model keeps a subagent's entry out. ``0`` when
    the result carries no entry for ``model`` (error results send ``{}``).
    """
    usage = event.get("modelUsage")
    entry = usage.get(model) if isinstance(usage, dict) and model else None
    if not isinstance(entry, dict):
        return 0
    try:
        return max(0, int(entry.get("contextWindow") or 0))
    except (TypeError, ValueError):
        return 0


# Opening this long is a restated/revised body, not a new post-tool segment.
# Witnessed (antigravity session 79f6494f4708 turn 6): streamed
# draft + later consolidated ``result`` shared ~790 opening chars, then
# the runner appended the whole revision and the Web UI showed the answer
# twice. Claude ``assistant`` and Codex ``item.completed`` have the same
# stream-then-restate shape.
_CLI_RESTATE_PREFIX_CHARS = 80


def _cli_consolidated_answer_delta(already: str, text: str) -> str:
    """Return the unstreamed suffix of a CLI consolidated answer, or ``""``.

    Antigravity / Claude / Codex emit live deltas and later a consolidated
    ``message`` / ``result`` / ``item.completed`` body. That body is often a
    revised restatement of text already streamed (same opening, slightly
    different middle). Appending it duplicates the visible answer.

    A genuine new segment after tools does not share a long opening with
    the prior answer, so it still returns ``text``.
    """
    if not text:
        return ""
    if not already:
        return text
    if text.startswith(already) and len(text) > len(already):
        return text[len(already) :]
    # Substring / suffix checks are only safe for a long restated body.
    # Short tokens ("e", " the", "\n") appear inside almost any draft.
    if len(text) >= 24 and (already.endswith(text) or text in already):
        return ""
    probe = min(len(already), len(text), _CLI_RESTATE_PREFIX_CHARS)
    if probe >= 24:
        shared = 0
        limit = min(len(already), len(text))
        while shared < limit and already[shared] == text[shared]:
            shared += 1
        if shared >= probe:
            return ""
        if already[:probe] in text or text[:probe] in already:
            return ""
    return text


class CliAnswerSink:
    """One answer accumulator for every CLI engine.

    Vendor runners still decode their own NDJSON envelopes. Every extracted
    answer string — incremental delta or consolidated body — goes through
    :meth:`push`. Tool markers stay outside this object.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        # Text streamed since the last consolidated body. A body restating it
        # (whole, or its tail after an earlier step) is a restatement at ANY
        # length. The length guards in ``_cli_consolidated_answer_delta`` are
        # for comparisons against the whole answer, where short tokens match
        # by accident — they let a short answer ("120") through twice.
        self._fresh = ""

    @property
    def text(self) -> str:
        return "".join(self._parts)

    def push_delta(self, chunk: str) -> str:
        """Always append an incremental token. Never run restatement checks."""
        if not chunk:
            return ""
        self._parts.append(chunk)
        self._fresh += chunk
        return chunk

    def push_consolidated(self, text: str) -> str:
        """Keep the unstreamed suffix of a restated body, or ``""``."""
        fresh, self._fresh = self._fresh, ""
        if fresh and text:
            if fresh.endswith(text):
                return ""
            if text.startswith(fresh):
                kept = text[len(fresh) :]
                self._parts.append(kept)
                return kept
        kept = _cli_consolidated_answer_delta(self.text, text)
        if kept:
            self._parts.append(kept)
        return kept

    def push_final(self, text: str) -> str:
        """Keep only what a terminal result adds to the answer, or ``""``.

        A terminal result (Antigravity ``result.response``) restates the
        answer its steps already delivered, streamed or consolidated, so a
        tail match against the whole answer is a restatement at any length.
        """
        if text and self.text.endswith(text):
            self._fresh = ""
            return ""
        return self.push_consolidated(text)

    def push(self, chunk: str) -> str:
        """Backward-compatible alias for :meth:`push_consolidated`."""
        return self.push_consolidated(chunk)


__all__ = [
    "_ANTIGRAVITY_MODE_MAP",
    "_CLAUDE_MODE_MAP",
    "_CLI_EFFORT_LEVELS",
    "_REASONING_BLOCK_TYPES",
    "_REASONING_DELTA_TYPES",
    "_REASONING_TOKEN_CONTAINERS",
    "_REASONING_TOKEN_KEYS",
    "_claude_effort_flags",
    "_claude_thinking_display_flags",
    "_claude_error_is_abort",
    "_claude_result_is_task_notification",
    "_claude_result_context_window",
    "_cli_consolidated_answer_delta",
    "CliAnswerSink",
    "ReasoningTextRelay",
    "_codex_effort_flags",
    "_codex_reasoning_summary_flags",
    "_codex_local_artifact_references",
    "_codex_raw_cumulative",
    "_codex_usage_delta",
    "_codex_usage_from_event",
    "_delegate_state_path",
    "_effective_cli_effort",
    "_format_cli_tool_use",
    "_is_numeric_zero",
    "_log_delegate_error",
    "_load_tool_utils",
    "_looks_like_context_overflow",
    "_reasoning_signal_chars",
    "_reasoning_text_chars",
    "_reasoning_tokens_from_usage",
    "_read_delegate_state",
    "_resolve_delegate_cwd",
    "_update_delegate_state",
    "_write_delegate_state",
    "log_event",
]
