"""Signal-first continuation decision for the native local engine.

Replaces the content-shape nudge heuristics. Truncation is decided by
``finish_reason`` and stream-termination facts, never by punctuation or
syntax shape.

The decision is a pure function of finish_reason, stream-termination
facts, pending-tool-call state, and delivered/reasoning character
counts — never of the reply's punctuation, syntax, or other content
shape. This decision function remains engine-state-free.

When a turn produces no visible content but some reasoning output, the
engine injects a distinct reasoning-only continuation nudge.
A legitimately terse complete answer is never mistaken for
this case because the gate requires produced_chars == 0
(or verdict_chars == 0 after the last reasoning/tool).
After a reasoning block finishes, the next act must be a
tool call or visible output — thinking duration is not a
stall trigger.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Action(enum.Enum):
    NONE = "none"
    RESUME_TRUNCATED = "resume_truncated"  # mechanical: stream cut off
    RESUME_EMPTY = "resume_empty"  # behavioral: clean stop, nothing delivered
    # behavioral: clean stop on the FIRST round, no tools — the model
    # produced reasoning but zero visible content (reasoning-only turn).
    RESUME_REASONING_ONLY = "resume_reasoning_only"


# action -> migration-compat nudge_kind (what the PrematureTurnEndGate hook and
# the canned nudge bodies key on). NONE maps to None. RESUME_REASONING_ONLY
# rides the SAME "silent" nudge machinery as RESUME_EMPTY (same gate); silent
# continues may retry up to max_nudges. The distinct ``reason_code`` selects a
# reasoning-only body inside ``_continuation_nudge``.
_KIND = {
    Action.RESUME_TRUNCATED: "stream_truncated",
    Action.RESUME_EMPTY: "silent",
    Action.RESUME_REASONING_ONLY: "silent",
    Action.NONE: None,
}


@dataclass(frozen=True)
class ContinuationDecision:
    action: Action
    reason_code: str
    normalized_finish_reason: str
    nudge_kind: str | None


def _d(action: Action, reason_code: str, nfr: str) -> ContinuationDecision:
    return ContinuationDecision(action, reason_code, nfr, _KIND[action])


# Shared cap for idle-empty continues across local LLM, ACP, and CLI engines.
MAX_IDLE_EMPTY_CONTINUES: int = 3

#: Nudging uses TWO budgets, not one, because a single cumulative cap conflates
#: "this model is stuck in a loop" with "this turn has needed help before".
#:
#: Observed (SDLC run 23bf022717d8, deepseek design stage): three
#: reasoning-starved rounds burned the whole cap, the 3rd nudge WORKED and three
#: productive tool rounds followed, then one relapse ended the turn with zero
#: budget left -- on exactly the kind of round the previous nudge had recovered.
#:
#: So the CONSECUTIVE streak is what bounds a loop, and it resets on real
#: progress; a separate lifetime ceiling keeps an alternating
#: productive/empty pattern from nudging forever.
MAX_TURN_NUDGES: int = 6

IDLE_EMPTY_CONTINUE_AFTER_TOOLS: str = (
    "The previous round produced no answer after "
    "your tool work. Treat that as a stall or error, not a finished "
    "turn. Continue from where you left off and write the answer. "
    "Do not restart the task."
)
IDLE_EMPTY_CONTINUE_REASONING: str = (
    "You reasoned but did not produce an answer. "
    "That is a stall, not a finished turn — the operator already saw "
    "the reasoning. Continue and deliver the answer, or call the next "
    "tool you need."
)
IDLE_EMPTY_CONTINUE_BLANK: str = (
    "The previous round produced no answer. "
    "Treat that as an error, not a finished turn. Continue toward the "
    "answer now."
)
IDLE_EMPTY_CONTINUE_STARVED: str = (
    "The previous round ran out of tokens while reasoning and produced no "
    "answer. Treat that as a stall, not a finished turn. Do not start "
    "another long reasoning pass — the same output ceiling will cut it "
    "again. Write the answer now (or call the next tool you need). Do not "
    "restart the task."
)

# Cap for CLI print-timeout continues (Antigravity ``--print-timeout``).
# Separate from idle-empty: a timeout mid-tool-chain is not "silent".
MAX_TIMEOUT_CONTINUES: int = 3

_PRINT_TIMEOUT_MARKERS = (
    "timeout waiting for response",
    "print-timeout",
    "print timeout",
)


def is_cli_print_timeout(message: str) -> bool:
    """True when a CLI result is the print-mode wait wall, not a user abort."""
    low = (message or "").lower()
    return any(marker in low for marker in _PRINT_TIMEOUT_MARKERS)


def antigravity_time_budget_notice(timeout_s: float) -> str:
    """Per-spawn instruction: finish a checkpoint inside the print-mode wall."""
    secs = max(1, int(round(float(timeout_s))))
    return (
        f"[sharedagents] This print-mode turn has a hard wall of {secs}s. "
        "Finish a useful checkpoint and return before that limit. "
        "Break remaining work into smaller tool/task chunks so each "
        "turn completes inside the limit. Do not start a long-running "
        "job you cannot finish in this turn. Keep at least one task in "
        "the foreground until you return."
    )


def cli_timeout_continue_body(timeout_s: float) -> str:
    """Small resume prompt after a CLI print-mode timeout."""
    secs = max(1, int(round(float(timeout_s))))
    return (
        "[sharedagents continuation] The previous print-mode turn hit the "
        f"{secs}s wall before you returned. Resume from the last "
        "completed checkpoint — do not restart the whole task. Finish "
        f"one smaller unit of work that can complete inside {secs}s, "
        "then return a status. Break any remaining work into later "
        "turns. Keep at least one task in the foreground; do not start "
        "a long job you cannot finish in this turn."
    )


def cli_timeout_continue_notice(*, attempt: int, timeout_s: float) -> str:
    """Operator-visible marker that a print-timeout continue was injected."""
    secs = max(1, int(round(float(timeout_s))))
    return f"\n\n↻ runtime: nudge · kind=timeout · reason=print_timeout · wall={secs}s · attempt={attempt}\n\n"


def turn_retry_continue_body(classification: str, message: str) -> str:
    """Prompt for an automatic retry of an attempt that had already acted.

    Replaying the operator's original text after the attempt ran tools or
    answered makes the model perform the request again ("merge to
    main restart and clear this worktree" re-issued twice). This is what the
    operator would type instead: continue, without repeating finished steps.
    """
    reason = " ".join(str(message or "").split())[:200]
    label = str(classification or "error")
    return (
        "[sharedagents retry] The previous attempt of this turn ended with a "
        f"transient {label} error ({reason}). Continue the same request from "
        "where it stopped. Check the current state first, do not repeat "
        "actions that already completed, and then finish or report status."
    )


def idle_empty_continue_body(reason_code: str) -> str:
    """Canned continue prompt for an idle empty final response."""
    if reason_code == "reasoning_only_no_output":
        return IDLE_EMPTY_CONTINUE_REASONING
    if reason_code == "empty_no_output":
        return IDLE_EMPTY_CONTINUE_BLANK
    if reason_code == "thinking_starved_length":
        return IDLE_EMPTY_CONTINUE_STARVED
    return IDLE_EMPTY_CONTINUE_AFTER_TOOLS


def idle_empty_continue_notice(reason_code: str, *, attempt: int) -> str:
    """Operator-visible marker that an idle-empty continue was injected.

    Not counted as model content by the classifier (yielded after the
    round's produced_chars are already recorded).
    """
    return f"\n\n↻ runtime: nudge · kind=silent · reason={reason_code} · attempt={attempt}\n\n"


def idle_empty_resume_prompt(
    *,
    finish_reason: str | None,
    terminal_finish_reason_seen: bool,
    final_content_chars: int,
    turn_visible_chars: int,
    reasoning_chars: int,
    tool_rounds_so_far: int,
    nudge_count: int,
    cancelled: bool = False,
    max_nudges: int = MAX_IDLE_EMPTY_CONTINUES,
    verdict_chars: int | None = None,
) -> tuple[str, str] | None:
    """If this idle stop should continue, return ``(body, reason_code)``.

    Used by ACP and CLI engines so they share ``classify_turn_end`` with
    the local LLM path. Truncation resumes are not returned — those
    engines only retry the idle-empty event (reasoning or tools, then
    no visible output, then stop).
    """
    decision = classify_turn_end(
        finish_reason=finish_reason,
        terminal_finish_reason_seen=terminal_finish_reason_seen,
        stream_transport_error=False,
        final_content_chars=final_content_chars,
        turn_visible_chars=turn_visible_chars,
        reasoning_chars=reasoning_chars,
        pending_tool_calls=False,
        tool_rounds_so_far=tool_rounds_so_far,
        nudge_count=nudge_count,
        max_nudges=max_nudges,
        cancelled=cancelled,
        verdict_chars=verdict_chars,
    )
    if decision.action in (Action.RESUME_EMPTY, Action.RESUME_REASONING_ONLY):
        return idle_empty_continue_body(decision.reason_code), decision.reason_code
    return None


def classify_turn_end(
    *,
    finish_reason: str | None,
    terminal_finish_reason_seen: bool,
    stream_transport_error: bool,
    final_content_chars: int,
    turn_visible_chars: int,
    reasoning_chars: int,
    pending_tool_calls: bool,
    tool_rounds_so_far: int,
    nudge_count: int,
    max_nudges: int,
    cancelled: bool,
    idle_streak: int = 0,
    max_idle_streak: int = MAX_IDLE_EMPTY_CONTINUES,
    min_text_chars: int = 1,
    verdict_chars: int | None = None,
    tool_call_truncated: bool = False,
) -> ContinuationDecision:
    """Decide whether (and why) to inject one continuation round.

    Pure: no engine state. Decision order is documented in the spec section 4.1.

    RESUME_EMPTY fires when the turn delivered ZERO non-whitespace visible
    content after a tool round — the long-standing "silent after tools"
    semantics (the operator sees the tools run and then nothing). This is
    delivery-based, NOT content-shape: a long answer that merely lacks
    terminal punctuation has ``turn_visible_chars`` well above zero and is
    never nudged (the ``stream_truncated`` false positive this redesign
    removes). Terse-but-real answers ("pong", "ok", "done.") are also
    complete deliveries — see the 2026-07-26 PING→PONG duplicate incident.

    Mechanical truncation (``length`` / transport drop / missing terminal
    chunk) is ``stream_truncated`` only when THIS round produced visible
    text to resume (``final_content_chars > 0``). A max-token hit with
    an empty round is thinking-only starvation — "resume from that exact
    point" with an empty tail is the wrong body. Instead the starved shape
    (``length`` + zero content in the FINAL round + non-zero reasoning)
    fires ``thinking_starved_length``: the model was still generating when
    killed and has not delivered anything user-visible, so it gets the
    reasoning-only continuation (live: 38,610 reasoning chars,
    content_chars=0, turn ended "complete" with no nudge). Clean
    ``stop`` with zero content falls through to the silent zero-output
    gates; a turn whose earlier rounds already delivered an answer ends
    ``complete``.

    ``tool_call_truncated`` is the worker's report that the round was still
    writing a tool call's arguments when the cap hit. With ``length`` that
    call never runs, so it resumes as ``tool_call_truncated`` ahead of both
    gates above: resuming a sentence or re-prompting for an answer sends the
    model back to the same oversized call.

    ``min_text_chars`` is retained for call-site compatibility but is no
    longer consulted: both behavioral empty gates require ``produced_chars
    == 0``. Callers may still pass it; it is ignored.
    """
    # Provider robustness: a pending tool call IS a tool turn regardless of the
    # wire label (some OpenAI-compatible servers mislabel as "stop").
    nfr = "tool_calls" if pending_tool_calls else (finish_reason or "")

    # "Did the MODEL produce an answer this turn?" is the authoritative measure
    # for the empty-nudge gate below — and it is the LARGER of what the user saw
    # streamed (``turn_visible_chars``) and the round's final content
    # (``final_content_chars``). They diverge when a real answer exists in the
    # final content but never incremented the visible counter: a delegated
    # sub-agent answer (rendered as the delegate's tool result, not the model's
    # content) or a stream reconciled/truncated after the fact. Nudging the model
    # to produce MORE in those cases is wrong — the answer already exists; the gap
    # is delivery, not production. Prior code ignored ``final_content_chars``
    # entirely, so those turns fired a spurious "empty_after_tools" nudge.
    produced_chars = max(turn_visible_chars, final_content_chars)
    _ = min_text_chars  # retained for API compat; behavioral gates use == 0 only

    if cancelled:
        return _d(Action.NONE, "cancelled", nfr)
    # Lifetime ceiling: the hard bound. An alternating productive/empty pattern
    # resets the streak below forever, so without this it could nudge without
    # end.
    if nudge_count >= max_nudges:
        return _d(Action.NONE, "max_nudges", nfr)
    if pending_tool_calls:
        # Real parsed calls are about to execute — this round is not a turn
        # end. A bare ``finish_reason="tool_calls"`` LABEL with no parsed
        # calls (malformed tool-call JSON) is deliberately NOT this case:
        # nothing will run, the turn is ending, and it must fall through to
        # the zero-output gates below ("stops with a tool call and no output
        # tokens received").
        return _d(Action.NONE, "tool_calls", nfr)

    # MECHANICAL: a tool call cut off by the output cap. The worker dispatches
    # calls only on ``finish_reason=tool_calls``, so the fragment never ran --
    # whatever text or reasoning came before it. Checked before the prose
    # truncation and thinking-starvation gates, whose bodies ("resume the
    # sentence", "you ran out while reasoning") sent the model back to the
    # same oversized call five rounds in a row.
    if finish_reason == "length" and tool_call_truncated:
        return _d(Action.RESUME_TRUNCATED, "tool_call_truncated", nfr)

    # MECHANICAL truncation -- signals only, never content shape, and
    # only when THIS round has visible text to resume from. Turn-level
    # ``produced_chars`` is the wrong gate: a tool chain that already
    # streamed an answer, then a thinking-only ``length`` stop, would
    # still fire ``stream_truncated`` with an empty tail. Zero-content
    # rounds fall through to the silent empty / reasoning-only gates
    # (or ``complete`` when an earlier round already delivered).
    if final_content_chars > 0:
        if finish_reason == "length":
            return _d(Action.RESUME_TRUNCATED, "length", nfr)
        if stream_transport_error:
            return _d(Action.RESUME_TRUNCATED, "transport_error", nfr)
        if not terminal_finish_reason_seen:
            return _d(Action.RESUME_TRUNCATED, "no_terminal", nfr)

    # BEHAVIORAL non-delivery -- the final response is idle (clean stop)
    # after reasoning or a tool chain, with no useful visible output.
    # Pending tool_calls above are not idle: those execute, then this
    # round re-enters. Continue up to max_nudges (default 3).
    #
    # First: reasoning then idle, no content, no tools this turn. Any
    # non-zero reasoning with ZERO visible content is the event — a
    # 20-char think-block that stops is still "the model produced
    # nothing useful". The old ``reasoning_chars > 100`` floor let
    # short reasoning-only stops fall through as complete.
    #
    # GATE TIGHTENED 2026-07-26 (incident: Telegram topic 1842 "PING"→"PONG"):
    # the prior gate ``produced_chars < min_text_chars`` (engine passes
    # ``NUDGE_MIN_TEXT_CHARS=10``) fired for ANY sub-10-char first-round
    # answer with reasoning — including a complete terse answer like
    # "PONG" (4 chars). The nudge then re-prompted the model, which produced
    # a DUPLICATE "PONG", so the operator saw "PONG\n\nPONG". A terse-but-real
    # answer is NOT "no output": the gate fires ONLY when ``produced_chars
    # == 0`` (truly empty / whitespace-only). That empty-content rule is
    # what protects terse answers, not a reasoning-volume floor.
    #
    # Consecutive-idle streak: the loop bound, and it applies ONLY to the
    # silent/idle gates below. It must sit AFTER mechanical truncation: a round
    # that produced visible text and hit `length` is a partial answer to resume,
    # not a nudge loop, and terminating it would lose real work.
    # Reset by real progress (a tool call that SUCCEEDED) -- never by a parsed
    # tool-call label, which a stuck model can emit without achieving anything.
    if idle_streak >= max_idle_streak:
        return _d(Action.NONE, "max_idle_streak", nfr)

    # THINKING-STARVATION cut-off (live 2026-08-18 and 2026-08-31): the FINAL
    # round exhausted its output budget on reasoning and was killed before
    # writing any visible content (``finish_reason=length``,
    # ``final_content_chars==0``, non-zero reasoning). The mid-sentence
    # truncation gate above needs a content tail to resume from and
    # deliberately skips this shape; the zero-output gate below needs the
    # WHOLE turn empty, but narration in earlier rounds ("let me check the
    # numbers") is not an answer. Result: the turn ended silently as
    # "complete" and the operator had to re-prompt by hand. Fire the
    # reasoning-only continuation instead — wire facts only, no content
    # shape. The consecutive-idle streak above bounds
    # repeated starved stops; real progress resets it.
    if finish_reason == "length" and final_content_chars == 0 and reasoning_chars > 0:
        return _d(Action.RESUME_REASONING_ONLY, "thinking_starved_length", nfr)

    # ZERO-OUTPUT GUARANTEE: no prompt may
    # finish without an output. Any terminal round that ends the turn with
    # ``produced_chars == 0`` gets a continuation — reasoning-only stops,
    # post-tool silence, a completely blank EOS, a dangling ``tool_calls``
    # label with nothing parsed, or an unrecognized provider label
    # (``content_filter``, ``repetition`` with the loop detector disabled for
    # the model, provider-specific strings). The gates are therefore NOT
    # keyed on ``nfr == "stop"``: the label cannot make silence acceptable.
    # ``produced_chars == 0`` keeps the false-positive surface at zero — a
    # delivered answer (however terse) never re-prompts; the cancelled and
    # max_nudges short-circuits above still bound the path.
    if produced_chars == 0:
        if tool_rounds_so_far == 0 and reasoning_chars > 0:
            return _d(
                Action.RESUME_REASONING_ONLY,
                "reasoning_only_no_output",
                nfr,
            )
        # Empty after one or more tool rounds. Same "truly empty only" rule
        # as the first-round gate — a short complete answer after tools
        # ("pong", "ok", "done.") is produced_chars > 0 and never lands here
        # (owner: ping→nudge FP).
        if tool_rounds_so_far >= 1:
            return _d(Action.RESUME_EMPTY, "empty_after_tools", nfr)
        # Nothing at all: no content, no reasoning, no tools this turn.
        return _d(Action.RESUME_EMPTY, "empty_no_output", nfr)

    # Post-reasoning / post-tool contract: earlier progress chatter is
    # not a verdict if the last act was thinking or a tool with no
    # later content. Thinking duration is irrelevant — PromptResponse
    # after a finished reasoning block is the stall event.
    if verdict_chars is not None and verdict_chars == 0 and (reasoning_chars > 0 or tool_rounds_so_far >= 1):
        if tool_rounds_so_far >= 1:
            return _d(Action.RESUME_EMPTY, "empty_after_tools", nfr)
        return _d(
            Action.RESUME_REASONING_ONLY,
            "reasoning_only_no_output",
            nfr,
        )

    return _d(Action.NONE, "complete", nfr)


# Tools that are not progress. A trailing bookkeeping call after a real
# answer must not clear the verdict (2026-09-10 local-LLM merge turn).
BOOKKEEPING_TOOLS: frozenset[str] = frozenset(
    {
        "delegate_teacher",
        "goal",
    }
)


def _tool_is_bookkeeping(name: str) -> bool:
    bare = (name or "").split(".")[-1].strip()
    return bare in BOOKKEEPING_TOOLS or bare.startswith("delegate_teacher")


def _tool_is_delivery(name: str) -> bool:
    """CreatePlan / ExitPlanMode publish the turn's artifact."""
    from sharedagents_agent.tools.widget_adapt import is_plan_tool_name

    return is_plan_tool_name(name)


@dataclass
class TurnEventTracker:
    """Engine-agnostic stream facts for hooks and idle-empty nudges."""

    _round_content: list[str] = field(default_factory=list)
    _verdict: list[str] = field(default_factory=list)
    _thinking: list[str] = field(default_factory=list)
    _extra_reasoning: int = 0
    turn_visible_chars: int = 0
    tool_rounds: int = 0
    round_idx: int = 0
    delivered_via_tool: bool = False
    tools_this_round: list[dict[str, Any]] = field(default_factory=list)

    def begin_round(self) -> None:
        # Per-round buffers only. Verdict persists until on_thinking or a
        # non-bookkeeping on_tool — a later empty generation after a real
        # answer (plus trailing delegate_teacher) must not look silent.
        self._round_content = []
        self._thinking = []
        self._extra_reasoning = 0
        self.tools_this_round = []

    def on_content(self, text: str) -> None:
        if text:
            self._round_content.append(text)
            self._verdict.append(text)

    def on_thinking(self, text: str) -> None:
        if text:
            self._thinking.append(text)
            self._verdict = []

    def on_reasoning_chars(self, chars: int) -> None:
        if chars > 0:
            self._extra_reasoning += int(chars)

    def on_tool(
        self,
        name: str = "",
        *,
        ok: bool | None = None,
        arguments: dict[str, Any] | None = None,
        result: str = "",
    ) -> None:
        self.tools_this_round.append(
            {
                "name": name or "",
                "ok": True if ok is None else bool(ok),
                "arguments": dict(arguments or {}),
                "result": result or "",
            }
        )
        if _tool_is_delivery(name) and (ok is not False):
            self.delivered_via_tool = True
            return
        if not _tool_is_bookkeeping(name):
            self._verdict = []
            self.tool_rounds += 1

    def commit_round(self) -> None:
        self.turn_visible_chars += self.final_content_chars
        self.round_idx += 1

    @property
    def verdict_chars(self) -> int:
        return len("".join(self._verdict).strip())

    @property
    def final_content_chars(self) -> int:
        return len("".join(self._round_content).strip())

    @property
    def reasoning_chars(self) -> int:
        return len("".join(self._thinking)) + self._extra_reasoning

    @property
    def has_work(self) -> bool:
        """True when this turn already thought or ran a non-bookkeeping tool.

        An ACP/CLI idle timeout with no work is a dead peer, not an
        empty-answer nudge.
        """
        return self.reasoning_chars > 0 or self.tool_rounds > 0

    def hook_payload(
        self,
        *,
        kind: str,
        nudge_count: int,
        should_fire_default: bool,
        tail: str = "",
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "tail": tail,
            "final_text_chars": self.final_content_chars,
            "verdict_chars": self.verdict_chars,
            "tool_rounds_so_far": self.tool_rounds,
            "nudge_count": nudge_count,
            "round_idx": self.round_idx,
            "should_fire_default": should_fire_default,
        }

    def decide_stop(
        self,
        *,
        finish_reason: str | None,
        cancelled: bool = False,
        nudge_count: int = 0,
        max_nudges: int = MAX_IDLE_EMPTY_CONTINUES,
        pending_tool_calls: bool = False,
        idle_streak: int = 0,
        terminal_finish_reason_seen: bool = True,
        stream_transport_error: bool = False,
        tool_call_truncated: bool = False,
    ) -> ContinuationDecision:
        if self.delivered_via_tool:
            return _d(Action.NONE, "complete", finish_reason or "")
        return classify_turn_end(
            finish_reason=finish_reason,
            terminal_finish_reason_seen=terminal_finish_reason_seen,
            stream_transport_error=stream_transport_error,
            final_content_chars=self.final_content_chars,
            turn_visible_chars=max(self.turn_visible_chars, self.final_content_chars),
            reasoning_chars=self.reasoning_chars,
            pending_tool_calls=pending_tool_calls,
            tool_rounds_so_far=self.tool_rounds,
            nudge_count=nudge_count,
            max_nudges=max_nudges,
            cancelled=cancelled,
            idle_streak=idle_streak,
            verdict_chars=self.verdict_chars,
            tool_call_truncated=tool_call_truncated,
        )


@dataclass
class NudgePlan:
    decision: ContinuationDecision
    body: str | None
    suppressed: bool = False
    kind: str | None = None
    tail: str | None = None
    system_messages: list[str] = field(default_factory=list)

    @property
    def is_idle_resume(self) -> bool:
        return self.body is not None and self.decision.action in (
            Action.RESUME_EMPTY,
            Action.RESUME_REASONING_ONLY,
        )


def _hook_kind_tail(outcome: Any, *, authorized: bool) -> tuple[str | None, str | None]:
    if not authorized or outcome is None:
        return None, None
    kind = None
    tail = None
    for r in reversed(getattr(outcome, "results", None) or []):
        j = getattr(r, "json_output", None)
        if not isinstance(j, dict):
            continue
        hso = j.get("hookSpecificOutput")
        if not isinstance(hso, dict):
            continue
        k = hso.get("kind")
        if isinstance(k, str) and k.strip():
            kind = k.strip()
        t = hso.get("tail")
        if isinstance(t, str):
            tail = t
        break
    return kind, tail


async def apply_nudge_hooks(
    tracker: TurnEventTracker,
    *,
    hook_engine: Any = None,
    hook_ctx: dict[str, Any] | None = None,
    finish_reason: str | None = "stop",
    cancelled: bool = False,
    nudge_count: int = 0,
    max_nudges: int = MAX_IDLE_EMPTY_CONTINUES,
    pending_tool_calls: bool = False,
    idle_streak: int = 0,
    fire_hook: Any = None,
    terminal_finish_reason_seen: bool = True,
    stream_transport_error: bool = False,
    tool_call_truncated: bool = False,
) -> NudgePlan:
    """Classify, then PrematureTurnEndGate, then PrematureTurnEnd.

    Hooks may suppress or substitute an already-authorized resume.
    They cannot invent a nudge on ``Action.NONE`` (PONG rule).
    """
    from sharedagents_agent.hooks import fire_hook_event

    fire = fire_hook or fire_hook_event
    decision = tracker.decide_stop(
        finish_reason=finish_reason,
        cancelled=cancelled,
        nudge_count=nudge_count,
        max_nudges=max_nudges,
        pending_tool_calls=pending_tool_calls,
        idle_streak=idle_streak,
        terminal_finish_reason_seen=terminal_finish_reason_seen,
        stream_transport_error=stream_transport_error,
        tool_call_truncated=tool_call_truncated,
    )
    authorized_idle = decision.action in (
        Action.RESUME_EMPTY,
        Action.RESUME_REASONING_ONLY,
    )
    authorized = authorized_idle or decision.action is Action.RESUME_TRUNCATED
    kind = decision.nudge_kind or ""
    tail = ""
    messages: list[str] = []
    override: str | None = None

    payload = tracker.hook_payload(
        kind=kind,
        nudge_count=nudge_count,
        should_fire_default=authorized,
        tail=tail,
    )
    gate = await fire(hook_engine, hook_ctx, "PrematureTurnEndGate", payload)
    messages.extend(getattr(gate, "system_messages", None) or [])
    gk, gt = _hook_kind_tail(gate, authorized=authorized)
    if gk:
        kind = gk
    if gt is not None:
        tail = gt
    if not getattr(gate, "should_continue", True):
        return NudgePlan(
            decision=decision,
            body=None,
            suppressed=True,
            kind=kind or None,
            tail=tail or None,
            system_messages=messages,
        )
    if authorized and getattr(gate, "blocked", False) and getattr(gate, "block_reason", ""):
        override = gate.block_reason

    pe_payload = tracker.hook_payload(
        kind=kind,
        nudge_count=nudge_count,
        should_fire_default=authorized,
        tail=tail,
    )
    pe = await fire(hook_engine, hook_ctx, "PrematureTurnEnd", pe_payload)
    messages.extend(getattr(pe, "system_messages", None) or [])
    if not getattr(pe, "should_continue", True):
        return NudgePlan(
            decision=decision,
            body=None,
            suppressed=True,
            kind=kind or None,
            tail=tail or None,
            system_messages=messages,
        )
    if authorized and getattr(pe, "blocked", False) and getattr(pe, "block_reason", ""):
        override = pe.block_reason

    if not authorized:
        return NudgePlan(
            decision=decision,
            body=None,
            suppressed=False,
            kind=kind or None,
            tail=tail or None,
            system_messages=messages,
        )
    # Idle-empty only. Truncation keeps hook suppress/substitute for local
    # LLM; ACP/CLI must not inject the after-tools canned body on length.
    if not authorized_idle:
        return NudgePlan(
            decision=decision,
            body=override,
            suppressed=False,
            kind=kind or None,
            tail=tail or None,
            system_messages=messages,
        )
    body = override if override else idle_empty_continue_body(decision.reason_code)
    return NudgePlan(
        decision=decision,
        body=body,
        suppressed=False,
        kind=kind or None,
        tail=tail or None,
        system_messages=messages,
    )
