"""Transcript message widgets for the sharedagents TUI (extracted from app.py)."""

from __future__ import annotations

import difflib
import json
import time

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.widgets import Collapsible, Static

from sharedagents_tui.helpers import (
    _THINK_TICK_INTERVAL_S,
    _build_metrics_markup,
    _default_tool_result_collapsed,
    _truncate_smart,
)


class RoundMetricsMessage(Static):
    """Standalone per-round telemetry line.

    Mounted as its OWN transcript row AFTER the round's tool call /
    result widgets, so the per-round output-token stats read as the
    round's closing line instead of appearing above its tools. (The
    final, no-tool round still stamps its cumulative line inline on the
    assistant bubble via :meth:`AssistantMessage.set_metrics`.)
    """

    def __init__(self, metrics: dict) -> None:
        super().__init__(
            _build_metrics_markup(metrics),
            markup=True,
            classes="assist-metrics round-metrics",
        )
        self.metrics = dict(metrics)


class UserMessage(Static):
    def __init__(self, text: str) -> None:
        super().__init__(text, markup=False, classes="user-msg")


class AssistantMessage(Vertical):
    """One assistant turn: optional collapsible thinking + streamed content.

    Layout order (top → bottom): prefill-counter → thinking-box →
    content → metrics. The prefill counter is a small live-ticking
    line that's visible from the moment the widget mounts, so a fresh
    turn shows "yes, work is happening" feedback during the multi-second
    prefill window before the first token arrives. It freezes the
    instant the first thinking or content delta lands and stays as a
    quiet `[dim]· prefill 1.23s[/dim]` breadcrumb above the thinking box.
    """

    def __init__(self, *, from_replay: bool = False) -> None:
        super().__init__(classes="assist-msg")
        self.thinking_text = ""
        self.content_text = ""
        self._thinking_static: Static | None = None
        self._thinking_box: Collapsible | None = None
        self._content_static = Static("", markup=False, classes="assist-content")
        # Per-model-call telemetry footer — one line of dim stats stamped
        # below the content when the bridge ships a `metrics` event for
        # this round. Each AssistantMessage corresponds to ONE model call
        # (a "round"), so this is per-call, not per-turn.
        self._metrics_static = Static("", markup=True, classes="assist-metrics")
        self._prefill_static = Static("", markup=True, classes="assist-prefill")
        self._prefill_started: float | None = None
        self._prefill_ended: float | None = None
        self._prefill_timer = None
        self._from_replay = from_replay
        self._final = False
        self._think_started: float | None = None
        self._think_ended: float | None = None
        # Monotonic receipt time of the LATEST thinking delta. The frozen
        # "Thought for Xs" title measures first→last thinking delta (NOT
        # first-delta→first-content) so it matches the reader's persisted
        # ``thinking_ms`` (a thinking_delta ts-span) that the passive
        # frontend replays — otherwise the driver over-counted by the
        # gap between the last reasoning token and the first content
        # token. (turn-replication: thinking-duration parity)
        self._think_last_delta_mono: float | None = None
        self._think_timer = None
        self.metrics: dict | None = None
        # Replay/render path: when set, the thinking title shows these
        # PERSISTED values instead of a (meaningless) monotonic re-measure
        # of an already-finished stream — so a passive frontend's
        # "Thought for Xs · N chars" matches the driver exactly.
        self._replay_think_ms: float | None = None
        self._replay_think_chars: int | None = None

    def compose(self) -> ComposeResult:
        if not self._from_replay:
            yield self._prefill_static
        yield self._content_static
        yield self._metrics_static

    def on_mount(self) -> None:
        """Kick off the prefill counter immediately on mount.

        Mounting happens at round-start (eager mount in `_stream_via_agent`
        and after `tool_result`), so the elapsed time we display matches
        the user's perceived "how long since I hit Enter" / "how long
        since the tool returned and the model went quiet again". Replay
        path skips this — its pft is already known and shown in metrics.
        """
        if self._from_replay:
            return
        self._prefill_started = time.monotonic()
        self._refresh_prefill_label(active=True)
        self._prefill_timer = self.set_interval(_THINK_TICK_INTERVAL_S, self._tick_prefill)

    def set_metrics(self, metrics: dict) -> None:
        """Render a one-line dim footer with turn-level telemetry.

        In the live streaming path this is called once per turn (from
        the end-of-turn ``finally`` block), with **cumulative** turn
        metrics — ``completion_tokens`` summed across all rounds,
        ``prompt_tokens`` from the final (already-cumulative) round,
        ``total_tokens`` recalculated as cumulative prompt + cumulative
        completion. Per-round ``usage_update`` events (from
        ``RoundComplete``) deliberately do NOT call ``set_metrics``:
        they arrive after the next round's assistant has been mounted,
        so mid-stream stamping would put per-round totals on the wrong
        bubble (see the ``event == "metrics"`` handler).

        On replay (``from_replay=True``), the saved turn's ``metrics``
        dict is stamped directly — this carries the cumulative
        turn-level values persisted by the engine audit update.

        Order (left → right):
            <model> · classify <ms> · plan <ms>
              · prefill <ms> @ <prefill_tps> tok/s · ttft <ms>
              · gen <gen_tps> tok/s
              · ctx in <pt> (<cached> cached, <fresh> fresh) /<window>
              · out <ct> tok (think <rt> / result <ot>)
              · total <total> tok · wt <s>

        Fields are skipped when their value is missing.
        """
        if not metrics:
            return
        self.metrics = dict(metrics)
        # Operator-facing alias resolved by the bridge from the serving
        # endpoint is folded in by ``_build_metrics_markup`` (rendered as
        # ``<alias>/<model>`` so the user sees BOTH the bucket they picked
        # AND the wire-level model id). The full field layout lives there
        # so this inline line and the standalone per-round line
        # (``RoundMetricsMessage``) can never drift apart.
        markup = _build_metrics_markup(metrics)
        if markup:
            self._metrics_static.update(markup)

    async def append_thinking(self, text: str) -> None:
        if not text:
            return
        self._freeze_prefill()
        now = time.monotonic()
        if self._think_started is None:
            self._think_started = now
        # Track the latest thinking delta so the frozen title ends at the
        # last reasoning token (mirrors the reader's ts-span).
        self._think_last_delta_mono = now
        self.thinking_text += text
        if self._thinking_box is None:
            self._thinking_static = Static("", markup=False, classes="thinking-body")
            self._thinking_box = Collapsible(
                self._thinking_static,
                title=self._think_title(active=True),
                collapsed=True,
                classes="thinking",
            )
            await self.mount(self._thinking_box, before=self._content_static)
            self._think_timer = self.set_interval(_THINK_TICK_INTERVAL_S, self._tick_think_title)
        assert self._thinking_static is not None
        self._thinking_static.update(self.thinking_text)
        self._refresh_think_title(active=True)

    async def append_thinking_replay(
        self,
        text: str,
        *,
        duration_ms: float | None = None,
        reasoning_chars: int | None = None,
    ) -> None:
        """Replay/render a finished thinking generation with PERSISTED
        display values. Unlike :meth:`append_thinking` (which clocks the
        live stream), this stamps the duration + char count the driver
        showed, so the title is identical across frontends. Mounts the
        collapsed thinking box in the same default state as live.
        (turn-replication: exact parity)
        """
        if not text:
            return
        self._replay_think_ms = duration_ms
        self._replay_think_chars = reasoning_chars
        self.thinking_text = text
        if self._thinking_box is None:
            self._thinking_static = Static("", markup=False, classes="thinking-body")
            self._thinking_box = Collapsible(
                self._thinking_static,
                title="",
                collapsed=True,
                classes="thinking",
            )
            await self.mount(self._thinking_box, before=self._content_static)
        assert self._thinking_static is not None
        self._thinking_static.update(self.thinking_text)
        # Force the finalized "✓ Thought for" title (no live timer).
        if self._think_started is None:
            self._think_started = time.monotonic()
        self._think_ended = self._think_started
        self._refresh_think_title(active=False)

    async def apply_canonical_thinking(
        self,
        thinking_text: str,
        duration_ms: float | None,
        reasoning_chars: int | None,
    ) -> None:
        """Reconcile this bubble's thinking box to the persisted
        (canonical) folded segment at end-of-turn, so the driver's
        finalized bubble equals what a passive frontend replays EXACTLY.

        Two cases:

        * A box was shown live → stamp the canonical duration + char
          count. The live "Thought for Xs" is measured monotonically as
          deltas arrive, but SSH-throttled paint (20 fps) inflates that
          span vs the reader's ``thinking_ms`` (a thinking_delta
          ts-span); overwrite it.
        * NO box was shown live but the folded segment HAS reasoning.
          This happens when the live stream delivered the content delta
          before/without the reasoning delta for a short turn, so the
          driver never mounted a box — yet the engine persisted the
          reasoning, so the renderer folds it into a thinking box. Mount
          the box now from the persisted text, matching the renderer's
          ``append_thinking_replay`` exactly (same mount condition:
          non-empty thinking).
        (turn-replication: thinking-box parity, driver side)
        """
        if self._thinking_box is None:
            if thinking_text:
                await self.append_thinking_replay(
                    thinking_text,
                    duration_ms=duration_ms,
                    reasoning_chars=reasoning_chars,
                )
            return
        if not thinking_text:
            # Canonical reasoning is empty (e.g. the persisted thinking
            # was scrubbed) — the renderer mounts NO box (its condition is
            # ``if th:``), so hide ours for parity AND so the driver
            # doesn't keep showing reasoning that was scrubbed from the
            # log. (turn-replication: thinking-box parity + scrub safety)
            try:
                self._thinking_box.display = False
            except Exception:  # noqa: BLE001
                pass
            return
        # Box exists: align its body text to the canonical persisted
        # reasoning too (the live stream may have shown a partial /
        # differently-scrubbed version), then stamp duration + chars so
        # the collapsed title matches the renderer.
        if thinking_text and thinking_text != self.thinking_text:
            self.thinking_text = thinking_text
            if self._thinking_static is not None:
                self._thinking_static.update(thinking_text)
        if duration_ms is not None:
            self._replay_think_ms = duration_ms
        if reasoning_chars is not None:
            self._replay_think_chars = reasoning_chars
        self._refresh_think_title(active=False)

    def append_content(self, text: str) -> None:
        if not text:
            return
        self._freeze_prefill()
        if self._think_started is not None and self._think_ended is None:
            self._end_thinking()
        self.content_text += text
        # Late-chunk handling: bridge sometimes emits trailing text
        # chunks AFTER the next ``tool_call`` event, which triggered
        # finalize(). Old behaviour dropped those chunks (``if _final:
        # return``) — the operator saw a mid-word cutoff like ``with
        # correct `name` par`` because "ameter." landed after finalize.
        # Now we still accept the chunk, append to ``content_text``,
        # and re-render through RichMarkdown if finalize already
        # promoted us to markdown.
        if self._final:
            try:
                self._content_static.update(RichMarkdown(self.content_text))
            except Exception:  # noqa: BLE001
                self._content_static.update(self.content_text)
        else:
            self._content_static.update(self.content_text)
        # Force a layout pass on every chunk so the Static grows in
        # lockstep with the streamed text — under SSH-throttled 20fps
        # the implicit paint cycle can lag a couple of seconds,
        # creating the visible "sentence cut off before the tool widget
        # appears" symptom. The Static still has the full content in
        # ``content_text``; the issue was purely layout/paint timing.
        try:
            self._content_static.refresh(layout=True)
            self.refresh(layout=True)
        except Exception:  # noqa: BLE001
            pass

    def finalize(self) -> None:
        self._freeze_prefill()
        self._hide_prefill_breadcrumb()
        if self._think_started is not None and self._think_ended is None:
            self._end_thinking()
        self._final = True
        if self.content_text:
            # Replace the streamed plain-text view with a rendered Markdown
            # block. Force a layout refresh on BOTH the inner Static and
            # the outer AssistantMessage — without the outer refresh, the
            # parent VerticalScroll holds the height it computed for the
            # plain-text Static and the next widget (a ToolCallMessage in
            # the tool-call path) appears to overlap or push out the last
            # streamed lines. This is the "messages not flushing right
            # before a tool call" bug — the text was there in `content_text`
            # but the widget hadn't grown to render the last lines.
            self._content_static.update(RichMarkdown(self.content_text))
            self._content_static.display = True
            self._content_static.refresh(layout=True)
            self.refresh(layout=True)
        else:
            # Thinking-only / tool-only rounds leave an empty content
            # Static that still reserved a blank band between the
            # thinking header and the next 🔧 pill.
            try:
                self._content_static.update("")
                self._content_static.display = False
            except Exception:  # noqa: BLE001
                pass

    def mark_aborted(self) -> None:
        self._freeze_prefill()
        self._hide_prefill_breadcrumb()
        if self._think_started is not None and self._think_ended is None:
            self._end_thinking()
        self._final = True
        suffix = "\n\n_[aborted]_" if self.content_text else "_[aborted]_"
        body = self.content_text + suffix
        self._content_static.update(RichMarkdown(body))

    def mark_error(self, message: str) -> None:
        self._freeze_prefill()
        self._hide_prefill_breadcrumb()
        if self._think_started is not None and self._think_ended is None:
            self._end_thinking()
        self._final = True
        body = (self.content_text + "\n\n" if self.content_text else "") + f"**[error]** {message}"
        self._content_static.update(RichMarkdown(body))

    # --- prefill (time-to-first-token) helpers ---

    def _hide_prefill_breadcrumb(self) -> None:
        """Clear the live prefill breadcrumb at end-of-turn.

        The breadcrumb (``▶ prefilling…`` → ``· prefill 1.04s``) is a
        LIVE-only affordance: ``compose`` doesn't even mount it on
        ``from_replay=True``, so a replayed/rendered turn never shows it.
        Clearing it when the driver's turn finalizes makes the driver's
        completed bubble match the passive frontend exactly — the
        prefill/ttft numbers still live on the metrics line below.
        (turn-replication: prefill-breadcrumb parity)
        """
        try:
            # `.assist-prefill` has `height: 1`, so an empty Static would
            # still occupy a blank row — but the renderer's `compose` never
            # mounts the breadcrumb on `from_replay`, so it has NO row.
            # `display = False` removes this widget from layout (0 rows),
            # matching the renderer exactly instead of leaving a stray blank
            # line above the thinking box.
            self._prefill_static.update("")
            self._prefill_static.display = False
        except Exception:  # noqa: BLE001
            pass

    def _refresh_prefill_label(self, *, active: bool) -> None:
        if self._prefill_started is None:
            return
        end = time.monotonic() if active else (self._prefill_ended or time.monotonic())
        elapsed = max(0.0, end - self._prefill_started)
        if active:
            # Bright-green pulse so it reads as "running, normal operation"
            # rather than an error/warning. The trailing dots cycle 1→2→3
            # so even on a slow tick rate (SSH: 2 Hz) the user sees motion
            # — and we never render zero dots, which would otherwise read
            # as "frozen" for a third of the cycle.
            phase = int(elapsed * 3) % 3 + 1
            dots = "." * phase + " " * (3 - phase)
            self._prefill_static.update(
                f"[bold green]▶ prefilling[/bold green][green]{dots}[/green] [green]{elapsed:.1f}s[/green]"
            )
        else:
            self._prefill_static.update(f"[dim]· prefill {elapsed:.2f}s[/dim]")

    def _tick_prefill(self) -> None:
        if self._prefill_ended is None:
            self._refresh_prefill_label(active=True)

    def _freeze_prefill(self) -> None:
        """Stop the live counter and freeze the line as a small breadcrumb.

        Called on the first thinking/content delta, on finalize, and on
        abort/error. Idempotent — late freezes from finalize/error after
        the first delta already froze it are no-ops.
        """
        if self._prefill_started is None or self._prefill_ended is not None:
            return
        self._prefill_ended = time.monotonic()
        if self._prefill_timer is not None:
            self._prefill_timer.stop()
            self._prefill_timer = None
        self._refresh_prefill_label(active=False)

    # --- thinking-title helpers ---

    def _think_title(self, *, active: bool) -> str:
        if self._think_started is None:
            return "Thinking"
        end = time.monotonic() if active else (self._think_ended or time.monotonic())
        elapsed = max(0.0, end - self._think_started)
        chars = len(self.thinking_text)
        # Hint flips with the box's current state so the user sees what Ctrl+R
        # will do at this moment.
        collapsed = self._thinking_box.collapsed if self._thinking_box is not None else True
        hint = "Ctrl+R to expand" if collapsed else "Ctrl+R to collapse"
        if active:
            return f"▶ Thinking… {elapsed:.1f}s · {chars} chars   ·   {hint}"
        # Replay/render: use persisted display values for exact parity
        # with the driver's finalized title.
        if self._replay_think_ms is not None:
            elapsed = self._replay_think_ms / 1000.0
        if self._replay_think_chars is not None:
            chars = self._replay_think_chars
        # A 0.0s stamp with non-empty thinking (common when replay sets
        # start==end without a persisted ms) reads as a bug — omit the
        # bogus duration and keep the char count.
        if elapsed < 0.05 and chars > 0:
            return f"✓ Thought · {chars} chars   ·   {hint}"
        return f"✓ Thought for {elapsed:.1f}s · {chars} chars   ·   {hint}"

    def _refresh_think_title(self, *, active: bool) -> None:
        if self._thinking_box is None:
            return
        self._thinking_box.title = self._think_title(active=active)

    def _tick_think_title(self) -> None:
        if self._think_ended is None:
            self._refresh_think_title(active=True)

    def _end_thinking(self) -> None:
        # Freeze the title at the LAST thinking delta, not "now" (which
        # would fold the wait-for-first-content gap into the reasoning
        # duration). This matches the reader's persisted ts-span so the
        # driver's "Thought for Xs" equals the passive frontend's.
        self._think_ended = self._think_last_delta_mono or time.monotonic()
        if self._think_timer is not None:
            self._think_timer.stop()
            self._think_timer = None
        self._refresh_think_title(active=False)


# NOTE: the sharedagents-context preamble that used to be built here
# (`_build_delegate_preamble`) moved to the BACKEND
# (`LocalLLMEngine._run_direct_tool_call` →
# `_build_direct_delegate_preamble`). Context injection is a backend
# concern so every frontend (TUI, Telegram, …) gets it uniformly; the
# TUI now sends the bare `/claude` / `/codex` / `/antigravity` prompt.


class ClaudeMessage(AssistantMessage):
    """Inline transcript widget for ``/claude <prompt>`` output.

    Subclass of :class:`AssistantMessage` so the layout — prefill
    counter → optional thinking box → content → metrics row — is
    visually identical to a native sharedagents turn. The ONLY difference
    is a one-line dim ``↳ via claude`` badge mounted above the
    content, so the operator can tell the answer was delegated to
    Claude Code without breaking the "just another turn" rhythm of
    the transcript.

    Public surface (kept compatible with the old Static-based widget):
    ``append_content(text)`` / ``finalize(session_id=None)`` /
    ``mark_aborted()`` / ``mark_error(message)`` and the
    ``content_text`` attribute.
    """

    DELEGATE_LABEL = "claude"
    DELEGATE_ID_LABEL = "session_id"

    def __init__(self, *, from_replay: bool = False) -> None:
        super().__init__(from_replay=from_replay)
        self._delegate_session_id: str | None = None
        self._aborted = False
        self._error_message: str | None = None
        # Lazily-mounted Static carrying the ``↳ via <delegate>`` badge.
        # We construct it now so it can be styled via CSS (`.delegate-badge`)
        # but mount it in ``on_mount`` (the parent class already mounts the
        # prefill / content / metrics children there).
        self._delegate_badge = Static(
            self._badge_text(),
            markup=True,
            classes="delegate-badge",
        )

    def _badge_text(self) -> str:
        # Inert dim, low-contrast — same vibe as the metrics row.
        return f"[dim]↳ via {self.DELEGATE_LABEL}[/dim]"

    def on_mount(self) -> None:
        super().on_mount()
        # Mount the delegate badge ABOVE the content static so it
        # sits between the prefill counter (or the top of the widget)
        # and the streaming content. ``before=self._content_static``
        # is the same anchor the thinking box uses when it appears.
        try:
            self.mount(self._delegate_badge, before=self._content_static)
        except Exception:  # noqa: BLE001
            # Mount can race with very-early lifecycle hooks; the badge
            # is purely visual, so a failure here is non-fatal.
            pass

    def finalize(self, session_id: str | None = None) -> None:
        self._delegate_session_id = session_id or None
        if self._aborted:
            self._render_aborted_inline()
            return
        if self._error_message is not None:
            self._render_error_inline(self._error_message)
            return
        super().finalize()
        # Append a dim trailer with the delegate session id so a
        # follow-up ``/claude`` can be grounded by the operator if they
        # need to debug. Same place the old widget put it.
        if session_id:
            trailer = f"\n\n[dim]_{self.DELEGATE_ID_LABEL}: {session_id}_[/dim]"
            try:
                self._content_static.update(RichMarkdown(self.content_text + trailer))
                self.refresh(layout=True)
            except Exception:  # noqa: BLE001
                pass

    def mark_aborted(self) -> None:
        self._aborted = True
        self._render_aborted_inline()

    def mark_error(self, message: str) -> None:
        self._error_message = message
        self._render_error_inline(message)

    def _render_aborted_inline(self) -> None:
        body = self.content_text + "\n\n_[aborted]_"
        try:
            self._content_static.update(RichMarkdown(body))
            self._final = True
            self._freeze_prefill()
            self.refresh(layout=True)
        except Exception:  # noqa: BLE001
            pass

    def _render_error_inline(self, message: str) -> None:
        prefix = (self.content_text + "\n\n") if self.content_text else ""
        body = f"{prefix}**[error]** {message}"
        try:
            self._content_static.update(RichMarkdown(body))
            self._final = True
            self._freeze_prefill()
            self.refresh(layout=True)
        except Exception:  # noqa: BLE001
            pass


class CodexMessage(ClaudeMessage):
    """Inline transcript widget for ``/codex <prompt>`` output.

    Same layout/streaming contract as :class:`ClaudeMessage` — only
    the badge label changes to ``↳ via codex`` and the trailing
    session-id label uses Codex's ``thread_id`` term.
    """

    DELEGATE_LABEL = "codex"
    DELEGATE_ID_LABEL = "thread_id"


class AntigravityMessage(ClaudeMessage):
    """Inline transcript widget for ``/antigravity <prompt>`` output.

    Same layout/streaming contract as :class:`ClaudeMessage` — only
    the badge label changes to ``↳ via antigravity`` and the trailing
    session-id field stays as ``session_id`` (Antigravity CLI's ``--resume``
    takes a UUID just like Claude).
    """

    DELEGATE_LABEL = "antigravity"
    DELEGATE_ID_LABEL = "session_id"


class ZCodeMessage(ClaudeMessage):
    """Inline transcript widget for ``/zcode <prompt>`` output.

    Same layout/streaming contract as :class:`ClaudeMessage` — only the
    badge label changes to ``↳ via zcode``. The trailing session-id field
    stays ``session_id`` (the ZCode CLI's ``--resume`` takes its own
    ``sess_…`` id).
    """

    DELEGATE_LABEL = "zcode"
    DELEGATE_ID_LABEL = "session_id"


# `/claude`, `/codex`, and `/antigravity` send the bare prompt; the gateway
# backend (LocalLLMEngine._run_direct_tool_call) injects the
# <sharedagents-context> preamble and forwards only the turns the delegate
# hasn't seen since its last same-kind call.


# Cursor/ACP metadata keys that must never appear in the 🔧 pill.
_TOOL_CALL_META_KEYS = frozenset({"_toolName", "toolName", "criticality"})


def format_tool_call_pill(name: str, args: dict | None) -> str:
    """Build the one-line ``🔧 name {args}`` transcript label.

    Cursor ACP often ships a shell command as ``name`` (markdown-fenced)
    and/or ``arguments={"_toolName": "updateTodos"}``. Either shape made
    the TUI look broken (multi-line fake user blocks, useless JSON).
    """
    raw_args = dict(args or {})
    clean = {k: v for k, v in raw_args.items() if k not in _TOOL_CALL_META_KEYS}
    label = (name or "tool").strip() or "tool"
    cmd = clean.get("command")
    if isinstance(cmd, str) and cmd.strip():
        bare = label.strip("`").strip()
        if len(label) > 60 or "\n" in label or (label.startswith("`") and label.endswith("`")) or bare == cmd.strip():
            label = "Shell"
    elif len(label) > 60 or "\n" in label:
        label = "tool"
    elif label.startswith("`") and label.endswith("`") and label.count("`") == 2:
        inner = label[1:-1]
        label = inner if len(inner) <= 40 else "tool"
    if not clean:
        return f"🔧 {label}"
    compact = _truncate_smart(json.dumps(clean, ensure_ascii=False), 200)
    return f"🔧 {label} {compact}"


class ToolCallMessage(Static):
    """Inline transcript pill: '🔧 tool(args)' shown when a tool starts running."""

    def __init__(self, name: str, args: dict) -> None:
        super().__init__(
            format_tool_call_pill(name, args),
            markup=False,
            classes="tool-call",
        )


class ToolResultMessage(Collapsible):
    """Tool-call output, collapsed by default.

    Subclasses Textual's `Collapsible` directly (not wrapped in a
    Vertical) so the visual is identical to `AssistantMessage`'s
    `_thinking_box`: collapsed-by-default header bar with a peek of
    the output, expand on `Ctrl+T`. All tool outputs — including
    streaming ones — start collapsed. Content accumulates in memory
    regardless of collapsed state; expanding mid-stream reveals
    everything received so far. Long outputs (a 64 KiB grep result,
    a web-fetched HTML page) stay folded out of the way until the
    user chooses to inspect them.
    """

    def __init__(
        self,
        call_id: str,
        ok: bool,
        content: str,
        *,
        streaming: bool = False,
        initial_collapsed: bool | None = None,
        outcome: str | None = None,
    ) -> None:
        self.call_id = call_id
        self.ok = ok
        self.content = (content or "").rstrip()
        self._peek = self._build_peek()
        self._prefix = "🔧" if streaming else ("✅" if ok else "❌")
        if not streaming and outcome in ("cancelled", "refused"):
            self._prefix = {"cancelled": "⏹ STOPPED", "refused": "⊘ REFUSED"}[outcome]
        # Set when a later compaction round trims or drops THIS tool
        # result on the wire to the upstream model. The transcript
        # widget still has the full content (set in __init__), but
        # this flag tells the user "the model didn't see all of this
        # on its last round" so they can correctly interpret answers.
        self._compaction_state: str | None = None  # None | "truncated" | "dropped"
        # `_streaming` flips to False on `finalize_streaming(...)`. While
        # True, the title shows a 🔧 spinner-equivalent and incoming
        # `tool_delta` chunks append to the body Static. Used for
        # `delegate_claude` tool calls so the model's autonomous Claude
        # delegations stream visibly, same as the slash-command path.
        self._streaming = streaming
        # `_split` is True when a streaming widget was frozen mid-stream
        # by ``mark_split()`` because a nested ``tool_call`` arrived
        # (e.g., delegate_claude's inner Grep call). The next tool_delta
        # for the same call_id mounts a NEW streaming widget below the
        # inner tool widgets — without splitting, all of Claude's text
        # would land in the ORIGINAL widget mounted at the time of the
        # first chunk, visually placing the inner tool widgets AFTER
        # Claude's entire answer. See ``mark_split`` + the tool_call
        # handler's split loop.
        self._split = False
        body = Static(self.content or "(empty)", markup=False, classes="tool-result-body")
        self._body = body  # held for incremental updates during streaming
        # Auto-collapse only LONG outputs. Short tool results
        # (a 1-line ls, a small grep) are far more useful expanded
        # by default — operators were complaining the TUI looked
        # like "nothing happened" when only the collapsed-peek
        # header was visible. Threshold picked so a 5-line snippet
        # stays open but a 64KB grep dump still folds away.
        # `initial_collapsed` (replay/render) overrides the heuristic so a
        # passive frontend matches the driver's collapse state exactly;
        # the live path leaves it None and uses the shared predicate.
        if initial_collapsed is not None:
            start_collapsed = bool(initial_collapsed)
        else:
            start_collapsed = _default_tool_result_collapsed(self.content, was_streaming=streaming)
        super().__init__(
            body,
            title=self._build_title(collapsed=start_collapsed),
            collapsed=start_collapsed,
            classes="tool-result",
        )

    def _build_peek(self) -> str:
        first = self.content.splitlines()[0] if self.content else "(empty)"
        return _truncate_smart(first, 80)

    def _build_title(self, *, collapsed: bool) -> Content:
        hint = "Ctrl+T to expand" if collapsed else "Ctrl+T to collapse"
        badge = ""
        if self._compaction_state == "truncated":
            badge = "   🗜️ upstream-truncated"
        elif self._compaction_state == "dropped":
            badge = "   🗜️ upstream-dropped"
        # Textual's Collapsible parses a *str* title as content markup, so
        # tool results containing [foo] or other bracketed content
        # (web_search JSON, exec stderr with traceback frames, a Dash
        # `html.Div([card(...config={...})])` snippet, …) get parsed as
        # markup tags. rich.markup.escape is NOT enough: its regex only
        # neutralizes a *complete* `[...]` tag, but `_peek` is truncated at
        # 80 chars (`_build_peek`) and routinely cuts off mid-`[…`, leaving
        # an UNCLOSED bracket that escape leaves untouched → Textual's
        # `Content.from_markup` raises MarkupError and crashes the TUI on
        # replay. Building a `Content` object instead bypasses markup
        # parsing entirely (`Content.from_text` returns a Content verbatim),
        # so ANY peek text is rendered literally and safely.
        return Content(f"{self._prefix} {self._peek}   ·   {hint}{badge}")

    def mark_compacted(self, state: str) -> None:
        """Record that this tool result was trimmed/dropped on the wire
        in a subsequent round. Idempotent; "dropped" supersedes "truncated".
        """
        if state not in ("truncated", "dropped"):
            return
        # Once dropped, never demote back to truncated.
        if self._compaction_state == "dropped":
            return
        self._compaction_state = state
        self.title = self._build_title(collapsed=self.collapsed)

    def watch_collapsed(self, collapsed: bool) -> None:
        # Textual's reactive watcher: fires whenever `collapsed` changes
        # (whether via mouse click on the header arrow or via
        # `action_toggle_tool_output`). Refreshes the inline hint so the
        # user always sees what Ctrl+T does *next*, not what it did last.
        # Textual's framework dispatches base-class watchers separately,
        # so no super() chain is needed.
        self.title = self._build_title(collapsed=collapsed)

    def toggle(self) -> bool:
        """Flip the collapsed state and return the new value.

        Used by `action_toggle_tool_output`. The reactive watcher above
        keeps the title hint in sync.
        """
        self.collapsed = not self.collapsed
        return self.collapsed

    def append_streaming(self, text: str) -> None:
        """Append a `tool_delta` chunk while the result is still streaming.

        No-op once `finalize_streaming` has run. The body Static is
        replaced wholesale (Textual's Static renders fastest from a
        single `update()` rather than incremental concatenation).
        """
        if not self._streaming or not text:
            return
        self.content = (self.content + text).rstrip()
        self._body.update(self.content or "(empty)")
        self._peek = self._build_peek()
        self.title = self._build_title(collapsed=self.collapsed)

    def finalize_streaming(self, *, ok: bool, content: str | None = None, outcome: str | None = None) -> None:
        """End-of-stream: stamp the result with success/failure and an
        authoritative content body (overrides streamed deltas if the
        final value differs — the agent always sends one).
        """
        self._streaming = False
        self.ok = ok
        if content is not None:
            self.content = content.rstrip()
            self._body.update(self.content or "(empty)")
        self._prefix = "✅" if ok else "❌"
        if outcome in ("cancelled", "refused"):
            self._prefix = {"cancelled": "⏹ STOPPED", "refused": "⊘ REFUSED"}[outcome]
        # Re-evaluate the collapse state from the FINAL content (purely
        # content-based, streaming no longer relevant once finished). A
        # result that streamed live started collapsed to avoid a growing
        # blob; now that it's done, collapse iff it's actually long — the
        # SAME rule a replay/render frontend applies (which can't see the
        # live-only streaming flag), so every frontend agrees on the
        # chevron. (turn-replication: exact cross-frontend parity)
        self.collapsed = _default_tool_result_collapsed(self.content, was_streaming=False)
        self._peek = self._build_peek()
        self.title = self._build_title(collapsed=self.collapsed)

    def mark_split(self) -> None:
        """Freeze a streaming widget mid-stream so a subsequent
        ``tool_delta`` for the same call_id mounts a NEW widget below
        whatever else just got placed in the transcript.

        Why: ``delegate_claude`` (and any future streaming-capable tool)
        emits ``tool_call_update`` deltas continuously. The inner Claude
        session may fire ``tool_use`` events in the MIDDLE of the
        outer's stream. Without splitting, every delta lands in the
        original streaming widget (mounted at the moment the FIRST
        delta arrived) — so the inner tool widgets, mounted later, end
        up visually BELOW Claude's entire combined output, including
        text that was emitted AFTER the inner tools ran. That looks to
        the operator like "the tool calls happened AFTER the final
        answer."

        Splitting flips ``_streaming`` to False so the
        ``tool_delta`` handler's ``next(... if w._streaming ...)``
        search returns None and a fresh streaming widget mounts. The
        prefix flips to ``🔧→`` so the operator sees this segment is a
        continuation, not the end. The final ``tool_result`` handler
        notices the prior split and skips the content-replacement
        path (the streamed segments together ARE the authoritative
        content; the agent's final `content` field would duplicate
        them).
        """
        if not self._streaming:
            return
        self._streaming = False
        self._split = True
        # Continuation marker — distinct from ✅ (final ok) and ❌ (final
        # fail). The next segment carries the final ✅/❌.
        self._prefix = "🔧→"
        self._peek = self._build_peek()
        self.title = self._build_title(collapsed=self.collapsed)


class PlanMessage(Collapsible):
    """ExitPlanMode plan body, rendered as a prominent transcript card.

    The model emits ``ExitPlanMode {plan: "<markdown>"}`` in plan mode
    when it's done planning and wants the operator to leave plan mode
    and execute. The default tool widgets hide the plan in two ways:
    the ToolCallMessage pill truncates JSON-escaped args at 200 chars,
    and the ToolResultMessage Collapsible auto-folds outputs > 12
    lines / > 800 chars — every real plan is bigger than both caps,
    so the operator saw only "Plan ready — confirm to leave plan mode
    and execute:" and never the plan itself. This widget renders the
    plan as full markdown inline.

    Subclasses ``Collapsible`` (default expanded) so the operator sees
    the plan on first render — that was the whole point of the May 13
    rewrite — but can collapse it once read via ``Ctrl+T`` (or by
    clicking the header arrow when mouse mode is on). Long plans that
    push the scroll position can be folded out of the way without
    losing the body.
    """

    def __init__(self, plan: str) -> None:
        self._plan = (plan or "").strip()
        body = Static(
            RichMarkdown(self._plan or "(empty plan)"),
            classes="plan-msg-body",
        )
        super().__init__(
            body,
            title=self._build_title(collapsed=False),
            collapsed=False,
            classes="plan-msg",
        )

    def _build_title(self, *, collapsed: bool) -> str:
        hint = "Ctrl+T to expand" if collapsed else "Ctrl+T to collapse"
        return f"📐 plan ready — review and approve to exit plan mode   ·   {hint}"

    def watch_collapsed(self, collapsed: bool) -> None:
        # Reactive watcher fires on every collapsed change (mouse click,
        # Ctrl+T, programmatic) so the inline hint always reflects what
        # the binding will do *next*, not what it just did.
        self.title = self._build_title(collapsed=collapsed)

    def toggle(self) -> bool:
        """Flip collapsed state and return the new value. Used by
        ``action_toggle_tool_output``."""
        self.collapsed = not self.collapsed
        return self.collapsed


class AskUserQuestionMessage(Vertical):
    """Inline transcript card for ``AskUserQuestion`` tool calls.

    The model's ``AskUserQuestion`` tool is intercepted in
    ``acp_bridge`` (see ``acp_bridge.py:_acp_handle_tool_calls``) and
    forwarded to the TUI as a JSON-RPC ``session/ask_user`` request.
    ``AgentChatApp._on_ask_user`` pops :class:`QuestionsModal`, the
    operator picks, and the modal vanishes — leaving the transcript
    with no record of what was asked or answered. This widget plugs
    that gap: mount it on the transcript right before pushing the
    modal so the operator (and anyone scrolling back later) sees
    the question and its outcome inline alongside every other turn
    artifact — same pattern as :class:`PlanMessage` for ExitPlanMode
    and :class:`UpdateMessage` for Edit/Update.

    Lifecycle is three states driven by the modal's resolution:

    * **pending** — header says "waiting…", each question panel lists
      its options as a numbered checklist (read-only echo).
    * **answered** — :meth:`mark_answered` flips the header to "✅
      answered", and each question panel appends ``→ selected: …``
      plus the operator's free-form ``notes`` if non-empty.
    * **cancelled** — :meth:`mark_cancelled` flips the header to
      "✗ cancelled" and recolors the card to the error palette.
    """

    def __init__(self, questions: list[dict]) -> None:
        super().__init__(classes="askq-msg")
        self._questions: list[dict] = [dict(q) for q in (questions or []) if isinstance(q, dict)]
        # None while waiting, True once answered, False on cancel.
        # Drives header text + .askq-msg-cancelled class.
        self._state: bool | None = None
        self._header_static: Static | None = None
        # Per-question footer Static (the "→ selected: …" line). Built
        # in compose() and addressed in mark_answered so we don't have
        # to walk the tree by id — keeps the answered-state update O(N)
        # in questions rather than O(widgets).
        self._answer_statics: list[Static] = []

    def _build_header(self) -> str:
        n = len(self._questions)
        plural = "question" if n == 1 else "questions"
        if self._state is True:
            marker, suffix = "✅", "answered"
        elif self._state is False:
            marker, suffix = "✗", "cancelled"
        else:
            marker, suffix = "❓", "waiting…"
        return f"{marker} AskUserQuestion — {n} {plural} · {suffix}"

    def compose(self) -> ComposeResult:
        self._header_static = Static(
            self._build_header(),
            markup=False,
            classes="askq-msg-header",
        )
        yield self._header_static
        if not self._questions:
            yield Static(
                "(no questions)",
                markup=False,
                classes="askq-msg-empty",
            )
            return
        for qi, q in enumerate(self._questions):
            header = str(q.get("header") or "").strip() or f"Question {qi + 1}"
            question = str(q.get("question") or "").strip() or "(no question text)"
            multi = bool(q.get("multiSelect"))
            options = q.get("options") or []
            with Vertical(classes="askq-msg-panel"):
                chip = f"[{header}]" + ("  (multi-select)" if multi else "")
                yield Static(chip, markup=False, classes="askq-msg-chip")
                yield Static(question, markup=False, classes="askq-msg-question")
                lines: list[str] = []
                for oi, opt in enumerate(options):
                    if not isinstance(opt, dict):
                        continue
                    label = str(opt.get("label") or "").strip()
                    desc = str(opt.get("description") or "").strip()
                    if not label:
                        continue
                    if desc:
                        lines.append(f"  {oi + 1}. {label} — {desc}")
                    else:
                        lines.append(f"  {oi + 1}. {label}")
                yield Static(
                    "\n".join(lines) if lines else "  (no options)",
                    markup=False,
                    classes="askq-msg-options",
                )
                # Placeholder for the "→ selected: …" line. Populated
                # by mark_answered; kept empty + hidden in pending /
                # cancelled states so the card height stays stable.
                ans = Static("", markup=False, classes="askq-msg-answer")
                ans.display = False
                self._answer_statics.append(ans)
                yield ans

    def _refresh_header(self) -> None:
        if self._header_static is not None:
            self._header_static.update(self._build_header())

    def mark_answered(self, result: dict | None) -> None:
        """Flip the card to its answered state and fill in selections.

        ``result`` is the wire-shape dict returned by
        :class:`QuestionsModal` — ``{"answers": [{"selected": [...],
        "notes": "..."}], ...}``. Missing / malformed entries render
        as ``(no selection)`` so the operator can tell a real empty
        answer from a layout bug.
        """
        self._state = True
        self._refresh_header()
        answers = []
        if isinstance(result, dict):
            raw = result.get("answers")
            if isinstance(raw, list):
                answers = raw
        for qi, ans in enumerate(self._answer_statics):
            entry = answers[qi] if qi < len(answers) and isinstance(answers[qi], dict) else {}
            selected = entry.get("selected") or []
            if not isinstance(selected, list):
                selected = []
            sel_text = ", ".join(str(s) for s in selected if s) or "(no selection)"
            notes = str(entry.get("notes") or "").strip()
            body = f"→ selected: {sel_text}"
            if notes:
                body += f"\n  notes: {notes}"
            ans.update(body)
            ans.display = True

    def mark_cancelled(self) -> None:
        """Flip the card to its cancelled state. No selections are
        rendered — there are none. The error class recolors the border
        so the cancellation is visible at a glance when scrolling back."""
        self._state = False
        self._refresh_header()
        self.set_class(True, "askq-msg-cancelled")


class UserQuestionMessage(Vertical):
    """Inline transcript card for ``UserQuestion`` tool calls.

    ``UserQuestion`` (``sharedagents_agent.tools.registry``) is the local
    agent's "ask the operator a clarifying question" tool. The runner
    just formats a markdown echo of the question and the model is
    instructed to end its turn — the operator's next chat message is
    interpreted as the answer (async-by-text, no modal). With the
    default tool widgets the call lands as a ``🔧 UserQuestion
    {"question": ..., "options": [...]}`` pill plus a collapsed
    ``❓ User question dispatched`` peek; operators reported "I can't
    tell what the model is asking." This widget renders the question
    + numbered options as a prominent inline card so the operator
    sees the ask immediately and can type their reply without
    expanding anything.

    Single-state by design: the model has already ended its turn by
    the time the card lands, and the answer arrives as a normal user
    message (which gets its own ``UserMessage`` widget in the
    transcript). There's no "answered" / "cancelled" lifecycle to
    track here — different from :class:`AskUserQuestionMessage`,
    which complements a synchronous modal round-trip.

    Wire-in: the tool-call dispatch in :meth:`AgentChatApp._run_turn`
    mounts this widget when ``tool_name == "UserQuestion"`` and adds
    the call_id to ``_user_question_call_ids`` so the matching
    ``tool_result`` (whose body is just the markdown echo) is
    suppressed on success — the card IS the question, no need for a
    redundant Collapsible underneath.
    """

    def __init__(self, args: dict | None) -> None:
        super().__init__(classes="userq-msg")
        args = args if isinstance(args, dict) else {}
        self._question: str = str(args.get("question") or "").strip()
        raw_opts = args.get("options")
        self._options: list[dict] = (
            [o for o in (raw_opts or []) if isinstance(o, dict)] if isinstance(raw_opts, list) else []
        )

    def compose(self) -> ComposeResult:
        yield Static(
            "❓ User question — waiting for your reply",
            markup=False,
            classes="userq-msg-header",
        )
        yield Static(
            self._question or "(no question text)",
            markup=False,
            classes="userq-msg-question",
        )
        lines: list[str] = []
        for i, opt in enumerate(self._options, 1):
            label = str(opt.get("label") or "").strip()
            if not label:
                continue
            desc = str(opt.get("description") or "").strip()
            if desc:
                lines.append(f"  {i}. {label} — {desc}")
            else:
                lines.append(f"  {i}. {label}")
        yield Static(
            "\n".join(lines) if lines else "  (no options)",
            markup=False,
            classes="userq-msg-options",
        )
        yield Static(
            "→ reply with your choice (or type freely) — the agent ended its turn and is waiting",
            markup=False,
            classes="userq-msg-hint",
        )


class OutputSectionMessage(Vertical):
    """One section of an ``OutputSection`` long-form answer.

    The model fires ``OutputSection {section_name, content, …}`` once
    per section when drafting reports or multi-step guides — the
    section body in ``content`` is the actual user-facing answer.
    Rendered through the generic tool widgets the prose ends up as
    a 200-char JSON-escaped pill (``\\n`` literals, brackets, the
    real content cut off) plus a collapsed "Section X/N saved" peek;
    operators report "the output is missing." This widget renders
    the section body as full markdown inline so each section reads
    as its own answer chunk.
    """

    def __init__(self, section_name: str, content: str) -> None:
        super().__init__(classes="section-msg")
        self._section_name = (section_name or "").strip() or "(untitled section)"
        self._content = (content or "").strip()

    def compose(self) -> ComposeResult:
        yield Static(
            f"📄 {self._section_name}",
            markup=False,
            classes="section-msg-header",
        )
        body = self._content or "(empty section)"
        yield Static(RichMarkdown(body), classes="section-msg-body")


class ContextMessage(Vertical):
    """Output of the ``/context`` slash command rendered as an inline
    boxed card in the transcript.

    Lands in the conversation pane (not the footer/events row) so the
    multi-line breakdown stays readable. ``markup=False`` on the body
    Static prevents Rich from chewing on bracketed tokens (``[OK]``,
    ``[red]``) that appear in the file-status lines.

    The header text mirrors openclaw's ``🧠 Context breakdown`` framing
    so operators reading either runtime see the same affordance. Body
    text is monospace plain text — the report is hand-formatted columns
    with chars/tokens, not markdown.
    """

    def __init__(self, body: str, *, sub: str = "") -> None:
        super().__init__(classes="context-msg")
        self._body = (body or "").rstrip() or "(empty)"
        self._sub = (sub or "").strip()

    def compose(self) -> ComposeResult:
        header = "🧠 /context"
        if self._sub:
            header = f"{header} {self._sub}"
        yield Static(header, markup=False, classes="context-msg-header")
        yield Static(self._body, markup=False, classes="context-msg-body")


class CommandResultMessage(Vertical):
    """Generic slash-command result card (Web UI panel analogue).

    One renderer per ``kind`` — never per command name.
    ``markup=False`` so bracketed tokens in reports stay literal.
    """

    def __init__(
        self,
        title: str,
        body: str,
        *,
        kind: str = "text",
        error: bool = False,
    ) -> None:
        classes = "cmd-card"
        if error or kind == "error":
            classes += " cmd-card-error"
        super().__init__(classes=classes)
        self._title = (title or kind or "result").strip()
        self._body = (body or "").rstrip() or ("error" if error else "(no output)")
        self._kind = kind or "text"

    def compose(self) -> ComposeResult:
        yield Static(self._title, markup=False, classes="cmd-card-header")
        yield Static(self._body, markup=False, classes="cmd-card-body")


class UpdateMessage(Vertical):
    """A file edit (`Update` / `Edit` tool) rendered as a colored code-diff card.

    Named after Claude Code's "Update" — the label Claude Code shows
    when the agent edits a file. The diff is computed *client-side*
    from the tool call's arguments (``old_string`` → ``new_string``),
    so the card appears the instant the call starts, before the tool
    even finishes — and the model's context stays lean because the
    `Update` runner only returns a one-line summary, not the diff.

    This widget is **TUI-only**. The Telegram frontend renders file
    edits as a plain ``🔧 Update…`` header with no diff body: it never
    echoes tool arguments, and the runner's result carries no diff.

    A green border-left marks a normal edit; ``mark_result(ok=False)``
    flips the card to the error palette if the tool reports failure.
    """

    # Cap the rendered diff so a giant edit doesn't flood the
    # transcript. The edit itself still applies in full — this only
    # trims the on-screen card.
    MAX_DIFF_LINES = 200

    def __init__(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
        call_id: str = "",
    ) -> None:
        super().__init__(classes="update-msg")
        self._file_path = (file_path or "").strip() or "(unknown file)"
        self._old = old_string or ""
        self._new = new_string or ""
        self._replace_all = bool(replace_all)
        # Set so the matching `tool_result` can find THIS card among
        # several un-stamped parallel-edit cards (see `mark_result`).
        self.call_id = call_id or ""
        self._header_static: Static | None = None
        # None while the tool is still running; True/False once the
        # matching tool_result lands (see `mark_result`).
        self._ok: bool | None = None

    def _build_header(self) -> str:
        # Mirror Claude Code's "Update(path)" headline. The marker
        # reflects tool state: ✎ in-flight, ✓ applied, ✗ failed.
        if self._ok is True:
            marker = "✓"
        elif self._ok is False:
            marker = "✗"
        else:
            marker = "✎"
        suffix = "  (replace all)" if self._replace_all else ""
        return f"{marker} Update({self._file_path}){suffix}"

    def _diff_text(self) -> Text:
        """Build a Rich ``Text`` of the unified diff with +/- coloring.

        Uses a ``Text`` object (not a markup string) so source lines
        containing ``[...]`` can't corrupt the render.
        """
        old_lines = self._old.splitlines()
        new_lines = self._new.splitlines()
        diff = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=3))
        # `unified_diff` prepends `--- ` / `+++ ` file-header lines from
        # the empty from/to names — drop them; our card header already
        # names the file.
        diff = [ln for ln in diff if not ln.startswith("--- ") and not ln.startswith("+++ ")]
        text = Text(no_wrap=False)
        if not diff:
            text.append("(no textual change)", style="dim")
            return text
        shown = diff[: self.MAX_DIFF_LINES]
        for i, line in enumerate(shown):
            if i:
                text.append("\n")
            if line.startswith("@@"):
                text.append(line, style="cyan")
            elif line.startswith("+"):
                text.append(line, style="green")
            elif line.startswith("-"):
                text.append(line, style="red")
            else:
                text.append(line, style="dim")
        hidden = len(diff) - len(shown)
        if hidden > 0:
            text.append(f"\n… {hidden} more diff line(s) trimmed", style="dim")
        return text

    def compose(self) -> ComposeResult:
        header = Static(self._build_header(), markup=False, classes="update-msg-header")
        self._header_static = header
        yield header
        yield Static(self._diff_text(), classes="update-msg-body")

    def mark_result(self, ok: bool) -> None:
        """Stamp the card with the tool's success/failure once the
        matching ``tool_result`` arrives. Idempotent — safe to call
        before the widget has composed (the header refresh is skipped
        until ``compose`` runs)."""
        self._ok = bool(ok)
        if self._header_static is not None:
            self._header_static.update(self._build_header())
        self.set_class(not self._ok, "update-msg-failed")


class TaskUpdateMessage(Vertical):
    """A TodoWrite task list rendered as an inline transcript card.

    Mounted when the model emits a ``TodoWrite`` tool call, showing
    the full task list with per-item status (☑ completed, ▶ in-progress,
    ☐ pending). Updates in place on subsequent TodoWrite calls so the
    transcript stays clean — no duplicate cards for status nudges.
    Follows the same visual convention as the persistent
    ``#plan-checklist`` footer widget but as a dedicated card in the
    turn transcript, mirroring Claude Code's inline task display.

    Layout (top → bottom): collapsible header with progress summary →
    scrollable item list. Default collapsed so a 20-item TodoWrite
    can't eat the screen; click the header (or focus + space/enter) to
    toggle. Body is wrapped in a :class:`textual.containers.VerticalScroll`
    whose ``max_height`` is capped to :attr:`BODY_HEIGHT_FRACTION` of
    the app's terminal height — recomputed on resize so a smaller
    window keeps the body proportionally bounded.
    """

    MAX_ITEMS = 12
    # Cap the body to 40% of the terminal height per MISSION.md D2.
    # A 60-row terminal → at most 24 rows of body. Recomputed in
    # ``on_mount`` and ``on_resize``.
    BODY_HEIGHT_FRACTION = 0.4

    BINDINGS = [
        Binding("space", "toggle_collapsed", "Toggle", show=False),
        Binding("enter", "toggle_collapsed", "Toggle", show=False),
        # MISSION.md D2 scroll bindings — forwarded to the body
        # VerticalScroll when the card is focused. Standard textual
        # scroll vocab so muscle memory carries over from other
        # scrollable widgets (kanban panel, monitor panel, etc).
        Binding("up", "body_scroll('line_up')", "Up", show=False),
        Binding("down", "body_scroll('line_down')", "Down", show=False),
        Binding("k", "body_scroll('line_up')", "Up", show=False),
        Binding("j", "body_scroll('line_down')", "Down", show=False),
        Binding("pageup", "body_scroll('page_up')", "PgUp", show=False),
        Binding("pagedown", "body_scroll('page_down')", "PgDn", show=False),
        Binding("home", "body_scroll('home')", "Home", show=False),
        Binding("end", "body_scroll('end')", "End", show=False),
    ]
    can_focus = True

    def __init__(
        self,
        items: list[dict],
        *,
        call_id: str = "",
        start_collapsed: bool = True,
    ) -> None:
        super().__init__(classes="task-update-msg")
        self._items: list[dict] = list(items)
        self._call_id = call_id or ""
        # MISSION.md D1: collapsed by default; the operator unfurls
        # explicitly via space / enter / header click. The flag
        # survives every ``_redraw`` so a status nudge from another
        # TodoWrite doesn't visually re-expand a card the operator
        # just collapsed.
        self._collapsed: bool = bool(start_collapsed)
        self._header_static: Static | None = None
        self._body_static: Static | None = None
        self._body_scroll: VerticalScroll | None = None

    def _build_progress(self) -> tuple[int, int]:
        done = sum(1 for it in self._items if it.get("status") in ("completed", "done"))
        return done, len(self._items)

    def _build_header(self) -> str:
        done, total = self._build_progress()
        # Match MISSION.md D1 example shape:
        # collapsed → "▶ 📝 Tasks · 9/17 done · click to expand"
        # expanded  → "▼ 📝 Tasks · 9/17 done"
        glyph = "▶" if self._collapsed else "▼"
        hint = " · [dim]click to expand[/dim]" if self._collapsed else ""
        if total == 0:
            return f"[bold magenta]{glyph} 📝 Tasks[/bold magenta]{hint}"
        return f"[bold magenta]{glyph} 📝 Tasks · {done}/{total} done[/bold magenta]{hint}"

    def _build_body(self) -> str:
        lines: list[str] = []
        for it in self._items:
            status = it.get("status") or "pending"
            content = (it.get("content") or "").strip()
            active = (it.get("activeForm") or "").strip()
            label = active if (status == "in_progress" and active) else content
            if len(label) > 90:
                label = label[:87] + "…"
            if status in ("completed", "done"):
                lines.append(f"  [dim strike]☑ {label}[/dim strike]")
            elif status == "in_progress":
                lines.append(f"  [bold cyan]▶ {label}[/bold cyan]")
            else:
                lines.append(f"  [dim]☐[/dim] {label}")
        return "\n".join(lines)

    def compose(self) -> ComposeResult:
        header = Static(self._build_header(), markup=True, classes="task-update-header")
        self._header_static = header
        yield header
        body = Static(self._build_body(), markup=True, classes="task-update-body")
        self._body_static = body
        # MISSION.md D2: wrap the body in a VerticalScroll so a long
        # task list scrolls inside the card instead of pushing the
        # streaming response off-screen. ``max_height`` is set in
        # ``_apply_body_height_cap`` (mount + resize) so it tracks
        # the live terminal size.
        scroll = VerticalScroll(body, classes="task-update-body-scroll")
        # Body hidden until the operator unfurls — collapsed by default
        # per D1. ``display`` is the cheapest way to hide a Textual
        # container; it stops taking layout space when False.
        scroll.display = not self._collapsed
        self._body_scroll = scroll
        yield scroll

    def on_mount(self) -> None:
        self._apply_body_height_cap()

    def on_resize(self, event=None) -> None:
        # Terminal resize → re-cap the body so a smaller window keeps
        # the panel bounded proportionally.
        self._apply_body_height_cap()

    def _apply_body_height_cap(self) -> None:
        if self._body_scroll is None:
            return
        try:
            screen_h = int(self.app.size.height)
        except Exception:  # noqa: BLE001
            return
        if screen_h <= 0:
            return
        # MISSION.md D2: "never occupy more than 40% of the terminal's
        # visible height ... Round down." A floor (``max(3, …)``)
        # would breach the cap on a tiny terminal (5-row term → 5*0.4
        # = 2, but max(3, 2) = 3). Drop the floor and trust the
        # floor of 0 cells == hidden, which is consistent with the
        # collapse-by-default contract anyway.
        cap = int(screen_h * self.BODY_HEIGHT_FRACTION)
        try:
            self._body_scroll.styles.max_height = cap
        except Exception:  # noqa: BLE001
            pass

    def action_toggle_collapsed(self) -> None:
        """Hotkey handler — space / enter when the card is focused."""
        self.set_collapsed(not self._collapsed)

    def action_body_scroll(self, direction: str) -> None:
        """Forward standard scroll keys to the body VerticalScroll
        when the card is focused. Mission D2 lists ``↑/↓`` + ``j/k``
        + ``PgUp/PgDn`` + ``Home/End``; all route through here."""
        if self._body_scroll is None or self._collapsed:
            return
        try:
            fn = {
                "line_up": self._body_scroll.scroll_up,
                "line_down": self._body_scroll.scroll_down,
                "page_up": self._body_scroll.scroll_page_up,
                "page_down": self._body_scroll.scroll_page_down,
                "home": self._body_scroll.scroll_home,
                "end": self._body_scroll.scroll_end,
            }[direction]
        except KeyError:
            return
        try:
            fn(animate=False)
        except TypeError:
            # Older textual versions don't take animate kwarg.
            fn()
        except Exception:  # noqa: BLE001
            pass

    def on_click(self, event) -> None:
        """Toggle on header click only. Body / scroll clicks pass
        through so the operator can interact with the scrollbar
        without the card collapsing under them."""
        target = getattr(event, "widget", None)
        if target is self._header_static and self._header_static is not None:
            self.set_collapsed(not self._collapsed)

    def set_collapsed(self, value: bool) -> None:
        """Programmatic toggle — also called from
        :meth:`action_toggle_collapsed` and :meth:`on_click`. Keeps
        the header text and body display in lockstep so a stale
        upsert can't desync them."""
        self._collapsed = bool(value)
        if self._body_scroll is not None:
            try:
                self._body_scroll.display = not self._collapsed
            except Exception:  # noqa: BLE001
                pass
        if self._header_static is not None:
            try:
                self._header_static.update(self._build_header())
            except Exception:  # noqa: BLE001
                pass

    def update_from_todos(self, items: list[dict]) -> None:
        """Update the card with a new TodoWrite payload.

        Merges statuses forward (pending → in_progress → completed)
        so a stale re-emit can't visually regress a row. Content
        matching is by ``content`` key. Best-effort when the widget
        hasn't composed yet (header/body Static references are None).
        """
        if not items:
            return
        rank = {"pending": 0, "in_progress": 1, "completed": 2, "done": 2}
        old_by_content = {it.get("content"): it for it in self._items}
        merged: list[dict] = []
        for n in items:
            content = (n.get("content") or "").strip()
            if not content:
                continue
            old = old_by_content.get(content, {})
            old_rank = rank.get(old.get("status") or "pending", 0)
            new_rank = rank.get(n.get("status") or "pending", 0)
            merged.append(
                {
                    "content": content,
                    "activeForm": (n.get("activeForm") or old.get("activeForm") or ""),
                    "status": (n["status"] if new_rank >= old_rank else old["status"]),
                }
            )
        self._items = merged[: self.MAX_ITEMS]
        if self._header_static is not None:
            self._header_static.update(self._build_header())
        if self._body_static is not None:
            self._body_static.update(self._build_body())


class HarnessTaskListMessage(Vertical):
    """Inline card for the harness ``Task*`` tool family (``TaskCreate``
    / ``TaskUpdate`` / ``TaskList`` / ``TaskGet`` / ``TaskOutput`` /
    ``TaskStop``). Mirrors :class:`TaskUpdateMessage` (the TodoWrite
    card) but keyed by numeric task id — Claude Code's harness Task
    tracker is a separate, persistent list from TodoWrite.

    Mounted lazily on the first ``Task*`` call; subsequent calls update
    the same card in place so the transcript stays clean — no duplicate
    pills for every TaskCreate / TaskUpdate the model fires.

    Same collapse + 40%-height-cap + scroll machinery as
    :class:`TaskUpdateMessage`. MISSION.md D1 + D2: default collapsed,
    click header (or focus + space/enter) to toggle, body wrapped in
    a :class:`VerticalScroll` capped at
    :attr:`BODY_HEIGHT_FRACTION` of the terminal height.
    """

    MAX_ITEMS = 30
    BODY_HEIGHT_FRACTION = 0.4

    BINDINGS = [
        Binding("space", "toggle_collapsed", "Toggle", show=False),
        Binding("enter", "toggle_collapsed", "Toggle", show=False),
        Binding("up", "body_scroll('line_up')", "Up", show=False),
        Binding("down", "body_scroll('line_down')", "Down", show=False),
        Binding("k", "body_scroll('line_up')", "Up", show=False),
        Binding("j", "body_scroll('line_down')", "Down", show=False),
        Binding("pageup", "body_scroll('page_up')", "PgUp", show=False),
        Binding("pagedown", "body_scroll('page_down')", "PgDn", show=False),
        Binding("home", "body_scroll('home')", "Home", show=False),
        Binding("end", "body_scroll('end')", "End", show=False),
    ]
    can_focus = True

    def __init__(
        self,
        *,
        start_collapsed: bool = True,
        id: str | None = None,
        hide_when_empty: bool = False,
    ) -> None:
        # ``id`` + ``hide_when_empty`` exist so the same class can do
        # double duty: as an inline transcript card (default — mounted
        # lazily on the first Task* call, always visible) AND as the
        # persistent ``#harness-tasks`` footer (id="harness-tasks",
        # hide_when_empty=True — yielded once from ``ChatApp.compose``
        # and toggled visible/hidden by the task state). Mission body:
        # the persistent footer used to be a flat
        # ``Static`` that grew unbounded; reusing this widget gives it
        # the same collapse-by-default + 40%-cap + scroll machinery as the
        # inline card. No duplication.
        super().__init__(classes="harness-task-msg", id=id)
        # tid -> {"subject": str, "status": str}
        self._rows: dict[str, dict] = {}
        # Insertion order so the card reads top-to-bottom in creation order.
        self._order: list[str] = []
        self._collapsed: bool = bool(start_collapsed)
        self._header_static: Static | None = None
        self._body_static: Static | None = None
        self._body_scroll: VerticalScroll | None = None
        # Footer mode: hide the entire widget while the task list is
        # empty so an idle session has no chrome. Toggled by
        # ``_redraw`` whenever ``_rows`` flips between empty / non-empty.
        # Default False preserves the inline-card invariant that the
        # card stays mounted (and shows ``(no tasks)``) even when its
        # row set is momentarily empty.
        self._hide_when_empty: bool = bool(hide_when_empty)
        if self._hide_when_empty:
            # No rows at construction time → hide immediately. The
            # widget still composes (header / body / scroll get
            # populated) so subsequent ``replace_rows`` can flip
            # ``display=True`` without re-mounting.
            try:
                self.display = False
            except Exception:  # noqa: BLE001
                pass

    def _build_progress(self) -> tuple[int, int]:
        done = sum(1 for r in self._rows.values() if r.get("status") in ("completed", "done"))
        return done, len(self._rows)

    def _build_header(self) -> str:
        done, total = self._build_progress()
        glyph = "▶" if self._collapsed else "▼"
        hint = " · [dim]click to expand[/dim]" if self._collapsed else ""
        if total == 0:
            return f"[bold blue]{glyph} 📋 Tasks[/bold blue]{hint}"
        return f"[bold blue]{glyph} 📋 Tasks · {done}/{total} done[/bold blue]{hint}"

    def _icon(self, status: str) -> str:
        return {
            "pending": "[dim]☐[/dim]",
            "in_progress": "[bold cyan]▶[/bold cyan]",
            "completed": "[dim strike]☑[/dim strike]",
            "done": "[dim strike]☑[/dim strike]",
            "deleted": "[red]✗[/red]",
        }.get(status, "[dim]?[/dim]")

    def _build_body(self) -> str:
        if not self._rows:
            return "  [dim](no tasks)[/dim]"
        lines: list[str] = []
        for tid in self._order[: self.MAX_ITEMS]:
            r = self._rows.get(tid)
            if not r:
                continue
            label = (r.get("subject") or "").strip()
            if len(label) > 90:
                label = label[:87] + "…"
            status = r.get("status") or "pending"
            icon = self._icon(status)
            if status in ("completed", "done", "deleted"):
                lines.append(f"  {icon} [dim strike]#{tid}[/dim strike] {label}")
            else:
                lines.append(f"  {icon} [bold]#{tid}[/bold] {label}")
        if len(self._order) > self.MAX_ITEMS:
            extra = len(self._order) - self.MAX_ITEMS
            lines.append(f"  [dim]… +{extra} more[/dim]")
        return "\n".join(lines)

    def compose(self) -> ComposeResult:
        header = Static(self._build_header(), markup=True, classes="harness-task-header")
        self._header_static = header
        yield header
        body = Static(self._build_body(), markup=True, classes="harness-task-body")
        self._body_static = body
        scroll = VerticalScroll(body, classes="harness-task-body-scroll")
        scroll.display = not self._collapsed
        self._body_scroll = scroll
        yield scroll

    def on_mount(self) -> None:
        self._apply_body_height_cap()

    def on_resize(self, event=None) -> None:
        self._apply_body_height_cap()

    def _apply_body_height_cap(self) -> None:
        if self._body_scroll is None:
            return
        try:
            screen_h = int(self.app.size.height)
        except Exception:  # noqa: BLE001
            return
        if screen_h <= 0:
            return
        # MISSION.md D2: "never occupy more than 40% of the terminal's
        # visible height ... Round down." A floor (``max(3, …)``)
        # would breach the cap on a tiny terminal (5-row term → 5*0.4
        # = 2, but max(3, 2) = 3). Drop the floor and trust the
        # floor of 0 cells == hidden, which is consistent with the
        # collapse-by-default contract anyway.
        cap = int(screen_h * self.BODY_HEIGHT_FRACTION)
        try:
            self._body_scroll.styles.max_height = cap
        except Exception:  # noqa: BLE001
            pass

    def action_toggle_collapsed(self) -> None:
        self.set_collapsed(not self._collapsed)

    def action_body_scroll(self, direction: str) -> None:
        if self._body_scroll is None or self._collapsed:
            return
        try:
            fn = {
                "line_up": self._body_scroll.scroll_up,
                "line_down": self._body_scroll.scroll_down,
                "page_up": self._body_scroll.scroll_page_up,
                "page_down": self._body_scroll.scroll_page_down,
                "home": self._body_scroll.scroll_home,
                "end": self._body_scroll.scroll_end,
            }[direction]
        except KeyError:
            return
        try:
            fn(animate=False)
        except TypeError:
            fn()
        except Exception:  # noqa: BLE001
            pass

    def on_click(self, event) -> None:
        target = getattr(event, "widget", None)
        if target is self._header_static and self._header_static is not None:
            self.set_collapsed(not self._collapsed)

    def set_collapsed(self, value: bool) -> None:
        self._collapsed = bool(value)
        if self._body_scroll is not None:
            try:
                self._body_scroll.display = not self._collapsed
            except Exception:  # noqa: BLE001
                pass
        if self._header_static is not None:
            try:
                self._header_static.update(self._build_header())
            except Exception:  # noqa: BLE001
                pass

    def _redraw(self) -> None:
        if self._header_static is not None:
            self._header_static.update(self._build_header())
        if self._body_static is not None:
            self._body_static.update(self._build_body())
        # Footer mode: collapse the row to zero height while the task
        # list is empty (mirrors what the old ``Static.update("")``
        # path did via ``height: auto``). Inline-card mode leaves
        # ``display`` alone so the card stays anchored in the
        # transcript even on a transient empty state.
        if self._hide_when_empty:
            try:
                self.display = bool(self._rows)
            except Exception:  # noqa: BLE001
                pass

    def upsert(
        self,
        tid: str,
        *,
        subject: str | None = None,
        status: str | None = None,
    ) -> None:
        """Insert or merge a task row. Status only advances forward
        (pending → in_progress → completed) so a stale re-emit can't
        visually regress a row. ``deleted`` overrides everything.
        """
        tid = str(tid or "").strip()
        if not tid:
            return
        row = self._rows.get(tid)
        if row is None:
            row = {"subject": "", "status": "pending"}
            self._rows[tid] = row
            self._order.append(tid)
        if subject is not None:
            sb = str(subject).strip()
            if sb:
                row["subject"] = sb
        if status is not None:
            ns = str(status).strip().lower()
            if ns == "deleted":
                row["status"] = "deleted"
            else:
                rank = {
                    "pending": 0,
                    "in_progress": 1,
                    "completed": 2,
                    "done": 2,
                }
                old_rank = rank.get(row.get("status") or "pending", 0)
                new_rank = rank.get(ns, 0)
                if new_rank >= old_rank:
                    row["status"] = ns
        self._redraw()

    def apply_tasklist_payload(self, payload: list[dict]) -> None:
        """Apply a TaskList result (full snapshot of the task store).
        Best-effort: each entry needs at least ``id``; missing fields
        leave the existing row untouched.
        """
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            tid = str(entry.get("id") or entry.get("taskId") or "")
            if not tid:
                continue
            self.upsert(
                tid,
                subject=entry.get("subject"),
                status=entry.get("status"),
            )

    def replace_rows(
        self,
        rows: dict[str, dict],
        order: list[str],
    ) -> None:
        """Wipe + re-seed the row set from an external (app-level)
        source, preserving the operator's collapse choice.

        Why: the persistent ``#harness-tasks`` footer is driven from
        the app's ``_harness_tasks`` dict + ``_harness_tasks_order``
        list, which are updated piecewise (TaskCreate / TaskUpdate /
        ``/task del``) and may shrink as well as grow. ``upsert``
        alone can't express deletion, and we don't want the footer
        and the app-level state to drift. Calling ``replace_rows``
        from ``_render_harness_tasks`` makes the widget a pure
        projection of the app state — every repaint rebuilds from the
        source of truth.

        ``_collapsed`` is *not* touched here: the operator's choice
        survives a TaskCreate / TaskUpdate flurry (mission D1
        "collapse state persists across panel re-renders"). Footer
        visibility is updated by ``_redraw`` via the
        ``_hide_when_empty`` toggle.
        """
        new_rows: dict[str, dict] = {}
        for tid, row in (rows or {}).items():
            if not isinstance(row, dict):
                continue
            sid = str(tid or "").strip()
            if not sid:
                continue
            new_rows[sid] = {
                "subject": str(row.get("subject") or "").strip(),
                "status": str(row.get("status") or "pending").strip().lower(),
            }
        new_order: list[str] = []
        seen: set[str] = set()
        for tid in order or []:
            sid = str(tid or "").strip()
            if sid and sid in new_rows and sid not in seen:
                new_order.append(sid)
                seen.add(sid)
        # Tolerate orphan rows that the caller forgot to include in
        # ``order`` — append them in their dict-iteration order so we
        # don't silently drop data.
        for sid in new_rows:
            if sid not in seen:
                new_order.append(sid)
        self._rows = new_rows
        self._order = new_order
        self._redraw()
