"""Requirements
------------
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from rich.markdown import Markdown as RichMarkdown
from textual.app import App
from textual.binding import Binding

from sharedagents_core import config as cfgmod
from sharedagents_tui.app_mixins import (  # noqa: F401
    HEALTH_POLL_INTERVAL_S,
    CommandsMixin,
    InputMixin,
    LifecycleMixin,
    ModalMixin,
    SessionMixin,
    StatusMixin,
    StreamMixin,
    SubagentMixin,
    TasksMixin,
)

# --- de-monolith: AgentChatApp/ChatApp method bodies live in mixins ---
from sharedagents_tui.app_mixins.chat_base import ChatBaseMixin  # noqa: F401

# --- de-monolith re-exports: keep every extracted symbol importable
#     from ``sharedagents_tui.app`` (tests/conftest depend on this). ---
from sharedagents_tui.helpers import (  # noqa: F401  re-export
    _DELEGATE_MARKER_PREFIX,
    _INLINE_TOOL_MARKER_RE,
    _IS_SSH,
    _KITTY_KEYBOARD_MODE,
    _PLAIN_KEYS,
    _RUNTIME_WARNING_GLYPHS,
    _RUNTIME_WARNING_HEADLINES,
    _RUNTIME_WARNING_PREFIX,
    _SPINNER_FRAMES,
    _SUBPROCESS_STDIO_LIMIT,
    _TASK_SUMMARY_LINE_RE,
    _THINK_TICK_INTERVAL_S,
    _TUI_FPS,
    CLAUDE_MODE_MAP,
    CODEX_MODE_LABELS,
    _alias_for_endpoint,
    _build_delegate_metrics,
    _build_metrics_markup,
    _claude_permission_mode,
    _codex_sandbox_label,
    _default_tool_result_collapsed,
    _detect_ssh,
    _enable_kitty_keyboard_enhancements,
    _engine_models_for,
    _gateway_alive,
    _gateway_url_and_token,
    _hostname_of,
    _humanize_age,
    _model_alias_id_label,
    _now,
    _parse_runtime_warning_chunk,
    _parse_task_list_text,
    _reconcile_subagent_sidecars,
    _rows_signature,
    _strip_inline_tool_markers,
    _truncate_smart,
    discover_model,
    humanise_delegate_marker,
)
from sharedagents_tui.widgets.input import (  # noqa: F401  re-export
    SLASH_COMMAND_HELP,
    CommandHistory,
    MessageInput,
    _longest_common_prefix,
    file_command_rows,
    merge_slash_command_rows,
)
from sharedagents_tui.widgets.messages import (  # noqa: F401  re-export
    AntigravityMessage,
    AskUserQuestionMessage,
    AssistantMessage,
    ClaudeMessage,
    CodexMessage,
    CommandResultMessage,
    ContextMessage,
    HarnessTaskListMessage,
    OutputSectionMessage,
    PlanMessage,
    RoundMetricsMessage,
    TaskUpdateMessage,
    ToolCallMessage,
    ToolResultMessage,
    UpdateMessage,
    UserMessage,
    UserQuestionMessage,
)
from sharedagents_tui.widgets.modals import (  # noqa: F401  re-export
    AcpPermissionModal,
    AgentMonitorModal,
    ConfirmModal,
    ModelPickerModal,
    MonitorPanelModal,
    PermissionModal,
    QuestionsModal,
    SessionPickerModal,
)
from sharedagents_tui.widgets.notices import (  # noqa: F401  re-export
    CompactionNotice,
    EmptyEosNotice,
    RuntimeWarningNotice,
)
from sharedagents_tui.widgets.panels import (  # noqa: F401  re-export
    PlanChecklistFooter,
    SubagentBoxMessage,
    SubagentListPanel,
)
from sharedagents_tui.widgets.scroll import (  # noqa: F401  re-export
    TranscriptScroll,
)

if TYPE_CHECKING:
    from sharedagents_agent.agent import AgentConfig


class ChatApp(ChatBaseMixin, App):
    CSS = """
    Screen { layout: vertical; }
    #transcript {
        background: $surface;
        padding: 0 2;
        height: 1fr;
    }
    .user-msg {
        background: $primary 20%;
        color: $text;
        padding: 0 1;
        margin: 1 0 0 0;
        border-left: thick $primary;
    }
    .assist-msg {
        margin: 1 0 0 0;
        height: auto;
    }
    .assist-content {
        padding: 0 1;
        height: auto;
    }
    /* Prefill (time-to-first-token) breadcrumb that ticks live above the
       thinking box until the first delta lands. Single-line, no border,
       no background — it's a quiet status line, not a panel. */
    .assist-prefill {
        height: 1;
        padding: 0 1;
        margin: 0;
    }
    /* Per-model-call stats line stamped under each AssistantMessage's
       content (model · ttft · gen tps · ctx · wt · …). Dim, single-line,
       no border so it reads as a quiet footnote — one row per round so
       multi-tool turns surface stats for every model call. */
    .assist-metrics {
        height: auto;
        padding: 0 1;
        margin: 0;
    }
    /* One-line dim badge mounted above the content of a delegated
       turn (Claude / Codex). Styled to read as a footnote — same
       low-contrast vibe as the metrics row — so the transcript still
       feels like a sequence of native sharedagents turns. */
    .delegate-badge {
        height: 1;
        padding: 0 1;
        margin: 0;
        color: $text-muted;
    }
    .thinking {
        background: $boost;
        margin: 0;
    }
    .thinking-body {
        color: $text-muted;
        padding: 0 1;
    }
    .tool-call {
        background: $boost;
        color: $accent;
        padding: 0 1;
        margin: 1 0 0 0;
        border-left: thick $accent;
    }
    .claude-msg {
        background: $surface;
        color: $text;
        padding: 1 1;
        margin: 1 0 0 0;
        border-left: thick $secondary;
    }
    /* `.tool-result` is a Collapsible (subclass), styled like the
       reasoning Collapsible so the two read as siblings — same visual
       treatment, same expand/collapse affordance, just a different hint
       (Ctrl+T vs Ctrl+R). Minimal here so Textual's Collapsible chrome
       (▶/▼ arrow, header bar) renders naturally. */
    .tool-result {
        background: $boost;
        margin: 0;
    }
    .tool-result-body {
        color: $text-muted;
        padding: 0 1;
    }
    /* `.plan-msg` shows the ExitPlanMode body inline so the operator
       can read the plan and decide whether to leave plan mode. It's a
       Collapsible (subclass), default expanded — the operator sees the
       plan on first render — but Ctrl+T (or the header arrow) folds it
       away once read. Green border-left distinguishes it from the
       surrounding tool widgets at a glance; without it the Collapsible
       chrome looks identical to `.tool-result` and operators lose the
       "this is a plan, not a tool result" cue. The neighbouring
       always-visible cards (`.context-msg`, `.section-msg`,
       `.update-msg`, `.harness-task-msg`) keep their accent borders
       too — collapsibility is the only thing different about this one. */
    .plan-msg {
        background: $boost;
        margin: 1 0 0 0;
        border-left: thick $success;
    }
    .plan-msg-body {
        color: $text;
        padding: 0 1 1 1;
    }
    /* `.askq-msg` is the inline transcript card for AskUserQuestion
       calls — mounted right before QuestionsModal pops so the question
       and its options stay visible after the modal closes, and the
       chosen answer is appended in place once the operator submits.
       Warning-colored border so it reads as "agent is waiting on you";
       `.askq-msg-cancelled` flips to the error palette to mark a user
       cancel. Never collapsed — short by construction (≤4 questions ×
       ≤4 options), and the operator wants to re-read what they chose. */
    .askq-msg {
        background: $boost;
        margin: 1 0 0 0;
        border-left: thick $warning;
        height: auto;
    }
    .askq-msg-cancelled {
        border-left: thick $error;
    }
    .askq-msg-header {
        color: $warning;
        text-style: bold;
        padding: 0 1;
    }
    .askq-msg-cancelled .askq-msg-header {
        color: $error;
    }
    .askq-msg-empty {
        color: $text-muted;
        padding: 0 1 1 1;
    }
    .askq-msg-panel {
        height: auto;
        padding: 0 1;
        margin-top: 1;
    }
    .askq-msg-chip {
        color: $warning;
        text-style: bold;
    }
    .askq-msg-question {
        color: $text;
    }
    .askq-msg-options {
        color: $text-muted;
    }
    .askq-msg-answer {
        color: $success;
        text-style: bold;
        margin-top: 1;
    }
    .askq-msg-cancelled .askq-msg-answer {
        color: $error;
    }
    /* `.userq-msg` renders a `UserQuestion` tool call as a prominent
       inline card — the model has issued a clarifying question, ended
       its turn, and is waiting for the operator's next chat message
       as the answer. Warning-colored border so it reads as "operator
       action required", contrasting with the cyan section card
       (model output) and green update card (file change applied).
       Never collapsed — the question IS the point, and the operator
       can't reply if they can't see it. */
    .userq-msg {
        background: $boost;
        margin: 1 0 0 0;
        border-left: thick $warning;
        height: auto;
        padding: 0 1 1 1;
    }
    .userq-msg-header {
        color: $warning;
        text-style: bold;
        padding: 0 0 1 0;
    }
    .userq-msg-question {
        color: $text;
        padding: 0 0 1 0;
    }
    .userq-msg-options {
        color: $text-muted;
    }
    .userq-msg-hint {
        color: $text-muted;
        text-style: italic;
        padding: 1 0 0 0;
    }
    /* `.section-msg` renders one OutputSection call's content inline as
       the model's actual answer body. Cyan/secondary border so it reads
       as model output (not a tool result, not a plan); never collapsed
       — operators want to see each section as it lands, just like a
       streamed AssistantMessage. */
    .section-msg {
        background: $surface;
        margin: 1 0 0 0;
        border-left: thick $secondary;
        height: auto;
    }
    .section-msg-header {
        color: $secondary;
        text-style: bold;
        padding: 0 1;
    }
    .section-msg-body {
        color: $text;
        padding: 0 1 1 1;
    }
    /* `.context-msg` renders `/context` slash-command output as an
       inline boxed card. Accent border-left distinguishes it from
       tool/section/plan cards; padding keeps the hand-formatted columns
       readable. Never collapsed — the breakdown IS the answer, and the
       operator typed the command to read it now. */
    .context-msg {
        background: $boost;
        margin: 1 0 0 0;
        border-left: thick $accent;
        height: auto;
    }
    .context-msg-header {
        color: $accent;
        text-style: bold;
        padding: 0 1;
    }
    .context-msg-body {
        color: $text;
        padding: 0 1 1 1;
    }
    /* `.cmd-card` is the TUI analogue of the Web UI command panel —
       one card per runtime kind (keyvals/list/table/checklist/detail).
       Error kind flips the border so a rejected slash is obvious. */
    .cmd-card {
        background: $boost;
        margin: 1 0 0 0;
        border-left: thick $accent;
        height: auto;
    }
    .cmd-card-error {
        border-left: thick $error;
    }
    .cmd-card-header {
        color: $accent;
        text-style: bold;
        padding: 0 1;
    }
    .cmd-card-error .cmd-card-header {
        color: $error;
    }
    .cmd-card-body {
        color: $text;
        padding: 0 1 1 1;
    }
    /* `.update-msg` renders an `Update`/`Edit` tool call as a colored
       code-diff card — Claude Code's "Update" treatment. Green
       border-left for a normal edit; `.update-msg-failed` flips the
       card to the error palette when the tool reports failure. Never
       collapsed — the diff IS the point. TUI-only: Telegram renders
       file edits as a plain `🔧 Update…` header with no diff body. */
    .update-msg {
        background: $boost;
        margin: 1 0 0 0;
        border-left: thick $success;
        height: auto;
    }
    .update-msg-failed {
        border-left: thick $error;
    }
    .update-msg-header {
        color: $success;
        text-style: bold;
        padding: 0 1;
    }
    .update-msg-failed .update-msg-header {
        color: $error;
    }
    .update-msg-body {
        padding: 0 1 1 1;
    }
    /* `.harness-task-msg` renders the harness ``Task*`` tool family
       (TaskCreate / TaskUpdate / TaskList / TaskGet / TaskOutput /
       TaskStop) as a single inline card that updates in place. Blue
       border-left to distinguish it from the TodoWrite plan (no
       border on plan-checklist footer) and from the cyan section /
       green plan / green update cards. Never collapsed — the card IS
       the task list view. */
    .harness-task-msg {
        background: $boost;
        margin: 1 0 0 0;
        border-left: thick $primary;
        height: auto;
    }
    .harness-task-header {
        color: $primary;
        text-style: bold;
        padding: 0 1;
    }
    .harness-task-body {
        color: $text;
        padding: 0 1 1 1;
    }
    /* `.subagent-box` frames one subagent run (delegate_claude /
       delegate_codex / delegate_antigravity / delegate_agent). Thick yellow
       border-left so the operator can spot the run boundary at a
       glance; the subagent's AssistantMessage + every inner tool
       widget mounts inside `.subagent-box-body`. Header flips to
       green ✅ / red ❌ on finalize. */
    .subagent-box {
        background: $surface;
        margin: 1 0 0 0;
        border-left: thick $warning;
        padding: 0 0 0 1;
        height: auto;
    }
    /* Operator-focused subagent (via ``alt+up`` / ``alt+down``) gets
       a brighter border so the navigation cursor is easy to spot.
       Bumps the background slightly too so even a deeply-nested
       sequence of subagents reads as "this one is the current pick." */
    .subagent-box-focused {
        background: $boost;
        border-left: thick $accent;
    }
    .subagent-box-header {
        color: $warning;
        text-style: bold;
        padding: 0 1;
        height: auto;
    }
    .subagent-box-body {
        padding: 0 0 0 0;
        height: auto;
    }
    /* `.compaction-notice` highlights when the runtime trimmed the
       upstream prompt mid-turn. Yellow border so it stands out from the
       standard tool-result Collapsible — easy to spot in a long
       transcript when scrolling back to debug a "why didn't the model
       see X?" answer. */
    .compaction-notice {
        background: $warning 15%;
        margin: 1 0 0 0;
        border-left: thick $warning;
    }
    .compaction-notice-body {
        color: $text;
        padding: 0 1;
    }
    /* `.empty-eos-notice` flags a premature-EOS recovery: the upstream
       model emitted finish_reason=stop with empty output on a tool
       followup round (Qwen3-fp8 multi-turn EOS bug) and the agent
       auto-injected a "continue" nudge. Same yellow-warning palette as
       the compaction notice so the two read as a family. */
    .empty-eos-notice {
        background: $warning 15%;
        color: $warning;
        margin: 1 0 0 0;
        padding: 0 1;
        border-left: thick $warning;
    }
    /* `.runtime-warning-notice` surfaces a runtime intervention
       (loop detected, distill recovery, force-terminate, premature-end
       nudge) that the bridge used to drop into the assistant message
       bubble as plain text — invisible to anyone not scanning carefully.
       Two variants: `.hard` (loop-hit, force-terminate) in red so it
       reads as an emergency; `.soft` (distill, nudge) in blue so it
       reads as "the runtime is handling this." Thick left border + full
       width so both stand out clearly in the transcript scrollback. */
    .runtime-warning-notice {
        margin: 1 0 0 0;
        padding: 0 1;
    }
    .runtime-warning-notice.hard {
        background: $error 15%;
        color: $error;
        border-left: thick $error;
    }
    .runtime-warning-notice.soft {
        background: $accent 15%;
        color: $accent;
        border-left: thick $accent;
    }
    #slash-popup {
        /* Slash-command autocomplete panel pinned just above the
           input row. ``height: auto`` so empty content collapses
           the row entirely and the input sits flush against the
           transcript when no completion is active. Each visible
           row is one matching command + description. */
        height: auto;
        padding: 0 1;
        background: $boost 30%;
        color: $text-muted;
    }
    #status-events {
        /* Transient-event line above the footer. Hosts one-shot
           messages (session created, /restart scheduled, command
           failed, ``ESC to cancel`` hints, etc.) without polluting
           the persistent identity-and-state footer. ``height: auto``
           so long messages wrap onto a second row on narrow terminals
           rather than getting clipped. */
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    #plan-checklist-wrap {
        /* Wrapper around the inner #plan-checklist Static. Keeps the
           historical border-top + padding on the wrapper so the
           visual separator survives whether the inner body is
           collapsed (hidden) or expanded (scrollable). */
        height: auto;
        padding: 0 1;
        border-top: thick $primary;
        color: $text;
    }
    .plan-checklist-header {
        /* Cursor-pointer hint so operator knows the header toggles. */
        height: 1;
        color: $text;
    }
    .plan-checklist-body-scroll {
        height: auto;
        padding: 0;
    }
    #plan-checklist {
        height: auto;
        padding: 0;
        color: $text;
    }
    #harness-tasks {
        /* Persistent harness-task footer. Rendered via the shared
           ``HarnessTaskListMessage`` widget so the collapse / 40%-
           cap / scroll machinery is inherited, but at the footer
           position we drop the inline card's margin / border so it
           sits flush against the surrounding sticky rows
           (plan-checklist / subagent-counter / status-footer). The
           widget toggles its own ``display`` via
           ``_hide_when_empty`` whenever the task list is empty so
           ``height: auto`` collapses the row to zero. */
        height: auto;
        margin: 0;
        padding: 0;
        background: transparent;
        border-left: none;
        color: $text-muted;
    }
    #harness-tasks .harness-task-header {
        padding: 0 1;
    }
    #harness-tasks .harness-task-body {
        padding: 0 1;
    }
    #harness-tasks .harness-task-body-scroll {
        /* The expanded body's max-height is set programmatically in
           ``HarnessTaskListMessage._apply_body_height_cap`` to 40%
           of the live terminal height (mission D2). Default to a
           ``height: auto`` shrink-to-content so a small task list
           doesn't reserve the full cap. */
        height: auto;
    }
    #subagent-counter {
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    #status-footer {
        /* Single consolidated footer: state · agent · mode · ctx · model ·
           upstream-status. ``height: auto`` so the footer word-wraps on
           narrow terminals — the operator gets to see every segment
           instead of having the right-hand side (upstream status,
           warnings) clipped off the edge. */
        height: auto;
        padding: 0 1;
        background: $primary 25%;
        color: $text;
    }
    #goal-scroll {
        /* Goal text, just above the teacher row. Word-wraps the full goal and
           SCROLLS when it exceeds the cap (long multi-line goals stay bounded).
           ``height: auto`` + ``max-height`` collapses to 0 when empty. */
        height: auto;
        max-height: 4;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #goal-text {
        height: auto;
        /* word-wrap is Rich's default; no truncation. */
    }
    #teacher-row {
        /* Teacher supervision indicator on its OWN row below the goal. */
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    MessageInput {
        /* `height` is driven dynamically by `MessageInput._autosize`
           (clamped to MIN_ROWS..MAX_ROWS based on wrapped line count) —
           Textual's `height: auto` on TextArea does NOT grow with content.
           The fixed default here just sets the initial render before the
           on_mount autosize runs (MIN_ROWS=1 → single-line empty prompt). */
        height: 1;
        width: 1fr;
        border: none;
        background: $boost;
        padding: 0 1;
    }
    MessageInput.-focus {
        border: none;
    }
    /* Pending-turn queue rendered above the input. `display: none` keeps
       it out of the layout while empty; `_refresh_queue_display` flips
       it on whenever `_turn_queue` is non-empty. */
    #queue {
        display: none;
        height: auto;
        max-height: 6;
        background: $surface;
        border-left: thick $warning;
        padding: 0;
        margin: 0;
    }
    #queue > ListItem {
        padding: 0 1;
        height: 1;
        background: $surface;
        color: $text-muted;
    }
    #queue > ListItem.--highlight {
        background: $warning 30%;
        color: $text;
    }
    /* Shell-style `>` prompt to the left of the input. */
    #input-row {
        height: auto;
        background: $boost;
    }
    #input-prompt {
        width: 2;
        height: auto;
        color: $accent;
        background: $boost;
        padding: 0 0 0 1;
        text-style: bold;
    }
    /* Top chrome: row 1 = app name + short hash; row 2 = session title
       (word-wrapped). Textual's default Header joins title/sub_title
       on one line with an em dash and clips with ellipsis. */
    Header {
        height: auto;
        min-height: 1;
    }
    HeaderTitle {
        text-wrap: wrap;
        text-overflow: clip;
        height: auto;
        content-align: center middle;
        text-align: center;
    }
    """

    # ctrl+p is commonly used by tmux / bash readline; remap palette to ctrl+\
    COMMAND_PALETTE_BINDING = "ctrl+backslash"

    # Cap Textual's render rate so SSH sessions don't drown in escape
    # sequences. Local sessions still get 60 FPS (Textual's default);
    # SSH sessions get 20 FPS unless `SHAREDAGENTS_TUI_FPS=<n>` overrides.
    # See `_detect_ssh()` above.
    MAX_FPS = _TUI_FPS

    # Disable mouse tracking by default so the terminal can handle native
    # text selection (shift+drag to copy). With mouse tracking on, the TUI
    # captures all mouse events and the terminal never sees a selection.
    # Collapsibles still work via Ctrl+R / Ctrl+T. Toggle with /mouse.
    mouse_support = False

    BINDINGS = [
        Binding("ctrl+/", "toggle_mouse", "Toggle mouse", show=True, priority=True),
        Binding("escape", "abort", "Abort", show=True, priority=True),
        Binding("ctrl+r", "toggle_thinking", "Toggle reasoning", show=True, priority=True),
        Binding("ctrl+t", "toggle_tool_output", "Toggle tool output", show=True, priority=True),
        # Live-follow (tail -f) toggle for the transcript. ON pins the
        # view to the live tail of the session event stream as it grows;
        # OFF freezes the scroll position so the operator can read back
        # without new deltas yanking them to the bottom. This is the
        # explicit complement to the implicit affinity (scroll-up auto-
        # pauses follow, scroll-to-bottom resumes it) — one keystroke to
        # jump back to the tail and re-arm, or to deliberately pause.
        Binding("ctrl+f", "toggle_follow", "Follow", show=True, priority=True),
        Binding("ctrl+s", "open_shells", "Shells", show=True, priority=True),
        Binding("ctrl+c", "request_quit", "Quit", show=True, priority=True),
        Binding("shift+tab", "cycle_mode", "Mode (plan/auto)", show=True, priority=True),
        # Queue navigation. Ctrl+Up/Down hops between input and the
        # pending-prompt queue (the raw Up/Down arrows are taken by
        # command-history scrolling inside ``MessageInput``). Once the
        # queue is focused, Textual ListView handles Up/Down natively,
        # Enter pulls a row back into the input via
        # ``on_list_view_selected``, and the bindings below add
        # delete (drop without re-edit) + shift+up/down (reorder).
        Binding("ctrl+up", "queue_focus_prev", "Queue ↑", show=True, priority=True),
        Binding("ctrl+down", "queue_focus_next", "Queue ↓", show=True, priority=True),
        # Subagent-box navigation. ``alt+up`` / ``alt+down`` step the
        # focus cursor through every SubagentBoxMessage currently
        # mounted in the transcript (oldest → newest); ``ctrl+o`` (or
        # ``alt+enter`` on terminals that encode it) pops a card that
        # prints the focused subagent's sidecar JSON (prompt + result +
        # elapsed) so the operator can re-read the run without
        # scrolling. Mirrors Claude Code's subagent navigation panel.
        #
        # ``alt+enter`` is kept because terminals that speak the Kitty
        # keyboard protocol DO deliver it distinctly — but most
        # terminals (and tmux's default ESC-prefix meta encoding)
        # collapse Alt+Enter to a bare ``enter`` keypress, so the
        # binding would silently never fire. ``ctrl+o`` ("open") is the
        # reliable, always-deliverable opener and is what the on-screen
        # hints advertise.
        Binding("alt+up", "subagent_focus_prev", "Subagent ↑", show=True, priority=True),
        Binding("alt+down", "subagent_focus_next", "Subagent ↓", show=True, priority=True),
        Binding("ctrl+o", "subagent_open", "Open subagent", show=True, priority=True),
        Binding("alt+enter", "subagent_open", "Open subagent", show=False, priority=True),
    ]

    @classmethod
    def via_agent(cls, agent_cfg, *, endpoint_override: str | None = None) -> AgentChatApp:
        """Construct an AgentChatApp that talks to the given running agent.

        `endpoint_override` (when set) is the URL the daemon should use
        for this session's worker spawns — the resolved form of the
        user's `--endpoint <alias|url>` flag. Sticky for the session
        until changed via the `/model` slash command.
        """
        return AgentChatApp(agent_cfg=agent_cfg, endpoint_override=endpoint_override)


class AgentChatApp(
    LifecycleMixin,
    SessionMixin,
    ModalMixin,
    StatusMixin,
    TasksMixin,
    SubagentMixin,
    InputMixin,
    CommandsMixin,
    StreamMixin,
    ChatApp,
):
    """ChatApp that proxies turns through an AgentClient instead of spawning workers."""

    _MODES_CYCLE = ("auto", "plan")

    # Auto-scroll affinity threshold in pixels (~3 text rows in
    # Textual's default Y-pixel unit, where 1 row = 1 unit). When the
    # transcript's bottom edge is within this distance from the
    # viewport's bottom edge, follow-mode is ON and ``scroll_end``
    # calls actually scroll. Above the threshold the user is reading
    # back-scroll and we leave them where they are. 3 was picked
    # empirically: less than the visible chunk of a typical
    # AssistantMessage (so a single new chunk landing at the bottom
    # while the user reads above doesn't yank them down), more than
    # the line-height jitter Textual's virtual_size emits during
    # mid-frame relayout.
    _SCROLL_FOLLOW_THRESHOLD = 3

    # Persistent follow-mode flag. Initial True so the empty
    # transcript starts in "follow the tail." We DON'T re-derive it
    # from ``virtual_size``/``scroll_y`` on each call (that loses
    # affinity across bulk mounts: a user who was at the bottom
    # right before an 8-row tool-result mount is suddenly 8 rows
    # away from it). Instead the flag is updated by
    # ``TranscriptScroll.watch_scroll_y`` (which fires for wheel,
    # touchpad, scrollbar-drag, PageUp/PageDown, and programmatic
    # scrolls in a single path).
    _scroll_follow: bool = True


_SESSION_SHOW_MAX_LIMIT = 24_000
_SESSION_SHOW_TOOL_CALLS_PER_TURN = 15
_SESSION_SHOW_TOOL_ARGS_CHARS = 160


def _print_tool_calls(turn: dict, indent: str) -> None:
    """``--tool-calls``: one bounded line per call (name + truncated args), so
    an agent taking over a session can see what its predecessor actually ran.
    """
    calls = [c for c in (turn.get("tool_calls") or []) if isinstance(c, dict)]
    for call in calls[:_SESSION_SHOW_TOOL_CALLS_PER_TURN]:
        args = call.get("arguments") or call.get("raw_arguments") or ""
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, default=str)
        args = " ".join(args.split())
        if len(args) > _SESSION_SHOW_TOOL_ARGS_CHARS:
            args = args[: _SESSION_SHOW_TOOL_ARGS_CHARS - 1] + "\u2026"
        name = " ".join(str(call.get("name") or "?").split())[:60]
        print(f"{indent}tool: {name} {args}".rstrip())
    hidden = len(calls) - _SESSION_SHOW_TOOL_CALLS_PER_TURN
    if hidden > 0:
        print(f"{indent}({hidden} more tool call(s) not shown)")


def _turn_text(raw: Any) -> str:
    if isinstance(raw, list):
        return "\n".join(str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in raw)
    return str(raw or "")


def _session_show(args: argparse.Namespace) -> int:
    """Read-only ledger view for `session show`; needs no gateway or agent config.

    A delegate CLI runs this from a scrubbed environment while the gateway may
    be busy serving that same delegate, so it reads the JSONL directly and
    bounds every call's output.
    """
    from sharedagents_core.kernel.layout import sessions_root
    from sharedagents_session.session_reader import load_session_dict

    if args.turn is None and (args.offset is not None or args.limit is not None or args.role is not None):
        print("--offset, --limit and --role need --turn", file=sys.stderr)
        return 2
    if args.start is None and args.count is not None:
        print("--count needs --from", file=sys.stderr)
        return 2
    offset = 0 if args.offset is None else int(args.offset)
    limit = 12_000 if args.limit is None else int(args.limit)
    role = args.role or "both"
    count = 50 if args.count is None else int(args.count)
    if args.turn is not None:
        if offset < 0:
            print("offset must be non-negative", file=sys.stderr)
            return 2
        if not 1 <= limit <= _SESSION_SHOW_MAX_LIMIT:
            print(
                f"limit must be between 1 and {_SESSION_SHOW_MAX_LIMIT}",
                file=sys.stderr,
            )
            return 2
    if args.last is not None and args.last < 1:
        print("--last must be at least 1", file=sys.stderr)
        return 2
    if args.start is not None and count < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2

    d = load_session_dict(
        sessions_root(),
        args.sid,
        expected_agent=args.agent or "jarvis",
    )
    if d is None:
        print(f"session not found: {args.sid}", file=sys.stderr)
        return 2
    turns = d.get("turns") or []
    total = len(turns)

    start, end = 0, total
    if args.turn is not None:
        if not 0 <= args.turn < total:
            print(
                f"turn index out of range: {args.turn} (session has {total} turn(s))",
                file=sys.stderr,
            )
            return 2
    elif args.last is not None:
        start = max(0, total - args.last)
    elif args.start is not None:
        if not 0 <= args.start < total:
            print(
                f"window start out of range: {args.start} (session has {total} turn(s))",
                file=sys.stderr,
            )
            return 2
        start, end = args.start, min(total, args.start + count)

    print(f"session {d.get('hash')}  ·  {d.get('title', '')}")
    print(f"  engine={d.get('engine', '')}  model={d.get('model', '')}  created={d.get('created_at', '')}")
    print(f"  {total} turn(s):")

    if args.turn is not None:
        roles = {"user": ("user",), "assistant": ("content",)}.get(
            role,
            ("user", "content"),
        )
        print(f"  exact context: turn={args.turn} offset={offset} limit={limit} role={role}")
        from sharedagents_core.attachment_context import model_user

        for field in roles:
            label = "assistant" if field == "content" else field
            value = turns[args.turn].get(field)
            if field == "user":
                # The exact context an agent saw, saved-file paths included.
                value = model_user(value, turns[args.turn].get("attachments"))
            text = _turn_text(value)
            fragment = text[offset : offset + limit]
            print(f"[turn {args.turn} {label} chars {offset}:{offset + len(fragment)} of {len(text)}]")
            print(fragment)
        if getattr(args, "tool_calls", False):
            _print_tool_calls(turns[args.turn], "  ")
        return 0

    if start > 0:
        print(f"  ({start} earlier turn(s) not shown; page with --from <index> [--count N] or --last N)")
    for i in range(start, end):
        u = _turn_text(turns[i].get("user"))[:70].replace("\n", " ")
        c = _turn_text(turns[i].get("content"))[:70].replace("\n", " ")
        print(f"   [{i}] user: {u}")
        print(f"       asst: {c}")
        if getattr(args, "tool_calls", False):
            _print_tool_calls(turns[i], "       ")
    if end < total:
        print(f"  ({total - end} later turn(s) not shown)")
    return 0


async def _cmd_session(args: argparse.Namespace) -> int:
    """Dispatch `sharedagents session {list,compact}`. Hits the same agent
    that the TUI / `-p` flows use (selected via `-a/--agent`), so the
    `/compact` slash command in the TUI and `sharedagents session compact`
    from the shell share the same server endpoint and audit format.
    """
    if getattr(args, "op", None) == "show":
        return _session_show(args)
    from sharedagents_tui.agent_client import AgentClient

    cfg = _resolve_agent_or_fail(args.agent or "jarvis")
    if cfg is None:
        return 2
    op = getattr(args, "op", None)
    gateway_url, gateway_token = _gateway_url_and_token()
    client = AgentClient(
        cfg.url,
        agent_name=cfg.name,
        gateway_url=gateway_url,
        gateway_token=gateway_token,
    )

    if op == "list":
        try:
            sessions = await client.list_sessions()
        except Exception as e:  # noqa: BLE001
            print(f"agent unreachable: {e}", file=sys.stderr)
            return 2
        # Newest last-turn activity first. ``last_activity`` is an
        # ISO-8601 timestamp, so a lexicographic sort is chronological;
        # empty values (never-active sessions) sink to the bottom. The
        # old code sorted on ``created_at`` — a key the client dict never
        # carries — so it silently fell back to a hash sort (alphabetical,
        # not by age) and printed an empty timestamp column.
        sessions.sort(key=lambda s: s.get("last_activity") or "", reverse=True)
        for s in sessions[: max(1, int(args.limit))]:
            h = s.get("hash", "?")
            from sharedagents_session.session import SESSION_TITLE_MAX_CHARS

            title = (s.get("title") or "").replace("\n", " ")[:SESSION_TITLE_MAX_CHARS]
            last_activity = s.get("last_activity", "")
            print(f"{h}  {last_activity}  {title}")
        return 0

    if op == "rebuild":
        from sharedagents_session.session_index import SessionIndex

        stats = SessionIndex(cfg.sessions_dir).rebuild()
        print(
            f"rebuilt sessions.db from JSONL: {stats['files']} file(s), "
            f"{stats['events']} event(s), {stats['errors']} error(s)"
        )
        return 0

    if op == "delete":
        from sharedagents_session.session_index import SessionIndex, connect_sessions_db
        from sharedagents_session.session_writer import close_session_writer

        sd = cfg.sessions_dir
        sid = args.sid
        # Cross-agent guard: the unified sessions.db means a known
        # sid for one agent would otherwise be deletable via
        # ``sharedagents -a other session delete <sid>``. Refuse when
        # the session's agent_name is set AND doesn't match the
        # caller's; empty agent_name stays deletable so legacy /
        # unattributed sessions can still be cleaned up.
        try:
            idx = SessionIndex(sd)
            idx.tail(sid)
            con = connect_sessions_db(idx.db_path)
            try:
                row = con.execute(
                    "SELECT agent_name FROM sessions_meta WHERE session_hash=?",
                    (sid,),
                ).fetchone()
            finally:
                con.close()
        except Exception:  # noqa: BLE001
            row = None
        sess_agent = (row["agent_name"] if row is not None else "") or ""
        own_agent = str(getattr(cfg, "name", "") or "")
        if sess_agent and own_agent and sess_agent != own_agent:
            print(
                f"session {sid} belongs to agent {sess_agent!r}, not {own_agent!r}",
                file=sys.stderr,
            )
            return 2

        close_session_writer(sd, sid)
        removed = []
        for name in (f"{sid}.jsonl", f".{sid}.lock"):
            p = sd / name
            try:
                p.unlink()
                removed.append(name)
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"warn: could not remove {name}: {e}", file=sys.stderr)
        try:
            SessionIndex(cfg.sessions_dir).remove_session(sid)
        except Exception as e:  # noqa: BLE001
            print(f"warn: index cleanup failed: {e}", file=sys.stderr)
        if not removed:
            print(f"session not found: {sid}", file=sys.stderr)
            return 2
        print(f"deleted session {sid} ({', '.join(removed)} + index rows)")
        return 0

    if op == "compact":
        sid = args.sid
        if not sid:
            # Convenience: pick the most recently active session so
            # `sharedagents session compact` (no args) operates on whatever the
            # user was last using. most-recent <sid>.jsonl by mtime.
            try:
                files = sorted(
                    cfg.sessions_dir.glob("*.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except OSError:
                files = []
            if not files:
                print("no saved sessions found", file=sys.stderr)
                return 2
            sid = files[0].stem
        try:
            result = await client.compact_session(
                sid,
                keep_recent_turns=args.keep_recent_turns,
                min_result_chars=args.min_result_chars,
                head_chars=args.head_chars,
            )
        except Exception as e:  # noqa: BLE001
            print(f"compact failed: {e}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result, indent=2))
            return 0
        audit = result.get("audit") or {}
        n_trunc = len(audit.get("truncated_tool_call_ids") or [])
        n_drop = len(audit.get("dropped_tool_call_ids") or [])
        n_turns = int(audit.get("dropped_turns") or 0)
        print(
            f"compacted session {sid}: "
            f"{result.get('removed_chars', 0)} chars removed · "
            f"{n_trunc} tool result(s) truncated · "
            f"{n_drop} dropped · {n_turns} old turn(s) discarded"
        )
        return 0

    print(f"unknown session op: {op}", file=sys.stderr)
    return 2


def _cmd_memory(args: argparse.Namespace) -> int:
    """Delegate to the memory admin CLI in sharedagents_mcp.memory (the
    DB-admin surface lives inside the memory package, so this module stays
    engine-free). See sharedagents_mcp.memory.admin_cli.cmd_memory."""
    from sharedagents_mcp.memory.admin_cli import cmd_memory

    return cmd_memory(args)


def _cmd_skills(args: argparse.Namespace) -> int:
    """Dispatch `sharedagents skills {list,show,import}`. See `--help`.

    `list` walks the search chain (env override → per-agent → shared
    → config `skills_dirs` bundles → builtin) and shows the
    unique-by-name set. `show <name>` prints the SKILL.md body.
    `import <name>` copies a skill from an `~/.openclaw/skills` dir
    into sharedagents's shared skills dir as a snapshot (an explicit,
    opt-in migration helper — not part of normal discovery).
    """
    from sharedagents_skill import skills as _skills_mod

    op = getattr(args, "op", None)
    if op == "list":
        # Show every discovered skill, disabled ones included, so an
        # operator can see what's available to re-enable. The `✗`
        # marker = `enabled: false` in ~/.sharedagents/config.json's
        # `skills` map; `discover_skills()` (no flag) is what the agent
        # actually sees, so the diff between the two is the disabled set.
        skills = _skills_mod.discover_skills(include_disabled=True)
        if not skills:
            print("(no skills installed; install or import to ~/.sharedagents/skills/<name>/)")
            return 0
        enabled_names = {s.name for s in _skills_mod.discover_skills()}
        for s in skills:
            mark = "  " if s.name in enabled_names else "✗ "
            line = f"  {mark}{s.name:<28s}  {s.short_description}"
            print(line)
            if getattr(args, "verbose", False):
                print(f"  {'':30s}    source: {s.source}")
        if len(enabled_names) < len(skills):
            print("\n  ✗ = disabled via the `skills` map in ~/.sharedagents/config.json")
        return 0
    if op == "show":
        name = args.name
        s = _skills_mod.find_skill(name)
        if s is None:
            sys.stderr.write(f"skill not found: {name!r}\n")
            return 1
        print(f"# {s.name}")
        if s.description:
            print(f"\n_{s.description}_\n")
        if s.version:
            print(f"version: {s.version}")
        if s.requires_env:
            print(f"requires env: {', '.join(s.requires_env)}")
        if s.requires_bins:
            print(f"requires bins: {', '.join(s.requires_bins)}")
        print(f"\n_(source: {s.source})_\n")
        print("---\n")
        print(s.body)
        return 0
    if op == "import":
        name = args.name
        try:
            dst = _skills_mod.import_openclaw_skill(name)
        except FileNotFoundError as e:
            sys.stderr.write(f"{e}\n")
            return 1
        except FileExistsError as e:
            sys.stderr.write(f"{e}\n")
            return 1
        print(f"imported to {dst}")
        # Re-discover so the user sees it active in the same shell.
        s = _skills_mod.find_skill(name)
        if s is not None:
            print(f"  {s.short_description}")
        return 0
    if op == "create":
        name = args.name
        target_root = Path(args.dir) if getattr(args, "dir", None) else None
        try:
            dst = _skills_mod.create_skill(
                name,
                target_root=target_root,
                description=getattr(args, "description", ""),
                create_venv=getattr(args, "venv", False),
            )
        except (ValueError, FileExistsError) as e:
            sys.stderr.write(f"error: {e}\n")
            return 1
        print(f"created skill {name} at {dst}")
        print(f"  - definition: {dst / 'SKILL.md'}")
        print(f"  - scripts:    {dst / 'scripts'}")
        if getattr(args, "venv", False):
            print(f"  - venv:       {dst / '.venv'}")
        return 0
    if op == "check":
        name = getattr(args, "name", None)
        if name:
            issues = _skills_mod.check_skill(name)
            if issues:
                print(f"Issues for {name}:")
                for iss in issues:
                    print(f"  - {iss}")
                return 1
            print(f"skill {name}: OK")
            return 0
        else:
            skills = _skills_mod.discover_skills(include_disabled=True)
            total_issues = 0
            for s in skills:
                issues = _skills_mod.check_skill(s.source)
                if issues:
                    total_issues += len(issues)
                    print(f"Issues for {s.name}:")
                    for iss in issues:
                        print(f"  - {iss}")
                else:
                    print(f"  {s.name:<28s} OK")
            if total_issues > 0:
                print(f"\nTotal issues found: {total_issues}")
                return 1
            print(f"\nAll {len(skills)} skills OK")
            return 0
    sys.stderr.write("unknown skills op; try `sharedagents skills --help`\n")
    return 2


def _cmd_configure(args: argparse.Namespace) -> int:
    """Interactive or flag-driven configuration.

    Interactive (no flags, no --show) re-uses the same wizard the
    first-run flow runs (`agent_cli.setup_main_wizard(reconfigure=True)`)
    so there is exactly one code path that writes the canonical
    `~/.sharedagents/config.json` + `~/.sharedagents/agents/<name>/config.json`
    files. Every AgentConfig variable is configurable through the
    wizard (advanced fields gated behind a single Y/N prompt).

    Flag-driven: set individual fields via --endpoint, --model,
    --telegram-bot-token, etc. — useful for scripted bootstraps.
    --show prints the effective canonical config.
    """
    from sharedagents_agent.agent import (
        AgentConfig,
        global_config_path,
        load_global_config,
        save_global_config,
    )

    agent_name = getattr(args, "agent", "jarvis") or "jarvis"
    tg_token = getattr(args, "telegram_bot_token", None)
    tg_ids = getattr(args, "telegram_allowed_chat_ids", None)
    tg_user_ids = getattr(args, "telegram_allowed_user_ids", None)
    tg_mention_required = getattr(args, "telegram_mention_required", None)

    # ---- --show ----
    if args.show:
        gpath = global_config_path()
        print(f"# global config: {gpath}{'  (does not exist yet)' if not gpath.exists() else ''}")
        print(json.dumps(load_global_config(), indent=2, ensure_ascii=False))
        try:
            acfg = AgentConfig.load(agent_name)
            print(f"\n# per-agent config: {acfg.config_path}")
            print(acfg.config_path.read_text(encoding="utf-8").strip())
        except FileNotFoundError:
            print(f"\n# per-agent config for {agent_name!r}: not configured")
        return 0

    # ---- Flag-driven: apply any flags and exit ----
    # Map the CLI flags (legacy `cfgmod.FIELDS` names) to canonical
    # AgentConfig / global-config field names. Keeps `--endpoint`,
    # `--model` etc. working as a scripted shortcut without writing the
    # legacy `~/.config/sharedagents/config.json` file the runtime ignores.
    FLAG_TO_GLOBAL = {
        "endpoint": "upstream_endpoint",
        "model": "upstream_model",
        "temperature": "temperature",
        "enable_thinking": "enable_thinking",
    }
    flag_global_changes = {
        FLAG_TO_GLOBAL[f]: getattr(args, f) for f in FLAG_TO_GLOBAL if getattr(args, f, None) is not None
    }
    # `--system` is a per-agent field; route it there.
    sys_extra_arg = getattr(args, "system", None)

    any_flag = (
        bool(flag_global_changes)
        or sys_extra_arg is not None
        or tg_token is not None
        or tg_ids is not None
        or tg_user_ids is not None
        or tg_mention_required is not None
    )

    if any_flag:
        # Global fields → ~/.sharedagents/config.json (canonical).
        if flag_global_changes:
            glob = load_global_config()
            glob.update(flag_global_changes)
            path = save_global_config(glob)
            print(f"wrote global config → {path}")

        # ``--endpoint`` / ``--model`` is the scripted fresh-install path:
        # write models[], seed jarvis, start the gateway. A partial flag
        # (temperature only, or one of endpoint/model while the other is
        # still empty) stays a field patch.
        glob = load_global_config()
        endpoint = str(glob.get("upstream_endpoint") or "").strip()
        model = str(glob.get("upstream_model") or "").strip()
        touched_upstream = "upstream_endpoint" in flag_global_changes or "upstream_model" in flag_global_changes
        if touched_upstream and endpoint and model:
            from sharedagents_gateway.agent_cli import apply_endpoint_model_config

            extras = {k: v for k, v in flag_global_changes.items() if k not in ("upstream_endpoint", "upstream_model")}
            cfg = apply_endpoint_model_config(
                endpoint,
                model,
                extras=extras or None,
                start_gateway=True,
            )
            if cfg is not None:
                print(f"wrote per-agent config → {cfg.config_path}")

        # Per-agent fields → ~/.sharedagents/agents/<name>/config.json.
        if sys_extra_arg is not None:
            try:
                acfg = AgentConfig.load(agent_name)
            except FileNotFoundError:
                print(
                    f"agent {agent_name!r} not configured. run `sharedagents configure` (no flags) for the full wizard."
                )
                return 2
            acfg.system = sys_extra_arg
            acfg.save()
            print(f"wrote per-agent config → {acfg.config_path}")

        # Telegram fields → ~/.sharedagents/telegram/config.json
        # (the bot is now its own service; agent config is the wrong
        # home for these).
        if tg_token is not None or tg_ids is not None or tg_user_ids is not None or tg_mention_required is not None:
            from sharedagents_telegram.config import TelegramConfig

            tg_cfg = TelegramConfig.load()
            if tg_token is not None:
                tg_cfg.bot_token = tg_token
            if tg_ids is not None:
                try:
                    tg_cfg.allowed_chat_ids = [int(x.strip()) for x in tg_ids.split(",") if x.strip()]
                except ValueError:
                    print(f"invalid chat_ids: {tg_ids!r} (expected comma-separated integers)")
                    return 2
            if tg_user_ids is not None:
                # Empty string is the "clear the allowlist" signal —
                # ``--telegram-allowed-user-ids ""`` reverts to "any
                # member of an allowed chat may interact". Non-empty
                # input is parsed strictly: each comma-separated token
                # must be an int (Telegram user ids ARE numeric).
                raw = tg_user_ids.strip()
                if not raw:
                    tg_cfg.allowed_user_ids = []
                else:
                    try:
                        tg_cfg.allowed_user_ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
                    except ValueError:
                        print(f"invalid user_ids: {tg_user_ids!r} (expected comma-separated integers)")
                        return 2
            if tg_mention_required is not None:
                # argparse stores ``--telegram-mention-required`` as
                # the canonical token (``on``/``off``); coerce to the
                # bool the config carries.
                token = str(tg_mention_required).strip().lower()
                if token in ("on", "true", "yes", "1", "enable", "enabled"):
                    tg_cfg.mention_required_in_groups = True
                elif token in ("off", "false", "no", "0", "disable", "disabled"):
                    tg_cfg.mention_required_in_groups = False
                else:
                    print(f"invalid --telegram-mention-required value {tg_mention_required!r} (expected on/off)")
                    return 2
            tg_cfg.save()
            print(f"wrote telegram config → {tg_cfg.config_path}")
        return 0

    # ---- Interactive: same wizard the first-run flow uses ----
    # `setup_main_wizard(reconfigure=True)` pre-fills every prompt from
    # the existing config (or falls through to first-run if nothing's
    # there yet) and covers every AgentConfig variable, including the
    # advanced fields gated behind a single Y/N.
    if not sys.stdin.isatty():
        print("stdin is not a tty — use flags (--endpoint, --model, --telegram-bot-token, …) or run interactively.")
        return 1

    if agent_name != "jarvis":
        # The wizard is hard-coded to the `jarvis` orchestrator (the only agent
        # with a first-run path; `main` retired). Other agents should be
        # configured via flags or by hand-editing their config.json.
        print(
            f"interactive `sharedagents configure` only supports the `jarvis` agent; "
            f"got --agent {agent_name!r}. Use flags (--endpoint, …) or edit "
            f"{(AgentConfig(name=agent_name).config_path)} directly."
        )
        return 2

    from sharedagents_gateway import agent_cli as agentcli

    cfg = agentcli.setup_main_wizard(reconfigure=True)
    return 0 if cfg is not None else 1


def _print_agent_ready(cfg, *, model_override: str | None = None) -> None:
    """One-line health summary printed to stderr right before the TUI starts.

    Probes the running sharedagents gateway at ``/acp/<agent>/acp`` —
    every agent is a stdio child of that gateway, never a standalone
    daemon. ``DOWN`` means the gateway isn't running (start it with
    ``sharedagents gateway install-service`` or ``sharedagents gateway start``).

    ``model_override`` is the ``--model <alias>`` the operator selected
    for THIS run. Only the oneshot ``-p`` path passes it (that path
    forwards ``--model`` to ``create_session``); the interactive TUI
    does NOT honour ``--model``, so its launcher leaves this ``None`` to
    keep the banner from advertising a model the TUI never runs. When
    set, the banner advertises the selected model id + its endpoint —
    not the agent's configured default ``upstream_model`` — so the
    readiness line agrees with the model the turn actually runs on (and
    with the post-turn metrics footer). Resolving via
    ``cfg.resolve_model`` also moves the ``upstream=`` field when the
    alias lives on another host (e.g. a remote ``deepseek`` row).
    """
    upstream = getattr(cfg, "upstream_endpoint", "") or "<unresolved>"
    model = getattr(cfg, "upstream_model", "") or "<unresolved>"
    if model_override:
        row: dict = {}
        try:
            row = cfg.resolve_model(model_override) or {}
        except Exception:  # noqa: BLE001
            row = {}
        # Prefer the resolved id/endpoint; fall back to the raw alias
        # when the row is synthetic/unknown so the banner still reflects
        # the operator's choice rather than silently snapping to default.
        model = str(row.get("id") or "") or model_override
        if row.get("baseUrl"):
            upstream = str(row["baseUrl"])
    gateway_url, gateway_token = _gateway_url_and_token()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if gateway_token:
        headers["Authorization"] = f"Bearer {gateway_token}"
    health = "DOWN"
    try:
        r = httpx.post(
            f"{gateway_url}/acp/{cfg.name}/acp",
            content=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": 1, "clientCapabilities": {}},
                }
            ).encode("utf-8"),
            headers=headers,
            timeout=2.0,
        )
        if r.status_code == 200:
            try:
                if isinstance(r.json().get("result"), dict):
                    health = "ok"
            except (ValueError, AttributeError):
                pass
    except httpx.HTTPError:
        pass
    gw_pid = "?"
    try:
        from sharedagents_gateway.gateway import _gateway_pidfile

        pf = _gateway_pidfile()
        if pf.is_file():
            gw_pid = pf.read_text().strip() or "?"
    except Exception:  # noqa: BLE001
        pass
    sys.stderr.write(f"  agent={cfg.name} gateway_pid={gw_pid} upstream={upstream} model={model} ({health})\n")
    if _IS_SSH:
        sys.stderr.write(
            f"  [ssh] tui throttled to {_TUI_FPS} fps; "
            f"override with SHAREDAGENTS_TUI_FPS=<n>. "
            f"For best feel, use `mosh` or `ssh -C`.\n"
        )
    sys.stderr.flush()


def _resolve_agent_or_fail(
    name: str,
    *,
    model_override: str | None = None,
) -> AgentConfig | None:
    """Resolve an agent and confirm the gateway is reachable.

    For the ``main`` agent: if it's not configured, run the first-run
    wizard so config exists on disk. We never spawn a process here —
    the sharedagents gateway (``sharedagents-gateway.service``) owns lifecycle
    and is the sole producer of agent stdio children. If the gateway
    isn't running, fail with a helpful systemctl pointer.
    """
    from sharedagents_agent.agent import AgentConfig
    from sharedagents_gateway import agent_cli as agentcli

    if name == "jarvis":
        cfg = agentcli.ensure_main_configured()
    else:
        try:
            cfg = AgentConfig.load(name)
        except FileNotFoundError:
            sys.stderr.write(f"agent {name!r} not configured.\nrun `sharedagents agent init {name}` first.\n")
            return None
    if cfg is None:
        return None

    if not _gateway_alive():
        # Self-bootstrap: bring the gateway up in user space (no sudo, no
        # systemd) instead of stranding the operator on a manual
        # `systemctl` / `gateway start` to-do. This is what makes a bare
        # `pip install sharedagents` + `sharedagents` "just work" — the TUI starts
        # its own gateway on demand and connects. A background gateway
        # doesn't survive a reboot, but the next `sharedagents` launch starts
        # it again; the survive-reboot systemd path stays an explicit
        # opt-in (`sharedagents gateway install-service`).
        sys.stderr.write("sharedagents stack not running — starting orchestrator + gateway…\n")
        try:
            from sharedagents_gateway.gateway_cli import start_gateway_background

            res = start_gateway_background()
        except Exception as e:  # noqa: BLE001
            res = {"ok": False, "error": f"could not import/start gateway: {e}"}
        if not (res.get("ok") or res.get("already_running")) or not _gateway_alive():
            sys.stderr.write(
                "could not start the sharedagents stack automatically:\n"
                f"  {res.get('error') or 'unknown error'}\n"
                "start it yourself with `sharedagents gateway start`, or run "
                "`sharedagents gateway install-service` for a boot service.\n"
            )
            return None
        sys.stderr.write(
            f"  ✓ stack up → gateway {res.get('url') or ''} orchestrator {res.get('orchestrator_url') or ''}\n"
        )

    gateway_url, gateway_token = _gateway_url_and_token()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if gateway_token:
        headers["Authorization"] = f"Bearer {gateway_token}"
    # Readiness handshake. The gateway returns HTTP 202 while the agent child
    # is still spawning OR is busy with another session — it is NOT unreachable,
    # just not-ready-yet. Poll ``initialize`` until it 200s (or a deadline)
    # instead of bailing, so a one-shot fired after an idle-reap or during
    # concurrent/rapid use WAITS for the agent instead of returning empty
    # output. (Fixes the "HTTP 202 not reachable" one-shot bail that made
    # overlapping ``sharedagents -p`` calls — e.g. benchmark drivers — drop answers.)
    import math as _math
    import time as _time

    try:
        _ready_timeout = float(os.environ.get("SHAREDAGENTS_ONESHOT_READY_TIMEOUT_S", "90") or 90)
    except (TypeError, ValueError):
        _ready_timeout = 90.0
    # Reject non-finite / negative overrides (``inf`` → infinite loop; ``nan`` →
    # comparisons always False → retries silently disabled).
    if not _math.isfinite(_ready_timeout) or _ready_timeout < 0:
        _ready_timeout = 90.0
    _deadline = _time.monotonic() + _ready_timeout
    _init_body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": 1, "clientCapabilities": {}},
        }
    ).encode("utf-8")
    while True:
        try:
            r = httpx.post(
                f"{gateway_url}/acp/{name}/acp",
                content=_init_body,
                headers=headers,
                timeout=5.0,
            )
        except httpx.HTTPError as e:
            # Transient connect/read error while the gateway/agent warms up.
            if _time.monotonic() < _deadline:
                _time.sleep(1.5)
                continue
            sys.stderr.write(
                f"agent {name!r} did not become ready on the gateway at "
                f"{gateway_url}/acp/{name} within {int(_ready_timeout)}s "
                f"(last error: {e})\n"
            )
            return None
        if r.status_code == 200:
            break
        # 202 = agent spawning / busy — not unreachable; wait and retry.
        if r.status_code == 202 and _time.monotonic() < _deadline:
            _time.sleep(1.5)
            continue
        if r.status_code == 202:
            sys.stderr.write(
                f"agent {name!r} did not become ready on the gateway at "
                f"{gateway_url}/acp/{name} within {int(_ready_timeout)}s "
                f"(still spawning/busy: HTTP 202)\n"
            )
        else:
            sys.stderr.write(
                f"agent {name!r} not reachable on the gateway at {gateway_url}/acp/{name}: HTTP {r.status_code}\n"
            )
        return None
    _print_agent_ready(cfg, model_override=model_override)
    return cfg


def _launch_tui_via_agent(args: argparse.Namespace) -> int:
    # NOTE: deliberately NOT forwarding ``--model`` to the banner here.
    # The interactive TUI does not honour ``--model`` (``ChatApp.via_agent``
    # builds the session from ``agent_cfg.upstream_model``), so reflecting
    # the override in the readiness banner would make the banner advertise
    # a model the TUI never actually runs. The ``-p`` oneshot path DOES
    # forward ``--model`` to ``create_session``, so it passes the override.
    cfg = _resolve_agent_or_fail(args.agent or "jarvis")
    if cfg is None:
        return 2
    # `--endpoint <alias|url>`: resolve against the agent's named-endpoint
    # list, then carry both the resolved URL (for the daemon) and the
    # display alias (for the status bar) into the AgentChatApp. We mutate
    # `cfg` in place too so the initial status bar — rendered before any
    # session is created — already shows the override; the resolved
    # value is also passed to `create_session` so the daemon actually
    # spawns workers against the alternate host.
    initial_endpoint_override: str | None = None
    if args.endpoint:
        resolved = cfg.resolve_endpoint(args.endpoint)
        if resolved:
            cfg.upstream_endpoint = resolved
            cfg.upstream_alias = cfg.resolve_alias(args.endpoint)
            initial_endpoint_override = resolved
    app = ChatApp.via_agent(cfg, endpoint_override=initial_endpoint_override)
    # CLI override of the initial permission mode (Shift+Tab still cycles
    # within the session). Useful for `sharedagents --plan` to start in plan
    # mode without having to remember to tab over.
    cli_mode = getattr(args, "mode", None)
    if cli_mode and hasattr(app, "_permission_mode"):
        app._permission_mode = cli_mode
    app.run()
    return 0


def _oneshot_state_path(engine_kind: Any, agent_name: str) -> Path:
    """Per-engine/cwd state for external one-shot `--continue`."""
    import hashlib

    from sharedagents_agent.external_registry import spec_for

    short_name = spec_for(engine_kind).short_name
    cwd = str(Path.cwd().resolve())
    cwd_hash = hashlib.sha256(cwd.encode("utf-8")).hexdigest()[:16]
    base = Path.home() / ".sharedagents" / f"oneshot_{short_name}" / agent_name
    return base / f"{cwd_hash}.json"


def _read_oneshot_state(engine_kind: Any, agent_name: str) -> str | None:
    path = _oneshot_state_path(engine_kind, agent_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    sid = data.get("session_id") if isinstance(data, dict) else None
    return sid if isinstance(sid, str) and sid else None


def _write_oneshot_state(engine_kind: Any, agent_name: str, sid: str) -> None:
    path = _oneshot_state_path(engine_kind, agent_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"session_id": sid}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _engine_session_id_from_sidecar(
    *,
    short_name: str,
    session_id_key: str | None,
    sa_sid: str,
    sessions_dir: str | None,
) -> str | None:
    """Read the vendor CLI session id from the delegate sidecar, if present."""
    if not session_id_key or not sa_sid:
        return None
    from sharedagents_agent.delegates.base import _read_delegate_state

    state = _read_delegate_state(short_name, sa_sid, sessions_dir)
    value = state.get(session_id_key)
    return value if isinstance(value, str) and value else None


async def _run_external_oneshot(
    engine_kind: Any,
    args: argparse.Namespace,
    prompt: str,
) -> int:
    """Run one registry-backed CLI delegate through the shared agent bridge."""
    from rich.console import Console

    from sharedagents_agent.agents.engine import EngineKind
    from sharedagents_agent.external_registry import spec_for
    from sharedagents_tui.agent_client import AgentClient

    spec = spec_for(engine_kind)
    if spec.adapter != "cli":
        raise ValueError(f"{spec.kind.value!r} is not a CLI one-shot engine")

    short_name = spec.short_name
    sa_mode = getattr(args, "mode", None) or "auto"
    if spec.kind is EngineKind.CLAUDE_CLI:
        mode_label = _claude_permission_mode(sa_mode)
    elif spec.kind is EngineKind.CODEX_CLI:
        mode_label = _codex_sandbox_label(sa_mode)
    else:
        mode_label = sa_mode
    continue_flag = bool(getattr(args, "bridge_continue", False))
    agent_name = args.agent or "jarvis"

    cfg = _resolve_agent_or_fail(agent_name)
    if cfg is None:
        return 2

    out_tty = sys.stdout.isatty()
    err_tty = sys.stderr.isatty()
    err_console = Console(stderr=True, soft_wrap=True, highlight=False)
    out_console = Console()

    gateway_url, gateway_token = _gateway_url_and_token()
    client = AgentClient(
        cfg.url,
        agent_name=cfg.name,
        gateway_url=gateway_url,
        gateway_token=gateway_token,
    )
    try:
        sid: str | None = None
        resumed = False
        if continue_flag:
            cached = _read_oneshot_state(spec.kind, agent_name)
            if cached:
                try:
                    await client.get_session(cached)
                    sid = cached
                    resumed = True
                except Exception:  # noqa: BLE001
                    sid = None
        if sid is None:
            sess = await client.create_session(
                workspace_cwd=os.getcwd(),
                workspace_origin=short_name,
            )
            sid = sess["hash"]

        sid_short = sid[:12]
        head = (
            f"── {short_name} ── mode={mode_label}"
            f"{' resume=' + sid_short if resumed else ' new'}"
            f"{' (' + agent_name + ')' if agent_name != 'jarvis' else ''}"
            f" ──"
        )
        if err_tty:
            err_console.print(f"[dim cyan]{head}[/dim cyan]")
        else:
            sys.stderr.write(head + "\n")
            sys.stderr.flush()

        answer_buf: list[str] = []
        engine_session_id: str | None = None
        turn_id_seen: str | None = None
        aborted = False
        delegate_done = False
        delegate_errored = False
        delegate_err = ""
        outer_stream_received = False
        outer_call_id: str | None = None

        def _emit_heartbeat(tool_name: str) -> None:
            if err_tty:
                err_console.print(f"[dim]… {tool_name}[/dim]")
            else:
                sys.stderr.write(f"… {tool_name}\n")
                sys.stderr.flush()

        try:
            async for event_name, data, tid in client.delegate(
                sid,
                prompt=prompt,
                tool=f"delegate_{short_name}",
                mode=sa_mode,
            ):
                turn_id_seen = tid or turn_id_seen
                if event_name == "started":
                    continue
                if event_name in ("delta", "tool_delta"):
                    call_id = str(data.get("call_id") or "")
                    if outer_call_id and call_id and call_id != outer_call_id:
                        continue
                    txt = data.get("text") or ""
                    if txt:
                        answer_buf.append(txt)
                        outer_stream_received = True
                        if not out_tty:
                            sys.stdout.write(txt)
                            sys.stdout.flush()
                    continue
                if event_name == "chunk":
                    delta = (data.get("choices") or [{}])[0].get("delta") or {}
                    txt = delta.get("content") or ""
                    if txt:
                        answer_buf.append(txt)
                        if not out_tty:
                            sys.stdout.write(txt)
                            sys.stdout.flush()
                    continue
                if event_name == "tool_call":
                    call_id = str(data.get("call_id") or "")
                    if outer_call_id is None:
                        outer_call_id = call_id
                        continue
                    if call_id != outer_call_id:
                        _emit_heartbeat(str(data.get("tool") or "tool"))
                    continue
                if event_name == "tool_result":
                    call_id = str(data.get("call_id") or "")
                    if outer_call_id is not None and call_id != outer_call_id:
                        continue
                    delegate_done = True
                    content = data.get("content") or ""
                    if not bool(data.get("ok", False)):
                        delegate_errored = True
                        delegate_err = (content or f"{short_name} failed")[:300]
                        continue
                    if not outer_stream_received and content:
                        answer_buf.append(content)
                        if not out_tty:
                            sys.stdout.write(content)
                            sys.stdout.flush()
                    continue
                if event_name == "error":
                    delegate_errored = True
                    delegate_err = (data.get("error") or data.get("message") or "delegate failed")[:300]
                    break
                if event_name == "done":
                    if spec.session_id_key:
                        value = data.get(spec.session_id_key)
                        if isinstance(value, str) and value:
                            engine_session_id = value
                    state = data.get("state")
                    if state == "aborted":
                        aborted = True
                    elif not delegate_done and not delegate_errored:
                        delegate_errored = True
                        delegate_err = ("".join(answer_buf) or f"{short_name} failed")[:300]
                    break
        except (KeyboardInterrupt, asyncio.CancelledError):
            aborted = True
            if turn_id_seen:
                try:
                    await client.turn_stop(turn_id_seen)
                except Exception:  # noqa: BLE001
                    pass

        if spec.session_id_key and not engine_session_id and sid:
            sdir = str(getattr(cfg, "sessions_dir", "") or os.environ.get("SHAREDAGENTS_SESSIONS_DIR") or "")
            engine_session_id = _engine_session_id_from_sidecar(
                short_name=short_name,
                session_id_key=spec.session_id_key,
                sa_sid=sid,
                sessions_dir=sdir or None,
            )

        if delegate_errored and not aborted:
            sys.stderr.write(f"\n{short_name} error: {delegate_err or f'{short_name} failed'}\n")
            return 1

        answer = "".join(answer_buf)
        if not answer.strip() and not aborted:
            sys.stderr.write("(empty answer)\n")
            return 1

        if out_tty and answer.strip():
            sys.stdout.write("\n")
            out_console.print(RichMarkdown(answer))
        else:
            if answer and not answer.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()

        _write_oneshot_state(spec.kind, agent_name, sid)

        bits = [f"sid={sid_short}"]
        if engine_session_id:
            bits.append(f"{short_name}={engine_session_id[:12]}")
        bits.append("`-c` to continue")
        line = "── " + " · ".join(bits) + " ──"
        if err_tty:
            err_console.print(f"[dim cyan]{line}[/dim cyan]")
        else:
            sys.stderr.write(line + "\n")

        return 130 if aborted else 0
    finally:
        await client.close()


async def _run_oneshot_via_agent(args: argparse.Namespace, prompt: str) -> int:
    from rich.console import Console

    from sharedagents_tui.agent_client import AgentClient

    cfg = _resolve_agent_or_fail(
        args.agent or "jarvis",
        model_override=getattr(args, "model", None),
    )
    if cfg is None:
        return 2
    # `--endpoint <alias|url>`: resolve via the agent's named-endpoint
    # list, then send the URL through `create_session` so the daemon
    # binds this oneshot session to the alternate upstream.
    endpoint_override: str | None = None
    if args.endpoint:
        resolved = cfg.resolve_endpoint(args.endpoint)
        if resolved:
            cfg.upstream_endpoint = resolved
            cfg.upstream_alias = cfg.resolve_alias(args.endpoint)
            endpoint_override = resolved

    # `--model <alias>`: pass the alias to ``create_session`` so it
    # calls ``session_set_config_option(sid, "model", <alias>)`` on the
    # bridge — which resolves via ``cfg.models[]`` and updates the
    # session's upstream model + endpoint. Previously ignored entirely.
    model_override: str | None = None
    if args.model:
        model_override = args.model

    out_tty = sys.stdout.isatty()
    err_tty = sys.stderr.isatty()
    err_console = Console(stderr=True, soft_wrap=True, highlight=False)
    out_console = Console()

    gateway_url, gateway_token = _gateway_url_and_token()
    client = AgentClient(
        cfg.url,
        agent_name=cfg.name,
        gateway_url=gateway_url,
        gateway_token=gateway_token,
    )
    try:
        sess = await client.create_session(
            workspace_cwd=os.getcwd(),
            workspace_origin="oneshot",
            endpoint=endpoint_override,
            model=model_override,
        )
        sid = sess["hash"]

        thinking_started = False
        content_buf: list[str] = []
        metrics: dict | None = None
        turn_id_seen: str | None = None
        aborted = False

        try:
            async for event, data, tid in client.session_turn(
                sid,
                user=prompt,
                enable_thinking=getattr(args, "enable_thinking", None),
                permission_mode=getattr(args, "mode", None),
            ):
                turn_id_seen = tid or turn_id_seen
                if event == "started":
                    continue
                if event == "chunk":
                    delta = (data.get("choices") or [{}])[0].get("delta") or {}
                    rc = delta.get("reasoning") or delta.get("reasoning_content")
                    if rc:
                        if not thinking_started:
                            if err_tty:
                                err_console.print("[dim cyan]── thinking ──[/dim cyan]")
                            else:
                                sys.stderr.write("── thinking ──\n")
                            thinking_started = True
                        if err_tty:
                            err_console.print(f"[dim]{rc}[/dim]", end="")
                        else:
                            sys.stderr.write(rc)
                            sys.stderr.flush()
                    txt = delta.get("content")
                    if txt:
                        content_buf.append(txt)
                elif event == "metrics":
                    metrics = data
                elif event == "error":
                    sys.stderr.write(f"\nerror: {data.get('message')}\n")
                    return 1
                elif event == "done":
                    pass
        except KeyboardInterrupt:
            aborted = True
            if turn_id_seen:
                try:
                    await client.turn_stop(turn_id_seen)
                except httpx.HTTPError:
                    pass

        if thinking_started:
            sys.stderr.write("\n")

        answer = "".join(content_buf)
        if not answer.strip():
            sys.stderr.write("(empty answer)\n")
            return 1

        if out_tty:
            if err_tty:
                err_console.print("[dim cyan]── answer ──[/dim cyan]")
            out_console.print(RichMarkdown(answer))
        else:
            sys.stdout.write(answer)
            if not answer.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()

        if metrics:
            parts = []
            if metrics.get("model"):
                parts.append(str(metrics["model"]))
            if metrics.get("ttft_ms") is not None:
                parts.append(f"ttft {metrics['ttft_ms']:.0f}ms")
            prefill_v = metrics.get("prefill_ms")
            accept_v = metrics.get("accept_ms")
            if accept_v is None:
                accept_v = metrics.get("pft_ms")
            if prefill_v is not None:
                parts.append(f"prefill {prefill_v:.0f}ms")
            elif accept_v is not None:
                parts.append(f"accept {accept_v:.0f}ms")
            if metrics.get("tps") is not None:
                est = "≈" if metrics.get("tokens_estimated") else ""
                parts.append(f"{metrics['tps']:.1f} tok/s{est}")
            if metrics.get("completion_tokens") is not None:
                parts.append(f"{metrics['completion_tokens']} tok")
            if metrics.get("wt_ms") is not None:
                parts.append(f"wt {metrics['wt_ms'] / 1000:.1f}s")
            line = "── " + " · ".join(parts) + " ──"
            if err_tty:
                err_console.print(f"[dim cyan]{line}[/dim cyan]")
            else:
                sys.stderr.write(line + "\n")

        return 130 if aborted else 0
    finally:
        await client.close()


def _launch_tui(args: argparse.Namespace) -> int:
    cfg = cfgmod.effective()
    endpoint = cfgmod.resolve_endpoint(args.endpoint, cfg)
    model = args.model if args.model is not None else cfg["model"]
    system = cfg["system"]
    # Resolve model-specific temperature overrides (e.g. kimi-k3 requires temp=1)
    from sharedagents_model.provider_presets import resolve_temperature

    temperature = resolve_temperature(model, cfg["temperature"])

    if not model:
        try:
            model = asyncio.run(discover_model(endpoint))
        except Exception as e:  # noqa: BLE001
            detail = str(e) or repr(e)
            sys.stderr.write(f"failed to discover model from {endpoint}: {detail}\n")
            sys.stderr.write(
                "fix the endpoint with `sharedagents configure --endpoint http://IP:PORT/v1`\n"
                "or pin a model with `sharedagents configure --model NAME`.\n"
            )
            return 2

    enable_thinking = bool(cfg["enable_thinking"])
    if getattr(args, "enable_thinking", None) is not None:
        enable_thinking = args.enable_thinking

    app = ChatApp(
        endpoint=endpoint,
        model=model,
        system=system,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
    app.run()
    return 0


def main() -> None:
    from sharedagents_core.launch import ensure_utf8_mode

    ensure_utf8_mode()

    # Make Textual's input thread tolerant of non-UTF-8 stdin bytes
    # before any TUI launches. A single stray byte (legacy X10 mouse
    # report, Alt/Meta high-bit key, non-UTF-8 paste over SSH) otherwise
    # raises UnicodeDecodeError inside Textual's strict input decoder,
    # panics the app, and leaves the TUI frozen/laggy. Uses
    # surrogateescape (not replace) so mouse/CSI bytes inside escape
    # sequences are not widened to U+FFFD and leaked into the input box.
    # See sharedagents_tui.input_hardening for the why and the mechanism.
    from sharedagents_tui.input_hardening import install_tolerant_input_decoder

    install_tolerant_input_decoder()

    parser = argparse.ArgumentParser(prog="sharedagents", description="Sovereign-inference TUI chat.")
    parser.add_argument("--endpoint", help="Endpoint alias (from config) or raw URL for this run only")
    parser.add_argument("--model", help="Override model for this run only (required — no auto-discover)")
    parser.add_argument(
        "-p",
        "--prompt",
        help="One-shot mode: send PROMPT, stream thinking to stderr, render answer to stdout, exit.",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Qwen3-style chat-template thinking for this run",
    )

    parser.add_argument(
        "-a",
        "--agent",
        default="jarvis",
        help="Name of the agent to talk to (default: jarvis).",
    )

    # Permission mode: pick one. The mutually-exclusive aliases (`--plan`,
    # `--ask`, `--auto`) are convenience shortcuts; `--mode <name>` is the
    # canonical form and accepts the literal mode names. If unset, the
    # agent's configured default applies (typically `auto`).
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--mode",
        choices=("plan", "ask", "auto"),
        default=None,
        help="Permission mode for this run (overrides agent default).",
    )
    mode_grp.add_argument(
        "--plan",
        dest="mode",
        action="store_const",
        const="plan",
        help="Shortcut for --mode plan (read-only, exhaustive answers).",
    )
    mode_grp.add_argument(
        "--ask",
        dest="mode",
        action="store_const",
        const="ask",
        help="Shortcut for --mode ask (prompt before mutations).",
    )
    mode_grp.add_argument(
        "--auto",
        dest="mode",
        action="store_const",
        const="auto",
        help="Shortcut for --mode auto (autonomous; pause on destructive).",
    )

    # External bridge flags route the prompt through the matching
    # ``delegate_*`` tool — the same path as the TUI slash command and local
    # model's autonomous tool emit. Active mode (--plan/--ask/--auto)
    # maps to the external agent's permission level. `--continue` /
    # `-c` resumes the most-recent `sharedagents --<bridge> -p` session for
    # this `(agent, cwd, bridge)` triple, so external-agent session
    # reuse chains across bash-bridge calls. Mutually exclusive: one
    # call goes to one bridge.
    bridge_grp = parser.add_mutually_exclusive_group()
    bridge_grp.add_argument(
        "--claude",
        action="store_true",
        default=False,
        help="One-shot via the agent's delegate_claude tool (requires a running agent).",
    )
    bridge_grp.add_argument(
        "--codex",
        action="store_true",
        default=False,
        help="One-shot via the agent's delegate_codex tool (requires a running agent).",
    )
    bridge_grp.add_argument(
        "--antigravity",
        action="store_true",
        default=False,
        help=("One-shot via the agent's delegate_antigravity tool (requires a running agent)."),
    )
    parser.add_argument(
        "--continue",
        "-c",
        dest="bridge_continue",
        action="store_true",
        default=False,
        help=(
            "(with an external bridge flag) resume the most recent "
            "`sharedagents --<bridge> -p` session for this (agent, cwd, bridge) "
            "triple so the external agent's resume keeps the same "
            "conversation across invocations."
        ),
    )

    sub = parser.add_subparsers(dest="cmd")

    # `sharedagents agent ...`
    from sharedagents_gateway import agent_cli as agentcli

    agentcli.add_subparser(sub)

    # `sharedagents agents` — detect/quick-install/configure external coding agents.
    agentcli.add_agents_subparser(sub)

    # `sharedagents monitor` — watch the background local-model install.
    agentcli.add_monitor_subparser(sub)

    # `sharedagents orchestrator ...` — cross-agent live view of the shared
    # sessions.db. Single SQL predicate to see what every agent is
    # doing right now.
    agentcli.add_orchestrator_subparser(sub)

    # `sharedagents migrate-layout` — one-shot archive + delete of legacy
    # per-agent sessions/logs into the unified layout.
    agentcli.add_migrate_layout_subparser(sub)

    # `sharedagents gateway ...` — north gateway + south orchestrator lifecycle.
    # See ``sharedagents_gateway.gateway`` for the full architecture.
    from sharedagents_gateway import gateway_cli as gw_cli

    gw_cli.add_subparser(sub)

    # `sharedagents telegram ...` — standalone Telegram bot lifecycle.
    # Lives in its own module so the bot's CLI surface stays
    # decoupled from any one agent.
    from sharedagents_telegram import cli as tg_cli

    tg_cli.add_subparser(sub)

    # `sharedagents whatsapp ...` — linked-device WhatsApp self-chat channel
    # (link/status/run/logout/install-service). Its subcommands register
    # ``set_defaults(_dispatch=...)`` and run through the generic
    # ``_dispatch`` fallback below.
    from sharedagents_whatsapp import cli as wa_cli

    wa_cli.add_subparser(sub)

    # `sharedagents web ...` — localhost web frontend lifecycle (the browser UI
    # that renders the shared command runtime). Daemon + systemd user
    # service, mirroring `sharedagents gateway` / `sharedagents telegram`.
    from sharedagents_gateway import web_cli as web_cli_mod

    web_cli_mod.add_subparser(sub)

    pc = sub.add_parser("configure", help="Set or show persistent config")
    pc.add_argument("--endpoint", help="vLLM/OpenAI-compatible base URL (e.g. http://host:8000/v1)")
    pc.add_argument("--model", help="Model id to pin (required — no auto-discover)")
    pc.add_argument("--system", help="System prompt")
    pc.add_argument("--temperature", type=float, help="Sampling temperature")
    pc.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Persist whether to send chat_template_kwargs.enable_thinking=true (Qwen3 family)",
    )
    pc.add_argument("--show", action="store_true", help="Print effective config and exit")
    pc.add_argument("--agent", default="jarvis", help="Agent name for per-agent fields (default: jarvis)")
    pc.add_argument("--telegram-bot-token", help="Telegram BotFather token (per-agent)")
    pc.add_argument("--telegram-allowed-chat-ids", help="Comma-separated list of allowed Telegram chat_ids (per-agent)")
    pc.add_argument(
        "--telegram-allowed-user-ids",
        help=(
            "Comma-separated Telegram user ids the bot will reply to. "
            "When set, every other sender (even inside an allowed chat) is "
            "ignored silently. Pass an empty string to clear and revert to "
            "'any member of an allowed chat may interact'."
        ),
    )
    pc.add_argument(
        "--telegram-mention-required",
        choices=["on", "off"],
        help=(
            "Reply only when the bot is @-mentioned (or replied to) in "
            "non-private chats. DMs are unaffected. Default off; flip on "
            "for shared groups where the bot should stay quiet otherwise."
        ),
    )

    # `sharedagents doctor ...` — let the agent fix its own logged errors.
    from sharedagents_gateway import doctor as doctor_mod

    doctor_mod.add_subparser(sub)

    # `sharedagents session ...` — operate on saved agent sessions from the shell.
    # Same agent the TUI/`-p` flows use (selected via `-a/--agent`).
    psess = sub.add_parser("session", help="List or compact agent sessions")
    psess_inner = psess.add_subparsers(dest="op", required=True)
    psess_list = psess_inner.add_parser("list", help="List saved sessions for the selected agent")
    psess_list.add_argument(
        "-n",
        "--limit",
        type=int,
        default=20,
        help="Max sessions to print (default: 20, most recent first).",
    )
    psess_compact = psess_inner.add_parser(
        "compact",
        help="Trim oversized tool results in a saved session (between-turn /compact).",
    )
    psess_compact.add_argument(
        "sid",
        nargs="?",
        help="Session hash. Omit to compact the most recently modified session.",
    )
    psess_compact.add_argument(
        "--keep-recent-turns",
        type=int,
        default=1,
        help="Trailing turns to leave fully intact (default 1).",
    )
    psess_compact.add_argument(
        "--min-result-chars",
        type=int,
        default=800,
        help="Minimum tool_result length to consider for trimming (default 800).",
    )
    psess_compact.add_argument(
        "--head-chars",
        type=int,
        default=400,
        help="Chars kept from the head of each trimmed result (default 400).",
    )
    psess_compact.add_argument(
        "--json",
        action="store_true",
        help="Print the raw audit JSON instead of a human-readable summary.",
    )
    psess_show = psess_inner.add_parser("show", help="Print a session's header + turns (folded from the JSONL).")
    psess_show.add_argument("sid", help="Session hash.")
    psess_show_window = psess_show.add_mutually_exclusive_group()
    psess_show_window.add_argument(
        "--turn",
        type=int,
        help="Read one turn's exact context instead of the preview list.",
    )
    psess_show_window.add_argument(
        "--last",
        type=int,
        help="List only the newest N turns.",
    )
    psess_show_window.add_argument(
        "--from",
        dest="start",
        type=int,
        help="List turns starting at this index (see --count).",
    )
    psess_show.add_argument(
        "--count",
        type=int,
        help="With --from, how many turns to list (default: 50).",
    )
    psess_show.add_argument(
        "--role",
        choices=("user", "assistant", "both"),
        help="With --turn, select user, assistant, or both fields (default: both).",
    )
    psess_show.add_argument(
        "--offset",
        type=int,
        help="With --turn, zero-based character offset (default: 0).",
    )
    psess_show.add_argument(
        "--limit",
        type=int,
        help=f"With --turn, characters per role to print (1-{_SESSION_SHOW_MAX_LIMIT}; default: 12000).",
    )
    psess_show.add_argument(
        "--tool-calls",
        dest="tool_calls",
        action="store_true",
        help="Also print each turn's tool calls (name + truncated arguments).",
    )
    psess_inner.add_parser(
        "rebuild",
        help="Rebuild sessions.db (the search/header index) from the JSONL "
        "event logs. Use after a sessions.db loss/corruption.",
    )
    psess_delete = psess_inner.add_parser("delete", help="Delete a session: its <sid>.jsonl, lock, and index rows.")
    psess_delete.add_argument("sid", help="Session hash to delete.")

    # `sharedagents memory ...` — inspect / search / rebuild the SQLite memory index.
    pm = sub.add_parser(
        "memory",
        help="Inspect, search, rebuild, or prune the agent's memory index",
    )
    pm_inner = pm.add_subparsers(dest="op", required=True)
    pm_status = pm_inner.add_parser("status", help="Show index health (chunks, db size, scope, last indexed)")
    pm_status.add_argument("-a", "--agent", default="jarvis", help="Agent name (default 'jarvis')")
    pm_search = pm_inner.add_parser("search", help="Run a memory search and print results")
    pm_search.add_argument("-a", "--agent", default="jarvis")
    pm_search.add_argument("query", help="Search query")
    pm_search.add_argument("-k", "--top-k", type=int, default=6)
    pm_search.add_argument(
        "--min-score", type=float, default=0.35, help="Absolute fused-score floor (0-1, default 0.35)."
    )
    pm_search.add_argument(
        "--relevance-floor",
        type=float,
        default=0.0,
        help="Relative-to-best filter as a fraction of the top hit's score (0-1, default 0 = off).",
    )
    pm_search.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of formatted snippets"
    )
    pm_index = pm_inner.add_parser("index", help="Re-index memory files now")
    pm_index.add_argument("-a", "--agent", default="jarvis")
    pm_index.add_argument(
        "--force",
        action="store_true",
        help="Drop chunks/FTS/vec first, then re-index from scratch. "
        "Embedding cache is KEPT — unchanged content reuses cached vectors.",
    )
    pm_index.add_argument(
        "--reembed",
        action="store_true",
        help="Drop chunks/FTS/vec AND the embedding cache, then re-embed every "
        "chunk via the configured provider's API. Implies --force. Use this "
        "when switching embedding models or recovering from a poisoned cache.",
    )
    pm_prune = pm_inner.add_parser(
        "prune", help="Drop chunks for daily logs older than N days (MEMORY.md is always kept)"
    )
    pm_prune.add_argument("-a", "--agent", default="jarvis")
    pm_prune.add_argument(
        "--older-than-days",
        type=int,
        default=365,
        help="Delete daily-log chunks older than this many days (default 365)",
    )
    pm_synth = pm_inner.add_parser(
        "synth",
        help=(
            "Force a memory synthesis pass — read MEMORY.md + recent "
            "dailies, rewrite MEMORY.md via the configured upstream"
        ),
    )
    pm_synth.add_argument("-a", "--agent", default="jarvis")
    pm_synth.add_argument(
        "--force",
        action="store_true",
        help="Bypass the 'no daily newer than MEMORY.md' guard so an "
        "operator can re-run synth on demand without touching a "
        "daily file.",
    )
    pm_synth.add_argument(
        "--model",
        default=None,
        help="Override the model id. Defaults to memory_flush_model (if set) → upstream_model.",
    )
    pm_synth.add_argument(
        "--endpoint",
        default=None,
        help="Override the upstream endpoint URL. Defaults to agent.upstream_endpoint.",
    )
    pm_synth.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout for the synth completion (default 60s).",
    )
    pm_synth.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw stats dict as JSON instead of a formatted line.",
    )

    # `sharedagents skills ...` — discover / show / import / create / check AgentSkills-format skills.
    ps = sub.add_parser("skills", help="List, show, import, create, or check AgentSkills-format skills")
    ps_inner = ps.add_subparsers(dest="op", required=True)
    ps_list = ps_inner.add_parser("list", help="List discovered skills (across the search chain)")
    ps_list.add_argument("-v", "--verbose", action="store_true", help="Show source paths")
    ps_show = ps_inner.add_parser("show", help="Print one skill's SKILL.md body")
    ps_show.add_argument("name", help="Skill name (as in `sharedagents skills list`)")
    ps_imp = ps_inner.add_parser("import", help="Snapshot an OpenClaw skill into ~/.sharedagents/skills/")
    ps_imp.add_argument("name", help="OpenClaw skill name (under ~/.openclaw/skills/)")
    ps_create = ps_inner.add_parser("create", help="Scaffold a new skill directory with SKILL.md")
    ps_create.add_argument("name", help="Skill name (kebab-case)")
    ps_create.add_argument("--dir", default=None, help="Target parent directory (defaults to ~/.sharedagents/skills/)")
    ps_create.add_argument("--venv", action="store_true", help="Create an isolated .venv in the skill directory")
    ps_create.add_argument("--description", default="", help="Skill description (for trigger and frontmatter)")
    ps_check = ps_inner.add_parser("check", help="Lint and validate a skill definition")
    ps_check.add_argument(
        "name", nargs="?", default=None, help="Skill name or path to validate (validates all if omitted)"
    )

    args = parser.parse_args()

    if args.cmd == "configure":
        sys.exit(_cmd_configure(args))
    if args.cmd == "agent":
        from sharedagents_gateway import agent_cli as agentcli

        sys.exit(agentcli.dispatch(args))
    if args.cmd == "agents":
        sys.exit(args._dispatch(args))
    if args.cmd == "monitor":
        sys.exit(args._dispatch(args))
    if args.cmd == "orchestrator":
        # ``orchestrator {status,all,serve}`` wire ``set_defaults(_dispatch=…)``;
        # without this branch the call falls through to ``_launch_tui_via_agent``
        # and ``orchestrator serve`` (the systemd ExecStart) would launch the TUI
        # instead of running the SOUTH orchestrator service (§13).
        dispatch = getattr(args, "_dispatch", None)
        if dispatch is None:
            sys.exit(0)
        sys.exit(dispatch(args))
    if args.cmd == "gateway":
        from sharedagents_gateway import gateway_cli as gw_cli

        sys.exit(gw_cli.dispatch(args))
    if args.cmd == "session":
        sys.exit(asyncio.run(_cmd_session(args)))
    if args.cmd == "memory":
        sys.exit(_cmd_memory(args))
    if args.cmd == "skills":
        sys.exit(_cmd_skills(args))
    if args.cmd == "doctor":
        from sharedagents_gateway import doctor as doctor_mod

        sys.exit(doctor_mod.dispatch(args))
    if args.cmd == "telegram":
        # ``sharedagents_telegram.cli`` wires ``set_defaults(_dispatch=...)``
        # on each ``telegram <cmd>`` parser; without this branch the
        # call would fall through to ``_launch_tui_via_agent`` and
        # ``sharedagents telegram stop`` / ``restart`` (used heavily by
        # test_08 and test_10) would silently launch the TUI and
        # never exit, surfacing as ``subprocess.TimeoutExpired``.
        dispatch = getattr(args, "_dispatch", None)
        if dispatch is None:
            # Bare ``sharedagents telegram`` with no subcommand — exit 0.
            sys.exit(0)
        sys.exit(dispatch(args))
    if args.cmd == "web":
        from sharedagents_gateway import web_cli as web_cli_mod

        sys.exit(web_cli_mod.dispatch(args))
    # Subparsers that register a handler via ``set_defaults(_dispatch=...)`` (e.g.
    # ``migrate-layout``) are invoked here. Without this, argparse recorded the
    # handler but the explicit cmd-branch dispatcher above never called it, so the
    # subcommand silently no-op'd instead of running.
    _dispatch = getattr(args, "_dispatch", None)
    if _dispatch is not None:
        sys.exit(_dispatch(args))
    if args.prompt is not None:
        from sharedagents_agent.agents.engine import EngineKind

        bridge_kind = next(
            (
                kind
                for attr, kind in (
                    ("claude", EngineKind.CLAUDE_CLI),
                    ("codex", EngineKind.CODEX_CLI),
                    ("antigravity", EngineKind.ANTIGRAVITY_CLI),
                )
                if getattr(args, attr, False)
            ),
            None,
        )
        if bridge_kind is not None:
            sys.exit(asyncio.run(_run_external_oneshot(bridge_kind, args, args.prompt)))
        sys.exit(asyncio.run(_run_oneshot_via_agent(args, args.prompt)))
    sys.exit(_launch_tui_via_agent(args))


# --- de-monolith facade monkeypatch liveness -------------------------------
# app.py was split into helpers.py + app_mixins/* + widgets/*. Tests monkeypatch
# module-level names on THIS facade (``sharedagents_tui.app``); pre-split those reads
# were intra-module. Post-split a relocated reader (e.g. helpers._enable_kitty
# reading _KITTY_KEYBOARD_MODE/_IS_SSH, lifecycle mixin reading _gateway_url_and_token)
# holds its own copy, so a facade patch would miss it. Propagate any attribute
# ASSIGNED on this module to every ``sharedagents_tui.*`` submodule that already defines
# that name — restoring pre-split monkeypatch semantics. Fires ONLY on external
# setattr (module-body assignments ran before this class swap); production unaffected.
import types as _types  # noqa: E402


class _AppFacadeModule(_types.ModuleType):
    def __setattr__(self, name, value):
        _types.ModuleType.__setattr__(self, name, value)
        # NEVER propagate module dunders: importlib.reload() re-assigns
        # __name__/__spec__/__file__/__loader__/__class__ etc. on the facade;
        # propagating those would rewrite each sibling's identity (and __class__
        # would convert siblings into _AppFacadeModule), corrupting them.
        if name.startswith("__"):
            return
        for _mn, _m in list(sys.modules.items()):
            if _m is not None and _mn.startswith("sharedagents_tui.") and _mn != __name__ and hasattr(_m, name):
                try:
                    _types.ModuleType.__setattr__(_m, name, value)
                except Exception:
                    pass


sys.modules[__name__].__class__ = _AppFacadeModule


if __name__ == "__main__":
    main()
