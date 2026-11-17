"""ChatApp method bodies (extracted verbatim from app.py).

Extracted from app.py (de-monolith); behavior-preserving.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from rich.markdown import Markdown as RichMarkdown
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Header, Input, ListView, Static

from sharedagents_agent import context as ctxmod
from sharedagents_session.session import Session
from sharedagents_tui.helpers import (
    _SUBPROCESS_STDIO_LIMIT,
    _enable_kitty_keyboard_enhancements,
    _now,
)
from sharedagents_tui.widgets.input import (
    MessageInput,
)
from sharedagents_tui.widgets.messages import (
    AssistantMessage,
    HarnessTaskListMessage,
    PlanMessage,
    TaskUpdateMessage,
    ToolResultMessage,
    UserMessage,
)
from sharedagents_tui.widgets.notices import (
    CompactionNotice,
)
from sharedagents_tui.widgets.panels import (
    PlanChecklistFooter,
    SubagentBoxMessage,
)
from sharedagents_tui.widgets.scroll import (
    TranscriptScroll,
)


class ChatBaseMixin:
    """ChatApp method bodies (extracted verbatim from app.py)."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        system: str,
        temperature: float | None = None,
        enable_thinking: bool = True,
    ) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.model = model
        # Inject SOUL/AGENTS/TOOLS/MEMORY from .sharedagents/ when present.
        self.system_prompt = ctxmod.load_system_prompt(system)
        self.temperature: float | None = temperature
        self.enable_thinking = enable_thinking
        self.reasoning_visibility = "on" if enable_thinking else "off"
        # Tool-call header (🔧 toolname {args}) visibility. Default OFF
        # to keep the transcript clean; the collapsed tool-result widget
        # still shows. Toggle live via /tool[s] [on|off].
        self.tool_output_visible: bool = False
        self.session: Session | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._abort_requested = False
        self._turn_render_frozen = False
        # Two-stage ESC arming. Holds the turn-id for which a graceful
        # stop (first ESC / ``/stop``) has already been sent. A second
        # ESC while this still matches the active turn escalates to a
        # hard kill (``/abort``). Self-resetting: a new turn has a new
        # id, so the first ESC against it is graceful again.
        self._hardkill_armed_tid: str | None = None
        self._current_assistant: AssistantMessage | None = None
        # The round whose per-round `metrics` event we're still waiting
        # for. A round's metrics arrive AFTER the next round's bubble has
        # already mounted (RoundComplete is emitted post-tool-dispatch),
        # so when a tool_call / auto_continue closes a round we stash its
        # bubble here and stamp the next `metrics` event onto it — that's
        # what surfaces latency/token stats for EVERY round, including
        # rounds whose only output was tool calls. See
        # `_close_round_assistant` / the `event == "metrics"` handler.
        self._round_pending_assistant: AssistantMessage | None = None
        # AssistantMessage bubbles mounted during the CURRENT turn, in
        # order — used to reconcile each bubble's thinking title against
        # the canonical folded ``render_segments`` once the turn lands on
        # disk. Reset at every turn start. (turn-replication)
        self._turn_assistants: list = []
        self._current_user_text = ""
        # Tool-call IDs announced this round but not yet resolved. With
        # parallel tool calls the bridge announces N `tool_call` events
        # back-to-back, then streams N `tool_result` events. We must NOT
        # mount the next-round AssistantMessage between those results
        # (the bug: round-2 chunks then land before remaining tool
        # widgets, so the answer renders ABOVE its tool results). Only
        # ensure a fresh AssistantMessage once this set is empty —
        # i.e., the LAST parallel result has landed.
        self._pending_tool_calls: set[str] = set()
        # ExitPlanMode call_ids whose plan body was rendered as a
        # PlanMessage at tool_call time. The matching tool_result event
        # would otherwise mount a redundant ToolResultMessage carrying
        # the same plan wrapped in "Plan ready — confirm…" boilerplate;
        # tracking ids here lets the tool_result branch skip that mount
        # while still doing pending-set housekeeping.
        self._plan_call_ids: set[str] = set()
        # OutputSection call_ids whose section body was rendered as an
        # OutputSectionMessage at tool_call time. The matching
        # tool_result body is just the "Section X/N saved, next: Y"
        # nudge meant for the model — drop it on success so the
        # transcript reads as a clean run of section bubbles. On
        # FAILURE we still mount the standard ToolResultMessage so the
        # operator sees the validation error.
        self._section_call_ids: set[str] = set()
        # `UserQuestion` call_ids whose question + options were rendered
        # as a UserQuestionMessage card at tool_call time. The matching
        # tool_result body is the markdown echo the runner produced
        # ("❓ **User question dispatched**\n\n…") — redundant with the
        # card. Drop the standard Collapsible on success; on FAILURE
        # (malformed args, missing label) keep it so the operator sees
        # the validation error.
        self._user_question_call_ids: set[str] = set()
        # `Update`/`Edit` call_ids whose diff was rendered as an
        # UpdateMessage card at tool_call time (Claude Code's "Update"
        # treatment). The matching tool_result body is just a one-line
        # "updated <path>" summary — drop the redundant Collapsible on
        # success and just stamp the card ✓/✗ via `mark_result`. On
        # FAILURE we still mount the standard ToolResultMessage so the
        # operator sees the error (old_string-not-found, etc.).
        self._update_call_ids: set[str] = set()
        # Reference to the most recent ``TaskUpdateMessage`` card
        # mounted in the transcript. On subsequent TodoWrite tool
        # calls we update this card in-place rather than stacking
        # duplicate plan cards — keeps the transcript clean.
        self._latest_task_update_widget: TaskUpdateMessage | None = None
        # Same slot for the harness ``Task*`` family
        # (TaskCreate / TaskUpdate / TaskList / TaskGet / TaskOutput /
        # TaskStop) — one inline card per session, updated in place on
        # every subsequent Task* call so the transcript stays clean.
        self._latest_harness_task_widget: HarnessTaskListMessage | None = None
        # Persistent harness-task state for the ``Task*`` tool family
        # (TaskCreate / TaskUpdate / TaskList / TaskGet / TaskOutput /
        # TaskStop). Rendered into the ``#harness-tasks`` footer Static
        # (analogous to the ``#plan-checklist`` footer for TodoWrite)
        # so the operator always sees the current task list pinned at
        # the bottom of the TUI — no scrolling back to find the card,
        # no duplicate 🔧 TaskCreate pills cluttering the transcript.
        # ``_harness_tasks`` is id → {"subject", "status"}; ``_harness_tasks_order``
        # preserves insertion order for the footer render.
        self._harness_tasks: dict[str, dict] = {}
        self._harness_tasks_order: list[str] = []
        # Replay-side harness Task* card + task-group flag (reset per
        # replayed turn in `_render_one_saved_turn`). Separate from the
        # live `_latest_harness_task_widget` so replay rebuilding doesn't
        # disturb live state. (turn-replication: Task* parity)
        self._replay_harness_card: HarnessTaskListMessage | None = None
        self._replay_task_group_open: bool = False
        # call_id → args dict for in-flight ``TaskCreate`` calls. The
        # numeric task id only lands in the matching tool_result body
        # (e.g. ``"Task #4 created successfully: …"``), so we hold the
        # args here and apply them when the result arrives.
        self._pending_task_create_calls: dict[str, dict] = {}
        # call_ids that the live ``tool_call`` branch routed to the
        # harness-task footer. Used by the matching ``tool_result``
        # branch to suppress the redundant ToolResultMessage Collapsible
        # (the footer already shows the row).
        self._harness_task_call_ids: set[str] = set()
        # MISSION.md D1: "clear the task panel when a new task group is
        # created in the same TUI session." Set ``True`` at the start
        # of every top-level user-prompt turn (``_send_user_turn``);
        # consumed (and reset) on the first ``TaskCreate`` of that
        # turn. While set, the next ``TaskCreate`` clears the prior
        # group's rows from the persistent ``#harness-tasks`` footer
        # AND resets ``_latest_harness_task_widget`` so a *fresh*
        # inline card mounts for the new group (the prior card stays
        # in the transcript scrollback as historical context).
        #
        # Heuristic: first TaskCreate per user-prompt turn — picked
        # because (b) "taskId resets to 1" never fires in-session
        # (``run_task_create`` strictly increments ``next_id`` across
        # turns, see runners.py:3569-3570) and (c) "all prior done"
        # would have missed the operator's verified repro (30 pending
        # tasks → new prompt → 5 new tasks; the 30 were never marked
        # done). Mid-turn ``/steer`` injections do NOT toggle this
        # flag — they continue the in-flight turn, not start a new
        # group. ``TaskUpdate`` / ``TaskList`` / ``TaskGet`` / ``TaskOutput``
        # / ``TaskStop`` arriving before any ``TaskCreate`` also do
        # NOT consume the flag — only a ``TaskCreate`` signals the
        # start of a new group.
        self._new_task_group_pending: bool = False
        # Operator-driven focus cursor over mounted SubagentBoxMessage
        # widgets. Stepped by ``alt+up`` / ``alt+down`` (see
        # ``_step_subagent_focus``); ``ctrl+o`` opens the focused
        # one. -1 means "no subagent currently focused."
        self._subagent_focus_idx: int = -1
        # Subagent counter state — total dispatched / completed since
        # session start. Rendered into the ``#subagent-counter``
        # footer Static. Lets the operator see overall fan-out
        # progress at a glance (the per-box ✅ flip is easy to miss
        # when boxes scroll out of the viewport).
        self._subagent_total: int = 0
        self._subagent_done: int = 0
        self._subagent_failed: int = 0
        # call_ids of INNER subagent tool calls whose widget mount we
        # suppressed (parent transcript stays clean per the
        # subagent-stream-offload rule). The matching tool_result
        # branch checks this set to skip its widget mount AND to bump
        # the SubagentBoxMessage's done-count, then discards.
        self._suppressed_subagent_calls: set[str] = set()
        # Stack of in-flight subagent containers. Each delegate_* tool
        # call pushes a SubagentBoxMessage; the matching tool_result
        # pops it. While a container is on the stack, inner widgets
        # (assistant deltas + nested tool calls) mount INSIDE the box
        # instead of into the parent transcript — visual containment.
        self._subagent_box_stack: list[SubagentBoxMessage] = []
        # call_id → SubagentBoxMessage so a tool_result can find and
        # finalize the matching box even if other delegates have been
        # pushed/popped above it on the stack.
        self._subagent_boxes_by_call: dict[str, SubagentBoxMessage] = {}
        # Plan checklist. Single plan per session, mirroring Claude
        # Code's TodoWrite semantic: one logical list whose statuses
        # advance over time — re-emits replace/merge into the
        # existing plan rather than stacking new ones. Stored as a
        # length-0-or-1 list (``[{"items": [{content, activeForm,
        # status}, ...]}]``) so the renderer + replay scaffolding
        # don't have to special-case "no plan." Cleared on /new.
        self._plans: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield TranscriptScroll(id="transcript")
        # Pending-turn queue. Hidden via `display: none` while empty;
        # `_refresh_queue_display` toggles it visible and populates rows
        # whenever `_turn_queue` changes. Rows are clickable / Enter-able:
        # selecting one pulls the text back into the input for editing.
        yield ListView(id="queue")
        # Slash-command autocomplete popup. Empty (and CSS-hidden) by
        # default; populates with matching ``/<command> — description``
        # rows whenever the input starts with ``/`` and contains no
        # space. Tab in the input completes to the longest common
        # prefix. See ``MessageInput._refresh_slash_popup``.
        yield Static("", id="slash-popup", markup=True)
        with Horizontal(id="input-row"):
            yield Static(">", id="input-prompt")
            yield MessageInput(placeholder="", id="input")
        # Plan checklist: persistent footer-area widget that lists
        # the multi-step todo from the most recent plan_pass.
        # Steps tick (☐ → ☑) as the model fires tool calls.
        # Wrapped in PlanChecklistFooter (Vertical with header +
        # VerticalScroll body) so it gets collapse-by-default +
        # ▶/▼ chevron + 40%-of-screen-height cap + scroll bindings
        # — matches the HarnessTaskListMessage shape for #harness-
        # tasks. The wrapper composes the inner ``Static`` with the
        # historical id so every ``query_one("#plan-checklist",
        # Static)`` selector at :meth:`_render_plan_checklist` still
        # resolves; the wrapper itself answers to the
        # ``#plan-checklist-wrap`` selector for
        # :meth:`notify_plan_changed` callbacks.
        yield PlanChecklistFooter()
        # Harness-task list footer. Persistent, pinned at the bottom
        # (above plan-checklist/status), populated from TaskCreate /
        # TaskUpdate tool calls. The footer reuses the inline-card
        # widget class so it gets the same collapse-by-default +
        # 40%-height-cap + scroll machinery as the inline card (the
        # footer used to be a flat
        # ``Static`` that grew unbounded — operators saw it eat ~40%
        # of the screen with 30 rows). ``hide_when_empty=True`` keeps
        # an idle session free of chrome — the widget toggles its
        # own ``display`` based on whether the task list has rows.
        yield HarnessTaskListMessage(
            id="harness-tasks",
            hide_when_empty=True,
        )
        # Subagent counter footer. Persistent, pinned at the bottom.
        # Bumps as each SubagentBoxMessage mounts (total) and finalizes
        # (done). Lets the operator see at a glance "3/8 subagents
        # complete" without having to scroll back through finalized
        # boxes that already left the viewport.
        yield Static("", id="subagent-counter", markup=True)
        # Single-row transient-event line above the footer — hosts
        # the one-shot ack messages (``session created``, ``/restart
        # scheduled``, ``command failed: …``) that used to clutter
        # the footer's state segment. Empty by default; cleared
        # whenever the operator types or a turn starts so stale
        # acks don't linger.
        yield Static("", id="status-events", markup=True)
        # Dedicated teacher/goal supervision area — kept OUT of the footer.
        # The goal is word-wrapped + scrollable (long goals stay bounded), and
        # the teacher indicator gets its OWN row below it. Both collapse to 0
        # height when there is no active goal.
        with VerticalScroll(id="goal-scroll"):
            yield Static("", id="goal-text", markup=True)
        yield Static("", id="teacher-row", markup=True)
        # Single consolidated footer (state · agent · mode · ctx ·
        # model · upstream-status). Identity + persistent
        # operational state only — transients live above.
        yield Static("", id="status-footer", markup=True)

    async def on_mount(self) -> None:
        # Textual dispatches on_mount on every class in the MRO that defines
        # one — child first, then parent. AgentChatApp does its own setup and
        # then this parent handler would otherwise stomp the status bar with
        # the legacy `endpoint=… model=… thinking=…` line. Gate it.
        if getattr(self, "_skip_legacy_setup", False):
            return
        _enable_kitty_keyboard_enhancements(self)
        self._start_new_session()
        think = "on" if self.enable_thinking else "off"
        self._set_status(f"endpoint={self.endpoint}  model={self.model}  thinking={think}")
        self.query_one(MessageInput).focus()
        # The TUI no longer auto-checkpoints memory on a timer. The
        # agent runtime persists every turn end-to-end already (session
        # JSON is the source of truth) and the OpenClaw heartbeat
        # scheduler running in the gateway covers the "proactive
        # memory maintenance" role this timer used to gesture at. See
        # `sharedagents.heartbeat.HeartbeatScheduler`.

    def _start_new_session(self) -> None:
        self.session = Session.new(
            system_prompt=self.system_prompt,
            model=self.model,
            endpoint=self.endpoint,
        )
        self.session.save()
        self._refresh_title()

    def format_title(self, title: str, sub_title: str) -> Content:
        """Row 1: app name (+ hash); row 2: session title (wraps)."""
        title_content = Content(title)
        session_title = (sub_title or "").strip()
        if not session_title:
            return title_content
        return Content.assemble(
            title_content,
            ("\n", ""),
            Content(session_title).stylize("dim"),
        )

    def _refresh_title(self) -> None:
        s = self.session
        if not s:
            self.title = "sharedagents"
            self.sub_title = ""
            return
        # Show the FULL session id (was truncated to 12 chars) so the
        # operator can read/copy the whole hash from the header. The id
        # is ~34 chars and fits a header line; with mouse capture off it
        # is selectable like any other on-screen text.
        full = (s.hash or "").strip()
        self.title = f"sharedagents  [{full}]" if full else "sharedagents"
        self.sub_title = (s.title or "").strip()

    def _set_status(self, msg: str) -> None:
        # Transient acks land on the events row above the footer.
        # The footer is identity-only and must never carry transient
        # messages — if the events row isn't mounted yet (compose,
        # older harness), silently drop the ack rather than leak it
        # into the footer.
        try:
            self.query_one("#status-events", Static).update(msg or "")
        except Exception:  # noqa: BLE001
            pass

    def _sibling_agent_names(self) -> set[str]:
        """Names of sharedagents-native sibling agents available for the
        ``/<name> <task>`` auto-dispatch path. Loaded from the local
        ``AgentConfig.acp_agents`` at TUI construction (the gateway is
        co-located so we don't need an ACP round-trip). Returns an
        empty set on the legacy ``ChatApp`` (no agent_cfg attached).
        """
        return getattr(self, "_sibling_agents_cache", set())

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        # Clear transient model-rehydration toast after the next turn.
        self._upstream_warn = None
        self._model_mismatch_paid = False
        self._set_status("")
        if text in ("/new", "/abort"):
            # ARCHITECTURE LAW: a slash command's behavior is
            # computed once, in the shared runtime. ``/new`` returns
            # EFFECT_NEW_SESSION and ``/abort`` returns EFFECT_CANCEL (with the
            # server-side hard kill already issued and ``force`` set);
            # ``_cmd_shared`` applies the returned effect. The
            # legacy gateway-less ``ChatApp`` has no ``_cmd_shared`` — fall back
            # to the local control action there.
            is_new = text == "/new"
            run = getattr(self, "_cmd_shared", None)
            if run is not None:
                self.run_worker(run(text), exclusive=False, name="control")
            elif is_new:
                await self._cmd_new()
            else:
                abort = getattr(self, "action_abort", None)
                if abort is not None:
                    await abort(via="abort")
            return
        if text == "/stop":
            # Graceful: let the in-flight tool / round finish, then end the
            # turn. Same as the first ESC press. (/stop is a frontend-local
            # graceful alias — not a runtime command.)
            await self.action_abort(via="stop")
            return
        if text == "/steer" or text.startswith("/steer "):
            steer_text = text[len("/steer") :].strip()
            if not steer_text:
                transcript = self.query_one("#transcript", VerticalScroll)
                await transcript.mount(UserMessage("❯ /steer"))
                self._set_status("usage: /steer <text> — redirect the running turn")
                return
            enqueue = getattr(self, "_enqueue_or_run", None)
            if enqueue is None:
                self._set_status("/steer is only available via the agent runtime")
                return
            transcript = self.query_one("#transcript", VerticalScroll)
            await transcript.mount(UserMessage(f"❯ {text}"))
            can_inject = getattr(self, "_can_inject_mid_turn", None)
            inject = getattr(self, "_inject_user_message_via_acp", None)
            if callable(can_inject) and can_inject() and inject is not None:
                # Native local-LLM turn streaming → true mid-turn inject;
                # the engine drains it at the next round boundary.
                self.run_worker(
                    inject(steer_text),
                    exclusive=False,
                    name="steer-inject",
                )
            elif self._is_busy():
                # A delegated CLI-bridge turn (claude/codex/antigravity) is
                # running — it can't accept a live inject, so gracefully
                # abort it (SIGINT → 15s → kill, preserving --resume state)
                # and queue the steer text to pivot once it winds down.
                await self.action_abort(via="steer")
                enqueue("prompt", steer_text)
                self._set_status("↳ steering — winding down current turn, then pivoting")
            else:
                # No turn in flight → just a normal prompt.
                enqueue("prompt", steer_text)
            return
        if text == "/restart" or text.startswith("/restart "):
            hard = "--hard" in text.split()[1:]
            # Echo to transcript mirrors every other slash command
            # (/agents, /models, /agent, /claude…). Without this, plain
            # /restart with no children to sweep was a silent no-op —
            # input cleared, nothing rendered, operator could not tell
            # the command was accepted and would retype it.
            transcript = self.query_one("#transcript", VerticalScroll)
            await transcript.mount(UserMessage(f"❯ {text}"))
            confirm = getattr(self, "_confirm_irreversible", None)
            if callable(confirm) and not await confirm(text):
                return
            # ``_dispatch_restart`` is only defined on ``AgentChatApp``
            # (gateway-aware path). The base ``ChatApp`` is still
            # instantiated directly by ``_launch_tui`` (legacy worker-
            # spawn mode, no gateway), so guarding the lookup mirrors
            # the same getattr pattern /sessions, /refresh, /compact
            # already use here — otherwise ``/restart`` in legacy mode
            # would AttributeError right after the user sees the echo.
            run = getattr(self, "_dispatch_restart", None)
            if run is not None:
                run(hard=hard)
            else:
                self._set_status("/restart is only available via the agent runtime")
            return
        if text == "/mouse":
            self.action_toggle_mouse()
            return
        if text == "/help":
            # Always render the local rich-Markdown help card. The
            # ACP adapter's ``session_command`` returns ``{"ok": True}``
            # with no text body for /help (it's a frontend-local view,
            # not an agent-state mutation), so routing through
            # ``_cmd_shared`` here would print "(no output)" instead
            # of the help table.
            await self._cmd_help()
            return
        if (
            text == "/status"
            or text == "/mode"
            or text.startswith("/mode ")
            or text == "/thinking"
            or text.startswith("/thinking ")
            or text == "/reasoning"
            or text.startswith("/reasoning ")
            or text == "/reason"
            or text.startswith("/reason ")
            or text == "/effort"
            or text.startswith("/effort ")
            or text in ("/on", "/off", "/stream")
            or text == "/tool"
            or text.startswith("/tool ")
            or text == "/tools"
            or text.startswith("/tools ")
            or text == "/teacher"
            or text.startswith("/teacher ")
            or text == "/history"
            or text.startswith("/history ")
            or text == "/context"
            or text.startswith("/context ")
            or text == "/ctx"
            or text.startswith("/ctx ")
            or text == "/budget"
            or text.startswith("/budget ")
            or text == "/loop"
            or text.startswith("/loop ")
            or text == "/title"
            or text.startswith("/title ")
            or text == "/memory"
            or text.startswith("/memory ")
        ):
            run = getattr(self, "_cmd_shared", None)
            if run is not None:
                self.run_worker(run(text), exclusive=False, name="command")
            else:
                self._set_status(f"{text.split()[0]} is only available via the agent runtime")
            return
        if text == "/compact":
            # Server-side runtime decides compaction and returns a prompt
            # effect; _cmd_shared applies it (submits the compact_context turn).
            self.run_worker(self._cmd_shared("/compact"), exclusive=False, name="compact")
            return
        if text == "/sessions" or text.startswith("/sessions "):
            arg = text[len("/sessions") :].strip()
            run = getattr(self, "_cmd_sessions", None)
            if run is not None:
                self.run_worker(run(arg), exclusive=False, name="sessions")
            else:
                self._set_status("/sessions is only available via the agent runtime")
            return
        if text == "/refresh":
            # Live subscribe-tail is the happy path, but a poisoned byte
            # cursor (or a missed remote turn) still needs a durable
            # catch-up. Prefer the real refresh command when present.
            run = getattr(self, "_cmd_refresh", None)
            if run is not None:
                self.run_worker(run(), exclusive=False, name="refresh")
            elif getattr(self, "_use_event_stream", False):
                self._set_status("session renderer is live — turns stream in automatically")
            else:
                self._set_status("/refresh is only available via the agent runtime")
            return
        if text.startswith("/claude "):
            rest = text[len("/claude ") :].strip()
            if rest:
                # `_run_claude` is defined on AgentChatApp; the legacy
                # ChatApp doesn't have a Claude bridge but we still
                # parse the slash so the user gets a helpful error.
                if hasattr(self, "_enqueue_or_run") and hasattr(self, "_run_claude"):
                    self._enqueue_or_run("claude", rest)
                elif hasattr(self, "_run_claude"):
                    self.run_worker(self._run_claude(rest), exclusive=False, name="claude")
                else:
                    self._set_status("/claude is only available via the agent runtime")
            return
        if text.startswith("/codex "):
            rest = text[len("/codex ") :].strip()
            if rest:
                # Mirror /claude: queue if AgentChatApp, direct-run if
                # legacy ChatApp, "not available" otherwise.
                if hasattr(self, "_enqueue_or_run") and hasattr(self, "_run_codex"):
                    self._enqueue_or_run("codex", rest)
                elif hasattr(self, "_run_codex"):
                    self.run_worker(self._run_codex(rest), exclusive=False, name="codex")
                else:
                    self._set_status("/codex is only available via the agent runtime")
            return
        if text.startswith("/antigravity "):
            rest = text[len("/antigravity ") :].strip()
            if rest:
                # Same dispatch shape as /claude and /codex.
                if hasattr(self, "_enqueue_or_run") and hasattr(self, "_run_antigravity"):
                    self._enqueue_or_run("antigravity", rest)
                elif hasattr(self, "_run_antigravity"):
                    self.run_worker(self._run_antigravity(rest), exclusive=False, name="antigravity")
                else:
                    self._set_status("/antigravity is only available via the agent runtime")
            return
        if text.startswith("/zcode "):
            rest = text[len("/zcode ") :].strip()
            if rest:
                # Same dispatch shape as /claude, /codex and /antigravity.
                if hasattr(self, "_enqueue_or_run") and hasattr(self, "_run_zcode"):
                    self._enqueue_or_run("zcode", rest)
                elif hasattr(self, "_run_zcode"):
                    self.run_worker(self._run_zcode(rest), exclusive=False, name="zcode")
                else:
                    self._set_status("/zcode is only available via the agent runtime")
            return
        if text == "/models" or text.startswith("/models "):
            arg = text[len("/models") :].strip()
            run = getattr(self, "_cmd_models", None)
            if run is not None:
                self.run_worker(run(arg), exclusive=False, name="models")
            else:
                self._set_status("/models is only available via the agent runtime")
            return
        if text == "/agents":
            # Sibling-agent list comes from the server-side runtime.
            self.run_worker(self._cmd_shared("/agents"), exclusive=False, name="agents")
            return
        if text == "/agent" or text.startswith("/agent "):
            arg = text[len("/agent") :].strip()
            run = getattr(self, "_cmd_agent", None)
            if run is not None:
                self.run_worker(run(arg), exclusive=False, name="agent-switch")
            else:
                self._set_status("/agent is only available via the agent runtime")
            return
        if text == "/todo" or text == "/todos":
            run = getattr(self, "_cmd_shared", None)
            if run is not None:
                self.run_worker(run("/todo"), exclusive=False, name="todo")
            else:
                self._set_status("/todo is only available via the agent runtime")
            return
        if text == "/task" or text.startswith("/task "):
            # Harness task store is read/written server-side via the shared
            # runtime (writes land in <sid>.tasks.json the same as the
            # model's Task tools); no client-side task logic.
            self.run_worker(self._cmd_shared(text), exclusive=False, name="task")
            return
        if text == "/subagents" or text.startswith("/subagents "):
            arg = text[len("/subagents") :].strip()
            # Bare listing comes from the server-side runtime (no client-side
            # sidecar globbing). `/subagents <N>` (enter a subagent session)
            # stays local — it rebinds THIS TUI's view, a frontend nav action.
            if not arg:
                self.run_worker(self._cmd_shared("/subagents"), exclusive=False, name="subagents")
                return
            run = getattr(self, "_cmd_subagents", None)
            if run is not None:
                self.run_worker(run(arg), exclusive=False, name="subagents")
            else:
                self._set_status("/subagents is only available via the agent runtime")
            return
        if text == "/monitor":
            # Cross-agent agent monitor (webui ``/agents`` port) — a
            # TUI-local interactive modal, like the ``/sessions`` picker.
            # Distinct from ``/monitors`` (shells): this navigates EVERY
            # agent's sessions + subagents, not subprocesses.
            run = getattr(self, "_cmd_monitor", None)
            if run is not None:
                self.run_worker(run(), exclusive=False, name="agent-monitor")
            else:
                self._set_status("/monitor is only available via the agent runtime")
            return
        if text == "/monitors" or text == "/shells":
            # ``/shells`` is the discoverable alias — the modal navigates
            # every live subprocess the model spawned via ``monitor_*``
            # tools. Same dispatcher; just two trigger names so the
            # operator can find it under either mental model
            # ("monitors" matches the tool name, "shells" matches the
            # concept).
            run = getattr(self, "_cmd_monitors", None)
            if run is not None:
                self.run_worker(run(), exclusive=False, name="monitors")
            else:
                self._set_status(f"{text} is only available via the agent runtime")
            return
        if text.startswith("/monitors ") or text.startswith("/shells "):
            # Subcommands (``kill <id>``, ``tail``) go through the shared
            # runtime so TUI gets the same list/kill contract as Web UI,
            # including the local confirm on ``kill``.
            self.run_worker(self._cmd_shared(text), exclusive=False, name="monitors")
            return
        if text == "/queue" or text.startswith("/queue "):
            run = getattr(self, "_cmd_queue", None)
            if run is not None:
                self.run_worker(run(), exclusive=False, name="queue")
            else:
                self._set_status("/queue is only available via the agent runtime")
            return
        if text in ("/skills", "/skill") or text.startswith("/skill ") or text.startswith("/skills "):
            # Data (skill list / one skill body) comes from the server-side
            # runtime via ``_cmd_shared`` — no client-side disk gather.
            self.run_worker(self._cmd_shared(text), exclusive=False, name="skills")
            return
        if text in ("/plugin", "/plugins") or text.startswith("/plugin ") or text.startswith("/plugins "):
            self.run_worker(self._cmd_shared(text), exclusive=False, name="plugin")
            return
        if text in ("/mcp", "/mcps") or text.startswith("/mcp ") or text.startswith("/mcps "):
            self.run_worker(self._cmd_shared(text), exclusive=False, name="mcp")
            return
        if text == "/model" or text.startswith("/model "):
            arg = text[len("/model") :].strip()
            run = getattr(self, "_cmd_model", None)
            if run is not None:
                self.run_worker(run(arg), exclusive=False, name="model")
            else:
                self._set_status("/model is only available via the agent runtime")
            return
        # Auto-discovery for sibling agents — `/coder <task>`, `/scraper
        # <task>`, future `/<name>`. Distinct from `/claude` / `/codex`
        # / `/antigravity` (handled above) which are external-LLM proxies
        # with bespoke per-vendor handlers. sharedagents-native siblings all
        # share the uniform ``delegate_agent(name, task)`` interface,
        # so a single fallthrough wires them all without per-agent code.
        # Source of truth: ``cfg.acp_agents`` from the local AgentConfig
        # (gateway is co-located, no wire trip needed).
        if text.startswith("/") and " " in text:
            head, _, rest = text[1:].partition(" ")
            head = head.strip()
            rest = rest.strip()
            if head and rest and head in self._sibling_agent_names():
                if hasattr(self, "_enqueue_or_run"):
                    # Auto-fill the pre-flight checklist the delegation gate
                    # requires when this sibling resolves
                    # to a frontier engine. `/coder <task>` is a DIRECT HUMAN
                    # command — the operator IS the intent + authorization —
                    # so the FRONTEND fills it; the model can't. It is inert
                    # (extra keys ignored) for local siblings, which the gate
                    # never gates anyway.
                    from sharedagents_agent.operator_delegation import (
                        operator_slash_checklist,
                    )

                    payload = "__SHAREDAGENTS_DELEGATE__:delegate_agent:" + json.dumps(
                        {
                            "name": head,
                            "task": rest,
                            **operator_slash_checklist(rest),
                        }
                    )
                    self._enqueue_or_run("prompt", payload)
                    return
                self._set_status(f"/{head} requires the agent runtime")
                return
        # File-based slash commands (Claude Code `commands/*.md`, incl.
        # everything-claude-code). `/commands` lists them; `/<name>
        # [args]` expands the command's body (with $ARGUMENTS / $1…
        # substitution) into a normal user turn.
        if text == "/commands" or text.startswith("/commands "):
            # File-command list comes from the server-side runtime.
            self.run_worker(self._cmd_shared(text), exclusive=False, name="commands")
            return
        if text.startswith("/"):
            _head, _, _rest = text[1:].partition(" ")
            _head = _head.strip().lower()
            if _head:
                try:
                    from sharedagents_command.file_commands import (
                        expand_command,
                        find_command,
                    )

                    _fc = find_command(_head)
                except Exception:  # noqa: BLE001
                    _fc = None
                if _fc is not None:
                    # `/goal` — the engine gate that turns teacher supervision
                    # on. ONE goal per session at a time:
                    #   /goal <cond>     → set it (OVERRIDES any previous goal)
                    #   /goal clear      → stop the goal (teacher dormant)
                    #   /goal (or list)  → show the current goal
                    # Setting a goal sends the goal preamble.
                    if _head == "goal":
                        _grest = _rest.strip()
                        _gl = _grest.lower()

                        async def _set_goal_opt(_val: str) -> str:
                            if not self._sid:
                                self._set_status("goal: no active session")
                                return ""
                            try:
                                _acp = await self._client._ensure_connected()
                                await _acp.session_set_config_option(
                                    self._sid,
                                    "goal",
                                    _val,
                                )
                                _sess = await self._client.get_session(self._sid)
                                return str(
                                    (_sess or {}).get("active_goal") or "",
                                )
                            except Exception as _exc:  # noqa: BLE001
                                self._set_status(f"goal: {_exc}"[:200])
                                return ""

                        from sharedagents_goal.goal_service import (
                            GOAL_CONTROL_ACTIONS,
                            parse_goal_subcommand,
                        )

                        _action, _, _ = parse_goal_subcommand(_grest)

                        if _action in GOAL_CONTROL_ACTIONS:
                            # Route read/control actions through the shared
                            # command runtime so the TUI renders the SAME
                            # registry view (goals.db) as Telegram and web:
                            # `/goal list` (and bare `/goal`) shows EVERY
                            # goal, not just this session's. Mutations still
                            # land durably in the registry (via
                            # _run_goal_command). Refresh the local goal
                            # cache + teacher indicator afterwards.
                            _cmd_text = f"/{_head}"
                            if _grest:
                                _cmd_text += f" {_grest}"
                            await self._cmd_shared(_cmd_text)
                            try:
                                _sess = await self._client.get_session(self._sid)
                                self._active_goal = str(
                                    (_sess or {}).get("active_goal") or "",
                                )
                            except Exception:  # noqa: BLE001
                                pass
                            self._recompute_teacher()
                            return

                        if not _grest:
                            self._set_status(f"goal: {self._active_goal}" if self._active_goal else "no active goal")
                            return
                        await _set_goal_opt(_grest)
                        self._active_goal = _grest
                        self._recompute_teacher()
                    _expanded = expand_command(_fc, _rest.strip())
                    # Stash a compact echo for the transcript — the
                    # full markdown body of a builtin command (e.g.
                    # /goal's 49-line preamble) goes to the model
                    # but the operator just wants to see ``❯ /goal
                    # <args>`` in the scrollback. ``_send_user_turn``
                    # picks this up, mounts it as the UserMessage,
                    # and clears the override so the next free-form
                    # prompt echoes verbatim.
                    _compact_echo = f"/{_head}"
                    if _rest.strip():
                        _compact_echo += f" {_rest.strip()}"
                    self._next_user_echo = _compact_echo
                    if hasattr(self, "_enqueue_or_run"):
                        self._enqueue_or_run("prompt", _expanded)
                    else:
                        await self._send_user_turn(_expanded)
                    return
            # Unknown slash — same as Web UI: ask the shared runtime
            # (unknown_command / file-command miss) instead of sending
            # `/notacommand` to the model as a chat turn.
            run = getattr(self, "_cmd_shared", None)
            if run is not None:
                self.run_worker(run(text), exclusive=False, name="slash")
                return
            self._set_status(f"unknown command: {text.split()[0]}")
            return
        # Plain text → user turn. AgentChatApp queues if a turn is already
        # in flight (so a second Enter doesn't get dropped); legacy ChatApp
        # falls through to its direct-spawn busy-rejection path.
        if hasattr(self, "_enqueue_or_run"):
            self._enqueue_or_run("prompt", text)
        else:
            await self._send_user_turn(text)

    async def _cmd_new(self) -> None:
        if self._proc is not None:
            await self.action_abort(via="new")
            for _ in range(50):
                if self._proc is None:
                    break
                await asyncio.sleep(0.02)
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.remove_children()
        self._start_new_session()
        assert self.session is not None
        self._set_status(f"new session [{self.session.hash}]")

    async def _cmd_help(self) -> None:
        """Render the sharedagents help card inline. Available in both the
        legacy direct-spawn `ChatApp` and the agent-backed `AgentChatApp`
        — sections that depend on the agent runtime (`/claude`, `/compact`)
        are still listed because they're the most common reason new users
        need help; the legacy app surfaces a "not available" status when
        they're invoked there.
        """
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage("❯ /help"))
        # Each "table-like" section is wrapped in a fenced code block.
        # Without the fences, Rich/CommonMark treats the per-row newlines
        # as soft line breaks and collapses every command into one
        # continuous paragraph (the "messy /help" symptom) — the leading
        # two-space indent isn't enough by itself to be an indented code
        # block under CommonMark because the parent is the section
        # header, not a bullet/paragraph that the indent would belong to.
        # Code fences preserve both the layout AND the monospace font.
        body = (
            "**sharedagents — quick reference**\n\n"
            "**Slash commands**\n"
            "```text\n"
            "/help                 — this card\n"
            "/new                  — abort, clear transcript, start a new session\n"
            "/stop                 — graceful stop: finish current step, then end (1st ESC)\n"
            "/abort                — hard kill now: worker + tools + subagents (2nd ESC)\n"
            "/steer <text>         — redirect the running turn: mid-turn inject for a\n"
            "                        local turn, or graceful-pivot a CLI-bridge turn\n"
            "/restart              — sweep every in-flight subprocess (tool_runners,\n"
            "                        subagents) under the gateway. Add --hard to\n"
            "                        also bounce the gateway parent (~3s reconnect).\n"
            "/mouse                — toggle mouse tracking (off = terminal text\n"
            "                        selection works; on = clickable UI)\n"
            "/compact              — squeeze oversized tool results from the saved\n"
            "                        session so the next turn's prompt is shorter\n"
            "/todo                 — show this session's todo checklist (the model\n"
            "                        writes it via the todo tool)\n"
            "/skills               — list installed skills (name + description)\n"
            "/skill <name>         — show one skill's full SKILL.md body\n"
            "/commands             — list file-based slash commands (commands/*.md)\n"
            "/<command> [args]     — run a file command (e.g. /refactor-clean)\n"
            "/subagents [N]        — list / enter delegated subagent sessions\n"
            "/monitors, /shells    — live panel of background shells the model\n"
            "                        spawned via the monitor_* tools\n"
            "/sessions             — list saved sessions (most recent first)\n"
            "/sessions <i|hash>    — switch to that session (index or hash prefix)\n"
            "/models               — list configured models (interactive picker)\n"
            "/model <alias|url>    — switch the live session to that model/upstream\n"
            "/mode [id]            — show or set this agent's advertised mode\n"
            "/effort [level]       — model-specific reasoning depth (or 'default')\n"
            "/status               — agent + endpoint + backend snapshot\n"
            "/history              — dump this session's transcript\n"
            "/context [N]          — token map filling the context window\n"
            "/task [add|in|done|del|list] — operator harness-task store\n"
            "/budget [usd]         — show or set this session's budget cap\n"
            "/queue                — list pending mid-turn / post-turn prompts\n"
            "/agents               — list configured agents (current marked)\n"
            "/agent <name> [mode]  — switch this conversation's agent\n"
            "/monitor              — cross-agent session monitor (Web /agents)\n"
            "/goal [text|list|clear] — set, list, or clear the session goal\n"
            "/loop /title /memory  — scheduled loops, session title, memory index\n"
            "/reasoning [on|off|stream] — show/stream/hide thinking tokens (default on)\n"
            "/on, /off, /stream    — shortcuts for /reasoning <level>\n"
            "/tools [on|off]       — toggle inline 🔧 tool-call header (default off)\n"
            "/teacher [on|off] | model <alias> — toggle teacher supervision / pick supervising model\n"
            "/refresh              — pull any new turns appended by another\n"
            "                        frontend (Telegram, API) into the transcript\n"
            "/claude <prompt>      — bridge the prompt to Claude Code (streams inline,\n"
            "                        chains across calls via --resume, includes the\n"
            "                        sharedagents conversation as context)\n"
            "/codex <prompt>       — same idea, bridges to OpenAI's Codex CLI\n"
            "                        (chains via 'codex exec resume', mirrors /claude)\n"
            "/antigravity <prompt>      — same idea, bridges to Google Antigravity CLI\n"
            "                        (chains via 'antigravity --resume', mirrors /claude)\n"
            "```\n\n"
            "**Key bindings**\n"
            "```text\n"
            "ESC                   — abort current operation (LLM round, tool,\n"
            "                        /claude, /codex, /antigravity)\n"
            "Ctrl+R                — toggle the most recent reasoning pane\n"
            "Ctrl+T                — toggle the most recent tool result / plan /\n"
            "                        compaction notice (expand/collapse)\n"
            "Ctrl+/                — toggle mouse tracking (off = terminal text\n"
            "                        selection works; on = clickable collapsibles)\n"
            "Shift+Tab             — cycle this agent's advertised modes\n"
            "Enter                 — send (in tmux/SSH, Enter/LF also sends)\n"
            "Ctrl+N                — insert a newline (Shift+Enter on kitty)\n"
            "Up / Down             — cycle through command history (like bash)\n"
            "Ctrl+Up / Ctrl+Down   — hop into the pending-prompt queue (focus +\n"
            "                        navigate). Inside the queue: Enter pulls the\n"
            "                        row back into the input for editing, Delete\n"
            "                        drops it, Shift+Up/Down reorders, Escape\n"
            "                        returns focus to the input.\n"
            "```\n\n"
            "**Mid-turn steering**\n\n"
            "Submit a fresh prompt while the agent is mid-plan and the new\n"
            "message lands at the next round boundary (no waiting for the\n"
            "current turn to end). The transcript marks it with\n"
            "`↳ injected mid-turn`. Queued entries for `/claude` / `/codex` /\n"
            "`/antigravity` bridges still wait for end-of-turn — those engines\n"
            "don't support mid-flight steering.\n\n"
            "**Status bar**\n"
            "```text\n"
            "agent · mode · model · ctx <used>/<max> · hb <age>s · uptime\n"
            "```\n"
            "`ctx` warns you before the runtime auto-compacts; `/compact` (or the\n"
            "model's `compact_context` tool call) trims it manually.\n\n"
            "**Permission modes**\n"
            "```text\n"
            "Native agents advertise their own catalogue (jarvis/plan/build:\n"
            "Allow all + Ask). External engines use their CLI catalogue\n"
            "(OpenCode Build|Plan, Claude/Codex/Cursor/… their own ids).\n"
            "Shift+Tab and /mode cycle only modes that agent actually\n"
            "advertises — there is no global plan↔auto pair anymore.\n"
            "```\n\n"
            "**CLI**\n"
            "```text\n"
            "Configure agents:    sharedagents agent {init,list,start,stop,restart,status,logs}\n"
            'One-shot:            sharedagents -p "..."\n'
            'Claude bridge:       sharedagents --claude --plan -p "..."\n'
            'Codex bridge:        sharedagents --codex --plan -p "..."\n'
            'Antigravity bridge:  sharedagents --antigravity --plan -p "..."\n'
            "```\n"
        )
        await transcript.mount(Static(RichMarkdown(body), classes="claude-msg"))
        self._maybe_scroll_end(transcript)

    def action_toggle_thinking(self) -> None:
        """Expand/collapse the most recent assistant turn's reasoning pane.

        Walks the transcript in reverse so the binding always lands on the
        currently-streaming turn first; falls back to the last finished one.
        Bound to `Ctrl+R`.
        """
        messages = list(self.query(AssistantMessage))
        for msg in reversed(messages):
            if msg._thinking_box is not None:
                msg._thinking_box.collapsed = not msg._thinking_box.collapsed
                # Refresh the title so the "Ctrl+R to expand/collapse" hint
                # flips immediately rather than waiting for the next tick.
                msg._refresh_think_title(active=msg._think_ended is None)
                state = "expanded" if not msg._thinking_box.collapsed else "collapsed"
                self._set_status(f"reasoning {state}")
                return
        self._set_status("no reasoning to toggle yet")

    def action_toggle_follow(self) -> None:
        """Toggle live-follow (``tail -f``) of the session transcript.

        Bound to ``Ctrl+F``. The transcript already auto-follows the live
        event stream when ``_scroll_follow`` is set: every passive delta
        routes through :meth:`_maybe_scroll_end`, which only scrolls while
        the flag is on, and :meth:`_update_scroll_follow` flips the flag
        off/on as the operator scrolls away from / back to the bottom.

        This action is the explicit, one-keystroke control over that same
        flag (no separate state — single source of truth):

        * Turning follow ON snaps to the live tail immediately and re-arms,
          so subsequent streamed events keep the view pinned to the bottom.
        * Turning follow OFF freezes the current scroll position so new
          events accumulate below without moving the viewport — letting the
          operator read back mid-turn.

        Frontend carries no business logic here: the event data is still
        the durable backend log tailed via ``subscribe_events``; this only
        governs *where the viewport sits* over that shared surface.
        """
        # Don't fight a modal on top (Shells, Kanban, pickers manage their
        # own scroll); the binding is for the main transcript view.
        if isinstance(self.screen, ModalScreen):
            return
        self._scroll_follow = not self._scroll_follow
        if self._scroll_follow:
            # Best-effort snap to the live tail, but report truthfully:
            # only claim "tailing live" once the snap actually succeeded.
            # If the transcript lookup / scroll fails the flag is still
            # armed (passive deltas will follow on the next render), so
            # report the softer "on — re-armed" rather than asserting a
            # snap that didn't happen.
            snapped = False
            try:
                transcript = self.query_one("#transcript", VerticalScroll)
                transcript.scroll_end(animate=False)
                snapped = True
            except Exception:  # noqa: BLE001
                snapped = False
            if snapped:
                self._set_status("follow: on — tailing live")
            else:
                self._set_status("follow: on — re-armed")
        else:
            self._set_status("follow: off — scroll frozen")

    def action_toggle_tool_output(self) -> None:
        """Expand/collapse the most recent toggleable transcript card. Bound to `Ctrl+T`.

        Symmetric to `action_toggle_thinking` — finds the most-recently
        mounted toggleable card (`ToolResultMessage`, `PlanMessage`, or
        `CompactionNotice`) and flips its collapsed state. Tool results
        default to collapsed (so a 64 KiB grep doesn't dominate the
        view); plans and compaction notices default to expanded (so the
        operator sees them); this binding flips whichever is most recent
        so the operator can clear space once a card has been read.
        """
        # Bail when a modal is on top: otherwise Ctrl+T silently flips
        # a transcript card hidden behind the modal — broadening from
        # tool-results-only to plans/compaction notices (both visually
        # loud and default-expanded) made that surprise easy to hit.
        if isinstance(self.screen, ModalScreen):
            return
        # Walk transcript children only — `self.query("*")` would also
        # traverse the input row, footer, and every modal screen, and
        # pay the CSS-selector match cost on every node before the
        # `isinstance` filter runs. `walk_children(reverse=True)` is
        # recursive (so a PlanMessage nested inside a SubagentBoxMessage
        # body still counts) and short-circuits on the first match.
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
        except Exception:  # noqa: BLE001
            self._set_status("nothing to toggle yet")
            return
        targets = (ToolResultMessage, PlanMessage, CompactionNotice)
        last = None
        for w in transcript.walk_children(with_self=False, reverse=True):
            if isinstance(w, targets):
                last = w
                break
        if last is None:
            self._set_status("nothing to toggle yet")
            return
        collapsed = last.toggle()
        label = {
            ToolResultMessage: "tool output",
            PlanMessage: "plan",
            CompactionNotice: "compaction notice",
        }.get(type(last), "card")
        self._set_status(f"{label} {'collapsed' if collapsed else 'expanded'}")

    def action_toggle_mouse(self) -> None:
        """Toggle mouse tracking on/off. Bound to Ctrl+/.

        Off (default): terminal handles native text selection (shift+drag).
        On: TUI captures mouse for clickable collapsibles, but you can't
        select/copy text with the mouse.
        """
        self.mouse_support = not self.mouse_support
        self._set_status(
            f"mouse {'enabled' if self.mouse_support else 'disabled'} (terminal selection {'blocked' if self.mouse_support else 'available'})"
        )

    async def action_request_quit(self) -> None:
        """Two-press Ctrl+C with first-press warning.

        First press → warn + arm a 3-second window.
        Second press within the window → flush memory and exit.
        """
        now = time.monotonic()
        last = getattr(self, "_quit_armed_at", None)
        if last is not None and (now - last) < 3.0:
            try:
                await self._on_exit_flush()
            except Exception:  # noqa: BLE001
                pass
            self.exit()
            return
        self._quit_armed_at = now
        self._set_status("press Ctrl+C again within 3s to quit (memory will be saved)")

    async def _on_exit_flush(self) -> None:
        """Hook for subclasses to persist state before quit. No-op here."""
        return None

    async def _send_user_turn(self, text: str) -> None:
        if self._proc is not None:
            self._set_status("busy — press ESC or /abort to cancel current turn")
            return

        assert self.session is not None
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage(f"❯ {text}"))
        assistant = AssistantMessage()
        await transcript.mount(assistant)
        self._maybe_scroll_end(transcript)

        self._current_assistant = assistant
        self._current_user_text = text

        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        for turn in self.session.turns:
            messages.append({"role": "user", "content": turn["user"]})
            content = turn.get("content") or ""
            if content:
                messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": text})

        request = {
            "endpoint": self.endpoint,
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "extra": {
                # vLLM's OpenAI-compat endpoint forwards this to the model's
                # chat template. Qwen3 family gates the <think> phase on it.
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            },
        }
        # Per-row upstream headers (OpenRouter Bearer + X-Title, etc.)
        # The TUI's legacy worker-spawn path bypasses the bridge, so
        # the same headers the bridge attaches in
        # ``_build_worker_request`` need to be plumbed in here too —
        # without this, switching to a model row with ``apiKeyEnv``
        # set (kimi → OpenRouter) hits HTTP 401 because the Bearer
        # token never reaches the worker. Local vLLM rows resolve to
        # an empty dict and are unaffected.
        try:
            upstream_hdrs = self.agent_cfg.upstream_headers(self.endpoint)
        except Exception:  # noqa: BLE001
            upstream_hdrs = {}
        if upstream_hdrs:
            request["headers"] = dict(upstream_hdrs)

        self.run_worker(self._stream_worker(request), exclusive=True, name="stream")

    async def _stream_worker(self, request: dict) -> None:
        assert self.session is not None
        assistant = self._current_assistant
        assert assistant is not None

        started = _now()
        self._abort_requested = False

        package_parent = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ)
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
            limit=_SUBPROCESS_STDIO_LIMIT,
        )
        self._proc = proc
        self._set_status("[dim]ESC to cancel[/dim]")

        assert proc.stdin is not None
        proc.stdin.write(json.dumps(request).encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        worker_aborted = False
        worker_error: str | None = None
        worker_metrics: dict | None = None
        transcript = self.query_one("#transcript", VerticalScroll)

        try:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    event = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                kind = event.get("kind")
                if kind == "thinking":
                    await assistant.append_thinking(event.get("text", ""))
                elif kind == "content":
                    assistant.append_content(event.get("text", ""))
                elif kind == "aborted":
                    worker_aborted = True
                elif kind == "error":
                    worker_error = event.get("message") or "unknown error"
                elif kind == "metrics":
                    worker_metrics = {k: v for k, v in event.items() if k != "kind"}
                elif kind == "done":
                    pass
                self._maybe_scroll_end(transcript)
        finally:
            await proc.wait()
            stderr_bytes = b""
            if proc.stderr is not None:
                try:
                    stderr_bytes = await proc.stderr.read()
                except Exception:  # noqa: BLE001
                    stderr_bytes = b""
            self._proc = None

        ended = _now()
        aborted = worker_aborted or self._abort_requested
        rc = proc.returncode or 0

        if worker_error:
            assistant.mark_error(worker_error)
            self._set_status(f"error: {worker_error[:80]}")
        elif aborted:
            assistant.mark_aborted()
            self._set_status("aborted")
        elif rc != 0:
            stderr_txt = stderr_bytes.decode("utf-8", "replace")[:300]
            msg = f"worker exited {rc}: {stderr_txt}".strip()
            assistant.mark_error(msg)
            self._set_status(f"error: rc={rc}")
        else:
            if worker_metrics:
                assistant.set_metrics(worker_metrics)
            assistant.finalize()
            # Footer no longer carries per-call stats — they live on the
            # assistant message itself. Clear any prior transient.
            self._set_status("")

        self.session.add_turn(
            user=self._current_user_text,
            thinking=assistant.thinking_text,
            content=assistant.content_text,
            started_at=started,
            ended_at=ended,
            aborted=aborted,
            error=worker_error,
            metrics=worker_metrics,
        )
        self.session.save()
        self._refresh_title()
