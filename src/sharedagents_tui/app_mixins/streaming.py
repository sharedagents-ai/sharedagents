"""AgentChatApp delegate runners + event render + stream loop.

Extracted from app.py (de-monolith); behavior-preserving.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from sharedagents_agent.agents.engine import EngineKind
from sharedagents_agent.external_registry import spec_for
from sharedagents_agent.tools.widget_adapt import (
    adapt_todo_args,
    is_todo_tool_name as _is_todo_tool_name,
)
from sharedagents_session.session import Session
from sharedagents_tui.helpers import (
    _RUNTIME_WARNING_HEADLINES,
    _build_delegate_metrics,
    _build_metrics_markup,
    _parse_runtime_warning_chunk,
    _parse_task_list_text,
    humanise_delegate_marker,
    user_prompt_text,
)
from sharedagents_tui.widgets.messages import (
    AntigravityMessage,
    AssistantMessage,
    ClaudeMessage,
    CodexMessage,
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
    ZCodeMessage,
)
from sharedagents_tui.widgets.notices import (
    CompactionNotice,
    EmptyEosNotice,
    RuntimeWarningNotice,
)
from sharedagents_tui.widgets.panels import (
    SubagentBoxMessage,
)


class StreamMixin:
    """AgentChatApp delegate runners + event render + stream loop."""

    def _tui_log(self, event: str, **fields) -> None:
        """Append one JSON line to the unified ``logs/tui.log``.

        Captures key TUI-side moments (ESC pressed, action_abort entry,
        turn_stop success/fail) so a post-mortem can prove whether a
        keystroke ever reached the TUI and whether the cancel hit the
        wire — gateway.log only sees the server side. Best-effort: any
        I/O failure here MUST NEVER bubble into the input handler.

        Routed through :mod:`sharedagents_core.jsonlog` so the line carries
        explicit origin tags (agent, service, session_id) and shares
        the cross-process flock with every other tui-log writer
        running against the same shared file.
        """
        try:
            from sharedagents_core.jsonlog import get_writer
            from sharedagents_core.layout import tui_logfile

            agent = getattr(self.agent_cfg, "name", None) or getattr(self, "_agent_name", None) or "tui"
            sid_full = self._sid or ""
            w = get_writer(tui_logfile(), service="tui")
            w.write(
                agent=str(agent),
                level="INFO",
                msg=event,
                kind=event.upper(),
                session_id=sid_full or None,
                k={k: v for k, v in fields.items() if v is not None and v != ""} or None,
            )
        except Exception:  # noqa: BLE001
            pass

    def _instant_abort_ui(self) -> None:
        """Hard-abort instant feedback — before the gateway cancel ack."""
        self._turn_render_frozen = True
        self._abort_requested = True
        if self._current_assistant is not None and not getattr(self._current_assistant, "_final", False):
            self._current_assistant.mark_aborted()
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
            for w in transcript.query(ToolResultMessage):
                if getattr(w, "_streaming", False):
                    w.finalize_streaming(ok=False, content=None)
        except Exception:  # noqa: BLE001
            pass
        self._set_worker_state("")

    async def action_abort(self, via: str = "esc") -> None:
        # Modal-aware Esc. The app-level ``Binding("escape", "abort",
        # priority=True)`` fires BEFORE the focused widget / active
        # screen sees the keystroke (Textual 8.x: app-level priority
        # bindings preempt modal-screen bindings AND modal-screen
        # ``on_key``). So when a modal is on top — ``/models`` picker,
        # ``/sessions`` picker, MonitorPanel, ``QuestionsModal``, … —
        # an Esc that the operator means as "close this dialog" was
        # firing turn-abort instead, surfacing "nothing to abort — no
        # turn in flight" while the modal stayed up. Detect that case
        # and dismiss the modal here. Only intercept when ``via="esc"``
        # — typed ``/abort`` and the teardown paths (``new``,
        # ``agent_switch``, ``session_switch``) must run the abort
        # logic unconditionally regardless of what's on top.
        if via == "esc" and isinstance(self.screen, ModalScreen):
            try:
                self.screen.dismiss(None)
            except Exception:  # noqa: BLE001
                # Defensive — a dismiss failure must NEVER swallow
                # the abort fallback. Fall through to the normal
                # abort path so Esc keeps working even if the modal
                # is in a weird state.
                pass
            else:
                return
        # Decide graceful vs hard. ``via`` carries the operator intent:
        #   * "stop"  ← typed ``/stop``  → graceful (let the in-flight
        #               tool / round finish, then end the turn; kill
        #               nothing).
        #   * "abort" ← typed ``/abort`` → hard kill now (SIGKILL +
        #               killpg the worker AND every runner / subagent).
        #   * "esc"   ← ESC key          → first press graceful; a
        #               second press on the SAME live turn escalates to
        #               a hard kill (the headline ESC-once / ESC-twice
        #               spec).
        #   * "new" / "agent_switch" / "session_switch" ← teardown paths
        #               that abandon the turn → hard.
        #   * anything else (e.g. "steer") → graceful.
        active_tid = self._claude_turn_id or self._codex_turn_id or self._antigravity_turn_id or self._current_turn_id
        _HARD_VIAS = {"abort", "new", "agent_switch", "session_switch"}
        if via in _HARD_VIAS:
            force = True
        elif via == "esc":
            force = bool(active_tid is not None and self._hardkill_armed_tid == active_tid)
        else:
            force = False

        # Instant feedback (fix C). Status BEFORE any await so the
        # operator sees the press land the same tick — without this a
        # slow gateway cancel ack was indistinguishable from "Textual
        # ate the keystroke" (2026-05-12 deepseek-v4-flash post-mortem).
        # ``mode`` is logged so a post-mortem can tell graceful from hard.
        pressed_at = time.strftime("%H:%M:%S")
        mode = "hard" if force else "graceful"
        self._set_status(f"{'KILL' if force else 'stop'} received {pressed_at} ({via}) — sending cancel…")
        self._tui_log(
            "abort_requested",
            via=via,
            mode=mode,
            ts_local=pressed_at,
            claude_tid=self._claude_turn_id,
            codex_tid=self._codex_turn_id,
            antigravity_tid=self._antigravity_turn_id,
            current_tid=self._current_turn_id,
            queued=len(self._turn_queue),
        )
        if force:
            self._instant_abort_ui()
        # ESC means "stop what I have going" — drop any pending prompts
        # in the queue too, so a long queued backlog doesn't keep firing
        # after the user explicitly bailed. The queued entries weren't
        # mounted to the transcript yet, so dropping them is silent.
        n_dropped = len(self._turn_queue)
        if n_dropped:
            self._turn_queue.clear()
            self._refresh_status_bar()
            self._refresh_queue_display()

        async def _send_cancel(tid: str, label: str) -> None:
            sent_at = time.strftime("%H:%M:%S")
            err: str | None = None
            try:
                await self._client.turn_stop(tid, force=force)
            except Exception as e:  # noqa: BLE001
                err = str(e) or repr(e)
            suffix = f" + dropped {n_dropped} queued" if n_dropped else ""
            if err:
                self._set_status(f"{label} cancel FAILED at {sent_at}: {err[:60]}{suffix}")
                self._tui_log(
                    "turn_stop_failed",
                    target=label,
                    tid=tid,
                    mode=mode,
                    error=err[:200],
                )
            elif force:
                self._set_status(f"{label} KILL sent at {sent_at} — SIGKILL now{suffix}")
                self._tui_log("turn_stop_ok", target=label, tid=tid, mode="hard")
            else:
                self._set_status(
                    f"{label} stop sent at {sent_at} — finishing current step; ESC again (or /abort) to kill{suffix}"
                )
                self._tui_log(
                    "turn_stop_ok",
                    target=label,
                    tid=tid,
                    mode="graceful",
                )

        # Arm / disarm the two-stage escalation. A graceful ESC arms the
        # active turn so the NEXT ESC hard-kills it; a hard kill (or no
        # live turn) clears the arming.
        if active_tid is not None and via == "esc" and not force:
            self._hardkill_armed_tid = active_tid
        elif force:
            self._hardkill_armed_tid = None

        # If a /claude delegate is in-flight, stop it via the same
        # `turn_stop` path used for normal turns — the server-side
        # `delegate_to_external` endpoint registers the tool runner
        # under that turn_id, so the cancel flows through the agent's
        # `TurnState.stop` and kills the claude CLI subprocess.
        if self._claude_turn_id is not None:
            await _send_cancel(self._claude_turn_id, "claude")
            return
        if self._codex_turn_id is not None:
            await _send_cancel(self._codex_turn_id, "codex")
            return
        if self._antigravity_turn_id is not None:
            await _send_cancel(self._antigravity_turn_id, "antigravity")
            return
        if self._current_turn_id is None:
            self._hardkill_armed_tid = None
            if n_dropped:
                self._set_status(f"dropped {n_dropped} queued prompt(s)")
                self._tui_log("esc_dropped_queue_only", queued=n_dropped)
            else:
                # No local turn id — still may be a ``waiting`` parent with
                # live background Agent children. ESC / /stop / /abort must
                # cancel that subtree (gateway force cascade), not no-op.
                sid = getattr(self, "_sid", None) or getattr(self, "_session_id", None)
                client = getattr(self, "_client", None)
                if sid and client is not None and hasattr(client, "cancel_prompt"):
                    # Child cascade only runs on force=True in the gateway.
                    # Waiting parents have no in-flight round to finish
                    # gracefully, so always force-kill the tree here.
                    try:
                        await client.cancel_prompt(str(sid), force=True)
                    except Exception as e:  # noqa: BLE001
                        self._set_status(f"tree cancel FAILED: {str(e)[:60]}")
                        self._tui_log(
                            "tree_cancel_failed",
                            sid=str(sid)[:12],
                            error=str(e)[:200],
                        )
                    else:
                        self._set_status("tree cancel sent — stopping subagents…")
                        self._tui_log(
                            "tree_cancel_ok",
                            sid=str(sid)[:12],
                            via=via,
                            mode="hard",
                        )
                else:
                    self._set_status("nothing to abort — no turn in flight")
                    self._tui_log("esc_noop")
            return
        await _send_cancel(self._current_turn_id, "agent")

    def on_worker_state_changed(self, event) -> None:  # noqa: ANN001
        """Capture worker exceptions into the structured error log.

        Textual `run_worker(coro, …)` swallows exceptions into the
        worker's `state = ERROR` and `_error` attributes, prints a
        traceback to the dev console, but doesn't crash the app. We
        listen for the state change here and persist the error so the
        agent (or the user) can review it later out-of-band.
        """
        try:
            from textual.worker import WorkerState
        except Exception:  # noqa: BLE001
            return
        worker = getattr(event, "worker", None)
        state = getattr(worker, "state", None)
        if state != WorkerState.ERROR or worker is None:
            return
        exc = getattr(worker, "_error", None) or getattr(worker, "error", None)
        if not isinstance(exc, BaseException):
            return
        try:
            from sharedagents_core import errors as _errors

            _errors.log_error(
                f"tui:{worker.name or 'worker'}",
                exc,
                summary=f"Textual worker '{worker.name}' raised {type(exc).__name__}",
                context={
                    "worker_name": worker.name,
                    "worker_description": str(worker.description)[:200],
                    "permission_mode": self._permission_mode,
                    "claude_session_id": self._claude_session_id,
                },
                session_id=getattr(self, "_session_id", None) or self._claude_session_id,
                agent_name=getattr(self.agent_cfg, "name", None),
            )
        except Exception:  # noqa: BLE001
            # Logging errors must never themselves break the app.
            pass

    async def _run_external_engine(
        self,
        engine_kind: EngineKind,
        prompt: str,
    ) -> None:
        """Desugar `/claude <prompt>` to the `delegate_claude` tool.

        Routes through the agent's `POST /v1/sessions/{sid}/delegate`
        endpoint, which spawns the same `delegate_claude` tool runner
        the local model would use if it emitted the call autonomously.
        One path, one runner, one kill mechanism (per the foundational
        rule in CLAUDE.md: external agent bridges are tools, not slash
        commands or frontend-special-cased subprocess spawns).

        - Claude session reuse: server-side state file at
          `<sessions_dir>/<sid>.delegate_claude.json` (managed by the
          tool runner) — no TUI tracking needed.
        - Preamble: built locally from unsynced sharedagents turns and
          prepended to the user's prompt so a follow-up like
          `/claude research more on this` resolves "this" against the
          prior local-model exchange. (When the local model emits the
          delegate tool call autonomously, it can pick context itself
          without the preamble; the slash command can't, since the
          user types raw text.)
        - Persistence: the server endpoint stores the result as a turn
          with `metrics.source = "claude"`, so `_build_messages` wraps
          it with attribution on the next local-model turn.
        - ESC: server tracks the running tool process under `turn_id`,
          so the abort path is the standard `turn_stop(turn_id)` we
          use for normal turns.
        - Streaming: Claude's text deltas flow through the runner's
          `on_delta` callback into `tool_call_update` ACP events,
          which the TUI renders into the inline `message_cls`
          widget.
        """
        engine_kind = EngineKind(engine_kind)
        spec = spec_for(engine_kind)
        engine_name = spec.short_name
        message_cls = {
            EngineKind.CLAUDE_CLI: ClaudeMessage,
            EngineKind.CODEX_CLI: CodexMessage,
            EngineKind.ANTIGRAVITY_CLI: AntigravityMessage,
            EngineKind.ZCODE_CLI: ZCodeMessage,
        }[engine_kind]
        turn_attr = {
            EngineKind.CLAUDE_CLI: "_claude_turn_id",
            EngineKind.CODEX_CLI: "_codex_turn_id",
            EngineKind.ANTIGRAVITY_CLI: "_antigravity_turn_id",
            EngineKind.ZCODE_CLI: "_zcode_turn_id",
        }[engine_kind]
        session_attr = {
            EngineKind.CLAUDE_CLI: "_claude_session_id",
            EngineKind.CODEX_CLI: "_codex_thread_id",
            EngineKind.ANTIGRAVITY_CLI: "_antigravity_session_id",
            EngineKind.ZCODE_CLI: "_zcode_session_id",
        }[engine_kind]

        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage(f"❯ /{engine_name} {prompt}"))
        msg: Any = message_cls()
        delegate_messages: list[Any] = [msg]
        await transcript.mount(msg)
        self._maybe_scroll_end(transcript)

        if self._sid is None or self._client is None:
            msg.mark_error(f"no active session — start one before using /{engine_name}")
            self._set_status(f"/{engine_name} requires an active session")
            return

        # Context injection is a BACKEND concern, not an interface one.
        # The gateway's direct-dispatch handler
        # (LocalLLMEngine._run_direct_tool_call) builds the
        # <sharedagents-context> preamble from the server-side session turns
        # for EVERY frontend (TUI, Telegram, …) and threads the parent
        # session id so the CLI session resumes. The TUI therefore sends
        # the bare prompt and lets the backend prepend context uniformly.
        full_prompt = prompt

        # Persistent indicator (survives transient `_set_status` writes
        # like "reasoning expanded") so the user always sees /claude is
        # in flight.
        self._set_worker_state(
            f"[bold magenta]🤖 {engine_name} running…[/bold magenta]",
            detail="[dim]ESC to cancel[/dim]",
        )

        delegate_turn_id: str | None = None
        # Direct-dispatch path emits ACP tool_call / tool_call_update
        # events. agent_client.session_turn translates them to
        # ``tool_call``, ``tool_delta``, ``tool_result``, ``done``.
        # Track terminal status so we can mark the message ok / error.
        delegate_done = False
        delegate_ok = False
        delegate_errored = False
        # The bridge announces ONE outer tool_call (title
        # ``delegate_claude``) plus zero-or-more inner tool_calls for
        # Claude's own ``Read`` / ``Bash`` / ``TodoWrite`` / etc.
        # blocks — the inner ones carry a ``.inner.`` segment in
        # their call_id. Distinguishing the two lets the outer
        # tool_delta / tool_result keep streaming into the
        # message_cls body while the inner ones get rendered as
        # native sharedagents widgets (ToolCallMessage + ToolResultMessage,
        # TodoWrite tick-through into the plan checklist).
        outer_call_id: str | None = None
        # Wall-clock timing for the per-call metrics row. Mirrors the
        # native-turn shape (``wt`` / ``ttft`` / ``gen`` / ``out``) so
        # /claude turns have the same telemetry footer as a gpu-a-served
        # turn — operators can compare them side by side. We don't get
        # real prefill/cache numbers from Claude Code, so the row is
        # marked ``tokens_estimated`` and the prompt-token segment is
        # omitted.
        delegate_t0 = time.monotonic()
        delegate_first_chunk_at: float | None = None
        delegate_chars = 0

        def _is_outer(call_id: str) -> bool:
            if outer_call_id is None:
                return True
            return call_id == outer_call_id

        async def _ensure_delegate_message() -> Any:
            nonlocal msg
            if msg is None:
                msg = message_cls()
                delegate_messages.append(msg)
                await transcript.mount(msg)
                self._maybe_scroll_end(transcript)
            return msg

        async def _close_delegate_segment_for_tool() -> None:
            """Freeze the current prose segment before an inner tool widget.

            Slash delegates stream all prose through one body widget by
            default. If the child CLI emits a tool event mid-stream, that
            single fixed-position body would keep receiving later prose above
            the new tool widgets, making tools appear after the final answer.
            Closing the segment makes the next delta mount a fresh body at the
            transcript tail, preserving event order.
            """
            nonlocal msg
            if msg is None:
                return
            try:
                await msg.wait_for_refresh()
            except Exception:  # noqa: BLE001
                pass
            msg.finalize()
            try:
                await msg.wait_for_refresh()
            except Exception:  # noqa: BLE001
                pass
            msg = None

        def _last_delegate_message() -> Any:
            return msg or delegate_messages[-1]

        try:
            async for event_name, data, turn_id in self._client.delegate(
                self._sid,
                prompt=full_prompt,
                tool=f"delegate_{engine_name}",
                mode=self._permission_mode,
            ):
                if turn_id and delegate_turn_id is None:
                    delegate_turn_id = turn_id
                    setattr(self, turn_attr, turn_id)
                if event_name == "started":
                    continue
                if event_name in ("delta", "tool_delta"):
                    # Outer-call streaming text only. Inner tool deltas
                    # would carry a different call_id; ignore them
                    # here so they don't leak Claude's tool stdout
                    # into the prose pane.
                    call_id = str(data.get("call_id") or "")
                    if outer_call_id and call_id and call_id != outer_call_id:
                        continue
                    txt = data.get("text") or ""
                    if txt:
                        if delegate_first_chunk_at is None:
                            delegate_first_chunk_at = time.monotonic()
                        delegate_chars += len(txt)
                        active_msg = await _ensure_delegate_message()
                        active_msg.append_content(txt)
                        self._maybe_scroll_end(transcript)
                    continue
                if event_name == "tool_call":
                    call_id = str(data.get("call_id") or "")
                    tool_name = str(data.get("tool") or "?")
                    # First tool_call IS the outer delegate_claude
                    # announce. Mission D2 invariant from MISSION.md:
                    #
                    #   User → tool-call header → response body → result
                    #
                    # message_cls was mounted upfront at app.py:10528
                    # so it already sits between User and the
                    # streaming chunks. Inject the header card BEFORE
                    # that body widget so the transcript order matches
                    # the mission contract; without ``before=msg`` the
                    # header would land below the body and the
                    # operator sees the answer first.
                    if outer_call_id is None:
                        outer_call_id = call_id
                        try:
                            await transcript.mount(
                                ToolCallMessage(
                                    tool_name,
                                    data.get("arguments") or {},
                                ),
                                before=msg,
                            )
                        except Exception:  # noqa: BLE001
                            # Defensive: a Textual mount-before failure
                            # should never block the delegate stream.
                            pass
                        continue
                    if call_id == outer_call_id:
                        continue
                    # Inner Claude tool: render as a native widget.
                    # TodoWrite additionally ticks the plan checklist
                    # via the existing handler — that's the path that
                    # makes ``/claude`` plans show up the same way a
                    # local-model plan does.
                    if _is_todo_tool_name(tool_name):
                        self._handle_todo_tool_call(
                            data.get("arguments") or {},
                        )
                    inner_args = data.get("arguments") or {}
                    await _close_delegate_segment_for_tool()
                    if tool_name == "ExitPlanMode":
                        plan_text = str(inner_args.get("plan") or "")
                        await transcript.mount(PlanMessage(plan_text))
                        if call_id:
                            self._plan_call_ids.add(call_id)
                    elif tool_name == "OutputSection":
                        section_name = str(inner_args.get("section_name") or "")
                        section_content = str(inner_args.get("content") or "")
                        await transcript.mount(OutputSectionMessage(section_name, section_content))
                        if call_id:
                            self._section_call_ids.add(call_id)
                    elif tool_name == "UserQuestion":
                        await transcript.mount(UserQuestionMessage(inner_args))
                        if call_id:
                            self._user_question_call_ids.add(call_id)
                    elif tool_name in ("Update", "Edit", "edit"):
                        # Claude's inner file edits get the same
                        # "Update" diff-card treatment as a local-model
                        # edit — Claude Code itself labels `Edit` as
                        # "Update", so this matches the delegate's own
                        # UI exactly.
                        await transcript.mount(
                            UpdateMessage(
                                str(inner_args.get("file_path") or ""),
                                str(inner_args.get("old_string") or ""),
                                str(inner_args.get("new_string") or ""),
                                replace_all=bool(inner_args.get("replace_all", False)),
                                call_id=call_id,
                            )
                        )
                        if call_id:
                            self._update_call_ids.add(call_id)
                    else:
                        await transcript.mount(ToolCallMessage(tool_name, inner_args))
                    self._maybe_scroll_end(transcript)
                    continue
                if event_name == "tool_result":
                    call_id = str(data.get("call_id") or "")
                    if outer_call_id is not None and not _is_outer(call_id):
                        # Inner Claude tool's terminal result. Mount a
                        # ToolResultMessage so the operator can fold
                        # open Read/Bash/etc. output. TodoWrite's
                        # result body is a recap of the plan — fine to
                        # show inline alongside the plan widget.
                        if call_id in self._plan_call_ids:
                            self._plan_call_ids.discard(call_id)
                            self._maybe_scroll_end(transcript)
                            continue
                        ok = bool(data.get("ok", True))
                        content = data.get("content") or ""
                        if call_id in self._section_call_ids:
                            self._section_call_ids.discard(call_id)
                            if ok:
                                self._maybe_scroll_end(transcript)
                                continue
                        if call_id in self._user_question_call_ids:
                            self._user_question_call_ids.discard(call_id)
                            if ok:
                                self._maybe_scroll_end(transcript)
                                continue
                        if call_id in self._update_call_ids:
                            self._update_call_ids.discard(call_id)
                            card = next(
                                (w for w in transcript.query(UpdateMessage) if w.call_id == call_id),
                                None,
                            )
                            if card is not None:
                                card.mark_result(ok)
                            if ok:
                                self._maybe_scroll_end(transcript)
                                continue
                        await transcript.mount(ToolResultMessage(call_id, ok, content))
                        self._maybe_scroll_end(transcript)
                        continue
                    # Outer delegate terminal: surface the body.
                    delegate_done = True
                    ok = bool(data.get("ok", False))
                    content = data.get("content") or ""
                    delegate_ok = ok
                    if not any(m.content_text for m in delegate_messages):
                        if delegate_first_chunk_at is None:
                            delegate_first_chunk_at = time.monotonic()
                        delegate_chars += len(content)
                        active_msg = await _ensure_delegate_message()
                        active_msg.append_content(content)
                        self._maybe_scroll_end(transcript)
                    # Mission D2: the terminal tool-result row must
                    # land AFTER the response body so the transcript
                    # reads ``User → 🔧 header → body → ✅ result``.
                    # We mount a fresh ``ToolResultMessage`` at the
                    # end of the transcript; the streaming delta
                    # path streamed the content into message_cls,
                    # so this row is a compact status marker (not
                    # a body duplicate) — pass ``content=""`` to
                    # avoid re-rendering Claude's whole response in
                    # a collapsible.
                    try:
                        await transcript.mount(
                            ToolResultMessage(call_id, ok, ""),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    if not ok:
                        delegate_errored = True
                        err = (content or f"{engine_name} failed")[:300]
                        _last_delegate_message().mark_error(err)
                        self._set_status(f"{engine_name} error: {err[:80]}")
                    continue
                if event_name == "error":
                    err = (data.get("error") or data.get("message") or "delegate failed")[:300]
                    delegate_errored = True
                    _last_delegate_message().mark_error(err)
                    self._set_status(f"{engine_name} error: {err[:80]}")
                    return
                if event_name == "done":
                    ok = delegate_ok if delegate_done else (data.get("state") != "error")
                    if not ok and not delegate_errored:
                        final_msg = _last_delegate_message()
                        err = (final_msg.content_text or f"{engine_name} failed")[:300]
                        final_msg.mark_error(err)
                        self._set_status(f"{engine_name} error: {err[:80]}")
                        return
                    if delegate_errored:
                        return
                    # Stamp a metrics row on the delegated turn so it
                    # reads the same as a native turn (model · ttft ·
                    # gen tps · out · wt).
                    delegate_metrics = _build_delegate_metrics(
                        label=engine_name,
                        t0=delegate_t0,
                        first_chunk_at=delegate_first_chunk_at,
                        chars=delegate_chars,
                    )
                    if delegate_metrics:
                        _last_delegate_message().set_metrics(delegate_metrics)
                    _last_delegate_message().finalize(getattr(self, session_attr, None))
                    # The agent appended one /claude turn to the saved
                    # session — keep `_rendered_turn_count` in sync with
                    # the disk row count so the cross-frontend renderer
                    # doesn't treat our just-finished /claude reply as
                    # "external" and re-render it.
                    if self._sid is not None:
                        try:
                            data = await self._client.get_session(self._sid)
                            self._rendered_turn_count = sum(
                                1 for _t in (data.get("turns") or []) if not _t.get("in_progress")
                            )
                            if self.session is not None:
                                self.session.turns = list(data.get("turns") or [])
                        except Exception:  # noqa: BLE001
                            self._rendered_turn_count += 1
                        # Our just-finished local delegate turn is now on disk;
                        # push the renderer's offset anchor past it so the
                        # subscribe-tail doesn't replay it as an "external" turn.
                        self._rendered_offset = max(
                            self._rendered_offset,
                            self._session_log_end_offset(self._sid),
                        )
                    sid_short = (getattr(self, session_attr, None) or "?")[:12]
                    self._set_status(f"{engine_name} done · session={sid_short} (saved)")
                    return
        except asyncio.CancelledError:
            # ESC landed during the delegate stream. Best-effort tell
            # the server to kill the tool runner — the streaming GET is
            # already cancelled, so the task we're in is unwinding.
            if delegate_turn_id:
                try:
                    await self._client.turn_stop(delegate_turn_id)
                except Exception:  # noqa: BLE001
                    pass
            _last_delegate_message().mark_aborted()
            raise
        except Exception as e:  # noqa: BLE001
            _last_delegate_message().mark_error(f"{e}"[:300])
            self._set_status(f"{engine_name} error: {e}"[:80])
        finally:
            setattr(self, turn_attr, None)
            self._set_worker_state("")
            # Drain the queue so any prompts/claude calls submitted while
            # this one was running fire next.
            self._maybe_start_next()

    async def _run_claude(self, prompt: str) -> None:
        await self._run_external_engine(EngineKind.CLAUDE_CLI, prompt)

    async def _run_codex(self, prompt: str) -> None:
        await self._run_external_engine(EngineKind.CODEX_CLI, prompt)

    async def _run_antigravity(self, prompt: str) -> None:
        await self._run_external_engine(EngineKind.ANTIGRAVITY_CLI, prompt)

    async def _run_zcode(self, prompt: str) -> None:
        await self._run_external_engine(EngineKind.ZCODE_CLI, prompt)

    async def _send_user_turn(self, text: str) -> None:
        # Defence-in-depth: callers should go through `_enqueue_or_run`
        # to avoid clobbering an in-flight turn, but if anything reaches
        # us while busy, push to the queue rather than dropping the input.
        if self._is_busy():
            self._turn_queue.append(("prompt", text))
            self._set_status(f"queued ({len(self._turn_queue)} pending) — will run when current turn finishes")
            self._refresh_status_bar()
            self._refresh_queue_display()
            return
        # Recover from a previous failed session-create (e.g. transient agent
        # or upstream blip): try once more on the user's next input rather
        # than locking them out until they restart the TUI.
        if self._sid is None and not await self._ensure_session():
            return
        # ``/restart`` (or a watcher tear-down) leaves ``self._client = None``
        # while the gateway is bouncing. Without a guard the submit path
        # crashes downstream with ``'NoneType' object has no attribute
        # 'session_turn'``. Drive an explicit reconnect on the user's next
        # input rather than asking them to re-type after the TUI heals.
        if self._client is None:
            self._set_status("⟳ reconnecting to gateway…")
            try:
                await self._reconnect_after_restart()
            except Exception as e:  # noqa: BLE001
                self._set_status(f"reconnect failed: {e} — retry the prompt")
                return
            if self._client is None:
                self._set_status("gateway still warming up — wait a moment then retry")
                return
            # Reattach to the existing session id if possible; if the agent
            # forgot the sid (process bounced), drop to a fresh session so
            # the prompt isn't sent against an orphan id.
            if self._sid is not None:
                try:
                    await self._client.attach_session(
                        self._sid,
                        cwd=os.getcwd(),
                    )
                except Exception:  # noqa: BLE001
                    self._sid = None
                    if not await self._ensure_session():
                        return

        transcript = self.query_one("#transcript", VerticalScroll)
        # Slash-command compact echo: when the prompt was expanded
        # from a builtin command file, the operator typed something
        # short (``/goal X``) and shouldn't see the full markdown
        # body printed in the transcript. The dispatcher stashes the
        # compact form in ``self._next_user_echo`` right before
        # calling here; we use it for the UserMessage, then clear it
        # so the next free-form prompt echoes verbatim.
        #
        # The ``humanise_delegate_marker`` pass strips the
        # ``__SHAREDAGENTS_DELEGATE__:<tool>:<json>`` wire sentinel the
        # ``/agent <sibling>`` direct-dispatch path at app.py:5138-5152
        # builds — without this the operator sees the raw marker JSON
        # instead of their ``/<sibling> <task>`` text.
        _echo = getattr(self, "_next_user_echo", None) or humanise_delegate_marker(text)
        self._next_user_echo = None
        await transcript.mount(UserMessage(f"❯ {_echo}"))
        # Two-phase scroll: the immediate call covers the common case;
        # `call_after_refresh` re-scrolls AFTER Textual lays out the new
        # widget, which is when scroll_end actually has the right
        # target height. Without the second call, the user's message
        # only became visible once the first thinking/content chunk
        # arrived and triggered another scroll — the user-reported bug.
        # User-action scroll: the operator hit Enter, they explicitly
        # want their new message visible. Bypass the auto-follow gate
        # so they land at the bottom even if they were reading back,
        # and re-arm follow-mode for subsequent passive chunks.
        self._scroll_follow = True
        transcript.scroll_end(animate=False)
        self.call_after_refresh(lambda: transcript.scroll_end(animate=False))
        # AssistantMessage is mounted lazily on the first `chunk` event of
        # each LLM round — that way the transcript reads chronologically:
        #     ❯ user → reasoning₁ → 🔧 tool_call → ✅ tool_result → reasoning₂ …
        # When a `tool_call` event arrives mid-turn we finalize the current
        # AssistantMessage and clear `_current_assistant`; the next chunk
        # mounts a fresh one below the tool widgets. Avoids the "all chunks
        # stack into one widget at the top, tools dangle below" bug.
        self._current_assistant = None
        self._round_pending_assistant = None
        self._current_user_text = text
        # Wipe any stale in-flight tool_call_ids from a prior turn (an
        # ESC abort or transport drop can leave the set non-empty);
        # otherwise the next-round eager assistant mount would be
        # suppressed forever on this fresh turn.
        self._pending_tool_calls.clear()
        self._plan_call_ids.clear()
        self._section_call_ids.clear()

        # MISSION.md D1: arm the "clear on first TaskCreate of this
        # new task group" trigger. Each top-level user prompt starts
        # a new logical task group; the prior group's rows on the
        # persistent ``#harness-tasks`` footer get cleared when the
        # model fires its first ``TaskCreate`` of this turn (see the
        # tool_call dispatch at the ``_is_harness_task_tool`` branch).
        # Set BEFORE spawning the streaming worker so the flag is
        # visible to the dispatch loop the worker drives. Mid-turn
        # ``/steer`` injection (``_inject_user_message_via_acp``) does
        # NOT set the flag — those continue the in-flight turn.
        self._new_task_group_pending = True
        # Arm the local-submit guard BEFORE spawning the worker so the
        # session renderer skips this turn even if its tail sees the
        # ``turn_started`` frame before the initiator does (no double-render).
        self._local_submit_pending = True
        # `exclusive=False`: the dispatcher (`_enqueue_or_run`) guarantees
        # we never overlap turns, so we don't need Textual to cancel old
        # workers — and `exclusive=True` would cancel a still-running
        # `_run_claude` if the user submits a plain prompt right after.
        self.run_worker(self._stream_via_agent(text), exclusive=False, name="stream")

    def _render_event(self, frame: dict) -> list[tuple[str, dict, str | None]]:
        """Map ONE canonical gateway event frame to the legacy TUI events
        the streaming dispatch already renders (spec §7 ``render(event)``).

        ``frame`` is a ``{offset, seq, spawnedBy, event}`` envelope from
        :meth:`AgentClient.subscribe_events`; ``event`` carries the
        canonical ``type`` (``text_delta``, ``thinking_delta``,
        ``tool_call``, ``tool_result``/``tool_result_meta``,
        ``round_ended``, ``turn_ended`` …). We translate to the SAME
        ``(event, data, turn_id)`` tuples ``session_turn`` historically
        produced so EVERY existing widget path (assistant bubble append,
        reasoning box, tool-call / tool-result widgets, subagent boxes,
        abort/error banner, per-round + turn metrics) renders unchanged —
        live, replay and observe all flow through this one map.

        The ``turn_id`` is the canonical ``turn_idx`` (stringified) so the
        TUI's ESC path can resolve it back to the running server-side turn
        via the cancel route. Returns zero-or-more tuples per frame
        (most events map 1:1; some, like ``turn_ended``, fan out to a
        terminal ``error`` + ``done`` pair).
        """
        ev = frame.get("event") if isinstance(frame, dict) else None
        if not isinstance(ev, dict):
            return []
        etype = ev.get("type") or ""
        ti = ev.get("turn_idx")
        tid = str(ti) if ti is not None else None

        if etype == "session_state_update":
            self._client.fold_session_state(ev)
            effective = ev.get("effective_mode") or {}
            if isinstance(effective, dict) and effective.get("id"):
                self._permission_mode = str(effective["id"])
            elif ev.get("effective_mode_id"):
                self._permission_mode = str(ev["effective_mode_id"])
            self._refresh_status_bar()
            return []
        if etype == "text_delta":
            delta = ev.get("delta", "") or ""
            if not delta:
                return []
            event_text = getattr(self, "_event_text_by_turn", None)
            if event_text is None:
                event_text = {}
                self._event_text_by_turn = event_text
            event_text[tid or ""] = event_text.get(tid or "", "") + delta
            return [("chunk", {"choices": [{"delta": {"content": delta}}]}, tid)]
        if etype == "thinking_delta":
            delta = ev.get("delta", "") or ""
            if not delta:
                return []
            return [
                (
                    "chunk",
                    {"choices": [{"delta": {"reasoning": delta}}]},
                    tid,
                )
            ]
        if etype == "tool_call":
            raw_args = ev.get("arguments", {}) or {}
            if isinstance(raw_args, dict):
                # Drop Cursor metadata keys so the 🔧 pill never shows
                # ``{"_toolName": "updateTodos"}`` from durable logs.
                raw_args = {k: v for k, v in raw_args.items() if k not in ("_toolName", "toolName", "criticality")}
            return [
                (
                    "tool_call",
                    {
                        "tool": ev.get("name", "") or "tool",
                        "call_id": ev.get("call_id", "") or "",
                        "arguments": raw_args,
                    },
                    tid,
                )
            ]
        if etype == "tool_delta":
            delta = ev.get("delta", "") or ""
            cid = ev.get("call_id", "") or ""
            if not delta or not cid:
                return []
            return [("tool_delta", {"call_id": cid, "text": delta}, tid)]
        if etype == "subagent_result":
            return [("subagent_result", ev, tid)]
        if etype == "tool_result_meta":
            # ``ok`` lives on the meta event; the body arrives separately
            # on ``tool_result``. The TUI's tool_result widget wants
            # ``{call_id, ok, content}`` in one shot. Stash ``ok`` keyed by
            # call_id and let the matching ``tool_result`` emit the tuple.
            cid = ev.get("call_id", "") or ""
            if cid:
                self._event_tool_ok[cid] = bool(ev.get("ok"))
                if not hasattr(self, "_event_tool_outcomes"):
                    self._event_tool_outcomes = {}
                self._event_tool_outcomes[cid] = ev.get("outcome")
            return []
        if etype == "tool_result":
            cid = ev.get("call_id", "") or ""
            ok = self._event_tool_ok.pop(cid, True)
            return [
                (
                    "tool_result",
                    {
                        "call_id": cid,
                        "ok": ok,
                        "content": ev.get("content", ""),
                        "outcome": getattr(self, "_event_tool_outcomes", {}).pop(cid, None),
                    },
                    tid,
                )
            ]
        if etype == "round_ended":
            metrics = dict(ev.get("metrics") or {})
            if ev.get("finish_reason") and "finish_reason" not in metrics:
                metrics["finish_reason"] = ev["finish_reason"]
            return [("metrics", metrics, tid)]
        if etype == "turn_ended":
            out: list[tuple[str, dict, str | None]] = []
            err = ev.get("error")
            aborted = bool(ev.get("aborted"))
            final_content = str(ev.get("final_content") or "")
            event_text = getattr(self, "_event_text_by_turn", None)
            streamed = event_text.pop(tid or "", "") if isinstance(event_text, dict) else ""
            # The durable session stream emits one terminal frame with the
            # complete answer. Rich native streams may have
            # already sent deltas, so emit only a missing suffix and never
            # replay the accumulated body.
            terminal_delta = ""
            if final_content:
                if not streamed:
                    terminal_delta = final_content
                elif final_content.startswith(streamed):
                    terminal_delta = final_content[len(streamed) :]
            if terminal_delta:
                out.append(
                    (
                        "chunk",
                        {"choices": [{"delta": {"content": terminal_delta}}]},
                        tid,
                    )
                )
            if ev.get("metrics"):
                out.append(("metrics", dict(ev["metrics"]), tid))
            if aborted:
                out.append(
                    (
                        "done",
                        {"state": "aborted", "error": err} if err else {"state": "aborted"},
                        tid,
                    )
                )
            elif err:
                out.append(("error", {"message": str(err)}, tid))
                out.append(("done", {"state": "error", "error": err}, tid))
            else:
                out.append(("done", {"state": "complete"}, tid))
            return out
        # session_started / round_started / header_update / session_closed /
        # lifecycle(subagent_spawned/plan_update) carry no live-render tuple
        # today — the subagent tree renders from the tool_call/tool_result
        # stream (delegate_* boxes) exactly as the ACP path did, and header
        # state is re-read from get_session at turn end. Drop them here.
        return []

    async def _turn_event_source(
        self,
        text: str,
    ) -> AsyncIterator[tuple[str, dict, str | None]]:
        """Yield this turn's ``(event, data, turn_id)`` tuples.

        Two transports, same tuple shape so the ``_stream_via_agent``
        dispatch body is identical either way:

        * **gateway / inverted (spec §9.1+§9.2)** — SUBMIT the prompt
          fire-and-forget, then SUBSCRIBE to the cursor stream from the
          returned ``start_offset`` and ``render(event)`` each canonical
          frame through :meth:`_render_event`. The initiator renders ONLY
          from the stream (no local echo / double-render). The loop ends
          when ``turn_ended`` for THIS turn lands.
        * **stdio / test fallback** — drive the ACP ``session_turn``
          iterator (unchanged), used when no gateway URL is configured.
        """
        if not getattr(self, "_use_event_stream", False):
            async for triple in self._client.session_turn(
                self._sid,
                user=text,
                enable_thinking=self.enable_thinking,
                permission_mode=self._permission_mode,
            ):
                yield triple
            return

        await self._client.get_session_state(self._sid)
        ack = await self._client.submit_prompt(
            self._sid,
            text,
            permission_mode=self._permission_mode or "",
        )
        start_offset = int(ack.get("start_offset") or 0)
        last_off = start_offset
        async for frame in self._client.subscribe_events(
            self._sid,
            from_offset=start_offset,
        ):
            ev = frame.get("event") if isinstance(frame, dict) else None
            if isinstance(frame, dict):
                off = frame.get("offset")
                if isinstance(off, int) and off > last_off:
                    last_off = off
            for triple in self._render_event(frame):
                yield triple
            # Terminal: this turn's ``turn_ended`` closes the subscribe
            # loop for the initiator. The session-level renderer (started
            # separately) keeps tailing for externally-driven turns.
            if isinstance(ev, dict) and ev.get("type") == "turn_ended":
                # We just rendered this LOCAL turn end-to-end; push the
                # renderer's offset anchor past it so the session-level tail
                # never replays it as an "external" turn (covers the race
                # where the renderer reaches this turn_started only after the
                # initiator already cleared ``_current_turn_id``).
                self._rendered_offset = max(self._rendered_offset, last_off)
                break

    @staticmethod
    def _unrendered_turn_offset(frame: Any, high_offset: int) -> int | None:
        """De-dup rule for the cross-frontend renderer.

        Return the byte ``offset`` of ``frame`` iff it opens a turn the renderer
        has NOT yet handled — i.e. a ``turn_started`` event whose offset is
        past ``high_offset`` — else ``None``.

        Identity is the SSE frame's monotonic byte offset, never ``turn_idx``:
        a busy session drains a queued prompt burst (Telegram "send 5 essays")
        back-to-back and stamps EVERY ``turn_started`` with the same turn_idx,
        so a turn_idx high-water collapsed the burst to its first turn. Offsets
        stay distinct, so each queued turn crosses the threshold exactly once.
        """
        if not isinstance(frame, dict):
            return None
        ev = frame.get("event")
        if not isinstance(ev, dict) or ev.get("type") != "turn_started":
            return None
        off = int(frame.get("offset") or 0)
        return off if off > high_offset else None

    @staticmethod
    def _session_log_end_offset(sid: str) -> int:
        """Byte size of the log ``subscribe_events`` tails for ``sid``.

        The log only grows, so its size is the offset just past the last
        event; a turn appended after this returns a strictly larger frame
        offset. 0 if absent/unreadable (a brand-new session has no history
        to skip, which is the correct anchor anyway).
        """
        if not sid:
            return 0
        try:
            from sharedagents_core.layout import sessions_root

            root = sessions_root()
            return int((root / f"{sid}.jsonl").stat().st_size)
        except (OSError, ValueError):
            return 0

    async def _handle_queue_event(self, ev: dict) -> None:
        """Handle a ``queue_msg_*`` lifecycle event from the session log.

        Triggered by ``_render_session_log`` for any event whose type
        starts with ``queue_msg_``.  Refreshes the queue widget and, for
        ``queue_msg_injected``, mounts the injected text as a
        ``UserMessage`` in the transcript so the operator can see what
        was consumed from the durable queue.
        """
        etype = ev.get("type", "")
        # Keep the widget up-to-date for every mutation event.
        await self._refresh_durable_queue()
        if etype == "queue_msg_injected":
            text = user_prompt_text(ev.get("text"), [f["name"] for f in ev.get("attached_files") or []])
            if text:
                try:
                    transcript = self.query_one("#transcript", VerticalScroll)
                    await transcript.mount(
                        UserMessage(f"❯ {humanise_delegate_marker(text)}"),
                    )
                    await transcript.mount(
                        Static(
                            "↳ dequeued and injected from durable queue",
                            classes="claude-msg",
                        )
                    )
                    self._maybe_scroll_end(transcript)
                except Exception:  # noqa: BLE001
                    pass

    async def _stream_via_agent(
        self,
        text: str,
        *,
        event_source: AsyncIterator[tuple[str, dict, str | None]] | None = None,
        from_log: bool = False,
    ) -> None:
        """Render one turn into the transcript from a stream of legacy
        ``(event, data, turn_id)`` tuples.

        The default source is :meth:`_turn_event_source` (submit+subscribe
        in gateway mode, ``session_turn`` in stdio/test mode) — the
        initiator path. ``event_source`` overrides it with a caller-built
        stream: the cross-frontend renderer passes a subscribe-from-offset
        stream of an externally-driven turn, so a rendered turn flows
        through the EXACT same widget dispatch (full fidelity) as a local
        one. ``from_log=True`` swaps the footer cue to "rendering…" and skips
        the local queue drain (the renderer doesn't own the local turn
        queue).
        """
        transcript = self.query_one("#transcript", VerticalScroll)
        self._set_worker_state(
            "[bold cyan]rendering…[/bold cyan]" if from_log else "[bold green]▶ streaming…[/bold green]",
            detail=None if from_log else "[dim]ESC to cancel[/dim]",
        )
        self._tui_log(
            "render_start" if from_log else "stream_start",
            prompt_chars=len(text),
        )
        worker_metrics: dict | None = None
        worker_error: str | None = None
        self._turn_render_frozen = False
        # Fresh bubble list for this turn's thinking-title reconcile.
        self._turn_assistants = []

        async def _ensure_assistant() -> AssistantMessage:
            """Lazily mount a new AssistantMessage; the active round writes here.

            Stays on `self._current_assistant` between events. A `tool_call`
            event finalizes and clears it; the next `chunk` triggers another
            mount. That's how the transcript ordering stays chronological
            across multi-round tool-use turns.

            Late-chunk catch-up: if the assistant was just finalized but
            no other widget has mounted yet (the bridge emitted a stray
            trailing chunk AFTER the tool_call event, common with
            deepseek-v4-pro mid-sentence tool dispatches), reuse the
            last finalized assistant so the trailing text appears in
            the same bubble — operator no longer sees ``with correct
            `name` par`` chopped before the tool widget. Only fires
            when the LAST transcript child is still the just-finalized
            assistant; once anything else mounts (the tool widget),
            the new round mounts a fresh bubble below.
            """
            if self._current_assistant is None:
                # If the last transcript child is still a finalized
                # AssistantMessage (no tool widget mounted since the
                # finalize call), reuse it — the new chunk is the
                # trailing tail of the just-closed round, NOT a new
                # round's lead. Without this, deepseek-v4-pro's
                # mid-sentence tool dispatches produced ``with
                # correct `name` par`` in bubble N and ``ameter.`` in
                # an orphan bubble N+1 below the tool widget — the
                # operator only saw the cutoff. ``append_content``
                # re-renders through RichMarkdown when the bubble is
                # already finalized.
                children = list(transcript.children)
                if children:
                    tail = children[-1]
                    if isinstance(tail, AssistantMessage) and getattr(tail, "_final", False):
                        self._current_assistant = tail
                        return tail
                a = AssistantMessage()
                await transcript.mount(a)
                self._current_assistant = a
                self._turn_assistants.append(a)
                self._maybe_scroll_end(transcript)
            return self._current_assistant

        # Eagerly mount the first round's AssistantMessage BEFORE the SSE
        # loop opens — its on_mount kicks off the prefill (TTFT) counter
        # so the user sees a live "▶ prefilling… 0.4s" line tick up while
        # the upstream model is still chewing on the prompt. Without this
        # eager mount the first visible widget after Enter would be the
        # thinking box that only appears once the first reasoning chunk
        # lands, leaving the screen looking frozen for several seconds on
        # large prompts.
        await _ensure_assistant()

        def _close_round_assistant() -> None:
            """Finalize the active AssistantMessage and clear the slot.

            ``_ensure_assistant`` re-routes any chunk that arrives
            before another widget mounts back to this same bubble (by
            checking whether the last transcript child is still a
            finalized AssistantMessage) so the "deepseek mid-sentence
            tool dispatch" race doesn't strand the trailing text in
            an orphan bubble below the tool widget.

            Records the just-closed bubble as ``_round_pending_assistant``
            so this round's per-round ``metrics`` event — which arrives
            AFTER the next round's bubble has mounted — stamps its
            latency/token stats onto THIS bubble. The bubble may be empty
            (a round whose only output was tool calls), but it still
            consumed thinking/output tokens worth showing.
            """
            if self._current_assistant is not None:
                self._round_pending_assistant = self._current_assistant
                self._current_assistant.finalize()
                self._current_assistant = None

        _source = event_source if event_source is not None else self._turn_event_source(text)
        try:
            async for event, data, tid in _source:
                if tid and not from_log:
                    # Only a locally-driven turn arms ESC/cancel against
                    # ``_current_turn_id``; a log-render must never let the
                    # operator cancel another frontend's turn from here.
                    self._current_turn_id = tid
                    # Bind tid → session so ESC/abort (``turn_stop``) resolves
                    # the gateway cancel route — the gateway-mode turn id comes
                    # from the stream, not the ACP path that self-registers.
                    try:
                        self._client.register_turn(tid, self._sid)
                    except Exception:  # noqa: BLE001
                        pass
                    # We now own the turn — disarm the submit→first-frame
                    # guard so the renderer falls back to ``_current_turn_id``.
                    self._local_submit_pending = False
                if event == "started":
                    continue
                if self._turn_render_frozen and event not in ("done", "error"):
                    continue
                if event == "chunk":
                    delta = (data.get("choices") or [{}])[0].get("delta") or {}
                    rc = delta.get("reasoning") or delta.get("reasoning_content")
                    txt = delta.get("content")
                    if not (rc or txt):
                        continue
                    # Runtime-warning sniff: the bridge formats loop /
                    # nudge / distill / force-terminate alerts as a
                    # single chunk whose body starts with one of the
                    # `_format_runtime_warning` glyph prefixes
                    # (acp_bridge.py L1153). Route those to a dedicated
                    # `RuntimeWarningNotice` widget instead of letting
                    # them blend into the assistant message bubble —
                    # without this branch a "loop detected, distilling
                    # and retrying" event is just three lines of text
                    # inside the model's normal prose, easy to miss
                    # entirely. Suppress the regular append so the
                    # warning doesn't ALSO show up inside the bubble.
                    if txt:
                        parsed = _parse_runtime_warning_chunk(txt)
                        if parsed is not None:
                            glyph, warn_event, fields = parsed
                            await transcript.mount(
                                RuntimeWarningNotice(glyph, warn_event, fields),
                            )
                            self._maybe_scroll_end(transcript)
                            # Surface in the status bar too so the
                            # operator notices even when scrolled back —
                            # the banner is the persistent record, this
                            # is the live cue.
                            try:
                                head = _RUNTIME_WARNING_HEADLINES.get(
                                    warn_event,
                                    warn_event,
                                )
                                self._set_status(f"{glyph} {head}")
                            except Exception:  # noqa: BLE001
                                pass
                            continue
                    a = await _ensure_assistant()
                    if rc and getattr(self, "reasoning_visibility", "on") != "off":
                        await a.append_thinking(rc)
                    if txt:
                        a.append_content(txt)
                    self._maybe_scroll_end(transcript)
                elif event == "metrics":
                    worker_metrics = data
                    # Per-round telemetry. Each round emits exactly one
                    # `metrics` event (RoundComplete -> usage_update),
                    # carrying THIS round's latency + token stats. It
                    # arrives AFTER the round's tool_result already mounted
                    # the NEXT round's bubble, so we stamp the bubble we
                    # stashed when the round closed
                    # (`_round_pending_assistant`) rather than
                    # `_current_assistant` — otherwise the previous round's
                    # numbers would land on the new round's bubble. This is
                    # what surfaces latency/token-generation stats for
                    # EVERY round, including rounds whose only output was
                    # tool calls (their bubble is empty but still consumed
                    # thinking/output tokens worth showing). The final
                    # (no-tool) round closes no bubble, so the pending slot
                    # is empty here; its cumulative turn stats are stamped
                    # once at turn end (the `finally` block below), where
                    # `worker_metrics` holds the cumulative TurnDone dict.
                    pending = self._round_pending_assistant
                    if pending is not None:
                        # Attach THIS round's metrics to its bubble so every
                        # round — INCLUDING a pure tool-only round (no
                        # thinking + no content deltas, e.g. with thinking
                        # off) — carries its per-round latency/token stats.
                        # The original per-round-stats feature (472616f)
                        # showed stats for EVERY round incl. tool-call-only;
                        # 21599f1 later gated the line on the bubble having
                        # content, which silently dropped tool-only rounds'
                        # stats — the regression this restores. We stamp the
                        # data on the bubble directly (not via ``set_metrics``,
                        # whose inline static would render the line ABOVE the
                        # round's tools) and surface the visible line as a
                        # standalone row BELOW the tools just below.
                        # (turn-replication: tool-only-round parity)
                        if data:
                            pending.metrics = dict(data)
                        if _build_metrics_markup(data):
                            # ORDER FIX: mount the per-round line as its OWN
                            # transcript row AFTER this round's tool call /
                            # result widgets (which already mounted between
                            # the pending bubble and the next round's eager
                            # bubble). Stamping it inline on the pending
                            # bubble — as before — rendered the output-token
                            # stats ABOVE the round's tools, so the operator
                            # saw the tool calls appear *after* the output
                            # tokens. ``_current_assistant`` here is the next
                            # round's freshly-mounted (empty) bubble, so
                            # mounting `before` it lands the line right after
                            # the round's tool result. The ``_build_metrics_markup``
                            # guard skips a dict that renders to nothing so we
                            # never mount an empty row.
                            # (tool-call ↔ output-token order)
                            round_line = RoundMetricsMessage(data)
                            anchor = self._current_assistant
                            try:
                                if anchor is not None and anchor.is_mounted:
                                    await transcript.mount(
                                        round_line,
                                        before=anchor,
                                    )
                                else:
                                    await transcript.mount(round_line)
                            except Exception:  # noqa: BLE001
                                try:
                                    await transcript.mount(round_line)
                                except Exception:  # noqa: BLE001
                                    pass
                            self._maybe_scroll_end(transcript)
                        self._round_pending_assistant = None
                    # Climb the `ctx X/Y` indicator after every round.
                    pt = data.get("prompt_tokens")
                    if isinstance(pt, (int, float)) and pt > 0:
                        self._last_prompt_tokens = int(pt)
                    ctx_window = data.get("context_window")
                    if isinstance(ctx_window, (int, float)) and ctx_window > 0:
                        self._last_context_window = int(ctx_window)
                    # Sync the goal widget from the engine's metrics — keeps the
                    # goal-bar correct when the teacher clears/overrides the goal,
                    # and lets a SECOND TUI on the same session replicate it (both
                    # clients receive the same usage_update metrics).
                    if "active_goal" in data:
                        self._active_goal = str(data.get("active_goal") or "")
                        self._recompute_teacher()
                    self._refresh_status_bar()
                elif event == "permission_request":
                    # Tool calls always come AFTER an assistant turn in the
                    # session_turn loop. Close the current round's
                    # AssistantMessage so the modal pops cleanly above the
                    # finalized turn — and so the tool widgets land in
                    # their own slot below.
                    _close_round_assistant()
                    await self._handle_permission_request(data)
                elif event == "tool_call":
                    tool_name = str(data.get("tool", "?"))
                    _incoming_cid = str(data.get("call_id") or "")
                    # Teacher-model review in flight → flip the footer to
                    # `teacher: reviewing…`. The engine emits this as a
                    # `delegate_teacher` tool node (title == the tool name);
                    # cleared in the matching tool_result branch below.
                    if tool_name == "delegate_teacher":
                        self._teacher_reviewing = True
                        if _incoming_cid:
                            self._teacher_review_cids.add(_incoming_cid)
                        self._refresh_status_bar()
                    # Predict whether this tool_call will be
                    # SUPPRESSED (inner subagent event hidden from
                    # the parent transcript) — if so, don't split the
                    # streaming widget. Splitting only to suppress
                    # leaves a useless empty `🔧→ (empty)` widget in
                    # the transcript. Only split when the new tool
                    # widget will actually mount BELOW it.
                    _will_suppress_inner = bool(
                        self._subagent_box_stack and _incoming_cid and _incoming_cid not in self._subagent_boxes_by_call
                    )
                    if not _will_suppress_inner:
                        # SPLIT BOUNDARY for any in-flight streaming
                        # ToolResultMessage. The new tool_call event
                        # is about to mount a ToolCallMessage at the
                        # END of the transcript. ``mark_split``
                        # freezes the streaming widget so the next
                        # delta mounts a fresh one BELOW the inner
                        # tool. See ToolResultMessage.mark_split.
                        for w in transcript.query(ToolResultMessage):
                            if getattr(w, "_streaming", False):
                                w.mark_split()
                    # Model-emitted ``todo`` tool: parse the args to
                    # update the bottom-of-TUI plan checklist. Two
                    # cases that mutate the visible list:
                    #   action=replace + items=[…]  → install a fresh
                    #       checklist of pending steps
                    #   action=add + items=[…]      → append to the
                    #       current list
                    #   action=complete + indexes=[…] → tick those
                    #       steps off
                    # The tool itself returns the canonical markdown
                    # checklist as ``content``; we re-parse from that
                    # in the matching ``tool_result`` branch so the
                    # rendered list always matches what the model
                    # sees on its side of the wire.
                    # Subagent-emitted TodoWrite must NOT leak into the
                    # parent session's `#plan-checklist` footer — the
                    # subagent's plan belongs to its own view, the
                    # parent has its own (separately-tracked) plan +
                    # the `#harness-tasks` footer. Gate on the
                    # subagent-box stack: any tool_call that arrives
                    # while a SubagentBoxMessage is on the stack is by
                    # definition an inner-subagent event.
                    if _is_todo_tool_name(tool_name) and not self._subagent_box_stack:
                        _todo_args = data.get("arguments") or {}
                        self._handle_todo_tool_call(_todo_args)
                        # Mirror the live TodoWrite into an inline
                        # ``TaskUpdateMessage`` card so the operator
                        # can scroll back to a previous plan state
                        # without losing it to a footer refresh. The
                        # card is collapsed-by-default (D1) and
                        # body-capped to 40% of screen (D2) so a
                        # 20-item TodoWrite no longer eats half the
                        # transcript. Mounted lazily on first
                        # TodoWrite; subsequent calls update in
                        # place via ``update_from_todos``.
                        _items: list[dict] = []
                        _adapted = adapt_todo_args(_todo_args)
                        if _adapted:
                            _items = list(_adapted.get("todos") or [])
                        if _items:
                            if self._latest_task_update_widget is None:
                                self._latest_task_update_widget = TaskUpdateMessage(_items)
                                try:
                                    await self._mount_target(transcript).mount(
                                        self._latest_task_update_widget,
                                    )
                                except Exception:  # noqa: BLE001
                                    self._latest_task_update_widget = None
                            elif self._latest_task_update_widget is not None:
                                try:
                                    self._latest_task_update_widget.update_from_todos(
                                        _items,
                                    )
                                except Exception:  # noqa: BLE001
                                    pass
                    # Plan-checklist cursor only advances on explicit
                    # `todo complete` actions handled above. The old
                    # "tick on every non-todo tool call" heuristic
                    # double-counted: a turn that fires 17 tool calls
                    # against a 4-step plan blew past the end and
                    # misled the operator into thinking the model was
                    # done while it was still on step 1.
                    # Capture the assistant BEFORE clearing the slot
                    # so we can await its own paint cycle below — the
                    # transcript-level ``wait_for_refresh`` further
                    # down waits for the SCROLL CONTAINER to repaint,
                    # which doesn't guarantee the child AssistantMessage
                    # has finished laying out to its new (taller)
                    # height. Under heavy / SSH-throttled paint, the
                    # tool widget can mount on top of a still-growing
                    # bubble and the tail prose stays clipped behind
                    # the new widget until an unrelated repaint.
                    _finalized_assistant = self._current_assistant
                    _close_round_assistant()
                    call_id = str(data.get("call_id") or "")
                    args_for_widget = data.get("arguments") or {}
                    # Subagent isolation — FAST PATH. When a
                    # SubagentBoxMessage is on the stack AND this
                    # call_id is NOT the outer Agent dispatch itself,
                    # this is an INNER tool fired by the subagent.
                    # Bail BEFORE the expensive ``wait_for_refresh``
                    # + worker-state update + widget-mount sequence:
                    # those events don't render anything in the
                    # parent's transcript (the SubagentBoxMessage
                    # header already shows live progress as
                    # ``N/M tools · last: <tool>``). Paying a paint
                    # cycle per inner event drags the parent's event
                    # loop and SERIALIZES sibling subagent boxes
                    # behind a backlog of suppressed inner refreshes
                    # — visible to the operator as the counter
                    # crawling 0/N → 1/N → … instead of jumping to
                    # 0/N immediately when N coders co-fire.
                    if self._subagent_box_stack and call_id and str(call_id) not in self._subagent_boxes_by_call:
                        try:
                            self._subagent_box_stack[-1].note_inner_call(
                                tool_name,
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        # Track the call_id so the matching
                        # tool_result branch also skips its widget
                        # mount AND bumps the done counter. Use the
                        # set's idempotency to avoid worrying about
                        # double-adds across the two add paths.
                        self._suppressed_subagent_calls.add(str(call_id))
                        # Still track in _pending_tool_calls so the
                        # parent's "round complete" detection works
                        # correctly when the inner call returns.
                        self._pending_tool_calls.add(str(call_id))
                        continue
                    # Parent-visible tool call: pay the paint cost
                    # + flip the footer to ``running tool: X``.
                    #
                    # Block until Textual has actually painted the
                    # just-finalized AssistantMessage at its grown size
                    # BEFORE we mount the tool widget below it. An
                    # earlier attempt used ``asyncio.sleep(0)`` to yield
                    # one event-loop tick, but under SSH-throttled FPS
                    # the compositor's paint timer hasn't fired by the
                    # time we get back — the tool widget then mounts on
                    # top of a still-laying-out assistant message and
                    # the prose before the tool call stays invisible
                    # for the entire (potentially long) tool run,
                    # reading as "agent stuck." ``wait_for_refresh``
                    # sets an ``asyncio.Event`` via
                    # ``call_after_refresh`` and awaits it, so we
                    # actually wait for a full paint cycle. A few extra
                    # ms here is fine; the bug it prevents looks like a
                    # hang.
                    #
                    # NOTE: must be called on a mounted widget, NOT on
                    # ``self`` (the App). Textual's ``MessagePump.
                    # wait_for_refresh`` asserts ``self._task is not
                    # None``, but on the App class ``_task`` is only
                    # set inside ``run_test`` — under normal
                    # ``run_async`` the App's message loop runs in
                    # ``run_process_messages`` without populating
                    # ``_task``, so ``await self.wait_for_refresh()``
                    # crashes every tool-call event with "Node must be
                    # running before calling wait_for_refresh".
                    # ``transcript`` is a started widget; its task is
                    # live, and ``call_after_refresh`` schedules onto
                    # the same app-level paint cycle either way.
                    #
                    # Wait on the CHILD AssistantMessage first so its
                    # post-finalize ``refresh(layout=True)`` (set in
                    # ``AssistantMessage.finalize`` at L583-600) has a
                    # chance to actually paint at its grown height,
                    # THEN wait on the transcript so the parent
                    # VerticalScroll re-flows around it. One wait on
                    # the parent alone misses the child's growth
                    # because Textual schedules child + parent refreshes
                    # independently — the parent can paint while the
                    # child is still pending layout.
                    if _finalized_assistant is not None:
                        try:
                            await _finalized_assistant.wait_for_refresh()
                        except Exception:  # noqa: BLE001
                            pass
                    await transcript.wait_for_refresh()
                    self._set_worker_state(
                        f"[bold yellow]🔧 running tool: {tool_name}[/bold yellow]",
                        detail="[dim]ESC to cancel[/dim]",
                    )
                    # Track this call as in-flight so the matching
                    # tool_result branch knows whether more parallel
                    # siblings are still pending before mounting the
                    # next-round AssistantMessage.
                    if call_id:
                        self._pending_tool_calls.add(call_id)
                    # Route inner mounts under a subagent box when one
                    # is on the stack (visual containment for
                    # delegate_* runs). Falls through to ``transcript``
                    # otherwise.
                    _mount = self._mount_target(transcript)
                    if tool_name == "ExitPlanMode":
                        plan_text = str(args_for_widget.get("plan") or "")
                        await _mount.mount(PlanMessage(plan_text))
                        if call_id:
                            self._plan_call_ids.add(call_id)
                    elif tool_name == "OutputSection":
                        section_name = str(args_for_widget.get("section_name") or "")
                        section_content = str(args_for_widget.get("content") or "")
                        await _mount.mount(OutputSectionMessage(section_name, section_content))
                        if call_id:
                            self._section_call_ids.add(call_id)
                    elif tool_name == "UserQuestion":
                        # Render the question + options as a prominent
                        # inline card; the matching tool_result is
                        # suppressed below (the runner's body is just a
                        # markdown echo of what the card already shows).
                        await _mount.mount(UserQuestionMessage(args_for_widget))
                        if call_id:
                            self._user_question_call_ids.add(call_id)
                    elif tool_name in ("Update", "Edit", "edit"):
                        # File edit → Claude Code's "Update" treatment:
                        # render the change as a colored code-diff card
                        # computed from the call's own args. The diff
                        # shows immediately, before the tool finishes;
                        # the matching tool_result just stamps it ✓/✗
                        # (see the `_update_call_ids` branch below).
                        await _mount.mount(
                            UpdateMessage(
                                str(args_for_widget.get("file_path") or ""),
                                str(args_for_widget.get("old_string") or ""),
                                str(args_for_widget.get("new_string") or ""),
                                replace_all=bool(args_for_widget.get("replace_all", False)),
                                call_id=call_id,
                            )
                        )
                        if call_id:
                            self._update_call_ids.add(call_id)
                    elif self._is_harness_task_tool(tool_name):
                        # Task* family — the persistent ``#harness-tasks``
                        # footer Static still carries the pinned list at
                        # the bottom of the TUI, AND an inline
                        # ``HarnessTaskListMessage`` card lands in the
                        # transcript so the operator can scroll back
                        # through the task history. The card is
                        # collapsible (MISSION.md D1 / D2) so a 22-task
                        # run no longer eats half the screen.
                        # ``TaskCreate``'s row needs the assigned id,
                        # which only arrives in the tool_result body;
                        # we stash the args here and apply them in the
                        # matching tool_result branch. ``TaskUpdate``
                        # is applied optimistically here so both the
                        # footer AND the inline card flip immediately
                        # on dispatch.
                        #
                        # Subagent gate: same logic as the TodoWrite
                        # branch above — a Task* fired by a subagent
                        # must not mutate the PARENT's footer / card.
                        # Skip the state update while a
                        # SubagentBoxMessage is on the stack (the
                        # subagent's own task list lives in its
                        # session view).
                        if not self._subagent_box_stack:
                            # MISSION.md D1 — "new task group" trigger.
                            # ``_send_user_turn`` arms ``_new_task_group_pending``
                            # at the start of every top-level user-prompt
                            # turn; the first ``TaskCreate`` of that turn
                            # consumes the flag (one-shot). When the prior
                            # group still has rows, drop them from the
                            # persistent ``#harness-tasks`` footer AND
                            # detach the inline card so a *fresh*
                            # ``HarnessTaskListMessage`` mounts below for
                            # the new group (the prior card stays in the
                            # transcript scrollback — operator can scroll
                            # back to see the history of group A).
                            #
                            # ``TaskUpdate`` / ``TaskList`` / ``TaskGet`` /
                            # ``TaskOutput`` / ``TaskStop`` do NOT consume
                            # the flag — they continue an existing group.
                            # Subagent-fired Task* calls were already
                            # gated out above (``not self._subagent_box_stack``).
                            if tool_name == "TaskCreate" and self._new_task_group_pending:
                                self._new_task_group_pending = False
                                if self._harness_tasks:
                                    self._clear_harness_tasks()
                                    # Detach the prior inline card so the
                                    # next mount-block below creates a
                                    # fresh one for the new group. The
                                    # OLD card stays in the transcript
                                    # at its scrollback position.
                                    self._latest_harness_task_widget = None
                            # Mount the inline card lazily on the first
                            # parent Task* call; subsequent calls reuse
                            # the same card via the cached ref so the
                            # transcript stays clean (no duplicate
                            # cards for a status nudge).
                            if self._latest_harness_task_widget is None:
                                self._latest_harness_task_widget = HarnessTaskListMessage()
                                try:
                                    await _mount.mount(
                                        self._latest_harness_task_widget,
                                    )
                                except Exception:  # noqa: BLE001
                                    self._latest_harness_task_widget = None
                            if tool_name == "TaskCreate":
                                if call_id:
                                    self._pending_task_create_calls[call_id] = (
                                        args_for_widget if isinstance(args_for_widget, dict) else {}
                                    )
                            elif tool_name == "TaskUpdate":
                                self._apply_task_update_args(args_for_widget)
                                # Mirror the update into the inline
                                # card so the operator sees both
                                # footer + transcript advance together.
                                if self._latest_harness_task_widget is not None and isinstance(args_for_widget, dict):
                                    _tid = str(args_for_widget.get("taskId") or "").strip()
                                    if _tid:
                                        try:
                                            self._latest_harness_task_widget.upsert(
                                                _tid,
                                                subject=args_for_widget.get("subject"),
                                                status=args_for_widget.get("status"),
                                            )
                                        except Exception:  # noqa: BLE001
                                            pass
                        if call_id:
                            self._harness_task_call_ids.add(call_id)
                    elif self._is_delegate_tool(tool_name):
                        # Subagent run → wrap inner widgets in a
                        # bordered container so the operator can tell
                        # the subagent's events from the parent's.
                        label, summary = self._delegate_summary(tool_name, args_for_widget)
                        box = SubagentBoxMessage(label, summary)
                        box.call_id = call_id
                        await _mount.mount(box)
                        if call_id:
                            self._subagent_boxes_by_call[call_id] = box
                            # Persist the full prompt/args to a per-call
                            # sidecar so the operator can re-read it
                            # later (and so the upcoming subagent-
                            # navigation feature can list it). The
                            # matching tool_result branch appends the
                            # result + elapsed time to the same file.
                            self._persist_subagent_call(
                                call_id,
                                tool_name,
                                args_for_widget,
                            )
                        self._subagent_box_stack.append(box)
                        # Bump the pinned ``#subagent-counter`` footer
                        # so the operator sees the fan-out total grow
                        # in real time. The matching tool_result
                        # branch bumps done / failed.
                        self._subagent_total += 1
                        self._render_subagent_counter()
                        # Still mount the generic announcement pill
                        # INSIDE the box so the operator can see the
                        # full args (prompt, model, etc.) collapsibly.
                        await box.body.mount(ToolCallMessage(tool_name, args_for_widget))
                    else:
                        # Always mount the inline 🔧 announcement widget
                        # so the user can see WHICH tool is running and
                        # with what args. Operators were reporting "the
                        # status bar shows exec but nothing's on the
                        # transcript" — that was this gate hiding the
                        # call until the result arrived. The widget is
                        # collapsed-by-default so it doesn't dominate
                        # the transcript; expand with Ctrl+T.
                        await _mount.mount(ToolCallMessage(tool_name, args_for_widget))
                    self._maybe_scroll_end(transcript)
                elif event == "tool_delta":
                    # Streaming-capable tool (today: `delegate_claude`)
                    # is producing output incrementally. Find or mount
                    # a ToolResultMessage in streaming mode and append
                    # the chunk so the user sees Claude's text arrive
                    # token-by-token, same UX as the slash command.
                    #
                    # NOTE the ``and w._streaming`` filter: previously
                    # this match was ``call_id`` only, which meant
                    # split widgets (frozen by mark_split when an inner
                    # tool_call interrupted) would be re-found and
                    # appended to. With the filter, split widgets are
                    # bypassed and the next delta mounts a fresh
                    # streaming widget at the END of the transcript —
                    # below whatever inner tool widgets just mounted.
                    call_id = data.get("call_id", "")
                    txt = data.get("text") or ""
                    if not txt or not call_id:
                        continue
                    # Drop tool_delta chunks for inner subagent calls —
                    # we've suppressed the call's widget so there's no
                    # bubble to stream into anyway.
                    if str(call_id) in self._suppressed_subagent_calls:
                        continue
                    streaming = next(
                        (w for w in transcript.query(ToolResultMessage) if w.call_id == call_id and w._streaming),
                        None,
                    )
                    if streaming is None:
                        streaming = ToolResultMessage(
                            call_id,
                            ok=True,
                            content="",
                            streaming=True,
                        )
                        # Codex round-3 D2: the SubagentBoxMessage for a
                        # delegate_* call mounts at ``box`` (transcript
                        # position N); historically the streaming
                        # ToolResultMessage mounted directly to
                        # ``transcript`` at position N+1, leaving the
                        # box visually empty in the middle while the
                        # output rendered BELOW it AND the terminal
                        # ToolResultMessage at the end landed back
                        # INSIDE the box (above the streaming body).
                        # Route the streaming widget through
                        # ``_mount_target`` so when a subagent box is
                        # on the stack it nests INSIDE ``box.body`` —
                        # the order becomes:
                        #   box header → streaming body → terminal ✅
                        # all inside the same container, matching the
                        # User → Header → Body → Result contract for
                        # delegate-class tools (incl. ``delegate_agent``).
                        await self._mount_target(transcript).mount(streaming)
                    streaming.append_streaming(txt)
                    self._maybe_scroll_end(transcript)
                elif event == "subagent_result":
                    cid = str(data.get("call_id") or "")
                    outcome = str(data.get("status") or "")
                    ok = bool(data.get("ok"))
                    if cid:
                        self._persist_subagent_result(cid, ok, str(data.get("content") or ""), outcome=outcome)
                        for child_box in transcript.query(SubagentBoxMessage):
                            if getattr(child_box, "call_id", None) == cid:
                                child_box.finalize(ok=ok, outcome=outcome)
                elif event == "tool_result":
                    # Reuse a streaming widget if one was mounted via
                    # `tool_delta` for this call_id; otherwise mount a
                    # fresh ToolResultMessage with the full content.
                    call_id = data.get("call_id", "")
                    ok = bool(data.get("ok", True))
                    content = data.get("content", "")
                    outcome = data.get("outcome")
                    # Teacher review finished → clear the `reviewing…` footer
                    # once all in-flight reviews for this turn have resolved.
                    if str(call_id) in self._teacher_review_cids:
                        self._teacher_review_cids.discard(str(call_id))
                        if not self._teacher_review_cids:
                            self._teacher_reviewing = False
                        self._refresh_status_bar()
                    # Inner subagent tool whose widget mount we
                    # suppressed at tool_call time: don't mount the
                    # ToolResultMessage either. Bump the subagent
                    # box's done counter so the header progress
                    # advances, then drop the housekeeping (pending
                    # tool set, worker state, next-round assistant
                    # gate) so the parent turn flows the same as
                    # always.
                    if str(call_id) in self._suppressed_subagent_calls:
                        self._suppressed_subagent_calls.discard(str(call_id))
                        try:
                            if self._subagent_box_stack:
                                self._subagent_box_stack[-1].note_inner_done()
                        except Exception:  # noqa: BLE001
                            pass
                        self._pending_tool_calls.discard(str(call_id or ""))
                        self._set_worker_state(
                            "[bold green]▶ streaming…[/bold green]",
                            detail="[dim]ESC to cancel[/dim]",
                        )
                        if not self._pending_tool_calls:
                            await _ensure_assistant()
                        self._maybe_scroll_end(transcript)
                        continue
                    # ExitPlanMode was rendered as a PlanMessage at
                    # tool_call time; the result body is just "Plan
                    # ready — confirm…\n\n<plan>" which duplicates the
                    # plan card. Skip the ToolResultMessage mount but
                    # keep the housekeeping (pending-set, worker state,
                    # next-round assistant) so the rest of the turn
                    # flows unchanged.
                    if str(call_id) in self._plan_call_ids:
                        self._plan_call_ids.discard(str(call_id))
                        self._pending_tool_calls.discard(str(call_id or ""))
                        self._set_worker_state(
                            "[bold green]▶ streaming…[/bold green]",
                            detail="[dim]ESC to cancel[/dim]",
                        )
                        if not self._pending_tool_calls:
                            await _ensure_assistant()
                        self._maybe_scroll_end(transcript)
                        continue
                    # OutputSection: on success the result body is the
                    # "Section X/N saved, next: Y" nudge meant for the
                    # model — we already rendered the section content
                    # as an OutputSectionMessage at tool_call time, so
                    # suppress the redundant Collapsible. On FAILURE
                    # fall through to the standard ToolResultMessage so
                    # the operator sees the validation error.
                    if str(call_id) in self._section_call_ids:
                        self._section_call_ids.discard(str(call_id))
                        if ok:
                            self._pending_tool_calls.discard(str(call_id or ""))
                            self._set_worker_state(
                                "[bold green]▶ streaming…[/bold green]",
                                detail="[dim]ESC to cancel[/dim]",
                            )
                            if not self._pending_tool_calls:
                                await _ensure_assistant()
                            self._maybe_scroll_end(transcript)
                            continue
                    # UserQuestion: the markdown echo the runner returns
                    # is exactly what the UserQuestionMessage card
                    # already shows — suppress the Collapsible on
                    # success. On FAILURE (malformed args) fall through
                    # to the standard ToolResultMessage so the operator
                    # sees the validation error.
                    if str(call_id) in self._user_question_call_ids:
                        self._user_question_call_ids.discard(str(call_id))
                        if ok:
                            self._pending_tool_calls.discard(str(call_id or ""))
                            self._set_worker_state(
                                "[bold green]▶ streaming…[/bold green]",
                                detail="[dim]ESC to cancel[/dim]",
                            )
                            if not self._pending_tool_calls:
                                await _ensure_assistant()
                            self._maybe_scroll_end(transcript)
                            continue
                    # Update/Edit: the diff was already rendered as an
                    # UpdateMessage card at tool_call time. Stamp it
                    # ✓/✗ via `mark_result` and — on success — skip the
                    # redundant ToolResultMessage (the result body is
                    # just a one-line "updated <path>" summary). On
                    # FAILURE fall through so the operator sees the
                    # error text (old_string-not-found, etc.).
                    if str(call_id) in self._update_call_ids:
                        self._update_call_ids.discard(str(call_id))
                        card = next(
                            (w for w in transcript.query(UpdateMessage) if w.call_id == call_id),
                            None,
                        )
                        if card is not None:
                            card.mark_result(ok)
                        if ok:
                            self._pending_tool_calls.discard(str(call_id or ""))
                            self._set_worker_state(
                                "[bold green]▶ streaming…[/bold green]",
                                detail="[dim]ESC to cancel[/dim]",
                            )
                            if not self._pending_tool_calls:
                                await _ensure_assistant()
                            self._maybe_scroll_end(transcript)
                            continue
                    # Harness Task* result: apply to the inline task
                    # card. TaskCreate's id comes from the result body;
                    # TaskUpdate was applied optimistically at call
                    # time. TaskList's result body is a JSON snapshot
                    # of the whole task store — fold it into the
                    # inline ``HarnessTaskListMessage`` via
                    # ``apply_tasklist_payload`` so a ``TaskList``
                    # response actually refreshes the transcript card.
                    # Suppress the
                    # redundant Collapsible on success (the card
                    # already shows the rows); on failure fall through
                    # so the error text is visible.
                    if str(call_id) in self._harness_task_call_ids:
                        self._harness_task_call_ids.discard(str(call_id))
                        if str(call_id) in self._pending_task_create_calls:
                            create_args = self._pending_task_create_calls.pop(str(call_id))
                            self._apply_task_create_from_result(create_args, str(content or ""))
                        # TaskList / TaskGet result body — parse the
                        # plain-text rows the harness emits (see
                        # ``sharedagents_core/tools/runners.py:_task_summary``
                        # which renders each row as
                        # ``#<id> [<status>] <subject> owner=X
                        # blockedBy=...``). The runner does NOT return
                        # JSON, so the TUI parses the text into
                        # ``apply_tasklist_payload``-compatible dicts.
                        # Best-effort: a body that doesn't match
                        # leaves the card untouched.
                        if ok and self._latest_harness_task_widget is not None and content:
                            tasks_list = _parse_task_list_text(content)
                            if tasks_list:
                                try:
                                    self._latest_harness_task_widget.apply_tasklist_payload(
                                        tasks_list,
                                    )
                                except Exception:  # noqa: BLE001
                                    pass
                        if ok:
                            self._pending_tool_calls.discard(str(call_id or ""))
                            self._set_worker_state(
                                "[bold green]▶ streaming…[/bold green]",
                                detail="[dim]ESC to cancel[/dim]",
                            )
                            if not self._pending_tool_calls:
                                await _ensure_assistant()
                            self._maybe_scroll_end(transcript)
                            continue
                    # Subagent (delegate_*) result: finalize the box,
                    # pop from the containment stack so subsequent
                    # parent-agent widgets mount in the transcript
                    # again. The subagent's own final content text was
                    # already streamed into widgets INSIDE the box; we
                    # still let the standard Collapsible mount below
                    # so the full final result is foldable.
                    # Append result + elapsed to the sidecar so the
                    # per-subagent log is complete. Done OUTSIDE the
                    # in-memory box guard below: a reattach / restart
                    # drops ``_subagent_boxes_by_call`` but the on-disk
                    # sidecar must still be flipped out of "running" or it
                    # leaks as a phantom forever. No-ops when this call_id
                    # has no subagent sidecar (the path-exists guard).
                    self._persist_subagent_result(
                        str(call_id),
                        ok,
                        str(content or ""),
                        outcome=outcome,
                    )
                    if str(call_id) in self._subagent_boxes_by_call:
                        box = self._subagent_boxes_by_call.pop(str(call_id))
                        try:
                            self._subagent_box_stack.remove(box)
                        except ValueError:
                            pass
                        box.finalize(ok=ok, outcome=outcome)
                        # Bump the pinned ``#subagent-counter`` footer
                        # so the operator sees ``N/M done`` tick up
                        # even when the finalized box has scrolled off
                        # the viewport.
                        if ok:
                            self._subagent_done += 1
                        elif outcome not in ("cancelled", "refused"):
                            self._subagent_failed += 1
                        self._render_subagent_counter()
                        # The result Collapsible should also land
                        # INSIDE the box so it stays visually grouped
                        # with the subagent's run.
                        if ok and not (content or "").strip():
                            self._pending_tool_calls.discard(str(call_id or ""))
                            self._set_worker_state(
                                "[bold green]▶ streaming…[/bold green]",
                                detail="[dim]ESC to cancel[/dim]",
                            )
                            if not self._pending_tool_calls:
                                await _ensure_assistant()
                            self._maybe_scroll_end(transcript)
                            continue
                        await box.body.mount(ToolResultMessage(call_id, ok, content, outcome=outcome))
                        self._pending_tool_calls.discard(str(call_id or ""))
                        self._set_worker_state(
                            "[bold green]▶ streaming…[/bold green]",
                            detail="[dim]ESC to cancel[/dim]",
                        )
                        if not self._pending_tool_calls:
                            await _ensure_assistant()
                        self._maybe_scroll_end(transcript)
                        continue
                    streaming = next(
                        (w for w in transcript.query(ToolResultMessage) if w.call_id == call_id and w._streaming),
                        None,
                    )
                    # Did we previously split this call's stream? If so,
                    # the streamed segments together ARE the
                    # authoritative content. The agent's final
                    # ``content`` field would duplicate the prior
                    # segments' text into the LAST one, re-showing
                    # everything that already rendered above the inner
                    # tool widgets. Pass content=None to keep streamed
                    # text and only stamp the ✅/❌ marker.
                    prior_splits = [
                        w
                        for w in transcript.query(ToolResultMessage)
                        if w.call_id == call_id and getattr(w, "_split", False)
                    ]
                    if streaming is not None:
                        if prior_splits:
                            streaming.finalize_streaming(ok=ok, content=None, outcome=outcome)
                        else:
                            streaming.finalize_streaming(ok=ok, content=content, outcome=outcome)
                    elif prior_splits:
                        # All segments were split (inner tool came right
                        # at the end with no trailing delta). Just
                        # stamp the FINAL ok marker on the LAST split
                        # widget — don't re-show the body.
                        last = prior_splits[-1]
                        last.ok = ok
                        last._prefix = "✅" if ok else "❌"
                        last._peek = last._build_peek()
                        last.title = last._build_title(collapsed=last.collapsed)
                    else:
                        await transcript.mount(ToolResultMessage(call_id, ok, content, outcome=outcome))
                    # Drop this call from the in-flight set. With
                    # parallel tool calls (one round, N tool_call
                    # events, then N tool_result events), if we mounted
                    # the next-round AssistantMessage now, remaining
                    # parallel results would land BELOW it and the
                    # next round's reasoning/content would render in
                    # the prematurely-mounted widget — visually the
                    # answer would appear ABOVE the still-pending tool
                    # widgets. Only re-arm the assistant once the LAST
                    # parallel sibling has resolved.
                    self._pending_tool_calls.discard(str(call_id or ""))
                    # Tool finished — model is about to re-prefill on the
                    # appended tool_result message and start streaming the
                    # next round. Flip the worker-state cue back to
                    # streaming so the user sees "still alive, model now"
                    # instead of leaving a stale "running tool" line up.
                    self._set_worker_state(
                        "[bold green]▶ streaming…[/bold green]",
                        detail="[dim]ESC to cancel[/dim]",
                    )
                    # Eagerly mount the next round's AssistantMessage so
                    # the prefill counter is visible during the upstream
                    # model's re-prefill on the just-grown chat history
                    # (tool result was just injected). Same fluidity goal
                    # as the eager mount before the SSE loop, but for
                    # round 2+ in a tool-use turn. Skipped while parallel
                    # siblings are still pending — see comment above.
                    if not self._pending_tool_calls:
                        await _ensure_assistant()
                    self._maybe_scroll_end(transcript)
                elif event == "auto_continue":
                    # Local-inference models often hit `finish_reason=length`
                    # mid-answer or run out of tool-call rounds in plan/ask.
                    # The agent auto-injects a continue prompt and runs
                    # another round. Close the current assistant so the
                    # next round mounts its own widget, and surface a
                    # status line so the user sees what's happening.
                    _close_round_assistant()
                    reason = data.get("reason", "?")
                    count = data.get("count", "?")
                    self._set_status(f"auto-continue: {reason} (#{count})")
                    # Premature EOS on a tool-followup round: surface a
                    # persistent inline alert so the user knows the
                    # upstream model glitched (qwen3-fp8 EOS-too-early
                    # bug) and the agent is retrying. Status bar fires
                    # for every auto_continue, but this gets a transcript
                    # widget so it's visible in scrollback.
                    if reason == "empty_stop":
                        try:
                            max_retries = int(data.get("max") or 0)
                        except (TypeError, ValueError):
                            max_retries = 0
                        try:
                            count_int = int(count)
                        except (TypeError, ValueError):
                            count_int = 0
                        try:
                            round_idx = int(data.get("round") or 0)
                        except (TypeError, ValueError):
                            round_idx = 0
                        await transcript.mount(
                            EmptyEosNotice(
                                count_int,
                                max_retries,
                                round_idx,
                            )
                        )
                        self._maybe_scroll_end(transcript)
                        self._set_status(f"premature EOS on tool call · retry {count_int}/{max_retries}")
                    # Compaction events (proactive `compacted` and
                    # post-overflow `compacted_after_overflow`) get a
                    # persistent inline notice so the user can tell at
                    # a glance — and after-the-fact when scrolling back —
                    # that some tool results were trimmed/dropped on the
                    # wire to the upstream model. The transcript widgets
                    # for those calls also get a "🗜️ upstream-…" badge.
                    if reason in ("compacted", "compacted_after_overflow"):
                        await transcript.mount(CompactionNotice(reason, data))
                        self._maybe_scroll_end(transcript)
                        # Update `ctx X/Y` immediately so the user sees
                        # the post-compaction prompt size without having
                        # to wait for the next round's metrics event.
                        new_pt = data.get("new_prompt_tokens")
                        if isinstance(new_pt, (int, float)) and new_pt > 0:
                            self._last_prompt_tokens = int(new_pt)
                            self._refresh_status_bar()
                        truncated = set(data.get("truncated_tool_call_ids") or [])
                        dropped = set(data.get("dropped_tool_call_ids") or [])
                        if truncated or dropped:
                            for w in transcript.query(ToolResultMessage):
                                if w.call_id in dropped:
                                    w.mark_compacted("dropped")
                                elif w.call_id in truncated:
                                    w.mark_compacted("truncated")
                    # Mount the next round's AssistantMessage now so the
                    # prefill counter ticks while the agent retries the
                    # upstream model. Same rationale as post-tool_result.
                    await _ensure_assistant()
                elif event == "error":
                    worker_error = data.get("message")
                elif event == "done":
                    # `state == "aborted"` was previously a no-op and
                    # left the user wondering why their turn just
                    # stopped. Surface the reason: ESC ("aborted") gets
                    # a soft message; tool_runner crashes carry a real
                    # error string in `data["error"]` and route through
                    # the same banner path as other errors.
                    state = data.get("state")
                    err = data.get("error")
                    if state == "aborted" and err:
                        worker_error = err
                    elif state == "aborted":
                        worker_error = "turn aborted"
                    elif state == "error" and err:
                        worker_error = err
        except Exception as e:  # noqa: BLE001
            worker_error = str(e) or repr(e)
        finally:
            self._turn_render_frozen = False
            ended_tid = self._current_turn_id
            self._current_turn_id = None
            # Clear the always-visible worker indicator now that we're done.
            self._set_worker_state("")
            self._tui_log(
                "stream_end",
                tid=ended_tid,
                error=(worker_error or "")[:200] or None,
            )

        # End-of-turn: close whatever's still open. The CUMULATIVE turn
        # stats are stamped here, from the TurnDone's usage_update event
        # captured in ``worker_metrics``. Each round's PER-ROUND stats
        # were already stamped on their own bubble as their RoundComplete
        # usage_update arrived (see the ``event == "metrics"`` handler) —
        # so a multi-round turn reads as "per-round line on every tool
        # round + turn-total line on the final bubble".
        if worker_error:
            if self._current_assistant is not None:
                self._current_assistant.mark_error(worker_error)
            self._set_status(f"error: {worker_error[:80]}")
        else:
            # Stamp the cumulative turn metrics on the last round's
            # assistant. For single-round turns ``worker_metrics``
            # carries the full (and only) round's stats. For multi-round
            # turns it carries the cumulative ``_turn_metrics`` from
            # ``TurnDone`` — which sums ``completion_tokens`` across
            # all rounds while keeping the last (cumulative) round's
            # ``prompt_tokens``.
            if worker_metrics:
                target = self._current_assistant
                if target is None or not (target.content_text or target.thinking_text):
                    # The active bubble is empty (turn ended on a tool-only
                    # round). The renderer stamps the cumulative turn metrics
                    # on the LAST assistant SEGMENT — a non-empty generation
                    # — never on an empty trailing bubble it never emits.
                    # Match that: route the total to the last NON-EMPTY
                    # bubble. If the whole turn produced no assistant
                    # text/thinking, the renderer shows no metrics line at all,
                    # so we mount no stray fallback bubble either.
                    # (turn-replication: tool-only-round parity)
                    target = next(
                        (b for b in reversed(self._turn_assistants) if b.content_text or b.thinking_text),
                        None,
                    )
                if target is not None:
                    target.set_metrics(worker_metrics)
            _close_round_assistant()
            self._set_status("")

        # The turn is fully drained; drop any per-round pending reference
        # so it can't carry across turns (it's re-armed per round, and
        # reset again at the next turn's start).
        self._round_pending_assistant = None

        # Reconcile any subagent sidecar still marked "running": the turn
        # is over, so no further tool_result will arrive for it. Without
        # this an aborted / interrupted subagent leaks at "running"
        # forever (the live result-write path never fires).
        self._reconcile_orphaned_subagents("aborted")

        # Refresh local session view from server
        try:
            data = await self._client.get_session(self._sid)
            self.session = Session(
                hash=data["hash"],
                title=data["title"],
                created_at=data["created_at"],
                system_prompt=data.get("system_prompt", ""),
                model=data["model"],
                endpoint=data["endpoint"],
                engine_model=data.get("engine_model") or "",
                turns=list(data.get("turns") or []),
                context_window=int(data.get("context_window") or 0),
            )
            self._refresh_title()
            # Repaint the footer from the freshly-reloaded session: this
            # is the turn that first SET (or CLEARED) ``engine_model`` for
            # an external CLI/ACP agent, and the model segment reads
            # ``self.session.engine_model``. The earlier ``finally`` refresh
            # ran BEFORE this ``get_session``, so without an explicit repaint
            # here the footer would keep showing the pre-turn label
            # (``claude`` / alias-only antigravity) until some unrelated
            # event triggered another refresh (post-turn footer model
            # staleness).
            try:
                self._refresh_status_bar()
            except Exception:  # noqa: BLE001
                pass
            # Reconcile THIS turn's thinking titles to the canonical
            # folded values. The driver's live "Thought for Xs" is
            # measured as deltas arrive and gets inflated by SSH-throttled
            # paint cadence, so it disagrees with the reader's persisted
            # ``thinking_ms`` (a thinking_delta ts-span) that a passive
            # frontend replays. Stamp the folded value onto each live
            # bubble (in mount order, mirroring the reader's per-generation
            # segmentation) so the driver's finalized titles equal the
            # renderer's exactly. (turn-replication: thinking-duration parity)
            try:
                _completed = [_t for _t in (data.get("turns") or []) if not _t.get("in_progress")]
                if _completed and self._turn_assistants:
                    _segs = [
                        _b for _b in (_completed[-1].get("render_segments") or []) if _b.get("kind") == "assistant"
                    ]
                    # The reader opens an assistant segment ONLY on a
                    # text/thinking delta, so it has no counterpart for
                    # the driver's EMPTY bubbles — the eager mount before
                    # the SSE loop and after each tool_result, plus the
                    # end-of-turn metrics-fallback bubble. A tool-first
                    # round leaves such an empty bubble at the FRONT/middle
                    # of ``_turn_assistants``, so a naive positional zip
                    # would pair it with the first real segment and shift
                    # every following pairing (mounting a thinking box on
                    # the wrong bubble). Drop empty bubbles first: a
                    # non-empty live bubble (got content or thinking) maps
                    # 1:1, in order, to a reader assistant segment.
                    _real = [_b for _b in self._turn_assistants if (_b.content_text or _b.thinking_text)]
                    # Only reconcile when the counts line up — if a rare
                    # boundary case (e.g. late-chunk bubble reuse) desyncs
                    # them, skip rather than risk mis-stamping; the live
                    # bubbles simply keep their own values.
                    if len(_real) == len(_segs):
                        for _bubble, _seg in zip(_real, _segs):
                            await _bubble.apply_canonical_thinking(
                                _seg.get("thinking") or "",
                                _seg.get("thinking_ms"),
                                _seg.get("reasoning_chars"),
                            )
                # Hide any bubble that finalized EMPTY after reconcile (no
                # content, no thinking, no metrics) — the eager mount for a
                # pure tool-only round. The reader opens no assistant
                # segment for a deltaless round, so the renderer mounts NO
                # widget; an empty AssistantMessage would otherwise leave a
                # margin-only blank row (`.assist-msg { margin: 1 0 0 0 }`)
                # on the driver alone. `display = False` drops it from
                # layout (0 rows), matching the renderer. Runs AFTER reconcile
                # so a bubble that just had its canonical thinking box
                # mounted (live stream missed the reasoning) is non-empty
                # and kept. (turn-replication: tool-only-round parity)
                for _b in self._turn_assistants:
                    try:
                        if not (_b.content_text or _b.thinking_text):
                            _b.display = False
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

        # Drain one queued entry, if any. Called after the session-view
        # refresh so the next turn starts from up-to-date state. The
        # cross-frontend renderer doesn't own the local turn queue, so it
        # must not pop from it.
        if not from_log:
            self._maybe_start_next()
