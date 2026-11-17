"""AgentChatApp session load/save/switch/replay/render.

Extracted from app.py (de-monolith); behavior-preserving.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from textual.containers import VerticalScroll
from textual.widgets import Static

from sharedagents_agent.tools.widget_adapt import is_todo_tool_name as _is_todo_tool_name
from sharedagents_session.session import Session
from sharedagents_tui.helpers import (
    _build_metrics_markup,
    _default_tool_result_collapsed,
    _parse_runtime_warning_chunk,
    _strip_inline_tool_markers,
    humanise_delegate_marker,
    user_prompt_text,
)
from sharedagents_tui.widgets.input import (
    MessageInput,
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
    ToolCallMessage,
    ToolResultMessage,
    UpdateMessage,
    UserMessage,
    UserQuestionMessage,
)
from sharedagents_tui.widgets.modals import (
    SessionPickerModal,
)
from sharedagents_tui.widgets.notices import (
    CompactionNotice,
    RuntimeWarningNotice,
)


class SessionMixin:
    """AgentChatApp session load/save/switch/replay/render."""

    @staticmethod
    def _pick_most_recent_session_hash(sessions: list[dict]) -> str | None:
        """Return the hash of the session with the newest ``last_activity``."""
        if not sessions:
            return None
        best = max(
            sessions,
            key=lambda s: str(s.get("last_activity") or ""),
        )
        return (best.get("hash") or best.get("sid") or "").strip() or None

    async def _bind_session_view(
        self,
        data: dict,
        *,
        replay: bool = False,
        clear_harness: bool = False,
    ) -> None:
        """Apply a ``get_session`` payload to local TUI state."""
        sid = str(data.get("hash") or "")
        self._sid = sid or None
        self.session = Session(
            hash=sid,
            title=data.get("title") or "",
            created_at=data.get("created_at") or "",
            system_prompt=data.get("system_prompt") or "",
            model=data.get("model") or "",
            engine_model=data.get("engine_model") or "",
            endpoint=data.get("endpoint") or "",
            turns=list(data.get("turns") or []),
            context_window=int(data.get("context_window") or 0),
        )
        # Re-anchor the cross-frontend renderer's high-water mark to the
        # turns just bound: completed turns here are already rendered, so
        # the renderer must not re-render them. Bumping the generation bounces
        # the renderer's tail onto the new session (see `_render_session_log`).
        self._rendered_turn_count = sum(1 for _t in self.session.turns if not _t.get("in_progress"))
        # Offset anchor: everything already in the log is on screen (replayed),
        # so the renderer only renders turns appended AFTER this point. Keyed on
        # byte offset — not turn_idx — so a later burst of same-idx queued
        # turns each still cross the threshold. ``get_session`` read the turns
        # from this same log a moment ago, so the end offset is the matching
        # boundary (a turn landing in the sub-ms gap streams in live as normal).
        self._rendered_offset = self._session_log_end_offset(sid)
        self._render_generation = getattr(self, "_render_generation", 0) + 1
        # Idle SSE heartbeats are swallowed inside subscribe_events, so the
        # renderer's ``async for`` never wakes to see the generation bump
        # after ``/new`` / ``/sessions``. Bounce the worker onto this sid now.
        self._bounce_session_render()
        if clear_harness:
            try:
                self._clear_harness_tasks()
            except Exception:  # noqa: BLE001
                pass
        self._refresh_title()
        if replay:
            # Drop the PREVIOUS session's plan before re-rendering: the
            # replay re-applies THIS session's own TodoWrite calls
            # (`_render_one_saved_turn` → `_handle_todo_tool_call`), so the
            # footer rebuilds to exactly what the driving frontend shows.
            # Only on the replay path — a no-replay refresh of the current
            # session must NOT clear (nothing would rebuild it).
            # (turn-replication: plan-footer parity on attach/resume)
            try:
                self.clear_plan_checklist()
            except Exception:  # noqa: BLE001
                pass
            # Same for the harness-task footer: an attach/resume that
            # replays must not retain the PREVIOUS session's Task* rows
            # (the `/sessions`-switch path already clears them). Replay
            # doesn't rebuild harness rows, but a stale cross-session
            # footer is worse than an empty one. (turn-replication)
            try:
                self._clear_harness_tasks()
            except Exception:  # noqa: BLE001
                pass
            await self._replay_session_into_transcript(data)
        # Sync the footer ``ctx used/window`` from the bound session's
        # last completed turn so a second TUI shows the same token count
        # as the driver instead of a stale/empty ``ctx —``.
        # (turn-replication: footer ctx parity on attach/resume)
        _bound_completed = [_t for _t in self.session.turns if not _t.get("in_progress")]
        if _bound_completed:
            self._last_prompt_tokens = None
            self._last_context_window = None
            self._sync_footer_from_turn_metrics(_bound_completed[-1].get("metrics") or {})
        upstream_error = data.get("upstream_error")
        if upstream_error:
            self._upstream_warn = f"upstream warn: {upstream_error[:60]}"
        else:
            self._upstream_warn = None
        # Replicate the session's active goal so a SECOND TUI on the same
        # session shows the same goal-bar on attach/resume/switch/refresh — the
        # goal lives in the session header (``active_goal``) which get_session
        # returns. (During a live turn the per-round metrics keep it in sync;
        # this covers binding to a session that already has a goal.)
        self._active_goal = str(data.get("active_goal") or "")
        # Replicate the per-session teacher override + model so attaching to a
        # session that had ``/teacher on`` (or ``/teacher model <alias>``)
        # restores the widget — the engine persists these to the header.
        self._teacher_override = str(data.get("teacher_override") or "")
        self._teacher_model_override = str(data.get("teacher_model_override") or "")
        self._teacher_reviewing = False
        self._turn_state = ""
        self._recompute_teacher()
        self._refresh_status_bar()
        # Replay mounts many transcript widgets; keep the composition box
        # focused so keystrokes land in the input rather than a stray
        # focusable transcript card.
        try:
            self.query_one(MessageInput).focus()
        except Exception:  # noqa: BLE001
            pass

    async def _try_resume_last_session(self, *, replay: bool) -> bool:
        """Attach to the most recently active session, if any."""
        try:
            sessions = await self._client.list_sessions()
        except Exception:  # noqa: BLE001
            return False
        last_hash = self._pick_most_recent_session_hash(sessions)
        if not last_hash:
            return False
        try:
            await self._client.attach_session(last_hash, cwd=os.getcwd())
            data = await self._client.get_session(last_hash)
        except Exception:  # noqa: BLE001
            return False
        await self._bind_session_view(data, replay=replay)
        title = (data.get("title") or "(untitled)").strip()
        n = len(data.get("turns") or [])
        msg = f"resumed [{last_hash[:12]}] — {title}"
        if replay and n:
            msg += f" ({n} turn(s) replayed)"
        self._set_status(msg)
        return True

    async def _ensure_session(
        self,
        *,
        initial: bool = False,
        force_new: bool = False,
    ) -> bool:
        """Ensure an active session exists.

        By default re-attaches to the most recently active session for
        this agent (``session/load`` + transcript replay on startup).
        Pass ``force_new=True`` (``/new``) to always ``session/new``.
        """
        if self._sid is not None:
            return True
        assert self._client is not None
        if not force_new and await self._try_resume_last_session(replay=initial):
            return True
        try:
            # Pass the TUI's launch directory so the agent injects a
            # `# WORKSPACE-CONTEXT` block at the tail of the system
            # prompt — model knows which folder the user is in.
            # Pass the resolved `--endpoint` override (when set) so the
            # daemon spawns workers against the alternate upstream.
            sess = await self._client.create_session(
                workspace_cwd=os.getcwd(),
                workspace_origin="tui",
                endpoint=self._endpoint_override,
            )
        except Exception as e:  # noqa: BLE001
            self.session = Session.new(
                system_prompt="",
                model="",
                endpoint=self.agent_cfg.url,
            )
            self._refresh_title()
            self._upstream_warn = None
            self._set_status(f"session create failed: {e}")
            return False
        data = {
            "hash": sess["hash"],
            "title": sess.get("title") or "",
            "created_at": sess.get("created_at") or "",
            "system_prompt": sess.get("system_prompt") or "",
            "model": sess.get("model") or "",
            "endpoint": sess.get("endpoint") or "",
            "turns": [],
            "context_window": int(sess.get("context_window") or 0),
            "upstream_error": sess.get("upstream_error"),
        }
        await self._bind_session_view(data, clear_harness=True)
        upstream_error = sess.get("upstream_error")
        if upstream_error:
            self._set_status(self._upstream_warn or f"new session [{self._sid[:12]}]")
        else:
            self._set_status(f"new session [{self._sid[:12]}]")
        return True

    async def _replay_session_into_transcript(self, data: dict) -> int:
        """Render the saved turns of `data` (a session JSON dict) into
        the transcript so the user can scroll back through prior history
        after a `/sessions` switch. Returns the number of turns replayed.

        The session JSON only retains AGGREGATE per-turn `thinking` /
        `content` (not per-round), so the replay can't reproduce the
        exact live interleaving of "round 0 reasoning → tool_call →
        tool_result → round 1 reasoning". Instead we render:

            ❯ user
            🔧 tool_call₁  /  ✅ tool_result₁   (per round, in order)
            🔧 tool_call₂  /  ✅ tool_result₂
            …
            <assistant content + collapsible thinking + metrics>

        which is enough for the user to see what happened and pick up
        the conversation. Turns from the `/claude` bridge are tagged
        via `metrics.source == "claude"` and render as a `ClaudeMessage`
        instead of `AssistantMessage` so attribution stays clear.
        """
        transcript = self.query_one("#transcript", VerticalScroll)
        turns = data.get("turns") or []
        # Replay COMPLETED turns only — a trailing in-flight turn (folded
        # while another frontend is mid-turn) is rendered live by the
        # session renderer (`_render_session_log`) once it lands. Returning
        # the completed count anchors `_rendered_turn_count` so the renderer's
        # subscribe-tail doesn't double-render what we just replayed.
        completed = [t for t in turns if not t.get("in_progress")]
        for t in completed:
            await self._render_one_saved_turn(transcript, t)
        self._maybe_scroll_end(transcript)
        return len(completed)

    async def _render_saved_tool_block(
        self,
        transcript: VerticalScroll,
        blk: dict,
    ) -> None:
        """Render one tool block from a turn's ``render_segments`` — the
        same widget shapes the live path uses (ExitPlanMode / OutputSection
        / Update·Edit cards, else 🔧 call + collapsible result), with the
        collapse state matched via the shared predicate. (turn-replication)
        """
        # Notice blocks (runtime warnings + compaction) — the reader emits
        # these in event order so a replayed/rendered turn mounts the same
        # RuntimeWarningNotice / CompactionNotice widgets the live driver
        # showed, instead of inlining the warning prose or dropping the
        # compaction marker. (turn-replication: notice-widget parity)
        if blk.get("kind") == "notice":
            sub = blk.get("subtype")
            if sub == "runtime_warning":
                parsed = _parse_runtime_warning_chunk(str(blk.get("text") or ""))
                if parsed is not None:
                    _glyph, _ev, _fields = parsed
                    await transcript.mount(RuntimeWarningNotice(_glyph, _ev, _fields))
            elif sub == "compaction":
                payload = blk.get("payload") if isinstance(blk.get("payload"), dict) else {}
                await transcript.mount(CompactionNotice(str(payload.get("reason") or "compacted"), payload))
                # Match the live `/compact` path (app.py ~10529): tag the
                # already-rendered ToolResultMessage widgets whose output
                # was trimmed on the wire with the "🗜️ upstream-truncated"
                # badge, so the renderer's tool-result titles match the
                # driver's. ``dropped`` is NOT applied here — its only live
                # mount is the dead `auto_continue` branch, so marking it
                # would be a replay-only (inverse) divergence.
                # (turn-replication: compaction tool-badge parity)
                _trunc = set(payload.get("truncated_tool_call_ids") or [])
                if _trunc:
                    for _w in transcript.query(ToolResultMessage):
                        if _w.call_id in _trunc:
                            _w.mark_compacted("truncated")
            return
        name = str(blk.get("name") or "?")
        args = blk.get("arguments") if isinstance(blk.get("arguments"), dict) else {}
        cid = str(blk.get("call_id") or "")
        _ok = blk.get("ok")
        ok = True if _ok is None else bool(_ok)
        content = blk.get("content") or ""

        async def _mount_result() -> None:
            await transcript.mount(
                ToolResultMessage(
                    cid,
                    ok,
                    content,
                    outcome=blk.get("outcome"),
                    initial_collapsed=_default_tool_result_collapsed(
                        content,
                        was_streaming=bool(blk.get("was_streaming")),
                    ),
                )
            )

        # Special tool cards. The live dispatch renders each as a dedicated
        # widget and SUPPRESSES the tool result on success, but falls
        # through to a ToolResultMessage on FAILURE so the error body is
        # visible (ExitPlanMode always suppresses). Replay must do the same
        # or a failed Update/OutputSection/UserQuestion would show its card
        # on both frontends but the error result on the driver only.
        # (turn-replication: special-tool result parity)
        if name == "ExitPlanMode":
            await transcript.mount(PlanMessage(str(args.get("plan") or "")))
            return  # plan result is always suppressed, success or fail
        if name == "OutputSection":
            await transcript.mount(
                OutputSectionMessage(
                    str(args.get("section_name") or ""),
                    str(args.get("content") or ""),
                )
            )
            if not ok:
                await _mount_result()
            return
        if name == "UserQuestion":
            await transcript.mount(UserQuestionMessage(args))
            if not ok:
                await _mount_result()
            return
        if name in ("Update", "Edit", "edit"):
            _card = UpdateMessage(
                str(args.get("file_path") or ""),
                str(args.get("old_string") or ""),
                str(args.get("new_string") or ""),
                replace_all=bool(args.get("replace_all", False)),
                call_id=cid,
            )
            await transcript.mount(_card)
            _card.mark_result(ok)
            if not ok:
                await _mount_result()
            return
        if self._is_harness_task_tool(name):
            # Harness Task* family — live renders ONE collapsible
            # HarnessTaskListMessage per task-group (updated in place) and
            # suppresses the generic call/result widgets. Replay rebuilds
            # the same card + footer from the persisted call+result.
            await self._replay_harness_task_block(transcript, name, args, content)
            return
        await transcript.mount(ToolCallMessage(name, args))
        if _is_todo_tool_name(name) and args:
            self._handle_todo_tool_call(args)
        await _mount_result()

    async def _replay_harness_task_block(
        self,
        transcript: VerticalScroll,
        name: str,
        args: dict,
        result_text: str,
    ) -> None:
        """Rebuild the harness-task footer + ONE inline
        :class:`HarnessTaskListMessage` per task-group from a replayed
        Task* call, matching the live dispatch (which mounts one card per
        group, updates it in place, and suppresses the generic call/result
        widgets). The first ``TaskCreate`` of a replayed turn opens a fresh
        group (clears the footer + starts a new card), mirroring the live
        ``_new_task_group_pending`` one-shot. (turn-replication: Task* parity)
        """
        _args = args if isinstance(args, dict) else {}
        if name == "TaskCreate" and not self._replay_task_group_open:
            self._replay_task_group_open = True
            try:
                self._clear_harness_tasks()
            except Exception:  # noqa: BLE001
                pass
            self._replay_harness_card = None
        try:
            if name == "TaskCreate":
                self._apply_task_create_from_result(_args, result_text or "")
            elif name == "TaskUpdate":
                self._apply_task_update_args(_args)
        except Exception:  # noqa: BLE001
            pass
        if self._replay_harness_card is None:
            self._replay_harness_card = HarnessTaskListMessage()
            try:
                await transcript.mount(self._replay_harness_card)
            except Exception:  # noqa: BLE001
                self._replay_harness_card = None
        if self._replay_harness_card is not None:
            rows = [
                {
                    "id": _tid,
                    "subject": (self._harness_tasks.get(_tid) or {}).get("subject", ""),
                    "status": (self._harness_tasks.get(_tid) or {}).get("status", "pending"),
                }
                for _tid in self._harness_tasks_order
            ]
            try:
                self._replay_harness_card.apply_tasklist_payload(rows)
            except Exception:  # noqa: BLE001
                pass

    async def _render_one_saved_turn(
        self,
        transcript: VerticalScroll,
        t: dict,
    ) -> None:
        """Render a single saved turn (user prompt + tool rounds +
        assistant content + thinking + abort/error markers) into the
        transcript. Used by `_replay_session_into_transcript` for cold
        history (a `/sessions` switch). Live cross-frontend turns now
        stream in via `_render_session_log` → `_render_event` instead.
        """
        # Per-turn harness Task* state: the first TaskCreate of THIS turn
        # opens a fresh task-group (clears the footer + starts a new inline
        # card), mirroring the live ``_new_task_group_pending`` one-shot
        # that fires per top-level user-prompt turn. (turn-replication)
        self._replay_harness_card = None
        self._replay_task_group_open = False
        user_text = user_prompt_text(t.get("user"), t.get("attachment_names") or [])
        metrics = t.get("metrics") or {}
        # Map a saved turn's ``metrics.source`` to the right replay
        # widget. The bridge writes ``source = <tool name>``
        # (``delegate_claude`` / ``delegate_codex`` / ``delegate_antigravity``);
        # legacy sessions persisted ``"claude"`` / ``"codex"`` directly
        # — accept both shapes so /history replays from old runs still
        # render with the delegation badge.
        delegate_source: str | None = None
        if isinstance(metrics, dict):
            raw = str(metrics.get("source") or "")
            if raw in ("claude", "delegate_claude"):
                delegate_source = "claude"
            elif raw in ("codex", "delegate_codex"):
                delegate_source = "codex"
            elif raw in ("antigravity", "delegate_antigravity"):
                delegate_source = "antigravity"
        from_delegate = delegate_source is not None
        # Back-compat alias for the few places below that still talk
        # in the original "claude vs not" terms.

        if user_text:
            # Humanise the wire sentinel. Bridge persists
            # ``__SHAREDAGENTS_DELEGATE__:<tool>:<json>`` as the saved
            # ``user`` field for direct-dispatch turns; the raw form
            # must never reach the rendered transcript on
            # ``/sessions`` switch or cross-frontend poll (D1).
            #
            # MISSION.md D1 explicitly wants the operator's actual
            # slash form preserved (``/claude <prompt>``,
            # ``/codex <prompt>``, ``/<sibling> <task>``) so the
            # transcript is greppable + replayable. Earlier revisions
            # also stripped the slash prefix here for delegate turns,
            # which made replay show only the bare prompt and
            # silently lost which delegate produced the answer —
            # keep the full humanised slash text exactly as
            # ``humanise_delegate_marker`` returned it.
            shown = humanise_delegate_marker(user_text)
            await transcript.mount(UserMessage(f"❯ {shown}"))

        # Preferred path: per-generation segments reconstructed from event
        # ORDER reproduce the live driver's interleaving EXACTLY — assistant
        # generation (thinking + content) → its tool call/result → next
        # generation — with per-round thinking duration + char count and the
        # same tool-result collapse state. Used for local-LLM turns; delegate
        # turns (claude/codex) keep the ClaudeMessage path below. Falls back
        # to the legacy flat render for old sessions with no segments.
        segments = t.get("render_segments") or []
        if segments and not from_delegate:
            last_assistant: AssistantMessage | None = None
            _n_seg = len(segments)
            # Deferred per-round metrics line awaiting its round's tool
            # widgets. A TOOL round (an assistant generation followed by a
            # tool call before the next generation) holds its line here and
            # drops it as a standalone ``RoundMetricsMessage`` AFTER the
            # tools — so the output-token stats close the round instead of
            # sitting ABOVE its tool calls (the ordering bug this fixes),
            # mirroring the live driver. It is flushed before the next
            # assistant block and at loop end. The turn's terminal text round
            # (no trailing tool) keeps its line INLINE and also carries the
            # cumulative turn metrics (stamped after the loop).
            # (turn-replication: tool-call ↔ output-token order)
            _pending_round_metrics: dict | None = None

            async def _flush_round_metrics() -> None:
                nonlocal _pending_round_metrics
                # Skip a metrics dict that renders to nothing so we never
                # mount an empty standalone row (the old inline path guarded
                # the same way with ``if parts:``).
                if _pending_round_metrics is not None and _build_metrics_markup(_pending_round_metrics):
                    await transcript.mount(RoundMetricsMessage(_pending_round_metrics))
                _pending_round_metrics = None

            for _idx, blk in enumerate(segments):
                kind = blk.get("kind")
                if kind == "assistant":
                    # New generation ⇒ the previous round's tools are behind
                    # us; drop its deferred line first.
                    await _flush_round_metrics()
                    a = AssistantMessage(from_replay=True)
                    await transcript.mount(a)
                    th = blk.get("thinking") or ""
                    if th:
                        await a.append_thinking_replay(
                            th,
                            duration_ms=blk.get("thinking_ms"),
                            reasoning_chars=blk.get("reasoning_chars"),
                        )
                    ct = blk.get("content") or ""
                    if ct:
                        a.append_content(ct)
                    a.finalize()
                    # TOOL round iff a tool block appears before the next
                    # assistant block. The look-ahead SKIPS ``notice`` blocks
                    # (runtime warnings / compaction) so a warning mounted
                    # between the prose and the tool doesn't fool the test
                    # into treating a real tool round as terminal — and so a
                    # plain no-tool turn that happens to be followed by a
                    # notice does NOT get a spurious standalone line.
                    _is_tool_round = False
                    for _j in range(_idx + 1, _n_seg):
                        _k = segments[_j].get("kind")
                        if _k == "assistant":
                            break
                        if _k == "tool":
                            _is_tool_round = True
                            break
                    _seg_m = blk.get("metrics")
                    if _seg_m and (ct or th) and _is_tool_round:
                        _pending_round_metrics = _seg_m
                    # else: terminal/no-tool round → leave inline; the
                    # cumulative line is stamped on it after the loop.
                    last_assistant = a
                elif kind == "user":
                    shown = user_prompt_text(blk.get("text"), [f["name"] for f in blk.get("attached_files") or []])
                    if shown:
                        await transcript.mount(UserMessage(f"❯ {shown}"))
                else:
                    # Tool or notice block — render in place. (Tool-only
                    # rounds emit no assistant block, so their tools sit
                    # adjacent to the previous round's; the previous visible
                    # round's line therefore lands after them at the next
                    # flush — still BELOW the tools, never above.)
                    await self._render_saved_tool_block(transcript, blk)
            # A turn that ends on a tool round still needs its trailing line
            # dropped after the final tool block.
            await _flush_round_metrics()
            # Cumulative turn metrics land on the final bubble, same as the
            # live end-of-turn path.
            saved_metrics = t.get("metrics") or {}
            if last_assistant is not None and saved_metrics:
                last_assistant.set_metrics(saved_metrics)
            if t.get("aborted"):
                from rich.markup import escape as _esc

                await transcript.mount(
                    Static(
                        f"[dim]{_esc('_[turn aborted]_')}[/dim]",
                        markup=True,
                        classes="claude-msg",
                    )
                )
            elif t.get("error"):
                from rich.markup import escape as _esc

                await transcript.mount(
                    Static(
                        f"[red]{_esc('_[error: ' + str(t['error']) + ']_')}[/red]",
                        markup=True,
                        classes="claude-msg",
                    )
                )
            return

        # Two on-disk shapes exist for tool calls on a saved turn:
        #
        # * Local-LLM engine: nested ``tool_rounds[].calls[]`` with
        #   per-call ``decision`` / ``result`` fields (one round per
        #   model<->tool exchange).
        # * CLI engines (claude_cli / codex_cli / antigravity_cli): flat
        #   ``tool_calls[]`` and ``tool_results[]`` lists paired by
        #   ``call_id`` — the CLI runs tools internally and sharedagents
        #   sees them as a single round.
        #
        # The legacy renderer only handled ``tool_rounds``, so CLI-engine
        # sessions on replay (``/sessions`` switch, cross-frontend poll)
        # showed the inline 🔧/↳ markers the runner glued into
        # ``content`` instead of expandable widgets. Render whichever
        # shape is present.
        tool_rounds_data = t.get("tool_rounds") or []
        flat_calls = t.get("tool_calls") or []
        flat_results = t.get("tool_results") or []
        # D2 — for delegate turns the live ``_run_*`` path suppresses
        # the OUTER ``delegate_*`` tool widget (see ``_run_claude``
        # outer_call_id continue at app.py:10519-10523). The replay
        # path must match: without suppression, the ``ClaudeMessage``
        # body mounts AFTER a "🔧 delegate claude" + "▼ ✅ (empty)"
        # card pair, putting the redundant header card *between* the
        # user line and the agent's response. Filter the outer
        # delegate calls out of the replay loop here so the header
        # ordering matches live.
        _OUTER_DELEGATE_NAMES = {
            "delegate_claude",
            "delegate_codex",
            "delegate_antigravity",
            "delegate_agent",
        }

        def _is_outer_delegate_replay(name: str) -> bool:
            return bool(from_delegate and name in _OUTER_DELEGATE_NAMES)

        had_widgets = False
        if tool_rounds_data:
            had_widgets = True
            for r in tool_rounds_data:
                for c in r.get("calls") or []:
                    name = c.get("name") or "?"
                    args = c.get("arguments") or {}
                    if _is_outer_delegate_replay(str(name)):
                        # Outer delegate call is implicit on a
                        # delegate-source turn — body renders inside
                        # ClaudeMessage below. Skip the redundant
                        # 🔧/✅ widgets so the operator sees:
                        # ``❯ user → ClaudeMessage`` with no empty
                        # header card between them.
                        continue
                    # ExitPlanMode / OutputSection replay as their
                    # dedicated cards — same shape as the live path —
                    # so a resumed session shows the body inline
                    # instead of the collapsed peek.
                    if name == "ExitPlanMode":
                        plan_text = ""
                        if isinstance(args, dict):
                            plan_text = str(args.get("plan") or "")
                        await transcript.mount(PlanMessage(plan_text))
                        continue
                    if name == "OutputSection":
                        section_name = ""
                        section_content = ""
                        if isinstance(args, dict):
                            section_name = str(args.get("section_name") or "")
                            section_content = str(args.get("content") or "")
                        await transcript.mount(OutputSectionMessage(section_name, section_content))
                        continue
                    if name in ("Update", "Edit", "edit"):
                        # Replay a file edit as its "Update" diff card,
                        # matching the live path; the saved decision
                        # tells us whether it applied.
                        _a = args if isinstance(args, dict) else {}
                        _card = UpdateMessage(
                            str(_a.get("file_path") or ""),
                            str(_a.get("old_string") or ""),
                            str(_a.get("new_string") or ""),
                            replace_all=bool(_a.get("replace_all", False)),
                            call_id=str(c.get("id") or ""),
                        )
                        await transcript.mount(_card)
                        _card.mark_result((c.get("decision") or "allow") == "allow")
                        continue
                    await transcript.mount(ToolCallMessage(name, args))
                    # Replay TodoWrite calls into the plan-checklist so a
                    # resumed session shows the accumulated plan in the
                    # footer (matches what was visible when the session
                    # was last live).
                    if _is_todo_tool_name(name) and isinstance(args, dict):
                        self._handle_todo_tool_call(args)
                    cid = c.get("id") or ""
                    decision = c.get("decision") or "allow"
                    result = c.get("result") or ""
                    ok = decision == "allow"
                    await transcript.mount(ToolResultMessage(cid, ok, result))
        elif flat_calls:
            had_widgets = True
            # Index results by call_id so we can pair them in the order
            # the calls fired. CLI engines emit results in call order in
            # practice, but be defensive — a missing result still mounts
            # an empty ToolResultMessage so the operator sees the call.
            results_by_cid: dict[str, dict] = {}
            for r in flat_results:
                if not isinstance(r, dict):
                    continue
                cid = str(r.get("call_id") or "")
                if cid:
                    results_by_cid[cid] = r
            for c in flat_calls:
                if not isinstance(c, dict):
                    continue
                name = str(c.get("name") or "?")
                args = c.get("arguments") or {}
                if _is_outer_delegate_replay(name):
                    # Same suppression as the tool_rounds branch — see
                    # the comment above the rounds loop. Without this,
                    # CLI-engine saved sessions on replay (``/sessions``
                    # switch, cross-frontend poll) render the empty
                    # ``🔧 delegate_claude`` + ``▼ ✅ (empty)`` pair
                    # before the ``ClaudeMessage`` body.
                    continue
                # ExitPlanMode / OutputSection replay as their
                # dedicated cards so a resumed CLI-engine session
                # shows the body, matching the live path.
                if name == "ExitPlanMode":
                    plan_text = ""
                    if isinstance(args, dict):
                        plan_text = str(args.get("plan") or "")
                    await transcript.mount(PlanMessage(plan_text))
                    continue
                if name == "OutputSection":
                    section_name = ""
                    section_content = ""
                    if isinstance(args, dict):
                        section_name = str(args.get("section_name") or "")
                        section_content = str(args.get("content") or "")
                    await transcript.mount(OutputSectionMessage(section_name, section_content))
                    continue
                if name in ("Update", "Edit", "edit"):
                    # Replay a file edit as its "Update" diff card,
                    # matching the live path.
                    _a = args if isinstance(args, dict) else {}
                    _cid = str(c.get("call_id") or "")
                    _r = results_by_cid.get(_cid) or {}
                    _card = UpdateMessage(
                        str(_a.get("file_path") or ""),
                        str(_a.get("old_string") or ""),
                        str(_a.get("new_string") or ""),
                        replace_all=bool(_a.get("replace_all", False)),
                        call_id=_cid,
                    )
                    await transcript.mount(_card)
                    _card.mark_result(bool(_r.get("ok", True)))
                    continue
                await transcript.mount(ToolCallMessage(name, args))
                if _is_todo_tool_name(name) and isinstance(args, dict):
                    self._handle_todo_tool_call(args)
                cid = str(c.get("call_id") or "")
                r = results_by_cid.get(cid) or {}
                ok = bool(r.get("ok", True))
                result_text = str(r.get("content") or "")
                await transcript.mount(ToolResultMessage(cid, ok, result_text, outcome=r.get("outcome")))

        content = t.get("content") or ""
        thinking = t.get("thinking") or ""
        # When we mounted widgets from the flat shape, drop the inline
        # 🔧/↳ markers from the prose body so the operator doesn't see
        # both. Safe no-op if the content carries no markers.
        if had_widgets and flat_calls and content:
            content = _strip_inline_tool_markers(content)
        if from_delegate:
            widget_cls = {
                "claude": ClaudeMessage,
                "codex": CodexMessage,
                "antigravity": AntigravityMessage,
            }[delegate_source]
            delegate_msg = widget_cls(from_replay=True)
            await transcript.mount(delegate_msg)
            if content:
                delegate_msg.append_content(content)
            saved_metrics = t.get("metrics") or {}
            if saved_metrics:
                delegate_msg.set_metrics(saved_metrics)
            delegate_msg.finalize()
        else:
            # `from_replay=True` skips the live prefill counter — these
            # are already-completed turns being re-rendered from the
            # session JSON, so a "▶ prefilling…" tick would be both
            # wrong (no prefill is happening) and visually noisy.
            a = AssistantMessage(from_replay=True)
            await transcript.mount(a)
            if thinking:
                await a.append_thinking(thinking)
            if content:
                a.append_content(content)
            saved_metrics = t.get("metrics") or {}
            if saved_metrics:
                a.set_metrics(saved_metrics)
            a.finalize()

        if t.get("aborted"):
            from rich.markup import escape as _esc

            await transcript.mount(
                Static(
                    f"[dim]{_esc('_[turn aborted]_')}[/dim]",
                    markup=True,
                    classes="claude-msg",
                )
            )
        elif t.get("error"):
            from rich.markup import escape as _esc

            await transcript.mount(
                Static(
                    f"[red]{_esc('_[error: ' + str(t['error']) + ']_')}[/red]",
                    markup=True,
                    classes="claude-msg",
                )
            )

    def _sync_footer_from_turn_metrics(self, m: dict) -> None:
        """Mirror a turn's token accounting into the footer ``ctx`` so a
        passive (non-driving) frontend shows the same ``ctx used/window``
        as the frontend that drove the turn. ``m`` is either a turn-level
        ``metrics`` dict or a single ``round_metrics`` entry — both carry
        ``prompt_tokens`` / ``context_window``.
        """
        if not isinstance(m, dict):
            return
        changed = False
        cw = m.get("context_window")
        if isinstance(cw, int) and cw > 0 and cw != self._last_context_window:
            self._last_context_window = cw
            changed = True
        pt = m.get("prompt_tokens")
        if isinstance(pt, int) and pt > 0 and pt != self._last_prompt_tokens:
            self._last_prompt_tokens = pt
            changed = True
        if changed:
            try:
                self._refresh_status_bar()
            except Exception:  # noqa: BLE001
                pass

    async def _cmd_sessions(self, arg: str = "") -> None:
        """`/sessions` — list saved sessions (no arg) or switch to one
        (`/sessions <index|hash-prefix>`).

        - No arg: render the most-recent-first list inline so the user
          can see indices, short hashes, titles, turn counts, last
          activity timestamps, and the workspace cwd each session was
          opened from.
        - With arg: resolve to a session and switch the TUI's active
          `_sid` to it. The transcript is cleared (we don't replay past
          turns yet — the model still has them in its session record;
          they'll re-enter `_build_messages` on the next turn). Active
          turn is aborted before the switch lands.
        """
        if self._client is None:
            self._set_status("agent not connected")
            return
        transcript = self.query_one("#transcript", VerticalScroll)
        if not arg:
            # Echo first so a slow list_sessions RPC cannot look like a
            # swallowed command (Web UI always echoes the slash).
            await transcript.mount(UserMessage("❯ /sessions"))
        try:
            sessions = await self._client.list_sessions()
        except Exception as e:  # noqa: BLE001
            self._set_status(f"/sessions list failed: {e}")
            return

        if not arg:
            # Echo already mounted. The modal lives on a separate
            # ModalScreen; this UserMessage is the durable evidence the
            # operator sees after dismiss without a pick.
            if not sessions:
                await transcript.mount(
                    Static(
                        "no sessions yet — start typing and the agent will create one.",
                        classes="claude-msg",
                    )
                )
                return
            # Inline the count + active id so the operator (and
            # tmux-capture-based tests) see proof that the list
            # was fetched even when the modal closes via cancel/Esc.
            await transcript.mount(
                Static(
                    f"{len(sessions)} session(s) — current [{(self._sid or '')[:12]}]. Pick one in the modal.",
                    classes="claude-msg",
                )
            )
            # Push the modal picker. `push_screen_wait` blocks this
            # worker until the user selects (returns hash) or cancels
            # (returns None / empty). The modal owns ↑/↓ navigation and
            # Enter/Esc bindings — same UX shape as `PermissionModal`.
            picked = await self.push_screen_wait(
                SessionPickerModal(sessions, current_sid=self._sid or ""),
            )
            if not picked:
                return  # cancelled
            target = next(
                (s for s in sessions if s.get("hash") == picked),
                None,
            )
            if target is None:
                self._set_status(f"/sessions: picked hash {picked[:12]} not in list")
                return
        else:
            # Arg-path: numeric index or hash-prefix. Hash-prefix
            # match wins when numeric parse fails or yields out-of-
            # range. Modal-path skips this and resolves directly above.
            target: dict | None = None
            if arg.isdigit():
                idx = int(arg)
                if 1 <= idx <= len(sessions):
                    target = sessions[idx - 1]
            if target is None:
                matches = [s for s in sessions if (s.get("hash") or "").startswith(arg)]
                if len(matches) == 1:
                    target = matches[0]
                elif len(matches) > 1:
                    self._set_status(
                        f"/sessions: ambiguous prefix {arg!r}, {len(matches)} matches — use a longer prefix"
                    )
                    return
            if target is None:
                self._set_status(f"/sessions: no match for {arg!r}")
                return

        new_sid = target.get("hash") or ""
        await self._switch_to_session_hash(
            new_sid,
            title=(target.get("title") or "").strip(),
        )

    async def _switch_to_session_hash(
        self,
        new_sid: str,
        title: str = "",
    ) -> bool:
        """Switch the TUI's active session to ``new_sid`` and replay it.

        Shared by ``/sessions`` (which resolves an index / hash-prefix to
        a hash first) and the agent monitor (``/monitor`` → Enter on any
        session row). Because the session store is unified across agents
        (``layout.sessions_root``), ``get_session`` hydrates ANY hash — so
        the monitor can open a session a *different* agent created, the
        operator scrolls its replayed history here, and the next turn
        continues it on whatever agent the TUI is bound to (the monitor's
        open path switches agent first when they differ). Returns True
        when the switch landed, False on a no-op / failure.
        """
        if not new_sid:
            self._set_status("session switch: empty hash")
            return False
        if new_sid == (self._sid or ""):
            self._set_status(f"already on session [{new_sid[:12]}]")
            return False
        transcript = self.query_one("#transcript", VerticalScroll)

        # Abort any in-flight turn before swapping the active session.
        if self._current_turn_id is not None:
            await self.action_abort(via="session_switch")
            for _ in range(50):
                if self._current_turn_id is None:
                    break
                await asyncio.sleep(0.02)

        # Wipe the current transcript before replaying — we don't want
        # the OLD session's widgets sitting above the NEW session's
        # replayed history.
        await transcript.remove_children()

        self._sid = new_sid
        # Attaching to a session whose previous process died mid-subagent
        # (reattach from another frontend, gateway restart) leaves orphan
        # sidecars at "running" with no recovery path — sweep them now.
        self._reconcile_orphaned_subagents("aborted")
        # The /claude bridge is per-sharedagents-session — switching means a
        # different `--resume` thread (no prior id).
        self._claude_session_id = None
        # Reset context tracking so the status bar doesn't show stale
        # token counts from the previous session.
        self._last_prompt_tokens = None
        self._last_context_window = None
        # Reset the persistent plan-checklist + harness-task footers so
        # the switched-TO session doesn't inherit the PREVIOUS session's
        # plan. The replay below re-applies this session's own TodoWrite
        # calls (``_render_one_saved_turn`` → ``_handle_todo_tool_call``),
        # so after the wipe the footer rebuilds to exactly what the
        # driving frontend shows. Without this, a second TUI that
        # ``/sessions``-switched onto a session kept a stale plan widget
        # above the new transcript. (turn-replication: plan-footer parity)
        try:
            self.clear_plan_checklist()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._clear_harness_tasks()
        except Exception:  # noqa: BLE001
            pass
        # Hydrate `self.session` so the title bar and metric helpers
        # don't lag behind the switch, then replay the saved turns
        # into the transcript so the user can scroll back through prior
        # history of the session they just switched to.
        replayed = 0
        try:
            data = await self._client.get_session(new_sid)
            await self._bind_session_view(
                {**data, "hash": data.get("hash") or new_sid},
                replay=True,
            )
            replayed = self._rendered_turn_count
        except Exception as e:  # noqa: BLE001
            self._set_status(f"switched to [{new_sid[:12]}] — replay failed: {e}")
            return False
        title = (title or "(untitled)").strip()
        self._set_status(f"switched to [{new_sid[:12]}] — {title} ({replayed} turn(s) replayed)")
        return True

    def _bounce_session_render(self) -> None:
        """Restart the cross-frontend subscribe-tail on the bound session.

        ``_render_session_log`` only notices a ``_render_generation`` bump
        when it receives a yielded SSE frame. Idle ``hb`` events are
        swallowed inside :meth:`AgentClient.subscribe_events`, so after
        ``/new`` or ``/sessions`` the tail can stay stuck on the previous
        session's stream indefinitely (BENCH-03 live Web→TUI miss). Cancel
        and re-arm the worker so the re-anchor is immediate.

        No-op until ``on_mount`` has armed ``_render_log_task`` (early binds
        during construction must not create a duplicate worker).
        """
        if not getattr(self, "_use_event_stream", False):
            return
        if not hasattr(self, "_render_log_task"):
            return
        # Prefer a cooperative exit (checked on SSE hb / reconnect) before
        # hard-cancelling the Textual worker — closes the httpx stream cleanly.
        wake = getattr(self, "_session_render_cancel", None)
        if wake is not None:
            try:
                wake.set()
            except Exception:  # noqa: BLE001
                pass
        old = self._render_log_task
        if old is not None:
            try:
                old.cancel()
            except Exception:  # noqa: BLE001
                pass
        self._render_log_task = self.run_worker(
            self._render_session_log(),
            exclusive=False,
            name="session-render",
        )

    async def _render_session_log(self) -> None:
        """Cross-frontend renderer: tail the session log and render turns
        driven by OTHER frontends through the same widget dispatch.

        Spec §8 unifies live == replay == observe: every subscriber tails
        one durable log and renders the SAME canonical events. Here the
        the renderer tails ``subscribe_events`` and, per turn, decides ownership:

        * a turn the LOCAL user initiated (``_current_turn_id`` armed, or
          one whose ``turn_idx`` we already rendered) is skipped — the
          initiator's own submit+subscribe loop renders it (no
          double-render, spec §8 "no double-render").
        * a turn driven elsewhere (Telegram / web / another TUI) is fed,
          frame by frame, into ``_stream_via_agent`` with a per-turn event
          source — so it renders with FULL fidelity (tools, reasoning,
          subagents, abort/error), replacing the old folded placeholder.

        Resilient: a transport blip is swallowed and the tail reopens from
        the last byte offset (``subscribe_events`` handles reconnect); the
        worker simply retries on the next session_live wake.
        """
        import asyncio as _asyncio

        while True:
            # Wait for a session to exist before tailing.
            sid = self._sid
            if sid is None:
                await _asyncio.sleep(0.2)
                continue
            # Snapshot the generation at the start of this tail; a
            # ``/sessions`` switch (``_bind_session_view``) bumps the
            # generation, which breaks us out to re-anchor on the new session.
            # The tail opens at ``_rendered_offset`` (anchored to the bound
            # log's end), so history is never restreamed; the offset gate then
            # de-dups any reconnect replay. We re-read ``_rendered_offset``
            # each (re)open so a session switch / local turn that advanced it
            # is honoured.
            gen = self._render_generation
            # Defence: if a prior bug anchored beyond the current session
            # log, remote turns are silently dropped. Drop back to 0 and skip
            # prompts already
            # present in the bound folded turns so missed remote turns
            # catch up without duplicating local history.
            try:
                end = int(self._session_log_end_offset(sid))
            except Exception:  # noqa: BLE001
                end = 0
            poison_catchup = bool(end and self._rendered_offset > end)
            if poison_catchup:
                self._rendered_offset = 0
            known_users: set[str] = set()
            if poison_catchup and self.session is not None:
                for _t in self.session.turns or []:
                    if not isinstance(_t, dict):
                        continue
                    for _k in ("user", "prompt", "content"):
                        _u = str(_t.get(_k) or "").strip()
                        if _u:
                            known_users.add(_u)
                            break
            # Collect an external turn's frames from THIS outer subscribe
            # (never open a nested SSE — that hung the renderer and dropped
            # every subsequent Web/Telegram turn).
            collecting_ti: int | None = None
            frame_q: _asyncio.Queue[dict | None] | None = None
            render_task: _asyncio.Task[int] | None = None
            cancel = _asyncio.Event()
            self._session_render_cancel = cancel
            try:
                async for frame in self._client.subscribe_events(
                    sid,
                    from_offset=self._rendered_offset,
                    cancel_event=cancel,
                ):
                    if cancel.is_set() or self._sid != sid or self._render_generation != gen:
                        break  # session switched — restart the tail
                    # Intercept durable-queue lifecycle events before the
                    # turn-start gate — they don't carry a turn_idx and would
                    # be silently dropped by _unrendered_turn_offset otherwise.
                    _ev_q = frame.get("event") if isinstance(frame, dict) else None
                    if isinstance(_ev_q, dict) and _ev_q.get("type", "").startswith("queue_msg_"):
                        await self._handle_queue_event(_ev_q)
                        continue

                    ev = frame.get("event") if isinstance(frame, dict) else None
                    if collecting_ti is not None and frame_q is not None:
                        _ti = None
                        if isinstance(ev, dict) and ev.get("turn_idx") is not None:
                            try:
                                _ti = int(ev.get("turn_idx"))
                            except (TypeError, ValueError):
                                _ti = None
                        if _ti == collecting_ti:
                            await frame_q.put(frame)
                            if ev.get("type") == "turn_ended":
                                await frame_q.put(None)
                                if render_task is not None:
                                    last = await render_task
                                    self._rendered_offset = max(
                                        self._rendered_offset,
                                        int(last or 0),
                                    )
                                collecting_ti = None
                                frame_q = None
                                render_task = None
                        continue

                    # Byte offset — not turn_idx — is the turn identity: a
                    # rapid burst of queued prompts (Telegram "send 5 essays")
                    # all share one turn_idx but land at distinct offsets.
                    off = self._unrendered_turn_offset(frame, self._rendered_offset)
                    if off is None:
                        continue  # history, a body frame, or already handled
                    if self._current_turn_id is not None or self._local_submit_pending:
                        # We're driving (or just submitted) a local turn; its
                        # initiator loop renders it. Advance past it so we
                        # don't re-render, and disarm the submit guard.
                        self._rendered_offset = off
                        self._local_submit_pending = False
                        continue
                    if not isinstance(ev, dict):
                        continue
                    if poison_catchup:
                        _user = str(ev.get("user") or "").strip()
                        if _user and _user in known_users:
                            self._rendered_offset = off
                            continue
                    ti = int(ev.get("turn_idx") or 0)
                    collecting_ti = ti
                    frame_q = _asyncio.Queue()
                    await frame_q.put(frame)
                    render_task = _asyncio.create_task(
                        self._render_logged_turn(sid, ti, frame_q),
                    )
            except Exception:  # noqa: BLE001
                # Transport / session switch blip — retry from the top.
                await _asyncio.sleep(0.5)
            finally:
                if getattr(self, "_session_render_cancel", None) is cancel:
                    self._session_render_cancel = None
                if frame_q is not None:
                    await frame_q.put(None)
                if render_task is not None:
                    try:
                        await render_task
                    except Exception:  # noqa: BLE001
                        pass

    async def _render_logged_turn(
        self,
        sid: str,
        turn_idx: int,
        frame_q: asyncio.Queue[dict | None],
    ) -> int:
        """Render one externally-driven turn from frames fed by the outer
        subscribe-tail (no nested SSE).

        Mounts the external prompt's ``UserMessage`` first (the renderer
        bypasses ``_send_user_turn``, which normally does this for local
        turns), then drives the dispatch in log-render mode so the footer reads
        "rendering…" and the local turn queue is left untouched.

        Returns the byte offset of the last frame consumed (this turn's
        ``turn_ended``), so the caller can advance the renderer's offset anchor
        past the whole turn.
        """
        transcript = self.query_one("#transcript", VerticalScroll)
        first_frame = await frame_q.get()
        if not isinstance(first_frame, dict):
            return self._rendered_offset
        user_text = ""
        ev0 = first_frame.get("event") or {}
        if isinstance(ev0, dict):
            user_text = user_prompt_text(ev0.get("user"), ev0.get("attachment_names") or [])
        if user_text:
            await transcript.mount(
                UserMessage(f"❯ {humanise_delegate_marker(user_text)}"),
            )
            self._maybe_scroll_end(transcript)

        # Reset the per-turn bubble bookkeeping the dispatch expects.
        self._current_assistant = None
        self._round_pending_assistant = None
        self._pending_tool_calls.clear()
        self._plan_call_ids.clear()
        self._section_call_ids.clear()

        last_off = int(first_frame.get("offset") or 0)

        async def _src() -> AsyncIterator[tuple[str, dict, str | None]]:
            nonlocal last_off
            frame: dict | None = first_frame
            while frame is not None:
                off = frame.get("offset")
                if isinstance(off, int) and off > last_off:
                    last_off = off
                for triple in self._render_event(frame):
                    yield triple
                ev = frame.get("event") if isinstance(frame, dict) else None
                if isinstance(ev, dict) and ev.get("type") == "turn_ended":
                    return
                nxt = await frame_q.get()
                frame = nxt if isinstance(nxt, dict) else None

        # ``_stream_via_agent(from_log=True)`` owns the "rendering…" footer cue
        # and clears it on completion; the ``_rendering_logged_turn`` flag is a
        # belt-and-suspenders marker other status paths can read.
        self._rendering_logged_turn = True
        try:
            await self._stream_via_agent(
                user_text,
                event_source=_src(),
                from_log=True,
            )
        finally:
            self._rendering_logged_turn = False
        return last_off
