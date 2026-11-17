"""Turn rendering + streaming methods for :class:`AcpTelegramBot`.

Extracted from the :class:`AcpTelegramBot` god-class (de-monolith). This is a
mixin: every method still binds ``self`` and dispatches identically through the
concrete class's MRO. Behaviour is unchanged — the method bodies moved verbatim.

The module-level helpers/constants these methods read stay owned by ``bot.py``
and are imported at the BOTTOM (after the class) so the ``bot`` <-> mixin import
cycle is deadlock-free in either import order.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def _session_probe_sid(sid: str, sessions_dir: Path) -> str:
    """Return the session SID used for local health checks."""
    del sessions_dir
    return sid


class _SessionArtifacts:
    """``ctx.artifacts`` for a locally submitted turn: each rendered file goes
    straight to the caller's list as ``(sid, event)``, so it survives the
    supervisor cancelling the dispatch; the caller uploads after its final
    paint."""

    __slots__ = ("_sink", "_sid")

    def __init__(self, sink: list, sid: str) -> None:
        self._sink, self._sid = sink, sid

    def append(self, event: dict) -> None:
        self._sink.append((self._sid, event))


class StreamingMixin:
    @staticmethod
    def _blocks_to_wire(blocks: list) -> list[dict]:
        """Serialise ACP content-block objects → plain dict blocks.

        The submit route (and the web/TUI path) takes JSON content blocks.
        Telegram builds ``acp.schema`` block models (``TextContentBlock`` /
        ``ImageContentBlock``) for voice-transcribed text, captions, and
        image/file attachments — so the SAME blocks ride the submit body that
        the legacy ``session_prompt`` drive sent. Pydantic models are dumped
        ``by_alias`` (so ``mime_type`` → ``mimeType``) with ``None`` fields
        stripped; already-dict blocks pass through unchanged.
        """
        out: list[dict] = []
        for b in blocks:
            if isinstance(b, dict):
                out.append(b)
                continue
            dump = getattr(b, "model_dump", None)
            if dump is not None:
                try:
                    out.append(dump(by_alias=True, exclude_none=True))
                    continue
                except Exception:  # noqa: BLE001
                    pass
            # Last-ditch: a bare text-ish object.
            text = getattr(b, "text", None)
            if text is not None:
                out.append({"type": "text", "text": str(text)})
        return out

    async def _render_event(
        self,
        event: dict,
        ctx: _TelegramRenderCtx,
    ) -> None:
        """Map ONE canonical event onto Telegram's existing renderers.

        This is the heart of the canonical-stream cutover (spec §Project 1):
        every interface renders the SAME canonical event log, natively. Web /
        TUI already do; this is Telegram's ``render(event)``. The mapping is a
        thin adapter onto the render callbacks packaged in ``ctx`` (the same
        ``_push_*`` closures the legacy ACP-iterator drive used), gated by the
        shared ``delivery.should_show`` semantics carried as ``ctx`` flags —
        so the reasoning/tools off·on·stream truth table is byte-identical to
        the bespoke path it replaces.

        Canonical event → render:

        * ``text_delta(delta)``      → ``push_content`` (answer, always shown)
        * ``thinking_delta(delta)``  → ``push_thought`` (gated by reasoning)
        * ``tool_call(call_id,name,arguments)`` → header (gated) / teacher badge
        * ``tool_result_meta(call_id,ok)``      → ``on_tool_status`` tick
        * ``tool_result(call_id,content)``      → ``push_tool_body`` (gated)
        * ``turn_ended(final_content,aborted,error)`` → finalize (sets
          ``ctx.stop_state`` so the per-turn loop ends)

        Events whose ``turn_idx`` belongs to a DIFFERENT turn than this
        dispatch owns are ignored, so a rendered / prior turn replayed on the
        same stream never bleeds into this reply. Structural events
        (round_*, header_update, lifecycle, session_*) are no-ops here — they
        carry no Telegram-visible body.
        """
        etype = event.get("type")
        if not etype:
            return
        # Scope to this dispatch's turn. ``turn_started`` / ``turn_ended``
        # carry ``turn_idx``; deltas/tool events carry it too. When this
        # dispatch hasn't pinned a turn yet (canonical path pins it from the
        # first event), accept the first turn we see.
        ev_turn = event.get("turn_idx")
        if etype == "turn_started":
            if ctx.turn_idx is None and isinstance(ev_turn, int):
                ctx.turn_idx = ev_turn
            # Render the user prompt from the canonical stream when an
            # ``echo_user`` sink is wired (the RENDER path — a turn started on
            # web/TUI/another chat, so its prompt isn't already in THIS chat).
            # The locally-submitted turn leaves ``echo_user`` None: the user's
            # own Telegram message is already visible, so we must not duplicate
            # it. This is what gives Telegram prompt parity with web/TUI.
            if ctx.echo_user is not None:
                user = event.get("user")
                text = (
                    user
                    if isinstance(user, str)
                    else (
                        "".join(b.get("text", "") for b in user if isinstance(b, dict))
                        if isinstance(user, list)
                        else ""
                    )
                )
                if text:
                    await ctx.echo_user(text)
            return
        if ctx.turn_idx is not None and isinstance(ev_turn, int) and ev_turn != ctx.turn_idx:
            return
        if ctx.turn_idx is None and isinstance(ev_turn, int):
            # First body event of the (only) turn we're tracking — pin it.
            ctx.turn_idx = ev_turn

        if etype == "artifact_reference":
            # A file the agent linked. Both render paths
            # send the collected files once the reply's final paint is done.
            ctx.artifacts.append(event)
            return

        # BLOCK-STREAM mode (the live local path): route each canonical event
        # straight to the 💭/🔧/💬 message-sequence renderer. The single-buffer
        # push_* path below is used only by the render-observer (external turns).
        if ctx.blockstream is not None:
            bs = ctx.blockstream
            # #1 (defensive) — a delegated CHILD's forwarded inner event tagged
            # ``subagent`` must not flood the parent reply. The compact
            # "🔎 … working…" delegation line already conveys progress, so drop
            # these unless the operator opted into full fidelity (/tools stream).
            # (On the canonical path these are already filtered upstream in
            # ``delivery.should_show`` at the gateway; this covers any other
            # transport that forwards them live.)
            if _delivery.is_subagent_event(event) and not ctx.tool_show_body:
                return
            if etype == "text_delta":
                _delta = event.get("delta") or event.get("chunk") or ""
                # Runtime-warning TextDeltas (loop/nudge/distill/retry/force-
                # terminate) are internal diagnostics the TUI renders as a
                # RuntimeWarningNotice widget, NOT as content. Mirror that
                # here by stripping the notice paragraph from EVERY text
                # delta — ``strip_runtime_notices`` is an exact no-op when no
                # notice paragraph is present, so this is safe for clean
                # deltas, and it correctly handles COALESCED deltas
                # (``SessionWriter`` merges consecutive content chunks) whose
                # body looks like ``A\n\n↻ runtime: …\n\nB``: those do NOT
                # start with the notice glyph, so the previous ``is_runtime_
                # notice_delta`` guard skipped them and the notice leaked into
                # the answer bubble. Stripping unconditionally preserves the
                # surrounding content (and its blank-line separation) and
                # never pollutes final_content with "↻ runtime: nudge …" prose.
                # (codex pre-deploy review round 2, MEDIUM #1)
                _delta = _delivery.strip_runtime_notices(_delta)
                if not _delta:
                    return
                await bs.text_delta(_delta)
            elif etype == "thinking_delta":
                if _delivery.should_show(event, {"reasoning": ctx.reasoning_level}):
                    await bs.thinking_delta(event.get("delta") or event.get("chunk") or "")
            elif etype == "tool_call":
                targs = event.get("arguments")
                if not targs and event.get("raw_arguments"):
                    targs = event.get("raw_arguments")
                await bs.tool_call(
                    event.get("call_id") or "",
                    event.get("name") or "tool",
                    targs,
                )
            elif etype == "tool_delta":
                tid = event.get("call_id") or ""
                title = bs._titles.get(tid) or "tool"
                gated = (title in _DIRECT_DISPATCH_TITLES or ctx.tool_show_body) and title not in _INTERNAL_TOOL_TITLES
                if gated:
                    await bs.tool_delta(tid, event.get("delta") or event.get("chunk") or "")
            elif etype == "tool_result":
                tid = event.get("call_id") or ""
                title = bs._titles.get(tid) or "tool"
                _ok = bool(event.get("ok", True))
                gated = (_direct_answer_ok(title, _ok) or ctx.tool_show_body) and title not in _INTERNAL_TOOL_TITLES
                await bs.tool_result(
                    tid,
                    (event.get("content") or "") if gated else "",
                    _ok,
                )
            elif etype == "turn_ended":
                await bs.turn_ended(
                    event.get("metrics") or {},
                    final_thinking=event.get("final_thinking") or "",
                    aborted=bool(event.get("aborted")),
                    error=event.get("error"),
                    final_content=str(event.get("final_content") or ""),
                )
                ctx.stop_state["done"] = True
                ctx.stop_state["aborted"] = bool(event.get("aborted"))
                ctx.stop_state["error"] = event.get("error")
                ctx.stop_state["final_content"] = event.get("final_content")
                ctx.stop_state["stop_reason"] = (
                    "cancelled" if event.get("aborted") else "error" if event.get("error") else "end_turn"
                )
                # Propagate the REAL terminal outcome (from the canonical
                # turn_ended event) into the shared per-turn ``_progress`` so
                # the lifecycle-reaction classifier in _stream_turn_for_chat
                # does NOT have to infer it from the assembled-text buffer —
                # which on the block-stream path is always "⚠️ empty reply"
                # (visible_parts is never populated there) and would otherwise
                # mis-classify every successful turn as an error. The branch-
                # level _turn_outcome (supervision CancelledError paths) still
                # takes precedence; this only feeds the normal (non-exception)
                # path. (codex pre-deploy review 2026-07-26, HIGH finding)
                _prog = getattr(bs, "_progress", None)
                if isinstance(_prog, dict):
                    if event.get("aborted"):
                        _prog["terminal_outcome"] = "abort"
                    elif event.get("error"):
                        _prog["terminal_outcome"] = "error"
                    else:
                        _fc_clean = _delivery.strip_runtime_notices(str(event.get("final_content") or "")).strip()
                        _prog["terminal_outcome"] = "success" if _fc_clean else "empty"
            # tool_result_meta / round_* / structural events: no block action.
            return

        if etype == "text_delta":
            piece = event.get("delta") or event.get("chunk") or ""
            if piece:
                # Reasoning→content split fires ONLY when a caller wires
                # ``on_split`` (the legacy single-message split mode). In the
                # TWO-MESSAGE mode the live path uses, ``on_split`` is None and
                # the answer streams contiguously into its OWN 💬 message — so
                # an interleaving model (qwen emits stray thinking tokens mid-
                # answer) can never fragment it (the cropping bug).
                if ctx.had_thinking and ctx.on_split is not None:
                    await ctx.on_split()
                    ctx.had_thinking = False
                ctx.push_content(piece)
                if ctx.on_chunk is not None:
                    await ctx.on_chunk("".join(ctx.visible_parts))
            return

        if etype == "thinking_delta":
            # FIX 2 — gate the LIVE reasoning stream through the SHARED
            # predicate over the ACTUAL canonical event, so Telegram's
            # off·on·stream truth table is byte-identical to web/TUI:
            #   off    → hide (no live deltas, no final)
            #   on     → hide LIVE deltas (the CONSOLIDATED final_thinking
            #            renders from ``turn_ended`` below instead)
            #   stream → show live deltas
            # ``should_show({"type":"thinking_delta"}, prefs)`` returns True
            # ONLY on "stream" (see delivery.should_show §7), which is exactly
            # the live-delta channel. The previous code re-derived visibility
            # from a ``tool_call`` probe over the reasoning level, which made
            # "on" render LIVE deltas — diverging from the shared semantics.
            if not _delivery.should_show(
                event,
                {"reasoning": ctx.reasoning_level},
            ):
                return
            piece = event.get("delta") or event.get("chunk") or ""
            if piece:
                ctx.had_thinking = True
                ctx.push_thought(piece)
                if ctx.on_think_chunk is not None:
                    # TWO-MESSAGE mode: paint the dedicated 💭 message.
                    await ctx.on_think_chunk("".join(ctx.think_parts))
                elif ctx.on_chunk is not None and ctx.thought_rendered():
                    # Single-message fallback (render-observer path).
                    await ctx.on_chunk("".join(ctx.visible_parts))
            return

        if etype == "tool_call":
            title = event.get("name") or "tool"
            tool_id = event.get("call_id") or ""
            targs = _parse_tool_args(event.get("arguments"))
            if tool_id:
                ctx.tool_titles[tool_id] = title
            _apply_tool_watchdog(ctx.progress, title, targs, tool_id)
            if tool_id and tool_id not in ctx.headers_emitted:
                ctx.headers_emitted.add(tool_id)
                if title == "delegate_teacher":
                    ctx.push_teacher_badge(event.get("arguments"))
                elif ctx.tool_show_header:
                    from sharedagents_agent.tools.runners import format_tool_header

                    ctx.push_tool_header(format_tool_header(title, targs))
                if ctx.on_chunk is not None:
                    # Force the prose tail to land NOW — the agent is about
                    # to enter a tool that may block for seconds-to-minutes.
                    await ctx.on_chunk("".join(ctx.visible_parts), force=True)
                # Split the message at the tool-call boundary: the current
                # round's content (reasoning + prose) is finalized as a
                # standalone message and a fresh placeholder opens for the
                # next round. Only the latest placeholder gets edited.
                if ctx.on_split is not None:
                    await ctx.on_split()
            if ctx.on_tool_status is not None:
                await ctx.on_tool_status(title, "in_progress")
            return

        if etype == "tool_delta":
            if ctx.progress is not None:
                ctx.progress["last"] = time.monotonic()
            piece = event.get("delta") or event.get("chunk") or ""
            tool_id = event.get("call_id") or ""
            title = ctx.tool_titles.get(tool_id) or "tool"
            if not piece:
                return
            if title in _INTERNAL_TOOL_TITLES:
                return
            if title not in _DIRECT_DISPATCH_TITLES and not ctx.tool_show_body:
                return
            ctx.streamed_tools.add(tool_id)
            ctx.push_tool_body(piece)
            if ctx.on_chunk is not None:
                await ctx.on_chunk("".join(ctx.visible_parts))
            return

        if etype == "tool_result_meta":
            tool_id = event.get("call_id") or ""
            title = ctx.tool_titles.get(tool_id) or "tool"
            _clear_tool_watchdog(ctx.progress, tool_id)
            # First sighting (tool that skipped tool_call) — plant header.
            if tool_id and tool_id not in ctx.headers_emitted:
                ctx.headers_emitted.add(tool_id)
                if ctx.tool_show_header:
                    ctx.push_tool_header(title)
                if ctx.on_chunk is not None:
                    await ctx.on_chunk("".join(ctx.visible_parts), force=True)
            status_val = "completed" if event.get("ok") else "failed"
            if ctx.on_tool_status is not None:
                await ctx.on_tool_status(title, status_val)
            return

        if etype == "tool_result":
            tool_id = event.get("call_id") or ""
            title = ctx.tool_titles.get(tool_id) or "tool"
            text_piece = event.get("content") or ""
            if not text_piece:
                return
            if title in _INTERNAL_TOOL_TITLES:
                return
            # Gate non-direct tool bodies via should_show over the actual
            # tool_result event (ctx.tool_show_body): off → suppress; on/stream
            # → show — the shared §7 semantics, identical to web/TUI.
            # A SUCCESSFUL direct-dispatch body IS the answer and always lands; a
            # FAILED one (ok=False) carries internal diagnostics, so it falls back
            # to the normal /tools gate instead of leaking to the user.
            if not _direct_answer_ok(title, bool(event.get("ok", True))) and not ctx.tool_show_body:
                return
            ctx.streamed_tools.add(tool_id)
            ctx.push_tool_body(text_piece)
            if ctx.on_chunk is not None:
                await ctx.on_chunk("".join(ctx.visible_parts), force=True)
            return

        if etype == "turn_ended":
            # FIX 2 — "on" reasoning shows the CONSOLIDATED final thinking
            # (not the live deltas, which were gated out above), matching
            # web/TUI which render the reasoning disclosure from
            # ``final_thinking`` at turn end. "off" shows nothing; "stream"
            # already streamed the live deltas, so we don't re-render here.
            if ctx.reasoning_level == "on":
                ft = event.get("final_thinking") or ""
                if ft:
                    if ctx.on_think_chunk is not None:
                        # TWO-MESSAGE mode: the consolidated reasoning is its
                        # OWN 💭 message (above the answer), painted once here.
                        ctx.think_parts[:] = ["💭 ", ft]
                        await ctx.on_think_chunk("".join(ctx.think_parts))
                    else:
                        # Single-message: prepend a 💭 section to the buffer
                        # (layout "on" has always produced 💭 then 💬 answer).
                        prefix = ["💭 ", ft]
                        if ctx.visible_parts:
                            prefix.append("\n\n")
                        ctx.visible_parts[0:0] = prefix
                        if ctx.on_chunk is not None:
                            await ctx.on_chunk(
                                "".join(ctx.visible_parts),
                                force=True,
                            )
            # Reconcile streamed answer with final_content before stats.
            _fc = str(event.get("final_content") or "")
            _metrics = event.get("metrics") or {}
            _line = ""
            if not event.get("aborted") and not event.get("error"):
                _line = _metrics_line(_metrics)
            if _fc and not event.get("aborted") and not event.get("error"):
                reconciled, patched = _delivery.reconcile_visible_with_final(
                    "".join(ctx.visible_parts),
                    _fc,
                )
                if patched:
                    ctx.visible_parts.clear()
                    ctx.visible_parts.append(reconciled)
                    # Defer paint when a stats row follows — one combined
                    # force-edit is enough (avoids a redundant Telegram edit
                    # before the terminal stats paint).
                    if ctx.on_chunk is not None and not _line:
                        await ctx.on_chunk(reconciled, force=True)
            # Cumulative turn stats row — the same dim telemetry line the
            # TUI stamps on the final bubble and the web renders under the
            # turn, built from the SAME shared formatter so the three
            # frontends can never drift. Appended LAST (after any final
            # thinking prepend) so it closes the message, mirroring TUI.
            # Skipped on abort/error (no meaningful totals) and when the
            # metrics dict renders to nothing.
            if not event.get("aborted") and not event.get("error") and _line:
                if ctx.visible_parts:
                    ctx.visible_parts.append("\n\n")
                ctx.visible_parts.append(_line)
                if ctx.on_chunk is not None:
                    await ctx.on_chunk(
                        "".join(ctx.visible_parts),
                        force=True,
                    )
            ctx.stop_state["done"] = True
            ctx.stop_state["aborted"] = bool(event.get("aborted"))
            ctx.stop_state["error"] = event.get("error")
            ctx.stop_state["final_content"] = event.get("final_content")
            if event.get("aborted"):
                ctx.stop_state["stop_reason"] = "cancelled"
            elif event.get("error"):
                ctx.stop_state["stop_reason"] = "error"
            else:
                ctx.stop_state["stop_reason"] = "end_turn"
            return

        if etype == "round_ended":
            # Per-round stats row for a TOOL round (a round that dispatched
            # tool calls, incl. a tool-call-ONLY round) — matches the TUI's
            # standalone ``RoundMetricsMessage`` mounted below each round's
            # tools. The terminal round (finish_reason "stop"/"length") is
            # covered by the cumulative line at ``turn_ended``, so it's
            # skipped here to avoid a duplicate. Same shared formatter.
            #
            # Gated like other per-round DETAIL: shown only when tools OR
            # reasoning is visible. With BOTH off the operator wants only the
            # cumulative turn line, so suppress the per-round line here.
            _detail_on = bool(ctx.tool_show_header or ctx.reason_shown)
            if _detail_on and str(event.get("finish_reason") or "") == "tool_calls":
                _line = _metrics_line(event.get("metrics") or {})
                if _line:
                    if ctx.visible_parts:
                        ctx.visible_parts.append("\n\n")
                    ctx.visible_parts.append(_line)
                    if ctx.on_chunk is not None:
                        await ctx.on_chunk(
                            "".join(ctx.visible_parts),
                            force=True,
                        )
            return

        # queue_msg_injected — a queued follow-up was dequeued and injected
        # as a user turn while the session was between turns. Render it as
        # an inbound user line (same echo_user path as turn_started uses for
        # cross-interface prompt parity). Block-stream path is handled in the
        # blockstream branch above (it falls through here for the single-buffer
        # path only).
        if etype == "queue_msg_injected":
            _inj_text = event.get("text")
            if _inj_text and ctx.echo_user is not None:
                await ctx.echo_user(str(_inj_text))
            return

        # round_started / header_update / lifecycle /
        # session_* → no Telegram-visible body; ignore.
        return

    async def _deliver_artifacts(
        self,
        chat_id: int,
        thread_id: int | None,
        items: list,
    ) -> None:
        """Send a finished turn's recorded files (photos / documents) to the chat.

        ``items`` are ``(sid, artifact_reference event)`` pairs. Call it only
        after the reply's final paint, and never from inside a supervised
        dispatch task. Best-effort: a failure is logged and noticed in the
        chat, never raised into the turn that already rendered.
        """
        base = self._http_gateway_base()
        if not items or self._api is None or not base:
            return
        if not self.tg_cfg.can_send(chat_id):
            return
        headers: dict[str, str] = {"X-Sharedagents-Principal": "local"}
        if self.acp_bearer_token:
            headers["Authorization"] = f"Bearer {self.acp_bearer_token}"
        by_sid: dict[str, list] = {}
        for sid, event in items:
            by_sid.setdefault(sid, []).append(event)
        for sid, events in by_sid.items():
            try:
                await deliver_artifacts(
                    self._api,
                    base=base,
                    headers=headers,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    sid=sid,
                    artifacts=events,
                    log=self._log,
                )
            except Exception:  # noqa: BLE001
                self._log.exception(
                    "artifact delivery failed chat=%s sid=%s",
                    chat_id,
                    sid,
                )

    # ---- live cross-interface renderer (FIX 1, spec §8) ------------------

    def _build_render_ctx(
        self,
        *,
        reasoning_level: str,
        tool_output_level: str,
        visible_parts: list[str],
        on_chunk: Any | None = None,
        on_tool_status: Any | None = None,
        echo_user: Any | None = None,
    ) -> _TelegramRenderCtx:
        """Build a :class:`_TelegramRenderCtx` over a fresh buffer.

        Packages the SAME render closures + shared visibility gates the
        local ``_dispatch_blocks`` path uses (so a log-rendered turn renders
        byte-identically to a locally-driven one), wired to the supplied
        async sinks. ``echo_user`` is set on the render path so an
        externally-started turn's prompt renders as an inbound bubble; the
        local path leaves it ``None`` (the user's own message is already in
        the chat). Visibility is gated through ``delivery.should_show`` over
        the SHARED predicate — the FIX-2 semantics — so off·on·stream is
        identical to web/TUI.
        """
        state = {"last_kind": None, "content_seen": False}
        pending_thoughts: list[str] = []
        # Reasoning shown AT ALL (not "off"); the live-vs-final distinction is
        # enforced in ``_render_event`` via should_show over the real event.
        reason_shown = reasoning_level != "off"
        tool_show_header = _delivery.should_show(
            {"type": "tool_call"},
            {"tools": tool_output_level},
        )
        # The tool RESULT BODY is gated by should_show over the ACTUAL
        # ``tool_result`` event — shown ONLY on "stream" (§7 / see
        # delivery.should_show), hidden on "on". The "on" mode only shows
        # the ``tool_call`` header (🔧 name {args}).
        tool_show_body = _delivery.should_show(
            {"type": "tool_result"},
            {"tools": tool_output_level},
        )

        def _flush_pending_thoughts() -> None:
            if not pending_thoughts:
                return
            if visible_parts:
                visible_parts.append("\n\n")
            visible_parts.append("💭 ")
            visible_parts.append("".join(pending_thoughts))
            pending_thoughts.clear()
            state["last_kind"] = "thinking"

        def push_thought(text: str) -> None:
            if not text or not reason_shown:
                return
            if reasoning_level == "on" and not state["content_seen"]:
                pending_thoughts.append(text)
                return
            if state["last_kind"] != "thinking":
                if visible_parts:
                    visible_parts.append("\n\n")
                visible_parts.append("💭 ")
                state["last_kind"] = "thinking"
            visible_parts.append(text)

        def push_content(text: str) -> None:
            if not text:
                return
            if pending_thoughts:
                _flush_pending_thoughts()
            if state["last_kind"] == "thinking":
                visible_parts.append("\n\n💬 ")
                state["last_kind"] = "content"
            elif state["last_kind"] in (None, "tool", "tool_body"):
                if visible_parts:
                    visible_parts.append("\n\n")
                state["last_kind"] = "content"
            state["content_seen"] = True
            visible_parts.append(text)

        _DELEGATE_BADGES = {
            "delegate_claude": "↳ via claude",
            "delegate_codex": "↳ via codex",
            "delegate_antigravity": "↳ via antigravity",
            "AgentSync": "↳ via agent",
            "AgentAsync": "↳ via agent",
        }

        def push_tool_header(title: str) -> None:
            badge = _DELEGATE_BADGES.get(title)
            if badge is not None:
                if visible_parts:
                    visible_parts.append("\n\n")
                visible_parts.append(f"_{badge}_")
                state["last_kind"] = "tool"
                return
            if not tool_show_header:
                return
            if visible_parts:
                visible_parts.append("\n\n")
            visible_parts.append(f"🔧 {title}…")
            state["last_kind"] = "tool"

        def push_teacher_badge(raw_input: Any) -> None:
            ri = raw_input if isinstance(raw_input, dict) else {}
            model = ri.get("model") or "teacher"
            trig = ri.get("trigger") or ""
            trig_label = {
                "pre_idle": "review before idle",
                "reasoning_loop": "reasoning loop",
                "premature_end": "premature end",
                "error-terminal": "error terminal",
                "stop-cap": "continuation cap",
            }.get(trig, trig)
            if visible_parts:
                visible_parts.append("\n\n")
            suffix = f": {trig_label}" if trig_label else ""
            visible_parts.append(f"_🧑‍🏫 teacher ({model}){suffix}_")
            state["last_kind"] = "tool"

        def push_tool_body(text: str) -> None:
            if not text:
                return
            if state["last_kind"] == "tool":
                visible_parts.append("\n")
                state["last_kind"] = "tool_body"
            visible_parts.append(text)

        return _TelegramRenderCtx(
            push_thought=push_thought,
            push_content=push_content,
            push_tool_header=push_tool_header,
            push_tool_body=push_tool_body,
            push_teacher_badge=push_teacher_badge,
            reason_shown=reason_shown,
            reasoning_level=reasoning_level,
            thought_rendered=lambda: not (reasoning_level == "on" and not state["content_seen"]),
            tool_show_header=tool_show_header,
            tool_show_body=tool_show_body,
            visible_parts=visible_parts,
            on_chunk=on_chunk,
            on_tool_status=on_tool_status,
            echo_user=echo_user,
        )

    def _ensure_render(
        self,
        chat_id: int,
        thread_id: int | None,
        sid: str,
        start_offset: int,
    ) -> None:
        """Idempotently start the live render observer for ``(chat, sid)``.

        One background task per (chat, thread, session): a chat bound to a
        session tails that session's canonical stream and renders turns
        started ELSEWHERE (web / TUI / another chat) into this chat. Cheap +
        idempotent: a second call for the same (chat, sid) is a no-op while
        the task is alive. A ``/new`` (new sid) gets a fresh observer; the old
        one is cancelled by ``_stop_render_key`` on the stale key.
        """
        if not self._http_gateway_base():
            return  # no control endpoint (ws/stdio fallback) — no renderer
        ck = TelegramConfig.chat_key(chat_id, thread_id)
        key = f"{ck}::{sid}"
        # Anchor the live offset the first time we bind this (chat, sid).
        self._rendered_offset.setdefault(ck, max(0, start_offset))
        existing = self._render_log_tasks.get(key)
        if existing is not None and not existing.done():
            return
        # A new sid for this chat → cancel any observer on the old sid.
        for k in list(self._render_log_tasks):
            if k.startswith(f"{ck}::") and k != key:
                self._stop_render_key(k)
        task = asyncio.create_task(
            self._render_session_log(chat_id, thread_id, sid),
        )
        self._render_log_tasks[key] = task

    def _stop_render_key(self, key: str) -> None:
        task = self._render_log_tasks.pop(key, None)
        if task is not None and not task.done():
            task.cancel()

    async def _render_session_log(
        self,
        chat_id: int,
        thread_id: int | None,
        sid: str,
    ) -> None:
        """Tail ``sid``'s canonical event stream and render EXTERNAL turns.

        The cross-interface renderer (FIX 1 / spec §8): subscribe to the SAME
        ``mode=events`` stream the local submit path uses, but render only
        turns this chat did NOT initiate locally — so a turn started on web /
        TUI / another chat shows up live in Telegram, satisfying the operator's
        #1 parity goal. Dedupe with the local path via a per-chat high-water
        turn mark + a ``_local_turn_pending`` guard, so each turn renders
        EXACTLY once. Subscribes with full fidelity (``reasoning=stream&tools=
        stream``) and gates client-side via the chat's prefs (the FIX-2 shared
        gating), so the off·on·stream behaviour matches the chat's toggles.

        Cancel-safe + bounded-reconnect: a stream drop reopens from the last
        byte offset (zero loss); cancellation (``/new`` / shutdown) propagates
        cleanly. Never blocks the main poller — it's its own asyncio task.
        """
        ck = TelegramConfig.chat_key(chat_id, thread_id)
        base = self._http_gateway_base()
        headers: dict[str, str] = {
            "X-Sharedagents-Principal": "local",
        }
        if self.acp_bearer_token:
            headers["Authorization"] = f"Bearer {self.acp_bearer_token}"
        stream_base = f"{base}/control/v1/orchestrator/sessions/{sid}/stream"
        # Start at the live offset captured by the first local submit so we
        # never replay the session's pre-existing history into the chat.
        cursor = max(0, self._rendered_offset.get(ck, 0))
        backoff = 0.5
        while True:
            url = f"{stream_base}?mode=events&from={cursor}&reasoning=stream&tools=stream"
            # True once a GET makes progress (a frame, a render, or a reconnect
            # cue). A GET that ends WITHOUT progress (degenerate empty stream)
            # gets a small idle sleep before reopening so we never hot-loop.
            progressed = False
            try:
                async with (
                    httpx.AsyncClient(
                        timeout=None,
                        headers=headers,
                    ) as http,
                    http.stream("GET", url) as sresp,
                ):
                    sresp.raise_for_status()
                    backoff = 0.5  # healthy connection — reset backoff
                    event_name = "log"
                    async for raw in sresp.aiter_lines():
                        line = raw.rstrip("\r")
                        if not line:
                            event_name = "log"
                            continue
                        if line.startswith("event:"):
                            event_name = line[len("event:") :].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if event_name == "hb":
                            continue
                        try:
                            frame = json.loads(payload)
                        except Exception:  # noqa: BLE001
                            continue
                        if event_name == "reconnect":
                            if isinstance(frame, dict):
                                nxt = frame.get("offset")
                                if isinstance(nxt, int) and nxt >= cursor:
                                    cursor = nxt
                            progressed = True
                            break  # reopen from cursor
                        if not isinstance(frame, dict):
                            continue
                        progressed = True
                        off = frame.get("offset")
                        ev = frame.get("event")
                        if not isinstance(ev, dict):
                            if isinstance(off, int) and off > cursor:
                                cursor = off
                            continue
                        if ev.get("type") == "session_state_update":
                            await self._apply_session_state(
                                chat_id,
                                thread_id,
                                sid,
                                ev,
                            )
                            if isinstance(off, int) and off > cursor:
                                cursor = off
                            continue
                        ti = ev.get("turn_idx")
                        if ti is None:
                            # Structural / sessionless frame — advance only.
                            if isinstance(off, int) and off > cursor:
                                cursor = off
                            continue
                        ti = int(ti)
                        hw = self._rendered_turn_count.get(ck, 0)
                        if ti < hw:
                            # Already rendered (local path or an earlier
                            # log-rendered turn) — skip, advance cursor.
                            if isinstance(off, int) and off > cursor:
                                cursor = off
                            continue
                        if self._local_turn_pending.get(ck):
                            # A local submit is in flight (or just landed):
                            # this next turn is locally-owned, the submit
                            # loop renders it. Claim it so we never double-
                            # render, and disarm the guard.
                            self._rendered_turn_count[ck] = max(hw, ti + 1)
                            self._local_turn_pending[ck] = False
                            if isinstance(off, int) and off > cursor:
                                cursor = off
                            continue
                        if ev.get("type") != "turn_started":
                            # Mid-turn frame for a turn we didn't open a
                            # render for — advance past it.
                            if isinstance(off, int) and off > cursor:
                                cursor = off
                            continue
                        # External turn — render it end-to-end, then bump
                        # the high-water past it and resume tailing after.
                        new_cursor = await self._render_external_turn(
                            chat_id,
                            thread_id,
                            sid,
                            ti,
                            ev,
                            int(off) if isinstance(off, int) else cursor,
                        )
                        self._rendered_turn_count[ck] = max(
                            self._rendered_turn_count.get(ck, 0),
                            ti + 1,
                        )
                        if new_cursor > cursor:
                            cursor = new_cursor
                        progressed = True
                        # Reopen the outer tail from the post-turn cursor.
                        break
                # A GET that ended WITHOUT any progress (degenerate empty
                # stream) — idle briefly before reopening so we never hot-loop.
                if not progressed:
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                # Transport blip / session not yet live — bounded backoff,
                # then reopen from the last byte offset (zero loss).
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2, 10.0)

    async def _render_external_turn(
        self,
        chat_id: int,
        thread_id: int | None,
        sid: str,
        turn_idx: int,
        first_event: dict,
        first_offset: int,
    ) -> int:
        """Render ONE externally-started turn (``turn_idx``) into the chat.

        Feeds the already-consumed ``turn_started`` (``first_event``) plus the
        rest of that turn's canonical frames (tailed from ``first_offset``)
        through :meth:`_render_event`, with ``echo_user`` wired so the external
        prompt shows as an inbound bubble, and the chat's reasoning/tools prefs
        applied (FIX-2 shared gating). The render streams into a fresh
        :class:`_RollingMessage` so it lands as its own message(s) — preserving
        the 4096-char rollover. Returns the byte offset to resume tailing from.
        """
        visibility = self._get_visibility(chat_id, thread_id)
        reasoning_level = str(visibility.get("reasoning_visibility") or "off")
        tool_output_level = str(visibility.get("tool_output") or "off").lower()

        roller = None
        if self._api is not None:
            placeholder_id = await self._api.send_message(
                chat_id,
                "💭 thinking…",
                message_thread_id=thread_id,
            )
            if placeholder_id is not None:
                roller = _RollingMessage(
                    self._api,
                    chat_id,
                    placeholder_id=placeholder_id,
                    log=self._log,
                    thread_id=thread_id,
                )

        async def _echo_user(text: str) -> None:
            # Render the external prompt as its own inbound bubble ABOVE the
            # reply. ``send_message`` gives it an earlier message id than the
            # rolling reply (which already has its placeholder), but Telegram
            # orders by id — so send a dedicated bubble. Best-effort.
            if self._api is None:
                return
            try:
                await self._api.send_message(
                    chat_id,
                    f"❯ {text}",
                    message_thread_id=thread_id,
                )
            except Exception:  # noqa: BLE001
                pass

        async def _on_chunk(buf: str, *, force: bool = False) -> None:
            if roller is not None:
                await roller.update(buf, force=force)

        visible_parts: list[str] = []
        ctx = self._build_render_ctx(
            reasoning_level=reasoning_level,
            tool_output_level=tool_output_level,
            visible_parts=visible_parts,
            on_chunk=_on_chunk,
            echo_user=_echo_user,
        )

        # Feed the already-pulled turn_started first.
        await self._render_event(first_event, ctx)

        cursor = first_offset
        base = self._http_gateway_base()
        headers: dict[str, str] = {
            "X-Sharedagents-Principal": "local",
        }
        if self.acp_bearer_token:
            headers["Authorization"] = f"Bearer {self.acp_bearer_token}"
        stream_base = f"{base}/control/v1/orchestrator/sessions/{sid}/stream"
        try:
            while not ctx.stop_state["done"]:
                url = f"{stream_base}?mode=events&from={cursor}&reasoning=stream&tools=stream"
                reopened = False
                async with (
                    httpx.AsyncClient(
                        timeout=None,
                        headers=headers,
                    ) as http,
                    http.stream("GET", url) as sresp,
                ):
                    sresp.raise_for_status()
                    event_name = "log"
                    async for raw in sresp.aiter_lines():
                        line = raw.rstrip("\r")
                        if not line:
                            event_name = "log"
                            continue
                        if line.startswith("event:"):
                            event_name = line[len("event:") :].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if event_name == "hb":
                            continue
                        try:
                            frame = json.loads(payload)
                        except Exception:  # noqa: BLE001
                            continue
                        if event_name == "reconnect":
                            if isinstance(frame, dict):
                                nxt = frame.get("offset")
                                if isinstance(nxt, int) and nxt >= cursor:
                                    cursor = nxt
                            reopened = True
                            break
                        if not isinstance(frame, dict):
                            continue
                        off = frame.get("offset")
                        if isinstance(off, int) and off > cursor:
                            cursor = off
                        ev = frame.get("event")
                        if not isinstance(ev, dict):
                            continue
                        if ev.get("type") == "session_state_update":
                            await self._apply_session_state(
                                chat_id,
                                thread_id,
                                sid,
                                ev,
                            )
                            continue
                        # Defensive: a gateway that replays ``from`` byte-
                        # inclusively could re-serve OUR already-rendered
                        # turn_started — skip it so echo_user fires once.
                        if ev.get("type") == "turn_started" and ev.get("turn_idx") == turn_idx:
                            continue
                        # Only this turn's frames drive the render; a frame
                        # from a later turn means our turn already ended.
                        if ev.get("turn_idx") != turn_idx:
                            if ev.get("type") == "turn_started":
                                ctx.stop_state["done"] = True
                                break
                            continue
                        await self._render_event(ev, ctx)
                        if ctx.stop_state["done"]:
                            break
                if ctx.stop_state["done"]:
                    break
                if not reopened:
                    continue
        finally:
            assembled = "".join(visible_parts).strip()
            if ctx.stop_state.get("stop_reason") not in (None, "end_turn"):
                sr = ctx.stop_state.get("stop_reason")
                assembled = (assembled + "\n\n").lstrip("\n") + f"_(stopped: {sr})_"
            if roller is not None:
                try:
                    await roller.finalize(assembled or "(no content)")
                except Exception:  # noqa: BLE001
                    self._log.exception(
                        "render finalize failed chat=%s",
                        chat_id,
                    )
        await self._deliver_artifacts(
            chat_id,
            thread_id,
            [(sid, event) for event in ctx.artifacts],
        )
        return cursor

    async def _dispatch_blocks(
        self,
        chat_id: int,
        blocks: list,
        *,
        on_chunk: Any | None = None,
        on_think_chunk: Any | None = None,
        on_tool_status: Any | None = None,
        on_split: Any | None = None,
        blockstream: Any | None = None,
        thread_id: int | None = None,
        queue_timeout_s: float | None = None,
        on_queued: Any | None = None,
        on_turn_start: Any | None = None,
        on_session_bound: Any | None = None,
        progress: dict | None = None,
        artifacts: list | None = None,
    ) -> str:
        """Send ``blocks`` to the agent on the chat's session, render reply.

        ``artifacts`` collects the files the turn records as ``(sid, event)``
        pairs. Uploading them is the CALLER's job, after its final paint:
        this task runs under turn supervision, which may cancel it as soon as
        ``turn_ended`` is on disk.

        ``on_chunk`` is an optional async callable invoked with the
        running visible-text buffer (cumulative ``str``) every time an
        ``AgentMessageChunk`` arrives. The long-poll loop wires it to
        :class:`_RollingMessage` so streaming edits land on Telegram in
        real time. Tests typically leave it ``None`` and assert on the
        final return value.

        ``on_split`` is an optional async callable invoked at tool-call
        boundaries (the instant a ``tool_call`` canonical event fires).
        The receiver should finalize the current round's message and
        start a fresh placeholder. If ``None``, no split occurs —
        streaming continues into the same message as before.

        Returns the assembled visible reply (concatenated
        ``agent_message_chunk`` content + any thoughts / tool headers
        the user has opted into). Rendering is gated by the per-
        (chat,thread) visibility map:

        * ``reasoning_visibility`` ∈ ``{"off", "on", "stream"}`` — when
          not ``"off"``, ``agent_thought_chunk`` content is rendered
          inline with a ``💭`` prefix. ``"stream"`` lets thoughts paint
          live; ``"on"`` buffers them until the first message chunk so
          the user sees the answer prefixed by the thinking that led
          to it. Toggle via ``/reasoning``.
        * ``tool_output_visible`` (bool) — when ``True``, every
          ``tool_call``/``tool_call_update`` plants a ``🔧 title…``
          header inline so tool runs are visible in scrollback even
          after they fall off the placeholder. Toggle via ``/tools``.

        Tool-call body deltas (the ``content`` array on
        ``tool_call_update``) always stream into the visible reply
        regardless of ``/tools`` — they ARE the answer for
        direct-dispatch turns like ``/claude`` and ``/codex``. The
        ``/tools`` toggle only governs the inline-header noise.

        The final ``PromptResponse`` adds a ``_(stopped: <reason>)_``
        marker when the stop reason is not ``end_turn`` so the user
        can see what happened.
        """
        import acp.schema as _acp

        # Serialize turns on this (chat, topic). A message that arrives while
        # a turn runs QUEUES here; ``_acquire_turn_slot`` lets the caller mark
        # it queued and drops it if it waits past ``queue_timeout_s`` instead
        # of answering a question the user has long since abandoned. Callers
        # that pass no timeout wait forever, exactly as the old
        # ``async with self.chat_lock`` did.
        _lock = self.chat_lock(chat_id, thread_id)
        if not await self._acquire_turn_slot(
            _lock,
            queue_timeout_s=queue_timeout_s,
            on_queued=on_queued,
        ):
            return _QUEUE_TIMEOUT_REPLY
        try:
            if on_turn_start is not None:
                # Slot acquired → the turn is REALLY starting now. Let the
                # caller flip "queued" → "thinking" + active reaction.
                try:
                    await on_turn_start()
                except Exception:  # noqa: BLE001
                    self._log.debug("on_turn_start raised", exc_info=True)
            client = await self._get_or_open_client(chat_id, thread_id)
            sid = self.get_session_id(chat_id, thread_id)
            visibility = self._get_visibility(chat_id, thread_id)
            reasoning_level = str(visibility.get("reasoning_visibility") or "off")
            if sid is None:
                agent_cfg = self._agent_cfg(chat_id, thread_id)
                sid = await client.session_new(cwd=str(agent_cfg.home))
                self._set_session_id(chat_id, sid, thread_id)
                # Seed the engine's per-session gate from this chat's
                # remembered preference (defaults to "off" — Telegram's
                # per-frontend default). Without this seed the engine
                # would swallow ``thinking`` deltas even when the chat
                # had previously set ``/reasoning on`` — the per-chat
                # visibility map is per-frontend, the engine's gate is
                # per-session.
                if reasoning_level != "off":
                    try:
                        await client.session_set_config_option(
                            sid,
                            "reasoning",
                            reasoning_level,
                        )
                    except Exception:  # noqa: BLE001
                        pass
            # Mirror the Telegram forum topic onto the session so the WebUI
            # Topics page groups it the same way the operator's Telegram client
            # does (General / DMs → "general"). Runs for BOTH a freshly-created
            # session and an existing one — the sync is idempotent per sid, so
            # this only pushes when the resolved topic actually changed (new
            # session, or a forum-topic rename since the last message).
            await self._sync_session_topic(client, chat_id, thread_id, sid)
            if on_session_bound is not None:
                try:
                    await on_session_bound(sid)
                except Exception:  # noqa: BLE001
                    self._log.debug("on_session_bound raised", exc_info=True)
            # 3-state tool-output level: "off" / "on" / "stream".
            #   off    → no header, no body
            #   on     → header only (body suppressed)
            #   stream → header + raw body streamed inline
            # Direct-dispatch tools (delegate_*, compact_context)
            # ignore this gate — see the body-push site below.
            tool_output_level = str(visibility.get("tool_output") or "off").lower()
            # Route Telegram's per-(chat,thread) on/off/stream gating
            # through the SHARED visibility predicate
            # (``sharedagents_agent.delivery.should_show``) instead of bespoke
            # inline ``if level == "off"`` checks — spec §7 "shared
            # visibility filter". The TOOL channels map onto the real
            # canonical event types whose ``should_show`` truth tables
            # match Telegram's exactly:
            #
            #   tool header visible  (show on on+stream, hide off)
            #     → should_show(tool_call,    {"tools": tool_output})
            #   tool body visible    (show only on stream, hide off/on)
            #     → should_show(tool_result, {"tools": tool_output})
            #
            # REASONING (FIX 2): the LIVE thinking-delta channel is gated in
            # ``_render_event`` via ``should_show({"type":"thinking_delta"},
            # {"reasoning": level})`` over the ACTUAL canonical event — which
            # is True ONLY on "stream" (§7). "on" renders the CONSOLIDATED
            # ``final_thinking`` from ``turn_ended`` instead (not live deltas),
            # matching web/TUI. So Telegram's off·on·stream truth table is now
            # IDENTICAL to the shared predicate. ``_REASON_SHOWN`` here means
            # only "is the reasoning channel shown AT ALL" (i.e. not "off") —
            # used by the ``_push_thought`` "off" short-circuit and the legacy
            # ``_run_iter`` fallback; the canonical path's live-vs-final
            # distinction is enforced in ``_render_event``.
            _REASON_SHOWN = reasoning_level != "off"
            tool_show_header = _delivery.should_show(
                {"type": "tool_call"},
                {"tools": tool_output_level},
            )
            # Body gated by should_show over the ACTUAL tool_result event
            # (shown only on stream), NOT the stream-only tool_result_meta — so the
            # local submit path matches web/TUI exactly, same as the render path.
            tool_show_body = _delivery.should_show(
                {"type": "tool_result"},
                {"tools": tool_output_level},
            )
            # Legacy bool kept for backwards-compat with any caller
            # outside this method that still reads the old name.
            tool_output_visible = tool_show_header

            visible_parts: list[str] = []
            # TWO-MESSAGE mode (``on_think_chunk`` wired): reasoning streams
            # into its OWN 💭 message via ``think_parts`` while ``visible_parts``
            # holds ONLY the 💬 answer — so an interleaving model can never
            # fragment the answer (the cropping bug). When ``on_think_chunk`` is
            # None the renderer keeps the legacy single-buffer behaviour.
            two_message = on_think_chunk is not None
            think_parts: list[str] = []
            # Tracks the most recent rendered "kind" so transitions
            # between thinking / content / tool plant a section
            # separator (``\n\n💭`` / ``\n\n💬`` / ``\n\n🔧``) without
            # restacking duplicate prefixes when many small chunks
            # arrive in a row.
            last_kind: str | None = None
            # ``"on"``-mode buffer: thoughts collected while no
            # content has streamed yet. Flushed (with a ``💭`` prefix)
            # the moment the first ``AgentMessageChunk`` lands, so
            # the user sees both the reasoning and the answer in one
            # rolling message.
            pending_thoughts: list[str] = []
            content_seen = False
            # Per-tool-call ids that already had their header planted
            # — keyed so repeated ``tool_call_update`` events for the
            # same call don't restamp ``🔧 title…``.
            headers_emitted: set[str] = set()
            # Per-tool-call titles seen on the initial ``tool_call`` /
            # ``ToolCallStart`` event. ``ToolCallProgress`` updates omit
            # ``title``, so without this map a delegate_claude/codex
            # update would fall back to the generic ``"tool"`` and
            # miss the ``_DIRECT_DISPATCH_TITLES`` gate — the body
            # would be dropped and the chat would render
            # ``⚠️ empty reply``.
            tool_titles: dict[str, str] = {}
            # Per-tool-call ids that already streamed in_progress
            # body deltas; the matching ``completed`` event then skips
            # its final body so we don't double-append.
            streamed_tools: set[str] = set()

            def _flush_pending_thoughts() -> None:
                nonlocal last_kind
                if not pending_thoughts:
                    return
                if visible_parts:
                    visible_parts.append("\n\n")
                visible_parts.append("💭 ")
                visible_parts.append("".join(pending_thoughts))
                pending_thoughts.clear()
                last_kind = "thinking"

            def _push_thought(text: str) -> None:
                nonlocal last_kind
                # Visibility gated by the SHARED predicate (see
                # ``_REASON_SHOWN`` above): shown on "on"/"stream",
                # hidden on "off" — identical to the old
                # ``reasoning_level == "off"`` short-circuit.
                if not text or not _REASON_SHOWN:
                    return
                if two_message:
                    # Reasoning is its OWN message: accumulate contiguously
                    # under a single 💭 prefix. (In "on" mode thinking_delta is
                    # gated out and the consolidated final_thinking is painted
                    # at turn_ended, so this only runs live in "stream".)
                    if not think_parts:
                        think_parts.append("💭 ")
                    think_parts.append(text)
                    return
                if reasoning_level == "on" and not content_seen:
                    # Buffer until first content chunk; mirrors the
                    # legacy bot's `buffering` flag for level "on".
                    pending_thoughts.append(text)
                    return
                # ``"stream"``, OR ``"on"`` after content already
                # appeared — render live with a section header.
                if last_kind != "thinking":
                    if visible_parts:
                        visible_parts.append("\n\n")
                    visible_parts.append("💭 ")
                    last_kind = "thinking"
                visible_parts.append(text)

            def _push_content(text: str) -> None:
                nonlocal last_kind, content_seen
                if not text:
                    return
                if two_message:
                    # Answer is its OWN message: contiguous, single 💬 prefix,
                    # never interleaved with reasoning. A tool header may land
                    # first (no prefix then); prose still extends contiguously.
                    if not visible_parts:
                        visible_parts.append("💬 ")
                    content_seen = True
                    last_kind = "content"
                    visible_parts.append(text)
                    return
                if pending_thoughts:
                    _flush_pending_thoughts()
                if last_kind == "thinking":
                    visible_parts.append("\n\n💬 ")
                    last_kind = "content"
                elif last_kind in (None, "tool", "tool_body"):
                    if visible_parts:
                        visible_parts.append("\n\n")
                    last_kind = "content"
                content_seen = True
                visible_parts.append(text)

            # Tools whose body IS the user-visible answer (the model
            # delegated and we stream the delegate's reply directly).
            # These get a subtle ``↳ via <delegate>`` badge instead of
            # the regular ``🔧 <title>…`` header so the chat reads as a
            # normal sharedagents turn with one extra italic line marking
            # the delegation.
            _DELEGATE_BADGES = {
                "delegate_claude": "↳ via claude",
                "delegate_codex": "↳ via codex",
                "delegate_antigravity": "↳ via antigravity",
            }

            def _push_tool_header(title: str) -> None:
                nonlocal last_kind
                badge = _DELEGATE_BADGES.get(title)
                if badge is not None:
                    # Always show the badge for delegate tools — it's
                    # a one-line attribution, not raw tool output, so
                    # the /tools gate doesn't apply. Render as
                    # markdown italic so it reads as a quiet footnote.
                    if visible_parts:
                        visible_parts.append("\n\n")
                    visible_parts.append(f"_{badge}_")
                    last_kind = "tool"
                    return
                if not tool_output_visible:
                    return
                if visible_parts:
                    visible_parts.append("\n\n")
                visible_parts.append(f"🔧 {title}…")
                last_kind = "tool"

            def _push_teacher_badge(raw_input: Any) -> None:
                """Always-visible teacher-supervision badge (bypasses the
                /tools gate) naming the model + trigger, e.g.
                ``🧑‍🏫 teacher (deepseek-v4-flash): review before idle``.
                Parity with the TUI, which always surfaces the teacher
                delegation node. The model is read from the delegate_teacher
                node's rawInput, so it always reflects the configured teacher."""
                nonlocal last_kind
                ri = raw_input if isinstance(raw_input, dict) else {}
                model = ri.get("model") or "teacher"
                trig = ri.get("trigger") or ""
                trig_label = {
                    "pre_idle": "review before idle",
                    "reasoning_loop": "reasoning loop",
                    "premature_end": "premature end",
                    "error-terminal": "error terminal",
                    "stop-cap": "continuation cap",
                }.get(trig, trig)
                if visible_parts:
                    visible_parts.append("\n\n")
                suffix = f": {trig_label}" if trig_label else ""
                visible_parts.append(f"_🧑‍🏫 teacher ({model}){suffix}_")
                last_kind = "tool"

            def _push_tool_body(text: str) -> None:
                """Append a streaming tool-output chunk.

                The ``🔧 title…`` header (gated by /tools) lands first
                via ``_push_tool_header``; the body itself lands here
                regardless of /tools — it IS the answer for direct-
                dispatch turns. We only insert a leading newline when
                the previous render kind was a header so the body
                starts on its own line under it.
                """
                nonlocal last_kind
                if not text:
                    return
                if last_kind == "tool":
                    visible_parts.append("\n")
                    last_kind = "tool_body"
                elif last_kind not in ("tool_body", None):
                    pass  # mid-stream within a tool body — no separator
                visible_parts.append(text)

            stop_reason: str | None = None

            # --- canonical render context (spec §Project 1) ----------------
            # Package the render closures + visibility gates into the ctx the
            # canonical-stream renderer (``_render_event``) maps onto. The
            # SAME closures serve both the canonical submit/subscribe path and
            # the legacy ACP-iterator fallback, so the rendered output is
            # identical regardless of which driver produced the events.
            ctx = _TelegramRenderCtx(
                push_thought=_push_thought,
                push_content=_push_content,
                push_tool_header=_push_tool_header,
                push_tool_body=_push_tool_body,
                push_teacher_badge=_push_teacher_badge,
                reason_shown=_REASON_SHOWN,
                reasoning_level=reasoning_level,
                # "on"-mode buffers thoughts until content lands; report
                # whether the LAST push actually rendered so the live-paint
                # decision matches the legacy inline check exactly.
                thought_rendered=lambda: not (reasoning_level == "on" and not content_seen),
                tool_show_header=tool_show_header,
                tool_show_body=tool_show_body,
                visible_parts=visible_parts,
                think_parts=think_parts,
                on_think_chunk=on_think_chunk,
                blockstream=blockstream,
                on_chunk=on_chunk,
                on_tool_status=on_tool_status,
                on_split=on_split,
                headers_emitted=headers_emitted,
                tool_titles=tool_titles,
                streamed_tools=streamed_tools,
                progress=progress,
            )

            async def _run_canonical(target_sid: str) -> None:
                """Submit one turn + subscribe to the canonical event stream.

                The production cutover (spec §Project 1): instead of driving
                the ACP ``session/prompt`` iterator and rendering its bespoke
                stream, Telegram POSTs the gateway submit route with the SAME
                content blocks (text + voice-transcribed text + image/file
                attachments), reads ``start_offset``, then tails
                ``GET …/stream?mode=events`` and feeds each canonical event to
                :meth:`_render_event`. This is the IDENTICAL submit→subscribe→
                render(event) path web + TUI use against the same gateway —
                same events, same order, same visibility — so a single session
                behaves the same on all three interfaces, and a turn started
                on web/TUI renders live in Telegram (it's the same log).

                We subscribe with full fidelity (``reasoning=stream&tools=
                stream``) and gate client-side via the ctx flags so Telegram's
                "on" reasoning mode (which the server-side filter would drop —
                thinking_delta is stream-only there) still renders. Ends on
                this turn's ``turn_ended``. Recoverable submit errors (session
                lost) propagate so the outer loop reopens + retries.
                """
                nonlocal stop_reason
                base = self._http_gateway_base()
                headers: dict[str, str] = {
                    "X-Sharedagents-Principal": "local",
                }
                if self.acp_bearer_token:
                    headers["Authorization"] = f"Bearer {self.acp_bearer_token}"
                wire_blocks = self._blocks_to_wire(blocks)
                submit_url = f"{base}/control/v1/orchestrator/sessions/{target_sid}/prompt"
                _ck = TelegramConfig.chat_key(chat_id, thread_id)
                body: dict[str, Any] = {"prompt": wire_blocks}
                # Permission mode: Telegram persists it on the session
                # server-side via ``/mode`` → ``session/set_mode`` (not a
                # per-turn frontend knob), and the submit driver's
                # ``load_session`` restores it before the turn runs — so it
                # rides the session, exactly as it did on the legacy
                # ``session_prompt`` drive. We only stamp the body when a
                # per-turn override is present (none today; reserved).
                pmode = str(visibility.get("permission_mode") or "")
                if pmode:
                    body["permission_mode"] = pmode
                # Arm the renderer dedupe guard BEFORE the submit lands: the
                # turn we're about to create is locally-owned (rendered by
                # THIS loop), so the live render observer must skip it. The
                # guard closes the submit→first-frame race where the renderer
                # could otherwise see the turn_started before we pin its idx.
                self._local_turn_pending[_ck] = True
                # Record the pre-turn log offset so the health supervisor only
                # recovers OUR turn's terminal record, never a prior turn we
                # were queued behind. Records
                # below this offset belong to earlier turns in the session.
                if progress is not None:
                    try:
                        from pathlib import Path as _Path

                        _sd = self._agent_cfg(chat_id, thread_id).sessions_dir
                        _probe_sid = _session_probe_sid(
                            target_sid,
                            _sd,
                        )
                        progress["probe_sid"] = _probe_sid
                        progress["min_offset"] = (_Path(_sd) / f"{_probe_sid}.jsonl").stat().st_size
                    except OSError:
                        progress["min_offset"] = 0
                async with httpx.AsyncClient(
                    timeout=30.0,
                    headers=headers,
                ) as http:
                    resp = await http.post(submit_url, json=body)
                if resp.status_code == 404:
                    # Session unknown to the gateway — recoverable: reopen +
                    # retry (raise the canonical phrase the classifier knows).
                    raise LookupError(f"session not found: {target_sid}")
                if resp.status_code == 409:
                    try:
                        refusal = resp.json()
                    except ValueError:
                        refusal = {}
                    raise RuntimeError(
                        f"{refusal.get('code') or 'session_busy'}: {refusal.get('message') or 'session is busy'}"
                    )
                if resp.status_code not in (200, 202):
                    raise RuntimeError(f"submit failed: HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                start_offset = int(data.get("start_offset") or 0)

                # Lazily start the live cross-interface renderer (FIX 1) the
                # first time this chat submits a turn, anchored at THIS
                # submit's live offset so the observer never replays the
                # session's pre-existing history into the chat — only turns
                # from here on (incl. turns started on web/TUI/another chat).
                self._ensure_render(
                    chat_id,
                    thread_id,
                    target_sid,
                    max(0, start_offset),
                )

                # --- subscribe + render until this turn ends ---------------
                ctx.artifacts = _SessionArtifacts(artifacts, target_sid) if artifacts is not None else []
                cursor = max(0, start_offset)
                stream_base = f"{base}/control/v1/orchestrator/sessions/{target_sid}/stream"
                while not ctx.stop_state["done"]:
                    url = f"{stream_base}?mode=events&from={cursor}&reasoning=stream&tools=stream"
                    reconnect = False
                    async with (
                        httpx.AsyncClient(
                            timeout=None,
                            headers=headers,
                        ) as http,
                        http.stream("GET", url) as sresp,
                    ):
                        sresp.raise_for_status()
                        event_name = "log"
                        async for raw in sresp.aiter_lines():
                            line = raw.rstrip("\r")
                            if not line:
                                event_name = "log"
                                continue
                            if line.startswith("event:"):
                                event_name = line[len("event:") :].strip()
                                continue
                            if not line.startswith("data:"):
                                continue
                            payload = line[len("data:") :].strip()
                            if event_name == "hb":
                                if progress is not None:
                                    progress["last"] = time.monotonic()
                                continue
                            try:
                                frame = json.loads(payload)
                            except Exception:  # noqa: BLE001
                                continue
                            if event_name == "reconnect":
                                if isinstance(frame, dict):
                                    nxt = frame.get("offset")
                                    if isinstance(nxt, int) and nxt >= cursor:
                                        cursor = nxt
                                reconnect = True
                                break
                            if not isinstance(frame, dict):
                                continue
                            off = frame.get("offset")
                            if isinstance(off, int) and off > cursor:
                                cursor = off
                            ev = frame.get("event")
                            if isinstance(ev, dict):
                                if ev.get("type") == "session_state_update":
                                    await self._apply_session_state(
                                        chat_id,
                                        thread_id,
                                        target_sid,
                                        ev,
                                    )
                                await self._render_event(ev, ctx)
                                # Once we know the locally-owned turn idx,
                                # advance the renderer high-water past it and
                                # disarm the submit guard so the observer
                                # never re-renders THIS turn (dedupe).
                                if ctx.turn_idx is not None:
                                    self._rendered_turn_count[_ck] = max(
                                        self._rendered_turn_count.get(_ck, 0),
                                        ctx.turn_idx + 1,
                                    )
                                    self._local_turn_pending[_ck] = False
                                if ctx.stop_state["done"]:
                                    break
                    if ctx.stop_state["done"]:
                        break
                    if not reconnect:
                        # Server generator ended without a reconnect cue (e.g.
                        # lifetime loop exited) before turn_ended — reopen
                        # from the cursor to keep tailing this turn.
                        continue
                # Translate the canonical terminal state into the legacy
                # ``stop_reason`` the assembled-reply tail uses.
                stop_reason = ctx.stop_state.get("stop_reason") or "end_turn"

            async def _run_iter(target_sid: str) -> None:
                """Drive one ``session/prompt`` iteration on ``target_sid``.

                The closure captures every render buffer
                (``visible_parts`` / ``pending_thoughts`` /
                ``headers_emitted`` / ``tool_titles`` /
                ``streamed_tools``) so the retry path doesn't have to
                replumb them. ``stop_reason`` is the only value
                assigned directly in here, hence the lone ``nonlocal``.
                Recoverable errors propagate to the caller — the
                outer loop classifies them and decides whether to
                re-issue the prompt against a freshly opened
                session id.
                """
                nonlocal stop_reason
                had_thinking = False
                async for item in client.session_prompt(target_sid, blocks):
                    if isinstance(item, _acp.SessionNotification):
                        upd = item.update
                        if isinstance(upd, _acp.AgentMessageChunk):
                            content = upd.content
                            piece = content.text if isinstance(content, _acp.TextContentBlock) else ""
                            if piece:
                                # Split at reasoning→content transition
                                # (parity with the canonical path).
                                if had_thinking and on_split is not None:
                                    await on_split()
                                    had_thinking = False
                                _push_content(piece)
                                if on_chunk is not None:
                                    await on_chunk("".join(visible_parts))
                        elif isinstance(upd, _acp.AgentThoughtChunk):
                            content = upd.content
                            piece = content.text if isinstance(content, _acp.TextContentBlock) else ""
                            if piece:
                                had_thinking = True
                                _push_thought(piece)
                                if two_message:
                                    # Reasoning paints its OWN 💭 message.
                                    if on_think_chunk is not None:
                                        await on_think_chunk("".join(think_parts))
                                # Only paint when we actually rendered
                                # — "on"-mode buffering returns early
                                # so the placeholder stays as-is until
                                # content lands.
                                elif (
                                    on_chunk is not None
                                    and _REASON_SHOWN
                                    and not (reasoning_level == "on" and not content_seen)
                                ):
                                    await on_chunk("".join(visible_parts))
                        elif isinstance(upd, _acp.ToolCallStart):
                            title = getattr(upd, "title", None) or "tool"
                            tool_id = getattr(upd, "tool_call_id", "") or ""
                            targs = _parse_tool_args(getattr(upd, "raw_input", None))
                            _apply_tool_watchdog(progress, title, targs, tool_id)
                            if tool_id:
                                tool_titles[tool_id] = title
                            if tool_id and tool_id not in headers_emitted:
                                headers_emitted.add(tool_id)
                                if title == "delegate_teacher":
                                    # Teacher supervision is always surfaced
                                    # (parity with the TUI) — a one-line badge
                                    # naming the model, regardless of /tools.
                                    _push_teacher_badge(getattr(upd, "raw_input", None))
                                elif tool_output_visible:
                                    from sharedagents_agent.tools.runners import (
                                        format_tool_header,
                                    )

                                    _push_tool_header(
                                        format_tool_header(title, targs),
                                    )
                                if on_chunk is not None:
                                    # FORCE the flush regardless of
                                    # header visibility: the agent is
                                    # about to enter the tool which may
                                    # block for seconds-to-minutes with
                                    # no further chunks. A throttled
                                    # drop here would strand the
                                    # assistant's prose tail (and the
                                    # tool header when visible) until
                                    # tool completion. With ``/tools
                                    # off`` (the default) there is no
                                    # header to flush, but the prose
                                    # tail still needs to land NOW.
                                    await on_chunk(
                                        "".join(visible_parts),
                                        force=True,
                                    )
                            # Split at tool-call boundary: finalize this
                            # round's message and start a fresh placeholder.
                            if on_split is not None:
                                await on_split()
                            if on_tool_status is not None:
                                await on_tool_status(title, "in_progress")
                        elif isinstance(upd, _acp.ToolCallProgress):
                            status_val = getattr(upd, "status", "") or ""
                            tool_id = getattr(upd, "tool_call_id", "") or ""
                            # ``tool_call_update`` events omit ``title``;
                            # remember it from the original ``tool_call``
                            # so the direct-dispatch gate below works.
                            update_title = getattr(upd, "title", None) or ""
                            title = update_title or tool_titles.get(tool_id) or "tool"
                            if status_val in ("completed", "failed"):
                                _clear_tool_watchdog(progress, tool_id)
                            if tool_id and update_title:
                                tool_titles[tool_id] = update_title
                            # First sighting of this call id (short-
                            # running tool that skipped ``ToolCallStart``)
                            # — plant the header if /tools is visible,
                            # and force-flush the assistant prose tail
                            # regardless. Same boundary semantics as
                            # ToolCallStart above: the tool may block
                            # the agent for seconds-to-minutes with no
                            # further chunks, so the throttled drop
                            # would otherwise strand the prose tail.
                            if tool_id and tool_id not in headers_emitted:
                                headers_emitted.add(tool_id)
                                if tool_output_visible:
                                    _push_tool_header(title)
                                if on_chunk is not None:
                                    await on_chunk(
                                        "".join(visible_parts),
                                        force=True,
                                    )
                                # Split at tool-call boundary for short-
                                # running tools too (parity with ToolCallStart).
                                if on_split is not None:
                                    await on_split()
                            # Internal classifier/scaffold tools shouldn't
                            # leak their body into the user-visible reply
                            # — those bodies are diagnostic strings like
                            # "plan_pass=False reason=Simple greeting…",
                            # not user-facing content. Header (🔧 …) still
                            # honors the /tools toggle separately.
                            _INTERNAL_TOOL_TITLES = {
                                "plan_classify",
                                "plan_pass",
                            }
                            # Direct-dispatch tools where the body IS the
                            # answer — /claude, /codex, /compact stream
                            # the subprocess output directly into the
                            # chat reply. For these, the body must
                            # land regardless of /tools (the user would
                            # see no reply otherwise) — the module-level
                            # ``_direct_answer_ok`` helper is the single
                            # source of truth (shared with the canonical
                            # renderer), so both transports gate failed
                            # delegations identically.
                            content_list = list(
                                getattr(upd, "content", None) or [],
                            )
                            for entry in content_list:
                                inner = getattr(entry, "content", None)
                                if not isinstance(inner, _acp.TextContentBlock):
                                    continue
                                text_piece = inner.text or ""
                                if not text_piece:
                                    continue
                                if title in _INTERNAL_TOOL_TITLES:
                                    # Skip body push; the header — when
                                    # /tools on — already tells the user
                                    # the classifier ran.
                                    continue
                                # Gate non-direct tool bodies on the
                                # 3-state /tools level:
                                #   off / on  → suppress raw body
                                #   stream    → render raw body
                                # A direct-dispatch body (delegate_*,
                                # compact_context, Agent) IS the answer and
                                # bypasses the gate — but ONLY when it
                                # succeeded. A FAILED direct-dispatch
                                # (status "failed", ok=False) carries internal
                                # diagnostic text, not the answer, and must
                                # NOT leak; it falls back to the normal /tools
                                # gate (shown only in stream mode). Same helper
                                # the canonical renderer uses, so both
                                # transports match.
                                _direct_ok = _direct_answer_ok(
                                    title,
                                    status_val != "failed",
                                )
                                if not _direct_ok and not tool_show_body:
                                    continue
                                if status_val == "in_progress":
                                    if progress is not None:
                                        progress["last"] = time.monotonic()
                                    streamed_tools.add(tool_id)
                                    _push_tool_body(text_piece)
                                elif status_val in ("completed", "failed"):
                                    if tool_id not in streamed_tools:
                                        _push_tool_body(text_piece)
                                if on_chunk is not None:
                                    # FORCE on terminal status: the
                                    # next event may be another heavy
                                    # tool — a throttled drop here
                                    # would hide the tool's final body
                                    # for the entire next gap. Mid-
                                    # stream ``in_progress`` deltas
                                    # stay throttled to avoid 429s on
                                    # chatty tools.
                                    await on_chunk(
                                        "".join(visible_parts),
                                        force=status_val
                                        in (
                                            "completed",
                                            "failed",
                                        ),
                                    )
                            if on_tool_status is not None:
                                await on_tool_status(title, status_val)
                    else:
                        stop_reason = item.stop_reason

            # ---- iterate, with one recovery retry for stale sessions ----
            #
            # The gateway raises ``LookupError: no live child owns
            # session …`` when the session id we send was idle-reaped
            # or the gateway parent itself restarted between turns.
            # The bot's stored sid is fine; the gateway just hasn't
            # re-attached it to a child yet. Recover by issuing
            # ``session/load`` (or ``session/new`` as a fallback) and
            # re-running the prompt once. Only safe BEFORE any
            # content / thoughts / tool headers have streamed —
            # otherwise the retry would re-emit the visible buffer
            # the user has already seen.
            # Driver selection: the production deployment exposes the gateway
            # over HTTP, so we use the canonical submit→subscribe→render(event)
            # path — the same one web + TUI use — which is the parity cutover.
            # The ws/stdio test-fallback transports have no control endpoint,
            # so they keep driving the legacy ACP ``session/prompt`` iterator.
            # Both feed the SAME render closures, so behaviour is identical.
            _gateway_base = self._http_gateway_base()
            _driver = _run_canonical if _gateway_base else _run_iter
            recover_attempted = False
            while True:
                try:
                    await _driver(sid)
                    break
                except Exception as exc:  # noqa: BLE001
                    can_recover = (
                        not recover_attempted
                        and _is_recoverable_session_error(exc)
                        and not visible_parts
                        and not pending_thoughts
                        and not headers_emitted
                    )
                    if not can_recover:
                        self._log.exception(
                            "session_prompt failed chat=%s sid=%s",
                            chat_id,
                            sid,
                        )
                        # Surface a real agent error to the self-improver (it is
                        # blind to bot-boundary failures); transient
                        # connectivity/restart blips are filtered out inside.
                        _maybe_record_agent_error(exc)
                        # Keep whatever already streamed and append the real
                        # error, rather than wiping the message with a bare
                        # pointer.
                        return _assemble_error_reply(
                            visible_parts,
                            pending_thoughts,
                            exc,
                        )
                    recover_attempted = True
                    self._log.warning(
                        "session_prompt: gateway lost session chat=%s sid=%s — attempting reopen (%s)",
                        chat_id,
                        sid,
                        exc,
                    )
                    # If the failure came from a dead transport (SSE
                    # stream torn down by an OOM-kill, etc.) the
                    # cached AcpClient is unusable — its read loop
                    # exited, ``_closed`` is True. Drop it so
                    # ``_get_or_open_client`` builds a fresh
                    # connection on the very next call.
                    if getattr(client, "_closed", False):
                        chat_key = TelegramConfig.chat_key(chat_id, thread_id)
                        self._clients.pop(chat_key, None)
                        try:
                            await client.close()
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            client = await self._get_or_open_client(
                                chat_id,
                                thread_id,
                            )
                        except Exception:  # noqa: BLE001
                            self._log.exception(
                                "client reopen failed chat=%s",
                                chat_id,
                            )
                            return _assemble_error_reply(
                                visible_parts,
                                pending_thoughts,
                                exc,
                            )
                    try:
                        sid = await self._reopen_session_after_loss(
                            client,
                            chat_id,
                            thread_id,
                            sid,
                        )
                        if on_session_bound is not None:
                            try:
                                await on_session_bound(sid)
                            except Exception:  # noqa: BLE001
                                self._log.debug(
                                    "on_session_bound raised after reopen",
                                    exc_info=True,
                                )
                    except Exception:  # noqa: BLE001
                        self._log.exception(
                            "session reopen failed chat=%s",
                            chat_id,
                        )
                        return _assemble_error_reply(
                            visible_parts,
                            pending_thoughts,
                            exc,
                        )
                    # Loop: re-issue the prompt against the fresh sid.

            # Flush any thoughts that were buffered for ``"on"`` mode
            # but never paired with a content chunk (e.g. the model
            # finished mid-thought, or stopped on max_tokens).
            if pending_thoughts:
                _flush_pending_thoughts()

            assembled = "".join(visible_parts).strip()
            if stop_reason and stop_reason != "end_turn":
                assembled = (assembled + "\n\n").lstrip("\n") + f"_(stopped: {stop_reason})_"
            if not assembled:
                assembled = "⚠️ empty reply"
            return assembled
        finally:
            _lock.release()

    async def _keep_typing(
        self,
        chat_id: int,
        thread_id: int | None,
    ) -> None:
        """Keep the "sharedagents is typing…" status visible for a whole turn.

        Telegram clears a chat action after ~5s, so this re-sends it every
        :data:`TYPING_REFRESH_S` seconds. Runs as a background task that the
        caller cancels once the turn finishes.
        """
        if self._api is None:
            return
        while True:
            await self._api.send_chat_action(
                chat_id,
                "typing",
                message_thread_id=thread_id,
            )
            await asyncio.sleep(_bot.TYPING_REFRESH_S)

    async def _stream_turn_for_chat(
        self,
        chat_id: int,
        *,
        text: str | None = None,
        voice_file_id: str | None = None,
        voice_mime: str = "",
        original_filename: str | None = None,
        blocks: list | None = None,
        thread_id: int | None = None,
        untrusted: bool = False,
        incoming_message_id: int | None = None,
    ) -> None:
        """Wire one Telegram message → one ACP turn → streaming edits.

        ``untrusted=True`` marks the inbound content as external data the
        sender merely RELAYED (a forwarded message), not their own prompt —
        it is wrapped in an injection-guard fence so the model treats it as
        data, not instructions.

        ``blocks`` is a pre-built ContentBlock list (the image path:
        ``[TextContentBlock(caption), ImageContentBlock(...)]``). It is
        dispatched as-is — the caller already applied any injection-guard
        fencing, so ``untrusted`` is NOT re-applied (no double-wrap).

        Sends a placeholder, opens the AcpClient (lazy on first message),
        builds the ``ContentBlock`` list (text or transcribed voice), and
        feeds ``session_prompt`` updates into a :class:`_RollingMessage`
        for in-place edits. The final ``PromptResponse`` triggers
        ``finalize`` so the placeholder ends in a meaningful state.

        ``thread_id`` (Telegram ``message_thread_id``) is propagated to
        every send/edit so replies in a forum topic land in the same
        topic, not the General room.
        """
        if self._api is None:
            return

        # Placeholder lifecycle state, shared with the on-turn-start flip
        # below: ``queued`` is True when this reply bubble was sent while a
        # turn was already running on this (chat, topic) — so it must show
        # "queued", not lie with "thinking".
        _ph_state = {"queued": False}
        # The text path uses THIS bubble as the reply; the voice path
        # repurposes it as the transcript echo and opens a fresh reply
        # bubble after STT, so only the text path marks it queued here.
        _reply_here = not voice_file_id
        _queued = _reply_here and self.chat_lock(chat_id, thread_id).locked()
        _ph_state["queued"] = _queued
        # Voice: the first bubble says "🎙️ hearing…" (audio received, transcribing)
        # — not "💭 thinking…", which lies before STT even runs. It is edited to
        # the "🎙️ heard: …" transcript echo after STT, then a fresh "💭 thinking…"
        # reply bubble opens for the turn.
        if voice_file_id:
            _initial_placeholder = _HEARING_PLACEHOLDER
        else:
            _initial_placeholder = _QUEUED_PLACEHOLDER if _queued else _THINKING_PLACEHOLDER
        placeholder_id = await self._api.send_message(
            chat_id,
            _initial_placeholder,
            message_thread_id=thread_id,
            reply_to_message_id=incoming_message_id,
        )
        if _queued:
            await self._set_status_reaction(chat_id, incoming_message_id, _REACT_QUEUED)
        if placeholder_id is None and not voice_file_id:
            self._log.warning(
                "TURN abort chat=%s thread=%s — placeholder send failed",
                chat_id,
                thread_id,
            )
            return
        if placeholder_id is None:
            self._log.warning(
                "TURN continue chat=%s thread=%s — voice placeholder send failed before STT",
                chat_id,
                thread_id,
            )

        # Visibility decides the block-sequence layout: 💭 reasoning (when
        # on/stream), 🔧 tool messages (per /tools), 💬 answer — each its own
        # message in chronological order.
        _vis = self._get_visibility(chat_id, thread_id)
        _reasoning_level = str(_vis.get("reasoning_visibility") or "off")
        _reasoning_shown = _reasoning_level != "off"
        _reasoning_consolidated = _reasoning_level == "on"
        _tool_level = str(_vis.get("tool_output") or "off").lower()
        _tool_show_header = _delivery.should_show({"type": "tool_call"}, {"tools": _tool_level})
        _tool_show_body = _delivery.should_show({"type": "tool_result"}, {"tools": _tool_level})
        _use_rich_drafts = await self._api.is_rich_drafts_enabled()
        roller = (
            _BlockStream(
                self._api,
                chat_id,
                placeholder_id=placeholder_id,
                log=self._log,
                thread_id=thread_id,
                reasoning_shown=_reasoning_shown,
                reasoning_consolidated=_reasoning_consolidated,
                tool_show_header=_tool_show_header,
                tool_show_body=_tool_show_body,
                reply_to=incoming_message_id,
                use_rich_drafts=_use_rich_drafts,
            )
            if placeholder_id is not None
            else None
        )
        _live_ck = TelegramConfig.chat_key(chat_id, thread_id)
        if roller is not None:
            self._live_turns[_live_ck] = {
                "roller": roller,
                "dispatch": None,
                "typing": None,
            }
        t0 = time.monotonic()
        inflight_key: str | None = None
        clear_inflight_on_exit = False

        # Build content blocks. Voice → STT → text block; text → text
        # block; image → pre-built blocks passed straight through.
        import acp.schema as _acp

        blocks = list(blocks) if blocks is not None else []

        if voice_file_id:
            if self._download_audio is None or self._stt is None:  # pragma: no cover — guarded by _handle_update
                if roller is not None:
                    await roller.finalize("⚠️ STT not configured")
                return
            try:
                dl = await self._download_audio(voice_file_id)
            except Exception as _exc:  # noqa: BLE001
                self._log.exception("voice download failed chat=%s", chat_id)
                _record_bot_incident(
                    "media_download_failure", f"voice download raised chat={chat_id}: {type(_exc).__name__}: {_exc}"
                )
                if roller is not None:
                    await roller.finalize("⚠️ couldn't download voice")
                return
            if not dl:
                _record_bot_incident(
                    "media_download_failure",
                    f"voice getFile returned no readable file chat={chat_id} "
                    f"(check telegram-bot-api --local / file root)",
                )
                if roller is not None:
                    await roller.finalize("⚠️ couldn't download voice")
                return
            audio_bytes, filename = dl
            try:
                transcript = await self._stt(
                    audio_bytes,
                    mime=voice_mime,
                    filename=filename,
                )
            except TranscriptionUnavailableError:
                # Transcription could not run at all (the
                # audio could not be decoded -- typically no resolvable
                # ffmpeg). Render a DISTINCT, actionable message so the operator
                # can fix the dependency, instead of the "empty transcription"
                # string a genuinely silent recording gets.
                self._log.warning("stt unavailable (decode) chat=%s", chat_id)
                if roller is not None:
                    await roller.finalize(STT_DECODE_UNAVAILABLE_MSG)
                return
            except Exception:  # noqa: BLE001
                self._log.exception("stt failed chat=%s", chat_id)
                if roller is not None:
                    await roller.finalize("⚠️ transcription failed")
                return
            if not transcript:
                # A genuinely silent / empty recording -- a
                # DISTINCT class from the decode-unavailable case above.
                if roller is not None:
                    await roller.finalize(STT_EMPTY_TRANSCRIPT_MSG)
                return
            self._log.info(
                "STT chat=%s len=%d preview=%r",
                chat_id,
                len(transcript),
                transcript[:120],
            )
            # Persist transcript to disk so call transcripts survive
            # across bot restarts and agent memory sweeps.
            try:
                agent_cfg = self._agent_cfg(chat_id, thread_id)
                xdir = agent_cfg.home / "transcriptions"
                xdir.mkdir(parents=True, exist_ok=True)
                now = datetime.now(timezone.utc)
                ts = now.strftime("%Y-%m-%d_%H-%M-%S")
                # Prefer the original filename from the Telegram message;
                # fall back to the getFile-derived name when absent (voice notes).
                display_stem = Path(original_filename).stem if original_filename else Path(filename).stem
                xpath = xdir / f"{ts}_{display_stem}.md"
                xpath.write_text(
                    f"# {display_stem}\n\n*Transcribed {now.isoformat()} | source: `{filename}`*\n\n{transcript}\n",
                    encoding="utf-8",
                )
                self._log.info("transcript saved to %s", xpath)
            except Exception:  # noqa: BLE001
                self._log.exception("transcript save failed chat=%s", chat_id)

            # Echo the transcript back so the user sees what the model
            # actually received — surfaces STT errors immediately.
            #
            # Telegram orders bubbles by message_id, not by edit time.
            # The reply streams INTO the placeholder, so the echo must
            # occupy an EARLIER message_id to render above it. Sending
            # the echo as a separate message gave it a LATER id — it
            # landed below the reply. Instead, reuse the first
            # placeholder (its earlier id) for the echo, then open a
            # fresh placeholder for the reply.
            async def _render_transcript_echo(echo_text: str) -> None:
                if placeholder_id is not None:
                    try:
                        ok, _retry_after = await self._api.edit_message(
                            chat_id,
                            placeholder_id,
                            echo_text,
                        )
                        if ok:
                            return
                        if _retry_after > 0:
                            try:
                                await asyncio.wait_for(
                                    asyncio.sleep(_retry_after + 0.5),
                                    timeout=_retry_after + 1.0,
                                )
                            except asyncio.TimeoutError:
                                pass
                            ok, _ = await self._api.edit_message(
                                chat_id,
                                placeholder_id,
                                echo_text,
                            )
                            if ok:
                                return
                    except Exception:  # noqa: BLE001
                        self._log.warning("transcript echo edit failed chat=%s", chat_id)
                try:
                    await self._api.send_message(
                        chat_id,
                        echo_text,
                        message_thread_id=thread_id,
                    )
                except Exception:  # noqa: BLE001
                    self._log.warning(
                        "transcript echo fallback send failed chat=%s",
                        chat_id,
                    )

            _LONG = 1500
            if len(transcript) > _LONG:
                _words = len(transcript.split())
                # Preserve the original filename the user sent; swap extension to .txt.
                doc_name = f"{Path(original_filename).stem}.txt" if original_filename else "transcript.txt"
                doc_message_id = None
                try:
                    doc_message_id = await self._api.send_document(
                        chat_id,
                        transcript.encode("utf-8"),
                        filename=doc_name,
                        caption=f"🎙️ transcript ({_words} words)",
                        message_thread_id=thread_id,
                    )
                except Exception:  # noqa: BLE001
                    self._log.warning("transcript doc send failed chat=%s", chat_id)
                if doc_message_id is not None:
                    await _render_transcript_echo(
                        f"🎙️ heard ({_words} words) — full transcript attached",
                    )
                else:
                    echo_text = f"🎙️ heard ({_words} words) — couldn't attach full transcript; sent to agent"
                    await _render_transcript_echo(
                        echo_text,
                    )
            else:
                await _render_transcript_echo(f'🎙️ heard: "{transcript}"')
            # Re-check the slot at the REPLY bubble: STT took time, so the
            # turn that was running when the voice note arrived may have
            # finished — or another may have started — since the echo bubble.
            _queued = self.chat_lock(chat_id, thread_id).locked()
            _ph_state["queued"] = _queued
            reply_placeholder_id = await self._api.send_message(
                chat_id,
                _QUEUED_PLACEHOLDER if _queued else _THINKING_PLACEHOLDER,
                message_thread_id=thread_id,
                reply_to_message_id=incoming_message_id,
            )
            if _queued:
                await self._set_status_reaction(chat_id, incoming_message_id, _REACT_QUEUED)
            if reply_placeholder_id is None:
                self._log.warning(
                    "TURN abort chat=%s thread=%s — reply placeholder send failed",
                    chat_id,
                    thread_id,
                )
                return
            placeholder_id = reply_placeholder_id
            roller = _BlockStream(
                self._api,
                chat_id,
                placeholder_id=placeholder_id,
                log=self._log,
                thread_id=thread_id,
                reasoning_shown=_reasoning_shown,
                reasoning_consolidated=_reasoning_consolidated,
                tool_show_header=_tool_show_header,
                tool_show_body=_tool_show_body,
                reply_to=incoming_message_id,
                use_rich_drafts=_use_rich_drafts,
            )
            _tx = transcript
            if untrusted:
                from sharedagents_core.injection_guard import wrap_untrusted

                _tx = wrap_untrusted(transcript, source="telegram:forwarded-audio")
            blocks.append(
                _acp.TextContentBlock(type="text", text=_tx),
            )
        elif text:
            _tt = text
            if untrusted:
                from sharedagents_core.injection_guard import wrap_untrusted

                _tt = wrap_untrusted(text, source="telegram:forwarded")
            blocks.append(_acp.TextContentBlock(type="text", text=_tt))
        elif not blocks:
            # Nothing to send (no text, no voice, no pre-built blocks).
            await roller.finalize("(no input)")
            return

        # Track whether the agent has streamed any visible content; we
        # use this to decide whether tool-status ticks may safely
        # overwrite the placeholder line.
        content_state = {"streamed": False}
        idle_timeout = _turn_idle_timeout_s()
        # No-progress watchdog state: ``last`` is bumped on every streamed
        # chunk / tool-status / tool_delta tick; a delegate that stalls
        # (the /claude hang: "thinking…" forever, no event) trips it.
        # ``tool_idle_timeout`` extends the budget while a foreground
        # ``Bash`` runs so Telegram never aborts before the tool would.
        _progress = {
            "last": time.monotonic(),
            "stalled": False,
            "supervision_cancel": False,
            "sid": None,
            "probe_sid": None,
            # Byte offset in the session log BELOW which turn records belong to
            # PRIOR turns, not ours. Set at submit time so the health
            # supervisor only recovers / surfaces OUR turn's terminal record —
            # never a queued-behind prior turn's (dup-response guard).
            "min_offset": 0,
            "idle_timeout": idle_timeout,
            "tool_idle_timeout": None,
            "tool_idle_budgets": {},
        }
        # The block-sequence renderer bumps the watchdog on every canonical
        # event (it's driven directly by _render_event, not via _on_chunk).
        if roller is not None:
            roller._progress = _progress

            async def _bs_answer_text(buf: str) -> None:
                content_state["streamed"] = True
                self._update_inflight_text(inflight_key, buf)

            roller._on_answer_text = _bs_answer_text

        async def _on_chunk(buf: str, *, force: bool = False) -> None:
            _progress["last"] = time.monotonic()
            if buf:
                content_state["streamed"] = True
                self._update_inflight_text(inflight_key, buf)
            # ``force=True`` bypasses the throttle/embargo gates inside
            # ``_RollingMessage.update`` (lines 1088-1094). Callers pass
            # it at tool-boundary transitions where the next event is
            # a (potentially heavy) tool that blocks the agent for
            # seconds-to-minutes: without a forced flush, the throttle
            # silently drops the edit (``now - last_edit_at <
            # cur_throttle``) and the tail of the assistant's prose +
            # the tool header stay hidden until the tool completes and
            # another chunk arrives to retry. With ``force``, the
            # edit lands immediately.
            await roller.update(buf, force=force)

        async def _on_think_chunk(buf: str, *, force: bool = False) -> None:
            # Reasoning tokens paint the dedicated 💭 message (kept separate
            # from the 💬 answer so an interleaving model can't fragment it).
            _progress["last"] = time.monotonic()
            await roller.update_think(buf, force=force)

        async def _on_session_bound(sid: str) -> None:
            _progress["sid"] = sid
            nonlocal inflight_key
            if inflight_key is not None:
                entry = self._inflight_turns.get(inflight_key) or {}
                if str(entry.get("sid") or "") == sid:
                    return
                self._clear_inflight_turn(inflight_key)
            agent_cfg = self._agent_cfg(chat_id, thread_id)
            inflight_key = self._record_inflight_turn(
                chat_id=chat_id,
                thread_id=thread_id,
                agent=agent_cfg.name,
                sid=sid,
                placeholder_id=placeholder_id,
            )

        async def _on_tool_status(title: str, status: str) -> None:
            _progress["last"] = time.monotonic()
            # Once real content has streamed in, never let a tool-
            # status tick overwrite it. Pre-prompt phases (plan_classify
            # / plan_pass) tick BEFORE any chunks so they always pass.
            if content_state["streamed"]:
                return
            # Route status ticks through ``update_status`` so they paint an
            # already-open message but never CREATE one — creating the answer
            # bubble here would steal the seed and push reasoning below the
            # answer.
            if title in ("plan_classify", "plan_pass"):
                emoji = "🧠" if title == "plan_classify" else "📐"
                if status == "in_progress":
                    await roller.update_status(f"{emoji} {title}…")
                elif status in ("completed", "failed"):
                    # Phase ended before any real content; restore the
                    # idle placeholder so the user knows the next phase
                    # (real prefill) has started.
                    await roller.update_status("💭 thinking…")
            else:
                # Other tool calls (web_search, exec, etc.) — show in
                # the placeholder until content lands. Consistent UX.
                if status == "in_progress":
                    await roller.update_status(f"🔧 {title}…")

        # Turn supervision: on the HTTP gateway path we poll the session log
        # every ~15s (:func:`_gateway_health_poll_s`) and keep waiting while
        # the orchestrator reports an in-flight turn. Real engine errors
        # surface within one poll; the legacy 300s silence timer applies only
        # to the ws/stdio fallback (no gateway session log).
        idle_timeout = _turn_idle_timeout_s()

        async def _on_turn_start() -> None:
            # Fired the instant ``_dispatch_blocks`` acquires the slot, so the
            # turn is REALLY running now: mark it active, and if this bubble
            # had been showing "queued", flip it to "thinking" before any
            # reply content streams in.
            await self._set_status_reaction(chat_id, incoming_message_id, _REACT_ACTIVE)
            if _ph_state["queued"]:
                try:
                    await self._api.edit_message(chat_id, placeholder_id, _THINKING_PLACEHOLDER)
                except Exception:  # noqa: BLE001
                    pass

        # Show "sharedagents is typing…" for as long as the turn runs. The task
        # loops sending the chat action until the finally block cancels it.
        _typing_task = asyncio.ensure_future(self._keep_typing(chat_id, thread_id))
        _live = self._live_turns.get(_live_ck)
        if _live is not None:
            _live["typing"] = _typing_task

        agent_cfg = self._agent_cfg(chat_id, thread_id)
        _progress["sessions_dir"] = agent_cfg.sessions_dir
        _use_gateway_health = bool(self._http_gateway_base())

        assembled = None
        # Explicit terminal outcome for the lifecycle reaction: one of
        # "success" / "error" / "abort" / "timeout" / "empty". Set in the
        # exception branches; None lets the finally infer from `assembled`
        # for the normal (non-exception) path. This avoids mis-classifying a
        # partial-output error or restart-abort (whose `assembled` starts
        # with the partial content, not ⚠️/⏹) as a success.
        _turn_outcome: str | None = None
        # Files the turn records, as (sid, event); uploaded below after the
        # final paint — outside the supervised dispatch task.
        _artifacts: list = []
        try:
            _dispatch_task = asyncio.ensure_future(
                self._dispatch_blocks(
                    chat_id,
                    blocks,
                    # The block-sequence renderer consumes canonical events
                    # directly in _render_event; on_chunk/on_think_chunk/
                    # on_tool_status are unused on this path.
                    blockstream=roller,
                    thread_id=thread_id,
                    queue_timeout_s=_queue_timeout_s(),
                    on_turn_start=_on_turn_start,
                    on_session_bound=_on_session_bound,
                    progress=_progress,
                    artifacts=_artifacts,
                )
            )
            if _live is not None:
                _live["dispatch"] = _dispatch_task
            assembled = await _await_with_turn_supervision(
                _dispatch_task,
                _progress,
                idle_timeout,
                sessions_dir=agent_cfg.sessions_dir,
                use_gateway_health=_use_gateway_health,
            )
            clear_inflight_on_exit = True
        except asyncio.CancelledError:
            if _progress.get("gateway_error"):
                self._log.warning(
                    "TURN supervision: engine error chat=%s sid=%s err=%r",
                    chat_id,
                    _progress.get("sid"),
                    _progress["gateway_error"],
                )
                _gw_err = RuntimeError(_progress["gateway_error"])
                # Close via BlockStream so 💭/🔧 blocks already painted stay
                # above the 💬 error footer — never ``finalize(full_text)``.
                await roller.turn_ended({}, error=_gw_err)
                assembled = _assemble_error_reply([], [], _gw_err)
                _turn_outcome = "error"
                clear_inflight_on_exit = True
            elif _progress.get("gateway_recover") is not None:
                _fc = _progress["gateway_recover"]
                _aborted = bool(_progress.get("gateway_recover_aborted"))
                self._log.info(
                    "TURN supervision: recovered final reply from log chat=%s sid=%s aborted=%s",
                    chat_id,
                    _progress.get("sid"),
                    _aborted,
                )
                # Reconcile through turn_ended (preserves think→tool→answer
                # order). Pass the recovered turn's REAL aborted flag so an
                # aborted recovery paints via abort_fast (partial content +
                # terminal state) instead of being shown as a clean success.
                await roller.turn_ended({}, final_content=_fc, aborted=_aborted)
                # The stream stalled before the turn's files arrived; the
                # log has them (the upload below de-duplicates by digest).
                _recover_sid = _progress.get("probe_sid") or _progress.get("sid")
                if _recover_sid:
                    _artifacts.extend((_recover_sid, _ev) for _ev in _progress.get("recovered_artifacts") or [])
                # Classify from the REAL terminal state + notice-stripped
                # content, NOT from a truthiness test on the raw final_content
                # (a notice-only recovery is truthy before stripping and would
                # wrongly get 👍; an empty recovery would miss the branch
                # entirely and fall through to a generic cancellation).
                # (codex pre-deploy review round 2, HIGH finding)
                _fc_clean = _delivery.strip_runtime_notices(_fc).strip()
                if _aborted:
                    _turn_outcome = "abort"
                elif _fc_clean:
                    _turn_outcome = "success"
                else:
                    _turn_outcome = "empty"
                assembled = (roller._last_full_text or _fc).strip()
                clear_inflight_on_exit = True
            elif _progress["stalled"]:
                secs = int(_effective_idle_timeout(_progress, idle_timeout))
                self._log.warning(
                    "TURN watchdog: delegate stalled chat=%s (no output for %ds)",
                    chat_id,
                    secs,
                )
                # Per-AGENT stall kind so each agent/engine gets a DISTINCT
                # self-improver fingerprint. A stall in one engine (e.g. cursor's
                # ACP engine) must NOT dedup away a stall in a
                # DIFFERENT engine (e.g. claude's run_delegate_claude_async) —
                # they need separate fixes.
                _stall_agent = self.tg_cfg.agent_for(chat_id, thread_id)
                _record_bot_incident(
                    f"delegate_stall:{_stall_agent}",
                    f"the {_stall_agent} delegate produced no output/tool-status "
                    f"for {secs}s (no turn_ended, no error) chat={chat_id}. The "
                    f"telegram watchdog aborted at {secs}s; the {_stall_agent} "
                    f"engine's own idle timeout never fired in time — find why "
                    f"{_stall_agent} stalls and/or bound its engine read timeout "
                    f"below the {secs}s watchdog.",
                )
                assembled = (
                    f"⚠️ the agent stalled — no output for {secs}s, so I aborted the turn. Try again or switch agents."
                )
                _turn_outcome = "error"
                clear_inflight_on_exit = True
            else:
                if self._stop.is_set():
                    latest = (roller._last_full_text or "").strip()
                    suffix = (
                        "⏹ bot restarted while this turn was running; "
                        "I will recover the final reply when the bot starts again."
                    )
                    assembled = f"{latest}\n\n{suffix}" if latest else suffix
                    _turn_outcome = "abort"  # restart-abort, even with partial content
                else:
                    assembled = "⏹ aborted"
                    _turn_outcome = "abort"
                    clear_inflight_on_exit = True
                raise
        finally:
            self._live_turns.pop(_live_ck, None)
            # Stop the "typing…" keepalive — the turn is finishing.
            _typing_task.cancel()
            elapsed = time.monotonic() - t0
            self._log.info(
                "TURN done chat=%s edits_ok=%d edits_429=%d rollovers=%d throttle=%.1fs t=%.1fs",
                chat_id,
                roller.n_edits_ok,
                roller.n_edits_429,
                roller.n_rollovers,
                roller.cur_throttle,
                elapsed,
            )
            try:
                await roller.finalize(assembled or "(no content)")
            except Exception:  # noqa: BLE001
                self._log.exception("finalize failed chat=%s", chat_id)
            if clear_inflight_on_exit:
                self._clear_inflight_turn(inflight_key)
            # Final lifecycle reaction on the user's message. Classify the
            # terminal outcome EXPLICITLY (set in each exception branch, or
            # inferred here for the normal path) rather than from the
            # `assembled` text prefix — a partial-output error or restart-
            # abort starts with the partial content, not ⚠️/⏹, so prefix
            # inference would mis-classify it as a success (👍). Error / abort
            # / empty envelopes CLEAR the active reaction so the user's
            # message doesn't keep showing "✍️ active" while the session is
            # actually idle.
            if _turn_outcome is None:
                # Normal (non-exception) path. Prefer the REAL terminal
                # outcome propagated from the canonical turn_ended event
                # (via _progress["terminal_outcome"]) — on the block-stream
                # path the assembled-text buffer is always "⚠️ empty reply"
                # (visible_parts is never populated), so text-prefix inference
                # would mis-classify every successful turn as an error. Fall
                # back to assembled-text inference only when no canonical
                # outcome was propagated (e.g. non-blockstream render path).
                _to = _progress.get("terminal_outcome")
                if _to in ("success", "error", "abort", "empty"):
                    _turn_outcome = _to
                elif assembled == _QUEUE_TIMEOUT_REPLY:
                    _turn_outcome = "timeout"
                elif not (assembled or "").strip() or assembled == "(no content)":
                    _turn_outcome = "empty"
                elif assembled.startswith("⚠️"):
                    _turn_outcome = "error"
                elif assembled.startswith("⏹"):
                    _turn_outcome = "abort"
                else:
                    _turn_outcome = "success"
            if _turn_outcome == "timeout":
                await self._set_status_reaction(chat_id, incoming_message_id, _REACT_EXPIRED)
            elif _turn_outcome == "success":
                await self._set_status_reaction(chat_id, incoming_message_id, _REACT_REPLIED)
            else:
                # error / abort / empty: clear the active reaction.
                await self._set_status_reaction(chat_id, incoming_message_id, None)
            # SI-detection (AFTER finalize so the final force-paint's 429 is
            # counted): a turn 429-throttled enough to freeze the live message is
            # a real UX defect the self-improver could not see — the turn still
            # SUCCEEDS, so no error session. Surface it as an incident so the SI
            # fixes the edit cadence. Best-effort, never blocks turn teardown.
            _maybe_record_throttle_incident(
                n_edits_429=roller.n_edits_429,
                n_rollovers=roller.n_rollovers,
                cur_throttle=roller.cur_throttle,
                elapsed=elapsed,
            )
            # Voice reply: when the user spoke, also synthesise the
            # assembled text reply and send it as an audio bubble next
            # to the text. Skips silently if TTS isn't wired, the reply
            # is just an error envelope, or the agent produced nothing.
            if (
                voice_file_id
                and self._tts is not None
                and assembled
                and not assembled.startswith(("⚠️", "(no content)", "⏹", "⌛"))
            ):
                try:
                    audio_bytes = await self._tts(assembled)
                except Exception:  # noqa: BLE001
                    self._log.exception("TTS reply failed chat=%s", chat_id)
                    audio_bytes = None
                if audio_bytes:
                    # WAV-from-local-TTS proxy needs honest mime/filename
                    # (send_voice's MP3 defaults would mislabel and break
                    # Telegram delivery). When we later switch the proxy
                    # to MP3 (or move to OGG/Opus + true sendVoice) these
                    # overrides go away.
                    try:
                        await self._api.send_voice(
                            chat_id,
                            audio_bytes,
                            message_thread_id=thread_id,
                            mime="audio/wav",
                            filename="voice.wav",
                        )
                    except Exception:  # noqa: BLE001
                        self._log.exception(
                            "voice-reply send failed chat=%s",
                            chat_id,
                        )
            # LAST: the files the reply linked. Everything above (inflight
            # record, reaction) is already settled, so a restart cancelling a
            # slow upload skips no cleanup; and a shutting-down bot never
            # starts one.
            if _artifacts and not self._stop.is_set():
                await self._deliver_artifacts(chat_id, thread_id, _artifacts)


# ---------------------------------------------------------------------------
# Shared helpers/constants owned by ``bot.py`` (imported at BOTTOM to keep the
# bot<->mixin import cycle deadlock-free — see module docstring).
# ---------------------------------------------------------------------------
from sharedagents_telegram import bot as _bot  # noqa: E402,F401  (live TYPING_REFRESH_S)
from sharedagents_telegram.attachments import deliver_artifacts  # noqa: E402
from sharedagents_telegram.bot import (  # noqa: E402
    _DIRECT_DISPATCH_TITLES,
    _HEARING_PLACEHOLDER,
    _INTERNAL_TOOL_TITLES,
    _QUEUE_TIMEOUT_REPLY,
    _QUEUED_PLACEHOLDER,
    _REACT_ACTIVE,
    _REACT_EXPIRED,
    _REACT_QUEUED,
    _REACT_REPLIED,
    _THINKING_PLACEHOLDER,
    STT_DECODE_UNAVAILABLE_MSG,
    STT_EMPTY_TRANSCRIPT_MSG,
    TelegramConfig,
    TranscriptionUnavailableError,
    _apply_tool_watchdog,
    _assemble_error_reply,
    _await_with_turn_supervision,
    _BlockStream,
    _clear_tool_watchdog,
    _delivery,
    _direct_answer_ok,
    _effective_idle_timeout,
    _is_recoverable_session_error,
    _maybe_record_agent_error,
    _maybe_record_throttle_incident,
    _metrics_line,
    _parse_tool_args,
    _queue_timeout_s,
    _record_bot_incident,
    _RollingMessage,
    _TelegramRenderCtx,
    _turn_idle_timeout_s,
)
