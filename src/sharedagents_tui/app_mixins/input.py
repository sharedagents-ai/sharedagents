"""AgentChatApp input/queue/key handling.

Extracted from app.py (de-monolith); behavior-preserving.
"""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import ListItem, ListView, Static

from sharedagents_tui.helpers import (
    humanise_delegate_marker,
)
from sharedagents_tui.widgets.input import (
    MessageInput,
)
from sharedagents_tui.widgets.messages import (
    CommandResultMessage,
    UserMessage,
)


class InputMixin:
    """AgentChatApp input/queue/key handling."""

    # ---- prompt / /claude / /codex / /antigravity queue --------------------

    def _is_busy(self) -> bool:
        """True iff a turn (local model, /claude, /codex, or /antigravity) is in flight."""
        return (
            self._current_turn_id is not None
            or self._claude_turn_id is not None
            or self._codex_turn_id is not None
            or self._antigravity_turn_id is not None
        )

    def _enqueue_or_run(self, kind: str, payload: str) -> None:
        """Enqueue a prompt / /claude / /codex call, or run immediately if idle.

        `kind` ∈ {"prompt", "claude", "codex", "antigravity"}. Always
        returns synchronously; the actual work runs as a background
        Textual worker.

        Mid-turn injection — when ``kind == "prompt"`` and a *native*
        sharedagents turn is already streaming on this sid (i.e. no
        delegated /claude / /codex / /antigravity bridge is running), the
        new prompt skips the local queue and gets pushed straight to
        the agent's inject buffer via ``session/inject_user_message``.
        The engine drains injects at every round boundary so the
        steering message lands at the next worker spawn — no waiting
        for the multi-round plan to terminate. The queue stays for
        the delegated-bridge / pre-turn-startup cases where the
        agent can't accept mid-flight steering.
        """
        if kind == "prompt" and self._can_inject_mid_turn() and not self._turn_queue:
            # A native sharedagents turn is live → enqueue durably so the message
            # survives a gateway restart and can be managed via /queue.
            # Fall back to _inject_user_message_via_acp only if the durable
            # path is unavailable (client None / no sid).
            if self._client is not None and self._sid is not None:
                self.run_worker(
                    self._queue_add_via_acp(payload),
                    exclusive=False,
                    name="queue-add",
                )
                return
            self.run_worker(
                self._inject_user_message_via_acp(payload),
                exclusive=False,
                name="inject-user-msg",
            )
            return
        if self._is_busy() or self._turn_queue:
            self._turn_queue.append((kind, payload))
            self._refresh_status_bar()  # show "+N queued" badge
            self._refresh_queue_display()
            self._set_status(f"queued ({len(self._turn_queue)} pending) — will run when current turn finishes")
            return
        self._start_turn_now(kind, payload)

    def _can_inject_mid_turn(self) -> bool:
        """True iff a native sharedagents turn is streaming and no delegated
        bridge (/claude / /codex / /antigravity) is in flight.

        Injecting only makes sense when the engine running ``run_turn``
        is the local LLM — delegated CLI bridges drive their own
        round loops outside the agent and can't be re-steered without
        aborting.
        """
        if self._client is None or self._sid is None:
            return False
        if self._current_turn_id is None:
            return False
        return not (
            self._claude_turn_id is not None or self._codex_turn_id is not None or self._antigravity_turn_id is not None
        )

    async def _inject_user_message_via_acp(self, text: str) -> None:
        """Push a mid-turn user message + render it inline in the transcript.

        Mounts a UserMessage immediately so the operator sees their
        steering input land, then fires the ACP call. On failure
        falls back to the local turn queue so the message isn't lost.
        """
        text = (text or "").strip()
        if not text:
            return
        transcript = self.query_one("#transcript", VerticalScroll)
        # Same humanisation as ``_send_user_turn`` — a
        # ``__SHAREDAGENTS_DELEGATE__:`` payload here would mean a delegate
        # slash command was mid-turn-injected; show the operator's
        # original ``/<tool> <prompt>`` rather than the wire sentinel.
        await transcript.mount(UserMessage(f"❯ {humanise_delegate_marker(text)}"))
        await transcript.mount(
            Static(
                "↳ injected mid-turn — will steer at next round boundary",
                classes="claude-msg",
            )
        )
        # User-action scroll: the operator just submitted, so they
        # explicitly want to see their input at the bottom regardless
        # of where the read-back scroll position was. Re-arm
        # follow-mode so subsequent passive deltas keep auto-
        # scrolling instead of holding the now-stale read-back
        # affinity.
        self._scroll_follow = True
        transcript.scroll_end(animate=False)
        try:
            ack = await self._client.inject_user_message(self._sid, text)
        except Exception as e:  # noqa: BLE001
            ack = {"ok": False, "error": str(e)}
        if not ack.get("ok"):
            # Engine doesn't support injection (e.g. older gateway) or
            # the call errored. Fall back to the post-turn queue.
            self._turn_queue.append(("prompt", text))
            self._refresh_status_bar()
            self._refresh_queue_display()
            self._set_status(
                f"inject not available ({ack.get('error') or ack.get('reason')}) "
                "— message queued for after the current turn"
            )
            return
        self._set_status("↳ injected — will land at next round boundary")

    async def _queue_add_via_acp(self, text: str) -> None:
        """Durably enqueue a prompt while a native turn is in flight.

        Calls ``session_queue_add`` on the backend so the message
        survives a gateway restart and can be inspected/removed via
        ``/queue``. On failure falls back to the local ``_turn_queue``
        exactly like ``_inject_user_message_via_acp``'s error path.
        """
        text = (text or "").strip()
        if not text or self._client is None or self._sid is None:
            return
        try:
            ack = await self._client.session_queue_add(
                self._sid,
                text,
                source="tui",
            )
        except Exception as e:  # noqa: BLE001
            ack = {"ok": False, "error": str(e)}
        if ack.get("ok"):
            self._set_status("🕒 queued")
            # Refresh the widget so the new durable item shows up.
            await self._refresh_durable_queue()
        else:
            # Durable enqueue unavailable (older gateway or error) —
            # fall back to the local post-turn queue so nothing is lost.
            self._turn_queue.append(("prompt", text))
            self._refresh_status_bar()
            self._refresh_queue_display()
            self._set_status(f"durable enqueue failed ({ack.get('error') or 'unknown'}) — added to local queue")

    async def _refresh_durable_queue(self) -> None:
        """Fetch the durable pending list and repaint the queue widget.

        Called after any durable queue mutation (add/remove/update) and
        on ``queue_msg_*`` events from the session log so the widget
        stays in sync with the backend without a manual ``/queue``.
        """
        if self._client is None or self._sid is None:
            return
        try:
            result = await self._client.session_queue_list(self._sid)
        except Exception:  # noqa: BLE001
            return
        self._durable_queue_items = list(result.get("pending") or [])
        self._refresh_queue_display()

    def action_queue_focus_prev(self) -> None:
        """``Ctrl+Up`` — move focus to the queue (or up within it).

        When the queue widget already has focus we let Textual's
        ListView handle the Up navigation natively (highlighting the
        previous row). When focus is elsewhere we hop into the queue
        and select the last row — the operator's "edit the prompt I
        just queued" muscle memory.
        """
        try:
            qview = self.query_one("#queue", ListView)
        except Exception:  # noqa: BLE001
            return
        if not self._turn_queue:
            return
        if qview.styles.display == "none":
            return
        if self.focused is qview:
            qview.action_cursor_up()
            return
        qview.focus()
        # Pick the last row so the most-recently-queued entry is on
        # top of the cursor — matches the "I just queued this and
        # want to fix it" workflow better than highlighting [0].
        qview.index = len(self._turn_queue) - 1

    def action_queue_focus_next(self) -> None:
        """``Ctrl+Down`` — move focus to the queue (or down within it).

        Symmetric to :meth:`action_queue_focus_prev`: from outside the
        queue we focus and select [0]; inside we forward Down to
        ListView so it scrolls one entry.
        """
        try:
            qview = self.query_one("#queue", ListView)
        except Exception:  # noqa: BLE001
            return
        if not self._turn_queue:
            return
        if qview.styles.display == "none":
            return
        if self.focused is qview:
            qview.action_cursor_down()
            return
        qview.focus()
        qview.index = 0

    def on_key(self, event) -> None:  # noqa: ANN001
        """Queue-focused shortcuts: Escape returns to input; Delete
        removes the highlighted row WITHOUT pulling it back into the
        input (a quick "discard" path); Shift+Up / Shift+Down swap
        the highlighted row with the neighbor (priority reordering
        without leaving the keyboard).

        Scroll-affinity keys are also handled here: Shift+Up
        deactivates auto-scroll (same as manual scroll-up), Shift+Down
        scrolls to end and reactivates it, Shift+End jumps to end
        and reactivates it.
        """
        # --- Queue-specific key handling ---
        try:
            qview = self.query_one("#queue", ListView)
        except Exception:  # noqa: BLE001
            qview = None

        if qview is not None and self.focused is qview:
            idx = qview.index
            if idx is not None and idx >= 0 and idx < len(self._turn_queue):
                if event.key == "escape":
                    try:
                        self.query_one(MessageInput).focus()
                    except Exception:  # noqa: BLE001
                        pass
                    event.stop()
                    return
                if event.key == "delete":
                    del self._turn_queue[idx]
                    self._refresh_queue_display()
                    self._refresh_status_bar()
                    self._set_status(f"dropped queue entry — {len(self._turn_queue)} remaining")
                    event.stop()
                    return
                if event.key == "shift+up" and idx > 0:
                    self._turn_queue[idx - 1], self._turn_queue[idx] = (
                        self._turn_queue[idx],
                        self._turn_queue[idx - 1],
                    )
                    self._refresh_queue_display()
                    qview.index = idx - 1
                    event.stop()
                    return
                if event.key == "shift+down" and idx < len(self._turn_queue) - 1:
                    self._turn_queue[idx + 1], self._turn_queue[idx] = (
                        self._turn_queue[idx],
                        self._turn_queue[idx + 1],
                    )
                    self._refresh_queue_display()
                    qview.index = idx + 1
                    event.stop()
                    return

        # --- Scroll-affinity key combos ---
        # PageUp disables auto-scroll and scrolls the transcript up
        # one page (explicit scroll so PageUp isn't consumed by the
        # TextArea cursor handler).
        # PageDown scrolls the transcript down one page and re-enables
        # auto-scroll; Shift+End jumps directly to the bottom.
        if event.key == "pageup":
            self._scroll_follow = False
            self._set_status("auto-scroll: off")
            try:
                transcript = self.query_one("#transcript", VerticalScroll)
                transcript.scroll_page_up(animate=False)
            except Exception:  # noqa: BLE001
                pass
            event.prevent_default()
            event.stop()
            return
        if event.key == "pagedown":
            self._scroll_follow = True
            try:
                transcript = self.query_one("#transcript", VerticalScroll)
                transcript.scroll_page_down(animate=False)
            except Exception:  # noqa: BLE001
                pass
            self._set_status("auto-scroll: on")
            event.prevent_default()
            event.stop()
            return
        if event.key == "shift+end":
            # Don't steal Shift+End from the input TextArea (select-to-end).
            try:
                if self.query_one("#input").has_focus:
                    return  # let TextArea handle it
            except Exception:  # noqa: BLE001
                pass
            self._scroll_follow = True
            try:
                transcript = self.query_one("#transcript", VerticalScroll)
                transcript.scroll_end(animate=False)
            except Exception:  # noqa: BLE001
                pass
            self._set_status("auto-scroll: on")
            event.prevent_default()
            event.stop()
            return

    async def _cmd_queue(self) -> None:
        """``/queue`` — list local + durable pending prompts (Web UI queue)."""
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage("❯ /queue"))
        await self._refresh_durable_queue()
        lines: list[str] = []
        for i, (kind, payload) in enumerate(self._turn_queue):
            preview = humanise_delegate_marker(payload).replace("\n", " ").strip()
            if len(preview) > 160:
                preview = preview[:157] + "…"
            tag = kind if kind in ("claude", "codex", "antigravity") else "prompt"
            lines.append(f"[{i + 1}] [{tag}] {preview}")
        durable = getattr(self, "_durable_queue_items", [])
        if durable:
            if lines:
                lines.append("── durable ──")
            for item in durable:
                qid = item.get("queue_id", "?")
                preview = (item.get("text") or "").replace("\n", " ").strip()
                if len(preview) > 160:
                    preview = preview[:157] + "…"
                lines.append(f"[D:{qid}] {preview}")
        body = "\n".join(lines) if lines else "queue empty"
        await transcript.mount(
            CommandResultMessage("/queue", body, kind="list"),
        )
        self._set_status(body.replace("\n", "  ")[:200])

    def _refresh_queue_display(self) -> None:
        """Repaint the queued-messages ListView from `_turn_queue` and the
        durable queue cache (`_durable_queue_items`).

        Hides the widget while empty so it takes no vertical space; shows
        and rebuilds the rows on every queue mutation. Local rows render
        as ``[N] [kind] text``; durable rows render as ``[D:id] text``
        under a separator. Selecting a local row pulls the text back into
        the input for editing (handled in `on_list_view_selected`).
        """
        try:
            qview = self.query_one("#queue", ListView)
        except Exception:  # noqa: BLE001
            return
        durable = getattr(self, "_durable_queue_items", [])
        if not self._turn_queue and not durable:
            qview.styles.display = "none"
            qview.clear()
            return
        qview.clear()
        for i, (kind, payload) in enumerate(self._turn_queue):
            # Humanise the delegate-marker sentinel when a ``/agent``
            # direct-dispatch payload sits in the queue, otherwise the
            # operator sees the raw ``__SHAREDAGENTS_DELEGATE__:`` JSON in
            # the queue ListView (D1 leak).
            preview = humanise_delegate_marker(payload).replace("\n", " ").strip()
            if len(preview) > 200:
                preview = preview[:197] + "…"
            tag = kind if kind in ("claude", "codex") else "prompt"
            row = f"[{i + 1}] [{tag}] {preview}"
            qview.append(ListItem(Static(row, markup=False)))
        if durable:
            if self._turn_queue:
                # Visual separator between local and durable sections.
                qview.append(ListItem(Static("── durable ──", markup=False)))
            for item in durable:
                qid = item.get("queue_id", "?")
                preview = (item.get("text") or "").replace("\n", " ").strip()
                if len(preview) > 200:
                    preview = preview[:197] + "…"
                row = f"[D:{qid}] {preview}"
                qview.append(ListItem(Static(row, markup=False)))
        qview.styles.display = "block"

    def on_list_view_selected(self, event) -> None:  # noqa: ANN001
        """Click / Enter on a queued row → pull text back into the input.

        Removes the entry from `_turn_queue` and places its payload into
        the message input so the user can edit and resubmit (or just
        clear it). Filters by widget id so it doesn't fire on unrelated
        ListViews mounted elsewhere.
        """
        try:
            lv = event.list_view
        except AttributeError:
            return
        if getattr(lv, "id", None) != "queue":
            return
        idx = lv.index
        if idx is None or idx < 0 or idx >= len(self._turn_queue):
            return
        kind, payload = self._turn_queue[idx]
        del self._turn_queue[idx]
        self._refresh_queue_display()
        self._refresh_status_bar()
        try:
            inp = self.query_one(MessageInput)
            if kind == "claude":
                prefix = "/claude "
            elif kind == "codex":
                prefix = "/codex "
            elif kind == "antigravity":
                prefix = "/antigravity "
            else:
                prefix = ""
            # A ``/agent <sibling>`` direct-dispatch queued the marker
            # sentinel itself as the payload — re-edit must show the
            # operator's original ``/<sibling> <task>`` form, not the
            # ``__SHAREDAGENTS_DELEGATE__:`` wire blob.
            humanised = humanise_delegate_marker(payload)
            inp.value = f"{prefix}{humanised}" if humanised is payload else humanised
            inp.focus()
        except Exception:  # noqa: BLE001
            pass
        self._set_status(f"unqueued — {len(self._turn_queue)} still pending")

    def _start_turn_now(self, kind: str, payload: str) -> None:
        """Spawn the worker for a queued (or directly submitted) entry."""
        if kind == "prompt":
            self.run_worker(
                self._send_user_turn(payload),
                exclusive=False,
                name="user-turn",
            )
        elif kind == "claude":
            self.run_worker(
                self._run_claude(payload),
                exclusive=False,
                name="claude",
            )
        elif kind == "codex":
            self.run_worker(
                self._run_codex(payload),
                exclusive=False,
                name="codex",
            )
        elif kind == "antigravity":
            self.run_worker(
                self._run_antigravity(payload),
                exclusive=False,
                name="antigravity",
            )
        elif kind == "zcode":
            self.run_worker(
                self._run_zcode(payload),
                exclusive=False,
                name="zcode",
            )

    def _maybe_start_next(self) -> None:
        """End-of-turn hook: drain one queued entry, if any.

        Called from the `finally` blocks of `_stream_via_agent` and
        `_run_claude` after they've cleared `_current_turn_id` /
        `_claude_turn_id`. Pops one entry, repaints, and starts it. The
        repaint matters even when the queue is empty: clearing the
        `+N queued` badge.
        """
        self._refresh_status_bar()
        self._refresh_queue_display()
        if self._is_busy():
            return  # another turn is somehow still running; will retry
        if not self._turn_queue:
            return
        kind, payload = self._turn_queue.popleft()
        self._refresh_queue_display()
        self._start_turn_now(kind, payload)
