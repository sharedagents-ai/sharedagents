"""AgentChatApp modal orchestration (permission/question).

Extracted from app.py (de-monolith); behavior-preserving.
"""

from __future__ import annotations

import asyncio
from typing import Any

from textual.containers import VerticalScroll

from sharedagents_tui.widgets.messages import (
    AskUserQuestionMessage,
)
from sharedagents_tui.widgets.modals import (
    AcpPermissionModal,
    PermissionModal,
    QuestionsModal,
)


class ModalMixin:
    """AgentChatApp modal orchestration (permission/question)."""

    # -------- composite status bar --------

    async def _handle_permission_request(self, data: dict) -> None:
        """Pop the modal, POST the user's answer, return.

        The agent has paused its dispatch loop on a Future. We block the
        SSE consumer here too — modal is awaited, response goes back via
        the agent client, then the loop continues. The modal returns one
        of "yes" / "yes_dont_ask" / "no"; on errors / cancel we send "no".
        """
        tid = data.get("turn_id") or self._current_turn_id
        call_id = data.get("call_id")
        if not tid or not call_id:
            return
        try:
            answer = await self.push_screen_wait(
                PermissionModal(
                    tool=data.get("tool", "?"),
                    kind=data.get("kind", "?"),
                    arguments=data.get("arguments") or {},
                )
            )
        except Exception:  # noqa: BLE001
            answer = "no"
        try:
            await self._client.submit_permission(tid, call_id, answer or "no")
        except Exception as e:  # noqa: BLE001
            self._set_status(f"permission send failed: {e}")

    async def _on_ask_user(self, params: dict) -> dict:
        """Pop ``QuestionsModal`` and return selections to ``_inbound_ask_user``.

        Hooked via :meth:`AgentClient.set_ask_user_callback` on every
        client construction. The handler runs inside the ACP client's
        read loop (a background asyncio task, NOT a Textual ``Worker``).

        We CANNOT use ``push_screen_wait`` here — in Textual 8.x it raises
        ``NoActiveWorker`` when called from a plain background task (the
        ``wait_for_dismiss=True`` path checks ``get_current_worker()``).
        Worse, ``push_screen`` ALREADY pushes the modal onto the screen
        stack BEFORE the worker check, so the exception strands a stale
        ``QuestionsModal`` on screen that intercepts keyboard input while
        the exception handler silently auto-answers with first-option
        selections — the TUI appears frozen to the operator.

        Fix: use ``push_screen`` (default ``wait_for_dismiss=False``
        — no worker check) with a callback that resolves an
        ``asyncio.Future`` we await directly. The callback fires from
        Textual's event loop when the modal dismisses, the future
        unblocks our task, and we return the user's selections.

        Before the modal pops we mount an :class:`AskUserQuestionMessage`
        card on the transcript so the question and its outcome stay in
        the conversation log after the modal closes — same persistence
        guarantee as :class:`PlanMessage` for ExitPlanMode. The card is
        stamped via ``mark_answered`` / ``mark_cancelled`` from the
        modal's return value; if mounting itself fails we just skip
        the card and let the modal run — better a missing inline
        record than a stuck agent.
        """
        questions = params.get("questions") if isinstance(params, dict) else []
        if not isinstance(questions, list) or not questions:
            return {"answers": [], "cancelled": True}
        card: AskUserQuestionMessage | None = None
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
            card = AskUserQuestionMessage(questions)
            await transcript.mount(card)
            self._maybe_scroll_end(transcript)
        except Exception:  # noqa: BLE001
            # Transcript not mounted yet (startup race) or the widget
            # failed to compose — fall through to the modal so the
            # agent still gets an answer.
            card = None
        # Use push_screen + callback + future instead of push_screen_wait
        # to avoid NoActiveWorker from a background task.
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        modal = QuestionsModal(questions)

        def _on_dismiss(result: object) -> None:
            if not future.done():
                future.set_result(result or {"answers": [], "cancelled": True})

        try:
            self.push_screen(modal, callback=_on_dismiss)
            result = await future
        except Exception:  # noqa: BLE001
            # Modal push / dismiss failed — the operator never saw the
            # modal, so flipping the card to "cancelled" would lie about
            # who cancelled. Leave the card in pending state; the
            # missing answer line + lingering "waiting…" header signals
            # something went wrong without misattributing the cancel.
            return {"answers": [], "cancelled": True}
        if card is not None:
            try:
                if (
                    not isinstance(result, dict)
                    or not isinstance(result.get("answers"), list)
                    or result.get("cancelled")
                ):
                    card.mark_cancelled()
                else:
                    card.mark_answered(result)
            except Exception:  # noqa: BLE001
                pass
        if not isinstance(result, dict) or not isinstance(result.get("answers"), list):
            return {"answers": [], "cancelled": True}
        return result

    async def _on_request_permission(self, params: dict) -> dict:
        """Pop :class:`AcpPermissionModal` and return the ACP outcome.

        Hooked via :meth:`AgentClient.set_permission_callback`. Like
        :meth:`_on_ask_user` it runs inside the ACP client's read loop,
        so we await the modal directly — the agent's outbound
        ``session/request_permission`` stays parked on its future until
        the user clicks or cancels.

        Uses ``push_screen`` + callback + future instead of
        ``push_screen_wait`` — Textual 8.x raises ``NoActiveWorker``
        when ``wait_for_dismiss=True`` is called from a plain background
        task (the ACP read loop). See :meth:`_on_ask_user` docstring for
        the full rationale.

        Esc / cancel / any error resolves to ``reject_once``: the
        agent only asks because the op is destructive, so "not an
        explicit allow" is a deny.

        Cancellation note: ``push_screen_wait``'s replacement also uses
        ``asyncio.shield`` (via the callback + future pattern without
        Textual's internal shielding), so cancelling our task (e.g. the
        ACP read loop tearing down after the agent subprocess was killed
        by ``/abort`` / Esc) raises ``CancelledError`` out here but
        leaves the modal stranded on the screen stack. We force-pop the
        modal on every exception path so the user is never left behind
        an unresponsive overlay.

        Returns the ACP ``RequestPermissionResponse`` mapping verbatim.
        """
        _DENY = {"outcome": {"outcome": "selected", "optionId": "reject_once"}}
        modal = AcpPermissionModal(params)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def _on_dismiss(result: object) -> None:
            if not future.done():
                future.set_result(result)

        try:
            self.push_screen(modal, callback=_on_dismiss)
            option_id = await future
        except asyncio.CancelledError:
            # Read loop is tearing down; the inbound JSON-RPC request
            # is gone — no one to answer. Pop the modal so the TUI is
            # interactive again, then propagate the cancellation so
            # the read loop teardown finishes cleanly.
            self._force_pop_acp_modal(modal)
            raise
        except Exception:  # noqa: BLE001 - modal teardown / app closing
            self._force_pop_acp_modal(modal)
            return _DENY
        if option_id not in ("allow_once", "allow_always", "reject_once", "reject_always"):
            return _DENY
        return {"outcome": {"outcome": "selected", "optionId": option_id}}

    def _force_pop_acp_modal(self, modal: AcpPermissionModal) -> None:
        """Forcibly remove an :class:`AcpPermissionModal` from the
        screen stack when the awaiter is unwound by something other
        than the user clicking a button — see :meth:`_on_request_permission`
        for why ``push_screen_wait`` doesn't pop on its own. Silent on
        every failure mode: the goal is "user can interact with the
        TUI again", and a stray exception in cleanup would defeat
        that."""
        try:
            modal._resolved = True
        except Exception:  # noqa: BLE001
            pass
        try:
            if modal in self.screen_stack:
                self.pop_screen()
        except Exception:  # noqa: BLE001
            pass
