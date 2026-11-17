"""Telegram streaming renderers — extracted from ``bot.py`` (de-monolith).

Holds the three classes that turn one ACP turn's ``session/update`` stream into
Telegram messages, plus the small render-only helpers/constants they use:

* :class:`_RollingMessage` — rolls a long reply across messages at the 4096-char
  cap instead of truncating, with adaptive 429 edit pacing + rollover.
* :class:`_BlockStream` — renders a turn as an ORDERED SEQUENCE of separate
  messages (💭 reasoning → 🔧 tools → 💬 answer), each its own rolling message.
* :class:`_TelegramRenderCtx` — the render context handed to
  ``AcpTelegramBot._render_event``.

These were extracted verbatim; ``bot.py`` re-exports every public name from here
so tests importing them from ``sharedagents_telegram.bot`` keep working. The shared
low-level constants/helpers they lean on stay owned by ``bot.py`` and are
imported at the BOTTOM of this module (see the note there) to keep the
``bot`` <-> ``rendering`` import cycle deadlock-free in either import order.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from sharedagents_agent import delivery as _delivery
from sharedagents_agent.metrics_format import metrics_line as _metrics_line

if TYPE_CHECKING:  # annotations only (PEP 563 strings at runtime)
    from sharedagents_telegram.api import _TelegramAPI

# Render-owned: referenced at CLASS-DEFINITION time by ``_BlockStream._THINK_IDLE``
# below, so it must exist before the classes execute (can't be bottom-imported).
# ``bot.py`` re-imports this name from here.
_THINKING_PLACEHOLDER = "💭 thinking…"


class _RollingMessage:
    """Streaming renderer that rolls a long reply across messages.

    Telegram caps text at 4096 chars per message. Once the rendered
    buffer outgrows the cap we:

      1. Pick a clean break point (``_split_at``: newline > space > hard).
      2. Force-edit the current placeholder to that head text.
      3. Send a fresh placeholder.
      4. Continue streaming into the new placeholder.

    Throttle/embargo state resets per placeholder because Telegram's
    per-message edit limit is independent. Earlier (frozen) messages
    are never re-edited.
    """

    _global_next_edit_at: float = 0.0

    def __init__(
        self,
        api: _TelegramAPI,
        chat_id: int,
        *,
        placeholder_id: int,
        idle_text: str = "💭 thinking…",
        log: logging.Logger | None = None,
        thread_id: int | None = None,
        use_rich_drafts: bool = False,
    ) -> None:
        self.api = api
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.idle_text = idle_text
        self._log = log or logging.getLogger("sharedagents.telegram")
        self.use_rich_drafts = use_rich_drafts
        self._draft_id = placeholder_id
        self._draft_started = False
        # message_ids[-1] is the active rolling placeholder; earlier
        # entries are frozen and read-only.
        self.message_ids: list[int] = [placeholder_id]
        self.frozen_chars: int = 0
        # The exact text already committed to frozen (read-only) messages.
        # ``frozen_chars == len(frozen_text)`` always. We slice the active
        # placeholder's body off the END of this prefix rather than a bare
        # byte offset so a buffer that gets REBUILT between streaming and
        # finalize (``.strip()``, or the "on"-mode ``final_thinking``
        # prepend at ``turn_ended``) can never misalign the offset and crop
        # the reply down to a stray fragment (the "I got 'A'" bug).
        self.frozen_text: str = ""
        self.last_edit_at: float = 0.0
        self.last_rendered: str = idle_text
        self.cur_throttle: float = _bot.EDIT_THROTTLE_S
        self.embargo_until: float = 0.0
        self.n_edits_ok: int = 0
        self.n_edits_429: int = 0
        self.n_rollovers: int = 0
        # Mid-stream rollovers triggered by sustained edit 429s (distinct from
        # ``n_rollovers``, which counts length-driven message splits). Bounded
        # by ``EDIT_429_MAX_ROLLOVERS`` so a perpetually-throttled chat can't
        # spam an unbounded number of follow-up bubbles.
        self.n_edits_429_rollovers: int = 0
        self._last_full_text: str = ""

    async def _start_draft(self) -> None:
        """Begin a rich-message draft stream (Bot API 10.1).

        Called once after construction when ``use_rich_drafts`` is True.
        Uses the placeholder message id as ``draft_id``. On failure,
        disables draft mode so subsequent updates fall back to edits.
        """
        if not self.use_rich_drafts or self._draft_started:
            return
        ok = await self.api.send_rich_message_draft(
            self.chat_id,
            self._draft_id,
            self.idle_text,
            message_thread_id=self.thread_id,
        )
        if ok:
            self._draft_started = True
        else:
            self.use_rich_drafts = False

    @classmethod
    def _claim_global_edit_slot(cls, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if now < cls._global_next_edit_at:
            return False
        cls._global_next_edit_at = now + _bot.EDIT_GLOBAL_MIN_GAP_S
        return True

    def _unfrozen(self, text: str) -> str:
        """The part of ``text`` not yet committed to a frozen message.

        Normally ``text`` extends the frozen prefix verbatim, so this is
        just ``text[len(frozen_text):]``. If the buffer was REBUILT since
        the last freeze — ``.strip()`` on finalize, or the "on"-mode
        ``final_thinking`` prepend at ``turn_ended`` — ``text`` no longer
        starts with the frozen prefix. Slicing by the raw byte offset then
        crops the reply down to a stray fragment (the "I got 'A'" bug).

        When that happens we RE-KEY ``frozen_text`` down to the longest
        common prefix and return everything after it, so the active
        placeholder (and any follow-up messages) re-show from the first
        divergence point. The already-frozen messages can't be rewritten,
        so the diverged region may repeat — but nothing is ever dropped,
        and overflow draining still terminates (the re-keyed prefix is now
        a true prefix of ``text``).
        """
        ft = self.frozen_text
        if not ft or text.startswith(ft):
            return text[len(ft) :]
        n = min(len(ft), len(text))
        i = 0
        while i < n and ft[i] == text[i]:
            i += 1
        self.frozen_text = text[:i]
        self.frozen_chars = i
        return text[i:]

    def _new_chars_since_render(self, pending: str) -> int:
        prev = self.last_rendered or ""
        if pending.startswith(prev):
            return max(0, len(pending) - len(prev))
        return len(pending)

    async def _commit_and_rollover(self, head: str) -> bool:
        ok = False
        retry_after = 0.0
        now = time.monotonic()
        if self._claim_global_edit_slot(now):
            ok, retry_after = await self.api.edit_rich_message(
                self.chat_id,
                self.message_ids[-1],
                head,
            )
        if ok:
            self.n_edits_ok += 1
        else:
            if retry_after > 0:
                self.n_edits_429 += 1
                self.embargo_until = time.monotonic() + retry_after + 0.5
                cur = float(getattr(self, "cur_throttle", _bot.EDIT_THROTTLE_S))
                self.cur_throttle = min(
                    _bot.EDIT_THROTTLE_MAX_S,
                    max(cur * 2, retry_after + 0.5),
                )
            sent = await self.api.send_rich_message(
                self.chat_id,
                head,
                message_thread_id=self.thread_id,
            )
            if sent is None:
                self._log.warning(
                    "rollover: head send failed chat=%s — overflow held back",
                    self.chat_id,
                )
                return False
            self.message_ids.append(sent)
        new_id = await self.api.send_rich_message(
            self.chat_id,
            "…",
            message_thread_id=self.thread_id,
        )
        if new_id is None:
            self._log.warning(
                "rollover: send failed chat=%s — overflow held back; next paint will retry",
                self.chat_id,
            )
            return False
        self.frozen_chars += len(head)
        self.frozen_text += head
        self.message_ids.append(new_id)
        self.last_edit_at = 0.0
        self.last_rendered = "…"
        self.cur_throttle = _bot.EDIT_THROTTLE_S
        self.embargo_until = 0.0
        self.n_rollovers += 1
        return True

    async def _drain_overflow(self) -> bool:
        full = self._last_full_text or ""
        while True:
            pending = self._unfrozen(full)
            if len(pending) <= _bot.TELEGRAM_MAX_TEXT:
                return True
            cut = _split_at(pending, _bot.TELEGRAM_MAX_TEXT)
            head = pending[:cut].rstrip()
            if not head:
                head = pending[: _bot.TELEGRAM_MAX_TEXT]
                cut = _bot.TELEGRAM_MAX_TEXT
            if not await self._commit_and_rollover(head):
                return False

    async def split(self) -> bool:
        """Force a round boundary at a reasoning→content / tool-call transition.

        Finalizes the current active message — the placeholder the user has
        been watching this round stream into — by FREEZING IT IN PLACE, then
        opens a fresh "💭 thinking…" placeholder for the next round. Only the
        LATEST placeholder is ever edited going forward.

        Freezing in place (force-edit the active message to the full pending
        text, then treat it as read-only) is what keeps the split duplicate-
        free: the round's content is already in ``message_ids[-1]`` from the
        streaming edits, so re-sending it as a separate message would leave
        the streamed placeholder orphaned above an identical frozen copy — the
        reasoning-stream duplication bug. This mirrors ``_commit_and_rollover``
        (the 4096-char overflow path), which has always frozen in place.

        Returns True on success; False if the fresh placeholder send
        failed (the accumulated text is still committed so no data is
        lost — the next round's streaming will restart with a stalled-
        message warning from the caller).
        """
        full = self._last_full_text or ""
        # Drain any overflow (4096-char cap) first so pending is clean.
        if not await self._drain_overflow():
            return False
        pending = self._unfrozen(full)
        # Only roll when the ACTIVE placeholder actually shows this round's
        # content. An empty / idle placeholder (e.g. a tool-first round with
        # ``/tools off``, where nothing has streamed yet) is reused for the
        # next round instead of being frozen as a junk "💭 thinking…" bubble.
        if not pending.strip() or pending.strip() == self.idle_text.strip():
            return True
        # Freeze the ACTIVE placeholder IN PLACE. It is already showing this
        # round's streamed content (``update`` edited it there as the tokens
        # arrived); re-sending that same text as a NEW message would DUPLICATE
        # it. Force the full ``pending`` onto the current message (the live
        # edits may be a throttled step behind), then mark ``message_ids[-1]``
        # frozen / read-only. No extra ``send_message`` for the committed text.
        ok, _retry_after = await self.api.edit_rich_message(
            self.chat_id,
            self.message_ids[-1],
            pending,
        )
        if ok:
            self.n_edits_ok += 1
        frozen_id = self.message_ids[-1]
        self.frozen_chars += len(pending)
        self.frozen_text += pending
        # Send a fresh placeholder for the next round.
        new_id = await self.api.send_rich_message(
            self.chat_id,
            self.idle_text,
            message_thread_id=getattr(self, "thread_id", None),
        )
        if new_id is None:
            self._log.warning(
                "split: new placeholder send failed chat=%s",
                self.chat_id,
            )
            return False
        self.message_ids.append(new_id)
        self.last_edit_at = 0.0
        self.last_rendered = self.idle_text
        self.cur_throttle = _bot.EDIT_THROTTLE_S
        self.embargo_until = 0.0
        # Reset the global edit cadence slot so the new placeholder can
        # be edited immediately on the next round's first chunk.
        _cls = type(self)
        _cls._global_next_edit_at = 0.0
        # Clear _last_full_text so the next round's content is treated
        # as fresh rather than as an extension of the committed round.
        self._last_full_text = ""
        # Fix #1: draft target desync — after rollover the old draft_id
        # points to the frozen message, so disable drafts for the new
        # placeholder. The new placeholder uses edit_rich_message instead.
        if self.use_rich_drafts and self._draft_started:
            self.use_rich_drafts = False
            self._draft_started = False
            self._log.info(
                "split: draft streaming disabled after rollover (old draft_id=%s frozen, new placeholder=%s)",
                self._draft_id,
                new_id,
            )
        self.n_rollovers += 1
        self._log.info(
            "split chat=%s pending_len=%d frozen_msg=%s new_msg=%s",
            self.chat_id,
            len(pending),
            frozen_id,
            new_id,
        )
        return True

    async def update(self, text: str, *, force: bool = False) -> None:
        if not text:
            text = self.idle_text
        self._last_full_text = text
        if not await self._drain_overflow():
            return
        pending = self._unfrozen(text) or self.idle_text
        # Telegram's ``editMessageText`` rejects whitespace-only bodies
        # with ``Bad Request: text must be non-empty``. Models often
        # emit a few leading ``\n``s before real content lands; skip
        # the edit until non-whitespace shows up so we don't burn an
        # EDIT FAIL per chunk and leave the chat stuck on the idle
        # placeholder.
        if not pending.strip():
            return
        if pending == self.last_rendered:
            return
        now = time.monotonic()
        if now < self.embargo_until:
            return
        if self.last_edit_at > 0:
            elapsed = now - self.last_edit_at
            if force and elapsed < _bot.EDIT_FORCE_MIN_GAP_S:
                return
            if not force:
                new_chars = self._new_chars_since_render(pending)
                enough_delta = new_chars >= _bot.EDIT_COALESCE_MIN_CHARS and elapsed >= _bot.EDIT_COALESCE_MIN_GAP_S
                if not enough_delta and elapsed < self.cur_throttle:
                    return
        if not self._claim_global_edit_slot(now):
            return
        if self.use_rich_drafts and self._draft_started:
            ok = await self.api.send_rich_message_draft(
                self.chat_id,
                self._draft_id,
                pending,
                message_thread_id=self.thread_id,
            )
            if ok:
                self.last_edit_at = now
                self.last_rendered = pending
                self.n_edits_ok += 1
                if self.cur_throttle > _bot.EDIT_THROTTLE_S:
                    self.cur_throttle = max(
                        _bot.EDIT_THROTTLE_S,
                        self.cur_throttle * 0.5,
                    )
                return
            # Draft rejected — fall back to in-place rich edits.
            self.use_rich_drafts = False
        ok, retry_after = await self.api.edit_rich_message(
            self.chat_id,
            self.message_ids[-1],
            pending,
        )
        now = time.monotonic()
        if ok:
            self.last_edit_at = now
            self.last_rendered = pending
            self.n_edits_ok += 1
            if self.cur_throttle > _bot.EDIT_THROTTLE_S:
                self.cur_throttle = max(
                    _bot.EDIT_THROTTLE_S,
                    self.cur_throttle * 0.5,
                )
            return
        backoff = self.cur_throttle * 2
        if retry_after > 0:
            self.n_edits_429 += 1
            self.embargo_until = now + retry_after + 0.5
            backoff = max(backoff, retry_after + 0.5)
            # Frozen-bubble escape: once sustained 429s have throttled this
            # message, stop retrying edits on it (they keep failing and the
            # bubble looks stuck). Roll the stream over to a fresh follow-up
            # message — a SEND, whose per-chat limit is INDEPENDENT of the
            # per-message EDIT 429 — so streamed progress stays visible.
            if (
                self.n_edits_429 >= _bot.EDIT_429_ROLLOVER_THRESHOLD
                and self.n_edits_429_rollovers < _bot.EDIT_429_MAX_ROLLOVERS
            ) and await self._rollover_to_followup(pending):
                return
        self.cur_throttle = min(_bot.EDIT_THROTTLE_MAX_S, backoff)
        self.last_edit_at = now

    async def _rollover_to_followup(self, text: str) -> bool:
        """Escape a frozen, edit-throttled placeholder: seal the stuck bubble
        and continue the stream in a fresh message.

        Delegates to the proven length-overflow primitive
        ``_commit_and_rollover``: the stuck message can't be edited (it 429s),
        so that path falls back to a SEND — whose per-chat limit is INDEPENDENT
        of the per-message EDIT 429 — then opens a fresh placeholder and, CRUCIALLY,
        advances ``frozen_text`` past the sealed content. That avoids the
        cumulative duplication a plain ``_send_followup`` would cause: across
        repeated rollovers each new bubble continues from the last seam instead
        of re-sending the whole reply from the start.
        Resets the 429 counter and disables drafts for the rest of the turn.
        Returns True if the rollover succeeded.
        """
        if not text.strip():
            return False
        if not await self._commit_and_rollover(text):
            return False
        self.n_edits_429_rollovers += 1
        # _commit_and_rollover already reset cur_throttle / embargo / last_edit_at
        # and advanced frozen_text; just clear the 429 streak and drafts.
        self.n_edits_429 = 0
        self.use_rich_drafts = False
        self._log.info(
            "edit-429 rollover chat=%s msg=%s n=%d",
            self.chat_id,
            self.message_ids[-1],
            self.n_edits_429_rollovers,
        )
        return True

    async def _send_followup(self, text: str, *, reason: str) -> bool:
        """Send the pending text as a new message instead of editing.

        Used by ``finalize()`` when Telegram refuses edits (embargo or 429).
        This avoids sleeping behind ``retry_after`` and delivers the answer
        immediately.
        """
        send_rich_message = getattr(self.api, "send_rich_message", None)
        if send_rich_message is None:
            return False
        new_id = await send_rich_message(
            self.chat_id,
            text,
            message_thread_id=getattr(self, "thread_id", None),
        )
        if new_id is None:
            self._log.warning(
                "finalize follow-up failed chat=%s reason=%s",
                self.chat_id,
                reason,
            )
            return False
        self.message_ids.append(new_id)
        self.last_edit_at = time.monotonic()
        self.last_rendered = text
        self.embargo_until = 0.0
        self._log.info(
            "finalize follow-up chat=%s msg=%s reason=%s len=%d",
            self.chat_id,
            new_id,
            reason,
            len(text),
        )
        return True

    async def finalize(self, text: str) -> None:
        if not text:
            text = self.idle_text
        self._last_full_text = text
        await self._drain_overflow()
        pending = self._unfrozen(text) or self.idle_text
        # Same Telegram constraint as ``update``: an all-whitespace
        # final reply 400s. If the model produced only newlines /
        # spaces (or nothing), fall back to the idle placeholder so
        # the user at least sees that the turn finished without an
        # API error.
        if not pending.strip():
            pending = self.idle_text
        # The active placeholder ALREADY displays the final text — the
        # ``turn_ended`` force-paint landed it via ``update``. Re-sending it
        # (or even re-editing) would DUPLICATE the reply, which is exactly the
        # "every turn shows the answer twice" bug: finalize used to fall
        # through to a follow-up ``send_message`` below. If the placeholder is
        # already correct, there is nothing to do.
        if pending == self.last_rendered:
            return
        now = time.monotonic()
        embargo_until = float(getattr(self, "embargo_until", 0.0) or 0.0)
        if now < embargo_until:
            # A genuine Telegram edit embargo (429 backoff) — streaming edits
            # have been skipped, so the placeholder is STALE and the only way
            # to deliver the final text is a follow-up message.
            if await self._send_followup(pending, reason="edit_embargo"):
                return
        # Terminal paint of the turn: edit the placeholder DIRECTLY. We do NOT
        # gate this on ``_claim_global_edit_slot`` — that soft client-side slot
        # exists to space out STREAMING edits, and the ``turn_ended``
        # force-paint just consumed it, so claiming here failed on nearly every
        # turn and dropped us into a follow-up ``send_message`` that DUPLICATED
        # the whole reply. One direct edit per turn is safe; a real 429 is
        # handled by the ``retry_after`` branch below.
        #
        # Bot API 10.1: try rich markdown first (enables tables, headings,
        # code blocks, etc.). Falls back to plain text edit automatically
        # if the server doesn't support it or the edit fails.
        #
        # Fix #2: draft finalize drops the reply-link and causes re-order/flicker.
        # The simplest and most reliable path is to skip the draft-finalize
        # (sendRichMessage + delete old) and just edit the existing placeholder
        # in-place with rich markdown. This preserves ordering, reply-links,
        # and avoids the delete/flicker. Drafts are already disabled by default
        # via SHAREDAGENTS_TELEGRAM_RICH_DRAFTS=off.
        ok, retry_after = await self.api.edit_rich_message(
            self.chat_id,
            self.message_ids[-1],
            pending,
        )
        if ok:
            self.last_edit_at = time.monotonic()
            self.n_edits_ok += 1
            self.last_rendered = pending
            return
        if not ok and retry_after > 0:
            # The FINAL force-paint also froze on a 429 — count it so the
            # throttle-incident tally reflects the freeze the user actually saw
            # (06-04: the final paint's 429 was invisible).
            self.n_edits_429 += 1
            now = time.monotonic()
            self.embargo_until = now + retry_after + 0.5
            cur = float(getattr(self, "cur_throttle", _bot.EDIT_THROTTLE_S))
            self.cur_throttle = min(
                _bot.EDIT_THROTTLE_MAX_S,
                max(cur * 2, retry_after + 0.5),
            )
            self.last_edit_at = now
            if await self._send_followup(pending, reason="final_edit_429"):
                return
            try:
                await asyncio.wait_for(
                    asyncio.sleep(retry_after + 0.5),
                    timeout=retry_after + 1.0,
                )
            except asyncio.TimeoutError:
                pass
            ok2, _ = await self.api.edit_rich_message(
                self.chat_id,
                self.message_ids[-1],
                pending,
            )
            if ok2:
                self.n_edits_ok += 1
        self.last_rendered = pending


class _BlockStream:
    """Renders one agent turn as an ORDERED SEQUENCE of separate Telegram
    messages — 💭 reasoning, 🔧 tool calls, 💬 answer — created in the order
    their first token arrives. Telegram orders by message id, so send-order is
    display-order; the sequence reads chronologically:

        💭 r1   💬 r1   🔧 tool   💭 r2   💬 r2   🔧 tool …   💬 final + stats

    Design (ed):
      * Within a round, reasoning accumulates into ONE 💭 message and content
        into ONE 💬 message regardless of interleave order — so a model that
        emits stray thinking mid-answer (qwen) can never fragment the answer
        (the cropping bug). Each block is a single :class:`_RollingMessage`,
        frozen in place at close (no duplication).
      * A ``tool_call`` CLOSES the round's 💭/💬 and opens a 🔧 message, then a
        fresh round begins below it — so tools land in correct sequence as
        their own messages, never bundled into the answer.
      * Ordering (B): on the first answer token the 💭 block for the round is
        opened FIRST (claiming the seed placeholder) whenever reasoning is shown,
        so reasoning always sits above the answer even when an engine (e.g.
        cursor_acp) emits ``text_delta`` before ``thinking_delta``; an empty 💭
        block is collapsed at close.
      * Fairness (C): a round-robin dirty queue spreads the shared global edit
        slot across all live blocks so the busy answer can't starve the 💭
        message (the "idle thinking…" bug).

    Reasoning-off → no 💭 blocks; the first content claims the seed as 💬.
    """

    _THINK_IDLE = _THINKING_PLACEHOLDER  # "💭 thinking…"
    _ANSWER_IDLE = "💬 …"

    def __init__(
        self,
        api: _TelegramAPI,
        chat_id: int,
        *,
        placeholder_id: int | None,
        log: logging.Logger | None = None,
        thread_id: int | None = None,
        reasoning_shown: bool = True,
        reasoning_consolidated: bool = False,
        tool_show_header: bool = True,
        tool_show_body: bool = True,
        progress: dict | None = None,
        on_answer_text: Any | None = None,
        reply_to: int | None = None,
        use_rich_drafts: bool = False,
    ) -> None:
        self._api = api
        self._chat_id = chat_id
        self._log = log or logging.getLogger("sharedagents.telegram")
        self._thread_id = thread_id
        self._use_rich_drafts = use_rich_drafts
        # The user's prompt message id — every block in this turn's sequence
        # is sent as a Telegram reply to it, so each answer links to its
        # question for clarity in a busy chat.
        self._reply_to = reply_to
        self._reasoning_shown = reasoning_shown
        # "on" reasoning: a SINGLE 💭 message pinned to the top of the turn,
        # painted ONCE with the consolidated final_thinking at turn end (the
        # live deltas are gated out upstream). "stream": per-round live 💭.
        self._reasoning_consolidated = reasoning_consolidated
        self._tool_show_header = tool_show_header
        self._tool_show_body = tool_show_body
        self._progress = progress
        self._on_answer_text = on_answer_text
        # The initial placeholder, claimed by the FIRST block opened.
        self._seed: int | None = placeholder_id
        self._blocks: list[dict] = []  # every block, in send order
        self._think: dict | None = None  # current round's 💭 block
        self._answer: dict | None = None  # current round's 💬 block
        self._tools: dict[str, dict] = {}  # call_id → 🔧 block
        self._titles: dict[str, str] = {}
        self._ended = False
        self._aborted = False
        # Hold whitespace-only leading deltas until real content lands, so a
        # stray "\n" before a tool call never opens an empty 💭/💬 bubble.
        self._pend_think = ""
        self._pend_answer = ""
        # Direct-dispatch tools (delegate_*/compact) whose body IS the answer:
        # their output routes into the 💬 answer (never a 🔧 message), with an
        # optional "↳ via <engine>" badge prepended once.
        self._direct_calls: set[str] = set()
        self._pending_badge = ""
        # Cluster block: multiple lightweight tool headers share ONE 🔧
        # message (and thus one sendMessage) to stay under rate limits.
        self._cluster_blk: dict | None = None
        # Round-robin dirty queue: roller-block → pending render text.
        self._dirty: OrderedDict[int, dict] = OrderedDict()
        # #2 — collapsed tool-status fold: ONE rolling 🔧 block that counts
        # every header-only tool call (``🔧 web search ×5 · fetch ×3``) instead
        # of one Telegram message per call. Active only when headers are shown
        # but bodies are NOT streamed (``/tools on``); ``/tools stream`` keeps
        # per-tool blocks so bodies still render, ``/tools off`` shows nothing.
        self._tool_status: dict | None = None
        self._tool_counts: OrderedDict[str, int] = OrderedDict()
        # #1 — compact delegation progress line (``🔎 research — working…``)
        # shown while a direct-dispatch delegate runs; deleted the instant the
        # child's answer starts streaming (or at turn end).
        self._deleg_status: dict | None = None
        # #5 (owner FLIP) — jarvis's OWN reframe is the authoritative answer.
        # The raw delegate/tool_result body is BUFFERED here (never rendered
        # inline); jarvis's own content stream renders as the single answer. If
        # jarvis produces NO own content after the delegation (pure pass-
        # through), ``turn_ended`` flushes the buffered body as a fallback — so
        # the reply is never empty and the body + a reframe never both show.
        self._delegate_body = ""  # buffered raw delegate body
        self._delegate_pending = False  # a buffered body awaits flush/supersede
        self._own_since_delegate = False  # jarvis rendered own content post-delegation
        # Delegate call_ids whose body already streamed via tool_delta, so the
        # terminal tool_result doesn't buffer the full body a second time.
        self._delegate_streamed: set[str] = set()

    # ---- block plumbing ------------------------------------------------
    async def _open(self, idle: str, prefix: str) -> dict | None:
        sid = self._seed
        self._seed = None
        if sid is None:
            if await self._api.is_rich_enabled():
                sid = await self._api.send_rich_message(
                    self._chat_id,
                    idle,
                    message_thread_id=self._thread_id,
                    reply_to_message_id=self._reply_to,
                )
            else:
                sid = await self._api.send_message(
                    self._chat_id,
                    idle,
                    message_thread_id=self._thread_id,
                    reply_to_message_id=self._reply_to,
                )
            if sid is None:
                return None
        rm = _RollingMessage(
            self._api,
            self._chat_id,
            placeholder_id=sid,
            idle_text=idle,
            log=self._log,
            thread_id=self._thread_id,
            use_rich_drafts=self._use_rich_drafts,
        )
        if self._use_rich_drafts:
            await rm._start_draft()
        blk = {"roller": rm, "buf": [], "prefix": prefix, "closed": False}
        self._blocks.append(blk)
        return blk

    @staticmethod
    def _text(blk: dict) -> str:
        body = "".join(blk["buf"])
        return (blk["prefix"] + body) if body.strip() else ""

    def _tick(self) -> None:
        if self._progress is not None:
            self._progress["last"] = time.monotonic()

    def _mark(self, blk: dict | None) -> None:
        if blk is None or blk["closed"]:
            return
        self._dirty[id(blk["roller"])] = blk
        self._dirty.move_to_end(id(blk["roller"]))

    async def _pump(self, *, force: bool = False) -> None:
        """Paint at most one dirty block per call, round-robin, honouring the
        shared global edit slot — so no single block monopolises the cadence."""
        for key in list(self._dirty.keys()):
            blk = self._dirty[key]
            if blk["closed"]:
                self._dirty.pop(key, None)
                continue
            text = self._text(blk)
            if not text:
                self._dirty.pop(key, None)
                continue
            if not force and not _RollingMessage._claim_global_edit_slot():
                return
            self._dirty.pop(key, None)
            await blk["roller"].update(text, force=True)
            if blk is self._answer and self._on_answer_text is not None:
                try:
                    await self._on_answer_text(text)
                except Exception:  # noqa: BLE001
                    pass
            return

    async def _close(self, blk: dict | None) -> None:
        if blk is None or blk["closed"]:
            return
        blk["closed"] = True
        self._dirty.pop(id(blk["roller"]), None)
        text = self._text(blk)
        if text:
            # Direct terminal edit (bypasses the global slot) so a block is
            # never left idle by throttle starvation.
            await blk["roller"].finalize(text)
        elif blk["prefix"].startswith("💭"):
            # A pinned/eager 💭 that never received real reasoning (e.g. an "on"
            # turn whose only output was a delegation) — delete it so the chat
            # isn't left with a stray "💭 thinking…".
            mids = blk["roller"].message_ids
            if mids:
                await self._api.delete_message(self._chat_id, mids[-1])
                blk["deleted"] = True

    async def _close_cluster(self) -> None:
        """Finalize the shared lightweight-tools cluster block, if any."""
        if self._cluster_blk is None:
            return
        blk = self._cluster_blk
        self._cluster_blk = None
        await self._close(blk)

    async def _ensure_top_think(self) -> None:
        """Pin a single 💭 message to the TOP of the turn (claiming the seed)
        for consolidated "on" reasoning, so it sits above any tools/answer."""
        if self._think is None:
            self._think = await self._open(self._THINK_IDLE, "💭 ")

    async def _top_think(self) -> dict | None:
        """The turn's canonical top 💭 block — never below 🔧/💬.

        Reuses the live seed-pinned 💭 block; only creates a new bubble when
        none exists yet. Never revives a closed block (that would duplicate
        terminal finalize edits) and never opens after ``turn_ended``.
        """
        if self._ended:
            return None
        if self._think is not None and not self._think.get("closed"):
            return self._think
        for blk in self._blocks:
            if blk["prefix"].startswith("💭") and not blk.get("deleted") and not blk.get("closed"):
                self._think = blk
                return blk
        await self._ensure_top_think()
        return self._think

    # ---- canonical event handlers --------------------------------------
    async def thinking_delta(self, delta: str) -> None:
        if self._ended:
            return  # terminal state — reject late deltas (round 3, MEDIUM #2)
        if not self._reasoning_shown or not delta:
            return
        self._tick()
        if self._think is None or self._think.get("closed"):
            # Hold whitespace-only deltas until real reasoning arrives.
            self._pend_think += delta
            if not self._pend_think.strip():
                return
            think = await self._top_think()
            if think is None:
                return
            think["buf"].append(self._pend_think)
            self._pend_think = ""
        else:
            self._think["buf"].append(delta)
        self._mark(self._think)
        await self._pump()

    async def text_delta(self, delta: str, *, _from_delegate: bool = False) -> None:
        # Implements #5 (jarvis reframe authoritative + fallback) + #1 (status).
        if self._ended:
            # Terminal state — a late queued delta (e.g. arriving after
            # abort_fast/turn_ended finalized the turn) must NOT open a new
            # 💬 block / send a stranded message. (codex pre-deploy review
            # round 3, MEDIUM #2 — terminal-block invariant.)
            return
        if not delta:
            return
        self._tick()
        if _from_delegate:
            # #5 (owner FLIP) — BUFFER the raw delegate body; do NOT render it
            # inline. jarvis's own content (below) is the authoritative answer.
            # Retire the transient "🔎 … working…" progress line. The buffered
            # body is emitted only as a fallback (turn_ended) if jarvis adds no
            # own content after the delegation.
            self._delegate_body += delta
            self._delegate_pending = True
            self._own_since_delegate = False
            await self._resolve_delegate_status()
            return
        # jarvis's OWN content — the authoritative answer. If it follows a
        # delegation it supersedes the buffered body (no fallback emitted).
        if self._delegate_pending:
            self._own_since_delegate = True
        if self._answer is None:
            # Hold whitespace-only deltas so a stray "\n" before a tool call
            # never opens an empty 💬 bubble (live repro).
            self._pend_answer += delta
            if not self._pend_answer.strip():
                return
            # Ordering (B): PIN a 💭 block above the answer before opening 💬.
            # Consolidated ("on") mode only knows reasoning at turn end; stream
            # mode *usually* receives thinking_delta first, but ACP engines
            # (cursor_acp, etc.) often interleave or lead with text_delta — if
            # we opened 💬 first it would claim the seed and 💭 would land BELOW
            # the answer (Telegram orders by message id). An empty pinned 💭 is
            # collapsed at ``_close`` so turns with no reasoning (e.g. /claude
            # delegation) never leave a stray idle bubble.
            if self._reasoning_shown and self._think is None:
                await self._ensure_top_think()
            self._answer = await self._open(self._ANSWER_IDLE, "💬 ")
            if self._answer is None:
                return
            if self._pending_badge:
                self._answer["buf"].append(f"_{self._pending_badge}_\n\n")
                self._pending_badge = ""
            self._answer["buf"].append(self._pend_answer)
            self._pend_answer = ""
        else:
            self._answer["buf"].append(delta)
        self._mark(self._answer)
        await self._pump()

    async def tool_call(self, call_id: str, name: str, arguments: Any) -> None:
        # Direct-dispatch bodies route to
        # 💬 (with a compact "🔎 working…" progress line); every other shown
        # tool call folds into ONE rolling 🔧 status block (headers-only mode).
        if self._ended:
            return  # terminal state — reject late tool events (round 3, MEDIUM #2)
        self._tick()
        self._titles[call_id] = name
        # Direct-dispatch tools (/claude, /codex, /compact, Agent …): the tool
        # BODY is the user-facing answer, so it must render as 💬 content —
        # ALWAYS, regardless of /tools — never as a 🔧 message (dropping it left
        # the chat stuck on "💭 thinking…", the /claude regression). Mark the
        # call so its body routes to the answer, queue a quiet attribution
        # badge, and show a compact "🔎 <who> — working…" line for the (often
        # long, silent) child turn. (#1)
        if name in _DIRECT_DISPATCH_TITLES:
            self._direct_calls.add(call_id)
            badge = _DELEGATE_BADGES.get(name)
            if badge:
                self._pending_badge = badge
            await self._open_delegate_status(name, arguments)
            return
        # Internal/hidden tools, or /tools off → no visible tool message.
        if name in _INTERNAL_TOOL_TITLES or not self._tool_show_header:
            return
        # A real tool starts a new round: close the round's 💬, keep the top 💭
        # pinned (qwen/vLLM often emits thinking_delta AFTER tool rounds, so
        # closing 💭 here would land a second bubble below 🔧/💬). Same
        # invariant for both reasoning modes.
        if self._reasoning_consolidated and self._reasoning_shown:
            await self._ensure_top_think()
        await self._close(self._answer)
        self._answer = None
        self._pend_think = ""
        self._pend_answer = ""
        if not self._tool_show_body:
            # #2 — headers-only (``/tools on``): fold EVERY shown tool call into
            # ONE rolling 🔧 status block with per-name ×N counts, so N tools
            # cost ONE Telegram message, not N. Supersedes the per-burst
            # lightweight cluster (which still runs in /tools stream below).
            await self._fold_tool_status(call_id, name)
            return
        # /tools stream: keep a per-tool 🔧 block so the body streams into it.
        _args = _parse_tool_args(arguments)
        from sharedagents_agent.tools.widget_adapt import (
            adapt_todo_args,
            format_todo_lines,
            is_todo_tool_name,
        )

        if is_todo_tool_name(name):
            adapted = adapt_todo_args(_args)
            if adapted and adapted.get("todos"):
                await self._close_cluster()
                header = format_todo_lines(adapted["todos"], heading="📝 todos")
                blk = await self._open("📝 todos…", "📝 ")
                if blk is None:
                    return
                blk["buf"].append(header)
                self._tools[call_id] = blk
                self._mark(blk)
                await self._pump(force=True)
                return
        from sharedagents_agent.tools.runners import format_tool_telegram_header

        header = format_tool_telegram_header(name, _args)
        cluster_key = _tool_cluster_key(name)
        if cluster_key is not None:
            # Fold lightweight read-only tools into ONE 🔧 message.
            if self._cluster_blk is None:
                blk = await self._open("🔧 tools…", "🔧 ")
                if blk is None:
                    return
                self._cluster_blk = blk
            else:
                blk = self._cluster_blk
            sep = "\n" if blk["buf"] else ""
            blk["buf"].append(sep + header)
            self._tools[call_id] = blk
            self._mark(blk)
            await self._pump(force=True)
            return
        # Non-clusterable tool: close any open cluster first, then its own 🔧.
        await self._close_cluster()
        blk = await self._open(f"🔧 {name}…", "🔧 ")
        if blk is None:
            return
        blk["buf"].append(header)
        self._tools[call_id] = blk
        self._mark(blk)
        await self._pump(force=True)

    async def _fold_tool_status(self, call_id: str, name: str) -> None:
        """#2 — accumulate ONE tool call into the single rolling 🔧 status
        block (``🔧 web search ×5 · fetch ×3``). All folded calls map to the
        same block so a later ``tool_result`` no-ops instead of closing it."""
        label = _tool_status_label(name)
        self._tool_counts[label] = self._tool_counts.get(label, 0) + 1
        if self._tool_status is None or self._tool_status.get("closed"):
            self._tool_status = await self._open("🔧 …", "🔧 ")
            if self._tool_status is None:
                return
        self._tool_status["buf"] = [_render_tool_counts(self._tool_counts)]
        self._tools[call_id] = self._tool_status
        self._mark(self._tool_status)
        await self._pump(force=True)

    async def _open_delegate_status(self, name: str, arguments: Any) -> None:
        """#1 — open/refresh the compact ``🔎 <who> — working…`` progress line
        for a running delegation. Retired by :meth:`_resolve_delegate_status`
        the instant the child's answer streams (or at turn end)."""
        if self._ended:
            return
        label = _delegate_status_label(name, arguments)
        if self._deleg_status is None or self._deleg_status.get("closed"):
            self._deleg_status = await self._open(label, "")
            if self._deleg_status is None:
                return
        self._deleg_status["buf"] = [label]
        self._mark(self._deleg_status)
        await self._pump(force=True)

    async def _resolve_delegate_status(self) -> None:
        """#1 — retire the transient delegation progress line: the child's
        answer (or turn end) supersedes it, so delete the message rather than
        leaving a stale "🔎 … working…" in the chat."""
        blk = self._deleg_status
        if blk is None:
            return
        self._deleg_status = None
        if blk.get("closed"):
            return
        blk["closed"] = True
        self._dirty.pop(id(blk["roller"]), None)
        mids = blk["roller"].message_ids
        if mids and not blk.get("deleted"):
            try:
                await self._api.delete_message(self._chat_id, mids[-1])
                blk["deleted"] = True
            except Exception:  # noqa: BLE001
                pass

    async def tool_delta(self, call_id: str, delta: str) -> None:
        # Direct-dispatch body streams straight into the 💬 answer.
        if call_id in self._direct_calls:
            # Delegate body streamed live → remember so the terminal
            # tool_result doesn't append the SAME body a second time. (#5)
            self._delegate_streamed.add(call_id)
            await self.text_delta(delta, _from_delegate=True)
            return
        blk = self._tools.get(call_id)
        if blk is None or not delta or not self._tool_show_body:
            return
        # A #2 folded status block carries counts only — never a body.
        if blk is self._tool_status:
            return
        self._tick()
        blk["buf"].append(delta)
        self._mark(blk)
        await self._pump()

    async def tool_result(self, call_id: str, content: str, ok: bool = True) -> None:
        if self._ended:
            return  # terminal state — reject late tool results (round 3, MEDIUM #2)
        if call_id in self._direct_calls:
            # #5 — the delegate body is BUFFERED (not rendered inline). If it
            # already streamed via tool_delta it is already in the buffer, so
            # just retire the progress line. Otherwise (canonical path: no
            # tool_delta, only this result) buffer the body now.
            if call_id in self._delegate_streamed:
                self._delegate_pending = True
                await self._resolve_delegate_status()
                return
            await self.text_delta(content, _from_delegate=True)
            return
        blk = self._tools.get(call_id)
        if blk is None:
            return
        self._tick()
        # #2 folded status block: results never close it (other tools are still
        # counting into the shared block); it finalizes at turn_ended.
        if blk is self._tool_status:
            return
        if content and self._tool_show_body:
            if blk is self._cluster_blk:
                # Keep the cluster open — append a compact result line.
                head = next((ln for ln in content.splitlines() if ln.strip()), "")
                summary = head.strip()[:140]
                arrow = "✗" if not ok else "↳"
                line = f"\n{arrow} {summary}" if summary else ""
                if line:
                    blk["buf"].append(line)
                    self._mark(blk)
                    await self._pump()
            else:
                blk["buf"].append("\n" + content)
                await self._close(blk)
        elif blk is not self._cluster_blk:
            await self._close(blk)

    async def abort_fast(self, msg: str = "⏹ aborted") -> None:
        """Hard abort: drop pending render queue, one minimal terminal edit.

        Used when ``/abort`` fires so we do NOT drain 100+ block finalizes
        (each hitting Telegram rate limits) after the engine is already dead.
        """
        if self._ended:
            return
        self._ended = True
        self._aborted = True
        await self._resolve_delegate_status()
        self._dirty.clear()
        await self._close_cluster()
        for blk in self._blocks:
            blk["closed"] = True
            self._dirty.pop(id(blk["roller"]), None)
        # Pick a LIVE (non-deleted) block to paint the terminal abort message
        # on. The original ``self._answer or self._last_block()`` could return a
        # DELETED block — e.g. a direct-delegate "🔎 … working…" progress
        # block that claimed the seed, then was deleted by
        # ``_resolve_delegate_status`` above, leaving neither a 💬 answer nor
        # an unclaimed seed. Finalizing a deleted block is a no-op, so the
        # user would see neither the partial content nor "⏹ aborted". (codex
        # pre-deploy review round 4, MEDIUM)
        target = self._answer if (self._answer is not None and not self._answer.get("deleted")) else None
        if target is None:
            for blk in reversed(self._blocks):
                if not blk.get("deleted"):
                    target = blk
                    break
        if target is None:
            # No live block exists at all (no 💬 answer, and every existing
            # block is deleted — the delegate-abort case, or the bare-seed
            # no-output abort). Open a NEW 💬 answer block — ``_open`` reuses
            # the seed if it's still set, otherwise sends a fresh message —
            # assign it to ``self._answer``, mark it closed, and paint the
            # abort message so the user sees the terminal state instead of a
            # deleted-seed hole or a stranded "💭 thinking…" placeholder.
            # (round 2/3 MEDIUM #2 + round 4 MEDIUM)
            target = await self._open(self._ANSWER_IDLE, "💬 ")
            if target is not None:
                self._answer = target
                target["closed"] = True
        if target is not None and not target.get("deleted"):
            try:
                await target["roller"].finalize(msg)
            except Exception:  # noqa: BLE001
                pass

    async def turn_ended(
        self,
        metrics: dict | None = None,
        *,
        final_thinking: str = "",
        aborted: bool = False,
        error: Any = None,
        final_content: str = "",
    ) -> None:
        self._tick()
        # #1 — a delegation that finished with no streamed body still leaves a
        # transient "🔎 … working…" line; retire it before the terminal render.
        await self._resolve_delegate_status()
        if aborted:
            # Strip runtime notices from the aborted turn's final_content
            # BEFORE painting via abort_fast, mirroring the normal path's
            # reconciliation stripping — otherwise an aborted recovery such as
            # "partial\n\n↻ runtime: nudge…" paints the notice verbatim, and a
            # notice-only aborted recovery paints the notice instead of "⏹
            # aborted". (codex pre-deploy review round 3, MEDIUM #1)
            _fc_clean = _delivery.strip_runtime_notices(final_content or "").strip()
            await self.abort_fast(_fc_clean or "⏹ aborted")
            return
        if error and not aborted:
            visible: list[str] = []
            if self._answer is not None:
                body = "".join(self._answer["buf"]).strip()
                if body:
                    visible.append(body)
            err_exc = error if isinstance(error, BaseException) else RuntimeError(str(error))
            footer_msg = _assemble_error_reply(visible, [], err_exc)
            if self._answer is None:
                self._answer = await self._open(self._ANSWER_IDLE, "💬 ")
            if self._answer is not None:
                self._answer["buf"] = [footer_msg]
            await self._close_cluster()
            for blk in self._blocks:
                await self._close(blk)
            self._ended = True
            return
        # #5 (owner FLIP) — FALLBACK: jarvis produced no own content after the
        # delegation (pure pass-through, no reframe), so render the BUFFERED raw
        # delegate body as the answer. This guarantees the reply is never empty,
        # and — because jarvis's own content path never re-renders the body —
        # the delegate body and a jarvis reframe are never both shown.
        if self._delegate_pending and not self._own_since_delegate and self._delegate_body.strip() and not self._ended:
            if self._answer is None:
                self._answer = await self._open(self._ANSWER_IDLE, "💬 ")
            if self._answer is not None:
                existing = "".join(self._answer["buf"])
                if not existing.strip() and self._pending_badge:
                    # Pure pass-through (no preamble) — attribute the reply.
                    self._answer["buf"].append(f"_{self._pending_badge}_\n\n")
                    self._pending_badge = ""
                elif existing.strip() and not existing.endswith(("\n", " ")):
                    # A preamble already streamed — separate it from the body.
                    self._answer["buf"].append("\n\n")
                self._answer["buf"].append(self._delegate_body)
        self._delegate_pending = False
        # "on" reasoning (and stream turns with gated live deltas): paint
        # consolidated final_thinking into the TOP 💭 block — never append a
        # fresh 💭 message below 🔧/💬 at turn end. Buffer only; ``_close``
        # below issues one terminal edit (no extra streaming ``_pump``).
        if final_thinking and self._reasoning_shown and not self._ended:
            top = await self._top_think()
            if top is not None and not "".join(top["buf"]).strip():
                # Cap to a single message: this 💭 is pinned ABOVE the turn's
                # already-sent 🔧/💬 messages, so overflow would roll into new
                # messages that land BELOW the answer (order bug). See
                # ``_cap_to_single_message``.
                top["buf"].append(_cap_to_single_message(final_thinking, prefix=top["prefix"]))
        # Reconcile streamed answer with canonical final_content when the
        # live delta stream ended short. Strip
        # runtime-warning paragraphs (↻/⚠ runtime: …) from final_content first
        # — the engine builds it from the raw content stream so it carries the
        # same notice marker the live deltas did, and we don't want the
        # reconciliation to re-introduce "↻ runtime: nudge …" into the answer.
        if final_content and not aborted and not error:
            final_content = _delivery.strip_runtime_notices(final_content)
            if not final_content.strip():
                final_content = ""
            if final_content:
                if self._answer is None:
                    self._answer = await self._open(self._ANSWER_IDLE, "💬 ")
                if self._answer is not None:
                    body = "".join(self._answer["buf"])
                    prefix = "💬 "
                    if body.startswith(prefix):
                        answer_text = body[len(prefix) :]
                        reconciled, patched = _delivery.reconcile_streamed_with_final(
                            answer_text,
                            final_content,
                        )
                        if patched:
                            self._answer["buf"] = [prefix + reconciled.lstrip()]
                    else:
                        reconciled, patched = _delivery.reconcile_streamed_with_final(
                            body,
                            final_content,
                        )
                        if patched:
                            self._answer["buf"] = [reconciled]
        # Stats line on the LAST visible message (prefer the answer).
        if not aborted and not error:
            line = _metrics_line(metrics or {})
            if line:
                target = self._answer or self._last_block()
                if target is None:
                    target = await self._open(self._ANSWER_IDLE, "💬 ")
                if target is not None:
                    sep = "\n\n" if "".join(target["buf"]).strip() else ""
                    target["buf"].append(sep + line)
        # Close every block (direct edits, ordered).
        await self._close_cluster()
        for blk in self._blocks:
            await self._close(blk)
        # If NO block was ever opened (e.g. a notice-only turn whose every
        # text_delta was a runtime notice that the renderer stripped, with no
        # reasoning/metrics either), the pre-sent seed placeholder is still
        # unclaimed — delete it so the chat isn't left with a stray
        # "💭 thinking…" bubble for a turn that produced no visible content.
        # (codex pre-deploy review 2026-07-26, MEDIUM finding)
        if self._seed is not None:
            try:
                await self._api.delete_message(self._chat_id, self._seed)
            except Exception:  # noqa: BLE001
                self._log.debug("seed placeholder delete failed chat=%s", self._chat_id)
            self._seed = None
        self._ended = True

    def _last_block(self) -> dict | None:
        for blk in reversed(self._blocks):
            if "".join(blk["buf"]).strip():
                return blk
        return self._blocks[-1] if self._blocks else None

    # ---- turn-loop compatibility surface -------------------------------
    async def finalize(self, text: str) -> None:
        """Terminal close for the turn-loop's ``finally``. If ``turn_ended``
        already ran this is a no-op; otherwise (abort / stall / error) we put
        ``text`` into the answer block and close everything."""
        if self._ended:
            return
        await self._resolve_delegate_status()
        if text and text.strip():
            if self._answer is None:
                if self._reasoning_shown and self._think is None:
                    self._think = await self._open(self._THINK_IDLE, "💭 ")
                self._answer = await self._open(self._ANSWER_IDLE, "💬 ")
            if self._answer is not None:
                # Replace the answer buffer with the final text (abort/stall
                # envelope) rather than appending to a partial stream.
                self._answer["buf"] = [text]
        for blk in self._blocks:
            await self._close(blk)
        self._ended = True

    @property
    def _last_full_text(self) -> str:
        if self._answer is not None:
            return self._answer["roller"]._last_full_text
        return self._last_block()["roller"]._last_full_text if self._blocks else ""

    @property
    def message_ids(self) -> list:
        ids: list = []
        for blk in self._blocks:
            ids += blk["roller"].message_ids
        return ids

    def _agg(self, attr: str) -> int:
        return sum(int(getattr(b["roller"], attr, 0) or 0) for b in self._blocks)

    @property
    def n_edits_ok(self) -> int:
        return self._agg("n_edits_ok")

    @property
    def n_edits_429(self) -> int:
        return self._agg("n_edits_429")

    @property
    def n_rollovers(self) -> int:
        return self._agg("n_rollovers")

    @property
    def cur_throttle(self) -> float:
        vals = [b["roller"].cur_throttle for b in self._blocks]
        return max(vals) if vals else _bot.EDIT_THROTTLE_S


class _TelegramRenderCtx:
    """Render context handed to :meth:`AcpTelegramBot._render_event`.

    Packages the per-turn render callbacks + visibility gates that the
    canonical-event renderer maps onto. ``_dispatch_blocks`` builds one of
    these over its own closures so the SAME renderers the legacy ACP-iterator
    drive used (``_push_thought`` / ``_push_content`` / ``_push_tool_header``
    / ``_push_tool_body`` / ``_push_teacher_badge``) are reused verbatim — the
    canonical cutover changes only the *source* of events (the gateway log
    stream), never the rendering. Hermetically testable: a test can construct
    one with fake callbacks and drive ``_render_event`` directly.

    The fields mirror the closures/flags inside ``_dispatch_blocks``:

    * ``push_thought(text)`` — render a thinking chunk (gated by reasoning).
    * ``push_content(text)`` — render an answer chunk (always visible).
    * ``push_tool_header(title)`` — plant a ``🔧 title…`` header (gated tools).
    * ``push_tool_body(text)`` — append raw tool body (direct-dispatch / stream).
    * ``push_teacher_badge(raw_input)`` — always-visible teacher badge.
    * ``reason_shown`` — bool: are thoughts shown at all (on/stream)?
    * ``tool_show_header`` / ``tool_show_body`` — the /tools gates.
    * ``visible_parts`` — the running render buffer (read for ``on_chunk``).
    * ``on_chunk`` / ``on_tool_status`` — async paint callbacks (may be None).
    * ``headers_emitted`` / ``tool_titles`` / ``streamed_tools`` — per-turn
      dedupe state shared across events.
    * ``turn_idx`` — the turn this dispatch owns; events from other turns
      (rendered / prior) are ignored so the per-turn loop stays scoped.
    """

    __slots__ = (
        "push_thought",
        "push_content",
        "push_tool_header",
        "push_tool_body",
        "push_teacher_badge",
        "reason_shown",
        "reasoning_level",
        "thought_rendered",
        "tool_show_header",
        "tool_show_body",
        "visible_parts",
        "on_chunk",
        "on_tool_status",
        "on_split",
        "headers_emitted",
        "tool_titles",
        "streamed_tools",
        "turn_idx",
        "stop_state",
        "echo_user",
        "progress",
        "had_thinking",
        "think_parts",
        "on_think_chunk",
        "blockstream",
        "artifacts",
    )

    def __init__(
        self,
        *,
        push_thought,
        push_content,
        push_tool_header,
        push_tool_body,
        push_teacher_badge,
        reason_shown,
        tool_show_header,
        tool_show_body,
        visible_parts,
        reasoning_level="off",
        thought_rendered=None,
        on_chunk=None,
        on_tool_status=None,
        on_split=None,
        headers_emitted=None,
        tool_titles=None,
        streamed_tools=None,
        turn_idx=None,
        echo_user=None,
        progress=None,
        think_parts=None,
        on_think_chunk=None,
        blockstream=None,
    ) -> None:
        self.push_thought = push_thought
        self.push_content = push_content
        self.push_tool_header = push_tool_header
        self.push_tool_body = push_tool_body
        self.push_teacher_badge = push_teacher_badge
        self.reason_shown = reason_shown
        self.reasoning_level = reasoning_level
        # Callable -> bool: did the most recent ``push_thought`` actually
        # render (vs buffer in "on" mode before content)? Controls whether a
        # thinking_delta triggers a live paint. Defaults to "always rendered".
        self.thought_rendered = thought_rendered or (lambda: True)
        self.tool_show_header = tool_show_header
        self.tool_show_body = tool_show_body
        self.visible_parts = visible_parts
        # Separate reasoning buffer + sink for the TWO-MESSAGE render mode
        # (``on_think_chunk`` set): thinking tokens accumulate here and paint
        # the dedicated 💭 message, so an interleaving model can never split
        # the answer. When ``on_think_chunk`` is None the renderer falls back
        # to the single-message behaviour (thoughts inline in visible_parts).
        self.think_parts = think_parts if think_parts is not None else []
        self.on_think_chunk = on_think_chunk
        # When set, the canonical events are routed to this :class:`_BlockStream`
        # (the per-turn 💭/🔧/💬 message sequence) instead of the single-buffer
        # push_* closures. The render-observer path leaves it None.
        self.blockstream = blockstream
        self.on_chunk = on_chunk
        self.on_tool_status = on_tool_status
        self.on_split = on_split
        self.headers_emitted = headers_emitted if headers_emitted is not None else set()
        self.tool_titles = tool_titles if tool_titles is not None else {}
        self.streamed_tools = streamed_tools if streamed_tools is not None else set()
        self.turn_idx = turn_idx
        # Async sink to render the user prompt from ``turn_started.user`` — set
        # ONLY on the render path (a turn started elsewhere). None for a locally
        # submitted turn (the user's own message is already in the chat).
        self.echo_user = echo_user
        # ``True`` after the first ``thinking_delta`` was rendered (or the
        # first ``AgentThoughtChunk`` in the legacy path). Used to detect
        # the reasoning→content transition for a split at reasoning end.
        self.had_thinking = False
        # Optional turn watchdog state (``last``, ``idle_timeout``, …).
        self.progress = progress
        # ``artifact_reference`` events of this turn: the files the agent
        # linked, sent as photos/documents after the reply's final paint.
        self.artifacts: list[dict] = []
        # Terminal turn state, set by the ``turn_ended`` mapping: the per-turn
        # subscribe loop polls ``stop_state["done"]`` to know when to stop.
        self.stop_state: dict[str, Any] = {
            "done": False,
            "aborted": False,
            "error": None,
            "final_content": None,
            "stop_reason": None,
        }


# Tool titles whose raw body IS the user-visible answer (model delegated and
# we stream the delegate's reply directly) — always rendered regardless of
# the /tools gate. Mirrors the legacy ``_DIRECT_DISPATCH_TITLES`` set.
_DIRECT_DISPATCH_TITLES = frozenset(
    {
        "delegate_claude",
        "delegate_codex",
        "delegate_antigravity",
        "compact_context",
        # The sibling-sharedagents-agent delegate (subagent_type=research/explore/general/
        # code/…) is recorded as the canonical ``Agent`` tool. Its body IS the child's
        # answer, exactly like the CLI-engine delegates above — without this, a
        # delegated reply arrives as an ``Agent`` tool_result, fails the /tools gate,
        # and its body is dropped, so the user sees only planner banners. (Delivery
        # bug found by test_live_bot_text_roundtrip_delivery.)
        "AgentSync",
        "AgentAsync",
    }
)
# Quiet attribution badge prepended to the 💬 answer for a delegate tool whose
# body IS the reply (rendered as the answer, never as a 🔧 tool message).
_DELEGATE_BADGES = {
    "delegate_claude": "↳ via claude",
    "delegate_codex": "↳ via codex",
    "delegate_antigravity": "↳ via antigravity",
    "AgentSync": "↳ via agent",
    "AgentAsync": "↳ via agent",
}


def _direct_answer_ok(title: str, ok: bool) -> bool:
    """A direct-dispatch delegate body IS the user answer and bypasses the /tools
    gate — but ONLY when the delegation SUCCEEDED. A failed ``Agent`` (missing
    subagent_type, delegation-gate refusal, runner error) returns ok=False with
    INTERNAL diagnostic/roster text; rendering that as the answer would leak
    internals to the user. On failure, fall back to the normal /tools gate (the
    status line shows 'failed'; the raw body appears only with /tools on)."""
    return title in _DIRECT_DISPATCH_TITLES and ok


# Internal classifier/scaffold tools whose body is diagnostic noise, never
# shown in the reply (the header, when /tools on, still announces the run).
_INTERNAL_TOOL_TITLES = frozenset({"plan_classify", "plan_pass"})
# Read-only / lightweight tools whose 🔧 headers are folded into ONE message
# per burst. Cuts sendMessage count so parallel forum topics don't trip
# Telegram's per-chat send rate limit (≈20/min).
_CLUSTERABLE_TOOL_KEYS = frozenset(
    {
        "grep",
        "glob",
        "read",
        "read file",
        "semanticsearch",
        "memory search",
        "memory get",
        "toolsearch",
        "skillsearch",
        "webfetch",
        "listmcpresources",
        "fetchmcpresource",
    }
)


def _tool_cluster_key(name: str) -> str | None:
    """Return a normalised key when ``name`` is clusterable, else ``None``."""
    key = name.strip().casefold().replace("_", " ")
    if key in _CLUSTERABLE_TOOL_KEYS:
        return key
    # OpenCode MCP read-only tools (``brave-search_brave_web_search``, …).
    if "web search" in key or key.endswith(" websearch"):
        return "webfetch"
    if key.endswith(" webfetch"):
        return "webfetch"
    return None


# Short human labels for the collapsed tool-status fold (#2). A raw tool name
# (``mcp__web__search``, ``Bash``) is mapped to a compact display token so the
# rolling status reads ``🔧 web search ×5 · fetch ×3``. Unknown names fall back
# to a de-namespaced form of the raw name.
_TOOL_STATUS_LABELS = {
    "mcp__web__search": "web search",
    "mcp__web__fetch": "web fetch",
    "mcp__memory__search": "memory search",
    "mcp__memory__get": "memory get",
    "mcp__memory__remember": "remember",
    "mcp__memory__session_search": "memory search",
}


def _tool_status_label(name: str) -> str:
    """Compact display token for one tool name in the collapsed status fold."""
    if not name:
        return "tool"
    lbl = _TOOL_STATUS_LABELS.get(name)
    if lbl:
        return lbl
    # De-namespace ``mcp__server__tool`` → ``tool`` (spaces for readability),
    # otherwise use the raw name verbatim (Bash, Read, Grep, …).
    if name.startswith("mcp__"):
        parts = name.split("__")
        return (parts[-1] or name).replace("_", " ")
    return name


def _render_tool_counts(counts: OrderedDict[str, int]) -> str:
    """Render the collapsed per-name tool counts as ``a ×5 · b ×3 · c``.

    A count of 1 shows the bare label (no ``×1``) so a single tool call reads
    naturally (``🔧 Bash``), while repeats carry the ``×N`` multiplier.
    """
    parts: list[str] = []
    for label, n in counts.items():
        parts.append(f"{label} ×{n}" if n > 1 else label)
    return " · ".join(parts)


def _delegate_status_label(name: str, arguments: Any) -> str:
    """Compact ``🔎 <who> — working…`` progress line for a delegation (#1).

    ``name`` is the direct-dispatch tool (``Agent`` / ``delegate_claude`` …);
    for the sibling ``Agent`` tool the ``subagent_type`` arg names the child
    (research / explore / code / …) so the operator sees WHICH delegate is
    running during the otherwise-silent child turn.
    """
    who = ""
    args = _parse_tool_args(arguments) if arguments is not None else {}
    if isinstance(args, dict):
        who = str(args.get("subagent_type") or args.get("name") or "").strip()
    if not who:
        who = {
            "delegate_claude": "claude",
            "delegate_codex": "codex",
            "delegate_antigravity": "antigravity",
        }.get(name, name or "agent")
    return f"🔎 {who} — working…"


# ---------------------------------------------------------------------------
# ``bot.py`` still owns the shared low-level constants + helpers these renderers
# use. The pacing/size CONSTANTS (EDIT_*, TELEGRAM_MAX_TEXT) are read through the
# ``_bot`` module alias — NOT copied via ``from bot import`` — so tests that
# monkeypatch ``sharedagents_telegram.bot.<NAME>`` (e.g. EDIT_GLOBAL_MIN_GAP_S) still
# take effect here, matching the original single-module semantics. The pure
# helper FUNCTIONS are never reassigned, so a plain re-import is equivalent.
# Both live at the BOTTOM (after the classes) so the bot<->rendering cycle can't
# deadlock in either import order. Do NOT hoist to the top.
# ---------------------------------------------------------------------------
from sharedagents_telegram import bot as _bot  # noqa: E402,F401
from sharedagents_telegram.bot import (  # noqa: E402
    _assemble_error_reply,
    _cap_to_single_message,
    _parse_tool_args,
    _split_at,
)
