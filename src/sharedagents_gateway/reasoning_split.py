"""Re-split a vLLM stream into reasoning and answer using the token ids.

vLLM's reasoning parser ends the reasoning at the FIRST reasoning-close token
(``</think>``, one token in Qwen's vocabulary) and never re-opens it. A model
that quotes the tag mid-thought -- a regex it read, a template it is editing --
emits that token, so the rest of the thought streams as the answer, or the
round stops with no answer at all. A later, real close is then dropped from the
text without a trace.

With ``return_token_ids`` every chunk carries the ids it decoded, so the worker
can tell the two apart: the model closes its reasoning on a line of its own
(the template's ``\\n</think>\\n\\n``), while a quoted tag follows ordinary text
on the same line. A quoted close is put back into the reasoning as text and the
stream stays in reasoning until a real close.
"""

from __future__ import annotations

CLOSE_TEXT = "</think>"


class ReasoningSplit:
    """Per-attempt state: ``reasoning`` -> (``literal`` ->) ``answer``.

    ``literal`` means a quoted close was seen and no real close yet: vLLM is
    streaming the continued thought as answer text, which goes back to the
    reasoning channel here.
    """

    def __init__(self, close_id: int) -> None:
        self.close_id = close_id
        self.phase = "reasoning"
        self.quoted_closes = 0
        self._tail = ""  # last characters routed to reasoning
        self.seen_reasoning = False
        # Only the ambiguous tail is held. Ordinary reasoning streams as it
        # arrives; this tail may still turn out to be the upstream's answer.
        self._pending: list[str] = []
        self._upstream_content: list[str] = []

    @property
    def unclosed(self) -> bool:
        """Reasoning ran and no real close came (quoted close or a stop)."""
        return self.seen_reasoning and self.phase != "answer"

    def _note(self, thinking: str) -> str:
        if thinking:
            self.seen_reasoning = True
            self._tail = (self._tail + thinking)[-4:]
        return thinking

    def _closes_line(self, before: str) -> bool:
        """A real close follows a newline (or opens an empty reasoning block)."""
        text = self._tail + before
        return text.endswith("\n") or (not self.seen_reasoning and not text)

    def feed(
        self,
        reasoning: str,
        content: str,
        token_ids: list[int] | None,
    ) -> tuple[str, str]:
        """Consume one chunk; return (reasoning_text, answer_text) to emit."""
        reasoning = reasoning or ""
        content = content or ""
        has_close = bool(token_ids) and self.close_id in token_ids
        if self.phase == "answer":
            if has_close:
                # The reasoning is over, so any further close is quoted text
                # that vLLM dropped.
                self.quoted_closes += 1
                content = self._restore(content, token_ids)
            return self._note(reasoning), content
        if self.phase == "reasoning":
            if not has_close:
                if content:
                    # Answer text with no close in this chunk: the close came
                    # earlier or thinking is off. Trust the upstream split.
                    self.phase = "answer"
                return self._note(reasoning), content
            # vLLM split this chunk at the close: ``reasoning`` is the text
            # before it, ``content`` the text after.
            if self._closes_line(reasoning):
                self.phase = "answer"
                return self._note(reasoning), content
            self.phase = "literal"
            self.quoted_closes += 1
            # ``content`` may itself hold a real close (vLLM drops it): run
            # the literal-phase split over it.
            head = self._note(reasoning)
            self._pending.append(self._note(CLOSE_TEXT))
            self._upstream_content.append(content)
            after = token_ids[token_ids.index(self.close_id) + 1 :]
            more_thinking, answer = self._literal(content, after)
            return head + more_thinking, answer
        self._upstream_content.append(content)
        return self._literal(reasoning + content, token_ids or [])

    def _literal(self, text: str, ids: list[int]) -> tuple[str, str]:
        """Route ``text`` (decoded from ``ids``) while a quoted close is open."""
        if self.close_id not in ids:
            self._pending.append(self._note(text))
            return "", ""
        # vLLM removed the close from ``text``. Place it where the text before
        # ends a line and the text after is empty or starts a new line -- the
        # shape of a real close. The close's position among the ids bounds
        # where that can be.
        k = ids.index(self.close_id)
        if k == 0:
            spots = [0]
        elif k == len(ids) - 1:
            spots = [len(text)]
        else:
            spots = range(1, len(text) + 1)
        for j in spots:
            if self._closes_line(text[:j]) and (j == len(text) or text[j] == "\n"):
                self.phase = "answer"
                thinking = "".join(self._pending) + self._note(text[:j])
                self._pending.clear()
                self._upstream_content.clear()
                return thinking, text[j:]
        self.quoted_closes += 1
        self._pending.append(self._note(self._restore(text, ids)))
        return "", ""

    def finish(self) -> tuple[str, str]:
        """On an unresolved final attempt, preserve vLLM's original answer.

        A forced/budget close can legitimately follow text on the same line.
        Never promote genuine reasoning: only bytes actually received in the
        upstream content channel are eligible for this bounded fallback.
        """
        content = "".join(self._upstream_content)
        self._pending.clear()
        self._upstream_content.clear()
        return "", content

    def _restore(self, text: str, ids: list[int]) -> str:
        """Put a dropped quoted close back where its place is certain."""
        if ids.count(self.close_id) != 1:
            return text
        k = ids.index(self.close_id)
        if k == 0:
            return CLOSE_TEXT + text
        if k == len(ids) - 1:
            return text + CLOSE_TEXT
        return text
