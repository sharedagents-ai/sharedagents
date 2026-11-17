"""Split inline ``<think>`` reasoning out of streamed model content."""

from __future__ import annotations

DEFAULT_OPEN = "<think>"
DEFAULT_CLOSE = "</think>"


class ThinkSplitter:
    """Split a stream of `delta.content` text into thinking vs final-content text.

    Models without a dedicated `reasoning_content` field (vLLM Qwen3 family with
    a buggy reasoning parser, raw chat templates, etc.) often emit
    `<think>...</think>` inline in `delta.content`. Only an initial configured
    reasoning block is structural; once answer text begins, tag-shaped text is
    preserved literally. This class buffers across chunks so partial initial
    tags spanning chunk boundaries split correctly.

    Tags are configurable so we can also parse Gemma's `<thought>...</thought>`
    pattern (vLLM `--reasoning-parser gemma4`) when it leaks back into
    `delta.content`. Defaults stay `<think>` / `</think>` for backward compat.
    """

    def __init__(
        self,
        open_tag: str = DEFAULT_OPEN,
        close_tag: str = DEFAULT_CLOSE,
        *,
        in_think: bool = False,
    ) -> None:
        self._open = open_tag
        self._close = close_tag
        self._hold = max(len(open_tag), len(close_tag)) - 1
        self._buf = ""
        # `in_think=True` lets callers prime the splitter when the prompt
        # was prefilled with the open tag (Gemma 4 thinking pattern) so
        # the model's first output token is already inside a thought block.
        self._state = "reasoning" if in_think else "initial"

    def feed(self, text: str) -> tuple[str, str]:
        """Consume a chunk; return (thinking_text, content_text) safe to emit now."""
        # initial/primed inline reasoning across chunk boundaries, then latch
        # permanently into answer mode so later delimiter text stays literal.
        self._buf += text
        thinking: list[str] = []
        content: list[str] = []
        while True:
            if self._state == "answer":
                content.append(self._buf)
                self._buf = ""
                break

            if self._state == "reasoning":
                end = self._buf.find(self._close)
                if end == -1:
                    safe = max(0, len(self._buf) - self._hold)
                    if safe:
                        thinking.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                thinking.append(self._buf[:end])
                self._buf = self._buf[end + len(self._close) :]
                self._state = "answer"
                continue

            # Initial formatting whitespace is not genuine answer text. Hold it
            # until the following bytes either complete the configured opening
            # delimiter or prove that this is an ordinary answer.
            prefix_len = 0
            while prefix_len < len(self._buf) and self._buf[prefix_len].isspace():
                prefix_len += 1
            candidate = self._buf[prefix_len:]
            if not candidate or self._open.startswith(candidate):
                break
            if candidate.startswith(self._open):
                content.append(self._buf[:prefix_len])
                self._buf = candidate[len(self._open) :]
                self._state = "reasoning"
                continue

            content.append(self._buf)
            self._buf = ""
            self._state = "answer"
            break
        return "".join(thinking), "".join(content)

    def flush(self) -> tuple[str, str]:
        """Drain the buffer when the stream ends."""
        # initial reasoning remains thinking; undecided/answer bytes stay content.
        if not self._buf:
            return "", ""
        if self._state == "reasoning":
            out = self._buf
            self._buf = ""
            return out, ""
        out = self._buf
        self._buf = ""
        return "", out
