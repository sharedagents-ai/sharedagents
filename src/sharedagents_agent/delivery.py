"""Shared delivery primitive — spec §6.2 / §7 / §11.

The architecture's rule (§14.9): *backend ships semantics, interfaces map
them.*  Today TUI, web, and Telegram each reinvent visibility-gating and
throttling.  This module is the single shared layer every interface reuses:

* ``should_show(event, prefs)`` — the one visibility predicate (§7 "Shared
  visibility filter").  Per-interface prefs (reasoning / tools on|off|stream)
  gate the live thinking stream and tool channels; canonical *structural*
  events are never gated.
* ``redact(event, prefs)`` — strips reasoning / tool *payload* to match prefs
  WITHOUT dropping the structural event (so the log stays a faithful skeleton
  even when a medium hides the bodies).  §11: canonical events are never
  dropped — only their gated fields are blanked.
* ``split_for_transport(text, limit)`` — Telegram-4096-style rollover.
* ``DeliveryConsumer`` — the single coalescing primitive (§6.2, from Hermes's
  ``GatewayStreamConsumer``): applies ``should_show``, redacts, and coalesces
  consecutive ``text_delta`` / ``thinking_delta`` chunks into one render block.

Pure logic: no I/O, no asyncio, no global state, no hidden clock (the spec
forbids hidden clocks in tests — the caller injects ``now_ms``).
"""

from __future__ import annotations

# Per-interface visibility preferences.  Each value is one of "off" / "on" /
# "stream".  "stream" opts into the live partial channels (live thinking, tool
# previews); "on" shows the finished structural blocks but not the live
# partials; "off" hides the channel entirely.
DEFAULT_PREFS: dict[str, str] = {
    "reasoning": "on",
    "tools": "on",
    "activity": "stream",
}

# Event types gated by the "tools" pref.
_TOOL_TYPES = frozenset({"tool_call", "tool_result_meta", "tool_result"})
# Body-only channel: shown only on "stream". ``tool_result_meta`` is NOT
# stream-only — it carries the finished ok/failed status the 🔧 header needs
# on tools=on (header without body). Hiding meta on "on" left widgets stuck
# at "Running" forever after the call completed.
_TOOL_STREAM_ONLY = frozenset({"tool_result"})


def _pref(prefs: dict[str, str] | None, key: str) -> str:
    """Resolve a single pref, falling back to ``DEFAULT_PREFS`` per-key."""
    if prefs is None:
        return DEFAULT_PREFS[key]
    value = prefs.get(key)
    if value is None:
        return DEFAULT_PREFS[key]
    return value


def is_subagent_event(event: dict) -> bool:
    """True when ``event`` is a delegated CHILD's inner event forwarded up
    for live display (not the parent's own canonical event).

    Tagged at the forwarding boundary (``delegate_shims`` ``on_event`` +
    ``local_llm`` ``from_subagent``) so every interface can gate a delegated
    run's internal tool-call / thinking / intermediate-text spray behind a
    single toggle — the child's authoritative ANSWER still reaches the parent
    as the (untagged) ``Agent`` / ``delegate_*`` tool_result, so suppressing
    these never drops the actual reply.

    Recognised markers: ``subagent`` truthy, or a positive depth carried as
    ``depth`` / ``agent_depth`` / ``child_depth``.
    """
    if event.get("subagent"):
        return True
    for key in ("depth", "agent_depth", "child_depth"):
        try:
            if int(event.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def should_show(event: dict, prefs: dict | None = None) -> bool:
    """The shared visibility predicate (§7).

    Truth table (columns are the pref value):

        event type          off    on     stream
        -----------------   ----   ----   ------
        thinking_delta      hide   hide   show     (reasoning pref)
        tool_call           hide   show   show     (tools pref)
        tool_result_meta    hide   show   show     (tools pref; header status)
        tool_result         hide   hide   show     (tools pref; stream-only body)
        subagent-tagged *   hide   hide   show     (tools pref; stream-only)
        everything else     ───── always show ─────

    A ``subagent``-tagged event (a delegated child's forwarded inner
    tool-call / thinking / intermediate text) is a STREAM-ONLY channel
    regardless of its own type: hidden in the default and ``on`` views so the
    parent's Telegram reply is not flooded with the child's internal run, and
    surfaced only when the operator opts into full fidelity via
    ``tools == "stream"``. The child's real answer is unaffected — it arrives
    as the parent's own (untagged) delegate tool_result.

    Unknown / missing ``type`` → ``True`` (fail-open for structural safety).
    """
    etype = event.get("type")
    if etype is None:
        return True

    # Delegated-child internals: one stream-only gate, before the per-type
    # channel rules, so a subagent's thinking / tool / intermediate-text
    # events are all suppressed together off the default reply.
    if is_subagent_event(event):
        if _pref(prefs, "tools") != "stream":
            return False
        # The Web activity channel composes with the pre-existing subagent
        # gate.  Returning early here used to let tagged tool/text events
        # bypass activity=off whenever tools=stream.
        if etype in _TOOL_TYPES or etype == "text_delta":
            return _pref(prefs, "activity") != "off"
        return True

    if etype == "thinking_delta":
        # The live reasoning stream is shown only when explicitly streaming.
        return _pref(prefs, "reasoning") == "stream"

    if etype in _TOOL_TYPES:
        tools = _pref(prefs, "tools")
        if tools == "off":
            return False
        if _pref(prefs, "activity") == "off":
            return False
        if etype in _TOOL_STREAM_ONLY:
            # The partial/preview channel is a streaming-only extra.
            return tools == "stream"
        return True  # "on" or "stream"

    if etype == "text_delta" and _pref(prefs, "activity") == "off":  # noqa: SIM103 - documented guard
        # Web Activity gear off hides live assistant prose. Callers that omit
        # activity keep the default ("stream") so text_delta stays structural.
        return False

    # Canonical structural events (text_delta, turn_*, round_*, header_update,
    # lifecycle, session_*, and any unknown type) are never gated.
    return True


def redact(event: dict, prefs: dict | None = None) -> dict:
    """Return a copy with reasoning / tool payload stripped to match prefs.

    Strips *fields*, never the structural event (§11):

    * ``reasoning == "off"``  → blank ``final_thinking`` on ``turn_ended``.
    * ``tools == "off"``      → blank a large ``tool_result`` body/content
      while keeping ``call_id`` + ``ok``.
    * Always: drop a ``tool_result``'s ``images`` (inline image data a tool
      showed the model). They exist for the model's history replay; no
      interface renders them, so they are never shipped to one.

    * Always: show a prompt the way the person typed it. A turn's saved-file
      references (or a legacy saved-attachment block in its text) become
      ``attached_files`` chips; the model-facing block and the saved paths
      are not shipped (``sharedagents_core.attachment_context.project_event``).

    Otherwise the event is returned unchanged (possibly the same object).  The
    input is never mutated.
    """
    etype = event.get("type")
    if etype in (
        "turn_started",
        "turn_ended",
        "queue_msg_injected",
        "queue_msg_added",
        "queue_msg_updated",
    ):
        from sharedagents_core.attachment_context import project_event

        event = project_event(event)

    if etype == "turn_ended" and _pref(prefs, "reasoning") == "off":
        if event.get("final_thinking"):
            out = dict(event)
            out["final_thinking"] = ""
            return out
        return event

    if etype == "tool_result" and "images" in event:
        event = {k: v for k, v in event.items() if k != "images"}

    if etype == "tool_result" and _pref(prefs, "tools") == "off":
        if event.get("body") or event.get("content"):
            out = dict(event)
            if "body" in out:
                out["body"] = ""
            if "content" in out:
                out["content"] = ""
            return out
        return event

    return event


def split_for_transport(text: str, limit: int) -> list[str]:
    """Split ``text`` into chunks each ``<= limit`` (Telegram-4096-style).

    Prefers to break at the last newline before the limit, else the last
    space, else a hard cut at ``limit``.  Concatenation of the returned parts
    equals ``text`` exactly (all characters preserved, in order).  Never
    returns an empty list for non-empty input.
    """
    if limit <= 0:
        # Degenerate limit: a single hard chunk is the only safe behaviour.
        return [text] if text else []
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = window.rfind("\n")
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit  # no break point — hard cut
        else:
            cut += 1  # keep the break character with the chunk it ends
        parts.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        parts.append(rest)
    return parts


class DeliveryConsumer:
    """The single coalescing / throttle primitive (§6.2).

    Applies ``should_show`` on ``feed``; shown events are redacted then
    accumulated.  Consecutive ``text_delta`` / ``thinking_delta`` chunks
    coalesce into one running text buffer; every other event is queued as a
    discrete item, in arrival order.

    Coalescing is deterministic and channel-boundaried: a run of same-type
    delta chunks is emitted as ONE synthetic block
    ``{"type": <delta type>, "text": <concatenated chunks>}``.  A discrete
    event, or a delta of a *different* type, closes the current run (flushing
    it into the output order) and starts a new channel.  ``drain`` returns the
    accumulated items and clears them.
    """

    def __init__(
        self,
        prefs: dict | None = None,
        *,
        flush_ms: int = 100,
        transport_limit: int | None = None,
    ) -> None:
        self.prefs = prefs
        self.flush_ms = flush_ms
        self.transport_limit = transport_limit
        # Output items already finalized (discrete events + closed text runs),
        # in arrival order.
        self._items: list[dict] = []
        # The open coalescing channel: (delta_type, accumulated_text) or None.
        self._buf_type: str | None = None
        self._buf_text: str = ""
        # Clock of the first buffered delta since the last drain (injected).
        self._first_delta_ms: int | None = None

    # -- internal -----------------------------------------------------------
    def _flush_buffer(self) -> None:
        """Close the open text run into the finalized item list."""
        if self._buf_type is not None:
            self._items.append({"type": self._buf_type, "text": self._buf_text})
            self._buf_type = None
            self._buf_text = ""

    # -- public -------------------------------------------------------------
    def feed(self, event: dict, now_ms: int | None = None) -> None:
        """Gate, redact, and accumulate one canonical event."""
        if not should_show(event, self.prefs):
            return
        event = redact(event, self.prefs)
        etype = event.get("type")

        if etype in ("text_delta", "thinking_delta"):
            # The canonical writer field is ``delta``; ``chunk`` is tolerated
            # for any adapter that still uses the older name.
            chunk = event.get("delta")
            if chunk is None:
                chunk = event.get("chunk", "")
            if self._buf_type is not None and self._buf_type != etype:
                # Channel boundary: a different delta kind closes the run.
                self._flush_buffer()
            if self._buf_type is None:
                self._buf_type = etype
                self._buf_text = ""
            self._buf_text += chunk
            if self._first_delta_ms is None and now_ms is not None:
                self._first_delta_ms = now_ms
            return

        # Discrete event closes any open text run, preserving arrival order.
        self._flush_buffer()
        self._items.append(event)

    def should_flush(self, now_ms: int) -> bool:
        """True once ``flush_ms`` has elapsed since the first buffered delta.

        Uses the injected ``now_ms`` clock only — no hidden time source.  With
        no buffered delta there is nothing pending, so this is ``False``.
        """
        if self._first_delta_ms is None:
            return False
        return (now_ms - self._first_delta_ms) >= self.flush_ms

    def drain(self) -> list[dict]:
        """Return and clear the events ready to render since the last drain.

        Coalesced text is emitted as a single synthetic block per channel
        boundary; discrete events pass through unchanged.  When
        ``transport_limit`` is set, each coalesced text block is split via
        ``split_for_transport`` into multiple same-type blocks.
        """
        self._flush_buffer()
        items = self._items
        self._items = []
        self._first_delta_ms = None

        if self.transport_limit is None:
            return items

        out: list[dict] = []
        for item in items:
            if "text" in item and item.get("text"):
                for part in split_for_transport(item["text"], self.transport_limit):
                    block = dict(item)
                    block["text"] = part
                    out.append(block)
            else:
                out.append(item)
        return out


# ---- stream / final-content reconciliation -------------------------------
#
# Live ``text_delta`` streams can end short of the engine's ``content_buf``
# (witnessed: Telegram showed a truncated answer + stats while
# ``turn_ended.final_content`` carried the full sentence). Frontends that
# only paint deltas must reconcile against ``final_content`` at turn end;
# the engine also emits a terminal tail ``TextDelta`` when it detects a gap.


def detect_stream_truncation(emitted: str, full: str) -> str | None:
    """Return missing suffix when ``full`` strictly extends ``emitted``, else None."""
    if not full:
        return None
    em = emitted or ""
    fu = full
    if fu.startswith(em):
        suffix = fu[len(em) :]
        return suffix if suffix else None
    em_r = em.rstrip()
    if em_r and fu.startswith(em_r) and len(fu) > len(em_r):
        suffix = fu[len(em_r) :]
        return suffix if suffix else None
    return None


_RUNTIME_NOTICE_PREFIXES = ("⚠", "↻")


def is_runtime_notice_delta(text: str) -> bool:
    """True when ``text`` is a runtime-warning TextDelta body the engine emits
    for loop/nudge/distill/upstream-retry/force-terminate recoveries.

    The body shape is ``\\n\\n{⚠|↻} runtime: <event> · …\\n\\n`` (see
    ``session_reader._seg_apply``). The live TUI sniffs the glyph prefix and
    mounts a RuntimeWarningNotice widget INSTEAD of appending the text to the
    content bubble; Telegram must mirror that so the user-facing reply never
    inlines ``↻ runtime: nudge …`` prose (which also pollutes ``final_content``
    and can break the terminal rich-markdown edit). Match by stripped first
    char + ``runtime:`` within the first 40 chars, exactly like the TUI.
    """
    if not text:
        return False
    ws = text.lstrip()
    return bool(ws) and ws[:1] in _RUNTIME_NOTICE_PREFIXES and "runtime:" in ws[:40]


def strip_runtime_notices(text: str) -> str:
    """Remove runtime-warning paragraphs from ``text``.

    Splits on blank-line boundaries and drops paragraphs whose stripped body
    is a runtime notice (``⚠/↻ runtime: …``). Keeps everything else verbatim,
    including leading/trailing blank lines produced by the removal (callers
    ``.strip()`` where needed). Used by the Telegram renderer so the canonical
    ``final_content`` — which the engine builds from the raw content stream and
    therefore ALSO carries the notice marker — does not re-introduce the notice
    into the answer bubble at ``turn_ended`` reconciliation time.
    """
    if not text:
        return text
    parts = text.split("\n\n")
    kept = [p for p in parts if not is_runtime_notice_delta(p)]
    if len(kept) == len(parts):
        return text
    return "\n\n".join(kept)


def reconcile_streamed_with_final(streamed: str, final: str) -> tuple[str, bool]:
    """Extend ``streamed`` with the missing tail of ``final`` when aligned."""
    if not final:
        return streamed, False
    s = streamed or ""
    f = final
    if not s.strip():
        return f, bool(f.strip())
    suffix = detect_stream_truncation(s, f)
    if suffix:
        return s + suffix, True
    fs = f.strip()
    ss = s.strip()
    if fs and ss and fs.startswith(ss) and len(fs) > len(ss):
        return f, True
    return streamed, False


def reconcile_visible_with_final(visible: str, final: str) -> tuple[str, bool]:
    """Reconcile a mixed tool+answer visible buffer against assistant ``final``."""
    if not final or not visible:
        return visible, False
    marker = "\n\n💬 "
    idx = visible.rfind(marker)
    if idx >= 0:
        prefix = visible[: idx + len(marker)]
        answer = visible[idx + len(marker) :]
        reconciled, patched = reconcile_streamed_with_final(answer, final)
        if patched:
            return prefix + reconciled, True
        return visible, False
    if visible.startswith("💬 "):
        reconciled, patched = reconcile_streamed_with_final(visible[3:], final)
        if patched:
            return "💬 " + reconciled.lstrip(), True
    reconciled, patched = reconcile_streamed_with_final(visible, final)
    return (reconciled, True) if patched else (visible, False)
