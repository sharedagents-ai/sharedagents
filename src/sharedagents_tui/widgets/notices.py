"""Inline transcript notice widgets (EOS / runtime-warning / compaction)."""

from __future__ import annotations

from textual.widgets import Collapsible, Static

from sharedagents_tui.helpers import (
    _RUNTIME_WARNING_HEADLINES,
)


class EmptyEosNotice(Static):
    """Inline transcript alert rendered when the upstream model emitted
    a premature EOS on a tool-followup round.

    Triggered by `auto_continue { reason: "empty_stop" }` from the agent
    runtime. The dispatch loop handles the recovery (it injects a nudge
    and resamples); this widget exists so the user sees, persistently in
    the scrollback, that the upstream model glitched on a tool followup.
    """

    def __init__(self, count: int, max_retries: int, round_idx: int) -> None:
        head = f"⚠️ premature EOS on tool call · auto-continuing (retry {count}/{max_retries}, round {round_idx})"
        super().__init__(head, markup=False, classes="empty-eos-notice")


class RuntimeWarningNotice(Static):
    """Inline transcript banner rendered when the runtime catches a
    recoverable failure mode mid-turn (loop / nudge / distill /
    force-terminate). Spans the full transcript width with a thick
    coloured border so the operator can see *at a glance* that the agent
    intervened — without this widget the only signal was a few lines of
    text inside the assistant message bubble, easy to miss when the
    model is also streaming prose around it.

    The data comes from the bridge's :func:`_format_runtime_warning`
    output (acp_bridge.py L1153). Each warning is emitted as a single
    ``agent_message_chunk`` whose body starts with one of two well-known
    glyph prefixes:

    - ``⚠ runtime: <event> · k1=v1 · k2=…`` — hard intervention
      (loop hit, force-terminate). Renders red.
    - ``↻ runtime: <event> · …`` — soft recovery (distill, nudge).
      Renders blue.

    The TUI sniffs the chunk prefix in the dispatch loop and routes here
    instead of appending to the assistant message bubble (where it used
    to blend into the model's own prose).
    """

    def __init__(self, glyph: str, event: str, fields: dict[str, str]) -> None:
        self.event = event
        self.fields = dict(fields)
        # Soft (↻ distill / ↻ nudge) vs hard (⚠ loop / ⚠ force-terminate).
        # Carries through to the CSS class so the palette differs and
        # `loop detected` doesn't visually read as `compaction OK`.
        self.soft = glyph == "↻"
        headline = _RUNTIME_WARNING_HEADLINES.get(event, event)
        # One-line body. Lead with the headline so the operator can
        # parse it at a glance; trail with the structured fields so the
        # underlying tier/stream/round/evidence are still visible
        # without expanding anything.
        bits = [f"{glyph} {headline}"]
        for key in ("stream", "tier", "round", "tool"):
            val = fields.get(key)
            if val:
                bits.append(f"{key}={val}")
        evidence = fields.get("evidence", "")
        if evidence:
            # Trim aggressively: the evidence is usually the offending
            # n-gram repeated, often >80 chars of mostly-whitespace
            # noise. A short preview is enough to tell two loops apart.
            preview = evidence.strip().replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:57] + "…"
            bits.append(f"evidence={preview!r}")
        cls = "runtime-warning-notice soft" if self.soft else "runtime-warning-notice hard"
        super().__init__(" · ".join(bits), markup=False, classes=cls)


class CompactionNotice(Collapsible):
    """Inline transcript marker rendered when the agent runtime compacted
    the upstream prompt mid-turn.

    Visible by default (so the user notices), and Ctrl+T-collapsible like
    the other expandable widgets. The body shows the full audit:
    estimated tokens removed, the post-compaction prompt size, the
    `tool_call_id`s that were truncated/dropped on the wire, and how
    many old user/assistant turns (if any) were discarded.

    The original transcript content for any affected `ToolResultMessage`
    is unchanged — `_apply_compaction_audit` *also* tags those widgets
    with a "🗜️ upstream-truncated/dropped" badge so the user can expand
    them and see exactly what the upstream model no longer has.
    """

    def __init__(self, reason: str, payload: dict) -> None:
        self.reason = reason
        self.payload = payload
        removed = int(payload.get("removed_tokens") or 0)
        new_size_raw = payload.get("new_prompt_tokens")
        new_size = int(new_size_raw) if isinstance(new_size_raw, (int, float)) and new_size_raw > 0 else None
        n_trunc = len(payload.get("truncated_tool_call_ids") or [])
        n_drop = len(payload.get("dropped_tool_call_ids") or [])
        n_turns = int(payload.get("dropped_turns") or 0)
        # LLM-summarization stats (present when the path went through
        # `_compact_via_llm` — the manual `/compact` endpoint always
        # does, the in-turn proactive trigger does too). When absent
        # the path was the deterministic post-overflow squeeze.
        summary_chars = int(payload.get("summary_chars") or 0)
        llm_pt = payload.get("prompt_tokens")
        llm_ct = payload.get("completion_tokens")
        llm_dur = payload.get("duration_s")
        llm_phase = payload.get("phase")
        llm_finish = payload.get("finish_reason")
        used_llm = summary_chars > 0 or llm_ct is not None

        emoji = "🗜️" if reason == "compacted" else "⚠️"
        label = "compacted" if reason == "compacted" else "compacted (after 400)"
        if used_llm:
            # Lead with proof the LLM ran: summary length + tokens-in/out
            # + wall-clock. This is the user's "yes the model summarised"
            # cue, otherwise `/compact` looks like a no-op.
            head = f"{emoji} {label} · summary {summary_chars} chars"
            tail_bits = []
            if isinstance(llm_pt, (int, float)) and llm_pt:
                tail_bits.append(f"pt {int(llm_pt)}")
            if isinstance(llm_ct, (int, float)) and llm_ct:
                tail_bits.append(f"ct {int(llm_ct)}")
            if isinstance(llm_dur, (int, float)) and llm_dur:
                tail_bits.append(f"{float(llm_dur):.1f}s")
            if tail_bits:
                head += " · " + " · ".join(tail_bits)
            if removed:
                head += f" · ~{removed} tok freed"
        else:
            head = f"{emoji} {label} · ~{removed} tok removed"
        if new_size is not None:
            head += f" · prompt now ~{new_size} tok"
        if n_trunc or n_drop or n_turns:
            bits = []
            if n_trunc:
                bits.append(f"{n_trunc} tool result(s) truncated")
            if n_drop:
                bits.append(f"{n_drop} tool result(s) dropped")
            if n_turns:
                bits.append(f"{n_turns} old turn(s) collapsed")
            head += " · " + ", ".join(bits)
        head += "   ·   Ctrl+T to expand"
        self._head_collapsed = head
        self._head_expanded = head.replace("Ctrl+T to expand", "Ctrl+T to collapse")

        # Body: structured dump. LLM stats are surfaced explicitly so
        # the user can confirm an actual summarization round happened.
        body_lines = [
            f"reason: {reason}",
            f"round: {payload.get('round')}",
            f"removed_tokens (est): {removed}",
            f"new_prompt_tokens (est): {new_size if new_size is not None else 'n/a (next turn)'}",
        ]
        if used_llm:
            body_lines.extend(
                [
                    "",
                    "LLM summarization:",
                    f"  summary_chars:     {summary_chars}",
                    f"  prompt_tokens:     {llm_pt}",
                    f"  completion_tokens: {llm_ct}",
                    f"  duration_s:        {llm_dur}",
                    f"  phase:             {llm_phase}",
                    f"  finish_reason:     {llm_finish}",
                ]
            )
        body_lines.extend(
            [
                "",
                f"truncated tool_call_ids ({n_trunc}):",
            ]
        )
        for tcid in payload.get("truncated_tool_call_ids") or []:
            body_lines.append(f"  · {tcid}")
        body_lines.append("")
        body_lines.append(f"dropped tool_call_ids ({n_drop}):")
        for tcid in payload.get("dropped_tool_call_ids") or []:
            body_lines.append(f"  · {tcid}")
        if n_turns:
            body_lines.append("")
            note = "old turns collapsed into the summary above" if used_llm else "old turns dropped"
            body_lines.append(f"{note}: {n_turns}")
        body_lines.append("")
        body_lines.append(
            "Note: the transcript above still has the original (untrimmed) "
            "tool outputs — only the upstream model's view was compacted."
        )
        body = Static("\n".join(body_lines), markup=False, classes="compaction-notice-body")
        super().__init__(
            body,
            title=self._head_collapsed,
            collapsed=False,  # visible by default — this is the whole point
            classes="compaction-notice",
        )

    def watch_collapsed(self, collapsed: bool) -> None:
        self.title = self._head_collapsed if collapsed else self._head_expanded

    def toggle(self) -> bool:
        self.collapsed = not self.collapsed
        return self.collapsed
