"""InjectionMixin — extracted from the ``local_llm`` monolith (behaviour-preserving)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from sharedagents_core.attachment_context import model_user as _attachment_model_user
from sharedagents_core.attachments import tool_image_message

from .state import _collapse_leading_system_messages

#: Bounded tail of prior turns a RESUME turn replays. An explicit next turn
#: whose WHOLE prompt is a rendered resume brief must not replay the full
#: prior-turn history — that would make the resumed turn nearly as heavy as
#: the capped one. The brief IS the machine-readable carryover; the model
#: additionally sees the most recent N turns so immediate prior tool results
#: stay visible.
_RESUME_BRIEF_TAIL_TURNS = 3

#: Per-turn content budget for the resumed tail. A
#: single capped turn can itself be hundreds of KB (large tool bodies,
#: long assistant wrap-ups); with only 1-2 prior turns the tail bound
#: alone would still replay it in full. On a RESUME turn every replayed
#: body (user content, tool-result body, assistant content) is ellipsized
#: past this cap, keeping the resume context compact regardless of how
#: many turns exist or how large they are.
_RESUME_BRIEF_MAX_TURN_CHARS = 32_000

#: AGGREGATE budget for the whole replayed tail on a
#: RESUME turn. The per-field caps alone are not enough: a turn with 25
#: tool results of ~31K chars each replays ~775K even though every single
#: field is under its cap. The tail is a carryover supplement (the brief
#: IS the machine-readable contract), so the entire replay is bounded to
#: what the tail turns themselves would be allowed to contribute (3 turns
#: × the per-turn cap). Oversized parts are dropped as a whole with an
#: explicit truncation marker — never replayed past the budget.
_RESUME_BRIEF_AGGREGATE_CHARS = _RESUME_BRIEF_TAIL_TURNS * _RESUME_BRIEF_MAX_TURN_CHARS

#: Per-ARGUMENT cap for replayed tool calls on a RESUME
#: turn. Tool-call arguments were the one replay field with no bound at
#: all (a ~200K-char argument string replayed in full). Arguments are
#: payload specs, not results — a tighter cap than the body cap keeps the
#: exchange compact while the model still sees the call shape.
_RESUME_BRIEF_MAX_ARG_CHARS = 8_000


def _ellipsize(text: str, limit: int) -> str:
    """Truncate ``text`` past ``limit`` chars, keeping the head and an
    explicit truncation marker. Below the limit the text is untouched
    (byte-stable replay for the common case).

    Boundary-safe for fenced bodies — when the cut lands INSIDE an
    untrusted-data fence (opener present, closer beyond the cut), the cut
    extends to the fence's closer so the replayed context never ends with
    a dangling opener. Extension is bounded in practice by the resume
    AGGREGATE budget (a fenced body whose close lies far past the cap makes
    the whole exchange exceed the budget and gets dropped as a unit)."""
    if len(text) <= limit:
        return text
    cut = limit
    for m in _FENCE_OPEN_RE.finditer(text[:limit]):
        closer = _FENCE_CLOSE_RE.search(text, m.end())
        if closer and closer.end() > limit:
            cut = max(cut, closer.end())
    return text[:cut] + f"\n…[truncated: {len(text)} chars total]"


def _ellipsize_user_content(
    content: str | list[dict[str, Any]],
    limit: int,
) -> str | list[dict[str, Any]]:
    """Per-field bound for replayed user content (str OR multipart)."""
    if isinstance(content, list):
        out: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = str(part.get("text") or "")
                out.append({**part, "text": _ellipsize(text, limit)})
            else:
                out.append(part)
        return out
    return _ellipsize(str(content or ""), limit)


def _ellipsize_tool_arguments(text: str, limit: int) -> str:
    """Keep an oversized historical call's preview inside valid JSON.

    Arguments are themselves a JSON document. Cutting that document and
    appending a raw newline/marker breaks strict upstream tool-history
    parsing, even though HTTPX correctly escapes the outer request body.
    Wrap the bounded preview as data, retaining fence boundaries and the
    explicit truncation marker. Short arguments keep their original bytes.
    """
    if len(text) <= limit:
        return text
    return json.dumps(
        {"_sharedagents_truncated_arguments": _ellipsize(text, limit)},
        ensure_ascii=False,
    )


def _content_chars(content: Any) -> int:
    """Char count for aggregate budgeting (str or multipart text parts)."""
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                total += len(str(part.get("text") or ""))
            elif isinstance(part, str):
                total += len(part)
        return total
    return len(str(content or ""))


#: Untrusted-data fence markers used for boundary-safe truncation. The square
#: mathematical brackets are the canonical runtime markers; retain the old
#: tortoise-shell form for already-persisted sessions.
_FENCE_OPEN_RE = re.compile(r"(?:⟦UNTRUSTED-DATA\b[^⟧]*⟧|〔UNTRUSTED-DATA\b[^〕]*〕)")
_FENCE_CLOSE_RE = re.compile(r"(?:⟦/UNTRUSTED-DATA\b[^⟧]*⟧|〔/UNTRUSTED-DATA\b[^〕]*〕)")


def _is_resume_brief_prompt(
    user_content: str | list[dict[str, Any]],
) -> bool:
    """Exact-match resume-brief detection: the whole prompt must BE a
    rendered resume brief (fence-first AND fence-last) — the byte shape
    ``render_resume_brief`` produces. A prompt that merely CONTAINS a pasted
    brief block is a normal turn and keeps full history."""
    if isinstance(user_content, list):
        return False
    s = str(user_content or "").strip()
    return s.startswith("[resume-brief]") and s.endswith("[/resume-brief]")


class InjectionMixin:
    def inject_user_message(self, sharedagents_sid: str, text: str) -> bool:
        """Queue a user message for the in-flight (or next) turn.

        Frontends call this when the user submits a fresh prompt while
        the model is already streaming a multi-round plan. The
        ``run_turn`` round-loop drains
        ``self._pending_user_messages[sid]`` at every round boundary,
        appending each entry to the live ``messages`` as a
        ``{"role": "user", "content": text}`` block before the next
        worker spawn — so the model sees the steering message on its
        very next round and can pivot.

        Soft-abort the current worker: when a turn is in flight, we
        ALSO call ``turn_state.stop()`` on the active round so the
        in-progress model generation gets SIGTERM'd. Without this,
        the model finishes the prior plan's final assistant content
        (sometimes 30K+ tokens — observed with qwen3.6-27b
        completing a categorized inventory before addressing the
        steer) and the inject only lands AFTER that wasted work. The
        engine's per-sid ``_cancelled`` flag is intentionally NOT
        set — only the worker subprocess dies; the round-loop then
        sees EOF, recovers gracefully, and the next iteration's
        drain picks up the inject so the model pivots.

        Returns ``True`` when the message was queued. ``False`` only
        if ``text`` is empty/whitespace. The method does NOT require an
        active turn — if no turn is running, the queued message will be
        picked up at the start of the next ``run_turn`` call for this
        sid (round 0 drain).
        """
        text = (text or "").strip()
        if not text:
            return False
        self._pending_user_messages.setdefault(sharedagents_sid, []).append(text)
        # Soft-abort the in-flight worker (if any) so the round-loop
        # pivots NOW rather than after the model finishes its current
        # generation. Don't touch ``_cancelled`` — that would terminate
        # the whole turn instead of just rolling to the next round.
        # The soft-abort flag tells the engine read loop to treat the
        # worker's eventual ``{kind: aborted}`` event as a normal
        # round end rather than a real cancellation.
        #
        # Only set the flag when there's a live worker to actually
        # interrupt. Setting it when ``stop()`` is a no-op (no active
        # turn, worker already exited between rounds) would leave
        # stale state that suppresses a later real cancel.
        turn_id = self._session_to_turn.get(sharedagents_sid)
        turn_state = self._active_turns.get(turn_id) if turn_id else None
        if turn_state is not None:
            self._soft_aborts[sharedagents_sid] = True
            try:
                turn_state.stop()
            except Exception:  # noqa: BLE001
                pass
        return True

    def _drain_pending_user_messages(self, sharedagents_sid: str) -> list[str]:
        """Atomically pull every pending inject for ``sharedagents_sid``.

        Returns the list in arrival order and clears the buffer. The
        round-loop calls this at the top of each round; appending the
        drained entries to ``messages`` is the caller's job (so the
        engine controls placement relative to preamble / compaction).
        """
        pending = self._pending_user_messages.get(sharedagents_sid)
        if not pending:
            return []
        out = list(pending)
        pending.clear()
        return out

    # -----------------------------------------------------------------
    # Phase 3a helpers — simple worker spawn + message build.
    # -----------------------------------------------------------------

    def _build_messages_simple(
        self,
        session: Any,
        user_content: str | list[dict[str, Any]],
        system_prompt: str | None,
        preamble: list[dict[str, Any]],
        live_started_at: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the per-round message list for the upstream worker.

        * Per-turn ``[Turn N · YYYY-MM-DD HH:MM:SS ±HHMM (Weekday)]\\n``
          marker for EVERY turn the model sees. Prior turns reuse
          their own frozen ``started_at`` (set once at record time,
          never recomputed) so the marker bytes stay stable across
          replays — vLLM APC / SGLang RadixAttention keep reusing
          the KV blocks for those prior-turn prefixes. The LIVE
          turn uses ``datetime.now()`` because there's no recorded
          ``started_at`` for it yet; the live position is the LAST
          message anyway, so its timestamp doesn't affect prior-
          turn caching.
        * Bare ``[Turn N]`` fallback when a prior turn's
          ``started_at`` is missing or unparseable.
        * Aborted / in-progress prior turns get one-line stubs
          rather than full replay — replaying tens of thousands of
          tokens of cancelled tool work would tax every subsequent
          turn's TTFT. Canonical lane handoffs already accepted into
          those turns are recovered from the durable model-input audit
          before the stub, because the conversation-store watermark has
          already advanced and will not send that handoff again.
        """
        import datetime as _dt

        msgs: list[dict[str, Any]] = []
        sys_text = system_prompt or session.system_prompt
        # Pin a BYTE-STABLE system head as message[0] (the cache anchor) and
        # defer the dynamic suffix (# WORKSPACE-CONTEXT / # MEMORY.md, split
        # at the SHAREDAGENTS_CACHE_BOUNDARY) to a rolling block emitted just
        # before the LIVE user turn (below). A mid-session MEMORY.md edit
        # (the `remember` tool) or cwd change then only invalidates that
        # small trailing block instead of busting the whole cached
        # [system head + conversation] prefix. Mirrors Hermes/openclaw
        # "stable system + dynamic rolling tail".
        _dyn_suffix = ""
        if sys_text:
            try:
                from sharedagents_agent.system_prompt import (
                    split_system_prompt_cache_boundary,
                )

                _split = split_system_prompt_cache_boundary(sys_text)
            except Exception:  # noqa: BLE001
                _split = None
            if _split is not None:
                _head = _split["stable_prefix"]
                _dyn_suffix = _split["dynamic_suffix"]
            else:
                _head = sys_text
            if _head:
                msgs.append({"role": "system", "content": _head})
        # NOTE: the Agent-composed preamble (pinned facts / rolling
        # compaction summary / todos / skill auto-invoke / memory recall)
        # is NO LONGER spliced here as leading `system` blocks. Those
        # collapse into message[0] (see _collapse_leading_system_messages)
        # and, being per-turn dynamic (hybrid memory recall), would bust
        # the whole cached prefix every time they change. Instead they ride
        # the rolling tail at the head of the LIVE user turn (built below),
        # together with the system prompt's dynamic suffix — keeping
        # message[0] + the conversation history byte-stable across turns.

        prior_turns = list(session.turns or [])

        # A RESUME turn (whole prompt = rendered resume brief) replays only
        # the bounded recent tail, never the full prior history — the brief
        # is the carryover contract and the tail keeps immediate prior tool
        # results visible. The brief's presence triggers the tail regardless
        # of turn count. Similarly, when a compaction summary is present,
        # earlier compacted turns are already summarized, so replay is
        # bounded to the recent tail. Non-resume/non-compacted turns keep
        # append-only full replay.
        has_compaction_summary = bool(
            str(getattr(session, "compaction_summary", "") or "").strip() or getattr(session, "compaction_count", 0) > 0
        )
        resume_compact = _is_resume_brief_prompt(user_content)
        _replay_start = 0
        # ``compaction_turn`` = turns recorded when the last compaction
        # committed. Anchoring the window there keeps the replayed prefix
        # byte-stable (append-only) until the NEXT compaction; a window sliding
        # with ``len(prior_turns)`` drops the oldest turn off the front every
        # turn, so the upstream prefix cache (vLLM APC) matches only the system
        # message and the whole history is re-prefilled each turn. On the first
        # turn after a compaction both formulas give the same start.
        _compaction_anchor = int(getattr(session, "compaction_turn", 0) or 0)
        _anchored = has_compaction_summary and not resume_compact and _compaction_anchor > 0
        if _anchored:
            _replay_start = max(
                0,
                _compaction_anchor - (_RESUME_BRIEF_TAIL_TURNS - 1),
            )
        elif resume_compact or has_compaction_summary:
            _replay_start = max(0, len(prior_turns) - _RESUME_BRIEF_TAIL_TURNS)

        def _format_ts(iso_str: str) -> str:
            """Render an ISO-8601 timestamp like the live marker does
            (``YYYY-MM-DD HH:MM:SS ±HHMM (Weekday)``). Returns empty
            string when the input is missing or unparseable, in
            which case the caller falls back to the bare ``[Turn N]``
            form. Same format on every turn keeps the per-turn
            prefix byte-stable once first written, so vLLM / SGLang
            prefix caching still reuses the prior-turn KV blocks.
            """
            if not iso_str:
                return ""
            try:
                # ``fromisoformat`` accepts the offset suffix in 3.11+.
                cleaned = str(iso_str).replace("Z", "+00:00")
                dt = _dt.datetime.fromisoformat(cleaned)
            except (TypeError, ValueError):
                return ""
            # ``%z`` is empty for naive datetimes (no tz info). Strip
            # the conditional space so a "naive" ISO ``2026-05-19T...``
            # doesn't render as ``2026-05-19 ...  (Tuesday)`` (two
            # spaces). The Session writer always emits tz-aware ISO,
            # so this only bites external/legacy corpora.
            tz_label = dt.strftime("%z")
            stamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            if tz_label:
                stamp = f"{stamp} {tz_label}"
            return f"{stamp} ({dt.strftime('%A')})"

        def _marker(idx: int, started_at: str = "") -> str:
            """Prior-turn header. Includes the turn's frozen
            ``started_at`` timestamp when available — the model sees
            when each prior turn happened, not just the index. The
            timestamp is set ONCE at turn-record time and never
            mutates, so the bytes of this marker stay stable across
            every subsequent turn the engine builds (cache-safe).
            """
            ts = _format_ts(started_at)
            if ts:
                return f"[Turn {idx} · {ts}]\n"
            return f"[Turn {idx}]\n"

        def _live_marker(idx: int) -> str:
            # Render the LIVE turn's marker from the SAME ``started_at``
            # that will be persisted onto this turn's record. Once this
            # turn becomes a prior turn it is replayed via ``_marker`` ->
            # ``_format_ts(started_at)``; using the identical source here
            # makes the live bytes byte-for-byte equal to the future
            # replay bytes, so the upstream prefix cache (DeepSeek /
            # vLLM APC / SGLang RadixAttention) keeps reusing this turn's
            # KV blocks on every subsequent turn instead of missing from
            # this boundary onward. Falls back to ``datetime.now()`` only
            # when no ``started_at`` was threaded in (legacy callers) —
            # that path is the old, cache-unstable behaviour but at least
            # still timestamps the turn.
            ts = _format_ts(live_started_at or "")
            if ts:
                return f"[Turn {idx} · {ts}]\n"
            now = _dt.datetime.now().astimezone()
            nts = now.strftime("%Y-%m-%d %H:%M:%S %z")
            day = now.strftime("%A")
            return f"[Turn {idx} · {nts} ({day})]\n"

        def _render_user(marker: str, content: Any) -> str | list[dict[str, Any]]:
            """Render a user turn (marker prefix + content) so a PRIOR
            turn's replay is byte-identical to how it was sent LIVE.

            * ``str`` content → ``marker + content`` (plain text).
            * Multi-part (``list``) content (images / mixed) → a
              text-marker block followed by the original parts, exactly
              like the live branch builds it. The previous code did
              ``marker + str(content)`` for prior turns, which stringified
              the Python list (e.g. ``"[{'type': 'image_url', ...}]"``):
              that both BUSTS the prefix cache (the live turn sent real
              parts, the replay sends a repr) and corrupts the content
              (an image becomes a literal Python repr the model can't
              use). Routing both paths through here keeps live == frozen.
            """
            if isinstance(content, str):
                return marker + content
            parts: list[dict[str, Any]] = [{"type": "text", "text": marker}]
            parts.extend(list(content))
            return parts

        import json as _json

        def _replay_tool_exchange(
            turn: Mapping[str, Any],
            *,
            compact_args: bool = False,
            is_compacted: bool = False,
        ) -> list[dict[str, Any]]:
            """Reconstruct a PRIOR turn's tool exchange as OpenAI messages
            (``assistant`` with ``tool_calls`` → one ``tool`` message per
            call) so tool / delegation (`/claude`, `/codex`) results
            PERSIST across turns instead of being dropped at the turn
            boundary. This makes the cross-turn context append-only: tool
            output stays until ``_maybe_compact`` is the one that removes
            it — satisfying "only change the context at compaction".

            Orphan repair (keeps the sequence upstream-valid — every
            assistant ``tool_call`` must be answered by a ``tool`` message):
            a call whose result body wasn't retained gets a stub result;
            results with no matching call are dropped. Returns ``[]`` when
            the turn made no tool calls.

            Bodies come from the in-memory turn record's ``tool_results``
            (``content``); after a gateway restart the on-disk record may
            have stripped bodies (privacy gate ``tool_output_visible``), in
            which case the stub keeps the exchange valid without the body.
            For compacted turns, bodies > 800 chars are cleared to
            ``_OC_CLEARED_PLACEHOLDER`` so historical tool data is not
            replayed alongside the compaction summary.
            """
            tcs = turn.get("tool_calls") or []
            if not tcs:
                return []
            res_by_id: dict[str, Any] = {}
            for r in turn.get("tool_results") or []:
                cid = str(r.get("call_id") or r.get("id") or "")
                if cid and "content" in r and cid not in res_by_id:
                    res_by_id[cid] = r
            oai_calls: list[dict[str, Any]] = []
            ordered_ids: list[str] = []
            for c in tcs:
                cid = str(c.get("call_id") or c.get("id") or "")
                if not cid:
                    continue
                args = c.get("arguments")
                if isinstance(args, (dict, list)):
                    # Deterministic serialization so the bytes stay stable
                    # across every rebuild (cache-safe).
                    args = _json.dumps(args, ensure_ascii=False, sort_keys=True)
                else:
                    args = str(args if args is not None else "")
                # Tool-call ARGUMENTS were the one
                # replay field without a bound — cap them on resume turns.
                if compact_args:
                    args = _ellipsize_tool_arguments(args, _RESUME_BRIEF_MAX_ARG_CHARS)
                oai_calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": str(c.get("name") or ""), "arguments": args},
                    }
                )
                ordered_ids.append(cid)
            if not oai_calls:
                return []
            # The turn records ONE reasoning text — the final round's
            # (``turn_ended.final_thinking`` replaces the accumulation). The
            # final reply below replays it; repeating it here would put
            # post-call reasoning before the call and pay for it twice.
            # Only a turn that ended without a reply keeps it here.
            thought = "" if turn.get("content") else turn.get("reasoning_content") or turn.get("thinking") or ""
            out: list[dict[str, Any]] = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": oai_calls,
                    "reasoning_content": thought,
                },
            ]
            image_items: list[tuple[str, str, Any]] = []
            names = {str(c.get("call_id") or c.get("id") or ""): c for c in tcs}
            for cid in ordered_ids:
                r = res_by_id.get(cid)
                body = r.get("content") if r is not None else None
                if body is None:
                    body = "[tool result not retained in context]"
                elif is_compacted and isinstance(body, str) and len(body) > 800:
                    from sharedagents_agent.compaction import _OC_CLEARED_PLACEHOLDER

                    body = _OC_CLEARED_PLACEHOLDER
                out.append({"role": "tool", "tool_call_id": cid, "content": str(body)})
                imgs = r.get("images") if r is not None else None
                if imgs and not is_compacted and not resume_compact:
                    call = names.get(cid) or {}
                    args = call.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = _json.loads(args)
                        except ValueError:
                            args = {}
                    from .tool_dispatch import _tool_image_label

                    name = str(call.get("name") or "tool")
                    image_items.append((name, _tool_image_label(name, args), imgs))
            # Images a tool showed the model come back in the same place
            # they were delivered live: one ``user`` message after the tool
            # results. Compacted and resume-brief turns drop them (the tool
            # body still says an image was read), like old attachments.
            image_msg = tool_image_message(image_items) if image_items else None
            if image_msg is not None:
                out.append(image_msg)
            return out

        # A lane handoff is accepted as soon as the upstream starts the
        # physical turn. If that turn later aborts, the conversation-store
        # watermark has legitimately advanced, but the normal cancelled-turn
        # stub below would drop the handoff from every future model prompt.
        # Recover only the runtime-generated wrapper from the persisted final
        # ``model_input_messages`` snapshot; never replay the rest of the
        # cancelled prompt (which can be very large and may contain tools).
        _handoff_block_re = re.compile(
            r"<sharedagents-canonical-handoff>\n(?P<body>.*?)\n"
            r"</sharedagents-canonical-handoff>",
            re.IGNORECASE | re.DOTALL,
        )
        _turn_marker_re = re.compile(
            r"(?:^|\n)\[Turn \d+(?:[^\]\n]*)\]\n",
        )
        _handoff_body_prefixes = (
            "Authoritative transcript/provenance of which prior messages ",
            "Canonical handoff checkpoint (lossy summary; source remains ",
        )
        _seen_handoff_hashes: set[str] = set()

        def _handoff_blocks_from_text(
            text: str,
            *,
            allow_standalone: bool,
        ) -> list[str]:
            recovered: list[str] = []
            for match in _handoff_block_re.finditer(text):
                body = match.group("body")
                if not body.startswith(_handoff_body_prefixes):
                    continue
                block = match.group(0)
                # Genuine local-LLM handoffs live in rolling preamble before
                # the LIVE ``[Turn N]`` marker. A recovered handoff from an
                # earlier cancelled turn can also appear as its own exact
                # user message. These placement checks avoid treating a
                # wrapper typed inside the cancelled user prompt as runtime
                # control-plane context.
                first_turn_marker = _turn_marker_re.search(text)
                is_runtime_preamble = bool(first_turn_marker and match.end() <= first_turn_marker.start())
                is_standalone = allow_standalone and text.strip() == block
                if not is_runtime_preamble and not is_standalone:
                    continue
                digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
                if digest in _seen_handoff_hashes:
                    continue
                _seen_handoff_hashes.add(digest)
                recovered.append(block)
            return recovered

        def _recover_cancelled_handoffs(
            turn: Mapping[str, Any],
        ) -> list[dict[str, Any]]:
            recovered: list[dict[str, Any]] = []
            for message in turn.get("model_input_messages") or []:
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    blocks = _handoff_blocks_from_text(
                        content,
                        allow_standalone=True,
                    )
                elif isinstance(content, list) and content:
                    # For multi-part live input, rolling preamble and the turn
                    # marker are both in the first synthetic text part. User
                    # attachment/text parts follow and must not be searched.
                    first = content[0]
                    first_text = first.get("text") if isinstance(first, dict) and first.get("type") == "text" else None
                    blocks = (
                        _handoff_blocks_from_text(
                            first_text,
                            allow_standalone=False,
                        )
                        if isinstance(first_text, str)
                        else []
                    )
                else:
                    blocks = []
                recovered.extend({"role": "user", "content": block} for block in blocks)
            return recovered

        # Aggregate remaining budget for the WHOLE resume-tail or compacted replay
        # (brief or compaction summary is the carryover; the tail is a bounded supplement).
        is_bounded = resume_compact or has_compaction_summary
        _agg_left = _RESUME_BRIEF_AGGREGATE_CHARS if is_bounded else None

        # Turns after the compaction watermark are the live, growing tail: they
        # are replayed uncapped (the context-overflow compaction bounds them).
        # An aggregate cap there would drop the NEWEST turns once the older ones
        # exhausted it.
        _uncapped = [False]

        def _charge(n: int) -> bool:
            """Charge ``n`` chars against the aggregate; False if exhausted."""
            nonlocal _agg_left
            if _agg_left is None or _uncapped[0]:
                return True
            if n > _agg_left:
                return False
            _agg_left -= n
            return True

        def _agg_truncate_marker() -> dict[str, str]:
            return {
                "role": "user",
                "content": ("…[truncated: tail aggregate budget exhausted; remaining prior context omitted]"),
            }

        def _append_injected(turn: Mapping[str, Any]) -> bool:
            """Replay durable mid-turn inputs; return False on budget stop."""
            for item in list(turn.get("injected_messages") or []):
                if isinstance(item, dict):
                    injected_text = str(item.get("text") or "")
                else:
                    # Backward-compatible with any early experimental record
                    # that stored bare strings.
                    injected_text = str(item or "")
                if not injected_text:
                    continue
                bounded_injected = (
                    _ellipsize(injected_text, _RESUME_BRIEF_MAX_TURN_CHARS) if is_bounded else injected_text
                )
                if not _charge(len(bounded_injected)):
                    msgs.append(_agg_truncate_marker())
                    return False
                msgs.append({"role": "user", "content": bounded_injected})
            return True

        for turn_idx in range(_replay_start, len(prior_turns)):
            t = prior_turns[turn_idx]
            _uncapped[0] = _anchored and turn_idx >= _compaction_anchor
            # Saved-file references live beside the typed text; the model
            # sees the attachment context exactly as on the live turn.
            u = _attachment_model_user(t.get("user"), t.get("attachments"))
            t_started = str(t.get("started_at") or "")
            if t.get("aborted"):
                msgs.extend(_recover_cancelled_handoffs(t))
                if u:
                    msgs.append({"role": "user", "content": _render_user(_marker(turn_idx, t_started), u)})
                    if not _append_injected(t):
                        break
                    msgs.append({"role": "assistant", "content": f"[Turn {turn_idx} aborted by user]"})
                elif not _append_injected(t):
                    break
                continue
            if t.get("in_progress"):
                msgs.extend(_recover_cancelled_handoffs(t))
                if u:
                    msgs.append({"role": "user", "content": _render_user(_marker(turn_idx, t_started), u)})
                    if not _append_injected(t):
                        break
                    msgs.append(
                        {
                            "role": "assistant",
                            "content": f"[Turn {turn_idx} interrupted before completing — treat as unanswered]",
                        }
                    )
                elif not _append_injected(t):
                    break
                continue
            if u:
                bounded_u = _ellipsize_user_content(u, _RESUME_BRIEF_MAX_TURN_CHARS) if is_bounded else u
                rendered = _render_user(_marker(turn_idx, t_started), bounded_u)
                if not _charge(_content_chars(rendered)):
                    msgs.append(_agg_truncate_marker())
                    break
                msgs.append({"role": "user", "content": rendered})
            # Append-only: replay this turn's tool exchange BEFORE the final
            # assistant text. On resume / compacted turns each body/arg is size-bounded
            # and the WHOLE exchange is charged against the aggregate — if
            # it does not fit, drop it as a unit (keep call↔result pairing)
            # with an explicit truncation marker.
            # Boundary-scoped compaction: a turn is compacted if it occurred
            # at or before the compaction watermark. Post-compaction turns
            # (turn_idx >= compaction_turn) retain their tool output in replay.
            compaction_turn = int(getattr(session, "compaction_turn", 0) or 0)
            turn_is_compacted = has_compaction_summary and (turn_idx < compaction_turn or compaction_turn == 0)
            exchange = _replay_tool_exchange(
                t,
                compact_args=is_bounded,
                is_compacted=turn_is_compacted,
            )
            if is_bounded and exchange:
                for _tx in exchange:
                    if isinstance(_tx.get("content"), str):
                        _tx["content"] = _ellipsize(_tx["content"], _RESUME_BRIEF_MAX_TURN_CHARS)
                exch_chars = sum(
                    _content_chars(_tx.get("content"))
                    + _content_chars(_tx.get("reasoning_content"))
                    + sum(
                        len(str((tc.get("function") or {}).get("arguments") or ""))
                        for tc in (_tx.get("tool_calls") or [])
                    )
                    for _tx in exchange
                )
                if not _charge(exch_chars):
                    msgs.append(_agg_truncate_marker())
                    break
                msgs.extend(exchange)
            else:
                msgs.extend(exchange)
            # A durable follow-up can enter the live model as a user-role
            # message after ``turn_started`` (notably a completed background
            # Agent result). It is intentionally stored outside ``turn.user``
            # by the event fold. Replay it here, after the reconstructed tool
            # exchange and before the turn's final assistant synthesis, so the
            # causal shape remains user -> tools -> injected evidence -> answer.
            # Do not rebuild this from ``render_segments``: that field is a UI
            # projection and would couple semantic replay to presentation.
            if not _append_injected(t):
                break
            c = t.get("content")
            if c:
                bounded_c = _ellipsize(str(c), _RESUME_BRIEF_MAX_TURN_CHARS) if is_bounded else str(c)
                thought = t.get("reasoning_content") or t.get("thinking") or ""
                # Keep recorded reasoning intact, but include it in the
                # bounded replay budget just like content and tool results.
                if not _charge(len(bounded_c) + _content_chars(thought)):
                    msgs.append(_agg_truncate_marker())
                    break
                asst: dict[str, Any] = {"role": "assistant", "content": bounded_c}
                if thought:
                    asst["reasoning_content"] = thought
                msgs.append(asst)

        # LIVE turn — index = number of prior turns. The live
        # marker uses ``datetime.now()`` (no recorded started_at
        # yet); prior-turn markers already pinned to their frozen
        # ``started_at`` upstream so the prefix bytes are stable.
        # Since the live position is the LAST message, recomputing
        # "now" here doesn't shift any earlier bytes — the cache
        # prefix up to (and including) the most recent PRIOR turn
        # is unchanged across builds.
        live_idx = len(prior_turns)
        live_marker = _live_marker(live_idx)
        # Rolling dynamic context rides at the HEAD of the LIVE user turn —
        # NOT as separate leading `system` messages. Two reasons: (1) a
        # mid-conversation `system` message is rejected by strict chat
        # templates (the qwen3.x vLLM server: "System message must be at the
        # beginning"); a `user` message is universally accepted. (2) Riding
        # on the live turn (the last message) means its turn-to-turn changes
        # (a `remember`/MEMORY.md write, query-dependent memory recall) do
        # NOT invalidate the cached [system head + history] prefix above —
        # mirrors Hermes/openclaw "stable system + dynamic user tail". None
        # of it is recorded onto the turn (built fresh here), so prior turns
        # never replay a stale copy. Order: preamble blocks, then the system
        # prompt's dynamic suffix, then the turn marker + user content.
        _rolling: list[str] = []
        for _block in preamble:
            _bc = str(_block.get("content") or "").strip()
            if _bc:
                _rolling.append(_bc)
        if _dyn_suffix:
            _rolling.append(_dyn_suffix)

        curr_agent = str(
            getattr(getattr(self, "cfg", None), "name", "")
            or getattr(session, "agent_name", "")
            or getattr(session, "agent", "")
            or ""
        ).strip()
        last_turn_agent = ""
        for pt in reversed(prior_turns):
            tagent = str(pt.get("agent_name") or pt.get("engine") or "").strip()
            if tagent:
                last_turn_agent = tagent
                break
        if curr_agent and last_turn_agent and curr_agent != last_turn_agent:
            _rolling.append(
                f"[Active Agent Context: Agent switched from {last_turn_agent} to {curr_agent}. "
                f"You are now acting as {curr_agent}. Adopt {curr_agent}'s persona, instructions, and role. "
                f"Do not introduce yourself as or impersonate {last_turn_agent}.]"
            )

        if _rolling:
            live_marker = "\n\n".join(_rolling) + "\n\n" + live_marker
        # Route through the SAME renderer the prior-turn replay uses so the
        # live bytes are byte-identical to this turn's future frozen replay
        # (str AND multi-part). Note the live marker differs from the frozen
        # marker only when no started_at was threaded in (legacy fallback);
        # with live_started_at set they are identical too.
        msgs.append({"role": "user", "content": _render_user(live_marker, user_content)})
        return _collapse_leading_system_messages(msgs)

    # -----------------------------------------------------------------
    # Message-list construction
    # -----------------------------------------------------------------

    def _apply_round_preamble(
        self,
        messages: list[dict[str, Any]],
        preamble: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Idempotently splice the Agent-composed preamble blocks.

        Reference: acp_bridge.py L1912–1947 (today's
        ``_apply_persistent_injections``). The semantic is preserved
        exactly: each block REPLACES any prior occurrence (matched by
        a stable marker comment at the top of the block), so calling
        every round keeps the prefix byte-stable. Order in the message
        list:

            system → pinned → rolling_summary → todos → skill_preamble
            → memory_recall → live tail

        The composition step (deciding WHAT goes into each block)
        lives on :class:`Agent` (see :meth:`Agent.compose_round_preamble`).
        The engine just splices what it's given — single composition
        owner.
        """
        if not preamble:
            return messages

        # Markers used by the preamble injection helpers — each block
        # carries a unique sentinel near the top so re-splicing is
        # idempotent (replaces in-place, doesn't duplicate). Marker
        # strings MUST match the actual prefix each producer emits;
        # mismatched markers let the previous block survive into
        # round_idx > 0 and the preamble duplicates:
        # * ``turn_runtime._inject_pinned_facts`` writes
        #   ``[PINNED FACTS — survive every compaction; …]``
        # * ``turn_runtime._inject_compaction_summary`` writes
        #   ``[ROLLING COMPACTION SUMMARY — …]``
        # * ``turn_runtime._inject_current_todos`` writes
        #   ``[CURRENT TODO LIST]`` (or a richer status header)
        # * ``skill_index.render_preamble`` writes
        #   ``[Auto-invoked skill: …]``
        # * ``memory_manager.MemoryManager.prefetch_and_format`` now wraps
        #   recalled memory in an injection-guard fence whose open marker
        #   reads ``⟦UNTRUSTED-DATA id=<nonce> source=memory:<provider>⟧``.
        #   We key on ``source=memory:`` (appears within the first 80 chars)
        #   so the fenced recall block is still recognized and replaced each
        #   round — without it the block is re-appended every round (context
        #   bloat). ``[RECALLED CONTEXT`` is kept for any legacy/cached block.
        MARKERS = (
            "[PINNED FACTS",
            "[ROLLING COMPACTION SUMMARY",
            "[CURRENT TODO LIST",
            "[Auto-invoked skill",
            "[RECALLED CONTEXT",
            "source=memory:",
        )

        def _marker_for(block: dict[str, Any]) -> str | None:
            content = str(block.get("content") or "")
            for m in MARKERS:
                if m in content[:80]:
                    return m
            return None

        # Find the system message (index 0 by convention). The
        # preamble sits *after* system, *before* the live transcript.
        out: list[dict[str, Any]] = []
        sys_idx = -1
        for i, m in enumerate(messages):
            if isinstance(m, dict) and m.get("role") == "system":
                sys_idx = i
                break

        # Strip any existing preamble blocks (matched by marker)
        # so the new ones land at their canonical position.
        existing = list(messages)
        head = existing[: sys_idx + 1] if sys_idx >= 0 else []
        tail = existing[sys_idx + 1 :] if sys_idx >= 0 else existing
        cleaned_tail = [m for m in tail if not (isinstance(m, dict) and _marker_for(m))]
        out.extend(head)
        for block in preamble:
            out.append(dict(block))
        out.extend(cleaned_tail)
        return _collapse_leading_system_messages(out)
