"""Streaming subprocess worker.

Reads one JSON request from stdin, opens a streaming chat-completions request
against an OpenAI-compatible endpoint (vLLM), and emits JSONL events on stdout
that the parent app consumes:

    {"kind": "thinking", "text": "..."}
    {"kind": "content",  "text": "..."}
    {"kind": "done"}
    {"kind": "aborted"}
    {"kind": "error",    "message": "..."}

The parent kills this process via SIGTERM (Popen.terminate()) on /abort or ESC.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
import time
from collections.abc import Mapping
from typing import Any

import httpx

from sharedagents_agent.turn_runtime import (
    LOOP_CHECK_STRIDE,
    LOOP_DETECT_ENABLED,
    _detect_loops_layered,
    _should_fire_with_sustain,
    tool_args_loop_should_abort,
)
from sharedagents_gateway.reasoning_split import ReasoningSplit
from sharedagents_model.bedrock_config import BEDROCK_API
from sharedagents_model.splitter import ThinkSplitter
from sharedagents_model.wire_replay import (
    ensure_required_reasoning_fields as _ensure_required_reasoning_fields,
    is_missing_reasoning_content_error as _is_missing_reasoning_content_error,
    record_rejected_field as _record_rejected_field,
    strip_unsupported_fields as _strip_unsupported_fields,
    unsupported_message_field as _unsupported_message_field,
)

# Module logger for anomaly surfacing (e.g. empty/degenerate turns).
# Warnings go to stderr and don't disturb the JSONL
# event stream the parent parses on stdout.
_log = logging.getLogger("sharedagents.worker")

#: A response that starts more tool calls than this is a runaway batch: live
#: 2026-09-30 a deepseek-flash response scripted 144 calls (bash, fetches,
#: todowrite x11, a plan write, question, ExitPlanMode x3) without waiting for
#: one result, hit the 32k output cap, and ran nothing. The week before, no
#: native deepseek-flash or qwen3.8 response used more than 11. The worker
#: stops reading at call N+1 and runs the first N as an ordinary tool round.
#: ``SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE`` overrides; 0 disables.
DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE = 32


def _max_tool_calls_per_response() -> int:
    raw = str(os.environ.get("SHAREDAGENTS_MAX_TOOL_CALLS_PER_RESPONSE") or "").strip()
    try:
        value = int(raw) if raw else DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE
    except ValueError:
        value = DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE
    return max(0, value)


def _call_complete(slot: Mapping[str, Any]) -> bool:
    """A call the cap may run: a name, and arguments that parse to an object.

    Checked strictly -- never "repaired" -- and an empty buffer is not ``{}``:
    a call announced before its arguments arrive has none yet.
    """
    raw = str(slot.get("arguments") or "")
    if not str(slot.get("name") or "").strip() or not raw.strip():
        return False
    try:
        return isinstance(json.loads(raw), dict)
    except ValueError:
        return False


# Invariant: worker watchdog >= mux inter-chunk read timeout. Keep this
# floor aligned with the mux default; operators increasing the mux timeout
# must also raise SHAREDAGENTS_WORKER_READ_TIMEOUT_S.
_MUX_UPSTREAM_READ_TIMEOUT_S = 120.0
_DEFAULT_WORKER_READ_TIMEOUT_S = 120.0
# Longest a provider may hold an accepted request with ONLY SSE keep-alive
# comments (``: keep-alive``) and no ``data:`` chunk before the round is failed
# fast. DeepSeek queues requests under load and keeps the connection open with
# these comments (for up to ~10 min); each one resets httpx's per-read timeout
# above, so without this guard a queued request waits out the engine's 240s
# idle timeout on every retry. Strictly below that engine timeout; 0 disables.
_DEFAULT_QUEUE_STALL_S = 90.0


# Liveness for the engine's idle timeout while there is nothing to show. A
# tool-call argument yields no event until the call is complete, and vLLM
# sends no bytes at all for an array/object argument, so the local vLLM
# router sends this comment while the replica is still generating
# (``sharedagents_vllm.proxy.KEEPALIVE_COMMENT``). Other providers' keep-alive
# comments are deliberately not liveness: see ``_DEFAULT_QUEUE_STALL_S``.
_MUX_KEEPALIVE_PREFIX = ": sharedagents-mux keep-alive"
# Minimum gap between liveness events while tool-call arguments stream in.
_TOOL_ARGS_KEEPALIVE_S = 30.0


def _queue_stall_timeout_s() -> float:
    try:
        value = float(os.environ.get("SHAREDAGENTS_WORKER_QUEUE_STALL_S", "") or _DEFAULT_QUEUE_STALL_S)
    except (TypeError, ValueError):
        return _DEFAULT_QUEUE_STALL_S
    return value if value == value and value >= 0 else _DEFAULT_QUEUE_STALL_S


# Fresh attempts allowed per round when the reasoning was cut (see
# ``reasoning_cut_mid_code``).
_REASONING_CUT_RESAMPLES = 2


def reasoning_cut_mid_code(reasoning: str) -> bool:
    """True when reasoning ended inside an open inline code span or code block.

    A reasoning model that quotes ``</think>`` (or ``<tool_call>``) emits the
    real token -- one token in Qwen's vocabulary -- and the server's reasoning
    parser ends the reasoning there, mid-quote. What follows is either the rest
    of the thought spilled into the answer or an empty stop. A reasoning block
    the model closes on purpose does not end between backticks.
    """
    if not reasoning or reasoning.endswith("\n"):
        return False
    if reasoning.count("```") % 2:
        return True
    last_line = reasoning.rsplit("\n", 1)[-1].replace("```", "")
    return last_line.count("`") % 2 == 1


class _RetryRound(Exception):
    """Internal signal: re-issue this round without the rejected field.

    A plain ``break``/``continue`` cannot express this: the retry decision
    is made inside ``with client.stream(...)``, and Python rejects
    ``continue`` there (it may only target an enclosing loop, not a block),
    while ``break`` would exit the enclosing ``for`` — which IS the retry
    loop, so the loop would end instead of retrying. Raising reaches the
    loop header from any depth.
    """


def emit(kind: str, **fields: Any) -> None:
    sys.stdout.write(json.dumps({"kind": kind, **fields}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def terminal_finish_reason_seen_from_chunks(chunks) -> bool:
    """True iff any chunk delivered a non-null choices[0].finish_reason.

    The include_usage trailer (choices == []) never counts; [DONE] is a
    sentinel, not a finish reason. Used by the continuation classifier to
    distinguish a clean stop from a dropped stream that ended before any
    terminal finish-reason chunk arrived.
    """
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if choices and choices[0].get("finish_reason"):
            return True
    return False


# --- Leaked tool-call recovery ------------------------------------------
# Some local models (observed: qwen3.6-27b-fp8 on vLLM) emit a tool call in
# a malformed / degenerate format that the upstream tool-call parser misses
# entirely, e.g. with the ``<tool_call>``/``<function=NAME>`` opening tags
# dropped and the tool name buried inside a pseudo-argument:
#
#     <parameter=thinking>…scratchpad…</parameter>
#     <parameter=generic_tool>call:default_api:Bash</parameter>
#     <parameter=command>du -sh /path/to/data/*</parameter>
#     </function>
#     </tool_call>
#
# vLLM then streams the whole block as ``content`` (or ``reasoning_content``,
# which we promote to content) and returns ``finish_reason="stop"`` with NO
# structured ``tool_calls``. The turn ends with the un-run tool leaked into
# the user-visible reply — a silent premature stop. ``recover_leaked_tool_calls``
# parses that markup client-side so the dispatch loop can still run the tool
# and the turn continues instead of dead-ending.

# ---- XML-based leak detection (legacy format from Qwen3/vLLM) -------
_LEAK_SENTINELS = (
    "</tool_call>",
    "<tool_call>",
    "</function>",
    "<function=",
    "<parameter=",
)
# Pseudo-parameters some models emit that are NOT real tool arguments: the
# inline scratchpad (``thinking``) and the degenerate name carrier
# (``generic_tool``, whose value is e.g. ``call:default_api:Bash``).
_LEAK_RESERVED_PARAMS = frozenset({"thinking", "generic_tool"})
_LEAK_FUNC_NAME_RE = re.compile(r"<function\s*=\s*([A-Za-z_]\w*)\s*>")
# ``<parameter=generic_tool>`` value of the form ``[call:][ns:…:]NAME`` —
# we take the final dotted/colon segment as the tool name.
_LEAK_GENERIC_TOOL_RE = re.compile(
    r"<parameter\s*=\s*generic_tool\s*>\s*"
    r"(?:call\s*:\s*)?(?:[\w.]+\s*:\s*)*([A-Za-z_]\w*)\s*</parameter>"
)
_LEAK_PARAM_RE = re.compile(r"<parameter\s*=\s*([A-Za-z_]\w*)\s*>(.*?)</parameter>", re.DOTALL)
# Opening markers of leaked tool-call markup.
_LEAK_OPENERS = ("<tool_call>", "<function=", "<parameter=")
# Markers that make a round a genuine serialization rather than prose that
# merely mentions a parameter tag. A bare ``<parameter=name>`` inside a
# sentence ("pass it as `<parameter=state>`") must NOT trigger cleanup: cutting
# there loses the rest of a real answer. A whole leaked block always carries a
# call wrapper or a closing tag, including the degenerate shape whose opening
# tags were dropped.
_LEAK_STRUCTURAL = ("<tool_call>", "</tool_call>", "<function=", "</function>")

# ---- JSON-based leak detection (Qwen3.6 fallback format) ------------
# When the XML tool-call parser fails, Qwen3.6 sometimes drops back to
# emitting raw JSON tool calls inside the assistant content, e.g.:
#   {"name": "Bash", "arguments": {"command": "ls"}}
# or wrapped in markdown:
#   ```json\n[{"name": "Bash", "arguments": {...}}]\n```
# or as an array:
#   [{"name": "Bash", "arguments": {...}}]
# Detect these by looking for a JSON object/array that has tool-like keys.
_JSON_LEAK_NAME_KEYS = frozenset({"name", "function_name", "tool_name"})
_JSON_LEAK_ARG_KEYS = frozenset({"arguments", "parameters", "args", "params", "input"})
_JSON_LEAK_CODE_BLOCK_RE = re.compile(r"```(?:json|tool_calls?)?\s*\n?(.*?)\n?```", re.DOTALL)
# Minimum length for a JSON leak candidate — avoids flagging short prose
# that happens to contain a tool name as a dictionary key.
_JSON_LEAK_MIN_TEXT_LEN = 40


def _try_json_tool_call(tc: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a single parsed JSON object as a possible tool call.

    Returns normalized ``{"name": str, "arguments": dict}`` or None.
    Handles doubly-encoded arguments (a JSON string that itself is JSON).
    """
    name: str | None = None
    for k in _JSON_LEAK_NAME_KEYS:
        v = tc.get(k)
        if isinstance(v, str) and v.strip():
            name = v.strip()
            break
    if not name:
        return None
    # Normalise tool name: strip "call:" prefix and namespace prefixes
    name = name.split(":")[-1]
    args_raw: Any = None
    for k in _JSON_LEAK_ARG_KEYS:
        v = tc.get(k)
        if v is not None:
            args_raw = v
            break
    if args_raw is None:
        # No arguments key found — use everything except name-like keys
        # as the arguments dict (some models put args at the top level).
        args = {k: v for k, v in tc.items() if k not in _JSON_LEAK_NAME_KEYS and k not in ("type", "id", "index")}
        return {"name": name, "arguments": args} if args else None
    if isinstance(args_raw, dict):
        return {"name": name, "arguments": args_raw}
    if isinstance(args_raw, str):
        # Doubly-encoded JSON: the arguments value is a JSON string
        # containing the actual arguments dict.
        try:
            parsed = json.loads(args_raw)
        except json.JSONDecodeError:
            # Try repair_truncated_json for truncated arguments
            try:
                from sharedagents_model.json_repair import repair_truncated_json as _repair

                parsed = _repair(args_raw)
            except Exception:  # noqa: BLE001
                return {"name": name, "arguments": {"_raw": args_raw}}
        if isinstance(parsed, dict):
            return {"name": name, "arguments": parsed}
        return {"name": name, "arguments": {"_raw": args_raw}}
    return {"name": name, "arguments": {"_value": args_raw}}


def _detect_json_leaked_tool_calls(text: str) -> list[dict[str, Any]]:
    """Detect JSON-formatted tool calls leaked into assistant content.

    Tries three strategies:
    1. JSON code block content (```json ... ```)
    2. Top-level JSON object containing tool call keys
    3. Top-level JSON array of tool call objects

    Returns empty list if nothing usable found.
    """
    if not text or len(text) < _JSON_LEAK_MIN_TEXT_LEN:
        return []

    # Strategy 1: check for JSON in markdown code blocks
    for m in _JSON_LEAK_CODE_BLOCK_RE.finditer(text):
        block = m.group(1).strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            # Try repair_truncated_json for truncated content
            try:
                from sharedagents_model.json_repair import repair_truncated_json as _repair

                parsed = _repair(block)
            except Exception:  # noqa: BLE001
                continue
        if isinstance(parsed, dict):
            result = _try_json_tool_call(parsed)
            if result:
                return [result]
        elif isinstance(parsed, list):
            results = []
            for item in parsed:
                if isinstance(item, dict):
                    result = _try_json_tool_call(item)
                    if result:
                        results.append(result)
            if results:
                return results

    # Strategy 2: try parsing the entire text as a JSON array
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, list):
            results = []
            for item in parsed:
                if isinstance(item, dict):
                    result = _try_json_tool_call(item)
                    if result:
                        results.append(result)
            if results:
                return results

    # Strategy 3: try finding JSON objects in the text via brace-matching
    # Look for '{' characters then try to extract balanced JSON structures
    _json_start = -1
    brace_depth = 0
    in_str = False
    str_char = None
    for i, ch in enumerate(text):
        if in_str:
            if ch == "\\" and i + 1 < len(text):
                continue
            if ch == str_char:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            str_char = ch
            continue
        if ch == "{":
            if _json_start == -1:
                _json_start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and _json_start >= 0:
                candidate = text[_json_start : i + 1]
                if len(candidate) >= _JSON_LEAK_MIN_TEXT_LEN:
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(parsed, dict):
                            result = _try_json_tool_call(parsed)
                            if result:
                                return [result]
                _json_start = -1

    return []


def recover_leaked_tool_calls(text: str) -> list[dict[str, Any]]:
    """Best-effort recovery of a tool call leaked into assistant content.

    Returns a list of ``{"name": str, "arguments": dict}`` calls, or an
    empty list when ``text`` shows no tool-call markup or no usable call can
    be reconstructed.

    Supports three classes of leak:
    1. XML malformed format (legacy Qwen3/vLLM): ``<function=Name>...``
    2. JSON format (Qwen3.6 fallback): raw JSON object/array in content
    3. JSON code blocks: wrapped in markdown ```json ... ```

    Intentionally conservative: each format requires both a resolvable tool
    name AND at least one real (non-reserved) parameter so ordinary prose
    that merely *mentions* the markup isn't misread as a call.
    """
    if not text:
        return []

    # 1. Try XML format recovery (legacy)
    if any(s in text for s in _LEAK_SENTINELS):
        # An unterminated block is a TRUNCATED serialization. Salvaging the
        # complete parameters out of it dispatches a call the model never
        # finished, with silently missing arguments (observed: a call whose
        # ``state`` was dropped and answered "nonempty state required" four
        # rounds in a row). Fail closed; the turn ends normally instead.
        if text.count("<parameter") > text.count("</parameter>"):
            return []
        if text.count("<tool_call>") > text.count("</tool_call>"):
            return []
        name: str | None = None
        m = _LEAK_FUNC_NAME_RE.search(text)
        if m:
            name = m.group(1).strip()
        if not name:
            m = _LEAK_GENERIC_TOOL_RE.search(text)
            if m:
                name = m.group(1).strip()
        if name:
            args: dict[str, Any] = {}
            for pm in _LEAK_PARAM_RE.finditer(text):
                key = pm.group(1).strip()
                if key in _LEAK_RESERVED_PARAMS:
                    continue
                args[key] = pm.group(2).strip()
            if args:
                return [{"name": name, "arguments": args}]

    # 2. Try JSON format recovery (Qwen3.6 fallback)
    json_results = _detect_json_leaked_tool_calls(text)
    if json_results:
        return json_results

    return []


def visible_content_delta(full: str, already_visible: int) -> tuple[str, int]:
    """Return ``(text_safe_to_show_now, new_visible_len)`` for a round's content.

    A model that drifts to XML tool-call serialization streams the markup as
    ordinary ``content``. It must not reach the user: once a structural marker
    appears, everything from it onward is a serialization, not prose, so the
    live stream is cut there. A trailing partial marker (``"<tool_ca"``) is
    held back too, so the tag never leaks in pieces.

    Only ``_LEAK_STRUCTURAL`` cuts the stream. A bare ``<parameter=name>`` in a
    sentence ("pass it as `<parameter=state>`") is prose: cutting there would
    silently swallow the rest of a real answer, which is worse than the few
    hundred characters a rare dropped-opener block leaves visible until the
    round ends (history is cleaned either way).

    ``already_visible`` is the number of characters of ``full`` already
    emitted; the caller threads the returned length into the next call.
    """
    if already_visible >= len(full):
        return "", already_visible
    cut = len(full)
    for marker in _LEAK_STRUCTURAL:
        index = full.find(marker)
        if index >= 0:
            cut = min(cut, index)
    if cut == len(full):
        # No marker yet — hold back a tail that could begin one.
        hold = 0
        for marker in _LEAK_STRUCTURAL:
            for size in range(len(marker) - 1, 0, -1):
                if full.endswith(marker[:size]):
                    hold = max(hold, size)
                    break
        cut = len(full) - hold
    if cut <= already_visible:
        return "", already_visible
    return full[already_visible:cut], cut


def prefer_recovered_arguments(
    tool_calls_acc: dict[Any, dict[str, Any]],
    recovered: list[dict[str, Any]],
) -> list[str]:
    """Replace truncated structured arguments with the recovered ones.

    An upstream that half-parses XML tool-call markup returns a call whose
    argument object is a strict subset of what the model wrote (measured:
    ``{"action": "score"}`` for a call that carried ``action``, ``state`` and
    ``questions``). The recovered markup is the model's actual intent, so when
    a recovered call has the same name and a strict superset of the structured
    keys, use it. Returns the names that were replaced.
    """
    by_name = {str(c.get("name") or ""): c for c in recovered}
    slots_by_name: dict[str, list[dict[str, Any]]] = {}
    for slot in tool_calls_acc.values():
        slots_by_name.setdefault(str(slot.get("name") or ""), []).append(slot)
    preferred: list[str] = []
    for name, rec in by_name.items():
        slots = slots_by_name.get(name) or []
        # Exactly one structured slot: with two same-name calls the single tail
        # block cannot say which one it belongs to, and copying its arguments
        # into both would destroy a call the upstream parsed correctly.
        if len(slots) != 1:
            continue
        slot = slots[0]
        try:
            current = json.loads(slot.get("arguments") or "{}")
        except (TypeError, ValueError):
            continue
        if not isinstance(current, dict):
            continue
        if set(rec.get("arguments") or {}) > set(current):
            slot["arguments"] = json.dumps(rec["arguments"])
            preferred.append(name)
    return preferred


def leaked_call_tail(text: str) -> str:
    """The final tool-call block in ``text``, or "" when the markup is quoted.

    A model that emits a call puts it LAST — its tool protocol says the call is
    the last thing it writes. So the block at the tail is the call. Markup
    anywhere else is a quotation (an example, a regression fixture, a
    docstring) and must neither be dispatched nor be allowed to override a real
    call's arguments. A round whose content merely MENTIONS the format keeps
    its text intact.

    Returns the block from its opener (or, for the degenerate shape whose
    opening tags were dropped, from its first parameter) to the end.
    """
    stripped = text.rstrip()
    if not stripped.endswith(("</tool_call>", "</function>")):
        return ""
    start = stripped.rfind("<tool_call>")
    if start < 0:
        start = stripped.rfind("<function=")
    if start < 0:
        start = stripped.rfind("<parameter")
        # Walk back over the contiguous run of parameter blocks so a
        # dropped-opener call keeps its name carrier (``generic_tool``) and the
        # arguments that preceded the last one.
        while start > 0:
            before = stripped[:start].rstrip()
            if not before.endswith("</parameter>"):
                break
            previous = before.rfind("<parameter")
            if previous < 0:
                break
            start = previous
    return stripped[start:] if start >= 0 else ""


def strip_leaked_tool_markup(text: str) -> str:
    """Return the legitimate prose before any leaked tool-call markup.

    ``recover_leaked_tool_calls`` reconstructs the call from ``text``; this
    returns the user-facing remainder with the raw ``<parameter=…>`` /
    ``<tool_call>`` markup or JSON code blocks removed so a recovered round
    doesn't persist (or re-feed the model) its own malformed tool-call
    serialization. ``text`` with no markup is returned unchanged.
    """
    # Strip XML-style markup at the first STRUCTURAL marker. A bare
    # ``<parameter=…>`` inside a sentence is prose; cutting there loses real
    # answer text (see ``visible_content_delta`` for the same rule).
    cuts = [i for i in (text.find(t) for t in _LEAK_STRUCTURAL) if i >= 0]
    if cuts:
        return text[: min(cuts)].rstrip()
    # Strip JSON code blocks that contain leaked tool calls
    if _detect_json_leaked_tool_calls(text):
        stripped = _JSON_LEAK_CODE_BLOCK_RE.sub("", text).strip()
        # Also strip bare JSON arrays/objects at start of text
        _strip_leading_json = re.compile(r'^\[?\s*\{.*?"name"\s*:\s*"[^"]+".*?\}\s*\]?\s*', re.DOTALL)
        if stripped:
            stripped = _strip_leading_json.sub("", stripped).strip()
        if stripped:
            return stripped
    return text


def _install_term_handler() -> None:
    """Install SIGTERM/SIGINT handlers that emit a terminal ``aborted``
    event before exit, so the parent never sees a silent mid-stream death.
    """

    def handler(signum, frame):  # noqa: ARG001
        emit("aborted")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


_SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "max_tokens",
    "seed",
    "reasoning_effort",
)


def _is_sse_content_type(ctype: str, *, responses_mode: bool = False) -> bool:
    """True when the upstream Content-Type is a usable SSE stream.

    ChatGPT Codex Responses returns HTTP 200 with a real ``event:`` /
    ``data:`` body but an empty Content-Type. Accept that only for the
    Responses transport; JSON/HTML 200s on chat-completions still fail.
    """
    c = (ctype or "").strip().lower()
    if "text/event-stream" in c:
        return True
    return bool(responses_mode and not c)


def forward_sampling_fields(request: dict[str, Any]) -> dict[str, Any]:
    """Pick the sampling knobs the caller actually set, verbatim.

    These come from the agent's `Model` (per-mode sampling profile) plus
    any per-turn override. Fields the caller didn't set — or set to
    ``null``/None — are dropped, so the upstream gets to apply its own
    default.

    There is deliberately NO unconditional temperature floor: when the
    model row or config explicitly sets temperature to ``null``/None, it
    never lands here, and we leave it off the wire entirely so the
    upstream (OpenAI, Bedrock, Anthropic, etc.) decides its own default.
    This matters for models like Claude 4.x (Bedrock) and o1/o3 (OpenAI)
    that reject the temperature parameter outright.
    """
    fields = {k: request[k] for k in _SAMPLING_FIELDS if k in request and request[k] is not None}
    # The model-row speed capability is intentionally config-driven.  The
    # request builder records the exact approved wire key; forwarding only
    # those names keeps arbitrary request keys out of the upstream payload.
    for key in request.get("_forward_fields", ()):
        if isinstance(key, str) and key and key in request and request[key] is not None:
            fields[key] = request[key]
    return fields


def cached_prompt_tokens(usage: Mapping[str, Any]) -> int | None:
    """Cached prompt-token count from an OpenAI-compatible ``usage`` object.

    Two upstream shapes exist and only one is ever present per response:

    * vLLM / OpenAI — ``usage.prompt_tokens_details.cached_tokens`` (the
      prefix-cache hit; ``0`` on a full miss). This is authoritative and
      wins when present.
    * DeepSeek — ``usage.prompt_cache_hit_tokens`` (paired with
      ``prompt_cache_miss_tokens``); it does NOT populate the
      ``prompt_tokens_details`` shape. Without reading it, every DeepSeek
      prompt token is priced at the full input rate ($0.14/$1.74 per M)
      instead of cached_read ($0.0028/$0.003625 per M) — a ~50x over-count
      on the cached portion, and a 0% ``cache_hit_pct`` on the stats row.

    Returns ``None`` when neither field is present (unknown → let the
    caller leave ``prompt_tokens_cached`` unset), never guessing a 0.
    """
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, Mapping):
        cached = details.get("cached_tokens")
        if cached is not None:
            try:
                return int(cached)
            except (TypeError, ValueError):
                return None
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is not None:
        try:
            return int(hit)
        except (TypeError, ValueError):
            return None
    return None


def main() -> int:
    """Read one stdin JSON request, stream one chat-completions turn, emit
    JSONL events on stdout.
    """
    _install_term_handler()

    try:
        request = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as e:
        emit("error", message=f"bad request json: {e}")
        return 2

    # Native Bedrock rows are dispatched to the Converse transport BEFORE any
    # OpenAI payload assembly, message sanitizing or SSE logic: `bedrock://`
    # is not an HTTP endpoint, and the AWS protocol is not OpenAI wire.
    # Traceability: plan/bedrock-native-converse.md (BED-01/03).
    if request.get("api") == BEDROCK_API:
        from sharedagents_gateway import bedrock_worker

        return bedrock_worker.run_bedrock_worker(request)

    endpoint = (request.get("endpoint") or "").rstrip("/")
    model = request.get("model")
    catalog_provider = str(request.get("provider") or "").strip()
    messages = request.get("messages") or []
    if not endpoint or not model or not messages:
        emit(
            "error",
            message=(
                "missing endpoint/model/messages: no model is configured for this agent; "
                "pick one with /model or the model picker"
            ),
        )
        return 2

    # Honour what this endpoint has already refused. The gateway
    # (``worker_request._build_worker_request``) reconciles the messages IT
    # assembles, but a round whose history was built elsewhere (a resumed
    # session, another caller) would otherwise re-discover a known-strict
    # host's 400 the hard way. Nothing learned for this endpoint means no
    # change at all.
    from sharedagents_model.providers import by_name as _provider_by_name
    from sharedagents_model.wire_replay import (
        defang_message_control_tokens as _defang_message_control_tokens,
        sanitize_messages as _sanitize_messages,
    )

    backend_name = _provider_by_name(request.get("backend", "")).name
    messages = _sanitize_messages(
        messages,
        {"baseUrl": endpoint, "provider": backend_name},
    )
    # vLLM / SGLang render the chat template server-side and tokenize control-
    # token text inside messages as the real token (see
    # ``defang_message_control_tokens``).
    if backend_name in ("vllm", "sglang"):
        messages = _defang_message_control_tokens(messages)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        # Ask vLLM to include token-count usage in the final SSE chunk so we
        # can compute tokens-per-second exactly instead of approximating from
        # character count.
        "stream_options": {"include_usage": True},
    }
    # Forward sampling knobs from the request (temperature, top_p, …).
    payload.update(forward_sampling_fields(request))
    # Optional tool-calling: the agent runtime sends `tools` (OpenAI
    # function-tool schemas) and `tool_choice`. We forward as-is.
    tools = request.get("tools")
    if tools:
        payload["tools"] = tools
        choice = request.get("tool_choice")
        if choice:
            payload["tool_choice"] = choice
    extra = request.get("extra") or {}
    payload.update(extra)

    from sharedagents_gateway.codex_responses import (
        ResponsesStreamState,
        apply_codex_responses_event,
        build_codex_responses_payload,
        is_codex_responses_request,
    )

    responses_mode = is_codex_responses_request(request, endpoint)
    if responses_mode:
        payload = build_codex_responses_payload(payload)

    # Per-model thinking-tag + prefill behaviour. `models.for_name` resolves
    # the upstream model id (e.g. `gemma-4-27b-it`) to a `Model` carrying
    # the right open/close tags for inline reasoning detection. Gemma 4
    # uses `<thought>...</thought>` rather than Qwen3's `<think>...</think>`.
    from sharedagents_model.models import for_name as _model_for_name

    mdl = _model_for_name(model)
    open_tag, close_tag = mdl.thinking_tags
    # Token-id reasoning split (``sharedagents_gateway.reasoning_split``): vLLM
    # streams each chunk's token ids, so a quoted ``</think>`` can be told
    # apart from the real close.
    close_id = mdl.reasoning_close_token_id if backend_name == "vllm" and not responses_mode else None
    if close_id is not None:
        payload["return_token_ids"] = True

    # Prefill: for Gemma 4 (and any future family with the same pattern),
    # when thinking is enabled, end the prompt with the open tag so the
    # model continues inside a thought block. Probe `enable_thinking` from
    # the chat-template kwargs the request builder set; if absent, default
    # to True (matches the rest of the codebase where thinking is opt-out).
    cte_kwargs = (extra.get("chat_template_kwargs") or {}) if isinstance(extra, dict) else {}
    enable_thinking = cte_kwargs.get("enable_thinking")
    if enable_thinking is None:
        enable_thinking = bool(request.get("enable_thinking", True))
    prefilled_thinking = False
    if (not responses_mode) and mdl.prefill_thinking_open_tag and enable_thinking:
        # Append an assistant message opened with the thinking tag and tell
        # vLLM to continue from it instead of starting a fresh assistant
        # turn. The splitter is primed with `in_think=True` so the first
        # streamed token is routed to the thinking channel until we see
        # the close tag.
        payload_msgs: list[Any] = list(payload.get("messages") or [])
        payload_msgs.append({"role": "assistant", "content": f"{open_tag}\n"})
        payload["messages"] = payload_msgs
        payload["continue_final_message"] = True
        payload["add_generation_prompt"] = False
        prefilled_thinking = True

    # The exact message list this attempt sends. Kept so a rejected retry
    # can rebuild the payload without the offending field — the prefill
    # above appends to `payload["messages"]`, and rebuilding from the
    # mutated copy would stack a second prefill turn onto the retry.
    base_messages: list[Any] = list(payload.get("messages") or [])

    def _drop_field_from_payload(name: str) -> None:
        payload["messages"] = _strip_unsupported_fields(
            base_messages,
            frozenset({name}),
        )

    def _ensure_reasoning_in_payload() -> None:
        payload["messages"] = _ensure_required_reasoning_fields(base_messages)

    # Per-chunk read timeout = the THINKING-INACTIVITY WATCHDOG. httpx's
    # ``read`` timeout is per socket-read, so it measures the gap BETWEEN
    # streamed chunks, and a streaming model sends one chunk per token —
    # vLLM/SGLang emit no bytes at all while the model is "thinking" with
    # nothing to show. So this is effectively "no output token for N
    # seconds": each arriving token resets the window, and a genuine stall
    # (model wedged mid-thinking, GPU idle, no 5xx) surfaces as
    # ``http error: ReadTimeout`` after N seconds instead of hanging the
    # whole turn for minutes. The recovery path (round-retry → loop/teacher
    # supervision) re-drives a genuinely-wedged model. Default 120s allows
    # local concurrency and cold prefill without timing out before the
    # mux's 120s inter-chunk budget. This is an inactivity window, not an
    # overall request deadline or a guarantee that mux failover finishes first.
    # A continuous stream is unaffected (tokens arrive every ~20ms at 50 tok/s,
    # far under the window). Override via ``SHAREDAGENTS_WORKER_READ_TIMEOUT_S``.
    try:
        read_timeout_s = float(
            os.environ.get("SHAREDAGENTS_WORKER_READ_TIMEOUT_S", "") or _DEFAULT_WORKER_READ_TIMEOUT_S
        )
    except (TypeError, ValueError):
        read_timeout_s = _DEFAULT_WORKER_READ_TIMEOUT_S
    # Clamp overrides below the mux default, including invalid NaN values.
    if not read_timeout_s >= _MUX_UPSTREAM_READ_TIMEOUT_S:
        _log.warning(
            "SHAREDAGENTS_WORKER_READ_TIMEOUT_S=%s is below the mux upstream "
            "read-timeout (%ss); clamping up to %ss to preserve the "
            "worker>=mux invariant.",
            read_timeout_s,
            _MUX_UPSTREAM_READ_TIMEOUT_S,
            _DEFAULT_WORKER_READ_TIMEOUT_S,
        )
        read_timeout_s = _DEFAULT_WORKER_READ_TIMEOUT_S
    timeout = httpx.Timeout(
        connect=10.0,
        read=read_timeout_s,
        write=10.0,
        pool=10.0,
    )
    headers = {"Accept": "text/event-stream"}
    extra_headers = request.get("headers") or {}
    if isinstance(extra_headers, dict):
        headers.update({str(k): str(v) for k, v in extra_headers.items() if v is not None})

    dump_path = os.environ.get("SHAREDAGENTS_DEBUG_DUMP")
    dump_fh = None
    if dump_path:
        try:
            dump_fh = open(dump_path, "w", encoding="utf-8")
            dump_fh.write(f"# request payload\n{json.dumps(payload, indent=2)}\n\n# raw sse stream\n")
            dump_fh.flush()
        except OSError:
            dump_fh = None

    # Per-backend differences (vLLM vs SGLang vs unknown) live in
    # `providers.py`. The worker request can carry an optional `backend`
    # field so we use the provider's preferred alias order; otherwise we
    # fall back to a UNKNOWN provider that probes every alias.
    from sharedagents_model.providers import by_name

    provider = by_name(request.get("backend", ""))
    REASONING_KEYS = provider.reasoning_keys

    t_start = time.monotonic()
    t_headers: float | None = None
    t_first_token: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    # Cached-token count from vLLM's `usage.prompt_tokens_details.cached_tokens`
    # (`--enable-prefix-caching` exposes this). Lets the agent surface
    # cache-hit ratio per round so an operator can tell "TTFT is climbing
    # because cache misses" vs "because the prompt grew." Also fits a
    # `cache_hit_pct` line on the status bar.
    prompt_tokens_cached: int | None = None
    # Authoritative reasoning-token count from
    # `usage.completion_tokens_details.reasoning_tokens` (OpenAI / DeepSeek
    # surface it). When present we report it directly; when absent we
    # estimate by char ratio between the streamed reasoning vs content.
    reasoning_tokens_usage: int | None = None
    reasoning_chars = 0
    content_chars = 0
    # Accumulator for streamed `delta.tool_calls` fragments. Keyed on the
    # tool-call's `index` (vLLM/SGLang both use position-indexed chunks):
    #   { 0: {"id": "...", "name": "...", "arguments": "<JSON fragments>"} }
    # On finish_reason="tool_calls" we parse `arguments` and emit a single
    # `tool_calls` event. `finish_reason` is recorded so the agent runtime
    # can tell "stop with content" from "tool_calls awaiting results".
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    max_tool_calls = _max_tool_calls_per_response()
    tool_calls_capped = 0
    # Calls left unfinished when the batch cap stopped the stream.
    capped_incomplete: list[dict[str, Any]] = []
    finish_reason: str | None = None
    # True once a chunk delivers a non-null finish_reason. Distinguishes a clean
    # terminal stop from a dropped stream (where ``finish_reason`` stays None and
    # must NOT be conflated with "stop"). Surfaced on the metrics event so the
    # continuation classifier can resume a truncated stream.
    terminal_finish_reason_seen = False
    # Accumulated visible content this round. Buffered so that, at finish,
    # we can scan it for a tool call the upstream parser missed (see
    # ``recover_leaked_tool_calls``).
    content_buf: list[str] = []
    # How much of ``content_buf`` has actually been emitted to the wire. Runs
    # behind ``content_chars`` when a leaked tool-call opener cuts the visible
    # prefix short (see ``visible_content_delta``).
    content_visible_chars = 0
    responses_state = ResponsesStreamState() if responses_mode else None

    post_url = f"{endpoint}/responses" if responses_mode else f"{endpoint}/chat/completions"
    # One extra attempt each is allowed for rejected token ids and a rejected
    # or required replayed reasoning field in Chat Completions, and up to
    # ``_REASONING_CUT_RESAMPLES`` fresh attempts for a cut reasoning block.
    # `t_attempt_headers` / `t_first_token` are PER ATTEMPT so the timing
    # is measured against the attempt that actually produced the stream: a
    # rejected first try contributes no phantom TTFT or prefill latency.
    max_attempts = 2 + _REASONING_CUT_RESAMPLES + (close_id is not None)
    retried_reasoning_replay = False
    cut_resamples = 0
    reasoning_replay_retry_enabled = not responses_mode and os.environ.get(
        "SHAREDAGENTS_DISABLE_REASONING_REPLAY_RETRY", "0"
    ) not in ("1", "true", "True")
    t_attempt_headers: float | None = None
    emitted_reasoning_prefix = ""
    for _attempt in range(max_attempts):
        headers_received = False
        t_attempt_headers = None
        t_first_token = None
        attempt_reasoning: list[str] = []
        answer_started = False
        split = ReasoningSplit(close_id) if close_id is not None else None
        saw_token_ids = False
        reasoning_prefix_offset = 0
        reasoning_prefix_matching = True
        # Usage and channel counts describe the accepted attempt, like its
        # TTFT. A discarded attempt's usage must not survive a missing usage
        # chunk on the replacement stream.
        completion_tokens = prompt_tokens = prompt_tokens_cached = None
        reasoning_tokens_usage = None
        reasoning_chars = content_chars = 0
        # ``content_buf`` is NOT reset per attempt, so neither is what the wire
        # has already seen: assume everything buffered so far was emitted.
        content_visible_chars = len("".join(content_buf))
        finish_reason = None
        terminal_finish_reason_seen = False
        splitter = ThinkSplitter(open_tag, close_tag, in_think=prefilled_thinking)
        try:
            with httpx.Client(timeout=timeout, headers=headers) as client:
                with client.stream("POST", post_url, json=payload) as resp:
                    # Entering the `with` returns once response headers are in.
                    # For a streaming SSE chat-completions call this happens at
                    # request ACCEPTANCE — the upstream's ASGI/uvicorn layer
                    # flushes status+headers as soon as the request is queued,
                    # NOT after prefill. So this stamp is "request accepted",
                    # and the real prefill+first-decode cost lives in
                    # `t_first_token - t_headers`. (Earlier comments here
                    # claimed headers came back after prefill; that was wrong
                    # and made `pft_ms` misleading on long-prompt diagnostics.)
                    t_headers = time.monotonic()
                    headers_received = True
                    t_attempt_headers = t_headers
                    if resp.status_code != 200:
                        body = resp.read().decode("utf-8", "replace")
                        if close_id is not None and resp.status_code in (400, 422) and "return_token_ids" in body:
                            # Upstream without token ids: stream without the
                            # split (the backtick check below still applies).
                            payload.pop("return_token_ids", None)
                            close_id = None
                            raise _RetryRound()
                        # A strict OpenAI-compatible host rejects a REPLAYED reasoning
                        # field outright (property messages.N.assistant.
                        # reasoning_content is unsupported, HTTP 400). The field is a
                        # DeepSeek extension, so it is dropped only when a provider
                        # actually refuses it: remember the refusal for this host and
                        # retry ONCE - a rejected round must not cost the turn. Every
                        # other 4xx/5xx keeps the historical fail-through to the
                        # engine's own round-retry.
                        rejected = None
                        if reasoning_replay_retry_enabled and not retried_reasoning_replay:
                            rejected = _unsupported_message_field(
                                resp.status_code,
                                body,
                            )
                        if rejected is not None:
                            _record_rejected_field(endpoint, rejected)
                            _drop_field_from_payload(rejected)
                            retried_reasoning_replay = True
                            emit(
                                "reasoning_replay_retry",
                                field=rejected,
                                endpoint=endpoint,
                            )
                            raise _RetryRound()
                        if (
                            reasoning_replay_retry_enabled
                            and not retried_reasoning_replay
                            and _is_missing_reasoning_content_error(
                                resp.status_code,
                                body,
                            )
                        ):
                            _ensure_reasoning_in_payload()
                            retried_reasoning_replay = True
                            emit(
                                "reasoning_replay_retry",
                                field="reasoning_content_required",
                                endpoint=endpoint,
                            )
                            raise _RetryRound()
                        emit(
                            "error",
                            message=f"HTTP {resp.status_code}: {body[:500]}",
                        )
                        return 1
                    # Validate Content-Type. A 200 response that ISN'T SSE
                    # (HTML interstitial, plain JSON, captive-portal page)
                    # would be parsed as garbage SSE below — every chunk
                    # silently fails JSON decode and the user sees an empty
                    # reply. Surface it as a structured error instead.
                    #
                    # ChatGPT Codex Responses streams real SSE (`event:` /
                    # `data:` frames) but omits Content-Type entirely. Treat
                    # a blank type as SSE only on that transport — other
                    # backends still fail closed.
                    ctype = resp.headers.get("content-type", "")
                    if not _is_sse_content_type(ctype, responses_mode=responses_mode):
                        body = resp.read().decode("utf-8", "replace")
                        emit(
                            "error",
                            message=(
                                "upstream returned non-SSE content-type "
                                f"{ctype!r} on a stream=true request "
                                f"(status=200). body[:300]={body[:300]!r}"
                            ),
                        )
                        return 1
                    json_decode_failures = 0
                    # Set once the upstream closes the stream with the SSE
                    # ``[DONE]`` sentinel (consumed by the post-loop handling).
                    saw_done_sentinel = False
                    queue_stall_s = _queue_stall_timeout_s()
                    saw_keepalive = saw_data = False
                    last_args_keepalive = time.monotonic()
                    for raw in resp.iter_lines():
                        # Accepted but queued: only keep-alive comments so far.
                        # Fail the round now (transient, so the engine retries
                        # it) instead of waiting out the engine idle timeout.
                        if (
                            queue_stall_s > 0
                            and saw_keepalive
                            and not saw_data
                            and t_attempt_headers is not None
                            and time.monotonic() - t_attempt_headers > queue_stall_s
                        ):
                            emit(
                                "error",
                                message=(
                                    f"upstream_unavailable: provider held the request "
                                    f"for {queue_stall_s:g}s with only SSE keep-alive "
                                    f"comments and no output (queued or overloaded); "
                                    f"timed out waiting for the first token"
                                ),
                            )
                            return 1
                        if not raw:
                            continue
                        line = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
                        if dump_fh is not None:
                            dump_fh.write(line + "\n")
                            dump_fh.flush()
                        if not line.startswith("data:"):
                            if line.startswith(":"):
                                saw_keepalive = True
                                if saw_data and line.startswith(_MUX_KEEPALIVE_PREFIX):
                                    emit("keepalive", source="mux")
                            continue
                        saw_data = True
                        data = line[5:].strip()
                        if data == "[DONE]":
                            saw_done_sentinel = True
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as je:
                            # Track silent decode failures so the parent can
                            # see when tokens are being lost. We don't bail
                            # the stream — a single malformed chunk shouldn't
                            # kill the turn — but the count surfaces in the
                            # metrics payload below so the operator notices
                            # and the SSE consumer logs it as a `decode_drop`.
                            json_decode_failures += 1
                            if json_decode_failures <= 3:
                                # First few only — emit so the caller sees
                                # the actual content; afterwards just count.
                                emit(
                                    "decode_drop",
                                    message=str(je),
                                    line_head=data[:200],
                                )
                            continue
                        if responses_mode and responses_state is not None:
                            if isinstance(chunk, dict):
                                apply_codex_responses_event(
                                    chunk,
                                    responses_state,
                                    now=time.monotonic(),
                                    emit=emit,
                                )
                                if responses_state.error:
                                    emit("error", message=responses_state.error)
                                    return 1
                                if responses_state.done:
                                    break
                            continue
                        # Defensive: vLLM (and some OpenAI-compat servers)
                        # surface a request-level 400 on a stream=true
                        # request as an SSE ``data:`` chunk shaped
                        # ``{"error": {"message": ..., "type": ..., "code": 400}}``
                        # rather than an HTTP 4xx. Without this branch we
                        # silently drop the chunk (no ``choices``, no
                        # ``usage``) and exit with ``completion_tokens=1
                        # tokens_estimated=true`` — looks like the model
                        # produced a single empty token, when in fact the
                        # upstream never started decoding. Surface as a
                        # structured ``error`` event so the parent agent
                        # sees the real reason (e.g. "min_p ... not yet
                        # supported with speculative decoding").
                        err = chunk.get("error")
                        if isinstance(err, dict):
                            msg = err.get("message") or json.dumps(err)
                            code = err.get("code")
                            if (
                                close_id is not None
                                and str(code) in ("400", "422")
                                and "return_token_ids" in str(msg)
                                and t_first_token is None
                                and not tool_calls_acc
                            ):
                                payload.pop("return_token_ids", None)
                                close_id = None
                                raise _RetryRound()
                            suffix = f" (code={code})" if code is not None else ""
                            emit(
                                "error",
                                message=f"upstream SSE error{suffix}: {msg}",
                            )
                            return 1
                        # vLLM with stream_options.include_usage=true sends a
                        # final chunk with empty choices and a populated usage
                        # field. Capture and continue.
                        usage = chunk.get("usage")
                        if usage:
                            if usage.get("completion_tokens") is not None:
                                completion_tokens = int(usage["completion_tokens"])
                            if usage.get("prompt_tokens") is not None:
                                prompt_tokens = int(usage["prompt_tokens"])
                            # Cached prompt-token count, across the two upstream
                            # shapes (vLLM/OpenAI `prompt_tokens_details.cached_tokens`
                            # and DeepSeek `prompt_cache_hit_tokens`). See
                            # ``cached_prompt_tokens`` for the resolution + rationale.
                            _cached = cached_prompt_tokens(usage)
                            if _cached is not None:
                                prompt_tokens_cached = _cached
                            # Some upstreams (OpenAI, DeepSeek) split the
                            # completion-side count into reasoning vs visible
                            # output under `completion_tokens_details`. When
                            # present we trust it over the char-ratio
                            # estimate built below.
                            comp_details = usage.get("completion_tokens_details") or {}
                            if isinstance(comp_details, dict):
                                rt = comp_details.get("reasoning_tokens")
                                if rt is not None:
                                    reasoning_tokens_usage = int(rt)
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        rc = None
                        for k in REASONING_KEYS:
                            v = delta.get(k)
                            if v:
                                rc = v
                                break
                        text = delta.get("content")
                        token_ids = choices[0].get("token_ids")
                        if split is not None:
                            if token_ids:
                                saw_token_ids = True
                            if saw_token_ids:
                                # Tool argument tokens belong to their own
                                # channel. A literal close there was never
                                # removed from delta.content by the parser.
                                split_ids = token_ids
                                if tool_calls_acc or delta.get("tool_calls"):
                                    split_ids = None
                                rc, text = split.feed(rc or "", text or "", split_ids)
                                if (
                                    split.phase == "literal"
                                    and time.monotonic() - last_args_keepalive >= _TOOL_ARGS_KEEPALIVE_S
                                ):
                                    last_args_keepalive = time.monotonic()
                                    emit("keepalive", source="reasoning_split")
                        if rc:
                            if t_first_token is None:
                                t_first_token = time.monotonic()
                            reasoning_chars += len(rc)
                            attempt_reasoning.append(rc)
                            visible_rc = rc
                            if saw_token_ids and reasoning_prefix_matching:
                                # Resampling may reproduce a prefix already
                                # streamed to Activity. Suppress that prefix
                                # across arbitrary chunk boundaries.
                                n = 0
                                while (
                                    n < len(rc)
                                    and reasoning_prefix_offset < len(emitted_reasoning_prefix)
                                    and rc[n] == emitted_reasoning_prefix[reasoning_prefix_offset]
                                ):
                                    n += 1
                                    reasoning_prefix_offset += 1
                                visible_rc = rc[n:]
                                if visible_rc:
                                    reasoning_prefix_matching = False
                                    if reasoning_prefix_offset < len(emitted_reasoning_prefix):
                                        emit("thinking", text="\n\n")
                            if visible_rc:
                                emit("thinking", text=visible_rc)
                        # Without token ids: at the first answer-side signal
                        # after reasoning, if the reasoning ended mid-quote,
                        # drop this attempt before any of its answer reaches
                        # the parent and resample.
                        if (
                            not saw_token_ids
                            and not answer_started
                            and (text or delta.get("tool_calls") or choices[0].get("finish_reason"))
                        ):
                            answer_started = True
                            if (
                                attempt_reasoning
                                and cut_resamples < _REASONING_CUT_RESAMPLES
                                and choices[0].get("finish_reason") in (None, "stop", "tool_calls")
                                and reasoning_cut_mid_code("".join(attempt_reasoning))
                            ):
                                cut_resamples += 1
                                _log.warning(
                                    "reasoning cut mid-quote; resampling the round (%d/%d)",
                                    cut_resamples,
                                    _REASONING_CUT_RESAMPLES,
                                )
                                emit(
                                    "reasoning_cut_resample",
                                    attempt=cut_resamples,
                                    tail="".join(attempt_reasoning)[-80:],
                                )
                                emit("thinking", text="\n\n")
                                raise _RetryRound()
                        if text:
                            if t_first_token is None:
                                t_first_token = time.monotonic()
                            thinking, content = splitter.feed(text)
                            if thinking:
                                reasoning_chars += len(thinking)
                                emit("thinking", text=thinking)
                            if content:
                                content_chars += len(content)
                                content_buf.append(content)
                                visible, content_visible_chars = visible_content_delta(
                                    "".join(content_buf),
                                    content_visible_chars,
                                )
                                if visible:
                                    emit("content", text=visible)
                        # Tool calls stream as positional deltas; accumulate.
                        tcalls = delta.get("tool_calls")
                        if tcalls:
                            for tc in tcalls:
                                idx = int(tc.get("index", 0))
                                if (
                                    max_tool_calls
                                    and idx not in tool_calls_acc
                                    and len(tool_calls_acc) >= max_tool_calls
                                ):
                                    # Skip, don't stop: later deltas in this
                                    # chunk may still finish a kept call.
                                    tool_calls_capped = max_tool_calls
                                    continue
                                slot = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                                if tc.get("id"):
                                    slot["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    slot["name"] = fn["name"]
                                arg_frag = fn.get("arguments")
                                if arg_frag:
                                    slot["arguments"] += arg_frag
                                    if time.monotonic() - last_args_keepalive >= _TOOL_ARGS_KEEPALIVE_S:
                                        last_args_keepalive = time.monotonic()
                                        emit("keepalive", source="tool_args")
                                    # Loop detection on the streamed
                                    # `arguments` JSON. The `delta.content`
                                    # / `delta.reasoning_content` detectors
                                    # in `acp_bridge.py` don't see this
                                    # channel — the worker buffers args
                                    # internally and only emits a single
                                    # `tool_calls` event at finish. On
                                    # gemma-4-31b-nvfp4-mtp we've observed
                                    # the model entering a token-collapse
                                    # loop *inside* the tool args (3022+
                                    # tokens emitted, finish_reason=
                                    # tool_calls, JSON unrecoverable).
                                    # Sample at LOOP_CHECK_STRIDE-char
                                    # intervals so the cost is bounded
                                    # over a long argument string.
                                    buf = slot["arguments"]
                                    last_check = slot.get("_loop_checked_at", 0)
                                    if LOOP_DETECT_ENABLED and len(buf) - last_check >= LOOP_CHECK_STRIDE:
                                        slot["_loop_checked_at"] = len(buf)
                                        hit = _detect_loops_layered(
                                            buf,
                                            channel="tool_args",
                                        )
                                        if hit is not None:
                                            tier, evidence = hit
                                            # Surface tiers (ngram/tight) are
                                            # TELEMETRY-ONLY on tool_args —
                                            # same FP class as thinking/content
                                            # (SSH boilerplate,
                                            # Write bodies, path lists, Bash
                                            # fragments aborted mid-call). Emit
                                            # soft notice and keep streaming;
                                            # real collapse is caught post-round
                                            # by ratio / server repetition /
                                            # ProgressTracker. Abort allowlist:
                                            # TOOL_ARGS_ABORT_TIERS (== entropy/
                                            # ratio); those tiers are currently
                                            # skipped inside the detector for
                                            # tool_args, so this path is soft
                                            # by default. Traceability:
                                            if not tool_args_loop_should_abort(
                                                tier,
                                            ):
                                                emit(
                                                    "tool_args_loop_soft",
                                                    index=idx,
                                                    name=slot.get("name") or "",
                                                    tier=tier,
                                                    evidence=evidence[:80],
                                                    args_chars=len(buf),
                                                    checked_at=len(buf),
                                                )
                                                continue
                                            # Ambiguous-chrome sustain gate
                                            # (see turn_runtime.LOOP_AMBIGUOUS_*).
                                            # Production false-positive 2026-05-28
                                            # session <session-id>: an Edit
                                            # tool_args containing a markdown
                                            # table separator (``--------``) hit
                                            # the tight tier on the first 330-char
                                            # check. With the coverage gate alone
                                            # a long separator clears 50%. The
                                            # sustain gate requires the SAME
                                            # evidence to persist for 30 s before
                                            # aborting — markdown chrome finishes
                                            # in <1 s, real collapse runs to
                                            # ``max_tokens``. Unambiguous heads
                                            # (``!``/``?``/alphanumeric) skip the
                                            # gate inside ``_should_fire_with_sustain``.
                                            # Scope: ``slot["_loop_sustain"]``
                                            # is per-tool-call, per-round. The
                                            # worker exits at round end and
                                            # ``tool_calls_acc`` is dropped, so
                                            # sustain state never crosses rounds.
                                            # That's correct: every documented
                                            # Qwen / Gemma collapse runs to
                                            # ``max_tokens`` within a single
                                            # round (vllm#27364 / #37554 /
                                            # #39407, sglang#19603 / #23687,
                                            # deepmind/gemma#622, OpenReview
                                            # lPwsHTEPQ9), so a 30 s grace per
                                            # round catches real collapse while
                                            # round-end is a meaningful "the
                                            # model recovered" checkpoint. The
                                            # post-round ratio-loop guard (see
                                            # ``LocalLLMEngine._check_ratio_loop``)
                                            # is an orthogonal backstop.
                                            sustain = slot.setdefault("_loop_sustain", {})
                                            try:
                                                should_fire = _should_fire_with_sustain(
                                                    evidence,
                                                    sustain,
                                                    time.monotonic(),
                                                )
                                            except Exception:  # noqa: BLE001
                                                # Defensive symmetry with the
                                                # engine-side wrapper: any
                                                # unexpected helper failure
                                                # should fall through to the
                                                # pre-fix abort behaviour
                                                # rather than hide a real
                                                # collapse behind a crashed
                                                # gate.
                                                should_fire = True
                                            if not should_fire:
                                                continue
                                            emit(
                                                "tool_args_loop",
                                                index=idx,
                                                name=slot.get("name") or "",
                                                tier=tier,
                                                evidence=evidence[:80],
                                                args_chars=len(buf),
                                                checked_at=len(buf),
                                            )
                                            # ``reason`` distinguishes a
                                            # worker self-protective abort
                                            # (loop detector tripped) from
                                            # an operator cancel (SIGTERM
                                            # / ESC, which emits bare
                                            # ``aborted`` from
                                            # ``_install_term_handler``).
                                            # The engine routes reasoned
                                            # aborts through the loop-
                                            # recovery cascade instead of
                                            # hard-cancelling the turn.
                                            emit(
                                                "aborted",
                                                reason="tool_args_loop",
                                            )
                                            return 0
                            if tool_calls_capped:
                                # Runaway batch (DEFAULT_MAX_TOOL_CALLS_PER_
                                # RESPONSE). DeepSeek and vLLM stream calls one
                                # after another, so the kept calls are normally
                                # complete -- but a provider that announces
                                # calls first or interleaves them can leave one
                                # without a name or arguments, and such a call
                                # must never be repaired and run.
                                for i in [i for i, s in tool_calls_acc.items() if not _call_complete(s)]:
                                    capped_incomplete.append(tool_calls_acc.pop(i))
                                _log.warning(
                                    "runaway tool-call batch: stopped the response at "
                                    "call %d; running %d complete call(s)",
                                    max_tool_calls + 1,
                                    len(tool_calls_acc),
                                )
                                # Leaving the loop closes the stream, which stops
                                # the upstream generating the rest.
                                finish_reason = "tool_calls" if tool_calls_acc else "length"
                                terminal_finish_reason_seen = True
                                break
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                            terminal_finish_reason_seen = True
            # A stream the upstream itself closed with the SSE ``[DONE]``
            # sentinel is a clean termination, not a truncation. Some
            # OpenAI-compatible gateways (e.g. the bedrock-access-gateway) end
            # the stream with ``[DONE]`` without ever delivering a chunk that
            # carries a non-null ``choices[0].finish_reason``; without this,
            # the round reads as a dropped stream and the continuation
            # classifier nudges a turn that actually stopped cleanly.
            # Synthesize a terminal ``stop`` when the stream ended on the
            # sentinel but no explicit finish reason was delivered. A real
            # ``length`` / ``tool_calls`` from a chunk wins, since we only fill
            # in while ``finish_reason`` is still ``None``.
            if saw_done_sentinel and not responses_mode and finish_reason is None:
                finish_reason = "stop"
                terminal_finish_reason_seen = True
            if split is not None and split.quoted_closes:
                emit("reasoning_quoted_close", count=split.quoted_closes)
            # The model stopped while still reasoning -- after a quoted close
            # or straight from the reasoning -- so nothing reached the answer
            # channel. Resample the round.
            if (
                split is not None
                and split.unclosed
                and not tool_calls_acc
                and not content_buf
                and finish_reason == "stop"
                and cut_resamples < _REASONING_CUT_RESAMPLES
                and _attempt + 1 < max_attempts
            ):
                cut_resamples += 1
                _log.warning(
                    "reasoning never closed; resampling the round (%d/%d)",
                    cut_resamples,
                    _REASONING_CUT_RESAMPLES,
                )
                emit("reasoning_cut_resample", attempt=cut_resamples)
                current_reasoning = "".join(attempt_reasoning)
                if not emitted_reasoning_prefix.startswith(current_reasoning):
                    emitted_reasoning_prefix = current_reasoning
                finish_reason = None
                terminal_finish_reason_seen = False
                raise _RetryRound()
            if split is not None and saw_token_ids:
                # The newline rule is a heuristic. On the last attempt (or
                # a length/truncated finish that must not be resampled),
                # return the upstream answer held behind an ambiguous close.
                # These bytes were never emitted as thinking or content.
                _, fallback_content = split.finish()
                if fallback_content:
                    thinking, content = splitter.feed(fallback_content)
                    if thinking:
                        reasoning_chars += len(thinking)
                        emit("thinking", text=thinking)
                    if content:
                        content_chars += len(content)
                        content_buf.append(content)
                        visible, content_visible_chars = visible_content_delta(
                            "".join(content_buf),
                            content_visible_chars,
                        )
                        if visible:
                            emit("content", text=visible)
            if responses_mode and responses_state is not None:
                tool_calls_acc = responses_state.tool_calls_acc
                content_buf = responses_state.content_buf
                reasoning_chars = responses_state.reasoning_chars
                content_chars = responses_state.content_chars
                t_first_token = responses_state.t_first_token
                finish_reason = responses_state.finish_reason
                terminal_finish_reason_seen = responses_state.terminal_finish_reason_seen
                prompt_tokens = responses_state.prompt_tokens
                completion_tokens = responses_state.completion_tokens
                prompt_tokens_cached = responses_state.prompt_tokens_cached
                reasoning_tokens_usage = responses_state.reasoning_tokens_usage
                if finish_reason is None:
                    finish_reason = "tool_calls" if tool_calls_acc else "stop"
            thinking, content = splitter.flush()
            if thinking:
                reasoning_chars += len(thinking)
                emit("thinking", text=thinking)
            if content:
                content_chars += len(content)
                content_buf.append(content)
                visible, content_visible_chars = visible_content_delta(
                    "".join(content_buf),
                    content_visible_chars,
                )
                if visible:
                    emit("content", text=visible)
            # The stream ended. Decide what the round's markup actually was,
            # then release everything that is NOT the call. A marker in the
            # middle of prose is a QUOTATION (an example, a regression fixture,
            # a docstring) and must be shown and kept; only a block at the tail
            # is the call the model meant to make. Never in responses mode —
            # that path emits content itself and does not run the suppression,
            # so releasing here would duplicate the whole answer.
            _round_text = "".join(content_buf)
            _tail_block = leaked_call_tail(_round_text)
            _tail_start = _round_text.rindex(_tail_block) if _tail_block else len(_round_text)
            if not responses_mode and content_visible_chars < _tail_start:
                emit("content", text=_round_text[content_visible_chars:_tail_start])
                content_visible_chars = _tail_start
            flush_content_chars = len(content) if content else 0

            # Reasoning is an Activity-only channel for every provider. Never
            # synthesize visible content from thinking: a reasoning-only stop
            # must remain contentless so the runtime's continuation policy can
            # recover it, while structured tool calls proceed independently.
            # The metric remains for schema compatibility and is invariantly false.
            reasoning_promoted_to_content = False

            # Leaked tool-call recovery. A model that drifted to XML tool-call
            # serialization streams the markup as ``content`` (see
            # ``recover_leaked_tool_calls``). Only the block at the TAIL counts:
            # the tool protocol puts the call last, so markup anywhere else is
            # a quotation and must neither be dispatched nor be allowed to
            # override a real call's arguments. Two cases at the tail:
            #
            # 1. No structured call at all — reconstruct it and dispatch, so a
            #    silent premature stop becomes a real tool round.
            # 2. A structured call AND markup — the upstream returned a PARTIAL
            #    call (measured: ``{"action": "score"}`` for a call whose markup
            #    carried three arguments). The recovered arguments are richer,
            #    so prefer them when they are a strict superset; otherwise keep
            #    the structured call.
            #
            # Cleanup is NOT gated on the absence of a structured call: the
            # markup must never persist into history or be re-fed to the model,
            # in either case. The recorded body keeps everything before the
            # call — including a quoted example — and drops only the call.
            if _tail_block:
                recovered = recover_leaked_tool_calls(_tail_block)
                recovered_names = [c["name"] for c in recovered]
                preferred: list[str] = []
                if recovered and not tool_calls_acc and finish_reason in ("stop", None):
                    for i, call in enumerate(recovered):
                        tool_calls_acc[i] = {
                            "id": f"recovered_{i}",
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        }
                    finish_reason = "tool_calls"
                elif recovered and tool_calls_acc:
                    preferred = prefer_recovered_arguments(tool_calls_acc, recovered)
                # The raw markup was already streamed as ``content`` and is in
                # ``content_buf``; ship the cleaned prose so the runtime can
                # drop the markup from the round's recorded content.
                emit(
                    "leaked_tool_call_recovered",
                    count=len(recovered),
                    names=recovered_names,
                    preferred=preferred,
                    cleaned_content=_round_text[:_tail_start].rstrip(),
                )
            elif not any(marker in _round_text for marker in _LEAK_STRUCTURAL):
                # No XML wrapper anywhere: the only remaining leak shape is a
                # raw JSON tool call (Qwen3.6 fallback), which the recovery
                # parser detects directly.
                recovered = recover_leaked_tool_calls(_round_text)
                if recovered:
                    if not tool_calls_acc and finish_reason in ("stop", None):
                        for i, call in enumerate(recovered):
                            tool_calls_acc[i] = {
                                "id": f"recovered_{i}",
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            }
                        finish_reason = "tool_calls"
                    emit(
                        "leaked_tool_call_recovered",
                        count=len(recovered),
                        names=[c["name"] for c in recovered],
                        preferred=[],
                        cleaned_content=strip_leaked_tool_markup(_round_text),
                    )
            t_end = time.monotonic()

            # If the model wants to call tools, parse each call's accumulated
            # `arguments` JSON and emit one `tool_calls` event so the agent
            # runtime can drive the dispatch loop. We don't validate the args
            # here — that's the runner's job.
            #
            # If `finish_reason == "tool_calls"` but the accumulator is empty
            # (some upstreams emit this when they truncate mid-tool-args),
            # downgrade to "stop" rather than emitting an empty `tool_calls`
            # event — the dispatch loop expects at least one call when the
            # finish_reason claims tool_calls, and an empty list there causes
            # a no-op or a confused dispatch.
            if not tool_calls_acc and finish_reason == "tool_calls":
                finish_reason = "stop"
            # Whether a *usable* tool call actually reached the runtime this turn.
            # Only the emission path below (``tool_calls_acc and
            # finish_reason == "tool_calls"``) dispatches a call. A partial /
            # mid-args-truncated accumulator that never satisfied that condition
            # (e.g. finish_reason downgraded to "stop"/"length" while a fragment
            # sits in ``tool_calls_acc``) is NOT a usable call and must not be
            # counted as one — see the empty-turn check below.
            tool_call_emitted = False
            if tool_calls_acc and finish_reason == "tool_calls":
                from sharedagents_model.json_repair import repair_truncated_json

                calls_out: list[dict[str, Any]] = []
                for idx in sorted(tool_calls_acc):
                    slot = tool_calls_acc[idx]
                    raw = slot["arguments"] or "{}"
                    # Streaming truncation: vLLM's gemma4 tool-call parser
                    # has been observed cutting long `arguments` (e.g. a
                    # multi-item TodoWrite) mid-string, producing JSON that
                    # `json.loads` can't parse. Try a structural repair
                    # (close strings + scopes, drop dangling key tails)
                    # before giving up — recovers the model's intent in
                    # the common case so the call still dispatches instead
                    # of bouncing the model with a misleading arg-shape
                    # error.
                    try:
                        parsed: Any = repair_truncated_json(raw)
                        if not isinstance(parsed, dict):
                            parsed = {"_arguments": parsed}
                    except json.JSONDecodeError:
                        parsed = {"_arguments_unparsed": raw}
                    calls_out.append(
                        {
                            "id": slot["id"] or f"call_{idx}",
                            "name": slot["name"],
                            "arguments": parsed,
                            "raw_arguments": raw,
                        }
                    )
                emit("tool_calls", calls=calls_out)
                tool_call_emitted = True
            # A call the model was still writing when the output cap hit
            # (finish_reason=length) is never dispatched -- the fragment is
            # dropped above. Report it, so the runtime can tell the model its
            # call did not run instead of mistaking the round for a cut-off
            # sentence or thinking starvation (live: one `write`
            # re-sent and cut off at 16,384 tokens, five rounds in a row).
            truncated_tool_calls = [
                {"name": slot["name"], "args_chars": len(slot["arguments"] or "")}
                for slot in [
                    *capped_incomplete,
                    *(slot for _, slot in sorted(tool_calls_acc.items()) if not tool_call_emitted),
                ]
            ]

            # Empty-turn invariant: "the model shall either
            # call a tool OR present output; failing to do so is considered
            # empty." A turn that produced NEITHER a usable tool call NOR any
            # visible output content is a degenerate turn and must be SURFACED —
            # not silently finished as a clean ``stop`` (which the finish_reason
            # downgrade above and the reasoning/leaked-recovery paths can produce).
            # Three cases, distinguished here:
            #   1. Output content present (non-blank visible text) -> not empty.
            #   2. A usable tool call was emitted (incl. a recovered leaked call,
            #      which lands in ``tool_calls_acc`` and dispatches above) -> not
            #      empty.
            #   3. NEITHER of the above -> empty. This also covers the
            #      partial/mid-args-truncated tool call: the accumulator held a
            #      fragment but no call was ever emitted, so it counts as neither.
            # "Output content" is whitespace-sensitive: blank/whitespace-only
            # assistant text is treated as no output, per the owner's reframe.
            # We do NOT raise or alter the normal stop-with-content / tool_calls
            # paths — only observe and flag.
            has_output_content = bool("".join(content_buf).strip())
            empty_turn = not tool_call_emitted and not has_output_content
            if empty_turn:
                _log.warning(
                    "empty turn: model produced neither a tool call nor output "
                    "content (finish_reason=%r, tool_calls_acc_slots=%d, "
                    "content_chars=%d, reasoning_chars=%d)",
                    finish_reason,
                    len(tool_calls_acc),
                    content_chars,
                    reasoning_chars,
                )

            # Build metrics payload.
            # Naming:
            #   accept_ms  = request acceptance — time from POST to upstream
            #                returning HTTP response headers (ASGI flushes as
            #                soon as the request is queued; prefill not done
            #                yet). Mostly reflects network + upstream queue.
            #   ttft_ms    = time-to-first-token — when the first reasoning
            #                or content delta hits us. Includes accept +
            #                prefill + first-decode-and-emit.
            #   prefill_ms = ttft_ms − accept_ms — the dominant component on
            #                long prompts (uncached-tail prefill plus the
            #                O(N) attention cost on the first decoded token).
            wt_ms = (t_end - t_start) * 1000.0
            accept_ms = (t_headers - t_start) * 1000.0 if t_headers is not None else None
            ttft_ms = (t_first_token - t_start) * 1000.0 if t_first_token is not None else None
            prefill_ms = (
                (t_first_token - t_attempt_headers) * 1000.0
                if (t_first_token is not None and t_attempt_headers is not None)
                else None
            )
            # Some upstreams (notably cold-start vLLM) hold the HTTP
            # response open until prefill is done — they return the
            # response headers AND the first token together at the end
            # of prefill. Our naïve decomposition then reads as
            # ``accept_ms ≈ 12_800ms`` and ``prefill_ms ≈ 0ms``, which
            # makes the operator-visible stats line lie about where the
            # latency went and (worse) divides ``prompt_tokens`` by ~0
            # to produce a 100M-tok/s prefill rate. Re-attribute when
            # the split looks pathological: if prefill_ms is near zero
            # but ttft_ms is non-trivial, treat ttft minus a small
            # network floor as the real prefill cost. Bump the floor
            # via $SHAREDAGENTS_PREFILL_FLOOR_MS for high-latency networks.
            try:
                floor_ms = float(os.environ.get("SHAREDAGENTS_PREFILL_FLOOR_MS", "50") or 50)
            except (TypeError, ValueError):
                floor_ms = 50.0
            if prefill_ms is not None and ttft_ms is not None and prefill_ms < floor_ms and ttft_ms > floor_ms * 4:
                prefill_ms = max(0.0, ttft_ms - floor_ms)
                if accept_ms is not None and accept_ms > floor_ms:
                    accept_ms = floor_ms
            gen_seconds = (t_end - t_first_token) if t_first_token is not None else None
            tokens_estimated = completion_tokens is None
            # Tool-call arguments are output too: a capped runaway batch (no
            # usage chunk) is mostly them. Counted as non-reasoning output.
            tool_args_chars = sum(
                len(slot.get("arguments") or "") for slot in [*tool_calls_acc.values(), *capped_incomplete]
            )
            if completion_tokens is None:
                # 1 token ≈ 4 chars of English text — rough but useful when the
                # server didn't return usage.
                completion_tokens = max(
                    1,
                    (reasoning_chars + content_chars + tool_args_chars) // 4,
                )
            # Decode throughput. `gen_seconds` (t_end − t_first_token) is the
            # streaming decode window and the right denominator for a truly
            # streaming upstream. But buffered / non-streaming upstreams
            # (notably OpenRouter's free tier and reasoning models that hold the
            # body until done) deliver the whole completion in a final burst:
            # t_first_token ≈ t_end, so gen_seconds collapses toward 0 and a
            # naïve completion/gen_seconds reports a nonsensical multi-thousand
            # tok/s. When the decode window is implausibly short for the token
            # count, fall back to the post-accept window (prefill+decode) for a
            # bounded, meaningful effective rate.
            decode_seconds = gen_seconds
            if (
                gen_seconds is not None
                and gen_seconds < 0.5
                and completion_tokens
                and completion_tokens > 20
                and accept_ms is not None
                and (wt_ms - accept_ms) > 0
            ):
                decode_seconds = (wt_ms - accept_ms) / 1000.0
            tps = (completion_tokens / decode_seconds) if (decode_seconds and decode_seconds > 0) else None
            # Prefill TPS is the prompt-side rate: how fast the upstream
            # ingested + KV-attended the (uncached tail of the) prompt. Only
            # meaningful when we have both a prompt-token count and a
            # prefill duration. Cached blocks aren't actually prefilled, so
            # exclude them from the numerator when the upstream returned a
            # ``prompt_tokens_cached`` count.
            prefill_seconds = (prefill_ms / 1000.0) if prefill_ms is not None else None
            prefill_tokens = (prompt_tokens - (prompt_tokens_cached or 0)) if prompt_tokens is not None else None
            prefill_tps = (
                (prefill_tokens / prefill_seconds)
                if (prefill_seconds and prefill_seconds > 0 and prefill_tokens)
                else None
            )
            total_tokens = (
                ((prompt_tokens or 0) + (completion_tokens or 0))
                if (prompt_tokens is not None or completion_tokens is not None)
                else None
            )

            cache_hit_pct = (
                round(100.0 * prompt_tokens_cached / prompt_tokens, 1)
                if prompt_tokens and prompt_tokens_cached is not None
                else None
            )
            # Per-call output-token split. Prefer the upstream's authoritative
            # `reasoning_tokens` when present; otherwise estimate by char
            # ratio so the TUI's stats line still has a "thinking vs result"
            # breakdown for upstreams that don't report the split.
            reasoning_tokens: int | None
            content_tokens: int | None
            if reasoning_tokens_usage is not None and completion_tokens is not None:
                reasoning_tokens = max(0, min(reasoning_tokens_usage, completion_tokens))
                content_tokens = max(0, completion_tokens - reasoning_tokens)
            elif completion_tokens is not None:
                total_chars = reasoning_chars + content_chars + tool_args_chars
                if total_chars > 0:
                    r = int(round(completion_tokens * (reasoning_chars / total_chars)))
                    reasoning_tokens = max(0, min(r, completion_tokens))
                    content_tokens = max(0, completion_tokens - reasoning_tokens)
                else:
                    reasoning_tokens = 0
                    content_tokens = completion_tokens
            else:
                reasoning_tokens = None
                content_tokens = None
            prompt_tokens_fresh = (
                max(0, prompt_tokens - (prompt_tokens_cached or 0)) if prompt_tokens is not None else None
            )
            # Token-loop ratio guard. When the model collapses into a token
            # loop (gemma-4-31b-nvfp4-mtp does this on long planning turns)
            # the upstream burns through completion tokens but the streamed
            # text barely grows: we've measured 3022 tokens for 1123 chars
            # (0.37 chars/token) — well below the ~3-4 chars/token English
            # baseline. The streaming detectors inside the round handle the
            # in-stream case; this is the post-stream backstop for round
            # types where streaming detection isn't wired (silent tool-args
            # rounds where `delta.content` was empty the whole time).
            # Per-model override (``Model.loop_chars_per_token_min``) wins
            # over the env default — quantization-fragile families like
            # gemma-4 NVFP4 tighten this to catch collapses one round
            # earlier than the BF16 baseline would.
            # Ratio floor: a hard lower bound for chars-per-completion-token
            # below which we suspect token-collapse. Empirical evidence:
            # real collapses (gemma-4 NVFP4 mtp) land at 0.03-0.43
            # chars/token; legitimate thinking-mode turns on Qwen3.6 / Deepseek
            # land at 1.0-1.2 chars/token (lots of compact reasoning tokens
            # followed by a clean tool call). The prior default of 1.5 sat
            # ABOVE the legitimate-reasoning floor and killed valid turns —
            # as LOOP_DETECTED ratio entries showed. 0.5 sits
            # well above any observed collapse but well below any observed
            # legitimate reasoning turn. Per-model override still wins.
            ratio_floor = (
                float(mdl.loop_chars_per_token_min)
                if getattr(mdl, "loop_chars_per_token_min", None) is not None
                else None
            )
            if ratio_floor is None:
                try:
                    ratio_floor = float(os.environ.get("SHAREDAGENTS_LOOP_CHARS_PER_TOKEN_MIN", "0.5") or 0.5)
                except (TypeError, ValueError):
                    ratio_floor = 0.5
            try:
                ratio_min_tokens = int(os.environ.get("SHAREDAGENTS_LOOP_RATIO_MIN_TOKENS", "500") or 500)
            except (TypeError, ValueError):
                ratio_min_tokens = 500
            emitted_chars = reasoning_chars + content_chars
            # Skip the ratio guard when the upstream gave a clean termination
            # signal ("stop" = end-of-turn, "tool_calls" = the model wants to
            # invoke a tool now). A clean finish means the model chose to end
            # this round on purpose; the chars/token ratio reflects how dense
            # the reasoning was, not a runaway loop. The guard still fires on
            # ``length`` (hit max_tokens — could be a loop that was capped),
            # ``None`` (upstream didn't say), or any other irregular state.
            _clean_finish = finish_reason in ("stop", "tool_calls")
            token_loop_suspected = (
                not _clean_finish
                and completion_tokens is not None
                and completion_tokens >= ratio_min_tokens
                and not tokens_estimated
                and emitted_chars / max(1, completion_tokens) < ratio_floor
            )
            emit(
                "metrics",
                wt_ms=round(wt_ms, 1),
                accept_ms=round(accept_ms, 1) if accept_ms is not None else None,
                ttft_ms=round(ttft_ms, 1) if ttft_ms is not None else None,
                prefill_ms=round(prefill_ms, 1) if prefill_ms is not None else None,
                prefill_tps=round(prefill_tps, 2) if prefill_tps is not None else None,
                tps=round(tps, 2) if tps is not None else None,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                content_tokens=content_tokens,
                prompt_tokens=prompt_tokens,
                prompt_tokens_cached=prompt_tokens_cached,
                prompt_tokens_fresh=prompt_tokens_fresh,
                total_tokens=total_tokens,
                cache_hit_pct=cache_hit_pct,
                reasoning_chars=reasoning_chars,
                content_chars=content_chars,
                flush_content_chars=flush_content_chars,
                tokens_estimated=tokens_estimated,
                model=model,  # which upstream model served this turn
                endpoint=endpoint,
                provider=catalog_provider,  # catalog provider the parent selected
                finish_reason=finish_reason,
                terminal_finish_reason_seen=terminal_finish_reason_seen,
                json_decode_failures=json_decode_failures,
                reasoning_promoted_to_content=reasoning_promoted_to_content,
                token_loop_suspected=token_loop_suspected,
                # Empty/degenerate turn marker: True when the
                # turn produced neither a usable tool call nor visible output.
                empty_turn=empty_turn,
                truncated_tool_calls=truncated_tool_calls,
                tool_calls_capped=tool_calls_capped,
                chars_per_token=(
                    round(emitted_chars / completion_tokens, 3) if completion_tokens and not tokens_estimated else None
                ),
            )
            emit("done")
            return 0
        except httpx.HTTPError as e:
            # Retry only when the failure hit BEFORE any response
            # header — nothing was streamed to the parent yet, so a
            # retry cannot duplicate output. Once headers are in,
            # bytes may already have reached the user.
            can_retry_replay = (
                reasoning_replay_retry_enabled
                and not retried_reasoning_replay
                and isinstance(e, httpx.HTTPStatusError)
                and e.response.status_code == 400
                and _attempt + 1 < max_attempts
                and not headers_received
            )
            rejected = _unsupported_message_field(400, str(e)) if can_retry_replay else None
            if rejected is not None:
                _record_rejected_field(endpoint, rejected)
                _drop_field_from_payload(rejected)
                retried_reasoning_replay = True
                emit(
                    "reasoning_replay_retry",
                    field=rejected,
                    endpoint=endpoint,
                )
                continue
            if can_retry_replay and _is_missing_reasoning_content_error(400, str(e)):
                _ensure_reasoning_in_payload()
                retried_reasoning_replay = True
                emit(
                    "reasoning_replay_retry",
                    field="reasoning_content_required",
                    endpoint=endpoint,
                )
                continue
            emit("error", message=f"http error: {e}")
            return 1
        except _RetryRound:
            # The replayed reasoning fields have been reconciled with the
            # upstream error, or a cut reasoning block was dropped; re-issue
            # the round. `continue` advances the attempt loop, so the retry is
            # bounded by `max_attempts`.
            continue
        except Exception as e:  # noqa: BLE001
            emit("error", message=f"{type(e).__name__}: {e}")
            return 1
        finally:
            if dump_fh is not None:
                try:
                    dump_fh.close()
                except OSError:
                    pass
                # A retried attempt must not write to the closed file.
                dump_fh = None

    # Attempts exhausted without a response we could stream.
    emit("error", message="upstream request failed after retry")
    return 1


if __name__ == "__main__":
    sys.exit(main())
