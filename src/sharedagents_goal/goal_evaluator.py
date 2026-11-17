"""Goal evaluator — Claude Code ``/goal`` semantics.

After every turn of a session with a running goal, a small fast model reads
the conversation transcript and judges whether the goal's condition is met.
It has no tools: it answers from transcript evidence only. Verdicts:

  * ``{"ok": true, "reason": ...}``                     → goal met
  * ``{"ok": false, "reason": ...}``                    → keep working; the
    reason is the guidance for the next turn
  * ``{"ok": false, "impossible": true, "reason": ...}`` → goal can never be
    met in this session; it ends as failed

The prompts below are Claude Code's stop-condition evaluator prompts.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Claude Code's evaluator system prompt for stop conditions.
SYSTEM_PROMPT = """\
You are evaluating a stop-condition hook in Claude Code. Read the conversation transcript carefully, then judge whether the user-provided condition is satisfied.

Your response must be a JSON object with one of these shapes:
- {"ok": true, "reason": "<quote evidence from the transcript that satisfies the condition>"}
- {"ok": false, "reason": "<quote what is missing or what blocks the condition>"}
- {"ok": false, "waiting_owner": true, "reason": "<quote the approval or owner-only input the remaining step needs>"}
- {"ok": false, "impossible": true, "reason": "<explain why the condition can never be satisfied>"}

Always include a "reason" field, quoting specific text from the transcript whenever possible. If the transcript does not contain clear evidence that the condition is satisfied, return {"ok": false, "reason": "insufficient evidence in transcript"}.

Use {"ok": false, "waiting_owner": true} when the transcript shows the only remaining step needs the owner: an approval the assistant asked for and has not received, a production or destructive action the owner has not authorized, or a value only the owner can supply. This is NONTERMINAL: the goal stays set, no automatic turn is admitted, and the owner's next message resumes it. Do not use it for ordinary slowness, an unresolved bug, or work the assistant can still do itself.

Only use {"ok": false, "impossible": true} when the condition is genuinely unachievable in this session — for example: the condition is self-contradictory, it depends on a resource or capability that is unavailable, or the assistant has explicitly tried, exhausted reasonable approaches, and stated it cannot be done. Apply your own judgment when deciding this — the assistant claiming the goal is impossible is evidence, not proof; independently confirm the condition is genuinely unachievable rather than deferring to the assistant's self-assessment. Do not use it just because the goal has not been reached yet or because progress is slow. When in doubt, return {"ok": false} without "impossible"."""

#: Claude Code's question to the evaluator, after the transcript.
QUESTION = (
    "Based on the conversation transcript above, has the following stopping "
    "condition been satisfied? Answer based on transcript evidence only.\n\n"
    "Condition: {condition}"
)

_OMITTED_NOTE = (
    "[Transcript truncated to fit the evaluator's context window — {n} "
    "earlier messages omitted. Evaluate the condition against the recent "
    "transcript below; if the required evidence may be in the omitted prefix, "
    'return {{"ok": false, "reason": "insufficient evidence in transcript"}}.]'
)

#: Transcript budget (characters) handed to the evaluator.
DEFAULT_TRANSCRIPT_CHARS = 120_000
_TOOL_ARGS_KEEP = 600
_TOOL_RESULT_KEEP = 2_000
#: Transient evaluator failures are retried this many times.
TRANSIENT_RETRIES = 3


@dataclass
class Verdict:
    ok: bool
    reason: str
    impossible: bool = False
    waiting_owner: bool = False


class EvaluatorError(Exception):
    """The evaluator could not produce a verdict.

    ``unrecoverable`` errors (auth, credit, unknown model, context too long)
    end the goal; transient ones (timeouts, 5xx, rate limits, a bad reply)
    are retried, then pause it."""

    def __init__(self, message: str, *, unrecoverable: bool = False) -> None:
        super().__init__(message)
        self.unrecoverable = unrecoverable


# --- transcript ---------------------------------------------------------------


#: Joins transcript messages. It never occurs inside a message (stripped on
#: the way in), so ``evaluate`` can split the transcript back into chat turns.
MESSAGE_SEP = "\n\x1e\n"

#: Claude Code's verdict tool: the evaluator must answer through it, so the
#: reply is always structured (a free-text "Yes —" cannot be misread).
VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "StructuredOutput",
        "description": (
            "Use this tool to return your verification result. You MUST call "
            "this tool exactly once at the end of your response."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean", "description": "Whether the condition was met"},
                "reason": {"type": "string", "description": "Reason, if the condition was not met"},
                "impossible": {
                    "type": "boolean",
                    "description": ("Whether the condition can never be satisfied (only meaningful when ok is false)"),
                },
                "waiting_owner": {
                    "type": "boolean",
                    "description": (
                        "Whether the only remaining step needs the owner "
                        "(approval, a production/destructive action, a value "
                        "only the owner has). Nonterminal: no automatic turn."
                    ),
                },
            },
            "required": ["ok"],
            "additionalProperties": False,
        },
    },
}


def _clip(text: str, keep: int) -> str:
    text = str(text or "").replace("\x1e", " ")
    return text if len(text) <= keep else text[:keep] + f"… [{len(text) - keep} more chars]"


def _read_events(path: Path) -> Iterator[dict[str, Any]]:
    from sharedagents_session.session_reader import decode_line

    try:
        with open(path, "rb") as fh:
            for raw in fh:
                if not raw.endswith(b"\n"):
                    break  # A writer has not finished appending this event.
                ev = decode_line(raw.rstrip(b"\r\n"))
                if isinstance(ev, dict):
                    yield ev
    except OSError:
        return


def transcript_messages(events: Iterable[dict[str, Any]]) -> tuple[list[str], int]:
    """Fold session events into transcript messages.

    Returns ``(messages, turns_ended)``. Each message is one line-block:
    the user's prompt, the assistant's text, a tool call or a tool result.
    """
    msgs: list[str] = []
    text_buf: list[str] = []
    names: dict[str, str] = {}
    turns_ended = 0
    turn_text: list[str] = []
    turn_text_indices: list[int] = []

    def flush_text() -> None:
        text = "".join(text_buf).strip()
        text_buf.clear()
        if text:
            turn_text_indices.append(len(msgs))
            msgs.append(f"Assistant: {text}")

    for ev in events:
        kind = ev.get("type")
        if kind == "turn_started":
            flush_text()
            turn_text.clear()
            turn_text_indices.clear()
            user = str(ev.get("user") or "").strip()
            if user:
                msgs.append(f"User: {user}")
        elif kind == "text_delta":
            delta = str(ev.get("delta") or "")
            text_buf.append(delta)
            turn_text.append(delta)
        elif kind == "tool_call":
            flush_text()
            name = str(ev.get("name") or "tool")
            names[str(ev.get("call_id") or "")] = name
            args = ev.get("arguments")
            args_text = json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args
            msgs.append(f"Assistant called {name}: {_clip(args_text, _TOOL_ARGS_KEEP)}")
        elif kind == "tool_result":
            flush_text()
            name = names.get(str(ev.get("call_id") or ""), "tool")
            msgs.append(f"Result of {name}: {_clip(ev.get('content'), _TOOL_RESULT_KEEP)}")
        elif kind == "round_ended":
            flush_text()
        elif kind == "turn_ended":
            flush_text()
            final = ev.get("final_content")
            if final is not None and str(final).strip() != "".join(turn_text).strip():
                # Engines can replace streamed drafts at finalization. Keep
                # tools as evidence, but do not judge superseded prose.
                for index in turn_text_indices:
                    msgs[index] = ""
                if str(final).strip():
                    msgs.append(f"Assistant: {str(final).strip()}")
            turn_text.clear()
            turn_text_indices.clear()
            turns_ended += 1
    flush_text()
    return [msg.replace("\x1e", " ") for msg in msgs if msg], turns_ended


def build_transcript(messages: list[str], *, max_chars: int = DEFAULT_TRANSCRIPT_CHARS) -> str:
    """The most recent messages that fit ``max_chars``, with Claude Code's
    omission note when earlier ones were dropped."""
    if not messages or max_chars <= 0:
        return ""
    # The separator never appears inside a message, whoever built the list.
    messages = [str(m).replace("\x1e", " ") for m in messages]

    def omission_note(n: int) -> str:
        if not n:
            return ""
        note = _OMITTED_NOTE.format(n=n) + MESSAGE_SEP
        if len(note) + 80 > max_chars:
            note = f"[Transcript truncated — {n} earlier messages omitted.]" + MESSAGE_SEP
        return note

    kept: list[str] = []
    used = 0
    clipped = ""
    for msg in reversed(messages):
        size = used + len(msg) + (len(MESSAGE_SEP) if kept else 0)
        note = omission_note(len(messages) - len(kept) - 1)
        if size + len(note) > max_chars:
            if not kept:
                clipped = "[Earlier text in latest message omitted.]\n"
                room = max_chars - len(note) - len(clipped)
                if room <= 0:
                    return (note + clipped)[:max_chars]
                kept.append(msg[-room:])
            break
        kept.append(msg)
        used = size
    kept.reverse()
    omitted = len(messages) - len(kept)
    return omission_note(omitted) + clipped + MESSAGE_SEP.join(kept)


def session_transcript(sessions_dir: Path | str, session_id: str) -> tuple[str, int]:
    """``(transcript_text, turns_ended)`` for a session JSONL."""
    path = Path(sessions_dir) / f"{session_id}.jsonl"
    messages, turns = transcript_messages(_read_events(path))
    return build_transcript(messages), turns


# --- model call -----------------------------------------------------------------


def _goal_row(rows: Any) -> dict[str, Any] | None:
    for row in rows or []:
        if (
            isinstance(row, dict)
            and str(row.get("role") or "").lower() == "goal"
            and row.get("baseUrl")
            and row.get("id")
        ):
            return dict(row)
    return None


def evaluator_model_row(cfg: Any) -> dict[str, Any] | None:
    """The evaluator model (Claude Code uses its small fast model):

    1. the agent's ``models[]`` row with ``role == "goal"``;
    2. the global config's ``role == "goal"`` row — CLI-engine agents (claude,
       codex, …) have no HTTP model of their own;
    3. the agent's small/utility model.
    """
    row = _goal_row(getattr(cfg, "models", None))
    if row:
        return row
    try:
        from sharedagents_agent.agent import load_global_config

        row = _goal_row((load_global_config() or {}).get("models"))
    except Exception:  # noqa: BLE001
        row = None
    if row:
        return row
    getter = getattr(cfg, "small_model_row", None)
    if callable(getter):
        try:
            row = getter()
        except Exception:  # noqa: BLE001
            row = None
        if isinstance(row, dict) and row.get("baseUrl") and row.get("id"):
            return dict(row)
    return None


def _load_caller():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "sharedagents_agent" / "agents" / "data" / "teacher_review.py"
    spec = importlib.util.spec_from_file_location("sharedagents_teacher_review", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod._call_teacher


def _classify(exc: BaseException) -> EvaluatorError:
    if isinstance(exc, urllib.error.HTTPError):
        code = int(getattr(exc, "code", 0) or 0)
        try:
            detail = exc.read(4096).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            detail = ""
        msg = f"HTTP {code}{': ' + detail[:300] if detail else ''}"
        lower = detail.lower()
        # Router timeouts ("context deadline exceeded"), token-generation
        # outages and temporary 404 routes are not permanent model errors.
        context_overflow = code == 400 and any(
            w in lower
            for w in (
                "context_length_exceeded",
                "maximum context length",
                "context window exceeded",
                "context length exceeded",
                "context too long",
                "prompt is too long",
                "too many tokens",
                "exceeds the context window",
            )
        )
        unknown_model = code in (400, 404) and any(
            w in lower
            for w in (
                "model_not_found",
                "unknown model",
                "model not found",
                "no such model",
            )
        )
        return EvaluatorError(
            msg,
            unrecoverable=code in (401, 402, 403) or context_overflow or unknown_model,
        )
    if isinstance(exc, (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError)):
        return EvaluatorError(f"{type(exc).__name__}: {exc}")
    return EvaluatorError(f"{type(exc).__name__}: {exc}")


def transcript_chat(transcript: str) -> list[dict[str, str]]:
    """The transcript as chat turns, the way Claude Code hands it to its
    evaluator: the user's prompts as user turns, everything the assistant
    did (text, tool calls, tool results) as assistant turns. Consecutive
    turns of one role are merged so strict chat templates accept them."""
    out: list[dict[str, str]] = []
    for part in transcript.split("\x1e"):
        part = part.strip("\n")
        if not part.strip():
            continue
        if part.startswith("User: "):
            role, text = "user", part[len("User: ") :]
        elif part.startswith("Assistant: "):
            role, text = "assistant", part[len("Assistant: ") :]
        elif part.startswith("["):
            role, text = "user", part  # omission note
        else:
            role, text = "assistant", part
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + text
        else:
            out.append({"role": role, "content": text})
    return out


def _verdict_from_reply(message: dict[str, Any]) -> dict[str, Any]:
    """The ``StructuredOutput`` call's arguments; a JSON object in the text
    reply is accepted as a fallback for servers without tool calling."""
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        if fn.get("name") == "StructuredOutput":
            args = fn.get("arguments")
            return json.loads(args) if isinstance(args, str) else dict(args or {})
    content = message.get("content") or ""
    return _load_caller().__globals__["_extract_verdict"](content)


def _call_with_verdict_tool(
    messages: list[dict[str, str]],
    *,
    endpoint: str,
    model: str,
    api_key: str,
    timeout: float,
    cfg: Any,
    row: dict[str, Any],
) -> dict[str, Any]:
    from sharedagents_model.sampling import apply_sampling, configured_sampling

    endpoint = (endpoint or "").rstrip("/")
    if not endpoint or not model:
        raise EvaluatorError("evaluator endpoint/model not configured", unrecoverable=True)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "tools": [VERDICT_TOOL],
        "tool_choice": "required",
    }
    apply_sampling(payload, configured_sampling(cfg, row, model, purpose="goal_review"))
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        endpoint + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8", "replace"))
    return _verdict_from_reply(obj["choices"][0]["message"])


def evaluate(
    condition: str,
    transcript: str,
    *,
    cfg: Any,
    row: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
    call: Any = None,
) -> Verdict:
    """One evaluator call. Raises :class:`EvaluatorError`."""
    row = row or evaluator_model_row(cfg)
    if not row:
        raise EvaluatorError(
            'no evaluator model: add a models[] row with role "goal" or set small_model',
            unrecoverable=True,
        )
    api_key = os.environ.get(str(row.get("apiKeyEnv") or ""), "") if row.get("apiKeyEnv") else ""
    # Claude Code's shape: system prompt, the transcript as chat turns, then
    # the question; the verdict comes back through the StructuredOutput tool.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += transcript_chat(transcript)
    question = QUESTION.format(condition=condition.strip())
    if len(messages) > 1 and messages[-1]["role"] == "user":
        messages[-1]["content"] += "\n\n" + question
    else:
        messages.append({"role": "user", "content": question})
    caller = call or _call_with_verdict_tool
    try:
        out = caller(
            messages,
            endpoint=str(row.get("baseUrl") or ""),
            model=str(row.get("id") or ""),
            api_key=api_key,
            timeout=timeout_s,
            cfg=cfg,
            row=row,
        )
    except EvaluatorError:
        raise
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise EvaluatorError(f"unreadable verdict: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise _classify(exc) from exc
    if (
        not isinstance(out, dict)
        or not isinstance(out.get("ok"), bool)
        or ("reason" in out and not isinstance(out["reason"], str))
        or ("impossible" in out and not isinstance(out["impossible"], bool))
        or ("waiting_owner" in out and not isinstance(out["waiting_owner"], bool))
    ):
        raise EvaluatorError(f"unreadable verdict: {str(out)[:200]}")
    ok = bool(out["ok"])
    waiting_owner = (not ok) and bool(out.get("waiting_owner"))
    impossible = (not ok) and not waiting_owner and bool(out.get("impossible"))
    # ``reason`` is optional in Claude Code's verdict schema.
    reason = str(out.get("reason") or "").strip() or (
        "condition met"
        if ok
        else "the remaining step needs the owner"
        if waiting_owner
        else "the evaluator judged the condition impossible"
        if impossible
        else "condition not met yet"
    )
    return Verdict(
        ok=ok,
        reason=reason,
        impossible=impossible,
        waiting_owner=waiting_owner,
    )
