"""Round-message assembly + direct-dispatch marker helpers.

Extracted from ``acp_bridge.py`` (behavior-preserving de-monolith).

* ``_build_assistant_round_msg`` — build the OpenAI-shaped assistant
  message (content + tool_calls, plus ``reasoning_content`` round-trip)
  that replays into the next worker round's message history.
* ``_parse_direct_delegate_marker`` + ``DIRECT_DELEGATE_PREFIX`` — the
  ``__SHAREDAGENTS_DELEGATE__:`` sentinel that opts a prompt out of the LLM
  round and dispatches a delegate-class tool directly.
* ``_text_tool_content`` — wrap plain text as a spec ``ToolCallContent``.

Re-exported from ``sharedagents_gateway.acp_bridge`` for import stability —
``local_llm`` imports ``_build_assistant_round_msg`` and gateway_shim /
``local_llm`` import ``_parse_direct_delegate_marker`` LAZILY from that
path; gateway_shim imports ``_text_tool_content``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from sharedagents_model.bedrock_config import BEDROCK_REASONING_FIELD


def _build_assistant_round_msg(
    *,
    content: str,
    thinking: str,
    tool_calls: list[Mapping[str, Any]],
    reasoning_replay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the assistant message that emitted this round's tool_calls.

    The message replays into ``messages`` so the next worker round
    sees the canonical OpenAI-shaped tool-call/tool-result history.

    Reasoning round-trip: attach ``thinking`` as ``reasoning_content``,
    using an empty string when no reasoning was recorded. DeepSeek (`deepseek-reasoner` /
    `deepseek-v4-flash`) HTTP-400s the tool-result follow-up with
    *"reasoning_content in the thinking mode must be passed back to
    the API"* when this field is missing.

    Attaching it here is deliberately UNCONDITIONAL, because this layer has no
    knowledge of the endpoint. The field is a DeepSeek extension, and an
    "OpenAI-compatible" host is not obliged to tolerate unknown message
    properties: building it here and reconciling it at the wire is what lets
    both audiences be served. Observed, an unconditional field was a
    hard 400 on two configured providers::

        api.cerebras.ai  {"message":"messages.N.assistant.reasoning_content:
                         property '…' is unsupported","code":"wrong_api_format"}
        api.groq.com     {"error":{"message":"'messages.N' : for 'role:
                         assistant' … property 'reasoning_content' is
                         unsupported]"}}

    The per-endpoint decision therefore lives in ONE place —
    :mod:`sharedagents_model.wire_replay` — which the gateway request builder
    (:mod:`sharedagents_gateway.worker_request`) and the worker both apply.
    Do NOT add endpoint checks here; add an endpoint to
    ``wire_replay.STRICT_REASONING_REPLAY_HOSTS`` instead.

    Opaque reasoning round-trip: ``reasoning_replay`` is a Bedrock worker's
    encrypted / signed reasoning for THIS round's tool calls (Grok 4.6 and
    GPT-6 return reasoning only as an encrypted blob). It rides as
    ``bedrock_reasoning``, is stripped from every non-Bedrock request by
    ``wire_replay.sanitize_messages``, and the Converse adapter replays it
    only to the model that produced it.
    """
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": c["id"],
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(
                        c.get("arguments") or {},
                        ensure_ascii=False,
                    ),
                },
            }
            for c in tool_calls
        ],
    }
    msg["reasoning_content"] = thinking or ""
    if reasoning_replay:
        msg[BEDROCK_REASONING_FIELD] = dict(reasoning_replay)
    return msg


# Sentinel marker the TUI / Telegram / one-shot CLI prepend on the
# first text block of a session/prompt to opt out of the LLM round
# entirely and dispatch a delegate-class tool directly — bypasses the
# local model so /claude, /codex, and /compact don't depend on a
# small model deciding to emit the right tool.
DIRECT_DELEGATE_PREFIX = "__SHAREDAGENTS_DELEGATE__:"


def _parse_direct_delegate_marker(
    blocks: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """If ``blocks`` opens with the direct-dispatch sentinel, parse out
    ``{"tool": str, "arguments": dict}``; otherwise return None.

    Format on the wire (single text block):

        __SHAREDAGENTS_DELEGATE__:<tool_name>:<json-args>

    where ``<json-args>`` is a JSON object passed straight through to
    the tool runner. Any malformed marker (missing tool, bad JSON,
    multi-block prompt) returns None so the caller falls back to the
    regular LLM round — fail-soft, never raise out of a bot path.
    """
    if not blocks:
        return None
    first = blocks[0]
    if not isinstance(first, Mapping) or first.get("type") != "text":
        return None
    text = str(first.get("text") or "")
    if not text.startswith(DIRECT_DELEGATE_PREFIX):
        return None
    rest = text[len(DIRECT_DELEGATE_PREFIX) :]
    head, sep, args_text = rest.partition(":")
    tool_name = head.strip()
    if not tool_name:
        return None
    args_obj: dict[str, Any] = {}
    if sep and args_text:
        try:
            parsed = json.loads(args_text)
            if isinstance(parsed, dict):
                args_obj = parsed
        except (json.JSONDecodeError, TypeError):
            return None
    return {"tool": tool_name, "arguments": args_obj}


def _text_tool_content(text: str) -> list[dict[str, Any]]:
    """Wrap a text string as a single ACP-spec ToolCallContent entry.

    Per ``$defs/ToolCallContent`` (canonical schema): tool_call_update
    ``content`` is a discriminated union over
    ``{"type": "content", "content": <ContentBlock>}`` |
    ``{"type": "diff", ...}`` |
    ``{"type": "terminal", ...}`` — the inner ContentBlock is itself a
    union (text/image/audio/resource_link/resource).

    Earlier emission sites in this module passed the raw ContentBlock
    inline (``[{"type": "text", "text": ...}]``) which trips the SDK's
    pydantic validator (``Input should be 'terminal'`` because the
    discriminator picks the wrong branch). Use this helper everywhere
    a tool result is plain text.
    """
    return [{"type": "content", "content": {"type": "text", "text": text}}]
