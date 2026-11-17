"""Codex ChatGPT OAuth Responses transport.

OpenClaw's ``openai-chatgpt-responses`` path: ChatGPT OAuth bearer →
``POST {chatgpt_base}/backend-api/codex/responses`` → streamed Responses
events. sharedagents already has OAuth read/refresh in model discovery; this
module is the generation half used by the local-LLM worker so jarvis /
build / plan can consume the Codex subscription without ``codex exec``.

Do not send the OAuth token to ``api.openai.com`` — it is a ChatGPT
subscription credential, not a platform API key.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

CODEX_RESPONSES_API = "codex-responses"
CODEX_OAUTH_AUTH = "codex-oauth"
CODEX_ALIAS_PREFIX = "codex:"

# Params the private Codex backend rejects (OpenClaw sanitizes these).
_UNSUPPORTED_PARAMS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "max_tokens",
    "max_completion_tokens",
    "stream_options",
    "metadata",
    "prompt_cache_retention",
    "service_tier",
    "chat_template_kwargs",
    "continue_final_message",
    "add_generation_prompt",
    "enable_thinking",
)


def is_codex_responses_request(request: Mapping[str, Any], endpoint: str = "") -> bool:
    """True when this worker request should use the Codex Responses wire."""
    api = str(request.get("api") or "").strip().lower()
    if api in (CODEX_RESPONSES_API, "openai-chatgpt-responses"):
        return True
    auth = str(request.get("auth") or "").strip().lower()
    if auth == CODEX_OAUTH_AUTH:
        return True
    return is_codex_responses_endpoint(endpoint or str(request.get("endpoint") or ""))


def is_codex_responses_endpoint(url: str) -> bool:
    """True for the ChatGPT Codex Responses base (prod or ``chatgpt_base_url``)."""
    u = (url or "").rstrip("/").lower()
    if not u:
        return False
    return u.endswith("/backend-api/codex") or u.endswith("/backend-api/codex/v1")


def _message_text(msg: Mapping[str, Any]) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _user_input_content(msg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A chat ``user`` message's content as Responses input parts.

    Text parts become ``input_text`` and ``image_url`` parts (a user
    attachment, or an image a tool returned) become ``input_image``, in
    order, so the model sees the image instead of losing it.
    """
    content = msg.get("content")
    if not isinstance(content, list):
        return [{"type": "input_text", "text": _message_text(msg)}]
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            if part:
                parts.append({"type": "input_text", "text": part})
            continue
        if not isinstance(part, Mapping):
            continue
        ptype = part.get("type")
        if ptype in ("text", "input_text"):
            text = str(part.get("text") or "")
            if text:
                parts.append({"type": "input_text", "text": text})
        elif ptype == "image_url":
            inner = part.get("image_url")
            url = inner.get("url") if isinstance(inner, Mapping) else inner
            if isinstance(url, str) and url:
                item: dict[str, Any] = {"type": "input_image", "image_url": url}
                detail = inner.get("detail") if isinstance(inner, Mapping) else None
                if detail:
                    item["detail"] = str(detail)
                parts.append(item)
        elif ptype == "input_image" and part.get("image_url"):
            parts.append(dict(part))
    return parts or [{"type": "input_text", "text": ""}]


def chat_tools_to_responses(tools: list[Any]) -> list[dict[str, Any]]:
    """Chat Completions ``tools[]`` → Responses function tools."""
    out: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, Mapping):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, Mapping):
            params = {"type": "object", "properties": {}}
        out.append(
            {
                "type": "function",
                "name": name,
                "description": str(fn.get("description") or ""),
                "parameters": dict(params),
            }
        )
    return out


def _responses_tool_choice(choice: Any) -> Any:
    if not isinstance(choice, Mapping):
        return choice
    fn = choice.get("function") if isinstance(choice.get("function"), Mapping) else None
    if str(choice.get("type") or "") == "function" and fn:
        return {"type": "function", "name": str(fn.get("name") or "")}
    return choice


def build_codex_responses_payload(chat_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a chat-completions worker payload to Codex Responses JSON."""
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for msg in chat_payload.get("messages") or []:
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role") or "")
        if role == "system":
            text = _message_text(msg).strip()
            if text:
                instructions.append(text)
            continue
        if role == "user":
            input_items.append(
                {
                    "role": "user",
                    "content": _user_input_content(msg),
                }
            )
            continue
        if role == "assistant":
            text = _message_text(msg).strip()
            if text:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, Mapping):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), Mapping) else {}
                args = fn.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {})
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": str(tc.get("id") or ""),
                        "name": str(fn.get("name") or ""),
                        "arguments": args,
                    }
                )
            continue
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(msg.get("tool_call_id") or ""),
                    "output": _message_text(msg) or "(no output)",
                }
            )
    if not input_items:
        input_items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        )
    payload: dict[str, Any] = {
        "model": chat_payload.get("model"),
        "input": input_items,
        "stream": True,
        "store": False,
    }
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    tools = chat_payload.get("tools")
    if tools:
        converted = chat_tools_to_responses(list(tools))
        if converted:
            payload["tools"] = converted
        choice = chat_payload.get("tool_choice")
        if choice:
            payload["tool_choice"] = _responses_tool_choice(choice)
    effort = str(chat_payload.get("reasoning_effort") or "").strip()
    if effort:
        payload["reasoning"] = {
            "effort": effort,
            **({} if effort == "none" else {"summary": "auto"}),
        }
    return payload


class ResponsesStreamState:
    """Mutable accumulators shared with the chat-completions worker finish path."""

    def __init__(self) -> None:
        self.tool_calls_acc: dict[int, dict[str, Any]] = {}
        self.item_index: dict[str, int] = {}
        self.content_buf: list[str] = []
        self.reasoning_chars = 0
        self.content_chars = 0
        self.t_first_token: float | None = None
        self.finish_reason: str | None = None
        self.terminal_finish_reason_seen = False
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.prompt_tokens_cached: int | None = None
        self.reasoning_tokens_usage: int | None = None
        self.error: str | None = None
        self.done = False


def _usage_from_response(state: ResponsesStreamState, usage: Mapping[str, Any]) -> None:
    if not isinstance(usage, Mapping):
        return
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    if inp is None:
        inp = usage.get("prompt_tokens")
    if out is None:
        out = usage.get("completion_tokens")
    if inp is not None:
        try:
            state.prompt_tokens = int(inp)
        except (TypeError, ValueError):
            pass
    if out is not None:
        try:
            state.completion_tokens = int(out)
        except (TypeError, ValueError):
            pass
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    if isinstance(details, Mapping) and details.get("cached_tokens") is not None:
        try:
            state.prompt_tokens_cached = int(details["cached_tokens"])
        except (TypeError, ValueError):
            pass
    out_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    if isinstance(out_details, Mapping) and out_details.get("reasoning_tokens") is not None:
        try:
            state.reasoning_tokens_usage = int(out_details["reasoning_tokens"])
        except (TypeError, ValueError):
            pass


def apply_codex_responses_event(
    event: Mapping[str, Any],
    state: ResponsesStreamState,
    *,
    now: float,
    emit,
) -> None:
    """Fold one Responses SSE event into worker emit/state. Sets ``state.error``."""
    typ = str(event.get("type") or "")
    if typ in ("error", "response.failed"):
        err = event.get("error") or event.get("response") or {}
        msg = err.get("message") or err.get("error") or json.dumps(err) if isinstance(err, Mapping) else str(err or typ)
        state.error = f"upstream Responses error: {msg}"
        return
    if typ == "response.output_text.delta":
        text = event.get("delta") or ""
        if text:
            if state.t_first_token is None:
                state.t_first_token = now
            state.content_chars += len(text)
            state.content_buf.append(text)
            emit("content", text=text)
        return
    if typ in (
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
        "response.reasoning.delta",
    ):
        text = event.get("delta") or ""
        if isinstance(text, Mapping):
            text = text.get("text") or ""
        if text:
            if state.t_first_token is None:
                state.t_first_token = now
            state.reasoning_chars += len(text)
            emit("thinking", text=str(text))
        return
    if typ == "response.output_item.added":
        item = event.get("item") or {}
        if not isinstance(item, Mapping):
            return
        if item.get("type") == "function_call":
            idx = len(state.tool_calls_acc)
            item_id = str(item.get("id") or "")
            if item_id:
                state.item_index[item_id] = idx
            state.tool_calls_acc[idx] = {
                "id": str(item.get("call_id") or ""),
                "name": str(item.get("name") or ""),
                "arguments": str(item.get("arguments") or ""),
            }
            if state.t_first_token is None:
                state.t_first_token = now
        return
    if typ == "response.function_call_arguments.delta":
        item_id = str(event.get("item_id") or "")
        idx = state.item_index.get(item_id)
        if idx is None and state.tool_calls_acc:
            idx = max(state.tool_calls_acc)
        if idx is None:
            return
        delta = event.get("delta") or ""
        if delta:
            state.tool_calls_acc[idx]["arguments"] += str(delta)
        return
    if typ == "response.function_call_arguments.done":
        item_id = str(event.get("item_id") or "")
        idx = state.item_index.get(item_id)
        args = event.get("arguments")
        if idx is not None and isinstance(args, str) and args:
            state.tool_calls_acc[idx]["arguments"] = args
        return
    if typ == "response.completed":
        resp = event.get("response") or {}
        if isinstance(resp, Mapping):
            _usage_from_response(state, resp.get("usage") or {})
            status = str(resp.get("status") or "")
            if status == "incomplete":
                state.finish_reason = "length"
            elif state.tool_calls_acc:
                state.finish_reason = "tool_calls"
            else:
                state.finish_reason = "stop"
        else:
            state.finish_reason = "tool_calls" if state.tool_calls_acc else "stop"
        state.terminal_finish_reason_seen = True
        state.done = True
        return
    if typ == "response.incomplete":
        state.finish_reason = "length"
        state.terminal_finish_reason_seen = True
        state.done = True
