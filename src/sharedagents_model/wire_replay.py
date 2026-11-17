"""Reasoning-replay wire compatibility — which endpoints accept what.

The engine replays an OpenAI-shaped message history into every worker round,
and ``round_build._build_assistant_round_msg`` decorates each assistant
message that carried thinking with ``reasoning_content``. That field is a
DeepSeek extension: ``deepseek-reasoner`` / ``deepseek-v4-*`` HTTP-400 a tool
follow-up when it is MISSING (*"reasoning_content in the thinking mode must
be passed back to the API"*), so it has to round-trip.

"OpenAI-compatible" is not the same as "tolerant of unknown fields". Several
hosted endpoints validate the message schema strictly and answer a replayed
reasoning field with a hard 400 — live 2026-09-17::

    api.cerebras.ai  HTTP 400 {"message":"messages.4.assistant.reasoning_content:
                     property 'messages.4.assistant.reasoning_content' is
                     unsupported","code":"wrong_api_format"}
    api.groq.com     HTTP 400 {"error":{"message":"'messages.1' : for
                     'role:assistant' the following must be satisfied[(
                     'messages.1' : property 'reasoning_content' is
                     unsupported)]"}}

Both reject ``reasoning_content`` / ``thinking`` / ``thinking_content`` and
accept a plain assistant message. Verified tolerant the same day:
``api.deepseek.com``, ``api.moonshot.ai``, ``api.x.ai``, ``openrouter.ai``
(the free local vLLM/SGLang servers ignore unknown fields by construction).

This module is the ONE place that knows that split. It has two layers:

* :data:`STRICT_REASONING_REPLAY_HOSTS` — the evidence-backed registry, so a
  known provider is handled on the FIRST round instead of erroring once.
* :func:`record_rejected_field` / :func:`rejected_assistant_fields` — a
  break-glass learner. An endpoint that is not in the registry but rejects a
  reasoning field with the standard validation-error shape gets that field
  stripped from then on, for this process and every later one (the cache is a
  small JSON file under the runtime root, the same idea as the model-list
  cache). A new strict provider therefore costs one rejected request ever,
  not a broken turn — and an unknown *field* can never be learned, so this
  cannot silently erode the payload.

Fail-soft by design: nothing here raises on the turn path. An unreadable
cache behaves like an empty one.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sharedagents_core.kernel.injection_guard import defang_control_tokens
from sharedagents_model.bedrock_config import BEDROCK_REASONING_FIELD, is_bedrock_row

#: Assistant-message fields that carry replayed chain-of-thought. Only these
#: are ever stripped or learned — anything else is left to the caller (a
#: provider complaining about, say, ``tool_calls`` must not silently lose
#: them).
REASONING_REPLAY_FIELDS: tuple[str, ...] = (
    "reasoning_content",
    "thinking",
    "thinking_content",
)

#: Hosts verified to reject a replayed reasoning field with HTTP 400.
#: Keyed on the endpoint hostname (exact match, lowercase). Evidence:
#:
#: * ``api.cerebras.ai`` — 2026-09-17 live probe, ``qwen-3.8-27b``:
#:   ``property 'messages.1.assistant.reasoning_content' is unsupported``.
#: * ``api.groq.com`` — 2026-09-17 live probe, ``qwen/qwen3.8-27b``:
#:   ``property 'reasoning_content' is unsupported``.
#:
#: Local vLLM / SGLang are NOT here: they ignore unknown fields, and the
#: local path is the one the field was designed for.
STRICT_REASONING_REPLAY_HOSTS: frozenset[str] = frozenset(
    {
        "api.cerebras.ai",
        "api.groq.com",
    }
)

#: Hosts verified to require reasoning_content on assistant messages in thinking mode.
REQUIRED_REASONING_REPLAY_HOSTS: frozenset[str] = frozenset(
    {
        "api.deepseek.com",
    }
)

#: Cache file name under the sharedagents runtime root.
_CACHE_FILENAME = "wire_replay_support.json"
#: Forget a learned rejection after this long. Long enough to cover a session
#: and then some; short enough that a provider fixing its schema is picked up
#: without a manual cache delete.
_LEARNED_TTL_S = 30 * 24 * 3600.0

_cache_lock = threading.Lock()
_learned: dict[str, dict[str, Any]] | None = None
_learned_stamp: float | None = None


def clear_learning_cache() -> None:
    """Drop the in-process snapshot (test / reload seam)."""
    global _learned, _learned_stamp
    with _cache_lock:
        _learned = None
        _learned_stamp = None


#: Backwards-compatible alias — the name the first tests used.
reset_learning_cache = clear_learning_cache


def _root() -> Path:
    try:
        from sharedagents_core.kernel.layout import sharedagents_root

        return Path(sharedagents_root())
    except Exception:  # noqa: BLE001 — a missing runtime root must not break the wire
        return Path.home() / ".sharedagents"


def learning_cache_path() -> Path:
    """Where learned rejections live (``<sharedagents root>/wire_replay_support.json``)."""
    return _root() / _CACHE_FILENAME


def endpoint_host(endpoint: Any) -> str:
    """Lowercase ``host[:port]`` of an endpoint URL ("" when there isn't one).

    The port is kept: the LEARNED cache is keyed on it, so a strict gateway on
    :8443 and a tolerant one on :443 of the same name never share a verdict.
    """
    raw = str(endpoint or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    return (parsed.netloc or "").lower()


def _hostname_of(host: str) -> str:
    """Strip a trailing ``:port`` from :func:`endpoint_host` output."""
    name = str(host or "")
    if name.startswith("["):  # IPv6 literal — [::1]:8000
        return name.partition("]")[0] + "]"
    return name.partition(":")[0]


def _endpoint_from_row(row: Mapping[str, Any] | None) -> str:
    payload = row if isinstance(row, Mapping) else {}
    return str(payload.get("baseUrl") or payload.get("base_url") or payload.get("endpoint") or "").strip()


def _row_identity(row: Mapping[str, Any] | None) -> str:
    """``provider``/host text used for substring matching (no model id).

    Kept in step with the identity rule the effort layer uses — provider
    label + endpoint host — so an operator who wrote ``"provider": "Cerebras"``
    on a proxied base URL still gets the strict-host behaviour.
    """
    payload = row if isinstance(row, Mapping) else {}
    return " ".join(
        (
            str(payload.get("provider") or ""),
            str(payload.get("owned_by") or ""),
            _hostname_of(endpoint_host(_endpoint_from_row(payload))),
        )
    ).lower()


def _load_learned() -> dict[str, dict[str, Any]]:
    """Snapshot of the on-disk learned rejections, re-read when it changes."""
    global _learned, _learned_stamp
    path = learning_cache_path()
    try:
        stamp = path.stat().st_mtime
    except OSError:
        stamp = None
    with _cache_lock:
        if _learned is not None and stamp == _learned_stamp:
            return _learned
        entries: dict[str, dict[str, Any]] = {}
        if stamp is not None:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                hosts = raw.get("hosts") if isinstance(raw, Mapping) else None
                if isinstance(hosts, Mapping):
                    now = time.time()
                    for host, fields in hosts.items():
                        if not isinstance(fields, Mapping):
                            continue
                        fresh = [
                            str(name)
                            for name, seen in fields.items()
                            if name in REASONING_REPLAY_FIELDS
                            and isinstance(seen, (int, float))
                            and (now - float(seen)) < _LEARNED_TTL_S
                        ]
                        if fresh:
                            entries[str(host).lower()] = {name: fields[name] for name in fresh}
            except (OSError, ValueError, TypeError):
                # Corrupt / half-written cache reads as "nothing learned".
                entries = {}
        _learned = entries
        _learned_stamp = stamp
        return _learned


def record_rejected_field(endpoint: Any, field: Any) -> bool:
    """Remember that *endpoint* rejected assistant field *field*.

    Returns True when the rejection was recorded (or already known), False
    when the field is not a reasoning-replay field — the caller only ever
    passes what :func:`unsupported_message_field` parsed out of a real
    upstream error, and this is the guard that keeps the learner narrow.
    """
    name = str(field or "").strip()
    host = endpoint_host(endpoint)
    if not host or name not in REASONING_REPLAY_FIELDS:
        return False
    path = learning_cache_path()
    try:
        with _cache_lock:
            existing: dict[str, dict[str, Any]] = {}
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                hosts = raw.get("hosts") if isinstance(raw, Mapping) else None
                if isinstance(hosts, Mapping):
                    for key, fields in hosts.items():
                        if isinstance(fields, Mapping):
                            existing[str(key).lower()] = dict(fields)
            except (OSError, ValueError, TypeError):
                existing = {}
            existing.setdefault(host, {})[name] = time.time()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps({"hosts": existing}), encoding="utf-8")
            tmp.replace(path)
    except OSError:
        return False
    global _learned, _learned_stamp
    with _cache_lock:
        _learned = None
        _learned_stamp = None
    return True


def rejected_assistant_fields(row: Mapping[str, Any] | None) -> frozenset[str]:
    """Reasoning fields this row's endpoint is known to reject.

    Empty for an unknown or tolerant endpoint — the fail-soft default is to
    send the field, so a provider that does not appear here keeps the
    historical behaviour unless it has actually rejected something.
    """
    payload = row if isinstance(row, Mapping) else {}
    host = endpoint_host(_endpoint_from_row(payload))
    if not host:
        return frozenset()
    reject: set[str] = set()
    if host in STRICT_REASONING_REPLAY_HOSTS:
        reject.update(REASONING_REPLAY_FIELDS)
    else:
        identity = _row_identity(payload)
        for known in STRICT_REASONING_REPLAY_HOSTS:
            if known in identity:
                reject.update(REASONING_REPLAY_FIELDS)
                break
    learned = _load_learned().get(host)
    if learned:
        reject.update(learned)
    return frozenset(reject)


def strip_unsupported_fields(
    messages: Sequence[Any],
    fields: frozenset[str],
) -> list[Any]:
    """Return *messages* with *fields* removed from every message.

    The input list and its entries are never mutated: the engine owns the
    history across rounds, and the same list can be re-sent to a different
    deployment after a failover.
    """
    if not fields:
        return list(messages)
    out: list[Any] = []
    for msg in messages:
        if isinstance(msg, Mapping):
            cleaned = {k: v for k, v in msg.items() if k not in fields}
            out.append(cleaned if len(cleaned) != len(msg) else msg)
        else:
            out.append(msg)
    return out


def requires_reasoning_content(row: Mapping[str, Any] | None) -> bool:
    """True if this endpoint requires reasoning_content on assistant messages."""
    payload = row if isinstance(row, Mapping) else {}
    host = endpoint_host(_endpoint_from_row(payload))
    if host and host in REQUIRED_REASONING_REPLAY_HOSTS:
        return True
    identity = _row_identity(payload)
    for known in REQUIRED_REASONING_REPLAY_HOSTS:
        if known in identity:
            return True
    return "deepseek" in identity


def ensure_required_reasoning_fields(
    messages: Sequence[Any],
) -> list[Any]:
    """Ensure assistant messages have reasoning_content (empty string if absent)."""
    out: list[Any] = []
    for msg in messages:
        if isinstance(msg, Mapping) and msg.get("role") == "assistant":
            if "reasoning_content" not in msg or msg.get("reasoning_content") is None:
                cleaned = dict(msg)
                cleaned["reasoning_content"] = ""
                out.append(cleaned)
                continue
        out.append(msg)
    return out


_DEFANG_TEXT_FIELDS = ("content", "reasoning_content", "reasoning")


def _defang_value(value: Any) -> Any:
    if isinstance(value, str):
        return defang_control_tokens(value)
    if isinstance(value, list):
        out = []
        for part in value:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                part = {**part, "text": defang_control_tokens(part["text"])}
            out.append(part)
        return out
    return value


def defang_message_control_tokens(messages: Sequence[Any]) -> list[Any]:
    """Copy of *messages* with control-token text defanged, system excepted.

    A server that renders the chat template itself (vLLM, SGLang) tokenizes a
    literal ``</think>``, ``<tool_call>`` or ``<|im_start|>`` in ANY message as
    the real token, so text the model reads -- a file, its own earlier tool
    arguments, a pasted prompt -- closes the template's blocks. The model then
    copies the token when it quotes the text, which ends its reasoning mid-
    sentence. System messages are operator-authored and keep their text.
    """
    out: list[Any] = []
    for msg in messages:
        if not isinstance(msg, Mapping) or msg.get("role") == "system":
            out.append(msg)
            continue
        cleaned = dict(msg)
        for field in _DEFANG_TEXT_FIELDS:
            if field in cleaned:
                cleaned[field] = _defang_value(cleaned[field])
        calls = cleaned.get("tool_calls")
        if isinstance(calls, list):
            new_calls = []
            for call in calls:
                fn = call.get("function") if isinstance(call, Mapping) else None
                if isinstance(fn, Mapping) and isinstance(fn.get("arguments"), str):
                    call = {
                        **call,
                        "function": {
                            **fn,
                            "arguments": _defang_value(fn["arguments"]),
                        },
                    }
                new_calls.append(call)
            cleaned["tool_calls"] = new_calls
        out.append(cleaned)
    return out


def sanitize_messages(
    messages: Sequence[Any],
    row: Mapping[str, Any] | None,
) -> list[Any]:
    """Wire-ready copy of *messages* for *row*'s endpoint."""
    rejected = rejected_assistant_fields(row)
    if not is_bedrock_row(row):
        # A Bedrock model's opaque reasoning (round_build attaches it for the
        # next Converse round) is provider-private: never on the OpenAI wire,
        # including after a mid-session switch off the Bedrock model.
        rejected = rejected | {BEDROCK_REASONING_FIELD}
    cleaned = strip_unsupported_fields(messages, rejected)
    # Actual endpoint rejections take precedence over provider identity,
    # including DeepSeek models served through a strict proxy.
    if "reasoning_content" not in rejected and requires_reasoning_content(row):
        return ensure_required_reasoning_fields(cleaned)
    return cleaned


def _body_text(body: Any) -> str:
    if isinstance(body, (bytes, bytearray)):
        return bytes(body).decode("utf-8", "replace")
    if isinstance(body, str):
        return body
    if isinstance(body, Mapping):
        try:
            return json.dumps(body)
        except (TypeError, ValueError):
            return str(body)
    return str(body or "")


def _error_context(body: Any) -> tuple[str, Any]:
    """``(message, code)`` for the two upstream error envelopes.

    OpenAI-compatible hosts answer either ``{"error": {"message": …}}``
    (Groq, OpenRouter, OpenAI) or a flat ``{"message": …}`` (Cerebras, vLLM).
    The worker's own error string prefixes the status, so a quoted-JSON body
    is accepted too.
    """
    text = _body_text(body).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            err = parsed.get("error")
            if isinstance(err, Mapping):
                return str(err.get("message") or ""), err.get("code") or parsed.get("code")
            if parsed.get("message"):
                return str(parsed.get("message") or ""), parsed.get("code")
    return text, None


def unsupported_message_field(status: Any, body: Any) -> str | None:
    """Reasoning field an upstream 400 says it cannot accept, else None.

    Matches the two observed shapes::

        messages.4.assistant.reasoning_content: property '…' is unsupported
        ('messages.1' : property 'reasoning_content' is unsupported)

    Only HTTP 400 and only the reasoning-replay aliases qualify, so a 401 /
    context-length / model-not-found 400 is never turned into a field strip.
    """
    if status != 400:
        return None
    message, _code = _error_context(body)
    low = message.lower()
    if "unsupported" not in low:
        return None
    for field in REASONING_REPLAY_FIELDS:
        if field in low:
            return field
    return None


def is_missing_reasoning_content_error(status: Any, body: Any) -> bool:
    """True if upstream 400 indicates reasoning_content is required/missing.

    Matches DeepSeek/vLLM error shape:
    The `reasoning_content` in the thinking mode must be passed back to the API.
    """
    if status != 400:
        return False
    message, _code = _error_context(body)
    low = message.lower()
    return "reasoning_content" in low and "must be passed back" in low
