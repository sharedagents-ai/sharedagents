"""Native Amazon Bedrock Converse worker — adapter (transport in Task 3).

This is the ONLY module that translates between sharedagents's OpenAI-shaped
worker contract and the AWS Bedrock Converse protocol. The normal worker
``main()`` dispatches here on ``request["api"] == "bedrock-converse"``
BEFORE any OpenAI payload assembly or SSE logic.

v1 scope (plan/bedrock-native-converse.md):

* ``build_converse_request`` — pure OpenAI→Converse translation with
  fail-closed validation (``BedrockAdapterError`` names the offending
  field, never a credential, header, or raw tool-argument body).
* Extended reasoning is not requested: no ``additionalModelRequestFields``
  for thinking. Models that reason anyway (Grok 4.6 and GPT-6 always do,
  as an encrypted ``redactedContent`` blob) get their opaque or signed
  reasoning on a tool-use turn handed back verbatim on the next round:
  the worker emits it as ``reasoning_replay``, the engine carries it on
  the round message (``bedrock_reasoning``), and the adapter replays it
  ahead of the matching ``toolUse`` — to the same model only. Plain
  unsigned reasoning is dropped (Bedrock accepts the follow-up without
  it). Assistant ``reasoning_content`` in history still fails closed:
  ``round_build.py`` persists it as plain text, which AWS would reject
  as a replay of signed reasoning.
* Images: data-URL (base64) sources only — the worker never fetches
  remote image URLs.

The boto3 import is lazy (Task 3) so a lean install without the
``[bedrock]`` extra never needs the AWS SDK.

Traceability: plan/bedrock-native-converse.md (BED-01/03)
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from typing import Any

from sharedagents_model.bedrock_config import (
    AUTH_API_KEY,
    AUTH_IAM,
    AUTH_VALID,
    BEDROCK_BASE_SCHEME,
    BEDROCK_BEARER_ENV,
    BEDROCK_REASONING_FIELD,
    BedrockRoute,
)

#: Bedrock Converse per-image limits (documented for the Converse API;
#: enforced client-side so an oversized image fails before any SDK call).
_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MiB per image
_MAX_IMAGES_PER_REQUEST = 100

_IMAGE_MIME_TO_FORMAT = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

_DATA_URL_RE = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,(.+)$", re.DOTALL)


class BedrockAdapterError(ValueError):
    """A worker request cannot be translated to a legal Converse request.

    Raised BEFORE any SDK call. The message names the offending field and,
    where useful, the tool name / call id — but never a credential, an
    Authorization header, a raw tool-argument body, or image bytes.
    """


def _err(where: str, detail: str) -> BedrockAdapterError:
    return BedrockAdapterError(f"Bedrock adapter: {where}: {detail}")


# ---------------------------------------------------------------------------
# content blocks
# ---------------------------------------------------------------------------


def _image_block(part: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one OpenAI ``image_url`` part to a Converse image block.

    Data-URL (base64) sources only; remote URLs are rejected, never
    fetched.
    """
    url = str((part.get("image_url") or {}).get("url") or "")
    m = _DATA_URL_RE.match(url)
    if not m:
        raise _err(
            "user image",
            "only base64 data-URL images are supported; remote image URLs are not fetched by the Bedrock worker",
        )
    mime = m.group(1).lower()
    fmt = _IMAGE_MIME_TO_FORMAT.get(mime)
    if fmt is None:
        raise _err(
            "user image",
            f"unsupported image MIME type {mime!r} (supported: {sorted(_IMAGE_MIME_TO_FORMAT)})",
        )
    try:
        raw = base64.b64decode(m.group(2), validate=True)
    except (ValueError, TypeError) as exc:
        raise _err("user image", "image payload is not valid base64") from exc
    if len(raw) > _MAX_IMAGE_BYTES:
        raise _err(
            "user image",
            f"image is {len(raw)} bytes; the Bedrock per-image limit is {_MAX_IMAGE_BYTES} (5 MiB)",
        )
    return {"image": {"format": fmt, "source": {"bytes": raw}}}


def _user_content_blocks(
    msg: Mapping[str, Any],
    idx: int,
    image_counter: list[int],
) -> list[dict[str, Any]]:
    """Translate one user message's content to Converse content blocks."""
    where = f"messages[{idx}] (user)"
    content = msg.get("content")
    if isinstance(content, str):
        if not content.strip():
            raise _err(where, "user message has empty content")
        return [{"text": content}]
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise _err(where, "content part is not an object")
            ptype = part.get("type")
            if ptype == "text":
                text = str(part.get("text") or "")
                if text.strip():
                    blocks.append({"text": text})
            elif ptype == "image_url":
                image_counter[0] += 1
                if image_counter[0] > _MAX_IMAGES_PER_REQUEST:
                    raise _err(
                        where,
                        f"more than {_MAX_IMAGES_PER_REQUEST} images in one request",
                    )
                blocks.append(_image_block(part))
            else:
                raise _err(
                    where,
                    f"unsupported user content part type {ptype!r} (v1 supports text and base64 image_url parts)",
                )
        if not blocks:
            raise _err(where, "user message has empty content")
        return blocks
    raise _err(where, "user content must be a string or a list of parts")


def _assistant_text_block(msg: Mapping[str, Any], idx: int) -> str | None:
    """The assistant message's text (or None), validating the shape."""
    where = f"messages[{idx}] (assistant)"
    content = msg.get("content")
    if content is None:
        return None
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "text":
                raise _err(
                    where,
                    "assistant multipart content supports text parts only (images are user-side in v1)",
                )
            text = str(part.get("text") or "")
            if text.strip():
                parts.append(text)
        return "\n".join(parts) if parts else None
    raise _err(where, "assistant content must be a string or text parts")


def _assistant_tool_use_blocks(
    msg: Mapping[str, Any],
    idx: int,
    seen_ids: set[str],
) -> list[dict[str, Any]]:
    """Translate OpenAI ``tool_calls`` to Converse ``toolUse`` blocks."""
    where = f"messages[{idx}] (assistant)"
    calls = msg.get("tool_calls") or []
    if not isinstance(calls, list):
        raise _err(where, "tool_calls must be a list")
    blocks: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise _err(where, "tool call entry is not an object")
        cid = str(call.get("id") or "").strip()
        if not cid:
            raise _err(where, "assistant tool call is missing its id")
        if cid in seen_ids:
            raise _err(where, f"duplicate toolUseId {cid!r}")
        fn = call.get("function") or {}
        if not isinstance(fn, Mapping):
            raise _err(where, f"tool call {cid!r} has no function object")
        name = str(fn.get("name") or "").strip()
        if not name:
            raise _err(where, f"tool call {cid!r} is missing its name")
        args = fn.get("arguments")
        if args is None:
            input_obj: Any = {}
        elif isinstance(args, Mapping):
            input_obj = dict(args)
        elif isinstance(args, str):
            try:
                input_obj = json.loads(args)
            except (ValueError, TypeError) as exc:
                # Redacted: the raw arguments body is NOT embedded in the
                # surfaced error (it can be large and operator-sensitive).
                raise _err(
                    where,
                    f"tool call {cid!r} ({name!r}) has arguments that are not valid JSON",
                ) from exc
        else:
            raise _err(where, f"tool call {cid!r} has non-string arguments")
        if not isinstance(input_obj, Mapping):
            raise _err(
                where,
                f"tool call {cid!r} ({name!r}) arguments must decode to a JSON object",
            )
        seen_ids.add(cid)
        blocks.append(
            {
                "toolUse": {"toolUseId": cid, "name": name, "input": input_obj},
            }
        )
    return blocks


def _assistant_reasoning_blocks(
    msg: Mapping[str, Any],
    idx: int,
    model_id: str,
) -> list[dict[str, Any]]:
    """Replay a tool round's opaque / signed reasoning, verbatim.

    The worker emits it on the ``tool_calls`` event and the engine carries
    it on the round message as ``bedrock_reasoning``. An encrypted blob only
    means something to the model that produced it (a foreign one is an
    opaque upstream 500 on xAI / OpenAI), so another model's payload — a
    mid-session model switch — is dropped, never forwarded.
    """
    payload = msg.get(BEDROCK_REASONING_FIELD)
    if BEDROCK_REASONING_FIELD not in msg:
        return []
    where = f"messages[{idx}] (assistant) {BEDROCK_REASONING_FIELD}"
    if not isinstance(payload, Mapping) or not isinstance(payload.get("blocks"), list):
        raise _err(where, "must be an object with a 'blocks' list")
    if not isinstance(payload.get("model"), str) or not payload["model"]:
        raise _err(where, "model must be a non-empty string")
    if payload["model"] != model_id:
        return []
    blocks: list[dict[str, Any]] = []
    for block in payload["blocks"]:
        if not isinstance(block, Mapping):
            raise _err(where, "reasoning block is not an object")
        if "redacted" in block:
            if set(block) != {"redacted"} or not isinstance(block["redacted"], str):
                raise _err(where, "redacted reasoning must contain only a base64 string")
            try:
                raw = base64.b64decode(block["redacted"], validate=True)
            except (ValueError, TypeError) as exc:
                raise _err(where, "redacted reasoning is not valid base64") from exc
            blocks.append({"reasoningContent": {"redactedContent": raw}})
        elif block.get("signature"):
            if (
                set(block) != {"text", "signature"}
                or not isinstance(block["text"], str)
                or not isinstance(block["signature"], str)
            ):
                raise _err(where, "signed reasoning requires text and signature strings")
            blocks.append(
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": block["text"],
                            "signature": block["signature"],
                        }
                    }
                }
            )
        else:
            raise _err(where, "reasoning block is neither redacted nor signed")
    return blocks


def _tool_result_block(
    msg: Mapping[str, Any],
    idx: int,
    seen_use_ids: set[str],
    seen_result_ids: set[str],
) -> dict[str, Any]:
    """Translate one ``role:'tool'`` message to a Converse ``toolResult``."""
    where = f"messages[{idx}] (tool)"
    tid = str(msg.get("tool_call_id") or "").strip()
    if not tid:
        raise _err(where, "tool result is missing its tool_call_id")
    if tid not in seen_use_ids:
        raise _err(
            where,
            f"tool result references unknown tool call {tid!r} (no matching assistant toolUse earlier in the history)",
        )
    if tid in seen_result_ids:
        raise _err(where, f"duplicate tool result for {tid!r}")
    seen_result_ids.add(tid)
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif content is None:
        text = ""
    elif isinstance(content, Mapping):
        try:
            text = json.dumps(content)
        except (TypeError, ValueError):
            text = str(content)
    else:
        text = str(content)
    status = "success"
    if msg.get("is_error") is True or str(msg.get("status") or "").lower() == "error":
        status = "error"
    return {
        "toolResult": {
            "toolUseId": tid,
            "status": status,
            "content": [{"text": text}],
        }
    }


# ---------------------------------------------------------------------------
# tools / sampling
# ---------------------------------------------------------------------------


def _tool_config(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """Translate the (already permission-filtered) tools + tool_choice."""
    tools = request.get("tools") or []
    if not tools:
        return None
    specs: list[dict[str, Any]] = []
    for i, tool in enumerate(tools):
        fn = (tool or {}).get("function") or {} if isinstance(tool, Mapping) else {}
        name = str(fn.get("name") or "").strip()
        if not name:
            raise _err(f"tools[{i}]", "tool schema is missing its name")
        spec: dict[str, Any] = {"name": name}
        desc = fn.get("description")
        if desc:
            spec["description"] = str(desc)
        params = fn.get("parameters")
        if isinstance(params, Mapping):
            spec["inputSchema"] = {"json": dict(params)}
        specs.append({"toolSpec": spec})
    config: dict[str, Any] = {"tools": specs}
    choice = request.get("tool_choice")
    if choice is None or choice == "auto":
        config["toolChoice"] = {"auto": {}}
    elif isinstance(choice, Mapping) and choice.get("type") == "function":
        name = str(((choice.get("function") or {}).get("name")) or "").strip()
        if not name:
            raise _err("tool_choice", "named tool_choice is missing its name")
        config["toolChoice"] = {"tool": {"name": name}}
    else:
        raise _err(
            "tool_choice",
            f"unsupported tool_choice {choice!r} (v1 supports 'auto' and named function choice)",
        )
    return config


def _inference_config(request: Mapping[str, Any]) -> dict[str, Any]:
    """The common sampling fields Converse accepts (and nothing else)."""
    ic: dict[str, Any] = {}
    max_tokens = request.get("max_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0:
        ic["maxTokens"] = max_tokens
    temperature = request.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        ic["temperature"] = float(temperature)
    top_p = request.get("top_p")
    if isinstance(top_p, (int, float)) and not isinstance(top_p, bool):
        ic["topP"] = float(top_p)
    return ic


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------


def build_converse_request(worker_request: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI-shaped worker request to a Converse request.

    Pure and side-effect free; the input mapping is never mutated. Raises
    :class:`BedrockAdapterError` (fail closed, redacted) before any SDK
    call when the history or request shape cannot be translated legally.

    Mapping (v1):

    * ``system`` message(s) → ``system: [{'text': …}]`` (order preserved)
    * ``user``/``assistant`` text → ``messages[].content[].text``
    * assistant ``tool_calls`` → same-message ``toolUse`` blocks with the
      exact ``toolUseId`` / ``name`` and JSON-decoded object ``input``
    * assistant ``bedrock_reasoning`` from THIS model → leading
      ``reasoningContent`` blocks (``redactedContent`` bytes, or
      ``reasoningText`` with its ``signature``); another model's is dropped
    * ``role:'tool'`` messages → user ``toolResult`` blocks referencing
      those ids; consecutive tool messages (and a tool result immediately
      followed by user text) coalesce into ONE user message so roles
      strictly alternate
    * user ``image_url`` data-URL parts → binary ``image`` blocks
    * permission-filtered ``tools`` → ``toolConfig.tools[].toolSpec``
      (``inputSchema.json``); empty tools omit ``toolConfig``;
      ``tool_choice`` ``'auto'`` → ``{'auto': {}}``, named → ``{'tool': …}``
    * ``max_tokens``/``top_p``/``temperature`` → ``inferenceConfig``
    * everything else (seed, penalties, ``reasoning_effort``,
      ``chat_template_kwargs``, ``stream_options``, ``extra``, …) is
      OMITTED — never guessed, never flattened
    * ``modelId`` is kept verbatim (foundation id or profile ARN)
    """
    if not isinstance(worker_request, Mapping):
        raise _err("request", "worker request must be a mapping")
    model_id = str(worker_request.get("model") or "").strip()
    if not model_id:
        raise _err("request", "model is required")
    messages = worker_request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _err("request", "messages must be a non-empty list")

    system_blocks: list[dict[str, Any]] = []
    converse_messages: list[dict[str, Any]] = []
    seen_use_ids: set[str] = set()
    seen_result_ids: set[str] = set()
    image_counter: list[int] = [0]

    def _push_user(blocks: list[dict[str, Any]]) -> None:
        # Coalesce: a toolResult (or follow-up user text) directly after
        # another user-role message would break Converse's strict
        # role alternation — merge into the previous user message.
        if converse_messages and converse_messages[-1]["role"] == "user":
            converse_messages[-1]["content"].extend(blocks)
        else:
            converse_messages.append({"role": "user", "content": list(blocks)})

    for idx, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            raise _err(f"messages[{idx}]", "message entry is not an object")
        role = msg.get("role")
        if role == "system":
            content = msg.get("content")
            if isinstance(content, str):
                if content.strip():
                    system_blocks.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, Mapping) or part.get("type") != "text":
                        raise _err(
                            f"messages[{idx}] (system)",
                            "system content supports text parts only",
                        )
                    text = str(part.get("text") or "")
                    if text.strip():
                        system_blocks.append({"text": text})
            elif content is not None:
                raise _err(
                    f"messages[{idx}] (system)",
                    "system content must be a string or text parts",
                )
            continue

        if role == "user":
            _push_user(_user_content_blocks(msg, idx, image_counter))

        elif role == "assistant":
            if msg.get("reasoning_content"):
                # v1 ships no extended reasoning: round_build.py persists
                # only plain (unsigned) reasoning, and AWS requires
                # unchanged signed reasoning/history on replay. Fail
                # closed before emitting an executable request.
                raise _err(
                    f"messages[{idx}] (assistant)",
                    "assistant reasoning content cannot be replayed to "
                    "Bedrock (v1 ships no extended reasoning); start a "
                    "fresh session or switch back to the original model",
                )
            blocks: list[dict[str, Any]] = []
            text = _assistant_text_block(msg, idx)
            if text:
                blocks.append({"text": text})
            tool_uses = _assistant_tool_use_blocks(msg, idx, seen_use_ids)
            blocks.extend(tool_uses)
            if not blocks:
                raise _err(
                    f"messages[{idx}] (assistant)",
                    "assistant message has neither text nor tool calls",
                )
            # The model's own reasoning leads the turn, as it was generated —
            # ahead of the text and the toolUse it produced. It was captured
            # for those tool calls: once compaction has stripped them all
            # (compaction.py keeps a message that still has text), the
            # reasoning is orphaned and is dropped rather than replayed.
            reasoning = _assistant_reasoning_blocks(msg, idx, model_id)
            if tool_uses:
                blocks[:0] = reasoning
            if converse_messages and converse_messages[-1]["role"] == "assistant":
                # Defensive: sharedagents history alternates, but never emit an
                # illegal consecutive-assistant message.
                converse_messages[-1]["content"].extend(blocks)
            else:
                converse_messages.append(
                    {"role": "assistant", "content": blocks},
                )

        elif role == "tool":
            _push_user(
                [
                    _tool_result_block(
                        msg,
                        idx,
                        seen_use_ids,
                        seen_result_ids,
                    )
                ]
            )

        else:
            raise _err(f"messages[{idx}]", f"unsupported role {role!r}")

    if not converse_messages or converse_messages[0]["role"] != "user":
        raise _err("messages", "the first message must be from the user")

    out: dict[str, Any] = {"modelId": model_id}
    if system_blocks:
        out["system"] = system_blocks
    out["messages"] = converse_messages
    ic = _inference_config(worker_request)
    if ic:
        out["inferenceConfig"] = ic
    tc = _tool_config(worker_request)
    if tc is not None:
        out["toolConfig"] = tc
    return out


# ---------------------------------------------------------------------------
# Transport (Task 3) — ConverseStream worker
# ---------------------------------------------------------------------------

#: Converse ``messageStop.stopReason`` → sharedagents ``finish_reason``.
_STOP_REASON_FINISH = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}

#: stopReason values that are NOT natural completions.
_STOP_REASON_ERRORS = {
    "model_context_window_exceeded": (
        "bedrock: model context window exceeded (prompt too large for the model's context window)"
    ),
    "content_filtered": ("bedrock: response was blocked by a content filter (stop reason 'content_filtered')"),
    "guardrail_intervention": ("bedrock: response was blocked by a guardrail (stop reason 'guardrail_intervention')"),
    "loop_detected": ("bedrock: model reported a loop (stop reason 'loop_detected')"),
}

#: ClientError codes that are transient upstream failures.
_TRANSIENT_CODES = {
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
    "RequestThrottledException",
    "ServiceUnavailableException",
    "InternalServerException",
    "SlowDownException",
}

#: ClientError codes that are credential / permission failures.
_AUTH_CODES = {
    "UnrecognizedClientException",
    "InvalidSignatureException",
    "ExpiredTokenException",
    "InvalidBearerTokenException",
    "AccessDeniedException",
}

#: Bearer-token env var name is owned by the shared route module.
_BEDROCK_BEARER_ENV = BEDROCK_BEARER_ENV
_DEFAULT_READ_TIMEOUT_S = 120.0
_MIN_READ_TIMEOUT_S = 10.0
_MAX_READ_TIMEOUT_S = 600.0
_CONNECT_TIMEOUT_S = 10.0
_MAX_ERROR_DETAIL = 300
# This is the SDK pair exercised by the no-network signing transport tests.
# Keep the optional dependency floor in pyproject.toml in sync.
_MIN_SDK_VERSION = (1, 43, 101)


def _emit_jsonl(kind: str, **fields: Any) -> None:
    """Serialize one worker event to a JSONL line on stdout and flush.

    Same contract as ``sharedagents_gateway.worker.emit`` (the Bedrock path is
    dispatched from ``worker.main`` before any OpenAI plumbing, so it owns
    its emitter to stay import-light and testable).
    """
    sys.stdout.write(json.dumps({"kind": kind, **fields}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _route_from_request(worker_request: Mapping[str, Any]) -> BedrockRoute:
    """Validate the route identity carried by the worker request.

    The gateway already validated the model row (Task 1); this is the
    worker-side re-check of the small request shape — fail closed if the
    identity is missing or internally inconsistent.
    """
    region = str(worker_request.get("region") or "").strip()
    if not region:
        raise BedrockAdapterError("bedrock request", "region", "worker request is missing 'region'")
    auth = str(worker_request.get("auth") or "").strip().lower()
    if auth not in AUTH_VALID:
        raise BedrockAdapterError(
            "bedrock request",
            "auth",
            f"'auth' must be one of {list(AUTH_VALID)}, got {auth!r}",
        )
    model_id = str(worker_request.get("model") or "").strip()
    if not model_id:
        raise BedrockAdapterError("bedrock request", "model", "worker request is missing 'model'")
    profile = worker_request.get("profile")
    if profile is not None:
        profile = str(profile).strip()
        if not profile or any(c.isspace() for c in profile):
            raise BedrockAdapterError(
                "bedrock request",
                "profile",
                "'profile' must be a non-blank boto3 profile name without whitespace",
            )
    else:
        profile = None
    endpoint = str(worker_request.get("endpoint") or "").strip()
    if endpoint:
        scheme, sep, rest = endpoint.partition("://")
        if not sep or scheme.lower() != BEDROCK_BASE_SCHEME or rest.rstrip("/") != region:
            raise BedrockAdapterError(
                "bedrock request",
                "endpoint",
                f"endpoint {endpoint!r} does not match region {region!r}",
            )
    return BedrockRoute(region=region, model_id=model_id, auth=auth, profile=profile)


def _check_auth_env(route: BedrockRoute) -> None:
    """Fail closed on credential-mode ambiguity BEFORE any SDK call.

    * ``bedrock-api-key`` requires the per-subprocess bearer token env var
      (injected by ``worker_env_for_model``); its absence is a hard error.
    * ``aws-iam`` requires the bearer token to be ABSENT, so botocore
      resolves the standard IAM chain and never silently prefers bearer.
    """
    token = os.environ.get(_BEDROCK_BEARER_ENV)
    if route.auth == AUTH_API_KEY:
        if not token:
            raise BedrockAdapterError(
                "bedrock request",
                "auth",
                "auth='bedrock-api-key' but "
                f"{_BEDROCK_BEARER_ENV} is not set in the worker environment "
                "(the gateway must inject it per-subprocess)",
            )
    elif route.auth == AUTH_IAM and token:
        raise BedrockAdapterError(
            "bedrock request",
            "auth",
            "auth='aws-iam' but "
            f"{_BEDROCK_BEARER_ENV} is present in the worker environment; "
            "refusing to resolve credentials ambiguously",
        )


def _read_timeout_s() -> float:
    raw = os.environ.get("SHAREDAGENTS_BEDROCK_READ_TIMEOUT_S", "")
    try:
        val = float(raw) if raw else _DEFAULT_READ_TIMEOUT_S
    except ValueError:
        val = _DEFAULT_READ_TIMEOUT_S
    return max(_MIN_READ_TIMEOUT_S, min(val, _MAX_READ_TIMEOUT_S))


def _make_client(route: BedrockRoute) -> Any:
    """Build the boto3 Bedrock client (lazy SDK import, bounded config)."""
    try:
        import boto3
        import botocore
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is not installed; install the optional extra: pip install 'sharedagents[bedrock]'"
        ) from exc
    for sdk in (boto3, botocore):
        version = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", sdk.__version__)
        if version is None or tuple(map(int, version.groups())) < _MIN_SDK_VERSION or int(version.group(1)) != 1:
            raise RuntimeError(
                "Bedrock requires boto3 and botocore >=1.43.101,<2; "
                "upgrade the optional extra: pip install --upgrade 'sharedagents[bedrock]'"
            )
    kwargs: dict[str, Any] = {
        "region_name": route.region,
        "config": Config(
            signature_version="bearer" if route.auth == AUTH_API_KEY else "v4",
            connect_timeout=_CONNECT_TIMEOUT_S,
            read_timeout=_read_timeout_s(),
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    }
    session = boto3.Session(profile_name=route.profile)
    return session.client("bedrock-runtime", **kwargs)


def _detail(exc: BaseException) -> str:
    """A bounded, redacted detail string for error messages.

    Never includes request/response headers or credentials — botocore
    exception text is generic API error text, truncated to keep the JSONL
    line bounded.
    """
    return str(exc)[:_MAX_ERROR_DETAIL]


def _client_error_message(exc: Any) -> str:
    """Normalize a ``botocore`` ``ClientError`` to a one-line message.

    The message is classifier-compatible for the parent turn loop
    (``_classify_upstream_error``): context overflow must contain
    ``context window``, throttling must contain ``rate limit``.
    """
    code = ""
    message = ""
    status = None
    try:
        err = exc.response.get("Error") or {}
        code = str(err.get("Code") or "")
        message = str(err.get("Message") or "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    except AttributeError:
        pass
    message = message[:_MAX_ERROR_DETAIL]
    if code == "ModelContextWindowExceededException":
        return f"bedrock: model context window exceeded: {message}"
    if code in _TRANSIENT_CODES:
        if code in (
            "ThrottlingException",
            "ProvisionedThroughputExceededException",
            "RequestThrottledException",
        ):
            return f"bedrock: rate limit (throttled by upstream, {code}): {message}"
        return f"bedrock: upstream 5xx ({code}): {message}"
    if code in _AUTH_CODES:
        if code == "AccessDeniedException":
            return (
                f"bedrock: access denied (check IAM permissions, e.g. bedrock:InvokeModelWithResponseStream): {message}"
            )
        return f"bedrock: auth failure ({code}): {message}"
    if code == "ValidationException":
        return f"bedrock: validation ({code}): {message}"
    if code in ("ResourceNotFoundException", "ModelNotFoundException"):
        return f"bedrock: model not found ({code}): {message}"
    if status in (429, 500, 502, 503, 504):
        return f"bedrock: upstream HTTP {status} ({code}): {message}"
    return f"bedrock: {code or 'error'}: {message}"


def _boto_error_message(exc: BaseException) -> str:
    """Normalize a ``botocore`` ``BotoCoreError`` (network/credential layer)."""
    cls = type(exc).__name__
    message = _detail(exc)
    if cls == "NoCredentialsError":
        return (
            "bedrock: no AWS credentials found in the worker environment "
            "(auth='aws-iam'; set ambient credentials or a profile)"
        )
    if cls == "NoAuthTokenError":
        return f"bedrock: no bearer auth token available (auth='bedrock-api-key'; {_BEDROCK_BEARER_ENV} must be set)"
    if cls in ("ConnectTimeoutError", "ReadTimeoutError"):
        return f"bedrock: request timed out ({cls}): {message}"
    if cls == "ConnectionError":
        return f"bedrock: connection error: {message}"
    if cls == "PartialCredentialsError":
        return f"bedrock: incomplete AWS credentials ({cls}): {message}"
    return f"bedrock: {cls}: {message}"


def _usage_metrics(usage: Any) -> dict[str, Any]:
    """Map Converse ``metadata.usage`` to sharedagents metrics fields.

    Missing usage → all token fields ``None`` + ``tokens_estimated=True``
    (never fabricate zeros for billing). v1 ships no extended reasoning,
    so there is no provider reasoning-token billing split:
    ``reasoning_tokens`` stays 0 and ``content_tokens`` = completion.
    """
    if not isinstance(usage, Mapping):
        return {
            "prompt_tokens": None,
            "prompt_tokens_cached": None,
            "prompt_tokens_cache_write": None,
            "prompt_tokens_fresh": None,
            "completion_tokens": None,
            "content_tokens": None,
            "reasoning_tokens": 0,
            "tokens_estimated": True,
        }
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    cache_read = usage.get("cacheReadInputTokens")
    cache_write = usage.get("cacheWriteInputTokens")
    fresh = None
    if input_tokens is not None:
        fresh = max(0, int(input_tokens) - int(cache_read or 0) - int(cache_write or 0))
    return {
        "prompt_tokens": input_tokens,
        "prompt_tokens_cached": cache_read,
        "prompt_tokens_cache_write": cache_write,
        "prompt_tokens_fresh": fresh,
        "completion_tokens": output_tokens,
        "content_tokens": output_tokens,
        "reasoning_tokens": 0,
        "tokens_estimated": (input_tokens is None and output_tokens is None),
    }


def _add_reasoning_delta(slot: dict[str, Any], delta: Any) -> None:
    """Accumulate one ``reasoningContent`` delta into its block's slot.

    ``ReasoningContentBlockDelta`` is a union: exactly one of ``text``,
    ``signature`` (a block is plain or signed text) or ``redactedContent``
    (an opaque blob, possibly split across deltas). Anything else — an
    unknown future member (botocore's ``SDK_UNKNOWN_MEMBER``), a wrong type,
    text mixed with a blob in one block — marks the block unreplayable
    instead of raising: a text turn discards reasoning anyway, and a
    tool-use turn then fails closed in :func:`run_bedrock_worker`.
    """
    if not isinstance(delta, Mapping) or len(delta) != 1:
        slot["invalid"] = "expected one reasoning delta member"
        return
    ((key, value),) = delta.items()
    if key == "redactedContent":
        if not isinstance(value, (bytes, bytearray)):
            slot["invalid"] = "redactedContent is not bytes"
        elif slot["text"] or slot["signature"]:
            slot["invalid"] = "redacted and text reasoning mixed in one block"
        else:
            slot["redacted"] += value
    elif key in ("text", "signature"):
        if not isinstance(value, str):
            slot["invalid"] = f"reasoning {key} is not a string"
        elif slot["redacted"]:
            slot["invalid"] = "redacted and text reasoning mixed in one block"
        elif key == "text":
            slot["text"].append(value)
        else:
            slot["signature"] += value
    else:
        slot["invalid"] = f"unsupported reasoning delta member {key!r}"


def _reasoning_replay(
    model_id: str,
    blocks: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """The reasoning a tool-use turn must hand back verbatim, or ``None``.

    Opaque (``redactedContent``) and signed blocks cannot be reconstructed,
    so they ride the ``tool_calls`` event (JSON-safe: blobs as base64) for
    the engine to replay next round. Plain unsigned text is dropped, as on a
    text turn — Bedrock accepts the tool follow-up without it (live probe
    2026-09-25: Kimi K2 Thinking / K3, gpt-oss-120b, MiniMax M2.5).
    """
    out: list[dict[str, str]] = []
    for idx in sorted(blocks):
        slot = blocks[idx]
        if slot["redacted"]:
            out.append(
                {
                    "redacted": base64.b64encode(bytes(slot["redacted"])).decode("ascii"),
                }
            )
        elif slot["signature"]:
            out.append({"text": "".join(slot["text"]), "signature": slot["signature"]})
    return {"model": model_id, "blocks": out} if out else None


def run_bedrock_worker(
    worker_request: Mapping[str, Any],
    *,
    client_factory: Any = None,
    emit_fn: Any = None,
) -> int:
    """Run one Bedrock ConverseStream turn, emitting JSONL worker events.

    Emits the same event contract as the OpenAI worker: ``content`` per
    text delta, one ``tool_calls`` event (buffer-validated JSON) when
    ``stopReason == 'tool_use'`` — carrying ``reasoning_replay`` when the
    turn's opaque / signed reasoning must be handed back next round —
    ``metrics`` (``provider='bedrock'``),
    then ``done`` — or a single terminal ``error`` event. Returns the
    process exit code (0 success, 1 failure).

    ``client_factory`` / ``emit_fn`` are test seams (a fake client / event
    sink); production passes nothing and gets the real boto3 client and
    stdout emitter.
    """
    emit = emit_fn or _emit_jsonl
    t0 = time.monotonic()

    # 1. Route identity (fail closed before any translation or SDK work).
    try:
        route = _route_from_request(worker_request)
    except BedrockAdapterError as e:
        emit("error", message=str(e))
        return 1

    # 2. OpenAI → Converse translation (pure).
    try:
        converse = build_converse_request(worker_request)
    except BedrockAdapterError as e:
        emit("error", message=str(e))
        return 1

    # 3. Credential-mode pre-check (fail closed, client never called).
    try:
        _check_auth_env(route)
    except BedrockAdapterError as e:
        emit("error", message=str(e))
        return 1

    # 4. Client (lazy SDK import; bounded connect/read timeouts + retries).
    try:
        client = (client_factory or _make_client)(route)
    except RuntimeError as e:
        emit("error", message=str(e))
        return 1
    except ImportError as e:
        # A lean install without the [bedrock] extra: name the fix rather
        # than surfacing a bare "No module named 'boto3'".
        emit(
            "error",
            message=(
                "bedrock: the AWS SDK is not installed — install the optional "
                f"extra: pip install 'sharedagents[bedrock]' ({_detail(e)})"
            ),
        )
        return 1
    except Exception as e:  # noqa: BLE001 — redacted, never a traceback
        emit(
            "error",
            message=f"bedrock: client init failed ({type(e).__name__}): {_detail(e)}",
        )
        return 1

    # 5. Stream.
    try:
        response = client.converse_stream(**converse)
    except Exception as e:  # noqa: BLE001
        emit("error", message=_normalize_sdk_error(e))
        return 1

    stop_reason: str | None = None
    usage: Any = None
    # Time-to-first-token: the first text/reasoning/tool delta, mirroring the
    # OpenAI worker's ttft_ms (accept + prefill + first decode-and-emit).
    t_first: float | None = None
    tool_blocks: dict[int, dict[str, str]] = {}  # index -> {id, name, input}
    # index -> {text: [str], signature: str, redacted: bytearray}
    reasoning_blocks: dict[int, dict[str, Any]] = {}

    def _reasoning_slot(idx: int) -> dict[str, Any]:
        return reasoning_blocks.setdefault(
            idx,
            {"text": [], "signature": "", "redacted": bytearray(), "invalid": None},
        )

    try:
        stream = response["stream"]
        for event in stream:
            if "messageStart" in event:
                continue
            if "contentBlockStart" in event:
                cbs = event["contentBlockStart"]
                idx = int(cbs.get("contentBlockIndex", 0))
                start = cbs.get("start") or {}
                if "toolUse" in start:
                    if t_first is None:
                        t_first = time.monotonic()
                    tu = start["toolUse"] or {}
                    tool_blocks[idx] = {
                        "id": str(tu.get("toolUseId") or ""),
                        "name": str(tu.get("name") or ""),
                        "input": "",
                    }
                elif "reasoningContent" in start:
                    _reasoning_slot(idx)
                    if t_first is None:
                        t_first = time.monotonic()
            elif "contentBlockDelta" in event:
                cbd = event["contentBlockDelta"]
                idx = int(cbd.get("contentBlockIndex", 0))
                delta = cbd.get("delta") or {}
                if "text" in delta:
                    text = delta.get("text")
                    if text:
                        if t_first is None:
                            t_first = time.monotonic()
                        emit("content", text=text)
                elif "toolUse" in delta:
                    if idx in tool_blocks:
                        tool_blocks[idx]["input"] += (delta.get("toolUse") or {}).get("input") or ""
                elif "reasoningContent" in delta:
                    _add_reasoning_delta(
                        _reasoning_slot(idx),
                        delta.get("reasoningContent") or {},
                    )
                    if t_first is None:
                        t_first = time.monotonic()
            elif "messageStop" in event:
                stop_reason = str((event["messageStop"] or {}).get("stopReason") or "")
            elif "metadata" in event:
                usage = (event["metadata"] or {}).get("usage")
            elif "exception" in event:
                err = (event["exception"] or {}).get("error") or {}
                etype = str(err.get("type") or "unknown")
                emsg = str(err.get("message") or "")[:_MAX_ERROR_DETAIL]
                emit("error", message=f"bedrock: stream error: {etype}: {emsg}")
                return 1
            # contentBlockStop / guardContent: no action needed.
    except Exception as e:  # noqa: BLE001 — mid-stream transport failure
        emit("error", message=_normalize_sdk_error(e))
        return 1

    # 6. Terminal discipline: only a valid messageStop counts.
    if not stop_reason:
        emit(
            "error",
            message=("bedrock: stream ended without a terminal messageStop; refusing to synthesize a finish reason"),
        )
        return 1

    if stop_reason in _STOP_REASON_ERRORS:
        emit("error", message=_STOP_REASON_ERRORS[stop_reason])
        return 1

    finish_reason = _STOP_REASON_FINISH.get(stop_reason)
    if finish_reason is None:
        emit(
            "error",
            message=f"bedrock: unrecognized stop reason {stop_reason!r}",
        )
        return 1

    latency_ms = (time.monotonic() - t0) * 1000.0

    if finish_reason == "tool_calls":
        unreplayable = next(
            (reasoning_blocks[i]["invalid"] for i in sorted(reasoning_blocks) if reasoning_blocks[i]["invalid"]),
            None,
        )
        if unreplayable:
            # Reasoning the worker could not capture verbatim cannot be handed
            # back next round; never run tools whose follow-up would lose it.
            emit(
                "error",
                message=(
                    "bedrock: model returned reasoning the worker cannot hand "
                    f"back ({unreplayable}); refusing to emit executable tool calls"
                ),
            )
            return 1
        calls: list[dict[str, Any]] = []
        for idx in sorted(tool_blocks):
            tb = tool_blocks[idx]
            if not tb["id"] or not tb["name"]:
                emit(
                    "error",
                    message=(
                        "bedrock: toolUse block is missing its toolUseId or "
                        "name; refusing to emit an unidentifiable tool call"
                    ),
                )
                return 1
            raw = tb["input"]
            try:
                args = json.loads(raw) if raw.strip() else {}
            except (ValueError, TypeError):
                emit(
                    "error",
                    message=(f"bedrock: tool call {tb['name']!r} returned arguments that are not valid JSON"),
                )
                return 1
            if not isinstance(args, dict):
                emit(
                    "error",
                    message=(f"bedrock: tool call {tb['name']!r} arguments must be a JSON object"),
                )
                return 1
            calls.append(
                {
                    "id": tb["id"],
                    "name": tb["name"],
                    "arguments": args,
                    "raw_arguments": raw,
                }
            )
        replay = _reasoning_replay(converse["modelId"], reasoning_blocks)
        if replay:
            emit("tool_calls", calls=calls, reasoning_replay=replay)
        else:
            emit("tool_calls", calls=calls)

    # 7. Metrics (provider attribution for pricing) + done.
    emit(
        "metrics",
        model=converse["modelId"],
        model_alias=None,
        provider="bedrock",
        **_usage_metrics(usage),
        latency_ms=round(latency_ms, 1),
        ttft_ms=round((t_first - t0) * 1000.0, 1) if t_first is not None else None,
        finish_reason=finish_reason,
    )
    emit("done")
    return 0


def _normalize_sdk_error(e: BaseException) -> str:
    """Normalize any SDK exception to a redacted, classifier-friendly line."""
    try:
        import botocore.exceptions as _be

        if isinstance(e, _be.ClientError):
            return _client_error_message(e)
        if isinstance(e, _be.BotoCoreError):
            return _boto_error_message(e)
    except ImportError:
        pass
    return f"bedrock: {type(e).__name__}: {_detail(e)}"
