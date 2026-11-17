"""JSON-RPC 2.0 types + newline-delimited JSON framing for ACP stdio.

This module is byte-fidelity-conscious. From ``CLAUDE.md``:

    Tool schemas must be deterministic. Prior turns' tool calls and
    tool results stay byte-identical when replayed — store the raw
    arguments JSON the model originally emitted (raw_arguments field)
    and replay it verbatim, don't re-serialize via json.dumps which
    can reorder keys.

The same rule applies to JSON-RPC framing on the wire: when prefix
caching is in play upstream (vLLM APC, SGLang RadixAttention), every
re-serialization can shift bytes and force a cache miss. Our
``JsonRpcRequest`` / ``JsonRpcResponse`` / ``JsonRpcNotification``
preserve the **raw** ``params`` / ``result`` JSON bytes alongside the
parsed Python view. ``parse_message`` populates both; callers that
care about byte-fidelity replay use ``raw_params`` /  ``raw_result``,
not ``json.dumps(params)``.

Framing is **newline-delimited JSON** — the Agent Client Protocol
stdio wire format that Zed and Cursor speak: one compact JSON object
per line, terminated by ``\\n`` (tolerant of ``\\r\\n`` on Windows),
with no length prefix::

    {"jsonrpc":"2.0","id":1,"method":"initialize",...}\\n
    {"jsonrpc":"2.0","id":1,"result":{...}}\\n

This is NOT the LSP / MCP-over-stdio ``Content-Length`` header format
— that framing is wrong for ACP and earlier versions of this module
got it wrong. ``read_message`` returns the line's bytes (terminator
stripped) as raw bytes — no parse happens at the framing layer.
Callers parse the body via ``parse_message`` (which rejects batch
arrays at the framing boundary; ACP's JSON-RPC never uses them).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

# ---------------------------------------------------------------------------
# Error codes
# ---------------------------------------------------------------------------


class JsonRpcErrorCode(IntEnum):
    """JSON-RPC 2.0 standard error codes plus the RFD-specific 501.

    Standard codes are those defined in the JSON-RPC 2.0 spec. The RFD
    Streamable HTTP transport adds a server-error-range value (501)
    used to reject batch requests over HTTP — we surface that here so
    the same code can be reused by the framing layer to flag batch
    rejection consistently across stdio / WS / HTTP.
    """

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    # ACP-defined: returned from session/new (and friends) when the
    # client must call ``authenticate`` before opening a session.
    # Spec ref: ``error.rs:AUTH_REQUIRED`` / schema $defs ErrorCode.
    AUTH_REQUIRED = -32000
    # Server-error-range values (-32000..-32099 reserved per JSON-RPC).
    BATCH_NOT_SUPPORTED = -32001
    # RFD-defined HTTP-status mapping for batch rejection.
    HTTP_BATCH_REJECTED = 501


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class JsonRpcParseError(Exception):
    """Raised when bytes cannot be parsed into a valid JSON-RPC message."""

    def __init__(self, message: str, *, code: int = JsonRpcErrorCode.PARSE_ERROR) -> None:
        super().__init__(message)
        self.code = code


class JsonRpcFramingError(Exception):
    """Raised when the newline-delimited frame on the wire is malformed
    (clean/mid-line EOF, or a line larger than the stream buffer)."""


# ---------------------------------------------------------------------------
# Message dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JsonRpcError:
    """JSON-RPC error object, transport-independent."""

    code: int
    message: str
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": int(self.code), "message": self.message}
        if self.data is not None:
            out["data"] = self.data
        return out


@dataclass(frozen=True)
class JsonRpcRequest:
    """A JSON-RPC 2.0 request (has ``id``, expects a response).

    ``params`` is the parsed Python view; ``raw_params`` is the original
    body of the request as bytes, suitable for byte-identical replay.
    Storing both lets handlers that care about prefix-cache stability
    pass through the on-wire bytes without re-serializing. Both fields
    refer to the same underlying value when the wire bytes are the
    canonical form.
    """

    id: str | int
    method: str
    params: Any = None
    raw_params: bytes | None = None
    raw_message: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def is_request(self) -> bool:
        return True


@dataclass(frozen=True)
class JsonRpcNotification:
    """A JSON-RPC 2.0 notification (no ``id``, no response expected)."""

    method: str
    params: Any = None
    raw_params: bytes | None = None
    raw_message: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def is_notification(self) -> bool:
        return True


@dataclass(frozen=True)
class JsonRpcResponse:
    """A JSON-RPC 2.0 response.

    Either ``result`` or ``error`` is set, not both. ``raw_result``
    preserves the original bytes for the result branch (mirror of
    ``raw_params`` on requests).
    """

    id: str | int | None
    result: Any = None
    error: JsonRpcError | None = None
    raw_result: bytes | None = None
    raw_message: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def is_response(self) -> bool:
        return True


JsonRpcMessage = JsonRpcRequest | JsonRpcResponse | JsonRpcNotification


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


# Generous high-water mark for the stdio read buffer. NDJSON framing
# pulls a whole line into memory before the ``\n`` delimiter, so the
# buffer must accommodate the largest single message — ACP prompts
# carry base64 image blocks that blow past asyncio's 64 KiB default.
# This is a ceiling, not a preallocation. The stdio stream / subprocess
# constructors size their reader off this value (see
# ``server/spawn.py`` and ``client/spawn.py``).
ACP_STREAM_LIMIT = 64 * 1024 * 1024  # 64 MiB


def _strip_eol(line: bytes) -> bytes:
    """Strip exactly one trailing ``\\n`` and an optional preceding
    ``\\r`` (Windows CRLF). Does not touch other bytes — a JSON value's
    escaped ``\\n`` is the two-char sequence and never a raw newline."""
    if line.endswith(b"\n"):
        line = line[:-1]
    if line.endswith(b"\r"):
        line = line[:-1]
    return line


async def read_message(reader: asyncio.StreamReader) -> bytes:
    """Read one newline-delimited JSON message from ``reader``.

    Reads a single line (everything up to and including the next
    ``\\n``) and returns the **raw bytes** of the body with the line
    terminator stripped (one ``\\n`` plus an optional preceding ``\\r``).
    No parsing happens here — the caller hands the bytes to
    ``parse_message`` (which keeps a reference for byte-fidelity replay).

    Blank lines are skipped: ACP emits exactly one JSON object per line,
    but tolerating stray blank lines keeps the reader robust.

    Raises:
        JsonRpcFramingError: on clean EOF (stream closed with nothing
            buffered) — message prefixed ``"EOF before any header
            bytes"`` so ``serve_until_close`` and the transport EOF
            paths shut down cleanly — or when a single line exceeds the
            stream buffer limit (``ACP_STREAM_LIMIT``).

    A final message with no trailing newline (EOF mid-line) is still
    returned: a well-formed object without a terminator is a message.
    """

    while True:
        try:
            line = await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError as exc:
            # Stream closed. ``partial`` holds bytes read since the last
            # newline: empty → clean EOF; non-empty → a trailing message
            # the peer didn't newline-terminate before closing.
            partial = bytes(exc.partial)
            if not partial:
                raise JsonRpcFramingError("EOF before any header bytes") from exc
            return _strip_eol(partial)
        except asyncio.LimitOverrunError as exc:
            raise JsonRpcFramingError(f"message exceeds stream buffer limit ({exc})") from exc

        body = _strip_eol(bytes(line))
        if body:
            return body
        # Blank line — skip it and read the next.


async def write_message(
    writer: asyncio.StreamWriter,
    payload: bytes | Mapping[str, Any] | JsonRpcRequest | JsonRpcResponse | JsonRpcNotification,
) -> None:
    """Write one newline-delimited JSON message to ``writer``.

    Accepts either pre-serialized bytes (preferred — preserves byte
    fidelity) or a dict / dataclass that we serialize via the standard
    JSON encoder. The body is written as one compact line followed by a
    single ``\\n`` terminator — no ``Content-Length`` header.

    Compact serialization (``separators=(",", ":")``, default
    ``ensure_ascii=True``) guarantees the body holds no raw newline, so
    each message occupies exactly one line. Pre-serialized bytes from
    callers (and ``raw_message`` captured by ``parse_message``, which
    strips the terminator on read) are likewise single-line; for
    byte-identical replay always pass bytes.
    """

    if isinstance(payload, (bytes, bytearray)):
        body = bytes(payload)
    elif isinstance(payload, (JsonRpcRequest, JsonRpcResponse, JsonRpcNotification)):
        # Prefer the original wire bytes if we have them.
        if payload.raw_message is not None:
            body = payload.raw_message
        else:
            body = json.dumps(_dataclass_to_dict(payload), separators=(",", ":")).encode("utf-8")
    elif isinstance(payload, Mapping):
        body = json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")
    else:
        raise TypeError(f"unsupported payload type: {type(payload).__name__}")

    writer.write(body + b"\n")
    await writer.drain()


def frame_ndjson_line(payload: bytes) -> bytes:
    """Validate ``payload`` as a single NDJSON body and append the ``\n``
    delimiter, returning the framed line.

    Every transport (stdio/HTTP/WS) emits exactly one
    newline-delimited JSON object per message. The stdio path frames via
    :func:`write_message`; the network transports (HTTP SSE, WebSocket) call
    this so all three share one framing contract. A payload carrying an
    INTERIOR newline is a corrupt multi-line frame (two objects glued
    together) — reject it with ``JsonRpcParseError`` (``INVALID_REQUEST``)
    rather than emitting a frame a peer's line reader would split wrong. An
    already-terminated payload (single trailing ``\n``) is accepted and
    re-normalized to exactly one terminator.
    """
    body = payload
    if body.endswith(b"\n"):
        body = body[:-1]
    if b"\n" in body:
        raise JsonRpcParseError(
            "multi-line payload cannot be framed as a single NDJSON line",
            code=JsonRpcErrorCode.INVALID_REQUEST,
        )
    return body + b"\n"


def _dataclass_to_dict(
    msg: JsonRpcRequest | JsonRpcResponse | JsonRpcNotification,
) -> dict[str, Any]:
    """Best-effort serialization fallback for messages built in-memory."""

    if isinstance(msg, JsonRpcRequest):
        out: dict[str, Any] = {"jsonrpc": "2.0", "id": msg.id, "method": msg.method}
        if msg.params is not None:
            out["params"] = msg.params
        return out
    if isinstance(msg, JsonRpcNotification):
        out = {"jsonrpc": "2.0", "method": msg.method}
        if msg.params is not None:
            out["params"] = msg.params
        return out
    # Response.
    out = {"jsonrpc": "2.0", "id": msg.id}
    if msg.error is not None:
        out["error"] = msg.error.to_dict()
    else:
        out["result"] = msg.result
    return out


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_message(buf: bytes) -> JsonRpcMessage:
    """Parse raw JSON-RPC bytes into a typed dataclass.

    Returns one of ``JsonRpcRequest`` / ``JsonRpcResponse`` /
    ``JsonRpcNotification``. The original ``buf`` is preserved as
    ``raw_message`` on the result, and the original ``params`` / ``result``
    sub-bytes (when present) are preserved as ``raw_params`` /
    ``raw_result``. Callers that care about byte-fidelity (prefix-cache
    stability) replay the raw fields verbatim instead of re-serializing.

    Rejects JSON-RPC 2.0 batch arrays at the framing boundary with
    ``JsonRpcParseError`` whose ``.code`` is ``INVALID_REQUEST``
    (-32600). Per the Streamable HTTP RFD batches are explicitly
    forbidden over HTTP, and we apply the rule consistently across all
    transports so callers don't have to special-case the wire.
    """

    try:
        # Decode strictly as UTF-8.
        text = buf.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JsonRpcParseError(f"non-UTF-8 message body: {exc}") from exc

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JsonRpcParseError(f"malformed JSON: {exc.msg}") from exc

    if isinstance(obj, list):
        raise JsonRpcParseError(
            "JSON-RPC batch arrays are not supported",
            code=JsonRpcErrorCode.INVALID_REQUEST,
        )

    if not isinstance(obj, dict):
        raise JsonRpcParseError(
            f"top-level JSON-RPC message must be an object, got {type(obj).__name__}",
            code=JsonRpcErrorCode.INVALID_REQUEST,
        )

    jsonrpc = obj.get("jsonrpc")
    if jsonrpc != "2.0":
        raise JsonRpcParseError(
            f"unsupported jsonrpc version: {jsonrpc!r} (must be '2.0')",
            code=JsonRpcErrorCode.INVALID_REQUEST,
        )

    has_id = "id" in obj
    has_method = "method" in obj
    has_result = "result" in obj
    has_error = "error" in obj

    # Preserve raw sub-bytes for params / result.
    raw_params = _extract_raw_field(buf, "params") if "params" in obj else None
    raw_result = _extract_raw_field(buf, "result") if has_result else None

    if has_method:
        method = obj["method"]
        if not isinstance(method, str):
            raise JsonRpcParseError(
                f"method must be a string, got {type(method).__name__}",
                code=JsonRpcErrorCode.INVALID_REQUEST,
            )
        params = obj.get("params")
        if has_id:
            req_id = obj["id"]
            if not isinstance(req_id, (str, int)) or isinstance(req_id, bool):
                raise JsonRpcParseError(
                    "id must be a string or integer (or null on response)",
                    code=JsonRpcErrorCode.INVALID_REQUEST,
                )
            return JsonRpcRequest(
                id=req_id,
                method=method,
                params=params,
                raw_params=raw_params,
                raw_message=buf,
            )
        return JsonRpcNotification(
            method=method,
            params=params,
            raw_params=raw_params,
            raw_message=buf,
        )

    if has_result or has_error:
        # Responses must have an `id` field (may be null per spec for
        # parse-error responses).
        if not has_id:
            raise JsonRpcParseError(
                "response missing 'id' field",
                code=JsonRpcErrorCode.INVALID_REQUEST,
            )
        if has_result and has_error:
            raise JsonRpcParseError(
                "response must not have both 'result' and 'error'",
                code=JsonRpcErrorCode.INVALID_REQUEST,
            )
        resp_id = obj["id"]
        if resp_id is not None and (not isinstance(resp_id, (str, int)) or isinstance(resp_id, bool)):
            raise JsonRpcParseError(
                "id must be a string, integer, or null",
                code=JsonRpcErrorCode.INVALID_REQUEST,
            )
        if has_error:
            err = obj["error"]
            if not isinstance(err, dict) or "code" not in err or "message" not in err:
                raise JsonRpcParseError(
                    "error must be an object with code and message",
                    code=JsonRpcErrorCode.INVALID_REQUEST,
                )
            return JsonRpcResponse(
                id=resp_id,
                error=JsonRpcError(
                    code=int(err["code"]),
                    message=str(err["message"]),
                    data=err.get("data"),
                ),
                raw_message=buf,
            )
        return JsonRpcResponse(
            id=resp_id,
            result=obj.get("result"),
            raw_result=raw_result,
            raw_message=buf,
        )

    raise JsonRpcParseError(
        "JSON-RPC message must have either 'method' or 'result'/'error'",
        code=JsonRpcErrorCode.INVALID_REQUEST,
    )


def _extract_raw_field(buf: bytes, field_name: str) -> bytes | None:
    """Return the byte slice of ``buf`` corresponding to the JSON value
    of ``field_name`` at the top level, or ``None`` if the field is
    absent or extraction fails.

    This is a best-effort byte-fidelity helper — it scans the bytes for
    the field key and uses the standard library ``json.JSONDecoder.raw_decode``
    to find the end of the value. Falls back to ``None`` (and callers
    must accept that re-serialization may be required) on any parse
    snag, never raises.
    """

    needle = f'"{field_name}"'.encode()
    idx = buf.find(needle)
    if idx < 0:
        return None
    # Skip past the key, the colon, and any whitespace.
    pos = idx + len(needle)
    n = len(buf)
    while pos < n and buf[pos : pos + 1] in (b" ", b"\t", b"\n", b"\r"):
        pos += 1
    if pos >= n or buf[pos : pos + 1] != b":":
        return None
    pos += 1
    while pos < n and buf[pos : pos + 1] in (b" ", b"\t", b"\n", b"\r"):
        pos += 1
    if pos >= n:
        return None
    try:
        decoder = json.JSONDecoder()
        # raw_decode wants str.
        text = buf[pos:].decode("utf-8")
        _, end = decoder.raw_decode(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    # Map ``end`` (string offset) back to a byte offset. UTF-8 decoding
    # already normalized the bytes; re-encode the slice to preserve
    # exact byte content for the value.
    return text[:end].encode("utf-8")


__all__ = [
    "ACP_STREAM_LIMIT",
    "JsonRpcError",
    "JsonRpcErrorCode",
    "JsonRpcFramingError",
    "JsonRpcMessage",
    "JsonRpcNotification",
    "JsonRpcParseError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "parse_message",
    "read_message",
    "write_message",
]
