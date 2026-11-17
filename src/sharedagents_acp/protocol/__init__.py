"""Public surface of the ``sharedagents_acp.protocol`` submodule.

Re-exports everything the rest of the codebase should grab from one
place: JSON-RPC types and framing helpers, the transport ABC, and
(lazily, only when ``schema`` itself is touched) the SDK schema
models.

The schema re-export is **not** imported eagerly — JSON-RPC framing
callers don't need the SDK's ~1700 lines of Pydantic models. Callers
that want schema models import them explicitly via
``from sharedagents_acp.protocol.schema import ...``.
"""

from __future__ import annotations

from .jsonrpc import (
    JsonRpcError,
    JsonRpcErrorCode,
    JsonRpcFramingError,
    JsonRpcMessage,
    JsonRpcNotification,
    JsonRpcParseError,
    JsonRpcRequest,
    JsonRpcResponse,
    parse_message,
    read_message,
    write_message,
)
from .transport import StdioTransport, Transport

__all__ = [
    "JsonRpcError",
    "JsonRpcErrorCode",
    "JsonRpcFramingError",
    "JsonRpcMessage",
    "JsonRpcNotification",
    "JsonRpcParseError",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "StdioTransport",
    "Transport",
    "parse_message",
    "read_message",
    "write_message",
]
