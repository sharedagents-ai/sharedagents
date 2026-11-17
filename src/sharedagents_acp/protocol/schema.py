"""Schema-model re-exports from the official ``agent-client-protocol`` SDK.

Schema source of truth: a pinned git SHA of
``agentclientprotocol/agent-client-protocol`` whose ``schema/schema.json``
is the authoritative reference (the docs-page summaries are not).

We pull the parsed Pydantic models out of the official SDK here so
callers grab them from one place. Swapping SDK versions becomes a
one-file change.

The pinned schema SHA is read from the repo's ``pyproject.toml``
``[tool.sharedagents-acp]`` table; ``SCHEMA_SHA`` exposes it as a
module-level constant so tests can assert SDK ↔ pin parity. Lookup is
lazy (only when this module is imported) so plain
``import sharedagents_acp`` does not pay the cost.

If the SDK isn't installed, ``import sharedagents_acp.protocol.schema``
raises ``ImportError`` with a clear message — client and server code
paths need the SDK; protocol-only callers should not import this
module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    # Standard-library tomllib (3.11+).
    import tomllib as _tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as _tomllib  # type: ignore[import-not-found,no-redef]


def _read_pinned_schema_sha() -> str | None:
    """Return the canonical schema SHA from ``pyproject.toml``, or
    ``None`` if it cannot be located.

    Walks up from this file looking for ``pyproject.toml``. Returns the
    value of ``[tool.sharedagents-acp].schema_sha`` if present.
    """

    cur: Path | None = Path(__file__).resolve()
    while cur is not None and cur.parent != cur:
        candidate = cur / "pyproject.toml"
        if candidate.is_file():
            try:
                with candidate.open("rb") as fh:
                    data = _tomllib.load(fh)
            except (OSError, ValueError):
                return None
            tool = data.get("tool", {}) if isinstance(data, dict) else {}
            block = tool.get("sharedagents-acp", {}) if isinstance(tool, dict) else {}
            if isinstance(block, dict) and isinstance(block.get("schema_sha"), str):
                return block["schema_sha"]
            return None
        cur = cur.parent
    return None


SCHEMA_SHA: str | None = _read_pinned_schema_sha()
"""The canonical ACP schema git SHA pinned in ``pyproject.toml``.

``None`` if the file isn't reachable (e.g. running from an installed
wheel without the source tree). Callers that need a hard guarantee
should assert ``SCHEMA_SHA is not None`` themselves.
"""


def _import_sdk() -> Any:
    try:
        import acp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "sharedagents_acp.protocol.schema requires the official "
            "`agent-client-protocol` package. Install it via "
            "`pip install agent-client-protocol`."
        ) from exc
    return acp


_acp = _import_sdk()


# Method/version constants
PROTOCOL_VERSION = _acp.PROTOCOL_VERSION
AGENT_METHODS = _acp.AGENT_METHODS
CLIENT_METHODS = _acp.CLIENT_METHODS

# Re-export the entire SDK module under ``schema_models`` so callers
# can grab any model the SDK defines (every Request/Response/Notification,
# ContentBlock variant, capability dataclass, StopReason Literal, etc.)
# without us having to enumerate them. Use the SDK's models directly;
# this module is the canonical front door.
schema_models = _acp


__all__ = [
    "AGENT_METHODS",
    "CLIENT_METHODS",
    "PROTOCOL_VERSION",
    "SCHEMA_SHA",
    "schema_models",
]
