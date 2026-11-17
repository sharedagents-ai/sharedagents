"""Frozen session-JSONL wire protocol.

The on-disk line codec and the required keys of known event types are
immutable. New event types and extra keys on existing types are additive
features. Renaming, removing, or retyping a required field is a protocol
break and is refused at write time.

Readers stay tolerant: they skip a bad line rather
than raise. This module gates the *writer* so a bad line is not produced
again.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Pinned json.dumps kwargs. Changing these changes every on-disk CRC.
CODEC_DUMP_KWARGS: dict[str, Any] = {
    "sort_keys": True,
    "ensure_ascii": False,
    "separators": (",", ":"),
}

# Historical aliases: copy onto the canonical name, keep the old key
# (additive). Never delete keys.
_FIELD_ALIASES: dict[str, tuple[tuple[str, str], ...]] = {
    "text_delta": (("chunk", "delta"), ("text", "delta")),
    "thinking_delta": (("chunk", "delta"), ("text", "delta")),
}

# Every session event, including new types, carries these.
REQUIRED_COMMON: frozenset[str] = frozenset({"type", "v", "ts"})

# Required keys *in addition to* REQUIRED_COMMON. Types not listed here
# are new/additive and only need REQUIRED_COMMON.
REQUIRED_BY_TYPE: dict[str, frozenset[str]] = {
    "text_delta": frozenset({"turn_idx", "delta"}),
    "thinking_delta": frozenset({"turn_idx", "delta"}),
    "turn_started": frozenset({"turn_idx", "user"}),
    "turn_ended": frozenset({"turn_idx"}),
    "round_started": frozenset({"turn_idx", "round_idx"}),
    "round_ended": frozenset({"turn_idx", "round_idx"}),
    "tool_call": frozenset({"turn_idx", "call_id", "name"}),
    "tool_result": frozenset({"turn_idx", "call_id"}),
    "tool_result_meta": frozenset({"turn_idx", "call_id"}),
    "session_started": frozenset(),
    "header_update": frozenset(),
    "session_closed": frozenset(),
}


class JsonlProtocolError(ValueError):
    """A session event is missing a frozen required field."""


def apply_field_aliases(event: dict[str, Any]) -> dict[str, Any]:
    """Copy known historical aliases onto canonical names; never drop keys."""
    etype = str(event.get("type") or "")
    for src, dst in _FIELD_ALIASES.get(etype, ()):
        if dst not in event and src in event:
            event[dst] = event[src]
    return event


def required_keys_for(event_type: str) -> frozenset[str]:
    extra = REQUIRED_BY_TYPE.get(str(event_type), frozenset())
    return REQUIRED_COMMON | extra


def prepare_session_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize aliases and refuse a write that would break the frozen schema.

    Extra keys pass through unchanged. Unknown ``type`` values are allowed
    (additive) as long as ``type`` / ``v`` / ``ts`` are present.
    """
    if not isinstance(event, dict):
        raise JsonlProtocolError("session jsonl event must be a dict")
    out = dict(event)
    if not str(out.get("type") or ""):
        raise JsonlProtocolError("session jsonl event missing type")
    apply_field_aliases(out)
    missing = sorted(k for k in required_keys_for(str(out["type"])) if k not in out)
    if missing:
        raise JsonlProtocolError(f"session jsonl {out.get('type')!r} missing required fields: " + ", ".join(missing))
    return out
