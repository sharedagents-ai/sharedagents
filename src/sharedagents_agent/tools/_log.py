"""Structured gateway log writer.

Every event from the runtime is appended to the ONE shared
``layout.gateway_logfile()`` (``~/.sharedagents/logs/gateway.log``) as a
single JSONL line carrying explicit origin tags — the calling agent
(``SHAREDAGENTS_AGENT_NAME``), the service (``gateway``), and any
session_id / round_idx / turn_idx the caller passes. The same line
shape is used by heartbeat / tui / telegram writers via
:mod:`sharedagents_core.jsonlog`, so a single ``jq`` query (or grep on
``"agent":"main"``) selects per-agent slices from the shared file.

Example line:

    {"ts":"2026-05-26T18:21:09.842+00:00","service":"gateway",
     "agent":"main","level":"INFO","msg":"turn_metrics",
     "kind":"TURN_METRICS",
     "session_id":"<session-id>",
     "round_idx":3,"turn_idx":1,
     "k":{"finish_reason":"stop","input_tokens":1402,
          "output_tokens":88}}

Backwards-compat: the public surface is ``log_event(event, **fields)``
— historical call sites pass an EVENT tag and free kwargs. We map
that onto :meth:`JsonLogWriter.write` by putting the EVENT into
``kind`` and the free kwargs into ``k``. Best-effort: any I/O error
is swallowed so a logging failure can never break a tool call.
"""

from __future__ import annotations

from typing import Any

from sharedagents_core.jsonlog import get_writer
from sharedagents_core.layout import current_agent_name, gateway_logfile

# Keys that the unified JSONL format promotes to top-level columns.
# Any kwarg on ``log_event`` matching one of these is hoisted out of
# the free ``k`` payload so log queries can filter on the structured
# field instead of digging through nested JSON.
_PROMOTED_FIELDS: tuple[str, ...] = (
    "session_id",
    "parent_session_id",
    "round_idx",
    "turn_idx",
    "level",
    "msg",
)


def log_event(event: str, **fields: Any) -> None:
    """Append a structured event to the unified gateway log.

    ``event`` becomes the line's ``kind`` (matches the legacy
    ``EVENT`` token). Recognised structural kwargs (session_id,
    round_idx, turn_idx, level, msg) are hoisted into top-level
    columns; everything else lands under the line's ``k`` payload
    map.
    """
    try:
        # Caller-provided ``agent=`` (e.g. ``acp_bridge._log_event``
        # passes ``self.cfg.name``) wins over the env-derived default
        # — a single gateway process hosts multiple agents and each
        # event must be attributed to the agent that produced it, not
        # the process's ``SHAREDAGENTS_AGENT_NAME`` startup tag. Falls back
        # to the env, then ``"unknown"`` for ad-hoc / one-shot callers.
        explicit_agent = fields.pop("agent", None)
        agent = (str(explicit_agent).strip() if explicit_agent else "") or current_agent_name() or "unknown"
        level = str(fields.pop("level", "INFO")) or "INFO"
        msg = str(fields.pop("msg", "") or "") or event.lower()
        session_id = fields.pop("session_id", None)
        parent_session_id = fields.pop("parent_session_id", None)
        round_idx = fields.pop("round_idx", None)
        turn_idx = fields.pop("turn_idx", None)
        w = get_writer(gateway_logfile(), service="gateway")
        w.write(
            agent=agent,
            level=level,
            msg=msg,
            kind=event,
            session_id=str(session_id) if session_id else None,
            parent_session_id=(str(parent_session_id) if parent_session_id else None),
            round_idx=int(round_idx) if round_idx is not None else None,
            turn_idx=int(turn_idx) if turn_idx is not None else None,
            k=dict(fields) if fields else None,
        )
    except OSError:
        # Logfile unwritable — silently drop. The session JSONL still
        # records the tool call; gateway.log is auxiliary telemetry.
        pass
    except ValueError:
        # Missing required field (agent/level/msg). Drop rather than
        # raise; logging must never break the tool call.
        pass


__all__ = ["log_event"]
