"""Session-lifecycle helper utilities extracted from ``turn_runtime``.

Small, dependency-light helpers used across the turn pipeline. Every
symbol here is re-exported from ``sharedagents_agent.turn_runtime`` so existing
``from sharedagents_agent.turn_runtime import ...`` call sites keep working.
"""

from __future__ import annotations

import secrets
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sharedagents_agent.agent import AgentConfig
    from sharedagents_session.session import Session


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _publish_turn_completed(
    sess: Session,
    user: Any,
    content: Any,
    cfg: AgentConfig,
) -> None:
    """Best-effort `turn_completed` publish on the in-process bus.

    Called from every site where a successful turn lands on disk
    (`sess.add_turn` followed by `sess.save`). Memory providers
    subscribe via `memory_manager.MemoryManager.wire_events()` and
    use the payload to drive `provider.sync_turn(user, assistant)`.
    No provider ships by default (file-only memory), so this
    only drives an external provider the operator registered.

    Capped at 5000 chars per side so a runaway turn can't push huge
    payloads onto the bus. List-shaped multi-part `user` values are
    flattened to their text parts only (vision images skipped).
    Failures here are swallowed: this is downstream notification, not
    part of the durable save contract.
    """
    try:
        if isinstance(user, list):
            user_text = "\n".join(
                str(p.get("text") or "")
                for p in user
                if isinstance(p, dict) and str(p.get("type") or "").lower() == "text"
            )
        else:
            user_text = str(user or "")
        assistant_text = str(content or "")
        from sharedagents_core.event_bus import get_bus

        get_bus().publish(
            "turn_completed",
            {
                "session_hash": sess.hash,
                "agent_home": str(cfg.home),
                "user": user_text[:5000],
                "assistant": assistant_text[:5000],
                "workspace_origin": sess.workspace_origin,
            },
        )
    except Exception as exc:  # noqa: BLE001
        # Downstream notification, not part of the durable save contract —
        # but route the turn-runtime crash boundary through structured
        # capture so the SessionError-hook machinery fires uniformly across
        # surfaces rather than only in the TUI. Kept
        # best-effort so a capture failure can never break the save caller.
        try:
            from sharedagents_core import errors as _errors

            _errors.log_error(
                "agent:turn_runtime",
                exc,
                summary="turn_completed publish failed",
                session_id=getattr(sess, "hash", None),
                agent_name=getattr(cfg, "name", None),
            )
        except Exception:  # noqa: BLE001
            pass


def _spawn_detached_restart(agent_name: str) -> int:
    """Fire off a detached child that waits ~1s, then calls
    `sharedagents agent restart <agent_name>`. Returns its PID.

    Used by the `/restart` slash command. The wait matters: the
    gateway's HTTP response must flush BEFORE `agent stop` SIGTERMs
    the daemon, otherwise the caller sees a connection reset and the
    Telegram bot reports "command failed". `start_new_session=True`
    detaches the helper so killing this process leaves it running.
    """
    import shlex
    import subprocess

    cmd = f"sleep 1; exec {shlex.quote(sys.executable)} -m sharedagents agent restart {shlex.quote(agent_name)}"
    proc = subprocess.Popen(  # noqa: S603
        ["/bin/sh", "-c", cmd],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return proc.pid


def _new_turn_id() -> str:
    return secrets.token_hex(6)


def _has_image(content: Any) -> bool:
    """True if an OpenAI-style content blob carries any image part.

    User content can be either:
      - a plain string (text only — never matches)
      - a list of multi-part dicts, each with `type` and a payload key.
        Vision payloads use `type=="image_url"` (Chat Completions) or
        `type=="input_image"` (Responses-API style). Both count as images.

    Anything else (None, dict, bytes, etc.) returns False so a
    malformed payload doesn't accidentally trigger a model swap.
    """
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").lower()
        if ptype in ("image_url", "input_image"):
            return True
    return False
