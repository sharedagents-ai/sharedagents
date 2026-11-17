"""Persistent WhatsApp channel state (``state.json``).

Holds, per chat, the active agent and one session id per agent (so switching
agents and back resumes the earlier conversation), the canonical-stream cursor
per session (so a restart resumes delivery where it stopped), and a bounded list
of processed WhatsApp message ids (so a redelivered message never runs twice).
v1 has a single chat, :data:`SELF_CHAT`; the layout is keyed for more.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

SELF_CHAT = "self"
MAX_PROCESSED_IDS = 500
_VERSION = 2

_log = logging.getLogger("sharedagents.whatsapp.state")


class ChannelState:
    """Chat → agent/session bindings, stream cursors and dedupe ids."""

    def __init__(self, path: Path, default_agent: str) -> None:
        self.path = Path(path)
        self.default_agent = default_agent
        self._chats: dict[str, dict[str, Any]] = {}
        self._cursors: dict[str, int] = {}
        # Last allowlist-routed model handled for each session. This lets a
        # live Channels-tab edit reconfigure an existing WhatsApp session
        # exactly once, while avoiding a /model command before every prompt.
        self._models: dict[str, str] = {}
        self._processed: list[str] = []
        self._processed_set: set[str] = set()

    @classmethod
    def load(cls, path: Path, default_agent: str) -> ChannelState:
        """Read ``path``; a missing file is an empty state, a corrupt one is set aside."""
        state = cls(path, default_agent)
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return state
        except (OSError, ValueError) as exc:
            aside = Path(f"{path}.corrupt-{int(time.time())}")
            with contextlib.suppress(OSError):
                os.replace(path, aside)
            _log.warning("state %s unreadable (%s); starting fresh, kept %s", path, exc, aside)
            return state
        if not isinstance(raw, dict):
            return state
        chats = raw.get("chats")
        if isinstance(chats, dict):
            for chat, entry in chats.items():
                if not isinstance(entry, dict):
                    continue
                sessions = entry.get("sessions")
                state._chats[str(chat)] = {
                    **({"agent": str(entry["agent"])} if entry.get("agent") else {}),
                    "sessions": {
                        str(a): str(s) for a, s in (sessions.items() if isinstance(sessions, dict) else ()) if a and s
                    },
                }
        cursors = raw.get("cursors")
        if isinstance(cursors, dict):
            state._cursors = {
                str(sid): int(off)
                for sid, off in cursors.items()
                if isinstance(off, int) and not isinstance(off, bool) and off >= 0
            }
        models = raw.get("models")
        if isinstance(models, dict):
            state._models = {
                str(sid): str(model)
                for sid, model in models.items()
                if sid and isinstance(model, str) and model.strip()
            }
        processed = raw.get("processed")
        if isinstance(processed, list):
            for msg_id in processed:
                if isinstance(msg_id, str) and msg_id:
                    state.mark_processed(msg_id)
        return state

    def explicit_agent_for(self, chat: str) -> str | None:
        """The explicitly switched agent for ``chat``, if any."""
        agent = self._chats.get(chat, {}).get("agent")
        return str(agent) if agent else None

    def agent_for(self, chat: str) -> str:
        return str(self._chats.get(chat, {}).get("agent") or self.default_agent)

    def set_agent(self, chat: str, agent: str) -> None:
        self._chats.setdefault(chat, {"sessions": {}})["agent"] = agent

    def session_for(self, chat: str, agent: str | None = None) -> str | None:
        sessions = self._chats.get(chat, {}).get("sessions", {})
        sid = sessions.get(agent or self.agent_for(chat))
        return str(sid) if sid else None

    def bind_session(self, chat: str, agent: str, sid: str) -> None:
        self._chats.setdefault(chat, {"sessions": {}}).setdefault("sessions", {})[agent] = sid
        self._cursors.setdefault(sid, 0)

    def unbind_session(self, chat: str, agent: str | None = None) -> str | None:
        sessions = self._chats.get(chat, {}).get("sessions", {})
        sid = sessions.pop(agent or self.agent_for(chat), None)
        if sid:
            self._cursors.pop(sid, None)
            self._models.pop(sid, None)
        return sid

    def model_for_session(self, sid: str) -> str:
        """The route last applied, defaulted, or explicitly overridden for ``sid``."""
        return self._models.get(sid, "")

    def set_session_model(self, sid: str, model: str) -> None:
        """Record the handled allowlist model, so unchanged routes stay consumed."""
        model = str(model or "").strip()
        if model:
            self._models[sid] = model
        else:
            self._models.pop(sid, None)

    def cursor_for(self, sid: str) -> int:
        return int(self._cursors.get(sid, 0))

    def set_cursor(self, sid: str, offset: int) -> None:
        """Advance ``sid``'s stream cursor; it never moves backwards."""
        if offset > self._cursors.get(sid, 0):
            self._cursors[sid] = int(offset)

    def is_processed(self, msg_id: str) -> bool:
        return msg_id in self._processed_set

    def mark_processed(self, msg_id: str) -> None:
        if msg_id in self._processed_set:
            return
        self._processed.append(msg_id)
        self._processed_set.add(msg_id)
        while len(self._processed) > MAX_PROCESSED_IDS:
            self._processed_set.discard(self._processed.pop(0))

    def current_bindings(self) -> list[tuple[str, str, str]]:
        """``(chat, agent, sid)`` for each chat's CURRENT agent that has a session."""
        out: list[tuple[str, str, str]] = []
        for chat in self._chats:
            agent = self.agent_for(chat)
            sid = self.session_for(chat, agent)
            if sid:
                out.append((chat, agent, sid))
        return out

    def all_bindings(self) -> list[tuple[str, str, str]]:
        """``(chat, agent, sid)`` for EVERY bound session, not only the current agent's.

        Each is observed: a delegated job finishing in an agent's session after
        the operator switched agents must still reach WhatsApp.
        """
        out: list[tuple[str, str, str]] = []
        for chat, entry in self._chats.items():
            for agent, sid in entry.get("sessions", {}).items():
                if sid:
                    out.append((chat, str(agent), str(sid)))
        return out

    def agent_of(self, chat: str, sid: str) -> str | None:
        """The agent whose session in ``chat`` is ``sid``, if any."""
        for agent, bound in self._chats.get(chat, {}).get("sessions", {}).items():
            if bound == sid:
                return str(agent)
        return None

    def save(self) -> None:
        """Write atomically (temp file + fsync + rename), owner-only (0600)."""
        payload = {
            "version": _VERSION,
            "chats": self._chats,
            "cursors": self._cursors,
            "models": self._models,
            "processed": self._processed,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
