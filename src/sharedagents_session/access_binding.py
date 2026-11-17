"""The grants a session's turns currently run under: ``<sid>.access.json``.

A Web user's prompt carries their grants (:class:`AccessStamp`). The turn
binds them here after taking the session lock, so every consumer in any
process — the tool gates, the ``Skill`` tool runner, delegation, the queue —
reads one file. No file means unrestricted (an operator's session).

* a restricted stamp writes the file; an unrestricted stamp (an admin's
  turn) deletes it — or, when a parent's restriction would still be
  inherited, records an explicit unrestricted binding; no stamp (TUI,
  Telegram, wake-ups, background results) leaves it as it is;
* a session with no file of its own inherits its parent's (walked through
  the immutable ``session_started`` header); a delegated child's turn is
  handed its parent turn's stamp and binds it, so the child keeps those
  limits after the parent moves on;
* an unreadable file fails closed (no grants);
* a running turn keeps the grants it bound: the stamp is pinned for the
  turn's own context (and handed to the processes it spawns through
  ``SHAREDAGENTS_ACCESS_PIN``), so a later turn re-binding the same session — in
  another process, or through another ``Agent`` object such as a delegation
  resuming it — never widens (or narrows) a turn already running.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from sharedagents_core.access_grants import AccessStamp, Grants

SIDECAR_SUFFIX = ".access.json"
_SID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,200}")
_MAX_DEPTH = 16
_PARENTS: dict[tuple[str, str], str] = {}

# The running turn's grants, for code that composes per-turn text without a
# session id (the skills catalogue in the system prompt). Set by the turn
# right after binding; ``None`` = unrestricted.
_TURN_GRANTS: ContextVar[Grants | None] = ContextVar("sharedagents_turn_grants", default=None)

# Running turns' stamps, ``((sid, stamp-or-None), …)`` (``None`` =
# unrestricted), innermost last. Readers in the turn's context (its rounds,
# tool gates, in-process delegation) see its pin instead of the shared file.
_PINS: ContextVar[tuple] = ContextVar("sharedagents_access_pins", default=())

# The same pin for a process the turn spawns (tool runner, subagent runner):
# ``{"sid": …, "stamp": {…} | null}``.
ACCESS_PIN_ENV = "SHAREDAGENTS_ACCESS_PIN"


def set_turn_grants(grants: Grants | None) -> None:
    _TURN_GRANTS.set(grants)


def turn_grants() -> Grants | None:
    return _TURN_GRANTS.get()


def sidecar_path(sessions_dir, sid: str) -> Path:
    sid = str(sid or "")
    if not _SID_RE.fullmatch(sid) or sid in (".", "..") or "/" in sid:
        raise ValueError(f"invalid session id: {sid!r}")
    return Path(sessions_dir) / f"{sid}{SIDECAR_SUFFIX}"


def _read(path: Path) -> AccessStamp | None:
    """The stamp in ``path``; ``None`` when absent; no grants when unreadable."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return AccessStamp("", Grants.none())
    try:
        stamp = AccessStamp.from_obj(json.loads(raw))
    except (TypeError, ValueError):
        stamp = None
    return stamp if stamp is not None else AccessStamp("", Grants.none())


def _write(path: Path, stamp: AccessStamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(stamp.to_obj(), fh, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _parent_of(sessions_dir, sid: str) -> str:
    key = (str(sessions_dir), sid)
    if key in _PARENTS:
        return _PARENTS[key]
    parent = ""
    try:
        from sharedagents_session.session_reader import tail_jsonl

        for event, _offset in tail_jsonl(Path(sessions_dir) / f"{sid}.jsonl", 0):
            if event.get("type") == "session_started":
                parent = str(event.get("parent_session_id") or "")
            break
    except (OSError, TypeError, ValueError):
        return ""
    _PARENTS[key] = parent  # a header never changes
    if len(_PARENTS) > 4096:
        _PARENTS.clear()
    return parent


def access_pin_env_value(sid: str, stamp: AccessStamp | None) -> str:
    """``ACCESS_PIN_ENV`` value pinning ``stamp`` (``None`` = unrestricted)."""
    return json.dumps(
        {"sid": str(sid), "stamp": stamp.to_obj() if stamp is not None else None},
        sort_keys=True,
    )


def _pinned(sid: str) -> tuple[bool, AccessStamp | None]:
    """``(True, stamp)`` when a running turn pinned ``sid`` for this context."""
    for pinned_sid, stamp in reversed(_PINS.get()):
        if pinned_sid == sid:
            return True, stamp
    raw = os.environ.get(ACCESS_PIN_ENV) or ""
    if raw:
        try:
            obj = json.loads(raw)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and obj.get("sid") == sid:
            if obj.get("stamp") is None:
                return True, None
            stamp = AccessStamp.from_obj(obj.get("stamp"))
            return True, stamp if stamp is not None else AccessStamp("", Grants.none())
    return False, None


@contextmanager
def pinned_turn_access(sid: str, stamp: AccessStamp | None) -> Iterator[None]:
    """Pin ``stamp`` for ``sid`` in this context while the block runs."""
    token = _PINS.set(_PINS.get() + ((str(sid), _effective(stamp)),))
    try:
        yield
    finally:
        _reset(token)


def apply_access_pin_env(env: dict, sid: str) -> None:
    """Hand a process spawned for ``sid``'s turn that turn's pinned stamp, or
    drop an inherited pin when there is none (it would belong to another
    turn)."""
    found, stamp = _pinned(str(sid or ""))
    if found:
        env[ACCESS_PIN_ENV] = access_pin_env_value(str(sid), stamp)
    else:
        env.pop(ACCESS_PIN_ENV, None)


def _effective(stamp: AccessStamp | None) -> AccessStamp | None:
    return stamp if stamp is not None and stamp.restricted else None


def _reset(token) -> None:
    try:
        _PINS.reset(token)
    except (ValueError, RuntimeError):
        pass  # finalized from another context: that context is going away


def session_access_stamp(
    sessions_dir,
    sid: str,
    *,
    pins: bool = True,
) -> AccessStamp | None:
    """The stamp bound to ``sid`` or its nearest ancestor (``None`` = none).

    A running turn's pin wins over the file (``pins=False`` reads the files
    only — what the next turn inherits).
    """
    cur, seen = str(sid or ""), set()
    for _ in range(_MAX_DEPTH):
        if not cur or cur in seen:
            return None
        seen.add(cur)
        if pins:
            found, stamp = _pinned(cur)
            if found:
                return stamp
        try:
            stamp = _read(sidecar_path(sessions_dir, cur))
        except ValueError:
            return None
        if stamp is not None:
            return stamp
        cur = _parent_of(sessions_dir, cur)
    return None


def session_access_grants(sessions_dir, sid: str) -> Grants | None:
    """Grants the session's turns run under; ``None`` = unrestricted."""
    stamp = session_access_stamp(sessions_dir, sid)
    if stamp is None or not stamp.restricted:
        return None
    return stamp.grants


def _bind(sessions_dir, sid: str, stamp: AccessStamp | None) -> AccessStamp | None:
    """Write ``stamp`` (see module doc); the restricted stamp now in force."""
    if stamp is None:
        return _effective(session_access_stamp(sessions_dir, sid, pins=False))
    path = sidecar_path(sessions_dir, sid)
    if stamp.restricted:
        _write(path, stamp)
        return stamp
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    inherited = session_access_stamp(sessions_dir, sid, pins=False)
    if inherited is not None and inherited.restricted:
        _write(path, AccessStamp(stamp.email, Grants.all()))
    return None


def bind_session_access(
    sessions_dir,
    sid: str,
    stamp: AccessStamp | None,
) -> Grants | None:
    """Bind a turn's stamp (see module doc); returns the grants now in force.

    An unrestricted stamp removes this session's own binding and, when a
    restriction would still be inherited from a parent, records an explicit
    unrestricted binding so an admin's turn in a restricted child really is
    unrestricted.
    """
    bound = _bind(sessions_dir, sid, stamp)
    return bound.grants if bound is not None else None


class TurnAccessPin:
    """A turn's binding, pinned for as long as the turn owns its session.

    ``async with lock, TurnAccessPin() as pin:`` then ``pin.bind(…)`` under
    the lock; leaving the block releases the pin.
    """

    def __init__(self) -> None:
        self._token = None

    async def __aenter__(self) -> TurnAccessPin:
        return self

    async def __aexit__(self, *_exc) -> bool:
        self.release()
        return False

    def bind(self, sessions_dir, sid: str, stamp: AccessStamp | None) -> Grants | None:
        bound = _bind(sessions_dir, sid, stamp)
        self.release()
        self._token = _PINS.set(_PINS.get() + ((str(sid), bound),))
        return bound.grants if bound is not None else None

    def release(self) -> None:
        token, self._token = self._token, None
        if token is not None:
            _reset(token)


def copy_session_access(sessions_dir, parent_sid: str, child_sid: str) -> Grants | None:
    """Give a delegated child its parent's grants for the child's lifetime.

    An unrestricted parent resuming a child that still carries an older
    restriction clears it (explicitly, so no ancestor binding leaks in).
    """
    stamp = session_access_stamp(sessions_dir, parent_sid)
    child_path = sidecar_path(sessions_dir, child_sid)
    if stamp is None or not stamp.restricted:
        current = session_access_stamp(sessions_dir, child_sid)
        if current is not None and current.restricted:
            _write(child_path, AccessStamp(stamp.email if stamp else "", Grants.all()))
        return None
    _write(child_path, stamp)
    return stamp.grants


# Internal runtime profiles (context compaction, titles, summaries, the goal
# teacher) are infrastructure, not agents a person picks: never refused.
RUNTIME_AGENTS = frozenset({"compaction", "title", "summary", "teacher"})

# The one direct-dispatch tool a restricted session may trigger: ``/compact``
# (context compaction). Every other ``__SHAREDAGENTS_DELEGATE__`` target would
# skip the model and the tool gate.
DIRECT_DISPATCH_ALWAYS = frozenset({"compact_context"})


def direct_dispatch_allowed(
    sessions_dir,
    sid: str,
    direct,
    *,
    incoming: AccessStamp | None = None,
) -> bool:
    """Whether a parsed direct-dispatch ``{"tool": …}`` may run in ``sid``.

    ``incoming`` is the submit's own stamp: when present it decides (the
    session may still be bound to the previous turn's submitter); a submit
    with no stamp runs under whatever the session is bound to.
    """
    tool = str((direct or {}).get("tool") or "")
    if tool in DIRECT_DISPATCH_ALWAYS:
        return True
    if incoming is not None:
        return not incoming.restricted
    return session_access_grants(sessions_dir, sid) is None
