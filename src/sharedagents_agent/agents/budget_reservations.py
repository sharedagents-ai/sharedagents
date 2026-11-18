"""Durable per-wave budget reservations for delegated children.

The session cost cap is the real limit, but two concurrent dispatchers read
the same "remaining" figure before either child has spent anything, so a wave
of frontier children can over-commit the budget. This module holds a
short-lived reservation per child/wave in a per-parent JSON file, guarded by
an exclusive platform lock: the second dispatcher sees the first one's claim and is
refused.

A reservation is admission control, not a spend cap. Actual spend still lands
in the usage ledger; ``release`` drops the claim when the child turn returns
(or the detached runner exits), and entries older than ``DEFAULT_TTL_S`` are
pruned so a crashed runner cannot wedge admission forever.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sharedagents_core.oscompat import file_lock

DEFAULT_TTL_S = 6 * 3600


def parse_budget_usd(value: Any) -> float:
    """Parse the checklist's ``budget_usd`` (number or ``"$6.00"``)."""
    text = str(value if value is not None else "").strip().lstrip("$").strip()
    try:
        return max(0.0, float(text))
    except (TypeError, ValueError):
        return 0.0


def _sessions_dir(sessions_dir: Any = None) -> Path:
    if sessions_dir:
        return Path(sessions_dir)
    from sharedagents_core.layout import sessions_root

    return Path(sessions_root())


def _path(parent_sid: str, sessions_dir: Any = None) -> Path:
    return _sessions_dir(sessions_dir) / f"{parent_sid}.budget_reservations.json"


@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with open(lock, "a+") as fh, file_lock(fh):
        yield


def _load(path: Path, *, now: float, ttl_s: int) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        try:
            created = float(entry.get("created_at") or 0)
            usd = float(entry.get("usd") or 0)
        except (TypeError, ValueError):
            continue
        if created and now - created > ttl_s:
            continue
        if usd <= 0:
            continue
        out[str(key)] = {**entry, "usd": usd, "created_at": created or now}
    return out


def _save(path: Path, data: dict[str, dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def active_reserved_usd(parent_sid: str, *, sessions_dir: Any = None, ttl_s: int = DEFAULT_TTL_S) -> float:
    """USD currently reserved by live children of ``parent_sid``."""
    if not parent_sid:
        return 0.0
    path = _path(parent_sid, sessions_dir)
    try:
        with _locked(path):
            return sum(e["usd"] for e in _load(path, now=time.time(), ttl_s=ttl_s).values())
    except OSError:
        return 0.0


@dataclass
class Reservation:
    parent_sid: str
    key: str
    usd: float
    sessions_dir: Any = None
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        path = _path(self.parent_sid, self.sessions_dir)
        try:
            with _locked(path):
                data = _load(path, now=time.time(), ttl_s=DEFAULT_TTL_S)
                data.pop(self.key, None)
                _save(path, data)
        except OSError:
            pass


def can_admit(parent_sid: str, usd: float, *, sessions_dir: Any = None, ttl_s: int = DEFAULT_TTL_S) -> bool:
    """Read-only admission check under the same lock as ``reserve_many``."""
    if not parent_sid or usd <= 0:
        return True
    path = _path(parent_sid, sessions_dir)
    try:
        with _locked(path):
            data = _load(path, now=time.time(), ttl_s=ttl_s)
            reserved = sum(e["usd"] for e in data.values())
            from sharedagents_goal.goal_service import remaining_for_session_hash

            return float(remaining_for_session_hash(parent_sid)) - reserved >= usd
    except OSError:
        return True


def reserve_many(
    parent_sid: str,
    amounts: Iterable[float],
    *,
    keys: Iterable[str] | None = None,
    sessions_dir: Any = None,
    ttl_s: int = DEFAULT_TTL_S,
) -> list[Reservation] | None:
    """Atomically admit every amount or none.

    Returns the reservations (empty for a zero-amount request) or ``None``
    when the session cannot cover them after its live reservations.
    """
    if not parent_sid:
        return []
    amounts = [max(0.0, float(a or 0)) for a in amounts]
    if not any(amounts):
        return []
    key_list = list(keys or [])
    if len(key_list) < len(amounts):
        key_list += [f"res_{uuid.uuid4().hex}" for _ in range(len(amounts) - len(key_list))]
    path = _path(parent_sid, sessions_dir)
    now = time.time()
    try:
        with _locked(path):
            data = _load(path, now=now, ttl_s=ttl_s)
            reserved = sum(e["usd"] for e in data.values())
            from sharedagents_goal.goal_service import remaining_for_session_hash

            remaining = float(remaining_for_session_hash(parent_sid))
            if remaining - reserved < sum(amounts):
                return None
            out: list[Reservation] = []
            for key, usd in zip(key_list, amounts):
                if usd <= 0:
                    continue
                data[key] = {"usd": usd, "created_at": now}
                out.append(Reservation(parent_sid, key, usd, sessions_dir))
            _save(path, data)
            return out
    except OSError:
        # A reservation store that cannot be written must not wedge dispatch;
        # the session cap still bounds the spend.
        return []


def reserve(
    parent_sid: str, usd: float, *, key: str | None = None, sessions_dir: Any = None, ttl_s: int = DEFAULT_TTL_S
) -> Reservation | None:
    out = reserve_many(
        parent_sid,
        [usd],
        keys=[key or f"res_{uuid.uuid4().hex}"],
        sessions_dir=sessions_dir,
        ttl_s=ttl_s,
    )
    if out is None:
        return None
    return out[0] if out else None


def refusal_message(parent_sid: str, usd: float, *, sessions_dir: Any = None) -> str:
    """Actionable refusal copy for a wave that does not fit."""
    from sharedagents_goal.goal_service import remaining_for_session_hash

    try:
        remaining = float(remaining_for_session_hash(parent_sid))
    except Exception:  # noqa: BLE001
        remaining = 0.0
    reserved = active_reserved_usd(parent_sid, sessions_dir=sessions_dir)
    return (
        f"[sharedagents budget] this dispatch asks for ${usd:.2f}, but the session "
        f"has ${remaining:.2f} left after ${reserved:.2f} already reserved by "
        "running children. Lower `budget_usd`, wait for a running child to "
        "finish, or top up with `/budget add`."
    )
