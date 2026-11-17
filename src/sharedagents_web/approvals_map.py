"""Map gateway pending-approval records to web session ids."""

from __future__ import annotations

from typing import Any


def map_gateway_approvals(
    raw: list[Any],
    store,
    owner: str,
    *,
    extra_session_ids: tuple[str, ...] | list[str] = (),
) -> list[dict[str, Any]]:
    """Filter gateway approvals to sessions this person may reach; attach web ids.

    Admins see every Web conversation's approvals (single operator); a user
    only those of conversations they own or that are shared with them —
    ``extra_session_ids`` can never widen that.
    """
    from sharedagents_web import access

    scoped = not access.is_admin(owner)
    gw_to_web: dict[str, str] = {}
    for rec in store.list_sessions(owner, limit=200):
        session_id = str(rec.get("id") or "").strip()
        if session_id:
            gw_to_web[session_id] = session_id
    for sid in extra_session_ids:
        sid = str(sid or "").strip()
        if sid:
            gw_to_web[sid] = sid
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        gw = str(item.get("session_hash") or "").strip()
        if gw not in gw_to_web:
            continue
        # A user only sees the approvals they may decide (manage access).
        if scoped and access.session_access(owner, gw) < access.MANAGE:
            continue
        mapped = dict(item)
        mapped["session_id"] = gw_to_web[gw]
        out.append(mapped)
    return out
