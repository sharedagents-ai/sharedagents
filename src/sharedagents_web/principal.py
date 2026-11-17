"""Canonical Web identity projection for gateway conversation ownership.

Every authenticated Web account maps to the gateway's ``local`` principal
(the gateway authenticates the Web with one bearer token and enforces nothing
per person). Who may reach which conversation, and what a non-admin account
may use, is decided by ``sharedagents_web.access`` before a request reaches the
gateway; a user's grants travel to the gateway turn runtime as the prompt's
``access_policy``.
"""

from __future__ import annotations

import os


def _local_operator_email() -> str:
    """Return the Web account authorized to act as the local installation owner."""
    configured = os.environ.get("SHAREDAGENTS_WEB_LOCAL_OPERATOR_EMAIL", "")
    if configured.strip():
        return configured.strip().lower()

    # The loopback-only debug identity is the local operator by definition.
    from sharedagents_web import server

    if server._loopback_debug_bypass_ok():
        return "debug@localhost"

    # Native account creation is CLI-only and documented as creating the
    # installation operator.  Persisted principal bindings make this choice
    # stable even if more accounts are added later.
    return server.auth_store.first_enabled_user() or ""


def canonical_gateway_principal(web_owner: str) -> str:
    """Map one authenticated Web owner to its durable gateway principal."""
    from sharedagents_web import server

    owner = str(web_owner or "").strip().lower()
    if not owner:
        raise ValueError("authenticated web owner required")
    return server.store.bind_gateway_principal(owner, local_owner=_local_operator_email())


def is_configured_local_operator(web_owner: str) -> bool:
    """Whether this owner may observe unbound local native sessions.

    Admins (role / env / operator / debug / fallback) may. A signed-in user
    reaches a TUI/Telegram conversation only when it is theirs or shared with
    them — never because every Web principal maps to gateway ``local``.
    """
    from sharedagents_web import access

    return access.is_admin(web_owner)


def may_adopt_gateway_session(web_owner: str) -> bool:
    """Only an admin may adopt a TUI/Telegram session into the Web store.

    Shared members use ``session_access`` to read and (when granted) prompt
    that hash without taking ownership.
    """
    from sharedagents_web import access

    return access.is_admin(web_owner)


def gateway_principal_headers(web_owner: str) -> dict[str, str]:
    return {
        "X-Sharedagents-Principal": canonical_gateway_principal(web_owner),
    }


def turn_actor(web_owner: str) -> dict[str, str]:
    """Attribution record for a prompt submitted by an authenticated Web user.

    Provenance only — it never grants or withholds anything: sharedagents stays
    single-user for access and every owner still maps to the
    ``local`` gateway principal. The display name is snapshotted at submit
    time so history reads like git authorship.
    """
    from sharedagents_session.actors import ACTOR_VIA_WEB, normalize_actor
    from sharedagents_web import server

    owner = str(web_owner or "").strip().lower()
    if not owner:
        raise ValueError("authenticated web owner required")
    actor = normalize_actor(
        {
            "email": owner,
            "name": server.auth_store.display_name(owner),
            "via": ACTOR_VIA_WEB,
        }
    )
    if actor is None:
        raise ValueError(f"unattributable web owner: {owner!r}")
    return actor
