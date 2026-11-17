"""Secrets gateway endpoints — ``/control/v1/secrets``.

The gateway owns secret writes; the web UI is a thin proxy. A value travels
once, in the request body, straight into a dotenv file
(:mod:`sharedagents_agent.secret_store`). It is never logged, never returned, and
never enters a chat, a session log or a model prompt — agents read it from the
environment variable of the same name. The orchestrator's mtime config
watcher applies the change within one poll interval; no restart.

Routes (literal, registered BEFORE the ``/{agent}/…`` patterns in
``control_endpoint.make_control_app``):

    GET    /secrets                    — ``{"secrets": [{"name", "agent"}],
                                          "agents": [<scopable agent>]}``
    PUT    /secrets                    — body ``{"name", "value", "agent"?}``
    DELETE /secrets/{name}?agent=<a>   — remove one name

Auth: the gateway bearer token. A gateway with no token configured (open,
loopback-only by deployment) answers these routes for loopback clients only,
so a misconfigured public bind can never read names or write values.
"""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from sharedagents_agent.secret_store import (
    SecretStoreError,
    delete_secret,
    list_secrets,
    secret_agents,
    set_secret,
)

_LOG = logging.getLogger("sharedagents.gateway.secrets")


def _err(exc: SecretStoreError) -> JSONResponse:
    return JSONResponse({"error": exc.message, "code": exc.code}, status_code=exc.status)


def _agent(raw: object) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise SecretStoreError(400, "invalid_agent", "Unknown agent.")
    return raw.strip()


async def _handle_list(request: Request) -> Response:
    return JSONResponse(
        {"secrets": [e.to_dict() for e in list_secrets()], "agents": secret_agents()},
        headers={"Cache-Control": "no-store"},
    )


async def _handle_put(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = None
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object", "code": "bad_request"}, status_code=400)
    try:
        entry = set_secret(body.get("name"), body.get("value"), _agent(body.get("agent")))
    except SecretStoreError as exc:
        return _err(exc)
    _LOG.info("[secrets] saved name=%s agent=%s", entry.name, entry.agent or "(shared)")
    return JSONResponse({"ok": True, **entry.to_dict()}, headers={"Cache-Control": "no-store"})


async def _handle_delete(request: Request) -> Response:
    name = str(request.path_params.get("name", "") or "")
    try:
        agent = _agent(request.query_params.get("agent"))
        removed = delete_secret(name, agent)
    except SecretStoreError as exc:
        return _err(exc)
    if not removed:
        return JSONResponse({"error": "No such secret.", "code": "not_found"}, status_code=404)
    _LOG.info("[secrets] deleted name=%s agent=%s", name, agent or "(shared)")
    return JSONResponse({"ok": True, "name": name, "agent": agent or None})


def _is_loopback(request: Request) -> bool:
    host = getattr(request.client, "host", "") or ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _loopback_only(
    handler: Callable[[Request], Awaitable[Response]],
) -> Callable[[Request], Awaitable[Response]]:
    async def guarded(request: Request) -> Response:
        if not _is_loopback(request):
            return JSONResponse(
                {"error": "secrets need a gateway auth token for remote callers", "code": "forbidden"},
                status_code=403,
            )
        return await handler(request)

    return guarded


def make_secrets_routes(bearer_token: Any = None) -> list[Route]:
    """Routes to splice into ``control_endpoint`` BEFORE the ``/{agent}``
    patterns (``/secrets/command`` must never reach the command route).

    ``bearer_token`` is the gateway's token (the auth middleware enforces
    it); without one, every route is limited to loopback clients.
    """
    wrap = (lambda h: h) if str(bearer_token or "").strip() else _loopback_only
    return [
        Route("/secrets", wrap(_handle_list), methods=["GET"]),
        Route("/secrets", wrap(_handle_put), methods=["PUT"]),
        Route("/secrets/{name}", wrap(_handle_delete), methods=["DELETE"]),
    ]


__all__ = ["make_secrets_routes"]
