"""Agent Builder gateway endpoints — ``/control/v1/agents/defs`` + registry.

The gateway OWNS all agent-definition mutations (per the Agent Builder
design): the web UI is a thin proxy, and every write goes through
:mod:`sharedagents_agent.agent_editor` (raw-JSON merge + atomic replace —
never ``AgentConfig.save()``, which drops operator keys).

Routes (all literal, registered BEFORE the ``/{agent}/…`` patterns in
``control_endpoint.make_control_app`` so they can never be swallowed by
the agent-param matcher):

    GET    /agents/defs                        — summary rows
    GET    /agents/defs/{agent}                — full definition + effective perms
    PATCH  /agents/defs/{agent}                — raw-JSON-merge config patch
    POST   /agents/defs/{agent}/validate       — dry-run patch validation
    GET    /agents/defs/{agent}/agents_md      — operator AGENTS.md
    PUT    /agents/defs/{agent}/agents_md      — persist AGENTS.md (sha-guarded)
    GET    /agents/defs/{agent}/system_prompt  — full prompt, hardcoded/AGENTS.md split
    GET    /agents/registry                    — builtin + MCP tool names

After a successful PATCH/PUT the live backend (when the gateway hosts the
agent) is reloaded immediately and the response carries the
``applied`` / ``restart_only`` report for the UI's restart banner; the
gateway's ambient mtime watcher remains the safety net for unhosted
agents.
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from sharedagents_agent.agent_editor import (
    AgentEditorError,
    list_agent_defs,
    patch_config,
    read_agents_md,
    read_definition,
    render_system_prompt,
    validate_model_membership,
    validate_patch,
    write_agents_md,
)

_LOG = logging.getLogger("sharedagents.gateway.agent_defs")


def _err(exc: AgentEditorError) -> JSONResponse:
    payload: dict[str, Any] = {"error": exc.message, "code": exc.code}
    payload.update(exc.extra or {})
    return JSONResponse(payload, status_code=exc.status)


def _agent_param(request: Request) -> str:
    return str(request.path_params.get("agent", "") or "")


async def _handle_defs_list(request: Request) -> Response:
    return JSONResponse({"agents": list_agent_defs()})


async def _handle_registry(request: Request) -> Response:
    from sharedagents_agent.tools.registry import (
        BUILTIN_TOOLS,
        ToolRegistry,
        discovered_mcp_tools_sync,
    )

    registry = ToolRegistry(list(BUILTIN_TOOLS))
    mcp = sorted(t.name for t in discovered_mcp_tools_sync())
    return JSONResponse(
        {
            "builtin": sorted(registry.names()),
            "mcp": mcp,
            "count": len(registry.names()) + len(mcp),
        }
    )


async def _handle_def_get(request: Request) -> Response:
    try:
        return JSONResponse(read_definition(_agent_param(request)))
    except AgentEditorError as exc:
        return _err(exc)


async def _handle_def_validate(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _err(AgentEditorError(422, "bad_json", "invalid JSON body"))
    try:
        cleaned = validate_patch(body)
    except AgentEditorError as exc:
        return _err(exc)
    # Dry-run parity with PATCH: run the SAME model-membership validation
    # against the PROSPECTIVE state (same helper, same 422 shape:
    # unknown_model / model_catalog_removed) so validate never blesses a
    # patch PATCH would reject.
    try:
        validate_model_membership(_agent_param(request), cleaned)
    except AgentEditorError as exc:
        return _err(exc)
    return JSONResponse({"valid": True})


async def _reload_report(ctx: Any, name: str) -> dict:
    """Hot-reload the live backend after a write, if the gateway hosts it.

    Returns the ConfigReloadReport shape for the UI's restart banner, or
    ``{"deferred": True}`` when the agent is not mounted here (the ambient
    config mtime watcher picks the change up on its next scan).
    """
    try:
        backend = ctx.manager.get(name)
    except Exception:  # noqa: BLE001 — unhosted/unknown agent
        return {"deferred": True}
    reload = getattr(backend, "reload_config", None)
    if reload is None:
        return {"deferred": True}
    try:
        report = reload()
    except Exception:  # noqa: BLE001 — reload must never fail a successful write
        return {"deferred": True}
    return {
        "applied": list(getattr(report, "applied", []) or []),
        "restart_only": list(getattr(report, "restart_only", []) or []),
    }


async def _handle_def_patch(request: Request) -> Response:
    name = _agent_param(request)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _err(AgentEditorError(422, "bad_json", "invalid JSON body"))
    if not isinstance(body, dict):
        return _err(AgentEditorError(422, "bad_patch", "patch body must be a JSON object"))
    sha = body.get("sha")
    patch = {k: v for k, v in body.items() if k != "sha"}
    try:
        result = patch_config(name, patch, sha=sha if isinstance(sha, str) and sha else None)
    except AgentEditorError as exc:
        return _err(exc)
    from sharedagents_gateway.control_endpoint import _ctx

    if "effort" in patch and isinstance(patch.get("upstream_model"), str):
        # "Set default" saves a model with its effort: remember the level for
        # that model too, so selecting it later loads the same level.
        await _remember_default_effort(name, patch["upstream_model"], patch.get("effort"))
    result["reload"] = await _reload_report(_ctx(request), name)
    return JSONResponse(result)


async def _remember_default_effort(agent: str, model: str, effort: Any) -> None:
    import asyncio

    from sharedagents_agent.agents.session_lifecycle import remember_model_effort

    try:
        await asyncio.to_thread(remember_model_effort, agent, model, effort)
    except Exception as exc:  # noqa: BLE001 — the config patch already succeeded
        _LOG.warning("remember default effort failed for %s (%s)", agent, type(exc).__name__)


async def _handle_agents_md_get(request: Request) -> Response:
    try:
        return JSONResponse(read_agents_md(_agent_param(request)))
    except AgentEditorError as exc:
        return _err(exc)


async def _handle_agents_md_put(request: Request) -> Response:
    name = _agent_param(request)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _err(AgentEditorError(422, "bad_json", "invalid JSON body"))
    if not isinstance(body, dict) or not isinstance(body.get("content"), str):
        return _err(AgentEditorError(422, "bad_content", "'content' (string) is required"))
    sha = body.get("sha")
    try:
        result = write_agents_md(
            name,
            body["content"],
            sha=sha if isinstance(sha, str) and sha else None,
        )
    except AgentEditorError as exc:
        return _err(exc)
    return JSONResponse(result)


async def _handle_system_prompt_get(request: Request) -> Response:
    try:
        return JSONResponse(render_system_prompt(_agent_param(request)))
    except AgentEditorError as exc:
        return _err(exc)


def make_agent_defs_routes() -> list[Route]:
    """Routes to splice into ``control_endpoint`` BEFORE the ``/{agent}``
    patterns (literal paths must win over the agent-param matcher)."""
    return [
        Route("/agents/defs", _handle_defs_list, methods=["GET"]),
        Route("/agents/defs/{agent}", _handle_def_get, methods=["GET"]),
        Route("/agents/defs/{agent}", _handle_def_patch, methods=["PATCH"]),
        Route("/agents/defs/{agent}/validate", _handle_def_validate, methods=["POST"]),
        Route("/agents/defs/{agent}/agents_md", _handle_agents_md_get, methods=["GET"]),
        Route("/agents/defs/{agent}/agents_md", _handle_agents_md_put, methods=["PUT"]),
        Route("/agents/defs/{agent}/system_prompt", _handle_system_prompt_get, methods=["GET"]),
        Route("/agents/registry", _handle_registry, methods=["GET"]),
    ]


__all__ = ["make_agent_defs_routes"]
