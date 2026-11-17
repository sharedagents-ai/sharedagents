"""sharedagents_web.routes.commands — Slash-command discovery + dispatch routes.

Extracted from server.py (Task 2 de-monolith). Handler bodies are
byte-identical to the originals; only the decorator target changed
(``@app.route`` -> ``@bp.route``). Registered by server.create_app().
"""

import logging

from flask import Blueprint, g, jsonify, request

from sharedagents_web.principal import (
    canonical_gateway_principal,
    gateway_principal_headers,
    may_adopt_gateway_session,
)
from sharedagents_web.server import (
    DEFAULT_AGENT,
    gateway_request,
    gw_session_history,
    require_auth,
    store,
)

log = logging.getLogger("sharedagents.web.commands")

bp = Blueprint("commands", __name__)


@bp.route("/api/commands")
@require_auth
def api_commands():
    """List dispatchable slash commands for an agent (for the UI palette).

    Pure proxy to the gateway control surface — the backend owns the list.
    """
    from sharedagents_web import access

    agent = request.args.get("agent", DEFAULT_AGENT)
    # Spliced into a gateway path: only a plain agent name may pass.
    if not access.valid_agent_name(agent):
        return jsonify({"commands": [], "error": "invalid agent name"}), 400
    admin = access.is_admin(g.user_email)
    if not admin and access.agent_refusal(g.user_email, agent):
        return jsonify({"commands": [], "error": "agent not granted"}), 403
    resp = gateway_request("GET", f"/control/v1/{agent}/commands")
    if resp is not None and resp.status_code == 200:
        body = resp.json()
        # A user's palette lists only the commands their account may run —
        # and nothing else from the gateway.
        if not admin:
            names = body.get("commands") if isinstance(body, dict) else []
            body = {"commands": [c for c in names or [] if str(c) in access.USER_COMMANDS]}
        return jsonify(body)
    return jsonify({"commands": [], "error": "Gateway unreachable"}), 502


@bp.route("/api/command", methods=["POST"])
@require_auth
def api_command():
    """Run a slash command via the gateway's shared command runtime.

    THE web frontend carries zero command logic: it POSTs the slash text
    here, we forward it to ``/control/v1/{agent}/command`` bound to this
    web session's persistent gateway session, and return the
    ``CommandResponse`` verbatim for a generic render. Session ownership
    is enforced (IDOR-safe) before we expose the gateway session id.
    """
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "empty command", "error_code": "bad_request"}), 400
    from sharedagents_web import access

    # Users run an allowlisted subset; /agent, /model and /<agent> need the
    # grant. Refused before anything reaches the gateway.
    refusal = access.command_refusal(g.user_email, text)
    if refusal:
        return jsonify({"ok": False, "error": access.human_reason(refusal), "error_code": "not_granted"}), 403
    session_id = data.get("session_id")
    sdata = store.get(session_id, g.user_email) if session_id else None
    if session_id and not sdata:
        # A gateway-native session (TUI / monitor) the user just opened —
        # admins adopt it; shared members attach without taking ownership.
        if may_adopt_gateway_session(g.user_email):
            bound_owner = store.gateway_binding_owner(session_id)
            if bound_owner is not None and bound_owner != g.user_email:
                return jsonify({"ok": False, "error": "Session not found", "error_code": "forbidden"}), 403
            hist = gw_session_history(session_id)
            if hist is None:
                return jsonify({"ok": False, "error": "Session not found", "error_code": "forbidden"}), 403
            try:
                sdata = store.adopt_gateway(
                    g.user_email,
                    session_id,
                    hist,
                    gateway_principal=canonical_gateway_principal(g.user_email),
                )
            except PermissionError:
                return jsonify({"ok": False, "error": "Session not found", "error_code": "forbidden"}), 403
        elif access.session_access(g.user_email, session_id) >= access.CHAT:
            # A conversation shared with them: its commands go to the agent
            # it runs on (the gateway's session index), never a default.
            from sharedagents_web.server import _gw_session_log_exists, _load_meta_headers

            meta = _load_meta_headers([session_id]).get(session_id) or {}
            if not meta.get("agent"):
                meta = gw_session_history(session_id) or {}
            if not _gw_session_log_exists(session_id) or not meta.get("agent"):
                return jsonify({"ok": False, "error": "Session not found", "error_code": "forbidden"}), 403
            sdata = {"id": session_id, "agent": str(meta["agent"])}
        else:
            return jsonify({"ok": False, "error": "Session not found", "error_code": "forbidden"}), 403
    admin = access.is_admin(g.user_email)
    if sdata:
        # A user's command runs on the conversation's own agent; a client
        # ``initial_agent`` only steers an admin's.
        requested = data.get("initial_agent") if admin else None
        agent = requested or sdata.get("agent") or DEFAULT_AGENT
        target_sid = str(sdata.get("id") or session_id)
    else:
        # Discovery without a session id remains sessionless and therefore
        # cannot materialize a New Chat placeholder.
        agent = data.get("agent", DEFAULT_AGENT)
        target_sid = ""
    # Spliced into the gateway path: only a plain agent name may pass.
    if not access.valid_agent_name(agent):
        return jsonify({"ok": False, "error": "invalid agent name", "error_code": "bad_request"}), 400
    if not sdata:
        agent_refusal = access.agent_refusal(g.user_email, str(agent or ""))
        if agent_refusal:
            return jsonify({"ok": False, "error": access.human_reason(agent_refusal), "error_code": "not_granted"}), 403

    payload = {"text": text, "frontend": "web"}
    # Forward the client's explicit confirmation for an irreversible
    # command so the gateway runtime executes it; without it the runtime
    # re-challenges (parity with the session-delete re-challenge). A direct API
    # caller that omits this hits the same server-side re-challenge.
    if bool(data.get("confirm")):
        payload["confirm"] = True
    if target_sid:
        payload["sessionId"] = target_sid
        if sdata:
            from sharedagents_web.server import _gw_session_log_exists

            if not _gw_session_log_exists(target_sid):
                if not admin:
                    from sharedagents_web.routes.sessions import _submit_refusal

                    refusal = _submit_refusal(sdata, sid=target_sid)
                    if refusal:
                        return jsonify(
                            {"ok": False, "error": access.human_reason(refusal), "error_code": "not_granted"}
                        ), 403
                # A command (e.g. /goal) is this new chat's first input: the
                # gateway creates the session from the same first intent the
                # first prompt would carry.
                payload["initial_agent"] = agent
                payload["initial_model"] = str(sdata.get("model") or "")
                payload["workspace_cwd"] = str(sdata.get("workspace_cwd") or "")
                payload["workspace_origin"] = "web"
                mode = data.get("permission_mode")
                if isinstance(mode, str) and mode.strip():
                    payload["permission_mode"] = mode.strip()
    resp = gateway_request(
        "POST",
        f"/control/v1/{agent}/command",
        json=payload,
        headers=gateway_principal_headers(g.user_email),
    )
    if resp is None:
        return jsonify({"ok": False, "error": "Gateway unreachable", "error_code": "gateway_unreachable"}), 502
    try:
        body = resp.json()
    except ValueError:
        return jsonify({"ok": False, "error": "bad gateway response", "error_code": "bad_gateway"}), 502
    if resp.status_code == 200 and body.get("ok") is not False:
        from sharedagents_loop.loop_service import LOOP_CONTROL_SUBCOMMANDS

        words = str(text or "").split(None, 2)
        loop_id = str((((body.get("result") or {}).get("data") or {}).get("loop") or {}).get("loop_id") or "")
        if (
            len(words) >= 2
            and words[0].lower() == "/loop"
            and words[1].lower() not in LOOP_CONTROL_SUBCOMMANDS
            and loop_id
            and (body.get("result") or {}).get("kind") != "error"
        ):
            # The gateway records created_by="chat"; the web is
            # the only tier that knows WHICH user created it. Control commands
            # must never backfill old/model-created loops. Never fails the command.
            try:
                from sharedagents_loop.loop_registry import get_loop_registry

                get_loop_registry().stamp_actor_if_empty(loop_id, g.user_email)
            except Exception:  # noqa: BLE001
                log.warning("loop %s: could not record actor", loop_id, exc_info=True)
    if (
        resp.status_code == 200
        and body.get("ok") is not False
        and (body.get("result") or {}).get("kind") != "error"
        and session_id
    ):
        from sharedagents_web.routes.ui_stream import notify_state_changed

        kind = str(text or "").lstrip().split(" ", 1)[0].lower()
        if kind in {"/budget", "/goal", "/model"}:
            if kind == "/model":
                patch: dict = {}
                for effect in body.get("effects") or []:
                    if not isinstance(effect, dict):
                        continue
                    if effect.get("type") != "state_patch":
                        continue
                    payload = effect.get("payload") or {}
                    if isinstance(payload, dict):
                        maybe = payload.get("patch")
                        if isinstance(maybe, dict):
                            patch = maybe
                            break
                model = str(patch.get("model") or "").strip()
                if model:
                    row = store.get(session_id, g.user_email)
                    if isinstance(row, dict) and not store.is_unused_web_placeholder(row):
                        row["model"] = model
                        from sharedagents_web.routes.ui_stream import mark_apply_inflight

                        mark_apply_inflight(row)
                        store._save(row)
            notify_state_changed(session_id)
    return jsonify(body), (200 if resp.status_code == 200 else resp.status_code)
