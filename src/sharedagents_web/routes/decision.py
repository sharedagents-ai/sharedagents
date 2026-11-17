"""Authenticated Decision playground routes using the shared scoring service."""

import json
import re
import time

from flask import Blueprint, Response, current_app, g, jsonify, render_template, request, stream_with_context

from sharedagents_core.decision import (
    DecisionRequestError,
    build_catalog,
    normalize_score_payload,
    score_decision,
)
from sharedagents_web.decision_suggestions import parse_suggestions, suggestion_input
from sharedagents_web.principal import gateway_principal_headers, turn_actor
from sharedagents_web.server import BASE_PATH, REALTIME_URL, gateway_request, gw_session_history, require_auth, store

bp = Blueprint("decision", __name__)
SUGGEST_TIMEOUT = 180


@bp.route("/api/decision/suggest", methods=["POST"])
@require_auth
def api_decision_suggest():
    body = request.get_json(silent=True)
    try:
        prompt, count = suggestion_input(body)
        agent, model = body.get("agent"), body.get("model")
        if not isinstance(agent, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", agent):
            raise DecisionRequestError("Select an agent")
        if not isinstance(model, str) or not model or len(model) > 250:
            raise DecisionRequestError("Select a model")
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 422

    # Resolve against chat's catalogues; never accept provider URLs or keys.
    roster = gateway_request("GET", "/openai/v1/models", timeout=5, retry_unavailable=False)
    if roster is None or roster.status_code != 200:
        return jsonify({"error": "Agent catalogue unavailable. Try again."}), 503
    try:
        from sharedagents_agent.discovery.cache_lock import discover_agent_models
        from sharedagents_web.routes.sessions import _agent_roster

        if agent not in {row["name"] for row in _agent_roster(roster.json())}:
            return jsonify({"error": "Selected agent is unavailable"}), 422
        rows = discover_agent_models(agent, force=False)
        if not any(model == str(row.get("id") or row.get("alias") or "") for row in rows):
            return jsonify({"error": "Selected model is unavailable. Reopen the picker."}), 422
    except Exception:
        return jsonify({"error": "Model catalogue unavailable. Try again."}), 503
    finally:
        roster.close()

    sdata = store.create(g.user_email, agent, model)
    sid = sdata["id"]
    sdata.update(title="Decision question suggestions", submitted=True)
    if not store._save(sdata):
        return jsonify({"error": "Could not save the suggestion session"}), 503
    headers = gateway_principal_headers(g.user_email)
    submitted = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{sid}/prompt",
        timeout=15,
        headers=headers,
        json={
            "prompt": prompt,
            "initial_agent": agent,
            "initial_model": model,
            "frontend": "web",
            "actor": turn_actor(g.user_email),
            "workspace_origin": "web",
        },
    )
    if submitted is None:
        return jsonify(
            {"error": "Agent submission could not be confirmed. Check the session before retrying.", "session_id": sid}
        ), 503
    try:
        accepted = submitted.status_code in (200, 202) and submitted.json().get("accepted") is True
    except (ValueError, AttributeError):
        accepted = False
    finally:
        submitted.close()
    if not accepted:
        return jsonify({"error": "The agent could not start this request", "session_id": sid}), 502

    @stream_with_context
    def events():
        yield json.dumps({"status": "running", "session_id": sid}) + "\n"
        deadline = time.monotonic() + SUGGEST_TIMEOUT
        try:
            while time.monotonic() < deadline:
                history = gw_session_history(sid, limit=10)
                if history:
                    state = history.get("session_state")
                    if state in {"ask", "permission", "error", "closed", "paused", "cancelled", "aborted"}:
                        raise DecisionRequestError("The agent needs attention. Open its session to continue.")
                    messages = [m for m in history.get("messages", []) if m.get("role") == "assistant"]
                    if messages and (
                        messages[-1].get("aborted")
                        or messages[-1].get("error")
                        or messages[-1].get("status") in {"error", "aborted"}
                    ):
                        raise DecisionRequestError(
                            "The suggestion was interrupted or failed. Open its session for details."
                        )
                    if messages and not history.get("active") and not history.get("tree_busy"):
                        if messages[-1].get("status") == "running":
                            time.sleep(1)
                            continue
                        questions = parse_suggestions(messages[-1]["content"], count)
                        yield json.dumps({"status": "done", "questions": questions, "session_id": sid}) + "\n"
                        return
                yield json.dumps({"status": "running", "session_id": sid}) + "\n"
                time.sleep(1)
            raise DecisionRequestError(
                "Still waiting for the agent. Open its session to check progress before retrying."
            )
        except DecisionRequestError as exc:
            yield json.dumps({"status": "error", "message": str(exc), "session_id": sid}) + "\n"
        except Exception:
            yield (
                json.dumps(
                    {
                        "status": "error",
                        "message": "Could not read the agent reply. Open its session to check progress.",
                        "session_id": sid,
                    }
                )
                + "\n"
            )

    return Response(
        events(), mimetype="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    )


@bp.route("/decision")
@require_auth
def decision_page():
    return render_template(
        "decision.html",
        base=BASE_PATH,
        realtime_url=REALTIME_URL,
    )


@bp.route("/api/decision/catalog")
@require_auth
def api_decision_catalog():
    try:
        return jsonify(build_catalog(force=request.args.get("refresh") == "1"))
    except Exception:
        current_app.logger.exception("failed to read decision catalog")
        return jsonify({"error": "Decision catalog is temporarily unavailable"}), 503


@bp.route("/api/decision/score", methods=["POST"])
@require_auth
def api_decision_score():
    try:
        payload = normalize_score_payload(request.get_json(silent=True))
    except DecisionRequestError as exc:
        return jsonify(
            {
                "error": "invalid_request",
                "message": str(exc),
            }
        ), 422
    except (TypeError, ValueError):
        return jsonify(
            {
                "error": "invalid_request",
                "message": "JSON object required",
            }
        ), 422
    if request.accept_mimetypes.best == "application/x-ndjson":

        @stream_with_context
        def events():
            yield json.dumps({"status": "running"}) + "\n"
            status, body = _score(payload)
            yield json.dumps({"status": "done" if status < 400 else "error", "result": body}) + "\n"

        return Response(
            events(),
            mimetype="application/x-ndjson",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )
    status, body = _score(payload)
    return jsonify(body), status


def _score(payload):
    try:
        return score_decision(payload)
    except DecisionRequestError as exc:
        return 503, {
            "error": "unconfigured",
            "message": str(exc),
        }
    except Exception:
        # Never log or echo state.
        current_app.logger.exception("decision score failed")
        return 503, {
            "error": "scoring_failed",
            "message": "decision scorer failed to score this request",
        }
