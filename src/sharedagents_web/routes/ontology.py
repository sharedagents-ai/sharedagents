"""Authenticated ontology exploration and human review, scoped by membership."""

import os
from pathlib import Path
from urllib.parse import urlsplit

from flask import Blueprint, Response, g, jsonify, render_template, request

from sharedagents_research.policy import authorization
from sharedagents_research.storage_client import configuration, open_runtime_ontology
from sharedagents_web.access import viewer_is_admin
from sharedagents_web.server import BASE_PATH, require_auth, require_auth_page

bp = Blueprint("ontology", __name__)


def scoped_store(operation="read"):
    permission = authorization(
        g.user_email, operator=viewer_is_admin(), workspace=request.args.get("workspace"), operation=operation
    )
    root = Path(
        os.environ.get("SHAREDAGENTS_ONTOLOGY_ROOT")
        or configuration().get("root")
        or Path(os.environ.get("SHAREDAGENTS_HOME", str(Path.home() / ".sharedagents"))) / "ontology"
    )
    return permission, open_runtime_ontology(root, permission["scope"], create=operation != "read")


def failure(exc):
    if isinstance(exc, PermissionError):
        return jsonify({"error": str(exc)}), 403
    if isinstance(exc, FileNotFoundError):
        return jsonify({"error": "No ontology yet. Start a research conversation to build it."}), 404
    if isinstance(exc, RuntimeError):
        return jsonify({"error": "Ontology storage unavailable. Check the configured data-layer dependency."}), 503
    return jsonify({"error": str(exc)}), 409 if "revision conflict" in str(exc) else 422


@bp.route("/ontology")
@require_auth_page
def page():
    try:
        permission = authorization(g.user_email, operator=viewer_is_admin(), workspace=request.args.get("workspace"))
        return render_template("ontology.html", base=BASE_PATH, ontology_role=permission["role"])
    except (ValueError, PermissionError, RuntimeError) as exc:
        return failure(exc)


@bp.route("/api/ontology", methods=["GET"])
@require_auth
def read():
    try:
        permission, store = scoped_store()
        action = request.args.get("action", "list")
        limit = min(200, max(1, int(request.args.get("limit", 100))))
        if action == "get":
            value = store.model_get(request.args["id"])
            if value is None:
                return jsonify({"error": "Ontology object not found"}), 404
        elif action == "search":
            value = store.model_search(request.args.get("q", ""), limit=limit)
        elif action == "history":
            value = store.model_history(
                key=request.args.get("id"), after=int(request.args.get("after", 0)), limit=limit
            )
        elif action == "export":
            from sharedagents_research.interchange import export_model

            value = export_model(store, request.args.get("format", "native"))
        elif action == "publications":
            value = store.publications(request.args["session_id"], limit=limit)
        elif action == "coverage":
            value = store.model_coverage(request.args.getlist("term"))
        elif action == "path":
            value = store.model_path(request.args["id"], request.args["target"])
        elif action == "evidence":
            value = store.search(request.args.get("q", ""), limit=limit)
        elif action == "source":
            value = store.get(request.args["id"], offset=int(request.args.get("offset", 0)), text_limit=16000)
        elif action in {"chart", "csv"}:
            from sharedagents_research.results import result_csv, result_view

            query = store.get(request.args["id"])
            if action == "csv":
                return Response(
                    result_csv(query),
                    mimetype="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="ontology-result.csv"'},
                )
            value = result_view(
                query, x=request.args.get("x"), y=request.args.get("y"), chart=request.args.get("chart", "bar")
            )
        elif action == "status":
            value = store.status()
        elif action == "list":
            value = store.model_list(kind=request.args.get("kind"), limit=limit, after=request.args.get("after", ""))
        else:
            raise ValueError("unknown ontology read action")
        return jsonify({"value": value, "role": permission["role"]})
    except (ValueError, KeyError, TypeError, PermissionError, RuntimeError, OSError) as exc:
        return failure(exc)


@bp.route("/api/ontology/research", methods=["POST"])
@require_auth
def research():
    from sharedagents_web import access
    from sharedagents_web.principal import gateway_principal_headers, turn_actor
    from sharedagents_web.server import gateway_request, store

    body = request.get_json(silent=True)
    origin = request.headers.get("Origin")
    if not request.is_json or origin and urlsplit(origin).netloc != request.host:
        return jsonify({"error": "Send a same-origin JSON research request"}), 403
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("question"), str)
        or not 0 < len(body["question"].strip()) <= 64000
    ):
        return jsonify({"error": "Research question required"}), 422
    refusal = access.agent_refusal(g.user_email, "research")
    if refusal:
        return jsonify({"error": refusal}), 403
    try:
        permission = authorization(
            g.user_email, operator=viewer_is_admin(), workspace=request.args.get("workspace"), operation="write"
        )
        default = authorization(g.user_email, operator=viewer_is_admin(), operation="write")
        if permission["scope"] != default["scope"]:
            raise ValueError(
                "Native research uses the configured default workspace; select it before submitting research"
            )
    except (PermissionError, ValueError) as exc:
        return failure(exc)
    session = store.create(g.user_email, "research", "")
    sid = session["id"]
    response = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{sid}/prompt",
        timeout=15,
        headers=gateway_principal_headers(g.user_email),
        json={
            "prompt": body["question"],
            "initial_agent": "research",
            "frontend": "web",
            "actor": turn_actor(g.user_email),
            "workspace_origin": "web",
        },
    )
    if response is None:
        return jsonify(
            {
                "error": "Research submission could not be confirmed; inspect this session before retrying",
                "session_id": sid,
            }
        ), 503
    try:
        accepted = response.status_code in (200, 202) and response.json().get("accepted") is True
    except (ValueError, AttributeError):
        accepted = False
    finally:
        response.close()
    if not accepted:
        return jsonify({"error": "Research request was not accepted", "session_id": sid}), 502
    return jsonify({"accepted": True, "session_id": sid}), 202


@bp.route("/api/ontology", methods=["POST"])
@require_auth
def write():
    if not request.is_json:
        return jsonify({"error": "Send a JSON document"}), 415
    origin = request.headers.get("Origin")
    if origin and urlsplit(origin).netloc != request.host:
        return jsonify({"error": "Cross-origin ontology changes are not accepted"}), 403
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "JSON object required"}), 422
    try:
        action = body.get("action", "edit")
        permission, store = scoped_store("review" if action == "review" else "write")
        if action == "review":
            value = store.model_review(
                body["id"],
                actor=permission["actor"],
                expected=body["expected_revision"],
                certified=body["certified"],
                reason=body.get("reason", ""),
            )
        elif action == "edit":
            value = store.model_put(
                body["kind"],
                body["data"],
                actor=permission["actor"],
                key=body.get("id"),
                expected=body.get("expected_revision"),
                origin="human",
            )
        elif action == "retire":
            value = store.model_retire(body["id"], actor=permission["actor"], expected=body["expected_revision"])
        elif action == "import":
            from sharedagents_research.interchange import import_model

            value = import_model(store, body["document"], actor=permission["actor"])
        else:
            raise ValueError("unknown ontology write action")
        return jsonify({"value": value})
    except (ValueError, KeyError, TypeError, PermissionError, RuntimeError, OSError) as exc:
        return failure(exc)
