"""Local routing + vLLM serving APIs and Models-tab redirects."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, redirect, request

from sharedagents_web.local_serving import vllm_serving_snapshot
from sharedagents_web.router_models import (
    RouterModelsError,
    delete_router_model,
    list_router_models,
    upsert_router_model,
)
from sharedagents_web.router_status import router_status_snapshot
from sharedagents_web.server import BASE_PATH, require_auth

bp = Blueprint("local", __name__)


def _models_tab(section: str):
    return redirect(BASE_PATH + "/models#" + section)


@bp.route("/local")
@require_auth
def local_page():
    return _models_tab("router")


@bp.route("/local/router")
@require_auth
def local_router_page():
    return _models_tab("router")


@bp.route("/local/vllm")
@require_auth
def local_vllm_page():
    return _models_tab("vllm")


@bp.route("/api/local/router-status")
@require_auth
def api_local_router_status():
    """Same payload as ``/api/usage/router-status``."""
    return jsonify(router_status_snapshot())


@bp.route("/api/local/vllm")
@require_auth
def api_local_vllm():
    """Live replica gauges, mux counters, process units, and log tails."""
    return jsonify(vllm_serving_snapshot())


@bp.route("/api/local/router-models", methods=["GET", "POST"])
@require_auth
def api_local_router_models():
    if request.method == "GET":
        return jsonify(list_router_models())
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(upsert_router_model(payload)), 201
    except RouterModelsError as exc:
        return jsonify({"error": str(exc)}), 400
    except OSError:
        current_app.logger.exception("failed to create router model")
        return jsonify({"error": "Router model list could not be saved"}), 500


@bp.route("/api/local/router-models/<path:model_id>", methods=["PUT", "DELETE"])
@require_auth
def api_local_router_model(model_id: str):
    try:
        if request.method == "DELETE":
            return jsonify(delete_router_model(model_id))
        payload = request.get_json(silent=True) or {}
        if not payload.get("id"):
            payload["id"] = model_id
        return jsonify(upsert_router_model(payload, replace_id=model_id))
    except RouterModelsError as exc:
        status = 404 if "is not configured" in str(exc) else 400
        return jsonify({"error": str(exc)}), status
    except OSError:
        current_app.logger.exception("failed to update router model")
        return jsonify({"error": "Router model list could not be saved"}), 500
