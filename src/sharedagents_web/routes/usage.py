"""Usage dashboard page and read-only ledger snapshot API."""

from __future__ import annotations

import sqlite3

from flask import Blueprint, current_app, jsonify, render_template, request

from sharedagents_core.layout import pricing_auto_path, pricing_path
from sharedagents_model.usage_ledger import get_pricing_table, list_generations
from sharedagents_model.usage_pricing import (
    PricingSaveError,
    delete_pricing_entry,
    save_pricing_table,
    serialize_pricing_table,
    update_pricing_rates,
    upsert_pricing_entry,
)
from sharedagents_web.router_models import (
    RouterModelsError,
    delete_router_model,
    list_router_models,
    upsert_router_model,
)
from sharedagents_web.router_status import router_status_snapshot
from sharedagents_web.server import BASE_PATH, REALTIME_URL, require_admin, require_auth
from sharedagents_web.subscription_limits import (
    ProviderUnavailable,
    catalog as subscription_catalog,
    get_pace_history,
    pace_history_window,
    probe_provider as subscription_probe,
    snapshot as subscription_snapshot,
)
from sharedagents_web.usage_stats import usage_snapshot

bp = Blueprint("usage", __name__)


def _activity_days(raw: str | None) -> int:
    try:
        return min(730, max(28, int(raw or 365)))
    except (TypeError, ValueError):
        return 365


@bp.route("/usage")
@require_auth
def usage_page():
    return render_template(
        "usage.html",
        base=BASE_PATH,
        realtime_url=REALTIME_URL,
    )


@bp.route("/api/usage/subscriptions/catalog", methods=["GET"])
@require_auth
def api_usage_subscriptions_catalog():
    """Immediate provider list so the UI can render per-card loaders.

    Rows are decorated with the external-agent health scan: every card
    carries its CLI ``version`` and an ``auth_required`` flag + ``auth_hint``
    when the engine is installed but has no usable credential — including
    engines the credential probes would otherwise have dropped (e.g. a
    signed-out Antigravity), which re-join the tab so the operator is
    instructed to authenticate instead of wondering where the card went.
    """
    try:
        from sharedagents_web import agent_health

        return jsonify(agent_health.decorate_catalog(subscription_catalog()))
    except Exception:
        current_app.logger.exception("failed to list subscription providers")
        return jsonify({"error": "Subscription catalog is temporarily unavailable"}), 503


@bp.route("/api/usage/subscriptions/<provider_id>", methods=["GET", "POST"])
@require_auth
def api_usage_subscription_provider(provider_id: str):
    """Probe one subscription or prepaid API provider."""
    from sharedagents_web import agent_health

    refresh = request.method == "POST" or str(request.args.get("refresh") or "").lower() in {
        "1",
        "true",
        "yes",
    }
    # Installed but not signed in: the credential probes drop these from the
    # available set, so ``subscription_probe`` would 404. Answer with a
    # renderable card carrying the sign-in instruction instead — the whole
    # point of the alert.
    card = agent_health.unauthenticated_provider_card(provider_id, refresh=refresh)
    if card is not None:
        return jsonify(
            {
                "updated_at": card["updated_at"],
                "source": "live",
                "provider": card,
            }
        )
    try:
        return jsonify(subscription_probe(provider_id, refresh=refresh))
    except ProviderUnavailable as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        current_app.logger.exception("failed to read subscription provider %s", provider_id)
        return jsonify({"error": "Subscription limits are temporarily unavailable"}), 503


@bp.route("/api/usage/subscriptions", methods=["GET", "POST"])
@require_auth
def api_usage_subscriptions():
    """Cached Codex / Claude / Cursor / AGY plan windows.

    GET returns the last snapshot (or a fresh probe when the cache is stale).
    POST forces a live refresh — the Subscriptions tab Update button.
    """
    refresh = request.method == "POST"
    try:
        return jsonify(subscription_snapshot(refresh=refresh))
    except Exception:
        current_app.logger.exception("failed to read subscription limits")
        return jsonify({"error": "Subscription limits are temporarily unavailable"}), 503


@bp.route("/api/usage/agents/health", methods=["GET"])
@require_auth
def api_usage_agents_health():
    """External-agent integrity scan: installed / version / authenticated.

    Backs the Usage > Subscriptions alerts and version chips. ``?refresh=1``
    re-probes the CLIs; the default path answers from the cached scan so the
    tab opens instantly.
    """
    from sharedagents_web import agent_health

    refresh = str(request.args.get("refresh") or "").lower() in {"1", "true", "yes"}
    try:
        return jsonify(agent_health.snapshot(force=refresh))
    except Exception:
        current_app.logger.exception("failed to scan external agent health")
        return jsonify({"error": "Agent health scan is temporarily unavailable"}), 503


@bp.route("/api/usage/agents/<agent>/auth-check", methods=["POST"])
@require_auth
def api_usage_agent_auth_check(agent: str):
    """Re-probe one agent (the card/selector 'check again' action)."""
    from sharedagents_web import agent_health

    row = agent_health.refresh_agent(agent)
    if row is None:
        return jsonify({"error": f"unknown agent {agent!r}"}), 404
    return jsonify(row)


@bp.route("/api/usage/agents/<agent>/auth", methods=["POST"])
@require_auth
def api_usage_agent_auth(agent: str):
    """Attempt the engine's OWN automatic authentication where one exists.

    Only ACP engines with a wired ``acp_auth_method`` (cursor, kimi) can
    authenticate themselves headlessly — the same ``authenticate`` replay
    ``AcpEngine`` performs at dispatch. Everything else is browser OAuth:
    the response says so and the UI shows the terminal command instead.
    """
    from sharedagents_web import agent_health

    try:
        result = agent_health.auto_auth(agent)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    status = 200 if result.get("ok") else 409
    return jsonify(result), status


@bp.route("/api/usage/agents/<agent>/update", methods=["POST"])
@require_admin
def api_usage_agent_update(agent: str):
    """Update one external agent CLI by re-running its installer (admin).

    Async: returns a job id immediately; poll ``/api/usage/agents/update/<id>``
    for status and the installer output tail.
    """
    from sharedagents_web import agent_health

    try:
        job = agent_health.start_update(agent)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        current_app.logger.exception("failed to start agent update for %s", agent)
        return jsonify({"error": "Agent update could not be started"}), 500
    return jsonify(job), 202


@bp.route("/api/usage/agents/update/<job_id>", methods=["GET"])
@require_auth
def api_usage_agent_update_status(job_id: str):
    from sharedagents_web import agent_health

    job = agent_health.update_status(job_id)
    if job is None:
        return jsonify({"error": "unknown update job"}), 404
    return jsonify(job)


@bp.route("/api/usage/pace-history", methods=["GET"])
@bp.route("/api/usage/subscriptions/history", methods=["GET"])
@require_auth
def api_usage_pace_history():
    """Historical pace and API balance data; ``?period=6h|24h|7d|30d`` returns
    one chart window (thinned), no period the whole store."""
    try:
        period = (request.args.get("period") or "").strip()
        return jsonify(pace_history_window(period) if period else get_pace_history())
    except Exception:
        current_app.logger.exception("failed to read pace history")
        return jsonify({"history": []})


@bp.route("/api/usage")
@require_auth
def api_usage():
    """All-time ledger totals plus the requested recent activity window."""

    try:
        data = usage_snapshot(days=_activity_days(request.args.get("days")))
    except (OSError, sqlite3.Error):
        current_app.logger.exception("failed to read usage ledger")
        return jsonify({"error": "Usage data is temporarily unavailable"}), 503
    return jsonify(data)


def _generations_query():
    try:
        limit = int(request.args.get("limit") or 20)
    except (TypeError, ValueError):
        limit = 20
    try:
        offset = int(request.args.get("offset") or 0)
    except (TypeError, ValueError):
        offset = 0
    return list_generations(
        limit=min(20, max(1, limit)),
        offset=offset,
        provider=str(request.args.get("provider") or "").strip(),
        agent_name=str(request.args.get("agent") or "").strip(),
        model=str(request.args.get("model") or "").strip(),
    )


@bp.route("/api/logs/generations")
@bp.route("/api/usage/generations")
@require_auth
def api_logs_generations():
    """Per-provider generation log written by sharedagents (not the vLLM router)."""

    try:
        return jsonify(_generations_query())
    except (OSError, sqlite3.Error):
        current_app.logger.exception("failed to read generation log")
        return jsonify({"error": "Generation log is temporarily unavailable"}), 503


@bp.route("/api/usage/pricing", methods=["GET", "POST"])
@require_auth
def api_usage_pricing():
    """Registered cache / input / output rates used to price ledger turns."""

    if request.method == "POST":
        payload = request.get_json(silent=True)
        try:
            data = save_pricing_table(payload or {})
        except PricingSaveError as exc:
            return jsonify({"error": str(exc)}), 400
        except (OSError, TypeError, ValueError):
            current_app.logger.exception("failed to save usage pricing")
            return jsonify({"error": "Pricing table could not be saved"}), 500
        return jsonify(data)

    table = get_pricing_table()
    data = serialize_pricing_table(table)
    user_path = pricing_path()
    data["source"] = str(user_path) if user_path.exists() else "bundled"
    # The three layers the effective table merges. Each row
    # already carries the layer that resolved it, so the tab can show whether a
    # rate is the operator's, the bundled audit's, or auto-populated.
    auto_path = pricing_auto_path()
    data["layers"] = {
        "bundled": "bundled default",
        "auto": str(auto_path) if auto_path.exists() else None,
        "operator": str(user_path) if user_path.exists() else None,
    }
    return jsonify(data)


@bp.route("/api/usage/pricing/sync", methods=["POST"])
@require_auth
def api_usage_pricing_sync():
    """Reconcile the auto pricing overlay with the model catalogue.

    Reports ONLY unless the body carries ``{"apply": true}``: a bare POST must
    not be able to reprice every discovered model by accident.
    """
    from sharedagents_model.pricing_sync import sync_pricing

    payload = request.get_json(silent=True) or {}
    only = payload.get("only")
    try:
        report = sync_pricing(
            dry_run=not bool(payload.get("apply")),
            only=[str(p) for p in only] if isinstance(only, (list, tuple)) else None,
        )
    except Exception:
        current_app.logger.exception("failed to sync model pricing")
        return jsonify({"error": "Pricing sync failed"}), 500
    return jsonify(report.to_dict())


@bp.route("/api/usage/pricing/entries", methods=["POST", "PATCH"])
@require_auth
def api_usage_pricing_create_entry():
    """Create one model entry or edit one row's rates without changing its match."""
    payload = request.get_json(silent=True) or {}
    try:
        if request.method == "PATCH":
            return jsonify(update_pricing_rates(payload))
        return jsonify(upsert_pricing_entry(payload)), 201
    except PricingSaveError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        current_app.logger.exception("failed to create pricing entry")
        return jsonify({"error": "Pricing entry could not be saved"}), 500


@bp.route("/api/usage/pricing/entries/<path:entry_key>", methods=["PUT", "DELETE"])
@require_auth
def api_usage_pricing_entry(entry_key: str):
    """Update or delete an individual model pricing entry."""
    try:
        if request.method == "DELETE":
            return jsonify(delete_pricing_entry(entry_key))
        payload = request.get_json(silent=True) or {}
        return jsonify(upsert_pricing_entry(payload, replace_key=entry_key))
    except PricingSaveError as exc:
        status = 404 if "not found" in str(exc) else 400
        return jsonify({"error": str(exc)}), status
    except Exception:
        current_app.logger.exception("failed to update pricing entry %s", entry_key)
        return jsonify({"error": "Pricing entry could not be updated"}), 500


@bp.route("/api/usage/router-status")
@require_auth
def api_usage_router_status():
    """Live vLLM router status for configured backend=vllm models.

    Always 200: per-endpoint failures are data (``available: false`` +
    ``error``), not transport errors, so the Usage tab keeps rendering.
    """
    return jsonify(router_status_snapshot())


@bp.route("/api/usage/router-models", methods=["GET", "POST"])
@require_auth
def api_usage_router_models():
    """Operator mux model list — empty until the owner adds pools."""

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


@bp.route("/api/usage/router-models/<path:model_id>", methods=["PUT", "DELETE"])
@require_auth
def api_usage_router_model(model_id: str):
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
