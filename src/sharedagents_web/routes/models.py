"""Models tab — model benchmarks.

The benchmark pane renders three hand-rolled SVG bar charts
(Intelligence, Speed, Cost per Task) from
:func:`sharedagents_web.model_benchmarks.fetch_benchmarks`, which is cache-first.
The network is only touched when the cache is missing/expired or the operator
asks for a refresh (``POST`` / ``?refresh=1``) — never on a page render.
"""

from __future__ import annotations

import json
import time

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

from sharedagents_web.local_benchmark import (
    DEFAULT_CONTEXT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    MANAGER,
    MAX_CONTEXT_TOKENS,
    BenchmarkError,
    list_targets,
)
from sharedagents_web.local_serving import vllm_serving_snapshot
from sharedagents_web.model_benchmarks import fetch_benchmarks
from sharedagents_web.server import BASE_PATH, REALTIME_URL, require_auth

bp = Blueprint("models", __name__)


@bp.route("/models")
@require_auth
def models_page():
    return render_template(
        "models.html",
        base=BASE_PATH,
        realtime_url=REALTIME_URL,
    )


@bp.route("/api/models/benchmarks", methods=["GET", "POST"])
@require_auth
def api_models_benchmarks():
    """Artificial Analysis benchmarks for the enabled-model overlay.

    GET returns the cached snapshot (a live fetch only when the cache is
    missing or past its TTL). POST — or ``?refresh=1`` — forces a live refresh,
    which is what the pane's refresh button calls. Either way the response is
    always 200 JSON: a missing key, a 401, a 429, or an HTTP failure is data
    (``error.code``), not a transport failure, so the page never breaks.
    """
    refresh = request.method == "POST" or str(request.args.get("refresh") or "").lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        return jsonify(fetch_benchmarks(refresh=refresh))
    except Exception:
        current_app.logger.exception("failed to read model benchmarks")
        return jsonify({"error": "Model benchmarks are temporarily unavailable"}), 503


@bp.route("/api/models/vllm")
@require_auth
def api_models_vllm():
    """Live replica gauges, mux counters, process units, and log tails."""
    try:
        return jsonify(vllm_serving_snapshot())
    except Exception:
        current_app.logger.exception("failed to read vLLM serving snapshot")
        return jsonify({"error": "vLLM serving status is temporarily unavailable"}), 503


@bp.route("/api/models/benchmark", methods=["GET", "POST"])
@require_auth
def api_models_benchmark():
    """Local replica benchmark targets and run control.

    GET returns the configured targets, defaults, and the current/last run
    snapshot. POST starts one simultaneous run; target ids are resolved
    server-side against ``vllm-router.yaml`` so the client cannot point the
    prober at arbitrary URLs.
    """
    if request.method == "GET":
        return jsonify(
            {
                "targets": [target.as_dict() for target in list_targets()],
                "defaults": {
                    "context_tokens": DEFAULT_CONTEXT_TOKENS,
                    "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                    "max_context_tokens": MAX_CONTEXT_TOKENS,
                },
                "run": MANAGER.last_snapshot(),
            }
        )
    payload = request.get_json(silent=True) or {}
    target_ids = payload.get("targets")
    if target_ids is not None and not isinstance(target_ids, list):
        return jsonify({"error": "targets must be a list"}), 400
    try:
        run = MANAGER.start(
            target_ids=[str(tid) for tid in (target_ids or [])],
            context_tokens=payload.get("context_tokens", DEFAULT_CONTEXT_TOKENS),
            max_output_tokens=payload.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
        )
    except BenchmarkError as exc:
        return jsonify({"error": str(exc)}), 409 if "already in progress" in str(exc) else 400
    return jsonify(run.snapshot()), 202


@bp.route("/api/models/benchmark/<run_id>")
@require_auth
def api_models_benchmark_state(run_id: str):
    snapshot = MANAGER.snapshot(run_id)
    if snapshot is None:
        return jsonify({"error": "benchmark run not found"}), 404
    return jsonify(snapshot)


@bp.route("/api/models/benchmark/<run_id>/stream")
@require_auth
def api_models_benchmark_stream(run_id: str):
    """SSE: replay the event log, then tail it until the run is done.

    Reads the shared on-disk log rather than an in-process queue, so a
    reconnect can land on any gunicorn worker.
    """
    if MANAGER.snapshot(run_id) is None:
        return jsonify({"error": "benchmark run not found"}), 404
    store = MANAGER.store

    def generate():
        offset = 0
        idle = 0
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            events, offset = store.read_events(run_id, offset)
            for event in events:
                yield _sse(event)
                if event.get("type") == "done":
                    return
            snapshot = store.read_snapshot(run_id)
            if snapshot is None:
                return
            if snapshot.get("status") != "running":
                # Drain anything written between the read and the status check.
                events, offset = store.read_events(run_id, offset)
                if not events:
                    return
                continue
            idle += 1
            if idle % 150 == 0:
                yield "event: hb\ndata: {}\n\n"
            time.sleep(0.1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
