"""Shared decision catalogue, validation and scoring for Web UI and native tools."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

import requests

QUESTION_TYPES = ("choice", "score", "noul")
MAX_BODY_BYTES = 32768
SCORE_TIMEOUT = 20.0
HEALTH_TIMEOUT = 2.0

# Discovery probes every configured provider sequentially; a slow one stalls
# the page and every score request. Targets are cached for this long, and a
# stale cache is served immediately while one background refresh runs.
TARGETS_TTL_S = 60.0
_targets_cache: dict[str, Any] = {"at": 0.0, "rows": None}
_targets_lock = threading.Lock()

SAMPLE_STATE = {"message": "I was charged twice for my subscription. Please help me get a refund."}
SAMPLE_QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "payments, invoices and refunds",
            "support": "technical problems and account access",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "Rate urgency.",
        "criteria": ["routine", "urgent"],
    },
    "needs_review": {
        "type": "noul",
        "instructions": "Does the message request review?",
    },
}


class DecisionRequestError(ValueError):
    """Caller payload failed the typed-decision contract."""


def derive_router_root(base_url: str) -> str:
    return str(base_url or "").rstrip("/").removesuffix("/v1")


def router_root() -> str:
    """vLLM mux root used for local ``/score`` traffic.

    Prefer an explicit decision router, then the Docker ``VLLM_URL``, then
    loopback. Never invent a LAN address.
    """
    for key in ("SHAREDAGENTS_DECISION_ROUTER", "VLLM_URL"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return derive_router_root(raw)
    return "http://127.0.0.1:8000"


def _safe_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not parsed.username


def _http() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _fetch_json(
    method: str, url: str, *, timeout: float, json_body: dict | None = None, headers: dict[str, str] | None = None
) -> tuple[int | None, Any, str]:
    if not _safe_http_url(url):
        return None, None, "invalid_url"
    try:
        resp = _http().request(
            method,
            url,
            json=json_body,
            headers=headers or {},
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.Timeout:
        return None, None, "timeout"
    except requests.RequestException as exc:
        return None, None, type(exc).__name__
    try:
        body = resp.json()
    except ValueError:
        body = None
    return resp.status_code, body, ""


def configured_models() -> list[dict[str, Any]]:
    from sharedagents_core.kernel.config import load_global_config

    return [r for r in load_global_config().get("models", []) if isinstance(r, dict)]


def _headers(row: dict[str, Any]) -> dict[str, str]:
    from sharedagents_agent.agent import _parse_dotenv, _warn_if_world_readable
    from sharedagents_core.kernel.layout import sharedagents_root

    secrets = {}
    shared = sharedagents_root() / "secrets.env"
    if shared.is_file():
        _warn_if_world_readable(shared)
        secrets = _parse_dotenv(shared.read_text(encoding="utf-8"))
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key_env = row.get("apiKeyEnv") or row.get("api_key_env")
    key = (os.environ.get(key_env) or secrets.get(key_env, "")) if key_env else ""
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _is_decision(row: dict[str, Any]) -> bool:
    capabilities = row.get("capabilities") or []
    architecture = row.get("architecture") or {}
    outputs = architecture.get("output_modalities", []) if isinstance(architecture, dict) else []
    return (
        row.get("kind") == "decision"
        or row.get("capability") in {"decision", "systemone"}
        or "decision" in capabilities
        or "systemone" in capabilities
        or row.get("decisionProtocol") in {"systemone", "score"}
        or "decisions" in outputs
    )


def _discover_targets_uncached() -> list[dict[str, Any]]:
    """Resolve server-owned targets; clients submit catalogue IDs, never URLs.

    Configured model rows share chat's models registry. Decision rows declare
    kind=decision and optionally decisionProtocol/decisionPath. Provider model
    lists may advertise decision capability metadata. OpenRouter advertises decision
    models through its output_modalities=decisions catalogue filter.
    """
    configured = configured_models()
    targets: dict[str, dict[str, Any]] = {}
    local_root = router_root()

    def add(row, base, provider, protocol="systemone", auth=None):
        wire_id = str(row.get("id") or "").strip()
        if not wire_id or not _safe_http_url(base):
            return
        ident = f"{provider}/{wire_id}"
        protocol = row.get("decisionProtocol") or protocol
        if protocol not in {"systemone", "score"}:
            return
        path = row.get("decisionPath") or ("/score" if protocol == "score" else "/v1/systemone")
        if not isinstance(path, str) or not path.startswith("/") or path.startswith("//") or "?" in path or "#" in path:
            return
        targets[ident] = {
            "id": ident,
            "wireId": wire_id,
            "label": row.get("label") or row.get("name") or wire_id,
            "provider": provider,
            "checkpoints": row.get("checkpoints") or [],
            "default_checkpoint": row.get("default_checkpoint") or "",
            "_url": derive_router_root(base) + path,
            "_auth": auth or row,
        }

    sources = {}
    for row in configured:
        base = str(row.get("baseUrl") or row.get("endpoint") or "").rstrip("/")
        provider = str(row.get("provider") or row.get("backend") or urlparse(base).netloc)
        if base == "https://openrouter.ai/api/v1":
            provider = "openrouter"
            row = {"apiKeyEnv": "OPENROUTER_API_KEY", **row}
        if base:
            sources[(base, provider)] = row
        if _is_decision(row):
            protocol = (
                "score" if row.get("backend") == "vllm" or derive_router_root(base) == local_root else "systemone"
            )
            add(row, base, provider, protocol)

    # This is a default source, not a provider/model allowlist.
    sources.setdefault(("https://openrouter.ai/api/v1", "openrouter"), {"apiKeyEnv": "OPENROUTER_API_KEY"})
    # Preserve the configured provider when it already owns the fallback
    # endpoint. Otherwise one scorer appears twice (e.g. vllm and local).
    if not any(derive_router_root(base) == local_root for base, _provider in sources):
        sources[(local_root + "/v1", "local")] = {}
    for (base, provider), source in sources.items():
        is_local_source = derive_router_root(base) == local_root
        catalog_url = local_root + "/v1/models" if is_local_source else base + "/models"
        if base == "https://openrouter.ai/api/v1":
            catalog_url += "?output_modalities=decisions"
        status, body, _ = _fetch_json("GET", catalog_url, timeout=HEALTH_TIMEOUT, headers=_headers(source))
        if status == 200 and isinstance(body, dict):
            for row in body.get("data", []):
                if not isinstance(row, dict):
                    continue
                if _is_decision(row):
                    if row.get("kind") == "decision" and row.get("replicas_healthy") == 0:
                        continue
                    ident = f"{provider}/{row.get('id')}"
                    if ident not in targets:
                        add(
                            row,
                            base,
                            provider,
                            "score" if is_local_source or source.get("backend") == "vllm" else "systemone",
                            source,
                        )
        # A direct local scorer advertises its model and checkpoints via health.
        if is_local_source:
            code, health, _ = _fetch_json("GET", derive_router_root(base) + "/health", timeout=HEALTH_TIMEOUT)
            if code == 200 and isinstance(health, dict) and health.get("loaded") is True and health.get("checkpoints"):
                existing = targets.get(f"{provider}/{health.get('model')}")
                if existing is not None:
                    # Health enriches discovery; explicit target paths, auth
                    # and labels retain the configured provider's overrides.
                    existing["checkpoints"] = existing["checkpoints"] or health["checkpoints"]
                    existing["default_checkpoint"] = existing["default_checkpoint"] or health.get(
                        "default_checkpoint", ""
                    )
                else:
                    add({**health, "id": health.get("model")}, base, provider, "score", source)
    return list(targets.values())


def invalidate_decision_targets() -> None:
    """Drop cached targets so the next call re-probes (tests, config edits)."""
    with _targets_lock:
        _targets_cache.update(at=0.0, rows=None)


def _refresh_targets_async() -> None:
    """Refresh the stale cache off the request path; one refresh at a time."""
    with _targets_lock:
        if _targets_cache.get("refreshing"):
            return
        _targets_cache["refreshing"] = True

    def run() -> None:
        try:
            rows = _discover_targets_uncached()
            with _targets_lock:
                _targets_cache.update(at=time.monotonic(), rows=rows)
        except Exception:
            pass  # keep serving the stale rows; the next call retries
        finally:
            with _targets_lock:
                _targets_cache["refreshing"] = False

    threading.Thread(target=run, name="decision-targets-refresh", daemon=True).start()


def discover_targets(force: bool = False) -> list[dict[str, Any]]:
    """Cached view of ``_discover_targets_uncached``.

    Fresh rows return immediately; stale rows return immediately while a
    single background refresh runs; a cold cache probes synchronously so the
    first caller after a restart still gets a catalogue. ``force`` re-probes
    synchronously and is meant for the page's refresh control.
    """
    with _targets_lock:
        rows, cached_at = _targets_cache["rows"], _targets_cache["at"]
    if rows is not None and not force and time.monotonic() - cached_at <= TARGETS_TTL_S:
        return [dict(r) for r in rows]
    if rows is not None and not force:
        _refresh_targets_async()
        return [dict(r) for r in rows]
    rows = _discover_targets_uncached()
    with _targets_lock:
        _targets_cache.update(at=time.monotonic(), rows=rows)
    return [dict(r) for r in rows]


def build_catalog(force: bool = False) -> dict[str, Any]:
    rows = discover_targets(force=force)
    jev = [r for r in rows if r["provider"] == "openrouter" and r["wireId"].lstrip("~").startswith("typesafe/jev-")]
    preferred = next((r["id"] for r in jev if r["wireId"].endswith("/jev-latest")), jev[0]["id"] if jev else "")
    return {
        "models": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        "default_model": preferred,
        "sample": {"state": SAMPLE_STATE, "questions": SAMPLE_QUESTIONS},
    }


def normalize_questions(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or not 1 <= len(raw) <= 8:
        raise DecisionRequestError("questions must contain 1 to 8 named questions")
    out: dict[str, Any] = {}
    for name, question in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise DecisionRequestError("each question must have a nonempty name")
        if not isinstance(question, dict):
            raise DecisionRequestError("each question must be an object")
        qtype = question.get("type")
        if qtype not in QUESTION_TYPES:
            raise DecisionRequestError("type must be choice, score or noul")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise DecisionRequestError("instructions must be nonempty text")
        shaped: dict[str, Any] = {
            "type": qtype,
            "instructions": instructions.strip(),
        }
        criteria = question.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 16:
                raise DecisionRequestError("choice requires 2 to 16 labeled options")
            if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) for k, v in criteria.items()):
                raise DecisionRequestError("choice labels and descriptions must be text")
            shaped["criteria"] = {str(k).strip(): str(v) for k, v in criteria.items()}
        elif qtype == "score":
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= 16
                or any(not isinstance(v, str) or not v.strip() for v in criteria)
            ):
                raise DecisionRequestError("score requires 2 to 16 ordered descriptions")
            shaped["criteria"] = [str(v).strip() for v in criteria]
        out[name.strip()] = shaped
    return out


def normalize_state(raw: Any) -> Any:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith(("{", "[")):
            try:
                raw = json.loads(text)
            except ValueError:
                pass
        if isinstance(raw, str):
            if not text:
                raise DecisionRequestError("nonempty state required")
            raw = {"text": raw}
    if raw in (None, "", {}, []):
        raise DecisionRequestError("nonempty state required")
    if not isinstance(raw, (dict, list)):
        raise DecisionRequestError("state must be text, an object or an array")
    return raw


def normalize_score_payload(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise DecisionRequestError("JSON object required")
    model = str(raw.get("model") or "").strip()
    if not model:
        raise DecisionRequestError("Select a decision model")
    checkpoint = str(raw.get("checkpoint") or "").strip()
    payload = {
        "model": model,
        "state": normalize_state(raw.get("state")),
        "questions": normalize_questions(raw.get("questions")),
    }
    if checkpoint:
        payload["checkpoint"] = checkpoint
    encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_BODY_BYTES:
        raise DecisionRequestError("request exceeds 32 KiB")
    return payload


def _target_for(payload: dict[str, Any]) -> tuple[str, dict[str, str], bool]:
    target = next((r for r in discover_targets() if r["id"] == payload["model"]), None)
    if not target:
        raise DecisionRequestError("Selected model is no longer available. Refresh the model list.")
    checkpoint = payload.get("checkpoint")
    if checkpoint and checkpoint not in target["checkpoints"]:
        raise DecisionRequestError("Selected checkpoint is not available")
    payload["model"] = target["wireId"]
    return target["_url"], _headers(target["_auth"]), target["provider"] == "local"


def score_decision(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    url, headers, _local = _target_for(payload)
    status, body, err = _fetch_json(
        "POST",
        url,
        timeout=SCORE_TIMEOUT,
        json_body=payload,
        headers=headers,
    )
    if err == "timeout":
        return 504, {
            "error": "scorer_timeout",
            "message": "decision scorer did not answer in time",
        }
    if err or status is None:
        return 503, {
            "error": "scorer_unreachable",
            "message": "decision scorer is not accepting connections",
        }
    if not isinstance(body, dict):
        return 502, {
            "error": "scorer_shape",
            "message": "decision scorer returned a non-object body",
        }
    if not 200 <= status < 300:
        detail = body.get("error") if isinstance(body.get("error"), (str, dict)) else body.get("detail")
        message = detail if isinstance(detail, str) else "decision scorer rejected the request"
        if isinstance(detail, dict) and isinstance(detail.get("message"), str):
            message = detail["message"]
        return status if status >= 400 else 502, {
            "error": "scorer_rejected",
            "message": message,
        }
    answers = body.get("answers")
    if not isinstance(answers, dict) or not answers or any(not isinstance(answer, dict) for answer in answers.values()):
        return 502, {
            "error": "scorer_shape",
            "message": "decision scorer returned no valid answers",
        }
    body.setdefault("model", payload["model"])
    return 200, body
