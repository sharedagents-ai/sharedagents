"""Advisory decision pools (``kind: decision``): serialized, queued, rate-limited.

An advisory decision scorer (a small single-worker HTTP service) has
behaviour the generic replica-walk cannot express. The
2026-09-20 gap test measured four properties that are treated here as hard
constraints:

1. **Strictly serial, ~1.6 req/s ceiling.** One lock, one uvicorn worker. The
   scorer never queues and never hangs — it *sheds* (16 simultaneous -> 1x200 + 15x503;
   3.33 req/s -> 8x200/7x503), and the winner is not slowed by the losers. So
   the router owns the queue: at most one outbound request is in flight per
   pool, sends are paced, and overlap waits in-process.
2. **Three 503s share one status code.** ``"scorer busy"``, ``"scorer loading"``
   and ``"scoring failed"`` differ *only* in the ``detail`` field. Folding them
   into one "provider down" signal would misread transient back-pressure as a
   fatal outage — or hide a real scoring failure behind a retry.
3. **A restart is a ~40 s hard outage.** For 36-42 s the port is *closed*
   (connection refused, not 503): uvicorn binds only after all three checkpoints
   load, which also makes the ``"scorer loading"`` branch effectively unreachable
   at startup. Connect refusal is therefore an expected steady-state condition
   with retry/backoff. It never marks the replica ``failed`` and never touches
   another pool.
4. **``checkpoint`` is always sent explicitly.** ``auto_task_detection`` is off,
   so ``auto`` silently routes every typed-decisions workflow to the ``english``
   head with no error. The router substitutes the pool's checkpoint for a
   missing or ``auto`` value.

*Advisory* means the answer is never authoritative: this module never falls
back to a hosted provider, a decision pool can never be another pool's fallback
target, and ``entropy_certainty`` is never surfaced as a confidence score — it
is explicitly *not* P(correct), and the scorer strips any upstream
``confidence`` key.
"""

from __future__ import annotations

import asyncio
import json
import logging

from .models import (
    DECISION_CHECKPOINTS,
    DECISION_DEFAULT_CHECKPOINT,
    Pool,
)
from .runtime import get

log = logging.getLogger("sharedagents_vllm")

# --- response classes (constraint 2) ---------------------------------------
OK = "ok"
BUSY = "busy"
LOADING = "loading"
SCORING_FAILED = "scoring_failed"
UPSTREAM_ERROR = "upstream_error"

# Matched against the ``detail`` field only; the status code is shared.
_DETAIL_CLASSES = (
    ("scorer busy", BUSY),
    ("scorer loading", LOADING),
    ("scoring failed", SCORING_FAILED),
)

KNOWN_CHECKPOINTS = DECISION_CHECKPOINTS
DEFAULT_CHECKPOINT = DECISION_DEFAULT_CHECKPOINT

_DROP_REQUEST_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
    "x-router-local-only",
}

_DROP_RESPONSE_HEADERS = {
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "content-encoding",
    "content-type",
}


def _json_or_none(payload: bytes):
    try:
        return json.loads(payload)
    except Exception:  # noqa: BLE001
        return None


def detail_of(body) -> str:
    """Extract the scorer's ``detail`` string, tolerating str/dict/error shapes."""
    if body is None:
        return ""
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return json.dumps(detail)
        error = body.get("error")
        if isinstance(error, str):
            return error
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return str(error["message"])
        return json.dumps(body)[:300]
    return str(body)[:300]


def classify(status: int | None, body) -> str:
    """Map a decision-pool response to a handling class.

    Returns one of ``ok``/``busy``/``loading``/``scoring_failed``/
    ``upstream_error``. A 503 is disambiguated *only* by its ``detail``.
    """
    if status is not None and 200 <= status < 300:
        return OK
    if status is None:
        return UPSTREAM_ERROR
    low = detail_of(body).lower()
    for marker, cls in _DETAIL_CLASSES:
        if marker in low:
            return cls
    return UPSTREAM_ERROR


def resolve_checkpoint(pool: Pool, parsed: dict | None) -> str:
    """Constraint 4: never let ``auto`` or an absent value pick the head."""
    default = (getattr(pool, "checkpoint", None) or DEFAULT_CHECKPOINT).strip()
    if isinstance(parsed, dict):
        given = parsed.get("checkpoint")
        if isinstance(given, str) and given.strip() and given.strip() != "auto":
            return given.strip()
    return default


def probe_payload(model: str, pool: Pool) -> dict:
    """Health-probe body. Sends the checkpoint explicitly, like user traffic."""
    return {
        "model": model,
        "state": "Synthetic health check; take no action.",
        "questions": {
            "probe": {
                "type": "choice",
                "instructions": "Choose a label.",
                "criteria": {"HOLD": "wait", "ABSTAIN": "no decision"},
            },
        },
        "checkpoint": resolve_checkpoint(pool, None),
    }


def verdict(status: int | None, body, err: str) -> tuple[bool, str]:
    """Health-probe verdict for a decision pool.

    ``busy``/``loading`` mean the scorer is *up and working*, merely occupied:
    the probe interval (~30 s) is far below a serial scorer's ~1.6 req/s
    ceiling, so a busy
    probe is transient back-pressure rather than a health failure. Reporting it
    as unhealthy is exactly the misread constraint 2 warns about.
    """
    if err:
        low = err.lower()
        if "connect" in low or "refused" in low:
            return False, "connect_refused"
        if "timeout" in low:
            return False, "inference_timeout"
        return False, err
    cls = classify(status, body)
    if cls == OK:
        answers = body.get("answers") if isinstance(body, dict) else None
        answer = answers.get("probe") if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or not isinstance(answer.get("choice"), str):
            return False, "inference_shape"
        probs = answer.get("probabilities")
        if not isinstance(probs, dict) or set(probs) != {"HOLD", "ABSTAIN"}:
            return False, "inference_shape"
        return True, ""
    if cls in (BUSY, LOADING):
        return True, f"scorer_{cls}"
    if cls == SCORING_FAILED:
        return False, "scoring_failed"
    return False, f"inference_http {status}"


# --- per-pool serialization / pacing / outage state ------------------------
class _Slot:
    """Serialization state for one decision pool, created inside the loop."""

    __slots__ = ("lock", "waiters", "next_slot", "outage_since")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.waiters = 0
        self.next_slot = 0.0  # mono time at/after which the next send may go
        self.outage_since = 0.0  # mono time of the current connect-refused run


_slots: dict[tuple, _Slot] = {}


def _fingerprint(pool: Pool) -> tuple:
    return (
        pool.model_id,
        pool.read_timeout_seconds,
        pool.queue_timeout_seconds,
        pool.queue_max_depth,
        pool.rate_limit_per_second,
        pool.connect_retry_seconds,
        pool.busy_retry_seconds,
        pool.outage_tolerance_seconds,
    )


def _slot_for(pool: Pool) -> _Slot:
    key = _fingerprint(pool)
    slot = _slots.get(key)
    if slot is None:
        slot = _Slot()
        _slots[key] = slot
    return slot


def reset_slots() -> None:
    """Drop all queue/pacing state (config reload and tests)."""
    _slots.clear()


async def _pace(slot: _Slot, pool: Pool) -> None:
    """Space outbound sends to stay under the scorer's ~1.6 req/s ceiling."""
    loop = asyncio.get_running_loop()
    now = loop.time()
    if slot.next_slot > now:
        await asyncio.sleep(slot.next_slot - now)
        now = loop.time()
    rate = float(pool.rate_limit_per_second or 0.0)
    slot.next_slot = max(now, slot.next_slot) + (1.0 / rate if rate > 0 else 0.0)


def _error(
    status: int,
    kind: str,
    message: str,
    *,
    retry_after: int | None = None,
    retryable: bool = False,
    detail: str = "",
):
    from fastapi.responses import JSONResponse

    error: dict = {"message": message, "type": kind, "retryable": bool(retryable)}
    if detail:
        error["upstream_detail"] = detail
    if retry_after is not None:
        error["retry_after"] = retry_after
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return JSONResponse(status_code=status, content={"error": error}, headers=headers)


def _passthrough(resp, payload: bytes, *, checkpoint: str):
    from fastapi.responses import Response

    headers = {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    headers["X-Router-Advisory"] = "true"
    headers["X-Router-Decision-Checkpoint"] = checkpoint
    return Response(
        content=payload,
        status_code=resp.status_code,
        headers=headers,
        media_type=resp.headers.get("content-type") or "application/json",
    )


def _forward_headers(req) -> dict[str, str]:
    headers = {k: v for k, v in req.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}
    headers["Content-Type"] = "application/json"
    return headers


def _with_checkpoint(body: bytes, parsed: dict | None, pool: Pool) -> tuple[bytes, str]:
    checkpoint = resolve_checkpoint(pool, parsed)
    if not isinstance(parsed, dict) or parsed.get("checkpoint") == checkpoint:
        return body, checkpoint
    out = dict(parsed)
    out["checkpoint"] = checkpoint
    return json.dumps(out).encode("utf-8"), checkpoint


def _queue_retry_after(pool: Pool) -> int:
    rate = float(pool.rate_limit_per_second or 0.0)
    if rate <= 0:
        return 1
    return max(1, int(round(1.0 / rate)))


async def dispatch_decision(
    req,
    pool: Pool,
    *,
    path: str,
    body: bytes,
    parsed: dict | None,
):
    """Serve one request from an advisory decision pool.

    Serialized by construction: one in-flight request per pool and an in-process
    bounded queue, because the scorer sheds rather than queues (constraint 1).
    """
    rt = get()
    st = rt.state
    model_id = pool.model_id
    slot = _slot_for(pool)

    if not pool.replicas:
        return _error(503, "no_replica", f"decision pool '{model_id}' has no replicas")
    if rt.client is None:
        return _error(503, "no_client", "router http client is not bound")

    # Constraint 4: pin the checkpoint before the body leaves the router.
    out_body, checkpoint = _with_checkpoint(body, parsed, pool)
    if isinstance(parsed, dict) and parsed.get("checkpoint") != checkpoint:
        st.bump(f"decision_checkpoint_forced:{model_id}")

    # Constraint 1: the router owns the queue. Refuse rather than pile up once
    # the bounded depth is reached, so callers get a truthful Retry-After.
    depth = int(pool.queue_max_depth or 0)
    if depth > 0 and slot.waiters >= depth:
        st.bump(f"decision_queue_full:{model_id}")
        return _error(
            503,
            "queue_full",
            f"decision pool '{model_id}' queue is full ({depth} waiting)",
            retry_after=_queue_retry_after(pool),
            retryable=True,
        )

    wait = float(pool.queue_timeout_seconds or 0.0)
    slot.waiters += 1
    try:
        if wait > 0:
            await asyncio.wait_for(slot.lock.acquire(), timeout=wait)
        else:
            await slot.lock.acquire()
    except (asyncio.TimeoutError, TimeoutError):
        st.bump(f"decision_queue_timeout:{model_id}")
        return _error(
            503,
            "queue_timeout",
            f"decision pool '{model_id}' queue wait exceeded {wait}s "
            f"(strictly serial scorer, ~{pool.rate_limit_per_second or 1.5} req/s)",
            retry_after=_queue_retry_after(pool),
            retryable=True,
        )
    finally:
        slot.waiters -= 1

    try:
        return await _attempt(
            req,
            pool,
            slot,
            pool.replicas[0],
            f"{pool.replicas[0].url}{path}",
            out_body,
            model_id,
            checkpoint,
        )
    finally:
        slot.lock.release()


async def _attempt(
    req,
    pool: Pool,
    slot: _Slot,
    replica,
    url: str,
    body: bytes,
    model_id: str,
    checkpoint: str,
):
    import httpx

    rt = get()
    st = rt.state
    loop = asyncio.get_running_loop()
    headers = _forward_headers(req)

    read_timeout = pool.read_timeout_seconds if pool.read_timeout_seconds is not None else rt.cfg.read_timeout_seconds
    timeout = httpx.Timeout(
        read_timeout,
        connect=rt.cfg.connect_timeout_seconds,
        write=min(15.0, float(read_timeout)),
        pool=min(15.0, float(read_timeout)),
    )

    connect_budget = float(pool.connect_retry_seconds or 0.0)
    busy_budget = float(pool.busy_retry_seconds or 0.0)
    outage_window = float(pool.outage_tolerance_seconds or 0.0)
    started = loop.time()

    while True:
        await _pace(slot, pool)
        try:
            resp = await rt.client.post(
                url,
                content=body,
                headers=headers,
                timeout=timeout,
            )
            payload = await resp.aread()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # Constraint 3: a restart is a ~40 s port-closed window, not a 40 ms
            # blip. Record it, retry briefly, then report a retryable condition.
            # The replica is never marked failed here, and no other pool is
            # touched — decision traffic cannot degrade chat/embed/asr/tts.
            if not slot.outage_since:
                slot.outage_since = loop.time()
                st.bump(f"decision_outage_start:{model_id}")
            elapsed = loop.time() - started
            if connect_budget > 0 and elapsed < connect_budget:
                backoff = min(0.25 * (2 ** int(elapsed / 0.5)), 1.0)
                await asyncio.sleep(max(0.0, min(backoff, connect_budget - elapsed)))
                continue
            remaining = outage_window - (loop.time() - slot.outage_since)
            st.bump(f"decision_unreachable:{model_id}")
            return _error(
                503,
                "scorer_unreachable",
                f"decision scorer for '{model_id}' is not accepting connections "
                f"(a restart closes the port for ~40 s while checkpoints load)",
                retry_after=max(1, min(40, int(round(remaining)) or 1)),
                retryable=True,
                detail=f"{type(exc).__name__}: {exc}"[:200],
            )
        except httpx.TimeoutException as exc:
            st.bump(f"decision_read_timeout:{model_id}")
            return _error(
                504,
                "scorer_timeout",
                f"decision scorer for '{model_id}' did not answer within {read_timeout}s",
                retryable=True,
                detail=type(exc).__name__,
            )
        except Exception as exc:  # noqa: BLE001
            st.bump(f"decision_transport_error:{model_id}")
            return _error(
                502,
                "scorer_transport_error",
                f"decision scorer for '{model_id}' transport failed",
                retryable=True,
                detail=f"{type(exc).__name__}: {exc}"[:200],
            )

        # It answered, so the port is open again.
        slot.outage_since = 0.0
        parsed_body = _json_or_none(payload)
        cls = classify(resp.status_code, parsed_body)
        st.bump(f"decision_class:{model_id}:{cls}")

        if cls == OK:
            replica.last_user_success = rt.now()
            return _passthrough(resp, payload, checkpoint=checkpoint)

        detail = detail_of(parsed_body)

        if cls in (BUSY, LOADING):
            # Constraint 2: transient back-pressure, NOT an outage. Wait and
            # retry inside this request; never mark the replica unhealthy.
            if busy_budget > 0 and (loop.time() - started) < busy_budget:
                await asyncio.sleep(0.2 if cls == BUSY else 0.5)
                continue
            st.bump(f"decision_{cls}_exhausted:{model_id}")
            return _error(
                503,
                f"scorer_{cls}",
                f"decision scorer for '{model_id}' is {cls}: transient "
                f"back-pressure from a strictly serial scorer, not an outage",
                retry_after=_queue_retry_after(pool),
                retryable=True,
                detail=detail,
            )

        if cls == SCORING_FAILED:
            # A real per-request scoring failure. Retrying is pointless and a
            # hosted fallback is forbidden, so surface it distinctly — a caller
            # must not be able to confuse this with "busy".
            st.bump(f"decision_scoring_failed:{model_id}")
            return _error(
                503,
                "scoring_failed",
                f"decision scorer for '{model_id}' failed to score this request",
                retryable=False,
                detail=detail,
            )

        st.bump(f"decision_upstream_error:{model_id}")
        return _error(
            502,
            "scorer_upstream_error",
            f"decision scorer for '{model_id}' returned HTTP {resp.status_code}",
            retryable=False,
            detail=detail,
        )
