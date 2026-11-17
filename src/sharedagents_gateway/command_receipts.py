"""Durable idempotency receipts for prompt and cancel HTTP commands.

A claimed request is never executed twice, including after a process restart.
An interrupted command is never reported as rejected: absence of a response is
not proof of rejection. While it may still finish it is ``pending``; once no
live handler can settle it, its final verdict is ``unknown`` so callers stop
waiting. Session events remain the authority for turn progress/completion.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from collections.abc import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# A claimed command with no handler left in this process gets this long to be
# settled by another one before its verdict becomes ``unknown``.
UNKNOWN_AFTER_S = 60.0
_RUNNING: set[tuple] = set()


def pending_receipt(request_id: str) -> dict:
    return {"request_id": request_id, "status": "pending", "pending": True}


def unknown_receipt(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "status": "unknown",
        "pending": False,
        "accepted": False,
        "error_code": "outcome_unknown",
        "error": "The server did not confirm this request. Check the conversation before sending it again.",
    }


def _receipt(key: tuple, fingerprint: str, result: tuple | None = None):
    from sharedagents_core.layout import sharedagents_root

    root = sharedagents_root()
    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "command_receipts.sqlite3", timeout=10) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS receipts (
            principal TEXT, session TEXT, action TEXT, request_id TEXT,
            fingerprint TEXT NOT NULL, status INTEGER, body TEXT,
            claimed_at REAL,
            PRIMARY KEY (principal, session, action, request_id))""")
        if result is not None:
            db.execute(
                """UPDATE receipts SET status=?, body=? WHERE
                principal=? AND session=? AND action=? AND request_id=?""",
                (*result, *key),
            )
            return None
        inserted = db.execute(
            "INSERT OR IGNORE INTO receipts VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)", (*key, fingerprint, time.time())
        ).rowcount
        row = db.execute(
            """SELECT fingerprint, status, body, claimed_at FROM receipts WHERE
            principal=? AND session=? AND action=? AND request_id=?""",
            key,
        ).fetchone()
        return inserted, row


async def _settle_unknown(key: tuple, fingerprint: str, request_id: str) -> Response:
    data = unknown_receipt(request_id)
    await asyncio.to_thread(_receipt, key, fingerprint, (500, json.dumps(data)))
    return JSONResponse(data, status_code=500)


async def execute_once(request: Request, action: str, handler: Callable[[Request], Awaitable[Response]]) -> Response:
    try:
        body = await request.json()
    except (ValueError, UnicodeError):
        return await handler(request)
    if not isinstance(body, dict) or "request_id" not in body:
        return await handler(request)  # older transports remain compatible
    request_id = body["request_id"]
    if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
        return JSONResponse({"error": "invalid request_id"}, status_code=400)
    key = (request.headers.get("X-SharedAgents-Principal", ""), request.path_params.get("hash", ""), action, request_id)
    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    inserted, row = await asyncio.to_thread(_receipt, key, fingerprint)
    if row[0] != fingerprint:
        return JSONResponse(
            {"request_id": request_id, "error": "request_id already used", "error_code": "request_id_conflict"},
            status_code=409,
        )
    if not inserted:
        if row[1] is not None:
            return JSONResponse(json.loads(row[2]), status_code=row[1])
        if key in _RUNNING or time.time() - float(row[3] or 0) < UNKNOWN_AFTER_S:
            return JSONResponse(pending_receipt(request_id), status_code=202)
        # The claiming process is gone (restart/crash): nothing can settle it.
        return await _settle_unknown(key, fingerprint, request_id)

    async def settle():
        try:
            response = await handler(request)
            data = json.loads(response.body)
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Command outcome remains unknown")
            # An unexpected exception could have followed a side effect, so a
            # retry must not execute it again, and it is not a rejection.
            return await _settle_unknown(key, fingerprint, request_id)
        data["request_id"] = request_id
        data.setdefault("status", "accepted" if data.get("accepted") else "rejected")
        await asyncio.to_thread(_receipt, key, fingerprint, (response.status_code, json.dumps(data)))
        return JSONResponse(data, status_code=response.status_code)

    async def run():
        try:
            return await settle()
        finally:
            _RUNNING.discard(key)  # only after the verdict is stored

    # A client disconnect/read timeout cannot cancel an admitted command.
    tasks = getattr(request.app.state, "command_tasks", None)
    if tasks is None:
        tasks = request.app.state.command_tasks = set()
    _RUNNING.add(key)
    task = asyncio.create_task(run())
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return await asyncio.shield(task)
