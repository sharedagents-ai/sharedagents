"""Slow command responses must not turn accepted work into a second command."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from starlette.testclient import TestClient

from sharedagents_gateway import control_endpoint as ce


class Backend:
    def __init__(self):
        self.prompts = []
        self.queued = []
        self.cancels = []

    async def prompt(self, sid, message_id, blocks):
        self.prompts.append(blocks)
        if False:
            yield

    async def queue_add(self, sid, text, **kwargs):
        self.queued.append(text)
        return {"ok": True, "queue_id": "q1"}

    async def cancel(self, sid, *, force=False):
        self.cancels.append((sid, force))
        await asyncio.sleep(0.03)  # simulate load, no model or server needed


@pytest.fixture
def stack(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    for key in ("SHAREDAGENTS_AGENTS_HOME", "SHAREDAGENTS_SESSIONS_DIR"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "example-session.jsonl").touch()
    monkeypatch.setattr(ce, "_owner_agent_for", lambda sid: "example-agent")
    backend = Backend()
    app = ce.make_control_app(
        agent_names=["example-agent"], manager=SimpleNamespace(get=lambda name: backend), bearer_token=None
    )
    return app, backend


BASE = "/orchestrator/sessions/example-session"


def test_prompt_retry_after_completion_is_deduplicated(stack):
    app, backend = stack
    body = {"prompt": "Hello", "request_id": "prompt-1"}
    with TestClient(app) as client:
        first = client.post(BASE + "/prompt", json=body)
        client.portal.call(ce._await_submit_for_tests, "example-session")
        retry = client.post(BASE + "/prompt", json=body)
    assert first.status_code == retry.status_code == 202
    assert first.json() == retry.json()
    assert retry.json()["request_id"] == "prompt-1"
    assert len(backend.prompts) == 1


def test_busy_prompt_retry_only_queues_once(stack, monkeypatch):
    app, backend = stack
    monkeypatch.setattr(ce, "_subagent_live_state", lambda sid: "streaming")
    body = {"prompt": "Later", "request_id": "queue-1"}
    with TestClient(app) as client:
        first = client.post(BASE + "/prompt", json=body)
        retry = client.post(BASE + "/prompt", json=body)
    assert first.json() == retry.json()
    assert retry.json()["queued"] is True
    assert backend.queued == ["Later"]


def test_concurrent_cancel_and_retry_only_execute_once(stack):
    app, backend = stack

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://example.com") as client:
            body = {"request_id": "cancel-1", "force": True}
            a, b = await asyncio.gather(*(client.post(BASE + "/cancel", json=body) for _ in range(2)))
            assert a.status_code == b.status_code == 202
            final = await client.post(BASE + "/cancel", json=body)
            assert final.json()["status"] == "accepted"
            assert final.json()["request_id"] == "cancel-1"

    asyncio.run(run())
    assert backend.cancels == [("example-session", True)]


def test_disconnect_does_not_cancel_admitted_command(stack):
    app, backend = stack

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://example.com") as client:
            body = {"request_id": "disconnect-1", "force": True}
            task = asyncio.create_task(client.post(BASE + "/cancel", json=body))
            while not backend.cancels:
                await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.gather(*app.state.command_tasks)
            response = await client.post(BASE + "/cancel", json=body)
            assert response.json()["status"] == "accepted"

    asyncio.run(run())
    assert len(backend.cancels) == 1


def test_receipt_survives_new_application_and_conflicting_body_is_rejected(stack):
    app, backend = stack
    body = {"request_id": "persistent-1", "force": True}
    with TestClient(app) as client:
        assert client.post(BASE + "/cancel", json=body).status_code == 202
    other = ce.make_control_app(
        agent_names=["example-agent"], manager=SimpleNamespace(get=lambda name: backend), bearer_token=None
    )
    with TestClient(other) as client:
        assert client.post(BASE + "/cancel", json=body).json()["status"] == "accepted"
        response = client.post(BASE + "/cancel", json={**body, "force": False})
    assert response.status_code == 409
    assert len(backend.cancels) == 1


@pytest.mark.parametrize("action", ["prompt", "cancel"])
@pytest.mark.parametrize(
    "error,status",
    [
        (httpx.ReadTimeout, 202),
        (httpx.ReadError, 202),
        (httpx.RemoteProtocolError, 202),
        (httpx.WriteError, 202),
        (httpx.ConnectError, 503),
        (httpx.ConnectTimeout, 503),
        (httpx.PoolTimeout, 503),
    ],
)
def test_gateway_timeout_is_pending_but_connection_failure_is_rejected(monkeypatch, action, error, status):
    async def request(self, method, url, **kwargs):
        raise error("simulated slow response")

    monkeypatch.setattr(httpx.AsyncClient, "request", request)
    ctx = SimpleNamespace(orchestrator_url="http://example.com", orchestrator_token="")
    response = asyncio.run(ce._proxy_command(ctx, "POST", BASE + "/" + action, json_body={"request_id": "slow-1"}))
    import json

    data = json.loads(response.body)
    assert response.status_code == status
    if status == 202:
        assert data == {"status": "pending", "pending": True, "request_id": "slow-1"}
        assert "accepted" not in data


@pytest.mark.parametrize("request_id", ["", 17, "x" * 129])
def test_invalid_request_id_cannot_bypass_deduplication(stack, request_id):
    app, backend = stack
    with TestClient(app) as client:
        response = client.post(BASE + "/cancel", json={"request_id": request_id})
    assert response.status_code == 400
    assert backend.cancels == []


def test_accepted_south_command_then_lost_response_is_pending_and_not_repeated(stack, monkeypatch):
    app, backend = stack
    original = httpx.AsyncClient.request

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://example.com") as south:
            calls = 0

            async def slow_response(self, method, url, **kwargs):
                nonlocal calls
                result = await original(south, method, url, **kwargs)
                calls += 1
                if calls == 1:
                    raise httpx.ReadTimeout("response lost after acceptance")
                return result

            monkeypatch.setattr(httpx.AsyncClient, "request", slow_response)
            ctx = SimpleNamespace(orchestrator_url="http://example.com", orchestrator_token="")
            body = {"request_id": "lost-response", "force": True}
            first = await ce._proxy_command(ctx, "POST", BASE + "/cancel", json_body=body)
            retry = await ce._proxy_command(ctx, "POST", BASE + "/cancel", json_body=body)
            import json

            assert json.loads(first.body)["pending"] is True
            assert json.loads(retry.body)["status"] == "accepted"

    asyncio.run(run())
    assert backend.cancels == [("example-session", True)]


def test_unknown_handler_outcome_is_never_reexecuted(stack, monkeypatch):
    app, backend = stack
    attempts = []

    async def interrupted(request):
        attempts.append(True)
        raise RuntimeError("simulated interruption after a side effect")

    monkeypatch.setattr(ce, "_handle_session_cancel_impl", interrupted)
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post(BASE + "/cancel", json={"request_id": "unknown-1"})
            # Final and not a rejection: the caller stops waiting.
            assert response.status_code == 500
            assert response.json()["status"] == "unknown"
            assert response.json()["pending"] is False
    assert attempts == [True]


def test_claim_left_by_a_dead_process_becomes_unknown_not_reexecuted(stack, monkeypatch):
    from sharedagents_gateway import command_receipts as cr

    app, backend = stack
    body = {"request_id": "orphan-1", "force": True}
    import hashlib
    import json

    key = ("", "example-session", "cancel", "orphan-1")
    fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    cr._receipt(key, fingerprint)  # claimed, then the process died
    clock = [__import__("time").time()]
    monkeypatch.setattr(cr.time, "time", lambda: clock[0])
    with TestClient(app) as client:
        assert client.post(BASE + "/cancel", json=body).json()["pending"] is True
        clock[0] += cr.UNKNOWN_AFTER_S + 1
        late = client.post(BASE + "/cancel", json=body)
        again = client.post(BASE + "/cancel", json=body)
    assert late.status_code == again.status_code == 500
    assert late.json() == again.json()
    assert late.json()["status"] == "unknown"
    assert backend.cancels == []


def test_slow_live_handler_stays_pending_past_the_unknown_window(stack, monkeypatch):
    from sharedagents_gateway import command_receipts as cr

    app, backend = stack
    monkeypatch.setattr(cr, "UNKNOWN_AFTER_S", 0.0)
    release = asyncio.Event()

    async def slow(request):
        backend.cancels.append("started")
        await release.wait()
        return ce.JSONResponse({"accepted": True}, status_code=202)

    monkeypatch.setattr(ce, "_handle_session_cancel_impl", slow)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://example.com") as client:
            body = {"request_id": "slow-live", "force": True}
            first = asyncio.create_task(client.post(BASE + "/cancel", json=body))
            while not backend.cancels:
                await asyncio.sleep(0.001)
            retry = await client.post(BASE + "/cancel", json=body)
            assert retry.json()["pending"] is True
            release.set()
            assert (await first).json()["status"] == "accepted"

    asyncio.run(run())
    assert backend.cancels == ["started"]
