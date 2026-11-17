"""The gateway carries the web actor to the agent."""

import time

from starlette.testclient import TestClient

from sharedagents_gateway import control_endpoint as ce
from sharedagents_session.actors import actor_from_blocks

ACTOR = {"email": "admin@sharedagents.ai", "name": "admin", "via": "web"}


class _Backend:
    def __init__(self):
        self.prompts, self.queued = [], []

    async def prompt(self, sid, message_id, blocks):
        self.prompts.append([dict(b) for b in blocks])
        return
        yield  # unreachable: makes this an async generator

    async def queue_add(self, sid, text, *, source="", author="", actor=None):
        self.queued.append({"text": text, "source": source, "author": author, "actor": actor})
        return {"ok": True, "queue_id": "q1"}


class _LegacyBackend(_Backend):
    async def queue_add(self, sid, text, *, source="", author=""):  # no actor kwarg
        self.queued.append({"text": text, "source": source, "author": author})
        return {"ok": True, "queue_id": "q1"}


class _Manager:
    def __init__(self, backend):
        self.backend = backend

    def get(self, name):
        return self.backend


def _app(tmp_path, monkeypatch, backend, *sids):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    for key in ("SHAREDAGENTS_AGENTS_HOME", "SHAREDAGENTS_SESSIONS_DIR"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "sessions").mkdir(exist_ok=True)
    for sid in sids:
        (tmp_path / "sessions" / f"{sid}.jsonl").write_bytes(b"")
    monkeypatch.setattr(ce, "_owner_agent_for", lambda sid: "jarvis")
    return ce.make_control_app(agent_names=["jarvis"], manager=_Manager(backend), bearer_token=None)


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.02)
    return pred()


def test_idle_submit_carries_actor_in_block_meta(tmp_path, monkeypatch):
    backend = _Backend()
    with TestClient(_app(tmp_path, monkeypatch, backend, "s-idle")) as client:
        r = client.post(
            "/orchestrator/sessions/s-idle/prompt", json={"prompt": "hi", "frontend": "web", "actor": ACTOR}
        )
        assert r.status_code == 202, r.text
        assert _wait(lambda: backend.prompts)
    assert actor_from_blocks(backend.prompts[0]) == ACTOR
    assert backend.prompts[0][0]["text"] == "hi"


def test_busy_submit_queues_with_actor(tmp_path, monkeypatch):
    backend = _Backend()
    monkeypatch.setattr(ce, "_subagent_live_state", lambda sid: "live")
    with TestClient(_app(tmp_path, monkeypatch, backend, "s-busy")) as client:
        r = client.post("/orchestrator/sessions/s-busy/prompt", json={"prompt": "later", "actor": ACTOR})
        assert r.status_code == 202 and r.json().get("queued") is True, r.text
    assert backend.queued == [{"text": "later", "source": "web", "author": "", "actor": ACTOR}]


def test_queue_post_carries_actor_and_legacy_backend_still_works(tmp_path, monkeypatch):
    backend = _Backend()
    with TestClient(_app(tmp_path, monkeypatch, backend, "s-q")) as client:
        client.post("/orchestrator/sessions/s-q/queue", json={"text": "q", "source": "web", "actor": ACTOR})
    assert backend.queued[-1]["actor"] == ACTOR
    legacy = _LegacyBackend()
    with TestClient(_app(tmp_path, monkeypatch, legacy, "s-l")) as client:
        r = client.post("/orchestrator/sessions/s-l/queue", json={"text": "q", "actor": ACTOR})
        assert r.status_code == 200, r.text
    assert legacy.queued[-1] == {"text": "q", "source": "", "author": ""}


def test_prompt_flatteners_ignore_actor_meta():
    from sharedagents_gateway.acp_bridge import _blocks_to_user_content, _flatten_prompt
    from sharedagents_session.actors import with_actor_meta

    blocks = [with_actor_meta({"type": "text", "text": "hi"}, ACTOR)]
    assert _flatten_prompt(blocks) == "hi"
    assert _blocks_to_user_content(blocks) == "hi"
    assert "admin@sharedagents.ai" not in ce._queue_prompt_text(blocks)


def test_wave_receipt_deduplicates_retry_after_acceptance(tmp_path, monkeypatch):
    """A different finalizer re-POSTs the same wave without a second parent turn."""
    from sharedagents_agent.agents import async_jobs
    from sharedagents_session.session_reader import tail_jsonl
    from sharedagents_session.session_writer import close_session_writer, get_session_writer

    sid = "wave-gateway-parent"
    sessions_dir = tmp_path / "sessions"

    class DurableBackend(_Backend):
        async def queue_add(self, sid, text, *, source="", author="", actor=None):
            qid = f"q{len(self.queued)}"
            item = {"queue_id": qid, "text": text, "source": source, "author": author}
            self.queued.append(item)
            get_session_writer(sessions_dir, sid).lifecycle(kind="queue_msg_added", payload=item)
            return {"ok": True, "queue_id": qid}

        async def queue_remove(self, sid, queue_id, *, reason=None):
            get_session_writer(sessions_dir, sid).lifecycle(
                kind="queue_msg_removed",
                payload={
                    "queue_id": queue_id,
                    "reason": reason,
                },
            )
            return {"ok": True}

        async def queue_list(self, sid):
            return {"pending": []}

        async def prompt(self, sid, message_id, blocks):
            self.prompts.append(blocks)
            writer = get_session_writer(sessions_dir, sid)
            writer.turn_started(turn_idx=1, user=blocks)
            writer.turn_ended(turn_idx=1, metrics={})
            if False:
                yield

    backend = DurableBackend()
    app = _app(tmp_path, monkeypatch, backend, sid)
    wid = async_jobs.create_wave(sid)
    members = [
        async_jobs.create(parent_sid=sid, agent=name, task=name, on_complete="notify", call_id=name, pid=0, wave_id=wid)
        for name in ("one", "two")
    ]
    async_jobs.seal_wave(sid, wid)
    # Both runner and recovery use this canonical, existing-format receipt key.
    key = min(m["job_id"] for m in members)
    note = f'<delegation_wave wave_id="{wid}">consolidated result</delegation_wave>'
    with TestClient(app) as client:
        response = client.post(
            f"/orchestrator/sessions/{sid}/prompt",
            json={
                "prompt": note,
                "background_job_id": key,
            },
        )
        assert response.status_code == 202, response.text
        assert _wait(lambda: backend.prompts)
        assert _wait(lambda: ce._background_turn_started(sessions_dir / f"{sid}.jsonl", key))
        response = client.post(
            f"/orchestrator/sessions/{sid}/prompt",
            json={
                "prompt": note,
                "background_job_id": key,
            },
        )
        assert response.status_code == 202 and response.json()["duplicate"], response.text
    events = [ev for ev, _ in tail_jsonl(sessions_dir / f"{sid}.jsonl", 0)]
    assert len([ev for ev in events if ev["type"] == "turn_started"]) == 1
    assert len(backend.prompts) == 1 and len(backend.queued) == 1
    close_session_writer(sessions_dir, sid)
