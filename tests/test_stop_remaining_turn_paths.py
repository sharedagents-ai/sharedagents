"""After a user Stop, nothing automatic starts a turn on the stopped tree.

Follow-up to the subagent stop cascade. Each test covers one remaining path:

1. cron jobs bound to the stopped session (fire, boot re-arm, resume),
2. the idle memory-flush maintenance turn,
3. the late result of a child that finished just before the Stop,
4. the local-LLM cancel path (queue purge and descendant walk),
5. a user Stop vs an external kill of the engine (retry decision),
plus the per-subagent Stop recording the Stop before it kills.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sharedagents_agent.agents import async_jobs, engine_pidfile
from sharedagents_loop import loop_service, loop_supervisor
from sharedagents_loop.loop_registry import STATUS_PAUSED, STATUS_RUNNING, LoopRegistry
from sharedagents_loop.loop_supervisor import LoopSupervisor
from sharedagents_session.operator_stop import operator_stop_mode, record_operator_stop
from sharedagents_session.session_index import connect_sessions_db, ensure_schema
from sharedagents_session.session_writer import SessionWriter

# -- fixtures ------------------------------------------------------------------


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path / "home"))
    sd = tmp_path / "sessions"
    sd.mkdir(parents=True)
    monkeypatch.setattr("sharedagents_core.layout.sessions_root", lambda: sd)
    monkeypatch.setattr(engine_pidfile, "kill_session_engine", lambda *a, **k: True)
    monkeypatch.setattr(async_jobs, "_kill_runner_tree", lambda *a, **k: True)
    return sd


@pytest.fixture
def registry(tmp_path, monkeypatch):
    reg = LoopRegistry(tmp_path / "loops.db")
    monkeypatch.setattr(loop_service, "_registry", lambda: reg)
    monkeypatch.setattr(loop_supervisor, "get_loop_registry", lambda: reg)
    monkeypatch.setattr(loop_supervisor, "cron_disabled", lambda: False)
    yield reg
    reg._conn.close()


def _session(sessions_dir: Path, sid: str, *, parent: str = "", state: str = "idle"):
    ensure_schema(sessions_dir / "sessions.db")
    w = SessionWriter(sessions_dir, sid, engine="claude")
    w.session_started({"agent_name": "rt", "parent_session_id": parent, "engine": "claude"})
    w.turn_started(turn_idx=1, user="first turn")
    w.close()
    con = connect_sessions_db(sessions_dir / "sessions.db")
    try:
        con.execute(
            "UPDATE sessions_meta SET session_state=?, live_turn_idx=1, "
            "last_turn_idx=1, parent_session_id=? WHERE session_hash=?",
            (state, parent, sid),
        )
        con.commit()
    finally:
        con.close()


def _state(sessions_dir: Path, sid: str) -> str:
    con = connect_sessions_db(sessions_dir / "sessions.db")
    try:
        return con.execute(
            "SELECT session_state FROM sessions_meta WHERE session_hash=?",
            (sid,),
        ).fetchone()["session_state"]
    finally:
        con.close()


def _fake_agent(sessions_dir: Path, queue_items: dict | None = None):
    queue_items = queue_items or {}
    return SimpleNamespace(
        name="rt",
        sessions_dir=sessions_dir,
        cfg=SimpleNamespace(),
        engine=SimpleNamespace(),
        loop_supervisor=MagicMock(),
        queue_list=MagicMock(side_effect=lambda s: list(queue_items.get(s, []))),
        queue_remove=MagicMock(return_value=True),
        _cancel_pending_flush=MagicMock(),
    )


def _cli_backend(agent):
    """The CLI-engine cancel path (``AgentACPBackend`` without a CoreBackend)."""
    from sharedagents_agent.agents.gateway_shim import AgentACPBackend

    shim = MagicMock(spec=AgentACPBackend)
    shim.agent = agent
    shim._core = None
    shim._cli_active_turns = {}
    shim.cancel = AgentACPBackend.cancel.__get__(shim, AgentACPBackend)
    shim.prompt = AgentACPBackend.prompt.__get__(shim, AgentACPBackend)
    return shim


def _core_backend(sessions_dir: Path, agent):
    """The local-LLM cancel path (``CoreBackend``)."""
    from sharedagents_gateway.acp_bridge import CoreBackend

    core = object.__new__(CoreBackend)
    core.cfg = SimpleNamespace(sessions_dir=sessions_dir, name="rt")
    core._new_agent = agent
    core._new_backend_active_turns = {}
    core._last_cancel_at = {}
    return core


def _backend(kind: str, sessions_dir: Path, agent):
    return _cli_backend(agent) if kind == "cli" else _core_backend(sessions_dir, agent)


# -- 1. cron jobs ----------------------------------------------------------------


def _job(reg: LoopRegistry, sid: str) -> dict:
    return reg.create_job(
        session_id=sid,
        agent_name="rt",
        prompt="check the build",
        human_schedule="every minute",
        cron="* * * * *",
        durable=True,
    )


@pytest.mark.parametrize("kind", ["cli", "core"])
@pytest.mark.parametrize("force", [True, False])
@pytest.mark.asyncio
async def test_stop_pauses_the_sessions_cron_jobs_and_boot_does_not_rearm_them(
    sessions_dir,
    registry,
    kind,
    force,
):
    _session(sessions_dir, "stopped")
    mine = _job(registry, "stopped")
    other = _job(registry, "other")
    agent = _fake_agent(sessions_dir)

    await _backend(kind, sessions_dir, agent).cancel("stopped", force=force)

    assert registry.get_job(mine["job_id"])["status"] == STATUS_PAUSED
    assert registry.get_job(other["job_id"])["status"] == STATUS_RUNNING
    agent.loop_supervisor.cancel.assert_any_call(f"job:{mine['job_id']}", pause=False)
    # Boot recovery re-arms only running durable jobs.
    sup = LoopSupervisor(SimpleNamespace(name="rt", cfg=SimpleNamespace()))
    sup.supervise_loop = AsyncMock(return_value={"ok": True})
    await sup.recover_running()
    kicked = [c.args[0] for c in sup.supervise_loop.await_args_list]
    assert kicked == [f"job:{other['job_id']}"]


@pytest.mark.asyncio
async def test_paused_job_never_fires_its_turn(sessions_dir, registry):
    job = _job(registry, "stopped")
    registry._execute(
        "UPDATE scheduled_jobs SET next_fire_at = ? WHERE job_id = ?",
        ("2000-01-01T00:00:00+00:00", job["job_id"]),
    )
    loop_service.stop_loops_for_session(_fake_agent(sessions_dir), "stopped")
    turns: list = []

    async def run_turn(sid, prompt, **_kw):
        turns.append(sid)
        if False:  # pragma: no cover — async generator
            yield None

    sup = LoopSupervisor(SimpleNamespace(name="rt", cfg=SimpleNamespace(), run_turn=run_turn))
    sup._claim(f"job:{job['job_id']}")  # a drive that already held the lease
    await asyncio.wait_for(sup._drive_one_job(job["job_id"]), 5)
    assert turns == []


@pytest.mark.asyncio
async def test_prompt_without_an_admitted_turn_does_not_resume_jobs(sessions_dir, registry):
    _session(sessions_dir, "stopped")
    job = _job(registry, "stopped")
    agent = _fake_agent(sessions_dir)
    shim = _cli_backend(agent)
    await shim.cancel("stopped", force=True)
    assert registry.get_job(job["job_id"])["status"] == STATUS_PAUSED

    async def core_prompt(*_a, **_kw):
        if False:  # pragma: no cover — async generator
            yield None

    shim._core = SimpleNamespace(prompt=core_prompt)
    background = [{"type": "text", "text": "late result", "_sharedagents_source": "background_task"}]
    [_ async for _ in shim.prompt("stopped", None, background)]
    assert registry.get_job(job["job_id"])["status"] == STATUS_PAUSED

    [_ async for _ in shim.prompt("stopped", None, [{"type": "text", "text": "go on"}])]
    # Merely routing a prompt (which may fail admission) is not a user turn.
    assert registry.get_job(job["job_id"])["status"] == STATUS_PAUSED


# -- 2. memory flush -------------------------------------------------------------


@pytest.mark.parametrize("kind", ["cli", "core"])
@pytest.mark.parametrize("force", [True, False])
@pytest.mark.asyncio
async def test_stop_cancels_the_pending_memory_flush_timer(sessions_dir, registry, kind, force):
    _session(sessions_dir, "parent")
    _session(sessions_dir, "child", parent="parent", state="streaming")
    agent = _fake_agent(sessions_dir)

    await _backend(kind, sessions_dir, agent).cancel("parent", force=force)

    cancelled = {c.args[0] for c in agent._cancel_pending_flush.call_args_list}
    assert cancelled == ({"parent", "child"} if force else {"parent"})


@pytest.mark.parametrize("mode", ["hard", "graceful"])
@pytest.mark.asyncio
async def test_memory_flush_never_starts_on_a_stopped_session(sessions_dir, mode):
    from sharedagents_agent.agents.memory_integration import MemoryIntegrationMixin

    class _Agent(MemoryIntegrationMixin):
        def __init__(self):
            self.cfg = SimpleNamespace(memory_flush_enabled=True, memory_flush_min_turns=1)
            self.sessions_dir = sessions_dir
            self._memory_flush_inflight = set()
            self._memory_flush_timers = {}
            self.turns: list = []

        def _session_is_live(self, sid):
            return False

        def _turns_since_flush(self, session):
            return 5

        def _effective_mode(self, session):
            return "auto"

        def memory_index_all(self):
            pass

        async def run_turn(self, sid, prompt, **kw):
            from sharedagents_agent.agents.engine import TurnDone

            self.turns.append(kw.get("source"))
            yield TurnDone(state="ok")

    _session(sessions_dir, "s1")
    agent = _Agent()
    session = SimpleNamespace(hash="s1", last_memory_flush_at="", save=lambda **k: None)
    record_operator_stop(sessions_dir, "s1", mode=mode)

    fired, why = await agent.maybe_memory_flush(session)
    assert (fired, why, agent.turns) == (False, "session_stopped", [])

    # The user's next prompt lifts the Stop: the flush may run again later.
    w = SessionWriter(sessions_dir, "s1", engine="claude")
    w.turn_started(turn_idx=2, user="user prompts again")
    w.close()
    fired, why = await agent.maybe_memory_flush(session)
    assert (fired, agent.turns) == (True, ["memory_flush"])


# -- 3. a child that finished just before the Stop -------------------------------


@pytest.mark.parametrize("kind", ["cli", "core"])
@pytest.mark.asyncio
async def test_result_of_a_child_finished_before_the_stop_does_not_wake_the_parent(
    sessions_dir,
    registry,
    kind,
):
    from sharedagents_gateway.control_endpoint import (
        _owner_stopped_since_last_turn,
        _schedule_background_queue_recovery,
    )

    _session(sessions_dir, "parent")
    job = async_jobs.create(
        parent_sid="parent",
        agent="explore",
        task="t",
        on_complete="resume",
        call_id="c-done",
        pid=os.getpid(),
    )
    async_jobs.claim(job["job_id"], os.getpid())
    async_jobs.try_finalize(job["job_id"], "done", True)  # finished, not cancelled
    assert not _owner_stopped_since_last_turn("parent")

    await _backend(kind, sessions_dir, _fake_agent(sessions_dir)).cancel("parent", force=True)

    assert async_jobs.read(job["job_id"])["status"] == "done"  # nothing to cancel
    assert _owner_stopped_since_last_turn("parent")
    calls: list = []

    async def prompt(sid, _cwd, blocks):
        calls.append(sid)
        if False:  # pragma: no cover — async generator
            yield None

    removed: list = []

    async def queue_remove(sid, queue_id, *, reason=None):
        removed.append((queue_id, reason))
        return {"ok": True}

    backend = SimpleNamespace(prompt=prompt, queue_remove=queue_remove, load_session=None)
    items = [{"source": "background_task", "consume_ids": ["q1"], "text": "late result"}]
    assert _schedule_background_queue_recovery(backend=backend, sid="parent", pending=items)
    await asyncio.sleep(0.05)
    assert calls == []
    assert removed == [("q1", "background_purged")]


@pytest.mark.asyncio
async def test_graceful_stop_still_lets_children_deliver(sessions_dir, registry):
    from sharedagents_gateway.control_endpoint import _owner_stopped_since_last_turn

    _session(sessions_dir, "parent")
    await _cli_backend(_fake_agent(sessions_dir)).cancel("parent", force=False)
    assert operator_stop_mode(sessions_dir, "parent") == "graceful"
    assert not _owner_stopped_since_last_turn("parent")


@pytest.mark.asyncio
async def test_submit_drops_a_late_child_result_after_a_stop(sessions_dir, registry, monkeypatch):
    from starlette.requests import Request

    from sharedagents_gateway import control_endpoint

    _session(sessions_dir, "parent")
    await _cli_backend(_fake_agent(sessions_dir)).cancel("parent", force=True)
    backend = MagicMock()
    backend.queue_add = AsyncMock()
    ctx = SimpleNamespace(
        manager=SimpleNamespace(get=lambda _a: backend),
        is_known_agent=lambda _a: True,
        agent_names=["rt"],
        orchestrator_url=None,
    )
    monkeypatch.setattr(control_endpoint, "_ctx", lambda _r: ctx)
    monkeypatch.setattr(control_endpoint, "_owner_agent_for", lambda _s: "rt")
    body = json.dumps({"prompt": "late result", "background_job_id": "job_" + "0" * 16}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    req = Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [],
            "path": "/control/v1/orchestrator/sessions/parent/prompt",
            "path_params": {"hash": "parent"},
        },
        receive,
    )
    resp = await control_endpoint._handle_prompt_submit(req)

    assert resp.status_code == 202
    assert json.loads(resp.body)["rejected"] == "session_stopped"
    backend.queue_add.assert_not_called()


# -- 4. local-LLM cancel path ------------------------------------------------------


@pytest.mark.asyncio
async def test_local_llm_stop_purges_queues_and_stops_every_descendant(
    sessions_dir,
    registry,
    monkeypatch,
):
    _session(sessions_dir, "parent", state="waiting")
    _session(sessions_dir, "idle_child", parent="parent", state="idle")
    _session(sessions_dir, "busy_grandchild", parent="idle_child", state="streaming")
    killed: list = []
    monkeypatch.setattr(
        engine_pidfile,
        "kill_session_engine",
        lambda _d, sid: killed.append(sid) or True,
    )
    queue_items = {
        "parent": [
            {"queue_id": "qb", "source": "background_task", "text": "result"},
            {"queue_id": "qu", "source": "web", "text": "typed follow-up"},
        ],
        "idle_child": [{"queue_id": "qc", "source": "web", "text": "child follow-up"}],
    }
    agent = _fake_agent(sessions_dir, queue_items)

    await _core_backend(sessions_dir, agent).cancel("parent", force=True)

    assert "busy_grandchild" in killed
    assert _state(sessions_dir, "busy_grandchild") == "error"
    assert _state(sessions_dir, "idle_child") == "idle"  # finished: left alone
    reasons = {c.args[1]: c.kwargs.get("reason") for c in agent.queue_remove.call_args_list}
    assert reasons == {"qb": "background_purged", "qu": "user_deleted", "qc": "user_deleted"}
    for sid in ("parent", "idle_child", "busy_grandchild"):
        assert operator_stop_mode(sessions_dir, sid) == "hard"


# -- the Stop is on record before anything is killed ----------------------------------


@pytest.mark.parametrize("kind", ["cli", "core"])
@pytest.mark.asyncio
async def test_stop_is_recorded_on_each_descendant_before_its_engine_dies(
    sessions_dir,
    registry,
    monkeypatch,
    kind,
):
    _session(sessions_dir, "parent", state="waiting")
    _session(sessions_dir, "child", parent="parent", state="streaming")
    seen_at_kill: dict = {}
    monkeypatch.setattr(
        engine_pidfile,
        "kill_session_engine",
        lambda d, sid: seen_at_kill.setdefault(sid, operator_stop_mode(d, sid)) or True,
    )

    await _backend(kind, sessions_dir, _fake_agent(sessions_dir)).cancel("parent", force=True)

    assert seen_at_kill.get("child") == "hard"


@pytest.mark.asyncio
async def test_per_subagent_stop_records_the_stop_before_the_kill(sessions_dir, registry, monkeypatch):
    from starlette.requests import Request

    from sharedagents_gateway import control_endpoint

    _session(sessions_dir, "parent", state="waiting")
    _session(sessions_dir, "child", parent="parent", state="streaming")
    seen_at_kill: dict = {}
    monkeypatch.setattr(
        engine_pidfile,
        "kill_session_engine",
        lambda d, sid: seen_at_kill.setdefault(sid, operator_stop_mode(d, sid)) or True,
    )
    child_job = _job(registry, "child")
    parent_job = _job(registry, "parent")
    monkeypatch.setattr(control_endpoint, "_ctx", lambda _r: SimpleNamespace(agent_names=["rt"]))
    monkeypatch.setattr(control_endpoint, "_owner_agent_for", lambda _s: "rt")
    monkeypatch.setattr(control_endpoint, "_deliver_owner_abort_no_job", AsyncMock())

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    req = Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [],
            "path": "/control/v1/orchestrator/sessions/child/abort",
            "path_params": {"hash": "child"},
        },
        receive,
    )
    resp = await control_endpoint._handle_subagent_abort(req)

    assert resp.status_code == 202
    assert seen_at_kill == {"child": "hard"}
    assert operator_stop_mode(sessions_dir, "parent") == ""  # parent untouched
    assert registry.get_job(child_job["job_id"])["status"] == STATUS_PAUSED
    assert registry.get_job(parent_job["job_id"])["status"] == STATUS_RUNNING


@pytest.mark.parametrize("kind", ["cli", "core"])
@pytest.mark.asyncio
async def test_hard_stop_pauses_descendant_jobs_too(sessions_dir, registry, kind):
    _session(sessions_dir, "parent")
    _session(sessions_dir, "child", parent="parent")
    job = _job(registry, "child")
    await _backend(kind, sessions_dir, _fake_agent(sessions_dir)).cancel("parent", force=True)
    assert registry.get_job(job["job_id"])["status"] == STATUS_PAUSED


def test_writer_rechecks_stop_at_durable_admission(sessions_dir):
    from sharedagents_session.operator_stop import OperatorStopped, TurnStopGuard
    from sharedagents_session.session_reader import tail_jsonl

    _session(sessions_dir, "s1")
    guard = TurnStopGuard.capture(sessions_dir, "s1", "user")
    record_operator_stop(sessions_dir, "s1")
    w = SessionWriter(sessions_dir, "s1", engine="claude")
    try:
        with pytest.raises(OperatorStopped):
            w.open_turn(turn_idx=2, user="stale prompt", stop_guard=guard)
        assert w._open_turn is None
        assert len([e for e, _ in tail_jsonl(w.jsonl_path, 0) if e["type"] == "turn_started"]) == 1
        assert operator_stop_mode(sessions_dir, "s1") == "hard"
    finally:
        w.close()


@pytest.mark.asyncio
async def test_boot_honours_stop_even_if_crash_preceded_job_pause(sessions_dir, registry):
    _session(sessions_dir, "stopped")
    job = _job(registry, "stopped")
    record_operator_stop(sessions_dir, "stopped")
    sup = LoopSupervisor(SimpleNamespace(name="rt", cfg=SimpleNamespace(), sessions_dir=sessions_dir))
    sup.supervise_loop = AsyncMock(return_value={"ok": True})
    await sup.recover_running()
    sup.supervise_loop.assert_not_awaited()
    assert registry.get_job(job["job_id"])["status"] == STATUS_PAUSED
