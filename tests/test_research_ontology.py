"""Real SharedData transactions, evidence validation and native completion gates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import permutations

import pytest

from sharedagents_research.native_tools import publication_required, runtime_context
from sharedagents_research.ontology import external_identity, grounded_quote, identity, open_ontology


@pytest.fixture
def store(tmp_path):
    pytest.importorskip("SharedData.SQLite")
    return open_ontology(tmp_path, "test-user")


def begin(store, run="run-1"):
    return store.begin(
        run, actor="research-user", question="What is Acme's capacity?", model="test-model", prompt_version="test-v1"
    )


def observation(store, run="run-1", value=100, *, index=1, quote=None, event_at="2026-01-01T10:00:00Z", corrects=None):
    text = quote or f"Acme's capacity is {value} units."
    source = store.capture(
        run,
        "research-user",
        url=f"https://example.com/{index}",
        text=text,
        published_at=f"2026-01-{index:02d}T11:00:00Z",
    )
    result = store.extract(
        run,
        "research-user",
        source["id"],
        entities=[{"key": "a", "name": "Acme", "type": "company", "quote": text}],
        concepts=[
            {"key": "capacity", "name": "capacity", "description": "Reported production capacity", "quote": text}
        ],
        claims=[
            {
                "subject": "a",
                "predicate": "capacity",
                "value": value,
                "unit": "units",
                "quote": text,
                "event_at": event_at,
                "corrects": corrects,
            }
        ],
    )
    return source, result


def finish(store, source, run="run-1"):
    return store.finish(
        run,
        "research-user",
        answer="Reported capacity.",
        citations=[source["id"]],
        limitations=["Source report, not independently verified."],
    )


def test_research_compounds_and_keeps_historical_receipt(store):
    begin(store)
    source, result = observation(store)
    original = finish(store, source)
    begin(store, "run-2")
    second, _ = observation(store, "run-2", value=120, index=2)
    latest = finish(store, second, "run-2")
    assert original["revision"] != latest["revision"]
    old_state = store.get(next(iter(original["states"].values())))
    new_state = store.get(next(iter(latest["states"].values())))
    assert old_state["value"] == 100
    assert new_state["status"] == "contested" and new_state["value"] is None
    assert len(new_state["alternatives"]) == 2
    assert store.get(original["revision"])["states"] == original["states"]
    assert store.search("capacity", kind="concept")
    assert store.status()["published_runs"] == 2
    assert store.export()[1]["research"]["revision"] == latest["revision"]
    assert store.get(result["claims"][0])["epistemic_status"] == "reported"
    begin(store, "run-3")
    reused = store.finish("run-3", "research-user", answer="Reused evidence.", citations=[source["id"]], limitations=[])
    assert reused["source_ids"] == [source["id"]] and reused["captured_source_ids"] == []
    assert reused["states"] == latest["states"]


def test_unextracted_source_blocks_success_and_retries_do_not_rewrite(store):
    begin(store)
    source = store.capture("run-1", "research-user", url="https://example.com/source", text="Acme.")
    with pytest.raises(ValueError, match="extraction"):
        finish(store, source)
    assert store.status()["published_runs"] == 0
    store.extract("run-1", "research-user", source["id"], entities=[], concepts=[], claims=[])
    original = finish(store, source)
    before = store.status()
    assert finish(store, source) == original
    assert store.status() == before
    with pytest.raises(ValueError, match="immutable"):
        store.finish("run-1", "research-user", answer="Changed", citations=[source["id"]], limitations=[])


def test_bad_quote_rolls_back_the_entire_extraction(store):
    begin(store)
    source = store.capture("run-1", "research-user", url="https://example.com/source", text="Acme has capacity.")
    before = store.status()
    with pytest.raises(ValueError, match="exact source quote"):
        store.extract(
            "run-1",
            "research-user",
            source["id"],
            entities=[{"key": "a", "name": "Acme", "quote": "Acme"}],
            concepts=[{"key": "b", "name": "capacity", "quote": "invented quote"}],
            claims=[],
        )
    assert store.status() == before


def test_retrieval_pages_source_text_without_rewriting_evidence(store):
    begin(store)
    text = "capacity " * 3000
    source = store.capture("run-1", "research-user", url="https://example.com/long", text=text)
    hit = store.search("capacity")[0]["object"]
    assert hit["excerpt"] and len(hit["text"]) == 1500
    first = store.get(source["id"], text_limit=16000)
    second = store.get(source["id"], offset=first["next_offset"], text_limit=16000)
    assert first["text"] + second["text"] == text
    assert store.get(source["id"])["text"] == text


def test_eight_concurrent_writers_share_consistent_indexes(store):
    def work(index):
        run = f"parallel-{index}"
        begin(store, run)
        source, _ = observation(store, run, value=100, index=index + 1)
        return finish(store, source, run)

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(work, range(8)))
    assert len({r["revision"] for r in receipts}) == 8
    status = store.status()
    assert status["published_runs"] == 8
    assert status["objects"]["entity"] == status["objects"]["concept"] == status["facts"] == 1
    assert status["objects"]["claim"] == 8


def test_correction_supersedes_exact_syndicated_copies(store):
    begin(store)
    first, claims = observation(store)
    observation(store, index=2)
    observation(store, index=3, value=120, corrects=claims["claims"][0])
    result = finish(store, first)
    state = store.get(next(iter(result["states"].values())))
    assert state["value"] == 120
    assert len(state["superseded"]) == 2


def test_correction_published_earlier_cannot_erase_later_report(store):
    begin(store)
    _, claim = observation(store, index=3)
    with pytest.raises(ValueError, match="earlier same-publisher"):
        observation(store, index=1, value=120, corrects=claim["claims"][0])
    assert store.status()["objects"]["claim"] == 1


def test_correction_chain_converges_for_every_arrival_order(store, tmp_path):
    begin(store)
    _, first = observation(store, index=1)
    _, second = observation(store, index=2, value=120, corrects=first["claims"][0])
    last, _ = observation(store, index=3, value=140, corrects=second["claims"][0])
    expected = store.get(next(iter(finish(store, last)["states"].values())))
    assert expected["value"] == 140 and len(expected["superseded"]) == 2
    targets = {1: None, 2: first["claims"][0], 3: second["claims"][0]}
    for order in permutations((1, 2, 3)):
        candidate = open_ontology(tmp_path / ("order-" + "".join(map(str, order))), "test-user")
        begin(candidate)
        for index in order:
            source, _ = observation(candidate, index=index, value=80 + 20 * index, corrects=targets[index])
        receipt = finish(candidate, source)
        actual = candidate.get(next(iter(receipt["states"].values())))
        assert actual == expected


def test_event_order_uses_utc_and_preserves_ties(store):
    begin(store)
    observation(store, index=2, event_at="2026-01-01T10:00:00Z")
    observation(store, index=1, event_at="2026-01-01T11:00:00+01:00")
    observation(store, index=3, event_at="2026-01-01T09:00:00Z")
    events = store.timeline()
    assert events[0]["event_at"] < events[1]["event_at"]
    assert events[1]["ordering"] == "undetermined" and len(events[1]["claims"]) == 2


def test_ontology_reopens_and_users_have_separate_stores(store, tmp_path):
    begin(store)
    source, _ = observation(store)
    receipt = finish(store, source)
    reopened = open_ontology(tmp_path, "test-user", create=False)
    assert reopened.get(receipt["revision"]) == receipt
    other = open_ontology(tmp_path, "other-user")
    assert other.get(source["id"]) is None and other.search("Acme") == []
    assert store.container.path.stat().st_mode & 0o077 == 0


def test_actor_is_checked_and_fake_citations_are_rejected(store):
    begin(store)
    with pytest.raises(PermissionError):
        store.capture("run-1", "other-actor", url="https://example.com/source", text="content")
    with pytest.raises(ValueError, match="citations"):
        store.finish("run-1", "research-user", answer="Claim.", citations=["not-stored"], limitations=[])
    with pytest.raises(ValueError, match="coverage limitation"):
        store.finish("run-1", "research-user", answer="No coverage.", citations=[], limitations=[])
    result = store.finish(
        "run-1", "research-user", answer="No coverage.", citations=[], limitations=["No evidence found."]
    )
    assert result["states"] == {} and store.status()["published_runs"] == 1


def test_publication_guard_requires_successful_finish_in_this_turn():
    call = {"id": "finish", "name": "ontology_update", "arguments": {"action": "finish"}}
    assert publication_required("research", [], [])
    assert publication_required("research", [call], [{"id": "finish", "ok": False}])
    assert publication_required("research", [call], [{"id": "another-call", "ok": True}])
    done = {"id": "finish", "name": "ontology_update", "action": "finish", "ok": True}
    assert not publication_required("research", [call], [done])
    assert publication_required("research", [call], [done, {"name": "ontology_source", "ok": True}])
    assert publication_required(
        "research",
        [call],
        [
            {"id": "finish", "name": "ontology_update", "action": "begin", "ok": True},
            {"id": "finish", "name": "ontology_update", "action": "finish", "ok": False},
        ],
    )
    assert not publication_required("build", [], [])


def test_runtime_context_is_pinned_to_authenticated_user(monkeypatch, tmp_path):
    monkeypatch.setenv("SHAREDAGENTS_SESSION_ID", "test-session")
    monkeypatch.setenv("SHAREDAGENTS_TURN_ID", "test-turn")
    monkeypatch.setenv("SHAREDAGENTS_SESSIONS_DIR", str(tmp_path))
    monkeypatch.setenv(
        "SHAREDAGENTS_ACCESS_PIN", '{"sid":"test-session","stamp":{"email":"user@example.com","grants":{}}}'
    )
    first = runtime_context()
    assert first["scope"] == "user:user@example.com"
    monkeypatch.setenv("SHAREDAGENTS_TURN_ID", "next-turn")
    assert runtime_context()["run_id"] != first["run_id"]
    assert first["run_id"] == identity(first["scope"], "test-session", "test-turn")
    monkeypatch.setenv("SHAREDAGENTS_ACCESS_PIN", "invalid")
    with pytest.raises(PermissionError, match="malformed"):
        runtime_context()


def test_quote_grounding_accepts_only_whitespace_changes():
    text = "A source describes its\n  actual findings."
    assert grounded_quote(text, "source describes its actual findings.") == "source describes its\n  actual findings."
    with pytest.raises(ValueError, match="exact source quote"):
        grounded_quote(text, "source describes imaginary findings.")


def test_explicit_repository_identity_survives_label_and_type_variants(store):
    begin(store)
    source_url = "https://raw.githubusercontent.com/example/project/main/README.md"
    variants = ["example/project", "github.com/example/project", "https://github.com/example/project/"]
    for index, external in enumerate(variants):
        source = store.capture("run-1", "research-user", url=source_url, text=f"Project label {index}.")
        store.extract(
            "run-1",
            "research-user",
            source["id"],
            entities=[
                {
                    "key": "entity",
                    "name": f"Project label {index}",
                    "type": f"classification-{index}",
                    "external_id": external,
                    "quote": f"Project label {index}.",
                }
            ],
            concepts=[],
            claims=[],
        )
    assert store.status()["objects"]["entity"] == 1
    assert len(store.search("Project label", kind="entity")) == 1
    assert store.status()["objects"]["definition"] == 3
    # Repository context does not turn an arbitrary document into a project.
    assert external_identity(source_url, source_url) == source_url


def test_native_ontology_capabilities_do_not_grant_arbitrary_file_access():
    from sharedagents_agent.folder_sandbox import needs_sandbox
    from sharedagents_core.access_grants import AccessStamp, Grants

    stamp = AccessStamp("user@example.com", Grants.none())
    for name in ("ontology_search", "ontology_source", "ontology_update"):
        assert not needs_sandbox(stamp, name)
    for name in ("read", "write", "bash", "mcp__unknown__write"):
        assert needs_sandbox(stamp, name)


@pytest.mark.asyncio
async def test_coordinator_runs_eight_native_sessions_and_carries_access():
    import asyncio
    from types import SimpleNamespace

    from sharedagents_agent.agents.engine import TurnDone
    from sharedagents_research.coordinator import research_many

    class NativeAgent:
        cfg = SimpleNamespace(engine="local_llm", agent_profile="research", name="research")
        created = 0
        active = 0
        peak = 0
        ready = asyncio.Event()

        def create_session(self, **kwargs):
            self.created += 1
            assert kwargs["parent_session_id"] == "parent"
            return SimpleNamespace(hash=str(self.created))

        async def run_turn(self, sid, brief, *, access, actors):
            assert access == "pinned-access" and actors == ["actor"]
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.active == 8:
                self.ready.set()
            await asyncio.wait_for(self.ready.wait(), timeout=2)
            self.active -= 1
            yield TurnDone(state="ok")

    agent = NativeAgent()
    results = await research_many(
        agent, ["brief"] * 8, parallel=8, access="pinned-access", actors=["actor"], parent_session_id="parent"
    )
    assert agent.peak == 8 and agent.active == 0
    assert len(results) == 8 and all(row["state"] == "ok" for row in results)


def test_explicit_sdk_worker_uses_shareddata_without_agent_dependencies(tmp_path):
    import os

    from sharedagents_research.storage_client import OntologyProcess

    python = os.environ.get("SHAREDAGENTS_TEST_SDK_PYTHON")
    if not python:
        pytest.skip("set SHAREDAGENTS_TEST_SDK_PYTHON for explicit SDK worker integration")
    worker = OntologyProcess(python, tmp_path, "test-worker", True)
    begin(worker)
    source, _ = observation(worker)
    result = finish(worker, source)
    assert worker.get(result["revision"]) == result
    assert worker.status()["published_runs"] == 1
    assert worker.search("capacity")
    with pytest.raises(PermissionError):
        worker.check_open("run-1", "wrong-actor")


def test_explicit_setup_preserves_runtime_settings_and_checks_backend_first(tmp_path, monkeypatch):
    import json
    import os

    from sharedagents_research.__main__ import configure_research

    python = os.environ.get("SHAREDAGENTS_TEST_SDK_PYTHON")
    if not python:
        pytest.skip("set SHAREDAGENTS_TEST_SDK_PYTHON for setup integration")
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.delenv("SHAREDAGENTS_ONTOLOGY_ROOT", raising=False)
    home = tmp_path / "agents" / "research"
    home.mkdir(parents=True)
    path = home / "config.json"
    original = {"engine": "local_llm", "name": "research", "upstream_model": "preserve-route", "custom": 17}
    path.write_text(json.dumps(original))
    (tmp_path / "config.json").write_text(json.dumps({"custom_global": True}))
    (home / "AGENTS.md").write_text("Legacy research instructions")
    with pytest.raises(OSError):
        configure_research(home, backend_python=str(tmp_path / "missing-interpreter"))
    assert json.loads(path.read_text()) == original
    assert (home / "AGENTS.md").read_text() == "Legacy research instructions"
    configure_research(home, backend_python=python)
    configured = json.loads(path.read_text())
    assert configured["upstream_model"] == "preserve-route" and configured["custom"] == 17
    assert {"ontology_update", "ontology_source", "ontology_search"} <= set(configured["tools_allowlist"])
    assert configured["skill_allowlist"] == []
    assert json.loads((tmp_path / "config.json").read_text())["custom_global"] is True
    assert "ontology_update" in (home / "AGENTS.md").read_text()


@pytest.mark.asyncio
async def test_native_engine_rejects_prose_only_completion(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from sharedagents_agent.agents.engine import EngineKind, EngineSessionState, TurnDone, TurnError, TurnRequest
    from sharedagents_agent.agents.local_llm import LocalLLMEngine
    from sharedagents_session.session import Session

    monkeypatch.setenv("SHAREDAGENTS_SESSIONS_DIR", str(tmp_path / "sessions"))
    cfg = SimpleNamespace(
        name="research",
        agent_profile="research",
        upstream_model="test-model",
        upstream_endpoint="https://example.com/v1",
        upstream_max_context=1000000,
        upstream_max_completion_tokens=None,
        compact_at_frac=0.85,
        compact_min_turns=3,
        models=[],
        sessions_dir=tmp_path / "sessions",
    )
    engine = LocalLLMEngine(cfg)
    session = Session(
        hash="research-publication-test",
        title="Research",
        created_at="2026-01-01T00:00:00Z",
        system_prompt="Research",
        model=cfg.upstream_model,
        endpoint=cfg.upstream_endpoint,
        context_window=1000000,
        system_mode="auto",
        agent_name="research",
    )
    monkeypatch.setattr(session, "save", Mock())
    engine._sessions[session.hash] = session
    monkeypatch.setattr(engine, "_log_event", Mock())
    monkeypatch.setattr(engine, "_log_round", Mock())
    monkeypatch.setattr(engine, "_teacher_enabled", lambda *args: False)
    spawned = []

    async def spawn(engine, session, messages, turn_id, env):
        spawned.append(messages)

        async def stream():
            yield {"kind": "content", "text": "I finished the research and updated the ontology with all findings."}
            yield {
                "kind": "metrics",
                "finish_reason": "stop",
                "terminal_finish_reason_seen": True,
                "prompt_tokens": 100,
                "completion_tokens": 15,
            }
            yield {"kind": "done"}

        return SimpleNamespace(proc=None, stderr_task=None), stream()

    monkeypatch.setattr(engine, "_worker_spawner", spawn)
    monkeypatch.setattr(
        engine, "_build_messages_simple", lambda *a, **kw: [{"role": "user", "content": "What did you learn?"}]
    )
    request = TurnRequest(prompt="What did you learn?", extra={"session": session})
    events = [
        event
        async for event in engine.run_turn(
            request,
            EngineSessionState(session.hash, EngineKind.LOCAL_LLM),
            sessions_dir=cfg.sessions_dir,
            turn_id="test-turn",
        )
    ]
    assert isinstance(events[-1], TurnDone) and events[-1].state == "error"
    assert any(isinstance(event, TurnError) and "ontology" in event.message for event in events)
    assert len(spawned) == 3
