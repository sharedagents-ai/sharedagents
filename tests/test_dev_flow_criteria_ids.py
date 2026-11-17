"""Stage-0 criteria use the REQ-SDLC-* IDs; the former prefix is rejected."""

import copy

import pytest

from sharedagents_web.dev_flow import CRITERION_IDS, FLOWCHART, FlowError, FlowRegistry

HUMAN = {
    "role": "owner",
    "actor_kind": "human",
    "profile": "web",
    "engine": "none",
    "model_id": "none",
    "model_version": "none",
    "session_id": "s1",
    "context_id": "c1",
}


def _create(registry, ids):
    return registry.create_flow(
        loop_id="loop-1",
        repo="example/repo",
        branch="main",
        issue_number=None,
        issue_url="",
        requirements="Do the thing.",
        flowchart=copy.deepcopy(FLOWCHART),
        criteria=[{"id": item, "text": f"criterion {item}"} for item in ids],
        actor=dict(HUMAN),
    )


def test_criterion_ids_are_the_ten_req_sdlc_ids():
    assert [f"REQ-SDLC-{i:04d}" for i in range(1, 11)] == CRITERION_IDS


def test_create_flow_accepts_the_current_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_GIT_TREE_ID", "0" * 40)
    flow = _create(FlowRegistry(tmp_path / "flows.sqlite3"), CRITERION_IDS)
    assert flow


def test_create_flow_rejects_the_former_id_prefix(tmp_path):
    legacy = ["OLD-" + item for item in CRITERION_IDS]
    with pytest.raises(FlowError, match="stable ratified IDs"):
        _create(FlowRegistry(tmp_path / "flows.sqlite3"), legacy)
