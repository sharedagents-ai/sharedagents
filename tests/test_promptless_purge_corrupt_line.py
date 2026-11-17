"""Boot purge of promptless sessions never deletes a log with a corrupt line."""

from __future__ import annotations

import os
import time

import pytest

from sharedagents_agent import orchestrator as orch


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.setenv("SHAREDAGENTS_AGENTS_HOME", str(tmp_path / "agents"))
    monkeypatch.delenv("SHAREDAGENTS_SESSIONS_DIR", raising=False)
    root = tmp_path / "sessions"
    root.mkdir()
    return root


def test_corrupt_turn_started_survives_and_empty_is_removed(sessions):
    from sharedagents_session.session_index import SessionIndex
    from sharedagents_session.session_reader import encode_line

    header = encode_line({"type": "session_started", "agent_name": "build"})
    turn = encode_line({"type": "turn_started", "turn_idx": 0, "user": "work"})
    corrupt_turn = b"00000000" + turn[8:]  # same payload, wrong CRC
    corrupt = sessions / "corruptTurn.jsonl"
    empty = sessions / "trulyEmpty.jsonl"
    corrupt.write_bytes(header + corrupt_turn)
    empty.write_bytes(header)
    past = time.time() - orch.PROMPTLESS_MIN_AGE_SECONDS - 60
    for p in (corrupt, empty):
        os.utime(p, (past, past))
    SessionIndex(sessions).refresh()

    assert orch.delete_promptless_sessions() == 1
    assert corrupt.exists()
    assert not empty.exists()
