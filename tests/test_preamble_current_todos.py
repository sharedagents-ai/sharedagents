"""Regression: the current-todos block must reach the round preamble."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sharedagents_agent.agents.system_prompt import SystemPromptMixin
from sharedagents_agent.tools.impl.planning import todos_path_for_session


def _fake_agent(sessions_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        sessions_dir=sessions_dir,
        memory_recall_context=lambda session, text: "",
    )


def test_todos_file_reaches_preamble(tmp_path: Path) -> None:
    sid = "abc123"
    todos_path_for_session(tmp_path, sid).write_text(
        json.dumps(
            [
                {"content": "write the parser", "status": "completed"},
                {"content": "add tests", "status": "pending"},
            ]
        ),
        encoding="utf-8",
    )
    session = SimpleNamespace(hash=sid, pinned_facts=[], compaction_summary="")

    out = SystemPromptMixin.compose_round_preamble(_fake_agent(tmp_path), session, "hi")

    todo_blocks = [m for m in out if str(m.get("content", "")).startswith("[CURRENT TODO LIST")]
    assert len(todo_blocks) == 1
    body = todo_blocks[0]["content"]
    assert todo_blocks[0]["role"] == "user"
    # OpenCode-style ``completed`` renders as checked, like ``done``.
    assert "- [x] 0: write the parser" in body
    assert "- [ ] 1: add tests" in body


def test_no_todos_file_no_block(tmp_path: Path) -> None:
    session = SimpleNamespace(hash="none", pinned_facts=[], compaction_summary="")
    out = SystemPromptMixin.compose_round_preamble(_fake_agent(tmp_path), session, "hi")
    assert not any(str(m.get("content", "")).startswith("[CURRENT TODO LIST") for m in out)
