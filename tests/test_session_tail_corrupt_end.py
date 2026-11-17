"""SessionIndex.tail steps past a corrupt tail but waits on a torn last line."""

from __future__ import annotations

import logging


def _cursor(idx, sid):
    con = idx._connect()
    try:
        row = con.execute("SELECT indexed_bytes FROM sessions_meta WHERE session_hash=?", (sid,)).fetchone()
    finally:
        con.close()
    return int(row["indexed_bytes"])


def test_file_ending_in_corrupt_lines_is_read_once(tmp_path, caplog):
    from sharedagents_session.session_index import SessionIndex
    from sharedagents_session.session_reader import encode_line

    good = encode_line({"type": "session_started", "agent_name": "build"})
    turn = encode_line({"type": "turn_started", "turn_idx": 0, "user": "work"})
    path = tmp_path / "corruptEnd.jsonl"
    path.write_bytes(good + b"00000000" + turn[8:] + b"not a record\n")
    idx = SessionIndex(tmp_path)

    with caplog.at_level(logging.WARNING):
        assert idx.refresh() == 1
        assert _cursor(idx, "corruptEnd") == path.stat().st_size
        assert idx.refresh() == 0
    dropped = [r for r in caplog.records if "dropped corrupt line" in r.getMessage()]
    assert len(dropped) == 2


def test_torn_last_line_waits_for_its_newline(tmp_path):
    from sharedagents_session.session_index import SessionIndex
    from sharedagents_session.session_reader import encode_line

    good = encode_line({"type": "session_started", "agent_name": "build"})
    turn = encode_line({"type": "turn_started", "turn_idx": 0, "user": "work"})
    path = tmp_path / "torn.jsonl"
    path.write_bytes(good + turn[:20])
    idx = SessionIndex(tmp_path)

    assert idx.tail("torn") == 1
    assert _cursor(idx, "torn") == len(good)
    with path.open("ab") as fp:
        fp.write(turn[20:])
    assert idx.tail("torn") == 1
    assert _cursor(idx, "torn") == path.stat().st_size
