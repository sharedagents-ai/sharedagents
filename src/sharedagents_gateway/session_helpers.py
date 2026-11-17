"""Session-store read helpers for :class:`CoreBackend`.

Extracted from ``acp_bridge.py`` (behavior-preserving de-monolith).

* ``list_sessions_acp`` — the shared ACP-shape ``session/list`` impl,
  used by ``CoreBackend.list_sessions`` and the CLI-engine fallback in
  ``gateway_shim.py``.
* ``_turn_has_terminal_event`` — durable JSONL check that a turn already
  ended (guards against appending a synthetic aborted close over a real
  terminal event on a hard-cancel race).
* ``_loads_json_list`` — tolerant JSON-array TEXT-column parser.

Re-exported from ``sharedagents_gateway.acp_bridge`` for import stability
(``gateway_shim`` and tests import ``list_sessions_acp`` by that path).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _loads_json_list(v: Any) -> list:
    """Parse a JSON-array TEXT column into a list; tolerate list/null/bad."""
    if isinstance(v, list):
        return v
    if not v:
        return []
    try:
        out = json.loads(v)
        return out if isinstance(out, list) else []
    except (ValueError, TypeError):
        return []


def _loads_json_dict(v: Any) -> dict:
    """Parse a JSON-object TEXT column into a dict; tolerate dict/null/bad."""
    if isinstance(v, dict):
        return v
    if not v:
        return {}
    try:
        out = json.loads(v)
        return out if isinstance(out, dict) else {}
    except (ValueError, TypeError):
        return {}


def list_sessions_acp(
    sessions_dir: Path,
    agent_home: Path,
    *,
    max_sessions: int = 20,
) -> dict[str, Any]:
    """Shared ACP-shape ``session/list`` impl, used by both:

    - :meth:`CoreBackend.list_sessions` — the local-LLM agent path
      (e.g., ``main``).
    - :meth:`AgentACPBackend.list_sessions` CLI-engine fallback in
      ``gateway_shim.py`` — the path for ``claude`` / ``codex`` /
      ``gemini`` agents whose engine doesn't carry a ``CoreBackend``.

    Returns the wire shape the ACP SDK validates against:
    ``{"sessions": [SessionInfo, ...], "nextCursor": None}``. CLI-engine
    agents previously fell through to :meth:`Agent.list_sessions`
    (which returns sharedagents-internal field names — ``hash`` / ``turn_count``
    / ``created_at``) and the resulting payload failed SDK validation,
    so the TUI's session picker silently returned an empty list.

    Rules (— reads ``sessions.db``):

    1. One SQL ``SELECT`` from ``sessions_meta`` (no per-file glob/parse).
    2. Drop empty-turn sessions (``last_turn_idx < 0`` AND no in-flight
       turn — created by ``/new`` + disconnect; no replay value). A
       session whose first turn is still streaming (or was interrupted
       mid-turn) keeps ``live_turn_idx`` set while ``last_turn_idx`` is
       still ``-1``; those carry real transcript and ARE listed, so the
       operator can resume them and so the *current* session shows up in
       its own ``/sessions`` picker mid-conversation.
    3. ACP ``SessionInfo`` requires a non-empty absolute ``cwd``; fall
       back to ``agent_home`` when ``workspace_cwd`` is empty.
    4. Most-recent first by last activity — ``MAX(last_turn_ts,
       updated_at)`` so a live mid-turn session (``last_turn_ts == 0``,
       but ``updated_at`` bumped on every event) sorts up instead of
       sinking below every finished one; cap at ``max_sessions``.
    5. ``updatedAt`` carries that same last-activity time (as ISO-8601),
       NOT ``created_at`` — every ``/sessions`` surface renders it as the
       relative age, so stamping it with creation time made a correctly
       *ordered* list *look* shuffled (a session created weeks ago but
       last touched a minute ago showed "weeks ago" at the top). The
       raw ``created_at`` and ``last_activity`` ms still ride along in
       ``_meta.sharedagents`` for callers that want them.
    """
    from sharedagents_session.session_index import SessionIndex, connect_sessions_db

    def _ms_to_iso(ms: int) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()

    candidates: list[dict[str, Any]] = []
    db = sessions_dir / "sessions.db"
    if not db.exists() and not sessions_dir.is_dir():
        return {"sessions": [], "nextCursor": None}
    # Scope to the caller's agent. ``sessions.db`` is shared across
    # every agent under the unified layout, so without this filter
    # ``session/list`` for ``claude`` would expose ``main``'s sessions
    # (and vice-versa) in the TUI / Telegram picker. The ``agent_home``
    # parameter's basename matches the writer's
    # ``session_started.agent_name`` for every legit producer.
    owner_agent = agent_home.name
    try:
        SessionIndex(sessions_dir).refresh()
        con = connect_sessions_db(db)
        try:
            cur = con.execute(
                "SELECT session_hash, title, workspace_cwd, created_at, "
                "engine, last_turn_idx, last_turn_ts, updated_at, "
                "live_turn_idx, agent_name "
                "FROM sessions_meta "
                "WHERE (last_turn_idx >= 0 OR live_turn_idx IS NOT NULL) "
                "  AND (agent_name = ? OR agent_name = '') "
                "ORDER BY MAX(last_turn_ts, updated_at) DESC, "
                "created_at DESC LIMIT ?",
                (owner_agent, int(max_sessions)),
            )
            for r in cur.fetchall():
                # ``turn_count`` reflects completed turns, but when a turn
                # is still in flight (``live_turn_idx`` set while
                # ``last_turn_idx`` is its pre-turn value) count the
                # in-progress turn too — otherwise a session mid-
                # conversation shows "0 turns" in the picker.
                turn_count = max(0, int(r["last_turn_idx"]) + 1)
                live_idx = r["live_turn_idx"]
                if live_idx is not None:
                    turn_count = max(turn_count, int(live_idx) + 1)
                # Same key the ORDER BY sorts on, so the rendered age is
                # monotonic with list position. Fall back to created_at
                # for a brand-new session whose epochs are still 0.
                last_activity_ms = max(
                    int(r["last_turn_ts"] or 0),
                    int(r["updated_at"] or 0),
                )
                updated_at = _ms_to_iso(last_activity_ms) if last_activity_ms > 0 else (r["created_at"] or "")
                candidates.append(
                    {
                        "sessionId": r["session_hash"],
                        "cwd": r["workspace_cwd"] or str(agent_home),
                        "title": r["title"] or f"untitled-{(r['session_hash'] or '')[-8:]}",
                        "updatedAt": updated_at,
                        "_meta": {
                            "sharedagents": {
                                "turn_count": turn_count,
                                "engine": r["engine"] or "",
                                "created_at": r["created_at"] or "",
                                "last_activity_ms": last_activity_ms,
                            },
                        },
                    }
                )
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return {"sessions": [], "nextCursor": None}
    return {"sessions": candidates, "nextCursor": None}


def _turn_has_terminal_event(
    sessions_dir: Path,
    session_id: str,
    turn_idx: int,
) -> bool:
    """True if the JSONL already contains ``turn_ended`` for this turn.

    A hard-cancel can race immediately after a completed turn: the client
    closes, connection teardown calls ``close_session()``, and a stale
    ``sessions_meta`` live row may still say the turn is streaming. The JSONL
    is the durable source of truth, so never append a synthetic aborted close
    over an existing terminal event.
    """
    path = sessions_dir / f"{session_id}.jsonl"
    if not path.exists():
        return False
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            brace = line.find("{")
            if brace < 0:
                continue
            try:
                ev = json.loads(line[brace:])
            except (TypeError, ValueError):
                continue
            if ev.get("type") == "turn_ended" and int(ev.get("turn_idx") or 0) == int(turn_idx):
                return True
    except OSError:
        return False
    return False
