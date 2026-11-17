"""Bind a Dev SDLC run to one already-implemented Jarvis chat + Tasks row.

No synthetic transcript. Chat is the standard session; ``sdlc`` is the lock.
"""

from __future__ import annotations

from sharedagents_web.server import store
from sharedagents_web.task_board import get_task_board_registry


def register_dev_conductor(
    *,
    owner: str,
    headline: str,
    session_id: str = "",
    agent: str = "jarvis",
) -> dict[str, str]:
    """Reuse or mint the Jarvis conductor session and track it.

    Title stays the default placeholder so the title agent can name it after
    the first real turn. The Tasks row has no pinned ``label`` so the board
    shows that same auto session title.
    """
    sid = str(session_id or "").strip()
    sdata = store.get(sid, owner) if sid else None
    if sdata is None:
        sdata = store.create(owner, agent, "", "dev")
        sid = str(sdata["id"])
    elif (sdata.get("topic") or "") != "dev":
        sdata["topic"] = "dev"
        store._save(sdata)
    # Empty label: Tasks falls through to the session title (placeholder,
    # then the title agent's name). A typed object used to pin the board
    # forever and hide the auto title.
    _ = headline
    task_id = get_task_board_registry().ensure(
        sid,
        owner=owner or None,
        label="",
    )
    return {"conductor_session_id": sid, "task_id": task_id}


def sync_run_title_from_session(run: dict, *, owner: str) -> dict:
    """Copy a real auto session title onto the run headline (and keep label empty)."""
    from sharedagents_session.session_title import is_default_title
    from sharedagents_web.dev_sdlc import get_sdlc_store

    sid = str(run.get("conductor_session_id") or "").strip()
    run_id = str(run.get("run_id") or "").strip()
    if not sid or not run_id:
        return run
    try:
        sdata = store.get(sid, owner) or store.get(sid) or {}
    except Exception:  # noqa: BLE001 — missing store must not hide the run
        return run
    title = str((sdata or {}).get("title") or "").strip()
    if not title or is_default_title(title):
        return run
    if str(run.get("headline") or "") == title:
        return run
    patched = get_sdlc_store().patch(run_id, headline=title)
    return patched or {**run, "headline": title}


def ensure_run_conductor(run: dict, *, owner: str) -> dict:
    """Attach a conductor session + task row to ``run`` if it has none."""
    from sharedagents_web.dev_sdlc import get_sdlc_store

    run_id = str(run.get("run_id") or "")
    headline = str(run.get("headline") or "Dev")
    existing = str(run.get("conductor_session_id") or "").strip()
    bound = register_dev_conductor(
        owner=owner,
        headline=headline,
        session_id=existing,
    )
    sid = bound["conductor_session_id"]
    try:
        sdata = store.get(sid)
        if sdata and run_id and sdata.get("sdlc_run_id") != run_id:
            sdata["sdlc_run_id"] = run_id
            store._save(sdata)
    except Exception:  # noqa: BLE001 — Chat fold has a conductor-id fallback
        pass
    if not existing:
        try:
            from sharedagents_agent.tools.impl.sdlc import seed_board_for_run
            from sharedagents_core.kernel.layout import sessions_root

            seed_board_for_run(sessions_root(), sid, {**run, "run_id": run_id})
        except Exception:  # noqa: BLE001 — card file is optional at mint
            pass
    if existing != sid and run_id:
        run = get_sdlc_store().patch(
            run_id,
            conductor_session_id=sid,
        ) or {**run, "conductor_session_id": sid}
    elif existing != sid:
        run = dict(run)
        run["conductor_session_id"] = sid
    return run
