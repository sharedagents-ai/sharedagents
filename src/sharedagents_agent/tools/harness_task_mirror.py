"""Mirror a subprocess engine's harness task/todo tool calls into sharedagents's
session stores.

For a ``claude_cli`` engine session, Claude Code's own ``TaskCreate`` /
``TaskUpdate`` / ``TaskStop`` / ``TodoWrite`` tools run INSIDE the subprocess.
sharedagents observes them only as ``tool_call`` / ``tool_result`` stream events and
never updates its OWN ``<sid>.tasks.json`` / ``<sid>.todos.json`` — so
``/task list`` and ``/todo`` return empty and every frontend's turn_ended
reconcile wipes the live task widget (names vanish, ghosts reappear next turn).

:class:`HarnessTaskMirror` watches the call/result pairs and writes them into the
same stores the native sharedagents Task*/Todo tools use, so the commands stay
authoritative for these sessions. Crucially it preserves CLAUDE's OWN task id
(parsed from the ``"Task #N created"`` result body) instead of sharedagents's
auto-increment, so a later ``TaskUpdate`` keyed on that id lands on the right row
rather than forking a nameless ghost.

Mirroring is best-effort: any failure is swallowed so a bad event can never break
the turn. Persistence reuses the runners' atomic-write helpers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sharedagents_agent.tools.runners import (
    _apply_link,
    _load_tasks,
    _normalize_task_id,
    _normalize_todo_items,
    _parse_id_list,
    _remove_link,
    _save_tasks,
    _save_todos,
)
from sharedagents_agent.tools.widget_adapt import adapt_todo_args, is_todo_tool_name

_TASK_TOOLS = frozenset({"TaskCreate", "TaskUpdate", "TaskStop"})
_ID_RE = re.compile(r"#(\d+)")
# Tag on every row this mirror writes, so a later create at a colliding id can
# Reserved metadata flag stamped on every row this mirror writes, so a later
# create at a colliding id — or any update/stop — can tell its own rows apart
# from operator/native ones it must not clobber. Kept separate from the
# user-facing ``metadata.source`` (which the model may set freely) so a model
# that passes its own ``source`` can't make the mirror disown its own row.
_MIRROR_FLAG = "_sharedagents_mirror"


def _subject(args: dict[str, Any]) -> str | None:
    """The task title, accepting the synonyms the runner accepts."""
    return args.get("subject") or args.get("title") or args.get("name") or args.get("content")


def _is_mirror_row(row: Any) -> bool:
    """True only for rows this mirror wrote (stamped with the reserved flag).
    Operator/native rows lack it and are therefore never mutated by the mirror."""
    if not isinstance(row, dict):
        return False
    meta = row.get("metadata")
    return isinstance(meta, dict) and meta.get(_MIRROR_FLAG) is True


def _delete_task_with_edges(state: dict[str, Any], tid: str) -> bool:
    """Remove a task and detach every reciprocal dependency edge, exactly as the
    native runner's ``status="deleted"`` branch does — so neighbours never keep a
    dangling ``blocks`` / ``blockedBy`` reference. Returns True if a row was
    removed."""
    if state["tasks"].get(tid) is None:
        return False
    _detach_edges(state, tid)
    del state["tasks"][tid]
    return True


def _detach_edges(state: dict[str, Any], tid: str) -> None:
    """Drop a row's reciprocal dependency edges from all its neighbours, leaving
    the row itself in place. Used before deleting a task and before overwriting a
    reused mirror id (so neighbours never keep a stale reference)."""
    task = state["tasks"].get(tid)
    if not isinstance(task, dict):
        return
    tid_int = int(tid)
    for child in list(task.get("blocks") or []):
        _remove_link(state, parent=tid_int, child=int(child))
    for parent in list(task.get("blockedBy") or []):
        _remove_link(state, parent=int(parent), child=tid_int)


def _task_id(args: dict[str, Any]) -> str | None:
    """The update key, accepting taskId / task_id / id. Delegates to the native
    normalizer so it matches the runner exactly — including the ``"#3"`` summary
    shape Claude copies from task listings."""
    for key in ("taskId", "task_id", "id"):
        val = args.get(key)
        if val is not None:
            nid = _normalize_task_id(val)
            return str(nid) if nid is not None else None
    return None


def _link_edge_if_both_mirror(state: dict[str, Any], *, parent: int, child: int, remove: bool = False) -> None:
    """Apply/remove a dependency edge ONLY when both endpoints are mirror-owned.
    Guards against an edge touching an operator/native row whose id collided with
    a subprocess id (which would show a false dependency in /task list)."""
    tasks = state.get("tasks") or {}
    if _is_mirror_row(tasks.get(str(parent))) and _is_mirror_row(tasks.get(str(child))):
        (_remove_link if remove else _apply_link)(state, parent=parent, child=child)


class HarnessTaskMirror:
    """Reflect subprocess Task*/TodoWrite tool events into the session stores.

    Drive it with :meth:`on_tool_call` (when a tool fires) and
    :meth:`on_tool_result` (when it completes) — the same call_id ties the two
    together so a ``TaskCreate``'s assigned id can be learned from its result.
    """

    def __init__(self, *, tasks_path: Path, todos_path: Path) -> None:
        self.tasks_path = tasks_path
        self.todos_path = todos_path
        # call_id → (tool_name, arguments) awaiting its result.
        self._pending: dict[str, tuple[str, dict[str, Any]]] = {}

    def on_tool_call(self, name: str, call_id: str, args: Any) -> None:
        if name in _TASK_TOOLS or is_todo_tool_name(name):
            self._pending[str(call_id)] = (
                name,
                dict(args) if isinstance(args, dict) else {},
            )

    def on_tool_result(self, call_id: str, content: Any, ok: Any) -> None:
        item = self._pending.pop(str(call_id), None)
        if item is None:
            return
        name, args = item
        if ok is False:  # the subprocess tool errored — mirror nothing
            return
        try:
            if name == "TaskCreate":
                self._mirror_create(args, content)
            elif name == "TaskUpdate":
                self._mirror_update(args)
            elif name == "TaskStop":
                self._mirror_stop(args)
            elif is_todo_tool_name(name):
                self._mirror_todos(args)
        except Exception:
            # Best-effort only: a mirroring failure must never break the turn.
            pass

    # -- tasks -----------------------------------------------------------------
    def _mirror_create(self, args: dict[str, Any], content: Any) -> None:
        m = _ID_RE.search(str(content or ""))
        if not m:
            return  # no id in the result body → can't key the row; skip
        tid = m.group(1)
        state = _load_tasks(self.tasks_path)
        existing = state["tasks"].get(tid)
        if existing is not None and not _is_mirror_row(existing):
            # The subprocess's id-counter collided with an operator/native task
            # already in this store (e.g. /task add). Never clobber it — its
            # data would be lost and /task list would go wrong. Skip the mirror;
            # the live web upsert still shows the task mid-turn.
            return
        if existing is not None:
            # Reused id for a brand-new task — detach the old row's edges first so
            # neighbours don't keep a reciprocal reference to the replacement.
            _detach_edges(state, tid)
        subject = _subject(args)
        metadata = dict(args.get("metadata")) if isinstance(args.get("metadata"), dict) else {}
        metadata[_MIRROR_FLAG] = True  # forced — never let the model's metadata disown us
        state["tasks"][tid] = {
            "id": int(tid),
            "subject": str(subject).strip() if subject else "",
            "description": str(args.get("description") or "").strip(),
            "activeForm": str(args.get("activeForm") or "").strip(),
            "status": "pending",
            "owner": str(args.get("owner") or "").strip(),
            "blocks": [],
            "blockedBy": [],
            "metadata": metadata,
        }
        state["next_id"] = max(int(state.get("next_id") or 1), int(tid) + 1)
        # Apply blockedBy edges (a supported TaskCreate field) so /task list and
        # the in_progress gate reflect real dependencies. Best-effort: a blocker
        # that isn't in the mirrored store yet is silently skipped (_apply_link
        # no-ops on a missing id) rather than failing the whole mirror.
        blocked_by, err = _parse_id_list(args.get("blockedBy"), field="blockedBy")
        if err is None and blocked_by:
            for blocker_id in blocked_by:
                _link_edge_if_both_mirror(state, parent=blocker_id, child=int(tid))
        _save_tasks(self.tasks_path, state)

    def _mirror_update(self, args: dict[str, Any]) -> None:
        tid = _task_id(args)
        if tid is None:
            return
        state = _load_tasks(self.tasks_path)
        existing = state["tasks"].get(tid)
        if existing is not None and not _is_mirror_row(existing):
            # The id belongs to an operator/native task (collision) — never
            # mutate or delete it on the subprocess's behalf.
            return
        status = args.get("status")
        if status == "deleted":
            if _delete_task_with_edges(state, tid):
                _save_tasks(self.tasks_path, state)
            return
        task = existing
        if task is None:
            # An update for a row we never saw created (e.g. created in a prior
            # turn before the mirror existed). Synthesize it so the widget still
            # reflects reality rather than dropping the event.
            subject = _subject(args)
            task = {
                "id": int(tid),
                "subject": str(subject).strip() if subject else "",
                "description": "",
                "activeForm": "",
                "status": "pending",
                "owner": "",
                "blocks": [],
                "blockedBy": [],
                "metadata": {_MIRROR_FLAG: True},
            }
            state["tasks"][tid] = task
            state["next_id"] = max(int(state.get("next_id") or 1), int(tid) + 1)
        subject = _subject(args)
        if subject:
            task["subject"] = str(subject).strip()
        if isinstance(status, str) and status:
            task["status"] = status
        # Mirror the other supported scalar fields when present (a missing key is
        # left untouched; the native TaskUpdate only writes what was passed).
        for field in ("description", "activeForm", "owner"):
            if isinstance(args.get(field), str):
                task[field] = args[field].strip()
        if isinstance(args.get("metadata"), dict):
            meta = task.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            # Native TaskUpdate semantics: a key set to null DELETES it; any other
            # value is set/overwritten.
            for k, v in args["metadata"].items():
                if v is None:
                    meta.pop(k, None)
                else:
                    meta[k] = v
            meta[_MIRROR_FLAG] = True  # never let an update disown the row
            task["metadata"] = meta
        # Mirror the supported dependency-edit fields so /task list reflects the
        # graph Claude actually built. Best-effort; a missing neighbour no-ops.
        tid_int = int(tid)
        add_blocks, e = _parse_id_list(args.get("addBlocks"), field="addBlocks")
        if e is None:
            for b in add_blocks:
                _link_edge_if_both_mirror(state, parent=tid_int, child=b)
        add_blocked_by, e = _parse_id_list(args.get("addBlockedBy"), field="addBlockedBy")
        if e is None:
            for x in add_blocked_by:
                _link_edge_if_both_mirror(state, parent=x, child=tid_int)
        rm_blocks, e = _parse_id_list(args.get("removeBlocks"), field="removeBlocks")
        if e is None:
            for b in rm_blocks:
                _link_edge_if_both_mirror(state, parent=tid_int, child=b, remove=True)
        rm_blocked_by, e = _parse_id_list(args.get("removeBlockedBy"), field="removeBlockedBy")
        if e is None:
            for x in rm_blocked_by:
                _link_edge_if_both_mirror(state, parent=x, child=tid_int, remove=True)
        _save_tasks(self.tasks_path, state)

    def _mirror_stop(self, args: dict[str, Any]) -> None:
        tid = _task_id(args)
        if tid is None:
            return
        state = _load_tasks(self.tasks_path)
        existing = state["tasks"].get(tid)
        if existing is not None and not _is_mirror_row(existing):
            return  # collision with an operator/native task — leave it alone
        if _delete_task_with_edges(state, tid):
            _save_tasks(self.tasks_path, state)

    # -- todos -----------------------------------------------------------------
    def _mirror_todos(self, args: dict[str, Any]) -> None:
        adapted = adapt_todo_args(args) or {}
        todos = adapted.get("todos")
        if not isinstance(todos, list):
            return  # only the Claude TodoWrite wholesale-replace shape
        normalized, err = _normalize_todo_items(todos)
        if err is not None or normalized is None:
            return
        _save_todos(self.todos_path, normalized)
