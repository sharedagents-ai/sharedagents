"""Extracted tool-runner family: planning (de-monolith of runners.py).

Behavior-preserving split — code moved verbatim.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ._utils import (
    ToolResult,
    _err,
    _truncate,
)

# ---- todo (session-scoped checklist) ----

# Persisted at `<sessions_dir>/<sid>.todos.json`. The runner resolves
# that path from `SHAREDAGENTS_SESSION_ID` + `SHAREDAGENTS_SESSIONS_DIR` (set by
# `_spawn_tool_runner` on every tool subprocess). The agent runtime
# also reads + formats this file directly for the `/todo` slash command,
# via `todos_path_for_session` and `format_todo_md` below.

_TODO_VALID_STATUSES: frozenset[str] = frozenset({"pending", "in_progress", "done", "completed", "cancelled"})


def _canonical_todo_status(status: Any) -> str:
    """Normalize a status value across sharedagents legacy and OpenCode todowrite.

    Both ``done`` and ``completed`` collapse to the sharedagents-internal
    ``done`` so downstream renderers / persistence stay stable;
    ``in_progress`` and OpenCode's ``cancelled`` pass through. Anything
    else falls back to ``pending``.
    """
    if not isinstance(status, str):
        return "pending"
    s = status.strip().lower()
    if s == "completed":
        return "done"
    if s in ("pending", "in_progress", "done", "cancelled"):
        return s
    return "pending"


def todos_path_for_session(sessions_dir: Path, sid: str) -> Path:
    """The session's todo file. Mirrors `<sid>.delegate_*.json` paths."""
    return sessions_dir / f"{sid}.todos.json"


def _todos_path_from_env() -> Path | None:
    sid = os.environ.get("SHAREDAGENTS_SESSION_ID")
    sdir = os.environ.get("SHAREDAGENTS_SESSIONS_DIR")
    if not sid or not sdir:
        return None
    return todos_path_for_session(Path(sdir), sid)


def load_todos(path: Path) -> list[dict[str, Any]]:
    """Read the todo list. Missing/corrupt → empty list."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        if not isinstance(content, str) or not content:
            continue
        status = entry.get("status")
        if status not in _TODO_VALID_STATUSES:
            status = "pending"
        item: dict[str, Any] = {"content": content, "status": status}
        priority = entry.get("priority")
        if priority in ("high", "medium", "low"):
            item["priority"] = priority
        active_form = entry.get("activeForm")
        if isinstance(active_form, str) and active_form:
            item["activeForm"] = active_form
        created_at = entry.get("created_at")
        if isinstance(created_at, str) and created_at:
            item["created_at"] = created_at
        out.append(item)
    return out


def _save_todos(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(items, indent=2, ensure_ascii=False) + "\n"
    # Atomic write: tmp + fsync + os.replace, same pattern Session.save
    # uses. Without this, a SIGTERM mid-write leaves an empty / partial
    # JSON the next `load_todos` discards as corrupt → silent reset of
    # the model's todo list.
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def format_todo_json(items: list[dict[str, Any]]) -> str:
    """Render the todo list as OpenCode ``todowrite`` output.

    OpenCode returns the todos as pretty-printed JSON (``JSON.stringify(todos,
    null, 2)``) with each item shaped ``{content, status, priority}``. The
    sharedagents-internal ``done`` status is surfaced as OpenCode's ``completed``;
    a missing priority defaults to ``medium``.
    """
    display: list[dict[str, Any]] = []
    for t in items:
        status = t.get("status") or "pending"
        if status == "done":
            status = "completed"
        display.append(
            {
                "content": (t.get("content") or "").strip(),
                "status": status,
                "priority": t.get("priority") or "medium",
            }
        )
    return json.dumps(display, indent=2, ensure_ascii=False)


def format_todo_md(items: list[dict[str, Any]]) -> str:
    """Render the list as the markdown checklist the model + UI see.

    Mirrors Claude Code's TodoWrite render: completed steps get ``[x]``,
    in_progress gets ``[~]`` (sharedagents extension; keeps the marker
    distinct from done), pending gets ``[ ]``. The trailing message
    matches Claude's "Todos have been modified successfully" so a
    delegated /claude turn and a native sharedagents turn produce
    indistinguishable confirmations.
    """
    if not items:
        return "Todos have been modified successfully. (list is empty)"
    lines = ["## Todo List", ""]
    for i, t in enumerate(items):
        status = t.get("status") or "pending"
        if status == "done":
            mark = "x"
        elif status == "in_progress":
            mark = "~"
        else:
            mark = " "
        content = (t.get("content") or "").strip()
        lines.append(f"- [{mark}] {i}: {content}")
    lines.append("")
    lines.append(
        "Todos have been modified successfully. Ensure that you continue to use the todo list to track your progress."
    )
    return "\n".join(lines)


def _normalize_todo_items(raw: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Coerce the model's todo items into ``{content, status, activeForm,
    created_at}`` dicts.

    Accepts:
    - plain strings (pending),
    - legacy ``{content, status?}`` dicts,
    - Claude TodoWrite ``{content, activeForm, status}`` dicts (where
      ``status`` ∈ {pending, in_progress, completed}).

    Returns ``(items, error)`` — exactly one is non-None.
    """
    if not isinstance(raw, list):
        return (
            None,
            "todos/items must be a list of strings or {content, activeForm, status} objects",
        )
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[dict[str, Any]] = []
    for it in raw:
        if isinstance(it, str):
            content = it.strip()
            if not content:
                return None, "items must not contain empty strings"
            out.append(
                {
                    "content": content,
                    "status": "pending",
                    "priority": "medium",
                    "activeForm": "",
                    "created_at": now,
                }
            )
            continue
        if isinstance(it, dict):
            content = it.get("content")
            if not isinstance(content, str) or not content.strip():
                return None, "each item dict must have non-empty content"
            status = _canonical_todo_status(it.get("status"))
            # OpenCode todowrite item = {content, status, priority}. Keep the
            # legacy sharedagents ``activeForm`` too so the TUI plan panel (which
            # still reads it, defaulting to "") keeps rendering.
            priority = it.get("priority")
            if priority not in ("high", "medium", "low"):
                priority = "medium"
            active_form = it.get("activeForm") or it.get("active_form") or ""
            if not isinstance(active_form, str):
                active_form = ""
            out.append(
                {
                    "content": content.strip(),
                    "status": status,
                    "priority": priority,
                    "activeForm": active_form.strip(),
                    "created_at": now,
                }
            )
            continue
        return None, "items must be strings or {content, status, priority} objects"
    return out, None


def run_todo(args: dict[str, Any]) -> ToolResult:
    """Session-scoped checklist tool — Claude Code ``TodoWrite`` shape.

    Primary primitive: pass ``todos: [{content, activeForm, status}]``
    and the runner overwrites the persisted list with this state.
    Status ∈ {``pending``, ``in_progress``, ``completed``}.
    sharedagents stores ``done`` internally for completed items so the
    legacy ``format_todo_md`` keeps rendering, but accepts both
    ``done`` and ``completed`` from the model.

    Legacy shape ``action="add|replace|complete|delete|list", items,
    indexes`` is still honored for callers that haven't migrated.

    Persists to ``<sessions_dir>/<sid>.todos.json``.
    """
    todos_arg = args.get("todos")
    action = args.get("action")
    if isinstance(todos_arg, list) and not action:
        # Claude TodoWrite shape — wholesale replace.
        action = "replace"
        items_raw = todos_arg
    else:
        items_raw = args.get("items")
    if action not in ("add", "replace", "complete", "delete", "list"):
        return _err(
            "either pass `todos: [...]` (Claude TodoWrite shape) or `action ∈ {add, replace, complete, delete, list}`"
        )
    path = _todos_path_from_env()
    if path is None:
        return _err("session todo file unresolved — SHAREDAGENTS_SESSION_ID / SHAREDAGENTS_SESSIONS_DIR not set in env")
    items = load_todos(path)
    if action in ("add", "replace"):
        normalized, err = _normalize_todo_items(items_raw)
        if err is not None:
            return _err(err)
        assert normalized is not None
        if action == "add":
            items = items + normalized
        else:
            # replace: if the incoming content list matches the
            # persisted one in order, treat it as a state-merge instead
            # of an overwrite. The model occasionally re-emits the whole
            # plan via `replace` (sometimes with all items pre-marked
            # done) when it should have used `complete`. A naive
            # overwrite either wipes existing done flags (when items
            # come back as "pending") or duplicates the plan in the TUI.
            # Merging keeps `created_at`, ORs the `done` status, and
            # leaves the persisted file representing actual progress.
            existing_contents = [it.get("content") for it in items]
            new_contents = [it.get("content") for it in normalized]
            if items and existing_contents == new_contents:
                merged: list[dict[str, Any]] = []
                for old, new in zip(items, normalized):
                    old_st = old.get("status")
                    new_st = new.get("status")
                    # Preserve "done" against accidental reversion when a model
                    # re-emits a plan with items defaulted back to "pending".
                    # However, honor explicit transitions to "cancelled",
                    # "in_progress", or "done".
                    if new_st in ("done", "cancelled", "in_progress"):
                        status = new_st
                    elif old_st == "done":
                        status = "done"
                    else:
                        status = new_st or old_st or "pending"
                    out_item = {**old, **new, "status": status}
                    if "created_at" in old:
                        out_item["created_at"] = old["created_at"]
                    elif "created_at" in new:
                        out_item["created_at"] = new["created_at"]
                    merged.append(out_item)
                items = merged
            else:
                items = normalized
    elif action in ("complete", "delete"):
        idxs_raw = args.get("indexes")
        if not isinstance(idxs_raw, list) or not idxs_raw:
            return _err("indexes must be a non-empty list of integers")
        idxs: list[int] = []
        for i in idxs_raw:
            if not isinstance(i, int) or isinstance(i, bool):
                return _err("indexes must contain integers")
            if i < 0 or i >= len(items):
                return _err(f"index out of range: {i} (have {len(items)} item(s))")
            idxs.append(i)
        if action == "complete":
            for i in idxs:
                items[i] = {**items[i], "status": "done"}
        else:
            drop = set(idxs)
            items = [t for n, t in enumerate(items) if n not in drop]
    if action != "list":
        try:
            _save_todos(path, items)
        except OSError as e:
            return _err(f"failed to write {path}: {e}")
    # OpenCode ``todowrite`` output = the todos as pretty JSON (title would be
    # ``<N> todos``; sharedagents ToolResult has no title field, so the JSON body is
    # the model-facing content).
    return ToolResult(content=format_todo_json(items), ok=True)


# ---- Task family (harness task tracker) ----

# Mirrors Claude Code's TaskCreate / TaskUpdate / TaskList / TaskGet
# tool family but persisted to the same per-session sidecar pattern
# used by TodoWrite. Distinct from TodoWrite:
#   - TodoWrite holds the model's working plan (single list, the
#     model rewrites it each turn).
#   - Task* holds a persistent task tracker with stable numeric ids,
#     subject + description + status, owner, and block/blockedBy
#     relations. Survives across turns and shows in the
#     ``#harness-tasks`` TUI footer.
#
# State file: ``<sessions_dir>/<sid>.tasks.json``. Shape:
#   {"next_id": int, "tasks": {id_str: {subject, description,
#     activeForm, status, owner, blocks: [id...], blockedBy: [id...],
#     metadata: {...}}}}


_TASK_VALID_STATUSES: frozenset[str] = frozenset({"pending", "in_progress", "completed", "deleted"})


def _tasks_path_from_env() -> Path | None:
    sid = os.environ.get("SHAREDAGENTS_SESSION_ID")
    sdir = os.environ.get("SHAREDAGENTS_SESSIONS_DIR")
    if not sid or not sdir:
        return None
    return Path(sdir) / f"{sid}.tasks.json"


def _load_tasks(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"next_id": 1, "tasks": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"next_id": 1, "tasks": {}}
    if not isinstance(data, dict):
        return {"next_id": 1, "tasks": {}}
    if not isinstance(data.get("tasks"), dict):
        data["tasks"] = {}
    if not isinstance(data.get("next_id"), int) or data["next_id"] < 1:
        data["next_id"] = (
            max(
                (int(k) for k in data["tasks"] if str(k).isdigit()),
                default=0,
            )
            + 1
        )
    return data


def _save_tasks(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _task_summary(t: dict[str, Any]) -> str:
    blockers = t.get("blockedBy") or []
    blocker_str = ""
    if blockers:
        blocker_str = f" blockedBy={','.join(str(b) for b in blockers)}"
    owner = t.get("owner") or ""
    owner_str = f" owner={owner}" if owner else ""
    return f"#{t.get('id')} [{t.get('status') or 'pending'}] {(t.get('subject') or '').strip()}{owner_str}{blocker_str}"


# ---------------------------------------------------------------------
# Task* dependency helpers — Claude-Code-parity DAG semantics on the
# session-scoped task store.
#
# Invariant: ``blocks`` and ``blockedBy`` are reverse edges of the same
# graph. If A.blocks contains B then B.blockedBy contains A, and vice
# versa. All mutation paths route through :func:`_apply_link` /
# :func:`_remove_link` so the two columns can't drift.
#
# Cycle policy: :func:`_would_cycle` BFS up the existing blockedBy
# chain from ``parent`` looking for ``child``; if reachable, adding
# ``parent.blocks=[child]`` (equivalently ``child.blockedBy=[parent]``)
# would close a cycle and is refused with a structured error the model
# can read.
#
# Status policy: :func:`_pending_blockers` returns blocker ids that
# aren't ``completed``. ``status="in_progress"`` is refused while
# pending blockers exist — the error names them so the model knows
# which tasks to finish first. ``completed`` is allowed (model may
# legitimately decide a task is moot mid-stream) but a soft warning is
# included in the result body.
# ---------------------------------------------------------------------


def _normalize_task_id(raw: Any) -> int | None:
    """Accept str / int / float-shaped-int → canonical int. Anything
    else returns None.

    Small models tend to emit ids as the string ``"3"`` or as a JSON
    integer; both must work without the wrapper having to second-guess.
    Strings tolerating a leading ``"#"`` (``"#3"``) match the format
    :func:`_task_summary` prints.
    """
    if isinstance(raw, bool):  # bool is a subclass of int in Python
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, float) and raw.is_integer() and raw > 0:
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip().lstrip("#")
        if s.isdigit():
            v = int(s)
            return v if v > 0 else None
    return None


def _parse_id_list(value: Any, *, field: str) -> tuple[list[int] | None, str | None]:
    """Parse a list of task ids. Tolerates mixed str / int. Returns
    ``(ids, error)`` — exactly one is ``None``. Empty list is valid.
    """
    if value is None:
        return [], None
    if not isinstance(value, list):
        return None, (f"`{field}` must be a list of task ids (got {type(value).__name__})")
    out: list[int] = []
    seen: set[int] = set()
    for item in value:
        nid = _normalize_task_id(item)
        if nid is None:
            return None, (f"`{field}` contains invalid task id {item!r} — expected positive integers")
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out, None


def _would_cycle(state: dict[str, Any], parent: int, child: int) -> bool:
    """Return True if making ``parent`` block ``child`` (equivalently
    making ``child`` blocked-by ``parent``) would close a cycle.

    BFS upward from ``parent`` via ``blockedBy`` edges; if we reach
    ``child`` then ``child → … → parent``, so adding ``parent → child``
    closes the loop.

    A self-edge (parent == child) is always a cycle.
    """
    if parent == child:
        return True
    tasks = state.get("tasks") or {}
    visited: set[int] = set()
    queue: list[int] = [parent]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        if node == child:
            return True
        node_row = tasks.get(str(node))
        if not isinstance(node_row, dict):
            continue
        for up in node_row.get("blockedBy") or []:
            try:
                up_int = int(up)
            except (TypeError, ValueError):
                continue
            if up_int not in visited:
                queue.append(up_int)
    return False


def _apply_link(state: dict[str, Any], parent: int, child: int) -> None:
    """Idempotently add the ``parent → child`` edge in both directions.

    No-op when either id is missing from ``state["tasks"]`` (the caller
    is expected to validate first, but be defensive — silent skip beats
    a half-applied edge).
    """
    tasks = state.get("tasks") or {}
    parent_row = tasks.get(str(parent))
    child_row = tasks.get(str(child))
    if not isinstance(parent_row, dict) or not isinstance(child_row, dict):
        return
    blocks = {int(x) for x in (parent_row.get("blocks") or [])}
    blocks.add(child)
    parent_row["blocks"] = sorted(blocks)
    blocked_by = {int(x) for x in (child_row.get("blockedBy") or [])}
    blocked_by.add(parent)
    child_row["blockedBy"] = sorted(blocked_by)


def _remove_link(state: dict[str, Any], parent: int, child: int) -> None:
    """Idempotently remove the ``parent → child`` edge from both
    directions. No-op if the edge doesn't exist.
    """
    tasks = state.get("tasks") or {}
    parent_row = tasks.get(str(parent))
    child_row = tasks.get(str(child))
    if isinstance(parent_row, dict):
        blocks = [int(x) for x in (parent_row.get("blocks") or []) if int(x) != child]
        parent_row["blocks"] = sorted(set(blocks))
    if isinstance(child_row, dict):
        blocked_by = [int(x) for x in (child_row.get("blockedBy") or []) if int(x) != parent]
        child_row["blockedBy"] = sorted(set(blocked_by))


def _graph_has_cycle(state: dict[str, Any]) -> bool:
    """Walk the whole ``blockedBy`` graph for any cycle. Three-colour
    DFS: WHITE = unvisited, GREY = on current path, BLACK = fully
    explored. A back-edge to a GREY node means a cycle is present.

    Catches the cross-field case the per-edge :func:`_would_cycle`
    misses: a single ``TaskUpdate`` with ``addBlocks=[B]`` AND
    ``addBlockedBy=[B]`` against a fresh A/B pair adds two acyclic
    edges individually but their union forms a 2-cycle. The post-
    apply guard runs this against the projected state.

    O(V + E) and the in-memory graph is small (one session's
    tasks), so the cost is negligible compared to the JSON
    round-trip on save.
    """
    tasks = state.get("tasks") or {}
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[int, int] = {}

    def visit(node: int) -> bool:
        c = colour.get(node, WHITE)
        if c == GREY:
            return True
        if c == BLACK:
            return False
        colour[node] = GREY
        row = tasks.get(str(node))
        if isinstance(row, dict):
            for raw in row.get("blockedBy") or []:
                try:
                    parent = int(raw)
                except (TypeError, ValueError):
                    continue
                if visit(parent):
                    return True
        colour[node] = BLACK
        return False

    for k in list(tasks.keys()):
        try:
            n = int(k)
        except (TypeError, ValueError):
            continue
        if colour.get(n, WHITE) == WHITE and visit(n):
            return True
    return False


def _pending_blockers(state: dict[str, Any], tid: int) -> list[int]:
    """Return blocker ids that aren't yet ``completed``. Excludes
    deleted blockers (the row is gone) and self-references (a malformed
    state — ignore silently). The result is sorted for stable error
    messages.
    """
    tasks = state.get("tasks") or {}
    row = tasks.get(str(tid))
    if not isinstance(row, dict):
        return []
    pending: list[int] = []
    for raw in row.get("blockedBy") or []:
        try:
            bid = int(raw)
        except (TypeError, ValueError):
            continue
        if bid == tid:
            continue
        blocker = tasks.get(str(bid))
        if not isinstance(blocker, dict):
            # Blocker was deleted — treat as resolved (no longer blocks).
            continue
        if blocker.get("status") != "completed":
            pending.append(bid)
    return sorted(set(pending))


def run_task_create(args: dict[str, Any]) -> ToolResult:
    """Create a new task in the session-scoped task store.

    Required: ``subject`` (synonyms ``title`` / ``name`` /
    ``content`` accepted). Optional: ``description``,
    ``activeForm``, ``owner``, ``metadata`` (dict),
    ``blockedBy`` (list[int]) — task ids that must reach
    ``status=completed`` before this task may transition to
    ``in_progress``. Status starts at ``pending``. Assigned id is
    returned in the result body so the TUI can parse it for the
    harness-task footer.
    """
    # Accept the common alternates that smaller-context LLMs reach
    # for first (deepseek-v4-pro / qwen / kimi tend to emit
    # ``title`` or ``name`` when the schema says ``subject``). The
    # canonical key remains ``subject``.
    subject = args.get("subject") or args.get("title") or args.get("name") or args.get("content")
    if not isinstance(subject, str) or not subject.strip():
        return _err("`subject` (non-empty string) required — synonyms title / name / content also accepted")
    description = args.get("description")
    if description is not None and not isinstance(description, str):
        return _err("`description` must be a string")
    active_form = args.get("activeForm")
    if active_form is not None and not isinstance(active_form, str):
        return _err("`activeForm` must be a string")
    owner = args.get("owner")
    if owner is not None and not isinstance(owner, str):
        return _err("`owner` must be a string")
    metadata = args.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return _err("`metadata` must be an object")
    blocked_by, err = _parse_id_list(args.get("blockedBy"), field="blockedBy")
    if err is not None:
        return _err(err)
    path = _tasks_path_from_env()
    if path is None:
        return _err("session task file unresolved — SHAREDAGENTS_SESSION_ID / SHAREDAGENTS_SESSIONS_DIR not set in env")
    state = _load_tasks(path)
    # Validate every blocker id exists BEFORE allocating a new id so a
    # bad call doesn't burn the counter.
    if blocked_by:
        missing = [b for b in blocked_by if str(b) not in state["tasks"]]
        if missing:
            return _err(f"`blockedBy` references unknown task ids: {sorted(missing)}")
    tid = int(state.get("next_id") or 1)
    state["next_id"] = tid + 1
    task = {
        "id": tid,
        "subject": subject.strip(),
        "description": (description or "").strip(),
        "activeForm": (active_form or "").strip(),
        "status": "pending",
        "owner": (owner or "").strip(),
        "blocks": [],
        "blockedBy": [],
        "metadata": metadata or {},
    }
    state["tasks"][str(tid)] = task
    # Mirror the blockedBy edges so the parents' ``blocks`` lists
    # stay in sync — same code path TaskUpdate uses for ``addBlockedBy``.
    # No cycle check possible against the brand-new id (no row could
    # already point at it), so just apply.
    for blocker_id in blocked_by:
        _apply_link(state, parent=blocker_id, child=tid)
    try:
        _save_tasks(path, state)
    except OSError as e:
        return _err(f"failed to write {path}: {e}")
    blocked_str = ""
    if blocked_by:
        blocked_str = f" — blocked by #{','.join(str(b) for b in sorted(blocked_by))}"
    return ToolResult(
        content=f"Task #{tid} created successfully: {subject.strip()}{blocked_str}",
        ok=True,
    )


def run_task_update(args: dict[str, Any]) -> ToolResult:
    """Update an existing task. Required: ``taskId`` (synonyms
    ``task_id`` / ``id``). Optional: ``status``, ``subject``,
    ``description``, ``activeForm``, ``owner``, ``metadata`` (merged),
    ``addBlocks`` / ``removeBlocks`` (list[int]),
    ``addBlockedBy`` / ``removeBlockedBy`` (list[int]).
    ``status="deleted"`` removes the task and detaches all its edges.

    Dependency rules:
     - ``addBlocks=[B]`` adds an edge ``this → B`` and mirrors
       ``B.blockedBy += [this]``. Refused if it would close a cycle.
     - ``addBlockedBy=[X]`` adds an edge ``X → this`` and mirrors
       ``X.blocks += [this]``. Refused if it would close a cycle.
     - ``removeBlocks`` / ``removeBlockedBy`` drop the edge in both
       directions. Missing edges are a silent no-op.
     - ``status="in_progress"`` is refused while ``blockedBy`` lists
       any non-completed task — the error names them so the model
       can finish them first.
    """
    tid_raw = (
        args.get("taskId")
        if args.get("taskId") is not None
        else args.get("task_id")
        if args.get("task_id") is not None
        else args.get("id")
    )
    tid_int = _normalize_task_id(tid_raw)
    if tid_int is None:
        return _err("`taskId` required (synonyms task_id / id also accepted) — expected a positive integer")
    tid = str(tid_int)
    path = _tasks_path_from_env()
    if path is None:
        return _err("session task file unresolved — SHAREDAGENTS_SESSION_ID / SHAREDAGENTS_SESSIONS_DIR not set in env")
    state = _load_tasks(path)
    task = state["tasks"].get(tid)
    if task is None:
        return _err(f"unknown taskId: {tid}")

    # status=deleted is its own branch — also detach every edge so
    # neighbours don't end up with dangling references.
    raw_status = args.get("status")
    if raw_status == "deleted":
        for child in list(task.get("blocks") or []):
            _remove_link(state, parent=tid_int, child=int(child))
        for parent in list(task.get("blockedBy") or []):
            _remove_link(state, parent=int(parent), child=tid_int)
        del state["tasks"][tid]
        try:
            _save_tasks(path, state)
        except OSError as e:
            return _err(f"failed to write {path}: {e}")
        return ToolResult(content=f"Task #{tid} deleted", ok=True)

    # Pre-parse all edge-update fields so a bad input fails the whole
    # call before any state mutation.
    add_blocks, err = _parse_id_list(args.get("addBlocks"), field="addBlocks")
    if err is not None:
        return _err(err)
    add_blocked_by, err = _parse_id_list(args.get("addBlockedBy"), field="addBlockedBy")
    if err is not None:
        return _err(err)
    remove_blocks, err = _parse_id_list(args.get("removeBlocks"), field="removeBlocks")
    if err is not None:
        return _err(err)
    remove_blocked_by, err = _parse_id_list(args.get("removeBlockedBy"), field="removeBlockedBy")
    if err is not None:
        return _err(err)

    # Every referenced id must exist as a task. The model should hear
    # about unknown ids before we touch any state.
    referenced = set(add_blocks) | set(add_blocked_by) | set(remove_blocks) | set(remove_blocked_by)
    missing = [x for x in referenced if str(x) not in state["tasks"]]
    if missing:
        return _err(f"unknown task ids referenced: {sorted(missing)} — use TaskList to see existing ids")

    # Cycle check: simulate each requested add against the existing
    # graph. _would_cycle BFS-walks the blockedBy chain — same
    # algorithm as kanban_db's link_tasks.
    for b in add_blocks:
        if _would_cycle(state, parent=tid_int, child=b):
            return _err(
                f"refusing addBlocks=#{b}: would create a cycle "
                f"(this task #{tid_int} is already transitively "
                f"blocked by #{b})"
            )
    for b in add_blocked_by:
        if _would_cycle(state, parent=b, child=tid_int):
            return _err(
                f"refusing addBlockedBy=#{b}: would create a cycle "
                f"(task #{b} is already transitively blocked by "
                f"this task #{tid_int})"
            )

    # Status enforcement. "in_progress" requires every blocker to be
    # completed. Project the proposed state (after the edge updates)
    # so a single call that adds a blocker AND moves to in_progress
    # is judged on the post-call graph, not the pre-call one.
    if raw_status is not None and raw_status not in _TASK_VALID_STATUSES:
        return _err(f"`status` must be one of {sorted(_TASK_VALID_STATUSES)}")

    # Apply edge updates first (so status enforcement sees the
    # new edge set).
    for b in remove_blocks:
        _remove_link(state, parent=tid_int, child=b)
    for b in remove_blocked_by:
        _remove_link(state, parent=b, child=tid_int)
    for b in add_blocks:
        _apply_link(state, parent=tid_int, child=b)
    for b in add_blocked_by:
        _apply_link(state, parent=b, child=tid_int)

    # Cross-field cycle guard. Per-edge _would_cycle above catches
    # cycles against the pre-call graph, but a single call with
    # ``addBlocks=[B]`` + ``addBlockedBy=[B]`` adds two acyclic
    # edges (each individually fine) whose union forms a 2-cycle.
    # The post-apply graph walk catches that and any larger
    # combinations of mixed-direction adds.
    if _graph_has_cycle(state):
        for b in add_blocks:
            _remove_link(state, parent=tid_int, child=b)
        for b in add_blocked_by:
            _remove_link(state, parent=b, child=tid_int)
        for b in remove_blocks:
            _apply_link(state, parent=tid_int, child=b)
        for b in remove_blocked_by:
            _apply_link(state, parent=b, child=tid_int)
        return _err(
            f"refusing edge updates on task #{tid_int}: the "
            f"combination of addBlocks={sorted(add_blocks)} + "
            f"addBlockedBy={sorted(add_blocked_by)} would create "
            "a cycle in the task graph. Apply one direction at a "
            "time so the conflict is visible per-edge."
        )

    if raw_status == "in_progress":
        pending = _pending_blockers(state, tid_int)
        if pending:
            # Roll back the edge updates we just applied so the call is
            # transactional from the model's perspective: a refused
            # status flip leaves the graph as it was at call entry.
            for b in add_blocks:
                _remove_link(state, parent=tid_int, child=b)
            for b in add_blocked_by:
                _remove_link(state, parent=b, child=tid_int)
            for b in remove_blocks:
                _apply_link(state, parent=tid_int, child=b)
            for b in remove_blocked_by:
                _apply_link(state, parent=b, child=tid_int)
            return _err(
                f"refusing status=in_progress: task #{tid_int} is "
                f"blocked by {sorted(pending)} (not yet completed). "
                "Finish or detach those tasks first."
            )
        task["status"] = raw_status
    elif raw_status is not None:
        task["status"] = raw_status

    for fld in ("subject", "description", "activeForm", "owner"):
        if (v := args.get(fld)) is not None:
            if not isinstance(v, str):
                return _err(f"`{fld}` must be a string")
            task[fld] = v.strip() if fld != "description" else v

    if (md := args.get("metadata")) is not None:
        if not isinstance(md, dict):
            return _err("`metadata` must be an object")
        merged = dict(task.get("metadata") or {})
        for k, v in md.items():
            if v is None:
                merged.pop(k, None)
            else:
                merged[k] = v
        task["metadata"] = merged

    try:
        _save_tasks(path, state)
    except OSError as e:
        return _err(f"failed to write {path}: {e}")

    # Soft notification: if this update just completed a task, list
    # the dependents that are now unblocked. Lets a small model spot
    # what to claim next without an extra TaskList round-trip.
    unblocked_msg = ""
    if raw_status == "completed":
        newly_ready: list[int] = []
        for child_id in task.get("blocks") or []:
            try:
                cid = int(child_id)
            except (TypeError, ValueError):
                continue
            child_row = state["tasks"].get(str(cid))
            if not isinstance(child_row, dict):
                continue
            if child_row.get("status") != "pending":
                continue
            if not _pending_blockers(state, cid):
                newly_ready.append(cid)
        if newly_ready:
            unblocked_msg = f" — now unblocked: #{','.join(str(x) for x in sorted(newly_ready))}"

    return ToolResult(
        content=f"Updated task #{tid}{unblocked_msg}",
        ok=True,
    )


def run_task_list(args: dict[str, Any]) -> ToolResult:
    """List all tasks in id order. Each row shows
    ``#id [status] subject [owner=X] [blockedBy=A,B]``.
    """
    path = _tasks_path_from_env()
    if path is None:
        return _err("session task file unresolved — SHAREDAGENTS_SESSION_ID / SHAREDAGENTS_SESSIONS_DIR not set in env")
    state = _load_tasks(path)
    tasks = state.get("tasks") or {}
    if not tasks:
        return ToolResult(content="(no tasks)", ok=True)
    ordered = sorted(
        tasks.values(),
        key=lambda t: int(t.get("id") or 0),
    )
    body = "\n".join(_task_summary(t) for t in ordered)
    return ToolResult(content=body, ok=True)


def run_task_get(args: dict[str, Any]) -> ToolResult:
    """Fetch a single task's full details for the model."""
    tid_raw = (
        args.get("taskId")
        if args.get("taskId") is not None
        else args.get("task_id")
        if args.get("task_id") is not None
        else args.get("id")
    )
    if not isinstance(tid_raw, (str, int)):
        return _err("`taskId` required (synonyms task_id / id also accepted)")
    tid = str(tid_raw)
    path = _tasks_path_from_env()
    if path is None:
        return _err("session task file unresolved — SHAREDAGENTS_SESSION_ID / SHAREDAGENTS_SESSIONS_DIR not set in env")
    state = _load_tasks(path)
    task = state["tasks"].get(tid)
    if task is None:
        return _err(f"unknown taskId: {tid}")
    return ToolResult(
        content=json.dumps(task, indent=2, ensure_ascii=False),
        ok=True,
    )


def _output_sections_path() -> Path | None:
    sid = os.environ.get("SHAREDAGENTS_SESSION_ID")
    sdir = os.environ.get("SHAREDAGENTS_SESSIONS_DIR")
    if not sid or not sdir:
        return None
    return Path(sdir) / f"{sid}.output_sections.json"


# ---- dispatcher ----

# Sync tools dispatch to a thread; async tools run on the event loop.


def run_meeting_search(args: dict[str, Any]) -> ToolResult:
    """Explicitly retrieve passive meeting chunks for a later human prompt.

    This tool is never auto-invoked or injected into context. It reuses the
    memory privacy origin decision because meeting transcripts are at least as
    private as ordinary recalled memory.
    """
    from sharedagents_realtime.meeting import MeetingTranscriptStore

    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return _err("query required")
    limit = args.get("limit", 6)
    try:
        hits = MeetingTranscriptStore().search(
            query.strip(), limit=int(limit) if isinstance(limit, (int, float)) else 6
        )
    except (OSError, ValueError) as exc:
        return _err(f"meeting_search failed: {exc}")
    if not hits:
        return ToolResult(content="No matching meeting transcript chunks.", ok=True)
    body = json.dumps({"results": hits}, ensure_ascii=False, indent=2)
    body, truncated = _truncate(body)
    return ToolResult(content=body, ok=True, truncated=truncated)


async def run_capture_delegation_pattern_async(args: dict[str, Any]) -> ToolResult:
    """Persist a successful delegation prompt as an orchestrator skill (deduped)."""
    import json as _json
    import os

    from sharedagents_agent.delegation_patterns import upsert_delegation_pattern

    agent_home = os.environ.get("SHAREDAGENTS_AGENT_HOME") or ""
    if not agent_home:
        return _err("capture_delegation_pattern: SHAREDAGENTS_AGENT_HOME not set")
    subagent_type = str(args.get("subagent_type") or "").strip()
    title = str(args.get("title") or "").strip()
    prompt_template = str(args.get("prompt_template") or args.get("prompt") or "").strip()
    tags = args.get("tags")
    tag_list: list[str] | None = None
    if isinstance(tags, list):
        tag_list = [str(t) for t in tags if t]
    result = upsert_delegation_pattern(
        agent_home,
        subagent_type=subagent_type,
        title=title,
        prompt_template=prompt_template,
        tags=tag_list,
    )
    if not result.get("ok"):
        return ToolResult(content=str(result.get("reason") or "failed"), ok=False)
    return ToolResult(content=_json.dumps(result, ensure_ascii=False), ok=True)


# Lightweight stand-in for AgentConfig — only the fields `remember`
# reads. Using a real AgentConfig here would force a full disk read
# of the global + per-agent config inside every tool runner.


def run_skill_get(args: dict[str, Any]) -> ToolResult:
    # Claude Code's ``Skill`` uses ``skill`` (and optional ``args``) —
    # accept both that and the legacy ``name`` / no-args shape.
    name = args.get("skill") or args.get("name")
    extra_args = args.get("args")
    if not isinstance(name, str) or not name:
        return _err("skill name required")
    # A Web user's session loads only the skills granted to them. This runs
    # in the tool-runner subprocess, which knows its session from the env
    import os as _os

    _sessions_dir = _os.environ.get("SHAREDAGENTS_SESSIONS_DIR", "")
    _sid = _os.environ.get("SHAREDAGENTS_SESSION_ID", "")
    _grants = None
    if _sessions_dir and _sid:
        from sharedagents_session.access_binding import session_access_grants

        _grants = session_access_grants(_sessions_dir, _sid)
        if _grants is not None and not _grants.allows("skills", name):
            return _err(f"skill {name!r} is not granted to this user")
    from sharedagents_skill import skills as _skills_mod

    _sk_allow, _sk_strict = _skills_mod.get_active_skill_allowlist()
    skill = _skills_mod.find_skill(name, allowlist=_sk_allow, allowlist_strict=_sk_strict)
    if skill is None:
        return _err(f"skill not found: {name!r}. Run `sharedagents skills list` to see what's installed.")
    body = skill.body or "(SKILL.md has frontmatter only — no body)"
    header = f"# {skill.name}\n_{skill.short_description}_\n\n_(source: {skill.source})_\n\n---\n\n"
    if isinstance(extra_args, str) and extra_args.strip():
        header += f"_(invoked with args: {extra_args.strip()})_\n\n"
    full = header + body
    # The live appendices are operator actions (WhatsApp send / allowlist /
    # device link, the provider credits snapshot): a Web user's turn gets the
    # skill text only.
    parts = (
        []
        if _grants is not None
        else [
            block
            for block in (
                _provider_route_appendix(skill, extra_args),
                _whatsapp_appendix(skill, extra_args),
            )
            if block
        ]
    )
    if parts:
        full = full + "\n\n" + "\n\n".join(parts)
    full, truncated = _truncate(full)
    return ToolResult(content=full, ok=True, truncated=truncated)


def _provider_route_appendix(skill: Any, extra_args: Any) -> str:
    """Attach the live credits snapshot when Jarvis loads provider-route."""
    meta = skill.metadata if isinstance(getattr(skill, "metadata", None), dict) else {}
    sa_meta = meta.get("sharedagents") if isinstance(meta.get("sharedagents"), dict) else {}
    if getattr(skill, "name", "") != "provider-route" and sa_meta.get("snapshot") != "provider-route":
        return ""
    refresh = False
    if isinstance(extra_args, str):
        refresh = extra_args.strip().lower() in {"refresh", "refresh=1", "1", "--refresh"}
    try:
        from sharedagents_web.subscription_limits import routing_digest, routing_digest_text

        text = routing_digest_text(routing_digest(refresh=refresh))
    except Exception as exc:  # noqa: BLE001
        return f"## Live snapshot\nUnavailable: {exc}"
    return "## Live snapshot\n" + text + "\n"


def _whatsapp_appendix(skill: Any, extra_args: Any) -> str:
    """Attach a live QR (or link status) when the on-demand WhatsApp skill loads."""
    meta = skill.metadata if isinstance(getattr(skill, "metadata", None), dict) else {}
    sa_meta = meta.get("sharedagents") if isinstance(meta.get("sharedagents"), dict) else {}
    if getattr(skill, "name", "") != "whatsapp" and sa_meta.get("snapshot") != "whatsapp":
        return ""
    raw = extra_args.strip() if isinstance(extra_args, str) else ""
    args = raw.lower()
    try:
        from sharedagents_whatsapp.allowlist_skill import run_allowlist_args
        from sharedagents_whatsapp.outbound import (
            account_snapshot,
            enqueue_send,
            parse_send_request,
        )
        from sharedagents_whatsapp.qr import render_snapshot, show_qr

        if args.startswith("send"):
            from sharedagents_whatsapp.config import WhatsAppConfig

            request = parse_send_request(raw, cfg=WhatsAppConfig.load())
            return enqueue_send(request.text, dest=request.dest, files=request.files)
        allow = run_allowlist_args(raw)
        if allow:
            return allow
        if args in {"status", "list", "users", "contacts", "account"}:
            return account_snapshot()
        if args in {"", "qr", "link", "show"}:
            return render_snapshot(show_qr())
        return account_snapshot()
    except Exception as exc:  # noqa: BLE001
        return f"## WhatsApp\nUnavailable: {exc}\n"


__all__ = [
    "_TODO_VALID_STATUSES",
    "_canonical_todo_status",
    "todos_path_for_session",
    "_todos_path_from_env",
    "load_todos",
    "_save_todos",
    "format_todo_json",
    "format_todo_md",
    "_normalize_todo_items",
    "run_todo",
    "_TASK_VALID_STATUSES",
    "_tasks_path_from_env",
    "_load_tasks",
    "_save_tasks",
    "_task_summary",
    "_normalize_task_id",
    "_parse_id_list",
    "_would_cycle",
    "_apply_link",
    "_remove_link",
    "_graph_has_cycle",
    "_pending_blockers",
    "run_task_create",
    "run_task_update",
    "run_task_list",
    "run_task_get",
    "_output_sections_path",
    "run_meeting_search",
    "run_capture_delegation_pattern_async",
    "run_skill_get",
]
