"""Widget adaptation dictionary for every external engine.

Each frontend (Web UI, TUI, Telegram) must ask this module what a foreign
tool *is*, not hard-code Cursor / Claude / OpenCode names. Add a new
engine by appending aliases here; do not fork matchers in the UIs.

Canonical kinds the three frontends already know how to paint:

- ``todo``     — sharedagents ``todowrite`` / ``/todo`` / 📝 checklist
- ``task``     — harness TaskCreate/Update/Stop / 🗒️
- ``plan``     — ExitPlanMode
- ``question`` — AskUserQuestion / question / any engine's spelling of it.
  Call :func:`is_ask_user_tool`; never hardcode the names in a frontend.
"""

from __future__ import annotations

from typing import Any

# alias key = alnum-lowercase of the wire title / tool id.
# value = canonical widget kind.
WIDGET_ALIASES: dict[str, str] = {
    # sharedagents native + OpenCode ACP
    "todowrite": "todo",
    "todo": "todo",
    "todos": "todo",
    # Cursor ACP
    "updatetodos": "todo",
    # Codex exec --json (todo_list item) + update_plan tool
    "todolist": "todo",
    "updateplan": "todo",
    # Claude Code CLI (in-process)
    # "todowrite" already covers TodoWrite after normalize
    # Harness tasks
    "taskcreate": "task",
    "taskupdate": "task",
    "taskstop": "task",
    "tasklist": "task",
    "taskget": "task",
    # Plan card (native ExitPlanMode + Cursor ACP Create Plan)
    "exitplanmode": "plan",
    "createplan": "plan",
    "create_plan": "plan",
    # Ask-user
    "question": "question",
    "askuserquestion": "question",
    "userquestion": "question",
    "askuser": "question",
}

# Human labels we have actually seen on the wire, grouped by engine.
# Documentation + drift tests — not a second matcher.
ENGINE_WIDGET_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "sharedagents": {
        "todo": ("todowrite", "TodoWrite", "todo"),
        "task": ("TaskCreate", "TaskUpdate", "TaskStop"),
        "plan": ("ExitPlanMode", "CreatePlan", "Create Plan"),
        "question": ("question", "AskUserQuestion"),
    },
    "cursor": {
        "todo": ("Update TODOs", "updateTodos"),
        "plan": ("Create Plan", "CreatePlan"),
    },
    "claude": {
        "todo": ("TodoWrite",),
        "task": ("TaskCreate", "TaskUpdate", "TaskStop"),
        "question": ("AskUserQuestion",),
    },
    "opencode": {
        "todo": ("todowrite",),
    },
    "antigravity": {
        "todo": ("TodoWrite",),
        "plan": ("exit_plan_mode",),
    },
    "codex": {
        "todo": ("todo_list", "update_plan"),
    },
    "hermes": {
        "todo": ("todo",),
    },
}


def norm_tool_key(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def widget_kind(name: str, *, acp_kind: str = "") -> str:
    """Return ``todo`` / ``task`` / ``plan`` / ``question`` / ````."""
    kind = WIDGET_ALIASES.get(norm_tool_key(name), "")
    if kind:
        return kind
    if norm_tool_key(acp_kind) == "todo":
        return "todo"
    return ""


def is_todo_tool_name(name: str, *, acp_kind: str = "") -> bool:
    return widget_kind(name, acp_kind=acp_kind) == "todo"


#: Canonical widget kind for the ask-user family.
ASK_WIDGET_KIND = "question"

#: Wire spellings seen for the ask family. Display / telemetry / compatibility
#: views only — MATCHING goes through :func:`is_ask_user_tool`, which also
#: accepts any normalized spelling (``askuser``, ``userQuestion``, …).
ASK_TOOL_WIRE_NAMES: tuple[str, ...] = (
    "question",
    "AskUserQuestion",
    "ask_user",
    "UserQuestion",
)


def is_ask_user_tool(name: str) -> bool:
    """True for ANY engine's spelling of the ask-user tool.

    The ask family is the one widget every agent can fire (``question``
    natively, ``AskUserQuestion`` from Claude Code, an ACP client's
    ``session/ask_user`` round-trip, …), and every frontend used to carry its
    own exact-match 4-name list. Those lists missed the normalized spellings
    this module already declares (``askuser``, ``userquestion``) as well as any
    future one (``ask_user_question``) — and a miss is not cosmetic: the
    transcript loses the question card *and the card's place in the turn*
    (see the placement contract in ``ask_user.js``), the history fold stops
    splitting the turn at the ask, and a parked question loses its
    no-idle-kill guard.

    Add a new engine by appending to :data:`WIDGET_ALIASES` — never by forking
    a matcher per frontend, per agent, or per engine.
    """
    return widget_kind(name) == ASK_WIDGET_KIND


def ask_tool_aliases() -> frozenset[str]:
    """Alias keys that resolve to the ask widget (docs / drift tests)."""
    return frozenset(key for key, kind in WIDGET_ALIASES.items() if kind == ASK_WIDGET_KIND)


def is_task_tool_name(name: str) -> bool:
    return widget_kind(name) == "task"


def is_plan_tool_name(name: str) -> bool:
    return widget_kind(name) == "plan"


def extract_plan_markdown(args: Any) -> str:
    """Pull the operator-facing plan body off CreatePlan / ExitPlanMode args."""
    obj = _as_args_dict(args)
    plan = obj.get("plan")
    if isinstance(plan, str) and plan.strip():
        return plan.strip()
    parts: list[str] = []
    name = obj.get("name")
    if isinstance(name, str) and name.strip():
        parts.append("# " + name.strip())
    overview = obj.get("overview")
    if isinstance(overview, str) and overview.strip():
        parts.append(overview.strip())
    return "\n\n".join(parts).strip()


def recover_cursor_plan_markdown(
    *,
    session_hint: str = "",
    after_mtime: float | None = None,
) -> str:
    """Read the newest Cursor ``*.plan.md`` when ACP sent an empty Create Plan.

    Cursor writes the artifact under ``~/.cursor/plans/`` (or
    ``$CURSOR_PLANS_DIR``) and does not put the markdown on the ACP
    ``rawInput`` / result. Strip YAML frontmatter so the 📐 card shows
    the body the operator approves.
    """
    import os
    from pathlib import Path

    roots: list[Path] = []
    env = (os.environ.get("CURSOR_PLANS_DIR") or os.environ.get("SHAREDAGENTS_CURSOR_PLANS_DIR") or "").strip()
    if env:
        roots.append(Path(env).expanduser())
    roots.append(Path.home() / ".cursor" / "plans")
    hint = "".join(ch for ch in (session_hint or "") if ch.isalnum())[:8].lower()
    ranked: list[tuple[float, Path]] = []
    for root in roots:
        try:
            paths = list(root.glob("*.plan.md"))
        except OSError:
            continue
        for path in paths:
            try:
                st = path.stat()
            except OSError:
                continue
            if after_mtime is not None and st.st_mtime < (after_mtime - 5):
                continue
            score = float(st.st_mtime)
            if hint and hint in path.name.lower():
                score += 1e12
            ranked.append((score, path))
    if not ranked:
        return ""
    ranked.sort(key=lambda row: row[0], reverse=True)
    try:
        text = ranked[0][1].read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    stripped = text.lstrip()
    if stripped.startswith("---"):
        rest = stripped[3:]
        end = rest.find("\n---")
        if end >= 0:
            text = rest[end + 4 :].lstrip("\n")
    return text.strip()


def _as_args_dict(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return args
    if isinstance(args, str) and args.strip():
        import json

        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _canonical_status(status: Any) -> str:
    if not isinstance(status, str):
        return "pending"
    s = status.strip().lower()
    if s in {"completed", "done"}:
        return "completed"
    if s in {"pending", "in_progress", "cancelled"}:
        return s
    return "pending"


def adapt_todo_item(raw: Any) -> dict[str, str] | None:
    """Cursor / Claude / OpenCode item → sharedagents ``{content, status, activeForm}``."""
    if isinstance(raw, str):
        content = raw.strip()
        if not content:
            return None
        return {"content": content, "status": "pending", "activeForm": "", "id": ""}
    if not isinstance(raw, dict):
        return None
    content = raw.get("content") or raw.get("text") or raw.get("body") or raw.get("step") or ""
    if not isinstance(content, str) or not content.strip():
        return None
    active = raw.get("activeForm") or raw.get("active_form") or ""
    if not isinstance(active, str):
        active = ""
    item_id = raw.get("id") or raw.get("todoId") or raw.get("todo_id") or ""
    status = raw.get("status") or raw.get("state")
    if status is None and isinstance(raw.get("completed"), bool):
        status = "completed" if raw.get("completed") else "pending"
    return {
        "id": str(item_id),
        "content": content.strip(),
        "status": _canonical_status(status),
        "activeForm": active.strip(),
    }


def adapt_todo_args(args: Any) -> dict[str, Any] | None:
    """Foreign todo payload → ``{"todos": [canonical items]}`` or ``None``."""
    obj = _as_args_dict(args)
    raw_list = obj.get("todos")
    if not isinstance(raw_list, list):
        raw_list = obj.get("items")
    if not isinstance(raw_list, list):
        raw_list = obj.get("plan")
    if not isinstance(raw_list, list):
        return None
    items = [it for it in (adapt_todo_item(x) for x in raw_list) if it]
    if not items:
        return None
    out: dict[str, Any] = {"todos": items}
    if obj.get("merge") is not None:
        out["merge"] = bool(obj.get("merge"))
    return out


def adapt_tool_call(
    name: str,
    args: Any = None,
    *,
    acp_kind: str = "",
) -> dict[str, Any]:
    """One wire tool call → ``{kind, name, args}`` with canonical todo args."""
    kind = widget_kind(name, acp_kind=acp_kind)
    adapted_args = _as_args_dict(args)
    if kind == "todo":
        todos = adapt_todo_args(adapted_args)
        if todos:
            adapted_args = todos
    return {"kind": kind, "name": name, "args": adapted_args}


def aliases_for_js() -> dict[str, str]:
    """Exact object the Web UI ``widget_adapt.js`` must keep in sync."""
    return dict(WIDGET_ALIASES)


def format_todo_lines(
    items: Any,
    *,
    heading: str = "todos:",
    limit: int = 50,
) -> str:
    """Canonical checklist text for Telegram / ``/todo`` / TUI fallback."""
    glyphs = {
        "completed": "[x]",
        "done": "[x]",
        "in_progress": "[~]",
        "pending": "[ ]",
        "cancelled": "[-]",
    }
    lines = [heading] if heading else []
    count = 0
    for item in items or []:
        if count >= limit:
            break
        row = adapt_todo_item(item)
        if row is None:
            continue
        count += 1
        lines.append(f"  {count}. {glyphs.get(row['status'], '[ ]')} {row['content']}")
    return "\n".join(lines)
