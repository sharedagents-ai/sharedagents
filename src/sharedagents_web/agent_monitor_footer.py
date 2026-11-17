"""Session-scoped Agent Monitor footer rows.

The chat footer only needs a flat lineage (roots → descendants), not the
full /agents tree. Agent activation history comes from the selected session's
JSONL-derived ``footer_state.sequence``.

**Every row is shipped.** There is no row budget here: the composer dock bounds
itself with a CSS scrollport (``.ama-body`` max-height + ``overflow-y: scroll``),
so the operator can scroll to any row. Truncating server-side instead hides
rows that no amount of scrolling can recover — a session whose lineage holds a
parent, a switch history AND several delegated children must paint all of them.

**Counts come from here.** Every payload carries ``counts`` =
``{active, total, agents}`` (:func:`footer_counts`); the page renders it and
never counts rows itself. ``active`` uses :func:`row_active`, the same rule the
page applies to a row's dot and Stop button.
"""

from __future__ import annotations

import glob
import json
import re
import time
from typing import Any

_SID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
_HIST_TTL_S = 30.0
_HIST_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _hkey(value: Any) -> str:
    return str(value or "").strip()[:24]


def _busy_states() -> frozenset[str]:
    from sharedagents_session.session_activity import TREE_BUSY_STATES

    # ``running`` is the flatten fallback for a live node without a state.
    return frozenset(TREE_BUSY_STATES) | {"running"}


FOOTER_BUSY_STATES: frozenset[str] = _busy_states()


def row_active(row: dict[str, Any] | None) -> bool:
    """One liveness rule for a footer row: its dot, Stop button and count.

    A busy state alone is not enough — a closed, abandoned turn can keep a
    legacy ``tool_running``. The row must also carry the backend ``live`` bit
    (open turn / fresh heartbeat, or promoted by the live child snapshot in
    :func:`reconcile_sub_liveness`).
    """
    if not isinstance(row, dict):
        return False
    return bool(row.get("live")) and str(row.get("state") or "") in FOOTER_BUSY_STATES


def footer_counts(footer: dict[str, Any] | None) -> dict[str, int]:
    """Header numbers for the footer label and the card header.

    ``total`` is every delegated subagent row (any depth, plus durable
    sidecar children), ``active`` the ones :func:`row_active` treats as
    running/queued/waiting, ``agents`` the lanes the operator is in (the
    current session and its ancestors).
    """
    base = footer or {}
    rows = [row for row in (base.get("rows") or []) if isinstance(row, dict)]
    subs = [row for row in rows if row.get("role") == "sub"]
    try:
        declared = int(base.get("subs") or 0)
    except (TypeError, ValueError):
        declared = 0
    return {
        "active": sum(1 for row in subs if row_active(row)),
        "total": max(declared, len(subs)),
        "agents": sum(1 for row in rows if row.get("role") in ("current", "parent")),
    }


def with_counts(footer: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``footer`` with fresh ``counts`` (the only count source)."""
    out = dict(footer or {})
    out["counts"] = footer_counts(out)
    return out


def _viewed_sub_keys(rows: list[dict[str, Any]]) -> set[str]:
    """Descendants of the current lane, excluding siblings in a child view."""
    current = next((row for row in rows if row.get("role") == "current"), None)
    viewed = _hkey((current or {}).get("hash"))
    if not viewed:
        return set()
    parents = {_hkey(row.get("hash")): _hkey(row.get("parent_hash")) for row in rows if row.get("hash")}
    descendants: set[str] = set()
    for row in rows:
        if row.get("role") != "sub":
            continue
        child = key = _hkey(row.get("hash"))
        seen: set[str] = set()
        while key and key not in seen and len(seen) < 128:
            if key == viewed:
                descendants.add(child)
                break
            seen.add(key)
            key = parents.get(key, "")
    return descendants


def descendants_running(footer: dict[str, Any] | None) -> bool:
    """Composer liveness from the reconciled footer's dot/count rule."""
    rows = [row for row in ((footer or {}).get("rows") or []) if isinstance(row, dict)]
    keys = _viewed_sub_keys(rows)
    return any(row_active(row) and _hkey(row.get("hash")) in keys for row in rows)


def reconcile_sub_liveness(
    footer: dict[str, Any] | None,
    live: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fold the viewed session's live activity snapshot into the sub rows.

    The tree's ``live`` bit is freshness-gated and can lag a tick behind the
    activity snapshot, so:

    * a child the snapshot still lists with a busy state is ACTIVE even if its
      tree ``live`` bit dropped (long silent tool call);
    * inside the viewed session's subtree, every busy-looking row is folded to
      idle when the viewed session itself is idle (nothing below an idle
      session can still be running), and a DIRECT child the snapshot no
      longer lists is folded to idle. Deeper descendants are not in the
      snapshot (it lists direct children only), so they keep the tree bit;
    * rows outside the viewed subtree (siblings in child view) keep the tree
      values — the tree is authoritative for them;
    * an ``error`` row is never live.
    """
    base = dict(footer or {})
    if not isinstance(live, dict):
        return with_counts(base)
    rows = [dict(row) for row in (base.get("rows") or []) if isinstance(row, dict)]
    session_state = str(live.get("session_state") or live.get("state") or "")
    session_busy = bool(live.get("own_live")) or session_state in FOOTER_BUSY_STATES
    listed: set[str] = set()
    for kid in live.get("subagents") or []:
        if isinstance(kid, dict):
            key = _hkey(kid.get("hash") or kid.get("session_id") or kid.get("id"))
            if key:
                listed.add(key)
    current = next((row for row in rows if row.get("role") == "current"), None)
    viewed = _hkey((current or {}).get("hash"))
    descendants = _viewed_sub_keys(rows)

    for row in rows:
        if row.get("role") != "sub":
            continue
        state = str(row.get("state") or "")
        if state == "error":
            row["live"] = False
            continue
        busy_state = state in FOOTER_BUSY_STATES
        key = _hkey(row.get("hash"))
        is_listed = bool(key) and key in listed
        if is_listed and busy_state:
            row["live"] = True
            continue
        if key not in descendants:
            continue
        direct = bool(viewed) and _hkey(row.get("parent_hash")) == viewed
        if not session_busy or (listed and direct and not is_listed):
            row["live"] = False
            if busy_state:
                row["state"] = "idle"
    base["rows"] = rows
    return with_counts(base)


def resolve_current_hash(
    tree: dict[str, Any] | None,
    viewed_hash: str = "",
    nodes: dict[str, dict[str, Any]] | None = None,
    parent_of: dict[str, dict[str, Any] | None] | None = None,
) -> str:
    """Pick the viewed session node as the footer's current row."""
    viewed = str(viewed_hash or "").strip()
    viewed_key = _hkey(viewed)
    if not viewed:
        return ""
    if nodes is None:
        return viewed
    match = nodes.get(viewed)
    if match is None and viewed_key:
        match = next(
            (node for key, node in nodes.items() if _hkey(key) == viewed_key),
            None,
        )
    if match is None:
        return viewed
    match_key = str(match.get("hash") or "").strip()
    return match_key


def unbound_footer_payload(
    agent: str = "",
    title: str = "",
    model: str = "",
) -> dict[str, Any]:
    """Synthetic footer for a New Chat row that has no gateway tree yet.

    ``model`` is the web store's picker model for the unbound chat (the model
    the conversation will run once the gateway binds it). It is optional: when
    it is genuinely unknown the row renders without a model label, exactly like
    a bound row whose session never resolved an engine model.
    """
    name = str(agent or "").strip() or "jarvis"
    return with_counts(
        {
            "session": "",
            "rows": [
                {
                    "hash": "",
                    "agent": name,
                    "title": str(title or ""),
                    "model": str(model or ""),
                    "state": "idle",
                    "live": False,
                    "depth": 0,
                    "role": "current",
                    "last_activity_at": 0,
                    "parent_hash": "",
                    "subagent_count": 0,
                }
            ],
            "subs": 0,
        }
    )


def flatten_footer_rows(
    tree: dict[str, Any] | None,
    viewed_hash: str = "",
) -> list[dict[str, Any]]:
    """Flatten a session-scoped DAG into footer rows."""
    if not isinstance(tree, dict):
        return []
    viewed = str(viewed_hash or "").strip()

    nodes: dict[str, dict[str, Any]] = {}
    parent_of: dict[str, dict[str, Any] | None] = {}

    def walk(node: dict[str, Any], parent: dict[str, Any] | None) -> None:
        if not isinstance(node, dict):
            return
        key = str(node.get("hash") or "").strip()
        if key and key not in nodes:
            nodes[key] = node
            parent_of[key] = parent
        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child, node)

    for bucket in tree.get("agents") or []:
        if not isinstance(bucket, dict):
            continue
        for root in bucket.get("roots") or []:
            if isinstance(root, dict):
                walk(root, None)

    if not nodes:
        return []

    roots = [node for key, node in nodes.items() if parent_of.get(key) is None]
    current_hash = resolve_current_hash(tree, viewed, nodes, parent_of)
    current_key = _hkey(current_hash)
    ancestors: set[str] = set()
    current = nodes.get(current_hash)
    if current is None and current_key:
        current = next(
            (node for key, node in nodes.items() if _hkey(key) == current_key),
            None,
        )
    guard = 0
    while current is not None and guard < 128:
        parent = parent_of.get(str(current.get("hash") or "").strip())
        if not parent:
            break
        ancestors.add(str(parent.get("hash") or "").strip())
        current = parent
        guard += 1

    def role_for(node: dict[str, Any], depth: int) -> str:
        key = str(node.get("hash") or "").strip()
        if current_key and (_hkey(key) == current_key):
            return "current"
        # ANY ancestor of the viewed session is a parent lane — not just the
        # depth-0 root. When the selected session is a nested SUBAGENT, the
        # chain above it is its ancestry: labelling it ``sub`` mis-read the
        # operator's own parent as a delegated child and counted it in
        # ``subs``. Siblings and their descendants stay ``sub``.
        if key in ancestors:
            return "parent"
        if depth == 0:
            return "previous"
        return "sub"

    def _to_ms(val: Any) -> int:
        if not val:
            return 0
        if isinstance(val, (int, float)):
            return int(val if val > 1_000_000_000_000 else val * 1000)
        try:
            from datetime import datetime, timezone

            s = str(val).strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            try:
                v = float(val)
                return int(v if v > 1_000_000_000_000 else v * 1000)
            except Exception:
                return 0

    def _hash_ms(value: Any) -> int:
        stamp = str(value or "").strip().split("-", 1)[0]
        if len(stamp) < 15 or stamp[8] != "T":
            return 0
        try:
            from datetime import datetime, timezone

            dt = datetime.strptime(stamp[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            micros = int((stamp[15:21] or "0").ljust(6, "0")[:6])
            return int(dt.timestamp() * 1000) + micros // 1000
        except Exception:
            return 0

    def creation_key(node: dict[str, Any]) -> tuple[int, str]:
        # Delegation / spawn time only — never last_turn_ts / last_activity_at.
        ms = _to_ms(node.get("created_at")) or _to_ms(node.get("started_at")) or _hash_ms(node.get("hash")) or 0
        return (ms, str(node.get("hash") or ""))

    rows: list[dict[str, Any]] = []

    def emit(node: dict[str, Any], depth: int) -> None:
        key = str(node.get("hash") or "").strip()
        parent = parent_of.get(key)
        footer_state = node.get("footer_state") or {}
        sequence = footer_state.get("sequence") or []
        current = footer_state.get("current") or {}

        kids = [child for child in (node.get("children") or []) if isinstance(child, dict)]
        # Newest delegated child first (operator reads the latest spawn at the top).
        kids.sort(key=creation_key, reverse=True)

        activations: list[dict[str, Any]] = []
        if depth == 0 and isinstance(sequence, list) and len(sequence) > 1:
            for idx, activation in enumerate(sequence):
                if not isinstance(activation, dict):
                    continue
                activations.append(
                    {
                        "index": idx,
                        "agent": activation.get("agent") or "?",
                        "model": activation.get("model") or "",
                        "timestamp": activation.get("timestamp") or 0,
                        "time_ms": _to_ms(activation.get("timestamp")),
                        "is_current": (idx == len(sequence) - 1),
                    }
                )
            # Model resolution is not an agent switch. Consecutive rows for
            # the same agent (cursor/auto then cursor/qwen) are one lane.
            collapsed: list[dict[str, Any]] = []
            for act in activations:
                if collapsed and collapsed[-1]["agent"] == act["agent"]:
                    prev = collapsed[-1]
                    if act["model"]:
                        prev["model"] = act["model"]
                    prev["is_current"] = act["is_current"]
                    prev["index"] = act["index"]
                    prev["timestamp"] = act["timestamp"] or prev["timestamp"]
                    prev["time_ms"] = act["time_ms"] or prev["time_ms"]
                else:
                    collapsed.append(dict(act))
            activations = collapsed
            for i, act in enumerate(activations):
                act["index"] = i

        if not activations:
            rows.append(
                {
                    "hash": key,
                    "agent": current.get("agent") or node.get("agent") or "?",
                    "model": current.get("model") or node.get("model") or "",
                    "title": node.get("title") or "",
                    "state": node.get("state") or ("running" if node.get("live") else "idle"),
                    "live": bool(node.get("live")),
                    "depth": int(depth),
                    "role": role_for(node, depth),
                    "last_activity_at": (
                        node.get("last_activity_at") or node.get("last_turn_ts") or node.get("state_updated_at") or 0
                    ),
                    "created_at": node.get("created_at") or "",
                    "parent_hash": str((parent or {}).get("hash") or ""),
                    "subagent_count": int(node.get("subagent_count") or 0),
                }
            )
            for child in kids:
                emit(child, depth + 1)
            return

        act_kids: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(activations))}
        for child in kids:
            child_ms = creation_key(child)[0]
            assigned_idx = 0
            for i, act in enumerate(activations):
                if act["time_ms"] and child_ms >= act["time_ms"]:
                    assigned_idx = i
                elif not act["time_ms"] and i == 0:
                    assigned_idx = 0
            act_kids[assigned_idx].append(child)

        for act in reversed(activations):
            idx = act["index"]
            is_cur = act["is_current"]
            role = role_for(node, depth) if is_cur else "previous"
            rows.append(
                {
                    "hash": key if is_cur else "",
                    "agent": act["agent"],
                    # Current lane: selected/picker model on the session node
                    # wins over a resolved engine id stamped on the activation
                    # (Auto vs qwen-3.8). Historical rows keep their record.
                    "model": (
                        (node.get("model") if is_cur else "")
                        or (current.get("model") if is_cur else "")
                        or act["model"]
                        or (node.get("engine_model") if is_cur else "")
                        or ""
                    ),
                    "title": node.get("title") or "",
                    "state": (node.get("state") or ("running" if node.get("live") else "idle"))
                    if is_cur
                    else "switched",
                    "live": bool(node.get("live")) if is_cur else False,
                    "depth": int(depth),
                    "role": role,
                    "last_activity_at": (
                        node.get("last_activity_at") or node.get("last_turn_ts") or node.get("state_updated_at") or 0
                    )
                    if is_cur
                    else (act.get("timestamp") or 0),
                    "created_at": (node.get("created_at") or "") if is_cur else (act.get("timestamp") or ""),
                    "parent_hash": str((parent or {}).get("hash") or "") if is_cur else "",
                    "subagent_count": len(act_kids[idx]),
                }
            )
            for child in act_kids[idx]:
                emit(child, depth + 1)

    for root in roots:
        emit(root, 0)
    return rows


def footer_payload(
    tree: dict[str, Any] | None,
    viewed_hash: str = "",
) -> dict[str, Any]:
    """Compact footer JSON: every lineage row, the sub count and ``counts``.

    No row budget — the dock scrolls. ``flatten_footer_rows`` already emits the
    whole viewed lineage (ancestors, the current lane, its switch history and
    every delegated child at any depth), so the payload is the flatten verbatim.
    """
    flat = flatten_footer_rows(tree, viewed_hash)
    return with_counts(
        {
            "session": str(viewed_hash or ""),
            "rows": flat,
            "subs": sum(1 for row in flat if row.get("role") == "sub"),
        }
    )


def stamp_viewed_current(
    footer: dict[str, Any] | None,
    *,
    agent: str,
    title: str = "",
    model: str = "",
    sid: str = "",
    state: str = "",
    live: bool | None = None,
) -> dict[str, Any]:
    """Force the painted ``current`` lane onto the open-session chrome.

    Picker, header, and Agent Monitor current row share one agent. Tree
    flatten / a leftover parent ``footer_state`` must not invent another.
    """
    name = str(agent or "").strip() or "jarvis"
    base = dict(footer or {})
    rows = [dict(row) for row in (base.get("rows") or [])]
    current = next((row for row in rows if row.get("role") == "current"), None)
    if current is None:
        current = {
            "hash": str(sid or ""),
            "agent": name,
            "title": str(title or ""),
            "model": str(model or ""),
            "state": str(state or "idle"),
            "live": bool(live) if live is not None else False,
            "depth": 0,
            "role": "current",
            "last_activity_at": 0,
            "parent_hash": "",
            "subagent_count": 0,
        }
        rows.insert(0, current)
    else:
        current["agent"] = name
        current["role"] = "current"
        if title:
            current["title"] = str(title)
        if model:
            current["model"] = str(model)
        if sid and not current.get("hash"):
            current["hash"] = str(sid)
        if state:
            current["state"] = str(state)
        if live is not None:
            current["live"] = bool(live)
    for row in rows:
        if row is not current and row.get("role") == "current":
            row["role"] = "previous"
    out = dict(base)
    if sid:
        out["session"] = str(sid)
    elif "session" not in out:
        out["session"] = ""
    out["rows"] = rows
    out["subs"] = int(out.get("subs") or 0)
    return with_counts(out)


def stamp_footer_from_web_row(
    footer: dict[str, Any] | None,
    sdata: dict[str, Any] | None,
    sid: str = "",
    *,
    title: str = "",
    model: str = "",
    state: str = "",
    live: bool | None = None,
) -> dict[str, Any]:
    """Stamp footer ``current`` from the web row the picker already uses."""
    from sharedagents_web.session_store import SessionStore

    row = sdata or {}
    return stamp_viewed_current(
        footer,
        agent=SessionStore.display_agent(row),
        title=str(title or row.get("title") or ""),
        model=str(model or row.get("model") or row.get("engine_model") or ""),
        sid=str(sid or row.get("id") or ""),
        state=str(state or ""),
        live=live,
    )


_ENGINE_AGENT = {
    "antigravity_cli": "antigravity",
    "antigravity": "antigravity",
    "zcode_cli": "zcode",
    "zcode": "zcode",
    "cursor": "cursor",
    "claude": "claude",
    "codex": "codex",
    "opencode": "opencode",
    "kimi": "kimi",
    "hermes": "hermes",
    "devin": "devin",
    "devin_acp": "devin",
    "local_llm": "general",
    "delegate_local_llm": "general",
    "delegate_antigravity": "antigravity",
}


def _agent_from_sidecar(data: dict[str, Any], filename: str) -> str:
    for key in ("agent", "agent_name", "child_agent"):
        val = str(data.get(key) or "").strip()
        if val:
            return val
    engine = str(data.get("engine") or "").strip().lower()
    if engine in _ENGINE_AGENT:
        return _ENGINE_AGENT[engine]
    if "delegate_" in filename:
        tail = filename.rsplit("delegate_", 1)[-1].replace(".json", "").strip()
        return _ENGINE_AGENT.get(tail, tail.replace("_cli", "") or "sub")
    return "sub"


def _agent_from_child_jsonl(path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            line = fh.readline()
    except Exception:
        return ""
    raw = line.strip()
    if not raw:
        return ""
    if raw[:1] != "{":
        sp = raw.find(" ")
        raw = raw[sp + 1 :] if sp > 0 else raw
    try:
        ev = json.loads(raw)
    except Exception:
        return ""
    if not isinstance(ev, dict):
        return ""
    return str(ev.get("agent_name") or ev.get("agent") or "").strip()


def historical_sub_rows(sid: str, sessions_dir) -> list[dict[str, Any]]:
    """Durable child rows from ``<sid>.subagent.*.json`` sidecars."""
    from pathlib import Path

    root = Path(sessions_dir)
    key = str(sid or "").strip()
    if not key or not _SID_RE.fullmatch(key) or not root.is_dir():
        return []
    now = time.monotonic()
    cached = _HIST_CACHE.get(key)
    if cached and now - cached[0] < _HIST_TTL_S:
        return list(cached[1])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    pattern = f"{glob.escape(key)}.subagent.*.json"
    for path in sorted(root.glob(pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        child = str(data.get("sharedagents_child_session_id") or "").strip()
        if not child or child in seen or not _SID_RE.fullmatch(child):
            continue
        seen.add(child)
        agent = _agent_from_child_jsonl(root / f"{child}.jsonl") or _agent_from_sidecar(
            data,
            path.name,
        )
        rows.append(
            {
                "hash": child,
                "agent": agent or "sub",
                "model": str(data.get("engine") or ""),
                "title": "",
                "state": "idle",
                "live": False,
                "depth": 1,
                "role": "sub",
                "last_activity_at": 0,
                "created_at": "",
                "parent_hash": key,
                "subagent_count": 0,
            }
        )
    _HIST_CACHE[key] = (now, rows)
    return list(rows)


def enrich_footer_with_history(
    footer: dict[str, Any] | None,
    sid: str,
    sessions_dir,
) -> dict[str, Any]:
    """Fill empty live-tree footers from durable subagent sidecars."""
    base: dict[str, Any] = dict(footer or {"session": str(sid or ""), "rows": [], "subs": 0})
    hist = historical_sub_rows(sid, sessions_dir)
    if not hist:
        return with_counts(base)
    rows = list(base.get("rows") or [])
    existing = {str(row.get("hash") or "") for row in rows if row.get("hash")}
    extras = [row for row in hist if row.get("hash") not in existing]
    live_subs = sum(1 for row in rows if row.get("role") == "sub")
    base["subs"] = max(int(base.get("subs") or 0), live_subs + len(extras), len(hist))
    if extras:
        base["rows"] = rows + extras
    return with_counts(base)
