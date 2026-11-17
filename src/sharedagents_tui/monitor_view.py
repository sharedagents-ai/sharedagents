"""Pure presentation logic for the TUI agent monitor.

A faithful Python port of the filter/flatten/age helpers in
``web/static/agents.js`` so the terminal monitor renders the SAME
cross-agent ``orchestrator.session_tree()`` the web ``/agents`` page
does — agent → session → subagent — with the same client-side filters
(a "seen ≤" age window and a free-text query) composing on top of the
server-side ``?all=1`` scope.

Everything here is a pure function over the JSON tree dict (no Textual
imports, no clock reads — ``now_ms`` is always injected) so it is
unit-testable in isolation and stays in lock-step with the JS.

Tree shape (from :func:`sharedagents_agent.orchestrator.session_tree`)::

    {
      "agents": [
        {agent, active_sessions, subagents_calling,
         roots: [ <node>, ... ]},
        ...
      ],
      "summary": [...], "totals": {...}, "live_states": [...],
    }

where each ``<node>`` is::

    {hash, title, agent, engine, model, state, live, stale, round, turn,
     last_turn_ts, state_updated_at, subagent_count, children: [...]}
"""

from __future__ import annotations

from typing import Any


def _last_seen_ms(node: dict[str, Any]) -> int:
    """Most-recent activity for a session node, in epoch-ms.

    Mirrors ``agents.js:lastSeenMs`` — prefer ``state_updated_at`` (set on
    every streamed delta / tool event), fall back to ``last_turn_ts``.
    The writer stores either epoch-seconds or epoch-ms depending on its
    vintage; values below 1e12 are seconds and get scaled to ms. ``0``
    means "never seen".
    """
    ts = node.get("state_updated_at") or node.get("last_turn_ts") or 0
    if not ts:
        return 0
    return int(ts) if ts > 1e12 else int(ts) * 1000


def within_age(node: dict[str, Any], max_age_sec: float, now_ms: int) -> bool:
    """True if ``node`` was seen recently enough for the age window.

    ``max_age_sec == 0`` is "any" (no cutoff). A zero-timestamp (never
    seen) node only passes when no window is set, so stale/unknown rows
    don't masquerade as fresh under a tight filter.
    """
    if not max_age_sec:
        return True
    ms = _last_seen_ms(node)
    if not ms:
        return False
    return (now_ms - ms) <= max_age_sec * 1000


def matches_query(node: dict[str, Any], query: str) -> bool:
    """True if the lower-cased ``query`` is a substring of the node's
    agent / title / hash. Empty query matches everything."""
    if not query:
        return True
    hay = (f"{node.get('agent', '')} {node.get('title', '')} {node.get('hash', '')}").lower()
    return query in hay


def node_kept(
    node: dict[str, Any],
    max_age_sec: float,
    query: str,
    now_ms: int,
) -> bool:
    """A session node is kept if it passes BOTH filters itself OR any
    descendant does — so an old/non-matching parent is never pruned out
    from under a fresh/matching subagent (mirrors ``agents.js:nodeKept``)."""
    if within_age(node, max_age_sec, now_ms) and matches_query(node, query):
        return True
    return any(node_kept(c, max_age_sec, query, now_ms) for c in (node.get("children") or []))


def build_rows(
    tree: dict[str, Any],
    *,
    collapsed: set[str],
    query: str = "",
    max_age_sec: float = 0,
    now_ms: int,
) -> list[dict[str, Any]]:
    """Flatten the tree into an ordered, navigable row list.

    Each row is a dict::

        {key, kind: "agent"|"session", depth, has_kids, collapsed,
         agent?: <agent dict>, node?: <session node>}

    - ``key`` is ``"agent:<name>"`` for agent rows and the session hash
      for session rows (the collapse-set key, and — for sessions — the
      hash the operator opens).
    - An agent whose sessions are all filtered out is dropped entirely.
    - A collapsed agent emits its header row but none of its sessions;
      a collapsed session emits its row but none of its subagents.
    """
    rows: list[dict[str, Any]] = []
    for agent in tree.get("agents") or []:
        kept_roots = [r for r in (agent.get("roots") or []) if node_kept(r, max_age_sec, query, now_ms)]
        if not kept_roots:
            continue
        akey = "agent:" + str(agent.get("agent", ""))
        is_collapsed = akey in collapsed
        rows.append(
            {
                "key": akey,
                "kind": "agent",
                "depth": 0,
                "agent": agent,
                "has_kids": True,
                "collapsed": is_collapsed,
            }
        )
        if is_collapsed:
            continue
        for root in kept_roots:
            _push_node(root, 1, rows, collapsed, query, max_age_sec, now_ms)
    return rows


def _push_node(
    node: dict[str, Any],
    depth: int,
    rows: list[dict[str, Any]],
    collapsed: set[str],
    query: str,
    max_age_sec: float,
    now_ms: int,
) -> None:
    kept_kids = [c for c in (node.get("children") or []) if node_kept(c, max_age_sec, query, now_ms)]
    hsh = node.get("hash") or ""
    is_collapsed = hsh in collapsed
    rows.append(
        {
            "key": hsh,
            "kind": "session",
            "depth": depth,
            "node": node,
            "has_kids": bool(kept_kids),
            "collapsed": is_collapsed,
        }
    )
    if is_collapsed:
        return
    for c in kept_kids:
        _push_node(c, depth + 1, rows, collapsed, query, max_age_sec, now_ms)


def age_str(ts: float | None, now_ms: int) -> str:
    """Relative age (``5s`` / ``3m`` / ``2h`` / ``4d``) for a timestamp.

    ``ts`` may be epoch-seconds or epoch-ms (normalised like
    :func:`_last_seen_ms`); ``0`` / ``None`` renders blank. Mirrors
    ``agents.js:ageStr``."""
    if not ts:
        return ""
    ms = int(ts) if ts > 1e12 else int(ts) * 1000
    s = max(0, (now_ms - ms) // 1000)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"
