"""Who may do what in the Web UI — the one place those decisions are made.

* **Roles** — an account is an admin (the single operator)
  or a user. :func:`admin_source` says why someone is an admin.
* **Grants** — the agents / models / tools / skills a user may use
  (:mod:`sharedagents_core.access_grants`); admins have All.
* **Conversation access** — :func:`session_access`: admins manage every
  conversation; an owner manages theirs; a person a conversation is shared
  with may read and chat in it; anyone else gets 404.
* **The gate** — :func:`gate_request`, run by ``require_auth`` for every
  authenticated request. A non-admin reaches only the endpoints listed in
  :data:`USER_ENDPOINTS` (default deny: a new route is admin-only until it is
  listed), and a listed session-scoped endpoint only for a conversation they
  can reach.

Every read goes to SQLite (gunicorn runs several workers; an in-process cache
would go stale when an admin changes someone). Results are memoised within one
request for a few seconds only — an SSE stream is one request that lives for
minutes, and its per-tick checks pass ``fresh=True``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from flask import g, jsonify, redirect, request

from sharedagents_core.access_grants import Grants, delegation_refusal
from sharedagents_web.native_auth import resolve_admin_source

NONE, VIEW, CHAT, MANAGE = 0, 1, 2, 3
LEVEL_NAMES = {NONE: "none", VIEW: "view", CHAT: "chat", MANAGE: "manage"}
_LINEAGE_MAX_DEPTH = 16
_MEMO_TTL_S = 5.0
_MISSING = object()


def _server():
    from sharedagents_web import server

    return server


def _norm(email) -> str:
    return str(email or "").strip().lower()


def _memo() -> dict:
    try:
        memo = g.get("_sa_access_memo")
        if memo is None:
            memo = {}
            g._sa_access_memo = memo
        return memo
    except RuntimeError:  # no request / app context
        return {}


def _memoised(key, compute, *, fresh: bool = False):
    memo = _memo()
    now = time.monotonic()
    hit = memo.get(key, _MISSING)
    if not fresh and hit is not _MISSING and now - hit[0] < _MEMO_TTL_S:
        return hit[1]
    value = compute()
    memo[key] = (now, value)
    return value


def forget() -> None:
    """Drop this request's memo (after an admin changes roles or sharing)."""
    try:
        g._sa_access_memo = {}
    except RuntimeError:
        pass


# -- roles ------------------------------------------------------------------


def _admin_source_uncached(email: str) -> str | None:
    srv = _server()
    if srv._loopback_debug_bypass_ok():
        return "debug"  # a loopback debug instance has one identity: the operator
    return resolve_admin_source(srv.auth_store, email)


def admin_source(email, *, fresh: bool = False) -> str | None:
    """Why ``email`` is an admin (``role``/``env``/``operator``/``debug``/
    ``fallback``), or ``None`` for a user."""
    e = _norm(email)
    if not e:
        return None
    return _memoised(("admin", e), lambda: _admin_source_uncached(e), fresh=fresh)


def is_admin(email, *, fresh: bool = False) -> bool:
    return admin_source(email, fresh=fresh) is not None


def viewer_is_admin() -> bool:
    return is_admin(getattr(g, "user_email", ""))


def grants_for(email) -> Grants:
    """What ``email`` may use: All for admins, none for unknown/disabled."""
    e = _norm(email)
    if is_admin(e):
        return Grants.all()
    acct = _server().auth_store.account(e)
    if acct is None or acct["disabled"]:
        return Grants.none()
    return acct["grants"]


# -- agents and models --------------------------------------------------------

_EXTERNAL_CACHE: dict[str, tuple[float, bool]] = {}
_EXTERNAL_TTL_S = 30.0


def agent_is_external(name: str) -> bool:
    """Whether ``name`` runs an external engine (Claude Code, Codex, …) that
    brings its own tools and skills. Unknown agents count as external (fail
    closed)."""
    key = str(name or "").strip().lower()
    now = time.monotonic()
    hit = _EXTERNAL_CACHE.get(key)
    if hit is not None and now - hit[0] < _EXTERNAL_TTL_S:
        return hit[1]
    try:
        from sharedagents_agent.agent import AgentConfig
        from sharedagents_agent.external_registry import EXTERNAL_ENGINE_VALUES

        engine = str(getattr(AgentConfig.load(key), "engine", "") or "")
        external = engine in EXTERNAL_ENGINE_VALUES
    except Exception:  # noqa: BLE001
        external = True
    _EXTERNAL_CACHE[key] = (now, external)
    return external


def valid_agent_name(name) -> bool:
    """An agent name safe to splice into a gateway path (``[A-Za-z0-9_.-]``)."""
    try:
        from sharedagents_agent.agent_editor import AgentEditorError, validate_agent_name
    except Exception:  # noqa: BLE001
        return False
    try:
        validate_agent_name(str(name or ""))
    except AgentEditorError:
        return False
    return True


def agent_refusal(email, agent: str) -> str | None:
    """Why ``email`` may not use ``agent``, else ``None`` (admins: never)."""
    if not valid_agent_name(agent):
        return "invalid agent name"
    if is_admin(email):
        return None
    name = str(agent or "").strip().lower()
    return delegation_refusal(
        grants_for(email),
        name,
        external=agent_is_external(name),
    )


_MODEL_ROW_KEYS = ("id", "alias", "wireId", "realId")


def _model_alias_index() -> dict[str, set[str]]:
    """Cached-catalogue alias index (``cache_lock.model_alias_index``)."""
    try:
        from sharedagents_agent.discovery.cache_lock import model_alias_index

        return model_alias_index()
    except Exception:  # noqa: BLE001
        return {}


def expand_model_names(*names: str) -> set[str]:
    """``names`` plus every alias/id the catalogue knows them by."""
    wanted = {str(n or "").strip() for n in names} - {""}
    index = _model_alias_index()
    out = set(wanted)
    for name in wanted:
        out |= index.get(name, set())
    return out


def model_refusal(email, *candidates: str) -> str | None:
    """Why ``email`` may not use a model known by any of ``candidates``."""
    if is_admin(email):
        return None
    wanted = [str(c).strip() for c in candidates if str(c or "").strip()]
    if not wanted:
        return None
    if grants_for(email).allows("models", *expand_model_names(*wanted)):
        return None
    return f"model {wanted[0]!r} is not granted to your account"


def _row_allowed(grants: Grants, row) -> bool:
    if grants.is_all("models"):
        return True
    if not isinstance(row, dict):
        return False
    names = [str(row.get(k) or "").strip() for k in _MODEL_ROW_KEYS]
    return grants.allows("models", *expand_model_names(*names))


def filter_model_rows(email, agent: str, rows) -> list:
    """The catalogue rows of ``agent`` that ``email`` may pick."""
    rows = list(rows or [])
    if is_admin(email):
        return rows
    if agent_refusal(email, agent):
        return []
    grants = grants_for(email)
    return [r for r in rows if _row_allowed(grants, r)]


def filter_models_catalogue(email, payload):
    """Per-viewer view of the stream's deduplicated ``models`` catalogue."""
    if not payload or is_admin(email):
        return payload
    import hashlib
    import json

    catalogues = payload.get("catalogues") or {}
    agents_out: dict[str, str] = {}
    lists: dict[str, list] = {}
    for agent, digest in (payload.get("agents") or {}).items():
        if agent_refusal(email, agent):
            continue
        rows = filter_model_rows(email, agent, catalogues.get(digest) or [])
        new_digest = hashlib.sha1(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        lists.setdefault(new_digest, rows)
        agents_out[str(agent)] = new_digest
    revision = hashlib.sha1(json.dumps(agents_out, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return {"ok": True, "revision": revision, "catalogues": lists, "agents": agents_out}


def usable_agents(email, names) -> list[str]:
    return [n for n in names if agent_refusal(email, n) is None]


# -- what a submit carries to the gateway ---------------------------------------

DIRECT_DISPATCH_PREFIX = "__SHAREDAGENTS_DELEGATE__:"


def access_policy_for(email) -> dict | None:
    """The ``access_policy`` a submit carries: ``None`` (unrestricted) for
    admins, else the submitter's grants."""
    if is_admin(email):
        return None
    from sharedagents_core.access_grants import AccessStamp

    return AccessStamp(_norm(email), grants_for(email)).to_obj()


def has_direct_dispatch(prompt, attachments=None) -> bool:
    """Whether a submit would open with the gateway's direct-dispatch
    sentinel (which runs a delegate tool without the model; TUI/Telegram
    only) for anything but ``/compact``.

    Judged on the text the turn will see: every block's text, attachments
    included, joined with no separator — so a sentinel split across blocks
    is caught.
    """
    parts: list = []
    for raw in (prompt, attachments):
        if isinstance(raw, str):
            parts.append(raw)
        elif isinstance(raw, dict):
            parts.append(str(raw.get("text") or ""))
        elif isinstance(raw, list):
            parts.extend(str(b.get("text") or "") for b in raw if isinstance(b, dict))
    flat = "".join(parts).lstrip()
    if not flat.startswith(DIRECT_DISPATCH_PREFIX):
        return False
    tool = flat[len(DIRECT_DISPATCH_PREFIX) :].partition(":")[0].strip()
    from sharedagents_session.access_binding import DIRECT_DISPATCH_ALWAYS

    return tool not in DIRECT_DISPATCH_ALWAYS


# Slash commands a user may run (canonical names, see sharedagents_command.commands
# parse_command). Anything else — /loop, /goal, /budget, /plugin, /mcp,
# /skill(s), /restart, /teacher, /models, /agents, /sessions, /memory,
# /steer, file commands — stays with admins. ``/<agent> …`` is allowed for
# granted agents.
USER_COMMANDS = frozenset(
    {
        "help",
        "status",
        "context",
        "todo",
        "model",
        "mode",
        "reasoning",
        "effort",
        "thinking",
        "tool",
        "history",
        "refresh",
        "compact",
        "abort",
        "new",
        "agent",
        "title",
        "subagents",
    }
)


def command_refusal(email, text: str) -> str | None:
    """Why ``email`` may not run slash command ``text``, else ``None``."""
    if is_admin(email):
        return None
    from sharedagents_command.commands import parse_command

    parsed = parse_command(text)
    if parsed is None:
        return "not a slash command"
    name, arg = parsed.name, str(parsed.arg or "").strip()
    if name in USER_COMMANDS:
        if name == "agent" and arg:
            return agent_refusal(email, arg.split()[0])
        if name == "model" and arg:
            target = arg.split()[0]
            if target.lower() in ("auto", "ask", "plan"):
                return None
            return model_refusal(email, target)
        return None
    try:
        from sharedagents_agent.agent import list_agents

        agents = {str(a).lower() for a in list_agents()}
    except Exception:  # noqa: BLE001
        agents = set()
    if name.lower() in agents:
        return agent_refusal(email, name)
    return f"/{name} is not available to your account"


# -- conversations ------------------------------------------------------------


def row_owner(row) -> str:
    """The person who owns a Web conversation row ('' = the operator).

    A gateway-native conversation (TUI, Telegram, CLI) belongs to the
    installation operator however it reached the Web store: adoption stamps
    the adopter's address, which must never make them its owner.
    """
    if not isinstance(row, dict) or str(row.get("origin") or "") == "gateway":
        return ""
    return _norm(row.get("owner"))


def _lineage(sid: str) -> list[tuple[str, dict | None]]:
    """``sid`` and its ancestors, nearest first, each with its Web row."""
    from sharedagents_web.routes.sessions import _canonical_gateway_parent

    srv = _server()
    out: list[tuple[str, dict | None]] = []
    seen: set[str] = set()
    cur = sid
    for _ in range(_LINEAGE_MAX_DEPTH):
        if not cur or cur in seen:
            break
        seen.add(cur)
        row = srv.store.get(cur)
        out.append((cur, row))
        if row_owner(row):
            break  # a Web-created conversation roots ownership
        ok, parent = _canonical_gateway_parent(cur)
        if not ok or not parent:
            break
        cur = parent
    return out


def _session_access_uncached(viewer: str, sid: str, fresh: bool = False) -> int:
    if is_admin(viewer, fresh=fresh):
        return MANAGE
    chain = _lineage(sid)
    for _cid, row in chain:
        if row_owner(row) == viewer:
            return MANAGE  # the owner's whole tree: the chat and its sub-agents
    shares = _server().sharing_store.get_many(cid for cid, _ in chain)
    if any(sh.admits(viewer) for sh in shares.values()):
        if chat_refusal(viewer, _conversation_row(sid, chain)):
            return VIEW
        return CHAT
    return NONE


def _conversation_row(sid: str, chain) -> dict | None:
    """The agent/model ``sid`` runs on: its Web row, else the gateway's
    session index (a TUI/Telegram conversation), else its nearest Web
    ancestor's."""
    own = chain[0][1] if chain and isinstance(chain[0][1], dict) else None
    if own is not None:
        return own
    try:
        meta = _server()._load_meta_headers([sid]).get(sid) or {}
    except Exception:  # noqa: BLE001
        meta = {}
    if meta.get("agent"):
        return {"agent": meta.get("agent"), "model": meta.get("model") or ""}
    return next((r for _cid, r in chain if isinstance(r, dict)), None)


def session_access(viewer, sid, *, fresh: bool = False) -> int:
    """``NONE`` / ``VIEW`` / ``CHAT`` / ``MANAGE`` for ``viewer`` on ``sid``.

    A sub-agent session inherits its parent conversation's access. A person
    the conversation is shared with who cannot use its agent or model gets
    ``VIEW`` (read, no prompt/answer/cancel). Owners stay ``MANAGE``.
    """
    v, s = _norm(viewer), str(sid or "").strip()
    if not v or not s:
        return NONE
    return _memoised(
        ("session", v, s),
        lambda: _session_access_uncached(v, s, fresh),
        fresh=fresh,
    )


def visible_shared_ids(viewer) -> set[str]:
    """Conversations shared with ``viewer`` (everyone, or by name)."""
    return _server().sharing_store.shared_ids_for(_norm(viewer))


def sharing_payload(sid) -> dict:
    """A conversation's sharing with display names, for the chat header."""
    srv = _server()
    out = srv.sharing_store.get(sid).to_dict()
    out["members"] = [{"email": m, "name": srv.auth_store.display_name(m)} for m in out["members"]]
    return out


def sharing_payloads(sids) -> dict[str, dict]:
    """``sharing_payload`` for many conversations: one bulk read of the
    sharing table, each member's display name looked up once."""
    srv = _server()
    names: dict[str, str] = {}

    def member(email: str) -> dict:
        if email not in names:
            names[email] = srv.auth_store.display_name(email)
        return {"email": email, "name": names[email]}

    out = {}
    for sid, sharing in srv.sharing_store.get_many(sids).items():
        payload = sharing.to_dict()
        payload["members"] = [member(m) for m in payload["members"]]
        out[sid] = payload
    return out


def session_view(viewer, sid, payload) -> dict:
    """Access fields the chat page renders for one opened conversation.

    ``access`` is ``manage`` / ``chat`` / ``view``: a person a conversation
    is shared with reads it but cannot prompt when its agent or model is not
    granted to them (``read_only_reason`` says why).
    """
    srv = _server()
    level = session_access(viewer, sid)
    owner = row_owner(srv.store.get(sid))
    info = {
        "access": LEVEL_NAMES.get(level, "none"),
        "owner_email": owner,
        "owner_name": srv.auth_store.display_name(owner) if owner else "",
        "sharing": sharing_payload(sid),
    }
    if level == VIEW:
        refusal = chat_refusal(viewer, payload) or "you can read this conversation"
        info["access"] = "view"
        info["read_only_reason"] = f"View only — {human_reason(refusal)}"
    return info


def human_reason(text) -> str:
    """A refusal as a person reads it: the ``[access]`` tag is for logs and
    models, and "this user" is the reader."""
    reason = str(text or "").strip()
    if reason.startswith("[access]"):
        reason = reason[len("[access]") :].strip()
    return reason.replace("to this user", "to you")


def chat_refusal(viewer, row) -> str | None:
    """Why ``viewer`` may read but not prompt a conversation, else ``None``."""
    if is_admin(viewer) or not isinstance(row, dict):
        return None
    agent = str(row.get("agent") or "").strip()
    if agent:
        refusal = agent_refusal(viewer, agent)
        if refusal:
            return refusal
    model = str(row.get("model") or "").strip()
    if model and model.lower() not in ("auto", "ask", "plan"):
        return model_refusal(viewer, model, str(row.get("engine_model") or ""))
    return None


# -- the gate -----------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """How a non-admin may use one endpoint.

    ``level`` — conversation access required (``NONE`` = not session-scoped);
    ``source`` — where the session id is (``view:<arg>``, ``args:<name>``,
    ``json:<name>``; ``|`` separates alternatives); ``optional`` — the id may
    be absent; ``by_method`` — per-HTTP-method level overrides.
    """

    level: int = NONE
    source: str = ""
    optional: bool = False
    by_method: tuple = ()

    def level_for(self, method: str) -> int:
        for name, level in self.by_method:
            if name == method:
                return level
        return self.level


_OPEN = Rule()
_VIEW_SID = Rule(VIEW, "view:sid")
_CHAT_SID = Rule(CHAT, "view:sid")

# Every endpoint a non-admin may call. Anything else is admin-only (403 /
# redirect to the chat). Handlers marked "scoped" filter what they return;
# "grants" check agent/model grants themselves.
USER_ENDPOINTS: dict[str, Rule] = {
    # the chat page and identity
    "chat.index": _OPEN,
    "auth.api_verify": _OPEN,
    "chat.api_health": _OPEN,
    "chat.api_extract_document": _OPEN,
    "voice.api_transcribe": _OPEN,
    "presence.api_presence": _OPEN,
    "presence.api_continue_here": _OPEN,  # scoped
    "presence.api_session_handoff": _VIEW_SID,
    # conversations
    "sessions.api_boot": _OPEN,  # scoped
    "sessions.api_list_sessions": _OPEN,  # scoped
    "sessions.api_create_session": _OPEN,  # grants
    "sessions.api_models": _OPEN,  # scoped (roster)
    "sessions.api_remember_model_effort": _OPEN,  # agent/model grants
    "sessions.api_agents_picker": _OPEN,  # scoped (roster + granted rows)
    "sessions.api_discover_models": Rule(VIEW, "json:session_id", optional=True),  # scoped
    "sessions.api_approvals": _OPEN,  # scoped
    "sessions.api_get_session": _VIEW_SID,
    "sessions.api_session_state": _VIEW_SID,
    "sessions.api_agent_capabilities": _VIEW_SID,
    "sessions.api_session_card": _VIEW_SID,
    "sessions.api_session_context": _VIEW_SID,
    "sessions.api_session_context_item": _VIEW_SID,
    "sessions.api_session_live": _VIEW_SID,
    "sessions.api_session_stream": _VIEW_SID,
    "sessions.api_session_events": _VIEW_SID,
    "sessions.api_session_queue_list": _VIEW_SID,
    "sessions.api_switch_agent": _CHAT_SID,  # grants
    "sessions.api_session_prompt": _CHAT_SID,  # grants
    "sessions.api_session_answer": _CHAT_SID,
    # Approving a tool call can widen what a turn does: owner/admin only, so a
    # person a chat is shared with never approves a wider-granted turn's tools.
    "sessions.api_session_permission": Rule(MANAGE, "view:sid"),
    "sessions.api_session_queue_add": _CHAT_SID,  # grants
    "sessions.api_session_queue_update": _CHAT_SID,  # re-stamped with the editor's grants
    "sessions.api_session_queue_delete": _CHAT_SID,
    "sessions.api_session_queue_prepare": _CHAT_SID,
    "sessions.api_session_queue_steer": _CHAT_SID,
    "sessions.api_session_cancel": _CHAT_SID,
    "sessions.api_delete_session": Rule(MANAGE, "view:sid"),
    "artifacts.serve_artifact": _VIEW_SID,
    "artifacts.api_session_artifact": _CHAT_SID,
    "commands.api_commands": _OPEN,  # scoped
    "commands.api_command": Rule(CHAT, "json:session_id", optional=True),  # grants
    "agents.api_agents_footer": Rule(VIEW, "args:session"),
    "agents.api_agents_abort": Rule(CHAT, "json:session_id|sid"),
    "goals.api_budget": Rule(VIEW, "view:session_hash"),
    "agent_builder.api_agent_def_get": _OPEN,  # scoped (projection)
    "topics.api_list_topics": _OPEN,  # scoped (empty)
    "sharing.api_session_sharing": Rule(VIEW, "view:sid", by_method=(("PUT", MANAGE),)),
    "users.api_user_directory": _OPEN,
    "tasks.tasks_page": _OPEN,
    "tasks.api_tasks_list": _OPEN,
}


def _session_id_from(source: str) -> str:
    kind, _, names = source.partition(":")
    for name in names.split("|"):
        if kind == "view":
            value = (request.view_args or {}).get(name)
        elif kind == "args":
            value = request.args.get(name)
        elif kind == "json":
            body = request.get_json(silent=True)
            value = body.get(name) if isinstance(body, dict) else None
        else:
            value = None
        value = str(value or "").strip()
        if value:
            return value
    return ""


def not_found():
    resp = jsonify({"error": "Session not found"})
    resp.status_code = 404
    return resp


def admin_only():
    if request.method == "GET" and not request.path.startswith("/api/"):
        return redirect((_server().BASE_PATH or "") + "/")
    resp = jsonify({"error": "admin only"})
    resp.status_code = 403
    return resp


def gate_request(email):
    """``None`` to let the request through, else the refusal response."""
    # Native token verification rejects disabled accounts already. The optional
    # dashboard login must obey the same local account status on every request.
    acct = _server().auth_store.account(_norm(email))
    if acct is not None and acct["disabled"]:
        resp = jsonify({"error": "account disabled"})
        resp.status_code = 401
        return resp
    if is_admin(email):
        return None
    rule = USER_ENDPOINTS.get(request.endpoint or "")
    if rule is None:
        return admin_only()
    level = rule.level_for(request.method)
    if level == NONE or not rule.source:
        return None
    sid = _session_id_from(rule.source)
    if not sid:
        return None if rule.optional else not_found()
    if session_access(email, sid) < level:
        return not_found()
    return None
