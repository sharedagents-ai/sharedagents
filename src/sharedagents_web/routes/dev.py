"""sharedagents_web.routes.dev — Dev tab: forge-backed closed development loops.

Devin-shaped: name an object + done-when, run inner laps on one branch, publish
one PR. Merge is automatic when independent verify A + verify B and LSP are
green; deploy stays a click. Live PR/CI overlay comes from the forge API.

This is not the scheduled Loops tab. Starting a Dev run mints one Jarvis
conductor session and registers it on the Tasks board so the same run is
reachable from Dev, Chat, and Tasks.

The ``/api/dev/sdlc/*`` surface underneath is the SDLC control plane: intake,
the run state machine, automatic merge, and the deploy call that stays the
operator's.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from flask import Blueprint, g, jsonify, render_template, request, send_from_directory

from sharedagents_web import forge_client
from sharedagents_web.dev_flow import FlowConflict, FlowError, get_flow_registry
from sharedagents_web.dev_forge import (
    cap_forge_panel,
    fill_forge_panel,
    forge_limit_arg,
)
from sharedagents_web.dev_loop import (
    branch_name,
    closed_loop_prompt,
    continue_prompt,
    has_issue,
    project_loop,
)
from sharedagents_web.dev_registry import get_dev_registry
from sharedagents_web.principal import gateway_principal_headers
from sharedagents_web.server import (
    BASE_PATH,
    DEFAULT_AGENT,
    REALTIME_URL,
    gateway_request,
    require_auth,
    store,
)

log = logging.getLogger(__name__)

bp = Blueprint("dev", __name__)


def _flow_actor(payload: dict, *, default_role: str, human: bool = False) -> dict:
    """Normalize persisted stage identity; content/control remains human-only."""
    supplied = payload.get("actor")
    if isinstance(supplied, dict):
        actor = dict(supplied)
    else:
        email = str(getattr(g, "user_email", "operator") or "operator")
        actor = {
            "role": "human" if human else default_role,
            "actor_kind": "human" if human else "ai",
            "profile": "web",
            "engine": "web",
            "model_id": email if human else "web-stage-runner",
            "model_version": "1",
            "session_id": "web-" + email,
            "context_id": "web-control",
        }
    actor.setdefault("role", "human" if human else default_role)
    if human and actor.get("actor_kind") != "human":
        raise PermissionError("human control is required")
    return actor


def _flow_detail(registry, flow_id: str) -> dict | None:
    flow = registry.get_flow(flow_id)
    if not flow:
        return None
    flow = dict(flow)
    flow["artifacts"] = registry.artifacts(flow_id)
    flow["stage_results"] = registry.stage_results(flow_id)
    flow["transitions"] = registry.events(flow_id)
    flow["disputes"] = registry.disputes(flow_id)
    flow["roles"] = registry.roles(flow_id)
    flow["configuration_index"] = registry.configuration_index(flow_id)
    return flow


def _require_repo(raw: str) -> str:
    repo = (raw or "").strip()
    if not repo:
        raise ValueError("repo is required")
    forge_client.split_repo(repo)
    return repo


def _optional_json_object() -> dict | None:
    """Parse an optional JSON object body.

    Empty or absent body is ``{}`` so query-only clients still work. A
    non-empty body that is malformed, JSON null, or any non-object is
    ``None`` so the caller can 400 before side effects.
    """
    raw = request.get_data(cache=True, as_text=True)
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _overlay_forge(row: dict) -> dict:
    """Project live forge state onto a loop row. Never raises."""
    out = dict(row)
    out["forge_configured"] = forge_client.configured()
    repo = str(out.get("repo") or "").strip()
    branch = out.get("branch") or ""
    if not forge_client.configured() or not branch or not repo:
        out.setdefault("live_pr", None)
        return project_loop(out, session_live=False, forge_configured=out.get("forge_configured"))
    try:
        status, pulls = forge_client.list_pulls(repo, state="open")
        out["forge_status"] = status
        match = next((p for p in pulls if p.get("head") == branch), None)
        # also see merged by listing closed if no open match and we have a number
        if match is None and out.get("pr_number"):
            st2, closed = forge_client.list_pulls(repo, state="closed")
            if st2 == 200:
                match = next(
                    (p for p in closed if int(p.get("number") or 0) == int(out["pr_number"])),
                    None,
                )
        out["live_pr"] = match
        if match:
            sha = match.get("sha") or ""
            out["last_sha"] = sha or out.get("last_sha") or ""
            out["pr_number"] = match.get("number") or out.get("pr_number")
            out["pr_url"] = match.get("html_url") or out.get("pr_url") or ""
            if match.get("merged"):
                out["state"] = "merged"
            elif out.get("state") in ("inner", "blocked"):
                out["state"] = "pr_open"
                if not out.get("published_at"):
                    out["published_at"] = match.get("updated_at") or ""
            if sha:
                _, ci = forge_client.commit_status(repo, sha)
                out["ci_status"] = ci.get("state") or ""
                if out["ci_status"] in ("failure", "error") and out["state"] not in (
                    "merged",
                    "aborted",
                ):
                    out["state"] = "ci_red"
                elif out["ci_status"] == "success" and out["state"] == "pr_open":
                    out["state"] = "ready"
            persist = {}
            for key in ("pr_number", "pr_url", "last_sha", "state", "ci_status", "published_at"):
                if out.get(key) not in (None, "") and out.get(key) != row.get(key):
                    persist[key] = out[key]
            if persist:
                try:
                    get_dev_registry().patch(str(row["id"]), **persist)
                except Exception:  # noqa: BLE001
                    pass
        else:
            st_b, br = forge_client.get_branch(repo, branch)
            if st_b == 200:
                out["last_sha"] = br.get("sha") or out.get("last_sha") or ""
    except Exception:  # noqa: BLE001
        log.warning("dev overlay failed for %s", out.get("id"))
    return project_loop(out, session_live=False, forge_configured=out.get("forge_configured"))


def _session_summary(row: dict | None) -> dict | None:
    if not row:
        return None
    sid = str(row.get("session_id") or "")
    return {
        "id": row.get("id"),
        "session_id": sid,
        "state": row.get("state") or "",
        "branch": row.get("branch") or "",
        "session_live": _session_live(sid) if sid else False,
        "has_session": bool(sid),
    }


def _issues_with_sessions(repo: str, issues: list[dict]) -> list[dict]:
    loops = [
        row
        for row in get_dev_registry().list_loops(include_done=True)
        if str(row.get("repo") or "") == repo and row.get("issue_number")
    ]
    by_num: dict[int, dict] = {}
    for row in loops:
        try:
            num = int(row["issue_number"])
        except (TypeError, ValueError):
            continue
        prev = by_num.get(num)
        if prev is None or str(row.get("updated_at") or "") >= str(prev.get("updated_at") or ""):
            by_num[num] = row
    out = []
    for issue in issues:
        rec = dict(issue)
        try:
            num = int(rec.get("number") or 0)
        except (TypeError, ValueError):
            num = 0
        bound = _session_summary(by_num.get(num))
        rec["loop"] = bound
        rec["has_session"] = bool(bound and bound.get("has_session"))
        out.append(rec)
    return out


def _session_live(session_id: str) -> bool:
    if not session_id:
        return False
    try:
        from sharedagents_goal.goal_registry import session_work_live

        live, _src = session_work_live(session_id)
        return bool(live)
    except Exception:  # noqa: BLE001
        return False


def _prompt_conductor(run: dict, prompt: str, *, agent: str = "") -> None:
    """Post one standard user message on the Jarvis conductor session."""
    from sharedagents_web.dev_sdlc import role_config

    sid = str(run.get("conductor_session_id") or "").strip()
    text = str(prompt or "").strip()
    if not sid or not text:
        return
    sdata = store.get(sid) or {}
    agent_name = agent or str(sdata.get("agent") or "jarvis")
    _generation_agent, selected_model = role_config(run, "generation")
    payload: dict | None = None
    http = 0
    for attempt in range(3):
        current_model = _session_model(sid)
        if (
            selected_model
            and current_model
            and not _apply_model(
                sid,
                agent=agent_name,
                model=selected_model,
                session_id=sid,
            )
        ):
            reason = getattr(_apply_model, "last_error", "") or "unknown reason"
            log.warning(
                "sdlc: conductor could not switch %s to %s: %s",
                sid,
                selected_model,
                reason,
            )
            return
        payload, http = _submit_prompt(
            sid,
            text,
            agent=agent_name,
            initial_model=(selected_model if selected_model and not current_model else ""),
        )
        if http < 400:
            return
        if http == 409 and attempt < 2:
            time.sleep(2)
            continue
        break
    log.warning(
        "sdlc: conductor prompt failed for %s (%s): %s",
        sid,
        http,
        (payload or {}).get("error") if isinstance(payload, dict) else payload,
    )


def _conductor_hash_set(run: dict) -> set[str]:
    """The conductor SID accepted by approval matching."""
    from sharedagents_web.dev_clarify import conductor_session_hashes

    sid = str(run.get("conductor_session_id") or "").strip()
    return conductor_session_hashes(sid)


def _last_turn_ended_at(run: dict) -> str:
    """ISO timestamp of the conductor's most recent ``turn_ended``, or "".

    An AskUserQuestion is answered by the WAITER inside a running turn. When
    that turn ends the waiter is gone, but the gateway's approvals list still
    advertises the call -- so the Dev tab offered a card whose submit could
    only ever return 404 "no pending question for this callId", and the stall
    guard saw "a question is pending" and refused to recover. Observed live on
    a real repo: the interview was frozen with an unanswerable question on
    screen and nothing able to move it.
    """
    for event in _conductor_jsonl_newest(run, limit=200):
        if str(event.get("type") or event.get("event") or "") == "turn_ended":
            return str(event.get("ts") or event.get("timestamp") or "")
    return ""


def _event_epoch(value) -> float | None:
    """Normalize gateway epoch numbers and JSONL ISO timestamps.

    Approval records use epoch seconds (occasionally milliseconds), while
    session events use timezone-qualified ISO strings.  Their raw string forms
    are not comparable: every epoch currently starts with ``1`` and every ISO
    year with ``2``, which made a newly raised question look older than an
    earlier completed turn.
    """
    raw = value
    if raw in (None, ""):
        return None
    try:
        stamp = float(raw)
    except (TypeError, ValueError):
        stamp = None
    if stamp is not None:
        # Accommodate seconds and the common higher-resolution epoch units.
        size = abs(stamp)
        if size >= 1e17:
            stamp /= 1e9
        elif size >= 1e14:
            stamp /= 1e6
        elif size >= 1e11:
            stamp /= 1e3
        return stamp
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    # A naive timestamp cannot be ordered safely against an epoch instant.
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


def _ask_is_orphaned(ask: dict, ended_at: str) -> bool:
    """True when this ask was raised BEFORE the last turn ended."""
    raised = _event_epoch((ask or {}).get("raised_at"))
    ended = _event_epoch(ended_at)
    # Unknown is live. Hiding an ask is safe only when both sides establish
    # its ordering; otherwise the operator loses the only answer surface.
    return bool(raised is not None and ended is not None and raised < ended)


def _conductor_pending_asks(run: dict) -> list[dict] | None:
    """Live AskUserQuestion cards on this run's Jarvis conductor.

    Returns a list when the gateway answered with a valid approvals
    schema (empty = nothing pending). Returns ``None`` when approval
    state is unknown — unreachable gateway, non-200, invalid JSON, or
    a 200 whose body is not ``{"ok": true, "approvals": [...]}``.
    Callers that only display cards may treat ``None`` like ``[]``;
    ratification must not.
    """
    from sharedagents_web.dev_clarify import pending_asks_for_session

    hashes = _conductor_hash_set(run)
    if not hashes:
        return []
    resp = gateway_request(
        "GET",
        "/control/v1/orchestrator/approvals",
        timeout=10,
    )
    if resp is None or getattr(resp, "status_code", None) != 200:
        return None
    try:
        body = resp.json()
    except (ValueError, TypeError, AttributeError):
        return None
    if not isinstance(body, dict):
        return None
    # Gateway success is always ``{"ok": True, "approvals": [...]}``. A 200
    # that lacks that list does not establish "none pending".
    if body.get("ok") is not True or "approvals" not in body:
        return None
    raw = body["approvals"]
    if not isinstance(raw, list):
        return None
    asks = pending_asks_for_session(raw, hashes)
    # Drop the ones nothing can answer any more. The gateway keeps advertising
    # an ask after the turn that raised it has ended, and presenting that as a
    # question makes the ONE human-in-the-loop step a dead end: submit returns
    # 404, and every recovery path reads "a question is pending" and stands
    # down.
    ended_at = _last_turn_ended_at(run)
    return [ask for ask in asks if not _ask_is_orphaned(ask, ended_at)]


def _close_conductor_interview(run: dict) -> int:
    """Resolve any question the conductor is still parked on. Returns how many.

    Ratification and the last answer race: the conductor can raise its NEXT
    question in the same second the operator freezes the contract. That ask is
    then unanswerable -- the intake gate closes as soon as the issue exists --
    so the conductor sits parked on it forever, and because every stage is
    dispatched to that same session, `plan` queues behind it and the run stalls
    at `plan/started` with no error and nothing running.

    Answering it (rather than cancelling the turn) keeps the conductor's own
    transcript coherent: it asked, and it is told the contract is frozen.
    """
    pending = _conductor_pending_asks(run) or []
    closed = 0
    for rec in pending:
        call_id = str(rec.get("call_id") or "").strip()
        gw = str(rec.get("session_hash") or "").strip()
        if not call_id or not gw:
            continue
        resp = gateway_request(
            "POST",
            f"/control/v1/orchestrator/sessions/{gw}/answer",
            json={
                "callId": call_id,
                "answers": [
                    {
                        "question": str(rec.get("summary") or ""),
                        "selected": [],
                        "notes": "The operator ratified the contract. Intake is closed; "
                        "this question is settled by the frozen requirements "
                        "and clearing evidence on the issue. Do not ask again.",
                    }
                ],
            },
            headers=gateway_principal_headers(g.user_email),
            timeout=15,
        )
        if resp is not None and resp.status_code < 400:
            closed += 1
        else:
            log.warning("sdlc: could not close the interview ask %s on %s", call_id, gw)
    return closed


def _guard_ratified_interview(run_id: str, owner: str, *, attempts: int = 90, delay: float = 1.0) -> None:
    """Close questions that race with ratification after its first snapshot.

    The conductor can be between model output and the question tool when the
    operator clicks Ratify. The immediate close then sees no approval, the
    question appears seconds later, and Plan queues behind a waiter no human
    should ever have to answer. Keep a short bounded drain across that race;
    the frozen issue contract is the answer authority.
    """
    from threading import Thread

    from sharedagents_web.dev_sdlc import get_sdlc_store
    from sharedagents_web.server import app

    def _go() -> None:
        try:
            with app.app_context():
                g.user_email = owner
                for attempt in range(max(1, int(attempts))):
                    fresh = get_sdlc_store().get(run_id)
                    if not fresh or not fresh.get("issue_number"):
                        return
                    _close_conductor_interview(fresh)
                    if attempt + 1 < attempts:
                        time.sleep(max(0.1, float(delay)))
        except Exception:  # noqa: BLE001 — guard failure must not break ratify
            log.exception("sdlc: late interview guard failed for %s", run_id)

    Thread(target=_go, name=f"sdlc-ratify-guard-{run_id}", daemon=True).start()


def _answer_conductor_ask(run: dict, entries: list[dict]) -> bool:
    """Unlock the Chat card with the same answers the Dev tab just recorded.

    Returns True only when the gateway accepted the answer. A missing card
    is not success — Chat still needs those answers once Jarvis asks.
    """
    from sharedagents_web.dev_clarify import chat_answers_for_pending_card

    pending = _conductor_pending_asks(run) or []
    if not pending:
        return False
    rec = pending[0]
    call_id = str(rec.get("call_id") or "").strip()
    hashes = _conductor_hash_set(run)
    gw = str(rec.get("session_hash") or "").strip() or next(iter(hashes), "")
    answers = chat_answers_for_pending_card(entries, rec)
    if not call_id or not gw or not answers:
        return False
    resp = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{gw}/answer",
        json={"callId": call_id, "answers": answers},
        headers=gateway_principal_headers(g.user_email),
        timeout=15,
    )
    if resp is None or resp.status_code >= 400:
        log.warning(
            "sdlc: could not unlock Chat ask for %s (%s)",
            run.get("run_id"),
            None if resp is None else resp.status_code,
        )
        return False
    return True


def _unlock_conductor_chat(run: dict, entries: list[dict], *, attempts: int = 90, delay: float = 1.0) -> bool:
    """Keep offering Dev answers until Chat's ask card appears and unlocks.

    Answering on Dev before Jarvis paints the card used to no-op, so Chat
    later showed the same questions still waiting.
    """
    for attempt in range(max(1, int(attempts))):
        if _answer_conductor_ask(run, entries):
            return True
        if attempt + 1 >= attempts:
            break
        time.sleep(max(0.1, float(delay)))
    return False


def _spawn_conductor_prompt(run: dict, text: str, *, wait_idle: bool = False) -> None:
    """Post to Jarvis after the HTTP response — Start must not wait on Chat."""
    sid = str(run.get("conductor_session_id") or "").strip()
    if not sid or not str(text or "").strip():
        return
    sdata = store.get(sid) or {}
    agent = str(sdata.get("agent") or "jarvis")
    owner = str(getattr(g, "user_email", "") or "")
    from threading import Thread

    from sharedagents_web.server import app

    def _go() -> None:
        try:
            with app.app_context():
                g.user_email = owner
                if wait_idle:
                    deadline = time.monotonic() + 90
                    while time.monotonic() < deadline:
                        if not _session_live(sid) and not _conductor_pending_asks(run):
                            break
                        time.sleep(1.0)
                    if _session_live(sid) or _conductor_pending_asks(run):
                        log.info(
                            "sdlc: skip conductor prompt for %s — still live or asking",
                            sid,
                        )
                        return
                _prompt_conductor(run, text, agent=agent)
        except Exception:  # noqa: BLE001
            log.exception("sdlc: conductor prompt thread failed for %s", run.get("run_id"))

    Thread(target=_go, name="sdlc-conductor", daemon=True).start()


def _conductor_jsonl_newest(run: dict, limit: int = 400) -> list[dict]:
    """Newest conductor session events from its single JSONL."""
    from sharedagents_session.session_reader import reverse_jsonl
    from sharedagents_web.server import GW_SESSIONS_DIR

    sid = str(run.get("conductor_session_id") or "").strip()
    if not sid:
        return []
    paths = [Path(GW_SESSIONS_DIR) / f"{sid}.jsonl"]
    newest: list[dict] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            for event, _start, _end in reverse_jsonl(path, max_bytes=2_000_000):
                if isinstance(event, dict):
                    newest.append(event)
                if len(newest) >= limit:
                    break
        except Exception:  # noqa: BLE001
            continue
        if newest:
            break
    return newest


def _latest_session_ask_questions(run: dict) -> list:
    """Newest unanswered question-tool payload on the conductor JSONL."""
    from sharedagents_web.dev_clarify import latest_ask_questions_from_newest

    return latest_ask_questions_from_newest(_conductor_jsonl_newest(run))


def _conductor_pending_ask_card(run: dict) -> tuple[dict | None, bool]:
    """``(card, known)`` — approvals first, then the JSONL.

    ``known`` is False when the gateway could not say. Collapsing that into
    "nothing is pending" is what let the client retire a LIVE question card on
    a transient hiccup, leaving the operator with a card they could not answer
    and no way to tell why.
    """
    from sharedagents_web.dev_clarify import open_ask_card_from_newest, pending_ask_card

    pending = _conductor_pending_asks(run)
    known = pending is not None
    rows = pending or []
    # The JSONL fallback covers UNKNOWN, never known-none. Once the gateway has
    # positively said nothing is pending -- orphans already dropped -- digging
    # a question out of history resurrects a dead one: the card renders, the
    # operator answers it, and the submit can only 404 because the turn that
    # would have received the answer is over.
    history = open_ask_card_from_newest(_conductor_jsonl_newest(run)) if not known else None
    if known and not rows:
        return None, True
    return pending_ask_card(
        rows[0] if rows else None,
        history,
        session_id=str(run.get("conductor_session_id") or ""),
    ), known


def _session_questions_for_run(run: dict) -> list[dict]:
    """Leftover Chat card wording, if any. Dev no longer overlays this."""
    from sharedagents_web.dev_clarify import session_ask_to_dev_questions

    pending = _conductor_pending_asks(run) or []
    tool_questions = _latest_session_ask_questions(run)
    return session_ask_to_dev_questions(
        pending[0] if pending else None,
        tool_questions,
    )


def _ensure_repo_clone(repo: str) -> str:
    """Best effort: put a local clone of *repo* on disk. Never raises.

    The pipeline needs one to cut a worktree from, and the intake interview
    needs one to read. Failing here must not block the intake -- the run will
    try again, loudly, when `ensure_worktree` runs.
    """
    from sharedagents_web.dev_sdlc import clone_repo, repo_dir

    name = str(repo or "").strip()
    if not name:
        return ""
    try:
        path = repo_dir(name)
        if (path / ".git").exists():
            return str(path)
        clone_repo(name, path)
        log.info("sdlc: cloned %s into %s for the intake interview", name, path)
        return str(path)
    except Exception as error:  # noqa: BLE001 — intake must survive a bad clone
        log.warning("sdlc: could not pre-clone %s: %s", name, error)
        return ""


def _start_conductor_interview(run: dict, *, wait_idle: bool | None = None) -> None:
    """Ask Jarvis to research, then interview via the chat question tool.

    ``wait_idle=False`` skips the "is the session busy?" wait. The stall
    watchdog passes it: it has ALREADY established that nothing has been
    written anywhere in the session tree for the grace period and that no
    question is outstanding. Re-asking `session_work_live()` there just
    re-litigates a settled question with a probe that is known to read True
    for a session parked on a dead ask -- and the spawn then gives up after
    90 seconds, silently, so a half-finished interview never resumed.
    """
    from sharedagents_web.dev_clarify import interview_prompt

    if _conductor_pending_asks(run):
        return
    sid = str(run.get("conductor_session_id") or "").strip()
    if not sid:
        return
    # The interview prompt's first instruction is "search and read the named
    # repo for features that already overlap". On a repo with no local clone
    # that is unanswerable, and the conductor spends its whole research budget
    # hunting the filesystem for a directory that is not there -- which is
    # every brand-new repo, exactly when the tab is most useful. Put the clone
    # on disk first; it is idempotent and the run needs it a stage later
    # anyway.
    repo_root = _ensure_repo_clone(str(run.get("repo") or ""))
    if repo_root:
        # Intake is the first turn on this session. Bind it before the gateway
        # materialises the physical session, otherwise its research child
        # inherits `/` and guesses `~/src` even though the clone is known.
        try:
            sdata = store.get(sid) or {}
            if str(sdata.get("workspace_cwd") or "") != repo_root:
                sdata["workspace_cwd"] = repo_root
                store._save(sdata)
        except Exception:  # noqa: BLE001 — the prompt still carries the root
            log.warning("sdlc: could not bind intake workspace for %s", sid)
    # Remember which turn the session was on BEFORE the interview is
    # dispatched. `_recover_stalled_intake` needs to tell "the conductor is
    # still thinking" from "the conductor finished and asked nothing", and the
    # liveness probe cannot answer that -- it reads False mid-turn.
    try:
        from sharedagents_web.dev_sdlc import get_sdlc_store

        before_turn = _sdlc_final_message(sid)[0] or "__no_completed_turn__"
        get_sdlc_store().patch(
            str(run.get("run_id") or ""),
            stage_dispatch_turn=before_turn,
        )
    except Exception:  # noqa: BLE001 — a missing marker only costs the watchdog
        log.debug("sdlc: could not stamp the intake turn for %s", run.get("run_id"))
    prompt_run = {**run, "repo_root": repo_root}
    _spawn_conductor_prompt(
        run,
        interview_prompt(prompt_run),
        wait_idle=_session_live(sid) if wait_idle is None else bool(wait_idle),
    )


#: Runs whose intake fallback is already in flight. The Dev tab polls every
#: two seconds and the fallback is a model call, so without this every poll
#: would stack another one.
_INTAKE_RECOVERING: set[str] = set()
_INTAKE_RECOVER_LOCK = Lock()


def _recover_stalled_intake(run: dict) -> dict:
    """Kick the intake fallback off the request thread; never block the poll.

    Deriving the questions costs a model round trip. This runs on the 2s poll
    route, so it must hand the work to a thread and answer immediately -- the
    next poll picks up whatever the thread has written.
    """
    run_id = str(run.get("run_id") or "")
    if not run_id or not _intake_looks_stalled(run):
        return run
    with _INTAKE_RECOVER_LOCK:
        if run_id in _INTAKE_RECOVERING:
            return run
        _INTAKE_RECOVERING.add(run_id)

    # The rescue talks to the gateway, and every gateway call signs itself with
    # `g.user_email`. A bare thread has no Flask context, so this died on
    # "Working outside of application context" on every single poll -- silently,
    # because the handler logged it and the poll answered 200 regardless. The
    # owner is captured HERE, on the request thread, while there is still a `g`.
    owner = str(getattr(g, "user_email", "") or "")
    from sharedagents_web.server import app

    def _go() -> None:
        try:
            with app.app_context():
                g.user_email = owner
                _intake_fallback_questions(run)
        except Exception:  # noqa: BLE001 — a failed rescue must not kill the poll
            log.exception("sdlc: intake fallback crashed for %s", run_id)
        finally:
            with _INTAKE_RECOVER_LOCK:
                _INTAKE_RECOVERING.discard(run_id)

    Thread(target=_go, name=f"sdlc-intake-rescue-{run_id}", daemon=True).start()
    return run


def _intake_looks_stalled(run: dict) -> bool:
    """Cheap, non-blocking test: is the interview stuck with nobody asking?

    Two shapes, and the second used to be invisible:
      * the conductor never asked anything at all; and
      * it asked a few, then its turn died -- a restart, a killed child, an
        upstream error -- leaving an INCOMPLETE interview with no pending
        question and nothing to restart it. The run then sits at two of five
        rounds forever, and the operator (who is supposed to be free after
        ratification) becomes the only thing that could notice.
    """
    from sharedagents_web.dev_clarify import answered_rounds

    if str(run.get("status") or "") != "intake" or run.get("issue_number"):
        return False
    if run.get("questions"):
        return False
    sid = str(run.get("conductor_session_id") or "").strip()
    if not sid:
        return False
    # File mtimes do not move while a hosted model is computing a response.
    # The session index does: never launch a competing fallback while the
    # authoritative conductor subtree is still live.
    if _session_live(sid):
        return False
    # A turn must have ENDED since the interview was dispatched. The liveness
    # probe cannot answer this -- it reads False mid-turn -- so compare the
    # last completed turn's identity with the one recorded at dispatch.
    #
    # That comparison only works for a conductor that has ASKED NOTHING. Once
    # an interview is under way the stamp and the last turn legitimately
    # coincide (and `turn_ended` often carries no physical turn id, so the
    # identity degrades to a timestamp that does not move). For a PARTIAL
    # interview the honest signal is different: the whole session tree has gone
    # quiet and no question is outstanding, so nobody is going to ask the next
    # one. `_conductor_tree_busy` already enforces the mid-turn grace period.
    ended_without_answer = False
    if not answered_rounds(run.get("transcript")):
        stamp = str(run.get("stage_dispatch_turn") or "")
        identity = _sdlc_final_message(sid)[0]
        # Only a guard when there IS something to compare against. The intake
        # dispatch records no stamp, so `stamp == ""` matched the empty
        # identity of a `turn_ended` that carries no physical turn id, and the
        # predicate returned False forever. Observed live on a real repo: the
        # conductor spent its turn EXPLORING the codebase, ended without asking
        # anything, and the run sat in `intake` with nobody able to notice --
        # the exact opposite of the recovery this function exists to provide.
        if stamp and (not identity or identity == stamp):
            return False
        ended_without_answer = bool(stamp and identity and identity != stamp)
        # Without a stamp, "is anyone still working?" is the honest question,
        # and `_conductor_tree_busy` below answers it with a quiet period.
    # A conductor that DELEGATES its research ends its own turn immediately and
    # is woken again when the child reports. `turn_ended` therefore does not
    # mean the interview is over, and firing here cut a working interview off
    # mid-research. Nothing may be moving anywhere in this session's tree.
    if _conductor_tree_busy(sid, include_self=False) if ended_without_answer else _conductor_tree_busy(sid):
        return False
    # `None` means the gateway could not say, which is not "none pending".
    return _conductor_pending_asks(run) == []


#: How long ratification may hold the intake claim. The critical section is
#: bounded by the forge: `create_issue` mints up to three labels (two requests
#: each) plus the issue itself, every request on `forge_client._request`'s
#: 12s timeout, so under 100s worst case; the lease is three times that and is
#: renewed once the forge answers. A worker that dies inside the section
#: therefore blocks synthesis for at most this long before stall recovery
#: continues. A claim shorter than the forge's worst case let another worker
#: take the still-unissued run and cancel the conductor mid-mint.
RATIFY_CLAIM_SECONDS = 300.0

#: How long the conductor's session tree must be silent before the interview
#: counts as abandoned rather than thinking.
INTAKE_QUIET_SECONDS = 90.0
# Contract freeze is a much shorter barrier than abandoned-turn recovery. It
# only needs to prove that the just-produced ready draft has settled and no
# late question is still being written into the session tree.
RATIFY_QUIET_SECONDS = 5.0


def _conductor_tree_busy(
    sid: str,
    *,
    quiet: float = INTAKE_QUIET_SECONDS,
    include_self: bool = True,
) -> bool:
    """Is anything still writing to this conductor's session or its children?

    Cheap: one directory scan, and only the files touched inside the window are
    opened -- and only their header line, which carries ``parent_session_id``.
    """
    from sharedagents_web.server import GW_SESSIONS_DIR

    cutoff = time.time() - max(0.0, quiet)
    try:
        entries = list(os.scandir(GW_SESSIONS_DIR))
    except OSError:
        return False
    for entry in entries:
        if not entry.name.endswith(".jsonl"):
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        if include_self and entry.name == f"{sid}.jsonl":
            return True
        try:
            with open(entry.path, encoding="utf-8", errors="replace") as handle:
                head = handle.readline()
        except OSError:
            continue
        # Session files prefix each line with a short id; the header JSON is
        # whatever follows the first space.
        _, _, raw = head.partition(" ")
        try:
            header = json.loads(raw or head)
        except ValueError:
            continue
        if str(header.get("parent_session_id") or "") == sid:
            return True
    return False


def _cancel_conductor_for_synthesis(run: dict, owner: str) -> tuple[bool, bool]:
    """Stop the conductor's turn so a server-side synthesis cannot race its next card.

    Returns ``(claimed, cancelled)``. Every pipeline stage is dispatched as a
    goal step on this same conductor session, so a force cancel that lands
    just after ratification kills the stage that was just started -- and a
    re-read before the cancel only narrows that window. The intake claim
    closes it: ratify holds the claim while it mints the issue, and the claim
    cannot be taken at all once the issue exists. ``claimed=False`` therefore
    means "the contract is being, or has been, frozen; leave the conductor
    alone and let ratification own the outcome".
    """
    from sharedagents_web.dev_sdlc import get_sdlc_store

    store = get_sdlc_store()
    run_id = str(run.get("run_id") or "")
    sid = str(run.get("conductor_session_id") or "").strip()
    if not run_id or not sid:
        return False, False
    token = secrets.token_hex(8)
    if not store.claim_intake(run_id, token):
        log.info("sdlc: %s is being ratified; leaving the conductor alone", run_id)
        return False, False
    try:
        try:
            resp = gateway_request(
                "POST",
                f"/control/v1/orchestrator/sessions/{sid}/cancel",
                json={"force": True},
                headers=gateway_principal_headers(owner),
                timeout=15,
            )
        except Exception:  # noqa: BLE001 — synthesis remains fail closed
            log.warning("sdlc: could not stop completed intake %s", sid)
            return True, False
        # A session the gateway no longer has (404) has no turn to race; any
        # other refusal leaves a possibly live conductor, and the caller must
        # not publish beside it.
        cancelled = bool(resp is not None and (resp.status_code < 400 or resp.status_code == 404))
        return True, cancelled
    finally:
        store.release_intake(run_id, token)


def _intake_fallback_questions(run: dict) -> dict:
    """Keep the intake alive when the conductor never asked anything.

    The Dev tab asks its questions through the chat question tool, so a
    conductor that ends its interview turn without calling that tool leaves the
    gate with no pending ask and no questions -- and the page waits on
    "Working out what I still need to know…" for as long as it is left open.
    Nothing times out, nothing is logged, and the run cannot be ratified.

    It is not hypothetical: a model that spends its whole output budget on
    reasoning emits no tool call at all, which is exactly what a local 27B does
    on this prompt. ``clarify()`` already derives deterministic gap questions
    for precisely this case -- ``_sdlc_clarify_round`` simply discards them
    because the happy path expects the chat card. So ask them here instead of
    stranding the run.

    Only fires once a turn has ENDED since the interview was dispatched, and
    only when the gateway positively reports no pending asks (``None`` means
    unknown, which is not the same as none).
    """
    from sharedagents_web.dev_sdlc import SdlcError, clarify, get_sdlc_store

    # Re-read: the conductor may have asked between the poll and this thread.
    fresh = get_sdlc_store().get(str(run.get("run_id") or "")) or run
    if not _intake_looks_stalled(fresh):
        return fresh
    run = fresh
    # Synthesize once when the conductor is idle, using ALL settled answers.
    # Re-dispatching the entire interview repeated research and forced filler
    # questions even when the request was already complete.
    from sharedagents_web.dev_clarify import MIN_CLARIFY_ROUNDS, answered_rounds

    transcript = list(run.get("transcript") or [])
    try:
        result = clarify(
            headline=str(run.get("headline") or ""),
            detail=str(run.get("detail") or ""),
            kind=str(run.get("kind") or "feature"),
            repo=str(run.get("repo") or ""),
            transcript=transcript,
        )
    except SdlcError as error:
        log.warning("sdlc: intake fallback failed for %s: %s", run.get("run_id"), error)
        return run
    # The fallback call may take minutes. Chat can ask and receive an answer
    # while it is in flight, so its pre-call stall decision is no longer
    # authority to write. Re-read and prove the same gap still exists.
    fresh = get_sdlc_store().get(str(run.get("run_id") or "")) or run
    if (
        list(fresh.get("transcript") or []) != transcript
        or fresh.get("detail") != run.get("detail")
        or fresh.get("issue_number")
        or not _intake_looks_stalled(fresh)
    ):
        log.info("sdlc: stale intake fallback discarded for %s", run.get("run_id"))
        return fresh
    run = fresh
    questions = list(result.get("questions") or [])
    ready = bool(result.get("ready")) and answered_rounds(transcript) >= MIN_CLARIFY_ROUNDS
    if ready:
        questions = []
    elif not questions:
        from sharedagents_web.dev_clarify import depth_followup_question

        questions = [
            depth_followup_question(
                transcript,
                headline=str(run.get("headline") or ""),
                detail=str(run.get("detail") or ""),
            )
        ]
    # This is an ownership handoff, not a second interviewer. Publish the
    # server question only after Chat can no longer wake up and race it --
    # and only while nobody is freezing the contract (the claim).
    claimed, cancelled = _cancel_conductor_for_synthesis(
        run,
        str(getattr(g, "user_email", "") or ""),
    )
    if not claimed:
        return get_sdlc_store().get(str(run["run_id"])) or run
    if not cancelled:
        log.warning("sdlc: fallback ownership handoff failed for %s", run.get("run_id"))
        return run
    log.info(
        "sdlc: conductor asked nothing for %s; falling back to %d gap question(s)", run.get("run_id"), len(questions)
    )
    # Conditional on the transcript this synthesis answered and on the
    # interview still being open: a newer answer or a ratification that
    # landed after the re-read above makes this write a no-op.
    updated = get_sdlc_store().patch_intake(
        str(run["run_id"]),
        expect_transcript=transcript,
        # One answer must be received before the next question is chosen.
        questions=questions[:1],
        requirements=result.get("requirements") or run.get("requirements") or [],
        criteria=result.get("criteria") or run.get("criteria") or [],
        summary=str(result.get("summary") or run.get("summary") or ""),
        status="ready" if ready else "intake",
        last_error="",
    )
    if updated is None:
        log.info("sdlc: stale intake fallback discarded for %s", run.get("run_id"))
        return get_sdlc_store().get(str(run["run_id"])) or run
    return updated


def _submit_prompt(
    sid: str,
    prompt: str,
    *,
    agent: str,
    initial_model: str = "",
) -> tuple[dict | None, int]:
    """Submit against the session's single durable id.

    ``initial_agent`` is what lets the gateway bootstrap a session that has no
    transcript yet.  A Dev run mints its conductor in the WEB store, which
    writes ``sessions.db`` and no ``<sid>.jsonl`` -- so the gateway's prompt
    route, which keys on that file, answered ``404 session not found`` to the
    FIRST prompt of every run.  The intake then waited forever on a clarifier
    that was never asked anything.  The chat route has always sent this
    (``routes/sessions.py``); the Dev tab did not, so it only ever worked on a
    conductor that had already taken a turn some other way.
    """
    body: dict = {
        "prompt": prompt,
        "frontend": "web",
        "initial_agent": agent or "jarvis",
        "workspace_origin": "web",
    }
    if initial_model:
        # The prompt endpoint creates an as-yet-unmaterialized session with
        # workspace_cwd and initial_model in one operation.  Running /model
        # first would lazily create the session at `/`, permanently detaching
        # its tool runner and LSP root from the isolated Dev worktree.
        body["initial_model"] = str(initial_model)
    sdata = store.get(sid) or {}
    if sdata.get("workspace_cwd"):
        body["workspace_cwd"] = str(sdata.get("workspace_cwd") or "")
    resp = gateway_request(
        "POST",
        f"/control/v1/orchestrator/sessions/{sid}/prompt",
        json=body,
        headers=gateway_principal_headers(g.user_email),
        timeout=30,
    )
    if resp is None:
        return {"error": "Gateway unreachable"}, 502
    try:
        payload = resp.json()
    except ValueError:
        payload = {"error": "bad gateway response"}
    return payload if isinstance(payload, dict) else {"ok": True}, resp.status_code


def _ensure_issue(row: dict) -> dict:
    """Mint or reuse one forge issue. Returns the row; caller fail-closes if unbound."""
    if has_issue(row):
        return row
    if not forge_client.configured():
        return row
    repo = str(row.get("repo") or "").strip()
    if not repo:
        return row
    title = str(row.get("object") or "").strip()
    kind = str(row.get("kind") or "feature")
    labels = ["type/bug" if kind == "fix" else "type/feature", "status/in-progress"]
    body = (
        f"## Done when\n{row.get('done_when') or ''}\n\n"
        f"## Tracker\nDEV_ID: {row.get('id')}\n"
        f"Branch: `{row.get('branch')}`\n"
    )
    try:
        _st, found = forge_client.find_open_issue(repo, title=title)
        issue = found if found.get("number") else {}
        if not issue:
            _st, issue = forge_client.create_issue(
                repo,
                title=title,
                body=body,
                labels=labels,
            )
        number = issue.get("number")
        url = str(issue.get("html_url") or "")
        if not number:
            return row
        patched = get_dev_registry().patch(
            str(row["id"]),
            issue_number=int(number),
            issue_url=url or forge_client.issue_html_url(repo, number),
        )
        return patched or row
    except Exception:  # noqa: BLE001 — caller returns issue_required
        log.warning("dev issue mint failed for %s", row.get("id"))
        return row


def _reject_without_issue(row: dict, *, fresh: bool):
    if has_issue(row):
        return None
    if fresh:
        try:
            get_dev_registry().discard(str(row["id"]))
        except Exception:  # noqa: BLE001
            log.warning("dev discard failed for %s", row.get("id"))
    return jsonify(
        {
            "error": "issue_required",
            "detail": "A Dev loop is not created without a forge issue.",
        }
    ), 503


def _web_loop(row: dict, *, session_live: bool | None = None) -> dict:
    projected = _overlay_forge(row)
    live = bool(session_live) if session_live is not None else _session_live(str(row.get("session_id") or ""))
    return project_loop(
        projected,
        session_live=live,
        forge_configured=projected.get("forge_configured"),
    )


@bp.route("/dev")
@bp.route("/dev/<loop_id>")
@require_auth
def dev_page(loop_id=None):
    from sharedagents_web import dev_sdlc

    return render_template(
        "dev.html",
        base=BASE_PATH,
        realtime_url=REALTIME_URL,
        initial_loop_id=loop_id or "",
        forge_configured=forge_client.configured(),
        # The lifecycle figure's legend is rendered from the same table the
        # driver walks, so a phase cannot be drawn as covered without a gate.
        phases=dev_sdlc.PHASES,
        unimplemented_phase=dev_sdlc.UNIMPLEMENTED_PHASE,
        vee_pairs=dev_sdlc.VEE_PAIRS,
    )


@bp.route("/api/dev/status")
@require_auth
def api_dev_status():
    return jsonify(
        {
            "forge_configured": forge_client.configured(),
            "forge_kind": forge_client.kind(),
            "forge_url": forge_client.forge_url(),
            "error": forge_client.configured_reason(),
        }
    )


@bp.route("/api/dev/repos")
@require_auth
def api_dev_repos():
    """List every repo the forge token can see (user + orgs)."""
    if not forge_client.configured():
        return jsonify(
            {
                "repos": [],
                "forge_configured": False,
                "error": forge_client.configured_reason(),
            }
        )
    status, repos = forge_client.list_accessible_repos()
    error = ""
    if status != 200:
        error = f"The forge listed no repos (HTTP {status})"
    elif not repos:
        error = "The forge returned an empty repo list for this token"
    return jsonify(
        {
            "repos": repos,
            "forge_configured": True,
            "forge_status": status,
            "forge_kind": forge_client.kind(),
            "forge_url": forge_client.forge_url(),
            "error": error,
        }
    )


@bp.route("/api/dev/issues")
@require_auth
def api_dev_issues():
    """List a repo's issues and flag which already have a Dev session."""
    try:
        repo = _require_repo(request.args.get("repo") or "")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    raw_state = request.args.get("state")
    state = str(raw_state or "open").strip().lower()
    if state not in {"open", "closed", "all"}:
        state = "open"
    filt = str(request.args.get("session") or "all").strip().lower()
    if filt not in {"all", "bound", "unbound"}:
        filt = "all"
    limit = forge_limit_arg(request.args.get("limit"))
    if not forge_client.configured():
        return jsonify(
            {
                "repo": repo,
                "issues": [],
                "total": 0,
                "limit": limit,
                "forge_configured": False,
                "state_filter": state,
                "session_filter": filt,
            }
        )
    status, issues = forge_client.list_issues(repo, state=state)
    tagged = _issues_with_sessions(repo, issues)
    if filt == "bound":
        tagged = [i for i in tagged if i.get("has_session")]
    elif filt == "unbound":
        tagged = [i for i in tagged if not i.get("has_session")]
    extra = []
    if raw_state is None and state == "open":
        _st, closed = forge_client.list_issues(repo, state="closed")
        extra = _issues_with_sessions(repo, closed)
        if filt == "bound":
            extra = [i for i in extra if i.get("has_session")]
        elif filt == "unbound":
            extra = [i for i in extra if not i.get("has_session")]
        shown, total = fill_forge_panel(tagged, extra, limit=limit)
    else:
        shown, total = cap_forge_panel(tagged, limit=limit)
    return jsonify(
        {
            "repo": repo,
            "forge_configured": True,
            "forge_status": status,
            "state_filter": state,
            "session_filter": filt,
            "issues": shown,
            "total": total,
            "limit": limit,
        }
    )


@bp.route("/api/dev/issues/<int:number>/close", methods=["POST"])
@require_auth
def api_dev_close_issue(number: int):
    """Close one forge issue. Does not delete a Dev run or merge a PR."""
    data = _optional_json_object()
    if data is None:
        return jsonify({"error": "invalid_json", "number": number}), 400
    try:
        repo = _require_repo(str(data.get("repo") or request.args.get("repo") or ""))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not forge_client.configured():
        return jsonify({"error": "forge_not_configured", "repo": repo}), 503
    lookup_status, lookup = forge_client.get_issue(repo, number)
    if lookup_status == 400 and isinstance(lookup, dict) and lookup.get("error") == "not_an_issue":
        return jsonify(
            {
                "error": "number refers to a pull request, not an issue",
                "forge_status": lookup_status,
                "repo": repo,
                "number": number,
            }
        ), 400
    if lookup_status != 200:
        return jsonify(
            {
                "error": (lookup.get("error") or lookup.get("message") if isinstance(lookup, dict) else None)
                or "issue lookup failed",
                "forge_status": lookup_status,
                "repo": repo,
                "number": number,
            }
        ), 502
    status, payload = forge_client.close_issue(repo, number)
    body = payload if isinstance(payload, dict) else {}
    if status not in (200, 201):
        return jsonify(
            {
                "error": body.get("error") or body.get("message") or "close failed",
                "forge_status": status,
                "repo": repo,
                "number": number,
            }
        ), 502
    return jsonify(
        {
            "ok": True,
            "repo": repo,
            "number": number,
            "state": body.get("state") or "closed",
            "issue": body,
        }
    )


@bp.route("/api/dev/pulls")
@require_auth
def api_dev_pulls():
    try:
        repo = _require_repo(request.args.get("repo") or "")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    raw_state = request.args.get("state")
    state = str(raw_state or "open").strip().lower()
    if state not in {"open", "closed", "all"}:
        state = "open"
    limit = forge_limit_arg(request.args.get("limit"))
    if not forge_client.configured():
        return jsonify(
            {
                "repo": repo,
                "pulls": [],
                "total": 0,
                "limit": limit,
                "forge_configured": False,
                "state_filter": state,
            }
        )
    status, pulls = forge_client.list_pulls(repo, state=state)
    if raw_state is None and state == "open":
        _st, closed = forge_client.list_pulls(repo, state="closed")
        shown, total = fill_forge_panel(pulls, closed, limit=limit)
    else:
        shown, total = cap_forge_panel(pulls, limit=limit)
    return jsonify(
        {
            "repo": repo,
            "forge_configured": True,
            "forge_status": status,
            "state_filter": state,
            "pulls": shown,
            "total": total,
            "limit": limit,
        }
    )


@bp.route("/api/dev/loops", methods=["GET"])
@require_auth
def api_dev_list():
    include_done = str(request.args.get("all") or "").lower() in {"1", "true", "yes"}
    repo_filter = str(request.args.get("repo") or "").strip()
    rows = get_dev_registry().list_loops(include_done=include_done)
    if repo_filter:
        rows = [r for r in rows if str(r.get("repo") or "") == repo_filter]
    loops = [_web_loop(row) for row in rows]
    return jsonify({"loops": loops, "forge_configured": forge_client.configured()})


@bp.route("/api/dev/loops/<loop_id>", methods=["GET"])
@require_auth
def api_dev_get(loop_id):
    row = get_dev_registry().get(loop_id)
    if row is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify(_web_loop(row))


@bp.route("/api/dev/loops", methods=["POST"])
@require_auth
def api_dev_create():
    """Start (or resume) a closed loop. Same repo+branch reuses the row.

    Object+done_when, one branch, no merge, no Tasks.
    An existing forge issue (created outside sharedagents) can be bound via
    ``issue_number``; object/done_when then default from that issue.
    """
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    try:
        repo = _require_repo(str(data.get("repo") or ""))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    issue_number_raw = data.get("issue_number") or data.get("issue")
    bound_issue: dict = {}
    if issue_number_raw not in (None, ""):
        if not forge_client.configured():
            return jsonify(
                {
                    "error": "issue_required",
                    "detail": "A Dev loop is not created without a forge issue.",
                }
            ), 503
        st, bound_issue = forge_client.get_issue(repo, issue_number_raw)
        if st == 400 and bound_issue.get("error") == "not_an_issue":
            return jsonify({"error": "not_an_issue"}), 400
        if st != 200 or not bound_issue.get("number"):
            return jsonify(
                {
                    "error": "issue_not_found",
                    "detail": f"Issue {issue_number_raw} was not found in {repo}.",
                }
            ), 404

    object_name = str(data.get("object") or data.get("object_name") or "").strip()
    done_when = str(data.get("done_when") or "").strip()
    if bound_issue:
        object_name = object_name or str(bound_issue.get("title") or "").strip()
        done_when = done_when or str(bound_issue.get("body") or "").strip()
        if not done_when:
            done_when = f"Close issue #{bound_issue.get('number')} as specified; one PR against main; do not merge."
    if not object_name:
        return jsonify({"error": "object is required"}), 400
    if not done_when:
        return jsonify({"error": "done_when is required"}), 400
    kind = str(data.get("kind") or "").strip().lower()
    if kind not in ("fix", "feature"):
        kind = forge_client.kind_from_labels(bound_issue.get("labels") or []) if bound_issue else "feature"
    branch = str(data.get("branch") or "").strip() or branch_name(kind, object_name)
    agent = str(data.get("agent") or "build").strip() or "build"
    from sharedagents_web.agent_visibility import conversational_agent_or_default

    agent = conversational_agent_or_default(agent, DEFAULT_AGENT)

    existing = None
    if bound_issue.get("number"):
        existing = get_dev_registry().get_by_issue(repo, int(bound_issue["number"]))
        if existing and existing.get("branch"):
            branch = str(existing["branch"])
    if existing is None:
        existing = get_dev_registry().get_by_branch(repo, branch)
        if (
            existing
            and bound_issue.get("number")
            and existing.get("issue_number")
            and int(existing["issue_number"] or 0) != int(bound_issue["number"])
        ):
            return jsonify(
                {
                    "error": "branch_in_use",
                    "detail": (f"`{branch}` is already bound to issue #{existing.get('issue_number')}."),
                    "loop": _web_loop(existing),
                }
            ), 409

    pr_url = (existing or {}).get("pr_url") or ""
    fresh = existing is None
    row = get_dev_registry().upsert(
        object_name=object_name,
        done_when=done_when,
        repo=repo,
        branch=branch,
        kind=kind,
        agent=agent,
        session_id=str((existing or {}).get("session_id") or ""),
    )
    if bound_issue.get("number") and not has_issue(row):
        patched = get_dev_registry().patch(
            str(row["id"]),
            issue_number=int(bound_issue["number"]),
            issue_url=str(bound_issue.get("html_url") or "")
            or forge_client.issue_html_url(repo, bound_issue["number"]),
        )
        row = patched or row
    else:
        row = _ensure_issue(row)
    rejected = _reject_without_issue(row, fresh=fresh)
    if rejected is not None:
        return rejected

    if existing and existing.get("state") not in ("merged", "aborted") and not data.get("force"):
        sid = str(row.get("session_id") or "")
        if sid and _session_live(sid):
            return jsonify(
                {
                    "loop": _web_loop(row),
                    "resumed": True,
                    "note": "same branch already running — inner lap in flight",
                }
            ), 200

    sdata = store.create(g.user_email, agent, "", "dev")
    sid = sdata["id"]
    row = get_dev_registry().patch(str(row["id"]), session_id=sid) or row
    prompt = closed_loop_prompt(
        object_name=object_name,
        done_when=done_when,
        repo=repo,
        branch=branch,
        kind=kind,
        pr_url=pr_url,
        loop_id=str(row["id"]),
        issue_number=str(row.get("issue_number") or ""),
        issue_url=str(row.get("issue_url") or ""),
    )
    payload, http = _submit_prompt(sid, prompt, agent=agent)
    get_dev_registry().bump_lap(row["id"])
    if http >= 400:
        get_dev_registry().patch(
            row["id"],
            state="blocked",
            last_error=str((payload or {}).get("error") or f"prompt_{http}"),
        )
    row = get_dev_registry().get(row["id"]) or row
    out = _web_loop(row, session_live=http < 400)
    out["prompt_result"] = payload
    return jsonify({"loop": out, "resumed": bool(existing)}), (201 if not existing else 200)


@bp.route("/api/dev/loops/<loop_id>/continue", methods=["POST"])
@require_auth
def api_dev_continue(loop_id):
    """Inner lap on the same branch. Never opens a new PR by itself.

    Continue is another inner lap, not a new loop.
    """
    row = get_dev_registry().get(loop_id)
    if row is None:
        return jsonify({"error": "not_found"}), 404
    if row.get("state") in ("merged", "aborted"):
        return jsonify({"error": "loop_closed", "state": row["state"]}), 409
    row = _ensure_issue(row)
    rejected = _reject_without_issue(row, fresh=False)
    if rejected is not None:
        return rejected
    data = request.get_json(silent=True) or {}
    reason = str((data or {}).get("reason") or "").strip() or "operator continue"
    sid = str(row.get("session_id") or "")
    if not sid:
        sdata = store.create(g.user_email, row.get("agent") or "build", "", "dev")
        sid = sdata["id"]
        get_dev_registry().patch(loop_id, session_id=sid)
    if _session_live(sid):
        return jsonify({"ok": True, "skipped": "session_live", "loop": _web_loop(row, session_live=True)}), 200
    prompt = continue_prompt(
        object_name=row["object"],
        done_when=row["done_when"],
        branch=row["branch"],
        reason=reason,
        kind=str(row.get("kind") or "feature"),
        loop_id=str(row.get("id") or loop_id),
        issue_number=str(row.get("issue_number") or ""),
        issue_url=str(row.get("issue_url") or ""),
    )
    payload, http = _submit_prompt(sid, prompt, agent=str(row.get("agent") or "build"))
    get_dev_registry().bump_lap(loop_id)
    from datetime import datetime, timezone

    get_dev_registry().patch(
        loop_id,
        last_continue_at=datetime.now(timezone.utc).isoformat(),
        last_error="" if http < 400 else str((payload or {}).get("error") or http),
        state="inner" if http < 400 else "blocked",
    )
    row = get_dev_registry().get(loop_id) or row
    out = _web_loop(row, session_live=http < 400)
    out["prompt_result"] = payload
    return jsonify(
        {
            "ok": http < 400,
            "loop": out,
            "prompt_result": payload,
        }
    ), (200 if http < 400 else http)


@bp.route("/api/dev/loops/<loop_id>/abort", methods=["POST"])
@require_auth
def api_dev_abort(loop_id):
    row = get_dev_registry().patch(loop_id, state="aborted")
    if row is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"loop": _web_loop(row)})


# ---------------------------------------------------------------------------
# Deterministic SDLC flow control plane.  These additive routes never merge
# or replace a forge Dev loop; they only persist server-owned flow evidence.
# ---------------------------------------------------------------------------


def _flow_error(error: Exception):
    if isinstance(error, PermissionError):
        return jsonify({"error": str(error)}), 403
    if isinstance(error, FlowConflict):
        return jsonify({"error": str(error)}), 409
    if isinstance(error, FlowError):
        return jsonify({"error": str(error)}), 400
    log.exception("dev flow request failed")
    return jsonify({"error": "flow request failed"}), 500


@bp.route("/api/dev/flows", methods=["GET"])
@require_auth
def api_dev_flows():
    registry = get_flow_registry()
    return jsonify({"flows": registry.list_flows(request.args.get("loop_id") or None)})


@bp.route("/api/dev/flows", methods=["POST"])
@require_auth
def api_dev_flow_create():
    payload = request.get_json(silent=True) or {}
    try:
        actor = _flow_actor(payload, default_role="human", human=True)
        loop_id = str(payload.get("loop_id") or "").strip()
        loop = get_dev_registry().get(loop_id)
        if not loop or not loop.get("issue_number") or not loop.get("issue_url"):
            return jsonify({"error": "issue-bound Dev loop is required"}), 400
        flow = get_flow_registry().create_flow(
            loop_id=loop_id,
            repo=str(loop.get("repo") or ""),
            branch=str(loop.get("branch") or ""),
            issue_number=loop.get("issue_number"),
            issue_url=str(loop.get("issue_url") or ""),
            requirements=str(payload.get("requirements") or ""),
            flowchart=payload.get("flowchart") or {},
            criteria=payload.get("criteria") or [],
            actor=actor,
            aal=str(payload.get("aal") or "C"),
            paac=payload.get("paac"),
        )
        return jsonify({"flow": _flow_detail(get_flow_registry(), flow["flow_id"])}), 201
    except Exception as error:  # noqa: BLE001
        return _flow_error(error)


@bp.route("/api/dev/flows/<flow_id>", methods=["GET"])
@require_auth
def api_dev_flow_detail(flow_id: str):
    detail = _flow_detail(get_flow_registry(), flow_id)
    if detail is None:
        return jsonify({"error": "flow not found"}), 404
    return jsonify({"flow": detail})


def _flow_payload() -> tuple[dict, int] | tuple[None, None]:
    payload = request.get_json(silent=True) or {}
    try:
        expected = int(payload.get("expected_version"))
    except (TypeError, ValueError):
        return None, None
    return payload, expected


@bp.route("/api/dev/flows/<flow_id>/advance", methods=["POST"])
@require_auth
def api_dev_flow_advance(flow_id: str):
    payload, expected = _flow_payload()
    if payload is None:
        return jsonify({"error": "expected_version is required"}), 400
    try:
        flow = get_flow_registry().advance(
            flow_id,
            expected_version=expected,
            action=str(payload.get("action") or "advance"),
            actor=_flow_actor(payload, default_role="ai"),
            evidence=payload.get("evidence"),
        )
        return jsonify({"flow": flow})
    except Exception as error:  # noqa: BLE001
        return _flow_error(error)


@bp.route("/api/dev/flows/<flow_id>/tests", methods=["POST"])
@require_auth
def api_dev_flow_tests(flow_id: str):
    payload, expected = _flow_payload()
    if payload is None:
        return jsonify({"error": "expected_version is required"}), 400
    try:
        flow = get_flow_registry().submit_tests(
            flow_id,
            expected_version=expected,
            actor=_flow_actor(payload, default_role="test_author"),
            cases=payload.get("cases") or [],
            manifest=payload.get("manifest") or {},
            run=payload.get("run") or {},
            negative_witnesses=payload.get("negative_witnesses"),
            baseline_decisions=payload.get("baseline_decisions"),
        )
        return jsonify({"flow": _flow_detail(get_flow_registry(), flow["flow_id"])})
    except Exception as error:  # noqa: BLE001
        return _flow_error(error)


@bp.route("/api/dev/flows/<flow_id>/spec", methods=["POST"])
@require_auth
def api_dev_flow_spec(flow_id: str):
    payload, expected = _flow_payload()
    if payload is None:
        return jsonify({"error": "expected_version is required"}), 400
    try:
        flow = get_flow_registry().submit_spec(
            flow_id,
            expected_version=expected,
            actor=_flow_actor(payload, default_role="spec_author"),
            spec=payload.get("spec") or {},
            adapter_binding=payload.get("adapter_binding") or {},
            consistency=payload.get("consistency") or {},
            test_hash=payload.get("test_hash"),
        )
        return jsonify({"flow": _flow_detail(get_flow_registry(), flow["flow_id"])})
    except Exception as error:  # noqa: BLE001
        return _flow_error(error)


@bp.route("/api/dev/flows/<flow_id>/implementation", methods=["POST"])
@require_auth
def api_dev_flow_implementation(flow_id: str):
    payload, expected = _flow_payload()
    if payload is None:
        return jsonify({"error": "expected_version is required"}), 400
    try:
        flow = get_flow_registry().submit_implementation(
            flow_id,
            expected_version=expected,
            actor=_flow_actor(payload, default_role="implementer"),
            code=payload.get("code") or {},
            runs=payload.get("runs") or {},
            test_hash=payload.get("test_hash"),
            rtm=payload.get("rtm"),
        )
        return jsonify({"flow": _flow_detail(get_flow_registry(), flow["flow_id"])})
    except Exception as error:  # noqa: BLE001
        return _flow_error(error)


@bp.route("/api/dev/flows/<flow_id>/disputes", methods=["POST"])
@require_auth
def api_dev_flow_dispute(flow_id: str):
    payload, expected = _flow_payload()
    if payload is None:
        return jsonify({"error": "expected_version is required"}), 400
    try:
        flow = get_flow_registry().open_dispute(
            flow_id,
            expected_version=expected,
            actor=_flow_actor(payload, default_role="reviewer"),
            reason=str(payload.get("reason") or ""),
        )
        return jsonify({"flow": _flow_detail(get_flow_registry(), flow["flow_id"])})
    except Exception as error:  # noqa: BLE001
        return _flow_error(error)


@bp.route("/api/dev/flows/<flow_id>/disputes/<dispute_id>/resolve", methods=["POST"])
@require_auth
def api_dev_flow_dispute_resolve(flow_id: str, dispute_id: str):  # noqa: ARG001
    payload, expected = _flow_payload()
    if payload is None:
        return jsonify({"error": "expected_version is required"}), 400
    try:
        flow = get_flow_registry().resolve_dispute(
            flow_id,
            expected_version=expected,
            disposition=str(payload.get("disposition") or ""),
            actor=_flow_actor(payload, default_role="human", human=True),
            new_cases=payload.get("new_cases"),
        )
        return jsonify({"flow": _flow_detail(get_flow_registry(), flow["flow_id"])})
    except Exception as error:  # noqa: BLE001
        return _flow_error(error)


@bp.route("/api/dev/flows/<flow_id>/review", methods=["POST"])
@require_auth
def api_dev_flow_review(flow_id: str):
    payload, expected = _flow_payload()
    if payload is None:
        return jsonify({"error": "expected_version is required"}), 400
    try:
        flow = get_flow_registry().submit_review(
            flow_id,
            expected_version=expected,
            actor=_flow_actor(payload, default_role="reviewer"),
            verdict=str(payload.get("verdict") or ""),
            checks=payload.get("checks") or {},
            runs=payload.get("runs") or {},
            rejection_type=payload.get("rejection_type"),
            reason=str(payload.get("reason") or ""),
        )
        return jsonify({"flow": _flow_detail(get_flow_registry(), flow["flow_id"])})
    except Exception as error:  # noqa: BLE001
        return _flow_error(error)


# ---------------------------------------------------------------------------
# SDLC control plane — the Dev tab as the only surface needed to ship.
#
# Intake questions the operator until nothing is ambiguous, the ratified answers
# become the issue, and plan -> design -> code -> tests -> deploy runs on its own,
# posting each finished stage back to that issue.  Merge and deploy stay under
# the operator's finger.
# ---------------------------------------------------------------------------

#: How long an armed deploy stays armed. Long enough to click, short enough
#: that a forgotten arm cannot be used later.
DEPLOY_ARM_SECONDS = 120


def _sdlc_final_message(web_sid: str) -> tuple[str, str, str]:
    """``(turn_identity, text, error)`` for the last COMPLETED turn of a session.

    A completed turn -- not the newest assistant message -- is what tells the
    driver the agent is finished.  An agent narrating its way through a long
    stage commits several assistant messages inside one turn; treating any of
    them as "done" would fail a stage that is still working, and the session
    liveness probe cannot be used instead: it reads ``False`` for a session
    that is demonstrably mid-turn.

    The identity is the physical turn id, so two stages ending on byte-identical
    markers are still distinguishable.  ``error`` is the turn's own failure
    string when it died upstream -- a 402, a 5xx, an abort -- because a stage
    that failed because the provider refused the request must not be reported
    as an agent that forgot to say it was done.  Falls back to the newest
    committed message only when no terminal event is on disk at all.
    """
    from sharedagents_session.session_reader import reverse_jsonl
    from sharedagents_web.server import GW_SESSIONS_DIR

    fallback: tuple[str, str, str] = ("", "", "")

    def _identity(event: dict, text: str) -> str:
        turn = str(event.get("physical_turn_id") or "").strip()
        if turn:
            return turn
        stamp = str(event.get("created_at") or event.get("ts") or "").strip()
        return stamp + ":" + str(len(text))

    candidates = [GW_SESSIONS_DIR / f"{web_sid}.jsonl"]
    for path in candidates:
        try:
            if not path.is_file():
                continue
            for event, _start, _end in reverse_jsonl(
                path,
                None,
                max_bytes=8 << 20,
                max_events=8000,
            ):
                inner = event.get("event") if isinstance(event.get("event"), dict) else event
                if inner.get("type") == "turn_ended":
                    text = str(inner.get("final_content") or "")
                    error = str(inner.get("error") or "").strip()
                    if not error and inner.get("aborted"):
                        error = "the turn was aborted"
                    classification = str(inner.get("error_classification") or "").strip()
                    if error and classification:
                        error = f"{classification}: {error}"
                    return _identity(inner, text) or "turn", text, error
                if not fallback[0] and event.get("kind") == "assistant_message_committed":
                    text = str(event.get("content") or "")
                    if text.strip():
                        fallback = (_identity(event, text), text, "")
        except (OSError, TypeError, ValueError):
            continue
    return fallback


def _digest_arg_summary(args: dict) -> str:
    """One line per tool call for the retry digest — never the payload."""
    parts: list[str] = []
    for key, value in (args or {}).items():
        text = " ".join(str(value or "").split())
        parts.append(f"{key}={text[:60]}")
        if sum(len(p) + 1 for p in parts) > 110:
            break
    return " ".join(parts)[:120]


def _digest_path_arg(args: dict) -> str:
    """The file path a read-like tool call named, for `files_read`."""
    for key, value in (args or {}).items():
        if (key == "path" or key.endswith("_path")) and str(value or "").strip():
            return " ".join(str(value).split())[:200]
    return ""


def _sdlc_session_digest(web_sid: str) -> dict:
    """P3: the last tool calls and files read from a worker session.

    Reads the same JSONL ``_sdlc_final_message`` reads, newest first, and
    reduces it to bounded one-line summaries. ``build_attempt_digest`` does
    the final bounding; this only gathers.
    """
    from sharedagents_session.session_reader import reverse_jsonl
    from sharedagents_web.server import GW_SESSIONS_DIR

    sid = str(web_sid or "").strip()
    if not sid:
        return {}
    calls: list[dict[str, str]] = []
    files: list[str] = []
    path = GW_SESSIONS_DIR / f"{sid}.jsonl"
    try:
        for event, _start, _end in reverse_jsonl(
            path,
            None,
            max_bytes=4 << 20,
            max_events=4000,
        ):
            inner = event.get("event") if isinstance(event.get("event"), dict) else event
            if inner.get("type") != "tool_call":
                continue
            args = inner.get("arguments")
            args = args if isinstance(args, dict) else {}
            calls.append(
                {
                    "name": str(inner.get("name") or "")[:60],
                    "args": _digest_arg_summary(args),
                }
            )
            found = _digest_path_arg(args)
            if found and found not in files:
                files.append(found)
            if len(calls) >= 12:
                break
    except (OSError, TypeError, ValueError):
        return {}
    return {"last_tool_calls": calls, "files_read": files}


def _sdlc_session_usage(web_sid: str) -> dict:
    """P7: token spend and tool-call count for the current worker turn."""
    from sharedagents_session.session_reader import reverse_jsonl
    from sharedagents_web.server import GW_SESSIONS_DIR

    sid = str(web_sid or "").strip()
    totals = {
        "reasoning_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_calls": 0,
    }
    if not sid:
        return totals
    path = GW_SESSIONS_DIR / f"{sid}.jsonl"
    try:
        for event, _start, _end in reverse_jsonl(
            path,
            None,
            max_bytes=4 << 20,
            max_events=4000,
        ):
            inner = event.get("event") if isinstance(event.get("event"), dict) else event
            if not isinstance(inner, dict):
                continue
            if inner.get("type") == "tool_call":
                totals["tool_calls"] += 1
            usage = inner.get("usage") if isinstance(inner.get("usage"), dict) else {}
            for src in (usage, inner):
                for key in ("reasoning_tokens", "prompt_tokens", "completion_tokens"):
                    raw = src.get(key)
                    if raw is None and key == "reasoning_tokens":
                        raw = src.get("reasoning")
                    try:
                        totals[key] = max(totals[key], int(raw or 0))
                    except (TypeError, ValueError):
                        continue
    except (OSError, TypeError, ValueError):
        return totals
    return totals


def _sdlc_last_activity_at(run: dict) -> float:
    """P2: the worker's last tool call, on the driver's monotonic clock.

    0.0 when nothing can be read — the driver then treats the stage start as
    the last activity, which fails closed toward "possibly idle" instead of
    granting a silent worker an endless allowance.
    """
    from sharedagents_session.session_reader import reverse_jsonl
    from sharedagents_web.server import GW_SESSIONS_DIR

    sid = str(run.get("author_session_id") or run.get("session_id") or "").strip()
    if not sid:
        return 0.0
    path = GW_SESSIONS_DIR / f"{sid}.jsonl"
    try:
        for event, _start, _end in reverse_jsonl(
            path,
            None,
            max_bytes=2 << 20,
            max_events=2000,
        ):
            inner = event.get("event") if isinstance(event.get("event"), dict) else event
            if inner.get("type") != "tool_call":
                continue
            stamp = str(inner.get("ts") or "").strip()
            if not stamp:
                return 0.0
            when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return time.monotonic() - (time.time() - when.timestamp())
    except (OSError, TypeError, ValueError):
        return 0.0
    return 0.0


class _Forge:
    """The forge writes the driver needs, degrading instead of raising."""

    def comment(self, repo: str, number: int, body: str):
        status, payload = forge_client.comment_issue(repo, number, body)
        if status not in (200, 201):
            return None
        try:
            return int(payload.get("id"))
        except (TypeError, ValueError):
            return None

    def label(self, repo: str, number: int, labels: list) -> bool:
        wanted = [name for name in labels if str(name or "").strip()]
        if not wanted:
            return False
        return forge_client.add_issue_labels(repo, number, wanted)

    def open_pull(self, repo: str, *, title: str, head: str, body: str) -> dict:
        status, payload = forge_client.create_pull(
            repo,
            title=title,
            head=head,
            body=body,
        )
        if status in (200, 201) and payload.get("number"):
            return payload
        # A PR already open for this head is the same PR, not a second one.
        # ``list_pulls`` projects ``head`` down to the branch NAME, not the
        # object the create endpoint returns. Reading it as a dict crashed the
        # one path that exists to avoid opening a second PR.
        _st, pulls = forge_client.list_pulls(repo, state="open")
        for pull in pulls:
            if str(pull.get("head") or "") == head:
                return pull
        return {"error": str(payload.get("message") or payload.get("error") or f"HTTP {status}")}


def _session_model(web_sid: str) -> str:
    """The model a session is currently bound to, from its own header.

    The session log's newest ``header_update`` carries the applied model id, so
    this is what the session actually uses -- not what anyone intended.
    """
    from sharedagents_session.session_reader import reverse_jsonl
    from sharedagents_web.server import GW_SESSIONS_DIR

    for path in (GW_SESSIONS_DIR / f"{web_sid}.jsonl",):
        try:
            if not path.is_file():
                continue
            for event, _start, _end in reverse_jsonl(
                path,
                None,
                max_bytes=4 << 20,
                max_events=4000,
            ):
                inner = event.get("event") if isinstance(event.get("event"), dict) else event
                if inner.get("type") == "header_update":
                    model = str((inner.get("patch") or {}).get("model") or "").strip()
                    if model:
                        return model
                if inner.get("type") == "session_started" or "session_hash" in inner:
                    model = str(inner.get("model") or "").strip()
                    if model:
                        return model
        except (OSError, TypeError, ValueError):
            continue
    return ""


def _model_ids(alias_or_id: str) -> set[str]:
    """Every name that identifies one configured model (alias and wire id)."""
    wanted = str(alias_or_id or "").strip()
    names = {wanted} if wanted else set()
    try:
        from sharedagents_core.kernel.config import load_global_config

        for row in load_global_config().get("models") or []:
            if not isinstance(row, dict):
                continue
            if wanted in (row.get("alias"), row.get("id")):
                names.update(str(row.get(key) or "") for key in ("alias", "id"))
    except Exception:  # noqa: BLE001
        pass
    return {name for name in names if name}


def _apply_model(web_sid: str, *, agent: str, model: str, session_id: str = "") -> bool:
    """Point a session at a specific model before its next prompt.

    Uses the same ``/model`` runtime command the chat tab's picker drives, so
    the Dev tab cannot drift into a second way of selecting a model.

    The gateway's command body names the session **``sessionId``** — camelCase,
    as ``routes/commands.py`` sends it. A ``session_id`` key is silently
    ignored and the command runs unbound, answering
    ``apply failed: session not found: ''`` with HTTP 200. That is how a run
    kept its old model after the operator changed it, and then died on the old
    provider's 402.
    """
    # Do not ask for a switch that is not needed. `/model` is refused outright
    # while a turn is running, so an unnecessary call turns a resumed run into a
    # hard failure for no reason.
    current = _session_model(web_sid)
    if current and current in _model_ids(model):
        return True
    # Prefer the id the binding just returned. Re-deriving it from the Web row
    # can lag behind a freshly materialized session, and `/model` on a stale id
    # answers "session not found" — which blocked a stage whose session was
    # perfectly healthy.
    target = session_id or web_sid
    if not target:
        return False
    try:
        resp = gateway_request(
            "POST",
            f"/control/v1/{agent}/command",
            json={
                "text": f"/model {model}",
                "sessionId": target,
                "frontend": "web",
            },
            headers=gateway_principal_headers(g.user_email),
            timeout=20,
        )
    except Exception:  # noqa: BLE001
        log.warning("sdlc: /model %s failed for %s", model, web_sid)
        return False
    if resp is None or resp.status_code >= 400:
        return False
    # The runtime answers 200 with ok=false for a rejected command, so the
    # status code alone does not mean the model was applied.
    try:
        body = resp.json()
    except ValueError:
        return False
    if not isinstance(body, dict) or not body.get("ok"):
        reason = str((body or {}).get("error") or "") if isinstance(body, dict) else ""
        log.warning("sdlc: /model %s rejected for %s: %s", model, web_sid, reason)
        _apply_model.last_error = reason  # type: ignore[attr-defined]
        return False
    return True


def _sdlc_driver(owner: str):
    """Build a driver bound to *owner*, safe to use off the request thread."""
    from sharedagents_web.dev_sdlc import (
        ROLE_LABELS,
        get_sdlc_store,
        role_config,
        validate_proposal as semantic_validate,
        verify_delivery as semantic_verify,
    )
    from sharedagents_web.dev_sdlc_driver import DriverHooks, SdlcDriver

    def _dispatch(web_sid: str, prompt: str, agent: str, model: str = ""):
        # Flask's ``g`` is request-scoped; the driver thread has none, so the
        # owner is captured here and pushed into a throwaway app context.
        from sharedagents_web.server import app

        with app.app_context():
            g.user_email = owner
            # A fresh Dev worker must be created by its first prompt: that
            # gateway operation stamps workspace_cwd and initial_model
            # atomically.  Calling /model first lazily materializes a session
            # rooted at `/`, and the later prompt cannot repair the root.
            current_model = _session_model(web_sid)
            if (
                model
                and current_model
                and not _apply_model(
                    web_sid,
                    agent=agent,
                    model=model,
                    session_id=web_sid,
                )
            ):
                # Running the stage on the wrong model burns a turn and can fail
                # on a provider the operator deliberately moved away from.
                reason = getattr(_apply_model, "last_error", "") or "unknown reason"
                return {
                    "error": f"could not switch the session to `{model}`: {reason}",
                }, 502
            return _submit_prompt(
                web_sid,
                prompt,
                agent=agent,
                initial_model=model if model and not current_model else "",
            )

    def _talk(run: dict, text: str) -> None:
        from sharedagents_web.server import app

        with app.app_context():
            g.user_email = owner
            _prompt_conductor(run, text)

    def _steer(web_sid: str, text: str) -> bool:
        """P1: one steering line into the RUNNING worker session.

        Best effort by design — a failed nudge must not fail the stage, and
        the watchdog still owns the kill decision.
        """
        from sharedagents_web.server import app

        sid = str(web_sid or "").strip()
        line = str(text or "").strip()
        if not sid or not line:
            return False
        with app.app_context():
            g.user_email = owner
            _payload, status = _submit_prompt(sid, line, agent="build")
        return int(status) < 400

    def _is_live(web_sid: str) -> bool:
        return _session_live(web_sid)

    def _new_session(workspace_cwd: str, *, agent: str = DEFAULT_AGENT) -> str:
        return str(
            store.create(
                owner,
                agent=agent,
                workspace_cwd=workspace_cwd,
            )["id"]
        )

    def _pending_interaction(web_sid: str) -> str:
        from sharedagents_core.kernel.layout import sessions_root
        from sharedagents_session.ask_pending import session_pending_interaction

        return session_pending_interaction(sessions_root(), web_sid)

    def _cancel_session(web_sid: str) -> bool:
        """Stop a post-ratification worker that requested human input."""
        from sharedagents_web.server import app

        with app.app_context():
            g.user_email = owner
            resp = gateway_request(
                "POST",
                f"/control/v1/orchestrator/sessions/{web_sid}/cancel",
                json={"force": True},
                headers=gateway_principal_headers(g.user_email),
                timeout=15,
            )
        return bool(resp is not None and resp.status_code < 400)

    gate_sessions: dict[tuple[str, str, str], str] = {}

    def _role_chat(role: str, run: dict, messages: list[dict[str, str]]) -> str:
        """Run one semantic gate through that run's selected agent/model.

        Gates are deliberately fresh sessions: each independent verifier must
        not inherit the generator's rationale or the other verifier's
        conclusions. The branch and the serialized gate transcript are their
        complete shared evidence.
        """
        agent, model = role_config(run, role)
        sid = _new_session(str(run.get("worktree") or ""))
        key = (str(run.get("run_id") or ""), str(run.get("stage") or ""), role)
        gate_sessions[key] = sid
        # Publish the concrete chat before dispatch.  Waiting for the gate's
        # verdict made Validate/Verify links appear only after they had stopped
        # running, so the Dev tab could never identify the active gate session.
        try:
            get_sdlc_store().record_gate_session_started(
                str(run.get("run_id") or ""),
                stage=str(run.get("stage") or ""),
                kind=role,
                session_id=sid,
                executor_agent=agent,
                executor_model=model,
            )
        except Exception:  # noqa: BLE001 — provenance must not stop the gate
            log.warning("sdlc: could not publish the live %s session for %s", role, run.get("run_id"), exc_info=True)
        completed = False
        failure = ""
        try:
            before = _sdlc_final_message(sid)[0]
            transcript = []
            for message in messages:
                label = str(message.get("role") or "user").upper()
                transcript.append(f"--- {label} ---\n{message.get('content') or ''}")
            prompt = (
                f"SDLC {(ROLE_LABELS.get(role) or role).upper()} GATE\n"
                f"RUN: {run.get('run_id') or ''}\n"
                f"STAGE: {run.get('stage') or ''}\n\n"
                + "\n\n".join(transcript)
                + "\n\nReturn only the response required by the gate contract above."
            )
            payload, status = _dispatch(sid, prompt, agent, model)
            if status >= 400:
                detail = str((payload or {}).get("error") or f"HTTP {status}")
                failure = f"dispatch failed: {detail}"
                raise RuntimeError(f"{role} agent `{agent}` dispatch failed: {detail}")
            deadline = time.monotonic() + float(os.environ.get("SHAREDAGENTS_SDLC_ROLE_GATE_TIMEOUT_S", "900"))
            while time.monotonic() < deadline:
                identity, text, error = _sdlc_final_message(sid)
                if identity and identity != before:
                    if error:
                        failure = f"turn failed: {error[:300]}"
                        raise RuntimeError(f"{role} agent `{agent}` failed: {error[:300]}")
                    if text.strip():
                        completed = True
                        return text
                time.sleep(2.0)
            failure = "timed out before a gate decision"
            raise TimeoutError(f"{role} agent `{agent}` timed out")
        finally:
            if not completed:
                gate_sessions.pop(key, None)
                try:
                    get_sdlc_store().record_gate_session_abandoned(
                        key[0],
                        stage=key[1],
                        kind=role,
                        session_id=sid,
                        note=failure,
                    )
                except Exception:  # noqa: BLE001 — provenance must not mask failure
                    log.warning(
                        "sdlc: could not close the abandoned %s session for %s", role, run.get("run_id"), exc_info=True
                    )

    def _validate(stage: str, run: dict):
        current = dict(run, stage=stage)
        gate_sessions.pop((str(current.get("run_id") or ""), stage, "validation"), None)
        return semantic_validate(
            stage,
            current,
            chat=lambda messages: _role_chat("validation", current, messages),
        )

    def _verify(stage: str, run: dict, marker: dict):
        current = dict(run, stage=stage)
        gate_sessions.pop((str(current.get("run_id") or ""), stage, "verification"), None)
        return semantic_verify(
            stage,
            current,
            marker,
            chat=lambda messages: _role_chat("verification", current, messages),
        )

    def _independent_reviews(stage: str, run: dict, marker: dict):
        """Two independent verifiers re-read the same oracles — one wait, two models."""
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

        # Imported UP HERE on purpose. It used to be imported after the block
        # below while being referenced INSIDE it, which makes it a local bound
        # too late: any reviewer error raised UnboundLocalError from the very
        # handler written to turn that error into a readable gate note.
        from sharedagents_web.dev_sdlc import (
            GATE_UNUSABLE,
            REVIEW_ANSWER_TIMEOUT,
            reviews_agree,
        )

        current = dict(run, stage=stage)

        def _one(role: str) -> tuple[bool, list]:
            gate_sessions.pop((str(current.get("run_id") or ""), stage, role), None)
            return semantic_verify(
                stage,
                current,
                marker,
                force=True,
                chat=lambda messages, _role=role: _role_chat(_role, current, messages),
            )

        # NOT a `with` block: its __exit__ joins the pool, so a reviewer stuck
        # on a socket would be waited on again right after we gave up on it.
        pool = ThreadPoolExecutor(max_workers=2)
        deadline = time.monotonic() + REVIEW_ANSWER_TIMEOUT
        try:
            future_a = pool.submit(_one, "validation")
            future_b = pool.submit(_one, "verification")

            def _collect(future, label: str) -> tuple[bool, list]:
                try:
                    # NOT max(1.0, ...): that hands the SECOND reviewer an
                    # extra second after the shared deadline has already gone,
                    # so B could be accepted purely because A used up the time.
                    return future.result(timeout=max(0.0, deadline - time.monotonic()))
                except FutureTimeout:
                    # UNUSABLE, not FAIL: a reviewer that never answered has
                    # said nothing about the code, and the caller already knows
                    # how to retry an unusable gate.
                    return False, [
                        f"{GATE_UNUSABLE} verify {label} did not answer within {int(REVIEW_ANSWER_TIMEOUT)}s",
                    ]
                except Exception as error:  # noqa: BLE001 — a missing reviewer is a fail
                    return False, [f"{GATE_UNUSABLE} verify {label} failed: {error}"]

            ok_a, notes_a = _collect(future_a, "A")
            ok_b, notes_b = _collect(future_b, "B")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return reviews_agree(ok_a, notes_a, ok_b, notes_b)

    def _sdlc_board(run: dict) -> dict:
        sid = str(run.get("conductor_session_id") or "").strip()
        if not sid:
            return {}
        try:
            from sharedagents_agent.tools.impl.sdlc import load_board, resolve_board_path
            from sharedagents_core.kernel.layout import sessions_root

            return load_board(resolve_board_path(sessions_root(), sid))
        except Exception:  # noqa: BLE001
            return {}

    def _arm_goal(run: dict, prompt: str) -> None:
        """Register ``/goal`` on the conductor the same way the Chat composer does."""
        from sharedagents_web.server import app

        line = str(prompt or "").strip().split("\n", 1)[0].strip()
        if not line.startswith("/goal"):
            return
        sid = str(run.get("conductor_session_id") or "").strip()
        if not sid:
            return
        with app.app_context():
            g.user_email = owner
            sdata = store.get(sid) or {}
            agent = str(sdata.get("agent") or "jarvis")
            resp = gateway_request(
                "POST",
                f"/control/v1/{agent}/command",
                json={"text": line, "frontend": "web", "sessionId": sid},
                headers=gateway_principal_headers(g.user_email),
                timeout=30,
            )
            if resp is None or resp.status_code >= 400:
                log.warning(
                    "sdlc: /goal command failed for %s (%s)",
                    sid,
                    getattr(resp, "status_code", "offline"),
                )

    def _mark_tests_ready(run: dict, evidence: str) -> None:
        from sharedagents_agent.tools.impl.sdlc import apply_drive_ready
        from sharedagents_core.kernel.layout import sessions_root

        sid = str(run.get("conductor_session_id") or "").strip()
        if not sid:
            raise RuntimeError("run has no conductor board")
        result = apply_drive_ready(
            sessions_root(),
            sid,
            evidence=evidence,
        )
        if str(result).startswith(("REFUSED", "JUMP")):
            raise RuntimeError(result)

    def _mark_stage_passed(run: dict, stage: str, note: str, sha: str) -> None:
        from sharedagents_agent.tools.impl.sdlc import apply_stage_pass
        from sharedagents_core.kernel.layout import sessions_root

        sid = str(run.get("conductor_session_id") or "").strip()
        if not sid:
            raise RuntimeError("run has no conductor board")
        result = apply_stage_pass(
            sessions_root(),
            sid,
            stage=stage,
            note=note,
            sha=sha,
        )
        if str(result).startswith(("REFUSED", "JUMP", "BLOCKED", "LOOP")):
            raise RuntimeError(result)

    return SdlcDriver(
        store=get_sdlc_store(),
        hooks=DriverHooks(
            dispatch=_dispatch,
            is_live=_is_live,
            final_message=_sdlc_final_message,
            new_session_id=_new_session,
            forge=_Forge(),
            validate_proposal=_validate,
            verify_delivery=_verify,
            independent_reviews=_independent_reviews,
            gate_session_id=lambda run_id, stage, role: gate_sessions.pop((run_id, stage, role), ""),
            prompt_conductor=_talk,
            sdlc_board=_sdlc_board,
            arm_goal=_arm_goal,
            mark_tests_ready=_mark_tests_ready,
            mark_stage_passed=_mark_stage_passed,
            pending_interaction=_pending_interaction,
            cancel_session=_cancel_session,
            last_activity_at=_sdlc_last_activity_at,
            steer=_steer,
            session_digest=_sdlc_session_digest,
            session_usage=_sdlc_session_usage,
        ),
    )


def _sdlc_live_role_sessions(run: dict, events: list[dict]) -> dict[str, str]:
    """Live session id keyed by the exact current ``stage:role``.

    Dev polls this projection every four seconds.  Inspect only the current
    stage's newest three role sessions and use the cached read-path probe; an
    indeterminate probe produces no running claim.
    """
    from sharedagents_goal.goal_registry import session_work_live_cheap
    from sharedagents_web.dev_sdlc import canonical_stage

    current_stage = canonical_stage(str(run.get("stage") or ""))
    newest: dict[tuple[str, str], str] = {}
    for event in events or []:
        stage = canonical_stage(str(event.get("stage") or ""))
        if not stage or stage != current_stage:
            continue
        status = str(event.get("status") or "")
        kind = str(event.get("gate_kind") or "")
        role = kind if kind in {"validation", "verification"} else ""
        if not role and status == "started":
            role = "generation"
        sid = str(event.get("executor_session_id") or "")
        key = (stage, role)
        if role and status == "gate_session_abandoned":
            if not sid or newest.get(key) == sid:
                newest.pop(key, None)
            continue
        if role and status in {"gate_decided", "gate_unusable"} and not sid:
            newest.pop(key, None)
            continue
        if role and sid:
            newest[(stage, role)] = sid
    # A legacy generation event may predate session provenance.  Keep the row
    # fallback used by the existing links so its live marker agrees with them.
    fallback = str(
        run.get("review_session_id")
        if canonical_stage(str(run.get("stage") or "")) == "deploy"
        else (run.get("author_session_id") or run.get("session_id") or "")
    )
    if fallback and current_stage:
        newest.setdefault((current_stage, "generation"), fallback)
    live: dict[str, str] = {}
    for (stage, role), sid in newest.items():
        is_live, _source, determinate, _terminal = session_work_live_cheap(
            sid,
        )
        if determinate and is_live:
            live[f"{stage}:{role}"] = sid
    return live


def _sdlc_role_reviews(run: dict) -> dict[str, dict[str, str]]:
    """Verify A/B rows from the conductor board — enough for the Dev chips."""
    sid = str(run.get("conductor_session_id") or "").strip()
    if not sid:
        return {}
    try:
        from sharedagents_agent.tools.impl.sdlc import load_board, sdlc_path_for_session
        from sharedagents_core.kernel.layout import sessions_root

        board = load_board(sdlc_path_for_session(sessions_root(), sid)) or {}
    except Exception:  # noqa: BLE001
        return {}
    reviews = board.get("reviews") if isinstance(board.get("reviews"), dict) else {}
    out: dict[str, dict[str, str]] = {}
    for role in ("validation", "verification"):
        row = reviews.get(role) if isinstance(reviews.get(role), dict) else {}
        out[role] = {
            "status": str(row.get("status") or ""),
            "verdict": str(row.get("verdict") or ""),
            "session_id": str(row.get("session_id") or ""),
            "sha": str(row.get("sha") or ""),
        }
    return out


def _conductor_snapshot(run: dict) -> dict | None:
    """Jarvis conductor id, chat href, and last-known store pulse."""
    from sharedagents_web.dev_sdlc import conductor_card

    sid = str(run.get("conductor_session_id") or run.get("session_id") or "").strip()
    if not sid:
        return None
    owner = str(getattr(g, "user_email", "") or "")
    sdata = {}
    try:
        sdata = store.get(sid, owner) or store.get(sid) or {}
    except Exception:  # noqa: BLE001 — a missing store row must not hide the link
        sdata = {}
    if not isinstance(sdata, dict):
        sdata = {}
    return conductor_card(
        sid,
        title=str(sdata.get("title") or "Jarvis"),
        agent=str(sdata.get("agent") or "jarvis"),
        state=str(sdata.get("state") or sdata.get("session_state") or ""),
        updated_at=str(sdata.get("updated_at") or ""),
        base=BASE_PATH,
    )


def _sdlc_view(run: dict) -> dict:
    from sharedagents_web.dev_sdlc import get_sdlc_store
    from sharedagents_web.dev_sdlc_driver import run_is_active
    from sharedagents_web.dev_task import sync_run_title_from_session

    run = sync_run_title_from_session(
        run,
        owner=str(getattr(g, "user_email", "") or ""),
    )
    out = dict(run)
    sdlc = get_sdlc_store()
    out["events"] = sdlc.events(str(run.get("run_id") or ""))
    # The gateway retained pre-migration gate chats that the event ledger did
    # not know how to name. Recover their links once, as new audit events; this
    # keeps the original verdict rows immutable and makes existing runs as
    # inspectable as new ones.
    try:
        recovered = _recover_legacy_gate_sessions(
            run,
            out["events"],
            str(getattr(g, "user_email", "") or ""),
            sdlc,
        )
        if recovered:
            out["events"] = sdlc.events(str(run.get("run_id") or ""))
    except Exception:  # noqa: BLE001 — a missing legacy link must not break Dev
        log.warning("sdlc: could not recover legacy gate sessions for %s", run.get("run_id"), exc_info=True)
    out["reviews"] = _sdlc_role_reviews(run)
    try:
        from sharedagents_web.dev_sdlc import backfill_review_gate_events

        backfilled = backfill_review_gate_events(run, out["events"], out["reviews"], sdlc)
        if backfilled:
            out["events"] = sdlc.events(str(run.get("run_id") or ""))
    except Exception:  # noqa: BLE001 — a missing A chip must not break Dev
        log.warning("sdlc: could not backfill review gates for %s", run.get("run_id"), exc_info=True)
    out["live_role_sessions"] = _sdlc_live_role_sessions(run, out["events"])
    out["driver_active"] = run_is_active(str(run.get("run_id") or ""))
    out["forge_configured"] = forge_client.configured()
    pulse = _conductor_snapshot(run)
    if pulse:
        out["conductor"] = pulse
    try:
        ask, ask_known = _conductor_pending_ask_card(run)
    except Exception:  # noqa: BLE001 — a missing card must not break the run GET
        log.warning("sdlc: could not project pending ask for %s", run.get("run_id"), exc_info=True)
        ask, ask_known = None, False
    # The client retires stale ask cards against this. It may only do so when
    # the answer is authoritative, so say which it is.
    out["pending_ask_known"] = bool(ask_known)
    if ask:
        out["pending_ask"] = ask
    sid = str(run.get("conductor_session_id") or "").strip()
    out["intake_quiescent"] = bool(
        str(run.get("status") or "") == "ready"
        and ask_known
        and not ask
        and sid
        and not _conductor_tree_busy(sid, quiet=RATIFY_QUIET_SECONDS)
    )
    try:
        aliases = sorted(_conductor_hash_set(run))
    except Exception:  # noqa: BLE001
        aliases = []
    if aliases:
        out["conductor_aliases"] = aliases
    # P4: the running-gate clock and the P5 fuse choices read these.
    from sharedagents_web.dev_sdlc import unreviewed_stages, watchdog_projection

    out.update(watchdog_projection(run, out.get("events") or []))
    if int(run.get("gate_unusable_count") or 0):
        # No fallback: if the merged identity is unknown the backend treats it
        # as unknown, and the page must not say "not blocked" over the top of
        # that.
        out["unreviewed_stages"] = unreviewed_stages(out["events"], str(run.get("merged_head_sha") or ""))
    return out


def _legacy_gate_session_matches(run_id: str, events: list[dict], sessions: list[dict], read_transcript) -> dict:
    """Newest preserved chat for each unlinked legacy semantic gate.

    Matching is semantic, not timestamp guesswork: fresh gate transcripts carry
    the run id, stage, and role in their dispatch preamble. Candidates are
    ordered by their immutable gateway creation id, so the first match is the
    newest retry/rework attempt even when recent activity reordered the list.
    Transcript content is data here; only fixed marker membership is inspected.
    """
    from sharedagents_web.dev_sdlc import canonical_stage

    linked: set[tuple[str, str]] = set()
    needed: dict[tuple[str, str], dict] = {}
    for event in events or []:
        raw_stage = str(event.get("stage") or "")
        stage = canonical_stage(raw_stage)
        kind = str(event.get("gate_kind") or "")
        key = (stage, kind)
        if kind in {"validation", "verification"} and event.get("executor_session_id"):
            linked.add(key)
        if (
            event.get("status") == "gate_decided"
            and kind in {"validation", "verification"}
            and str(event.get("executor_agent") or "")
        ):
            # Legacy ``spec`` and current ``design`` are one visible stage.
            # Keep only the newest decision for that semantic gate so an old
            # recovered spec chat cannot overwrite an already-linked design
            # retry in the UI.
            needed[key] = dict(event, raw_stage=raw_stage, canonical_stage=stage)
    needed = {key: event for key, event in needed.items() if key not in linked}
    needed_agents = {str(event.get("executor_agent") or "") for event in needed.values()}
    found: dict[tuple[str, str], dict] = {}

    def _created(item: dict) -> str:
        sid = str(item.get("id") or "")
        stamp = "".join(c for c in sid.split("-", 1)[0] if c.isdigit())
        return stamp[:20] if len(stamp) >= 14 else ""

    ordered = sorted(sessions or [], key=_created, reverse=True)
    for session in ordered:
        if len(found) == len(needed):
            break
        if str(session.get("agent") or "") not in needed_agents:
            continue
        sid = str(session.get("id") or "")
        if not sid:
            continue
        try:
            transcript = str(read_transcript(sid) or "")
        except Exception:  # noqa: BLE001 — one corrupt chat does not hide others
            continue
        # The dispatch preamble is first. A verifier may quote validator text,
        # so arbitrary marker membership would let one chat answer two gates.
        role_positions = [
            (transcript.find(f"SDLC {role.upper()} GATE"), role) for role in ("validation", "verification")
        ]
        transcript_roles = [item for item in role_positions if item[0] >= 0]
        transcript_kind = min(transcript_roles)[1] if transcript_roles else ""
        for (stage, kind), event in needed.items():
            raw_stage = str(event.get("raw_stage") or stage)
            result_key = (raw_stage, kind)
            if result_key in found:
                continue
            if str(session.get("agent") or "") != str(event.get("executor_agent") or ""):
                continue
            if kind != transcript_kind:
                continue
            # Do not attach a newer failed/rework chat to an older decision.
            # Gateway ids and event times both begin with sortable UTC digits.
            started = "".join(c for c in sid.split("-", 1)[0] if c.isdigit())
            decided = "".join(c for c in str(event.get("created_at") or "") if c.isdigit())
            # Seconds are present in both formats; fractional seconds are not
            # guaranteed on older event rows.
            if len(started) >= 14 and len(decided) >= 14 and started[:14] > decided[:14]:
                continue
            if f"RUN: {run_id}" in transcript and f"STAGE: {raw_stage}" in transcript:
                found[result_key] = {
                    "session_id": sid,
                    "agent": str(event.get("executor_agent") or ""),
                    "model": str(event.get("executor_model") or ""),
                    "canonical_stage": str(event.get("canonical_stage") or raw_stage),
                }
                break
    return found


def _recover_legacy_gate_sessions(run: dict, events: list[dict], owner: str, sdlc) -> int:
    """Persist exact links for gate chats created before provenance existed."""
    from sharedagents_web.dev_sdlc import canonical_stage
    from sharedagents_web.server import GW_SESSIONS_DIR

    run_id = str(run.get("run_id") or "")
    if not run_id or not owner:
        return 0
    if any(event.get("status") == "gate_session_recovery_checked" for event in events or []):
        return 0
    linked = {
        (canonical_stage(str(event.get("stage") or "")), str(event.get("gate_kind") or ""))
        for event in events or []
        if event.get("executor_session_id")
    }
    eligible = any(
        event.get("status") == "gate_decided"
        and str(event.get("gate_kind") or "") in {"validation", "verification"}
        and str(event.get("executor_agent") or "")
        and (canonical_stage(str(event.get("stage") or "")), str(event.get("gate_kind") or "")) not in linked
        for event in events or []
    )
    if not eligible:
        sdlc.mark_gate_session_recovery_checked(run_id)
        return 0
    candidates = store.list_session_bindings(owner)
    root = Path(GW_SESSIONS_DIR).resolve()

    def _read(session_id: str) -> str:
        # Gateway ids are filenames. Prove the resolved target stays below the
        # session root before reading the bounded prompt prefix.
        if (
            not session_id
            or len(session_id) > 200
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in session_id)
        ):
            return ""
        for path in ((root / f"{session_id}.jsonl").resolve(),):
            if path.parent != root or not path.is_file():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return handle.read(1024 * 1024)
        return ""

    matches = _legacy_gate_session_matches(run_id, events, candidates, _read)
    count = 0
    for (stage, kind), match in matches.items():
        if sdlc.record_gate_session_link(
            run_id,
            stage=stage,
            kind=kind,
            session_id=match["session_id"],
            executor_agent=match["agent"],
            executor_model=match["model"],
        ):
            count += 1
    sdlc.mark_gate_session_recovery_checked(run_id)
    return count


@bp.route("/api/dev/sdlc/runs", methods=["GET"])
@require_auth
def api_sdlc_list():
    from sharedagents_web.dev_sdlc import (
        default_model,
        default_role_agents,
        deploy_targets,
        get_sdlc_store,
    )

    repo = str(request.args.get("repo") or "").strip()
    include_archived = str(request.args.get("include_archived") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    runs = get_sdlc_store().list(repo=repo, include_archived=include_archived)
    # The Dev tab polls this route every 2s while an intake is open; it is the
    # only heartbeat a waiting run gets, so the stall watchdog rides on it.
    runs = [_recover_stalled_intake(row) if str(row.get("status") or "") == "intake" else row for row in runs]
    generator = str(request.args.get("generation_agent") or request.args.get("agent") or "jarvis")
    defaults = default_role_agents(generator)
    return jsonify(
        {
            "runs": runs,
            "default_model": default_model(generator),
            "default_role_agents": defaults,
            "default_role_models": {role: default_model(agent) for role, agent in defaults.items()},
            "deploy_targets": deploy_targets(repo),
            "forge_configured": forge_client.configured(),
        }
    )


@bp.route("/api/dev/sdlc/runs", methods=["POST"])
@require_auth
def api_sdlc_create():
    """Open an intake and ask the first round of clarifying questions."""
    from sharedagents_web.dev_sdlc import SdlcError, get_sdlc_store, save_attachments

    payload = request.get_json(silent=True) or {}
    try:
        run = get_sdlc_store().create(
            repo=_require_repo(str(payload.get("repo") or "")),
            kind=str(payload.get("kind") or "feature"),
            headline=str(payload.get("object") or payload.get("headline") or ""),
            detail=str(payload.get("detail") or ""),
            agent="jarvis",
            model=str(payload.get("model") or payload.get("generation_model") or ""),
            validation_agent=str(payload.get("validation_agent") or ""),
            validation_model=str(payload.get("validation_model") or ""),
            generation_agent="jarvis",
            generation_model=str(payload.get("generation_model") or payload.get("model") or ""),
            verification_agent=str(payload.get("verification_agent") or ""),
            verification_model=str(payload.get("verification_model") or ""),
            max_laps=int(payload.get("max_laps") or 3),
        )
    except (SdlcError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    # Attachments are folded into the detail AFTER create, because they are
    # stored under the run's own id and that id does not exist until now. A
    # failure here must not lose the description the operator just typed, so it
    # degrades to "no attachments" rather than failing the intake.
    attachments = payload.get("attachments")
    if isinstance(attachments, list) and attachments:
        try:
            saved, markdown = save_attachments(
                run["run_id"],
                attachments,
                url_for=lambda rid, name: f"{BASE_PATH}/api/dev/sdlc/runs/{rid}/files/{name}",
            )
            if markdown:
                run = get_sdlc_store().patch(run["run_id"], detail=(str(run.get("detail") or "") + markdown)) or run
        except Exception as error:  # noqa: BLE001
            log.warning("sdlc: attachments dropped for %s: %s", run["run_id"], error)
    # One Jarvis chat + one Tasks row for this run. Label is board-only so
    # the title agent can still name the conversation.
    from sharedagents_web.dev_task import ensure_run_conductor

    run = ensure_run_conductor(run, owner=str(getattr(g, "user_email", "") or ""))
    # Mint the conductor + Tasks row, then start the Jarvis interview.
    # Questions come from the chat question tool over SSE, not a second form.
    return _sdlc_clarify_round(run, kick_research=True)


@bp.route("/api/dev/sdlc/runs/<run_id>/files/<path:name>")
@require_auth
def api_sdlc_run_file(run_id: str, name: str):
    """Serve one intake attachment, jailed to that run's own directory.

    Single-user surface, so @require_auth is the access
    control; the jail is what stops a crafted run_id or name from reading
    anything else. 404 rather than 403 so the response does not report which
    runs exist.
    """
    from sharedagents_web.dev_sdlc import SdlcError, attachment_dir, safe_attachment_name

    safe = safe_attachment_name(name)
    if not safe:
        return jsonify({"error": "not found"}), 404
    try:
        base = attachment_dir(run_id)
    except SdlcError:
        return jsonify({"error": "not found"}), 404
    target = (base / safe).resolve()
    # Prove the resolved path is still inside the run's own directory: this is
    # what defeats a symlink planted under it, not just `..` in the name.
    if target.parent != base or not target.is_file():
        return jsonify({"error": "not found"}), 404
    return send_from_directory(str(base), safe, as_attachment=False)


def fold_chat_answers_into_run(sid: str, body: dict) -> None:
    """When Chat answers a clarify card, record the same answers on Dev.

    Chat ``/answer`` already unblocks Jarvis. This keeps the Dev transcript
    in lockstep so the intake form does not stay on questions the operator
    already settled in the session.
    """
    from sharedagents_web.dev_clarify import (
        MAX_CLARIFY_ROUNDS,
        answered_rounds,
        chat_answers_to_transcript,
    )
    from sharedagents_web.dev_sdlc import get_sdlc_store

    sdata = store.get(sid) or {}
    run_id = str(sdata.get("sdlc_run_id") or "").strip()
    sdlc = get_sdlc_store()
    if not run_id:
        run_id = str(sdlc.conductor_run_ids([sid]).get(sid) or "").strip()
        if run_id and sdata:
            sdata["sdlc_run_id"] = run_id
            try:
                store._save(sdata)
            except Exception:  # noqa: BLE001
                pass
    if not run_id:
        return
    run = sdlc.get(run_id)
    if not run or run.get("status") not in ("intake", "ready"):
        return
    added = chat_answers_to_transcript(
        list(body.get("answers") or []),
        _session_questions_for_run(run) or list(run.get("questions") or []),
    )
    if not added:
        return
    # An answer reopens the interview: server questions are consumed and a
    # `ready` contract computed before it is no longer ratifiable. The append
    # is conditional on the transcript it was appended to, so it can neither
    # land on a ratified run nor drop a concurrent answer it never saw; a lost
    # write is retried on the fresh row.
    transcript: list[dict] = []
    for _attempt in range(3):
        existing = {str(row.get("question") or "") for row in (run.get("transcript") or []) if isinstance(row, dict)}
        fresh = [row for row in added if row["question"] not in existing]
        if not fresh:
            return
        base = list(run.get("transcript") or [])
        transcript = base + fresh
        landed = sdlc.patch_intake(
            run_id,
            expect_transcript=base,
            transcript=transcript,
            questions=[],
            status="intake",
        )
        if landed is not None:
            run = landed
            break
        run = sdlc.get(run_id) or {}
        if not run or run.get("issue_number") or run.get("status") not in ("intake", "ready"):
            return
    else:
        log.warning("sdlc: chat answer for %s could not be mirrored", run_id)
        return
    if answered_rounds(transcript) < MAX_CLARIFY_ROUNDS:
        # The conductor Chat is the sole sequential interviewer. Starting a
        # Dev-side clarify call here creates a competing fallback question
        # whose answer cannot inform Chat's next card.
        return

    # The bounded Chat interview is complete. Synthesis stays asynchronous so
    # Chat /answer can return, but first stops the conductor before it can ask
    # a ninth question that races the frozen contract.
    # Chat /answer must return immediately. Clarify (intake LLM) can take
    # minutes and used to stall the Flask worker — same hang as Start.
    owner = str(getattr(g, "user_email", "") or "")
    from threading import Thread

    from sharedagents_web.server import app

    def _go() -> None:
        try:
            with app.app_context():
                g.user_email = owner
                latest = get_sdlc_store().get(run_id)
                # Re-check AFTER the wait, not just before it. `clarify()` is a
                # model call that takes minutes, and the operator ratifies in
                # the middle of it -- so this thread was landing on a run that
                # had already been frozen, reverting `running` to `ready` and
                # overwriting the ratified requirements and criteria with a
                # fresh guess. The contract is frozen; nothing here may edit it.
                if latest and not latest.get("issue_number"):
                    claimed, cancelled = _cancel_conductor_for_synthesis(
                        latest,
                        owner,
                    )
                    # A lost claim means ratification owns the outcome now. A
                    # refused cancel leaves a possibly live conductor, and
                    # publishing beside it would give the interview two
                    # owners; the run is `intake` with no question, which is
                    # exactly the shape the stall recovery resumes from.
                    if claimed and cancelled:
                        _sdlc_clarify_round(latest)
                    else:
                        log.info(
                            "sdlc: %s synthesis deferred to stall recovery (claimed=%s cancelled=%s)",
                            run_id,
                            claimed,
                            cancelled,
                        )
        except Exception:  # noqa: BLE001
            log.exception("sdlc: fold clarify failed for %s", run_id)

    Thread(target=_go, name="sdlc-fold-clarify", daemon=True).start()


def _sdlc_clarify_round(run: dict, *, kick_research: bool = False):
    from sharedagents_web.dev_sdlc import SdlcError, clarify, get_sdlc_store
    from sharedagents_web.dev_task import ensure_run_conductor

    run = ensure_run_conductor(
        run,
        owner=str(getattr(g, "user_email", "") or ""),
    )
    sdlc = get_sdlc_store()
    # Chat may research once at Start. Later answer rounds stay on Dev.
    if kick_research and not _conductor_pending_asks(run):
        _start_conductor_interview(run)
    rows = list(run.get("transcript") or [])
    if kick_research and not rows:
        updated = (
            sdlc.patch(
                str(run["run_id"]),
                questions=[],
                status="intake",
                last_error="",
            )
            or run
        )
        return jsonify({"run": _sdlc_view(updated)})
    # The invariant, next to the only writer that can break it: once the issue
    # exists the contract is frozen, and a clarify round that lands late must
    # not rewrite it. The caller re-checks too; this is what makes it true for
    # every caller.
    if run.get("issue_number"):
        return jsonify({"run": _sdlc_view(sdlc.get(str(run["run_id"])) or run)})
    try:
        result = clarify(
            headline=str(run.get("headline") or ""),
            detail=str(run.get("detail") or ""),
            kind=str(run.get("kind") or "feature"),
            repo=str(run.get("repo") or ""),
            transcript=rows,
        )
    except SdlcError as error:
        updated = sdlc.patch(str(run["run_id"]), last_error=str(error))
        return jsonify({"run": _sdlc_view(updated or run), "error": str(error)}), 503
    # A slow synthesis has no authority over a newer answer or a ratified run.
    fresh = sdlc.get(str(run["run_id"])) or run
    if (
        fresh.get("issue_number")
        or fresh.get("status") not in ("intake", "ready")
        or list(fresh.get("transcript") or []) != rows
        or fresh.get("detail") != run.get("detail")
    ):
        return jsonify({"run": _sdlc_view(fresh)})
    run = fresh
    # A configured interview floor is optional. Contract defects and pending
    # questions remain authoritative even when no minimum was configured.
    from sharedagents_web.dev_clarify import (
        MIN_CLARIFY_ROUNDS,
        answered_rounds,
        depth_followup_question,
        question_was_answered,
    )

    deep_enough = answered_rounds(rows) >= MIN_CLARIFY_ROUNDS
    if not deep_enough:
        log.info(
            "sdlc: %s answered %d of %d clarify rounds; still interviewing",
            run.get("run_id"),
            answered_rounds(rows),
            MIN_CLARIFY_ROUNDS,
        )
        # Before the floor, the server owns the sequence. A model that returns
        # not-ready with the same question is just as stalled as one that
        # returns ready with no question. Always hand the operator one new,
        # answer-dependent safety card.
        result["ready"] = False
        result["questions"] = [
            depth_followup_question(
                rows,
                headline=str(run.get("headline") or ""),
                detail=str(run.get("detail") or ""),
            )
        ]
    elif not result["ready"]:
        proposed = list(result.get("questions") or [])
        if not proposed or all(question_was_answered(row, rows) for row in proposed):
            # A completed answer must never reopen the identical question.
            # Keep a silent/repetitive clarifier moving without inventing a
            # batch: one causal fallback is derived from the newest answer.
            result["questions"] = [
                depth_followup_question(
                    rows,
                    headline=str(run.get("headline") or ""),
                    detail=str(run.get("detail") or ""),
                )
            ]
        else:
            result["questions"] = proposed[:1]
    if result.get("questions"):
        result["questions"] = result["questions"][:1]
    # A later round must not DESTROY what an earlier one established. The
    # clarifier degrades: one live round returned 24 requirements and zero
    # criteria, wiping eight good given/when/then/fail-if lines and dropping
    # the run back to `intake` with nothing to freeze. An empty list is "I did
    # not produce any this time", never "there are none".
    kept_reqs = result["requirements"] or list(run.get("requirements") or [])
    kept_crit = result["criteria"] or list(run.get("criteria") or [])
    # Conditional: the re-read above can be overtaken by a ratification or a
    # newer answer before this statement runs. The row decides, not the read.
    updated = sdlc.patch_intake(
        str(run["run_id"]),
        expect_transcript=rows,
        questions=[] if (result["ready"] and deep_enough) else list(result.get("questions") or []),
        requirements=kept_reqs,
        criteria=kept_crit,
        summary=result["summary"] or str(run.get("summary") or ""),
        status="ready" if (result["ready"] and deep_enough) else "intake",
        last_error="",
    )
    if updated is None:
        log.info("sdlc: stale clarify round discarded for %s", run.get("run_id"))
        return jsonify({"run": _sdlc_view(sdlc.get(str(run["run_id"])) or run)})
    return jsonify({"run": _sdlc_view(updated)})


@bp.route("/api/dev/sdlc/runs/<run_id>", methods=["GET"])
@require_auth
def api_sdlc_get(run_id: str):
    from sharedagents_web.dev_sdlc import get_sdlc_store

    run = get_sdlc_store().get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    if str(run.get("status") or "") == "intake":
        run = _recover_stalled_intake(run)
    return jsonify({"run": _sdlc_view(run)})


@bp.route("/api/dev/sdlc/runs/<run_id>/answers", methods=["POST"])
@require_auth
def api_sdlc_answers(run_id: str):
    """Record the operator's answers and ask the next round."""
    from sharedagents_web.dev_sdlc import get_sdlc_store

    sdlc = get_sdlc_store()
    run = sdlc.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    if run.get("status") not in ("intake", "ready"):
        return jsonify({"error": "intake is closed for this run"}), 409
    payload = request.get_json(silent=True) or {}
    answers = payload.get("answers")
    transcript = list(run.get("transcript") or [])
    for entry in answers if isinstance(answers, list) else []:
        if not isinstance(entry, dict):
            continue
        question = str(entry.get("question") or entry.get("text") or "").strip()
        answer = str(entry.get("answer") or "").strip()
        # Provenance matters: a suggestion the operator accepted unchanged is
        # the model agreeing with itself, and must not read as fresh evidence
        # in the next round.
        source = "suggested" if entry.get("source") == "suggested" else "operator"
        if question and answer:
            transcript.append(
                {
                    "question": question,
                    "answer": answer,
                    "source": source,
                }
            )
    note = str(payload.get("note") or "").strip()
    if note:
        transcript.append(
            {
                "question": "(operator note)",
                "answer": note,
                "source": "operator",
            }
        )
    prior = len(run.get("transcript") or [])
    if len(transcript) == prior:
        return jsonify({"error": "no answers supplied"}), 400
    added = transcript[prior:]
    server_owned = bool(run.get("questions"))
    # An answer reopens the interview: the server question is consumed and a
    # `ready` contract computed before it is no longer ratifiable. Conditional
    # on the transcript this request read: it cannot land on a run ratified
    # since, nor on top of a concurrent answer it never saw, and reset either.
    landed = sdlc.patch_intake(
        run_id,
        expect_transcript=run.get("transcript") or [],
        transcript=transcript,
        questions=[],
        status="intake",
    )
    if landed is None:
        fresh = sdlc.get(run_id) or run
        if fresh.get("issue_number") or fresh.get("status") not in ("intake", "ready"):
            return jsonify({"error": "intake is closed for this run", "run": _sdlc_view(fresh)}), 409
        on_row = {
            (str(r.get("question") or ""), str(r.get("answer") or ""))
            for r in (fresh.get("transcript") or [])
            if isinstance(r, dict)
        }
        if all((row["question"], row["answer"]) in on_row for row in added):
            # Chat mirrored this same answer first. The row already says what
            # this request meant to say; that is not a conflict.
            return jsonify({"run": _sdlc_view(fresh)})
        if list(fresh.get("transcript") or []) == list(run.get("transcript") or []):
            # Nothing moved on the row, so the write lost to the claim: the
            # contract is being frozen right now.
            return jsonify(
                {"error": "The scope is being finalized. Try again in a moment.", "run": _sdlc_view(fresh)}
            ), 409
        return jsonify({"error": "the interview moved on; answer the current question", "run": _sdlc_view(fresh)}), 409
    run = landed
    # Unlock Chat only while Chat owns intake. A server-owned question means
    # the conductor was cancelled before that question was published.
    if not server_owned and not _answer_conductor_ask(run, added):
        owner = str(getattr(g, "user_email", "") or "")
        from threading import Thread

        from sharedagents_web.server import app

        def _unlock() -> None:
            try:
                with app.app_context():
                    g.user_email = owner
                    _unlock_conductor_chat(run, added)
            except Exception:  # noqa: BLE001
                log.exception(
                    "sdlc: Chat unlock failed for %s",
                    run.get("run_id"),
                )

        Thread(target=_unlock, name="sdlc-unlock-chat", daemon=True).start()
    from sharedagents_web.dev_clarify import MAX_CLARIFY_ROUNDS, answered_rounds

    if answered_rounds(transcript) < MAX_CLARIFY_ROUNDS:
        if server_owned:
            # Choose exactly one next question from the answer just received.
            return _sdlc_clarify_round(run)
        # Chat is the one sequential interviewer. Calling `clarify()` here as
        # well produced a second form question before Chat's next card, so the
        # two models raced and the interview never converged. The append
        # above already reset questions and status.
        return jsonify({"run": _sdlc_view(run)})

    # The bounded interview is complete. Stop the Chat turn before its next
    # card can race the one synthesis pass below. A lost claim means the
    # contract is being frozen; a refused cancel means a possibly live
    # conductor. Either way the run is now `intake` with no question -- the
    # shape the stall recovery resumes from -- and a stale `ready` can no
    # longer be ratified, so nothing is stranded and nothing gets two owners.
    claimed, cancelled = _cancel_conductor_for_synthesis(
        run,
        str(getattr(g, "user_email", "") or ""),
    )
    if not (claimed and cancelled):
        log.info("sdlc: %s synthesis deferred to stall recovery (claimed=%s cancelled=%s)", run_id, claimed, cancelled)
        return jsonify({"run": _sdlc_view(sdlc.get(run_id) or run)})
    return _sdlc_clarify_round(run)


@bp.route("/api/dev/sdlc/runs/<run_id>/ratify", methods=["POST"])
@require_auth
def api_sdlc_ratify(run_id: str):
    """Freeze the contract, mint the issue, and start the pipeline."""
    from sharedagents_web.dev_sdlc import (
        contract_defects,
        get_sdlc_store,
    )

    sdlc = get_sdlc_store()
    run = sdlc.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    if run.get("issue_number"):
        return jsonify({"error": "already ratified", "run": _sdlc_view(run)}), 409
    payload = request.get_json(silent=True) or {}
    requirements = [
        str(item).strip()
        for item in (payload.get("requirements") or run.get("requirements") or [])
        if str(item).strip()
    ]
    criteria = [
        str(item).strip() for item in (payload.get("criteria") or run.get("criteria") or []) if str(item).strip()
    ]
    if not requirements or not criteria:
        return jsonify(
            {
                "error": "condition (for plan) and clearing evidence (for test) are both required",
            }
        ), 400
    defects = contract_defects(requirements, criteria)
    if defects:
        return jsonify(
            {
                "error": "contract is not freezeable",
                "detail": "; ".join(defects),
            }
        ), 400
    # The deploy the operator authorises HERE, before any code exists, is what
    # lets the rest of the run finish unattended: the pipeline merges and then
    # runs exactly this target. Omitting it keeps the old behaviour, where the
    # deploy is a separate arm + POST after the merge.
    from sharedagents_web.dev_sdlc import deploy_targets

    planned_deploy = str(payload.get("deploy_target") or "").strip()
    if planned_deploy and planned_deploy not in {row["id"] for row in deploy_targets(str(run.get("repo") or ""))}:
        return jsonify(
            {
                "error": f"unknown deploy target: {planned_deploy}",
            }
        ), 400
    # Intake is not a formality: an issue minted mid-interrogation is exactly
    # the ambiguous contract this flow exists to prevent. `ready` is the
    # clarifier's own verdict. Chat questions live on the conductor, not on
    # `run.questions`, so a live pending ask also blocks.
    from sharedagents_web.dev_clarify import ratification_blocked

    forced = bool(payload.get("force"))
    pending_ask = False
    approval_unknown = False
    try:
        found = _conductor_pending_asks(run)
    except Exception:  # noqa: BLE001 — fail closed: state is unknown
        found = None
    if found is None:
        approval_unknown = True
    else:
        pending_ask = bool(found)
    from sharedagents_web.dev_clarify import answered_rounds

    sid = str(run.get("conductor_session_id") or "").strip()
    conductor_busy = bool(sid and _conductor_tree_busy(sid, quiet=RATIFY_QUIET_SECONDS))
    blocked = ratification_blocked(
        status=str(run.get("status") or ""),
        force=forced,
        pending_ask=pending_ask,
        approval_unknown=approval_unknown,
        conductor_busy=conductor_busy,
        rounds=answered_rounds(run.get("transcript")),
    )
    if blocked:
        return jsonify(
            {
                "error": "intake is not finished",
                "detail": blocked,
            }
        ), 409
    if not forge_client.configured():
        return jsonify({"error": "issue_required", "detail": "The forge is not configured."}), 503
    # Freezing the contract and stopping the conductor for a synthesis are
    # mutually exclusive: the claim is held from here until the issue is on
    # the row, so no fallback can force-cancel the conductor in the moment the
    # pipeline starts dispatching stages to it. A synthesis holds the claim
    # only for its cancel, so the wait here is short; a claim that stays taken
    # is reported, never silently skipped.
    claim = secrets.token_hex(8)
    for _ in range(15):
        if sdlc.claim_intake(run_id, claim, seconds=RATIFY_CLAIM_SECONDS):
            break
        time.sleep(0.2)
    else:
        fresh = sdlc.get(run_id) or run
        if fresh.get("issue_number"):
            return jsonify({"error": "already ratified", "run": _sdlc_view(fresh)}), 409
        return jsonify(
            {
                "error": "intake is not finished",
                "detail": "The clarification is still being finalized. Try again in a moment.",
            }
        ), 409
    try:
        # The claim protects the row from here on; what was read BEFORE it is
        # only a preflight. Re-read under the claim and freeze that, or refuse:
        # an answer that landed meanwhile reopened the interview and the
        # operator must approve the scope that answer produced.
        fresh = sdlc.get(run_id) or run
        if fresh.get("issue_number"):
            return jsonify({"error": "already ratified", "run": _sdlc_view(fresh)}), 409
        blocked = ratification_blocked(
            status=str(fresh.get("status") or ""),
            force=forced,
            pending_ask=pending_ask,
            approval_unknown=approval_unknown,
            conductor_busy=conductor_busy,
            rounds=answered_rounds(fresh.get("transcript")),
        )
        if blocked:
            return jsonify({"error": "intake is not finished", "detail": blocked, "run": _sdlc_view(fresh)}), 409
        if list(fresh.get("transcript") or []) != list(run.get("transcript") or []):
            return jsonify(
                {
                    "error": "intake is not finished",
                    "detail": "A new answer arrived while you were approving. "
                    "Review the updated scope, then approve again.",
                    "run": _sdlc_view(fresh),
                }
            ), 409
        return _ratify_claimed(
            run_id,
            fresh,
            payload,
            forced=forced,
            planned_deploy=planned_deploy,
            requirements=requirements,
            criteria=criteria,
            claim=claim,
        )
    finally:
        sdlc.release_intake(run_id, claim)


def _ratify_claimed(
    run_id: str,
    run: dict,
    payload: dict,
    *,
    forced: bool,
    planned_deploy: str,
    requirements: list[str],
    criteria: list[str],
    claim: str = "",
):
    """The part of ratification that runs under the intake claim."""
    from sharedagents_web.dev_sdlc import (
        STAGE_TAGS,
        SdlcError,
        get_sdlc_store,
        issue_body,
    )
    from sharedagents_web.dev_sdlc_driver import start_run_thread

    sdlc = get_sdlc_store()
    repo = str(run.get("repo") or "")
    kind = str(run.get("kind") or "feature")
    headline = str(run.get("headline") or "")
    requested_branch = str(payload.get("branch") or "").strip()
    branch = requested_branch or branch_name(kind, headline)
    if (
        branch in {"main", "master"}
        or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]{0,126}[A-Za-z0-9])?", branch)
        or ".." in branch
        or "//" in branch
        or branch.endswith(".lock")
    ):
        return jsonify({"error": "invalid branch", "detail": branch}), 400
    publish_mode = str(payload.get("publish_mode") or "merge").strip().lower()
    if publish_mode not in {"pr", "merge"}:
        return jsonify({"error": "invalid publish mode", "detail": publish_mode}), 400
    requested_isolation = str(payload.get("test_isolation") or "docker").strip().lower()
    if requested_isolation != "docker":
        return jsonify({"error": "test isolation must be docker"}), 400
    # One branch, one worktree, one run. The claim is a unique index, not a
    # read-then-write check two concurrent ratifications could both pass.
    try:
        run = sdlc.claim_branch(run_id, branch) or run
    except SdlcError as error:
        return jsonify({"error": "branch is busy", "detail": str(error)}), 409
    body = issue_body(
        headline=headline,
        kind=kind,
        requirements=requirements,
        criteria=criteria,
        transcript=list(run.get("transcript") or []),
        run_id=run_id,
        branch=branch,
    )
    labels = [
        "type/bug" if kind == "fix" else "type/feature",
        "status/in-progress",
        STAGE_TAGS["intake"],
    ]
    status, issue = forge_client.create_issue(
        repo,
        title=headline,
        body=body,
        labels=labels,
    )
    # The forge answered inside the lease; extend it for the write below.
    if claim:
        sdlc.claim_intake(run_id, claim, seconds=RATIFY_CLAIM_SECONDS)
    number = issue.get("number") if isinstance(issue, dict) else None
    if status not in (200, 201) or not number:
        return jsonify(
            {
                "error": "issue_required",
                "detail": str((issue or {}).get("message") or f"forge HTTP {status}"),
            }
        ), 503
    fields: dict = {
        "requirements": requirements,
        "criteria": criteria,
        "questions": [],
        "status": "running",
        "issue_number": int(number),
        "issue_url": str(issue.get("html_url") or forge_client.issue_html_url(repo, number)),
        "last_error": "",
        "publish_mode": publish_mode,
        "test_isolation": "docker",
    }
    if planned_deploy:
        fields["deploy_target"] = planned_deploy
    run = sdlc.patch(run_id, **fields) or run
    sdlc.add_event(
        run_id,
        stage="intake",
        status="passed",
        note=(
            f"{len(requirements)} requirements, {len(criteria)} criteria ratified"
            + f" — branch `{branch}` — publish `{publish_mode}`"
            " — tests in isolated Docker"
            + (f" — deploy `{planned_deploy}` authorised for after the merge" if planned_deploy else "")
            + (" — FORCED past open questions by the operator" if forced else "")
        ),
    )
    # The contract is frozen, so the interview is over. Anything the conductor
    # is still parked on would block every stage dispatched to that session.
    closed = _close_conductor_interview(run)
    if closed:
        sdlc.add_event(
            run_id, stage="intake", status="passed", note=f"closed {closed} open interview question(s) at ratification"
        )

    # Mirror the run onto the Dev-loop board so one object is one card there too.
    try:
        loop = get_dev_registry().create(
            object=headline,
            done_when="\n".join(f"- {text}" for text in criteria),
            repo=repo,
            branch=branch,
            kind=kind,
            agent=str(run.get("agent") or "build"),
        )
        get_dev_registry().patch(
            str(loop["id"]),
            issue_number=int(number),
            issue_url=str(run.get("issue_url") or ""),
        )
        run = sdlc.patch(run_id, loop_id=str(loop["id"])) or run
    except Exception:  # noqa: BLE001 — the board mirror is not the contract
        log.warning("sdlc: dev-loop mirror failed for %s", run_id)

    start_run_thread(_sdlc_driver(str(g.user_email)), run_id)
    _guard_ratified_interview(run_id, str(g.user_email or ""))
    fresh = sdlc.get(run_id) or run
    try:
        from sharedagents_agent.tools.impl.sdlc import sync_board_after_ratify
        from sharedagents_core.kernel.layout import sessions_root

        sid = str(fresh.get("conductor_session_id") or "").strip()
        if sid:
            sync_board_after_ratify(sessions_root(), sid, fresh)
    except Exception:  # noqa: BLE001 — card sync must not fail ratify
        log.warning("sdlc: board sync after ratify failed for %s", run_id)
    return jsonify({"run": _sdlc_view(fresh)}), 201


@bp.route("/api/dev/sdlc/runs/<run_id>/resume", methods=["POST"])
@require_auth
def api_sdlc_resume(run_id: str):
    """Re-arm a blocked run after the operator has unblocked it."""
    from sharedagents_web.dev_sdlc import MAX_NO_PROGRESS_ATTEMPTS, get_sdlc_store
    from sharedagents_web.dev_sdlc_driver import run_is_active, start_run_thread

    sdlc = get_sdlc_store()
    run = sdlc.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    if not run.get("issue_number"):
        return jsonify({"error": "run has no issue yet"}), 409
    if run_is_active(run_id):
        return jsonify({"run": _sdlc_view(run), "started": False})
    payload = request.get_json(silent=True) or {}
    override = payload.get("first_artifact_budget_s")
    # P5: a stage that burned its budget twice with nothing to hand over is
    # not a transient failure. Restarting it a third time spends the same
    # tokens on the same dead end, so the line stops here until the operator
    # changes the policy: raise the budget (accepted below), edit the scope,
    # or abort.
    fused = int(run.get("no_progress_attempts") or 0) >= MAX_NO_PROGRESS_ATTEMPTS
    if fused and override is None:
        return jsonify(
            {
                "error": (
                    f"`{run.get('blocked_stage') or run.get('stage') or 'stage'}` "
                    f"failed {run.get('no_progress_attempts')}x with no artifact "
                    "and no progress. Raise the stage budget "
                    "(first_artifact_budget_s), edit the scope, or abort."
                ),
                "blocked_stage": run.get("blocked_stage") or "",
                "no_progress_attempts": int(run.get("no_progress_attempts") or 0),
                "requires": "first_artifact_budget_s",
            }
        ), 409
    if override is not None:
        try:
            budget = int(float(override))
        except (TypeError, ValueError):
            return jsonify({"error": "first_artifact_budget_s must be a number"}), 400
        if budget <= 0:
            return jsonify({"error": "first_artifact_budget_s must be positive"}), 400
        # A policy input, written before the driver can read it — not a
        # status/lifecycle write, so it does not race the lease claim. The
        # override is also the operator's statement that the situation
        # changed: the fuse counts identical failures, and this is no longer
        # one.
        sdlc.patch(run_id, first_artifact_budget_s=budget, no_progress_attempts=0)
        run = sdlc.get(run_id) or run
    # A ratified run whose conductor is parked on an unanswerable question
    # cannot make progress no matter how often it is resumed: every stage is
    # dispatched to that session and queues behind the ask. Clear it first, or
    # the resume is a no-op that looks like one more silent stall.
    _close_conductor_interview(run)
    # An operator resuming IS the signal that the situation changed -- they
    # looked, and they decided this run can go on. Without restoring the
    # recovery budget, a run that had exhausted it stays exhausted, so every
    # subsequent block needs another human: the intervention the supervisor
    # exists to avoid. The lift is itself budgeted, because this route cannot
    # prove a human is behind the request.
    # The lease claim comes FIRST and nothing is written before it. Writing the
    # status and putting it back on failure let a losing request restore its
    # stale value over the winner's live run: the loser must not touch
    # the run at all. `run()` gates only on "aborted", so the driver is happy to
    # start while the row still says blocked.
    started = start_run_thread(_sdlc_driver(str(g.user_email)), run_id)
    if not started:
        return jsonify(
            {
                "run": _sdlc_view(run),
                "started": False,
                "recovery_budget_restored": False,
                "error": "another worker already owns this run",
            }
        ), 409
    # The status is set by start_run_thread, under the lease and before the
    # driver can write anything. Writing it here would clobber whatever the
    # driver has since decided.
    lift = sdlc.operator_resume(run_id, actor=str(g.user_email))
    return jsonify(
        {
            "run": _sdlc_view(sdlc.get(run_id) or run),
            "started": True,
            "recovery_budget_restored": bool(lift.get("lifted")),
        }
    )


@bp.route("/api/dev/sdlc/runs/<run_id>/builder", methods=["POST"])
@require_auth
def api_sdlc_builder(run_id: str):
    """Change which agent and model build a run that has not finished.

    A run blocked because its model was unreachable, too weak, or out of credit
    is otherwise unrecoverable: the choice was fixed at intake and the operator
    would have to describe the whole change again.
    """
    from sharedagents_web.dev_sdlc import get_sdlc_store, validate_role_agents
    from sharedagents_web.dev_sdlc_driver import run_is_active

    sdlc = get_sdlc_store()
    run = sdlc.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    if str(run.get("status")) in ("done", "aborted"):
        return jsonify({"error": "this run has finished"}), 409
    if run_is_active(run_id):
        return jsonify(
            {
                "error": "the run is working",
                "detail": "Abort or wait for the current stage before changing the builder.",
            }
        ), 409
    payload = request.get_json(silent=True) or {}
    fields: dict = {}
    for role in ("validation", "generation", "verification"):
        agent_key, model_key = f"{role}_agent", f"{role}_model"
        if agent_key in payload:
            fields[agent_key] = str(payload.get(agent_key) or "").strip()
        if model_key in payload:
            fields[model_key] = str(payload.get(model_key) or "").strip()
    # Orchestrate is always Jarvis. Model stays selectable; ignore agent swaps.
    if "generation_agent" in fields or "agent" in payload:
        fields["generation_agent"] = "jarvis"
    if "model" in payload and "generation_model" not in fields:
        fields["generation_model"] = str(payload.get("model") or "").strip()
    if not fields:
        return jsonify({"error": "nothing to change"}), 400
    proposed = {
        f"{role}_agent": fields.get(f"{role}_agent", run.get(f"{role}_agent"))
        for role in ("validation", "generation", "verification")
    }
    proposed["generation_agent"] = "jarvis"
    proposed.update(
        {
            f"{role}_model": fields.get(f"{role}_model", run.get(f"{role}_model"))
            for role in ("validation", "generation", "verification")
        }
    )
    try:
        validate_role_agents(proposed)
    except Exception as error:
        return jsonify({"error": str(error)}), 400
    if fields.get("generation_agent") == "jarvis" and run.get("generation_agent") != "jarvis":
        # Sessions are owned by the selected agent.  Reusing one after a
        # role change can silently dispatch the next stage to the old
        # executor through the gateway's session affinity.
        fields["author_session_id"] = ""
        fields["session_id"] = ""
        fields["agent"] = "jarvis"
    elif "generation_agent" in fields:
        fields["agent"] = "jarvis"
    if "generation_model" in fields:
        fields["model"] = fields["generation_model"]
    return jsonify({"run": _sdlc_view(sdlc.patch(run_id, **fields) or run)})


@bp.route("/api/dev/sdlc/runs/<run_id>/prompt", methods=["POST"])
@require_auth
def api_sdlc_prompt(run_id: str):
    """Send a follow-up on the conductor after the operator stops a question."""
    from sharedagents_web.dev_sdlc import get_sdlc_store
    from sharedagents_web.routes.sessions import _resolve_gw_hash

    sdlc = get_sdlc_store()
    run = sdlc.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    if run.get("status") not in ("intake", "ready") or run.get("issue_number"):
        return jsonify({"error": "intake is closed for this run"}), 409
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("prompt") or payload.get("detail") or "").strip()
    if not text:
        return jsonify({"error": "prompt required"}), 400
    sid = str(run.get("conductor_session_id") or "").strip()
    if sid:
        gw, err = _resolve_gw_hash(sid)
        if err is None and gw:
            gateway_request(
                "POST",
                f"/control/v1/orchestrator/sessions/{gw}/cancel",
                json={"force": True},
                headers=gateway_principal_headers(g.user_email),
                timeout=15,
            )
    _prompt_conductor(run, text)
    return jsonify({"run": _sdlc_view(run), "accepted": True})


@bp.route("/api/dev/sdlc/runs/<run_id>/abort", methods=["POST"])
@require_auth
def api_sdlc_abort(run_id: str):
    from sharedagents_web.dev_sdlc import get_sdlc_store

    sdlc = get_sdlc_store()
    if not sdlc.get(run_id):
        return jsonify({"error": "unknown run"}), 404
    run = sdlc.patch(run_id, status="aborted", stage_status="", last_error="aborted by operator")
    return jsonify({"run": _sdlc_view(run or {})})


@bp.route("/api/dev/sdlc/runs/<run_id>/merge", methods=["POST"])
@require_auth
def api_sdlc_merge(run_id: str):
    """Operator override of the run's single PR. Pipeline merge is automatic."""
    from sharedagents_web.dev_sdlc import STAGE_TAGS, apply_merge, get_sdlc_store

    payload = request.get_json(silent=True) or {}
    forced = payload.get("force") is True or str(payload.get("force")).strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }
    result = apply_merge(
        run_id,
        method=str(payload.get("method") or "merge"),
        force=forced,
        actor="operator",
    )
    run = result.get("run") or get_sdlc_store().get(run_id) or {}
    if not result.get("ok"):
        body = {"error": result.get("error") or "merge failed", "run": _sdlc_view(run)}
        for key in ("blockers", "warnings", "facts"):
            if key in result:
                body[key] = result[key]
        return jsonify(body), int(result.get("http") or 409)
    overrode = str(result.get("overrode") or "")
    pr_number = result.get("pr_number") or run.get("pr_number")
    repo = str(run.get("repo") or "")
    try:
        _Forge().comment(
            repo,
            int(run.get("issue_number") or 0),
            f"**`{STAGE_TAGS['merged']}`** — PR #{pr_number} merged by the operator."
            + (f"\n\n> Release validation was OVERRIDDEN. Blockers at merge time: {overrode}" if overrode else ""),
        )
        _Forge().label(repo, int(run.get("issue_number") or 0), [STAGE_TAGS["merged"]])
    except Exception:  # noqa: BLE001
        pass
    return jsonify({"run": _sdlc_view(get_sdlc_store().get(run_id) or run)})


@bp.route("/api/dev/sdlc/runs/<run_id>", methods=["DELETE"])
@bp.route("/api/dev/sdlc/runs/<run_id>/archive", methods=["POST", "DELETE"])
@require_auth
def api_sdlc_archive(run_id: str):
    """Manual delete: close the issue, delete the branch, remove the Dev row."""
    from sharedagents_web.dev_sdlc import STAGE_TAGS, delete_run

    def _cancel_bound_sessions(run: dict) -> list[str]:
        from sharedagents_web.dev_sdlc import get_sdlc_store

        session_ids = {
            str(run.get(key) or "").strip()
            for key in (
                "session_id",
                "conductor_session_id",
                "author_session_id",
                "review_session_id",
            )
        }
        session_ids.update(
            str(event.get("executor_session_id") or "").strip()
            for event in get_sdlc_store().events(str(run.get("run_id") or ""))
        )
        session_ids.discard("")
        cancelled = 0
        failed = 0
        for sid in sorted(session_ids):
            resp = gateway_request(
                "POST",
                f"/control/v1/orchestrator/sessions/{sid}/cancel",
                json={"force": True},
                headers=gateway_principal_headers(g.user_email),
                timeout=15,
            )
            if resp is not None and resp.status_code < 400:
                cancelled += 1
            else:
                failed += 1
        notes = [f"cancelled {cancelled} run session(s)"] if cancelled else []
        if failed:
            notes.append(f"could not cancel {failed} run session(s)")
        return notes

    result = delete_run(
        run_id,
        actor="operator",
        cancel_sessions=_cancel_bound_sessions,
    )
    snapshot = result.get("snapshot") or result.get("run") or {}
    if not result.get("ok"):
        body = {"error": result.get("error") or "delete failed", "run": _sdlc_view(snapshot) if snapshot else {}}
        for key in ("blockers", "warnings", "facts"):
            if key in result:
                body[key] = result[key]
        return jsonify(body), int(result.get("http") or 409)
    try:
        _Forge().comment(
            str(snapshot.get("repo") or ""),
            int(snapshot.get("issue_number") or 0),
            f"**`{STAGE_TAGS['archived']}`** — Dev run deleted by the operator. "
            "Issue closed; branch removed."
            + ((" " + "; ".join(result.get("notes") or [])) if result.get("notes") else ""),
        )
        _Forge().label(
            str(snapshot.get("repo") or ""),
            int(snapshot.get("issue_number") or 0),
            [STAGE_TAGS["archived"]],
        )
    except Exception:  # noqa: BLE001
        pass
    return jsonify(
        {
            "deleted": True,
            "run_id": run_id,
            "run": None,
            "notes": result.get("notes") or [],
        }
    )


@bp.route("/api/dev/sdlc/runs/<run_id>/arm", methods=["POST"])
@require_auth
def api_sdlc_arm(run_id: str):
    """Arm this run's deploy for a short window, once.

    The UI toggle is an affordance. This is the fact: a token the deploy call
    must present, which expires on its own and is consumed on use, so a
    production restart can never be a single replayable POST.
    """
    import secrets
    from datetime import datetime, timedelta, timezone

    from sharedagents_web.dev_sdlc import get_sdlc_store

    sdlc = get_sdlc_store()
    run = sdlc.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    if not run.get("merged"):
        return jsonify({"error": "merge before arming a deploy"}), 409
    token = secrets.token_urlsafe(18)
    until = datetime.now(timezone.utc) + timedelta(seconds=DEPLOY_ARM_SECONDS)
    sdlc.patch(run_id, deploy_armed_token=token, deploy_armed_until=until.isoformat())
    return jsonify(
        {
            "armed_token": token,
            "expires_at": until.isoformat(),
            "expires_in": DEPLOY_ARM_SECONDS,
            "run": _sdlc_view(sdlc.get(run_id) or run),
        }
    )


def launch_deploy(run_id: str, target: str, *, forced: bool = False, actor: str = "operator") -> dict:
    """Run the gate checks and launch one deploy unit. No arm check here.

    Split out of the route so the PIPELINE can run a deploy the operator chose
    during the plan phase without a second visit to the tab -- and run it
    through exactly the same refusals. The arm token stays where it was: it
    guards the HTTP surface, so a production restart is never one replayable
    POST. A pipeline deploy is authorised instead by the target the operator
    ratified before any code existed, which is a recorded decision rather than
    a click after the fact.

    Returns ``{"ok", "error", "http", "unit", "run", "verify_required"}``.
    """
    import subprocess
    import time

    from sharedagents_web.dev_sdlc import (
        STAGE_TAGS,
        deploy_command,
        deploy_verify_command,
        get_sdlc_store,
        unreviewed_stages,
    )

    sdlc = get_sdlc_store()
    run = sdlc.get(run_id) or {}
    repo = str(run.get("repo") or "")
    # A forced merge carries its blockers PAST the merge gate, so the deploy
    # must ask again rather than inheriting the override. Otherwise
    # force-merge -> arm -> deploy ships work no gate ever reviewed
    # Deploying anyway needs its own explicit decision.
    # The identity that matters is what was MERGED, not what was reviewed at
    # some earlier point: a force-push between review and merge would otherwise
    # let an H1 decision vouch for H2.
    merged_at = str(run.get("merged_head_sha") or "")
    outstanding = unreviewed_stages(sdlc.events(run_id), merged_at)
    unusable = len(outstanding)
    if not merged_at and int(run.get("gate_unusable_count") or 0):
        # A forced merge can land without the forge naming the commit. When the
        # run carries marks, "no identity, therefore no mismatch, therefore
        # fine" is exactly backwards -- there is nothing to prove it WITH
        # Runs with no marks are unaffected, including every run
        # merged before this identity was recorded at all.
        outstanding = [
            "the merged commit was never recorded, so the gates that "
            "decided nothing cannot be shown to have been reviewed"
        ]
        unusable = len(outstanding)
    if unusable and not forced:
        sdlc.add_event(
            run_id,
            stage="deployed",
            status="failed",
            note=f"refused: never reviewed against this commit: {', '.join(outstanding)}",
        )
        return {
            "ok": False,
            "http": 409,
            "error": "this work advanced past a gate that decided nothing and "
            "has not been reviewed since; re-run those stages, or "
            "force this deploy deliberately",
            "unreviewed_stages": outstanding,
            "run": run,
        }
    if unusable:
        sdlc.add_event(
            run_id,
            stage="deployed",
            status="started",
            note=f"FORCED past unreviewed stage(s): {', '.join(outstanding)}",
        )

    inner = deploy_command(repo, target)
    if not inner:
        return {"ok": False, "http": 400, "run": run, "error": f"`{repo}` declares no command for `{target}`"}
    # ``systemd-run`` detaches the work from this request's cgroup so a restart
    # of sharedagents-web cannot kill the deploy it just launched.
    # Recorded so the probe can tell "the service restarted" from "a service is
    # answering" -- the process being replaced answers right up to the moment it
    # is replaced.
    started_at = time.time()
    unit = f"sharedagents-sdlc-deploy-{run_id}-{int(started_at)}"
    command = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--collect",
        "/bin/bash",
        "-lc",
        inner,
    ]
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30, env=env)
    except (OSError, subprocess.SubprocessError) as error:
        run = sdlc.patch(run_id, deploy_target=target, deploy_status=f"failed: {error}", deploy_verified="") or run
        return {"ok": False, "http": 502, "error": str(error), "run": run}
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:400]
        run = sdlc.patch(run_id, deploy_target=target, deploy_status=f"failed: {detail}", deploy_verified="") or run
        return {"ok": False, "http": 502, "run": run, "error": detail or "systemd-run failed"}
    run = (
        sdlc.patch(
            run_id,
            deploy_target=target,
            deploy_status=f"launched:{unit}",
            deploy_verified="",
            deploy_started_at=str(int(started_at)),
        )
        or run
    )
    try:
        if run.get("issue_number"):
            _Forge().comment(
                str(run.get("repo") or ""),
                int(run["issue_number"]),
                f"**`{STAGE_TAGS['deployed']}`** — deploy `{target}` launched as `{unit}` by the {actor}.",
            )
    except Exception:  # noqa: BLE001
        pass
    # `systemd-run` exiting 0 means the unit was ACCEPTED, not that the deploy
    # worked -- and this request cannot wait, because the deploy restarts the
    # process serving it. So the launch is recorded as unverified and VERIFY is
    # a separate call the page makes once the web is answering again.
    sdlc.add_event(
        run_id, stage="deployed", status="started", note=f"target={target} unit={unit} — launched, not yet verified"
    )
    return {
        "ok": True,
        "http": 200,
        "run": sdlc.get(run_id) or run,
        "unit": unit,
        "verify_required": bool(deploy_verify_command(repo, target)),
    }


@bp.route("/api/dev/sdlc/runs/<run_id>/deploy", methods=["POST"])
@require_auth
def api_sdlc_deploy(run_id: str):
    """Operator-chosen deploy. Detached: restarting web kills this process."""
    from sharedagents_web.dev_sdlc import deploy_targets, get_sdlc_store

    sdlc = get_sdlc_store()
    run = sdlc.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    payload = request.get_json(silent=True) or {}
    target = str(payload.get("target") or "none").strip()
    repo = str(run.get("repo") or "")
    if target not in {row["id"] for row in deploy_targets(repo)}:
        return jsonify({"error": f"unknown deploy target: {target}"}), 400
    if target == "none":
        run = sdlc.patch(run_id, deploy_target="none", deploy_status="skipped") or run
        return jsonify({"run": _sdlc_view(run)})
    if not run.get("merged"):
        return jsonify({"error": "merge before deploying"}), 409
    deploy_forced = payload.get("force") is True or str(payload.get("force") or "").strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }
    # The arm is a server-held, expiring, one-shot token. A client-asserted
    # "armed": true would make a production restart one replayable POST.
    from datetime import datetime, timezone

    supplied = str(payload.get("armed_token") or "")
    try:
        expires = datetime.fromisoformat(str(run.get("deploy_armed_until") or ""))
    except ValueError:
        expires = None
    fresh = bool(expires and expires > datetime.now(timezone.utc))
    # Consume FIRST and atomically. Checking then clearing let two concurrent
    # deploys both validate the same token before either spent it.
    if not fresh or not sdlc.consume_deploy_arm(run_id, supplied):
        return jsonify(
            {
                "error": "deploy is not armed",
                "detail": f"Arm the deploy, then run it within {DEPLOY_ARM_SECONDS}s.",
            }
        ), 409

    result = launch_deploy(run_id, target, forced=deploy_forced, actor="operator")
    if not result.get("ok"):
        body = {"error": result.get("error"), "run": _sdlc_view(result.get("run") or run)}
        if "unreviewed_stages" in result:
            body["unreviewed_stages"] = result["unreviewed_stages"]
        return jsonify(body), int(result.get("http") or 409)
    return jsonify(
        {
            "run": _sdlc_view(result.get("run") or run),
            "unit": result.get("unit"),
            "verify_required": result.get("verify_required"),
        }
    )


#: How long after a launch a failing probe still means "not yet".
DEPLOY_VERIFY_GRACE = 90.0


@bp.route("/api/dev/sdlc/runs/<run_id>/deploy/verify", methods=["POST"])
@require_auth
def api_sdlc_deploy_verify(run_id: str):
    """VERIFY, the closing bookend: did the deploy actually take?

    Run after the deploy, from a request the deploy did not kill. The target's
    own ``verify`` command is the verdict; a target that declares none can only
    report ``unverified``, never a pass.
    """
    import math
    import subprocess
    import time

    from sharedagents_web.dev_sdlc import STAGE_TAGS, deploy_verify_command, get_sdlc_store

    sdlc = get_sdlc_store()
    run = sdlc.get(run_id)
    if not run:
        return jsonify({"error": "unknown run"}), 404
    repo = str(run.get("repo") or "")
    target = str(run.get("deploy_target") or "")
    if not target or target == "none":
        return jsonify({"error": "this run has no deploy to verify"}), 409
    status_now = str(run.get("deploy_status") or "")
    if not status_now.startswith("launched:"):
        # A launch that failed still recorded its target. Verifying that would
        # let a healthy service vouch for a deploy that never ran.
        return jsonify(
            {
                "error": "the last deploy did not launch, so there is nothing to verify",
                "deploy_status": status_now or "(never deployed)",
            }
        ), 409

    try:
        since = float(str(run.get("deploy_started_at") or "").strip())
    except (TypeError, ValueError):
        since = 0.0
    # `float()` happily parses "nan" and "inf". NaN fails every comparison, so
    # it slips past `<= 0` and then `int()` raises; Infinity would make the
    # grace window eternal. Anything not finite is an unknown launch time.
    if not math.isfinite(since) or since <= 0:
        # SINCE=0 makes the "did it restart?" half of the probe vacuous, so a
        # pre-existing process would vouch for this deploy. Unknown is not OK.
        sdlc.patch(run_id, deploy_verified="unverified")
        return jsonify(
            {
                "error": "this deploy did not record when it started, so a restart "
                "cannot be told from a service that was already running",
                "verified": False,
                "unverified": True,
                "run": _sdlc_view(sdlc.get(run_id) or run),
            }
        ), 409

    probe = deploy_verify_command(repo, target)
    if not probe:
        sdlc.patch(run_id, deploy_verified="unverified")
        sdlc.add_event(
            run_id,
            stage="deployed",
            status="failed",
            note=f"`{target}` declares no verify command, so the deploy cannot be proved",
        )
        return jsonify(
            {
                "verified": False,
                "unverified": True,
                "detail": f"`{target}` declares no verify command",
                "run": _sdlc_view(sdlc.get(run_id) or run),
            }
        )
    env = dict(os.environ)
    env["SHAREDAGENTS_DEPLOY_SINCE"] = str(int(since))
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    try:
        proc = subprocess.run(["/bin/bash", "-lc", probe], env=env, capture_output=True, text=True, timeout=90)
        ok = proc.returncode == 0
        detail = (proc.stdout or proc.stderr or "").strip()[:400]
    except (OSError, subprocess.SubprocessError) as error:
        ok, detail = False, str(error)[:400]

    # A detached deploy takes time. Until the grace period is up, "not yet" is
    # a retry rather than a verdict -- reporting failure early would be wrong
    # about a deploy that was merely still running.
    if not ok and time.time() - since < DEPLOY_VERIFY_GRACE:
        return jsonify(
            {
                "pending": True,
                "verified": False,
                "detail": f"waiting for `{target}` to come up",
                "run": _sdlc_view(run),
            }
        )

    sdlc.patch(run_id, deploy_verified="passed" if ok else f"failed: {detail}")
    sdlc.add_event(
        run_id,
        stage="deployed",
        status="passed" if ok else "failed",
        note=(
            f"verified: `{target}` is serving" if ok else f"deploy did NOT verify: {detail or 'probe exited non-zero'}"
        ),
    )
    try:
        if run.get("issue_number"):
            _Forge().comment(
                repo,
                int(run["issue_number"]),
                f"**`{STAGE_TAGS['deployed']}`** — deploy `{target}` "
                + (
                    "verified: the service is answering."
                    if ok
                    else f"FAILED verification: {detail or 'probe exited non-zero'}"
                ),
            )
    except Exception:  # noqa: BLE001
        pass
    return jsonify({"verified": ok, "detail": detail, "run": _sdlc_view(sdlc.get(run_id) or run)})


# A run's driver is a daemon thread, so it dies with this process. Sweep once
# on the first request rather than at import: the owner identity and the
# gateway are only reliable inside an application context.
_recovery_done = False
_recovery_lock = __import__("threading").Lock()
_last_supervise = 0.0
#: How often the supervisor sweeps for blocked runs after dependencies are
#: ready. During boot it polls readiness every two seconds instead.
SUPERVISE_EVERY_S = 120.0


def start_sdlc_supervisor() -> bool:
    """SDLC supervisor removed — never start a blocked-run sweep."""
    return False


@bp.after_app_request
def _sdlc_recover_once(response):
    """Re-arm unfinished runs, then keep sweeping for blocked ones.

    Two jobs share this hook:

    * ONCE per process, `recover_runs` re-arms everything unfinished — the
      driver is a daemon thread and dies with the web process.
    * PERIODICALLY, `supervise_runs` restarts runs that BLOCKED while the
      server was up. Without it a blocked run waits for a human, which is
      exactly the intervention this pipeline exists to remove.

    AFTER the request, not before: ``@require_auth`` populates ``g.user_email``
    inside the view, so a ``before_request`` hook always saw it empty and the
    recovery never ran at all.
    """
    return response
