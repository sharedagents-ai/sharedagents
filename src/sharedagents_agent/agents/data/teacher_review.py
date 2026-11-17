"""sharedagents teacher-model — verdict engine (stdlib only, pure library).

The shared review brain. Its ONLY entry point is the pure callable
:func:`run_review`, which the engine imports and calls IN-PROCESS — no
shell, no subprocess, no hooks (see ``local_llm.py::_teacher_review_native``
and ``teacher_supervisor``). It:

  1. reads the session's transcript + prior sidecar state,
  2. resolves the ORIGINAL GOAL (passed explicitly, else the first user
     message),
  3. builds a Tier-1 digest (and Tier-2 when a stop would be approved or
     the situation is ambiguous — the anti-false-done gate, spec §3.1),
  4. calls the teacher endpoint (OpenAI-compatible /v1/chat/completions),
  5. parses a strict JSON verdict, updates the per-session sidecar,
  6. RETURNS the control dict — ``{"decision":"block","reason":...}`` to
     re-drive the small model with a continuation, or ``{"continue":true}``
     to allow a verified stop.

Endpoint / model / key / thresholds / prompt arrive via the ``settings``
dict the caller builds from the agent config (``cfg.teacher_model()`` +
policy). There is NO environment / config-file / stdin input here.

Zero third-party deps. Any internal failure fails OPEN
(``{"continue":true}``) so a turn is never wedged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from sharedagents_model.sampling import apply_sampling, configured_sampling, sampling_row

# ---- event → kind -------------------------------------------------------
# The event slug passed by the engine.
#
# FIRE_BACK_EVENTS: a "not done" verdict re-drives the small model with a
#   continuation prompt (returns ``decision:block``). These are the events
#   the engine fires a continuation BACK into.
# STOP_ATTEMPT_EVENTS: the model is trying to END the turn → the anti-false-
#   done gate forces a Tier-2 (deep) review before a completion can be
#   approved.
# DEEP_REVIEW_EVENTS: proactive mid-turn triggers that always need Tier-2
#   evidence even though they are NOT stop attempts — ``tool-errors`` fires
#   after a run of failed tool calls and the teacher must see the error
#   bodies to diagnose them.
FIRE_BACK_EVENTS = {"stop", "premature-end", "reasoning-loop", "pre_idle", "periodic", "tool-errors", "no-result"}
STOP_ATTEMPT_EVENTS = {"stop", "premature-end", "reasoning-loop", "pre_idle"}
DEEP_REVIEW_EVENTS = {"tool-errors"}


# ---- transcript fold (minimal, stdlib) ----------------------------------
def _read_events(path: str) -> list[dict]:
    """Read a sharedagents JSONL transcript into a list of event dicts.

    Tolerant: skips blank/garbage lines, accepts either a JSONL stream or
    a single JSON object with an ``events``/``messages`` list.
    """
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    # Single-JSON-object form?
    stripped = text.lstrip()
    if (
        stripped.startswith("{")
        and '"events"' in stripped[:200]
        or stripped.startswith("{")
        and '"messages"' in stripped[:200]
    ):
        try:
            obj = json.loads(text)
            return list(obj.get("events") or obj.get("messages") or [])
        except json.JSONDecodeError:
            pass
    # sharedagents session JSONL: each line is "<crc8-hex> <json>" (see
    # sharedagents_session.session_reader.encode_line). Strip the optional CRC
    # prefix before parsing; plain JSONL (no prefix) still works.
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[0-9a-fA-F]{6,8}\s+", "", line)
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _is_injected_continuation(text: str) -> bool:
    """True for the framework's continuation-echo markers streamed into the
    content channel when the teacher (or a Stop hook) re-drives a turn
    (``"[Stop hook requested continuation] …"``). These are SUPERVISOR
    instructions, not assistant output, and must be excluded from the digest
    the teacher reviews — otherwise the teacher reads its own past
    continuations back as the assistant parroting them."""
    s = (text or "").lstrip()
    return s.startswith("[Stop hook requested continuation]") or s.startswith("[Stop hook]")


# A user "message" that is wholly an ASR / harness placeholder sentinel —
# e.g. "[segment 1 unavailable]" emitted when a voice segment couldn't be
# transcribed, or "[content unavailable]". These carry NO user intent and
# must never become the (implicit) goal: anchoring supervision to one made
# the teacher invent un-satisfiable subgoals and re-drive forever (session
# <session-id>). Matches only a string that is ENTIRELY
# a single bracketed "…unavailable…" token, so a real sentence that merely
# mentions "unavailable" is untouched.
_PLACEHOLDER_USER_RE = re.compile(r"^\[[^\]]*\bunavailable\b[^\]]*\]$", re.IGNORECASE)


def _is_placeholder_user(text: str) -> bool:
    """True for ASR/harness 'segment unavailable' sentinels (not real intent)."""
    return bool(_PLACEHOLDER_USER_RE.match((text or "").strip()))


def _is_real_user_text(text) -> bool:
    """A genuine user ask: non-empty, not a placeholder sentinel, and not a
    supervisor/Stop-hook continuation echoed into the user channel."""
    if not isinstance(text, str) or not text.strip():
        return False
    return not _is_placeholder_user(text) and not _is_injected_continuation(text)


def _is_teacher_node(ev: dict) -> bool:
    """True for the teacher's OWN review machinery in the transcript. Every
    review is rendered as a ``delegate_teacher`` tool node whose ``call_id`` is
    prefixed ``"teacher-"`` (tool_call + tool_result + tool_result_meta share
    it). The supervisor must never see its own interventions: folding them in
    made the teacher read its past reviews as the ASSISTANT loop-calling a
    tool, so it fired endless "stop calling delegate_teacher" continuations the
    model could never satisfy (an infinite stop-cap loop)."""
    if ev.get("name") == "delegate_teacher":
        return True
    cid = ev.get("call_id") or ""
    return isinstance(cid, str) and cid.startswith("teacher-")


def _digest(events: list[dict], *, deep: bool, max_tool_body: int = 1200) -> dict:
    """Fold events into a compact digest for the teacher.

    Tier-1 (deep=False): message boundaries, round count, final assistant
    text, and tool names + ok/err status only.
    Tier-2 (deep=True): also include (truncated) tool_result bodies.

    The teacher's own review artifacts are filtered out (see
    ``_is_teacher_node`` / ``_is_injected_continuation``) so the supervisor
    never reviews its own interventions.
    """
    user_msgs: list[str] = []
    assistant_final = ""
    text_delta_buf: list[str] = []
    rounds = 0
    tools: list[dict] = []
    # call_id → the single row that call owns (see the tool-event fold below).
    _tool_rows: dict[str, dict[str, Any]] = {}
    # Current-turn tracking. ``current_goal`` is the latest REAL user ask and
    # ``current_turn_index`` the turn it belongs to — together they are the
    # implicit goal + scope key the supervisor uses when no explicit ``/goal``
    # is set, so supervision tracks the CURRENT turn instead of anchoring to
    # the session's first-ever message.
    current_goal = ""
    current_turn_index = 0
    _ordinal = -1  # fallback turn counter when events carry no `turn_idx`
    for ev in events:
        t = ev.get("type") or ev.get("event") or ""
        role = ev.get("role")
        # The teacher's OWN review nodes must never enter the digest it reviews.
        if _is_teacher_node(ev):
            continue
        # sharedagents schema: the user goal for a turn rides on `turn_started.user`.
        if t == "turn_started":
            _ordinal += 1
            _ti = ev.get("turn_idx")
            current_turn_index = _ti if isinstance(_ti, int) else _ordinal
            # A new turn starts with NO ask: clear the carried-over goal so a
            # placeholder/empty current turn doesn't inherit a prior turn's ask
            # (which would supervise a stale goal instead of going observe-only).
            current_goal = ""
            txt = ev.get("user") or ""
            # Skip placeholder sentinels / supervisor echoes — never intent.
            if _is_real_user_text(txt):
                user_msgs.append(txt.strip())
                current_goal = txt.strip()
            # A new turn supersedes any earlier turn's assistant output. Reset
            # the per-turn accumulators so the digest judges the CURRENT turn's
            # deliverable — not a stale prior `turn_ended.final_content`. This
            # matters at a `pre_idle` review, which fires BEFORE the current
            # turn writes its own `turn_ended`: the report exists only as
            # streamed `text_delta`, and without this reset a prior turn's
            # final_content (e.g. an earlier "hi" greeting) would win, making
            # the teacher judge the report request against the wrong message
            # and wrongly fire a continuation.
            assistant_final = ""
            text_delta_buf = []
        elif t in ("user_message", "user") or role == "user":
            txt = ev.get("content") or ev.get("text") or ev.get("user") or ""
            if _is_real_user_text(txt):
                _ti = ev.get("turn_idx")
                if isinstance(_ti, int):
                    current_turn_index = _ti
                else:
                    _ordinal += 1
                    current_turn_index = _ordinal
                user_msgs.append(txt.strip())
                current_goal = txt.strip()
        elif t == "round_ended":
            rounds += 1
        elif t == "text_delta":
            # visible-content stream (thinking_delta is the reasoning channel
            # and is intentionally excluded — the teacher judges the answer).
            # Framework-injected continuation echoes ("[Stop hook requested
            # continuation] …") are the teacher's own instructions, not
            # assistant output — drop them. They only reach the digest via this
            # fallback path (used at pre_idle, before turn_ended is written).
            d = ev.get("delta")
            if isinstance(d, str) and not _is_injected_continuation(d):
                text_delta_buf.append(d)
        elif t in ("assistant_message", "final_content", "turn_ended") or role == "assistant":
            txt = ev.get("final_content") or ev.get("content") or ev.get("text") or ""
            if isinstance(txt, str) and txt.strip():
                assistant_final = txt.strip()
        elif t in ("tool_call", "tool_result", "tool_result_meta"):
            # ONE row per TOOL CALL. The transcript splits a single call across
            # three events sharing one ``call_id``: ``tool_call`` (name/args),
            # ``tool_result_meta`` (status) and ``tool_result`` (body). Appending
            # a row per EVENT reported 15 calls as 40 rows, 27 of them
            # permanently "pending", which read to the teacher as unfinished
            # work — a completed task was judged "unverified" (observed live
            # 2026-09-10). Merge by ``call_id``; an id-less legacy event appends.
            _cid = str(ev.get("call_id") or "").strip()
            _name = str(ev.get("name") or ev.get("tool") or "").strip()
            _fallback = tools[-1]["name"] if tools else "?"
            entry = _tool_rows.get(_cid) if _cid else None
            if entry is None:
                entry = {"name": _name or _fallback or "?", "status": "pending"}
                tools.append(entry)
                if _cid:
                    _tool_rows[_cid] = entry
            elif _name:
                entry["name"] = _name
            if t in ("tool_result", "tool_result_meta"):
                # Status sources, in priority order:
                #   err  — an explicit error flag (``is_error``/``error``
                #          truthy) or the sharedagents JSONL ``"ok": false`` on
                #          ``tool_result_meta`` (session_reader writes ``ok``
                #          there, NOT ``is_error``). STICKY: a later statusless
                #          body must never upgrade an error to a success;
                #   ok   — an explicit success marker (``ok`` True or
                #          ``is_error`` explicitly False — older transcripts mark
                #          success that way), so a real success BREAKS the
                #          trailing-error streak;
                #   (a statusless ``tool_result`` body leaves the row's status
                #    untouched, so the meta verdict that precedes it survives.)
                if ev.get("is_error") or ev.get("error") or ev.get("ok") is False:
                    entry["status"] = "err"
                elif (ev.get("ok") is True or ev.get("is_error") is False) and entry["status"] != "err":
                    entry["status"] = "ok"
                if deep:
                    body = ev.get("content") or ev.get("output") or ev.get("result") or ""
                    if isinstance(body, (dict, list)):
                        body = json.dumps(body)[: max_tool_body * 2]
                    body = str(body)
                    if body:
                        if len(body) > max_tool_body:
                            head = body[: max_tool_body // 2]
                            tail = body[-max_tool_body // 2 :]
                            body = f"{head}\n…[truncated]…\n{tail}"
                            entry["truncated"] = True
                        entry["body"] = body
    # Fall back to the accumulated visible-content stream when no explicit
    # final_content event was present (e.g. a turn still streaming, or an
    # engine that emits only deltas).
    if not assistant_final and text_delta_buf:
        assistant_final = "".join(text_delta_buf).strip()
    # tool-failure streak: trailing run of "err" results. A "pending" status
    # (a statusless ``tool_result`` body row, or an unresolved ``tool_call``) is
    # TRANSPARENT: the loop neither counts it nor breaks on it, so a production
    # ``meta(ok:false)`` + ``body`` pair keeps the streak intact and only a real
    # success ("ok") breaks the run. (Regression-guarded by
    # test_digest_streak_honors_ok_false_meta_with_body.)
    streak = 0
    for tl in reversed(tools):
        _st = tl.get("status")
        if _st == "err":
            streak += 1
        elif _st == "ok":
            break
        # else: "pending" → skip (transparent to the trailing-error streak).
    return {
        "goal": user_msgs[0] if user_msgs else "",
        "later_user_msgs": user_msgs[1:6],
        # Current-turn intent — the implicit goal + scope key for goal-less
        # supervision (see run_review). Empty when the only user content was
        # placeholder sentinels (e.g. an untranscribable voice segment).
        "current_goal": current_goal,
        "current_turn_index": current_turn_index,
        "rounds": rounds,
        "final_assistant_message": assistant_final[:4000],
        "tools": tools[-40:],
        "tool_failure_streak": streak,
    }


# ---- sidecar ------------------------------------------------------------
def _sidecar_path(cwd: str, agent: str, sid: str) -> str:
    home = os.environ.get("SHAREDAGENTS_HOME") or os.path.join(os.path.expanduser("~"), ".sharedagents")
    d = os.path.join(home, "agents", agent or "jarvis", "teacher_reviews")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{sid or 'unknown'}.json")


def _load_sidecar(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_sidecar(path: str, data: dict) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def _goal_hash(goal: str) -> str:
    return hashlib.sha256((goal or "").encode("utf-8", "replace")).hexdigest()


# ---- teacher endpoint ---------------------------------------------------
def _load_prompt(prompt_text: str = "", prompt_path: str = "") -> str:
    """Resolve the teacher system prompt from SOURCE.

    Callers pass ``prompt_text`` or a source-tree ``prompt_path`` via
    ``settings``. Empty → the embedded ``_DEFAULT_PROMPT`` (also in source).
    No environment / ~/.sharedagents lookups.
    """
    if prompt_text:
        return prompt_text
    if prompt_path and os.path.exists(prompt_path):
        try:
            with open(prompt_path, encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            pass
    return _DEFAULT_PROMPT


def _call_teacher(
    system_prompt: str,
    user_payload: str,
    *,
    endpoint: str,
    model: str,
    api_key: str = "",
    timeout: float = 45.0,
    temperature: float | None = None,
    cfg=None,
    row=None,
    purpose: str = "teacher",
    sampling=None,
    max_tokens: int = 0,
    disable_thinking: bool = False,
) -> dict:
    """POST to the OpenAI-compatible endpoint. Raises on transport error.

    Endpoint/model/key/timeout are passed explicitly (native path resolves
    them from ``cfg.teacher_model()``); no env reads here.
    """
    endpoint = (endpoint or "").rstrip("/")
    if not endpoint or not model:
        raise RuntimeError("teacher endpoint/model not configured")
    url = endpoint + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
        "stream": False,
    }
    override = dict(sampling or {})
    if temperature is not None:
        override["temperature"] = temperature
    apply_sampling(
        payload,
        configured_sampling(
            cfg,
            row if row is not None else sampling_row(cfg, endpoint, model),
            model,
            purpose=purpose,
            override=override,
        ),
    )
    if max_tokens > 0:
        payload["max_tokens"] = int(max_tokens)
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    obj = json.loads(raw)
    content = obj["choices"][0]["message"]["content"]
    return _extract_verdict(content)


def _extract_verdict(text: str) -> dict:
    """Pull the first balanced JSON object out of the model's reply."""
    if not isinstance(text, str):
        raise ValueError("non-string content")
    # strip <think>…</think> and code fences
    import re

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```json", "```")
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in verdict")
    # Braces in quoted evidence are ordinary string characters. Counting
    # them by hand can reject a valid verdict and pause an autonomous goal.
    value, _end = json.JSONDecoder().raw_decode(text, start)
    return value


def _review_summary(verdict: dict) -> str:
    """Human-readable digest of the teacher's verdict, surfaced into the
    session under the ``delegate_teacher`` node so the operator sees the
    teacher's actual reasoning — not just the terse decision line. Shown
    on EVERY verdict (complete included), where previously a "complete"
    emitted nothing readable."""
    if not isinstance(verdict, dict):
        return ""
    v = verdict.get("verdict") or "?"
    head = f"verdict: {v}"
    gm = verdict.get("goal_met")
    if gm is not None:
        head += f" · goal_met={bool(gm)}"
    conf = verdict.get("confidence")
    if conf is not None:
        try:
            head += f" · confidence={float(conf):.2f}"
        except (TypeError, ValueError):
            pass
    lines = [head]
    if verdict.get("goal_restated"):
        lines.append(f"goal: {verdict['goal_restated']}")
    if verdict.get("progress"):
        lines.append(f"progress: {verdict['progress']}")
    for s in (verdict.get("subgoals") or [])[:8]:
        if isinstance(s, dict):
            mark = "✓" if s.get("done") else "○"
            lines.append(f"  {mark} {s.get('desc', '')}")
    gaps = [g for g in (verdict.get("gaps") or []) if g]
    if gaps:
        lines.append("gaps: " + "; ".join(str(g) for g in gaps[:6]))
    if verdict.get("rationale"):
        lines.append(f"rationale: {verdict['rationale']}")
    return "\n".join(lines).strip()


def _fail_open(sidecar_path: str, sidecar: dict, reason: str, *, mandatory: bool) -> dict:
    """Never wedge a turn. If a mandatory-first review failed, mark the
    session supervision_unavailable rather than a silent verified stop.
    Returns the control dict (caller emits / consumes it)."""
    sidecar["terminal_state"] = "supervision_unavailable" if mandatory else sidecar.get("terminal_state", "in_progress")
    sidecar.setdefault("history", []).append({"ts": int(time.time()), "fail_open": reason})
    _save_sidecar(sidecar_path, sidecar)
    return {
        "continue": True,
        "systemMessage": f"[teacher] supervision unavailable this turn: {reason}",
        "suppressOutput": True,
        "goal_unavailable": True,
    }


# ---- review (the one and only entry point) ------------------------------
def run_review(
    event: str,
    *,
    session_id: str,
    transcript_path: str,
    cwd: str,
    goal: str = "",
    agent: str = "jarvis",
    settings: dict | None = None,
) -> dict:
    """Compute a teacher verdict and return the control dict.

    The single source of review logic, called by the native in-process engine
    path (``local_llm.py::_teacher_review_native`` and ``teacher_supervisor``),
    which passes ``settings`` resolved from ``cfg.teacher_model()`` + policy.
    It does NO stdin/stdout I/O and reads NO environment — it runs completely
    inside the sharedagents process.

    The returned dict: ``{"decision":"block","reason":…,"review":…}`` to keep
    the small model working, ``{"continue":true,…}`` to allow a verified stop,
    or a record carrying the review. Never raises — any internal failure
    returns a fail-open ``{"continue":true}``.
    """
    event = (event or "stop").lower()
    settings = settings or {}
    sid = session_id or "unknown"
    transcript = transcript_path or ""

    sc_path = _sidecar_path(cwd, agent, sid)
    sidecar = _load_sidecar(sc_path)

    events = _read_events(transcript)
    tier1 = _digest(events, deep=False)
    # Resolve the goal to supervise against, and the SCOPE KEY that gates the
    # persistent sidecar reset.
    #
    # 1. Explicit ``/goal`` (passed by the engine) always wins — a durable
    #    objective that spans turns. Scope is the goal text.
    # 2. Otherwise (``/teacher on`` with no ``/goal``) there is no durable
    #    objective, so supervise the CURRENT turn's ask (``current_goal``),
    #    NOT the session's first-ever message. Anchoring to the first message
    #    is how the teacher latched onto an ASR "[segment 1 unavailable]"
    #    sentinel and re-drove forever. Scope is per-turn so the sidecar resets
    #    between turns even when two turns share identical text ("continue").
    # 3. If goal-less AND there is no real current ask (only placeholder
    #    sentinels), there is nothing to supervise — return observe-only and
    #    NEVER fire back a continuation against a garbage/empty goal.
    if (goal or "").strip():
        goal = goal.strip()
        scope_key = "explicit:" + _goal_hash(goal)
    else:
        goal = (tier1.get("current_goal") or "").strip()
        if not goal:
            return {"continue": True, "suppressOutput": True}
        scope_key = "implicit:{}:{}".format(tier1.get("current_turn_index"), _goal_hash(goal))
    gh = _goal_hash(goal)
    # Reset the per-session sidecar when the SCOPE changes (new task, or — for
    # implicit goals — a new turn). Keyed on scope_key, not goal text, so a
    # repeated "continue"/"try again" on a later turn doesn't reuse the prior
    # turn's intervention budget and subgoals.
    if sidecar.get("scope_key"):
        if sidecar["scope_key"] != scope_key:
            sidecar = {}  # new scope → reset
    elif sidecar.get("goal_hash") and sidecar["goal_hash"] != gh:
        # Legacy sidecar written before scope_key existed: fall back to the old
        # goal-hash reset so a pre-upgrade task's interventions/stalls/subgoals
        # don't leak into a new goal's scope.
        sidecar = {}
    sidecar.setdefault("session_id", sid)
    sidecar["scope_key"] = scope_key
    sidecar["goal_hash"] = gh  # retained for inspection / back-compat
    sidecar.setdefault("interventions", 0)
    sidecar.setdefault("stalls", 0)
    sidecar.setdefault("reviewed_at_least_once", False)
    sidecar.setdefault("terminal_state", "in_progress")
    sidecar.setdefault("subgoals", [])

    # FIRE-BACK events are the ones where a "not done" verdict re-drives the
    # small model with a continuation prompt (decision:block). This is the
    # whole point of the teacher: stop / premature-end / reasoning-loop AND
    # the clean-stop ``pre_idle`` review + mid-turn ``periodic``. (The shell
    # path only recognised the first three, so ``pre_idle`` silently failed
    # open and never re-drove — the native port fixes that.) Terminal events
    # (error-terminal / stop-cap) are NOT fire-back: the engine routes them
    # through ``_teacher_terminal_review`` which never re-drives.
    fire_back = event in FIRE_BACK_EVENTS
    # Anti-false-done: any review that could approve a STOP must use Tier-2
    # evidence; never approve "done" on Tier-1 alone. All stop-attempt
    # boundaries qualify (stop / premature-end / reasoning-loop / pre_idle).
    mandatory = (not sidecar["reviewed_at_least_once"]) and fire_back
    deep_tier = settings.get("deep_tier", "auto")
    # A non-positive ``tool_failure_threshold`` DISABLES the streak→deep
    # escalation (operators set 0 to turn the proactive tool-failure trigger
    # off) — guard ``> 0`` so ``streak >= 0`` does not silently force every
    # review to the expensive deep tier.
    try:
        _tft = int(settings.get("tool_failure_threshold", 5))
    except (TypeError, ValueError):
        _tft = 5
    want_deep = deep_tier == "always" or (
        deep_tier != "never"
        and (
            event in STOP_ATTEMPT_EVENTS
            or event in DEEP_REVIEW_EVENTS
            or (_tft > 0 and tier1["tool_failure_streak"] >= _tft)
        )
    )
    digest = _digest(events, deep=True) if want_deep else tier1

    user_payload = json.dumps(
        {
            "event": event,
            "original_goal": goal,
            "prior_subgoals": sidecar.get("subgoals", []),
            "interventions_so_far": sidecar["interventions"],
            "consecutive_stalls": sidecar["stalls"],
            "tier": "deep" if want_deep else "shallow",
            "transcript_digest": digest,
        },
        indent=2,
    )

    # --- call the teacher, with bounded resilience -----------------------
    retries = 2 if mandatory else 1  # the one obligatory review gets a retry
    verdict = None
    last_err = ""
    system_prompt = _load_prompt(settings.get("prompt_text", ""), settings.get("prompt_path", ""))
    for attempt in range(retries):
        try:
            verdict = _call_teacher(
                system_prompt,
                user_payload,
                endpoint=settings.get("endpoint", ""),
                model=settings.get("model", ""),
                api_key=settings.get("api_key", ""),
                timeout=float(settings.get("timeout", 45.0)),
                sampling=settings.get("sampling"),
            )
            break
        except (urllib.error.URLError, OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5 * (attempt + 1))
    if verdict is None:
        if fire_back:
            return _fail_open(sc_path, sidecar, last_err, mandatory=mandatory)
        _save_sidecar(sc_path, sidecar)
        return {"continue": True, "suppressOutput": True}

    # --- fold verdict into sidecar --------------------------------------
    sidecar["reviewed_at_least_once"] = True
    sidecar["last_verdict"] = verdict.get("verdict", "")
    if verdict.get("subgoals"):
        sidecar["subgoals"] = verdict["subgoals"]
    progress = bool(verdict.get("progress_made", True))
    if progress:
        sidecar["stalls"] = 0
    else:
        sidecar["stalls"] += 1
    sidecar.setdefault("history", []).append(
        {
            "ts": int(time.time()),
            "event": event,
            "verdict": verdict.get("verdict"),
            "tier": "deep" if want_deep else "shallow",
            "progress_made": progress,
        }
    )

    v = (verdict.get("verdict") or "").lower()
    goal_met = bool(verdict.get("goal_met"))
    confidence = float(verdict.get("confidence") or 0.0)
    min_conf = float(settings.get("min_confidence", 0.6))
    cont = (verdict.get("continuation_prompt") or "").strip()
    # Readable digest surfaced into the session on EVERY verdict.
    review = _review_summary(verdict)

    # --- decision: return the control dict (caller renders / re-drives) --
    if v == "complete" and goal_met and confidence >= min_conf and want_deep:
        sidecar["terminal_state"] = "completed_by_small_model"
        _save_sidecar(sc_path, sidecar)
        # No fire-back: the goal is verified met → let the turn end (idle).
        return {
            "continue": True,
            "suppressOutput": True,
            "review": review,
            "goal_met": True,
        }

    if v == "needs_human":
        sidecar["terminal_state"] = "needs_human"
        _save_sidecar(sc_path, sidecar)
        return {
            "continue": True,
            "review": review,
            "systemMessage": f"[teacher] needs human: {verdict.get('rationale', '')[:300]}",
            "goal_blocked": True,
        }

    # continue / off_track / stuck / looping → keep the small model working.
    over_budget = sidecar["interventions"] >= int(settings.get("max_interventions", 6))
    stalled_out = sidecar["stalls"] >= int(settings.get("stall_limit", 2))
    if over_budget or stalled_out:
        # Escalation territory. The continuation still fires back (so the
        # model re-attempts) but is flagged; the escalation model-swap ladder
        # (when wired) reads the ``escalate`` sidecar flag.
        sidecar["terminal_state"] = sidecar.get("terminal_state", "in_progress")
        sidecar["escalate"] = True
        _save_sidecar(sc_path, sidecar)
        if fire_back:
            reason = cont or "[teacher] Re-attempt with a different approach; escalation pending."
            return {"decision": "block", "reason": _tag(reason), "review": review, "escalate": True}
        return {"continue": True, "suppressOutput": True, "review": review, "escalate": True}

    sidecar["interventions"] += 1
    _save_sidecar(sc_path, sidecar)
    if fire_back:
        if not cont:
            cont = "Your work does not yet meet the original goal. Keep going on the remaining gaps: " + "; ".join(
                verdict.get("gaps", []) or ["complete the task"]
            )
        return {"decision": "block", "reason": _tag(cont), "review": review}
    # Non-fire-back (terminal) event: record the verdict + surface the review
    # under the delegate_teacher node, but never re-drive.
    return {
        "continue": True,
        "suppressOutput": True,
        "teacher_action": "inject",
        "teacher_text": _tag(cont),
        "review": review,
    }


def _tag(text: str) -> str:
    text = text or ""
    return text if text.lstrip().startswith("[teacher") else f"[teacher] {text}"


_DEFAULT_PROMPT = """\
You are the TEACHER: a strong model supervising a smaller local model that is
performing an agentic task. You are given the ORIGINAL GOAL, the prior sub-goal
checklist, and a digest of what the small model has done so far.

Decide whether the original goal is actually met. Be skeptical of premature
"done" claims — the small model often stops or loops early. Do NOT approve a
completion unless the evidence (tool results, final message) genuinely shows the
goal achieved.

Return ONLY a strict JSON object, no prose:
{
  "goal_restated": "<one sentence>",
  "subgoals": [{"desc": "<concrete sub-goal>", "done": true|false}],
  "progress": "<what was actually accomplished>",
  "progress_made": true|false,   // did anything advance vs the prior checklist?
  "verdict": "complete|continue|off_track|looping|stuck|needs_human",
  "goal_met": true|false,
  "confidence": 0.0-1.0,
  "needs_detail": true|false,     // set true if you cannot judge without tool bodies
  "gaps": ["<what is still missing>"],
  "continuation_prompt": "<direct next-step instructions for the small model; empty if complete>",
  "escalate": true|false,
  "rationale": "<why this verdict>"
}
Use verdict "needs_human" ONLY for a genuine external blocker the user alone can
resolve (missing credential, truly unrecoverable ambiguity) — never just because
the task is hard.

IF THE ORIGINAL GOAL IS EMPTY OR UNCLEAR: do NOT return needs_human. DEDUCE the
implicit goal from what the assistant has already been doing (its messages, tool
calls, artifacts), restate it in "goal_restated", and review against it. A
missing/empty goal is never a reason to involve a human — infer intent and keep
the small model working toward it.
"""
