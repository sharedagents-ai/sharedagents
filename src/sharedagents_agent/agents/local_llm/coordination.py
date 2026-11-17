"""Per-physical-turn workflow enforcement for the Jarvis coordinator."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sharedagents_agent.tools.registry import is_agent_tool
from sharedagents_agent.tools.widget_adapt import is_ask_user_tool

WAVE_SUMMARY_PROMPT = (
    "Your delegation wave has started. Now explain to the owner what was "
    "actually dispatched and the consolidated output you expect when all "
    "children finish. Then wait for the wave result. Do not dispatch another "
    "wave, poll, or claim that the work has already finished."
)

# Runtime-authored result envelopes. A turn made only of these is a delivery
# of earlier work, not a new owner prompt. Child output sits inside an
# untrusted-data fence whose glyphs the content can never contain, so the
# fenced spans are removed FIRST: a child that writes "</delegation_wave>" or
# a wave tag in its answer cannot end the envelope or name another wave.
_FENCED_RE = re.compile(
    r"⟦UNTRUSTED-DATA id=([0-9a-f]+)[^\n⟧]*⟧.*?⟦/UNTRUSTED-DATA id=\1[^\n⟧]*⟧",
    re.S,
)
_WAVE_TAG_RE = re.compile(r'<delegation_wave wave_id="(wave_[0-9a-f]{32})"')
_ENVELOPE_RES = (
    re.compile(r"<delegation_wave\b[^>]*>.*?</delegation_wave>", re.S),
    re.compile(r"<task_(?:result|error)\b[^>]*>.*?</task_(?:result|error)>", re.S),
)
_RUNTIME_TRAILERS = (
    "All children in this delegation wave are terminal. Incorporate the "
    "consolidated outcomes and continue. Cancelled or aborted children "
    "were explicitly stopped; do not retry or redispatch them.",
    "Incorporate this completed task and continue.",
    "The user explicitly stopped this delegated task. Do NOT retry or "
    "re-dispatch this task. Acknowledge that the task was cancelled.",
    "The delegated task did not complete. Decide whether to retry, adjust the approach, or report it.",
)
# Sources that never carry an owner message. ``queue_drain`` is absent on
# purpose: it can join wave results with follow-ups the owner typed meanwhile.
_NON_OWNER_SOURCES = frozenset(
    {
        "background_task",
        "delegate",
        "heartbeat",
        "loop_supervisor",
        "memory_flush",
        "teacher",
        "goal_supervisor",
    }
)
# A child brief that drives the shared Chrome. One agent per profile at a
# time: a second one resets the page the owner or the first agent is using.
_BROWSER_RE = re.compile(
    r"(?<![\w/.-])(?:browser|nodriver|chrome|chromium|devtools|display\s*:\d+|"
    r"live\s+profile)(?![\w-]|\.[A-Za-z0-9])",
    re.I,
)


def is_dispatch(name: str, arguments: dict) -> bool:
    return is_agent_tool(name) and str(arguments.get("action") or "dispatch").strip().lower() not in {
        "list",
        "abort",
        "steer",
    }


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        return " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in prompt)
    return str(prompt or "")


def _has_non_text_part(prompt: Any) -> bool:
    return isinstance(prompt, list) and any(isinstance(p, dict) and p.get("type") not in (None, "text") for p in prompt)


def _runtime_text(prompt: Any) -> str:
    """The prompt with every fenced (untrusted) span removed."""
    return _FENCED_RE.sub(" ", _prompt_text(prompt))


def delivered_wave_ids(prompt: Any) -> list[str]:
    """Wave ids whose consolidated result this prompt delivers."""
    return list(dict.fromkeys(_WAVE_TAG_RE.findall(_runtime_text(prompt))))


def has_owner_text(prompt: Any, source: str = "") -> bool:
    """True when the prompt carries words the owner wrote, not only results."""
    if str(source or "user").strip().lower() in _NON_OWNER_SOURCES:
        return False
    if _has_non_text_part(prompt):
        # An image or file the owner attached is an owner prompt by itself.
        return True
    text = _runtime_text(prompt)
    for rx in _ENVELOPE_RES:
        text = rx.sub(" ", text)
    for trailer in _RUNTIME_TRAILERS:
        text = text.replace(trailer, " ")
    return bool(text.strip())


def new_prompt_origin() -> str:
    return "prompt_" + uuid.uuid4().hex


# Negation must sit in the same clause: "do not use curl, use the browser"
# is an affirmative browser instruction.
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|don'?t|cannot|can'?t|must\s+not|stay\s+out\s+of|"
    r"off-limits|forbidden|avoid)\b[^.;:,\n]*$",
    re.I,
)


def is_browser_task(text: Any) -> bool:
    """A brief that drives the browser, ignoring "do not touch the browser"."""
    body = _prompt_text(text)
    for m in _BROWSER_RE.finditer(body):
        before = body[max(0, m.start() - 60) : m.start()]
        if not _NEGATION_RE.search(before):
            return True
    return False


_MAX_RUN_DEPTH = 200


def _decode_runs(runs: Any) -> list:
    """The batch runner's accepted ``runs`` forms as a list of entries."""
    if isinstance(runs, str) and runs.strip():
        try:
            import json

            runs = json.loads(runs)
        except ValueError:
            return []
    if isinstance(runs, dict):
        return [runs]
    return list(runs) if isinstance(runs, list) else []


def _decode_entry(entry: Any) -> Any:
    if isinstance(entry, str) and entry.strip():
        try:
            import json

            return json.loads(entry)
        except ValueError:
            return None
    return entry


def _run_entries(arguments: Any, inherited: dict | None = None, depth: int = 0) -> list[dict]:
    """Every leaf run of a dispatch, normalized the way the batch runner
    executes it: JSON-encoded ``runs`` and entries are decoded, top-level
    fields (prompt, subagent_type, ...) are inherited by entries that lack
    them, and nested batches are flattened at any depth."""
    if not isinstance(arguments, dict):
        return []
    merged = {**(inherited or {}), **arguments}
    runs = _decode_runs(arguments.get("runs"))
    if not runs:
        merged.pop("runs", None)
        return [merged]
    if depth >= _MAX_RUN_DEPTH:
        # Absurd nesting: judge the whole subtree as several runs carrying
        # all of its text, so any browser mention in it is refused.
        import json

        blob = {"prompt": json.dumps(merged, default=str)}
        return [blob, dict(blob)]
    shared = {k: v for k, v in merged.items() if k not in ("runs", "task_id")}
    out: list[dict] = []
    for entry in runs:
        out.extend(_run_entries(_decode_entry(entry), shared, depth + 1))
    return out


def _dispatch_prompts(arguments: dict) -> list[str]:
    return [
        # Every field the delegate shims accept as the task text.
        " ".join(str(e.get(k) or "") for k in ("description", "prompt", "task", "message", "input"))
        for e in _run_entries(arguments)
    ]


def _job_label(job: dict) -> str:
    tid = str(job.get("child_sid") or job.get("job_id") or "?")
    return f"{job.get('agent') or '?'} task_id={tid}"


@dataclass
class CoordinationTurn:
    """Never reset by worker rounds, Stop hooks, or in-turn nudges.

    ``origin`` names the owner prompt this turn works for. An owner prompt
    starts a new origin; a turn that only delivers wave results inherits the
    origin of those waves. Each origin may have one live wave at a time, so a
    new owner question about another subject can start its own wave while an
    earlier wave is still running. An empty origin (automatic work with no
    known owner prompt, or a wave from before origins were recorded) keeps
    the strict rule: no new wave while any wave of the session is open.
    """

    delegated: bool = False
    summary_requested: bool = False
    owner_boundary: bool = False
    origin: str = ""
    owner_prompt: bool = False
    delivered_waves: list[str] = field(default_factory=list)

    def refusal(
        self,
        name: str,
        arguments: dict,
        *,
        has_todos: bool,
        open_jobs: bool | Iterable[dict],
        open_wave_origins: dict[str, str] | None = None,
    ) -> str | None:
        # Planning and owner/child controls remain available even without a
        # plan; in particular, a stop must never be blocked by this gate.
        control = is_agent_tool(name) and str(arguments.get("action") or "dispatch").strip().lower() in {
            "list",
            "abort",
            "steer",
        }
        if name == "todowrite" or control or is_ask_user_tool(name):
            return None
        if not has_todos:
            return (
                "Create a non-empty todo list with todowrite before acting. "
                "Wait for that tool to succeed, then carry out the plan."
            )
        if not is_dispatch(name, arguments):
            return None
        if self.delegated:
            return (
                "Only one delegation wave is allowed per physical turn. Put all "
                "independent subagents in one AgentAsync(runs=[...]) call; a "
                "run may continue a finished child by passing its task_id with "
                "a new prompt. This turn already dispatched its wave; explain "
                "the expected consolidated output and wait for all children."
            )
        jobs = self._open_list(open_jobs)
        browser = self._browser_conflict(arguments, jobs)
        if browser:
            return browser
        if not jobs:
            return None
        origins = open_wave_origins or {}
        if not self.origin:
            return (
                "A delegation wave is still running. Wait until all its "
                "children finish before dispatching the next wave. "
                "Use steer to change a live child, or abort only when needed "
                "for owner control."
            )
        same = [j for j in jobs if origins.get(str(j.get("wave_id") or "")) == self.origin]
        if same:
            return (
                "A delegation wave for this same request is still running ("
                + ", ".join(_job_label(j) for j in same[:6])
                + '). Wait for it to finish, or use action="steer" with a '
                "task_id to redirect a live child. Do not start a second wave "
                "for the same request."
            )
        return None

    @staticmethod
    def _open_list(open_jobs: bool | Iterable[dict]) -> list[dict]:
        if isinstance(open_jobs, bool):
            # Legacy callers only know whether something is open; treat that
            # as one job of unknown origin.
            return [{}] if open_jobs else []
        return [j for j in (open_jobs or []) if isinstance(j, dict)]

    @staticmethod
    def _browser_conflict(arguments: dict, jobs: list[dict]) -> str | None:
        wanted = sum(1 for p in _dispatch_prompts(arguments) if is_browser_task(p))
        if not wanted:
            return None
        if wanted > 1:
            return (
                "Only one agent may drive the browser at a time, and this "
                "dispatch has several runs that use it. Give the browser work "
                "to one child (in order, in one brief) and resend."
            )
        busy = [j for j in jobs if is_browser_task(j.get("task"))]
        if not busy:
            return None
        return (
            "Another child is already driving the browser ("
            + ", ".join(_job_label(j) for j in busy[:3])
            + "). Only one agent may use a Chrome profile at a time; a second "
            "agent resets the page the first one or the owner is working on. "
            'Steer that child with action="steer" and its task_id, or wait '
            "for it to finish and continue it with its task_id."
        )

    def observe(self, name: str, arguments: dict, *, ok: bool, background_handle: bool) -> None:
        # A partially refused batch can still contain live children. The
        # structured handle, never the result's prose, consumes the wave.
        if is_dispatch(name, arguments) and (ok or background_handle):
            self.delegated = True
        if ok and (
            is_ask_user_tool(name) or (is_agent_tool(name) and str(arguments.get("action")).strip().lower() == "abort")
        ):
            self.owner_boundary = True

    def should_nudge(
        self,
        *,
        pending: list[str],
        open_jobs: bool,
        stopped: bool,
        limited: bool,
        nudge_count: int,
        max_nudges: int,
        disposition: str = "working",
    ) -> bool:
        return bool(
            disposition == "working"
            and pending
            and not self.delegated
            and not self.owner_boundary
            and not open_jobs
            and not stopped
            and not limited
            and nudge_count < max_nudges
        )


# An item that is waiting on the owner or on an outside event is not stalled
# work: re-dispatching it burns a child without changing the answer. The todo
# text is the durable carrier (it is persisted in ``<sid>.todos.json``), so a
# marker written once survives every later turn and compaction.
_BLOCKED_APPROVAL_RE = re.compile(
    r"\[(?:blocked|waiting)\s*:\s*(?:approval|owner|user)\]"
    r"|\b(?:awaiting|waiting\s+(?:on|for))\s+(?:the\s+)?owner\b"
    r"|\b(?:awaiting|needs?)\s+(?:owner\s+)?approval\b"
    r"|\bowner\s+approval\s+(?:is\s+)?required\b",
    re.I,
)
_BLOCKED_EXTERNAL_RE = re.compile(
    r"\[(?:blocked|waiting)\s*:\s*(?:external|ci|deploy|event)\]"
    r"|\b(?:awaiting|waiting\s+(?:on|for))\s+(?:an?\s+)?"
    r"(?:external|outside|ci|deploy|event)\b",
    re.I,
)
_GOAL_BLOCKED_TEXT_RE = re.compile(r"\bGOAL\s+BLOCKED\b", re.I)


def workflow_disposition(pending: Iterable[str], last_content: str = "") -> str:
    """The turn's workflow state, derived from durable evidence.

    ``working``          — there is pending work this turn may dispatch.
    ``waiting_approval`` — the remaining item needs the owner (production step,
                           destructive action, a value only the owner has).
    ``waiting_external`` — the remaining item waits on an outside event (CI, a
                           deploy, a third party) that a child cannot advance.
    ``complete``         — nothing is pending.

    A non-``working`` disposition suppresses the ``pending_todos_without_
    delegation`` nudge. This is the harness-side counterpart to the prompt's
    ``GOAL BLOCKED`` stop: even a model that forgets to say it cannot be
    re-driven into spending on an approval it does not have.
    """
    items = [str(x) for x in (pending or []) if str(x).strip()]
    if not items:
        return "complete"
    if any(_BLOCKED_APPROVAL_RE.search(x) for x in items):
        return "waiting_approval"
    if any(_BLOCKED_EXTERNAL_RE.search(x) for x in items):
        return "waiting_external"
    if _GOAL_BLOCKED_TEXT_RE.search(str(last_content or "")):
        return "waiting_approval"
    return "working"


def start_turn(prompt: Any, source: str = "", wave_origin_of: Any = None) -> CoordinationTurn:
    """Build the coordination state for one physical turn.

    ``wave_origin_of`` maps a wave id to its recorded origin (``""`` when the
    wave predates origins).
    """
    waves = delivered_wave_ids(prompt)
    owner = has_owner_text(prompt, source)
    origin = ""
    if owner:
        origin = new_prompt_origin()
    elif waves and wave_origin_of is not None:
        for wid in waves:
            try:
                origin = str(wave_origin_of(wid) or "")
            except Exception:  # noqa: BLE001 — origin is advisory
                origin = ""
            if origin:
                break
    return CoordinationTurn(
        origin=origin,
        owner_prompt=owner,
        delivered_waves=waves,
    )


def live_children_note(jobs: Iterable[dict]) -> str:
    """Owner-prompt reminder of earlier work still running in the background."""
    rows = [j for j in jobs if isinstance(j, dict)]
    if not rows:
        return ""
    lines = [f"- {_job_label(j)}: {str(j.get('task') or '').strip()[:160]}" for j in rows[:8]]
    return (
        "[sharedagents delegation status] Children from earlier requests are still "
        "running:\n" + "\n".join(lines) + "\n"
        "If this new message is about the same subject as one of them, steer "
        'that child (AgentAsync action="steer", task_id, prompt) instead of '
        "dispatching. If it is a different subject, you may start one new "
        "wave for it. Prefer continuing a finished child (task_id plus a new "
        "prompt) over a fresh child when the task is the same."
    )
