"""Dev-tab SDLC control plane: intake -> issue -> plan -> design -> code ->
tests -> deploy-readiness -> PR -> merge -> deploy.

The Dev tab is meant to be the ONLY surface needed to ship a feature or a fix.
This module owns the half of that promise the closed-loop registry does not:

* **Validate (once).** Clarify freezes two artifacts: a WHEN/SHALL condition
  for plan, and given/when/then/fail-if evidence for test. Nothing is minted
  until both parse and the operator ratifies them. That freeze is the only
  validation — later stages do not re-ask whether the requirements are right.
* **Develop.** ``plan -> code -> tests`` on the fused path (design is
  skipped unless the operator splits the V). Generate without further
  prompting. Each completed stage is posted to the forge issue under its own
  tag so the issue is a readable build log rather than a stub.
* **Verify.** LSP after code (is it even legal). Tests drive the frozen
  evidence on the named surface. Two independent models then re-read the same
  evidence. Deploy-readiness is deterministic once those reviews and LSP are
  green.
* **Sign-off.** Merge is automatic when independent verification A, independent
  verification B, and LSP are green. Deploy stays a human click.

Stage completion is deterministic, not inferred: an agent finishes a stage by
emitting a ``SDLC-STAGE-DONE:`` marker line, and the driver reads that marker.
An idle session with no marker is a failed stage, never a silent pass.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NamedTuple

from sharedagents_core.layout import sharedagents_root
from sharedagents_model.sampling import apply_sampling, configured_sampling

log = logging.getLogger(__name__)


#: P5: after this many no-progress failures of one stage, the run stops
#: auto-resuming. The failed calculator run restarted identically three times
#: because the supervisor treated every block as recoverable; a stage that
#: burned its whole budget twice with nothing to hand over needs an operator
#: decision (edit scope, raise the budget, or abort), not a third restart.
MAX_NO_PROGRESS_ATTEMPTS = int(os.environ.get("SHAREDAGENTS_SDLC_MAX_NO_PROGRESS_ATTEMPTS") or 2)

#: P3: the retry digest is a handoff, not a transcript. Everything above this
#: bound is dropped oldest-first so one verbose attempt cannot bloat the next
#: worker's prompt.
ATTEMPT_DIGEST_MAX_BYTES = 2048

#: P4: fraction of the first-artifact budget after which the flight strip
#: switches from ok to warn. Must match the driver's STEER_AT_FRACTION.
WATCHDOG_WARN_FRACTION = 0.6

#: P7: per-stage spend ceiling (prompt + completion). A breach is a
#: no-progress-style failure so P5's fuse can stop the line.
STAGE_TOKEN_BUDGET = int(os.environ.get("SHAREDAGENTS_SDLC_STAGE_TOKEN_BUDGET") or 400_000)

#: P7: a round that burns this many reasoning tokens with no tool call is
#: starving. Stop it instead of paying for another empty 16k-token turn.
REASONING_STARVATION_TOKENS = int(os.environ.get("SHAREDAGENTS_SDLC_REASONING_STARVATION_TOKENS") or 12_000)

#: Visible attempt cap shown next to the running gate. Matches the driver's
#: MAX_STAGE_ATTEMPTS without importing the driver (circular).
STAGE_ATTEMPTS_MAX = 2


def stage_artifact_paths(
    stage: str,
    branch: str,
    *,
    fuse_plan_design: bool = True,
) -> dict[str, str]:
    """Named artifact locations the worker must write. Shared by the prompt
    and the P8 path guard so they cannot drift."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(branch or "run").lower()).strip("-") or "run"
    paths = {"plan": f"plan/{slug}.md"}
    if not fuse_plan_design or stage == "design":
        paths["design"] = f"spec/{slug}.md"
    if stage in {"tests", "deploy"}:
        paths["checks"] = ".sharedagents/verify.json"
        paths["isolation"] = "Dockerfile.test"
    return paths


def prepare_artifact_parents(worktree: str, paths: dict[str, str]) -> list[str]:
    """P8: make each artifact's parent exist before the worker is dispatched."""
    notes: list[str] = []
    root = Path(worktree) if worktree else None
    if root is None or not root.is_dir():
        return notes
    for rel in paths.values():
        parent = (root / rel).parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:  # noqa: BLE001 — reported, not raised
            notes.append(f"could not create {parent}: {error}")
    return notes


def stray_artifact_note(worktree: str, paths: dict[str, str]) -> str:
    """P8: name sibling directories that appeared instead of the artifact."""
    root = Path(worktree) if worktree else None
    if root is None or not root.is_dir():
        return ""
    found: list[str] = []
    for rel in paths.values():
        expected = root / rel
        if expected.is_file():
            continue
        parent = expected.parent
        if not parent.is_dir():
            continue
        try:
            children = list(parent.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                found.append(str(child.relative_to(root)).rstrip("/") + "/")
    if not found:
        return ""
    return "stray artifact path(s): " + ", ".join(sorted(set(found)))


def phase_rules_prefix(prompt: str) -> str:
    """The byte-stable rules block; RUN CONTEXT is stripped (P7)."""
    return prompt.split("RUN CONTEXT (JSON data):", 1)[0]


def _first_write_budget_s(run: dict[str, Any], stage: str) -> float:
    override = float(run.get("first_artifact_budget_s") or 0.0)
    if override:
        return override
    env_key = {
        "plan": "SHAREDAGENTS_SDLC_PLAN_FIRST_WRITE_TIMEOUT",
        "code": "SHAREDAGENTS_SDLC_CODE_FIRST_WRITE_TIMEOUT",
        "tests": "SHAREDAGENTS_SDLC_TEST_FIRST_WRITE_TIMEOUT",
    }.get(stage)
    if not env_key:
        return 0.0
    return float(os.environ.get(env_key) or 900.0)


def watchdog_projection(
    run: dict[str, Any],
    events: Iterable[dict[str, Any]] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """P4: the numbers the running-gate clock renders.

    Computed here so the flight strip and the tests share one source; the
    driver is not imported (it already imports this module).
    """
    now = now or datetime.now(timezone.utc)
    stage = str(run.get("stage") or "")
    status = str(run.get("status") or "")
    started: dict[str, Any] | None = None
    started_count = 0
    tokens = int(run.get("stage_tokens") or 0)
    for event in events or []:
        if not isinstance(event, dict):
            continue
        if str(event.get("stage") or "") != stage:
            continue
        if event.get("status") == "started":
            started = event
            started_count += 1
        tokens += int(event.get("tokens") or 0)
    attempts = started_count or 1
    if status == "running" and int(run.get("no_progress_attempts") or 0):
        attempts = int(run.get("no_progress_attempts") or 0) + 1
    elif status != "running" and int(run.get("no_progress_attempts") or 0):
        attempts = int(run.get("no_progress_attempts") or 0)
    budget = _first_write_budget_s(run, stage)
    started_at = str((started or {}).get("created_at") or "")
    deadline = ""
    elapsed_s = 0.0
    if started_at:
        try:
            start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            elapsed_s = max(0.0, (now - start_dt).total_seconds())
            if budget:
                deadline = (start_dt + timedelta(seconds=budget)).isoformat()
        except ValueError:
            pass
    fused = int(run.get("no_progress_attempts") or 0) >= MAX_NO_PROGRESS_ATTEMPTS
    expected = {
        "plan": "plan brief",
        "code": "product change",
        "tests": "proof artifact",
    }.get(stage, "stage artifact")
    state = "ok"
    note = ""
    if fused:
        state = "cancelled"
        spent = int(elapsed_s) if elapsed_s else int(budget or 0)
        note = f"{stage or 'stage'} produced no {expected} after {spent}s"
    elif status == "running" and budget and elapsed_s >= WATCHDOG_WARN_FRACTION * budget:
        state = "warn"
        note = "No artifact yet; first-artifact budget is more than 60% spent."
    return {
        "stage_started_at": started_at,
        "stage_attempt": attempts,
        "stage_attempts_max": STAGE_ATTEMPTS_MAX,
        "first_artifact_deadline": deadline,
        "stage_tokens": tokens,
        "watchdog_state": state,
        "watchdog_note": note,
        "no_progress_fused": fused,
    }


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

#: Pipeline order, exactly as the Dev tab advertises it.
STAGES: tuple[str, ...] = ("plan", "design", "code", "tests", "deploy")
FUSED_STAGES: tuple[str, ...] = ("plan", "code", "tests", "deploy")


def stages_for_run(run: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Stages this run actually dispatches. Design is skipped on the fused path."""
    if (run or {}).get("fuse_plan_design", True):
        return FUSED_STAGES
    return STAGES


ROLE_NAMES: tuple[str, ...] = ("validation", "generation", "verification")
#: Orchestrate always runs as Jarvis on the conductor. Only the model is a choice.
GENERATION_AGENT = "jarvis"
#: Verify A / verify B: the local builder or an external engine. Never the
#: conductor (jarvis) and never the read-only planner.
REVIEWER_AGENTS: tuple[str, ...] = (
    "build",
    "claude",
    "cursor",
    "codex",
    "antigravity",
    "opencode",
    "kimi",
    "hermes",
    "devin",
)
NOT_REVIEWERS: frozenset[str] = frozenset({GENERATION_AGENT, "plan"})
VERIFY_A_DEFAULTS: tuple[str, ...] = ("build", "claude", "codex", "antigravity")
VERIFY_B_DEFAULTS: tuple[str, ...] = (
    "cursor",
    "claude",
    "antigravity",
    "opencode",
    "kimi",
    "hermes",
    "build",
)
#: Reviewers that run inside the gateway. They need no CLI and no out-of-band
#: login, so one of them is always dispatchable -- which is what makes them the
#: right last resort for a default that has to work unattended.
IN_PROCESS_REVIEWERS: tuple[str, ...] = ("build", "general", "explore")

#: How to tell, cheaply, that an external engine has credentials. Either an
#: API key in the environment, or the login file its own CLI writes. Neither
#: proves the credential is still valid; both prove one was configured, which
#: is the difference between "the operator set this up" and "this CLI is merely
#: installed".
_EXTERNAL_CREDENTIALS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # agent: ((env vars), (paths under $HOME))
    "claude": (("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"), (".claude/.credentials.json", ".claude.json")),
    "codex": (("OPENAI_API_KEY", "CODEX_API_KEY"), (".codex/auth.json",)),
    "cursor": (("CURSOR_API_KEY",), (".config/cursor/auth.json",)),
    "opencode": (("OPENCODE_API_KEY",), (".local/share/opencode/auth.json",)),
    "kimi": (("KIMI_API_KEY", "MOONSHOT_API_KEY"), (".kimi/auth.json",)),
    "hermes": (("HERMES_API_KEY", "OPENROUTER_API_KEY"), (".hermes/auth.json",)),
    "devin": (("WINDSURF_API_KEY",), (".local/share/devin/credentials.toml",)),
    "antigravity": (
        ("ANTIGRAVITY_API_KEY", "GEMINI_API_KEY"),
        (".antigravity/auth.json", ".config/antigravity/auth.json"),
    ),
    # ZCode authenticates with its own Z.AI OAuth login; ZCODE_API_KEY (the
    # coding-plan key the CLI provisions) is an optional additional
    # credential for API access.
    "zcode": (("ZCODE_API_KEY",), (".zcode/v2/credentials.json",)),
}


def reviewer_dispatchable(agent: str) -> bool:
    """Can this deployment actually give *agent* a turn?

    A default that cannot run is not a default. ``cursor`` ships in the Docker
    dev image, so it is on PATH and was chosen as verify B -- and every run
    reached the review stage only to die on "Authentication required. Please
    run 'agent login'". The pipeline is supposed to finish without a person,
    and a first-boot deployment could not finish at all.

    In-process reviewers are always dispatchable. An external engine counts
    only when its CLI is installed AND some credential for it exists.
    """
    import shutil

    name = str(agent or "").strip()
    if not name or name in NOT_REVIEWERS:
        return False
    if name in IN_PROCESS_REVIEWERS:
        return True
    try:
        # Keyed by EngineKind, so look it up by the short name the Dev tab uses.
        # `EXTERNAL_REGISTRY.get("cursor")` is always None and would silently
        # fall back to probing a binary called `cursor` -- there isn't one; the
        # CLI is `cursor-agent`.
        from sharedagents_agent.external_registry import spec_for_short_name

        spec = spec_for_short_name(name)
    except Exception:  # noqa: BLE001 — registry absent: judge by name alone
        spec = None
    command = getattr(spec, "cli_command", "") or name
    if not shutil.which(command):
        return False
    envs, paths = _EXTERNAL_CREDENTIALS.get(name, ((), ()))
    if any(os.environ.get(key, "").strip() for key in envs):
        return True
    home = Path.home()
    if any((home / rel).exists() for rel in paths):
        return True
    # No credential convention recorded for this engine: installed is the only
    # signal there is, so take it rather than refusing a working setup.
    return not envs and not paths


#: Storage keys stay validation/generation/verification (DB + API). The jobs
#: they do: generation builds; the other two are independent verifiers after
#: tests. Validation of the *requirements* is the clarify/contract freeze,
#: not a third post-test scorer.
ROLE_LABELS: dict[str, str] = {
    "validation": "verify A",
    "generation": "orchestrate",
    "verification": "verify B",
}
#: Independent LLM verification runs once, after tests, in parallel. Plan /
#: design / code use generation + cheap structural checks (LSP on code).
#: Deploy is deterministic once those reviews and LSP are green.
LLM_REVIEW_STAGE = "tests"


def uses_llm_gate(stage: str) -> bool:
    """True only for the one stage that pays for independent model review."""
    return canonical_stage(stage) == LLM_REVIEW_STAGE


# Bump only when an unfinished run deserves one fresh, bounded recovery budget
# because the pipeline engine itself changed.  The stored revision makes that
# refund a migration, not an unbounded retry loophole.
PIPELINE_REVISION = 3

#: Runs created before Plan and Design were separate stages carry ``spec``.
#: It meant "the technical design", so it resumes as ``design``.
LEGACY_STAGES: dict[str, str] = {
    "spec": "design",
    # Runs created before the terminal gate was named for its purpose recorded
    # it as ``review``.  Canonicalising at every read keeps those runs resumable
    # while all newly emitted labels/events use ``sdlc/deploy``.
    "review": "deploy",
}


def canonical_stage(stage: str) -> str:
    """The current name for *stage*, which may be a retired one."""
    return LEGACY_STAGES.get(str(stage or ""), str(stage or ""))


#: A real chat session id, in either form sharedagents has ever minted.
#:
#: ``new_session_id()`` returns ``YYYYMMDDTHHMMSSffffff-<12 hex>`` -- NOT a
#: UUID. This pattern used to accept only the UUID, so it rejected every id the
#: running system produces, exactly as if it were the ``verify-a-obs``
#: placeholder it exists to filter out. Two things broke silently:
#:
#:   * the Dev tab unbound the conductor on every paint, so the clarify
#:     question card rendered but its Submit was a no-op -- the ONE gate a
#:     human has to pass could not be passed from the Dev tab; and
#:   * ``independent_reviews_ready`` could never be true, so the pipeline's
#:     own merge could never fire.
#:
#: Placeholders stay rejected: they match neither shape.
_SESSION_ID = re.compile(
    r"^(?:"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"  # legacy UUID
    r"|\d{8}T\d{12}-[0-9a-f]{12}"  # sharedagents stamp
    r")$",
    re.IGNORECASE,
)
#: Retained name for callers that imported the old symbol.
_SESSION_UUID = _SESSION_ID


def usable_session_id(value: Any) -> str:
    """A gateway/web session id, or empty when the value is a placeholder.

    Observation stamps such as ``verify-a-obs`` are not chats. Linking them
    from the Dev tab would 404; the chip can still show PASS without an href.
    """
    sid = str(value or "").strip()
    return sid if _SESSION_ID.fullmatch(sid) else ""


def independent_reviews_ready(
    reviews: dict[str, Any] | None,
    *,
    generation_agent: str = "",
) -> bool:
    """True only when verify A and verify B both decided with real chats.

    A board stamp with ``verify-a-obs`` is not a reviewer. Both session ids
    must be distinct UUIDs, and neither agent may be the generator.
    """
    if not isinstance(reviews, dict):
        return False
    gen = str(generation_agent or "").strip().lower()
    seen_sids: set[str] = set()
    seen_agents: set[str] = set()
    for role in ("validation", "verification"):
        row = reviews.get(role) if isinstance(reviews.get(role), dict) else {}
        if str(row.get("status") or "") != "decided":
            return False
        if str(row.get("verdict") or "").upper() != "PASS":
            return False
        sid = usable_session_id(row.get("session_id"))
        if not sid or sid in seen_sids:
            return False
        seen_sids.add(sid)
        agent = str(row.get("agent") or "").strip().lower()
        if agent and gen and agent == gen:
            return False
        if agent:
            seen_agents.add(agent)
    return len(seen_agents) != 1


def chat_href(session_id: str, *, base: str = "") -> str:
    """Chat URL for the conductor. Tasks title and Dev chat use this."""
    return f"{base}/?session={session_id}"


def conductor_card(
    session_id: str,
    *,
    title: str = "Jarvis",
    agent: str = "jarvis",
    state: str = "",
    updated_at: str = "",
    base: str = "",
) -> dict:
    """Small Dev-tab pulse for the Jarvis conductor session.

    ``state`` is EMPTY when unknown, never "idle". The web session store does
    not track a live turn, so defaulting to "idle" made this projection assert
    something it cannot observe -- and the Dev tab, polling it every two
    seconds, repainted "idle" over the live probe's "needs input" and back
    again. Unknown has to be representable or the caller cannot tell the
    difference between "not running" and "no idea".
    """
    sid = str(session_id or "").strip()
    return {
        "session_id": sid,
        "href": chat_href(sid, base=base),
        "title": str(title or "Jarvis"),
        "agent": str(agent or "jarvis"),
        "state": str(state or ""),
        "updated_at": str(updated_at or ""),
    }


def backfill_review_gate_events(run: dict[str, Any], events: list[dict], reviews: dict, sdlc: SdlcStore) -> int:
    """Persist a missing tests validation/verification decision from the board.

    Runs that stamped A/B on the conductor card never wrote ``gate_decided``
    for verify A, so the chip stayed pending after both agreed PASS.
    """
    run_id = str(run.get("run_id") or "")
    if not run_id or not reviews:
        return 0
    have = {
        str(event.get("gate_kind") or "")
        for event in events or []
        if event.get("status") == "gate_decided"
        and str(event.get("gate_kind") or "") in {"validation", "verification"}
        and canonical_stage(str(event.get("stage") or "")) == "tests"
    }
    count = 0
    for kind in ("validation", "verification"):
        if kind in have:
            continue
        row = reviews.get(kind) if isinstance(reviews.get(kind), dict) else {}
        if str(row.get("status") or "") != "decided":
            continue
        digest = str(row.get("sha") or run.get("merged_head_sha") or "tests")
        agent, model = role_config(run, kind)
        sdlc.record_gate_decided(
            run_id,
            stage="tests",
            kind=kind,
            digest=digest,
            executor_agent=agent,
            executor_model=model,
            executor_session_id=usable_session_id(row.get("session_id")),
        )
        count += 1
    return count


#: Forge label applied to the issue while a stage runs, and used as the tag on
#: the stage's completion comment.  ``sdlc/`` namespaced so it never collides
#: with the repo's own ``type/`` and ``status/`` labels.
STAGE_TAGS: dict[str, str] = {
    "intake": "sdlc/intake",
    "worktree": "sdlc/worktree",
    "plan": "sdlc/plan",
    "design": "sdlc/design",
    "spec": "sdlc/design",  # retired name, kept so old runs still tag
    "code": "sdlc/code",
    "tests": "sdlc/tests",
    "review": "sdlc/deploy",  # legacy key for old stored events
    "deploy": "sdlc/deploy",
    "pr": "sdlc/pr",
    "merged": "sdlc/merged",
    "deployed": "sdlc/deployed",
    "blocked": "sdlc/blocked",
    "archived": "sdlc/archived",
}

#: The classic seven-phase SDLC, and the gate in this pipeline that owns each.
#:
#: This table is the figure on the Dev tab: the page renders its legend from
#: here, so a phase cannot be drawn as covered without a gate actually behind
#: it. Phase 7 is deliberately EMPTY — a run ends at its approved delivery
#: boundary; maintenance is a new run against that version.
PHASES: tuple[dict[str, Any], ...] = (
    {
        "n": 1,
        "name": "Identify and gather requirements",
        "gates": ("describe", "clarify", "contract"),
        "note": "This is the validation. Clarify freezes a WHEN/SHALL condition "
        "for plan and given/when/then/fail-if evidence for test. It may "
        "use web search as a tool to check current APIs and tools — "
        "that is not a separate stage. Those two lists become the issue.",
    },
    {
        "n": 2,
        "name": "Plan",
        "gates": ("plan",),
        "note": "Fused brief: approach, file/function map, and out of scope. "
        "Design is skipped on this path unless the operator splits the V.",
    },
    {
        "n": 3,
        "name": "Design",
        "gates": ("design",),
        "note": "Optional split-V only. The fused path clears this gate with the plan brief.",
    },
    {
        "n": 4,
        "name": "Implement",
        "gates": ("code",),
        "note": "Built in small committed increments, so a stage that runs out of "
        "budget still leaves work behind. LSP then checks the code is "
        "even legal.",
    },
    {
        "n": 5,
        "name": "Test",
        "gates": ("tests", "deploy"),
        "note": "Prove the agreed behavior on the named surface, then gather "
        "two independent reviews of the same evidence.",
    },
    {
        "n": 6,
        "name": "Deploy",
        "gates": ("ship",),
        "note": "After checks and independent reviews pass, follow the approved "
        "delivery choice: reviewed pull request, merge, or merge and "
        "deployment to the authorized target.",
    },
    {
        "n": 7,
        "name": "Maintain",
        "gates": (),
        "note": "Open a new change against the delivered version for fixes or further improvements.",
    },
)

#: The one phase this pipeline does not implement.
UNIMPLEMENTED_PHASE = 7

#: The V-model's claim: every definition on the left arm is answered by a check
#: on the right arm. Validation (are these the right requirements?) happens
#: once, at clarify/contract. Everything after that is development or
#: verification (does this output satisfy those frozen inputs?).
#:
#: Listed here are the pairs that are REAL in this pipeline. It deliberately
#: does not claim per-stage LLM validate, or the V-model's four separate test
#: levels: there is one tests stage and one deploy-readiness stage.
VEE_PAIRS: tuple[dict[str, str], ...] = (
    {
        "defines": "Condition + clearing evidence",
        "gate": "contract",
        "answered_by": "tests · verify A/B",
        "how": "This freeze is the validation. Tests drive each evidence line on "
        "the named surface; two independent models re-read the same "
        "evidence.",
    },
    {
        "defines": "Plan",
        "gate": "plan",
        "answered_by": "design",
        "how": "Generate a design that can satisfy every frozen condition. No second validation of the contract.",
    },
    {
        "defines": "Design",
        "gate": "design",
        "answered_by": "code",
        "how": "Generate the implementation the design named.",
    },
    {
        "defines": "Code",
        "gate": "code",
        "answered_by": "LSP · tests",
        "how": "LSP checks the code is legal. Tests drive each frozen evidence line on the real surface.",
    },
    {
        "defines": "Release",
        "gate": "ship",
        "answered_by": "merge · deploy",
        "how": "Merge when verify A, verify B, and LSP are green. Deploy stays "
        "a click; a post-deploy probe checks the service came back.",
    },
)


#: Run status values.  ``intake`` and ``ready`` are pre-issue; everything from
#: ``running`` on has a forge issue behind it.
RUN_STATUSES = frozenset(
    {
        "intake",
        "ready",
        "running",
        "awaiting_pr",
        "pr_open",
        "blocked",
        "done",
        "aborted",
    }
)

#: The marker an agent prints to close a stage.  Deterministic on purpose: an
#: idle session without it is a FAILED stage, never an inferred pass.
STAGE_MARKER_RE = re.compile(
    r"^\s*SDLC-STAGE-DONE:\s*(?P<stage>[a-z]+)\s+STATUS=(?P<status>PASS|FAIL)"
    r"(?:\s+NOTE=(?P<note>.*))?\s*$",
    re.IGNORECASE,
)

CRITERIA: tuple[dict[str, str], ...] = (
    {
        "id": "REQ-DEV-SDLC-0001",
        "text": "WHEN the operator describes a change THE SYSTEM SHALL "
        "interrogate until a WHEN/SHALL condition for plan and "
        "given/when/then/fail-if evidence for test both parse; that "
        "freeze is the validation. Clarify may look up current facts "
        "with web search; research is a tool of clarify, not a stage. "
        "NON-GOAL: skipping clarify, a research gate, or treating "
        "'tests pass' as evidence.",
    },
    {
        "id": "REQ-DEV-SDLC-0002",
        "text": "WHEN either frozen artifact is missing or misshapen THE SYSTEM "
        "SHALL NOT mint a forge issue. The operator ratifies both.",
    },
    {
        "id": "REQ-DEV-SDLC-0003",
        "text": "WHEN the operator ratifies THE SYSTEM SHALL write the condition "
        "and the clearing evidence into the issue body verbatim.",
    },
    {
        "id": "REQ-DEV-SDLC-0004",
        "text": "WHEN the issue is minted THE SYSTEM SHALL advance "
        "plan -> design -> code -> tests without further operator "
        "prompting; deploy-readiness is deterministic after the "
        "independent verifications and LSP.",
    },
    {"id": "REQ-DEV-SDLC-0005", "text": "Each finished stage is posted to the issue under its own sdlc/ tag."},
    {
        "id": "REQ-DEV-SDLC-0006",
        "text": "A rejecting independent-verification or LSP verdict loops back to code, bounded by max_laps.",
    },
    {"id": "REQ-DEV-SDLC-0007", "text": "A passing deploy-readiness verdict opens exactly one pull request."},
    {
        "id": "REQ-DEV-SDLC-0008",
        "text": "WHEN independent verification A, independent verification B, "
        "and LSP diagnostics are green THE SYSTEM SHALL merge; deploy "
        "stays operator-triggered.",
    },
    {"id": "REQ-DEV-SDLC-0009", "text": "An idle stage session without a completion marker fails the stage."},
    {"id": "REQ-DEV-SDLC-0010", "text": "Every run works in its own git worktree; a run without one never starts."},
    {
        "id": "REQ-DEV-SDLC-0011",
        "text": "WHEN a stage after the contract runs THE SYSTEM SHALL generate "
        "or verify against the frozen artifacts; it SHALL NOT "
        "re-validate whether those artifacts are the right "
        "requirements. Verification is LSP after code, driving the "
        "frozen evidence at tests, and two independent model re-reads "
        "of that same evidence.",
    },
    {
        "id": "REQ-DEV-SDLC-0012",
        "text": "Generation defaults to the local vLLM. The two independent "
        "verifiers use distinct selected agents, distinct from "
        "generation and from each other.",
    },
    {
        "id": "REQ-DEV-SDLC-0015",
        "text": "A blocked run resumes itself, bounded by a backoff, an attempt "
        "cap, and a reset on progress; it never waits for a human for "
        "an ordinary, self-clearing failure.",
    },
    {
        "id": "REQ-DEV-SDLC-0017",
        "text": "A gate's evidence is never crowded out by its context, and any "
        "truncation is announced so absence from an excerpt is not read "
        "as absence from the branch.",
    },
    {
        "id": "REQ-DEV-SDLC-0016",
        "text": "Traceability covers every requirement at function AND test "
        "level, and the tooling can see every requirement domain.",
    },
    {
        "id": "REQ-DEV-SDLC-0014",
        "text": "Intake accepts pasted or attached files; text is inlined into the "
        "detail and other files are stored and linked.",
    },
    {
        "id": "REQ-DEV-SDLC-0013",
        "text": "WHEN a PR is about to merge THE SYSTEM SHALL refuse unless "
        "independent verification A, independent verification B, and "
        "LSP are green against that commit; a stale review never "
        "ships. Deploy stays a click.",
    },
    {
        "id": "REQ-DEV-SDLC-0018",
        "text": "Generate stages record the generation executor. After tests "
        "the run records two distinct verification executors. "
        "Deploy-readiness records no third LLM.",
    },
    {
        "id": "REQ-DEV-SDLC-0019",
        "text": "The terminal automatic stage is sdlc/deploy, and runs stored under the retired review name resume without replaying finished work.",
    },
    {
        "id": "REQ-DEV-SDLC-0020",
        "text": "The Dev tab exposes an independent agent and model selector for each role on desktop and small mobile screens.",
    },
    {
        "id": "REQ-DEV-SDLC-0021",
        "text": "WHEN a later SDLC gate is submitted while an earlier gate is OPEN THE SYSTEM SHALL refuse the jump; CLEAR unlocks the next gate and LOOP names defects and returns to the producer.",
    },
    {
        "id": "REQ-DEV-SDLC-0022",
        "text": "WHEN a Dev run starts THE SYSTEM SHALL mint one Jarvis conductor session, register it on the Tasks board, and keep the Dev tab, the chat page, and the Tasks row pointing at that same session.",
    },
    {
        "id": "REQ-DEV-SDLC-0023",
        "text": "WHEN Jarvis conducts a run THE SYSTEM SHALL keep only the gate "
        "card in Jarvis context; child reasoning stays on the child "
        "session and is linked from the issue.",
    },
    {
        "id": "REQ-DEV-SDLC-0024",
        "text": "WHEN design runs THE SYSTEM SHALL use live web search for "
        "current APIs and tools and SHALL write spec/<branch-slug>.md "
        "on the run worktree.",
    },
    {
        "id": "REQ-DEV-SDLC-0025",
        "text": "WHEN a LOOP repeats the same SHA and the same defect set THE "
        "SYSTEM SHALL count no-progress; WHEN attempts reach 3 or "
        "no-progress reaches 2 THE SYSTEM SHALL BLOCK. NON-GOAL: "
        "Jarvis inferring a pass.",
    },
    {
        "id": "REQ-DEV-SDLC-0026",
        "text": "WHEN tests run THE SYSTEM SHALL drive each frozen evidence "
        "line on the named real surface. NON-GOAL: inventing a pytest "
        "suite as the oracle.",
    },
)


class SdlcError(RuntimeError):
    """Invalid SDLC control-plane input or state."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _unjson(raw: Any, default: Any) -> Any:
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Intake clarifier
# --------------------------------------------------------------------------


def _clarify_system(kind: str) -> str:
    """Tool-free synthesis uses the same compact contract for both run kinds."""
    from sharedagents_web.dev_phase_prompts import CLARIFY_KINDS, CLARIFY_SYSTEM

    kind = "fix" if str(kind or "").strip() == "fix" else "feature"
    return CLARIFY_SYSTEM + "\n" + CLARIFY_KINDS[kind]


def _config() -> dict[str, Any]:
    try:
        from sharedagents_core.kernel.config import load_global_config

        cfg = load_global_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:  # noqa: BLE001 — intake degrades, never 500s
        return {}


def _clarifier_model(cfg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Pick the model that drives intake questioning and the semantic gates.

    Prefers a LOCAL (``backend: vllm``) model. These calls are frequent — intake
    rounds plus a validate and a verify per stage — so defaulting to whatever
    the global config calls primary can quietly bill a provider several times
    per run, and stop the pipeline dead when that provider runs out.

    ``SHAREDAGENTS_SDLC_INTAKE_MODEL`` wins, so the operator can point it anywhere.
    """
    cfg = cfg if cfg is not None else _config()
    rows = [row for row in (cfg.get("models") or []) if isinstance(row, dict)]
    if not rows:
        return None
    wanted = (os.environ.get("SHAREDAGENTS_SDLC_INTAKE_MODEL") or "").strip()
    if wanted:
        for row in rows:
            if wanted in (row.get("alias"), row.get("id")):
                return row
    local = [row for row in rows if str(row.get("backend") or "").lower() == "vllm"]
    for candidates in (local, rows):
        for row in candidates:
            if str(row.get("role") or "") == "primary":
                return row
        if candidates:
            return candidates[0]
    return None


#: A verdict is a small JSON object, but a thinking model emits its reasoning
#: first and that is what has to fit. Applied ONLY to gates; intake questioning
#: shares `_chat` and has no business being truncated to a gate's budget.
#:
#: 2048 was set from a 26KB prompt that used 258 completion tokens. It was far
#: too tight, and the harness said so: after the prompt budget rose to 120KB,
#: EVERY gate in production came back "reply was cut off at the token limit"
#: and was recorded unusable. Measured again on a real 62KB gate prompt
#: (14,856 prompt tokens): 1,782 completion tokens, finish_reason=stop -- 13%
#: under the old cap, with production evidently past it.
#:
#: A cut-off reply is strictly worse than a slow one: a slow gate still
#: decides, while a truncated gate decides nothing and marks the run. So the
#: headroom is generous -- ~4.5x the largest measured reply, still a small
#: fraction of the model's context beside a 50k-token prompt.
MAX_GATE_COMPLETION_TOKENS = int(os.environ.get("SHAREDAGENTS_SDLC_GATE_COMPLETION_TOKENS") or 8192)


def _gate_chat(messages: list[dict[str, str]]) -> str:
    """`_chat` with the gate's completion ceiling."""
    return _chat(messages, max_tokens=MAX_GATE_COMPLETION_TOKENS)


_CLARIFY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "websearch",
            "description": (
                "Search the live web. Use for current APIs, SDKs, CLIs, "
                "library versions, vendor docs, and product surfaces."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Web search query.",
                    },
                    "numResults": {
                        "type": "number",
                        "description": "How many results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_repo",
            "description": (
                "List files in the named local clone. Use on a fix to find "
                "the failing surface before asking the operator."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path inside the clone (default .).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_repo",
            "description": (
                "Read a file from the named local clone. Use on a fix to "
                "inspect the failing code and try to reproduce the error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path inside the clone.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]
MAX_CLARIFY_SEARCHES = 6


def _run_websearch(query: str, *, count: int = 5) -> str:
    """Brave websearch for clarify. Failures become text, not a thrown gate."""
    q = " ".join(str(query or "").split())
    if not q:
        return "(websearch skipped: empty query)"
    try:
        import asyncio

        from sharedagents_agent.tools.impl.search import run_web_search_async

        result = asyncio.run(
            run_web_search_async(
                {
                    "query": q,
                    "numResults": max(1, min(8, int(count or 5))),
                }
            )
        )
        text = str(getattr(result, "content", "") or "").strip()
        return text[:4000] if text else "(no search results)"
    except Exception as exc:  # noqa: BLE001 — search is a tool, not a veto
        log.warning("sdlc: clarify websearch failed: %s", exc)
        return f"(websearch unavailable: {type(exc).__name__}: {exc})"


def _repo_jail(repo: str, rel: str) -> Path | None:
    """Resolve ``rel`` inside the named clone, or None if it escapes."""
    try:
        root = repo_dir(repo).resolve()
    except Exception:  # noqa: BLE001 — missing repo is a tool miss, not a crash
        return None
    if not root.is_dir():
        return None
    rel = str(rel or ".").lstrip("/")
    if ".." in Path(rel).parts:
        return None
    target = (root / rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def _list_repo(repo: str, rel: str = ".") -> str:
    target = _repo_jail(repo, rel)
    if not target:
        return f"(clone not found for {repo or '(no repo)'})"
    try:
        root = repo_dir(repo).resolve()
    except Exception:  # noqa: BLE001
        return f"(clone not found for {repo})"
    if target.is_file():
        return str(target.relative_to(root))
    rows: list[str] = []
    for path in sorted(target.rglob("*")):
        parts = path.relative_to(root).parts
        if any(part == ".git" or part.startswith(".") for part in parts):
            continue
        if path.is_file():
            rows.append(str(path.relative_to(root)))
        if len(rows) >= 80:
            rows.append("…")
            break
    return "\n".join(rows) or "(empty)"


def _read_repo(repo: str, rel: str) -> str:
    if not str(rel or "").strip():
        return "(read_repo needs a path)"
    target = _repo_jail(repo, rel)
    if not target or not target.is_file():
        return f"(file not found: {rel})"
    data = target.read_bytes()[:12000]
    return data.decode("utf-8", errors="replace")


def _attempt_fix_reproduction(repo: str, headline: str, detail: str) -> str:
    """Inspect the clone so a fix interview starts from the actual product."""
    lines = [
        f"object: {headline}",
        f"detail: {detail}" if detail else "",
    ]
    listing = _list_repo(repo, ".")
    lines.append("clone files:")
    lines.append(listing)
    for name in ("index.html", "README.md", "src/index.html"):
        text = _read_repo(repo, name)
        if text.startswith("(file not found"):
            continue
        lines.append(f"--- {name} ---")
        lines.append(text[:8000])
    return "\n".join(item for item in lines if item)[:14000]


_REPRO_MARKERS = (
    "actual",
    "expected",
    "currently",
    "instead",
    "reproduc",
    "failing sequence",
    "i pressed",
    "i saw",
    "then press",
)


def _has_reproduction(
    transcript: Iterable[dict[str, str]],
    requirements: list[str],
    criteria: list[str],
) -> bool:
    """True when the failing sequence and actual-vs-expected are on record."""
    blob = " ".join(
        [str(entry.get("answer") or "") for entry in transcript] + list(requirements) + list(criteria)
    ).lower()
    if not any(marker in blob for marker in _REPRO_MARKERS):
        return False
    return any(token in blob for token in ("when ", "given ", "press", "click"))


def _fix_reproduce_questions(headline: str, detail: str) -> list[dict[str, str]]:
    object_line = " ".join((headline or "").split()) or "the reported failure"
    detail_line = " ".join((detail or "").split())
    suggestion = (
        f"WHEN a user performs the failing sequence for {object_line} "
        f"THE SYSTEM SHALL show the expected new behavior and SHALL NOT "
        f"keep the current wrong one."
    )
    evidence = (
        f"given the current product for {object_line} / when the failing "
        f"sequence in the object/detail is driven / then the expected "
        f"observation appears / fail-if the current wrong observation remains"
    )
    if detail_line:
        suggestion += f" Sequence: {detail_line[:240]}"
    return [
        {
            "id": "q1",
            "text": (
                "I will try to reproduce this error on the named clone. Confirm "
                "the failing steps, what you see now, and what you want instead."
            ),
            "why": "for the evidence: a fix cannot freeze until the error is reproduced",
            "suggestion": f"{suggestion} Evidence: {evidence}",
        }
    ]


def _completion(
    messages: list[dict[str, Any]],
    *,
    timeout: float = 120.0,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    """One chat/completions call. Returns ``(message, finish_reason)``."""
    model = _clarifier_model()
    if not model:
        raise SdlcError("no model is configured for intake questioning")
    base = str(model.get("baseUrl") or "").rstrip("/")
    if not base:
        raise SdlcError("configured intake model has no baseUrl")
    body: dict[str, Any] = {
        "model": str(model.get("id") or ""),
        "messages": messages,
        "stream": False,
    }
    apply_sampling(
        body,
        configured_sampling(
            _config(),
            model,
            str(model.get("id") or ""),
            purpose="intake",
        ),
    )
    if max_tokens:
        body["max_tokens"] = int(max_tokens)
    if tools:
        body["tools"] = tools
    # Qwen3-class local models otherwise think past the urllib timeout.
    body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key_env = str(model.get("apiKeyEnv") or "").strip()
    if key_env and os.environ.get(key_env):
        headers["Authorization"] = f"Bearer {os.environ[key_env]}"
    for name, value in (model.get("headers") or {}).items():
        headers[str(name)] = str(value)
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(MAX_MODEL_RESPONSE_BYTES + 1)
            if len(raw) > MAX_MODEL_RESPONSE_BYTES:
                raise SdlcError("model response exceeded the size limit")
            payload = json.loads(raw.decode("utf-8", errors="replace") or "{}")
    except urllib.error.HTTPError as exc:
        hint = (
            " (prompt may exceed the model's context; set SHAREDAGENTS_SDLC_GATE_PROMPT_CHARS lower)"
            if exc.code == 400
            else ""
        )
        raise SdlcError(f"intake model HTTP {exc.code}{hint}") from exc
    except Exception as exc:  # noqa: BLE001
        raise SdlcError(f"intake model unreachable: {type(exc).__name__}") from exc
    try:
        choice = payload["choices"][0]
        message = dict(choice.get("message") or {})
    except (KeyError, IndexError, TypeError) as exc:
        raise SdlcError("intake model returned no content") from exc
    if str(choice.get("finish_reason") or "") == "length":
        raise SdlcError("intake model reply was cut off at the token limit")
    return message, str(choice.get("finish_reason") or "")


def _chat(messages: list[dict[str, str]], *, timeout: float = 120.0, max_tokens: int | None = None) -> str:
    """One blocking OpenAI-compatible completion against the configured model."""
    message, _reason = _completion(
        messages,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    content = str(message.get("content") or "")
    if not content:
        raise SdlcError("intake model returned no content")
    return content


def _clarify_chat(
    messages: list[dict[str, Any]], *, timeout: float = 180.0, max_tokens: int | None = None, repo: str = ""
) -> str:
    """Clarify completion that can inspect the clone or search before freezing."""
    working = [dict(item) for item in messages]
    for _ in range(MAX_CLARIFY_SEARCHES + 1):
        message, reason = _completion(
            working,
            timeout=timeout,
            max_tokens=max_tokens,
            tools=_CLARIFY_TOOLS,
        )
        calls = list(message.get("tool_calls") or [])
        if calls and reason == "tool_calls":
            working.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": calls,
                }
            )
            for call in calls[:MAX_CLARIFY_SEARCHES]:
                fn = call.get("function") if isinstance(call, dict) else {}
                name = str((fn or {}).get("name") or "")
                raw_args = str((fn or {}).get("arguments") or "") or "{}"
                try:
                    args = json.loads(raw_args)
                except (TypeError, ValueError):
                    args = {}
                if name == "websearch":
                    result = _run_websearch(
                        str(args.get("query") or ""),
                        count=int(args.get("numResults") or 5),
                    )
                elif name == "list_repo":
                    result = _list_repo(repo, str(args.get("path") or "."))
                elif name == "read_repo":
                    result = _read_repo(repo, str(args.get("path") or ""))
                else:
                    result = f"(unknown tool {name})"
                working.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or "websearch"),
                        "content": result,
                    }
                )
            continue
        content = str(message.get("content") or "")
        if not content:
            raise SdlcError("intake model returned no content")
        return content
    raise SdlcError("clarify used websearch too many times without freezing")


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Best-effort strict-JSON extraction from a model reply."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        start, end = text.find("{"), text.rfind("}")
        parsed = None
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except (TypeError, ValueError):
                parsed = None
        if parsed is None:
            try:
                from sharedagents_model.json_repair import repair_truncated_json

                parsed = repair_truncated_json(text)
            except Exception:  # noqa: BLE001
                parsed = None
    if not isinstance(parsed, dict):
        raise SdlcError("intake model did not return a JSON object")
    return parsed


def _strings(value: Any, *, limit: int = 40) -> list[str]:
    out: list[str] = []
    for item in value if isinstance(value, list) else []:
        text = str(item if not isinstance(item, dict) else item.get("text") or "").strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


_VAGUE_CONDITION = (
    "fix it",
    "implement the feature",
    "make it work",
    "as discussed",
    "do the thing",
    "the change",
    "ship it",
)


def _is_condition_line(text: str) -> bool:
    """A plan-facing condition: EARS-shaped, not a test procedure."""
    low = " ".join((text or "").lower().split())
    if not low or low in _VAGUE_CONDITION:
        return False
    if "fail-if" in low or "fail if" in low:
        return False
    return any(
        token in low
        for token in (
            "shall ",
            "must ",
            "when ",
            "if ",
            "the system",
            "non-goal",
        )
    )


#: Phrases that assert nothing about what "correct" would look like.
_VAGUE_ORACLES: tuple[str, ...] = (
    "works as expected",
    "looks good",
    "no regressions",
    "tests pass",
    "should work",
    "as discussed",
)


def _then_clause_is_only_platitude(low: str) -> bool:
    """True when the THEN clause says nothing checkable once platitudes go.

    `low` is already lowercased and whitespace-collapsed.
    """
    if not any(re.search(rf"\b{re.escape(p)}\b", low) for p in _VAGUE_ORACLES):
        return False
    segments = [seg.strip() for seg in low.split("/")]
    then = next(
        (seg for seg in segments if seg.startswith("then ")),
        next((seg for seg in segments if " then " in seg), low),
    )
    # A numeric process status is a deterministic oracle even when phrased
    # compactly ("the exit code is 0").  The generic residue-length heuristic
    # below rejected that exact phrase because ``exit code 0`` is eleven
    # characters, while accepting the semantically identical longer wording
    # "the process exits with code 0".
    if re.search(
        r"\b(?:exit|status|return)\s+code\s+(?:is\s+|equals?\s+)?\d+\b",
        then,
    ):
        return False
    rest = then
    for phrase in _VAGUE_ORACLES:
        rest = re.sub(rf"\b{re.escape(phrase)}\b", " ", rest)
    # Drop the scaffolding words that carry no assertion of their own.
    rest = re.sub(r"\b(then|and|the|a|an|all|is|are|it|that|of|to|with)\b", " ", rest)
    rest = re.sub(r"[^a-z0-9]+", " ", rest).strip()
    return len(rest) < 12


def _is_evidence_line(text: str) -> bool:
    """A test-facing oracle: given / when / then / fail-if."""
    low = " ".join((text or "").lower().split())
    if not low:
        return False
    # These are rejected as THE oracle -- "then tests pass" says nothing about
    # what passing would mean. But only as the WHOLE oracle: a line may mention
    # one alongside something checkable, and rejecting that is wrong twice
    # over. Both misfires happened live and both blocked ratification on the
    # clarifier's own output:
    #   * a bare substring test rejected "...reports all unit tests passing"
    #     (it contains "tests pass" inside "tests passing"); and
    #   * word boundaries alone still rejected "then the process exits with
    #     code 0 and all tests pass", whose oracle is an exit code.
    # So strip the platitude out of the `then` clause and ask whether anything
    # checkable is left.
    if _then_clause_is_only_platitude(low):
        return False
    has_given = "given " in low or low.startswith("given")
    has_when = " when " in low or low.startswith("when ")
    has_then = " then " in low or " pass " in low or "/ then" in low
    has_fail = "fail-if" in low or "fail if" in low
    return has_given and has_when and has_then and has_fail


def contract_defects(requirements: list[str], criteria: list[str]) -> list[str]:
    """Why this contract cannot yet feed plan and test. Empty means freezeable."""
    defects: list[str] = []
    if not requirements:
        defects.append("no condition for plan")
    else:
        for index, line in enumerate(requirements, 1):
            if not _is_condition_line(line):
                defects.append(f"condition {index} is not a clear WHEN/SHALL statement for plan")
    if not criteria:
        defects.append("no clearing evidence for test")
    else:
        for index, line in enumerate(criteria, 1):
            if not _is_evidence_line(line):
                defects.append(f"evidence {index} is not given/when/then/fail-if")
    defects += _duplicate_defects(requirements, "condition")
    defects += _duplicate_defects(criteria, "evidence")
    return defects


def _duplicate_defects(lines: list[str], what: str) -> list[str]:
    """A repeated line is a line that was never written.

    A clarifier can degenerate and emit the same sentence ten times. Every one
    of them is well-formed, so the shape checks above pass and the contract
    freezes claiming ten behaviours while covering one -- then `tests` judges
    the delivery against nine criteria that were never really asked for. Seen
    live: ten identical "when the user presses 1 ... display shows '1'" lines.
    """
    seen: dict[str, int] = {}
    defects: list[str] = []
    for index, line in enumerate(lines or [], 1):
        key = " ".join(str(line or "").lower().split())
        if not key:
            continue
        first = seen.get(key)
        if first is None:
            seen[key] = index
        else:
            defects.append(
                f"{what} {index} repeats {what} {first} word for word — each line must cover something different"
            )
    return defects


def _gap_questions(
    defects: list[str],
    *,
    kind: str = "feature",
    headline: str = "",
    detail: str = "",
) -> list[dict[str, str]]:
    """The single next hole. The following hole waits for this answer."""
    object_line = " ".join((headline or "").split()) or "the named object"
    detail_line = " ".join((detail or "").split())
    fix = str(kind or "").strip() == "fix"
    out: list[dict[str, str]] = []
    for index, defect in enumerate(defects[:1], 1):
        if fix and ("no condition" in defect or defect.startswith("condition")):
            text = (
                "What is the failing sequence, what happens now, and what shall happen instead? Write one WHEN/SHALL."
            )
            why = "for the condition: " + defect
            suggestion = (
                f"WHEN a user performs the failing sequence for {object_line} "
                f"THE SYSTEM SHALL show the expected new behavior and SHALL NOT "
                f"keep the current wrong one."
            )
            if detail_line:
                suggestion += f" Sequence: {detail_line[:240]}"
        elif "no condition" in defect:
            text = "What must be true of the system when this run is done?"
            why = "for the condition: plan needs a WHEN/SHALL line"
            suggestion = (
                f"WHEN the operator uses the surface for {object_line} "
                f"THE SYSTEM SHALL show the new behavior and SHALL NOT keep "
                f"the old one."
            )
        elif "no clearing evidence" in defect:
            text = "How will we see that condition become true on the real surface?"
            why = "for the evidence: test needs given/when/then/fail-if"
            suggestion = (
                f"given the current product for {object_line} / when the "
                f"operator performs the action / then the new behavior is "
                f"visible / fail-if the old behavior still appears"
            )
        elif defect.startswith("condition"):
            text = f"Rewrite the condition about {object_line} as a WHEN <situation> THE SYSTEM SHALL <state> line."
            why = "for the condition: " + defect
            suggestion = (
                f"WHEN the operator uses {object_line} THE SYSTEM SHALL "
                f"show the named new behavior and SHALL NOT change anything "
                f"outside that scope."
            )
        else:
            text = "Rewrite the clearing evidence as given <setup> / when <action> / then <pass> / fail-if <red>."
            why = "for the evidence: " + defect
            suggestion = (
                f"given the product for {object_line} / when the operator "
                f"performs the action / then the pass observation is visible "
                f"/ fail-if the red observation appears"
            )
        out.append(
            {
                "id": f"q{index}",
                "text": text,
                "why": why,
                "suggestion": suggestion,
            }
        )
    return out


def _questions(value: Any, *, limit: int = 1) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for index, item in enumerate(value if isinstance(value, list) else []):
        if isinstance(item, str):
            text, why, suggestion = item.strip(), "", ""
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("question") or "").strip()
            why = str(item.get("why") or item.get("reason") or "").strip()
            suggestion = str(
                item.get("suggestion") or item.get("suggested") or item.get("default") or item.get("answer") or ""
            ).strip()
        else:
            continue
        if not text:
            continue
        # A "suggestion" that merely echoes the question is worse than none:
        # it looks answered in the UI while deciding nothing.
        if suggestion and suggestion.rstrip("?").strip().lower() == text.rstrip("?").strip().lower():
            suggestion = ""
        out.append(
            {
                "id": f"q{index + 1}",
                "text": text,
                "why": why,
                "suggestion": suggestion,
            }
        )
        if len(out) >= limit:
            break
    return out


#: How much of the survey may ride in the intake prompt.
SURVEY_MAX_CHARS = int(os.environ.get("SHAREDAGENTS_SDLC_SURVEY_CHARS") or 3000)

#: Files whose names describe the project rather than implement it.
_SURVEY_DOCS = (
    "AGENTS.md",
    "README.md",
    "README.rst",
    "CONTRIBUTING.md",
    "ARCHITECTURE.md",
    "CHANGES.rst",
    "CHANGELOG.md",
    "docs/index.rst",
    "docs/index.md",
)

_POLICY_TERMS = re.compile(
    r"\b(branch|worktree|merge|production|deploy|rollback|index|lock|mutex|"
    r"deadlock|mmap|sidecar|atomic|test|docker)\b",
    re.IGNORECASE,
)

_POLICY_REFERENCE = re.compile(
    r"(?P<path>(?:docs/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:md|rst))",
    re.IGNORECASE,
)

_CAUSAL_RISK_TERMS = re.compile(
    r"\b(?:non[- ]?reentrant|self[- ]?deadlocks?|deadlocks?|brief[- ]?hold|"
    r"do not hold|must not|never|corrupt|use-after|stale (?:view|pointer)|"
    r"spin(?:s|ning)?|atomic(?:ally)?|rollback)\b",
    re.IGNORECASE,
)


def _survey_policy_sources(tree: Path, files: list[str]) -> list[str]:
    """Policy roots plus tracked documents they explicitly require reading."""
    tracked = set(files)
    sources = [name for name in ("AGENTS.md", "CONTRIBUTING.md") if name in tracked]
    # One bounded level is enough to follow "read docs/locking.md" without
    # turning intake into a documentation crawler.
    for name in list(sources):
        try:
            text = (tree / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _POLICY_REFERENCE.finditer(text):
            candidate = match.group("path").replace("\\", "/")
            if (
                candidate in tracked
                and candidate not in sources
                and not candidate.startswith("/")
                and ".." not in candidate.split("/")
            ):
                sources.append(candidate)
            if len(sources) >= 8:
                return sources
    return sources


def _survey_policy(tree: Path, files: list[str]) -> list[str]:
    """Bounded policy facts needed before a branch or safety question exists."""
    critical: list[str] = []
    ordinary: list[str] = []
    for name in _survey_policy_sources(tree, files):
        try:
            lines = (tree / name).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            clean = " ".join(line.strip().lstrip("-#| ").split())
            if not clean or not (_POLICY_TERMS.search(clean) or _CAUSAL_RISK_TERMS.search(clean)):
                continue
            fact = f"{name}: {clean}"[:300]
            bucket = critical if _CAUSAL_RISK_TERMS.search(clean) else ordinary
            if fact not in critical and fact not in ordinary:
                bucket.append(fact)
    # High-blast warnings have to survive the prompt ceiling. Generic layout
    # facts follow them rather than crowding them out.
    return (critical + ordinary)[:24]


def _survey_languages(files: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in files:
        if "." not in name.rsplit("/", 1)[-1]:
            continue
        suffix = "." + name.rsplit(".", 1)[-1].lower()
        if suffix in {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".cjs",
            ".mjs",
            ".go",
            ".rs",
            ".rb",
            ".java",
            ".c",
            ".h",
            ".cpp",
        }:
            counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:5])


def _survey_packages(files: list[str]) -> list[str]:
    """Import roots, i.e. where the implementation actually lives."""
    roots: dict[str, int] = {}
    for name in files:
        parts = name.split("/")
        if len(parts) < 2 or _looks_like_test(name):
            continue
        if parts[-1] == "__init__.py":
            roots["/".join(parts[:-1])] = roots.get("/".join(parts[:-1]), 0) + 1
        elif parts[0] in ("src", "lib", "app") and len(parts) > 2:
            roots["/".join(parts[:2])] = roots.get("/".join(parts[:2]), 0) + 1
    ordered = sorted(roots.items(), key=lambda kv: (kv[0].count("/"), -kv[1]))
    return [name for name, _n in ordered[:8]]


def _survey_hot_files(path: Path, *, runner: Any = None) -> list[str]:
    """Files that change most often -- the ones a change usually touches.

    Cheap stand-in for a co-change graph: no index, one `git log`.
    """
    code, out = _git(["log", "--name-only", "--pretty=format:", "-n", "150"], cwd=path, runner=runner)
    if code != 0:
        return []
    seen: dict[str, int] = {}
    for line in out.splitlines():
        name = line.strip()
        # CODE only. On a freshly imported repo the top of this list was
        # `.editorconfig`, `.flake8` and `.github/ISSUE_TEMPLATE/*` -- true,
        # and useless for deciding where a change goes.
        if (
            name
            and not _looks_like_test(name)
            and not _is_manifest(name)
            and "." + name.rsplit(".", 1)[-1].lower() in _MUTABLE_SUFFIXES
        ):
            seen[name] = seen.get(name, 0) + 1
    return [n for n, _c in sorted(seen.items(), key=lambda kv: -kv[1])[:8]]


def _survey_relevant(files: list[str], headline: str) -> list[str]:
    """Paths whose names overlap the request. Keyword-cheap, not semantic."""
    words = {
        w
        for w in re.findall(r"[a-z_]{4,}", str(headline or "").lower())
        if w
        not in {
            "this",
            "that",
            "with",
            "from",
            "when",
            "then",
            "should",
            "which",
            "issue",
            "test",
            "tests",
            "repository",
        }
    }
    if not words:
        return []
    scored: list[tuple[int, str]] = []
    for name in files:
        if _looks_like_test(name) or _is_manifest(name):
            continue
        if "." + name.rsplit(".", 1)[-1].lower() not in _MUTABLE_SUFFIXES:
            continue  # prose cannot be "where the change goes"
        # WHOLE tokens: substring matching put README.rst at the top of a
        # request containing the word "read".
        tokens = set(re.findall(r"[a-z_]+", name.lower()))
        hits = len(words & tokens)
        if hits:
            scored.append((hits, name))
    scored.sort(key=lambda pair: (-pair[0], len(pair[1])))
    return [name for _h, name in scored[:8]]


def _survey_by_content(tree: Path, files: list[str], headline: str, *, runner: Any = None) -> list[str]:
    """Files whose CONTENT mentions the request. Used when paths do not.

    Path matching works when filenames mirror the domain (`config.py` for a
    request about config) and finds NOTHING when they do not: on sharedagents,
    "contract coverage gate" matches no filename, while `git grep tautology`
    lands on the exact module. One bounded grep per word, no index.
    """
    words = sorted(
        {
            w
            for w in re.findall(r"[a-z_]{5,}", str(headline or "").lower())
            if w
            not in {"about", "should", "which", "there", "these", "where", "issue", "tests", "repository", "parameter"}
        },
        key=len,
        reverse=True,
    )[:3]
    if not words:
        return []
    mutable = set(mutable_files(files))
    hits: dict[str, int] = {}
    for word in words:
        code, out = _git(["grep", "-l", "-i", "-F", "--", word], cwd=tree, runner=runner)
        if code != 0:
            continue  # 1 = no match, which is not an error
        for line in out.splitlines()[:200]:
            name = line.strip()
            if name in mutable:
                hits[name] = hits.get(name, 0) + 1
    ordered = sorted(hits.items(), key=lambda kv: (-kv[1], len(kv[0])))
    return [name for name, _n in ordered[:8]]


_OUTLINE_LINE = re.compile(
    r"^[ \t]*(?:async\s+)?(?:def|class|function|export\s+(?:default\s+)?function)\s+[A-Za-z_$][\w$]*\s*\(?[^\n]{0,120}",
    re.MULTILINE,
)


def _survey_outline(tree: Path, name: str, *, limit: int = 14) -> list[str]:
    """The signatures in one file -- what "let me go read it" would return.

    The interview kept answering `{"questions": [], "summary": "Researching the
    existing implementation before freezing requirements"}`: it deferred to
    research and produced nothing, which is the stall. Naming the file is not
    enough; hand it the shape of the file so there is nothing left to go and
    look up.
    """
    path = Path(tree) / name
    try:
        if not path.is_file() or path.stat().st_size > 400_000:
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    seen: list[str] = []
    for match in _OUTLINE_LINE.finditer(text):
        line = match.group(0).strip().rstrip(":").rstrip("{").strip()
        if line and line not in seen:
            seen.append(line)
        if len(seen) >= limit:
            break
    return seen


def survey_repo(repo: str, *, headline: str = "", branch: str = "", runner: Any = None) -> dict[str, Any]:
    """What a developer would read BEFORE asking the operator anything.

    The intake conductor used to open its interview knowing nothing about the
    repository, so on a real codebase it spent its whole first turn EXPLORING
    and ended without asking a single question -- the run then sat in `intake`
    with nobody able to move it. Reading the repo is not the operator's job to
    answer, so it should not be a question: it is derived here, deterministically
    and without a model, and handed to the interview as fact.

    Everything in here is cheap: one `git ls-files`, one `git log`, and the
    check detection the verify stage already does.
    """
    out: dict[str, Any] = {"repo": repo}
    try:
        path = repo_dir(repo)
    except WorktreeError as error:
        return {**out, "unavailable": str(error)}
    worktree = (existing_worktree(path, branch, runner=runner) if branch else "") or str(path)
    tree = Path(worktree)
    if not (tree / ".git").exists() and not path.is_dir():
        return {**out, "unavailable": "no local checkout"}
    code, listing = _git(["ls-files"], cwd=tree, runner=runner)
    files = [line.strip() for line in listing.splitlines() if line.strip()] if code == 0 else []
    if not files:
        return {**out, "unavailable": "git listed no files"}

    tests = [name for name in files if _looks_like_test(name)]
    # `_looks_like_test` matches any path containing "test", which on flask
    # meant `.github/workflows/tests.yaml` and `docs/tutorial/tests.rst` were
    # offered as examples of what a test looks like.
    test_code = sorted(
        (name for name in tests if "." + name.rsplit(".", 1)[-1].lower() in _MUTABLE_SUFFIXES),
        # Shallowest first: the project's own suite, not the one inside
        # `examples/`, is what a new test should be modelled on.
        key=lambda name: (name.count("/"), name),
    )
    out["files"] = len(files)
    out["languages"] = _survey_languages(files)
    out["packages"] = _survey_packages(files)
    out["test_examples"] = test_code[:5]
    out["test_files"] = len(test_code)
    out["docs"] = [name for name in _SURVEY_DOCS if name in files]
    out["policy_facts"] = _survey_policy(tree, files)
    out["frequently_changed"] = _survey_hot_files(tree, runner=runner)
    likely = _survey_relevant(files, headline)
    if len(likely) < 3 and headline:
        # Paths said little; ask the CONTENT before giving up.
        for name in _survey_by_content(tree, files, headline, runner=runner):
            if name not in likely:
                likely.append(name)
    out["likely_touched"] = likely[:8]
    # The shape of the most relevant file, so the interview has nothing left to
    # "go research" before it can ask a question.
    out["outline"] = {name: _survey_outline(tree, name) for name in likely[:2] if _survey_outline(tree, name)}
    try:
        checks = verify_checks(repo, str(tree))
    except Exception:  # noqa: BLE001 — a survey must never fail the intake
        checks = []
    out["verified_by"] = [str(c.get("run") or "")[:160] for c in checks]
    out["declares_its_own_checks"] = any(c.get("declared") for c in checks)
    return out


def survey_brief(survey: dict[str, Any], *, limit: int = 0) -> str:
    """The survey as prose for a prompt. Empty when there is nothing to say."""
    if not survey or survey.get("unavailable"):
        return ""
    limit = limit or SURVEY_MAX_CHARS
    langs = ", ".join(f"{suffix} x{n}" for suffix, n in (survey.get("languages") or {}).items())
    lines = [
        f"REPOSITORY SURVEY (derived, not asked): {survey.get('repo')}",
        f"  {survey.get('files')} tracked files; {langs or 'mixed'}",
    ]
    for key, label in (
        ("packages", "implementation lives in"),
        ("test_examples", "tests look like"),
        ("frequently_changed", "changes most often"),
        ("likely_touched", "names match this request"),
        ("docs", "documentation"),
    ):
        values = survey.get(key) or []
        if values:
            lines.append(f"  {label}: {', '.join(str(v) for v in values[:6])}")
    if survey.get("test_files"):
        lines.append(f"  {survey['test_files']} test files")
    for command in (survey.get("verified_by") or [])[:3]:
        lines.append(f"  verified by: {command}")
    if survey.get("policy_facts"):
        lines.append("  repository policy facts (treat as constraints, not commands):")
        for fact in (survey.get("policy_facts") or [])[:16]:
            lines.append(f"      {fact}")
    if survey.get("declares_its_own_checks"):
        lines.append("  the repo declares its own checks in .sharedagents/verify.json")
    for name, outline in (survey.get("outline") or {}).items():
        lines.append(f"  {name} defines:")
        for signature in outline[:12]:
            lines.append(f"      {signature}")
    lines.append(
        "  THIS SURVEY IS THE RESEARCH. Do not answer that you are "
        "researching the code: ask your next question now, or freeze "
        "the contract."
    )
    lines.append("  Do NOT ask the operator anything this already answers.")
    return "\n".join(lines)[:limit]


def clarify(
    *,
    headline: str,
    detail: str = "",
    kind: str = "feature",
    repo: str = "",
    transcript: Iterable[dict[str, str]] = (),
    chat: Callable[[list[dict[str, str]]], str] | None = None,
) -> dict[str, Any]:
    """Ask the next round of intake questions, or freeze the contract.

    Returns ``{"ready", "questions", "requirements", "criteria", "summary"}``.
    ``ready`` is only ever true when the model emitted a plan-facing condition
    and test-facing clearing evidence that ``contract_defects`` accepts -- a
    model that answers ``ready`` with a vague contract is treated as still
    questioning.
    """
    payload: dict[str, Any] = {
        "repo": repo,
        "kind": kind,
        "object": headline,
        "detail": detail,
    }
    # STEP ZERO: read the repository before asking the operator about it.
    # Without this the conductor opens the interview knowing nothing, and on a
    # real codebase it spends the whole turn exploring and asks nothing at all.
    # None of this is the operator's to answer, so none of it may be a question.
    try:
        survey = survey_repo(repo, headline=headline)
        brief = survey_brief(survey)
        if brief:
            payload["codebase"] = brief
    except Exception:  # noqa: BLE001 — never let the survey block the interview
        log.warning("sdlc: repo survey failed for %s", repo, exc_info=True)
    rows = list(transcript)
    from sharedagents_web.dev_clarify import MAX_CLARIFY_ROUNDS, answered_rounds

    payload["answered_rounds"] = len(rows)
    if answered_rounds(rows) >= MAX_CLARIFY_ROUNDS:
        payload["convergence"] = (
            "The required sequential interview is complete. The newest "
            "operator answer supersedes conflicting earlier suggestions, but "
            "it does NOT override surveyed repository safety constraints. Do "
            "not ask again about a subject any answer "
            "settled. Synthesize ready=true requirements and evidence now; ask "
            "one more question only when a decision is genuinely absent from "
            "the entire transcript."
        )
    if str(kind or "").strip() == "fix":
        payload["reproduction_attempt"] = _attempt_fix_reproduction(
            repo,
            headline,
            detail,
        )
        payload["instruction"] = (
            "This is a FIX. Use the supplied request and reproduction notes; "
            "source inspection is not an executed reproduction. Ask only for "
            "missing failing steps, actual observation, or expected behavior. "
            "If those facts are complete, freeze without a confirmation round."
        )
    rounds = [
        {"role": "system", "content": _clarify_system(kind)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    for entry in rows:
        question = str(entry.get("question") or "").strip()
        answer = str(entry.get("answer") or "").strip()
        source = str(entry.get("source") or "operator").strip()
        if question:
            rounds.append({"role": "assistant", "content": question})
        if answer:
            prefix = "[source=suggested] " if source == "suggested" else ""
            rounds.append({"role": "user", "content": prefix + answer})
    # Thinking models can spend the whole urlopen budget on reasoning and
    # then raise TimeoutError, which is what blocked the calculator prove
    # on the dev sandbox (intake model unreachable). Cap the completion so the
    # JSON freeze returns; disable thinking when the server accepts it.
    try:
        if chat:
            reply = chat(rounds)
        else:
            # Repo inspection is already in the user payload. A tool-calling
            # loop against the local intake model can hang the Start request
            # past gunicorn; keep the completion tool-free so the interview
            # returns. _clarify_chat stays for callers that opt in.
            reply = _chat(rounds, timeout=180.0, max_tokens=2048)
        parsed = _parse_json_object(reply)
        # Small/local models often keep asking a differently worded version of
        # the same question after the minimum interview is complete. Give the
        # same model one explicit synthesis turn over the full transcript. It
        # may still refuse with an invalid contract; contract_defects remains
        # the fail-closed authority below.
        if answered_rounds(rows) >= MAX_CLARIFY_ROUNDS and not bool(parsed.get("ready")):
            freeze_rounds = rounds + [
                {
                    "role": "user",
                    "content": (
                        "INTERVIEW COMPLETE. Freeze the contract now. The latest "
                        "answer wins conflicts with earlier answers, never with a "
                        "surveyed code safety constraint. If such a conflict "
                        "remains, keep ready=false and ask exactly one question "
                        "that names its failure mode and safe recommendation. "
                        "Otherwise return ready=true, no questions, "
                        "and complete WHEN/SHALL requirements plus matching "
                        "given/when/then/fail-if evidence. Include the causal code "
                        "path, compatibility blast boundary, rollback strategy, "
                        "Docker-only synthetic proof, no production, and no merge."
                    ),
                }
            ]
            freeze_reply = chat(freeze_rounds) if chat else _chat(freeze_rounds, timeout=180.0, max_tokens=4096)
            frozen = _parse_json_object(freeze_reply)
            if isinstance(frozen, dict):
                parsed = frozen
    except SdlcError:
        # A bad completion must still open the interview — gap / reproduce
        # questions below fill what the model failed to freeze.
        parsed = {}
    except Exception:  # noqa: BLE001
        parsed = {}
    questions = _questions(parsed.get("questions"))
    requirements = _strings(parsed.get("requirements") or parsed.get("conditions"))
    criteria = _strings(parsed.get("criteria") or parsed.get("evidence") or parsed.get("clearing_evidence"))
    defects = contract_defects(requirements, criteria)
    ready = bool(parsed.get("ready")) and not defects
    if str(kind or "").strip() == "fix" and not _has_reproduction(
        rows,
        requirements,
        criteria,
    ):
        ready = False
        if not questions:
            questions = _fix_reproduce_questions(headline, detail)
    if ready:
        questions = []
    elif not questions:
        questions = _gap_questions(
            defects or ["no condition for plan"],
            kind=kind,
            headline=headline,
            detail=detail,
        )
    return {
        "ready": ready,
        "questions": questions,
        "requirements": requirements,
        "criteria": criteria,
        "summary": str(parsed.get("summary") or "").strip(),
    }


# --------------------------------------------------------------------------
# Issue rendering
# --------------------------------------------------------------------------


def _agent_brings_its_own_model(agent: str) -> bool:
    """True for an agent whose engine supplies its own model out of band.

    claude / codex / cursor / opencode run their own CLI and take no `/model`
    alias from our catalogue. Handing one a local alias creates a run that
    cannot dispatch at all.
    """
    name = str(agent or "").strip()
    if not name:
        return False
    try:
        from sharedagents_core.layout import agents_root

        raw = (Path(agents_root()) / name / "config.json").read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return False
        engine = str(parsed.get("engine") or "").strip().lower()
    except (OSError, ValueError, ImportError):
        return False
    return bool(engine) and engine != "local_llm"


def default_model(agent: str = "") -> str:
    """The alias a new run uses when the operator does not pick one.

    The local vLLM router, not whatever the global chat config points at. A Dev
    run is long and multi-stage, so the default should be the model that is free
    and always reachable — the alternative is a pipeline that silently bills a
    cloud provider, and stops dead when that provider runs out of credit, which
    is exactly what happened to the first two live runs.

    ``SHAREDAGENTS_SDLC_DEFAULT_MODEL`` overrides. An operator who picks a model in
    the tab always wins over both.
    """
    if _agent_brings_its_own_model(agent):
        # Its engine chooses; forcing an alias would fail at dispatch.
        return ""
    declared = (os.environ.get("SHAREDAGENTS_SDLC_DEFAULT_MODEL") or "").strip()
    if declared:
        return declared
    rows = [row for row in (_config().get("models") or []) if isinstance(row, dict)]
    local = [row for row in rows if str(row.get("backend") or "").lower() == "vllm"]
    # Prefer the local model the config already calls primary, else the first.
    for row in local:
        if str(row.get("role") or "") == "primary":
            return str(row.get("alias") or row.get("id") or "")
    if local:
        return str(local[0].get("alias") or local[0].get("id") or "")
    return ""


def default_role_agents(_generator: str = "jarvis") -> dict[str, str]:
    """Three distinct executors for a new or legacy run.

    Orchestrate is always Jarvis. Verify A defaults to ``build``; verify B
    defaults to an external engine. Neither reviewer may be Jarvis.
    """
    generation = GENERATION_AGENT

    def _pick(order: tuple[str, ...], taken: set[str]) -> str:
        eligible = [name for name in (*order, *IN_PROCESS_REVIEWERS) if name not in taken and name not in NOT_REVIEWERS]
        for name in eligible:
            if reviewer_dispatchable(name):
                return name
        # Nothing verifiably runnable: keep the declared order rather than
        # returning nothing, so the tab still offers a choice to override.
        return eligible[0] if eligible else ""

    validation = _pick(VERIFY_A_DEFAULTS, {generation})
    verification = _pick(VERIFY_B_DEFAULTS, {generation, validation})
    return {
        "validation": validation,
        "generation": generation,
        "verification": verification,
    }


def role_config(run: dict[str, Any], role: str) -> tuple[str, str]:
    """Return the persisted ``(agent, model)`` for one pipeline role.

    Rows written before role-scoped columns existed — or that stored Jarvis
    as a reviewer — are upgraded logically on read. Generate stays Jarvis.
    """
    if role not in ROLE_NAMES:
        raise SdlcError(f"unknown SDLC role: {role}")
    defaults = default_role_agents(str(run.get("agent") or GENERATION_AGENT))
    agent = str(run.get(f"{role}_agent") or "").strip()
    if role == "generation":
        agent = GENERATION_AGENT
    elif not agent or agent in NOT_REVIEWERS:
        agent = defaults[role]
    model = str(run.get(f"{role}_model") or "").strip()
    if not model and role == "generation":
        model = str(run.get("model") or "").strip()
    gen_model = str(run.get("generation_model") or run.get("model") or "").strip()
    if role != "generation" and model and gen_model and model == gen_model:
        model = ""
    return agent, model


def validate_role_agents(config: dict[str, str]) -> None:
    """Reject a run that only labels the same executor three different ways.

    Independent agents (and named models) are what makes automatic merge
    safe: the same model cannot both write the code and sign it off.
    Verify A and verify B cannot be Jarvis.
    """
    agents = [str(config.get(f"{role}_agent") or "").strip() for role in ROLE_NAMES]
    if any(not agent for agent in agents):
        raise SdlcError("validation, generation, and verification agents are required")
    if str(config.get("generation_agent") or "").strip() != GENERATION_AGENT:
        raise SdlcError("orchestrate is always jarvis")
    for role in ("validation", "verification"):
        name = str(config.get(f"{role}_agent") or "").strip()
        if name in NOT_REVIEWERS:
            raise SdlcError(f"{ROLE_LABELS[role]} must be build or an external agent, not {name}")
    if len(set(agents)) != len(agents):
        raise SdlcError("validation, generation, and verification must use three different agents")
    models = [str(config.get(f"{role}_model") or "").strip() for role in ROLE_NAMES]
    named = [model for model in models if model]
    if len(named) != len(set(named)):
        raise SdlcError("named validation, generation, and verification models must be distinct")


def issue_body(
    *,
    headline: str,
    kind: str,
    requirements: list[str],
    criteria: list[str],
    transcript: list[dict[str, str]],
    run_id: str,
    branch: str,
) -> str:
    """The frozen contract, as it lands in the forge issue."""
    lines = [f"## Object\n{headline.strip()}\n", "## Condition (for plan)"]
    lines += [f"{i}. {text}" for i, text in enumerate(requirements, 1)] or ["_none stated_"]
    lines += ["", "## Clearing evidence (for test)"]
    lines += [f"- [ ] {text}" for text in criteria] or ["- [ ] _none stated_"]
    if transcript:
        lines += ["", "## Intake transcript", "<details><summary>Clarifying questions</summary>", ""]
        for entry in transcript:
            question = str(entry.get("question") or "").strip()
            answer = str(entry.get("answer") or "").strip()
            if question:
                lines.append(f"**Q:** {question}")
            if answer:
                lines.append(f"**A:** {answer}")
            lines.append("")
        lines += ["</details>"]
    lines += [
        "",
        "## Tracker",
        f"SDLC_RUN: `{run_id}`",
        f"Kind: `{kind}`",
        f"Branch: `{branch}`",
        "",
        "_Stages are posted below as they finish: " + ", ".join(f"`{STAGE_TAGS[s]}`" for s in STAGES) + "._",
    ]
    return "\n".join(lines)


def stage_comment(
    *,
    stage: str,
    status: str,
    note: str = "",
    evidence: str = "",
    lap: int = 0,
) -> str:
    """The issue comment posted when a stage finishes."""
    tag = STAGE_TAGS.get(stage, f"sdlc/{stage}")
    icon = {"passed": "PASS", "failed": "FAIL", "started": "RUN"}.get(status, status.upper())
    head = f"**`{tag}`** — {icon}" + (f" (lap {lap})" if lap else "")
    parts = [head]
    if note:
        parts += ["", note.strip()]
    if evidence:
        parts += ["", "<details><summary>Evidence</summary>", "", "```", evidence.strip()[:8000], "```", "</details>"]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Stage prompts
# --------------------------------------------------------------------------

_MARKER_RULE = """\
END OF TURN CONTRACT — the driver reads this line and nothing else:
Print, as the LAST line of your final message, exactly one of:
  SDLC-STAGE-DONE: {stage} STATUS=PASS NOTE=<one line of what you proved>
  SDLC-STAGE-DONE: {stage} STATUS=FAIL NOTE=<one line of what blocked you>
No marker means the stage FAILED. Do not print the marker before the work is done."""


def build_attempt_digest(
    *,
    attempt: int,
    note: str,
    session_activity: dict[str, Any] | None = None,
    artifacts: Iterable[str] | None = None,
    max_bytes: int = ATTEMPT_DIGEST_MAX_BYTES,
) -> dict[str, Any]:
    """P3: the compact handoff a failed attempt leaves for its successor.

    A no-progress retry used to be a cold restart: the next worker received
    the same prompt and a one-line note, so it repeated the same exploration
    and died the same way. This carries what it learned — the last tool calls
    by name with a one-line argument summary, the files it read, the diff it
    produced, and the watchdog note — never the transcript, never file
    contents. Every text field is truncated at write time and the whole
    payload is bounded: one verbose attempt must not tax every later prompt.
    """
    activity = session_activity if isinstance(session_activity, dict) else {}
    calls: list[dict[str, str]] = []
    for row in (activity.get("last_tool_calls") or [])[:12]:
        if not isinstance(row, dict):
            continue
        calls.append(
            {
                "name": str(row.get("name") or "")[:60],
                "args": " ".join(str(row.get("args") or "").split())[:120],
            }
        )
    files: list[str] = []
    for path in activity.get("files_read") or []:
        text = str(path or "").strip()
        if text and text not in files:
            files.append(text[:200])
        if len(files) >= 12:
            break
    paths: list[str] = []
    for path in artifacts or []:
        text = str(path or "").strip()
        if text and text not in paths:
            paths.append(text[:200])
        if len(paths) >= 12:
            break
    digest: dict[str, Any] = {
        "attempt": max(1, int(attempt or 1)),
        "note": " ".join(str(note or "").split())[:300],
        "last_tool_calls": calls,
        "files_read": files,
        "artifacts": paths,
    }
    encoded = lambda d: len(  # noqa: E731 — small, local, and obvious
        json.dumps(d, ensure_ascii=False).encode()
    )
    while encoded(digest) > max_bytes:
        if len(digest["last_tool_calls"]) > 1:
            digest["last_tool_calls"].pop()  # oldest call (list is newest-first)
        elif digest["files_read"]:
            digest["files_read"].pop()
        elif digest["artifacts"]:
            digest["artifacts"].pop()
        elif len(digest["note"]) > 64:
            digest["note"] = digest["note"][:64]
        else:
            digest["note"] = "previous attempt made no progress"
            if encoded(digest) <= max_bytes:
                break
            digest["last_tool_calls"] = []
            digest["files_read"] = []
            digest["artifacts"] = []
            break
    return digest


def stage_prompt(
    *,
    stage: str,
    headline: str,
    kind: str,
    repo: str,
    branch: str,
    requirements: list[str],
    criteria: list[str],
    issue_number: Any = "",
    issue_url: str = "",
    run_id: str = "",
    lap: int = 0,
    feedback: str = "",
    worktree: str = "",
    attempt: int = 1,
    previous: str = "",
    validation_guidance: list[str] | None = None,
    detail: str = "",
    transcript: Iterable[dict[str, Any]] = (),
    head_sha: str = "",
    fuse_plan_design: bool = True,
    first_artifact_budget_s: float = 0.0,
    stage_ceiling_s: float = 0.0,
    previous_attempt: dict[str, Any] | None = None,
) -> str:
    """Stable phase rules followed by the minimum lossless run context.

    Only PLAN consumes intake. Its committed brief carries the decisions to
    later fresh workers. Frozen conditions/criteria are never clipped or
    renumbered; retries carry current feedback instead of old conversations.
    """
    from sharedagents_web.dev_phase_prompts import COMMON_RULES, ISOLATION_RULES, PHASE_RULES

    stage = canonical_stage(stage)
    if stage not in PHASE_RULES:
        raise SdlcError(f"unknown SDLC phase: {stage}")
    # An explicit/legacy design dispatch is the split flow even when an old
    # caller has no stored preference. Its prompt must not say design is skipped.
    fuse_plan_design = fuse_plan_design and stage != "design"
    paths = stage_artifact_paths(
        stage,
        branch,
        fuse_plan_design=fuse_plan_design,
    )
    context: dict[str, Any] = {
        "run_id": run_id,
        "repo": repo,
        "kind": kind,
        "object": headline,
        "issue": {"number": issue_number, "url": issue_url},
        "branch": branch,
        "worktree": worktree,
        "head_sha": head_sha,
        "phase": stage,
        "flow": list(FUSED_STAGES if fuse_plan_design else STAGES),
        "artifact_paths": paths,
        "conditions": {f"R{i}": text for i, text in enumerate(requirements, 1)},
        "criteria": {f"C{i}": text for i, text in enumerate(criteria, 1)},
    }
    if stage == "plan":
        # Do not clip or summarize operator decisions: the brief is the one
        # place they are reconciled with the frozen contract. Unanswered cards
        # and arbitrary transcript metadata are not inputs to implementation.
        decisions = [
            {
                "question": str(row.get("question") or ""),
                "answer": str(row["answer"]),
                "source": str(row.get("source") or "operator"),
            }
            for row in transcript
            if isinstance(row, dict) and str(row.get("answer") or "").strip()
        ]
        if detail or decisions:
            context["intake"] = {"request": detail, "settled_answers": decisions}
    if attempt > 1 or feedback or previous:
        context["rework"] = {
            "attempt": attempt,
            "lap": lap,
            "feedback": feedback.strip() or previous.strip() or "Previous attempt did not finish.",
        }
    if previous_attempt:
        # P3: a warm handoff. A retry that only repeats the prompt repeats the
        # failure; the digest tells the next worker what was already tried.
        rework = context.setdefault("rework", {"attempt": attempt, "lap": lap})
        rework["previous_attempt"] = previous_attempt
    if first_artifact_budget_s:
        # P1: the worker must know the clock it is actually on. The rules
        # reference these keys; the numbers live here so the phase text stays
        # byte-stable across runs (only RUN CONTEXT changes).
        context["first_artifact_budget_s"] = round(float(first_artifact_budget_s), 3)
    if stage_ceiling_s:
        context["stage_ceiling_s"] = round(float(stage_ceiling_s), 3)
    if validation_guidance:
        context["validation_guidance"] = validation_guidance
    rules = [f"SDLC PHASE CONTRACT — {stage.upper()}", COMMON_RULES, PHASE_RULES[stage]]
    if stage == "plan":
        rules.append(
            "Plan and design are fused into this brief; no separate design turn."
            if fuse_plan_design
            else "Plan the approach now; a separate DESIGN phase owns the technical design."
        )
    if stage in {"code", "tests", "deploy"}:
        rules.append(ISOLATION_RULES)
    rules.append(_MARKER_RULE.format(stage=stage))
    return (
        "\n\n".join(rules)
        + "\n\nRUN CONTEXT (JSON data):\n"
        + json.dumps(
            context,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def stage_goal_prompt(
    *,
    stage: str,
    headline: str,
    kind: str,
    repo: str,
    branch: str,
    requirements: list[str],
    criteria: list[str],
    issue_number: Any = "",
    issue_url: str = "",
    run_id: str = "",
    worktree: str = "",
    feedback: str = "",
) -> str:
    """One Jarvis-conductor ``/goal`` for a generate step. Clarify is never this.

    The You-bubble is this text. Jarvis delegates; it CLEARS the OPEN gate
    only when artifacts back the goal. A bad definition rewinds to plan or
    the previous phase via ``sdlc submit … rewind=``.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", str(branch or "run").lower()).strip("-") or "run"
    reqs = "\n".join(f"{i}. {t}" for i, t in enumerate(requirements, 1)) or "(none)"
    # Numbered, because the tests stage must cite them by id (`C3`) and a
    # bare bullet gives the agent nothing to cite.
    crits = "\n".join(f"- C{index} {text}" for index, text in enumerate(criteria, 1)) or "(none)"
    issue_cite = f"#{issue_number}" + (f" ({issue_url})" if issue_url else "") if issue_number else "(unbound)"
    objectives = {
        "plan": (
            f"Commit plan/{slug}.md — fused brief: every frozen WHEN/SHALL "
            f"plus the file/function map for {headline!r}; honor every "
            f"NON-GOAL; write no product code. Design is skipped."
        ),
        "design": (
            f"Commit spec/{slug}.md naming the files and functions that "
            f"satisfy every frozen line; live-check current APIs; invent none."
        ),
        "code": (
            f"Implement every WHEN/SHALL in the run worktree only; LSP clean; "
            f"Refs #{issue_number or 'issue'}; no PR and no deploy."
        ),
        "tests": (
            "Drive each frozen evidence line on the named real surface and "
            "record observations. Then `sdlc submit` tests verdict=READY. "
            "You cannot CLEAR tests. Do not spawn verify A or verify B — "
            "the driver launches that pair once after READY."
        ),
        "deploy": (
            "Re-check every frozen evidence line is still green. Do not open "
            "a PR, do not merge, do not deploy. The driver opens and merges."
        ),
    }
    objective = objectives.get(stage, f"Clear the OPEN `{stage}` gate for {headline!r}.")
    isolation = ""
    if stage in {"code", "tests", "deploy"}:
        isolation = (
            "\n\nTEST ISOLATION (frozen): verification runs only in nested Docker. "
            "The branch MUST declare `.sharedagents/verify.json` with "
            "`isolation.type=docker`, `isolation.dockerfile=Dockerfile.test`, "
            "and checks whose commands run inside that image at `/workspace`. "
            "`Dockerfile.test` MUST COPY the repository to `/workspace`. Use "
            "synthetic fixtures only. Do not access production paths, services, "
            "credentials, host mounts, or network resources. The driver, not "
            "the child, builds and executes the proof image.\n"
        )
    rewind = (
        "If a DEFINITION is wrong (freeze, plan, or spec), do not patch "
        "around it. Call `sdlc submit` FAIL with `rewind=plan` (freeze/plan "
        "wrong) or `rewind=previous` (last phase only). Implementation-only "
        "defects rewind to the producer (usually code)."
    )
    extra = f"\n\nREWIND / REWORK:\n{feedback.strip()}\n" if feedback.strip() else ""
    return f"""/goal {objective}

You are the Jarvis conductor for SDLC_RUN `{run_id}` (kind {kind}).
This is the same session for the whole run. Do not start another Jarvis.

OPEN gate: `{stage}`
ISSUE: {issue_cite}
REPO: `{repo}`
BRANCH: `{branch}`
WORKTREE: `{worktree}` — children work ONLY there. Never edit local main.

CONDITION (frozen — do not re-validate whether these are the right requirements):
{reqs}

CLEARING EVIDENCE (frozen):
{crits}
{isolation}
{extra}
Delegate ONE child for this OPEN gate. You do not write files.
You HAVE the ``sdlc`` tool on this session — call it yourself.
Do not delegate submit. Do not hunt for a CLI or HTTP /submit.
Do not ask a child to paste full source, a full plan, or the board JSON.
Child report = path + SHA + checks + defects.

Every stage ARTIFACT is written by `build`. The `plan` profile is READ-ONLY —
it cannot create `plan/<slug>.md`, and asking it to will burn the stage on
`[permission denied] plan-mode` instead of producing anything. Use `plan` (or
`explore`) to READ and research; use `build` to write.

Typical handoff: fused brief → `build` (research first with `plan` if the
repo is unfamiliar, then you ``sdlc submit``);
implement → `build` (then you ``sdlc submit`` with the SHA);
drive a named surface → whoever can drive it. The driver launches
verify A and verify B after READY. Do not spawn them.

When — and only when — you can see the artifacts that satisfy this goal,
call `sdlc submit` with stage=`{stage}` verdict=PASS, the path, and the SHA.
Exception: stage `tests` — you cannot PASS/CLEAR. Submit verdict=READY
after the drive and wait. Tests CLEAR only when both reviewers agree PASS.
No artifact, no CLEAR. Do not infer a pass. Do not print a worker
SDLC-STAGE-DONE marker as a substitute.

{rewind}

Do not open a second issue. Do not open a PR. Do not merge.
Do not deploy. Deploy stays a CANON-A click.
"""


# --------------------------------------------------------------------------
# Phase bookends: every stage starts with validation and ends with verification
# --------------------------------------------------------------------------
#
# A stage used to be trusted on its word: it printed PASS and the pipeline moved
# on. That is the weakest link in the whole chain, because the thing most likely
# to be wrong about whether work happened is the agent that was supposed to do
# it. So each stage is now bookended:
#
#   VALIDATE (entry)  — are this stage's inputs actually present? Checked BEFORE
#                       dispatch, so a stage that cannot succeed costs no turn.
#   VERIFY  (exit)    — is the claim consistent with what is on the branch?
#                       A PASS that leaves no trace is downgraded to a failure.
#
# Both are deterministic and git-based. They prove STRUCTURE, not correctness:
# "a spec file was committed", not "the spec is good". Correctness is the
# reviewer's job. The point is that neither can be satisfied by saying so.


def commit_plan_if_uncommitted(run: dict[str, Any], *, runner: Any = None) -> str:
    """Controller commit for ``plan/*.md`` so plan-mode need not run git.

    Returns the new SHA or empty if there was nothing to commit.
    """
    root = Path(str(run.get("worktree") or ""))
    if not root.is_dir():
        return ""
    code, status = _git(["status", "--porcelain"], cwd=root, runner=runner)
    if code != 0 or not status.strip():
        return ""
    dirty = [line[3:].strip() for line in status.splitlines() if line[3:].strip()]
    if not dirty or any(not name.startswith("plan/") for name in dirty):
        return ""
    add_code, _ = _git(["add", "--", "plan"], cwd=root, runner=runner)
    if add_code != 0:
        return ""
    issue = run.get("issue_number") or "issue"
    msg = f"plan: fused brief Refs #{issue}"
    c_code, _ = _git(["commit", "-m", msg], cwd=root, runner=runner)
    if c_code != 0:
        return ""
    _, sha = _git(["rev-parse", "HEAD"], cwd=root, runner=runner)
    return sha.strip()


def _branch_commits(repo: str, branch: str, *, base: str = "", runner: Any = None) -> list[tuple[str, str]]:
    """``(sha, subject)`` for each commit this branch adds over *base*."""
    path = repo_dir(repo)
    worktree = existing_worktree(path, branch, runner=runner) or str(path)
    ref = base or default_base(repo, runner=runner)
    code, out = _git(
        ["log", "--no-merges", "--format=%h\x1f%s", f"{ref}..HEAD"],
        cwd=Path(worktree),
        runner=runner,
    )
    if code != 0:
        return []
    rows: list[tuple[str, str]] = []
    for line in out.splitlines():
        sha, _, subject = line.partition("\x1f")
        if sha.strip():
            rows.append((sha.strip(), subject.strip()))
    return rows


def _branch_files(
    repo: str, branch: str, *, base: str = "", runner: Any = None, committed_only: bool = True
) -> list[str]:
    """Paths this branch has touched relative to its base.

    ``committed_only`` is the default because the bookends claim the BRANCH
    corroborates a stage, and an uncommitted edit is not on the branch — it is
    one `git checkout` from never having existed. Entry validation may look at
    the working tree too, since there it is only deciding whether there is
    something to work with.
    """
    path = repo_dir(repo)
    worktree = existing_worktree(path, branch, runner=runner) or str(path)
    ref = base or default_base(repo, runner=runner)
    seen: list[str] = []
    sources = [["diff", "--name-only", f"{ref}...HEAD"]]
    if not committed_only:
        sources.append(["status", "--porcelain"])
    # Only the PRIMARY source decides `ok`. The diff against the base is what
    # says which files this branch changed; `status` is a supplement for
    # uncommitted work, and a failure there does not mean the listing is
    # unknown.
    ok = True
    for index, args in enumerate(sources):
        code, out = _git(args, cwd=Path(worktree), runner=runner)
        if code != 0:
            if index == 0:
                ok = False
            continue
        for line in out.splitlines():
            name = line[3:].strip() if args[0] == "status" else line.strip()
            if name and name not in seen:
                seen.append(name)
    listing = FileList(seen)
    listing.ok = ok
    return listing


#: Where a design document is recognised. The stage prompts write to the repo's
#: own `plan/` and `spec/` directories -- these are deliverables reviewed with
#: the code, not agent scratch, so they must not land in `.claude/`, which is
#: gitignored in most checkouts and is for throwaway output.
#: `.claude/plans/` stays recognised so runs created before this still verify.
#: The rest are conventions other repos use. This is a heuristic and is
#: documented as one.
#: Bounds on everything the semantic gates read. A gate that can stall for
#: minutes on a local model, or blow its context, is worse than no gate.
MAX_SPEC_FILE_BYTES = 256 * 1024
MAX_DIFF_FILES = 200
#: The buffered git path has always had this; the bounded one must not lose it.
GIT_TIMEOUT_S = 120
DIFF_CONTEXT_LINES = 2
#: Per-item cap for an operator-authored requirement or clearing-evidence line.
#:
#: This was 400, and 400 is not a criterion. The clarify interview itself
#: authors evidence as `given … / when … / then … / fail-if …`, which lands
#: between 200 and 450 characters -- so the pipeline routinely froze contracts
#: its OWN gate could not judge. Clipping any item marks the material partial,
#: `definition_partial` makes the `tests` gate GATE-UNUSABLE, and no retry can
#: ever clear it: a live run died on two criteria that were 22 and 25
#: characters too long, reported as "saw only 2787 of 2834 characters".
#:
#: The real pressure is the GATE budget (120k by default), not this: 40 items
#: at this cap is ~48k, well inside it. Override for a smaller context.
MAX_BULLET_CHARS = int(os.environ.get("SHAREDAGENTS_SDLC_BULLET_CHARS") or 1200)
#: How much of the definition and the evidence a gate may be shown.
#:
#: This was 24,000 characters, and that number was the root of both gate
#: failures seen in production: an implementation reported missing because the
#: evidence was cut away, and then an input REJECTED with "the visible design is
#: truncated" -- a complaint about the excerpt, not about the work.
#:
#: It was also arbitrary. Measured against the configured model: 200,000
#: characters is 81,880 prompt tokens and answers in 11 seconds; 120,000 is
#: 49,154 tokens in 8.2 seconds, no slower than 24,000 was. A real design
#: (47,085 characters on the run that hit this) plus its evidence fits inside
#: 120,000 with room to spare, so the gate simply sees the whole thing.
#:
#: Override for a model with a smaller context. A prompt that overruns one
#: shows up as an unusable gate -- recorded, blocking, and obvious -- never as
#: a wrong verdict.
MAX_GATE_PROMPT_CHARS = int(os.environ.get("SHAREDAGENTS_SDLC_GATE_PROMPT_CHARS") or 120000)

#: A memory ceiling on material collection, NOT a review ceiling. Collection
#: used to clip documents at 16,000 characters and evidence at 24,000, which
#: meant a 47KB design was already gone before the gate budget was consulted --
#: so `evidence_cut` said False and the gate rejected an incomplete view it did
#: not know was incomplete. Collection now returns everything and the GATE
#: decides what fits, because only the gate can page through the rest.
MAX_MATERIAL_CHARS = MAX_GATE_PROMPT_CHARS * 8


class FileList(list):
    """Branch filenames, carrying whether git actually answered.

    An empty list from a FAILED git command is indistinguishable from a branch
    with no files, so a collection failure became complete-looking empty
    evidence -- the original bug's exact shape.
    """

    ok: bool = True


class Material(str):
    """Text a gate will judge, carrying whether it is the WHOLE of its source.

    A plain `str` cannot say "there was more of me". Clipping upstream and
    returning a bare string is how a partial view came to be judged as if it
    were complete. This behaves as a string everywhere it already did.
    """

    total: int
    shown: int

    def __new__(cls, text: str, total: int | None = None) -> Material:
        self = super().__new__(cls, text)
        self.shown = len(text)
        self.total = int(total if total is not None else len(text))
        return self

    @property
    def partial(self) -> bool:
        """Was source discarded that this gate can never page back in?"""
        return self.shown < self.total


def _compose(*parts: Any) -> Material:
    """Join material, keeping "something was discarded" true if any part was.

    Composing Materials into a bare string threw the provenance away, which is
    how a clipped design reached a gate looking whole.
    """
    text = "".join(str(part) for part in parts)
    missing = sum(max(0, getattr(part, "total", 0) - getattr(part, "shown", 0)) for part in parts)
    return Material(text, total=len(text) + missing)


def _or_else(value: Any, fallback: str) -> Material:
    """`value` if it has content, else `fallback` -- KEEPING the provenance.

    `Material("", total=1)` is falsey, so a plain `or "(empty)"` swapped a
    known-incomplete collection for a complete-looking placeholder and the gate
    judged it as whole. "There was more of me" has to survive the
    fallback, because that is exactly the case the fallback fires on.
    """
    if str(value):
        return value if isinstance(value, Material) else Material(str(value))
    missing = max(0, int(getattr(value, "total", 0)) - int(getattr(value, "shown", 0)))
    return Material(fallback, total=len(fallback) + missing)


#: The evidence is the SUBJECT of a gate's judgement; the definition is only
#: context. When both cannot fit, the definition yields — a gate that cannot see
#: the work cannot judge it, and silently losing the diff is how a stage that
#: genuinely delivered gets reported as unimplemented.
GATE_EVIDENCE_SHARE = 0.6


class GateFit(NamedTuple):
    """What actually reached the model, and how much of the source it was.

    ``evidence_shown`` counts RAW evidence characters included -- NOT
    ``len(evidence)``, which also carries the truncation note. Inferring the
    offset from the fitted string silently skips the difference, so a re-check
    would never look at those characters.
    """

    definition: str
    evidence: str
    evidence_cut: bool
    evidence_shown: int
    evidence_budget: int
    #: The DEFINITION side can be clipped while the evidence fits, and a gate
    #: judging against half a design can pass work that never met the other
    #: half. Tracking only the evidence left that hole open.
    definition_cut: bool = False


def _fit_gate_prompt(definition: str, evidence: str, cap: int = MAX_GATE_PROMPT_CHARS, overhead: int = 1200) -> GateFit:
    """Share the prompt budget so the evidence is never crowded out.

    A gate is asked whether something was implemented. If the context is laid
    down first and the whole prompt is then chopped to fit, the answer is
    decided by the chop: the evidence is what falls off the end. That is not a
    hypothetical -- it is how a real implementation came to be reported missing.
    """
    budget = max(0, int(cap) - int(overhead))
    definition, evidence = str(definition or ""), str(evidence or "")
    if len(definition) + len(evidence) <= budget:
        return GateFit(definition, evidence, False, len(evidence), max(budget - len(definition), 1), False)
    ev_budget = int(budget * GATE_EVIDENCE_SHARE)
    def_budget = budget - ev_budget
    if len(evidence) < ev_budget:
        def_budget += ev_budget - len(evidence)
        ev_budget = len(evidence)
    elif len(definition) < def_budget:
        ev_budget += def_budget - len(definition)
        def_budget = len(definition)

    shown: dict[str, int] = {}

    def _clip(text: str, limit: int, what: str) -> str:
        if len(text) <= limit:
            shown[what] = len(text)
            return text
        # The note lives inside the budget, so "showing N" must describe the
        # excerpt, not the limit -- otherwise it overstates by its own length.
        note = (
            f"\n\n[... {what} truncated: showing {{n}} of {len(text)} "
            f"characters. Absence from this excerpt is NOT evidence of "
            f"absence from the branch. ...]"
        )
        keep = max(0, limit - len(note.format(n=limit)))
        shown[what] = keep
        return text[:keep] + note.format(n=keep)

    fitted_def = _clip(definition, def_budget, "definition")
    fitted_ev = _clip(evidence, ev_budget, "evidence")
    return GateFit(
        fitted_def,
        fitted_ev,
        shown["evidence"] < len(evidence),
        shown["evidence"],
        max(ev_budget, 1),
        shown["definition"] < len(definition),
    )


MAX_MODEL_RESPONSE_BYTES = 2 * 1024 * 1024

SPEC_MARKERS: tuple[str, ...] = (
    "plan/",
    "spec/",
    "requirements/",
    ".claude/plans/",
    "/plans/",
    "docs/",
    "design",
    "spec",
    "rfc",
    "adr",
)


def _looks_like_spec(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in SPEC_MARKERS)


def _doc_kind(name: str) -> str:
    """``"plan"``, ``"design"`` or ``""`` for a design document.

    Plan and Design are separate stages committing separate documents, so "a
    spec-shaped file exists" is not enough: the plan's own file would otherwise
    satisfy design's entry check, and then code's. They now live in different
    directories, so the directory answers it; the filename is only a fallback
    for runs made before the move.
    """
    lowered = name.lower().lstrip("./")
    # The DIRECTORY is the kind now: the prompts write `plan/<slug>.md` and
    # `spec/<slug>.md`, so the containing folder says what a document is
    # without depending on anyone naming the file well. This also ends the
    # ambiguity a branch slug used to create -- a plan on a branch called
    # `add-design-page` read as a design.
    if lowered.startswith("plan/") or "/plan/" in lowered:
        return "plan"
    if lowered.startswith(("spec/", "requirements/")) or "/spec/" in lowered:
        return "design"

    # Fall back to the filename for pre-move runs and for repos that keep
    # design notes elsewhere.
    base = lowered.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    for kind in ("plan", "design"):
        if stem.endswith(f"-{kind}") or stem == kind:
            return kind
    if "design" in stem or "spec" in stem:
        return "design"
    if "plan" in stem:
        return "plan"
    return ""


def _has_doc(files: list[str], kind: str) -> bool:
    """Is a document for *kind* committed?

    An untagged document satisfies any kind ONLY while nothing is tagged -- a
    repo that does not follow the naming should not be failed. Once some
    document names its kind, the naming is being used, so an exact match is
    required: otherwise the plan's own file silently satisfies design's entry
    check, and then code's.
    """
    docs = [n for n in files if _looks_like_spec(n)]
    kinds = [_doc_kind(n) for n in docs]
    if any(k == kind for k in kinds):
        return True
    return bool(docs) and not any(kinds)


#: Filename shapes that mean "this is a test", across the ecosystems a repo
#: driven by this tab might plausibly use. Narrow matching failed correct runs
#: on `test/foo.go`, `foo_test.go` and `.spec.ts`.
_TEST_DIRS = ("tests/", "test/", "spec/", "specs/", "__tests__/", "e2e/", "it/")
_TEST_SUFFIXES = (
    "_test.py",
    "_test.go",
    "_test.rb",
    "_test.exs",
    "_test.rs",
    ".test.js",
    ".test.ts",
    ".test.jsx",
    ".test.tsx",
    ".spec.js",
    ".spec.ts",
    ".spec.jsx",
    ".spec.tsx",
    ".spec.rb",
    "test.java",
    "tests.py",
    "_spec.rb",
)


def _looks_like_test(name: str) -> bool:
    lowered = name.lower()
    leaf = lowered.rsplit("/", 1)[-1]
    stem = leaf.rsplit(".", 1)[0] if "." in leaf else leaf
    return (
        any(lowered.startswith(d) or f"/{d}" in lowered for d in _TEST_DIRS)
        or leaf.startswith(("test_", "test-", "spec_"))
        or leaf.endswith(_TEST_SUFFIXES)
        # A single-file suite at the root -- `tests.js`, `test.py`, `spec.ts` --
        # matched none of the above, so it counted as IMPLEMENTATION. The
        # self-passing probe then emptied the test file itself, every check
        # trivially passed, and a perfectly good suite was reported as
        # self-passing. It also meant `_test_surface` never saw it.
        or stem in ("test", "tests", "spec", "specs")
    )


#: Build metadata, not behaviour. Emptying one of these does not remove an
#: implementation -- it removes the ability to RUN anything, so the whole suite
#: dies for a reason that says nothing about the code. Live: `package.json` was
#: gutted alongside the implementation, `npm test` could no longer start, and
#: every criterion came back unattributable.
_MANIFEST_FILES: frozenset[str] = frozenset(
    {
        "package.json",
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "tsconfig.json",
        "jsconfig.json",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "tox.ini",
        "go.mod",
        "go.sum",
        "Cargo.toml",
        "Cargo.lock",
        "Makefile",
        "Dockerfile",
    }
)


def _is_manifest(name: str) -> bool:
    return str(name).rsplit("/", 1)[-1] in _MANIFEST_FILES


#: Suffixes whose behaviour the mutant can actually remove. Anything else --
#: `.md`, `.editorconfig`, a YAML workflow -- carries no behaviour a test can
#: depend on, and "gutting" it only breaks the tree for unrelated reasons.
#: Live on flask: with no branch diff to work from, the fallback walked the
#: repo and nominated 83 files including `.github/SECURITY.md`, so shape
#: preservation failed and every criterion came back UNPROVEN.
_MUTABLE_SUFFIXES: frozenset[str] = frozenset({".py", ".js", ".cjs"})


def mutable_files(files: list[str]) -> list[str]:
    """Implementation files whose behaviour can actually be removed."""
    return [name for name in _implementation_files(files) if "." + str(name).rsplit(".", 1)[-1] in _MUTABLE_SUFFIXES]


def _implementation_files(files: list[str]) -> list[str]:
    """The files that are neither the spec nor a test — i.e. the actual change."""
    return [
        name for name in files if not _looks_like_spec(name) and not _looks_like_test(name) and not _is_manifest(name)
    ]


def _run_cmd(argv: list[str], *, cwd: Path | None = None, timeout: float = 30.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        text = (proc.stderr or proc.stdout or "").strip()
        return proc.returncode, text
    except FileNotFoundError:
        return 127, f"{argv[0]} not found"
    except Exception as exc:  # noqa: BLE001
        return 1, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Functional verification: the harness RUNS the work, and that run is evidence
# --------------------------------------------------------------------------
#
# Before this, the `tests` gate read a diff of test files and the agent's own
# note about them. Nothing executed anything, so "the tests pass" was a claim
# with the same standing as any other sentence the model wrote -- and a test
# that asserts nothing reads exactly like one that asserts everything. The
# reviewers noticed, and kept asking for "driven observation" the pipeline had
# no way to supply.
#
# So the harness runs the checks itself, in the worktree, and the transcript it
# captures IS the evidence: real commands, real exit codes, real output. A
# model cannot fabricate an exit code it did not produce.

#: Wall-clock ceiling for one check. A hung suite must not hold a stage open.
VERIFY_CHECK_TIMEOUT = float(os.environ.get("SHAREDAGENTS_SDLC_VERIFY_TIMEOUT") or 600)

#: How long BOTH independent reviewers together get to answer. A reviewer is a
#: model turn over the network with no timeout of its own, and the wait for it
#: happens INSIDE the stage polling loop -- so a reviewer that never answers
#: does not merely fail the stage, it stops the loop from ever reaching its own
#: 90-minute deadline. Observed live: verify A and verify B were created at
#: 20:17:30, never reached the gateway at all, and the run sat in `tests`
#: indefinitely with the driver thread parked on `future.result()`.
REVIEW_ANSWER_TIMEOUT = float(os.environ.get("SHAREDAGENTS_SDLC_REVIEW_TIMEOUT") or 1200)
#: How much of one check's output is kept as evidence.
VERIFY_OUTPUT_CHARS = 4000
#: Criterion citation, e.g. a test named "[C3] divide by zero shows Error".
#: The contract's clearing evidence is numbered from 1 in the order it was
#: frozen, so `C3` is the third line and nothing has to be kept in sync.
#: `[C1] adds` (TAP titles) AND `test_C1_adds` / `::test_C1_adds` (python
#: identifiers, where `\b` fails because `_` is a word character). Rejects
#: `C3PO`: a letter may not follow the number.
CRITERION_CITE = re.compile(r"(?<![A-Za-z0-9])C(\d{1,3})(?![A-Za-z0-9])")
#: `ok 3 - name` / `not ok 3 - name`, which node:test and pytest-tap both emit.
_TAP_LINE = re.compile(r"^\s*(not\s+)?ok\s+\d+\s*[-–]?\s*(.+?)\s*$", re.MULTILINE)


def criterion_ids(criteria: list[str] | None) -> list[str]:
    """`C1..Cn` for the frozen clearing evidence, by position."""
    return [f"C{index}" for index, _ in enumerate(list(criteria or []), 1)]


def _parse_test_results(output: str) -> dict[str, bool]:
    """`{test name: passed}` from TAP output. Empty when the runner emits none.

    Subtests repeat under their parent in node:test output, so the STRICTER
    result wins: a name that failed anywhere is recorded as failed.
    """
    return {name: all(runs) for name, runs in _parse_test_occurrences(output).items()}


_PYTEST_LINE = re.compile(
    r"^(?P<id>[\w./\\-]+\.py::[^\s]+)\s+(?P<verdict>PASSED|FAILED|ERROR|XFAIL|XPASS)", re.MULTILINE
)
_PYTEST_SUMMARY = re.compile(r"^(?P<verdict>PASSED|FAILED|ERROR)\s+(?P<id>[\w./\\-]+\.py::[^\s]+)", re.MULTILINE)


def _parse_pytest_occurrences(output: str) -> dict[str, list[bool]]:
    """`{node id: [passed, ...]}` from pytest's own reporting.

    pytest does not speak TAP, so a Python repo had NO per-test results at all
    and every criterion was unattributable before this. Both the verbose line
    (`path::test PASSED`) and the summary line (`FAILED path::test`) are read.
    """
    runs: dict[str, list[bool]] = {}
    for pattern in (_PYTEST_LINE, _PYTEST_SUMMARY):
        for match in pattern.finditer(str(output or "")):
            node = match.group("id").strip()
            verdict = match.group("verdict")
            runs.setdefault(node, []).append(verdict in ("PASSED", "XFAIL"))
    return runs


def _parse_test_occurrences(output: str) -> dict[str, list[bool]]:
    """`{test name: [passed, ...]}` — EVERY line, not one verdict per name.

    Collapsing to a single bool loses a test. Two same-named `[C1]` tests both
    passing on the branch, then one passing and one gone under mutation, read
    as "still passed" and the criterion is called uncovered even though a test
    for it demonstrably broke.
    """
    runs: dict[str, list[bool]] = {}
    for failed, name in _TAP_LINE.findall(str(output or "")):
        label = name.strip()
        if not label or label.startswith("#"):
            continue
        runs.setdefault(label, []).append(not failed)
    for node, verdicts in _parse_pytest_occurrences(output).items():
        runs.setdefault(node, []).extend(verdicts)
    return runs


def _criterion_coverage(
    criteria: list[str] | None,
    baseline: dict[str, bool],
    mutated: dict[str, bool],
    *,
    probe_ran: bool | None = None,
    baseline_runs: dict[str, list[bool]] | None = None,
    mutated_runs: dict[str, list[bool]] | None = None,
    counts_reliable: bool = True,
    citations_reliable: bool = True,
    isolated: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Which frozen criteria are cleared by a test that DEPENDS on the code.

    Running green proves the suite works. It does not prove the suite is about
    the contract: a real, non-self-passing suite can test something else
    entirely and this gate used to pass it. So each criterion is cleared only
    when a test citing it (``[C3] …``) passed on the branch AND demonstrably
    stopped passing once the implementation was emptied.

    Three states, because "did not pass under mutation" and "was not seen under
    mutation" are not the same claim:

    * COVERED -- a citing test is PRESENT AND FAILING in the mutated run. That
      is the only signal that names one test: the runner reached it, ran it,
      and it reported that it no longer holds.
    * UNCOVERED -- nothing cites the criterion, or every citing test passed
      again with the implementation emptied. The producer's to fix; blocks.
    * UNPROVEN -- the mutated run is silent about it, in whole or in part.
      Tempting to read as "the suite died before reaching it", but a fail-fast
      runner, a truncated tail, a parallel runner, or an UNCITED check crashing
      first all produce the same silence, and each would credit a tautology.
      Reported, never credited, and never blamed on the producer.
    """
    wanted = criterion_ids(criteria)
    if not wanted:
        return {
            "attempted": False,
            "why": "the contract froze no criteria",
            "wanted": [],
            "covered": [],
            "uncovered": [],
            "unproven": [],
        }
    # A probe that RAN and parsed no test names is evidence (the suite died
    # outright); a probe that never ran is not. Both arrive here as an empty
    # `mutated`, so the caller has to say which -- inferring it from emptiness
    # lets a skipped probe pass a criterion on the baseline alone.
    ran = bool(mutated) if probe_ran is None else bool(probe_ran)

    def _verdict(name: str) -> str:
        """``"broke"`` | ``"survived"`` | ``"unknown"`` for one citing test."""
        # An ISOLATED experiment outranks every whole-suite reading: it ran
        # this test and nothing else, against a control that proved the filter
        # selects it. None of the confounders below apply to it.
        decided = (isolated or {}).get(name)
        if decided in ("broke", "survived"):
            return decided
        if mutated.get(name) is True:
            if not counts_reliable:
                # Only the TAIL was kept. "Passing" here is what SURVIVED the
                # truncation, not what the run reported: a failing occurrence
                # of the same name may have scrolled off. Blaming the producer
                # for that is a gate they cannot pass by writing better tests.
                return "unknown"
            if (
                baseline_runs is not None
                and mutated_runs is not None
                and len(mutated_runs.get(name) or []) < len(baseline_runs.get(name) or [])
            ):
                # One occurrence of this name stopped happening. Tempting to
                # call that broken, but it is the absence problem again: a
                # fail-fast runner cut off by an UNCITED check ends the run
                # early, and the occurrence that vanished may be a tautology
                # that simply never ran. Unattributable in BOTH directions --
                # so neither credited nor blamed.
                return "unknown"
            return "survived"
        if name in mutated:
            return "broke"  # present and failing: it depended on the code
        # ABSENT from a run that DID report other tests. Tempting to read as
        # "the suite died before reaching it", but the silence is not
        # attributable: a fail-fast runner stops at the first failure, the
        # output keeps only its tail, and a parallel runner does not even
        # report in a fixed order. Position cannot settle it either -- several
        # check commands are concatenated, so one command's later test can sit
        # "after" another command's failure. Anything credited here would be
        # credited for a test that never ran.
        # Even "nothing reported at all" does not attribute: an UNCITED check
        # crashing first kills the run before the citing test would have
        # printed anything, and the tautology gets the credit. There is no
        # reading of silence that is sound, so silence proves nothing.
        return "unknown"

    cited: dict[str, list[str]] = {cid: [] for cid in wanted}
    for name in baseline:
        for number in CRITERION_CITE.findall(name):
            cid = f"C{int(number)}"
            if cid in cited:
                cited[cid].append(name)
    covered, uncovered, unproven = [], [], []
    matrix: list[dict[str, Any]] = []
    for cid in wanted:
        tests = cited[cid]
        if not tests:
            if not citations_reliable:
                # The BASELINE output was truncated, so "nothing names it" may
                # just be the citation scrolling off. Blocking on that is a
                # gate no amount of correct work can pass.
                unproven.append(cid)
            else:
                uncovered.append((cid, "no test names it"))
            continue
        if not ran:
            # No mutation evidence exists, so nothing here can be CLAIMED. It
            # is not the producer's failure either -- the probe skips for
            # environmental reasons like tree size -- and failing on it would
            # be an unsatisfiable gate. Say plainly that the contract is
            # unproven instead of quietly crediting a passing test.
            unproven.append(cid)
            continue
        # `mutated` is only meaningful when the probe ran; without it, a citing
        # test that passes is the most that can be established.
        #
        # ABSENT IS NOT PASSED. When the implementation is emptied the suite
        # often dies before it reaches a given test, so that test simply does
        # not appear in the mutated results -- which is the STRONGEST evidence
        # it depends on the implementation. Scoring a missing name as "still
        # passed" made the gate reject work that was already correct, over and
        # over, with no way for the agent to satisfy it.
        verdicts = {name: _verdict(name) for name in tests if baseline.get(name)}
        matrix.append(
            {
                "criterion": cid,
                "tests": list(tests),
                "verdicts": dict(verdicts),
                "text": (criteria or [])[int(cid[1:]) - 1] if 0 < int(cid[1:]) <= len(criteria or []) else "",
            }
        )
        if "broke" in verdicts.values():
            covered.append(cid)
        elif "unknown" in verdicts.values():
            # Environmental, not the producer's failure -- same reasoning as a
            # probe that never ran.
            unproven.append(cid)
        elif not verdicts:
            uncovered.append((cid, "its test did not pass"))
        else:
            uncovered.append((cid, "its test still passed with the implementation emptied"))
    return {
        "attempted": True,
        "wanted": wanted,
        "covered": covered,
        "uncovered": uncovered,
        "unproven": unproven,
        "matrix": matrix,
    }


#: Ceiling on the throwaway copy the self-passing probe works in.
#: How many citing tests may be re-run in isolation per observation. Each one
#: costs two extra process launches (intact + mutant).
TEST_ISOLATION_CAP = int(os.environ.get("SHAREDAGENTS_SDLC_ISOLATION_CAP") or 12)


def isolation_command(run_cmd: str, test_name: str) -> str:
    """A command running ONLY *test_name*, or "" when the runner cannot filter.

    Isolation is what makes a mutation attributable. Running the whole suite
    and reading who is missing cannot distinguish "this test broke" from "an
    unrelated test crashed the process first" -- and the dominant real case is
    exactly that: emptying an implementation breaks the module IMPORT, so the
    runner dies before reporting anything and every criterion goes UNPROVEN.

    Ask for one test and silence becomes evidence: nothing else was in the run.
    """
    cmd = str(run_cmd or "").strip()
    name = str(test_name or "").strip()
    if not cmd or not name:
        return ""
    low = cmd.lower()
    # Never stack a second filter on a command that already has one.
    if "--test-name-pattern" in low or " -k " in low or " -run " in low:
        return ""
    pattern = re.escape(name)
    if "node --test" in low or "node:test" in low:
        return f"{cmd} --test-name-pattern {shlex.quote(pattern)}"
    if "npm test" in low or "npm run test" in low:
        # npm forwards everything after `--` to the script's runner.
        return f"{cmd} -- --test-name-pattern {shlex.quote(pattern)}"
    if "pytest" in low:
        # A NODE ID selects exactly one test; `-k` is a substring expression
        # and cannot express a free-text title at all.
        #
        # The selection has to REPLACE what the command already names, not be
        # appended to it. `pytest tests/test_config.py <nodeid>` runs all of
        # test_config.py AND the target -- so the "isolated" run was the whole
        # suite, every failure in it counted against the target, and a
        # tautology was credited as broken.
        base = _pytest_without_targets(cmd)
        if not base:
            return ""
        if "::" in name and ".py" in name:
            return f"{base} -v {shlex.quote(name)}"
        ident = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        return f"{base} -v -k {shlex.quote(name)}" if ident else ""
    return ""


def _pytest_without_targets(cmd: str) -> str:
    """*cmd* with its positional test paths removed, so one can be named."""
    kept: list[str] = []
    for token in cmd.split():
        bare = token.strip("\"'")
        if bare.endswith(".py") or ".py::" in bare:
            continue  # a positional target we are replacing
        # Nothing else is touched. `-v` here is DOCKER's volume flag as often
        # as it is pytest's verbosity, and stripping it broke the mount so the
        # isolated run had no code to test at all.
        kept.append(token)
    return " ".join(kept)


VERIFY_PROBE_MAX_FILES = 400
VERIFY_PROBE_MAX_BYTES = 24 * 1024 * 1024
_VERIFY_PROBE_SKIP = {
    ".git",
    "node_modules",
    ".worktrees",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".pytest_cache",
}

#: What to run when the repo declares nothing: (marker predicate, id, command).
#: Ordered, and every match runs -- a repo with both unit tests and a linter
#: should answer for both.
_VERIFY_AUTODETECT: tuple[tuple[str, str, str], ...] = (
    ("tests.js", "node-tests", "node tests.js"),
    ("test.js", "node-test", "node test.js"),
    ("package.json", "npm-test", "npm test --silent"),
    ("pytest.ini", "pytest", "python -m pytest -v"),
    ("pyproject.toml", "pytest", "python -m pytest -v"),
)

#: A bare `test/` or `tests/` directory says nothing about what is IN it. The
#: table above used to map `tests/` to `node --test` and `test/` to pytest, so
#: a Node project that happened to name its directory `test/` had pytest run
#: against it -- a check that fails for a reason the branch is not responsible
#: for, and the run rewound to `code` chasing it. Ask the language first.
_VERIFY_TEST_DIRS: tuple[str, ...] = ("tests", "test", "spec", "__tests__")
_NODE_SUFFIXES = (".js", ".mjs", ".cjs", ".ts")
_PY_SUFFIXES = (".py",)


def _language_checks(root: Path) -> list[dict[str, Any]]:
    """A runner for each test directory, chosen by what the files actually are."""
    found: list[dict[str, Any]] = []
    for name in _VERIFY_TEST_DIRS:
        folder = root / name
        if not folder.is_dir():
            continue
        suffixes = {path.suffix.lower() for path in folder.rglob("*") if path.is_file()}
        if suffixes & set(_NODE_SUFFIXES):
            found.append(
                {
                    "id": f"node-{name}",
                    "run": f"node --test {name}/",
                    "timeout": VERIFY_CHECK_TIMEOUT,
                    "declared": False,
                }
            )
        elif suffixes & set(_PY_SUFFIXES):
            found.append(
                {
                    "id": f"pytest-{name}",
                    "run": f"python -m pytest -v {name}",
                    "timeout": VERIFY_CHECK_TIMEOUT,
                    "declared": False,
                }
            )
    return found


def verify_checks(repo: str, worktree: str) -> list[dict[str, Any]]:
    """How this branch proves its own behaviour.

    A repo declares its checks in ``.sharedagents/verify.json``::

        {"checks": [{"id": "unit", "run": "node tests.js", "timeout": 300}]}

    That is the authority. With no declaration the worktree is sniffed for the
    usual entry points, so a brand-new repo still gets executed rather than
    silently believed.
    """
    root = Path(str(worktree or ""))
    if not root.is_dir():
        return []
    declared: list[dict[str, Any]] = []
    try:
        raw = json.loads((root / ".sharedagents" / "verify.json").read_text(encoding="utf-8"))
        for row in (raw.get("checks") or []) if isinstance(raw, dict) else []:
            command = str((row or {}).get("run") or "").strip()
            if command:
                declared.append(
                    {
                        "id": str(row.get("id") or "check").strip() or "check",
                        "run": command,
                        "timeout": float(row.get("timeout") or VERIFY_CHECK_TIMEOUT),
                        "declared": True,
                    }
                )
    except (OSError, ValueError, AttributeError):
        declared = []
    if declared:
        return declared
    seen: set[str] = set()
    found: list[dict[str, Any]] = []
    for marker, check_id, command in _VERIFY_AUTODETECT:
        if check_id in seen or not (root / marker).exists():
            continue
        # `package.json` without a test script runs, fails, and would be read
        # as a broken build rather than an absent check.
        if marker == "package.json":
            try:
                pkg = json.loads((root / marker).read_text(encoding="utf-8"))
                if not ((pkg.get("scripts") or {}).get("test") or "").strip():
                    continue
            except (OSError, ValueError, AttributeError):
                continue
        seen.add(check_id)
        found.append({"id": check_id, "run": command, "timeout": VERIFY_CHECK_TIMEOUT, "declared": False})
    # Directory heuristics are the LAST resort. When the project already
    # declares how to run its tests -- an npm `test` script, pytest config, a
    # root suite file -- that is authoritative, and synthesising a second
    # invocation of the SAME tests adds a way to fail without adding anything
    # to know. Live: `npm test` (node --test) passed while the extra
    # `node --test test/` failed, and the run rewound to `code` over it.
    if not found:
        found.extend(_language_checks(root))
    return found


def _run_one_check(check: dict[str, Any], cwd: Path) -> dict[str, Any]:
    """Run one check through a shell and keep what it actually said."""
    import subprocess

    command = str(check.get("run") or "")
    try:
        proc = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=float(check.get("timeout") or VERIFY_CHECK_TIMEOUT),
        )
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        code = proc.returncode
    except subprocess.TimeoutExpired:
        return {**check, "code": 124, "output": "timed out", "timed_out": True}
    except Exception as exc:  # noqa: BLE001
        return {**check, "code": 1, "output": f"{type(exc).__name__}: {exc}"}
    return {
        **check,
        "code": code,
        "output": output[-VERIFY_OUTPUT_CHARS:],
        # Only the TAIL is kept, so early TAP lines can vanish. Counting
        # occurrences across a truncated run would read a dropped line as a
        # test that stopped running -- coverage credited for nothing.
        "truncated": len(output) > VERIFY_OUTPUT_CHARS,
    }


def _docker_verify_spec(root: Path) -> tuple[dict[str, str] | None, str]:
    """Read the committed sandbox contract for a Docker-isolated run.

    The test commands may be project-specific, but the isolation boundary is
    not: the driver always supplies the network, mounts, capabilities, user
    namespace and read-only flags.  A repository therefore cannot weaken the
    boundary from ``verify.json``.
    """
    try:
        raw = json.loads((root / ".sharedagents" / "verify.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, AttributeError) as exc:
        return None, f"Docker verification requires .sharedagents/verify.json: {exc}"
    isolation = raw.get("isolation") if isinstance(raw, dict) else None
    if not isinstance(isolation, dict) or isolation.get("type") != "docker":
        return None, ("Docker verification requires isolation.type=docker in .sharedagents/verify.json")
    dockerfile = str(isolation.get("dockerfile") or "Dockerfile.test").strip()
    workdir = str(isolation.get("workdir") or "/workspace").strip()
    candidate = (root / dockerfile).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None, "Docker verification dockerfile must stay inside the worktree"
    if not candidate.is_file():
        return None, f"Docker verification dockerfile does not exist: {dockerfile}"
    if not workdir.startswith("/") or "\x00" in workdir:
        return None, "Docker verification workdir must be an absolute container path"
    return {"dockerfile": dockerfile, "workdir": workdir}, ""


class _DockerCheckRunner:
    """Build each immutable tree once, then run every check in nested Docker.

    ``DOCKER_HOST`` points at the Dev stack's dedicated DinD daemon.  No host
    path is mounted into a test container and no environment variable from the
    web process is forwarded.  The build context is the only input.
    """

    def __init__(self, spec: dict[str, str], run_id: str = ""):
        self.spec = dict(spec)
        self.run_id = re.sub(r"[^a-z0-9]", "", str(run_id).lower())[:12] or "run"
        self.images: dict[str, str] = {}
        self.build_errors: dict[str, str] = {}

    @staticmethod
    def _call(argv: list[str], *, cwd: Path, timeout: float) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                argv,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()
        except subprocess.TimeoutExpired:
            return 124, "timed out"
        except Exception as exc:  # noqa: BLE001
            return 125, f"{type(exc).__name__}: {exc}"

    def _image(self, cwd: Path, timeout: float) -> tuple[str, str]:
        key = str(cwd.resolve())
        if key in self.images:
            return self.images[key], ""
        if key in self.build_errors:
            return "", self.build_errors[key]
        tag = f"sharedagents-verify-{self.run_id}-{len(self.images) + 1}-{uuid.uuid4().hex[:8]}"
        code, output = self._call(
            [
                "docker",
                "build",
                "--pull=false",
                "--label",
                f"sharedagents.dev.run={self.run_id}",
                "-f",
                self.spec["dockerfile"],
                "-t",
                tag,
                ".",
            ],
            cwd=cwd,
            timeout=max(300.0, timeout),
        )
        if code != 0:
            detail = f"sandbox image build failed [exit {code}]: {output[-VERIFY_OUTPUT_CHARS:]}"
            self.build_errors[key] = detail
            return "", detail
        self.images[key] = tag
        return tag, ""

    def __call__(self, check: dict[str, Any], cwd: Path) -> dict[str, Any]:
        timeout = float(check.get("timeout") or VERIFY_CHECK_TIMEOUT)
        image, error = self._image(cwd, timeout)
        if not image:
            return {**check, "code": 125, "output": error, "isolation": "docker", "timed_out": False}
        command = str(check.get("run") or "")
        code, output = self._call(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,noexec,size=256m",
                "--tmpfs",
                "/run:rw,nosuid,nodev,noexec,size=16m",
                "--workdir",
                self.spec["workdir"],
                image,
                "/bin/bash",
                "-lc",
                command,
            ],
            cwd=cwd,
            timeout=timeout,
        )
        return {
            **check,
            "code": code,
            "output": output[-VERIFY_OUTPUT_CHARS:],
            "truncated": len(output) > VERIFY_OUTPUT_CHARS,
            "isolation": "docker",
            "sandbox_image": image,
            "timed_out": code == 124,
        }

    def close(self) -> None:
        # These images live only in the nested Dev daemon. Removing them is
        # bounded cleanup and cannot address host or production images.
        for image in self.images.values():
            self._call(["docker", "image", "rm", "-f", image], cwd=Path.cwd(), timeout=60.0)


def _run_capturing_stdout(
    argv: list[str], *, cwd: Path | None = None, timeout: float = 60.0, runner: Any = None
) -> tuple[int, str]:
    """``(code, STDOUT)``. Distinct from `_run_cmd`, which prefers stderr --
    here stdout carries a JSON answer that a stray warning must not displace."""
    if runner is not None:
        return runner(argv, cwd=cwd)
    import subprocess

    try:
        proc = subprocess.run(argv, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, ""
    except Exception:  # noqa: BLE001
        return 1, ""
    return proc.returncode, (proc.stdout or "")


#: Poison an implementation while KEEPING the module importable.
_JS_MUTANT_TEMPLATE = """// sharedagents mutant: exports preserved, behaviour removed
function __sharedagents_boom(name) {
  throw new Error('sharedagents mutant: ' + name + ' has no implementation');
}
%s
"""


def _js_export_shape(path: Path, *, runner: Any = None) -> dict[str, Any] | None:
    """``{"kind": "object"|"function", "keys": {name: typeof}}`` or None.

    Asks NODE what the module exports rather than parsing it, because the
    question is what `require` returns, and only node answers that exactly.
    """
    script = (
        "const m=require(process.argv[1]);"
        "const keys={};"
        "for (const k of Object.keys(m||{})) { try { keys[k]=typeof m[k]; }"
        " catch(e) { keys[k]='unknown'; } }"
        "console.log(JSON.stringify({kind: typeof m, keys}));"
    )
    code, out = _run_capturing_stdout(["node", "-e", script, str(path)], runner=runner)
    if code != 0:
        return None
    for line in reversed(out.strip().splitlines()):
        try:
            shape = json.loads(line)
        except ValueError:
            continue
        if isinstance(shape, dict) and "keys" in shape:
            return shape
    return None


def _write_js_mutant(path: Path, shape: dict[str, Any]) -> bool:
    """Replace every exported function with one that throws. Keep the rest."""
    keys = shape.get("keys") or {}
    if not isinstance(keys, dict):
        return False
    lines: list[str] = []
    if str(shape.get("kind")) == "function":
        lines.append("module.exports = function (...a) { return __sharedagents_boom('default'); };")
    else:
        lines.append("module.exports = {};")
    for name, kind in keys.items():
        if not re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", str(name)):
            continue
        if str(kind) == "function":
            lines.append(
                f"module.exports[{json.dumps(name)}] = function (...a) "
                f"{{ return __sharedagents_boom({json.dumps(name)}); }};"
            )
        else:
            # A data export: keep the KEY (so destructuring still works) and
            # remove the value, so anything asserting on it fails.
            lines.append(f"module.exports[{json.dumps(name)}] = undefined;")
    if len(lines) <= 1 and str(shape.get("kind")) != "function":
        return False  # nothing exported: gutting and preserving are identical
    path.write_text(_JS_MUTANT_TEMPLATE % "\n".join(lines), encoding="utf-8")
    return True


def _write_py_mutant(path: Path) -> bool:
    """Replace every function BODY with a raise, keeping the module importable.

    Signatures, decorators, classes, imports and module-level constants all
    survive, so `import` still works and every name still resolves -- only the
    behaviour is gone. Emptying the file instead would break the import and
    kill every test in it alike, including ones that never call the code.
    """
    import ast

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return False

    changed = 0

    class _Poison(ast.NodeTransformer):
        def _replace(self, node):
            self.generic_visit(node)
            nonlocal changed
            # Dunders are called while the module is still being IMPORTED --
            # `class Flask: testing = ConfigAttribute("TESTING")` runs
            # `ConfigAttribute.__init__` in the class body. Poisoning them
            # breaks the import, which kills every test in the file alike and
            # credits the tautologies among them.
            if node.name.startswith("__") and node.name.endswith("__"):
                return node
            doc = ast.get_docstring(node, clean=False)
            body: list[ast.stmt] = []
            if doc is not None:
                body.append(ast.Expr(value=ast.Constant(value=doc)))
            body.append(
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                        args=[ast.Constant(value=f"sharedagents mutant: {node.name} has no implementation")],
                        keywords=[],
                    ),
                    cause=None,
                )
            )
            node.body = body
            changed += 1
            return node

        visit_FunctionDef = _replace
        visit_AsyncFunctionDef = _replace

    try:
        poisoned = ast.fix_missing_locations(_Poison().visit(tree))
        text = ast.unparse(poisoned)
    except (AttributeError, ValueError, RecursionError):
        return False
    if not changed:
        # No behaviour to remove. Leaving the file ALONE is the honest mutation
        # of it -- and it still imports, which is what preservation means.
        return True
    path.write_text(text + "\n", encoding="utf-8")
    return True


def mutate_target(path: Path, *, runner: Any = None) -> tuple[bool, bool]:
    """``(module still imports, behaviour was actually removed)``.

    The second flag matters: if NOTHING could be poisoned the mutation is
    vacuous, and a suite that then passes has survived nothing. Reporting that
    as self-passing would fail honest work.
    """
    before = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    preserved = shape_preserving_mutant(path, runner=runner)
    if not preserved:
        return False, False
    after = path.read_text(encoding="utf-8", errors="replace")
    return True, after != before


def shape_preserving_mutant(path: Path, *, runner: Any = None) -> bool:
    """Poison *path*'s behaviour while keeping it importable. True on success.

    Emptying a file is a mutation of the MODULE, not of the behaviour: the
    import fails, the whole file's tests die together, and a pure tautology
    sitting beside a real test dies with it and is credited. Verified live --
    `assert.equal(1,1)` cited a criterion and the gate called it covered.

    Keeping the exports present and making each throw is the mutation we
    actually mean: a test that calls the implementation fails, and a test that
    never touches it still passes and is exposed.
    """
    if path.suffix == ".py":
        return _write_py_mutant(path)
    if path.suffix not in (".js", ".cjs"):
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    if "export " in text and "module.exports" not in text:
        return False  # ESM: `require` cannot introspect it
    shape = _js_export_shape(path, runner=runner)
    if not shape:
        return False
    return _write_js_mutant(path, shape)


def _exit_code(result: dict[str, Any]) -> int:
    """The process exit code, treating a MISSING one as failure.

    Not `int(result.get("code") or 1)`: a real exit code of 0 is falsy, so that
    reads success as failure. It cost a debugging cycle here, and on the mutant
    branch it would have been worse than a nuisance -- a test that SURVIVED the
    mutation (exit 0) would have been recorded as broken, crediting coverage to
    a test that proves nothing.
    """
    code = result.get("code")
    if code is None:
        return 1
    try:
        return int(code)
    except (TypeError, ValueError):
        return 1


#: pytest: 0 ok, 1 tests failed, 2 interrupted, 3 internal, 4 usage, 5 none
#: collected. Only 0 and 1 are statements ABOUT THE TEST; the rest mean it
#: never ran, and reading those as "it broke" credits a test that did nothing.
_PYTEST_RAN = frozenset({0, 1})


def _mutant_verdict(result: dict[str, Any], name: str, command: str) -> str:
    """``"broke"`` | ``"survived"`` | ``""`` for one isolated mutant run.

    ``""`` means the run did not answer -- a collection error, a crashed
    interpreter, a usage mistake. That is NOT evidence the test depends on the
    implementation, and treating it as such is how an import that the mutant
    itself broke gets credited to a tautology.
    """
    code = _exit_code(result)
    reported = _parse_test_results(result.get("output") or "").get(name)
    if reported is True and code == 0:
        return "survived"
    if reported is False:
        return "broke"  # it ran and it failed: attributable
    if "pytest" in command.lower() and code not in _PYTEST_RAN:
        return ""  # never collected -- says nothing about this test
    if code == 0:
        return "survived"
    return "broke"


def _isolate_citing_tests(
    intact: Path,
    mutant: Path,
    checks: list[dict[str, Any]],
    names: list[str],
    *,
    execute: Callable[[dict[str, Any], Path], dict[str, Any]] = _run_one_check,
) -> dict[str, str]:
    """Per-test mutation experiment. ``{test name: "broke"|"survived"|""}``.

    For each citing test, twice: run it ALONE against the intact tree, then
    ALONE against the mutant. The first run is the control -- it proves the
    filter really selects a passing test, so a filter that matches nothing
    cannot be mistaken for a test that broke. Only then is the second run
    evidence, and because nothing else is in that run, even a crash attributes
    to this test.

    ``""`` means the experiment could not be run (no filter for this runner,
    or the control did not pass); the caller treats that as UNPROVEN, never as
    a verdict.
    """
    out: dict[str, str] = {}
    for name in names[:TEST_ISOLATION_CAP]:
        for check in checks:
            command = isolation_command(str(check.get("run") or ""), name)
            if not command:
                continue
            probe = {
                **check,
                "run": command,
                "timeout": min(float(check.get("timeout") or VERIFY_CHECK_TIMEOUT), 120.0),
            }
            control = execute(probe, intact)
            if _exit_code(control) != 0:
                continue  # the filter selected nothing, or the test is flaky
            if not _parse_test_results(control.get("output") or "").get(name):
                continue  # it ran, but this is not the test we meant
            broken = execute(probe, mutant)
            verdict = _mutant_verdict(broken, name, str(check.get("run") or ""))
            if verdict:
                out[name] = verdict
            break
    return out


def _probe_tests_are_not_self_passing(
    worktree: Path,
    checks: list[dict[str, Any]],
    files: list[str],
    cite_tests: list[str] | None = None,
    *,
    execute: Callable[[dict[str, Any], Path], dict[str, Any]] = _run_one_check,
) -> dict[str, Any]:
    """Break the implementation in a COPY and re-run. Passing then is a lie.

    A suite that passes against a gutted implementation is not testing the
    implementation. This is the difference between "the tests pass" and "the
    tests establish anything", and it is the only cheap check that tells them
    apart mechanically.

    Nothing touches the real worktree: the probe works in a throwaway copy, so
    a crash mid-probe cannot damage the branch.
    """
    import shutil
    import tempfile

    cite_tests = list(cite_tests or [])

    targets = [name for name in mutable_files(files) if (worktree / name).is_file()]
    # Did git tell us what this branch CHANGED? If not, everything below is a
    # guess about the whole repository, and poisoning a whole package breaks
    # imports far away from the change -- on flask it broke `import flask`
    # itself, so a tautology's own module failed to load and was credited.
    scoped = bool(targets)
    if not targets:
        # git listed nothing usable (a fresh branch, a runner stub, a failed
        # `ls-files`). Walking the worktree keeps the probe running instead of
        # skipping it -- and a skipped probe reads as "not self-passing", which
        # is exactly the fail-open this exists to close.
        walked: list[str] = []
        for path in sorted(worktree.rglob("*")):
            # Ignore entries INSIDE this worktree, not its ancestors. Every
            # managed run itself lives beneath .worktrees/.
            if any(part in _VERIFY_PROBE_SKIP for part in path.relative_to(worktree).parts):
                continue
            if path.is_file():
                walked.append(str(path.relative_to(worktree)))
        targets = [name for name in mutable_files(walked) if (worktree / name).is_file()]
    # Never empty a file the checks THEMSELVES run. `node tests.js` against an
    # emptied `tests.js` exits 0 for the most trivial reason there is, and the
    # probe would report a working suite as self-passing.
    # Never empty a file the checks THEMSELVES run -- but compare PATHS, not
    # substrings. `src/flask/config.py` was excluded because its basename
    # `config.py` occurs inside `tests/test_config.py` in the command line, so
    # the one file the change actually touched was never mutated and every
    # criterion came back unattributable.
    run_paths: set[str] = set()
    for check in checks:
        for token in re.split(r"[\s;&|]+", str(check.get("run") or "")):
            token = token.strip("\"'()")
            if token:
                run_paths.add(token)
                run_paths.add(token.rsplit("/", 1)[-1])
    targets = [name for name in targets if name not in run_paths and name.rsplit("/", 1)[-1] not in run_paths]
    if not targets or not checks:
        return {
            "ran": False,
            "scoped": scoped,
            "isolated": {},
            "why": "no implementation files whose behaviour can be removed",
        }
    total = 0
    for path in worktree.rglob("*"):
        if any(part in _VERIFY_PROBE_SKIP for part in path.relative_to(worktree).parts):
            continue
        if path.is_file():
            total += 1
            if total > VERIFY_PROBE_MAX_FILES:
                return {"ran": False, "why": "worktree too large to copy safely"}
    tmp = Path(tempfile.mkdtemp(prefix="sharedagents-selfpass-"))
    try:
        copy = tmp / "wt"
        shutil.copytree(
            worktree,
            copy,
            ignore=shutil.ignore_patterns(*_VERIFY_PROBE_SKIP),
            symlinks=True,
        )
        size = sum(p.stat().st_size for p in copy.rglob("*") if p.is_file())
        if size > VERIFY_PROBE_MAX_BYTES:
            return {"ran": False, "why": "worktree too large to copy safely"}
        preserved = 0
        poisoned = 0
        for name in targets:
            kept, changed_it = mutate_target(copy / name)
            if kept:
                preserved += 1
                poisoned += 1 if changed_it else 0
                continue
            (copy / name).write_text(
                "// gutted by the sharedagents self-passing probe\n",
                encoding="utf-8",
            )
            poisoned += 1
        # Isolation can only tell a tautology from a real test when the module
        # still IMPORTS. If any target had to be gutted outright, a crash on
        # import fails every test in that file alike and would credit them all.
        shape_preserved = bool(targets) and preserved == len(targets)
        if not poisoned:
            # Every target was data only. Nothing was removed, so a green suite
            # proves nothing about self-passing -- and saying it does would
            # reject work that is fine.
            return {"ran": False, "why": "no behaviour to remove from the changed files"}
        survivors = []
        probe_truncated = False
        mutated_runs: dict[str, list[bool]] = {}
        for check in checks:
            result = execute(check, copy)
            probe_truncated = probe_truncated or bool(result.get("truncated"))
            # Per-test results from the MUTATED run are what tie a criterion to
            # the implementation: a citing test that survives this proves only
            # that it runs, not that it checks anything. Occurrences, not one
            # verdict per name -- a same-named test that stopped running is the
            # evidence a collapsed bool throws away.
            for name, runs in _parse_test_occurrences(result.get("output") or "").items():
                mutated_runs.setdefault(name, []).extend(runs)
            if int(result.get("code") or 0) == 0:
                survivors.append(str(check.get("id")))
        mutated = {name: all(runs) for name, runs in mutated_runs.items()}
        # Isolation is only attributable when the mutation was SCOPED to the
        # change. Unscoped, a citing test can fail because an unrelated module
        # stopped importing, which is not evidence about that test.
        isolated = (
            _isolate_citing_tests(worktree, copy, checks, cite_tests, execute=execute)
            if (shape_preserved and scoped)
            else {}
        )
        return {
            "ran": True,
            "broke": targets,
            "survivors": survivors,
            "self_passing": bool(survivors),
            "mutated_tests": mutated,
            "mutated_runs": mutated_runs,
            "truncated": probe_truncated,
            "isolated": isolated,
            "shape_preserved": shape_preserved,
            "scoped": scoped,
        }
    except Exception as exc:  # noqa: BLE001 — a probe that dies is not a verdict
        return {"ran": False, "why": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_verify_checks(run: dict[str, Any], *, probe: bool = True, runner: Any = None) -> dict[str, Any]:
    """Execute the branch's own checks and report what actually happened."""
    worktree = Path(str(run.get("worktree") or ""))
    repo = str(run.get("repo") or "")
    if not worktree.is_dir():
        return {
            "ok": False,
            "ran": 0,
            "failed": 0,
            "checks": [],
            "note": "no worktree to run anything in",
            "transcript": "",
        }
    checks = verify_checks(repo, str(worktree))
    if not checks:
        return {
            "ok": False,
            "ran": 0,
            "failed": 0,
            "checks": [],
            "note": (
                "nothing executable was found: the branch declares no "
                "`.sharedagents/verify.json` and none of the usual entry "
                "points exist, so no behaviour could be observed"
            ),
            "transcript": "",
        }
    docker_runner: _DockerCheckRunner | None = None
    execute: Callable[[dict[str, Any], Path], dict[str, Any]] = _run_one_check
    if str(run.get("test_isolation") or "").strip().lower() == "docker":
        spec, isolation_error = _docker_verify_spec(worktree)
        if not spec:
            return {
                "ok": False,
                "ran": 0,
                "failed": 0,
                "checks": [],
                "note": isolation_error,
                "transcript": f"[DOCKER ISOLATION] REFUSED: {isolation_error}",
                "isolation": "docker",
            }
        docker_runner = _DockerCheckRunner(spec, str(run.get("run_id") or ""))
        execute = docker_runner
    results = [execute(check, worktree) for check in checks]
    failed = [r for r in results if int(r.get("code") or 0) != 0]
    files = _branch_files(repo, str(run.get("branch") or ""), runner=runner, committed_only=False)
    selfpass: dict[str, Any] = {"ran": False, "why": "not attempted"}
    if probe and not failed:
        # Which test names cite a criterion? The probe needs them so it can run
        # each one in isolation, which is the only attributable experiment.
        wanted_ids = set(criterion_ids(list(run.get("criteria") or [])))
        cite_tests: list[str] = []
        for r in results:
            for name in _parse_test_occurrences(r.get("output") or ""):
                if name in cite_tests:
                    continue
                if any(f"C{int(n)}" in wanted_ids for n in CRITERION_CITE.findall(name)):
                    cite_tests.append(name)
        selfpass = _probe_tests_are_not_self_passing(
            worktree, checks, list(files), cite_tests=cite_tests, execute=execute
        )
    lines = []
    if docker_runner:
        lines.append(
            "[DOCKER ISOLATION] nested daemon; network=none; rootfs=read-only; "
            "capabilities=none; host mounts=none; inherited environment=none"
        )
    for r in results:
        lines.append(
            f"$ {r['run']}\n[exit {r['code']}]"
            f"{' DECLARED' if r.get('declared') else ' auto-detected'}\n"
            f"{r.get('output') or '(no output)'}"
        )
    if selfpass.get("ran"):
        broke = ", ".join(selfpass.get("broke") or [])
        if selfpass.get("self_passing"):
            lines.append(
                f"[SELF-PASSING PROBE] with {broke} emptied, these checks STILL "
                f"passed: {', '.join(selfpass.get('survivors') or [])}. They do "
                f"not exercise the implementation."
            )
        else:
            lines.append(
                f"[SELF-PASSING PROBE] every check failed once {broke} was "
                f"emptied, so the checks do exercise the implementation."
            )
    else:
        lines.append(f"[SELF-PASSING PROBE] not run: {selfpass.get('why')}")
    # Does the suite answer the CONTRACT, or just itself? Green plus
    # not-self-passing proves the tests are real; only this proves they are
    # about the nine things the operator froze.
    baseline_runs: dict[str, list[bool]] = {}
    truncated = False
    for r in results:
        truncated = truncated or bool(r.get("truncated"))
        for name, runs in _parse_test_occurrences(r.get("output") or "").items():
            baseline_runs.setdefault(name, []).extend(runs)
    baseline = {name: all(runs) for name, runs in baseline_runs.items()}
    probe_ran = bool(selfpass.get("ran"))
    mutated_runs = dict(selfpass.get("mutated_runs") or {}) if probe_ran else {}
    coverage = _criterion_coverage(
        list(run.get("criteria") or []),
        baseline,
        dict(selfpass.get("mutated_tests") or {}) if probe_ran else {},
        # Told, never inferred: a probe that ran and parsed nothing is evidence;
        # one that never ran is not, and both look like an empty dict.
        probe_ran=probe_ran,
        baseline_runs=baseline_runs,
        mutated_runs=mutated_runs or None,
        counts_reliable=not truncated and not selfpass.get("truncated"),
        # Citations come from the BASELINE alone, so a truncated probe says
        # nothing about whether a criterion was named.
        citations_reliable=not truncated,
        isolated=dict(selfpass.get("isolated") or {}) if probe_ran else None,
    )
    if coverage.get("attempted"):
        if coverage["uncovered"]:
            lines.append(
                "[CONTRACT COVERAGE] these frozen criteria are not cleared by "
                "any test that depends on the implementation:\n"
                + "\n".join(f"  {cid}: {why}" for cid, why in coverage["uncovered"])
                + "\nName each test for the criterion it clears, e.g. "
                "`[C3] divide by zero shows Error`."
            )
        elif coverage.get("unproven"):
            # Distinct from "cleared": nothing was demonstrated. Not a block --
            # the probe skips for environmental reasons and failing on that
            # would be a gate nobody can satisfy -- but it must not read as
            # proof either.
            lines.append(
                "[CONTRACT COVERAGE] UNPROVEN for "
                + ", ".join(coverage["unproven"])
                + ": a test names each of them and passed, but the mutation "
                "probe did not run, so nothing shows those tests depend on "
                "the implementation."
            )
        else:
            lines.append(
                f"[CONTRACT COVERAGE] all {len(coverage['wanted'])} frozen "
                f"criteria are cleared by a test that fails when the "
                f"implementation is emptied."
            )
        # The traceability matrix: requirement -> test -> what the mutation
        # showed. Written out so the link can be READ, not just asserted.
        rows = coverage.get("matrix") or []
        if rows:
            lines.append("[TRACEABILITY] requirement -> test -> mutation verdict")
            for row in rows:
                head = f"  {row['criterion']}: {str(row.get('text') or '')[:90]}"
                lines.append(head)
                if not row.get("tests"):
                    lines.append("    (no test names this criterion)")
                for name in row.get("tests") or []:
                    verdict = (row.get("verdicts") or {}).get(name) or "not run"
                    lines.append(f"    {name} -> {verdict}")
            changed = ", ".join(selfpass.get("broke") or []) or "(nothing)"
            lines.append(f"  mutation was applied to: {changed}")
    ok = not failed and not selfpass.get("self_passing") and not (coverage.get("attempted") and coverage["uncovered"])
    # The NOTE is what survives into the rewind feedback; the transcript is
    # clipped long before the producer sees it. So the instruction has to be
    # here, or the agent is told it failed without being told what to change --
    # which is exactly how two runs failed the same way four times.
    note = f"{len(results)} check(s) ran, {len(failed)} failed" + (
        "; the suite is self-passing" if selfpass.get("self_passing") else ""
    )
    if coverage.get("uncovered"):
        note += (
            f"; {len(coverage['uncovered'])} of {len(coverage['wanted'])} frozen "
            f"criteria are cleared by NO test: "
            + ", ".join(f"{cid} ({why})" for cid, why in coverage["uncovered"])
            + ". FIX: rename the tests so each one cites the criterion it "
            "clears -- the clearing evidence is numbered from 1 in the order "
            "it was frozen, so a test for the third line is named "
            "`[C3] <what it checks>`. Every criterion needs a citing test "
            "that passes now and FAILS when the implementation is emptied."
        )
    answer = {
        "ok": ok,
        "ran": len(results),
        "failed": len(failed),
        "checks": results,
        "self_passing": bool(selfpass.get("self_passing")),
        "probe": selfpass,
        "coverage": coverage,
        "note": note,
        "transcript": "\n\n".join(lines),
        "isolation": "docker" if docker_runner else "local",
    }
    if docker_runner:
        docker_runner.close()
    return answer


def _is_runtime_lsp_error(message: str) -> bool:
    """Syntax / cannot-run. Type-checker nits do not block merge."""
    low = " ".join((message or "").lower().split())
    if any(
        token in low
        for token in (
            "expected type",
            "not assignable",
            "is constant",
            "is not a known attribute",
            "type arguments",
        )
    ):
        return False
    return any(
        token in low
        for token in (
            "was not closed",
            "invalid syntax",
            "unexpected indent",
            "expected parameter",
            "expected expression",
            "unterminated",
            "syntaxerror",
            "indentationerror",
        )
    )


def _runtime_lsp_blockers(block: str) -> list[str]:
    out: list[str] = []
    for match in re.finditer(
        r'<diagnostics file="([^"]+)">ERROR \[(\d+):(\d+)\] (.*?)</diagnostics>',
        block or "",
        re.S,
    ):
        message = match.group(4).strip()
        if _is_runtime_lsp_error(message):
            out.append(f"{match.group(1)}:{match.group(2)}: {message}")
    return out


def unmerged_paths(run: dict[str, Any], *, runner: Any = None) -> list[str]:
    """Files left conflicted in this run's worktree, if any.

    An OPEN merge is a state the harness creates on purpose (see
    ``merge_base_into_branch``), and while it is open every other verdict about
    the tree is noise -- an LSP run on a file full of conflict markers reports
    syntax errors that say nothing about the agent's work. Observed live: the
    run blocked at `code` on "LSP reported code errors" for markers the
    pipeline itself had just written, without ever dispatching the agent that
    was supposed to resolve them.
    """
    root = Path(str(run.get("worktree") or ""))
    if not root.is_dir():
        return []
    files = _conflicted_files(root, runner=runner)
    if files:
        return files
    # Resolved but never committed still owes a commit.
    unfinished = _merge_in_progress(root, runner=runner)
    if unfinished:
        return [f"(the {unfinished} is resolved but not committed)"]
    return []


def lsp_code_errors(run: dict[str, Any], *, runner: Any = None) -> dict[str, Any]:
    """Fail-closed language-server check on the branch change.

    LSP must run. A missing language server is a blocker, not a skip.
    Only syntax / cannot-run diagnostics block merge; type nits do not.
    ``py_compile`` / ``node --check`` back the same files.
    """
    repo = str(run.get("repo") or "")
    branch = str(run.get("branch") or "")
    root = Path(str(run.get("worktree") or ""))
    blockers: list[str] = []
    if not root.is_dir():
        return {"ok": False, "blockers": ["no worktree to run LSP against"], "files": []}
    files = _implementation_files(_branch_files(repo, branch, runner=runner))
    py_files = [name for name in files if name.endswith(".py")]
    js_files = [
        name for name in files if name.endswith(".js") and "/vendor/" not in name and not name.startswith("vendor/")
    ]
    html_files = [name for name in files if name.endswith(".html") and "/vendor/" not in name]
    checked = py_files + js_files + html_files
    if py_files:
        try:
            from sharedagents_agent.tools.impl._lsp_gate import (
                lsp_errors_block,
                resolve_server_cmd_extended,
            )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "blockers": [f"LSP gate unavailable: {exc}"], "files": checked}
        if resolve_server_cmd_extended() is None:
            blockers.append("no language server on PATH — install basedpyright; merge stays closed")
        else:
            abs_paths = [str(root / name) for name in py_files]
            block = lsp_errors_block(
                abs_paths,
                root=str(root),
                timeout=45.0,
            )
            blockers.extend(_runtime_lsp_blockers(block))
        for rel in py_files:
            path = root / rel
            if not path.is_file():
                continue
            code, out = _run_cmd(
                ["python3", "-m", "py_compile", str(path)],
                timeout=20.0,
            )
            if code != 0:
                blockers.append(f"{rel}: {out[:400] or f'py_compile exit {code}'}")
    for rel in js_files:
        path = root / rel
        if not path.is_file():
            continue
        code, out = _run_cmd(["node", "--check", str(path)], timeout=20.0)
        if code != 0:
            blockers.append(f"{rel}: {out[:400] or f'node --check exit {code}'}")
    for rel in html_files:
        path = root / rel
        if not path.is_file():
            continue
        blockers.extend(f"{rel}: {msg}" for msg in _html_syntax_blockers(path))
    return {"ok": not blockers, "blockers": blockers, "files": checked}


def _html_syntax_blockers(path: Path) -> list[str]:
    """Language-appropriate HTML check — never open HTML as Python.

    Parses markup; if ``node`` exists, also ``node --check`` each inline
    ``<script>`` that is not ``type=module`` JSON/ld.
    """
    from html.parser import HTMLParser

    class _Scan(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.scripts: list[tuple[str, str]] = []
            self._buf: list[str] = []
            self._in_script = False
            self._script_kind = "js"
            self.errors: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag != "script":
                return
            kinds = {k: (v or "") for k, v in attrs}
            typ = kinds.get("type", "").lower()
            if typ and typ not in {
                "",
                "text/javascript",
                "application/javascript",
                "module",
            }:
                self._in_script = False
                return
            self._in_script = True
            self._script_kind = "mjs" if typ == "module" else "js"
            self._buf = []

        def handle_endtag(self, tag: str) -> None:
            if tag == "script" and self._in_script:
                self.scripts.append((self._script_kind, "".join(self._buf)))
                self._in_script = False
                self._buf = []

        def handle_data(self, data: str) -> None:
            if self._in_script:
                self._buf.append(data)

        def close(self) -> None:
            if self._in_script:
                self.errors.append("unterminated script element")
            super().close()

    raw = path.read_text(encoding="utf-8")
    scan = _Scan()
    try:
        scan.feed(raw)
        scan.close()
    except Exception as exc:  # noqa: BLE001 — parse failure is a blocker
        return [f"HTML parse failed: {exc}"]
    if scan.errors:
        return list(scan.errors)
    scripts = [(kind, text.strip()) for kind, text in scan.scripts if text.strip()]
    if not scripts:
        return []
    probe, probe_msg = _run_cmd(["node", "--version"], timeout=5.0)
    if probe == 127:
        return [f"node is required to check inline JavaScript: {probe_msg or 'not found'}"]
    out: list[str] = []
    import tempfile

    for index, (kind, text) in enumerate(scripts, 1):
        suffix = ".mjs" if kind == "mjs" else ".js"
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as handle:
            handle.write(text)
            tmp = handle.name
        try:
            code, msg = _run_cmd(["node", "--check", tmp], timeout=15.0)
        finally:
            Path(tmp).unlink(missing_ok=True)
        if code == 127:
            return [f"node is required to check inline JavaScript: {msg or 'not found'}"]
        if code != 0:
            out.append(f"inline script {index}: {msg[:300] or f'node --check exit {code}'}")
    return out


def preflight_stage(stage: str, run: dict[str, Any], *, runner: Any = None) -> list[str]:
    """Is the raw material for *stage* even PRESENT? Empty list means go.

    This is not validation -- it asks nothing about whether the work makes
    sense, only whether there is anything to look at. It is cheap and runs
    first so a stage with no input never costs an agent turn, and so the
    semantic validation below always has something to read.
    """
    repo = str(run.get("repo") or "")
    branch = str(run.get("branch") or "")
    missing: list[str] = []
    if not run.get("worktree"):
        missing.append("no worktree is provisioned")
    if not run.get("requirements"):
        missing.append("the contract has no condition for plan")
    if not run.get("criteria"):
        missing.append("the contract has no clearing evidence for test")
    if not run.get("issue_number"):
        missing.append("the run has no issue")
    if missing or stage == "plan":
        return missing

    files = _branch_files(repo, branch, runner=runner, committed_only=False)
    commits = _branch_commits(repo, branch, runner=runner)
    fused = run.get("fuse_plan_design", True)
    if stage == "design" and not _has_doc(files, "plan"):
        missing.append("the plan stage left no plan on the branch")
    if stage == "code":
        has_design = _has_doc(files, "design")
        has_plan = _has_doc(files, "plan")
        if not has_design and not (fused and has_plan):
            missing.append(
                "the plan stage left no brief on the branch"
                if fused
                else "the design stage left no design on the branch"
            )
    if stage == "tests" and not _implementation_files(files):
        # The design commit is not code. Counting any commit let a `code` stage
        # that wrote nothing satisfy the next stage's entry check.
        missing.append("the code stage committed no implementation to test")
    if stage == "deploy" and not commits:
        missing.append("there is nothing on the branch to assess for deploy readiness")
    return missing


def trace_stage(stage: str, run: dict[str, Any], marker: dict[str, Any], *, runner: Any = None) -> tuple[bool, str]:
    """Did *stage* leave any trace at all? ``(ok, why not)``.

    Not verification either: it only catches a stage that claimed PASS having
    committed nothing of its own kind. Whether what it committed satisfies the
    definition is `verify_delivery`'s question.
    """
    if str(marker.get("status") or "").upper() != "PASS":
        # A FAIL needs no corroboration, but it does need a reason.
        return (True, "") if str(marker.get("note") or "").strip() else (False, "reported FAIL without saying why")
    repo = str(run.get("repo") or "")
    branch = str(run.get("branch") or "")
    files = _branch_files(repo, branch, runner=runner)
    commits = _branch_commits(repo, branch, runner=runner)
    if stage in ("plan", "design") and not _has_doc(files, stage):
        return False, f"claimed PASS but no {stage} file is committed on the branch"
    if stage == "code" and not _implementation_files(files):
        # `commits` alone was satisfied by the design commit, so a `code` stage
        # that wrote nothing could claim PASS and advance.
        return False, "claimed PASS but committed no implementation, only the design"
    if stage == "tests":
        from sharedagents_agent.tools.impl.sdlc import _looks_like_driven_evidence

        note = str(marker.get("note") or "")
        evidence = str(marker.get("evidence") or "")
        stored = str(run.get("drive_evidence") or marker.get("drive_evidence") or "")
        missing: list[str] = []
        if not any(_looks_like_test(name) for name in files):
            if not _looks_like_driven_evidence(f"{note}\n{evidence}\n{stored}"):
                missing.append("a committed acceptance test or driven evidence")
        if "Dockerfile.test" not in files:
            missing.append("committed Dockerfile.test")
        if ".sharedagents/verify.json" not in files:
            missing.append("committed .sharedagents/verify.json")
        if missing:
            return False, "claimed PASS but left no " + ", ".join(missing)
        return True, ""
    if stage == "deploy" and not commits:
        return False, "claimed PASS but there is nothing on the branch"
    return True, ""


def is_test_stage_path(path: str) -> bool:
    """Whether TEST owns this repository path.

    The prompt is guidance; this predicate is the enforceable handoff. A test
    worker that discovers a product defect must leave it red for CODE rather
    than quietly becoming a second product author.
    """
    name = str(path or "").strip().replace("\\", "/")
    return name in {"Dockerfile.test", ".sharedagents/verify.json"} or _looks_like_test(name)


def changed_paths_since(run: dict[str, Any], baseline: str, *, runner: Any = None) -> list[str]:
    """Committed and working-tree paths changed since a stage started."""
    if not str(baseline or "").strip():
        return []
    repo = str(run.get("repo") or "")
    branch = str(run.get("branch") or "")
    path = repo_dir(repo)
    worktree = existing_worktree(path, branch, runner=runner) or str(path)
    root = Path(worktree)
    seen: list[str] = []
    for args, porcelain in (
        (["diff", "--name-only", f"{baseline}..HEAD"], False),
        (["status", "--porcelain"], True),
    ):
        code, out = _git(args, cwd=root, runner=runner)
        if code != 0:
            continue
        for line in out.splitlines():
            name = line[3:].strip() if porcelain else line.strip()
            if " -> " in name:
                name = name.rsplit(" -> ", 1)[-1].strip()
            if name and name not in seen:
                seen.append(name)
    return seen


def worktree_clean(run: dict[str, Any], *, runner: Any = None) -> tuple[bool, list[str]]:
    """Return whether the run worktree was inspectable and its dirty paths.

    A committed SHA is not a complete stage handoff while a newer edit remains
    only in the working tree.  Keep this observation separate from
    :func:`changed_paths_since`: that helper deliberately combines commits and
    working-tree edits for stage ownership, while this one asks the binary
    acceptance question "is HEAD the whole artifact?".
    """
    root = Path(str(run.get("worktree") or ""))
    if not root.is_dir():
        return False, ["run worktree is unavailable"]
    code, out = _git(
        ["status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        runner=runner,
    )
    if code != 0:
        return False, [out.strip() or "git status failed"]
    dirty: list[str] = []
    for line in out.splitlines():
        name = line[3:].strip() if len(line) >= 4 else line.strip()
        if " -> " in name:
            name = name.rsplit(" -> ", 1)[-1].strip()
        if name and name not in dirty:
            dirty.append(name)
    return True, dirty


# --------------------------------------------------------------------------
# The real bookends: validate the proposal, verify the delivery
# --------------------------------------------------------------------------
#
# The operator's rule, in their words:
#
#   VALIDATE  — "check if what's proposed makes sense"
#   VERIFY    — "check what's defined has been implemented"
#
# Both are judgements about MEANING, so neither can be a file-existence check.
# A stage therefore opens by reading the artifact it is about to build on and
# asking whether it holds up, and closes by comparing what it produced against
# what the previous artifact said it would produce.
#
# They run on the intake model -- local and free -- because they are short
# reads, and because a pipeline that bills a provider to second-guess itself
# would not survive contact with a long run.

#: What each stage is handed, and what it is answerable for.
STAGE_CONTRACT: dict[str, dict[str, str]] = {
    "plan": {
        "validates": "the ratified condition (what must be true) and the clearing evidence TEST will drive",
        "delivers": "an approach that satisfies every condition, an order of work, and an explicit out-of-scope list",
    },
    "design": {
        "validates": "the plan against every frozen condition",
        "delivers": "a technical design naming the concrete files, functions and "
        "endpoints that will satisfy every condition",
    },
    "code": {
        "validates": "the design, or the fused plan when design is skipped",
        "delivers": "an implementation of everything the spec named",
    },
    "tests": {
        "validates": "the implementation against the design or fused plan",
        "delivers": "a driven observation for every clearing-evidence line",
    },
    "deploy": {
        "validates": "the tests as evidence for the frozen clearing evidence",
        "delivers": "a verdict on whether every evidence line is still green",
    },
}

_VERIFY_SYSTEM = """\
You are the verification gate at the END of one stage of a software pipeline.

You are answering one question: HAS WHAT WAS DEFINED ACTUALLY BEEN IMPLEMENTED?

You are given what the stage was supposed to deliver and the evidence of what
it produced. For each defined element, decide whether it is really there in the
evidence. An element that is described, promised, or planned but absent from the
evidence is NOT delivered.

Do NOT judge whether the implementation is good, elegant, or complete beyond
what was defined. Only: was each defined thing delivered?

Reply with STRICT JSON and nothing else:
{"ok": bool, "missing": [str], "delivered": [str], "summary": str}
"""


def _decided(value: Any) -> bool:
    """Strict truth for a model's ``ok`` field.

    ``bool("false")`` is True, so a model that answers with the STRING "false"
    would have been read as a pass. Only a real boolean, or an unambiguous
    textual yes, counts.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "ok", "pass", "1")
    return False


def _bullets(
    items: Iterable[str], limit: int = 40, chars: int = MAX_BULLET_CHARS, *, summary: bool = False
) -> Material:
    """Requirements/criteria are operator-authored and therefore unbounded.

    They are also AUTHORITATIVE: what a gate judges against, not background
    So a list capped at `limit`, or an item clipped at `chars`, is
    material the gate did not see, and the result says so.

    ``summary=True`` is for the orientation lists -- which test files exist,
    which files the branch touches. Those are context, not the artefact under
    judgement, and this repository has hundreds of test files: marking them
    partial would make every `tests` stage GATE-UNUSABLE forever.
    """
    rows: list[str] = []
    dropped = 0
    seen = list(items)
    for index, item in enumerate(seen):
        text = str(item).strip()
        if not text:
            continue
        if len(text) > chars:
            dropped += len(text) - chars
            text = text[:chars] + "…"
        rows.append("- " + text)
        if len(rows) >= limit:
            rest = [str(x).strip() for x in seen[index + 1 :] if str(x).strip()]
            if rest:
                dropped += sum(len(x) for x in rest)
                rows.append(f"- … {len(rest)} more item(s) NOT shown …")
            break
    body = "\n".join(rows) or "(none)"
    return Material(body, total=len(body) + (0 if summary else dropped))


def stage_evidence(
    stage: str, run: dict[str, Any], marker: dict[str, Any], *, runner: Any = None, limit: int = MAX_MATERIAL_CHARS
) -> Material:
    """What *stage* actually produced -- what verification reads."""
    repo, branch = str(run.get("repo") or ""), str(run.get("branch") or "")
    reported = str(marker.get("note") or "").strip()
    if stage in ("plan", "design"):
        body = _compose(
            f"{stage.upper()} AS COMMITTED:\n", _or_else(_doc_text(repo, branch, stage, runner=runner), "(none)")
        )
    elif stage == "code":
        # The design is the definition for this stage and designs commonly name
        # required test files. Excluding tests made implemented designs
        # impossible to verify: the gate was asked to find files hidden from it.
        body = _compose(
            "IMPLEMENTATION DIFF:\n",
            _or_else(_branch_diff(repo, branch, runner=runner, exclude_tests=False), "(empty)"),
        )
    elif stage == "tests":
        # Judged on the whole test surface, not only what this stage touched: a
        # criterion already covered by an existing test IS covered, and calling
        # it missing would loop a correct run until its budget ran out.
        surface = _test_surface(repo, branch, runner=runner)
        listed = _bullets(surface, limit=120, summary=True)
        if not getattr(surface, "ok", True):
            listed = Material(str(listed), total=len(listed) + 1)
        # The FIRST thing a reviewer sees is what actually happened when the
        # branch was executed -- commands, exit codes, output, and whether the
        # suite survives its implementation being emptied. A diff of test files
        # says what was written; only this says what any of it does.
        observed = str(marker.get("verify_transcript") or "").strip()
        body = _compose(
            "OBSERVED BEHAVIOUR — the harness ran this itself in the worktree.\n"
            "These exit codes were not reported by an agent; treat them as the\n"
            "primary evidence and the narrative below as a claim about them.\n\n",
            observed or "(the harness ran nothing — see the blockers)",
            "\n\nTESTS ADDED OR CHANGED ON THIS BRANCH:\n",
            _or_else(_branch_diff(repo, branch, runner=runner, tests_only=True), "(none)"),
            "\n\nALL TEST FILES PRESENT IN THE WORKTREE:\n",
            listed,
        )
    else:
        # `deploy` delivers a deploy-readiness verdict, so its verdict is the evidence — a list
        # of changed filenames says nothing about whether criteria were met.
        files = _branch_files(repo, branch, runner=runner)
        shown = _bullets(files, limit=120, summary=True)
        if not getattr(files, "ok", True):
            shown = Material(str(shown), total=len(shown) + 1)
        body = _compose(
            "THE DEPLOY-READINESS VERDICT:\n", reported or "(nothing)", "\n\nWHAT IS ON THE BRANCH:\n", shown
        )
    # The deploy verdict is already in body. Repeating it wastes reviewer
    # context without adding evidence.
    whole = (
        body
        if stage not in ("plan", "design", "code", "tests")
        else _compose(
            body,
            "\n\nWHAT THE STAGE REPORTED:\n",
            reported or "(nothing)",
        )
    )
    return Material(whole[:limit], total=whole.total)


def _doc_text(
    repo: str, branch: str, kind: str = "", *, runner: Any = None, limit: int = MAX_MATERIAL_CHARS
) -> Material:
    """The committed plan/design prose on this branch, concatenated.

    *kind* narrows to one document; empty takes every design document, which is
    what the old single-spec stage wanted.
    """
    try:
        path = repo_dir(repo)
    except WorktreeError:
        return ""
    worktree = Path(existing_worktree(path, branch, runner=runner) or path).resolve()
    chunks: list[str] = []
    dropped = 0
    budget = limit
    names = _branch_files(repo, branch, runner=runner)
    if not getattr(names, "ok", True):
        # git could not list the branch. Unknown is not empty.
        dropped += 1
    for name in names:
        if not _looks_like_spec(name):
            continue
        if budget <= 0:
            # Out of budget is not "nothing more to read": these documents are
            # part of the definition and the gate never sees them.
            dropped += 1
            continue
        if kind and _doc_kind(name) not in (kind, ""):
            continue
        target = worktree / name
        try:
            # A committed symlink under a spec-marked path would otherwise be
            # followed straight out of the worktree and posted to the model.
            resolved = target.resolve()
            resolved.relative_to(worktree)
            if not resolved.is_file():
                continue
            size = resolved.stat().st_size
            if size > MAX_SPEC_FILE_BYTES:
                # Not "nothing to see": the whole document is missing from what
                # the gate will read, and it has to know that.
                dropped += int(size)
                chunks.append(f"--- {name} (too large to read: {size} bytes) ---")
                continue
            whole = resolved.read_text(encoding="utf-8", errors="replace")
            body = whole[:budget]
            dropped += len(whole) - len(body)
        except (OSError, ValueError):
            # A document we could not read is still part of the definition.
            dropped += 1
            continue
        chunk = f"--- {name} ---\n{body}"
        budget -= len(chunk)
        chunks.append(chunk)
    body = "\n\n".join(chunks)
    return Material(body[:limit], total=len(body) + dropped)


def _test_surface(repo: str, branch: str, *, runner: Any = None, limit: int = 400) -> list[str]:
    """Every test file in the worktree, not just the ones this branch touched.

    A success criterion can be satisfied by a test that already existed. Asking
    only "what did you add" would call that criterion uncovered, and the run
    would rework itself to death trying to satisfy a gate that was never open.
    """
    try:
        path = repo_dir(repo)
    except WorktreeError:
        failed = FileList()
        failed.ok = False
        return failed
    worktree = Path(existing_worktree(path, branch, runner=runner) or path)
    code, out = _git(["ls-files"], cwd=worktree, runner=runner)
    if code != 0:
        # git did not answer. An empty surface would read as "this branch has
        # no tests", which is a very different claim.
        failed = FileList()
        failed.ok = False
        return failed
    return FileList([name for name in out.splitlines() if _looks_like_test(name)][:limit])


def _branch_diff(
    repo: str,
    branch: str,
    *,
    runner: Any = None,
    tests_only: bool = False,
    exclude_tests: bool = False,
    limit: int = MAX_MATERIAL_CHARS,
) -> Material:
    """The branch's committed diff, optionally narrowed to (or away from) tests."""
    try:
        path = repo_dir(repo)
    except WorktreeError:
        return ""
    worktree = Path(existing_worktree(path, branch, runner=runner) or path)
    ref = default_base(repo, runner=runner)
    # Filtering through a comprehension returns a plain list and throws `.ok`
    # away -- the same provenance loss this whole change is about, one level
    # down. The listing is captured first and the filter keeps its verdict.
    listing = _branch_files(repo, branch, runner=runner)
    names = FileList(
        name
        for name in listing
        if not _looks_like_spec(name)
        and (not tests_only or _looks_like_test(name))
        and (not exclude_tests or not _looks_like_test(name))
    )
    names.ok = bool(getattr(listing, "ok", True))
    if not getattr(names, "ok", True):
        # Unknown is not empty: a gate must not read a collection failure as a
        # branch with nothing on it.
        return Material("", total=1)
    if not names:
        return Material("")
    # Files past the cap are not in the diff at all. `--stat-count` shapes the
    # diffstat, not the patch, so it bounds nothing here: the omission has to
    # be reported, and the capture has to be bounded for real.
    omitted = max(0, len(names) - MAX_DIFF_FILES)
    code, out = _git(
        [
            "diff",
            f"--unified={DIFF_CONTEXT_LINES}",
            "--no-color",
            f"--stat-count={MAX_DIFF_FILES}",
            f"{ref}...HEAD",
            "--",
            *names[:MAX_DIFF_FILES],
        ],
        cwd=worktree,
        runner=runner,
        max_bytes=limit * 2,
    )
    if code != 0:
        return Material("", total=1)
    if omitted:
        out += f"\n… {omitted} more changed file(s) are NOT included in this diff (cap {MAX_DIFF_FILES}) …"
    if len(out) > limit or omitted:
        head = out[:limit] + f"\n… diff truncated at {limit} characters …" if len(out) > limit else out
        # `omitted` files were never captured, so their size is unknown; count
        # one character each so `partial` is TRUE and no verdict is trusted.
        return Material(head, total=max(len(out), len(head)) + omitted)
    return out


MAX_GATE_WINDOWS = 6


_RECHECK_SYSTEM = """\
You are checking whether specific items appear in an excerpt of a larger body of
material. You are NOT judging quality, completeness or design.

Answer ONLY with a JSON object of this exact shape:

  {"present": [{"id": <integer>, "quote": "<text copied verbatim from the
   excerpt that shows the item is present>"}]}

Rules:
- "id" must be one of the numbered items given to you. Never invent an id.
- "quote" must be copied CHARACTER FOR CHARACTER from the excerpt. A quote that
  does not appear verbatim is discarded and the item stays open.
- Include an item only if the excerpt itself shows it. Say nothing about items
  you cannot see here -- another excerpt may contain them.
- An empty list is a perfectly good answer.
"""

MIN_RECHECK_QUOTE = 12


_RECHECK_SYSTEM = """\
You are checking whether specific items appear in an excerpt of a larger body of
material. You are NOT judging quality, completeness or design.

Answer ONLY with a JSON object of this exact shape:

  {"present": [{"id": <integer>, "quote": "<text copied verbatim from the
   excerpt that shows THAT item is present>"}]}

Rules:
- "id" must be one of the numbered items given to you. Never invent an id.
- "quote" must be copied CHARACTER FOR CHARACTER from the excerpt, and must be
  the text that shows THAT SPECIFIC item is present. A quote that does not
  appear verbatim, or that has nothing to do with the item, is discarded and
  the item stays open.
- Include an item only if the excerpt itself shows it. Say nothing about items
  you cannot see here -- another excerpt may contain them.
- An empty list is a perfectly good answer.
"""

MIN_RECHECK_QUOTE = 12


#: Words too common in requirement prose to identify anything on their own.
_QUOTE_STOPWORDS = frozenset(
    [
        "must",
        "should",
        "shall",
        "will",
        "need",
        "needs",
        "needed",
        "implement",
        "implemented",
        "implementing",
        "implementation",
        "support",
        "supports",
        "supported",
        "provide",
        "provides",
        "provided",
        "ensure",
        "ensures",
        "ensured",
        "handle",
        "handles",
        "handled",
        "code",
        "tests",
        "test",
        "testing",
        "there",
        "their",
        "this",
        "that",
        "these",
        "those",
        "with",
        "without",
        "from",
        "into",
        "when",
        "where",
        "which",
        "while",
        "have",
        "has",
        "had",
        "been",
        "being",
        "does",
        "doing",
        "done",
        "make",
        "makes",
        "made",
        "also",
        "only",
        "just",
        "still",
        "even",
    ]
)

#: A quote asserting ABSENCE is not proof of presence, however verbatim it is.
_NEGATION = re.compile(
    r"\b(not|no|never|without|missing|absent|lacks?|lacking|except|instead|"
    r"todo|fixme|unimplemented|unsupported|pending|stub|placeholder|"
    r"should|would|could)\b",
    re.I,
)


def _looks_like_symbol(word: str) -> bool:
    """Does this word name a concrete thing in code rather than describe one?"""
    return (
        "_" in word
        or (word.isupper() and len(word) >= 4)
        or (word[:1].islower() and any(c.isupper() for c in word[1:]))
    )


def _item_tokens(item: str) -> set[str]:
    """The symbol(s) a quote must carry to be about this item.

    ONLY symbols. A prose complaint cannot be answered by quoting text: every
    looser rule was exploitable, the last being that "reset user password" was
    cleared by "The user guide documents password storage; reset the cache
    after deployment" -- each word present, nothing proved.

    Prose items therefore stay outstanding, which fails the stage or blocks it.
    That is the safe direction and costs little: the false negative this whole
    mechanism exists to fix was a gate reporting named SYMBOLS missing --
    CANONICAL_STATES, TRANSITION_TABLE -- which is exactly what a symbol item
    is.
    """
    words = [w for w in re.findall(r"[A-Za-z0-9_]+", str(item)) if len(w) >= 4 and w.lower() not in _QUOTE_STOPWORDS]
    return {w.lower() for w in words if _looks_like_symbol(w)}


def _mentions(text: str, token: str) -> bool:
    """Is `token` present in `text` as a whole word?

    Boundaries are unicode-aware and treat `_` as part of a word. ASCII-only
    guards let "carte" and "cart<CJK>" clear "cart", and excluding `_` let
    "foo_bar" clear "foo" -- a different identifier entirely.
    """
    return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) is not None


def _quote_supports(item: str, quote: str, raw: str) -> bool:
    """Does this quote actually show THIS item, in material we really read?

    Every loosening here has been exploited in review, so each rule earns its
    place. The quote must be long enough to identify something; must appear in
    the RAW window rather than in the truncation note this module writes
    itself; must not be asserting the item's ABSENCE; and must carry every
    SYMBOL the item names, at unicode word boundaries.

    An item naming no symbol cannot be cleared by a quote at all. Judging prose
    by word overlap kept producing quotes that mentioned every word and proved
    nothing.
    """
    quote, item = str(quote or ""), str(item or "")
    if len(quote.strip()) < MIN_RECHECK_QUOTE:
        return False
    if quote not in raw:  # the RAW window, never the note
        return False
    if _NEGATION.search(quote):
        return False
    wanted = _item_tokens(item)
    if not wanted:
        return False
    low = re.sub(r"\s+", " ", quote.lower())
    return all(_mentions(low, token) for token in wanted)


def _resolve_over_remaining(
    *,
    stage: str,
    kind: str,
    outstanding: list[str],
    material: str,
    start: int,
    window: int,
    chat: Any,
    deadline: float | None = None,
) -> tuple[list[str], bool]:
    """Show the judge what it never saw, asking only about outstanding items.

    A negative verdict reached on a PARTIAL view is not a finding -- it is a
    question about the part that was cut. Converting it to PASS invents a false
    positive; converting it to FAIL invents the false negative that hit
    production. The honest move is to go and look at the rest.

    The cursor advances by what the model was ACTUALLY shown, not by the
    nominal window size. Pre-partitioning the material and then fitting each
    part again drops whatever the second fit trimmed, while still reporting
    full coverage -- about 120 characters per window with a large outstanding
    list.

    Returns ``(still_outstanding, fully_covered)``; ``fully_covered`` is False
    when the material was not read to the end, and the caller must NOT read
    that as a pass.
    """
    if window <= 0:
        return outstanding, False
    if start >= len(material):
        return outstanding, True
    if deadline is None:
        deadline = _gate_deadline()
    cursor, reads = start, 0
    while cursor < len(material) and reads < MAX_GATE_WINDOWS - 1:
        if time.monotonic() >= deadline:
            # Out of time is the same as a model that cannot answer: report it
            # as an unusable gate rather than a verdict about the work.
            raise SdlcError(f"{kind} re-check ran out of time for {stage}")
        if not outstanding:
            # Coverage describes how much MATERIAL was read. Clearing every
            # item early does not mean the rest of it was examined.
            return outstanding, cursor >= len(material)
        raw = material[cursor : cursor + window]
        numbered = "\n".join(f"[{i}] {item}" for i, item in enumerate(outstanding, 1))
        lead = (
            f"You judged the {kind} for stage '{stage}' incomplete, but you "
            f"were shown only PART of it. Here is ANOTHER PORTION of the "
            f"same material.\n\nOUTSTANDING ITEMS:\n"
        )
        mid = f"\n\nANOTHER PORTION OF THE {kind.upper()}:\n"
        # The recovery path must not overflow the budget it exists to protect.
        fit = _fit_gate_prompt(numbered, raw, overhead=len(lead) + len(mid))
        shown = fit.evidence_shown
        if shown <= 0:
            # The outstanding list alone fills the budget: no progress is
            # possible, and pretending otherwise would loop forever.
            log.warning("sdlc: %s re-check cannot fit any material for %s", kind, stage)
            return outstanding, False
        seen_raw = raw[:shown]
        try:
            parsed = _ask_gate(
                chat,
                [
                    {"role": "system", "content": _RECHECK_SYSTEM},
                    {"role": "user", "content": lead + fit.definition + mid + fit.evidence},
                ],
                what=f"{kind} re-check",
                deadline=deadline,
            )
        except Exception as error:  # noqa: BLE001 -- an unreachable model does
            # not decide the verdict. This is the same failure as a gate that
            # cannot answer, and is reported as such rather than becoming an
            # INCONCLUSIVE block.
            log.warning("sdlc: %s re-check unavailable for %s: %s", kind, stage, error)
            raise SdlcError(f"{kind} re-check unavailable: {error}") from error
        cleared: set[int] = set()
        for row in parsed.get("present") or []:
            if not isinstance(row, dict):
                continue
            index = row.get("id")
            # A strict integer in range: bools, floats and "01" are not ids.
            if isinstance(index, bool) or not isinstance(index, int):
                continue
            if not 1 <= index <= len(outstanding):
                continue
            if _quote_supports(outstanding[index - 1], row.get("quote"), seen_raw):
                cleared.add(index)
            else:
                log.info("sdlc: %s re-check quote for item %s does not support it; the item stays open", kind, index)
        if cleared:
            outstanding = [item for i, item in enumerate(outstanding, 1) if i not in cleared]
        cursor += shown
        reads += 1
    return outstanding, cursor >= len(material)


#: Prefix marking a gate answer that could not be used as a verdict. It rides
#: back on an ``ok=True`` result: the stage advances, but the run carries the
#: fact durably so a consequential action can refuse it later.
#: A manual resume lifts the unattended recovery cap, but the route cannot
#: prove a human is behind it, so the lift is budgeted too.
MANUAL_RESUME_MAX = 10

GATE_UNUSABLE = "GATE-UNUSABLE:"


def _review_blocked(ok: bool, notes: Iterable[str]) -> bool:
    """A reviewer that did not decide, or decided fail, cannot clear tests."""
    if not ok:
        return True
    return any(str(note).startswith(GATE_UNUSABLE) or str(note).startswith(GATE_INCONCLUSIVE) for note in notes)


def reviews_agree(
    ok_a: bool,
    notes_a: list[str],
    ok_b: bool,
    notes_b: list[str],
) -> tuple[bool, list[str]]:
    """Tests PASS only when verify A and verify B both decide and both pass.

    Disagreement, an unusable/inconclusive gate, or a missing reviewer is a
    FAIL. The tests stage cannot be voided by one reviewer passing alone.
    """
    notes = [f"verify-a: {n}" for n in notes_a] + [f"verify-b: {n}" for n in notes_b]
    blocked_a = _review_blocked(ok_a, notes_a)
    blocked_b = _review_blocked(ok_b, notes_b)
    if not blocked_a and not blocked_b:
        return True, notes
    if blocked_a != blocked_b or ok_a != ok_b:
        notes.append("verify A and verify B did not agree — tests cannot pass")
    else:
        notes.append("verify A and verify B both rejected — tests cannot pass")
    return False, notes


#: A gate that could not finish reading. It BLOCKS rather than failing open,
#: but it reviewed nothing either, so it is recorded like any undecided gate --
#: the driver was only recording notes with the other prefix.
GATE_INCONCLUSIVE = "INCONCLUSIVE:"

#: A gate mark is permanent, so a blip must not create one. The model is asked
#: again before a gate is declared unusable -- this is where a transient outage
#: is forgiven, rather than by letting a later verdict erase the record.
GATE_ATTEMPTS = 4
#: Backoff between attempts. Flat half-seconds retried straight back into the
#: same congestion: the gates share one vLLM with the build agents, so a stall
#: lasts as long as somebody else's long generation, not milliseconds.
GATE_RETRY_BACKOFF_S = (2.0, 8.0, 30.0)

#: The gate's own sleep, so a test can suppress the backoff without patching
#: `time.sleep` for the whole process -- `dev_sdlc.time` IS the time module, so
#: patching through it stops every unrelated poll in the suite too.
_gate_sleep = time.sleep
#: Retrying a model that is genuinely down must not outlast the driver's lease
#: on the run: four 120-second timeouts plus 40s of backoff is about nine
#: minutes of a two-hour lease per gate, and there are two gates and a re-check
#: per stage. The driver renews around gate work and before each stage retry.
#:
#: This bounds when a gate may START new model work, not when a call returns --
#: a request already in flight is not abandoned, because the time is spent
#: either way and discarding a good answer only wastes it. So the real ceiling
#: is this budget plus ONE model timeout (600 + 120 = 720s), which is what has
#: to stay well inside the two-hour lease.
#: Raised from 180s once a mark became PERMANENT: at 180 a gate got two
#: attempts, so a four-minute contention window was enough to block a release
#: for good. At 600 a mark means the model was unusable for ten minutes, which
#: is worth blocking on. Still far inside the two-hour driver lease.
GATE_TIME_BUDGET_S = 600.0


def _gate_deadline() -> float:
    """When ALL of one gate's model work must be finished by.

    Computed once per gate and threaded through every call it makes, because a
    per-call budget is not a bound: each re-check window started a fresh one,
    so a gate could still outlast the driver's lease.
    """
    return time.monotonic() + GATE_TIME_BUDGET_S


def _ask_gate(
    chat: Any,
    messages: list[dict[str, str]],
    *,
    what: str,
    attempts: int = GATE_ATTEMPTS,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Ask a gate, retrying a model that cannot answer. Raises if none does."""
    last: Exception | None = None
    if deadline is None:
        deadline = _gate_deadline()
    for attempt in range(1, max(1, int(attempts)) + 1):
        if attempt > 1 and time.monotonic() >= deadline:
            # Checked before STARTING work, never to discard an answer already
            # paid for.
            break
        try:
            return _parse_json_object(chat(messages))
        except Exception as error:  # noqa: BLE001 — retried, then reported
            last = error
            if attempt >= attempts or time.monotonic() >= deadline:
                break
            pause = GATE_RETRY_BACKOFF_S[min(attempt - 1, len(GATE_RETRY_BACKOFF_S) - 1)]
            log.info(
                "sdlc: %s gate attempt %d/%d failed (%s); retrying in %.0fs", what, attempt, attempts, error, pause
            )
            if pause:
                _gate_sleep(pause)
    raise last if last else SdlcError(f"{what} gate produced nothing")


def _unjustified(kind: str, stage: str) -> tuple[bool, list[str]]:
    """A refusal that names nothing is an UNUSABLE answer, not a verdict.

    Blocking on it stalls stages with no producer to lap back to (`plan` has
    none), and passing it silently -- the original behaviour -- turns "I am not
    sure" into approval. It is the same class of failure as a timeout or an
    unparseable payload, which this module already fails OPEN on.

    Failing open is safe only because it is not silent. A log line is not
    enough: a model answering this way every time would quietly turn both gates
    into no-ops and nothing downstream would know. The result
    carries a durable marker the driver records against the run, so a run whose
    gates went unusable cannot reach a PR unremarked.
    """
    log.warning(
        "sdlc: %s gate for %s refused but named nothing specific; treating as UNUSABLE and failing open", kind, stage
    )
    return True, [
        f"{GATE_UNUSABLE} the {kind} gate for '{stage}' refused without naming anything, so it decided nothing"
    ]


def validate_proposal(
    stage: str, run: dict[str, Any], *, chat: Callable[[list[dict[str, str]]], str] | None = None, runner: Any = None
) -> tuple[bool, list[str]]:
    """Per-stage LLM validate is retired.

    Validation of the requirements happens once at clarify/contract
    (``contract_defects`` + operator ratification). After that, stages
    generate and then verify. This hook stays so older driver patches keep
    importing it; it always passes and records nothing useful.
    """
    return True, []


def verify_delivery(
    stage: str,
    run: dict[str, Any],
    marker: dict[str, Any],
    *,
    chat: Callable[[list[dict[str, str]]], str] | None = None,
    runner: Any = None,
    force: bool = False,
) -> tuple[bool, list[str]]:
    """Has what this stage defined actually been implemented? ``(ok, missing)``.

    Only a claimed PASS is checked -- a FAIL is already telling the truth.
    ``force=True`` is an independent reviewer: that path fails CLOSED so a
    missing verify A or verify B cannot void the tests gate.
    """
    if not force and not uses_llm_gate(stage):
        return True, []
    contract = STAGE_CONTRACT.get(stage)
    if not contract or str(marker.get("status") or "").upper() != "PASS":
        return True, []
    evidence = stage_evidence(stage, run, marker, runner=runner)
    # Each stage answers to a different definition. `code` implements the
    # design; `tests` and `deploy` answer to the clearing evidence. Judging the
    # tests against prose design text was asking the wrong question of them.
    definition = (
        _doc_text(str(run.get("repo") or ""), str(run.get("branch") or ""), "design", runner=runner)
        if stage == "code"
        else _bullets(run.get("criteria") or [])
    )
    # The criteria are context too, so they go INSIDE the budget — added after
    # the fit they simply push the evidence back out again.
    # Built from the same strings that size the reserve -- see validate_proposal.
    prefix = (
        f"STAGE THAT JUST RAN: {stage}\n"
        f"IT WAS ANSWERABLE FOR: {contract['delivers']}\n\n"
        f"WHAT WAS DEFINED, AND THE CLEARING EVIDENCE:\n"
    )
    middle = "\n\nEVIDENCE OF WHAT IT PRODUCED:\n"
    # Capture provenance BEFORE the definition is rebuilt into a plain string:
    # the rebuild is where "there was more of me" was being thrown away.
    criteria_text = _bullets(run.get("criteria") or [])
    definition_partial = bool(getattr(definition, "partial", False) or criteria_text.partial)
    definition_shown = int(getattr(definition, "shown", 0))
    definition_total = int(getattr(definition, "total", 0))
    # Only code has a separate design definition. Other gates already use
    # criteria as their definition; duplicating every criterion burns context
    # in BOTH independent reviews and can crowd out real evidence.
    if stage == "code":
        definition = _compose(
            _or_else(definition, "(nothing explicit)"), "\n\nCLEARING EVIDENCE (for test):\n", criteria_text
        )
    else:
        definition = _or_else(definition, "(nothing explicit)")
    fit = _fit_gate_prompt(definition, evidence, overhead=len(prefix) + len(middle))
    prompt = prefix + fit.definition + middle + fit.evidence
    deadline = _gate_deadline()
    try:
        parsed = _ask_gate(
            (chat or _gate_chat),
            [
                {"role": "system", "content": _VERIFY_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            what="verification",
            deadline=deadline,
        )
    except Exception as error:  # noqa: BLE001 — a single structural verify
        # fails open; an independent reviewer (force=True) fails closed.
        log.warning("sdlc: verification unavailable for %s: %s", stage, error)
        notes = [f"{GATE_UNUSABLE} the verification gate for '{stage}' could not be reached: {error}"]
        return (False, notes) if force else (True, notes)
    missing = _strings(parsed.get("missing"))
    ok = _decided(parsed.get("ok"))
    # Two answers that are not verdicts: one that refuses without naming
    # anything, and one reached on an excerpt. Neither may decide the stage.
    if not ok and not missing:
        return _unjustified("verification", stage)
    if getattr(evidence, "partial", False):
        return True, [
            f"{GATE_UNUSABLE} the verification gate for '{stage}' saw "
            f"only {getattr(evidence, 'shown', 0)} of "
            f"{getattr(evidence, 'total', 0)} characters of the "
            f"evidence, so its answer is about an excerpt"
        ]
    if definition_partial:
        return True, [
            f"{GATE_UNUSABLE} the verification gate for '{stage}' saw "
            f"only {definition_shown} of {definition_total} characters "
            f"of what the stage was meant to deliver"
        ]
    if (fit.evidence_cut or fit.definition_cut) and ok and not missing:
        # The DEFINITION side can be clipped while the evidence fits, and a
        # gate judging against half a design passes work that never met the
        # other half.
        return True, [
            f"{GATE_UNUSABLE} the verification gate for '{stage}' "
            f"passed without being shown all of "
            f"{'the evidence' if fit.evidence_cut else 'what it was meant to deliver'}"
        ]
    if missing and fit.evidence_cut:
        # Judged on an excerpt. Go and look at the rest before failing a stage
        # -- this is exactly how a real implementation was reported missing.
        try:
            missing, covered = _resolve_over_remaining(
                stage=stage,
                kind="evidence",
                outstanding=missing,
                material=evidence,
                start=fit.evidence_shown,
                window=fit.evidence_budget,
                chat=(chat or _gate_chat),
                deadline=deadline,
            )
        except Exception as error:  # noqa: BLE001 — an unreachable re-check is
            # an unusable gate, not a verdict against the work.
            return True, [
                f"{GATE_UNUSABLE} the verification gate for '{stage}' could not finish reading the evidence: {error}"
            ]
        if not missing:
            return True, []
        if not covered:
            # Never a PASS: uncertainty must not advance the stage.
            return False, [
                f"{GATE_INCONCLUSIVE} the evidence is too large to review "
                "in full and these remain unaccounted for: " + "; ".join(missing)
            ]
    # Gaps that survived a full read are a real failure, whatever the
    # first-pass boolean said.
    return (False, missing) if missing else (ok, [])


def parse_stage_marker(text: str, *, stage: str) -> dict[str, Any] | None:
    """Read the stage completion marker off the END of a session's final text.

    Only the last non-blank line counts.  A marker quoted mid-message -- an
    agent reciting its instructions, pasting a previous stage's output, or
    reasoning aloud about what it will print -- is not a verdict, and treating
    it as one would close a stage the agent never finished.

    Returns ``None`` when the agent did not end on one for *stage*; the driver
    treats that as a failure rather than guessing.
    """
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    match = STAGE_MARKER_RE.match(lines[-1])
    marker_stage = canonical_stage(match.group("stage").strip().lower()) if match else ""
    wanted_stage = canonical_stage(stage.strip().lower())
    if not match or marker_stage != wanted_stage:
        return None
    return {
        "stage": wanted_stage,
        "status": match.group("status").strip().upper(),
        "note": (match.group("note") or "").strip(),
    }


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

_CREATE = """
CREATE TABLE IF NOT EXISTS dev_sdlc_runs (
    run_id TEXT PRIMARY KEY,
    loop_id TEXT NOT NULL DEFAULT '',
    repo TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'feature',
    agent TEXT NOT NULL DEFAULT 'build',
    model TEXT NOT NULL DEFAULT '',
    validation_agent TEXT NOT NULL DEFAULT '',
    validation_model TEXT NOT NULL DEFAULT '',
    generation_agent TEXT NOT NULL DEFAULT '',
    generation_model TEXT NOT NULL DEFAULT '',
    verification_agent TEXT NOT NULL DEFAULT '',
    verification_model TEXT NOT NULL DEFAULT '',
    pipeline_revision INTEGER NOT NULL DEFAULT 3,
    headline TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'intake',
    stage TEXT NOT NULL DEFAULT '',
    stage_status TEXT NOT NULL DEFAULT '',
    transcript_json TEXT NOT NULL DEFAULT '[]',
    questions_json TEXT NOT NULL DEFAULT '[]',
    requirements_json TEXT NOT NULL DEFAULT '[]',
    criteria_json TEXT NOT NULL DEFAULT '[]',
    summary TEXT NOT NULL DEFAULT '',
    issue_number INTEGER,
    issue_url TEXT NOT NULL DEFAULT '',
    branch TEXT NOT NULL DEFAULT '',
    worktree TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    conductor_session_id TEXT NOT NULL DEFAULT '',
    author_session_id TEXT NOT NULL DEFAULT '',
    stage_dispatch_turn TEXT NOT NULL DEFAULT '',
    stage_executor_model TEXT NOT NULL DEFAULT '',
    deploy_armed_token TEXT NOT NULL DEFAULT '',
    deploy_armed_until TEXT NOT NULL DEFAULT '',
    review_session_id TEXT NOT NULL DEFAULT '',
    pr_number INTEGER,
    pr_url TEXT NOT NULL DEFAULT '',
    merged INTEGER NOT NULL DEFAULT 0,
    publish_mode TEXT NOT NULL DEFAULT 'merge',
    test_isolation TEXT NOT NULL DEFAULT 'docker',
    deploy_target TEXT NOT NULL DEFAULT '',
    deploy_status TEXT NOT NULL DEFAULT '',
    deploy_verified TEXT NOT NULL DEFAULT '',
    deploy_started_at TEXT NOT NULL DEFAULT '',
    validated_sha TEXT NOT NULL DEFAULT '',
    blocked_stage TEXT NOT NULL DEFAULT '',
    driver_lease_at TEXT NOT NULL DEFAULT '',
    driver_lease_token TEXT NOT NULL DEFAULT '',
    auto_resume_count INTEGER NOT NULL DEFAULT 0,
    gate_unusable_count INTEGER NOT NULL DEFAULT 0,
    manual_resume_count INTEGER NOT NULL DEFAULT 0,
    auto_resume_at TEXT NOT NULL DEFAULT '',
    -- P5: consecutive no-progress failures of the blocked stage. Reset on any
    -- stage pass and by an operator resume that raises the stage budget.
    no_progress_attempts INTEGER NOT NULL DEFAULT 0,
    -- P1/P5: the operator's explicit stage-budget override (seconds); 0 means
    -- the engine defaults apply.
    first_artifact_budget_s INTEGER NOT NULL DEFAULT 0,
    stage_ceiling_s INTEGER NOT NULL DEFAULT 0,
    stage_tokens INTEGER NOT NULL DEFAULT 0,
    -- P3: the previous attempt's digest (JSON), handed to the next attempt.
    rework_digest TEXT NOT NULL DEFAULT '',
    reviewed_sha TEXT NOT NULL DEFAULT '',
    merged_head_sha TEXT NOT NULL DEFAULT '',
    lap INTEGER NOT NULL DEFAULT 0,
    max_laps INTEGER NOT NULL DEFAULT 3,
    last_error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS dev_sdlc_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    tag TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    lap INTEGER NOT NULL DEFAULT 0,
    comment_id INTEGER,
    -- Which gate, and over what content. AUDIT ONLY: these were policy inputs
    -- to a settle mechanism that was removed for laundering marks, and nothing
    -- reads them to decide anything now. They are kept because "which gate
    -- failed, on which commit" is what a human needs when reading the log.
    gate_kind TEXT NOT NULL DEFAULT '',
    gate_digest TEXT NOT NULL DEFAULT '',
    executor_agent TEXT NOT NULL DEFAULT '',
    executor_model TEXT NOT NULL DEFAULT '',
    executor_session_id TEXT NOT NULL DEFAULT '',
    -- P6: did the driver OBSERVE the worker go idle after cancelling it?
    -- NULL means no cancel was involved; 0 means a cancel was issued and its
    -- effect could never be confirmed (the F4 token burn).
    cancellation_confirmed INTEGER,
    cancellation_idle_at TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dev_sdlc_events_run ON dev_sdlc_events(run_id, id);
-- One branch, one live run. A partial unique index makes the claim atomic
-- rather than a check two concurrent ratifications can both pass. `blocked` is
-- deliberately INSIDE the claim: a blocked run is resumable and still owns its
-- worktree.
CREATE UNIQUE INDEX IF NOT EXISTS idx_dev_sdlc_live_branch
    ON dev_sdlc_runs(repo, branch)
    WHERE branch != '' AND status NOT IN ('done', 'aborted');
"""

_LIST_FIELDS = (
    "run_id",
    "loop_id",
    "repo",
    "kind",
    "agent",
    "model",
    "headline",
    "detail",
    "status",
    "validation_agent",
    "validation_model",
    "generation_agent",
    "generation_model",
    "verification_agent",
    "verification_model",
    "pipeline_revision",
    "stage",
    "stage_status",
    "summary",
    "issue_number",
    "issue_url",
    "branch",
    "worktree",
    "session_id",
    "conductor_session_id",
    "author_session_id",
    "review_session_id",
    "stage_dispatch_turn",
    "stage_executor_model",
    "deploy_armed_until",
    "pr_number",
    "pr_url",
    "merged",
    "publish_mode",
    "test_isolation",
    "deploy_target",
    "deploy_status",
    "deploy_verified",
    "deploy_started_at",
    "reviewed_sha",
    "merged_head_sha",
    "auto_resume_count",
    "gate_unusable_count",
    "manual_resume_count",
    "auto_resume_at",
    "validated_sha",
    "blocked_stage",
    "no_progress_attempts",
    "first_artifact_budget_s",
    "stage_ceiling_s",
    "stage_tokens",
    "rework_digest",
    "driver_lease_at",
    "driver_lease_token",
    "lap",
    "max_laps",
    "last_error",
    "created_at",
    "updated_at",
    "archived_at",
)


def _default_run_headline(headline: str, detail: str = "") -> str:
    """Interim run name until the auto session title lands.

    The operator no longer types a title. Use the first detail line, or
    ``Dev``, so the row is never empty. The title agent names the session;
    ``sync_run_title_from_session`` copies that onto the run and the task.
    """
    text = str(headline or "").strip()
    if text:
        return text[:160]
    first = next((line.strip() for line in str(detail or "").splitlines() if line.strip()), "")
    return first[:80] or "Dev"


class SdlcStore:
    """SQLite store for intake transcripts and pipeline runs."""

    def __init__(self, db_path: Path | str | None = None):
        self.path = Path(db_path) if db_path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns to a store created by an earlier version.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op on an existing table, so a new
        field has to be added explicitly or every read of it returns nothing.
        """
        columns = {str(row[1]) for row in self._conn.execute("PRAGMA table_info(dev_sdlc_runs)")}
        for name in (
            "worktree",
            "model",
            "author_session_id",
            "review_session_id",
            "conductor_session_id",
            "validation_agent",
            "validation_model",
            "generation_agent",
            "generation_model",
            "verification_agent",
            "verification_model",
            "stage_dispatch_turn",
            "stage_executor_model",
            "deploy_armed_token",
            "deploy_armed_until",
            "reviewed_sha",
            "deploy_verified",
            "deploy_started_at",
            "auto_resume_at",
            "validated_sha",
            "blocked_stage",
            "driver_lease_at",
            "driver_lease_token",
            "merged_head_sha",
            "archived_at",
            "publish_mode",
            "test_isolation",
            "intake_claim_token",
            "intake_claim_until",
            "rework_digest",
        ):
            if name not in columns:
                self._add_column("dev_sdlc_runs", name, "TEXT NOT NULL DEFAULT ''")
        # Integer columns need their own DEFAULT: added with the TEXT default
        # above, every existing row would hold '' and arithmetic on it raises.
        # `CREATE TABLE IF NOT EXISTS` never alters a live table, and
        # production's database already has rows.
        ecols = {row[1] for row in self._conn.execute("PRAGMA table_info(dev_sdlc_events)")}
        for name in (
            "gate_kind",
            "gate_digest",
            "executor_agent",
            "executor_model",
            "executor_session_id",
            "cancellation_idle_at",
        ):
            if name not in ecols:
                self._add_column("dev_sdlc_events", name, "TEXT NOT NULL DEFAULT ''")
        for name in (
            "auto_resume_count",
            "gate_unusable_count",
            "manual_resume_count",
            "no_progress_attempts",
            "first_artifact_budget_s",
            "stage_ceiling_s",
            "stage_tokens",
            "cancellation_confirmed",
        ):
            if name not in columns:
                self._add_column("dev_sdlc_runs", name, "INTEGER NOT NULL DEFAULT 0")
        if "cancellation_confirmed" not in ecols:
            # NULL on historical rows: no cancel was ever involved with them.
            self._add_column("dev_sdlc_events", "cancellation_confirmed", "INTEGER")
        if "pipeline_revision" not in columns:
            # Existing rows start behind the current engine.  A brand-new
            # database gets the current default from _CREATE and never enters
            # this path.
            self._add_column("dev_sdlc_runs", "pipeline_revision", "INTEGER NOT NULL DEFAULT 1")

        self._upgrade_pipeline_revision()

        # `gate_unusable_settled` rows are from a settle mechanism that was
        # removed for laundering marks; they were real marks and are counted,
        # so a database written by that version re-blocks rather than reading
        # as clean.
        # The event log is the single source of truth for this blocker, so the
        # counter is DERIVED from it on every open rather than trusted. A run
        # that recorded events before the column existed would otherwise arrive
        # counted zero and merge freely, and any historical mismatch -- a count
        # of 2 against one event -- would block that run forever (Codex,
        # rounds 5 and 7).
        has_marks = self._conn.execute(
            "SELECT 1 FROM dev_sdlc_events WHERE status IN ('gate_unusable', 'gate_unusable_settled') LIMIT 1"
        ).fetchone()
        if has_marks:
            self._reconcile_gate_counts()

    def _upgrade_pipeline_revision(self) -> None:
        """Move legacy runs to this engine and refund one exhausted budget.

        A bounded supervisor correctly leaves a permanently failing run alone.
        But after the gate semantics and executors themselves are replaced,
        carrying that exhausted budget forward strands exactly the runs the
        migration is meant to repair.  The conditional revision update makes
        the refund happen once across any number of gunicorn workers.
        """
        rows = self._conn.execute(
            "SELECT run_id, stage, status FROM dev_sdlc_runs WHERE COALESCE(pipeline_revision, 1) < ?",
            (PIPELINE_REVISION,),
        ).fetchall()
        for row in rows:
            run_id = str(row["run_id"])
            unfinished = str(row["status"] or "") == "blocked"
            cur = self._conn.execute(
                "UPDATE dev_sdlc_runs SET pipeline_revision=?,"
                " auto_resume_count=CASE WHEN ? THEN 0 ELSE auto_resume_count END,"
                " auto_resume_at=CASE WHEN ? THEN '' ELSE auto_resume_at END"
                " WHERE run_id=? AND COALESCE(pipeline_revision, 1) < ?",
                (PIPELINE_REVISION, unfinished, unfinished, run_id, PIPELINE_REVISION),
            )
            if cur.rowcount == 1 and unfinished:
                self._conn.execute(
                    "INSERT INTO dev_sdlc_events"
                    " (run_id, stage, status, tag, note, lap, created_at)"
                    " VALUES (?, ?, 'pipeline_upgraded', '', ?, 0, ?)",
                    (
                        run_id,
                        canonical_stage(str(row["stage"] or "blocked")),
                        f"pipeline revision {PIPELINE_REVISION} restored one bounded recovery budget",
                        _now(),
                    ),
                )

    def _add_column(self, table: str, name: str, decl: str) -> None:
        """ALTER that tolerates another worker having just done it.

        Several workers open this database at boot and can all observe a column
        as absent before any of them adds it. The loser of that race would
        crash on "duplicate column name" -- a startup failure caused only by
        starting twice.
        """
        try:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError as error:
            if "duplicate column" not in str(error).lower():
                raise
            log.debug("sdlc: %s.%s already added by another worker", table, name)

    def _reconcile_gate_counts(self) -> None:
        """Make the counter agree with the log. The log is the truth."""
        self._conn.execute(
            "UPDATE dev_sdlc_runs SET gate_unusable_count = ("
            "  SELECT COUNT(*) FROM dev_sdlc_events"
            "   WHERE dev_sdlc_events.run_id = dev_sdlc_runs.run_id"
            "     AND dev_sdlc_events.status IN"
            "         ('gate_unusable', 'gate_unusable_settled'))"
            " WHERE gate_unusable_count != ("
            "  SELECT COUNT(*) FROM dev_sdlc_events"
            "   WHERE dev_sdlc_events.run_id = dev_sdlc_runs.run_id"
            "     AND dev_sdlc_events.status IN"
            "         ('gate_unusable', 'gate_unusable_settled'))"
        )

    # -- projection ------------------------------------------------------
    @staticmethod
    def _public(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        out = {key: data.get(key) for key in _LIST_FIELDS}
        out["merged"] = bool(data.get("merged"))
        out["archived_at"] = str(data.get("archived_at") or "")
        out["archived"] = bool(out["archived_at"].strip())
        out["transcript"] = _unjson(data.get("transcript_json"), [])
        out["questions"] = _unjson(data.get("questions_json"), [])
        out["requirements"] = _unjson(data.get("requirements_json"), [])
        out["criteria"] = _unjson(data.get("criteria_json"), [])
        out["stage"] = canonical_stage(str(out.get("stage") or ""))
        for role in ROLE_NAMES:
            agent, model = role_config(data, role)
            out[f"{role}_agent"] = agent
            out[f"{role}_model"] = model
        # Historical clients still read these fields. They remain aliases for
        # generation, never a fourth role.
        out["agent"] = out["generation_agent"]
        out["model"] = out["generation_model"]
        out["stages"] = list(STAGES)
        out["tags"] = dict(STAGE_TAGS)
        out["summary"] = data.get("summary") or ""
        return out

    # -- runs ------------------------------------------------------------
    def create(
        self,
        *,
        repo: str,
        kind: str,
        headline: str,
        detail: str = "",
        agent: str = "build",
        model: str = "",
        validation_agent: str = "",
        validation_model: str = "",
        generation_agent: str = "",
        generation_model: str = "",
        verification_agent: str = "",
        verification_model: str = "",
        max_laps: int = 3,
    ) -> dict[str, Any]:
        if not str(repo or "").strip():
            raise SdlcError("repo is required")
        headline = _default_run_headline(headline, detail)
        if not headline:
            raise SdlcError("object is required")
        defaults = default_role_agents(GENERATION_AGENT)
        roles = {
            "validation_agent": str(validation_agent or defaults["validation"]).strip(),
            "generation_agent": GENERATION_AGENT,
            "verification_agent": str(verification_agent or defaults["verification"]).strip(),
        }
        role_models = {
            "validation_model": str(validation_model or "").strip(),
            "generation_model": str(generation_model or model or "").strip(),
            "verification_model": str(verification_model or "").strip(),
        }
        validate_role_agents({**roles, **role_models})
        run_id = uuid.uuid4().hex[:12]
        now = _now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO dev_sdlc_runs
                   (run_id, repo, kind, agent, model,
                    validation_agent, validation_model, generation_agent,
                    generation_model, verification_agent, verification_model,
                    headline, detail, status, max_laps, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'intake',?,?,?)""",
                (
                    run_id,
                    repo.strip(),
                    "fix" if str(kind).lower() == "fix" else "feature",
                    roles["generation_agent"],
                    role_models["generation_model"] or default_model(roles["generation_agent"]),
                    roles["validation_agent"],
                    role_models["validation_model"] or default_model(roles["validation_agent"]),
                    roles["generation_agent"],
                    role_models["generation_model"] or default_model(roles["generation_agent"]),
                    roles["verification_agent"],
                    role_models["verification_model"] or default_model(roles["verification_agent"]),
                    headline.strip(),
                    (detail or "").strip(),
                    max(1, int(max_laps)),
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return self.get(run_id)  # type: ignore[return-value]

    def delete(self, run_id: str) -> bool:
        """Remove the run and its events. The forge issue / PR stay."""
        rid = str(run_id or "").strip()
        if not rid:
            return False
        with self._lock:
            self._conn.execute("DELETE FROM dev_sdlc_events WHERE run_id=?", (rid,))
            cur = self._conn.execute("DELETE FROM dev_sdlc_runs WHERE run_id=?", (rid,))
            self._conn.commit()
            gone = cur.rowcount == 1
        if gone:
            try:
                base = attachment_dir(rid)
            except Exception:  # noqa: BLE001 — missing artifacts root must not keep the row
                return True
            if base.is_dir():
                shutil.rmtree(base, ignore_errors=True)
        return gone

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM dev_sdlc_runs WHERE run_id=?", (run_id,)).fetchone()
        return self._public(row) if row else None

    def list(self, *, repo: str = "", limit: int = 100, include_archived: bool = True) -> list[dict[str, Any]]:
        # Failed and finished runs stay until the operator deletes them.
        # include_archived is kept for callers; hiding is no longer the product.
        hidden = ""
        with self._lock:
            if repo:
                rows = self._conn.execute(
                    "SELECT * FROM dev_sdlc_runs WHERE repo=?" + hidden + " ORDER BY updated_at DESC LIMIT ?",
                    (repo, int(limit)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM dev_sdlc_runs WHERE 1=1" + hidden + " ORDER BY updated_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
        return [self._public(row) for row in rows]

    def conductor_run_meta(self, session_ids: list[str]) -> dict[str, dict[str, str]]:
        """Newest run id+status for each Jarvis conductor (live first)."""
        ids = [str(sid).strip() for sid in session_ids if str(sid or "").strip()]
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                "SELECT conductor_session_id, run_id, status FROM dev_sdlc_runs"
                f" WHERE conductor_session_id IN ({placeholders})"
                " ORDER BY CASE WHEN status IN ('done', 'aborted') THEN 1 ELSE 0 END,"
                " updated_at DESC",
                ids,
            ).fetchall()
        out: dict[str, dict[str, str]] = {}
        for row in rows:
            sid = str(row["conductor_session_id"] or "")
            if sid and sid not in out:
                out[sid] = {
                    "run_id": str(row["run_id"] or ""),
                    "status": str(row["status"] or ""),
                }
        return out

    def conductor_run_ids(self, session_ids: list[str]) -> dict[str, str]:
        """Newest run_id for each Jarvis conductor session (live first)."""
        return {sid: meta["run_id"] for sid, meta in self.conductor_run_meta(session_ids).items() if meta.get("run_id")}

    def consume_deploy_arm(self, run_id: str, token: str) -> bool:
        """Spend the deploy arm, atomically. True only for the caller that won.

        A read-then-clear in the route let two concurrent deploys both validate
        the same token before either cleared it. The UPDATE names the token in
        its WHERE clause, so exactly one row-change happens however many callers
        race.
        """
        if not str(token or "").strip():
            return False
        with self._lock:
            cur = self._conn.execute(
                "UPDATE dev_sdlc_runs SET deploy_armed_token='', deploy_armed_until='',"
                " updated_at=? WHERE run_id=? AND deploy_armed_token=? AND deploy_armed_token!=''",
                (_now(), run_id, token),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def due_for_auto_resume(self, *, now: Any = None, max_attempts: int = 5) -> list[dict[str, Any]]:
        """Runs the supervisor may restart unattended, right now.

        A run is due when it is BLOCKED, its backoff has expired, and it has
        not already been auto-resumed ``max_attempts`` times.

        The cap is what stops a permanently-broken run from spinning: autonomy
        means recovering from a transient failure, not retrying a wrong answer
        forever. The counter resets whenever a stage passes, so a long healthy
        run is never penalised for a hiccup twenty stages ago.

        P5 adds a sharper stop: a run whose blocked stage failed
        ``MAX_NO_PROGRESS_ATTEMPTS`` times with nothing to hand over is not a
        transient failure — it is a standing policy problem (the budget, the
        scope, or the builder). Restarting it again spends the same tokens on
        the same dead end, so it stays blocked until an operator either raises
        the budget or aborts.
        """
        stamp = str(now) if now is not None else _now()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM dev_sdlc_runs"
                " WHERE issue_number IS NOT NULL"
                "   AND status = 'blocked'"
                "   AND COALESCE(archived_at,'') = ''"
                "   AND COALESCE(last_error,'') NOT LIKE '%human input is prohibited%'"
                "   AND COALESCE(no_progress_attempts, 0) < ?"
                "   AND COALESCE(auto_resume_count, 0) < ?"
                "   AND (auto_resume_at = '' OR auto_resume_at <= ?)"
                " ORDER BY updated_at ASC",
                (int(MAX_NO_PROGRESS_ATTEMPTS), int(max_attempts), stamp),
            ).fetchall()
        return [self._public(r) for r in rows]

    def claim_auto_resume(self, run_id: str, *, expected_count: int, backoff_seconds: int) -> bool:
        """Claim one supervised restart. True only for the winner.

        The claim is the DATABASE's, not a process lock: with more than one
        gunicorn worker each has its own lock and its own supervisor, so a
        process-local claim lets two workers restart the same run and spend two
        attempts on it. The conditional UPDATE decides — exactly one caller can
        move the count from `expected_count` to `expected_count + 1`.

        It COMMITS before returning, so a crash between claiming and launching
        cannot roll the attempt back and hand the run a fresh budget on the next
        boot. Losing an attempt to a crash is the safe direction; losing the
        record of it is not.
        """
        nxt = (datetime.now(timezone.utc) + timedelta(seconds=int(backoff_seconds))).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE dev_sdlc_runs"
                " SET auto_resume_count = COALESCE(auto_resume_count, 0) + 1,"
                "     auto_resume_at = ?, updated_at = ?"
                " WHERE run_id = ? AND COALESCE(auto_resume_count, 0) = ?",
                (nxt, _now(), run_id, int(expected_count)),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def note_auto_resume(self, run_id: str, *, backoff_seconds: int) -> None:
        """Unconditional variant, for tests and manual bookkeeping."""
        nxt = (datetime.now(timezone.utc) + timedelta(seconds=int(backoff_seconds))).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE dev_sdlc_runs"
                " SET auto_resume_count = COALESCE(auto_resume_count, 0) + 1,"
                "     auto_resume_at = ?, updated_at = ?"
                " WHERE run_id = ?",
                (nxt, _now(), run_id),
            )
            self._conn.commit()

    def operator_resume(self, run_id: str, *, actor: str = "") -> dict[str, Any]:
        """Record a manual resume, and restore the recovery budget if allowed.

        An operator resuming is a human deciding this run can go on, which is
        the condition the unattended cap exists to protect against being
        absent. But the ROUTE cannot prove a human: any authenticated client
        can POST it in a loop, and resetting unconditionally would make
        recovery unbounded through the front door.

        So the lift is budgeted -- and the budget is spent only while the run
        is NOT progressing, because ``clear_auto_resume`` (progress) resets it
        too. "Ten manual resumes without getting anywhere", not "ten ever":
        a permanent per-run limit would make a run that had been legitimately
        helped ten times human-dependent for the rest of its life.

        The whole thing is ONE transaction: read-then-write under a per-process
        lock is not atomic across gunicorn workers, which have their own locks
        and their own connections. Two workers could both see the count below
        the limit, both refund, and lose an increment between them; a
        concurrent ``claim_auto_resume`` could commit into the gap. The
        conditional UPDATE and the event are now decided and written together.

        Returns ``{"lifted": bool, "manual": int, "previous": int}``.
        """
        now = _now()
        tag = STAGE_TAGS.get("resume", "sdlc/resume")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT auto_resume_count, manual_resume_count  FROM dev_sdlc_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return {"lifted": False, "manual": 0, "previous": 0}
                previous = int(row["auto_resume_count"] or 0)
                manual = int(row["manual_resume_count"] or 0) + 1
                lifted = manual <= MANUAL_RESUME_MAX
                if lifted:
                    self._conn.execute(
                        "UPDATE dev_sdlc_runs SET auto_resume_count = 0,"
                        "       auto_resume_at = '', manual_resume_count = ?,"
                        "       updated_at = ? WHERE run_id = ?",
                        (manual, now, run_id),
                    )
                else:
                    self._conn.execute(
                        "UPDATE dev_sdlc_runs SET manual_resume_count = ?,       updated_at = ? WHERE run_id = ?",
                        (manual, now, run_id),
                    )
                note = f"{actor or 'operator'} resumed this run; " + (
                    f"recovery budget restored (was {previous} used)"
                    if lifted
                    else f"budget NOT restored: {manual} manual resumes "
                    f"without progress exceeds the limit of "
                    f"{MANUAL_RESUME_MAX}"
                )
                # In the SAME transaction: a reset whose record is written
                # afterwards can be left silent by a crash between them.
                self._conn.execute(
                    """INSERT INTO dev_sdlc_events
                       (run_id, stage, status, tag, note, lap, comment_id,
                        gate_kind, gate_digest, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (run_id, "resume", "manual_resume", tag, note, 0, None, "", "", now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return {"lifted": lifted, "manual": manual, "previous": previous}

    def clear_auto_resume(self, run_id: str) -> None:
        """Progress happened: give the run its full recovery budget back.

        Both budgets. The manual-lift limit exists to stop a caller looping on
        a run that is getting nowhere; a run that HAS got somewhere is not that
        case, and a limit that never resets would leave a legitimately
        much-helped run permanently human-dependent.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE dev_sdlc_runs SET auto_resume_count = 0,"
                "       auto_resume_at = '', manual_resume_count = 0,"
                "       updated_at = ?"
                " WHERE run_id = ?",
                (_now(), run_id),
            )
            self._conn.commit()

    def claim_driver_lease(self, run_id: str, *, lease_seconds: int = 7200, token: str = "") -> tuple[bool, str]:
        """Claim the right to advance *run_id*. True only for the winner.

        `run_is_active()` is process-local, so with several gunicorn workers each
        one thinks no driver is running and they all restart the same abandoned
        run — dispatching the same stage concurrently and spending the agent
        turns twice. The lease is the DATABASE's: a conditional UPDATE that only
        succeeds when nobody holds it or the previous holder's lease has expired.

        Expiry matters as much as the claim. A worker that dies holding a lease
        must not lock the run out forever, so the lease is a deadline rather
        than a flag. Under systemd, every worker in one service invocation
        shares ``INVOCATION_ID``. A replacement invocation may therefore take
        a lease from the service instance that was just stopped immediately;
        sibling workers in the same live invocation still cannot steal it.
        """
        now = datetime.now(timezone.utc)
        until = (now + timedelta(seconds=int(lease_seconds))).isoformat()
        invocation = service_instance_id()
        tok = token or ((invocation + ":") if invocation else "") + uuid.uuid4().hex[:12]
        invocation_pattern = f"{invocation}:%" if invocation else ""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE dev_sdlc_runs"
                " SET driver_lease_at = ?, driver_lease_token = ?, updated_at = ?"
                " WHERE run_id = ?"
                "   AND (driver_lease_at = '' OR driver_lease_at <= ?"
                "        OR (? != '' AND driver_lease_token NOT LIKE ?))",
                (until, tok, _now(), run_id, now.isoformat(), invocation, invocation_pattern),
            )
            self._conn.commit()
            return (cur.rowcount == 1), tok

    def renew_driver_lease(self, run_id: str, token: str, *, lease_seconds: int = 7200) -> bool:
        """Extend a lease this worker still holds. False if it lost it.

        A stage can run far longer than any sane lease, so the holder renews
        rather than the lease being sized for the worst case.
        """
        until = (datetime.now(timezone.utc) + timedelta(seconds=int(lease_seconds))).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE dev_sdlc_runs SET driver_lease_at = ?, updated_at = ?"
                " WHERE run_id = ? AND driver_lease_token = ?",
                (until, _now(), run_id, token),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def release_driver_lease(self, run_id: str, token: str = "") -> None:
        """Give the lease back — but only if this worker still holds it.

        An unconditional release let a worker whose lease had already expired
        and been taken over clear the NEW holder's lease on its way out, so two
        drivers could end up advancing one run.
        """
        if not token:
            # No token, no release. An unconditional path defeats the whole
            # guarantee: it is exactly how a stale worker cleared the new
            # holder's lease. An expired lease is reclaimed by `claim`, so
            # nothing needs a force-release.
            return
        with self._lock:
            self._conn.execute(
                "UPDATE dev_sdlc_runs SET driver_lease_at = '',"
                " driver_lease_token = ''"
                " WHERE run_id = ? AND driver_lease_token = ?",
                (run_id, token),
            )
            self._conn.commit()

    def resumable(self) -> list[dict[str, Any]]:
        """Runs that have an issue and have not finished.

        The driver is a daemon thread: it dies with the web process, leaving a
        `running` row with nobody advancing it.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM dev_sdlc_runs WHERE issue_number IS NOT NULL"
                " AND status NOT IN ('done','aborted','intake','ready')"
                " AND COALESCE(archived_at,'') = ''"
                " ORDER BY updated_at DESC"
            ).fetchall()
        return [self._public(row) for row in rows]

    def claim_branch(self, run_id: str, branch: str) -> dict[str, Any] | None:
        """Take the (repo, branch) claim for this run, or raise.

        The uniqueness is enforced by a partial index, so two ratifications
        racing for the same branch cannot both win the way a read-then-write
        check allows.
        """
        try:
            return self.patch(run_id, branch=branch)
        except sqlite3.IntegrityError as error:
            raise SdlcError(f"another live run already owns `{branch}`") from error

    def _set_clause(self, fields: dict[str, Any]) -> tuple[list[str], list[Any]]:
        allowed = {
            "loop_id",
            "status",
            "stage",
            "stage_status",
            "summary",
            "issue_number",
            "issue_url",
            "branch",
            "worktree",
            "session_id",
            "conductor_session_id",
            "author_session_id",
            "stage_executor_model",
            "review_session_id",
            "stage_dispatch_turn",
            "deploy_armed_token",
            "deploy_armed_until",
            "pr_number",
            "pr_url",
            "merged",
            "publish_mode",
            "test_isolation",
            "deploy_target",
            "deploy_status",
            "deploy_verified",
            "deploy_started_at",
            "reviewed_sha",
            "merged_head_sha",
            # gate_unusable_count is DERIVED from the event log and is
            # deliberately NOT writable: a store invariant that any caller can
            # overwrite is not an invariant.
            "auto_resume_count",
            "auto_resume_at",
            "validated_sha",
            "blocked_stage",
            "driver_lease_at",
            "driver_lease_token",
            "first_artifact_budget_s",
            "stage_ceiling_s",
            "rework_digest",
            "no_progress_attempts",
            "stage_tokens",
            "lap",
            "max_laps",
            "last_error",
            "agent",
            "model",
            "headline",
            "detail",
            "validation_agent",
            "validation_model",
            "generation_agent",
            "generation_model",
            "verification_agent",
            "verification_model",
            "archived_at",
        }
        json_fields = {
            "transcript": "transcript_json",
            "questions": "questions_json",
            "requirements": "requirements_json",
            "criteria": "criteria_json",
        }
        sets, args = [], []
        for key, value in fields.items():
            if key in json_fields:
                sets.append(f"{json_fields[key]}=?")
                args.append(_json(value))
            elif key in allowed:
                sets.append(f"{key}=?")
                args.append(int(value) if key == "merged" else value)
        return sets, args

    def patch(self, run_id: str, **fields: Any) -> dict[str, Any] | None:
        sets, args = self._set_clause(fields)
        if not sets:
            return self.get(run_id)
        sets.append("updated_at=?")
        args += [_now(), run_id]
        with self._lock:
            self._conn.execute(f"UPDATE dev_sdlc_runs SET {', '.join(sets)} WHERE run_id=?", args)
            self._conn.commit()
        return self.get(run_id)

    def patch_intake(
        self, run_id: str, *, expect_transcript: Iterable[dict[str, Any]], **fields: Any
    ) -> dict[str, Any] | None:
        """Write to the interview only if it is the interview the caller read.

        Used by every intake writer: a synthesis result, and an appended
        answer (which passes the new transcript in *fields* and the one it
        was appended to as *expect_transcript*).

        A synthesis is a model call that takes minutes; the operator can answer
        again or ratify while it runs. Re-reading before the write narrows that
        window, it does not close it: between the re-read and the UPDATE a
        ratification can still land, and the stale result would then overwrite
        frozen requirements and drop the run out of ``running``. The predicate
        is therefore on the row itself, in one statement, across every worker:
        no issue yet, still an interview, and the transcript is byte-for-byte
        the one the synthesis read -- and nobody holds the intake claim, so an
        answer cannot slip in between ratify's re-read and its issue (the
        issue body would then describe an older transcript than the row).
        ``None`` means the write lost and the caller must discard its result.
        """
        sets, args = self._set_clause(fields)
        if not sets:
            return self.get(run_id)
        sets.append("updated_at=?")
        args += [_now(), run_id, _json(list(expect_transcript)), _now()]
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE dev_sdlc_runs SET {', '.join(sets)}"
                " WHERE run_id=? AND issue_number IS NULL"
                " AND status IN ('intake', 'ready') AND transcript_json=?"
                " AND (intake_claim_token='' OR intake_claim_until='' OR intake_claim_until<?)",
                args,
            )
            self._conn.commit()
            won = cur.rowcount == 1
        return self.get(run_id) if won else None

    def claim_intake(self, run_id: str, token: str, seconds: float = 30.0) -> bool:
        """Take the short exclusive claim that guards stopping the conductor.

        Every pipeline stage is dispatched as a goal step on the conductor
        session, so a force cancel that lands just after ratification kills
        the stage that was just started. Ratify holds this claim while it
        mints the issue; a synthesis takes it only around its cancel. The
        predicate refuses a run that already has an issue, so the claim can
        never be taken on a frozen contract at all. Expiry keeps a crashed
        holder from wedging the run.
        """
        now = _now()
        until = (datetime.now(timezone.utc) + timedelta(seconds=max(1.0, float(seconds)))).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE dev_sdlc_runs SET intake_claim_token=?, intake_claim_until=?"
                " WHERE run_id=? AND issue_number IS NULL"
                " AND status IN ('intake', 'ready')"
                " AND (intake_claim_token='' OR intake_claim_token=?"
                "      OR intake_claim_until='' OR intake_claim_until<?)",
                (token, until, run_id, token, now),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def release_intake(self, run_id: str, token: str) -> None:
        """Give the claim back; only the holder can."""
        with self._lock:
            self._conn.execute(
                "UPDATE dev_sdlc_runs SET intake_claim_token='', intake_claim_until=''"
                " WHERE run_id=? AND intake_claim_token=?",
                (run_id, token),
            )
            self._conn.commit()

    # -- events ----------------------------------------------------------
    def add_event(
        self,
        run_id: str,
        *,
        stage: str,
        status: str,
        note: str = "",
        lap: int = 0,
        comment_id: int | None = None,
        executor_agent: str = "",
        executor_model: str = "",
        executor_session_id: str = "",
        cancellation_confirmed: bool | None = None,
        cancellation_idle_at: str = "",
    ) -> dict[str, Any]:
        if status == "gate_unusable":
            # This event carries a release blocker, and the two must be written
            # together. Going through the generic writer would leave the count
            # to be repaired at the next store open.
            self.record_gate_unusable(run_id, stage=stage, kind="", digest="", notes=[note], lap=lap)
            return {
                "run_id": run_id,
                "stage": stage,
                "status": status,
                "tag": STAGE_TAGS.get(stage, f"sdlc/{stage}"),
                "note": note,
                "lap": lap,
                "comment_id": comment_id,
                "created_at": _now(),
            }
        now = _now()
        tag = STAGE_TAGS.get(stage, f"sdlc/{stage}")
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO dev_sdlc_events
                   (run_id, stage, status, tag, note, lap, comment_id,
                    executor_agent, executor_model, executor_session_id,
                    cancellation_confirmed, cancellation_idle_at, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    stage,
                    status,
                    tag,
                    note,
                    int(lap),
                    comment_id,
                    str(executor_agent or ""),
                    str(executor_model or ""),
                    str(executor_session_id or ""),
                    (None if cancellation_confirmed is None else int(bool(cancellation_confirmed))),
                    str(cancellation_idle_at or ""),
                    now,
                ),
            )
            self._conn.commit()
            event_id = int(cur.lastrowid or 0)
        return {
            "id": event_id,
            "run_id": run_id,
            "stage": stage,
            "status": status,
            "tag": tag,
            "note": note,
            "lap": lap,
            "comment_id": comment_id,
            "executor_agent": str(executor_agent or ""),
            "executor_model": str(executor_model or ""),
            "executor_session_id": str(executor_session_id or ""),
            "created_at": now,
        }

    def record_gate_unusable(
        self,
        run_id: str,
        *,
        stage: str,
        kind: str,
        digest: str,
        notes: list[str],
        lap: int = 0,
        executor_agent: str = "",
        executor_model: str = "",
        executor_session_id: str = "",
    ) -> int:
        """Record undecided gate answers AND their blocker in one transaction.

        One event per note, so the count and the log cannot disagree -- passing
        a count of 2 while writing one event left a counter nothing could ever
        settle.

        Committing the events first and patching the counter after leaves a
        window where a crash keeps the visible record but loses the thing that
        actually blocks a merge. The blocker is the part that must not be lost.
        """
        notes = [str(n) for n in (notes or []) if str(n).strip()]
        if not notes or not run_id:
            return 0
        now = _now()
        tag = STAGE_TAGS.get(stage, f"sdlc/{stage}")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for note in notes:
                    self._conn.execute(
                        """INSERT INTO dev_sdlc_events
                           (run_id, stage, status, tag, note, lap, comment_id,
                            gate_kind, gate_digest, executor_agent,
                            executor_model, executor_session_id, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            run_id,
                            stage,
                            "gate_unusable",
                            tag,
                            note,
                            int(lap),
                            None,
                            str(kind or ""),
                            str(digest or ""),
                            str(executor_agent or ""),
                            str(executor_model or ""),
                            str(executor_session_id or ""),
                            now,
                        ),
                    )
                self._conn.execute(
                    "UPDATE dev_sdlc_runs"
                    "   SET gate_unusable_count = COALESCE(gate_unusable_count, 0) + ?,"
                    "       updated_at = ?"
                    " WHERE run_id = ?",
                    (len(notes), now, run_id),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return len(notes)

    def record_gate_decided(
        self,
        run_id: str,
        *,
        stage: str,
        kind: str,
        digest: str,
        executor_agent: str = "",
        executor_model: str = "",
        executor_session_id: str = "",
    ) -> None:
        """A gate ANSWERED, about this exact content.

        Nothing recorded this, so nothing downstream could ever tell whether a
        stage marked unreviewed had since been reviewed properly. That left one
        transient defect blocking a run for its whole life, with a human
        override as the only way past -- and an override is a rubber stamp when
        there is no evidence to weigh.

        This never clears a mark. Marks are permanent, and every mechanism for
        clearing one turned out to launder an unreviewed stage. It is a
        SEPARATE fact, and the release gate compares the two.
        """
        if not run_id or not kind or not digest:
            return
        # Insert-then-annotate-the-max-id let two writers attach one writer's
        # kind and digest to the other's event, and `unreviewed_stages` trusts
        # those fields -- a false exact match. One statement, one row.
        now = _now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO dev_sdlc_events
                   (run_id, stage, status, tag, note, lap, comment_id,
                    gate_kind, gate_digest, executor_agent, executor_model,
                    executor_session_id, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    stage,
                    "gate_decided",
                    STAGE_TAGS.get(stage, f"sdlc/{stage}"),
                    f"the {kind} gate decided about {digest[:12]}",
                    0,
                    None,
                    str(kind),
                    str(digest),
                    str(executor_agent or ""),
                    str(executor_model or ""),
                    str(executor_session_id or ""),
                    now,
                ),
            )
            self._conn.commit()

    def record_gate_session_link(
        self, run_id: str, *, stage: str, kind: str, session_id: str, executor_agent: str = "", executor_model: str = ""
    ) -> bool:
        """Append recovered session provenance without rewriting a verdict.

        Releases before ``executor_session_id`` preserved gate conversations
        in the gateway but not in this ledger. Recovery is an audit fact of its
        own: mutating the old ``gate_decided`` row would make history look as if
        the id had always been captured. Idempotence is transactional because
        more than one Web worker may notice the same legacy row.
        """
        if not run_id or kind not in {"validation", "verification"} or not session_id:
            return False
        now = _now()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                exists = self._conn.execute(
                    "SELECT 1 FROM dev_sdlc_events"
                    " WHERE run_id=? AND stage=? AND status='gate_session_linked'"
                    " AND gate_kind=? AND executor_session_id=? LIMIT 1",
                    (run_id, stage, kind, session_id),
                ).fetchone()
                if exists:
                    self._conn.rollback()
                    return False
                self._conn.execute(
                    """INSERT INTO dev_sdlc_events
                       (run_id, stage, status, tag, note, lap, comment_id,
                        gate_kind, gate_digest, executor_agent, executor_model,
                        executor_session_id, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id,
                        stage,
                        "gate_session_linked",
                        STAGE_TAGS.get(stage, f"sdlc/{stage}"),
                        f"recovered the legacy {kind} session link",
                        0,
                        None,
                        kind,
                        "",
                        str(executor_agent or ""),
                        str(executor_model or ""),
                        session_id,
                        now,
                    ),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def record_gate_session_started(
        self, run_id: str, *, stage: str, kind: str, session_id: str, executor_agent: str = "", executor_model: str = ""
    ) -> bool:
        """Persist a gate chat as soon as it starts.

        A completed ``gate_decided`` event is too late for the Dev tab to show
        which validator or verifier is working right now.  This provenance-only
        event gives the UI the concrete link while the turn is live; it carries
        no verdict and therefore cannot satisfy a validation/release gate.
        """
        if not run_id or kind not in {"validation", "verification"} or not session_id:
            return False
        now = _now()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                exists = self._conn.execute(
                    "SELECT 1 FROM dev_sdlc_events"
                    " WHERE run_id=? AND stage=? AND status='gate_session_started'"
                    " AND gate_kind=? AND executor_session_id=? LIMIT 1",
                    (run_id, stage, kind, session_id),
                ).fetchone()
                if exists:
                    self._conn.rollback()
                    return False
                self._conn.execute(
                    """INSERT INTO dev_sdlc_events
                       (run_id, stage, status, tag, note, lap, comment_id,
                        gate_kind, gate_digest, executor_agent, executor_model,
                        executor_session_id, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id,
                        stage,
                        "gate_session_started",
                        STAGE_TAGS.get(stage, f"sdlc/{stage}"),
                        f"the {kind} session started",
                        0,
                        None,
                        kind,
                        "",
                        str(executor_agent or ""),
                        str(executor_model or ""),
                        session_id,
                        now,
                    ),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def record_gate_session_abandoned(
        self, run_id: str, *, stage: str, kind: str, session_id: str, note: str = ""
    ) -> bool:
        """Close live provenance when a gate turn cannot produce a decision.

        The event is deliberately separate from the immutable started row.  A
        later projection can therefore distinguish a live gate from an
        abandoned dispatch without rewriting history or treating either row as
        a gate verdict.
        """
        if not run_id or kind not in {"validation", "verification"} or not session_id:
            return False
        now = _now()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                exists = self._conn.execute(
                    "SELECT 1 FROM dev_sdlc_events"
                    " WHERE run_id=? AND stage=? AND status='gate_session_abandoned'"
                    " AND gate_kind=? AND executor_session_id=? LIMIT 1",
                    (run_id, stage, kind, session_id),
                ).fetchone()
                if exists:
                    self._conn.rollback()
                    return False
                self._conn.execute(
                    """INSERT INTO dev_sdlc_events
                       (run_id, stage, status, tag, note, lap, comment_id,
                        gate_kind, gate_digest, executor_agent, executor_model,
                        executor_session_id, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id,
                        stage,
                        "gate_session_abandoned",
                        STAGE_TAGS.get(stage, f"sdlc/{stage}"),
                        str(note or f"the {kind} session ended without a decision"),
                        0,
                        None,
                        kind,
                        "",
                        "",
                        "",
                        session_id,
                        now,
                    ),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def mark_gate_session_recovery_checked(self, run_id: str) -> bool:
        """Record one generic completion marker across concurrent Web workers."""
        if not run_id:
            return False
        now = _now()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                exists = self._conn.execute(
                    "SELECT 1 FROM dev_sdlc_events WHERE run_id=? AND status='gate_session_recovery_checked' LIMIT 1",
                    (run_id,),
                ).fetchone()
                if exists:
                    self._conn.rollback()
                    return False
                self._conn.execute(
                    """INSERT INTO dev_sdlc_events
                       (run_id, stage, status, tag, note, lap, comment_id,
                        gate_kind, gate_digest, executor_agent, executor_model,
                        executor_session_id, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id,
                        "migration",
                        "gate_session_recovery_checked",
                        "sdlc/migration",
                        "legacy gate session recovery checked",
                        0,
                        None,
                        "",
                        "",
                        "",
                        "",
                        "",
                        now,
                    ),
                )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM dev_sdlc_events WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def default_db_path() -> Path:
    return Path(sharedagents_root()) / "web" / "dev_sdlc.db"


_store_lock = threading.Lock()
_store: SdlcStore | None = None


def get_sdlc_store() -> SdlcStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = SdlcStore()
        return _store


def reset_sdlc_store_for_tests(store: SdlcStore | None) -> None:
    """Point the module singleton at a throwaway store (tests only)."""
    global _store
    with _store_lock:
        _store = store


# --------------------------------------------------------------------------
# Worktree provisioning
# --------------------------------------------------------------------------


class WorktreeError(SdlcError):
    """A run cannot start without an isolated checkout."""


def dev_src_root() -> Path:
    """Where local clones of forge repos live."""
    return Path(os.environ.get("SHAREDAGENTS_DEV_SRC_ROOT") or (Path.home() / "src"))


def repo_dir(repo: str) -> Path:
    """``owner/name`` -> the local clone directory."""
    name = str(repo or "").strip().strip("/").split("/")[-1]
    if not name:
        raise WorktreeError("repo is required")
    return dev_src_root() / name


def service_instance_id() -> str:
    """A stable id for THIS run of the service. Empty when it cannot be known.

    The driver lease lets a replacement service instance take a run over from
    the instance that was just stopped, while sibling workers inside the SAME
    instance still cannot steal from each other. That was keyed on systemd's
    ``INVOCATION_ID`` alone — so outside systemd (Docker, and this repo's own
    dev stack) there was no instance identity at all, and a container restart
    stranded every in-flight run behind a dead worker's lease for the full two
    hours. An operator resume answered "another worker already owns this run"
    against a worker that no longer existed.

    PID 1's start time is the portable equivalent: a new container has a new
    one, every process inside a container shares it, and on a bare host it
    changes only on reboot — which is the same conservative answer the lease
    wants there.
    """

    def _alnum(value: str) -> str:
        return "".join(char for char in value if char.isalnum())

    invocation = _alnum(str(os.environ.get("INVOCATION_ID") or ""))[:32]
    if invocation:
        return invocation
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8")
        # `/proc/1/stat` field 22 is starttime; the comm field can contain
        # spaces and parentheses, so split after the LAST ')'.
        stat = Path("/proc/1/stat").read_text(encoding="utf-8")
        started = stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return ""
    # HASH the pair rather than truncating it. The boot id alone is already 32
    # alphanumeric characters, so `(boot + started)[:32]` threw the start time
    # away and every container restart produced the SAME id -- which is the one
    # thing this must never do, since it is what tells a replacement instance
    # apart from the one it replaced.
    import hashlib

    return hashlib.sha256((_alnum(boot.strip()) + ":" + _alnum(started)).encode()).hexdigest()[:32]


def forge_clone_url(repo: str) -> str:
    """The HTTPS clone URL for ``owner/name`` on the active forge."""
    from sharedagents_web import forge_client

    name = str(repo or "").strip().strip("/")
    if not name:
        return ""
    base = forge_client.forge_url()
    return f"{base}/{name}.git" if base else ""


@contextmanager
def clone_write_lock(path: Path, *, timeout: float = 180.0):
    """Serialise MUTATING git on one clone, across threads AND processes.

    Every run on a repo shares one clone, and the operator runs many at once.
    ``git fetch``, ``git worktree add`` and ``git push`` all take
    ``.git/index.lock`` or rewrite ``.git/worktrees``; two runs starting
    together raced and one died on "Unable to create '…/index.lock': File
    exists" -- which the tab reported as "no worktree, nothing dispatched".

    The lock is a FILE lock, not a `threading.Lock`, because the web runs
    several gunicorn workers: a lock inside one process protects nothing from
    the others. Read-only git (log, diff, ls-files) is deliberately NOT
    serialised -- it does not take git's locks, and holding this across a large
    diff would stall every sibling run.

    Best effort: a lock that cannot be taken lets the caller proceed rather
    than failing the run, because the pre-existing behaviour was no lock at all.
    """
    import errno
    import fcntl

    handle = None
    try:
        lock_dir = path / ".git"
        lock_dir.mkdir(parents=True, exist_ok=True)
        handle = open(lock_dir / "sharedagents-sdlc.lock", "w")
    except OSError:
        handle = None
    if handle is None:
        yield False
        return
    deadline = time.monotonic() + max(0.0, timeout)
    held = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    break
                if time.monotonic() >= deadline:
                    log.warning(
                        "sdlc: clone lock on %s still held after %ss; proceeding without it", path, int(timeout)
                    )
                    break
                time.sleep(0.25)
        yield held
    finally:
        if held:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            handle.close()
        except OSError:
            pass


def clone_repo(repo: str, path: Path, *, runner: Any = None) -> None:
    """Clone *repo* from the forge into *path*. Never overwrites anything.

    The Dev tab lists every repo the forge token can see, so the operator can
    pick one that has no local checkout -- a brand-new repo always is one. The
    run needs a clone to cut its worktree from, and refusing to make one meant
    a new repo could never be worked on at all.

    Fails closed on a non-empty path: the local source root is the operator's,
    and a directory that is not ours is never emptied to make room.
    """
    if path.exists() and any(path.iterdir()):
        raise WorktreeError(f"{path} already exists and is not a git clone of {repo}")
    url = forge_clone_url(repo)
    if not url:
        raise WorktreeError(f"no local clone of {repo} and no forge is configured to clone it from")
    path.parent.mkdir(parents=True, exist_ok=True)
    code, out = _git(
        [*_origin_auth_args(), "clone", "--quiet", url, str(path)],
        cwd=path.parent,
        runner=runner,
    )
    if code != 0 or not (path / ".git").exists():
        raise WorktreeError(f"git clone of {repo} failed: {_redact_git_secret(out)[:300]}")


def _origin_auth_args() -> list[str]:
    """``git -c http.extraHeader=...`` so fetch/push can use the forge token."""
    from sharedagents_web import forge_client

    token = forge_client.forge_token()
    if not token:
        return []
    return ["-c", f"http.extraHeader=Authorization: token {token}"]


def _redact_git_secret(text: str) -> str:
    from sharedagents_web import forge_client

    token = forge_client.forge_token()
    return text.replace(token, "***") if token else text


def _git(args: list[str], *, cwd: Path, runner: Any = None, max_bytes: int = 0) -> tuple[int, str]:
    """Run one git command. Returns ``(returncode, combined output)``.

    ``max_bytes`` bounds what is kept. A handful of large changed files can
    otherwise be captured whole -- and copied again through composition and
    prompt assembly -- to have all but the first few kilobytes thrown away.
    """
    import subprocess

    if max_bytes and runner is None:
        # Genuinely bounded: `capture_output=True` buffers git's whole output
        # before anything can truncate it, so the RETURN was bounded while peak
        # memory was not. Read what is wanted, in binary, under the same
        # deadline the buffered path has, then stop the process.
        import os
        import select

        try:
            proc = subprocess.Popen(
                ["git", "-C", str(cwd), *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return 1, f"{type(exc).__name__}: {exc}"

        chunks: list[bytes] = []
        total = 0
        hit_ceiling = False
        deadline = time.monotonic() + GIT_TIMEOUT_S
        timed_out = False
        try:
            assert proc.stdout is not None
            fd = proc.stdout.fileno()
            while total < int(max_bytes):
                left = deadline - time.monotonic()
                if left <= 0:
                    timed_out = True
                    break
                # A slow producer must not park here forever: the buffered path
                # had a 120s timeout and this one lost it.
                if not select.select([fd], [], [], min(1.0, left))[0]:
                    continue
                chunk = os.read(fd, min(65536, int(max_bytes) - total))
                if not chunk:
                    break  # git finished on its own
                chunks.append(chunk)
                total += len(chunk)
            else:
                hit_ceiling = True
        except OSError as exc:
            return 1, f"{type(exc).__name__}: {exc}"
        finally:
            if hit_ceiling or timed_out:
                proc.kill()
            try:
                proc.wait(timeout=GIT_TIMEOUT_S)
            except subprocess.SubprocessError:
                proc.kill()  # never leave it unreaped
                try:
                    proc.wait(timeout=5)
                except subprocess.SubprocessError:
                    pass
            if proc.stdout is not None:
                proc.stdout.close()

        out = b"".join(chunks).decode("utf-8", errors="replace").strip()
        if timed_out:
            return 1, f"git timed out after {GIT_TIMEOUT_S}s"
        if hit_ceiling:
            # WE stopped it -- a bounded read, not a git failure. Unless git had
            # ALREADY exited badly by the time we reached the ceiling, which at
            # exactly max_bytes it can have.
            code = proc.returncode
            if code is not None and code > 0:
                return int(code), out
            return 0, out
        # Git exited on its own, so its verdict is the verdict: returning 0
        # unconditionally made a fatal error look like a valid empty diff, and
        # `_branch_diff` would have treated the error text as evidence.
        return int(proc.returncode or 0), out

    run = runner or subprocess.run
    try:
        result = run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    out = ((result.stdout or "") + (result.stderr or "")).strip()
    if max_bytes and len(out) > max_bytes:
        out = out[:max_bytes]
    return int(result.returncode), out


def existing_worktree(repo_path: Path, branch: str, *, runner: Any = None) -> str:
    """The path already checked out on *branch*, or ''.

    ``git worktree list --porcelain`` prints one stanza per checkout; a branch
    is reported as a full ref, so compare against ``refs/heads/<branch>``.
    """
    code, out = _git(["worktree", "list", "--porcelain"], cwd=repo_path, runner=runner)
    if code != 0:
        return ""
    path = ""
    for line in out.splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree ") :].strip()
        elif line.startswith("branch ") and path and line[len("branch ") :].strip() == f"refs/heads/{branch}":
            return path
    return ""


def rebase_onto_base(repo: str, branch: str, *, runner: Any = None) -> tuple[bool, str]:
    """Replay this branch on top of the current base. ``(ok, detail)``.

    Parallel runs on one repository race to land, and the loser's review was
    made against the tree the winner replaced. Rebasing is how the loser gets
    a tree worth re-reviewing; a CONFLICT is real work for the producer, so it
    is reported rather than resolved here, and the branch is restored first.
    """
    branch = str(branch or "").strip()
    if not branch:
        return False, "no branch"
    try:
        path = repo_dir(repo)
    except WorktreeError as error:
        return False, str(error)
    worktree = existing_worktree(path, branch, runner=runner)
    if not worktree:
        return False, "no worktree for this branch"
    with clone_write_lock(path):
        _git([*_origin_auth_args(), "fetch", "--quiet", "--prune", "origin"], cwd=path, runner=runner)
    base = default_base(repo, runner=runner)
    code, out = _git(["rebase", base], cwd=Path(worktree), runner=runner)
    if code != 0:
        # Leave the branch exactly as it was; a half-rebased worktree is worse
        # than a stale one. If the abort ITSELF fails the tree really is
        # half-rebased, and saying "conflict" about it would send the producer
        # to resolve markers that are not there.
        acode, aout = _git(["rebase", "--abort"], cwd=Path(worktree), runner=runner)
        detail = _redact_git_secret(out)[:300]
        # An abort can "fail" simply because the rebase never started (a dirty
        # worktree stops it up front), and that leaves nothing half-done.
        # Only a rebase that is REALLY still in progress is the bad state.
        if acode != 0 and _rebase_in_progress(Path(worktree), runner=runner):
            return False, (
                "the rebase failed AND could not be aborted, so the "
                "worktree is mid-rebase: " + _redact_git_secret(aout)[:200]
            )
        return False, detail
    return True, f"rebased onto {base}"


def merge_base_into_branch(repo: str, branch: str, *, runner: Any = None) -> tuple[str, str, list[str]]:
    """Bring the base INTO the branch. ``(outcome, detail, conflicted_files)``.

    ``outcome`` is ``"merged"``, ``"conflict"`` or ``"error"``.

    The rebase path restores the branch on conflict, which is right when the
    harness must not guess -- but it leaves the branch behind the base forever,
    so the producer reworks against a tree that can never merge and the run
    loops. Observed live: three rounds of genuine test rework, each one refused
    for being "5 commit(s) behind".

    So on a conflict this leaves the merge IN THE WORKING TREE, markers and
    all, and reports which files. Resolving conflict markers is work an agent
    can actually do; running the right git incantation from a prompt is not
    (it was asked, and it reworked the tests instead). ``ensure_worktree``
    reuses a worktree without resetting it, so the conflicted state survives
    into the next `code` stage.
    """
    branch = str(branch or "").strip()
    if not branch:
        return "error", "no branch", []
    try:
        path = repo_dir(repo)
    except WorktreeError as error:
        return "error", str(error), []
    worktree = existing_worktree(path, branch, runner=runner)
    if not worktree:
        return "error", "no worktree for this branch", []
    with clone_write_lock(path):
        _git([*_origin_auth_args(), "fetch", "--quiet", "origin"], cwd=path, runner=runner)
    base = default_base(repo, runner=runner)
    tree = Path(worktree)
    # A merge already in progress from a previous attempt is the state we
    # want; do not start a second one on top of it.
    conflicts = _conflicted_files(tree, runner=runner)
    if conflicts:
        return "conflict", f"merge of {base} already open", conflicts
    unfinished = _merge_in_progress(tree, runner=runner)
    if unfinished:
        # Staged and never committed. Do NOT start another merge on top -- that
        # aborts and discards the resolutions.
        return (
            "conflict",
            f"a {unfinished} is resolved but not committed",
            [f"(the {unfinished} is resolved but not committed)"],
        )
    code, out = _git(["merge", "--no-edit", base], cwd=tree, runner=runner)
    detail = _redact_git_secret(out)[:400]
    if code == 0:
        return "merged", detail or f"merged {base}", []
    conflicts = _conflicted_files(tree, runner=runner)
    if conflicts:
        return "conflict", detail, conflicts
    # Failed for some other reason -- an unrelated dirty tree, a bad ref. Leave
    # nothing half-done.
    _git(["merge", "--abort"], cwd=tree, runner=runner)
    return "error", detail, []


def _conflicted_files(tree: Path, *, runner: Any = None) -> list[str]:
    """Paths git reports as unmerged, newest state, no caching."""
    code, out = _git(["diff", "--name-only", "--diff-filter=U"], cwd=tree, runner=runner)
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _merge_in_progress(tree: Path, *, runner: Any = None) -> str:
    """The unfinished git operation in this worktree, or "".

    An agent that resolves and ``git add``s but dies before committing leaves
    NO unmerged entries and an open operation. Reading only the index calls
    that tree clean, lets the run advance, and then blocks it at merge -- and a
    second ``git merge`` there aborts, throwing the resolutions away.

    A rebase and a cherry-pick leave the same shape and are just as unfinished,
    so all three are reported.
    """
    for head, label in (("MERGE_HEAD", "merge"), ("CHERRY_PICK_HEAD", "cherry-pick"), ("REVERT_HEAD", "revert")):
        code, _out = _git(["rev-parse", "--verify", "--quiet", head], cwd=tree, runner=runner)
        if code == 0:
            return label
    if _rebase_in_progress(tree, runner=runner):
        return "rebase"
    return ""


def _rebase_in_progress(tree: Path, *, runner: Any = None) -> bool:
    """``.git/rebase-merge`` / ``rebase-apply`` — REBASE_HEAD is not enough."""
    code, out = _git(["rev-parse", "--git-path", "rebase-merge"], cwd=tree, runner=runner)
    code2, out2 = _git(["rev-parse", "--git-path", "rebase-apply"], cwd=tree, runner=runner)
    for ok, raw in ((code, out), (code2, out2)):
        if ok != 0:
            continue
        candidate = Path(raw.strip())
        if not candidate.is_absolute():
            candidate = Path(tree) / candidate
        if candidate.exists():
            return True
    return False


def branch_already_landed(repo: str, branch: str, *, runner: Any = None) -> bool:
    """Is this branch's head already contained in the base?

    Then there is nothing left to do, whatever the run row says. A merge that
    lands while the run's bookkeeping is lost (a dropped lease, a worker that
    died between merging and patching) leaves the row saying `merged=0`, and
    the driver re-enters the pipeline on a branch whose diff against the base
    is now EMPTY -- so every trace check reports the artifact missing. Seen
    live: a merged run rewound to `plan` and failed with "no plan file is
    committed on the branch" against a branch that plainly had one.
    """
    branch = str(branch or "").strip()
    if not branch:
        return False
    try:
        path = repo_dir(repo)
    except WorktreeError:
        return False
    if not (path / ".git").exists():
        return False
    with clone_write_lock(path):
        _git([*_origin_auth_args(), "fetch", "--quiet", "origin"], cwd=path, runner=runner)
    base = default_base(repo, runner=runner)
    head = existing_worktree(path, branch, runner=runner) or str(path)
    code, _out = _git(["merge-base", "--is-ancestor", branch, base], cwd=Path(head), runner=runner)
    return code == 0


def default_base(repo: str, *, runner: Any = None) -> str:
    """The repo's own default branch ref, e.g. ``origin/main`` or ``origin/trunk``.

    Hardcoding ``origin/main`` broke every repo that does not use it: worktrees
    could not be created, and a missing ref made the bookend checks report false
    failures. Resolved from ``origin/HEAD`` where the clone records it.
    """
    try:
        path = repo_dir(repo)
    except WorktreeError:
        return "origin/main"
    code, out = _git(["symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"], cwd=path, runner=runner)
    if code == 0 and out.strip().startswith("refs/remotes/"):
        return out.strip()[len("refs/remotes/") :]
    for candidate in ("origin/main", "origin/master"):
        if _git(["rev-parse", "--verify", "--quiet", candidate], cwd=path, runner=runner)[0] == 0:
            return candidate
    return "origin/main"


def _assert_clone_of(path: Path, origin: str, repo: str) -> None:
    """Refuse a clone that is a DIFFERENT forge repo with the same basename.

    ``~/src`` is keyed by basename, so ``acme/widgets`` and ``other/widgets``
    both resolve to ``~/src/widgets`` -- and a run would branch into whichever
    happens to be there.

    Only a remote-looking origin is checked. A clone whose origin is a local
    path (a fixture, a mirror, a worktree of a worktree) carries no comparable
    ``owner/name``, and refusing those would break working setups to guard a
    case they cannot be in.
    """
    url = str(origin or "").strip()
    if not url:
        return
    remote = "://" in url or ("@" in url and ":" in url.split("@", 1)[1])
    if not remote:
        return
    tail = url.removesuffix(".git").rstrip("/")
    parts = [segment for segment in tail.replace(":", "/").split("/") if segment]
    if len(parts) < 2:
        return
    actual = "/".join(parts[-2:]).lower()
    wanted = str(repo).strip().strip("/").lower()
    if wanted and actual != wanted:
        raise WorktreeError(f"{path} is a clone of {url}, not {repo}")


def ensure_worktree(
    repo: str,
    branch: str,
    *,
    base: str = "",
    runner: Any = None,
) -> str:
    """Guarantee an isolated checkout for *branch* and return its path.

    Every run gets one.  The stage prompts then state the path as a fact rather
    than asking the agent to create it -- an agent that "forgot" would commit on
    the operator's main checkout, which is the one thing this must never allow.

    Idempotent: an existing worktree for the branch is reused, never reset.
    """
    branch = str(branch or "").strip()
    if not branch:
        raise WorktreeError("branch is required")
    path = repo_dir(repo)
    if not (path / ".git").exists():
        # A repo the operator picked in the Dev tab but has never checked out
        # locally used to block the run at `worktree`, before a single stage was
        # dispatched -- so the pipeline could not start work on a NEW repo at
        # all, which is precisely the case an autonomous run has to handle.
        # Clone it, but only into a path that is absent or empty: an existing
        # directory holding anything else is the operator's, not ours.
        clone_repo(repo, path, runner=runner)
    if not (path / ".git").exists():
        raise WorktreeError(f"{path} is not a git clone of {repo}")
    code, origin = _git(["remote", "get-url", "origin"], cwd=path, runner=runner)
    if code == 0:
        _assert_clone_of(path, origin, repo)

    found = existing_worktree(path, branch, runner=runner)
    if found:
        return found

    target = path / ".worktrees" / branch
    target.parent.mkdir(parents=True, exist_ok=True)
    # Everything from here rewrites the clone, so it happens under the lock and
    # the "does this branch exist" answers cannot go stale between the question
    # and `worktree add`.
    with clone_write_lock(path):
        # Another run may have created this very worktree while we queued.
        found = existing_worktree(path, branch, runner=runner)
        if found:
            return found
        # Best effort: a stale remote ref would branch from the wrong commit,
        # but a fetch failure (offline, auth) must not block local work. Prune
        # is required: deleting the forge branch does not itself erase this
        # clone's `refs/remotes/origin/<branch>`.
        _git([*_origin_auth_args(), "fetch", "--quiet", "--prune", "origin"], cwd=path, runner=runner)

        local_exists = (
            _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=path, runner=runner)[0] == 0
        )
        remote_exists = (
            _git(["rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}"], cwd=path, runner=runner)[0] == 0
        )
        if local_exists:
            args = ["worktree", "add", str(target), branch]
        elif remote_exists:
            args = ["worktree", "add", "--track", "-b", branch, str(target), f"origin/{branch}"]
        else:
            args = ["worktree", "add", "-b", branch, str(target), base or default_base(repo, runner=runner)]

        code, out = _git(args, cwd=path, runner=runner)
    if code != 0:
        # A sibling run that won the race may have made it after our check.
        found = existing_worktree(path, branch, runner=runner)
        if found:
            return found
        raise WorktreeError(f"git worktree add failed: {out[:300]}")
    return str(target)


def remove_worktree(
    repo: str,
    branch: str,
    *,
    runner: Any = None,
    path_hint: str = "",
) -> tuple[bool, str]:
    """Remove a run's checkout *and* its exact local branch.

    A discarded run must not seed a later run that reuses the same branch
    name.  In particular, the checkout may already be gone while the local ref
    survives, so branch removal is not conditional on finding a worktree.
    """
    branch = str(branch or "").strip()
    if not branch:
        return True, "no branch"
    if branch in {"main", "master"}:
        return False, f"refused to delete protected branch {branch}"
    try:
        clone = repo_dir(repo)
    except WorktreeError as exc:
        return False, str(exc)
    base_branch = default_base(repo, runner=runner).removeprefix("origin/")
    if branch == base_branch:
        return False, f"refused to delete protected branch {branch}"
    found = existing_worktree(clone, branch, runner=runner) or str(path_hint or "")
    removed_note = "no worktree"
    with clone_write_lock(clone):
        if found:
            found_p = Path(found).resolve()
            allowed = (clone / ".worktrees").resolve()
            if found_p != allowed and allowed not in found_p.parents:
                return False, "worktree is outside the clone .worktrees/ directory"
            code, out = _git(
                ["worktree", "remove", "--force", str(found_p)],
                cwd=clone,
                runner=runner,
            )
            if code != 0:
                return False, _redact_git_secret(out)[:240]
            removed_note = f"cleared worktree {found_p}"

        # `branch -D` reports failure when the ref is already absent.  The
        # postcondition, not that command's wording, is authoritative.
        code, out = _git(["branch", "-D", "--", branch], cwd=clone, runner=runner)
        tracking_ref = f"refs/remotes/origin/{branch}"
        _git(["update-ref", "-d", tracking_ref], cwd=clone, runner=runner)
        ref_code, ref_out = _git(
            ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=clone,
            runner=runner,
        )
        tracking_code, tracking_out = _git(
            ["show-ref", "--verify", "--quiet", tracking_ref],
            cwd=clone,
            runner=runner,
        )
    if ref_code == 0:
        detail = _redact_git_secret(out or ref_out)[:240]
        return False, f"local branch {branch} remains: {detail or 'deletion failed'}"
    if ref_code not in {1}:
        return False, f"could not verify deletion of local branch {branch}: " + (
            _redact_git_secret(ref_out)[:240] or f"git exited {ref_code}"
        )
    if tracking_code == 0:
        return False, f"remote-tracking ref origin/{branch} remains: " + (
            _redact_git_secret(tracking_out)[:240] or "deletion failed"
        )
    if tracking_code not in {1}:
        return False, f"could not verify deletion of origin/{branch}: " + (
            _redact_git_secret(tracking_out)[:240] or f"git exited {tracking_code}"
        )
    branch_note = f"deleted local branch {branch}" if code == 0 else f"no local branch {branch}"
    return True, f"{removed_note}; {branch_note}; cleared origin/{branch}"


def remote_branch_sha(repo: str, branch: str, *, runner: Any = None) -> str:
    """The locally-known `origin/<branch>` sha, or "" — no fetch, on purpose.

    Captured BEFORE a rebase fetches, so a `--force-with-lease` built from it
    refuses to clobber anything that appeared in between. A lease taken after
    the fetch would already have adopted that update and would overwrite it.
    """
    try:
        path = repo_dir(repo)
    except WorktreeError:
        return ""
    code, out = _git(["rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}"], cwd=path, runner=runner)
    return out.strip() if code == 0 else ""


def push_branch(
    repo: str, branch: str, *, runner: Any = None, rewritten: bool = False, expect_remote: str = ""
) -> tuple[bool, str]:
    """Publish the run's branch so a pull request has something to point at.

    The stages commit; nobody pushed. Opening a PR against a branch that exists
    only on this disk fails with the forge's unhelpful "The target couldn't be
    found", which reads like a bad repo or a bad base rather than an unpublished
    head.

    This is the driver's job, not an agent instruction: a PR is only openable if
    the branch is really there, and that must not depend on an agent
    remembering.
    """
    branch = str(branch or "").strip()
    if not branch:
        return False, "no branch"
    path = repo_dir(repo)
    if not (path / ".git").exists():
        return False, f"{path} is not a git clone"
    worktree = existing_worktree(path, branch, runner=runner) or str(path)
    # The clone remotes are bare HTTPS. API mint uses FORGE_TOKEN
    # in the web process; `git push` does not, and then the forge asks for a
    # username on a headless TTY ("No such device or address").
    auth = _origin_auth_args()
    if not auth:
        return False, ("FORGE_TOKEN is not available to git — refuse to prompt for a username on a headless TTY")
    # A worktree shares the clone's refs and object store, so a push from one
    # takes the same locks a sibling run's fetch does.
    # A REBASE rewrites the branch, so an ordinary push to an already-published
    # PR branch is a non-fast-forward and is rejected -- forever, since every
    # retry replays the same rejection. `--force-with-lease` publishes the new
    # history while still refusing if someone else moved the remote ref.
    args = [*auth, "push", "--set-upstream"]
    if rewritten:
        # Pinned to the sha the rebase replaced when we know it: a bare lease
        # is taken against whatever the last fetch adopted, which can already
        # include somebody else's push.
        # `<branch>:` with an EMPTY expectation requires the remote ref not to
        # exist. A bare `--force-with-lease` would instead be measured against
        # whatever the rebase's fetch discovered, so a branch created in the
        # meantime could be overwritten.
        args.append(f"--force-with-lease={branch}:{expect_remote}")
    args += ["origin", branch]
    with clone_write_lock(path):
        code, out = _git(args, cwd=Path(worktree), runner=runner)
    out = _redact_git_secret(out)
    if code != 0:
        return False, out[:300]
    return True, out[:300]


# --------------------------------------------------------------------------
# Intake attachments
# --------------------------------------------------------------------------
#
# A screenshot describes a UI bug better than a paragraph, and a stack trace is
# better pasted than retyped. Files dropped on "what you already know" are
# folded into the run's `detail`, which is what the clarifier reads and what
# lands verbatim in the issue body -- so both the model and every later reader
# get them.
#
# Text is INLINED (the clarifier cannot open a link); everything else is stored
# and LINKED. Storage is jailed under the run's own id using the same
# resolve-and-prove pattern as the session artifact store.

#: Per-file and per-run ceilings. Intake is a text box, not an upload service.
ATTACHMENT_MAX_COUNT = 10
ATTACHMENT_MAX_BYTES = 8 * 1024 * 1024
ATTACHMENT_TOTAL_BYTES = 24 * 1024 * 1024
#: How much of a text file is worth putting in front of the model.
ATTACHMENT_INLINE_CHARS = 12000

#: Extensions inlined as text rather than linked. Everything else must be in
#: ``_ARTIFACT_MIME`` (the server's existing whitelist) or it is rejected.
ATTACHMENT_TEXT_EXT = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".log",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".sh",
    ".sql",
    ".html",
    ".css",
    ".diff",
    ".patch",
}


def attachment_dir(run_id: str) -> Path:
    """``<artifacts>/dev-sdlc/<run_id>``, proven to stay inside the root.

    ``run_id`` is a 12-char uuid4 hex, so it is a single well-formed segment --
    but it is checked rather than trusted, because a crafted id must not be able
    to write outside the jail.
    """
    from sharedagents_web.server import ARTIFACTS_DIR

    rid = str(run_id or "")
    if (
        not rid
        or rid in (".", "..")
        or "/" in rid
        or "\\" in rid
        or "\x00" in rid
        or not all(c in "0123456789abcdefABCDEF-" for c in rid)
    ):
        raise SdlcError("bad run id")
    root = (Path(ARTIFACTS_DIR) / "dev-sdlc").resolve()
    base = (root / rid).resolve()
    if base != root and not str(base).startswith(str(root) + os.sep):
        raise SdlcError("bad run id")
    return base


def safe_attachment_name(name: str) -> str:
    """A basename safe to write and to serve, or '' if it is not allowed."""
    from sharedagents_web.server import _ARTIFACT_MIME

    base = os.path.basename(str(name or "").replace("\\", "/")).strip()
    if not base or base.startswith(".") or "\x00" in base:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", base)[:120]
    ext = os.path.splitext(cleaned)[1].lower()
    if ext not in _ARTIFACT_MIME and ext not in ATTACHMENT_TEXT_EXT:
        return ""
    return cleaned


def save_attachments(
    run_id: str, items: list[dict[str, Any]], *, url_for: Any = None
) -> tuple[list[dict[str, Any]], str]:
    """Store *items* and return ``(saved, markdown)`` to append to the detail.

    Each item is ``{"name": str, "data_base64": str}``. Anything unreadable,
    oversized, or outside the whitelist is SKIPPED rather than failing the whole
    intake -- losing one screenshot must not lose the description the operator
    just typed.
    """
    saved: list[dict[str, Any]] = []
    lines: list[str] = []
    total = 0
    base = attachment_dir(run_id)

    for item in list(items or [])[:ATTACHMENT_MAX_COUNT]:
        if not isinstance(item, dict):
            continue
        name = safe_attachment_name(item.get("name"))
        if not name:
            continue
        try:
            raw = base64.b64decode(str(item.get("data_base64") or ""), validate=True)
        except (ValueError, TypeError):
            continue
        if not raw or len(raw) > ATTACHMENT_MAX_BYTES:
            continue
        if total + len(raw) > ATTACHMENT_TOTAL_BYTES:
            break
        total += len(raw)

        ext = os.path.splitext(name)[1].lower()
        if ext in ATTACHMENT_TEXT_EXT:
            # Inlined: a link is useless to the model reading this prompt.
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
            clipped = text[:ATTACHMENT_INLINE_CHARS]
            fence = "```"
            while fence in clipped:
                fence += "`"
            lines.append(
                f"**{name}**\n{fence}\n{clipped}\n{fence}"
                + ("\n_(truncated)_" if len(text) > ATTACHMENT_INLINE_CHARS else "")
            )
            saved.append({"name": name, "bytes": len(raw), "inlined": True})
            continue

        base.mkdir(parents=True, exist_ok=True)
        (base / name).write_bytes(raw)
        url = url_for(run_id, name) if url_for else f"./runs/{run_id}/files/{name}"
        is_image = ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
        lines.append(("!" if is_image else "") + f"[{name}]({url})")
        saved.append({"name": name, "bytes": len(raw), "inlined": False, "url": url})

    markdown = ("\n\n## Attached\n\n" + "\n\n".join(lines)) if lines else ""
    return saved, markdown


# --------------------------------------------------------------------------
# The ship phase: verify the release is still the reviewed commit, then
# the operator deploys, then a probe verifies the service came back.
# --------------------------------------------------------------------------
#
#   VERIFY    is this still the commit independent verify A/B and LSP signed?
#             Deploy-readiness judged the branch in isolation, at a commit that
#             may be hours and many merges old. Merging puts it into main AS
#             MAIN IS NOW. Those are different questions, and the gap between
#             them is time.
#   DEPLOY    the operator's action.
#   VERIFY    did the deploy actually take? Not "did the command exit 0" — the
#             deploy is launched detached precisely because it restarts the
#             process serving the request, so its exit code says nothing.


def unreviewed_stages(events: Iterable[dict[str, Any]], head: str) -> list[str]:
    """Gate marks that have not received the required later decision.

    Both independent verifiers must decide about the exact merge head because
    they judge the generated artifact. A mark of either kind is repaired only
    by a later real decision for the same stage and kind whose digest names
    that head.

    Marks remain in the immutable event log. A later decision repairs only its
    own stage/kind pair; a verification decision about any other commit cannot
    vouch for the merge candidate.
    """
    # Independent verify A/B both judge the OUTPUT head. A mark is repaired
    # only by a later decision for the same stage/kind whose digest is the
    # commit about to merge.
    latest_mark: dict[tuple[str, str], int] = {}
    latest_decision: dict[tuple[str, str], tuple[int, str]] = {}
    for index, event in enumerate(events or []):
        key = (canonical_stage(str(event.get("stage") or "")), str(event.get("gate_kind") or ""))
        status = str(event.get("status") or "")
        if status == "gate_unusable":
            latest_mark[key] = index
        elif status == "gate_decided":
            latest_decision[key] = (index, str(event.get("gate_digest") or ""))
    outstanding: set[tuple[str, str]] = set()
    for key, marked_at in latest_mark.items():
        decided_at, digest = latest_decision.get(key, (-1, ""))
        repaired = decided_at > marked_at and bool(digest)
        if key[1] in {"validation", "verification"}:
            repaired = repaired and bool(head) and digest == head
        if not repaired:
            outstanding.add(key)
    return sorted(f"{stage or '?'}/{kind or '?'}" for stage, kind in outstanding)


def configured_model_aliases() -> list[str]:
    """Every model alias this deployment could dispatch, de-duplicated."""
    seen: list[str] = []
    for row in _config().get("models") or []:
        if not isinstance(row, dict):
            continue
        alias = str(row.get("alias") or row.get("id") or "").strip()
        if alias and alias not in seen:
            seen.append(alias)
    return seen


def effective_role_models(run: dict[str, Any]) -> dict[str, str]:
    """What each role will REALLY run, after defaults resolve.

    `role_config` blanks a reviewer's model when it matches the producer's, but
    blanking only defers the choice to that agent's default -- which can, and
    in practice did, resolve straight back to the producer's model. Reading the
    stored column therefore says nothing about independence; this resolves it.

    An empty string means the agent's own engine chooses (an external CLI), and
    that is NOT evidence of sameness.
    """
    out: dict[str, str] = {}
    for role in ROLE_NAMES:
        agent, model = role_config(run, role)
        out[role] = (model or default_model(agent) or "").strip()
    return out


def single_model_review(run: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """``(reason, facts)`` -- reason is "" when the review had a second mind.

    Three distinct AGENTS and three distinct SESSIONS are enforced elsewhere.
    Neither makes the review independent: agents are seats, and a seat is only
    a second opinion if something different is sitting in it. Observed live --
    generator, verify A and verify B all resolved to `dpsk-pro`, and the run
    merged reporting "verify A and verify B agreed PASS".
    """
    models = effective_role_models(run)
    facts: dict[str, Any] = {"role_models": models}
    gen = models.get("generation", "")
    a = models.get("validation", "")
    b = models.get("verification", "")
    if not gen or not a or not b:
        return "", facts  # an engine choosing for itself cannot be proven same
    if a != gen or b != gen:
        return "", facts
    return (
        f"verify A and verify B both ran `{gen}`, the same model as the "
        f"producer -- that is one model in three seats, not an independent "
        f"review. Pick a different model for verify A or verify B."
    ), facts


def release_checks(
    run: dict[str, Any],
    *,
    pull: dict[str, Any] | None = None,
    runner: Any = None,
    events: Iterable[dict[str, Any]] | None = None,
    landed_sha: str = "",
) -> dict[str, Any]:
    """Is this release still safe to merge? ``{ok, blockers, warnings, facts}``.

    Deterministic -- no model. Every check answers one question: is the thing
    the review approved still the thing about to be merged?

    It FAILS CLOSED. A check it could not run is a blocker, not a skip: a gate
    that waves work through when git is unreadable is worse than no gate,
    because it reads as an assurance. The operator can still override, but the
    override has to be deliberate.
    """
    repo = str(run.get("repo") or "")
    branch = str(run.get("branch") or "")
    blockers: list[str] = []
    warnings: list[str] = []
    facts: dict[str, Any] = {}

    # Independence is a property of what RAN, not of how many seats were
    # filled. Blocking only when the deployment actually offers a second model,
    # so a single-model install is told rather than wedged.
    same_model, model_facts = single_model_review(run)
    facts.update(model_facts)
    if same_model:
        if len(configured_model_aliases()) > 1:
            blockers.append(same_model)
        else:
            warnings.append(
                same_model + " Only one model is configured, so this cannot be "
                "fixed here -- the review is recorded as single-model."
            )

    # A gate that advanced a stage without deciding anything did not review
    # this work. Recording that and then merging anyway makes the record an
    # annotation rather than a gate: the PR still opens so a
    # human can see and repair it, but it cannot merge or deploy unremarked.
    # Computed FIRST so neither the no-PR nor the already-merged early return
    # can skip past it.
    unusable = int(run.get("gate_unusable_count") or 0)
    if unusable:
        facts["gate_unusable_count"] = unusable
        # The identity of what is about to merge must come from the PULL, not
        # from an older review: substituting `reviewed_sha` when the candidate
        # is unknown is not fail-closed, it reuses the old identity.
        head = str((pull or {}).get("head", {}).get("sha") or "")
        if events is None:
            # Fail closed: without the log this cannot tell whether those
            # stages were reviewed later, and "cannot tell" is not "fine".
            blockers.append(
                f"{unusable} gate answer(s) decided nothing, so this work "
                f"advanced unreviewed; re-run those stages or override "
                f"deliberately"
            )
        else:
            outstanding = unreviewed_stages(events, head)
            facts["unreviewed_stages"] = outstanding
            if outstanding:
                blockers.append(
                    f"these stages advanced with a gate that decided nothing, "
                    f"and none has been reviewed against the commit about to "
                    f"merge: {', '.join(outstanding)}"
                )
            else:
                warnings.append(
                    f"{unusable} gate answer(s) decided nothing earlier in this "
                    f"run; every affected stage has since been reviewed against "
                    f"the commit about to merge"
                )

    if not run.get("pr_number"):
        blockers.append("there is no pull request yet")
        return {"ok": False, "blockers": blockers, "warnings": warnings, "facts": facts}
    if landed_sha:
        reviewed = str(run.get("reviewed_sha") or "").strip()
        landed = str(landed_sha or "").strip()
        facts["already_landed"] = True
        facts["reviewed_sha"] = reviewed
        facts["landed_sha"] = landed
        if not reviewed:
            blockers.append("no reviewed commit was recorded, so the landed SHA cannot be bound to A/B")
        elif reviewed != landed:
            blockers.append(
                f"the review approved `{reviewed[:8]}` but the PR landed `{landed[:8]}` -- that work was never reviewed"
            )
        return {
            "ok": not blockers,
            "blockers": blockers,
            "warnings": warnings,
            "facts": facts,
        }
    if run.get("merged"):
        # Merged is not a pardon: a deploy still asks this question, and work
        # that advanced past an unusable gate is no safer for having landed.
        return {"ok": not blockers, "blockers": blockers, "warnings": ["already merged"], "facts": facts}
    if not repo or not branch:
        blockers.append("the run does not name a repo and branch to check")
        return {"ok": False, "blockers": blockers, "warnings": warnings, "facts": facts}

    # Gitea reports `mergeable: null` while it is still computing. Unknown is
    # not the same as clean.
    mergeable = (pull or {}).get("mergeable")
    facts["mergeable"] = mergeable
    if mergeable is False:
        blockers.append("the pull request does not merge cleanly")
    elif mergeable is not True:
        blockers.append("the forge has not confirmed this pull request is mergeable")

    base = default_base(repo, runner=runner)
    facts["base"] = base
    if not base:
        blockers.append("could not determine the base branch to compare against")

    try:
        path = repo_dir(repo)
    except WorktreeError as exc:
        path = None
        blockers.append(f"cannot read the repository to check freshness: {exc}")
    if path is not None and not (path / ".git").exists():
        blockers.append(f"`{path}` is not a git checkout, so freshness is unknown")
        path = None

    if path is not None and base:
        code, out = _git(
            [*_origin_auth_args(), "fetch", "--quiet", "origin"],
            cwd=path,
            runner=runner,
        )
        if code != 0:
            blockers.append("could not fetch `origin`, so freshness is unknown: " + _redact_git_secret(out)[:200])
        else:
            behind, ok_b = _rev_count(path, f"{branch}..{base}", runner=runner)
            ahead, ok_a = _rev_count(path, f"{base}..{branch}", runner=runner)
            facts["behind"], facts["ahead"] = behind, ahead
            if not (ok_b and ok_a):
                blockers.append(f"could not compare `{branch}` with `{base}`, so freshness is unknown")
            elif behind:
                # The one that bites: the suite was green against a tree that
                # no longer exists.
                blockers.append(
                    f"the branch is {behind} commit(s) behind `{base}` -- the review "
                    f"and the test run happened against a different tree"
                )
            elif not ahead:
                blockers.append(f"the branch adds nothing over `{base}`")

    reviewed = str(run.get("reviewed_sha") or "")
    local, local_ok = _head_sha(repo, branch, runner=runner)
    # The authority on what will merge is the PULL, not this worktree. A local
    # checkout can lag the remote, so comparing the review against a stale local
    # head passes a review of H1 while the forge is about to merge H2.
    candidate = str((pull or {}).get("head", {}).get("sha") or "")
    head = candidate or local
    facts["head_sha"] = head
    facts["local_head_sha"], facts["reviewed_sha"] = local, reviewed
    facts["candidate_sha"] = candidate
    if not reviewed:
        blockers.append(
            "no reviewed commit was recorded, so there is nothing to prove the review saw what is about to merge"
        )
    elif not candidate and not (local_ok and local):
        blockers.append(
            "could not read the commit this pull request would merge, so the review cannot be compared with it"
        )
    elif reviewed != head:
        where = "the pull request head" if candidate else "the branch head"
        blockers.append(
            f"the review approved `{reviewed[:8]}` but {where} is now `{head[:8]}` -- that work was never reviewed"
        )
    elif candidate and local_ok and local and local != candidate:
        warnings.append(f"this checkout is at `{local[:8]}` while the pull request would merge `{candidate[:8]}`")

    try:
        validate_role_agents(
            {f"{role}_agent": str(run.get(f"{role}_agent") or "") for role in ROLE_NAMES}
            | {f"{role}_model": str(run.get(f"{role}_model") or "") for role in ROLE_NAMES}
        )
    except SdlcError as error:
        blockers.append(str(error))

    lsp = lsp_code_errors(run, runner=runner)
    facts["lsp_files"] = list(lsp.get("files") or [])
    if not lsp["ok"]:
        blockers.extend(f"LSP: {item}" for item in lsp["blockers"])

    return {"ok": not blockers, "blockers": blockers, "warnings": warnings, "facts": facts}


def _rev_count(path: Path, span: str, *, runner: Any = None) -> tuple[int, bool]:
    """``(count, could_read)``. Zero-because-unreadable is not zero commits."""
    code, out = _git(["rev-list", "--count", span], cwd=path, runner=runner)
    if code != 0:
        return 0, False
    try:
        return int(out.strip()), True
    except ValueError:
        return 0, False


def _head_sha(repo: str, branch: str, *, runner: Any = None) -> tuple[str, bool]:
    """``(sha, could_read)`` for *branch*, preferring its worktree."""
    try:
        path = repo_dir(repo)
    except WorktreeError:
        return "", False
    worktree = Path(existing_worktree(path, branch, runner=runner) or path)
    code, out = _git(["rev-parse", "HEAD"], cwd=worktree, runner=runner)
    return (out.strip(), True) if code == 0 else ("", False)


def _already_landed_sha(
    run: dict[str, Any],
    pull: dict[str, Any] | None,
    *,
    runner: Any = None,
) -> tuple[str, str]:
    """Return ``(sha, reason)`` when the reviewed work is already on the base.

    A sibling PR (or a later catch-up of this branch onto ``origin/main``)
    can leave this run's PR open with nothing to merge. Asking the forge to
    merge that empty PR hangs or returns "Please try again later"; the run
    should record the land instead.
    """
    pull = pull or {}
    forge_sha = _forge_landed_sha(pull)
    if pull.get("merged") or pull.get("merged_at"):
        if not forge_sha:
            return "", ""
        return forge_sha, "this pull request is already merged on the forge"
    reviewed = str(run.get("reviewed_sha") or "")
    if not reviewed or not forge_sha:
        return "", ""
    repo = str(run.get("repo") or "")
    try:
        path = repo_dir(repo)
    except WorktreeError:
        return "", ""
    if not path or not (path / ".git").exists():
        return "", ""
    base = default_base(repo, runner=runner)
    if not base:
        return "", ""
    fetched, _ = _git(
        [*_origin_auth_args(), "fetch", "--quiet", "origin"],
        cwd=path,
        runner=runner,
    )
    if fetched != 0:
        return "", ""
    # default_base() already returns a remote ref such as origin/main.
    code, _ = _git(
        ["merge-base", "--is-ancestor", reviewed, base],
        cwd=path,
        runner=runner,
    )
    if code != 0:
        return "", ""
    code, _ = _git(
        ["merge-base", "--is-ancestor", forge_sha, base],
        cwd=path,
        runner=runner,
    )
    if code != 0:
        return "", ""
    return forge_sha, f"reviewed commit already on {base}"


def _forge_landed_sha(pull: dict[str, Any] | None) -> str:
    pull = pull or {}
    return str(
        pull.get("merge_commit_sha") or pull.get("merge_commit_id") or (pull.get("head") or {}).get("sha") or ""
    ).strip()


def _usable_pull(pull: Any) -> dict[str, Any] | None:
    """A forge PR object with a nonempty forge-supplied SHA."""
    if not isinstance(pull, dict) or not pull:
        return None
    if not _forge_landed_sha(pull):
        return None
    if pull.get("merged") or pull.get("merged_at") or pull.get("merge_commit_sha"):
        return pull
    if isinstance(pull.get("head"), dict):
        return pull
    return None


def _record_already_landed(
    sdlc: Any,
    run: dict[str, Any],
    *,
    pull: dict[str, Any],
    landed_sha: str,
    landed_why: str,
    actor: str,
    force: bool,
    checks: dict[str, Any],
    overrode: str = "",
    forge_note: str = "",
) -> dict[str, Any]:
    """Accept a concurrent land only after post-land release checks bind the SHA."""
    from sharedagents_web import forge_client

    run_id = str(run.get("run_id") or "")
    pr_number = run.get("pr_number")
    repo = str(run.get("repo") or "")
    if actor != "operator":
        force = False
    if not checks.get("ok") and not force:
        sdlc.add_event(
            run_id,
            stage="merged",
            status="failed",
            note="already landed but release verification refused: " + "; ".join(checks.get("blockers") or []),
        )
        return {
            "ok": False,
            "error": "already landed but this release did not pass independent verification",
            "blockers": checks.get("blockers") or [],
            "warnings": checks.get("warnings") or [],
            "facts": checks.get("facts") or {},
            "http": 409,
            "run": run,
            "already_landed": landed_why,
        }
    who = "pipeline" if actor != "operator" else "the operator"
    extra = f"; forge: {forge_note[:80]}" if forge_note else ""
    run = (
        sdlc.patch(
            run_id,
            merged=True,
            status="done",
            stage="merged",
            stage_status="passed",
            last_error="",
            merged_head_sha=landed_sha,
        )
        or run
    )
    sdlc.add_event(
        run_id,
        stage="merged",
        status="passed",
        note=f"PR #{pr_number} already landed ({landed_why}{extra}) — recorded by {who}",
    )
    if not pull.get("merged") and not pull.get("merged_at"):
        owner, name = forge_client.split_repo(repo)
        forge_client._request(
            "PATCH",
            f"/repos/{owner}/{name}/pulls/{int(pr_number)}",
            {"state": "closed"},
        )
    return {
        "ok": True,
        "run": sdlc.get(run_id) or run,
        "overrode": overrode,
        "pr_number": pr_number,
        "actor": actor,
        "already_landed": landed_why,
    }


def apply_merge(
    run_id: str,
    *,
    method: str = "merge",
    force: bool = False,
    actor: str = "pipeline",
    pull: dict[str, Any] | None = None,
    store: Any = None,
) -> dict[str, Any]:
    """Merge the run's PR when release checks (incl. LSP) are green.

    ``actor`` is ``pipeline`` (automatic, never forced) or ``operator``.
    Automatic merge cannot override blockers.
    """
    from sharedagents_web import forge_client

    sdlc = store or get_sdlc_store()
    run = sdlc.get(run_id)
    if not run:
        return {"ok": False, "error": "unknown run", "http": 404}
    pr_number = run.get("pr_number")
    if not pr_number:
        return {"ok": False, "error": "this run has no pull request yet", "http": 409}
    repo = str(run.get("repo") or "")
    if pull is None:
        status, fetched = forge_client.get_pull(repo, int(pr_number))
        if status != 200 or _usable_pull(fetched) is None:
            return {
                "ok": False,
                "error": "could not fetch this pull request to bind the landed SHA",
                "http": 502,
                "run": run,
            }
        pull = fetched
    elif _usable_pull(pull) is None:
        return {
            "ok": False,
            "error": "could not bind the landed SHA: pull object is not a forge PR",
            "http": 409,
            "run": run,
        }
    landed_sha, landed_why = _already_landed_sha(run, pull)
    checks = release_checks(
        run,
        pull=pull or {},
        events=sdlc.events(run_id),
        landed_sha=landed_sha,
    )
    if landed_sha:
        return _record_already_landed(
            sdlc,
            run,
            pull=pull or {},
            landed_sha=landed_sha,
            landed_why=landed_why,
            actor=actor,
            force=force,
            checks=checks,
        )
    if actor != "operator":
        force = False
    if not checks["ok"] and not force:
        sdlc.add_event(
            run_id,
            stage="merged",
            status="failed",
            note="release verification refused: " + "; ".join(checks["blockers"]),
        )
        return {
            "ok": False,
            "error": "this release did not pass independent verification",
            "blockers": checks["blockers"],
            "warnings": checks["warnings"],
            "facts": checks["facts"],
            "http": 409,
            "run": run,
        }
    overrode = ""
    if not checks["ok"]:
        overrode = "; ".join(checks["blockers"])
        sdlc.add_event(
            run_id,
            stage="merged",
            status="started",
            note=f"{actor} OVERRODE release verification: {overrode}",
        )
    candidate = str((pull or {}).get("head", {}).get("sha") or "")
    if not candidate and not force:
        return {
            "ok": False,
            "error": "the forge did not report which commit this pull request "
            "would merge, so the checks cannot be bound to it",
            "http": 409,
            "run": run,
        }
    status, result = forge_client.merge_pull(
        repo,
        int(pr_number),
        method=method,
        title=f"{run.get('headline')} (#{pr_number})",
        message=f"Closes #{run.get('issue_number')}" if run.get("issue_number") else "",
        head_sha=candidate,
    )
    if status not in (200, 201, 204):
        st, latest = forge_client.get_pull(repo, int(pr_number))
        if st != 200 or _usable_pull(latest) is None:
            detail = str(result.get("message") or result.get("error") or f"forge HTTP {status}")
            sdlc.patch(run_id, last_error=f"merge failed: {detail}")
            sdlc.add_event(
                run_id,
                stage="merged",
                status="failed",
                note=(
                    f"forced merge failed: {detail}"
                    if overrode
                    else f"merge failed: {detail}; could not re-fetch the PR to bind the landed SHA"
                ),
            )
            return {"ok": False, "error": detail, "http": 502, "run": sdlc.get(run_id) or run}
        landed_sha, landed_why = _already_landed_sha(run, latest)
        if landed_sha:
            landed_checks = release_checks(
                run,
                pull=latest,
                events=sdlc.events(run_id),
                landed_sha=landed_sha,
            )
            return _record_already_landed(
                sdlc,
                run,
                pull=latest,
                landed_sha=landed_sha,
                landed_why=landed_why,
                actor=actor,
                force=force,
                checks=landed_checks,
                overrode=overrode,
                forge_note=str(result.get("message") or status),
            )
        detail = str(result.get("message") or result.get("error") or f"forge HTTP {status}")
        sdlc.patch(run_id, last_error=f"merge failed: {detail}")
        sdlc.add_event(
            run_id,
            stage="merged",
            status="failed",
            note=(f"forced merge failed: {detail}" if overrode else f"merge failed: {detail}"),
        )
        return {"ok": False, "error": detail, "http": 502, "run": sdlc.get(run_id) or run}
    who = "pipeline" if actor != "operator" else "the operator"
    run = (
        sdlc.patch(
            run_id,
            merged=True,
            status="done",
            stage="merged",
            stage_status="passed",
            last_error="",
            merged_head_sha=candidate,
        )
        or run
    )
    sdlc.add_event(
        run_id,
        stage="merged",
        status="passed",
        note=(
            f"PR #{pr_number} merged by {who} — verification OVERRIDDEN ({overrode})"
            if overrode
            else f"PR #{pr_number} merged by {who}"
        ),
    )
    branch = str(run.get("branch") or "").strip()
    cleanup_notes: list[str] = []
    if repo and branch:
        try:
            status_del, _ = forge_client.delete_branch(repo, branch)
            if status_del in (200, 204, 404):
                cleanup_notes.append(f"cleared remote branch {branch}")
            else:
                cleanup_notes.append(f"remote branch cleanup status {status_del}")
        except Exception as exc:  # noqa: BLE001
            cleanup_notes.append(f"remote branch cleanup failed: {exc}")
        try:
            removed, wt_note = remove_worktree(repo, branch, path_hint=str(run.get("worktree") or ""))
            if removed:
                cleanup_notes.append(wt_note)
        except Exception as exc:  # noqa: BLE001
            cleanup_notes.append(f"worktree cleanup failed: {exc}")
    if cleanup_notes:
        sdlc.add_event(
            run_id,
            stage="merged",
            status="passed",
            note="branch cleanup: " + "; ".join(cleanup_notes),
        )
    return {
        "ok": True,
        "run": sdlc.get(run_id) or run,
        "overrode": overrode,
        "pr_number": pr_number,
        "actor": actor,
    }


def delete_run(
    run_id: str,
    *,
    actor: str = "operator",
    store: Any = None,
    cancel_sessions: Callable[[dict[str, Any]], list[str]] | None = None,
    quiesce_timeout: float = 20.0,
) -> dict[str, Any]:
    """Manual delete: close the issue, delete the branch, remove the Dev row.

    Does not merge. Failed and finished runs stay until the operator clicks
    Delete.
    """
    from sharedagents_web import forge_client

    sdlc = store or get_sdlc_store()
    run = sdlc.get(run_id)
    if not run:
        return {"ok": False, "error": "unknown run", "http": 404}
    notes: list[str] = []
    # Tombstone FIRST. The supervisor includes blocked/running rows in its
    # recovery sweep; deleting the checkout while a driver still owns a lease
    # lets that driver recreate the same branch from its stale snapshot after
    # this function returns. `aborted` is terminal to every driver and excludes
    # the row from recovery while its sessions are being cancelled.
    run = (
        sdlc.patch(
            run_id,
            status="aborted",
            stage_status="",
            last_error="deletion in progress",
        )
        or run
    )
    if cancel_sessions is not None:
        try:
            notes.extend(str(item) for item in (cancel_sessions(run) or []) if str(item))
        except Exception as exc:  # noqa: BLE001 — fail closed below on activity
            notes.append(f"session cancellation failed: {type(exc).__name__}: {exc}")

    # Cancellation is asynchronous. Do not race branch removal against a turn
    # that can still commit or against another web worker holding the DB lease.
    from sharedagents_web.dev_sdlc_driver import run_is_active

    deadline = time.monotonic() + max(0.0, float(quiesce_timeout))
    while True:
        fresh = sdlc.get(run_id) or {}
        busy = run_is_active(run_id) or bool(fresh.get("driver_lease_token"))
        if not busy or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    if busy:
        return {
            "ok": False,
            "error": "run is still active after cancellation; cleanup not started",
            "http": 409,
            "run": sdlc.get(run_id) or run,
            "snapshot": dict(run),
            "notes": notes,
        }
    repo = str(run.get("repo") or "")
    issue_number = run.get("issue_number")
    branch = str(run.get("branch") or "").strip()
    if repo and issue_number:
        try:
            status, payload = forge_client.close_issue(repo, int(issue_number))
        except Exception as exc:  # noqa: BLE001
            status, payload = 0, {"error": type(exc).__name__}
        if status in (200, 201):
            notes.append(f"closed issue #{issue_number}")
        else:
            notes.append("issue still open: " + str(payload.get("message") or payload.get("error") or status))

    removed, wt_note = remove_worktree(
        repo,
        branch,
        path_hint=str(run.get("worktree") or ""),
    )
    if removed:
        notes.append(wt_note)
    else:
        # Do not erase the run record while its local branch could contaminate
        # the next run that reuses this branch name.
        return {
            "ok": False,
            "error": f"local cleanup failed: {wt_note}",
            "http": 409,
            "run": run,
            "snapshot": dict(run),
            "notes": notes,
        }

    if repo and branch:
        try:
            status, payload = forge_client.delete_branch(repo, branch)
        except Exception as exc:  # noqa: BLE001
            status, payload = 0, {"error": type(exc).__name__}
        if status in (200, 204, 404):
            notes.append(f"deleted branch {branch}" if status != 404 else f"no remote branch {branch}")
        else:
            notes.append("branch still on the forge: " + str(payload.get("message") or payload.get("error") or status))

    snapshot = dict(run)
    if not sdlc.delete(run_id):
        return {"ok": False, "error": "could not delete run", "http": 500, "run": snapshot}
    notes.append("deleted Dev run")
    return {
        "ok": True,
        "deleted": True,
        "run_id": run_id,
        "run": None,
        "snapshot": snapshot,
        "notes": notes,
    }


def archive_run(
    run_id: str,
    *,
    method: str = "merge",
    force: bool = False,
    actor: str = "operator",
    store: Any = None,
) -> dict[str, Any]:
    """Compatibility alias — delete is the only removal, and it is manual."""
    del method, force
    return delete_run(run_id, actor=actor, store=store)


def deploy_verify_command(repo: str, target: str) -> str:
    """The shell line that proves a deploy of *target* actually took, or ''.

    Empty means "this target declares no proof" -- the caller must report that
    as unverified rather than as a pass. A repo supplies its own by adding a
    ``verify`` key beside ``run`` in ``.sharedagents/deploy.json``.
    """
    for row in deploy_targets(repo):
        if row["id"] == target:
            return str(row.get("verify") or "")
    return ""


# --------------------------------------------------------------------------
# Deploy targets
# --------------------------------------------------------------------------


def _UNIT_SERVING(unit: str) -> str:
    """A unit is only deployed if it is active AND actually answering.

    `systemctl is-active` alone was never proof: a process can be up and
    serving 500s. sharedagents-web binds the LAN address, so a loopback probe
    returns 000 and would report a healthy deploy as broken.
    """
    host = os.environ.get("SHAREDAGENTS_WEB_HOST", "127.0.0.1")
    port = os.environ.get("SHAREDAGENTS_WEB_PORT", "8110")
    # `is-active` is not proof and neither is "answered something": a worker
    # that boots, fails to import a blueprint and serves 500s is up. Accept only
    # the statuses this route really returns -- 200 when authed, 302/401 when
    # not -- so a 404 (blueprint missing) or 5xx (broken worker) fails.
    return (
        f"systemctl --user is-active --quiet {unit}"
        # ...and it entered active AFTER this deploy started. Without this the
        # process the deploy is about to replace answers the probe itself.
        f' && [ "$(date -d "$(systemctl --user show -p ActiveEnterTimestamp'
        f' --value {unit} 2>/dev/null)" +%s 2>/dev/null || echo 0)"'
        f' -ge "${{SHAREDAGENTS_DEPLOY_SINCE:-0}}" ]'
        f" && case \"$(curl -sS --max-time 8 -o /dev/null -w '%{{http_code}}'"
        f' http://{host}:{port}/dev || echo 000)" in'
        f" 200|302|401) : ;; *) exit 1 ;; esac"
    )


#: Deploy targets for sharedagents itself. Any OTHER repo gets whatever its own
#: `.sharedagents/deploy.json` declares — the Dev tab is meant to ship "a feature or
#: a fix to sharedagents OR ANY CODEBASE", and hardcoding one project's systemd units
#: as the universal answer would not be that.
SHAREDAGENTS_DEPLOY_TARGETS: tuple[dict[str, str], ...] = (
    {
        "id": "web",
        "label": "Web only",
        "detail": "Restart sharedagents-web (UI-only changes).",
        "run": "systemctl --user restart sharedagents-web",
        "verify": _UNIT_SERVING("sharedagents-web"),
    },
    {
        "id": "core",
        "label": "Core services",
        "detail": "Restart orchestrator + gateway + web.",
        "run": "systemctl --user restart sharedagents-orchestrator sharedagents-gateway sharedagents-web",
        "verify": "systemctl --user is-active --quiet sharedagents-orchestrator"
        " && systemctl --user is-active --quiet sharedagents-gateway"
        " && " + _UNIT_SERVING("sharedagents-web"),
    },
    {
        "id": "full",
        "label": "Full prod deploy",
        "detail": "deploy-prod.sh: fast-forward, restart all units, health-verify.",
        "run": "~/.sharedagents/deploy-prod.sh",
        "verify": "systemctl --user is-active --quiet sharedagents-orchestrator"
        " && systemctl --user is-active --quiet sharedagents-gateway"
        " && systemctl --user is-active --quiet sharedagents-telegram"
        " && " + _UNIT_SERVING("sharedagents-web"),
    },
)

#: Always offered, always first, always the default: shipping a PR does not
#: imply restarting anything.
NO_DEPLOY = {"id": "none", "label": "No deploy", "detail": "Merge only; restart nothing.", "run": ""}


def self_repo() -> str:
    """The ``owner/name`` of the sharedagents checkout this code is running from.

    Only that exact repo gets sharedagents's built-in deploy commands. Comparing
    basenames would give any repo called ``sharedagents`` -- from any owner -- the
    ability to run this machine's production deploy.

    Returns ``""`` when the identity cannot be established: an unreadable or
    missing ``origin`` must mean NO repo gets the built-ins, not that an
    unverified checkout is assumed to be this one. Set ``SHAREDAGENTS_SELF_REPO`` to
    state it explicitly.
    """
    declared = os.environ.get("SHAREDAGENTS_SELF_REPO", "").strip().strip("/").lower()
    if declared:
        return declared
    try:
        root = Path(__file__).resolve().parents[2]
        code, origin = _git(["remote", "get-url", "origin"], cwd=root)
    except (OSError, ValueError):
        return ""
    if code != 0 or not origin:
        return ""
    tail = origin.strip().removesuffix(".git").rstrip("/")
    parts = [seg for seg in tail.replace(":", "/").split("/") if seg]
    if len(parts) < 2:
        return ""
    return "/".join(parts[-2:]).lower()


def deploy_targets(repo: str = "") -> list[dict[str, str]]:
    """What the operator may pick after merging *repo*.

    A repo declares its own deploys in ``.sharedagents/deploy.json`` at the root of
    its checkout::

        {"targets": [{"id": "staging", "label": "Staging",
                      "detail": "restart the staging unit",
                      "run": "systemctl --user restart myapp-staging",
                      "verify": "curl -fsS http://localhost:9000/healthz"}]}

    ``verify`` is optional and is the proof the deploy took: it runs after the
    deploy and its exit status is the verdict. A target without one can only
    ever report "unverified".

    It runs with ``SHAREDAGENTS_DEPLOY_SINCE`` set to the epoch second the deploy was
    launched. A probe that only asks "is it answering?" will pass against the
    process the deploy is about to replace, so a good one also checks the
    service restarted at or after that moment.

    Only ``none`` is universal. sharedagents keeps its own built-ins so the tab works
    out of the box on the repo it lives in.
    """
    targets: list[dict[str, str]] = [dict(NO_DEPLOY)]
    slug = str(repo or "").strip().strip("/").lower()
    if not slug:
        return targets
    # FULL owner/name, never the basename: matching on basename alone would hand
    # `someone-else/sharedagents` this machine's production restart commands.
    mine = self_repo()
    if mine and slug == mine:
        targets += [dict(row) for row in SHAREDAGENTS_DEPLOY_TARGETS]
        return targets
    try:
        declared = json.loads((repo_dir(repo) / ".sharedagents" / "deploy.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, WorktreeError):
        return targets
    for row in (declared.get("targets") or []) if isinstance(declared, dict) else []:
        if not isinstance(row, dict):
            continue
        target_id = str(row.get("id") or "").strip()
        command = str(row.get("run") or "").strip()
        if not target_id or target_id == "none" or not command:
            continue
        targets.append(
            {
                "id": target_id,
                "label": str(row.get("label") or target_id),
                "detail": str(row.get("detail") or command),
                "run": command,
                # Optional: the shell line that proves this deploy took. A target
                # without one is reported as unverified, never as verified.
                "verify": str(row.get("verify") or "").strip(),
            }
        )
    return targets


def deploy_command(repo: str, target: str) -> str:
    """The shell line for one target, or '' if the repo does not declare it."""
    for row in deploy_targets(repo):
        if row["id"] == target:
            return str(row.get("run") or "")
    return ""
