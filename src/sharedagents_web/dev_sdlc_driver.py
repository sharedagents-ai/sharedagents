"""The thread that walks one SDLC run from plan to an open pull request.

Everything the driver touches is injected: dispatching a prompt, asking whether
a session is still working, reading its final text, and writing to the forge.
That keeps the state machine testable without a gateway, a repo, or a network,
and it keeps Flask's request globals out of a background thread.

The driver never infers a pass.  A stage closes only on an explicit
``SDLC-STAGE-DONE`` marker; an idle session without one fails the *attempt*,
not the run. An independent-verification or LSP rejection loops back to code.
The run blocks only when those loops exhaust ``max_laps``, or when there is
no earlier stage to amend (the intake contract itself).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from sharedagents_web.dev_sdlc import (
    GATE_INCONCLUSIVE,
    GATE_UNUSABLE,
    REASONING_STARVATION_TOKENS,
    STAGE_TAGS,
    STAGE_TOKEN_BUDGET,
    STAGES,
    SdlcStore,
    WorktreeError,
    build_attempt_digest,
    canonical_stage,
    changed_paths_since,
    default_model,
    ensure_worktree,
    independent_reviews_ready,
    is_test_stage_path,
    parse_stage_marker,
    preflight_stage,
    prepare_artifact_parents,
    push_branch,
    role_config,
    stage_artifact_paths,
    stage_comment,
    stage_prompt,
    stages_for_run,
    stray_artifact_note,
    trace_stage,
    usable_session_id,
    validate_proposal,
    verify_delivery,
    worktree_clean,
)

log = logging.getLogger(__name__)


class _NotObserved(RuntimeError):
    """The branch was run and did not demonstrate what it claims."""


def _review_notes_block(notes: list[str]) -> bool:
    return any(
        str(n).startswith(GATE_UNUSABLE)
        or str(n).startswith(GATE_INCONCLUSIVE)
        or str(n).startswith("verify A failed")
        or str(n).startswith("verify B failed")
        for n in notes
    )


class Forge(Protocol):
    """The slice of the forge API the pipeline writes to."""

    def comment(self, repo: str, number: int, body: str) -> int | None: ...

    def label(self, repo: str, number: int, labels: list[str]) -> bool: ...

    def open_pull(
        self,
        repo: str,
        *,
        title: str,
        head: str,
        body: str,
    ) -> dict[str, Any]: ...


def _head_sha_default(repo: str, branch: str) -> str:
    """Real git, used unless a test injects something else."""
    from sharedagents_web.dev_sdlc import _head_sha

    sha, ok = _head_sha(repo, branch)
    return sha if ok else ""


@dataclass
class DriverHooks:
    """Injected effects."""

    dispatch: Callable[..., tuple[dict | None, int]]
    #: Advisory only -- shown in the UI, never used to decide a stage.
    is_live: Callable[[str], bool]
    #: ``(turn_identity, text, error)`` of a session's last COMPLETED turn. The
    #: identity -- not the text -- is what makes a report NEW: two stages can
    #: legitimately end on byte-identical markers, and comparing text would
    #: reject the second one forever. ``error`` is the turn's own upstream
    #: failure, so a provider 402 is not reported as a forgetful agent.
    final_message: Callable[[str], tuple[str, str, str]]
    #: Mint a clean Web session already bound to the run worktree. Artifact
    #: producers and independent verify A/B each get fresh sessions.
    new_session_id: Callable[..., str]
    forge: Forge
    #: Provision the run's isolated checkout. Injected so the state machine
    #: stays testable without git, but NEVER optional: a run with no worktree
    #: would edit the operator's main checkout.
    ensure_worktree: Callable[[str, str], str] = ensure_worktree
    #: The commit at the tip of a run's branch. Injected so the state machine
    #: stays testable without git; used to pin what the review approved.
    head_sha: Callable[[str, str], str] = _head_sha_default
    #: The four gates around one stage, injected so the state machine stays
    #: testable without a repo, a model, or a network.
    #:   preflight — is the raw material present at all? (cheap, structural)
    #:   validate  — does what it will work from MAKE SENSE? (semantic)
    #:   trace     — did it leave a trace of its own kind? (cheap, structural)
    #:   verify    — has what was DEFINED been IMPLEMENTED? (semantic)
    preflight_stage: Callable[[str, dict], list[str]] = preflight_stage
    validate_proposal: Callable[[str, dict], tuple[bool, list[str]]] = validate_proposal
    trace_stage: Callable[..., tuple[bool, str]] = trace_stage
    verify_delivery: Callable[..., tuple[bool, list[str]]] = verify_delivery
    #: After tests: two independent verifiers judge the same oracle evidence.
    #: Default is a single verify so unit hooks stay one-call.
    independent_reviews: Callable[..., tuple[bool, list[str]]] | None = None
    #: The newest fresh session used by a semantic gate. Production records
    #: this while dispatching; the default keeps existing/test hooks compatible.
    gate_session_id: Callable[[str, str, str], str] = lambda _run, _stage, _role: ""
    #: Publish the branch before the PR is opened. Injected for the same reason
    #: as ensure_worktree: the state machine must be testable without git.
    #: ``(repo, branch, *, rewritten=False)`` — `rewritten` after a rebase.
    push_branch: Callable[..., tuple[bool, str]] = push_branch
    #: Post a standard user message on the Jarvis conductor (subagent return).
    prompt_conductor: Callable[[dict, str], None] = lambda _run, _text: None
    #: Jarvis conductor ``sdlc`` card. Driver waits for CLEAR/LOOP here.
    sdlc_board: Callable[[dict], dict] = lambda _run: {}
    #: Register ``/goal <objective>`` on the conductor before the prompt lands.
    arm_goal: Callable[[dict, str], None] = lambda _run, _text: None
    #: Record that the driver actually exercised the tests gate. A producer
    #: finishing its turn is not evidence that any criterion ran.
    mark_tests_ready: Callable[[dict, str], None] = lambda _run, _text: None
    #: Advance the conductor board after the driver's own branch verification.
    #: Tests are still cleared exclusively by the two-reviewer path.
    mark_stage_passed: Callable[[dict, str, str, str], None] = lambda _run, _stage, _note, _sha: None
    #: Human input ends at ratification. A worker that parks on a question or
    #: tool permission has violated the unattended contract and must not hold
    #: the run until the generic 90-minute stage timeout.
    pending_interaction: Callable[[str], str] = lambda _sid: ""
    #: Cancel the exact worker turn after a forbidden interaction is detected,
    #: resolving its waiter and preventing an orphaned process/session.
    cancel_session: Callable[[str], bool] = lambda _sid: False
    #: Enforce producer ownership after TEST. Prompt wording is not a safety
    #: boundary: compare the actual repository paths to the stage-start SHA.
    changed_paths_since: Callable[[dict, str], list[str]] = changed_paths_since
    #: A stage may hand off only a complete commit.  A clean HEAD with a dirty
    #: successor in the worktree is stale evidence, not a finished artifact.
    worktree_clean: Callable[[dict], tuple[bool, list[str]]] = worktree_clean
    #: P2: the monotonic timestamp of the worker session's most recent tool
    #: call (0.0 when unknown). The default never lies about activity — it
    #: reads zero, so a session with no observable activity is treated as
    #: idle, which is the fail-closed direction.
    last_activity_at: Callable[[dict], float] = lambda _run: 0.0
    #: P1: push one steering line into the RUNNING worker session. Best
    #: effort: returning False is acceptable and simply skips the nudge.
    steer: Callable[[str, str], bool] = lambda _sid, _text: False
    #: P3: the raw activity feed for one worker session, reduced by
    #: ``build_attempt_digest`` into the bounded handoff payload.
    session_digest: Callable[[str], dict] = lambda _sid: {}
    #: P7: live token use for the current worker turn. Keys: reasoning_tokens,
    #: prompt_tokens, completion_tokens, tool_calls. Missing keys are zero.
    session_usage: Callable[[str], dict] = lambda _sid: {}
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic


#: How many times one stage may be attempted before the run blocks. A stage
#: that runs out of budget mid-task leaves its commits on the branch, so the
#: next attempt continues rather than restarts — but an agent that cannot make
#: progress must stop costing turns.
MAX_STAGE_ATTEMPTS = 2

# A stale-reconciled parent can appear quiet between sequential sibling Agent
# calls: child A exits, then the still-running tool dispatcher starts child B a
# few seconds later.  Replacing the stage in that gap creates two writers for
# one worktree.  Require several complete driver polls of whole-tree silence
# before interpreting an aborted parent as finished.
ABORTED_SUBTREE_QUIET_POLLS = 3

# PLAN receives the original request and settled intake answers. A local Qwen3.8
# trace can spend more than five minutes reasoning across a large legacy module
# while still emitting fresh repository calls. Match the producer allowance so
# that live local inference reaches its first durable artifact before the
# fail-closed watchdog decides it made no progress.
PLAN_FIRST_WRITE_TIMEOUT_S = float(os.environ.get("SHAREDAGENTS_SDLC_PLAN_FIRST_WRITE_TIMEOUT") or 900.0)

# A model can keep producing healthy-looking reasoning tokens while making no
# repository progress. That is not a 90-minute stage; it is a design loop. Give
# enough time to read the already-settled brief and write a vertical slice,
# then cancel it while the branch is still pristine.  A real Qwen3.8 CODE
# trace spent just over five minutes using repository tools and working through
# an index invariant before it reached the write call, so five minutes was a
# premature kill rather than evidence of a dead worker. Fifteen minutes still
# leaves the no-progress gate far inside the generic 90-minute stage ceiling.
CODE_FIRST_WRITE_TIMEOUT_S = float(os.environ.get("SHAREDAGENTS_SDLC_CODE_FIRST_WRITE_TIMEOUT") or 900.0)

# TEST still has its own no-progress bound, but constructing criterion-mapped
# fixtures for a legacy library can require the same local-model reasoning and
# context-compaction time as CODE. Keep all producer stages consistent while
# retaining the generic 90-minute absolute stage timeout.
TEST_FIRST_WRITE_TIMEOUT_S = float(os.environ.get("SHAREDAGENTS_SDLC_TEST_FIRST_WRITE_TIMEOUT") or 900.0)

# P2: how long a worker may sit with no tool call and no artifact before the
# watchdog stops treating it as working. The failed run's 30+ command rounds
# each landed inside this window, so investigating would have survived; the
# token-burning silence would not.
IDLE_NO_PROGRESS_S = float(os.environ.get("SHAREDAGENTS_SDLC_IDLE_NO_PROGRESS_S") or 240.0)

# P2: the unconditional per-stage ceiling. Progress or not, a stage that runs
# this long is a design loop, not a task.
STAGE_CEILING_S = float(os.environ.get("SHAREDAGENTS_SDLC_STAGE_CEILING_S") or 3600.0)

# P1: fraction of the first-artifact budget after which the driver steers the
# worker once, while it can still act on the nudge.
STEER_AT_FRACTION = 0.6

# Jarvis owns the durable contract board. Artifact production is a separate,
# fresh build-role turn so a conductor that merely *reasons* about delegation
# cannot strand the run without ever calling the delegation tool.
STAGE_EXECUTOR_AGENT = "build"


def _stage_dispatch_model(
    selected: str,
    agent: str,
    default_provider: Callable[[str], str] = default_model,
) -> str:
    """Carry the operator's selected model to every fresh stage worker.

    ``default_provider`` describes the Dev run default, not the concrete
    worker session's runtime route. Dropping an equal value therefore made a
    run recorded as Codex silently execute on Build's Qwen default. A fresh
    session can resolve the explicit alias atomically with its worktree, so
    preserving it is both cheaper and safer than guessing what it inherits.
    """
    del agent, default_provider
    return str(selected or "").strip()


def _stage_selected_model(stage: str, selected: str) -> str:
    """Allow producer stages to use a purpose-built model, with provenance."""
    suffix = "TEST" if stage == "tests" else str(stage or "").upper()
    override = os.environ.get(f"SHAREDAGENTS_SDLC_{suffix}_MODEL", "").strip()
    return override or str(selected or "").strip()


def _stored_rework(run: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """``(next attempt number, digest)`` a blocked run carries into its retry.

    P3/F3: the supervisor restarts a blocked run into a brand-new worker
    session, so the only bridge between attempts is the durable row. A run
    with nothing stored starts clean at attempt 1 with no digest.
    """
    raw = str(run.get("rework_digest") or "").strip()
    digest: dict[str, Any] = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                digest = parsed
        except ValueError:
            digest = {}
    count = int(run.get("no_progress_attempts") or 0)
    return count + 1, digest


@dataclass
class DriverTiming:
    poll_seconds: float = 8.0
    #: A real stage is minutes of agent work, sometimes tens of them. This is
    #: the only bound on one; there is no separate "did it start" timeout,
    #: because nothing observable distinguishes a slow start from slow work.
    stage_timeout: float = 5400.0


_threads: dict[str, threading.Thread] = {}
#: run_id -> the lease token this process holds, so the driver can renew it.
_leases: dict[str, str] = {}
_threads_lock = threading.Lock()


@dataclass
class SdlcDriver:
    store: SdlcStore
    hooks: DriverHooks
    timing: DriverTiming = field(default_factory=DriverTiming)

    # -- forge helpers ---------------------------------------------------
    def _post(
        self,
        run: dict[str, Any],
        *,
        stage: str,
        status: str,
        note: str = "",
        evidence: str = "",
        lap: int = 0,
        cancellation_confirmed: bool | None = None,
        cancellation_idle_at: str = "",
    ) -> None:
        """Record the stage outcome locally and on the issue. Never raises."""
        comment_id = None
        repo, number = str(run.get("repo") or ""), run.get("issue_number")
        if repo and number:
            body = stage_comment(
                stage=stage,
                status=status,
                note=note,
                evidence=evidence,
                lap=lap,
            )
            # The issue IS the build log, so one transient failure should not
            # silently lose a stage. Two tries, then record the gap — a forge
            # outage still must not stall the run itself.
            for attempt in (1, 2):
                try:
                    comment_id = self.hooks.forge.comment(repo, int(number), body)
                except Exception:  # noqa: BLE001 — a forge outage must not stall the run
                    log.warning("sdlc: issue comment failed for %s", run.get("run_id"))
                    comment_id = None
                if comment_id is not None:
                    break
                if attempt == 1:
                    self.hooks.sleep(min(self.timing.poll_seconds, 3.0))
            if comment_id is None:
                # The issue is the build log. If a stage never reached it, the
                # run must say so rather than present a complete local history.
                note = (note + " " if note else "") + "[not posted to the issue]"
            try:
                self.hooks.forge.label(repo, int(number), [STAGE_TAGS.get(stage, "")])
            except Exception:  # noqa: BLE001
                pass
        child = str(
            run.get("review_session_id")
            if stage == "deploy"
            else run.get("author_session_id") or run.get("session_id") or ""
        )
        if stage in STAGES:
            self.store.add_event(
                str(run["run_id"]),
                stage=stage,
                status=status,
                note=note,
                lap=lap,
                comment_id=comment_id,
                executor_agent=STAGE_EXECUTOR_AGENT,
                executor_model=str(run.get("stage_executor_model") or role_config(run, "generation")[1]),
                executor_session_id=child,
                cancellation_confirmed=cancellation_confirmed,
                cancellation_idle_at=cancellation_idle_at,
            )
        else:
            self.store.add_event(
                str(run["run_id"]),
                stage=stage,
                status=status,
                note=note,
                lap=lap,
                comment_id=comment_id,
                cancellation_confirmed=cancellation_confirmed,
                cancellation_idle_at=cancellation_idle_at,
            )
        # The next step is a new ``/goal`` on this same conductor. Do not
        # inject a spectator "subagent return" turn.

    def _sync_board_worktree(self, run: dict[str, Any], worktree: str) -> None:
        sid = str(run.get("conductor_session_id") or "").strip()
        if not sid or not worktree:
            return
        try:
            from sharedagents_agent.tools.impl.sdlc import set_board_worktree
            from sharedagents_core.kernel.layout import sessions_root

            set_board_worktree(sessions_root(), sid, worktree)
        except Exception as error:  # noqa: BLE001 — board miss must not stall
            log.debug("sdlc: could not stamp worktree on board: %s", error)

    # -- one stage -------------------------------------------------------
    def _verified(self, stage: str, run: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any]:
        """VERIFY: downgrade a claim the branch does not corroborate.

        The thing most likely to be wrong about whether work happened is the
        agent that was supposed to do it, so a PASS that left no trace on the
        branch is turned into a failure with the reason.
        """

        def _downgrade(why: str) -> dict[str, Any]:
            return {
                "status": "FAIL",
                "note": f"{marker.get('note') or 'reported PASS'} — but {why}",
                "evidence": str(marker.get("evidence") or ""),
                "verdict": True,  # still a verdict: the stage did report
                "unverified": True,
            }

        def _clean_handoff() -> tuple[bool, str]:
            try:
                observed, dirty = self.hooks.worktree_clean(run)
            except Exception as error:  # noqa: BLE001 — unknown is not clean
                return False, f"worktree cleanliness could not be checked: {error}"
            if not observed:
                return False, (
                    "worktree cleanliness could not be checked"
                    + (": " + "; ".join(str(x) for x in dirty) if dirty else "")
                )
            if dirty:
                return False, (
                    "the stage left uncommitted work outside its accepted SHA: " + ", ".join(str(x) for x in dirty[:20])
                )
            return True, ""

        clean, clean_why = _clean_handoff()
        if not clean:
            return _downgrade(clean_why)

        # TRACE: cheap structural sanity — did it commit anything of its kind?
        try:
            traced, why = self.hooks.trace_stage(stage, run, marker)
        except Exception as error:  # noqa: BLE001 — fail closed
            log.warning("sdlc: trace check errored for %s: %s", stage, error)
            return _downgrade(f"trace check could not run: {error}")
        if not traced:
            return _downgrade(why)

        # VERIFY: has what was DEFINED been IMPLEMENTED? The stage's own word is
        # the least reliable evidence for that, so it is checked against the
        # artifact the previous stage left.
        # Verification judges what the stage COMMITTED, so the head is read
        # fresh here -- and read again afterwards, so a tree that moved during
        # the judgement yields no claim at all.
        judged_before = self._read_head(run)
        if not self._renew_lease(str(run.get("run_id") or "")):
            # Another worker owns this run. Returning the marker UNCHANGED
            # would hand back the PASS the stage claimed, which the caller then
            # persists as a passed stage -- a verdict recorded by a worker that
            # no longer owns the run. Stop instead.
            return {"status": "", "note": "lease lost", "lease_lost": True}
        # tests already required both reviewers to agree on the sdlc board
        # before CLEAR. Do not run a third semantic review.
        if stage == "tests":
            judged = self._judged(run, judged_before)
            clean, clean_why = _clean_handoff()
            if not clean:
                return _downgrade(clean_why)
            if not judged:
                return _downgrade("the branch moved while its evidence was being accepted")
            marker["accepted_sha"] = judged
            return marker
        pair = False
        review = self.hooks.verify_delivery
        try:
            delivered, missing = review(stage, run, marker)
        except Exception as error:  # noqa: BLE001
            log.warning("sdlc: verification errored for %s: %s", stage, error)
            kinds = ("validation", "verification") if pair else ("verification",)
            for kind in kinds:
                self._record_gate(
                    run,
                    stage,
                    kind,
                    ran=False,
                    notes=[f"{GATE_UNUSABLE} the {kind} verification for '{stage}' failed: {error}"],
                    session_id=self._gate_session(run, stage, kind),
                )
            return _downgrade("verify A and verify B did not both finish: " + str(error))
        examined = str(marker.get("status") or "").upper() == "PASS"
        digest = self._judged(run, judged_before)
        if not digest:
            return _downgrade("the branch moved while its evidence was being judged")
        if pair:
            notes_a = [n[len("verify-a: ") :] for n in missing if str(n).startswith("verify-a: ")]
            notes_b = [n[len("verify-b: ") :] for n in missing if str(n).startswith("verify-b: ")]
            extra = [n for n in missing if not str(n).startswith("verify-a: ") and not str(n).startswith("verify-b: ")]
            self._record_gate(
                run,
                stage,
                "validation",
                ran=examined and delivered,
                notes=notes_a,
                digest=digest,
                session_id=self._gate_session(run, stage, "validation"),
            )
            self._record_gate(
                run,
                stage,
                "verification",
                ran=examined and delivered,
                notes=notes_b,
                digest=digest,
                session_id=self._gate_session(run, stage, "verification"),
            )
            missing = list(missing) + extra
        else:
            if stage == "tests":
                # Independent A already ran (or was stamped) before CLEAR.
                # Recording only B left the verify A chip pending after PASS.
                self._record_gate(
                    run,
                    stage,
                    "validation",
                    ran=examined,
                    notes=[],
                    digest=digest,
                    session_id=self._gate_session(run, stage, "validation"),
                )
            self._record_gate(
                run,
                stage,
                "verification",
                ran=examined,
                notes=missing,
                digest=digest,
                session_id=self._gate_session(run, stage, "verification"),
            )
        if not delivered:
            return _downgrade("this was defined and is not implemented: " + "; ".join(missing))
        if stage == "code":
            from sharedagents_web.dev_sdlc import lsp_code_errors, unmerged_paths

            # An open merge is OURS -- opened when a sibling landed first. Grade
            # nothing until it is resolved, or the run blocks on syntax errors
            # in markers the pipeline wrote, and the agent is never asked.
            conflicted = unmerged_paths(run)
            if conflicted:
                return _downgrade(
                    "the base has been merged into this branch and is waiting "
                    "conflicted -- resolve the conflict markers in: "
                    + ", ".join(conflicted)
                    + ". Keep BOTH behaviours, `git add` those files and commit "
                    "the merge. Do not rebase, reset or re-clone."
                )
            lsp = lsp_code_errors(run)
            if not lsp.get("ok"):
                return _downgrade(
                    "LSP reported code errors: " + "; ".join(str(item) for item in lsp.get("blockers") or [])
                )
        clean, clean_why = _clean_handoff()
        if not clean:
            return _downgrade(clean_why)
        if self._read_head(run) != digest:
            return _downgrade("the branch moved after verification and before handoff")
        marker["accepted_sha"] = digest
        return marker

    def _renew_lease(self, run_id: str) -> bool:
        """Keep the claim alive across gate work. False means it is LOST.

        The stage timeout covers agent-turn polling only; the gates, their
        retries and the re-check windows all run outside it while the lease is
        renewed only at stage boundaries. A slow agent turn followed by slow
        gates could outlast the lease and let another worker take the run over
        mid-stage.

        The answer must be OBEYED, not logged. Renewing and continuing anyway
        masks a lease that was already lost and lets two workers write to one
        run. A caller that cannot stop must not call this.
        """
        token = _leases.get(run_id, "")
        if not token:
            return True  # not leased: nothing to lose
        try:
            held = bool(self.store.renew_driver_lease(run_id, token))
        except Exception as error:  # noqa: BLE001 — an unreadable lease is not
            # proof it was lost; the stage-boundary check remains authoritative.
            log.warning("sdlc: could not renew lease for %s: %s", run_id, error)
            return True
        if not held:
            log.warning("sdlc: %s lost its lease during gate work; another worker has it", run_id)
        return held

    def _read_head(self, run: dict) -> str:
        """The branch head RIGHT NOW, never a memo."""
        try:
            return str(self.hooks.head_sha(str(run.get("repo") or ""), str(run.get("branch") or "")) or "")
        except Exception:  # noqa: BLE001
            return ""

    def _judged(self, run: dict, before: str) -> str:
        """The content id a gate may claim, or '' if the tree moved under it.

        Capturing the SHA on one side of the judgement proves nothing: a
        force-push inside that window produces a `gate_decided` digest matching
        a tree the gate never read. The head has to be the same before
        AND after, or there is no proof and the stage stays outstanding.
        """
        after = self._read_head(run)
        if before and after and before == after:
            return before
        log.info(
            "sdlc: the branch moved while a gate was judging (%s -> %s); recording no decision", before[:12], after[:12]
        )
        return ""

    def _gate_digest(self, run: dict, stage: str = "") -> str:
        """What the gate is judging, as a content id. '' when unknown.

        Memoised per stage attempt: both gates of one stage judge the same
        branch state, and asking git twice would be a second answer to a
        question already asked.
        """
        memo = getattr(self, "_head_memo", {})
        cached = memo.get((str(run.get("run_id") or ""), stage or str(run.get("stage") or "")))
        if cached:
            return cached
        try:
            return str(self.hooks.head_sha(str(run.get("repo") or ""), str(run.get("branch") or "")) or "")
        except Exception:  # noqa: BLE001 — unknown content settles nothing
            return ""

    def _record_gate(
        self,
        run: dict,
        stage: str,
        kind: str,
        *,
        notes: list[str],
        ran: bool,
        digest: str | None = None,
        session_id: str = "",
    ) -> None:
        """Account for a gate that advanced a stage without deciding anything.

        A gate fails OPEN when it cannot decide, so a stage with no producer to
        lap back to cannot stall. That is only safe while it is visible: a model
        answering that way every time would turn both gates into no-ops and
        nothing downstream would know. The record is what a consequential
        action -- merging, deploying -- refuses on.

        A mark remains a blocker until a LATER real decision of the same stage
        and kind repairs it. Verification must decide on the final head;
        validation deliberately decides on the pre-generation input digest.
        A skipped retry, crashed gate, or FAIL marker never records a decision,
        so none can launder the mark.

        ``digest`` identifies the content actually judged.
        ``ran`` only distinguishes "the gate answered" from "no gate ran"; it
        is not persisted.
        """
        run_id = str(run.get("run_id") or "")
        if not run_id:
            # A malformed run reaching here must not raise out of a gate.
            return
        # An INCONCLUSIVE gate blocks rather than failing open, but it decided
        # nothing either -- recording only one of the two prefixes let the
        # other pass unrecorded.
        unusable = [
            n for n in (notes or []) if str(n).startswith(GATE_UNUSABLE) or str(n).startswith(GATE_INCONCLUSIVE)
        ]
        if digest is None:
            digest = self._gate_digest(run, stage)
        if not unusable:
            if ran:
                # It ANSWERED, about this exact content. Recorded separately
                # from any mark, so the release gate can tell whether a stage
                # once marked unreviewed has since been reviewed properly.
                agent, model = role_config(run, kind)
                self.store.record_gate_decided(
                    run_id,
                    stage=stage,
                    kind=kind,
                    digest=digest,
                    executor_agent=agent,
                    executor_model=model,
                    executor_session_id=session_id,
                )
            return
        log.warning(
            "sdlc: run %s stage %s advanced with an UNUSABLE %s gate: %s", run_id, stage, kind, "; ".join(unusable)
        )
        agent, model = role_config(run, kind)
        self.store.record_gate_unusable(
            run_id,
            stage=stage,
            kind=kind,
            digest=digest,
            notes=unusable,
            lap=int(run.get("lap") or 0),
            executor_agent=agent,
            executor_model=model,
            executor_session_id=session_id,
        )

    def _gate_session(self, run: dict, stage: str, kind: str) -> str:
        """Newest concrete chat session used by one semantic gate."""
        try:
            return str(self.hooks.gate_session_id(str(run.get("run_id") or ""), stage, kind) or "")
        except Exception:  # noqa: BLE001 — provenance must not break a gate
            return ""

    def _final_message(self, sid: str) -> tuple[str, str, str]:
        """``(turn_identity, text, error)`` of a session's last completed turn."""
        try:
            result = self.hooks.final_message(sid)
        except Exception:  # noqa: BLE001
            return "", "", ""
        # Tolerate the 2-tuple shape so a stubbed hook cannot crash a run.
        identity, text, error = (tuple(result) + ("", "", ""))[:3]
        return str(identity or ""), str(text or ""), str(error or "")

    def _turn_identity(self, sid: str) -> str:
        return self._final_message(sid)[0]

    def _cancel_worker_confirmed(self, sid: str) -> tuple[bool, float]:
        """Cancel a worker and report what was actually OBSERVED (P6/F4).

        ``cancel_session`` returning True is only the gateway accepting the
        request. F4 burned tokens for eighteen minutes after every watchdog
        kill because acceptance was reported as outcome. So: cancel, wait out
        a bounded settle window, and only claim confirmation when the session
        is actually seen idle. A still-live worker is cancelled once more
        (the force escalation) and then reported honestly as unconfirmed —
        the caller still fails the stage, it just does not lie about it.
        """
        try:
            self.hooks.cancel_session(sid)
        except Exception:  # noqa: BLE001 — a failed cancel must still be reported
            pass
        settle_deadline = self.hooks.now() + max(self.timing.poll_seconds, 2.0) * 2
        while self.hooks.now() < settle_deadline:
            try:
                live = bool(self.hooks.is_live(sid))
            except Exception:  # noqa: BLE001 — unknown liveness is not proof
                live = True
            if not live:
                return True, self.hooks.now()
            self.hooks.sleep(min(self.timing.poll_seconds, 2.0))
        # Escalate once, then report the truth either way.
        try:
            self.hooks.cancel_session(sid)
        except Exception:  # noqa: BLE001
            pass
        try:
            live = bool(self.hooks.is_live(sid))
        except Exception:  # noqa: BLE001
            live = True
        if not live:
            return True, self.hooks.now()
        log.warning("sdlc: worker %s cancellation could not be confirmed", sid)
        return False, 0.0

    def _store_rework(
        self, run: dict[str, Any], sid: str, note: str, *, attempt: int, stage_start_head: str = ""
    ) -> dict:
        """Persist the failed attempt's digest for the next one (P3/F3).

        The failure that made the failed run repeat itself identically three
        times was not the kill — it was the cold restart that followed. What
        the attempt learned (tools used, files read, diff produced) is worth
        one bounded paragraph in the next worker's context.
        """
        try:
            activity = self.hooks.session_digest(sid) or {}
        except Exception:  # noqa: BLE001 — a digest is best effort
            activity = {}
        try:
            artifacts = list(self.hooks.changed_paths_since(run, stage_start_head))[:20]
        except Exception:  # noqa: BLE001 — a digest is best effort
            artifacts = []
        digest = build_attempt_digest(
            attempt=attempt,
            note=note,
            session_activity=activity,
            artifacts=artifacts,
        )
        try:
            self.store.patch(str(run["run_id"]), rework_digest=json.dumps(digest, ensure_ascii=False))
        except Exception:  # noqa: BLE001 — the failure itself still stands
            log.debug("sdlc: could not store the rework digest for %s", run.get("run_id"))
        return digest

    def _stage_failure(
        self,
        note: str,
        *,
        no_progress: bool = False,
        cancellation_confirmed: bool | None = None,
        cancellation_idle_at: float = 0.0,
        run: dict[str, Any] | None = None,
        stage: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        """One failure exit: fail the stage and report cancel provenance."""
        if run is not None and stage:
            extra_note = stray_artifact_note(
                str(run.get("worktree") or ""),
                stage_artifact_paths(
                    stage,
                    str(run.get("branch") or ""),
                    fuse_plan_design=bool(run.get("fuse_plan_design", True)),
                ),
            )
            if extra_note:
                note = f"{note}; {extra_note}"
        result: dict[str, Any] = {"status": "FAIL", "note": note}
        if no_progress:
            result["no_progress"] = True
        if cancellation_confirmed is not None:
            # Only a cancelled stage reports this; None means "no cancel was
            # involved" and the key is omitted rather than guessed.
            result["cancellation_confirmed"] = bool(cancellation_confirmed)
            if cancellation_confirmed and cancellation_idle_at:
                result["cancellation_idle_at"] = cancellation_idle_at
        result.update(extra)
        return result

    #: Merge refusals that a fresh base would resolve.
    STALE_BASE_MARKERS = ("behind", "does not merge cleanly", "different tree")

    def _handle_merge_refusal(self, run_id: str, note: str) -> dict[str, Any]:
        """Decide what a refused merge means: replay, reopen `code`, or stop.

        Split out from `merge_when_green` so the decision can be tested
        without a live forge -- it is the whole of fire-and-forget under
        parallel load.
        """
        fresh = self.store.get(run_id) or {}
        if self._already_landed(fresh):
            self.store.patch(
                run_id,
                merged=True,
                status="done",
                last_error="",
                blocked_stage="",
                stage="merged",
                stage_status="passed",
            )
            self._post(fresh, stage="merged", status="passed", note="already on the base branch; recorded as merged")
            return {"ok": True, "merged": True, "reconciled": True}

        # THE parallel case. Runs on one repo race to land, so a sibling
        # merging first moves the base out from under a branch whose review
        # already happened -- "the branch is 5 commit(s) behind
        # `origin/main`; the review and the test run happened against a
        # different tree". The gate is right to refuse: merging now would
        # land a tree nobody reviewed. Refusing forever is not an answer
        # though, so take the new base and let the gates decide again.
        outcome, detail = self._rebase_for_merge(fresh, note)
        if outcome == "rebased":
            why = (
                "a sibling run landed first; this branch has been rebased "
                "onto the new base. Re-run every acceptance-criterion test "
                "against the rebased tree -- the previous PASS was earned "
                "on a tree that no longer exists."
            )
            # `blocked_stage` + `last_error` are what the resumed producer
            # prompt reads. An event alone is invisible to the agent.
            self.store.patch(run_id, status="running", last_error=why, blocked_stage="tests")
            self._post(fresh, stage="tests", status="invalidated", note=why)
            return {"ok": False, "error": note, "rebased": True}
        if outcome == "conflict":
            # Two runs edited the same file. Telling the agent to "re-apply on
            # the new base" did not work -- it reworked the tests and left the
            # branch behind, so the same refusal came back. Put the base into
            # the branch HERE, leave the conflict markers in the worktree, and
            # ask for the one thing an agent is actually good at: resolving
            # them.
            files, how, opened = self._open_base_merge(fresh)
            if not opened:
                # The base could not be merged in at all -- not a conflict the
                # producer can resolve. Reopening `code` for it would cycle
                # between `code` and `merged` forever, and orphan adoption does
                # not spend a recovery attempt, so nothing would ever stop it.
                self.store.patch(run_id, status="blocked", last_error=how, blocked_stage="merged")
                self._post(fresh, stage="merged", status="failed", note=how)
                return {"ok": False, "error": how}
            # The instruction has to ride on `last_error`/`blocked_stage`, not
            # just an event: `run()` seeds the producer's feedback from those
            # two fields and nothing else. Posting only the event is how the
            # first version of this silently asked for nothing -- the agent
            # reworked its tests three laps running and never touched the base.
            self.store.patch(run_id, status="running", last_error=how, blocked_stage="code")
            self._post(fresh, stage="code", status="invalidated", note=how)
            return {"ok": False, "error": note, "conflict": True, "conflicted_files": files}
        # Anything else is a stop, not a spin: `blocked` puts the run under
        # the bounded auto-resume instead of being re-adopted every sweep.
        # Carry what git said -- "mid-rebase worktree" is the actionable half of
        # the story and storing only the merge refusal throws it away.
        stopped = f"{note} -- {detail[:300]}" if detail else note
        self.store.patch(run_id, status="blocked", last_error=stopped, blocked_stage="merged")
        self._post(fresh, stage="merged", status="failed", note=stopped)
        return {"ok": False, "error": stopped}

    def _open_base_merge(
        self,
        run: dict[str, Any],
    ) -> tuple[list[str], str, bool]:
        """Merge the base in. ``(files, note, opened)``.

        ``opened`` is False when there is no merge for the producer to finish --
        the caller must stop rather than reopen `code` on work that does not
        exist.
        """
        repo = str(run.get("repo") or "")
        branch = str(run.get("branch") or "")
        try:
            from sharedagents_web.dev_sdlc import merge_base_into_branch

            outcome, detail, files = merge_base_into_branch(repo, branch)
        except Exception as error:  # noqa: BLE001
            log.warning("sdlc: could not open the base merge for %s: %s", run.get("run_id"), error)
            return (
                [],
                (
                    "a sibling run landed first and this branch is behind "
                    f"the base, and the merge could not be started: {error}"
                ),
                False,
            )
        if outcome == "merged":
            return (
                [],
                (
                    "a sibling run landed first; the base has been merged "
                    "into this branch cleanly. Re-run the checks against "
                    "the merged tree and make sure BOTH behaviours still "
                    "work."
                ),
                True,
            )
        if outcome == "conflict":
            listed = ", ".join(files) or "the files git lists as unmerged"
            return (
                files,
                (
                    "a sibling run landed first and its change touches the same "
                    "code as yours. The base is ALREADY MERGED INTO YOUR WORKTREE "
                    "and left conflicted on purpose -- do not rebase, reset, or "
                    "re-clone.\n"
                    f"Resolve the conflict markers in: {listed}\n"
                    "Keep BOTH behaviours -- theirs and yours. Then `git add` those "
                    "files, commit the merge, and re-run the tests for every "
                    "acceptance criterion.\n"
                    f"git said:\n{detail[:500]}"
                ),
                True,
            )
        return (
            [],
            (
                "a sibling run landed first and this branch is behind the "
                f"base, which could not be merged automatically: {detail[:400]}"
            ),
            False,
        )

    def _rebase_for_merge(self, run: dict[str, Any], why: str) -> tuple[str, str]:
        """``("rebased"|"conflict"|"no", detail)`` for a stale-base refusal.

        Only for a refusal the base actually explains. A clean rebase re-opens
        the gates; a CONFLICT is real work, and the producer is the only thing
        that knows how to keep both behaviours -- so it is reported as such
        rather than swallowed.
        """
        lowered = str(why or "").lower()
        if not any(marker in lowered for marker in self.STALE_BASE_MARKERS):
            return "no", ""
        repo = str(run.get("repo") or "")
        branch = str(run.get("branch") or "")
        try:
            from sharedagents_web.dev_sdlc import rebase_onto_base, remote_branch_sha

            # Before the rebase fetches: the lease must name the commit we
            # decided to replace, not whatever the fetch adopts on the way.
            expect = remote_branch_sha(repo, branch)
            ok, detail = rebase_onto_base(repo, branch)
        except Exception as error:  # noqa: BLE001
            log.warning("sdlc: rebase failed for %s: %s", run.get("run_id"), error)
            return "no", str(error)
        if not ok:
            log.info("sdlc: %s could not be rebased: %s", run.get("run_id"), detail)
            return ("conflict" if "conflict" in detail.lower() else "no"), detail
        pushed, push_detail = self.push_branch_safely(repo, branch, rewritten=True, expect_remote=expect)
        if not pushed:
            log.warning("sdlc: rebased %s but could not push: %s", run.get("run_id"), push_detail)
            return "no", push_detail
        return "rebased", detail

    def _already_landed(self, run: dict[str, Any]) -> bool:
        """True when the branch head is an ancestor of the base. Never raises."""
        from sharedagents_web.dev_sdlc import branch_already_landed

        try:
            return branch_already_landed(str(run.get("repo") or ""), str(run.get("branch") or ""))
        except Exception:  # noqa: BLE001 — an unanswerable question is a "no"
            return False

    @staticmethod
    def _landed_reconciliation_eligible(run: dict[str, Any]) -> bool:
        """Whether an ancestor branch can represent a lost merge result.

        Every newly-created worktree initially points at the base commit, so
        ancestry alone is not evidence that its requested change landed.  The
        lost-bookkeeping recovery this guard serves happens after a pull
        request was opened and merged; at that point ``pr_number`` has already
        been persisted.  Requiring it prevents a pristine run from skipping
        plan, design, code, tests, and review as an instant false merge.
        """
        return bool(run.get("pr_number"))

    #: How many laps the producer gets to make an UNPROVEN criterion provable
    #: before the run stops asking and the merge gate takes over.
    UNPROVEN_REWORK_LAPS = int(os.environ.get("SHAREDAGENTS_SDLC_UNPROVEN_LAPS") or 2)

    def _press_on_unproven(self, run: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
        """Send the run back while a criterion is merely UNPROVEN -- bounded.

        UNPROVEN means the suite passed and nothing showed those tests depend
        on the implementation: usually a runner the harness cannot ask for one
        test, an ESM module it cannot poison without breaking the import, or a
        criterion no isolable test names. That is fixable BY THE PRODUCER, so
        ask -- but only for a bounded number of laps, because some of it is
        environmental and a gate nobody can satisfy is worse than no gate.

        After the budget the stage is allowed through and the merge gate
        refuses instead, so a human decides rather than a loop deciding.
        """
        coverage = observed.get("coverage") or {}
        unproven = list(coverage.get("unproven") or [])
        if not observed.get("ok") or not unproven:
            return observed
        lap = int(run.get("lap") or 0)
        if lap >= self.UNPROVEN_REWORK_LAPS:
            observed["unproven_exhausted"] = True
            return observed
        listed = ", ".join(unproven)
        observed["ok"] = False
        observed["note"] = (
            f"the suite passes, but {listed} are UNPROVEN: nothing shows the "
            f"tests naming them depend on the implementation. Make each one "
            f"provable -- give it a test the runner can select by name "
            f"(node:test / pytest), keep the module importable (CommonJS, not "
            f"ESM-only), and assert on a value the implementation produces "
            f"rather than a constant. Previous note: " + str(observed.get("note") or "")
        )[:1200]
        return observed

    def _observe_branch(self, run: dict[str, Any]) -> dict[str, Any]:
        """Run the branch's own checks. Never raises; a crash is a failure."""
        from sharedagents_web.dev_sdlc import run_verify_checks

        try:
            return run_verify_checks(run)
        except Exception as error:  # noqa: BLE001
            log.warning("sdlc: could not observe %s: %s", run.get("run_id"), error)
            return {
                "ok": False,
                "ran": 0,
                "failed": 0,
                "transcript": "",
                "note": f"the checks could not be run: {error}",
            }

    def _record_observation(self, run: dict[str, Any], observed: dict[str, Any]) -> None:
        """Put what actually ran on the issue, passing or not."""
        status = "passed" if observed.get("ok") else "failed"
        detail = str(observed.get("note") or "")
        if observed.get("self_passing"):
            detail += " — the suite passes against an emptied implementation"
        unproven = list((observed.get("coverage") or {}).get("unproven") or [])
        if unproven:
            # A FIRST-CLASS status, so the merge gate can ask "is the newest
            # observation still unproven?" instead of sniffing prose.
            try:
                self.store.add_event(
                    str(run.get("run_id") or ""),
                    stage="tests",
                    status="observed_unproven",
                    note=("no evidence ties these to the implementation: " + ", ".join(unproven))[:500],
                )
            except Exception:  # noqa: BLE001
                log.debug("sdlc: could not record the unproven criteria")
        try:
            self.store.add_event(
                str(run.get("run_id") or ""),
                stage="tests",
                status=f"observed_{status}",
                note=detail[:500],
            )
        except Exception:  # noqa: BLE001 — bookkeeping must not fail the stage
            log.debug("sdlc: could not record the observation")
        self._post(
            run,
            stage="tests",
            status=status,
            note=f"observed behaviour: {detail}",
            evidence=str(observed.get("transcript") or "")[:8000],
        )

    def _apply_test_reviews(
        self,
        run: dict[str, Any],
        board: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Run verify A/B and write each verdict onto the conductor sdlc board.

        Jarvis stays on the OPEN tests gate. The tool CLEARS only after both
        reviewers agree PASS. This method never CLEARS as Jarvis.  A failed
        deterministic observation is returned to ``run_stage`` immediately;
        no review was launched and waiting on the still-open board would be a
        permanent deadlock.
        """
        sid = str(run.get("conductor_session_id") or "").strip()
        if not sid or not self.hooks.independent_reviews:
            return
        reviews = board.get("reviews") if isinstance(board.get("reviews"), dict) else {}
        generation_agent, _ = role_config(run, "generation")
        if independent_reviews_ready(reviews, generation_agent=generation_agent):
            self._record_board_review_gates(run, reviews)
            return
        pinned = self._read_head(run)
        notes: list[str] = []
        agreed_ok = False
        if not pinned:
            notes = [
                "verify-a: no HEAD SHA to bind the review to",
                "verify-b: no HEAD SHA to bind the review to",
            ]
        else:
            marker = {
                "status": "PASS",
                "note": str((board.get("gates") or {}).get("tests", {}).get("note") or ""),
            }
            # Run the branch's own checks BEFORE asking anyone's opinion, and
            # carry the transcript into the review as primary evidence. Two
            # things are settled here that no reviewer can settle: whether the
            # suite passes, and whether it passes for a reason. A suite that
            # still goes green against an emptied implementation is not
            # evidence of anything, and it used to read identically to one that
            # works.
            observed = self._press_on_unproven(run, self._observe_branch(run))
            marker["verify_transcript"] = str(observed.get("transcript") or "")
            self._record_observation(run, observed)
            if not observed.get("ok"):
                why = str(observed.get("note") or "the branch could not be run")
                # A real failing acceptance command is product feedback.  A
                # suite that ran green but lacked criterion attribution or
                # survived mutation is proof-authoring feedback.  Preserve
                # that ownership boundary so TEST never edits product code.
                rewind = "code" if int(observed.get("failed") or 0) else "tests"
                return {
                    "status": "FAIL",
                    "note": (
                        "the branch does not demonstrate its own behaviour: "
                        f"{why}. Observed:\n"
                        f"{observed.get('transcript') or '(nothing ran)'}"
                    ),
                    "verdict": True,
                    "rewind": rewind,
                    "observation_failed": True,
                }
            if observed.get("ok") and not board.get("drive_ready"):
                try:
                    self.hooks.mark_tests_ready(
                        run,
                        str(
                            observed.get("transcript")
                            or observed.get("note")
                            or "driver executed the declared Docker checks"
                        ),
                    )
                except Exception as error:  # noqa: BLE001 — fail closed below
                    observed["ok"] = False
                    observed["note"] = f"could not record Docker drive readiness: {error}"
            try:
                if not observed.get("ok"):
                    why = str(observed.get("note") or "the branch could not be run")
                    raise _NotObserved(
                        f"the branch does not demonstrate its own behaviour: {why}. "
                        f"Observed:\n{observed.get('transcript') or '(nothing ran)'}"
                    )
                agreed_ok, notes = self.hooks.independent_reviews("tests", run, marker)
            except _NotObserved as error:
                # Deterministic, and it is a real FAIL rather than an unusable
                # gate: the harness ran the branch and the branch did not hold
                # up. Nothing here is a reviewer being unreachable.
                notes = [f"verify-a: {error}", f"verify-b: {error}"]
                agreed_ok = False
            except Exception as error:  # noqa: BLE001
                log.warning("sdlc: test reviews failed to start: %s", error)
                notes = [
                    f"verify-a: {GATE_UNUSABLE} verify A failed: {error}",
                    f"verify-b: {GATE_UNUSABLE} verify B failed: {error}",
                ]
                agreed_ok = False
            after = self._read_head(run)
            if after != pinned:
                notes = list(notes) + [
                    f"verify-a: HEAD moved during review ({pinned[:12]} -> {after[:12]})",
                    f"verify-b: HEAD moved during review ({pinned[:12]} -> {after[:12]})",
                ]
                pinned = ""
                agreed_ok = False
        notes_a = [n[len("verify-a: ") :] for n in notes if str(n).startswith("verify-a: ")]
        notes_b = [n[len("verify-b: ") :] for n in notes if str(n).startswith("verify-b: ")]
        extra = [n for n in notes if not str(n).startswith("verify-a: ") and not str(n).startswith("verify-b: ")]
        blocked_a = (
            (agreed_ok is not True)
            or _review_notes_block(notes_a)
            or any(token in str(n).lower() for n in extra for token in ("did not agree", "both rejected"))
        )
        blocked_b = (
            (agreed_ok is not True)
            or _review_notes_block(notes_b)
            or any(token in str(n).lower() for n in extra for token in ("did not agree", "both rejected"))
        )
        # A combined fail with no per-role notes means both failed to decide.
        if extra and not notes_a and not notes_b:
            blocked_a = blocked_b = True
            notes_a = notes_b = extra
        if not pinned:
            blocked_a = blocked_b = True
        try:
            from sharedagents_agent.tools.impl.sdlc import apply_review
            from sharedagents_core.kernel.layout import sessions_root

            root = sessions_root()
            sha = pinned
            sid_a = self._gate_session(run, "tests", "validation")
            sid_b = self._gate_session(run, "tests", "verification")
            apply_review(
                root,
                sid,
                role="validation",
                verdict="FAIL" if blocked_a else "PASS",
                note="; ".join(notes_a) or ("PASS" if not blocked_a else "FAIL"),
                sha=sha,
                session_id=sid_a,
            )
            apply_review(
                root,
                sid,
                role="verification",
                verdict="FAIL" if blocked_b else "PASS",
                note="; ".join(notes_b) or ("PASS" if not blocked_b else "FAIL"),
                sha=sha,
                session_id=sid_b,
            )
            digest = sha or self._gate_digest(run, "tests")
            self._record_gate(
                run, "tests", "validation", ran=not blocked_a, notes=notes_a, digest=digest, session_id=sid_a
            )
            self._record_gate(
                run, "tests", "verification", ran=not blocked_b, notes=notes_b, digest=digest, session_id=sid_b
            )
            if not blocked_a and not blocked_b and sha:
                self._pin_reviewed_sha(run, sha)
        except Exception:  # noqa: BLE001 — board write must not kill the wait loop
            log.exception("sdlc: could not record test reviews on the conductor board")

    def _pin_reviewed_sha(self, run: dict[str, Any], sha: str) -> None:
        rid = str(run.get("run_id") or "")
        token = str(sha or "").strip()
        if rid and token:
            self.store.patch(rid, reviewed_sha=token)

    def _same_commit(self, left: str, right: str) -> bool:
        a, b = (left or "").strip().lower(), (right or "").strip().lower()
        return bool(a and b and a == b)

    def _tests_still_at_review(self, run: dict[str, Any], gate: dict[str, Any]) -> tuple[bool, str]:
        agreed = str(gate.get("sha") or "").strip()
        head = self._read_head(run)
        if not agreed:
            return False, "tests CLEAR has no A/B SHA"
        if not head:
            return False, "could not read HEAD to bind the A/B SHA"
        if not self._same_commit(agreed, head):
            return False, (f"HEAD moved after A/B ({agreed[:12]} -> {head[:12]}); reviews must rerun")
        return True, agreed

    def _invalidate_stale_tests(self, run: dict[str, Any], why: str) -> None:
        rid = str(run.get("run_id") or "")
        if rid:
            self.store.patch(rid, reviewed_sha="")
        sid = str(run.get("conductor_session_id") or "").strip()
        if not sid:
            return
        try:
            from sharedagents_agent.tools.impl.sdlc import (
                _alias_board,
                load_board,
                reopen_tests,
                resolve_board_path,
            )
            from sharedagents_core.kernel.layout import sessions_root

            root = sessions_root()
            path = resolve_board_path(root, sid)
            board = load_board(path)
            reopen_tests(board, note=why)
            extras = list(board.get("session_aliases") or [])
            extras.append(sid)
            _alias_board(path, board, extras)
        except Exception as error:  # noqa: BLE001
            log.warning("sdlc: could not reopen tests after stale A/B: %s", error)
        if rid:
            self.store.add_event(
                rid,
                stage="tests",
                status="invalidated",
                note=why,
            )
            self.store.add_event(
                rid,
                stage="deploy",
                status="invalidated",
                note=why,
            )

    def _record_board_review_gates(self, run: dict[str, Any], reviews: dict[str, Any]) -> None:
        """Copy already-decided A/B board rows onto the event ledger.

        Skip a role that already has a ``gate_decided`` so a retried tests
        wait does not append a second empty verdict over a real session.
        """
        run_id = str(run.get("run_id") or "")
        if not run_id:
            return
        existing = {
            str(event.get("gate_kind") or "")
            for event in self.store.events(run_id)
            if event.get("status") == "gate_decided"
            and str(event.get("gate_kind") or "") in {"validation", "verification"}
        }
        sha = self._read_head(run)
        for kind in ("validation", "verification"):
            if kind in existing:
                continue
            row = reviews.get(kind) if isinstance(reviews.get(kind), dict) else {}
            sid = usable_session_id(row.get("session_id")) or self._gate_session(run, "tests", kind)
            verdict = str(row.get("verdict") or "").upper()
            self._record_gate(
                run,
                "tests",
                kind,
                ran=verdict == "PASS",
                notes=[str(row.get("note") or "")],
                digest=sha or str(row.get("sha") or ""),
                session_id=sid,
            )
        sha_a = str((reviews.get("validation") or {}).get("sha") or "")
        sha_b = str((reviews.get("verification") or {}).get("sha") or "")
        if (
            str((reviews.get("validation") or {}).get("verdict") or "").upper() == "PASS"
            and str((reviews.get("verification") or {}).get("verdict") or "").upper() == "PASS"
            and sha_a
            and self._same_commit(sha_a, sha_b)
            and (not sha or self._same_commit(sha_a, sha))
        ):
            self._pin_reviewed_sha(run, sha_a)

    def run_stage(
        self,
        run: dict[str, Any],
        stage: str,
        *,
        feedback: str = "",
        attempt: int = 1,
        previous: str = "",
        previous_attempt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch one stage and block until it reports. Returns the marker."""
        run_id = str(run["run_id"])
        lap = int(run.get("lap") or 0)
        # Every attempt starts clean. The branch/worktree is the durable handoff
        # between stages; conversational context is deliberately not. This
        # prevents a retry from inheriting stale requirements, conclusions, or
        # a previous turn's test strategy.
        sid = str(run.get("conductor_session_id") or "").strip()
        if not sid:
            return {
                "status": "FAIL",
                "note": "no Jarvis conductor session — cannot post a /goal step",
                "not_validated": True,
            }
        fields = {"review_session_id": sid} if stage == "deploy" else {"author_session_id": sid}
        self.store.patch(
            run_id,
            status="running",
            stage=stage,
            stage_status="started",
            session_id=sid,
            last_error="",
            stage_dispatch_turn=self._turn_identity(sid),
            **fields,
        )
        # Keep this invocation's projection in sync with the durable row. The
        # final event is written from this object after the turn completes.
        run.update(fields)
        run["session_id"] = sid
        # PREFLIGHT: is there anything to work with? Cheap, and it runs first so
        # a stage with no input costs no agent turn.
        try:
            absent = list(self.hooks.preflight_stage(stage, run))
        except Exception as error:  # noqa: BLE001
            absent = [f"preflight failed: {type(error).__name__}: {error}"]
        if absent:
            return {
                "status": "FAIL",
                "note": f"`{stage}` cannot start: " + "; ".join(absent),
                "not_validated": True,
            }
        if stage == "deploy":
            from sharedagents_web.dev_sdlc import lsp_code_errors

            lsp = lsp_code_errors(run)
            if not lsp.get("ok"):
                return {
                    "status": "FAIL",
                    "verdict": True,
                    "note": "deploy blocked by LSP: " + "; ".join(str(item) for item in lsp.get("blockers") or []),
                }

        stage_start_head = self._read_head(run)

        # Jarvis owns the board, but a fresh build-role session owns the
        # artifact. Routing this in the driver is a safety property: a local
        # model can narrate that it will delegate and end the turn without a
        # tool call. Retrying that same conductor only repeats the deadlock.
        _generation_agent, generation_model = role_config(run, "generation")
        stage_model = _stage_selected_model(stage, generation_model)
        worker_agent = STAGE_EXECUTOR_AGENT
        worktree = str(run.get("worktree") or "")
        if self._hook_accepts(self.hooks.new_session_id, "agent"):
            worker_sid = self.hooks.new_session_id(worktree, agent=worker_agent)
        else:
            worker_sid = self.hooks.new_session_id(worktree)
        if not worker_sid:
            return {
                "status": "FAIL",
                "note": f"could not create the `{worker_agent}` stage session",
                "dispatch_error": True,
            }
        sid = str(worker_sid)
        fields = {"review_session_id": sid} if stage == "deploy" else {"author_session_id": sid}
        before_turn = self._turn_identity(sid)
        self.store.patch(
            run_id,
            session_id=sid,
            stage_dispatch_turn=before_turn,
            **fields,
        )
        run.update(fields)
        run["session_id"] = sid
        self.store.add_event(
            run_id,
            stage=stage,
            status="started",
            lap=lap,
            executor_agent=worker_agent,
            executor_model=stage_model,
            executor_session_id=sid,
        )
        self.store.patch(run_id, stage_executor_model=stage_model)
        run["stage_executor_model"] = stage_model
        # P1: the budget the worker is TOLD must be the budget the watchdog
        # enforces. An operator override on the run row wins over the engine
        # default (P5's "raise the stage budget and resume").
        first_write_timeout = {
            "plan": PLAN_FIRST_WRITE_TIMEOUT_S,
            "code": CODE_FIRST_WRITE_TIMEOUT_S,
            "tests": TEST_FIRST_WRITE_TIMEOUT_S,
        }.get(stage)
        budget_s = float(run.get("first_artifact_budget_s") or 0.0) or float(first_write_timeout or 0.0)
        ceiling_s = float(run.get("stage_ceiling_s") or 0.0) or STAGE_CEILING_S
        prompt = stage_prompt(
            stage=stage,
            headline=str(run.get("headline") or ""),
            kind=str(run.get("kind") or "feature"),
            repo=str(run.get("repo") or ""),
            branch=str(run.get("branch") or ""),
            requirements=list(run.get("requirements") or []),
            criteria=list(run.get("criteria") or []),
            issue_number=run.get("issue_number") or "",
            issue_url=str(run.get("issue_url") or ""),
            run_id=run_id,
            worktree=str(run.get("worktree") or ""),
            feedback=feedback,
            previous=previous,
            attempt=attempt,
            lap=lap,
            detail=str(run.get("detail") or ""),
            transcript=list(run.get("transcript") or []),
            head_sha=stage_start_head,
            fuse_plan_design=bool(run.get("fuse_plan_design", True)),
            first_artifact_budget_s=budget_s if first_write_timeout else 0.0,
            stage_ceiling_s=ceiling_s if first_write_timeout else 0.0,
            previous_attempt=previous_attempt,
        )
        # P8: the named parent must exist before the worker starts, so a
        # stray `plan/<something>/` cannot be mistaken for the artifact.
        paths = stage_artifact_paths(
            stage,
            str(run.get("branch") or ""),
            fuse_plan_design=bool(run.get("fuse_plan_design", True)),
        )
        layout_notes = prepare_artifact_parents(str(run.get("worktree") or ""), paths)
        if layout_notes:
            log.warning("sdlc: artifact layout for %s: %s", run_id, "; ".join(layout_notes))
        payload, status = self.hooks.dispatch(
            sid,
            prompt,
            worker_agent,
            _stage_dispatch_model(stage_model, worker_agent),
        )
        if status >= 400:
            detail = str((payload or {}).get("error") or f"HTTP {status}")
            return {"status": "FAIL", "note": f"dispatch failed: {detail}", "dispatch_error": True}

        stage_started = self.hooks.now()
        deadline = stage_started + self.timing.stage_timeout
        first_write_deadline = stage_started + first_write_timeout if first_write_timeout is not None else None
        # P1: one steer at 60% of the artifact budget; P2: the unconditional
        # ceiling. Both exist so the deadline below is a decision, not a hope.
        steer_at = stage_started + STEER_AT_FRACTION * budget_s if budget_s else None
        ceiling_deadline = stage_started + ceiling_s
        stage_label = "TEST" if stage == "tests" else stage.upper()
        saw_stage_write = False
        steered = False
        reviews_launched = False
        dispatched_turn = before_turn
        aborted_subtree_quiet_polls = 0
        # A stage that "ends" seconds after dispatch has not run; what the
        # check is seeing is the PREVIOUS turn on the same session -- often one
        # aborted by a restart. The turn identity degrades to a timestamp when
        # `turn_ended` carries no physical id, so it cannot always tell them
        # apart. A short dwell can: a real turn takes minutes, and the stall it
        # exists to catch was still caught in 73 seconds.
        while True:
            latest = self.store.get(run_id)
            if not latest or str(latest.get("status") or "") == "aborted":
                # Delete tombstones before cancelling the worker. Never
                # overwrite that terminal decision or start a retry from this
                # turn's stale run snapshot.
                # F4: the abort must also stop the worker, or it keeps
                # burning tokens with nobody watching the clock.
                confirmed, idle_at = self._cancel_worker_confirmed(sid)
                result = self._stage_failure(
                    "run was aborted while the stage was active"
                    + ("" if confirmed else "; worker cancellation was not confirmed"),
                    cancellation_confirmed=confirmed,
                    cancellation_idle_at=idle_at,
                )
                result["run_aborted"] = True
                return result
            if first_write_deadline is not None:
                now = self.hooks.now()
                if not saw_stage_write:
                    # An artifact (committed or working tree) is the strongest
                    # progress signal; check it every poll so a late writer is
                    # never mistaken for a stalled one.
                    try:
                        saw_stage_write = bool(self.hooks.changed_paths_since(run, stage_start_head))
                    except Exception as error:  # noqa: BLE001 — fail closed
                        try:
                            confirmed, idle_at = self._cancel_worker_confirmed(sid)
                        except Exception:  # noqa: BLE001
                            confirmed, idle_at = False, 0.0
                        return self._stage_failure(
                            f"{stage_label} first-write watchdog could not inspect the branch: {error}",
                            no_progress=True,
                            cancellation_confirmed=confirmed,
                            cancellation_idle_at=idle_at,
                            run=run,
                            stage=stage,
                        )
                if not saw_stage_write:
                    now = self.hooks.now()
                    # P1: tell the worker its clock while it can still act.
                    if steer_at is not None and not steered and now >= steer_at:
                        steered = True
                        minutes = max(1, int(round((first_write_deadline - now) / 60.0)))
                        text = (
                            f"No artifact yet. Commit the smallest complete "
                            f"{stage_label} artifact now; you have ~{minutes} "
                            "minutes."
                        )
                        try:
                            self.hooks.steer(sid, text)
                        except Exception:  # noqa: BLE001 — the nudge is advisory
                            log.debug("sdlc: could not steer %s for %s", sid, run_id)
                    # P2: cancel on SILENCE, not on mere absence of an
                    # artifact. The failed run's attempts ran 30+ productive
                    # repository commands and every one of them scored zero
                    # under the old artifact-only predicate.
                    try:
                        activity = float(self.hooks.last_activity_at(run) or 0.0)
                    except Exception:  # noqa: BLE001 — unknown is not progress
                        activity = 0.0
                    last_activity = max(stage_started, activity)
                    try:
                        usage = dict(self.hooks.session_usage(sid) or {})
                    except Exception:  # noqa: BLE001 — unknown spend is not progress
                        usage = {}
                    reasoning = int(usage.get("reasoning_tokens") or 0)
                    tools = int(usage.get("tool_calls") or 0)
                    spent = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
                    if spent:
                        try:
                            self.store.patch(run_id, stage_tokens=spent)
                        except Exception:  # noqa: BLE001 — the clock is advisory
                            pass
                    if not saw_stage_write and tools == 0 and reasoning >= REASONING_STARVATION_TOKENS:
                        confirmed, idle_at = self._cancel_worker_confirmed(sid)
                        result = self._stage_failure(
                            f"{stage_label} burned {reasoning} reasoning "
                            "tokens with no tool call; the starved round was "
                            "stopped for a shorter retry",
                            cancellation_confirmed=confirmed,
                            cancellation_idle_at=idle_at,
                            run=run,
                            stage=stage,
                            reasoning_starved=True,
                        )
                        result["digest"] = self._store_rework(
                            run, sid, result["note"], attempt=attempt, stage_start_head=stage_start_head
                        )
                        return result
                    if spent >= STAGE_TOKEN_BUDGET:
                        confirmed, idle_at = self._cancel_worker_confirmed(sid)
                        result = self._stage_failure(
                            f"{stage_label} hit the {STAGE_TOKEN_BUDGET} "
                            "token stage budget; raise the budget, edit the "
                            "scope, or abort",
                            no_progress=True,
                            cancellation_confirmed=confirmed,
                            cancellation_idle_at=idle_at,
                            run=run,
                            stage=stage,
                            budget_breach=True,
                        )
                        result["digest"] = self._store_rework(
                            run, sid, result["note"], attempt=attempt, stage_start_head=stage_start_head
                        )
                        return result
                    if now >= ceiling_deadline:
                        confirmed, idle_at = self._cancel_worker_confirmed(sid)
                        result = self._stage_failure(
                            f"{stage_label} hit the {int(ceiling_s)}s stage "
                            "ceiling with no "
                            + (
                                {"plan": "plan brief", "code": "product change", "tests": "proof artifact"}.get(
                                    stage, "stage artifact"
                                )
                            )
                            + " committed; the stage was cancelled",
                            no_progress=True,
                            cancellation_confirmed=confirmed,
                            cancellation_idle_at=idle_at,
                            run=run,
                            stage=stage,
                        )
                        result["digest"] = self._store_rework(
                            run, sid, result["note"], attempt=attempt, stage_start_head=stage_start_head
                        )
                        return result
                    if now - last_activity >= IDLE_NO_PROGRESS_S:
                        confirmed, idle_at = self._cancel_worker_confirmed(sid)
                        result = self._stage_failure(
                            f"{stage_label} made no committed or working-tree "
                            + (
                                {
                                    "plan": "plan brief change",
                                    "code": "product change",
                                    "tests": "proof artifact change",
                                }.get(stage, "stage artifact change")
                            )
                            + f" and no tool activity for "
                            f"{int(now - last_activity)}s (silent longer than "
                            f"{int(IDLE_NO_PROGRESS_S)}s); the design loop was "
                            "cancelled for a fresh attempt"
                            + ("" if confirmed else " (worker cancellation was not confirmed)"),
                            no_progress=True,
                            cancellation_confirmed=confirmed,
                            cancellation_idle_at=idle_at,
                            run=run,
                            stage=stage,
                        )
                        result["digest"] = self._store_rework(
                            run, sid, result["note"], attempt=attempt, stage_start_head=stage_start_head
                        )
                        return result
            board = {}
            try:
                board = self.hooks.sdlc_board(run) or {}
            except Exception:  # noqa: BLE001
                board = {}
            gates = board.get("gates") if isinstance(board.get("gates"), dict) else {}
            gate = gates.get(stage) if isinstance(gates.get(stage), dict) else {}
            gstatus = str(gate.get("status") or "")
            opened = str(board.get("open") or "")
            if gstatus == "cleared":
                if stage == "tests":
                    ok, detail = self._tests_still_at_review(run, gate)
                    if not ok:
                        self._invalidate_stale_tests(run, detail)
                        return {
                            "status": "FAIL",
                            "note": detail,
                            "verdict": True,
                            "rewind": "tests",
                        }
                    self._pin_reviewed_sha(run, detail)
                marker = {
                    "status": "PASS",
                    "note": str(gate.get("note") or "conductor cleared"),
                    "evidence": str(board.get("drive_evidence") or gate.get("note") or ""),
                    "drive_evidence": str(board.get("drive_evidence") or ""),
                    "verdict": True,
                }
                return self._verified(stage, run, marker)
            if gstatus in {"looped", "blocked"} or board.get("blocked"):
                return {
                    "status": "FAIL",
                    "note": str(gate.get("note") or board.get("blocked") or "looped"),
                    "verdict": True,
                    "rewind": opened if opened in STAGES or opened == "clarify" else "",
                    "definition_invalid": opened in {"plan", "clarify", "design", "contract"},
                }
            try:
                interaction = str(self.hooks.pending_interaction(sid) or "").strip()
            except Exception:  # noqa: BLE001 — unknown is not evidence of a waiter
                interaction = ""
            if interaction in {"question", "permission"}:
                # P6: the policy boundary still cancels, but now it reports
                # what the cancel actually achieved.
                confirmed, idle_at = self._cancel_worker_confirmed(sid)
                kind = "operator question" if interaction == "question" else "tool permission"
                result = self._stage_failure(
                    f"unattended `{stage}` worker requested {kind}; "
                    "human input is prohibited after ratification"
                    + ("; worker cancelled" if confirmed else "; worker cancellation could not be confirmed"),
                    cancellation_confirmed=confirmed,
                    cancellation_idle_at=idle_at,
                )
                result["verdict"] = True
                result["policy_violation"] = True
                return result
            # The conductor's turn ENDING without a verdict is a finished stage
            # that decided nothing -- an agent that narrated its way to the end,
            # or whose child was killed under it. Waiting out the full
            # stage_timeout after that is 90 minutes of polling a session that
            # will never write again. Notice it, and let the attempt loop
            # redispatch straight away.
            # NOT after `tests` has gone READY: there the conductor is SUPPOSED
            # to finish its turn and wait while verify A and verify B decide,
            # and failing the stage then would kill a review that is working.
            if self.hooks.now() > deadline:
                return {
                    "status": "FAIL",
                    "note": f"goal timed out after {int(self.timing.stage_timeout)}s without sdlc CLEAR on `{stage}`",
                    "timeout": True,
                }
            handed_off = reviews_launched or bool(board.get("drive_ready"))
            # Every artifact attempt now owns a freshly-created worker
            # session, so its completed turn can never be the previous stage's
            # stale turn. The old conductor reuse needed a 60-second dwell;
            # retaining it here would manufacture a minute of apparent idle
            # time after every worker had already finished.
            settled = True
            if not handed_off and settled:
                # A parent can end as aborted while descendants it launched
                # are still auditing the branch. Dispatching a replacement at
                # that instant creates two writers for one stage and races the
                # child reports. The live hook is tree-aware: wait until the
                # complete stage subtree is quiet before interpreting the
                # parent's final event.
                try:
                    tree_live = bool(self.hooks.is_live(sid))
                except Exception:  # noqa: BLE001 — unknown liveness is not proof
                    tree_live = True
                if tree_live:
                    aborted_subtree_quiet_polls = 0
                    self.hooks.sleep(self.timing.poll_seconds)
                    continue
                identity, text, turn_error = self._final_message(sid)
                if turn_error:
                    aborted_subtree_quiet_polls += 1
                    if aborted_subtree_quiet_polls < ABORTED_SUBTREE_QUIET_POLLS:
                        self.hooks.sleep(self.timing.poll_seconds)
                        continue
                else:
                    aborted_subtree_quiet_polls = 0
                if identity and identity != dispatched_turn:
                    current_head = self._read_head(run)
                    committed_after_start = bool(stage_start_head and current_head and current_head != stage_start_head)
                    # A turn that ended may still have DONE the work: the plan
                    # stage commits its brief and then finishes without calling
                    # `sdlc submit`. Failing on the missing call throws away a
                    # committed artifact and redispatches the same stage until
                    # its attempts run out. Adopt the marker if the turn left
                    # one -- it is still verified against the branch like any
                    # other result, so this accepts work, never a claim.
                    if stage == "tests" and (not turn_error or committed_after_start):
                        # The producer is done but never submitted READY. It
                        # may be adopted only after the BRANCH proves that it
                        # contains the complete test/isolation artifact set.
                        # Previously any finished TEST turn launched review;
                        # a failed observation then set drive_ready, so retry 2
                        # was reviewed and rewound while its worker was still
                        # running. The synthetic PASS asks trace_stage about
                        # artifacts rather than trusting a worker's marker.
                        reported = parse_stage_marker(text, stage=stage) if text else None
                        artifact_claim = {
                            "status": "PASS",
                            "note": str((reported or {}).get("note") or "tests worker finished"),
                            "evidence": str((reported or {}).get("evidence") or ""),
                        }
                        try:
                            touched = list(self.hooks.changed_paths_since(run, stage_start_head))
                        except Exception as error:  # noqa: BLE001 — fail closed
                            return {
                                "status": "FAIL",
                                "verdict": True,
                                "rewind": "code",
                                "stage_boundary_violation": True,
                                "note": f"TEST ownership check failed: {error}",
                            }
                        violations = [name for name in touched if not is_test_stage_path(name)]
                        if violations:
                            return {
                                "status": "FAIL",
                                "verdict": True,
                                "rewind": "code",
                                "stage_boundary_violation": True,
                                "note": (
                                    "TEST changed product-owned paths: "
                                    + ", ".join(violations[:20])
                                    + ". Leave the failing proof committed; a "
                                    "fresh CODE worker must fix the product."
                                ),
                            }
                        try:
                            traced, why = self.hooks.trace_stage(stage, run, artifact_claim)
                        except Exception as error:  # noqa: BLE001 — fail closed
                            traced, why = False, f"test artifact trace failed: {error}"
                        if not traced:
                            prior = str((reported or {}).get("note") or "").strip()
                            return {
                                "status": "FAIL",
                                "note": ((prior + " — ") if prior else "") + why,
                                "verdict": True,
                                "unverified": True,
                            }
                        log.info(
                            "sdlc: tests turn ended with committed proof "
                            "artifacts for %s; starting independent reviews",
                            run_id,
                        )
                        review_result = self._apply_test_reviews(run, board)
                        if review_result is not None:
                            return review_result
                        reviews_launched = True
                        self.hooks.sleep(self.timing.poll_seconds)
                        continue
                    marker = parse_stage_marker(text, stage=stage) if text else None
                    # NEVER for `tests`. That gate clears only when verify A
                    # and verify B both agree, and adopting a silent turn there
                    # produced a PASS with no reviewed commit -- which the merge
                    # gate then refused, correctly, as "no reviewed commit".
                    # Letting tests pass unreviewed is the exact hazard the
                    # independent review exists to prevent.
                    if marker is None and stage != "tests" and (not turn_error or committed_after_start):
                        # No marker, but the turn is over and it may still have
                        # committed the artifact -- the code stage writes and
                        # commits the implementation and then simply stops.
                        # Propose a PASS and let `_verified` interrogate the
                        # BRANCH: trace check first, then the semantic gate. A
                        # stage that produced nothing fails exactly as before,
                        # so this accepts work rather than a claim.
                        marker = {
                            "status": "PASS",
                            "note": (
                                f"`{stage}` finished without calling sdlc"
                                + (f" ({turn_error[:160]})" if turn_error else "")
                                + "; judged on what it committed"
                            ),
                        }
                    if marker and (not turn_error or committed_after_start):
                        marker.setdefault("evidence", "")
                        marker["verdict"] = True
                        log.info("sdlc: adopted a finished %s turn for %s that never submitted", stage, run_id)
                        return self._verified(stage, run, marker)
                    note = f"the `{stage}` turn ended without an sdlc verdict" + (
                        f": {turn_error[:300]}" if turn_error else ""
                    )
                    result = self._stage_failure(note)
                    result["turn_error"] = bool(turn_error)
                    # P3: this is the auto-resume path — the run blocks and
                    # the supervisor restarts it, so the next worker needs
                    # what this one learned.
                    result["digest"] = self._store_rework(
                        run, sid, note, attempt=attempt, stage_start_head=stage_start_head
                    )
                    return result
            if self.hooks.now() > deadline:
                return {
                    "status": "FAIL",
                    "note": f"goal timed out after {int(self.timing.stage_timeout)}s without sdlc CLEAR on `{stage}`",
                    "timeout": True,
                }
            self.hooks.sleep(self.timing.poll_seconds)

    # -- publish ---------------------------------------------------------
    @staticmethod
    def _hook_accepts(hook: Any, name: str) -> bool:
        """Does *hook* take keyword *name*? Asked, not discovered by failing.

        UNKNOWABLE counts as yes. Some callables (a `Mock(spec=...)`, a C
        builtin) have no readable signature, and answering "no" there would
        silently drop `rewritten` and push without the force flag -- a
        rejection, or a plain push where a force was required. Guessing yes
        fails LOUDLY instead: a hook that really cannot take it raises
        TypeError, which is returned as the error it is.
        """
        import inspect

        try:
            params = inspect.signature(hook).parameters
        except (TypeError, ValueError):
            return True
        found = params.get(name)
        if found is not None:
            # Positional-only cannot be passed by keyword, whatever it is called.
            return found.kind is not inspect.Parameter.POSITIONAL_ONLY
        return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def push_branch_safely(
        self, repo: str, branch: str, *, rewritten: bool = False, expect_remote: str = ""
    ) -> tuple[bool, str]:
        """``rewritten=True`` after a REBASE: the ordinary push is a
        non-fast-forward against the published PR branch and is rejected
        forever, so every retry replays the same rejection.

        The hook's signature is INSPECTED rather than probed by catching
        TypeError -- catching it would also swallow a TypeError raised from
        inside a correct hook, and then push a second time without the flag.
        """
        try:
            extra: dict[str, Any] = {}
            if rewritten and self._hook_accepts(self.hooks.push_branch, "rewritten"):
                extra["rewritten"] = True
                if self._hook_accepts(self.hooks.push_branch, "expect_remote"):
                    extra["expect_remote"] = expect_remote
            return self.hooks.push_branch(repo, branch, **extra)
        except Exception as error:  # noqa: BLE001
            return False, f"{type(error).__name__}: {error}"

    def open_pull_request(self, run: dict[str, Any]) -> dict[str, Any]:
        """Open exactly one PR. Reuses an existing one rather than duplicating."""
        run_id = str(run["run_id"])
        if run.get("merged"):
            return {"ok": True, "reused": True, "already_merged": True}
        if run.get("pr_number"):
            return {"ok": True, "reused": True}
        # The stages commit locally; nothing has published the branch yet, and a
        # PR against an unpublished head fails as "target couldn't be found".
        pushed, detail = self.push_branch_safely(
            str(run.get("repo") or ""),
            str(run.get("branch") or ""),
        )
        if not pushed:
            self.store.patch(
                run_id, status="blocked", blocked_stage="pr", last_error=f"could not push the branch: {detail}"
            )
            self._post(run, stage="pr", status="failed", note=f"could not push `{run.get('branch')}`: {detail}")
            return {"ok": False, "error": detail}
        number = run.get("issue_number")
        body_lines = [
            f"Fixes #{number}" if number else "",
            "",
            "## Clearing evidence (for test)",
            *[f"- [x] {text}" for text in (run.get("criteria") or [])],
            "",
            f"_Opened by the Dev tab SDLC pipeline (`{run_id}`) after `{STAGE_TAGS['deploy']}` passed._",
        ]
        try:
            pr = self.hooks.forge.open_pull(
                str(run.get("repo") or ""),
                title=str(run.get("headline") or run_id),
                head=str(run.get("branch") or ""),
                body="\n".join(line for line in body_lines if line is not None),
            )
        except Exception as exc:  # noqa: BLE001
            self.store.patch(run_id, status="blocked", blocked_stage="pr", last_error=f"pr failed: {exc}")
            self._post(run, stage="pr", status="failed", note=str(exc))
            return {"ok": False, "error": str(exc)}
        pr_number = pr.get("number")
        if not pr_number:
            detail = str(pr.get("error") or "forge returned no PR number")
            self.store.patch(run_id, status="blocked", blocked_stage="pr", last_error=detail)
            self._post(run, stage="pr", status="failed", note=detail)
            return {"ok": False, "error": detail}
        url = str(pr.get("html_url") or "")
        self.store.patch(
            run_id,
            status="pr_open",
            stage="pr",
            stage_status="passed",
            pr_number=int(pr_number),
            pr_url=url,
        )
        fresh = self.store.get(run_id) or run
        self._post(
            fresh,
            stage="pr",
            status="passed",
            note=f"PR #{pr_number} opened — {url}" if url else f"PR #{pr_number} opened",
        )
        return {"ok": True, "pr_number": int(pr_number), "pr_url": url}

    def merge_when_green(self, run_id: str) -> dict[str, Any]:
        """Merge after independent verify A/B + LSP. Never force."""
        from sharedagents_web.dev_sdlc import STAGE_TAGS, _usable_pull, apply_merge
        from sharedagents_web.forge_client import get_pull

        run = self.store.get(run_id) or {}
        repo = str(run.get("repo") or "")
        pr_number = run.get("pr_number")
        # Publish the reviewed commit BEFORE asking to merge. The PR is opened
        # once, but a rework lap moves the branch on afterwards -- so the head
        # the forge would merge stayed at the commit that existed when the PR
        # was created, and the release gate refused with "the review approved
        # `8331772` but the pull request head is now `a59b5a0`". It was right
        # to: what was reviewed was not what would land. Pushing makes them the
        # same commit instead of arguing about it.
        pushed, detail = self.push_branch_safely(repo, str(run.get("branch") or ""))
        if not pushed:
            log.warning("sdlc: could not refresh the PR head for %s: %s", run_id, detail)
        last_ok: dict[str, Any] | None = None
        if repo and pr_number:
            for _ in range(8):
                status, body = get_pull(repo, int(pr_number))
                usable = _usable_pull(body) if status == 200 else None
                if usable is not None:
                    last_ok = usable
                    if usable.get("mergeable") is True:
                        break
                else:
                    last_ok = None
                self.hooks.sleep(2)
        result = apply_merge(
            run_id,
            method="merge",
            force=False,
            actor="pipeline",
            pull=last_ok,
            store=self.store,
        )
        fresh = self.store.get(run_id) or run
        if not result.get("ok"):
            note = str(result.get("error") or "merge blocked")
            blockers = result.get("blockers") or []
            if blockers:
                note = note + ": " + "; ".join(str(item) for item in blockers)
            return self._handle_merge_refusal(run_id, note)
        who = f"PR #{result.get('pr_number')} merged by pipeline after independent verify A/B + LSP"
        self._post(fresh, stage="merged", status="passed", note=who)
        try:
            self.hooks.forge.label(
                repo,
                int(fresh.get("issue_number") or 0),
                [STAGE_TAGS["merged"]],
            )
        except Exception:  # noqa: BLE001
            pass
        self.deploy_as_authorised(run_id)
        return result

    def deploy_as_authorised(self, run_id: str) -> dict[str, Any]:
        """Run the deploy the operator authorised at ratification, if any.

        This is what makes the human's involvement end at the plan phase. The
        target was chosen before any code existed and recorded on the run, so
        the pipeline is carrying out a decision rather than making one; a run
        that named no target is untouched and stays operator-driven exactly as
        before.

        The same refusals apply as on the HTTP route -- notably, work that got
        past a gate which decided nothing is never deployed. This never forces.
        """
        run = self.store.get(run_id) or {}
        target = str(run.get("deploy_target") or "").strip()
        if not target or target == "none":
            return {"ok": True, "skipped": True}
        if str(run.get("deploy_status") or ""):
            return {"ok": True, "skipped": True}  # already launched or refused
        try:
            from sharedagents_web.routes.dev import launch_deploy

            result = launch_deploy(run_id, target, forced=False, actor="pipeline")
        except Exception as error:  # noqa: BLE001 — a merged PR is not undone by this
            log.warning("sdlc: authorised deploy failed for %s: %s", run_id, error)
            self.store.patch(run_id, deploy_status=f"failed: {error}")
            return {"ok": False, "error": str(error)}
        fresh = self.store.get(run_id) or run
        if result.get("ok"):
            self._post(
                fresh,
                stage="deployed",
                status="passed",
                note=f"deploy `{target}` launched as `{result.get('unit')}` — authorised at ratification",
            )
        else:
            self._post(fresh, stage="deployed", status="failed", note=str(result.get("error") or "deploy refused"))
        return result

    # -- the whole run ---------------------------------------------------
    def _adopt_orphaned_stage(self, run: dict[str, Any], stage: str) -> dict[str, Any] | None:
        """Claim a stage that finished while no driver was watching.

        The driver thread dies with the web process. If it died mid-stage the
        agent kept working in the gateway and may well have reported;
        re-dispatching would spend that stage twice and confuse the agent with
        a repeated instruction. Adopt the marker instead -- but only for a
        stage whose last recorded event is `started`, so this can never pick up
        a marker from an earlier lap.
        """
        last = None
        for event in self.store.events(str(run["run_id"])):
            # Provenance recovery is append-only audit metadata, not a stage
            # transition. It may be written after ``started`` while the agent
            # keeps running outside a restarted Web process; treating it as
            # the last lifecycle event prevents adoption and redispatches the
            # already-finished stage.
            if canonical_stage(str(event.get("stage"))) == stage and str(event.get("status") or "") not in {
                "gate_session_linked",
                "gate_session_recovery_checked",
            }:
                last = event
        if not last or str(last.get("status")) != "started":
            return None
        sid = str(
            run.get("review_session_id")
            if stage == "deploy"
            else run.get("author_session_id") or run.get("session_id") or ""
        )
        if not sid:
            return None
        identity, text, error = self._final_message(sid)
        if not identity or identity == str(run.get("stage_dispatch_turn") or ""):
            # Nothing has completed since this stage was dispatched, so anything
            # on the session belongs to an earlier one.
            return None
        if error:
            return {
                "status": "FAIL",
                "note": f"the stage turn failed: {error[:300]}",
                "evidence": text[-2000:],
                "turn_error": True,
            }
        marker = parse_stage_marker(text, stage=stage)
        if not marker:
            return None
        # An adopted result is still a stage result: it carries a verdict, and
        # its claim is verified against the branch like any other. Skipping this
        # let an unverified PASS advance, and left an adopted review FAIL
        # without `verdict` so it blocked instead of starting a rework lap.
        marker.setdefault("evidence", "")
        marker["verdict"] = True
        return self._verified(stage, run, marker)

    #: Stages a rework lap must redo. A deploy-readiness rejection sends the run back to
    #: `code` -- it did not reject the plan or the design -- so those still
    #: stand on lap 1 having passed on lap 0. Keeping this list next to the loop that
    #: implements the same rule stops the two from drifting apart.
    REWORK_STAGES = ("code", "tests", "deploy")

    #: Validation of stage N refused the artifact stage N-1 produced. Loop back
    #: to that producer instead of blocking for the operator. ``plan`` has no
    #: producer in this pipeline (its input is the ratified contract).
    PREVIOUS_STAGE = {
        "design": "plan",
        "code": "design",
        "tests": "code",
        "deploy": "tests",
    }

    def _pipeline(self, run: dict[str, Any] | None) -> tuple[str, ...]:
        return stages_for_run(run)

    def _previous_producer(self, run: dict[str, Any] | None, stage: str) -> str | None:
        pipe = self._pipeline(run)
        if stage not in pipe:
            return None
        index = pipe.index(stage)
        return pipe[index - 1] if index else None

    def _input_rework_count(self, run_id: str, stage: str) -> int:
        events = self.store.events(run_id)
        # A pipeline revision is a new bounded engine attempt. Historical
        # invalidations stay in the audit log, but cannot consume the entire
        # amendment allowance before the replacement validator ever runs.
        # The migration event is conditional on the persisted revision, so
        # this boundary can occur only once per revision across all workers.
        boundary = max(
            (index for index, event in enumerate(events) if str(event.get("status")) == "pipeline_upgraded"),
            default=-1,
        )
        return sum(
            1
            for index, event in enumerate(events)
            if index > boundary
            if canonical_stage(str(event.get("stage"))) == stage and str(event.get("status")) == "invalidated"
        )

    def _resume_index(self, run_id: str) -> int:
        """The first stage this run still owes, after a restart or a rework."""
        run = self.store.get(run_id) or {}
        lap = int(run.get("lap") or 0)
        passed: set[str] = set()
        for event in self.store.events(run_id):
            stage = canonical_stage(str(event.get("stage")))
            status = str(event.get("status") or "")
            if status == "invalidated":
                # Next-stage validation refused this artifact; it is no
                # longer a pass even if an earlier event said it was.
                passed.discard(stage)
                continue
            if status != "passed":
                continue
            if stage in self.REWORK_STAGES and int(event.get("lap") or 0) < lap:
                continue  # earned on an earlier lap; this lap must redo it
            passed.add(stage)
        # A run that predates the plan/design split has no `plan` event and can
        # never grow one. Crediting it keeps the resume pointer moving forward
        # instead of restarting at a stage it never had. `spec` ANYWHERE is the
        # marker -- including a `started` event or the run's own stage column,
        # so a run interrupted mid-spec resumes into `design` rather than
        # rewinding to `plan` and then redoing the design work.
        legacy = (
            any(str(e.get("stage")) == "spec" for e in self.store.events(run_id))
            or str(run.get("stage") or "") == "spec"
        )
        if legacy and "plan" not in passed:
            passed.add("plan")
        pipe = self._pipeline(run)
        if "design" not in pipe:
            passed.add("design")
        for position, stage in enumerate(pipe):
            if stage not in passed:
                return position
        return len(pipe)

    def _revalidate_tests_pin(self, run: dict[str, Any]) -> None:
        """If HEAD moved after A/B, drop the tests/deploy pass and reopen tests."""
        reviewed = str(run.get("reviewed_sha") or "").strip()
        if not reviewed:
            return
        head = self._read_head(run)
        if head and self._same_commit(reviewed, head):
            return
        why = (
            "could not read HEAD to bind the A/B SHA"
            if not head
            else f"HEAD moved after A/B ({reviewed[:12]} -> {head[:12]})"
        )
        self._invalidate_stale_tests(run, why)

    def _release_lease(self, run_id: str, token: str = "") -> None:
        try:
            self.store.release_driver_lease(run_id, token)
        except Exception:  # noqa: BLE001
            log.debug("sdlc: could not release the lease for %s", run_id)

    def run(self, run_id: str) -> dict[str, Any]:
        """Walk the run pipeline to an open pull request. Blocking."""
        # Resume where the run actually got to. The driver thread dies with the
        # web process (a restart, a crash), so `run()` is re-entered on a run
        # that may already have stages behind it — replaying them would spend
        # the turns again and re-post their comments.
        resumed = self.store.get(run_id) or {}
        self._revalidate_tests_pin(resumed)
        index = self._resume_index(run_id)
        resumed = self.store.get(run_id) or resumed
        pipe = self._pipeline(resumed)
        owed_stage = pipe[index] if index < len(pipe) else ""
        # A verifier rejection blocks the current stage and the unattended
        # supervisor later re-enters here. The agent session cannot see the
        # verifier's separate chat, so dropping ``last_error`` makes it repeat
        # the same PASS until the recovery cap is exhausted. Carry the exact
        # objection into the resumed producer prompt.
        feedback = (
            str(resumed.get("last_error") or "")
            if (canonical_stage(str(resumed.get("blocked_stage") or "")) == owed_stage)
            else ""
        )
        prepared = False
        adopt = True
        while True:
            run = self.store.get(run_id)
            if not run:
                return {"ok": False, "error": "unknown run"}
            if not prepared:
                # an instruction. Nothing is dispatched until the checkout exists.
                try:
                    path = self.hooks.ensure_worktree(
                        str(run.get("repo") or ""),
                        str(run.get("branch") or ""),
                    )
                except WorktreeError as error:
                    # `stage` is not bound yet: the worktree is provisioned
                    # before the stage loop starts.
                    self.store.patch(run_id, status="blocked", blocked_stage="worktree", last_error=str(error))
                    self._post(run, stage="blocked", status="failed", note=f"no worktree, nothing dispatched: {error}")
                    return {"ok": False, "error": str(error), "stage": "worktree"}
                run = self.store.patch(run_id, worktree=path) or run
                self._sync_board_worktree(run, path)
                self.store.add_event(run_id, stage="worktree", status="passed", note=path)
                prepared = True
            if run.get("status") == "aborted":
                return {"ok": False, "error": "aborted"}
            # Whatever the row says, a branch already contained in the base has
            # nothing left to do -- and replaying stages against it is worse
            # than useless, because every diff is empty and every trace check
            # reports the artifact missing. Reconcile the row to the repository
            # instead of arguing with it.
            if not run.get("merged") and self._landed_reconciliation_eligible(run) and self._already_landed(run):
                self.store.patch(
                    run_id,
                    merged=True,
                    status="done",
                    last_error="",
                    blocked_stage="",
                    stage="merged",
                    stage_status="passed",
                )
                self._post(
                    self.store.get(run_id) or run,
                    stage="merged",
                    status="passed",
                    note="branch is already contained in the base — recorded as merged",
                )
                log.info("sdlc: %s was already landed; reconciled", run_id)
                return {"ok": True, "already_merged": True}
            pipe = self._pipeline(run)
            if index >= len(pipe):
                opened = self.open_pull_request(run)
                if not opened.get("ok"):
                    return opened
                fresh = self.store.get(run_id) or run
                if str(fresh.get("publish_mode") or "merge") == "pr":
                    # The operator froze this terminal state during intake.
                    # Opening the reviewed PR fulfils the run; merge and deploy
                    # are deliberately not invoked and require a later explicit
                    # owner action outside this autonomous pipeline.
                    self.store.add_event(
                        run_id,
                        stage="pr",
                        status="passed",
                        note="delivery contract complete: PR open; merge prohibited",
                    )
                    return {**opened, "stopped": "pr_open"}
                return self.merge_when_green(run_id)
            stage = pipe[index]
            # A pipeline runs far longer than any one lease, so the holder
            # renews at each stage boundary. Losing the renewal means another
            # worker has taken the run over, and continuing would duplicate its
            # work — so stop rather than race it.
            _tok = _leases.get(run_id, "")
            if _tok and not self.store.renew_driver_lease(run_id, _tok):
                log.warning("sdlc: %s lost its lease; another worker has it", run_id)
                return {"ok": False, "error": "lease lost", "stage": stage}
            lap = int(run.get("lap") or 0)
            result = None
            if adopt:
                adopt = False  # one chance, on the first stage after a resume
                result = self._adopt_orphaned_stage(run, stage)
                if result:
                    log.info("sdlc: adopted unwatched %s for %s", stage, run_id)
            if result is None:
                # A stage that ran out of budget leaves its commits behind, so a
                # second attempt continues the work rather than repeating it.
                previous = ""
                # P3/P5: a blocked run carries what its last attempt learned.
                # The attempt number is the GLOBAL one for this stage, so an
                # auto-resumed restart continues the count instead of
                # presenting itself as a fresh first try.
                attempt_start, rework_digest = _stored_rework(run)
                if canonical_stage(str(run.get("blocked_stage") or "")) != stage:
                    attempt_start, rework_digest = 1, {}
                for attempt in range(attempt_start, MAX_STAGE_ATTEMPTS + 1):
                    result = self.run_stage(
                        run,
                        stage,
                        feedback=feedback,
                        attempt=attempt,
                        previous=previous,
                        previous_attempt=rework_digest or None,
                    )
                    latest = self.store.get(run_id)
                    if result.get("run_aborted") or not latest or str(latest.get("status") or "") == "aborted":
                        return {"ok": False, "error": "aborted", "stage": stage}
                    if result.get("lease_lost"):
                        return {"ok": False, "error": "lease lost", "stage": stage}
                    if str(result.get("status") or "").upper() == "PASS":
                        break
                    if (
                        stage == "deploy"
                        or result.get("dispatch_error")
                        or result.get("policy_violation")
                        or result.get("not_validated")
                        or result.get("no_progress")
                    ):
                        break  # a rejection is the lap; the others are fatal
                    previous = str(result.get("note") or "")
                    if attempt < MAX_STAGE_ATTEMPTS:
                        # A stage attempt can spend a full stage timeout plus a
                        # gate budget, and the boundary renewal is a whole
                        # stage away. Without this, a retried stage can run
                        # past the lease and another worker takes the run over
                        # underneath it.
                        if not self._renew_lease(run_id):
                            return {"ok": False, "error": "lease lost", "stage": stage}
                        log.info("sdlc: retrying %s for %s (attempt %s)", stage, run_id, attempt + 1)
                        self._post(run, stage=stage, status="failed", note=f"attempt {attempt}: {previous}", lap=lap)
                        run = self.store.get(run_id) or run
                        lap = int(run.get("lap") or 0)
            if result.get("no_progress"):
                # P5: one more identical no-progress failure on this stage.
                # The count is what the supervisor's fuse reads; the digest
                # (already stored by the failing attempt) is what the next
                # attempt reads.
                counted = self.store.get(run_id) or {}
                self.store.patch(run_id, no_progress_attempts=int(counted.get("no_progress_attempts") or 0) + 1)
            passed = str(result.get("status") or "").upper() == "PASS"
            note = str(result.get("note") or "")
            evidence = str(result.get("evidence") or "")
            accepted_sha = str(result.get("accepted_sha") or "")
            if passed and (not accepted_sha or self._read_head(run) != accepted_sha):
                passed = False
                note = "verified artifact was not pinned to the current clean HEAD; the stage must be verified again"
                result = {"status": "FAIL", "note": note, "unverified": True}
            if passed and stage != "tests":
                try:
                    self.hooks.mark_stage_passed(
                        run,
                        stage,
                        note,
                        accepted_sha,
                    )
                except Exception as error:  # noqa: BLE001 — board is authoritative
                    passed = False
                    note = f"verified artifact could not advance the SDLC board: {error}"
                    result = {"status": "FAIL", "note": note, "dispatch_error": True}
            fields: dict[str, Any] = {
                "stage": stage,
                "stage_status": "passed" if passed else "failed",
            }
            self.store.patch(run_id, **fields)
            fresh = self.store.get(run_id) or run
            self._post(
                fresh,
                stage=stage,
                status="passed" if passed else "failed",
                note=note,
                evidence=evidence,
                lap=lap,
                cancellation_confirmed=result.get("cancellation_confirmed"),
                cancellation_idle_at=str(result.get("cancellation_idle_at") or ""),
            )
            if passed:
                # P5: any stage pass is progress; the no-progress fuse and the
                # handoff digest belong to a failure that is now over.
                self.store.patch(run_id, no_progress_attempts=0, rework_digest="")
                # Reset the recovery budget only on progress PAST the point the
                # run was last blocked at. Resetting on ANY stage pass re-opened
                # the loop the cap exists to close (blocked -> supervisor
                # resumes -> the replayed producer passes -> budget refilled ->
                # same refusal -> forever), because replaying a stage is not
                # progress past what is wrong. Getting FURTHER than the block
                # is, so a long run with unrelated transient failures still
                # earns its budget back.
                try:
                    blocked_at = str(fresh.get("blocked_stage") or "")
                    # `worktree`, `pr` and `driver` are not stages; they happen
                    # around the pipeline rather than inside it, so any stage
                    # passing is progress past them. Treating them as unknown
                    # meant those recoveries never earned the budget back.
                    before_everything = bool(blocked_at) and blocked_at not in pipe
                    if before_everything or (blocked_at in pipe and (pipe.index(stage) > pipe.index(blocked_at))):
                        self.store.clear_auto_resume(run_id)
                        self.store.patch(run_id, blocked_stage="")
                except Exception:  # noqa: BLE001 — bookkeeping must not fail a pass
                    log.debug("sdlc: could not evaluate recovery reset for %s", run_id)
                feedback = ""
                index += 1
                continue
            # Only an agent's own STATUS=FAIL is a verdict. A dispatch that
            # never reached one, a turn that died upstream, a turn that was
            # aborted — those are infrastructure, and treating them as review
            # rejections burned a rework lap per outage on a run whose code and
            # tests had both just passed.
            rewind = str(result.get("rewind") or "").strip()
            if rewind == "clarify":
                self.store.patch(
                    run_id,
                    status="blocked",
                    last_error=note,
                    blocked_stage="clarify",
                )
                self._post(
                    fresh,
                    stage="blocked",
                    status="failed",
                    note=f"definition invalid — back to clarify: {note}",
                )
                return {"ok": False, "error": note, "stage": "clarify"}
            if rewind == "design" and "design" not in pipe:
                rewind = "plan"
            if rewind in pipe and GATE_UNUSABLE in note:
                # A gate that could not DECIDE is infrastructure, not a review
                # rejection: the reviewer's engine errored, or it was shown less
                # evidence than the stage produced. Rewinding to `code` on that
                # redoes work nobody criticised and spends a rework lap per
                # outage -- the same distinction the driver already draws for a
                # dispatch that never reached a verdict. Block instead; the
                # supervisor's bounded auto-resume retries it, and a persistent
                # outage stops safely rather than looping.
                self.store.patch(run_id, status="blocked", last_error=note, blocked_stage=stage)
                self._post(
                    fresh,
                    stage="blocked",
                    status="failed",
                    note=f"`{stage}` could not be decided (not a rejection of the work): {note}",
                )
                return {"ok": False, "error": note, "stage": stage}
            if rewind in pipe:
                log.info("sdlc: %s rewound to %s for %s", stage, rewind, run_id)
                self._post(
                    fresh,
                    stage=rewind,
                    status="invalidated",
                    lap=lap,
                    note=f"`{stage}` sent the run back to `{rewind}`: {note}",
                )
                self.store.patch(run_id, status="running", last_error="")
                feedback = "INPUT: " + note
                index = pipe.index(rewind)
                continue
            if result.get("not_validated"):
                prev = self._previous_producer(fresh, stage)
                max_laps = int(run.get("max_laps") or 3)
                if prev is None or self._input_rework_count(run_id, prev) >= max_laps:
                    self.store.patch(run_id, status="blocked", last_error=note, blocked_stage=stage)
                    self._post(
                        fresh, stage="blocked", status="failed", note=f"`{stage}` never returned a verdict: {note}"
                    )
                    return {"ok": False, "error": note, "stage": stage}
                log.info("sdlc: %s input refused; looping back to %s for %s", stage, prev, run_id)
                self._post(
                    fresh,
                    stage=prev,
                    status="invalidated",
                    lap=lap,
                    note=f"`{stage}` refused this `{prev}` output: {note}",
                )
                self.store.patch(run_id, status="running", last_error="")
                feedback = "INPUT: " + note
                index = pipe.index(prev)
                continue
            if not result.get("verdict"):
                self.store.patch(run_id, status="blocked", last_error=note, blocked_stage=stage)
                self._post(fresh, stage="blocked", status="failed", note=f"`{stage}` never returned a verdict: {note}")
                return {"ok": False, "error": note, "stage": stage}
            # A deploy-readiness rejection is the loop; anything else is a stop.
            if stage != "deploy":
                self.store.patch(run_id, status="blocked", last_error=note, blocked_stage=stage)
                self._post(
                    fresh, stage="blocked", status="failed", note=f"stopped at `{STAGE_TAGS.get(stage, stage)}`: {note}"
                )
                return {"ok": False, "error": note, "stage": stage}
            if lap + 1 > int(run.get("max_laps") or 3):
                self.store.patch(
                    run_id,
                    status="blocked",
                    blocked_stage=stage,
                    last_error=f"deploy readiness rejected {lap + 1}x: {note}",
                )
                self._post(
                    fresh,
                    stage="blocked",
                    status="failed",
                    note=f"rework budget exhausted after {lap + 1} laps: {note}",
                )
                return {"ok": False, "error": "max laps", "stage": stage}
            self.store.patch(run_id, lap=lap + 1)
            feedback = note
            index = pipe.index("code")


def start_run_thread(driver: SdlcDriver, run_id: str) -> bool:
    """SDLC pipeline removed — never start a driver thread."""
    return False


#: How long a supervised restart waits before the next, doubling each time.
#: Bounded so a broken run backs off instead of spinning, and so a transient
#: outage is retried soon rather than in an hour.
AUTO_RESUME_BACKOFF_S = (60, 300, 900, 1800, 3600)
#: How many times the supervisor restarts one run before leaving it alone.
AUTO_RESUME_MAX = 5


_sweep_lock = threading.Lock()


def supervise_runs(
    driver_for: Callable[[], SdlcDriver],
    *,
    dependencies_ready: Callable[[], bool] = lambda: True,
) -> list[str]:
    """Restart blocked runs whose backoff has expired. Returns the run ids.

    `recover_runs` fires once per process, so a run that blocked while the
    server was UP stayed blocked until a human pressed resume. This is the
    periodic half: it is what makes the pipeline unattended rather than merely
    restart-tolerant.

    Every failure here is deliberately non-fatal — a supervisor that can take
    the server down with it is worse than no supervisor.
    """
    started: list[str] = []
    # Timer and request hook both call this. Without a claim, two sweeps can
    # see the same due run and each spend an attempt on one actual restart.
    if not _sweep_lock.acquire(blocking=False):
        return started
    try:
        # Do not spend a bounded recovery attempt while the command path is
        # still booting. Readiness is checked inside the sweep lock so two Web
        # workers cannot both pass it and race the database claim.
        try:
            if not dependencies_ready():
                return started
        except Exception:  # noqa: BLE001 — unavailable means not ready
            return started
        try:
            driver = driver_for()
            due = driver.store.due_for_auto_resume(max_attempts=AUTO_RESUME_MAX)
        except Exception:  # noqa: BLE001
            log.exception("sdlc: supervisor could not enumerate blocked runs")
            return started
        # A run can be `running` and yet have nobody driving it: the thread
        # dies with its worker, or exits on a path that leaves the status
        # alone. `due_for_auto_resume` only enumerates BLOCKED runs, so those
        # sat mid-stage forever -- observed live, two runs stuck at
        # `tests/started` for half an hour with the work already done. Adopt
        # them here; `start_run_thread` claims the database lease, so a run
        # another worker really is driving is refused and skipped.
        try:
            due = list(due) + [
                run
                for run in driver.store.list()
                if str(run.get("status") or "") in ("running", "pr_open")
                and not run_is_active(str(run.get("run_id") or ""))
                and str(run.get("run_id") or "") not in {str(r.get("run_id") or "") for r in due}
            ]
        except Exception:  # noqa: BLE001 — adoption is best effort
            log.exception("sdlc: supervisor could not enumerate orphaned runs")
        for run in due:
            run_id = str(run.get("run_id") or "")
            if not run_id:
                continue
            # An orphan is not a recovery: it has not failed, it has merely
            # lost its driver. Restart it without spending a bounded attempt.
            if str(run.get("status") or "") in ("running", "pr_open"):
                try:
                    if start_run_thread(driver, run_id):
                        started.append(run_id)
                        log.info(
                            "sdlc: adopted orphaned run %s at %s/%s", run_id, run.get("stage"), run.get("stage_status")
                        )
                except Exception:  # noqa: BLE001
                    log.exception("sdlc: could not adopt orphaned run %s", run_id)
                continue
            attempt = int(run.get("auto_resume_count") or 0)
            backoff = AUTO_RESUME_BACKOFF_S[min(attempt, len(AUTO_RESUME_BACKOFF_S) - 1)]
            try:
                # A DATABASE claim, not a process one: with several gunicorn
                # workers each has its own lock and its own supervisor, so only
                # a conditional UPDATE can decide which restarts the run. It
                # commits before returning, so a crash between claiming and
                # launching cannot refund the attempt.
                if not driver.store.claim_auto_resume(run_id, expected_count=attempt, backoff_seconds=backoff):
                    continue  # another worker got it
                # Start FIRST, then mark running. Patching first left the run
                # `running` with no driver whenever the start was refused (a
                # live thread or a lease held elsewhere) — and `running` is not
                # in the supervisor's query, so it would never be swept again.
                if start_run_thread(driver, run_id):
                    # Deliberately NOT writing status here. `run_stage` sets
                    # `running` the moment it begins, and a fast driver can
                    # finish or block before this line would land — an
                    # unconditional write then clobbers `pr_open` or `blocked`
                    # and strands the run. The driver owns its own status; the
                    # supervisor's job is only to start it.
                    started.append(run_id)
                    log.info("sdlc: supervisor resumed %s (attempt %d, next in %ds)", run_id, attempt + 1, backoff)
            except Exception:  # noqa: BLE001
                log.exception("sdlc: supervisor could not resume %s", run_id)
        return started
    finally:
        _sweep_lock.release()


_supervisor_thread: threading.Thread | None = None
_supervisor_lock = threading.Lock()


def start_supervisor(
    driver_for: Callable[[], SdlcDriver],
    *,
    every_seconds: float = 120.0,
    dependencies_ready: Callable[[], bool] = lambda: True,
) -> bool:
    """Sweep for blocked runs on a timer, not on request traffic.

    The request hook alone was not autonomy: with nobody visiting the Dev tab,
    a blocked run waited indefinitely without ever exhausting a bound. Nothing
    about "recover from a transient failure" should depend on a human happening
    to load a page.

    A DAEMON thread, so it still dies with the process and cannot outlive the
    app — the property the request-driven design was reaching for. Idempotent:
    repeated calls keep the one thread.
    """
    return False


def recover_runs(
    driver_for: Callable[[], SdlcDriver],
    *,
    dependencies_ready: Callable[[], bool] = lambda: True,
) -> list[str]:
    """Re-arm runs the DRIVER THREAD abandoned, after a process restart.

    The driver is a daemon thread, so it dies with the web process and leaves a
    `running` row that nothing is advancing. Recovery is safe because `run()`
    resumes at the first stage the run still owes and adopts a stage that
    finished unwatched, so nothing is redone.

    BLOCKED runs are deliberately NOT included. They are the supervisor's, and
    the supervisor is what bounds them — restarting one here would start it with
    no backoff and no counter increment, so a run that had already exhausted its
    five attempts would get five more on every process restart. That defeats the
    cap entirely, which is worse than never retrying: an unattended loop that
    looks like progress.
    """
    started: list[str] = []
    try:
        if not dependencies_ready():
            return started
    except Exception:  # noqa: BLE001 — unavailable means not ready
        return started
    try:
        driver = driver_for()
        runs = [r for r in driver.store.resumable() if str(r.get("status") or "") != "blocked"]
    except Exception:  # noqa: BLE001 — a failed sweep must not stop the server
        log.exception("sdlc: could not enumerate resumable runs")
        return started
    for run in runs:
        run_id = str(run.get("run_id") or "")
        if not run_id or run_is_active(run_id):
            continue
        # `run_is_active` is process-local, so with several workers each thinks
        # nobody is driving this run and they all restart it, dispatching the
        # same stage concurrently. The lease is the database's.
        if start_run_thread(driver, run_id):
            started.append(run_id)
    if started:
        log.info("sdlc: resumed %d run(s) after restart: %s", len(started), started)
    return started


def run_is_active(run_id: str) -> bool:
    with _threads_lock:
        thread = _threads.get(run_id)
    return bool(thread and thread.is_alive())
