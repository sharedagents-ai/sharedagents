"""Persisted records for background (async) delegations.

One JSON file per job under ``<SHAREDAGENTS_HOME>/jobs/<job_id>.json``, written
atomically (tmp + ``os.replace``) so a reader never sees a half-written
record. Survives gateway/orchestrator restarts; the detached runner, the
reconciler, and the submitting shim are the only writers.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat

TERMINAL = frozenset({"done", "completed", "failed", "cancelled", "aborted"})
_CANCEL_FENCE_TTL_S = 10.0
_STARTTIME_UNSET = object()
# create_wave() ids: "wave_" + uuid4().hex.
_WAVE_ID_RE = re.compile(r"wave_[0-9a-f]{32}")


def _home() -> Path:
    return Path(os.environ.get("SHAREDAGENTS_HOME") or os.path.expanduser("~/.sharedagents"))


def _dir() -> Path:
    d = _home() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(job_id: str) -> Path:
    return _dir() / f"{job_id}.json"


def _cancel_fences_path() -> Path:
    return _dir() / ".cancel_fences.json"


def _read_cancel_fences(now: float | None = None) -> dict[str, float]:
    """Return unexpired session cancellation fences."""
    current = time.time() if now is None else float(now)
    try:
        raw = json.loads(_cancel_fences_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for sid, expires_at in raw.items():
        try:
            expiry = float(expires_at)
        except (TypeError, ValueError):
            continue
        if sid and expiry > current:
            out[str(sid)] = expiry
    return out


def _write_cancel_fences(fences: dict[str, float]) -> None:
    path = _cancel_fences_path()
    tmp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex[:8]}")
    try:
        tmp.write_text(json.dumps(fences, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _extend_cancel_fences(session_ids: Any) -> None:
    """Keep subtree publication fenced while a cancellation is in progress."""
    targets = {str(sid) for sid in (session_ids or []) if sid}
    if not targets:
        return
    with _locked("cancel-fences"):
        now = time.time()
        fences = _read_cancel_fences(now)
        expiry = now + _CANCEL_FENCE_TTL_S
        for target in targets:
            fences[target] = max(fences.get(target, 0.0), expiry)
        _write_cancel_fences(fences)


@contextlib.contextmanager
def _locked(job_id: str) -> Iterator[None]:
    """Per-job exclusive lock so concurrent read-modify-write updates from
    the submitting shim and the detached runner can't lose each other's
    fields (e.g. the runner's status write clobbering the parent's pid).
    """
    lock_path = _dir() / f"{job_id}.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        oscompat.file_lock(fd)
        yield
    finally:
        try:
            oscompat.file_unlock(fd)
        finally:
            os.close(fd)


def _write(rec: dict[str, Any]) -> None:
    p = _path(rec["job_id"])
    # Unique temp per writer so two concurrent writers never collide on the
    # same scratch file; os.replace is atomic on the final path.
    tmp = _dir() / f"{rec['job_id']}.json.tmp.{uuid.uuid4().hex[:8]}"
    try:
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def wave_membership_final(parent_sid: str, wave_id: str) -> bool:
    """True once a wave can no longer accept members (caller holds no locks).

    A wave is EDITABLE mid flight (same-goal appends) until a delivery attempt
    begins: the first claim pins the gateway receipt key in the manifest, and
    any delivered member id marks the wave complete. From then on membership
    is final, so the receipt key and the exactly-once digest stay stable.
    """
    manifest = read("." + wave_id) if wave_id else None
    if manifest is not None and manifest.get("receipt_key"):
        return True
    delivered, claims = _read_marker(parent_sid)
    locked = {jid for c in claims.values() for jid in (c.get("ids") or [])}
    members = {j["job_id"] for j in list_jobs() if j.get("parent_sid") == parent_sid and j.get("wave_id") == wave_id}
    return bool(members & (delivered | locked))


def requested_wave_id(value: Any) -> str:
    """A caller's ``wave_id`` argument; empty means "open a new wave".

    Weak models fill an unused optional field with a null-like placeholder,
    which is not a request to join anything.
    """
    text = str(value or "").strip()
    return "" if text.lower() in ("null", "none", "undefined") else text


def ensure_wave_joinable(parent_sid: str, wave_id: str) -> None:
    """Raise ``ValueError`` unless ``wave_id`` can still accept members.

    Lets a dispatcher fail fast (one clean error) instead of per-member
    mid-dispatch; :func:`create` re-validates authoritatively under its locks.
    Every runtime wave has a manifest (:func:`create_wave`), so an id without
    one is a mistake. It must not become a wave without a receipt-key pin.
    """
    manifest = read("." + wave_id) if _WAVE_ID_RE.fullmatch(wave_id or "") else None
    if manifest is None:
        raise ValueError(f"unknown delegation wave {wave_id!r}; omit wave_id to start a new wave")
    if manifest.get("parent_sid") != parent_sid:
        raise ValueError("delegation wave belongs to another parent session; dispatch a new wave")
    if wave_membership_final(parent_sid, wave_id):
        raise ValueError("delegation wave is sealed and no longer accepting members; dispatch a new wave")


def create(
    *, parent_sid: str, agent: str, task: str, on_complete: str, call_id: str, pid: int, wave_id: str = ""
) -> dict[str, Any]:
    now = time.time()
    # Serialize job publication with cancellation-fence installation. A create
    # that wins the lock first is visible in the canceller's snapshot; one that
    # loses is born terminal and its runner's claim is refused.
    with _locked("cancel-fences"), _locked("resume:" + parent_sid):
        if wave_id:
            # A sealed registration barrier still accepts same-goal members
            # while the wave is in flight; only a wave whose delivery has
            # begun (or finished) is closed.
            ensure_wave_joinable(parent_sid, wave_id)
        fences = _read_cancel_fences(now)
        fenced = bool(parent_sid and parent_sid in fences)
        _write_cancel_fences(fences)
        rec = {
            "job_id": "job_" + uuid.uuid4().hex[:16],
            # parent_sid identifies the JSONL under the UNIFIED sessions_root()
            # (all agents share one sessions folder since the 2026-05-26 layout
            # refactor; attribution is sessions_meta.agent_name, not the path).
            "parent_sid": parent_sid,
            "wave_id": wave_id,
            # The session this job's runner drives; recorded early by the runner so
            # cascade-cancel can find the job that owns a given session. Empty until
            # the child turn resolves it.
            "child_sid": "",
            "agent": agent,
            "task": task,
            "on_complete": on_complete,
            "call_id": call_id,
            "pid": int(pid),
            "pid_starttime": _proc_starttime(int(pid)) if int(pid) > 0 else None,
            "status": "cancelled" if fenced else "started",
            "ok": False if fenced else None,
            "error": "cancelled (session fence)" if fenced else None,
            "created_at": now,
            "ended_at": now if fenced else None,
        }
        _write(rec)
    return rec


def read(job_id: str) -> dict[str, Any] | None:
    try:
        return json.loads(_path(job_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def update(job_id: str, **fields: Any) -> dict[str, Any] | None:
    # Serialize the whole read-modify-write under the per-job lock so a
    # concurrent writer can't overwrite fields it never saw.
    with _locked(job_id):
        rec = read(job_id)
        if rec is None:
            return None
        rec.update(fields)
        if fields.get("status") in TERMINAL and rec.get("ended_at") is None:
            rec["ended_at"] = time.time()
        _write(rec)
        return rec


def list_jobs() -> list[dict[str, Any]]:
    out = []
    for f in sorted(_dir().glob("job_*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def open_jobs() -> list[dict[str, Any]]:
    return [j for j in list_jobs() if j.get("status") not in TERMINAL]


def open_jobs_for(parent_sid: str) -> list[dict[str, Any]]:
    """OPEN (non-terminal) background jobs spawned BY ``parent_sid``."""
    return [j for j in open_jobs() if j.get("parent_sid") == parent_sid]


# A terminal job not yet delivered within this window is still "on its way"
# to the parent (delivery normally lands within a second or two). Older ones
# are leftovers (a crashed delivery, an aborted parent) owned by the
# reconciler.
UNDELIVERED_WINDOW_S = 120.0


def undelivered_results_for(
    parent_sid: str, *, now: float | None = None, window_s: float = UNDELIVERED_WINDOW_S
) -> list[dict[str, Any]]:
    """Terminal jobs of ``parent_sid`` whose result has not reached it yet.

    Between a child finishing and its result being queued for (or posted to)
    the parent there is a short gap. During it the parent has no open job and
    an empty queue, yet an event is about to arrive.
    """
    if not parent_sid:
        return []
    current = time.time() if now is None else float(now)
    delivered = _read_covered(parent_sid)
    out = []
    for j in list_jobs():
        if j.get("parent_sid") != parent_sid or j.get("status") not in TERMINAL:
            continue
        if j.get("job_id") in delivered or j.get("delivery_silenced"):
            continue
        ended = float(j.get("ended_at") or 0)
        if ended and current - ended <= window_s:
            out.append(j)
    return out


# --- per-wave parent-delivery claim -------------------------------------------
#
# When a background delegation finishes, the parent orchestrator (jarvis) is
# idle — its dispatching turn ended at the "started" ack. Nothing re-invokes it,
# so the ``subagent_result`` just sits on the log unconsumed. The runner (and, at
# boot, reconcile) POST one continuation per terminal wave (legacy jobs solo).
#
# Two-phase marker (Residual 1 — crash-between-claim-and-POST). The durable
# per-parent marker file records TWO distinct sets, never one:
#   * ``delivered`` — tasks whose resume POST returned 2xx. A DELIVERED task is
#     FINAL: reconcile skips it, so it
#     is never re-fired. This is the exactly-once guarantee.
#   * ``claims`` — in-flight RESERVATIONS ("I am attempting the resume"), keyed by
#     the claimant job_id and carrying the *claiming process pid*. A reservation
#     is NOT delivered: it only stops a concurrent finalizer (or a same-boot
#     reconcile) from double-firing the task WHILE the attempt is live. If the
#     claiming process CRASHES between reserving and a successful POST, its pid is
#     dead → the reservation is STALE → reconcile re-claims and re-attempts the
#     task exactly once. A stale claim therefore NEVER strands the parent, and a
#     truly delivered task is NEVER re-fired.
#
# ``claim_idle_resume_ids`` writes only a RESERVATION; the durable ``delivered``
# marker is written by :func:`mark_delivered` ONLY after the POST is confirmed
# after the POST is confirmed. A failed POST calls
# :func:`release_resume_claim`, which drops the reservation so the task is
# immediately re-claimable.


def _resume_marker_path(parent_sid: str) -> Path:
    # Safe filename regardless of parent_sid shape; never matches the job_*.json
    # glob so list_jobs() ignores it.
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in parent_sid)
    return _dir() / f".resumed_{safe}.json"


def _read_marker(parent_sid: str) -> tuple[set, dict]:
    """Return ``(delivered_ids, claims)`` from the durable per-parent marker.

    ``claims`` maps ``claimant_job_id -> {"pid": int, "at": float, "ids": [...]}``.
    Back-compat: a legacy bare-list marker (the pre-two-phase format) is read as
    ``delivered`` with no claims — treating any previously-"covered" wave as final
    is the safe interpretation (it is never re-fired), and there are no live
    reservations to recover from a format that never wrote any.
    """
    try:
        data = json.loads(_resume_marker_path(parent_sid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set(), {}
    if isinstance(data, list):
        return set(data), {}
    if isinstance(data, dict):
        delivered = set(data.get("delivered") or [])
        claims = {k: dict(v) for k, v in (data.get("claims") or {}).items() if isinstance(v, dict)}
        return delivered, claims
    return set(), {}


def _read_covered(parent_sid: str) -> set:
    """The DELIVERED set — tasks that are final and must never be re-fired.

    A mere reservation is NOT covered: reconcile deliberately skips only
    ``delivered`` so a claimed-but-crashed wave stays recoverable (Residual 1).
    """
    return _read_marker(parent_sid)[0]


def _write_marker(parent_sid: str, delivered: set, claims: dict) -> None:
    p = _resume_marker_path(parent_sid)
    tmp = p.with_suffix(f".json.tmp.{uuid.uuid4().hex[:8]}")
    payload = {"delivered": sorted(delivered), "claims": claims}
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def create_wave(parent_sid: str, origin: str = "") -> str:
    """Open a dispatch barrier before any member can finish.

    Membership is the jobs bearing this id, scoped to the parent. Only the
    runtime assigns ids; a sealed wave still accepts same-goal members while
    it is in flight (see :func:`wave_membership_final`). A dead dispatcher
    leaves a recoverable sealed snapshot, never an eternal barrier. No
    records or migration are required for legacy solo jobs.

    ``origin`` names the owner prompt the wave works for (see
    ``local_llm.coordination``); the coordinator allows one live wave per
    origin. Waves written before origins existed simply have none.
    """
    wave_id = "wave_" + uuid.uuid4().hex
    rec = {
        "job_id": "." + wave_id,
        "parent_sid": parent_sid,
        "sealed": False,
        "pid": os.getpid(),
        "pid_starttime": _proc_starttime(os.getpid()),
    }
    if origin:
        rec["origin"] = str(origin)
    _write(rec)
    return wave_id


def wave_origin(wave_id: str) -> str:
    """The owner-prompt origin recorded on a wave, or ``""``."""
    if not _WAVE_ID_RE.fullmatch(str(wave_id or "")):
        return ""
    manifest = read("." + wave_id)
    return str((manifest or {}).get("origin") or "")


def seal_wave(parent_sid: str, wave_id: str) -> list[str]:
    """Close registration; callers attempt delivery after releasing this lock."""
    with _locked("resume:" + parent_sid):
        update("." + wave_id, sealed=True)
        return [j["job_id"] for j in list_jobs() if j.get("parent_sid") == parent_sid and j.get("wave_id") == wave_id]


def close_wave(parent_sid: str, wave_id: str) -> list[str]:
    """Seal registration AND freeze membership for an aborting wave.

    Pinning the receipt key at abort time keeps the gateway idempotency key
    stable and refuses late same-goal appends: an aborted wave is final even
    while its cancellation bookkeeping is still settling.
    """
    with _locked("resume:" + parent_sid):
        ids = [j["job_id"] for j in list_jobs() if j.get("parent_sid") == parent_sid and j.get("wave_id") == wave_id]
        manifest = read("." + wave_id) if wave_id else None
        if manifest is not None:
            fields: dict[str, Any] = {"sealed": True}
            if ids and not manifest.get("receipt_key"):
                fields["receipt_key"] = min(ids)
            update("." + wave_id, **fields)
        return ids


def wave_jobs(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Legacy records without a wave id are ALWAYS independent solo waves."""
    wave_id = rec.get("wave_id")
    if not wave_id:
        return [rec]
    return sorted(
        (j for j in list_jobs() if j.get("parent_sid") == rec.get("parent_sid") and j.get("wave_id") == wave_id),
        key=lambda j: j["job_id"],
    )


def claim_idle_resume_ids(parent_sid: str, job_id: str, pid: int | None = None) -> list[str] | None:
    """Reserve ONE continuation for a fully terminal, fully dispatched wave.

    The parent lock serializes competing runners/reconcilers. All member ids
    are reserved and delivered together. Legacy jobs keep their solo delivery
    contract. Reservations recover after a dead claimant; endpoint receipts
    deduplicate a POST accepted just before the claimant crashed.
    """
    if pid is None:
        pid = os.getpid()
    with _locked("resume:" + parent_sid):
        delivered, claims = _read_marker(parent_sid)
        rec = read(job_id)
        if rec is None or rec.get("parent_sid") != parent_sid:
            return None
        wave_id = rec.get("wave_id")
        manifest = read("." + wave_id) if wave_id else None
        if manifest and not manifest.get("sealed") and _job_runner_alive(manifest):
            return None
        members = wave_jobs(rec)
        if not members or any(j.get("status") not in TERMINAL for j in members):
            return None
        ids = [j["job_id"] for j in members]
        if set(ids) <= delivered:
            return None
        claims = {cj: c for cj, c in claims.items() if _pid_alive(int(c.get("pid") or 0))}
        live_locked = {jid for c in claims.values() for jid in c.get("ids", [])}
        if set(ids) & live_locked:
            return None
        claims[job_id] = {"pid": int(pid), "at": time.time(), "ids": ids}
        if wave_id:
            manifest = read("." + wave_id)
            if manifest is not None and not manifest.get("receipt_key"):
                # Freeze the gateway receipt key at first claim: membership is
                # final from here on, so a retry after a lost POST response
                # re-uses the same key and delivery stays exactly-once.
                update("." + wave_id, receipt_key=min(ids))
        _write_marker(parent_sid, delivered, claims)
        return ids


def claim_idle_resume(parent_sid: str, job_id: str) -> bool:
    """Bool view of :func:`claim_idle_resume_ids`."""
    return claim_idle_resume_ids(parent_sid, job_id) is not None


def mark_delivered(parent_sid: str, job_ids: Any) -> None:
    """Promote a reservation to DELIVERED after a confirmed resume POST.

    Only a DELIVERED task is skipped by :func:`reconcile`, so this is what makes
    the resume exactly-once ACROSS crashes/restarts: written ONLY after the POST
    succeeds (Residual 1), never at reservation time. Idempotent — re-delivering
    an already-delivered wave is a no-op. Drops the matching in-flight
    reservation.
    """
    ids = set(job_ids or [])
    if not ids:
        return
    with _locked("resume:" + parent_sid):
        delivered, claims = _read_marker(parent_sid)
        delivered = delivered | ids
        # Drop any reservation now fully covered by `delivered`.
        claims = {cj: c for cj, c in claims.items() if not (set(c.get("ids") or []) <= delivered)}
        _write_marker(parent_sid, delivered, claims)


def release_resume_claim(parent_sid: str, job_ids: Any) -> None:
    """Drop the in-flight RESERVATION for ``job_ids`` (the token from
    :func:`claim_idle_resume_ids`) — the resume POST was NOT confirmed.

    The task becomes immediately re-claimable so reconcile or a retry re-attempts
    instead of the parent staying stranded. Never touches
    ``delivered``, so a different task already delivered for this parent stays
    delivered (no accidental double-wake). A no-op if the
    reservation is already gone.
    """
    ids = set(job_ids or [])
    if not ids:
        return
    with _locked("resume:" + parent_sid):
        delivered, claims = _read_marker(parent_sid)
        remaining = {cj: c for cj, c in claims.items() if not (set(c.get("ids") or []) & ids)}
        if remaining != claims:
            _write_marker(parent_sid, delivered, remaining)


def _event_epoch(ts: Any) -> float | None:
    """Parse a session-log ``ts`` (ISO-8601, ms precision) to epoch seconds."""
    if not ts:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(str(ts)).timestamp()
    except (ValueError, TypeError):
        return None


# --- cascading cancellation -----------------------------------------------
#
# A background job's runner is a session/group leader (spawned with
# start_new_session), so killpg(runner_pid) reaps the runner AND any CLI engine
# (claude/codex) it spawned. Background GRANDCHILDREN are their own detached
# sessions, reached by walking the session DAG (sessions_meta.parent_session_id)
# DOWN from the target — never up, so cancelling a middle subagent never touches
# its parent/orchestrator.


def _descendant_sessions(sid: str) -> set:
    """All sessions transitively under ``sid`` via sessions_meta.parent_session_id.
    Excludes ``sid`` itself. Empty on any DB error (cancel still kills direct
    jobs via parent_sid/child_sid == sid).
    """
    out: set = set()
    try:
        from sharedagents_core.layout import sessions_root
        from sharedagents_session.session_index import connect_sessions_db

        db = sessions_root() / "sessions.db"
        if not db.exists():
            return out
        con = connect_sessions_db(db)
        try:
            frontier = [sid]
            seen = {sid}
            while frontier:
                cur = frontier.pop()
                rows = con.execute(
                    "SELECT session_hash FROM sessions_meta WHERE parent_session_id=?", (cur,)
                ).fetchall()
                for (child,) in rows:
                    if child and child not in seen:
                        seen.add(child)
                        out.add(child)
                        frontier.append(child)
        finally:
            con.close()
    except Exception:  # noqa: BLE001 — best-effort; never block a cancel
        pass
    return out


def _proc_starttime(pid: int) -> int | None:
    """Linux process birth identity from ``/proc/<pid>/stat`` field 22."""
    if oscompat.IS_WINDOWS or pid <= 0:
        return None
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    rparen = data.rfind(b")")
    if rparen < 0:
        return None
    try:
        return int(data[rparen + 2 :].split()[19])
    except (IndexError, ValueError):
        return None


def _descendant_pids(root_pid: int) -> list[int]:
    """All transitive child PIDs of ``root_pid`` (Linux ``/proc`` walk),
    DOWNWARD only — never an ancestor, so the gateway/orchestrator is never in
    the kill set. Best-effort; never raises. (Mirrors tools.runners._descendant_pids
    but kept here so async_jobs stays stdlib-only / importable by the runner.)
    """
    children: dict[int, list[int]] = {}
    if oscompat.IS_WINDOWS or root_pid <= 0:
        return []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                data = fh.read()
            rparen = data.rfind(b")")  # comm may contain spaces / ')'
            ppid = int(data[rparen + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    out: list[int] = []
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        for ch in children.get(cur, []):
            out.append(ch)
            stack.append(ch)
    return out


def _kill_runner_tree(
    pid: int,
    *,
    expected_starttime: Any = _STARTTIME_UNSET,
) -> bool:
    """SIGTERM→grace→SIGKILL the runner AND its whole descendant tree.

    The detached runner is a session leader, but the CLI engines it spawns
    (claude/codex/antigravity) are ALSO ``start_new_session`` → their own
    process groups, which a lone ``killpg(runner)`` misses. So walk ``/proc``
    for every descendant (downward only) and signal each pid directly, plus
    ``killpg`` each one that is its own group leader (reaches that group's
    children too). Never signals ancestors → the orchestrator is safe.
    """
    import signal
    import time as _t

    if pid <= 0:  # never os.kill(0/-1, …) — that broadcasts to a group
        return False
    if oscompat.IS_WINDOWS:
        # No /proc birth identity or descendant sweep on Windows. As with
        # gateway pidfiles, this is best-effort cancellation of the saved pid.
        try:
            oscompat.kill_process(pid, force=True)
            return True
        except OSError:
            return False
    if expected_starttime is _STARTTIME_UNSET:
        # Direct/internal diagnostic callers without a durable record snapshot
        # bind to the process that exists at function entry.
        expected_starttime = _proc_starttime(pid)
    if expected_starttime is None or _proc_starttime(pid) != expected_starttime:
        return False
    identities: dict[int, int] = {pid: int(expected_starttime)}
    for descendant in _descendant_pids(pid):
        starttime = _proc_starttime(descendant)
        if descendant > 0 and starttime is not None:
            identities[descendant] = starttime

    def _blast(sig: int) -> bool:
        signalled = False
        for p, starttime in identities.items():
            if p <= 0:
                continue
            # Revalidate before EACH TERM/KILL. A process can exit and its PID
            # can be recycled during the grace window.
            if _proc_starttime(p) != starttime:
                continue
            try:
                os.kill(p, sig)
                signalled = True
            except (ProcessLookupError, PermissionError):
                pass
            try:
                if os.getpgid(p) == p:  # own-session leader → reap its group
                    os.killpg(p, sig)
                    signalled = True
            except (ProcessLookupError, PermissionError):
                pass
        return signalled

    signalled = _blast(signal.SIGTERM)
    if _proc_starttime(pid) == expected_starttime and _pid_alive(pid):
        _t.sleep(0.4)
    return _blast(signal.SIGKILL) or signalled


def try_finalize(job_id: str, status: str, ok: bool, error: Any = None) -> bool:
    """Terminalize a job to done/failed ONLY if it is still ``running`` — never
    overwrite a cancellation that landed mid-turn (which would resurrect a
    cancelled job as completed). Returns True if it wrote the terminal status.
    """
    with _locked(job_id):
        rec = read(job_id)
        if rec is None or rec.get("status") != "running":
            return False
        rec["status"] = status
        rec["ok"] = bool(ok)
        rec["error"] = error
        rec["ended_at"] = time.time()
        _write(rec)
        return True


def fail_submission(job_id: str, error: str) -> None:
    """A spawn failure can only fail an unclaimed job, never undo cancellation."""
    with _locked(job_id):
        rec = read(job_id)
        if rec is None or rec.get("status") != "started":
            return
        rec.update(status="failed", ok=False, error=error, ended_at=time.time())
        _write(rec)


def claim(job_id: str, pid: int) -> bool:
    """Atomically transition a job to ``running`` with the runner's real pid —
    UNLESS it is already terminal (e.g. cancelled during startup), in which case
    return False so the runner exits WITHOUT driving the child. Closes the
    resurrection race where a cancel landing in the pid=0 window could be undone
    by the runner's status write.
    """
    with _locked(job_id):
        rec = read(job_id)
        if rec is None or rec.get("status") in TERMINAL:
            return False
        rec["status"] = "running"
        rec["pid"] = int(pid)
        rec["pid_starttime"] = _proc_starttime(int(pid))
        _write(rec)
        return True


def _claim_for_cancel(
    job_id: str,
    reason: str,
    *,
    return_identity: bool = False,
) -> Any:
    """Under the lock: if the job is still open, mark it cancelled and return
    the CURRENT pid to kill; if already terminal, return None. Re-reading the
    pid inside the lock (vs. a stale snapshot) closes the TOCTOU where the
    runner records its real pid between a snapshot and the kill.
    """
    with _locked(job_id):
        rec = read(job_id)
        if rec is None or rec.get("status") in TERMINAL:
            return None
        pid = int(rec.get("pid") or 0)
        rec["status"] = "cancelled"
        rec["ok"] = False
        rec["error"] = reason
        rec["ended_at"] = time.time()
        _write(rec)
        if return_identity:
            return pid, rec.get("pid_starttime")
        return pid


def cancel_subtree(sid: str, *, include_driver: bool = True) -> list[str]:
    """Cancel every OPEN background job in the session subtree rooted at ``sid``:
    jobs spawned BY sid or any descendant (parent_sid ∈ targets), and — when
    ``include_driver`` is true — the job whose runner DRIVES one of those
    sessions (child_sid ∈ targets). Kill each runner's whole process tree.
    Walks DOWN only. Returns the cancelled job_ids.

    Turn-retry must pass ``include_driver=False``: the job driving ``sid``
    is the retrying turn itself. Cancelling it rewrote a recoverable
    ``delegate_stall`` as ``cancelled (session subtree)``.
    """
    targets = {sid} | _descendant_sessions(sid)
    # Fence these sessions before taking the job snapshot. create() shares the
    # same lock: jobs published before the fence are already durable; jobs
    # attempted afterward are terminal at birth and cannot start a runner.
    _extend_cancel_fences(targets)
    candidates = [
        job
        for job in open_jobs()
        if job.get("parent_sid") in targets or (include_driver and job.get("child_sid") in targets)
    ]
    cancelled: list[str] = []
    for j in candidates:
        # Large cascades may exceed one lease. Refresh before each bounded
        # TERM/KILL cycle so later descendants cannot publish through expiry.
        _extend_cancel_fences(targets)
        identity = _claim_for_cancel(
            j["job_id"],
            "cancelled (session subtree)",
            return_identity=True,
        )
        if identity is None:  # raced to terminal — skip
            continue
        pid, starttime = identity
        _kill_runner_tree(pid, expected_starttime=starttime)
        cancelled.append(j["job_id"])
    return cancelled


def silence_deliveries(
    job_ids: Any,
    *,
    reason: str = "cancelled (parent abort)",
) -> None:
    """Record durable terminal results and mark them DELIVERED without waking
    any parent.

    Used by hard ``/abort`` cascades: killing background children must not let
    the live reconciler / delivery path POST a ``<task_error>…retry``
    prompt that revives the aborted parent ("unstoppable jarvis").
    """
    ids = [str(j) for j in (job_ids or []) if j]
    if not ids:
        return
    from sharedagents_core.layout import sessions_root
    from sharedagents_session.session_writer import get_session_writer

    sroot = sessions_root()
    by_parent: dict[str, list[str]] = {}
    for jid in ids:
        rec = read(jid)
        if rec is None:
            continue
        parent = str(rec.get("parent_sid") or "")
        if not parent:
            continue
        # A silenced member carries its reason so a wave sibling delivered
        # later can still recognize a parent abort and stay silent too.
        update(jid, delivery_silenced=reason or "cancelled")
        prior = _find_result_event(sroot, parent, jid)
        if prior is None:
            err = reason or rec.get("error") or "cancelled"
            try:
                get_session_writer(sroot, parent).lifecycle(
                    kind="subagent_result",
                    payload={
                        "job_id": jid,
                        "parent_sid": parent,
                        "child_sid": rec.get("child_sid") or "",
                        "agent": rec.get("agent") or "",
                        "call_id": rec.get("call_id") or "",
                        "status": rec.get("status"),
                        "ok": False,
                        "content": "",
                        "error": err,
                    },
                )
            except Exception:  # noqa: BLE001 — abort must not fail on bookkeeping
                pass
        by_parent.setdefault(parent, []).append(jid)
    for parent, jids in by_parent.items():
        mark_delivered(parent, jids)


def cancel_subtree_for_abort(sid: str) -> list[str]:
    """Hard-abort cascade: kill the background subtree and suppress parent wakes.

    ``/abort`` on a session owns the stop. Cancelling its background jobs must
    leave the aborted session dead — durable ``subagent_result`` yes, automatic
    parent resume no.
    """
    cancelled = cancel_subtree(sid)
    if cancelled:
        silence_deliveries(
            cancelled,
            reason="cancelled (parent abort)",
        )
    return cancelled


def cancel_job_ids(job_id: str) -> list[str]:
    """Cancel ONE job and the subtree it spawned: kill its runner tree, then
    cascade into its child session. Returns every job id newly cancelled,
    including descendants, so each one can be delivered independently.
    """
    rec = read(job_id)
    if rec is None:
        return []
    cancelled: list[str] = []
    identity = _claim_for_cancel(
        job_id,
        "cancelled (job)",
        return_identity=True,
    )
    if identity is not None:  # was open → kill its runner tree
        pid, starttime = identity
        _kill_runner_tree(pid, expected_starttime=starttime)
        cancelled.append(job_id)
    # Cascade into the child session subtree REGARDLESS of whether the root was
    # still open: a finished root can have left running background grandchildren.
    # Re-read for the freshest child_sid (the runner may have recorded it after
    # our first read but before the claim).
    fresh = read(job_id) or rec
    child = fresh.get("child_sid")
    if child:
        cancelled.extend(cancel_subtree(child))
    return list(dict.fromkeys(cancelled))


def cancel_job(job_id: str) -> bool:
    """Cancel a job subtree while preserving the historical boolean API.

    ``True`` means the named job exists, including when it was already terminal;
    terminal roots are still traversed so any open descendants are cancelled.
    Callers that must independently deliver every cancellation use
    :func:`cancel_job_ids`.
    """
    if read(job_id) is None:
        return False
    cancel_job_ids(job_id)
    return True


def _pid_alive(pid: int) -> bool:
    return oscompat.process_alive(pid)


def _job_runner_alive(job: dict[str, Any]) -> bool:
    """Identity-aware liveness for durable async-job runner records."""
    pid = int(job.get("pid") or 0)
    if not _pid_alive(pid):
        return False
    expected = job.get("pid_starttime")
    if expected is None:
        # Legacy and bounded pre-claim records did not carry a birth identity.
        return True
    try:
        return _proc_starttime(pid) == int(expected)
    except (TypeError, ValueError):
        return False


# Maximum age for a terminal job to be eligible for parent-resume recovery.
# Set to 0: no jobs are ever re-fired on orchestrator restart — the live
# reconciler (2s loop) handles runtime recovery while the process stays up.
# Boot recovery is intentionally disabled to prevent any stale conversation
# reactivation.
RECONCILE_MAX_AGE_SECONDS = 0  # no boot recovery — live reconciler only


def _job_age_seconds(job: dict[str, Any], now: float | None = None) -> float | None:
    """Return the age in seconds of a terminal job.

    Uses ``ended_at`` when available (set atomically with the terminal status),
    falling back to ``created_at``. Returns None when neither is parseable
    (treat-as-recent to preserve existing recovery behavior for records with
    missing timestamps).
    """
    if now is None:
        now = time.time()
    ts = job.get("ended_at") or job.get("created_at")
    if ts is None:
        return None
    try:
        return now - float(ts)
    except (ValueError, TypeError):
        return None


def _find_result_event(sessions_dir: Any, parent_sid: str, job_id: str) -> dict[str, Any] | None:
    """Return the last ``subagent_result`` for ``job_id`` already on the
    parent log, or None. Lets reconcile stay idempotent: a runner that
    delivered the result but died before terminalizing the job must NOT get
    a duplicate (error) result re-emitted.
    """
    from sharedagents_session import session_reader

    path = Path(sessions_dir) / f"{parent_sid}.jsonl"
    found = None
    for ev, _ in session_reader.tail_jsonl(path):
        if ev.get("type") == "subagent_result" and ev.get("job_id") == job_id:
            found = ev
    return found


def reconcile(max_age_seconds: int = 0) -> list[dict[str, Any]]:
    """At boot: any open job whose runner pid is dead never finished. If the
    result was already delivered to the parent log, just terminalize the
    record (no duplicate emit); otherwise mark failed AND emit a
    ``subagent_result`` error so no consumer waits forever. (Wired into
    orchestrator boot in Task 7.)

    A runner can die AFTER terminalizing a job but BEFORE waking the parent.
    Therefore a second pass independently claims every terminal task that is not
    delivered and whose runner is gone. Waves wait for all members and claim
    one durable continuation; legacy jobs remain independent.

    ``max_age_seconds`` controls the recovery window for parent-resume.
    Boot callers pass 0 (no recovery — prevents stale conversation
    reactivation on restart). The live reconciler passes 600 (10 min).

    Returns the list of resume intents (possibly empty): each is
    ``{"parent_sid","agent","ok","content","error","covered_ids"}``. Best-effort;
    the durable ``subagent_result`` is already on the log, and ``covered_ids`` is
    the claim token the caller releases if the resume POST later fails.
    """
    from sharedagents_core.layout import sessions_root
    from sharedagents_session.session_writer import get_session_writer

    sroot = sessions_root()
    # Pass 1: terminalize every open job whose runner died mid-flight. Adopt a
    # delivered result's outcome, else mark failed and emit an interrupted result.
    for j in open_jobs():
        if (
            j.get("status") == "started"
            and int(j.get("pid") or 0) == 0
            and time.time() - float(j.get("created_at") or 0) < 10.0
        ):
            # Submission writes pid=0 before spawn/update. A live reconciler
            # must not mistake that bounded startup window for a dead runner.
            continue
        if _job_runner_alive(j):
            continue
        prior = _find_result_event(sroot, j["parent_sid"], j["job_id"])
        if prior is not None:
            ok = bool(prior.get("ok"))
            update(
                j["job_id"],
                status=prior.get("status") or ("done" if ok else "failed"),
                ok=ok,
                error=None if ok else prior.get("error"),
            )
        else:
            error = "interrupted (runner gone)"
            update(j["job_id"], status="failed", ok=False, error=error)
            try:
                w = get_session_writer(sroot, j["parent_sid"])
                w.lifecycle(
                    kind="subagent_result",
                    payload={
                        "job_id": j["job_id"],
                        "parent_sid": j["parent_sid"],
                        "child_sid": "",
                        "agent": j.get("agent", ""),
                        "status": "failed",
                        "call_id": j.get("call_id") or "",
                        "ok": False,
                        "content": "",
                        "error": error,
                    },
                )
            except Exception:  # noqa: BLE001 — boot must not crash on a bad record
                pass
    # Pass 2: every independently-undelivered terminal task is a candidate
    # stranded parent continuation, including tasks terminalized before a crash.
    #
    # Time guard: terminal jobs older than max_age_seconds are permanently
    # silenced rather than re-fired. Boot callers pass 0 → no recovery.
    # Live reconciler passes 600 → recovers recent in-process crashes.
    resumes: list[dict[str, Any]] = []
    now = time.time()
    for j in list_jobs():
        if j.get("status") not in TERMINAL:
            continue
        parent_sid = j["parent_sid"]
        if j["job_id"] in _read_covered(parent_sid):
            continue  # already DELIVERED (final) → never re-fire (Residual 1)
        if _job_runner_alive(j):
            continue  # runner still alive → it will resume itself; don't race it
        try:
            covered_ids = claim_idle_resume_ids(parent_sid, j["job_id"])
            if covered_ids is None:
                continue
            # Age the WAVE from its latest terminal member. An early finisher
            # may wait hours for a sibling and must never be aged out alone.
            members = [read(jid) or {} for jid in covered_ids]
            ages = [_job_age_seconds(member, now) for member in members]
            known_ages = [age for age in ages if age is not None]
            if known_ages and min(known_ages) > max_age_seconds:
                mark_delivered(parent_sid, covered_ids)
                continue
            prior = _find_result_event(sroot, parent_sid, j["job_id"])
            ok = bool(prior.get("ok")) if prior is not None else bool(j.get("ok"))
            content = str((prior or {}).get("content") or "")
            error = None if ok else ((prior or {}).get("error") if prior is not None else j.get("error"))
            if prior is None:
                # Cancellation may kill its runner before the runner can emit.
                # Restore the durable terminal result before scheduling delivery.
                try:
                    get_session_writer(sroot, parent_sid).lifecycle(
                        kind="subagent_result",
                        payload={
                            "job_id": j["job_id"],
                            "parent_sid": parent_sid,
                            "child_sid": j.get("child_sid", ""),
                            "agent": j.get("agent", ""),
                            "status": j.get("status"),
                            "call_id": j.get("call_id") or "",
                            "ok": ok,
                            "content": content,
                            "error": error,
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
            resumes.append(
                {
                    "parent_sid": parent_sid,
                    "agent": j.get("agent", ""),
                    "ok": ok,
                    "content": content,
                    "error": error,
                    "job_id": j["job_id"],
                    "task_id": j.get("child_sid", ""),
                    "covered_ids": covered_ids,
                }
            )
        except Exception:  # noqa: BLE001 — never let recovery crash boot
            pass
    return resumes
