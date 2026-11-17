"""Execution leases + capability kill — spec §10 (minimal capability layer).

The spec's boundary is the OS (userns / cgroup v2 / pidns / mount-ns / seccomp);
this module adds the *capability model* that governs it without re-implementing
the OS primitives we already have. Each untrusted execution unit (a tool exec,
a spawned ``*_cli``/``acp`` agent, a subagent) runs under an
:class:`ExecutionLease` issued by the :class:`SandboxBroker`; the lease is the
only authority for *who may kill whom*.

Kill is **capability-based and downward-only** (§10.3): a unit may kill only its
own lineage descendants — never a sibling, an ancestor, or the gateway /
orchestrator supervisor. We reuse the existing process-group kill
(``tools.runners._kill_group`` → ``os.killpg``) as the *mechanism* and keep
``tools.safety.kills_sharedagents_gateway`` as defense-in-depth *behind* this
boundary (§10.3 "string guards survive only as defense-in-depth"). The OS
isolation tiers (§10.4) plug in as substrates later; the lease contract here is
substrate-independent so they need no changes to this layer.
"""

from __future__ import annotations

import contextvars
import enum
import logging
import os
import signal
import threading
from dataclasses import dataclass, field

from sharedagents_core import oscompat

log = logging.getLogger(__name__)

# The privileged supervisor lineage id — the gateway+orchestrator. Nothing an
# agent is issued may ever kill it (§10.2 immutable supervisor cgroup).
SUPERVISOR_ID = "__supervisor__"

# --- the one shape, one knob (spec §4.2 / §6.4) --------------------------
# All caps derive from a 3-wide, 3-deep agent tree. Never a bare literal:
# the full tree below a session root has 3 + 9 + 27 = 39 agent nodes, which
# is *also* the unique constant under which the cross-process per-subtree
# budget split (§6.4) equals the global budget with no remainder.
BRANCHING = 3
DEPTH = 3
MAX_LIVE_AGENTS = sum(BRANCHING**d for d in range(1, DEPTH + 1))  # = 39


# Orchestration cap — SEPARATE from the resource DEPTH above and enforced at a
# DIFFERENT layer. The resource tree (DEPTH=3, 39 nodes) governs the budget/kill
# accounting and must stay intact. This constant governs how many agents may chain
# in a *delegation*, enforced by the ENGINE stripping the Agent/BatchAgent/
# delegate_* tools from an agent already at this depth (wire + dispatch gate) — NOT
# by the broker (which still books the full resource tree). Default 2 (agent_depth)
# = **3 levels** counting the root: root(0) → A(1) → B(2); B is a leaf that holds no
# delegation tool, so a 4th level is never created. Env-overridable.
# Owner-set 2026-07-16 after a main→plan→code→code 4-level chain.
def _deleg_env_int(_name: str, _default: int) -> int:
    try:
        return int(os.environ.get(_name, str(_default)))
    except (TypeError, ValueError):
        return _default


MAX_DELEGATION_DEPTH = max(1, min(DEPTH, _deleg_env_int("SHAREDAGENTS_MAX_DELEGATION_DEPTH", 2)))


def _subtree_entitlement(agent_depth: int) -> int:
    """Agents an agent at ``agent_depth`` may have *below* it (spec §6.4).

    ``E(d) = Σ_{k=1..(DEPTH−d)} BRANCHING**k`` → E(1)=12, E(2)=3, E(3)=0.
    A cross-process reservation books ``1 + E(d)`` atomically so a child
    process can never spend more than its subtree was ever entitled to.
    """
    return sum(BRANCHING**k for k in range(1, max(0, DEPTH - agent_depth) + 1))


class LeaseClass(str, enum.Enum):
    """What an execution lease counts as (spec §4.1).

    Only ``AGENT`` leases are counted against the session budget and
    agent-depth-checked; ``TOOL``/``BOOKKEEPING`` anchor lineage for the
    capability-kill model but are free.
    """

    AGENT = "agent"  # a delegated agent (subagent / acp / *_cli). COUNTED, depth-checked, pre-spawn reserved.
    TOOL = "tool"  # a tool exec (Bash / monitor). NOT counted, post-spawn, best-effort.
    BOOKKEEPING = "book"  # turn roots / child-process-local roots. NOT counted; anchors lineage.


# Counted-agent ``kind`` tags that MUST go through :func:`reserve_agent`
# (pre-spawn, spec §5.1) — the delegated-agent fan-out primitives. lease_process
# fails closed on these so a fan-out site can't drift back to post-spawn leasing.
#
# NOTE: ``cli``/``acp`` are NOT here. Per spec §3.1 the counting unit is the
# *agent*, not the OS process: a delegated CLI/ACP agent is counted once at its
# agent-turn (``_drive_child_agent_turn`` → :func:`reserve_agent`), and the
# ``lease_process(kind="cli"/"acp")`` call at its engine's spawn is only the
# capability-kill bookkeeping lease for that same agent's process — uncounted,
# never a second agent.
_AGENT_KINDS = frozenset({"agent", "subagent"})


class SpawnRefused(RuntimeError):
    """An AGENT-class spawn was refused by the sandbox broker (spec §5.1).

        Loud by design: callers MUST surface this to the model/operator
    ; it is never converted to ``None``.
    """

    def __init__(
        self, *, reason: str, session_key: str, live: int, cap: int, agent_depth: int, kind: str, parent_id: str | None
    ) -> None:
        self.reason = reason  # "depth" | "session_total"
        self.session_key = session_key
        self.live = live
        self.cap = cap
        self.agent_depth = agent_depth
        self.kind = kind
        self.parent_id = parent_id
        if reason == "depth":
            msg = f"agent spawn refused: agent-depth cap {DEPTH} exceeded (would be depth {agent_depth}; kind={kind})"
        else:
            msg = (
                f"agent spawn refused: session {session_key!r} at live-agent cap "
                f"({live}/{cap}; kind={kind}, parent={parent_id})"
            )
        super().__init__(msg)


@dataclass(frozen=True)
class ExecutionLease:
    """The capability an execution unit runs under (§10.1).

    ``lineage`` is the root→self chain of agent ids; ``kill_scope`` is the set
    of lineage ids this lease may signal (filled by the broker as descendants
    are issued). ``pgid`` is the OS process-group the lease's worker runs in
    (the kill mechanism). ``signature`` is reserved for remote-node leases.
    """

    lease_id: str
    session_id: str
    turn_id: str
    agent_id: str
    parent_id: str | None
    lineage: tuple[str, ...]
    fs_roots: tuple[str, ...] = ()
    network_policy: str = "deny"
    pgid: int | None = None
    signature: str | None = None
    kill_scope: set[str] = field(default_factory=set)
    lease_class: str = LeaseClass.AGENT.value
    # Number of AGENT-class ancestors incl. self (spec §4.1); the measure the
    # depth cap applies to. TOOL/BOOKKEEPING leases inherit their parent's.
    agent_depth: int = 0

    @property
    def depth(self) -> int:
        return len(self.lineage) - 1


class SandboxBroker:
    """Issues execution leases and enforces capability-based kill (§10).

    One broker per orchestrator (the only component holding OS resources, §5).
    Thread-safe: leases are issued/revoked from the turn supervisor and tool
    runners across threads.
    """

    def __init__(
        self,
        *,
        max_depth: int = DEPTH,
        max_live_agents: int = MAX_LIVE_AGENTS,
        agent_depth_base: int = 0,
        session_key_override: str | None = None,
    ) -> None:
        """Construct a broker.

        The shipped constructor uses the derivation constants (spec §4.3);
        the kwargs exist for unit tests and for the env-delegated CHILD broker
        (§6.3), whose ``agent_depth_base``/``session_key_override``/
        ``max_live_agents`` come from the parent-supplied grant.
        """
        self._lock = threading.RLock()
        self._leases: dict[str, ExecutionLease] = {}
        # parent agent_id -> set of child agent_ids (the live spawn tree).
        self._children: dict[str, set[str]] = {}
        # process group id -> agent_id, so a kill site that only has the OS
        # pgid (the shared ``_kill_group`` chokepoint) can resolve the lease.
        self._by_pgid: dict[int, str] = {}
        self._max_depth = max_depth
        self._max_live_agents = max_live_agents
        self._agent_depth_base = agent_depth_base
        self._session_key_override = session_key_override
        # session_key -> live AGENT budget consumed (concurrently-live, §3.4).
        self._session_live: dict[str, int] = {}
        # agent_id -> (session_key, booked_amount) for a live reservation.
        self._reservations: dict[str, tuple[str, int]] = {}
        self._counter = 0

    # -- issue / revoke -----------------------------------------------------

    def issue_lease(
        self,
        *,
        session_id: str,
        turn_id: str,
        agent_id: str,
        parent_id: str | None = None,
        fs_roots: tuple[str, ...] = (),
        network_policy: str = "deny",
        pgid: int | None = None,
        lease_class: str = LeaseClass.AGENT.value,
    ) -> ExecutionLease:
        """Register a lease under ``parent_id``'s lineage (the low-level primitive).

        This does NOT enforce the session/depth caps — that is done atomically
        in :meth:`reserve_agent` for AGENT-class spawns (spec §5.1), which is
        the ONLY path that may create a counted, depth-limited agent. TOOL and
        BOOKKEEPING leases are uncounted and depth-exempt (spec §4.1 — fixes
        L1, where a depth-3 agent's Bash exec was refused then run unleased).
        Raises ``ValueError`` only on an unknown parent.
        """
        with self._lock:
            if parent_id is not None and parent_id != SUPERVISOR_ID:
                parent = self._lease_by_agent(parent_id)
                if parent is None:
                    raise ValueError(f"unknown parent lease: {parent_id!r}")
                lineage = parent.lineage + (agent_id,)
                parent_agent_depth = parent.agent_depth
            else:
                lineage = (agent_id,)
                parent_agent_depth = self._agent_depth_base
            is_agent = lease_class == LeaseClass.AGENT.value
            agent_depth = parent_agent_depth + (1 if is_agent else 0)

            self._counter += 1
            lease = ExecutionLease(
                lease_id=f"lease-{self._counter}",
                session_id=session_id,
                turn_id=turn_id,
                agent_id=agent_id,
                parent_id=parent_id,
                lineage=lineage,
                fs_roots=fs_roots,
                network_policy=network_policy,
                pgid=pgid,
                kill_scope={agent_id},  # a lease may always kill its own subtree
                lease_class=lease_class,
                agent_depth=agent_depth,
            )
            self._leases[agent_id] = lease
            if pgid is not None:
                self._by_pgid[pgid] = agent_id
            if parent_id:
                self._children.setdefault(parent_id, set()).add(agent_id)
                # Every ancestor's kill_scope gains this new descendant.
                for anc in lease.lineage[:-1]:
                    anc_lease = self._leases.get(anc)
                    if anc_lease is not None:
                        anc_lease.kill_scope.add(agent_id)
            return lease

    # -- pre-spawn agent gating (spec §5) -----------------------------------

    def reserve_agent(
        self,
        *,
        kind: str,
        session_id: str = "",
        turn_id: str = "",
        parent_id: str | None = SUPERVISOR_ID,
        cross_process: bool = False,
        agent_id: str | None = None,
    ) -> AgentReservation:
        """Atomically book an AGENT slot BEFORE its process is spawned (spec §5.1).

        The reservation *is* the count: a single broker-locked op that either
        books the slot(s) or raises :class:`SpawnRefused` — never ``None``,
        never after the fact. Enforces both caps:

        * The new agent's agent-depth must be ``<= DEPTH``.
        * Its booked amount must fit the session budget.

        With ``cross_process=True`` the *entire* subtree entitlement
        ``1 + E(depth)`` is booked up front (spec §6.4) so a child OS process
        can be handed a fixed grant and can never overspend it.
        """
        with self._lock:
            if parent_id is not None and parent_id != SUPERVISOR_ID:
                parent = self._leases.get(parent_id)
                parent_agent_depth = parent.agent_depth if parent is not None else self._agent_depth_base
            else:
                parent_agent_depth = self._agent_depth_base
            agent_depth = parent_agent_depth + 1
            skey = self._session_key(session_id, parent_id)

            if agent_depth > self._max_depth:
                raise SpawnRefused(
                    reason="depth",
                    session_key=skey,
                    live=self._session_live.get(skey, 0),
                    cap=self._max_live_agents,
                    agent_depth=agent_depth,
                    kind=kind,
                    parent_id=parent_id,
                )
            subtree_grant = _subtree_entitlement(agent_depth) if cross_process else 0
            amount = 1 + subtree_grant

            live = self._session_live.get(skey, 0)
            # Re-issue of the same agent_id replaces, never double-counts (§4.3).
            prior = self._reservations.get(agent_id or "")
            prior_amount = prior[1] if prior else 0
            new_live = live - prior_amount + amount
            if new_live > self._max_live_agents:
                raise SpawnRefused(
                    reason="session_total",
                    session_key=skey,
                    live=live,
                    cap=self._max_live_agents,
                    agent_depth=agent_depth,
                    kind=kind,
                    parent_id=parent_id,
                )

            self._counter += 1
            aid = agent_id or f"{kind}-{self._counter}"
            if aid in self._reservations:
                self._release_reservation(aid)
            lease = self.issue_lease(
                session_id=session_id,
                turn_id=turn_id,
                agent_id=aid,
                parent_id=parent_id,
                pgid=None,
                lease_class=LeaseClass.AGENT.value,
            )
            # issue_lease derives agent_depth from parent lineage; for a
            # SUPERVISOR-parented lease that equals base+1 — the same value.
            self._session_live[skey] = self._session_live.get(skey, 0) + amount
            self._reservations[aid] = (skey, amount)
            return AgentReservation(
                broker=self,
                agent_id=aid,
                lease=lease,
                subtree_grant=subtree_grant,
                agent_depth=agent_depth,
                session_key=skey,
                amount=amount,
            )

    def _session_key(self, session_id: str, parent_id: str | None) -> str:
        """Bucket a spawn to a session (spec §4.3). A CHILD broker collapses all
        buckets to its inherited session so a spawn can't dodge the budget."""
        if self._session_key_override is not None:
            return self._session_key_override
        if session_id:
            return session_id
        if parent_id and parent_id != SUPERVISOR_ID:
            parent = self._leases.get(parent_id)
            if parent is not None:
                return parent.session_id or (parent.lineage[0] if parent.lineage else parent_id)
            return parent_id
        return ""

    def _bind_pgid(self, agent_id: str, pgid: int | None) -> None:
        """Attach the spawned OS pgid to a reservation's lease (spec §5.1 bind)."""
        if pgid is None:
            return
        with self._lock:
            lease = self._leases.get(agent_id)
            if lease is None:
                return
            object.__setattr__(lease, "pgid", pgid)
            self._by_pgid[pgid] = agent_id

    def _release_reservation(self, agent_id: str) -> None:
        """Return a reservation's booked amount to its session budget and revoke
        the lease. Idempotent (spec §5.1 release)."""
        with self._lock:
            booked = self._reservations.pop(agent_id, None)
            if booked is not None:
                skey, amount = booked
                remaining = self._session_live.get(skey, 0) - amount
                if remaining > 0:
                    self._session_live[skey] = remaining
                else:
                    self._session_live.pop(skey, None)
            self.revoke(agent_id)

    def session_live(self, session_key: str) -> int:
        """Live AGENT budget consumed by ``session_key`` (test/observability)."""
        with self._lock:
            return self._session_live.get(session_key, 0)

    def session_has_live_lease(self, session_key: str) -> bool:
        """Return whether any in-process lease still owns ``session_key``.

        ``session_live`` intentionally counts only AGENT reservations for the
        fan-out budget.  The stale-session reconciler needs a different fact:
        a local turn root or an uncounted tool/engine lease is proof that the
        session still has an in-process owner and must not be folded as a
        ghost merely because it has not emitted a JSONL delta recently.
        """
        key = str(session_key or "")
        if not key:
            return False
        with self._lock:
            return any(str(lease.session_id or "") == key for lease in self._leases.values())

    def revoke(self, agent_id: str) -> None:
        """Drop a lease and detach it from its parent's child set."""
        with self._lock:
            lease = self._leases.pop(agent_id, None)
            if lease is None:
                return
            if lease.pgid is not None and self._by_pgid.get(lease.pgid) == agent_id:
                self._by_pgid.pop(lease.pgid, None)
            if lease.parent_id:
                kids = self._children.get(lease.parent_id)
                if kids:
                    kids.discard(agent_id)
            for anc in lease.lineage[:-1]:
                anc_lease = self._leases.get(anc)
                if anc_lease is not None:
                    anc_lease.kill_scope.discard(agent_id)

    # -- capability checks --------------------------------------------------

    def can_kill(self, killer_id: str, target_id: str) -> bool:
        """May ``killer_id`` signal ``target_id``? Downward-only (§10.3).

        True iff ``target`` is a strict lineage descendant of ``killer``.
        Never a sibling, an ancestor, the killer itself, or the supervisor.
        The orchestrator drives all kills, so ``killer_id=SUPERVISOR_ID``
        (the root capability) may kill any issued lease.
        """
        with self._lock:
            if target_id == SUPERVISOR_ID:
                return False  # nothing may kill the supervisor
            target = self._leases.get(target_id)
            if target is None:
                return False
            if killer_id == SUPERVISOR_ID:
                return True
            if killer_id == target_id:
                return False  # self-kill is the cancel path, not a capability
            # Downward-only: killer must be a proper ancestor in target lineage.
            return killer_id in target.lineage[:-1]

    def descendants(self, agent_id: str) -> set[str]:
        """All lineage descendants of ``agent_id`` (its kill cascade set)."""
        with self._lock:
            out: set[str] = set()
            stack = list(self._children.get(agent_id, ()))
            while stack:
                child = stack.pop()
                if child in out:
                    continue
                out.add(child)
                stack.extend(self._children.get(child, ()))
            return out

    # -- kill mechanism (reuses the existing process-group primitive) -------

    def capability_kill(
        self,
        *,
        killer_id: str,
        target_id: str,
        sig: int = signal.SIGTERM,
        command: str = "",
    ) -> bool:
        """Kill ``target``'s process group iff the capability check passes.

        Returns ``True`` if the signal was authorised + delivered. Refuses
        (``False``) when the lineage check fails — the OS boundary the spec
        mandates (pidns) makes this physically true; here it is enforced in
        the capability layer. ``command`` is checked against the gateway
        self-protection guard as defense-in-depth behind the boundary.
        """
        from sharedagents_agent.tools.safety import kills_sharedagents_gateway

        if command and kills_sharedagents_gateway(command):
            return False  # defense-in-depth: never the gateway/supervisor
        if not self.can_kill(killer_id, target_id):
            return False
        with self._lock:
            target = self._leases.get(target_id)
            pgid = target.pgid if target is not None else None
        if pgid is None:
            return False
        try:
            oscompat.signal_process_group(pgid, sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    # -- internal -----------------------------------------------------------

    def _lease_by_agent(self, agent_id: str) -> ExecutionLease | None:
        return self._leases.get(agent_id)


@dataclass
class AgentReservation:
    """A booked AGENT slot returned by :meth:`SandboxBroker.reserve_agent`.

    The lease is issued immediately (``pgid=None``); the caller spawns the
    process, calls :meth:`bind` with its pgid, and :meth:`release` in a
    ``finally``. ``release`` is idempotent and returns ``1 + subtree_grant``
    to the session budget, so a SIGKILLed child can't leak budget (spec §6.2).
    """

    broker: SandboxBroker
    agent_id: str
    lease: ExecutionLease
    subtree_grant: int
    agent_depth: int
    session_key: str
    amount: int
    _released: bool = False

    def bind(self, pgid: int | None) -> None:
        """Register the spawned process group on the reservation's lease."""
        self.broker._bind_pgid(self.agent_id, pgid)

    def release(self) -> None:
        """Return the booked budget and revoke the lease (idempotent)."""
        if self._released:
            return
        self._released = True
        try:
            self.broker._release_reservation(self.agent_id)
        except Exception:  # noqa: BLE001 — release must never raise
            pass

    def child_env(self, base_env: dict) -> dict:
        """Env vars handing this reservation's subtree grant to a child process
        (spec §6.3/§6.5). The child's :func:`get_broker` initialises from these.
        A copy is returned; ``base_env`` is not mutated."""
        env = dict(base_env)
        env["SHAREDAGENTS_SANDBOX_CHILD"] = "1"
        env["SHAREDAGENTS_SANDBOX_GRANT"] = str(self.subtree_grant)
        env["SHAREDAGENTS_SANDBOX_AGENT_DEPTH"] = str(self.agent_depth)
        env["SHAREDAGENTS_SANDBOX_SESSION"] = self.session_key or ""
        return env


# Process-wide broker — one per orchestrator process (the only component
# holding OS resources, §5). Tool runners + the turn supervisor share it so a
# lease issued at spawn is the same capability the kill cascade validates.
_BROKER: SandboxBroker | None = None
_BROKER_LOCK = threading.Lock()


def _child_broker_from_env() -> SandboxBroker | None:
    """Build a CHILD broker from the parent-delegated env grant (spec §6.3/§6.5).

    Only honoured under a child-boot (``SHAREDAGENTS_SANDBOX_CHILD=1``) so an
    operator's shell env can never reshape the orchestrator's root broker.
    Clamps ``grant <= MAX_LIVE_AGENTS`` and ``0 <= agent_depth_base <= DEPTH``
    (no shipped/inherited config raises the cap); an
    out-of-range value is treated as absent and logged.
    """
    if os.environ.get("SHAREDAGENTS_SANDBOX_CHILD") != "1":
        return None
    if "SHAREDAGENTS_SANDBOX_GRANT" not in os.environ:
        return None
    try:
        grant = int(os.environ.get("SHAREDAGENTS_SANDBOX_GRANT", ""))
        depth_base = int(os.environ.get("SHAREDAGENTS_SANDBOX_AGENT_DEPTH", "0"))
    except (TypeError, ValueError):
        log.warning("sandbox: malformed child-grant env; using full-cap broker")
        return None
    if not (0 <= grant <= MAX_LIVE_AGENTS) or not (0 <= depth_base <= DEPTH):
        log.warning(
            "sandbox: out-of-range child grant (grant=%s depth=%s); using full-cap broker",
            grant,
            depth_base,
        )
        return None
    session = os.environ.get("SHAREDAGENTS_SANDBOX_SESSION", "") or "child"
    return SandboxBroker(
        max_live_agents=grant,
        agent_depth_base=depth_base,
        session_key_override=session,
    )


def get_broker() -> SandboxBroker:
    """The shared orchestrator-process SandboxBroker (lazily created).

    In a child process spawned with an env grant (spec §6.3) the broker starts
    pre-loaded from that grant instead of empty — so cross-process fan-out can
    no longer escape the session budget."""
    global _BROKER
    if _BROKER is None:
        with _BROKER_LOCK:
            if _BROKER is None:
                _BROKER = _child_broker_from_env() or SandboxBroker()
    return _BROKER


def _reset_broker_for_tests() -> None:
    global _BROKER
    _BROKER = None


# --- spawn-site helpers ---------------------------------------------------
#
# Every untrusted execution unit (the Bash tool, a spawned ``*_cli``/``acp``
# agent, a background shell monitor) wires isolation with the SAME three calls:
# lease the spawned process group, route its kill through the capability layer,
# and release the lease on exit. These wrappers keep that pattern a one-liner at
# each site and are all best-effort — leasing never blocks a spawn, and the
# caller's existing group-kill stays the guaranteed fallback so isolation can
# never cost the existing kill behaviour.
_LEASE_SEQ = 0
_LEASE_SEQ_LOCK = threading.Lock()


# Per-turn lineage context. The orchestrator opens a turn-root lease and binds
# it here; tool/subagent leases issued during the turn nest UNDER it (so a
# lease carries the real session/turn and parent lineage, not a bare
# SUPERVISOR parent). contextvars propagate into the turn's async sub-tasks.
_current_turn: contextvars.ContextVar[dict | None] = contextvars.ContextVar("sharedagents_turn_lease", default=None)


def begin_turn(session_id: str, turn_id: str) -> tuple[object, str]:
    """Open a turn-root lease (under the supervisor) and bind it as the current
    turn context. Returns ``(token, root_agent_id)`` for :func:`end_turn`."""
    root = f"turn:{session_id}:{turn_id}"
    try:
        get_broker().issue_lease(
            session_id=session_id,
            turn_id=turn_id,
            agent_id=root,
            parent_id=SUPERVISOR_ID,
            pgid=None,
            lease_class=LeaseClass.BOOKKEEPING.value,
        )
    except Exception:  # noqa: BLE001 — a duplicate/again is harmless
        pass
    token = _current_turn.set({"session_id": session_id, "turn_id": turn_id, "agent_id": root})
    return token, root


def end_turn(token: object, root_agent_id: str) -> None:
    """Revoke the turn-root lease (cascading bookkeeping) and unbind context."""
    try:
        _current_turn.reset(token)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        pass
    try:
        broker = get_broker()
        # Revoke the whole turn subtree so a crashed tool can't leak leases.
        for child in broker.descendants(root_agent_id):
            broker.revoke(child)
        broker.revoke(root_agent_id)
    except Exception:  # noqa: BLE001
        pass


def current_turn() -> dict | None:
    """The bound turn context ``{session_id, turn_id, agent_id}`` or ``None``."""
    return _current_turn.get()


def reserve_agent(
    *,
    kind: str,
    session_id: str = "",
    turn_id: str = "",
    parent_id: str | None = None,
    cross_process: bool = False,
    agent_id: str | None = None,
) -> AgentReservation:
    """Module-level :meth:`SandboxBroker.reserve_agent` bound to the current turn.

    This is the ONLY way to obtain an AGENT-class lease. Like
    :func:`lease_process`, it defaults its lineage to the bound turn context
    (set by :func:`begin_turn`) so the agent nests under the real session/turn
    rather than a bare SUPERVISOR parent. Raises :class:`SpawnRefused`; never
    returns ``None``.
    """
    ctx = _current_turn.get()
    if ctx and not session_id and not turn_id and (parent_id is None or parent_id == SUPERVISOR_ID):
        session_id = ctx.get("session_id", "")
        turn_id = ctx.get("turn_id", "")
        parent_id = ctx.get("agent_id", SUPERVISOR_ID)
    if not session_id:
        # The ContextVar above is the preferred source, but it does NOT cross a
        # thread or a freshly-created event loop — and the delegate spawn path
        # does both. When it is unset every spawn books against the SAME bucket
        # keyed on the empty string, so MAX_LIVE_AGENTS (39), meant to be PER
        # SESSION, becomes one global counter that fills across every session
        # and turn and never drains.
        #
        # Observed: "agent spawn refused: session '' at live-agent cap
        # (39/39)". A five-way review fan-out got three results; runs 3 and 4
        # were refused in 0.00s and surfaced to the model as EMPTY answers, which
        # is why this read for months as "the external agents return nothing".
        # It is position-dependent, not engine-dependent, and degrades with
        # uptime — a restart clears it because the bucket is in-process.
        #
        # $SHAREDAGENTS_SESSION_ID is the process-scoped identity every log and lease
        # already keys on, so it is the correct fallback.
        import os as _os

        session_id = (_os.environ.get("SHAREDAGENTS_SESSION_ID") or "").strip()
    if parent_id is None:
        parent_id = SUPERVISOR_ID
    return get_broker().reserve_agent(
        kind=kind,
        session_id=session_id,
        turn_id=turn_id,
        parent_id=parent_id,
        cross_process=cross_process,
        agent_id=agent_id,
    )


def reserve_agent_or_precounted(
    args: dict | None,
    *,
    kind: str,
    cross_process: bool = False,
) -> AgentReservation | None:
    """Reserve an AGENT slot unless this dispatch was already counted upstream.

    A child-process top turn was already booked by the parent's
    ``cross_process`` reservation (spec §6.3); it arrives carrying a
    ``_sharedagents_precounted`` marker (set by the spawning helper). This consumes
    that marker and returns ``None`` (no double-count); otherwise it reserves
    a fresh slot. Raises :class:`SpawnRefused` on refusal.
    """
    if isinstance(args, dict) and args.pop("_sharedagents_precounted", False):
        return None
    return reserve_agent(kind=kind, cross_process=cross_process)


def lease_process(
    *,
    pgid: int | None,
    kind: str,
    session_id: str = "",
    turn_id: str = "",
    parent_id: str = SUPERVISOR_ID,
) -> str | None:
    """Issue a bookkeeping (uncounted) lease for a freshly spawned process group.

    ``kind`` is a capability-kill bookkeeping tag: ``exec`` / ``monitor`` (a
    genuine TOOL exec) OR ``cli`` / ``acp`` (a delegated agent's ENGINE process,
    already counted once at its agent-turn — spec §3.1). All are uncounted,
    depth-exempt, best-effort, and never block a spawn. The delegated-agent
    FAN-OUT kinds (``subagent`` / ``agent``) are refused with ``ValueError`` —
    they MUST go through :func:`reserve_agent` (pre-spawn, spec §5.1) so no
    fan-out site can drift back to the swallowed-``None``, post-spawn contract
    that let the cap be breached. Returns ``None`` on any other
    failure (caller falls back to its raw group kill).
    """
    if kind in _AGENT_KINDS:
        raise ValueError(
            f"lease_process refuses AGENT kind {kind!r}: use reserve_agent() (pre-spawn gating — spec §5.1)"
        )
    global _LEASE_SEQ
    # Bind to the current turn lineage by default (set by begin_turn): the
    # lease then carries the real session/turn and nests under the turn-root
    # lease instead of a bare SUPERVISOR parent. An explicit override still
    # wins. If the turn-root parent isn't registered, issue_lease will raise
    # and we degrade to None (caller keeps its legacy kill).
    ctx = _current_turn.get()
    if ctx and not session_id and not turn_id and parent_id == SUPERVISOR_ID:
        session_id = ctx.get("session_id", "")
        turn_id = ctx.get("turn_id", "")
        parent_id = ctx.get("agent_id", SUPERVISOR_ID)
    # ``pgid=None`` is allowed: a lineage-only lease (e.g. a spawned ACP server
    # whose group leadership is unknown) brings the unit into the capability
    # model; its capability kill then validates lineage and defers to the
    # caller's own kill mechanism rather than signalling a process group.
    with _LEASE_SEQ_LOCK:
        _LEASE_SEQ += 1
        agent_id = f"{kind}-{pgid}-{_LEASE_SEQ}"
    try:
        return (
            get_broker()
            .issue_lease(
                session_id=session_id,
                turn_id=turn_id,
                agent_id=agent_id,
                parent_id=parent_id,
                pgid=pgid,
                lease_class=LeaseClass.TOOL.value,
            )
            .agent_id
        )
    except Exception:  # noqa: BLE001
        return None


def release_process(lease_id: str | None) -> None:
    """Revoke a process lease (best-effort no-op when ``lease_id`` is None)."""
    if not lease_id:
        return
    try:
        get_broker().revoke(lease_id)
    except Exception:  # noqa: BLE001
        pass


def session_has_live_lease(session_key: str) -> bool:
    """Return whether the broker has any live lease for ``session_key``."""
    try:
        return get_broker().session_has_live_lease(session_key)
    except Exception:  # noqa: BLE001
        return False


def release_pgid(pgid: int | None) -> None:
    """Revoke whatever lease is registered for ``pgid`` (best-effort).

    For kill sites that only carry the OS pgid (background monitors) and want
    to release on stop without threading the lease id through.
    """
    if pgid is None:
        return
    broker = get_broker()
    with broker._lock:
        agent_id = broker._by_pgid.get(pgid)
    if agent_id:
        try:
            broker.revoke(agent_id)
        except Exception:  # noqa: BLE001
            pass


def enforce_kill_pgid(pgid: int, sig: int, *, command: str = "") -> str:
    """Capability-ENFORCING kill of a process group, looked up by pgid.

    Unlike :func:`try_capability_kill_pgid` (advisory boolean), this is the
    enforcement boundary the shared ``_kill_group`` uses, returning a status so
    the caller knows whether it may legacy-fall-back:

    * ``"no_lease"`` — no lease for this pgid; the caller MAY raw-``killpg`` it
      (legacy, un-leased process — no capability claim was ever made).
    * ``"killed"``  — leased + authorised + the signal was delivered.
    * ``"gone"``    — leased + authorised but the group was already absent.
    * ``"denied"``  — leased but the downward-only capability check (or the
      gateway self-protection guard) REFUSED. The caller MUST NOT raw-kill —
      that would bypass the boundary (§10.3).

    For a leased group the capability layer is therefore the ONLY path to the
    kill; the raw fallback exists solely for genuinely un-leased processes.
    """
    broker = get_broker()
    with broker._lock:
        agent_id = broker._by_pgid.get(pgid)
        lease = broker._leases.get(agent_id) if agent_id else None
    if not agent_id or lease is None:
        return "no_lease"
    if command:
        from sharedagents_agent.tools.safety import kills_sharedagents_gateway

        if kills_sharedagents_gateway(command):
            return "denied"
    if not broker.can_kill(SUPERVISOR_ID, agent_id):
        return "denied"
    if lease.pgid is None:
        return "denied"  # lineage-only lease — no pgid target to signal
    try:
        oscompat.signal_process_group(lease.pgid, sig)
        return "killed"
    except (ProcessLookupError, PermissionError, OSError):
        return "gone"


def try_capability_kill_pgid(pgid: int, sig: int, *, command: str = "") -> bool:
    """Capability-validated kill of a leased process group, looked up by pgid.

    The shared ``_kill_group`` chokepoint only has the OS pgid, so it resolves
    the lease here. Returns ``True`` iff a lease exists for ``pgid`` AND the
    downward-only capability check passed AND the signal was delivered; the
    caller falls back to a raw ``killpg`` otherwise (an un-leased pgid — e.g. a
    legacy spawn — therefore behaves exactly as before).
    """
    broker = get_broker()
    with broker._lock:
        agent_id = broker._by_pgid.get(pgid)
    if not agent_id:
        return False
    try:
        return broker.capability_kill(
            killer_id=SUPERVISOR_ID,
            target_id=agent_id,
            sig=sig,
            command=command,
        )
    except Exception:  # noqa: BLE001
        return False


def capability_kill_proc(
    lease_id: str | None,
    *,
    sig: int,
    command: str = "",
) -> bool:
    """Capability-validated downward-only kill of a leased process group.

    Returns ``True`` iff the signal was authorised AND delivered; the caller
    MUST fall back to its raw group kill when this returns ``False`` so a
    missing/stale lease never leaves a runaway group alive.
    """
    if not lease_id:
        return False
    try:
        return get_broker().capability_kill(
            killer_id=SUPERVISOR_ID,
            target_id=lease_id,
            sig=sig,
            command=command,
        )
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "SandboxBroker",
    "ExecutionLease",
    "SUPERVISOR_ID",
    "LeaseClass",
    "SpawnRefused",
    "AgentReservation",
    "BRANCHING",
    "DEPTH",
    "MAX_LIVE_AGENTS",
    "get_broker",
    "_reset_broker_for_tests",
    "reserve_agent",
    "reserve_agent_or_precounted",
    "lease_process",
    "release_process",
    "session_has_live_lease",
    "release_pgid",
    "capability_kill_proc",
    "try_capability_kill_pgid",
    "enforce_kill_pgid",
    "begin_turn",
    "end_turn",
    "current_turn",
]
