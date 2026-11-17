"""Deterministic, test-first SDLC flow used by the Dev control plane.

The module deliberately keeps the state machine and its persistence together:
all legal transitions are decided under a SQLite write lock, while artifacts
are immutable content-addressed records.  Stage runners submit evidence from
isolated contexts; this module never executes the submitted commands.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FlowError(RuntimeError):
    """Base error for invalid flow input or integrity evidence."""


class FlowConflict(FlowError):
    """A CAS, state, validation, or immutable-artifact guard rejected input."""


class HumanControlRequired(FlowConflict):
    """A human-only content/control operation was attempted by an AI actor."""


# Ratified Stage-0 instance.  canonical_hash preserves array order and sorts
# only object keys.  The pinned hashes record the instance as originally
# ratified, before the criterion IDs took the REQ-SDLC- prefix and the
# flowchart gained its current schema name; nothing validates against them,
# and they do not match the current CRITERION_IDS or FLOWCHART.
CRITERION_IDS = [f"REQ-SDLC-{i:04d}" for i in range(1, 11)]
RATIFIED_STAGE0_HASHES = {
    "criteria_sha256": "49643c72336bb8adc15b50591b18d24a574da864e9f3734fa2be8f30f0dd52c0",
    "flowchart_sha256": "3f923277658eb2bb79f71d4e7adc8afce6dba774ad175b82a0d566832936ec72",
}
FLOWCHART: dict[str, Any] = {
    "schema": "sharedagents-sdlc-flow/v1",
    "start": "S0",
    "nodes": [
        {"id": "S0", "type": "stage", "name": "Human: requirements + flowchart + success criteria"},
        {"id": "S1", "type": "stage", "name": "AI: generate + freeze acceptance tests"},
        {"id": "S2", "type": "stage", "name": "AI: technical spec + adapter binding"},
        {"id": "S3", "type": "stage", "name": "AI: implement code"},
        {"id": "S4", "type": "stage", "name": "Separate agent: review + run frozen manifest"},
        {"id": "DISPUTE", "type": "state", "name": "Test dispute (human decision)"},
        {"id": "DONE", "type": "terminal"},
        {"id": "FAILED", "type": "terminal"},
        {"id": "NEW_RUN", "type": "terminal"},
    ],
    "edges": [
        {"from": "S0", "to": "S1", "guard": "reqs+flow+criteria present, IDs valid, flowchart valid"},
        {
            "from": "S1",
            "to": "S2",
            "guard": "each test @trace to a criterion; suite red or BASELINE_ALREADY_SATISFIED; hash recorded",
        },
        {"from": "S2", "to": "S3", "guard": "spec names concrete files/tables/endpoints; consistent with frozen tests"},
        {"from": "S3", "to": "S4", "guard": "frozen tests green; test hash unchanged"},
        {"from": "S4", "to": "DONE", "guard": "reviewer APPROVE and manifest green"},
        {"from": "S4", "to": "S3", "guard": "REJECT type=impl"},
        {"from": "S4", "to": "S2", "guard": "REJECT type=spec"},
        {"from": "S4", "to": "DISPUTE", "guard": "REJECT type=test"},
        {"from": "S4", "to": "NEW_RUN", "guard": "REJECT type=requirement"},
        {"from": "DISPUTE", "to": "S3", "guard": "human UPHOLD_TEST"},
        {"from": "DISPUTE", "to": "S1", "guard": "human SUPERSEDE_TEST (new frozen baseline, old hash preserved)"},
        {"from": "DISPUTE", "to": "NEW_RUN", "guard": "human AMEND_REQUIREMENT"},
        {"from": "S0", "to": "FAILED", "guard": "validation fails"},
        {"from": "S1", "to": "FAILED", "guard": "validation fails"},
        {"from": "S2", "to": "FAILED", "guard": "validation fails"},
        {"from": "S3", "to": "FAILED", "guard": "validation fails"},
    ],
}


# The action names are stable API commands; destinations are exactly the
# ratified flowchart's destinations.  Failure destinations are represented by
# the validation-fail action for each stage.
TRANSITION_TABLE: tuple[dict[str, str], ...] = (
    {"source": "S0", "action": "submit_tests", "guard": "stage0_valid", "destination": "S1"},
    {"source": "S1", "action": "freeze_tests", "guard": "tests_traceable_red_or_baseline", "destination": "S2"},
    {"source": "S2", "action": "submit_spec", "guard": "spec_concrete_and_consistent", "destination": "S3"},
    {"source": "S3", "action": "submit_implementation", "guard": "tests_green_hash_unchanged", "destination": "S4"},
    {"source": "S4", "action": "approve", "guard": "review_approve_manifest_green", "destination": "DONE"},
    {"source": "S4", "action": "reject_impl", "guard": "typed_reject_impl", "destination": "S3"},
    {"source": "S4", "action": "reject_spec", "guard": "typed_reject_spec", "destination": "S2"},
    {"source": "S4", "action": "reject_test", "guard": "typed_reject_test", "destination": "DISPUTE"},
    {"source": "S4", "action": "reject_requirement", "guard": "typed_reject_requirement", "destination": "NEW_RUN"},
    {"source": "DISPUTE", "action": "UPHOLD_TEST", "guard": "human_disposition", "destination": "S3"},
    {"source": "DISPUTE", "action": "SUPERSEDE_TEST", "guard": "human_new_frozen_baseline", "destination": "S1"},
    {"source": "DISPUTE", "action": "AMEND_REQUIREMENT", "guard": "human_disposition", "destination": "NEW_RUN"},
    {"source": "S0", "action": "validation_failed", "guard": "stage0_invalid", "destination": "FAILED"},
    {"source": "S1", "action": "validation_failed", "guard": "stage1_invalid", "destination": "FAILED"},
    {"source": "S2", "action": "validation_failed", "guard": "stage2_invalid", "destination": "FAILED"},
    {"source": "S3", "action": "validation_failed", "guard": "stage3_invalid", "destination": "FAILED"},
)

# Public aliases make the exhaustive graph inspectable by the Dev tab and by
# independent verifier tests without exposing mutable registry internals.
EXHAUSTIVE_TRANSITION_TABLE = TRANSITION_TABLE
STATE_TRANSITIONS = TRANSITION_TABLE

_TRANSITIONS = {(row["source"], row["action"]): row["destination"] for row in TRANSITION_TABLE}
_TERMINAL = frozenset({"DONE", "FAILED", "NEW_RUN"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def append_event_hash(event: dict[str, Any], prior_event_hash: str) -> str:
    """Return the deterministic digest for one audit event."""
    return canonical_hash({"event": event, "prior_event_hash": prior_event_hash})


def validate_flowchart(flowchart: dict[str, Any]) -> dict[str, Any]:
    """Validate the machine-readable graph and return observable terminals."""
    if not isinstance(flowchart, dict) or flowchart.get("schema") != "sharedagents-sdlc-flow/v1":
        raise FlowError("unsupported flowchart schema")
    nodes = flowchart.get("nodes")
    edges = flowchart.get("edges")
    start = flowchart.get("start")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(start, str):
        raise FlowError("flowchart requires start, nodes, and edges")
    node_map: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise FlowError("node id is required")
        node_id = node["id"]
        if node_id in node_map or node.get("type") not in {"stage", "state", "terminal"}:
            raise FlowError("duplicate or invalid node")
        node_map[node_id] = node
    if start not in node_map:
        raise FlowError("start node is unknown")
    outgoing: dict[str, list[str]] = {key: [] for key in node_map}
    for edge in edges:
        if (
            not isinstance(edge, dict)
            or not edge.get("from")
            or not edge.get("to")
            or not str(edge.get("guard") or "").strip()
        ):
            raise FlowError("every edge requires from, to, and a non-empty guard")
        if edge["from"] not in node_map or edge["to"] not in node_map:
            raise FlowError("edge endpoint is unknown")
        if node_map[edge["from"]].get("type") == "terminal":
            raise FlowError("terminal cannot have outgoing edge")
        outgoing[edge["from"]].append(edge["to"])
    seen: set[str] = set()
    todo = [start]
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        todo.extend(outgoing[current])
    if seen != set(node_map):
        raise FlowError(f"unreachable nodes: {sorted(set(node_map) - seen)}")
    terminals = sorted(node_id for node_id, node in node_map.items() if node.get("type") == "terminal")
    reachable_terminals = sorted(set(terminals) & seen)
    if not reachable_terminals:
        raise FlowError("flowchart has no reachable terminal")
    branch_nodes = [node_id for node_id, destinations in outgoing.items() if len(destinations) > 1]
    for node_id in branch_nodes:
        guards = [str(edge.get("guard") or "").strip() for edge in edges if edge.get("from") == node_id]
        if len(guards) != len(set(guards)) or any(not guard for guard in guards):
            raise FlowError(f"unlabeled or duplicate branch guard at {node_id}")
    if flowchart == FLOWCHART:
        return {"ok": True, "reachable_terminals": reachable_terminals}
    return {"ok": True, "reachable_terminals": reachable_terminals}


def extract_rtm(cases: Iterable[dict[str, Any]], criterion_ids: Iterable[str]) -> dict[str, Any]:
    criteria = {str(value.get("id") if isinstance(value, dict) else value) for value in criterion_ids}
    case_rows = list(cases)
    untraced_tests: list[str] = []
    traced: dict[str, list[str]] = {}
    for case in case_rows:
        case_id = str(case.get("case_id") or case.get("id") or "")
        refs = case.get("criterion_ids") or case.get("criteria") or []
        refs = [str(ref) for ref in refs]
        if not case_id or not refs or any(ref not in criteria for ref in refs):
            untraced_tests.append(case_id)
        for ref in refs:
            traced.setdefault(ref, []).append(case_id)
    untested = sorted(criteria - set(traced))
    return {
        "untraced_tests": sorted(untraced_tests),
        "untested_criteria": untested,
        "traced": {key: sorted(value) for key, value in sorted(traced.items())},
    }


def route_rework(kind: str) -> str:
    route = {"impl": "S3", "spec": "S2", "test": "DISPUTE", "requirement": "NEW_RUN"}.get(str(kind))
    if route is None:
        raise FlowError(f"unknown problem type: {kind}")
    return route


def _json(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def _unjson(value: str | None, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _actor_identity(actor: dict[str, Any]) -> str:
    return canonical_hash(
        {
            "actor_kind": actor.get("actor_kind"),
            "profile": actor.get("profile"),
            "engine": actor.get("engine"),
            "model_id": actor.get("model_id"),
            "model_version": actor.get("model_version"),
            "session_id": actor.get("session_id"),
            "context_id": actor.get("context_id"),
        }
    )


def _validate_actor(actor: dict[str, Any], *, human: bool = False) -> None:
    if not isinstance(actor, dict):
        raise FlowError("actor identity is required")
    required = ("role", "actor_kind", "profile", "engine", "model_id", "model_version", "session_id", "context_id")
    if any(not str(actor.get(key) or "").strip() for key in required):
        raise FlowError("complete role/model/session/context identity is required")
    if human and actor.get("actor_kind") != "human":
        raise HumanControlRequired("human control is required")


def _git_tree_id() -> str:
    supplied = os.environ.get("SHAREDAGENTS_GIT_TREE_ID", "").strip()
    if supplied:
        return supplied
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        value = result.stdout.strip()
        if result.returncode == 0 and value:
            return value
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _trace_baseline() -> dict[str, Any]:
    """Capture the existing function-level trace debt before this run."""
    try:
        repo = Path(__file__).resolve().parents[2]
        scanner_path = repo / "src" / "sharedagents_skill" / "builtin_skills" / "sdlc-trace" / "graph" / "func_trace.py"
        spec = importlib.util.spec_from_file_location("sharedagents_trace_baseline", scanner_path)
        if spec is None or spec.loader is None:
            raise OSError("trace scanner unavailable")
        scanner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scanner)
        result = scanner.scan(str(repo))
        symbols = [f"{row['file']}:{row['line']} {row['symbol']}" for row in result.get("untraced", [])]
        return {
            "significant": len(result.get("significant", [])),
            "traced": len(result.get("traced", [])),
            "derived": len(result.get("derived", [])),
            "untraced_symbols": symbols,
            "git_tree_id": _git_tree_id(),
        }
    except (OSError, ImportError, AttributeError, TypeError):
        return {
            "significant": 0,
            "traced": 0,
            "derived": 0,
            "untraced_symbols": [],
            "git_tree_id": _git_tree_id(),
            "scanner_error": True,
        }


_CREATE = """
CREATE TABLE IF NOT EXISTS dev_flows (
    flow_id TEXT PRIMARY KEY,
    loop_id TEXT NOT NULL,
    repo TEXT NOT NULL,
    branch TEXT NOT NULL,
    issue_number INTEGER,
    issue_url TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL,
    current_stage TEXT NOT NULL DEFAULT 'S0',
    current_phase TEXT NOT NULL DEFAULT 'validation',
    version INTEGER NOT NULL DEFAULT 0,
    frozen_test_hash TEXT NOT NULL DEFAULT '',
    previous_frozen_test_hash TEXT NOT NULL DEFAULT '',
    input_hash TEXT NOT NULL,
    criteria_hash TEXT NOT NULL,
    flowchart_hash TEXT NOT NULL,
    configuration_index_hash TEXT NOT NULL DEFAULT '',
    aal TEXT NOT NULL DEFAULT 'C',
    gate_status TEXT NOT NULL DEFAULT 'VALIDATED',
    validation_json TEXT NOT NULL DEFAULT '{}',
    criteria_json TEXT NOT NULL,
    flowchart_json TEXT NOT NULL,
    requirements TEXT NOT NULL,
    rtm_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    terminal_reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_dev_flows_loop ON dev_flows(loop_id, updated_at);
CREATE TABLE IF NOT EXISTS dev_flow_artifacts (
    artifact_hash TEXT PRIMARY KEY,
    flow_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    version INTEGER NOT NULL,
    content_json TEXT NOT NULL,
    actor_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_flow_roles (
    flow_id TEXT NOT NULL,
    role TEXT NOT NULL,
    actor_kind TEXT NOT NULL,
    profile TEXT NOT NULL,
    engine TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    session_id TEXT NOT NULL,
    context_id TEXT NOT NULL,
    identity_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(flow_id, role)
);
CREATE TABLE IF NOT EXISTS dev_flow_stage_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    actor_json TEXT NOT NULL,
    input_refs_json TEXT NOT NULL DEFAULT '[]',
    output_refs_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    command_manifest_json TEXT NOT NULL DEFAULT '{}',
    result_digest TEXT NOT NULL DEFAULT '',
    test_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_flow_transitions (
    flow_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    prior_event_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    command TEXT NOT NULL,
    actor_json TEXT NOT NULL,
    result TEXT NOT NULL,
    input_refs_json TEXT NOT NULL DEFAULT '[]',
    output_refs_json TEXT NOT NULL DEFAULT '[]',
    verification_json TEXT NOT NULL DEFAULT '{}',
    command_manifest_json TEXT NOT NULL DEFAULT '{}',
    result_digest TEXT NOT NULL DEFAULT '',
    request_digest TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    PRIMARY KEY(flow_id, event_id)
);
CREATE TABLE IF NOT EXISTS dev_flow_problem_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    destination TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dev_flow_disputes (
    dispute_id TEXT PRIMARY KEY,
    flow_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    old_hash TEXT NOT NULL,
    decision TEXT NOT NULL DEFAULT '',
    decision_actor_json TEXT NOT NULL DEFAULT '{}',
    new_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    decided_at TEXT NOT NULL DEFAULT ''
);
"""


class FlowRegistry:
    """SQLite-backed CAS registry for immutable flow runs."""

    def __init__(self, db_path: str | os.PathLike[str]):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = self.path.parent / f"{self.path.stem}_artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_CREATE)
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(dev_flows)")}
            if "current_stage" not in columns:
                conn.execute("ALTER TABLE dev_flows ADD COLUMN current_stage TEXT NOT NULL DEFAULT 'S0'")
            if "current_phase" not in columns:
                conn.execute("ALTER TABLE dev_flows ADD COLUMN current_phase TEXT NOT NULL DEFAULT 'validation'")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _row(self, conn: sqlite3.Connection, flow_id: str) -> dict[str, Any] | None:
        row = conn.execute("SELECT * FROM dev_flows WHERE flow_id=?", (flow_id,)).fetchone()
        return dict(row) if row else None

    def _public(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        for key in ("validation_json", "criteria_json", "flowchart_json", "rtm_json"):
            out[key[:-5] if key.endswith("_json") else key] = _unjson(out.pop(key), {})
        out["allowed_actions"] = self.allowed_actions(out["state"])
        return out

    def get_flow(self, flow_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = self._row(conn, flow_id)
        return self._public(row) if row else None

    def list_flows(self, loop_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if loop_id:
                rows = conn.execute(
                    "SELECT * FROM dev_flows WHERE loop_id=? ORDER BY updated_at DESC", (loop_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM dev_flows ORDER BY updated_at DESC").fetchall()
        return [self._public(dict(row)) for row in rows]

    @staticmethod
    def allowed_actions(state: str) -> list[str]:
        return {
            "S0": ["submit_tests"],
            "S1": ["freeze_tests"],
            "S2": ["submit_spec"],
            "S3": ["submit_implementation"],
            "S4": ["approve", "reject_impl", "reject_spec", "reject_test", "reject_requirement"],
            "DISPUTE": ["UPHOLD_TEST", "SUPERSEDE_TEST", "AMEND_REQUIREMENT"],
        }.get(state, [])

    def _append_event(
        self,
        conn: sqlite3.Connection,
        flow_id: str,
        *,
        from_state: str,
        to_state: str,
        command: str,
        actor: dict[str, Any],
        result: str,
        input_refs: list[str] | None = None,
        output_refs: list[str] | None = None,
        verification: dict[str, Any] | None = None,
        command_manifest: dict[str, Any] | None = None,
        result_digest: str = "",
        request_digest: str = "",
    ) -> dict[str, Any]:
        previous = conn.execute(
            "SELECT event_id,event_hash FROM dev_flow_transitions WHERE flow_id=? ORDER BY event_id DESC LIMIT 1",
            (flow_id,),
        ).fetchone()
        event_id = int(previous["event_id"]) + 1 if previous else 1
        prior_hash = str(previous["event_hash"]) if previous else "0" * 64
        event = {
            "flow_id": flow_id,
            "event_id": event_id,
            "from_state": from_state,
            "to_state": to_state,
            "command": command,
            "actor": actor,
            "result": result,
            "input_refs": input_refs or [],
            "output_refs": output_refs or [],
            "verification": verification or {},
            "command_manifest": command_manifest or {},
            "result_digest": result_digest,
            "request_digest": request_digest,
            "created_at": _now(),
        }
        event_hash = append_event_hash(event, prior_hash)
        conn.execute(
            """INSERT INTO dev_flow_transitions
               (flow_id,event_id,prior_event_hash,event_hash,from_state,to_state,command,actor_json,result,
                input_refs_json,output_refs_json,verification_json,command_manifest_json,result_digest,
                request_digest,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                flow_id,
                event_id,
                prior_hash,
                event_hash,
                from_state,
                to_state,
                command,
                _json(actor),
                result,
                _json(input_refs or []),
                _json(output_refs or []),
                _json(verification or {}),
                _json(command_manifest or {}),
                result_digest,
                request_digest,
                event["created_at"],
            ),
        )
        return {**event, "prior_event_hash": prior_hash, "event_hash": event_hash}

    def _reject(
        self,
        conn: sqlite3.Connection,
        row: dict[str, Any],
        actor: dict[str, Any],
        command: str,
        reason: str,
        expected_version: Any,
    ) -> None:
        request_digest = canonical_hash({"command": command, "expected_version": expected_version, "reason": reason})
        self._append_event(
            conn,
            row["flow_id"],
            from_state=row["state"],
            to_state=row["state"],
            command=command,
            actor=actor if isinstance(actor, dict) else {"actor": str(actor)},
            result="REJECTED_" + reason.upper(),
            verification={"reason": reason, "expected_version": expected_version, "actual_version": row["version"]},
            request_digest=request_digest,
        )

    def _begin(
        self, flow_id: str, expected_version: int, actor: dict[str, Any], command: str
    ) -> tuple[sqlite3.Connection, dict[str, Any]]:
        _validate_actor(actor)
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        row = self._row(conn, flow_id)
        if not row:
            conn.rollback()
            conn.close()
            raise FlowError("unknown flow")
        if int(row["version"]) != int(expected_version):
            self._reject(conn, row, actor, command, "CAS", expected_version)
            conn.commit()
            conn.close()
            raise FlowConflict("wrong expected_version")
        if row["state"] in _TERMINAL:
            self._reject(conn, row, actor, command, "TERMINAL", expected_version)
            conn.commit()
            conn.close()
            raise FlowConflict("terminal flow")
        return conn, row

    def _finish(self, conn: sqlite3.Connection, flow_id: str, to_state: str) -> dict[str, Any]:
        row = self._row(conn, flow_id)
        assert row is not None
        now = _now()
        conn.execute(
            "UPDATE dev_flows SET state=?, current_stage=?, current_phase=?, version=version+1, updated_at=?, gate_status=? WHERE flow_id=?",
            (
                to_state,
                to_state,
                "verification" if to_state in _TERMINAL else "validation",
                now,
                "TERMINAL" if to_state in _TERMINAL else "READY",
                flow_id,
            ),
        )
        updated = self._row(conn, flow_id)
        assert updated is not None
        return updated

    def _store_artifact(
        self, conn: sqlite3.Connection, flow_id: str, kind: str, content: Any, actor: dict[str, Any], version: int
    ) -> str:
        digest = canonical_hash(content)
        payload = _json(content)
        existing = conn.execute(
            "SELECT content_json FROM dev_flow_artifacts WHERE artifact_hash=?", (digest,)
        ).fetchone()
        if existing and existing["content_json"] != payload:
            raise FlowConflict("content-addressed artifact collision")
        if not existing:
            conn.execute(
                "INSERT INTO dev_flow_artifacts (artifact_hash,flow_id,kind,version,content_json,actor_json,created_at) VALUES (?,?,?,?,?,?,?)",
                (digest, flow_id, kind, version, payload, _json(actor), _now()),
            )
            target = self.artifact_root / f"{digest}.json"
            if target.exists() and target.read_bytes() != payload.encode("utf-8"):
                raise FlowConflict("artifact file collision")
            if not target.exists():
                target.write_bytes(payload.encode("utf-8"))
        return digest

    def _stage_result(
        self,
        conn: sqlite3.Connection,
        flow_id: str,
        stage: str,
        phase: str,
        status: str,
        actor: dict[str, Any],
        evidence: Any,
        manifest: Any,
        inputs: list[str],
        outputs: list[str],
        test_hash: str = "",
    ) -> None:
        result_digest = canonical_hash(
            {"stage": stage, "phase": phase, "status": status, "evidence": evidence, "manifest": manifest}
        )
        conn.execute(
            """INSERT INTO dev_flow_stage_results
            (flow_id,stage,phase,status,actor_json,input_refs_json,output_refs_json,evidence_json,command_manifest_json,result_digest,test_hash,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                flow_id,
                stage,
                phase,
                status,
                _json(actor),
                _json(inputs),
                _json(outputs),
                _json(evidence),
                _json(manifest),
                result_digest,
                test_hash,
                _now(),
            ),
        )

    def _refresh_configuration_index(
        self, conn: sqlite3.Connection, flow_id: str, actor: dict[str, Any], version: int
    ) -> str:
        rows = conn.execute(
            "SELECT artifact_hash FROM dev_flow_artifacts WHERE flow_id=? AND kind!='configuration_index' ORDER BY version,created_at",
            (flow_id,),
        ).fetchall()
        content = {
            "schema": "sharedagents-sdlc-config/v1",
            "flow_id": flow_id,
            "artifact_hashes": [str(row["artifact_hash"]) for row in rows],
            "git_tree_ids": [_git_tree_id()],
            "updated_at": _now(),
        }
        digest = self._store_artifact(conn, flow_id, "configuration_index", content, actor, version)
        conn.execute("UPDATE dev_flows SET configuration_index_hash=? WHERE flow_id=?", (digest, flow_id))
        return digest

    def _fail_stage(
        self,
        conn: sqlite3.Connection,
        row: dict[str, Any],
        stage: str,
        actor: dict[str, Any],
        reason: str,
        evidence: Any,
        manifest: Any,
    ) -> None:
        """Persist a failed validation and take the ratified FAILED edge."""
        self._stage_result(conn, row["flow_id"], stage, "validation", "FAIL", actor, evidence, manifest, [], [])
        self._append_event(
            conn,
            row["flow_id"],
            from_state=row["state"],
            to_state="FAILED",
            command="validation_failed",
            actor=actor,
            result="FAILED",
            verification={"reason": reason, "evidence": evidence},
            command_manifest=manifest,
            result_digest=canonical_hash({"reason": reason, "evidence": evidence}),
        )
        self._finish(conn, row["flow_id"], "FAILED")

    def create_flow(
        self,
        *,
        loop_id: str,
        repo: str,
        branch: str,
        issue_number: int | None,
        issue_url: str,
        requirements: str,
        flowchart: dict[str, Any],
        criteria: list[dict[str, Any]],
        actor: dict[str, Any],
        aal: str = "C",
        paac: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _validate_actor(actor, human=True)
        validation = validate_flowchart(flowchart)
        if not str(requirements or "").strip() or not isinstance(criteria, list) or not criteria:
            raise FlowError("requirements and criteria are required")
        ids = [str(item.get("id") or "") for item in criteria if isinstance(item, dict)]
        if len(ids) != len(criteria) or len(set(ids)) != len(ids) or any(item not in CRITERION_IDS for item in ids):
            raise FlowError("criteria IDs must be stable ratified IDs")
        if set(ids) != set(CRITERION_IDS):
            raise FlowError("all ratified criteria are required")
        flow_id = "flow_" + uuid.uuid4().hex
        requirements = str(requirements)
        criteria_hash = canonical_hash(criteria)
        flowchart_hash = canonical_hash(flowchart)
        input_hash = canonical_hash({"requirements": requirements, "flowchart": flowchart, "criteria": criteria})
        paac_value = paac or {"aal": aal or "C", "blast_radius": "default", "coverage_floor": 0.8, "reviewer_quorum": 1}
        now = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM dev_flows WHERE flow_id=?", (flow_id,)).fetchone():
                conn.rollback()
                raise FlowConflict("flow id collision")
            conn.execute(
                """INSERT INTO dev_flows
                (flow_id,loop_id,repo,branch,issue_number,issue_url,state,current_stage,current_phase,version,input_hash,criteria_hash,flowchart_hash,aal,gate_status,validation_json,criteria_json,flowchart_json,requirements,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    flow_id,
                    loop_id,
                    repo,
                    branch,
                    issue_number,
                    issue_url,
                    "S0",
                    "S0",
                    "validation",
                    0,
                    input_hash,
                    criteria_hash,
                    flowchart_hash,
                    str(paac_value.get("aal") or aal or "C"),
                    "VALIDATED",
                    _json({"status": "PASS", **validation}),
                    _json(criteria),
                    _json(flowchart),
                    requirements,
                    now,
                    now,
                ),
            )
            input_ref = self._store_artifact(
                conn,
                flow_id,
                "ratified_stage0_input",
                {"requirements": requirements, "flowchart": flowchart, "criteria": criteria},
                actor,
                0,
            )
            paac_ref = self._store_artifact(conn, flow_id, "paac_lite", paac_value, actor, 0)
            baseline_ref = self._store_artifact(
                conn, flow_id, "rtm_baseline", {**_trace_baseline(), "captured_at": now}, actor, 0
            )
            index = {
                "schema": "sharedagents-sdlc-config/v1",
                "flow_id": flow_id,
                "artifact_hashes": [input_ref, paac_ref, baseline_ref],
                "git_tree_ids": [_git_tree_id()],
                "created_at": now,
            }
            index_ref = self._store_artifact(conn, flow_id, "configuration_index", index, actor, 0)
            conn.execute("UPDATE dev_flows SET configuration_index_hash=? WHERE flow_id=?", (index_ref, flow_id))
            event = self._append_event(
                conn,
                flow_id,
                from_state="NONE",
                to_state="S0",
                command="START",
                actor=actor,
                result="ACCEPTED",
                output_refs=[input_ref, paac_ref, baseline_ref, index_ref],
                verification=validation,
                request_digest=input_hash,
            )
            conn.commit()
        return self.get_flow(flow_id) or {"flow_id": flow_id, "version": event["event_id"], "state": "S0"}

    def _validate_cases(self, flow: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
        if not cases:
            raise FlowError("at least one declarative test is required")
        rtm = extract_rtm(cases, _unjson(flow["criteria_json"], []))
        if rtm["untraced_tests"] or rtm["untested_criteria"]:
            raise FlowConflict("bidirectional RTM failed")
        for case in cases:
            if not str(case.get("behavior") or "").strip() or not str(case.get("expected_outcome") or "").strip():
                raise FlowError("declarative case needs behavior and expected outcome")
            if case.get("kind") not in {"normal", "robustness"}:
                raise FlowError("test kind must be normal or robustness")
        return rtm

    def submit_tests(
        self,
        flow_id: str,
        *,
        expected_version: int,
        actor: dict[str, Any],
        cases: list[dict[str, Any]],
        manifest: dict[str, Any],
        run: dict[str, Any],
        negative_witnesses: dict[str, Any] | None = None,
        baseline_decisions: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        conn, row = self._begin(flow_id, expected_version, actor, "submit_tests")
        try:
            if row["state"] != "S0":
                self._reject(conn, row, actor, "submit_tests", "STATE", expected_version)
                conn.commit()
                raise FlowConflict("tests can only start at S0")
            if actor.get("role") not in {"test_author", "tests", "ai"} or actor.get("actor_kind") != "ai":
                raise FlowConflict("test generation must use an AI test-author role")
            try:
                rtm = self._validate_cases(row, cases)
            except (FlowError, FlowConflict) as error:
                self._fail_stage(conn, row, "S1", actor, str(error), {"cases": cases}, manifest)
                conn.commit()
                raise
            status = str(run.get("status") or "").upper()
            baseline = baseline_decisions or {}
            if not (
                (status == "RED" and int(run.get("failures") or 0) > 0 and int(run.get("errors") or 0) == 0) or baseline
            ):
                self._stage_result(conn, flow_id, "S1", "validation", "FAIL", actor, run, manifest, [], [])
                self._append_event(
                    conn,
                    flow_id,
                    from_state="S0",
                    to_state="FAILED",
                    command="validation_failed",
                    actor=actor,
                    result="FAILED",
                    verification=run,
                )
                self._finish(conn, flow_id, "FAILED")
                conn.commit()
                raise FlowConflict("stage-1 suite must be RED or baseline-decided")
            witnesses = negative_witnesses or {}
            criterion_ids = [
                str(item.get("id") if isinstance(item, dict) else item) for item in _unjson(row["criteria_json"], [])
            ]
            if not baseline and any(not witnesses.get(cid) for cid in criterion_ids):
                error = FlowConflict("negative witness missing")
                self._fail_stage(conn, row, "S1", actor, str(error), {"cases": cases}, manifest)
                conn.commit()
                raise error
            content = {
                "cases": cases,
                "manifest": manifest,
                "run": run,
                "negative_witnesses": witnesses,
                "baseline_decisions": baseline,
                "rtm": rtm,
            }
            artifact = self._store_artifact(conn, flow_id, "declarative_tests", content, actor, int(row["version"]) + 1)
            self._ensure_distinct_role(conn, row, actor, "test_author")
            self._stage_result(
                conn, flow_id, "S1", "implementation", "PASS", actor, content, manifest, [row["input_hash"]], [artifact]
            )
            self._append_event(
                conn,
                flow_id,
                from_state="S0",
                to_state="S1",
                command="submit_tests",
                actor=actor,
                result="ACCEPTED",
                output_refs=[artifact],
                verification=rtm,
                command_manifest=manifest,
                result_digest=canonical_hash(content),
                request_digest=canonical_hash(content),
            )
            current = self._finish(conn, flow_id, "S1")
            self._stage_result(
                conn, flow_id, "S1", "verification", "PASS", actor, run, manifest, [artifact], [artifact]
            )
            self._append_event(
                conn,
                flow_id,
                from_state="S1",
                to_state="S2",
                command="freeze_tests",
                actor=actor,
                result="FROZEN",
                output_refs=[artifact],
                verification={"rtm": rtm, "run": run},
                command_manifest=manifest,
                result_digest=canonical_hash(content),
                request_digest=canonical_hash(content),
            )
            conn.execute(
                "UPDATE dev_flows SET state='S2', current_stage='S2', current_phase='verification', frozen_test_hash=?, rtm_json=?, version=version+1, updated_at=?, gate_status='READY' WHERE flow_id=?",
                (artifact, _json(rtm), _now(), flow_id),
            )
            self._refresh_configuration_index(conn, flow_id, actor, int(row["version"]) + 2)
            conn.commit()
            return self.get_flow(flow_id) or self._public(current)
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            conn.close()
            raise

    def _same_hash(self, row: dict[str, Any], supplied: str | None) -> bool:
        return bool(supplied) and str(supplied) == str(row.get("frozen_test_hash") or "")

    def _frozen_manifest(self, conn: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
        artifact = conn.execute(
            "SELECT content_json FROM dev_flow_artifacts WHERE artifact_hash=? AND flow_id=?",
            (row.get("frozen_test_hash") or "", row["flow_id"]),
        ).fetchone()
        content = _unjson(artifact["content_json"], {}) if artifact else {}
        return content.get("manifest") if isinstance(content, dict) else {}

    def submit_spec(
        self,
        flow_id: str,
        *,
        expected_version: int,
        actor: dict[str, Any],
        spec: dict[str, Any],
        adapter_binding: dict[str, Any],
        consistency: dict[str, Any],
        test_hash: str | None = None,
    ) -> dict[str, Any]:
        conn, row = self._begin(flow_id, expected_version, actor, "submit_spec")
        try:
            if row["state"] != "S2":
                self._reject(conn, row, actor, "submit_spec", "STATE", expected_version)
                conn.commit()
                raise FlowConflict("spec requires S2")
            if test_hash is not None and not self._same_hash(row, test_hash):
                self._reject(conn, row, actor, "submit_spec", "HASH", expected_version)
                conn.commit()
                raise FlowConflict("frozen test hash mismatch")
            files = spec.get("files") or spec.get("modules") or []
            endpoints = spec.get("endpoints") or spec.get("tables") or []
            if not files or not endpoints or str(consistency.get("status") or "").upper() != "PASS":
                error = FlowConflict("concrete spec and consistency evidence required")
                self._fail_stage(conn, row, "S2", actor, str(error), spec, consistency)
                conn.commit()
                raise error
            spec_ref = self._store_artifact(conn, flow_id, "technical_spec", spec, actor, int(row["version"]) + 1)
            binding_ref = self._store_artifact(
                conn,
                flow_id,
                "adapter_binding",
                {"binding": adapter_binding, "consistency": consistency, "frozen_test_hash": row["frozen_test_hash"]},
                actor,
                int(row["version"]) + 1,
            )
            self._stage_result(
                conn,
                flow_id,
                "S2",
                "implementation",
                "PASS",
                actor,
                spec,
                consistency,
                [row["frozen_test_hash"]],
                [spec_ref, binding_ref],
                row["frozen_test_hash"],
            )
            self._append_event(
                conn,
                flow_id,
                from_state="S2",
                to_state="S3",
                command="submit_spec",
                actor=actor,
                result="ACCEPTED",
                output_refs=[spec_ref, binding_ref],
                verification=consistency,
                result_digest=canonical_hash({"spec": spec, "binding": adapter_binding}),
            )
            self._finish(conn, flow_id, "S3")
            self._refresh_configuration_index(conn, flow_id, actor, int(row["version"]) + 1)
            conn.commit()
            return self.get_flow(flow_id) or {}
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            conn.close()
            raise

    def submit_implementation(
        self,
        flow_id: str,
        *,
        expected_version: int,
        actor: dict[str, Any],
        code: dict[str, Any],
        runs: dict[str, Any],
        test_hash: str | None = None,
        rtm: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        conn, row = self._begin(flow_id, expected_version, actor, "submit_implementation")
        try:
            if row["state"] != "S3":
                self._reject(conn, row, actor, "submit_implementation", "STATE", expected_version)
                conn.commit()
                raise FlowConflict("implementation requires S3")
            if not self._same_hash(row, test_hash):
                dispute_id = "dispute_" + uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO dev_flow_disputes (dispute_id,flow_id,reason,old_hash,created_at) VALUES (?,?,?,?,?)",
                    (dispute_id, flow_id, "frozen test hash changed", row["frozen_test_hash"], _now()),
                )
                self._append_event(
                    conn,
                    flow_id,
                    from_state="S3",
                    to_state="DISPUTE",
                    command="test_hash_changed",
                    actor=actor,
                    result="TEST_DISPUTE",
                    verification={"expected": row["frozen_test_hash"], "actual": test_hash or ""},
                )
                self._finish(conn, flow_id, "DISPUTE")
                conn.commit()
                raise FlowConflict("frozen test hash changed; TEST_DISPUTE")
            files = code.get("files") or []
            if (
                not files
                or str(runs.get("compile") or "").upper() != "PASS"
                or str(runs.get("acceptance") or "").upper() != "PASS"
            ):
                error = FlowConflict("implementation compile and acceptance evidence must be green")
                self._fail_stage(conn, row, "S3", actor, str(error), code, runs)
                conn.commit()
                raise error
            gaps = rtm or {}
            if gaps.get("new_gaps") or gaps.get("worsened_gaps"):
                error = FlowConflict("new or worsened RTM gaps block implementation")
                self._fail_stage(conn, row, "S3", actor, str(error), gaps, runs)
                conn.commit()
                raise error
            code_ref = self._store_artifact(
                conn,
                flow_id,
                "implementation",
                {"code": code, "runs": runs, "rtm": gaps, "frozen_test_hash": row["frozen_test_hash"]},
                actor,
                int(row["version"]) + 1,
            )
            self._ensure_distinct_role(conn, row, actor, "implementer")
            self._stage_result(
                conn,
                flow_id,
                "S3",
                "implementation",
                "PASS",
                actor,
                code,
                runs,
                [row["frozen_test_hash"]],
                [code_ref],
                row["frozen_test_hash"],
            )
            self._append_event(
                conn,
                flow_id,
                from_state="S3",
                to_state="S4",
                command="submit_implementation",
                actor=actor,
                result="ACCEPTED",
                output_refs=[code_ref],
                verification=runs,
                command_manifest=runs,
                result_digest=canonical_hash({"code": code, "runs": runs}),
            )
            self._finish(conn, flow_id, "S4")
            self._refresh_configuration_index(conn, flow_id, actor, int(row["version"]) + 1)
            conn.commit()
            return self.get_flow(flow_id) or {}
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            conn.close()
            raise

    def _ensure_distinct_role(
        self, conn: sqlite3.Connection, row: dict[str, Any], actor: dict[str, Any], role: str
    ) -> None:
        identity = _actor_identity(actor)
        prior = conn.execute(
            "SELECT role,identity_hash FROM dev_flow_roles WHERE flow_id=?", (row["flow_id"],)
        ).fetchall()
        if any(str(item["identity_hash"]) == identity for item in prior if item["role"] != role):
            raise FlowConflict("role identities must be distinct")
        conn.execute(
            """INSERT INTO dev_flow_roles (flow_id,role,actor_kind,profile,engine,model_id,model_version,session_id,context_id,identity_hash,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(flow_id,role) DO UPDATE SET actor_kind=excluded.actor_kind,profile=excluded.profile,engine=excluded.engine,model_id=excluded.model_id,model_version=excluded.model_version,session_id=excluded.session_id,context_id=excluded.context_id,identity_hash=excluded.identity_hash""",
            (
                row["flow_id"],
                role,
                actor["actor_kind"],
                actor["profile"],
                actor["engine"],
                actor["model_id"],
                actor["model_version"],
                actor["session_id"],
                actor["context_id"],
                identity,
                _now(),
            ),
        )

    def submit_review(
        self,
        flow_id: str,
        *,
        expected_version: int,
        actor: dict[str, Any],
        verdict: str,
        checks: dict[str, Any],
        runs: dict[str, Any],
        rejection_type: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        conn, row = self._begin(flow_id, expected_version, actor, "review")
        try:
            if row["state"] != "S4" or actor.get("actor_kind") != "reviewer":
                self._reject(conn, row, actor, "review", "STATE", expected_version)
                conn.commit()
                raise FlowConflict("review requires separate reviewer at S4")
            self._ensure_distinct_role(conn, row, actor, "reviewer")
            verdict = str(verdict or "").upper()
            if verdict == "APPROVE":
                required = ("semantic", "anti_tautology", "manifest", "rtm")
                if (
                    any(not checks.get(key) for key in required)
                    or str(runs.get("acceptance") or "").upper() != "PASS"
                    or str(runs.get("regression") or "").upper() != "PASS"
                ):
                    raise FlowConflict("review approval requires all checks and manifest green")
                frozen_manifest = self._frozen_manifest(conn, row)
                submitted_manifest = runs.get("manifest")
                if not isinstance(submitted_manifest, dict) or canonical_hash(submitted_manifest) != canonical_hash(
                    frozen_manifest
                ):
                    raise FlowConflict("review must execute the exact frozen manifest")
                review_ref = self._store_artifact(
                    conn,
                    flow_id,
                    "review",
                    {"verdict": verdict, "checks": checks, "runs": runs},
                    actor,
                    int(row["version"]) + 1,
                )
                summary_ref = self._store_artifact(
                    conn,
                    flow_id,
                    "accomplishment_summary",
                    {"flow_id": flow_id, "verdict": verdict, "checks": checks, "runs": runs},
                    actor,
                    int(row["version"]) + 1,
                )
                self._stage_result(
                    conn,
                    flow_id,
                    "S4",
                    "verification",
                    "PASS",
                    actor,
                    checks,
                    runs,
                    [row["frozen_test_hash"]],
                    [review_ref, summary_ref],
                    row["frozen_test_hash"],
                )
                self._append_event(
                    conn,
                    flow_id,
                    from_state="S4",
                    to_state="DONE",
                    command="approve",
                    actor=actor,
                    result="APPROVE",
                    output_refs=[review_ref, summary_ref],
                    verification={"checks": checks, "runs": runs},
                    command_manifest=runs,
                    result_digest=canonical_hash({"review": review_ref, "summary": summary_ref}),
                )
                self._finish(conn, flow_id, "DONE")
                self._refresh_configuration_index(conn, flow_id, actor, int(row["version"]) + 1)
                conn.commit()
                return self.get_flow(flow_id) or {}
            if (
                verdict != "REJECT"
                or rejection_type not in {"impl", "spec", "test", "requirement"}
                or not reason.strip()
            ):
                raise FlowError("typed rejection with reason is required")
            destination = route_rework(rejection_type)
            conn.execute(
                "INSERT INTO dev_flow_problem_reports (flow_id,kind,destination,reason,evidence_json,created_at) VALUES (?,?,?,?,?,?)",
                (flow_id, rejection_type, destination, reason, _json({"checks": checks, "runs": runs}), _now()),
            )
            review_ref = self._store_artifact(
                conn,
                flow_id,
                "review",
                {"verdict": verdict, "type": rejection_type, "reason": reason, "checks": checks, "runs": runs},
                actor,
                int(row["version"]) + 1,
            )
            if destination == "DISPUTE":
                dispute_id = "dispute_" + uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO dev_flow_disputes (dispute_id,flow_id,reason,old_hash,created_at) VALUES (?,?,?,?,?)",
                    (dispute_id, flow_id, reason, row["frozen_test_hash"], _now()),
                )
            self._append_event(
                conn,
                flow_id,
                from_state="S4",
                to_state=destination,
                command="reject_" + rejection_type,
                actor=actor,
                result="REJECT",
                output_refs=[review_ref],
                verification={"type": rejection_type, "reason": reason},
                result_digest=canonical_hash(reason),
            )
            self._finish(conn, flow_id, destination)
            conn.commit()
            return self.get_flow(flow_id) or {}
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            conn.close()
            raise

    def open_dispute(
        self, flow_id: str, *, expected_version: int, actor: dict[str, Any], reason: str
    ) -> dict[str, Any]:
        conn, row = self._begin(flow_id, expected_version, actor, "reject_test")
        try:
            if row["state"] != "S4" or not str(reason).strip():
                self._reject(conn, row, actor, "reject_test", "STATE", expected_version)
                conn.commit()
                raise FlowConflict("test dispute requires S4 and a reason")
            dispute_id = "dispute_" + uuid.uuid4().hex
            conn.execute(
                "INSERT INTO dev_flow_disputes (dispute_id,flow_id,reason,old_hash,created_at) VALUES (?,?,?,?,?)",
                (dispute_id, flow_id, reason, row["frozen_test_hash"], _now()),
            )
            self._append_event(
                conn,
                flow_id,
                from_state="S4",
                to_state="DISPUTE",
                command="reject_test",
                actor=actor,
                result="TEST_DISPUTE",
                verification={"reason": reason},
            )
            self._finish(conn, flow_id, "DISPUTE")
            conn.commit()
            return self.get_flow(flow_id) or {}
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            conn.close()
            raise

    def resolve_dispute(
        self,
        flow_id: str,
        *,
        expected_version: int,
        disposition: str,
        actor: dict[str, Any],
        new_cases: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        _validate_actor(actor, human=True)
        conn, row = self._begin(flow_id, expected_version, actor, disposition)
        try:
            if row["state"] != "DISPUTE":
                self._reject(conn, row, actor, disposition, "STATE", expected_version)
                conn.commit()
                raise FlowConflict("flow is not in TEST_DISPUTE")
            dispute = conn.execute(
                "SELECT * FROM dev_flow_disputes WHERE flow_id=? AND decision='' ORDER BY created_at DESC LIMIT 1",
                (flow_id,),
            ).fetchone()
            if not dispute:
                raise FlowConflict("pending dispute not found")
            destination = {"UPHOLD_TEST": "S3", "SUPERSEDE_TEST": "S1", "AMEND_REQUIREMENT": "NEW_RUN"}.get(disposition)
            if not destination:
                raise FlowError("unknown dispute disposition")
            new_hash = ""
            if disposition == "SUPERSEDE_TEST":
                cases = new_cases or []
                self._validate_cases(row, cases)
                content = {
                    "cases": cases,
                    "supersedes": row["frozen_test_hash"],
                    "rtm": extract_rtm(cases, _unjson(row["criteria_json"], [])),
                }
                new_hash = self._store_artifact(
                    conn, flow_id, "declarative_tests_superseding", content, actor, int(row["version"]) + 1
                )
                if new_hash == row["frozen_test_hash"]:
                    raise FlowConflict("superseding baseline must differ")
                conn.execute(
                    "UPDATE dev_flows SET previous_frozen_test_hash=?, frozen_test_hash=? WHERE flow_id=?",
                    (row["frozen_test_hash"], new_hash, flow_id),
                )
            conn.execute(
                "UPDATE dev_flow_disputes SET decision=?,decision_actor_json=?,new_hash=?,decided_at=? WHERE dispute_id=?",
                (disposition, _json(actor), new_hash, _now(), dispute["dispute_id"]),
            )
            self._append_event(
                conn,
                flow_id,
                from_state="DISPUTE",
                to_state=destination,
                command=disposition,
                actor=actor,
                result="HUMAN_DECISION",
                output_refs=[new_hash] if new_hash else [],
                verification={"disposition": disposition, "old_hash": row["frozen_test_hash"], "new_hash": new_hash},
            )
            self._finish(conn, flow_id, destination)
            self._refresh_configuration_index(conn, flow_id, actor, int(row["version"]) + 1)
            conn.commit()
            return self.get_flow(flow_id) or {}
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            conn.close()
            raise

    def advance(
        self,
        flow_id: str,
        *,
        expected_version: int,
        action: str,
        actor: dict[str, Any],
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generic command endpoint; only explicitly implemented actions pass."""
        conn, row = self._begin(flow_id, expected_version, actor, action)
        try:
            self._reject(conn, row, actor, action, "VALIDATION", expected_version)
            conn.commit()
            raise FlowConflict("advance requires a validated stage command")
        finally:
            if conn.isolation_level is not None and conn.in_transaction:
                conn.rollback()
            conn.close()

    def events(self, flow_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dev_flow_transitions WHERE flow_id=? ORDER BY event_id", (flow_id,)
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            for key in (
                "actor_json",
                "input_refs_json",
                "output_refs_json",
                "verification_json",
                "command_manifest_json",
            ):
                item[key[:-5]] = _unjson(item.pop(key), {})
            out.append(item)
        return out

    def verify_event_chain(self, flow_id: str) -> bool:
        previous = "0" * 64
        for event in self.events(flow_id):
            expected = append_event_hash(
                {
                    "flow_id": event["flow_id"],
                    "event_id": event["event_id"],
                    "from_state": event["from_state"],
                    "to_state": event["to_state"],
                    "command": event["command"],
                    "actor": event["actor"],
                    "result": event["result"],
                    "input_refs": event["input_refs"],
                    "output_refs": event["output_refs"],
                    "verification": event["verification"],
                    "command_manifest": event["command_manifest"],
                    "result_digest": event["result_digest"],
                    "request_digest": event["request_digest"],
                    "created_at": event["created_at"],
                },
                previous,
            )
            if event["prior_event_hash"] != previous or event["event_hash"] != expected:
                return False
            previous = event["event_hash"]
        return True

    def roles(self, flow_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM dev_flow_roles WHERE flow_id=? ORDER BY role", (flow_id,)).fetchall()
        return [dict(row) for row in rows]

    def artifacts(self, flow_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dev_flow_artifacts WHERE flow_id=? ORDER BY version,created_at", (flow_id,)
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["content"] = _unjson(item.pop("content_json"), {})
            item["actor"] = _unjson(item.pop("actor_json"), {})
            out.append(item)
        return out

    def stage_results(self, flow_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dev_flow_stage_results WHERE flow_id=? ORDER BY id", (flow_id,)
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            for key in ("actor_json", "input_refs_json", "output_refs_json", "evidence_json", "command_manifest_json"):
                item[key[:-5]] = _unjson(item.pop(key), {})
            out.append(item)
        return out

    def disputes(self, flow_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM dev_flow_disputes WHERE flow_id=? ORDER BY created_at", (flow_id,)
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["decision_actor"] = _unjson(item.pop("decision_actor_json"), {})
            out.append(item)
        return out

    def configuration_index(self, flow_id: str) -> dict[str, Any]:
        flow = self.get_flow(flow_id)
        if not flow:
            raise FlowError("unknown flow")
        artifacts = self.artifacts(flow_id)
        return {
            "flow_id": flow_id,
            "artifact_hashes": [item["artifact_hash"] for item in artifacts],
            "git_tree_ids": [_git_tree_id()],
            "configuration_index_hash": flow.get("configuration_index_hash", ""),
        }

    def replace_artifact(self, artifact_hash: str, content: Any) -> None:
        raise FlowConflict("artifacts are immutable")

    def delete_event(self, flow_id: str, event_id: int) -> None:
        raise FlowConflict("audit events are append-only")


def default_flow_db() -> Path:
    configured = os.environ.get("SHAREDAGENTS_DEV_FLOW_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    root = os.environ.get("SHAREDAGENTS_HOME", "").strip()
    if root:
        return Path(root).expanduser() / "web" / "dev_flows.db"
    return Path.home() / ".sharedagents" / "web" / "dev_flows.db"


_registry: FlowRegistry | None = None


def get_flow_registry() -> FlowRegistry:
    global _registry
    path = default_flow_db()
    if _registry is None or _registry.path != path:
        _registry = FlowRegistry(path)
    return _registry


def reset_flow_registry() -> None:
    global _registry
    _registry = None


__all__ = [
    "CRITERION_IDS",
    "FLOWCHART",
    "RATIFIED_STAGE0_HASHES",
    "TRANSITION_TABLE",
    "EXHAUSTIVE_TRANSITION_TABLE",
    "STATE_TRANSITIONS",
    "FlowConflict",
    "FlowError",
    "FlowRegistry",
    "HumanControlRequired",
    "append_event_hash",
    "canonical_hash",
    "extract_rtm",
    "get_flow_registry",
    "reset_flow_registry",
    "route_rework",
    "validate_flowchart",
]
