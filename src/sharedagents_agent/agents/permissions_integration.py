"""Permission gating + ASK protocol integration.

The permission policy lives in two places:

* ``sharedagents_agent.tools.permissions.PermissionEvaluator`` — pure logic:
  given ``(tool_name, args, criticality, mode)``, returns
  ``"allow"`` / ``"ask"`` / ``"deny"``.
* ``permissions_integration.ask_user`` (this module) +
  ``sharedagents_gateway.acp_bridge._ask_user_outbound`` — the ACP wire
  protocol that surfaces an ``ask`` decision to the user and waits
  for an answer.

These are Agent-level (orthogonal to engine — apply whether the
engine is local-LLM or a CLI). The Agent class and the gateway
shim agree on this surface.

The permission sentinel is NOT defined
here — it lives in :mod:`.engine` as the :class:`PermissionRequested`
event variant so it can flow through the standard
:data:`AgentTurnEvent` stream the engine already yields. This module
owns only the policy table + the ACP wire round-trip.

================================================================
Permission policy (unchanged from today)
================================================================

The matrix of (criticality × mode → decision) lives in
``permissions.py`` and DOES NOT change in the refactor:

============== =========== =========== ===========
criticality \\ mode   plan       ask         auto
============== =========== =========== ===========
read_only            allow      allow       allow
mutating             ask        ask         allow
destructive          deny       ask         allow*
network              ask        ask         allow

* plan-mode ``network`` stays in the ask path. Read-only web research reaches
  plan mode via the read_only row instead: ``mcp__web__*`` declares
  ``readOnlyHint`` so discovery resolves it to read_only. Other
  NETWORK-kind tools (Agent/acp_call, which execute) stay in the ask path.
** Auto never uses the permission widget. Production-impact briefing and
  a single batched ``question`` (impact + rollback for the whole set) are
  the agent's job, not per-tool ``request_permission`` cards.
============== =========== =========== ===========

(Today's table — verify against ``permissions.PermissionEvaluator`` at
implementation time. Same evaluator used for local-LLM tool calls
AND for CLI engines' OWN tool decisions — the CLI runs in its own
permission space (``--permission-mode <x>``) but the sharedagents side
still gates which delegate kinds the model is allowed to invoke.)

================================================================
ASK protocol
================================================================

ACP defines ``request_permission`` (server → client) with the
client responding via ``permission_response``. :func:`ask_user`
below issues the ``request_outbound("session/request_permission",
…)`` call directly through the bound :class:`AcpServer`, with a
auto-answer fast path (``_ask_user_auto_answers``) for auto-mode runs
where there's no user to ask. Interactive permission gates otherwise
remain parked until the operator answers or explicitly cancels the turn;
elapsed wall-clock time is not a permission decision.

After refactor the Agent.run_turn flow is:

1. Engine encounters a tool call (or, for CLI engines, a delegate
   tool the model invoked through the CLI's own tools — rare, since
   CLI engines mostly run their own tools).
2. Engine yields :class:`PermissionRequested` (an
   :data:`AgentTurnEvent` variant) instead of dispatching directly.
   (Local LLM does this inside ``_dispatch_tool_calls``; CLI engines
   never do.)
3. :class:`Agent.run_turn` catches the sentinel, runs the evaluator,
   decides ``allow`` / ``deny`` immediately OR emits an ACP
   ``request_permission`` and awaits the response via a `Future`.
4. Engine resumes with the decision.

The new sentinel event:
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Delegation primitive(s) carved out of plan-mode's mutating/destructive deny
# (build & plan MAY delegate). Only the delegation call itself is admitted; the
# delegated turn is gated in the child (ask/auto, never plan).
_DELEGATION_TOOLS = frozenset({"AgentSync", "AgentAsync"})


def _plan_path_of(arguments: dict[str, Any] | None) -> str:
    a = arguments or {}
    return str(a.get("path") or a.get("filePath") or a.get("file_path") or a.get("target_file") or "")


# ---------------------------------------------------------------------
# Shell command-prefix grants ("Allow always" on a bash call).
# ---------------------------------------------------------------------
# A shell call cannot be remembered by signature (every command differs),
# so its grant is the command PREFIX — ``git worktree`` covers
# ``git worktree list`` and ``git worktree add …`` — mirroring OpenCode's
# BashArity prefix semantics. A grant is only ever created by an explicit
# operator click on a read_only/mutating call; destructive and forbidden
# calls are never rememberable.
_BASH_TOOLS = frozenset({"bash", "Bash", "shell", "Shell"})

# Prefix length per leading token (default 1). ``git`` → 2 means the grant
# is ``git <subcommand>`` rather than all of git.
_BASH_PREFIX_ARITY: dict[str, int] = {
    "git": 2,
    "docker": 2,
    "kubectl": 2,
    "podman": 2,
    "npm": 2,
    "pnpm": 2,
    "yarn": 2,
    "cargo": 2,
    "go": 2,
    "pip": 2,
    "pip3": 2,
    "poetry": 2,
    "uv": 2,
    "make": 2,
    "systemctl": 2,
    "journalctl": 2,
    "gh": 3,
    "aws": 3,
    "gcloud": 3,
    "terraform": 2,
    "helm": 2,
    "pytest": 2,
}

# Segments that carry no grantable command of their own.
_BASH_SKIP_TOKENS = frozenset(
    {
        "cd",
        "true",
        "false",
        ":",
        "export",
        "set",
        "source",
        ".",
    }
)


def _shell_argv(segment: str) -> list[str]:
    """Tokenise one shell segment, dropping leading ``VAR=value`` assignments."""
    import shlex

    try:
        tokens = shlex.split(segment)
    except ValueError:
        return []
    i = 0
    while i < len(tokens):
        head = tokens[i].split("=", 1)[0]
        if "=" in tokens[i] and head and "/" not in head and head.replace("_", "").isalnum():
            i += 1
            continue
        break
    return tokens[i:]


def _bash_command_prefixes(command: str) -> list[str]:
    """Every grantable prefix in *command* — one per real shell segment."""
    from ..tools.safety import _split_command

    try:
        segments, _has_write, _has_sub = _split_command(command or "")
    except Exception:  # noqa: BLE001 — prefix math must never break the gate
        return []
    out: list[str] = []
    for seg in segments:
        tokens = _shell_argv(seg)
        if not tokens or tokens[0] in _BASH_SKIP_TOKENS:
            continue
        base = tokens[0].rsplit("/", 1)[-1]
        arity = max(1, int(_BASH_PREFIX_ARITY.get(base, 1)))
        out.append(" ".join(tokens[: min(arity, len(tokens))]))
    return out


def bash_command_prefix(command: str) -> str:
    """The prefix a fresh grant would cover (``"git worktree"``), or ``""``."""
    prefixes = _bash_command_prefixes(command)
    return prefixes[0] if prefixes else ""


# =====================================================================
# Permission protocol — flow through the event stream.
# =====================================================================
#
# The engine yields a :class:`PermissionRequested` event (defined in
# :mod:`.engine`) carrying ``response: asyncio.Future[str]``. The
# Agent intercepts it BEFORE the gateway translator sees it (it never
# appears on the ACP wire), runs :meth:`PermissionPolicy.evaluate`,
# and for ``"ask"`` decisions runs :func:`ask_user` to get an operator
# answer. Then ``response.set_result("allow"|"deny")`` unblocks the
# engine.


# =====================================================================
# Evaluator façade
# =====================================================================


class PermissionPolicy:
    """Adapter over today's :class:`PermissionEvaluator`.

    Reference: ``sharedagents_agent.tools.permissions`` (module).

    The Agent holds one of these per session (mode may be per-session-
    overridden via ``/mode``). Engines call ``evaluate(...)`` to get an
    immediate decision; the Agent layer is what bridges ``"ask"`` to
    the user.
    """

    # Public alias re-exported at module scope (see VALID_PERMISSION_MODES
    # at the bottom of this module) so callers like ``Agent._effective_mode``
    # don't reach into a private attribute on a sibling class.
    _VALID_MODES = ("plan", "ask", "auto")

    def __init__(self, *, mode: str, allowlist: set[str] | None = None) -> None:
        """``mode ∈ {plan, ask, auto}``.

        ``allowlist`` — a **shared** set of remembered ``"<tool>:<pattern>"``
        strings. The Agent constructs a fresh policy per tool call (mode may
        change mid-session), so the allowlist must be owned by the session and
        passed in here, or ``remember()`` would populate a set that is discarded
        the moment the call resolves — which is exactly why "don't ask again"
        was a no-op when the set was private to each call. When ``None`` (tests, one-off use)
        a private set is created.
        """
        if mode not in self._VALID_MODES:
            raise ValueError(f"unknown mode: {mode!r} — expected one of {', '.join(self._VALID_MODES)}")
        self._mode = mode
        # Session allowlist of ``"<tool>:<pattern>"`` strings — once
        # the user picks ``allow_always`` for a tool, later calls
        # with matching args auto-allow. Keyed by ``tool_name`` plus
        # a coarse argument hash so a one-off allow doesn't blanket
        # the whole tool. Owned by the session (see docstring) so it
        # survives across the per-call policy instances.
        self._allowlist: set[str] = allowlist if allowlist is not None else set()

    def set_mode(self, mode: str) -> None:
        """Per-session ``/mode`` override.

        Reference: ``CoreBackend.set_mode`` (acp_bridge.py L1654).
        Validates ``mode ∈ {plan, ask, auto}``.
        """
        if mode not in self._VALID_MODES:
            raise ValueError(f"unknown mode: {mode!r} — expected one of {', '.join(self._VALID_MODES)}")
        self._mode = mode

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        criticality: str,
    ) -> tuple[str, str]:
        """Return ``(decision, reason)`` per the policy table.

        ``decision ∈ {"allow", "deny", "ask"}``. ``reason`` is a
        short user-facing string used when the decision is non-allow.
        """
        crit = (criticality or "mutating").lower()
        if crit not in {"read_only", "mutating", "destructive", "network", "forbidden"}:
            crit = "mutating"

        # Hard deny BEFORE the mode-aware branches — ``forbidden`` is the
        # heuristic's "never even ask" class (rm -rf ~/, rm -rf /,
        # fork bomb, whole-disk wipe). Bypassing the ASK/auto-allowlist
        # paths is the point: a model must not have a route to get an
        # operator to allow these by reflex.
        if crit == "forbidden":
            return "deny", "forbidden: blast radius too large to ever allow"

        # An explicit operator "allow always" grant
        # auto-allows the calls it covers. The grant IS the human decision
        # the mode matrix would otherwise ask for (interactive plan mode
        # already admits a mutating call on a one-off operator Allow), so it
        # is consulted BEFORE the matrix. It can never admit a destructive
        # or forbidden call: those criticalities are excluded here and at
        # ``remember()`` time.
        if crit in ("read_only", "mutating", "network") and self._remembered(
            tool_name,
            arguments,
        ):
            return "allow", ""

        mode = self._mode

        # Decision matrix per the module docstring.
        if mode == "plan":
            if crit == "read_only":
                return "allow", ""
            # Delegation carve-out: the ``Agent`` delegation
            # primitive is the sole mutating/destructive-floored call plan mode
            # admits. Its gating is ceded to the child (the delegated
            # turn runs ask/auto under its own gate, never plan); build & plan
            # may delegate. Every OTHER
            # mutating/destructive tool stays denied in plan mode.
            if tool_name in _DELEGATION_TOOLS:
                return "allow", ""
            # OpenCode plan-file write carve-out (the single
            # live table now owns this — moved here from the offline
            # ``PermissionEvaluator``). A mutating Write/Edit/Update to the
            # operator plan sandbox (``.sharedagents/plans/*.md``) is allowed so the
            # plan agent can author its plan file while in plan mode.
            if tool_name in ("write", "edit", "Write", "Edit", "Update") and crit == "mutating":
                from ..tools.permissions import _is_plan_file_path

                if _is_plan_file_path(_plan_path_of(arguments)):
                    return "allow", ""
            # Web research reaches plan mode via the read_only branch above:
            # ``mcp__web__*`` declares ``readOnlyHint`` so discovery resolves it
            # to read_only (not NETWORK/mutating), and a planner can search the
            # web. The former by-name WebSearch/WebFetch allow-list is retired
            # now that they are MCP tools. ``acp_call`` (raw CLI execution) is NOT
            # read_only and NOT the delegation primitive, so it stays gated in
            # plan mode; only ``Agent`` (above) is carved out.
            if crit == "destructive":
                return "deny", "plan-mode: destructive blocked"
            return "ask", "plan-mode: needs confirmation"

        if mode == "auto":
            # Auto never waits on the permission widget.
            # FORBIDDEN already returned deny above.
            return "allow", ""

        # ask mode = OpenCode without --auto: cards appear for mutating /
        # network / destructive. read_only proceeds. FORBIDDEN already denied.
        # Allow-always grants are consulted above and never cover destructive.
        if crit == "read_only":
            return "allow", ""
        return "ask", "ask-mode: needs confirmation"

    # Shell tools carry a command-PREFIX grant instead of a signature grant
    # (module-level set re-exposed for the Agent layer's option building).
    _BASH_TOOLS = _BASH_TOOLS

    @staticmethod
    def bash_prefix_for(command: str) -> str:
        """The prefix an "Allow always" click on *command* would remember."""
        return bash_command_prefix(command)

    # Curated structured tools plus any named/MCP tool may be remembered.
    # Destructive / FORBIDDEN never enter the allowlist. Shell grants are a
    # command PREFIX (see ``_BASH_TOOLS`` / ``remember``).
    _REMEMBERABLE_TOOLS = frozenset(
        {
            "read",
            "write",
            "edit",
            "MultiEdit",
            "NotebookEdit",
            "glob",
            "grep",
            "list",
            "LS",
        }
    )
    _REMEMBERABLE_CRITICALITY = frozenset({"read_only", "mutating", "network"})

    def remember(self, tool_name: str, arguments: dict[str, Any], criticality: str = "mutating") -> bool:
        """Add this call pattern to the session allowlist, if safe.

        Returns ``True`` iff the pattern was actually remembered. Named and
        MCP tools grant the **tool id** so a later call with different args
        is not re-asked. Shell tools use a bounded command-prefix grant.
        Destructive and forbidden calls are never remembered.

        Called by the Agent layer when the user picks "allow, don't ask again".
        """
        if (criticality or "mutating").lower() not in self._REMEMBERABLE_CRITICALITY:
            return False
        if tool_name in _BASH_TOOLS:
            prefix = bash_command_prefix(
                str((arguments or {}).get("command", "")),
            )
            if not prefix:
                return False
            self._allowlist.add(f"bash-prefix:{prefix}")
            return True
        name = (tool_name or "").strip()
        if not name:
            return False
        self._allowlist.add(self._pattern_for(name, arguments))
        return True

    def is_rememberable(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        criticality: str = "mutating",
    ) -> bool:
        """True when Allow always can persist a grant for this call."""
        if (criticality or "mutating").lower() not in self._REMEMBERABLE_CRITICALITY:
            return False
        if tool_name in _BASH_TOOLS:
            return bool(
                bash_command_prefix(
                    str((arguments or {}).get("command", "")),
                )
            )
        return bool((tool_name or "").strip())

    def _remembered(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> bool:
        """True when a session grant covers this call (see ``remember``).

        Shell grants match by command PREFIX: a grant of ``git worktree``
        covers ``git worktree list`` / ``git worktree add …``. Named and MCP
        tools match by tool id, ignoring args.
        """
        if not self._allowlist:
            return False
        if tool_name in _BASH_TOOLS:
            granted = {key[len("bash-prefix:") :] for key in self._allowlist if key.startswith("bash-prefix:")}
            if not granted:
                return False
            prefixes = _bash_command_prefixes(
                str((arguments or {}).get("command", "")),
            )
            # A compound command is covered only when EVERY executable segment
            # is covered. Matching just one segment would let an unrelated
            # trailing command ride on a remembered prefix.
            return bool(prefixes) and all(
                any(prefix == grant or prefix.startswith(grant + " ") for grant in granted) for prefix in prefixes
            )
        return self._pattern_for(tool_name, arguments) in self._allowlist

    @staticmethod
    def _pattern_for(tool_name: str, arguments: dict[str, Any]) -> str:
        """Tool-id grant. Args are ignored so a later URL/path is covered."""
        return f"tool:{(tool_name or '').strip()}"


def allowlist_path(sessions_dir: Any, session_hash: str) -> Path | None:
    """``{sessions_dir}/{hash}.allowlist.json`` or None when unset."""
    sid = (session_hash or "").strip()
    if not sid or sessions_dir in (None, ""):
        return None
    try:
        return Path(sessions_dir) / f"{sid}.allowlist.json"
    except (TypeError, ValueError):
        return None


def load_persisted_allowlist(sessions_dir: Any, session_hash: str) -> set[str]:
    """Load persisted Allow-always grants. Missing/corrupt files are empty."""
    path = allowlist_path(sessions_dir, session_hash)
    if path is None:
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return set()
    if isinstance(raw, list):
        return {str(item) for item in raw if item}
    if isinstance(raw, dict):
        grants = raw.get("grants")
        if isinstance(grants, list):
            return {str(item) for item in grants if item}
    return set()


def persist_allowlist(
    sessions_dir: Any,
    session_hash: str,
    grants: set[str],
) -> Path | None:
    """Write the session allowlist. Returns the path or None on skip/failure."""
    path = allowlist_path(sessions_dir, session_hash)
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(sorted(grants), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return None
    return path


# =====================================================================
# ASK protocol — ACP wire path.
# =====================================================================


@dataclass(slots=True, frozen=True)
class AskQuestion:
    """One question to surface via ACP ``request_permission``.

    Reference: ``sharedagents_acp/protocol/schema.py`` for the wire shape.
    Today's ``_ask_user_outbound`` constructs equivalent dicts.

    The ACP wire field names (``optionId``, ``label``, ``kind``) are
    preserved by the translation layer; this dataclass is the
    Python-friendly version.
    """

    id: str
    title: str
    detail: str
    # ``default_index`` is auto-selected in ``auto`` mode (skipping the
    # round-trip) when ``self.id`` matches a known "is this safe?"
    # question pattern. Today: ``_ask_user_auto_answers`` (acp_bridge.py
    # L581–610) handles this.
    default_index: int = 0
    options: tuple[str, ...] = ("Allow", "Deny")
    # The host-resolved criticality of the underlying call (the more
    # cautious of the model's declared value and the runtime heuristic).
    # ``auto_answer`` gates on this so a destructive call never auto-approves
    # in auto mode regardless of how safe the command TEXT looks
    criticality: str = "mutating"


async def ask_user(
    questions: list[AskQuestion],
    *,
    session_id: str,
    mode: str,
    note: str = "",
    outbound: Any | None = None,  # an ACP outbound channel
) -> list[dict[str, Any]]:
    """Run the ASK round-trip and return the answers.

    Behavior:

    * If ``mode == "auto"``, short-circuit via
      :func:`auto_answer` for every question recognized by the
      auto-answer pattern matcher; surface only the unrecognized ones.
    * If ``mode == "plan"``, surface mutating requests to an attached
      operator channel; without one, deny closed.
    * Otherwise call ``outbound.request_permission(...)`` and
      ``await`` the user's response.
    """
    answers: list[dict[str, Any]] = []
    mode = (mode or "").lower()
    # Split gateway+orchestrator: the session's bound outbound is often the
    # non-interactive plumbing link, which answers -32601 for
    # ``session/request_permission``. Prefer the in-turn web channel
    # (same registry ``_ask_user_outbound`` uses) so a live operator
    # approval actually reaches the destructive bash gate.
    channel = outbound
    try:
        from sharedagents_gateway.ask_user import _web_ask_channel_for

        web = _web_ask_channel_for(session_id)
        if web is not None:
            channel = web
    except Exception:  # noqa: BLE001
        channel = outbound

    for q in questions:
        # Auto-mode auto-answer fast path.
        if mode == "auto":
            auto = auto_answer(q, note)
            if auto is not None:
                answers.append(auto)
                continue
        # ASK / plan-mutating / auto-destructive — go through the outbound
        # channel if one exists. Plan remains fail-closed when no interactive
        # channel is attached, but a live operator can now approve once or add
        # the offered session-scoped command-prefix grant.
        if channel is None:
            answers.append(
                {
                    "questionId": q.id,
                    "optionId": "deny",
                    "decision": "deny",
                    "reason": "no outbound channel attached",
                }
            )
            continue
        try:
            payload = {
                "id": q.id,
                "title": q.title,
                "detail": q.detail,
                "options": [{"optionId": opt, "label": opt} for opt in q.options],
                "defaultIndex": q.default_index,
            }
            # AcpServer exposes ``request_outbound(method, params)``
            # — the generic outbound-request entry point. We surface
            # permission prompts via the ACP ``session/request_permission``
            # method. Match the historical legacy-bridge
            # payload shape (sessionId / toolCall / options) so a
            # frontend built against the prior wire stays compatible.
            request_fn = getattr(channel, "request_outbound", None)
            if request_fn is None:
                # Last-resort: some stub backends expose a direct
                # ``request_permission(session_id=, question=)``
                # helper. Honor it if present so test fakes that
                # pre-date the AcpServer rewrite keep working.
                legacy_fn = getattr(channel, "request_permission", None)
                if legacy_fn is None:
                    answers.append(
                        {
                            "questionId": q.id,
                            "optionId": "deny",
                            "decision": "deny",
                            "reason": "outbound has no request_permission / request_outbound",
                        }
                    )
                    continue
                res = legacy_fn(session_id=session_id, question=payload)
            else:
                # ACP spec ``PermissionOption.kind`` must be one of
                # ``allow_once`` / ``allow_always`` / ``reject_once`` /
                # ``reject_always``. The dataclass ``options`` field
                # carries display labels (``"Allow"`` / ``"Deny"``);
                # map each label to a spec-conforming triple so a
                # strict ACP client doesn't reject the request.
                _OPT_KIND = {
                    "allow": "allow_once",
                    "yes": "allow_once",
                    "y": "allow_once",
                    "ok": "allow_once",
                    "allow_once": "allow_once",
                    "allow always": "allow_always",
                    "allow, don't ask again": "allow_always",
                    "allow_always": "allow_always",
                    "deny": "reject_once",
                    "no": "reject_once",
                    "n": "reject_once",
                    "reject": "reject_once",
                    "reject_once": "reject_once",
                    "reject_always": "reject_always",
                }
                spec_options = []
                for opt in q.options or ["Allow", "Deny"]:
                    kind = _OPT_KIND.get(str(opt).lower(), "reject_once")
                    spec_options.append(
                        {
                            "optionId": kind,
                            "name": str(opt),
                            "kind": kind,
                        }
                    )
                req_payload = {
                    "sessionId": session_id,
                    "toolCall": {
                        "toolCallId": q.id,
                        "title": q.title or "tool",
                        "kind": "other",
                        "status": "pending",
                        "rawInput": {"detail": q.detail or ""},
                    },
                    "options": spec_options,
                }
                res = request_fn(
                    "session/request_permission",
                    req_payload,
                )
            if hasattr(res, "__await__"):
                # No wall-clock timeout: a parked gate is already fail-closed
                # because the tool cannot run until this await resolves. The
                # operator may answer later, or cancel/tear down the owning turn.
                res = await res
            # Unwrap the AcpServer-shape outcome:
            #   {"outcome": {"outcome": "selected"|"cancelled",
            #                "optionId": "allow_once"|...}}
            if isinstance(res, dict) and "outcome" in res:
                outcome = res.get("outcome") or {}
                if isinstance(outcome, dict):
                    kind = outcome.get("outcome")
                    if kind == "cancelled":
                        res = {"optionId": "deny", "reason": "user cancelled"}
                    elif kind == "selected":
                        opt_id = str(outcome.get("optionId") or "")
                        res = {"optionId": opt_id}
                    else:
                        res = {"optionId": "deny", "reason": f"unknown outcome: {kind!r}"}
            # Normalize: caller may return {"optionId": ...} or a string.
            # ``allow_once`` / ``allow_always`` are the ACP-spec
            # PermissionOption ids the legacy bridge
            # used; map them to ``"allow"`` so callers don't need to
            # care about which option-id naming the wire uses.
            if isinstance(res, dict):
                opt = res.get("optionId") or res.get("decision") or "deny"
                ans = dict(res)
                ans.setdefault("questionId", q.id)
                ans.setdefault("optionId", opt)
                ans["decision"] = (
                    "allow"
                    if str(opt).lower()
                    in (
                        "allow",
                        "allow_once",
                        "allow_always",
                        "allow always",
                        "allow, don't ask again",
                        "yes",
                        "y",
                        "0",
                    )
                    else "deny"
                )
                answers.append(ans)
            else:
                opt = str(res or "deny")
                answers.append(
                    {
                        "questionId": q.id,
                        "optionId": opt,
                        "decision": "allow" if opt.lower() in ("allow", "yes", "y") else "deny",
                    }
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            answers.append(
                {
                    "questionId": q.id,
                    "optionId": "deny",
                    "decision": "deny",
                    "reason": f"ask_user error: {exc}",
                }
            )
    return answers


def auto_answer(question: AskQuestion, note: str = "") -> dict[str, Any] | None:
    """Auto-mode default for one question.

    Reference: ``_ask_user_auto_answers`` (acp_bridge.py L581–610).
    Returns the auto-selected answer dict, or ``None`` when the
    question doesn't match any auto-answer pattern.
    """
    title = (question.title or "").lower()
    detail = (question.detail or "").lower()

    # A call the host classified ``destructive`` NEVER
    # auto-approves in auto mode, regardless of how safe the command TEXT
    # looks. The host-resolved criticality (registered kind + runtime heuristic,
    # with any legacy claim able to escalate only) is authoritative here; decline the
    # auto-answer fast path so the operator is asked via the outbound channel.
    if (getattr(question, "criticality", "") or "").lower() == "destructive":
        return None

    # Tool-permission questions in auto mode: allow safe-looking ones,
    # defer destructive-looking ones to the operator. This
    # keyword set must be at least as conservative as the tool-level
    # ``tools.safety.classify_command`` machinery — the old nine-phrase list
    # let an irreversible action phrased with a synonym ("purge", "erase",
    # "tear down") auto-approve with no operator visibility. Widened to cover
    # the common irreversible-action verbs so a destructive-sounding question
    # defers to the operator rather than silently taking the default option.
    destructive_signals = (
        "rm -rf",
        "drop table",
        "drop database",
        "drop schema",
        "truncate",
        "force push",
        "force-push",
        "delete",
        "destroy",
        "wipe",
        "purge",
        "erase",
        "expunge",
        "obliterate",
        "clobber",
        "overwrite",
        "revoke",
        "deallocate",
        "tear down",
        "teardown",
        "uninstall",
        "shutdown",
        "reboot",
        "format disk",
        "reformat",
        "mkfs",
        "shred",
    )
    text = f"{title} {detail}"
    if any(sig in text for sig in destructive_signals):
        return None

    # Tools that should always auto-approve in auto-mode.
    safe_prefixes = (
        "allow read",
        "allow grep",
        "allow glob",
        "allow webfetch",
        "allow websearch",
        "allow ls",
    )
    if any(title.startswith(p) for p in safe_prefixes):
        return {
            "questionId": question.id,
            "optionId": "allow",
            "decision": "allow",
            "auto": True,
            "note": note,
        }

    # ``request_permission`` style — answer with the default option.
    if question.options:
        default_label = question.options[max(0, min(question.default_index, len(question.options) - 1))]
        return {
            "questionId": question.id,
            "optionId": default_label,
            "decision": "allow" if default_label.lower() in ("allow", "yes", "y") else "deny",
            "auto": True,
            "note": note,
        }

    return None


# =====================================================================
# UserQuestion tool — frontend-driven ASK
# =====================================================================
#
# The ``UserQuestion`` tool (registry.py — searchable) is the model-
# facing version of this: the LOCAL LLM emits a ``UserQuestion`` tool
# call when it needs operator input mid-turn. Today this is
# orchestrator-only (the ``tools_allowlist`` excludes ``UserQuestion``
# for sub-agents because they don't have a user to ask).
#
# After refactor: same behavior. ``Agent.apply_pre_tool_hooks``
# returns ``(False, "no user attached")`` for ``UserQuestion`` calls
# from a non-orchestrator agent. Orchestrator ``UserQuestion`` calls go through
# the same ``ask_user`` helper above.


# Public re-export of the canonical mode set. ``Agent._effective_mode``
# and ``acp_bridge.set_mode`` / ``gateway_shim.set_mode`` import this
# instead of reaching into ``PermissionPolicy._VALID_MODES`` (private).
#
# This is an IDENTITY binding — both names refer to the same tuple
# object created in the class body above. Equality holds today
# trivially (`VALID_PERMISSION_MODES is PermissionPolicy._VALID_MODES`),
# but a future edit that REBINDS ``_VALID_MODES`` to a new tuple in
# ``PermissionPolicy`` would silently desync the public alias.
# The assertion below trips at import time if anyone breaks the link.
VALID_PERMISSION_MODES = PermissionPolicy._VALID_MODES
assert VALID_PERMISSION_MODES is PermissionPolicy._VALID_MODES, (
    "VALID_PERMISSION_MODES drift: a code change rebound "
    "PermissionPolicy._VALID_MODES without updating the public alias "
    "in this module. Re-pin or remove the alias."
)


def coerce_permission_mode(mode_id: str) -> str:
    """Map a picker/native/permission id onto ``plan`` / ``ask`` / ``auto``.

    Cursor's advertised default is native ``agent``, OpenCode's is ``build``,
    Hermes' is ``dont_ask``. Those are permission-layer ``auto``. Reject
    anything that is neither a permission mode nor a classified native
    alias — same ``ValueError`` text as ``PermissionPolicy``.
    """
    mid = str(mode_id or "").strip()
    if mid in VALID_PERMISSION_MODES:
        return mid
    from sharedagents_agent.acp_mode_dict import classify_native_mode

    classified = classify_native_mode(mid)
    if classified in VALID_PERMISSION_MODES:
        return classified
    raise ValueError(f"unknown mode: {mode_id!r} — expected one of {', '.join(VALID_PERMISSION_MODES)}")
