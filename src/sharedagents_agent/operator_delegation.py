"""Frontend helper: auto-fill the pre-flight checklist for an OPERATOR
slash-command delegation.

The pre-flight delegation gate (:mod:`sharedagents_agent.delegation_gate`) BLOCKS an
expensive/third-party delegate call whose checklist — ``goal``, ``constraints``,
``acceptance_criteria``, ``budget_usd`` — is incomplete. That
gate is correct for the *model's* ``Agent`` tool path, where the schema forces
the model to state intent, constraints, acceptance, and budget before spending
frontier tokens.

An operator typing ``/claude <prompt>`` / ``/codex <prompt>`` / ``/antigravity
<prompt>`` (or ``/<sibling> <task>``) is a DIFFERENT case: it is a DIRECT HUMAN
command. The operator IS the intent and the authorization — the checklist the
gate wants to see is answered implicitly the moment a supervising human types
the command. Blocking that behind a form the human already answered would break
operator slash delegations for no safety gain.

So the FRONTEND command handler — never the model — auto-fills the checklist
for operator slash delegations and threads it into the delegate call args
alongside the ``prompt``. Two properties this preserves:

* **Frontend-set, not model-reachable.** This helper is invoked only from the
  TUI (:mod:`sharedagents_tui.agent_client`, :mod:`sharedagents_tui.app`) and Telegram
  (:mod:`sharedagents_telegram.bot`) slash-command handlers. The model's ``Agent``
  tool path does not call it; that path fills the checklist itself via the tool
  schema, so the model can never mint an auto-approved expensive delegation.
* **Gate-compatible.** The returned values are chosen to pass
  :func:`sharedagents_agent.delegation_gate.evaluate_preflight` — each prose field is
  non-empty and not a placeholder token, and ``budget_usd`` parses to a positive
  number.
"""

from __future__ import annotations

from typing import Any

# Default budget for an operator-initiated slash delegation. Any positive
# number satisfies the gate's ``budget_usd`` check; this is a sane per-call
# ceiling for a single supervised frontier consult.
_OPERATOR_SLASH_BUDGET_USD = 5.0


def operator_slash_checklist(prompt: str, *, budget_usd: float = _OPERATOR_SLASH_BUDGET_USD) -> dict[str, Any]:
    """Return the four auto-filled pre-flight checklist fields for an operator
    slash-command delegation.

    ``prompt`` — the operator's typed prompt/task; its first ~200 chars become
    the ``goal`` so the audit record names what the operator asked for.

    The result is merged into the delegate-tool call args by the frontend
    handler (the ``prompt``/``task`` field is added by the caller, not here).
    Every value is chosen to pass
    :func:`sharedagents_agent.delegation_gate.evaluate_preflight` on an expensive
    engine.
    """
    text = (prompt or "").strip()
    snippet = text[:200] if text else "(no prompt supplied)"
    return {
        "goal": f"operator-initiated: {snippet}",
        "constraints": "operator-initiated direct slash command; operator supervises.",
        "acceptance_criteria": "operator judges the result.",
        "budget_usd": budget_usd,
    }


__all__ = ("operator_slash_checklist",)
