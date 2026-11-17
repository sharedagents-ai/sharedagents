"""``AskUserQuestion`` outbound routing for :class:`CoreBackend`.

Extracted from ``acp_bridge.py`` (behavior-preserving de-monolith). The
model's ``AskUserQuestion`` / ``question`` tool can't run inside the
sandboxed subprocess runner — it round-trips through the connected
frontend (TUI modal / web / Telegram) via ``session/ask_user``. This
module owns:

* the web-turn interactive channel registry
  (``register_web_ask_channel`` / ``unregister_web_ask_channel`` /
  ``_web_ask_channel_for``), populated by ``control_endpoint._drive``;
* the wire-shape helpers (``_normalize_ask_questions``,
  ``_format_ask_answers_for_model``, ``_ask_user_auto_answers``);
* the outbound issuer (``_ask_user_outbound``) with its robust
  degradation to synthesized **unanswered** replies (fail-closed;
  never forge option-0 consent).

Re-exported from ``sharedagents_gateway.acp_bridge`` for import stability
(``control_endpoint`` and tests import these by that path).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .outbound import _resolve_outbound_channel

if TYPE_CHECKING:
    from sharedagents_gateway.acp_bridge import CoreBackend


# --- AskUserQuestion outbound -------------------------------------------


# Web-submitted turns (REST ``/prompt``) have no interactive bidirectional ACP
# client of their own. ``control_endpoint._drive`` registers a browser-backed
# ``session/ask_user`` channel here for the turn's duration. ``_ask_user_outbound``
# PREFERS this over whatever outbound channel is bound to the session, because in
# a split gateway+orchestrator deployment the orchestrator's per-session outbound
# channel is the (non-interactive) gateway↔orchestrator ``RemoteBackend`` ACP
# link — it doesn't implement ``session/ask_user`` and answers ``-32601``, which
# would otherwise force an unanswered synth even though a human is sitting
# at the browser. Keyed by session id (one in-flight turn per session).
_WEB_ASK_CHANNELS: dict[str, Any] = {}


def register_web_ask_channel(session_id: str, channel: Any) -> None:
    """Register ``channel`` as the interactive ask_user channel for a web turn."""
    if session_id:
        _WEB_ASK_CHANNELS[session_id] = channel


def unregister_web_ask_channel(session_id: str, channel: Any | None = None) -> None:
    """Drop the web ask_user channel for ``session_id``.

    When ``channel`` is given, only drops it if it's still the current one
    (a concurrent re-submit could have replaced it). Idempotent.
    """
    if not session_id:
        return
    if channel is None or _WEB_ASK_CHANNELS.get(session_id) is channel:
        _WEB_ASK_CHANNELS.pop(session_id, None)


def _web_ask_channel_for(session_id: str) -> Any | None:
    """Return the registered interactive web ask_user channel, or ``None``."""
    return _WEB_ASK_CHANNELS.get(session_id)


def _sdlc_one_question_only(session_id: str, questions: list[Any]) -> int:
    """How many questions to drop because this is an SDLC clarify session.

    The Dev-tab interview is worth having only if each question is written
    AFTER the previous answer -- that is the whole difference between an
    interview and a form. A model that batches five questions into one card
    has written all five before learning anything, so the fourth cannot
    depend on the second.

    The prompt asks for one at a time and the conductor has complied, but a
    prompt is a request. This makes it a property: on a session that owns an
    SDLC board, only the first question survives, and the model is told the
    rest were dropped so it re-asks them one by one with the answer in hand.

    Scoped to SDLC sessions on purpose -- Chat and Telegram legitimately ask
    several things at once, and capping them would be a regression.
    """
    if len(questions) <= 1:
        return 0
    try:
        from sharedagents_agent.tools.impl.sdlc import sdlc_path_for_session
        from sharedagents_core.kernel.layout import sessions_root

        path = sdlc_path_for_session(sessions_root(), str(session_id or ""))
    except Exception:  # noqa: BLE001 — not an SDLC deployment; leave it alone
        return 0
    try:
        if not path or not Path(path).exists():
            return 0
    except Exception:  # noqa: BLE001
        return 0
    return len(questions) - 1


def _normalize_ask_questions(questions: list[Any]) -> list[Any]:
    """Mirror opencode's ``multiple`` onto the wire's ``multiSelect`` field.

    The model-facing ``question`` tool schema uses opencode's ``multiple``
    (tool parity), but the connected frontends (TUI modal / web) read
    ``multiSelect`` off the ``session/ask_user`` wire payload. Copy ``multiple``
    → ``multiSelect`` (without dropping either) so multi-select keeps working
    after the rename. Non-mapping entries pass through untouched.
    """
    out: list[Any] = []
    for q in questions:
        if isinstance(q, Mapping) and "multiple" in q and "multiSelect" not in q:
            q = {**q, "multiSelect": bool(q.get("multiple"))}
        out.append(q)
    return out


def _is_system_unanswered_note(note: str) -> bool:
    """True when ``notes`` is a fail-closed synth / Stop, not a typed Other."""
    n = str(note or "").strip().lower()
    if not n:
        return True
    if n in {"stopped", "cancelled", "canceled"}:
        return True
    return n.startswith("(unanswered") or n.startswith("unanswered")


def _format_ask_answers_for_model(questions: list[Any], answers: list[Any]) -> str:
    """Render answers as opencode's ``question`` tool output string.

    Matches opencode (tool/question.ts): ``User has answered your questions:
    "q"="a, b", … . You can now continue with the user's answers in mind.``
    Answers pair with questions by index; an empty/absent selection renders as
    ``Unanswered`` unless the operator typed an Other answer in ``notes``.
    """
    parts: list[str] = []
    quoted_from_notes: set[str] = set()
    selected_labels: set[str] = set()
    truly_unanswered = False
    for i, q in enumerate(questions):
        qtext = ""
        if isinstance(q, Mapping):
            qtext = str(q.get("question") or "")
        selected: list[str] = []
        notes_i = ""
        if i < len(answers) and isinstance(answers[i], Mapping):
            sel = answers[i].get("selected")
            if isinstance(sel, list):
                selected = [str(s) for s in sel if str(s)]
            notes_i = str(answers[i].get("notes") or "").strip()
        selected_labels.update(selected)
        if selected:
            joined = ", ".join(selected)
        elif notes_i and not _is_system_unanswered_note(notes_i):
            joined = notes_i
            quoted_from_notes.add(notes_i)
        else:
            joined = "Unanswered"
            truly_unanswered = True
        parts.append(f'"{qtext}"="{joined}"')
    formatted = ", ".join(parts)
    out = f"User has answered your questions: {formatted}. You can now continue with the user's answers in mind."
    # Preserve auto-answer transparency: when the answers were synthesized
    # (delivery failed / no interactive channel), each carries a ``notes``
    # string. Surface it so the model knows the answer wasn't user-picked.
    # Do not repeat a typed Other that is already the quoted answer.
    notes = []
    for a in answers:
        if isinstance(a, Mapping):
            n = str(a.get("notes") or "").strip()
            if n and n not in notes and n not in quoted_from_notes and n not in selected_labels:
                notes.append(n)
    if notes:
        out += " " + " ".join(notes)
    # Fail-closed hint for local orchestrators: empty selection is never
    # consent (Off29: timeout/option-0 synth forged a destructive yes).
    # A typed Other in notes is an answer — do not attach this clause.
    if truly_unanswered:
        out += " Unanswered means no consent — do not proceed with destructive or privileged actions; stop or re-ask."
    return out


def _ask_user_auto_answers(questions: list[Any], note: str) -> list[dict[str, Any]]:
    """Synthesize **unanswered** replies for each question (fail-closed).

    Used when the connected frontend doesn't implement ``session/ask_user``,
    the outbound request errors, the web channel times out, or
    ``server_handle`` is None. Never selects option 0 — forged consent
    caused Off29. ``note`` is preserved so the model sees the degradation.

    (silence ≠ yes).
    """
    answers: list[dict[str, Any]] = []
    for q in questions:
        if not isinstance(q, Mapping):
            continue
        answers.append(
            {
                "question": str(q.get("question") or ""),
                "selected": [],
                "notes": note,
            }
        )
    return answers


async def _ask_user_outbound(
    backend: CoreBackend,
    session_id: str,
    call: Mapping[str, Any],
) -> tuple[bool, str]:
    """Issue ``session/ask_user`` to the client and return ``(ok, content)``.

    The model's ``AskUserQuestion`` tool can't be executed inside the
    sandboxed subprocess runner — it has to round-trip through the
    connected frontend (TUI modal / Telegram / etc.) so a human can
    pick answers. We hijack the normal ``_default_run_tool`` dispatch
    on tool-name match and send the call out over the wire instead.

    Wire shape (request):

    .. code:: json

        {
          "sessionId": "...",
          "callId": "...",
          "questions": [
            {
              "question": "...", "header": "...", "multiSelect": false,
              "options": [
                {"label": "...", "description": "..."},
                ...
              ]
            },
            ...
          ]
        }

    Wire shape (response):

    .. code:: json

        {
          "answers": [
            {"question": "...", "selected": ["<label>", ...], "notes": "..."},
            ...
          ]
        }

    The returned ``content`` string is a stable JSON dump of the answers
    list so the model gets a deterministic tool_result it can parse.

    Robust degradation: frontends that don't implement
    ``session/ask_user`` (Telegram, OpenAI endpoint) return
    methodNotFound on the wire; rather than failing the tool we
    synthesize **unanswered** replies + a transparent ``notes`` string
    so the model never hangs and knows no human consented.
    """
    args = call.get("arguments") or {}
    questions_raw = args.get("questions") if isinstance(args, Mapping) else None
    if not isinstance(questions_raw, list) or not questions_raw:
        return False, "AskUserQuestion: arguments.questions must be a non-empty list"
    # Mirror opencode's ``multiple`` onto the wire's ``multiSelect`` so the
    # frontends (which read ``multiSelect``) still honor multi-select after the
    # tool-schema rename.
    questions: list[Any] = _normalize_ask_questions(list(questions_raw))
    dropped = _sdlc_one_question_only(session_id, questions)
    if dropped:
        questions = questions[:1]

    # Tell the model what happened, or it will believe the dropped questions
    # were asked and answered. Rebinding the formatter keeps the five return
    # paths below untouched.
    _fmt = _format_ask_answers_for_model
    if dropped:

        def _fmt(qs, ans, _n=dropped, _base=_format_ask_answers_for_model):
            return _base(qs, ans) + (
                f" NOTE: {_n} further question(s) in that same call were NOT "
                "asked. This interview is one question at a time. Ask the next "
                "one now, written against the answer above."
            )

    payload: dict[str, Any] = {
        "sessionId": session_id,
        "callId": str(call.get("id") or ""),
        "questions": questions,
    }

    # Web-submitted turns register an interactive browser channel (see
    # ``register_web_ask_channel``). Prefer it over the session's bound outbound
    # channel: in the split deployment that bound channel is the non-interactive
    # gateway↔orchestrator link, which answers ``-32601`` and would force an
    # auto-answer even though a human is at the browser. The web channel parks a
    # future the ``/answer`` route resolves. It waits until that explicit
    # answer or ``/cancel`` — no wall-clock timeout (the human may reply
    # the next day). Delivery failure still synthesizes unanswered.
    web_chan = _web_ask_channel_for(session_id)
    if web_chan is not None:
        try:
            result = await web_chan.request_outbound("session/ask_user", payload)
        except asyncio.CancelledError:
            # Task teardown, not a delivery failure — must propagate.
            raise
        except Exception as exc:  # noqa: BLE001
            # Off29 P0: delivery failure → unanswered synth,
            # never a hang and never option-0 consent.
            synth = _ask_user_auto_answers(questions, f"(unanswered: web channel failed — {exc})")
            return True, _fmt(questions, synth)
        answers = result.get("answers") if isinstance(result, Mapping) else None
        if not isinstance(answers, list):
            answers = _ask_user_auto_answers(questions, "(unanswered: web channel malformed response)")
        return True, _fmt(questions, answers)

    server_handle = _resolve_outbound_channel(backend, session_id)
    if server_handle is None:
        # No outbound channel (self-contained task / tests / stub
        # server). Synthesize unanswered so the model gets a parseable
        # tool_result without forged consent.
        synth = _ask_user_auto_answers(questions, "(unanswered: no outbound channel)")
        return True, _fmt(questions, synth)

    try:
        result = await server_handle.request_outbound("session/ask_user", payload)
    except asyncio.CancelledError:
        # Task teardown, not a delivery failure — must propagate.
        raise
    except Exception as exc:  # noqa: BLE001
        # Off29 P0: methodNotFound / transport error →
        # unanswered synth (turn never hangs; silence ≠ yes).
        synth = _ask_user_auto_answers(questions, f"(unanswered: outbound failed — {exc})")
        return True, _fmt(questions, synth)

    if not isinstance(result, Mapping):
        return False, "AskUserQuestion: malformed response (not a mapping)"
    answers = result.get("answers")
    if not isinstance(answers, list):
        return False, "AskUserQuestion: malformed response (answers not a list)"

    return True, _fmt(questions, answers)
