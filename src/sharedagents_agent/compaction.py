"""Context compaction / distillation extracted from ``turn_runtime``.

Message compaction, tool-output pruning, transcript summarization, the
distiller recovery path, and the pinned-facts / rolling-summary / todo
injectors. Every symbol is re-exported from ``sharedagents_agent.turn_runtime``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from sharedagents_model.sampling import apply_sampling, configured_sampling, sampling_row

from .loop_detection import _env_float, _env_int

# Distiller prompt body lives in ``<home>/DISTILL.md`` (operator-
# editable), loaded fresh on every recovery call via
# ``load_prompt_pair``. The file uses ``<!-- system -->`` / ``<!--
# user -->`` markers; the user section may carry ``{user}`` and
# ``{thinking}`` placeholders that this module fills in. If the file
# is missing or either section is empty, ``_distill_partial_reasoning``
# raises so the caller can fall through to its non-distill recovery
# path rather than send a half-formed prompt to the model.

# Thinking budgets bound synthesis independently of optional sampling config.
_DISTILL_THINKING_BUDGET: int = _env_int("SHAREDAGENTS_DISTILL_THINKING_BUDGET", 2048)
#: Bounded budget for the native-Bedrock one-shot distill. The OpenAI HTTP
#: path streams with an unbounded read (a slow upstream must not trip a read
#: timeout mid-stream); a worker subprocess needs an explicit ceiling so a
#: wedged SDK call cannot pin the recovery path forever.
_BEDROCK_ONESHOT_TIMEOUT_S: float = _env_float("SHAREDAGENTS_BEDROCK_ONESHOT_TIMEOUT_S", 90.0)


# Built-in fallback distiller prompt — used when the agent's
# ``<home>/DISTILL.md`` is missing or empty. Mirrors the shape of
# ``main``'s on-disk template so newly-scaffolded sub-agents (e.g.
# ``coder``) recover from loops out of the box. Operators can still
# override per-agent by dropping a real ``DISTILL.md`` into the
# agent's home — the file-based load wins when present.
_DEFAULT_DISTILL_SYSTEM: str = (
    "You compress in-progress reasoning into a structured handoff so "
    "a downstream model can finish the reply without re-deriving "
    "anything. The downstream model will VERIFY the facts you list, "
    "so do not over-claim — only state what the reasoning trace "
    "actually established. Mark anything still uncertain in "
    "`open_questions`. Output a SINGLE JSON object matching this "
    "schema and nothing else — no preamble, no markdown fences, no "
    "commentary:\n"
    "{\n"
    '  "established_facts": ["...", "..."],\n'
    '  "open_questions":    ["...", "..."],\n'
    '  "next_action":       "..."\n'
    "}\n"
    "Each bullet must be ONE concise sentence. Empty arrays / empty "
    "string are valid when the reasoning genuinely produced nothing "
    "in that category."
)
_DEFAULT_DISTILL_USER: str = (
    "The reasoning below was produced for the user request:\n"
    "<user_request>\n{user}\n</user_request>\n\n"
    "It was cut off before a final answer (the model either looped, "
    "ran out of thinking budget, or stopped without writing the "
    "reply). Extract a structured JSON handoff per the schema in the "
    "system prompt:\n"
    "- `established_facts`: claims supported by tool results / "
    "context in the trace below. NOT speculation.\n"
    "- `open_questions`: things the trace was uncertain about, or "
    "that would need re-verification before relying on.\n"
    "- `next_action`: the single most concrete next step the trace "
    "was moving toward — a tool call, a sub-question, or the final "
    "answer shape — in one sentence.\n\n"
    "REASONING SO FAR:\n"
    "<thinking>\n{thinking}\n</thinking>\n"
)

# ── LLM-based context compaction (Hermes-style) ──────────────────────
# Unlike ``compact_messages`` (which DELETES old turns), this summarizes the
# dropped span with the model so the small model never loses the thread —
# the dominant cause of mid-task false-positive loops / abrupt failures on
# qwen3.6-27b. The summary is anti-poisoning by construction: observed facts
# only, exact paths/errors/commands preserved verbatim, unknowns explicit,
# and a conservative next step (codex design review 2026-06-19).
_COMPACT_THINKING_BUDGET: int = _env_int("SHAREDAGENTS_COMPACT_THINKING_BUDGET", 1024)
# Cap on the transcript text (chars) fed to the summarizer in one round so a
# huge head can't itself overflow the summarizer call. Oldest content past the
# cap is dropped before summarizing (least relevant); the cap is generous.
_COMPACT_MAX_TRANSCRIPT_CHARS: int = _env_int(
    "SHAREDAGENTS_COMPACT_MAX_TRANSCRIPT_CHARS",
    240_000,
)

# ── OpenCode-style tool-output pruning (compaction.ts PRUNE_PROTECT) ──
_PRUNE_PROTECT_TOKENS: int = _env_int("SHAREDAGENTS_PRUNE_PROTECT_TOKENS", 40_000)
_PRUNE_MINIMUM_TOKENS: int = _env_int("SHAREDAGENTS_PRUNE_MINIMUM_TOKENS", 20_000)
# Fraction of the model context window kept as protected recent tool output.
# OpenCode's fixed 40k ≈ 20% of a 200k window; scale up for 1M contexts so
# prune does not wipe history while the window is still mostly empty.
_PRUNE_PROTECT_FRAC: float = _env_float("SHAREDAGENTS_PRUNE_PROTECT_FRAC", 0.20)
# Reference window the fixed floors were tuned against (~200k-class models).
_PRUNE_SCALE_REF_CONTEXT: int = _env_int("SHAREDAGENTS_PRUNE_SCALE_REF_CONTEXT", 200_000)
_PRUNE_TAIL_TURNS: int = _env_int("SHAREDAGENTS_PRUNE_TAIL_TURNS", 2)
_PRUNE_NEVER_TOOLS: frozenset[str] = frozenset(
    {
        # names compared lowercase (_tool_name_for_call lowercases). Memory
        # recall moved to mcp__memory__*, so protect those too — else a long
        # transcript could prune the recalled-memory output the builtin used to keep.
        "skillsearch",
        "remember",
        "memory_search",
        "memory_get",
        "mcp__memory__search",
        "mcp__memory__get",
        "mcp__memory__session_search",
        "mcp__memory__remember",
        "taskcreate",
        "taskupdate",
        "tasklist",
        "taskget",
    }
)
_PRUNED_MARKER_PREFIX = "[pruned:"


def scaled_prune_protect(context_window: int = 0) -> int:
    """PRUNE_PROTECT scaled to the model context window.

    Floors at the OpenCode 40k constant for ≤200k windows; grows at
    ``_PRUNE_PROTECT_FRAC`` of the window for larger contexts so 1M-class
    models keep a proportional recent tool-output tail.
    """
    floor = _PRUNE_PROTECT_TOKENS
    ctx = int(context_window or 0)
    if ctx <= 0 or ctx <= _PRUNE_SCALE_REF_CONTEXT:
        return floor
    frac = float(_PRUNE_PROTECT_FRAC or 0.0)
    if frac <= 0:
        return floor
    return max(floor, int(ctx * frac))


def scaled_prune_minimum(context_window: int = 0) -> int:
    """PRUNE_MINIMUM companion — half of protect, floored at OpenCode 20k."""
    protect = scaled_prune_protect(context_window)
    return max(_PRUNE_MINIMUM_TOKENS, protect // 2)


# OpenCode compaction.ts — safety buffer before auto-compact fires.
_COMPACT_SAFETY_BUFFER: int = _env_int("SHAREDAGENTS_COMPACT_SAFETY_BUFFER", 20_000)

# OpenCode packages/core/src/session/compaction.ts SUMMARY_TEMPLATE.
_SUMMARY_TEMPLATE: str = (
    "Output exactly the Markdown structure shown inside <template> and keep "
    "the section order unchanged. Do not include the <template> tags in your "
    "response.\n"
    "<template>\n"
    "## Goal\n"
    "- [single-sentence task summary]\n\n"
    "## Constraints & Preferences\n"
    '- [user constraints, preferences, specs, or "(none)"]\n\n'
    "## Progress\n"
    "### Done\n"
    '- [completed work or "(none)"]\n\n'
    "### In Progress\n"
    '- [current work or "(none)"]\n\n'
    "### Blocked\n"
    '- [blockers or "(none)"]\n\n'
    "## Key Decisions\n"
    '- [decision and why, or "(none)"]\n\n'
    "## Next Steps\n"
    '- [ordered next actions or "(none)"]\n\n'
    "## Critical Context\n"
    '- [important technical facts, errors, open questions, or "(none)"]\n\n'
    "## Relevant Files\n"
    '- [file or directory path: why it matters, or "(none)"]\n'
    "</template>\n\n"
    "Rules:\n"
    "- Keep every section, even when empty.\n"
    "- Use terse bullets, not prose paragraphs.\n"
    "- Preserve exact file paths, commands, error strings, and identifiers "
    "when known.\n"
    "- Do not mention the summary process or that context was compacted.\n"
    "- Do not record the session budget, spend figures, cost limits or a "
    "budget-exhausted state; the runtime supplies the live budget every turn."
)

# OpenCode packages/opencode/src/agent/prompt/compaction.txt (shipped fallback).
_DEFAULT_COMPACT_SYSTEM: str = (
    "You are an anchored context summarization assistant for coding sessions.\n\n"
    "Summarize only the conversation history you are given. The newest turns may "
    "be kept verbatim outside your summary, so focus on the older context that "
    "still matters for continuing the work.\n\n"
    "If the prompt includes a <previous-summary> block, treat it as the current "
    "anchored summary. Update it with the new history by preserving still-true "
    "details, removing stale details, and merging in new facts.\n\n"
    "Always follow the exact output structure requested by the user prompt. "
    "Keep every section, preserve exact file paths and identifiers when known, "
    "and prefer terse bullets over paragraphs.\n\n"
    "Rules:\n"
    "- OBSERVED FACTS ONLY — never infer success you did not see\n"
    "- Preserve verbatim: file paths, symbol names, error messages, commands, "
    "key numbers\n"
    "- Keep unresolved blockers and open questions\n"
    "- Do not answer the conversation itself\n"
    "- Do not mention that you are summarizing, compacting, or merging context\n"
    "- Respond in the same language as the conversation"
)


def compaction_agent_home() -> Path:
    """Agent config dir for the hidden compaction summarizer."""
    from sharedagents_core.layout import agent_config_dir

    return agent_config_dir("compaction")


COMPACTION_SUBAGENT_NAME = "compaction"
COMPACTION_SUBAGENT_TITLE = "Compact context"

#: Hidden utility agents whose calls are shown as delegation sessions: the
#: operator can open them and read the turn, but never continue them.
#: ``summary`` is the Tasks-board completion recap (parent-session side
#: call); its child sessions are view-only and monitor-hidden.
INTERNAL_AGENT_NAMES = frozenset({"compaction", "title", "summary"})
_INTERNAL_AGENT_TITLES = {
    "compaction": COMPACTION_SUBAGENT_TITLE,
    "title": "Generate title",
    "summary": "Task recap",
}


#: Internal children the monitor tree hides (see orchestrator.MONITOR_HIDDEN_AGENTS).
_MONITOR_HIDDEN_CHILDREN = frozenset({"title", "summary"})


def is_internal_agent(name: Any) -> bool:
    """True for the hidden agents whose sessions are view-only."""
    return str(name or "").strip().lower() in INTERNAL_AGENT_NAMES


class VisibleCompactionSubagent:
    """Foreground-only compaction child visible in the Agent Monitor.

    Same session/tree contract as explore/general: ``parent_session_id`` on
    the child header, ``subagent_spawned`` on the parent stream, a live
    ``turn_started`` while the summarizer runs, then ``turn_ended``. The
    caller waits (never ``async_jobs``) and applies the child's result as
    the parent's replacement context.
    """

    __slots__ = ("child_sid", "writer", "sessions_dir")

    def __init__(self, child_sid: str, writer: Any, sessions_dir: Path) -> None:
        self.child_sid = child_sid
        self.writer = writer
        self.sessions_dir = sessions_dir

    def finish(
        self,
        *,
        content: str = "",
        aborted: bool = False,
        error: str | None = None,
    ) -> None:
        try:
            self.writer.turn_ended(
                turn_idx=1,
                final_content=content or "",
                aborted=aborted,
                error=error,
            )
        except Exception:  # noqa: BLE001 — monitor spawn must not break /compact
            pass
        try:
            from sharedagents_session.session_writer import close_session_writer

            close_session_writer(self.sessions_dir, self.child_sid)
        except Exception:  # noqa: BLE001
            pass


def spawn_visible_internal_subagent(
    parent: Any,
    *,
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    engine: str | None = None,
    title: str | None = None,
) -> VisibleCompactionSubagent | None:
    """Create a visible, view-only child session for a hidden utility call
    (compaction, title, summary).

    Same session/tree contract as a delegation: ``parent_session_id`` on the
    child header, ``subagent_spawned`` on the parent stream, a live
    ``turn_started`` while the call runs, then ``turn_ended`` (see
    :class:`VisibleCompactionSubagent`). ``engine`` is the engine that RUNS the
    call; pass it explicitly because the parent header's ``engine`` can be stale
    after an agent switch. Skips (returns None) when the parent has no durable
    sessions dir. Never raises.
    """
    parent_sid = str(getattr(parent, "hash", "") or "").strip()
    sessions_dir = getattr(parent, "_storage_root", None)
    if not parent_sid or sessions_dir is None:
        return None
    sessions_dir = Path(sessions_dir)
    child_title = title or _INTERNAL_AGENT_TITLES.get(agent_name, agent_name)
    try:
        from sharedagents_agent.agents.session_log import record_session_creation
        from sharedagents_session.session import Session
        from sharedagents_session.session_writer import get_session_writer

        child = Session.new(
            system_prompt=system_prompt,
            model=str(getattr(parent, "model", "") or ""),
            endpoint=str(getattr(parent, "endpoint", "") or ""),
            sessions_dir=sessions_dir,
            context_window=int(getattr(parent, "context_window", 0) or 0),
            parent_session_id=parent_sid,
            workspace_origin=f"delegate:{parent_sid}",
        )
        child.title = child_title
        record_session_creation(
            child,
            agent_name=agent_name,
            engine=str(engine or getattr(parent, "engine", "") or "local_llm"),
            system_prompt=system_prompt,
            sessions_dir=sessions_dir,
            parent_session_id=parent_sid,
            announce_in_parent=agent_name not in _MONITOR_HIDDEN_CHILDREN,
        )
        writer = get_session_writer(
            sessions_dir,
            child.hash,
            engine=str(child.engine or "local_llm"),
        )
        writer.header_update(patch={"title": child_title})
        # ``agent_name`` matters: the gateway resolves a session's owner from the
        # last ``turn_started``, which otherwise falls back to the ENGINE name
        # ("local_llm") and hides that this is a view-only internal agent.
        writer.turn_started(
            turn_idx=1,
            user=user_prompt,
            permission_mode="plan",
            agent_name=agent_name,
        )
        return VisibleCompactionSubagent(child.hash, writer, sessions_dir)
    except Exception:  # noqa: BLE001 — the utility call must still run
        return None


def spawn_visible_compaction_subagent(
    parent: Any,
    *,
    user_prompt: str = "Compact the parent session context.",
    engine: str | None = None,
) -> VisibleCompactionSubagent | None:
    """Create a visible compaction child — the WebUI Compact button AND every
    automatic mid-turn compaction that actually shrinks the prompt.

    ``engine`` is the engine that RUNS the compaction (in-process, the
    local_llm engine). Pass it explicitly: the parent header's ``engine`` can
    be stale after an agent switch (prod d422f7ca7358 stamped the child
    ``antigravity_cli`` for a deepseek compaction).
    """
    from sharedagents_agent.agent_profiles import COMPACTION, load_profile_agents_md

    system_msg = ""
    try:
        system_msg = load_profile_agents_md(
            compaction_agent_home(),
            COMPACTION,
        ).strip()
    except Exception:  # noqa: BLE001
        system_msg = ""
    return spawn_visible_internal_subagent(
        parent,
        agent_name=COMPACTION_SUBAGENT_NAME,
        system_prompt=system_msg or _DEFAULT_COMPACT_SYSTEM,
        user_prompt=user_prompt,
        engine=engine,
        title=COMPACTION_SUBAGENT_TITLE,
    )


def _build_compact_user_prompt(
    *,
    task: str,
    transcript: str,
    previous_summary: str = "",
) -> str:
    """OpenCode ``buildPrompt`` — anchor + SUMMARY_TEMPLATE + task + transcript."""
    anchor = (
        "Update the anchored summary below using the conversation history above.\n"
        "Preserve still-true details, remove stale details, and merge in new "
        "facts.\n"
        f"<previous-summary>\n{previous_summary.strip()}\n</previous-summary>"
        if previous_summary.strip()
        else "Create a new anchored summary from the conversation history."
    )
    task_line = (task or "(no current task line)").strip()
    return "\n\n".join(
        [
            anchor,
            _SUMMARY_TEMPLATE,
            (f"The agent is mid-task. The user's current/most-recent request is:\n<task>\n{task_line}\n</task>"),
            (
                "Below is the EARLIER transcript that is about to be dropped from "
                "context to stay under the model's window:\n\n"
                f"<transcript>\n{transcript}\n</transcript>"
            ),
        ]
    )


def load_compaction_prompts(
    *,
    task: str,
    transcript: str,
    previous_summary: str = "",
) -> tuple[str, str]:
    """Resolve compaction (system, user) prompts.

    Operators customize summarization behavior by editing
    ``agents/compaction/AGENTS.md`` (re-read from disk every call). The user
    turn is built by the runtime (OpenCode ``buildPrompt`` + transcript).
    Falls back to the built-in ``_DEFAULT_COMPACT_SYSTEM`` template
    (observed-facts-only, anti-poisoning) when no profile system message
    is configured.
    """
    from sharedagents_agent.agent_profiles import COMPACTION, load_profile_agents_md

    home = compaction_agent_home()
    system_msg = load_profile_agents_md(home, COMPACTION).strip()
    if not system_msg:
        system_msg = _DEFAULT_COMPACT_SYSTEM

    task_line = (task or "(no current task line)").strip()
    prev = (previous_summary or "").strip()
    user_msg = _build_compact_user_prompt(
        task=task_line,
        transcript=transcript,
        previous_summary=prev,
    )
    return system_msg, user_msg


# --- plan-pass for complex turns --------------------------------------
#
# Plan-and-Act pattern (arXiv 2503.09572) applied to a single-model
# setup: when the user's turn looks like planning, run a one-shot
# planner call BEFORE the main reasoning+tool loop, with high thinking
# budget and no tools, asking the model to break the task into ≤5
# ordered subtasks. The resulting plan is injected as a `[PLAN]`
# scaffold so every subsequent round of the main loop has narrower
# per-step scope (which empirically reduces loop frequency on small
# models).
#
# Activation has two tiers:
#   1. FAST SKIP (zero cost) — short prompts that match a greeting /
#      ack / single-word-imperative regex. No upstream call.
#   2. CLASSIFIER (one ~500ms LLM call) — everything else asks the
#      upstream model to classify the prompt. JSON output:
#      `{plan_pass, kind, reason}`. On classifier failure the gate
#      fails CLOSED (no plan-pass) — better to skip than to fire on
#      a wrong heuristic.
#
# (Plan-pass / classifier was removed 2026-05-06 — the model now
# calls the canonical ``TodoWrite`` tool when it identifies multi-step
# work, instead of the bridge running mandatory pre-prompt LLM rounds.
# The persistent injections below — pinned facts + compaction summary
# — still apply between system prompt and live conversation tail.)


# Cap on persistent injection blocks so a runaway compaction summary
# or pinned-facts list can't push the cache prefix past the model's
# context window.
_PINNED_FACTS_MAX_CHARS: int = _env_int(
    "SHAREDAGENTS_PINNED_FACTS_MAX_CHARS",
    2000,
)
_COMPACTION_SUMMARY_MAX_CHARS: int = _env_int(
    "SHAREDAGENTS_COMPACTION_SUMMARY_MAX_CHARS",
    3000,
)


def _inject_pinned_facts(
    messages: list[dict],
    pinned_facts: list[str],
    *,
    max_chars: int = _PINNED_FACTS_MAX_CHARS,
) -> list[dict]:
    """Insert a `[PINNED FACTS]` user message right after the system
    prompt so the model sees these facts on every round, regardless of
    compaction.

    Rendered shape:
        [PINNED FACTS — survive every compaction; do not contradict]
        - fact 1
        - fact 2
        - …

    Idempotent: if a `[PINNED FACTS]` message is already present at the
    expected slot, it's REPLACED rather than duplicated. If
    `pinned_facts` is empty, any prior pinned-facts message is removed.
    Truncates to `max_chars` chars to keep the prompt budget bounded.
    """
    out = list(messages)
    sys_idx = 0 if out and out[0].get("role") == "system" else -1
    insert_idx = sys_idx + 1
    # Strip any existing pinned-facts marker.
    stripped = []
    for _i, m in enumerate(out):
        c = m.get("content")
        if isinstance(c, str) and c.startswith("[PINNED FACTS") and m.get("role") == "user":
            continue
        stripped.append(m)
    out = stripped
    if not pinned_facts:
        return out
    body = "\n".join(f"- {f}" for f in pinned_facts if str(f).strip())
    if not body:
        return out
    rendered = f"[PINNED FACTS — survive every compaction; do not contradict]\n{body}"
    if len(rendered) > max_chars:
        rendered = rendered[: max_chars - 3] + "..."
    msg = {"role": "user", "content": rendered}
    return out[:insert_idx] + [msg] + out[insert_idx:]


def _inject_compaction_summary(
    messages: list[dict],
    compaction_summary: str,
) -> list[dict]:
    """Insert the rolling cross-compaction summary right after pinned
    facts (or the system prompt if there are no pinned facts). Keeps
    one slot per session — replaces any existing
    `[ROLLING COMPACTION SUMMARY]` message in place.

    The point of having both this AND the in-turn `_splice_summary`
    (which compresses the CURRENT history into a single 'EARLIER
    CONVERSATION SUMMARY' message) is that `_splice_summary` only
    captures what was in the prompt at THIS compaction time. When
    successive compactions fire, each new in-turn summary's input is
    the previous summary plus new turns — so old facts can drop out
    silently. The rolling summary persists across compactions, gets
    merged forward, and acts as a long-term memory floor.
    """
    if not compaction_summary or not compaction_summary.strip():
        return list(messages)
    # Budget state goes stale the moment the cap moves; a frozen "budget
    # exhausted" line here outvoted the live figures for 13 hours in a jarvis
    # session. Also cleans summaries stored before this rule.
    compaction_summary = _strip_budget_state(compaction_summary)
    if not compaction_summary.strip():
        return list(messages)
    out = list(messages)
    # Strip any existing rolling summary marker.
    stripped = []
    for m in out:
        c = m.get("content")
        if isinstance(c, str) and c.startswith("[ROLLING COMPACTION SUMMARY") and m.get("role") == "user":
            continue
        stripped.append(m)
    out = stripped
    sys_idx = 0 if out and out[0].get("role") == "system" else -1
    # Insert AFTER pinned facts if present.
    insert_idx = sys_idx + 1
    while (
        insert_idx < len(out)
        and out[insert_idx].get("role") == "user"
        and isinstance(out[insert_idx].get("content"), str)
        and out[insert_idx]["content"].startswith("[PINNED FACTS")
    ):
        insert_idx += 1
    rendered = (
        "[ROLLING COMPACTION SUMMARY — persistent memory floor across "
        "compactions; treat as already-established context]\n" + compaction_summary.strip()
    )
    msg = {"role": "user", "content": rendered}
    return out[:insert_idx] + [msg] + out[insert_idx:]


def _inject_current_todos(
    messages: list[dict],
    todos: list[dict],
) -> list[dict]:
    """Insert a `[CURRENT TODO LIST]` user message after the system
    prompt (and after pinned facts / rolling summary if present), so
    the model sees the live plan on every round.

    Without this, prior turns are replayed via ``_build_messages`` using
    only the assistant's final text — tool calls and tool results are
    dropped. When a prior turn aborted with no final text, the model
    starts the next turn with zero visibility into the persisted
    ``.todos.json`` and dutifully calls ``todo(replace=…)`` again,
    overwriting the prior plan with a near-duplicate. This injection
    surfaces the existing list so the model picks ``complete``/``add``
    instead.

    Idempotent: replaces any prior `[CURRENT TODO LIST]` message at the
    same slot. Empty/missing list → strips any prior marker (so a
    completed-and-cleared plan disappears from the prefix on the next
    round).
    """
    out = list(messages)
    # Strip any existing marker.
    out = [
        m
        for m in out
        if not (
            m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m["content"].startswith("[CURRENT TODO LIST")
        )
    ]
    if not todos:
        return out
    sys_idx = 0 if out and out[0].get("role") == "system" else -1
    insert_idx = sys_idx + 1
    # Insert AFTER pinned facts and rolling summary if present.
    while (
        insert_idx < len(out)
        and out[insert_idx].get("role") == "user"
        and isinstance(out[insert_idx].get("content"), str)
        and (
            out[insert_idx]["content"].startswith("[PINNED FACTS")
            or out[insert_idx]["content"].startswith("[ROLLING COMPACTION SUMMARY")
        )
    ):
        insert_idx += 1
    lines = []
    for i, t in enumerate(todos):
        mark = "x" if t.get("status") in ("done", "completed") else " "
        content = (t.get("content") or "").strip()
        if content:
            lines.append(f"- [{mark}] {i}: {content}")
    if not lines:
        return out
    rendered = (
        "[CURRENT TODO LIST — persisted from earlier in this session; "
        "to make changes call `TodoWrite` with action=complete/add/delete. "
        "Do NOT call action=replace unless starting a fresh, unrelated task]\n" + "\n".join(lines)
    )
    msg = {"role": "user", "content": rendered}
    return out[:insert_idx] + [msg] + out[insert_idx:]


def _merge_compaction_summary(
    prior: str,
    fresh: str,
    *,
    max_chars: int = _COMPACTION_SUMMARY_MAX_CHARS,
) -> str:
    """Deterministic rolling merge of the cross-compaction summary.

    The fresh per-compaction summary already covers everything that
    was visible at this compaction; we keep the prior rolling summary
    as a "this is what older compactions established" prefix and
    append a "...and most recently:" tail. If the combined length
    exceeds `max_chars`, the prior summary is truncated tail-first
    (oldest content drops earliest). Pure deterministic: an LLM-based
    re-summarisation would be more accurate but blocks compaction on
    yet another upstream call.
    """
    prior = _strip_budget_state(prior or "").strip()
    fresh = _strip_budget_state(fresh or "").strip()
    if not prior:
        return fresh[:max_chars]
    if not fresh:
        return prior[:max_chars]
    sep = "\n\n--- and most recently ---\n\n"
    combined = prior + sep + fresh
    if len(combined) <= max_chars:
        return combined
    # Truncate prior from the FRONT (drop oldest material) until the
    # combined fits. Fresh summary gets to keep its full text. The
    # `[older context truncated]` marker counts against the budget.
    marker = "[older context truncated]\n"
    overhead = len(sep) + len(fresh) + len(marker)
    room_for_prior = max(0, max_chars - overhead)
    if room_for_prior <= 0:
        # Fresh summary alone (with marker + sep) is already at/over the
        # cap. Take the tail of fresh; prior is dropped entirely.
        return fresh[-max_chars:]
    truncated_prior = prior[-room_for_prior:]
    # Drop a leading partial sentence so the truncation point is on
    # a sentence boundary, but only if doing so leaves non-trivial
    # prior content (otherwise the truncation removes everything).
    cut = truncated_prior.find(". ")
    if 0 <= cut < 200 and cut + 2 < len(truncated_prior) - 20:
        truncated_prior = truncated_prior[cut + 2 :]
    out = marker + truncated_prior + sep + fresh
    # Final safety cap — the sentence-boundary trim above can leave us
    # very slightly over budget if `prior` lacked a clean cut point.
    if len(out) > max_chars:
        out = out[-max_chars:]
    return out


def _parse_distill_json(raw: str) -> tuple[dict | None, str]:
    """Best-effort JSON extraction from the distiller's output.

    Qwen sometimes wraps JSON in a `<think>` block, in markdown fences,
    or emits a small leading prose preamble despite the system-prompt
    instruction to output JSON only. We try the cheap parses first
    (raw, then strip fences, then locate the first `{` and last `}`)
    and fall through to "raw text fallback" so the caller can decide
    whether to use the structured form or treat the whole blob as the
    legacy free-text distill.

    Returns `(parsed_dict_or_None, fallback_text)`. The dict is None
    when nothing parseable was found; in that case the caller may
    use `fallback_text` (the original raw stripped of fences and
    `<think>` blocks) as a non-structured handoff instead of failing
    outright.
    """
    if not raw:
        return None, ""
    # Strip <think>...</think> blocks — present when the model leaked
    # its reasoning despite the prompt asking for JSON only.
    import re as _re

    cleaned = _re.sub(
        r"<thinking?\b[^>]*>.*?</thinking?\s*>",
        "",
        raw,
        flags=_re.IGNORECASE | _re.DOTALL,
    ).strip()
    # Strip markdown code fences (```json, ```).
    if cleaned.startswith("```"):
        # remove leading fence line and trailing fence
        cleaned = _re.sub(r"^```[a-zA-Z0-9]*\s*\n", "", cleaned, count=1)
        cleaned = _re.sub(r"\n```\s*$", "", cleaned)
        cleaned = cleaned.strip()
    # Try direct JSON parse.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, cleaned
    except (json.JSONDecodeError, ValueError):
        pass
    # Try locating the first balanced `{...}` block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, candidate
        except (json.JSONDecodeError, ValueError):
            pass
    return None, cleaned


def _format_distill_for_recovery(parsed: dict | None, fallback_text: str) -> str:
    """Render a distill result into the recovery-prompt body.

    Structured form (parsed JSON):
        Established facts:
          - …
          - …
        Open questions:
          - …
        Next action: …

    Unstructured fallback: just the raw cleaned text. The recovery
    nudge wraps this with the "verify before relying" instruction.
    """
    if not isinstance(parsed, dict):
        return fallback_text.strip()
    lines: list[str] = []
    facts = parsed.get("established_facts") or []
    if isinstance(facts, list) and facts:
        lines.append("Established facts:")
        for f in facts:
            lines.append(f"  - {str(f).strip()}")
    open_qs = parsed.get("open_questions") or []
    if isinstance(open_qs, list) and open_qs:
        lines.append("Open questions:")
        for q in open_qs:
            lines.append(f"  - {str(q).strip()}")
    nxt = parsed.get("next_action")
    if isinstance(nxt, str) and nxt.strip():
        lines.append(f"Next action: {nxt.strip()}")
    return "\n".join(lines).strip() or fallback_text.strip()


_DISTILL_SUCCESS_MARKERS: tuple[str, ...] = (
    "succeed",
    "success",
    "complete",
    "deployed",
    "passed",
    "works",
    "working",
    "resolved",
    "fixed",
    "verified",
    "confirmed",
    "merged",
    "shipped",
    "installed",
)


def _reclassify_unearned_success(
    parsed: dict | None,
    thinking: str,
) -> dict | None:
    """Demote fabricated/inferred success out of the established-facts list.

    Anti-poisoning: a distilled ``established_fact``
    may only carry a claim the reasoning trace actually established. An upstream
    distiller can hallucinate a success the trace never observed ("The
    deployment succeeded.") — carrying that into the next round as an
    authoritative fact is a trust-boundary failure. So any established fact that
    asserts success/completion while the reasoning trace itself shows no
    corroborating success signal is moved to ``open_questions`` (rephrased as a
    "Whether ..." item) for the downstream model to re-verify, never presented
    as fact.
    """
    if not isinstance(parsed, dict):
        return parsed
    facts = parsed.get("established_facts")
    if not isinstance(facts, list) or not facts:
        return parsed
    trace = (thinking or "").lower()
    trace_has_success = any(mk in trace for mk in _DISTILL_SUCCESS_MARKERS)
    kept: list = []
    demoted: list[str] = []
    for f in facts:
        text = str(f).strip()
        low = text.lower()
        claims_success = any(mk in low for mk in _DISTILL_SUCCESS_MARKERS)
        # An asserted success the trace never corroborates is "inferred
        # success you did not see" — demote it to an open question.
        if claims_success and not trace_has_success:
            q = text.rstrip(". ").strip()
            if q:
                demoted.append("Whether " + q[0].lower() + q[1:])
            continue
        kept.append(f)
    if not demoted:
        return parsed
    out = dict(parsed)
    out["established_facts"] = kept
    open_qs = list(out.get("open_questions") or [])
    open_qs.extend(demoted)
    out["open_questions"] = open_qs
    return out


async def _distill_partial_reasoning(
    *,
    endpoint: str,
    model: str,
    headers: dict,
    user_prompt: str,
    thinking: str,
    home: Path,
    cfg: Any = None,
) -> tuple[str, dict]:
    """Compress an in-flight CoT into a structured JSON handoff.

    Distiller runs with `enable_thinking=True` (Qwen 3.6 27B's
    no-thinking mode degrades synthesis quality) but capped at a
    small `thinking_budget` so it can't inherit the main turn's
    loop. Optional sampling comes from side_sampling.distill. The output
    is parsed as JSON; on parse failure we
    fall back to the raw text (still useful as a free-form summary).
    Streams the response so a slow upstream never triggers a read
    timeout. Returns
    `(formatted_handoff, debug_info)`. Raises on HTTP/transport failure,
    empty output, or a missing/incomplete ``<home>/DISTILL.md`` —
    recovery cannot itself recursively recover.
    """
    from sharedagents_agent.system_prompt import load_prompt_pair

    prompt_pair = load_prompt_pair(home / "DISTILL.md")
    system_msg = (prompt_pair.get("system") or "").strip()
    user_template = (prompt_pair.get("user") or "").strip()
    if not system_msg or not user_template:
        # Built-in fallback so newly-scaffolded agents (e.g. ``coder``
        # without an authored DISTILL.md) still recover from loops
        # instead of cancelling on first hit. The packaged template
        # mirrors the structure ``main``'s on-disk DISTILL.md uses
        # (system role tells the model what to produce; user role
        # carries the looped trace + original prompt). Operators can
        # override per-agent by dropping a real DISTILL.md into
        # ``<home>/`` — the load above wins when that file exists.
        system_msg = _DEFAULT_DISTILL_SYSTEM
        user_template = _DEFAULT_DISTILL_USER
    try:
        user_msg = user_template.format(
            user=(user_prompt or "(no original user prompt)").strip(),
            thinking=thinking,
        )
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"distill skipped: {home}/DISTILL.md template missing placeholder: {e}") from e

    resolved_sampling = configured_sampling(
        cfg,
        sampling_row(cfg, endpoint, model),
        model,
        purpose="distill",
    )
    bedrock_row = _bedrock_row_for(cfg, endpoint, model)
    if bedrock_row is not None:
        # Native Bedrock row: never POST to ``bedrock://.../chat/completions``.
        # A bounded one-shot Converse worker carries the row's own credential
        # mode; prompts, DISTILL.md formatting and the post-processing below
        # stay identical to the HTTP path.
        started_bedrock = time.monotonic()
        out_bedrock, metrics = await _run_bedrock_oneshot(
            cfg,
            bedrock_row,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            sampling=resolved_sampling,
            timeout_s=_BEDROCK_ONESHOT_TIMEOUT_S,
        )
        out_bedrock = out_bedrock.strip()
        if not out_bedrock:
            raise RuntimeError("distill returned empty content (bedrock oneshot)")
        return _finalize_distill(
            out_bedrock,
            thinking,
            usage={
                "prompt_tokens": metrics.get("prompt_tokens"),
                "completion_tokens": metrics.get("completion_tokens"),
            },
            finish_reason=metrics.get("finish_reason"),
            duration_s=time.monotonic() - started_bedrock,
        )

    payload = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }
    apply_sampling(payload, resolved_sampling)
    if _sends_chat_template_kwargs(cfg, endpoint, model):
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "thinking_budget": _DISTILL_THINKING_BUDGET,
        }
    url = endpoint.rstrip("/") + "/chat/completions"
    timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
    sse_headers = {"Accept": "text/event-stream", **headers}
    started = time.monotonic()
    chunks: list[str] = []
    finish_reason: str | None = None
    usage: dict = {}
    saw_data_line: bool = False
    saw_choices: bool = False
    response_content_type: str = ""
    async with httpx.AsyncClient(timeout=timeout, headers=sse_headers) as c:
        async with c.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise RuntimeError(f"distill upstream HTTP {resp.status_code}: {body}")
            response_content_type = str(resp.headers.get("content-type") or "")
            async for raw in resp.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                saw_data_line = True
                data_str = raw[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                u = obj.get("usage")
                if isinstance(u, dict):
                    usage = u
                choices = obj.get("choices") or []
                if not choices:
                    continue
                saw_choices = True
                ch0 = choices[0] or {}
                delta = ch0.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    chunks.append(piece)
                fr = ch0.get("finish_reason")
                if fr:
                    finish_reason = fr
    duration_s = round(time.monotonic() - started, 3)
    out = "".join(chunks).strip()
    if not out:
        # Distinguish between failure modes so an operator reading
        # the session record can debug:
        #   - never saw a `data:` line at all → upstream returned a
        #     non-SSE 200 (raw JSON / plain text). Most likely
        #     `stream=true` was ignored or stripped en route, or the
        #     proxy collapsed the SSE response. Surface the response
        #     content-type so they know where to look.
        #   - saw `data:` lines but never any `choices` → SSE was
        #     emitted, but only `usage` / heartbeat frames; the model
        #     produced zero output for this prompt.
        #   - saw choices but `chunks` is empty → all content fields
        #     came back empty/None (model genuinely produced nothing).
        if not saw_data_line:
            raise RuntimeError(
                "no SSE data lines received — check Content-Type and "
                "stream support on the upstream "
                f"(response Content-Type={response_content_type!r}, "
                f"finish_reason={finish_reason!r})"
            )
        if not saw_choices:
            raise RuntimeError(
                "SSE stream carried no `choices` payloads — upstream "
                "responded but produced zero output (usage-only / "
                "heartbeat frames). "
                f"finish_reason={finish_reason!r}"
            )
        raise RuntimeError(f"distill returned empty content (finish_reason={finish_reason!r})")
    return _finalize_distill(
        out,
        thinking,
        usage=usage,
        finish_reason=finish_reason,
        duration_s=duration_s,
    )


def _finalize_distill(
    out: str,
    thinking: str,
    *,
    usage: dict,
    finish_reason: str | None,
    duration_s: float,
) -> tuple[str, dict]:
    """Shared distill post-processing for both transports.

    The Converse one-shot path and the OpenAI HTTP path must produce
    byte-identical recovery handoffs: prompts, DISTILL.md formatting and
    the structured-JSON parsing are transport-independent.
    """
    parsed, fallback_text = _parse_distill_json(out)
    # Strip any invented/inferred success the trace
    # never established before it can enter the recovery handoff.
    parsed = _reclassify_unearned_success(parsed, thinking)
    formatted = _format_distill_for_recovery(parsed, fallback_text)
    debug = {
        "duration_s": duration_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": finish_reason,
        "raw_chars": len(out),
        "structured": parsed is not None,
        "established_facts_count": (len(parsed.get("established_facts") or []) if isinstance(parsed, dict) else 0),
        "open_questions_count": (len(parsed.get("open_questions") or []) if isinstance(parsed, dict) else 0),
        "next_action_chars": (len(parsed.get("next_action") or "") if isinstance(parsed, dict) else 0),
    }
    return formatted, debug


def _sends_chat_template_kwargs(cfg: Any, endpoint: str, model: str) -> bool:
    """True when the ``(endpoint, model)`` row accepts vLLM-only body knobs.

    ``chat_template_kwargs`` (enable_thinking / thinking_budget) is a
    vLLM/SGLang extension; hosted OpenAI-compatible upstreams such as Cerebras
    reject it with ``HTTP 400: property 'chat_template_kwargs' is unsupported``.
    Mirrors the worker request builder's ``openai_compat`` classification so
    the out-of-band summarizer and the main turn cannot disagree. An unknown
    row (no ``cfg`` / no match) keeps the legacy local-vLLM behavior.
    """
    row: Mapping[str, Any] = {}
    if cfg is not None:
        try:
            resolver = getattr(cfg, "model_row_for_endpoint", None)
            row = (resolver(endpoint, model) if callable(resolver) else None) or {}
        except Exception:  # noqa: BLE001 — never fail the caller's fallback path
            row = {}
    from sharedagents_model.models import row_is_hosted

    return not row_is_hosted(row)


def _bedrock_row_for(
    cfg: Any,
    endpoint: str,
    model: str,
) -> Mapping[str, Any] | None:
    """Return the native Bedrock model row for ``(endpoint, model)``, else None.

    The out-of-band inference paths take an ``endpoint``/``model`` pair and
    must not be recast around a session. When the pair resolves to a native
    Bedrock row they call the Converse worker subprocess instead of POSTing
    to ``<endpoint>/chat/completions`` (``bedrock://`` is not HTTP).

    Traceability: plan/bedrock-native-converse.md (BED-03/05)
    """
    if cfg is None:
        return None
    try:
        resolver = getattr(cfg, "model_row_for_endpoint", None)
        row = resolver(endpoint, model) if callable(resolver) else None
    except Exception:  # noqa: BLE001 — never fail the caller's fallback path
        return None
    if not row:
        return None
    try:
        from sharedagents_model.bedrock_config import is_bedrock_row
    except ImportError:  # pragma: no cover — bedrock_config ships in-repo
        return None
    return row if is_bedrock_row(row) else None


async def _run_bedrock_oneshot(
    cfg: Any,
    row: Mapping[str, Any],
    messages: list[dict[str, Any]],
    *,
    sampling: dict | None = None,
    timeout_s: float,
) -> tuple[str, dict]:
    """Run one bounded Bedrock one-shot and return ``(text, metrics)``.

    Thin, function-local wrapper so this agent module never imports the
    gateway at module scope (``sharedagents_gateway.worker_request`` imports this
    package's ``turn_runtime`` — a module-level import would be cyclic).
    Errors propagate as :class:`RuntimeError` so each caller keeps its
    existing mechanical-fallback behavior.
    """
    from sharedagents_gateway.worker_request import run_bedrock_oneshot

    text, metrics = await run_bedrock_oneshot(
        cfg,
        row,
        messages,
        sampling=sampling,
        timeout_s=timeout_s,
    )
    return text, (metrics or {})


async def _summarize_context_span(
    *,
    endpoint: str,
    model: str,
    headers: dict,
    transcript: str,
    task: str,
    home: Path | None = None,
    previous_summary: str = "",
    timeout_s: float = 90.0,
    cfg: Any = None,
) -> str:
    """LLM-summarize an earlier conversation span into a compact running memo.

    Prompts resolve from ``agents/compaction/AGENTS.md`` so operators can
    customize the hidden compaction agent without touching parent-agent config.
    The ``home`` argument is kept for backward compatibility but is unused.

    Returns a formatted, human- and model-readable Markdown summary block.
    Raises on HTTP/transport failure, timeout, or empty output so the caller
    can fall back to mechanical compaction.
    """
    del home  # prompts always come from agents/compaction/
    system_msg, user_msg = load_compaction_prompts(
        task=task,
        transcript=transcript,
        previous_summary=previous_summary,
    )

    resolved_sampling = configured_sampling(
        cfg,
        sampling_row(cfg, endpoint, model),
        model,
        purpose="compaction",
    )
    bedrock_row = _bedrock_row_for(cfg, endpoint, model)
    if bedrock_row is not None:
        # Native Bedrock row: the memo is produced by a bounded one-shot
        # Converse worker (no /chat/completions at a ``bedrock://`` endpoint).
        # Prompt selection and ``_wrap_compact_summary`` formatting are shared
        # with the HTTP path so the persisted memo is identical.
        out_bedrock, _metrics = await _run_bedrock_oneshot(
            cfg,
            bedrock_row,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            sampling=resolved_sampling,
            timeout_s=timeout_s,
        )
        out_bedrock = out_bedrock.strip()
        if not out_bedrock:
            raise RuntimeError("compact summarizer returned empty (bedrock oneshot)")
        return _wrap_compact_summary(out_bedrock)

    payload = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }
    apply_sampling(payload, resolved_sampling)
    if _sends_chat_template_kwargs(cfg, endpoint, model):
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "thinking_budget": _COMPACT_THINKING_BUDGET,
        }
    url = endpoint.rstrip("/") + "/chat/completions"
    # Bounded total time: a broken/slow summarizer must not stall the turn —
    # the caller treats a timeout as a failure and uses mechanical compaction.
    timeout = httpx.Timeout(connect=10.0, read=timeout_s, write=10.0, pool=10.0)
    sse_headers = {"Accept": "text/event-stream", **headers}
    chunks: list[str] = []
    finish_reason: str | None = None
    async with httpx.AsyncClient(timeout=timeout, headers=sse_headers) as c:
        async with c.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:300]
                raise RuntimeError(f"compact summarizer HTTP {resp.status_code}: {body}")
            async for raw in resp.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                data_str = raw[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch0 = choices[0] or {}
                piece = (ch0.get("delta") or {}).get("content")
                if piece:
                    chunks.append(piece)
                fr = ch0.get("finish_reason")
                if fr:
                    finish_reason = fr
    out = "".join(chunks).strip()
    if not out:
        raise RuntimeError(f"compact summarizer returned empty (finish_reason={finish_reason!r})")
    return _wrap_compact_summary(out)


def _strip_compact_fences(raw: str) -> str:
    """Drop accidental markdown fences from summarizer output."""
    text = (raw or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _strip_budget_state(text: str) -> str:
    try:
        from sharedagents_goal.goal_service import strip_budget_state
    except Exception:  # noqa: BLE001 — never block compaction
        return text
    return strip_budget_state(text)


def _wrap_compact_summary(raw: str) -> str:
    """Wrap OpenCode-style Markdown compaction output for downstream injection."""
    body = _strip_budget_state(_strip_compact_fences(raw))
    if not body:
        return ""
    marker = "[conversation summary so far"
    if body.lower().startswith(marker):
        return body
    header = "[conversation summary so far — earlier turns were compacted; treat these as established context]\n\n"
    return header + body


def _format_compact_summary(parsed: dict | None, fallback_text: str) -> str:
    """Render a compaction memo into a model-readable block.

    OpenCode parity path: ``fallback_text`` is Markdown from the summarizer.
    Legacy JSON memos (tests / older summaries) still render when ``parsed`` is
    a dict.
    """
    if isinstance(parsed, dict):

        def _lines(key: str, label: str) -> list[str]:
            vals = parsed.get(key) or []
            if isinstance(vals, str):
                vals = [vals]
            out = [f"- {str(v).strip()}" for v in vals if str(v).strip()]
            return [f"{label}:", *out] if out else []

        task = str(parsed.get("task") or parsed.get("goal") or "").strip()
        nxt = str(parsed.get("next_step") or "").strip()
        sections: list[str] = [
            "[conversation summary so far — earlier turns were compacted; treat these as established context]",
        ]
        if task:
            sections.append(f"Task: {task}")
        sections += _lines("done", "Done")
        sections += _lines("progress", "Progress")
        sections += _lines("key_results", "Key results")
        sections += _lines("key_decisions", "Key decisions")
        sections += _lines("files_touched", "Files touched")
        sections += _lines("open", "Open / unresolved")
        if nxt:
            sections.append(f"Next step: {nxt}")
        return "\n".join(sections).strip()

    return _wrap_compact_summary(fallback_text)


def _msg_chars(messages: list[dict]) -> int:
    """Total UTF-8 char count over the JSON-serialized messages list."""
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)


def _user_field_chars(u) -> int:
    """Char count of a `turn.user` field. Multipart user content (an
    OpenAI-style content list, e.g. Telegram photo turns shipped as
    `[image_url, text]`) is JSON-serialized so a 192k-char base64 image
    counts as 192k, not as `len(list) == 2`. The latter mistake made
    `_pretrim_loaded_session_for_overflow` undercount image-heavy
    sessions by ~7×, leaving them stuck at 450k tokens against a 256k
    context (fingerprint: empty-summary on session 69f621f9…).
    """
    if isinstance(u, str):
        return len(u)
    if isinstance(u, (list, dict)):
        try:
            return len(json.dumps(u, ensure_ascii=False))
        except (TypeError, ValueError):
            return 0
    return 0


def _content_has_image(content) -> bool:
    """True if a user content payload is a multipart list with at least
    one `image_url` part. Used to decide whether to strip image bytes
    from older turns when replaying history.
    """
    if not isinstance(content, list):
        return False
    return any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)


def _strip_images_from_user_content(content):
    """Replace `image_url` parts in a multipart user content list with a
    short placeholder string. Strings pass through unchanged.

    Why: each saved turn keeps the original `user` payload on disk so
    we can replay the conversation byte-for-byte. For image turns that
    payload includes a `data:image/...;base64,…` URL whose body can
    easily be 100–500 KB. `_build_messages` replays every past turn on
    every subsequent turn, so without stripping a single image bloats
    the prompt by its full size FOREVER. Net effect: 5 image-bearing
    turns ⇒ ~1 MB of base64 sent on every reply, blowing past any
    reasonable context window.

    Strategy: keep text parts, count images, return a single-string
    summary like `"caption text\\n\\n[2 image(s) attached, stripped from
    earlier history]"`. `_build_messages` calls this on every past
    `turn.user`; the live new user message (passed as the `user`
    parameter) still carries the image so the model can describe it
    once, and the model's text description on `turn.content` provides
    full text-only continuity on every replay thereafter.
    """
    if not isinstance(content, list):
        return content
    text_parts: list[str] = []
    image_count = 0
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            t = part.get("text") or ""
            if t:
                text_parts.append(t)
        elif ptype == "image_url":
            image_count += 1
        # Unknown part types are dropped — conservative: an
        # unrecognised image-like part would re-bloat the history.
    text = "\n".join(text_parts)
    if image_count:
        suffix = f"[{image_count} image(s) attached, stripped from earlier history]"
        text = f"{text}\n\n{suffix}" if text else suffix
    return text or "[multipart message]"


def _hash_prefix(messages: list[dict], *, max_chars: int = 1024) -> str:
    """Short hex hash of the JSON-serialized message-list prefix.

    Used by the round logger to detect byte drift across rounds: if
    round N+1's hash differs from round N's, vLLM's prefix cache will
    miss everything from the divergence point forward. The hash covers
    the first `max_chars` characters of the concatenated prefix —
    enough to catch system-prompt drift, tool-schema drift, or
    raw_arguments reorderings without paying the cost of hashing the
    whole prompt.
    """
    import hashlib

    blob: list[str] = []
    written = 0
    for m in messages:
        s = json.dumps(m, ensure_ascii=False, sort_keys=False)
        blob.append(s)
        written += len(s)
        if written >= max_chars:
            break
    text = "".join(blob)[:max_chars]
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


def _hash_tools_payload(tools: list[dict]) -> str:
    """Hash of the tool-schemas list — independent from the messages
    hash because tools end up at a different spot in vLLM's chat
    template (typically before the conversation), so they cache
    separately. Lets us tell "tools drifted" apart from "messages
    drifted" when the cache-hit ratio drops.
    """
    import hashlib

    text = json.dumps(tools, ensure_ascii=False, sort_keys=False)
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


# Conservative chars-per-token floor. Empirically observed ratios:
#   - dense code/JSON: ~3.5–4.0 chars/token
#   - english prose:   ~4.0+ chars/token
#   - HTML / minified: ~2.5–3.0 chars/token  (the dangerous case — a
#     65 KiB web_fetch payload tokenized to ~25k tokens, ratio 2.6)
# Using 3.0 keeps us safe on prose/code without dramatically over-
# trimming HTML payloads. The proactive compaction guard combines this
# with `compact_at_frac` (default 1.0 — only compact when the request
# would actually overflow the upstream context); HTML-dense rounds
# that slip past the estimate are caught by the post-overflow reactive
# squeeze on a context-length 400.
_FALLBACK_CHARS_PER_TOKEN = 3.0


def _approx_tokens(messages: list[dict], chars_per_token: float = _FALLBACK_CHARS_PER_TOKEN) -> int:
    """Cheap token count for OpenAI-style messages.

    Authoritative counts from upstream `/usage.prompt_tokens` are
    preferred when available (see `session_turn`'s use of
    `last_metrics["prompt_tokens"]`). This estimator is the fallback
    for the very first round and the starting point for the learned
    per-session ratio.
    """
    return int(_msg_chars(messages) / max(0.5, chars_per_token))


# An inline ``image_url`` part is billed by the model as a bounded number
# of VISUAL tokens (Qwen-VL: ~256–1024; we use the upper bound), NOT the
# length of its base64 data URL. Counting the base64 bytes as text (a
# ~600KB image → ~200k "tokens") is what spuriously tripped the compaction
# / memory-flush machinery on agent 'main' 2026-06-22.
_IMAGE_PART_TOKEN_COST = 1024


def _approx_tokens_image_aware(messages: list[dict], chars_per_token: float = _FALLBACK_CHARS_PER_TOKEN) -> int:
    """Like :func:`_approx_tokens`, but charge each multimodal ``image_url``
    part a flat :data:`_IMAGE_PART_TOKEN_COST` instead of the (huge) char
    length of its base64 data URL.

    Used by the compaction gate's first-round fallback so an inline image
    can't masquerade as ~200k tokens of text before the authoritative
    upstream ``prompt_tokens`` is available. Text parts and non-multimodal
    messages are counted exactly as ``_approx_tokens`` does.
    """
    cpt = max(0.5, chars_per_token)
    toks = 0.0
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    toks += _IMAGE_PART_TOKEN_COST
                else:
                    try:
                        toks += len(json.dumps(part, ensure_ascii=False)) / cpt
                    except (TypeError, ValueError):
                        toks += len(str(part)) / cpt
            # small allowance for the role / wrapper framing the list omits
            toks += (len(str(m.get("role", ""))) + 16) / cpt
        else:
            try:
                toks += len(json.dumps(m, ensure_ascii=False)) / cpt
            except (TypeError, ValueError):
                toks += len(str(m)) / cpt
    return int(toks)


#: OpenCode's TOOL_OUTPUT_MAX_CHARS: each tool result is cut to this many
#: characters in the text handed to the summarizer, which keeps the summary
#: request small enough to fit even when the conversation just overflowed.
_OC_TOOL_OUTPUT_MAX_CHARS = 2_000


def _render_messages_as_transcript(
    messages: list[dict],
    *,
    max_chars: int = _COMPACT_MAX_TRANSCRIPT_CHARS,
) -> str:
    """Render a span of chat messages as readable transcript TEXT.

    The summarizer receives this as a single user-prompt blob rather than raw
    role-tagged messages (passing raw ``tool``-role messages
    into the summarizer call risks its own chat-shape 400s). When the rendered
    text exceeds ``max_chars`` the OLDEST content is dropped (least relevant to
    the running state) and a marker is prepended.
    """
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "?")
        content = m.get("content")
        text = content if isinstance(content, str) else ""
        if role == "assistant":
            calls = m.get("tool_calls") or []
            call_strs = []
            for c in calls:
                if not isinstance(c, dict):
                    continue
                fn = c.get("function") if isinstance(c.get("function"), dict) else {}
                nm = fn.get("name") or c.get("name") or "?"
                args = fn.get("arguments")
                if not isinstance(args, str):
                    try:
                        args = json.dumps(args, ensure_ascii=False)
                    except Exception:  # noqa: BLE001
                        args = str(args)
                call_strs.append(f"{nm}({(args or '')[:600]})")
            seg = "ASSISTANT: " + text.strip()
            if call_strs:
                seg += "\n  → tool_call: " + "; ".join(call_strs)
            parts.append(seg)
        elif role == "tool":
            nm = m.get("name") or m.get("tool_call_id") or "tool"
            body = text.strip()
            if len(body) > _OC_TOOL_OUTPUT_MAX_CHARS:  # OpenCode ``truncate``
                body = body[:_OC_TOOL_OUTPUT_MAX_CHARS] + "\n[truncated]"
            parts.append(f"TOOL_RESULT[{nm}]: {body}")
        elif role == "user":
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                stripped = _strip_images_from_user_content(content)
                text = stripped if isinstance(stripped, str) else str(stripped)
            else:
                text = str(content or "")
            parts.append("USER: " + text.strip())
        elif role == "system":
            # System content rarely belongs in a mid-task memo; keep a stub.
            parts.append("SYSTEM: " + text.strip()[:400])
        else:
            parts.append(f"{role.upper()}: {text.strip()}")
    transcript = "\n\n".join(p for p in parts if p.strip())
    if len(transcript) > max_chars:
        # Drop the OLDEST text; keep the most recent ``max_chars``.
        transcript = "[… older turns omitted to fit the summarizer window …]\n\n" + transcript[-max_chars:]
    return transcript


def _split_for_summary(
    messages: list[dict],
    target_tokens: int,
) -> tuple[dict | None, list[dict], list[dict]] | None:
    """Split ``messages`` into ``(system, head, tail)`` for LLM compaction.

    ``head`` is the older span to summarize; ``tail`` is the recent span kept
    verbatim. The tail is grown from the END until it holds ~half of
    ``target_tokens``, then its start is SNAPPED to a ROUND boundary — a
    ``user`` OR ``assistant`` message, never a ``tool`` result. This:

    * keeps the tail self-contained — a ``tool`` result is never orphaned from
      its preceding ``assistant.tool_calls`` (a vLLM 400); an assistant message
      carries its own tool_calls and the following tool results stay with it.

    CRITICAL: agentic turns have exactly ONE user message (the task) followed
    by many assistant/tool rounds — snapping to ``user`` only would never find
    an intermediate boundary and LLM compaction would never engage. Assistant
    messages are valid round boundaries, so we snap to either. The caller is
    responsible for supplying a user message in the rebuilt prompt (it inserts
    the memo as a user message when the tail starts at an assistant round).

    Returns ``None`` (→ caller falls back to mechanical compaction) when there
    is nothing useful to summarize: no system/body split is possible, no round
    boundary exists, or the head would be empty.
    """
    if not messages or target_tokens <= 0:
        return None
    sys_msg = messages[0] if messages[0].get("role") == "system" else None
    body = messages[1:] if sys_msg is not None else messages[:]
    if len(body) < 3:
        return None

    target_tail = max(1, target_tokens // 2)
    # Grow the tail from the end until it reaches ~half the target budget.
    acc = 0
    split = len(body)
    for i in range(len(body) - 1, -1, -1):
        acc += _approx_tokens([body[i]])
        split = i
        if acc >= target_tail:
            break

    def _is_boundary(msg: dict) -> bool:
        return msg.get("role") in ("user", "assistant")

    # Snap the tail start to a round boundary. Prefer the first boundary AT or
    # AFTER the provisional split (keeps the tail near target); else fall back
    # to the LAST boundary before the split (keeps more in the tail, still
    # valid). Never start the tail on a ``tool`` message (orphaned result).
    snap = None
    for i in range(split, len(body)):
        if _is_boundary(body[i]):
            snap = i
            break
    if snap is None:
        for i in range(min(split, len(body)) - 1, -1, -1):
            if _is_boundary(body[i]):
                snap = i
                break
    if snap is None or snap <= 0:
        # No boundary, or it's the very first body message (nothing older to
        # summarize). Let the caller use mechanical compaction.
        return None

    head = body[:snap]
    tail = body[snap:]
    if not head or not tail:
        return None
    return sys_msg, head, tail


def _tool_name_for_call(head: list[dict], tool_idx: int, tool_call_id: str) -> str:
    """Best-effort tool name lookup from the preceding assistant message."""
    for j in range(tool_idx - 1, -1, -1):
        a = head[j]
        if a.get("role") != "assistant":
            continue
        tcalls = a.get("tool_calls")
        if not isinstance(tcalls, list):
            continue
        for tc in tcalls:
            if not isinstance(tc, dict):
                continue
            if tc.get("id") == tool_call_id:
                fn = tc.get("function") or {}
                if isinstance(fn, dict):
                    return str(fn.get("name") or "").lower()
        break
    return ""


def _stage0_prune_tool_outputs(
    head: list[dict],
    *,
    prune_protect: int,
    prune_minimum: int,
    audit: dict,
) -> None:
    """OpenCode PRUNE_PROTECT: replace old tool bodies beyond protected tail."""
    candidates: list[tuple[int, str, int, dict]] = []
    for i, m in enumerate(head):
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if not isinstance(content, str) or content.startswith(_PRUNED_MARKER_PREFIX):
            continue
        tcid = str(m.get("tool_call_id") or "")
        tname = _tool_name_for_call(head, i, tcid)
        if tname in _PRUNE_NEVER_TOOLS:
            continue
        tokens = max(1, len(content) // 3)
        candidates.append((i, tcid, tokens, m))

    if not candidates:
        return

    protected_total = 0
    to_prune: list[tuple[int, str, int, dict]] = []
    for entry in reversed(candidates):
        _i, _tcid, tokens, _m = entry
        protected_total += tokens
        if protected_total <= prune_protect:
            continue
        to_prune.append(entry)

    pruned_tokens = sum(t for _, _, t, _ in to_prune)
    if pruned_tokens < prune_minimum:
        return

    for _i, tcid, _tokens, m in to_prune:
        old = m.get("content") or ""
        if isinstance(old, str) and not old.startswith(_PRUNED_MARKER_PREFIX):
            m["content"] = f"{_PRUNED_MARKER_PREFIX} {len(old)} chars of tool output removed to reclaim context]"
            if tcid:
                audit["pruned_tool_call_ids"].append(tcid)


# ── Manual /compact target math (opencode parity) ──────────────────────────
# The operator-facing ``compact_context`` direct-dispatch path (WebUI 🗜
# button / /compact → ``__SHAREDAGENTS_DELEGATE__:compact_context:{}``) ALWAYS
# performs a real trim, no matter how full the window is. opencode's
# /compact squeezes regardless of window size; the old window-fraction
# target (``ctx_window - output_budget - margin``) made a small context in a
# huge window a no-op (prod: 156K used of a 1M window → "~0 tokens removed").
#
# Target = HALF the current usage so the squeeze always bites, capped at the
# safe window (``ctx_window - output_budget - margin``) so an over-full
# context still lands inside the NEXT request's budget — the old emergency
# ceiling, preserved. A trivial context (below the floor, e.g. < 8K) has
# nothing worth reclaiming: the trim legitimately lands near zero, but the
# caller still produces + persists the summary and header stats (opencode
# writes its summary message too).
#
# The AUTO path is untouched: ``_maybe_compact`` / ``_compaction_target``
# stay threshold-gated on ``compact_at_frac × ctx_window``.
_MANUAL_COMPACT_TRIVIAL_FLOOR_TOKENS: int = 8_192
_MANUAL_COMPACT_MARGIN_TOKENS: int = 256


def _split_for_summary_oc(
    messages: list[dict],
    preserve_tokens: int,
) -> tuple[dict | None, list[dict], list[dict]] | None:
    """OpenCode ``select``: ``(system, head, tail)`` with a strict tail budget.

    Walk whole rounds from the newest backwards, keeping each while the running
    estimate stays within ``preserve_tokens``; the first round that does not fit
    ends the tail (OpenCode's ``splitTurn`` is a no-op here because a round is
    the smallest unit that keeps a tool result with its call). The tail may be
    EMPTY (everything summarized) when even the newest round exceeds the budget,
    which is what stops one 25K-token fetch result from riding along after every
    compaction. A round starts at a ``user`` or ``assistant`` message and takes
    the ``tool`` results that follow it. ``None`` when there is nothing to
    summarize.
    """
    if not messages or preserve_tokens <= 0:
        return None
    sys_msg = messages[0] if messages[0].get("role") == "system" else None
    body = messages[1:] if sys_msg is not None else messages[:]
    if len(body) < 2:
        return None
    starts = [i for i, m in enumerate(body) if m.get("role") in ("user", "assistant")]
    if not starts:
        return None
    bounds = starts + [len(body)]
    total = 0
    keep_from = len(body)
    for k in range(len(starts) - 1, -1, -1):
        size = _approx_tokens(body[bounds[k] : bounds[k + 1]])
        if total + size > preserve_tokens:
            break
        total += size
        keep_from = bounds[k]
    head, tail = body[:keep_from], body[keep_from:]
    if not head:
        return None
    return sys_msg, head, tail


def _manual_compact_target(
    before_tokens: int,
    ctx_window: int,
    output_budget: int,
    margin: int = _MANUAL_COMPACT_MARGIN_TOKENS,
) -> int:
    """Compaction target for a MANUAL /compact, sized off current usage.

    ``max(1024, ctx_window - output_budget - margin)`` is the safety budget:
    the largest prompt the next worker request can carry without overflow.
    The manual target never exceeds it, and for a trivial context
    (≤ ``_MANUAL_COMPACT_TRIVIAL_FLOOR_TOKENS``) it IS the safety budget —
    nothing meaningful to reclaim, but callers still record summary + stats.
    Returns 0 when the window is unknown (caller skips the mechanical pass).
    """
    if ctx_window <= 0:
        return 0
    safety_budget = max(1024, ctx_window - output_budget - margin)
    if before_tokens <= _MANUAL_COMPACT_TRIVIAL_FLOOR_TOKENS:
        return safety_budget
    return min(int(before_tokens * 0.5), safety_budget)


def compact_messages(
    messages: list[dict],
    target_tokens: int,
    *,
    prune_protect_tokens: int | None = None,
    prune_minimum_tokens: int | None = None,
    context_window: int = 0,
) -> tuple[list[dict], int, dict]:
    """Squeeze old tool results until estimated tokens ≤ target_tokens.

    Strategy (in order, oldest first):

    0. Token-based tool output pruning (OpenCode PRUNE_PROTECT): scan
       backward, keep the last ``PRUNE_PROTECT_TOKENS`` of tool output
       intact; replace older bodies with a tiny marker when ≥
       ``PRUNE_MINIMUM_TOKENS`` are reclaimable. When
       ``prune_protect_tokens`` is omitted, thresholds scale with
       ``context_window`` (see :func:`scaled_prune_protect`).
    1. Replace `role: tool` content > 800 chars with `<truncated>`
       (keep the first 400 chars + `\\n\\n... [N chars trimmed]`).
    2. If still over: drop the truncated `tool` messages entirely
       (oldest first) and the matching `assistant.tool_calls`.
    3. If still over: drop oldest droppable non-user/assistant head
       messages (every user/assistant turn is kept).

    Always preserves:
    - The `system` message at index 0.
    - The MOST RECENT user message (the live request).
    - The most recent assistant content message (the live answer
      assembling, if mid-loop).

    Returns `(new_messages, removed_tokens_estimate, audit)`. `audit` is:
        {
          "truncated_tool_call_ids": [str, ...],  # stage 1
          "dropped_tool_call_ids":   [str, ...],  # stage 2
          "dropped_turns":           int,         # stage 3
        }
    The TUI uses `audit` to surface, per-call_id, which transcript
    `ToolResultMessage` widgets had their content truncated or dropped
    on the wire even though the local widget still has the original.
    """
    audit = {
        "truncated_tool_call_ids": [],
        "dropped_tool_call_ids": [],
        "dropped_turns": 0,
        "pruned_tool_call_ids": [],
    }
    if not messages:
        return messages, 0, audit

    before = _approx_tokens(messages)
    if before <= target_tokens:
        return messages, 0, audit

    sys_msg = messages[0] if messages and messages[0].get("role") == "system" else None
    body = messages[1:] if sys_msg is not None else messages[:]

    # Anchor: don't touch the last 2 messages (the user prompt that
    # triggered this round + any assistant we're mid-replying to).
    keep_tail = 2
    if len(body) <= keep_tail:
        return messages, 0, audit

    head = body[:-keep_tail]
    tail = body[-keep_tail:]

    # CRITICAL: vLLM's chat template rejects with HTTP 400 "No user query
    # found in messages." if the prompt has no `user` role anywhere.
    # `keep_tail=2` only protects the tail by position — when the tail
    # is `[assistant_tool_calls, tool_result]` (common after a tool
    # round), the user message lives in `head` and stage 3 will happily
    # pop it. Identify the most-recent user message by reference and
    # protect it through every stage. Bug surfaced as fingerprint
    # `4aea829f` after a `compact_context` tool call mid-turn.
    protected_user: dict | None = None
    for m in reversed(body):
        if m.get("role") == "user":
            protected_user = m
            break

    # Also collect tool_call_ids that live in the kept TAIL window so
    # stage 3 doesn't drop the assistant message that issued them —
    # leaving a tool message with no preceding `assistant.tool_calls`
    # is a separate vLLM 400 ("tool message without preceding assistant
    # tool_call"). We grow this set as stage 2 mutates the head, but
    # the *initial* set captures every tool_call_id referenced by tool
    # messages anywhere in the kept tail.
    tail_tool_call_ids: set[str] = set()
    for m in tail:
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if isinstance(tcid, str) and tcid:
                tail_tool_call_ids.add(tcid)

    # Stage 0: OpenCode PRUNE_PROTECT token-based tool output pruning.
    protect = prune_protect_tokens if prune_protect_tokens is not None else scaled_prune_protect(context_window)
    minimum = prune_minimum_tokens if prune_minimum_tokens is not None else scaled_prune_minimum(context_window)
    _stage0_prune_tool_outputs(
        head,
        prune_protect=protect,
        prune_minimum=minimum,
        audit=audit,
    )
    rebuilt = ([sys_msg] if sys_msg else []) + head + tail
    if _approx_tokens(rebuilt) <= target_tokens:
        return rebuilt, before - _approx_tokens(rebuilt), audit

    # Stage 1: truncate large tool results.
    from sharedagents_core.injection_guard import reseal_fence

    for m in head:
        if m.get("role") != "tool":
            continue
        c = m.get("content")
        if isinstance(c, str) and len(c) > 800:
            trimmed = len(c) - 400
            # reseal_fence: tool-result content may be an injection-guard
            # untrusted-data envelope; slicing to [:400] severs the closing
            # fence. Re-close it so the truncated body stays fenced as data
            # (fail-safe) instead of floating as instructions (fail-open).
            m["content"] = reseal_fence(c[:400] + f"\n\n<truncated> ... [{trimmed} chars trimmed]")
            tcid = m.get("tool_call_id")
            if tcid:
                audit["truncated_tool_call_ids"].append(tcid)
    rebuilt = ([sys_msg] if sys_msg else []) + head + tail
    if _approx_tokens(rebuilt) <= target_tokens:
        return rebuilt, before - _approx_tokens(rebuilt), audit

    # Stage 2: drop tool messages entirely, oldest first. Also drop
    # the matching tool_call entries from the preceding assistant
    # message so the tool_call_id refs don't go dangling.
    while head:
        # Find oldest tool message; drop it. Then walk backward to
        # find its assistant.tool_calls and prune the matching entry.
        idx = next(
            (i for i, m in enumerate(head) if m.get("role") == "tool"),
            None,
        )
        if idx is None:
            break
        tool_call_id = head[idx].get("tool_call_id")
        head.pop(idx)
        if tool_call_id:
            audit["dropped_tool_call_ids"].append(tool_call_id)
            # Promote: it's no longer "truncated", it's gone.
            try:
                audit["truncated_tool_call_ids"].remove(tool_call_id)
            except ValueError:
                pass
        # Strip the corresponding tool_calls[] entry from any earlier
        # assistant message; if that leaves it empty, remove the
        # tool_calls key (and drop the whole assistant message too when
        # it has no content) — UNLESS it would also drop the protected
        # user (assistants can't be users, but this guards the index
        # math from underflow).
        for j in range(idx - 1, -1, -1):
            a = head[j]
            if a.get("role") == "assistant" and isinstance(a.get("tool_calls"), list):
                a["tool_calls"] = [tc for tc in a["tool_calls"] if tc.get("id") != tool_call_id]
                if not a["tool_calls"]:
                    # An empty ``tool_calls`` list is invalid on the wire:
                    # OpenAI-compatible backends reject it with HTTP 400
                    # "Invalid 'messages[N].tool_calls': empty array".
                    # Drop the key entirely; if the
                    # assistant also carries no content, drop the whole
                    # message (the original behaviour).
                    a.pop("tool_calls", None)
                    if not (a.get("content") or "").strip():
                        head.pop(j)
                break
        rebuilt = ([sys_msg] if sys_msg else []) + head + tail
        if _approx_tokens(rebuilt) <= target_tokens:
            return rebuilt, before - _approx_tokens(rebuilt), audit

    # Stage 3: drop oldest user/assistant turns until fits — but skip
    # the protected user AND any assistant whose tool_calls reference a
    # tool_call_id still kept in the tail. Dropping such an assistant
    # would dangle the surviving tool message → vLLM 400 "tool message
    # without preceding assistant tool_call". If the only messages
    # left in head are protected, exit; sending a prompt that's still
    # slightly over the soft target just makes the runtime's later
    # max_tokens rescale work harder (or triggers the post-overflow
    # reactive squeeze, which is recoverable).
    def _is_load_bearing_assistant(m: dict) -> bool:
        """True when dropping this assistant message would orphan a tool
        result still kept in the protected tail.
        """
        if m.get("role") != "assistant":
            return False
        tcalls = m.get("tool_calls")
        if not isinstance(tcalls, list):
            return False
        return any(isinstance(tc, dict) and tc.get("id") in tail_tool_call_ids for tc in tcalls)

    while head:
        # Find the oldest message that is NOT the protected user and
        # NOT a load-bearing assistant.
        drop_idx = None
        for i, m in enumerate(head):
            # Compaction never drops or truncates user- or
            # assistant-role content under any circumstance. Only oversized
            # tool results are shed (stages 1/2, and the stage-4 last resort
            # that truncates a protected-tail tool body). Stage 3 therefore
            # skips every user/assistant message; when only such messages
            # remain the pass exits still-over-target and the runtime's later
            # max_tokens rescale absorbs the small overage.
            if m.get("role") in ("user", "assistant"):
                continue
            if m is protected_user:
                continue
            if _is_load_bearing_assistant(m):
                continue
            drop_idx = i
            break
        if drop_idx is None:
            # head only contains protected messages; can't compact
            # further without breaking the upstream contract.
            break
        head.pop(drop_idx)
        audit["dropped_turns"] += 1
        rebuilt = ([sys_msg] if sys_msg else []) + head + tail
        if _approx_tokens(rebuilt) <= target_tokens:
            return rebuilt, before - _approx_tokens(rebuilt), audit

    # Stage 4 (last resort): if we're STILL over target after dropping every
    # droppable head turn, the remaining weight lives in PROTECTED messages —
    # in practice an oversized `tool` result in the kept tail (a large
    # web_fetch / Read in the LIVE turn). Stages 1-3 only ever touch the
    # head, so a giant tool result in the tail kept the input floor pinned
    # above the window no matter how many times we compacted. That is the
    # exact production deadlock (qwen3.6-27b @ 65536: input stuck ~50178,
    # the reactive max_tokens fallback oscillating 14334↔15359 until
    # FALLBACK_EXHAUSTED → the raw HTTP 400 the operator saw, including on a
    # `/compact` recovery attempt). Truncating a tool message's CONTENT is
    # structurally safe — it dangles no tool_call_id and drops no user query
    # — so do it here, oldest-first, until we fit. NEVER touch user/assistant
    # content (that would corrupt the live request / answer).
    rebuilt = ([sys_msg] if sys_msg else []) + head + tail
    if _approx_tokens(rebuilt) > target_tokens:
        from sharedagents_core.injection_guard import reseal_fence

        for m in rebuilt:
            if _approx_tokens(rebuilt) <= target_tokens:
                break
            if m.get("role") != "tool":
                continue
            c = m.get("content")
            if isinstance(c, str) and len(c) > 800:
                trimmed = len(c) - 400
                m["content"] = reseal_fence(c[:400] + f"\n\n<truncated> ... [{trimmed} chars trimmed]")
                tcid = m.get("tool_call_id")
                if isinstance(tcid, str) and tcid and tcid not in audit["truncated_tool_call_ids"]:
                    audit["truncated_tool_call_ids"].append(tcid)

    rebuilt = ([sys_msg] if sys_msg else []) + head + tail
    return rebuilt, before - _approx_tokens(rebuilt), audit


def compact_session_data_in_place(
    data: dict,
    *,
    keep_recent: int = 1,
    min_chars: int = 800,
    head_chars: int = 400,
) -> tuple[int, list[str]]:
    """Apply deterministic truncation to oversized tool-call results
    stored on a session dict, in place.

    For every `tool_rounds[].calls[].result` longer than `min_chars`
    in the compactable slice, replace it with the first `head_chars`
    chars + a `[compacted: N chars trimmed]` marker. Preserves the most
    recent `keep_recent` turns intact so the active reasoning context
    keeps full fidelity.

    Returns `(removed_chars, affected_call_ids)`. The latter is the
    list of `tool_call_id`s whose results were trimmed — same shape as
    `compact_messages`'s `audit["truncated_tool_call_ids"]` so callers
    can surface the same TUI badging.

    Single source of truth for two callers:
    - `POST /v1/sessions/{sid}/compact` (between-turn manual via the
      TUI's `/compact` slash command).
    - `_pretrim_loaded_session_for_overflow` (load-time auto-trim
      when a stored session would already overflow upstream's
      context window — see that method's docstring for why).
    """
    turns = data.get("turns") or []
    compactable = turns[:-keep_recent] if keep_recent and len(turns) > keep_recent else list(turns)
    from sharedagents_core.injection_guard import reseal_fence

    affected_ids: list[str] = []
    removed_chars = 0
    for turn in compactable:
        for r in turn.get("tool_rounds") or []:
            for c in r.get("calls") or []:
                res = c.get("result")
                if not isinstance(res, str) or len(res) <= min_chars:
                    continue
                trimmed = len(res) - head_chars
                # reseal_fence: keep a possibly-fenced result well-formed
                # after truncation (see Stage-1 note in _compact_messages).
                c["result"] = reseal_fence(res[:head_chars] + f"\n\n... [compacted: {trimmed} chars trimmed]")
                removed_chars += trimmed
                cid = c.get("id")
                if cid:
                    affected_ids.append(cid)
    return removed_chars, affected_ids


def compact_session_turns_in_place(
    session: Any,
    *,
    keep_recent: int = 1,
    min_chars: int = 800,
    head_chars: int = 400,
) -> tuple[int, list[str]]:
    """Durably truncate oversized tool-result bodies on a LIVE Session's
    in-memory turns (the ``tool_results[].content`` the wire rebuild
    replays via ``_replay_tool_exchange``).

    Sibling to :func:`compact_session_data_in_place`, which targets the
    on-disk ``tool_rounds[].calls[].result`` schema; this one targets the
    in-memory :class:`sharedagents_session.session.Session` schema
    (``turns[].tool_results[].content``). Used by the ``compact_context``
    direct-dispatch path (``/compact``) so an operator-initiated compaction
    PERSISTS instead of only squeezing one ephemeral round.

    Best-effort: the in-memory mirror often stores only tool-result
    metadata (no ``content`` body — it's gated by
    ``persist_tool_result_bodies`` / lives on disk), in which case those
    turns are a safe no-op. Preserves the most-recent ``keep_recent`` turns
    intact so the active reasoning context keeps full fidelity.

    Returns ``(removed_chars, affected_call_ids)``.
    """
    turns = list(getattr(session, "turns", None) or [])
    if not turns:
        return 0, []
    compactable = turns[:-keep_recent] if keep_recent and len(turns) > keep_recent else turns
    from sharedagents_core.injection_guard import reseal_fence

    affected: list[str] = []
    removed_chars = 0
    for turn in compactable:
        if not isinstance(turn, dict):
            continue
        for r in turn.get("tool_results") or []:
            if not isinstance(r, dict):
                continue
            c = r.get("content")
            if not isinstance(c, str) or len(c) <= min_chars:
                continue
            trimmed = len(c) - head_chars
            # reseal_fence: keep a possibly-fenced result well-formed after
            # truncation (see Stage-1 note in compact_messages).
            r["content"] = reseal_fence(c[:head_chars] + f"\n\n... [compacted: {trimmed} chars trimmed]")
            removed_chars += trimmed
            cid = r.get("call_id") or r.get("id")
            if cid:
                affected.append(str(cid))
    return removed_chars, affected


# ── OpenCode prune() parity ────────────────────────────────────────────────
# Exact constants copied from OpenCode
# (packages/opencode/src/session/compaction.ts): the durable "erase old tool
# outputs to reclaim context WITHOUT summarization" pass.
_OC_TOOL_OUTPUT_MAX_CHARS: int = 2_000  # TOOL_OUTPUT_MAX_CHARS
_OC_PRUNE_MINIMUM: int = 20_000  # PRUNE_MINIMUM — min tokens freed to commit
_OC_PRUNE_PROTECT: int = 40_000  # PRUNE_PROTECT — recent tool tokens kept intact
_OC_PRUNE_TAIL_TURNS: int = _PRUNE_TAIL_TURNS  # skip the last N turns (DEFAULT_TAIL_TURNS)
_OC_PRUNE_PROTECTED_TOOLS: frozenset = frozenset({"skill"})  # PRUNE_PROTECTED_TOOLS
# OpenCode fully CLEARS a pruned tool output; its model-message conversion emits
# this exact placeholder in place of the body. We store it directly on the
# folded tool_result so the wire rebuild replays the same cleared text (and a
# later prune stops when it sees it).
_OC_CLEARED_PLACEHOLDER: str = "[Old tool result content cleared]"


def _oc_estimate_tokens(text: str) -> int:
    """OpenCode's ``Token.estimate`` — ceil(chars / 3). Used by prune() to size
    tool outputs against the PRUNE_PROTECT / PRUNE_MINIMUM token budgets so the
    thresholds land where OpenCode's do (sharedagents's ``_approx_tokens`` expects a
    message list, not a raw string, and would over-count ~3×)."""
    return -(-len(text) // 3) if text else 0


def prune_session_tool_outputs(
    session: Any,
    *,
    context_window: int = 0,
) -> tuple[int, list[str]]:
    """Port of OpenCode's ``SessionCompaction.prune`` — durably erase OLD
    tool-call outputs to reclaim context WITHOUT summarizing.

    Faithful to OpenCode's algorithm (compaction.ts):
    - Walk turns newest→oldest, SKIPPING the last ``_PRUNE_TAIL_TURNS`` (2) turns.
    - Stop scan if a turn carries an assistant summary or an already-cleared output.
    - Keep the most-recent protect TOKENS of tool output intact
      (accumulate an estimate as we walk back); protected tools
      (``_OC_PRUNE_PROTECTED_TOOLS``) never count against the budget and are
      never pruned. Protect/minimum scale with ``context_window`` (or
      ``session.context_window``) via :func:`scaled_prune_protect`.
    - Every older tool result past the protect budget is replaced with
      ``_OC_CLEARED_PLACEHOLDER``.
    - Only COMMIT the prune if it frees strictly more than minimum tokens;
      otherwise it is a no-op (OpenCode does not commit a tiny prune).

    Operates on the in-memory ``session.turns[].tool_results[].content`` mirror
    (sharedagents's schema); a body-less mirror is a safe no-op. Returns
    ``(removed_chars, affected_call_ids)``.
    """
    turns = list(getattr(session, "turns", None) or [])
    tail_turns = _PRUNE_TAIL_TURNS
    if len(turns) <= tail_turns:
        return 0, []
    walkable = turns[: len(turns) - tail_turns]

    ctx = int(context_window or getattr(session, "context_window", 0) or 0)
    protect = scaled_prune_protect(ctx) if ctx > 0 else _OC_PRUNE_PROTECT
    minimum = scaled_prune_minimum(ctx) if ctx > 0 else _OC_PRUNE_MINIMUM

    # Dry-run pass (newest→oldest) mirroring OpenCode's prune() exactly:
    # accumulate each completed tool output's token estimate FIRST, keep it while
    # the running total stays within PRUNE_PROTECT, and prune (fully CLEAR) the
    # first output that pushes the total past PRUNE_PROTECT and every older one.
    # Protected tools never count and are never pruned. Stop the whole scan at an
    # already-cleared output or a summary turn (everything older is already compacted).
    total = 0
    plan: list[dict] = []
    freed_tokens = 0
    stop = False
    for turn in reversed(walkable):
        if stop:
            break
        if not isinstance(turn, dict):
            continue
        # OpenCode parity: stop scan if a turn carries an assistant summary,
        # compaction event, or manual/delegate compaction sentinel
        if (
            turn.get("compaction_events")
            or str(turn.get("user") or "").startswith("__SHAREDAGENTS_DELEGATE__:compact_context:")
            or str(turn.get("compaction_summary") or "").strip()
            or turn.get("summary")
        ):
            break
        # OpenCode exempts by tool NAME; the folded tool_results carry only
        # call_id, so resolve the name from this turn's tool_calls by call_id.
        names: dict[str, str] = {}
        for tc in turn.get("tool_calls") or []:
            if isinstance(tc, dict):
                names[str(tc.get("call_id") or tc.get("id") or "")] = str(tc.get("name") or "")
        for r in reversed(turn.get("tool_results") or []):
            if not isinstance(r, dict):
                continue
            c = r.get("content")
            if not isinstance(c, str):
                continue
            if c == _OC_CLEARED_PLACEHOLDER:
                stop = True  # already compacted → OpenCode stops scanning
                break
            if r.get("ok") is not True:
                continue  # OpenCode prunes only successfully-COMPLETED parts
            cid = str(r.get("call_id") or r.get("id") or "")
            name = str(r.get("name") or names.get(cid, ""))
            if name in _OC_PRUNE_PROTECTED_TOOLS:
                continue  # never counts, never pruned
            tok = _oc_estimate_tokens(c)
            total += tok  # OpenCode: add THEN test
            if total <= protect:
                continue  # still within the protected recent-token budget
            plan.append(r)
            freed_tokens += tok  # fully cleared → full estimate freed

    # OpenCode commits only when it frees STRICTLY MORE than PRUNE_MINIMUM.
    if freed_tokens <= minimum:
        return 0, []

    removed_chars = 0
    affected: list[str] = []
    for r in plan:
        removed_chars += len(r["content"])  # fully cleared, like OpenCode
        r["content"] = _OC_CLEARED_PLACEHOLDER
        cid = r.get("call_id") or r.get("id")
        if cid:
            affected.append(str(cid))
    return removed_chars, affected


PRUNE_CACHE_WARNING = (
    "compact_prune_on_load is enabled: erasing old tool results outside a "
    "compaction rewrites earlier prompt messages, which invalidates the "
    "upstream prefix cache from that point and makes the next call re-prefill "
    "the history (slower time to first token). Leave it off; tool results are "
    "pruned and summarized when the session compacts."
)
_prune_cache_warned = False


def prune_session_on_load(
    session: Any,
    *,
    enabled: bool = False,
) -> tuple[int, list[str]]:
    """Load-time prune for a session folded back from disk (OpenCode
    ``compaction.prune``), OFF by default.

    Tool results are normally erased only when the session compacts
    (``_maybe_compact`` / the summary path), the one point where the prompt
    prefix is rebuilt anyway. Pruning on load changes earlier messages between
    two requests and costs the prefix cache; ``enabled`` is the operator's
    explicit opt-in (``compact_prune_on_load``) and logs
    :data:`PRUNE_CACHE_WARNING` once per process.
    """
    global _prune_cache_warned
    if not enabled:
        return 0, []
    if not _prune_cache_warned:
        _prune_cache_warned = True
        logging.getLogger("sharedagents.agent.compaction").warning(PRUNE_CACHE_WARNING)
    return prune_session_tool_outputs(session)


def _format_compaction_summary(removed: int, audit: dict, new_prompt_tokens: int) -> str:
    """Human-readable one-liner for a compaction tool_result body.

    Used both by the model-callable `compact_context` tool and by the
    `POST /v1/sessions/{sid}/compact` endpoint so the audit reads the
    same way wherever it surfaces.
    """
    n_trunc = len(audit.get("truncated_tool_call_ids") or [])
    n_drop = len(audit.get("dropped_tool_call_ids") or [])
    n_turns = int(audit.get("dropped_turns") or 0)
    bits: list[str] = [f"~{removed} tokens removed", f"prompt now ~{new_prompt_tokens} tokens"]
    if n_trunc:
        bits.append(f"{n_trunc} tool result(s) truncated")
    if n_drop:
        bits.append(f"{n_drop} tool result(s) dropped")
    if n_turns:
        bits.append(f"{n_turns} old turn(s) dropped")
    return "compacted: " + " · ".join(bits)
