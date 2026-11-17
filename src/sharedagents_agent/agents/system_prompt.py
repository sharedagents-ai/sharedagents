"""System-prompt + per-round preamble composition for :class:`Agent`.

The on-disk section-composition half of the Agent: builds the system
prompt from the agent's home directory (SOUL/AGENTS/TOOLS/USER/MEMORY/
MODE-*.md via ``turn_runtime._compose_system_prompt``) and the
below-cache-boundary preamble blocks (pinned facts, rolling compaction
summary, current todos, skill auto-invoke, memory recall) for one round.

Split out of ``agents/agent.py`` as a mixin so the composition root keeps
only turn orchestration. Methods resolve against the ``Agent`` instance
via normal MRO — cross-mixin calls (``self.memory_recall_context``,
``self._load_hook_engine``, ``self._run_hook_event``,
``self._append_memory_recall_async``) are satisfied by the sibling mixins
also mixed into ``Agent``.

``__slots__ = ()`` is REQUIRED: ``Agent`` is ``@dataclass(slots=True)`` and
relies on undeclared-attribute assignment raising ``AttributeError``. A
mixin base without ``__slots__`` would re-introduce ``__dict__`` and break
that guarantee.
"""

from __future__ import annotations

from typing import Any


class SystemPromptMixin:
    """On-disk system-prompt + per-round preamble composition."""

    __slots__ = ()

    def compose_system_prompt(
        self,
        *,
        mode: str | None = None,
        workspace_cwd: str = "",
        workspace_origin: str = "",
        skip_memory: bool = False,
    ) -> str:
        """Build the system prompt from the agent's home directory.

        Forwards to ``turn_runtime._compose_system_prompt`` (the
        single source of truth — reading SOUL.md / AGENTS.md /
        TOOLS.md / USER.md / MEMORY.md / MODE-*.md, plus the
        skills + MCP + deferred-tools catalogs, with the cache
        boundary inserted before the dynamic ``# WORKSPACE-CONTEXT``
        tail).
        """
        from sharedagents_agent.turn_runtime import _compose_system_prompt

        return _compose_system_prompt(
            self.cfg,
            mode or self.permission_mode,
            workspace_cwd=workspace_cwd,
            workspace_origin=workspace_origin,
            skip_memory=skip_memory,
        )

    def compose_round_preamble(
        self,
        session: Any,
        user_text: str,
        *,
        suppressed_blocks: set[str] | None = None,
        block_overrides: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the below-cache-boundary preamble blocks for one round.

        **Single owner of dynamic-context composition.** Neither :class:`LocalLLMEngine` nor any
        :class:`ExternalCliEngine` reaches into the agent's .md files
        or the memory provider on its own — they consume what this
        method returns via :attr:`TurnRequest.preamble`.

        Returns the list of (role, content) pairs that go AFTER the
        system prompt but BEFORE the live transcript:

        * Pinned facts block (today's ``_inject_pinned_facts``).
        * Rolling compaction summary (today's
          ``_inject_compaction_summary``).
        * Current-todos block (today's ``_inject_current_todos``).
        * Skill auto-invoke preamble (when
          ``cfg.skill_auto_invoke.enabled`` — ``SkillIndex.search``
          + ``render_preamble``).
        * Memory recall block (when
          ``cfg.memory_recall_mode in {"context", "hybrid"}`` —
          ``memory_manager.prefetch_for_query``).

        Per-engine semantics:

        * Local LLM: this method is invoked once per turn; the engine
          re-splices the same preamble blocks every round within the
          turn (idempotent via stable marker comments).
        * CLI engines: invoked ONCE on the first call of a session;
          the Agent folds the result into the composed system prompt
          sent via ``--append-system-prompt`` (or first-message
          injection). NOT re-invoked on resume — the CLI session has
          a snapshot. If the operator edits a SOUL/AGENTS/TOOLS file
          mid-session, the change won't reach CLI engines until the
          next first-turn (sha mismatch detection via
          :meth:`Engine.system_prompt_policy`).
        """
        # Reuses the legacy injection helpers — calling each in
        # sequence against an empty message list collects the
        # pinned-facts / rolling-summary / current-todos blocks
        # without duplicating the rendering logic. Skills / memory
        # recall integrations are Phase 4 follow-ups; for now we wire
        # only the three high-traffic injections that fire every
        # round today.
        from sharedagents_agent.turn_runtime import (
            _inject_compaction_summary,
            _inject_current_todos,
            _inject_pinned_facts,
        )

        out: list[dict[str, Any]] = []
        # ``suppressed_blocks`` lets a ``PreambleInject`` hook (fired
        # by ``compose_round_preamble_async`` before calling this
        # sync method) opt out of specific built-in injectors;
        # ``block_overrides`` lets a hook substitute the body text
        # (handed in as ``{"kind": "<custom body>"}``).
        suppressed_blocks = suppressed_blocks or set()
        block_overrides = block_overrides or {}

        # Preamble blocks are ALL ``role="user"`` (below the cache
        # boundary, after the single system prompt). A ``role="system"``
        # block here lands after the user-role pinned/summary/todos
        # blocks, and strict chat templates (strict local Qwen/vLLM backends)
        # reject any system message that isn't first with HTTP 400
        # "System message must be at the beginning" — killing the turn.
        # The canonical injectors already emit
        # user-role; the hook-override + recall + skill paths below match
        # that so no preamble block is ever system-role.
        pinned = list(getattr(session, "pinned_facts", None) or [])
        if pinned and "pinned_facts" not in suppressed_blocks:
            if "pinned_facts" in block_overrides:
                # Hook supplied a substitute body — render as a single
                # user message (matches the canonical multi-line block).
                out.append(
                    {
                        "role": "user",
                        "content": block_overrides["pinned_facts"],
                    }
                )
            else:
                out = _inject_pinned_facts(out, [str(x) for x in pinned])

        # Realtime voice_mode turns only (turn-scoped; never durable). Typed
        # Jarvis on the same session must not see this block.
        try:
            from sharedagents_agent.voice_tts import (
                is_voice_tts_turn,
                voice_tts_preamble_block,
            )

            if is_voice_tts_turn(getattr(session, "hash", None)):
                out.append(voice_tts_preamble_block())
        except Exception:  # noqa: BLE001 — never break preamble composition
            pass

        summary = str(getattr(session, "compaction_summary", "") or "")
        if summary.strip() and "compaction_summary" not in suppressed_blocks:
            if "compaction_summary" in block_overrides:
                out.append(
                    {
                        "role": "user",
                        "content": block_overrides["compaction_summary"],
                    }
                )
            else:
                out = _inject_compaction_summary(out, summary)

        # Todos — needs the session id + the agent's sessions_dir to
        # locate <sessions_dir>/<sid>.todos.json. Fail-soft on
        # missing/corrupt todos: caller still gets a valid preamble.
        if "current_todos" not in suppressed_blocks:
            try:
                from sharedagents_agent.tools.impl.planning import (
                    load_todos,
                    todos_path_for_session,
                )

                sid = getattr(session, "hash", None)
                if isinstance(sid, str) and sid:
                    todos = load_todos(todos_path_for_session(self.sessions_dir, sid))
                    if "current_todos" in block_overrides:
                        out.append(
                            {
                                "role": "user",
                                "content": block_overrides["current_todos"],
                            }
                        )
                    else:
                        out = _inject_current_todos(out, todos)
            except (OSError, ValueError, ImportError):
                pass

        # Skill auto-invoke preamble is appended by
        # ``compose_round_preamble_async`` (see ``_skill_auto_invoke_block``)
        # — it MUST be ``await``ed because ``SkillIndex.search`` is a
        # coroutine, so it cannot run from this sync method. The previous
        # inline version here called ``SkillIndex(self.cfg)`` (wrong ctor —
        # it takes ``(embedder, agent_home)``) and used the un-awaited
        # coroutine, which ``render_preamble`` then threw on and a bare
        # ``except`` swallowed: the feature never produced a preamble even
        # when enabled. Run-turn always goes through the async path
        # (``compose_round_preamble_async``), so no engine loses it.

        # Memory recall preamble — when
        # ``cfg.memory_recall_mode in {context, hybrid}``, prefetch the
        # memory-manager's recall block for the current user prompt.
        # Same migration story as skill auto-invoke: used to live in
        # the legacy turn_runtime; now runs here uniformly.
        try:
            recall = self.memory_recall_context(session, user_text)
        except Exception:  # noqa: BLE001
            recall = ""
        if recall:
            # user-role (see preamble-role note above): a system message
            # after the user-role blocks breaks strict chat templates.
            out.append({"role": "user", "content": recall})

        return out

    async def compose_round_preamble_async(
        self,
        session: Any,
        user_text: str,
    ) -> list[dict[str, Any]]:
        """Async variant — fires ``PreambleInject`` hooks per built-in
        injector before composing the preamble, so an ECC plugin can
        suppress / substitute pinned-facts, compaction-summary,
        current-todos blocks instead of only wrapping them.

        Hook semantics per block:
        * ``decision: block`` (exit 2) → suppress the block entirely.
        * ``decision: block`` + ``reason: <body>`` → substitute the
          hook's body for the canonical injector output.
        * ``hookSpecificOutput.replacementBody`` → same as the reason
          form via the JSON control protocol.
        * No-op → original built-in injector runs.

        Returns the same shape as :meth:`compose_round_preamble`.
        """
        engine = self._load_hook_engine()
        if engine is None or not engine.has_event("PreambleInject"):
            out = self.compose_round_preamble(session, user_text)
            await self._append_skill_auto_invoke(out, user_text)
            await self._append_memory_recall_async(out, session, user_text)
            return out

        # Probe each injector: skip the hook entirely when there's
        # nothing to render (no pinned facts, no summary, no todos).
        candidates: list[tuple[str, str]] = []
        if list(getattr(session, "pinned_facts", None) or []):
            candidates.append(("pinned_facts", "pinned facts"))
        if str(getattr(session, "compaction_summary", "") or "").strip():
            candidates.append(("compaction_summary", "compaction summary"))
        # current_todos always considered — load_todos is the truth.
        candidates.append(("current_todos", "current todos"))

        suppressed: set[str] = set()
        overrides: dict[str, str] = {}
        for kind, description in candidates:
            outcome = await self._run_hook_event(
                "PreambleInject",
                session,
                {
                    "kind": kind,
                    "description": description,
                    "user_text_chars": len(user_text or ""),
                },
            )
            if outcome.blocked:
                if outcome.block_reason:
                    overrides[kind] = outcome.block_reason
                else:
                    suppressed.add(kind)
            # Honour replacementBody from JSON control protocol.
            for r in getattr(outcome, "results", []) or []:
                j = getattr(r, "json_output", None)
                if not isinstance(j, dict):
                    continue
                hso = j.get("hookSpecificOutput")
                if not isinstance(hso, dict):
                    continue
                rb = hso.get("replacementBody")
                if isinstance(rb, str) and rb.strip():
                    overrides[kind] = rb
                    suppressed.discard(kind)
                    break

        out = self.compose_round_preamble(
            session,
            user_text,
            suppressed_blocks=suppressed,
            block_overrides=overrides,
        )
        await self._append_skill_auto_invoke(out, user_text)
        await self._append_memory_recall_async(out, session, user_text)
        return out

    async def _append_skill_auto_invoke(
        self,
        out: list[dict[str, Any]],
        user_text: str,
    ) -> None:
        """Append the embedding-backed skill auto-invoke preamble.

        No-op unless ``cfg.skill_auto_invoke.enabled`` is set AND an
        embedder is configured. Disabled by default — the OpenCode-style
        inline skill catalog + ``Skill`` tool replace auto-injection.
        Embeds the live user prompt against the per-agent
        :class:`SkillIndex` (cached on ``self._skill_index`` so we don't
        re-embed the skill corpus every turn) and appends any skills
        whose description scores above threshold. Fail-soft: any
        error (embedder down, no skills) yields no preamble — never
        raises into the turn loop.

        This is the awaited home of the auto-invoke that ``SkillIndex``
        was always meant to back; the old inline version in
        :meth:`compose_round_preamble` was dead code (un-awaited
        coroutine, wrong constructor).
        """
        sai_cfg = getattr(self.cfg, "skill_auto_invoke", None) or {}
        if isinstance(sai_cfg, dict):
            enabled = bool(sai_cfg.get("enabled"))
            top_k = int(sai_cfg.get("top_k") or 3)
            threshold = float(sai_cfg.get("threshold", 0.6))
        else:
            enabled = bool(getattr(sai_cfg, "enabled", False))
            top_k = int(getattr(sai_cfg, "top_k", 0) or 3)
            threshold = float(getattr(sai_cfg, "threshold", 0.6))
        if not enabled or not (user_text or "").strip():
            return
        try:
            from sharedagents_skill.skill_index import SkillIndex, render_preamble

            idx = getattr(self, "_skill_index", None)
            if idx is None:
                # Embed the skill corpus over the memory wire
                # (WireEmbedder proxy), never the in-process engine. None → no
                # embedder configured → SkillIndex.is_available False (fail-safe).
                from sharedagents_mcp.memory.client import get_memory_client

                _sk_home = getattr(self.cfg, "home", None)
                idx = SkillIndex(
                    await get_memory_client(_sk_home).embedder(),
                    _sk_home,
                )
                self._skill_index = idx
            if not idx.is_available:
                return
            matches = await idx.search(
                user_text,
                top_k=top_k,
                threshold=threshold,
            )
            block = render_preamble(matches)
            if block:
                # Skill bodies are TRUSTED instruction-tier content, NOT
                # untrusted external data, so they are injected unfenced (a
                # fence would tell the model to ignore the skill's guidance,
                # defeating the feature). This is safe because skills are
                # discovered ONLY from operator-controlled search paths
                # (SHAREDAGENTS_SKILLS_DIRS, <agent_home>/skills, ~/.sharedagents/skills,
                # ~/.claude/skills, installed plugins — see
                # skills.skill_search_paths) and NEVER from the working
                # directory or a cloned repo. They sit in the same trust tier
                # as AGENTS.md / SOUL.md / the system prompt. (An operator who
                # installs a malicious skill has already been trusted with the
                # system prompt — out of the prompt-injection threat model,
                # which targets non-user EXTERNAL content.)
                #
                # user-role (see preamble-role note in
                # compose_round_preamble): the skill block is appended
                # after the user-role pinned/summary/todos blocks, so a
                # system role here would not be leading and strict chat
                # templates reject it ("System message must be at the
                # beginning").
                out.append({"role": "user", "content": block})
        except Exception:  # noqa: BLE001
            pass
