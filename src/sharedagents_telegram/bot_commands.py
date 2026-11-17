"""Slash-command + agent-delegation handlers for :class:`AcpTelegramBot`.

Extracted from the :class:`AcpTelegramBot` god-class (de-monolith). This is a
mixin: every method still binds ``self`` and dispatches identically through the
concrete class's MRO. Behaviour is unchanged — the method bodies moved verbatim.

The module-level helpers/constants these methods read stay owned by ``bot.py``
and are imported at the BOTTOM (after the class) so the ``bot`` <-> mixin import
cycle is deadlock-free in either import order.
"""

from __future__ import annotations

import json
from typing import Any

from sharedagents_agent.tools.widget_adapt import (
    adapt_todo_args,
    format_todo_lines,
    is_todo_tool_name,
)


def _session_mode_label(state: dict[str, Any]) -> str:
    """Return the authoritative public mode label for Telegram status."""
    nested = state.get("effective_mode")
    if isinstance(nested, dict):
        label = nested.get("label") or nested.get("id")
        if label:
            return str(label)
    elif nested:
        return str(nested)
    return str(state.get("effective_mode_label") or state.get("effective_mode_id") or "")


def _status_probe_sid(sessions_dir: Any, sid: str) -> str:
    """Return the session SID used for local metrics."""
    del sessions_dir
    return sid


class CommandsMixin:
    async def _handle_command(
        self,
        chat_id: int,
        cmd: str,
        thread_id: int | None = None,
    ) -> None:
        """Route a slash command. ``thread_id`` is propagated to every
        reply so messages in a forum topic stay in that topic.
        """
        if self._api is None:
            return  # purely a unit-test path

        async def reply(text: str) -> None:
            await self._api.send_message(
                chat_id,
                text,
                message_thread_id=thread_id,
            )

        if cmd in ("/help", "/start"):
            caps_state = "fs.readTextFile=on (trusted)" if chat_id in self.trusted_chats else "text-only"
            # Live listing of file-based slash commands so bundled
            # ones (e.g. /goal) AND any operator-installed
            # ``commands/*.md`` show up in /help. The total reply has
            # to fit Telegram's 4096-char ceiling though, so cap the
            # inline listing at ``_HELP_FILE_CMDS_SHOW`` and point at
            # /commands for the full list when there's overflow —
            # otherwise a 80-file ``commands/`` dir pushes the rest of
            # the card past the truncation boundary and the user only
            # sees a half-listed file-commands section. Fail-soft: a
            # malformed commands dir falls back to the generic hint.
            _HELP_FILE_CMDS_SHOW = 10
            try:
                from sharedagents_command.file_commands import discover_commands

                _file_cmds = discover_commands()
            except Exception:  # noqa: BLE001
                _file_cmds = []
            if _file_cmds:
                fc_lines = ["", "File-based commands (commands/*.md):"]
                for c in _file_cmds[:_HELP_FILE_CMDS_SHOW]:
                    hint = f" {c.argument_hint}" if c.argument_hint else ""
                    fc_lines.append(f"  /{c.name}{hint}  — {c.short_description}")
                extra = len(_file_cmds) - _HELP_FILE_CMDS_SHOW
                if extra > 0:
                    fc_lines.append(f"  … {extra} more — see /commands for the full list")
                file_commands_block = "\n".join(fc_lines) + "\n"
            else:
                file_commands_block = "  /<command> [args]      — run a file command (commands/*.md)\n"
            body = (
                "sharedagents — local-LLM agent on Telegram (ACP).\n\n"
                "Slash commands:\n"
                "  /help                  — this card\n"
                "  /new                   — drop session, next msg starts fresh\n"
                "  /status                — agent + bound session id\n"
                "  /stop, /cancel       — graceful stop (finish current step)\n"
                "  /abort                — hard kill now (SIGKILL + killpg)\n"
                "  /steer <text>          — redirect the running turn\n"
                "  /sessions              — list saved sessions\n"
                "  /history [N]           — last N turns (default 5)\n"
                "  /todo, /todos          — todos from the model's todo tool\n"
                "  /compact               — squeeze the running prompt now\n"
                "  /context [list|detail|json] — context tokens & turns report\n"
                "  /skills                — list installed skills\n"
                "  /skill <name>          — show one skill's SKILL.md body\n"
                "  /commands              — list file-based slash commands\n"
                + file_commands_block
                + "  /refresh               — re-load session state from disk\n"
                + "  /restart               — restart the sharedagents gateway\n"
                "  /mode [plan|ask|auto]  — permission mode for tool calls\n"
                "  /models                — list configured models\n"
                "  /model <alias|url>     — switch the live session model\n"
                "  /reasoning [on|off|stream] — thinking-token visibility\n"
                "  /on, /off, /stream     — shortcuts for /reasoning <level>\n"
                "  /effort <level|default> — model-specific reasoning depth\n"
                "  /thinking [on|off]     — enable model thinking\n"
                "  /tools [on|off|stream] — tool visibility (off=none, "
                "on=header only, stream=header+output)\n"
                "  /teacher [on|off] | /teacher model <alias> — teacher-model supervision\n"
                "  /plugin {list|install|enable|disable|remove} — manage plugin bundles\n"
                "  /mcp [filter]          — list MCP servers\n"
                "  /teacher [on|off|model <alias>] — toggle/pick teacher-model supervision\n"
                "  /subagents [N]         — list / enter delegated subagent sessions\n"
                "  /monitors, /shells     — background shells the model spawned\n"
                "  /mouse [on|off]        — toggle TUI mouse capture (no-op here)\n"
                "  /task, /tasks          — operator harness-task store\n"
                "  /claude <prompt>       — bridge to Claude Code via delegate_claude\n"
                "  /codex <prompt>        — bridge to Codex CLI via delegate_codex\n"
                "  /antigravity <prompt>       — bridge to Antigravity CLI via delegate_antigravity\n"
                "  /agents                — list configured sharedagents agents\n"
                "  /agent <name>          — switch this chat to <name> "
                "(new session in target)\n"
                "  /<agent> <task>        — one-shot delegation to <agent> "
                "via delegate_agent\n\n"
                f"Capabilities: {caps_state}\n"
                f"Your chat_id is {chat_id}."
            )
            # Wrap in a MarkdownV2 fenced code block so Telegram
            # renders the body in MONOSPACE — that's the only way the
            # space-padded column alignment (``  /cmd <spaces> —``)
            # looks right; Telegram's default proportional font
            # destroyed the layout under plain text and the user saw a
            # ragged blob. Inside ``` ... ``` only ``\`` and ``` ` ```
            # need escaping per Telegram's MarkdownV2 spec; everything
            # else (—, *, _, |, etc.) is rendered as-is.
            escaped = body.replace("\\", "\\\\").replace("`", "\\`")
            fenced = f"```\n{escaped}\n```"
            await self._api.send_message(
                chat_id,
                fenced,
                message_thread_id=thread_id,
                parse_mode="MarkdownV2",
            )
            if cmd == "/start":
                # Force-create a fresh session for this (chat,topic).
                self._drop_session_id(chat_id, thread_id)
            return
        if cmd == "/new":
            agent_cfg = self._agent_cfg(chat_id, thread_id)
            engine = (getattr(agent_cfg, "engine", "") or "").strip()
            external_engine = engine in (
                "claude_cli",
                "codex_cli",
                "antigravity_cli",
                "zcode_cli",
                "acp",
                "cursor_acp",
            )
            sid = self.get_session_id(chat_id, thread_id)
            current_model = ""
            # Read the model the live (about-to-be-dropped) session is on —
            # for EVERY engine, not just external CLI/ACP ones. The figure
            # bug: a ``local_llm`` agent the operator
            # switched via ``/model dpsk`` showed ``qwen3.6-27b-fp8`` (the
            # stale disk-config ``upstream_model``) in the ``/new`` banner
            # while ``/models`` correctly reported ``deepseek-v4-flash``.
            # ``/model`` is session-scoped (mutates ``sess.model`` only); the
            # bot's ``agent_cfg`` is a disk-read snapshot. Mirror ``/models``'
            # source of truth (the per-session ``engine_model``/``model`` on
            # disk) here for the banner of the session being dropped.
            if sid:
                try:
                    from sharedagents_session.session_reader import load_session_dict

                    raw = load_session_dict(
                        agent_cfg.sessions_dir,
                        sid,
                        expected_agent=str(getattr(agent_cfg, "name", "") or ""),
                    )
                    if isinstance(raw, dict):
                        current_model = raw.get("engine_model") or raw.get("model") or ""
                except Exception as exc:  # noqa: BLE001
                    self._log.debug("/new session model read failed: %s", exc)
            # The live sid can be unavailable even when the agent HAS a
            # concrete model: after a gateway restart ``verify_sessions_on_
            # startup`` marks the binding stale, so ``get_session_id``
            # returns None and the read above is skipped. The sessions DB
            # still holds the agent's last ``engine_model`` — consult it so
            # the banner shows ``claude-opus-4-8`` rather than collapsing to
            # the bare engine name ``claude``.
            if not current_model and external_engine:
                try:
                    from sharedagents_session.session_reader import (
                        latest_engine_model_for_agent,
                    )

                    current_model = latest_engine_model_for_agent(
                        agent_cfg.sessions_dir,
                        str(getattr(agent_cfg, "name", "") or ""),
                    )
                except Exception as exc:  # noqa: BLE001
                    self._log.debug("/new latest engine_model read failed: %s", exc)
            self._drop_session_id(chat_id, thread_id)
            # CLI-engine agents (claude/codex/antigravity) don't drive an
            # HTTP upstream — the CLI itself owns the model call, and
            # ``upstream_model`` belongs to a different (local-LLM)
            # deployment. Mirror the TUI status-bar rendering
            # (app.py:5911-5921) so we don't falsely advertise the
            # global config's upstream id.
            if current_model:
                from sharedagents_command.commands import resolve_session_model

                model = resolve_session_model(current_model, agent_cfg)
            elif cold_model := _cold_external_engine_model(engine):
                model = cold_model
            else:
                from sharedagents_command.commands import cold_external_model_for

                discovered = cold_external_model_for(agent_cfg)
                if discovered:
                    model = discovered
                elif engine in ("claude_cli", "codex_cli", "antigravity_cli", "zcode_cli"):
                    model = engine[:-4]  # "claude" / "codex" / "antigravity" / "zcode"
                elif external_engine:
                    # Generic ACP agents (``acp``) drive an external ACP server,
                    # NOT an HTTP upstream — the inherited ``upstream_model``
                    # belongs to a different (local-LLM) deployment and must
                    # never be advertised here.
                    model = (str(getattr(agent_cfg, "name", "") or "")) or "(default)"
                else:
                    model = getattr(agent_cfg, "upstream_model", "") or "(default)"
            await reply(f"✨ new session started with agent `{agent_cfg.name}` model `{model}`.")
            return
        if cmd == "/status":
            sid = self.get_session_id(chat_id, thread_id)
            sid_disp = sid[:12] + "…" if sid else "(none — next message creates one)"
            topic_line = f"\ntopic: {thread_id}" if thread_id is not None else ""
            session_line = ""
            if sid:
                try:
                    await self._get_or_open_client(
                        chat_id,
                        thread_id,
                    )
                    state = await self._get_session_state(sid)
                    await self._apply_session_state(
                        chat_id,
                        thread_id,
                        sid,
                        state,
                    )
                    mode_label = _session_mode_label(state)
                    session_line = f"\nmode: {mode_label or '?'}"
                except Exception:  # noqa: BLE001
                    pass
            # Context-length probe: only meaningful when a session is
            # open. Reads the persisted session JSON directly off
            # disk — the SDK's ``LoadSessionResponse`` doesn't carry
            # ``turns`` or ``system_prompt`` (it was returning 0
            # tokens even on a freshly-bootstrapped agent that has
            # SOUL/AGENTS/TOOLS context worth thousands of tokens).
            #
            # Path: ``<agent_home>/sessions/<sid>.json``. The session
            # file is the canonical source of truth — ``Session.save``
            # writes it atomically — so reading it here costs one
            # ``stat`` + ``read`` and stays consistent with what the
            # next prompt round will replay.
            ctx_line = ""
            raw: dict = {}
            agent_cfg = self._agent_cfg(chat_id, thread_id)
            if sid is not None:
                try:
                    from sharedagents_session.session_reader import load_session_dict

                    probe_sid = _status_probe_sid(
                        agent_cfg.sessions_dir,
                        sid,
                    )
                    raw = load_session_dict(
                        agent_cfg.sessions_dir,
                        probe_sid,
                        expected_agent=str(getattr(agent_cfg, "name", "") or ""),
                    )
                    if not raw:
                        raise FileNotFoundError("active physical session unavailable")
                    turns = list(raw.get("turns") or [])
                    system_prompt = str(raw.get("system_prompt") or "")
                    # Count chars across system prompt + every turn's
                    # user/content/thinking text. 4 chars ≈ 1 token,
                    # matches turn_runtime._approx_tokens.
                    char_count = len(system_prompt)
                    for t in turns:
                        if not isinstance(t, dict):
                            continue
                        for key in ("user", "content", "thinking"):
                            v = t.get(key)
                            if isinstance(v, str):
                                char_count += len(v)
                            elif isinstance(v, list):
                                for blk in v:
                                    if isinstance(blk, dict):
                                        char_count += len(blk.get("text") or "")
                    approx_tokens = char_count // 4
                    # Per-session contextWindow takes precedence over
                    # the global default (different models have
                    # different windows).
                    sess_ctx = int(raw.get("context_window") or 0)
                    if not sess_ctx:
                        sess_endpoint = str(raw.get("endpoint") or "")
                        for m in getattr(agent_cfg, "models", None) or []:
                            if not isinstance(m, dict):
                                m = m.__dict__ if hasattr(m, "__dict__") else {}
                            if (m.get("baseUrl") or "").rstrip("/") == sess_endpoint.rstrip("/"):
                                sess_ctx = int(m.get("contextWindow") or 0)
                                break
                    if not sess_ctx:
                        sess_ctx = int(getattr(agent_cfg, "upstream_max_context", 0) or 0)
                    pct = (approx_tokens * 100 / sess_ctx) if sess_ctx else 0.0
                    if sess_ctx:
                        ctx_line = f"\ncontext: ~{approx_tokens} / {sess_ctx} tokens ({pct:.1f}%, {len(turns)} turns)"
                    else:
                        ctx_line = f"\ncontext: ~{approx_tokens} tokens ({len(turns)} turns)"
                except Exception as exc:  # noqa: BLE001
                    ctx_line = f"\ncontext: (probe failed: {type(exc).__name__})"
            # Teacher-model supervision indicator (parity with the TUI footer
            # widget). Reflects the EFFECTIVE per-session state — the
            # ``/teacher on|off`` override and the active goal both gate it,
            # not just the agent-config policy — via the shared
            # ``teacher_supervisor.is_teacher_active`` predicate, reading the
            # persisted session header (``teacher_override`` / ``active_goal``
            # / ``teacher_model_override``) that the engine writes.
            try:
                from sharedagents_agent.teacher_supervisor import (
                    effective_teacher_model,
                    is_teacher_active,
                )

                _tpol = (agent_cfg.teacher_policy() if hasattr(agent_cfg, "teacher_policy") else {}) or {}
                _tov = str(raw.get("teacher_override") or "")
                _tmov = str(raw.get("teacher_model_override") or "")
                _goal = str(raw.get("active_goal") or "")
                if is_teacher_active(_tpol, _tov, _goal):
                    _trow = (agent_cfg.teacher_model() if hasattr(agent_cfg, "teacher_model") else None) or {}
                    _tmodel = effective_teacher_model(_trow, _tmov) or "?"
                    teacher_line = f"\nteacher: 🧑‍🏫 on ({_tmodel})"
                else:
                    teacher_line = "\nteacher: off"
            except Exception:  # noqa: BLE001
                teacher_line = ""
            await reply(
                f"agent: {agent_cfg.name}\n"
                f"session: {sid_disp}{topic_line}{session_line}"
                f"{ctx_line}{teacher_line}\n"
                f"capabilities: " + ("fs.readTextFile (trusted)" if chat_id in self.trusted_chats else "text-only"),
            )
            return
        if cmd in ("/stop", "/cancel"):
            await self._dispatch_abort(
                chat_id,
                thread_id,
                via=cmd.lstrip("/"),
                reply=reply,
            )
            return

        # /steer <text> — redirect the running turn. Carries its own
        # argument, so split before the generic head/arg handling below.
        _steer_head, _, _steer_arg = cmd.partition(" ")
        if _steer_head.split("@", 1)[0].lower() == "/steer":
            await self._dispatch_steer(chat_id, thread_id, _steer_arg, reply)
            return

        # Skills — list installed skills, or show one skill's SKILL.md.
        # Handled before the generic head/arg split below since
        # ``/skill <name>`` carries its own argument.
        _sk_head, _, _sk_arg = cmd.partition(" ")
        _sk_norm = _sk_head.split("@", 1)[0].lower()
        if _sk_norm in ("/skills", "/skill"):
            await self._handle_skills_command(
                chat_id,
                _sk_arg.strip(),
                thread_id,
            )
            return
        if _sk_norm in ("/plugin", "/plugins"):
            await self._handle_plugin_command(
                chat_id,
                _sk_arg.strip(),
                thread_id,
            )
            return
        if _sk_norm in ("/mcp", "/mcps"):
            await self._handle_mcp_command(
                chat_id,
                _sk_arg.strip(),
                thread_id,
            )
            return

        # Configuration commands — route through ACP. ``cmd`` here is
        # the full text including any argument (e.g. "/model kimi"),
        # not just the verb. parse_command handles the head + arg
        # split and the alias collapse (/on → /reasoning on, etc.).
        head, _, arg = cmd.partition(" ")
        head_lower = head.split("@", 1)[0].lower()
        cfg_cmds = {
            "/mode",
            "/model",
            "/models",
            "/reasoning",
            "/reason",
            "/on",
            "/off",
            "/stream",
            "/effort",
            "/thinking",
            "/tools",
            "/tool",
        }
        if head_lower in cfg_cmds:
            await self._handle_config_command(chat_id, cmd, thread_id)
            return
        session_cmds = {
            "/sessions",
            "/history",
            "/todo",
            "/todos",
            "/abort",
            "/refresh",
            "/restart",
            "/compact",
            "/context",
            "/ctx",
            "/teacher",
        }
        if head_lower in session_cmds:
            await self._handle_session_command(chat_id, cmd, thread_id)
            return
        # Delegate sugars — same shape the TUI uses (synthesize a user
        # message that biases the local model to emit delegate_claude /
        # delegate_codex / delegate_antigravity). Keeps "external bridges
        # are tools, not slash commands" while giving the same UX on
        # every frontend.
        if head_lower in ("/claude", "/codex", "/antigravity"):
            await self._handle_delegate_command(
                chat_id,
                head_lower,
                arg,
                thread_id,
            )
            return

        # Multi-agent routing commands. ``/agents`` (the sibling list) is a
        # runtime command — it falls through to the server-first passthrough
        # below so Telegram carries no command logic of its own.
        # Only ``/agent <name>`` (switch THIS chat's bound
        # agent) stays local: it rebinds Telegram-side session state.
        if head_lower == "/agent":
            await self._handle_agent_switch(
                chat_id,
                arg.strip(),
                thread_id,
            )
            return
        # ``/<agent_name> <task>`` — delegate sugar that targets a
        # sharedagents agent the way ``/claude`` targets Claude Code.
        # Resolved against the configured agent list so a typo
        # falls through to the "unknown command" reply instead of
        # spawning a phantom child.
        agent_target = head_lower[1:] if head_lower.startswith("/") else ""
        if agent_target and self._is_known_agent(agent_target):
            await self._handle_agent_delegate(
                chat_id,
                agent_target,
                arg,
                thread_id,
            )
            return

        # File-based slash commands (Claude Code ``commands/*.md``,
        # including everything-claude-code). ``/commands`` lists them;
        # ``/<name> [args]`` expands the command's prompt body (with
        # $ARGUMENTS / $1… substitution) and runs it as a normal turn.
        # ``/commands`` (list file-backed commands) is a runtime command —
        # it falls through to the server-first passthrough below rather than a
        # Telegram-local listing.
        _fc_name = agent_target
        if _fc_name:
            try:
                from sharedagents_command.file_commands import (
                    expand_command,
                    find_command,
                )

                _fc = find_command(_fc_name)
            except Exception:  # noqa: BLE001
                _fc = None
            if _fc is not None:
                expanded = expand_command(_fc, arg.strip())
                await self._stream_turn_for_chat(
                    chat_id,
                    text=expanded,
                    thread_id=thread_id,
                )
                return

        # Server-first fallback: any command the shared runtime can
        # dispatch but Telegram doesn't special-case locally (e.g.
        # /teacher, /subagents, /monitors, /shells, /mouse,
        # /task) routes through the SAME ACP ``session/command``
        # passthrough the config/session commands use. This keeps the
        # bot free of per-command logic (ARCHITECTURE LAW: zero command
        # behavior on frontends) and at automatic parity with the
        # TUI/web for every present AND future runtime command. Only a
        # name the runtime itself can't dispatch falls through to the
        # genuine "unknown command" reply.
        from sharedagents_command.command_runtime import list_commands as _list_cmds
        from sharedagents_command.commands import parse_command as _parse_cmd

        _parsed = _parse_cmd(cmd)
        if _parsed is not None and _parsed.name in set(_list_cmds()):
            await self._handle_session_command(chat_id, cmd, thread_id)
            return

        await reply(f"unknown command: {cmd}\nsend /help to list commands")

    async def _handle_plugin_command(
        self,
        chat_id: int,
        arg: str,
        thread_id: int | None = None,
    ) -> None:
        """``/plugin {list|install|enable|disable|remove}`` — manage
        plugin bundles (one dir packaging skills + commands + agents +
        hooks). Same surface as the TUI ``/plugin``.
        """
        if self._api is None:
            return
        from sharedagents_agent import plugins as _plug

        async def reply(text: str) -> None:
            await self._api.send_message(
                chat_id,
                text,
                message_thread_id=thread_id,
            )

        parts = (arg or "list").strip().split(None, 1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub == "list":
            items = _plug.list_plugins()
            if not items:
                await reply(
                    "no plugins installed — try `/plugin search <query>` then `/plugin install <name-or-git-url>`",
                )
                return
            lines = [f"{len(items)} plugin(s) at ~/.sharedagents/plugins/:", ""]
            for p in items:
                mark = "" if p.enabled else "✗ "
                comps = p.components()
                summary = ", ".join(f"{k}={v}" for k, v in comps.items()) or "(empty)"
                lines.append(f"• {mark}{p.name} ({p.version or '-'}) — {summary}")
                if p.description:
                    lines.append(f"     {p.description[:80]}")
                if p.source:
                    lines.append(f"     source: {p.source}")
            await reply("\n".join(lines))
            return

        if sub == "search":
            results = _plug.search(rest or "")
            if not results:
                await reply(
                    f"no marketplace matches for {rest!r}\n"
                    "tip: try a broader keyword like `skill`, `hook`, `mcp`, "
                    "or drop ~/.sharedagents/plugin-marketplace.json with your "
                    "own entries"
                )
                return
            label = f"matches for {rest!r}" if rest else "marketplace listing"
            lines = [f"{len(results)} {label}:", ""]
            for e in results:
                comps = ",".join(e.components) or "-"
                tags = ",".join(e.tags) or "-"
                lines.append(f"• {e.name} [{comps}]  tags={tags}")
                if e.description:
                    lines.append(f"     {e.description[:90]}")
                lines.append(f"     install: {e.install_hint()}")
            lines.append("")
            lines.append(
                "tip: `/plugin install <name>` installs by marketplace name, or paste any git URL / local path"
            )
            await reply("\n".join(lines))
            return

        if sub == "install":
            if not rest:
                await reply("usage: /plugin install <git-url|local-path>")
                return
            try:
                p = _plug.install(rest.strip())
            except Exception as exc:  # noqa: BLE001
                await reply(f"install failed: {exc}")
                return
            comps = ", ".join(f"{k}={v}" for k, v in p.components().items()) or "(no components detected)"
            await reply(f"installed {p.name} → {p.path}\n  components: {comps}")
            return

        if sub in ("enable", "disable"):
            if not rest:
                await reply(f"usage: /plugin {sub} <name>")
                return
            try:
                fn = _plug.enable if sub == "enable" else _plug.disable
                p = fn(rest)
            except Exception as exc:  # noqa: BLE001
                await reply(f"{sub} failed: {exc}")
                return
            await reply(f"{p.name}: {'enabled' if p.enabled else 'disabled'}")
            return

        if sub == "remove":
            if not rest:
                await reply("usage: /plugin remove <name> [--purge]")
                return
            purge = " --purge" in rest or rest.endswith(" --purge")
            name = rest.replace(" --purge", "").strip()
            try:
                where = _plug.remove(name, purge=purge)
            except Exception as exc:  # noqa: BLE001
                await reply(f"remove failed: {exc}")
                return
            await reply(f"removed {name} → {where}" + ("" if purge else " (in .trash/)"))
            return

        await reply(
            f"unknown /plugin subcommand: {sub!r}\nusage: /plugin {{list|search|install|enable|disable|remove}} [args]"
        )

    async def _handle_mcp_command(
        self,
        chat_id: int,
        arg: str,
        thread_id: int | None = None,
    ) -> None:
        """``/mcp [filter]`` — list MCP servers visible to the agent.

        Mirrors the TUI's ``_cmd_mcp`` so Telegram operators see the
        same Claude-Code-compatible view: name, transport, discovery
        source. A filter argument scopes by substring against name +
        transport + description.
        """
        if self._api is None:
            return

        async def reply(text: str) -> None:
            await self._api.send_message(
                chat_id,
                text,
                message_thread_id=thread_id,
            )

        from sharedagents_mcp.mcp_client import list_mcp_servers_with_source

        servers = list_mcp_servers_with_source()
        flt = (arg or "").strip().lower()
        if flt:
            servers = [
                (s, src)
                for (s, src) in servers
                if flt in s.name.lower() or flt in (s.transport or "").lower() or flt in (s.description or "").lower()
            ]
        if not servers:
            if flt:
                await reply(
                    f"no MCP servers match {arg!r}.\nconfigured servers can be listed with `/mcp` (no argument)."
                )
            else:
                await reply(
                    "no MCP servers configured.\n"
                    "tip: add one in ~/.sharedagents/config.json under "
                    "`mcpServers`, drop ~/.claude/.mcp.json, or "
                    "install a plugin bundle that ships an `.mcp.json`."
                )
            return
        lines = [f"{len(servers)} MCP server(s):", ""]
        for server, src in servers:
            head = f"• {server.name} [{server.transport}] [{src}]"
            lines.append(head)
            if server.transport == "stdio":
                cmd_line = server.command
                if server.args:
                    cmd_line += " " + " ".join(server.args)
                if cmd_line:
                    lines.append(f"     exec: {cmd_line}")
            else:
                if server.url:
                    lines.append(f"     url:  {server.url}")
            if server.description:
                lines.append(f"     desc: {server.description[:80]}")
        lines.append("")
        lines.append(
            "sources: [config]=~/.sharedagents/config.json  [claude]=~/.claude/.mcp.json  [plugin]=bundle .mcp.json"
        )
        await reply("\n".join(lines))

    async def _handle_commands_list(
        self,
        chat_id: int,
        thread_id: int | None = None,
    ) -> None:
        """``/commands`` — list file-based slash commands (Claude Code
        ``commands/*.md``, including everything-claude-code). Each
        ``/<name> [args]`` expands that command's prompt body into a
        normal user turn.
        """
        if self._api is None:
            return
        from sharedagents_command import file_commands as _fc

        async def reply(text: str) -> None:
            await self._api.send_message(
                chat_id,
                text,
                message_thread_id=thread_id,
            )

        cmds = _fc.discover_commands()
        if not cmds:
            await reply(
                "no file commands installed — add a dir to "
                "`commands_dirs` in ~/.sharedagents/config.json, or drop one "
                "in ~/.sharedagents/commands/<name>.md",
            )
            return
        lines = [f"{len(cmds)} command(s) — type /<name> [args] to run one:", ""]
        for c in cmds:
            hint = f" {c.argument_hint}" if c.argument_hint else ""
            lines.append(f"• /{c.name}{hint} — {c.short_description}")
        text = "\n".join(lines)
        limit = 3800
        buf = ""
        for line in text.split("\n"):
            if buf and len(buf) + len(line) + 1 > limit:
                await reply(buf)
                buf = ""
            buf = f"{buf}\n{line}" if buf else line
        if buf:
            await reply(buf)

    async def _handle_skills_command(
        self,
        chat_id: int,
        arg: str,
        thread_id: int | None = None,
    ) -> None:
        """``/skills`` — list installed skills; ``/skill <name>`` — show
        one skill's SKILL.md body.

        Skills are the agent's on-demand runbooks. This is the
        operator-facing view of the same discovery chain the agent
        sees (per-agent → ``~/.sharedagents/skills`` → config
        ``skills_dirs`` bundles → builtin), so it doubles as a quick
        check that an everything-claude-code / OpenClaw bundle
        registered. Long output is chunked on line boundaries to fit
        Telegram's ~4 KB message cap.
        """
        if self._api is None:
            return
        from sharedagents_skill import skills as _skills_mod

        async def reply(text: str) -> None:
            await self._api.send_message(
                chat_id,
                text,
                message_thread_id=thread_id,
            )

        async def reply_chunked(text: str, limit: int = 3800) -> None:
            # Split on line boundaries so a skill row is never cut mid-word.
            buf = ""
            for line in text.split("\n"):
                if buf and len(buf) + len(line) + 1 > limit:
                    await reply(buf)
                    buf = ""
                buf = f"{buf}\n{line}" if buf else line
            if buf:
                await reply(buf)

        arg = (arg or "").strip()
        if arg:
            skill = _skills_mod.find_skill(arg)
            if skill is None:
                await reply(
                    f"skill not found: {arg!r}\nsend /skills to list what's installed",
                )
                return
            head = [f"# {skill.name}", skill.short_description, ""]
            if skill.version:
                head.append(f"version: {skill.version}")
            if skill.requires_env:
                head.append(f"requires env: {', '.join(skill.requires_env)}")
            if skill.requires_bins:
                head.append(f"requires bins: {', '.join(skill.requires_bins)}")
            head.append(f"source: {skill.source}")
            head.append("")
            body = skill.body or "(SKILL.md has frontmatter only — no body)"
            await reply_chunked("\n".join(head) + "\n" + body)
            return
        skills = _skills_mod.discover_skills(include_disabled=True)
        if not skills:
            await reply(
                "no skills installed — add a dir to `skills_dirs` in "
                "~/.sharedagents/config.json, or drop one in "
                "~/.sharedagents/skills/<name>/",
            )
            return
        enabled = {s.name for s in _skills_mod.discover_skills()}
        lines = [f"{len(skills)} skill(s) — /skill <name> to read one:", ""]
        for s in skills:
            mark = "" if s.name in enabled else "✗ "
            lines.append(f"• {mark}{s.name} — {s.short_description}")
        if len(enabled) < len(skills):
            lines.append("")
            lines.append("✗ = disabled via the `skills` map in ~/.sharedagents/config.json")
        await reply_chunked("\n".join(lines))

    def _render_command_result(self, result: dict) -> str:
        """Render a ``CommandResponse.result`` for Telegram (a text-only
        transport) by dispatching SOLELY on ``result.get("kind")`` — never on
        which command produced it. The runtime sets ``text``
        on every response, so that is the primary body; ``kind`` selects a
        structured fallback when a response carries only ``data``.
        """
        if not isinstance(result, dict):
            return "(no output)"
        kind = result.get("kind") or "text"
        body = result.get("text")
        if isinstance(body, str) and body.strip():
            return body
        data = result.get("data") or {}
        if kind == "error":
            return (isinstance(body, str) and body) or "error"
        if kind == "checklist":
            glyph = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]", "deleted": "[-]"}
            lines = [
                f"{glyph.get(it.get('status'), '[ ]')} {it.get('content', '')}"
                for it in (data.get("items") or [])
                if isinstance(it, dict)
            ]
            return "\n".join(lines) or "(no items)"
        if kind in ("list", "table"):
            lines = []
            title = data.get("title")
            if title:
                lines.append(str(title))
            for it in data.get("items") or []:
                if isinstance(it, dict):
                    lines.append(
                        "  "
                        + str(
                            it.get("title") or it.get("content") or it.get("id") or "",
                        )
                    )
                else:
                    lines.append("  " + str(it))
            return "\n".join(lines) or "(empty)"
        if kind in ("keyvals", "detail"):
            pairs = data.get("pairs") or data.get("items") or []
            if isinstance(pairs, dict):
                pairs = list(pairs.items())
            lines = []
            for pr in pairs:
                if isinstance(pr, (list, tuple)) and len(pr) == 2:
                    lines.append(f"{pr[0]}: {pr[1]}")
                elif isinstance(pr, dict):
                    lines.append(
                        f"{pr.get('key', pr.get('label', ''))}: {pr.get('value', '')}",
                    )
            return "\n".join(lines) or "(no output)"
        return "(no output)"

    async def _handle_session_command(
        self,
        chat_id: int,
        text: str,
        thread_id: int | None = None,
    ) -> None:
        """Apply session-management slash commands.

        ``/sessions`` — list saved sessions (most recent first).
        ``/history [N]`` — show last N turns from the cached view.
        ``/todo[s]`` — list session todos written by the ``todo`` tool.
        ``/abort`` — hard kill now (SIGKILL + killpg); ``/stop`` / ``/cancel``
        are graceful (finish the in-flight step, then end).
        ``/refresh`` — re-bind the chat to the current session id (no-op
        on Telegram since we're always synced via the wire).
        ``/restart`` — TUI-only (daemon respawn); on Telegram, ack with
        a hint.
        """
        from sharedagents_command.commands import parse_command

        if self._api is None:
            return

        async def reply(t: str) -> None:
            await self._reply(chat_id, t, thread_id)

        parsed = parse_command(text)
        if parsed is None:
            await reply(f"unknown command: {text}")
            return

        if parsed.name == "abort":
            await self._dispatch_abort(
                chat_id,
                thread_id,
                via="abort",
                reply=reply,
            )
            return

        if parsed.name == "restart":
            # /restart bounces the sharedagents gateway — the single
            # process hosting every agent — and drops this chat's
            # cached client so the next message reconnects against
            # the respawned gateway. Bouncing the gateway cascades
            # to every child (every agent), so all chats see a brief
            # reconnect; per-chat session ids are persisted on disk
            # and re-loaded by the gateway's children on the next
            # ``session/load``. Allowed for any allowlisted chat —
            # the allowlist is the trust boundary.
            #
            # ``/restart --hard`` extends the bounce to THIS bot's own
            # systemd service (sharedagents-telegram). The restart helper
            # runs detached and sleeps ~1s, so the "restarting…" reply
            # below flushes before systemd stops us — the same trick
            # the gateway bounce already relies on.
            from sharedagents_core.restart import restart_agent_detached

            hard = "--hard" in (parsed.arg or "").split()
            agent_cfg = self._agent_cfg(chat_id, thread_id)
            res = restart_agent_detached(agent_cfg.name, hard=hard)
            if not res.get("ok"):
                await reply(f"⚠️ restart failed: {res.get('error')}")
                return
            chat_key = TelegramConfig.chat_key(chat_id, thread_id)
            client = self._clients.pop(chat_key, None)
            try:
                if client is not None and hasattr(client, "close"):
                    await client.close()
            except Exception:  # noqa: BLE001
                pass
            # Stdio fallback only — production path (HTTP-to-gateway)
            # has no per-chat child process; ``_child_procs`` stays
            # empty for HTTP-mode chats.
            proc = self._child_procs.pop(chat_key, None)
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:  # noqa: BLE001
                    pass
            if res.get("telegram"):
                await reply(
                    "♻ restarting sharedagents gateway AND this Telegram bot "
                    "(graceful, via systemd). The bot will go quiet for a "
                    "few seconds and reconnect — your session is preserved.",
                )
            elif hard:
                # --hard asked for, but the bot isn't under systemd here
                # (or the unit is missing): only the gateway can bounce.
                await reply(
                    "♻ restarting sharedagents gateway. The Telegram bot isn't "
                    "under systemd here, so it stays up — restart "
                    "sharedagents-telegram manually if you need it bounced. "
                    "Your session is preserved.",
                )
            else:
                await reply(
                    "♻ restarting sharedagents gateway. "
                    "Your session is preserved — the next message will "
                    "land on the respawned agent.",
                )
            return

        try:
            client = await self._get_or_open_client(chat_id, thread_id)
        except Exception as exc:  # noqa: BLE001
            self._log.exception(
                "session command client open failed chat=%s thread=%s",
                chat_id,
                thread_id,
            )
            await reply(f"⚠️ agent unreachable: {exc}")
            return
        agent_cfg = self._agent_cfg(chat_id, thread_id)

        # Server-first: route feature commands through the agent's shared
        # runtime (sharedagents_command.command_runtime via ACP session/command) so
        # Telegram renders the SAME output as the TUI and web, carrying no
        # command logic of its own. Session-navigation commands that rebind
        # THIS chat (`/sessions`, `/refresh`) and process control
        # (`/abort`, `/restart`, handled above) stay local. Falls back to
        # the legacy in-process handling when the gateway is too old to
        # expose `session/command` or doesn't recognize the command.
        if parsed.name not in ("sessions", "refresh"):
            sid_for_cmd = self.get_session_id(chat_id, thread_id)
            # A runtime-known command sent with no session yet must STILL
            # reach the shared runtime — the legacy local fallback below
            # only implements a handful of commands, so anything else
            # (e.g. /teacher sent before the first message) would drop to
            # the misleading "unknown command" reply at the end of this
            # method. Open a session on demand, exactly as
            # ``_handle_config_command`` does for /mode, /model, etc. Skip
            # the commands that carry their OWN no-session UX locally
            # (/history, /todo, /context, /compact) so they keep behaving
            # as before. See tests/test_telegram_command_parity.py.
            if not sid_for_cmd and parsed.name not in (
                "history",
                "todo",
                "context",
                "compact",
            ):
                from sharedagents_command.command_runtime import (
                    list_commands as _list_cmds,
                )

                if parsed.name in set(_list_cmds()):
                    try:
                        sid_for_cmd = await client.session_new(
                            cwd=str(agent_cfg.home),
                        )
                    except Exception as exc:  # noqa: BLE001
                        await reply(
                            f"⚠️ couldn't open a session for /{parsed.name}: {exc}",
                        )
                        return
                    self._set_session_id(chat_id, sid_for_cmd, thread_id)
            if sid_for_cmd:
                from sharedagents_acp.client.client import AcpRemoteError
                from sharedagents_acp.protocol.jsonrpc import JsonRpcErrorCode

                fell_back = False
                try:
                    env = await client.session_command(
                        sid_for_cmd,
                        text,
                        frontend="telegram",
                    )
                except AcpRemoteError as exc:
                    # ONLY a genuine capability gap (old gateway) falls back to
                    # the legacy local path — never a command-level error, so a
                    # rejected command can't get a second divergent reading.
                    if getattr(exc, "code", None) != JsonRpcErrorCode.METHOD_NOT_FOUND:
                        await reply(f"⚠️ {exc}")
                        return
                    fell_back = True
                    env = None
                except Exception as exc:  # noqa: BLE001
                    await reply(f"⚠️ command failed: {exc}")
                    return
                if not fell_back:
                    out = self._render_command_result(
                        (env.get("result") or {}),
                    )
                    await reply(out)
                    return

        if parsed.name == "sessions":
            try:
                resp = await client.session_list()
            except Exception as exc:  # noqa: BLE001
                await reply(f"⚠️ session/list failed: {exc}")
                return
            entries = list(getattr(resp, "sessions", None) or [])
            if not entries:
                await reply("no sessions yet — send a message to start one.")
                return
            current_sid = self.get_session_id(chat_id, thread_id)

            def _entry_sid(entry: Any) -> str:
                return getattr(entry, "session_id", None) or getattr(entry, "sessionId", None) or ""

            def _entry_title(entry: Any) -> str:
                meta = (
                    getattr(entry, "field_meta", None)
                    or getattr(entry, "meta", None)
                    or getattr(entry, "_meta", None)
                    or {}
                )
                rt = (meta or {}).get("sharedagents", {}) if isinstance(meta, dict) else {}
                return getattr(entry, "title", None) or rt.get("title") or "(untitled)"

            displayed = entries[:20]

            arg = (parsed.arg or "").strip()
            if arg:
                target: Any | None = None
                if arg.isdigit():
                    idx = int(arg)
                    if 1 <= idx <= len(displayed):
                        target = displayed[idx - 1]
                if target is None:
                    matches = [e for e in entries if _entry_sid(e).startswith(arg)]
                    if len(matches) == 1:
                        target = matches[0]
                    elif len(matches) > 1:
                        await reply(
                            f"/sessions: ambiguous prefix {arg!r}, {len(matches)} matches — use a longer prefix"
                        )
                        return
                if target is None:
                    await reply(f"/sessions: no match for {arg!r}")
                    return
                new_sid = _entry_sid(target)
                if not new_sid:
                    await reply("/sessions: matched entry has no session_id")
                    return
                if new_sid == (current_sid or ""):
                    await reply(f"already on session [{new_sid[:12]}]")
                    return
                # Validate the session is loadable for this agent before
                # rebinding — surfaces a clear error instead of silently
                # writing a binding the next message will fail to use.
                try:
                    await client.session_load(new_sid, cwd=str(agent_cfg.home))
                except Exception as exc:  # noqa: BLE001
                    await reply(f"⚠️ session/load failed: {exc}")
                    return
                # Abort any in-flight turn so the rebind doesn't strand a
                # streaming response on the old session.
                await self.cancel_chat(chat_id, thread_id, via="session_switch")
                self._set_session_id(chat_id, new_sid, thread_id)
                title = _entry_title(target)
                await reply(f"switched to [{new_sid[:12]}] — {title}")
                return

            lines = ["saved sessions:"]
            for i, entry in enumerate(displayed, 1):
                sid = _entry_sid(entry) or "?"
                # The SDK's ``SessionInfo`` exposes ``_meta`` under the
                # python-side attr ``field_meta`` (pydantic alias). The
                # ``meta`` / ``_meta`` getattrs stay as forward-compat
                # fallbacks for SDKs that don't apply the alias rename.
                meta = (
                    getattr(entry, "field_meta", None)
                    or getattr(entry, "meta", None)
                    or getattr(entry, "_meta", None)
                    or {}
                )
                rt = (meta or {}).get("sharedagents", {}) if isinstance(meta, dict) else {}
                title = _entry_title(entry)
                turns = rt.get("turn_count") or 0
                updated_at = (
                    getattr(entry, "updated_at", None)
                    or getattr(entry, "updatedAt", None)
                    or rt.get("last_activity")
                    or ""
                )
                mark = "→ " if sid == current_sid else "  "
                age = _format_relative_age(updated_at)
                lines.append(f"{mark}{i}. {age} · {title} ({turns} turns)")
            lines.append("")
            lines.append("switch: /sessions <index|hash-prefix>")
            await reply("\n".join(lines))
            return

        if parsed.name == "history":
            sid = self.get_session_id(chat_id, thread_id)
            if sid is None:
                await reply("no session yet.")
                return
            try:
                n = int(parsed.arg) if parsed.arg.strip() else 5
            except ValueError:
                await reply(f"usage: /history [N]; got {parsed.arg!r}")
                return
            # ACP ``LoadSessionResponse`` carries ``modes`` and
            # ``configOptions`` only — turns are not on the wire. Use
            # ``session/load`` to verify the session is loadable on
            # the agent (so a stale binding surfaces an error here
            # instead of confusing "no turns yet"), then read the
            # persisted JSON directly. The agent's own ``Session.save``
            # is the authoritative writer for ``turns``; the bridge
            # never re-projects it through ACP.
            try:
                await client.session_load(sid, cwd=str(agent_cfg.home))
            except Exception as exc:  # noqa: BLE001
                await reply(f"⚠️ session/load failed: {exc}")
                return
            from sharedagents_session.session_reader import load_session_dict

            blob = load_session_dict(
                agent_cfg.sessions_dir, sid, expected_agent=str(getattr(agent_cfg, "name", "") or "")
            )
            if blob is None:
                await reply("⚠️ couldn't read session.")
                return
            all_turns = blob.get("turns") if isinstance(blob, dict) else None
            turns = list(all_turns or [])[-max(1, n) :]
            if not turns:
                await reply("no turns yet.")
                return
            lines = [f"last {len(turns)} turn(s):"]
            for i, t in enumerate(turns, 1):
                user = ""
                assistant = ""
                if isinstance(t, dict):
                    # Persisted turns use ``content`` for the model's
                    # reply (``Session.save`` shape); ``assistant`` is
                    # an older field-name kept here as a back-compat
                    # fallback so historic JSONs still render.
                    user = str(t.get("user") or "")[:80]
                    assistant = str(
                        t.get("content") or t.get("assistant") or "",
                    )[:80]
                lines.append(f"  {i}. ❯ {user}")
                if assistant:
                    lines.append(f"     ↳ {assistant}")
            await reply("\n".join(lines))
            return

        if parsed.name == "todo":
            sid = self.get_session_id(chat_id, thread_id)
            if sid is None:
                await reply(
                    "no session yet — todos appear once the model writes them via the `todo` tool.",
                )
                return
            # Same store Web UI / TUI / /todo command read — Cursor
            # Update TODOs is mirrored here by HarnessTaskMirror.
            try:
                from sharedagents_agent.tools import load_todos, todos_path_for_session

                stored = load_todos(todos_path_for_session(agent_cfg.sessions_dir, sid))
            except Exception:  # noqa: BLE001
                stored = []
            if stored:
                await reply(format_todo_lines(stored, heading="session todos:"))
                return
            try:
                resp = await client.session_load(sid, cwd=str(agent_cfg.home))
            except Exception as exc:  # noqa: BLE001
                await reply(f"⚠️ session/load failed: {exc}")
                return
            todos: list[Any] = []
            meta = getattr(resp, "meta", None) or getattr(resp, "_meta", None) or {}
            rt = (meta or {}).get("sharedagents", {}) if isinstance(meta, dict) else {}
            if isinstance(rt, dict):
                todos = list(rt.get("todos") or [])
            adapted_meta = adapt_todo_args({"todos": todos}) or adapt_todo_args({"items": todos})
            if adapted_meta and adapted_meta.get("todos"):
                await reply(
                    format_todo_lines(
                        adapted_meta["todos"],
                        heading="session todos:",
                    )
                )
                return
            # Fall back to scanning saved turns for any engine's todo
            # tool — Cursor Update TODOs, Claude TodoWrite, OpenCode
            # todowrite, native todo.
            turns = list(getattr(resp, "turns", None) or [])
            for t in reversed(turns):
                rounds = (t or {}).get("tool_rounds") if isinstance(t, dict) else None
                if not rounds:
                    continue
                for r in reversed(rounds):
                    for c in (r or {}).get("calls") or []:
                        cname = (c or {}).get("name")
                        if is_todo_tool_name(cname):
                            adapted = adapt_todo_args((c or {}).get("arguments"))
                            if adapted and adapted.get("todos"):
                                await reply(
                                    format_todo_lines(
                                        adapted["todos"],
                                        heading="session todos:",
                                    )
                                )
                                return
                        if cname == "todo" and c.get("ok"):
                            # The runner persists the rendered list
                            # in `result`; surface that verbatim.
                            content = c.get("result") or ""
                            if content:
                                # send_message splits over the cap, so
                                # pass the full list — no lossy pre-cut.
                                await reply(content)
                                return
            await reply(
                "no todos yet — the model adds them via the `todo` tool when planning multi-step work.",
            )
            return

        if parsed.name == "refresh":
            sid = self.get_session_id(chat_id, thread_id)
            if sid is None:
                await reply("no session yet — send a message to open one.")
                return
            try:
                resp = await client.session_load(sid, cwd=str(agent_cfg.home))
            except Exception as exc:  # noqa: BLE001
                await reply(f"⚠️ session/load failed: {exc}")
                return
            turns = list(getattr(resp, "turns", None) or [])
            await reply(
                f"🔄 synced — {len(turns)} turn(s) loaded for session {sid[:12]}…",
            )
            return

        if parsed.name == "context":
            from sharedagents_agent.context_report import (
                SessionTokens,
                build_context_report,
                format_context_detail,
                format_context_help,
                format_context_json,
                format_context_list,
            )
            from sharedagents_command.commands import normalize_context_subcommand

            sub = normalize_context_subcommand(parsed.arg)
            if not sub or sub == "help":
                await reply(format_context_help())
                return
            if sub not in {"list", "show", "detail", "deep", "json"}:
                await reply("Unknown /context mode.\nUse: /context, /context list, /context detail, or /context json")
                return

            sid = self.get_session_id(chat_id, thread_id)
            if sid is None:
                await reply("no session yet — send a message to open one.")
                return

            system_prompt = ""
            ctx_window = 0
            model = ""
            permission_mode = ""
            workspace_cwd = ""
            turns: list[dict[str, Any]] = []
            try:
                from sharedagents_session.session_reader import load_session_dict

                raw = load_session_dict(
                    agent_cfg.sessions_dir, sid, expected_agent=str(getattr(agent_cfg, "name", "") or "")
                )
                if isinstance(raw, dict):
                    turns = list(raw.get("turns") or [])
                    ctx_window = int(raw.get("context_window") or 0)
                    model = raw.get("model") or ""
                    system_prompt = raw.get("system_prompt") or ""
                    permission_mode = raw.get("system_mode") or ""
                    workspace_cwd = raw.get("workspace_cwd") or ""
            except Exception as exc:  # noqa: BLE001
                self._log.debug("/context disk read failed: %s", exc)

            latest: dict[str, Any] = {}
            for t in reversed(turns):
                m = t.get("metrics") if isinstance(t, dict) else None
                if isinstance(m, dict) and m:
                    latest = m
                    break

            tools_schemas: list[dict[str, Any]] = []
            tool_summaries: list[tuple[str, str]] = []
            deferred_catalog_text = ""
            skills_block_text = ""
            skills_list: list[Any] = []
            try:
                from sharedagents_agent.tools import BUILTIN_TOOLS, ToolRegistry

                registry = ToolRegistry(BUILTIN_TOOLS)
                tools_schemas = registry.openai_schemas()
                tool_summaries = [(t.name, t.description or "") for t in registry.tools]
                deferred_catalog_text = registry.deferred_catalog_block()
            except Exception as exc:  # noqa: BLE001
                self._log.debug("/context tool registry read failed: %s", exc)
            try:
                from sharedagents_skill.skills import (
                    discover_skills,
                    skills_summary_block,
                )

                skills_block_text = skills_summary_block(agent_cfg.home)
                skills_list = list(discover_skills(agent_cfg.home))
            except Exception as exc:  # noqa: BLE001
                self._log.debug("/context skills read failed: %s", exc)

            report = build_context_report(
                agent_home=agent_cfg.home,
                mode=permission_mode or "auto",
                system_prompt=system_prompt,
                tools_schemas=tools_schemas,
                tool_summaries=tool_summaries,
                deferred_catalog_text=deferred_catalog_text,
                skills_block_text=skills_block_text,
                skills=skills_list,
                model=model,
                workspace_cwd=workspace_cwd,
                permission_mode=permission_mode,
                source="run" if system_prompt else "estimate",
            )
            session_tok = SessionTokens(
                total_tokens=(
                    int(latest.get("prompt_tokens")) if isinstance(latest.get("prompt_tokens"), (int, float)) else None
                ),
                total_tokens_fresh=(
                    int(latest.get("prompt_tokens_fresh"))
                    if isinstance(latest.get("prompt_tokens_fresh"), (int, float))
                    else None
                ),
                input_tokens=(
                    int(latest.get("prompt_tokens")) if isinstance(latest.get("prompt_tokens"), (int, float)) else None
                ),
                output_tokens=(
                    int(latest.get("completion_tokens"))
                    if isinstance(latest.get("completion_tokens"), (int, float))
                    else None
                ),
                context_tokens=ctx_window or None,
            )

            if sub == "json":
                # Telegram MarkdownV2 escapes break JSON output — wrap in
                # a code fence so the structure renders verbatim.
                payload = format_context_json(report, session_tok)
                await reply(f"```\n{payload}\n```")
            elif sub in ("detail", "deep"):
                await reply(format_context_detail(report, session_tok))
            else:
                await reply(format_context_list(report, session_tok))
            return

        if parsed.name == "compact":
            sid = self.get_session_id(chat_id, thread_id)
            if sid is None:
                await reply("no session yet — nothing to compact.")
                return
            wrapped = "__SHAREDAGENTS_DELEGATE__:compact_context:{}"
            # Route through the same rolling-message pipeline a normal
            # text prompt uses so the user sees a placeholder, the
            # tool-call header, and the assistant's confirmation —
            # ``_dispatch_blocks`` alone returns the assembled string
            # into the void (same bug pattern as the original
            # /claude /codex handlers had).
            await self._stream_turn_for_chat(
                chat_id,
                text=wrapped,
                thread_id=thread_id,
            )
            return

        await reply(f"unknown command: {text}\nsend /help to list commands")

    async def _handle_delegate_command(
        self,
        chat_id: int,
        head: str,
        arg: str,
        thread_id: int | None = None,
    ) -> None:
        """Apply ``/claude <prompt>`` and ``/codex <prompt>`` on Telegram.

        Per CLAUDE.md "external bridges are tools, not slash commands":
        we synthesize a user prompt that biases the local model to
        emit a ``delegate_claude`` / ``delegate_codex`` tool call, then
        let the regular session_prompt stream surface the tool call +
        result. Same shape the TUI uses — one path, one runner, ESC
        kills via ``session/cancel`` like every other turn.
        """
        if self._api is None:
            return
        prompt = (arg or "").strip()
        if not prompt:
            await self._reply(chat_id, f"usage: {head} <prompt>", thread_id)
            return
        tool_name = {
            "/claude": "delegate_claude",
            "/codex": "delegate_codex",
            "/antigravity": "delegate_antigravity",
        }.get(head, "delegate_claude")
        # Direct dispatch sentinel — bridge intercepts and runs the
        # delegate tool without an LLM round. Route through the same
        # rolling-message turn pipeline a normal text prompt uses so
        # the user sees a placeholder, live tool deltas, stats footer
        # and a finalized reply — instead of _dispatch_blocks
        # returning an assembled string into the void.
        #
        # Auto-fill the pre-flight checklist the delegation gate
        # (sharedagents_agent.delegation_gate) requires for
        # an expensive frontier engine. An operator's `/claude <prompt>`
        # is a DIRECT HUMAN command — the operator IS the intent and the
        # authorization — so the FRONTEND fills the checklist here; it is
        # never settable by the model.
        from sharedagents_agent.operator_delegation import operator_slash_checklist

        args = {"prompt": prompt, **operator_slash_checklist(prompt)}
        marker = f"__SHAREDAGENTS_DELEGATE__:{tool_name}:{json.dumps(args)}"
        await self._stream_turn_for_chat(
            chat_id,
            text=marker,
            thread_id=thread_id,
        )

    async def _handle_agents_list(
        self,
        chat_id: int,
        thread_id: int | None = None,
    ) -> None:
        """``/agents`` — list configured agents with the chat's
        current selection marked.

        Walks ``~/.sharedagents/agents/*`` for every directory carrying a
        ``config.json``. The chat's CURRENT routed agent (per-thread
        override → bare-chat override → ``default_agent``) is shown
        with a ``▶`` marker so the operator sees where messages
        land right now.
        """
        agents = self._list_configured_agents()
        if not agents:
            await self._reply(
                chat_id,
                "no agents configured. Run `sharedagents agent init <name>` on the server to scaffold one.",
                thread_id,
            )
            return
        current = self.tg_cfg.agent_for(chat_id, thread_id)
        lines = ["agents:"]
        for name in agents:
            marker = "▶" if name == current else " "
            lines.append(f"  {marker} {name}")
        lines.append("")
        lines.append(
            f"current: {current} (use `/agent <name>` to switch — the session and transcript stay unchanged).",
        )
        await self._reply(chat_id, "\n".join(lines), thread_id)

    async def _handle_agent_switch(
        self,
        chat_id: int,
        name: str,
        thread_id: int | None = None,
    ) -> None:
        """Read or mutate the agent serving this session.

        Forms: ``/agent`` and ``/agent <name> [mode]``.
        """
        chat_key = TelegramConfig.chat_key(chat_id, thread_id)
        sid = self.get_session_id(chat_id, thread_id)
        if not sid:
            # First-run: no session exists yet for this chat/topic. The
            # operator MUST be able to choose the agent BEFORE the first
            # message — otherwise a fresh chat is locked to
            # ``default_agent`` until after the first turn (which defeats
            # the purpose of switching on a brand-new chat). Pin this
            # chat's route in ``agent_routes`` so the FIRST message
            # creates its session with the chosen agent instead of the
            # global default. ``agent_for`` consults the same map, and the
            # client factory resolves the agent through it, so the very
            # first turn lands on the requested agent. Persisted to
            # config.json so the pending switch survives a bot restart
            # before the first turn (the documented intent of
            # ``agent_routes`` — config.py notes the dynamic ``/agent``
            # command writes here).
            parts = (name or "").split()
            if len(parts) > 2:
                await self._reply(
                    chat_id,
                    "usage: /agent <name> [auto|ask|plan]",
                    thread_id,
                )
                return
            target = parts[0] if parts else ""
            if not target:
                current = self.tg_cfg.agent_for(chat_id, thread_id)
                await self._reply(
                    chat_id,
                    f"no conversation yet — current route: {current} "
                    "(use `/agent <name>` to set the agent for your "
                    "first message).",
                    thread_id,
                )
                return
            if not self._is_known_agent(target):
                await self._reply(
                    chat_id,
                    f"unknown agent: {target}",
                    thread_id,
                )
                return
            self.tg_cfg.agent_routes[chat_key] = target
            try:
                self.tg_cfg.save()
            except Exception:  # noqa: BLE001
                self._log.debug(
                    "agent_routes save failed chat_key=%s",
                    chat_key,
                )
            await self._reply(
                chat_id,
                f"✅ agent set: {target} — your next message starts a new session with this agent.",
                thread_id,
            )
            return
        if (
            chat_key in self._live_turns
            or self.chat_lock(
                chat_id,
                thread_id,
            ).locked()
        ):
            await self._reply(
                chat_id,
                "session_busy: wait for the active turn to finish or cancel it before switching.",
                thread_id,
            )
            return

        parts = (name or "").split()
        if len(parts) > 2:
            await self._reply(
                chat_id,
                "usage: /agent <name> [auto|ask|plan]",
                thread_id,
            )
            return
        target = parts[0] if parts else ""
        target_mode = parts[1] if len(parts) == 2 else None
        lock = self.chat_lock(chat_id, thread_id)
        async with lock:
            try:
                state = await self._get_session_state(sid)
                choices = await self._list_session_modes(sid)
            except Exception as exc:  # noqa: BLE001
                await self._reply(
                    chat_id,
                    "agent state unavailable: " + self._session_error_text(exc),
                    thread_id,
                )
                return
            await self._apply_session_state(chat_id, thread_id, sid, state)
            if not target:
                await self._reply(
                    chat_id,
                    self._render_telegram_agent_state(state, choices),
                    thread_id,
                )
                return
            prev = str(state.get("active_agent") or self.tg_cfg.agent_for(chat_id, thread_id))
            try:
                result = await self._switch_session_agent(
                    sid=sid,
                    target_agent=target,
                    target_mode=target_mode,
                )
            except Exception as exc:  # noqa: BLE001
                await self._reply(
                    chat_id,
                    "switch refused: " + self._session_error_text(exc),
                    thread_id,
                )
                return
            await self._apply_session_state(
                chat_id,
                thread_id,
                sid,
                result,
            )

        active = str(result.get("active_agent") or target)
        if active not in self._agent_cfg_cache:
            try:
                self._agent_cfg_cache[active] = AgentConfig.load(active)
            except Exception:  # noqa: BLE001
                pass
        mode = result.get("effective_mode") or {}
        if isinstance(mode, dict):
            mode_label = str(
                mode.get("label")
                or mode.get("id")
                or result.get("effective_mode_label")
                or result.get("effective_mode_id")
                or "Default"
            )
            behavior = str(mode.get("behavior_kind") or result.get("behavior_kind") or "")
        else:
            mode_label = str(mode or result.get("effective_mode_label") or "Default")
            behavior = str(result.get("behavior_kind") or "")
        await self._reply(
            chat_id,
            f"switched {prev} → {active} · {mode_label}" + (f" ({behavior})" if behavior else "") + ". "
            "Conversation and transcript preserved.",
            thread_id,
        )

    @staticmethod
    def _session_error_text(exc: Exception) -> str:
        data = getattr(exc, "data", None)
        if isinstance(data, dict):
            code = str(data.get("code") or "")
            message = str(data.get("message") or "")
            if code:
                return f"{code}: {message}" if message else code
        return str(exc) or type(exc).__name__

    @staticmethod
    def _render_telegram_agent_state(
        state: dict,
        choices: list[dict],
    ) -> str:
        mode = state.get("effective_mode") or {}
        if isinstance(mode, dict):
            label = str(
                mode.get("label")
                or mode.get("id")
                or state.get("effective_mode_label")
                or state.get("effective_mode_id")
                or "Default"
            )
            behavior = str(mode.get("behavior_kind") or state.get("behavior_kind") or "")
        else:
            label = str(mode or state.get("effective_mode_label") or "Default")
            behavior = str(state.get("behavior_kind") or "")
        lines = [
            f"agent: {state.get('active_agent') or '?'}",
            f"mode: {label}" + (f" ({behavior})" if behavior else ""),
            "available:",
        ]
        for row in choices:
            agent = str(row.get("agent_name") or row.get("name") or row.get("agent") or "")
            if not agent:
                continue
            modes = [
                str(item.get("id") or item.get("requested_mode_id") or "")
                for item in (row.get("modes") or row.get("availableModes") or [])
                if isinstance(item, dict)
                and (item.get("id") or item.get("requested_mode_id"))
                and item.get("supported", True)
            ]
            lines.append(f"  {agent}: {', '.join(modes) or 'default'}")
        return "\n".join(lines)

    async def _handle_agent_delegate(
        self,
        chat_id: int,
        agent_name: str,
        arg: str,
        thread_id: int | None = None,
    ) -> None:
        """``/<agent_name> <task>`` — one-shot delegate to ``<agent_name>``.

        Parallels ``/claude`` / ``/codex``: the chat stays on its
        currently-routed agent; we synthesize a user message that
        biases the local model to emit ``delegate_agent(name=...,
        task=...)``. The bridge intercepts the marker and runs the
        delegate tool without an LLM round, returning the spawned
        agent's reply as a tool result inline. Result is rendered
        through the same rolling-message pipeline the rest of the
        bot uses.
        """
        if self._api is None:
            return
        task = (arg or "").strip()
        if not task:
            await self._reply(
                chat_id,
                f"usage: /{agent_name} <task>",
                thread_id,
            )
            return
        # Auto-fill the pre-flight checklist the delegation gate
        # (sharedagents_agent.delegation_gate) requires for an
        # expensive frontier engine. A dynamic sibling slash `/mysibling <task>`
        # is a DIRECT HUMAN command exactly like `/claude` / `/codex` — the
        # operator IS the intent and the authorization — but unlike those paths
        # it previously shipped only name/task, so a sibling scaffolded on an
        # external engine hit an incomplete-checklist REFUSAL. Fill the checklist
        # here at the FRONTEND (never settable by the model) so the sibling path
        # rides the identical enforcement the `/claude` path does.
        from sharedagents_agent.operator_delegation import operator_slash_checklist

        args = {"name": agent_name, "task": task, **operator_slash_checklist(task)}
        marker = f"__SHAREDAGENTS_DELEGATE__:delegate_agent:{json.dumps(args)}"
        await self._stream_turn_for_chat(
            chat_id,
            text=marker,
            thread_id=thread_id,
        )

    async def _handle_config_command(
        self,
        chat_id: int,
        text: str,
        thread_id: int | None = None,
    ) -> None:
        """Apply a configuration slash command via the ACP wire.

        ``/mode <id>`` → ``session/set_mode``.
        ``/model <alias|url>`` → ``session/set_config_option``
        (option_id="model").
        ``/reasoning`` and ``/tools`` mutate per-(chat,thread) display
        state on the bot — the agent always emits
        ``agent_thought_chunk`` / ``tool_call_update`` over the ACP
        wire; the bot decides whether to render them inline. Mirrors
        the TUI's ``state_patch`` flow for the same toggles.
        ``/thinking`` is a model-side flag that only takes effect at
        ``session/new`` time and so just acks here.
        """
        from sharedagents_command.commands import (
            normalize_mode,
            normalize_reasoning_level,
            normalize_thinking,
            normalize_tool_output,
            parse_command,
        )

        if self._api is None:
            return

        async def reply(t: str) -> None:
            await self._reply(chat_id, t, thread_id)

        parsed = parse_command(text)
        if parsed is None:
            await reply(f"unknown command: {text}")
            return

        try:
            client = await self._get_or_open_client(chat_id, thread_id)
        except Exception as exc:  # noqa: BLE001
            self._log.exception(
                "config command client open failed chat=%s thread=%s",
                chat_id,
                thread_id,
            )
            await reply(f"⚠️ agent unreachable: {exc}")
            return

        agent_cfg = self._agent_cfg(chat_id, thread_id)
        sid = self.get_session_id(chat_id, thread_id)
        if sid is None:
            # Auto-open a fresh session so config commands like /mode,
            # /model, /reasoning, /tools work straight after /new (or
            # on a brand-new chat). Previous behaviour was to reject
            # with "no session yet — send a message first…", which made
            # the natural sequence "/new → /reasoning stream → prompt"
            # fail on step 2.
            try:
                sid = await client.session_new(cwd=str(agent_cfg.home))
            except Exception as exc:  # noqa: BLE001
                self._log.exception(
                    "auto-open session for config command failed chat=%s",
                    chat_id,
                )
                await reply(f"⚠️ couldn't open a new session: {exc}")
                return
            self._set_session_id(chat_id, sid, thread_id)

        if parsed.name == "mode":
            level = normalize_mode(parsed.arg)
            if not parsed.arg:
                await reply("usage: /mode [plan|ask|auto]")
                return
            if level is None:
                await reply(
                    f"unknown mode {parsed.arg!r}. use plan, ask, or auto",
                )
                return
            try:
                # Declare the telegram frontend so the server capability check
                # sees an ask-capable origin.
                await client.session_set_mode(sid, level, frontend="telegram")
            except Exception as exc:  # noqa: BLE001
                await reply(f"⚠️ set_mode failed: {exc}")
                return
            await reply(f"mode → {level}")
            return

        if parsed.name == "model":
            if not parsed.arg:
                await reply("usage: /model <alias|url>")
                return
            # Route through the shared server-side runtime so /model is gated
            # identically to every other frontend — including the turn-active
            # refusal and typo/external-id resolution
            # Instead of a raw set_config_option that
            # bypasses run_command.
            try:
                env = await client.session_command(
                    sid,
                    text,
                    frontend="telegram",
                )
            except Exception as exc:  # noqa: BLE001
                await reply(f"⚠️ model switch failed: {exc}")
                return
            out = self._render_command_result(env.get("result") or {})
            await reply(out)
            return

        if parsed.name == "effort":
            # Route through the shared runtime so Telegram receives the same
            # validation, persistence, state_patch, and error envelope as
            # the TUI and WebUI.
            try:
                env = await client.session_command(
                    sid,
                    text,
                    frontend="telegram",
                )
            except Exception as exc:  # noqa: BLE001
                await reply(f"⚠️ effort set failed: {exc}")
                return
            await reply(self._render_command_result(env.get("result") or {}))
            return

        if parsed.name == "reasoning":
            level = normalize_reasoning_level(parsed.arg) if parsed.arg else None
            if not parsed.arg:
                cur = self._get_visibility(chat_id, thread_id).get(
                    "reasoning_visibility",
                    "off",
                )
                await reply(
                    f"reasoning is {cur} — usage: /reasoning [on|off|stream]",
                )
                return
            if level is None:
                await reply(
                    f"unknown reasoning level {parsed.arg!r}. use on, off, or stream",
                )
                return
            self._set_reasoning_visibility(chat_id, thread_id, level)
            # Push to the agent: the engine's per-session gate decides
            # whether ``thinking`` deltas reach Telegram at all (defense
            # in depth — Telegram also gates client-side via the per-chat
            # ``reasoning_visibility`` map). Without this, flipping to
            # "on" only changes the local rendering policy for a stream
            # that never carries any thinking events.
            try:
                await client.session_set_config_option(sid, "reasoning", level)
            except Exception as exc:  # noqa: BLE001
                await reply(f"⚠️ reasoning push to agent failed: {exc}")
                return
            await reply(f"reasoning → {level}")
            return

        if parsed.name == "thinking":
            val = normalize_thinking(parsed.arg) if parsed.arg else None
            if not parsed.arg:
                await reply("usage: /thinking [on|off]")
                return
            if val is None:
                await reply(
                    f"unknown thinking value {parsed.arg!r}. use on or off",
                )
                return
            await reply(
                f"thinking → {'on' if val else 'off'} (model-side flag; "
                "applies to the next session via the agent runtime)",
            )
            return

        if parsed.name == "tool":
            val = normalize_tool_output(parsed.arg) if parsed.arg else None
            if not parsed.arg:
                cur = self._get_visibility(chat_id, thread_id).get(
                    "tool_output",
                    "off",
                )
                await reply(
                    f"tools is {cur} — usage: /tools [on|off|stream]\n"
                    "  off    — hide tool calls and results\n"
                    "  on     — show tool-call header only "
                    "(🔧 web_search…)\n"
                    "  stream — show header + raw tool output",
                )
                return
            if val is None:
                await reply(
                    f"unknown tool visibility {parsed.arg!r}. use on, off, or stream",
                )
                return
            self._set_tool_output_level(chat_id, thread_id, val)
            await reply(f"tools → {val}")
            return

        if parsed.name == "models":
            from sharedagents_command.commands import (
                engine_model_rows,
                format_model_list,
                local_model_rows,
            )

            # Catalogue follows the agent's configured engine, not a leftover
            # session header. A conversation founded on OpenCode keeps
            # ``engine: opencode_acp`` on the founding session id after
            # ``/agent jarvis`` / ``/agent cursor``; using that header here
            # listed OpenCode models under those agents.
            cur_endpoint = ""
            cur_model = ""
            read_failed = False
            try:
                from sharedagents_session.session_reader import load_session_dict

                raw = load_session_dict(
                    agent_cfg.sessions_dir,
                    sid,
                    expected_agent=str(getattr(agent_cfg, "name", "") or ""),
                )
                if isinstance(raw, dict):
                    cur_model = raw.get("engine_model") or raw.get("model") or ""
                    cur_endpoint = raw.get("endpoint") or ""
            except Exception as exc:  # noqa: BLE001
                read_failed = True
                self._log.debug("/models session read failed: %s", exc)
            _engine_rows = engine_model_rows(agent_cfg)
            _engine_owned = _engine_rows is not None
            rows = _engine_rows if _engine_owned else local_model_rows(agent_cfg)
            if not rows and not _engine_owned:
                await reply(
                    "no models configured — add some via "
                    "`sharedagents agent setup` or by editing "
                    f"{agent_cfg.config_path}",
                )
                return
            if read_failed:
                text_out = format_model_list(rows)
                await reply(text_out + "\n⚠️ couldn't read session state — " + "current model unknown")
                return
            # Fall back to the global default ONLY when the session header
            # carries nothing yet (e.g. first turn not run) — that IS what
            # the session will use when its first prompt runs. For
            # engine-owned models (external CLI/ACP) the global
            # ``upstream_*`` belongs to a different deployment, so DON'T
            # assert it as "current".
            if not _engine_owned:
                if not cur_model:
                    cur_model = getattr(agent_cfg, "upstream_model", "") or ""
                if not cur_endpoint:
                    cur_endpoint = getattr(agent_cfg, "upstream_endpoint", "") or ""
            from sharedagents_command.commands import resolve_session_model

            text_out = format_model_list(
                rows,
                current_endpoint=cur_endpoint,
                current_model=resolve_session_model(cur_model, agent_cfg),
            )
            # send_message rolls anything over the cap into additional
            # messages, so no lossy pre-truncation here.
            await reply(text_out)
            return

        await reply(f"unknown command: {text}\nsend /help to list commands")


# ---------------------------------------------------------------------------
# Shared helpers/constants owned by ``bot.py`` (imported at BOTTOM to keep the
# bot<->mixin import cycle deadlock-free — see module docstring).
# ---------------------------------------------------------------------------
from sharedagents_telegram.bot import (  # noqa: E402
    AgentConfig,
    TelegramConfig,
    _cold_external_engine_model,
    _format_relative_age,
)
