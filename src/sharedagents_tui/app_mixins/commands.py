"""AgentChatApp slash-command handlers (_cmd_*).

Extracted from app.py (de-monolith); behavior-preserving.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from textual.containers import VerticalScroll
from textual.widgets import Static

from sharedagents_tui.command_render import (
    card_title,
    confirm_copy_for,
    format_kind_body,
)
from sharedagents_tui.helpers import (
    _engine_models_for,
    _model_alias_id_label,
)
from sharedagents_tui.widgets.messages import (
    CommandResultMessage,
    ContextMessage,
    UserMessage,
)
from sharedagents_tui.widgets.modals import (
    AgentMonitorModal,
    ConfirmModal,
    ModelPickerModal,
    MonitorPanelModal,
)


class CommandsMixin:
    """AgentChatApp slash-command handlers (_cmd_*)."""

    async def _cmd_new(self) -> None:
        if self._current_turn_id is not None:
            await self.action_abort(via="new")
        # Mirror /restart's child sweep: action_abort only cancels the
        # *tracked* turn; parallel batch runners, mid-flight subagents,
        # or anything whose registration was already cleared can survive
        # and keep writing to the just-discarded session. Reap them
        # before the new session starts so the first prompt against the
        # fresh /new doesn't race orphaned children of the prior turn.
        # Idempotent: a clean gateway with no children is a no-op.
        self._sweep_gateway_children(reason="new")
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.remove_children()
        # Force a fresh session by clearing the current id, then go through
        # _ensure_session so we get the same retry/health-render behaviour.
        self._sid = None
        # The /claude bridge is per-sharedagents-session: a new session means
        # a new Claude --resume thread (no prior id).
        self._claude_session_id = None
        # /codex bridge is also per-sharedagents-session — same reset rule.
        self._codex_thread_id = None
        # Reset context tracking so the status bar doesn't show stale
        # token counts from the previous session.
        self._last_prompt_tokens = None
        self._last_context_window = None
        # Clear the persistent plan-checklist so the bottom-of-TUI
        # row doesn't carry over stale steps from the prior session.
        self.clear_plan_checklist()
        # Same treatment for the harness-task footer: a new session
        # starts at task #1 again, and any stale rows from the prior
        # session would be confusing.
        self._clear_harness_tasks()
        # A fresh session has no goal and no teacher activity yet — clear the
        # goal/teacher widget so it doesn't carry the prior session's goal or
        # per-session ``/teacher`` override.
        self._active_goal = ""
        self._teacher_override = ""
        self._teacher_model_override = ""
        self._teacher_reviewing = False
        self._teacher_review_cids.clear()
        self._recompute_teacher()
        if not await self._ensure_session(force_new=True):
            return
        self._set_status(f"new session [{self.session.hash}]")

    async def _cmd_agents(self) -> None:
        """``/agents`` — list configured agents under
        ``~/.sharedagents/agents/<name>/config.json`` with a marker for the
        currently-active one. Use ``/agent <name>`` to switch.
        """
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage("❯ /agents"))
        agents_root = Path.home() / ".sharedagents" / "agents"
        if not agents_root.is_dir():
            await transcript.mount(
                Static(
                    "no agents configured at ~/.sharedagents/agents/",
                    classes="claude-msg",
                )
            )
            return
        rows: list[str] = []
        current = self.agent_cfg.name
        for entry in sorted(agents_root.iterdir()):
            if not entry.is_dir():
                continue
            if not (entry / "config.json").is_file():
                continue
            mark = "●" if entry.name == current else " "
            # Pull a friendly description: engine kind + upstream
            # alias for local-LLM agents, or just the engine kind
            # tag for CLI-engine agents. Cheap read, never blocks.
            engine_kind = "local_llm"
            try:
                from sharedagents_agent.agent import AgentConfig

                cfg = AgentConfig.load(entry.name)
                engine_kind = getattr(cfg, "engine", "") or "local_llm"
                if engine_kind == "local_llm":
                    hint = getattr(cfg, "upstream_alias", "") or getattr(cfg, "upstream_endpoint", "") or "—"
                else:
                    # CLI-engine agents (claude_cli / codex_cli /
                    # antigravity_cli) — show the engine tag, not an
                    # upstream URL.
                    hint = f"engine={engine_kind}"
            except Exception:  # noqa: BLE001
                hint = "—"
            rows.append(f"  [bold]{mark}[/bold]  {entry.name:<16}  [dim]{hint}[/dim]")
        if not rows:
            await transcript.mount(
                Static(
                    "no agents found (no <name>/config.json directories)",
                    classes="claude-msg",
                )
            )
            return
        # Mix of Rich-markup chrome (bold/dim) + plain text — render
        # via ``Static(markup=True)``, NOT RichMarkdown, since Markdown
        # passes through ``[bold]...[/bold]`` as literal text.
        body = (
            f"[bold]configured agents[/bold]  "
            f"(current: [bold cyan]{current}[/bold cyan]; "
            f"[dim]/agent <name> to switch[/dim])\n\n" + "\n".join(rows)
        )
        await transcript.mount(
            Static(
                body,
                classes="claude-msg",
                markup=True,
            )
        )

    async def _cmd_task(self, arg: str = "") -> None:
        """``/task <verb> [args]`` — operator-side steering of the
        harness-task footer + on-disk task store. Lets the operator
        add / mark / delete tasks directly without going through the
        model.

        Verbs:
          ``/task add <subject>``            create
          ``/task in <N>``                   flip N → in_progress
          ``/task done <N>``                 flip N → completed
          ``/task del <N>``                  remove N
          ``/task list``  (or just /task)    print current
          ``/task help``                     this help
        """
        from rich.markup import escape as _esc

        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage("❯ /task" + (f" {arg}" if arg else "")))
        sid = getattr(self, "_sid", None)
        if not sid:
            await transcript.mount(
                Static(
                    "no active session — start one before /task",
                    classes="claude-msg",
                )
            )
            return

        # Resolve the on-disk task-store path. Mirrors the runner's
        # ``_tasks_path_from_env`` resolution but uses the TUI's
        # agent config (which knows ``sessions_dir`` directly).
        from pathlib import Path as _Path

        sessions_dir = _Path(self.agent_cfg.sessions_dir)
        task_path = sessions_dir / f"{sid}.tasks.json"

        def _load() -> dict:
            if not task_path.exists():
                return {"next_id": 1, "tasks": {}}
            try:
                d = json.loads(task_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return {"next_id": 1, "tasks": {}}
            if not isinstance(d, dict):
                return {"next_id": 1, "tasks": {}}
            d.setdefault("tasks", {})
            d.setdefault(
                "next_id",
                max(
                    (int(k) for k in d["tasks"] if str(k).isdigit()),
                    default=0,
                )
                + 1,
            )
            return d

        def _save(state: dict) -> None:
            task_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = task_path.with_name(f".{task_path.name}.tmp")
            tmp.write_text(
                json.dumps(state, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp.replace(task_path)

        def _render_list() -> str:
            rows: list[str] = ["[bold blue]📋 Tasks[/bold blue]"]
            for tid in self._harness_tasks_order:
                r = self._harness_tasks.get(tid)
                if not r:
                    continue
                status = r.get("status") or "pending"
                subj = (r.get("subject") or "").strip()
                icon = {
                    "pending": "[dim]☐[/dim]",
                    "in_progress": "[bold cyan]▶[/bold cyan]",
                    "completed": "[dim strike]☑[/dim strike]",
                    "deleted": "[red]✗[/red]",
                }.get(status, "?")
                if status in ("completed", "deleted"):
                    rows.append(f"  {icon} [dim strike]#{tid} {_esc(subj)}[/dim strike]")
                else:
                    rows.append(f"  {icon} [bold]#{tid}[/bold] {_esc(subj)}")
            return "\n".join(rows) if len(rows) > 1 else "[dim](no tasks)[/dim]"

        parts = arg.split(None, 1) if arg else []
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        if verb in ("", "list", "ls"):
            await transcript.mount(
                Static(
                    _render_list(),
                    markup=True,
                    classes="claude-msg",
                )
            )
            return

        if verb in ("help", "?", "h"):
            await transcript.mount(
                Static(
                    "[bold]/task[/bold] — operator-side task steering\n"
                    "  [bold]/task add[/bold] <subject>      append a new task\n"
                    "  [bold]/task in[/bold] <N>             flip N → in_progress\n"
                    "  [bold]/task done[/bold] <N>           flip N → completed\n"
                    "  [bold]/task del[/bold] <N>            delete task N\n"
                    "  [bold]/task list[/bold]               print tasks (default)\n"
                    "[dim]Changes are mirrored to "
                    f"{_esc(str(task_path))} so the model's TaskList call "
                    "sees them on the next round.[/dim]",
                    classes="claude-msg",
                    markup=True,
                )
            )
            return

        if verb == "add":
            subject = rest.strip()
            if not subject:
                await transcript.mount(
                    Static(
                        "usage: /task add <subject>",
                        classes="claude-msg",
                    )
                )
                return
            state = _load()
            tid = str(state["next_id"])
            state["next_id"] = int(tid) + 1
            state["tasks"][tid] = {
                "id": int(tid),
                "subject": subject,
                "description": "",
                "activeForm": "",
                "status": "pending",
                "owner": "operator",
                "blocks": [],
                "blockedBy": [],
                "metadata": {"source": "operator"},
            }
            try:
                _save(state)
            except OSError as e:
                await transcript.mount(
                    Static(
                        f"failed to write {task_path}: {e}",
                        classes="claude-msg",
                    )
                )
                return
            # Mirror into the footer state so the operator sees it
            # immediately without waiting for the next TaskList round.
            self._upsert_harness_task(
                tid,
                subject=subject,
                status="pending",
            )
            await transcript.mount(
                Static(
                    f"[green]✓[/green] Task #{tid} added: {_esc(subject)}\n{_render_list()}",
                    classes="claude-msg",
                    markup=True,
                )
            )
            return

        if verb in ("in", "in-progress", "start", "claim"):
            tid = rest.strip().lstrip("#")
            if not tid.isdigit():
                await transcript.mount(
                    Static(
                        "usage: /task in <N>",
                        classes="claude-msg",
                    )
                )
                return
            state = _load()
            t = state["tasks"].get(tid)
            if t is None:
                await transcript.mount(
                    Static(
                        f"unknown task #{tid}",
                        classes="claude-msg",
                    )
                )
                return
            t["status"] = "in_progress"
            _save(state)
            self._upsert_harness_task(tid, status="in_progress")
            await transcript.mount(
                Static(
                    f"[cyan]▶[/cyan] Task #{tid} → in_progress\n{_render_list()}",
                    classes="claude-msg",
                    markup=True,
                )
            )
            return

        if verb in ("done", "complete", "finish"):
            tid = rest.strip().lstrip("#")
            if not tid.isdigit():
                await transcript.mount(
                    Static(
                        "usage: /task done <N>",
                        classes="claude-msg",
                    )
                )
                return
            state = _load()
            t = state["tasks"].get(tid)
            if t is None:
                await transcript.mount(
                    Static(
                        f"unknown task #{tid}",
                        classes="claude-msg",
                    )
                )
                return
            t["status"] = "completed"
            _save(state)
            self._upsert_harness_task(tid, status="completed")
            await transcript.mount(
                Static(
                    f"[green]☑[/green] Task #{tid} → completed\n{_render_list()}",
                    classes="claude-msg",
                    markup=True,
                )
            )
            return

        if verb in ("del", "delete", "rm", "drop"):
            tid = rest.strip().lstrip("#")
            if not tid.isdigit():
                await transcript.mount(
                    Static(
                        "usage: /task del <N>",
                        classes="claude-msg",
                    )
                )
                return
            state = _load()
            if tid not in state["tasks"]:
                await transcript.mount(
                    Static(
                        f"unknown task #{tid}",
                        classes="claude-msg",
                    )
                )
                return
            del state["tasks"][tid]
            _save(state)
            # Also drop from the footer state.
            self._harness_tasks.pop(tid, None)
            try:
                self._harness_tasks_order.remove(tid)
            except ValueError:
                pass
            self._render_harness_tasks()
            await transcript.mount(
                Static(
                    f"[red]✗[/red] Task #{tid} deleted\n{_render_list()}",
                    classes="claude-msg",
                    markup=True,
                )
            )
            return

        await transcript.mount(
            Static(
                f"unknown /task verb: {_esc(verb)}  (try /task help)",
                classes="claude-msg",
                markup=True,
            )
        )

    async def _cmd_monitors(self) -> None:
        """``/monitors`` — open the live monitor panel.

        Pushes ``MonitorPanelModal``: a navigable list of every
        monitor for the active session, refreshed every 1.5 s, with
        ``x`` to SIGTERM the selected monitor's PID and ``q`` /
        ``Esc`` to dismiss. Reads the same sidecar (``<sid>.monitors.json``)
        the runner code writes, so the panel and ``monitor_list``
        stay in sync.

        For non-interactive contexts (scripts, log review),
        ``monitor_list`` printed in a tool round gives the same data
        as a one-shot text dump.
        """
        from pathlib import Path as _Path

        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage("❯ /monitors"))
        sid = getattr(self, "_sid", None)
        if not sid:
            await transcript.mount(
                Static(
                    "no active session — start one before opening the monitor panel",
                    classes="claude-msg",
                )
            )
            return
        sessions_dir = _Path(self.agent_cfg.sessions_dir)
        await self.push_screen(MonitorPanelModal(sessions_dir, sid))

    async def _cmd_monitor(self) -> None:
        """``/monitor`` — open the cross-agent agent monitor modal.

        The TUI port of the web ``/agents`` page: a live agent → session
        → subagent tree spanning EVERY agent, so the operator can
        navigate to ANY session, not just ones spawned on this one (the
        legacy ``/subagents`` panel is single-session). Picking a session
        switches the TUI onto it — hopping to the session's owning agent
        first when it differs from the current one, so a continued turn
        runs on the right engine. Because the session store is unified
        (``layout.sessions_root``), the replayed history loads regardless
        of which agent created the session.
        """
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage("❯ /monitor"))
        if self._client is None:
            await transcript.mount(
                Static(
                    "agent not connected — can't open the monitor",
                    classes="claude-msg",
                )
            )
            return
        picked = await self.push_screen_wait(AgentMonitorModal())
        if not picked or not isinstance(picked, dict):
            return  # cancelled
        new_sid = (picked.get("hash") or "").strip()
        if not new_sid:
            return
        target_agent = (picked.get("agent") or "").strip()
        title = (picked.get("title") or "").strip()
        # Hop to the session's owning agent first when it's a known,
        # different sharedagents agent — so the next turn continues on the
        # right engine. Unknown / external / same agent → switch session
        # in place (the unified store still hydrates the history).
        cur = getattr(self.agent_cfg, "name", "") or ""
        if target_agent and target_agent != cur:
            from sharedagents_agent.agent import AgentConfig

            try:
                AgentConfig.load(target_agent)
                known = True
            except Exception:  # noqa: BLE001
                known = False
            if known:
                await self._cmd_agent(target_agent)
        await self._switch_to_session_hash(new_sid, title=title)

    async def _cmd_skill(self, arg: str = "") -> None:
        """``/skills`` — list installed skills; ``/skill <name>`` — show
        one skill's SKILL.md body.

        Skills are the agent's on-demand runbooks (the ``skill_get``
        tool). This is the operator-facing view of the same discovery
        chain the agent sees — per-agent → ``~/.sharedagents/skills`` →
        config ``skills_dirs`` bundles → builtin — so it doubles as a
        quick check that an everything-claude-code / OpenClaw bundle
        registered. Disabled skills (``skills`` map in the global
        config) show with a ``✗`` marker.
        """
        from sharedagents_skill import skills as _skills_mod

        transcript = self.query_one("#transcript", VerticalScroll)
        arg = (arg or "").strip()
        await transcript.mount(UserMessage("❯ /skill " + arg if arg else "❯ /skills"))
        if arg:
            skill = _skills_mod.find_skill(arg)
            if skill is None:
                await transcript.mount(
                    Static(
                        f"skill not found: {arg!r} — /skills to list what's installed",
                        classes="claude-msg",
                    )
                )
                return
            lines = [f"# {skill.name}", skill.short_description, ""]
            if skill.version:
                lines.append(f"version: {skill.version}")
            if skill.requires_env:
                lines.append(f"requires env: {', '.join(skill.requires_env)}")
            if skill.requires_bins:
                lines.append(f"requires bins: {', '.join(skill.requires_bins)}")
            lines.append(f"source: {skill.source}")
            lines.append("")
            lines.append("─" * 40)
            lines.append(skill.body or "(SKILL.md has frontmatter only — no body)")
            await transcript.mount(Static("\n".join(lines), classes="claude-msg"))
            return
        skills = _skills_mod.discover_skills(include_disabled=True)
        if not skills:
            await transcript.mount(
                Static(
                    "no skills installed — add a dir to `skills_dirs` in "
                    "~/.sharedagents/config.json, or drop one in ~/.sharedagents/skills/<name>/",
                    classes="claude-msg",
                )
            )
            return
        enabled = {s.name for s in _skills_mod.discover_skills()}
        lines = [f"{len(skills)} skill(s)  ·  /skill <name> to read one", ""]
        for s in skills:
            mark = "  " if s.name in enabled else "✗ "
            lines.append(f"{mark}{s.name:<30s} {s.short_description}")
        if len(enabled) < len(skills):
            lines.append("")
            lines.append("✗ = disabled via the `skills` map in ~/.sharedagents/config.json")
        await transcript.mount(Static("\n".join(lines), classes="claude-msg"))

    async def _cmd_plugin(self, arg: str = "") -> None:
        """``/plugin list|install|enable|disable|remove`` — manage plugin
        bundles (one dir packaging skills + commands + agents + hooks
        as a unit). Installed plugins live at ``~/.sharedagents/plugins/``
        and are discovered automatically — disabled ones use a
        ``.disabled`` suffix.
        """
        from sharedagents_agent import plugins as _plug

        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(
            UserMessage(
                "❯ /plugin" + (f" {arg}" if arg else " list"),
            )
        )
        parts = (arg or "list").strip().split(None, 1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""

        def _say(text: str) -> None:
            # Closure: mount a Static into the transcript.
            import asyncio as _aio

            _aio.create_task(
                transcript.mount(
                    Static(text, classes="claude-msg"),
                )
            )

        if sub == "list":
            items = _plug.list_plugins()
            if not items:
                await transcript.mount(
                    Static(
                        "no plugins installed — try `/plugin search <query>` then `/plugin install <name-or-git-url>`",
                        classes="claude-msg",
                    )
                )
                return
            lines = [f"{len(items)} plugin(s) at ~/.sharedagents/plugins/", ""]
            for p in items:
                mark = "  " if p.enabled else "✗ "
                comps = p.components()
                summary = ", ".join(f"{k}={v}" for k, v in comps.items()) or "(empty)"
                head = f"{mark}{p.name:<20s} {p.version or '-':<10s} {summary}"
                if p.description:
                    head += f"  — {p.description[:60]}"
                lines.append(head)
                if p.source:
                    lines.append(f"     source: {p.source}")
            if any(not p.enabled for p in items):
                lines.append("")
                lines.append("✗ = disabled (rename drops `.disabled` to re-enable)")
            await transcript.mount(Static("\n".join(lines), classes="claude-msg"))
            return

        if sub == "search":
            # ``/plugin search [query]`` — substring-rank the curated
            # marketplace registry (sharedagents_agent.plugins.search). Empty
            # query lists the full registry sorted by name. Mirrors
            # the Telegram bot's parity path so a plugin discovered on
            # one frontend is installable from the other.
            results = _plug.search(rest or "")
            if not results:
                await transcript.mount(
                    Static(
                        f"no marketplace matches for {rest!r}\n"
                        "tip: try a broader keyword like `skill`, `hook`, `mcp`, or "
                        "drop ~/.sharedagents/plugin-marketplace.json with your own entries",
                        classes="claude-msg",
                    )
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
            await transcript.mount(Static("\n".join(lines), classes="claude-msg"))
            return

        if sub == "install":
            if not rest:
                await transcript.mount(
                    Static(
                        "usage: /plugin install <git-url|local-path> [--ref <branch>]",
                        classes="claude-msg",
                    )
                )
                return
            src = rest
            ref: str = ""
            if " --ref " in src:
                src, _, ref = src.partition(" --ref ")
                src, ref = src.strip(), ref.strip()
            try:
                p = _plug.install(src.strip(), ref=ref or None)
            except Exception as exc:  # noqa: BLE001
                await transcript.mount(
                    Static(
                        f"install failed: {exc}",
                        classes="claude-msg",
                    )
                )
                return
            comps = (
                ", ".join(f"{k}={v}" for k, v in p.components().items()) or "(no skills/commands/agents/hooks detected)"
            )
            await transcript.mount(
                Static(
                    f"installed {p.name} -> {p.path}\n  components: {comps}\n"
                    "  (skills + commands are discovered automatically)",
                    classes="claude-msg",
                )
            )
            return

        if sub in ("enable", "disable"):
            if not rest:
                await transcript.mount(
                    Static(
                        f"usage: /plugin {sub} <name>",
                        classes="claude-msg",
                    )
                )
                return
            try:
                fn = _plug.enable if sub == "enable" else _plug.disable
                p = fn(rest)
            except Exception as exc:  # noqa: BLE001
                await transcript.mount(
                    Static(
                        f"{sub} failed: {exc}",
                        classes="claude-msg",
                    )
                )
                return
            await transcript.mount(
                Static(
                    f"{p.name}: {'enabled' if p.enabled else 'disabled'}",
                    classes="claude-msg",
                )
            )
            return

        if sub == "remove":
            if not rest:
                await transcript.mount(
                    Static(
                        "usage: /plugin remove <name> [--purge]",
                        classes="claude-msg",
                    )
                )
                return
            purge = False
            name = rest
            if " --purge" in name or name.endswith(" --purge"):
                name = name.replace(" --purge", "").strip()
                purge = True
            try:
                where = _plug.remove(name, purge=purge)
            except Exception as exc:  # noqa: BLE001
                await transcript.mount(
                    Static(
                        f"remove failed: {exc}",
                        classes="claude-msg",
                    )
                )
                return
            await transcript.mount(
                Static(
                    f"removed {name} -> {where}" + ("" if purge else " (in .trash/; can be restored)"),
                    classes="claude-msg",
                )
            )
            return

        await transcript.mount(
            Static(
                f"unknown /plugin subcommand: {sub!r}\n"
                "usage: /plugin {list|search|install|enable|disable|remove} [args]",
                classes="claude-msg",
            )
        )

    async def _cmd_mcp(self, arg: str = "") -> None:
        """``/mcp`` — list every MCP server visible to the agent, with
        its transport and discovery source.

        Mirrors Claude Code's ``/mcp`` command (CLAUDE_CODE_FEATURE_MAP
        sec 1). Reads from ``sharedagents_mcp.list_mcp_servers_with_source``
        which folds three sources in precedence order:

          1. ``~/.sharedagents/config.json`` → ``mcp_servers`` (legacy /
             snake_case) or ``mcpServers`` (Claude Code spec). Tagged
             ``[config]``.
          2. ``~/.claude/.mcp.json`` (Claude Code's user-level MCP
             file). Tagged ``[claude]``.
          3. Each enabled plugin bundle's ``.mcp.json``. Tagged
             ``[plugin]``.

        A filter argument scopes by substring match against name +
        transport + description: ``/mcp gitea`` lists only servers
        whose name or description mentions "gitea".
        """
        try:
            transcript = self.query_one("#chat-transcript")
        except Exception:  # noqa: BLE001
            return
        await transcript.mount(
            Static(
                "❯ /mcp" + (f" {arg}" if arg else ""),
                classes="user-msg",
            )
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
                msg = f"no MCP servers match {arg!r}.\nconfigured servers can be listed with `/mcp` (no argument)."
            else:
                msg = (
                    "no MCP servers configured.\n"
                    "tip: add one in ~/.sharedagents/config.json under "
                    "`mcpServers`, drop ~/.claude/.mcp.json (Claude "
                    "Code spec), or install a plugin bundle that "
                    "ships an `.mcp.json`."
                )
            await transcript.mount(Static(msg, classes="claude-msg"))
            return
        lines = [f"{len(servers)} MCP server(s):", ""]
        for server, src in servers:
            badge = f"[{src}]"
            head = f"  {server.name:<24s} {server.transport:<5s} {badge}"
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
        await transcript.mount(Static("\n".join(lines), classes="claude-msg"))

    async def _cmd_commands(self, arg: str = "") -> None:
        """``/commands`` — list file-based slash commands (Claude Code
        ``commands/*.md``, including everything-claude-code). Each
        ``/<name> [args]`` expands that command's prompt body into a
        normal user turn — type one to run it.
        """
        from sharedagents_command import file_commands as _fc

        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage("❯ /commands"))
        cmds = _fc.discover_commands()
        if not cmds:
            await transcript.mount(
                Static(
                    "no file commands installed — add a dir to `commands_dirs` "
                    "in ~/.sharedagents/config.json, or drop one in "
                    "~/.sharedagents/commands/<name>.md",
                    classes="claude-msg",
                )
            )
            return
        lines = [
            f"{len(cmds)} command(s)  ·  type /<name> [args] to run one",
            "",
        ]
        for c in cmds:
            hint = f" {c.argument_hint}" if c.argument_hint else ""
            lines.append(f"  /{c.name}{hint}".ljust(32) + f" {c.short_description}")
        await transcript.mount(Static("\n".join(lines), classes="claude-msg"))

    async def _cmd_agent(self, arg: str) -> None:
        """Show or atomically switch this session's agent/mode.

        ``/agent`` is a status/capability view. Mutating forms are
        ``/agent <name> [mode]``. The gateway owns the mutation; this
        method never creates a session, clears the transcript, or searches
        the target agent's session history.
        """
        transcript = self.query_one("#transcript", VerticalScroll)
        raw = (arg or "").strip()
        await transcript.mount(
            UserMessage(
                "❯ /agent" + (f" {raw}" if raw else ""),
            )
        )
        if self._sid is None or self._client is None:
            self._set_status("no session yet")
            return
        if self._current_turn_id is not None or self._turn_queue:
            msg = "conversation busy — wait for the active/queued turn or abort it before switching"
            self._set_status(msg)
            await transcript.mount(Static(f"[yellow]{msg}[/yellow]", classes="claude-msg"))
            return

        parts = raw.split()
        if len(parts) > 2:
            await transcript.mount(
                Static(
                    "usage: /agent <name> [auto|ask|plan]",
                    classes="claude-msg",
                )
            )
            return
        target = parts[0] if parts else ""
        target_mode = parts[1] if len(parts) == 2 else None

        try:
            state = await self._client.get_session_state(self._sid)
            choices = await self._client.list_agent_modes(self._sid)
        except Exception as e:  # noqa: BLE001
            msg = self._session_switch_error(e)
            self._set_status(msg)
            await transcript.mount(
                Static(
                    f"[red]agent state unavailable:[/red] {msg}",
                    classes="claude-msg",
                )
            )
            return

        if not target:
            msg = self._render_session_agent_status(state, choices)
            await transcript.mount(
                Static(
                    msg,
                    classes="claude-msg",
                    markup=True,
                )
            )
            # Footer already shows the active agent — don't park a sticky
            # ``agent <name>`` line in #status-events (it never auto-clears).
            self._set_status("")
            return

        old_agent = str(state.get("active_agent") or self.agent_cfg.name)
        self._set_status(f"switching {old_agent} → {target}" + (f" ({target_mode})" if target_mode else "") + "…")
        try:
            result = await self._client.switch_agent(
                self._sid,
                target,
                target_mode,
                frontend="tui",
            )
        except Exception as e:  # noqa: BLE001
            msg = self._session_switch_error(e)
            self._set_status(f"agent switch refused: {msg}")
            await transcript.mount(
                Static(
                    f"[red]switch refused:[/red] {msg}",
                    classes="claude-msg",
                )
            )
            return

        active = str(result.get("active_agent") or target)
        mode = self._effective_mode_parts(result)
        msg = f"switched [bold]{old_agent} → {active}[/bold] · {mode[1]} ({mode[2] or 'profile behavior'})"
        self._set_status(re.sub(r"\[/?[^]]+\]", "", msg))
        await transcript.mount(
            Static(
                msg,
                classes="claude-msg",
                markup=True,
            )
        )

    @staticmethod
    def _effective_mode_parts(state: dict) -> tuple[str, str, str]:
        mode = state.get("effective_mode") or {}
        if isinstance(mode, str):
            return (
                mode,
                str(state.get("effective_mode_label") or mode),
                str(state.get("behavior_kind") or ""),
            )
        if not isinstance(mode, dict):
            mode = {}
        mode_id = str(mode.get("id") or state.get("effective_mode_id") or "")
        return (
            mode_id,
            str(mode.get("label") or state.get("effective_mode_label") or mode_id or "Default"),
            str(mode.get("behavior_kind") or state.get("behavior_kind") or ""),
        )

    @staticmethod
    def _session_switch_error(exc: Exception) -> str:
        data = getattr(exc, "data", None)
        if isinstance(data, dict):
            code = str(data.get("code") or "")
            message = str(data.get("message") or "")
            if code and message:
                return f"{code}: {message}"
            if code:
                return code
        text = str(exc).strip()
        for code in (
            "session_busy",
            "unknown_agent",
            "agent_unavailable",
            "agent_forbidden",
            "unsupported_mode",
            "native_session_unavailable",
            "native_mode_rejected",
            "switch_recovery_required",
            "feature_disabled",
        ):
            if code in text:
                return text[text.find(code) :]
        return text or type(exc).__name__

    def _render_session_agent_status(
        self,
        state: dict,
        choices: list[dict],
    ) -> str:
        active = str(state.get("active_agent") or self.agent_cfg.name)
        _, label, behavior = self._effective_mode_parts(state)
        lines = [
            f"[bold]agent[/bold] {active} · [bold]mode[/bold] {label} ({behavior or 'profile behavior'})",
            "",
            "[bold]available[/bold]",
        ]
        for row in choices:
            name = str(row.get("agent_name") or row.get("name") or row.get("agent") or "")
            if not name:
                continue
            modes = row.get("modes") or row.get("availableModes") or []
            labels: list[str] = []
            for mode in modes:
                if not isinstance(mode, dict):
                    continue
                mode_id = str(mode.get("id") or mode.get("requested_mode_id") or "")
                mode_label = str(mode.get("label") or mode_id)
                kind = str(mode.get("behavior_kind") or "")
                labels.append(f"{mode_id}={mode_label}" + (f"/{kind}" if kind else ""))
            status = str(row.get("status") or "")
            suffix = ", ".join(labels) or status or "default"
            mark = "●" if name == active else " "
            lines.append(f"  {mark} [bold]{name}[/bold]: {suffix}")
        lines.extend(
            (
                "",
                "[dim]/agent <name> [mode] switches this conversation without clearing its transcript.[/dim]",
            )
        )
        return "\n".join(lines)

    async def _cmd_refresh(self) -> None:
        """Re-fold the durable session log into this transcript.

        Live subscribe-tail is the happy path for cross-frontend turns, but
        a poisoned renderer cursor (or a missed remote turn) still needs an
        operator-facing catch-up. This clears the local transcript widgets
        and replays the gateway's folded view, then re-anchors the live
        renderer on the session-log byte cursor.
        """
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage("❯ /refresh"))
        if self._sid is None or self._client is None:
            self._set_status("no session yet — wait for the agent to come up")
            return
        if self._current_turn_id is not None or self._turn_queue:
            msg = "turn in flight — abort first, then /refresh"
            self._set_status(msg)
            await transcript.mount(
                Static(
                    f"[yellow]{msg}[/yellow]",
                    classes="claude-msg",
                )
            )
            return
        self._set_status("refreshing from durable session log…")
        try:
            data = await self._client.get_session(self._sid)
        except Exception as e:  # noqa: BLE001
            msg = f"refresh failed: {e}"
            self._set_status(msg)
            await transcript.mount(
                Static(
                    f"[red]{msg}[/red]",
                    classes="claude-msg",
                )
            )
            return
        await transcript.remove_children()
        await self._bind_session_view(data, replay=True)
        n = sum(1 for t in (data.get("turns") or []) if not t.get("in_progress"))
        self._set_status(f"refreshed · {n} turn(s) from durable log")

    async def _cmd_compact(self) -> None:
        """Manual `/compact` — between-turn equivalent of the model's
        `compact_context` tool call.

        Synthesizes a user prompt that biases the local model to emit
        ``compact_context``. The bridge intercepts that tool call and
        squeezes the live message list in-place, persisting a rolling
        summary onto the session. Same shape Telegram uses, so the
        two frontends stay in lock-step.
        """
        if self._sid is None:
            self._set_status("no session yet — wait for the agent to come up")
            return
        if self._current_turn_id is not None:
            # Mid-turn compaction is the model's job (`compact_context`).
            # Doing it from here would race the dispatch loop.
            self._set_status("turn in flight — abort first or call compact_context")
            return
        assert self._client is not None
        self._set_status("compacting…")
        # Hand off to the regular turn dispatcher with a synthesized
        # tool-use directive. The bridge intercept on compact_context
        # surfaces a tool_call_update with the audit; the standard TUI
        # rendering shows it as the round's tool result, plus the
        # status bar updates after `_send_user_turn` returns.
        compact_directive = (
            "Call the `compact_context` tool now to squeeze the running "
            "message list. Then briefly confirm in one sentence what was "
            "trimmed."
        )
        await self._send_user_turn(compact_directive)

    def _apply_command_state_patch(self, patch: dict) -> None:
        mode = patch.get("permission_mode") or patch.get("system_mode")
        if isinstance(mode, str):
            self._permission_mode = mode
        if "thinking_enabled" in patch:
            self.enable_thinking = bool(patch.get("thinking_enabled"))
        reasoning = patch.get("reasoning_visibility")
        if isinstance(reasoning, str):
            self.reasoning_visibility = reasoning
        if "tool_output_visible" in patch:
            self.tool_output_visible = bool(patch.get("tool_output_visible"))
        if "effort" in patch:
            self._effort = str(patch.get("effort") or "default")
        if "speed" in patch:
            self._speed = str(patch.get("speed") or "default")
        _teacher_touched = False
        if "teacher_override" in patch:
            # The PERSISTED per-session toggle ("on"/"off"/""). This is the
            # authoritative value to track — NOT the effective gate — so a
            # later goal change recomputes correctly and a goal-driven session
            # (override "") is never mistaken for an explicit `/teacher on`.
            self._teacher_override = str(patch.get("teacher_override") or "")
            if self._teacher_override == "off":
                self._teacher_reviewing = False
                self._teacher_review_cids.clear()
            _teacher_touched = True
        if "teacher_model" in patch:
            # `/teacher model <alias>` picked a different supervising model;
            # record the per-session override (empty → configured teacher row).
            self._teacher_model_override = str(patch.get("teacher_model") or "").strip()
            _teacher_touched = True
        if _teacher_touched:
            # Recompute the EFFECTIVE gate from override + goal + policy — this
            # agrees with the engine and with the patch's ``teacher_active``.
            self._recompute_teacher()
        elif "teacher_active" in patch:
            # No override/model change (e.g. a bare status echo) — trust the
            # runtime's effective gate directly so the widget still updates.
            self._teacher_enabled = bool(patch.get("teacher_active"))
            self._refresh_goal_bar()
        self._refresh_status_bar()

    async def _confirm_irreversible(self, text: str) -> bool:
        """TUI analogue of the Web UI confirm card.

        Server re-challenge is scoped to ``frontend=web``. TUI confirms
        locally so irreversible slashes cannot fire on a stray Enter.
        """
        copy = confirm_copy_for(text)
        if copy is None:
            return True
        title, prompt = copy
        try:
            ok = await self.push_screen_wait(
                ConfirmModal(title=title, prompt=prompt),
            )
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            self._set_status("cancelled")
            try:
                transcript = self.query_one("#transcript", VerticalScroll)
                await transcript.mount(
                    Static(
                        "cancelled — command not run",
                        classes="claude-msg",
                    )
                )
            except Exception:  # noqa: BLE001
                pass
            return False
        return True

    async def _cmd_shared(self, text: str, *, confirmed: bool = False) -> None:
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage(f"❯ {text}"))
        if self._sid is None or self._client is None:
            self._set_status("no session yet — wait for the agent to come up")
            return
        # Mid-turn: do not invent a stricter gate than the runtime.
        # Web UI sends every slash to session/command; TURN_UNSAFE_COMMANDS
        # (/model, /compact) are rejected server-side.
        # A TUI-only allowlist used to drop /teacher /budget /goal while
        # a turn was streaming.
        if not confirmed and not await self._confirm_irreversible(text):
            return
        try:
            result = await self._client.session_command(
                self._sid,
                text,
                frontend="tui",
                confirmed=True,
            )
        except Exception as e:  # noqa: BLE001
            self._set_status(f"command failed: {e}")
            return
        self._apply_command_state_patch(result.get("state_patch") or {})
        # Apply imperative effects the runtime returned (generic, like the
        # web frontend): prompt → submit a turn, cancel/new_session → control.
        for _eff in result.get("effects") or []:
            if not isinstance(_eff, dict):
                continue
            _t = _eff.get("type")
            _p = _eff.get("payload") or {}
            if _t == "prompt" and _p.get("text"):
                self._enqueue_or_run("prompt", _p["text"])
                self._set_status(result.get("text") or "running…")
                return
            if _t == "cancel":
                # The runtime already issued the server-side cancel; mirror the
                # force level locally (bare /abort = hard kill).
                await self.action_abort(via="abort" if _p.get("force") else "stop")
                return
            if _t == "new_session":
                await self._cmd_new()
                return
            if _t == "switch_session" and _p.get("session_id"):
                switch = getattr(self, "_switch_to_session", None)
                if callable(switch):
                    await switch(_p["session_id"])
                    return
            if _t == "switch_agent" and _p.get("agent"):
                run = getattr(self, "_cmd_agent", None)
                if callable(run):
                    await run(str(_p["agent"]))
                    return
        kind = str(result.get("kind") or "text")
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        msg = format_kind_body(kind, result.get("text") or "", data)
        title = card_title(result.get("command") or text.split()[0], kind, data)
        is_error = (not result.get("ok")) or kind == "error" or bool(result.get("error"))
        # Render by KIND, never by command name.
        if kind == "detail":
            parts = text.split(None, 1)
            sub = parts[1] if len(parts) > 1 else ""
            await transcript.mount(ContextMessage(msg, sub=sub))
        else:
            await transcript.mount(
                CommandResultMessage(title, msg, kind=kind, error=is_error),
            )
        self._set_status(msg.replace("\n", "  ")[:200])

    async def _cmd_models(self, arg: str = "") -> None:
        """`/models` — list configured models (interactive picker).

        Mirrors openclaw's `/models`: list-only entry point. The single
        configured-model case prints an inline summary; multi-model
        configs open the modal picker so the user can pick one. To
        switch directly without the picker, use `/model <alias|url>`.

        For external-agent engines whose model list is engine-owned, the rows
        come from :func:`_engine_models_for` instead of ``cfg.models`` so the
        picker shows the right thing for every registry entry.
        """
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage("❯ /models" + (f" {arg}" if arg else "")))
        if arg:
            self._set_status("usage: /models  (no args). switch with /model <alias|url>")
            return
        cfg = self.agent_cfg
        # Discovery may do a (cached) network call — keep it off the UI thread.
        engine_rows = await asyncio.to_thread(_engine_models_for, cfg)
        if engine_rows is not None and not engine_rows:
            # External CLI/ACP engine with no discoverable or owned menu
            # (generic ACP or an unavailable catalogue): the model is owned
            # out-of-band, so the local
            # cfg.models rows aren't available to this agent — show only the
            # active model instead of offering a picker of unusable options.
            engine = (getattr(cfg, "engine", "") or "").strip() or "engine"
            cur = (self.session.model if self.session else "") or ""
            label = cur or engine
            await transcript.mount(
                Static(
                    f"current: {label}\n\n"
                    f"this agent runs the `{engine}` engine — its model is "
                    "set by that CLI/server, not from sharedagents's model list.",
                    classes="claude-msg",
                )
            )
            self._set_status("model info shown")
            return
        if engine_rows is not None:
            rows = engine_rows
        else:
            from sharedagents_command.commands import local_model_rows

            rows = local_model_rows(cfg) or []
            if not rows:
                rows = [cfg._primary_model_row()]
        current = self.session.endpoint if self.session else cfg.upstream_endpoint
        current_norm = (current or "").rstrip("/")
        # Engine-owned rows have no upstream URL; match by alias instead.
        current_alias = ""
        if engine_rows is not None:
            current_alias = self._current_engine_model_alias()
        # Wire model id of the live session — picker uses it to break
        # ties when multiple rows share the same upstream URL.
        current_model_id = (self.session.model if self.session else "") or ""

        if len(rows) == 1:
            m = rows[0]
            alias = m.get("alias") or "?"
            raw_model_id = m.get("id") or ""
            model_id = raw_model_id or "(missing id)"
            url = (m.get("baseUrl") or "").rstrip("/")
            ctx = m.get("contextWindow") or 0
            ctx_label = f"{ctx // 1024}k" if ctx else "?"
            model_label = _model_alias_id_label(alias, model_id)
            lines = [
                f"current: {model_label}  ctx={ctx_label}",
                f"  {url}",
                "",
                "Add more models via `sharedagents agent setup` or by editing ~/.sharedagents/config.json",
            ]
            await transcript.mount(Static("\n".join(lines), classes="claude-msg"))
            self._set_status("model info shown")
            return

        picked = await self.push_screen_wait(
            ModelPickerModal(
                rows,
                current_url=current_norm,
                current_alias=current_alias,
                current_model_id=current_model_id,
            ),
        )
        if not picked:
            return
        await self._switch_to_model_row(picked)

    def _current_engine_model_alias(self) -> str:
        """Return the selected alias for any registry-backed external engine.

        If a fresh session still carries the engine label instead of a model,
        return the registry's configured default where one exists. ACP
        engines without a configured default use the portable ``auto``
        sentinel; native CLI engines leave the alias empty until their own
        model is reported.

        Used to highlight the right row in the model picker and to
        render the footer's external model segment.
        """
        engine = (getattr(self.agent_cfg, "engine", "") or "").strip()
        try:
            from sharedagents_agent.external_registry import (
                is_external_engine_kind,
                spec_for,
            )

            if not is_external_engine_kind(engine):
                return ""
            spec = spec_for(engine) if engine != "acp" else None
        except Exception:  # noqa: BLE001
            return ""
        sel = ""
        if self.session is not None:
            sel = (getattr(self.session, "model", "") or "").strip()
        engine_label = str(getattr(spec, "short_name", "") or "")
        if sel and sel not in {engine, engine_label}:
            return sel
        configured_default = str(getattr(spec, "default_model", "") or "").strip()
        if configured_default:
            return configured_default
        selector_default = str(getattr(spec, "default_model_alias", "") or "").strip()
        if selector_default:
            return selector_default
        if getattr(spec, "adapter", "") == "acp" or engine == "acp":
            return "auto"
        # A native CLI owns its default and may not have reported it yet.
        # Do not mark an arbitrary picker row as current.
        return ""

    async def _cmd_model(self, arg: str = "") -> None:
        """`/model <alias|url>` — switch the live session's model/upstream.

        Each configured model row in `cfg.models` carries an alias,
        baseUrl, id, backend, and contextWindow. The alias names the
        model from the user's POV. To list/pick interactively, use
        `/models`.
        """
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(UserMessage(f"❯ /model {arg}".rstrip()))
        if not arg:
            self._set_status("usage: /model <alias|url>. list available models with /models")
            return

        if self._sid is None:
            self._set_status("no session yet — wait for the agent to come up")
            return
        if self._current_turn_id is not None:
            self._set_status("turn in flight — abort first")
            return
        assert self._client is not None
        try:
            result = await self._client.session_command(self._sid, f"/model {arg}", frontend="tui")
        except Exception as e:  # noqa: BLE001
            self._set_status(f"model switch failed: {e}")
            return
        data = result.get("data") or {}
        if not result.get("ok"):
            self._set_status(result.get("text") or "model command failed")
            return
        if self.session is not None:
            self.session.endpoint = data.get("endpoint") or data.get("baseUrl") or self.session.endpoint
            self.session.model = data.get("model") or data.get("id") or self.session.model
            if data.get("contextWindow"):
                self.session.context_window = int(data.get("contextWindow") or 0)
        self.endpoint = data.get("endpoint") or data.get("baseUrl") or self.endpoint
        if data.get("contextWindow"):
            self._last_context_window = int(data.get("contextWindow") or 0)
        if data.get("model") or data.get("id"):
            self.model = data.get("model") or data.get("id")
        self._refresh_title()
        try:
            self._refresh_status_bar()
        except Exception:  # noqa: BLE001
            pass
        msg = result.get("text") or "model switched"
        self._set_status(msg.replace("\n", "  "))
        await transcript.mount(Static(msg, classes="claude-msg"))

    async def _switch_to_model_row(self, row: dict) -> None:
        """Rebind the session to a model row dict and update local state.

        Extracted from `_cmd_model` so both the modal picker path and
        the `/model <alias>` path share the same switch logic.
        """
        resolved = (row.get("baseUrl") or "").rstrip("/")
        # Registry-backed external rows carry no sharedagents upstream URL —
        # they're selected by the engine model id/alias. Only the local-LLM /
        # openrouter path requires a URL.
        engine = (getattr(self.agent_cfg, "engine", "") or "").strip()
        from sharedagents_agent.external_registry import is_external_engine_kind

        engine_owned = is_external_engine_kind(engine)
        if not resolved and not engine_owned:
            self._set_status(f"unknown model alias {row.get('alias', '?')!r}")
            return
        assert self._client is not None
        # Pass ``alias=`` so the bridge resolves the row by alias
        # (the only key it accepts on ``set_config_option(model=…)``),
        # and ``id=`` so the local view's stats line picks up the
        # right wire model id without a round-trip. Without
        # ``alias`` the picker silently fails on the wire because
        # the bridge's ``_resolve_model_row`` only matches by alias.
        row_id = row.get("id") or row.get("model")
        row_alias = row.get("alias") or ""
        try:
            result = await self._client.update_session_model(
                self._sid,
                baseUrl=resolved,
                id=row_id or None,
                alias=row_alias or None,
                backend=row.get("backend") or None,
                contextWindow=row.get("contextWindow") or None,
            )
        except Exception as e:  # noqa: BLE001
            self._set_status(f"model switch failed: {e}")
            return
        if not result.get("ok"):
            # Wire-side rejection (unknown alias, upstream
            # unreachable…). Surface it so the operator doesn't see
            # a green ack while messages keep landing on the
            # previous model.
            err = str(result.get("error") or "unknown error")
            self._set_status(f"model switch failed: {err[:100]}")
            transcript = self.query_one("#transcript", VerticalScroll)
            await transcript.mount(
                Static(
                    f"[red]model switch failed:[/red] {err}",
                    classes="claude-msg",
                ),
            )
            return
        # ``update_session_model`` returns the local-view shape
        # (``endpoint`` / ``model`` / ``context_window`` snake-case),
        # not the picker-row shape (``baseUrl`` / ``id`` /
        # ``contextWindow``). Read both so the post-switch ack works
        # whether the caller is the picker or a manual ``/model``.
        new_endpoint = result.get("endpoint") or result.get("baseUrl") or resolved
        new_model_id = result.get("model") or result.get("id") or row_id or ""
        new_ctx = result.get("context_window") or result.get("contextWindow") or row.get("contextWindow") or 0
        if self.session is not None:
            self.session.endpoint = new_endpoint
            if new_model_id:
                self.session.model = new_model_id
            if new_ctx:
                self.session.context_window = int(new_ctx)
        self.endpoint = new_endpoint
        if new_ctx:
            self._last_context_window = int(new_ctx)
        if new_model_id:
            self.model = new_model_id
        self._refresh_title()
        # Repaint the footer NOW so model/ctx update without waiting
        # for the next health poll — matches what the picker user
        # expects: pick → footer flips immediately.
        try:
            self._refresh_status_bar()
        except Exception:  # noqa: BLE001
            pass
        transcript = self.query_one("#transcript", VerticalScroll)
        warn = result.get("upstream_error")
        alias = row.get("alias") or "?"
        ctx_label = f" · ctx={new_ctx // 1024}k" if new_ctx else ""
        # Friendlier ack: alias is what the user picked from the
        # modal; model id is the wire identifier the agent will
        # actually request.
        model_label = _model_alias_id_label(alias, new_model_id)
        msg = f"model → {model_label or alias}{ctx_label}"
        msg += f"\n  {new_endpoint}"
        if warn:
            msg += f"  ·  upstream warn: {warn[:60]}"
        self._set_status(msg.replace("\n", "  ")[:200])
        await transcript.mount(Static(msg, classes="claude-msg"))
