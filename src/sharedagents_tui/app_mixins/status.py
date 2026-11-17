"""AgentChatApp status bar / scroll-follow / worker state.

Extracted from app.py (de-monolith); behavior-preserving.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Static

from sharedagents_tui.agent_client import _model_label
from sharedagents_tui.helpers import (
    _SPINNER_FRAMES,
    _alias_for_endpoint,
    _hostname_of,
)


class StatusMixin:
    """AgentChatApp status bar / scroll-follow / worker state."""

    def action_cycle_mode(self) -> None:
        """Cycle the active agent's advertised modes for this session."""
        client = getattr(self, "_client", None)
        sid = getattr(self, "_sid", None)
        if client is not None and sid:
            self.run_worker(
                self._cycle_session_mode(),
                exclusive=False,
                name="session-mode",
            )
            return
        try:
            i = self._MODES_CYCLE.index(self._permission_mode)
        except ValueError:
            i = 0
        self._permission_mode = self._MODES_CYCLE[(i + 1) % len(self._MODES_CYCLE)]
        self._refresh_status_bar()
        # Surface the change as a transient turn-state suffix so the
        # user sees it acknowledged without having to read the wrapped
        # right-side fields.
        self._set_status(f"permission mode: {self._permission_mode}")

    async def _cycle_session_mode(self) -> None:
        if self._current_turn_id is not None or self._turn_queue:
            self._set_status("conversation busy — mode unchanged")
            return
        assert self._client is not None and self._sid is not None
        try:
            state = await self._client.get_session_state(self._sid)
            rows = await self._client.list_agent_modes(self._sid)
            active = str(state.get("active_agent") or self.agent_cfg.name)
            row = next(
                (
                    item
                    for item in rows
                    if str(item.get("agent_name") or item.get("name") or item.get("agent") or "") == active
                ),
                {},
            )
            modes = [
                str(mode.get("id") or mode.get("requested_mode_id") or "")
                for mode in (row.get("modes") or row.get("availableModes") or [])
                if isinstance(mode, dict)
                and (mode.get("id") or mode.get("requested_mode_id"))
                and mode.get("supported", True)
            ]
            if len(modes) < 2:
                self._set_status(f"{active} has no alternate supported mode")
                return
            effective = state.get("effective_mode") or {}
            current = str(effective.get("id") or "") if isinstance(effective, dict) else str(effective)
            try:
                idx = modes.index(current)
            except ValueError:
                idx = -1
            target = modes[(idx + 1) % len(modes)]
            result = await self._client.switch_agent(
                self._sid,
                active,
                target,
                frontend="tui",
            )
            effective = result.get("effective_mode") or {}
            self._permission_mode = (
                str(effective.get("id") or target) if isinstance(effective, dict) else str(effective or target)
            )
            self._refresh_status_bar()
            self._set_status(f"mode: {self._permission_mode}")
        except Exception as exc:  # noqa: BLE001
            render = getattr(self, "_session_switch_error", None)
            message = render(exc) if callable(render) else str(exc)
            self._set_status(f"mode switch refused: {message}")

    def action_open_shells(self) -> None:
        """Ctrl+S — open the Shells modal (alias for the ``/shells`` /
        ``/monitors`` slash command). Dispatches into ``_cmd_monitors``
        via the same worker pattern the slash dispatcher uses so the
        modal mount path is identical."""
        run = getattr(self, "_cmd_monitors", None)
        if run is None:
            self._set_status("shells panel only available via the agent runtime")
            return
        self.run_worker(run(), exclusive=False, name="shells")

    def _update_scroll_follow(
        self,
        container: VerticalScroll | None = None,
    ) -> None:
        """Read the transcript's scroll position and update the follow flag.

        Called from ``TranscriptScroll.watch_scroll_y`` whenever
        ``scroll_y`` changes — covers wheel, touchpad, scrollbar-drag,
        and key-driven scrolls.

        * Scrolled away from the bottom → disable follow.
        * At/near the bottom (≤ threshold) → re-enable follow.
        """
        if container is None:
            try:
                container = self.query_one("#transcript", VerticalScroll)
            except Exception:  # noqa: BLE001
                return
        try:
            virtual_height = container.virtual_size.height
            view_height = container.size.height
            scroll_y = container.scroll_y
        except Exception:  # noqa: BLE001
            return
        if virtual_height <= view_height:
            # Content fits in the viewport — nothing to scroll away from.
            self._scroll_follow = True
            return
        max_scroll = virtual_height - view_height
        if (max_scroll - scroll_y) > self._SCROLL_FOLLOW_THRESHOLD:
            self._scroll_follow = False
        else:
            self._scroll_follow = True

    def _is_at_bottom(
        self,
        container: Any = None,
    ) -> bool:
        """Whether auto-scroll-end should fire.

        Returns the cached ``_scroll_follow`` flag — updated by
        :meth:`_update_scroll_follow` (wheel/page-key/scrollbar
        scroll-away → disable, scroll-to-bottom → enable),
        :meth:`on_key` (PageUp → disable, PageDown/Shift+End → enable),
        and user-submit paths (re-arm).
        This deliberately does NOT re-derive from post-mutation
        geometry: that path loses bottom affinity across bulk mounts
        (a multi-row tool result widget mounted under a user who was
        at the bottom pushes the bottom edge out of the threshold,
        so the helper wrongly concludes the user has scrolled away).
        The cached flag holds the user's actual intent regardless of
        how the content under them grew.
        """
        # ``container`` arg kept for back-compat with callers that
        # pass an explicit container; the cached flag is global to
        # the transcript so we ignore it.
        del container
        return self._scroll_follow

    def _maybe_scroll_end(
        self,
        container: Any = None,
        *,
        animate: bool = False,
    ) -> None:
        """Guarded version of ``container.scroll_end(...)``.

        Used as the canonical auto-scroll entry point so live
        streaming chunks, mid-turn renders, and queue/state repaints
        only follow when the operator hasn't scrolled away. Without
        this, every chunk on a long-running turn snaps the viewport
        back to the bottom, making it impossible to read prior turns
        until the model finishes.

        Operator submits + explicit "user is acting" events (sending
        a prompt, /new) still call ``scroll_end`` directly so the
        user's own action always brings them to the live tail; only
        the *passive* repaints route through this guard.

        Affinity caveat: by the time this fires, content has already
        been mounted/appended and ``virtual_size`` reflects the new
        layout — a user who was at the bottom right before the mount
        is no longer within the threshold. For passive flows where
        callers need "preserve bottom affinity across this update,"
        see :meth:`_scroll_end_if_was_at_bottom`. Here we accept the
        post-mount affinity check — works fine for small streaming
        chunks (single delta < 3 rows), trades correctness on bulk
        mounts (tool result cards) for a single hot-path helper that
        any call site can drop in without preceding bookkeeping.
        """
        if container is None:
            try:
                container = self.query_one(
                    "#transcript",
                    VerticalScroll,
                )
            except Exception:  # noqa: BLE001
                return
        if not self._is_at_bottom(container):
            return
        try:
            container.scroll_end(animate=animate)
        except Exception:  # noqa: BLE001
            pass

    def _scroll_end_if_was_at_bottom(
        self,
        container: Any,
        was_at_bottom: bool,
        *,
        animate: bool = False,
    ) -> None:
        """Variant that uses a caller-captured affinity snapshot.

        Pattern:

            anchor = self._is_at_bottom(transcript)
            await transcript.mount(BigWidget())   # virtual_size jumps
            self._scroll_end_if_was_at_bottom(transcript, anchor)

        Lets passive bulk mounts (tool result cards, restored turns,
        compaction notices) preserve "follow the tail" affinity that
        ``_maybe_scroll_end`` alone can't infer post-mount.
        """
        if not was_at_bottom:
            return
        try:
            container.scroll_end(animate=animate)
        except Exception:  # noqa: BLE001
            pass

    def _read_shell_count(self) -> tuple[int, int]:
        """Return (live, exited) counts for monitors of the active
        session. Cached for ~3 s so the spinner's 0.25-s repaint
        doesn't beat on the sidecar JSON. Returns the cache when no
        active session yet."""
        sid = getattr(self, "_sid", None)
        if not sid:
            return (0, 0)
        now = time.monotonic()
        if now - self._shell_count_at < 3.0:
            return self._shell_count_cache
        try:
            from sharedagents_agent.tools.monitor import live_monitors_for_session

            sessions_dir = Path(self.agent_cfg.sessions_dir)
            rows = live_monitors_for_session(sessions_dir, sid)
            live = sum(1 for r in rows if r.get("alive"))
            exited = len(rows) - live
            self._shell_count_cache = (live, exited)
        except Exception:  # noqa: BLE001
            # Sidecar missing / unreadable — leave the previous cached
            # value so a transient FS hiccup doesn't blank the segment.
            pass
        self._shell_count_at = now
        return self._shell_count_cache

    def _refresh_status_bar(self) -> None:
        """Repaint the single consolidated ``#status-footer``.

        Order (left → right):
            <state> · agent <name> · mode <mode> · ctx <used>/<max>
              · shells <live>[+<exited>] · model <alias>/<id>
              · upstream <ok|DOWN: …>

        Per-call detail (ttft · gen tps · wt …) lives on each
        AssistantMessage's own dim metrics row, but the footer
        carries the identity + operational fields the operator
        needs at a glance even when their eye is on the transcript:
        which agent, which mode, how full the context is, how many
        background shells are running (so Ctrl+S is discoverable),
        which model is wired, and whether the upstream is healthy.
        """
        parts: list[str] = []

        # State (always visible) + queued badge.
        # Transient acks (errors, "aborting…", "session created") go
        # to ``#status-events`` above the footer, NOT here — the
        # footer is identity + persistent state only.
        #
        # While a turn is in flight (``_worker_state`` set) prepend a
        # braille spinner frame so the user has a continuous "still
        # alive" cue even when chunks pause — the primary user-
        # reported failure mode here was "TUI looks frozen right
        # before a long tool call". The frame advances on a fast
        # timer (``_animate_status``); idle state shows no spinner.
        qn = len(self._turn_queue)
        if self._worker_state:
            frame = _SPINNER_FRAMES[self._anim_tick % len(_SPINNER_FRAMES)]
            state_seg = f"[bold cyan]{frame}[/bold cyan] {self._worker_state}"
        else:
            state_seg = "[bold]idle[/bold]"
        if qn:
            state_seg = f"{state_seg} [yellow]+{qn} queued[/yellow]"
        parts.append(state_seg)

        # Agent name.
        _session_agent = str(getattr(getattr(self, "_client", None), "active_agent", "") or "")
        parts.append(f"agent [bold]{_session_agent or self.agent_cfg.name}[/bold]")

        # Permission mode — colour-coded by danger level so a glance
        # at the status bar tells the operator whether the agent is in
        # a guarded mode (plan/ask) or autonomous (auto). Mapping:
        #   plan → green   (read-only, safest)
        #   ask  → yellow  (prompts before mutations)
        #   auto → red     (bypassPermissions; agent runs unrestricted)
        # Unknown modes fall through to plain bold so a typo doesn't
        # blank the indicator. ``[/]`` closes the most-recent opened
        # span — Rich's recommended form, immune to the name-matched
        # closing-tag edge cases.
        _mode_style = {
            "plan": "bold green",
            "ask": "bold yellow",
            "auto": "bold red",
        }.get(self._permission_mode, "bold")
        _effective = getattr(
            getattr(self, "_client", None),
            "effective_mode",
            {},
        )
        if isinstance(_effective, dict) and _effective:
            _mode_label = str(_effective.get("label") or _effective.get("id") or self._permission_mode)
            _mode_behavior = str(_effective.get("behavior_kind") or "")
            _mode_display = f"{_mode_label}/{_mode_behavior}" if _mode_behavior else _mode_label
        else:
            _mode_display = self._permission_mode
        parts.append(f"mode: [{_mode_style}]{_mode_display}[/]")

        # Goal + teacher indicators live in the dedicated goal widget
        # (``#goal-text`` / ``#teacher-row``, see ``_refresh_goal_bar``), NOT
        # the footer — keep the footer to identity/state. Refresh them
        # alongside the footer so they stay in sync from every call site.
        self._refresh_goal_bar()

        # Context used/max — pulled from the latest worker metrics
        # (``prompt_tokens`` per round) and the upstream's advertised
        # ``max_context``. Falls back to the live session's
        # ``contextWindow`` (set on /model and on session_new), then
        # to the upstream-health baseline so a fresh session before
        # any turn still shows a number.
        upstream = (self._health or {}).get("upstream") or {}
        cli_engine = (getattr(self.agent_cfg, "engine", "") or "").strip()
        from sharedagents_agent.external_registry import (
            is_external_engine_kind,
            spec_for,
        )

        _is_external_engine = is_external_engine_kind(cli_engine)
        try:
            _external_spec = spec_for(cli_engine) if _is_external_engine and cli_engine != "acp" else None
        except Exception:  # noqa: BLE001
            _external_spec = None
        # External engines do not use the local cfg-level context window: the
        # CLI/ACP server owns that dimension, and ACP may report a live value
        # through ``_last_context_window``. Suppress the inherited local
        # fallback for every registry-backed external kind.
        if _is_external_engine:
            is_acp = cli_engine == "acp" or getattr(_external_spec, "adapter", "") == "acp"
            max_ctx = self._last_context_window if is_acp else 0
        else:
            max_ctx = (
                self._last_context_window
                or upstream.get("max_context")
                or getattr(self.agent_cfg, "upstream_max_context", 0)
                or 0
            )
        if isinstance(max_ctx, int) and max_ctx > 0:
            if self._last_prompt_tokens is not None:
                used_disp = str(self._last_prompt_tokens)
            else:
                baseline = upstream.get("baseline_prompt_tokens")
                used_disp = (
                    f"~{baseline}"
                    if isinstance(baseline, int) and baseline > 0
                    else "\u2014"  # em dash — "no data yet"
                )
            parts.append(f"ctx {used_disp}/{max_ctx}")

        # Background-shells counter. Surfaces ``shells N`` (cyan) when
        # the model has live monitor subprocesses for this session, so
        # the operator notices the Ctrl+S / /shells panel exists at
        # all — without this badge a long-running background shell
        # has no on-screen presence until the operator thinks to
        # open the modal. ``+M`` follows when there are also exited
        # entries the panel will still show until they're cleared.
        # Suppressed entirely at (0, 0) so the footer stays clean
        # when no monitors are in play.
        live_n, exited_n = self._read_shell_count()
        if live_n or exited_n:
            seg = f"shells [bold cyan]{live_n}[/bold cyan]"
            if exited_n:
                seg += f"[dim]+{exited_n}[/dim]"
            parts.append(seg)

        # Upstream model — alias/<model_id>. Resolution prefers the
        # live session's endpoint match against
        # ``agent_cfg.models[]``, then falls back to the configured
        # upstream alias / hostname.
        #
        # CLI-engine agents (claude/codex/antigravity) don't drive an HTTP
        # upstream — the CLI itself owns the model call. The session
        # carries ``model = <engine name>`` and an empty endpoint, and
        # the inherited cfg.upstream_* fields belong to a different
        # (local-LLM) deployment. Render the engine name as-is so the
        # footer stops falsely advertising the global config's upstream
        # (e.g. ``gpu-a/qwen3.6-27b-fp8``).
        # The REAL model the engine reported (claude-opus-4-…/gpt-5.5-codex/
        # composer-2.5/antigravity's concrete id), folded into the session
        # header at turn end and re-read here via the post-turn
        # ``get_session`` refresh. Preferred over the placeholder/alias
        # ``model`` for every external engine. Empty until the first turn.
        sess_engine_model = (getattr(self.session, "engine_model", "") if self.session else "") or ""
        is_external = _is_external_engine
        spec = _external_spec
        if is_external:
            # All external engines own the model dimension; none should fall
            # through to the local upstream alias. Registry metadata supplies
            # the display name and adapter type, while the live session fields
            # supply the selected/reported model.
            engine_name = str(getattr(spec, "short_name", "") or cli_engine.removesuffix("_cli"))
            selected_model = (getattr(self.session, "model", "") if self.session else "") or ""
            if getattr(spec, "adapter", "") == "acp" or cli_engine == "acp":
                acp_label = getattr(self.agent_cfg, "name", "") or engine_name
                acp_model_id = (sess_engine_model or selected_model).strip()
                if not acp_model_id or acp_model_id.lower() in {
                    "local",
                    "default",
                    engine_name.lower(),
                    cli_engine.lower(),
                }:
                    acp_model_id = "auto"
                parts.append(f"model [bold]{_model_label(acp_label, acp_model_id)}[/bold]")
            else:
                shown = (sess_engine_model or selected_model).strip()
                if shown and shown.lower() not in {
                    engine_name.lower(),
                    cli_engine.lower(),
                }:
                    parts.append(f"model [bold]{_model_label(engine_name, shown)}[/bold]")
                else:
                    parts.append(f"model [bold]{engine_name}[/bold]")
        else:
            sess_endpoint = (self.session.endpoint if self.session else "") or upstream.get("endpoint", "")
            model_id = (self.session.model if self.session else "") or upstream.get("model") or ""
            alias = (
                _alias_for_endpoint(
                    getattr(self.agent_cfg, "models", None),
                    sess_endpoint,
                    model_id,
                )
                or (
                    getattr(self.agent_cfg, "upstream_alias", "")
                    if (sess_endpoint or "").rstrip("/") == (self.agent_cfg.upstream_endpoint or "").rstrip("/")
                    else ""
                )
                or _hostname_of(sess_endpoint)
                or "upstream"
            )
            model_label = _model_label(alias, model_id)
            if model_label and (model_id or alias != "upstream"):
                parts.append(f"model [bold]{model_label}[/bold]")

        # Upstream status. Agent unreachable (``self._health_error``)
        # shadows everything else; otherwise we surface ``ok`` / DOWN
        # for the LLM upstream. The green ``ok`` badge is the
        # operator's at-a-glance "the model is reachable" signal.
        if self._health_error:
            parts.append(f"[red]agent unreachable: {self._health_error[:60]}[/red]")
        elif upstream.get("ok") is False:
            err = str(upstream.get("error") or "")[:60]
            parts.append(f"[red]upstream DOWN{(': ' + err) if err else ''}[/red]")
        elif upstream.get("ok") is True:
            # ``probed=False`` means we couldn't / wouldn't probe the
            # endpoint live (cloud providers, where every poll is a
            # paid API hit). Render a softer "cloud" badge so the
            # operator knows the green isn't a live reachability
            # signal — a real outage there surfaces as a turn-time
            # error via the ``upstream warn`` row instead.
            if upstream.get("probed") is False:
                parts.append("[dim green]upstream cloud[/dim green]")
            else:
                parts.append("[green]upstream ok[/green]")

        line = "  ·  ".join(parts)
        try:
            self.query_one("#status-footer", Static).update(line)
        except Exception:  # noqa: BLE001
            # Widget may not be mounted yet on the first poll.
            pass

    def _recompute_teacher(self) -> None:
        """Recompute the EFFECTIVE teacher gate (``_teacher_enabled``) and the
        supervising-model label from the policy + the per-session override +
        the active goal, then repaint the widget. Mirrors the engine's
        ``_teacher_enabled`` resolution (and the shared
        ``teacher_supervisor.is_teacher_active``) so the footer widget agrees
        with what actually supervises the session."""
        try:
            from sharedagents_agent.teacher_supervisor import is_teacher_active

            pol = {
                "enabled": self._teacher_policy_enabled,
                "require_goal": self._teacher_require_goal,
            }
            self._teacher_enabled = is_teacher_active(
                pol,
                self._teacher_override,
                self._active_goal,
            )
        except Exception:  # noqa: BLE001
            self._teacher_enabled = False
        self._teacher_model = self._teacher_model_override or self._teacher_cfg_model
        self._refresh_goal_bar()

    def _refresh_goal_bar(self) -> None:
        """Repaint the teacher/goal supervision widget (two rows).

        The widget is a TEACHER widget — it appears whenever supervision is
        active (a goal is set OR ``/teacher on`` was issued) and DISAPPEARS
        when the teacher is off:
          • teacher inactive → both rows empty (collapse to 0 height)
          • teacher active → ``#teacher-row`` shows the supervising model
            (or ``reviewing…``); ``#goal-text`` shows the FULL goal when one
            is set (word-wrapped, scrollable), or a "no goal" hint when the
            teacher is on without a goal.
        """
        try:
            goal_text = self.query_one("#goal-text", Static)
            teacher_row = self.query_one("#teacher-row", Static)
        except Exception:  # noqa: BLE001
            return
        if not getattr(self, "_teacher_enabled", False):
            goal_text.update("")
            teacher_row.update("")
            return
        # Full goal, word-wrapped (no truncation); the #goal-scroll container
        # caps the height and scrolls when it overflows. Collapse internal
        # newlines so the wrap is driven by width, not the operator's hard
        # breaks. With ``/teacher on`` and no goal, show a dim hint instead.
        if self._active_goal:
            _g = " ".join(self._active_goal.split())
            goal_text.update(f"🎯 [bold cyan]{_g}[/]")
        else:
            goal_text.update("🎯 [dim]no goal — supervising on /teacher on[/]")
        _t = "[bold magenta]reviewing…[/]" if self._teacher_reviewing else f"[magenta]{self._teacher_model or 'on'}[/]"
        teacher_row.update(f"🧑‍🏫 teacher: {_t}")

    def _animate_status(self) -> None:
        """Advance the footer spinner one frame while a worker is in flight.

        No-op when ``_worker_state`` is empty (idle) so the timer is
        essentially free outside an active turn. The spinner exists so
        the operator can tell the difference between "TUI is alive,
        upstream is slow or a tool is running" and "TUI is hung" —
        previously the static ``▶ streaming…`` / ``🔧 running tool``
        label gave no motion cue when chunks paused.
        """
        if not self._worker_state:
            return
        self._anim_tick += 1
        self._refresh_status_bar()

    def _set_status(self, msg: str) -> None:
        """Override: write transient acks to the events row above
        the footer.

        Transient messages (``done · …``, ``session created``,
        ``reasoning expanded``, command-failure replies, etc.) land
        on ``#status-events`` — a single row dedicated to one-shot
        feedback. The persistent ``#status-footer`` stays identity-
        only (state · agent · mode · ctx · model · upstream), so
        the operator's at-a-glance signals never wash out under
        ack noise.

        ``_turn_state`` is kept up to date so existing call sites
        that read it for diagnostic logging keep working, but it's
        no longer rendered into the footer.
        """
        self._turn_state = msg or ""
        try:
            self.query_one("#status-events", Static).update(msg or "")
        except Exception:  # noqa: BLE001
            # Widget not yet mounted (test harness, mid-compose);
            # fall through silently — the footer is repainted
            # separately and won't pick up the lost ack.
            pass

    def _set_worker_state(
        self,
        msg: str,
        *,
        detail: str = "",
    ) -> None:
        """Set the always-visible worker-state indicator on the footer,
        with an optional ``detail`` blurb that goes to ``#status-events``
        (the row above) instead of bleeding into the identity strip.

        Empty ``msg`` clears the footer state (turn done / nothing in
        flight). ``detail`` is the long-form transient (``mode=…  new
        session  ESC to abort``) that earlier code shoved straight into
        ``msg`` and ended up smearing the footer's `state · agent ·
        mode · ctx · model · upstream` strip into one unreadable line.
        Splitting the surfaces keeps the footer at-a-glance readable
        while the transient hint still appears (one row up).

        Call shape going forward:
            self._set_worker_state(
                "[bold green]▶ streaming…[/bold green]",
                detail="[dim]ESC to cancel[/dim]",
            )
        """
        self._worker_state = msg or ""
        # Reset the footer spinner frame whenever the state label
        # changes (``▶ streaming…`` → ``🔧 running tool: bash`` →
        # ``▶ streaming…`` …) so the operator gets a clean visual
        # restart on each phase transition rather than the spinner
        # smearing across phases at whatever frame it happened to be
        # on.
        self._anim_tick = 0
        if detail or msg:
            try:
                self.query_one("#status-events", Static).update(detail)
            except Exception:  # noqa: BLE001
                pass
        else:
            # Clearing both at once: wipe the events row too so a stale
            # "ESC to abort" detail doesn't outlive the worker state.
            try:
                self.query_one("#status-events", Static).update("")
            except Exception:  # noqa: BLE001
                pass
        self._refresh_status_bar()
