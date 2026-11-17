"""AgentChatApp lifecycle: init/mount/health/reconnect/exit.

Extracted from app.py (de-monolith); behavior-preserving.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Static

from sharedagents_tui.helpers import (
    _IS_SSH,
    _enable_kitty_keyboard_enhancements,
    _gateway_url_and_token,
)
from sharedagents_tui.widgets.input import (
    MessageInput,
)

HEALTH_POLL_INTERVAL_S = 15.0  # /v1/control/status poll cadence


class LifecycleMixin:
    """AgentChatApp lifecycle: init/mount/health/reconnect/exit."""

    def __init__(self, *, agent_cfg, endpoint_override: str | None = None) -> None:
        super().__init__(
            endpoint=agent_cfg.upstream_endpoint,
            model=agent_cfg.upstream_model or agent_cfg.name,
            system=agent_cfg.system,
            temperature=agent_cfg.temperature,
            enable_thinking=agent_cfg.enable_thinking,
        )
        self.agent_cfg = agent_cfg
        # Cached sibling-agent registry for the `/<name> <task>` auto-
        # dispatch fallthrough in `on_input_submitted`. Source: the
        # global ``~/.sharedagents/config.json`` ``acp_agents`` map (this
        # isn't a dataclass field on ``AgentConfig`` so we read the
        # raw dict directly — same source the ``delegate_agent``
        # runner uses, so the slash registry can never disagree with
        # the dispatch path). Excludes the current agent's own name.
        try:
            from sharedagents_agent.agent import load_global_config

            siblings_cfg = (load_global_config() or {}).get("acp_agents") or {}
            my_name = getattr(agent_cfg, "name", "") or ""
            self._sibling_agents_cache: set[str] = {
                str(name) for name in siblings_cfg if isinstance(name, str) and name != my_name
            }
        except Exception:  # noqa: BLE001
            self._sibling_agents_cache = set()
        # The resolved upstream URL the next-created session should
        # bind to. None = let the agent daemon use its configured
        # primary. Set initially from `--endpoint <alias|url>`; the
        # `/model` slash command rebinds the LIVE session via a
        # daemon-side mutation rather than touching this field.
        self._endpoint_override: str | None = endpoint_override
        # Tell ChatApp.on_mount to skip its legacy direct-spawn startup —
        # Textual would otherwise also fire it via MRO and stomp the bar.
        self._skip_legacy_setup = True
        self._client = None  # populated in on_mount
        self._sid: str | None = None
        self._current_turn_id: str | None = None
        # Composite status-bar state. Three layers, rendered together by
        # `_refresh_status_bar`:
        #   1. `_worker_state` — ALWAYS-VISIBLE indicator while a turn is in
        #      flight: "▶ streaming…", "🔧 running tool: bash", "🤖 claude
        #      running…". Set by the dispatch loop, cleared at end-of-turn.
        #      Lives BESIDE `_turn_state` so an unrelated transient ("reasoning
        #      expanded") doesn't clobber the streaming-is-alive cue — that's
        #      the user's primary signal that the agent isn't stuck.
        #   2. `_turn_state` — short transient feedback ("done · …", "session
        #      created", "compacting…", "reasoning expanded"). Updated by
        #      `_set_status`.
        #   3. health portion (refreshed by the 15s poll) + warn.
        self._health: dict | None = None
        self._health_error: str | None = None
        self._worker_state: str = ""
        self._turn_state: str = ""
        self._upstream_warn: str | None = None
        # Set after /restart reconnect when the rehydrated session model is
        # a paid (cloud) upstream but the agent config's default model is
        # local — gate the next turn until the operator acknowledges.
        self._model_mismatch_paid: bool = False
        # Pending-turn queue. New user prompts and /claude calls submitted
        # while a turn is already in flight land here as `(kind, payload)`
        # tuples — `kind` is "prompt" or "claude". Drained sequentially in
        # the order they were submitted by `_maybe_start_next` after each
        # turn ends. Without this queue the second Enter while the first
        # turn is still streaming would race the dispatcher: the busy
        # check rejected the input and silently dropped the user's text.
        from collections import deque as _deque

        self._turn_queue: _deque = _deque()
        # Durable queue items from the last session_queue_list fetch.
        # Populated async; read by _refresh_queue_display to merge with
        # _turn_queue rows. Keyed as list of {queue_id, text, source, author}.
        self._durable_queue_items: list = []
        self._health_timer = None
        # Wall-clock timestamp of the last successful `/v1/control/status`
        # poll. Drives the "hb Xs" heartbeat-age display in the status
        # bar so the user can tell at a glance whether the agent is
        # currently reachable. None means we haven't successfully polled
        # yet (or the most recent poll failed).
        self._last_health_at: float | None = None
        # Latest authoritative prompt_tokens reported by the worker; the
        # status bar renders this as `ctx <last>/<max>` so the user can
        # see how close they are to the upstream context limit. None
        # until the first round's metrics arrive.
        self._last_prompt_tokens: int | None = None
        self._last_context_window: int | None = None
        # Display-refresh tick — re-renders the status bar so the
        # heartbeat age counts up between health polls. Doesn't fetch
        # anything new; SSH-aware (slower interval to save bandwidth).
        self._display_timer = None
        # Footer spinner. ``_anim_tick`` advances on a fast timer while
        # ``_worker_state`` is non-empty so the operator sees motion on
        # the state segment — the user's primary "TUI isn't frozen"
        # signal during long tool calls or stretches where the upstream
        # pauses between streaming chunks. Idle = timer is a no-op.
        self._anim_tick: int = 0
        self._anim_timer = None
        # Cached shells-count for the footer's ``shells N`` segment.
        # The spinner ticks every 0.25 s and calls _refresh_status_bar,
        # so we mustn't re-read the per-session monitors.json sidecar
        # on every paint — the cache TTL throttles disk IO to ~3 s
        # while still updating fast enough that a freshly-spawned
        # monitor shows up in the footer without the operator having
        # to open the /shells modal.
        self._shell_count_cache: tuple[int, int] = (0, 0)  # (live, exited)
        self._shell_count_at: float = 0.0
        # Permission mode for this TUI session. Defaults to whatever the
        # agent's config says; cycled by Shift+Tab.
        self._permission_mode: str = getattr(agent_cfg, "permission_mode", "auto") or "auto"
        # Teacher-model supervisor visibility (see `_recompute_teacher`).
        # `_teacher_enabled` is the EFFECTIVE supervision gate the widget
        # shows on — recomputed from the policy, the `/teacher on|off`
        # override, and the active goal (a goal OR `/teacher on` makes it
        # active; `/teacher off` or no-goal-no-override hides it).
        # `_teacher_reviewing` is
        # True while a `delegate_teacher` review is in flight (set on its
        # tool_call, cleared on its tool_result) so the footer shows
        # `teacher: reviewing…` instead of looking like a frozen turn during
        # the 30–120s the deepseek review takes. `_teacher_review_cids` tracks
        # the in-flight review call_ids so overlapping reviews clear correctly.
        try:
            _tp = (agent_cfg.teacher_policy() if hasattr(agent_cfg, "teacher_policy") else {}) or {}
        except Exception:  # noqa: BLE001
            _tp = {}
        # Teacher supervision policy (config defaults) + per-session live
        # state. ``_teacher_enabled`` is the EFFECTIVE gate the widget shows
        # on — recomputed by ``_recompute_teacher`` from the policy, the
        # ``/teacher on|off`` override, and whether a goal is active (the
        # teacher fires goal-driven OR on explicit ``/teacher on``). Seeded
        # dormant: policy-enabled alone (no goal, no override) is NOT active
        # when ``require_goal`` holds.
        self._teacher_policy_enabled: bool = bool(_tp.get("enabled"))
        self._teacher_require_goal: bool = bool(_tp.get("require_goal", True))
        self._teacher_override: str = ""  # "on"/"off"/"" (/teacher on|off)
        self._teacher_model_override: str = ""  # /teacher model <alias>
        try:
            _trow = (agent_cfg.teacher_model() if hasattr(agent_cfg, "teacher_model") else None) or {}
        except Exception:  # noqa: BLE001
            _trow = {}
        self._teacher_cfg_model: str = str(_trow.get("id") or _trow.get("alias") or "")
        self._teacher_model: str = self._teacher_cfg_model
        self._teacher_enabled: bool = False
        self._teacher_reviewing: bool = False
        self._teacher_review_cids: set[str] = set()
        # Active goal/completion-condition for the bound session (set by
        # `/goal <condition>`, cleared by `/goal clear`). Drives the footer's
        # `goal:` segment, and the teacher is only ACTIVE while this is set —
        # the footer reflects that. Seeded from the resumed session below if it
        # carries one.
        self._active_goal: str = ""
        # `/claude <prompt>` chains across slash invocations within a single
        # sharedagents session via Claude Code's `--resume <session_id>`. The id
        # is captured from the first call's stream-json events and reused
        # for subsequent calls. None until the first /claude runs.
        self._claude_session_id: str | None = None
        # Server-side `delegate_claude` tool runner's turn id, captured
        # from the SSE response of `_run_claude`. Used by ESC to stop
        # the running tool through `turn_stop`. None when no /claude
        # call is in flight.
        self._claude_turn_id: str | None = None
        # (sharedagents-context injection is backend-owned — the gateway's
        # LocalLLMEngine._run_direct_tool_call builds the preamble from
        # the server-side session for every frontend. The TUI no longer
        # tracks a per-delegate "synced turn index".)
        #
        # `/codex <prompt>` mirrors `/claude` against OpenAI's Codex CLI:
        # `_codex_thread_id` is captured from `thread.started` events
        # and reused via `codex exec resume <id>` on follow-ups;
        # `_codex_turn_id` lets ESC stop the in-flight tool runner.
        self._codex_thread_id: str | None = None
        self._codex_turn_id: str | None = None
        # `/antigravity <prompt>` mirrors `/claude` and `/codex` against the
        # Google Antigravity CLI. ``_antigravity_session_id`` is the Gemini-side
        # UUID surfaced via ``init`` events and reused via
        # ``--resume <id>`` on follow-ups.
        self._antigravity_session_id: str | None = None
        self._antigravity_turn_id: str | None = None
        # Number of turns we've already rendered into the transcript
        # (live + replayed). The poll timer compares this against the
        # session JSON's actual turn count to detect turns that landed
        # on disk while we weren't streaming — typically a Telegram or
        # API client writing to the same session — and renders them
        # inline so the user sees both sides of a multi-frontend
        # conversation in real time.
        # Inverted submit+subscribe driver state (spec §7 / §16 step 2).
        # ``_use_event_stream`` is flipped on in ``on_mount`` / reconnect
        # whenever a real gateway URL is configured; it switches the live
        # turn driver off the ACP ``session_turn`` iterator and onto
        # submit_prompt + subscribe_events + _render_event. Default off so
        # the stdio / unit-test path keeps driving ``session_turn``.
        self._use_event_stream: bool = False
        # Per-call ``ok`` flag stash: the canonical stream splits a tool
        # result into ``tool_result_meta`` (carries ok) + ``tool_result``
        # (carries body). ``_render_event`` joins them via this map.
        self._event_tool_ok: dict[str, bool] = {}
        # Canonical streams may include a full ``final_content`` on
        # ``turn_ended``. Track deltas per turn so the terminal fallback can
        # render a terminal-only frame without duplicating content on
        # richer streams that already emitted ``text_delta`` frames.
        self._event_text_by_turn: dict[str, str] = {}
        # Single per-session subscribe-tail task: the unified live == replay
        # == observe renderer (spec §8). It tails the session log from the
        # current byte offset and feeds EVERY frame (incl. externally-driven
        # turns from Telegram / web / another TUI) through the SAME
        # ``_render_event`` → ``_stream_via_agent`` dispatch the initiator
        # uses, so a rendered turn now renders with full fidelity (tools,
        # reasoning, subagents, abort/error) instead of the old folded
        # placeholder. Replaces ``_poll_session_for_external_turns``.
        self._render_log_task = None
        # High-water turn_idx the renderer has rendered up to (anything below
        # is already on screen). Re-anchored by ``_bind_session_view`` on a
        # ``/sessions`` switch; ``_render_generation`` bumps to bounce the
        # tail onto the freshly-bound session.
        #
        # NOTE: ``turn_idx`` is NOT a unique turn identity. When several
        # prompts are submitted to a busy session in quick succession (a
        # Telegram message burst, the OpenAI-compat API, another TUI), the
        # orchestrator drains them back-to-back and stamps EVERY resulting
        # ``turn_started`` with the SAME ``turn_idx`` — the engine's turn
        # counter does not advance per queued prompt. A turn_idx high-water
        # therefore collapsed "send 5 essays" down to the first essay only.
        # The renderer now de-dups on the SSE frame's monotonic byte ``offset``
        # (``_rendered_offset``), which is distinct for every event; the
        # turn_idx field below is retained for telemetry/back-compat only.
        self._rendered_turn_count: int = 0
        # Byte offset into the bound session's JSONL up to which the renderer
        # has already handled events (history replayed at bind + turns it has
        # itself rendered + local turns the initiator owns). A ``turn_started``
        # frame past this offset is a not-yet-shown external turn. Re-anchored
        # to the session log's end on every bind / switch.
        self._rendered_offset: int = 0
        self._render_generation: int = 0
        # Cooperative stop for the active subscribe-tail (set by
        # ``_bounce_session_render`` so idle hb can exit without a hard cancel).
        self._session_render_cancel: Any | None = None
        # Armed by ``_send_user_turn`` the instant a LOCAL turn is submitted,
        # before the initiator's first stream frame sets ``_current_turn_id``.
        # Closes the submit→first-frame race where the renderer could otherwise
        # render the local turn too (double-render). Cleared by the initiator
        # once it owns the turn, or by the renderer once it has skipped it.
        self._local_submit_pending: bool = False
        # True while the footer shows the passive-render "rendering…" state
        # (another frontend is driving an in-flight turn we're reflecting).
        self._rendering_logged_turn: bool = False

    async def on_mount(self) -> None:
        from sharedagents_tui.agent_client import AgentClient

        gateway_url, gateway_token = _gateway_url_and_token()
        self._client = AgentClient(
            self.agent_cfg.url,
            agent_name=self.agent_cfg.name,
            gateway_url=gateway_url,
            gateway_token=gateway_token,
        )
        # A real gateway URL means we can use the inverted submit+subscribe
        # turn driver (spec §16 step 2). Stdio / test mode (no URL) keeps
        # the legacy ``session_turn`` path.
        self._use_event_stream = bool(gateway_url)
        self._wire_client_callbacks(self._client)
        # Reconnect watcher state. The watcher fires every
        # ``RECONNECT_WATCH_INTERVAL_S`` and rebuilds the client when
        # the stdio child has died. The lock dedupes concurrent
        # callers (eager poll-driven retry + scheduled tick) so we
        # never spawn two stdio children for one disconnect — that
        # used to leak an orphaned acp-stdio child each time both
        # paths fired in the same loop tick.
        self._reconnect_lock: asyncio.Lock = asyncio.Lock()
        self._reconnect_attempts: int = 0
        self._last_reconnect_at: float = 0.0
        _enable_kitty_keyboard_enhancements(self)
        self.query_one(MessageInput).focus()

        # 0. First-run background local-model install in flight? Show a
        #    NON-blocking indicator (you keep chatting on the free model) and
        #    notify when the local model is ready to /model-switch to.
        try:
            from sharedagents_model import local_gpu as _lg

            if _lg.read_provision_marker() is not None:
                self._set_status(
                    "⏳ installing local model in the background — "
                    "`sharedagents monitor` to watch; `/model` to switch when ready"
                )
                self._provision_timer = self.set_interval(3.0, self._provision_tick)
        except Exception:  # noqa: BLE001
            pass

        # 1. Initial health probe so the status bar isn't blank for 15s.
        await self._poll_health()

        # 2. Schedule the recurring health poll. The status bar's durable
        #    portion (agent up/down, upstream up/down, model) is refreshed
        #    on every tick; transient turn-state ("streaming…", "done · …",
        #    "error: …") is set independently by _set_status.
        self._health_timer = self.set_interval(HEALTH_POLL_INTERVAL_S, self._poll_health)

        # 2.1 Reconnect watcher — runs at 2 s on a fresh boot, longer
        #     under SSH to keep link bandwidth modest. Detects a dead
        #     stdio child WITHOUT a network round-trip (just checks
        #     ``self._client._proc.returncode``); when dead, rebuilds
        #     the client and reattaches via ``session/load`` to keep
        #     the operator's active session id valid.
        reconnect_tick = 5.0 if _IS_SSH else 2.0
        self._reconnect_timer = self.set_interval(
            reconnect_tick,
            self._watch_connection,
        )

        # 2.5 Schedule a faster, *no-fetch* tick so the heartbeat-age
        #    indicator (`hb Xs`) counts up between polls. SSH-aware:
        #    slower interval so we don't waste link bandwidth on a
        #    1-second-resolution age display.
        display_tick = 10.0 if _IS_SSH else 5.0
        self._display_timer = self.set_interval(display_tick, self._refresh_status_bar)

        # 2.55 Spinner tick — fast cadence so the operator sees motion
        #      on the worker-state segment while a turn is in flight.
        #      The handler bails out immediately when ``_worker_state``
        #      is empty (idle), so this is essentially free except
        #      during an active turn. SSH-aware so we don't burn
        #      bandwidth on idle frames over a slow link.
        anim_tick = 0.5 if _IS_SSH else 0.25
        self._anim_timer = self.set_interval(
            anim_tick,
            self._animate_status,
        )

        # 2.6 Cross-frontend renderer (spec §8, unify live == replay ==
        #     observe). Instead of polling the folded session every ~1.5s
        #     and rendering a coarse placeholder for in-flight external
        #     turns, a single subscribe-tail task feeds EVERY canonical
        #     frame (incl. turns driven by Telegram / web / another TUI)
        #     through the SAME ``_render_event`` → dispatch the initiator
        #     uses — so a rendered turn renders with full fidelity (tools,
        #     reasoning, subagents, abort/error), woken ~5ms by
        #     ``session_live`` rather than on a poll tick. Only armed in
        #     gateway mode; stdio / test mode has no shared log to tail.
        if self._use_event_stream:
            self._render_log_task = self.run_worker(
                self._render_session_log(),
                exclusive=False,
                name="session-render",
            )

        # 3. If the agent is reachable, try to create a session. If it fails
        #    (e.g. transient upstream blip), keep the error visible —
        #    `_send_user_turn` will retry rather than locking us out.
        if self._health is not None:
            await self._ensure_session(initial=True)

    def _provision_tick(self) -> None:
        """Non-blocking poll of the background local-model install. Updates the
        events line with the download bar; on ready/error, notifies once and
        stops. The free model stays active throughout — no surprise switch."""
        try:
            from sharedagents_model import local_gpu as _lg

            st = _lg.provision_status()
        except Exception:  # noqa: BLE001
            st = None
        if st is None:
            # Marker cleared → the worker finished successfully (it only clears
            # on a serving model; failures keep the marker with state=error).
            # Don't go silent — tell the user the local model is ready.
            self._set_status("✓ local model ready — switch to it with `/model`")
            self._stop_provision_timer()
            return
        if st.get("server_up"):
            self._set_status(f"✓ local model ready: {st.get('served')} — switch with `/model`")
            self._stop_provision_timer()
            return
        if st.get("phase") == "error":
            self._set_status(
                f"✗ local model install failed: {st.get('error') or 'unknown'} "
                "(still on the free model; `sharedagents monitor` for details)"
            )
            self._stop_provision_timer()
            return
        served = st.get("served") or "model"
        bar = (st.get("bar") or "provisioning…").strip()
        phase = st.get("phase")
        if phase == "installing_vllm":
            # vLLM itself is being installed (bare box) — name it explicitly so a
            # long, weight-download-free wait doesn't look stuck.
            msg = f"⏳ {bar} (then downloads {served})"
        elif phase == "downloading":
            msg = f"⏳ downloading {served} — {bar}"
        elif phase == "loading":
            msg = f"⏳ {served}: {bar}"
        else:
            msg = f"⏳ installing {served} — {bar}"
        self._set_status(f"{msg}  (/model to switch when ready)")

    def _stop_provision_timer(self) -> None:
        t = getattr(self, "_provision_timer", None)
        if t is not None:
            try:
                t.stop()
            except Exception:  # noqa: BLE001
                pass
            self._provision_timer = None

    async def _poll_health(self) -> None:
        """One tick of the 15s health poll — repaints the status bar.

        On a connection failure, kicks the reconnect watcher so it
        notices the dead client even before its own 2 s tick fires.
        Without this, a network failure during a slow turn would
        leave the user staring at "error: connection refused" until
        the next watcher tick (up to 5 s on SSH).
        """
        if self._client is None:
            return
        try:
            # Probe the session's active upstream, not the config default.
            sess_endpoint = (getattr(self.session, "endpoint", "") or "").rstrip("/")
            self._health = await self._client.status(
                endpoint=sess_endpoint or None,
            )
            self._health_error = None
            self._last_health_at = time.monotonic()
        except Exception as e:  # noqa: BLE001
            self._health = None
            self._health_error = str(e) or repr(e)
            # Health failure usually means the stdio child is gone or
            # ACP transport is closed. Schedule an immediate reconnect
            # attempt rather than waiting for the next watcher tick.
            self.run_worker(
                self._watch_connection(),
                exclusive=False,
                name="reconnect-eager",
            )
        self._refresh_status_bar()

    async def _watch_connection(self) -> None:
        """Reconnect watcher tick.

        Cheap path: if the client looks live (``is_connected()``),
        return. Otherwise acquire ``self._reconnect_lock`` and
        rebuild — the lock dedupes concurrent callers (eager path
        from ``_poll_health`` + scheduled timer tick) so we never
        spawn two stdio children for one disconnect.

        Backoff: 1, 2, 4, 8, … seconds, capped at 30 s. Reset on
        success. We don't sleep here — we just *skip* if the prior
        attempt was less than the current backoff window ago.
        """
        if self._client is None:
            return
        if self._client.is_connected():
            # Healthy — reset backoff state.
            if self._reconnect_attempts:
                self._reconnect_attempts = 0
            return
        # Lock dedupes concurrent callers. ``locked()`` skip path
        # avoids piling up a queue of waiters behind a long
        # in-flight reconnect.
        if self._reconnect_lock.locked():
            return
        async with self._reconnect_lock:
            # Recheck inside the lock — another caller may have
            # already healed the client between our pre-lock peek
            # and acquisition.
            if self._client is not None and self._client.is_connected():
                self._reconnect_attempts = 0
                return
            backoff_s = min(2**self._reconnect_attempts, 30)
            if time.monotonic() - self._last_reconnect_at < backoff_s:
                return
            self._reconnect_attempts += 1
            self._last_reconnect_at = time.monotonic()
            await self._reconnect()

    async def _reconnect(self) -> None:
        """Tear down the dead ``AgentClient``, spawn a fresh one,
        and re-attach to the active session via ``session/load``.

        Best-effort: a reattach failure (e.g. the agent's session
        store rolled away) leaves ``self._sid = None`` so the next
        operation will create a fresh session instead of failing
        in a confusing way. Status bar tracks state across the
        reconnect window so the operator sees what's happening.
        """
        from sharedagents_tui.agent_client import AgentClient

        self._set_status(
            f"⟳ reconnecting (attempt {self._reconnect_attempts})…",
        )
        # Tear down the dead client. ``close`` will SIGTERM/wait the
        # zombie child if any.
        try:
            await self._client.close()
        except Exception:  # noqa: BLE001
            pass
        # Build a fresh client against the gateway. The gateway owns
        # every agent's stdio child; reconnecting just rebuilds the
        # HTTP transport, no process spawn happens here.
        gateway_url, gateway_token = _gateway_url_and_token()
        new_client = AgentClient(
            self.agent_cfg.url,
            agent_name=self.agent_cfg.name,
            gateway_url=gateway_url,
            gateway_token=gateway_token,
        )
        try:
            # Probe by calling status() — that triggers the lazy
            # ``_ensure_connected`` path which spawns the child.
            await new_client.status()
        except Exception as e:  # noqa: BLE001
            # Mirror the probe failure into the footer state so the
            # operator doesn't see a stale-green ``upstream ok`` /
            # missing-badge footer above a red ``reconnect failed:
            # …`` events row. _poll_health only ticks every 15 s and
            # only writes _health_error from its own probe — between
            # a previously-successful poll and the next tick, a
            # mid-cycle drop+reconnect attempt would leave the
            # footer reading the connection as healthy while the
            # events row carried the live failure string. Now both
            # surfaces flip together on the same probe.
            self._health_error = str(e) or repr(e)
            self._health = None
            self._set_status(
                f"reconnect failed: {e} — retrying in {min(2**self._reconnect_attempts, 30)}s",
            )
            self._refresh_status_bar()
            try:
                await new_client.close()
            except Exception:  # noqa: BLE001
                pass
            return
        # New client up. Try to re-attach to the existing session id;
        # if that fails (session_load not supported, or the agent
        # forgot the sid), fall back to a fresh session so the
        # operator can keep working without restarting the TUI.
        self._client = new_client
        self._wire_client_callbacks(self._client)
        if self._sid is not None:
            try:
                await new_client.attach_session(
                    self._sid,
                    cwd=os.getcwd(),
                )
                try:
                    state = await new_client.get_session_state(self._sid)
                    mode = state.get("effective_mode") or {}
                    if isinstance(mode, dict) and mode.get("id"):
                        self._permission_mode = str(mode["id"])
                except Exception as state_exc:  # noqa: BLE001
                    self._log.debug("session state refresh failed: %s", state_exc)
                self._set_status(
                    f"✓ reconnected — session {self._sid[:12]} resumed",
                )
                # §9b: toast on model mismatch after session rehydration.
                _saved_model = getattr(self.session, "model", "") or ""
                _cfg_alias = getattr(self.agent_cfg, "upstream_alias", "") or ""
                if _saved_model and _cfg_alias and _saved_model != _cfg_alias:
                    self._upstream_warn = (
                        f"model rehydrated as {_saved_model} from saved "
                        f"session — type /model {_cfg_alias} to switch back"
                    )
                    self._set_status(self._upstream_warn)
                    # Determine if saved model is paid (apiKeyEnv) and
                    # config default is local (no apiKeyEnv).
                    _models = getattr(self.agent_cfg, "models", None) or []
                    _saved_row = next(
                        (m for m in _models if m.get("alias") == _saved_model or m.get("id") == _saved_model),
                        {},
                    )
                    _cfg_row = next(
                        (m for m in _models if m.get("alias") == _cfg_alias or m.get("id") == _cfg_alias),
                        {},
                    )
                    _saved_paid = bool(_saved_row.get("apiKeyEnv"))
                    _cfg_local = not bool(_cfg_row.get("apiKeyEnv"))
                    if _saved_paid and _cfg_local:
                        self._model_mismatch_paid = True
                        self._set_status(
                            f"session was saved on {_saved_model} (paid) — "
                            f"config default is {_cfg_alias} (local). "
                            "Type /model to switch or submit to continue on saved model."
                        )
            except Exception as e:  # noqa: BLE001
                # Reattach failed — the agent might have lost the
                # session record across the restart. Drop sid so
                # the next op spawns a fresh one. Reset the
                # plan-checklist and harness-task footers too, since
                # both lived only in the prior session and would
                # otherwise show stale rows above the new prompt.
                old_sid = self._sid
                self._sid = None
                self.session = None
                self._rendered_turn_count = 0
                self._rendered_offset = 0
                try:
                    self.clear_plan_checklist()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    self._clear_harness_tasks()
                except Exception:  # noqa: BLE001
                    pass
                self._set_status(
                    f"reconnected but couldn't resume {old_sid}: {e}. next message starts a fresh session."
                )
        else:
            self._set_status("✓ reconnected — no active session")
        # Refresh status bar so the durable footer reflects the new
        # health probe immediately rather than on the next 15 s tick.
        await self._poll_health()

    def _wire_client_callbacks(self, client) -> None:
        """Install AskUserQuestion + permission UI hooks on a fresh
        ``AgentClient``. Called from each instantiation path — startup and
        reconnect-watcher — so newly built
        clients pop the modals instead of falling back to the
        synthetic auto-answer / deny-by-default."""
        try:
            client.set_ask_user_callback(self._on_ask_user)
        except Exception:  # noqa: BLE001
            # Older clients may lack the method on partial upgrades;
            # don't crash the App for a missing hook.
            pass
        try:
            client.set_permission_callback(self._on_request_permission)
        except Exception:  # noqa: BLE001
            pass

    def _sweep_gateway_children(self, *, reason: str) -> int:
        """Reap the gateway's recursive descendants. Returns the
        number of PIDs SIGTERMed (0 when there's nothing to kill).

        Shared by ``/restart`` (always) and ``/new`` (so a fresh
        session doesn't race an orphaned tool_runner from the prior
        turn — observed live: pid 570457 survived three
        abort attempts and required manual ``kill -9``).
        """
        try:
            from sharedagents_core.restart import (
                gateway_pid,
                kill_gateway_children,
            )
        except ImportError as e:
            self._set_status(f"sweep helper missing: {e}")
            return 0
        gw_pid = gateway_pid()
        if gw_pid is None:
            return 0
        result = kill_gateway_children(gw_pid, grace_s=3.0)
        killed = result.get("killed") or []
        if not killed:
            return 0
        # Compact one-line ack: "swept 3 children (2 tool_runner, 1
        # subagent) on /restart". Operator-facing; the gateway log
        # already carries the per-PID detail via
        # kill_gateway_children's INFO/WARNING records.
        by_kind: dict[str, int] = {}
        for rec in killed:
            by_kind[rec["kind"]] = by_kind.get(rec["kind"], 0) + 1
        breakdown = ", ".join(f"{n} {k}" for k, n in sorted(by_kind.items(), key=lambda kv: -kv[1]))
        n = len(killed)
        plural = "child" if n == 1 else "children"
        self._set_status(f"swept {n} {plural} ({breakdown}) on /{reason}")
        return n

    def _dispatch_restart(self, *, hard: bool = False) -> None:
        """``/restart`` — sweep gateway children (default) or bounce
        the gateway (``--hard``), keep the TUI up.

        Default mode (no flag) is a **child sweep only**: walks the
        gateway PID's recursive descendants and SIGTERMs every one
        (tool_runners, subagents, anything spawned under the gateway).
        The gateway parent stays up, so the ACP connection survives
        and the next prompt lands on the same gateway — minimum
        disruption. Before 2026-05-16 ``/restart`` only invoked
        ``restart_agent_detached`` which collapses to a SIGTERM of
        the gateway parent; ``TurnState.stop`` reaped only the
        *tracked* worker / tool runner per active session, so any
        parallel batch leftovers, mid-flight subagent runners, or
        runners whose registration had been cleared survived and
        kept writing. Live evidence 2026-05-16: pid 570457
        (tool_runner child of gateway pid 172813) outlived Esc x3 +
        ``/restart`` + ``/new`` and required manual ``kill -9``.

        ``--hard`` runs the sweep AND then schedules a detached
        gateway-bounce via
        :func:`sharedagents_core.restart.restart_agent_detached` with
        ``hard=True`` — which, under systemd, also bounces the
        standalone Telegram bot service so one command brings the whole
        stack back gracefully. The TUI
        loses its connection for a few seconds; we schedule an
        explicit :meth:`_reconnect` after the gateway's typical
        respawn window — the connection-watch loop alone can't
        recover because (a) it short-circuits when ``self._client is
        None`` (the pre-mount guard), and (b) the HTTP-mode
        ``is_connected()`` is just a "client exists" check, never a
        live probe, so the watcher can't even detect a dead remote.
        Observed: prompt after ``/restart`` crashed with
        ``'NoneType' object has no attribute 'session_turn'``.
        """
        # Step 1 (NEW 2026-05-16): always reap the gateway's recursive
        # descendants first. Idempotent: zero children == no-op.
        n_swept = self._sweep_gateway_children(reason="restart")
        if not hard:
            # Default /restart: child-sweep only, gateway stays up.
            # Always render a transcript result line so the operator can
            # see WHAT the command did — _sweep_gateway_children's
            # status row only fires when killed > 0, so without this
            # zero-children sweeps left the operator with no feedback
            # at all and they would retype /restart.
            if n_swept == 0:
                msg = (
                    "✔ /restart — no in-flight subprocesses to sweep "
                    "(gateway stayed up). "
                    "Use /restart --hard to bounce the gateway + Telegram bot."
                )
            else:
                plural = "child" if n_swept == 1 else "children"
                msg = (
                    f"♻ /restart — swept {n_swept} {plural}; "
                    "gateway stayed up. "
                    "Use /restart --hard to bounce the gateway + Telegram bot."
                )
            try:
                transcript = self.query_one("#transcript", VerticalScroll)
                transcript.mount(Static(msg, classes="claude-msg"))
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            from sharedagents_core.restart import restart_agent_detached
        except ImportError as e:
            self._set_status(f"restart helper missing: {e}")
            return
        agent_name = getattr(self.agent_cfg, "name", "jarvis")
        result = restart_agent_detached(agent_name, hard=True)
        if not result.get("ok"):
            self._set_status(f"restart failed: {result.get('error')}")
            return
        # Drop the live ACP client so the next operation rebuilds
        # against the respawned gateway.
        self._client = None
        # ``--hard`` also bounces the standalone Telegram bot when
        # systemd manages it (result["telegram"]); say so honestly.
        svc = "sharedagents gateway + Telegram bot" if result.get("telegram") else "sharedagents gateway"
        self._set_status(f"♻ /restart --hard: swept {n_swept} children, bouncing {svc} (~3s)…")
        try:
            transcript = self.query_one("#transcript", VerticalScroll)
            transcript.mount(
                Static(
                    f"♻ swept {n_swept} in-flight subprocesses\n"
                    f"♻ restart scheduled — {svc}\n"
                    "Reconnecting once the gateway is back up.",
                    classes="claude-msg",
                ),
            )
        except Exception:  # noqa: BLE001
            pass
        # Schedule the reconnect. Wait > the restart_agent_detached
        # helper's typical respawn window (~3s) plus a small buffer
        # so the new daemon is bound to its port by the time we
        # probe. The reconnect path will fall back to its own
        # exponential backoff if the gateway is still warming up.
        try:
            self.set_timer(4.0, self._reconnect_after_restart)
        except Exception:  # noqa: BLE001
            pass

    async def _reconnect_after_restart(self) -> None:
        """Re-establish the AgentClient after a ``/restart``.

        Resets reconnect bookkeeping so the post-restart attempt
        doesn't inherit the previous session's backoff window, then
        defers to :meth:`_reconnect`. If the gateway hasn't bound
        its port yet, ``_reconnect`` sets back to ``self._client =
        None`` on probe failure — so we re-schedule ourselves with
        a small backoff until the gateway is up (or we've burned
        the attempt budget).
        """
        self._reconnect_attempts = 0
        self._last_reconnect_at = 0.0
        attempt = 0
        max_attempts = 8  # ~30s total at 1+2+3+4+5+5+5+5
        while attempt < max_attempts:
            async with self._reconnect_lock:
                await self._reconnect()
            if self._client is not None and self._client.is_connected():
                return
            attempt += 1
            backoff = min(attempt, 5)
            await asyncio.sleep(backoff)
        self._set_status(
            "reconnect after /restart timed out — submit a prompt to retry, or relaunch the TUI",
        )
