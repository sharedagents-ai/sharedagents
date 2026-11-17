"""Session / topic / client-management methods for :class:`AcpTelegramBot`.

Extracted from the :class:`AcpTelegramBot` god-class (de-monolith). This is a
mixin: every method still binds ``self`` and dispatches identically through the
concrete class's MRO. Behaviour is unchanged — the method bodies moved verbatim.

The module-level helpers/constants these methods read stay owned by ``bot.py``
and are imported at the BOTTOM (after the class) so the ``bot`` <-> mixin import
cycle is deadlock-free in either import order.
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

STARTUP_ACP_RETRY_DELAYS_S = (0.5, 1.0, 2.0, 4.0, 8.0, 10.0)


async def _retry_startup_acp_open(
    open_client: Any,
    *,
    sleep: Any = asyncio.sleep,
    retry_delays: tuple[float, ...] = STARTUP_ACP_RETRY_DELAYS_S,
) -> Any:
    """Open an ACP client while the gateway finishes its startup sequence.

    ``After=sharedagents-gateway.service`` orders the Telegram unit after the
    gateway process, but does not mean the gateway is already listening.  A
    bounded retry keeps that normal readiness window out of the error log
    while still surfacing a genuinely unavailable gateway to the existing
    stale-session handling.
    """
    last_error: BaseException | None = None
    for delay in (0.0, *retry_delays):
        if delay:
            await sleep(delay)
        try:
            return await open_client()
        except Exception as exc:  # noqa: BLE001 - preserve existing handling
            last_error = exc
    assert last_error is not None
    raise last_error


class SessionsMixin:
    # ---- per-chat agent resolution -------------------------------------

    async def _session_control_call(
        self,
        sid: str,
        endpoint: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = self._http_gateway_base()
        if not base:
            raise RuntimeError("feature_disabled: session control requires gateway")
        import httpx

        url = f"{base}/control/v1/orchestrator/sessions/{sid}/{endpoint}"
        headers: dict[str, str] = {"X-Sharedagents-Principal": "local"}
        token = getattr(self, "acp_bearer_token", None)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as http:
            response = await http.post(url, json=body or {}) if body is not None else await http.get(url)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code >= 400:
            raise RuntimeError(
                f"{payload.get('code') or response.status_code}: {payload.get('message') or response.text[:200]}"
            )
        return dict(payload) if isinstance(payload, dict) else {}

    async def _get_session_state(self, sid: str) -> dict[str, Any]:
        return await self._session_control_call(sid, "state")

    async def _list_session_modes(self, sid: str) -> list[dict[str, Any]]:
        result = await self._session_control_call(sid, "capabilities")
        rows = result.get("agents") or result.get("available_agent_modes") or []
        return [dict(row) for row in rows if isinstance(row, dict)]

    async def _switch_session_agent(
        self,
        *,
        sid: str,
        target_agent: str,
        target_mode: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "target_agent": target_agent,
            "frontend": "telegram",
        }
        if target_mode:
            params["target_mode"] = target_mode
        return await self._session_control_call(sid, "agent", body=params)

    async def _apply_session_state(
        self,
        chat_id: int,
        thread_id: int | None,
        sid: str,
        state: dict[str, Any],
    ) -> None:
        """Apply the active agent reported for a session."""
        key = self._session_key(chat_id, thread_id)
        before = self.tg_cfg.agent_for(chat_id, thread_id)
        after = str(state.get("active_agent") or before)
        if after == before:
            return
        self.tg_cfg.agent_routes[key] = after
        self._sessions.setdefault(key, {})[after] = sid
        _save_acp_session_map(self.tg_cfg, self._sessions)
        try:
            self.tg_cfg.save()
        except Exception:  # noqa: BLE001
            self._log.debug("agent route save failed chat_key=%s", key)

    def _inflight_key(
        self,
        chat_id: int,
        thread_id: int | None,
        agent: str,
        sid: str,
    ) -> str:
        chat_key = TelegramConfig.chat_key(chat_id, thread_id)
        return f"{chat_key}::{agent}::{sid}"

    def _record_inflight_turn(
        self,
        *,
        chat_id: int,
        thread_id: int | None,
        agent: str,
        sid: str,
        placeholder_id: int | None,
    ) -> str:
        key = self._inflight_key(chat_id, thread_id, agent, sid)
        self._inflight_turns[key] = {
            "chat_id": int(chat_id),
            "thread_id": thread_id,
            "agent": agent,
            "sid": sid,
            "placeholder_id": placeholder_id,
            "last_text": "",
            "started_at": time.time(),
        }
        _save_inflight_turns(self.tg_cfg, self._inflight_turns)
        return key

    def _update_inflight_text(self, key: str | None, text: str) -> None:
        if not key:
            return
        entry = self._inflight_turns.get(key)
        if entry is None:
            return
        entry["last_text"] = text
        _save_inflight_turns(self.tg_cfg, self._inflight_turns)

    def _clear_inflight_turn(self, key: str | None) -> None:
        if not key:
            return
        if self._inflight_turns.pop(key, None) is not None:
            _save_inflight_turns(self.tg_cfg, self._inflight_turns)

    @staticmethod
    def _assistant_text_from_turn(turn: dict[str, Any]) -> str:
        text = str(
            turn.get("content") or turn.get("assistant") or turn.get("final_content") or "",
        ).strip()
        if text:
            return text
        err = turn.get("error")
        if err:
            return f"⚠️ agent error: {err}"
        return ""

    async def _recover_inflight_turns_on_startup(self) -> None:
        if self._api is None or not self._inflight_turns:
            return
        recovered = dict(self._inflight_turns)
        for key, entry in recovered.items():
            chat_id = int(entry["chat_id"])
            thread_id = entry.get("thread_id")
            if thread_id is not None:
                thread_id = int(thread_id)
            sid = str(entry.get("sid") or "")
            agent_name = str(entry.get("agent") or "")  # the session's on-disk owner
            # A record owned by the retired managed `main` can no longer AgentConfig
            # .load("main") after quarantine. Heal the LOAD TARGET to jarvis so a
            # valid config resolves the SHARED sessions dir; the session READ below
            # still uses expected_agent=agent_name (the original owner), which is how
            # the completed turn was attributed on disk. A live
            # custom `main` is not healed (it loads normally). No-op for other agents.
            load_target = agent_name
            if agent_name == "main" and _retired_main_should_heal():
                load_target = "jarvis"
            agent_cfg = self._agent_cfg_cache.get(load_target)
            if agent_cfg is None:
                try:
                    agent_cfg = AgentConfig.load(load_target)
                    self._agent_cfg_cache[load_target] = agent_cfg
                except Exception:  # noqa: BLE001
                    self._log.warning(
                        "inflight recovery: agent load failed agent=%s sid=%s",
                        load_target,
                        sid,
                    )
                    continue
            text = ""
            deadline = time.monotonic() + INFLIGHT_RECOVERY_WAIT_S
            while True:
                try:
                    from sharedagents_session.session_reader import load_session_dict

                    raw = load_session_dict(
                        agent_cfg.sessions_dir,
                        sid,
                        # The session's on-disk owner — NOT the (possibly healed)
                        # load target — so a retired-`main`-owned turn still matches
                        # its persisted agent_name attribution.
                        expected_agent=agent_name,
                    )
                except Exception:  # noqa: BLE001
                    raw = None
                turns = list((raw or {}).get("turns") or [])
                turn = turns[-1] if turns else {}
                if isinstance(turn, dict):
                    text = self._assistant_text_from_turn(turn)
                    if text and not turn.get("in_progress"):
                        break
                if time.monotonic() >= deadline:
                    text = text or str(entry.get("last_text") or "").strip()
                    break
                await asyncio.sleep(INFLIGHT_RECOVERY_POLL_S)
            if not text:
                text = "⚠️ bot restarted while this turn was running; no reply text was recoverable."
            try:
                await self._api.send_message(
                    chat_id,
                    "♻️ recovered reply after bot restart:\n\n" + text,
                    message_thread_id=thread_id,
                )
                self._log.info(
                    "inflight recovery sent chat=%s thread=%s sid=%s len=%d",
                    chat_id,
                    thread_id,
                    sid,
                    len(text),
                )
            except Exception:  # noqa: BLE001
                self._log.exception(
                    "inflight recovery send failed chat=%s sid=%s",
                    chat_id,
                    sid,
                )
                continue
            self._clear_inflight_turn(key)

    def _agent_cfg(
        self,
        chat_id: int,
        thread_id: int | None = None,
    ) -> AgentConfig:
        """Resolve the :class:`AgentConfig` that handles
        ``(chat_id, thread_id)`` and memoize it.

        Routing uses :meth:`TelegramConfig.agent_for`. Agents are loaded lazily;
        a config edit
        on disk is picked up by the next bot restart, not the next
        message (intentional — multi-agent routing means we don't
        want to file-stat every agent on every dispatch).
        """
        name = self.tg_cfg.agent_for(chat_id, thread_id)
        cached = self._agent_cfg_cache.get(name)
        if cached is not None:
            return cached
        cfg = AgentConfig.load(name)
        self._agent_cfg_cache[name] = cfg
        return cfg

    @staticmethod
    def _list_configured_agents() -> list[str]:
        """Walk ``agents_root()`` and return every directory that
        carries a ``config.json`` (i.e. is a configured agent).

        Sorted alphabetically — operators see a stable list across
        ``/agents`` invocations regardless of filesystem walk order.
        """
        from sharedagents_agent.agent import agents_root

        root = agents_root()
        if not root.is_dir():
            return []
        names: list[str] = []
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if (entry / "config.json").is_file():
                names.append(entry.name)
        names.sort()
        return names

    def _is_known_agent(self, name: str) -> bool:
        """Quick membership check for the slash-command dispatcher.

        Memoised in ``_agent_cfg_cache`` once an agent is loaded;
        otherwise probed directly against the on-disk config so a
        freshly-scaffolded agent shows up without a bot restart.
        """
        if not name:
            return False
        if name in self._agent_cfg_cache:
            return True
        try:
            from sharedagents_agent.agent import agents_root

            return (agents_root() / name / "config.json").is_file()
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _session_key(chat_id: int, thread_id: int | None = None) -> str:
        """Compose the persistence key for a (chat, topic) pair.

        Forum topics need separate sessions so messages in different
        topics don't conflate. Non-forum chats and the General topic
        (Telegram doesn't set ``message_thread_id`` there) keep the
        legacy bare-chat-id key, so existing on-disk mappings survive.
        """
        if thread_id is None:
            return str(chat_id)
        return f"{chat_id}:{thread_id}"

    def _stale_key(self, chat_key: str, agent_name: str) -> str:
        """Compose the per-(chat, agent) staleness key.

        ``_stale`` is a flat set of strings; staleness is per-(chat,
        agent) so a chat that switched agents doesn't drag the new
        agent's binding into "needs session_new" the next message.
        """
        return f"{chat_key}@{agent_name}"

    def get_session_id(
        self,
        chat_id: int,
        thread_id: int | None = None,
        agent_name: str | None = None,
    ) -> str | None:
        """Persisted session id for ``(chat, thread, agent)`` or None.

        ``agent_name`` defaults to the route resolved by
        :meth:`TelegramConfig.agent_for` so callers that don't care
        about explicit overrides keep working unchanged. Returns
        ``None`` for the (chat, agent) tuple if it was marked stale
        by :meth:`verify_sessions_on_startup` — the next message
        opens a fresh session.
        """
        key = self._session_key(chat_id, thread_id)
        agent = agent_name or self.tg_cfg.agent_for(chat_id, thread_id)
        if self._stale_key(key, agent) in self._stale:
            return None
        per_agent = self._sessions.get(key) or {}
        return per_agent.get(agent)

    def _set_session_id(
        self,
        chat_id: int,
        session_id: str,
        thread_id: int | None = None,
        agent_name: str | None = None,
    ) -> None:
        key = self._session_key(chat_id, thread_id)
        agent = agent_name or self.tg_cfg.agent_for(chat_id, thread_id)
        per_agent = self._sessions.setdefault(key, {})
        per_agent[agent] = session_id
        self._stale.discard(self._stale_key(key, agent))
        _save_acp_session_map(self.tg_cfg, self._sessions)

    async def _reopen_session_after_loss(
        self,
        client: Any,
        chat_id: int,
        thread_id: int | None,
        sid: str | None,
    ) -> str:
        """Recover when the gateway no longer recognizes ``sid``.

        Two failure modes share this path:
          * the gateway's child idle-reaped the session between turns;
          * the gateway parent itself restarted (OOM, redeploy) so the
            in-memory ``child.sessions`` set is empty for everyone.

        Strategy: try ``session/load`` first — if persistence still
        has the turn history, the gateway re-attaches the id to a
        fresh child and we keep continuity. If load fails (no record
        on disk, schema mismatch, agent forgot), allocate a new id
        via ``session/new`` and rebind the chat. Either way the
        caller gets a session id that's live on the gateway right
        now.

        On success the chat's persisted session map is updated so
        the next turn doesn't repeat the recovery dance. The old
        ``sid`` (if reused) stays in place; a brand-new id replaces
        it.
        """
        agent_cfg = self._agent_cfg(chat_id, thread_id)
        cwd = str(agent_cfg.home)
        if sid:
            try:
                await client.session_load(sid, cwd=cwd)
                self._log.info(
                    "session reopen via session/load chat=%s sid=%s",
                    chat_id,
                    sid,
                )
                return sid
            except Exception as exc:  # noqa: BLE001
                self._log.warning(
                    "session/load after loss failed chat=%s sid=%s err=%s — falling back to session/new",
                    chat_id,
                    sid,
                    exc,
                )
        new_sid = await client.session_new(cwd=cwd)
        self._set_session_id(chat_id, new_sid, thread_id)
        await self._sync_session_topic(client, chat_id, thread_id, new_sid)
        self._log.info(
            "session reopen via session/new chat=%s old_sid=%s new_sid=%s",
            chat_id,
            sid,
            new_sid,
        )
        return new_sid

    def _drop_session_id(
        self,
        chat_id: int,
        thread_id: int | None = None,
        agent_name: str | None = None,
        all_agents: bool = False,
    ) -> None:
        """Drop the session binding for ``(chat, thread, agent)``.

        With ``all_agents=True`` clears every per-agent binding for
        that chat — matches the legacy "/new wipes the chat" UX.
        Otherwise only the routed (or named) agent's binding is
        cleared, so switching agents back later resumes the OTHER
        agent's session intact.
        """
        key = self._session_key(chat_id, thread_id)
        per_agent = self._sessions.get(key)
        changed = False
        if per_agent is not None:
            if all_agents:
                self._sessions.pop(key, None)
                changed = True
            else:
                agent = agent_name or self.tg_cfg.agent_for(chat_id, thread_id)
                if agent in per_agent:
                    per_agent.pop(agent, None)
                    if not per_agent:
                        self._sessions.pop(key, None)
                    changed = True
        if changed:
            _save_acp_session_map(self.tg_cfg, self._sessions)
        # Stale set: scoped to (chat, agent). Clear matching entries.
        if all_agents:
            self._stale = {s for s in self._stale if not s.startswith(f"{key}@")}
        else:
            agent = agent_name or self.tg_cfg.agent_for(chat_id, thread_id)
            self._stale.discard(self._stale_key(key, agent))
        # Drop the visibility cache too — a fresh /new should land on
        # default (off/off) just like the TUI does.
        self._visibility.pop(key, None)
        # Tear down the live render observer(s) bound to this chat — the
        # session they tail is gone; a fresh observer starts on the next
        # submit against the new session (FIX 1, unbind path). Also reset the
        # dedupe marks so the new session's turns aren't skipped.
        for k in list(self._render_log_tasks):
            if k.startswith(f"{key}::"):
                self._stop_render_key(k)
        self._rendered_turn_count.pop(key, None)
        self._local_turn_pending.pop(key, None)
        self._rendered_offset.pop(key, None)

    # ---- forum-topic name tracking -------------------------------------

    def _remember_topic_name(
        self,
        chat_id: int,
        thread_id: int | None,
        msg: dict,
    ) -> None:
        """Cache a forum topic's human name from an incoming update.

        Telegram surfaces the topic NAME only on a handful of message shapes:

        * ``forum_topic_created`` — the service message that opens a topic
          (its own ``message_thread_id`` == the topic's thread id, and when
          absent equals the service message's ``message_id``);
        * ``forum_topic_edited`` — a rename (carries ``name`` when the title
          changed);
        * ``reply_to_message.forum_topic_created`` — present on a reply whose
          thread root is the creation service message.

        We stash whatever name we can see keyed by ``(chat, thread)`` so the
        next session bound in that thread is labelled with the real topic
        rather than the ``topic-<id>`` fallback. Best-effort: any malformed
        shape is ignored.
        """
        if not isinstance(msg, dict):
            return
        topic_names = self.__dict__.setdefault("_topic_names", {})
        name: str | None = None
        tid: int | None = thread_id
        created = msg.get("forum_topic_created")
        edited = msg.get("forum_topic_edited")
        reply = msg.get("reply_to_message")
        if isinstance(created, dict) and created.get("name"):
            name = str(created["name"]).strip()
            # The creation service message's own thread id is its message_id
            # when ``message_thread_id`` is absent.
            if tid is None:
                mid = msg.get("message_id")
                tid = int(mid) if isinstance(mid, int) else None
        elif isinstance(edited, dict) and edited.get("name"):
            name = str(edited["name"]).strip()
        elif isinstance(reply, dict):
            rcreated = reply.get("forum_topic_created")
            if isinstance(rcreated, dict) and rcreated.get("name"):
                name = str(rcreated["name"]).strip()
        if name and tid is not None:
            topic_names.setdefault(str(chat_id), {})[int(tid)] = name

    def _topic_name(self, chat_id: int, thread_id: int | None) -> str:
        """Best-known topic name for ``(chat, thread)``.

        The General topic and every non-forum chat (``thread_id is None``) map
        to ``"general"`` — the universal default every frontend shares. A named
        forum topic resolves to its cached name, falling back to
        ``topic-<thread_id>`` when the name hasn't been seen yet (cold cache).
        """
        if thread_id is None:
            return "general"
        topic_names = self.__dict__.setdefault("_topic_names", {})
        name = topic_names.get(str(chat_id), {}).get(int(thread_id))
        return name or f"topic-{thread_id}"

    async def _sync_session_topic(
        self,
        client: Any,
        chat_id: int,
        thread_id: int | None,
        sid: str,
    ) -> None:
        """Stamp the Telegram forum topic onto a bound session.

        Pushes the best-known topic name through the ACP ``topic`` config
        option (same channel ``/reasoning`` uses), which the engine persists to
        the session header — so the WebUI Topics page groups this conversation
        under the same topic the operator sees in Telegram.

        Idempotent + rename-aware: the topic last applied to ``sid`` is cached,
        so a no-op dispatch skips the round-trip, but a ``forum_topic_edited``
        rename (which updates ``_topic_names`` → a new resolved name) re-pushes
        on the thread's next message, keeping an ALREADY-bound session in sync.
        Best-effort: a transport hiccup never breaks the turn (the session
        keeps its prior topic and we retry on the next dispatch).
        """
        if not sid:
            return
        applied = self.__dict__.setdefault("_applied_topic", {})
        topic = self._topic_name(chat_id, thread_id)
        if applied.get(sid) == topic:
            return
        try:
            await client.session_set_config_option(sid, "topic", topic)
            applied[sid] = topic
        except Exception:  # noqa: BLE001
            self._log.debug(
                "topic sync failed chat=%s thread=%s sid=%s",
                chat_id,
                thread_id,
                sid,
                exc_info=True,
            )

    # ---- per-(chat,thread) display visibility --------------------------

    def _get_visibility(
        self,
        chat_id: int,
        thread_id: int | None = None,
    ) -> dict[str, Any]:
        """Return the ``{reasoning_visibility, tool_output}`` map for
        ``(chat, thread)``. Auto-initializes to defaults (reasoning
        ``"on"``, tool_output ``"on"``) on first access so callers
        can read without checking presence.

        ``tool_output`` is a 3-state string: ``"off"`` / ``"on"`` /
        ``"stream"`` — see ``normalize_tool_output`` for semantics.
        Migrates legacy bool ``tool_output_visible`` records on read.
        """
        key = self._session_key(chat_id, thread_id)
        cur = self._visibility.setdefault(
            key,
            {
                "reasoning_visibility": "on",
                "tool_output": "on",
            },
        )
        # Backwards-compat: an old session record may still carry the
        # legacy bool ``tool_output_visible``. Translate True → "on",
        # False → "off" so the new gate-by-string code path doesn't
        # crash on stale state.
        if "tool_output" not in cur and "tool_output_visible" in cur:
            cur["tool_output"] = "on" if cur.get("tool_output_visible") else "off"
        cur.setdefault("tool_output", "on")
        cur.setdefault("reasoning_visibility", "on")
        return cur

    def _set_reasoning_visibility(
        self,
        chat_id: int,
        thread_id: int | None,
        level: str,
    ) -> None:
        """Set the per-(chat,thread) reasoning visibility. Caller is
        expected to validate ``level`` against
        ``normalize_reasoning_level`` first.
        """
        self._get_visibility(chat_id, thread_id)["reasoning_visibility"] = level
        _save_visibility_map(self.tg_cfg, self._visibility)

    def _set_tool_output_level(
        self,
        chat_id: int,
        thread_id: int | None,
        level: str,
    ) -> None:
        """Set the per-(chat,thread) tool-output visibility level.

        Caller validates ``level`` against ``normalize_tool_output``
        first; the engine accepts ``"off"`` / ``"on"`` / ``"stream"``.
        """
        self._get_visibility(chat_id, thread_id)["tool_output"] = level
        _save_visibility_map(self.tg_cfg, self._visibility)

    async def verify_sessions_on_startup(self) -> None:
        """For each persisted ``(chat, agent)`` pair, call
        ``session/load`` to verify.

        With per-(chat, agent) routing the same chat may carry one
        binding per agent it has used; we only verify the binding
        for the agent the chat is CURRENTLY routed to. Other-agent
        bindings are left untouched on disk — switching back later
        will pick them up and verify lazily.
        """
        if not self._sessions:
            return
        for key in list(self._sessions.keys()):
            # ``key`` is the chat_key — bare chat_id or
            # ``"<chat_id>:<thread_id>"`` for forum topics. Parse the
            # thread_id back out so the per-chat agent resolver picks
            # the right route (a thread-pinned agent must be honored
            # at startup verification too).
            chat_id, thread_id = _parse_chat_key(key)
            if chat_id is None:
                # Malformed key — mark every per-agent binding stale.
                for agent_name in self._sessions.get(key) or {}:
                    self._stale.add(self._stale_key(key, agent_name))
                continue
            agent_cfg = self._agent_cfg(chat_id, thread_id)
            sid = (self._sessions.get(key) or {}).get(agent_cfg.name)
            if not sid:
                # No binding for the routed agent — nothing to
                # verify. Other agents' bindings will be lazily
                # verified the first time the chat switches to
                # them.
                continue
            try:
                client = await _retry_startup_acp_open(
                    lambda: self._get_or_open_client(chat_id, thread_id),
                )
            except Exception:  # noqa: BLE001
                self._log.exception(
                    "verify_sessions_on_startup: open client failed for chat=%s agent=%s; marking stale",
                    key,
                    agent_cfg.name,
                )
                self._stale.add(self._stale_key(key, agent_cfg.name))
                continue
            try:
                await client.session_load(sid, cwd=str(agent_cfg.home))
            except Exception as e:  # noqa: BLE001
                self._log.warning(
                    "verify_sessions_on_startup: session_load failed chat=%s agent=%s sid=%s err=%s — marking stale",
                    key,
                    agent_cfg.name,
                    sid,
                    e,
                )
                self._stale.add(self._stale_key(key, agent_cfg.name))

    # ---- client / capability wiring ------------------------------------

    async def _default_client_factory(
        self,
        chat_id: int,
        capabilities,
        thread_id: int | None = None,
    ):
        """Open one ``AcpClient`` for ``(chat_id, thread_id)``.

        Production path: the north gateway is reachable over Streamable
        HTTP at ``self.acp_url``. We open one HTTP connection per
        ``(chat, thread)`` pair against the agent the chat is
        routed to. The gateway holds each agent in-memory via its
        :class:`AgentRegistry`; we never spawn a per-agent child.

        ``acp_url`` accepts ``http(s)://host:port`` (gateway) or
        ``ws(s)://host:port/acp`` (legacy WS endpoint, preserved for
        tests that fake an ACP server). Empty / unset falls through to
        a stdio spawn — kept solely as a test-time fallback so
        unit tests that don't run the gateway can still drive the bot
        end-to-end.
        """
        agent_cfg = self._agent_cfg(chat_id, thread_id)

        if self.acp_url:
            scheme = self.acp_url.split("://", 1)[0].lower()
            if scheme in ("http", "https"):
                from sharedagents_acp.client.http_spawn import connect_http

                # Gateway routes by mount path; per-agent endpoint is
                # ``<gateway-url>/acp/<agent>``. ``connect_http`` then
                # appends ``/acp`` for the actual JSON-RPC route, so
                # the wire URL ends up at ``…/acp/<agent>/acp`` —
                # matches gateway.py's ``Mount("/acp/<agent>", …)``
                # over a sub-app whose own routes live under ``/acp``.
                agent_url = f"{self.acp_url.rstrip('/')}/acp/{agent_cfg.name}"
                client = await connect_http(
                    agent_url,
                    client_capabilities=capabilities,
                    bearer_token=self.acp_bearer_token or None,
                    frontend="telegram",  # Pin identity at init
                )
                # ``connect_http`` already runs initialize() during
                # connect (it has to — the connection id is in the
                # initialize response headers).
                return client
            if scheme in ("ws", "wss"):
                from sharedagents_acp.client.ws_spawn import connect_ws

                client, ws_session = await connect_ws(
                    self.acp_url,
                    client_capabilities=capabilities,
                    frontend="telegram",  # Pin identity at init
                )
                await client.initialize()
                client._telegram_ws_session = ws_session
                return client
            raise ValueError(f"unsupported acp_url scheme {scheme!r} — expected http/https (gateway) or ws/wss")

        # Test-only fallback: spawn a stdio child of the bot. Production
        # always sets ``acp_url`` so the gateway owns the agent process.
        from sharedagents_acp.client.spawn import spawn_stdio_agent

        cmd = [
            sys.executable,
            "-m",
            "sharedagents_gateway",
            "acp-stdio",
            agent_cfg.name,
        ]
        client, proc = await spawn_stdio_agent(
            cmd,
            cwd=str(agent_cfg.home),
            client_capabilities=capabilities,
        )
        await client.initialize()
        self._child_procs[TelegramConfig.chat_key(chat_id, thread_id)] = proc
        return client

    async def _get_or_open_client(
        self,
        chat_id: int,
        thread_id: int | None = None,
    ):
        key = TelegramConfig.chat_key(chat_id, thread_id)
        client = self._clients.get(key)
        if client is not None:
            # ``AcpClient._closed`` flips True in two cases:
            #   * explicit ``close()`` (we did it) — caller is reopening,
            #   * read-loop exited on transport error (gateway crashed,
            #     SSE stream torn down, OOM-kill etc.) — the cached
            #     instance can't service requests anymore.
            # Either way, evict + recreate. Without this the next
            # ``session_prompt`` would either hang on a future the
            # dead read loop will never resolve, or fail with
            # ``AcpTransportClosed`` on every retry.
            if not getattr(client, "_closed", False):
                return client
            self._log.warning(
                "evicting closed ACP client for chat_key=%s; reopening",
                key,
            )
            self._clients.pop(key, None)
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
        capabilities = _capabilities_for_chat(
            chat_id,
            trusted_chats=self.trusted_chats,
        )
        # Default factory accepts ``thread_id`` so per-thread agent
        # routing can resolve the right stdio target. Test stubs that
        # override ``client_factory`` without the kwarg keep working
        # because we pass it positionally only when supported.
        try:
            client = await self._client_factory(
                chat_id,
                capabilities,
                thread_id=thread_id,
            )
        except TypeError:
            # Legacy two-arg stub — drop thread_id so test fixtures
            # don't have to be updated in lockstep.
            client = await self._client_factory(chat_id, capabilities)
        # Wire the permission-prompt UI on every freshly built client,
        # regardless of which factory produced it. Destructive tool
        # calls round-trip ``session/request_permission`` through an
        # inline-keyboard message in THIS chat (the handler closes over
        # chat_id / thread_id). Best-effort — a client stub without the
        # method just doesn't get the hook, and the agent's no-channel
        # path then safely defaults to deny.
        register = getattr(client, "register_request_permission", None)
        if callable(register):
            try:
                register(self._make_permission_handler(chat_id, thread_id))
            except Exception:  # noqa: BLE001
                self._log.debug(
                    "register_request_permission failed for chat_key=%s",
                    key,
                    exc_info=True,
                )
        self._clients[key] = client
        return client

    def _http_gateway_base(self) -> str:
        """The HTTP base for the control submit/subscribe/cancel routes.

        Returns the gateway URL when ``acp_url`` is an ``http(s)://`` base
        (the production deployment shape) so Telegram can POST the submit
        route + tail the canonical event stream there. Returns ``""`` for the
        ws/stdio test-fallback transports, which have no control endpoint —
        the caller then drives the turn through the legacy ACP iterator (the
        renderers are shared either way, so behaviour is identical).
        """
        url = (getattr(self, "acp_url", "") or "").rstrip("/")
        if url.split("://", 1)[0].lower() in ("http", "https"):
            return url
        return ""


# ---------------------------------------------------------------------------
# Shared helpers/constants owned by ``bot.py`` (imported at BOTTOM to keep the
# bot<->mixin import cycle deadlock-free — see module docstring).
# ---------------------------------------------------------------------------
from sharedagents_telegram.bot import (  # noqa: E402
    INFLIGHT_RECOVERY_POLL_S,
    INFLIGHT_RECOVERY_WAIT_S,
    AgentConfig,
    TelegramConfig,
    _capabilities_for_chat,
    _parse_chat_key,
    _retired_main_should_heal,
    _save_acp_session_map,
    _save_inflight_turns,
    _save_visibility_map,
)
