/* render.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
async function deleteSession(sid) {
  try {
    // Session deletion is irreversible, so the server
    // re-challenges any DELETE that arrives without an explicit confirmation.
    // The user already confirmed via the dialog above, so pass confirm=1; a
    // direct API call that skips the dialog hits the 409 re-challenge.
    await apiDelete(`/api/sessions/${sid}?confirm=1`);
    delete state.sessions[sid];
    if (state.currentSessionId === sid) {
      ++state.navEpoch; // clearing the active session supersedes its in-flight loaders
      state.currentSessionId = null;
      clearLastSession(); // don't restore a deleted conversation on next load
      closeLiveStream();
      clearMessages();
      refreshSessionWidgets(); // no active session → hide the footer status bar
    }
    renderSessionList();
  } catch (e) {
    console.error("Failed to delete session:", e);
  }
}

// ── Messages ──

function clearMessages() {
  state.pendingPrompt = null;
  state.pendingCancel = null;
  state.pendingEcho = null; // switching/clearing — drop any unmatched echo
  state.liveTurn = null; // transcript identity; SSE close must not drop this
  state.livePhysicalTurnId = "";
  state.liveEngine = "";
  state.closedPhysicalTurnIds = {};
  state.locallyAbortedTurnIds = {};
  state.liveMemoryFlushEl = null;
  if (typeof _stopStreaming === "function") {
    _stopStreaming();
  }
  state.promptSubmitting = false;
  state._askStopReleased = false;
  state._instantAbort = false;
  state.askUserCards = {}; // reset ask-user cards on session/clear
  state.permissionCards = {}; // ditto for inline tool-permission cards
  state.realtimeArmedThisSession = false;
  state.realtimeCaptureMode = "wake";
  clearPastedImages(); // discard clipboard-pasted image previews
  VirtualScroller.reset();
  // Keep the welcome screen placeholder.
  if (welcomeScreen.parentNode !== messagesContainer) {
    messagesContainer.innerHTML = "";
    messagesContainer.appendChild(welcomeScreen);
  } else {
    // Re-insert welcome at the end (should be the only child).
    welcomeScreen.remove();
    messagesContainer.appendChild(welcomeScreen);
  }
  // Ensure sentinel and bottom marker are present after reset.
  VirtualScroller.init();
  setWelcomeVisible(true);
}

/** Toggle the welcome/empty-state screen. Also refreshes the realtime (Jarvis)
 *  button so its enabled/title state matches the current session. */
function setWelcomeVisible(visible) {
  if (!welcomeScreen) return;
  welcomeScreen.style.display = visible ? "" : "none";
  updateRealtimeVisibility();
}

// Initial history and older network pages must cache the same transcript
// rows BEFORE materialising DOM. Otherwise a question found while replaying
// an older assistant is appended to the live tail instead of its ask point.
function _splitAskHistoryMessage(msg) {
  if (msg.role !== "assistant" || !Array.isArray(msg.segments)
      || !msg.segments.length || typeof _splitAskSegments !== "function") {
    return null;
  }
  const parts = _splitAskSegments(msg.segments);
  if (!parts.some((part) => part.kind === "ask")) return null;
  const rows = [];
  const meta = { agent: msg.agent, timestamp: msg.timestamp };
  // An unanswered question settles with its turn, exactly like the live
  // turn_ended does. Left open, a reload re-mounts it as a pending ask and
  // holds the composer on Stop for a session that is idle.
  const closedAs = (msg.aborted || msg.status === "aborted")
    ? "⏹ Question aborted"
    : msg.status === "error" ? "✗ Turn failed" : "";
  const prose = (segments) => segments
    .filter((seg) => seg && (seg.kind === "assistant" || seg.kind === "answer"))
    .map((seg) => String(seg.content || seg.text || ""))
    .join("");
  const allProse = prose(msg.segments);
  const committed = typeof msg.content === "string" ? msg.content : "";
  // Only a genuinely missing trailing answer can use the committed text as a
  // fallback. A completed ask with prose solely BEFORE it often has that same
  // prose in msg.content; echoing it below the card invents a continuation.
  const trailing = parts[parts.length - 1];
  let fallback = "";
  if (msg.status !== "running"
      && (!trailing || trailing.kind === "ask" || !prose(trailing.segments).trim())) {
    if (committed.trim() && committed.trim() !== allProse.trim()) {
      fallback = committed.startsWith(allProse)
        ? committed.slice(allProse.length) : committed;
    }
  }
  parts.forEach((part, index) => {
    if (part.kind === "ask") {
      const ask = part.segment;
      const args = typeof _toolArgsObj === "function"
        ? _toolArgsObj(ask.arguments != null ? ask.arguments : ask.raw_arguments)
        : (ask.arguments || {});
      const answered = ask.ok != null || !!ask.content;
      const closed = !answered && !!closedAs;
      rows.push({
        ...meta, role: "ask_user", content: "",
        askCall: {
          callId: String(ask.call_id || ""),
          args,
          content: ask.content || "",
          ok: closed ? false : ask.ok,
          finalized: answered || closed,
          ...(closed ? { statusOverride: closedAs } : {}),
        },
      });
      return;
    }
    const last = index === parts.length - 1;
    rows.push(last
      ? { ...msg, content: prose(part.segments) || fallback, segments: part.segments }
      : { ...meta, role: "assistant", content: "", segments: part.segments });
  });
  if (trailing && trailing.kind === "ask" && fallback.trim()) {
    rows.push({ ...msg, content: fallback, segments: [] });
  }
  return rows;
}

function renderMessages(messages) {
  clearMessages(); // this calls VirtualScroller.reset() + init()
  if (messages.length === 0) return;

  setWelcomeVisible(false);

  // Cache EVERY message (lightweight — no DOM), then render only the latest
  // window. Older messages stay in the cache and are prepended on scroll-up
  // (VirtualScroller.loadOlder). This keeps the browser light on long logs
  // without the buggy scroll-position→index pruning that blanked the view.
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    // User content may be a content-block list carrying images — split it the
    // same way the live turn_started path does, so a rehydrated history bubble
    // shows its picture (and never stringifies an array into the text).
    // ``msg.agent`` is the lane that authored the turn (survives agent switch).
    if (msg.role === "user") {
      const parts = _userPromptParts(msg.content, msg.attached_files);
      const text = parts.text;
      const images = parts.files.length ? [...parts.images, ...parts.files] : parts.images;
      // Omit realtime TTS "read out the result" relays — they dump the child
      // transcript as a fake operator prompt on reload (canonical delivery is
      // the ``<task_result>`` card).
      if (typeof _shouldOmitUserHistoryPrompt === "function"
          && _shouldOmitUserHistoryPrompt(text)) {
        continue;
      }
      // Pair a MEMORY FLUSH prompt with its assistant reply into one collapsed
      // widget so the silent end-of-session turn doesn't spam the transcript.
      if (typeof _isMemoryFlushPrompt === "function" && _isMemoryFlushPrompt(text)) {
        let reply = "";
        const next = messages[i + 1];
        if (next && next.role === "assistant") {
          reply = typeof next.content === "string" ? next.content : "";
          i += 1;
        }
        VirtualScroller.add(
          "memory_flush",
          { prompt: text, reply },
          null, null, null, msg.agent, null, msg.timestamp,
        );
        continue;
      }
      VirtualScroller.add("user", text, null, images, null, msg.agent, null, msg.timestamp, undefined, undefined, undefined, undefined, authorFromMessage(msg));
    } else {
      const askRows = _splitAskHistoryMessage(msg);
      if (askRows) {
        askRows.forEach((row) => {
          const idx = VirtualScroller.add(
            row.role, row.content, null, null, row.segments, row.agent,
            row.artifacts, row.timestamp, row.aborted, row.status, row.error,
            row.ended_at,
          );
          if (row.askCall && idx >= 0 && state.msgCache[idx]) {
            state.msgCache[idx].askCall = row.askCall;
          }
        });
      } else if (
        msg.role === "assistant"
        && typeof _splitNudgeSegments === "function"
        && Array.isArray(msg.segments)
        && msg.segments.length
      ) {
        const nudgeParts = _splitNudgeSegments(msg.segments);
        if (nudgeParts.length > 1) {
          nudgeParts.forEach((p) => {
            if (p.type === "nudge") {
              VirtualScroller.add(
                "nudge",
                (p.notice && p.notice.raw) || "nudge",
                null, null, null, msg.agent, null, msg.timestamp,
              );
            } else {
              VirtualScroller.add(
                "assistant", msg.content, null, null, p.segments, msg.agent,
                msg.artifacts, msg.timestamp, msg.aborted, msg.status,
                msg.error, msg.ended_at,
              );
            }
          });
        } else {
          VirtualScroller.add(
            msg.role, msg.content, null, null, msg.segments, msg.agent,
            msg.artifacts, msg.timestamp, msg.aborted, msg.status, msg.error,
            msg.ended_at,
          );
        }
      } else if (
        msg.role === "assistant"
        && typeof splitAssistantOnNudgeNotices === "function"
      ) {
        const rawParts = splitAssistantOnNudgeNotices(msg.content);
        if (rawParts.length > 1) {
          rawParts.forEach((p) => {
            if (p.notice) {
              VirtualScroller.add(
                "nudge",
                (p.notice && p.notice.raw) || "nudge",
                null, null, null, msg.agent, null, msg.timestamp,
              );
            } else {
              VirtualScroller.add(
                "assistant", p.text, null, null, null, msg.agent,
                msg.artifacts, msg.timestamp, msg.aborted, msg.status,
                msg.error, msg.ended_at,
              );
            }
          });
        } else {
          VirtualScroller.add(
            msg.role, msg.content, null, null, msg.segments, msg.agent,
            msg.artifacts, msg.timestamp, msg.aborted, msg.status, msg.error,
            msg.ended_at,
          );
        }
      } else {
        VirtualScroller.add(
          msg.role, msg.content, null, null, msg.segments, msg.agent,
          msg.artifacts, msg.timestamp, msg.aborted, msg.status, msg.error,
          msg.ended_at,
        );
      }
    }
  }
  VirtualScroller.renderTail();
  scrollToBottom();
}

function appendMessage(
  role, content, scroll = true, images = null, agent = null, artifacts = null,
  timestamp = null, author = null, options = {},
) {
  setWelcomeVisible(false);

  // A user bubble mid-turn (optimistic send, queue inject / native steer)
  // must split the live assistant the same way ask-user cards do. Otherwise
  // later Activity rows keep painting into the previous bubble and the
  // steer appears under the entire turn.
  if (role === "user" && !options.preserveLiveTurn && typeof _sealLiveAssistant === "function") {
    _sealLiveAssistant();
  }

  if (
    role === "assistant"
    && typeof splitAssistantOnNudgeNotices === "function"
  ) {
    const parts = splitAssistantOnNudgeNotices(content);
    if (parts.length > 1) {
      let last = null;
      for (let i = 0; i < parts.length; i++) {
        const p = parts[i];
        if (p.notice) {
          last = appendNudgeYouCard(p.notice, timestamp);
        } else {
          last = appendMessage(
            "assistant", p.text, scroll, images, agent, artifacts, timestamp,
          );
        }
      }
      return last;
    }
  }

  // Fold a flush assistant reply into the preceding MEMORY FLUSH card when
  // the live/history path emits them as two messages.
  if (
    role === "assistant"
    && typeof _isMemoryFlushReply === "function"
    && _isMemoryFlushReply(content)
  ) {
    const last = state.msgCache[state.msgCache.length - 1];
    if (
      last
      && last.el
      && last.el.classList
      && last.el.classList.contains("message-memory-flush")
    ) {
      _fillMemoryFlushReply(last.el, content);
      if (scroll) scrollToBottom();
      return last.el;
    }
  }

  if (role === "user" && state.pendingEcho && state.pendingEcho.el) {
    const last = state.msgCache[state.msgCache.length - 1];
    if (last && last.el === state.pendingEcho.el && last.role === "user") {
      const sameText = String(last.content || "").replace(/\s+/g, " ").trim()
        === String(content || "").replace(/\s+/g, " ").trim();
      if (sameText) {
        if (author && Array.isArray(author.actors) && author.actors.length) {
          _applyMessageAuthor(last.el, author);
        }
        return last.el;
      }
    }
  }

  const div = _createMessageElement(role, content, images, agent, artifacts, timestamp, undefined, undefined, undefined, undefined, author);
  if (!div) return div;

  if (role === "user") {
    state.scroll.autoScroll = true;
    if (typeof VirtualScroller !== "undefined") {
      VirtualScroller._hideNewMsgBtn();
    }
  }

  // Cache the data and register this message index.
  const idx = VirtualScroller.add(
    role, content, div, images, null, agent, artifacts, timestamp,
    undefined, undefined, undefined, undefined, author,
  );

  if (role === "assistant") {
    div.dataset.raw = content || "";
    div.classList.toggle("has-content", !!(content && content.trim()));
  }

  // Attach through the virtual scroller (which manages auto-scroll).
  // An AskUserQuestion card is an ordinary transcript row: it sits where the
  // ask happened, and everything the turn produces later — the post-answer
  // continuation, a mid-turn steer — appends BELOW it, never above. Never
  // relocate a later row above a card to keep the card last (see ask_user.js).
  VirtualScroller.attach(div, idx);

  if (scroll) {
    if (role === "user" && typeof VirtualScroller !== "undefined") {
      VirtualScroller._forcePin();
      requestAnimationFrame(() => VirtualScroller._forcePin());
    } else {
      scrollToBottom();
    }
  }
  return div;
}

// Set an assistant message's content from RAW gateway text (which may carry a
// leading <think>…</think> reasoning/tool block). Raw is stashed on the element
// so streaming deltas accumulate deterministically instead of being re-derived
// from rendered DOM text.
function setAssistantContent(el, raw) {
  el.dataset.raw = raw || "";
  const contentEl = el.querySelector(".message-content");
  if (contentEl) contentEl.innerHTML = renderAssistantContent(raw);
  el.classList.toggle("has-content", !!(raw && raw.trim()));

  // Sync with virtual cache (find this el's index).
  const idx = state.msgCache.findIndex((m) => m.el === el);
  if (idx >= 0) {
    VirtualScroller.update(idx, raw);
  }
}

function updateMessageContent(el, content) {
  setAssistantContent(el, content);
}

function getAgentInitial() {
  const name = state.currentAgent || "jarvis";
  return name.charAt(0).toUpperCase();
}

// The engine-model badge label prefers the operator-facing
// alias (e.g. "jarvis") and falls back to the raw wire id (e.g.
// "qwen3.6-27b-fp8") when the config maps none; the tooltip shows both when
// they differ. Pure — the vector tests extract and exercise this directly.
function engineModelBadgeLabel(modelState) {
  if (!modelState.engineModel) return { label: "", title: "", hidden: true };
  const alias = modelState.engineModelAlias || "";
  return {
    label: alias || modelState.engineModel,
    title: alias && alias !== modelState.engineModel
      ? `Model: ${alias} (${modelState.engineModel})`
      : `Model: ${modelState.engineModel}`,
    hidden: false,
  };
}

function updateHeader() {
  currentAgentEl.textContent = state.currentAgent;
  const effectiveModeEl = document.getElementById("effective-mode");
  if (effectiveModeEl) {
    const mode = state.effectiveMode || {};
    effectiveModeEl.textContent = mode.label || mode.id || "Unknown mode";
    effectiveModeEl.title = [
      mode.label || mode.id || "Unknown mode",
      mode.behavior_kind ? `behavior: ${mode.behavior_kind}` : "",
    ].filter(Boolean).join(" · ");
  }
  // Conversation title — primary heading. Hide default stubs
  // (untitled-… / New Chat - HHMMSS) so the heading stays "New Chat"
  // until a real title arrives. The sidebar still lists the timestamp
  // stub so empty chats stay distinguishable.
  const titleEl = document.getElementById("session-title");
  if (titleEl) {
    const sess = state.sessions && state.sessions[state.currentSessionId];
    const raw = sess && sess.title;
    const title =
      (raw && !isDefaultSessionTitle(raw)) ? raw : "New Chat";
    titleEl.textContent = title;
    titleEl.title = title;
  }
  // Engine-model badge: the real model the agent last ran (claude-opus-4-…
  // / gpt-5.5-codex / composer-2.5). Empty until the first turn reports it.
  // Prefer the operator-facing alias, fall back to the raw
  // id; tooltip shows both when they differ (engineModelBadgeLabel is pure
  // and exercised by the tests/web vectors).
  const badge = document.getElementById("engine-model");
  if (badge) {
    const rendered = engineModelBadgeLabel(state);
    badge.textContent = rendered.label;
    badge.style.display = rendered.hidden ? "none" : "";
    badge.title = rendered.title;
  }
  // Keep the model picker selection honest: re-resolve the active alias now that
  // the engine model may have changed (e.g. a turn just reported the real model
  // for a freshly created session). Cheap, and stops the picker drifting.
  if (modelSelect && state.models && state.models.length) {
    const want = pickerModelAlias();
    if (want && modelSelect.value !== want) modelSelect.value = want;
  }
  updateTeacher();
  syncOptionsControls();
  // Share control reflects the open conversation.
  if (window.ShareControl) window.ShareControl.sync();
}

// Reflect the live mode into the Options-card select, and the gear
// expand toggles from state.visExpand (not gateway visibility).
function syncOptionsControls() {
  const info = state.statusInfo || {};
  const optMode = document.getElementById("opt-mode");
  const optActivity = document.getElementById("opt-activity");
  const optReasoning = document.getElementById("opt-reasoning");
  const optTools = document.getElementById("opt-tools");
  const optCollapse = document.getElementById("opt-collapse");
  if (optMode) {
    const agent = state.pickerAgent || state.currentAgent;
    renderAgentModeSelect(agent);
    const allowed = pickerModes(agent).map((mode) => mode.id);
    const capDefault = typeof preferredModeFor === "function"
      ? preferredModeFor(agent, "")
      : "";
    const liveId = (state.effectiveMode && state.effectiveMode.id) || "";
    const session = state.sessions && state.sessions[state.currentSessionId];
    const pending = (session && session.pending_mode_intent) || "";
    const fresh = typeof isFreshUnboundSession === "function"
      && isFreshUnboundSession(state.currentSessionId);
    // Fresh New Chat: honor an explicit operator pick, else the agent's
    // capability default (Auto for native agents). Do not keep a leftover
    // Ask/Plan chip over that default.
    let wanted;
    if (fresh && !state.modePickerTouched) {
      wanted = pending || capDefault || liveId || "auto";
    } else {
      wanted = liveId || pending || info.mode || state.currentMode || capDefault || "auto";
    }
    const fallback = allowed.includes(capDefault) ? capDefault : (allowed[0] || "auto");
    optMode.value = allowed.includes(wanted) ? wanted : fallback;
  }
  if (optActivity) syncVisExpandToggle(optActivity, !!(state.visExpand && state.visExpand.activity));
  if (optReasoning) syncVisExpandToggle(optReasoning, !!(state.visExpand && state.visExpand.reasoning));
  if (optTools) syncVisExpandToggle(optTools, !!(state.visExpand && state.visExpand.tools));
  if (typeof syncTransportToggle === "function") syncTransportToggle();
  if (optCollapse) {
    optCollapse.setAttribute("aria-pressed", String(!!state.collapseReplies));
    optCollapse.classList.toggle("is-active", !!state.collapseReplies);
  }
  updateTeacher();
}

function syncVisExpandToggle(btn, on) {
  if (!btn) return;
  btn.setAttribute("aria-pressed", on ? "true" : "false");
  btn.classList.toggle("is-active", !!on);
}

// Teacher-supervision indicator + control. The badge in the header-info row
// mirrors the TUI footer widget — visible only while a teacher model
// supervises this session, hidden otherwise. The toggle/selector in
// header-actions runs /teacher on|off and /teacher model <alias>.
function updateTeacher() {
  if (teacherBadge) {
    if (state.teacherActive) {
      teacherBadge.textContent = state.teacherModel
        ? `🧑‍🏫 teacher: ${state.teacherModel}`
        : "🧑‍🏫 teacher";
      teacherBadge.style.display = "";
      teacherBadge.title = state.teacherModel
        ? `Teacher supervising with ${state.teacherModel}`
        : "Teacher-model supervision active";
    } else {
      teacherBadge.style.display = "none";
    }
  }
  if (teacherToggle) {
    teacherToggle.setAttribute(
      "aria-pressed",
      state.teacherActive ? "true" : "false",
    );
    teacherToggle.classList.toggle("teacher-on", !!state.teacherActive);
    teacherToggle.classList.toggle("is-active", !!state.teacherActive);
  }
  if (teacherToggleState) {
    teacherToggleState.textContent = state.teacherActive ? "on" : "off";
  }
  renderTeacherModelSelect();
  renderTeacherWidget();
}

// ── Session status bar (mirrors the TUI footer) ─────────────────────────
// Exceptional widgets: todo/tasks at the top of the composer card;
// teacher supervision and compact progress under the toolbar. The stable
// toolbar itself owns agent/model/context; the Agent Monitor summary owns budget.

function renderSessionStatusBar() {
  if (!sessionStatus) return;
  sessionStatus.hidden = !state.currentSessionId;
  renderStatusSegments();
  renderTeacherWidget();
  renderHarnessWidget();
}

// Composer-toolbar segments: agent/state · context.
// Values only — no labels. Larger font.
function renderStatusSegments() {
  if (!statusSegments) return;

  // The chip is an EFFECTIVE-state display. An open draft must not change it.
  const sheetEffective = state.modelSheet && state.modelSheet.effective;
  const agentName = (sheetEffective && sheetEffective.agent) ||
    state.currentAgent || "jarvis";

  // State icon only — the idle/live ticker lives on #status-text.
  let stateIcon = "⏹";
  if (state.isStreaming || _turn) {
    const ph = _PHASE[_turn?.phase] || _PHASE.warming;
    stateIcon = ph.icon;
  } else if (state.treeLive) {
    stateIcon = "⏸";
  }

  // Context: used / max
  let ctxStr = "";
  let maxWindow = state.lastContextMax;
  if (!maxWindow || maxWindow <= 0) {
    const curModel = state.currentModel || (state.sessions[state.currentSessionId] && state.sessions[state.currentSessionId].model);
    const modelObj = (state.models || []).find((m) => m && (m.id === curModel || m.alias === curModel));
    if (modelObj && modelObj.contextWindow) {
      maxWindow = modelObj.contextWindow;
    } else if (agentName === "antigravity" || agentName === "cursor" || agentName === "kimi" || agentName === "hermes") {
      maxWindow = 1000000;
    } else if (agentName === "claude" || agentName === "codex" || agentName === "opencode") {
      maxWindow = 200000;
    }
  }
  if (maxWindow > 0) {
    const ti = (n, maxN) => {
      const v = Number(n) || 0;
      const cap = Number(maxN) || 0;
      if (v >= 1000000) return (v / 1000000).toFixed(1) + "M";
      if (v >= 10000) return (v / 1000).toFixed(0) + "K";
      if (v >= 1000) return (v / 1000).toFixed(1) + "K";
      // Keep units with the window so 580 next to 200K cannot be read as 580k.
      if (cap >= 1000) return (v / 1000).toFixed(1) + "K";
      return String(v);
    };
    ctxStr = ti(state.lastContextUsed || 0, maxWindow) + "/" + ti(maxWindow, maxWindow);
  }

  const segs = [];
  // The chip that DISPLAYS agent+model is also where they are
  // selected. A button (not a span) so it is keyboard-reachable; the popover it
  // toggles lives outside this node because innerHTML below replaces everything
  // here on every status update. The active model is shown inline so it is
  // readable without opening anything — it previously required opening the
  // session-options card.
  const modelLabel = typeof pickerOrSessionModelLabel === "function"
    ? pickerOrSessionModelLabel(agentName)
    : String(pickerModelAlias(agentName) || "").trim();
  segs.push(
    `<button type="button" class="seg seg-agent seg-agent-btn" id="agent-model-trigger"` +
    ` aria-haspopup="true" aria-expanded="false"` +
    ` title="Change agent or model">${stateIcon}` +
    (typeof pickerAgentLogo === "function"
      ? pickerAgentLogo(agentName, "sm", { eager: true })
      : "") +
    ` ${escapeHtml(agentName)}` +
    (modelLabel
      ? `<span class="seg-model">${escapeHtml(modelLabel)}</span>`
      // Never omit the chip outright: a cold load (no catalogue/boot yet) must
      // still show the slot with a neutral placeholder until the real label
      // lands via the boot payload or an `event: models` frame.
      : `<span class="seg-model seg-model-pending" title="Model loading…">…</span>`) +
    `<span class="seg-caret">▾</span></button>`
  );
  if (ctxStr) {
    // The context meter opens the Context sheet (context_viewer.js), which
    // lists what the model works with and carries the Compact action. This
    // node is re-created on EVERY status render, so its click handler is
    // delegated from the stable #status-segments container below (same
    // mechanism as #agent-model-trigger); never bind to the node itself.
    segs.push(
      `<button type="button" class="seg seg-ctx" id="compact-context-btn"` +
      ` title="Show context" aria-label="Show context" aria-haspopup="dialog"` +
      ` aria-controls="ctx-sheet">${ctxStr}</button>`
    );
  }
  statusSegments.innerHTML = segs.join('<span class="seg-sep">·</span>');
  renderAgentMonitorBudget();
  // Keep the Agents footer chip on the same selected model as this trigger.
  if (typeof updateAgentMonAccordion === "function") updateAgentMonAccordion();
  // DELEGATION, not per-render binding. renderStatusSegments replaces
  // statusSegments.innerHTML on EVERY status update, which destroys the trigger
  // node. A handler bound directly to that node dies with it — and worse, a tap
  // whose pointerdown landed on the old node never produces a click on its
  // replacement. On an idle session the chip is stable and it appears to work
  // (why this survived testing); on a live session the chip re-renders
  // constantly and the picker simply never opens. Binding once to the stable
  // container makes the handler independent of how often the chip re-renders.
  const _pop = document.getElementById("agent-model-pop");
  const _back = document.getElementById("amp-backdrop");
  if (_pop && statusSegments && !statusSegments.dataset.ampBound) {
    statusSegments.dataset.ampBound = "1";
    const close = function () {
      if (typeof closeModelSheet === "function") closeModelSheet();
    };
    statusSegments.addEventListener("click", function (e) {
      const compactHit =
        e.target && e.target.closest && e.target.closest("#compact-context-btn");
      if (compactHit) {
        e.stopPropagation();
        // Same delegation rule as the agent picker — the
        // handler must live on the container, not the re-created button node.
        if (typeof openContextSheet === "function") openContextSheet();
        return;
      }
      const hit = e.target && e.target.closest && e.target.closest("#agent-model-trigger");
      if (!hit) return;
      e.stopPropagation();
      const opening = _pop.hasAttribute("hidden");
      if (opening && typeof closeBudgetEditor === "function") closeBudgetEditor();
      if (opening && typeof openModelSheet === "function") openModelSheet();
      else close();
      // Opening is the moment the operator needs the roster. If it is empty
      // (failed fetch, gateway blip, an init step that threw), repair it now
      // rather than presenting a card with nothing to choose.
      if (opening && !(typeof state !== "undefined" && (state.agents || []).length)
          && typeof loadAgents === "function" && !_pop.dataset.rosterFetching) {
        _pop.dataset.rosterFetching = "1";
        // loadAgents() catches internally, so this settles either way; the
        // in-flight flag keeps repeated taps from racing last-response-wins.
        Promise.resolve(loadAgents())
          .then(function () {
            if (typeof renderAgentSelect === "function") renderAgentSelect();
          })
          .finally(function () { delete _pop.dataset.rosterFetching; });
      }
    });
    if (typeof bindModelSheetControls === "function") bindModelSheetControls();
  }
}

/** Paint cost/budget in the Agent Monitor summary, aligned at its right edge.
 *  The host is stable while this button is replaced on every budget/status
 *  refresh, so its editor click is delegated once from the host. */
function renderAgentMonitorBudget() {
  const host = document.getElementById("ama-budget");
  if (!host) return;
  let html = "";
  const budget = state.budget;
  if (state.currentSessionId && budget && budget.total &&
      typeof budgetSegment === "function") {
    html = budgetSegment(budget);
  } else if (state.currentSessionId && typeof state.sessionCostUsd === "number") {
    html =
      `<button type="button" class="seg seg-cost seg-budget seg-budget-btn"` +
      ` id="budget-edit-trigger" aria-haspopup="dialog" aria-expanded="false"` +
      ` title="Edit whole-tree budget"><span class="bud bud-ok">` +
      `${formatUsd(state.sessionCostUsd)}</span>` +
      `<span class="seg-caret" aria-hidden="true">▾</span></button>`;
  }
  host.innerHTML = html;
  host.hidden = !html;
  if (!host.dataset.budgetBound) {
    host.dataset.budgetBound = "1";
    host.addEventListener("click", function (event) {
      const hit = event.target && event.target.closest &&
        event.target.closest("#budget-edit-trigger");
      if (!hit) return;
      event.preventDefault();
      event.stopPropagation();
      if (typeof openBudgetEditor === "function") openBudgetEditor();
    });
  }
}

// >>> AGENT_SWITCH_CORE_START
function normalizeEffectiveMode(raw) {
  if (typeof raw === "string")
    return { id: raw, label: raw, behavior_kind: "" };
  raw = raw || {};
  return {
    id: raw.id || raw.effective_mode_id || raw.requested_mode_id || "",
    label: raw.label || raw.effective_mode_label || raw.id || "Unknown mode",
    behavior_kind: raw.behavior_kind || "",
  };
}

function normalizeSessionState(raw) {
  let value = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
  for (const key of ["session_state", "session", "state", "result"]) {
    const candidate = value[key];
    if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
      value = candidate;
      break;
    }
  }
  return {
    active_agent: value.active_agent || value.agent || "",
    effective_mode: normalizeEffectiveMode(
      value.effective_mode || {
        id: value.effective_mode_id || value.requested_mode_id,
        label: value.effective_mode_label,
        behavior_kind: value.behavior_kind,
      },
    ),
    revision: Number(value.revision) || 0,
    model: value.model || "",
    engine_model: value.engine_model || "",
    engine_model_alias: value.engine_model_alias || "",
  };
}

function installAgentCapabilities(raw) {
  const data = (raw && (raw.capabilities || raw.result)) || raw || {};
  const rows = Array.isArray(data)
    ? data
    : (data.agents || data.agent_capabilities || data.available_agents || []);
  const next = {};
  if (Array.isArray(rows)) {
    rows.forEach((row) => {
      const name = row.agent_name || row.name || row.agent;
      if (!name) return;
      next[name] = _capabilityEntry(row);
    });
  } else {
    Object.keys(rows || {}).forEach((name) => {
      next[name] = _capabilityEntry(rows[name] || {});
    });
  }
  state.availableAgentModes = next;
  const sid = state.currentSessionId;
  if (
    !state.modePickerTouched
    && typeof isFreshUnboundSession === "function"
    && isFreshUnboundSession(sid)
    && typeof applyFreshAgentModeIntent === "function"
  ) {
    applyFreshAgentModeIntent(sid, state.pickerAgent || state.currentAgent, "");
  }
  return next;
}

function _capabilityEntry(row) {
  return {
    engine: row.engine || row.engine_kind || "",
    default_mode_id:
      row.default_mode_id || row.defaultModeId || row.default_mode || "",
    modes: (
      row.modes || row.available_modes || row.availableModes || []
    ).map((mode) => ({
      ...normalizeEffectiveMode(mode),
      supported: mode.supported !== false,
      unsupported_reason: mode.unsupported_reason || mode.reason || "",
      native_behavior: !!mode.native_behavior,
    })),
  };
}

function engineForAgent(agent) {
  const entry = state.availableAgentModes[agent] || {};
  return String(entry.engine || (state.agentEngines || {})[agent] || "");
}

function profileLockedModes(agent) {
  // Native conversational: permission Allow all | Ask.
  // Native non-conversational: Allow all only.
  // External engines: null → use capabilities / catalogue.
  if (agent === "jarvis" || agent === "plan" || agent === "build") {
    return [
      { id: "auto", label: "Allow all", supported: true },
      { id: "ask", label: "Ask", supported: true },
    ];
  }
  if (
    agent === "explore" ||
    agent === "general" ||
    (typeof isNonConversationalAgent === "function" &&
      isNonConversationalAgent(agent))
  ) {
    return [{ id: "auto", label: "Allow all", supported: true }];
  }
  return null;
}

function cataloguePickerModes(engine, agent) {
  const locked = profileLockedModes(agent);
  if (locked) return locked;
  if (engine === "opencode_acp") {
    return [
      { id: "build", label: "Build", supported: true },
      { id: "plan", label: "Plan", supported: true },
    ];
  }
  if (engine === "cursor_acp") {
    return [
      { id: "agent", label: "Agent", supported: true },
      { id: "plan", label: "Plan", supported: true },
      { id: "ask", label: "Ask", supported: true },
    ];
  }
  if (engine === "kimi_acp") {
    return [
      { id: "auto", label: "Auto", supported: true },
      { id: "plan", label: "Plan", supported: true },
      { id: "default", label: "Default", supported: true },
    ];
  }
  if (engine === "hermes_acp") {
    return [
      { id: "dont_ask", label: "Don't Ask", supported: true },
      { id: "accept_edits", label: "Accept Edits", supported: true },
      { id: "default", label: "Default", supported: true },
    ];
  }
  if (engine === "devin_acp") {
    return [
      { id: "bypass", label: "Bypass Permissions", supported: true },
      { id: "accept-edits", label: "Code", supported: true },
      { id: "ask", label: "Ask", supported: true },
      { id: "plan", label: "Plan", supported: true },
    ];
  }
  if (
    engine === "claude_cli" ||
    engine === "codex_cli" ||
    engine === "antigravity_cli"
  ) {
    return [
      { id: "auto", label: "Allow all", supported: true },
      { id: "ask", label: "Ask", supported: true },
      { id: "plan", label: "Plan", supported: true },
    ];
  }
  return [{ id: "auto", label: "Allow all", supported: true }];
}

function pickerModes(agent) {
  const locked = profileLockedModes(agent);
  if (locked) return locked;
  const entry = state.availableAgentModes[agent] || {};
  const engine = engineForAgent(agent);
  const fromCaps = (entry.modes || []).filter((mode) => mode.supported);
  if (fromCaps.length) return fromCaps;
  return cataloguePickerModes(engine, agent);
}

function agentModes(agent) {
  return (state.availableAgentModes[agent] || {}).modes || [];
}

function renderAgentModeSelect(agent) {
  const select = document.getElementById("opt-mode");
  const row = document.querySelector(".opt-mode-row");
  if (!select) return;
  if (state.monitorReadOnly) {
    if (
      typeof isNonConversationalAgent === "function" &&
      isNonConversationalAgent(agent)
    ) {
      select.innerHTML = '<option value="auto">Allow all</option>';
      select.disabled = true;
      return;
    }
    select.innerHTML = '<option value="monitor">read only</option>';
    select.disabled = true;
    return;
  }
  const modes = pickerModes(agent);
  if (!modes.length || modes.length === 1) {
    if (modes.length === 1) {
      select.innerHTML = `<option value="${escapeHtml(modes[0].id)}">${escapeHtml(modes[0].label)}</option>`;
    } else {
      select.innerHTML = "";
    }
    select.disabled = true;
    if (row) row.hidden = true;
    return;
  }
  if (row) row.hidden = false;
  select.innerHTML = modes.map((mode) => {
    const reason = mode.unsupported_reason
      ? ` — ${mode.unsupported_reason}`
      : "";
    return `<option value="${escapeHtml(mode.id)}" ${mode.supported ? "" : "disabled"}>${escapeHtml(mode.label + reason)}</option>`;
  }).join("");
  select.disabled = !!state.switchPending;
}

function setSwitchPending(pending, message, isError) {
  state.switchPending = !!pending;
  state.switchError = isError ? (message || "") : "";
  const mode = document.getElementById("opt-mode");
  const status = document.getElementById("agent-switch-status");
  if (typeof agentSelect !== "undefined" && agentSelect)
    agentSelect.disabled = !!pending || !!state.monitorReadOnly;
  if (mode)
    mode.disabled =
      !!pending || !!state.monitorReadOnly || !agentModes(state.currentAgent).length;
  if (typeof sendBtn !== "undefined" && sendBtn) {
    // Don't grey out Stop/Queue while the tree is busy — updateSendState owns that.
    if (pending) sendBtn.disabled = true;
    else if (typeof updateSendState === "function") updateSendState();
    else sendBtn.disabled = !!state.isStreaming;
  }
  if (status) {
    status.hidden = !message;
    status.textContent = message || "";
    status.classList.toggle("error", !!isError);
  }
}

function switchFailureMessage(error) {
  // The person reads the message; the machine code stays in the payload.
  const data = (error && error.data) || {};
  return data.message || data.error || (error && error.message) || "Switch failed";
}

function applySessionState(raw, opts = {}) {
  const next = normalizeSessionState(raw);
  if (!next.active_agent) return false;
  if (
    !opts.force &&
    next.revision &&
    state.sessionRevision &&
    next.revision < state.sessionRevision
  ) return false;
  // Goals are session-scoped. Do not reset the tracker when active_agent flips;
  // a state refresh must not hide a still-live goal row.
  // Hide stays on session switch, /goal clear, met/archived, and the X.
  state.currentAgent = next.active_agent;
  state.sessionRevision = next.revision || state.sessionRevision;
  state.effectiveMode = next.effective_mode;
  state.currentMode = next.effective_mode.id;
  if (next.model) {
    state.currentModel = next.model;
  }
  if (next.engine_model) {
    state.engineModel = next.engine_model;
    // Alias travels with the engine model; absent alias
    // (or a cleared engine model) means the raw id is the display label.
    state.engineModelAlias = next.engine_model_alias || "";
  }
  const session = state.sessions && state.sessions[state.currentSessionId];
  if (session) {
    session.intent_materialized = true;
    session.agent = state.currentAgent;
    session.effective_mode = state.effectiveMode;
    session.revision = state.sessionRevision;
    if (next.model) session.model = next.model;
    if (next.engine_model) session.engine_model = next.engine_model;
  }
  state.commands = null;
  renderAgentSelect();
  updateHeader();
  renderStatusSegments();
  // A committed send-time switch changes the scoped monitor roots. Refresh
  // immediately so the new lane appears above the old one without waiting for
  // a subagent event or turn-end widget reconciliation.
  if (typeof updateAgentMonAccordion === "function") {
    updateAgentMonAccordion();
  }
  return true;
}

async function refreshAgentCapabilities(sid, epoch = state.navEpoch) {
  if (state.monitorReadOnly) return {};
  if (!sid) return {};
  const data = await apiGet(
    `/api/sessions/${encodeURIComponent(sid)}/capabilities`,
  );
  if (epoch !== state.navEpoch || sid !== state.currentSessionId) return {};
  const result = installAgentCapabilities(data);
  renderAgentModeSelect(state.currentAgent);
  return result;
}

async function refreshSessionState(sid, epoch = state.navEpoch) {
  if (state.monitorReadOnly) return null;
  if (!sid) return null;
  const data = await apiGet(
    `/api/sessions/${encodeURIComponent(sid)}/state`,
  );
  if (epoch !== state.navEpoch || sid !== state.currentSessionId) return null;
  applySessionState(data, { force: true });
  return normalizeSessionState(data);
}

function switchRequestId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function")
    return globalThis.crypto.randomUUID();
  return `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function preferredModeFor(agent, requested) {
  const entry = state.availableAgentModes[agent] || {};
  const supported = pickerModes(agent);
  if (requested && supported.some((mode) => mode.id === requested)) return requested;
  if (entry.default_mode_id && supported.some((mode) => mode.id === entry.default_mode_id))
    return entry.default_mode_id;
  return supported.length ? supported[0].id : "";
}

function isUnusedWebPlaceholder(sid) {
  if (!sid) return false;
  const session = (state.sessions || {})[sid];
  if (!session || session.origin === "gateway") return false;
  const messages = Array.isArray(session.messages) ? session.messages : [];
  const spoken = messages.some((m) => m && (m.role === "user" || m.role === "assistant"));
  return (
    !session.intent_materialized &&
    !Number(session.revision || 0) &&
    !Number(session.message_count || 0) &&
    !spoken
  );
}

function isFreshUnboundSession(sid) {
  if (!sid || state.isStreaming) return false;
  return isUnusedWebPlaceholder(sid);
}

// Snapshot one immutable browser intent before any first-submit side effect.
function firstIntentForSession(sid, agent = selectedPickerAgent()) {
  if (!isFreshUnboundSession(sid)) return Object.freeze({});
  const session = (state.sessions || {})[sid] || {};
  const selectedAgent = String(agent || "").trim();
  if (!selectedAgent) return Object.freeze({});
  const selectedMode =
    session.pending_mode_intent || preferredModeFor(selectedAgent, "");
  return Object.freeze({
    initial_agent: selectedAgent,
    ...(selectedMode ? { permission_mode: selectedMode } : {}),
  });
}

function applyFreshAgentModeIntent(sid, agent, requestedMode) {
  state.currentAgent = agent || state.currentAgent;
  const desiredMode = preferredModeFor(
    state.currentAgent, requestedMode || "",
  );
  const selectedMode = agentModes(state.currentAgent).find(
    (mode) => mode.supported && mode.id === desiredMode,
  );
  if (selectedMode) {
    state.effectiveMode = normalizeEffectiveMode(selectedMode);
    state.currentMode = state.effectiveMode.id;
  } else if (desiredMode) {
    state.effectiveMode = normalizeEffectiveMode(desiredMode);
    state.currentMode = desiredMode;
  } else {
    // Capabilities are not available yet. Do not carry another agent's mode
    // into turn 1; omitting the intent lets the selected agent's server-side
    // default remain authoritative.
    state.effectiveMode = normalizeEffectiveMode("");
    state.currentMode = "";
  }
  const session = sid && state.sessions && state.sessions[sid];
  if (session) {
    session.agent = state.currentAgent;
    session.effective_mode = state.effectiveMode;
    session.pending_agent_intent = state.currentAgent;
    session.pending_mode_intent =
      (state.effectiveMode && state.effectiveMode.id) || "";
  }
  if (typeof updateAgentMonAccordion === "function") {
    updateAgentMonAccordion();
  }
  return desiredMode;
}

async function switchSessionAgent(agent, requestedMode) {
  if (state.monitorReadOnly) return false;
  const sid = state.currentSessionId;
  // Before the first submit, the picker is local preselection. The first prompt
  // carries the intent and materializes the gateway session.
  if (!sid || isFreshUnboundSession(sid)) {
    const switchingAgent = agent !== state.currentAgent;
    // A mode token accompanying an agent change is leftover intent, not a
    // same-agent /mode selection. Keep first-bind consistent with the durable
    // SessionRuntime rule and let the target agent choose its default.
    applyFreshAgentModeIntent(
      sid, agent, switchingAgent ? "" : requestedMode,
    );
    state.engineModel = "";
    state.engineModelAlias = ""; // No model → no alias
    // Keep this agent's picker alias (or its catalogue default). Deleting
    // pickerModels[agent] then resetting the sheet to "" hid `.seg-model`
    // on New Chat until a later catalogue load — which never re-rendered
    // the chip.
    const kept = String(
      ((state.pickerModels || {})[agent])
      || (typeof catalogueDefaultAlias === "function" && catalogueDefaultAlias(agent))
      || "",
    ).trim();
    state.currentModel = kept;
    if (typeof modelSheetSelectionFor === "function") {
      if (typeof ensureModelSheetState === "function") ensureModelSheetState();
      state.modelSheet.effective = modelSheetSelectionFor(agent, kept);
      if (typeof cloneModelSelection === "function")
        state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
    }
    renderAgentSelect();
    renderAgentModeSelect(state.currentAgent);
    updateHeader();
    renderStatusSegments();
    // Persist picker intent and wake `event: tree` so the footer is not
    // stuck on the New Chat default (jarvis) while the composer shows cursor.
    try {
      await apiPost(
        `/api/sessions/${encodeURIComponent(sid)}/agent`,
        {
          target_agent: agent,
          target_mode: switchingAgent ? null : (requestedMode || null),
          frontend: "web",
        },
      );
    } catch (_e) { /* local picker already updated */ }
    // Keep configured choices fresh; this never imports backend active state.
    loadModelsForCurrentAgent({ loading: true });
    return true;
  }
  const epoch = ++state.navEpoch;
  const oldAgent = state.currentAgent;
  const oldMode = state.effectiveMode;
  // This is the send-time backend transaction. The frontend picker already
  // owns its agent/model intent and is deliberately untouched here.
  state.pendingAgent = agent;
  setSwitchPending(true, `Switching from ${oldAgent}…`, false);
  renderAgentModeSelect(agent);
  let desiredMode = requestedMode || "";
  try {
    await refreshAgentCapabilities(sid, epoch);
    if (epoch !== state.navEpoch) return false;
    desiredMode = preferredModeFor(agent, desiredMode);
    renderAgentModeSelect(agent);
    // Picking an AGENT is not picking a MODE. preferredModeFor() guesses from
    // the capabilities currently loaded — which describe the agent we are
    // leaving, not the one we are switching to — so for an external agent it
    // proposes a mode that agent cannot serve and the server refuses the whole
    // switch with `native_mode_rejected` ("native readiness handshake failed").
    // Measured: target_mode "auto" -> rejected after 11s; target_mode null ->
    // succeeds. Native agents happened to accept the guess, which is why this
    // looked like "the picker is broken for some agents".
    // So only send a mode the operator actually asked for; otherwise let the
    // server pick one the target supports.
    const submittedMode = requestedMode ? (desiredMode || null) : null;
    const submittedModel = typeof sendableInitialModel === "function"
      ? (sendableInitialModel(agent, state.currentModel) || null)
      : null;
    const submit = async () => apiPost(
      `/api/sessions/${encodeURIComponent(sid)}/agent`,
      {
        target_agent: agent,
        target_mode: submittedMode,
        target_model: submittedModel,
        expected_revision: state.sessionRevision,
        request_id: switchRequestId(),
        frontend: "web",
      },
    );
    let result;
    try {
      result = await submit();
    } catch (error) {
      // A prior rapid intent may have committed first. Re-read the authority
      // and retry only the still-latest intent once at the new revision.
      if (error && (error.code === "stale_revision" || String(error.message || "").includes("409") || error.status === 409) && epoch === state.navEpoch) {
        await refreshSessionState(sid, epoch);
        if (epoch !== state.navEpoch) return false;
        result = await submit();
      } else {
        throw error;
      }
    }
    if (epoch !== state.navEpoch || sid !== state.currentSessionId) return false;
    applySessionState(result, { force: true });
    if (typeof loadModelsForCurrentAgent === "function") {
      loadModelsForCurrentAgent({ loading: true });
    }
    state.pendingAgent = "";
    setSwitchPending(false, [
      state.currentAgent,
      state.effectiveMode.label,
    ].filter(Boolean).join(" · "), false);
    // The SID and its JSONL remain unchanged across an agent switch.
    closeSidebarMobile();
    return true;
  } catch (error) {
    if (epoch !== state.navEpoch || sid !== state.currentSessionId) return false;
    state.pendingAgent = "";
    state.effectiveMode = oldMode;
    updateHeader();
    renderAgentModeSelect(oldAgent);
    setSwitchPending(false, switchFailureMessage(error), true);
    return false;
  }
}

// Atomic same-session agent switch.
async function applyAgentSwitch(agent) {
  if (!agent || (
    agent === (state.pendingAgent || state.currentAgent) &&
    state.currentSessionId &&
    !isFreshUnboundSession(state.currentSessionId)
  )) {
    renderAgentSelect();
    return false;
  }
  state.modePickerTouched = false;
  const pending = switchSessionAgent(agent, "");
  state.agentSwitchPromise = pending;
  try {
    return await pending;
  } finally {
    if (state.agentSwitchPromise === pending) state.agentSwitchPromise = null;
  }
}

async function applyAgentModeSwitch(mode) {
  // Mode picker is permission / engine-native mode — never an agent switch.
  // Plan and Build stay in the agent picker.
  if (!mode || (
    state.effectiveMode &&
    mode === state.effectiveMode.id &&
    state.currentSessionId
  )) return false;
  return switchSessionAgent(state.currentAgent, mode);
}
// <<< AGENT_SWITCH_CORE_END

// Teacher-supervision indicator — COMPACT for small screens: just the 🧑‍🏫
// emoji with a hover/tap label carrying the full state (active / enabled /
// model), instead of three long "teacher_xxx=false" chips. The emoji lights up
// (.on) when supervision is actually active.
function renderTeacherWidget() {
  if (!teacherWidget) return;
  teacherWidget.hidden = true;
}

// Pull the goal-progress snapshot (terminal state + turns/budget) out of any
// status/state_patch payload into local state. Pure adapter — the backend
// (command_runtime._goal_state) computes the tracker; the JS only stores it.
