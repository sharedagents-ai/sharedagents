/* sharedagents-web — Frontend application (bootstrap / init).
 *
 * De-monolith Task 9: the former 8087-line app.js was split into ordered
 * classic scripts that share the global / window.App scope (same mechanism
 * nav.js/agents.js/goals.js/topics.js already use). They load BEFORE this file
 * (see templates/index.html): state.js (window.App + state + $/$$/BASE),
 * virtual_scroll, attachments, api, sessions, streaming, render, widgets, send,
 * voice, realtime_ui, markdown, commands, polish, launch. This file
 * keeps only bootstrap: DOM-element refs, init(), setupEventListeners(), and
 * the DOMContentLoaded entry. (The split rewrote no references; the device-id /
 * multi-tab-handoff / presence-polling block that used to live here was later
 * removed with the continue-here chip — see the note at the end of this file.)
 */

// ── DOM element references (resolved at load; app.js runs at end of <body>) ──
// `$`/`$$` come from state.js, which loads first.

const sidebar = $("#sidebar");
const sidebarBackdrop = $("#sidebar-backdrop");
const sessionList = $("#session-list");
const agentSelect = $("#agent-select");
const messagesContainer = $("#messages");
const messageInput = $("#message-input");
const sendBtn = $("#send-btn");
const treeStopBtn = $("#tree-stop-btn");
const micBtn = $("#mic-btn");
const realtimeBtn = $("#realtime-btn");
const realtimeVoiceStatus = $("#realtime-voice-status");
const realtimeListeningCaption = $("#realtime-listening-caption");
const newChatBtn = $("#new-chat-btn");
const toggleSidebarBtn = $("#toggle-sidebar");
const settingsBtn = $("#settings-btn");
const settingsModal = $("#settings-modal");
const closeSettingsBtn = $("#close-settings");
const currentAgentEl = $("#current-agent");
const modelSelect = $("#model-select");
const teacherToggle = $("#teacher-toggle");
const teacherToggleState = $("#teacher-toggle-state");
const teacherModelSelect = $("#teacher-model-select");
const teacherBadge = $("#teacher-badge");
const statusText = $("#status-text");
const sessionStatus = $("#session-status");
const statusSegments = $("#status-segments");
const teacherWidget = $("#teacher-widget");
const goalWidget = $("#goal-widget");
const harnessWidget = $("#harness-widget");
const queueWidget = $("#queue-widget");
const welcomeScreen = $("#welcome-screen");
const chatView = $("#chat-view");
const pastePreview = $("#paste-preview");

// Viewport helper — the drawer/bottom-sheet breakpoint mirrors the CSS media query.
const isMobile = () => window.matchMedia("(max-width: 768px)").matches;

// ── Init ──

// ══════════════════════════════════════════════════════════════════════
// Virtual Scroller — infinite scroll / virtual rendering engine
//
// Prevents DOM bloat on long conversations by rendering only a window of
// messages near the viewport. Messages far above/below are pruned from the
// DOM (data stays in state.msgCache for fast re-hydration).
//
// Architecture:
//   messagesContainer:
//     [sentinelTop]         ← invisible, height = scrolled-off content
//     [message N]           ← rendered window: state.scroll.windowStart..
//     [message N+1]              ..state.scroll.windowEnd
//     ...
//     [bottomMarker]        ← invisible, detected by IntersectionObserver
//
// The sentinelTop's height shrinks/grows as the user scrolls up/down,
// simulating the full scrollbar. When the sentinel enters the viewport,
// the window shifts up to load older messages. When the bottom marker is
// visible, autoScroll is enabled (new messages jump to bottom).
//
// Constants:
async function init() {
  VirtualScroller.init();
  initializePickerPreferences();
  setupEventListeners();
  autoResizeInput();
  // The whole boot: ONE REST read (GET /api/boot) + ONE conversation stream —
  // all an untouched chat page may call. With nothing to resume, bootChat
  // opens a New Chat draft (no request until the first send). The sidebar
  // lists ride the boot read when the menu is already visible, and load
  // lazily when the operator opens it later.
  await bootChat();
  renderAgentSelect();
  seedSelectedModelOnly();
  // The chip names the agent's saved default model (carried by the boot
  // read) — no catalogue load until the operator opens the Agent Selector.
  try {
    await ensurePickerDefaultModel(undefined, { passive: true });
  } catch (_e) { /* chip stays empty until the operator opens the picker */ }
  renderSessionStatusBar(); // hidden until a session is bound (bootChat may bind one)
  // The `state` frame on the single stream keeps the conn badge + health live.
  initMic();
  initRealtimeVoice();
  registerServiceWorker(); // Installable PWA (manifest + SW)
  // Notifications: reconcile stored prefs against the authoritative
  // Notification.permission BEFORE any turn event can fire. Must follow
  // registerServiceWorker() — every notification is displayed through the
  // worker's registration, never `new Notification()` (plan §3.3).
  if (window.Notifications) {
    window.Notifications.loadNotificationPrefs();
    window.Notifications.syncSettingsUi();
    // Permission can be granted from Chrome's own address-bar menu, which
    // fires no event — re-reconcile when the tab returns to the front.
    window.Notifications.watchForExternalPermissionChange();
  }
  // ?settings=1 deep-link opens the settings sheet on load.
  if (new URLSearchParams(window.location.search).get("settings") === "1") {
    settingsModal.classList.add("active");
  }
}

// ── Image attachments: preview, normalize, capture ───────────────────

/** True when the composer holds something submittable — typed text or a staged
 *  image / text-file attachment. */
// ── Notification controls (plan §6.0/§6.1) ───────────────────────────
// Every "enable" path calls Notifications.enableFromGesture() DIRECTLY from
// the click handler — no await, no wrapper that yields first. The handler
// creates and resumes the AudioContext synchronously before it requests
// permission; yielding first spends the user activation and the chime is then
// silent forever, with permission granted so nothing looks broken (plan §4.2).
function setupNotificationListeners() {
  const N = window.Notifications;
  if (!N) return;
  const on = (id, fn) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", fn);
  };
  on("notif-optin-yes", () => N.answerOptIn("yes"));
  on("notif-optin-later", () => N.answerOptIn("later"));
  on("notif-optin-never", () => N.answerOptIn("never"));
  on("enable-notifications", () => N.onSettingsButton());
  // Backgrounded delivery is the ONLY case that matters on a phone, and it is
  // unobservable while you are looking at the page — hence a delayed test.
  on("notif-bg-test", () => N.scheduleBackgroundTest(5));
  [
    ["notif-turn-complete", "turn_complete"],
    ["notif-turn-error", "turn_error"],
    ["notif-question", "question"],
  ].forEach(([id, key]) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", () => N.setTrigger(key, el.checked));
  });
}

function setupEventListeners() {
  setupNotificationListeners();
  // Send message
  sendBtn.addEventListener("click", (e) => {
    e.preventDefault();
    sendMessage();
  });
  if (treeStopBtn) {
    treeStopBtn.addEventListener("click", () => {
      if (typeof cancelStreaming === "function") cancelStreaming();
    });
  }

  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // ── Paste handler: capture files from clipboard ──
  messageInput.addEventListener("paste", (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    const files = [];
    for (const item of items) {
      if (item.kind !== "file") continue;
      const f = item.getAsFile();
      if (f) files.push(f);
    }
    if (!files.length) return; // plain text paste — let it through
    e.preventDefault(); // suppress text fallback when files are present
    attachFiles(files); // images stage; non-images get a visible notice
  });

  // Paste-preview clicks — ONE delegated listener (renderPastePreview rebuilds
  // the thumbs; per-render binding would re-stack handlers). Remove buttons
  // strip a staged attachment; clicking the thumb itself opens the lightbox
  // so the operator can inspect the full image before send.
  if (pastePreview)
    pastePreview.addEventListener("click", (e) => {
      // Image thumbnail remove
      let btn = e.target.closest(".paste-thumb-remove");
      if (btn) {
        const idx = parseInt(btn.dataset.idx, 10);
        if (!isNaN(idx) && idx >= 0 && idx < state.pastedImages.length) {
          state.pastedImages.splice(idx, 1);
          renderPastePreview();
        }
        return;
      }
      // Text-file card remove
      btn = e.target.closest(".file-remove");
      if (btn) {
        const idx = parseInt(btn.dataset.idx, 10);
        if (!isNaN(idx) && idx >= 0 && idx < state.attachedTextFiles.length) {
          state.attachedTextFiles.splice(idx, 1);
          renderPastePreview();
        }
        return;
      }
      // Click staged image thumb (not the remove control) → lightbox.
      const thumb = e.target.closest(".paste-thumb");
      if (thumb && !e.target.closest(".paste-thumb-remove")) {
        const url = thumb.dataset.artifactUrl || "";
        const name = thumb.dataset.artifactName || "Attached image";
        if (url && typeof openMediaLightbox === "function") {
          openMediaLightbox(url, name, thumb);
        } else if (url && typeof _openArtifactLightbox === "function") {
          _openArtifactLightbox({ url, name, opener: thumb });
        }
      }
    });

  // ── Annex (📎) button: file-picker route into the same pipeline ──
  const attachBtn = $("#attach-btn");
  const attachInput = $("#attach-input");
  if (attachBtn && attachInput) {
    attachBtn.addEventListener("click", () => attachInput.click());
    attachInput.addEventListener("change", () => {
      attachFiles(Array.from(attachInput.files || []));
      attachInput.value = ""; // re-selecting the same file must re-fire change
    });
  }

  // New chat — also dismiss the drawer on mobile so the fresh chat is visible.
  newChatBtn.addEventListener("click", () => {
    createSession(undefined, { explicit: true });
    closeSidebarMobile();
  });

  // Copy an assistant answer (markdown, reasoning excluded) to the clipboard.
  messagesContainer.addEventListener("click", (e) => {
    // Collapse-replies mode: tapping a folded assistant reply expands it (and
    // tapping again re-folds). Ignore clicks on links / buttons / the reasoning
    // disclosure so those still work.
    if (
      state.collapseReplies &&
      !e.target.closest("a, button, .msg-copy, details, summary")
    ) {
      const am = e.target.closest(".message-assistant");
      if (am) {
        am.classList.toggle("msg-expanded");
        return;
      }
    }
    const btn = e.target.closest(".msg-copy");
    if (!btn) return;
    const msg = btn.closest(".message");
    const raw = (msg && msg.dataset.raw) || "";
    const answer = parseAssistantContent(raw).answer || raw;
    const done = (label) => {
      btn.textContent = label;
      setTimeout(() => {
        btn.textContent = "Copy";
      }, 1200);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(answer).then(
        () => done("Copied"),
        () => done("Copy failed"),
      );
    } else {
      done("Copy failed");
    }
  });

  // Microphone button — toggle audio recording
  if (micBtn) {
    micBtn.addEventListener("click", toggleRecording);
  }

  // Realtime voice button (rightmost) — toggle the live streaming session.
  if (realtimeBtn) {
    realtimeBtn.addEventListener("click", toggleRealtimeVoice);
  }

  // Toggle sidebar — drawer on mobile, collapsible column on desktop.
  toggleSidebarBtn.addEventListener("click", toggleSidebar);
  sidebarBackdrop.addEventListener("click", closeSidebarMobile);

  // Agent selection updates the current session. Before the first turn it only
  // preselects the agent.
  agentSelect.addEventListener("change", async () => {
    selectPickerAgent(agentSelect.value);
    const sid = state.currentSessionId;
    if (sid && isFreshUnboundSession(sid)) {
      applyFreshAgentModeIntent(sid, agentSelect.value, "");
      state.currentAgent = agentSelect.value;
      const sess = state.sessions && state.sessions[sid];
      if (sess) {
        sess.agent = agentSelect.value;
        sess.pending_agent = agentSelect.value;
      }
      try {
        await apiPost(
          `/api/sessions/${encodeURIComponent(sid)}/agent`,
          {
            target_agent: agentSelect.value,
            target_mode: null,
            frontend: "web",
          },
        );
      } catch (_e) {}
      if (typeof refreshSessionWidgets === "function") refreshSessionWidgets();
    }
  });

  // Session selection (sidebar)
  sessionList.addEventListener("click", async (e) => {
    // Delete button
    const delBtn = e.target.closest(".delete-session");
    if (delBtn) {
      e.stopPropagation();
      // Destructive + irreversible — confirm first (covers gateway-adopted
      // conversations that aren't just local scratch chats).
      const ok = await confirmCard({
        title: "Delete this conversation?",
        okLabel: "Delete",
        danger: true,
      });
      if (ok) deleteSession(delBtn.dataset.delete);
      return;
    }

    const item = e.target.closest(".session-item");
    if (!item) return;
    loadSession(item.dataset.session);
    closeSidebarMobile();
  });

  // Topic quick-jump (sidebar) — open that topic's most recent session.
  const topicList = document.getElementById("topic-list");
  if (topicList) {
    topicList.addEventListener("click", (e) => {
      // "+" button: start a NEW chat bound to this topic (takes priority over
      // the row's open-latest behaviour, and must not bubble into it).
      const newBtn = e.target.closest(".topic-new-chat");
      if (newBtn) {
        e.stopPropagation();
        const topic = newBtn.dataset.topicNew || "";
        createSession(topic, { explicit: true });
        closeSidebarMobile();
        return;
      }
      const item = e.target.closest(".topic-item");
      if (!item) return;
      const sid = item.dataset.topicSession;
      if (!sid) return;
      loadSession(sid);
      closeSidebarMobile();
    });
  }

  // Pending-approval quick-jump (sidebar) — open the
  // conversation that raised the decision so the operator can action it there.
  const approvalsList = document.getElementById("pending-approvals");
  if (approvalsList) {
    approvalsList.addEventListener("click", (e) => {
      // An in-place permission decision resolves the round-trip
      // without leaving the panel.
      const permBtn = e.target.closest("[data-perm-option]");
      if (permBtn) {
        e.stopPropagation();
        // resolvePermission now PROPAGATES delivery failures (it used to
        // swallow them). The panel row is re-rendered from the refreshed feed
        // either way, so there is nothing to re-arm here — but the rejection
        // must be consumed or it surfaces as an unhandled promise rejection.
        Promise.resolve(
          resolvePermission(
            permBtn.dataset.permSession,
            permBtn.dataset.permRequest,
            permBtn.dataset.permOption,
          ),
        ).catch(() => {});
        return;
      }
      const row = e.target.closest(".pending-approval");
      if (!row) return;
      const sid = row.dataset.approvalSession;
      if (!sid) return;
      loadSession(sid);
      closeSidebarMobile();
    });
  }

  // Model select — switch the active model through the shared runtime.
  modelSelect.addEventListener("change", () => switchModel(modelSelect.value));

  // Teacher controls — toggle supervision + pick the supervising model.
  if (teacherToggle)
    teacherToggle.addEventListener("click", () => toggleTeacher());
  if (teacherModelSelect) {
    teacherModelSelect.addEventListener("change", () =>
      selectTeacherModel(teacherModelSelect.value),
    );
  }

  // Agent mode is capability-resolved by the server and uses the same atomic
  // switch transaction as agent identity. Reasoning/tools remain ordinary
  // lane-local runtime commands.
  const optMode = $("#opt-mode"),
    optActivity = $("#opt-activity"),
    optReasoning = $("#opt-reasoning"),
    optTools = $("#opt-tools");
  // Update BOTH the live display state AND the cached statusInfo so the change
  // sticks immediately (otherwise the stale statusInfo would revert the control).
  if (optMode)
    optMode.addEventListener("change", async () => {
      state.modePickerTouched = true;
      await applyAgentModeSwitch(optMode.value);
    });
  const bindVisExpandToggle = (btn, key) => {
    if (!btn) return;
    btn.addEventListener("click", () => {
      if (!state.visExpand) state.visExpand = { activity: true, reasoning: false, tools: false };
      state.visExpand[key] = !state.visExpand[key];
      try {
        localStorage.setItem(
          `sharedagents.web.visExpand.${key}`,
          state.visExpand[key] ? "1" : "0",
        );
      } catch (_e) {}
      syncOptionsControls();
      if (typeof applyVisExpand === "function") applyVisExpand(null, { force: true });
    });
  };
  bindVisExpandToggle(optActivity, "activity");
  bindVisExpandToggle(optReasoning, "reasoning");
  bindVisExpandToggle(optTools, "tools");
  const optTransport = $("#opt-transport");
  if (optTransport) {
    if (typeof syncTransportToggle === "function") syncTransportToggle();
    optTransport.addEventListener("click", (e) => {
      const choice = e.target && e.target.closest
        ? e.target.closest(".opt-live-choice")
        : null;
      const next = choice && choice.getAttribute("data-transport");
      if (!next || typeof setLiveTransport !== "function") return;
      setLiveTransport(next);
    });
  }
  // Collapse-replies: fold every assistant reply to a short preview so a long
  // conversation reads as a list of prompts (tap a reply to expand it). The
  // global toggle lives here; per-reply expand is handled on the messages click.
  const optCollapse = $("#opt-collapse");
  if (optCollapse) {
    try {
      state.collapseReplies = !!localStorage.getItem(
        "sharedagents.web.collapseReplies",
      );
    } catch (_e) {}
    const applyCollapse = () => {
      optCollapse.setAttribute("aria-pressed", String(!!state.collapseReplies));
      optCollapse.classList.toggle("is-active", !!state.collapseReplies);
      if (messagesContainer)
        messagesContainer.classList.toggle(
          "collapse-replies",
          !!state.collapseReplies,
        );
    };
    optCollapse.addEventListener("click", () => {
      state.collapseReplies = !state.collapseReplies;
      try {
        localStorage.setItem(
          "sharedagents.web.collapseReplies",
          state.collapseReplies ? "1" : "",
        );
      } catch (_e) {}
      applyCollapse();
    });
    applyCollapse();
  }

  // On open, reflect current values and (on phones) anchor the full-width panel
  // just below the header — the header grows when the title wraps, so measure it.
  const optionsCard = document.getElementById("options-card");
  if (optionsCard)
    optionsCard.addEventListener("toggle", () => {
      if (!optionsCard.open) return;
      syncOptionsControls();
      const body = optionsCard.querySelector(".options-body");
      const header = document.getElementById("chat-header");
      if (body) {
        body.style.top =
          body && header && window.matchMedia("(max-width: 600px)").matches
            ? header.getBoundingClientRect().bottom + 4 + "px"
            : "";
      }
    });
  // The options card is a <details> popover — natively it stays open until the
  // ⚙ is clicked again. Close it on any pointer-down outside, like a dropdown.
  if (optionsCard)
    document.addEventListener("pointerdown", (e) => {
      if (optionsCard.open && !optionsCard.contains(e.target))
        optionsCard.open = false;
    });

  // Settings modal
  settingsBtn.addEventListener("click", () => {
    settingsModal.classList.add("active");
  });

  closeSettingsBtn.addEventListener("click", () => {
    settingsModal.classList.remove("active");
  });

  settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) {
      settingsModal.classList.remove("active");
    }
  });

  // Escape — cancel a stream, else dismiss the drawer / panel / settings.
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (state.isStreaming) {
      cancelStreaming();
      return;
    }
    if (document.body.classList.contains("sidebar-open")) {
      closeSidebarMobile();
      return;
    }
    if (!panel.hidden) {
      closePanel();
      return;
    }
    if (settingsModal.classList.contains("active"))
      settingsModal.classList.remove("active");
  });
}

// ── Start ──

document.addEventListener("DOMContentLoaded", () => {
  init();
  setupCommandUI();
});

// The client half of cross-device presence (presence reporting + the
// "Live on another device — continue here" chip) was removed:
// the chip sat pinned over the composer on mobile and the handoff offer was
// unwanted. The server endpoints (/api/presence, /api/presence/continue-here,
// /api/sessions/<sid>/handoff) are intentionally left in place — the browser
// simply no longer reports presence or polls for an offer, so nothing renders.

document.addEventListener("DOMContentLoaded", () => {
  const saveGw = document.getElementById("save-gateway");
  if (saveGw) saveGw.addEventListener("click", switchGateway);
});

// Sidebar lists — MenuState runs a loader only when the drawer is visible and
// the section is open. Defaults: topics + conversations expanded once the menu
// is shown. A menu already visible at load gets its lists from the boot read;
// one opened later fetches them then.
document.addEventListener("DOMContentLoaded", () => {
  if (window.MenuState) {
    MenuState.registerLoader("sidebar-conversations", () =>
      sidebarListAfterBoot("sessions", () => loadSessionList({ limit: 20 })));
    MenuState.registerLoader("sidebar-topics", () =>
      sidebarListAfterBoot("topics", () => loadTopicList()));
  }
  // The "all →" link inside the Topics summary must navigate, not ALSO
  // toggle the parent <details> (both would otherwise fire on one click).
  const topicsLink = document.querySelector("#sidebar-topics .section-label-link");
  if (topicsLink) topicsLink.addEventListener("click", (e) => e.stopPropagation());
});
