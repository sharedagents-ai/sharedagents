/* sessions.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
function renderAgentSelect() {
  const colors = {
    main: "#58a6ff",
    claude: "#d29922",
    codex: "#f85149",
    antigravity: "#bc8cff",
    cursor: "#79c0ff",
  };
  renderAgentCard();
  if (typeof agentSelect === "undefined" || !agentSelect) return;
  const shownAgent = selectedPickerAgent();
  agentSelect.innerHTML = state.agents
    .map(
      (agent) => `
    <option value="${escapeHtml(agent)}" ${agent === shownAgent ? "selected" : ""}>${escapeHtml(agent)}</option>
  `,
    )
    .join("");
  // Keep the header current-agent text in sync
  if (typeof currentAgentEl !== "undefined" && currentAgentEl) {
    currentAgentEl.textContent = shownAgent;
  }
  if (typeof updateHeader === "function") updateHeader();
}

function applyMonitorReadOnlyUi() {
  const readOnly = !!state.monitorReadOnly;
  const lock = (node) => {
    if (!node) return;
    if (readOnly) {
      if (!node.dataset.monitorLock) {
        node.dataset.monitorLock = "1";
        node.dataset.monitorWasDisabled = node.disabled ? "1" : "0";
      }
      node.disabled = true;
    } else if (node.dataset.monitorLock) {
      node.disabled = node.dataset.monitorWasDisabled === "1";
      delete node.dataset.monitorLock;
      delete node.dataset.monitorWasDisabled;
    }
  };
  [
    agentSelect,
    modelSelect,
    messageInput,
    sendBtn,
    micBtn,
    realtimeBtn,
    document.getElementById("attach-btn"),
    document.getElementById("commands-btn"),
    document.getElementById("opt-mode"),
    document.getElementById("opt-activity"),
    document.getElementById("opt-reasoning"),
    document.getElementById("opt-tools"),
    teacherToggle,
    teacherModelSelect,
  ].forEach(lock);
  if (messageInput) {
    if (readOnly) {
      if (!messageInput.dataset.monitorPlaceholder)
        messageInput.dataset.monitorPlaceholder = messageInput.placeholder || "";
      messageInput.placeholder = state.readOnlyReason || "Read-only subagent session";
      messageInput.title = state.readOnlyReason ||
        "Opened from Agent Monitor — history and live output are read-only";
    } else if (messageInput.dataset.monitorPlaceholder != null) {
      messageInput.placeholder = messageInput.dataset.monitorPlaceholder;
      messageInput.title = "";
      delete messageInput.dataset.monitorPlaceholder;
    }
  }
  if (statusText) {
    const lockedText = state.viewerLock && state.readOnlyReason === state.viewerLock
      ? "no agents yet"
      : (state.readOnlyReason ? "view only" : "read-only subagent session");
    if (readOnly) statusText.textContent = lockedText;
    else if (
      statusText.textContent === "read-only subagent session" ||
      statusText.textContent === "view only" ||
      statusText.textContent === "no agents yet"
    )
      statusText.textContent = "";
  }
  updateSendState();
}

// ── Sessions ──

function mergeSessionListRow(ex, s) {
  if (!ex) return s;
  const out = Object.assign({}, ex, s);
  if (isDefaultSessionTitle(s.title) && !isDefaultSessionTitle(ex.title))
    out.title = ex.title;
  if (!s.model && ex.model) out.model = ex.model;
  if (!s.engine_model && ex.engine_model) out.engine_model = ex.engine_model;
  if (!s.engine_model_alias && ex.engine_model_alias)
    out.engine_model_alias = ex.engine_model_alias;
  if ((!s.messages || !s.messages.length) && ex.messages)
    out.messages = ex.messages;
  return out;
}

function loadSessionList(opts = {}) {
  // The lazy sidebar loads the REST of the conversations: the last 20,
  // newest-activity first (the server's response order — no client sort).
  const limit = Number(opts.limit) || 20;
  return apiGet(`/api/sessions?limit=${limit}`)
    .then((data) => applySessionListRows(data.sessions || []))
    .catch(() => {});
}

// Install one Conversations page — from GET /api/sessions, or from the boot
// read when the menu was already visible at load.
function applySessionListRows(sessions) {
  const prev = state.sessions || {};
  const next = {};
  sessions.forEach((s) => {
    next[s.id] = mergeSessionListRow(prev[s.id], s);
  });
  // A monitor deep-link may open a child the sidebar list omitted.
  const cur = state.currentSessionId;
  if (cur && prev[cur] && !next[cur]) next[cur] = prev[cur];
  state.sessions = next;
  renderSessionList();
  // Title / summary may have landed server-side (first-turn derive or
  // OpenCode title agent) — refresh the chat header too, not only the
  // sidebar list.
  if (typeof updateHeader === "function") updateHeader();
  if (typeof syncModelSheetEffectiveFromSession === "function")
    syncModelSheetEffectiveFromSession();
}

// Open a conversation by gateway hash IN-PLACE (no full reload). Exposed for
// the chat-docked Agent Monitor accordion: clicking a parent / sibling /
// subagent row switches the chat to that session. Resolves to an already-bound
// web conversation when one exists, else opens the gateway session by hash.
window.saOpenConversation = function (hash, options = {}) {
  if (!hash) return;
  const match = Object.values(state.sessions).find(
    (s) => s.id === hash,
  );
  const target = match ? match.id : hash;
  if (options.fromMonitor) {
    // Keep the parent/sibling breadcrumb, but open the same canonical Web/
    // logical session target used by the full Agent Monitor tab. The response
    // explicitly reports the mounted session history.
    loadSession(target, {
      monitorNavigation: {
        returnSessionId: String(
          state.monitorReturnSessionId || state.currentSessionId || "",
        ),
        scopeHash: String(options.scopeHash || ""),
      },
    });
    return;
  }
  loadSession(target);
};

// Per-agent accent — keys the leading conversation dot to the engine that
// last ran the session (mirrors renderAgentSelect's palette). Falls back to a
// neutral hue for unknown agents.
const AGENT_DOT_COLORS = {
  main: "#58a6ff",
  claude: "#d29922",
  codex: "#f85149",
  antigravity: "#bc8cff",
  cursor: "#79c0ff",
  "self-improver": "#5aff8a",
};
function agentDotColor(agent) {
  return AGENT_DOT_COLORS[agent] || "var(--text-muted)";
}

// "4h" / "2d" / "now" — compact relative age, Open-WebUI style. Returns '' if
// the timestamp is missing/unparseable so the slot collapses cleanly.
function relativeAge(iso) {
  if (!iso) return "";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "";
  const secs = Math.max(0, (Date.now() - then) / 1000);
  if (secs < 60) return "now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return mins + "m";
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + "h";
  const days = Math.floor(hrs / 24);
  if (days < 7) return days + "d";
  const wks = Math.floor(days / 7);
  if (wks < 5) return wks + "w";
  const mos = Math.floor(days / 30);
  if (mos < 12) return mos + "mo";
  return Math.floor(days / 365) + "y";
}

// Date-bucket a session into the sidebar groups Open WebUI uses: Today /
// Yesterday / Previous 7 Days / Previous 30 Days / Older. Unknown timestamps
// sink to the bottom bucket so they never jump above dated conversations.
function sessionBucket(iso) {
  const t = iso ? Date.parse(iso) : NaN;
  if (Number.isNaN(t)) return 4;
  const now = new Date();
  const startOfToday = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate(),
  ).getTime();
  const dayMs = 86400000;
  if (t >= startOfToday) return 0; // Today
  if (t >= startOfToday - dayMs) return 1; // Yesterday
  if (t >= startOfToday - 7 * dayMs) return 2; // Previous 7 Days
  if (t >= startOfToday - 30 * dayMs) return 3; // Previous 30 Days
  return 4; // Older
}
const SESSION_BUCKET_LABELS = [
  "Today",
  "Yesterday",
  "Previous 7 Days",
  "Previous 30 Days",
  "Older",
];

function renderSessionList() {
  const sessions = Object.values(state.sessions).sort((a, b) =>
    (b.updated_at || "").localeCompare(a.updated_at || ""),
  );

  let lastBucket = -1;
  const rows = [];
  for (const s of sessions) {
    const bucket = sessionBucket(s.updated_at);
    if (bucket !== lastBucket) {
      rows.push(
        `<div class="session-group">${SESSION_BUCKET_LABELS[bucket]}</div>`,
      );
      lastBucket = bucket;
    }
    // Conversations that originated outside the web (e.g. started in the TUI)
    // get a small agent chip so they're recognisable; they're fully openable
    // + continuable here all the same. The leading dot is colour-keyed to the
    // agent regardless of origin.
    const agent = s.agent || (s.origin === "gateway" ? "tui" : "");
    const tag =
      s.origin === "gateway"
        ? `<span class="session-origin" title="Started outside the web (TUI / agent)">${escapeHtml(s.agent || "tui")}</span>`
        : "";
    // Shared conversations: a people mark, and whose it is
    // when someone else owns it.
    const shared =
      s.visibility === "public" || (s.sharing && s.sharing.visibility === "public")
        ? `<span class="session-shared" title="Shared" aria-label="Shared"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" aria-hidden="true"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7"/></svg></span>`
        : "";
    const ownerTag = s.owner_name
      ? `<span class="session-owner" title="${escapeAttr("Owned by " + (s.owner_email || s.owner_name))}">${escapeHtml(s.owner_name)}</span>`
      : "";
    const title =
      s.title && !isPlaceholderTitle(s.title) ? s.title : "New Chat";
    const titleAttr = title;
    const age = relativeAge(s.updated_at);
    const isLoading = s.id === state.loadingSessionId;
    const livePill = isLoading
      ? `<span class="live-pill session-loading-pill" title="Loading session…">Loading…</span>`
      : sessionActivityPill(s);
    rows.push(`
    <div class="session-item ${s.id === state.currentSessionId ? "active" : ""} ${isLoading ? "is-loading" : ""}" data-session="${escapeHtml(s.id)}">
      <span class="session-dot" style="background:${agentDotColor(agent)}" title="${escapeHtml(agent || "web")}"></span>
      <span class="session-title" title="${escapeAttr(titleAttr)}">${escapeHtml(title)}</span>
      <span class="session-meta">
        ${livePill}
        ${age ? `<span class="session-time">${escapeHtml(age)}</span>` : ""}
        ${shared}${ownerTag}
        ${tag}
      </span>
      ${s.can_manage === false ? "" : `<button class="delete-session" data-delete="${escapeHtml(s.id)}" title="Delete">×</button>`}
    </div>`);
  }
  sessionList.innerHTML = rows.join("");
}

/** Sidebar activity pill from the shared V2 projection (legacy fallback kept). */
function sessionActivityPill(s) {
  if (!s) return "";
  const a = s.activity && typeof s.activity === "object" ? s.activity : null;
  if (a && ["question", "permission", "operator_paused"].includes(a.attention)) {
    const label = a.display && a.display.label ? a.display.label : "Needs input";
    return `<span class="live-pill waiting-pill" title="${escapeAttr(label)}">${escapeHtml(label)}</span>`;
  }
  if (a && a.stale) {
    return `<span class="live-pill waiting-pill" title="Stale activity snapshot">Stale</span>`;
  }
  if (a && a.status === "needs_input") {
    const label = a.display && a.display.label ? a.display.label : "Needs input";
    return `<span class="live-pill waiting-pill" title="${escapeAttr(label)}">${escapeHtml(label)}</span>`;
  }
  if (s.state === "waiting") {
    return `<span class="live-pill waiting-pill" title="Waiting — background subagents still running">Waiting</span>`;
  }
  if (s.live) {
    return `<span class="live-pill" title="Running — a turn is in progress">Live</span>`;
  }
  return "";
}

// ── Topics quick-jump (sidebar) ──
// Conversations grouped by topic (synced from Telegram forum topics). Each row
// opens that topic's MOST RECENT session — /api/topics returns each topic's
// `sessions` newest-first, so sessions[0] is the latest. Pure read view; the
// full Discord-style browser is at /topics.
// ── PWA install ────────────────────────────────────
// Register the (non-caching) service worker so the app is installable to the
// home screen. Scope = BASE + "/" so it covers the app whether served at the
// root or under a reverse-proxy base path. Silent no-op where SW is
// unsupported (non-HTTPS, old browser) — installability simply won't offer.
function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  const scope = (BASE || "") + "/";
  navigator.serviceWorker
    .register((BASE || "") + "/sw.js", { scope })
    .catch((e) => console.debug("SW registration skipped:", e));
}

// ── Pending-approvals panel ────────────────────────
// Every decision awaiting THIS operator, aggregated across all their sessions,
// so a pending approval is reachable no matter which conversation raised it.
// Read-only surface: a row deep-links to its conversation, where it is actioned
// through the ordinary answer card (the panel is a view, not a second channel).
// Mount inline ask-user + tool-permission cards from the pending-approvals
// feed once the active session id is known (boot, sidebar switch, SSE state).
function _hydratePendingInteractiveCards() {
  if (typeof hydratePendingPermissionCards === "function") {
    hydratePendingPermissionCards();
  }
}

function _applyBootApprovals(boot) {
  if (!boot || !Array.isArray(boot.approvals)) return;
  state.pendingApprovals = boot.approvals;
  if (typeof renderPendingApprovals === "function") renderPendingApprovals();
}

function loadPendingApprovals() {
  return apiGet("/api/approvals")
    .then((data) => {
      state.pendingApprovals = Array.isArray(data && data.approvals)
        ? data.approvals
        : [];
      renderPendingApprovals();
    })
    .catch(() => {});
}

function renderPendingApprovals() {
  const listEl = document.getElementById("pending-approvals");
  const sectionEl = document.getElementById("sidebar-approvals");
  const countEl = document.getElementById("pending-approvals-count");
  if (!listEl || !sectionEl) {
    if (typeof hydratePendingPermissionCards === "function") {
      hydratePendingPermissionCards();
    }
    return;
  }
  const items = Array.isArray(state.pendingApprovals)
    ? state.pendingApprovals.filter((a) => a && a.session_id)
    : [];
  if (countEl) countEl.textContent = String(items.length);
  if (!items.length) {
    sectionEl.hidden = true;
    listEl.innerHTML = "";
    if (typeof hydratePendingPermissionCards === "function") {
      hydratePendingPermissionCards();
    }
    return;
  }
  sectionEl.hidden = false;
  listEl.innerHTML = items
    .map((a) => {
      const sid = a.session_id || "";
      const summary = (a.summary || "Decision requested").toString();
      const active = sid === state.currentSessionId;
      // A tool-permission request is actionable IN PLACE — the
      // panel is a view, not a second channel; actioning still goes through the
      // authenticated, session-bound /permission route. Other approval types keep
      // the deep-link-to-conversation behavior.
      if (a.type === "tool_permission") {
        const reqId = a.call_id || "";
        const opts = Array.isArray(a.options) && a.options.length
          ? a.options
          : ["allow_once", "reject_once"];
        const buttons = opts
          .map((o) =>
            `<button type="button" class="pending-approval-btn perm-${escapeAttr(o)}" data-perm-session="${escapeAttr(sid)}" data-perm-request="${escapeAttr(reqId)}" data-perm-option="${escapeAttr(o)}">${escapeHtml(permOptionLabel(o))}</button>`,
          )
          .join("");
        return `
    <div class="pending-approval pending-approval-permission ${active ? "active" : ""}" title="${escapeAttr(a.detail || summary)}">
      <span class="pending-approval-dot" aria-hidden="true"></span>
      <span class="pending-approval-text">${escapeHtml(summary)}</span>
      <span class="pending-approval-actions">${buttons}</span>
    </div>`;
      }
      return `
    <div class="pending-approval ${active ? "active" : ""}" data-approval-session="${escapeAttr(sid)}" title="Open the conversation awaiting your decision">
      <span class="pending-approval-dot" aria-hidden="true"></span>
      <span class="pending-approval-text">${escapeHtml(summary)}</span>
    </div>`;
    })
    .join("");
  if (typeof hydratePendingPermissionCards === "function") {
    hydratePendingPermissionCards();
  }
}

// Friendly label for an ACP permission option kind.
function permOptionLabel(kind) {
  switch (String(kind)) {
    case "allow_once": return "Allow";
    case "allow_always": return "Allow always";
    case "reject_once": return "Deny";
    case "reject_always": return "Deny always";
    default: return String(kind);
  }
}

// Resolve a tool-permission round-trip with the operator's pick.
// Posts the ACP outcome the gateway unwraps; refreshes the panel so the resolved
// item clears.
// PROPAGATES delivery failures. The old ``.catch(() => {})`` swallowed every
// rejection, so a 404/500/offline POST looked identical to success: the sidebar
// row simply stayed put and the inline card hung on its
// pending "…" with the buttons locked until the gateway's 600s timeout denied it.
// The panel still refreshes either way — but the caller now learns it failed and
// can re-arm. Every call site must attach a .catch().
// ``optionId`` null/"" posts a bare ``cancelled`` — the standing deny. Used when
// the gate recorded no options to choose from, where the route (correctly)
// refuses ``selected`` for anything, since nothing was ever offered.
function resolvePermission(sid, requestId, optionId) {
  const outcome = optionId
    ? { outcome: "selected", optionId: optionId }
    : { outcome: "cancelled" };
  return apiPost("/api/sessions/" + encodeURIComponent(sid) + "/permission", {
    requestId: requestId,
    outcome: outcome,
  }).then(
    (res) => Promise.resolve(loadPendingApprovals()).then(() => res),
    (err) => Promise.resolve(loadPendingApprovals()).then(() => {
      throw err;
    }),
  );
}

function loadTopicList() {
  return apiGet("/api/topics") // compact default: no per-topic session arrays
    .then((data) => applyTopicList(data && data.topics))
    .catch(() => {});
}

// Install the Topics list — from GET /api/topics, or from the boot read when
// the menu was already visible at load.
function applyTopicList(topics) {
  state.topics = Array.isArray(topics) ? topics : [];
  state.topicsLoaded = true;
  renderTopicList();
}

// Per-topic emoji: a keyword map for known topics, with a deterministic
// hash-keyed fallback so any unknown topic still gets a stable glyph (same
// topic → same emoji every render). Replaces the old "#" hash marker.
// KEEPER: nav.js mirrors these two constants byte-for-byte for the shared
// sidebar on Tasks / Goals / Agent Monitor / Topics / Loops (enforced by
// tests/web/test_sidebar_topic_emoji.js) — keep them in sync.
const TOPIC_EMOJI_MAP = [
  [/\bgeneral\b/, "💬"],
  [/web ?ui|webui|\bweb\b|frontend|ui\b/, "🖥️"],
  [/health|medical|doctor/, "🩺"],
  [/docs?|research/, "📚"],
  [/telegram/, "✈️"],
  [/gpu|cuda|vllm/, "🎮"],
  [/shared ?data|database|\bdata\b/, "🗄️"],
  [/sharedagents|claw/, "🐾"],
  [/\btest|qa\b/, "🧪"],
  [/claude/, "🤖"],
  [/codex|openai/, "🧠"],
  [/voice|audio|realtime/, "🎙️"],
  [/bug|fix|error|debug/, "🐛"],
  [/deploy|infra|ops|server/, "⚙️"],
  [/news|tweet|social/, "📰"],
  [/dashboard|chart/, "📊"],
  [/cursor/, "🖱️"],
  [/self.?improv/, "🔁"],
  [/worker/, "👷"],
  [/agent/, "🛰️"],
];
const TOPIC_EMOJI_FALLBACK = ["💬", "📌", "🗂️", "🔖", "📁", "🏷️", "📨", "🧭", "🔹", "✳️", "📋", "🗨️"];
function topicEmoji(name) {
  const s = String(name || "").toLowerCase();
  for (const [re, e] of TOPIC_EMOJI_MAP) if (re.test(s)) return e;
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return TOPIC_EMOJI_FALLBACK[h % TOPIC_EMOJI_FALLBACK.length];
}

function renderTopicList() {
  const listEl = document.getElementById("topic-list");
  const sectionEl = document.getElementById("sidebar-topics");
  if (!listEl || !sectionEl) return;
  // Topics load lazily on menu expand — until then the section stays a closed,
  // tappable <details> (never auto-hidden before the first fetch).
  if (!state.topicsLoaded) return;

  // Only topics that actually have a session to open, with a real name. Hide
  // the lone "general" bucket — it carries no grouping signal and every chat
  // would land in it, so a one-row "general" list is just noise.
  const topics = (Array.isArray(state.topics) ? state.topics : []).filter(
    (t) => t && (
      (Number(t.count) > 0)
      || (t.latest_id)
      || (Array.isArray(t.sessions) && t.sessions.length > 0)
    ),
  );
  // Drop the catch-all "general" bucket — it's the ABSENCE of a topic (every
  // ungrouped chat lands there), already covered by the conversation list, so
  // a "general" row would just open the latest chat overall. A distinct named
  // topic like "General" (capitalised, a real forum topic) is kept.
  const meaningful = topics.filter(
    (t) => (t.topic || "general") !== "general",
  );
  if (!meaningful.length) {
    sectionEl.hidden = true;
    listEl.innerHTML = "";
    return;
  }
  sectionEl.hidden = false;

  const rows = meaningful.map((t) => {
    const name = t.topic || "general";
    const sessions = Array.isArray(t.sessions) ? t.sessions : [];
    const latest = sessions[0];
    const sid = t.latest_id || (latest && latest.id) || "";
    const n = Number(t.count);
    const count = Number.isFinite(n) ? n : sessions.length;
    const cur = state.sessions[state.currentSessionId];
    // Active when the open conversation belongs to this topic (not only when
    // it's the latest one), so the highlight tracks where the user actually is.
    const active = (cur && (cur.topic || "general") === name)
      || sessions.some((s) => s && s.id === state.currentSessionId);
    // Running when any conversation in the topic (or the topic aggregate) is live.
    const live = t.live || sessions.some((s) => s && s.live);
    const waiting = !!t.waiting
      || sessions.some((s) => s && s.state === "waiting")
      || t.state === "waiting";
    const livePill = waiting
      ? `<span class="live-pill waiting-pill" title="Waiting — background subagents still running">Waiting</span>`
      : live
        ? `<span class="live-pill" title="Running — a turn is in progress">Live</span>`
        : "";
    // escapeAttr (quote-safe) for attribute values; escapeHtml for text nodes.
    return `
    <div class="topic-item ${active ? "active" : ""}" data-topic-session="${escapeAttr(sid)}" title="Open latest in #${escapeAttr(name)}">
      <span class="topic-hash topic-emoji">${topicEmoji(name)}</span>
      <span class="topic-name">${escapeHtml(name)}</span>
      <span class="topic-actions">
        ${livePill}
        <span class="topic-count" title="${escapeAttr(count + " conversation" + (count === 1 ? "" : "s"))}">${escapeHtml(count)}</span>
        <button type="button" class="topic-new-chat" data-topic-new="${escapeAttr(name)}" title="New chat in #${escapeAttr(name)}" aria-label="New chat in ${escapeAttr(name)}">+</button>
      </span>
    </div>`;
  });
  listEl.innerHTML = rows.join("");
}

async function createSession(topic, opts) {
  opts = opts || {};
  if (!document.getElementById("message-input")) {
    const base =
      (document.querySelector('meta[name="sharedagents-base"]') || {}).content || "";
    window.location.href = base + "/?new=1";
    return false;
  }
  const explicit = !!opts.explicit;
  const commit = !!opts.commit;
  const intent = String(state._openSessionIntent || "").trim();
  // Empty-boot / send-path mint must not steal a deep-link that claimed the
  // tab while /api/boot was in flight. Only New Chat / ?new / ?q may steal.
  if (intent && !explicit && !commit) return false;
  if (explicit) {
    state._openSessionIntent = "";
    if (intent) state._createSessionInFlight = null;
  }

  if (!commit) {
    return startNewChatDraft(topic, opts);
  }

  if (state._createSessionInFlight) {
    return state._createSessionInFlight;
  }
  const run = _commitSession(topic, opts);
  state._createSessionInFlight = run;
  try {
    return await run;
  } finally {
    if (state._createSessionInFlight === run) state._createSessionInFlight = null;
  }
}

function refreshViewerLock() {
  // Locked because nobody had granted an agent at page load: ask again, so
  // a grant made meanwhile unlocks New Chat without a reload.
  if (!state.viewerLock || typeof apiGet !== "function") return;
  apiGet("/api/boot?session=").then(async (boot) => {
    if (!boot || !boot.viewer || boot.viewer.can_chat === false) return;
    state.viewerLock = "";
    // The roster was empty at page load: fetch the agents granted since and
    // start New Chat on one of them, not on whatever conversation was open.
    if (typeof loadAgents === "function") await loadAgents();
    if (!state.currentSessionId && typeof adoptStartablePickerAgent === "function" &&
        typeof startablePickerAgent === "function") {
      adoptStartablePickerAgent(startablePickerAgent());
    }
    if (!state.currentSessionId && state.readOnlyReason && !state.monitorReturnSessionId) {
      state.monitorReadOnly = false;
      state.readOnlyReason = "";
      applyMonitorReadOnlyUi();
    }
    if (typeof updateHeader === "function") updateHeader();
  }).catch(() => {});
}

function startNewChatDraft(topic, opts) {
  const t = typeof topic === "string" ? topic.trim() : "";
  const epoch = ++state.navEpoch;
  state.monitorReadOnly = !!state.viewerLock;
  state.readOnlyReason = state.viewerLock || "";
  state.monitorReturnSessionId = "";
  state.monitorScopeHash = "";
  applyMonitorReadOnlyUi();
  refreshViewerLock();

  state.currentSessionId = "";
  state.liveOffset = 0;
  state.history = {
    total: 0,
    tailFrom: null,
    beforeCursor: null,
    historyTruncated: false,
    tailContinuation: false,
    loading: false,
  };
  state.draftTopic = t;
  state.statusInfo = {};
  state._openSessionIntent = "";
  state._createSessionInFlight = null;

  // Detach any prior session's stream. While on a new chat draft page,
  // no stream connects and no updates shall be received or rendered.
  closeLiveStream();

  // Address bar: clear ?session= and ?new= so URL reflects the draft state.
  try {
    const _u = new URL(window.location.href);
    if (_u.searchParams.has("session") || _u.searchParams.has("new")) {
      _u.searchParams.delete("session");
      _u.searchParams.delete("new");
      const qs = _u.searchParams.toString();
      history.replaceState(null, "", _u.pathname + (qs ? "?" + qs : "") + _u.hash);
    }
  } catch (_e) {}

  // New Chat must never inherit a delegation-only agent left in the picker.
  const agent = (typeof startablePickerAgent === "function"
    ? startablePickerAgent() : selectedPickerAgent()) || "jarvis";
  if (typeof adoptStartablePickerAgent === "function")
    adoptStartablePickerAgent(agent);
  else if (typeof selectPickerAgent === "function")
    selectPickerAgent(agent);

  const posted = pickerModelAlias(agent)
    || (typeof catalogueDefaultAlias === "function" ? catalogueDefaultAlias(agent) : "")
    || "";
  state.currentModel = posted;
  if (posted) {
    state.pickerModels = state.pickerModels || {};
    state.pickerModels[agent] = posted;
  }
  if (typeof modelSheetSelectionFor === "function") {
    if (typeof ensureModelSheetState === "function") ensureModelSheetState();
    state.modelSheet.effective = modelSheetSelectionFor(agent, posted);
    if (typeof cloneModelSelection === "function")
      state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
  }
  // Catalogue is lazy on sheet-open. New Chat still needs the default id
  // on `.seg-model` and on the later POST /api/sessions body. A draft the
  // boot opened (nobody clicked) resolves it without a request.
  const passive = !!(opts && opts.passive);
  if (typeof ensurePickerDefaultModel === "function")
    Promise.resolve(ensurePickerDefaultModel(agent, { passive })).catch(function () {});

  state.todos = [];
  state.todosSid = "";
  state.harnessTasks = [];
  state.harnessTasksSid = "";
  state.subagents = [];
  // New Chat must drop the previous conversation's goal row too. Neither
  // path goes through openSession's `prevSid !== sid` guard here, and the
  // self-heal is blocked because applyGoalDetail/refreshGoalCost bail when
  // state.goalSid no longer matches the session.
  if (typeof resetGoalTracker === "function") resetGoalTracker();
  state.treeLive = false;
  state.engineModel = "";
  state.engineModelAlias = "";
  state.lastContextUsed = 0;
  state.lastContextMax = 0;

  renderSessionList();
  clearMessages();
  updateHeader();
  refreshSessionWidgets();
  if (t && !passive && typeof loadTopicList === "function") loadTopicList();
  if (typeof messageInput !== "undefined" && messageInput) {
    messageInput.value = "";
    messageInput.focus();
  }
  if (typeof clearStagedAttachments === "function") clearStagedAttachments();
  if (typeof updateSendState === "function") updateSendState();

  return true;
}

async function _commitSession(topic, opts) {
  opts = opts || {};
  const t = typeof topic === "string" ? topic.trim() : (state.draftTopic || "");
  const epoch = ++state.navEpoch;
  state.monitorReadOnly = !!state.viewerLock;
  state.readOnlyReason = state.viewerLock || "";
  state.monitorReturnSessionId = "";
  state.monitorScopeHash = "";
  applyMonitorReadOnlyUi();

  let agent = opts.agent || (typeof startablePickerAgent === "function"
    ? startablePickerAgent() : selectedPickerAgent()) || "jarvis";
  const priorAgent = state.currentAgent || state.pickerAgent || "";
  const switchingAgent = !!priorAgent && agent !== priorAgent;
  if (switchingAgent) state.modePickerTouched = false;
  if (typeof adoptStartablePickerAgent === "function")
    adoptStartablePickerAgent(agent);
  else if (typeof selectPickerAgent === "function")
    selectPickerAgent(agent);

  let model = opts.model || pickerModelAlias(agent) || "";
  if (!model && typeof ensurePickerDefaultModel === "function") {
    try { model = await ensurePickerDefaultModel(agent) || ""; }
    catch (_e) { model = ""; }
  }
  if (!model && typeof catalogueDefaultAlias === "function")
    model = catalogueDefaultAlias(agent) || "";
  const body = {
    agent,
    model,
    reserve: false,
  };
  if (t) body.topic = t;

  const data = await apiPost("/api/sessions", body);
  const session = data.session;
  if (epoch !== state.navEpoch) return false;

  for (const id of (data.discarded || [])) {
    if (id && id !== session.id) delete state.sessions[id];
  }
  state.sessions[session.id] = session;
  state.currentSessionId = session.id;
  state.draftTopic = "";
  state.statusInfo = {};
  // The server may start the chat on another agent than the picked one (a
  // person's first granted agent): the composer follows it.
  const served = String((session && session.agent) || "").trim();
  if (served && served !== agent) {
    agent = served;
    if (typeof adoptStartablePickerAgent === "function") adoptStartablePickerAgent(served);
    state.currentAgent = served;
  }

  const optModeEl = document.getElementById("opt-mode");
  const sameAgent = !switchingAgent;
  const pickerMode = sameAgent && opts.mode !== undefined
    ? opts.mode
    : (sameAgent && state.modePickerTouched
      ? ((optModeEl && optModeEl.value) || "")
      : "");
  if (!sameAgent) state.modePickerTouched = false;
  applyFreshAgentModeIntent(session.id, agent, pickerMode);
  // Keep the model we just posted on the unused row. Wiping currentModel /
  // pickerModels here left the composer chip nameless on New Chat while the
  // Agents footer still showed store.model (qwen3.8-27b-fp8).
  const posted = String((session && session.model) || "").trim();
  state.currentModel = posted;
  if (posted) {
    state.pickerModels = state.pickerModels || {};
    state.pickerModels[agent] = posted;
  }
  if (typeof modelSheetSelectionFor === "function") {
    if (typeof ensureModelSheetState === "function") ensureModelSheetState();
    state.modelSheet.effective = modelSheetSelectionFor(agent, posted);
    if (typeof cloneModelSelection === "function")
      state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
  }
  persistLastSession(session.id); // remember across browser restarts

  try {
    const _u = new URL(window.location.href);
    if (_u.searchParams.get("session") !== String(session.id)) {
      _u.searchParams.set("session", String(session.id));
      history.replaceState(null, "", _u.pathname + _u.search + _u.hash);
    }
  } catch (_e) {}

  state.todos = [];
  state.todosSid = "";
  state.harnessTasks = [];
  state.harnessTasksSid = "";
  state.subagents = [];
  // _commitSession sets state.currentSessionId directly (bypassing
  // openSession's goal reset) — the committed session must not inherit the
  // previous conversation's goal row.
  if (typeof resetGoalTracker === "function") resetGoalTracker();
  state.treeLive = false;
  state.engineModel = "";
  state.engineModelAlias = "";
  state.lastContextUsed = 0;
  state.lastContextMax = 0;

  renderSessionList();
  openLiveStream(session.id, 0);
  updateHeader();
  refreshSessionWidgets();
  if (t && typeof loadTopicList === "function") loadTopicList();

  return session || true;
}

async function loadSession(sid, opts = {}) {
  // Claim a fresh nav epoch up-front so any in-flight loader from a previous
  // session/agent bails, and so OUR own GET can detect being superseded by a
  // later click before it applies (no wrong-session paint).
  const epoch = ++state.navEpoch;
  // FAIL OPEN on the read-only lock. It is set from the response at the bottom
  // of this function, AFTER the epoch guard — so a load that is superseded (a
  // second click) or that throws leaves the PREVIOUS session's lock in place,
  // and the operator lands in their OWN conversation with the composer stuck
  // on "Read-only subagent session", unable to type or stop anything.
  //
  // The server had it right all along: it reports monitor_read_only=false for
  // these sessions. The lock was purely stale client state. Clearing it up
  // front means the worst case is a brief unlocked composer on a genuinely
  // read-only session, instead of a permanently locked one on your own.
  state.monitorReadOnly = !!state.viewerLock;
  state.loadingSessionId = sid;
  const progress = typeof document !== "undefined" ? document.getElementById("turn-progress") : null;
  if (progress) {
    progress.classList.add("running");
    progress.removeAttribute("hidden");
  }
  if (typeof renderSessionList === "function") renderSessionList();
  const monitorNavigation = opts.monitorNavigation || null;
  if (!monitorNavigation) {
    state.monitorReturnSessionId = "";
    state.monitorScopeHash = "";
  }
  state.treeLive = false;
  if (state.scroll) {
    state.scroll.olderUnlocked = false;
    state.scroll.fillViewport = false;
    state.scroll.viewportFillHops = 0;
  }
  try {
    // Monitor navigation is UI context, not a competing history source. The
    // normal endpoint prefers the canonical logical transcript (same as the
    // full monitor tab) and returns that session's payload.
    const data = await apiGet(`/api/boot?session=${encodeURIComponent(sid)}`);
    if (epoch !== state.navEpoch) return; // a newer load/switch superseded this one
    if (data && data.session) {
      openSession(data.session, { epoch, monitorNavigation, boot: data });
      return;
    }
    // Boot can omit a resolvable sid (ownership miss, aux timeout). Do not
    // leave the chat empty with no EventSource — fall back to the session GET.
    const fallback = await apiGet(`/api/sessions/${encodeURIComponent(sid)}`);
    if (epoch !== state.navEpoch) return;
    if (fallback && fallback.session) {
      openSession(fallback.session, { epoch, monitorNavigation, boot: data || {} });
      return;
    }
    throw new Error("Session not found");
  } catch (e) {
    console.error("Failed to load session:", e);
    appendSystemMessage(
      `Could not open this session: ${escapeHtml(e.message || "access denied")}`,
      true,
    );
  } finally {
    if (state.loadingSessionId === sid) {
      state.loadingSessionId = null;
      const progress = typeof document !== "undefined" ? document.getElementById("turn-progress") : null;
      if (progress && !state.isStreaming) {
        progress.classList.remove("running");
        progress.setAttribute("hidden", "");
      }
      if (typeof renderSessionList === "function") renderSessionList();
    }
  }
}

function openSession(session, opts = {}) {
  // Applies a session already fetched by loadSession() (GET /api/boot) or
  // created by createSession() (POST). The page opens with a single boot read.
  const epoch = opts.epoch || 0;
  const monitorNavigation = opts.monitorNavigation || null;
  const boot = opts.boot || {};
  if (!session) return;
  const sid = session.id || "";
  const prevSid = state.currentSessionId;

  state._openSessionIntent = sid;
  state.currentSessionId = sid;
  if (prevSid !== sid) {
    state.liveOffset = 0;
    resetGoalTracker();
    // The footer falls back to this spend when the new session has no budget
    // snapshot yet; the previous conversation's total must not show there.
    state.sessionCostUsd = null;
    state.goalDismissedAfterMet = false;
    state.goalTextExpanded = false;
    // Session-scoped widget rows must not leak across a switch: `state.todos`
    // / `state.harnessTasks` are global, and the previous conversation's rows
    // would otherwise survive the whole visit — the new session's first state
    // frame is authoritative for it, but the _applyState guard only protects
    // the ACTIVE session's live rows, so a stale session's rows outlived an
    // empty (valid) snapshot from a subagent that never wrote .todos.json.
    state.todos = [];
    state.todosSid = "";
    state.harnessTasks = [];
    state.harnessTasksSid = "";
    // Repaint the harness widget SYNCHRONOUSLY from the just-cleared arrays.
    // The post-switch refreshSessionWidgets() below is throttled (500 ms) and
    // may be discarded without rescheduling — without this repaint the
    // previous session's rendered chip stays visible until the new session's
    // first state frame arrives. renderHarnessWidget is a pure repaint of
    // state.todos/state.harnessTasks (the only readers of those arrays) — no
    // fetch, and deliberately NOT routed through the throttled refresh.
    if (typeof renderHarnessWidget === "function") renderHarnessWidget();
  }
  // A conversation the server lets this person chat in proves they were
  // granted an agent since the page loaded: the boot-time lock is stale.
  if (session.access === "chat" || session.access === "manage") state.viewerLock = "";
  state.monitorReadOnly = !!session.monitor_read_only || !!state.viewerLock;
  state.readOnlyReason = String(session.read_only_reason || state.viewerLock || "");
  _applyBootApprovals(boot);
  // Never let the previous conversation's effort snapshot seed this one's
  // slider while the fresh /status request is in flight.
  state.statusInfo = {};
  // INLINE (not a helper call) so the URL stays copyable — tests/web slices this
  // function out of the file and evals it standalone, where a sibling helper is
  // a ReferenceError. Same constraint as the context-meter assignments below.
  // Clicking a session must leave a copyable URL.
  try {
    const _u = new URL(window.location.href);
    const _monitorParam = monitorNavigation ? "1" : "";
    if (
      _u.searchParams.get("session") !== String(state.currentSessionId)
      || (_u.searchParams.get("monitor") || "") !== _monitorParam
    ) {
      _u.searchParams.set("session", String(state.currentSessionId));
      if (_monitorParam) _u.searchParams.set("monitor", _monitorParam);
      else _u.searchParams.delete("monitor");
      history.replaceState(null, "", _u.pathname + _u.search + _u.hash);
    }
  } catch (_e) {}
  if (monitorNavigation) {
    state.monitorReturnSessionId = String(
      monitorNavigation.returnSessionId || "",
    );
    state.monitorScopeHash = String(monitorNavigation.scopeHash || "");
  }
  applyMonitorReadOnlyUi();
  persistLastSession(state.currentSessionId); // remember across browser restarts
  // Register/refresh the conversation in the sidebar — covers gateway-native
  // sessions (TUI / monitor deep-link) that weren't in the list yet, so they
  // appear and stay selectable. Keyed by the session's real id.
  state.sessions[state.currentSessionId] = session;
  if (window.ShareControl) window.ShareControl.sync();
  // Open-session agent is the only chrome source. A leftover parent
  // footer ``current`` (jarvis waiting on a child) must not steal the
  // picker — stamp the footer from this agent instead.
  state.currentAgent = session.agent || "jarvis";
  // An existing session's server-owned agent is authoritative.  The picker
  // preference is global browser state, so leaving it untouched here let a
  // stale "jarvis" preference override a loaded Codex session in both the
  // selector and footer after refresh.
  state.pickerAgent = state.currentAgent;
  state.sessionRevision = state.monitorReadOnly
    ? 0
    : Number(session.revision) || 0;
  if (state.monitorReadOnly) {
    if (
      typeof isNonConversationalAgent === "function" &&
      isNonConversationalAgent(state.currentAgent)
    ) {
      state.effectiveMode = {
        id: "auto",
        label: "Allow all",
        behavior_kind: "profile_build",
      };
    } else {
      state.effectiveMode = {
        id: "monitor",
        label: "Read only",
        behavior_kind: "monitor",
      };
    }
    state.availableAgentModes = {};
  } else if (session.effective_mode || session.effective_mode_id) {
    state.effectiveMode = normalizeEffectiveMode(
      session.effective_mode || {
        id: session.effective_mode_id,
        label: session.effective_mode_label,
        behavior_kind: session.behavior_kind,
      },
    );
  }
  state.currentModel = session.model || "";
  if (typeof isForeignAgentModel === "function" &&
      isForeignAgentModel(state.currentAgent, state.currentModel))
    state.currentModel = "";
  state.engineModel = session.engine_model || ""; // restore the real-model badge
  state.engineModelAlias = session.engine_model_alias || ""; // badge alias
  if (typeof isForeignAgentModel === "function") {
    if (isForeignAgentModel(state.currentAgent, state.engineModel))
      state.engineModel = "";
    if (isForeignAgentModel(state.currentAgent, state.engineModelAlias))
      state.engineModelAlias = "";
  }
  if (typeof syncModelSheetEffectiveFromSession === "function")
    syncModelSheetEffectiveFromSession();
  // Seed the footer's context "used / max" segment from the session header so
  // it shows the moment a session opens. Live round/header events refine these
  // values during a turn; the header remains the terminal backstop. Kept
  // INLINE (not factored into applyContextHeader) because tests/web slices
  // this function out of the file and evals it standalone: a call to a
  // sibling helper is a ReferenceError in that harness.
  if (typeof session.context_window === "number" && session.context_window > 0)
    state.lastContextMax = session.context_window;
  if (typeof session.last_prompt_tokens === "number" && session.last_prompt_tokens > 0)
    state.lastContextUsed = session.last_prompt_tokens;
  // Seed the compaction-stats state from the session header so the chat-area
  // compaction turn card can render the moment a session opens. INLINE (not
  // factored into applyContextHeader) for the same tests/web-slicing reason
  // as the context-window seed above.
  if (session.compaction_last_stats &&
      typeof session.compaction_last_stats === "object" &&
      !Array.isArray(session.compaction_last_stats))
    state.compactionLastStats = session.compaction_last_stats;
  if (typeof session.compaction_summary === "string")
    state.compactionSummary = session.compaction_summary;
  if (typeof session.compaction_count === "number")
    state.compactionCount = session.compaction_count;
  if (state.monitorReadOnly) {
    state.commands = [];
  } else if (state._commandsAgent && state._commandsAgent !== state.currentAgent) {
    state.commands = null;
  }
  // Slash palette is loaded on first `/` or the commands button — not on
  // every existing-chat enter (that was a second 8 kB GET /api/commands).
  renderSessionList();
  renderTopicList(); // move the topic highlight to the now-open conversation
  renderAgentSelect(); // reflect the session's agent in the sidebar
  updateHeader();
  // Render the footer synchronously so a session-load to a
  // different-agent session shows the new agent immediately — refreshSessionWidgets
  // below is async and would otherwise leave the footer on the old agent briefly.
  renderStatusSegments();
  if (!state.monitorReadOnly) {
    seedSelectedModelOnly(); // selected alias only; catalogue loads on sheet open
    // Teacher comes from the one post-bulk /status. A /teacher command
    // here was a fifth POST on every existing-chat enter.
  } else {
    state.models = [];
    state.statusInfo = {};
    state.todos = [];
    state.todosSid = "";
    state.harnessTasks = [];
    state.harnessTasksSid = "";
    state.subagents = [];
    state.queue = [];
    state.steeringQueueId = "";
    state.teacherActive = false;
    state.teacherEnabled = false;
    state.budget = null;
    renderModelSelect();
    renderSessionStatusBar();
  }
  renderGoalWidget(); // show the goal bar for this session
  // Populate goal from session data or boot if available
  const initialGoal = session.goal || (opts && opts.boot && opts.boot.goal);
  if (initialGoal && typeof applyGoalDetail === "function") {
    applyGoalProgress({ goal_id: initialGoal.goal_id || initialGoal.id || "" });
    applyGoalDetail(initialGoal, sid);
  } else if (session.goal_id) {
    applyGoalProgress({ goal_id: session.goal_id });
  }
  const initialBudget = session.budget || (opts && opts.boot && opts.boot.budget);
  const initialGoalForBudget = initialGoal || session.goal || (opts && opts.boot && opts.boot.goal);
  let stampedBudget = initialBudget;
  if (typeof stampBudgetFromGoal === "function") {
    stampedBudget = stampBudgetFromGoal(initialBudget, initialGoalForBudget);
  } else if (initialBudget && initialGoalForBudget) {
    const spent = Number((initialBudget.total && initialBudget.total.spent_usd) || 0);
    const goalCost = Number(initialGoalForBudget.cost_usd);
    if (spent === 0 && Number.isFinite(goalCost) && goalCost > 0) {
      stampedBudget = Object.assign({}, initialBudget, {
        total: Object.assign({}, initialBudget.total || {}, { spent_usd: goalCost }),
      });
    }
  }
  if (stampedBudget) {
    state.budget = stampedBudget.total ? stampedBudget : null;
    if (typeof renderStatusSegments === "function") renderStatusSegments();
  }
  // Hydrate Agent Monitor footer from pre-computed lineage in boot/session.
  // Set chrome (`cur`) first so a stale parent footer cannot paint as current.
  if (typeof updateAgentMonAccordion === "function") updateAgentMonAccordion();
  const initialFooter = session.footer || (opts && opts.boot && opts.boot.footer);
  if (initialFooter && window.SaAgentMon && typeof window.SaAgentMon.applySnapshot === "function") {
    window.SaAgentMon.applySnapshot(initialFooter);
  }
  state.history = {
    total: session.history_total || 0,
    tailFrom: Number.isFinite(session.tail_from_offset) ? session.tail_from_offset : null,
    beforeCursor:
      Number.isFinite(session.before_cursor)
        ? session.before_cursor
        : null,
    historyTruncated: !!session.history_truncated,
    tailContinuation: !!session.tail_continuation,
    loading: false,
  };
  // Kill smooth scrolling for the entire load: the container has CSS
  // `scroll-behavior: smooth`, so ANY pin (renderMessages' scrollToBottom, the
  // settle loop) would ANIMATE from the top down — the "page scrolls down"
  // you see. _teardownBulk restores it for live following afterwards.
  if (messagesContainer) messagesContainer.style.scrollBehavior = "auto";
  const hasTail = Number.isFinite(session.tail_from_offset);
  const sessState = String(session.state || session.session_state || "");
  const compacting = sessState === "compacting";
  // A JSONL unmatched turn_started can keep session.active true after an
  // error/abort finalize (session_state already idle/error). Do not light
  // the warming / Stop UI unless the durable parent turn is still live.
  const turnLive = !!session.active && (
    typeof _parentTurnIsLive === "function"
      ? _parentTurnIsLive(sessState) || sessState === "ask"
      : ["prompting", "prefill", "reasoning", "streaming", "tool_running",
         "compacting", "paused", "ask", "retrying"]
          .includes(sessState)
  );
  if (hasTail) {
    // REST carries lazy one-liner shells (incl. the open turn). SSE tails
    // from EOF and delivers only future canonical events.
    renderMessages(session.messages || []);
    VirtualScroller.beginBulkLoad();
    const ownLive = !!(
      session.own_live
      || (session.activity && session.activity.own_live)
      || (typeof _parentTurnIsLive === "function" && _parentTurnIsLive(sessState))
    );
    if (turnLive || compacting || ownLive) {
      const phase = compacting ? "compacting"
        : sessState === "tool_running" ? "tool"
        : sessState === "retrying" ? "retrying"
        : sessState;
      _applyActiveTurnUI(
        (phase && phase !== "warming") ? { phase: phase } : undefined,
      );
    } else if (
      session.state === "waiting"
      || session.session_state === "waiting"
      || (session.tree_busy && !ownLive)
    ) {
      if (typeof _setTreeLive === "function") _setTreeLive(true);
    }
    // Only join an in-progress REST shell. Adopting a completed closer on
    // idle reload glues the next turn's SSE into the previous card.
    if (turnLive && typeof adoptInFlightAssistant === "function") {
      adoptInFlightAssistant();
    }
    openLiveStream(state.currentSessionId, session.tail_from_offset);
  } else {
    // Web-only session with no gateway log: paint the stored transcript
    // (a Dev conductor seed is one system line; skip it and the welcome
    // screen claims this is a brand-new chat).
    renderMessages(session.messages || []);
    VirtualScroller.beginBulkLoad();
    if (compacting && typeof _applyActiveTurnUI === "function") {
      _applyActiveTurnUI({ phase: "compacting" });
    }
    closeLiveStream();
    // Always attach the conversation EventSource. Unbound New Chat hits the
    // local control-frame stream (ready/reconnect) until the first prompt
    // binds a gateway log — skipping it left some sessions with no SSE and
    // an empty chat. Painted rows need a concrete resume cursor; empty rows
    // can start at 0. First send reuses a healthy stream.
    openLiveStream(state.currentSessionId, 0);
  }
  if (compacting && typeof restoreCompactWidgetFromSession === "function") {
    restoreCompactWidgetFromSession(session);
  }
  // Reconcile session state + capabilities from the single boot read.
  if (!state.monitorReadOnly) {
    if (boot.session_state && typeof applySessionState === "function")
      applySessionState(boot.session_state, { force: true });
    if (boot.agent_capabilities && typeof installAgentCapabilities === "function") {
      installAgentCapabilities(boot.agent_capabilities);
      renderAgentModeSelect(state.currentAgent);
    }
    // Production: finishBulkLoad() runs the one widget refresh. Tests that
    // stub VirtualScroller without that hook still need the footer populated.
    if (
      typeof VirtualScroller === "undefined" ||
      typeof VirtualScroller._refreshWidgetsAfterBulk !== "function"
    ) {
      refreshSessionWidgets();
    }
  }
  _hydratePendingInteractiveCards();
  applyMonitorReadOnlyUi();
  }

function _historyMessageFingerprint(m) {
  if (!m) return "";
  const c = m.content;
  let text = "";
  if (typeof c === "string") text = c;
  else if (c && typeof c === "object") {
    if (typeof c.prompt === "string") text = c.prompt;
    else if (typeof c.text === "string") text = c.text;
    else {
      try {
        text = JSON.stringify(c);
      } catch (_e) {
        text = "";
      }
    }
  }
  return String(m.role || "") + "\0" + text;
}

/** Messages from an older page that are not already a prefix of the cache.
 *  If the whole page overlaps the visible tail, the prefix is empty. */
function _uniqueOlderPrefix(older) {
  const cache =
    typeof state !== "undefined" && Array.isArray(state.msgCache)
      ? state.msgCache
      : [];
  if (!older.length) return older;
  if (!cache.length) return older;
  const olderFp = older.map(_historyMessageFingerprint);
  const cacheFp = cache.map(_historyMessageFingerprint);
  let overlap = Math.min(olderFp.length, cacheFp.length);
  while (overlap > 0) {
    let match = true;
    for (let i = 0; i < overlap; i += 1) {
      if (olderFp[olderFp.length - overlap + i] !== cacheFp[i]) {
        match = false;
        break;
      }
    }
    if (match) break;
    overlap -= 1;
  }
  return overlap ? older.slice(0, older.length - overlap) : older;
}

// Fetch the next OLDER page of history and prepend it to the scroller. Triggered
// on scroll-up when the top sentinel is reached and an older page exists.
async function loadOlderPage() {
  const h = state.history;
  if (!h || h.loading || h.beforeCursor == null) return;
  const sid = state.currentSessionId;
  if (!sid) return;
  const epoch = state.navEpoch; // guard against a same-sid reload race (Codex #5)
  const maxEmptyHops = 8;
  const seenCursors = new Set();
  let cursor = h.beforeCursor;
  let prepended = false;
  h.loading = true;
  try {
    // A bounded reverse scan can encounter one or more giant/noncanonical
    // records without completing a turn. Those pages legitimately contain no
    // messages but advertise a strictly older continuation cursor. Follow a
    // small bounded chain in this SAME sentinel interaction; otherwise the
    // observer may never fire again and all earlier history becomes stranded.
    for (let hop = 0; hop < maxEmptyHops && cursor != null; hop += 1) {
      if (seenCursors.has(cursor)) {
        console.warn("Stopping older-history pagination at repeated cursor", cursor);
        h.beforeCursor = null;
        break;
      }
      seenCursors.add(cursor);
      const data = await apiGet(
        `/api/sessions/${sid}?before=${cursor}`,
      );
      // Bail if the user switched sessions OR reopened the same one while we waited.
      if (state.currentSessionId !== sid || state.navEpoch !== epoch) return;
      const session = data.session || {};
      const older = Array.isArray(session.messages) ? session.messages : [];
      const nextCursor =
        Number.isFinite(session.before_cursor)
          ? session.before_cursor
          : null;
      h.total = session.history_total || h.total;
      h.historyTruncated = !!session.history_truncated;
      h.tailContinuation = !!session.tail_continuation;
      if (older.length) {
        // The first ?before= after open can repeat the latest page (same
        // messages, older byte cursor). Prepending that doubles the tail.
        // Drop any suffix that already sits at the start of the cache.
        const unique = _uniqueOlderPrefix(older);
        if (!unique.length) {
          if (nextCursor == null) {
            h.beforeCursor = null;
            break;
          }
          if (nextCursor >= cursor || seenCursors.has(nextCursor)) {
            console.warn(
              "Stopping older-history pagination at overlapping nondecreasing cursor",
              cursor,
              nextCursor,
            );
            h.beforeCursor = null;
            break;
          }
          h.beforeCursor = nextCursor;
          cursor = nextCursor;
          continue;
        }
        VirtualScroller.prependOlder(unique);
        prepended = true;
        h.beforeCursor = nextCursor;
        break;
      }
      if (nextCursor == null) {
        h.beforeCursor = null;
        break;
      }
      if (nextCursor >= cursor || seenCursors.has(nextCursor)) {
        console.warn(
          "Stopping older-history pagination at nondecreasing cursor",
          cursor,
          nextCursor,
        );
        h.beforeCursor = null;
        break;
      }
      h.beforeCursor = nextCursor;
      cursor = nextCursor;
    }
  } catch (e) {
    console.error("Failed to load older history:", e);
  } finally {
    h.loading = false;
    if (
      typeof VirtualScroller !== "undefined" &&
      typeof VirtualScroller.onOlderPageDone === "function"
    ) {
      VirtualScroller.onOlderPageDone(prepended);
    } else if (
      typeof VirtualScroller !== "undefined" &&
      typeof VirtualScroller.ensureReachableHistory === "function"
    ) {
      VirtualScroller.ensureReachableHistory();
    }
  }
}

// ── Realtime conversation stream ───────────────────────────────────────
// The OPEN conversation subscribes to the gateway session log's canonical
// incremental event stream (mode=events), so a turn the user starts here OR
// continues in the TUI (or any other frontend) renders in realtime as the log
// is written. Only the selected conversation streams — switching away closes
// it — so the web stays light. live == replay == observe: from=0 replays the
// whole session, then tails.

// The stream's ONE control frame (`event: state`) consolidates everything the
// old eight-frame multiplexer pushed: active-session header, live activity,
// health, approvals, sidebar pills, budget, goal, Agent-Monitor tree, widgets.
function _menuSectionOpen(id) {
  if (window.MenuState && typeof MenuState.sectionOpen === "function") {
    return MenuState.sectionOpen(id);
  }
  const el = document.getElementById(id);
  return !!el && el.open;
}

function mergeSessionList(rows) {
  if (!Array.isArray(rows)) return;
  let changed = false;
  for (const fresh of rows) {
    if (!fresh || !fresh.id) continue;
    const current = state.sessions[fresh.id];
    if (!current) {
      state.sessions[fresh.id] = fresh;
      changed = true;
      continue;
    }
    const merged = (typeof mergeSessionListRow === "function")
      ? mergeSessionListRow(current, fresh)
      : Object.assign({}, current, fresh);
    for (const key of [
      "title", "updated_at", "agent", "origin", "live", "state",
      "session_summary", "model", "activity", "version", "phase",
      "attention", "lifecycle", "last_outcome", "own_live", "tree_busy",
      "stale", "active_descendant_count", "open_job_count",
      "phase_changed_at", "heartbeat_at", "outcome_at", "status",
      "display", "capabilities",
    ]) {
      if (key === "session_summary" && merged[key] == null) continue;
      if (current[key] !== merged[key]) {
        current[key] = merged[key];
        changed = true;
      }
    }
  }
  if (changed) {
    renderSessionList();
    if (typeof updateHeader === "function") updateHeader();
  }
}

function _applyDiscoveredModels(agent, rawList) {
  if (!agent || !Array.isArray(rawList)) return;
  // The same row mapping and default rule as every other model source.
  const models = normalizeModelRows(agent, rawList);
  // An empty frame is a failed discovery, never a reason to drop a good list.
  const cached = state.modelOptionsByAgent[agent];
  if (!models.length && Array.isArray(cached) && cached.length) return;
  // The frame is the saved catalogue of every provider: it replaces the rows
  // of each provider it lists and keeps any other row (a default model the
  // picker summary named that no catalogue lists yet).
  storeAgentModelRows(agent, models, [...new Set(models.map((m) => m.provider))]);
  // Live frames carry saved catalogues, so the list is neither degraded nor
  // partial any more.
  if (state.modelCatalogueDegraded) delete state.modelCatalogueDegraded[agent];
  state.modelCatalogueFull = state.modelCatalogueFull || {};
  state.modelCatalogueFull[agent] = true;
  if (typeof refreshModelCataloguePartial === "function") refreshModelCataloguePartial(agent);
  if (typeof selectedPickerAgent === "function" && selectedPickerAgent() === agent) {
    if (typeof renderModelSheet === "function") renderModelSheet();
    if (typeof renderModelSelect === "function") renderModelSelect();
  }
}

// Deduplicated catalogue frame: `catalogues` carries each DISTINCT model list
// ONCE under its content digest, and `agents` maps agent -> digest. The native
// catalogue is shared by every native agent, so sending it per agent repeated
// the same 481-entry list six times (~851 KB per state tick).
function _applyCatalogue(frame) {
  if (!frame || typeof frame !== "object") return;
  const lists = frame.catalogues;
  const byAgent = frame.agents;
  if (!lists || typeof lists !== "object" || !byAgent || typeof byAgent !== "object") return;
  for (const [agent, digest] of Object.entries(byAgent)) {
    const list = lists[digest];
    if (Array.isArray(list)) _applyDiscoveredModels(agent, list);
  }
}

// Discovered models ride the `models` frame in two shapes: a single-agent delta
// (`{agent, models}`, broadcast by POST /api/models/discover) or the full
// deduplicated catalogue (sent once on connect + whenever it changes). Accept
// both shapes so the picker has ONE entry point.
function _applyModelsFrame(frame) {
  if (!frame || typeof frame !== "object") return;
  if (frame.agent && Array.isArray(frame.models)) {
    _applyDiscoveredModels(frame.agent, frame.models);
    return;
  }
  _applyCatalogue(frame);
}

function _applyRemoteAgentFromState(remoteAgent, extra) {
  if (!state.currentSessionId) return false;
  // Other-browser / other-tab agent switch. The 4s `event: state` tick (and
  // notify_state_changed wake) already carries session.agent, but the picker
  // chip reads modelSheet.effective / pickerAgent / currentAgent — none of
  // those were updated, so prod tabs stayed on the old agent. Do not steal
  // the chip while this tab is mid-switch.
  const want = String(remoteAgent || "").trim();
  if (!want) return false;
  if (state.switchPending || state.pendingAgent) return false;
  // Unused New Chat: the picker is local intent. A stale `event: state`
  // still carrying the previous agent (cursor while the chip shows build)
  // must not steal the footer or the composer.
  if (
    typeof isUnusedWebPlaceholder === "function"
    && isUnusedWebPlaceholder(state.currentSessionId)
  ) {
    const local = (
      (typeof selectedPickerAgent === "function" && selectedPickerAgent())
      || state.pickerAgent
      || ""
    );
    if (local && want !== local) return false;
  }
  const shown = (
    (state.modelSheet && state.modelSheet.effective && state.modelSheet.effective.agent)
    || state.pickerAgent
    || state.currentAgent
    || ""
  );
  if (want === shown && want === (state.currentAgent || "")) return false;
  if (typeof applySessionState === "function") {
    applySessionState(Object.assign({
      active_agent: want,
      agent: want,
    }, extra || {}), { force: true });
  } else {
    state.currentAgent = want;
  }
  if (typeof adoptStartablePickerAgent === "function") {
    adoptStartablePickerAgent(want);
  } else {
    state.pickerAgent = want;
  }
  if (typeof renderAgentSelect === "function") renderAgentSelect();
  if (typeof renderStatusSegments === "function") renderStatusSegments();
  if (typeof updateAgentMonAccordion === "function") updateAgentMonAccordion();
  return true;
}

function _footerCurrentModel(payload) {
  const footer = (payload && payload.tree && (payload.tree.footer || payload.tree)) || {};
  const rows = footer.rows || [];
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    if (row && row.role === "current" && row.model) return String(row.model);
  }
  if (footer.current && footer.current.model) return String(footer.current.model);
  const widgets = payload && payload.widgets && payload.widgets.status;
  if (widgets && widgets.model) return String(widgets.model);
  return "";
}

function _promoteLiveSessionModel(s, fallbackModel) {
  if (!state.currentSessionId) return;
  const model = String((s && s.model) || fallbackModel || "").trim();
  const engine = String((s && s.engine_model) || fallbackModel || "").trim();
  const alias = String((s && s.engine_model_alias) || "").trim();
  const agent = state.currentAgent || (s && s.agent) || "";
  if (s && s.agent && state.currentAgent && String(s.agent).trim() !== String(state.currentAgent).trim()) {
    return;
  }
  const foreign = (id) =>
    (typeof isLocalLlmAgent === "function" && isLocalLlmAgent(agent) && typeof isModeSentinel === "function" && isModeSentinel(id)) ||
    (typeof isForeignAgentModel === "function" && isForeignAgentModel(agent, id));
  if (model && !foreign(model)) state.currentModel = model;
  if (engine && !foreign(engine)) {
    state.engineModel = engine;
    state.engineModelAlias = alias && !foreign(alias) ? alias : "";
  } else if (alias && !foreign(alias)) {
    state.engineModelAlias = alias;
    if (!state.engineModel) state.engineModel = alias;
  }
  if (typeof syncModelSheetEffectiveFromSession === "function")
    syncModelSheetEffectiveFromSession();
}

function _applyState(payload) {
  if (!payload || payload.ok === false) return;
  const sid = state.currentSessionId;
  if (!sid) return;
  if (String(payload.session_id || "") !== String(sid || "")) return;
  // A state tick may have been assembled before the terminal log frame that
  // just closed the turn. Never let that older projection repaint the UI.
  if (typeof _isStaleLiveSnapshot === "function" &&
      _isStaleLiveSnapshot(sid, payload.live)) return;

  if (Array.isArray(payload.delegations)) {
    state.delegationSnapshot = { sessionId: sid, jobs: payload.delegations };
    if (typeof _applyDelegationUpdates === "function")
      _applyDelegationUpdates(payload.delegations);
  }

  // Legacy in-state model field. Models now ride their own `models` frame
  // (catalogue on connect + deltas), but a not-yet-restarted gunicorn worker
  // can still serve the old shape, so keep resolving it.
  if (payload.models_by_agent && typeof payload.models_by_agent === "object") {
    for (const [agent, list] of Object.entries(payload.models_by_agent)) {
      if (Array.isArray(list)) {
        _applyDiscoveredModels(agent, list);
      }
    }
  }

  // Active session header → title, agent/model badge, stop button, sidebar row.
  const s = payload.session;
  if (s && sid) {
    const row = state.sessions[sid] || (state.sessions[sid] = s);
    // A live round/header event is the freshest context reading. The state
    // snapshot is assembled from the session index, which can trail the
    // append-only event log while a tool-driven turn is still running. Do not
    // let that stale projection erase a value the stream has already painted.
    // Once the turn is idle, the normal header reconciliation below remains
    // the terminal backstop for a missed/reconnected event.
    const preserveLiveContext = !!(s.live || state.isStreaming) && (
      Number(state.lastContextUsed) > 0 || Number(state.lastContextMax) > 0
    );
    for (const key of ["title", "agent", "model", "engine_model",
      "engine_model_alias", "updated_at", "session_summary", "revision", "state",
      "context_window", "last_prompt_tokens", "compaction_last_stats",
      "compaction_summary", "compaction_count"]) {
      if (s[key] == null) continue;
      if (preserveLiveContext && (key === "context_window" || key === "last_prompt_tokens"))
        continue;
      if (
        (key === "agent" || key === "model")
        && typeof isUnusedWebPlaceholder === "function"
        && isUnusedWebPlaceholder(sid)
      ) {
        const local = (
          (typeof selectedPickerAgent === "function" && selectedPickerAgent())
          || state.pickerAgent
          || ""
        );
        if (key === "agent" && local && String(s[key]) !== String(local)) continue;
        if (
          key === "model" &&
          (
            (local && s.agent && String(s.agent) !== String(local)) ||
            (typeof isLocalLlmAgent === "function" && isLocalLlmAgent(local || state.currentAgent) && typeof isModeSentinel === "function" && isModeSentinel(s[key]))
          )
        ) {
          continue;
        }
      }
      // State ticks read the web store, which still says "New Chat" until
      // something persists the gateway title — never clobber a generated
      // title the stream already applied via header_update.
      if (
        key === "title" &&
        typeof isDefaultSessionTitle === "function" &&
        isDefaultSessionTitle(s[key]) &&
        !isDefaultSessionTitle(row[key])
      ) {
        continue;
      }
      row[key] = s[key];
    }
    row.live = !!s.live;
    if (typeof applyContextHeader === "function") {
      applyContextHeader(s, { preserveLiveContext });
    }
    _applyRemoteAgentFromState(s.agent, {
      revision: s.revision,
      engine_model: s.engine_model,
      engine_model_alias: s.engine_model_alias,
    });
    // Agent-unchanged ticks used to skip applySessionState, so the composer
    // chip never got engine_model (Agents accordion still did, from the tree).
    if (
      typeof isUnusedWebPlaceholder !== "function" ||
      !isUnusedWebPlaceholder(sid) ||
      !s.agent ||
      String(s.agent).trim() === String(state.currentAgent || "").trim()
    ) {
      _promoteLiveSessionModel(s, _footerCurrentModel(payload));
    }
    if (typeof updateHeader === "function") updateHeader();
    if (typeof renderStatusSegments === "function") renderStatusSegments();
  }

  // Live activity → subagents, tree-live flag, streaming reconcile.
  const live = payload.live;
  if (live && sid) {
    if (typeof _applyLiveSnapshotToSession === "function") {
      _applyLiveSnapshotToSession(sid, live);
    } else {
      const row = state.sessions[sid];
      if (row) {
        row.state = live.session_state || row.state || "idle";
        row.live = !!live.tree_busy;
        row.tree_busy = !!live.tree_busy;
      }
    }
    if (Array.isArray(live.subagents)) {
      state.subagents = live.subagents;
      if (typeof renderHarnessWidget === "function") renderHarnessWidget();
      if (typeof renderSessionStatusBar === "function") renderSessionStatusBar();
    }
    if (typeof _setTreeLive === "function") _setTreeLive(!!live.tree_busy);
    if (typeof _reconcileStreamingFromActivity === "function") {
      _reconcileStreamingFromActivity(live);
    }
    if (typeof updateSendState === "function") updateSendState();
    renderSessionList();
  }

  // Gateway health → conn badge.
  if (payload.health && typeof payload.health.healthy === "boolean") {
    const ok = payload.health.healthy;
    const badge = document.getElementById("conn-badge");
    if (badge) {
      badge.className = "conn-badge " + (ok ? "conn-live" : "conn-down");
      badge.title = ok ? "Gateway connected" : "Gateway unreachable";
    }
    if (typeof applyComposerConnClasses === "function") applyComposerConnClasses(ok);
    renderStatusSegments();
  }

  // Pending approvals panel + inline permission/ask cards in the transcript.
  if (payload.approvals && Array.isArray(payload.approvals.approvals)) {
    state.pendingApprovals = payload.approvals.approvals;
    if (typeof renderPendingApprovals === "function") renderPendingApprovals();
    else _hydratePendingInteractiveCards();
  }

  // Sidebar live/waiting pills — only merge once the menu has been opened
  // (the rest sessions list itself is a lazy REST load on first expand).
  if (payload.live_sessions && Array.isArray(payload.live_sessions.sessions) &&
      _menuSectionOpen("sidebar-conversations")) {
    mergeSessionList(payload.live_sessions.sessions);
  }

  // Budget / goal / Agent-Monitor tree / footer widgets.
  if (payload.budget && payload.budget.ok !== false) {
    const incoming = payload.budget.budget && payload.budget.budget.total
      ? payload.budget.budget : null;
    if (!(typeof ignoreStaleBudgetSnapshot === "function" &&
        ignoreStaleBudgetSnapshot(sid,
          incoming && incoming.total && incoming.total.budget_usd))) {
      state.budget = incoming;
      renderStatusSegments();
    }
  }
  if (payload.goal && payload.goal.ok !== false) {
    const goalId = String(payload.goal.goal_id || "");
    const goal = payload.goal.goal;
    if (goal && goalId) {
      if (state.goalDismissedAfterMet && goal.status !== "met") {
        state.goalDismissedAfterMet = false;
      }
      applyGoalProgress({ goal_id: goalId });
      applyGoalDetail(goal, sid);
    } else if (!goalId && state.goalId && state.goalSid === sid &&
               typeof refreshGoalCost === "function") {
      // Re-consult only a goal already bound to THIS session.
      refreshGoalCost();
    }
  }
  if (payload.tree && payload.tree.ok !== false &&
      window.SaAgentMon && typeof window.SaAgentMon.applySnapshot === "function") {
    // Prefer the precomputed footer rows (includes current/previous after a
    // switch). Flattening payload.tree.tree alone drops that lineage.
    // Refresh chrome first so applySnapshot stamps current from session.agent.
    if (typeof updateAgentMonAccordion === "function") updateAgentMonAccordion();
    const footer = payload.tree.footer || payload.tree;
    if (state.sessions[sid]) state.sessions[sid].footer = footer;
    window.SaAgentMon.applySnapshot(footer);
  }
  if (payload.widgets && payload.widgets.ok !== false) {
    const sd = payload.widgets.status && typeof payload.widgets.status === "object"
      ? payload.widgets.status : {};
    if (typeof applyStatusSnapshot === "function") applyStatusSnapshot(sd);
    if (sd.agent) {
      _applyRemoteAgentFromState(sd.agent, {
        engine_model: sd.model,
      });
    }
    if (sd.model) _promoteLiveSessionModel({ engine_model: sd.model }, sd.model);
    const todoItems = Array.isArray(payload.widgets.todos) ? payload.widgets.todos : [];
    const taskItems = Array.isArray(payload.widgets.tasks) ? payload.widgets.tasks : [];
    state.queue = Array.isArray(payload.widgets.queue) ? payload.widgets.queue : [];
    // "Don't wipe during a live turn" is session-scoped: it protects THIS
    // session's live TodoWrite/Task* rows from a stale or empty state tick,
    // but rows stamped for ANOTHER session are never protected — a snapshot
    // is always authoritative for the session it describes (a subagent with
    // no .todos.json legitimately reports []).
    const sameTodoSession = state.todosSid === sid;
    if (!sameTodoSession) {
      state.todos = todoItems;
      state.todosSid = sid;
    } else if (!state.isStreaming) {
      const completedExisting = (state.todos || []).filter((t) => {
        const st = String((t && t.status) || "").toLowerCase();
        return st === "completed" || st === "done";
      });
      if (completedExisting.length) {
        const contentSet = new Set(todoItems.map((t) => t.content));
        state.todos = todoItems.concat(
          completedExisting.filter((t) => !contentSet.has(t.content))
        );
      } else {
        state.todos = todoItems;
      }
      state.todosSid = sid;
    }
    const sameTaskSession = state.harnessTasksSid === sid;
    if (
      taskItems.length ||
      !sameTaskSession ||
      !(state.isStreaming && (state.harnessTasks || []).length)
    ) {
      state.harnessTasks = taskItems;
      state.harnessTasksSid = sid;
    }
    if (typeof renderSessionStatusBar === "function") renderSessionStatusBar();
    if (typeof renderGoalWidget === "function") renderGoalWidget();
    if (typeof renderHarnessWidget === "function") renderHarnessWidget();
    if (typeof renderQueueWidget === "function") renderQueueWidget();
  }
}

window.__saLiveSources = window.__saLiveSources || new Set();

function closeLiveStream() {
  _clearLiveStreamWatchdog();
  if (window.__saLiveSources) {
    for (const s of window.__saLiveSources) {
      try { s.close(); } catch (_e) {}
    }
    window.__saLiveSources.clear();
  }
  if (state.liveSource) {
    try {
      state.liveSource.close();
    } catch (_e) {}
  }
  state.liveSource = null;
  state.liveSid = null;
  // The assistant bubble belongs to the SESSION, not the EventSource.
  // Dropping liveTurn here was the empty SUCCEEDED card: reconnect/watchdog
  // closed the pipe, then turn_ended called ensureLiveAssistant() and minted
  // a second header-only bubble. clearMessages() drops liveTurn on navigate.
  // Don't reset virtual scroller here — loadSession/clearMessages handles it.
}

// The gateway sends a `log` or `hb` frame at least every
// _SESSION_STREAM_HEARTBEAT (15s server-side) whenever the connection is
// actually healthy. 45s gives 3x margin for network jitter while still
// recovering well within the minutes-long stalls this was written for (see
// openLiveStream's `error`/`gw_error`/`reconnect` handlers — this watchdog
// covers the gap those leave: a connection that goes silent WITHOUT ever
// emitting an error/reconnect event, which native EventSource does not
// consider a failure on its own).
// A first open that never delivers ready/state/log/hb is a buffering proxy,
// not jitter — try fetch SSE once, then flip to poll after LIVE_STREAM_COLD_MS.
// Auto-poll is not sticky forever: shadow-probe SSE later (see sseRetryAllowed).
const LIVE_STREAM_WATCHDOG_MS = 45000;
const LIVE_STREAM_COLD_MS = 8000;
const LIVE_TRANSPORT_KEY = "sharedagents.web.transport";
const LIVE_TRANSPORT_AUTO_KEY = "sharedagents.web.transport.auto";
const LIVE_SSE_PROBE_BACKOFF_MS = [15000, 30000, 120000, 300000, 900000];
const LIVE_VIS_HIDDEN_MIN_MS = 2000;

function sseRetryBackoffMs(failCount) {
  const n = Math.max(0, failCount | 0);
  return LIVE_SSE_PROBE_BACKOFF_MS[Math.min(n, LIVE_SSE_PROBE_BACKOFF_MS.length - 1)];
}

function sseRetryAllowed(opts) {
  if (!opts) return false;
  if (opts.forcedOff || opts.userPinned || opts.urlPinned) return false;
  if (!opts.autoPolling) return false;
  if (opts.transport !== "poll") return false;
  if (opts.probeBusy) return false;
  return true;
}

function sseProbePromoteEvent(name) {
  return name === "ready" || name === "state" || name === "hb";
}

function liveStreamAlreadyFramed(sid) {
  return !!(sid && state._liveStreamFramedSid && state._liveStreamFramedSid === sid);
}

function _markLiveStreamFramed(sid) {
  if (sid) state._liveStreamFramedSid = sid;
}

/** First-open cold failures may flip EventSource → fetch SSE → poll.
 *  A cancel-induced drop on a session that already streamed must not;
 *  that is the multi-device desync (one tab stays on SSE, the other
 *  silently polls every 4s and misses the continuation turn). */
function _shouldAutoDowngradeLiveTransport(sid, sawFrameOnThisConnection) {
  if (sawFrameOnThisConnection) return false;
  if (typeof liveTransport === "function" && liveTransport() === "poll") return false;
  return !liveStreamAlreadyFramed(sid);
}

function _sseForcedOff() {
  const meta = document.querySelector('meta[name="sharedagents-disable-sse"]');
  return !!(meta && meta.content === "1");
}

function _urlPinsPoll() {
  try {
    if (state._urlPinReleased || typeof location === "undefined") return false;
    return (new URLSearchParams(location.search || "").get("transport") || "").toLowerCase() === "poll";
  } catch (_e) {
    return false;
  }
}

function liveTransport() {
  if (_sseForcedOff()) return "poll";
  return state._preferPolling ? "poll" : "sse";
}

function syncTransportToggle() {
  const root = document.getElementById("opt-transport");
  if (!root) return;
  const poll = liveTransport() === "poll";
  const choices = root.querySelectorAll(".opt-live-choice");
  for (let i = 0; i < choices.length; i++) {
    const on = (choices[i].getAttribute("data-transport") === "poll") === poll;
    choices[i].setAttribute("aria-pressed", String(on));
    choices[i].classList.toggle("is-active", on);
  }
}

function setLiveTransport(mode, opts) {
  const forced = _sseForcedOff();
  const reason = (opts && opts.reason) || "user";
  // An explicit choice overrides the URL for this page lifetime.
  if (reason === "user") state._urlPinReleased = mode === "sse";
  // Gear Poll, ?transport=poll, and the persisted user key are a hard pin.
  // Auto-fallback and shadow probes must never convert that into auto-poll
  // (which would schedule SSE retries) or promote back to Stream.
  const pinned = !forced && !!(state._userPinnedPoll || _urlPinsPoll());
  if (pinned && reason !== "user") {
    state._preferPolling = true;
    state._autoPolling = false;
    state._userPinnedPoll = true;
    _clearSseProbe();
    try { localStorage.removeItem(LIVE_TRANSPORT_AUTO_KEY); } catch (_e) {}
    try { localStorage.setItem(LIVE_TRANSPORT_KEY, "poll"); } catch (_e) {}
    syncTransportToggle();
    if (mode === "sse") return;
    const reconnectPinned = !opts || opts.reconnect !== false;
    if (!reconnectPinned) return;
    const pinSid = state.currentSessionId;
    if (!pinSid) return;
    const pinResume = state.liveOffset || 0;
    if (state.liveSource) {
      try { state.liveSource.close(); } catch (_e) {}
    }
    state.liveSource = null;
    openLiveStream(pinSid, pinResume);
    return;
  }
  const wantPoll = forced || mode === "poll";
  if (wantPoll && reason === "auto" && !forced) {
    state._preferPolling = true;
    state._autoPolling = true;
    state._userPinnedPoll = false;
    try { localStorage.setItem(LIVE_TRANSPORT_AUTO_KEY, "1"); } catch (_e) {}
    _scheduleSseProbe("auto-fallback");
  } else if (wantPoll) {
    state._preferPolling = true;
    state._autoPolling = false;
    state._userPinnedPoll = !forced;
    _clearSseProbe();
    try { localStorage.removeItem(LIVE_TRANSPORT_AUTO_KEY); } catch (_e) {}
    try { localStorage.setItem(LIVE_TRANSPORT_KEY, "poll"); } catch (_e) {}
  } else {
    state._preferPolling = false;
    state._autoPolling = false;
    state._userPinnedPoll = false;
    if (reason !== "probe") state._preferFetchSse = false;
    _clearSseProbe();
    try { localStorage.removeItem(LIVE_TRANSPORT_AUTO_KEY); } catch (_e) {}
    try { localStorage.setItem(LIVE_TRANSPORT_KEY, "sse"); } catch (_e) {}
  }
  syncTransportToggle();
  const reconnect = !opts || opts.reconnect !== false;
  if (!reconnect) return;
  const sid = state.currentSessionId;
  if (!sid) return;
  const resume = state.liveOffset || 0;
  if (state.liveSource) {
    try { state.liveSource.close(); } catch (_e) {}
  }
  state.liveSource = null;
  openLiveStream(sid, resume);
}

function _sseRetryOpts() {
  return {
    forcedOff: _sseForcedOff(),
    userPinned: !!state._userPinnedPoll,
    urlPinned: _urlPinsPoll(),
    autoPolling: !!state._autoPolling,
    transport: liveTransport(),
    probeBusy: !!state._sseProbeBusy,
  };
}

function _clearSseProbeTimer() {
  if (state._sseProbeTimer) {
    clearTimeout(state._sseProbeTimer);
    state._sseProbeTimer = null;
  }
}

function _clearSseProbe() {
  state._sseProbeGen = (state._sseProbeGen || 0) + 1;
  _clearSseProbeTimer();
  if (state._sseProbeDeadline != null) {
    clearTimeout(state._sseProbeDeadline);
    state._sseProbeDeadline = null;
  }
  const probe = state._sseProbeSource;
  state._sseProbeSource = null;
  state._sseProbeBusy = false;
  if (probe) {
    try { probe.close(); } catch (_e) {}
  }
}

function _scheduleSseProbe(reason) {
  _clearSseProbeTimer();
  if (!sseRetryAllowed(_sseRetryOpts())) return;
  const delay = sseRetryBackoffMs(state._sseProbeFails || 0);
  state._sseProbeTimer = setTimeout(() => {
    state._sseProbeTimer = null;
    _runSseProbe(reason || "schedule");
  }, delay);
}

function _runSseProbe(reason) {
  if (!sseRetryAllowed(_sseRetryOpts())) return;
  const sid = state.currentSessionId;
  if (!sid) {
    _scheduleSseProbe(reason || "no-sid");
    return;
  }
  _clearSseProbe();
  const generation = state._sseProbeGen;
  state._sseProbeBusy = true;
  const fromOffset = state.liveOffset || 0;
  const url = `${BASE}/api/sessions/${encodeURIComponent(sid)}/stream` +
    `?mode=events&from=${encodeURIComponent(fromOffset)}` +
    `&activity=stream&reasoning=stream&tools=stream`;
  const useFetch = state._preferFetchSse || typeof EventSource === "undefined";
  let probe = null;
  const finish = (ok) => {
    if (state._sseProbeGen !== generation || state._sseProbeSource !== probe) return;
    _clearSseProbe();
    // Completion releases probeBusy before checking the current policy:
    // a pin or forced-off setting may have changed while the probe ran.
    if (!sseRetryAllowed(_sseRetryOpts())) return;
    if (ok) {
      console.info("Live stream probe: SSE healthy — leaving poll", reason || "");
      state._sseProbeFails = 0;
      setLiveTransport("sse", { reason: "probe" });
    } else if (!useFetch) {
      // Reloads forget native failures. Try fetch in this cycle before backoff.
      state._preferFetchSse = true;
      _runSseProbe("probe-native-fail");
    } else {
      state._sseProbeFails = (state._sseProbeFails || 0) + 1;
      _scheduleSseProbe("probe-fail");
    }
  };
  try {
    probe = useFetch ? new FetchEventSource(url) : new EventSource(url);
  } catch (_e) {
    finish(false);
    return;
  }
  state._sseProbeSource = probe;
  state._sseProbeDeadline = setTimeout(() => finish(false), LIVE_STREAM_COLD_MS);
  ["ready", "state", "hb"].forEach((name) => {
    probe.addEventListener(name, () => {
      if (!sseProbePromoteEvent(name)) return;
      finish(true);
    });
  });
  probe.addEventListener("error", () => {
    finish(false);
  });
}

function _bindSseProbeSignals() {
  if (state._sseProbeBound) return;
  state._sseProbeBound = true;
  window.addEventListener("online", () => {
    if (!state._autoPolling) return;
    state._sseProbeFails = 0;
    // Network may have recovered while a dying native probe is in flight.
    // Cancel that deadline/source so a replacement cannot be corrupted.
    _clearSseProbe();
    _runSseProbe("online");
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "visible") {
      state._sseHiddenAt = Date.now();
      return;
    }
    const hiddenFor = Date.now() - (state._sseHiddenAt || 0);
    if (hiddenFor < LIVE_VIS_HIDDEN_MIN_MS) return;
    if (!state._autoPolling) return;
    _clearSseProbe();
    _runSseProbe("visible");
  });
}

function _fallbackLiveTransport(sid, resume) {
  if (liveTransport() === "poll") {
    openLiveStream(sid, resume);
    return;
  }
  if (!state._preferFetchSse && typeof EventSource !== "undefined") {
    state._preferFetchSse = true;
    console.warn("Live stream: no first frame — retrying via fetch SSE");
    openLiveStream(sid, resume);
    return;
  }
  console.warn("Live stream: SSE blocked — falling back to poll");
  setLiveTransport("poll", { reconnect: false, reason: "auto" });
  openLiveStream(sid, resume);
}

function _armLiveStreamWatchdog(sid, sawFrame) {
  _clearLiveStreamWatchdog();
  const delay = sawFrame ? LIVE_STREAM_WATCHDOG_MS : LIVE_STREAM_COLD_MS;
  state._liveStreamWatchdog = setTimeout(() => {
    if (state.liveSid !== sid || state.currentSessionId !== sid) return; // stale
    const resume = state.liveOffset;
    if (state.liveSource) {
      try { state.liveSource.close(); } catch (_e) {}
    }
    state.liveSource = null;
    if (!sawFrame && liveTransport() !== "poll") {
      _fallbackLiveTransport(sid, resume);
      return;
    }
    console.warn(
      `Live stream watchdog: no frame in ${delay}ms — forcing reconnect`,
    );
    openLiveStream(sid, resume);
  }, delay);
}

function _clearLiveStreamWatchdog() {
  if (state._liveStreamWatchdog) {
    clearTimeout(state._liveStreamWatchdog);
    state._liveStreamWatchdog = null;
  }
}

// Subscribe to the canonical incremental event stream (spec §9.2,
// mode=events). live == replay == observe: every frame is `event: log` with
// `{offset, seq, spawnedBy, event}`; `from=<byte>` replays from that offset
// then tails live. Default from=0 replays the whole session (mirrors a turn
// continued in the TUI / another frontend). Each canonical event is dispatched
// to renderCanonicalEvent, which drives the SAME dom/render helpers used for
// restored history (reasoning disclosure, tool rows, model badge).
// Fetch-based SSE reader for enterprise proxies that block native EventSource
// (they often still allow a same-origin GET with Accept: text/event-stream).
function FetchEventSource(url) {
  this.url = url;
  this.readyState = 0; // CONNECTING
  this._listeners = Object.create(null);
  this._abort = new AbortController();
  this._buf = "";
  const self = this;
  fetch(url, {
    method: "GET",
    headers: { Accept: "text/event-stream" },
    credentials: "same-origin",
    signal: this._abort.signal,
  }).then((resp) => {
    if (self.readyState === 2) return;
    if (!resp.ok || !resp.body) {
      self.readyState = 2;
      self._emit("error", {});
      return;
    }
    self.readyState = 1;
    self._emit("open", {});
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    const pump = () => reader.read().then(({ value, done }) => {
      if (self.readyState === 2) return;
      if (done) {
        self.readyState = 2;
        self._emit("error", {});
        return;
      }
      self._buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = self._buf.indexOf("\n\n")) >= 0) {
        const raw = self._buf.slice(0, idx);
        self._buf = self._buf.slice(idx + 2);
        let type = "message";
        const dataLines = [];
        raw.split("\n").forEach((line) => {
          if (line.startsWith("event:")) type = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
        });
        self._emit(type, { data: dataLines.join("\n") });
      }
      return pump();
    }).catch(() => {
      if (self.readyState === 2) return;
      self.readyState = 2;
      self._emit("error", {});
    });
    return pump();
  }).catch(() => {
    if (self.readyState === 2) return;
    self.readyState = 2;
    self._emit("error", {});
  });
}
FetchEventSource.prototype.addEventListener = function (type, fn) {
  (this._listeners[type] || (this._listeners[type] = [])).push(fn);
};
FetchEventSource.prototype.close = function () {
  this.readyState = 2;
  try { this._abort.abort(); } catch (_e) {}
};
FetchEventSource.prototype._emit = function (type, ev) {
  const list = this._listeners[type] || [];
  for (let i = 0; i < list.length; i++) {
    try { list[i](ev); } catch (_e) {}
  }
};

// Cursor-based polling: the fallback when SSE is disabled or blocked.
// GET /api/sessions/<sid>/events?from=<offset>&timeout=10
function PollingEventSource(sid, fromOffset) {
  this.sid = sid;
  this.offset = fromOffset || 0;
  this.readyState = 0;
  this._listeners = Object.create(null);
  this._abort = new AbortController();
  this._closed = false;
  this._pollTimer = null;
  // Signatures of what this client already holds. Each is reported back to the
  // server (`state_sig` / `catalogue_rev`) so an unchanged payload never rides
  // the poll loop while a CHANGE still arrives on the very next poll.
  // Empty string = hold nothing = ask to be given it.
  this._stateSig = "";
  this._catalogueRev = "";
  const self = this;

  function doPoll() {
    if (self._closed) return;
    const initial = self.readyState === 0;
    const url = `${BASE}/api/sessions/${encodeURIComponent(self.sid)}/events` +
      `?from=${encodeURIComponent(self.offset)}&timeout=${initial ? 0 : 10}` +
      // There is no `include_state` switch: the server sends the snapshot when
      // we are MISSING it (we report no signature) or when it MOVED.
      (self._stateSig ? `&state_sig=${encodeURIComponent(self._stateSig)}` : "") +
      (self._catalogueRev
        ? `&catalogue_rev=${encodeURIComponent(self._catalogueRev)}`
        : "&catalogue=1");
    fetch(url, {
      method: "GET",
      headers: { Accept: "application/json" },
      credentials: "same-origin",
      signal: self._abort.signal,
    }).then((resp) => {
      if (self._closed) return;
      if (!resp.ok) {
        self._emit("error", {});
        self._pollTimer = setTimeout(doPoll, 2000);
        return;
      }
      if (self.readyState === 0) {
        self.readyState = 1;
        self._emit("open", {});
      }
      return resp.json().then((data) => {
        if (self._closed) return;
        if (data && data.ok) {
          // Adopt the server's signature for what we now hold — including on a
          // poll that carried no state, where it restates the value we sent.
          // An empty signature (an undigestible snapshot) keeps us asking,
          // which is the intended fail-open.
          if (typeof data.state_sig === "string") self._stateSig = data.state_sig;
          if (data.state) {
            self._emit("state", { data: JSON.stringify(data.state) });
          }
          // Catalogue (bootstrap or a changed revision): surface it as the same
          // `models` event the SSE path delivers, so both transports share one
          // client entry point, and remember the revision we now hold.
          if (data.models_catalogue) {
            self._catalogueRev = data.models_catalogue.revision || "";
            self._emit("models", { data: JSON.stringify(data.models_catalogue) });
          }
          if (Array.isArray(data.events)) {
            for (let i = 0; i < data.events.length; i++) {
              self._emit("log", { data: JSON.stringify(data.events[i]) });
            }
          }
          if (typeof data.next_offset === "number") self.offset = data.next_offset;
        }
        self._pollTimer = setTimeout(doPoll, 250);
      });
    }).catch(() => {
      if (self._closed) return;
      self._emit("error", {});
      self._pollTimer = setTimeout(doPoll, 2000);
    });
  }
  setTimeout(doPoll, 10);
}
PollingEventSource.prototype.addEventListener = function (type, fn) {
  (this._listeners[type] || (this._listeners[type] = [])).push(fn);
};
PollingEventSource.prototype.close = function () {
  this._closed = true;
  this.readyState = 2;
  if (this._pollTimer) clearTimeout(this._pollTimer);
  try { this._abort.abort(); } catch (_e) {}
};
PollingEventSource.prototype._emit = function (type, ev) {
  const list = this._listeners[type] || [];
  for (let i = 0; i < list.length; i++) {
    try { list[i](ev); } catch (_e) {}
  }
};

function _sseDisabled() {
  return liveTransport() === "poll";
}

function createLiveEventSource(sid, fromOffset, url) {
  let source;
  // ask_user.js still calls createLiveEventSource(url) with one argument.
  if (typeof sid === "string" && sid.indexOf("/") !== -1 && url == null) {
    url = sid;
    if (state._preferFetchSse || typeof EventSource === "undefined") {
      source = new FetchEventSource(url);
    } else {
      source = new EventSource(url);
    }
  } else if (_sseDisabled()) {
    source = new PollingEventSource(sid, fromOffset);
  } else if (state._preferFetchSse || typeof EventSource === "undefined") {
    source = new FetchEventSource(url);
  } else {
    source = new EventSource(url);
  }
  window.__saLiveSources = window.__saLiveSources || new Set();
  window.__saLiveSources.add(source);
  return source;
}

function openLiveStream(sid, from) {
  if (!sid) {
    closeLiveStream();
    return;
  }
  const intent = String(state._openSessionIntent || "").trim();
  if (intent && sid !== intent) {
    return;
  }
  let fromOffset = Number.isFinite(from) && from < Number.MAX_SAFE_INTEGER ? from : 0;
  const tailFrom = (state.history && Number.isFinite(state.history.tailFrom)
      && state.history.tailFrom >= 0 && state.history.tailFrom < Number.MAX_SAFE_INTEGER)
    ? state.history.tailFrom
    : null;
  const liveFrom = (Number.isFinite(state.liveOffset)
      && state.liveOffset >= 0 && state.liveOffset < Number.MAX_SAFE_INTEGER)
    ? state.liveOffset
    : null;
  // Same-session recovery: the live cursor is newer than the REST watermark.
  // No live cursor: use tailFrom for a bound log, else 0 (New Chat / web-only).
  if (fromOffset === 0) {
    if (liveFrom != null && liveFrom > 0) fromOffset = liveFrom;
    else if (tailFrom != null) fromOffset = tailFrom;
  }
  // Reuse existing healthy connection for the same session and offset to avoid duplicate streams
  if (state.liveSource && state.liveSid === sid && state.liveOffset === fromOffset) {
    try {
      if (state.liveSource.readyState === 1 || state.liveSource.readyState === 0) {
        return;
      }
    } catch (_e) {}
  }
  closeLiveStream();
  // Reconnect must keep painting into the already-open assistant card.
  if (!state.liveTurn && state.isStreaming &&
      typeof adoptInFlightAssistant === "function") {
    adoptInFlightAssistant();
  }
  let es;
  try {
    // Disclosure toggles only expand/collapse already-received cards. Keep
    // full fidelity on the wire so a collapsed Tools card can be opened later
    // and direct-dispatch tool bodies still reach the user-facing answer.
    es = createLiveEventSource(
      sid,
      fromOffset,
      `${BASE}/api/sessions/${encodeURIComponent(sid)}/stream` +
        `?mode=events&from=${encodeURIComponent(fromOffset)}` +
        `&activity=stream&reasoning=stream&tools=stream`,
    );
  } catch (_e) {
    return;
  }
  state.liveSource = es;
  state.liveSid = sid;
  state.liveOffset = fromOffset;
  _bindSseProbeSignals();
  if (state._autoPolling && liveTransport() === "poll"
      && !state._sseProbeTimer && !state._sseProbeBusy) {
    _scheduleSseProbe("open");
  }
  // Loop guard for the `reconnect` handler: the offset this stream MOUNTED at,
  // and whether it ever delivered a frame. Together they answer "did this
  // connection make progress?" — a reconnect that did neither must back off
  // instead of reopening synchronously.
  const _mountOffset = fromOffset;
  let _sawFrame = false;
  _armLiveStreamWatchdog(sid, false);
  es.addEventListener("ready", () => {
    if (state.liveSid !== sid || state.currentSessionId !== sid) return;
    _sawFrame = true;
    _markLiveStreamFramed(sid);
    _armLiveStreamWatchdog(sid, true);
  });
  // One consolidated control frame — header/live/health/approvals/sidebar.
  es.addEventListener("state", (e) => {
    if (state.liveSid !== sid || state.currentSessionId !== sid) return;
    let frame;
    try {
      frame = JSON.parse(e.data);
    } catch (_e) {
      return;
    }
    _sawFrame = true;
    _markLiveStreamFramed(sid);
    _armLiveStreamWatchdog(sid, true);
    _applyState(frame);
  });
  // Discovered models frame — updates available models for agents live.
  es.addEventListener("models", (e) => {
    let frame;
    try {
      frame = JSON.parse(e.data);
    } catch (_e) {
      return;
    }
    _applyModelsFrame(frame);
  });
  // `log` frames carry one canonical event each — render incrementally.
  es.addEventListener("log", (e) => {
    if (state.liveSid !== sid || state.currentSessionId !== sid) return; // stale
    let frame;
    try {
      frame = JSON.parse(e.data);
    } catch (_e) {
      return;
    }
    _sawFrame = true; // this connection delivered — reconnect may reopen at once
    _markLiveStreamFramed(sid);
    // Any frame proves the connection is alive, regardless of whether THIS
    // tab is the one driving the turn — re-arm before the isStreaming-gated
    // watchdog below, which only matters for a locally-sent prompt.
    _armLiveStreamWatchdog(sid, true);
    // Track the byte offset so a reconnect resumes with zero loss (§9.2).
    if (Number.isFinite(frame.offset) && frame.offset < Number.MAX_SAFE_INTEGER) {
      state.liveOffset = frame.offset;
    }
    // Any live frame from the gateway is proof the turn is still progressing —
    // refresh the streaming watchdog so it stays an INACTIVITY guard. Without
    // this a healthy long turn (e.g. a cursor-agent CLI doing minutes of work)
    // trips the 5-min absolute timer mid-flight, the running indicator vanishes
    // and the session looks stale even though it is still streaming.
    if (state.isStreaming) _startStreamingWatchdog();
    if (frame.event) renderCanonicalEvent(frame.event, frame);
    // Keep the bulk-load idle timer alive across the replay; when the stream
    // goes quiet (history done) finishBulkLoad() fires the one-time scroll pin.
    VirtualScroller._noteBulkActivity();
  });
  // Heartbeats keep proxies warm and prove the logical stream is still alive
  // even when no turn events are flowing. Re-arm the inactivity watchdog so a
  // long tool/CLI silence does not clear the running UI while SSE is healthy.
  es.addEventListener("hb", () => {
    if (state.liveSid !== sid || state.currentSessionId !== sid) return;
    _sawFrame = true;
    _markLiveStreamFramed(sid);
    _armLiveStreamWatchdog(sid, true);
    // Heartbeats prove the SSE connection, not the parent turn. Re-arm the
    // running-UI watchdog only while durable state still says the turn is live;
    // otherwise an error/idle finalize stays "warming up" forever.
    if (state.isStreaming) {
      const cur = state.sessions && state.sessions[sid];
      const st = cur && (cur.state || cur.session_state);
      if (!st || typeof _parentTurnIsLive !== "function" || _parentTurnIsLive(st))
        _startStreamingWatchdog();
    }
  });
  // Web proxy emits gw_error when the upstream gateway is unreachable / times
  // out. Treat it like a transport death and resume from liveOffset — do not
  // leave a half-open EventSource that still owns liveSid.
  es.addEventListener("gw_error", () => {
    if (state.liveSource !== es) return;
    if (state.liveSid !== sid || state.currentSessionId !== sid) return;
    _clearLiveStreamWatchdog(); // this handler owns the reconnect below
    const resume = state.liveOffset;
    try { es.close(); } catch (_e) {}
    if (state.liveSource === es) state.liveSource = null;
    setTimeout(() => {
      if (state.liveSid !== sid || state.currentSessionId !== sid) return;
      if (state.liveSource) return;
      openLiveStream(sid, resume);
    }, 1000);
  });
  // The gateway caps a stream's lifetime and emits `reconnect` with the last
  // byte offset; reopen from there so no event is missed across the gap.
  es.addEventListener("reconnect", (e) => {
    if (state.liveSid !== sid || state.currentSessionId !== sid) return; // stale
    let resume = state.liveOffset;
    try {
      const d = JSON.parse(e.data);
      if (typeof d.offset === "number") resume = d.offset;
    } catch (_e) {}
    // A reconnect frame is a gateway keepalive even when no events flowed —
    // count it as activity so a silent-but-alive turn keeps the watchdog fresh.
    // Same gate as hb: do not keep a finalized error/idle session "running".
    if (state.isStreaming) {
      const cur = state.sessions && state.sessions[sid];
      const st = cur && (cur.state || cur.session_state);
      if (!st || typeof _parentTurnIsLive !== "function" || _parentTurnIsLive(st))
        _startStreamingWatchdog();
    }
    // An IMMEDIATE reconnect (the proxy's one-shot `{}` frame for a session
    // with nothing to stream yet) carries no offset and ends the response in
    // 27 bytes. Reopening it synchronously spins a tight loop: prod logged
    // 2107 opens in 6 h on one session and 4.5 h of sustained looping on
    // another, every one at from=0 — i.e. the whole transcript re-requested
    // over and over. Only an ADVANCING stream may reopen at once; a frame
    // that delivered no bytes backs off exactly like `error`/`gw_error`.
    const progressed = resume !== _mountOffset || _sawFrame;
    if (progressed) {
      openLiveStream(sid, resume);
      return;
    }
    try { es.close(); } catch (_e) {}
    if (state.liveSource === es) state.liveSource = null;
    setTimeout(() => {
      if (state.liveSid !== sid || state.currentSessionId !== sid) return;
      if (state.liveSource) return;
      openLiveStream(sid, resume);
    }, 1000);
  });
  // TRANSPORT error. Distinct from the `reconnect` frame above: that one is the
  // gateway politely handing us a resume offset; this is the connection simply
  // dying. Without this handler native EventSource auto-reconnects BY ITSELF,
  // reusing the URL it was constructed with — whose `from=` is the MOUNT-TIME
  // offset. Everything since mount then replays, and because `turn_started`
  // appends with no de-duplication the bottom of the transcript walks back
  // through old turns: the operator's "shows the last message then flickers to
  // some other turn".
  //
  // Idle deaths used to need ~11 minutes because the logical stream sent no
  // heartbeats and the web proxy closed the upstream after a 720-second read
  // timeout. Heartbeats now keep that path warm; this handler still covers
  // gateway restarts and hostile middlebox drops.
  //
  // Take every reconnect away from the browser — including a transient proxy
  // or service restart, where readyState is CONNECTING rather than CLOSED.
  // Close the source and reopen from `state.liveOffset`, which every `log`
  // frame has been advancing all along. Guarded on identity so a stale source
  // cannot resurrect a session the operator already left, and backed off so
  // an unreachable gateway does not become a reconnect loop.
  es.addEventListener("error", () => {
    if (state.liveSource !== es) return;              // already superseded
    if (state.liveSid !== sid || state.currentSessionId !== sid) return;
    _clearLiveStreamWatchdog();
    const resume = state.liveOffset;
    try { es.close(); } catch (_e) {}
    if (state.liveSource === es) state.liveSource = null;
    if (_shouldAutoDowngradeLiveTransport(sid, _sawFrame)) {
      if (!state._preferFetchSse && typeof EventSource !== "undefined") {
        state._preferFetchSse = true;
      } else {
        setLiveTransport("poll", { reconnect: false, reason: "auto" });
      }
    }
    setTimeout(() => {
      if (state.liveSid !== sid || state.currentSessionId !== sid) return;
      if (state.liveSource) return;                   // something reopened first
      openLiveStream(sid, resume);
    }, 1000);
  });
}

// ── Canonical event renderer (spec §7: render(event)) ───────────────────
// Maps ONE canonical log event to the EXISTING dom/render helpers. The live
// assistant bubble is built up from raw glyph text (the same shape the legacy
// folded path used), so reasoning (thinking_delta) and tool activity
// (tool_call/tool_result) flow through the UNCHANGED reasoning/tool renderers
// (parseAssistantContent → reasoningToHtml: the reasoning disclosure + tool
// rows). text_delta appends answer content; turn_ended finalizes + syncs the
// model badge; header_update drives state; user prompts render as user bubbles.
//
// `state.liveTurn = { el, bs }` is the scratch for the assistant bubble
// currently streaming. ``bs`` is a :class:`WebBlockStream` — the per-turn
// 💭/🔧/💬 block model ported from Telegram's ``_BlockStream``. Each canonical
// event feeds the model; ``renderBlockTurn`` repaints the bubble's blocks.

// ══════════════════════════════════════════════════════════════════════
// Block-stream renderer — Telegram ``_BlockStream`` parity (spec: render a
// turn as an ORDERED SEQUENCE of SEPARATE blocks, in arrival order:
//   💭 r1 · 💬 r1 · 🔧 tool · 💭 r2 · 💬 r2 … 💬 final + stats
// Rules (identical to sharedagents_telegram._BlockStream): no duplication (each
// block painted once; reasoning never copied into the answer); no fragmentation
// (within a round ALL reasoning → ONE 💭, ALL content → ONE 💬 regardless of
// think↔content interleave); tool calls are their own blocks in sequence
// EXCEPT direct-dispatch (delegate_*/compact) whose body IS the answer (routed
// to 💬 with a quiet ``↳ via <engine>`` badge) and internal tools (no block);
// 💭 sits above 💬; "on" reasoning is a single consolidated 💭 pinned at the
// top, filled once from ``turn_ended.final_thinking`` (live deltas gated out);
// the stats line closes the turn on the last/answer block.
//
// The PURE model below (between the BLOCKSTREAM_CORE sentinels) touches no DOM
// / globals so it is unit-testable in node — ``web/tests/test_block_render.js``
// extracts and drives it. The DOM painter (``renderBlockTurn``) lives outside.
// >>> BLOCKSTREAM_CORE_START (pure, DOM-free — do not reference document/state)


// ---------------------------------------------------------------------------
// Context meter ("used / max") — the header seeds/reconciles it, while live
// round/header events keep it current during an active turn.
// ---------------------------------------------------------------------------

function applyContextHeader(session, options) {
  if (!session) return false;
  const preserveLiveContext = !!(options && options.preserveLiveContext);
  let changed = false;
  if (typeof session.context_window === "number" && session.context_window > 0
      && !preserveLiveContext
      && state.lastContextMax !== session.context_window) {
    state.lastContextMax = session.context_window;
    changed = true;
  }
  if (typeof session.last_prompt_tokens === "number"
      && session.last_prompt_tokens > 0
      && !preserveLiveContext
      && state.lastContextUsed !== session.last_prompt_tokens) {
    state.lastContextUsed = session.last_prompt_tokens;
    changed = true;
  }
  // Manual-compaction stats (chat-area compaction turn card). The header is
  // the single source of truth; the card just renders whatever it last carried.
  const stats = session.compaction_last_stats;
  if (stats && typeof stats === "object" && !Array.isArray(stats) &&
      JSON.stringify(state.compactionLastStats || null) !== JSON.stringify(stats)) {
    state.compactionLastStats = stats;
    changed = true;
  }
  if (typeof session.compaction_summary === "string" &&
      (state.compactionSummary || "") !== session.compaction_summary) {
    state.compactionSummary = session.compaction_summary;
    changed = true;
  }
  if (typeof session.compaction_count === "number" &&
      state.compactionCount !== session.compaction_count) {
    state.compactionCount = session.compaction_count;
    changed = true;
  }
  return changed;
}

// Re-seed the footer's context segment from the session header after a turn.
// During a live turn, streaming.js applies each round's metrics directly from
// the canonical event stream; this REST reconciliation remains the terminal
// backstop when a consumer joined after the round frames.
//
// WHY THIS EXISTS: the live ``turn_ended`` frame was once synthesised
// from a conversation-lane ``assistant_message_committed`` event with no
// ``metrics``, and the footer sat at ``0/256K``. The gateway now streams the
// session log's raw events (``_stream_events_mode``), so live turn_ended and
// round_ended frames carry their metrics and the footer follows them directly.
// Reading the header here stays as the backstop for a consumer that joined
// without an open stream — it returns early while SSE is open.
async function syncContextMeterFromHeader() {
  const sid = state.currentSessionId;
  if (!sid || state.monitorReadOnly) return;
  if (typeof sseIsOpen === "function" && sseIsOpen()) return;
  try {
    const resp = await apiGet(`/api/sessions/${encodeURIComponent(sid)}`);
    const session = (resp && resp.session) || null;
    if (!session) return;
    if (state.currentSessionId !== sid) return; // a switch superseded us
    if (applyContextHeader(session)) renderSessionStatusBar();
  } catch (_e) {
    // Never let a footer refresh break the turn-end path.
  }
}


// The agent picker names what each agent is FOR.
// A card of rows (name + description) rather than a bare <select>: the
// distinctions that matter — which agent edits files, which is read-only, which
// is an external engine — live in the descriptions and were invisible before.
// Clicking a row calls the SAME applyAgentSwitch helper as #agent-select, so
// switching keeps one code path and the model list reloads for the newly
// selected agent exactly as it always did. Keep the card open after a successful
// switch: choosing an agent is step one of the agent -> model flow, not a dismiss
// action. The card's backdrop/outside-click, close button, and Escape handlers
// own dismissal.
// One line of feedback inside the picker itself. Anything the operator needs to
// read after tapping an agent must appear where they tapped, not in a collapsed
// panel elsewhere.
function _agentCardNotice(host, message) {
  if (!host) return;
  let note = host.querySelector(".agent-card-notice");
  if (!note) {
    note = document.createElement("div");
    note.className = "agent-card-notice";
    host.appendChild(note);
  }
  note.textContent = message;
}


function renderAgentCard() {
  const host = document.getElementById("agent-card");
  if (!host) return;
  const descs = state.agentDescriptions || {};
  const cur = selectedPickerAgent();
  // An empty roster must never render as an empty card. Whatever the upstream
  // cause — the roster fetch failed, the gateway was unreachable, or an earlier
  // init step threw before renderAgentSelect() ran — the operator sees a card
  // with nothing in it and no way to tell why, which is indistinguishable from
  // "the picker is broken". Say so, and offer the retry.
  if (!(state.agents || []).length) {
    host.innerHTML =
      '<div class="agent-row-empty">No agents loaded.' +
      '<button type="button" id="agent-card-retry">Retry</button></div>';
    const again = document.getElementById("agent-card-retry");
    if (again) {
      again.addEventListener("click", function (e) {
        e.stopPropagation();
        if (typeof loadAgents === "function") {
          Promise.resolve(loadAgents()).then(function () { renderAgentSelect(); });
        }
      });
    }
    return;
  }
  const catalog = window.AgentModelCatalogView;
  if (!catalog) {
    host.innerHTML = '<div class="agent-row-empty">Agent/model catalogue unavailable.</div>';
    return;
  }
  catalog.renderAgents({
    host,
    current: cur,
    agents: (state.agents || []).map(function (name) {
      const flags = (state.agentOrigins || {})[name] || {};
      const summary = typeof pickerAgentSummary === "function" ? pickerAgentSummary(name) : null;
      const def = summary && summary.default;
      const pace = (state.agentPaces || {})[name] || ((state.usageSummary || {}).agent_paces || {})[name];
      const credits = (state.agentCredits || {})[name] || ((state.usageSummary || {}).provider_credits || {})[name];
      const health = ((state.usageSummary || {}).agent_health || {})[name] || {};
      return {
        name,
        description: descs[name] || "",
        origin: flags.origin || "native",
        external: !!flags.external,
        // The agent's default model, from the picker summary.
        model: def ? String(def.label || def.alias || def.id || "") : "",
        pace,
        credits,
        // External-agent integrity: CLI version chip + sign-in alert.
        version: String(health.version || ""),
        auth_state: String(health.auth_state || ""),
        auth_hint: String(health.auth_hint || ""),
        installed: health.installed !== false,
      };
    }),
    // Kept for callers that still pass it; the sectioning itself now reads the
    // server-authoritative origin/external flags.
    isExternal: function (row) { return isExternalPickerAgent(row.name); },
    disabled: state.modelSheet && state.modelSheet.phase === "APPLYING",
  });
  host.querySelectorAll(".agent-row").forEach(function (row) {
    row.addEventListener("click", async function () {
      const want = row.getAttribute("data-agent") || "";
      if (!want) return;
      if (want === selectedPickerAgent()) {
        if (state.modelLoadError && state.modelLoadError.agent === want)
          loadModelsForAgent(want, { loading: true });
        return;
      }
      // A read-only session refuses the switch outright (switchConversation()
      // returns false immediately), so closing the card on tap made a refusal
      // look exactly like a broken picker: tap an agent, card vanishes, nothing
      // changes. Say why instead, and leave the card open.
      if (state.monitorReadOnly) {
        _agentCardNotice(host, "This session is read-only — take control to switch agents.");
        return;
      }
      // Unused New Chat: selectPickerAgent updates the Agent Monitor now
      // (and persists pending_agent). Bound sessions stay send-time.
      selectPickerAgent(want);
    });
  });
  const loadFailed = !!(state.modelLoadError && state.modelLoadError.agent === cur);
  if (loadFailed && !(state.modelOptionsByAgent[cur] || []).length) {
    const wrap = document.createElement("div");
    wrap.className = "agent-row-empty";
    wrap.appendChild(document.createTextNode(
      state.modelLoadError.message || "Could not load models for this agent.",
    ));
    const again = document.createElement("button");
    again.type = "button";
    again.id = "agent-models-retry";
    again.textContent = "Retry";
    again.addEventListener("click", function (e) {
      e.stopPropagation();
      loadModelsForAgent(cur, { loading: true });
    });
    wrap.appendChild(again);
    host.appendChild(wrap);
  }
}

const MODEL_SHEET_PHASES = new Set([
  "CLOSED", "OPEN_CLEAN", "OPEN_DIRTY", "APPLYING", "OPEN_ERROR",
]);

function setModelSheetPhase(phase, error = "") {
  ensureModelSheetState();
  if (!MODEL_SHEET_PHASES.has(phase)) return false;
  state.modelSheet.phase = phase;
  state.modelSheet.error = phase === "OPEN_ERROR" ? String(error || "") : "";
  renderModelSheet();
  if (typeof updateSendState === "function") updateSendState();
  return true;
}

// An explicit alias is the authoritative answer of the caller — for sync it
// is the model reported by a loaded session. When the currently-CACHED
// catalogue cannot represent it (the list predates the model, or belongs to
// an earlier agent state), the explicit alias is PRESERVED instead of being
// silently replaced by the active/default row: a stale display cache must
// never contradict the session. The active/default/first fallback applies
// only when the caller has no explicit model. The fresh catalogue refresh
// re-resolves the preserved alias onto a real row (loadModelsForAgent), so a
// full reload and an in-page navigation converge on the same selection.
function modelSheetSelectionFor(agent, alias) {
  const models = state.modelOptionsByAgent[agent] || [];
  const explicit = String(alias || "").trim();
  // With an explicit alias (the model reported by a loaded session), the
  // catalogue is consulted ONLY to normalize id/realId — a miss must not fall
  // through to the active/default row. Without an explicit alias the
  // catalogue default still applies.
  const row = explicit
    ? (models.find((m) => m.id === explicit) ||
       models.find((m) => m.realId && m.realId === explicit) ||
       models.find((m) => m.alias && m.alias === explicit) ||
       models.find((m) => m.wireId && m.wireId === explicit) ||
       null)
    : (models.find((m) => m.active) ||
       agentDefaultRow(agent, models));
  return cloneModelSelection({
    agent,
    provider: row ? row.provider : "",
    // A catalogue hit normalizes to the row id (realId → id); a miss keeps the
    // explicit alias rather than degrading to the catalogue default.
    alias: row ? row.id : explicit || String(alias || ""),
    wire_id: row ? row.wireId : "",
    reasoning: row
      ? (typeof reasoningForSelectedModel === "function"
        ? reasoningForSelectedModel(row.reasoning, agent, row)
        : null)
      : null,
    speed: row ? supportedDraftValue(row.speed, null) : null,
  });
}

// Reconcile only while closed: a loaded session is authoritative and an open
// draft must never be overwritten by polling/navigation paint.
function syncModelSheetEffectiveFromSession() {
  ensureModelSheetState();
  if (state.modelSheet.phase !== "CLOSED") return;
  const agent = state.currentAgent || selectedPickerAgent();
  const prior = state.modelSheet.effective || {};
  // Seed the picker from the operator-facing alias when
  // one exists — the catalogue key is the alias, so normalizing a raw wire
  // id would otherwise show the raw id until the catalogue catches up.
  const fresh = typeof isFreshUnboundSession === "function"
    && isFreshUnboundSession(state.currentSessionId);
  const raw = state.currentModel || (
    fresh ? "" : (state.engineModelAlias || state.engineModel || "")
  );
  const fromPicker = (prior.agent === agent ? prior.alias : "") ||
    (state.pickerModels || {})[agent] || "";
  // A session-stamped id that is merely ABSENT from a possibly-stale client
  // catalogue is not foreign to this session: keep it as the effective
  // selection instead of silently substituting the catalogue default (the
  // sheet renders it unmatched). Engine kinds, mode sentinels and inherited
  // local routes stay foreign (otherwise a gpt-6-astra session could be
  // rewritten to the default qwen row).
  const rawUnmatched = raw
    && isForeignAgentModel(agent, raw)
    && !isEngineKindLabel(raw)
    && !isModeSentinel(raw)
    && !looksLikeInheritedLocalRoute(agent, raw);
  const alias = (raw && !isForeignAgentModel(agent, raw) && raw) ||
    (rawUnmatched && raw) ||
    (fromPicker && !isForeignAgentModel(agent, fromPicker) && fromPicker) ||
    "";
  state.modelSheet.effective = modelSheetSelectionFor(agent, alias);
  state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
  state.pickerAgent = state.modelSheet.effective.agent;
  if (state.modelSheet.effective.alias &&
      !isForeignAgentModel(agent, state.modelSheet.effective.alias))
    state.pickerModels[agent] = state.modelSheet.effective.alias;
  else if (isForeignAgentModel(agent, (state.pickerModels || {})[agent]))
    delete state.pickerModels[agent];
}

// Render the server-resolved session effort on the closed composer.
// Open sheets render the selected model row from their picker response.
function syncModelSheetEffortFromStatus() {
  ensureModelSheetState();
  if (state.modelSheet.phase !== "CLOSED") return false;
  const selection = state.modelSheet.effective || {};
  const rows = state.modelOptionsByAgent[selection.agent] || [];
  const row = rows.find((model) =>
    model.id === selection.alias ||
    (selection.wire_id && model.wireId === selection.wire_id)
  );
  const value = state.statusInfo && state.statusInfo.effort_selected;
  if (!row || value == null) return false;
  state.modelSheet.effective = cloneModelSelection({ ...selection, reasoning: value });
  state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
  return true;
}

function modelSheetNodes() {
  return {
    sheet: document.getElementById("agent-model-pop"),
    backdrop: document.getElementById("amp-backdrop"),
    tabAgent: document.getElementById("amp-tab-btn-agent"),
    tabModel: document.getElementById("amp-tab-btn-model"),
    panelAgent: document.getElementById("amp-tab-agent"),
    panelModel: document.getElementById("amp-tab-model"),
    badgeAgent: document.getElementById("amp-active-agent-badge"),
    badgeModel: document.getElementById("amp-active-model-badge"),
    providers: document.getElementById("amp-provider-tabs"),
    models: document.getElementById("amp-model-cards"),
    search: document.getElementById("amp-model-search"),
    reasoningSection: document.getElementById("amp-reasoning-section"),
    reasoning: document.getElementById("amp-reasoning"),
    reasoningHint: document.getElementById("amp-reasoning-hint"),
    speedSection: document.getElementById("amp-speed-section"),
    speeds: document.getElementById("amp-speed-options"),
    error: document.getElementById("amp-error"),
    save: document.getElementById("amp-save"),
    close: document.getElementById("amp-close"),
    useDefault: document.getElementById("amp-default"),
    useSetDefault: document.getElementById("amp-set-default"),
    discard: document.getElementById("amp-discard"),
  };
}

function switchModelSheetTab(tabName) {
  ensureModelSheetState();
  state.modelSheet.activeTab = tabName === "model" ? "model" : "agent";
  const n = modelSheetNodes();
  if (state.modelSheet.activeTab === "agent") {
    if (n.tabAgent) {
      n.tabAgent.classList.add("is-active");
      n.tabAgent.setAttribute("aria-selected", "true");
      n.tabAgent.setAttribute("tabindex", "0");
    }
    if (n.tabModel) {
      n.tabModel.classList.remove("is-active");
      n.tabModel.setAttribute("aria-selected", "false");
      n.tabModel.setAttribute("tabindex", "-1");
    }
    if (n.panelAgent) {
      n.panelAgent.removeAttribute("hidden");
      n.panelAgent.classList.add("is-active");
    }
    if (n.panelModel) {
      n.panelModel.setAttribute("hidden", "");
      n.panelModel.classList.remove("is-active");
    }
    if (n.reasoningSection) n.reasoningSection.hidden = true;
    renderAgentCard();
  } else {
    if (typeof ensureSheetProviderLoaded === "function") ensureSheetProviderLoaded();
    if (n.tabModel) {
      n.tabModel.classList.add("is-active");
      n.tabModel.setAttribute("aria-selected", "true");
      n.tabModel.setAttribute("tabindex", "0");
    }
    if (n.tabAgent) {
      n.tabAgent.classList.remove("is-active");
      n.tabAgent.setAttribute("aria-selected", "false");
      n.tabAgent.setAttribute("tabindex", "-1");
    }
    if (n.panelModel) {
      n.panelModel.removeAttribute("hidden");
      n.panelModel.classList.add("is-active");
    }
    if (n.panelAgent) {
      n.panelAgent.setAttribute("hidden", "");
      n.panelAgent.classList.remove("is-active");
    }
    renderModelSheet();
  }
}

function openModelSheet(preferredTab) {
  if (state.monitorReadOnly) return false;
  // Only one footer sheet may be open at a time.
  if (window.SaContextSheet && window.SaContextSheet.isOpen()) {
    window.SaContextSheet.close({ restoreFocus: false });
  }
  ensureModelSheetState();
  syncModelSheetEffectiveFromSession();
  state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
  state.modelSheet.search = "";
  // Each opening asks a provider again only when its tab is opened.
  state.modelSheet.providerLoads = {};
  state.modelSheet.pendingStageProvider = "";
  if (preferredTab === "model" || preferredTab === "agent") {
    state.modelSheet.activeTab = preferredTab;
  } else {
    // The picker trigger always opens the agents/defaults view. A previous
    // Model view must not turn a later opening into a provider request.
    state.modelSheet.activeTab = "agent";
  }
  setModelSheetPhase("OPEN_CLEAN");
  state.modelSheet.ignoreOutsideUntil = Date.now() + 400;
  // Only one footer sheet may be open at a time.
  if (window.SaAgentMon && window.SaAgentMon.close) {
    window.SaAgentMon.close({ restoreFocus: false });
  }
  const n = modelSheetNodes();
  if (n.sheet) n.sheet.removeAttribute("hidden");
  if (n.backdrop) n.backdrop.removeAttribute("hidden");
  switchModelSheetTab(state.modelSheet.activeTab);
  const trigger = document.getElementById("agent-model-trigger");
  if (trigger) trigger.setAttribute("aria-expanded", "true");
  // Agent file logos load only when the picker is shown — not on page init.
  if (typeof hydratePickerLogos === "function") {
    hydratePickerLogos(n.sheet || document.getElementById("agent-card") || document);
  }
  // Opening asks for the agents and their default models only — no model
  // provider. A provider's models load when its tab (or the Model view
  // showing it) is opened.
  if (typeof loadPickerSummary === "function") {
    loadPickerSummary().then(function () {
      if (typeof reconcileDraftWithLoadedModels === "function")
        reconcileDraftWithLoadedModels(state.modelSheet.draft.agent);
      else renderModelSheet();
    }).catch(function () {
      // The roster the page already holds stays; the next opening retries.
      loadModelsForAgent(state.modelSheet.draft.agent, { loading: true });
    });
  }
  window.setTimeout(() => {
    const first = n.sheet && n.sheet.querySelector("button:not([disabled]), input:not([disabled])");
    if (first) first.focus();
  }, 0);
  return true;
}

function closeModelSheet() {
  ensureModelSheetState();
  if (state.modelSheet.phase === "APPLYING") return false;
  state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
  state.modelSheet.phase = "CLOSED";
  state.modelSheet.error = "";
  const n = modelSheetNodes();
  if (n.sheet) n.sheet.setAttribute("hidden", "");
  if (n.backdrop) n.backdrop.setAttribute("hidden", "");
  const trigger = document.getElementById("agent-model-trigger");
  if (trigger) {
    trigger.setAttribute("aria-expanded", "false");
    trigger.focus();
  }
  if (typeof updateSendState === "function") updateSendState();
  return true;
}

function bindModelSheetControls() {
  const _pop = document.getElementById("agent-model-pop");
  const _back = document.getElementById("amp-backdrop");
  if (!_pop || _pop.dataset.ampSheetBound) return;
  _pop.dataset.ampSheetBound = "1";
  const close = function () {
    if (typeof closeModelSheet === "function") closeModelSheet();
  };
  const ignoreOpeningClick = function () {
    return !!(state.modelSheet && Date.now() < (state.modelSheet.ignoreOutsideUntil || 0));
  };
  _pop.addEventListener("click", function (e) { e.stopPropagation(); });
  if (_back) _back.addEventListener("click", function () {
    if (ignoreOpeningClick()) return;
    close();
  });
  const btn = document.getElementById("amp-close");
  if (btn) btn.addEventListener("click", close);
  const tabAgent = document.getElementById("amp-tab-btn-agent");
  const tabModel = document.getElementById("amp-tab-btn-model");
  if (tabAgent) tabAgent.addEventListener("click", function (e) {
    e.stopPropagation();
    if (typeof switchModelSheetTab === "function") switchModelSheetTab("agent");
  });
  if (tabModel) tabModel.addEventListener("click", function (e) {
    e.stopPropagation();
    if (typeof switchModelSheetTab === "function") switchModelSheetTab("model");
  });
  const tabsNav = _pop.querySelector(".amp-tabs-nav");
  if (tabsNav) tabsNav.addEventListener("keydown", function (e) {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      const next = state.modelSheet && state.modelSheet.activeTab === "agent" ? "model" : "agent";
      if (typeof switchModelSheetTab === "function") switchModelSheetTab(next);
      const activeBtn = next === "agent" ? tabAgent : tabModel;
      if (activeBtn) activeBtn.focus();
    }
  });
  const discard = document.getElementById("amp-discard");
  if (discard) discard.addEventListener("click", close);
  const reset = document.getElementById("amp-default");
  if (reset) reset.addEventListener("click", function () {
    if (typeof stageDefaultModel === "function") stageDefaultModel();
  });
  const setDefaultBtn = document.getElementById("amp-set-default");
  if (setDefaultBtn) setDefaultBtn.addEventListener("click", function () {
    if (typeof setAgentDefaultFromSheet === "function") setAgentDefaultFromSheet();
  });
  const save = document.getElementById("amp-save");
  if (save) save.addEventListener("click", function () {
    if (typeof saveModelSheet === "function") saveModelSheet();
  });
  const providers = document.getElementById("amp-provider-tabs");
  if (providers) providers.addEventListener("click", function (e) {
    const hit = e.target.closest("[data-provider]");
    if (hit && typeof stageModelSheetProvider === "function") {
      stageModelSheetProvider(hit.dataset.provider);
    }
  });
  if (providers) providers.addEventListener("keydown", function (e) {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key) ||
        state.modelSheet.phase === "APPLYING") return;
    const tabs = [...providers.querySelectorAll('[role="tab"]:not([disabled])')];
    if (!tabs.length) return;
    const current = Math.max(0, tabs.indexOf(document.activeElement));
    let next = current;
    if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (current - 1 + tabs.length) % tabs.length;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (current + 1) % tabs.length;
    if (e.key === "Home") next = 0;
    if (e.key === "End") next = tabs.length - 1;
    e.preventDefault();
    const provider = tabs[next].dataset.provider;
    if (!provider || typeof stageModelSheetProvider !== "function") return;
    stageModelSheetProvider(provider);
    const selected = [...providers.querySelectorAll("[data-provider]")]
      .find((tab) => tab.dataset.provider === provider);
    if (selected) selected.focus();
  });
  const models = document.getElementById("amp-model-cards");
  const search = document.getElementById("amp-model-search");
  if (search) search.addEventListener("input", function () {
    ensureModelSheetState();
    if (state.modelSheet.phase === "APPLYING") return;
    const draft = state.modelSheet.draft;
    const query = search.value || "";
    state.modelSheet.searchByAgent = state.modelSheet.searchByAgent || {};
    if (draft && draft.agent) state.modelSheet.searchByAgent[draft.agent] = query;
    state.modelSheet.search = query;
    if (typeof renderModelSheet === "function") renderModelSheet();
  });
  if (models) models.addEventListener("click", function (e) {
    const hit = e.target.closest("[data-model]");
    if (hit && typeof selectPickerModel === "function") selectPickerModel(hit.dataset.model);
  });
  const reasoning = document.getElementById("amp-reasoning");
  if (reasoning) reasoning.addEventListener("change", function () {
    const draft = state.modelSheet.draft;
    const row = (state.modelOptionsByAgent[draft.agent] || [])
      .find((model) => model.id === draft.alias);
    const cap = (row && row.reasoning) || (typeof normalizeAdvancedCapability === "function" ? normalizeAdvancedCapability(null, "reasoning") : null);
    if (!cap || !cap.values || state.modelSheet.phase === "APPLYING") return;
    const idx = Number(reasoning.value);
    if (!Number.isInteger(idx) || idx < -1 || idx >= cap.values.length) return;
    draft.reasoning = idx === -1 ? "default" : cap.values[idx];
    setModelSheetPhase("OPEN_DIRTY");
  });
  const speeds = document.getElementById("amp-speed-options");
  if (speeds) speeds.addEventListener("click", function (e) {
    const hit = e.target.closest("[data-speed]");
    if (!hit || state.modelSheet.phase === "APPLYING") return;
    state.modelSheet.draft.speed = hit.dataset.speed;
    setModelSheetPhase("OPEN_DIRTY");
  });
  document.addEventListener("click", function (e) {
    if (_pop.hasAttribute("hidden")) return;
    if (ignoreOpeningClick()) return;
    const hit = e.target && e.target.closest && e.target.closest(
      "#agent-model-pop, #agent-model-trigger, .channel-amp-trigger"
    );
    if (hit) return;
    close();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !_pop.hasAttribute("hidden")) close();
    if (e.key !== "Tab" || _pop.hasAttribute("hidden")) return;
    const focusable = [..._pop.querySelectorAll(
      "button:not([disabled]), input:not([disabled]), select:not([disabled])",
    )].filter((node) => !node.closest("[hidden]"));
    if (!focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  });
}

// ── Compaction statistics in the view-only compaction child ──
// Parent control turns stay out of the chat. These helpers can show the child's
// structured stats beside its summary, collapsed by default. A per-turn
// {stats, summary, count} snapshot survives virtual-scroller rebuilds; the child
// session header is the fallback when one is available after a reload.

// Compact token-count formatting ("16.5K", "1.3M"). Pure — no DOM, unit-tested.
function fmtCompactTokens(n) {
  n = Number(n) || 0;
  if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "K";
  return String(n);
}

// Pure body markup for the stats card — no DOM, unit-tested. Self-contained
// escaper + formatter so the function evals standalone in the node harness.
function compactStatsBodyHtml(stats, summary) {
  const s = stats && typeof stats === "object" ? stats : {};
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => (
    c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" :
    c === '"' ? "&quot;" : "&#39;"
  ));
  const fmtTok = (n) => {
    n = Number(n) || 0;
    if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
    if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "K";
    return String(n);
  };
  const rows = [];
  const push = (label, value) => rows.push(
    '<div class="csp-row"><span class="csp-label">' + esc(label) + "</span>" +
    '<span class="csp-value">' + esc(value) + "</span></div>",
  );
  if (s.before_tokens != null && s.after_tokens != null) {
    push("Tokens", fmtTok(s.before_tokens) + " → " + fmtTok(s.after_tokens));
  }
  if (s.removed_tokens != null) push("Removed", fmtTok(s.removed_tokens) + " tokens");
  if (s.removed_chars != null) push("Removed chars", fmtTok(s.removed_chars));
  if (s.messages_trimmed != null) push("Messages trimmed", String(s.messages_trimmed));
  if (s.at) {
    const iso = new Date(String(s.at)).toISOString();
    push("When", iso ? iso.slice(0, 16).replace("T", " ") : String(s.at));
  }
  const summaryHtml = summary
    ? '<p class="csp-summary">' + esc(summary) + "</p>" : "";
  if (!rows.length && !summaryHtml) return "";
  return '<div class="csp-rows">' + rows.join("") + "</div>" + summaryHtml;
}

// Pure collapsed-line markup for the card ("🗜 Compaction #2 · 208.4K →
// 117.1K · −91.2K tokens"). Self-contained (no DOM / globals) so the node
// harness can eval it; falls back to the summary text when the stats dict
// has no token numbers.
function compactionCardLineHtml(stats, count, summary) {
  const s = stats && typeof stats === "object" ? stats : {};
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => (
    c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" :
    c === '"' ? "&quot;" : "&#39;"
  ));
  const n = Number(count) || 0;
  const title = n > 0 ? "🗜 Compaction #" + n : "🗜 Compaction";
  const parts = [];
  if (s.before_tokens != null && s.after_tokens != null) {
    parts.push(fmtCompactTokens(s.before_tokens) + " → " +
      fmtCompactTokens(s.after_tokens));
  }
  const removed = Number(s.removed_tokens) || 0;
  if (removed > 0) parts.push("−" + fmtCompactTokens(removed) + " tokens");
  let line = parts.join(" · ");
  if (!line) {
    // No numeric stats (e.g. a failed/edge compaction) — surface the summary.
    line = (summary || "").trim();
    if (line.length > 80) line = line.slice(0, 77) + "…";
  }
  return '<span class="mcs-title">' + esc(title) + "</span>" +
    (line ? '<span class="mcs-line">' + esc(line) + "</span>" : "");
}

// Toggle delegation for the in-chat card — bound ONCE on the stable messages
// container (the card node itself is re-created by the virtual scroller, same
// rule as .msg-copy): click toggles the details disclosure, Escape collapses
// an open card. No native dialogs anywhere.
function bindCompactionCardDelegation() {
  const host = (typeof messagesContainer !== "undefined" && messagesContainer) ||
    document.getElementById("messages");
  if (!host || host.dataset.mcsBound) return;
  host.dataset.mcsBound = "1";
  host.addEventListener("click", function (e) {
    const toggle = e.target && e.target.closest &&
      e.target.closest(".mcs-toggle");
    if (!toggle) return;
    e.stopPropagation();
    const card = toggle.closest(".message-compaction-stats");
    const details = card && card.querySelector(".mcs-details");
    if (!details) return;
    const open = !details.hasAttribute("hidden");
    if (open) details.setAttribute("hidden", "");
    else details.removeAttribute("hidden");
    toggle.setAttribute("aria-expanded", open ? "false" : "true");
  });
  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    const details = document.querySelector(
      ".message-compaction-stats .mcs-details:not([hidden])",
    );
    if (!details) return;
    const card = details.closest(".message-compaction-stats");
    const toggle = card && card.querySelector(".mcs-toggle");
    details.setAttribute("hidden", "");
    if (toggle) {
      toggle.setAttribute("aria-expanded", "false");
      toggle.focus(); // restore the trigger, like the old pop's focus restore
    }
  });
}

// Attach (or re-attach) the compaction-stats card inside an assistant message
// bubble. Idempotent: re-mounting with identical stats leaves the current DOM
// (and its open state) untouched; mounting on a non-compaction message removes
// any stale card. `snapshot` is the per-turn {stats, summary, count}; when
// omitted/null the header values are used (history re-hydration).
function mountCompactionStatsCard(el, snapshot) {
  if (!el || !el.classList || !el.classList.contains("message-assistant")) {
    return null;
  }
  bindCompactionCardDelegation();
  const raw = (el.dataset && el.dataset.raw) || "";
  const isCompactionTurn = state.currentAgent === "compaction" && typeof raw === "string" &&
    raw.trim().startsWith("compacted");
  const existing = el.querySelector(":scope > .message-compaction-stats");
  if (!isCompactionTurn) {
    if (existing) existing.remove();
    return null;
  }
  const snap = snapshot && typeof snapshot === "object" ? snapshot : {};
  const stats = (snap.stats && typeof snap.stats === "object" &&
    !Array.isArray(snap.stats)) ? snap.stats : null;
  const summary = typeof snap.summary === "string" ? snap.summary : "";
  const count = typeof snap.count === "number" ? snap.count : 0;
  // Header fallback: after a reload the header keeps the latest compaction's
  // values, which for the most recent compaction turn IS this turn's snapshot.
  const statsFor = stats ||
    ((typeof state !== "undefined" && state.compactionLastStats &&
      typeof state.compactionLastStats === "object" &&
      !Array.isArray(state.compactionLastStats))
      ? state.compactionLastStats : null);
  const summaryFor = summary ||
    (typeof state !== "undefined" && typeof state.compactionSummary === "string"
      ? state.compactionSummary : "");
  const countFor = count ||
    (typeof state !== "undefined" && typeof state.compactionCount === "number"
      ? state.compactionCount : 0);
  if (!statsFor && !summaryFor) return null; // compaction turn, no data yet
  if (existing) {
    const same = existing.dataset.statsJson === JSON.stringify(statsFor || null);
    if (same) return existing; // keep the open/closed state across re-mounts
    existing.remove();
  }
  const card = document.createElement("div");
  card.className = "message-compaction-stats";
  card.dataset.statsJson = JSON.stringify(statsFor || null);
  card.innerHTML =
    '<button type="button" class="mcs-toggle" aria-expanded="false">' +
    compactionCardLineHtml(statsFor, countFor, summaryFor) +
    '<span class="mcs-caret" aria-hidden="true">▾</span></button>' +
    '<div class="mcs-details" hidden>' +
    compactStatsBodyHtml(statsFor, summaryFor) + "</div>";
  const contentEl = el.querySelector(".message-content");
  if (contentEl && contentEl.nextSibling) {
    el.insertBefore(card, contentEl.nextSibling);
  } else {
    el.appendChild(card);
  }
  return card;
}

// The provider tab the Model view shows: the draft's provider when it is one
// of the agent's tabs, else the first tab.
function visibleSheetProvider() {
  const draft = state.modelSheet && state.modelSheet.draft;
  if (!draft) return "";
  const tabs = typeof pickerProviderTabs === "function" ? pickerProviderTabs(draft.agent) : [];
  const ids = tabs.length
    ? tabs.map((t) => t.id)
    : [...new Set((state.modelOptionsByAgent[draft.agent] || []).map((m) => m.provider || "Other"))];
  return ids.includes(draft.provider) ? draft.provider : (ids[0] || "");
}

// Ask the server for one provider tab's models, once per sheet opening. A tab
// the server already marked as an error (no key) is not requested: its
// message is rendered as sent.
function ensureSheetProviderLoaded(provider) {
  ensureModelSheetState();
  const draft = state.modelSheet.draft;
  if (!draft || !draft.agent || state.modelSheet.phase === "CLOSED") return null;
  const agent = draft.agent;
  const tab = String(provider || visibleSheetProvider() || "").trim();
  if (!tab) return null;
  // Loaded, loading or failed: the Retry / Refresh buttons ask again.
  if (sheetProviderLoads(agent)[tab]) return null;
  const entry = pickerProviderTabs(agent).find((t) => t.id === tab);
  if (entry && entry.status === "error") return null;
  return loadModelsForAgent(agent, { provider: tab, loading: true }).then(function () {
    // The operator opened this tab before its rows existed: stage it now,
    // exactly as a click on a loaded tab does.
    if (state.modelSheet.pendingStageProvider !== tab) return;
    const draftNow = state.modelSheet.draft;
    const rows = (state.modelOptionsByAgent[agent] || []).filter((m) => m.provider === tab);
    if (draftNow && draftNow.agent === agent && draftNow.provider === tab && rows.length) {
      stageModelSheetProvider(tab);
    } else if (!rows.length) {
      renderModelSheet();
    }
  });
}

function stageModelSheetProvider(provider) {
  if (!["OPEN_CLEAN", "OPEN_DIRTY", "OPEN_ERROR"].includes(state.modelSheet.phase))
    return false;
  const agent = state.modelSheet.draft.agent;
  // A click asks that provider (once per opening; never the others).
  ensureSheetProviderLoaded(provider);
  const models = (state.modelOptionsByAgent[agent] || [])
    .filter((m) => m.provider === provider);
  if (!models.length) {
    // Nothing to stage yet: show the tab (loading, or the server's error)
    // and stage it when its rows arrive.
    const known = pickerProviderTabs(agent).some((t) => t.id === provider);
    if (!known) return false;
    state.modelSheet.draft.provider = provider;
    state.modelSheet.pendingStageProvider = provider;
    renderModelSheet();
    return true;
  }
  state.modelSheet.pendingStageProvider = "";
  const row = models.find((m) => m.id === state.modelSheet.draft.alias) ||
    models.find((m) => m.isDefault) || models[0];
  state.modelSheet.draft = cloneModelSelection({
    ...state.modelSheet.draft,
    provider,
    alias: row.id,
    wire_id: row.wireId,
    reasoning: (typeof reasoningForSelectedModel === "function"
      ? reasoningForSelectedModel(row.reasoning, state.modelSheet.draft.agent, row)
      : null),
    speed: supportedDraftValue(row.speed, state.modelSheet.draft.speed),
  });
  setModelSheetPhase("OPEN_DIRTY");
  return true;
}

function stageDefaultModel() {
  const draft = state.modelSheet.draft;
  const models = state.modelOptionsByAgent[draft.agent] || [];
  // "Use default model" means the AGENT's saved default model.
  const row = agentDefaultRow(draft.agent, models);
  if (!row || state.modelSheet.phase === "APPLYING") return false;
  state.modelSheet.draft = cloneModelSelection({
    agent: draft.agent,
    provider: row.provider,
    alias: row.id,
    wire_id: row.wireId,
    reasoning: row.effortSelected,
    speed: supportedDraftValue(row.speed, null),
  });
  setModelSheetPhase("OPEN_DIRTY");
  return true;
}

function renderModelSheet() {
  const n = modelSheetNodes();
  if (!n.sheet) return;
  ensureModelSheetState();
  const phase = state.modelSheet.phase;
  const applying = phase === "APPLYING";
  const draft = state.modelSheet.draft;
  n.sheet.dataset.phase = phase;
  n.sheet.setAttribute("aria-busy", applying ? "true" : "false");
  renderAgentCard();

  if (n.badgeAgent) {
    n.badgeAgent.textContent = draft.agent || "";
    n.badgeAgent.style.display = draft.agent ? "inline-block" : "none";
  }
  if (n.badgeModel) {
    n.badgeModel.textContent = draft.alias || "";
    n.badgeModel.style.display = draft.alias ? "inline-block" : "none";
  }

  // Ensure current tab panel is rendered cleanly
  if (state.modelSheet.activeTab === "agent") {
    if (n.tabAgent) { n.tabAgent.classList.add("is-active"); n.tabAgent.setAttribute("aria-selected", "true"); }
    if (n.tabModel) { n.tabModel.classList.remove("is-active"); n.tabModel.setAttribute("aria-selected", "false"); }
    if (n.panelAgent) { n.panelAgent.removeAttribute("hidden"); n.panelAgent.classList.add("is-active"); }
    if (n.panelModel) { n.panelModel.setAttribute("hidden", ""); n.panelModel.classList.remove("is-active"); }
  } else {
    if (n.tabModel) { n.tabModel.classList.add("is-active"); n.tabModel.setAttribute("aria-selected", "true"); }
    if (n.tabAgent) { n.tabAgent.classList.remove("is-active"); n.tabAgent.setAttribute("aria-selected", "false"); }
    if (n.panelModel) { n.panelModel.removeAttribute("hidden"); n.panelModel.classList.add("is-active"); }
    if (n.panelAgent) { n.panelAgent.setAttribute("hidden", ""); n.panelAgent.classList.remove("is-active"); }
  }

  const allModels = state.modelOptionsByAgent[draft.agent] || [];
  state.modelSheet.searchByAgent = state.modelSheet.searchByAgent || {};
  const query = String(state.modelSheet.searchByAgent[draft.agent] || "");
  state.modelSheet.search = query;
  // Provider tabs come from the picker summary (server-decided, with their
  // status); before it arrives, from the rows the page holds.
  const tabs = typeof pickerProviderTabs === "function" ? pickerProviderTabs(draft.agent) : [];
  const providerNames = tabs.length
    ? tabs.map((t) => t.id)
    : [...new Set(allModels.map((m) => m.provider || "Other"))];
  // With no tab known yet, the selection keeps its own provider.
  if (providerNames.length && !providerNames.includes(draft.provider))
    draft.provider = providerNames[0];
  const tab = tabs.find((t) => t.id === draft.provider) || null;
  const load = (typeof sheetProviderLoads === "function" ? sheetProviderLoads(draft.agent) : {})[draft.provider] || null;
  const visible = allModels.filter((m) => (m.provider || "Other") === draft.provider);
  const loading = (!!load && load.phase === "loading") ||
    (state.modelLoadingAgent === draft.agent && !visible.length);
  // The server's own words for a tab that cannot list models: a missing key
  // (from the summary) or what its request answered (no_models / unreachable).
  const tabError = (tab && tab.status === "error" && tab.error) ||
    (load && load.error) || null;
  const loadFailed = !!(load && load.transport) || (!tabs.length &&
    !!(state.modelLoadError && state.modelLoadError.agent === draft.agent));
  if (n.search) {
    if (n.search.value !== query) n.search.value = query;
    n.search.disabled = applying;
    n.search.hidden = false;
  }
  const refreshBtn = document.getElementById("amp-refresh-models");
  if (refreshBtn) {
    refreshBtn.disabled = applying || loading;
    refreshBtn.classList.toggle("is-refreshing", loading);
    refreshBtn.title = loading ? "Discovering models…" : "Refresh models for this provider";
    if (!refreshBtn.dataset.bound) {
      refreshBtn.dataset.bound = "1";
      refreshBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        const currentDraft = state.modelSheet && state.modelSheet.draft;
        const provider = typeof visibleSheetProvider === "function" ? visibleSheetProvider() : "";
        if (currentDraft && currentDraft.agent && provider) {
          loadModelsForAgent(currentDraft.agent, { provider, force: true, loading: true });
        }
      });
    }
  }
  n.sheet.setAttribute("aria-busy", (applying || loading) ? "true" : "false");
  const catalog = window.AgentModelCatalogView;
  const noticeHtml = tabError && tabError.message
    ? '<p class="amp-empty amp-provider-error" role="alert" data-error-code="' +
      escapeHtml(String(tabError.code || "")) + '">' + escapeHtml(tabError.message) + "</p>"
    : "";
  const emptyHtml = loading
    ? '<p class="amp-empty amp-provider-loading">Loading models…</p>'
    : (noticeHtml
      ? ""
      : (query.trim() && visible.length
        ? '<p class="amp-empty">No models match this search.</p>'
        : (loadFailed
          ? '<p class="amp-empty">Could not load models.' +
            '<button type="button" id="models-card-retry">Retry</button></p>'
          : '<p class="amp-empty">No models advertised for this provider.</p>')));
  if (catalog) {
    const rendered = catalog.renderModels({
      providersHost: n.providers,
      modelsHost: n.models,
      models: allModels,
      providers: tabs,
      provider: draft.provider,
      selectedModel: draft.alias,
      query,
      disabled: applying,
      emptyHtml,
      noticeHtml,
    });
    if (rendered.provider) draft.provider = rendered.provider;
  }
  const modelsRetry = n.sheet.querySelector("#models-card-retry");
  if (modelsRetry) {
    modelsRetry.addEventListener("click", function (e) {
      e.stopPropagation();
      const provider = typeof visibleSheetProvider === "function" ? visibleSheetProvider() : "";
      loadModelsForAgent(draft.agent, provider ? { provider, loading: true } : { loading: true });
    });
  }

  const row = allModels.find((m) => m.id === draft.alias);
  const reasoning = (row && row.reasoning) || (typeof normalizeAdvancedCapability === "function" ? normalizeAdvancedCapability(null, "reasoning") : null);
  if (n.reasoningSection) {
    n.reasoningSection.hidden = !reasoning || state.modelSheet.activeTab !== "model";
    if (!reasoning && n.reasoningHint) n.reasoningHint.hidden = true;
    if (reasoning) {
      if (!draft.reasoning || draft.reasoning !== "default" && !reasoning.values.includes(draft.reasoning)) {
        draft.reasoning = row ? row.effortSelected : null;
      }
      const index = reasoning.values.indexOf(draft.reasoning);
      n.reasoning.innerHTML = `<option value="-1">Default</option>` + reasoning.values.map((value, i) =>
        `<option value="${i}">${escapeHtml(value.charAt(0).toUpperCase() + value.slice(1))}</option>`
      ).join("");
      n.reasoning.value = String(index);
      n.reasoning.disabled = applying || !reasoning.values.length;
      // Surface the "not configured" state. `applies` is the server's verdict
      // on whether the assembled request will carry the selected level: false
      // means the wire omits it and the provider's own default is used. It is
      // undefined for rows whose path makes no such claim (external engines,
      // legacy shapes), which keep the plain selector they always had.
      if (n.reasoningHint) {
        if (reasoning.applies === false) {
          n.reasoningHint.textContent =
            "Effort levels aren't confirmed for this model, so no reasoning level is sent and the provider applies its own default. Configure this model's effort list to control it.";
          n.reasoningHint.hidden = false;
        } else {
          n.reasoningHint.hidden = true;
        }
      }
    }
  }
  const speed = row && row.speed;
  if (n.speedSection) {
    n.speedSection.hidden = !(speed && speed.applyable);
  }
  if (n.speeds) {
    n.speeds.innerHTML = speed ? speed.values.map((value) =>
      `<button type="button" role="radio" class="amp-speed${value === draft.speed ? " is-selected" : ""}"` +
      ` aria-checked="${value === draft.speed}" data-speed="${escapeHtml(value)}" ${applying ? "disabled" : ""}>` +
      `${escapeHtml(value)}</button>`
    ).join("") : "";
  }

  n.error.textContent = state.modelSheet.error || "";
  n.error.hidden = phase !== "OPEN_ERROR";
  // Long external-engine catalogues (notably Codex) can place the alert below
  // the independently scrolling sheet body while the sticky Save row remains
  // visible. Bring every rejection into view so a failed switch never looks
  // like a silent click.
  if (phase === "OPEN_ERROR" && typeof n.error.scrollIntoView === "function")
    n.error.scrollIntoView({ block: "nearest", inline: "nearest" });
  n.save.disabled = applying || !["OPEN_DIRTY", "OPEN_ERROR"].includes(phase);
  n.save.textContent = applying ? "Saving…" : "Save";
  n.close.disabled = applying;
  n.useDefault.disabled = applying;
  if (n.useSetDefault) {
    n.useSetDefault.disabled = applying || !draft || !draft.agent;
  }
  n.discard.disabled = applying;
  n.sheet.querySelectorAll(".agent-card button").forEach((button) => {
    button.disabled = applying;
  });
  if (typeof agentSelect !== "undefined" && agentSelect)
    agentSelect.disabled = applying || !!state.monitorReadOnly;
  if (typeof modelSelect !== "undefined" && modelSelect)
    modelSelect.disabled =
      applying || !!state.monitorReadOnly || !(state.models || []).length;
}
