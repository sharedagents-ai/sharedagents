/* nav.js — navigation-only behaviour for the shared left menu on non-chat tabs.
 *
 * The complete markup lives in templates/_menu.html and is also rendered by
 * the chat page.  Chat owns in-place session/topic actions; this file gives
 * Usage, Tasks, Goals, Agent Monitor, and Topics the same data (sessions, topics,
 * pending approvals) and mobile drawer behaviour without duplicating the
 * component.  The pending-approvals panel mirrors sessions.js exactly so the
 * PENDING YOUR DECISION section renders identically on every tab. */
(function () {
  "use strict";
  const BASE =
    document.querySelector('meta[name="sharedagents-base"]')?.content || "";

  function esc(v) {
    return String(v ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  const AGENT_DOT_COLORS = {
    main: "#58a6ff", claude: "#d29922", codex: "#f85149",
    antigravity: "#bc8cff", cursor: "#79c0ff", "self-improver": "#5aff8a",
  };
  const dot = (a) => AGENT_DOT_COLORS[a] || "var(--text-muted)";
  // Per-topic emoji — MUST stay byte-identical to sessions.js (the New Chat
  // sidebar is the keeper; tests/web/test_sidebar_topic_emoji.js enforces
  // parity). The old short map + single 📌 fallback made the Usage / Tasks /
  // Goals / Agent Monitor / Topics sidebar show different icons than New Chat for the
  // same topic. Unknown topics fall back to the same deterministic hash.
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
  function isPlaceholder(t) {
    return typeof t === "string" && (t === "(untitled)" || t.startsWith("untitled-"));
  }
  function displayTitle(t) { return (t && !isPlaceholder(t)) ? t : "New Chat"; }

  function relAge(iso) {
    if (!iso) return "";
    const then = Date.parse(iso);
    if (Number.isNaN(then)) return "";
    const s = Math.max(0, (Date.now() - then) / 1000);
    if (s < 60) return "now";
    const m = Math.floor(s / 60); if (m < 60) return m + "m";
    const h = Math.floor(m / 60); if (h < 24) return h + "h";
    const d = Math.floor(h / 24); if (d < 7) return d + "d";
    const w = Math.floor(d / 7); if (w < 5) return w + "w";
    const mo = Math.floor(d / 30); if (mo < 12) return mo + "mo";
    return Math.floor(d / 365) + "y";
  }

  const BUCKETS = ["Today", "Yesterday", "Previous 7 Days", "Previous 30 Days", "Older"];
  function bucket(iso) {
    const t = iso ? Date.parse(iso) : NaN;
    if (Number.isNaN(t)) return 4;
    const now = new Date();
    const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const day = 86400000;
    if (t >= start) return 0;
    if (t >= start - day) return 1;
    if (t >= start - 7 * day) return 2;
    if (t >= start - 30 * day) return 3;
    return 4;
  }

  function init() {
    const menuBtn = document.getElementById("mon-menu-btn");
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("sidebar-backdrop");
    const sessionList = document.getElementById("session-list");
    const topicList = document.getElementById("topic-list");
    const topicSection = document.getElementById("sidebar-topics");
    const newChatBtn = document.getElementById("new-chat-btn");
    const settingsBtn = document.getElementById("settings-btn");
    if (!sidebar) return;
    // Agent Monitor already has its own 1s session poll; do not also hammer
    // GET /api/approvals on a timer while that tab is open. Other non-chat
    // tabs keep the 5s sidebar refresh. A one-shot still fills the pinned
    // desktop PENDING section.
    const menuPage =
      (sidebar.getAttribute && sidebar.getAttribute("data-menu-page")) || "";
    const isAgentMonitor = menuPage === "agents";
    const isTasks = menuPage === "tasks";
    const desktopMenu = window.matchMedia
      ? window.matchMedia("(min-width: 769px)")
      : null;

    function renderSessions(sessions) {
      if (!sessionList) return;
      if (!sessions.length) {
        sessionList.innerHTML = '<div class="session-empty">No conversations yet</div>';
        return;
      }
      sessions.sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));
      let last = -1;
      const rows = [];
      for (const s of sessions) {
        const b = bucket(s.updated_at);
        if (b !== last) {
          rows.push('<div class="session-group">' + BUCKETS[b] + "</div>");
          last = b;
        }
        const agent = s.agent || (s.origin === "gateway" ? "tui" : "");
        const tag = s.origin === "gateway"
          ? '<span class="session-origin" title="Started outside the web">' + esc(s.agent || "tui") + "</span>"
          : "";
        const title = displayTitle(s.title);
        const age = relAge(s.updated_at);
        const a = s.activity && typeof s.activity === "object" ? s.activity : null;
        const live = a && ["question", "permission", "operator_paused"].includes(a.attention)
          ? '<span class="live-pill waiting-pill">' + esc((a.display || {}).label || "Needs input") + '</span>'
          : a && a.status === "needs_input"
            ? '<span class="live-pill waiting-pill">' + esc((a.display || {}).label || "Needs input") + '</span>'
            : s.state === "waiting"
              ? '<span class="live-pill waiting-pill">Waiting</span>'
              : s.live ? '<span class="live-pill">Live</span>' : "";
        rows.push(
          '<div class="session-item" data-session="' + esc(s.id) + '">' +
            '<span class="session-dot" style="background:' + dot(agent) + '" title="' + esc(agent || "web") + '"></span>' +
            '<span class="session-title" title="' + esc(title) + '">' + esc(title) + "</span>" +
            '<span class="session-meta">' + live + (age ? '<span class="session-time">' + esc(age) + "</span>" : "") + tag + "</span>" +
          "</div>"
        );
      }
      sessionList.innerHTML = rows.join("");
    }

    function renderTopics(topics) {
      if (!topicList || !topicSection) return;
      const meaningful = (topics || []).filter((t) =>
        t && (t.topic || "general") !== "general" && (
          Number(t.count) > 0 || t.latest_id ||
          (Array.isArray(t.sessions) && t.sessions.length > 0)
        )
      );
      if (!meaningful.length) {
        topicSection.hidden = true;
        topicList.innerHTML = "";
        return;
      }
      topicSection.hidden = false;
      topicList.innerHTML = meaningful.map((t) => {
        const name = t.topic || "general";
        const sessions = Array.isArray(t.sessions) ? t.sessions : [];
        const latest = sessions[0] || {};
        const sid = t.latest_id || latest.id || "";
        const count = Number.isFinite(Number(t.count)) ? Number(t.count) : sessions.length;
        const live = t.waiting || t.state === "waiting" || sessions.some((s) => s && s.state === "waiting")
          ? '<span class="live-pill waiting-pill">Waiting</span>'
          : (t.live || sessions.some((s) => s && s.live))
            ? '<span class="live-pill">Live</span>' : "";
        return '<div class="topic-item" data-topic-session="' + esc(sid) + '" title="Open latest in #' + esc(name) + '">' +
          '<span class="topic-hash topic-emoji">' + topicEmoji(name) + "</span>" +
          '<span class="topic-name">' + esc(name) + "</span>" +
          '<span class="topic-actions">' + live + '<span class="topic-count">' + esc(count) + "</span></span>" +
          "</div>";
      }).join("");
    }

    function installSearch() {
      if (!sessionList || document.getElementById("conv-search")) return;
      const wrap = document.createElement("div");
      wrap.className = "conv-search-wrap";
      wrap.innerHTML =
        '<svg class="conv-search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>' +
        '<input id="conv-search" type="search" placeholder="Search chats…" aria-label="Search conversations" autocomplete="off">';
      sessionList.parentNode.insertBefore(wrap, sessionList);
      const input = wrap.querySelector("input");
      input.addEventListener("input", function () {
        const q = input.value.trim().toLowerCase();
        sessionList.querySelectorAll(".session-item, .session-group").forEach((row) => {
          row.style.display = !q || (row.textContent || "").toLowerCase().includes(q) ? "" : "none";
        });
      });
    }

    // ── Pending-approvals panel ────────────────────────
    // Mirror of sessions.js renderPendingApprovals: same rows, count badge and
    // section visibility, so the PENDING YOUR DECISION section on Tasks /
    // Usage / Goals / Agent Monitor / Topics is byte-for-byte the chat sidebar's.
    // Read-only surface: a row deep-links to its conversation; a tool-permission
    // row resolves in place through the authenticated session-bound route.
    const PERM_OPTION_LABELS = {
      allow_once: "Allow",
      allow_always: "Allow always",
      reject_once: "Deny",
      reject_always: "Deny always",
    };
    function permOptionLabel(kind) {
      return PERM_OPTION_LABELS[String(kind)] || String(kind);
    }

    function renderPendingApprovals(approvals) {
      const listEl = document.getElementById("pending-approvals");
      const sectionEl = document.getElementById("sidebar-approvals");
      const countEl = document.getElementById("pending-approvals-count");
      if (!listEl || !sectionEl) return;
      const items = Array.isArray(approvals)
        ? approvals.filter((a) => a && a.session_id)
        : [];
      if (countEl) countEl.textContent = String(items.length);
      if (!items.length) {
        sectionEl.hidden = true;
        listEl.innerHTML = "";
        return;
      }
      sectionEl.hidden = false;
      listEl.innerHTML = items.map((a) => {
        const sid = a.session_id || "";
        const summary = String(a.summary || "Decision requested");
        // A tool-permission request is actionable IN PLACE.
        if (a.type === "tool_permission") {
          const reqId = a.call_id || "";
          const opts = Array.isArray(a.options) && a.options.length
            ? a.options
            : ["allow_once", "reject_once"];
          const buttons = opts
            .map((o) =>
              '<button type="button" class="pending-approval-btn perm-' + esc(o) + '" data-perm-session="' + esc(sid) + '" data-perm-request="' + esc(reqId) + '" data-perm-option="' + esc(o) + '">' + esc(permOptionLabel(o)) + "</button>",
            )
            .join("");
          return '<div class="pending-approval pending-approval-permission" title="' + esc(a.detail || summary) + '">' +
            '<span class="pending-approval-dot" aria-hidden="true"></span>' +
            '<span class="pending-approval-text">' + esc(summary) + "</span>" +
            '<span class="pending-approval-actions">' + buttons + "</span>" +
            "</div>";
        }
        return '<div class="pending-approval" data-approval-session="' + esc(sid) + '" title="Open the conversation awaiting your decision">' +
          '<span class="pending-approval-dot" aria-hidden="true"></span>' +
          '<span class="pending-approval-text">' + esc(summary) + "</span>" +
          "</div>";
      }).join("");
    }

    function loadPendingApprovals() {
      if (isTasks) return;
      // The chat page's conversation stream carries the pending-approvals list
      // in its consolidated `state` frame (sessions.js `_applyState` →
      // renderPendingApprovals()), refreshed every tick. Polling
      // /api/approvals on top of a live stream is redundant; non-chat pages
      // have no stream, so they keep the REST refresh.
      if (typeof uiSseIsOpen === "function" && uiSseIsOpen()) return;
      if (window.MenuState && !MenuState.sidebarVisible()) return;
      fetch(BASE + "/api/approvals", { credentials: "same-origin" })
        .then((r) => r.ok ? r.json() : { approvals: [] })
        .then((d) => renderPendingApprovals(d.approvals || []))
        .catch(() => {});
    }

    function resolvePermission(sid, requestId, optionId) {
      fetch(BASE + "/api/sessions/" + encodeURIComponent(sid) + "/permission", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          requestId: requestId,
          outcome: { outcome: "selected", optionId: optionId },
        }),
      })
        .catch(() => {})
        .then(() => {
          if (isTasks) {
            if (
              typeof document.dispatchEvent === "function" &&
              typeof Event === "function"
            ) {
              document.dispatchEvent(new Event("sharedagents:tasks-refresh"));
            }
          } else {
            loadPendingApprovals();
          }
        });
    }

    function applyTasksSnapshot(snapshot) {
      const data = snapshot || {};
      renderSessions(data.sessions || []);
      renderTopics(data.topics || []);
      renderPendingApprovals(data.approvals || []);
    }

    function loadMenuData() {
      // The Tasks board owns one combined bootstrap request. Reuse it here so
      // the shared sidebar does not independently fan out to sessions/topics.
      if (isTasks) {
        if (window.__saTasksSnapshot) {
          applyTasksSnapshot(window.__saTasksSnapshot);
        }
        return;
      }
      if (sessionList && (!window.MenuState || MenuState.sectionOpen("sidebar-conversations"))) {
        fetch(BASE + "/api/sessions", { credentials: "same-origin" })
          .then((r) => r.ok ? r.json() : { sessions: [] })
          .then((d) => renderSessions(d.sessions || []))
          .catch(() => {});
      }
      if (topicList && (!window.MenuState || MenuState.sectionOpen("sidebar-topics"))) {
        fetch(BASE + "/api/topics", { credentials: "same-origin" })
          .then((r) => r.ok ? r.json() : { topics: [] })
          .then((d) => renderTopics(d.topics || []))
          .catch(() => {});
      }
    }

    if (isTasks) {
      document.addEventListener("sharedagents:tasks-snapshot", (event) => {
        applyTasksSnapshot(event.detail || {});
      });
      if (window.__saTasksSnapshot) {
        applyTasksSnapshot(window.__saTasksSnapshot);
      }
    }

    function open() {
      if (window.MenuState) {
        MenuState.setSidebarVisible(true);
        return;
      }
      if (desktopMenu && desktopMenu.matches) return;
      document.body.classList.add("sidebar-open");
      if (backdrop) backdrop.hidden = false;
      sidebar.setAttribute("aria-hidden", "false");
      sidebar.inert = false;
      if (menuBtn) menuBtn.setAttribute("aria-expanded", "true");
      loadMenuData();
    }
    function close(returnFocus = true) {
      if (window.MenuState) {
        MenuState.setSidebarVisible(false);
        if (returnFocus && menuBtn) menuBtn.focus();
        return;
      }
      document.body.classList.remove("sidebar-open");
      if (backdrop) backdrop.hidden = true;
      sidebar.setAttribute("aria-hidden", "true");
      sidebar.inert = true;
      if (menuBtn) {
        menuBtn.setAttribute("aria-expanded", "false");
        if (returnFocus) menuBtn.focus();
      }
    }
    function syncViewport() {
      if (window.MenuState) {
        const want = desktopMenu && !desktopMenu.matches ? false : MenuState.readPrefs().sidebar;
        MenuState.setSidebarVisible(want, { persist: false });
        return;
      }
      if (desktopMenu && desktopMenu.matches) {
        document.body.classList.remove("sidebar-open");
        if (backdrop) backdrop.hidden = true;
        sidebar.setAttribute("aria-hidden", "false");
        sidebar.inert = false;
        if (menuBtn) menuBtn.setAttribute("aria-expanded", "false");
        loadMenuData();
      } else if (!document.body.classList.contains("sidebar-open")) {
        close(false);
      }
    }

    installSearch();
    if (window.MenuState) {
      if (isTasks) {
        let tasksApplied = false;
        const applyTasks = () => {
          if (tasksApplied) return;
          tasksApplied = true;
          loadMenuData();
        };
        MenuState.registerLoader("sidebar-conversations", applyTasks);
        MenuState.registerLoader("sidebar-topics", applyTasks);
      } else {
        MenuState.registerLoader("sidebar-conversations", () => {
          if (!sessionList) return;
          fetch(BASE + "/api/sessions", { credentials: "same-origin" })
            .then((r) => r.ok ? r.json() : { sessions: [] })
            .then((d) => renderSessions(d.sessions || []))
            .catch(() => {});
        });
        MenuState.registerLoader("sidebar-topics", () => {
          if (!topicList) return;
          fetch(BASE + "/api/topics", { credentials: "same-origin" })
            .then((r) => r.ok ? r.json() : { topics: [] })
            .then((d) => renderTopics(d.topics || []))
            .catch(() => {});
        });
        MenuState.registerLoader("approvals", loadPendingApprovals);
      }
    }
    if (menuBtn) menuBtn.addEventListener("click", () => {
      if (window.MenuState) MenuState.toggleSidebar();
      else open();
    });
    if (backdrop) backdrop.addEventListener("click", close);
    if (newChatBtn) newChatBtn.addEventListener("click", () => {
      // Channels (and other non-chat tabs) load sessions.js for the agent
      // picker, so createSession exists — but there is no composer here.
      // Always leave for the chat page unless #message-input is present.
      const onChat = !!document.getElementById("message-input");
      if (onChat && typeof createSession === "function") {
        createSession(undefined, { explicit: true });
        close(false);
      } else {
        window.location.href = BASE + "/?new=1";
      }
    });
    if (settingsBtn) settingsBtn.addEventListener("click", () => { window.location.href = BASE + "/?settings=1"; });
    if (sessionList) sessionList.addEventListener("click", function (e) {
      const item = e.target.closest(".session-item");
      if (item?.dataset.session) window.location.href = BASE + "/?session=" + encodeURIComponent(item.dataset.session);
    });
    if (topicList) topicList.addEventListener("click", function (e) {
      const item = e.target.closest(".topic-item");
      if (item?.dataset.topicSession) window.location.href = BASE + "/?session=" + encodeURIComponent(item.dataset.topicSession);
    });
    // Pending-approval quick-jump (sidebar) — same behaviour
    // as the chat page: in-place permission resolution or deep-link to the
    // conversation awaiting the decision.
    const approvalsList = document.getElementById("pending-approvals");
    if (approvalsList) {
      approvalsList.addEventListener("click", (e) => {
        const permBtn = e.target.closest("[data-perm-option]");
        if (permBtn) {
          e.stopPropagation();
          resolvePermission(
            permBtn.dataset.permSession,
            permBtn.dataset.permRequest,
            permBtn.dataset.permOption,
          );
          return;
        }
        const row = e.target.closest(".pending-approval");
        if (!row) return;
        const sid = row.dataset.approvalSession;
        if (!sid) return;
        window.location.href = BASE + "/?session=" + encodeURIComponent(sid);
      });
    }
    if (!isTasks) {
      if (!window.MenuState) loadPendingApprovals();
      if (!isAgentMonitor) setInterval(loadPendingApprovals, 5000);
    }
    document.querySelector("#sidebar-topics .section-label-link")?.addEventListener("click", (e) => e.stopPropagation());
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && document.body.classList.contains("sidebar-open")) close();
    });
    if (desktopMenu) {
      if (desktopMenu.addEventListener) desktopMenu.addEventListener("change", syncViewport);
      else if (desktopMenu.addListener) desktopMenu.addListener(syncViewport);
    }
    syncViewport();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
