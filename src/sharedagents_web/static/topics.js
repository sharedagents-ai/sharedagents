// Topics page — Discord-style navigation. Fetches /api/topics and renders two
// panes: a channel rail (each Telegram forum topic as a #channel) and a messages
// pane (each conversation in the selected channel as a Discord-style message row
// that deep-links to /?session=<id>). Pure read view — topics are authored in
// Telegram; this page only surfaces the grouping in a navigable shape.
//
// Layout: desktop shows both panes side-by-side; mobile shows one at a time and
// toggles via the .dc-show-messages class on #topics-app (back arrow returns to
// the channel list — mirrors Discord mobile).
(function () {
  "use strict";

  const BASE =
    document.querySelector('meta[name="sharedagents-base"]')?.content || "";

  const appEl = document.getElementById("topics-app");
  const channelListEl = document.getElementById("dc-channel-list");
  const msgListEl = document.getElementById("dc-msg-list");
  const channelTitleEl = document.getElementById("dc-channel-title");
  const msgCountEl = document.getElementById("dc-msg-count");
  const backBadgeEl = document.getElementById("dc-back-badge");
  const totalsEl = document.getElementById("topics-totals");
  const updatedEl = document.getElementById("topics-updated");
  const connEl = document.getElementById("topics-conn");
  const filterEl = document.getElementById("topics-filter");

  let lastTopics = [];
  let lastCapped = false;
  let lastForumSync = "";

  // Per-topic emoji (mirrors the sidebar quick-jump in app.js): keyword map +
  // deterministic hash fallback so any topic gets a stable glyph.
  const TOPIC_EMOJI_MAP = [
    [/\bgeneral\b/, "💬"], [/web ?ui|webui|\bweb\b|frontend|ui\b/, "🖥️"],
    [/health|medical|doctor/, "🩺"], [/market|trad|order|oms|portfolio|risk/, "📈"],
    [/docs?|research/, "📚"], [/telegram/, "✈️"], [/gpu|cuda|vllm/, "🎮"],
    [/shared ?data|database|\bdata\b/, "🗄️"], [/sharedagents|claw/, "🐾"],
    [/\btest|qa\b/, "🧪"], [/claude/, "🤖"], [/codex|openai/, "🧠"],
    [/voice|audio|realtime/, "🎙️"], [/bug|fix|error|debug/, "🐛"],
    [/deploy|infra|ops|server/, "⚙️"], [/news|tweet|social/, "📰"],
    [/dashboard|chart/, "📊"], [/cursor/, "🖱️"], [/self.?improv/, "🔁"],
    [/worker/, "👷"], [/agent/, "🛰️"],
  ];
  const TOPIC_EMOJI_FALLBACK = ["💬", "📌", "🗂️", "🔖", "📁", "🏷️", "📨", "🧭", "🔹", "✳️", "📋", "🗨️"];
  function topicEmoji(name) {
    const s = String(name || "").toLowerCase();
    for (const pair of TOPIC_EMOJI_MAP) if (pair[0].test(s)) return pair[1];
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return TOPIC_EMOJI_FALLBACK[h % TOPIC_EMOJI_FALLBACK.length];
  }
  let filterText = "";
  let selectedTopic = null; // channel name, or null until first render picks one
  let pollTimer = null;

  // ── helpers ──────────────────────────────────────────────────────────
  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function isPlaceholderTitle(t) {
    return typeof t === "string" && t.startsWith("untitled-");
  }

  function formatAge(iso) {
    if (!iso) return "";
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return "";
    const secs = Math.max(0, (Date.now() - t) / 1000);
    if (secs < 60) return "just now";
    const mins = Math.floor(secs / 60);
    if (mins < 60) return mins + "m ago";
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return hrs + "h ago";
    const days = Math.floor(hrs / 24);
    if (days < 30) return days + "d ago";
    return new Date(t).toLocaleDateString();
  }

  // Calendar-day label for the date divider ("Today" / "Yesterday" / full date).
  function dayKey(iso) {
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return "unknown";
    const d = new Date(t);
    return d.getFullYear() + "-" + d.getMonth() + "-" + d.getDate();
  }
  function dayLabel(iso) {
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return "Earlier";
    const d = new Date(t);
    const now = new Date();
    const dk = d.getFullYear() + "-" + d.getMonth() + "-" + d.getDate();
    const today = now.getFullYear() + "-" + now.getMonth() + "-" + now.getDate();
    const y = new Date(now.getTime() - 86400000);
    const yk = y.getFullYear() + "-" + y.getMonth() + "-" + y.getDate();
    if (dk === today) return "Today";
    if (dk === yk) return "Yesterday";
    return d.toLocaleDateString(undefined, {
      year: "numeric", month: "long", day: "numeric",
    });
  }

  // Deterministic avatar color from a string (stable per author).
  const AVATAR_COLORS = [
    "#5aff8a", "#8ab4ff", "#ffb454", "#ff7b9c",
    "#c792ea", "#4dd0e1", "#f78c6c", "#82d96b",
  ];
  function avatarColor(seed) {
    let h = 0;
    const s = String(seed || "?");
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) & 0xffffffff;
    return AVATAR_COLORS[Math.abs(h) % AVATAR_COLORS.length];
  }
  function initial(s) {
    const c = String(s || "?").trim().replace(/^[#@]/, "");
    return (c.charAt(0) || "?").toUpperCase();
  }

  function setConn(state) {
    if (!connEl) return;
    connEl.className = "mon-conn mon-conn-" + (state || "unknown");
  }

  function authorFor(s) {
    return s.agent || (s.origin === "gateway" ? "tui" : "web");
  }
  function roleFor(s) {
    // Discord-style role tag: where the conversation lives.
    return s.origin === "gateway" ? (s.agent || "tui") : "web";
  }

  // ── filtering ────────────────────────────────────────────────────────
  // Returns topics (optionally narrowed to sessions matching the filter).
  function visibleTopics() {
    if (!filterText) return lastTopics.slice();
    const q = filterText.toLowerCase();
    const out = [];
    for (const topic of lastTopics) {
      const nameHit = (topic.topic || "").toLowerCase().includes(q);
      const sessions = (topic.sessions || []).filter(function (s) {
        return (s.title || "").toLowerCase().includes(q);
      });
      if (nameHit) {
        out.push(topic); // whole channel matches by name → keep all sessions
      } else if (sessions.length) {
        out.push(Object.assign({}, topic, { sessions: sessions, count: sessions.length }));
      }
    }
    return out;
  }

  function findTopic(topics, name) {
    return topics.find(function (t) { return (t.topic || "general") === name; }) || null;
  }

  // ── channel rail ─────────────────────────────────────────────────────
  function channelRow(topic) {
    const name = topic.topic || "general";
    const count = topic.count != null ? topic.count : (topic.sessions || []).length;
    const active = name === selectedTopic;
    const live = topic.live || (topic.sessions || []).some(function (s) { return s && s.live; });
    const waiting = topic.waiting || (topic.sessions || []).some(function (s) {
      return s && s.state === "waiting";
    });
    const livePill = waiting
      ? '<span class="live-pill waiting-pill" title="Waiting — background subagents still running">Waiting</span>'
      : live
        ? '<span class="live-pill" title="Running — a turn is in progress">Live</span>'
        : "";
    const badge = count
      ? '<span class="dc-channel-badge' + (active ? "" : " muted") + '">' + escapeHtml(count) + "</span>"
      : "";
    // A row wrapper so the "+ new conversation" control is a SIBLING of the
    // channel button (a <button> can't legally nest another button), and so a
    // click on "+" doesn't bubble into channel-select.
    return (
      '<div class="dc-channel-row' + (active ? " active" : "") + '">' +
        '<button class="dc-channel' + (active ? " active" : "") + '" type="button" role="listitem"' +
          ' data-topic="' + escapeHtml(name) + '" title="' + escapeHtml(name) + '">' +
          '<span class="dc-channel-hash dc-channel-emoji">' + topicEmoji(name) + "</span>" +
          '<span class="dc-channel-name">' + escapeHtml(name) + "</span>" +
          livePill +
          badge +
        "</button>" +
        '<button class="dc-channel-new" type="button"' +
          ' data-topic-new="' + escapeHtml(name) + '"' +
          ' title="New conversation in ' + escapeHtml(name) + '"' +
          ' aria-label="New conversation in ' + escapeHtml(name) + '">+</button>' +
      "</div>"
    );
  }

  function renderChannels(topics) {
    if (!channelListEl) return;
    if (!topics.length) {
      channelListEl.innerHTML = filterText
        ? '<div class="dc-channel-empty">No topics match “' + escapeHtml(filterText) + '”</div>'
        : '<div class="dc-channel-empty">No topics yet. Telegram forum topics appear here once synced.</div>';
      return;
    }
    const withSessions = topics.filter(function (t) { return (t.count || (t.sessions || []).length) > 0; });
    const empties = topics.filter(function (t) { return !(t.count || (t.sessions || []).length); });
    let html = "";
    if (withSessions.length) {
      html += '<div class="dc-section-label">Active topics</div>';
      html += withSessions.map(channelRow).join("");
    }
    if (empties.length) {
      html += '<div class="dc-section-label">Empty topics</div>';
      html += empties.map(channelRow).join("");
    }
    channelListEl.innerHTML = html;
  }

  // ── messages pane ────────────────────────────────────────────────────
  function messageRow(s) {
    const title = s.title && !isPlaceholderTitle(s.title) ? s.title : "";
    const author = authorFor(s);
    const role = roleFor(s);
    const roleClass = s.origin === "gateway" ? "" : " web";
    const age = formatAge(s.updated_at);
    const content = title
      ? '<div class="dc-content" title="' + escapeHtml(title) + '">' + escapeHtml(title) + "</div>"
      : '<div class="dc-content untitled">New conversation — open to continue</div>';
    return (
      '<button class="dc-msg" type="button" role="listitem" data-session="' + escapeHtml(s.id) + '"' +
        ' title="Open conversation">' +
        '<span class="dc-avatar" style="background:' + avatarColor(author) + '">' + escapeHtml(initial(author)) + "</span>" +
        '<span class="dc-msg-body">' +
          '<span class="dc-msg-meta">' +
            '<span class="dc-author">' + escapeHtml(author) + "</span>" +
            '<span class="dc-role' + roleClass + '">' + escapeHtml(role) + "</span>" +
            (age ? '<span class="dc-time">' + escapeHtml(age) + "</span>" : "") +
          "</span>" +
          content +
        "</span>" +
      "</button>"
    );
  }

  function renderMessages(topics) {
    if (!msgListEl) return;
    const topic = findTopic(topics, selectedTopic);
    const name = (topic && topic.topic) || selectedTopic || "general";
    if (channelTitleEl) channelTitleEl.textContent = name;
    const sessions = (topic && topic.sessions) || [];
    if (msgCountEl) {
      msgCountEl.textContent = sessions.length
        ? sessions.length + " conversation" + (sessions.length === 1 ? "" : "s")
        : "";
    }
    if (!sessions.length) {
      msgListEl.innerHTML =
        '<div class="dc-empty">' +
          '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>' +
          "<div>No conversations in <b>#" + escapeHtml(name) + "</b> yet.</div>" +
          "<div>Start one from chat and bind it to this topic in Telegram.</div>" +
        "</div>";
      return;
    }
    // Date-divider grouping (sessions already newest-first from the API).
    let html = "";
    let lastDay = null;
    for (const s of sessions) {
      const dk = dayKey(s.updated_at);
      if (dk !== lastDay) {
        html += '<div class="dc-divider">' + escapeHtml(dayLabel(s.updated_at)) + "</div>";
        lastDay = dk;
      }
      html += messageRow(s);
    }
    msgListEl.innerHTML = html;
  }

  // ── render orchestration ─────────────────────────────────────────────
  function render() {
    const topics = visibleTopics();
    // Keep the selected channel valid; otherwise fall back to the first visible.
    if (!findTopic(topics, selectedTopic)) {
      selectedTopic = topics.length ? (topics[0].topic || "general") : null;
    }
    renderChannels(topics);
    renderMessages(topics);

    if (totalsEl) {
      const tcount = lastTopics.length;
      const scount = lastTopics.reduce(function (n, t) {
        return n + (t.count || (t.sessions || []).length);
      }, 0);
      totalsEl.textContent =
        tcount + " topic" + (tcount === 1 ? "" : "s") +
        " · " + scount + " conversation" + (scount === 1 ? "" : "s") +
        (lastCapped ? " · capped" : "");
    }
    // Back-arrow unread-style badge = conversations in the current channel.
    if (backBadgeEl) {
      const topic = findTopic(topics, selectedTopic);
      const n = topic ? (topic.count || (topic.sessions || []).length) : 0;
      if (n > 0) { backBadgeEl.textContent = n > 99 ? "99+" : String(n); backBadgeEl.hidden = false; }
      else backBadgeEl.hidden = true;
    }
  }

  function selectChannel(name) {
    if (!name) return;
    selectedTopic = name;
    render();
    // Mobile: slide to the messages pane.
    if (appEl) appEl.classList.add("dc-show-messages");
  }

  function showChannels() {
    if (appEl) appEl.classList.remove("dc-show-messages");
  }

  // ── data ─────────────────────────────────────────────────────────────
  function load() {
    return fetch(BASE + "/api/topics?full=1", { credentials: "same-origin" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        lastTopics = (data && data.topics) || [];
        lastCapped = !!(data && data.capped);
        lastForumSync = (data && data.forum_topics_updated_at) || "";
        setConn("ok");
        if (updatedEl) {
          updatedEl.textContent = lastForumSync
            ? new Date().toLocaleTimeString() + " · forum " + lastForumSync.slice(0, 19)
            : new Date().toLocaleTimeString();
        }
        render();
      })
      .catch(function (err) {
        setConn("down");
        if (!lastTopics.length && channelListEl) {
          channelListEl.innerHTML =
            '<div class="dc-channel-empty">Failed to load topics: ' +
            escapeHtml(err.message || String(err)) + "</div>";
        }
      });
  }

  function openSession(sid) {
    if (sid) window.location.href = BASE + "/?session=" + encodeURIComponent(sid);
  }

  // Create a web-local topic (no Telegram needed). It appears as an empty
  // channel until a conversation is started in it. POSTs to /api/topics, then
  // reloads and selects the new channel.
  async function createTopic() {
    const raw = await promptCard({
      title: "New topic name",
      okLabel: "Create",
    });
    if (raw == null) return; // cancelled
    const name = raw.trim();
    if (!name) return;
    fetch(BASE + "/api/topics", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name }),
    })
      .then(function (r) {
        return r.json().then(function (d) { return { ok: r.ok, d: d }; });
      })
      .then(function (res) {
        if (!res.ok) {
          alertCard({
            title: "Could not add topic",
            message: (res.d && res.d.error) || "unknown error",
          });
          return;
        }
        const created = (res.d && res.d.topic) || name;
        return load().then(function () { selectChannel(created); });
      })
      .catch(function (err) {
        alertCard({
          title: "Could not add topic",
          message: err.message || String(err),
        });
      });
  }

  // Navigate to the main chat view with a request to create a fresh session
  // bound to this topic (app.js handles ?new&topic= on load).
  function newConversation(name) {
    const topic = (name || "").trim();
    window.location.href =
      BASE + "/?new=1" + (topic ? "&topic=" + encodeURIComponent(topic) : "");
  }

  // ── bindings ─────────────────────────────────────────────────────────
  function bindChannels() {
    if (!channelListEl) return;
    channelListEl.addEventListener("click", function (e) {
      // "+" → open a NEW chat bound to this topic in the main chat view.
      const nb = e.target.closest(".dc-channel-new");
      if (nb) {
        e.stopPropagation();
        newConversation(nb.getAttribute("data-topic-new"));
        return;
      }
      const ch = e.target.closest(".dc-channel");
      if (ch) selectChannel(ch.getAttribute("data-topic"));
    });
  }

  function bindMessages() {
    if (!msgListEl) return;
    msgListEl.addEventListener("click", function (e) {
      const msg = e.target.closest(".dc-msg");
      if (msg) openSession(msg.getAttribute("data-session"));
    });
  }

  function bindBack() {
    const back = document.getElementById("dc-back");
    if (back) back.addEventListener("click", showChannels);
  }

  function bindFilter() {
    if (!filterEl) return;
    filterEl.addEventListener("input", function () {
      filterText = filterEl.value.trim();
      render();
    });
  }

  function bindRefresh() {
    const btn = document.getElementById("topics-refresh");
    if (btn) btn.addEventListener("click", function () { load(); });
  }

  function bindAddTopic() {
    const btn = document.getElementById("dc-add-topic");
    if (btn) btn.addEventListener("click", createTopic);
  }

  // Navigation drawer + conversations are handled by the shared nav.js
  // (loaded before this script). See templates/_menu.html.

  bindChannels();
  bindMessages();
  bindBack();
  bindFilter();
  bindRefresh();
  bindAddTopic();

  load();
  pollTimer = setInterval(function () { load(); }, 5000);
  window.addEventListener("beforeunload", function () {
    if (pollTimer) clearInterval(pollTimer);
  });
})();
