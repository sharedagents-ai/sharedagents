(function () {
  const BASE =
    document.querySelector('meta[name="sharedagents-base"]')?.content || "";
  const INITIAL_LOOP =
    document.querySelector('meta[name="sharedagents-initial-loop"]')?.content || "";

  const STATUS_ORDER = ["needs_input", "working", "completed"];
  // Backend loop statuses remain running/paused — loops are DELETED when
  // they end, so the board has no archived state.
  const STATUS_LABELS = {
    needs_input: "Needs input",
    working: "Working",
    completed: "Completed",
  };

  const appEl = document.getElementById("loops-app");
  const listEl = document.getElementById("loops-list");
  const detailEl = document.getElementById("loops-detail");
  const totalsEl = document.getElementById("loops-totals");
  const updatedEl = document.getElementById("loops-updated");
  const connEl = document.getElementById("loops-conn");
  const createToggle = document.getElementById("loops-create-toggle");
  const createForm = document.getElementById("loops-create-form");

  let selectedId = INITIAL_LOOP.trim();
  // Status the open detail panel was last rendered for. The poll watches this:
  // when the selected loop transitions (e.g. running → paused) or its
  // `live` flag flips (supervisor died → Stalled), the panel re-renders itself
  // so the badge and available controls update without a click.
  let detailStatus = null;
  let detailLive = null;
  let detailInFlight = null;
  let lastLoops = []; // the FILTERED subset currently rendered
  let allLoops = []; // raw, unfiltered loops from the API
  let pollTimer = null;
  // D1: sid of the session minted by the two-step create, cached so a failed
  // loop-create step can retry without minting another orphan session.
  // The cache is keyed by agent+topic and survives form close
  // — it is cleared only on successful loop creation or when the key changes.
  let newSessionId = null;
  let newSessionKey = null;
  // Agent + model picker (shared AgentModelPicker class — the same roster,
  // model catalogue, defaults and localStorage keys the chat composer picker
  // uses). Null when the shared script failed to load.
  let agentModelPicker = null;

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function api(path, opts) {
    return fetch(BASE + path, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      ...opts,
    }).then(async function (r) {
      const data = await r.json().catch(function () { return {}; });
      if (!r.ok) {
        const fallback = r.status ? "HTTP " + r.status : "request failed";
        const err = new Error(data.error || r.statusText || fallback);
        err.status = r.status;
        err.data = data;
        throw err;
      }
      return data;
    });
  }

  function setConnection(ok) {
    if (!connEl) return;
    connEl.classList.remove("mon-conn-live", "mon-conn-down", "mon-conn-unknown");
    connEl.classList.add(ok ? "mon-conn-live" : "mon-conn-down");
  }

  function statusBadge(status, live, cardKey) {
    const key = String(cardKey || status || "unknown");
    let label = STATUS_LABELS[key] || status || "unknown";
    let extra = "";
    // A running loop whose supervisor heartbeat has gone stale is no longer
    // being driven — render it "Stalled" (muted, no pulse) so this page agrees
    // with the control endpoint's live flag.
    if (key === "working" && status === "running" && live === false) {
      label = "Stalled";
      extra = " is-stalled";
    }
    return (
      '<span class="badge badge-' + escapeHtml(key) + extra + '">' +
        '<i class="loop-dot"></i>' +
        '<span>' + escapeHtml(label) + "</span>" +
      "</span>"
    );
  }

  function cardFor(loop) {
    const status = String((loop || {}).status || "");
    if (status === "paused") return "needs_input";
    if (status === "running") {
      // Idle between ticks is Completed. Working is only an in-flight turn.
      return loop && loop.in_flight ? "working" : "completed";
    }
    // Archived state no longer exists (loops are deleted when
    // they end) — a straggler row reads as idle-completed, never archived.
    if (status === "archived") return "completed";
    return "needs_input";
  }

  // Status-aware controls. A running loop can be paused; a paused loop can be
  // resumed; the terminal action is Delete — the loop is permanently removed
  // (record + run history), never archived.
  function loopActions(status) {
    const btn = function (act, label, extra) {
      return '<button type="button" class="loop-action-btn' + (extra ? " " + extra : "") + '" data-act="' + act + '">' + label + "</button>";
    };
    const out = [];
    if (status === "running") out.push(btn("pause", "Pause"));
    if (status === "paused") out.push(btn("resume", "Resume"));
    out.push(btn("delete", "Delete", "is-danger"));
    return out.join("");
  }

  function sessionHref(sessionId) {
    // Reuse the canonical chat deep-link contract so a loop
    // opens the conversation that is actually driving it.
    return BASE + "/?session=" + encodeURIComponent(String(sessionId || ""));
  }

  function sessionLink(loop, label, extraClass) {
    if (!loop.session_id) {
      return '<span class="loop-session-unavailable">Session not available</span>';
    }
    return (
      '<a class="loop-session-link ' + escapeHtml(extraClass || "") + '" data-session-link ' +
        'href="' + escapeHtml(sessionHref(loop.session_id)) + '" ' +
        'title="Open session ' + escapeHtml(loop.session_id) + '">' +
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">' +
          '<path d="M8 5h11v11"/><path d="M19 5 5 19"/>' +
        '</svg>' +
        '<span>' + escapeHtml(label || "Open session") + '</span>' +
      '</a>'
    );
  }

  function formatDuration(milliseconds) {
    const secs = Math.max(0, Math.floor(Number(milliseconds || 0) / 1000));
    if (secs < 60) return "<1m";
    const mins = Math.floor(secs / 60);
    if (mins < 60) return mins + "m";
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) {
      const remainder = mins % 60;
      return hrs + "h" + (hrs < 10 && remainder >= 5 ? " " + remainder + "m" : "");
    }
    const days = Math.floor(hrs / 24);
    if (days < 14) return days + "d";
    const weeks = Math.floor(days / 7);
    if (weeks < 9) return weeks + "w";
    const months = Math.floor(days / 30);
    if (months < 18) return months + "mo";
    return Math.floor(days / 365) + "y";
  }

  function runLabel(loop) {
    const now = Date.now();
    const lastIso = loop.last_run_at || "";
    const nextIso = loop.next_run_at || "";
    const hasLast = lastIso && !Number.isNaN(Date.parse(lastIso));
    const hasNext = nextIso && !Number.isNaN(Date.parse(nextIso));
    const lastPart = hasLast
      ? "ran " + formatDuration(now - Date.parse(lastIso)) + " ago"
      : "never ran";
    if (loop.status === "paused") return hasLast ? lastPart + " · paused" : "paused · never ran";
    if (hasNext) {
      const nextMs = Date.parse(nextIso) - now;
      if (nextMs <= 0) {
        // A scheduled run whose time has passed is
        // overdue — never clamp it to a fake "Next run in <1m".
        return "Overdue · " + lastPart;
      }
      const next = "Next run in " + formatDuration(nextMs);
      return next + " · " + lastPart;
    }
    return hasLast ? lastPart + " · no next run" : "no next run";
  }

  function formatDate(iso) {
    if (!iso) return "-";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString(undefined, {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: true,
    });
  }

  function updateSummaries(loops) {
    const working = loops.filter(function (l) { return cardFor(l) === "working"; }).length;
    const needsInput = loops.filter(function (l) { return cardFor(l) === "needs_input"; }).length;
    if (totalsEl) totalsEl.textContent = working + " working · " + needsInput + " need input";
    if (updatedEl) updatedEl.textContent = loops.length ? "Updated now" : "";
  }

  function renderLoopCard(l) {
    const selected = l.loop_id === selectedId ? " selected" : "";
    const openLabel = l.status === "running" ? "Open loop session" : "Open session";
    return (
      '<article class="loop-card' + selected + '" data-id="' + escapeHtml(l.loop_id) + '">' +
        '<button type="button" class="loop-card-summary" data-loop-details="' + escapeHtml(l.loop_id) + '">' +
          '<span class="loop-card-head">' +
            '<span class="loop-card-title">' +
              '<span class="loop-session-title">' + escapeHtml(l.session_title) + '</span>' +
              '<span class="loop-agent-chip">' + escapeHtml(l.agent_name) + '</span>' +
            '</span>' +
            '<span class="loop-period-chip" title="Runs ' + escapeHtml(l.period_label || "") + '">' +
              escapeHtml(l.period_label || "on schedule") +
            '</span>' +
          '</span>' +
          '<span class="loop-prompt">' + escapeHtml(l.prompt_text) + '</span>' +
          '<span class="loop-card-meta">' +
            statusBadge(l.status, l.live, cardFor(l)) +
            '<span class="loop-run-meta">' + escapeHtml(runLabel(l)) + '</span>' +
          '</span>' +
        '</button>' +
        '<div class="loop-card-footer">' +
          sessionLink(l, openLabel, "is-primary") +
          '<span class="loop-card-actions">' +
            '<button type="button" class="loop-action-btn is-danger" data-act="delete" data-loop-delete="' + escapeHtml(l.loop_id) + '">Delete</button>' +
            '<button type="button" class="loop-details-btn" data-loop-details="' + escapeHtml(l.loop_id) + '">Details</button>' +
          '</span>' +
        '</div>' +
      "</article>"
    );
  }

  function renderList(loops) {
    lastLoops = loops;
    const buckets = Object.fromEntries(STATUS_ORDER.map(function (key) { return [key, []]; }));
    loops.forEach(function (loop) {
      const key = cardFor(loop);
      buckets[key].push(loop);
    });
    listEl.innerHTML = STATUS_ORDER.map(function (key) {
      const rows = buckets[key];
      return '<section class="loop-state-column' + (!rows.length ? ' is-empty' : '') + '" data-card="' + key + '">' +
        '<header class="loop-state-header"><span>' + escapeHtml(STATUS_LABELS[key]) + '</span><span>' + rows.length + '</span></header>' +
        '<div class="loop-state-body">' + (rows.length ? rows.map(renderLoopCard).join("") : '<div class="loop-state-empty">—</div>') + '</div>' +
      '</section>';
    }).join("");

    listEl.querySelectorAll("[data-loop-details]").forEach(function (el) {
      el.addEventListener("click", function () {
        openDetail(el.dataset.loopDetails);
      });
    });
    listEl.querySelectorAll("[data-loop-delete]").forEach(function (el) {
      el.addEventListener("click", function () {
        runCardDelete(el.dataset.loopDelete);
      });
    });
  }

  function renderListError(err) {
    updateSummaries([]);
    listEl.innerHTML =
      '<div class="loops-load-error">' +
        "Failed to load loops: " + escapeHtml(err.message || String(err)) +
      "</div>";
  }

  function detailShell(content) {
    detailEl.hidden = false;
    appEl.classList.add("has-detail");
    detailEl.innerHTML = '<div class="loops-detail-inner">' + content + "</div>";
  }

  function closeDetail() {
    selectedId = "";
    detailStatus = null;
    detailLive = null;
    detailEl.hidden = true;
    detailEl.innerHTML = "";
    appEl.classList.remove("has-detail");
    listEl.querySelectorAll(".loop-card").forEach(function (el) {
      el.classList.remove("selected");
    });
    if (window.location.pathname !== BASE + "/loops") {
      history.replaceState(null, "", BASE + "/loops");
    }
  }

  function kv(label, value, href) {
    const body = href
      ? '<a href="' + escapeHtml(href) + '">' + escapeHtml(value || "-") + "</a>"
      : "<b>" + escapeHtml(value || "-") + "</b>";
    return '<div class="loop-detail-kv"><span>' + escapeHtml(label) + "</span>" + body + "</div>";
  }

  function renderEvents(events) {
    if (!events || !events.length) {
      return '<div class="loops-empty">No events yet.</div>';
    }
    return events.map(function (ev) {
      const payload = ev.payload && Object.keys(ev.payload).length
        ? JSON.stringify(ev.payload, null, 2)
        : "";
      return (
        '<div class="loop-event">' +
          '<div class="loop-event-type">' + escapeHtml(ev.event_type || "event") + "</div>" +
          '<div class="loop-event-meta">' + escapeHtml(formatDate(ev.created_at)) + "</div>" +
          (payload ? '<div class="loop-event-payload">' + escapeHtml(payload) + "</div>" : "") +
        "</div>"
      );
    }).join("");
  }

  async function loadList() {
    try {
      const data = await api("/api/loops");
      allLoops = data.loops || [];
      setConnection(true);
      updateSummaries(allLoops);
       renderList(allLoops);
      return allLoops;
    } catch (err) {
      setConnection(false);
      renderListError(err);
      throw err;
    }
  }

  async function openDetail(loopId) {
    if (!loopId) return;
    selectedId = loopId;
    listEl.querySelectorAll(".loop-card").forEach(function (el) {
      el.classList.toggle("selected", el.dataset.id === loopId);
    });
    detailShell('<div class="loops-loading">Loading loop...</div>');
    try {
      const l = await api("/api/loops/" + encodeURIComponent(loopId));
      setConnection(true);
      detailStatus = l.status;
      detailLive = !!l.live;
      detailInFlight = !!l.in_flight;
      const rootHref = l.session_id ? sessionHref(l.session_id) : "";
      const lastError = l.last_error || "";
      // A direct /loops/<id> link should also make the matching card visible.
      if (!lastLoops.some(function (item) { return item.loop_id === loopId; })) {
        renderList(allLoops);
      }
      detailShell(
        '<div class="loops-detail-header">' +
          '<h2>' + escapeHtml(l.session_title) + '</h2>' +
          '<button type="button" class="loop-detail-close" data-close-detail aria-label="Close loop details" title="Close details">&times;</button>' +
        '</div>' +
         '<div class="loop-detail-status">' + statusBadge(l.status, l.live, cardFor(l)) + '</div>' +
        '<div class="loop-detail-section">' +
          '<h3>What runs</h3>' +
          '<p class="loop-detail-prompt">' + escapeHtml(l.prompt_text) + '</p>' +
        '</div>' +
        '<div class="loop-detail-section">' +
          '<h3>Schedule</h3>' +
          '<div class="loop-detail-grid">' +
            kv("Period", l.period_label || "-") +
            // next_run_at only describes a real next run
            // while the loop is scheduled; paused shows a dash.
            kv("Next run", l.status === "running" ? formatDate(l.next_run_at) : "—") +
            kv("Last ran", formatDate(l.last_run_at)) +
            kv("Runs", String(l.run_count || 0)) +
            kv("Missed runs", String(l.misses || 0)) +
          '</div>' +
        '</div>' +
        '<div class="loop-detail-section">' +
          '<h3>Loop controls</h3>' +
          '<div class="loop-detail-actions">' + loopActions(l.status) + '</div>' +
        '</div>' +
        '<div class="loop-detail-section">' +
          sessionLink(l, l.status === "running" ? "Open loop session" : "Open session", "is-primary is-large") +
        '</div>' +
        '<details class="loop-disclosure">' +
          '<summary><span>Loop details</span><span class="loop-disclosure-hint">IDs &amp; timing</span></summary>' +
          '<div class="loop-disclosure-body">' +
            '<div class="loop-detail-grid">' +
              kv("Agent", l.agent_name) +
              kv("Session", l.session_id || "-", rootHref) +
              (l.bound_model ? kv("Session model", l.bound_model) : "") +
              kv("Loop ID", l.loop_id) +
              // A daily-time loop has no interval_s — show
              // its schedule label (e.g. "daily at 09:30") instead of the
              // misleading "model-paced".
              kv("Schedule", l.interval_s ? formatDuration(l.interval_s * 1000) : (l.cron ? (l.period_label || l.cron) : "model-paced")) +
              kv("Created", formatDate(l.created_at)) +
              kv("Created by", l.actor_name || l.actor_email || "") +
              kv("Lifetime", "No automatic expiry") +
              kv("Budget per run", (l.budget_usd != null && l.budget_usd !== "") ? ("$" + Number(l.budget_usd).toFixed(2)) : "session default") +
            '</div>' +
            '<label class="loops-run-in-option" style="margin-top:10px">' +
              '<input type="checkbox" id="loops-detail-fresh-session" autocomplete="off"' +
              (l.fresh_session_each_run ? " checked" : "") + '>' +
              '<span><strong>New session each run</strong><small>off unless explicitly enabled — reuse the bound conversation</small></span>' +
            '</label>' +
            (lastError ? '<div class="loop-detail-note is-error"><strong>Last error</strong><p>' + escapeHtml(lastError) + '</p></div>' : '') +
          '</div>' +
        '</details>' +
        '<details class="loop-disclosure">' +
          '<summary><span>Activity history</span><span class="loop-disclosure-hint">' + escapeHtml(String((l.events || []).length)) + ' events</span></summary>' +
          '<div class="loop-disclosure-body"><div class="loop-events">' + renderEvents(l.events || []) + '</div></div>' +
        '</details>' +
        '<div class="loop-detail-error" role="alert"></div>'
      );
      detailEl.querySelector("[data-close-detail]")?.addEventListener("click", closeDetail);
      detailEl.querySelectorAll("[data-act]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          runAction(loopId, btn.dataset.act, l.agent_name);
        });
      });
      const freshToggle = detailEl.querySelector("#loops-detail-fresh-session");
      if (freshToggle) {
        freshToggle.addEventListener("change", function () {
          patchFreshSession(loopId, !!freshToggle.checked);
        });
      }
      if (window.location.pathname !== BASE + "/loops/" + encodeURIComponent(loopId)) {
        history.replaceState(null, "", BASE + "/loops/" + encodeURIComponent(loopId));
      }
    } catch (err) {
      setConnection(false);
      detailShell(
        '<button type="button" class="loop-detail-close" data-close-detail aria-label="Close loop details" title="Close details">&times;</button>' +
        '<div class="loops-load-error">Failed to load loop: ' + escapeHtml(err.message || String(err)) + "</div>",
      );
      detailEl.querySelector("[data-close-detail]")?.addEventListener("click", closeDetail);
    }
  }

  function showDetailError(msg) {
    const errEl = detailEl.querySelector(".loop-detail-error");
    if (errEl) errEl.textContent = msg;
  }

  async function patchFreshSession(loopId, enabled) {
    const toggle = detailEl.querySelector("#loops-detail-fresh-session");
    if (toggle) toggle.disabled = true;
    showDetailError("");
    try {
      await api("/api/loops/" + encodeURIComponent(loopId), {
        method: "PATCH",
        body: JSON.stringify({ fresh_session_each_run: !!enabled }),
      });
      await loadList();
      await openDetail(loopId);
    } catch (err) {
      showDetailError("Couldn't update session option: " + (err.message || String(err)));
      if (toggle) toggle.checked = !enabled;
    } finally {
      if (toggle) toggle.disabled = false;
    }
  }

  // Permanently delete the loop: record + run/event history + wakeups.
  async function deleteLoop(loopId) {
    await api("/api/loops/" + encodeURIComponent(loopId), { method: "DELETE" });
    await loadList();
    if (selectedId === loopId) closeDetail();
  }

  async function runAction(loopId, action, agentName) {
    const buttons = detailEl.querySelectorAll("[data-act]");
    buttons.forEach(function (b) { b.disabled = true; });
    showDetailError("");
    try {
      if (action === "delete") {
        await deleteLoop(loopId);
        return; // the loop is gone — there is no detail to re-open
      }
      await api("/api/loops/" + encodeURIComponent(loopId) + "/" + action, {
        method: "POST",
        body: JSON.stringify({ agent_name: agentName || "" }),
      });
      await loadList();
      await openDetail(loopId);
    } catch (err) {
      showDetailError((action === "delete" ? "Delete failed: " : "Action failed: ") + (err.message || String(err)));
    } finally {
      buttons.forEach(function (b) { b.disabled = false; });
    }
  }

  // Card-level Delete: the terminal action right from the portfolio, no
  // detail pane required (mobile-first). Same DELETE endpoint as the detail
  // controls — the loop vanishes from the board (never archived).
  async function runCardDelete(loopId) {
    const btn = listEl.querySelector('[data-loop-delete="' + loopId + '"]');
    if (btn) btn.disabled = true;
    try {
      await deleteLoop(loopId);
    } catch (err) {
      if (btn) btn.disabled = false;
      const cardEl = listEl.querySelector('.loop-card[data-id="' + loopId + '"]');
      const msg = "Delete failed: " + (err.message || String(err));
      if (cardEl) {
        let errEl = cardEl.querySelector(".loop-card-error");
        if (!errEl) {
          errEl = document.createElement("div");
          errEl.className = "loop-card-error";
          cardEl.appendChild(errEl);
        }
        errEl.textContent = msg;
      } else if (selectedId === loopId) {
        showDetailError(msg);
      }
    }
  }

  async function loadSessionOptions() {
    const select = document.getElementById("loops-create-session");
    if (!select) return;
    try {
      const data = await api("/api/sessions");
      const sessions = data.sessions || [];
      if (!sessions.length) {
        select.innerHTML = '<option value="">No sessions yet</option>';
        select.disabled = true;
        return;
      }
      select.innerHTML = sessions.map(function (s) {
        const sid = String(s.id || "");
        const title = String(s.title || "Session");
        return '<option value="' + escapeHtml(sid) + '">' +
          escapeHtml(title + " (" + (s.agent || "main") + ")") +
          "</option>";
      }).join("");
      select.disabled = false;
    } catch (_) {
      select.innerHTML = '<option value="">Sessions unavailable</option>';
      select.disabled = true;
    }
  }

  // Shared AgentModelPicker (agent_model_picker.js): drives the create
  // form's Agent + Model selects from the same roster / per-agent model
  // catalogue / persisted picker preferences the chat page uses. Agent
  // options keep the per-agent description the old loader rendered.
  function initAgentModelPicker() {
    if (typeof AgentModelPicker !== "function") return null;
    const agentSelect = document.getElementById("loops-create-agent");
    const modelSelect = document.getElementById("loops-create-model");
    if (!agentSelect || !modelSelect) return null;
    return new AgentModelPicker({
      agentSelect: agentSelect,
      modelSelect: modelSelect,
      agentLabel: function (name, meta) {
        return meta && meta.description
          ? name + " — " + meta.description
          : name;
      },
    });
  }

  async function loadTopicOptions() {
    // Closed list from /api/topics — same inventory as the sidebar Topics
    // section. Free-text topics are not allowed; "No topic" is the empty
    // optional choice. Drop the catch-all "general" bucket (absence of a
    // topic), matching sessions.js renderTopicList.
    const select = document.getElementById("loops-create-topic");
    if (!select) return;
    const prev = String(select.value || "");
    let names = [];
    try {
      const data = await api("/api/topics");
      const seen = new Set();
      (data.topics || []).forEach(function (t) {
        const name = String((t && t.topic) || "").trim();
        if (!name || name === "general") return;
        const fold = name.toLowerCase();
        if (seen.has(fold)) return;
        seen.add(fold);
        names.push(name);
      });
    } catch (_) {
      names = [];
    }
    const opts = ['<option value="">No topic</option>'];
    names.forEach(function (name) {
      opts.push(
        '<option value="' + escapeHtml(name) + '">' + escapeHtml(name) + "</option>",
      );
    });
    select.innerHTML = opts.join("");
    if (prev && names.some(function (n) { return n === prev; })) {
      select.value = prev;
    } else {
      select.value = "";
    }
  }

  // Web-created loops fire their first run at the creation
  // instant (next_run_at = now), so the hints say the first run starts right
  // away — no interval wait, no jitter.
  const FREQUENCY_HINTS = {
    "30m": "Runs every 30 minutes · first run starts right away.",
    "1h": "Runs hourly · first run starts right away.",
    "2h": "Runs every 2 hours · first run starts right away.",
    "4h": "Runs every 4 hours · first run starts right away.",
    "12h": "Runs every 12 hours · first run starts right away.",
    "1d": "Runs daily · first run starts right away.",
    "7d": "Runs weekly · first run starts right away.",
    custom: "Type any interval the scheduler understands — e.g. 90m, every 2h, 1800. The first run starts right away.",
    time: "Runs once a day at the time you pick · first run starts right away, then daily at that time.",
  };
  const FREQUENCY_VALUES = { "30m": "30m", "1h": "1h", "2h": "2h", "4h": "4h", "12h": "12h", "1d": "1d", "7d": "7d", custom: null, time: null };

  function applyFrequencyPreset(value) {
    const input = document.getElementById("loops-create-interval");
    const hint = document.getElementById("loops-frequency-hint");
    const field = document.getElementById("loops-create-interval-field");
    const timeField = document.getElementById("loops-create-time-field");
    const mapped = FREQUENCY_VALUES[value];
    const isTime = value === "time";
    const isCustom = value === "custom";
    // The interval text box appears ONLY for a custom interval; the specific
    // time input appears ONLY for the time-of-day mode.
    if (field) field.hidden = !isCustom;
    if (timeField) timeField.hidden = !isTime;
    if (input) {
      if (mapped) {
        input.value = mapped;
        input.readOnly = true;
      } else {
        input.readOnly = false;
      }
    }
    if (hint) hint.textContent = FREQUENCY_HINTS[value] || "";
  }

  function bindFrequencyPresets() {
    const preset = document.getElementById("loops-create-frequency");
    const input = document.getElementById("loops-create-interval");
    if (!preset || !input) return;
    preset.addEventListener("change", function () {
      applyFrequencyPreset(preset.value);
    });
    input.addEventListener("input", function () {
      if (preset.value !== "custom") preset.value = "custom";
      input.readOnly = false;
      applyFrequencyPreset("custom");
    });
    applyFrequencyPreset(preset.value);
  }

  function bindRunInToggle() {
    const newFields = document.getElementById("loops-new-session-fields");
    const existingFields = document.getElementById("loops-existing-session-fields");
    document.querySelectorAll('input[name="loops-run-in"]').forEach(function (radio) {
      radio.addEventListener("change", function () {
        const isNew = radio.value === "new";
        newFields.hidden = !isNew;
        existingFields.hidden = isNew;
      });
    });
  }

  // Client-side mirror of loop_service.parse_interval for
  // the web form — known suffixes s/m/h/d + bare seconds (plus the backend's
  // "every N <unit>" clause), range 60s .. 28d. Returns seconds or null.
  function intervalSeconds(text) {
    const t = String(text || "").trim().toLowerCase();
    const every = t.match(/^every\s+(\d+)\s*(s|m|h|d|minutes?|mins?|hours?|hrs?|days?)?$/);
    const plain = t.match(/^(\d+)([smhd])?$/);
    const m = every || plain;
    if (!m) return null;
    const value = Number(m[1]);
    if (!(value > 0)) return null;
    const unitWord = m[2] || (every ? "m" : "s");
    const unit = unitWord[0] === "m" ? "m" : unitWord[0] === "h" ? "h" : unitWord[0] === "d" ? "d" : "s";
    const seconds = value * (unit === "s" ? 1 : unit === "m" ? 60 : unit === "h" ? 3600 : 86400);
    if (seconds < 60 || seconds > 28 * 86400) return null;
    return seconds;
  }

  function bindCreateForm() {
    const errEl = document.getElementById("loops-create-error");
    const closeForm = function () {
      createForm.hidden = true;
      createToggle.setAttribute("aria-expanded", "false");
      errEl.textContent = "";
      // F3: deliberately do NOT clear the retry cache here — closing the form
      // must not orphan a session minted by a failed create. Success (below)
      // and a changed agent+topic key are the only cache invalidations.
    };
    createToggle.addEventListener("click", function () {
      const next = createForm.hidden;
      createForm.hidden = !next;
      createToggle.setAttribute("aria-expanded", String(next));
      if (next) {
        loadSessionOptions();
        loadTopicOptions();
        // Roster fetch may have failed at page boot (gateway down) — retry
        // lazily when the operator actually opens the form.
        if (agentModelPicker && !agentModelPicker.isLoaded()) {
          agentModelPicker.load();
        }
        document.getElementById("loops-create-prompt").focus();
      }
    });
    ["loops-create-close", "loops-create-cancel"].forEach(function (id) {
      document.getElementById(id)?.addEventListener("click", closeForm);
    });

    createForm.addEventListener("submit", async function (ev) {
      ev.preventDefault();
      const submit = createForm.querySelector('button[type="submit"]');
      const promptEl = document.getElementById("loops-create-prompt");
      const agentEl = document.getElementById("loops-create-agent");
      const topicEl = document.getElementById("loops-create-topic");
      errEl.textContent = "";
      const promptText = document.getElementById("loops-create-prompt").value.trim();
      if (!promptText) {
        errEl.textContent = "Write what the loop should run.";
        promptEl.focus();
        return;
      }
      const runIn = createForm.querySelector('input[name="loops-run-in"]:checked')?.value || "new";
      let sessionId;
      if (runIn === "new") {
        const agent = agentModelPicker ? agentModelPicker.getAgent() : (agentEl ? agentEl.value : "");
        if (!agent) {
          errEl.textContent = "Pick an agent for the new session.";
          return;
        }
      } else {
        sessionId = document.getElementById("loops-create-session").value;
        if (!sessionId) {
          errEl.textContent = "Pick an existing session, or choose New session.";
          return;
        }
      }
      // Validate the schedule BEFORE minting anything (F3): an invalid
      // interval/time must never orphan a freshly created session. The POST
      // sends the VALIDATED SECONDS as interval_s or the validated HH:MM as
      // `time` — the backend accepts the
      // same surface, so client-accepted === backend-accepted by construction.
      const scheduleMode = document.getElementById("loops-create-frequency").value;
      let bodySchedule;
      if (scheduleMode === "time") {
        const timeValue = document.getElementById("loops-create-time").value;
        if (!/^\d{2}:\d{2}$/.test(timeValue)) {
          errEl.textContent = "Pick a specific time of day (24-hour HH:MM).";
          document.getElementById("loops-create-time").focus();
          return;
        }
        bodySchedule = { time: timeValue };
      } else {
        const intervalText = document.getElementById("loops-create-interval").value;
        const intervalSecs = intervalSeconds(intervalText);
        if (!intervalSecs) {
          errEl.textContent = "Interval must be 1 minute to 28 days — e.g. 30m, every 2h, 1800.";
          document.getElementById("loops-create-interval").focus();
          return;
        }
        bodySchedule = { interval_s: intervalSecs };
      }
      const freshEach = !!document.getElementById("loops-create-fresh-session")?.checked;
      const budgetRaw = document.getElementById("loops-create-budget")?.value;
      const budgetUsd = budgetRaw === undefined || String(budgetRaw).trim() === ""
        ? null
        : Number(budgetRaw);
      if (budgetUsd != null && (!Number.isFinite(budgetUsd) || budgetUsd < 0 || budgetUsd > 1000)) {
        errEl.textContent = "Budget per run must be between 0 and 1000 USD.";
        document.getElementById("loops-create-budget").focus();
        return;
      }
      // Lock the whole form BEFORE the first await (F3): a double-click must
      // not mint two sessions/loops.
      const controls = Array.from(createForm.querySelectorAll("input, select, textarea, button"));
      controls.forEach(function (c) { c.disabled = true; });
      submit.disabled = true;
      try {
        if (runIn === "new") {
          const agent = agentModelPicker ? agentModelPicker.getAgent() : (agentEl ? agentEl.value : "");
          const topic = topicEl ? topicEl.value.trim() : "";
          const key = agent + "\u0000" + topic;
          if (newSessionKey !== key) {
            newSessionKey = key;
            newSessionId = null;
          }
          if (!newSessionId) {
            const created = await api("/api/sessions", {
              method: "POST",
              body: JSON.stringify({
                agent: agent,
                topic: topic,
                // The picked model rides onto the minted web row so the
                // loop's first materialized session runs it (the same model
                // the chat page stores on New Chat). Omitted when the
                // catalogue never loaded — the agent default applies.
                model: agentModelPicker ? (agentModelPicker.getModelId() || undefined) : undefined,
              }),
            });
            newSessionId = created.session && created.session.id;
          }
          sessionId = newSessionId;
        }
        const data = await api("/api/loops", {
          method: "POST",
          body: JSON.stringify(Object.assign({
            prompt_text: promptText,
            session_id: sessionId,
            // Omit the key unless the operator explicitly opted in.
            // Sending false is also safe (strict parser), but omit-by-default
            // keeps unconfigured loops from minting a session each tick.
            budget_usd: budgetUsd != null && budgetUsd > 0 ? budgetUsd : undefined,
          }, freshEach ? { fresh_session_each_run: true } : {}, bodySchedule)),
        });
        newSessionId = null;
        newSessionKey = null;
        closeForm();
        promptEl.value = "";
        document.getElementById("loops-create-interval").value = "30m";
        const timeEl = document.getElementById("loops-create-time");
        if (timeEl) timeEl.value = "09:00";
        const preset = document.getElementById("loops-create-frequency");
        if (preset) preset.value = "30m";
        applyFrequencyPreset("30m");
        if (topicEl) topicEl.value = "";
        const freshEl = document.getElementById("loops-create-fresh-session");
        if (freshEl) freshEl.checked = false;
        const budgetEl = document.getElementById("loops-create-budget");
        if (budgetEl) budgetEl.value = "";
        await loadList();
        if (data.loop) await openDetail(data.loop.loop_id);
      } catch (err) {
        errEl.textContent = "Couldn't start the loop: " + (err.message || String(err));
      } finally {
        controls.forEach(function (c) { c.disabled = false; });
      }
    });
  }

  function bindFilters() {
    document.getElementById("loops-refresh").addEventListener("click", function () {
      loadList().catch(function () {});
      if (selectedId) openDetail(selectedId);
    });
  }

  // Navigation drawer + conversations are handled by the shared nav.js
  // (loaded before this script). See templates/_menu.html.

  bindCreateForm();
  bindFrequencyPresets();
  bindRunInToggle();
  agentModelPicker = initAgentModelPicker();
  if (agentModelPicker) agentModelPicker.load();
  loadTopicOptions();
  bindFilters();

  loadList()
    .then(function () {
      if (selectedId) return openDetail(selectedId);
      return null;
    })
    .catch(function () {});

  pollTimer = setInterval(function () {
    loadList()
      .then(function () {
        // Re-render the open detail panel only when its loop actually changed
        // state — a status transition (paused) OR a `live` flip
        // (supervisor died/attached, badge goes live↔stalled). Re-rendering on
        // every poll would clobber a disclosure mid-open, so we gate on those
        // transitions (rare) rather than refreshing unconditionally.
        if (!selectedId) return;
        const l = allLoops.find(function (x) { return x.loop_id === selectedId; });
        if (l && (l.status !== detailStatus || !!l.live !== detailLive
            || !!l.in_flight !== detailInFlight))
          openDetail(selectedId);
      })
      .catch(function () {});
  }, 4000);
  window.addEventListener("beforeunload", function () {
    if (pollTimer) clearInterval(pollTimer);
  });
})();
