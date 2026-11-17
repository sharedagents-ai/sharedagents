(function () {
  const BASE =
    document.querySelector('meta[name="sharedagents-base"]')?.content || "";
  const INITIAL_GOAL =
    document.querySelector('meta[name="sharedagents-initial-goal"]')?.content || "";

  const STATUS_ORDER = ["running", "blocked", "paused", "met"];
  // need attention; completed history stays one click away.
  const VIEW_STATUSES = {
    active: new Set(["running", "blocked", "paused"]),
    completed: new Set(["met"]),
    all: new Set(STATUS_ORDER),
  };
  // Operator-facing labels. Backend status keys remain unchanged.
  const STATUS_LABELS = {
    running: "Working",
    blocked: "Needs input",
    paused: "Paused",
    met: "Completed",
  };
  const COLLAPSED_TITLE_LIMIT = 160;

  const listEl = document.getElementById("goals-list");
  const viewTabsEl = document.getElementById("goals-view-tabs");
  const agentFilter = document.getElementById("goals-filter-agent");
  const agentFilterLabel = document.getElementById("goals-filter-label");
  const totalsEl = document.getElementById("goals-totals");
  const updatedEl = document.getElementById("goals-updated");
  const connEl = document.getElementById("goals-conn");
  const createToggle = document.getElementById("goals-create-toggle");
  const createForm = document.getElementById("goals-create-form");
  const createAgentSelect = document.getElementById("goals-create-agent");
  const agentOptions = document.getElementById("goals-agent-options");

  let selectedId = INITIAL_GOAL.trim();
  let currentView = "active";
  // Status the open detail panel was last rendered for. The poll watches this:
  // when the selected goal transitions (e.g. running → met/Completed) the panel
  // re-renders itself so the badge and available controls update without a click.
  let detailStatus = null;
  let detailSupervised = null;
  let lastGoals = []; // the FILTERED subset currently rendered
  let allGoals = []; // raw, unfiltered goals from the API
  let pollTimer = null;
  let agentFilterTimer = null;
  let detailRequest = 0;

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function collapsedGoalTitle(value) {
    const chars = Array.from(String(value ?? "").trim());
    if (chars.length <= COLLAPSED_TITLE_LIMIT) return chars.join("");
    return chars.slice(0, COLLAPSED_TITLE_LIMIT - 1).join("").trimEnd() + "…";
  }

  function goalDomKey(goalId) {
    return String(goalId || "goal").replace(/[^a-zA-Z0-9_-]/g, "-");
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

  function statusBadge(status, supervisedLive, extraClass, livenessUnknown) {
    const key = String(status || "unknown");
    let label = STATUS_LABELS[status] || status || "unknown";
    let extra = "";
    // A running goal whose supervisor heartbeat has gone stale is no longer
    // driving turns — render it "Stalled" (muted, no pulse) so this page agrees
    // with the Agent Monitor, which likewise stops showing it as live "working".
    // When the liveness probe could not answer (unreadable index, absent root
    // row) the honest state is a neutral "Unknown" — never a confident
    // "Stalled" built on evidence we could not read.
    if (key === "running" && supervisedLive === false) {
      if (livenessUnknown) {
        label = "Unknown";
        extra = " is-unknown";
      } else {
        label = "Stalled";
        extra = " is-stalled";
      }
    }
    return (
      '<span class="badge badge-' + escapeHtml(key) + extra +
        (extraClass ? " " + escapeHtml(extraClass) : "") + '">' +
        '<i class="goal-dot" aria-hidden="true"></i>' +
        '<span>' + escapeHtml(label) + "</span>" +
      "</span>"
    );
  }

  // Status-aware controls. A running goal can be paused; a paused or
  // needs-input (blocked) goal can be resumed. Every goal can be deleted
  // (manual) or restarted on a fresh session.
  function goalActions(status) {
    const btn = function (act, label) {
      return '<button type="button" class="goal-action-btn" data-act="' + act + '">' + label + "</button>";
    };
    const out = [];
    if (status === "running") out.push(btn("pause", "Pause"));
    if (status === "paused" || status === "blocked") out.push(btn("resume", "Resume"));
    if (status !== "met") out.push(btn("restart", "Restart"));
    out.push(btn("delete", "Delete"));
    return out.join("");
  }

  function sessionHref(sessionId) {
    // Reuse the canonical chat deep-link contract so a goal
    // opens the conversation that is actually driving it.
    return BASE + "/?session=" + encodeURIComponent(String(sessionId || ""));
  }

  function sessionLink(goal, label, extraClass) {
    if (!goal.root_session_id) {
      return '<span class="goal-session-unavailable">Session not available</span>';
    }
    return (
      '<a class="goal-session-link ' + escapeHtml(extraClass || "") + '" data-session-link ' +
        'href="' + escapeHtml(sessionHref(goal.root_session_id)) + '" ' +
        'title="Open session ' + escapeHtml(goal.root_session_id) + '">' +
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

  function elapsedLabel(g) {
    const started = Date.parse(g.started_at || g.created_at || "");
    if (Number.isNaN(started)) return "";
    const terminalIso =
      g.status === "met" ? g.completed_at :
      g.status === "archived" ? g.archived_at :
      g.status === "paused" ? g.paused_at :
      g.status === "blocked" ? g.updated_at : "";
    const terminal = terminalIso ? Date.parse(terminalIso) : Date.now();
    return formatDuration((Number.isNaN(terminal) ? Date.now() : terminal) - started) + " elapsed";
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

  // The budget is never unbounded — a 0/missing value floors to
  // the default ($25, mirroring goal_service.DEFAULT_BUDGET_USD). Never renders ∞.
  const DEFAULT_BUDGET_USD = 25.0;
  const effBudget = (v) => (Number(v) > 0 ? Number(v) : DEFAULT_BUDGET_USD);

  function updateSummaries(goals) {
    const counts = Object.fromEntries(STATUS_ORDER.map(function (s) { return [s, 0]; }));
    goals.forEach(function (g) {
      counts[g.status] = (counts[g.status] || 0) + 1;
    });
    const active = (counts.running || 0) + (counts.blocked || 0) + (counts.paused || 0);
    if (totalsEl) totalsEl.textContent = active + " active";
    if (updatedEl) updatedEl.textContent = goals.length ? "Updated now" : "";
  }

  function renderList(goals) {
    lastGoals = goals;
    // Polling re-renders the compact rows. Preserve the expanded node itself so
    // open disclosures, a partially edited budget, and its event handlers stay
    // intact until the selected goal actually changes state.
    const preservedDetail = selectedId ? detailContainer(selectedId) : null;
    if (!goals.length) {
      const empty = currentView === "active"
        ? "No active goals. Start one when you have an outcome that needs ongoing work."
        : currentView === "completed"
          ? "No completed goals yet."
          : "No goals match this filter.";
      listEl.innerHTML = '<div class="goals-empty"><strong>' + escapeHtml(empty) + '</strong></div>';
      return;
    }

    listEl.innerHTML = goals.map(function (g) {
      const isSelected = g.goal_id === selectedId;
      const selected = isSelected ? " selected" : "";
      const title = isSelected
        ? String(g.condition_text || "").trim()
        : collapsedGoalTitle(g.condition_text);
      const elapsed = elapsedLabel(g);
      const key = goalDomKey(g.goal_id);
      return (
        '<article class="goal-card' + selected + '" data-id="' + escapeHtml(g.goal_id) + '" ' +
          'data-title="' + escapeHtml(g.condition_text) + '">' +
          '<button type="button" class="goal-card-summary" data-goal-toggle="' + escapeHtml(g.goal_id) + '" ' +
            'id="goal-summary-' + escapeHtml(key) + '" aria-expanded="' + String(isSelected) + '" ' +
            'aria-controls="goal-detail-' + escapeHtml(key) + '" title="' + escapeHtml(g.condition_text) + '">' +
            '<span class="goal-summary-copy">' +
              '<span class="condition">' + escapeHtml(title) + '</span>' +
              '<span class="goal-summary-meta">' +
                statusBadge(g.status, g.supervised_live, "goal-summary-status", g.liveness_unknown) +
                '<span class="goal-summary-agent">' + escapeHtml(g.agent_name || "unknown") + '</span>' +
                (elapsed ? '<span class="goal-summary-elapsed">' + escapeHtml(elapsed) + '</span>' : '') +
              '</span>' +
            '</span>' +
            '<svg class="goal-card-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" ' +
              'stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>' +
          '</button>' +
          '<div class="goal-card-detail" id="goal-detail-' + escapeHtml(key) + '" ' +
            'role="region" aria-labelledby="goal-summary-' + escapeHtml(key) + '"' + (isSelected ? '' : ' hidden') + '></div>' +
        "</article>"
      );
    }).join("");

    if (preservedDetail && selectedId) {
      const placeholder = detailContainer(selectedId);
      if (placeholder) placeholder.replaceWith(preservedDetail);
    }

    listEl.querySelectorAll("[data-goal-toggle]").forEach(function (el) {
      el.addEventListener("click", function () {
        toggleDetail(el.dataset.goalToggle);
      });
    });
  }

  function renderListError(err) {
    updateSummaries([]);
    listEl.innerHTML =
      '<div class="goals-load-error">' +
        "Failed to load goals: " + escapeHtml(err.message || String(err)) +
      "</div>";
  }

  function setAgentOptions(names) {
    const clean = Array.from(new Set(["jarvis"].concat(names || [])))
      .map(function (name) { return String(name || "").trim(); })
      .filter(Boolean)
      .sort(function (a, b) {
        if (a === "jarvis") return -1;
        if (b === "jarvis") return 1;
        return a.localeCompare(b);
      });
    const current = createAgentSelect.value || "jarvis";
    createAgentSelect.innerHTML = clean.map(function (name) {
      return '<option value="' + escapeHtml(name) + '">' + escapeHtml(name) + "</option>";
    }).join("");
    if (clean.includes(current)) createAgentSelect.value = current;
    if (agentOptions) {
      agentOptions.innerHTML = clean.map(function (name) {
        return '<option value="' + escapeHtml(name) + '"></option>';
      }).join("");
    }
  }

  async function loadAgentOptions() {
    try {
      const data = await api("/api/models");
      const names = (data.models || []).map(function (m) {
        return m.name || (String(m.id || "").replace(/^sharedagents\//, ""));
      });
      setAgentOptions(names);
    } catch (_) {
      setAgentOptions(["jarvis"]);
    }
  }

  function cardForGoal(goalId) {
    return Array.from(listEl.querySelectorAll(".goal-card")).find(function (card) {
      return card.dataset.id === goalId;
    }) || null;
  }

  function detailContainer(goalId) {
    return cardForGoal(goalId)?.querySelector(".goal-card-detail") || null;
  }

  function setCardExpanded(card, expanded) {
    if (!card) return;
    card.classList.toggle("selected", expanded);
    const toggle = card.querySelector("[data-goal-toggle]");
    if (toggle) toggle.setAttribute("aria-expanded", String(expanded));
    const title = card.querySelector(".condition");
    if (title) {
      const fullTitle = String(card.dataset.title || "").trim();
      title.textContent = expanded ? fullTitle : collapsedGoalTitle(fullTitle);
    }
    const detail = card.querySelector(".goal-card-detail");
    if (!detail) return;
    detail.hidden = !expanded;
    if (!expanded) detail.innerHTML = "";
  }

  function detailShell(goalId, content) {
    const card = cardForGoal(goalId);
    if (!card) return null;
    setCardExpanded(card, true);
    const detail = detailContainer(goalId);
    detail.innerHTML = '<div class="goal-card-detail-inner">' + content + "</div>";
    return detail;
  }

  function closeDetail(updateHistory) {
    const oldId = selectedId;
    detailRequest += 1;
    selectedId = "";
    detailStatus = null;
    detailSupervised = null;
    setCardExpanded(cardForGoal(oldId), false);
    if (updateHistory !== false && window.location.pathname !== BASE + "/goals") {
      history.replaceState(null, "", BASE + "/goals");
    }
  }

  function toggleDetail(goalId) {
    if (selectedId === goalId) {
      closeDetail();
      return;
    }
    openDetail(goalId);
  }

  function kv(label, value, href) {
    const body = href
      ? '<a href="' + escapeHtml(href) + '">' + escapeHtml(value || "-") + "</a>"
      : "<b>" + escapeHtml(value || "-") + "</b>";
    return '<div class="goal-kv"><span>' + escapeHtml(label) + "</span>" + body + "</div>";
  }

  function renderEvents(events) {
    if (!events || !events.length) {
      return '<div class="goals-empty">No events yet.</div>';
    }
    return events.map(function (ev) {
      const payload = ev.payload && Object.keys(ev.payload).length
        ? JSON.stringify(ev.payload, null, 2)
        : "";
      return (
        '<div class="goal-event">' +
          '<div class="goal-event-type">' + escapeHtml(ev.event_type || "event") + "</div>" +
          '<div class="goal-event-meta">' + escapeHtml(formatDate(ev.created_at)) + "</div>" +
          (payload ? '<div class="goal-event-payload">' + escapeHtml(payload) + "</div>" : "") +
        "</div>"
      );
    }).join("");
  }

  function renderViewTabs() {
    if (!viewTabsEl) return;
    viewTabsEl.querySelectorAll("[data-view]").forEach(function (tab) {
      const selected = tab.dataset.view === currentView;
      tab.classList.toggle("active", selected);
      tab.setAttribute("aria-pressed", String(selected));
    });
  }

  // The page opens on active work. Completed and archived history stays one
  // click away instead of competing with work that may need attention now.
  function renderFiltered() {
    const statuses = VIEW_STATUSES[currentView] || VIEW_STATUSES.active;
    renderList(allGoals.filter(function (g) {
      return statuses.has(g.status);
    }));
  }

  async function loadList() {
    // Status filtering is client-side, so only the agent narrows the request;
    // the API returns every status (including archived) which we then filter.
    const params = new URLSearchParams();
    if (agentFilter.value.trim()) params.set("agent", agentFilter.value.trim());
    const q = params.toString();
    try {
      const data = await api("/api/goals" + (q ? "?" + q : ""));
      allGoals = data.goals || [];
      setConnection(true);
      updateSummaries(allGoals);
      renderFiltered();
      return allGoals;
    } catch (err) {
      setConnection(false);
      renderListError(err);
      throw err;
    }
  }

  async function openDetail(goalId) {
    if (!goalId) return;
    if (selectedId && selectedId !== goalId) closeDetail(false);
    selectedId = goalId;
    const request = ++detailRequest;
    setCardExpanded(cardForGoal(goalId), true);
    detailShell(goalId, '<div class="goals-loading">Loading goal...</div>');
    try {
      const g = await api("/api/goals/" + encodeURIComponent(goalId));
      if (request !== detailRequest || selectedId !== goalId) return;
      setConnection(true);
      detailStatus = g.status;
      detailSupervised = !!g.supervised_live;
      const rootHref = g.root_session_id ? sessionHref(g.root_session_id) : "";
      const elapsed = elapsedLabel(g);
      const summary = g.terminal_summary || "";
      const lastError = g.last_error || "";
      // A direct /goals/<id> link should also make the matching card visible.
      if (!lastGoals.some(function (item) { return item.goal_id === goalId; })) {
        currentView = g.status === "met" ? "completed" : "active";
        renderViewTabs();
        renderFiltered();
      }
      const detailEl = detailShell(
        goalId,
        '<div class="goals-detail-hero">' +
          '<div class="goal-detail-status">' + statusBadge(g.status, g.supervised_live, "", g.liveness_unknown) +
            (elapsed ? '<span class="goal-elapsed">' + escapeHtml(elapsed) + '</span>' : '') + '</div>' +
          (g.wait_label ? '<div class="goal-wait-label">' + escapeHtml(g.wait_label) + '</div>' : '') +
          sessionLink(g, g.status === "running" ? "Open running session" : "Open session", "is-primary is-large") +
        '</div>' +
        '<div class="goal-detail-controls">' +
          '<span class="goal-detail-controls-label">Goal controls</span>' +
          '<div class="goals-actions">' + goalActions(g.status) + '</div>' +
        "</div>" +
        '<details class="goal-disclosure">' +
          '<summary><span>Budget &amp; usage</span><span class="goal-disclosure-hint">Optional</span></summary>' +
          '<div class="goal-disclosure-body">' +
            '<div class="goal-detail-grid">' +
              kv("Spent", "$" + Number(g.cost_usd || 0).toFixed(4)) +
              kv("Budget", "$" + effBudget(g.max_cost_usd).toFixed(2)) +
            '</div>' +
            '<div class="goal-budget-edit">' +
              '<label class="goal-budget-label" for="goal-budget-input">Change budget (USD)</label>' +
              '<input type="number" step="0.01" min="0" class="goal-budget-input" ' +
                'id="goal-budget-input" value="' + escapeHtml(String(Number(g.max_cost_usd || 0))) + '">' +
              '<button type="button" class="goal-action-btn" data-budget-set>Save budget</button>' +
              '<span class="goal-budget-hint">0 uses the default $25 budget</span>' +
            '</div>' +
          '</div>' +
        '</details>' +
        '<details class="goal-disclosure">' +
          '<summary><span>Goal details</span><span class="goal-disclosure-hint">IDs, timing &amp; last check</span></summary>' +
          '<div class="goal-disclosure-body">' +
            '<div class="goal-detail-grid">' +
              kv("Agent", g.agent_name) +
              kv("Running session", g.root_session_id || "-", rootHref) +
              kv("Goal ID", g.goal_id) +
              kv("Turns", String(g.turn_count || 0)) +
              kv("Started", formatDate(g.started_at || g.created_at)) +
              kv("Last updated", formatDate(g.updated_at || g.last_activity_at)) +
              kv("Outcome", g.terminal_reason || "-") +
            '</div>' +
            (summary ? '<div class="goal-detail-note"><strong>Evidence</strong><p>' + escapeHtml(summary) + '</p></div>' : '') +
            (lastError ? '<div class="goal-detail-note is-error"><strong>Last error</strong><p>' + escapeHtml(lastError) + '</p></div>' : '') +
          '</div>' +
        '</details>' +
        '<details class="goal-disclosure">' +
          '<summary><span>Activity history</span><span class="goal-disclosure-hint">' + escapeHtml(String((g.events || []).length)) + ' events</span></summary>' +
          '<div class="goal-disclosure-body"><div class="goal-events">' + renderEvents(g.events || []) + '</div></div>' +
        '</details>'
      );
      if (!detailEl) return;
      detailEl.querySelectorAll("[data-act]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          runAction(goalId, btn.dataset.act, g.agent_name);
        });
      });
      const budgetBtn = detailEl.querySelector("[data-budget-set]");
      if (budgetBtn) {
        budgetBtn.addEventListener("click", function () {
          const input = detailEl.querySelector("#goal-budget-input");
          const usd = parseFloat(input && input.value);
          if (!isFinite(usd) || usd < 0) {
            showBudgetError(goalId, "Enter a finite budget ≥ 0 (0 = default $25).");
            return;
          }
          setBudget(goalId, usd, g.agent_name);
        });
      }
      if (window.location.pathname !== BASE + "/goals/" + encodeURIComponent(goalId)) {
        history.replaceState(null, "", BASE + "/goals/" + encodeURIComponent(goalId));
      }
    } catch (err) {
      if (request !== detailRequest || selectedId !== goalId) return;
      setConnection(false);
      detailShell(
        goalId,
        '<div class="goals-load-error">Failed to load goal: ' + escapeHtml(err.message || String(err)) + "</div>",
      );
    }
  }

  async function runAction(goalId, action, agentName) {
    const detailEl = detailContainer(goalId);
    if (!detailEl) return;
    const buttons = detailEl.querySelectorAll("[data-act]");
    buttons.forEach(function (b) { b.disabled = true; });
    try {
      await api("/api/goals/" + encodeURIComponent(goalId) + "/" + action, {
        method: "POST",
        body: JSON.stringify({ agent_name: agentName || "jarvis" }),
      });
      await loadList();
      if (action === "delete" || action === "archive") {
        closeDetail();
        return;
      }
      await openDetail(goalId);
    } catch (err) {
      detailContainer(goalId)?.querySelector(".goal-card-detail-inner")?.insertAdjacentHTML(
        "afterbegin",
        '<div class="goals-load-error">Action failed: ' + escapeHtml(err.message || String(err)) + "</div>",
      );
    } finally {
      buttons.forEach(function (b) { b.disabled = false; });
    }
  }

  function showBudgetError(goalId, msg) {
    const detailEl = detailContainer(goalId);
    if (!detailEl) return;
    // Replace any prior error so repeated clicks don't stack messages.
    detailEl.querySelectorAll(".goals-load-error").forEach(function (e) { e.remove(); });
    detailEl.querySelector(".goal-card-detail-inner")?.insertAdjacentHTML(
      "afterbegin",
      '<div class="goals-load-error">' + escapeHtml(msg) + "</div>",
    );
  }

  async function setBudget(goalId, usd, agentName) {
    const detailEl = detailContainer(goalId);
    if (!detailEl) return;
    const btn = detailEl.querySelector("[data-budget-set]");
    if (btn) btn.disabled = true;
    try {
      await api("/api/goals/" + encodeURIComponent(goalId) + "/set-budget", {
        method: "POST",
        body: JSON.stringify({ agent_name: agentName || "jarvis", max_cost_usd: usd }),
      });
      await loadList();
      await openDetail(goalId);
    } catch (err) {
      showBudgetError(goalId, "Set budget failed: " + (err.message || String(err)));
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function bindCreateForm() {
    const closeForm = function () {
      createForm.hidden = true;
      createToggle.setAttribute("aria-expanded", "false");
      document.getElementById("goals-create-error").textContent = "";
    };
    createToggle.addEventListener("click", function () {
      const next = createForm.hidden;
      createForm.hidden = !next;
      createToggle.setAttribute("aria-expanded", String(next));
      if (next) document.getElementById("goals-create-condition").focus();
    });
    ["goals-create-close", "goals-create-cancel"].forEach(function (id) {
      document.getElementById(id)?.addEventListener("click", closeForm);
    });

    createForm.addEventListener("submit", async function (ev) {
      ev.preventDefault();
      const errEl = document.getElementById("goals-create-error");
      const submit = createForm.querySelector('button[type="submit"]');
      errEl.textContent = "";
      submit.disabled = true;
      try {
        const data = await api("/api/goals", {
          method: "POST",
          body: JSON.stringify({
            condition_text: document.getElementById("goals-create-condition").value,
            agent_name: document.getElementById("goals-create-agent").value || "jarvis",
            max_cost_usd: document.getElementById("goals-create-budget").value || null,
          }),
        });
        closeForm();
        document.getElementById("goals-create-condition").value = "";
        document.getElementById("goals-create-budget").value = "";
        const options = createForm.querySelector(".goals-create-options");
        if (options) options.open = false;
        await loadList();
        if (data.goal) await openDetail(data.goal.goal_id);
      } catch (err) {
        errEl.textContent = err.message || String(err);
      } finally {
        submit.disabled = false;
      }
    });
  }

  function bindFilters() {
    document.getElementById("goals-refresh").addEventListener("click", function () {
      loadList().catch(function () {});
      if (selectedId) openDetail(selectedId);
    });
    if (viewTabsEl) {
      viewTabsEl.addEventListener("click", function (e) {
        const tab = e.target.closest("[data-view]");
        if (!tab || !VIEW_STATUSES[tab.dataset.view]) return;
        if (tab.dataset.view !== currentView && selectedId) closeDetail();
        currentView = tab.dataset.view;
        renderViewTabs();
        renderFiltered();
      });
    }
    agentFilter.addEventListener("input", function () {
      if (selectedId) closeDetail();
      const agent = agentFilter.value.trim();
      if (agentFilterLabel) agentFilterLabel.textContent = agent ? "Agent: " + agent : "Filter";
      clearTimeout(agentFilterTimer);
      agentFilterTimer = setTimeout(function () {
        loadList().catch(function () {});
      }, 180);
    });
    document.getElementById("goals-filter-clear")?.addEventListener("click", function () {
      if (selectedId) closeDetail();
      agentFilter.value = "";
      if (agentFilterLabel) agentFilterLabel.textContent = "Filter";
      document.getElementById("goals-filter-menu")?.removeAttribute("open");
      loadList().catch(function () {});
    });
  }

  // Navigation drawer + conversations are handled by the shared nav.js
  // (loaded before this script). See templates/_menu.html.

  renderViewTabs();
  bindCreateForm();
  bindFilters();
  loadAgentOptions();

  loadList()
    .then(function () {
      if (selectedId) return openDetail(selectedId);
      return null;
    })
    .catch(function () {});

  pollTimer = setInterval(function () {
    loadList()
      .then(function () {
        // Re-render the open detail panel only when its goal actually changed
        // state — a status transition (met/blocked) OR a supervised_live flip
        // (supervisor died/attached, badge goes live↔stale). Re-rendering on
        // every poll would clobber the budget input mid-edit, so we gate on
        // those transitions (rare) rather than refreshing unconditionally.
        if (!selectedId) return;
        const g = allGoals.find(function (x) { return x.goal_id === selectedId; });
        if (g && (g.status !== detailStatus || !!g.supervised_live !== detailSupervised))
          openDetail(selectedId);
      })
      .catch(function () {});
  }, 4000);
  window.addEventListener("beforeunload", function () {
    if (pollTimer) clearInterval(pollTimer);
  });
})();
