/* Agent Monitor — live agent → session → subagent tree.
 *
 * Transport: poll GET /api/agents/monitor every 1s with the tab's session
 * filters applied server-side (limit / age / agent / model / engine / state /
 * search). This page monitors sessions, not goals.
 *
 * Rendering: a flat, ordered list of visible rows reconciled BY KEY against
 * the live DOM. Moving an existing element with appendChild does not restart
 * its CSS animation, and we only rewrite a status dot's class when the state
 * actually changed — so live pulses stay smooth across snapshots (the
 * Phoenix "don't full-re-render" lesson). Collapse state survives snapshots
 * and reloads (localStorage); agents are always collapsed by default: the
 * operator expands an agent to see its sessions with subagents still hidden
 * (each subagent row stays expandable on click). Session and subagent
 * collapse preferences persist across reloads; agent headers re-collapse
 * on every page load so the monitor always starts clean.
 */
(function () {
  "use strict";

  // Public path prefix when reverse-proxied (e.g. "/sharedagents"); empty at root.
  // From the <meta name="sharedagents-base"> (not an inline script → CSP-safe).
  const BASE = ((document.querySelector('meta[name="sharedagents-base"]') || {}).content || "");

  const treeEl = document.getElementById("mon-tree");
  const emptyEl = document.getElementById("mon-empty");
  const summaryEl = document.getElementById("mon-summary");
  const totalsEl = document.getElementById("mon-totals");
  const connEl = document.getElementById("mon-conn");
  const updatedEl = document.getElementById("mon-updated");
  const windowEl = document.getElementById("mon-window");
  const limitEl = document.getElementById("mon-limit");
  const searchEl = document.getElementById("mon-search");
  const groupByEl = document.getElementById("mon-groupby");

  // hash/agent-key -> row element, for keyed reconciliation.
  const rowEls = new Map();
  // Liveness clock — bumped by ANY inbound frame (open / snapshot / heartbeat
  // / poll), not just tree changes, so the connection dot doesn't falsely go
  // "stale" while a healthy stream is only sending heartbeats.
  let lastMsgAt = 0;
  let pollTimer = null;

  // ---- persisted collapse state --------------------------------------
  const COLLAPSE_KEY = "sharedagents.monitor.collapsed";
  let collapsed = new Set();
  // True once a stored collapse preference exists (at load, or after the
  // first manual toggle). While false, render() seeds sessions-with-subagents
  // collapsed so a fresh monitor opens with agent headers only; the first
  // toggle turns the seeded set into a real persisted preference and stops
  // the seeding, so session and subagent collapse preferences persist across
  // reloads. Agents themselves are handled separately (agentsSeededThisLoad
  // below): they ALWAYS start collapsed on a fresh page load.
  let hasStoredPref = false;
  // One-shot flag: agents are seeded collapsed exactly once per page load so
  // the monitor always starts clean. Re-seeding on EVERY render (as the
  // first draft of this change did) instantly undid an expansion, because
  // toggleCollapse() re-renders — the toggle deleted the key and the render
  // re-added it. Once seeded, the operator's toggles own the set for the
  // rest of the page lifetime.
  let agentsSeededThisLoad = false;
  try {
    const raw = localStorage.getItem(COLLAPSE_KEY);
    if (raw) {
      collapsed = new Set(JSON.parse(raw));
      hasStoredPref = true;
    }
  } catch (_e) { /* ignore */ }
  function persistCollapsed() {
    try { localStorage.setItem(COLLAPSE_KEY, JSON.stringify([...collapsed])); } catch (_e) {}
  }
  function toggleCollapse(key) {
    if (collapsed.has(key)) collapsed.delete(key); else collapsed.add(key);
    hasStoredPref = true; // first toggle turns the seeded set into a real preference
    persistCollapsed();
    if (lastTree) render(lastTree);
  }

  // ---- persisted filter state ----------------------------------------
  // CLIENT-SIDE filters (the data scope is always full — see streamUrl):
  //   • sessionLimit — server-side cap on most-recent sessions (sqlite LIMIT)
  //   • maxAgeSec — hide sessions whose last activity is older than this
  //   • query     — free-text over agent / title / hash
  //   • groupBy   — how the tree is grouped (agent | model | engine | state | none)
  //   • value-filters (fgSel) — which agent/model/engine/STATE values to show
  // All survive reload (localStorage). On a FIRST visit the default is
  // "last 15 min" (seen ≤ 15m); the STATE filter shows idle by default.
  const FILTER_KEY = "sharedagents.monitor.filters";
  const GROUP_BYS = ["agent", "model", "engine", "state", "none"];
  // Guarded: a profile that denies storage (privacy mode) throws on getItem —
  // it must not abort the whole monitor before the tree renders.
  let firstVisit = true;
  try { firstVisit = !localStorage.getItem(FILTER_KEY); } catch (_e) { /* storage denied */ }
  let maxAgeSec = firstVisit ? 900 : 0;   // first visit → last 15 min
  let sessionLimit = firstVisit ? 100 : 200; // first visit → 100 most-recent rows
  let query = "";                          // lower-cased substring
  // Grouping dimension. Default: agent — the grouped view is what lets the
  // fresh-start "all agents collapsed" default show headers-only (see
  // render()/buildRows); "none" is the flat, age-sorted list.
  let groupBy = "agent";
  try {
    const raw = localStorage.getItem(FILTER_KEY);
    if (raw) {
      const f = JSON.parse(raw);
      if (Number.isFinite(f.maxAgeSec)) maxAgeSec = f.maxAgeSec;
      if (Number.isFinite(f.sessionLimit)) sessionLimit = f.sessionLimit;
      if (typeof f.query === "string") query = f.query;
      if (GROUP_BYS.indexOf(f.groupBy) !== -1) groupBy = f.groupBy;
    }
  } catch (_e) { /* ignore */ }
  function persistFilters() {
    try {
      localStorage.setItem(FILTER_KEY, JSON.stringify({ maxAgeSec, sessionLimit, query, groupBy }));
    } catch (_e) {}
  }
  // Collapsed-card chip: a terse summary of the active filters so the user can
  // see what's applied without expanding the card.
  function updateFilterSummary() {
    const el = document.getElementById("mon-filter-active");
    if (!el) return;
    const parts = ["by " + groupBy];
    if (maxAgeSec) parts.push("≤ " + (windowEl
      ? (windowEl.options[windowEl.selectedIndex] || {}).text || (maxAgeSec + "s")
      : maxAgeSec + "s"));
    if (sessionLimit) parts.push("last " + sessionLimit);
    if (query) parts.push('"' + query + '"');
    let fgActive = 0;
    for (const d of fgDims) { if (fgSel[d] && fgSel[d].size) fgActive++; }
    if (fgActive) parts.push("⚑" + fgActive);
    el.textContent = parts.join(" · ");
  }

  // ---- value-level multi-select filtering (compose with the rest) ----
  // Per dimension { agent|model|engine|state : Set<value> }; an empty set means
  // "all" for that dimension, a non-empty set restricts to the chosen values.
  // Options are rebuilt from the live tree each render (rebuildFGOptions). Lives
  // in the SAME collapsible Filters card as the group-by / age / search controls.
  const FG_KEY = "sharedagents.monitor.fg";
  // The STATE dimension REPLACES the old "show idle" toggle: it lists the real
  // session states (idle / error / streaming / …) so the user picks which to
  // show. "idle" is shown by default (empty selection = all), and the data scope
  // always includes idle/closed sessions so they're available to select.
  const fgDims = ["agent", "model", "engine", "state"];
  let fgSel = {};
  fgDims.forEach((d) => { fgSel[d] = new Set(); });
  try {
    const raw = localStorage.getItem(FG_KEY);
    if (raw) {
      const f = JSON.parse(raw);
      for (const d of fgDims) if (Array.isArray(f[d])) fgSel[d] = new Set(f[d]);
    }
  } catch (_e) { /* storage denied / bad json */ }
  function persistFG() {
    try {
      localStorage.setItem(FG_KEY, JSON.stringify({
        agent: [...fgSel.agent], model: [...fgSel.model],
        engine: [...fgSel.engine], state: [...fgSel.state],
      }));
    } catch (_e) {}
  }
  // The filterable value of a node for a dimension — matches how group-by
  // buckets read it (state folds a non-live session to "idle").
  function fgVal(node, d) {
    if (d === "agent") return node.agent || "";
    if (d === "state") return node.stale ? "stale" : (node.state || "idle");
    return node[d] || "";
  }
  function passesFG(node) {
    for (const d of fgDims) {
      const sel = fgSel[d];
      if (sel && sel.size > 0 && !sel.has(fgVal(node, d))) return false;
    }
    return true;
  }
  function collectFGValues(node, dimValues) {
    for (const d of fgDims) {
      const v = fgVal(node, d);
      if (v) dimValues[d].add(v);
    }
    for (const c of (node.children || [])) collectFGValues(c, dimValues);
  }
  // Rebuild the per-dimension checkbox option lists from the latest snapshot, so
  // options track live data; called on every render.
  function rebuildFGOptions(tree) {
    const dimValues = {};
    for (const d of fgDims) dimValues[d] = new Set();
    for (const agent of tree.agents || []) {
      dimValues.agent.add(agent.agent);
      for (const root of (agent.roots || [])) collectFGValues(root, dimValues);
    }
    for (const d of fgDims) {
      const fs = document.getElementById("mon-fg-" + d);
      if (!fs) continue;
      for (const el of fs.querySelectorAll('label.mon-fg-opt')) el.remove();
      for (const v of [...dimValues[d]].sort()) {
        const label = document.createElement("label");
        label.className = "mon-fg-opt";
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.className = "mon-fg-cb";
        cb.dataset.fg = d; cb.value = v;
        cb.checked = fgSel[d].size === 0 || fgSel[d].has(v);
        label.appendChild(cb);
        label.appendChild(document.createTextNode(" " + v));
        fs.appendChild(label);
      }
      const allCb = fs.querySelector('input.mon-fg-cb[value="__all__"]');
      if (allCb) allCb.checked = fgSel[d].size === 0;
    }
  }

  // Most-recent activity for a session node, in epoch-ms. Prefer the server-
  // computed ``last_activity_at`` (sqlite-backed; idle rows use last_turn_ts
  // so a restart sweep cannot masquerade old sessions as fresh).
  function lastSeenMs(node) {
    const ts = node.last_activity_at || node.last_turn_ts || node.state_updated_at || 0;
    if (!ts) return 0;
    return ts > 1e12 ? ts : ts * 1000;
  }
  // A node passes the age window if it's recent enough. 0 ts (never seen) is
  // only shown when no window is set, so stale/unknown rows don't masquerade
  // as fresh under a tight filter.
  function withinAge(node) {
    if (!maxAgeSec) return true;
    const ms = lastSeenMs(node);
    if (!ms) return false;
    return (Date.now() - ms) <= maxAgeSec * 1000;
  }
  function matchesQuery(node) {
    if (!query) return true;
    const hay = (
      (node.agent || "") + " " + (node.title || "") + " " + (node.hash || "")
    ).toLowerCase();
    return hay.indexOf(query) !== -1;
  }
  // A session row is kept if it passes ALL filters (age, query, value-filters)
  // itself, OR any descendant does — so an old parent is never pruned out from
  // under a fresh subagent.
  function nodeKept(node) {
    if (withinAge(node) && matchesQuery(node) && passesFG(node)) return true;
    return (node.children || []).some(nodeKept);
  }

  // ---- deterministic per-agent color ---------------------------------
  function agentColor(name) {
    let h = 0;
    for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
    return `hsl(${h % 360} 70% 62%)`;
  }

  // ---- timestamp → relative age --------------------------------------
  function ageStr(ts) {
    if (!ts) return "";
    // Stored as epoch seconds or ms depending on writer; normalise to ms.
    const ms = ts > 1e12 ? ts : ts * 1000;
    let s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m";
    if (s < 86400) return Math.floor(s / 3600) + "h";
    return Math.floor(s / 86400) + "d";
  }

  // ---- build the visible row list ------------------------------------
  let lastTree = null;

  // All top-level session nodes across every agent (each node carries its own
  // .agent/.model/.engine/.state), for non-agent grouping.
  function rootNodes(tree) {
    const out = [];
    for (const agent of tree.agents || []) {
      for (const r of (agent.roots || [])) out.push(r);
    }
    return out;
  }
  // The group bucket a session falls in for the chosen dimension. Empty
  // model/engine (e.g. a CLI agent that hasn't reported one) bucket as "(none)".
  function groupKeyOf(node, by) {
    if (by === "state") {
      if (node.stale) return "stale";
      const activity = node.activity && typeof node.activity === "object"
        ? node.activity : null;
      // V2 status keeps failed/detached rows visible in their own bucket;
      // legacy payloads retain the old raw-state behavior.
      return activity && activity.status
        ? activity.status : (node.state || "idle");
    }
    const v = (node[by] == null ? "" : String(node[by])).trim();
    return v || "(none)";
  }

  // Sort sessions newest-first by last-turn age (most recent activity on top).
  function byAgeDesc(a, b) { return lastSeenMs(b) - lastSeenMs(a); }

  function buildRows(tree) {
    const rows = [];
    // No grouping: one flat, age-sorted list of every session (subagents stay
    // nested under their parent).
    if (groupBy === "none") {
      const roots = rootNodes(tree).filter(nodeKept).sort(byAgeDesc);
      for (const node of roots) pushNode(node, 0, rows, false);
      return rows;
    }
    // Default dimension: the server already nests sessions under their agent.
    if (groupBy === "agent") {
      for (const agent of tree.agents || []) {
        const keptRoots = (agent.roots || []).filter(nodeKept).sort(byAgeDesc);
        // Drop an agent entirely once the filters hide all of its sessions.
        if (keptRoots.length === 0) continue;
        const akey = "agent:" + agent.agent;
        rows.push({ key: akey, kind: "agent", depth: 0, agent,
                    hasKids: keptRoots.length > 0 });
        if (collapsed.has(akey)) continue;
        for (const root of keptRoots) pushNode(root, 1, rows, false);
      }
      return rows;
    }
    // Other dimensions: regroup all root sessions client-side by their value.
    const groups = new Map();
    for (const node of rootNodes(tree)) {
      if (!nodeKept(node)) continue;
      const gv = groupKeyOf(node, groupBy);
      if (!groups.has(gv)) groups.set(gv, []);
      groups.get(gv).push(node);
    }
    // Busiest bucket first, then alphabetical — a stable, predictable order.
    const keys = [...groups.keys()].sort(function (a, b) {
      return (groups.get(b).length - groups.get(a).length) || a.localeCompare(b);
    });
    for (const gv of keys) {
      const nodes = groups.get(gv).sort(byAgeDesc);
      // Encode the value so a model/title containing ":" can't collide with
      // another bucket and poison the collapse/reconcile keyspace.
      const gkey = "group:" + groupBy + ":" + encodeURIComponent(gv);
      rows.push({ key: gkey, kind: "group", depth: 0,
                  group: { by: groupBy, value: gv, count: nodes.length },
                  hasKids: nodes.length > 0 });
      if (collapsed.has(gkey)) continue;
      for (const node of nodes) pushNode(node, 1, rows, false);
    }
    return rows;
  }
  // isSub marks a row as a (nested) subagent vs a top-level/root session — used
  // by the header counters so they reflect exactly the rows on screen.
  function pushNode(node, depth, rows, isSub) {
    const keptKids = (node.children || []).filter(nodeKept);
    rows.push({ key: node.hash, kind: "session", depth, node,
                isSub: !!isSub, hasKids: keptKids.length > 0 });
    if (collapsed.has(node.hash)) return;
    for (const c of keptKids) pushNode(c, depth + 1, rows, true);
  }

  // Seed one session node (and, recursively, its subagents) into the
  // collapsed set — only nodes that actually HAVE children, since leaves
  // carry no twisty. Keyed by node.hash, the SAME key pushNode() consults
  // when deciding whether to recurse, so a seeded session hides exactly its
  // nested subagent rows (at any depth) while staying expandable on click.
  function seedSessionCollapsed(node) {
    const kids = node.children || [];
    if (kids.length > 0 && node.hash != null) collapsed.add(node.hash);
    for (const c of kids) seedSessionCollapsed(c);
  }

  // ---- element factories ---------------------------------------------
  function makeRow(key) {
    const row = document.createElement("div");
    row.className = "tree-row";
    row.dataset.key = key;
    row.innerHTML =
      '<span class="tree-twisty"></span>' +
      '<span class="glyph"></span>' +
      '<span class="tree-label"></span>' +
      '<span class="tree-meta"></span>';
    const twisty = row.querySelector(".tree-twisty");
    twisty.addEventListener("click", function (e) {
      e.stopPropagation();
      toggleCollapse(key);
    });
    // Click a SESSION row to jump to its conversation in the chat UI.
    // `key` is the session hash (== gateway session id); the chat resolves it
    // to the owning web conversation. The twisty keeps its own click so a
    // parent with children can still expand without navigating. Agent rows
    // aren't conversations, so a double-click there just toggles their subtree.
    row.addEventListener("click", function (e) {
      if (row.dataset.kind !== "session") return;
      e.preventDefault();
      openConversation(key);
    });
    row.addEventListener("dblclick", function (e) {
      e.preventDefault();
      if (row.dataset.kind !== "session") toggleCollapse(key);
    });
    return row;
  }

  // Navigate to the chat UI focused on this gateway session. The chat reads
  // the ?session= param on load and opens the matching web conversation.
  function openConversation(hash) {
    if (!hash) return;
    window.location.href = BASE + "/?session=" + encodeURIComponent(hash);
  }

  function updateTwisty(row, item) {
    const t = row.querySelector(".tree-twisty");
    if (!item.hasKids) {
      t.className = "tree-twisty leaf";
      t.textContent = "";
      return;
    }
    const isCollapsed = collapsed.has(item.key);
    t.className = "tree-twisty" + (isCollapsed ? " collapsed" : "");
    t.textContent = "▾"; // ▾
  }

  function renderAgentRow(row, item) {
    const a = item.agent;
    row.classList.add("is-agent");
    row.classList.remove("idle");
    row.style.paddingLeft = "8px";
    updateTwisty(row, item);

    // Build the agent-row shape once (kind stable per key) — avoids churning
    // innerHTML on every reconcile.
    if (row.dataset.kind !== "agent") {
      row.dataset.kind = "agent";
      row.querySelector(".glyph").className = "glyph agent-swatch";
      row.querySelector(".tree-label").innerHTML = "";
    }
    const glyph = row.querySelector(".glyph");
    glyph.style.setProperty("--agc", agentColor(a.agent));
    row.querySelector(".tree-label").textContent = a.agent;

    const meta = row.querySelector(".tree-meta");
    const sub = a.subagents_calling || 0;
    meta.innerHTML =
      '<span class="badge">' + (a.active_sessions || 0) + " active</span>" +
      '<span class="badge sub' + (sub ? " hot" : "") + '">' + sub + " subagents</span>";
  }

  // Header row for a non-agent group (model / engine / state). Reuses the
  // bold agent-row chrome; the swatch is coloured by the group value.
  function renderGroupRow(row, item) {
    const g = item.group;
    row.classList.add("is-agent");
    row.classList.remove("idle");
    row.style.paddingLeft = "8px";
    updateTwisty(row, item);

    if (row.dataset.kind !== "group") {
      row.dataset.kind = "group";
      row.querySelector(".glyph").className = "glyph agent-swatch";
      row.querySelector(".tree-label").innerHTML =
        '<span class="group-by-tag"></span><span class="group-val"></span>';
    }
    row.querySelector(".glyph").style.setProperty("--agc", agentColor(g.value));
    row.querySelector(".group-by-tag").textContent = g.by + ":";
    row.querySelector(".group-val").textContent = g.value;

    const meta = row.querySelector(".tree-meta");
    meta.innerHTML =
      '<span class="badge">' + g.count + " session" + (g.count === 1 ? "" : "s") + "</span>";
  }

  // Session activity comes from the gateway tree node itself — this page
  // monitors sessions, not goals.
  function sessionView(n) {
    const state = n.state || "idle";
    const a = n.activity && typeof n.activity === "object" ? n.activity : {};
    const live = a.own_live == null ? !!n.live : !!a.own_live;
    const treeBusy = a.tree_busy == null ? !!(n.live || n.subagent_count) : !!a.tree_busy;
    return {
      state: state,
      live: live,
      treeBusy: treeBusy,
      status: a.status || "",
      attention: String(a.attention || ""),
      outcome: a.last_outcome || "",
      phase: String(a.phase || ""),
    };
  }

  // Legend states only. Never use Tasks-board display labels (Ready /
  // Needs review) — those painted idle rows red and hid real errors.
  function monitorVisual(n, eff) {
    const LIVE = ["prompting", "prefill", "reasoning", "streaming", "tool_running", "compacting", "ask"];
    if (n.stale) {
      return {
        code: "stale",
        text: "stale · " + (eff.state || "idle"),
        cls: "state stale",
      };
    }
    if (eff.attention === "question" || eff.attention === "permission" ||
        eff.attention === "operator_paused") {
      return { code: "needs_input", text: "needs input", cls: "state needs_input" };
    }
    const livePhase = LIVE.includes(eff.phase) ? eff.phase
      : (LIVE.includes(eff.state) ? eff.state : "");
    if (eff.live && livePhase) {
      const text = livePhase === "tool_running" ? "tool" : livePhase;
      return { code: livePhase, text: text, cls: "state live" };
    }
    if (eff.state === "waiting" || (eff.treeBusy && !eff.live)) {
      return { code: "waiting", text: "waiting", cls: "state waiting" };
    }
    if (eff.state === "paused") {
      return { code: "paused", text: "paused", cls: "state" };
    }
    if (eff.outcome === "failed" || eff.outcome === "interrupted" ||
        eff.state === "error") {
      return { code: "error", text: "error", cls: "state error" };
    }
    return { code: "idle", text: "idle", cls: "state" };
  }

  function renderSessionRow(row, item) {
    const n = item.node;
    const eff = sessionView(n);
    row.classList.remove("is-agent");
    row.classList.add("is-session");
    row.title = "Click to open this conversation";
    row.classList.toggle("idle", !eff.live && !eff.treeBusy);
    row.style.paddingLeft = (8 + item.depth * 18) + "px";
    updateTwisty(row, item);

    // Build the session-row shape once (kind stable per key), then update text
    // in place so a snapshot never flickers/rebuilds the label DOM.
    if (row.dataset.kind !== "session") {
      row.dataset.kind = "session";
      row.querySelector(".glyph").className = "glyph dot st-idle";
      row.querySelector(".tree-label").innerHTML =
        '<span class="agent-tag"></span>' +
        '<span class="title"></span>' +
        '<span class="hash"></span>';
    }

    // Status dot — only rewrite class when the state changed, so a running
    // pulse animation is never reset by an unrelated snapshot. A stale row
    // (live state that failed the server freshness gate — only seen in
    // "show idle" mode) gets a distinct dashed glyph, never the live pulse.
    const vis = monitorVisual(n, eff);
    const glyph = row.querySelector(".glyph");
    const wantDot = "glyph dot st-" + vis.code;
    if (glyph.className !== wantDot) glyph.className = wantDot;

    const label = row.querySelector(".tree-label");
    const tag = label.querySelector(".agent-tag");
    tag.textContent = n.agent;
    tag.style.color = agentColor(n.agent);
    label.querySelector(".title").textContent = displaySessionTitle(n.title);
    label.querySelector(".hash").textContent = (n.hash || "").slice(0, 8);

    const meta = row.querySelector(".tree-meta");
    // Defence-in-depth: every gateway-supplied value below is escaped/coerced
    // before being concatenated into innerHTML — state is a server enum and
    // round/subagent_count are numbers today, but never trust upstream shape.
      let html = "";
      html += '<span class="' + vis.cls + '">' + escapeHtml(vis.text) + "</span>";
    if (n.live && n.round != null) html += "<span>r" + Number(n.round) + "</span>";
    const age = ageStr(n.last_activity_at || n.last_turn_ts || n.state_updated_at);
    if (age) html += "<span>" + escapeHtml(age) + "</span>";
    if (n.subagent_count) {
      html += '<span class="badge sub hot">' + Number(n.subagent_count) + " sub</span>";
    }
    meta.innerHTML = html;
  }

  // ---- keyed reconcile -----------------------------------------------
  function render(tree) {
    lastTree = tree;
    // Agents are seeded collapsed ONCE per page load ("agent headers
    // re-collapse on every page load so the monitor always starts clean") —
    // never on every render, which would instantly undo an expansion
    // (toggleCollapse → render). Sessions/subagents are seeded only on a
    // FRESH start (no stored preference yet) so expanding an agent reveals
    // its sessions with their subagents still hidden; once the operator has
    // toggled anything, the stored preference rules and the seeding stops.
    if (tree && tree.agents) {
      if (!agentsSeededThisLoad) {
        agentsSeededThisLoad = true;
        for (const a of tree.agents) {
          if (a && a.agent != null) collapsed.add("agent:" + a.agent);
        }
      }
      // Fresh-start session seeding is scoped to the agent-grouped view:
      // the flat "none" view keeps its old fresh-start behaviour (subagents
      // visible), so drop any session-level seeds there.
      if (!hasStoredPref) {
        if (groupBy === "agent") {
          for (const a of tree.agents) {
            for (const root of (a.roots || [])) seedSessionCollapsed(root);
          }
        } else {
          for (const key of Array.from(collapsed)) {
            if (!key.startsWith("agent:")) collapsed.delete(key);
          }
        }
      }
    }
    rebuildFGOptions(tree);   // keep value-filter options in sync with live data
    const rows = buildRows(tree);

    if (rows.length === 0) {
      emptyEl.style.display = "";
      let fgActive = false;
      for (const d of fgDims) { if (fgSel[d] && fgSel[d].size) fgActive = true; }
      const filtered = maxAgeSec || query || fgActive;
      emptyEl.textContent = filtered
        ? "No sessions match the current filters."
        : "No sessions recorded yet.";
    } else {
      emptyEl.style.display = "none";
    }

    const seen = new Set();
    let prev = null;
    for (const item of rows) {
      seen.add(item.key);
      let row = rowEls.get(item.key);
      if (!row) { row = makeRow(item.key); rowEls.set(item.key, row); }
      if (item.kind === "agent") renderAgentRow(row, item);
      else if (item.kind === "group") renderGroupRow(row, item);
      else renderSessionRow(row, item);
      // Place in order. insertBefore on an already-correct node is a cheap
      // no-op-ish move that does not restart CSS animations.
      const want = prev ? prev.nextSibling : treeEl.firstChild;
      if (row !== want) treeEl.insertBefore(row, want);
      prev = row;
    }
    // Drop rows that vanished.
    for (const [key, el] of rowEls) {
      if (!seen.has(key)) { el.remove(); rowEls.delete(key); }
    }

    renderSummary(tree);
    renderTotals(rows);
    updatedEl.textContent = new Date().toLocaleTimeString();
  }

  function renderSummary(tree) {
    const items = (tree.summary || []).filter(
      (s) => s.active_sessions || s.subagents_calling
    );
    summaryEl.innerHTML = items.map(function (s) {
      const busy = s.active_sessions || s.subagents_calling;
      return (
        '<div class="agent-pill' + (busy ? " busy" : "") + '">' +
          '<span class="ap-name">' +
            '<span class="agent-swatch" style="--agc:' + agentColor(s.agent) + '"></span>' +
            escapeHtml(s.agent) +
          "</span>" +
          '<span class="ap-stats">' +
            '<span class="ap-stat" title="' + s.active_sessions + ' active session' +
              (s.active_sessions === 1 ? "" : "s") + '">⚡<b class="' +
              (s.active_sessions ? "hot" : "") + '">' + s.active_sessions + "</b></span>" +
            '<span class="ap-stat" title="' + s.subagents_calling + ' subagent' +
              (s.subagents_calling === 1 ? "" : "s") + ' delegating">🔗<b class="' +
              (s.subagents_calling ? "hot" : "") + '">' + s.subagents_calling + "</b></span>" +
          "</span>" +
        "</div>"
      );
    }).join("");
  }

  // Header counters reflect exactly what's RENDERED below — the server's
  // `totals` block counts only live/active sessions and disagreed with the
  // (idle-inclusive, filtered) row list, so it read "0 / 0 / 0" under a list of
  // visible rows. Count the actual session rows instead: agents = distinct
  // agent names on screen, sessions = top-level rows, subagents = nested rows.
  function renderTotals(rows) {
    const agents = new Set();
    let sessions = 0;
    let subagents = 0;
    for (const it of rows) {
      if (it.kind !== "session") continue;
      if (it.node && it.node.agent) agents.add(it.node.agent);
      if (it.isSub) subagents++; else sessions++;
    }
    totalsEl.innerHTML =
      '<span class="mon-total-chip' + (agents.size ? " hot" : "") + '"><b>' +
        agents.size + "</b> agents</span>" +
      '<span class="mon-total-chip' + (sessions ? " hot" : "") + '"><b>' +
        sessions + "</b> sessions</span>" +
      '<span class="mon-total-chip' + (subagents ? " hot" : "") + '"><b>' +
        subagents + "</b> subagents</span>";
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // ---- connection status ---------------------------------------------
  // Liveness (is the transport delivering frames at all?) is tracked
  // SEPARATELY from gateway health (is the upstream gateway reachable?).
  // `touch()` records that a frame — of ANY kind, including a gw_error —
  // arrived, so the stale watchdog only fires when the stream itself wedges.
  // The dot colour is set independently: live (flowing + gateway ok), down
  // (stream alive but gateway unreachable), stale (no frames at all).
  function setConn(cls) { connEl.className = "mon-conn mon-conn-" + cls; }
  function touch() { lastMsgAt = Date.now(); }
  function markAlive() { touch(); setConn("live"); }
  function markDown() { touch(); setConn("down"); }
  setInterval(function () {
    if (!lastMsgAt) return;
    // Polls arrive every 1s; a >8s gap means the poller itself is dead.
    if (Date.now() - lastMsgAt > 8000) setConn("stale");
  }, 2000);

  // ---- transport: 1s poll of the filtered session tree ---------------
  // Browse scope (?all=1) so idle sessions stay available to the STATE
  // filter; the remaining tab filters go on the wire so the payload is
  // already pruned.
  const MONITOR_POLL_MS = 1000;
  let inFlight = false;
  let lastTreeSig = "";
  function monitorQuery() {
    const params = new URLSearchParams();
    params.set("all", "1");
    params.set("limit", String(sessionLimit || 200));
    if (maxAgeSec) params.set("max_age_sec", String(maxAgeSec));
    if (query) params.set("q", query);
    for (const d of fgDims) {
      if (fgSel[d] && fgSel[d].size) params.set(d, [...fgSel[d]].join(","));
    }
    return params.toString();
  }
  function pollMonitor() {
    if (typeof document !== "undefined" && document.hidden) return;
    if (inFlight) return;
    inFlight = true;
    fetch(BASE + "/api/agents/monitor?" + monitorQuery(), { credentials: "same-origin" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (tree) {
        markAlive();
        const sig = JSON.stringify(tree);
        if (sig === lastTreeSig) return;
        lastTreeSig = sig;
        render(tree);
      })
      .catch(function () { markDown(); })
      .finally(function () { inFlight = false; });
  }
  function startPolling() {
    if (pollTimer) return;
    pollMonitor();
    pollTimer = setInterval(pollMonitor, MONITOR_POLL_MS);
  }
  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  // ---- client-side filters (age window + text search) ----------------
  // Reflect any persisted selection into the controls, then re-render the
  // already-cached tree locally for instant UI; kick a poll so the next
  // payload matches the new server-side filter.
  if (windowEl) windowEl.value = String(maxAgeSec);
  if (limitEl) limitEl.value = String(sessionLimit);
  if (searchEl) searchEl.value = query;
  if (groupByEl) groupByEl.value = groupBy;
  if (windowEl) windowEl.addEventListener("change", function () {
    maxAgeSec = parseInt(windowEl.value, 10) || 0;
    persistFilters();
    updateFilterSummary();
    lastTreeSig = "";
    if (lastTree) render(lastTree);
    pollMonitor();
  });
  if (limitEl) limitEl.addEventListener("change", function () {
    sessionLimit = parseInt(limitEl.value, 10) || 200;
    persistFilters();
    updateFilterSummary();
    lastTree = null;
    lastTreeSig = "";
    pollMonitor();
  });
  if (searchEl) searchEl.addEventListener("input", function () {
    query = (searchEl.value || "").trim().toLowerCase();
    persistFilters();
    updateFilterSummary();
    if (lastTree) render(lastTree);
    lastTreeSig = "";
    pollMonitor();
  });
  if (groupByEl) groupByEl.addEventListener("change", function () {
    if (GROUP_BYS.indexOf(groupByEl.value) !== -1) groupBy = groupByEl.value;
    persistFilters();
    updateFilterSummary();
    if (lastTree) render(lastTree);
  });
  // Value-filter checkboxes (delegated — options are rebuilt every render). An
  // "All" toggle clears/sets the whole dimension; a value toggle adds/removes it.
  const fgContainer = document.getElementById("mon-fg-groups");
  if (fgContainer) fgContainer.addEventListener("change", function (e) {
    const cb = e.target;
    if (!cb || !cb.classList || !cb.classList.contains("mon-fg-cb")) return;
    const dim = cb.dataset.fg;
    const fs = cb.closest("fieldset");
    if (cb.value === "__all__") {
      if (cb.checked) {
        fgSel[dim].clear();
      } else {
        fs.querySelectorAll('input.mon-fg-cb:not([value="__all__"])')
          .forEach((o) => fgSel[dim].add(o.value));
      }
    } else {
      if (cb.checked) fgSel[dim].add(cb.value); else fgSel[dim].delete(cb.value);
      const allCb = fs.querySelector('input.mon-fg-cb[value="__all__"]');
      if (allCb) allCb.checked = fgSel[dim].size === 0;
    }
    persistFG();
    updateFilterSummary();
    lastTreeSig = "";
    if (lastTree) render(lastTree);
    pollMonitor();
  });
  // First visit applied non-trivial defaults (show idle + ≤15m) — persist them
  // now so a reload is stable, and seed the collapsed-card summary chip.
  if (firstVisit) persistFilters();
  updateFilterSummary();

  // The Filters card is a <details> popover — natively it stays open until the
  // summary is clicked again. Close it on any pointer-down outside, like a
  // normal dropdown.
  const filterCard = document.getElementById("mon-filter-card");
  if (filterCard) document.addEventListener("pointerdown", function (e) {
    if (filterCard.open && !filterCard.contains(e.target)) filterCard.open = false;
  });

  // Navigation drawer + conversations are handled by the shared nav.js
  // (loaded before this script). See templates/_menu.html. These two title
  // helpers stayed behind because the tree (renderNode → .title) uses
  // displaySessionTitle for its session labels — keep them here.
  function isPlaceholderTitle(t) {
    return typeof t === "string" && (t === "(untitled)" || t.startsWith("untitled-"));
  }
  function displaySessionTitle(t) {
    return (t && !isPlaceholderTitle(t)) ? t : "New Chat";
  }

  window.addEventListener("beforeunload", function () {
    stopPolling();
  });
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) pollMonitor();
  });

  // Go.
  startPolling();
})();
