/* Agent Monitor (mini) — the Agents footer inside the composer input card.
 *
 * The footer is a one-line trigger; clicking it opens #agent-mon-pop, a card
 * styled like the agent selector (fixed sheet + backdrop, × / Escape /
 * backdrop close, bottom sheet on phones). Rows paint into the card so the
 * chat never shrinks. Only one footer sheet is open at a time.
 *
 * Scope: the current session lineage only.
 * Data: GET /api/agents/footer?session=<sid> (flat rows) or the
 * chat SSE `event: tree` `footer.rows` field. The full /agents tree is
 * not fetched. Switches render latest-first; children render newest
 * delegated first (delegation time, never last activity). Paint EVERY row —
 * ancestors, the switch history, and every delegated child at any depth —
 * into the scrollport. There is no row budget: the card bounds itself with
 * its CSS max-height and scrolls, so no row can be hidden from the operator.
 * The CURRENT row overlays the chat live status so an idle session cannot
 * stay "reasoning 0s". Sub rows arrive reconciled by the backend, and the
 * label ("3 active · 17 subs") renders the payload's backend ``counts`` —
 * the page never counts rows.
 * The card header Active filter hides only inactive subagents; switch history stays.
 * Do not render an empty "no delegations" row.
 *
 * CSP-safe: external file, no inline JS.
 */
(function () {
  "use strict";

  const BASE = ((document.querySelector('meta[name="sharedagents-base"]') || {}).content || "");
  const acc = document.getElementById("agent-mon-acc");
  const body = document.getElementById("ama-body");
  // Default is closed (`hidden` in the template). Never open the card from
  // paint/SSE — only the operator's click on the footer opens it.
  const pop = document.getElementById("agent-mon-pop");
  const backdrop = document.getElementById("ama-backdrop");
  const trigger = acc && acc.querySelector(".ama-summary");
  const countEl = document.getElementById("ama-count");
  const popCountEl = document.getElementById("ama-pop-count");
  const filterBtn = document.getElementById("ama-filter");
  if (!acc || !body) return;

  const FILTER_KEY = "sharedagents.web.amaFilter";
  let filterMode = "all";
  try {
    const saved = localStorage.getItem(FILTER_KEY);
    if (saved === "active" || saved === "all") filterMode = saved;
  } catch (_e) {}

  const MOCK = /[?&]ama_mock=1\b/.test(window.location.search);

  let cur = null;
  let lastRows = null;
  let lastCounts = null;
  let pollTimer = null;
  let fetchSeq = 0;
  let refreshInFlight = false;
  let abortCtl = null;

  function _sseOpen() {
    return typeof uiSseIsOpen === "function" && uiSseIsOpen();
  }

  const stopPressedAt = new Map();
  const STOP_HARD_WINDOW_MS = 10000;

  function sweepStopPressed() {
    const cutoff = Date.now() - 60000;
    for (const [h, ts] of stopPressedAt) {
      if (ts < cutoff) stopPressedAt.delete(h);
    }
  }

  function agentColor(name) {
    name = String(name || "");
    let h = 0;
    for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
    return "hsl(" + (h % 360) + " 70% 62%)";
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function ageStr(ts) {
    if (!ts) return "";
    const numeric = Number(ts);
    const ms = Number.isFinite(numeric)
      ? (numeric > 1e12 ? numeric : numeric * 1000)
      : Date.parse(String(ts));
    if (!Number.isFinite(ms)) return "";
    const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m";
    if (s < 86400) return Math.floor(s / 3600) + "h";
    return Math.floor(s / 86400) + "d";
  }
  const hkey = (h) => String(h || "").slice(0, 24);

  const TREE_BUSY_STATES = [
    "prompting", "prefill", "reasoning", "streaming", "tool_running",
    "compacting", "paused", "waiting", "ask", "retrying", "running",
  ];
  function rowBusy(row) {
    if (!row) return false;
    // The backend keeps open turns live even during silent tools. A legacy
    // busy state alone can belong to a closed, abandoned turn. Use the same
    // authoritative bit for the dot, Active filter and Stop.
    return !!row.live && TREE_BUSY_STATES.indexOf(String(row.state || "")) !== -1;
  }
  function rowOrDescendantBusy(row, rows, visited) {
    if (rowBusy(row)) return true;
    const key = hkey(row && row.hash);
    const seen = visited || new Set();
    if (!key || seen.has(key) || seen.size >= 128) return false;
    seen.add(key);
    for (const child of (rows || [])) {
      if (hkey(child.parent_hash) === key && rowOrDescendantBusy(child, rows, seen)) {
        return true;
      }
    }
    return false;
  }

  function currentSessionState() {
    // Same source as #status-text: the live session row, then chrome.
    const sid = state && state.currentSessionId;
    const row = sid && state.sessions && state.sessions[sid];
    const fromRow = row
      ? String(row.state || row.session_state
        || (row.activity && row.activity.phase) || "").trim()
      : "";
    const fromChrome = (cur && cur.stateLabel) || "";
    // A local turn paints chrome (reasoning / tool_running) before the
    // session row leaves "idle". An idle row must not clobber that.
    if (fromRow && fromRow !== "idle") return fromRow;
    if (fromChrome && fromChrome !== "idle") return fromChrome;
    return fromRow || fromChrome || "idle";
  }

  /** Overlay the open-session chrome onto the CURRENT row only. Sub rows
   *  arrive already reconciled against the live activity snapshot by the
   *  backend (``reconcile_sub_liveness``), which also counts them
   *  (``counts``); the page paints them as given. */
  function reconcileFooterRows(rows) {
    if (!rows || !rows.length) return rows || [];
    const status = currentSessionState();
    return rows.map(function (row) {
      if (row.role !== "current") return row;
      const next = Object.assign({}, row);
      if (cur && cur.agent) next.agent = cur.agent;
      if (cur && cur.title && cur.title !== "this session") next.title = cur.title;
      if (cur && cur.model) next.model = cur.model;
      if (status === "error" || String(next.state || "") === "error") {
        next.state = "error";
        next.live = false;
      } else {
        // Always overlay the live session status. A stale tree snapshot
        // used to keep "idle" on the current row while #status-text
        // already showed reasoning / tool_running / waiting.
        next.state = status;
        next.live = status === "waiting" || status === "running"
          || TREE_BUSY_STATES.indexOf(status) !== -1;
      }
      return next;
    });
  }

  function rowHtml(o, depth, role) {
    const color = agentColor(o.agent);
    let label = String(o.stateLabel || (o.live ? "running" : "idle"));
    const busy = rowBusy({ live: o.live, state: label });
    if (!busy && TREE_BUSY_STATES.indexOf(label) !== -1) label = "idle";
    const stateCls = label === "error" ? "error"
      : label === "waiting" ? "waiting"
      : (busy ? "live" : "idle");
    const tag = role === "current"
      ? '<span class="ama-tag ama-tag-current">● current</span>'
      : role === "previous"
        ? '<span class="ama-tag ama-tag-previous">↳ previous</span>'
      : role === "parent"
        ? '<span class="ama-tag ama-tag-parent">parent</span>'
        : '<span class="ama-tag ama-tag-sub">🔗 sub</span>';
    let meta = "";
    if (label) meta += '<span class="ama-state ama-state-' + stateCls + '">' + esc(label) + "</span>";
    if (o.age) meta += '<span class="ama-dim">' + esc(o.age) + "</span>";
    if (o.subcount) meta += '<span class="ama-subcount">🔗' + Number(o.subcount) + "</span>";
    const hashAttr = o.hash ? ' data-hash="' + esc(o.hash) + '"' : "";
    const scopeAttr = o.scopeHash
      ? ' data-scope-hash="' + esc(o.scopeHash) + '"'
      : "";
    const navigable = !!o.hash && (role === "parent" || role === "sub");
    const clickable = navigable ? " ama-clickable" : "";
    const titleAttr = navigable ? ' title="Open this session"' : "";
    const stopHtml = o.stopVisible
      ? '<button type="button" class="ama-stop" data-stop-hash="' + esc(o.hash) + '"' +
        ' title="Stop this subagent (its own children keep running)"' +
        ' aria-label="Stop subagent">✕</button>'
      : "";
    return (
      '<div class="ama-row' + (role === "current" ? " is-current" : "") + clickable + '"' +
        hashAttr + scopeAttr + titleAttr +
        ' style="padding-left:' + (8 + depth * 16) + 'px">' +
        '<span class="ama-dot ama-st-' + stateCls + '"' + (busy ? ' data-live="1"' : "") + '></span>' +
        '<span class="ama-agent" style="color:' + color + '">' + esc(o.agent || "?") + "</span>" +
        (o.model ? '<span class="ama-model ama-dim">' + esc(o.model) + "</span>" : "") +
        '<span class="ama-ttl">' + esc(o.title || "") + "</span>" +
        tag +
        '<span class="ama-meta">' + meta + "</span>" +
        stopHtml +
      "</div>"
    );
  }

  function flattenFooterRows(tree, viewedHash) {
    if (!tree || typeof tree !== "object") return [];
    const viewedKey = hkey(viewedHash);
    const nodes = new Map();
    const parentOf = new Map();
    function walk(node, parent) {
      const key = String((node && node.hash) || "");
      if (key && !nodes.has(key)) {
        nodes.set(key, node);
        parentOf.set(key, parent || null);
      }
      for (const child of ((node && node.children) || [])) walk(child, node);
    }
    for (const bucket of (tree.agents || [])) {
      for (const root of ((bucket && bucket.roots) || [])) walk(root, null);
    }
    if (!nodes.size) return [];
    let roots = [];
    nodes.forEach(function (node, key) {
      if (!parentOf.get(key)) roots.push(node);
    });
    let currentHash = viewedHash;
    let current = nodes.get(viewedHash) || null;
    if (!current && viewedKey) {
      nodes.forEach(function (node, key) {
        if (!current && hkey(key) === viewedKey) current = node;
      });
    }
    if (current) {
      currentHash = String(current.hash || "");
    }
    const currentKey = hkey(currentHash);
    const ancestors = new Set();
    for (let guard = 0; current && guard < 128; guard++) {
      const parent = parentOf.get(String(current.hash || ""));
      if (!parent) break;
      ancestors.add(String(parent.hash || ""));
      current = parent;
    }
    function roleFor(node, depth) {
      if (currentKey && hkey(node.hash) === currentKey) return "current";
      // Mirror of the server rule: every ancestor of the viewed session is a
      // parent lane, so selecting a nested subagent still shows its whole
      // tree with the chain above it marked as ancestry (and excluded from
      // the delegated-child count).
      if (ancestors.has(String(node.hash || ""))) return "parent";
      if (depth === 0) return "previous";
      return "sub";
    }
    function toMs(val) {
      if (!val) return 0;
      if (typeof val === "number") return val > 1000000000000 ? val : val * 1000;
      const parsed = Date.parse(String(val));
      if (!isNaN(parsed)) return parsed;
      const num = Number(val);
      if (!isNaN(num) && num > 0) return num > 1000000000000 ? num : num * 1000;
      return 0;
    }
    function hashTimeMs(hash) {
      const stamp = String(hash || "").split("-")[0];
      if (stamp.length < 15 || stamp.charAt(8) !== "T") return 0;
      const y = Number(stamp.slice(0, 4));
      const mo = Number(stamp.slice(4, 6)) - 1;
      const d = Number(stamp.slice(6, 8));
      const h = Number(stamp.slice(9, 11));
      const mi = Number(stamp.slice(11, 13));
      const se = Number(stamp.slice(13, 15));
      const ms = Math.floor(Number(stamp.slice(15, 21) || "0") / 1000);
      const t = Date.UTC(y, mo, d, h, mi, se, ms);
      return Number.isFinite(t) ? t : 0;
    }
    function delegationMs(node) {
      return toMs(node && node.created_at) || toMs(node && node.started_at)
        || hashTimeMs(node && node.hash) || 0;
    }
    function creationKey(node) {
      return [delegationMs(node), String((node && node.hash) || "")];
    }
    const rows = [];
    function emit(node, depth) {
      const parent = parentOf.get(String(node.hash || ""));
      const key = String((node && node.hash) || "");
      const footerState = (node && node.footer_state) || {};
      const sequence = footerState.sequence || [];
      const current = footerState.current || {};

      const kids = ((node && node.children) || []).slice().sort(function (a, b) {
        const ka = creationKey(a), kb = creationKey(b);
        if (ka[0] !== kb[0]) return kb[0] - ka[0];
        return ka[1] < kb[1] ? 1 : ka[1] > kb[1] ? -1 : 0;
      });

      const activations = [];
      if (depth === 0 && Array.isArray(sequence) && sequence.length > 1) {
        for (let idx = 0; idx < sequence.length; idx++) {
          const act = sequence[idx];
          if (!act || typeof act !== "object") continue;
          activations.push({
            index: idx,
            agent: act.agent || "?",
            model: act.model || "",
            timestamp: act.timestamp || 0,
            time_ms: toMs(act.timestamp),
            is_current: (idx === sequence.length - 1),
          });
        }
        const collapsed = [];
        for (let i = 0; i < activations.length; i++) {
          const act = activations[i];
          const prev = collapsed.length ? collapsed[collapsed.length - 1] : null;
          if (prev && prev.agent === act.agent) {
            if (act.model) prev.model = act.model;
            prev.is_current = act.is_current;
            prev.index = act.index;
            prev.timestamp = act.timestamp || prev.timestamp;
            prev.time_ms = act.time_ms || prev.time_ms;
          } else {
            collapsed.push(act);
          }
        }
        activations.length = 0;
        for (let i = 0; i < collapsed.length; i++) {
          collapsed[i].index = i;
          activations.push(collapsed[i]);
        }
      }

      if (!activations.length) {
        rows.push({
          hash: key,
          agent: current.agent || node.agent || "?",
          model: current.model || node.model || "",
          title: node.title || "",
          state: node.state || (node.live ? "running" : "idle"),
          live: !!node.live,
          depth: depth,
          role: roleFor(node, depth),
          last_activity_at: node.last_activity_at || node.last_turn_ts || node.state_updated_at || 0,
          created_at: node.created_at || "",
          parent_hash: (parent && parent.hash) || "",
          subagent_count: node.subagent_count || 0,
        });
        for (let i = 0; i < kids.length; i++) emit(kids[i], depth + 1);
        return;
      }

      const actKids = {};
      for (let i = 0; i < activations.length; i++) actKids[i] = [];
      for (let i = 0; i < kids.length; i++) {
        const child = kids[i];
        const childMs = creationKey(child)[0];
        let assignedIdx = 0;
        for (let j = 0; j < activations.length; j++) {
          const act = activations[j];
          if (act.time_ms && childMs >= act.time_ms) {
            assignedIdx = j;
          } else if (!act.time_ms && j === 0) {
            assignedIdx = 0;
          }
        }
        actKids[assignedIdx].push(child);
      }

      for (let i = activations.length - 1; i >= 0; i--) {
        const act = activations[i];
        const idx = act.index;
        const isCur = act.is_current;
        const role = isCur ? roleFor(node, depth) : "previous";
        rows.push({
          hash: isCur ? key : "",
          agent: act.agent,
          // The CURRENT lane falls back to the freshest session model
          // (footer_state.current / node model) — an activation recorded
          // before model resolution can carry an empty model while the
          // session header already knows the real one. Historical
          // "previous"/"switched" rows keep their recorded activation
          // model; an empty value simply renders without a label.
          model: isCur
            ? (node.model || current.model || act.model || node.engine_model || "")
            : (act.model || ""),
          title: node.title || "",
          state: isCur ? (node.state || (node.live ? "running" : "idle")) : "switched",
          live: isCur ? !!node.live : false,
          depth: depth,
          role: role,
          last_activity_at: isCur ? (node.last_activity_at || node.last_turn_ts || node.state_updated_at || 0) : (act.timestamp || 0),
          created_at: isCur ? (node.created_at || "") : (act.timestamp || ""),
          parent_hash: isCur ? ((parent && parent.hash) || "") : "",
          subagent_count: actKids[idx].length,
        });
        for (let j = 0; j < actKids[idx].length; j++) {
          emit(actKids[idx][j], depth + 1);
        }
      }
    }
    for (let i = 0; i < roots.length; i++) emit(roots[i], 0);
    return rows;
  }

  function persistFilter() {
    try { localStorage.setItem(FILTER_KEY, filterMode); } catch (_e) {}
  }
  function syncFilterBtn() {
    if (!filterBtn) return;
    const active = filterMode === "active";
    filterBtn.classList.toggle("is-on", active);
    filterBtn.setAttribute("aria-pressed", active ? "true" : "false");
    filterBtn.textContent = "Active";
    filterBtn.title = active
      ? "Hiding inactive subagents — click to show all subagents"
      : "Hide inactive subagents";
  }
  function visibleFooterRows(rows) {
    if (!rows || !rows.length || filterMode !== "active") return rows || [];
    return rows.filter(function (row) {
      if (!row || row.role !== "sub") return true;
      return rowBusy(row);
    });
  }

  function reorderSubsByDelegation(rows) {
    if (!rows || rows.length < 2) return rows || [];
    function hashTimeMs(hash) {
      const stamp = String(hash || "").split("-")[0];
      if (stamp.length < 15 || stamp.charAt(8) !== "T") return 0;
      const y = Number(stamp.slice(0, 4));
      const mo = Number(stamp.slice(4, 6)) - 1;
      const d = Number(stamp.slice(6, 8));
      const h = Number(stamp.slice(9, 11));
      const mi = Number(stamp.slice(11, 13));
      const se = Number(stamp.slice(13, 15));
      const ms = Math.floor(Number(stamp.slice(15, 21) || "0") / 1000);
      const t = Date.UTC(y, mo, d, h, mi, se, ms);
      return Number.isFinite(t) ? t : 0;
    }
    function toMsLocal(val) {
      if (!val) return 0;
      const numeric = Number(val);
      if (Number.isFinite(numeric) && numeric > 0) {
        return numeric > 1e12 ? numeric : numeric * 1000;
      }
      const parsed = Date.parse(String(val));
      return Number.isFinite(parsed) ? parsed : 0;
    }
    function delegationMs(row) {
      return toMsLocal(row && row.created_at) || toMsLocal(row && row.started_at)
        || hashTimeMs(row && row.hash) || 0;
    }
    function pack(list) {
      if (!list.length) return [];
      const depth = Number(list[0].depth || 0);
      const chunks = [];
      let i = 0;
      while (i < list.length) {
        const row = list[i];
        const d = Number(row.depth || 0);
        if (d < depth) break;
        let j = i + 1;
        while (j < list.length && Number(list[j].depth || 0) > d) j += 1;
        chunks.push({ row: row, descendants: pack(list.slice(i + 1, j)) });
        i = j;
      }
      const subIdx = [];
      for (let k = 0; k < chunks.length; k++) {
        if ((chunks[k].row.role || "sub") === "sub") subIdx.push(k);
      }
      if (subIdx.length > 1) {
        const ordered = subIdx.slice().sort(function (a, b) {
          const da = delegationMs(chunks[a].row);
          const db = delegationMs(chunks[b].row);
          if (da !== db) return db - da;
          const ha = String(chunks[a].row.hash || "");
          const hb = String(chunks[b].row.hash || "");
          return ha < hb ? 1 : ha > hb ? -1 : 0;
        });
        const copies = ordered.map(function (idx) { return chunks[idx]; });
        for (let n = 0; n < subIdx.length; n++) chunks[subIdx[n]] = copies[n];
      }
      const out = [];
      for (let k = 0; k < chunks.length; k++) {
        out.push(chunks[k].row);
        for (let n = 0; n < chunks[k].descendants.length; n++) {
          out.push(chunks[k].descendants[n]);
        }
      }
      return out;
    }
    return pack(rows.slice());
  }

  // The footer and the card header show the same label, rendered from the
  // backend ``counts`` ({active, total, agents}); the page never counts rows.
  function countsFrom(payload) {
    if (!payload || typeof payload !== "object") return null;
    const c = payload.counts;
    if (c && typeof c === "object") {
      return {
        active: Number(c.active) || 0,
        total: Number(c.total) || 0,
        agents: Number(c.agents) || 0,
      };
    }
    if (typeof payload.subs === "number") {
      return { active: 0, total: payload.subs, agents: 0 };
    }
    return null;
  }
  function countHtml(c) {
    if (!c) return "";
    const parts = [];
    if (c.agents > 1) parts.push(esc(c.agents + " agents"));
    if (c.active > 0) {
      parts.push('<span class="ama-count-active">' + esc(c.active + " active") + "</span>");
    }
    if (c.total > 0) parts.push(esc(c.total + " sub" + (c.total === 1 ? "" : "s")));
    return parts.join(" · ");
  }
  function setCount(c) {
    const html = countHtml(c);
    if (countEl) countEl.innerHTML = html;
    if (popCountEl) popCountEl.innerHTML = html;
  }

  function paintRows(rows) {
    if (!cur || !cur.active) {
      body.innerHTML = '<div class="ama-empty">No session open.</div>';
      setCount(null);
      return;
    }
    if (!rows || !rows.length) {
      // No lineage rows yet (fresh session before the first tree frame, or a
      // session whose monitor tree is empty). Render the single current-lane
      // row with the model the chat page already knows — it must not vanish
      // just because the tree snapshot has not landed.
      body.innerHTML = rowHtml({
        agent: cur.agent, title: cur.title,
        model: (cur && cur.model) || "",
        live: cur.live, stateLabel: cur.stateLabel || (cur.live ? "running" : "idle"),
      }, 0, "current");
      setCount(lastCounts);
      return;
    }
    const reconciled = reconcileFooterRows(rows);
    // No row budget: the dock scrolls, so every row the session's lineage
    // produced is painted (the `Active` toggle is a filter, not a cap).
    const filtered = reorderSubsByDelegation(visibleFooterRows(reconciled));
    let html = "";
    for (let i = 0; i < filtered.length; i++) {
      const row = filtered[i];
      const role = row.role || "sub";
      const isDelegation = !!row.hash && (
        role === "sub" || (role === "current" && (row.depth || 0) > 0)
      );
      html += rowHtml({
        agent: row.agent,
        model: row.model || "",
        title: row.title || "",
        hash: row.hash || "",
        live: rowBusy(row),
        stateLabel: row.state || (row.live ? "running" : "idle"),
        age: ageStr(row.last_activity_at),
        subcount: filterMode === "active"
          ? filtered.filter(function (r) {
              return r.role === "sub" && hkey(r.parent_hash) === hkey(row.hash);
            }).length
          : (row.subagent_count || 0),
        scopeHash: row.parent_hash || row.hash || "",
        stopVisible: isDelegation && rowOrDescendantBusy(row, filtered),
      }, Number(row.depth || 0), role);
    }
    body.innerHTML = html;
    setCount(lastCounts);
  }

  function applySnapshot(payload) {
    if (MOCK) { renderMock(); return; }
    const counts = countsFrom(payload && payload.footer) || countsFrom(payload);
    if (counts) lastCounts = counts;
    if (payload && payload.footer && Array.isArray(payload.footer.rows)) {
      lastRows = payload.footer.rows;
      paintRows(lastRows);
      return;
    }
    if (payload && Array.isArray(payload.rows)) {
      lastRows = payload.rows;
      paintRows(lastRows);
      return;
    }
    if (Array.isArray(payload)) {
      lastRows = payload;
      paintRows(lastRows);
      return;
    }
    const tree = payload && payload.tree ? payload.tree : payload;
    if (!tree || typeof tree !== "object") return;
    lastRows = flattenFooterRows(tree, cur && cur.hash);
    paintRows(lastRows);
  }

  async function refresh(opts) {
    if (MOCK) { renderMock(); return; }
    if (!cur || !cur.active) {
      lastRows = null;
      body.innerHTML = '<div class="ama-empty">No session open.</div>';
      setCount(null);
      return;
    }
    if (!cur.hash) { paintRows([]); return; }
    // The single session stream's `state` frame and initial boot payload carry the footer rows
    // (applySnapshot). Never fetch /api/agents/footer from REST during ordinary chat or stream.
    if (lastRows && lastRows.length) {
      paintRows(lastRows);
      return;
    }
    const force = !!(opts && opts.allowFetch);
    if (!force) return;
    if (refreshInFlight) return;
    refreshInFlight = true;
    try {
      await fetchFooter();
    } finally {
      refreshInFlight = false;
    }
  }

  async function fetchFooter() {
    const seq = ++fetchSeq;
    const scopeHash = cur.scopeHash || cur.hash;
    if (abortCtl) {
      try { abortCtl.abort(); } catch (_e) {}
    }
    abortCtl = (typeof AbortController !== "undefined") ? new AbortController() : null;
    try {
      const r = await fetch(
        BASE + "/api/agents/footer?session=" + encodeURIComponent(scopeHash),
        {
          headers: { Accept: "application/json" },
          signal: abortCtl ? abortCtl.signal : undefined,
        },
      );
      if (seq !== fetchSeq) return;
      if (!r.ok) { paintRows(lastRows || []); return; }
      const payload = await r.json();
      if (seq !== fetchSeq) return;
      applySnapshot(payload);
    } catch (e) {
      if (e && e.name === "AbortError") return;
      paintRows(lastRows || []);
    }
  }

  body.addEventListener("click", function (e) {
    const stopBtn = e.target.closest(".ama-stop");
    if (stopBtn) {
      e.stopPropagation();
      e.preventDefault();
      const hash = stopBtn.getAttribute("data-stop-hash") || "";
      if (!hash) return;
      sweepStopPressed();
      const now = Date.now();
      const prev = stopPressedAt.get(hash);
      const force = !!prev && (now - prev) < STOP_HARD_WINDOW_MS;
      stopPressedAt.set(hash, now);
      stopBtn.disabled = true;
      stopBtn.classList.add("ama-stop-busy");
      apiPost(BASE + "/api/agents/abort", {
        session_id: hash,
        force: force || undefined,
      }).then(function () {
        refresh({ force: true });
      }).catch(function (err) {
        if (window.appendSystemMessage) {
          appendSystemMessage("Abort error: " + esc(err.message), true);
        }
      }).finally(function () {
        stopBtn.disabled = false;
        stopBtn.classList.remove("ama-stop-busy");
      });
      return;
    }
    const row = e.target.closest(".ama-row[data-hash]");
    if (!row || !row.classList.contains("ama-clickable")) return;
    const hash = row.getAttribute("data-hash");
    if (!hash) return;
    const parent = row.querySelector(".ama-tag-parent");
    const sub = row.querySelector(".ama-tag-sub");
    closeCard({ restoreFocus: false });
    if (window.saOpenConversation) {
      if (parent && cur && cur.returnSessionId) {
        window.saOpenConversation(cur.returnSessionId);
      } else if (sub) {
        window.saOpenConversation(hash, {
          fromMonitor: true,
          scopeHash: row.getAttribute("data-scope-hash") || "",
        });
      } else {
        window.saOpenConversation(hash);
      }
    }
    else window.location.href = BASE + "/?session=" + encodeURIComponent(hash);
  });

  function syncPoll() {
    const want = isOpen() && cur && cur.active && cur.live &&
      !(typeof uiSseIsOpen === "function" && uiSseIsOpen());
    if (want && !pollTimer) pollTimer = setInterval(function () {
      if (document.hidden) return;
      refresh({ force: true });
    }, 4000);
    else if (!want && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  if (filterBtn) {
    syncFilterBtn();
    filterBtn.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      filterMode = filterMode === "active" ? "all" : "active";
      persistFilter();
      syncFilterBtn();
      paintRows(lastRows || []);
    });
  }

  // ── Card open / close (same contract as the agent selector sheet) ──
  let ignoreBackdropUntil = 0;

  function isOpen() {
    return !!pop && !pop.hidden;
  }

  function openCard() {
    if (!pop || isOpen()) return;
    // Only one footer sheet may be open at a time.
    if (typeof closeModelSheet === "function") {
      const sheet = document.getElementById("agent-model-pop");
      if (sheet && !sheet.hidden) closeModelSheet();
    }
    if (typeof closeBudgetEditor === "function") {
      const editor = document.getElementById("budget-editor");
      if (editor && !editor.hidden) closeBudgetEditor();
    }
    if (window.SaContextSheet && window.SaContextSheet.isOpen()) {
      window.SaContextSheet.close({ restoreFocus: false });
    }
    ignoreBackdropUntil = Date.now() + 400;
    pop.hidden = false;
    if (backdrop) backdrop.hidden = false;
    if (trigger) trigger.setAttribute("aria-expanded", "true");
    refresh();
    syncPoll();
    window.setTimeout(function () {
      const close = document.getElementById("ama-pop-close");
      if (close && isOpen()) close.focus();
    }, 0);
  }

  function closeCard(opts) {
    if (!pop || !isOpen()) return;
    pop.hidden = true;
    if (backdrop) backdrop.hidden = true;
    if (trigger) {
      trigger.setAttribute("aria-expanded", "false");
      if (!opts || opts.restoreFocus !== false) trigger.focus();
    }
    syncPoll();
  }

  if (trigger && pop) {
    trigger.addEventListener("click", function (e) {
      // The budget control stops propagation; the /agents link
      // must keep navigating instead of opening the card.
      if (e.target.closest && e.target.closest(".ama-full")) return;
      openCard();
    });
    trigger.addEventListener("keydown", function (e) {
      if (e.target !== trigger) return;
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      openCard();
    });
    const closeBtn = document.getElementById("ama-pop-close");
    if (closeBtn) closeBtn.addEventListener("click", function () { closeCard(); });
    if (backdrop) backdrop.addEventListener("click", function () {
      if (Date.now() < ignoreBackdropUntil) return;
      closeCard();
    });
    document.addEventListener("keydown", function (e) {
      if (!isOpen()) return;
      if (e.key === "Escape") {
        // Dismiss the modal before the application's stream-cancel shortcut.
        e.preventDefault();
        e.stopImmediatePropagation();
        closeCard();
        return;
      }
      if (e.key !== "Tab") return;
      const focusable = [...pop.querySelectorAll("button:not([disabled])")];
      if (!focusable.length) return;
      const first = focusable[0], last = focusable[focusable.length - 1];
      // A live repaint can remove the focused Stop button. Recover focus
      // inside the modal rather than tabbing into the background composer.
      if (!pop.contains(document.activeElement)) {
        e.preventDefault(); (e.shiftKey ? last : first).focus();
      } else if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus();
      }
    }, true);
  }

  window.SaAgentMon = {
    update: function (data) {
      const next = data || { active: false };
      const hashChanged = !cur || hkey(cur.hash) !== hkey(next.hash);
      const agentChanged = !cur || String(cur.agent || "") !== String(next.agent || "");
      cur = next;
      if (hashChanged) {
        lastRows = null;
        lastCounts = null;
      }
      if (next.footer) {
        applySnapshot(next.footer);
      } else if (hashChanged || agentChanged) {
        paintRows(lastRows || []);
      } else if (lastRows) {
        paintRows(lastRows);
      }
      syncPoll();
    },
    applySnapshot: applySnapshot,
    syncPoll: syncPoll,
    open: openCard,
    close: closeCard,
    isOpen: isOpen,
  };

  function renderMock() {
    if (!cur || !cur.active) {
      cur = {
        active: true,
        agent: "opencode",
        title: "summarize what was done in this session",
        hash: "dddd4444opencode",
        live: true,
        stateLabel: "streaming",
      };
    }
    const now = Date.now() / 1000;
    lastRows = [
      { agent: "opencode", title: "summarize what was done in this session", hash: "20260903T101000000000-opencode",
        live: true, state: "streaming", last_activity_at: now - 8, created_at: now - 8 * 60,
        depth: 0, role: "current", parent_hash: "", subagent_count: 0 },
      { agent: "jarvis", title: "route the footer fix", hash: "20260903T100500000000-jarvis",
        live: false, state: "switched", last_activity_at: now - 16 * 60, created_at: now - 16 * 60,
        depth: 0, role: "previous", parent_hash: "", subagent_count: 3 },
      { agent: "build", title: "dock monitor inside input card", hash: "20260903T100700000000-build",
        live: true, state: "streaming", last_activity_at: now - 40, created_at: now - 10 * 60,
        depth: 1, role: "sub", parent_hash: "20260903T100500000000-jarvis", subagent_count: 0 },
      { agent: "explore", title: "find footer markup", hash: "20260903T100600000000-explore",
        live: true, state: "tool_running", last_activity_at: now - 2, created_at: now - 11 * 60,
        depth: 1, role: "sub", parent_hash: "20260903T100500000000-jarvis", subagent_count: 0 },
      { agent: "cursor", title: "semaphore lock report", hash: "20260903T100550000000-cursor",
        live: true, state: "tool_running", last_activity_at: now - 1, created_at: now - 12 * 60,
        depth: 1, role: "sub", parent_hash: "20260903T100500000000-jarvis", subagent_count: 0 },
      { agent: "plan", title: "design the footer dock", hash: "20260903T100000000000-plan",
        live: false, state: "switched", last_activity_at: now - 20 * 60, created_at: now - 20 * 60,
        depth: 0, role: "previous", parent_hash: "", subagent_count: 0 },
      { agent: "antigravity", title: "earlier investigation", hash: "20260903T095000000000-agy",
        live: false, state: "switched", last_activity_at: now - 30 * 60, created_at: now - 30 * 60,
        depth: 0, role: "previous", parent_hash: "", subagent_count: 0 },
    ];
    lastCounts = { active: 3, total: 3, agents: 1 };
    paintRows(lastRows);
  }
  if (MOCK) openCard();
})();
