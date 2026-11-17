/* context_viewer.js — the full-screen Context sheet.
 *
 * The composer's context meter opens it. The backend decides what the context
 * is, pages it and sizes every item (GET /api/sessions/<sid>/context); this
 * file only renders the pages it is given.
 *
 * Infinite scroll: an IntersectionObserver on a sentinel below the list asks
 * for the next page. The list is WINDOWED — rows have one fixed height and only
 * the rows near the viewport exist in the DOM — so a context of thousands of
 * items (an 800K-token session) keeps a small, constant DOM. One item at a
 * time can be expanded; its body scrolls inside a bounded box.
 *
 * One sheet at a time with the agent picker, the Agents card and the budget
 * editor. Escape / × / backdrop close it and focus returns to the meter.
 */
(function () {
  "use strict";

  const OVERSCAN = 8;
  const PAGE_LIMIT = 50;
  const PREFETCH_PX = 900;

  const $ = (id) => document.getElementById(id);
  const sheet = $("ctx-sheet");
  const backdrop = $("ctx-backdrop");
  if (!sheet) return;
  const scroller = $("ctx-scroll");
  const list = $("ctx-list");
  const sentinel = $("ctx-sentinel");
  const status = $("ctx-status");
  const totalNode = $("ctx-total");
  const meterNode = $("ctx-meter");
  const compactBtn = $("ctx-compact");
  const closeBtn = $("ctx-close");
  const confirmNode = $("ctx-confirm");
  const noticeNode = $("ctx-notice");

  const view = {
    sid: "",
    gen: 0,
    items: [],
    snapshot: "",
    nextCursor: null,
    loading: false,
    error: "",
    expanded: -1,
    expandedHeight: 0,
    full: {},        // index -> {text, nextOffset, loading}
    rowHeight: 56,
    rows: new Map(), // index -> row node currently in the DOM
    canCompact: false,
    compactNote: "",
  };
  let ignoreBackdropUntil = 0;
  let observer = null;
  let rafPending = false;

  function base() {
    return typeof BASE === "string" ? BASE : "";
  }

  function currentSid() {
    return (typeof state !== "undefined" && state && state.currentSessionId) || "";
  }

  function isOpen() {
    return !sheet.hidden;
  }

  // ── data ──────────────────────────────────────────────────────────────

  async function fetchJson(path) {
    const resp = await fetch(base() + path, { credentials: "same-origin" });
    const data = await resp.json().catch(() => ({}));
    return { status: resp.status, data: data || {} };
  }

  function contextUrl(cursor) {
    const q = new URLSearchParams({ limit: String(PAGE_LIMIT) });
    if (cursor) q.set("cursor", cursor);
    return "/api/sessions/" + encodeURIComponent(view.sid) + "/context?" + q.toString();
  }

  function resetView() {
    view.gen += 1;
    view.items = [];
    view.snapshot = "";
    view.nextCursor = null;
    view.loading = false;
    view.error = "";
    view.expanded = -1;
    view.expandedHeight = 0;
    view.full = {};
    view.rows.forEach((node) => node.remove());
    view.rows.clear();
    scroller.scrollTop = 0;
    layout();
  }

  async function loadPage(first) {
    if (view.loading) return;
    if (!first && !view.nextCursor) return;
    const gen = view.gen;
    view.loading = true;
    view.error = "";
    sheet.setAttribute("aria-busy", "true");
    paintStatus();
    let res;
    try {
      res = await fetchJson(contextUrl(first ? "" : view.nextCursor));
    } catch (e) {
      res = { status: 0, data: { message: "Gateway unreachable" } };
    }
    if (gen !== view.gen) return; // closed / reloaded meanwhile
    view.loading = false;
    sheet.setAttribute("aria-busy", "false");
    const d = res.data || {};
    if (res.status === 409 && d.code === "context_changed") {
      reload();
      return;
    }
    if (res.status !== 200 || d.ok === false) {
      view.error = d.message || d.error || "Could not load the context";
      paintStatus();
      return;
    }
    if (first) applyHeader(d);
    view.snapshot = d.snapshot || view.snapshot;
    view.nextCursor = d.next_cursor || null;
    for (const item of d.items || []) view.items.push(item);
    if (first && !view.items.length) view.error = "";
    layout();
    paintStatus(d.empty_text || "");
    maybePrefetch();
  }

  function reload() {
    resetView();
    return loadPage(true);
  }

  function applyHeader(d) {
    const t = d.totals || {};
    totalNode.textContent = t.label || "";
    meterNode.textContent = t.meter_label || "";
    meterNode.hidden = !t.meter_label;
    const lines = [];
    if (d.notice) lines.push(d.notice);
    for (const note of d.notes || []) lines.push(note);
    noticeNode.textContent = lines.join(" ");
    noticeNode.hidden = !lines.length;
    view.canCompact = !!d.can_compact;
    paintCompact();
  }

  function paintStatus(emptyText) {
    if (view.error) {
      status.textContent = view.error;
      status.dataset.state = "error";
    } else if (view.loading) {
      status.textContent = "Loading…";
      status.dataset.state = "loading";
    } else if (!view.items.length) {
      status.textContent = emptyText || "";
      status.dataset.state = "empty";
    } else {
      status.textContent = "";
      status.dataset.state = "";
    }
  }

  function maybePrefetch() {
    if (!isOpen() || view.loading || !view.nextCursor) return;
    const remaining = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
    if (remaining < PREFETCH_PX) loadPage(false);
  }

  // ── windowed list ─────────────────────────────────────────────────────

  function extra() {
    return view.expanded >= 0 ? Math.max(0, view.expandedHeight - view.rowHeight) : 0;
  }

  function topOf(i) {
    return i * view.rowHeight + (view.expanded >= 0 && i > view.expanded ? extra() : 0);
  }

  function indexAt(y) {
    const h = view.rowHeight;
    if (view.expanded >= 0) {
      const expTop = view.expanded * h;
      if (y >= expTop && y < expTop + view.expandedHeight) return view.expanded;
      if (y >= expTop + view.expandedHeight) {
        return Math.floor((y - extra()) / h);
      }
    }
    return Math.floor(y / h);
  }

  function layout() {
    list.style.height = (view.items.length * view.rowHeight + extra()) + "px";
    paintWindow();
  }

  function schedulePaint() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(function () {
      rafPending = false;
      paintWindow();
      maybePrefetch();
    });
  }

  function paintWindow() {
    const n = view.items.length;
    if (!n || !isOpen()) {
      view.rows.forEach((node) => node.remove());
      view.rows.clear();
      return;
    }
    const top = scroller.scrollTop;
    const bottom = top + (scroller.clientHeight || 600);
    const first = Math.max(0, indexAt(top) - OVERSCAN);
    const last = Math.min(n - 1, indexAt(bottom) + OVERSCAN);
    view.rows.forEach(function (node, i) {
      if (i < first || i > last) {
        node.remove();
        view.rows.delete(i);
      }
    });
    for (let i = first; i <= last; i++) {
      let node = view.rows.get(i);
      const expanded = i === view.expanded;
      if (!node || node.dataset.expanded !== String(expanded)) {
        if (node) node.remove();
        node = buildRow(view.items[i], expanded);
        view.rows.set(i, node);
        list.appendChild(node);
      }
      node.style.top = topOf(i) + "px";
    }
  }

  function buildRow(item, expanded) {
    const row = document.createElement("div");
    row.className = "ctx-row";
    row.setAttribute("role", "listitem");
    row.dataset.index = String(item.index);
    row.dataset.kind = item.kind || "";
    row.dataset.expanded = String(expanded);
    if (item.withheld) row.dataset.withheld = "true";

    const head = document.createElement("button");
    head.type = "button";
    head.className = "ctx-row-head";
    head.setAttribute("aria-expanded", String(expanded));
    const idx = document.createElement("span");
    idx.className = "ctx-idx";
    idx.textContent = "#" + item.index;
    const kind = document.createElement("span");
    kind.className = "ctx-kind";
    kind.textContent = item.kind_label || item.kind || "";
    const label = document.createElement("span");
    label.className = "ctx-label";
    label.textContent = item.label || "";
    const tok = document.createElement("span");
    tok.className = "ctx-tok";
    tok.textContent = item.tokens_label || "";
    tok.title = (item.chars_label || "") + (item.images ? " · " + item.images + " image(s)" : "");
    head.append(idx, kind, label, tok);
    row.appendChild(head);

    if (!expanded) {
      const preview = document.createElement("div");
      preview.className = "ctx-preview";
      preview.textContent = item.preview || "";
      row.appendChild(preview);
      return row;
    }
    const body = document.createElement("pre");
    body.className = "ctx-body";
    const full = view.full[item.index];
    body.textContent = item.withheld ? (item.preview || "") : (full ? full.text : item.content || "");
    row.appendChild(body);
    const more = full ? full.nextOffset != null : !!item.truncated;
    if (more) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ctx-more";
      btn.dataset.index = String(item.index);
      const shown = full ? full.text.length : (item.content || "").length;
      btn.textContent = (full && full.loading)
        ? "Loading…"
        : "Show more (" + shown.toLocaleString() + " of " + Number(item.chars || 0).toLocaleString() + " chars)";
      btn.disabled = !!(full && full.loading);
      row.appendChild(btn);
    }
    return row;
  }

  function measureExpanded() {
    const node = view.rows.get(view.expanded);
    if (!node) return;
    const h = Math.ceil(node.getBoundingClientRect().height);
    if (h && h !== view.expandedHeight) {
      view.expandedHeight = h;
      layout();
    }
  }

  function toggle(index) {
    const was = view.expanded;
    view.expanded = was === index ? -1 : index;
    view.expandedHeight = view.expanded >= 0 ? view.rowHeight : 0;
    layout();
    if (view.expanded >= 0) measureExpanded();
    const node = view.rows.get(index);
    const head = node && node.querySelector(".ctx-row-head");
    if (head) head.focus({ preventScroll: true });
  }

  async function loadMore(index) {
    const item = view.items[index];
    if (!item) return;
    let full = view.full[index];
    if (!full) {
      full = view.full[index] = { text: "", nextOffset: 0, loading: false };
    }
    if (full.loading || full.nextOffset == null) return;
    full.loading = true;
    rebuildRow(index);
    const gen = view.gen;
    const q = new URLSearchParams({ snapshot: view.snapshot, offset: String(full.nextOffset) });
    let res;
    try {
      res = await fetchJson("/api/sessions/" + encodeURIComponent(view.sid) +
        "/context/items/" + index + "?" + q.toString());
    } catch (e) {
      res = { status: 0, data: {} };
    }
    if (gen !== view.gen) return;
    full.loading = false;
    const d = res.data || {};
    if (res.status === 409 && d.code === "context_changed") {
      reload();
      return;
    }
    if (res.status === 200 && d.ok !== false) {
      full.text += d.content || "";
      full.nextOffset = d.next_offset == null ? null : d.next_offset;
    }
    rebuildRow(index);
  }

  function rebuildRow(index) {
    const node = view.rows.get(index);
    if (!node) return;
    node.remove();
    view.rows.delete(index);
    paintWindow();
    if (index === view.expanded) measureExpanded();
  }

  // ── compact (the shared compact widget's state machine) ───────────────

  function paintCompact() {
    const w = typeof compactWidgetState === "function" ? compactWidgetState() : { state: "idle" };
    compactBtn.hidden = !view.canCompact;
    compactBtn.disabled = w.state === "confirm" || w.state === "compacting";
    confirmNode.textContent = "";
    confirmNode.dataset.state = w.state;
    if (w.state === "confirm") {
      const text = document.createElement("span");
      text.className = "ctx-confirm-text";
      text.textContent = w.prompt || "Compaction cannot be undone. Continue?";
      const ok = document.createElement("button");
      ok.type = "button";
      ok.id = "ctx-compact-ok";
      ok.className = "ctx-confirm-ok";
      ok.textContent = "Compact";
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.id = "ctx-compact-cancel";
      cancel.textContent = "Cancel";
      confirmNode.append(text, ok, cancel);
      confirmNode.hidden = false;
      return;
    }
    if (w.state === "compacting") {
      confirmNode.textContent = "Compacting context…";
      confirmNode.hidden = false;
      return;
    }
    if (w.state === "error") {
      const text = document.createElement("span");
      text.className = "ctx-confirm-text";
      text.textContent = w.error || "Compaction failed";
      const dismiss = document.createElement("button");
      dismiss.type = "button";
      dismiss.id = "ctx-compact-dismiss";
      dismiss.textContent = "Dismiss";
      confirmNode.append(text, dismiss);
      confirmNode.hidden = false;
      return;
    }
    if (w.state === "done") {
      confirmNode.textContent = "Compaction complete";
      confirmNode.hidden = false;
      return;
    }
    confirmNode.hidden = true;
  }

  let lastCompactState = "idle";
  function compactChanged(w) {
    const next = (w && w.state) || "idle";
    const prev = lastCompactState;
    lastCompactState = next;
    if (!isOpen()) return;
    paintCompact();
    if (next === "done" && prev !== "done") reload();
    if (next === "confirm") {
      const ok = $("ctx-compact-ok");
      if (ok) ok.focus();
    }
  }

  // ── open / close ──────────────────────────────────────────────────────

  function closeOtherSheets() {
    const picker = $("agent-model-pop");
    if (picker && !picker.hidden && typeof closeModelSheet === "function") closeModelSheet();
    const editor = $("budget-editor");
    if (editor && !editor.hidden && typeof closeBudgetEditor === "function") closeBudgetEditor();
    if (window.SaAgentMon && window.SaAgentMon.isOpen && window.SaAgentMon.isOpen()) {
      window.SaAgentMon.close({ restoreFocus: false });
    }
  }

  function readRowHeight() {
    const v = parseFloat(getComputedStyle(sheet).getPropertyValue("--ctx-row-h"));
    view.rowHeight = v > 0 ? v : 56;
  }

  function open() {
    const sid = currentSid();
    if (!sid) return false;
    if (isOpen() && sid === view.sid) return true;
    closeOtherSheets();
    ignoreBackdropUntil = Date.now() + 400;
    sheet.hidden = false;
    if (backdrop) backdrop.hidden = false;
    readRowHeight();
    view.sid = sid;
    totalNode.textContent = "";
    meterNode.textContent = "";
    noticeNode.hidden = true;
    lastCompactState = typeof compactWidgetState === "function" ? compactWidgetState().state : "idle";
    paintCompact();
    if (!observer && typeof IntersectionObserver === "function") {
      observer = new IntersectionObserver(function (entries) {
        if (entries.some((e) => e.isIntersecting)) loadPage(false);
      }, { root: scroller, rootMargin: "0px 0px " + PREFETCH_PX + "px 0px" });
      observer.observe(sentinel);
    }
    reload();
    window.setTimeout(function () { if (isOpen()) closeBtn.focus(); }, 0);
    return true;
  }

  function close(opts) {
    if (!isOpen()) return;
    // A pending confirmation belongs to this sheet's Compact button.
    if (typeof compactWidgetState === "function" && compactWidgetState().state === "confirm"
        && typeof compactContextCancel === "function") {
      compactContextCancel();
    }
    sheet.hidden = true;
    if (backdrop) backdrop.hidden = true;
    view.gen += 1;
    view.rows.forEach((node) => node.remove());
    view.rows.clear();
    view.items = [];
    view.full = {};
    if (!opts || opts.restoreFocus !== false) {
      const meter = $("compact-context-btn");
      if (meter) meter.focus();
    }
  }

  // ── events ────────────────────────────────────────────────────────────

  scroller.addEventListener("scroll", schedulePaint, { passive: true });
  window.addEventListener("resize", function () { if (isOpen()) { readRowHeight(); layout(); } });
  closeBtn.addEventListener("click", function () { close(); });
  if (backdrop) {
    backdrop.addEventListener("click", function () {
      if (Date.now() < ignoreBackdropUntil) return;
      close();
    });
  }
  compactBtn.addEventListener("click", function () {
    if (typeof compactContextStart === "function") compactContextStart();
  });
  confirmNode.addEventListener("click", function (e) {
    const t = e.target;
    if (!t || !t.closest) return;
    if (t.closest("#ctx-compact-ok") && typeof compactContextConfirm === "function") compactContextConfirm();
    else if (t.closest("#ctx-compact-cancel") && typeof compactContextCancel === "function") compactContextCancel();
    else if (t.closest("#ctx-compact-dismiss") && typeof compactContextDismiss === "function") compactContextDismiss();
  });
  list.addEventListener("click", function (e) {
    const t = e.target;
    if (!t || !t.closest) return;
    const more = t.closest(".ctx-more");
    if (more) {
      loadMore(Number(more.dataset.index));
      return;
    }
    const head = t.closest(".ctx-row-head");
    if (head) {
      const row = head.closest(".ctx-row");
      if (row) toggle(Number(row.dataset.index));
    }
  });
  // Capture phase so the sheet handles Escape before the chat's
  // stream-cancel shortcut, and Tab stays inside the dialog.
  window.addEventListener("keydown", function (e) {
    if (!isOpen()) return;
    if (e.key === "Escape") {
      e.preventDefault();
      e.stopImmediatePropagation();
      close();
      return;
    }
    if (e.key !== "Tab") return;
    const focusable = [...sheet.querySelectorAll("button:not([disabled]):not([hidden])")]
      .filter((b) => b.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (!sheet.contains(document.activeElement)) {
      e.preventDefault(); (e.shiftKey ? last : first).focus();
    } else if (e.shiftKey && document.activeElement === first) {
      e.preventDefault(); last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault(); first.focus();
    }
  }, true);

  window.SaContextSheet = {
    open: open,
    close: close,
    isOpen: isOpen,
    reload: function () { return isOpen() ? reload() : null; },
    compactChanged: compactChanged,
    // Test/diagnostic surface: how much is loaded vs. in the DOM.
    stats: function () {
      return {
        loaded: view.items.length,
        rendered: view.rows.size,
        nodes: sheet.getElementsByTagName("*").length,
        nextCursor: view.nextCursor,
        expanded: view.expanded,
      };
    },
  };
  window.openContextSheet = open;
  window.closeContextSheet = close;
})();
