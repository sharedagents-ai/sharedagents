/* virtual_scroll.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
const SCROLL_WINDOW_HALF = 20; // messages rendered above/below viewport center
const SCROLL_WINDOW_MAX = 60; // hard cap on rendered messages
const INITIAL_RENDER = 30; // messages rendered on load (latest N); rest lazy
const OLDER_BATCH = 20; // messages prepended per scroll-up
const PRUNE_DELAY_MS = 1500; // debounce for pruning off-screen messages
const ESTIMATED_MSG_HEIGHT = 80; // px, fallback when DOM not yet rendered
// ══════════════════════════════════════════════════════════════════════

const VirtualScroller = {
  // ── Lifecycle ─────────────────────────────────────────────────────

  /** Initialise sentinel element + observer. Call once on page load. */
  init() {
    const mc = messagesContainer;
    if (!mc) return;

    // Remove any stale virtual scroller elements first.
    const oldSentinel = document.getElementById("vs-sentinel-top");
    if (oldSentinel) oldSentinel.remove();
    const oldBm = document.getElementById("vs-bottom-marker");
    if (oldBm) oldBm.remove();
    const oldIndicator = document.getElementById("vs-new-msg-indicator");
    if (oldIndicator) oldIndicator.remove();
    const oldEarlier = document.getElementById("vs-earlier-btn");
    if (oldEarlier) oldEarlier.remove();

    // Create sentinel (inserted as first child of messagesContainer).
    const sentinel = document.createElement("div");
    sentinel.id = "vs-sentinel-top";
    sentinel.style.cssText = "height:0;overflow:hidden;pointer-events:none;";
    mc.insertBefore(sentinel, mc.firstChild);

    // Bottom marker — invisible element after the last message.
    const bm = document.createElement("div");
    bm.id = "vs-bottom-marker";
    bm.style.cssText = "height:1px;overflow:hidden;";
    mc.appendChild(bm);

    // "Scroll to bottom" / "New messages" floating button — created once, shown/hidden.
    let indicator = document.getElementById("vs-new-msg-indicator");
    if (!indicator) {
      indicator = document.createElement("div");
      indicator.id = "vs-new-msg-indicator";
      indicator.innerHTML =
        '<button id="vs-new-msg-btn" class="new-msg-btn">↓ Scroll to bottom</button>';
      mc.appendChild(indicator);
      indicator
        .querySelector("#vs-new-msg-btn")
        .addEventListener("click", () => {
          state.scroll.autoScroll = true;
          this._hasNewMessages = false;
          this._forcePin();
          this._hideNewMsgBtn();
        });
    }

    state.scroll.sentinelTop = sentinel;
    state.scroll.bottomMarker = bm;
    state.scroll.autoScroll = true;

    // Observe both sentinel and bottom marker.
    if (state.scroll.observer) state.scroll.observer.disconnect();
    state.scroll.observer = new IntersectionObserver(
      (entries) => this._onIntersect(entries),
      { root: mc, rootMargin: "200px 0px 200px 0px", threshold: 0 },
    );
    state.scroll.observer.observe(sentinel);
    state.scroll.observer.observe(bm);

    // Also observe scroll events for fine-grained auto-scroll detection.
    // Reuse ONE bound handler so repeated init() (every session switch) doesn't
    // accumulate listeners — remove-then-add is a no-op the first time.
    if (!this._boundScroll) this._boundScroll = () => this._onScroll();
    mc.removeEventListener("scroll", this._boundScroll, { passive: true });
    mc.addEventListener("scroll", this._boundScroll, { passive: true });
    if (!this._boundWheel) {
      this._boundWheel = (e) => {
        if (e.deltaY >= 0) return;
        state.scroll.olderUnlocked = true;
        // Debounce the wheel burst: one older page per settled gesture,
        // not a GET on every tick (that re-fetched the last-N page).
        this._syncEarlierBtn();
        this._scheduleOlderFromGesture();
      };
    }
    if (!this._boundTouchStart) {
      this._boundTouchStart = (e) => {
        this._touchY = e.touches && e.touches[0] ? e.touches[0].clientY : 0;
      };
      this._boundTouchMove = (e) => {
        const y = e.touches && e.touches[0] ? e.touches[0].clientY : 0;
        if (y - (this._touchY || 0) < 24) return;
        this._touchY = y;
        state.scroll.olderUnlocked = true;
        this._syncEarlierBtn();
        this._scheduleOlderFromGesture();
      };
    }
    mc.removeEventListener("wheel", this._boundWheel, { passive: true });
    mc.addEventListener("wheel", this._boundWheel, { passive: true });
    mc.removeEventListener("touchstart", this._boundTouchStart, { passive: true });
    mc.addEventListener("touchstart", this._boundTouchStart, { passive: true });
    mc.removeEventListener("touchmove", this._boundTouchMove, { passive: true });
    mc.addEventListener("touchmove", this._boundTouchMove, { passive: true });
  },

  /** Full reset (on session switch / clearMessages). */
  reset() {
    state.msgCache = [];
    state.scroll.windowStart = 0;
    state.scroll.windowEnd = 0;
    state.scroll.renderedStart = 0;
    state.scroll.totalHeight = 0;
    state.scroll.autoScroll = true;
    state.scroll.rendering = false;
    this._hasNewMessages = false;
    this._hideNewMsgBtn();
    // Cancel any pending bulk-load pin from a prior session so it can't fire
    // against the new one (session switch mid-load). The in-flight rAF pin is
    // additionally epoch-guarded in finishBulkLoad().
    state.scroll.bulkLoading = false;
    this._teardownBulk();
    this._olderNetworkLock = false;
    if (this._olderGestureTimer) {
      clearTimeout(this._olderGestureTimer);
      this._olderGestureTimer = null;
    }
    if (this._olderLockReleaseTimer) {
      clearTimeout(this._olderLockReleaseTimer);
      this._olderLockReleaseTimer = null;
    }

    // Keep sentinel + bottom marker, remove everything between them.
    const mc = messagesContainer;
    const sentinel = state.scroll.sentinelTop;
    const bm = state.scroll.bottomMarker;
    if (mc && sentinel && bm) {
      // Remove every child between sentinel and bottom marker.
      while (
        sentinel.nextElementSibling &&
        sentinel.nextElementSibling !== bm
      ) {
        sentinel.nextElementSibling.remove();
      }
      sentinel.style.height = "0";
    }
  },

  // ── Cache management ─────────────────────────────────────────────

  /** Register a new message in the cache. Returns its index. */
  add(role, content, el, images, segments, agent, artifacts, timestamp, aborted, status, error, endedAt, author) {
    const idx = state.msgCache.length;
    state.msgCache.push({
      id: `msg-${idx}-${Date.now()}`,
      role,
      content,
      // Author of a user bubble ({actors, source}); null for other roles.
      // Cached so a pruned-then-rebuilt bubble keeps its name.
      author: role === "user" && author && typeof author === "object"
        ? { actors: Array.isArray(author.actors) ? author.actors.slice() : [],
            source: String(author.source || "") }
        : null,
      // Server-stamped turn timestamp (ISO-8601 UTC). Cached so a
      // pruned-then-rebuilt bubble keeps its time slot. Empty when the
      // source message carried none (the header then renders no time).
      timestamp: timestamp ? String(timestamp) : "",
      aborted: !!aborted || status === "aborted",
      status: status ? String(status) : "",
      error: error != null && String(error) ? String(error) : "",
      endedAt: endedAt ? String(endedAt) : "",
      lastTs: endedAt ? String(endedAt) : (timestamp ? String(timestamp) : ""),
      // Attached images (user prompts) — data/URL strings rendered below the
      // text. Cached so a pruned-then-rebuilt bubble still shows the picture.
      images: Array.isArray(images) && images.length ? images : null,
      // Lossless completed-turn blocks supplied by history replay. Cached so
      // virtual-scroll pruning cannot degrade tool/reasoning cards to prose.
      segments:
        Array.isArray(segments) && segments.length ? segments : null,
      // Per-turn agent that authored this bubble. Empty falls back to the active
      // picker agent at render time.
      agent: agent ? String(agent) : "",
      // Structured agent-produced outputs. Only public display metadata and a
      // SHA-256 reach the browser; local paths remain gateway-side.
      artifacts:
        Array.isArray(artifacts) && artifacts.length ? artifacts.slice() : null,
      // Native disclosure state for the lazy reasoning/tool Activity card.
      // It lives with the cached message so pruning and re-materialisation do
      // not collapse rows the operator had opened.
      activityDisclosureState: null,
      el: el || null,
      height: 0,
    });
    return idx;
  },

  /** Update cached content (used by setAssistantContent for streaming). */
  update(idx, content) {
    if (idx >= 0 && idx < state.msgCache.length) {
      state.msgCache[idx].content = content;
    }
  },

  /** Update cached el reference for a message. */
  setEl(idx, el) {
    if (idx >= 0 && idx < state.msgCache.length) {
      state.msgCache[idx].el = el;
    }
  },

  /** Measure a message element's height and cache it. */
  measure(idx, el) {
    if (idx >= 0 && idx < state.msgCache.length && el) {
      const h = el.hidden ? 0 : el.offsetHeight || ESTIMATED_MSG_HEIGHT;
      state.msgCache[idx].height = h;
    }
  },

  /** Get total estimated height for all messages in a range. */
  _estimateHeight(fromIdx, toIdx) {
    let h = 0;
    for (let i = fromIdx; i < toIdx; i++) {
      const m = state.msgCache[i];
      h += (m && m.height) ?? ESTIMATED_MSG_HEIGHT;
    }
    return h;
  },

  // ── Rendering ────────────────────────────────────────────────────

  /** Attach a new message element to the DOM (via the scroller). */
  attach(el, idx) {
    const mc = messagesContainer;
    if (!mc || !el) return;
    this.setEl(idx, el);

    if (state.scroll.autoScroll) {
      // User is at bottom — insert before bottomMarker so it's visible.
      const bm = state.scroll.bottomMarker;
      if (bm && bm.parentNode === mc) {
        mc.insertBefore(el, bm);
      } else {
        mc.appendChild(el);
      }
      // Immediately measure the height (needs layout)
      requestAnimationFrame(() => this.measure(idx, el));
      // During a bulk load we do NOT scroll per event — that races async height
      // measurement + pruning and lands the view at a non-deterministic spot.
      // One force-pin runs in finishBulkLoad() once the stream settles.
      if (!state.scroll.bulkLoading)
        requestAnimationFrame(() => scrollToBottom());
    } else {
      // User scrolled up — show "new messages" indicator.
      this._showNewMsgBtn(true);
      // Insert before bottom marker so the marker stays at the end.
      const bm = state.scroll.bottomMarker;
      if (bm && bm.parentNode === mc) {
        mc.insertBefore(el, bm);
      } else {
        mc.appendChild(el);
      }
      requestAnimationFrame(() => this.measure(idx, el));
    }

    // Keep the DOM light: detach the oldest rendered messages once we exceed
    // the window — but ONLY ones fully scrolled off the top (above the
    // viewport), so we never remove what the user is looking at. This is
    // safe regardless of autoScroll and replaces the flawed height-estimate
    // pruning that blanked long logs. Detached messages stay in the cache and
    // re-materialise on scroll-up (loadOlder).
    this._trimTop();
  },

  /** Detach oldest rendered messages that are fully above the viewport. */
  _trimTop() {
    const cache = state.msgCache;
    const mc = messagesContainer;
    if (!mc) return;
    let start = state.scroll.renderedStart || 0;
    while (cache.length - start > SCROLL_WINDOW_MAX) {
      const m = cache[start];
      if (!m || !m.el || !m.el.parentNode) {
        start++;
        continue;
      }
      // Stop as soon as a message is not fully above the viewport — keeping
      // everything from here down preserves what the user can see.
      const elBottom = m.el.offsetTop + m.el.offsetHeight;
      if (elBottom >= mc.scrollTop) break;
      this.measure(start, m.el); // capture REAL height for the sentinel
      m.el.remove();
      m.el = null;
      start++;
    }
    if (start !== (state.scroll.renderedStart || 0)) {
      state.scroll.renderedStart = start;
      this._updateTopSentinel();
    }
  },

  /** Recompute visible window based on scroll position and re-render. */
  renderWindow() {
    if (state.scroll.rendering) return;
    state.scroll.rendering = true;

    const mc = messagesContainer;
    const cache = state.msgCache;
    if (!mc || !cache.length) {
      state.scroll.rendering = false;
      return;
    }

    const scrollTop = mc.scrollTop;
    const clientH = mc.clientHeight;
    const scrollBottom = scrollTop + clientH;

    // Find which cache index corresponds to the top of the viewport.
    let leadingH = 0;
    let topIdx = 0;
    for (let i = 0; i < cache.length; i++) {
      const h = cache[i].height ?? ESTIMATED_MSG_HEIGHT;
      if (leadingH + h > scrollTop) {
        topIdx = i;
        break;
      }
      leadingH += h;
    }

    // Window: SCROLL_WINDOW_HALF before topIdx, SCROLL_WINDOW_HALF after.
    const newStart = Math.max(0, topIdx - SCROLL_WINDOW_HALF);
    const newEnd = Math.min(cache.length, topIdx + SCROLL_WINDOW_HALF * 2);

    // Only re-render if the window changed significantly (avoid thrash).
    const oldStart = state.scroll.windowStart;
    const oldEnd = state.scroll.windowEnd;
    if (newStart === oldStart && newEnd === oldEnd) {
      state.scroll.rendering = false;
      return;
    }

    // --- Phase 1: detach elements leaving the window (prune above/below) ---
    // Elements above newStart.
    for (let i = oldStart; i < newStart && i < oldEnd; i++) {
      const m = cache[i];
      if (m && m.el && m.el.parentNode) {
        this.measure(i, m.el);
        // Keep the one immediately above newStart for visual continuity
        if (i < newStart - 1) {
          m.el.remove();
        }
      }
    }
    // Elements below newEnd.
    for (let i = Math.max(oldStart, newEnd); i < oldEnd; i++) {
      const m = cache[i];
      if (m && m.el && m.el.parentNode) {
        this.measure(i, m.el);
        m.el.remove();
      }
    }

    // --- Phase 2: attach elements entering the window ---
    // We need to insert them AFTER the sentinel but BEFORE the bottom marker
    // (or before the first existing element from the old window).
    const sentinel = state.scroll.sentinelTop;
    const bm = state.scroll.bottomMarker;

    // Find the reference node: first DOM element >= newStart that's still attached.
    let refNode = bm;
    for (let i = newEnd - 1; i >= newStart; i--) {
      const m = cache[i];
      if (m && m.el && m.el.parentNode) {
        refNode = m.el;
        break;
      }
    }

    // Attach elements in ascending order before refNode.
    for (let i = newStart; i < newEnd; i++) {
      const m = cache[i];
      if (!m) continue;
      if (!m.el || !m.el.parentNode) {
        // Need to create/re-create this element's DOM.
        // But we don't create DOM here — appendMessage does that.
        // If the element is missing, we signal that it needs rebuilding.
        continue;
      }
    }

    // If the first window elements don't exist yet, signal they need creation
    // by calling into the message rendering (handled by appendMessage itself).

    // --- Phase 3: update sentinel height to reflect pruned-off top content ---
    const prunedHeight = this._estimateHeight(0, newStart);
    if (sentinel) {
      sentinel.style.height = `${prunedHeight}px`;
    }

    state.scroll.windowStart = newStart;
    state.scroll.windowEnd = newEnd;
    state.scroll.rendering = false;

    // If the bottom marker is visible, keep autoScroll on.
    this._checkAutoScroll();
  },

  /** Detect if the user is at the bottom. */
  _checkAutoScroll() {
    const mc = messagesContainer;
    if (!mc) return;
    // While bulk-loading, the latch is frozen — growing content would otherwise
    // flip autoScroll false against an intermediate non-bottom position, which is
    // exactly what left long sessions parked near the top.
    if (state.scroll.bulkLoading) return;
    const threshold = 80; // px from bottom counts as "at bottom"
    const atBottom =
      mc.scrollHeight - mc.scrollTop - mc.clientHeight < threshold;
    state.scroll.autoScroll = atBottom;
    if (atBottom) {
      this._hideNewMsgBtn();
    } else {
      // User is not at the bottom and there is scrollable content
      if (mc.scrollHeight > mc.clientHeight + threshold) {
        this._showNewMsgBtn(false);
      }
    }
  },

  // ── Bulk-load lifecycle (deterministic scroll pin) ──────────────────
  // History now renders synchronously from the REST page (renderMessages), but
  // its markdown/image heights settle ASYNC afterward — a fixed-delay pin lands
  // a few px short. So we keep the view pinned to the bottom across a ResizeObserver
  // settle window (re-pin on every content-size change), then finish once the
  // layout is quiet. A user wheel/touch cancels the pin (they want to read up).

  /** Hard pin to the bottom, bypassing the autoScroll/bulk guards in scrollToBottom.
   * MUST be instant: the container has `scroll-behavior: smooth` (style.css), so a
   * plain `scrollTop =` animates over ~1-2s — that was the "strange fast scroll".
   * `behavior:'instant'` overrides the CSS and jumps straight to the bottom. */
  _forcePin() {
    const mc = messagesContainer;
    if (!mc) return;
    mc.scrollTo({ top: mc.scrollHeight, behavior: "instant" });
  },
  /** Enter bulk-load mode: freeze the autoScroll latch, pin + watch for settle. */
  beginBulkLoad() {
    state.scroll.bulkLoading = true;
    state.scroll.autoScroll = true; // we intend to land at the bottom
    // Token so a late pin can't fire against a session the user switched to in
    // the meantime (Codex #2). navEpoch is bumped by every loadSession.
    this._bulkEpoch = state.navEpoch;
    this._bulkUserInterrupt = false;
    if (!this._boundBulkInterrupt)
      this._boundBulkInterrupt = () => this._cancelBulkPin();
    const mc = messagesContainer;
    if (mc) {
      mc.addEventListener("wheel", this._boundBulkInterrupt, { passive: true });
      mc.addEventListener("touchstart", this._boundBulkInterrupt, {
        passive: true,
      });
    }
    // rAF settle-loop: pin every frame, finish once scrollHeight is stable for
    // BULK_STABLE_FRAMES frames (catches async markdown/image height growth that
    // a ResizeObserver on the fixed-size container would miss).
    this._bulkStable = 0;
    this._bulkLastH = -1;
    this._bulkDeadline = performance.now() + BULK_SETTLE_MAX_MS;
    this._bulkTick();
  },
  _bulkTick() {
    if (!state.scroll.bulkLoading) return;
    this._forcePin();
    const mc = messagesContainer;
    const h = mc ? mc.scrollHeight : 0;
    if (h === this._bulkLastH) this._bulkStable += 1;
    else {
      this._bulkStable = 0;
      this._bulkLastH = h;
    }
    if (
      this._bulkStable >= BULK_STABLE_FRAMES ||
      performance.now() >= this._bulkDeadline
    ) {
      this.finishBulkLoad();
      return;
    }
    this._bulkRAF = requestAnimationFrame(() => this._bulkTick());
  },
  /** A live event arrived during bulk (in-flight turn tail) — keep pinning. */
  _noteBulkActivity() {
    if (!state.scroll.bulkLoading) return;
    // Live turns must follow with scrollToBottom, not the history settle
    // loop. Extending the deadline on every log frame froze autoscroll
    // for the whole turn.
    if (state.isStreaming) {
      this.finishBulkLoad();
      return;
    }
    this._bulkStable = 0; // content changed → not settled yet
    this._bulkDeadline = performance.now() + BULK_SETTLE_MAX_MS;
  },
  /** Stop the settle loop + remove interrupt listeners + restore smooth scrolling. */
  _teardownBulk() {
    if (this._bulkRAF) {
      cancelAnimationFrame(this._bulkRAF);
      this._bulkRAF = null;
    }
    const mc = messagesContainer;
    if (mc) {
      mc.style.scrollBehavior = ""; // restore CSS smooth for live following
      if (this._boundBulkInterrupt) {
        mc.removeEventListener("wheel", this._boundBulkInterrupt, {
          passive: true,
        });
        mc.removeEventListener("touchstart", this._boundBulkInterrupt, {
          passive: true,
        });
      }
    }
  },
  _refreshWidgetsAfterBulk() {
    const epoch = this._bulkEpoch;
    if (typeof refreshSessionWidgets !== "function") return;
    if (this._widgetsRefreshedEpoch === epoch) return;
    this._widgetsRefreshedEpoch = epoch;
    setTimeout(() => {
      if (epoch !== state.navEpoch) return;
      refreshSessionWidgets();
    }, 0);
  },
  /** User grabbed the view during load — exit bulk mode WITHOUT the force-pin. */
  _cancelBulkPin() {
    if (!state.scroll.bulkLoading) return;
    this._bulkUserInterrupt = true;
    state.scroll.bulkLoading = false;
    this._teardownBulk();
    this._checkAutoScroll(); // reflect where the user actually is
    this._refreshWidgetsAfterBulk();
  },
  /** Exit bulk-load mode, pin once more after layout settled. Idempotent. */
  finishBulkLoad() {
    if (!state.scroll.bulkLoading) {
      this._teardownBulk();
      this._refreshWidgetsAfterBulk();
      this.ensureReachableHistory();
      return;
    }
    if (this._bulkUserInterrupt) {
      state.scroll.bulkLoading = false;
      this._teardownBulk();
      this._checkAutoScroll();
      this._refreshWidgetsAfterBulk();
      this.ensureReachableHistory();
      return;
    }
    const epoch = this._bulkEpoch;
    this._teardownBulk();
    state.scroll.bulkLoading = false;
    state.scroll.autoScroll = true;
    // Final pin now, plus a couple of guarded late re-pins to absorb LATE layout
    // shifts (image decode, web-font swap, syntax highlight) that land after the
    // settle loop exits (Codex #4). Each bails if the session changed or the user
    // has since scrolled up (autoScroll flipped false by the scroll handler).
    const latePin = () => {
      if (epoch !== state.navEpoch) return;
      if (state.scroll.bulkLoading || this._bulkUserInterrupt) return;
      if (!state.scroll.autoScroll) return; // user scrolled away → respect it
      this._forcePin();
      this.ensureReachableHistory();
    };
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        latePin();
        this._hideNewMsgBtn();
      }),
    );
    setTimeout(latePin, 250);
    setTimeout(latePin, 600);
    this._refreshWidgetsAfterBulk();
    // Observer already saw the top sentinel during bulk load and will not
    // re-fire if the tail still fits. Pull a few older pages so a refresh
    // of a short last-N window gets a scrollbar (or a clickable control).
    this.ensureReachableHistory();
  },

  _viewFits() {
    const mc = messagesContainer;
    return !!(mc && mc.scrollHeight <= mc.clientHeight + 8);
  },

  _historyReachable() {
    const h = state.history;
    if (h && h.beforeCursor != null) return true;
    return (state.scroll.renderedStart || 0) > 0;
  },

  /** After open / each older page: show the Earlier control when more
   * history exists. Boot does not auto-page. After the user unlocks
   * (wheel / touch / scrollbar / button), infinite scroll loads one
   * older page per settled gesture. */
  ensureReachableHistory() {
    if (state.scroll.bulkLoading) return;
    if (state.history && state.history.loading) return;
    state.scroll.fillViewport = false;
    this._syncEarlierBtn();
  },

  /** Collapse a wheel/touch burst into one older-page request. */
  _scheduleOlderFromGesture() {
    if (this._olderGestureTimer) clearTimeout(this._olderGestureTimer);
    this._olderGestureTimer = setTimeout(() => {
      this._olderGestureTimer = null;
      this.requestOlder();
    }, 120);
  },

  /** Cache rematerialise always; network older page at most once while
   * the lock is held. Scrolling away from the top, a unique prepend, or
   * the Earlier button re-arms. Overlapping last-N pages keep the lock
   * so the same tail is not GET-looped. */
  requestOlder() {
    if (state.scroll.bulkLoading) return;
    if ((state.scroll.renderedStart || 0) > 0) {
      this.loadOlder();
      return;
    }
    const h = state.history;
    if (!h || h.beforeCursor == null) {
      this._syncEarlierBtn();
      return;
    }
    if (!state.scroll.olderUnlocked) return;
    if (h.loading) return;
    if (this._olderNetworkLock) return;
    this._olderNetworkLock = true;
    this.loadOlder();
  },

  onOlderPageDone(prepended) {
    if (prepended) {
      if (this._olderLockReleaseTimer) clearTimeout(this._olderLockReleaseTimer);
      this._olderLockReleaseTimer = setTimeout(() => {
        this._olderLockReleaseTimer = null;
        this._olderNetworkLock = false;
        this._syncEarlierBtn();
      }, 400);
    }
    this.ensureReachableHistory();
  },

  _atLoadedTop() {
    const mc = messagesContainer;
    return !!mc && mc.scrollTop <= 8 && (state.scroll.renderedStart || 0) <= 0;
  },

  _syncEarlierBtn() {
    const mc = messagesContainer;
    if (!mc) return;
    let btn = document.getElementById("vs-earlier-btn");
    const show = !state.scroll.bulkLoading
      && this._historyReachable()
      && (this._viewFits() || this._atLoadedTop());
    if (!show) {
      if (btn) btn.hidden = true;
      return;
    }
    if (!btn) {
      btn = document.createElement("button");
      btn.id = "vs-earlier-btn";
      btn.type = "button";
      btn.className = "earlier-msg-btn";
      btn.textContent = "↑ Earlier messages";
      btn.addEventListener("click", () => {
        state.scroll.olderUnlocked = true;
        this._olderNetworkLock = false;
        if (this._olderGestureTimer) {
          clearTimeout(this._olderGestureTimer);
          this._olderGestureTimer = null;
        }
        this.requestOlder();
      });
      const sentinel = state.scroll.sentinelTop;
      if (sentinel && sentinel.parentNode === mc)
        mc.insertBefore(btn, sentinel.nextSibling);
      else mc.insertBefore(btn, mc.firstChild);
    }
    btn.hidden = false;
  },

  // ── Pruning ───────────────────────────────────────────────────────

  _pruneTimer: null,

  _schedulePrune() {
    if (this._pruneTimer) clearTimeout(this._pruneTimer);
    this._pruneTimer = setTimeout(() => this._prune(), PRUNE_DELAY_MS);
  },

  /** Aggressively prune DOM elements far outside the viewport. */
  _prune() {
    const mc = messagesContainer;
    const cache = state.msgCache;
    if (!mc || !cache.length) return;

    const scrollTop = mc.scrollTop;
    const clientH = mc.clientHeight;
    const scrollBottom = scrollTop + clientH;
    // Keep a generous buffer: render everything within 3x viewport height.
    const keepPx = clientH * 3;

    let accH = 0;
    let keepStart = 0;
    let keepEnd = cache.length;

    // Find keepStart: first message whose bottom is within keepPx of scrollTop.
    for (let i = 0; i < cache.length; i++) {
      const m = cache[i];
      const h = (m && m.height) ?? ESTIMATED_MSG_HEIGHT;
      if (accH + h > scrollTop - keepPx) {
        keepStart = i;
        break;
      }
      accH += h;
    }

    // Find keepEnd: last message whose top is within keepPx of scrollBottom.
    accH = 0;
    for (let i = 0; i < cache.length; i++) {
      const m = cache[i];
      const h = (m && m.height) ?? ESTIMATED_MSG_HEIGHT;
      if (accH > scrollBottom + keepPx) {
        keepEnd = i;
        break;
      }
      accH += h;
    }

    // Prune elements below keepEnd.
    for (let i = keepEnd; i < cache.length; i++) {
      const m = cache[i];
      if (m && m.el && m.el.parentNode) {
        this.measure(i, m.el);
        m.el.remove();
      }
    }

    // Prune elements above keepStart (but keep the first rendered one for continuity).
    for (let i = 0; i < keepStart; i++) {
      const m = cache[i];
      if (m && m.el && m.el.parentNode) {
        this.measure(i, m.el);
        if (i < keepStart - 1) {
          m.el.remove();
        }
      }
    }

    // Recompute total pruned height for the sentinel.
    // Use keepStart (prune boundary) so the sentinel reflects what's scrolled off.
    const prunedTop = this._estimateHeight(0, keepStart);
    const sentinel = state.scroll.sentinelTop;
    if (sentinel) sentinel.style.height = `${prunedTop}px`;

    // Update the window state to reflect pruning.
    state.scroll.windowStart = Math.max(state.scroll.windowStart, keepStart);
    state.scroll.windowEnd = Math.min(state.scroll.windowEnd, keepEnd);
  },

  // ── New-messages / Scroll-to-bottom indicator ───────────────────────

  _showNewMsgBtn(isNewMsg) {
    if (isNewMsg) {
      this._hasNewMessages = true;
    }
    const btn = document.getElementById("vs-new-msg-btn");
    if (!btn) return;
    if (this._hasNewMessages) {
      btn.textContent = "↓ New messages";
      btn.classList.add("has-new");
    } else {
      btn.textContent = "↓ Scroll to bottom";
      btn.classList.remove("has-new");
    }
    btn.classList.add("visible");
  },

  _hideNewMsgBtn() {
    const btn = document.getElementById("vs-new-msg-btn");
    if (btn) {
      btn.classList.remove("visible");
      btn.classList.remove("has-new");
    }
    this._hasNewMessages = false;
  },

  // ── Event handlers ────────────────────────────────────────────────

  _onIntersect(entries) {
    // Frozen during bulk load: neither flip autoScroll nor trigger loadOlder
    // while the initial history is still streaming in.
    if (state.scroll.bulkLoading) return;
    for (const entry of entries) {
      if (entry.target.id === "vs-sentinel-top" && entry.isIntersecting) {
        if (state.history && state.history.loading) continue;
        // Pruned cached rows: rematerialise. At the loaded top, one
        // unlocked gesture loads the next older page (not every
        // intersection tick — requestOlder holds a lock).
        if ((state.scroll.renderedStart || 0) > 0) {
          this.loadOlder();
          continue;
        }
        if (state.scroll.olderUnlocked) this.requestOlder();
        else this._syncEarlierBtn();
      }
      if (entry.target.id === "vs-bottom-marker") {
        // User reached the bottom — enable auto-scroll, hide indicator.
        if (entry.isIntersecting) {
          state.scroll.autoScroll = true;
          this._hideNewMsgBtn();
        } else {
          state.scroll.autoScroll = false;
          if (!state.scroll.bulkLoading) {
            this._checkAutoScroll();
          }
        }
      }
    }
  },

  _onScroll() {
    const mc = messagesContainer;
    if (mc && mc.scrollTop > 40) {
      this._olderNetworkLock = false;
    }
    // Scrollbar to the top after leaving the bottom: same unlock as wheel.
    // Boot pin keeps autoScroll true on a fitting tail, so we do not
    // auto-page history on open.
    if (
      mc
      && mc.scrollTop <= 8
      && !state.scroll.autoScroll
      && (state.scroll.renderedStart || 0) <= 0
    ) {
      state.scroll.olderUnlocked = true;
      this._scheduleOlderFromGesture();
    }
    // Throttle: check autoScroll and update scroll-to-bottom indicator periodically.
    if (!this._scrollTimer) {
      this._scrollTimer = setTimeout(() => {
        this._scrollTimer = null;
        this._checkAutoScroll();
      }, 50);
    }
  },

  /** Rebuild a message's DOM element that was pruned. Returns the new el. */
  rebuild(idx) {
    const m = state.msgCache[idx];
    if (!m) return null;
    // We need to call appendMessage to rebuild, but that will add to cache again.
    // Instead, clone the original appendMessage behavior for cached data.
    const el = _createMessageElement(
      m.role, m.content, m.images, m.agent, m.artifacts, m.timestamp,
      m.aborted, m.status, m.error, m.endedAt, m.author,
    );
    if (el) {
      this.attach(el, idx);
      if (m.role === "assistant" && m.content) {
        el.dataset.raw = m.content;
        el.classList.toggle("has-content", !!(m.content && m.content.trim()));
      }
      if (m.role === "assistant" && m.liveBs &&
          (m.liveBs.isCompactionOnly() || m.liveBs.blocks.some((blk) => blk.delegate))) {
        // Retain hidden control turns and detached children across DOM rebuilds.
        renderBlockTurn({ el, bs: m.liveBs }, false);
      } else if (m.role === "assistant" && m.segments) {
        renderBlockTurn(
          {
            el,
            bs: blockStreamFromSavedSegments(
              m.segments,
              m.content,
              state.vis,
              _replayOptsFromMessage(m),
            ),
          },
          false,
        );
      } else if (
        m.role === "assistant"
        && m.status === "running"
        && (m.content || "").trim()
        && typeof blockStreamFromSavedSegments === "function"
      ) {
        renderBlockTurn(
          {
            el,
            bs: blockStreamFromSavedSegments(
              [{ kind: "assistant", thinking: "", content: m.content }],
              m.content,
              state.vis,
              _replayOptsFromMessage(m),
            ),
          },
          false,
        );
      }
      // Re-attach the compaction-stats card (sessions.js) so an off-screen
      // rebuild of a compaction turn keeps its card — and its per-turn stats
      // snapshot when one was captured live (history reloads fall back to the
      // header values inside the mount helper).
      if (typeof mountCompactionStatsCard === "function") {
        mountCompactionStatsCard(el, m.compactionStats || null);
      }
    }
    if (m.role === "ask_user") _remountAskUserFromCache(m, el);
    return el;
  },

  // ── Tail rendering + lazy older-message loading ─────────────────────
  // Render only the most recent INITIAL_RENDER messages; build one DOM node
  // per shown message through the SAME _createMessageElement → renderAssistant
  // Content path, so reasoning disclosures + tool rows render exactly as
  // before. Older messages are materialised from the cache on scroll-up.

  _materialise(i, beforeNode) {
    const m = state.msgCache[i];
    if (!m) return null;
    const el = _createMessageElement(
      m.role, m.content, m.images, m.agent, m.artifacts, m.timestamp,
      m.aborted, m.status, m.error, m.endedAt, m.author,
    );
    if (!el) return null;
    m.el = el;
    if (m.role === "assistant") {
      el.dataset.raw = m.content || "";
      el.classList.toggle("has-content", !!(m.content && m.content.trim()));
      if (m.liveBs && (m.liveBs.isCompactionOnly() || m.liveBs.blocks.some((blk) => blk.delegate))) {
        renderBlockTurn({ el, bs: m.liveBs }, false);
      } else if (m.segments) {
        renderBlockTurn(
          {
            el,
            bs: blockStreamFromSavedSegments(
              m.segments,
              m.content,
              state.vis,
              _replayOptsFromMessage(m),
            ),
          },
          false,
        );
      } else if (
        m.status === "running"
        && (m.content || "").trim()
        && typeof blockStreamFromSavedSegments === "function"
      ) {
        renderBlockTurn(
          {
            el,
            bs: blockStreamFromSavedSegments(
              [{ kind: "assistant", thinking: "", content: m.content }],
              m.content,
              state.vis,
              _replayOptsFromMessage(m),
            ),
          },
          false,
        );
      }
      // Re-attach the compaction-stats card (sessions.js) — see rebuild().
      if (typeof mountCompactionStatsCard === "function") {
        mountCompactionStatsCard(el, m.compactionStats || null);
      }
    }
    if (m.role === "ask_user") _remountAskUserFromCache(m, el);
    const mc = messagesContainer;
    if (beforeNode && beforeNode.parentNode === mc)
      mc.insertBefore(el, beforeNode);
    else if (
      state.scroll.bottomMarker &&
      state.scroll.bottomMarker.parentNode === mc
    )
      mc.insertBefore(el, state.scroll.bottomMarker);
    else mc.appendChild(el);
    return el;
  },

  _updateTopSentinel() {
    const sentinel = state.scroll.sentinelTop;
    if (!sentinel) return;
    const older = state.scroll.renderedStart || 0;
    sentinel.style.height =
      older > 0 ? `${this._estimateHeight(0, older)}px` : "0";
  },

  /** Render the latest window of the cache (called once after a full load). */
  renderTail() {
    const cache = state.msgCache;
    const start = Math.max(0, cache.length - INITIAL_RENDER);
    state.scroll.renderedStart = start;
    for (let i = start; i < cache.length; i++) this._materialise(i, null);
    this._updateTopSentinel();
  },

  /** Prepend the next older batch above the current top. Preserves scroll. */
  loadOlder() {
    const cache = state.msgCache;
    const oldStart = state.scroll.renderedStart || 0;
    if (oldStart <= 0) {
      // Cache exhausted at the top — pull the next OLDER page from the server
      // (backward pagination). prependOlder() splices it in below.
      loadOlderPage();
      return;
    }
    const mc = messagesContainer;
    const prevH = mc.scrollHeight,
      prevTop = mc.scrollTop;
    const anchor =
      (cache[oldStart] && cache[oldStart].el) || state.scroll.bottomMarker;
    const newStart = Math.max(0, oldStart - OLDER_BATCH);
    for (let i = newStart; i < oldStart; i++) this._materialise(i, anchor);
    state.scroll.renderedStart = newStart;
    this._updateTopSentinel();
    // Content grew above the viewport — keep the user where they were.
    mc.scrollTop = prevTop + (mc.scrollHeight - prevH);
  },

  /** Prepend an OLDER network page to the cache, preserving scroll position.
   * unshift shifts every absolute index by k, so we bump the scroller's index
   * pointers in lockstep (the index-shift fix). Position-agnostic (Codex #2): if
   * the view is at the top we materialise the new rows in place; otherwise we
   * just grow the cache + top sentinel and let loadOlder() materialise them on a
   * later scroll-up — so a user who scrolled down mid-fetch isn't corrupted. */
  prependOlder(olderMsgs) {
    const cache = state.msgCache;
    const mc = messagesContainer;
    const k = (olderMsgs && olderMsgs.length) || 0;
    if (!k || !mc) return;
    const prevH = mc.scrollHeight,
      prevTop = mc.scrollTop;
    const wasAtTop = (state.scroll.renderedStart || 0) === 0;
    const anchor = wasAtTop
      ? (cache[0] && cache[0].el) || state.scroll.bottomMarker
      : null;
    const stamp = Math.round(performance.now());
    const entries = [];
    olderMsgs.forEach((m, i) => {
      if (m == null) return; // consumed as MEMORY FLUSH reply pair
      let role = m.role,
        content = m.content,
        images = null,
        segments = Array.isArray(m.segments) ? m.segments : null;
      const artifacts = Array.isArray(m.artifacts) ? m.artifacts : null;
      if (role === "user") {
        const p = _userPromptParts(m.content, m.attached_files);
        content = p.text;
        images = p.files.length ? [...p.images, ...p.files] : p.images;
        if (_shouldOmitUserHistoryPrompt(content)) return;
        // Pair MEMORY FLUSH + following assistant into one collapsed widget
        // when both land in the same older-history page.
        if (_isMemoryFlushPrompt(content)) {
          let reply = "";
          const next = olderMsgs[i + 1];
          if (next && next.role === "assistant") {
            reply = typeof next.content === "string" ? next.content : "";
            olderMsgs[i + 1] = null; // skip on next iteration
          }
          entries.push({
            id: `msg-old-${stamp}-${i}`,
            role: "memory_flush",
            author: null,
            content: { prompt: content, reply },
            images: null,
            segments: null,
            artifacts: null,
            agent: m.agent ? String(m.agent) : "",
            timestamp: m.timestamp ? String(m.timestamp) : "",
            activityDisclosureState: null,
            el: null,
            height: 0,
          });
          return;
        }
      }
      const askRows = typeof _splitAskHistoryMessage === "function"
        ? _splitAskHistoryMessage(m) : null;
      const rows = askRows || [{ ...m, role, content, images, segments, artifacts }];
      rows.forEach((row, part) => {
        entries.push({
          id: `msg-old-${stamp}-${i}-${part}`,
          role: row.role,
          author: row.role === "user" ? authorFromMessage(row) : null,
          content: row.content,
          images: Array.isArray(row.images) && row.images.length ? row.images : null,
          segments: row.segments && row.segments.length ? row.segments : null,
          artifacts: row.artifacts && row.artifacts.length ? row.artifacts.slice() : null,
          askCall: row.askCall || null,
          agent: row.agent ? String(row.agent) : "",
          timestamp: row.timestamp ? String(row.timestamp) : "",
          aborted: !!row.aborted || row.status === "aborted",
          status: row.status ? String(row.status) : "",
          error: row.error != null && String(row.error) ? String(row.error) : "",
          endedAt: row.endedAt || row.ended_at ? String(row.endedAt || row.ended_at) : "",
          lastTs: row.lastTs || row.endedAt || row.ended_at || row.timestamp
            ? String(row.lastTs || row.endedAt || row.ended_at || row.timestamp)
            : "",
          activityDisclosureState: null,
          el: null,
          height: 0,
        });
      });
    });
    if (!entries.length) return;
    const kKept = entries.length;
    cache.unshift(...entries);
    // Index-shift fix: every existing index moved up by the kept count.
    state.scroll.windowStart = (state.scroll.windowStart || 0) + kKept;
    state.scroll.windowEnd = (state.scroll.windowEnd || 0) + kKept;
    state.scroll.renderedStart = (state.scroll.renderedStart || 0) + kKept;
    if (wasAtTop) {
      // Materialise the new rows [0,kKept) before the old top; all rendered from 0.
      for (let i = 0; i < kKept; i++) this._materialise(i, anchor);
      state.scroll.renderedStart = 0;
      this._updateTopSentinel();
      mc.scrollTop = prevTop + (mc.scrollHeight - prevH); // keep the user in place
    } else {
      // User scrolled down mid-fetch: just reflect the new unrendered older rows
      // in the sentinel; they materialise via loadOlder() when scrolled back up.
      this._updateTopSentinel();
    }
  },
};

// A completed background subagent is delivered to the parent as the parent's
// NEXT PROMPT — a machine-generated ``<task_result …>`` / ``<task_error …>``
// envelope (see ``_resume_parent`` in async_delegate_runner.py). Rendering that
// as a plain "You" bubble dumped the whole child transcript into the main chat
// and buried the actual conversation. Recognise the envelope so it can render
// as a collapsed card instead: OpenCode keeps child-session output out of the
// parent thread entirely (the child is a separate, navigable session), and this
// is the equivalent — the payload stays reachable, just not shouted.
//
// Returns ``null`` for anything an operator could plausibly have typed; the
// match is anchored and requires the machine-generated attribute set.
// The body match is GREEDY: a non-greedy `[\s\S]*?` stops at the FIRST
// `</task_result>` line, so a child whose report quotes that literal (an agent
// discussing this very code does) had its body truncated and the remainder
// silently hidden as trailer. Greedy anchors on the LAST close instead.
//
// `job_id` must be canonical — that is what makes this machine provenance rather
// than a shape an operator could stumble into. `task_id` may be EMPTY: a
// delegation that fails before its child session exists yields
// `<task_error task_id="" …>`, and requiring canonical-nonempty there would
// render real failure envelopes as raw "You" bubbles.
const _TASK_ENVELOPE_RE =
  /^<task_(result|error)\s+task_id="([0-9]{8}T[0-9]{12}-[0-9a-f]{12}|)"\s+job_id="(job_[0-9a-f]{16})"\s+agent="([^"]*)"\s*>\n([\s\S]*)\n<\/task_\1>\s*([\s\S]*)$/;

function _parseTaskEnvelope(text) {
  if (typeof text !== "string" || !text.startsWith("<task_")) return null;
  const m = _TASK_ENVELOPE_RE.exec(text);
  if (!m) return null;
  return {
    failed: m[1] === "error",
    taskId: m[2],
    agent: m[4],
    body: m[5],
    trailer: (m[6] || "").trim(), // the "Incorporate this…" instruction line
  };
}

// Only the canonical machine envelope is a wave card. Fence ids must match.
// The digest is ONE line of JSON (json.dumps escapes newlines), so the body is
// matched as a single line: a quoted close tag inside a member's report cannot
// end it early, and a second wave delivered by the same process (same fence
// nonce) cannot be swallowed into the first wave's body.
const _WAVE_ENVELOPE_RE =
  /^<delegation_wave wave_id="(wave_[0-9a-f]{32})" terminal="([0-9]+)" total="([0-9]+)">\n⟦UNTRUSTED-DATA id=([0-9a-f]{16}) source=subagent-wave:\1⟧\n([^\n]*)\n⟦\/UNTRUSTED-DATA id=\4⟧\n<\/delegation_wave>\s*([\s\S]*)$/;

// The runtime instruction line that follows every wave envelope.
const _WAVE_NOTE_RE = /^All children in this delegation wave are terminal\.[^\n]*/;

// A busy parent receives queued wakes joined into ONE prompt, so a prompt can
// hold several waves back to back. Parse every leading wave; whatever follows
// them is returned as ``trailer`` and rendered, never dropped.
//
// A digest stored before the scrubber stopped eating escape backslashes has
// `<<REDACTED:KIND>>"` where the original was `<<REDACTED:KIND>>\"`. That is
// invalid JSON, so repair exactly that one shape before giving up — otherwise
// a real, machine-generated wave would fall through and paint its whole body
// as a "Background task" bubble. A body that is still unusable after the repair keeps its envelope
// but renders with no payload at all: suppression is the contract, so an
// unreadable wave must never dump its text either.
const _WAVE_SENTINEL_QUOTE_RE = /(<<REDACTED:[A-Z_]+>>)"/g;

function _repairWaveBody(body) {
  return String(body || "").replace(_WAVE_SENTINEL_QUOTE_RE, '$1\\"');
}

function _parseWaveEnvelope(text) {
  if (typeof text !== "string" || !text.startsWith("<delegation_wave ")) return null;
  const waves = [];
  let rest = text;
  while (rest.startsWith("<delegation_wave ")) {
    const m = _WAVE_ENVELOPE_RE.exec(rest);
    if (!m) break;
    let members = null;
    try { members = JSON.parse(m[5]); } catch (_) {
      try { members = JSON.parse(_repairWaveBody(m[5])); } catch (_) { members = null; }
    }
    const usable = Array.isArray(members) &&
      !members.some((row) => !row || typeof row !== "object" || Array.isArray(row));
    waves.push(usable
      ? { waveId: m[1], terminal: Number(m[2]), total: Number(m[3]), members }
      : { waveId: m[1], terminal: Number(m[2]), total: Number(m[3]), members: [], unreadable: true });
    rest = (m[6] || "").replace(_WAVE_NOTE_RE, "").trim();
  }
  if (!waves.length) return null;
  return { ...waves[0], waves, trailer: rest };
}

// A wave card never paints a child's report. The child is a separate,
// navigable session — the same contract as a delegate card row — so each
// member links to that session and the transcript stays one click away
// instead of being dumped into the parent thread. Only the failure reason
// (status metadata, not output) stays on the row.
const _WAVE_LABELS = { done: "ok", failed: "failed", cancelled: "stopped", aborted: "stopped", refused: "refused" };
const _WAVE_VERBS = { done: "Completed", failed: "Failed", cancelled: "Stopped", aborted: "Stopped", refused: "Refused" };
const _WAVE_ICONS = { done: "✓", failed: "✗", cancelled: "⏹", aborted: "⏹", refused: "⊘" };

// The child session id the digest carries: `task_id` for current waves,
// `result_ref.child_sid` for the same record shape read by other consumers.
function _waveMemberSessionId(member) {
  const ref = member && member.result_ref && typeof member.result_ref === "object"
    ? member.result_ref : {};
  return String(member.task_id || ref.child_sid || "");
}

function _waveCardHtml(wave) {
  if (typeof _bindDelegationCardLinks === "function") _bindDelegationCardLinks();
  const plural = wave.total === 1 ? "" : "s";
  // No usable digest (unreadable body, or a wave with no members): still a
  // card, still no payload. The child sessions remain reachable from the
  // Agents panel even when the digest cannot name them.
  if (!wave.members.length) {
    return `<div class="task-card wave-card" title="${escapeAttr(wave.waveId)}">` +
      `<div class="task-card-summary wave-card-head">` +
      `<span class="task-card-icon">↩</span><span class="task-card-agent">Subagent wave</span>` +
      `<span class="task-card-verb">${wave.total} subagent${plural} · output suppressed</span></div></div>`;
  }
  const counts = new Map();
  const agents = new Map();
  const rows = wave.members.map((member) => {
    const label = _WAVE_LABELS[member.status] || String(member.status || "");
    counts.set(label, (counts.get(label) || 0) + 1);
    const agent = String(member.agent || "subagent");
    agents.set(agent, (agents.get(agent) || 0) + 1);
    const verb = _WAVE_VERBS[member.status] || String(member.status || "");
    const icon = _WAVE_ICONS[member.status] || "·";
    const error = String(member.error || "");
    const sessionId = _waveMemberSessionId(member);
    const href = sessionId
      ? (typeof BASE === "string" ? BASE : "") + "/?session=" + encodeURIComponent(sessionId)
      : "";
    const tag = href ? "a" : "span";
    const attrs = href
      ? ` href="${escapeAttr(href)}" data-delegate-session="${escapeAttr(sessionId)}"` +
        ` title="Open subagent session"` +
        ` aria-label="${escapeAttr(`${agent} subagent session (${verb})`)}"`
      : "";
    const reason = error
      ? `<span class="delegate-description" title="${escapeAttr(error)}">` +
        `${escapeHtml(error.length > 120 ? error.slice(0, 117) + "…" : error)}</span>`
      : "";
    return `<${tag} class="delegate-child wave-member" data-status="${escapeAttr(member.status || "")}"${attrs}>` +
      `<span class="tc-status" aria-hidden="true">${icon}</span>` +
      `<span class="delegate-identity"><span class="tc-name">${escapeHtml(agent)}</span>${reason}</span>` +
      `<span class="delegate-verb" role="status">${escapeHtml(verb)}</span></${tag}>`;
  }).join("");
  const outcome = [...counts].map(([label, count]) => `${count} ${label}`).join(" · ");
  const who = [...agents].map(([agent, count]) => (count > 1 ? `${agent} ×${count}` : agent)).join(", ");
  // Lead with what the reader cares about; the wave id stays a tooltip.
  return `<div class="task-card wave-card" title="${escapeAttr(wave.waveId)}">` +
    `<div class="task-card-summary wave-card-head">` +
    `<span class="task-card-icon">↩</span><span class="task-card-agent">${escapeHtml(who)}</span>` +
    `<span class="task-card-verb">${wave.total} subagent${plural}` +
    `${wave.terminal === wave.total ? " finished" : ` · ${wave.terminal} terminal`} · ${escapeHtml(outcome)}</span></div>` +
    `<div class="wave-members">${rows}</div></div>`;
}

function _createWaveResultElement(parsed) {
  const div = document.createElement("div");
  div.className = "message message-task message-wave";
  div.dataset.waveId = parsed.waveId;
  const trailer = parsed.trailer
    ? `<div class="message-content wave-trailer">${renderMarkdown(parsed.trailer)}</div>` : "";
  div.innerHTML = (parsed.waves || [parsed]).map(_waveCardHtml).join("") + trailer;
  return div;
}

// Realtime `_voice_result` (session.py) re-submits a long background-subagent
// body as a NEW parent prompt so Jarvis can summarize it for TTS. That prompt
// is machine-authored and must never paint as a "You" bubble on live turns or
// history reload — it dumps the child's entire transcript into the chat as if
// the operator typed it. Match the exact lead-in; the body is usually a
// `<task id="…" state="…">…</task>` wrapper but may be bare text.
const _VOICE_RELAY_RE =
  /^Read out only the result below for the user, in one or two short spoken sentences\.\s*Ignore any planning, reasoning, or tool-use narration it contains:\s*\n+([\s\S]+)$/;
const _VOICE_RELAY_TASK_RE =
  /^<task\s+id="([0-9]{8}T[0-9]{12}-[0-9a-f]{12}|)"[^>]*>\s*([\s\S]*?)\s*<\/task>\s*$/;

function _parseVoiceRelayPrompt(text) {
  if (typeof text !== "string" || !text.startsWith("Read out only the result below")) {
    return null;
  }
  const m = _VOICE_RELAY_RE.exec(text);
  if (!m) return null;
  let body = (m[1] || "").trim();
  let taskId = "";
  const wrapped = _VOICE_RELAY_TASK_RE.exec(body);
  if (wrapped) {
    taskId = wrapped[1] || "";
    body = wrapped[2] || "";
  }
  return {
    failed: false,
    taskId,
    agent: "voice",
    body,
    trailer: "",
    verb: "relayed for speech",
  };
}

/** Voice-TTS relay prompts duplicate the canonical ``<task_result>`` card and
 * must be omitted from history / live paint (not shown as "You"). */
function _shouldOmitUserHistoryPrompt(text) {
  return !!_parseVoiceRelayPrompt(text);
}

/** True when the user prompt is the silent end-of-session MEMORY.md flush. */
function _isMemoryFlushPrompt(text) {
  return typeof text === "string" && /^\s*\[MEMORY FLUSH\]/.test(text);
}

/** True when the assistant reply is the flush turn's done marker.
 * Anchored at the start like ``_isMemoryFlushPrompt``: a normal turn that
 * merely QUOTES ``[MEMORY_FLUSH_DONE]`` mid-text (e.g. a code review of this
 * widget) must not be wrapped as a memory-flush card. */
function _isMemoryFlushReply(text) {
  // Merge resolution: prefix-anchored match — a real flush
  // reply may carry a trailing note ("[MEMORY_FLUSH_DONE]\n\nPersisted …"),
  // while quoted mid-text markers must never match.
  return typeof text === "string" && /^\s*\[MEMORY_FLUSH_DONE\]/.test(text);
}

/** Fill / reveal the reply slot inside a memory-flush card. */
function _fillMemoryFlushReply(cardEl, replyText) {
  if (!cardEl) return;
  const slot = cardEl.querySelector(".memory-flush-reply");
  if (!slot) return;
  const raw = typeof replyText === "string" ? replyText : "";
  slot.hidden = !raw.trim();
  slot.innerHTML = raw.trim() ? renderAssistantContent(raw) : "";
  slot.dataset.raw = raw;
}

/** Collapsed expandable widget for a MEMORY FLUSH turn. */
function _createMemoryFlushElement(promptText, replyText, timestamp) {
  const div = document.createElement("div");
  div.className = "message message-memory-flush";
  const prompt = typeof promptText === "string" ? promptText : "";
  const reply = typeof replyText === "string" ? replyText : "";
  const replyHidden = reply.trim() ? "" : " hidden";
  const timeLabel = formatMessageTime(timestamp);
  div.innerHTML = `
    <details class="task-card memory-flush-card">
      <summary class="task-card-summary">
        <span class="task-card-icon">M</span>
        <span class="task-card-agent">Memory flush</span>
        <span class="task-card-verb">idle snapshot</span>
        ${timeLabel ? `<span class="message-time">${timeLabel}</span>` : ""}
      </summary>
      <div class="task-card-body">
        <div class="memory-flush-prompt">${escapeHtml(prompt).replace(/\n/g, "<br>")}</div>
        <div class="memory-flush-reply message-content"${replyHidden}></div>
      </div>
    </details>`;
  if (reply.trim()) _fillMemoryFlushReply(div, reply);
  return div;
}

/** Collapsed card for a background subagent's delivered result. */
function _createTaskResultElement(task) {
  const div = document.createElement("div");
  div.className =
    "message message-task" + (task.failed ? " message-task-failed" : "");
  const agent = escapeHtml(task.agent || "subagent");
  const chars = (task.body || "").length;
  const size =
    chars >= 1000 ? `${(chars / 1000).toFixed(1)}k chars` : `${chars} chars`;
  const verb = task.verb || (task.failed ? "failed" : "returned");
  div.innerHTML = `
    <details class="task-card">
      <summary class="task-card-summary">
        <span class="task-card-icon">${task.failed ? "✕" : "↩"}</span>
        <span class="task-card-agent">${agent}</span>
        <span class="task-card-verb">${escapeHtml(verb)}</span>
        <span class="task-card-size">${escapeHtml(size)}</span>
      </summary>
      <div class="task-card-body message-content">${renderMarkdown(task.body)}</div>
    </details>`;
  div.dataset.taskId = task.taskId || "";
  return div;
}

// Wire sentinel the /compact button and /claude|/codex|/agent slash
// commands persist as the user turn. Never paint it as a "You" bubble —
// operators should see a small card, not `__SHAREDAGENTS_DELEGATE__:…`.
const _DELEGATE_MARKER_PREFIX = "__SHAREDAGENTS_DELEGATE__:";

function _isCompactUserText(text) {
  const t = String(text || "").trim();
  if (!t) return false;
  if (t.startsWith(_DELEGATE_MARKER_PREFIX + "compact_context")) return true;
  return t === "/compact" || t.startsWith("/compact ") || t.startsWith("/compact\n");
}

function _parseDelegateMarker(text) {
  if (typeof text !== "string" || !text.startsWith(_DELEGATE_MARKER_PREFIX)) {
    return null;
  }
  const rest = text.slice(_DELEGATE_MARKER_PREFIX.length);
  const colon = rest.indexOf(":");
  if (colon < 0) return null;
  const tool = rest.slice(0, colon);
  if (!tool) return null;
  let args;
  try {
    args = JSON.parse(rest.slice(colon + 1));
  } catch (_) {
    return null;
  }
  if (!args || typeof args !== "object" || Array.isArray(args)) return null;
  return { tool, args };
}

function _delegateCardInfo(parsed) {
  const tool = String((parsed && parsed.tool) || "");
  const args = (parsed && parsed.args) || {};
  const prompt = String(args.prompt || args.task || "").trim();
  if (tool === "compact_context") {
    return { icon: "🗜", title: "Compact context", detail: "", compact: true };
  }
  let name = tool;
  if (tool === "delegate_agent") {
    name = String(args.name || "agent").trim() || "agent";
  } else if (name.startsWith("delegate_")) {
    name = name.slice("delegate_".length) || "agent";
  }
  let detail = prompt;
  if (detail.length > 80) detail = detail.slice(0, 77) + "…";
  return { icon: "↪", title: "/" + name, detail, compact: false };
}

/** Small static card replacing a `__SHAREDAGENTS_DELEGATE__` user turn. */
function _createDelegateCardElement(parsed, timestamp) {
  const info = _delegateCardInfo(parsed);
  const div = document.createElement("div");
  div.className =
    "message message-delegate" +
    (info.compact ? " message-delegate-compact" : "");
  // /compact is a control action; its output lives in the compaction child.
  div.hidden = info.compact;
  const timeLabel = formatMessageTime(timestamp);
  const detail = info.detail
    ? `<span class="task-card-verb">${escapeHtml(info.detail)}</span>`
    : "";
  div.innerHTML = `
    <div class="task-card delegate-card">
      <div class="task-card-summary">
        <span class="task-card-icon">${escapeHtml(info.icon)}</span>
        <span class="task-card-agent">${escapeHtml(info.title)}</span>
        ${detail}
        ${timeLabel ? `<span class="message-time">${timeLabel}</span>` : ""}
      </div>
    </div>`;
  return div;
}

/** Create a message DOM element from role + content without appending.
 * ``images`` (user prompts) are rendered as thumbnails below the text.
 * ``agent`` is the per-turn author; falls back to the active picker agent. */
/** Locale-aware "date + time to the second" for a message bubble.
 * Reuses the web UI's established date convention (same ``toLocaleString``
 * options as goals.js / loops.js), so every timestamp render in the app
 * shares one shape. ``hour12`` is pinned so AM/PM stays stable regardless of
 * browser locale. Returns "" when the timestamp is missing or unparseable
 * so the bubble simply omits the time slot — never crashes, never renders a
 * bogus date. */
function formatMessageTime(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });
}

function _sessionReplayPhase() {
  const sid = typeof state !== "undefined" && state.currentSessionId;
  const row = sid && state.sessions && state.sessions[sid];
  return String((row && (row.state || row.session_state)) || "");
}

function _replayOptsFromMessage(m) {
  return {
    aborted: !!(m && (m.aborted || m.status === "aborted")),
    error: (m && m.error) || (m && m.status === "error" ? "error" : null),
    inProgress: !!(m && m.status === "running"),
    phase: _sessionReplayPhase(),
  };
}

function _historyTurnStatus(aborted, status, error) {
  if (aborted || status === "aborted") return "aborted";
  if (status === "error" || (error != null && String(error))) return "error";
  if (status === "running") return "running";
  if (status === "ok") return "ok";
  return "";
}

function _historyTurnPillHtml(status) {
  const labels = {
    aborted: "aborted",
    error: "error",
    ok: "succeeded",
    running: "running",
  };
  const classes = {
    aborted: "aborted-pill",
    error: "error-pill",
    ok: "ok-pill",
    running: "running-pill",
  };
  const label = labels[status];
  const cls = classes[status];
  if (!label || !cls) return "";
  return `<span class="live-pill ${cls}">${label}</span>`;
}

function _remountAskUserFromCache(m, el) {
  if (!m || !el || !m.askCall || typeof renderAskUserCard !== "function") return;
  const call = m.askCall;
  const callId = String(call.callId || "");
  if (!callId) return;
  if (state.askUserCards && state.askUserCards[callId]) {
    delete state.askUserCards[callId];
  }
  const host = el.querySelector(".message-content") || el;
  renderAskUserCard(callId, call.args || {}, host);
  if (call.submitted && typeof _restoreSubmittedAsk === "function") {
    _restoreSubmittedAsk(callId, call.submitted);
  }
  if (call.finalized || call.content || call.ok != null || call.statusOverride) {
    if (typeof _finalizeAskUserCard === "function") {
      _finalizeAskUserCard(
        callId,
        call.content || "",
        call.ok,
        call.statusOverride,
      );
    }
  }
}

function _createMessageElement(role, content, images, agent, artifacts, timestamp, aborted, status, error, endedAt, author) {
  if (role === "memory_flush") {
    // Paired history entry: content is ``{prompt, reply}`` or a plain prompt.
    if (content && typeof content === "object") {
      return _createMemoryFlushElement(content.prompt || "", content.reply || "", timestamp);
    }
    return _createMemoryFlushElement(String(content || ""), "", timestamp);
  }
  if (role === "user") {
    // Realtime speech-relay dumps: skip entirely (canonical delivery is the
    // ``<task_result>`` card that follows). Returning null lets appendMessage
    // no-op on the live path.
    if (_shouldOmitUserHistoryPrompt(content)) return null;
    if (_isMemoryFlushPrompt(content)) {
      return _createMemoryFlushElement(content, "", timestamp);
    }
    const wave = _parseWaveEnvelope(content);
    if (wave) return _createWaveResultElement(wave);
    const task = _parseTaskEnvelope(content);
    if (task) return _createTaskResultElement(task);
    const delegate = _parseDelegateMarker(content);
    if (delegate) return _createDelegateCardElement(delegate, timestamp);
    if (_isCompactUserText(content)) {
      return _createDelegateCardElement(
        { tool: "compact_context", args: {} }, timestamp,
      );
    }
  }
  if (role === "assistant" && _isMemoryFlushReply(content)) {
    // Standalone flush reply (e.g. older-history page split) — same widget.
    return _createMemoryFlushElement("", content);
  }
  if (role === "system") {
    const div = document.createElement("div");
    div.className = `message message-system`;
    div.innerHTML = `
      <div class="message-header">
        <div class="message-avatar system">/</div>
        <span>command</span>
      </div>
      <div class="message-content">${renderMarkdown(content)}</div>`;
    return div;
  }
  if (role === "nudge") {
    const notice =
      typeof parseRuntimeNoticeLine === "function"
        ? parseRuntimeNoticeLine(content)
        : null;
    if (typeof _createNudgeCardElement === "function") {
      return _createNudgeCardElement(notice, timestamp);
    }
  }
  if (role === "ask_user") {
    const div = document.createElement("div");
    div.className = "message message-system ask-user-card";
    div.innerHTML = `
      <div class="message-header">
        <div class="message-avatar system">?</div>
        <span>question</span>
      </div>
      <div class="message-content"></div>`;
    return div;
  }

  const div = document.createElement("div");
  div.className = `message message-${role}`;
  if (role === "assistant" && aborted) div.classList.add("message-aborted");

  // Prefer the per-message agent over the active picker. After a build→cursor
  // switch, older bubbles must still say "build", not the current lane.
  const agentName = String(agent || state.currentAgent || "agent");
  // A user bubble names its author for every viewer — never
  // "You". `author` = {actors, source} from the canonical event / history.
  const who = role === "user" ? authorInfo(author) : null;
  const avatar = who
    ? escapeHtml(who.initial)
    : escapeHtml(agentName.charAt(0).toUpperCase());
  const label = who ? escapeHtml(who.label) : escapeHtml(agentName);
  if (role === "assistant" && agentName) div.dataset.agent = agentName;
  if (who) {
    div.dataset.authorEmails = who.emails.join(",");
    div.dataset.authorLabel = who.label;
  }

  let bodyHtml =
    role === "assistant"
      ? renderAssistantContent(content)
      : escapeHtml(content).replace(/\n/g, "<br>");

  // Attached images (e.g. a photo sent from Telegram / WebUI attach) render as
  // thumbnails below the prompt text. Click opens the shared artifact lightbox
  // — browsers often refuse multi-MB data: new tabs, and authenticated /files
  // URLs are the durable source of truth.
  // Only safe media schemes are rendered — never javascript:/etc., even if a
  // malformed upstream event smuggles one in (the URL lands in data-attr + src).
  if (role === "user" && Array.isArray(images) && images.length) {
    // Saved non-image files arrive from the backend as ``{name, size, mime}``
    // chips (``attached_files``); render them as compact pills.
    const chips = images
      .filter((item) => item && typeof item === "object")
      .map((file) => {
        const name = String(file.name || "attachment");
        const size = Number(file.size) || 0;
        const sizeStr = size < 1024 ? `${size} B` : `${(size / 1024).toFixed(1)} KB`;
        const icon = typeof fileIcon === "function" ? fileIcon(name) : "📎";
        return (
          `<span class="user-file-chip" title="${escapeHtml(name).replace(/"/g, "&quot;")}">` +
          `<span class="file-icon" aria-hidden="true">${icon}</span>` +
          `<span class="file-name">${escapeHtml(name)}</span>` +
          `<span class="file-size">${escapeHtml(sizeStr)}</span></span>`
        );
      })
      .join("");
    if (chips) bodyHtml += `<div class="user-files">${chips}</div>`;
    const imgs = images
      .filter((src) => typeof src === "string")
      .map((src) => {
        const safe = _safeMediaUrl(src);
        if (!safe) return "";
        const s = escapeHtml(safe).replace(/"/g, "&quot;");
        return (
          `<button class="user-image-link" type="button"` +
          ` data-artifact-url="${s}" data-artifact-name="Attached image"` +
          ` aria-label="Open attached image fullscreen">` +
          `<img class="user-image" src="${s}" alt="attached image" loading="lazy"></button>`
        );
      })
      .join("");
    if (imgs) bodyHtml += `<div class="user-images">${imgs}</div>`;
  }

  const actions =
    role === "assistant"
      ? '<div class="message-actions"><button class="msg-copy" type="button" title="Copy answer as Markdown">Copy</button></div>'
      : "";

  const timeLabel = formatMessageTime(timestamp);
  const outcome = role === "assistant"
    ? _historyTurnStatus(aborted, status, error)
    : "";

  div.innerHTML = `
    <div class="message-header">
      <div class="message-avatar ${escapeHtml(role)}">${avatar}</div>
      ${who
        ? `<span class="message-author" title="${escapeHtml(who.title).replace(/"/g, "&quot;")}">${label}</span>`
        : `<span>${label}</span>`}
      ${timeLabel ? `<span class="message-time">${timeLabel}</span>` : ""}
    </div>
    <div class="message-content">${bodyHtml}</div>
    ${role === "assistant" ? '<div class="message-artifacts" hidden></div>' : ""}
    ${role === "assistant" ? '<div class="message-telemetry"></div>' : ""}
    ${actions}
  `;
  if (role === "assistant") {
    if (outcome) div.dataset.turnStatus = outcome;
    if (timestamp) div.dataset.turnStarted = String(timestamp);
    if (endedAt) div.dataset.turnEnded = String(endedAt);
    if (outcome === "aborted") div.classList.add("message-aborted");
    else if (outcome === "error") div.classList.add("message-error");
    else if (outcome === "ok") div.classList.add("message-ok");
    else if (outcome === "running") div.classList.add("message-running");
    renderMessageArtifacts(div, artifacts, state.currentSessionId);
    if (typeof renderMessageTelemetry === "function") renderMessageTelemetry(div);
  } else if (
    role === "user" &&
    Array.isArray(images) &&
    images.length &&
    typeof _bindArtifactOutputUi === "function"
  ) {
    // Wire user-thumb lightbox clicks (same helper as agent output images).
    _bindArtifactOutputUi(div);
  }
  return div;
}


/** Re-label an already-painted user bubble (optimistic echo → canonical
 *  turn_started author) and its cache row. */
function _applyMessageAuthor(el, author) {
  if (!el || !el.classList || !el.classList.contains("message-user")) return;
  const who = authorInfo(author);
  const nameEl = el.querySelector(".message-author");
  const avatarEl = el.querySelector(".message-avatar.user");
  if (nameEl) {
    nameEl.textContent = who.label;
    nameEl.title = who.title;
  }
  if (avatarEl) avatarEl.textContent = who.initial;
  el.dataset.authorEmails = who.emails.join(",");
  el.dataset.authorLabel = who.label;
  const row = state.msgCache.find((m) => m && m.el === el);
  if (row) {
    row.author = {
      actors: Array.isArray(author && author.actors) ? author.actors.slice() : [],
      source: String((author && author.source) || ""),
    };
  }
}
