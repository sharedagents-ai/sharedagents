/* menu_state.js — persist left-menu expand/hide; fetch list APIs only when shown.
 *
 * Defaults: sidebar hidden; MENU / TOPICS / CONVERSATIONS expanded once the
 * drawer is opened. Sidebar preferences are scoped to this browser tab and
 * restored across its page navigations on desktop (width > 768px). On mobile
 * or narrow viewports (width <= 768px), the menu collapses on navigation and
 * link selection so it never obstructs page content. Loaders registered by
 * app.js / nav.js fire only when the sidebar is visible AND that section is
 * open, and only once per page load.
 */
(function (global) {
  "use strict";
  const KEY = "sharedagents.web.menu";
  const DEFAULTS = {
    sidebar: false,
    nav: true,
    topics: true,
    conversations: true,
  };
  const SECTION_IDS = {
    nav: "nav-section",
    topics: "sidebar-topics",
    conversations: "sidebar-conversations",
  };
  const MQ_MOBILE = "(max-width: 768px)";

  function readPrefs() {
    try {
      const raw = JSON.parse(sessionStorage.getItem(KEY) || "{}") || {};
      return Object.assign({}, DEFAULTS, raw);
    } catch (_e) {
      return Object.assign({}, DEFAULTS);
    }
  }

  function writePrefs(next) {
    try { sessionStorage.setItem(KEY, JSON.stringify(next)); } catch (_e) { /* private mode */ }
  }

  let prefs = readPrefs();
  const loaded = Object.create(null);
  const loaders = Object.create(null);

  function isMobile() {
    if (global.matchMedia) {
      return !!global.matchMedia(MQ_MOBILE).matches;
    }
    return typeof global.innerWidth === "number" ? global.innerWidth <= 768 : false;
  }

  function sidebarEl() {
    return document.getElementById("sidebar");
  }

  function sidebarVisible() {
    const el = sidebarEl();
    if (!el) return false;
    if (isMobile()) return document.body.classList.contains("sidebar-open");
    return !el.classList.contains("collapsed");
  }

  function applySidebarVisible(want) {
    const el = sidebarEl();
    if (!el) return;
    const open = !!want;
    if (isMobile()) {
      document.body.classList.toggle("sidebar-open", open);
      const backdrop = document.getElementById("sidebar-backdrop");
      if (backdrop) backdrop.hidden = !open;
    } else {
      el.classList.toggle("collapsed", !open);
      document.body.classList.remove("sidebar-open");
      const backdrop = document.getElementById("sidebar-backdrop");
      if (backdrop) backdrop.hidden = true;
    }
    el.setAttribute("aria-hidden", open ? "false" : "true");
    el.inert = !open;
    const menuBtn = document.getElementById("mon-menu-btn");
    if (menuBtn) menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    const toggle = document.getElementById("toggle-sidebar");
    if (toggle) toggle.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function applySections() {
    for (const [key, id] of Object.entries(SECTION_IDS)) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.open = prefs[key] !== false;
    }
  }

  function persistSection(id, open) {
    const key = Object.keys(SECTION_IDS).find((k) => SECTION_IDS[k] === id);
    if (!key) return;
    prefs[key] = !!open;
    writePrefs(prefs);
  }

  function persistSidebar(open) {
    prefs.sidebar = !!open;
    writePrefs(prefs);
  }

  function maybeLoad(id) {
    if (loaded[id]) return false;
    if (!sidebarVisible()) return false;
    const el = document.getElementById(id);
    if (!el || !el.open) return false;
    const fn = loaders[id];
    if (typeof fn !== "function") return false;
    loaded[id] = true;
    // A loader that settles to false loaded nothing (the chat boot read owed
    // this list and could not carry it): the next open — a click — retries.
    const done = fn();
    if (done && typeof done.then === "function") {
      done.then((ok) => { if (ok === false) loaded[id] = false; }, () => {});
    }
    return true;
  }

  function loadOpenSections() {
    for (const id of Object.values(SECTION_IDS)) maybeLoad(id);
    if (typeof loaders.approvals === "function" && sidebarVisible()) {
      if (!loaded.approvals) {
        loaded.approvals = true;
        loaders.approvals();
      }
    }
  }

  function registerLoader(id, fn) {
    loaders[id] = fn;
    if (sidebarVisible()) {
      maybeLoad(id);
    }
  }

  function setSidebarVisible(want, opts) {
    const persist = !opts || opts.persist !== false;
    applySidebarVisible(want);
    if (persist && !isMobile()) persistSidebar(want);
    if (want) loadOpenSections();
  }

  function toggleSidebar() {
    setSidebarVisible(!sidebarVisible());
  }

  function sectionOpen(id) {
    const el = document.getElementById(id);
    return !!el && !!el.open && sidebarVisible();
  }

  function syncViewport(e) {
    const mobile = (e && typeof e.matches === "boolean") ? e.matches : isMobile();
    if (mobile) {
      applySidebarVisible(false);
    } else {
      applySidebarVisible(prefs.sidebar);
      if (prefs.sidebar) loadOpenSections();
    }
  }

  function bind() {
    prefs = readPrefs();
    applySidebarVisible(isMobile() ? false : prefs.sidebar);
    applySections();
    for (const id of Object.values(SECTION_IDS)) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.addEventListener("toggle", () => {
        persistSection(id, el.open);
        if (el.open) maybeLoad(id);
      });
    }

    const navList = document.getElementById("nav-list");
    if (navList) {
      navList.addEventListener("click", (e) => {
        if (!e.target.closest("a.nav-item")) return;
        if (isMobile()) setSidebarVisible(false, { persist: false });
      });
    }

    const logo = document.querySelector("#sidebar a.logo");
    if (logo) {
      logo.addEventListener("click", () => {
        if (isMobile()) setSidebarVisible(false, { persist: false });
      });
    }

    const sidebar = sidebarEl();
    if (sidebar) {
      sidebar.addEventListener("click", (e) => {
        if (!isMobile()) return;
        const link = e.target.closest("a[href], .session-item");
        if (link) {
          setSidebarVisible(false, { persist: false });
        }
      });
    }

    const backdrop = document.getElementById("sidebar-backdrop");
    if (backdrop) {
      backdrop.addEventListener("click", () => {
        if (isMobile()) setSidebarVisible(false, { persist: false });
      });
    }
  }

  const mql = global.matchMedia ? global.matchMedia(MQ_MOBILE) : null;
  if (mql) {
    if (mql.addEventListener) mql.addEventListener("change", syncViewport);
    else if (mql.addListener) mql.addListener(syncViewport);
  }

  const api = {
    KEY,
    readPrefs: () => Object.assign({}, prefs),
    sidebarVisible,
    setSidebarVisible,
    toggleSidebar,
    sectionOpen,
    registerLoader,
    maybeLoad,
    loadOpenSections,
    syncViewport,
    isMobile,
    bind,
  };
  global.MenuState = api;
  global._menuSectionOpen = sectionOpen;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})(window);
