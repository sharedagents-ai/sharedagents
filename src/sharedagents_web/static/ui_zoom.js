/* sharedagents-web — UI zoom preference.
 *
 * Chrome/Android CSS `zoom` scales the whole chrome (the phone complaint).
 * Persist to localStorage so Settings +/- survive reloads. Loaded in <head>
 * so the first paint is already at the saved scale.
 */
(function () {
  "use strict";

  var KEY = "sharedagents.web.uiZoom";
  var MIN = 70;
  var MAX = 130;
  var STEP = 10;
  var DEFAULT = 100;

  function clamp(n) {
    n = Math.round(Number(n) / STEP) * STEP;
    if (!isFinite(n)) return DEFAULT;
    return Math.min(MAX, Math.max(MIN, n));
  }

  function read() {
    try {
      var raw = localStorage.getItem(KEY);
      if (raw == null || raw === "") return DEFAULT;
      return clamp(raw);
    } catch (_e) {
      return DEFAULT;
    }
  }

  function write(pct) {
    try {
      localStorage.setItem(KEY, String(pct));
    } catch (_e) { /* private mode */ }
  }

  function supportsZoom() {
    try {
      return typeof CSS !== "undefined" && CSS.supports && CSS.supports("zoom", "1");
    } catch (_e) {
      return false;
    }
  }

  function applyFallback(root, factor) {
    if (factor === 1) {
      root.style.transform = "";
      root.style.transformOrigin = "";
      root.style.width = "";
      return;
    }
    root.style.transform = "scale(" + factor + ")";
    root.style.transformOrigin = "top left";
    root.style.width = 100 / factor + "%";
  }

  function syncControls(pct) {
    var label = document.getElementById("zoom-value");
    if (label) label.textContent = pct + "%";
    var out = document.getElementById("zoom-out-btn");
    var inn = document.getElementById("zoom-in-btn");
    if (out) out.disabled = pct <= MIN;
    if (inn) inn.disabled = pct >= MAX;
  }

  function apply(pct) {
    pct = clamp(pct);
    var root = document.documentElement;
    var factor = pct / 100;
    root.dataset.uiZoom = String(pct);
    root.style.setProperty("--ui-zoom", String(factor));
    if (supportsZoom()) {
      root.style.zoom = String(factor);
      applyFallback(root, 1);
    } else {
      root.style.zoom = "";
      applyFallback(root, factor);
    }
    write(pct);
    syncControls(pct);
    return pct;
  }

  function bind() {
    var out = document.getElementById("zoom-out-btn");
    var inn = document.getElementById("zoom-in-btn");
    var reset = document.getElementById("zoom-reset-btn");
    if (out) {
      out.addEventListener("click", function () {
        apply(read() - STEP);
      });
    }
    if (inn) {
      inn.addEventListener("click", function () {
        apply(read() + STEP);
      });
    }
    if (reset) {
      reset.addEventListener("click", function () {
        apply(DEFAULT);
      });
    }
    apply(read());
  }

  apply(read());
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }

  window.UiZoom = {
    KEY: KEY,
    MIN: MIN,
    MAX: MAX,
    STEP: STEP,
    DEFAULT: DEFAULT,
    read: read,
    apply: apply,
  };
})();
