/* confirm-card.js — in-UI replacement for the native browser dialog trio.
   window.confirm / window.prompt / window.alert are silently blocked in
   embedded contexts (the original "/compact errors on web" report), so every
   call site now routes through ONE reusable promise-based card:

       confirmCard({title, message, okLabel?, cancelLabel?, danger?})
         → Promise<boolean>          true only on explicit OK
       promptCard({title, message?, value?, okLabel?, cancelLabel?})
         → Promise<string|null>      null on cancel; trimmed value on OK
                                     (empty string allowed — caller validates)
       alertCard({title, message, okLabel?})
         → Promise<void>             resolves on OK / Escape / backdrop click

   Behavior contract (matches the native dialogs it replaces):
     * one DOM node, created lazily on first use, hidden with [hidden]
       (style.css supplies the .confirm-card* surface — amp/budget
       bottom-sheet conventions: dimmed fixed backdrop at z-index 9998,
       bottom-anchored panel at 9999, mobile full-width sheet <520px);
     * role="dialog" aria-modal="true" aria-labelledby=<title id>;
     * focus moves to the first control on open (prompt input, else Cancel
       for confirm, OK for alert) and is RESTORED to the triggering element
       on close;
     * Escape cancels, backdrop click cancels, Tab focus-traps inside the
       panel (render.js focus-trap pattern).

   Classic script sharing the global scope; loaded FIRST (before state.js
   per index.html) so any later file can call the card API. Call sites use
   the bare global names so the same source runs in the browser (window
   scope) and in the node harness (global scope).

   embedded contexts; the whole class goes through this one card. */
(function () {
  "use strict";

  var root = null;      // .confirm-card (backdrop + panel)
  var panel = null;
  var titleEl = null;
  var messageEl = null;
  var inputEl = null;
  var cancelBtn = null;
  var okBtn = null;
  var pending = null;   // { mode, resolve, lastFocus }

  function buildCard() {
    root = document.createElement("div");
    root.className = "confirm-card";
    root.hidden = true;

    var backdrop = document.createElement("div");
    backdrop.className = "confirm-card-backdrop";
    backdrop.addEventListener("click", function () { closeCard("cancel"); });

    panel = document.createElement("div");
    panel.className = "confirm-card-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-labelledby", "confirm-card-title");
    panel.addEventListener("click", function (e) { e.stopPropagation(); });

    titleEl = document.createElement("h3");
    titleEl.className = "confirm-card-title";
    titleEl.id = "confirm-card-title";

    messageEl = document.createElement("div");
    messageEl.className = "confirm-card-message";

    inputEl = document.createElement("input");
    inputEl.className = "confirm-card-input";
    inputEl.type = "text";
    inputEl.setAttribute("autocomplete", "off");
    inputEl.hidden = true;
    inputEl.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        okBtn.click();
      }
    });

    var actions = document.createElement("div");
    actions.className = "confirm-card-actions";

    cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "confirm-card-btn confirm-card-cancel";
    cancelBtn.textContent = "Cancel";
    cancelBtn.addEventListener("click", function () { closeCard("cancel"); });

    okBtn = document.createElement("button");
    okBtn.type = "button";
    okBtn.className = "confirm-card-btn confirm-card-ok";
    okBtn.textContent = "OK";
    okBtn.addEventListener("click", function () { closeCard("ok"); });

    actions.appendChild(cancelBtn);
    actions.appendChild(okBtn);
    panel.appendChild(titleEl);
    panel.appendChild(messageEl);
    panel.appendChild(inputEl);
    panel.appendChild(actions);
    root.appendChild(backdrop);
    root.appendChild(panel);
    (document.body || document.documentElement).appendChild(root);

    // Focus trap + Escape handling — registered once; a hidden card is a
    // no-op. Follows the render.js model-sheet trap.
    document.addEventListener("keydown", function (e) {
      if (root.hidden) return;
      if (e.key === "Escape") {
        e.preventDefault();
        closeCard("cancel");
        return;
      }
      if (e.key !== "Tab") return;
      var focusable = panel.querySelectorAll(
        "button:not([disabled]), input:not([disabled]), select:not([disabled])",
      );
      focusable = Array.prototype.filter.call(focusable, function (node) {
        return !node.closest("[hidden]");
      });
      if (!focusable.length) return;
      var first = focusable[0];
      var last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    });
  }

  // Resolve the pending promise for `kind` ("ok" | "cancel"), restore focus
  // to the triggering element, hide the card. Safe to call once; a second
  // close is a no-op.
  function closeCard(kind) {
    if (!pending) return;
    var p = pending;
    pending = null;
    root.hidden = true;
    var lastFocus = p.lastFocus;
    if (lastFocus && typeof lastFocus.focus === "function" &&
        lastFocus.isConnected !== false) {
      lastFocus.focus();
    }
    if (p.mode === "prompt") {
      p.resolve(kind === "ok" ? String(inputEl.value).trim() : null);
    } else if (p.mode === "confirm") {
      p.resolve(kind === "ok");
    } else {
      p.resolve(undefined); // alert
    }
  }

  // Open the card for `mode` ("confirm" | "prompt" | "alert") with the
  // caller's options; returns the mode-specific promise.
  function openCard(mode, opts) {
    opts = opts || {};
    if (!root) buildCard();
    // A second call while one dialog is open (should not happen in the UI):
    // the earlier caller gets the cancel value and the new one takes over.
    if (pending) {
      var stale = pending;
      pending = null;
      root.hidden = true;
      stale.resolve(null);
    }
    var lastFocus = document.activeElement || null;

    titleEl.textContent = opts.title || "";
    messageEl.textContent = opts.message || "";
    messageEl.hidden = !opts.message;

    var isPrompt = mode === "prompt";
    inputEl.hidden = !isPrompt;
    inputEl.value = isPrompt && opts.value != null ? String(opts.value) : "";

    cancelBtn.hidden = mode === "alert";
    cancelBtn.textContent = opts.cancelLabel || "Cancel";
    okBtn.textContent = opts.okLabel || "OK";
    okBtn.classList.toggle("confirm-card-danger", !!opts.danger);

    root.hidden = false;

    // Focus the first control: the input for prompts, Cancel for confirms
    // (never auto-focus a destructive action), OK for alerts.
    if (isPrompt) {
      inputEl.focus();
      if (typeof inputEl.select === "function") inputEl.select();
    } else if (mode === "alert") {
      okBtn.focus();
    } else {
      cancelBtn.focus();
    }

    return new Promise(function (resolve) {
      pending = { mode: mode, resolve: resolve, lastFocus: lastFocus };
    });
  }

  function confirmCard(opts) {
    return openCard("confirm", opts);
  }

  function promptCard(opts) {
    return openCard("prompt", opts);
  }

  function alertCard(opts) {
    return openCard("alert", opts);
  }

  window.confirmCard = confirmCard;
  window.promptCard = promptCard;
  window.alertCard = alertCard;
})();
