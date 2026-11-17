/* notifications.js — system notifications + sound when a turn finishes.

   Phase 1 of plan/mobile-notifications.md. Target platform is Chrome on
   Android: the operator has sharedagents open on their phone, switches to another
   app, and wants to be told (visibly + audibly) when the agent is done.

   THREE THINGS IN HERE ARE LOAD-BEARING — see the plan before changing them:

   1. Notifications are displayed ONLY through
      ServiceWorkerRegistration.showNotification(). `new Notification()` throws
      `Illegal constructor` on Android Chrome — the constructor is not and will
      never be implemented there, because it assumes the page outlives the
      notification. It works on desktop, so this bug is invisible to every
      desktop test. There is exactly one display path, no platform sniff.
      (plan §3.3)

   2. The app asks BEFORE the browser does. `Notification.requestPermission()`
      is one-shot and unrecoverable — a dismissal sets permission to "denied"
      forever, fixable only in Chrome's site settings. So the native dialog is
      only ever opened from inside a tap on our own in-page "Enable" card.
      (plan §6.0)

   3. The AudioContext is created and resumed SYNCHRONOUSLY, before the first
      await in the gesture handler. A user gesture only covers the synchronous
      part of its handler; awaiting permission first consumes it and every
      notification is then silent forever. (plan §4.2)

   The chime is synthesized with an OscillatorNode rather than decoded from a
   base64 data: URI as the plan sketched — the app's CSP is `default-src 'self'`
   with no media-src, so a data: audio source would be blocked outright.

   Delivery ceiling (plan §3.5): this rides the SSE stream, so it only fires
   while the page is running. Android freezes backgrounded tabs after ~5min,
   which drops EventSource; turns longer than that are notified late (when the
   stream reconnects and replays) or not at all. Web Push (Phase 2) is the fix.

*/

(function () {
  "use strict";

  const LS_KEY = "sharedagents.notifications";
  const SNOOZE_MS = 7 * 24 * 60 * 60 * 1000; // "Not now" → re-ask in a week
  const COOLDOWN_MS = 30 * 1000; // floor between two notifications
  const BODY_MAX = 120; // chars of answer text in the body

  const DEFAULTS = {
    enabled: false,
    sound: true,
    // ask.state: unasked | later | never | granted-flow | denied
    ask: { state: "unasked", asked_at: 0 },
    triggers: {
      turn_complete: true,
      turn_error: true,
      question: true,
    },
  };

  let prefs = JSON.parse(JSON.stringify(DEFAULTS));
  let audioCtx = null;
  let lastNotifyAt = 0;
  let promptShownThisSession = false;

  // ── Capability gate (plan §6.3) ──────────────────────────────────────
  // False ⇒ the feature is INVISIBLE, not broken: no card, no dead button,
  // no console noise. Covers iOS Chrome (no in-tab Notification API) and any
  // non-secure origin (LAN http://<lan-ip>:8110 — see plan §3.4).
  function notificationsSupported() {
    return (
      typeof window !== "undefined" &&
      "Notification" in window &&
      "serviceWorker" in navigator &&
      window.isSecureContext === true
    );
  }

  function permission() {
    try {
      return notificationsSupported() ? Notification.permission : "unsupported";
    } catch (_e) {
      return "unsupported";
    }
  }

  // ── Preferences (plan §6.2) ──────────────────────────────────────────
  // localStorage is a cache of INTENT. Notification.permission is the source
  // of truth and can change out from under us (user resets site settings,
  // clears Chrome data), so every load reconciles rather than trusting the
  // stored flag and throwing at notify() time.
  function loadNotificationPrefs() {
    let stored = null;
    try {
      stored = JSON.parse(localStorage.getItem(LS_KEY) || "null");
    } catch (_e) {
      stored = null; // storage denied or corrupt — fall back to defaults
    }
    prefs = {
      enabled: !!(stored && stored.enabled),
      sound: stored && "sound" in stored ? !!stored.sound : true,
      ask: {
        state: (stored && stored.ask && stored.ask.state) || "unasked",
        asked_at: (stored && stored.ask && stored.ask.asked_at) || 0,
      },
      triggers: Object.assign(
        {},
        DEFAULTS.triggers,
        (stored && stored.triggers) || {},
      ),
    };
    const perm = permission();
    if (prefs.enabled && perm !== "granted") prefs.enabled = false;
    if (perm === "denied") prefs.ask.state = "denied";
    // The browser now says "granted" but our flag is still off. That means the
    // operator allowed the site from Chrome's OWN menu (the only recovery path
    // once a site is blocked / auto-blocked — requestPermission() then returns
    // "denied" with no prompt and can never re-ask). A browser grant is
    // explicit consent, so adopt it — from ANY prior state except an in-app
    // "Never". This must fire from `denied` too, which is the whole point:
    // Adopting only from 'attempted' left a blocked user stuck even after
    // they fixed it in Chrome settings.
    // Adopt from any state EXCEPT the two explicit in-app declines — a "Never"
    // or a "Not now" must not be silently upgraded into enabled (that would not
    // be consent). Crucially this now includes "denied": a user who was blocked
    // and then allowed the site in Chrome settings is enabled on return.
    if (
      perm === "granted" &&
      !prefs.enabled &&
      prefs.ask.state !== "never" &&
      prefs.ask.state !== "later"
    ) {
      prefs.enabled = true;
      prefs.ask.state = "granted-flow";
    }
    return prefs;
  }

  /** Permission can be changed OUTSIDE the page (Chrome's address-bar menu),
   *  which fires no event. Re-reconcile whenever the tab comes back to front —
   *  that is exactly the moment the operator returns from those settings. */
  function watchForExternalPermissionChange() {
    if (typeof document.addEventListener !== "function") return;
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState !== "visible") return;
      const was = prefs.enabled;
      loadNotificationPrefs();
      if (prefs.enabled !== was) saveNotificationPrefs();
      else syncSettingsUi();
    });
  }

  function saveNotificationPrefs(next) {
    if (next) prefs = next;
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(prefs));
    } catch (_e) {
      /* storage denied — preferences are per-session only */
    }
    syncSettingsUi();
    return prefs;
  }

  // ── Sound ────────────────────────────────────────────────────────────
  // Two short descending tones. Synthesized, so nothing is fetched and the
  // CSP is irrelevant. Created lazily but ALWAYS from inside a user gesture
  // (see enableFromGesture) — a context created outside one starts suspended
  // and stays that way on mobile.
  function ensureAudioContext() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    if (!audioCtx) {
      try {
        audioCtx = new Ctx();
      } catch (_e) {
        return null;
      }
    }
    if (audioCtx.state === "suspended") {
      try {
        audioCtx.resume();
      } catch (_e) {
        /* resume is best-effort; a later gesture may unlock it */
      }
    }
    return audioCtx;
  }

  function playNotificationSound() {
    if (!prefs.sound) return;
    const ctx = audioCtx;
    // Deliberately does NOT create a context here: outside a gesture it would
    // be born suspended and never play, while masking the real problem.
    if (!ctx || ctx.state !== "running") return;
    try {
      const now = ctx.currentTime;
      [
        [880, 0],
        [660, 0.14],
      ].forEach(function (pair) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "sine";
        osc.frequency.value = pair[0];
        const t0 = now + pair[1];
        gain.gain.setValueAtTime(0.0001, t0);
        gain.gain.exponentialRampToValueAtTime(0.18, t0 + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.13);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(t0);
        osc.stop(t0 + 0.15);
      });
    } catch (_e) {
      /* audio is a bonus; never let it break the notification */
    }
  }

  // ── Display (plan §3.3) ──────────────────────────────────────────────
  // The ONLY place a notification is shown. Never `new Notification(...)`.
  // Ungated on purpose: the test paths need to bypass the foreground/cooldown
  // rules. `notify()` below owns the policy.
  async function deliver(title, body, tag) {
    try {
      const reg = await navigator.serviceWorker.ready;
      await reg.showNotification(title, {
        body: body || "",
        tag: tag || "sharedagents",
        renotify: true,
        icon: BASE + "/static/icon-192.png",
        badge: BASE + "/static/badge-72.png",
        data: { url: location.href },
      });
      playNotificationSound();
      lastResult = new Date().toLocaleTimeString() + " — delivered";
      renderDiagnostics();
      return true;
    } catch (e) {
      // Surface the real reason on the device instead of failing silently.
      lastResult = "FAILED — " + (e && e.message ? e.message : String(e));
      renderDiagnostics();
      console.debug("notification failed:", e);
      return false;
    }
  }

  async function notify(title, body, tag) {
    if (!prefs.enabled || permission() !== "granted") {
      lastResult = "suppressed — notifications are off";
      renderDiagnostics();
      return false;
    }
    // Foreground check: hasFocus() alone misreports on Android often enough to
    // both double-alert and silently skip, so require both signals.
    if (document.hasFocus() && document.visibilityState === "visible") {
      lastResult = new Date().toLocaleTimeString() + " — suppressed (tab in front)";
      renderDiagnostics();
      return false;
    }
    const now = Date.now();
    if (now - lastNotifyAt < COOLDOWN_MS) {
      lastResult = new Date().toLocaleTimeString() + " — suppressed (30s cooldown)";
      renderDiagnostics();
      return false;
    }
    lastNotifyAt = now;
    return deliver(title, body, tag);
  }

  // ── The enable gesture (plan §4.2) ───────────────────────────────────
  // ORDER IS NORMATIVE. Audio first and synchronous; permission after.
  /** Ask for permission across BOTH shapes of the API, and treat
   *  `Notification.permission` as the authority rather than the return value.
   *
   *  `Notification.requestPermission()` has two forms: the modern
   *  promise-returning one, and a legacy callback form that returns
   *  `undefined`. Some Android builds still implement only the callback. The
   *  previous code did `perm = await Notification.requestPermission()`, so on
   *  those builds `perm` became `undefined` — treated as "not granted" even
   *  when the user had just tapped Allow and `Notification.permission` said
   *  "granted". Passing a callback AND handling the promise covers both; a
   *  guard makes whichever fires first win.
   *
   *  Resolves with the browser's own permission string, never the raw return.
   */
  function requestPermissionCompat() {
    return new Promise(function (resolve) {
      let settled = false;
      let timer = null;
      const done = function (hint) {
        if (settled) return;
        settled = true;
        // Always release the safety timer — whichever path answered first,
        // leaving it armed leaks a pending timer per enable attempt.
        if (timer !== null) {
          clearTimeout(timer);
          timer = null;
        }
        // permission() re-reads Notification.permission — the source of truth.
        const authoritative = permission();
        resolve(authoritative === "default" && hint ? hint : authoritative);
      };
      let ret;
      try {
        ret = Notification.requestPermission(done);
      } catch (_e) {
        // Some engines reject a callback argument outright; retry bare.
        try {
          ret = Notification.requestPermission();
        } catch (e2) {
          return done();
        }
      }
      if (ret && typeof ret.then === "function") {
        ret.then(done, function () { done(); });
      }
      // If the prompt is suppressed and NEITHER path ever settles, don't leave
      // the button wedged forever — fall back to whatever the browser reports.
      if (!settled) timer = setTimeout(function () { done(); }, 60000);
    });
  }

  async function enableFromGesture() {
    if (!notificationsSupported()) return "unsupported";

    // 1+2. Synchronous, still inside the gesture: unlock audio, play the
    //      confirmation chime. Anything after the first `await` below is no
    //      longer user-activated.
    //      `resume()` is ASYNC — checking ctx.state right after calling it
    //      still reads "suspended" on Android, which silently skipped the
    //      chime. The gesture unlocks the CONTEXT; playing a moment later off
    //      the resume promise is fine and is what actually works on a phone.
    const ctx = ensureAudioContext();
    if (ctx) {
      prefs.sound = true;
      if (ctx.state === "running") {
        playNotificationSoundNow();
      } else if (typeof ctx.resume === "function") {
        try {
          Promise.resolve(ctx.resume()).then(playNotificationSoundNow, function () {});
        } catch (_e) {
          /* chime is a bonus */
        }
      }
    }

    // 3. Now the native dialog — opened from a tap the user already agreed to.
    const before = permission();
    let returned = before;
    try {
      if (before === "default") returned = await requestPermissionCompat();
      recordEnableAttempt(before, returned, null);
    } catch (e) {
      recordEnableAttempt(before, permission(), e);
    }

    // AUTHORITATIVE state = the persistent Notification.permission, NOT the
    // value requestPermission() returned. Verified via CDP + a real phone:
    // Chrome's quieter-notifications can RETURN "denied" while leaving the
    // persistent permission at "default" (recoverable, still promptable). The
    // old code wrote ask.state="denied" off that transient "denied" and locked
    // the operator out of ever re-asking, for a state that was never actually
    // blocked. Decide from the persistent value; the return is only a message.
    const perm = permission();
    prefs.ask.asked_at = Date.now();
    if (perm === "granted") {
      // 4. Await the worker so the first real notification can't race
      //    registration (display goes through the registration, not the page).
      try {
        await navigator.serviceWorker.ready;
      } catch (_e) {
        /* registration is retried by registerServiceWorker() */
      }
      prefs.enabled = true;
      prefs.ask.state = "granted-flow";
    } else {
      prefs.enabled = false;
      // Only a PERSISTENT "denied" is a dead end. "default" (dismissed or
      // quieter-suppressed) stays "attempted": recoverable, still re-askable,
      // and adopted the moment the browser reports "granted" (see
      // loadNotificationPrefs + the visibilitychange watcher).
      prefs.ask.state = perm === "denied" ? "denied" : "attempted";
    }
    saveNotificationPrefs();
    hideOptInCard();
    return perm;
  }

  // Bypasses the cooldown/foreground gates — this is the confirmation chime
  // for a tap the user just made, not an event notification.
  function playNotificationSoundNow() {
    const saved = lastNotifyAt;
    lastNotifyAt = 0;
    playNotificationSound();
    lastNotifyAt = saved;
  }

  // ── On-device diagnostics ────────────────────────────────────────────
  // A phone cannot be remote-debugged casually, and "it doesn't work" has at
  // least six distinct causes here (permission, secure context, OS-level
  // Chrome block, SW not ready, feature off, suppressed-because-foreground).
  // This renders the actual state in plain language so the failing one is
  // readable on the device in five seconds.
  let lastResult = "never";
  // `Notification.permission === "default"` CANNOT distinguish "never asked"
  // from "asked, and Chrome returned default" — which on Android happens when
  // the sheet is dismissed (back / tap-outside leaves it "default", NOT
  // "denied") or when quieter-permissions suppresses the sheet entirely and
  // shows only a bell icon in the omnibox. Reporting all three as "not
  // requested yet" sent the operator hunting for a button they had already
  // pressed. Record what actually happened.
  let lastEnable = null; // { at, before, after, persistent, error }

  function recordEnableAttempt(before, after, error) {
    lastEnable = {
      at: new Date().toLocaleTimeString(),
      before: before,
      after: after,
      // The persistent, authoritative state right after the request — this is
      // what distinguishes a real block from a transient quieter-notifications
      // "denied" that left the permission recoverable at "default".
      persistent: permission(),
      error: error ? (error.message || String(error)) : null,
    };
    renderDiagnostics();
  }

  function enableAttemptLine() {
    if (!lastEnable) return "never tapped in this tab";
    if (lastEnable.error) return lastEnable.at + " → ERROR: " + lastEnable.error;
    if (lastEnable.persistent === "granted") return lastEnable.at + " → granted";
    // Persistently blocked — the only true dead end. requestPermission() will
    // never prompt again; recovery is a manual allow in Chrome's own settings.
    if (lastEnable.persistent === "denied") {
      return lastEnable.at +
        " → blocked in Chrome (no prompt can appear). Fix: tap the icon left " +
        "of the address bar → Permissions/Notifications → Allow.";
    }
    // Persistent state is "default" but the request returned "denied": Chrome's
    // quieter notifications suppressed the prompt. NOT actually blocked — the
    // request can be retried, and a bell/blocked icon is usually in the omnibox.
    if (lastEnable.after === "denied") {
      return lastEnable.at +
        " → Chrome suppressed the prompt (quieter notifications) — you were " +
        "not asked. Tap the notification icon in the address bar and choose " +
        "Allow, or just tap Enable again.";
    }
    return lastEnable.at +
      " → Chrome returned '" + lastEnable.after + "' (prompt dismissed, or " +
      "suppressed by Chrome). Use the address-bar icon → Permissions → " +
      "Notifications → Allow.";
  }

  function diagnosticLines() {
    const perm = permission();
    const out = [];
    out.push(["Address", location.origin]);
    out.push(["Secure context", window.isSecureContext ? "yes" : "NO — notifications cannot work on http://"]);
    out.push(["Notification API", "Notification" in window ? "yes" : "NO — not available in this browser"]);
    out.push(["Service worker", "serviceWorker" in navigator
      ? (navigator.serviceWorker.controller ? "active" : "registered, not yet controlling — reload once")
      : "NO"]);
    out.push(["Browser permission", perm === "granted" ? "granted"
      : perm === "denied" ? "BLOCKED — address-bar icon → Permissions → Notifications → Allow"
      : "not granted"]);
    out.push(["Enable attempt", enableAttemptLine()]);
    out.push(["Alerts", prefs.enabled ? "on" : "OFF"]);
    out.push(["Last alert", lastResult]);
    if (perm === "granted" && prefs.enabled) {
      out.push(["Note", "alerts are suppressed while this tab is in front — " +
        "use the 5-second test below"]);
    }
    return out;
  }

  function renderDiagnostics() {
    // Never let a rendering problem take down delivery — this is called from
    // inside deliver()'s try block.
    try {
      paintDiagnostics();
    } catch (_e) {
      /* diagnostics are advisory */
    }
  }

  function paintDiagnostics() {
    const host = document.getElementById("notification-diagnostic");
    if (!host) return;
    host.textContent = "";
    diagnosticLines().forEach(function (pair) {
      const row = document.createElement("div");
      row.className = "notif-diag-row";
      const k = document.createElement("span");
      k.className = "notif-diag-k";
      k.textContent = pair[0];
      const v = document.createElement("span");
      v.className = "notif-diag-v";
      v.textContent = pair[1];
      if (/^(NO|BLOCKED)\b|ERROR:|not granted|Chrome returned/.test(pair[1])) {
        v.classList.add("notif-diag-bad");
      }
      row.appendChild(k);
      row.appendChild(v);
      host.appendChild(row);
    });
  }

  /** Fire a notification after a delay so the operator can background the app
   *  first — the only way to observe real backgrounded delivery on a phone. */
  function scheduleBackgroundTest(seconds) {
    ensureAudioContext();
    const host = document.getElementById("notification-status");
    let left = seconds;
    const tickMsg = function () {
      if (host) {
        host.textContent = left > 0
          ? "Switch to another app now — alert in " + left + "s"
          : "Sent. Check your notification tray.";
      }
    };
    tickMsg();
    const iv = setInterval(function () {
      left -= 1;
      tickMsg();
      if (left <= 0) {
        clearInterval(iv);
        deliver("sharedagents", "Backgrounded delivery test — if you can see this, " +
          "notifications work.", "sharedagents-bgtest");
      }
    }, 1000);
    return iv;
  }

  async function sendTestNotification() {
    ensureAudioContext();
    if (permission() !== "granted") {
      lastResult = "cannot test — permission is '" + permission() + "'";
      renderDiagnostics();
      return false;
    }
    return deliver(
      "sharedagents",
      "Test notification — this is what a finished turn looks like.",
      "sharedagents-test",
    );
  }

  // ── Opt-in card (plan §6.0) ──────────────────────────────────────────
  function shouldShowOptIn() {
    if (!notificationsSupported()) return false;
    if (promptShownThisSession) return false;
    if (permission() !== "default") return false; // granted or denied ⇒ nothing to ask
    const st = prefs.ask.state;
    // "attempted" ⇒ they already engaged with the card and tapped Enable;
    // re-showing it would just offer a button that Chrome will ignore.
    if (st === "never" || st === "denied" || st === "granted-flow" ||
        st === "attempted") {
      return false;
    }
    if (st === "later" && Date.now() - (prefs.ask.asked_at || 0) < SNOOZE_MS) {
      return false;
    }
    return true;
  }

  function optInCard() {
    return document.getElementById("notif-optin");
  }

  function maybeShowOptInPrompt() {
    if (!shouldShowOptIn()) return false;
    const card = optInCard();
    if (!card) return false;
    promptShownThisSession = true;
    card.hidden = false;
    return true;
  }

  function hideOptInCard() {
    const card = optInCard();
    if (card) card.hidden = true;
  }

  function answerOptIn(answer) {
    if (answer === "never") {
      prefs.ask.state = "never";
      prefs.ask.asked_at = Date.now();
      saveNotificationPrefs();
      hideOptInCard();
      return Promise.resolve("never");
    }
    if (answer === "later") {
      prefs.ask.state = "later";
      prefs.ask.asked_at = Date.now();
      saveNotificationPrefs();
      hideOptInCard();
      return Promise.resolve("later");
    }
    return enableFromGesture();
  }

  // ── Settings surface (plan §6.1) ─────────────────────────────────────
  function syncSettingsUi() {
    applySettingsUi();
    renderDiagnostics();
    // The backgrounded-delivery test only makes sense once alerts can fire.
    const bg = document.getElementById("notif-bg-test");
    if (bg) bg.hidden = !(permission() === "granted" && prefs.enabled);
  }

  function applySettingsUi() {
    const group = document.getElementById("notif-setting-group");
    const btn = document.getElementById("enable-notifications");
    const status = document.getElementById("notification-status");
    const triggers = document.getElementById("notification-triggers");
    if (!group || !btn || !status) return;

    if (!("Notification" in window) || !("serviceWorker" in navigator)) {
      group.hidden = true;
      return;
    }
    group.hidden = false;

    const perm = permission();
    if (!window.isSecureContext) {
      btn.disabled = true;
      status.textContent =
        "Requires HTTPS or localhost. Current origin: " + location.origin;
      if (triggers) triggers.hidden = true;
      return;
    }
    btn.disabled = false;
    if (perm === "denied") {
      btn.disabled = true;
      btn.textContent = "Enable notifications + sound";
      status.textContent =
        "Blocked in browser settings. Chrome → ⋮ → Site settings → " +
        "Notifications → Allow.";
      if (triggers) triggers.hidden = true;
      return;
    }
    if (perm === "granted" && prefs.enabled) {
      btn.textContent = "Send test notification";
      status.textContent = "Notifications on";
      if (triggers) triggers.hidden = false;
    } else if (perm === "granted") {
      btn.textContent = "Enable notifications + sound";
      status.textContent = "Permission granted — currently off";
      if (triggers) triggers.hidden = true;
    } else if (lastEnable && lastEnable.after !== "granted" && !lastEnable.error) {
      // Tapped, but Chrome did not grant. The single most useful thing we can
      // say is the recovery path, because tapping Enable again very often
      // does nothing (Chrome will not re-prompt after a dismissal).
      btn.textContent = "Enable notifications + sound";
      status.textContent =
        "Chrome didn't grant permission. Tap the icon left of the address " +
        "bar → Permissions → Notifications → Allow, then reopen Settings.";
      if (triggers) triggers.hidden = true;
    } else {
      btn.textContent = "Enable notifications + sound";
      status.textContent = "";
      if (triggers) triggers.hidden = true;
    }

    Object.keys(prefs.triggers).forEach(function (key) {
      const el = document.getElementById("notif-" + key.replace(/_/g, "-"));
      if (el) el.checked = !!prefs.triggers[key];
    });
  }

  function onSettingsButton() {
    if (permission() === "granted" && prefs.enabled) return sendTestNotification();
    return enableFromGesture();
  }

  function setTrigger(key, on) {
    prefs.triggers[key] = !!on;
    saveNotificationPrefs();
  }

  // ── Event → notification (plan §5) ───────────────────────────────────
  function sessionLabel() {
    const sid = state.currentSessionId;
    const sess = (sid && state.sessions && state.sessions[sid]) || null;
    const title = sess && sess.title;
    if (title && typeof isPlaceholderTitle === "function" &&
        !isPlaceholderTitle(title)) {
      return title;
    }
    return title || "";
  }

  function tidySnippet(raw) {
    raw = String(raw || "");
    raw = raw.replace(/<think>[\s\S]*?<\/think>/gi, "");
    raw = raw.replace(/```[\s\S]*?```/g, " (code) ");
    raw = raw.replace(/[#*`_>|]/g, "").replace(/\s+/g, " ").trim();
    if (!raw) return "";
    return raw.length > BODY_MAX ? raw.slice(0, BODY_MAX - 1) + "…" : raw;
  }

  /** Last assistant text, read SYNCHRONOUSLY (no DOM scraping, no deferral).
   *  Prefers the live turn's accumulated raw text — the streaming deltas have
   *  already written it by the time turn_ended arrives — and falls back to the
   *  virtual-scroller cache for a replayed/reconnected turn. */
  function lastAnswerSnippet() {
    const live = state && state.liveTurn;
    if (live && live.el && live.el.dataset && live.el.dataset.raw) {
      const s = tidySnippet(live.el.dataset.raw);
      if (s) return s;
    }
    const cache = (state && state.msgCache) || [];
    for (let i = cache.length - 1; i >= 0; i--) {
      if (cache[i] && cache[i].role === "assistant") {
        return tidySnippet(cache[i].content);
      }
    }
    return "";
  }

  function onTurnEnded(event) {
    // An abort is the operator's own action — they are right there; never
    // notify for it. `_instantAbort` is the local Stop-button variant, which
    // is still set when this fires from the top of the turn_ended case.
    if (event && event.aborted) return;
    if (state && state._instantAbort) return;
    // MUST stay synchronous. This originally deferred via setTimeout(…, 0) to
    // read the settled transcript — which put the whole feature behind the one
    // primitive Android throttles: a backgrounded tab's timers drop to 1/sec,
    // then 1/MINUTE under intensive throttling, and stop entirely once the tab
    // is frozen (~5min). The notification is ONLY wanted while backgrounded,
    // so that deferral broke exactly the case it exists for. The answer text
    // is read from the live turn element instead, which is already populated
    // by the streaming deltas at this point.
    emitTurnEnded(event);
  }

  function emitTurnEnded(event) {
    const agent =
      (event && event.agent_name) || state.liveTurnAgent || state.currentAgent ||
      "sharedagents";
    const label = sessionLabel();
    const sid = state.currentSessionId || "";
    // tag keyed on the session so a replayed turn_ended (SSE reconnect after
    // the phone unfreezes the tab) collapses onto the same tray entry instead
    // of stacking a duplicate.
    const tag = "sharedagents-turn-" + sid;

    if (event && event.error) {
      if (!prefs.triggers.turn_error) return;
      notify(
        agent + " hit an error",
        String(event.error).slice(0, BODY_MAX) + (label ? " — " + label : ""),
        tag,
      );
      return;
    }
    if (!prefs.triggers.turn_complete) return;
    const snippet = lastAnswerSnippet();
    const body = [label, snippet].filter(Boolean).join(" — ") || "Turn complete.";
    notify(agent + " finished", body, tag);
  }

  function onQuestion(event) {
    if (!prefs.triggers.question) return;
    const label = sessionLabel();
    notify(
      "Question for you",
      (label ? label + " — " : "") + "the agent needs your answer to continue.",
      "sharedagents-ask-" + ((event && event.call_id) || state.currentSessionId || ""),
    );
  }

  /** Single entry point called from renderCanonicalEvent (streaming.js). */
  function notificationTrigger(kind, event) {
    try {
      if (kind === "turn_started") {
        maybeShowOptInPrompt();
      } else if (kind === "turn_ended") {
        onTurnEnded(event);
      } else if (kind === "question") {
        onQuestion(event);
      }
    } catch (e) {
      console.debug("notificationTrigger:", e);
    }
  }

  // ── Exports (classic script — shared global scope, like the rest) ─────
  window.notificationTrigger = notificationTrigger;
  window.Notifications = {
    loadNotificationPrefs,
    saveNotificationPrefs,
    watchForExternalPermissionChange,
    notificationsSupported,
    permission,
    notify,
    playNotificationSound,
    enableFromGesture,
    sendTestNotification,
    onSettingsButton,
    scheduleBackgroundTest,
    renderDiagnostics,
    diagnosticLines,
    maybeShowOptInPrompt,
    answerOptIn,
    setTrigger,
    syncSettingsUi,
    prefs: function () {
      return prefs;
    },
    _reset: function () {
      // test seam
      audioCtx = null;
      lastNotifyAt = 0;
      promptShownThisSession = false;
    },
  };
})();
