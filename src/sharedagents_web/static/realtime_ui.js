/* realtime_ui.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
function realtimeServerUrl() {
  const meta = (document.querySelector('meta[name="sharedagents-realtime"]') || {})
    .content;
  const v = (meta || "").trim();
  if (v) return v.endsWith("/") ? v : v + "/";
  return `${location.protocol}//${location.hostname}:8123/`;
}

/** Wire the inline realtime engine once, if the script loaded. */
function loadScriptOnce(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector('script[src="' + src + '"]');
    if (existing) {
      if (existing.getAttribute("data-loaded") === "1") return resolve();
      existing.addEventListener("load", () => resolve());
      existing.addEventListener("error", () => reject(new Error("script " + src)));
      return;
    }
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => { s.setAttribute("data-loaded", "1"); resolve(); };
    s.onerror = () => reject(new Error("script " + src));
    document.body.appendChild(s);
  });
}

function ensureRealtimeScripts() {
  if (window.SharedagentsRealtime) return Promise.resolve();
  if (window._rtScriptsP) return window._rtScriptsP;
  const meta = (name) =>
    ((document.querySelector('meta[name="' + name + '"]') || {}).content || "").trim();
  const onnx = meta("sharedagents-rt-onnx");
  const wake = meta("sharedagents-rt-wake");
  const engine = meta("sharedagents-rt-engine");
  window._rtScriptsP = Promise.resolve()
    .then(() => (onnx ? loadScriptOnce(onnx) : null))
    .then(() => (wake ? loadScriptOnce(wake) : null))
    .then(() => (engine ? loadScriptOnce(engine) : null));
  return window._rtScriptsP;
}

function initRealtimeVoice() {
  if (!realtimeBtn) return;
  if (
    !navigator.mediaDevices ||
    !navigator.mediaDevices.getUserMedia
  ) {
    state.realtimeState = "unsupported";
    realtimeBtn.disabled = true;
    realtimeBtn.title = "Realtime voice not available";
    return;
  }
  realtimeBtn.disabled = false;
  if (!window.SharedagentsRealtime) {
    realtimeBtn.title = "Realtime voice";
    return;
  }
  window.SharedagentsRealtime.configure({
    url: realtimeServerUrl(),
    assetBase: BASE + "/static/", // wake-word models/wasm are served by the web origin
    wake: true, // client "Hey Jarvis" gate — matches the standalone page
    // and the server's client_wake. Critically this makes
    // the engine send wake_client_gated=true, so the server
    // does NOT also require the literal "jarvis" in the
    // transcript (which made continuous mode feel dead).
    hooks: realtimeHooks,
  });
  realtimeBtn.disabled = false;
  updateRealtimeButton();
}

// Live STT caption state for the chat-page "Hearing…" surface. Partials
// replace by item_id; finalized VAD segments accumulate across the utterance.
// `_rtPendingTurnText` survives `session.wake_reset` (server emits that BEFORE
// `response.created`) so we can still paint the user bubble before the answer.
let _rtCaptionCommitted = "";
let _rtCaptionPartial = "";
let _rtPendingTurnText = "";

function clearRealtimeListeningCaption() {
  _rtCaptionCommitted = "";
  _rtCaptionPartial = "";
  renderRealtimeListeningCaption();
}

function renderRealtimeListeningCaption() {
  if (!realtimeListeningCaption) return;
  const textEl = realtimeListeningCaption.querySelector(".rt-caption-text");
  const parts = [];
  if (_rtCaptionCommitted) parts.push(_rtCaptionCommitted);
  if (_rtCaptionPartial) parts.push(_rtCaptionPartial);
  const text = parts.join(" ").replace(/\s+/g, " ").trim();
  if (!text) {
    realtimeListeningCaption.hidden = true;
    if (textEl) textEl.textContent = "";
    realtimeListeningCaption.classList.remove("partial");
    return;
  }
  realtimeListeningCaption.hidden = false;
  realtimeListeningCaption.classList.toggle("partial", !!_rtCaptionPartial);
  if (textEl) textEl.textContent = text;
}

// Hooks: the engine calls these instead of touching its standalone DOM.
const realtimeHooks = {
  onConnected(live) {
    state.realtimeState = live ? "live" : "idle";
    if (live) {
      state.realtimeArmedThisSession = true;
      showRealtimeCoaching(); // raw-cascade path (canonical re-shows post-load)
    }
    if (!live) {
      state.realtimeAssistantEl = null;
      state.realtimeAssistantRaw = "";
      // Leave the bound gateway session on screen (it's a normal session now);
      // just clear the canonical flag so a later raw-cascade connect can use the
      // plain-transcript fallback again.
      state.realtimeCanonical = false;
      state.realtimeVoiceTurn = false;
      _rtPendingTurnText = "";
      setRealtimeVoiceStatus(""); // hide the phase indicator
      clearRealtimeListeningCaption();
      _unbindRealtimePttHold();
    }
    updateRealtimeButton();
    updateMicButton();
  },
  onModeChange(nextMode) {
    state.realtimeCaptureMode = nextMode === "ptt" ? "ptt" : "wake";
    if (state.realtimeState === "live" || state.realtimeState === "connecting") {
      if (state.realtimeCaptureMode === "ptt") {
        _bindRealtimePttHold();
        appendSystemMessage(
          "Wake word unavailable on this device — hold the Jarvis button to talk, release when done. Tap the button to stop.",
          false,
        );
      } else {
        _unbindRealtimePttHold();
      }
    }
    updateRealtimeButton();
  },
  // An unexpected connection loss must NOT hide the way back
  // in. Enter a distinct "reconnecting" state (still "busy" for visibility, so
  // the realtime/driver button stays on screen and acts as the Stop/cancel
  // control) while realtime.js auto-reconnects + replays buffered audio; return
  // to "live" once it's back. We never fall to "idle" on a loss — only a user
  // stop or a failed initial connect does that.
  onReconnecting(active) {
    if (active) {
      state.realtimeState = "reconnecting";
    } else if (state.realtimeState === "reconnecting") {
      state.realtimeState = "live";
    }
    updateRealtimeButton();
    updateMicButton();
  },
  onStatus(text, live) {
    // Voice phase ("say Hey Jarvis" / listening / transcribing / answering /
    // speaking) goes to its OWN surface so the canonical turn indicator
    // (#status-text "thinking…") can't clobber it. Ignore late statuses once
    // the session is idle so the pill doesn't reappear after stop.
    const active = live || state.realtimeState !== "idle";
    setRealtimeVoiceStatus(active ? text || "" : "", live);
  },
  onMicStreamStart(stream) {
    startAudioVisualizer(stream);
  },
  onMicStreamStop() {
    stopAudioVisualizer();
  },
  // Agent mode: bind the realtime server's gateway (Jarvis) session and render
  // it through the NORMAL session pipeline — loadSession() handles a
  // gateway-native hash exactly like a typed/monitor session (history + the
  // canonical live stream → reasoning, collapsed tool calls, stats). Max reuse.
  onGatewaySession(sid) {
    if (!sid) return;
    state.realtimeCanonical = true; // set BEFORE the load so plain hooks no-op
    // loadSession() clears the message list, so re-show the coaching afterwards
    // (the voice-status pill lives outside the list and is unaffected). On
    // failure, fall back to plain transcript bubbles rather than render nothing.
    Promise.resolve(loadSession(sid))
      .then(() => {
        if (state.realtimeState !== "idle") showRealtimeCoaching();
      })
      .catch((err) => {
        console.error("realtime: failed to bind gateway session", err);
        state.realtimeCanonical = false;
      });
  },
  // Live STT caption — always on (agent mode + raw cascade).
  onUserDelta(ev) {
    _rtCaptionPartial = (ev && ev.delta) || "";
    renderRealtimeListeningCaption();
  },
  onUserFinal(ev) {
    const transcript = (ev && ev.transcript) || "";
    _rtCaptionPartial = "";
    if (transcript) {
      _rtCaptionCommitted = (_rtCaptionCommitted
        ? `${_rtCaptionCommitted} ${transcript}`
        : transcript
      ).replace(/\s+/g, " ").trim();
      // Stash for the response that follows wake_reset (which clears the
      // visible caption but must not lose the question text).
      _rtPendingTurnText = _rtCaptionCommitted;
      renderRealtimeListeningCaption();
    } else if (!_rtCaptionCommitted) {
      clearRealtimeListeningCaption();
    } else {
      renderRealtimeListeningCaption();
    }
    // Raw-cascade: mirror each finalized VAD segment into the chat (no gateway
    // session). Agent mode waits for onAssistantStart so one bubble covers the
    // whole utterance (segments accumulate until "over").
    if (!state.realtimeCanonical && transcript) {
      appendMessage("user", transcript, true, null, null, null, new Date().toISOString());
    }
  },
  onWakeReset() {
    // Server re-arms wake BEFORE response.created. Hide the caption, but keep
    // `_rtPendingTurnText` so the user bubble can still be painted first.
    _rtCaptionPartial = "";
    _rtCaptionCommitted = "";
    renderRealtimeListeningCaption();
  },
  onAssistantStart() {
    // Paint user THEN assistant. Prefer the wake_reset-surviving stash —
    // `_rtCaptionCommitted` is usually already cleared by then.
    const heard = (_rtPendingTurnText || _rtCaptionCommitted || "").trim();
    _rtPendingTurnText = "";
    clearRealtimeListeningCaption();
    // Hold until turn_ended so late SSE cannot open a second user/assistant
    // bubble (the bug behind "messages duplicated in realtime mode").
    state.realtimeVoiceTurn = true;
    if (heard && state.realtimeCanonical) {
      if (state.currentSessionId) {
        state.pendingEcho = {
          sid: state.currentSessionId,
          text: heard,
          images: 0,
        };
      }
      // Skip if turn_started already painted the same prompt (race).
      const last = (state.msgCache || []).slice().reverse().find((m) => m && m.role === "user");
      const lastText = last ? String(last.content || "").replace(/\s+/g, " ").trim() : "";
      const heardNorm = heard.replace(/\s+/g, " ").trim();
      if (!lastText || lastText.toLowerCase() !== heardNorm.toLowerCase()) {
        appendMessage("user", heard, true, null, null, null, new Date().toISOString());
      }
    }
    state.realtimeAssistantRaw = "";
    state.realtimeAssistantEl = appendMessage(
      "assistant", "", true, null, null, null, new Date().toISOString(),
    );
  },
  onAssistantDelta(delta) {
    if (!state.realtimeAssistantEl) {
      state.realtimeAssistantRaw = "";
      state.realtimeAssistantEl = appendMessage(
        "assistant", "", true, null, null, null, new Date().toISOString(),
      );
    }
    state.realtimeAssistantRaw += delta || "";
    setAssistantContent(state.realtimeAssistantEl, state.realtimeAssistantRaw);
    scrollToBottom();
  },
  onAssistantDone() {
    // Keep realtimeVoiceTurn until turn_ended (or the grace timer) — clearing
    // it here lets turn_ended's ensureLiveAssistant open a duplicate bubble.
    state.realtimeAssistantEl = null;
    state.realtimeAssistantRaw = "";
    try { clearTimeout(state._rtVoiceTurnTimer); } catch (_) {}
    state._rtVoiceTurnTimer = setTimeout(() => {
      state.realtimeVoiceTurn = false;
      state._rtVoiceTurnTimer = null;
    }, 8000);
  },
  onAssistantCancelled() {
    state.realtimeAssistantEl = null;
    state.realtimeAssistantRaw = "";
    try { clearTimeout(state._rtVoiceTurnTimer); } catch (_) {}
    state._rtVoiceTurnTimer = null;
    state.realtimeVoiceTurn = false;
  },
  onError(ev) {
    const msg = (ev && ev.error && ev.error.message) || "Realtime voice error";
    appendSystemMessage(msg, true);
  },
};

function toggleRealtimeVoice() {
  if (_rtPttSuppressClick) {
    _rtPttSuppressClick = false;
    return;
  }
  if (
    state.realtimeState === "live" ||
    state.realtimeState === "connecting" ||
    state.realtimeState === "reconnecting"
  ) {
    stopRealtimeVoice();
    return;
  }
  if (!window.SharedagentsRealtime) {
    ensureRealtimeScripts()
      .then(() => {
        initRealtimeVoice();
        if (window.SharedagentsRealtime) startRealtimeVoice();
        else appendSystemMessage("Realtime voice failed to load", true);
      })
      .catch(() => {
        appendSystemMessage("Realtime voice failed to load", true);
      });
    return;
  }
  startRealtimeVoice();
}

function _realtimeErrorText(e) {
  if (!e) return "unknown error";
  if (typeof e === "string") return e;
  if (e.message) return e.message;
  if (e.type) return String(e.type);
  try { return JSON.stringify(e); } catch (_e) { return String(e); }
}

function startRealtimeVoice() {
  if (!window.SharedagentsRealtime) return;
  if (state.monitorReadOnly) return;
  // One mic at a time — don't start while push-to-talk is recording/processing,
  // and don't start while a typed/main-chat turn is still streaming.
  if (state.audioState === "recording" || state.audioState === "processing")
    return;
  if (state.isStreaming) return;
  state.realtimeState = "connecting";
  state.realtimeArmedThisSession = true;
  updateRealtimeButton();
  updateMicButton();
  setRealtimeVoiceStatus("Connecting…", true);
  window.SharedagentsRealtime.connect().catch((e) => {
    // Tear the engine fully down so a quick retry isn't blocked by a half-open ws.
    try {
      window.SharedagentsRealtime.disconnect();
    } catch (_) {}
    state.realtimeState = "idle";
    stopAudioVisualizer();
    setRealtimeVoiceStatus(""); // clear the "Connecting…" pill on failure
    updateRealtimeButton();
    updateMicButton();
    appendSystemMessage(
      "Realtime voice failed: " + _realtimeErrorText(e),
      true,
    );
  });
}

function stopRealtimeVoice() {
  _unbindRealtimePttHold();
  try {
    window.SharedagentsRealtime.disconnect();
  } catch (e) {}
  state.realtimeState = "idle";
  state.realtimeCaptureMode = "wake";
  state.realtimeAssistantEl = null;
  state.realtimeAssistantRaw = "";
  state.realtimeCanonical = false;
  state.realtimeVoiceTurn = false;
  try { clearTimeout(state._rtVoiceTurnTimer); } catch (_) {}
  state._rtVoiceTurnTimer = null;
  _rtPendingTurnText = "";
  stopAudioVisualizer();
  updateRealtimeButton();
  updateMicButton();
  setRealtimeVoiceStatus("");
  clearRealtimeListeningCaption();
}

// One-time-per-view coaching: the prod Jarvis server needs the end-word "over"
// to dispatch (the wake word starts the turn). Idempotent via DOM presence so it
// survives loadSession()'s clearMessages without duplicating.
function showRealtimeCoaching() {
  const exists = Array.prototype.some.call(
    document.querySelectorAll(".message"),
    (m) =>
      /hey .?jarvis/i.test(m.textContent || "") &&
      /over/i.test(m.textContent || ""),
  );
  if (!exists) {
    appendSystemMessage(
      "Realtime voice on. Say “Hey Jarvis”, your request, then “over” when you’re done.",
      false,
    );
  }
}

// Render the voice-phase indicator. Maps the engine's setStatus() text to a
// data-phase for styling; hides the surface when empty.
function setRealtimeVoiceStatus(text, live) {
  if (!realtimeVoiceStatus) return;
  const t = text || "";
  realtimeVoiceStatus.textContent = t;
  realtimeVoiceStatus.hidden = !t;
  realtimeVoiceStatus.classList.toggle("live", !!live);
  const phase = /hey jarvis|say /i.test(t)
    ? "waiting"
    : /listening|hearing|connecting/i.test(t)
      ? "listening"
      : /transcrib/i.test(t)
        ? "transcribing"
        : /answer|speaking|🔊/i.test(t)
          ? "answering"
          : "";
  realtimeVoiceStatus.dataset.phase = phase;
}

/** Realtime voice stays on the composer for every chat — fresh or ongoing.
 *  Hiding it after the first text turn made the control look removed.
 *  The entry/cancel control must remain reachable during
 *  connect, live, and reconnect; never hiding it satisfies that. */
function updateRealtimeVisibility() {
  if (!realtimeBtn) return;
  realtimeBtn.hidden = false;
}

let _rtPttHeld = false;
let _rtPttBound = false;
let _rtPttSuppressClick = false;
let _rtPttHoldTimer = null;

function _realtimePttDown(e) {
  if (state.realtimeCaptureMode !== "ptt" || state.realtimeState !== "live")
    return;
  if (!window.SharedagentsRealtime || typeof window.SharedagentsRealtime.beginPushToTalk !== "function")
    return;
  // Delay engage so a short tap still acts as Stop (click).
  if (_rtPttHoldTimer) clearTimeout(_rtPttHoldTimer);
  _rtPttHoldTimer = setTimeout(() => {
    _rtPttHoldTimer = null;
    _rtPttHeld = true;
    try {
      window.SharedagentsRealtime.beginPushToTalk();
    } catch (_) {}
  }, 280);
}

function _realtimePttUp(e) {
  if (_rtPttHoldTimer) {
    clearTimeout(_rtPttHoldTimer);
    _rtPttHoldTimer = null;
    // Short tap — do not suppress; toggleRealtimeVoice will Stop.
    return;
  }
  if (!_rtPttHeld) return;
  if (e) e.preventDefault();
  _rtPttHeld = false;
  _rtPttSuppressClick = true; // ignore the synthetic click that follows a hold
  try {
    if (window.SharedagentsRealtime && typeof window.SharedagentsRealtime.endPushToTalk === "function")
      window.SharedagentsRealtime.endPushToTalk();
  } catch (_) {}
}

function _bindRealtimePttHold() {
  if (!realtimeBtn || _rtPttBound) return;
  realtimeBtn.addEventListener("pointerdown", _realtimePttDown);
  realtimeBtn.addEventListener("pointerup", _realtimePttUp);
  realtimeBtn.addEventListener("pointercancel", _realtimePttUp);
  realtimeBtn.addEventListener("pointerleave", _realtimePttUp);
  _rtPttBound = true;
}

function _unbindRealtimePttHold() {
  if (_rtPttHoldTimer) {
    clearTimeout(_rtPttHoldTimer);
    _rtPttHoldTimer = null;
  }
  if (!realtimeBtn || !_rtPttBound) return;
  realtimeBtn.removeEventListener("pointerdown", _realtimePttDown);
  realtimeBtn.removeEventListener("pointerup", _realtimePttUp);
  realtimeBtn.removeEventListener("pointercancel", _realtimePttUp);
  realtimeBtn.removeEventListener("pointerleave", _realtimePttUp);
  _rtPttBound = false;
  if (_rtPttHeld) {
    _rtPttHeld = false;
    try {
      if (window.SharedagentsRealtime && typeof window.SharedagentsRealtime.endPushToTalk === "function")
        window.SharedagentsRealtime.endPushToTalk();
    } catch (_) {}
  }
}

function updateRealtimeButton() {
  if (!realtimeBtn) return;
  updateRealtimeVisibility();
  realtimeBtn.classList.remove("connecting", "live");
  const icon = realtimeBtn.querySelector(".realtime-icon");
  const stop = realtimeBtn.querySelector(".realtime-stop-icon");
  const showStop = (on) => {
    if (icon) icon.style.display = on ? "none" : "";
    if (stop) stop.style.display = on ? "" : "none";
  };
  if (state.monitorReadOnly) {
    realtimeBtn.disabled = true;
    realtimeBtn.title = "Read-only subagent session";
    showStop(false);
    return;
  }
  switch (state.realtimeState) {
    case "connecting":
      realtimeBtn.classList.add("connecting");
      realtimeBtn.title = "Connecting…";
      realtimeBtn.disabled = false;
      showStop(true);
      break;
    case "live":
      realtimeBtn.classList.add("live");
      if (state.realtimeCaptureMode === "ptt") {
        realtimeBtn.title = "Hold to talk · tap to stop";
        _bindRealtimePttHold();
      } else {
        realtimeBtn.title = "Stop realtime voice";
        _unbindRealtimePttHold();
      }
      realtimeBtn.disabled = false;
      showStop(true);
      break;
    case "reconnecting":
      // Connection dropped; auto-reconnecting. Keep the control visible and
      // enabled so it doubles as the cancel button.
      realtimeBtn.classList.add("connecting");
      realtimeBtn.title = "Reconnecting… (click to stop)";
      realtimeBtn.disabled = false;
      showStop(true);
      break;
    case "unsupported":
      realtimeBtn.disabled = true;
      realtimeBtn.title = "Realtime voice not available";
      showStop(false);
      break;
    default: // idle
      _unbindRealtimePttHold();
      // Don't start realtime while push-to-talk owns the mic or a turn streams.
      realtimeBtn.disabled =
        state.audioState === "recording" ||
        state.audioState === "processing" ||
        !!state.isStreaming;
      realtimeBtn.title = state.realtimeArmedThisSession
        ? "Continue realtime voice"
        : "Realtime voice — talk live";
      showStop(false);
      break;
  }
}

// ══════════════════════════════════════════════════════════════════════
// Reasoning & tool-call rendering (Hermes-style)
//
// The gateway folds an agent's intermediate reasoning + tool activity into
// the assistant content stream as a leading <think>…</think> block (see
// sharedagents_agent/openai_endpoint.format_session_updates_for_openai):
//   <think>
//   …free-form thought text…
//   🔧 tool_name(arg=…, arg2=…)
//   ← OK            (or "← failed")
//   📋 plan:
//   • step one
//   </think>
//   the final answer…
// We parse that block out and render it as a collapsible disclosure with
// structured tool-call rows, instead of leaking the raw markers into chat.
// ══════════════════════════════════════════════════════════════════════

// Split raw assistant text into { segments, answer, open }. `segments` is null
// when there is no reasoning block. `open` is true while the block is still
// streaming (no closing </think> yet) so it stays expanded mid-thought.
