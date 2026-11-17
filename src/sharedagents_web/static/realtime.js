/* sharedagents realtime voice client.
 *
 * Mic → AudioWorklet → PCM16 mono 24 kHz → base64 `input_audio_buffer.append`.
 * `response.output_audio.delta` (PCM16 24 kHz) → scheduled AudioBuffer queue.
 * Repeating the local wake phrase sends `session.interrupt`; no other detected
 * speech can stop playback.
 *
 * The local wake engine gates every interactive microphone frame. If it cannot
 * load, capture fails closed and the explicit push-to-talk control is the only
 * degraded channel-open act; continuous interactive egress does not exist.
 */
"use strict";

const WIRE_RATE = 24000;
const CHUNK_SAMPLES = 960; // 40 ms
const CHUNK_MS = 1000 * CHUNK_SAMPLES / WIRE_RATE; // 40
// White-label: the product name (substituted into realtime.html by the
// realtime server from SHAREDAGENTS_WEB_TITLE). Embedded contexts have no meta
// tag and fall back to the historical default.
const SITE_TITLE = (document.querySelector('meta[name="sharedagents-site-title"]') || {}).content || "sharedagents";

const ui = {
  btn: document.getElementById("btn"),
  mode: document.getElementById("mode"),
  meeting: document.getElementById("meeting"),
  status: document.getElementById("status"),
  log: document.getElementById("log"),
  vu: document.querySelector("#vu > div"),
  metrics: document.getElementById("metrics"),
};

// ── Embed API ───────────────────────────────────────────────────────────────
// The standalone page (realtime.html) has all the #btn/#log/#vu/#status/#metrics
// elements, so `ui.*` is populated and the engine renders into them directly.
// When this same script is loaded into the MAIN chat (index.html) those elements
// don't exist (ui.* are null); the chat calls SharedagentsRealtime.configure({hooks,
// url, assetBase, wake}) to drive rendering through the chat's own widgets and to
// point the WebSocket at the realtime server (a different origin than the page).
let hooks = {};               // semantic render callbacks supplied by the embedder
let realtimeBaseUrl = "";     // realtime-server base (http/https URL or path); "" = page-relative
let realtimeAssetBase = "";   // base for wake-word model/wasm assets; "" = page dir
let embedWake = null;         // tri-state: null = default, true/false = force wake on/off

// Resolve the realtime server's WS + HTTP base. Standalone: same origin as the
// page. Embedded: derived from the injected realtime_url (may be cross-origin in
// dev, or a same-origin proxy path like "/sharedagentsrt/" in prod).
function realtimeBase() {
  if (realtimeBaseUrl) {
    try {
      const u = new URL(realtimeBaseUrl, location.href);
      // Accept either an http(s) base (the documented form) or a ws(s) one and
      // derive both schemes from it.
      const secure = u.protocol === "https:" || u.protocol === "wss:";
      const wsProto = secure ? "wss:" : "ws:";
      const httpProto = secure ? "https:" : "http:";
      const dir = u.pathname.endsWith("/") ? u.pathname : u.pathname + "/";
      return { ws: `${wsProto}//${u.host}${dir}v1/realtime`,
               http: `${httpProto}//${u.host}${dir}` };
    } catch (e) { /* fall through to page-relative */ }
  }
  const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
  const dir = pageDir();
  return { ws: `${wsProto}//${location.host}${dir}v1/realtime`,
           http: location.origin + dir };
}

let ws = null, ctx = null, micNode = null, micStream = null;
let micWorkletReady = false;
// Egress accumulator: raw mic samples accrete here until a full CHUNK_SAMPLES
// chunk is ready to leave the device. It is filled ONLY while the explicit
// channel state is OPEN — see onMicChunk.
let sendBuf = new Int16Array(CHUNK_SAMPLES), sendFill = 0;
// Pre-wake accumulator, SEPARATE from the egress accumulator above.
// Wake-idle samples fill THIS (feeding the local `preroll` ring), never sendBuf, so a
// pre-wake partial can never straddle the wake→engaged boundary into an egressed/buffered
// chunk. engage() also resets both fills as defense-in-depth.
let prerollBuf = new Int16Array(CHUNK_SAMPLES), prerollFill = 0;
let playHead = 0, liveSources = [];
let currentBot = null;

// Capture gating. "wake" is hands-free local detection; "ptt" is the explicit
// fail-closed degraded mode. In both cases only streaming===true may egress.
let mode = "wake";
let streaming = false;          // wake mode: currently engaged & sending
let responding = false;
// Opening is a server-acknowledged handshake. A local wake can fire while the
// socket is down, but egress remains closed; only this marker request is queued.
let pendingOpen = null; // { requestId, marker, replay }
let pendingOpenTimer = null;
let channelGeneration = 0;
const OPEN_ACK_TIMEOUT_MS = 5000;
let wakeEngine = null;
let clientCfg = null;
let preroll = [];               // recent PCM16 chunks held while idle in wake mode
let prerollMax = Math.round(1500 / CHUNK_MS); // ~1.5 s
// Hard safety: never stay engaged forever. If a server turn hangs (response.created
// with no response.done — and engaged mode suppresses speech_started), this forces
// a re-arm so a new "Hey Jarvis" works again.
let maxEngageMs = 45000;
let engageTimer = null;
// Live user captions (partial transcription): item_id → caption <span>. The
// server re-decodes the in-progress utterance and sends the FULL hypothesis
// each tick (it can rewrite the tail), so we REPLACE the bubble text, then the
// matching …completed finalizes it (or removes it if the turn was dropped).
const partials = {};
// Playback is routed through a MediaStream → <audio> element rather than
// ctx.destination. On Android, an active mic forces the AudioContext output
// onto the earpiece/communication channel (silent on the loudspeaker); a media
// element plays on the media channel, restoring audible output.
let sinkNode = null, sinkEl = null;

function playbackTarget() {
  return sinkNode || (ctx && ctx.destination);
}

function setStatus(text, live) {
  if (hooks.onStatus) hooks.onStatus(text, !!live);
  if (!ui.status) return;
  ui.status.textContent = text;
  ui.status.classList.toggle("live", !!live);
}

function addTurn(cls, who, text) {
  // Embedded (no standalone log): surface error turns through the chat and skip
  // DOM rendering. Returns a detached span so any caller that reads the return
  // value stays safe (only the standalone transcript path does, and that path is
  // bypassed when hooks are configured).
  if (!ui.log) {
    if (cls === "err" && hooks.onError)
      hooks.onError({ error: { message: who + ": " + (text || "") } });
    return document.createElement("span");
  }
  const div = document.createElement("div");
  div.className = "turn " + cls;
  // Build the speaker label as TEXT, never innerHTML: ``who`` is now the
  // operator-configured product name (SITE_TITLE), so interpolating it into
  // markup would make the brand string an injection sink.
  const whoEl = document.createElement("div");
  whoEl.className = "who";
  whoEl.textContent = who;
  const textEl = document.createElement("span");
  textEl.textContent = text || "";
  div.append(whoEl, textEl);
  ui.log.appendChild(div);
  ui.log.scrollTop = ui.log.scrollHeight;
  return div.querySelector("span");
}

/* ---------------- base64 <-> PCM ---------------- */

function pcmToB64(int16) {
  const bytes = new Uint8Array(int16.buffer, int16.byteOffset, int16.byteLength);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000)
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

/** Linear resample Float32 PCM between sample rates (speech-grade). Desktop
 *  browsers often ignore AudioContext({sampleRate}) and run at 44.1/48 kHz;
 *  the wire protocol and wake models need exact rates. */
function resampleFloat32(input, fromRate, toRate) {
  if (!input || !input.length) return input;
  if (!fromRate || !toRate || Math.abs(fromRate - toRate) < 1) return input;
  const ratio = fromRate / toRate;
  const outLen = Math.max(1, Math.round(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const src = i * ratio;
    const i0 = Math.floor(src);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const t = src - i0;
    out[i] = input[i0] * (1 - t) + input[i1] * t;
  }
  return out;
}

function b64ToFloat32(b64) {
  const bin = atob(b64);
  const n = bin.length >> 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
    if (v >= 0x8000) v -= 0x10000;
    out[i] = v / 32768;
  }
  return out;
}

/* ---------------- playback ---------------- */

let audioChunks = 0;

// Mobile browsers start an AudioContext suspended and only let it produce
// sound after a user gesture. Resume it, build the media-element sink (so audio
// plays on the loudspeaker, not the earpiece, while the mic is live), and play
// a short audible beep confirming the OUTPUT path works.
async function unlockAudio() {
  if (!ctx) return;
  try { await ctx.resume(); } catch (e) {}
  // Route WebAudio → MediaStream → <audio> element = media (loudspeaker) path.
  try {
    if (!sinkNode) {
      sinkNode = ctx.createMediaStreamDestination();
      sinkEl = new Audio();
      sinkEl.autoplay = true;
      sinkEl.playsInline = true;
      sinkEl.srcObject = sinkNode.stream;
    }
    await sinkEl.play().catch(() => {});
  } catch (e) { sinkNode = null; }    // fall back to ctx.destination
  try {
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.frequency.value = 660;
    g.gain.value = 0.06;
    osc.connect(g); g.connect(playbackTarget());
    osc.start();
    osc.stop(ctx.currentTime + 0.15);
  } catch (e) {}
}

/* Play a short oscillator beep on the correct output path.
 * Used for wake-word / turn-end audio feedback. Fails silently. */
function playBeep(freq, duration) {
  if (!ctx || ctx.state === "suspended") return;
  try {
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.frequency.value = freq;
    osc.type = "sine";
    g.gain.value = 0.12;
    g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
    osc.connect(g); g.connect(playbackTarget());
    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) {}
}

function playDelta(b64) {
  const samples = b64ToFloat32(b64);
  if (!samples.length || !ctx) return;
  if (ctx.state === "suspended") ctx.resume().catch(() => {});
  if (sinkEl && sinkEl.paused) sinkEl.play().catch(() => {});
  audioChunks++;
  setStatus(`🔊 speaking… (${audioChunks})`, true);
  try {
    const buf = ctx.createBuffer(1, samples.length, WIRE_RATE);
    buf.copyToChannel(samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(playbackTarget());
    playHead = Math.max(playHead, ctx.currentTime + 0.06);
    src.start(playHead);
    playHead += buf.duration;
    liveSources.push(src);
    src.onended = () => { liveSources = liveSources.filter((s) => s !== src); };
  } catch (e) {
    addTurn("err", "audio", "playback failed: " + e.message);
  }
}

function flushPlayback() {
  for (const s of liveSources) { try { s.stop(); } catch (e) {} }
  liveSources = [];
  playHead = 0;
}

/* ---------------- mic capture ---------------- */

const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("sa-tap", Tap);`;

async function startMic() {
  if (!ctx || ctx.state === "closed") {
    ctx = new AudioContext({ sampleRate: WIRE_RATE });
    micWorkletReady = false;
  }
  // NS/AGC OFF: openWakeWord is trained on near-raw speech, and the wake
  // engine shares this same track (one mic permission, one capture). EC stays
  // on so barge-in still works while Jarvis is speaking.
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
  });
  if (!micWorkletReady) {
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    try { await ctx.audioWorklet.addModule(url); }
    finally { URL.revokeObjectURL(url); }
    micWorkletReady = true;
  }
  const srcNode = ctx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(ctx, "sa-tap");
  srcNode.connect(micNode);
  micNode.port.onmessage = (e) => onMicChunk(e.data);
  // Hand the live mic stream to the embedder so it can drive its own waveform
  // (the main chat reuses its existing #audio-visualizer canvas).
  if (hooks.onMicStreamStart) try { hooks.onMicStreamStart(micStream); } catch (e) {}
}

// ── capture egress gate ──────────────────────────────────
// The SINGLE predicate that decides whether a captured chunk may leave the edge
// device. Both the live send (onMicChunk) and the durable outage buffer
// gate on THIS function, so wake-idle (pre-wake) audio can never egress or be
// buffered. Pure + free of module state so it is unit-testable in isolation
// (see tests/test_realtime_egress_gate.js).
function egressAllowed(mode, streaming) {
  return (mode === "wake" || mode === "ptt") && streaming === true;
}

// A captured chunk may enter the durable
// outage buffer ONLY while a reconnect is in flight AND the chunk is egress-eligible
// (explicitly opened). This re-asserts the wake gate at the buffer boundary — the
// same gate as live egress — so a reconnect can never persist or resend pre-wake
// audio. Pure + unit-testable.
function shouldBufferOutage(reconnecting, mode, streaming) {
  return reconnecting === true && egressAllowed(mode, streaming);
}

function sendAudioChunk(int16) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio: pcmToB64(int16) }));
    return;
  }
  // The socket dropped mid-conversation. Hold the chunks
  // that WOULD have egressed (only while explicitly open) in a bounded outage
  // buffer and replay them on reconnect, so the user's speech during the outage
  // isn't lost. The shouldBufferOutage() gate enforces the wake rule at the
  // buffer boundary so wake-idle (pre-wake) audio is NEVER buffered even if a
  // caller mis-gates; anything not egress-eligible (or a permanent stop) is dropped.
  if (shouldBufferOutage(reconnecting, mode, streaming)) bufferOutageAudio(int16);
}

function onMicChunk(f32) {
  // Desktop AudioContexts often ignore the requested 24 kHz rate — resample
  // so the server ASR always sees WIRE_RATE PCM (otherwise transcripts die).
  if (ctx && Math.abs(ctx.sampleRate - WIRE_RATE) > 1) {
    f32 = resampleFloat32(f32, ctx.sampleRate, WIRE_RATE);
  }
  // VU meter
  let peak = 0;
  for (let i = 0; i < f32.length; i++) peak = Math.max(peak, Math.abs(f32[i]));
  if (ui.vu) ui.vu.style.width = Math.min(100, peak * 140) + "%";
  if (hooks.onVu) hooks.onVu(peak);

  // Keep capturing while a reconnect is in flight — the
  // SAME gating below decides what leaves the device. In wake-idle the chunk
  // only updates the local pre-roll ring (never buffered/sent, so pre-wake
  // audio still never egresses); while explicitly
  // open, sendAudioChunk() diverts to the bounded outage buffer for replay.
  if ((!ws || ws.readyState !== WebSocket.OPEN) && !reconnecting) return;
  // Route each SAMPLE to the accumulator that matches its egress
  // eligibility BEFORE it is accumulated, so a partial (sub-chunk) can never straddle
  // the wake→engaged boundary. The egress accumulator (sendBuf/sendFill) fills ONLY
  // while the channel is open; a wake-idle sample fills ONLY the local
  // preroll accumulator (prerollBuf/prerollFill) and reaches the preroll ring alone —
  // never sendBuf. This structurally prevents a boundary chunk that mixes pre-wake and
  // post-wake samples. (Previously ALL samples filled sendBuf and were classified
  // per-whole-chunk, so a pre-wake partial lingering in sendBuf became the prefix of
  // the first post-wake chunk and was buffered/resent across a reconnect — the residual
  // pre-wake leak this closes; engage() also resets sendFill/prerollFill.)
  // NOTE: channel state changes run on event-loop boundaries, never within this
  // synchronous sample loop, so egressAllowed is constant for the whole frame.
  // Meeting is the one sanctioned no-wake path. It is represented by its own
  // state (never by opening the interactive assistant gate), and only a live,
  // signal-completed meeting may use it.
  const meetingCapture = meetingState === "active" && connected();
  if (meetingState === "active" && !meetingCapture) return;
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767)));
    if (meetingCapture || egressAllowed(mode, streaming)) {
      sendBuf[sendFill++] = s;
      if (sendFill === CHUNK_SAMPLES) {
        sendAudioChunk(sendBuf.slice(0)); // copy out of the reused buffer
        sendFill = 0;
      }
    } else {
      // Wake mode, idle: hold recent audio in a LOCAL ring buffer so the wake word
      // + lead-in can be flushed upstream when detection fires (keeps the server text
      // gate matching and avoids clipping the command). This accumulator is SEPARATE
      // from the egress one, so pre-wake samples never touch sendBuf.
      prerollBuf[prerollFill++] = s;
      if (prerollFill === CHUNK_SAMPLES) {
        preroll.push(prerollBuf.slice(0));
        if (preroll.length > prerollMax) preroll.shift();
        prerollFill = 0;
      }
    }
  }
}

function stopMic(keepAudioOutput) {
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (micNode) try { micNode.disconnect(); } catch (e) {}
  micNode = micStream = null;
  if (!keepAudioOutput) {
    if (sinkEl) { try { sinkEl.pause(); } catch (e) {} sinkEl.srcObject = null; }
    if (ctx) ctx.close();
    ctx = sinkNode = sinkEl = null;
    micWorkletReady = false;
  }
  if (ui.vu) ui.vu.style.width = "0%";
  if (hooks.onMicStreamStop) try { hooks.onMicStreamStop(); } catch (e) {}
}

/* ---------------- passive meeting mode ---------------- */

// Meeting is a distinct capture state, not wake/PTT assistant mode. The server
// issues a one-use signal challenge for each transition. Completion is returned
// only after this real WebAudio chime finishes; failure leaves start closed.
let meetingState = "idle"; // idle | starting | active | stopping
let meetingReturnPending = false;
let meetingGen = 0;
let meetingChallenge = null;

function invalidateMeetingTransition() {
  meetingGen++;
  meetingChallenge = null;
}

function meetingTransitionCurrent(gen, socket, phase, token, expectedState) {
  return meetingGen === gen
    && ws === socket
    && connected()
    && meetingState === expectedState
    && meetingChallenge
    && meetingChallenge.gen === gen
    && meetingChallenge.phase === phase
    && meetingChallenge.token === token;
}

function updateMeetingUI() {
  if (!ui.meeting) return;
  const active = meetingState === "active" || meetingState === "stopping";
  ui.meeting.textContent = active ? "■ stop meeting" : "◉ meeting";
  ui.meeting.classList.toggle("active", active);
  ui.meeting.disabled = meetingState === "starting" || meetingState === "stopping";
}

async function playMeetingSignal(phase) {
  if (!ctx || ctx.state === "closed") throw new Error("audio output unavailable");
  await ctx.resume();
  if (ctx.state !== "running") throw new Error("audio output is not running");
  if (sinkEl && sinkEl.paused) await sinkEl.play();
  const notes = phase === "start" ? [523.25, 659.25, 783.99] : [783.99, 659.25, 523.25];
  for (const frequency of notes) {
    await new Promise((resolve, reject) => {
      try {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.frequency.value = frequency;
        osc.type = "sine";
        gain.gain.setValueAtTime(0.16, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.22);
        osc.connect(gain); gain.connect(playbackTarget());
        osc.onended = resolve;
        osc.start();
        osc.stop(ctx.currentTime + 0.24);
      } catch (e) { reject(e); }
    });
  }
}

async function startMeeting() {
  if (meetingState !== "idle") throw new Error("meeting transition already in progress");
  if (!connected()) await connect();
  if (!connected()) throw new Error("not connected");
  // Mode transition invalidates any in-flight interactive OPEN generation
  // before the meeting intent can leave this client.
  rejectPendingOpen("", false);
  invalidateMeetingTransition();
  meetingState = "starting";
  updateMeetingUI();
  setStatus("preparing audible meeting notice…");
  ws.send(JSON.stringify({ type: "meeting.start" }));
}

function stopMeeting() {
  if (meetingState !== "active" || !connected()) return;
  invalidateMeetingTransition();
  meetingState = "stopping";
  updateMeetingUI();
  // Capture/egress ceases locally before the stop request leaves the client.
  stopMic(true);
  ws.send(JSON.stringify({ type: "meeting.stop" }));
  setStatus("finalizing meeting…");
}

/* ---------------- wake-word gating ---------------- */

// Build paths relative to the page dir so it works at "/" or under a proxy
// prefix like "/sharedagentsrt/" (mirrors connect()).
function pageDir() {
  return location.pathname.endsWith("/")
    ? location.pathname
    : location.pathname.replace(/[^/]*$/, "");
}

async function loadClientConfig() {
  try {
    // Same-origin standalone; embedded points at the realtime server (the fetch
    // may be cross-origin in dev and simply fail → built-in defaults, harmless).
    const r = await fetch(realtimeBase().http + "v1/realtime/clientconfig",
                          { credentials: "include" });
    if (r.ok) return await r.json();
  } catch (e) { /* fall back to built-in defaults */ }
  return null;
}

function updateModeUI() {
  if (ui.mode) {
    ui.mode.textContent = mode === "wake" ? "🅦 local wake" : "hold to talk";
    ui.mode.classList.toggle("active", mode === "wake");
    ui.mode.title = mode === "wake"
      ? "Audio leaves this device only after local wake detection."
      : "Wake engine unavailable. Hold to deliberately open the channel.";
  }
  // Every supported interactive mode is explicitly gated on this edge device.
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: "session.update",
                             session: { wake_client_gated: true } }));
}

// Engagement lifecycle. The server's stop-word event closes the channel; this
// watchdog is only a privacy safety net for a hung/incomplete turn.
function armWatchdog(ms) {
  clearTimeout(engageTimer);
  engageTimer = setTimeout(() => {
    if (streaming) { addTurn("err", "wake", "turn timed out — re-arming wake word"); disengage(); }
  }, ms != null ? ms : maxEngageMs);
}

function holdEngaged() {        // activity in flight → no lull, extend watchdog
  if (streaming) armWatchdog();
}

function rejectPendingOpen(reason, surfaceError) {
  clearTimeout(pendingOpenTimer);
  pendingOpenTimer = null;
  pendingOpen = null;
  streaming = false;
  sendFill = prerollFill = 0;
  preroll.length = 0;
  // Buffered audio was eligible under the old connection, but a new server
  // session that did not acknowledge OPEN must never receive it.
  outageBuffer = [];
  outageDropped = false;
  if (surfaceError && reason) addTurn("err", "wake", reason);
  if (connected())
    setStatus(mode === "wake" ? "say “Hey Jarvis”…" : "hold to talk", true);
}

function sendPendingOpen() {
  if (!pendingOpen || !connected()) return;
  ws.send(JSON.stringify({ type: pendingOpen.marker,
                           request_id: pendingOpen.requestId }));
}

function requestChannelOpen(marker, replay) {
  if (pendingOpen) return;
  streaming = false;
  sendFill = prerollFill = 0;
  preroll.length = 0;
  const requestId = `open-${++channelGeneration}-${Date.now()}`;
  pendingOpen = { requestId, marker, replay: !!replay };
  clearTimeout(pendingOpenTimer);
  pendingOpenTimer = setTimeout(() => {
    if (pendingOpen && pendingOpen.requestId === requestId)
      rejectPendingOpen("channel open timed out; audio stayed local", true);
  }, OPEN_ACK_TIMEOUT_MS);
  sendPendingOpen();
  setStatus("opening audio channel…", true);
}

function acknowledgeChannelOpen(ev, marker) {
  if (!pendingOpen || pendingOpen.marker !== marker
      || ev.request_id !== pendingOpen.requestId) return;
  const replay = pendingOpen.replay;
  clearTimeout(pendingOpenTimer);
  pendingOpenTimer = null;
  pendingOpen = null;
  // Samples captured while awaiting acknowledgement were routed to the local
  // accumulator. Drop them so the first egress sample is strictly post-ACK.
  sendFill = prerollFill = 0;
  preroll.length = 0;
  streaming = true;
  playBeep(880, 0.15);
  setStatus("listening…", true);
  armWatchdog();
  if (replay) flushOutageBuffer();
}

function engage() {
  if (mode !== "wake") return;
  if (pendingOpen) return;
  const interrupt = responding || liveSources.length > 0;
  if (interrupt) {
    flushPlayback();
  } else if (streaming) {
    armWatchdog();
    return;
  }
  requestChannelOpen(interrupt ? "session.interrupt" : "input_audio_buffer.wake", false);
  if (hooks.onWakeDetected) try { hooks.onWakeDetected(); } catch (e) {}
}

function disengage() {
  clearTimeout(engageTimer); engageTimer = null;
  rejectPendingOpen("", false);
  playBeep(440, 0.15); // lower chirp: "turn ended, processing"
  if (hooks.onTurnEnd) try { hooks.onTurnEnd(); } catch (e) {}
  preroll.length = 0;
  if (ws && ws.readyState === WebSocket.OPEN)
    setStatus(mode === "wake" ? "say “Hey Jarvis”…" : "hold to talk", true);
}

// Generation token: each start is tagged; stopWakeEngine() bumps it so an
// in-flight load/start that resumes after a mode switch / disconnect aborts
// instead of binding a stale engine or opening a mic post-disconnect.
let wakeGen = 0;

function connected() { return !!ws && ws.readyState === WebSocket.OPEN; }

async function startWakeEngine() {
  if (!connected()) throw new Error("not connected");
  const gen = ++wakeGen;
  const w = (clientCfg && clientCfg.wake) || {};
  const assetBase = realtimeAssetBase || pageDir();
  const engine = new WakeWordEngine({
    keyword: w.keyword || "hey_jarvis",
    modelBase: assetBase + "models",
    ortWasmPath: assetBase + "vendor/onnxruntime/",
    threshold: w.threshold != null ? w.threshold : 0.28,
    cooldownMs: w.cooldown_ms != null ? w.cooldown_ms : 1200,
    vadGate: !!(w.vad_gate),
  });
  await engine.load();
  if (gen !== wakeGen || mode !== "wake" || !connected()) { // superseded while loading
    try { await engine.stop(); } catch (e) {}
    return;
  }
  engine.on("detect", engage);
  engine.on("error", (e) => failWakeClosed("wake engine error: " + ((e && e.message) || e)));
  // Share the realtime mic track (already opened with NS/AGC off). A second
  // getUserMedia competed for the device and was a common miss on Linux/Chrome.
  if (!micStream) throw new Error("mic not started");
  await engine.start({ mediaStream: micStream });
  if (gen !== wakeGen) { try { await engine.stop(); } catch (e) {} return; }
  wakeEngine = engine;
}

async function stopWakeEngine() {
  wakeGen++; // cancel any in-flight startWakeEngine
  clearTimeout(engageTimer); engageTimer = null;
  const e = wakeEngine;
  wakeEngine = null;
  if (e) { try { await e.stop(); } catch (err) {} }
}

function failWakeClosed(reason) {
  mode = "ptt";
  rejectPendingOpen("", false);
  stopWakeEngine();
  updateModeUI();
  if (hooks.onModeChange) try { hooks.onModeChange("ptt"); } catch (e) {}
  if (reason) addTurn("err", "wake", reason + " — hands-free disabled; hold to talk");
  if (ws && ws.readyState === WebSocket.OPEN) setStatus("hold to talk", true);
}

// Load the wake engine. Any failure stays locally gated and offers PTT.
async function enterWakeMode() {
  mode = "wake";
  rejectPendingOpen("", false);
  updateModeUI();
  if (hooks.onModeChange) try { hooks.onModeChange("wake"); } catch (e) {}
  if (!micStream) return; // not connected yet — engine starts on connect()
  try {
    if (typeof WakeWordEngine === "undefined") throw new Error("wake engine unavailable");
    if (!wakeEngine) await startWakeEngine();
    if (ws && ws.readyState === WebSocket.OPEN) setStatus("say “Hey Jarvis”…", true);
  } catch (e) {
    let msg = "wake init failed: " + ((e && e.message) || e);
    // The openWakeWord models are CC BY-NC-SA and cannot ship in the package;
    // a load failure usually means they were never fetched to this host.
    if (/model load failed|404|not found/i.test(String(e && e.message))) {
      msg += " — run `sharedagents web fetch-voice-models` on the web host";
    }
    failWakeClosed(msg);
  }
}

function beginPushToTalk() {
  if (mode !== "ptt" || !connected() || streaming || pendingOpen) return;
  requestChannelOpen("input_audio_buffer.wake", false);
}

function endPushToTalk() {
  if (mode === "ptt" && pendingOpen) {
    rejectPendingOpen("", false);
    return;
  }
  if (mode !== "ptt" || !streaming) return;
  streaming = false;
  clearTimeout(engageTimer); engageTimer = null;
  if (connected()) ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
  else {
    // Releasing PTT closes this turn. Audio accumulated while its socket was
    // down belongs to the abandoned turn and may not cross into a later OPEN.
    outageBuffer = [];
    outageDropped = false;
  }
  setStatus("hold to talk", true);
}

/* ---------------- protocol ---------------- */

async function onEvent(ev) {
  switch (ev.type) {
    case "session.created":
      ws.send(JSON.stringify({
        type: "session.update",
        session: { turn_detection: { type: "server_vad", silence_duration_ms: 550 } },
      }));
      if (meetingReturnPending) {
        meetingReturnPending = false;
        meetingState = "idle";
        updateMeetingUI();
        await startMic();
        await unlockAudio();
        await enterWakeMode();
      }
      setStatus(pendingOpen ? "opening audio channel…"
                : (mode === "wake" && !streaming
                   ? "say “Hey Jarvis”…" : "listening…"), true);
      break;
    case "meeting.signal": {
      const phase = ev.phase;
      const token = ev.token;
      const socket = ws;
      const expectedState = phase === "start" ? "starting" : "stopping";
      const gen = meetingGen;
      if (!connected() || meetingState !== expectedState || !token) break;
      meetingChallenge = { gen, phase, token };
      try {
        if (phase === "start") {
          // Stop every interactive capture mechanism before notice. Meeting
          // capture itself is opened only after the chime has fully ended.
          rejectPendingOpen("", false);
          await stopWakeEngine();
          if (!meetingTransitionCurrent(gen, socket, phase, token, expectedState)) break;
          stopMic(true);
        }
        await playMeetingSignal(phase);
        if (!meetingTransitionCurrent(gen, socket, phase, token, expectedState)) break;
        if (phase === "start") {
          await startMic();
          if (!meetingTransitionCurrent(gen, socket, phase, token, expectedState)) {
            stopMic(true);
            break;
          }
        }
        if (!meetingTransitionCurrent(gen, socket, phase, token, expectedState)) break;
        if (phase === "stop") meetingReturnPending = true;
        socket.send(JSON.stringify({ type: "meeting.signal.completed", phase, token }));
      } catch (e) {
        if (!meetingTransitionCurrent(gen, socket, phase, token, expectedState)) {
          if (phase === "start") stopMic(true);
          break;
        }
        meetingReturnPending = true;
        socket.send(JSON.stringify({ type: "meeting.signal.failed", phase, token }));
        invalidateMeetingTransition();
        addTurn("err", "meeting", "audible notice failed: " + e.message);
      }
      break;
    }
    case "meeting.started":
      if (!connected() || meetingState !== "starting" || !meetingChallenge) break;
      meetingChallenge = null;
      meetingState = "active";
      updateMeetingUI();
      setStatus("meeting transcription active", true);
      if (hooks.onMeetingStarted) hooks.onMeetingStarted(ev);
      break;
    case "meeting.transcription.completed":
      if (hooks.onMeetingTranscript) hooks.onMeetingTranscript(ev);
      else if (ui.log) addTurn("user", "meeting record", ev.transcript || "");
      break;
    case "meeting.stopped":
      if (meetingState !== "stopping") break;
      setStatus(`meeting saved (${ev.chunk_count || 0} chunks)`);
      if (hooks.onMeetingStopped) hooks.onMeetingStopped(ev);
      break;
    case "meeting.failed":
      // Server-side ASR/persistence faults fail capture closed. Stop the local
      // mic immediately, audibly announce that recording has ceased, then let
      // the following fresh interactive session reopen its own gated mic.
      invalidateMeetingTransition();
      const failureGen = meetingGen;
      const failureSocket = ws;
      meetingState = "stopping";
      updateMeetingUI();
      stopMic(true);
      await stopWakeEngine();
      if (meetingGen !== failureGen || ws !== failureSocket || !connected()
          || meetingState !== "stopping") break;
      try { await playMeetingSignal("stop"); }
      catch (e) { addTurn("err", "meeting", "stop notice failed: " + e.message); }
      if (meetingGen !== failureGen || ws !== failureSocket || !connected()
          || meetingState !== "stopping") break;
      meetingReturnPending = true;
      if (hooks.onMeetingFailed) hooks.onMeetingFailed(ev);
      break;
    case "meeting.signal.acknowledged":
      break;
    case "input_audio_buffer.wake_acknowledged":
      acknowledgeChannelOpen(ev, "input_audio_buffer.wake");
      break;
    case "session.interrupt_acknowledged":
      acknowledgeChannelOpen(ev, "session.interrupt");
      break;
    case "session.config":
      // Agent mode: the server tells us which gateway session it drives, so the
      // embedder (main chat) can render the turn through its normal canonical
      // pipeline instead of plain transcript bubbles.
      // Remember the server-issued RESUME TOKEN
      // (not the raw gateway sid) so an unexpected disconnect can RESUME the same
      // conversation — the token is what proves to the server that we originated
      // it, so a foreign/guessed sid can't rebind our voice. On a successful
      // resume the server re-announces the same gateway id with a fresh token; on
      // a failed resume it announces a fresh session — either way we track
      // whatever it last told us. gateway_session_id is still used (below) to
      // subscribe the main chat to the canonical event stream.
      if (ev.gateway_session_id) gatewaySessionId = ev.gateway_session_id;
      if (ev.resume_token) resumeToken = ev.resume_token;
      if (hooks.onGatewaySession && ev.gateway_session_id)
        hooks.onGatewaySession(ev.gateway_session_id, ev.agent);
      break;
    case "input_audio_buffer.speech_started":
      setStatus("hearing you…", true);
      holdEngaged();
      break;
    case "input_audio_buffer.speech_stopped":
      setStatus("transcribing…", true);
      break;
    case "conversation.item.input_audio_transcription.delta": {
      if (hooks.onUserDelta) { hooks.onUserDelta(ev); break; }
      if (!ui.log) break; // embedded without a partials hook → wait for the final
      // Partial caption while the user is still speaking. Replace (don't append)
      // because re-decoding can rewrite the unsettled tail.
      let span = ev.item_id ? partials[ev.item_id] : null;
      if (!span) {
        span = addTurn("user", "you", "");
        span.closest(".turn").classList.add("partial");
        if (ev.item_id) partials[ev.item_id] = span;
      }
      span.textContent = ev.delta || "";
      ui.log.scrollTop = ui.log.scrollHeight;
      break;
    }
    case "conversation.item.input_audio_transcription.completed": {
      if (hooks.onUserFinal) { hooks.onUserFinal(ev); break; }
      if (!ui.log) break; // embedded fallback safety
      const span = ev.item_id ? partials[ev.item_id] : null;
      if (span) {
        delete partials[ev.item_id];
        const turn = span.closest(".turn");
        if (ev.transcript) {
          span.textContent = ev.transcript;   // authoritative final, replace
          turn.classList.remove("partial");
        } else {
          turn.remove();                       // dropped (noise/no wake) → clear
        }
      } else if (ev.transcript) {
        addTurn("user", "you", ev.transcript); // partials off / no live bubble
      } else {
        setStatus("(heard nothing intelligible)", true);
      }
      break;
    }
    case "response.created":
      audioChunks = 0;
      responding = true;
      setStatus("answering…", true);
      if (hooks.onAssistantStart) { hooks.onAssistantStart(ev); break; }
      currentBot = addTurn("bot", SITE_TITLE, "");
      currentBot.classList.add("pulse");
      break;
    case "response.output_audio_transcript.delta":
      if (hooks.onAssistantDelta) { hooks.onAssistantDelta(ev.delta || "", ev); break; }
      if (currentBot) currentBot.textContent += ev.delta;
      if (ui.log) ui.log.scrollTop = ui.log.scrollHeight;
      break;
    case "response.output_audio.delta":
      playDelta(ev.delta);
      break;
    case "response.done": {
      responding = false;
      if (currentBot) currentBot.classList.remove("pulse");
      currentBot = null;
      const m = ev.response && ev.response.metrics;
      if (hooks.onAssistantDone) hooks.onAssistantDone(ev);
      if (m && ui.metrics) ui.metrics.textContent =
        `asr ${m.asr_ms}ms · llm first token ${m.llm_ttft_ms}ms · ` +
        `first audio ${m.tts_first_audio_ms}ms · spoke ${(m.audio_ms / 1000).toFixed(1)}s · ` +
        `pipeline ${m.total_ms}ms · audio chunks rx ${audioChunks}`;
      // Audio was produced server-side but none reached us → transport issue,
      // not playback. Surface it instead of failing silently.
      if (m && m.audio_ms > 0 && audioChunks === 0)
        addTurn("err", "audio", "server sent audio but none arrived in the browser");
      setStatus(streaming ? "listening…"
                : (mode === "wake" ? "say “Hey Jarvis”…" : "hold to talk"), true);
      break;
    }
    case "response.cancelled":
      responding = false;
      if (currentBot) { currentBot.classList.remove("pulse"); currentBot = null; }
      flushPlayback();
      setStatus(streaming ? "listening…"
                : (mode === "wake" ? "say “Hey Jarvis”…" : "hold to talk"), true);
      if (hooks.onAssistantCancelled) hooks.onAssistantCancelled(ev);
      break;
    case "session.wake_reset":
      // Server tells us the conversation ended (e.g. user said "over").
      // Re-arm the wake word — the next turn needs "Hey Jarvis" again.
      if (streaming) disengage();
      if (hooks.onWakeReset) try { hooks.onWakeReset(ev); } catch (e) {}
      break;
    case "error":
      if (meetingState === "starting") {
        invalidateMeetingTransition();
        stopMic(true);
        meetingState = "stopping";
        meetingReturnPending = true;
        updateMeetingUI();
      }
      if (pendingOpen)
        rejectPendingOpen((ev.error && ev.error.message)
                          || "channel open rejected; audio stayed local", true);
      if (hooks.onError) { hooks.onError(ev); break; }
      addTurn("err", "error", ev.error && ev.error.message);
      break;
  }
}

/* ---------------- connect / reconnect / disconnect ----------------
 *
 * The realtime conversation is PERSISTENT across a
 * connection loss. Reconnect state machine:
 *
 *   connected ──(user clicks stop)──────────────▶ disconnect()  [full teardown]
 *   connected ──(unexpected ws.onclose)─────────▶ RECONNECTING
 *   RECONNECTING ──(socket reopens)─────────────▶ connected (resume + replay)
 *   RECONNECTING ──(user clicks stop)───────────▶ disconnect()  [cancel]
 *
 * The distinction is `userClosing`: a button-driven stop sets it, so ws.onclose
 * tears down as before. An *unexpected* close (network drop / server restart)
 * leaves it false → we KEEP the mic, wake engine, and — critically — the entry
 * control (we never call onConnected(false) on a loss, so the embedder's
 * driver-mode button stays visible). We buffer captured audio (bounded) and
 * auto-reconnect with capped exponential backoff, retrying UNTIL the socket
 * comes back or the user cancels (no hard attempt cap — the entry control is
 * always present to cancel, per the "never lock the owner out" requirement). On
 * reconnect we resume the SAME gateway session (the server-issued token, offered
 * as a WS subprotocol — never in the URL) and replay the buffered audio in
 * order.
 *
 * Privacy ("pre-wake audio is
 * never buffered or resent"). The outage buffer is fed ONLY by bufferOutageAudio,
 * whose ONLY caller is sendAudioChunk's shouldBufferOutage() branch. Enumerate
 * every path that could reach a chunk into that branch; none is wake-gated:
 *
 *   1. onMicChunk → sendAudioChunk — each captured SAMPLE is routed by
 *      egressAllowed(mode, streaming) BEFORE accumulation: OPEN samples fill
 *      the egress accumulator (sendBuf/sendFill) and a completed chunk reaches
 *      sendAudioChunk; wake-idle samples fill a SEPARATE accumulator (prerollBuf/
 *      prerollFill) that feeds the LOCAL preroll ring only. Because the two accumulators
 *      are distinct — and engage() resets sendFill/prerollFill on the wake→engaged
 *      transition — a pre-wake partial can never straddle the boundary into a sendBuf
 *      chunk. Wake-idle audio is never passed onward.
 *   2. flushOutageBuffer → sendAudioChunk — runs only after the socket is OPEN
 *      (attemptReconnect), so it takes the ws.send branch; by (1) the buffer
 *      it drains only ever held explicitly-open audio.
 *
 * shouldBufferOutage() re-asserts egressAllowed() at the buffer boundary as
 * defense-in-depth. Net: no pre-wake audio is ever buffered or resent across a
 * reconnect; if a wake engagement was mid-flight when the socket dropped, only the
 * post-wake portion (captured with streaming===true) is buffered and replayed.
 */

// Connection generation: a stop (disconnect) during an in-flight connect() must
// abort the rest of connect() after its awaits, so a session can't resurrect
// after the user already stopped it. Bumped on every connect() and disconnect().
let connectGen = 0;

// Reconnect state.
let userClosing = false;        // true = the user intends a full teardown
let reconnecting = false;       // true = in the auto-reconnect loop after a loss
let reconnectAttempts = 0;      // consecutive failed reopen attempts (backoff)
let reconnectTimer = null;      // pending setTimeout for the next attempt
const RECONNECT_BASE_MS = 500;  // first backoff delay
const RECONNECT_CAP_MS = 15000; // max backoff between attempts
let gatewaySessionId = null;    // last gateway session id (for canonical-stream subscribe)
let resumeToken = null;         // last server-issued resume token (the reconnect handle)
// Bounded outage buffer: PCM16 chunks captured while the socket is down, replayed
// in order on reconnect. Capped by DURATION so it can never grow without bound.
let outageBuffer = [];
let outageDropped = false;      // one-shot "buffer full" status note
const OUTAGE_BUFFER_MS = 30000; // ~30 s of captured audio, then drop oldest
let outageBufferMax = Math.round(OUTAGE_BUFFER_MS / CHUNK_MS);

// Base WS subprotocol always offered so the server has one to echo back.
const REALTIME_SUBPROTOCOL = "sharedagents.realtime.v1";

function resumeProtocols() {
  // Carry the server-issued resume TOKEN in the WebSocket SUBPROTOCOL
  // list — a handshake HEADER (Sec-WebSocket-Protocol), NOT the URL query. A
  // bearer token in the URL leaks via the server access log, browser history,
  // and Referer; a subprotocol offer is in none of those. Only a token the
  // server minted for us resolves back to our sid, so a foreign/guessed value
  // can't rebind our voice. token_urlsafe → [A-Za-z0-9_-] plus the "rt_" prefix
  // are all valid subprotocol token chars, so it rides verbatim (no encoding).
  return resumeToken
    ? [REALTIME_SUBPROTOCOL, "sharedagents.resume." + resumeToken]
    : [REALTIME_SUBPROTOCOL];
}

function bufferOutageAudio(int16) {
  outageBuffer.push(int16);
  if (outageBuffer.length > outageBufferMax) {
    outageBuffer.shift();       // drop oldest — bounded, never unbounded growth
    if (!outageDropped) {
      outageDropped = true;
      setStatus("reconnecting… (buffer full — dropping oldest audio)");
    }
  }
}

function flushOutageBuffer() {
  const pending = outageBuffer;
  outageBuffer = [];
  outageDropped = false;
  for (const c of pending) sendAudioChunk(c);   // ws is OPEN now → real send
}

// Wire a socket's steady-state handlers. The open handshake (below) overrides
// onopen/onerror transiently; onclose is what drives the reconnect machine.
function wireSocket(sock) {
  let eventChain = Promise.resolve();
  sock.onmessage = (e) => {
    eventChain = eventChain
      .then(() => onEvent(JSON.parse(e.data)))
      .catch((err) => addTurn("err", "meeting", err.message || String(err)));
  };
  sock.onerror = () => setStatus("websocket error");
  sock.onclose = () => {
    // Ignore a socket we've already replaced (a superseded reconnect attempt)
    // or a close the user asked for (full teardown handled in disconnect()).
    if (sock !== ws || userClosing) return;
    const interruptedStart = meetingState === "starting" && meetingChallenge !== null;
    invalidateMeetingTransition();
    if (meetingState === "starting") {
      // If the challenge had arrived, its handler already stopped the mic.
      // Reopen only through the fresh interactive session after reconnect.
      meetingState = "idle";
      meetingReturnPending = interruptedStart;
      updateMeetingUI();
      scheduleReconnect();
      return;
    }
    if (meetingState === "active" || meetingState === "stopping") {
      // A reconnect creates an interactive server session, so replaying meeting
      // audio into it would violate the record-not-prompt boundary. Stop locally
      // and announce that stop; meeting mode never enters interactive reconnect.
      meetingState = "stopping";
      stopMic(true);
      playMeetingSignal("stop").catch((e) => {
        addTurn("err", "meeting", "stop notice failed: " + e.message);
      }).finally(() => {
        meetingState = "idle";
        updateMeetingUI();
        disconnect(true);
      });
      return;
    }
    scheduleReconnect();        // unexpected loss → persist + reconnect
  };
}

// Open a fresh WebSocket, wire it, and resolve when it's OPEN (reject on error).
// Used by BOTH the initial connect() and each reconnect attempt so the resume
// subprotocol offer and handlers stay identical. The token is passed as a WS
// subprotocol (a handshake header), never in the URL.
function openSocket() {
  const url = realtimeBase().ws;
  const sock = new WebSocket(url, resumeProtocols());
  wireSocket(sock);
  ws = sock;
  return new Promise((res, rej) => { sock.onopen = res; sock.onerror = rej; });
}

function scheduleReconnect() {
  if (userClosing) return;
  if (!reconnecting) {
    reconnecting = true;
    if (hooks.onReconnecting) try { hooks.onReconnecting(true); } catch (e) {}
  }
  clearTimeout(reconnectTimer);
  const delay = Math.min(RECONNECT_CAP_MS,
                         RECONNECT_BASE_MS * Math.pow(2, reconnectAttempts));
  setStatus(`reconnecting… (attempt ${reconnectAttempts + 1})`);
  reconnectTimer = setTimeout(attemptReconnect, delay);
}

async function attemptReconnect() {
  if (userClosing) return;
  reconnectAttempts++;
  try {
    await openSocket();         // resolves on OPEN, rejects on error
  } catch (e) {
    scheduleReconnect();        // failed → back off and retry (until cancel)
    return;
  }
  if (userClosing) { try { ws.close(); } catch (e) {} return; }
  // Reconnected. The server sends a fresh session.created (its RealtimeSession
  // is new even though the gateway conversation is rebound), so re-assert the
  // client-gating flag BEFORE replaying, then flush the buffered audio.
  reconnecting = false;
  reconnectAttempts = 0;
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  if (hooks.onReconnecting) try { hooks.onReconnecting(false); } catch (e) {}
  if (hooks.onConnected) hooks.onConnected(true);   // re-mark live (kept visible)
  updateModeUI();               // re-sends wake_client_gated on the new session
  // A request whose ACK was lost is re-sent with the same generation. An
  // already-open turn must re-open on the new server session before replay;
  // streaming flips false until the matching acknowledgement arrives.
  if (pendingOpen) sendPendingOpen();
  else if (streaming) requestChannelOpen("input_audio_buffer.wake", true);
  else {
    // Defense in depth: without OPEN or a pending marker there is no turn to
    // which buffered audio can belong. Never carry it into a future wake/PTT.
    outageBuffer = [];
    outageDropped = false;
    setStatus(mode === "wake" ? "say “Hey Jarvis”…" : "hold to talk", true);
  }
}

async function connect() {
  const gen = ++connectGen;
  // Fresh, user-initiated connect: clear all reconnect/outage state and start a
  // NEW conversation (no resume handle carried over from a prior session).
  userClosing = false;
  reconnecting = false;
  reconnectAttempts = 0;
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  outageBuffer = [];
  outageDropped = false;
  gatewaySessionId = null;
  resumeToken = null;
  // Fail safe to the GATED state before any capture starts: a prior session may
  // have ended with an open channel, and the mic starts producing chunks the
  // instant the ws opens — without this reset those chunks could go
  // upstream before client config re-selects wake mode (a privacy leak on
  // reconnect). Default = wake + not streaming → onMicChunk only buffers.
  mode = "wake";
  responding = false;
  rejectPendingOpen("", false);
  // Auth is the dashboard JWT cookie, sent automatically on the WS handshake —
  // no token in the URL. Standalone: page-relative (works at "/" or under a
  // reverse-proxy prefix like "/sharedagentsrt/"). Embedded: realtimeBase() points at
  // the realtime server (the injected realtime_url).
  await openSocket();
  if (gen !== connectGen) return;   // stopped while the socket was opening
  await startMic();
  if (gen !== connectGen) { stopMic(); return; }   // stopped during mic startup
  await unlockAudio();   // resume AudioContext + confirmation beep (mobile)
  if (gen !== connectGen) { stopMic(); return; }
  if (ui.btn) { ui.btn.textContent = "■ disconnect"; ui.btn.classList.add("active"); }
  if (hooks.onConnected) hooks.onConnected(true);

  // Load local wake detection. Missing/disabled/unavailable detection fails
  // closed to explicit push-to-talk; it never enables continuous capture.
  clientCfg = await loadClientConfig();
  if (gen !== connectGen) return;   // stopped while fetching client config
  const w = (clientCfg && clientCfg.wake) || {};
  if (w.preroll_ms != null) prerollMax = Math.max(1, Math.round(w.preroll_ms / CHUNK_MS));
  const wakeOn = (embedWake === null ? w.enabled !== false : embedWake)
                 && typeof WakeWordEngine !== "undefined";
  if (wakeOn) await enterWakeMode();
  else failWakeClosed("local wake detection unavailable");
}

function disconnect(fromClose) {
  // A user-intended teardown (button, or a failed initial connect). This is the
  // ONLY path that stops the reconnect machine and drops the resume handle — an
  // unexpected loss never routes here (it goes to scheduleReconnect instead).
  userClosing = true;
  connectGen++;   // abort any in-flight connect() past its awaits
  invalidateMeetingTransition();
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  reconnecting = false;
  reconnectAttempts = 0;
  outageBuffer = [];
  outageDropped = false;
  gatewaySessionId = null;
  resumeToken = null;
  if (hooks.onReconnecting) try { hooks.onReconnecting(false); } catch (e) {}
  stopWakeEngine();
  stopMic();
  flushPlayback();
  responding = false;
  rejectPendingOpen("", false);
  meetingState = "idle";
  meetingReturnPending = false;
  updateMeetingUI();
  preroll.length = 0;
  const prev = ws;
  ws = null;
  if (prev && !fromClose) { try { prev.close(); } catch (e) {} }
  if (ui.btn) { ui.btn.textContent = "▶ connect"; ui.btn.classList.remove("active"); }
  setStatus("idle");
  if (hooks.onConnected) hooks.onConnected(false);
}

if (ui.btn) ui.btn.onclick = () => {
  // While connected OR mid-reconnect, the button cancels (full teardown). Only
  // from a truly idle state does it start a fresh connect.
  if (meetingState === "active") stopMeeting();
  else if (ws || reconnecting) disconnect();
  else connect().catch((e) => { setStatus("failed: " + e.message); disconnect(); });
};

if (ui.meeting) ui.meeting.onclick = () => {
  if (meetingState === "active") stopMeeting();
  else startMeeting().catch((e) => {
    invalidateMeetingTransition();
    meetingState = "idle";
    updateMeetingUI();
    addTurn("err", "meeting", e.message || String(e));
  });
};

// In fail-closed mode this control is push-to-talk. It cannot switch a healthy
// wake session into continuous capture.
if (ui.mode) {
  ui.mode.onpointerdown = (e) => { e.preventDefault(); beginPushToTalk(); };
  ui.mode.onpointerup = endPushToTalk;
  ui.mode.onpointercancel = endPushToTalk;
  ui.mode.onpointerleave = (e) => { if (e.buttons) endPushToTalk(); };
}

// Pause the wake engine while the tab is backgrounded (saves phone battery) and
// resume on return — resume() also un-suspends a context the OS parked.
document.addEventListener("visibilitychange", () => {
  if (!wakeEngine) return;
  if (document.hidden) wakeEngine.pause();
  else wakeEngine.resume();
});

updateModeUI();
updateMeetingUI();

// Public API for embedding the engine in another page (the main chat). The
// standalone page ignores this and drives itself via #btn. configure() must be
// called before connect() when embedding.
//
// Hooks the embedder may supply (all optional): onConnected(live),
// onReconnecting(active) — fired true when an unexpected
// close starts the auto-reconnect loop and false once reconnected/torn down, so
// the embedder can show a "reconnecting…" state WITHOUT hiding the entry
// control — onStatus, onError, onGatewaySession, onMicStreamStart/Stop, and the
// transcript render hooks.
window.SharedagentsRealtime = {
  configure(opts) {
    opts = opts || {};
    hooks = opts.hooks || {};
    realtimeBaseUrl = opts.url || "";
    realtimeAssetBase = opts.assetBase || "";
    if (typeof opts.wake === "boolean") embedWake = opts.wake;
  },
  connect,
  disconnect,
  connected,
  beginPushToTalk,
  endPushToTalk,
  getMode() { return mode; },
  startMeeting,
  stopMeeting,
};
