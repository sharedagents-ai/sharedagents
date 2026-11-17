/* state.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
window.App = window.App || {};

const state = {
  sessions: {}, // session_id → { messages: [...], agent, model, title }
  topics: [], // [{topic, count, sessions: [...]}] from /api/topics (newest-first)
  currentSessionId: null,
  // A delegated session can be observable history rather than an owned chat
  // for a non-local principal. Mutation controls stay
  // disabled while the server sets this flag.
  monitorReadOnly: false,
  // Why the composer is locked for a shared, view-only conversation;
  // empty = the Agent Monitor's subagent view.
  readOnlyReason: "",
  // Set from GET /api/boot when nobody has granted this person an agent yet:
  // every composer stays locked with this reason.
  viewerLock: "",
  // Which cursor/stream namespace produced the mounted history page. This is
  // deliberately independent of monitorReadOnly: the local operator may write
  // to a monitor-opened child while viewing that child's JSONL.
  // Agent Monitor drill-down keeps the parent session as a breadcrumb while a
  // delegated session is open.
  monitorReturnSessionId: "",
  monitorScopeHash: "",
  // User-facing default agent: the SDLC orchestrator jarvis, NOT the internal
  // `main` fallback. A fresh New Chat and any session with
  // no bound agent default here. Kept in sync with the server DEFAULT_AGENT.
  currentAgent: "jarvis",
  // Server-authoritative session switching state.
  sessionRevision: 0,
  // Native conversational default: Allow all (OpenCode --auto, no cards).
  effectiveMode: {
    id: "auto",
    label: "Allow all",
    behavior_kind: "profile_build",
  },
  modePickerTouched: false,
  availableAgentModes: {},
  switchPending: false,
  switchError: "",
  pendingAgent: "", // backend agent transition in progress; never drives the picker
  agentSwitchPromise: null,
  // Composer-owned intent. Backend session/status events never write these;
  // they are applied to the session only when the operator sends a turn.
  pickerAgent: "",
  pickerModels: {}, // agent -> selected model alias
  modelOptionsByAgent: {}, // agent -> last configured option list
  // agent -> true while its list is the server's degraded answer (static
  // fallback or nothing). Such a list cannot prove a pick is unavailable.
  modelCatalogueDegraded: {},
  // agent -> its SAVED default from config.json ({effort, model}), read back
  // from /api/agents/defs/<agent>. See "Set default".
  agentDefaults: {},
  modelLoadingAgent: "", // agent whose /models request is in flight
  agentEngines: {}, // agent -> engine kind (local_llm vs acp/cli)
  // agent -> {origin, external} for the picker's sections.
  // Server-authoritative: ORIGIN decides first, the engine second.
  agentOrigins: {},
  // The footer sheet stages routing separately from the last acknowledged
  // effective tuple. Legacy picker fields above are compatibility projections
  // only; modelSheet is the owner.
  modelSheet: {
    phase: "CLOSED",
    effective: {
      agent: "jarvis", provider: "", alias: "", wire_id: "",
      reasoning: null, speed: null,
    },
    draft: {
      agent: "jarvis", provider: "", alias: "", wire_id: "",
      reasoning: null, speed: null,
    },
    search: "",
    error: "",
  },
  currentModel: "",
  engineModel: "", // real model the agent last ran (claude-opus-4-…/gpt-5.5-codex)
  engineModelAlias: "", // operator-facing alias for engineModel ("" = none configured)
  teacherActive: false, // teacher model is supervising THIS session (goal OR /teacher on)
  teacherEnabled: false, // effective supervision gate echo (== teacherActive server-side)
  teacherModel: "", // the supervising model label (effective)
  activeGoal: "", // goal text (set by /goal <condition>, cleared by /goal clear)
  goalId: "", // durable goal record id (links to /goals/<id>)
  goalSid: "", // session that owns the bound goal id
  goalTerminal: "", // "" | "met" | "blocked" | "paused" — outer-loop terminal state
  goalWait: "", // backend label of why the goal stopped ("" while it runs)
  goalDismissedAfterMet: false, // next ordinary turn hid a completed badge
  goalTurnCount: 0, // re-drive turns consumed by the goal outer loop
  maxGoalTurns: 0, // configured re-drive budget (0 = unknown)
  sessionCostUsd: null, // cumulative session spend (USD); null = no data yet
  budget: null, // {is_subagent, downstream?, total, delegate} from /api/budget
  goalCostUsd: null, // cumulative active-goal spend (USD); null = no data yet
  goalMaxCostUsd: null, // active-goal cost budget (USD); floored to $25 default, never ∞
  statusInfo: {}, // last /status snapshot (agent, model, mode, reasoning, tool, …)
  todos: [], // [{content, status, activeForm}] from /todo (TodoWrite plan)
  todosSid: "", // session id that owns the current todos rows ("" = none)
  harnessTasks: [], // [{content, status}] from /task (harness Task* family)
  harnessTasksSid: "", // session id that owns the current harness-task rows
  subagents: [], // [{id, title, subtitle}] from /subagents + live spawns
  queue: [], // [{queue_id, text, source, author}] pending follow-ups
  _renderedInjects: new Set(), // queue_ids already rendered as user bubbles (dedup)
  queueSubmitting: false, // Web follow-up POST/PATCH currently in flight
  promptSubmitting: false, // first-click lock: ignore extra Send/Enter until POST settles
  voiceSendFlushing: false, // first Send click is draining the armed mic into the composer
  voiceFlushTimeoutMs: 0, // test override; 0 → sendMessage's default 30s drain bound
  steeringQueueId: "", // queue item currently being stopped-and-sent
  hwOpen: { todo: true, task: false, sub: false }, // todo starts expanded at the composer top
  hwScroll: { todo: null, task: null }, // list scrollTop persisted across re-renders
  hwScrolledOnce: { todo: false, task: false }, // first paint scrolls to in_progress
  models: [], // [{ id, name, label }]
  modelLoadEpoch: 0, // latest per-agent model request owns the dropdown
  agents: [], // agent names
  isStreaming: false,
  // True while any session in this conversation's descendant tree is live
  // (background Agent children after the parent turn_ended). Keeps Stop /
  // tree-kill available. Refreshed from /api/agents/tree.
  treeLive: false,
  abortController: null,
  liveSource: null, // EventSource for the OPEN conversation's live stream
  liveSid: null, // session id the live stream is bound to
  liveOffset: 0, // last byte offset seen on the events stream (resume cursor)
  // Live transport. Three poll reasons, only one of them may retry SSE:
  //   • user/URL pin (`sharedagents.web.transport=poll` or ?transport=poll)
  //   • server force (`sharedagents-disable-sse`)
  //   • auto-fallback (`sharedagents.web.transport.auto`) after a cold/blocked stream
  // Auto-fallback starts this tab on poll (no 8s blank) and may shadow-probe
  // SSE later. A user pin never auto-promotes.
  _urlPinReleased: false, // An explicit Stream choice releases the URL pin until reload.
  _userPinnedPoll: (function () {
    try {
      if (typeof location !== "undefined") {
        const q = new URLSearchParams(location.search || "");
        if ((q.get("transport") || "").toLowerCase() === "poll") return true;
      }
      return typeof localStorage !== "undefined"
        && localStorage.getItem("sharedagents.web.transport") === "poll";
    } catch (_e) {
      return false;
    }
  })(),
  _autoPolling: (function () {
    try {
      if (typeof location !== "undefined") {
        const q = new URLSearchParams(location.search || "");
        if ((q.get("transport") || "").toLowerCase() === "poll") return false;
      }
      if (typeof localStorage !== "undefined"
          && localStorage.getItem("sharedagents.web.transport") === "poll") {
        return false;
      }
      return typeof localStorage !== "undefined"
        && localStorage.getItem("sharedagents.web.transport.auto") === "1";
    } catch (_e) {
      return false;
    }
  })(),
  _preferPolling: false,
  _preferFetchSse: false,
  _sseProbeFails: 0,
  _sseProbeBusy: false,
  _sseProbeTimer: null,
  _sseProbeDeadline: null,
  _sseProbeGen: 0,
  _sseProbeSource: null,
  _sseHiddenAt: 0,
  // Web reasoning/tools visibility — gateway still gets STREAM + ON so the
  // cards exist. The gear toggles (visExpand: activity default on, reasoning/tools default off) only expand them:
  // off = shown collapsed; on = expanded (reasoning also pins to the last
  // tokens). Persisted to localStorage so browser preference survives session
  // switches and page reloads. First-turn apply still uses visToApplyBeforeTurn().
  vis: { reasoning: "stream", tools: "on" },
  visExpand: (function () {
    try {
      const a = typeof localStorage !== "undefined" ? localStorage.getItem("sharedagents.web.visExpand.activity") : null;
      const t = typeof localStorage !== "undefined" ? localStorage.getItem("sharedagents.web.visExpand.tools") : null;
      const r = typeof localStorage !== "undefined" ? localStorage.getItem("sharedagents.web.visExpand.reasoning") : null;
      return {
        activity: a === null ? true : a === "1",
        reasoning: r === "1",
        tools: t === "1",
      };
    } catch (_e) {
      return { activity: true, reasoning: false, tools: false };
    }
  })(),
  me: _readViewerIdentity(), // signed-in viewer; optimistic author only
  liveTurnAuthor: null, // {actors, source} of the live turn
  pendingPrompt: null, // transport receipt, settled by POST or canonical echo
  pendingCancel: null, // reused request id until backend settlement
  pendingEcho: null, // optimistically-rendered user prompt awaiting its turn_started
  // Durable physical identity of the most recently rendered turn_started.
  // Reconnect/adoption retries may replay that boundary; rendering it twice
  // would duplicate the operator's user bubble and reset the live assistant.
  livePhysicalTurnId: "",
  // UUIDs that already got turn_ended. Abort replay of the same id must not
  // remount a second card; a later prompt uses a new UUID and must stream.
  closedPhysicalTurnIds: {},
  // UUIDs this tab already sealed with Stop/Steer. Their turn_ended can land
  // after the cancel-settlement timeout, when the next prompt is already
  // live; it must settle quietly instead of closing the newer turn.
  locallyAbortedTurnIds: {},
  liveTurn: null, // per-turn render scratch: { el, bs:WebBlockStream } for the
  // assistant bubble currently being built from canonical events
  // Collapsed Memory-flush card for the in-flight silent flush turn (if any).
  // Set on turn_started when the prompt is ``[MEMORY FLUSH]…``; cleared on
  // turn_ended. Assistant deltas stream into the card's reply slot.
  liveMemoryFlushEl: null,
  navEpoch: 0, // bumped on every session/agent switch; in-flight async
  // loaders bail if it changed (no wrong-session paint)
  _openSessionIntent: "", // URL/boot target sid; blocks a racing New Chat stream

  // ── Image attachments (clipboard paste + 📎 picker) ──
  pastedImages: [], // [{dataUrl, name}] — images staged for the next prompt
  pendingImageReads: [], // in-flight FileReader/downscale promises — sendMessage
  // awaits these so paste→Enter can't drop an image

  // ── Text-file attachments (.md via 📎 picker) ──
  attachedTextFiles: [], // [{content, name, size}] — text files staged for the next prompt
  pendingTextReads: [], // in-flight FileReader promises for text files

  // ── Audio / voice input state ──
  audioState: "idle", // 'idle' | 'recording' | 'processing' | 'unsupported'
  recorder: null, // MediaRecorder instance
  recordingStream: null, // MediaStream from getUserMedia
  audioContext: null, // AudioContext for the live waveform visualizer
  analyserNode: null, // AnalyserNode feeding the waveform

  // ── Realtime voice (inline) state ──
  realtimeState: "idle", // 'idle' | 'connecting' | 'live' | 'unsupported'
  // Once the owner starts realtime in this chat view, keep the Start/Stop
  // control visible after Stop so they can continue without New Chat. Cleared
  // on clearMessages / new session.
  realtimeArmedThisSession: false,
  // Engine capture mode mirrored for the embed UI ('wake' | 'ptt').
  realtimeCaptureMode: "wake",
  realtimeAssistantEl: null, // assistant bubble currently being spoken into
  realtimeAssistantRaw: "", // accumulated assistant transcript for that bubble
  // Agent mode: realtime server bound a gateway (Jarvis) session. Voice I/O
  // still runs; chat bubbles for the voice utterance are painted from the
  // voice path (order + latency), and gateway SSE must not double-paint them.
  realtimeCanonical: false,
  // True from response.created until turn_ended for a voice utterance. Voice
  // paints user+assistant; streaming.js skips turn_started user echo,
  // text/thinking deltas, and turn_ended's ensureLiveAssistant while set —
  // otherwise the chat shows every message twice.
  realtimeVoiceTurn: false,

  // ── Safety: streaming watchdog — if isStreaming stays true for this long
  // without a turn_ended, auto-clear it. Prevents a stuck indicator if the
  // events stream dies mid-turn.
  _streamingWatchdog: null,

  // ── Safety: live-stream liveness watchdog — independent of isStreaming
  // (which only tracks a turn THIS tab started). Re-armed on every frame
  // (log/hb/reconnect) from the open EventSource; if it fires, the stream
  // has gone silent longer than the gateway's own heartbeat cadence should
  // ever allow, so it is forced closed and reopened from the last known
  // offset. Protects tabs that are only OBSERVING a turn driven elsewhere
  // (a background delegate, another interface) — those never set
  // isStreaming, so _streamingWatchdog above never engages for them.
  _liveStreamWatchdog: null,

  lastContextUsed: 0,      // latest prompt_tokens carried by the event stream
  lastContextMax: 0,       // latest context_window carried by the event stream

  // ── Infinite scroll / virtual rendering ──
  msgCache: [], // ordered message data: { id, role, content, el, height }
  scroll: {
    windowStart: 0, // first cache index with a DOM element
    windowEnd: 0, // one-past-last cache index with a DOM element
    totalHeight: 0, // estimated full content height (used for sentinel)
    sentinelTop: null, // invisible <div> above the message list
    observer: null, // IntersectionObserver for sentinel + tail
    autoScroll: true, // true when user is at bottom (follow new messages)
    bottomMarker: null, // invisible element after the last message to detect bottom
    rendering: false, // guard against re-entrant renderWindow calls
    bulkLoading: false, // true while a session's history is replaying/loading —
    // freezes the autoScroll latch so the scroll handler /
    // IntersectionObserver can't flip it false mid-load; a
    // single deterministic force-pin runs when load settles.
    olderUnlocked: false, // user scrolled up — allow unbounded older pages
    fillViewport: false, // after open, fetch a few older pages until the view fills
    viewportFillHops: 0,
  },
};
state._preferPolling = !!(state._userPinnedPoll || state._autoPolling);

// Bulk-load settle: re-pin to the bottom each frame until scrollHeight stops
// growing for STABLE_FRAMES consecutive frames (async markdown/image height
// settle), or until MAX_MS elapses as a hard backstop.
const BULK_STABLE_FRAMES = 8; // ~130ms of no growth = settled
const BULK_SETTLE_MAX_MS = 1500; // hard cap so a never-quiet load still finishes

// ── DOM refs ──

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// The gateway's raw event stream (mode=events) replays `header_update` events
// whose patch may carry the in-memory placeholder title `untitled-<hash8>`
// (emitted when a config save() runs before the turn-derived title is set).
// The server-side projector ignores these (session_index._apply_header_patch);
// the web must mirror that guard or opening a conversation clobbers its real
// sidebar title with "untitled-…".
function isPlaceholderTitle(t) {
  return typeof t === "string" && t.startsWith("untitled-");
}

// Matches sharedagents_session.session_title.is_default_title. Used when merging
// the session LIST so a ``New Chat - HHMMSS`` stub cannot clobber a generated
// title the GET already loaded. The sidebar still DISPLAYS the timestamp
// placeholder until sessions_meta has a real title (test_title_two_phase).
function isDefaultSessionTitle(t) {
  if (t == null) return true;
  const s = String(t).trim();
  if (!s) return true;
  if (s.startsWith("untitled-")) return true;
  if (/^New Chat - \d{6}$/.test(s)) return true;
  if (s.startsWith("New session")) return true;
  if (s === "New Chat" || s === "Untitled" || s === "Session") return true;
  return false;
}

// Public path prefix when reverse-proxied (e.g. "/sharedagents"); empty at root.
// All same-origin API/stream URLs are built through it so the app works under a
// subpath. Read from the <meta name="sharedagents-base"> the server emits (a meta tag
// rather than an inline script so the strict script-src 'self' CSP can't block it).
const BASE =
  (document.querySelector('meta[name="sharedagents-base"]') || {}).content || "";

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}
window.escapeHtml = escapeHtml;


// Signed-in viewer ({email, name, via}) from the server-rendered meta tags;
// null when absent. Only the optimistic echo uses it — the canonical author
// always arrives on turn_started.
function _readViewerIdentity() {
  if (typeof document === "undefined") return null;
  const read = (n) =>
    ((document.querySelector(`meta[name="${n}"]`) || {}).content || "").trim();
  const email = read("sharedagents-user-email").toLowerCase();
  if (!email) return null;
  return { email, name: read("sharedagents-user-name"), via: "web" };
}

// Expose the shared primitives on the explicit App namespace (same object
// refs — App.state === state). Other scripts also reach these via the shared
// classic-script global scope, exactly as when app.js was one file.
window.App.state = state;
window.App.$ = $;
window.App.$$ = $$;
window.App.BASE = BASE;
window.App.isPlaceholderTitle = isPlaceholderTitle;
window.App.isDefaultSessionTitle = isDefaultSessionTitle;
