/* launch.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */

// ══════════════════════════════════════════════════════════════════════
// Conversation entry & persistence — bootChat() resolves the target session
// from the URL (?new / ?session / ?q) or the remembered last chat and runs
// the page's single boot read (GET /api/boot), then remembers the session id
// across reloads.
// ══════════════════════════════════════════════════════════════════════
const LS_LAST_SESSION = "sharedagents:lastSession";

function persistLastSession(sid) {
  try {
    if (sid) localStorage.setItem(LS_LAST_SESSION, sid);
    else localStorage.removeItem(LS_LAST_SESSION);
  } catch (_e) {
    /* private mode / blocked storage — feature degrades silently */
  }
}
function clearLastSession() {
  persistLastSession(null);
}

// The launch prompt: ?q= (open-webui/ai2), with ?prompt= / ?ai= aliases.
function getLaunchPrompt() {
  try {
    const p = new URLSearchParams(window.location.search);
    return (p.get("q") || p.get("prompt") || p.get("ai") || "").trim();
  } catch (_e) {
    return "";
  }
}

// Resolve a requested agent name against the configured agents, tolerating
// naming variants: exact first, then normalized (lowercase, non-alphanumerics
// stripped) so "self-improver" / "Self Improver" / "self_improver" all map to
// the "self-improver" agent.
function resolveLaunchAgent(raw) {
  const want = String(raw || "").trim();
  if (!want || !Array.isArray(state.agents) || !state.agents.length) return "";
  if (state.agents.includes(want)) return want;
  const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
  const target = norm(want);
  return state.agents.find((a) => norm(a) === target) || "";
}

// Launcher config on the URL (a dashboard command forwards these to the
// embedded UI): ?agent=<name> selects the agent, ?model=<name> the model.
// Applied BEFORE the fresh session is created for ?new / ?q, so the new
// conversation is bound to the requested agent (e.g. research) not "main".
async function applyLaunchConfig() {
  let p;
  try { p = new URLSearchParams(window.location.search); } catch (_e) { return; }
  const rawAgent = String(p.get("agent") || p.get("a") || "").trim();
  const rawModel = String(p.get("model") || p.get("m") || "").trim();
  // sharedagents exposes each AGENT as an OpenAI "model" (sharedagents/<agent>), so the
  // dashboard's OpenAI-style launcher passes the agent as ?model=<agent>
  // (e.g. ?model=research). Resolve the agent from ?agent= first, then from
  // ?model= — both fuzzy-matched so "self-improver" → the "self-improver" agent.
  const agent = resolveLaunchAgent(rawAgent) || resolveLaunchAgent(rawModel);
  const modelIsAgent = !!(rawModel && resolveLaunchAgent(rawModel));
  if (agent && agent !== selectedPickerAgent()) {
    selectPickerAgent(agent);
    try { await loadModelsForCurrentAgent({ loading: true }); } catch (_e) {}
  } else if (rawAgent && !resolveLaunchAgent(rawAgent)) {
    console.warn("[launch] requested agent not available:", rawAgent, "— keeping", selectedPickerAgent());
  }
  // Only treat ?model= as a within-agent model if it did NOT name an agent.
  if (rawModel && !modelIsAgent && Array.isArray(state.models) &&
      state.models.some((x) => (x && (x.id || x.name)) === rawModel || x === rawModel)) {
    selectPickerModel(rawModel);
  }
}

// The left-menu lists the page shows without being asked: the menu is
// already visible at load (desktop, left open earlier in this tab) and the
// section is expanded. Those ride the boot read; a menu hidden at load loads
// them lazily when the operator opens it.
function bootSidebarLists() {
  if (typeof _menuSectionOpen !== "function") return [];
  const lists = [];
  if (_menuSectionOpen("sidebar-conversations")) lists.push("conversations");
  if (_menuSectionOpen("sidebar-topics")) lists.push("topics");
  return lists;
}

// The boot read's key for each list ``?lists=`` names.
const BOOT_LIST_KEYS = { conversations: "sessions", topics: "topics" };

// Install the lists the boot read carried; returns the keys it installed.
// Each list stands alone: rows that cannot be applied leave that list owed.
function applyBootSidebar(sidebar) {
  const lists = sidebar && typeof sidebar === "object" ? sidebar : {};
  const applied = new Set();
  for (const [key, apply] of [["sessions", applySessionListRows], ["topics", applyTopicList]]) {
    if (!Array.isArray(lists[key])) continue;
    try {
      apply(lists[key]);
      applied.add(key);
    } catch (e) {
      console.warn(`[boot] could not apply the ${key} list:`, e);
    }
  }
  return applied;
}

// MenuState runs a visible section's loader as soon as it registers — before
// the boot read has answered. Wait for it:
//   * a list the boot read carried is not fetched again;
//   * a list the boot read OWED (the menu was open at load) but did not
//     carry — boot failed, or the server could not build it — is not fetched
//     either: nobody clicked. Resolving false re-arms MenuState, so the next
//     open of the menu or section (a click) loads it;
//   * any other list (menu opened later — a click) loads as before.
function sidebarListAfterBoot(key, fetchList) {
  const pending = state.bootLists;
  if (!pending) return fetchList();
  return pending.then((boot) => {
    if (boot.applied.has(key)) return true;
    if (boot.owed.delete(key)) return false;
    return fetchList();
  });
}

// The whole chat-page boot: ONE REST read (GET /api/boot) + ONE conversation
// stream — all an untouched chat page may call.
// Resolves the target from the URL or the remembered last chat; with nothing
// to resume it opens a New Chat draft, which makes no request until the
// first send. The read also carries what the page would otherwise fetch on
// its own: the rendered agent's saved default model/effort, and — only when
// the left menu is already visible — the Conversations/Topics lists.
async function bootChat() {
  let params;
  try {
    params = new URLSearchParams(window.location.search);
  } catch (_e) {
    params = new URLSearchParams();
  }
  const wantNew = params.has("new");
  const prompt = wantNew ? "" : getLaunchPrompt();
  const sessionHash = (wantNew || prompt) ? "" : (params.get("session") || "");
  const monitorNav = params.get("monitor") === "1"
    ? { returnSessionId: "", scopeHash: "" } : null;

  // Deep-link owns the tab: never mint a New Chat beside ?session=, and
  // drop any leftover EventSource from a prior `/` login boot.
  const bootEpoch = (state.navEpoch = (state.navEpoch || 0) + 1);
  if (sessionHash) {
    state._openSessionIntent = sessionHash;
    if (typeof closeLiveStream === "function") closeLiveStream();
  } else {
    state._openSessionIntent = "";
  }
  const bootStillCurrent = () => bootEpoch === state.navEpoch;

  // ONE REST read: session (or null) + health + approvals + the picker
  // agent's saved defaults (+ the menu's lists when the menu is showing).
  const query = new URLSearchParams({ session: sessionHash || "" });
  const pickerAgent = selectedPickerAgent();
  if (pickerAgent) query.set("agent", pickerAgent);
  const lists = bootSidebarLists();
  if (lists.length) query.set("lists", lists.join(","));
  const owed = new Set(lists.map((name) => BOOT_LIST_KEYS[name]));
  let applied = new Set();
  let settleLists = () => {};
  state.bootLists = new Promise((resolve) => { settleLists = resolve; });
  const gatewayEpoch = state.gatewayEpoch || 0;
  let boot = { session: null, agents: [], health: {}, approvals: [] };
  let bootAnswered = false;
  try {
    boot = await apiGet(`/api/boot?${query}`);
    bootAnswered = true;
  } catch (_e) { /* web/gateway down — the history fallback below retries */ }
  // Nobody has granted this person an agent yet: every composer says so
  // instead of offering a chat the server would refuse.
  state.viewerLock = boot && boot.viewer && boot.viewer.can_chat === false
    ? String(boot.viewer.chat_block_reason || "No agents are available to you yet.")
    : "";
  if (state.viewerLock) {
    state.monitorReadOnly = true;
    state.readOnlyReason = state.viewerLock;
    if (typeof applyMonitorReadOnlyUi === "function") applyMonitorReadOnlyUi();
  }
  try {
    // Page-global, so applied even when a navigation superseded this boot —
    // unless the operator switched gateway meanwhile: then this read speaks
    // for the old one, and the lists load from the new one instead.
    if ((state.gatewayEpoch || 0) === gatewayEpoch) {
      if (boot && boot.usage_summary) state.usageSummary = boot.usage_summary;
      rememberAgentDefaults(boot.agent_defaults);
      applied = applyBootSidebar(boot.sidebar);
    } else {
      owed.clear();
    }
  } catch (e) {
    console.warn("[boot] could not apply the boot read:", e);
  } finally {
    settleLists({ applied, owed });
  }
  if (!bootStillCurrent()) return state.currentSessionId;
  seedAgents(boot.agents || []);
  // Apply ?agent=/?model= launcher config after the roster is known.
  await applyLaunchConfig();
  if (!bootStillCurrent()) return state.currentSessionId;

  if (wantNew) {
    if (!bootStillCurrent()) return state.currentSessionId;
    const topic = (params.get("topic") || "").trim();
    try { history.replaceState(null, "", window.location.pathname); } catch (_e) {}
    return createSession(topic, { explicit: true, passive: true });
  }
  if (prompt) {
    if (!bootStillCurrent()) return state.currentSessionId;
    try { history.replaceState(null, "", window.location.pathname); } catch (_e) {}
    const ok = await createSession(undefined, { explicit: true, passive: true });
    if (ok) {
      messageInput.value = prompt;
      updateSendState();
      await sendMessage(); // routes prompt (or /command) into the new session
    }
    return state.currentSessionId;
  }
  if (boot.session) {
    if (!bootStillCurrent()) return state.currentSessionId;
    // openSession applies boot.approvals AFTER currentSessionId is set so
    // hydratePendingPermissionCards can mount inline bash-permission cards.
    openSession(boot.session, { boot, monitorNavigation: monitorNav });
    return boot.session.id;
  }
  // A requested ?session= that boot could not resolve must NOT be replaced
  // with a brand-new empty chat — that is the "SSE never connects / empty
  // transcript" failure. Attach SSE to the requested sid so a later bind can
  // still paint. Re-read the history first only when the boot read itself
  // failed: when it answered, GET /api/sessions/<sid> runs the same
  // _session_payload and cannot answer differently.
  if (sessionHash) {
    if (!bootAnswered) {
      try {
        const data = await apiGet(`/api/sessions/${encodeURIComponent(sessionHash)}`);
        if (!bootStillCurrent()) return state.currentSessionId;
        if (data && data.session) {
          openSession(data.session, { boot: Object.assign({}, boot, data), monitorNavigation: monitorNav });
          return data.session.id;
        }
      } catch (_e) { /* fall through to stream-only attach */ }
    }
    if (!bootStillCurrent()) return state.currentSessionId;
    state.currentSessionId = sessionHash;
    persistLastSession(sessionHash);
    if (typeof openLiveStream === "function") openLiveStream(sessionHash, 0);
    if (typeof appendSystemMessage === "function") {
      appendSystemMessage("Could not load this session's history. Live stream attached.", true);
    }
    return sessionHash;
  }
  // Stale or missing last session — open a New Chat draft.
  if (!bootStillCurrent()) return state.currentSessionId;
  clearLastSession();
  return createSession(undefined, { passive: true });
}

// ═══════════════════════════════════════════════════════════════════════════
// Live gateway switching. Additive, self-initialising block
// — no changes to init(). (The cross-device continue-here chip that used to
// share this block was removed.)
// ═══════════════════════════════════════════════════════════════════════════

async function switchGateway() {
  const input = document.getElementById("setting-gateway");
  const status = document.getElementById("gateway-switch-status");
  if (!input) return;
  const url = (input.value || "").trim();
  if (status) status.textContent = "Switching…";
  // A boot read still in flight now speaks for the previous gateway.
  state.gatewayEpoch = (state.gatewayEpoch || 0) + 1;
  try {
    const r = await apiPost("/api/settings/gateway", { url });
    if (status) status.textContent = "Now connected to " + (r.gateway || url);
    // Switching bumps the sessions view to the new gateway with no restart
    // (only when the sidebar list has been opened).
    if (typeof _menuSectionOpen === "function" && _menuSectionOpen("sidebar-conversations"))
      loadSessionList();
  } catch (e) {
    if (status)
      status.textContent = "Switch failed — previous gateway still active";
  }
}
