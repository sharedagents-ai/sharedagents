/* api.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
function applyComposerConnClasses(ok) {
  const live = !!ok;
  const inputWrapper = document.querySelector(".input-wrapper");
  if (inputWrapper) {
    inputWrapper.classList.toggle("conn-live", live);
    inputWrapper.classList.toggle("conn-down", !live);
  }
  const composerCard = document.querySelector(".composer-card");
  if (composerCard) {
    composerCard.classList.toggle("conn-live", live);
    composerCard.classList.toggle("conn-down", !live);
  }
}

function sseIsOpen() {
  const es = state && state.liveSource;
  if (!es) return false;
  if (typeof EventSource !== "undefined" && es.readyState === EventSource.OPEN)
    return true;
  return es.readyState === 1;
}

// The chat page's control stream is the single conversation EventSource (the
// old /api/ui/stream is gone); REST fallback polling keys off it.
function uiSseIsOpen() {
  return sseIsOpen();
}

// ── API ──

async function apiGet(path) {
  const resp = await fetch(BASE + path);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw apiError("GET", path, resp.status, data);
  return data;
}

async function apiPost(path, body) {
  const resp = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw apiError("POST", path, resp.status, data);
  return data;
}

function apiError(method, path, status, data) {
  const detail = data && (data.message || data.error);
  const err = new Error(detail || `${method} ${path} → ${status}`);
  err.status = status;
  err.data = data || {};
  err.code = err.data.code || err.data.error_code || "";
  return err;
}

async function apiDelete(path) {
  const resp = await fetch(BASE + path, { method: "DELETE" });
  if (!resp.ok) {
    // Keep the server's answer so callers can tell its codes apart.
    const error = new Error(`DELETE ${path} → ${resp.status}`);
    error.status = resp.status;
    error.data = await resp.json().catch(() => null);
    throw error;
  }
  return resp.json();
}

async function apiPatch(path, body) {
  const resp = await fetch(BASE + path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const error = new Error(`PATCH ${path} → ${resp.status}`);
    error.status = resp.status;
    error.data = await resp.json().catch(() => null);
    throw error;
  }
  return resp.json();
}

// ── Models & Agents ──

// The gateway's /api/models endpoint returns one entry per AGENT (id
// `sharedagents/<agent>`). That's the agent roster — NOT the model list — so it
// populates only the sidebar agent picker and the settings default-agent box.
function isNonConversationalAgent(name) {
  return /^(explore|general|docs|compaction|title|summary)$/.test(
    String(name || "").trim(),
  );
}

function seedAgents(models) {
  state.agents = (models || [])
    .map((m) => m.name)
    .filter((name) => !isNonConversationalAgent(name));
  // Keep the per-agent description so the picker can say
  // what each agent is FOR instead of listing bare identifiers.
  state.agentDescriptions = {};
  state.agentEngines = {};
  // Sectioning inputs. Server-authoritative: ORIGIN first,
  // engine second, so a user-created agent on an external engine stays in the
  // "User created" section.
  state.agentOrigins = {};
  state.agentPaces = {};
  state.agentCredits = {};
  (models || []).forEach((m) => {
    if (m && m.name) {
      state.agentDescriptions[m.name] = m.description || "";
      state.agentEngines[m.name] = String(m.engine || "").trim();
      state.agentOrigins[m.name] = {
        origin: String(m.origin || "native"),
        external: !!m.external,
      };
      if (m.pace) state.agentPaces[m.name] = m.pace;
      if (m.credits != null) state.agentCredits[m.name] = m.credits;
    }
  });
  const el = document.getElementById("setting-default-agent");
  if (el) {
    el.innerHTML = state.agents
      .map((a) => `<option value="${escapeHtml(a)}">${escapeHtml(a)}</option>`)
      .join("");
  }
  if (typeof renderAgentModeSelect === "function") {
    renderAgentModeSelect(state.pickerAgent || state.currentAgent);
  }
}

async function loadAgents() {
  try {
    const data = await apiGet("/api/models");
    if (data && data.usage_summary) {
      state.usageSummary = data.usage_summary;
    }
    seedAgents(data.models || []);
    if (typeof renderStatusSegments === "function") {
      renderStatusSegments();
    }
  } catch (e) {
    console.error("Failed to load agents:", e);
    if (typeof statusText !== "undefined" && statusText) {
      statusText.textContent = "Gateway unreachable";
    }
  }
}

const PICKER_AGENT_KEY = "sharedagents.web.pickerAgent";
const PICKER_MODELS_KEY = "sharedagents.web.pickerModels";

function ensureModelSheetState() {
  if (!state.modelSheet) {
    const seed = {
      agent: state.pickerAgent || state.currentAgent || "jarvis",
      provider: "", alias: "", wire_id: "", reasoning: null, speed: null,
    };
    state.modelSheet = {
      phase: "CLOSED", effective: { ...seed }, draft: { ...seed },
      search: "", error: "",
    };
  }
  if (state.modelSheet.search == null) state.modelSheet.search = "";
  return state.modelSheet;
}

function cloneModelSelection(value) {
  const v = value || {};
  return {
    agent: String(v.agent || state.currentAgent || "jarvis"),
    provider: String(v.provider || ""),
    alias: String(v.alias || ""),
    wire_id: String(v.wire_id || ""),
    reasoning: v.reasoning == null ? null : String(v.reasoning),
    speed: v.speed == null ? null : String(v.speed),
  };
}

function selectedPickerAgent() {
  const sheet = ensureModelSheetState();
  const selection = sheet.phase && sheet.phase !== "CLOSED"
    ? sheet.draft : sheet.effective;
  return (selection && selection.agent) ||
    state.pickerAgent || state.currentAgent || "jarvis";
}

function isStartablePickerAgent(name) {
  const agent = String(name || "").trim();
  if (!agent) return false;
  // Delegation-only utilities (gateway opencode_mode subagent|hidden).
  // Viewing one of those sessions used to leak the name into New Chat.
  if (isNonConversationalAgent(agent))
    return false;
  if (Array.isArray(state.agents) && state.agents.length)
    return state.agents.includes(agent);
  return true;
}

function startablePickerAgent() {
  const picked = selectedPickerAgent();
  if (isStartablePickerAgent(picked)) return picked;
  if (isStartablePickerAgent(state.pickerAgent)) return state.pickerAgent;
  if (Array.isArray(state.agents) && state.agents.includes("jarvis"))
    return "jarvis";
  if (Array.isArray(state.agents) && state.agents.length)
    return state.agents[0];
  return "jarvis";
}

function adoptStartablePickerAgent(agent) {
  const want = String(agent || "").trim() || "jarvis";
  if (typeof selectPickerAgent === "function" && selectPickerAgent(want))
    return want;
  state.pickerAgent = want;
  ensureModelSheetState();
  const next = cloneModelSelection({
    agent: want,
    alias: (state.pickerModels || {})[want] || "",
  });
  state.modelSheet.effective = next;
  state.modelSheet.draft = cloneModelSelection(next);
  return want;
}

function modelRowsForAgent(agent) {
  const rows = (state.modelOptionsByAgent && state.modelOptionsByAgent[agent]) || [];
  return Array.isArray(rows) ? rows : [];
}

// The agent's list is the server's degraded answer to a failed discovery
// (its static fallback or nothing) — or a partial one (the picker loads one
// provider at a time) — so a pick missing from it may still exist.
function modelCatalogueDegraded(agent) {
  return modelCatalogueFailed(agent) || !!((state.modelCataloguePartial || {})[agent]);
}

// Only the failed-discovery half: a list that is merely partial is still a
// good list and must not be replaced by a fallback answer.
function modelCatalogueFailed(agent) {
  return !!((state.modelCatalogueDegraded || {})[agent]);
}

function modelIdMatchesRow(row, id) {
  const want = String(id || "").trim();
  if (!row || !want) return false;
  return (
    row.id === want ||
    row.realId === want ||
    row.wireId === want ||
    row.alias === want
  );
}

function looksLikeInheritedLocalRoute(agent, id) {
  // Agents that own the global local-LLM / OpenRouter catalogue. Every
  // other name (cursor, opencode, claude, custom ACP) must not display
  // or persist those ids as if they were its own selection.
  const localAgents = {
    jarvis: 1, build: 1, plan: 1, explore: 1, general: 1, main: 1,
  };
  if (localAgents[String(agent || "").trim()]) return false;
  const t = String(id || "").trim();
  if (!t) return false;
  const lower = t.toLowerCase();
  if (lower === "local" || lower === "or-auto") return true;
  if (lower.startsWith("openrouter/")) return true;
  if (lower.startsWith("qwen")) return true;
  return false;
}

function isExternalPickerAgent(name) {
  const n = String(name || "").trim();
  if (!n) return false;
  const engine = String((state.agentEngines || {})[n] || "").trim();
  if (engine === "local_llm") return false;
  if (engine) return true;
  const named = {
    claude: 1, codex: 1, cursor: 1, antigravity: 1,
    opencode: 1, kimi: 1, hermes: 1, devin: 1,
  };
  if (named[n]) return true;
  const desc = String((state.agentDescriptions || {})[n] || "");
  return /\(ext\)/i.test(desc);
}

const LOCAL_ONLY_PROVIDERS = {
  vllm: 1, "openai-compat": 1, configured: 1, unknown: 1,
};
const AGENT_PROVIDERS = {
  antigravity: "antigravity",
  claude: "anthropic",
  codex: "codex",
  cursor: "cursor",
  opencode: "opencode",
  kimi: "kimi",
  hermes: "hermes",
  devin: "devin",
};
const NATIVE_AGENTS = { jarvis: 1, build: 1, plan: 1, review: 1, local_llm: 1 };

function isLocalOnlyProvider(name) {
  return !String(name || "").trim() || !!LOCAL_ONLY_PROVIDERS[String(name).trim().toLowerCase()];
}

/** Catalogue provider for the Agent & model sheet.
 *  vLLM is ONLY the provider of local models (jarvis/build/plan).
 *  External engines keep their own provider — never inherit vLLM. */
function modelPickerProvider(it, agent) {
  const row = it && typeof it === "object" ? it : {};
  const agentKey = String(agent || "").trim().toLowerCase();
  const mapped = AGENT_PROVIDERS[agentKey] || "";
  const explicit = String(row.provider || "").trim();
  const backend = String(row.backend || "").trim();
  const url = String(row.baseUrl || row.body || row.endpoint || "");
  const isLocalUrl = /127\.0\.0\.1|localhost|:8000\b/.test(url);
  const isNative = !agentKey || !!NATIVE_AGENTS[agentKey];

  if (isNative || (isLocalUrl && !mapped)) {
    if (explicit && !isLocalOnlyProvider(explicit)) return explicit;
    const key = backend.toLowerCase();
    if (key && key !== "openai" && key !== "unknown" && key !== "configured") return backend;
    return backend || "vllm";
  }

  if (mapped) {
    // OpenCode advertises real upstream providers on each model id.
    if (agentKey === "opencode" && explicit && !isLocalOnlyProvider(explicit)) {
      return explicit;
    }
    if (mapped === "antigravity" && explicit.toLowerCase() === "google") {
      return "antigravity";
    }
    return mapped;
  }

  if (explicit && !isLocalOnlyProvider(explicit)) return explicit;
  const mid = String(row.id || row.alias || "").trim().toLowerCase();
  if (/cerebras\.ai/i.test(url) || mid.startsWith("cerebras")) return "Cerebras";
  if (mid.startsWith("gemini")) return "antigravity";
  if (mid.startsWith("claude")) return "anthropic";
  if (mid.startsWith("codex:") || /^(gpt-5\.[4-6]|gpt-6-|codex-)/.test(mid)) return "codex";
  if (/^(gpt-|o1|o3|o4)/.test(mid)) return "openai";
  return backend || agentKey || "vllm";
}

function modelMatchesQuery(model, query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return true;
  if (!model) return false;
  const hay = [
    model.id, model.alias, model.label, model.wireId, model.realId,
    model.subtitle, model.provider,
  ];
  return hay.some((part) => String(part || "").toLowerCase().includes(q));
}

function isEngineKindLabel(id) {
  const kinds = {
    local_llm: 1, claude_cli: 1, codex_cli: 1, antigravity_cli: 1,
    acp: 1, cursor_acp: 1, opencode_acp: 1, kimi_acp: 1, hermes_acp: 1,
    devin_acp: 1,
  };
  return !!kinds[String(id || "").trim()];
}

function isModeSentinel(id) {
  const v = String(id || "").trim().toLowerCase();
  return v === "auto" || v === "ask" || v === "plan";
}

function isLocalLlmAgent(agent) {
  const engine = typeof engineForAgent === "function" ? engineForAgent(agent) : "";
  return !engine || engine === "local_llm";
}

function isForeignAgentModel(agent, id) {
  const want = String(id || "").trim();
  const name = String(agent || "").trim();
  if (!want || !name) return false;
  if (isEngineKindLabel(want)) return true;
  if (name !== "cursor" && isModeSentinel(want)) return true;
  const own = modelRowsForAgent(name);
  if (own.some((row) => modelIdMatchesRow(row, want))) return false;
  if (looksLikeInheritedLocalRoute(name, want)) return true;
  // Unknown own catalogue is not "foreign". New Chat lazy-loads models on
  // picker open; if jarvis (or another local agent) already has this id,
  // treating it as stolen hid the composer-chip model on an unused build
  // row even though the Agents footer already showed sess.model.
  if (!own.length) return false;
  const byAgent = state.modelOptionsByAgent || {};
  for (const other of Object.keys(byAgent)) {
    if (other === name) continue;
    if ((byAgent[other] || []).some((row) => modelIdMatchesRow(row, want)))
      return true;
  }
  return false;
}

function acceptedModelLabel(agent, id) {
  const want = String(id || "").trim();
  if (!want || isEngineKindLabel(want) || isForeignAgentModel(agent, want)) return "";
  if (String(agent || "").trim() !== "cursor" && isModeSentinel(want)) return "";
  const row = modelRowsForAgent(agent).find((m) => modelIdMatchesRow(m, want));
  return row ? row.id : (isModeSentinel(want) ? "" : want);
}

function sendableInitialModel(agent, turnModel) {
  const id = String(turnModel || "").trim();
  if (!id || isModeSentinel(id)) return "";
  if (typeof isForeignAgentModel === "function" && isForeignAgentModel(agent, id))
    return "";
  const rows = modelRowsForAgent(agent);
  if (!rows.length) return "";
  if (!rows.some((row) => modelIdMatchesRow(row, id))) return "";
  return id;
}

// The model turn 1 names (``initial_model``). Only an id the agent's catalogue
// vouches for is sent (sendableInitialModel). An untouched page loads no
// catalogue, so a selection other than the agent's SAVED default — a resumed
// placeholder that carries another model — is checked here, at Send (a click,
// where loading the catalogue is allowed). Unchecked, the gateway would start
// turn 1 on the saved default under a chip naming the other model.
//
// A list that failed to load, or the server's degraded answer, cannot rule
// the pick out. Then the pick is named anyway: the gateway validates it and
// refuses a bad id visibly, while omitting it silently started the agent's
// default under a chip that names the pick.
async function firstTurnInitialModel(agent, model) {
  const id = String(model || "").trim();
  if (id && !modelRowsForAgent(agent).length && id !== agentDefaultModel(agent)) {
    try { await loadModelsForAgent(agent); } catch (_e) { /* send without it */ }
  }
  const vouched = sendableInitialModel(agent, id);
  if (vouched || !id) return vouched;
  if (modelRowsForAgent(agent).length && !modelCatalogueDegraded(agent)) return "";
  if (id === agentDefaultModel(agent) || isModeSentinel(id)) return "";
  if (isForeignAgentModel(agent, id)) return "";
  return id;
}

function catalogueDefaultAlias(agent) {
  const options = modelRowsForAgent(agent);
  const saved = typeof agentDefaultModel === "function" ? agentDefaultModel(agent) : "";
  if (saved && typeof rowMatchesModelId === "function") {
    const hit = options.find((m) => rowMatchesModelId(m, saved));
    if (hit) return hit.id;
  }
  const local = isLocalLlmAgent(agent);
  let row = options.find((m) => m.isDefault);
  if (!row && !local) {
    row = options.find((m) => m.id === "auto" || m.realId === "auto");
  }
  if (!row) {
    row = local
      ? options.find((m) => !isModeSentinel(m.id) && !isModeSentinel(m.realId))
      : options[0];
  }
  if (row) return row.id;
  // Cursor advertises Auto as its session default. Show that instead of
  // the backend kind (cursor_acp) while the catalogue is still loading.
  if (String(agent || "").trim() === "cursor") return "auto";
  return "";
}

// Resolve the model id the composer shows for *agent*: the operator's picker
// choice, else the catalogue default. With no catalogue to check either
// against, a selection the page already holds (a resumed session's model)
// stands, and an empty one names the agent's SAVED default model (seeded by
// GET /api/boot) — the model a new session runs when turn 1 names none.
//
// ``opts.passive`` — the page is booting and nobody has clicked anything: an
// untouched chat page makes only its two calls (the boot read + the session
// stream), so a passive resolve never loads a catalogue. A click-driven one
// (New Chat, send) still may.
async function ensurePickerDefaultModel(agent, opts = {}) {
  agent = String(agent || selectedPickerAgent() || "jarvis").trim();
  if (!agent) return "";
  const resolveAlias = () => String(
    (typeof pickerModelAlias === "function" ? pickerModelAlias(agent) : "")
    || (typeof catalogueDefaultAlias === "function" ? catalogueDefaultAlias(agent) : "")
    || "",
  ).trim();
  let alias = resolveAlias();
  if (!alias && !opts.passive) {
    await loadModelsForAgent(agent, { loading: true });
    alias = resolveAlias();
  }
  let unchecked = false;
  if (!alias) {
    const sheet = ensureModelSheetState();
    const held = sheet.phase && sheet.phase !== "CLOSED" ? sheet.draft : sheet.effective;
    const heldAlias = held && held.agent === agent ? String(held.alias || "").trim() : "";
    if (heldAlias) return heldAlias;
    alias = agentDefaultModel(agent);
    unchecked = true;
  }
  if (alias && agent !== "cursor" && typeof isModeSentinel === "function"
      && isModeSentinel(alias))
    alias = "";
  if (alias) {
    state.currentModel = alias;
    // Only a catalogue-checked id becomes the remembered picker choice.
    if (!unchecked) {
      state.pickerModels = state.pickerModels || {};
      state.pickerModels[agent] = alias;
    }
    if (typeof modelSheetSelectionFor === "function") {
      if (typeof ensureModelSheetState === "function") ensureModelSheetState();
      state.modelSheet.effective = modelSheetSelectionFor(agent, alias);
      if (typeof cloneModelSelection === "function")
        state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
    }
  }
  if (typeof renderStatusSegments === "function") renderStatusSegments();
  return alias;
}

function pickerOrSessionModelLabel(agentName) {
  const sess = (state.sessions && state.sessions[state.currentSessionId]) || {};
  const sessionModel = String(state.currentModel || sess.model || "").trim();
  let modelLabel = acceptedModelLabel(agentName, sessionModel);
  if (!modelLabel) {
    const engineLabel = String(state.engineModelAlias || state.engineModel || "").trim();
    const own = modelRowsForAgent(agentName);
    if (engineLabel && own.some((m) => modelIdMatchesRow(m, engineLabel)))
      modelLabel = acceptedModelLabel(agentName, engineLabel);
  }
  if (!modelLabel) {
    const sheet = state.modelSheet && state.modelSheet.effective;
    if (sheet && sheet.agent === agentName
        && modelRowsForAgent(agentName).some((m) => modelIdMatchesRow(m, sheet.alias)))
      modelLabel = acceptedModelLabel(agentName, sheet.alias);
  }
  // Session-stamped id rejected as foreign (another agent's catalogue has it,
  // ours is stale or lacks the row — e.g. a Codex OAuth transport id the
  // native cache hasn't picked up): show the RAW stamped id rather than the
  // catalogue default. The chip must never name a model the session is not
  // actually running. Engine kinds, mode sentinels and inherited local routes
  // still fall through to the default.
  if (!modelLabel) {
    const raw = sessionModel;
    if (raw && !isEngineKindLabel(raw) && !isModeSentinel(raw)
        && !looksLikeInheritedLocalRoute(agentName, raw))
      modelLabel = raw;
  }
  if (!modelLabel)
    modelLabel = String(pickerModelAlias(agentName) || catalogueDefaultAlias(agentName) || "").trim();
  // Persisted picker/sheet values require catalogue validation. While the
  // catalogue is unknown, renderStatusSegments paints the neutral chip.
  return modelLabel;
}

function scrubForeignPickerModels() {
  const saved = state.pickerModels || {};
  for (const agent of Object.keys(saved)) {
    if (isForeignAgentModel(agent, saved[agent])) delete saved[agent];
  }
}

function pickerModelAlias(agent = selectedPickerAgent()) {
  // Never fall back to state.models — that list can still hold another
  // agent catalogue (jarvis) while cursor/opencode options are loading.
  const options = modelRowsForAgent(agent);
  const local = isLocalLlmAgent(agent);
  const sheet = ensureModelSheetState();
  const active = sheet.phase && sheet.phase !== "CLOSED"
    ? sheet.draft : sheet.effective;
  const selected = active && active.agent === agent
    ? active.alias : (state.pickerModels || {})[agent] || "";
  // A session-reported wire id (realId) is a valid selection too — only an
  // alias the catalogue genuinely cannot represent falls back to row 0.
  // A degraded list (failed discovery) proves nothing, so the pick stands.
  // Foreign leftovers (openrouter/auto on cursor) are not a valid selection.
  if (selected && !isForeignAgentModel(agent, selected) &&
      (!local || !isModeSentinel(selected)) &&
      (options.some((m) => modelIdMatchesRow(m, selected)) ||
        modelCatalogueDegraded(agent)))
    return selected;
  return catalogueDefaultAlias(agent);
}

function persistPickerPreferences() {
  try {
    const agent = typeof startablePickerAgent === "function"
      ? startablePickerAgent() : selectedPickerAgent();
    localStorage.setItem(PICKER_AGENT_KEY, agent);
    localStorage.setItem(PICKER_MODELS_KEY, JSON.stringify(state.pickerModels || {}));
  } catch (_e) {}
}

function initializePickerPreferences() {
  ensureModelSheetState();
  let savedAgent = "";
  let savedModels = {};
  try {
    savedAgent = localStorage.getItem(PICKER_AGENT_KEY) || "";
    savedModels = JSON.parse(localStorage.getItem(PICKER_MODELS_KEY) || "{}");
  } catch (_e) {}
  // The roster has not loaded yet on a cold boot (initializePickerPreferences
  // runs before any roster/boot). An empty roster is "unknown", not "the
  // saved agent is gone" — discarding the saved pickerAgent there blanked the
  // model chip, because every later lookup keyed on the wrong agent. Keep the
  // saved selection until the roster can actually validate it.
  state.pickerAgent = (!state.agents.length && savedAgent)
    ? savedAgent
    : state.agents.includes(savedAgent)
      ? savedAgent
      : (state.agents.includes(state.currentAgent) ? state.currentAgent : state.agents[0] || "jarvis");
  state.pickerModels = savedModels && typeof savedModels === "object"
    ? savedModels : {};
  scrubForeignPickerModels();
  const alias = state.pickerModels[state.pickerAgent] || "";
  state.modelSheet.effective = cloneModelSelection({
    agent: state.pickerAgent, alias,
  });
  state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
}

function selectPickerAgent(agent) {
  agent = String(agent || "").trim();
  if (!agent || !state.agents.includes(agent)) return false;
  const priorAgent = state.pickerAgent || state.currentAgent || "";
  if (priorAgent && priorAgent !== agent) state.modePickerTouched = false;
  ensureModelSheetState();
  if (state.modelSheet.phase === "APPLYING") return false;
  const sheetOpen = state.modelSheet.phase !== "CLOSED";
  const next = cloneModelSelection(sheetOpen
    ? state.modelSheet.draft : state.modelSheet.effective);
  const models = state.modelOptionsByAgent[agent] || [];
  // Seed from the agent's SAVED default model (its config.json), then the
  // catalogue's own default flag, then the first row.
  const defaultRow = agentDefaultRow(agent, models);
  const savedDef = typeof agentDefaultModel === "function" ? agentDefaultModel(agent) : "";
  next.agent = agent;
  next.provider = defaultRow ? (defaultRow.provider || "") : "";
  next.alias = defaultRow ? (defaultRow.id || "") : (savedDef || "");
  next.wire_id = defaultRow ? (defaultRow.wireId || "") : (savedDef || "");
  next.reasoning = null;
  next.speed = null;
  state.modelSheet.searchByAgent = state.modelSheet.searchByAgent || {};
  state.modelSheet.search = state.modelSheet.searchByAgent[agent] || "";
  const sid = state.currentSessionId;
  const unused = !!(
    sid
    && typeof isFreshUnboundSession === "function"
    && isFreshUnboundSession(sid)
  );
  if (sheetOpen) {
    state.modelSheet.draft = next;
    if (typeof setModelSheetPhase === "function")
      setModelSheetPhase("OPEN_DIRTY");
    else state.modelSheet.phase = "OPEN_DIRTY";
    // Unused New Chat: the tap is the session agent. Commit effective now
    // so Close cannot copy the previous agent back over the footer.
    if (unused) state.modelSheet.effective = cloneModelSelection(next);
  } else {
    state.modelSheet.effective = next;
    state.modelSheet.draft = cloneModelSelection(next);
    if (next.alias) {
      state.currentModel = next.alias;
      state.pickerModels = state.pickerModels || {};
      state.pickerModels[agent] = next.alias;
      if (typeof persistPickerPreferences === "function") persistPickerPreferences();
    }
  }
  state.pickerAgent = agent;
  state.models = state.modelOptionsByAgent[agent] || [];
  if (typeof renderAgentSelect === "function") renderAgentSelect();
  if ((!sheetOpen || unused) && typeof renderStatusSegments === "function") renderStatusSegments();
  if (state.models.length) renderModelSelect();
  else renderModelSelect(true);
  // Unused New Chat: the picker IS the session agent. Update currentAgent
  // and the Agent Monitor now — do not wait for Save or Send. A later
  // state frame that still has leftover currentAgent (cursor) would
  // otherwise stamp the footer backward.
  if (unused && typeof applyFreshAgentModeIntent === "function") {
    applyFreshAgentModeIntent(sid, agent, "");
    if (typeof apiPost === "function") {
      apiPost(`/api/sessions/${encodeURIComponent(sid)}/agent`, {
        target_agent: agent,
        frontend: "web",
      }).catch(function () { /* local chrome already updated */ });
    }
  }
  if (typeof updateAgentMonAccordion === "function")
    updateAgentMonAccordion();
  // Refresh the agent's summary (default row + provider tabs) in the
  // background — no provider is asked. With the Model view open, its shown
  // provider loads next. No bound session id is sent, so backend session
  // state cannot choose or rewrite this frontend setting.
  Promise.resolve(loadModelsForAgent(agent, { loading: !state.models.length })).then(function () {
    if (sheetOpen && state.modelSheet.activeTab === "model" &&
        typeof ensureSheetProviderLoaded === "function")
      ensureSheetProviderLoaded();
  }).catch(function () { /* the summary keeps what the page holds */ });
  return true;
}

function selectPickerModel(id) {
  id = String(id || "").trim();
  const agent = selectedPickerAgent();
  ensureModelSheetState();
  if (state.modelSheet.phase === "APPLYING") return false;
  if (!id || !state.models.some((m) => m.id === id)) return false;
  const model = state.models.find((m) => m.id === id);
  const next = cloneModelSelection(state.modelSheet.phase !== "CLOSED"
    ? state.modelSheet.draft : state.modelSheet.effective);
  next.agent = agent;
  next.provider = model.provider;
  next.alias = model.id;
  next.wire_id = model.wireId;
  next.reasoning = reasoningForSelectedModel(model.reasoning, agent, model);
  next.speed = supportedDraftValue(model.speed, next.speed);
  if (state.modelSheet.phase !== "CLOSED") {
    state.modelSheet.draft = next;
    if (typeof setModelSheetPhase === "function")
      setModelSheetPhase("OPEN_DIRTY");
    else state.modelSheet.phase = "OPEN_DIRTY";
  } else {
    state.modelSheet.effective = next;
    state.modelSheet.draft = cloneModelSelection(next);
  }
  state.pickerModels[agent] = id;
  if (typeof modelSelect !== "undefined" && modelSelect) modelSelect.value = id;
  if (state.modelSheet.phase === "CLOSED") {
    persistPickerPreferences();
    renderStatusSegments();
  }
  if (typeof renderModelSheet === "function") renderModelSheet();
  return true;
}

function normalizeAdvancedCapability(raw, kind = "reasoning") {
  const isObject = raw && typeof raw === "object" && !Array.isArray(raw);
  if (!isObject) return null;
  const values = Array.isArray(raw.values)
    ? raw.values.filter((v) => typeof v === "string" && v.length) : [];
  if (!values.length) return null;
  const fallback = raw.default == null ? null : String(raw.default);
  return {
    values,
    default: values.includes(fallback) ? fallback : null,
    applyable: kind === "reasoning" || !!String(raw.wire || "").trim(),
    source: String(raw.source || ""),
    applies: typeof raw.applies === "boolean" ? raw.applies : undefined,
  };
}

function supportedDraftValue(capability, value) {
  if (!capability) return null;
  const current = value == null ? "" : String(value);
  return capability.values.includes(current) ? current : capability.default;
}

// ── The agent's SAVED default model + effort ──────────────────────────────
//
// "Set default" in the Agent & Model sheet PATCHes the per-agent
// ``config.json`` (agent_editor ``PATCHABLE_KEYS``: ``upstream_model`` +
// ``effort``), and the runtime treats that file as the agent-wide default —
// both wire builders resolve ``session.effort or cfg.effort`` before any
// model-row vocabulary is consulted (worker_request.py, turn_execution.py).
// The selector therefore has to READ IT BACK: without it every reload
// re-derived the displayed effort from the model row alone, so a saved
// ``xhigh`` re-rendered (and was applied) as the row's own ``medium`` — the
// saved default looked like it had never persisted.
//
// Read fail-OPEN: an unknown agent or an unreachable gateway leaves the
// model row's advertised default standing, exactly as before this read-back
// existed. Never invent an effort.
function ensureAgentDefaults() {
  if (!state.agentDefaults || typeof state.agentDefaults !== "object")
    state.agentDefaults = {};
  return state.agentDefaults;
}

function agentDefaultEffort(agent) {
  const name = String(agent || "").trim();
  if (!name) return "";
  const entry = ensureAgentDefaults()[name];
  return entry && typeof entry.effort === "string" ? entry.effort : "";
}

// "128K ctx" from a catalogue row's ``contextWindow``. The server also ships a
// short badge string for the same number (command_runtime), used only as a
// fallback so the tile still says something when a row has no window.
function modelContextLabel(row) {
  const ctx = Number((row && row.contextWindow) || 0);
  if (ctx > 0) {
    const k = ctx / 1024;
    return (k >= 1000 ? (k / 1024).toFixed(1) + "M" : Math.round(k) + "K") + " ctx";
  }
  const badge = String((row && row.badge) || "").trim();
  return badge ? badge + " ctx" : "";
}

// One line naming WHERE a model row runs: provider, then the wire backend.
function modelRouteLabel(row) {
  const provider = String((row && row.provider) || "").trim();
  const backend = String((row && row.backend) || "").trim();
  if (provider && backend && backend.toLowerCase() !== provider.toLowerCase())
    return provider + " · " + backend;
  return provider || backend;
}

// The agent's SAVED default MODEL — the ``model`` half of the same
// ``/api/agents/defs/<agent>`` read-back (the id the runtime sends on the
// wire for a new session: ``cfg.upstream_model``).
function agentDefaultModel(agent) {
  const name = String(agent || "").trim();
  if (!name) return "";
  const entry = ensureAgentDefaults()[name];
  return entry && typeof entry.model === "string" ? entry.model.trim() : "";
}

// Does *row* denote the model *id* the operator saved? Row identity is
// carried in four fields across the catalogue shapes (id / alias / wireId /
// realId) — a saved ``upstream_model`` may be any of them.
function rowMatchesModelId(row, id) {
  const want = String(id || "").trim();
  if (!row || !want) return false;
  return (
    String(row.id || "") === want ||
    String(row.alias || "") === want ||
    String(row.wireId || "") === want ||
    String(row.realId || "") === want
  );
}

// Resolve the catalogue row the sheet should seed for *agent*.
//
//   1. the model SAVED for this agent (``config.json`` → ``upstream_model``,
//      read back through ``/api/agents/defs/<agent>``),
//   2. the catalogue's own explicit ``default: true`` row,
//   3. the first row (legacy behaviour, unchanged).
//
// Step 1 is not redundant with step 2: the runtime stamps ``default: true``
// only on the NATIVE discovery branch (``cache_lock.discover_agent_models``);
// external CLI/ACP agents publish no flag at all, and the client's documented
// index-0 fallback then made "select the agent" land on whichever row the
// catalogue listed first instead of the model the operator saved (a custom
// agent saved on a hosted model rendered the catalogue's first local row,
// whose effort vocabulary then silently downgraded the saved ``high``).
function agentDefaultRow(agent, models) {
  const rows = Array.isArray(models) ? models : [];
  if (!rows.length) return undefined;
  const saved = agentDefaultModel(agent);
  if (saved) {
    const hit = rows.find((m) => rowMatchesModelId(m, saved));
    if (hit) return hit;
  }
  return rows.find((m) => m.isDefault) || rows[0];
}

// The cached shape of one ``/api/agents/defs/<agent>`` read-back.
function agentDefaultEntry(def) {
  return {
    effort: typeof def.effort === "string" ? def.effort.trim().toLowerCase() : "",
    model: String(def.model || ""),
  };
}

// Seed the cache from GET /api/boot, which carries that same read-back for
// the agent(s) the page renders — an untouched page makes no def request.
function rememberAgentDefaults(defs) {
  if (!defs || typeof defs !== "object") return;
  const cache = ensureAgentDefaults();
  for (const [name, def] of Object.entries(defs)) {
    if (!name || cache[name] || !def || typeof def !== "object") continue;
    cache[name] = agentDefaultEntry(def);
    reconcileAgentDefaultEffort(name);
  }
}

// Cache the default the operator JUST saved so the sheet keeps showing (and
// applying) what was persisted, with no extra round trip.
function recordAgentDefaultEffort(agent, effort, model) {
  const name = String(agent || "").trim();
  if (!name) return "";
  const value = effort == null ? "" : String(effort).trim().toLowerCase();
  ensureAgentDefaults()[name] = { effort: value, model: String(model || "") };
  if (typeof reconcileAgentDefaultEffort === "function")
    reconcileAgentDefaultEffort(name);
  return value;
}

// Fetch the agent's saved default (once per agent) from the Agent Builder's
// own authoritative endpoint and reconcile the sheet with it.
async function hydrateAgentDefaultEffort(agent, options = {}) {
  const name = String(agent || "").trim();
  if (!name) return "";
  const cache = ensureAgentDefaults();
  const entry = cache[name];
  if (entry && !options.force) return entry.effort || "";
  let effort = "";
  let model = "";
  try {
    const def = await apiGet("/api/agents/defs/" + encodeURIComponent(name));
    if (def && typeof def === "object") {
      const entry = agentDefaultEntry(def);
      effort = entry.effort;
      model = entry.model;
    }
  } catch (_err) {
    // Gateway unreachable / agent not configured: never invent an effort and
    // never cache the miss (a later load retries). The row's own default
    // stands — exactly the behavior before this read-back existed.
    if (typeof reconcileAgentDefaultEffort === "function")
      reconcileAgentDefaultEffort(name);
    return entry ? (entry.effort || "") : "";
  }
  const prior = entry && typeof entry.effort === "string" ? entry.effort : "";
  cache[name] = { effort, model };
  if (prior !== effort && typeof reconcileAgentDefaultEffort === "function")
    reconcileAgentDefaultEffort(name);
  return effort;
}

// Refresh an untouched sheet from its server-resolved row after hydration.
function reconcileAgentDefaultEffort(agent) {
  ensureModelSheetState();
  const phase = state.modelSheet.phase;
  const selected = phase === "CLOSED"
    ? state.modelSheet.effective
    : (phase === "OPEN_CLEAN" ? state.modelSheet.draft : null);
  if (!selected || String(selected.agent || "") !== String(agent)) return false;
  const rows = modelRowsForAgent(agent);
  const row = rows.find((m) => m.id === selected.alias ||
      (selected.wire_id && m.wireId === selected.wire_id)) ||
    agentDefaultRow(agent, rows);
  if (!row || !row.reasoning) return false;
  const next = typeof reasoningForSelectedModel === "function"
    ? reasoningForSelectedModel(row.reasoning, agent, row)
    : null;
  if (next == null || next === selected.reasoning) return false;
  const updated = cloneModelSelection({ ...selected, reasoning: next });
  if (phase === "CLOSED") {
    state.modelSheet.effective = updated;
    state.modelSheet.draft = cloneModelSelection(updated);
  } else {
    state.modelSheet.draft = updated;
  }
  if (typeof renderModelSheet === "function") renderModelSheet();
  if (typeof renderStatusSegments === "function") renderStatusSegments();
  return true;
}

// The server resolves session/model/agent precedence for each picker row.
function reasoningForSelectedModel(_capability, _agent, row) {
  return row ? row.effortSelected : null;
}

// Build the one acknowledged /model transaction used by both the sheet Save
// path and the pre-turn apply path. The selected level is always sent
// explicitly: the server stores it concretely and remembers it for the model.
function modelApplyCommand(selection, row, _previousReasoning = null) {
  const target = String(
    (selection && (selection.wire_id || selection.alias)) || "",
  ).trim();
  if (!target) return "";
  let command = `/model ${target}`;
  const reasoning = selection && selection.reasoning != null
    ? String(selection.reasoning) : null;
  if (row && row.reasoning && reasoning != null) {
    command += ` effort=${encodeURIComponent(reasoning || "default")}`;
  }
  const speed = selection && selection.speed != null
    ? String(selection.speed) : null;
  if (row && row.speed && row.speed.applyable && speed != null) {
    command += ` speed=${encodeURIComponent(speed)}`;
  }
  return command;
}

// One picker row per server row — the single mapping for every source (a
// provider's rows, the picker summary's default rows, the stream's catalogue
// frame, the legacy /models command). The server names the provider tab each
// row sits under (``tab``); rows without one (the /models command) keep the
// older provider mapping.
function normalizeModelRow(it, agent) {
  it = it || {};
  return {
    id: it.id || it.alias,
    alias: it.alias || it.id,
    label: it.display || it.title || it.label || it.id,
    subtitle: it.subtitle || "",
    wireId: String(it.wire_id || it.model_id || it.id || ""),
    realId: String(it.wire_id || it.model_id || it.id || ""),
    provider: it.tab ? String(it.tab) : modelPickerProvider(it, agent),
    active: !!it.active,
    serverDefault: it.default === true,
    isDefault: false,
    badge: (it.badges || [])[0] || "",
    contextWindow: it.contextWindow || 0,
    // The tile names the route (provider · backend) and the exact id that
    // goes on the wire — the two things the operator needs to know what
    // they are actually selecting.
    backend: String(it.backend || ""),
    // The level the server remembers for this provider/model ("" = none).
    effortPref: String(it.effort_pref || ""),
    effortSelected: it.effort_selected == null ? null : String(it.effort_selected),
    maxTokens: Number(it.maxTokens || it.max_tokens || 0) || 0,
    reasoning: normalizeAdvancedCapability(
      it.selection_capabilities
        ? it.selection_capabilities.reasoning
        : (it.capabilities && !Array.isArray(it.capabilities)
          ? it.capabilities.reasoning : (it.reasoning || it.effort)),
      "reasoning",
    ),
    speed: normalizeAdvancedCapability(
      it.selection_capabilities
        ? it.selection_capabilities.speed
        : (it.capabilities && !Array.isArray(it.capabilities)
          ? it.capabilities.speed : it.speed),
      "speed",
    ),
  };
}

// ``isDefault`` over a whole list. A payload naming its own default row must
// SUPPRESS the index-0 fallback — with both applied, ``find(m => m.isDefault)``
// returned the catalogue's first row instead of the agent's saved default
// (row 0 AND the saved row further down were both marked default).
function markDefaultModelRows(agent, models) {
  const savedDefault = typeof agentDefaultModel === "function" ? agentDefaultModel(agent) : "";
  const hasSavedMatch = !!savedDefault && models.some((m) => rowMatchesModelId(m, savedDefault));
  const explicitDefault = hasSavedMatch || models.some((m) => m.serverDefault);
  models.forEach((m, index) => {
    if (hasSavedMatch) m.isDefault = rowMatchesModelId(m, savedDefault);
    else if (m.serverDefault) m.isDefault = true;
    else m.isDefault = !explicitDefault && index === 0;
  });
  return models;
}

function normalizeModelRows(agent, items) {
  return markDefaultModelRows(agent, (items || []).map((it) => normalizeModelRow(it, agent)));
}

// Store an agent's rows. ``replaceProviders`` names the provider tabs whose
// rows ``models`` replace (a provider just loaded); every other row stays, so
// loading one provider never drops another provider's models.
function storeAgentModelRows(agent, models, replaceProviders) {
  const replace = new Set(replaceProviders || []);
  const ids = new Set(models.map((m) => m.id));
  const kept = modelRowsForAgent(agent)
    .filter((m) => !replace.has(m.provider) && !ids.has(m.id));
  const merged = markDefaultModelRows(agent, kept.concat(models));
  state.modelOptionsByAgent[agent] = merged;
  if (selectedPickerAgent() === agent) state.models = merged;
  return merged;
}

// What the picker summary says about an agent: its default row, its saved
// defaults and its provider tabs (``{id, label, status, error?}``).
function pickerAgentSummary(agent) {
  return ((state.pickerSummary || {})[String(agent || "")]) || null;
}

function pickerProviderTabs(agent) {
  const summary = pickerAgentSummary(agent);
  return summary && Array.isArray(summary.providers) ? summary.providers : [];
}

// Provider tabs loaded while the sheet is open: ``{agent: {tab: {phase,
// error}}}`` (phase ``loading`` | ``loaded`` | ``failed``). Reset when the
// sheet opens, so each opening asks a provider again only when its tab is
// clicked (the server's own cache decides whether that reaches the provider).
function sheetProviderLoads(agent) {
  const sheet = ensureModelSheetState();
  sheet.providerLoads = sheet.providerLoads || {};
  const key = String(agent || "");
  sheet.providerLoads[key] = sheet.providerLoads[key] || {};
  return sheet.providerLoads[key];
}

// A list is partial until every provider tab of the agent has loaded (or the
// stream's full catalogue arrived): a pick missing from it may still exist.
function refreshModelCataloguePartial(agent) {
  state.modelCataloguePartial = state.modelCataloguePartial || {};
  if ((state.modelCatalogueFull || {})[agent]) {
    delete state.modelCataloguePartial[agent];
    return;
  }
  const tabs = pickerProviderTabs(agent).filter((t) => t.status !== "error");
  const loaded = (state.providerLoadedEver || {})[agent] || {};
  const complete = tabs.length > 0 && tabs.every((t) => loaded[t.id]);
  if (complete) delete state.modelCataloguePartial[agent];
  else state.modelCataloguePartial[agent] = true;
}

// Apply GET /api/agents/picker rows: roster (when the whole list came back),
// saved defaults, provider tabs, and each agent's default model row.
function applyPickerSummary(rows, options = {}) {
  if (!Array.isArray(rows)) return;
  state.pickerSummary = state.pickerSummary || {};
  if (options.roster && rows.length && typeof seedAgents === "function") seedAgents(rows);
  const defaults = {};
  rows.forEach((row) => {
    const name = String((row && row.name) || "").trim();
    if (!name) return;
    const def = row.default ? normalizeModelRow(row.default, name) : null;
    state.pickerSummary[name] = {
      providers: Array.isArray(row.providers) ? row.providers : [],
      default: def,
    };
    if (row.defaults && typeof row.defaults === "object") defaults[name] = row.defaults;
  });
  if (typeof rememberAgentDefaults === "function") rememberAgentDefaults(defaults);
  rows.forEach((row) => {
    const name = String((row && row.name) || "").trim();
    const summary = state.pickerSummary[name];
    if (!summary) return;
    const had = modelRowsForAgent(name).length;
    if (summary.default && !modelRowsForAgent(name).some((m) => m.id === summary.default.id))
      storeAgentModelRows(name, [summary.default], []);
    else if (had) markDefaultModelRows(name, modelRowsForAgent(name));
    refreshModelCataloguePartial(name);
  });
}

async function loadPickerSummary(agent) {
  const name = String(agent || "").trim();
  const params = [];
  if (name) params.push("agent=" + encodeURIComponent(name));
  if (state.currentSessionId) params.push("session_id=" + encodeURIComponent(state.currentSessionId));
  const data = await apiGet("/api/agents/picker" + (params.length ? "?" + params.join("&") : ""));
  applyPickerSummary((data && data.agents) || [], { roster: !name });
  return data;
}

// Apply one provider's answer (POST /api/models/discover with ``provider``).
// A failed or empty answer never replaces rows the agent already holds.
function applyProviderModels(agent, provider, resp) {
  const loads = sheetProviderLoads(agent);
  const summary = pickerAgentSummary(agent);
  if (summary && Array.isArray(resp.providers) && resp.providers.length)
    summary.providers = resp.providers;
  else if (!summary && Array.isArray(resp.providers))
    (state.pickerSummary = state.pickerSummary || {})[agent] = { providers: resp.providers, default: null };
  const loaded = Array.isArray(resp.loaded) && resp.loaded.length ? resp.loaded : [provider];
  const models = normalizeModelRows(agent, resp.models || []);
  const degraded = !!resp.degraded;
  const error = resp.error && resp.error.message ? resp.error : null;
  const keepCached = (degraded || !models.length) &&
    modelRowsForAgent(agent).some((m) => loaded.includes(m.provider)) &&
    !modelCatalogueFailed(agent);
  if (!keepCached && models.length) {
    storeAgentModelRows(agent, models, loaded);
    state.modelCatalogueDegraded = state.modelCatalogueDegraded || {};
    if (degraded) state.modelCatalogueDegraded[agent] = true;
    else delete state.modelCatalogueDegraded[agent];
  } else if (!models.length && degraded && !modelRowsForAgent(agent).length) {
    state.modelCatalogueDegraded = state.modelCatalogueDegraded || {};
    state.modelCatalogueDegraded[agent] = true;
  }
  state.providerLoadedEver = state.providerLoadedEver || {};
  const ever = state.providerLoadedEver[agent] = state.providerLoadedEver[agent] || {};
  loaded.forEach((tab) => {
    const failed = !!error && tab === (resp.provider || provider);
    loads[tab] = { phase: failed ? "failed" : "loaded", error };
    if (!failed || !degraded) ever[tab] = true;
  });
  if (resp.provider && resp.provider !== provider) loads[provider] = loads[resp.provider];
  refreshModelCataloguePartial(agent);
}

// Load what the picker needs for *agent*.
//
//   * ``options.provider`` — that provider tab's models, asking only that
//     provider (POST /api/models/discover {agent, provider}). Called when the
//     tab (or the Model view showing it) is opened.
//   * otherwise — the agent's picker summary (GET /api/agents/picker): its
//     default model row and provider tabs. No provider is asked, so New Chat,
//     Send and an agent pick only ever cost this cheap read.
//
// This picker is frontend-owned, so model discovery ALWAYS queries without a
// bound session id. A session-bound `/models` response carries backend-active
// state and must never be allowed to choose or reset the operator's setting.
async function loadModelsForAgent(agent, options = {}) {
  if (state.monitorReadOnly) {
    state.models = [];
    renderModelSelect();
    return;
  }
  agent = String(agent || "").trim();
  if (!agent) return;
  const provider = String(options.provider || "").trim();
  let requestEpoch;
  let current;
  if (provider) {
    state.providerLoadEpoch = state.providerLoadEpoch || {};
    const key = agent + "\u0000" + provider;
    requestEpoch = Number(state.providerLoadEpoch[key] || 0) + 1;
    state.providerLoadEpoch[key] = requestEpoch;
    current = () => state.providerLoadEpoch[key] === requestEpoch;
    sheetProviderLoads(agent)[provider] = { phase: "loading", error: null };
  } else {
    state.modelLoadEpoch = Number(state.modelLoadEpoch || 0) + 1;
    requestEpoch = state.modelLoadEpoch;
    current = () => requestEpoch === state.modelLoadEpoch;
    state.modelLoadingAgent = agent;
  }
  const cached = modelRowsForAgent(agent);
  // Keep any cached rows visible. Wiping them made the sheet flash
  // "No models advertised for this provider" for the whole /models
  // wait (Cursor discovery is a 1-8s cursor-agent spawn on a cold cache).
  if (options.loading && selectedPickerAgent() === agent && !cached.length)
    renderModelSelect(true);
  if (typeof renderModelSheet === "function") renderModelSheet();
  try {
    if (provider) {
      // The agent's SAVED default effort is part of what a catalogue load
      // resolves (the sheet's draft, the composer's effective tuple the next
      // session inherits, and the "Set default" round trip all read it). One
      // cached fetch per agent; a failed read leaves the model's own default
      // standing. The summary path carries it.
      await hydrateAgentDefaultEffort(agent);
      const resp = await apiPost("/api/models/discover", {
        agent: agent,
        provider: provider,
        force: !!options.force,
        session_id: state.currentSessionId,
      });
      if (!current()) return;
      applyProviderModels(agent, provider, resp || {});
    } else {
      // A failed summary keeps the last view; never turn it into a provider
      // discovery or a gateway /models command behind the operator's back.
      await loadPickerSummary(agent);
      if (!current()) return;
      // Recompute after the defaults read-back: the saved default row wins.
      markDefaultModelRows(agent, modelRowsForAgent(agent));
    }
    if (state.modelLoadError && state.modelLoadError.agent === agent)
      state.modelLoadError = null;
  } catch (e) {
    if (!current()) return;
    if (provider) {
      sheetProviderLoads(agent)[provider] = { phase: "failed", error: null, transport: true };
    }
    state.modelLoadError = {
      agent,
      provider,
      message: String((e && e.message) || "Could not load models"),
    };
    if (selectedPickerAgent() === agent && !state.modelOptionsByAgent[agent])
      state.models = [];
  } finally {
    if (!provider && current() && state.modelLoadingAgent === agent)
      state.modelLoadingAgent = "";
  }
  if (!current()) return;
  reconcileDraftWithLoadedModels(agent);
}

// After rows arrive, re-resolve the open (or closed) selection onto a real row.
function reconcileDraftWithLoadedModels(agent) {
  if (selectedPickerAgent() !== agent) {
    if (typeof renderModelSheet === "function") renderModelSheet();
    return;
  }
  const draft = state.modelSheet.draft;
  if (draft && draft.agent === agent && state.modelSheet.phase !== "APPLYING") {
    const loaded = state.modelOptionsByAgent[agent] || [];
    // Match the preserved selection by id OR real wire id (the session may
    // report a wire id the stale catalogue could not resolve) before any
    // default-row fallback — same resolution a full reload performs in
    // modelSheetSelectionFor.
    const aliasHit = loaded.find((m) => rowMatchesModelId(m, draft.alias)) || null;
    // A degraded list cannot show the pick is gone: keep it, unresolved.
    const keepPick = !aliasHit && !!draft.alias && modelCatalogueDegraded(agent);
    const chosen = keepPick ? null : (aliasHit || agentDefaultRow(agent, loaded));
    if (chosen) {
      // A re-resolved OPERATOR selection keeps the provider tab they were
      // working in; an unresolved one (fresh agent seed — empty or stale
      // alias) adopts the resolved default row's provider, so the sheet
      // lands on THAT row instead of the provider's first card. Seeding
      // jarvis-vllm leftovers onto a Codex-defaulted agent used to show
      // gpt-6-astra for a saved gpt-6-sol.
      // A provider tab the operator just opened stays open while it loads.
      const opening = state.modelSheet.pendingStageProvider &&
        state.modelSheet.pendingStageProvider === draft.provider;
      draft.provider = opening || (aliasHit && loaded.some((m) => m.provider === draft.provider))
        ? draft.provider : (chosen.provider || draft.provider);
      const visible = loaded.filter((m) => m.provider === draft.provider);
      const supported = visible.find((m) => m.id === draft.alias) ||
        visible.find((m) => m.realId && m.realId === draft.alias) ||
        (visible.includes(chosen) ? chosen : visible[0]) || chosen;
      draft.alias = supported.id;
      draft.wire_id = supported.wireId;
      draft.reasoning = reasoningForSelectedModel(supported.reasoning, agent, supported);
      draft.speed = supportedDraftValue(supported.speed, draft.speed);
      // Closed sheet: the composer chip reads effective, not draft.
      // New Chat starts with an empty alias — adopt the catalogue default
      // so `.seg-model` and the next POST /api/sessions carry a real id.
      if (state.modelSheet.phase === "CLOSED") {
        const cur = String(
          (state.modelSheet.effective && state.modelSheet.effective.alias)
          || state.currentModel || "",
        ).trim();
        if (!cur || (agent !== "cursor" && typeof isModeSentinel === "function"
            && isModeSentinel(cur))
            || (typeof isForeignAgentModel === "function" && isForeignAgentModel(agent, cur))) {
          state.currentModel = supported.id;
          state.pickerModels = state.pickerModels || {};
          state.pickerModels[agent] = supported.id;
          if (typeof persistPickerPreferences === "function") persistPickerPreferences();
          if (typeof modelSheetSelectionFor === "function") {
            state.modelSheet.effective = modelSheetSelectionFor(agent, supported.id);
            state.modelSheet.draft = typeof cloneModelSelection === "function"
              ? cloneModelSelection(state.modelSheet.effective) : draft;
          } else {
            state.modelSheet.effective = cloneModelSelection(draft);
          }
        }
      }
    }
  }
  renderModelSelect();
  if (typeof syncModelSheetEffortFromStatus === "function")
    syncModelSheetEffortFromStatus();
  if (typeof renderModelSheet === "function") renderModelSheet();
  // Catalogue arrival must refresh the composer chip — New Chat wipes
  // engineModel and used to leave `.seg-model` blank until a later tick.
  if (typeof renderStatusSegments === "function") renderStatusSegments();
}

function loadModelsForCurrentAgent(options = {}) {
  const agent = options.agent || selectedPickerAgent();
  if (
    !options.loading && !options.force
    && Array.isArray(state.modelOptionsByAgent[agent])
    && state.modelOptionsByAgent[agent].length
  ) {
    if (selectedPickerAgent() === agent)
      state.models = state.modelOptionsByAgent[agent];
    renderModelSelect();
    if (typeof renderModelSheet === "function") renderModelSheet();
    // The catalogue is already warm, but the agent's saved default effort
    // may not be (page load / session open) — hydrate it and let the
    // reconcile repaint if it differs.
    if (typeof hydrateAgentDefaultEffort === "function")
      hydrateAgentDefaultEffort(agent).catch(() => "");
    return Promise.resolve();
  }
  return loadModelsForAgent(agent, options);
}

function seedSelectedModelOnly(agent) {
  agent = String(agent || selectedPickerAgent() || "").trim();
  const cached = agent ? state.modelOptionsByAgent[agent] : null;
  if (Array.isArray(cached) && cached.length) {
    if (selectedPickerAgent() === agent) state.models = cached;
    renderModelSelect();
    return;
  }
  const alias = String(((state.pickerModels || {})[agent]) || "").trim();
  if (!alias || isForeignAgentModel(agent, alias)) {
    if (!agent || selectedPickerAgent() === agent) state.models = [];
    renderModelSelect();
    return;
  }
  const row = {
    id: alias,
    alias,
    label: alias,
    subtitle: "",
    wireId: "",
    realId: "",
    provider: "configured",
    active: true,
    isDefault: true,
    badge: "",
    reasoning: null,
    speed: null,
  };
  if (selectedPickerAgent() === agent) state.models = [row];
  renderModelSelect();
}

function renderModelSelect(loading = false) {
  if (typeof modelSelect === "undefined" || !modelSelect) return;
  if (state.monitorReadOnly) {
    modelSelect.innerHTML = '<option value="">read only</option>';
    modelSelect.disabled = true;
    renderTeacherModelSelect();
    return;
  }
  if (loading) {
    modelSelect.innerHTML = '<option value="">Loading models…</option>';
    modelSelect.disabled = true;
    renderTeacherModelSelect();
    return;
  }
  if (!state.models.length) {
    modelSelect.innerHTML = '<option value="">— no models —</option>';
    modelSelect.disabled = true;
    renderTeacherModelSelect(); // teacher dropdown shares the model list
    return;
  }
  modelSelect.disabled = false;
  modelSelect.innerHTML = state.models
    .map(
      (m) =>
        `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}${m.badge ? ` · ${escapeHtml(m.badge)}` : ""}</option>`,
    )
    .join("");
  // Reflect only the composer-owned choice. Backend current/engine model fields
  // are intentionally not consulted here.
  const agent = selectedPickerAgent();
  const shown = pickerModelAlias(agent);
  // Resolve the shown alias onto its catalogue row — a session-reported wire id
  // (realId) must render as the catalogue id, not a value no option has.
  const row = state.models.find((m) => m.id === shown) ||
    state.models.find((m) => m.realId && m.realId === shown);
  modelSelect.value = row ? row.id : shown;
  // Outside an open sheet, capture the frontend default so Send always carries
  // a concrete acknowledged projection. An open sheet may alter draft only.
  if (shown && state.modelSheet.phase === "CLOSED") {
    state.modelSheet.effective = cloneModelSelection({
      ...state.modelSheet.effective,
      agent,
      provider: row ? row.provider : "",
      // Normalize a wire-id hit onto its catalogue id (same as
      // modelSheetSelectionFor) so effective/draft/select all agree.
      alias: row ? row.id : shown,
      wire_id: row ? row.wireId : "",
    });
    state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
    if (!isForeignAgentModel(agent, shown))
      state.pickerModels[agent] = shown;
    else
      delete state.pickerModels[agent];
    persistPickerPreferences();
  }
  if (typeof renderModelSheet === "function") renderModelSheet();
  // The teacher dropdown draws from the same model list — repopulate it once
  // the models arrive so it isn't stuck empty when teacher is on.
  renderTeacherModelSelect();
}

// Decide which model (if any) must be force-applied to a session's gateway turn
// BEFORE its first prompt. A session whose engine model is still unknown (no
// turn has reported one yet) was born running the agent's DEFAULT model — NOT
// the alias the picker shows — so the visible selection must be applied first
// or turn 1 silently runs the wrong model and the picker then reconciles TO it.
// Once a turn HAS reported a real engine model, or the session history already
// contains a turn, the session is running a known model. The latter matters for
// engines such as local_llm whose durable header does not populate
// `engine_model`; relying on that field alone would re-issue `/model` when a
// resumed chat is already in flight. Explicit picker changes go through
// `switchModel`, which applies `/model` itself, so return '' in both cases.
function modelToApplyBeforeTurn(selectedModel, engineModel, hasPriorTurn) {
  if (engineModel || hasPriorTurn) return "";
  const selected = (selectedModel == null ? "" : String(selectedModel)).trim();
  if (!selected) return "";
  const agent = (typeof selectedPickerAgent === "function" && selectedPickerAgent()) || "";
  // Auto on antigravity means the CLI default — never force a leftover
  // local-LLM wire id (qwen3.8-27b-fp8) onto ``--model``.
  if (agent === "antigravity" && selected.toLowerCase() === "auto") return "";
  // Mode picker values are not model ids. Cursor ACP still uses ``auto``.
  if (isLocalLlmAgent(agent) && isModeSentinel(selected)) return "";
  if (typeof isForeignAgentModel === "function" && isForeignAgentModel(agent, selected)) return "";
  return selected;
}

// Decide which reasoning/tools-visibility commands must be force-applied to a
// session's gateway turn BEFORE its first prompt — the visibility analogue of
// modelToApplyBeforeTurn. A fresh gateway session is born with the backend
// defaults (reasoning=stream, tools=on); but a session adopted from another
// interface (e.g. the TUI) can carry different values, and the web's own
// defaults must win on turn 1. So while the engine model is still unknown (no
// turn has run yet) we assert the web's `state.vis` to the gateway session so
// the FIRST message renders streamed reasoning + tool headers as configured.
// Once a turn HAS reported a real engine model the session's prefs are known
// and explicit toggles flow through the Options card / `/reasoning`|`/tools`
// slash commands, so we return [] to avoid clobbering a mid-session change.
// Returns an ordered list of slash commands (possibly empty).
function visToApplyBeforeTurn(vis, engineModel) {
  if (engineModel) return []; // a turn already ran — prefs are known
  const v = vis || {};
  const cmds = [];
  const r = (v.reasoning == null ? "" : String(v.reasoning)).trim();
  const t = (v.tools == null ? "" : String(v.tools)).trim();
  if (r) cmds.push(`/reasoning ${r}`);
  if (t) cmds.push(`/tools ${t}`);
  return cmds;
}

// Resolve which alias option should be selected, given the session's current
// model. Try (1) a direct alias match, (2) the real engine model id carried in
// each row's subtitle, (3) a backend-marked `active` row, then (4) the first
// row. Returns '' when there are no models.
function activeModelAlias() {
  const ms = state.models;
  if (!ms.length) return "";
  // Candidates, best-first: an explicit selection (alias or real id) the user
  // made, then the engine's real model from the last turn (covers a freshly
  // created session that adopted the gateway default but never had `/model`
  // run, so only `engineModel` knows the truth).
  const cands = [state.currentModel, state.engineModel].filter(Boolean);
  for (const cur of cands) {
    const hit =
      ms.find((m) => m.id === cur) ||
      ms.find((m) => m.realId && m.realId === cur);
    if (hit) return hit.id;
  }
  return (ms.find((m) => m.active) || ms[0]).id; // backend active flag, else first
}

// Picker changes are frontend settings. The chosen alias is applied by
// sendMessage immediately before the next prompt, never from this change event.
async function switchModel(id) {
  return selectPickerModel(id);
}
