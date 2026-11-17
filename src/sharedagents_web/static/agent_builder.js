/* Agent Builder — prompt, capability, and runtime control plane.
 *
 * The browser presents source-attributed prompt sections and catalog-driven
 * editors for tools, MCP functions, and skills. Effective access ALWAYS comes
 * from the backend (compose_tool_allowlist / discover_skills); the catalog UI
 * only edits the same raw operator-owned allowlist fields as the advanced
 * textareas. Allowlist resolution is never reimplemented here.
 *
 * All writes are sha-guarded: the server 409s on a lost update and this
 * script re-merges by reloading the agent.
 */
(function () {
  "use strict";

  const BASE = ((document.querySelector('meta[name="sharedagents-base"]') || {}).content || "");

  const selectEl = document.getElementById("builder-agent-select");
  const updatedEl = document.getElementById("builder-updated");
  const emptyEl = document.getElementById("builder-empty");
  const mainEl = document.getElementById("builder-main");
  const overviewEl = document.getElementById("builder-overview");
  const tabsEl = document.getElementById("builder-tabs");
  const cardPrompt = document.getElementById("card-system-prompt");
  const cardPerms = document.getElementById("card-permissions");
  const cardRuntime = document.getElementById("card-runtime");
  const sectionsEl = document.getElementById("sp-sections");
  const degradedEl = document.getElementById("sp-degraded");
  const hcToggleEl = document.getElementById("sp-hc-toggle");
  const hcNoteEl = document.getElementById("sp-hc-note");
  const bannerEl = document.getElementById("builder-restart-banner");
  const bannerFieldsEl = document.getElementById("builder-restart-fields");
  const bannerDismissEl = document.getElementById("builder-banner-dismiss");
  const savebarEl = document.getElementById("builder-savebar");

  const promptFilterEl = document.getElementById("prompt-filter");
  const promptExpandEl = document.getElementById("prompt-expand-all");
  const promptExactEl = document.getElementById("prompt-exact-badge");
  const toolPolicyEl = document.getElementById("tool-policy");
  const skillPolicyEl = document.getElementById("skill-policy");
  const toolCatalogEl = document.getElementById("tool-catalog");
  const skillCatalogEl = document.getElementById("skill-catalog");
  const toolCatalogFilterEl = document.getElementById("tool-catalog-filter");
  const skillCatalogFilterEl = document.getElementById("skill-catalog-filter");

  const permsForm = document.getElementById("perms-form");
  const modeSel = document.getElementById("perms-mode");
  const delegationSel = document.getElementById("perms-delegation");
  const modelSel = document.getElementById("perms-model");
  const modelNoteEl = document.getElementById("perms-model-note");
  const toolsEl = document.getElementById("perms-tools");
  const toolsStrictEl = document.getElementById("perms-tools-strict");
  const skillsEl = document.getElementById("perms-skills");
  const skillsStrictEl = document.getElementById("perms-skills-strict");
  const disabledEl = document.getElementById("perms-disabled");
  const descInputEl = document.getElementById("builder-agent-description-input");
  const permsDescEl = document.getElementById("perms-description");
  const permsSaveBtn = document.getElementById("perms-save");
  const permsReloadBtn = document.getElementById("perms-reload");
  const permsStatusEl = document.getElementById("perms-status");
  const effToolsList = document.getElementById("eff-tools-list");
  const effToolsCount = document.getElementById("eff-tools-count");
  const effToolsFilter = document.getElementById("eff-tools-filter");
  const effSkillsList = document.getElementById("eff-skills-list");
  const effSkillsCount = document.getElementById("eff-skills-count");

  const state = {
    agents: [],
    def: null,        // GET /api/agents/defs/<agent>
    md: null,         // GET /api/agents/defs/<agent>/agents_md
    prompt: null,     // GET /api/agents/defs/<agent>/system_prompt
    registry: { builtin: [], mcp: [] },
    editingMd: false,
    dirty: false,
    mdDirty: false,
    permsDirty: false,
    loading: false,
    loadGeneration: 0,
    loadController: null,
    mdSaveButtons: new Set(),
    toolsFilter: "",
    toolCatalogFilter: "",
    skillCatalogFilter: "",
    promptFilter: "",
    activeTab: "prompt",
    promptExpanded: false,
    showHardcoded: false,   // hardcoded prompt sections are hidden by default
    modelInitial: "",       // model drop-down selection at last renderPerms()
    delegationInitial: "",  // delegation_mode select at last renderPerms()
    modelManagedBy: "",     // "engine" | "profile" | "" — who owns the DEFAULT model (note only, never a gate)
    cardToggles: {},        // hardcoded section id -> enabled (true/false), the enable/disable cards
  };

  const SOURCE_LABEL = {
    hardcoded: "runtime",
    agents_md: "AGENTS.md",
    system: "system",
    dynamic: "dynamic",
    unknown: "source",
  };

  const SOURCE_DETAIL = {
    hardcoded: "sharedagents canonical policy",
    agents_md: "operator-owned instructions",
    system: "agent config",
    dynamic: "session context",
    unknown: "unattributed",
  };

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function lines(value) {
    return String(value || "")
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
  }

  function utf8Bytes(value) {
    const text = String(value == null ? "" : value);
    if (typeof TextEncoder === "function") {
      return new TextEncoder().encode(text).length;
    }
    // Exact fallback for older browsers: ``for...of`` iterates Unicode code
    // points (not UTF-16 code units), so astral characters count as 4 bytes.
    let size = 0;
    for (const char of text) {
      const cp = char.codePointAt(0);
      size += cp <= 0x7f ? 1 : cp <= 0x7ff ? 2 : cp <= 0xffff ? 3 : 4;
    }
    return size;
  }

  function fmtTime() {
    return new Date().toLocaleTimeString();
  }

  async function api(method, path, body, signal) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    if (signal) opts.signal = signal;
    const resp = await fetch(BASE + path, opts);
    let payload = null;
    try {
      payload = await resp.json();
    } catch (_e) { /* non-JSON */ }
    return { ok: resp.ok, status: resp.status, payload };
  }

  function syncDirty() {
    state.dirty = state.mdDirty || state.permsDirty;
    // AGENTS.md has its own inline Save/Cancel controls. The global bar owns
    // config-backed changes only (prompt toggles, capability rules, runtime).
    savebarEl.hidden = !state.permsDirty;
  }

  function clearDirty() {
    state.mdDirty = false;
    state.permsDirty = false;
    syncDirty();
  }

  function setLoading(loading) {
    state.loading = loading;
    permsSaveBtn.disabled = loading;
    permsReloadBtn.disabled = loading;
    for (const save of state.mdSaveButtons) save.disabled = loading;
  }

  function currentAgentMatches(name) {
    return !state.loading && !!name && !!state.def &&
      state.def.name === name && selectEl.value === name;
  }

  function discardUnsavedChanges() {
    const resetPrompt = state.mdDirty || state.editingMd;
    const resetPerms = state.permsDirty;
    clearDirty();
    if (resetPrompt && state.prompt) renderPrompt();
    if (resetPerms && state.def) renderPerms();
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value == null || value === "" ? "—" : String(value);
  }

  function uniqueSorted(values) {
    return Array.from(new Set((values || []).filter(Boolean).map(String))).sort((a, b) => a.localeCompare(b));
  }

  function activateTab(name) {
    state.activeTab = ["prompt", "capabilities", "runtime"].includes(name) ? name : "prompt";
    for (const button of tabsEl.querySelectorAll("[data-builder-tab]")) {
      const active = button.dataset.builderTab === state.activeTab;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    }
    cardPrompt.hidden = state.activeTab !== "prompt";
    cardPerms.hidden = state.activeTab !== "capabilities";
    cardRuntime.hidden = state.activeTab !== "runtime";
    if (history.replaceState) history.replaceState(null, "", "#" + state.activeTab);
    mainEl.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ---------------------------------------------------------------- init

  async function init() {
    let listResponse;
    let registryResponse;
    try {
      [listResponse, registryResponse] = await Promise.all([
        api("GET", "/api/agents/defs"),
        api("GET", "/api/agents/registry"),
      ]);
    } catch (err) {
      emptyEl.textContent = "Failed to load agents: " + (err.message || "gateway unreachable");
      return;
    }
    const { ok, payload } = listResponse;
    if (!ok) {
      emptyEl.textContent = "Failed to load agents: " + ((payload || {}).error || "gateway unreachable");
      emptyEl.hidden = false;
      return;
    }
    if (registryResponse && registryResponse.ok) {
      state.registry = {
        builtin: uniqueSorted((registryResponse.payload || {}).builtin),
        mcp: uniqueSorted((registryResponse.payload || {}).mcp),
      };
    }
    state.agents = (payload || {}).agents || [];
    if (!state.agents.length) {
      emptyEl.hidden = false;
      return;
    }
    emptyEl.hidden = true;
    selectEl.innerHTML = "";
    for (const a of state.agents) {
      const opt = document.createElement("option");
      opt.value = a.name;
      opt.textContent = a.name + (a.agent_profile ? " (" + a.agent_profile + ")" : "");
      selectEl.appendChild(opt);
    }
    selectEl.addEventListener("change", async () => {
      if (state.dirty) {
        const ok = await confirmCard({
          title: "Discard unsaved changes?",
          okLabel: "Discard",
        });
        if (!ok) {
          selectEl.value = state.def ? state.def.name : selectEl.value;
          return;
        }
        discardUnsavedChanges();
      }
      loadAgent(selectEl.value);
    });
    for (const button of tabsEl.querySelectorAll("[data-builder-tab]")) {
      button.addEventListener("click", () => activateTab(button.dataset.builderTab));
    }
    const requestedTab = String(location.hash || "").replace(/^#/, "");
    if (["prompt", "capabilities", "runtime"].includes(requestedTab)) state.activeTab = requestedTab;
    selectEl.value = state.agents[0].name;
    await loadAgent(selectEl.value);
  }

  async function loadAgent(name) {
    const generation = ++state.loadGeneration;
    if (state.loadController) state.loadController.abort();
    const controller = typeof AbortController === "function" ? new AbortController() : null;
    state.loadController = controller;
    setLoading(true);
    overviewEl.hidden = true;
    tabsEl.hidden = true;
    cardPrompt.hidden = true;
    cardPerms.hidden = true;
    cardRuntime.hidden = true;
    permsStatusEl.textContent = "";
    bannerEl.hidden = true;
    let responses;
    try {
      responses = await Promise.all([
        api("GET", "/api/agents/defs/" + encodeURIComponent(name), undefined, controller && controller.signal),
        api("GET", "/api/agents/defs/" + encodeURIComponent(name) + "/agents_md", undefined, controller && controller.signal),
        api("GET", "/api/agents/defs/" + encodeURIComponent(name) + "/system_prompt", undefined, controller && controller.signal),
      ]);
    } catch (err) {
      if (generation !== state.loadGeneration || (err && err.name === "AbortError")) return;
      state.loadController = null;
      setLoading(false);
      selectEl.value = state.def ? state.def.name : name;
      emptyEl.textContent = "Failed to load agent " + name + ": " + (err.message || "gateway error");
      emptyEl.hidden = false;
      overviewEl.hidden = !state.def;
      tabsEl.hidden = !state.def;
      if (state.def) activateTab(state.activeTab);
      return;
    }
    if (generation !== state.loadGeneration || selectEl.value !== name) return;
    const [def, md, prompt] = responses;
    if (!def.ok || !md.ok || !prompt.ok) {
      const failed = [def, md, prompt].find((response) => !response.ok);
      const err = ((failed || {}).payload || {}).error || "gateway error";
      state.loadController = null;
      setLoading(false);
      selectEl.value = state.def ? state.def.name : name;
      emptyEl.textContent = "Failed to load agent " + name + ": " + err;
      emptyEl.hidden = false;
      overviewEl.hidden = !state.def;
      tabsEl.hidden = !state.def;
      if (state.def) activateTab(state.activeTab);
      return;
    }
    state.loadController = null;
    emptyEl.hidden = true;
    state.def = def.payload;
    state.md = md.payload;
    state.prompt = prompt.payload;
    state.editingMd = false;
    clearDirty();
    state.showHardcoded = false;
    hcToggleEl.checked = false;
    hcNoteEl.hidden = true;
    state.toolsFilter = "";
    state.toolCatalogFilter = "";
    state.skillCatalogFilter = "";
    state.promptFilter = "";
    state.promptExpanded = false;
    effToolsFilter.value = "";
    toolCatalogFilterEl.value = "";
    skillCatalogFilterEl.value = "";
    promptFilterEl.value = "";
    promptExpandEl.textContent = "Expand all";
    selectEl.value = name;
    updatedEl.textContent = "loaded " + fmtTime();
    renderPrompt();
    renderPerms();
    renderOverview();
    overviewEl.hidden = false;
    tabsEl.hidden = false;
    activateTab(state.activeTab);
    setLoading(false);
  }

  // ------------------------------------------------------ system prompt

  function sectionBadgeClass(source) {
    return {
      hardcoded: "sp-badge-hc",
      agents_md: "sp-badge-md",
      system: "sp-badge-sys",
      dynamic: "sp-badge-dyn",
    }[source] || "sp-badge-unknown";
  }

  function renderOverview() {
    const d = state.def || {};
    const meta = state.prompt || {};
    const tools = d.effective_tools || [];
    const mcpNames = new Set(state.registry.mcp || []);
    const mcpCount = tools.filter((name) => mcpNames.has(name) || String(name).startsWith("mcp__")).length;
    const avatar = document.getElementById("builder-agent-avatar");
    const name = String(d.name || "agent");
    avatar.textContent = name.slice(0, 1).toUpperCase();
    setText("builder-agent-name", name);
    setText("builder-agent-profile", d.agent_profile ? "profile · " + d.agent_profile : "custom agent");
    setText("builder-agent-engine", d.engine || "local_llm");
    const descVal = d.description || "";
    if (descInputEl) descInputEl.value = descVal;
    if (permsDescEl) permsDescEl.value = descVal;
    setText("summary-model", d.model || "inherit global");
    setText("summary-mode", d.permission_mode || "auto");
    setText("summary-prompt", Number(meta.chars || 0).toLocaleString() + " chars");
    setText("summary-tools", tools.length);
    setText("summary-mcp", mcpCount);
    setText("summary-skills", (d.effective_skills || []).length);
    setText("runtime-profile", d.agent_profile || "none (custom)");
    setText("runtime-engine", d.engine || "local_llm");
    setText("runtime-model-owner", d.model_managed_by || "agent / global config");
    setText("runtime-delegation-source", d.delegation_mode ? "agent override" : "profile default");
    setText("runtime-config-sha", d.config_sha ? d.config_sha.slice(0, 12) : "unavailable");
    renderRuntimeExplainers();
  }

  function renderPromptMetrics() {
    const meta = state.prompt || {};
    const sections = meta.sections || [];
    const active = sections.filter((section) => !section.disabled);
    const operator = active.filter((section) => section.source === "agents_md" || section.source === "system");
    const chars = Number(meta.chars || 0);
    setText("prompt-char-count", chars.toLocaleString());
    setText("prompt-token-estimate", "≈ " + Math.ceil(chars / 4).toLocaleString());
    setText("prompt-section-count", active.length);
    setText("prompt-operator-count", operator.length);
    promptExactEl.textContent = meta.exact ? "Exact runtime match" : "Unattributed prompt";
    promptExactEl.className = "status-badge " + (meta.exact ? "status-ok" : "status-neutral");
  }

  function applyPromptFilter() {
    const query = state.promptFilter.toLowerCase();
    for (const section of sectionsEl.querySelectorAll(".sp-section")) {
      const haystack = (section.dataset.title + " " + section.dataset.source + " " + section.textContent).toLowerCase();
      section.hidden = !!query && !haystack.includes(query);
    }
  }

  function renderPrompt() {
    state.mdSaveButtons.clear();
    state.editingMd = false;
    state.cardToggles = {};
    sectionsEl.innerHTML = "";
    degradedEl.hidden = true;
    const meta = state.prompt;
    if (!meta || !meta.sections) {
      degradedEl.hidden = false;
      return;
    }
    renderPromptMetrics();
    if (!meta.exact) {
      degradedEl.hidden = false;
    }
    for (const sec of meta.sections) {
      const src = sec.source || "unknown";
      // Hardcoded sections render as enable/disable CARDS: the
      // checkbox persists through disabled_prompt_sections (toggle state
      // only — the client never sends a composed prompt string). Every
      // other source stays in the flat list, hidden by default via the
      // visibility toggle.
      if (src === "hardcoded" && sec.id) {
        const el = document.createElement("div");
        el.className = "sp-section sp-card" + (sec.disabled ? " sp-card-disabled" : "");
        el.dataset.source = src;
        el.dataset.title = sec.title || "";
        const body = document.createElement("pre");
        body.className = "sp-body";
        body.hidden = true;
        body.textContent = sec.text || "";
        const head = document.createElement("div");
        head.className = "sp-section-head";
        // Enable/disable switch: a <label> wrapping the checkbox so the
        // whole control is clickable; the head's collapse handler below
        // ignores clicks that land on it.
        const toggleLabel = document.createElement("label");
        toggleLabel.className = "sp-card-toggle";
        const toggle = document.createElement("input");
        toggle.type = "checkbox";
        toggle.className = "sp-card-enable";
        toggle.dataset.sectionId = sec.id;
        toggle.checked = !sec.disabled;
        toggleLabel.title = sec.disabled
          ? "Section disabled — the agent's prompt omits it. Check to re-enable."
          : "Section enabled — uncheck to remove it from the agent's prompt.";
        toggle.addEventListener("change", () => {
          state.cardToggles[sec.id] = toggle.checked;
          sec.disabled = !toggle.checked;
          el.classList.toggle("sp-card-disabled", !toggle.checked);
          renderPromptMetrics();
          markPermsDirty();
        });
        toggleLabel.appendChild(toggle);
        state.cardToggles[sec.id] = !sec.disabled;
        head.innerHTML =
          '<span class="sp-badge ' + sectionBadgeClass(src) + '">' + esc(SOURCE_LABEL[src] || src) + "</span>" +
          '<span class="sp-title">' + esc(sec.title || "") + "</span>" +
          '<span class="sp-source-detail">' + esc(SOURCE_DETAIL[src] || SOURCE_DETAIL.unknown) + "</span>" +
          '<span class="sp-meta">' + esc(String((sec.text || "").length) + " chars") +
          ' <span class="sp-chevron">▸</span></span>';
        // Enable/disable switch, first in the row (before the badge).
        head.insertBefore(toggleLabel, head.firstChild);
        head.addEventListener("click", (ev) => {
          // The checkbox (or its label) keeps its own behavior.
          if (ev.target === toggle || ev.target === toggleLabel) return;
          body.hidden = !body.hidden;
          head.querySelector(".sp-chevron").textContent = body.hidden ? "▸" : "▾";
        });
        el.appendChild(head);
        el.appendChild(body);
        sectionsEl.appendChild(el);
        continue;
      }
      const el = document.createElement("div");
      el.className = "sp-section";
      el.dataset.source = src;
      el.dataset.title = sec.title || "";
      const body = document.createElement("pre");
      body.className = "sp-body";
      body.hidden = true;
      body.textContent = sec.text || "";
      const head = document.createElement("div");
      head.className = "sp-section-head";
      head.innerHTML =
        '<span class="sp-badge ' + sectionBadgeClass(src) + '">' + esc(SOURCE_LABEL[src] || src) + "</span>" +
        '<span class="sp-title">' + esc(sec.title || "") + "</span>" +
        '<span class="sp-source-detail">' + esc(SOURCE_DETAIL[src] || SOURCE_DETAIL.unknown) + "</span>" +
        '<span class="sp-meta">' + esc(String((sec.text || "").length) + " chars") +
        ' <span class="sp-chevron">▸</span></span>';
      head.addEventListener("click", () => {
        body.hidden = !body.hidden;
        head.querySelector(".sp-chevron").textContent = body.hidden ? "▸" : "▾";
      });
      el.appendChild(head);
      el.appendChild(body);
      // AGENTS.md sections get the edit affordance (the one configurable part).
      if (src === "agents_md") {
        appendMdEditor(el, body, sec);
      }
      sectionsEl.appendChild(el);
    }
    // Hardcoded sections stay in the DOM (cards, chevrons, layout, tags
    // intact) but are hidden by default via a container class — no reload,
    // no API change.
    sectionsEl.classList.toggle("sp-hc-hidden", !state.showHardcoded);
    updateHcNote();
    applyPromptFilter();
  }

  // Re-sync the hardcoded cards IN PLACE after a successful save (no
  // renderPrompt() re-call — that would rebuild the AGENTS.md editor and
  // drop its content/scroll state). Checkbox state + disabled styling
  // follow the freshly loaded definition's persisted toggle list.
  function syncPromptCards() {
    const disabled = new Set(state.def && state.def.disabled_prompt_sections || []);
    for (const section of ((state.prompt || {}).sections || [])) {
      if (section.source === "hardcoded" && section.id) section.disabled = disabled.has(section.id);
    }
    for (const el of sectionsEl.querySelectorAll(".sp-card")) {
      const toggle = el.querySelector(".sp-card-enable");
      if (!toggle) continue;
      const id = toggle.dataset.sectionId || "";
      const enabled = !disabled.has(id);
      toggle.checked = enabled;
      el.classList.toggle("sp-card-disabled", !enabled);
      if (id) state.cardToggles[id] = enabled;
    }
    renderPromptMetrics();
  }

  function updateHcNote() {
    const hc = (state.prompt && state.prompt.sections || [])
      .filter((s) => (s.source || "unknown") === "hardcoded");
    const chars = hc.reduce((n, s) => n + (s.text || "").length, 0);
    if (!hc.length || state.showHardcoded) {
      hcNoteEl.hidden = true;
    } else {
      hcNoteEl.hidden = false;
      hcNoteEl.textContent = hc.length + " hardcoded section" +
        (hc.length === 1 ? "" : "s") + " hidden (" + chars + " chars) — tick the box above to reveal.";
    }
  }

  function appendMdEditor(sectionEl, bodyEl, sec) {
    const tools = document.createElement("div");
    tools.className = "sp-edit-tools";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "builder-btn";
    btn.textContent = "Edit";
    const status = document.createElement("span");
    status.className = "sp-edit-status";
    tools.appendChild(btn);
    tools.appendChild(status);
    sectionEl.appendChild(tools);

    const showEditor = () => {
      const agentName = state.def && state.def.name;
      if (!currentAgentMatches(agentName)) return;
      state.editingMd = true;
      btn.hidden = true;
      status.textContent = "Editing the AGENTS.md that gets injected into the prompt. Save to persist.";
      const ta = document.createElement("textarea");
      ta.className = "sp-editor";
      ta.value = state.md ? state.md.content : (sec.text || "");
      const row = document.createElement("div");
      row.className = "sp-edit-tools";
      const save = document.createElement("button");
      save.type = "button";
      save.className = "builder-btn builder-btn-primary";
      save.textContent = "Save";
      save.disabled = state.loading;
      state.mdSaveButtons.add(save);
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "builder-btn builder-btn-ghost";
      cancel.textContent = "Cancel";
      row.appendChild(save);
      row.appendChild(cancel);
      sectionEl.insertBefore(ta, tools);
      sectionEl.insertBefore(row, tools);
      tools.hidden = true;
      bodyEl.hidden = true;

      const exit = () => {
        state.editingMd = false;
        state.mdSaveButtons.delete(save);
        ta.remove();
        row.remove();
        tools.hidden = false;
        btn.hidden = false;
        status.textContent = "";
      };
      ta.addEventListener("input", () => {
        state.mdDirty = true;
        syncDirty();
      });
      cancel.addEventListener("click", () => {
        state.mdDirty = false;
        syncDirty();
        exit();
      });
      save.addEventListener("click", async () => {
        const targetName = state.def && state.def.name;
        if (!currentAgentMatches(targetName)) {
          status.className = "sp-edit-status err";
          status.textContent = state.loading
            ? "Wait for the selected agent to finish loading."
            : "Agent selection changed — save refused.";
          return;
        }
        if (utf8Bytes(ta.value) > 64 * 1024) {
          status.className = "sp-edit-status err";
          status.textContent = "Too large — AGENTS.md is capped at 64 KiB.";
          return;
        }
        const content = ta.value;
        const sha = state.md ? state.md.sha : undefined;
        save.disabled = true;
        save.textContent = "Saving…";
        const res = await api("PUT", "/api/agents/defs/" + encodeURIComponent(targetName) + "/agents_md", {
          content,
          sha,
        });
        if (!currentAgentMatches(targetName)) {
          save.disabled = state.loading;
          return;
        }
        if (res.ok) {
          state.md = { ...state.md, content, sha: res.payload.agents_md_sha, exists: true };
          status.className = "sp-edit-status ok";
          status.textContent = "Saved. New sessions use this content.";
          updatedEl.textContent = "instructions saved " + fmtTime();
          state.mdDirty = false;
          syncDirty();
          exit();
          // Refresh the render so the prompt view reflects the new file.
          const p = await api("GET", "/api/agents/defs/" + encodeURIComponent(targetName) + "/system_prompt");
          if (p.ok && currentAgentMatches(targetName)) {
            state.prompt = p.payload;
            renderPrompt();
          }
        } else {
          const err = res.payload || {};
          save.disabled = false;
          save.textContent = "Save";
          if (res.status === 409) {
            status.className = "sp-edit-status err";
            status.textContent = "AGENTS.md changed on disk — reloading.";
            exit();
            const md = await api("GET", "/api/agents/defs/" + encodeURIComponent(targetName) + "/agents_md");
            if (md.ok && currentAgentMatches(targetName)) state.md = md.payload;
            const pr = await api("GET", "/api/agents/defs/" + encodeURIComponent(targetName) + "/system_prompt");
            if (pr.ok && currentAgentMatches(targetName)) { state.prompt = pr.payload; renderPrompt(); }
          } else {
            if (currentAgentMatches(targetName)) {
              status.className = "sp-edit-status err";
              status.textContent = err.error || ("Save failed (" + res.status + ")");
            }
          }
        }
      });
    };
    btn.addEventListener("click", showEditor);
  }

  // ------------------------------------------------------- permissions

  function fillFromLines(el, value) {
    el.value = Array.isArray(value) ? value.join("\n") : "";
  }

  function markPermsDirty() {
    state.permsDirty = true;
    syncDirty();
  }

  function updatePolicyButtons(container, policy) {
    for (const button of container.querySelectorAll("[data-policy]")) {
      const active = button.dataset.policy === policy;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-checked", active ? "true" : "false");
    }
  }

  function allToolNames() {
    return uniqueSorted([
      ...(state.registry.builtin || []),
      ...(state.registry.mcp || []),
      ...((state.def || {}).effective_tools || []),
      ...lines(toolsEl.value),
      ...lines(disabledEl.value),
    ]);
  }

  function allSkillNames() {
    return uniqueSorted([
      ...((state.def || {}).available_skills || []),
      ...((state.def || {}).effective_skills || []),
      ...lines(skillsEl.value),
    ]);
  }

  function renderCatalogGroup(parent, title, names, selected, unrestricted, sourceLabel, kind, disabled) {
    if (!names.length) return;
    const section = document.createElement("section");
    section.className = "catalog-group";
    const head = document.createElement("div");
    head.className = "catalog-group-head";
    head.innerHTML = "<span>" + esc(title) + "</span><span>" + names.length + " available</span>";
    const items = document.createElement("div");
    items.className = "catalog-items";
    for (const name of names) {
      const label = document.createElement("label");
      label.className = "catalog-item";
      label.title = name;
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = kind === "tool" && unrestricted
        ? !(disabled || new Set()).has(name)
        : unrestricted || selected.has(name);
      input.addEventListener("change", () => {
        if (kind === "tool" && unrestricted) {
          // An unrestricted policy plus one unchecked catalog entry is an
          // explicit deny, not a new snapshot of every currently discovered
          // tool. Persisting the deny keeps the choice stable as MCP/tools
          // catalogs change and avoids turning one click into a large,
          // stale allowlist.
          const denied = new Set(disabled || []);
          if (input.checked) denied.delete(name); else denied.add(name);
          disabledEl.value = uniqueSorted(Array.from(denied)).join("\n");
        } else {
          const values = new Set(unrestricted ? (kind === "tool" ? allToolNames() : allSkillNames()) : Array.from(selected));
          if (input.checked) values.add(name); else values.delete(name);
          const target = kind === "tool" ? toolsEl : skillsEl;
          target.value = uniqueSorted(Array.from(values)).join("\n");
        }
        if (kind === "tool") renderToolCatalog(); else renderSkillCatalog();
        markPermsDirty();
      });
      const text = document.createElement("span");
      text.textContent = name;
      label.appendChild(input);
      label.appendChild(text);
      if (sourceLabel) {
        const source = document.createElement("em");
        source.textContent = sourceLabel;
        label.appendChild(source);
      }
      items.appendChild(label);
    }
    section.appendChild(head);
    section.appendChild(items);
    parent.appendChild(section);
  }

  function renderToolCatalog() {
    const raw = lines(toolsEl.value);
    const unrestricted = raw.length === 0;
    const selected = new Set(raw);
    const disabled = new Set(lines(disabledEl.value));
    const query = state.toolCatalogFilter.toLowerCase();
    const builtinSet = new Set(state.registry.builtin || []);
    const mcpSet = new Set(state.registry.mcp || []);
    const filter = (items) => uniqueSorted(items).filter((name) => !query || name.toLowerCase().includes(query));
    const builtin = filter(Array.from(builtinSet));
    const mcp = filter(Array.from(mcpSet));
    const other = filter(allToolNames().filter((name) => !builtinSet.has(name) && !mcpSet.has(name)));
    toolCatalogEl.innerHTML = "";
    renderCatalogGroup(toolCatalogEl, "Built-in tools", builtin, selected, unrestricted, "core", "tool", disabled);
    renderCatalogGroup(toolCatalogEl, "MCP functions", mcp, selected, unrestricted, "MCP", "tool", disabled);
    renderCatalogGroup(toolCatalogEl, "Profile & custom rules", other, selected, unrestricted, "rule", "tool", disabled);
    if (!toolCatalogEl.children.length) toolCatalogEl.innerHTML = '<div class="catalog-empty">No tools match this filter.</div>';
    updatePolicyButtons(toolPolicyEl, unrestricted ? "all" : "custom");
    setText("tool-catalog-count", allToolNames().length + " available");
    toolsStrictEl.disabled = unrestricted;
  }

  function renderSkillCatalog() {
    const raw = lines(skillsEl.value);
    const unrestricted = raw.length === 0;
    const selected = new Set(raw);
    const query = state.skillCatalogFilter.toLowerCase();
    const names = allSkillNames().filter((name) => !query || name.toLowerCase().includes(query));
    skillCatalogEl.innerHTML = "";
    renderCatalogGroup(skillCatalogEl, "Discovered skills", names, selected, unrestricted, "skill", "skill");
    if (!skillCatalogEl.children.length) skillCatalogEl.innerHTML = '<div class="catalog-empty">No skills match this filter.</div>';
    updatePolicyButtons(skillPolicyEl, unrestricted ? "all" : "custom");
    setText("skill-catalog-count", allSkillNames().length + " discovered");
    skillsStrictEl.disabled = unrestricted;
  }

  function renderCatalogs() {
    renderToolCatalog();
    renderSkillCatalog();
  }

  function renderRuntimeExplainers() {
    const permissionCopy = {
      auto: "Allowed actions run without per-call approval. Explicit deny rules and unconditional safety blocks still apply.",
      ask: "Read-only and ordinary actions run normally; destructive operations pause for an operator decision.",
      plan: "The agent can inspect and reason, while mutating and destructive actions are denied by the host.",
    };
    const delegationCopy = {
      "": "Uses the role’s default: orchestrators usually detach work; build and plan specialists usually wait inline.",
      background: "Child work detaches. The parent can continue orchestrating while results arrive asynchronously.",
      foreground: "The parent waits for child work to finish, keeping completion in one visible execution chain.",
      default: "No Builder pin is applied; the model or environment chooses the execution mode.",
    };
    setText("permission-mode-explainer", permissionCopy[modeSel.value] || permissionCopy.auto);
    setText("delegation-mode-explainer", delegationCopy[delegationSel.value] || delegationCopy[""]);
  }

  function renderPerms(options) {
    options = options || {};
    const d = state.def;
    modeSel.value = d.permission_mode || "auto";
    delegationSel.value = d.delegation_mode || "";
    const descVal = d.description || "";
    if (descInputEl) descInputEl.value = descVal;
    if (permsDescEl) permsDescEl.value = descVal;
    state.descriptionInitial = descVal;
    // Model drop-down: options come from the agent's effective models[]
    // catalog (the same rows the runtime resolves / the TUI picker lists)
    // for EVERY agent class — engine/profile-managed agents included.
    // An explicit 'inherit
    // global (default)' option (value '') heads the list and is preselected
    // whenever the agent has NO per-agent override (state.def.model is then
    // just the merged/global effective id), so the override can be cleared
    // from the UI. model_managed_by is INFORMATIONAL only: it renders a
    // note with the managed default but never disables
    // the select — the backend validates every class against the catalog
    // and an override set here wins for new sessions.
    const models = d.models || [];
    const effective = d.model || "";
    const hasOverride = Array.isArray(d.on_disk_keys) &&
      d.on_disk_keys.indexOf("upstream_model") !== -1;
    const modelManagedBy = d.model_managed_by || "";
    state.modelManagedBy = modelManagedBy;
    modelSel.disabled = false;
    modelSel.innerHTML = "";
    if (models.length) {
      const inherit = document.createElement("option");
      inherit.value = "";
      inherit.textContent = "inherit global (default)";
      modelSel.appendChild(inherit);
      for (const m of models) {
        const mid = String(m.id || "");
        if (!mid) continue;
        const opt = document.createElement("option");
        opt.value = mid;
        const provider = String(m.provider || m.backend || "").trim();
        opt.textContent = (m.alias ? m.alias + " — " : "") + mid +
          (provider && provider !== m.alias ? " (" + provider + ")" : "");
        modelSel.appendChild(opt);
      }
    } else {
      const opt = document.createElement("option");
      opt.value = effective;
      opt.textContent = effective || "inherit global";
      opt.disabled = true;
      modelSel.appendChild(opt);
    }
    // Find the option by direct value comparison — interpolating the id
    // into a CSS selector string would throw on ids with quotes/escapes.
    // Preselect the per-agent override when one is on disk; otherwise the
    // inherit-global option (the concrete id would just be the merged
    // effective id and hide the inherit path).
    const hasEffective = Array.from(modelSel.options).some((o) => o.value === effective);
    modelSel.value = hasOverride ? (hasEffective ? effective : "") : "";
    state.modelInitial = modelSel.value;
    state.delegationInitial = delegationSel.value;
    // Managed-default note: only meaningful while the effective id is the
    // MANAGED default (no override on disk); with an override the effective
    // id IS the override, and the note would mislabel it.
    if (modelNoteEl) {
      modelNoteEl.textContent = (modelManagedBy && !hasOverride)
        ? "default from " + modelManagedBy + (effective ? ": " + effective : " (none configured)")
        : "";
    }
    fillFromLines(toolsEl, d.tools_allowlist);
    toolsStrictEl.checked = !!d.tools_allowlist_strict;
    fillFromLines(skillsEl, d.skill_allowlist);
    skillsStrictEl.checked = !!d.skill_allowlist_strict;
    fillFromLines(disabledEl, d.disabled_tools);
    renderCatalogs();
    renderEffective();
    renderRuntimeExplainers();
    if (!options.preserveStatus) permsStatusEl.textContent = "";
  }

  function renderEffective() {
    const tools = state.def.effective_tools || [];
    effToolsCount.textContent = state.def.effective_tools_unrestricted
      ? "(" + tools.length + " — unrestricted)"
      : "(" + tools.length + ")";
    const q = state.toolsFilter.toLowerCase();
    const shown = q ? tools.filter((t) => t.toLowerCase().includes(q)) : tools;
    const mcpNames = new Set(state.registry.mcp || []);
    effToolsList.innerHTML = shown.length ? shown.map((t) => {
      const mcp = mcpNames.has(t) || String(t).startsWith("mcp__");
      return '<li class="' + (mcp ? "is-mcp" : "") + '"><span>' + esc(t) +
        '</span><span class="tag-source">' + (mcp ? "MCP" : "core") + "</span></li>";
    }).join("") : '<li class="effective-empty">No effective tools</li>';
    const skills = state.def.effective_skills || [];
    effSkillsCount.textContent = state.def.effective_skills_unrestricted
      ? "(" + skills.length + " — all)"
      : "(" + skills.length + ")";
    effSkillsList.innerHTML = skills.length ? skills.map((s) => '<li class="is-skill"><span>' + esc(s) +
      '</span><span class="tag-source">skill</span></li>').join("") : '<li class="effective-empty">No effective skills</li>';
  }

  function buildPatch() {
    const tools = lines(toolsEl.value);
    const skills = lines(skillsEl.value);
    const disabled = lines(disabledEl.value);
    const patch = {
      permission_mode: modeSel.value,
      tools_allowlist: tools.length ? tools : null,
      tools_allowlist_strict: tools.length ? toolsStrictEl.checked : false,
      skill_allowlist: skills.length ? skills : null,
      skill_allowlist_strict: skills.length ? skillsStrictEl.checked : false,
      disabled_tools: disabled.length ? disabled : null,
      sha: state.def.config_sha,
    };
    // Delegation-mode select: ship ONLY when the operator actually changed
    // it — leaving the default preselected writes nothing. Clearing the
    // select (value '') emits _UNSET so the key is removed and the profile
    // default (jarvis ⇒ background, build/plan ⇒ foreground) applies again.
    const currentDesc = (descInputEl ? descInputEl.value : (permsDescEl ? permsDescEl.value : "")).trim();
    const initialDesc = (state.descriptionInitial || "").trim();
    if (currentDesc !== initialDesc) {
      patch.description = currentDesc ? currentDesc : "_UNSET";
    }
    if (delegationSel.value !== state.delegationInitial) {
      patch.delegation_mode = delegationSel.value === ""
        ? "_UNSET"
        : delegationSel.value;
    }
    // Model override ships ONLY when the operator actually changed it —
    // leaving the default preselected writes nothing, so behavior is
    // byte-identical to before this feature. Switching to 'inherit
    // global' (value '') emits _UNSET so an existing per-agent override
    // is cleared from the UI. This applies to EVERY agent class: the select is always
    // populated from the catalogue and never disabled.
    if (modelSel.value !== state.modelInitial) {
      patch.upstream_model = modelSel.value === "" ? "_UNSET" : modelSel.value;
    }
    // Hardcoded prompt-section cards: ship ONLY the toggle state —
    // the sorted identifiers of the unchecked (disabled) cards. `null`
    // (server removes the key) when every card is enabled. NEVER a
    // composed prompt string: the server recomputes the prompt from its
    // canonical sections.
    const offSections = Object.keys(state.cardToggles)
      .filter((id) => !state.cardToggles[id])
      .sort();
    patch.disabled_prompt_sections = offSections.length ? offSections : null;
    return patch;
  }

  function showBanner(fields) {
    bannerFieldsEl.textContent = fields.join(", ");
    bannerEl.hidden = false;
  }

  permsForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const targetName = state.def && state.def.name;
    if (!currentAgentMatches(targetName)) {
      permsStatusEl.className = "perms-status err";
      permsStatusEl.textContent = state.loading
        ? "Wait for the selected agent to finish loading."
        : "Agent selection changed — save refused.";
      permsSaveBtn.disabled = state.loading;
      return;
    }
    const patch = buildPatch();
    permsStatusEl.className = "perms-status";
    permsStatusEl.textContent = "Saving…";
    permsSaveBtn.disabled = true;
    const res = await api("PATCH", "/api/agents/defs/" + encodeURIComponent(targetName), patch);
    if (!currentAgentMatches(targetName)) {
      permsSaveBtn.disabled = state.loading;
      return;
    }
    permsSaveBtn.disabled = false;
    if (res.ok) {
      state.def = { ...state.def, config_sha: res.payload.config_sha };
      state.permsDirty = false;
      syncDirty();
      updatedEl.textContent = "configuration saved " + fmtTime();
      permsStatusEl.className = "perms-status ok";
      permsStatusEl.textContent = "Saved (" + (res.payload.applied_keys || []).join(", ") + ").";
      const reload = res.payload.reload || {};
      if (reload.restart_only && reload.restart_only.length) {
        showBanner(reload.restart_only);
      } else if (reload.restart_only) {
        bannerEl.hidden = true;
      }
      // Refresh effective surface (backend recomputes it).
      const def = await api("GET", "/api/agents/defs/" + encodeURIComponent(targetName));
      if (def.ok && currentAgentMatches(targetName)) {
        state.def = def.payload;
        renderPerms({ preserveStatus: true });
        syncPromptCards(); // cards follow the persisted toggle state
        renderOverview();
      }
    } else if (res.status === 409) {
      permsStatusEl.className = "perms-status err";
      permsStatusEl.textContent = "Config changed on disk — reloading latest.";
      const def = await api("GET", "/api/agents/defs/" + encodeURIComponent(targetName));
      if (def.ok && currentAgentMatches(targetName)) {
        state.def = def.payload;
        state.permsDirty = false;
        syncDirty();
        renderPerms({ preserveStatus: true });
        syncPromptCards();
        renderOverview();
      }
    } else {
      const err = res.payload || {};
      const fieldErr = err.field_errors ? " — " + JSON.stringify(err.field_errors) : "";
      permsStatusEl.className = "perms-status err";
      permsStatusEl.textContent = (err.error || ("Save failed (" + res.status + ")")) + fieldErr;
    }
  });

  permsReloadBtn.addEventListener("click", async () => {
    const targetName = state.def && state.def.name;
    if (!currentAgentMatches(targetName)) return;
    const def = await api("GET", "/api/agents/defs/" + encodeURIComponent(targetName));
    if (def.ok && currentAgentMatches(targetName)) {
      state.def = def.payload;
      state.permsDirty = false;
      syncDirty();
      renderPerms();
      syncPromptCards();
      renderOverview();
    }
    const md = await api("GET", "/api/agents/defs/" + encodeURIComponent(targetName) + "/agents_md");
    if (md.ok && currentAgentMatches(targetName)) state.md = md.payload;
  });

  effToolsFilter.addEventListener("input", () => {
    state.toolsFilter = effToolsFilter.value;
    renderEffective();
  });

  bannerDismissEl.addEventListener("click", () => { bannerEl.hidden = true; });

  hcToggleEl.addEventListener("change", () => {
    state.showHardcoded = hcToggleEl.checked;
    sectionsEl.classList.toggle("sp-hc-hidden", !state.showHardcoded);
    updateHcNote();
  });

  promptFilterEl.addEventListener("input", () => {
    state.promptFilter = promptFilterEl.value;
    applyPromptFilter();
  });

  promptExpandEl.addEventListener("click", () => {
    state.promptExpanded = !state.promptExpanded;
    promptExpandEl.textContent = state.promptExpanded ? "Collapse all" : "Expand all";
    for (const section of sectionsEl.querySelectorAll(".sp-section")) {
      if (section.hidden || section.offsetParent === null) continue;
      const body = section.querySelector(".sp-body");
      const chevron = section.querySelector(".sp-chevron");
      if (body) body.hidden = !state.promptExpanded;
      if (chevron) chevron.textContent = state.promptExpanded ? "▾" : "▸";
    }
  });

  function bindPolicy(container, kind) {
    for (const button of container.querySelectorAll("[data-policy]")) {
      button.addEventListener("click", () => {
        const target = kind === "tool" ? toolsEl : skillsEl;
        const strict = kind === "tool" ? toolsStrictEl : skillsStrictEl;
        const isAll = lines(target.value).length === 0;
        if (button.dataset.policy === "all") {
          if (isAll && (kind !== "tool" || lines(disabledEl.value).length === 0)) return;
          target.value = "";
          strict.checked = false;
        } else {
          if (!isAll) return;
          const denied = kind === "tool" ? new Set(lines(disabledEl.value)) : new Set();
          target.value = (kind === "tool" ? allToolNames() : allSkillNames())
            .filter((name) => !denied.has(name))
            .join("\n");
        }
        if (kind === "tool") renderToolCatalog(); else renderSkillCatalog();
        markPermsDirty();
      });
    }
  }

  bindPolicy(toolPolicyEl, "tool");
  bindPolicy(skillPolicyEl, "skill");

  toolCatalogFilterEl.addEventListener("input", () => {
    state.toolCatalogFilter = toolCatalogFilterEl.value;
    renderToolCatalog();
  });
  skillCatalogFilterEl.addEventListener("input", () => {
    state.skillCatalogFilter = skillCatalogFilterEl.value;
    renderSkillCatalog();
  });
  toolsEl.addEventListener("input", renderToolCatalog);
  skillsEl.addEventListener("input", renderSkillCatalog);
  modeSel.addEventListener("change", renderRuntimeExplainers);
  delegationSel.addEventListener("change", renderRuntimeExplainers);

  for (const el of [modeSel, delegationSel, modelSel, toolsEl, toolsStrictEl, skillsEl, skillsStrictEl, disabledEl]) {
    el.addEventListener("input", markPermsDirty);
  }
  if (descInputEl) {
    descInputEl.addEventListener("input", () => {
      if (permsDescEl && permsDescEl.value !== descInputEl.value) permsDescEl.value = descInputEl.value;
      markPermsDirty();
    });
  }
  if (permsDescEl) {
    permsDescEl.addEventListener("input", () => {
      if (descInputEl && descInputEl.value !== permsDescEl.value) descInputEl.value = permsDescEl.value;
      markPermsDirty();
    });
  }

  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });

  init();
})();
