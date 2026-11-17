/* agent_model_picker.js — ONE shared Agent + Model picker.

   Every surface where the operator decides WHO runs a session and on WHICH
   model uses this class so the roster, the per-agent model catalogue, the
   defaults, and the persisted preferences are identical everywhere:

     * the chat composer picker preferences it reads/writes (the
       ``sharedagents.web.pickerAgent`` / ``sharedagents.web.pickerModels`` keys) are
       the SAME keys api.js/sessions.js use — pick in chat, the loop form is
       pre-seeded, and vice-versa;
     * the model list is the SAME ``POST /api/models/discover`` catalogue the
       chat picker renders (option label + badge), falling back to the
       legacy ``/api/command /models`` read when discovery is unavailable;
     * agents come from the same ``GET /api/models`` roster, minus the
       delegation-only names the chat picker hides (explore/general/…).

   The class drives two <select> elements the host page supplies (a plain
   agent select + a model select it populates lazily per agent). Host page
   responsibilities stay minimal: provide the markup, listen to
   ``onChange``, read ``getSelection()`` when submitting.

   Classic script sharing the global scope (window.AgentModelPicker). Load it
   BEFORE the page script that instantiates it.

   "Schedule a loop" form (picking the model for the agent that runs the
   loop).
 */
(function () {
  "use strict";

  function ampkEscape(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function ampkStorageGet(key) {
    try {
      return window.localStorage.getItem(key) || "";
    } catch (_e) {
      return "";
    }
  }

  function ampkStorageSet(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (_e) { /* storage unavailable — preference simply not kept */ }
  }

  function ampkBase() {
    try {
      const meta = document.querySelector('meta[name="sharedagents-base"]');
      return (meta && meta.content) || "";
    } catch (_e) {
      return "";
    }
  }

  // Delegation-only utilities the chat picker never offers as a session
  // agent (mirror of api.js isNonConversationalAgent).
  var NON_CONVERSATIONAL = /^(explore|general|docs|compaction|title|summary)$/;

  // Persistence keys — MUST stay byte-identical to api.js so the chat page
  // and every other AgentModelPicker share one preference set.
  var PICKER_AGENT_KEY = "sharedagents.web.pickerAgent";
  var PICKER_MODELS_KEY = "sharedagents.web.pickerModels";

  function ampkSavedAgent() {
    return ampkStorageGet(PICKER_AGENT_KEY).trim();
  }

  function ampkSavedModels() {
    try {
      var raw = ampkStorageGet(PICKER_MODELS_KEY);
      if (!raw) return {};
      var parsed = JSON.parse(raw);
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (_e) {
      return {};
    }
  }

  function ampkSavedModelFor(agent) {
    var map = ampkSavedModels();
    return String(map[agent] || "").trim();
  }

  function ampkPersist(agent, modelId) {
    var map = ampkSavedModels();
    if (modelId) map[agent] = modelId;
    else delete map[agent];
    ampkStorageSet(PICKER_MODELS_KEY, JSON.stringify(map));
    ampkPersistAgent(agent);
  }

  function ampkPersistAgent(agent) {
    if (agent) ampkStorageSet(PICKER_AGENT_KEY, agent);
  }

  class AgentModelPicker {
    /* opts: {
     *   agentSelect, modelSelect   — <select> elements (required)
     *   onChange?                  — (agentName, modelRow|null) => void
     *   agentLabel?                — (name, meta) => text for an option;
     *                                default: bare name
     *   initialAgent?              — force a starting agent (else the saved
     *                                chat picker agent, then jarvis)
     *   initialModel?              — force a starting model alias (else the
     *                                saved per-agent model, then default row)
     *   rosterUrl?                 — alternate picker-compatible roster API;
     *                                default remains GET /api/models
     *   filterAgents?              — fn(name) => bool, default hides
     *                                delegation-only agents like api.js
     *   filterModels?              — fn(normalizedRow) => bool; hosts may
     *                                restrict the catalogue to supported models
     * }
     */
    constructor(opts) {
      opts = opts || {};
      this.agentSelect = opts.agentSelect;
      this.modelSelect = opts.modelSelect;
      this.onChange = typeof opts.onChange === "function" ? opts.onChange : null;
      this.agentLabel = typeof opts.agentLabel === "function" ? opts.agentLabel : null;
      this.filterAgents = typeof opts.filterAgents === "function"
        ? opts.filterAgents
        : function (name) { return !NON_CONVERSATIONAL.test(String(name || "").trim()); };
      this.filterModels = typeof opts.filterModels === "function" ? opts.filterModels : () => true;
      this._forcedAgent = String(opts.initialAgent || "").trim();
      this._persistChanges = opts.persistChanges !== false;
      this._forcedModel = String(opts.initialModel || "").trim();
      this.rosterUrl = String(opts.rosterUrl || "/api/models");
      this._roster = [];            // [{name, description, engine}]
      this._modelsByAgent = {};     // agent -> normalized rows
      this._agent = "";
      this._model = null;           // selected row (catalogue id/label/wireId)
      this._loadingAgent = "";
      this._loadSeq = 0;
      this._rosterLoaded = false;

      if (this.agentSelect) {
        this.agentSelect.addEventListener("change", this._onAgentChange.bind(this));
      }
      if (this.modelSelect) {
        this.modelSelect.addEventListener("change", this._onModelChange.bind(this));
      }
      this._renderAgentsLoading();
      this._renderModelsLoading();
    }

    // ── public API ──

    getAgent() {
      return this._agent || (this.agentSelect ? String(this.agentSelect.value || "") : "");
    }

    // Selected model row: {id, label, wireId} | null. ``id`` is the catalogue
    // alias the chat picker stores/applies; ``wireId`` the real engine id.
    getModel() {
      return this._model;
    }

    // Alias id only (what POST /api/sessions stores / /model applies).
    getModelId() {
      var m = this.getModel();
      return m ? m.id : "";
    }

    // Read-only snapshots for card/sheet hosts that render the same catalogue
    // as the select projection.
    getAgents() {
      return this._roster.map(function (row) { return Object.assign({}, row); });
    }

    getModels(agent) {
      const name = String(agent || this._agent || "");
      return (this._modelsByAgent[name] || []).map(function (row) {
        return Object.assign({}, row);
      });
    }

    async selectAgent(agent) {
      const name = String(agent || "").trim();
      if (!name || !this._roster.some(function (row) { return row.name === name; })) return false;
      this._agent = name;
      if (this.agentSelect) this.agentSelect.value = name;
      if (this._persistChanges) ampkPersistAgent(name);
      this._model = null;
      await this._loadModelsFor(name);
      this._emitChange();
      return true;
    }

    selectModel(modelId) {
      const id = String(modelId || "").trim();
      const row = (this._modelsByAgent[this._agent] || []).find(function (item) {
        return item && item.id === id;
      }) || (this._modelsByAgent[this._agent] || []).find(function (item) {
        return item && item.alias === id;
      }) || null;
      if (!row) return false;
      this._model = row;
      if (this.modelSelect) this.modelSelect.value = row.id;
      if (this._persistChanges) ampkPersist(this._agent, row.id);
      this._emitChange();
      return true;
    }

    setDisabled(disabled) {
      if (this.agentSelect) this.agentSelect.disabled = !!disabled;
      if (this.modelSelect) {
        this.modelSelect.disabled = !!disabled ||
          !(this._modelsByAgent[this._agent] || []).length;
      }
    }

    commitSelection() {
      ampkPersist(this.getAgent(), this.getModelId());
    }

    // True once the roster fetch finished (even via a fallback).
    isLoaded() {
      return this._rosterLoaded;
    }

    // Load the roster + the current agent's models. Idempotent; safe to call
    // again after the agent catalogue changes server-side.
    async load() {
      await this._loadRoster();
      await this._loadModelsFor(this._agent);
      this._emitChange();
    }

    // ── roster ──

    async _loadRoster() {
      let options = [];
      let rosterFromModels = false;
      try {
        const resp = await window.fetch(ampkBase() + this.rosterUrl, {
          credentials: "same-origin",
        });
        const data = await resp.json().catch(function () { return {}; });
        if (resp.ok && Array.isArray(data.models)) {
          rosterFromModels = true;
          (data.models || []).forEach(function (m) {
            if (!m || !m.name) return;
            if (!this.filterAgents(m.name)) return;
            options.push({
              name: String(m.name),
              description: String(m.description || ""),
              engine: String(m.engine || ""),
              origin: String(m.origin || ""),
              external: !!m.external,
              // External-agent integrity chips (version + sign-in alert).
              version: String(m.version || ""),
              auth_state: String(m.auth_state || ""),
              auth_hint: String(m.auth_hint || ""),
              installed: m.installed !== false,
              auto_auth: !!m.auto_auth,
            });
          }, this);
        } else if (!resp.ok) {
          throw new Error(data.error || "HTTP " + resp.status);
        }
      } catch (_e) {
        // Gateway unreachable — fall back to agents seen on existing
        // sessions (same fallback the previous loops agent select used).
        try {
          const resp = await window.fetch(ampkBase() + "/api/sessions", {
            credentials: "same-origin",
          });
          const data = await resp.json().catch(function () { return {}; });
          const seen = {};
          (data.sessions || []).forEach(function (s) {
            const a = String((s && s.agent) || "");
            if (a && this.filterAgents(a) && !seen[a]) {
              seen[a] = 1;
              options.push({ name: a, description: "", engine: "" });
            }
          }, this);
        } catch (_e2) { /* keep default */ }
      }
      if (!options.length) options.push({ name: "jarvis", description: "", engine: "" });
      // Deterministic default: the operator's default agent ("jarvis") first
      // when present — API order must not decide the default (mirror of the
      // loops form's F5 rule).
      const jarvisIndex = options.findIndex(function (o) { return o.name === "jarvis"; });
      if (jarvisIndex > 0) {
        const picked = options.splice(jarvisIndex, 1)[0];
        options.unshift(picked);
      }
      this._roster = options;
      this._rosterFromModels = rosterFromModels;
      const saved = ampkSavedAgent();
      // The saved chat-composer agent is inherited ONLY
      // when the options came from /api/models (exact-match precondition).
      const wanted = this._forcedAgent ||
        (rosterFromModels && saved &&
          options.some(function (o) { return o.name === saved; }) ? saved : "") ||
        (options.some(function (o) { return o.name === "jarvis"; }) ? "jarvis" : options[0].name);
      this._agent = wanted;
      this._rosterLoaded = true;
      this._renderAgents();
    }

    _renderAgentsLoading() {
      if (!this.agentSelect) return;
      this.agentSelect.innerHTML =
        '<option value="">Loading agents…</option>';
      this.agentSelect.disabled = true;
    }

    _renderAgents() {
      if (!this.agentSelect) return;
      this.agentSelect.innerHTML = this._roster.map(function (o) {
        const text = this.agentLabel
          ? this.agentLabel(o.name, o)
          : o.name;
        return '<option value="' + ampkEscape(o.name) + '">' +
          ampkEscape(text) + "</option>";
      }, this).join("");
      this.agentSelect.disabled = false;
      this.agentSelect.value = this._agent;
    }

    // ── per-agent model catalogue ──

    // Normalized rows mirror the chat picker's loadModelsForAgent mapping
    // (api.js): the catalogue id is what the chat page stores in
    // sharedagents.web.pickerModels and what /model applies.
    _normalizeRow(item, index, hasExplicitDefault) {
      item = item || {};
      const id = String(item.id || item.alias || "");
      return {
        id: id,
        alias: String(item.alias || item.id || ""),
        label: String(item.display || item.title || item.label || item.id || ""),
        subtitle: String(item.subtitle || ""),
        wireId: String(item.wire_id || item.model_id || item.id || ""),
        realId: String(item.wire_id || item.model_id || item.id || ""),
        // An explicitly flagged default row suppresses the index-0 fallback
        // (parity with the chat picker's mapping).
        isDefault: item.default === true || (!hasExplicitDefault && index === 0),
        badge: String((item.badges || [])[0] || ""),
        contextWindow: Number(item.contextWindow || 0),
        provider: String(item.provider || ""),
      };
    }

    async _loadModelsFor(agent) {
      this._agent = agent;
      if (!agent) {
        this._model = null;
        this._renderModels();
        return;
      }
      this._loadSeq += 1;
      const seq = this._loadSeq;
      this._loadingAgent = agent;
      this._renderModelsLoading();
      let items = [];
      try {
        const resp = await window.fetch(ampkBase() + "/api/models/discover", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agent: agent, force: false }),
        });
        const data = await resp.json().catch(function () { return {}; });
        if (resp.ok && Array.isArray(data.models)) {
          items = data.models;
        } else {
          throw new Error((data && data.error) || "HTTP " + resp.status);
        }
      } catch (_e) {
        // Legacy fallback the chat picker uses when discovery is down.
        try {
          const resp = await window.fetch(ampkBase() + "/api/command", {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: "/models", agent: agent }),
          });
          const data = await resp.json().catch(function () { return {}; });
          items = (((data || {}).result || {}).data || {}).items || [];
        } catch (_e2) { /* keep empty */ }
      }
      if (seq !== this._loadSeq) return; // superseded by a newer agent switch
      this._loadingAgent = "";
      const explicitDefault = (items || []).some(
        function (it) { return it && it.default === true; });
      const rows = (items || []).map(
        (it, index) => this._normalizeRow(it, index, explicitDefault)).filter(this.filterModels);
      this._modelsByAgent[agent] = rows;
      this._applyModelSelection(agent, rows);
      this._emitChange();
    }

    _applyModelSelection(agent, rows) {
      const saved = ampkSavedModelFor(agent);
      let chosen = null;
      if (saved) {
        chosen = rows.find(function (r) { return r.id === saved; }) ||
          rows.find(function (r) { return r.realId === saved; }) || null;
      }
      if (!chosen && this._forcedModel && agent === this._forcedAgent) {
        chosen = rows.find(function (r) { return r.id === this._forcedModel; }, this) ||
          rows.find(function (r) { return r.realId === this._forcedModel; }, this) || null;
      }
      if (!chosen) {
        chosen = rows.find(function (r) { return r.isDefault; }) ||
          rows[0] || null;
      }
      this._model = chosen;
      this._renderModels();
    }

    _renderModelsLoading() {
      if (!this.modelSelect) return;
      this.modelSelect.innerHTML = '<option value="">Loading models…</option>';
      this.modelSelect.disabled = true;
    }

    _renderModels() {
      if (!this.modelSelect) return;
      const rows = (this._modelsByAgent[this._agent] || []).filter(Boolean);
      if (!rows.length) {
        this.modelSelect.innerHTML =
          '<option value="">' + (this._loadingAgent ? "Loading models…" : "— no models —") + "</option>";
        // A catalogue with no rows is not a choice — keep the select locked
        // (same "— no models —" disabled state the chat picker renders).
        this.modelSelect.disabled = true;
        return;
      }
      this.modelSelect.innerHTML = rows.map(function (r) {
        const badge = r.badge ? " · " + r.badge : "";
        return '<option value="' + ampkEscape(r.id) + '">' +
          ampkEscape(r.label + badge) + "</option>";
      }).join("");
      this.modelSelect.disabled = false;
      if (this._model) this.modelSelect.value = this._model.id;
    }

    // ── events ──

    _onAgentChange() {
      const next = this.agentSelect ? String(this.agentSelect.value || "") : "";
      if (!next || next === this._agent) return;
      this._agent = next;
      // Remember the agent choice ONLY — the previous agent's saved model
      // preference stays in the map untouched; the new agent's selection
      // starts at its own saved/default row.
      if (this._persistChanges) ampkPersistAgent(next);
      this._model = null;
      this._loadModelsFor(next);
      this._emitChange();
    }

    _onModelChange() {
      if (!this.modelSelect) return;
      const id = String(this.modelSelect.value || "");
      const rows = (this._modelsByAgent[this._agent] || []).filter(Boolean);
      const row = rows.find(function (r) { return r.id === id; }) || null;
      this._model = row;
      if (row && this._persistChanges) ampkPersist(this._agent, row.id);
      this._emitChange();
    }

    _emitChange() {
      if (!this.onChange) return;
      try {
        this.onChange(this.getAgent(), this.getModel());
      } catch (_e) { /* a broken listener must not break the picker */ }
    }
  }

  // Classic script: global on window (browser) / globalThis (node harness).
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { AgentModelPicker: AgentModelPicker };
  }
  window.AgentModelPicker = AgentModelPicker;
})();
