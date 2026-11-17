/* Card popup reusing chat's catalogue renderer, styles and picker preferences. */
(function () {
  "use strict";
  let sequence = 0;
  class AgentModelDialog {
    constructor({ onSave, fixedAgent = "", title = "Agent & Model", filterModels,
                  persistSelection = true, allowDefault = false }) {
      const id = `shared-amp-${++sequence}`;
      this.onSave = onSave;
      this.fixedAgent = fixedAgent;
      this.persistSelection = persistSelection;
      this.provider = "";
      this.tab = "agent";
      this.busy = false;
      this.dialog = document.createElement("dialog");
      this.dialog.className = "agent-model-pop shared-agent-model-dialog";
      this.dialog.setAttribute("aria-labelledby", `${id}-title`);
      this.dialog.innerHTML = `
        <div class="amp-head"><span id="${id}-title" class="amp-title">Agent &amp; Model</span>
          <button type="button" class="amp-close" data-dismiss aria-label="Close agent and model picker">×</button></div>
        <div class="amp-tabs-nav" role="tablist" aria-label="Selection mode">
          <button type="button" class="amp-tab-btn" data-tab="agent" role="tab" id="${id}-agent-tab" aria-controls="${id}-agents">Agent <span class="amp-tab-badge" data-agent-badge></span></button>
          <button type="button" class="amp-tab-btn" data-tab="model" role="tab" id="${id}-model-tab" aria-controls="${id}-models">Model <span class="amp-tab-badge" data-model-badge></span></button>
        </div>
        <div class="amp-body">
          <section id="${id}-agents" data-panel="agent" role="tabpanel" aria-labelledby="${id}-agent-tab"><div class="agent-card" data-agents></div></section>
          <section id="${id}-models" data-panel="model" role="tabpanel" aria-labelledby="${id}-model-tab" hidden>
            <div class="amp-model-grid"><div class="amp-provider-tabs" data-providers role="tablist" aria-label="Providers"></div>
              <div class="amp-model-pane"><label class="amp-model-search-wrap"><input class="amp-model-search" type="search" aria-label="Search models" placeholder="Search models"></label>
                <div class="amp-model-cards" data-models></div></div></div>
          </section>
        </div>
        <p class="amp-error" role="alert" data-error hidden></p>
        <div class="amp-actions"><button type="button" class="amp-action" data-default hidden>Use default</button>
          <button type="button" class="amp-action" data-dismiss>Discard</button>
          <button type="button" class="amp-action amp-save" data-save disabled>Save</button></div>`;
      this.dialog.querySelector(".amp-title").textContent = title;
      this.dialog.querySelector("[data-default]").hidden = !allowDefault;
      this.dialog.querySelector(".amp-tabs-nav").hidden = !!fixedAgent;
      document.body.appendChild(this.dialog);
      this.picker = new AgentModelPicker({
        agentSelect: document.createElement("select"), modelSelect: document.createElement("select"),
        persistChanges: false, onChange: () => this.render(),
        initialAgent: fixedAgent, filterModels,
        ...(fixedAgent ? { filterAgents: name => name === fixedAgent } : {}),
      });
      this.dialog.addEventListener("click", async (ev) => {
        const btn = ev.target.closest("button");
        if (!btn) return;
        if (btn.hasAttribute("data-dismiss")) this.dialog.close();
        if (btn.hasAttribute("data-tab")) { this.tab = btn.dataset.tab; this.render(); }
        if (btn.hasAttribute("data-agent") && !this.busy) {
          this.busy = true; this.render();
          try { await this.picker.selectAgent(btn.dataset.agent); }
          finally { this.busy = false; }
          this.provider = this.picker.getModel()?.provider || "";
          this.tab = "model";
          this.dialog.querySelector("input").value = "";
          this.render();
        }
        if (btn.hasAttribute("data-provider")) { this.provider = btn.dataset.provider; this.render(); }
        if (btn.hasAttribute("data-model")) this.picker.selectModel(btn.dataset.model);
        if (btn.hasAttribute("data-save") && !this.busy && this.picker.getModelId())
          await this.save({ agent: this.picker.getAgent(), model: this.picker.getModelId(),
                            alias: this.picker.getModel().alias, label: this.picker.getModel().label });
        if (btn.hasAttribute("data-default") && !this.busy)
          await this.save({ agent: this.picker.getAgent(), model: "", label: "Default model" });
      });
      this.dialog.addEventListener("keydown", (ev) => {
        const tab = ev.target.closest('[role="tab"]');
        if (!tab || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(ev.key)) return;
        ev.preventDefault();
        const siblings = [...tab.parentElement.querySelectorAll('[role="tab"]')];
        const index = siblings.indexOf(tab);
        const next = ev.key === "Home" ? 0 : ev.key === "End" ? siblings.length - 1 : (index + (ev.key === "ArrowRight" ? 1 : -1) + siblings.length) % siblings.length;
        const attr = tab.hasAttribute("data-tab") ? "data-tab" : "data-provider";
        const value = siblings[next].getAttribute(attr);
        siblings[next].click();
        [...this.dialog.querySelectorAll(`[${attr}]`)].find(el => el.getAttribute(attr) === value)?.focus();
      });
      this.dialog.querySelector("input").addEventListener("input", (ev) => {
        const match = this.picker.getModels().find(row => AgentModelCatalogView.matchesQuery(row, ev.target.value));
        if (match) this.provider = match.provider || "Other";
        this.render();
      });
    }
    async open(selection) {
      this.tab = this.fixedAgent ? "model" : "agent";
      this.dialog.querySelector("input").value = "";
      this.dialog.querySelector("[data-error]").hidden = true;
      this.busy = true;
      this.dialog.showModal();
      this.render();
      try {
        await this.picker.load();
        if (selection?.agent && !this.fixedAgent) {
          await this.picker.selectAgent(selection.agent);
        }
        if (selection?.model) this.picker.selectModel(selection.model);
        this.provider = this.picker.getModel()?.provider || "";
      } finally { this.busy = false; this.render(); }
    }
    async save(selection) {
      this.busy = true;
      this.render();
      const error = this.dialog.querySelector("[data-error]");
      error.hidden = true;
      try {
        await this.onSave(selection);
        if (this.persistSelection) this.picker.commitSelection();
        this.dialog.close();
      } catch (err) {
        error.textContent = err.message || "Could not save model.";
        error.hidden = false;
      } finally { this.busy = false; this.render(); }
    }
    render() {
      if (!this.picker) return;
      const root = this.dialog;
      root.setAttribute("aria-busy", String(this.busy));
      root.querySelector("[data-agent-badge]").textContent = this.picker.getAgent();
      root.querySelector("[data-model-badge]").textContent = this.picker.getModel()?.label || "";
      root.querySelectorAll("[data-tab]").forEach(el => {
        const active = el.dataset.tab === this.tab;
        el.classList.toggle("is-active", active); el.setAttribute("aria-selected", String(active)); el.tabIndex = active ? 0 : -1;
      });
      root.querySelectorAll("[data-panel]").forEach(el => { el.hidden = el.dataset.panel !== this.tab; });
      AgentModelCatalogView.renderAgents({ host: root.querySelector("[data-agents]"), agents: this.picker.getAgents(), current: this.picker.getAgent(), disabled: this.busy });
      const view = AgentModelCatalogView.renderModels({
        models: this.picker.getModels(), provider: this.provider, selectedModel: this.picker.getModelId(), disabled: this.busy,
        query: root.querySelector("input").value, providersHost: root.querySelector("[data-providers]"), modelsHost: root.querySelector("[data-models]"),
        emptyHtml: '<p class="amp-empty">No models available. Reopen the picker to retry.</p>',
      });
      this.provider = view.provider;
      root.querySelector("[data-save]").disabled = this.busy || !this.picker.getModelId();
      root.querySelector("[data-default]").disabled = this.busy;
    }
  }
  window.AgentModelDialog = AgentModelDialog;
})();
