/* secrets.js — admin "Add secret" dialog + the Settings secrets list.

   The value goes from the password field to PUT /api/secrets (the gateway's
   secrets.env store) and is then wiped from the page. It never enters the
   composer, the chat, a session log or the model; agents read it from the
   environment variable of the same name. Loaded only for admins. */
(function () {
  "use strict";
  const modal = document.getElementById("secret-modal");
  if (!modal) return;
  const form = document.getElementById("secret-form");
  const nameEl = document.getElementById("secret-name");
  const valueEl = document.getElementById("secret-value");
  const agentEl = document.getElementById("secret-agent");
  const errEl = document.getElementById("secret-error");
  const saveBtn = document.getElementById("secret-save");
  const listEl = document.getElementById("secret-list");
  const listStatus = document.getElementById("secret-list-status");
  let fromComposer = false;
  let dialogGeneration = 0;

  async function call(method, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(BASE + "/api/secrets", opts);
    let data = {};
    try { data = await resp.json(); } catch (_) { /* empty body */ }
    if (!resp.ok) throw new Error(data.error || "Request failed");
    return data;
  }

  // The gateway names the agents a secret can be scoped to.
  function agentOptions(agents, selected) {
    const opts = ['<option value="">All agents</option>'].concat(
      agents.map((a) => `<option value="${escapeHtml(a)}">${escapeHtml(a)}</option>`));
    agentEl.innerHTML = opts.join("");
    agentEl.value = selected || "";
  }

  function wipe() {
    valueEl.value = "";
    errEl.textContent = "";
  }

  function open(opts) {
    const generation = ++dialogGeneration;
    fromComposer = !!(opts && opts.fromComposer);
    wipe();
    nameEl.value = "";
    agentOptions([], "");
    saveBtn.disabled = false;
    modal.classList.add("active");
    nameEl.focus();
    call("GET").then((data) => {
      if (generation === dialogGeneration && modal.classList.contains("active") && !agentEl.value)
        agentOptions(Array.isArray(data.agents) ? data.agents : [], "");
    }).catch((e) => {
      if (generation === dialogGeneration) errEl.textContent = e.message;
    });
  }

  function close() {
    ++dialogGeneration;
    wipe();
    modal.classList.remove("active");
  }

  function scopeText(agent) {
    return agent ? agent : "all agents";
  }

  async function refreshList() {
    if (!listEl) return;
    try {
      const data = await call("GET");
      const rows = Array.isArray(data.secrets) ? data.secrets : [];
      listEl.innerHTML = rows.map((s) =>
        `<li class="secret-row"><code>${escapeHtml(s.name)}</code>` +
        `<span class="secret-scope">${escapeHtml(scopeText(s.agent))}</span>` +
        `<button type="button" class="icon-btn secret-del" data-name="${escapeHtml(s.name)}"` +
        ` data-agent="${escapeHtml(s.agent || "")}" aria-label="Delete ${escapeHtml(s.name)}"` +
        ` title="Delete">&times;</button></li>`).join("");
      if (listStatus) listStatus.textContent = rows.length ? "" : "None saved";
    } catch (e) {
      listEl.innerHTML = "";
      if (listStatus) listStatus.textContent = e.message;
    }
  }

  function insertReference(name) {
    if (typeof messageInput === "undefined" || !messageInput) return;
    const ref = "Secret `" + name + "` is saved; use $" + name +
      " without printing or returning its value.";
    const draft = messageInput.value;
    messageInput.value = draft ? draft.replace(/\s*$/, "") + "\n\n" + ref : ref;
    if (typeof updateSendState === "function") updateSendState();
    if (typeof resizeComposer === "function") resizeComposer();
    messageInput.focus();
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errEl.textContent = "";
    const generation = dialogGeneration;
    const name = nameEl.value.trim();
    const agent = agentEl.value || null;
    saveBtn.disabled = true;
    try {
      await call("PUT", { name, value: valueEl.value, agent });
    } catch (err) {
      if (generation !== dialogGeneration) return;
      errEl.textContent = err.message;
      saveBtn.disabled = false;
      return;
    }
    // A response from a dismissed dialog must not wipe a new entry.
    if (generation !== dialogGeneration) return;
    // The current chat's agent gets this secret only when it is shared or
    // scoped to that agent; only then does a reference make sense.
    const reaches = !agent || agent === state.currentAgent;
    close();
    if (fromComposer && reaches) insertReference(name);
    refreshList();
  });

  document.getElementById("secret-close").addEventListener("click", close);
  document.getElementById("secret-cancel").addEventListener("click", close);
  modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
  modal.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.stopPropagation(); close(); }
  });

  const fromSettings = document.getElementById("secret-add-settings");
  if (fromSettings) fromSettings.addEventListener("click", () => open({}));
  const fromOptions = document.getElementById("opt-secret");
  if (fromOptions) fromOptions.addEventListener("click", () => {
    const card = document.getElementById("options-card");
    if (card) card.open = false;
    open({ fromComposer: true });
  });

  if (listEl) {
    listEl.addEventListener("click", async (e) => {
      const btn = e.target.closest(".secret-del");
      if (!btn) return;
      const name = btn.dataset.name;
      const agent = btn.dataset.agent || "";
      const ok = typeof confirmCard === "function"
        ? await confirmCard({
            title: "Delete secret",
            message: `Delete ${name} (${scopeText(agent)})?`,
            okLabel: "Delete",
            danger: true,
          })
        : true;
      if (!ok) return;
      try {
        await call("DELETE", { name, agent });
      } catch (err) {
        if (listStatus) listStatus.textContent = err.message;
        return;
      }
      refreshList();
    });
    const settingsBtn = document.getElementById("settings-btn");
    if (settingsBtn) settingsBtn.addEventListener("click", refreshList);
  }

  window.SecretDialog = { open, refreshList };
})();
