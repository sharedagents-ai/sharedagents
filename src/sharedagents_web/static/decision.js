/* Decision tab — manual typed-decision playground. */
(function () {
  "use strict";

  const BASE =
    document.querySelector('meta[name="sharedagents-base"]')?.content || "";

  const appEl = document.getElementById("decision-app");
  const connEl = document.getElementById("decision-conn");
  const updatedEl = document.getElementById("decision-updated");
  const statusEl = document.getElementById("decision-status");
  const backendsEl = document.getElementById("decision-backends");
  const backendMetaEl = document.getElementById("decision-backend-meta");
  const checkpointWrapEl = document.getElementById("decision-checkpoint-wrap");
  const checkpointEl = document.getElementById("decision-checkpoint");
  const stateEl = document.getElementById("decision-state");
  const questionsEl = document.getElementById("decision-questions");
  const resultsEl = document.getElementById("decision-results");
  const resultsEmptyEl = document.getElementById("decision-results-empty");
  const rawEl = document.getElementById("decision-raw");
  const showRawEl = document.getElementById("decision-show-raw");
  const scoreBtn = document.getElementById("decision-score");
  const latencyEl = document.getElementById("decision-latency");
  const progressEl = document.getElementById("decision-progress");

  let catalog = { models: [], sample: null };
  let selectedProvider = "";
  let running = false;
  let suggesting = false;
  let suggestionSelection = null;
  const suggestBtn = document.getElementById("decision-suggest");
  const suggestPickerBtn = document.getElementById("decision-suggest-picker");
  const suggestStatus = document.getElementById("decision-suggest-status");
  const suggestSession = document.getElementById("decision-suggest-session");
  const modelDialog = document.getElementById("decision-model-dialog");
  const suggestionPicker = new AgentModelDialog({ onSave(selection) {
    suggestionSelection = selection;
    suggestPickerBtn.textContent = `${selection.agent} · ${selection.label}`;
    syncQuestions();
  }});
  const providersEl = document.getElementById("decision-providers");
  const searchEl = document.getElementById("decision-model-search");
  let selectedModel = "";
  let lastResult = null;
  let questionSeq = 0;

  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function setStatus(message, isError) {
    if (!statusEl) return;
    if (!message) {
      statusEl.hidden = true;
      statusEl.textContent = "";
      statusEl.classList.remove("is-error");
      return;
    }
    statusEl.hidden = false;
    statusEl.textContent = message;
    statusEl.classList.toggle("is-error", !!isError);
  }

  function setConn(ok) {
    if (!connEl) return;
    connEl.classList.toggle("mon-conn-live", !!ok);
    connEl.classList.toggle("mon-conn-unknown", !ok);
    connEl.title = ok ? "Decision API reachable" : "Decision API status unknown";
  }

  function selectedBackend() {
    return (catalog.models || []).find((row) => row.id === selectedModel) || null;
  }

  function renderBackends() {
    const view = AgentModelCatalogView.renderModels({
      models: catalog.models, provider: selectedProvider, selectedModel,
      query: searchEl.value, providersHost: providersEl, modelsHost: backendsEl,
      disabled: running || suggesting,
      emptyHtml: '<p class="amp-empty">No matching decision models. Check your model configuration or refresh.</p>',
    });
    selectedProvider = view.provider;
    const row = selectedBackend();
    if (backendMetaEl) backendMetaEl.textContent = row ? `${row.provider} / ${row.label}` : "Select a model";
    document.getElementById("decision-score-hint").textContent = row ? `${row.provider} · ${row.label}` : "Choose a model to begin";
    scoreBtn.disabled = running || suggesting || !row || !questionsEl.children.length;
    renderCheckpoint();
  }

  function renderCheckpoint() {
    const row = selectedBackend();
    const checkpoints = (row && row.checkpoints) || [];
    if (!checkpointWrapEl || !checkpointEl) return;
    const previous = checkpointEl.value;
    checkpointWrapEl.hidden = checkpoints.length === 0;
    checkpointEl.innerHTML = checkpoints.map((name) => (
      `<option value="${esc(name)}">${esc(name)}</option>`
    )).join("");
    if (checkpoints.includes(previous)) checkpointEl.value = previous;
    else if (row && row.default_checkpoint) checkpointEl.value = row.default_checkpoint;
  }

  function fitQuestionText(root = questionsEl) {
    root.querySelectorAll("textarea.decision-autogrow").forEach(field => {
      if (!field.getClientRects().length) return;
      field.style.height = "auto";
      field.style.height = `${field.scrollHeight + 2}px`;
    });
  }
  // Width can change through the sidebar as well as a viewport resize.
  let questionWidth = 0;
  new ResizeObserver(entries => {
    const width = entries[0].contentRect.width;
    if (width !== questionWidth) { questionWidth = width; fitQuestionText(); }
  }).observe(questionsEl);

  function addQuestion(seed) {
    if (questionsEl.children.length >= 8) { setStatus("You can ask up to 8 questions per run.", true); return; }
    questionSeq += 1;
    const id = "q" + questionSeq;
    const name = (seed && seed.name) || ("q" + questionSeq);
    const type = (seed && seed.type) || "choice";
    const instructions = (seed && seed.instructions) || "";
    const card = document.createElement("article");
    card.className = "decision-question";
    const expanded = !seed || questionsEl.children.length === 0;
    card.dataset.qid = id;
    card.innerHTML =
      `<header class="decision-question-toolbar">` +
      `<button type="button" class="decision-question-toggle" data-toggle-q aria-expanded="${expanded}" aria-controls="${id}-body">` +
      `<svg class="decision-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m9 5 7 7-7 7"/></svg>` +
      `<span class="decision-question-label">${esc(name)}</span><span class="decision-question-kind"></span></button>` +
      `<button type="button" class="tasks-btn tasks-btn-ghost decision-delete-btn" data-remove-q>Delete</button></header>` +
      `<div class="decision-question-body" id="${id}-body"${expanded ? "" : " hidden"}>` +
      `<div class="decision-question-head">` +
      `<label class="decision-field"><span>Name</span>` +
      `<textarea class="decision-q-name decision-autogrow" rows="1">${esc(name)}</textarea></label>` +
      `<label class="decision-field"><span>Type</span>` +
      `<select class="decision-q-type">` +
      `<option value="choice">Choice</option>` +
      `<option value="score">Rating</option>` +
      `<option value="noul">Yes / No</option>` +
      `</select></label>` +
      `</div>` +
      `<label class="decision-field"><span>Instructions</span>` +
      `<textarea class="decision-q-instructions decision-autogrow" rows="1">${esc(instructions)}</textarea></label>` +
      `<div class="decision-q-criteria"></div>` +
      `<button type="button" class="tasks-btn tasks-btn-ghost" data-add-option>Add option</button></div>`;
    questionsEl.appendChild(card);
    card.querySelector(".decision-q-type").value = type;
    const criteria = seed && seed.criteria;
    if (type === "choice" && criteria && typeof criteria === "object" && !Array.isArray(criteria)) {
      Object.entries(criteria).forEach(([key, desc]) => addOptionRow(card, key, desc));
    } else if (type === "score" && Array.isArray(criteria)) {
      criteria.forEach((level) => addLevelRow(card, level));
    } else if (type === "choice") {
      addOptionRow(card, "HOLD", "wait for review");
      addOptionRow(card, "ABSTAIN", "insufficient evidence");
    } else if (type === "score") {
      addLevelRow(card, "routine");
      addLevelRow(card, "urgent");
    }
    syncCriteriaChrome(card);
    fitQuestionText(card);
    syncQuestions();
    if (!seed) card.querySelector(".decision-q-name").focus();
  }

  function syncQuestions() {
    const count = questionsEl.children.length;
    document.getElementById("decision-questions-empty").hidden = count !== 0;
    document.getElementById("decision-question-count").textContent = `${count} / 8`;
    document.getElementById("decision-add-question").disabled = running || suggesting || count >= 8;
    scoreBtn.disabled = running || suggesting || !selectedBackend() || count === 0;
    suggestBtn.disabled = running || suggesting || count >= 8 || !stateEl.value.trim() || !suggestionSelection;
    suggestPickerBtn.disabled = running || suggesting;
  }

  function suggestionSessionLink(sid) {
    if (!sid) return;
    suggestSession.href = BASE + "/?session=" + encodeURIComponent(sid);
    suggestSession.hidden = false;
  }

  async function suggestQuestions() {
    if (running || suggesting || !suggestionSelection) return;
    suggesting = true;
    suggestBtn.textContent = "Sending…";
    suggestStatus.textContent = "Sending context to the selected agent…";
    suggestStatus.classList.remove("is-error");
    suggestSession.hidden = true;
    syncQuestions();
    // Keep context and existing questions unchanged while the reply is pending.
    stateEl.disabled = true;
    questionsEl.inert = true;
    document.getElementById("decision-load-sample").disabled = true;
    let reader;
    try {
      const existingNames = [...questionsEl.querySelectorAll(".decision-q-name")].map(el => el.value.trim());
      const resp = await fetch(BASE + "/api/decision/suggest", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...suggestionSelection, state: parseState(stateEl.value), existing_names: existingNames }),
      });
      if (!resp.ok) {
        const error = await resp.json();
        suggestionSessionLink(error.session_id);
        throw new Error(error.error || "Could not suggest questions");
      }
      if (!resp.headers.get("Content-Type")?.includes("application/x-ndjson")) throw new Error("Unexpected response. Refresh the page and try again.");
      reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      let questions;
      while (true) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        const lines = pending.split("\n");
        pending = lines.pop();
        if (done && pending.trim()) lines.push(pending);
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          suggestionSessionLink(event.session_id);
          if (event.status === "running") { suggestBtn.textContent = "Suggesting…"; suggestStatus.textContent = "Running — drafting questions from your context…"; }
          if (event.status === "error") throw new Error(event.message || "Could not suggest questions");
          if (event.status === "done") questions = event.questions;
        }
        if (done) break;
      }
      if (!questions || typeof questions !== "object" || !Object.keys(questions).length) throw new Error("No questions returned. Try again.");
      const used = new Set(existingNames);
      Object.entries(questions).forEach(([name, question]) => {
        let unique = name;
        for (let suffix = 2; used.has(unique); suffix++) unique = `${name}_${suffix}`;
        used.add(unique);
        addQuestion({ ...question, name: unique });
      });
      renderResult(null);
      setProgress("", "");
      suggestStatus.textContent = `Done — added ${Object.keys(questions).length} questions. Review or edit them before running.`;
    } catch (err) {
      suggestStatus.textContent = `Error — ${err.message || "Could not suggest questions. Try again."}`;
      suggestStatus.classList.add("is-error");
    } finally {
      if (reader) { await reader.cancel().catch(() => {}); reader.releaseLock(); }
      suggesting = false;
      stateEl.disabled = false;
      questionsEl.inert = false;
      document.getElementById("decision-load-sample").disabled = false;
      suggestBtn.textContent = "Suggest questions";
      syncQuestions();
    }
  }

  function addOptionRow(card, key, desc) {
    const box = card.querySelector(".decision-q-criteria");
    const row = document.createElement("div");
    row.className = "decision-option-row";
    row.innerHTML =
      `<label class="decision-field"><span>Label</span>` +
      `<textarea class="decision-opt-key decision-autogrow" rows="1">${esc(key || "")}</textarea></label>` +
      `<label class="decision-field"><span>Description</span>` +
      `<textarea class="decision-opt-desc decision-autogrow" rows="1">${esc(desc || "")}</textarea></label>` +
      `<button type="button" class="decision-icon-btn" data-remove-row aria-label="Remove option" title="Remove option">&times;</button>`;
    box.appendChild(row);
    fitQuestionText(row);
  }

  function addLevelRow(card, level) {
    const box = card.querySelector(".decision-q-criteria");
    const row = document.createElement("div");
    row.className = "decision-level-row";
    row.innerHTML =
      `<label class="decision-field"><span>Level</span>` +
      `<textarea class="decision-opt-level decision-autogrow" rows="1">${esc(level || "")}</textarea></label>` +
      `<button type="button" class="decision-icon-btn" data-remove-row aria-label="Remove option" title="Remove option">&times;</button>`;
    box.appendChild(row);
    fitQuestionText(row);
  }

  function syncCriteriaChrome(card) {
    const type = card.querySelector(".decision-q-type").value;
    const addBtn = card.querySelector("[data-add-option]");
    addBtn.hidden = type === "noul";
    addBtn.textContent = type === "score" ? "Add level" : "Add option";
    card.querySelector("[data-remove-q]").setAttribute("aria-label", `Delete question ${card.querySelector(".decision-q-name").value || "Untitled question"}`);
    card.querySelector(".decision-question-kind").textContent = {choice: "Choice", score: "Rating", noul: "Yes / No"}[type];
  }

  function collectQuestions() {
    const questions = Object.create(null);
    questionsEl.querySelectorAll(".decision-question").forEach((card) => {
      const name = card.querySelector(".decision-q-name").value.trim();
      const type = card.querySelector(".decision-q-type").value;
      const instructions = card.querySelector(".decision-q-instructions").value.trim();
      if (!name) throw new Error("Give every question a name.");
      if (Object.hasOwn(questions, name)) throw new Error("Question names must be unique.");
      if (!instructions) throw new Error("Add instructions to every question.");
      const item = { type, instructions };
      if (type === "choice") {
        const criteria = Object.create(null);
        card.querySelectorAll(".decision-option-row").forEach((row) => {
          const key = row.querySelector(".decision-opt-key").value.trim();
          const desc = row.querySelector(".decision-opt-desc").value;
          if (!key || Object.hasOwn(criteria, key)) throw new Error("Option labels must be nonempty and unique.");
          criteria[key] = desc;
        });
        item.criteria = criteria;
      } else if (type === "score") {
        item.criteria = Array.from(card.querySelectorAll(".decision-opt-level"))
          .map((el) => el.value.trim())
          .filter(Boolean);
      }
      questions[name] = item;
    });
    if (!Object.keys(questions).length) throw new Error("Add at least one question.");
    return questions;
  }

  function parseState(raw) {
    const text = String(raw || "");
    if (!text.trim()) throw new Error("Add some context before running a decision.");
    if (text.trim().startsWith("{") || text.trim().startsWith("[")) {
      try { return JSON.parse(text); } catch (_err) { /* Preserve plain text verbatim. */ }
    }
    return { text };
  }

  function setProgress(stage, message) {
    progressEl.dataset.stage = stage;
    progressEl.textContent = message || "";
    progressEl.classList.toggle("is-error", stage === "error");
    progressEl.hidden = !stage;
    resultsEl.setAttribute("aria-busy", String(stage === "sending" || stage === "running"));
    if (stage === "sending") scoreBtn.textContent = "Sending…";
    if (stage === "running") scoreBtn.textContent = "Running…";
  }

  async function readScoreResponse(resp) {
    if (!resp.ok || !resp.headers.get("Content-Type")?.includes("application/x-ndjson")) {
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.message || data.error || `Request failed (${resp.status})`);
      return data;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    let result;
    try {
      while (true) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value, { stream: !done });
        const lines = pending.split("\n");
        pending = lines.pop();
        if (done && pending.trim()) lines.push(pending);
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (event.status === "running") setProgress("running", "Running — waiting for the model…");
          else if (event.status === "error") throw new Error(event.result?.message || event.result?.error || "Decision failed");
          else if (event.status === "done") result = event.result;
        }
        if (done) break;
      }
    } finally {
      await reader.cancel();
      reader.releaseLock();
    }
    if (!result || !result.answers || !Object.keys(result.answers).length) throw new Error("The model returned no answers. Try again.");
    return result;
  }

  function loadSample() {
    renderResult(null);
    setStatus("", false);
    setProgress("", "");
    const sample = catalog.sample || {
      state: { message: "Synthetic test: please review a duplicate invoice. No action is authorized." },
      questions: {
        action: {
          type: "choice",
          instructions: "Choose a review action.",
          criteria: { HOLD: "wait for review", ABSTAIN: "insufficient evidence" },
        },
        urgency: { type: "score", instructions: "Rate urgency.", criteria: ["routine", "urgent"] },
        needs_review: { type: "noul", instructions: "Does the message request review?" },
      },
    };
    stateEl.value = typeof sample.state === "string"
      ? sample.state
      : JSON.stringify(sample.state, null, 2);
    questionsEl.innerHTML = "";
    Object.entries(sample.questions || {}).sort((a, b) =>
      ["choice", "score", "noul"].indexOf(a[1].type) - ["choice", "score", "noul"].indexOf(b[1].type)
    ).forEach(([name, q]) => {
      addQuestion({ name, type: q.type, instructions: q.instructions, criteria: q.criteria });
    });
  }

  function renderResult(body, elapsedMs) {
    lastResult = body;
    latencyEl.hidden = !Number.isFinite(elapsedMs);
    latencyEl.textContent = Number.isFinite(elapsedMs) ? `· Latency: ${Math.round(elapsedMs).toLocaleString()} ms` : "";
    const answers = body && body.answers;
    const hasAnswers = answers && typeof answers === "object" && Object.keys(answers).length;
    resultsEmptyEl.hidden = !!hasAnswers;
    resultsEl.hidden = !hasAnswers;
    rawEl.textContent = body ? JSON.stringify(body, null, 2) : "";
    rawEl.hidden = !(showRawEl && showRawEl.checked);
    if (!hasAnswers) {
      resultsEl.innerHTML = "";
      return;
    }
    resultsEl.innerHTML = Object.entries(answers).map(([name, answer]) => {
      const value = answer.choice != null
        ? answer.choice
        : (answer.score != null ? String(answer.score) : (answer.noul != null ? String(answer.noul) : "—"));
      const probs = answer.probabilities || {};
      const bars = Object.entries(probs).map(([label, p]) => {
        const pct = Math.max(0, Math.min(100, Math.round(Number(p) * 1000) / 10));
        return `<div class="decision-bar"><span>${esc(label)}</span>` +
          `<div class="decision-bar-track"><div class="decision-bar-fill" style="width:${pct}%"></div></div>` +
          `<span>${pct.toFixed(1)}%</span></div>`;
      }).join("");
      const extra = [
        answer.type || "",
        answer.entropy_certainty != null ? `entropy ${answer.entropy_certainty}` : "",
        body.latency_ms != null ? `scorer ${body.latency_ms} ms` : "",
      ].filter(Boolean).join(" · ");
      return `<article class="decision-answer" data-answer="${esc(name)}">` +
        `<div class="decision-answer-head"><span class="decision-answer-name">${esc(name)}</span>` +
        `<span class="decision-answer-value">${esc(value)}</span></div>` +
        `<div class="decision-meta">${esc(extra)}</div>${bars}</article>`;
    }).join("");
  }

  async function loadCatalog(force = false) {
    try {
      const resp = await fetch(BASE + "/api/decision/catalog" + (force ? "?refresh=1" : ""));
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.error || "catalog failed");
      catalog = data;
      if (!catalog.models.some(row => row.id === selectedModel)) {
        selectedModel = catalog.default_model || "";
        selectedProvider = selectedBackend()?.provider || "";
      }
      setStatus(selectedModel ? "" : "Select an available model to begin.", false);
      setConn(true);
      if (updatedEl) updatedEl.textContent = `${catalog.models.length} models`;
      renderBackends();
      syncQuestions();
      appEl.classList.remove("is-loading");
    } catch (err) {
      setConn(false);
      setStatus(err.message || "catalog failed", true);
      appEl.classList.remove("is-loading");
    }
  }

  async function score() {
    if (running || suggesting) return;
    setStatus("", false);
    setProgress("", "");
    running = true;
    syncQuestions();
    document.getElementById("decision-refresh").disabled = true;
    searchEl.disabled = true;
    checkpointEl.disabled = true;
    renderBackends();
    renderResult(null);
    document.querySelector(".decision-editor").inert = true;
    let startedAt;
    try {
      const body = {
        model: selectedModel,
        state: parseState(stateEl.value),
        questions: collectQuestions(),
      };
      stateEl.value = JSON.stringify(body.state, null, 2);
      if (checkpointEl && !checkpointWrapEl.hidden) body.checkpoint = checkpointEl.value;
      setProgress("sending", "Sending — submitting your decision…");
      startedAt = performance.now();
      const resp = await fetch(BASE + "/api/decision/score", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/x-ndjson" },
        body: JSON.stringify(body),
      });
      const data = await readScoreResponse(resp);
      const elapsedMs = performance.now() - startedAt;
      renderResult(data, elapsedMs);
      setProgress("done", "Done — answers are ready.");
    } catch (err) {
      const message = err.message || "Decision failed. Try again.";
      setProgress("error", `Error — ${message}`);
      setStatus(message, true);
      renderResult({ error: message }, startedAt === undefined ? undefined : performance.now() - startedAt);
    } finally {
      running = false;
      document.getElementById("decision-refresh").disabled = false;
      searchEl.disabled = false;
      checkpointEl.disabled = false;
      document.querySelector(".decision-editor").inert = false;
      scoreBtn.textContent = "Run decision";
      renderBackends();
      syncQuestions();
    }
  }

  backendsEl.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-model]");
    if (!btn || running) return;
    selectedModel = btn.getAttribute("data-model");
    checkpointEl.value = "";
    renderBackends();
    modelDialog.close();
  });
  providersEl.addEventListener("click", (ev) => {
    const tab = ev.target.closest("[data-provider]");
    if (!tab || running) return;
    selectedProvider = tab.dataset.provider;
    renderBackends();
  });
  providersEl.addEventListener("keydown", (ev) => {
    if (running || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(ev.key)) return;
    const tabs = Array.from(providersEl.querySelectorAll("[data-provider]"));
    const index = tabs.indexOf(document.activeElement);
    if (index < 0) return;
    ev.preventDefault();
    const next = ev.key === "Home" ? 0 : ev.key === "End" ? tabs.length - 1 : (index + (ev.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    selectedProvider = tabs[next].dataset.provider;
    renderBackends();
    providersEl.querySelectorAll("[data-provider]")[next].focus();
  });
  searchEl.addEventListener("input", () => {
    const match = catalog.models.find(row => AgentModelCatalogView.matchesQuery(row, searchEl.value));
    if (match) selectedProvider = match.provider;
    renderBackends();
  });
  questionsEl.addEventListener("click", (ev) => {
    const card = ev.target.closest(".decision-question");
    if (!card || running) return;
    if (ev.target.closest("[data-toggle-q]")) {
      const toggle = card.querySelector("[data-toggle-q]");
      const expanded = toggle.getAttribute("aria-expanded") !== "true";
      toggle.setAttribute("aria-expanded", String(expanded));
      card.querySelector(".decision-question-body").hidden = !expanded;
      if (expanded) fitQuestionText(card);
      return;
    }
    if (ev.target.closest("[data-remove-q]")) {
      const focusTarget = card.nextElementSibling || card.previousElementSibling;
      card.remove();
      renderResult(null);
      syncQuestions();
      (focusTarget?.querySelector("[data-toggle-q]") || document.getElementById("decision-add-question")).focus();
      return;
    }
    if (ev.target.closest("[data-remove-row]")) {
      ev.target.closest(".decision-option-row, .decision-level-row")?.remove();
      return;
    }
    if (ev.target.closest("[data-add-option]")) {
      const type = card.querySelector(".decision-q-type").value;
      if (type === "score") addLevelRow(card, "");
      else addOptionRow(card, "", "");
    }
  });
  questionsEl.addEventListener("change", (ev) => {
    if (!ev.target.classList.contains("decision-q-type")) return;
    const card = ev.target.closest(".decision-question");
    const box = card.querySelector(".decision-q-criteria");
    box.innerHTML = "";
    if (ev.target.value === "choice") {
      addOptionRow(card, "HOLD", "wait");
      addOptionRow(card, "ABSTAIN", "no decision");
    } else if (ev.target.value === "score") {
      addLevelRow(card, "low");
      addLevelRow(card, "high");
    }
    syncCriteriaChrome(card);
  });
  questionsEl.addEventListener("input", (ev) => {
    if (ev.target.matches("textarea.decision-autogrow")) fitQuestionText(ev.target.closest(".decision-question"));
    if (!ev.target.classList.contains("decision-q-name")) return;
    const card = ev.target.closest(".decision-question");
    card.querySelector(".decision-question-label").textContent = ev.target.value || "Untitled question";
    syncCriteriaChrome(card);
  });
  document.getElementById("decision-add-question").addEventListener("click", () => addQuestion());
  document.getElementById("decision-model-open").addEventListener("click", () => modelDialog.showModal());
  document.getElementById("decision-model-close").addEventListener("click", () => modelDialog.close());
  suggestPickerBtn.addEventListener("click", () => suggestionPicker.open(suggestionSelection));
  suggestBtn.addEventListener("click", suggestQuestions);
  stateEl.addEventListener("input", syncQuestions);
  document.getElementById("decision-load-sample").addEventListener("click", loadSample);
  document.getElementById("decision-refresh").addEventListener("click", () => loadCatalog(true));
  scoreBtn.addEventListener("click", score);
  showRawEl.addEventListener("change", () => {
    rawEl.hidden = !(showRawEl.checked && lastResult);
  });

  syncQuestions();
  loadCatalog();
})();
