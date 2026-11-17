/* commands.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
// ══════════════════════════════════════════════════════════════════════
// Slash commands — generic, render-only surface over the backend runtime.
// The frontend NEVER branches on the command name; it renders by the
// response's `kind` and applies `effects`/`refresh`. All command logic
// lives in the gateway (sharedagents_core.command_runtime).
// ══════════════════════════════════════════════════════════════════════

const panel = $("#panel");
const panelTitle = $("#panel-title");
const panelBody = $("#panel-body");
const panelClose = $("#panel-close");
const commandsBtn = $("#commands-btn");
const slashMenu = $("#slash-menu");

// Self-improver panel poll handle (cleared when the panel closes).
let siPollTimer = null;

// Rich kinds render in the side panel; text/error render inline in chat.
const PANEL_KINDS = new Set([
  "keyvals",
  "list",
  "table",
  "checklist",
  "detail",
]);

// A slash command that mutates per-session state needs a REAL gateway session
// before it runs. `/goal` registers/clears a goal against a session id; sent
// session-less it is rejected by the backend with "session not found: ''" and
// the goal is never created (the /goal-in-a-fresh-chat bug). `/mode`,
// `/effort`, `/teacher`, `/budget`, `/loop` and `/model <alias>` write the
// same per-session state and fail the same way in a fresh chat (bare `/model`
// only lists models). Pure + top-level so the slash handler can gate
// createSession() on it and tests can pin it.
function commandRequiresSession(text) {
  const t = (text || "").trim();
  return /^\/(goal|mode|effort|teacher|budget|loop)(\s|$)/i.test(t) || /^\/model\s+\S/i.test(t);
}

// True when `/goal <text>` is a SET (new objective), not a control verb.
// The SET still registers via /api/command, then send.js falls through to
// the same /prompt path a typed message uses.
function slashGoalSetPrompt(text) {
  const raw = String(text || "").trim();
  const m = raw.match(/^\/goal(?:\s+([\s\S]*))?$/i);
  if (!m) return "";
  let rest = (m[1] || "").trim();
  if (rest.startsWith("--force ")) rest = rest.slice(8).trim();
  else if (rest === "--force") return "";
  if (!rest) return "";
  const sp = rest.indexOf(" ");
  const head = (sp < 0 ? rest : rest.slice(0, sp)).toLowerCase();
  const tail = sp < 0 ? "" : rest.slice(sp + 1).trim();
  const first = tail.split(/\s+/, 1)[0] || "";
  if (["clear", "stop", "reset", "off", "none", "done", "list", "status", "show", "budget", "cancel"].includes(head)) {
    return "";
  }
  if (["pause", "resume", "restart", "archive", "topup", "met"].includes(head)) {
    if (!tail || first.startsWith("goal_")) return "";
  }
  return rest.replace(/\s*--(?:budget|cost)\s+\$?\d+(?:\.\d+)?/g, " ").replace(/\s+/g, " ").trim();
}

// Format a USD amount for the cost labels (goal widget + session footer).
// Pure; exported only for tests. 2dp normally ("$0.00" for null/NaN/0), but
// sub-cent spend (0 < n < $0.01) keeps 4dp so a real, tiny goal cost doesn't
// collapse to "$0.00" (cents-vs-$5 footer/Goals parity). Anything that still
// rounds to nothing at 4dp shows "<$0.01". goals.js replicates this verbatim
// so the footer and the Goals page render byte-identical cost text.
function formatUsd(n) {
  const v = Number(n) || 0;
  if (v > 0 && v < 0.01) {
    const fine = v.toFixed(4);
    return fine === "0.0000" ? "<$0.01" : "$" + fine;
  }
  return "$" + v.toFixed(2);
}

async function runCommand(text, opts = {}) {
  const silent = !!opts.silent;
  const throwOnError = !!opts.throwOnError;
  const sid = state.currentSessionId;
  const epoch = state.navEpoch;
  const explicitFirstIntent = Object.prototype.hasOwnProperty.call(
    opts, "firstIntent",
  );
  const firstIntent = explicitFirstIntent
    ? (opts.firstIntent || {})
    : firstIntentForSession(sid);
  const session = sid && state.sessions && state.sessions[sid];
  const refreshMaterializedAuthority = !!(
    sid &&
    firstIntent.initial_agent &&
    !(session && session.intent_materialized)
  );
  const refreshFirstIntentAuthority = async () => {
    if (
      !refreshMaterializedAuthority ||
      epoch !== state.navEpoch ||
      sid !== state.currentSessionId
    ) return;
    try {
      await refreshSessionState(sid, epoch);
    } catch (_) {
      // The server may already have persisted authority even when command
      // dispatch failed. A later prompt/state refresh will reconcile it.
    }
  };
  if (!silent) statusText.textContent = `running ${text}…`;
  let resp;
  try {
    resp = await apiPost("/api/command", {
      text,
      session_id: sid,
      agent: firstIntent.initial_agent || state.currentAgent,
      ...firstIntent,
      // On the confirmed re-run, tell the server the user accepted
      // the irreversible-command re-challenge (parity with session delete's
      // ?confirm=1). Absent → the server re-challenges before executing.
      ...(opts.confirmed ? { confirm: true } : {}),
    });
  } catch (e) {
    await refreshFirstIntentAuthority();
    if (!silent)
      appendSystemMessage(`command failed: ${escapeHtml(e.message)}`, true);
    statusText.textContent = "";
    if (throwOnError) throw e;
    return;
  }
  // Materialization is authoritative only after the coordinator projection is
  // read back and applied. Keep refresh best-effort so setup-command behavior
  // does not become a new prompt-blocking policy.
  await refreshFirstIntentAuthority();
  if (!silent) statusText.textContent = "";
  // The server re-challenges an irreversible command (/compact,
  // /restart, /abort, /task del|delete|rm, /monitors kill) that arrives
  // without an explicit confirmation. Surface the SAME confirm card used for
  // session delete (naming the object + what is irreversible), then re-run once
  // with the confirmation set. A direct API caller that skips this still hits
  // the server re-challenge.
  if (resp && resp.error_code === "confirmation_required" && !opts.confirmed) {
    // The footer /compact button drives its OWN in-widget
    // confirm flow (native dialogs are blocked in embedded contexts — the
    // original "compact errors on web" report). With noConfirmDialog the
    // re-challenge envelope is returned untouched so the caller can render
    // that flow; every other caller gets the in-UI confirm card below.
    if (opts.noConfirmDialog) return resp;
    const d = (resp.result && resp.result.data) || {};
    const msg = d.prompt
      ? `${d.prompt}\n\nProceed with ${text}?`
      : `Run ${text}? This cannot be undone.`;
    const ok = await confirmCard({
      title: "Confirm irreversible command",
      message: msg,
      okLabel: "Run",
      danger: true,
    });
    if (!ok) return;
    return runCommand(text, { ...opts, confirmed: true });
  }
  // Visibility (reasoning/tools) is mirrored from the command's state_patch
  // EFFECT in applyStatePatch — never derived from result.text
  renderCommandResponse(resp, text, { silent });
  const result = (resp && resp.result) || {};
  if (!resp || resp.ok === false || result.kind === "error" || resp.error) {
    const detail =
      result.text ||
      (resp && (resp.error || resp.message)) ||
      "command was rejected";
    const error = new Error(detail);
    error.data = resp || {};
    error.code = (resp && (resp.error_code || resp.code)) || "";
    if (throwOnError) throw error;
  }
  return resp;
}

// Commit one staged sheet tuple through the shared acknowledged /model
// transaction.
async function saveModelSheet() {
  ensureModelSheetState();
  if (!["OPEN_DIRTY", "OPEN_ERROR"].includes(state.modelSheet.phase)) return false;
  const draft = cloneModelSelection(state.modelSheet.draft);
  const prior = cloneModelSelection(state.modelSheet.effective);
  const rows = state.modelOptionsByAgent[draft.agent] || [];
  const row = rows.find((model) =>
    model.id === draft.alias &&
    model.provider === draft.provider &&
    model.wireId === draft.wire_id
  ) || rows.find((model) =>
    (model.id === draft.alias || model.alias === draft.alias || model.wireId === draft.wire_id) &&
    (!draft.provider || model.provider === draft.provider)
  ) || rows.find((model) => model.id === draft.alias || model.wireId === draft.wire_id) || rows[0];
  // Some ACP servers (OpenCode, Kimi, and generic ACP agents) advertise
  // models only after session/new. Their sessionless picker catalogue is
  // intentionally empty. An agent-only change remains valid, but must never
  // invent or apply a model id from another engine.
  const agentOnly = !rows.length && !draft.provider &&
    !draft.alias && !draft.wire_id;
  if (!row && !agentOnly) {
    setModelSheetPhase("OPEN_ERROR", "That model is no longer available. Refresh the selection and retry.");
    return false;
  }
  if (row && ((draft.reasoning != null && !row.reasoning) ||
      (draft.speed != null && !(row.speed && row.speed.applyable)))) {
    setModelSheetPhase("OPEN_ERROR", "The selected advanced setting is not supported by this model.");
    return false;
  }

  if (typeof window.sharedagentsModelSheetApply === "function") {
    setModelSheetPhase("APPLYING");
    try {
      const ok = await window.sharedagentsModelSheetApply(draft);
      if (!ok) {
        setModelSheetPhase("OPEN_ERROR", "Could not save agent/model.");
        return false;
      }
      state.modelSheet.effective = cloneModelSelection(draft);
      state.modelSheet.draft = cloneModelSelection(draft);
      state.pickerAgent = draft.agent;
      state.pickerModels[draft.agent] = draft.alias;
      state.modelSheet.phase = "CLOSED";
      state.modelSheet.error = "";
      const n = modelSheetNodes();
      if (n.sheet) n.sheet.setAttribute("hidden", "");
      if (n.backdrop) n.backdrop.setAttribute("hidden", "");
      return true;
    } catch (error) {
      setModelSheetPhase("OPEN_ERROR", (error && error.message) || "Save failed");
      return false;
    }
  }

  setModelSheetPhase("APPLYING");
  let agentChanged = false;
  let modelApplyAttempted = false;
  try {
    const sid = state.currentSessionId;
    if (!sid || isFreshUnboundSession(sid)) {
      applyFreshAgentModeIntent(sid, draft.agent, "");
      const targetModel = draft.wire_id || draft.alias || "";
      state.currentModel = targetModel;
      state.currentAgent = draft.agent;
      const sess = state.sessions && state.sessions[sid];
      if (sess) {
        sess.agent = draft.agent;
        sess.pending_agent = draft.agent;
        sess.model = targetModel;
      }
      if (sid && draft.agent) {
        try {
          await apiPost(
            `/api/sessions/${encodeURIComponent(sid)}/agent`,
            {
              target_agent: draft.agent,
              target_model: targetModel || null,
              target_mode: null,
              frontend: "web",
            },
          );
        } catch (_e) {}
      }
      // No session runs /model yet: report the level shown in the sheet so
      // the server remembers it for the model (turn 1 carries it as
      // initial_effort).
      if (targetModel && row && row.reasoning && draft.reasoning != null) {
        try {
          await apiPost("/api/models/effort", {
            agent: draft.agent, model: targetModel, effort: String(draft.reasoning),
          });
        } catch (_e) { /* the first prompt still carries it */ }
      }
      if (typeof refreshSessionWidgets === "function") refreshSessionWidgets();
    } else {
      if (draft.agent !== state.currentAgent) {
        const switched = await applyAgentSwitch(draft.agent);
        if (!switched || state.currentAgent !== draft.agent)
          throw new Error(state.switchError || `Could not switch to ${draft.agent}`);
        agentChanged = true;
      }
      if (row) {
        modelApplyAttempted = true;
        const command = modelApplyCommand(draft, row, prior.reasoning);
        await runCommand(command, {
          silent: true,
          throwOnError: true,
        });
        if (String(state.currentModel || "") !== draft.wire_id &&
            String(state.currentModel || "") !== draft.alias) {
          throw new Error(
            `Model acknowledgement mismatch: expected ${draft.wire_id}, got ${state.currentModel || "no model"}`,
          );
        }
      }
    }

    // Refresh the server-resolved row after an acknowledged effort change.
    if (draft.provider) await loadModelsForAgent(draft.agent, { provider: draft.provider });
    state.modelSheet.effective = cloneModelSelection(draft);
    state.modelSheet.draft = cloneModelSelection(draft);
    state.pickerAgent = draft.agent;
    state.pickerModels[draft.agent] = draft.alias;
    persistPickerPreferences();
    state.modelSheet.phase = "CLOSED";
    state.modelSheet.error = "";
    const n = modelSheetNodes();
    if (n.sheet) n.sheet.setAttribute("hidden", "");
    if (n.backdrop) n.backdrop.setAttribute("hidden", "");
    renderAgentSelect();
    renderStatusSegments();
    if (typeof updateSendState === "function") updateSendState();
    const trigger = document.getElementById("agent-model-trigger");
    if (trigger) {
      trigger.setAttribute("aria-expanded", "false");
      trigger.focus();
    }
    return true;
  } catch (error) {
    // Agent and model changes are separate acknowledgements. Restore
    // the prior tuple if the second step fails so no partial selection remains.
    if ((agentChanged || state.currentAgent !== prior.agent) && prior.agent) {
      try {
        await switchSessionAgent(prior.agent, "");
      } catch (_rollbackError) {
        // The sheet remains fail-closed and visibly errored; never claim the
        // draft became effective when rollback itself could not be proved.
      }
    }
    if (modelApplyAttempted && prior.alias &&
        state.currentAgent === prior.agent) {
      try {
        const priorRows = state.modelOptionsByAgent[prior.agent] || [];
        const priorRow = priorRows.find((model) => model.id === prior.alias);
        const rollback = modelApplyCommand(prior, priorRow, draft.reasoning);
        await runCommand(rollback, {
          silent: true, throwOnError: true,
        });
      } catch (_rollbackError) {
        // Same fail-closed rule: retain prior effective UI and expose ERROR.
      }
    }
    state.modelSheet.effective = prior;
    state.modelSheet.draft = draft;
    setModelSheetPhase("OPEN_ERROR", switchFailureMessage(error));
    renderStatusSegments();
    return false;
  }
}

async function setAgentDefaultFromSheet() {
  ensureModelSheetState();
  const draft = cloneModelSelection(state.modelSheet.draft);
  if (!draft || !draft.agent) return false;
  const targetAgent = draft.agent;
  const n = modelSheetNodes();
  if (n.useSetDefault) {
    n.useSetDefault.disabled = true;
    n.useSetDefault.textContent = "Setting…";
  }

  try {
    const patch = {};
    if (draft.alias) {
      patch.upstream_model = draft.alias;
    }
    if (draft.reasoning != null) {
      patch.effort = draft.reasoning;
    }
    const res = await apiPatch(`/api/agents/defs/${encodeURIComponent(targetAgent)}`, patch);
    if (!res || !res.ok) {
      const errDetail = (res && res.error) || (res && res.message) || "Failed to set default";
      throw new Error(errDetail);
    }
    // Update local cached models/agents so future opens reflect the new default
    if (state.modelOptionsByAgent && state.modelOptionsByAgent[targetAgent]) {
      const rows = state.modelOptionsByAgent[targetAgent];
      rows.forEach((m) => {
        m.isDefault = typeof rowMatchesModelId === "function"
          ? rowMatchesModelId(m, draft.alias)
          : (m.id === draft.alias);
      });
    }
    // Keep the agent-default cache truthful for the rest of this page life:
    // the saved effort is what every later derivation must resolve to (the
    // server value is re-read on the next catalogue load).
    if (typeof recordAgentDefaultEffort === "function") {
      recordAgentDefaultEffort(targetAgent, draft.reasoning, draft.alias);
    }
    // The agent tile names the saved default: re-read the agent's summary.
    if (typeof loadPickerSummary === "function") {
      loadPickerSummary(targetAgent).then(function () {
        if (typeof renderModelSheet === "function") renderModelSheet();
      }).catch(function () { /* the tile keeps the previous name */ });
    }
    if (draft.alias) {
      state.pickerModels = state.pickerModels || {};
      state.pickerModels[targetAgent] = draft.alias;
      if (typeof persistPickerPreferences === "function") {
        persistPickerPreferences();
      }
    }
    if (n.useSetDefault) {
      n.useSetDefault.textContent = "Default set ✓";
      setTimeout(() => {
        if (n.useSetDefault) {
          n.useSetDefault.textContent = "Set default";
          n.useSetDefault.disabled = false;
        }
      }, 2000);
    }
    if (typeof showToast === "function") {
      showToast(`Set ${draft.alias || "model"} (${draft.reasoning || "default effort"}) as default for ${targetAgent}`);
    }
    return true;
  } catch (err) {
    console.error("setAgentDefaultFromSheet error:", err);
    if (n.useSetDefault) {
      n.useSetDefault.textContent = "Failed ✕";
      setTimeout(() => {
        if (n.useSetDefault) {
          n.useSetDefault.textContent = "Set default";
          n.useSetDefault.disabled = false;
        }
      }, 2000);
    }
    setModelSheetPhase("OPEN_ERROR", `Could not set default: ${err.message || err}`);
    return false;
  }
}

function renderCommandResponse(resp, sourceText, opts = {}) {
  const { silent } = opts;
  const result = resp.result || {};
  const kind = result.kind || "text";

  // Panel rendering — always (panels need refresh).
  if (PANEL_KINDS.has(kind)) {
    state.lastPanelCommand = sourceText || state.lastPanelCommand;
    openPanel(panelTitleFor(result, resp.command), renderKind(kind, result));
  } else if (kind === "error") {
    // Errors always surface for user-typed commands; suppressed for
    // auto-applied internal commands (e.g. pre-turn /model apply).
    if (!silent) {
      appendSystemMessage(result.text || resp.error || "error", true);
    }
  } else if (!silent) {
    // text — render inline chat bubble; also render any root actions.
    // Suppressed when silent (e.g. internal /model, /reasoning, /tools).
    const el = appendSystemMessage(result.text || "(no output)", false);
    if (result.actions && result.actions.length) {
      el.querySelector(".message-content").appendChild(
        renderActions(result.actions),
      );
    }
  }

  // Side effects — always apply (state patches, refresh directives).
  // "Silent" only suppresses chat text, not structural effects.
  applyEffects(resp.effects || []);
  applyRefresh(resp.refresh || []);
  // Chat `/budget <usd>` must move the Agent Monitor Σ chip immediately.
  // The conversation SSE also emits `event: budget` after this command
  // (notify_budget_changed); this local paint covers the sending tab
  // before that frame arrives.
  if (typeof applyLocalBudgetCap === "function" &&
      typeof parseBudgetEditorValue === "function") {
    const parts = String(sourceText || "").trim().split(/\s+/);
    const verb = String(parts[1] || "").toLowerCase();
    if (parts[0] && parts[0].toLowerCase() === "/budget" &&
        parts.length > 1 &&
        verb !== "add" && verb !== "+" && verb !== "topup") {
      const parsed = parseBudgetEditorValue(parts[1]);
      if (parsed.ok) applyLocalBudgetCap(parsed.value);
    }
  }
  if (resp.warnings && resp.warnings.length) {
    // Warnings for internal commands are noise; only surface for user-typed.
    if (!silent)
      appendSystemMessage(
        "⚠ " + resp.warnings.map(escapeHtml).join("; "),
        false,
      );
  }
}

function panelTitleFor(result, command) {
  return (result.data && result.data.title) || command || "result";
}

// ── Generic renderers (one per kind — NO command-specific logic) ──

function renderKind(kind, result) {
  const data = result.data || {};
  switch (kind) {
    case "keyvals":
      return renderKeyvals(data);
    case "list":
      return renderList(data);
    case "table":
      return renderTable(data);
    case "checklist":
      return renderChecklist(data);
    case "detail":
      return renderDetail(data);
    default:
      return renderTextBlock(result.text || "");
  }
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function renderActions(actions) {
  const wrap = el("div", "cmd-actions");
  (actions || []).forEach((a) => {
    const b = el(
      "button",
      `cmd-action cmd-action-${escapeHtml(a.style || "secondary")}`,
      escapeHtml(a.label),
    );
    b.dataset.cmd = a.command;
    if (a.confirm) b.dataset.confirm = "1";
    if (a.disabled) b.disabled = true;
    wrap.appendChild(b);
  });
  return wrap;
}

function renderTextBlock(text) {
  return el("div", "panel-text", renderMarkdown(text));
}

function renderKeyvals(data) {
  const tbl = el("table", "kv-table");
  (data.pairs || []).forEach(([k, v]) => {
    const tr = el("tr");
    tr.appendChild(el("td", "kv-key", escapeHtml(String(k))));
    tr.appendChild(el("td", "kv-val", escapeHtml(String(v))));
    tbl.appendChild(tr);
  });
  return tbl;
}

function renderList(data) {
  const wrap = el("div", "rlist");
  (data.items || []).forEach((it) => {
    const row = el("div", "rlist-item");
    const head = el("div", "rlist-head");
    head.appendChild(
      el("span", "rlist-title", escapeHtml(it.title || it.id || "")),
    );
    if (it.subtitle)
      head.appendChild(el("span", "rlist-sub", escapeHtml(it.subtitle)));
    row.appendChild(head);
    if (it.body) row.appendChild(el("div", "rlist-body", escapeHtml(it.body)));
    if (it.badges && it.badges.length) {
      const bd = el("div", "rlist-badges");
      it.badges
        .filter(Boolean)
        .forEach((b) =>
          bd.appendChild(el("span", "badge", escapeHtml(String(b)))),
        );
      row.appendChild(bd);
    }
    if (it.actions && it.actions.length)
      row.appendChild(renderActions(it.actions));
    wrap.appendChild(row);
  });
  if (!(data.items || []).length)
    wrap.appendChild(el("div", "panel-empty", "empty"));
  return wrap;
}

function renderTable(data) {
  const cols = data.columns || [];
  const tbl = el("table", "rtable");
  const thead = el("thead");
  const htr = el("tr");
  cols.forEach((c) => htr.appendChild(el("th", null, escapeHtml(String(c)))));
  thead.appendChild(htr);
  tbl.appendChild(thead);
  const tbody = el("tbody");
  (data.rows || []).forEach((r) => {
    const tr = el("tr");
    cols.forEach((c) =>
      tr.appendChild(
        el("td", null, escapeHtml(String(r[c] != null ? r[c] : ""))),
      ),
    );
    tbody.appendChild(tr);
  });
  tbl.appendChild(tbody);
  return tbl;
}

function renderChecklist(data) {
  const wrap = el("div", "checklist");
  const glyph = { completed: "✓", in_progress: "◐", pending: "○" };
  (data.items || []).forEach((it) => {
    const row = el(
      "div",
      `check-item check-${escapeHtml(it.status || "pending")}`,
    );
    row.appendChild(el("span", "check-glyph", glyph[it.status] || "○"));
    row.appendChild(el("span", "check-text", escapeHtml(it.content || "")));
    wrap.appendChild(row);
  });
  if (!(data.items || []).length)
    wrap.appendChild(el("div", "panel-empty", "no todos"));
  return wrap;
}

function renderDetail(data) {
  const wrap = el("div", "detail");
  if (data.subtitle)
    wrap.appendChild(el("div", "detail-sub", escapeHtml(data.subtitle)));
  if (data.fields && data.fields.length) {
    const tbl = el("table", "kv-table");
    data.fields.forEach(([k, v]) => {
      const tr = el("tr");
      tr.appendChild(el("td", "kv-key", escapeHtml(String(k))));
      tr.appendChild(el("td", "kv-val", escapeHtml(String(v))));
      tbl.appendChild(tr);
    });
    wrap.appendChild(tbl);
  }
  if (data.body)
    wrap.appendChild(el("div", "panel-text", renderMarkdown(data.body)));
  return wrap;
}

// ── Effects / refresh ──

// Submit a command's follow-up prompt as a normal turn. A typed slash command
// runs inside sendMessage(), which holds the first-click lock until the command
// returns — sending now would be dropped and leave the text in the composer.
function submitEffectPrompt(text) {
  const submit = () => {
    messageInput.value = text;
    sendMessage();
  };
  if (state.promptSubmitting) setTimeout(submit, 0);
  else submit();
}

function applyEffects(effects) {
  effects.forEach((eff) => {
    const p = eff.payload || {};
    switch (eff.type) {
      case "state_patch":
        applyStatePatch(p.patch || {});
        break;
      case "prompt":
        if (p.text) {
          // A delegate bridge (`/claude <prompt>`) carries a target agent —
          // switch to it before submitting the prompt as a normal turn.
          if (
            p.agent &&
            state.agents.includes(p.agent) &&
            p.agent !== state.currentAgent
          ) {
            applyAgentSwitch(p.agent).then((ok) => {
              if (!ok) return;
              submitEffectPrompt(p.text);
            });
            break;
          }
          submitEffectPrompt(p.text);
        }
        break;
      case "new_session":
        createSession(undefined, { explicit: true });
        break;
      case "switch_agent":
        if (p.revision || p.effective_mode || p.effective_mode_id)
          applySessionState(p);
        else if (state.currentSessionId)
          refreshSessionState(state.currentSessionId, state.navEpoch)
            .catch((error) =>
              setSwitchPending(false, switchFailureMessage(error), true),
            );
        else if (p.agent && state.agents.includes(p.agent))
          applyAgentSwitch(p.agent);
        break;
      case "switch_session":
        // Session switching on web is managed via the sidebar (web sessions);
        // the gateway session id here is informational.
        statusText.textContent = `gateway session ${(p.session_id || "").slice(0, 12)}`;
        break;
      case "cancel":
        cancelStreaming();
        break;
    }
  });
}
// Only these keys are ever applied from a state_patch — defends against
// state injection even though the backend already scopes them.
//
// The goal keys (active_goal + the goal_* tracker) are whitelisted for parity,
// but their PRIMARY live channel is the engine's turn metrics (TurnDone /
// RoundComplete → applyGoalProgress at "turn_ended"), not a state_patch effect.
// The backend stamps the full tracker — goal_terminal / goal_turn_count /
// max_goal_turns (engine local_llm._goal_progress_metrics) alongside
// active_goal — onto every turn's metrics, and /status re-derives the same
// snapshot via command_runtime._goal_state. The whitelist guards the
// state_patch channel should a producer ever route them there too.
const STATE_PATCH_KEYS = new Set([
  "mode",
  "reasoning_visibility",
  "thinking_enabled",
  "tool_output_visible",
  "tool_output",
  "model",
  "endpoint",
  "context_window",
  "effort",
  "effort_selected",
  "speed",
  "teacher_enabled",
  "teacher_model",
  "teacher_active",
  "active_goal",
  "goal_id",
  "goal_terminal",
  "goal_turn_count",
  "max_goal_turns",
  "session_cost_usd",
  "goal_cost_usd",
  "goal_max_cost_usd",
  "goal_wait",
]);

function applyStatePatch(patch) {
  const safe = {};
  Object.keys(patch || {}).forEach((k) => {
    if (STATE_PATCH_KEYS.has(k)) safe[k] = patch[k];
  });
  if (safe.model) {
    const agent = state.currentAgent ||
      (typeof selectedPickerAgent === "function" ? selectedPickerAgent() : "");
    state.currentModel = (typeof isForeignAgentModel === "function" &&
      isForeignAgentModel(agent, safe.model)) ? "" : safe.model;
    // A typed `/model <alias>` must also move the composer's selection;
    // otherwise the next first-turn send re-applies the stale picker model
    // (with the new model's effort) and the prompt is refused.
    if (typeof syncModelSheetEffectiveFromSession === "function")
      syncModelSheetEffectiveFromSession();
  }
  // A model/endpoint switch can change the context window — keep the footer's
  // "max" segment in sync (it's whitelisted above but was otherwise ignored).
  if (typeof safe.context_window === "number" && safe.context_window > 0) {
    state.lastContextMax = safe.context_window;
  }
  if (safe.mode) {
    state.currentMode = safe.mode;
  }
  if ("effort_selected" in safe) state.statusInfo.effort_selected = safe.effort_selected;
  if ("effort" in safe) {
    state.statusInfo.effort = safe.effort || "default";
    if (typeof syncModelSheetEffortFromStatus === "function")
      syncModelSheetEffortFromStatus();
  }
  if ("speed" in safe) {
    state.statusInfo.speed = safe.speed || "default";
  }
  // Teacher supervision: ``teacher_active`` is the effective gate the
  // indicator shows/hides on; ``teacher_enabled`` mirrors it (kept distinct
  // for the explicit footer widget); ``teacher_model`` is the supervising model.
  if ("teacher_active" in safe) {
    state.teacherActive = !!safe.teacher_active;
  }
  if ("teacher_enabled" in safe) {
    state.teacherEnabled = !!safe.teacher_enabled;
  }
  if ("teacher_model" in safe) {
    state.teacherModel = safe.teacher_model || "";
  }
  if ("active_goal" in safe) {
    const nextGoal = safe.active_goal || "";
    // A goal that just got set OR cleared starts a fresh cycle: drop the old
    // terminal badge + turn count so a new goal never inherits a stale
    // "paused"/"met" state. If the SAME patch also carries fresh
    // progress numbers (applied just below) they win. EXCEPTION: an empty
    // active_goal that arrives alongside a terminal state (a met/blocked
    // settle) must NOT wipe that terminal — the badge has to outlive the goal
    // text; applyGoalProgress below restores it.
    if (nextGoal !== state.activeGoal) {
      state.goalTerminal = "";
      state.goalTurnCount = 0;
    }
    state.activeGoal = nextGoal;
  }
  applyGoalProgress(safe);
  // Reasoning / tools visibility — keep BOTH the live display state and the
  // cached statusInfo in sync so a /reasoning or /tools change (from the Options
  // card OR a typed slash command) reflects in the status bar + the controls.
  if ("reasoning_visibility" in safe) {
    state.vis.reasoning = safe.reasoning_visibility;
    if (state.statusInfo)
      state.statusInfo.reasoning = safe.reasoning_visibility;
  }
  if ("tool_output" in safe) {
    // Authoritative three-state level (off|on|stream) from the state_patch.
    state.vis.tools = safe.tool_output;
    if (state.statusInfo) state.statusInfo.tool = safe.tool_output;
  } else if ("tool_output_visible" in safe) {
    const v = safe.tool_output_visible ? "on" : "off";
    state.vis.tools = v;
    if (state.statusInfo) state.statusInfo.tool = v;
  }
  if (safe.mode && state.statusInfo) {
    state.statusInfo.mode = safe.mode;
  }
  updateHeader();
  updateSelects();
  // The patch is reflected in the header + status bar + teacher indicator; do
  // NOT dump a raw "key=value …" string into the footer status-text (that left a
  // duplicate "teacher_active=false …" line under the input).
  renderSessionStatusBar();
  renderGoalWidget();
}

function applyRefresh(refresh) {
  if (!refresh.length) return;
  if ((refresh.includes("session") || refresh.includes("models")) &&
      typeof _menuSectionOpen === "function" &&
      _menuSectionOpen("sidebar-conversations")) {
    loadSessionList();
    loadTopicList();
  }
  // A "models" refresh must actually reload the model picker (it used to only
  // refresh the session list), so the active-model marker stays in sync.
  if (refresh.includes("models")) loadModelsForCurrentAgent();
  // Re-run the last panel command so a mutating action (kanban complete,
  // plugin enable, …) updates the open panel without the frontend knowing
  // what changed.
  if (state.lastPanelCommand && !panel.hidden) {
    runCommand(state.lastPanelCommand, { silent: true });
  }
}

// ── Panel ──

function openPanel(title, contentEl) {
  panelTitle.textContent = title;
  panelBody.innerHTML = "";
  panelBody.appendChild(contentEl);
  panel.hidden = false;
  document.getElementById("app").classList.add("panel-open");
}

function closePanel() {
  panel.hidden = true;
  document.getElementById("app").classList.remove("panel-open");
  if (siPollTimer) {
    clearInterval(siPollTimer);
    siPollTimer = null;
  }
}

// ── Self-improver dashboard panel ──
//
// Read-only view over GET /api/self-improver (the SI runs as CLI subprocesses,
// not sessions, so it never appears in the conversation list). Polls every 30s
// while the panel is open; the server caches for 60s so this is cheap.

function siIssueUrl(data, n) {
  return n && data.gitea ? `${data.gitea.issues_base}/${n}` : "";
}

function siLink(url, text) {
  const a = el("a", "si-link", escapeHtml(text));
  if (url) {
    a.href = url;
    a.target = "_blank";
    a.rel = "noopener";
  }
  return a;
}

function siSection(root, title, count) {
  const h = el("div", "si-section");
  h.appendChild(el("span", "si-section-title", escapeHtml(title)));
  if (count !== undefined) h.appendChild(el("span", "si-count", String(count)));
  root.appendChild(h);
  return h;
}

function siEmpty(root, label) {
  root.appendChild(el("div", "si-empty", escapeHtml(label || "none")));
}

function siIssueRows(root, data, list) {
  if (!list || !list.length) {
    siEmpty(root);
    return;
  }
  list.forEach((i) => {
    const row = el("div", "si-row");
    row.appendChild(
      siLink(i.url || siIssueUrl(data, i.number), `#${i.number}`),
    );
    row.appendChild(
      el("span", "si-sum", escapeHtml(i.title || i.summary || "")),
    );
    (i.labels || [])
      .filter((l) => l !== "self-improver")
      .forEach((l) => row.appendChild(el("span", "si-label", escapeHtml(l))));
    root.appendChild(row);
  });
}
function renderSelfImproverInto(root, d) {
  if (!d || d.available === false) {
    siEmpty(
      root,
      "Self-improver data unavailable" + (d && d.error ? `: ${d.error}` : ""),
    );
    return;
  }
  const s = d.status || {};

  // (a) status header
  const head = el("div", "si-head");
  head.appendChild(
    el(
      "span",
      `si-pill ${s.enabled ? "on" : "off"}`,
      s.enabled ? "enabled" : "disabled",
    ),
  );
  const meta = [
    `mode ${s.mode || "—"}`,
    s.model_alias || "—",
    `≤${s.max_rounds || 0} rounds`,
  ];
  if (s.auto_merge) meta.push("auto-merge");
  if (s.auto_restart) meta.push("auto-restart");
  head.appendChild(el("span", "si-meta", escapeHtml(meta.join(" · "))));
  root.appendChild(head);

  const sw = s.last_sweep;
  root.appendChild(
    el(
      "div",
      "si-substat",
      escapeHtml(
        `${s.analyzed_sessions || 0} analyzed · ${s.open_issues || 0} open issues · ` +
          `${s.ledger_patterns || 0} patterns` +
          (sw
            ? ` · last sweep ${sw.targets} targets / ${sw.incidents} failures`
            : ""),
      ),
    ),
  );

  // (b) currently fixing
  siSection(root, "Currently fixing");
  const c = d.currently_fixing;
  if (c) {
    const row = el("div", "si-row si-active");
    row.appendChild(
      siLink(siIssueUrl(d, c.issue_number), `#${c.issue_number || "?"}`),
    );
    row.appendChild(el("span", "si-sum", escapeHtml(c.summary || "")));
    if (c.age_seconds != null)
      row.appendChild(
        el("span", "si-age", `${Math.floor(c.age_seconds / 60)}m`),
      );
    root.appendChild(row);
  } else {
    siEmpty(root, "idle — no fix in flight");
  }

  // (c) backlog
  const b = d.backlog || {};
  siSection(root, "Needs human", (b.needs_human || []).length);
  siIssueRows(root, d, b.needs_human);
  siSection(root, "Open SI issues", (b.open_issues || []).length);
  siIssueRows(root, d, b.open_issues);
  siSection(root, "No agreement", (b.no_agreement || []).length);
  if ((b.no_agreement || []).length) {
    b.no_agreement.forEach((r) => {
      const row = el("div", "si-row");
      row.appendChild(
        siLink(
          r.url || siIssueUrl(d, r.issue_number),
          `#${r.issue_number || "?"}`,
        ),
      );
      row.appendChild(el("span", "si-sum", escapeHtml(r.summary || "")));
      row.appendChild(el("span", "si-age", `${r.rounds}r`));
      root.appendChild(row);
    });
  } else {
    siEmpty(root);
  }

  // (d) recent fixes
  siSection(root, "Recent fixes", (d.recent_fixes || []).length);
  if ((d.recent_fixes || []).length) {
    d.recent_fixes.forEach((r) => {
      const row = el("div", "si-row");
      row.appendChild(
        siLink(
          r.issue_url || siIssueUrl(d, r.issue_number),
          `#${r.issue_number || "?"}`,
        ),
      );
      row.appendChild(el("span", "si-sum", escapeHtml(r.summary || "")));
      if (r.pr_number) row.appendChild(siLink(r.pr_url, `PR #${r.pr_number}`));
      root.appendChild(row);
    });
  } else {
    siEmpty(root);
  }

  // (e) orphaned
  siSection(
    root,
    "Orphaned (fix merged, issue open)",
    (d.orphaned || []).length,
  );
  if ((d.orphaned || []).length) {
    d.orphaned.forEach((r) => {
      const row = el("div", "si-row");
      row.appendChild(
        siLink(
          r.issue_url || siIssueUrl(d, r.issue_number),
          `#${r.issue_number || "?"}`,
        ),
      );
      row.appendChild(el("span", "si-sum", escapeHtml(r.summary || "")));
      root.appendChild(row);
    });
  } else {
    siEmpty(root);
  }

  // (f) patterns
  siSection(root, "Patterns", (d.patterns || []).length);
  if ((d.patterns || []).length) {
    d.patterns.forEach((p) => {
      const row = el("div", "si-row");
      row.appendChild(el("span", "si-label", escapeHtml(p.status || "?")));
      row.appendChild(el("span", "si-age", `${p.count}×`));
      row.appendChild(el("span", "si-sum", escapeHtml(p.symptom || "")));
      root.appendChild(row);
    });
  } else {
    siEmpty(root);
  }

  if ((d.errors || []).length)
    root.appendChild(el("div", "si-errs", escapeHtml(d.errors.join("; "))));
}

async function refreshSelfImprover() {
  let data;
  try {
    data = await apiGet("/api/self-improver");
  } catch (e) {
    return;
  } // transient: keep the last good render
  const body = el("div", "si-dash");
  renderSelfImproverInto(body, data);
  panelTitle.textContent = "Self-Improver";
  panelBody.innerHTML = "";
  panelBody.appendChild(body);
}

async function openSelfImproverPanel() {
  const loading = el("div", "si-dash");
  siEmpty(loading, "loading…");
  openPanel("Self-Improver", loading);
  await refreshSelfImprover();
  if (siPollTimer) clearInterval(siPollTimer);
  siPollTimer = setInterval(refreshSelfImprover, 30000);
}

// ── Inline system messages (for text/error command output) ──

function appendSystemMessage(text, isError) {
  setWelcomeVisible(false);
  const div = _createMessageElement("system", text);
  if (isError) div.classList.add("message-error");
  const idx = VirtualScroller.add("system", text, div);
  VirtualScroller.attach(div, idx);
  scrollToBottom();
  return div;
}

// ── Command palette + slash autocomplete ──

async function loadCommands() {
  if (
    state.commands && state.commands.length
    && state._commandsAgent === state.currentAgent
  ) return state.commands;
  if (state._commandsPromise && state._commandsAgent === state.currentAgent) {
    return state._commandsPromise;
  }
  const epoch = state.navEpoch; // bail if an agent/session switch supersedes us
  const p = (async () => {
    try {
      const data = await apiGet(
        `/api/commands?agent=${encodeURIComponent(state.currentAgent)}`,
      );
      if (epoch !== state.navEpoch) return state.commands || []; // stale agent palette
      state.commands = data.commands || [];
      state._commandsAgent = state.currentAgent;
    } catch (e) {
      if (epoch === state.navEpoch) state.commands = [];
    } finally {
      state._commandsPromise = null;
    }
    return state.commands || [];
  })();
  state._commandsPromise = p;
  state._commandsAgent = state.currentAgent;
  return p;
}

async function openCommandPalette() {
  const cmds = await loadCommands();
  const wrap = el("div", "rlist");
  cmds.forEach((name) => {
    const row = el("div", "rlist-item");
    const head = el("div", "rlist-head");
    head.appendChild(el("span", "rlist-title", `/${escapeHtml(name)}`));
    row.appendChild(head);
    row.appendChild(
      renderActions([{ id: name, label: "run", command: `/${name}` }]),
    );
    wrap.appendChild(row);
  });
  openPanel("commands", wrap);
}

function showSlashMenu(prefix) {
  const cmds = (state.commands || []).filter((n) => n.startsWith(prefix));
  if (!cmds.length) {
    hideSlashMenu();
    return;
  }
  slashMenu.innerHTML = cmds
    .slice(0, 12)
    .map(
      (n) =>
        `<div class="slash-item" data-name="${escapeHtml(n)}">/${escapeHtml(n)}</div>`,
    )
    .join("");
  slashMenu.hidden = false;
}

function hideSlashMenu() {
  slashMenu.hidden = true;
}

// ── Wire up command UI ──

function setupCommandUI() {
  commandsBtn.addEventListener("click", openCommandPalette);
  panelClose.addEventListener("click", closePanel);

  // Delegated action-button clicks (panel + inline messages).
  document.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-cmd]");
    if (!btn) return;
    // A danger action (data-confirm) confirms client-side here; pass the
    // confirmation through so the server re-challenge isn't a second
    // dialog for the same click. Non-danger actions run one-shot.
    if (btn.dataset.confirm) {
      const ok = await confirmCard({
        title: "Confirm command",
        message: `Run ${btn.dataset.cmd}? This cannot be undone.`,
        okLabel: "Run",
        danger: true,
      });
      if (!ok) return;
      runCommand(btn.dataset.cmd, { confirmed: true });
      return;
    }
    runCommand(btn.dataset.cmd);
  });

  // Slash autocomplete in the input.
  messageInput.addEventListener("input", () => {
    const v = messageInput.value;
    if (v.startsWith("/") && !v.includes(" ") && !v.includes("\n")) {
      const prefix = v.slice(1).toLowerCase();
      if (state.commands && state.commands.length) {
        showSlashMenu(prefix);
        return;
      }
      loadCommands().then(() => {
        if (messageInput.value === v) showSlashMenu(prefix);
      });
    } else {
      hideSlashMenu();
    }
  });
  slashMenu.addEventListener("click", (e) => {
    const item = e.target.closest(".slash-item");
    if (!item) return;
    messageInput.value = `/${item.dataset.name} `;
    hideSlashMenu();
    messageInput.focus();
    updateSendState();
  });
}

// ══════════════════════════════════════════════════════════════════════
// Footer /compact button + progress widget.
//
// The footer compact button lives in #status-segments (re-rendered on every
// status update); its click is delegated from that container in render.js.
// This controller owns the widget state machine:
//     idle → confirm → compacting → done | error
// The button path must NEVER use a native dialog — blocked dialogs in embedded
// contexts are the reported failure this widget fixes — so the
// re-challenge is answered inside the widget's own UI. The slash-command
// /compact path answers the same re-challenge with the shared in-UI confirm
// card (confirm-card.js), never a native dialog.
//
// Completion is signalled by the compaction turn's turn_ended (streaming.js
// calls compactContextTurnEnded); the footer context meter refreshes
// automatically via syncContextMeterFromHeader() on that same event. If no
// turn_ended ever arrives the widget keeps waiting and shows elapsed time
// (with a "still working" note past COMPACT_WIDGET_TIMEOUT_MS).
// ══════════════════════════════════════════════════════════════════════

const COMPACT_WIDGET_TIMEOUT_MS = 120000;

// A fresh idle widget record. Pure — no DOM, unit-tested.
function compactWidgetIdle() {
  return {
    state: "idle", // idle | confirm | compacting | done | error
    prompt: "", // confirm-prompt text (server-provided)
    error: "", // sticky error text
    startAt: 0, // compacting start (Date.now)
    timeoutNoted: false, // "still working" hint past the timeout
    stats: null, // last compaction stats (header compaction_last_stats)
    summary: "", // last compaction summary (header compaction_summary)
  };
}

// Pure widget-state transition core — no DOM, unit-tested.
function compactWidgetReduce(w, action) {
  switch (action.type) {
    case "BEGIN":
      return { state: "confirm", prompt: action.prompt || "", error: "", startAt: 0, timeoutNoted: false };
    case "CONFIRMED":
      return { state: "compacting", prompt: "", error: "", startAt: action.startAt || Date.now(), timeoutNoted: false };
    case "SIGNAL":
      if (w.state !== "compacting") return w;
      return { ...w, timeoutNoted: true };
    case "DONE":
      if (w.state !== "compacting") return w;
      // Carry the just-completed compaction's stats/summary (from the header
      // via compactContextTurnEnded) so the done state can show what happened.
      return {
        ...w, state: "done", error: "",
        stats: action.stats || w.stats || null,
        summary: action.summary || w.summary || "",
      };
    case "ERROR":
      if (w.state === "idle") return w;
      return { ...w, state: "error", error: action.text || "Compaction failed" };
    case "DISMISS":
      return compactWidgetIdle();
    default:
      return w;
  }
}

// Format the compaction elapsed time ("12s", "1m 05s"). Pure.
function fmtCompactElapsed(ms) {
  const s = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
  if (s < 60) return s + "s";
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r ? m + "m " + r + "s" : m + "m";
}

// Map a failed /compact attempt to the exact user-facing message. Pure.
function compactFailureText(resp, err) {
  if (err) {
    if (err.status === 403) return "Session not found or not adoptable";
    if (err.status === 502) return "Gateway unreachable";
    return err.message || "Command failed";
  }
  if (!resp) return "Command failed — no response from the server";
  if (resp.error_code === "turn_active") {
    return "A turn is running — finish it or /abort it first";
  }
  const detail = (resp.result && resp.result.text) || resp.error || resp.message;
  return detail || "Command rejected";
}

// ── Widget store + timers ──

let _compactWidget = compactWidgetIdle();
let _compactTicker = null;
let _compactDismissTimer = null;

// Read access for render.js (button disabled state) and streaming.js (turn end).
function compactWidgetState() {
  return _compactWidget;
}

function restoreCompactWidgetFromSession(session) {
  const st = String(
    (session && (session.state || session.session_state)) || "",
  );
  if (st !== "compacting") return;
  if (_compactWidget.state === "compacting") return;
  _compactWidgetSet(compactWidgetReduce(compactWidgetIdle(), { type: "CONFIRMED" }));
  _compactWidgetStartTicker();
}

function _compactWidgetSet(w) {
  _compactWidget = w;
  renderCompactWidget();
  renderStatusSegments(); // keep the footer button's disabled state in sync
  // The Context sheet's Compact button drives this same state machine.
  if (window.SaContextSheet) window.SaContextSheet.compactChanged(w);
}

function _compactWidgetClearTimers() {
  if (_compactTicker) { clearInterval(_compactTicker); _compactTicker = null; }
  if (_compactDismissTimer) { clearTimeout(_compactDismissTimer); _compactDismissTimer = null; }
}

function _compactWidgetStartTicker() {
  _compactWidgetClearTimers();
  _compactTicker = setInterval(function () {
    const w = _compactWidget;
    if (w.state !== "compacting") { _compactWidgetClearTimers(); return; }
    // Timeout fallback: 120s without a turn_ended — keep waiting and surface
    // the elapsed time; a long compaction must not look like an error.
    if (Date.now() - w.startAt >= COMPACT_WIDGET_TIMEOUT_MS && !w.timeoutNoted) {
      _compactWidgetSet(compactWidgetReduce(w, { type: "SIGNAL" }));
    } else {
      renderCompactWidget();
    }
  }, 1000);
}

// ── Flow: button → confirm → run → track ──

// Delegated from #status-segments (render.js). Guarded: only starts when idle.
async function compactContextStart() {
  const w = _compactWidget;
  if (w.state === "confirm" || w.state === "compacting") return;
  if (!state.currentSessionId) {
    _compactWidgetSet(compactWidgetReduce(w, {
      type: "ERROR", text: "No session is bound — open or start a session first.",
    }));
    return;
  }
  let resp = null;
  let err = null;
  try {
    // noConfirmDialog: the re-challenge envelope comes back to the
    // widget instead of popping a confirm card (blocked native dialogs are
    // the failure this button fixes). Confirmation happens in the widget's UI.
    resp = await runCommand("/compact", {
      silent: true, throwOnError: true, noConfirmDialog: true,
    });
  } catch (e) {
    err = e;
  }
  if (err) {
    _compactWidgetSet(compactWidgetReduce(w, { type: "ERROR", text: compactFailureText(null, err) }));
    return;
  }
  if (resp && resp.error_code === "confirmation_required") {
    const d = (resp.result && resp.result.data) || {};
    _compactWidgetSet(compactWidgetReduce(w, {
      type: "BEGIN",
      prompt: d.prompt || "Compaction cannot be undone. Continue?",
    }));
    return;
  }
  if (!resp || resp.ok === false) {
    _compactWidgetSet(compactWidgetReduce(w, { type: "ERROR", text: compactFailureText(resp, null) }));
    return;
  }
  _compactWidgetSet(compactWidgetReduce(w, { type: "CONFIRMED" }));
  _compactWidgetStartTicker();
}

// OK on the in-widget confirm prompt — re-run with the confirmation set.
async function compactContextConfirm() {
  const w = _compactWidget;
  if (w.state !== "confirm") return;
  let resp = null;
  let err = null;
  try {
    resp = await runCommand("/compact", {
      silent: true, throwOnError: true, noConfirmDialog: true, confirmed: true,
    });
  } catch (e) {
    err = e;
  }
  if (err) {
    _compactWidgetSet(compactWidgetReduce(w, { type: "ERROR", text: compactFailureText(null, err) }));
    return;
  }
  if (!resp || resp.ok === false) {
    _compactWidgetSet(compactWidgetReduce(w, { type: "ERROR", text: compactFailureText(resp, null) }));
    return;
  }
  _compactWidgetSet(compactWidgetReduce(w, { type: "CONFIRMED" }));
  _compactWidgetStartTicker();
}

function compactContextCancel() {
  if (_compactWidget.state !== "confirm") return;
  _compactWidgetClearTimers();
  _compactWidgetSet(compactWidgetReduce(_compactWidget, { type: "DISMISS" }));
}

// Dismiss the sticky error state (its own button).
function compactContextDismiss() {
  if (_compactWidget.state !== "error") return;
  _compactWidgetClearTimers();
  _compactWidgetSet(compactWidgetReduce(_compactWidget, { type: "DISMISS" }));
}

// turn_ended hook (streaming.js): the compaction turn is over.
function compactContextTurnEnded(ev) {
  const w = _compactWidget;
  if (w.state !== "compacting") return;
  _compactWidgetClearTimers();
  if (ev && (ev.error || ev.aborted)) {
    _compactWidgetSet(compactWidgetReduce(w, {
      type: "ERROR",
      text: ev.error || (ev.aborted ? "Compaction was aborted." : "Compaction failed."),
    }));
    return;
  }
  _compactWidgetSet(compactWidgetReduce(w, {
    type: "DONE",
    // The engine persisted these to the session header during the tool turn;
    // a header_update event lands before turn_ended, so state carries the
    // fresh values here. Absent (never-arrived header) → the done state just
    // omits the summary line; the chip renders from state independently.
    stats: (state.compactionLastStats &&
            typeof state.compactionLastStats === "object" &&
            !Array.isArray(state.compactionLastStats))
      ? state.compactionLastStats : null,
    summary: typeof state.compactionSummary === "string"
      ? state.compactionSummary : "",
  }));
  // Auto-dismiss so the confirmation is readable, then return to idle.
  _compactDismissTimer = setTimeout(function () {
    if (_compactWidget.state === "done") {
      _compactWidgetSet(compactWidgetReduce(_compactWidget, { type: "DISMISS" }));
    }
  }, 3000);
}

// ── Widget DOM ──

function renderCompactWidget() {
  const node = document.getElementById("compact-status-widget");
  if (!node) return;
  // Delegated clicks for the widget's own buttons — bound once on the static
  // #session-status container (the widget node itself is stable, but its
  // innerHTML is replaced per state, so the handlers live on the parent).
  const host = document.getElementById("session-status");
  if (host && !host.dataset.cwBound) {
    host.dataset.cwBound = "1";
    host.addEventListener("click", function (e) {
      const t = e.target;
      if (!t || !t.closest) return;
      if (t.closest("#compact-confirm-ok")) { e.stopPropagation(); compactContextConfirm(); return; }
      if (t.closest("#compact-confirm-cancel")) { e.stopPropagation(); compactContextCancel(); return; }
      if (t.closest("#compact-error-dismiss")) { e.stopPropagation(); compactContextDismiss(); return; }
    });
  }
  const w = _compactWidget;
  node.className = "status-widget compact-widget compact-" + w.state;
  if (w.state === "idle") {
    node.hidden = true;
    node.innerHTML = "";
    return;
  }
  node.hidden = false;
  if (w.state === "confirm") {
    node.innerHTML =
      '<span class="cw-icon">⚠️</span>' +
      '<span class="cw-text">' + escapeHtml(w.prompt) + "</span>" +
      '<button type="button" class="cw-btn cw-ok" id="compact-confirm-ok">OK</button>' +
      '<button type="button" class="cw-btn" id="compact-confirm-cancel">Cancel</button>';
    return;
  }
  if (w.state === "compacting") {
    const elapsed = fmtCompactElapsed(Date.now() - w.startAt);
    node.innerHTML =
      '<span class="cw-spinner" aria-hidden="true"></span>' +
      '<span class="cw-text">Compacting context… ' + escapeHtml(elapsed) +
      (w.timeoutNoted ? " — still working" : "") + "</span>";
    return;
  }
  if (w.state === "done") {
    node.innerHTML =
      '<span class="cw-icon">✅</span><span class="cw-text">Compaction complete</span>' +
      (w.summary ? '<span class="cw-summary">' + escapeHtml(w.summary) + "</span>" : "");
    return;
  }
  node.innerHTML =
    '<span class="cw-icon">❌</span>' +
    '<span class="cw-text">' + escapeHtml(w.error) + "</span>" +
    '<button type="button" class="cw-btn" id="compact-error-dismiss">Dismiss</button>';
}
