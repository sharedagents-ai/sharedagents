/* widgets.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
function applyGoalProgress(src, opts) {
  if (!src || typeof src !== "object") return;
  // Distinguish ABSENT from EXPLICIT-EMPTY: an absent ``goal_id`` leaves the
  // current id alone (a partial metrics/header patch), while an explicit
  // empty string CLEARS it — that is what lets a ``/goal clear`` state_patch
  // detach the widget (and stops refreshGoalCost from re-polling a stale,
  // now-archived goal id).
  //
  // ``allowClear: false`` (the live turn-metrics channel, see
  // applyTurnGoalMetrics below) refuses to CLEAR a known goal id / terminal on
  // an explicit empty: the engine stamps ``active_goal`` from its in-memory
  // projection, which can be momentarily empty for a goal that is still
  // running in the registry. The authoritative channels (state_patch,
  // header_update via applyHeaderGoalPatch, and /status's goal_id /
  // goal_terminal writes) keep the default and still detach cleanly;
  // active_goal TEXT on /status and header_update is identity-aware
  // (applyIncomingGoalText).
  const allowClear = !(opts && opts.allowClear === false);
  if ("goal_id" in src) {
    const gid = src.goal_id || "";
    if (gid || allowClear) {
      state.goalId = gid;
      state.goalSid = gid ? state.currentSessionId : "";
    }
  }
  if ("goal_terminal" in src) {
    const term = src.goal_terminal || "";
    if (term || allowClear) state.goalTerminal = term;
  }
  if ("goal_wait" in src) state.goalWait = String(src.goal_wait || "");
  if ("goal_turn_count" in src)
    state.goalTurnCount = Number(src.goal_turn_count) || 0;
  if ("max_goal_turns" in src)
    state.maxGoalTurns = Number(src.max_goal_turns) || 0;
  // Realtime cost: spend (session + goal) and the goal's budget. ``null``
  // budget floored to the $25 default. Stamped by command_runtime._goal_state and every
  // turn_ended; the goal widget + footer render straight from these.
  if ("session_cost_usd" in src)
    state.sessionCostUsd = src.session_cost_usd == null ? null : Number(src.session_cost_usd);
  if ("goal_cost_usd" in src)
    state.goalCostUsd = src.goal_cost_usd == null ? null : Number(src.goal_cost_usd);
  if ("goal_max_cost_usd" in src)
    state.goalMaxCostUsd = src.goal_max_cost_usd == null ? null : Number(src.goal_max_cost_usd);
}

// Live turn-metrics goal sync — the ONLY entry point for the goal slice of a
// turn_ended / round metrics frame (streaming.js). The engine stamps
// ``active_goal`` from its IN-MEMORY projection (turn_loop.py:
// ``self._active_goals.get(sid, "") or ""``) while ``goal_id`` comes from the
// persisted Session (cost.py ``_goal_progress_metrics``) — the two can
// disagree: a goal that is still RUNNING in the registry may have an empty
// in-memory text (goal set via the Goals page, after an engine restart, or a
// supervisor-driven turn) while ``goal_id`` is present. An empty
// ``active_goal`` on this channel is therefore NO INFORMATION, not a clear:
// wiping ``state.activeGoal`` here is what made the widget vanish on idle
// The persistent budget indicator must stay up while a
// goal is active).
//
// Reconcile against the durable registry instead: refreshGoalCost() is the
// authoritative hide path (met/archived → resetGoalTracker) and restores the
// goal text + cost while the goal is running. Genuinely cleared goals still
// detach — the explicit-empty clears arrive on the authoritative channels
// (state_patch / /goal clear, header_update, /status), which keep their
// normal clear semantics.
function applyTurnGoalMetrics(metrics) {
  if (!metrics || typeof metrics !== "object") return;
  if ("active_goal" in metrics) {
    const ag = String(metrics.active_goal || "").trim();
    if (ag) {
      state.activeGoal = ag;
    } else if ((state.goalId || "").trim()) {
      // Known goal, empty text: keep the last good text and let the registry
      // decide (hide on met/archived, restore text+cost on running).
      refreshGoalCost();
    } else {
      state.activeGoal = "";
    }
  }
  applyGoalProgress(metrics, { allowClear: false });
}

// Goal widget — mirrors the TUI's #goal-scroll / #goal-text. Shows the active
// goal text with a 🎯 icon PLUS a progress tracker (terminal state badge +
// turns-consumed/budget the outer goal loop reports). Collapses to zero height
// when no goal AND no terminal state. All numbers/labels are computed by the
// backend (command_runtime._goal_state for /status, the engine's live turn
// metrics for streaming) — this renders, it never derives.
const GOAL_TERMINAL_META = {
  met: { label: "completed", cls: "goal-met", icon: "✓" },
  blocked: { label: "blocked", cls: "goal-blocked", icon: "🚫" },
  paused: { label: "paused", cls: "goal-paused", icon: "⏸" },
};

// applyIncomingGoalText — identity-aware write of ``active_goal`` TEXT from
// an incoming snapshot (the /status reconcile and header_update). Empty
// text is NO INFORMATION only when the snapshot identifies the SAME goal,
// or when it is a badge-only linger (valid terminal, no incoming id). A
// DIFFERENT incoming goal_id must not inherit the previous goal's text
// even if it carries a valid terminal (codex round 2: validTerminal
// accepted without checking the incoming id).
function applyIncomingGoalText(src) {
  if (!src || typeof src !== "object") return;
  if (!("active_goal" in src)) return;
  const ag = String(src.active_goal || "").trim();
  const incomingGoalId = String(src.goal_id || "").trim();
  const incomingTerminal = String(src.goal_terminal || "").trim();
  const sameGoal =
    !!incomingGoalId && incomingGoalId === String(state.goalId || "").trim();
  const validTerminal =
    !!incomingTerminal &&
    Object.prototype.hasOwnProperty.call(
      GOAL_TERMINAL_META,
      incomingTerminal.toLowerCase()
    );
  if (ag) {
    state.activeGoal = ag;
  } else if (sameGoal || (validTerminal && !incomingGoalId)) {
    // Preserve: same goal, or badge-only linger with no incoming id.
  } else {
    state.activeGoal = "";
  }
}

function applyStatusSnapshot(sd) {
  if (!sd || typeof sd !== "object") return;
  if (!Object.keys(sd).length) return;
  state.statusInfo = Object.assign({}, state.statusInfo || {}, sd);
  if (typeof syncModelSheetEffortFromStatus === "function")
    syncModelSheetEffortFromStatus();
  if ("teacher_active" in sd) state.teacherActive = !!sd.teacher_active;
  if ("teacher_enabled" in sd) state.teacherEnabled = !!sd.teacher_enabled;
  if ("teacher_model" in sd) state.teacherModel = sd.teacher_model || "";
    if ("active_goal" in sd) {
      applyIncomingGoalText(sd);
    }
    applyGoalProgress(sd);
}

// applyHeaderGoalPatch — sole header_update entry for the goal slice.
// ``applyGoalProgress`` writes identity (id/terminal/cost) but never the
// visible text, so a met/archive projection (goal_registry.py: empty
// goal_id + empty active_goal + empty goal_terminal) used to clear the
// id while leaving the old label on screen. An explicit detached
// projection resets the tracker; any other goal-bearing patch consumes
// active_goal identity-aware, applies progress, and re-renders.
function applyHeaderGoalPatch(patch) {
  if (!patch || typeof patch !== "object") return;
  const hasGoalSlice =
    "goal_id" in patch ||
    "active_goal" in patch ||
    "goal_terminal" in patch ||
    "goal_turn_count" in patch ||
    "max_goal_turns" in patch ||
    "goal_cost_usd" in patch ||
    "goal_max_cost_usd" in patch ||
    "session_cost_usd" in patch;
  if (!hasGoalSlice) return;
  const incomingGoalId = String(patch.goal_id || "").trim();
  const incomingText = String(patch.active_goal || "").trim();
  const incomingTerminal = String(patch.goal_terminal || "").trim();
  const detached =
    "goal_id" in patch &&
    "active_goal" in patch &&
    !incomingGoalId &&
    !incomingText &&
    (!("goal_terminal" in patch) || !incomingTerminal);
  if (detached) {
    resetGoalTracker();
    return;
  }
  if (state.goalDismissedAfterMet && incomingTerminal.toLowerCase() === "met") {
    return;
  }
  applyIncomingGoalText(patch);
  applyGoalProgress(patch);
  renderGoalWidget();
}

// GOAL_WIDGET_VIEW — pure, DOM-free presentation model for the goal widget.
// Given the goal slice of `state`, decides whether the widget is visible and
// returns the label + tracker HTML to mount. Zero logic/classification here:
// the terminal state ("met"/"blocked"/"paused") is decided by core and arrives
// pre-classified in `goalTerminal`; this only FORMATS it. Extracted so it is
// unit-testable without a DOM (see web/tests/test_goal_widget.js).
function goalWidgetView(gs) {
  const goal = (gs.activeGoal || "").trim();
  const term = (gs.goalTerminal || "").toLowerCase();
  const meta = GOAL_TERMINAL_META[term];
  // The widget stays visible while a goal is active OR while a TERMINAL state
  // (met/blocked/paused) lingers — the engine clears the goal text the instant
  // it settles, so the met/blocked badge has to outlive active_goal or it would
  // never be seen. A met badge persists until the next ordinary
  // user turn (send.js dismisses it); blocked/paused persist until a new goal
  // is set or the session switches. Collapse only when there is neither
  // a goal nor a terminal state to show.
  if (!goal && !meta) return { visible: false, label: "", trackerHtml: "" };

  // Progress tracker: a terminal-state badge once the outer loop settles, and
  // (while the goal is still working) a COST label — spend / budget in USD,
  // floored to the $25 default (never ∞). There is intentionally NO turn counter
  // and NO progress bar: cost — not a turn cap — governs the loop, and the bar
  // didn't flex well. Cost shows once the backend has stamped it (goalCostUsd
  // is a number); a brand-new goal with no spend yet shows no tracker.
  let trackerHtml = "";
  const wait = term === "blocked" || term === "paused" ? String(gs.goalWait || "").trim() : "";
  if (meta && wait) {
    // A goal waiting on the owner says for what (backend label, verbatim).
    trackerHtml =
      `<span class="goal-status goal-waiting" title="${escapeHtml(wait)}">` +
      `⏳ ${escapeHtml(wait)}</span>`;
  } else if (meta) {
    trackerHtml =
      `<span class="goal-status ${meta.cls}" title="goal ${meta.label}">` +
      `${meta.icon} ${meta.label}</span>`;
  } else if (typeof gs.goalCostUsd === "number") {
    const spend = formatUsd(gs.goalCostUsd);
    // Never ∞ — floor to the default when no explicit budget.
    const budget = formatUsd(effBudget(gs.goalMaxCostUsd));
    trackerHtml =
      `<span class="goal-status goal-running goal-cost" ` +
      `title="goal spend / budget">${spend} / ${budget}</span>`;
  }

  // When the goal text is gone but a terminal badge lingers, label the row
  // "goal <state>" so the badge has context (the original condition text is no
  // longer on the session). While a goal is active, show its text.
  const label = goal
    ? escapeHtml(goal.replace(/\s+/g, " "))
    : `<span class="goal-text-dim">goal ${meta ? meta.label : ""}</span>`;
  const goalId = (gs.goalId || "").trim();
  const base = String(gs.portfolioBase || "").trim();
  const goalLinkHtml = goalId && base
    ? `<a class="goal-portfolio-link" href="${escapeHtml(base)}/goals/${escapeHtml(goalId)}" ` +
      `title="Open in goals portfolio">↗</a>`
    : "";

  // Status-aware action button: pause a running goal, resume a paused/blocked
  // one. Terminal states (met, archived) get no button.
  let actionButtonHtml = "";
  if (goalId) {
    if (!term) {
      actionButtonHtml =
        `<button class="goal-action-btn" data-action="pause" ` +
        `title="Pause goal">⏸</button>`;
    } else if (term === "paused" || term === "blocked") {
      actionButtonHtml =
        `<button class="goal-action-btn" data-action="resume" ` +
        `title="Resume goal">▶</button>`;
    }
  }

  return { visible: true, label, trackerHtml, goalLinkHtml, actionButtonHtml };
}

function renderGoalWidget() {
  if (!goalWidget) return;
  const view = goalWidgetView({
    activeGoal: state.activeGoal,
    goalId: state.goalId,
    goalTerminal: state.goalTerminal,
    goalWait: state.goalWait,
    goalCostUsd: state.goalCostUsd,
    goalMaxCostUsd: state.goalMaxCostUsd,
    portfolioBase: BASE,
  });
  if (!view.visible) {
    goalWidget.hidden = true;
    goalWidget.classList.remove("goal-active");
    goalWidget.innerHTML = "";
    return;
  }
  goalWidget.hidden = false;
  goalWidget.classList.add("goal-active");
  const term = (state.goalTerminal || "").toLowerCase();
  const icon = term === "met" ? "✓" : "🎯";
  goalWidget.innerHTML = `
    <div class="goal-row">
      <span class="goal-icon">${icon}</span>
      <span class="goal-text">${view.label}</span>
      ${view.goalLinkHtml || ""}
      ${view.trackerHtml}
      ${view.actionButtonHtml || ""}
      <button class="goal-clear" title="Clear goal (/goal clear)">✕</button>
    </div>`;
  goalWidget.querySelector(".goal-clear").addEventListener("click", () => {
    runCommand("/goal clear");
  });
  const actionBtn = goalWidget.querySelector(".goal-action-btn");
  if (actionBtn) {
    actionBtn.addEventListener("click", async () => {
      const action = actionBtn.dataset.action;
      const gid = (state.goalId || "").trim();
      if (!gid) return;
      actionBtn.disabled = true;
      try {
        await apiPost("/api/goals/" + encodeURIComponent(gid) + "/" + action, {});
        await refreshGoalCost();
        renderGoalWidget();
      } catch (err) {
        console.error("Goal action failed:", err);
      } finally {
        actionBtn.disabled = false;
      }
    });
  }
  // Goal text is clamped to 2 lines by default so a long goal can't crush the
  // chat; let the user reveal the whole thing on demand (tap to expand, hover
  // tooltip carries the full text). Expansion is sticky across the frequent
  // re-renders during a live turn (state.goalTextExpanded).
  const textEl = goalWidget.querySelector(".goal-text");
  if (textEl) {
    const fullGoal = (state.activeGoal || "").replace(/\s+/g, " ").trim();
    if (fullGoal) textEl.title = fullGoal;
    // Detect overflow against the collapsed (clamped) box, then apply the
    // sticky expanded state.
    const clamped = textEl.scrollHeight > textEl.clientHeight + 1;
    if (clamped) textEl.classList.add("is-clamped");
    if (state.goalTextExpanded) textEl.classList.add("expanded");
    textEl.addEventListener("click", () => {
      state.goalTextExpanded = textEl.classList.toggle("expanded");
    });
  }
}

// Authoritative goal-cost sync. The streaming turn metrics stamp goalCostUsd
// only at turn_ended, so between turns (or in a brand-new goal) the footer
// widget can read $0 while the goal store has already accrued spend.
function applyGoalDetail(g, sid = state.currentSessionId) {
  if (!g) return;
  const gid = (g.goal_id || g.id || "").trim();
  // Only boot/state/header data can bind a goal. A delayed detail response
  // cannot rebind the widget after navigation, goal replacement, or clear.
  if (!sid || sid !== state.currentSessionId || state.goalSid !== sid) return;
  if (!gid || gid !== state.goalId) return;
  if (g.root_session_id && g.root_session_id !== sid) return;
  if (g.status === "archived") {
    resetGoalTracker();
    return;
  }
  if (state.goalDismissedAfterMet && g.status === "met") {
    resetGoalTracker();
    return;
  }
  if (g.cost_usd != null) state.goalCostUsd = Number(g.cost_usd);
  if ("wait_label" in g) state.goalWait = String(g.wait_label || "");
  if (g.max_cost_usd != null) state.goalMaxCostUsd = Number(g.max_cost_usd);
  var _st = { blocked: "blocked", paused: "paused", met: "met" };
  var _nextTerm = _st[g.status] || "";
  if (_nextTerm !== state.goalTerminal) {
    state.goalTerminal = _nextTerm;
  }
  if (g.condition_text) {
    state.activeGoal = g.condition_text;
  }
  renderGoalWidget();
}
window.applyGoalDetail = applyGoalDetail;

// Goals page (/api/goals → cost_usd) and this footer must agree. Poll the
// goal's persisted cost_usd / max_cost_usd and overwrite the local slice with
// that single source of truth so both views read the same number.
async function refreshGoalCost() {
  const gid = (state.goalId || "").trim();
  const sid = state.currentSessionId;
  if (!gid || !sid || state.goalSid !== sid) return;
  try {
    const g = await apiGet("/api/goals/" + encodeURIComponent(gid));
    // Re-check the LIVE goal id (not just the API echo): the active goal may
    // have switched while this fetch was in flight, so a stale response must
    // not overwrite the now-current goal's cost (mirrors goals.js's guard).
    if (state.currentSessionId !== sid || state.goalSid !== sid) return;
    if ((state.goalId || "").trim() !== gid) return;
    if (!g || g.goal_id !== gid) return; // API echoed a different goal
    applyGoalDetail(g, sid);
  } catch (_e) { /* leave the streamed value in place on a fetch blip */ }
}

// The budget is NEVER unbounded. The backend floors every budget to
// its DEFAULT_BUDGET_USD, so the widget shows a real cap, not ∞. This mirror is a
// defensive fallback ONLY (a null should never reach here); keep it in step with
// goal_service.DEFAULT_BUDGET_USD, which is the single source of truth.
const DEFAULT_BUDGET_USD = 25.0;
const effBudget = (v) => (typeof v === "number" && v > 0 ? v : DEFAULT_BUDGET_USD);

// Render the whole-tree cost/budget pair. The Agent Monitor
// header intentionally exposes only `Σ spent / root budget`; detailed
// downstream/delegate slices remain available from the underlying API.
// Severity of a spend/budget pair — healthy → warning → critical, mapped to the brand
// semantic tokens (see spec/brand-book.md); never a divide-by-zero (budget is floored).
function budgetSeverity(p) {
  // Prefer the backend's verdict (single source of truth, tested there).
  if (p && (p.severity === "ok" || p.severity === "warn" || p.severity === "crit")) {
    return p.severity;
  }
  if (!p) return "ok";
  const r = p.spent_usd / effBudget(p.budget_usd);
  if (r >= 0.9) return "crit";   // at/over a hard cap → --error (red)
  if (r >= 0.75) return "warn";  // approaching → --warning (amber)
  return "ok";                   // healthy → --success (green)
}

function budgetPair(p) {
  return formatUsd(p.spent_usd) + " / " + formatUsd(effBudget(p.budget_usd));
}

// One Agent Monitor cost/budget segment: only `Σ spent / root budget`, in
// severity color. The monitor segment remains the budget indicator and editor
// trigger; the API still carries the detailed hierarchy for other consumers.
function stampBudgetFromGoal(budget, goal) {
  const view = budget && typeof budget === "object" ? Object.assign({}, budget) : { is_subagent: false, total: {} };
  const total = Object.assign({}, view.total || {});
  const spent = Number(total.spent_usd || 0);
  const goalCost = goal ? Number(goal.cost_usd) : NaN;
  if (spent === 0 && Number.isFinite(goalCost) && goalCost > 0) {
    total.spent_usd = goalCost;
    view.total = total;
  }
  return view.total ? view : budget;
}
if (typeof window !== "undefined") window.stampBudgetFromGoal = stampBudgetFromGoal;

function budgetSegment(b) {
  const part = (p, glyph, title) =>
    `<span class="bud bud-${budgetSeverity(p)}" title="${title}">${glyph} ${budgetPair(p)}</span>`;
  const parts = [part(b.total, "Σ", "whole-tree spend / root budget")];
  return (
    `<button type="button" class="seg seg-cost seg-budget seg-budget-btn"` +
    ` id="budget-edit-trigger" aria-haspopup="dialog" aria-expanded="false"` +
    ` title="Edit whole-tree budget">${parts.join(" ")}` +
    `<span class="seg-caret" aria-hidden="true">▾</span></button>`
  );
}

function sessionGwHash(sid) {
  return String(sid || "").trim();
}

function applyLocalBudgetCap(usd) {
  const cap = effBudget(Number(usd));
  const spentRaw = (state.budget && state.budget.total && state.budget.total.spent_usd);
  const spent = Number(spentRaw != null ? spentRaw : state.sessionCostUsd);
  const pair = {
    spent_usd: Number.isFinite(spent) ? spent : 0,
    budget_usd: cap,
  };
  if (!state.budget) {
    state.budget = { is_subagent: false, total: pair };
  } else {
    state.budget.total = Object.assign({}, state.budget.total || {}, pair);
    if (state.budget.delegate) {
      state.budget.delegate = Object.assign({}, state.budget.delegate, {
        budget_usd: cap,
      });
    }
    if (state.budget.downstream) {
      state.budget.downstream = Object.assign({}, state.budget.downstream, {
        budget_usd: cap,
      });
    }
  }
  state._budgetWrite = {
    sid: String(state.currentSessionId || "").trim(),
    budgetUsd: cap,
    at: Date.now(),
  };
  renderStatusSegments();
}

function ignoreStaleBudgetSnapshot(sid, incomingUsd) {
  const w = state._budgetWrite;
  if (!w) return false;
  if (String(w.sid || "") !== String(sid || "").trim()) return false;
  if ((Date.now() - w.at) > 15000) {
    state._budgetWrite = null;
    return false;
  }
  if (incomingUsd == null || incomingUsd === "") return true;
  const incoming = Number(incomingUsd);
  if (!Number.isFinite(incoming)) return true;
  if (Math.abs(incoming - w.budgetUsd) < 1e-6) {
    state._budgetWrite = null;
    return false;
  }
  return true;
}

async function refreshBudget() {
  const sid = (state.currentSessionId || "").trim();
  if (!sid) { state.budget = null; return; }
  const hash = sessionGwHash(sid);
  if (!hash) { state.budget = null; return; }
  try {
    const v = await apiGet("/api/budget/" + encodeURIComponent(hash));
    if ((state.currentSessionId || "").trim() !== sid) return; // stale response, session switched
    const incoming = v && v.total ? v : null;
    if (ignoreStaleBudgetSnapshot(sid, incoming && incoming.total && incoming.total.budget_usd))
      return;
    state.budget = incoming;
    renderStatusSegments(); // repaint the Agent Monitor budget segment
  } catch (_e) { /* leave last render in place on a fetch blip */ }
}

// Convert the numeric editor value to the shared slash-command surface.
// Keeping this pure makes the validation/command text independently testable.
// Zero intentionally means "reset to the configured default".
function parseBudgetEditorValue(raw) {
  const text = String(raw == null ? "" : raw).trim();
  if (!text) return { ok: false, error: "Enter a budget in USD." };
  const value = Number(text);
  if (!Number.isFinite(value) || value < 0) {
    return { ok: false, error: "Budget must be a finite number of USD, 0 or greater." };
  }
  return { ok: true, value, command: "/budget " + String(value) };
}

function budgetEditorNodes() {
  return {
    sheet: document.getElementById("budget-editor"),
    backdrop: document.getElementById("budget-backdrop"),
    form: document.getElementById("budget-form"),
    input: document.getElementById("budget-input"),
    context: document.getElementById("budget-context"),
    error: document.getElementById("budget-error"),
    save: document.getElementById("budget-save"),
  };
}

function setBudgetEditorOpen(open) {
  const n = budgetEditorNodes();
  if (!n.sheet) return;
  if (open) n.sheet.removeAttribute("hidden");
  else n.sheet.setAttribute("hidden", "");
  if (n.backdrop) {
    if (open) n.backdrop.removeAttribute("hidden");
    else n.backdrop.setAttribute("hidden", "");
  }
  const trigger = document.getElementById("budget-edit-trigger");
  if (trigger) trigger.setAttribute("aria-expanded", open ? "true" : "false");
}

function openBudgetEditor() {
  if (!(state.currentSessionId || "").trim()) return false;
  const n = budgetEditorNodes();
  if (!n.sheet || !n.input) return false;

  // Only one footer sheet may be open at a time.
  const agentSheet = document.getElementById("agent-model-pop");
  const agentBackdrop = document.getElementById("amp-backdrop");
  if (agentSheet) agentSheet.setAttribute("hidden", "");
  if (agentBackdrop) agentBackdrop.setAttribute("hidden", "");
  if (window.SaAgentMon && window.SaAgentMon.close) {
    window.SaAgentMon.close({ restoreFocus: false });
  }
  if (window.SaContextSheet && window.SaContextSheet.isOpen()) {
    window.SaContextSheet.close({ restoreFocus: false });
  }

  const total = state.budget && state.budget.total;
  const budget = Number(total && total.budget_usd);
  const spent = Number(total && total.spent_usd);
  n.input.value = Number.isFinite(budget) && budget >= 0 ? String(budget) : "";
  if (n.context) {
    n.context.textContent =
      "Whole-tree spend " + formatUsd(Number.isFinite(spent) ? spent : 0) +
      ". Enter 0 to restore the default budget.";
  }
  if (n.error) {
    n.error.textContent = "";
    n.error.setAttribute("hidden", "");
  }
  setBudgetEditorOpen(true);
  window.setTimeout(() => {
    n.input.focus();
    n.input.select();
  }, 0);
  return true;
}

function closeBudgetEditor() {
  const n = budgetEditorNodes();
  if (n.save) n.save.disabled = false;
  setBudgetEditorOpen(false);
}

async function saveBudgetEditor() {
  const n = budgetEditorNodes();
  if (!n.input || !n.save) return;
  const parsed = parseBudgetEditorValue(n.input.value);
  if (!parsed.ok) {
    if (n.error) {
      n.error.textContent = parsed.error;
      n.error.removeAttribute("hidden");
    }
    n.input.focus();
    return;
  }

  const sid = (state.currentSessionId || "").trim();
  if (!sid) {
    closeBudgetEditor();
    return;
  }
  if (
    typeof isFreshUnboundSession === "function" &&
    isFreshUnboundSession(sid)
  ) {
    if (n.error) {
      n.error.textContent =
        "Send a message first to bind this chat before setting a budget.";
      n.error.removeAttribute("hidden");
    }
    return;
  }
  n.save.disabled = true;
  if (n.error) n.error.setAttribute("hidden", "");
  try {
    // Deliberately reuse the shared, authenticated command runtime rather than
    // adding a Web-only mutation route.
    const resp = await apiPost("/api/command", {
      text: parsed.command,
      session_id: sid,
      agent: state.currentAgent,
    });
    const result = (resp && resp.result) || {};
    if (!resp || resp.ok === false || result.kind === "error") {
      throw new Error(result.text || resp.error || "Budget update failed.");
    }
    if ((state.currentSessionId || "").trim() === sid) {
      applyLocalBudgetCap(parsed.value);
      if (typeof applyEffects === "function") applyEffects(resp.effects || []);
    }
    closeBudgetEditor();
    if ((state.currentSessionId || "").trim() === sid) await refreshBudget();
  } catch (err) {
    if (n.error) {
      n.error.textContent = (err && err.message) || "Budget update failed.";
      n.error.removeAttribute("hidden");
    }
  } finally {
    n.save.disabled = false;
  }
}

function bindBudgetEditor() {
  const n = budgetEditorNodes();
  if (!n.form || n.form.dataset.bound) return;
  n.form.dataset.bound = "1";
  n.form.addEventListener("submit", function (e) {
    e.preventDefault();
    saveBudgetEditor();
  });
  const cancel = document.getElementById("budget-cancel");
  const close = document.getElementById("budget-close");
  if (cancel) cancel.addEventListener("click", closeBudgetEditor);
  if (close) close.addEventListener("click", closeBudgetEditor);
  if (n.backdrop) n.backdrop.addEventListener("click", closeBudgetEditor);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && n.sheet && !n.sheet.hasAttribute("hidden")) {
      closeBudgetEditor();
    }
  });
}

document.addEventListener("DOMContentLoaded", bindBudgetEditor);

// resetGoalTracker — drop the FULL goal slice (text + terminal badge + turn
// counters/budget) and repaint the widget. Hide authority is session switch,
// /goal clear, met/archived, and the X button. Pure-state + render.
function resetGoalTracker() {
  state.activeGoal = "";
  state.goalId = "";
  state.goalSid = "";
  state.goalTerminal = "";
  state.goalWait = "";
  state.goalTurnCount = 0;
  state.maxGoalTurns = 0;
  state.goalCostUsd = null;
  state.goalMaxCostUsd = null;
  renderGoalWidget();
}

// Map the composer ticker phase onto the same session_state enum the
// Agents footer current row paints. Generic "running" hid reasoning /
// tool_running while #status-text already had the live phase.
function _phaseToFooterState(phase, snap) {
  const p = String(phase || "");
  const live = String(snap || "").trim();
  if (p === "tool") return "tool_running";
  if (p === "writing") return live && live !== "idle" ? live : "streaming";
  if (p === "sending" || p === "received" || p === "warming") {
    return live && live !== "idle" ? live : "prompting";
  }
  if (p === "retrying" || p === "compacting" || p === "ask") return p;
  if (p === "reasoning" || p === "prefill" || p === "prompting" || p === "streaming") {
    return p;
  }
  return live || "running";
}

// Feed the chat-docked Agent Monitor accordion for this SID.
function updateAgentMonAccordion() {
  if (!window.SaAgentMon) return;
  if (!state.currentSessionId) {
    window.SaAgentMon.update({ active: false });
    return;
  }
  const sess = state.sessions[state.currentSessionId] || {};
  const unused = typeof isUnusedWebPlaceholder === "function"
    && isUnusedWebPlaceholder(state.currentSessionId);
  const chromeAgent = unused
    ? ((typeof selectedPickerAgent === "function" && selectedPickerAgent())
      || state.pickerAgent || state.currentAgent || "jarvis")
    : (state.currentAgent || "jarvis");
  const live = !unused && !!(state.isStreaming || state.treeLive || _turn);
  // Same status the top #status-text ticker uses: live session_state, then
  // the in-flight turn phase. Do not coarsen a live turn to "running".
  // A never-bound New Chat row cannot be "running" — leaked isStreaming
  // from the previous turn must not paint the placeholder.
  const snap = String(
    sess.state || sess.session_state
    || (sess.activity && sess.activity.phase) || "",
  ).trim();
  let stateLabel = "idle";
  if (!unused && (state.isStreaming || _turn)) {
    stateLabel = _phaseToFooterState(_turn && _turn.phase, snap);
  } else if (!unused && state.treeLive) {
    stateLabel = snap && snap !== "idle" ? snap : "waiting";
  } else if (!unused && snap && snap !== "idle") {
    const ownLive = typeof _ownTurnLiveFrom === "function"
      ? _ownTurnLiveFrom(sess)
      : false;
    const ignoreStale = typeof _shouldIgnoreStaleOwnLive === "function"
      && _shouldIgnoreStaleOwnLive();
    if (ownLive && !ignoreStale) stateLabel = snap;
    else if (snap === "waiting" || snap === "ask" || snap === "error") stateLabel = snap;
  } else if (!unused) {
    const cache = state.msgCache || [];
    for (let i = cache.length - 1; i >= 0; i -= 1) {
      const m = cache[i];
      if (!m || m.role !== "assistant") continue;
      if (m.status === "error" || (m.error != null && String(m.error))) {
        stateLabel = "error";
      }
      break;
    }
  }
  const hash = sessionGwHash(state.currentSessionId);
  const returnSessionId = String(state.monitorReturnSessionId || "");
  window.SaAgentMon.update({
    active: true,
    hash: hash,
    scopeHash: returnSessionId
      ? String(state.monitorScopeHash || hash)
      : hash,
    returnSessionId: returnSessionId,
    agent: chromeAgent,
    // Same label the composer selector shows (selected/picker alias). The
    // resolved engine id (qwen-3.8 under Cursor Auto) must not appear as a
    // second identity for the same agent.
    model: (typeof pickerOrSessionModelLabel === "function"
      ? pickerOrSessionModelLabel(chromeAgent || sess.agent || "")
      : "") || sess.model || "",
    title: sess.title || "this session",
    live: live,
    stateLabel: stateLabel,
    // Rows come from boot/stream applySnapshot only. Re-feeding sess.footer
    // here would paint a stale parent current over the open-session chrome.
  });
}

// Harness widget — todo / tasks counters with expandable lists.
// Mounted at the top of the composer card, above the idle/live ticker.
// Todo is expanded by default; the chip row toggles the list, which grows
// upward (CSS column-reverse) and scrolls after 5 rows. List scrollTop is
// captured before the innerHTML wipe and restored after; the first paint
// for a session scrolls the in_progress row into view.
function renderHarnessWidget() {
  // Keep the chat-docked Agent Monitor accordion in lock-step with the footer's
  // subagent widget — same per-session `state.subagents` source, same triggers.
  updateAgentMonAccordion();
  renderQueueWidget();
  if (!harnessWidget) return;
  if (state._hwScrollSid !== state.currentSessionId) {
    state._hwScrollSid = state.currentSessionId || "";
    state.hwScroll = { todo: null, task: null };
    state.hwScrolledOnce = { todo: false, task: false };
  }
  harnessWidget.querySelectorAll(".hw-list").forEach((list) => {
    const key = list.dataset.key;
    if (key) {
      state.hwScroll = state.hwScroll || {};
      state.hwScroll[key] = list.scrollTop;
    }
  });
  const todos = state.todos || [];
  const tasks = state.harnessTasks || [];
  if (!todos.length && !tasks.length) {
    harnessWidget.hidden = true;
    harnessWidget.innerHTML = "";
    return;
  }
  harnessWidget.hidden = false;
  const groups = [];
  if (todos.length) groups.push(hwGroup("todo", "📋", "todo", todos, true));
  if (tasks.length) groups.push(hwGroup("task", "🗒️", "tasks", tasks, true));
  harnessWidget.innerHTML = groups.join("");
  // Wire chip toggles (re-bound on every render; cheap for ≤4 chips).
  harnessWidget.querySelectorAll(".hw-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const key = chip.dataset.key;
      state.hwOpen[key] = !state.hwOpen[key];
      renderHarnessWidget();
    });
  });
  harnessWidget.querySelectorAll(".hw-list").forEach((list) => {
    const key = list.dataset.key;
    if (!key) return;
    state.hwScroll = state.hwScroll || {};
    state.hwScrolledOnce = state.hwScrolledOnce || {};
    const restore = () => {
      if (!state.hwScrolledOnce[key]) {
        const active = list.querySelector("li.in_progress");
        if (active) {
          const listBox = list.getBoundingClientRect();
          const rowBox = active.getBoundingClientRect();
          const mid = (list.clientHeight - rowBox.height) / 2;
          list.scrollTop += rowBox.top - listBox.top - mid;
        }
        state.hwScrolledOnce[key] = true;
        state.hwScroll[key] = list.scrollTop;
      } else if (state.hwScroll[key] != null) {
        list.scrollTop = state.hwScroll[key];
      }
    };
    restore();
    list.addEventListener("scroll", () => {
      state.hwScroll[key] = list.scrollTop;
    }, { passive: true });
  });
  // Build one chip + (optionally) its expanded list. ``checklist`` items carry
  // a status glyph; subagents render as a plain titled list.
  function hwGroup(key, icon, label, items, checklist) {
    const open = !!state.hwOpen[key];
    let countLabel;
    if (checklist) {
      const done = items.filter((i) => (i.status || "") === "completed").length;
      countLabel = `${done}/${items.length}`;
    } else {
      countLabel = String(items.length);
    }
    const chip =
      `<button type="button" class="hw-chip${open ? " open" : ""}" data-key="${key}" ` +
      `aria-expanded="${open ? "true" : "false"}" title="${label}">${icon} ` +
      `<span class="hw-count">${countLabel}</span> ${label}` +
      `<span class="hw-caret">›</span></button>`;
    if (!open) return `<span class="hw-group">${chip}</span>`;
    const glyph = { completed: "✔", in_progress: "▸", pending: "○" };
    const rows = items
      .map((it) => {
        if (checklist) {
          const st = it.status || "pending";
          const cls =
            st === "completed"
              ? "done"
              : st === "in_progress"
                ? "in_progress"
                : "";
          const text =
            st === "in_progress" && it.activeForm
              ? it.activeForm
              : it.content || "";
          return `<li class="${cls}"><span class="hw-mark">${glyph[st] || "○"}</span><span>${escapeHtml(text)}</span></li>`;
        }
        return `<li><span class="hw-mark">•</span><span>${escapeHtml(it.title || it.id || "")}</span></li>`;
      })
      .join("");
    return `<span class="hw-group open">${chip}<ul class="hw-list" data-key="${key}">${rows}</ul></span>`;
  }

}

function mergeQueuedText(items) {
  return (items || [])
    .map((item) => String((item && item.text) || "").trim())
    .filter(Boolean)
    .join("\n\n");
}

function queuedMessageIds(items) {
  return [...new Set(
    (items || [])
      .map((item) => String((item && item.queue_id) || ""))
      .filter(Boolean),
  )];
}

// Follow-up queue — deliberately mounted immediately above the composer.
// Multiple durable entries can arrive from another client, so present them as
// one compact message and apply edit/remove/Steer to the whole aggregate.
function renderQueueWidget() {
  const widget = queueWidget || document.getElementById("queue-widget");
  if (!widget) return;
  const queue = state.queue || [];
  if (!queue.length) {
    widget.hidden = true;
    widget.innerHTML = "";
    return;
  }
  const attr = (value) => escapeHtml(String(value == null ? "" : value))
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  const busy = typeof treeBusy === "function"
    ? treeBusy()
    : !!state.isStreaming;
  const locked = !!state.steeringQueueId || !!state.queueSubmitting;
  const ids = queuedMessageIds(queue);
  const first = queue[0] || {};
  const qid = attr(ids[0] || "");
  // The backend sends ``display_text`` (typed words) + ``attached_files``
  // chips for follow-ups that carry saved files; ``text`` stays the
  // model-facing payload a steer sends on.
  const text = mergeQueuedText(queue.map((item) => ({
    text: item && typeof item.display_text === "string" ? item.display_text : (item && item.text),
  })));
  const fileNames = queue.flatMap((item) =>
    (item && Array.isArray(item.attached_files) ? item.attached_files : [])
      .map((file) => `📎 ${file.name}`));
  const shown = [text, fileNames.join(" ")].filter(Boolean).join(" ");
  const preview = shown.length > 180 ? shown.slice(0, 180) + "…" : shown;
  const source = queue.length > 1
    ? `<span class="queue-source">${queue.length} merged</span>`
    : first.source
      ? `<span class="queue-source">${escapeHtml(first.source)}</span>`
      : "";
  const steering = !!state.steeringQueueId;
  const actionDisabled = !ids.length || (locked && !steering);
  const rows = (
    `<li class="queue-item${steering ? " is-steering" : ""}">` +
      `<div class="queue-item-body">` +
        `<span class="queue-item-mark" aria-hidden="true">○</span>` +
        `<span class="queue-item-text" title="${attr(shown)}">${escapeHtml(preview)}</span>` +
        source +
      `</div>` +
      `<div class="queue-item-actions">` +
        `<button type="button" class="queue-steer" data-queue-id="${qid}"` +
          ` title="${busy ? "Stop the current run and send this message" : "Send this queued message now"}"` +
          `${steering || actionDisabled ? " disabled" : ""}>` +
          `${steering ? "Steering…" : "Steer"}</button>` +
        `<button type="button" class="queue-edit" data-queue-id="${qid}"` +
          ` title="Edit queued message"${locked || !ids.length ? " disabled" : ""}>Edit</button>` +
        `<button type="button" class="queue-delete" data-queue-id="${qid}"` +
          ` title="Remove queued message"${locked || !ids.length ? " disabled" : ""}>Remove</button>` +
      `</div>` +
    `</li>`
  );
  widget.hidden = false;
  widget.innerHTML =
    `<div class="queue-heading">` +
      `<span class="queue-heading-title">📥 Queued message</span>` +
      `<span class="queue-count">1</span>` +
      `<span class="queue-heading-hint">sent after the current run</span>` +
    `</div>` +
    `<ol class="queue-list">${rows}</ol>`;

  widget.querySelectorAll(".queue-steer").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      steerQueueMessages(ids);
    });
  });
  widget.querySelectorAll(".queue-edit").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      queueEditMessages(ids, text);
    });
  });
  widget.querySelectorAll(".queue-delete").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      queueDeleteMessages(ids);
    });
  });
}

// LIVE upsert of the footer task chip from a Task* tool_call. The footer must
// reflect the harness store even for claude_cli sessions, where the tasks live
// inside the subprocess and sharedagents's own `<sid>.tasks.json` is empty — so the
// stream is the only source of truth until turn_ended's refresh.
//
// Matching is by ID, not by content. A TaskCreate carries only a subject (no id
// yet) — we stash the new row under its call_id and learn the real id from the
// result body ("Task #N created…") via `_liveHarnessTaskResult`, the SAME way
// the inline-card path does. TaskUpdate/TaskStop arrive with `taskId` + status
// and NO subject, so a content/prefix match could never find the created row —
// it used to append a bogus bare `#<id>` row each time, losing the name,
// dropping the status, and never clearing stopped tasks. Match on the learned
// id instead; TaskStop removes the row so it actually clears.
const _harnessPendingCreate = {}; // call_id → row awaiting its id from the result
function _liveUpsertHarnessTask(tool, a, callId) {
  const rows = (state.harnessTasks = state.harnessTasks || []);
  state.harnessTasksSid = state.currentSessionId; // live rows belong to this session
  // Synonyms the runner accepts (smaller models reach for title/name/content,
  // and task_id/id for the update key).
  const subject = a.subject || a.title || a.name || a.content;
  if (tool === "TaskCreate") {
    if (subject) {
      const row = { id: "", content: String(subject), status: "pending" };
      rows.push(row);
      if (callId) _harnessPendingCreate[callId] = row;
    }
    return;
  }
  const tidRaw =
    a.taskId != null ? a.taskId : a.task_id != null ? a.task_id : a.id;
  const tid = tidRaw != null ? String(tidRaw) : "";
  // Match by learned id first; fall back to subject (covers a result we never
  // saw) so an update still lands on its row instead of forking a duplicate.
  const row = rows.find(
    (r) =>
      (tid && String(r.id || "") === tid) ||
      (subject && String(r.content || "") === String(subject)),
  );
  if (tool === "TaskStop") {
    if (row) rows.splice(rows.indexOf(row), 1); // stopped → clear it
    return;
  }
  const status = a.status ? String(a.status) : "";
  if (row) {
    if (subject) row.content = String(subject);
    if (status === "deleted") {
      rows.splice(rows.indexOf(row), 1);
      return;
    }
    if (status) row.status = status;
  } else if (tid || subject) {
    // No created row seen (e.g. a session reattach mid-life) — best-effort row.
    if (status === "deleted") return;
    rows.push({
      id: tid,
      content: String(subject || `#${tid}`),
      status: status || "pending",
    });
  }
}

// A TaskCreate's row learns its real id from the result body ("Task #N
// created…"), mirroring the inline-card path. Without this, TaskUpdate/TaskStop
// (which key on the id) can never find the row.
function _liveHarnessTaskResult(callId, content, ok) {
  const row = callId ? _harnessPendingCreate[callId] : null;
  if (!row) return;
  delete _harnessPendingCreate[callId];
  const m = /#(\d+)/.exec(String(content || ""));
  if (m && ok !== false) row.id = m[1];
}

// Repaint the harness widgets (todo / tasks / subagents) + the session status
// bar from `state`. The data itself arrives on the multiplexed `widgets` SSE
// state frame (per-session snapshot) and on live tool events — this function
// never fetches, so it is safe for read-only monitor views too.
async function refreshSessionWidgets(opts = {}) {
  // One in-flight burst, one queued follow-up. Session open + turn_ended +
  // bulk-finish used to stack 3–4 POSTs each (/status /todo /task /subagents)
  // and a long replay made that look like a command storm.
  const now = Date.now();
  if (!opts.force && state._lastWidgetRefreshTs && (now - state._lastWidgetRefreshTs < 500)) {
    return state._widgetRefreshInFlight || null;
  }
  if (state._widgetRefreshInFlight) {
    state._widgetRefreshQueued = true;
    return state._widgetRefreshInFlight;
  }
  if (state.scroll && state.scroll.bulkLoading && !state.isStreaming) {
    state._widgetRefreshQueued = true;
    return null;
  }
  const run = _refreshSessionWidgetsNow();
  state._widgetRefreshInFlight = run;
  try {
    const res = await run;
    state._lastWidgetRefreshTs = Date.now();
    return res;
  } finally {
    state._widgetRefreshInFlight = null;
    if (state._widgetRefreshQueued) {
      state._widgetRefreshQueued = false;
      if (!(state.scroll && state.scroll.bulkLoading && !state.isStreaming)) {
        refreshSessionWidgets();
      }
    }
  }
}

async function _refreshSessionWidgetsNow() {
  if (!state.currentSessionId) {
    state.statusInfo = {};
    state.todos = [];
    state.todosSid = "";
    state.harnessTasks = [];
    state.harnessTasksSid = "";
    state.subagents = [];
    state.queue = [];
    state.steeringQueueId = "";
    state._renderedInjects = new Set(); // clear dedup set on session leave
    renderSessionStatusBar();
    return;
  }
  // Unbound New Chat placeholders must not fire session-scoped /api/command
  // probes before the operator's first real intent (picker → plan, then send).
  if (
    typeof isFreshUnboundSession === "function" &&
    isFreshUnboundSession(state.currentSessionId)
  ) {
    state.statusInfo = {};
    state.todos = [];
    state.todosSid = "";
    state.harnessTasks = [];
    state.harnessTasksSid = "";
    state.subagents = [];
    state.queue = [];
    state.steeringQueueId = "";
    state._renderedInjects = new Set();
    renderSessionStatusBar();
    renderGoalWidget();
    renderHarnessWidget();
    return;
  }
  const epoch = state.navEpoch;
  // Queue state arrives in the same consolidated state frame as the other
  // widgets. Queue transitions then ride the session event stream.
  if (epoch !== state.navEpoch) return;
  renderSessionStatusBar();
  renderGoalWidget();
  renderHarnessWidget();
  renderQueueWidget();
}

// Populate the teacher-model dropdown from the agent's model list (the
// supervising model is one of the configured models[] aliases). Keep the
// selector visible while teacher is off so the operator can choose the model
// before enabling supervision.
function renderTeacherModelSelect() {
  if (!teacherModelSelect) return;
  teacherModelSelect.style.display = "";
  const models = Array.isArray(state.models) ? state.models : [];
  const opts = ['<option value="">— default teacher —</option>'].concat(
    models.map(
      (m) =>
        `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label)}</option>`,
    ),
  );
  teacherModelSelect.innerHTML = opts.join("");
  teacherModelSelect.disabled = state.monitorReadOnly || !models.length;
  if (
    state.teacherModel &&
    models.some((m) => m.id === state.teacherModel)
  ) {
    teacherModelSelect.value = state.teacherModel;
  } else {
    teacherModelSelect.value = "";
  }
}

// Flip teacher supervision for the current session. Needs a live session,
// so create one first (mirrors switchModel).
async function toggleTeacher() {
  const want = state.teacherActive ? "off" : "on";
  try {
    localStorage.setItem("sharedagents.web.teacherActive", want === "on" ? "1" : "0");
  } catch (_e) {}
  if (!state.currentSessionId && !(await createSession(state.draftTopic, { commit: true }))) return;
  await runCommand(`/teacher ${want}`, { silent: true });
}

async function selectTeacherModel(alias) {
  if (!state.currentSessionId && !(await createSession(state.draftTopic, { commit: true }))) return;
  await runCommand(`/teacher model ${alias || "default"}`, { silent: true });
}

// Pull the EFFECTIVE teacher state for the current session from the gateway
// (the no-arg /teacher command returns it as a state_patch). Applied
// silently — no chat output — so loading a session restores the indicator.
async function syncTeacherState() {
  if (state.scroll && state.scroll.bulkLoading) return;
  if (state.monitorReadOnly) return;
  if (!state.currentSessionId) {
    state.teacherActive = false;
    state.teacherModel = "";
    updateTeacher();
    return;
  }
  // Teacher flags arrive on the multiplexed `widgets` SSE frame.
  if (typeof updateTeacher === "function") updateTeacher();
}

function updateSelects() {
  // Backend state patches describe the running lane, not the composer setting.
  // Repaint only from frontend-owned picker state.
  if (state.models && state.models.length) {
    const want = pickerModelAlias();
    if (want) modelSelect.value = want;
  }
}

// ── Follow-up queue CRUD ──
