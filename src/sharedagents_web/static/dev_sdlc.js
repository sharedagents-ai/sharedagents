/* Dev tab — describe → clarify → contract, then the pipeline.
 * Clarify is the validation. Orchestrate walks plan/design/code/tests.
 * Independent verify A + verify B run once after tests, in parallel.
 * LSP checks code. Then merge. Deploy stays a click.
 */
(function () {
  "use strict";
  // Node imports this file for the pure rules below and has no document.
  const doc = typeof document === "undefined" ? null : document;
  const BASE = doc?.querySelector('meta[name="sharedagents-base"]')?.content || "";
  const $ = (id) => (doc ? doc.getElementById(id) : null);
  const esc = (v) => String(v ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  /* The strip, in order. `auto` gates are driven by the server. */
  const GATES = [
    { id: "describe", auto: false },
    { id: "clarify", auto: false },
    { id: "contract", auto: false },
    { id: "plan", auto: true },
    { id: "design", auto: true },
    { id: "code", auto: true },
    { id: "tests", auto: true },
    { id: "deploy", auto: true },
    { id: "ship", auto: false },
  ];
  const AUTO = GATES.filter((g) => g.auto).map((g) => g.id);
  const INTAKE_GATES = new Set(["describe", "clarify", "contract"]);

  let run = null;
  let runs = [];
  let focusedClarify = "";
  // Finished intake gates collapse by default but remain inspectable. Keep the
  // operator's disclosure choices across polling repaints for this run.
  const expandedIntake = new Set();
  /* Question ids the operator has actually edited this round. Kept in JS, not
   * read back off the DOM: paint() renders the gate note before the question
   * list exists, so a DOM count is always one render behind. */
  const touched = new Set();
  let poll = null;
  let busy = false;
  let busyTail = Promise.resolve();
  const pendingActions = new Set();
  let runsSeq = 0;
  let selectionSeq = 0;
  let refreshSeq = 0;
  let selecting = "";
  let preparingStart = false;
  let runsSyncFailed = false;

  /** Presentation follows persisted delivery policy, never assumes a merge. */
  function runPresentation(value) {
    const r = value;
    if (!r) return { step: "describe", label: "Ready", attention: "Start here",
      title: "What would you like to build?",
      detail: "Describe the outcome. Clarify any open questions, then approve the scope and let development run." };
    const step = r.issue_number ? (r.pr_number || r.merged ? "deliver" : "build") : "clarify";
    if (r.status === "blocked" || String(r.deploy_status || "").startsWith("failed") || String(r.deploy_verified || "").startsWith("failed")) {
      return { step, label: "Blocked", tone: "bad", attention: "Needs attention",
        title: "Development needs attention", detail: r.last_error || r.deploy_verified || r.deploy_status || "Review the stage result below, then resume when the blocker is resolved." };
    }
    if (r.status === "aborted") return { step, label: "Stopped", attention: "Run stopped",
      title: "This change is stopped", detail: "The run history remains available. Start a new change when you’re ready." };
    if (r.merged) {
      const deployed = r.deploy_verified === "passed";
      const planned = r.deploy_target && r.deploy_target !== "none";
      return { step: "deliver", label: deployed ? "Deployed" : "Merged", tone: "ok", attention: "Delivery",
        title: deployed ? "Deployment verified" : "Your change is merged",
        detail: deployed ? "The deployed service passed its verification check. Open a new change for follow-up work."
          : planned ? "Deployment was authorized. Its verification status is shown below."
            : "No automatic deployment was requested. The pull request and evidence are linked below." };
    }
    if (r.pr_number) {
      const automatic = (r.publish_mode || "merge") === "merge";
      return { step: "deliver", label: automatic ? "Awaiting merge" : "PR ready", tone: automatic ? "running" : "ok", attention: "Delivery",
        title: automatic ? "Merging after checks pass" : "Your pull request is ready",
        detail: automatic ? "The pipeline will merge when the required checks and independent reviews pass."
          : "Delivery is complete at the pull request. Review the linked changes and merge when you choose." };
    }
    if (r.issue_number) return { step: "build", label: r.driver_active ? "Running" : "Queued", tone: "running", attention: "Automatic development",
      title: r.driver_active ? "Development is in progress" : "Waiting for the next stage",
      detail: "Planning, implementation, tests, and independent review follow the approved scope. You can leave this page and return later." };
    // A late question outranks a ready contract: activeGate() already keeps
    // the operator on Clarify and disables Approve, so the overview must not
    // say "Review scope" over a card that still needs an answer.
    const asking = !!(r.pending_ask && r.pending_ask.call_id) || !!(r.questions || []).length;
    if (asking) return { step: "clarify", label: "Needs input", attention: "Your next step",
      title: "One decision to clarify",
      detail: "Answer the question below. We’ll ask again only if an important choice remains unresolved." };
    if (r.status === "ready") return { step: "clarify", label: "Review scope", attention: "Your next step",
      title: "Review the scope and start development", detail: "Check what will change, how it will be verified, and where delivery should stop." };
    return { step: "clarify", label: "Clarifying", attention: "Preparing your change",
      title: "Checking the request and repository",
      detail: "We’re preparing the requirements and acceptance checks. Any open question will appear here." };
  }

  function paintOverview() {
    const view = runPresentation(run);
    say("sdlc-attention", view.attention);
    say("sdlc-next-title", view.title);
    say("sdlc-next-detail", view.detail);
    const order = ["describe", "clarify", "build", "deliver"];
    doc.querySelectorAll("[data-step]").forEach((el) => {
      const current = el.dataset.step === view.step;
      el.classList.toggle("is-current", current);
      el.classList.toggle("is-complete", order.indexOf(el.dataset.step) < order.indexOf(view.step));
      if (current) el.setAttribute("aria-current", "step");
      else el.removeAttribute("aria-current");
    });
    const badge = $("sdlc-run-state");
    if (badge) { badge.textContent = view.label; badge.className = "sdlc-state " + (view.tone || ""); }
  }

  function syncStatus(message) {
    const el = $("sdlc-sync-status");
    if (el) { el.textContent = message || ""; el.hidden = !message; }
  }

  function resetIntakeFields() {
    ["sdlc-note", "sdlc-requirements", "sdlc-criteria", "sdlc-plan-branch"].forEach((id) => {
      const el = $(id);
      if (el) { el.value = ""; delete el.dataset.dirty; }
    });
    const publish = $("sdlc-publish-mode");
    if (publish) { publish.value = "pr"; delete publish.dataset.chosen; }
    const target = $("sdlc-plan-deploy-target");
    if (target) target.value = "";
    const arm = $("sdlc-arm");
    if (arm) arm.checked = false;
    const force = $("sdlc-merge-force");
    if (force) force.checked = false;
    const forceRow = $("sdlc-merge-force-row");
    if (forceRow) forceRow.hidden = true;
    touched.clear();
    const questions = $("sdlc-questions");
    if (questions) { questions.innerHTML = ""; delete questions.dataset.qids; }
    ["sdlc-start-status", "sdlc-intake-status", "sdlc-contract-status", "sdlc-publish-status"].forEach((id) => say(id, ""));
  }

  function newChange() {
    if (pendingActions.size || preparingStart) return;
    selectionSeq += 1;
    refreshSeq += 1;
    selecting = "";
    run = null;
    rememberRun("");
    expandedIntake.clear();
    resetIntakeFields();
    const box = composer();
    if (box) { box.clear(); box.setFrozen(false); }
    if (poll) clearTimeout(poll);
    syncStatus("");
    paint(); renderRuns();
  }

  function api(path, opts) {
    return fetch(BASE + path, {
      credentials: "same-origin",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      ...opts,
    }).then(async (r) => {
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const err = new Error(data.error || data.detail || r.statusText || "request failed");
        err.status = r.status;
        err.body = data;
        throw err;
      }
      return data;
    });
  }

  const repo = () => ($("dev-repo") || {}).value?.trim() || "";
  const lines = (v) => String(v || "").split("\n").map((s) => s.trim()).filter(Boolean);
  // Mirrors the chat link contract used by dev.js; this file is a separate
  // browser IIFE, so it cannot call that private helper directly.
  const sessionHref = (sid) => BASE + "/?session=" + encodeURIComponent(String(sid || ""));

  function say(id, text, kind) {
    const el = $(id);
    if (!el) return;
    el.textContent = text || "";
    el.classList.toggle("bad", kind === "bad");
    el.classList.toggle("ok", kind === "ok");
  }

  /* ---------- where are we ---------- */

  /** The gate the operator (or the pipeline) is currently on. */
  function activeGate() {
    if (!run) return "describe";
    if (run.merged || (run.pr_number && !AUTO.includes(canonicalStage(run.stage)))) return "ship";
    if (run.issue_number) {
      // Server-driven: the run's own stage, or the first not-yet-passed one.
      const stage = canonicalStage(run.stage);
      if (AUTO.includes(stage)) return stage;
      if (stage === "pr") return "ship";
      return "plan";
    }
    if (run.pending_ask && run.pending_ask.call_id) return "clarify";
    // A replayed/finalizing client card is only a fallback when the server
    // could not determine approval state.  `pending_ask_known: true` with no
    // pending call is authoritative; letting stale client state win here
    // leaves the completed Clarify panel open and hides Ratify in a collapsed
    // Contract panel.
    if (run.pending_ask_known !== true
        && typeof askUserHasPendingCard === "function"
        && askUserHasPendingCard()) {
      return "clarify";
    }
    if ((run.requirements || []).length && !(run.questions || []).length) return "contract";
    if ((run.questions || []).length) return "clarify";
    return "describe";
  }

  /** Whether one intake disclosure is open. Pure for the Node vectors. */
  function intakeGateOpen(here, gate, expanded) {
    const chosen = expanded instanceof Set ? expanded : new Set(expanded || []);
    return gate === here || (INTAKE_GATES.has(gate) && chosen.has(gate));
  }

  /** Verdict per gate: "done" | "here" | "bad" | "" (not reached).
   *
   * A stage's verdict is its LAST event, never any earlier one. A retried stage
   * accumulates events — failed, then started again — and reading them all made
   * a stage that was busy working show red from the attempt before it.
   */
  function stageVerdicts() {
    const here = activeGate();
    const idx = GATES.findIndex((g) => g.id === here);
    const out = {};
    GATES.forEach((g, i) => { out[g.id] = i < idx ? "done" : (i === idx ? "here" : ""); });
    if (run && run.merged) out.ship = "done";

    for (const stage of AUTO) {
      const ev = lastEvent(stage);
      if (!ev) continue;
      if (ev.status === "passed") out[stage] = stage === here ? "here" : "done";
      else if (ev.status === "failed") out[stage] = "bad";
      else if (ev.status === "started") out[stage] = stage === here ? "here" : out[stage];
    }
    // The run being blocked is the authority on the gate it stopped at.
    if (run && run.status === "blocked" && out[here] !== "done") out[here] = "bad";
    return out;
  }

  /* ---------- notes per gate ---------- */

  /* Runs created before Plan and Design were separate stages carry `spec`,
   * which meant the design stage. Resolving it here and nowhere else keeps the
   * strip, the notes and the clocks agreeing about what a stage is called. */
  const LEGACY_STAGES = { spec: "design", review: "deploy" };
  const canonicalStage = (s) => LEGACY_STAGES[String(s || "")] || String(s || "");

  /** Latest validation/generation/verification session for one stage.
   *
   * Events are append-only, so assigning while walking them deliberately
   * selects the newest attempt. Runs created before session provenance was
   * added can still recover their shared generation chat from the run row.
   */
  // Mirror of dev_sdlc.usable_session_id. A real session id is EITHER a legacy
  // UUID or the stamp `new_session_id()` actually mints,
  // `YYYYMMDDTHHMMSSffffff-<12 hex>`. Accepting only the UUID rejected every
  // live session, so `runSessionId()` returned "" forever, the conductor was
  // unbound on every paint, and the clarify card's Submit silently did
  // nothing -- the one place a human must act.
  const SESSION_ID_RE =
    /^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\d{8}T\d{12}-[0-9a-f]{12})$/i;

  function usableSessionId(value) {
    const sid = String(value || "").trim();
    return SESSION_ID_RE.test(sid) ? sid : "";
  }

  function runSessionId() {
    return usableSessionId(run && (run.conductor_session_id || run.session_id));
  }

  function ageLabel(ts) {
    if (!ts) return "";
    let ms;
    if (typeof ts === "number") ms = ts > 1e12 ? ts : ts * 1000;
    else ms = Date.parse(ts);
    if (!ms || Number.isNaN(ms)) return "";
    const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
    if (s < 15) return "just now";
    if (s < 60) return s + "s ago";
    if (s < 3600) return Math.floor(s / 60) + "m ago";
    if (s < 86400) return Math.floor(s / 3600) + "h ago";
    return Math.floor(s / 86400) + "d ago";
  }

  let jarvisTimer = null;
  let jarvisSid = "";
  let jarvisTitle = "";
  let jarvisTitleSid = "";
  // The conductor's RUNTIME state is known only to the live probe
  // (`/api/sessions/<sid>/live`). The 2s run poll carries a `conductor.state`
  // the web session store cannot actually know, and it was always the literal
  // "idle" -- so the poll repainted "idle" over the live probe's "needs input"
  // every two seconds and the probe put it back every four. That alternation
  // IS the flicker. Remember the last state we genuinely learned, and let a
  // pulse that carries none leave it alone.
  let jarvisState = "";
  let jarvisStateSid = "";

  function applyJarvisTitle(raw, sid) {
    const want = String(sid || runSessionId() || "");
    if (want && jarvisTitleSid && want !== jarvisTitleSid) jarvisTitle = "";
    if (want) jarvisTitleSid = want;
    const t = String(raw || "").trim();
    if (!t) return;
    const isPh = typeof isPlaceholderTitle === "function" && isPlaceholderTitle(t);
    const isDefault = typeof isDefaultSessionTitle === "function"
      ? isDefaultSessionTitle(t)
      : /^New Chat - \d{6}$/.test(t);
    if (isPh) return;
    if (isDefault && jarvisTitle) return;
    if (!isDefault) {
      jarvisTitle = t;
      if (run && run.conductor) run.conductor.title = t;
    }
    const title = $("sdlc-jarvis-title");
    if (title) title.textContent = jarvisTitle || t || "Jarvis session";
  }

  function seedJarvisSession(sid) {
    if (!sid) return;
    fetch(BASE + "/api/sessions/" + encodeURIComponent(sid), {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    }).then((r) => r.json()).then((data) => {
      if (runSessionId() !== sid) return;
      const sess = (data && data.session) || data || {};
      applyJarvisTitle(sess.title, sid);
    }).catch(() => {});
  }

  function paintJarvisCard(pulse) {
    const card = $("sdlc-jarvis");
    if (!card) return;
    const sid = (pulse && pulse.session_id) || runSessionId();
    if (!sid) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    const st = $("sdlc-jarvis-state");
    if (jarvisStateSid && jarvisStateSid !== sid) jarvisState = "";
    jarvisStateSid = sid;
    const given = String((pulse && (pulse.display || pulse.state)) || "").trim();
    if (given) jarvisState = given;
    if (st) {
      const raw = jarvisState || "idle";
      st.textContent = raw;
      const live = /stream|wait|ask|busy|run|input/i.test(raw);
      st.className = "sdlc-jarvis-state" + (live ? " is-live" : "");
    }
    applyJarvisTitle((pulse && pulse.title) || jarvisTitle, sid);
    const act = $("sdlc-jarvis-activity");
    if (act) {
      const age = ageLabel((pulse && (pulse.updated_at || pulse.heartbeat_at)) || "");
      act.textContent = age ? ("Last activity " + age) : "Last activity unknown";
    }
  }

  function syncJarvisLive() {
    const sid = runSessionId();
    if (jarvisTimer && jarvisSid === sid) return;
    if (jarvisTimer) {
      clearTimeout(jarvisTimer);
      jarvisTimer = null;
    }
    jarvisSid = sid;
    if (!sid) return;
    seedJarvisSession(sid);

    const tick = () => {
      if (runSessionId() !== sid) return;
      fetch(BASE + "/api/sessions/" + encodeURIComponent(sid) + "/live", {
        credentials: "same-origin",
        headers: { Accept: "application/json" },
      }).then((r) => r.json()).then((data) => {
        if (runSessionId() !== sid) return;
        const label = data && data.display && data.display.label;
        paintJarvisCard(Object.assign({}, (run && run.conductor) || {}, {
          session_id: sid,
          href: sessionHref(sid),
          title: jarvisTitle || ((run && run.conductor && run.conductor.title) || ""),
          state: label || (data && (data.session_state || data.status)) || "idle",
          updated_at: (data && (data.heartbeat_at || data.outcome_at
            || data.phase_changed_at)) || (run && run.conductor && run.conductor.updated_at) || "",
        }));
      }).catch(() => {}).then(() => {
        if (runSessionId() === sid) jarvisTimer = setTimeout(tick, 4000);
      });
    };
    jarvisTimer = setTimeout(tick, 200);
  }

  function stageSessionRefs(events, stage, fallbackGeneration) {
    const refs = { validation: null, generation: null, verification: null };
    let sawGeneration = null;
    for (const ev of events || []) {
      if (canonicalStage(ev.stage) !== canonicalStage(stage)) continue;
      const kind = String(ev.gate_kind || "");
      const generation = !kind && ev.status === "started";
      if (!generation && kind !== "validation" && kind !== "verification") continue;
      const role = generation ? "generation" : kind;
      if (generation) sawGeneration = ev;
      const sid = usableSessionId(ev.executor_session_id);
      if (ev.status === "gate_session_abandoned") {
        if (!sid || (refs[role] && refs[role].session_id === sid)) refs[role] = null;
        continue;
      }
      if (ev.status === "gate_decided" || ev.status === "gate_unusable") {
        // An empty later verdict must not hide a session that already started.
        // It still marks the role decided so verify A can paint green after PASS
        // even when the chat id was never captured.
        refs[role] = {
          session_id: sid || (refs[role] && refs[role].session_id) || "",
          agent: String(ev.executor_agent || (refs[role] && refs[role].agent) || ""),
          model: String(ev.executor_model || (refs[role] && refs[role].model) || ""),
          decided: ev.status === "gate_decided",
        };
        continue;
      }
      if (sid) refs[role] = {
        session_id: sid,
        agent: String(ev.executor_agent || ""),
        model: String(ev.executor_model || ""),
        decided: false,
      };
    }
    if (!refs.generation && sawGeneration && fallbackGeneration) {
      refs.generation = {
        session_id: String(fallbackGeneration),
        agent: String(sawGeneration.executor_agent || ""),
        model: String(sawGeneration.executor_model || ""),
        decided: false,
      };
    }
    return refs;
  }

  function renderStageSessions(stage) {
    const host = $("sdlc-sessions-" + stage);
    if (!host) return;
    const currentStage = canonicalStage(run && run.stage);
    const generationSid = run && currentStage === canonicalStage(stage) && (stage === "deploy"
      ? run.review_session_id
      : (run.author_session_id || run.session_id));
    const refs = stageSessionRefs((run && run.events) || [], stage, generationSid);
    const reviews = (run && run.reviews) || {};
    const testsPassed = stage === "tests" && !!(run && (
      /verify A and verify B agreed PASS/i.test(String((lastEvent("tests") || {}).note || ""))
      || run.merged
    ));
    const labels = stage === "tests"
      ? [["validation", "verify A"], ["generation", "orchestrate"], ["verification", "verify B"]]
      : [["generation", "orchestrate"]];
    const markup = labels.map(([role, label]) => {
      const ref = refs[role] || {};
      const review = reviews[role] || {};
      const sid = usableSessionId(ref.session_id) || usableSessionId(review.session_id);
      const passed = !!ref.decided
        || String(review.verdict || "").toUpperCase() === "PASS"
        || (role === "validation" && testsPassed)
        || (role === "verification" && testsPassed);
      if (!sid && !passed) {
        return `<span class="sdlc-stage-session is-pending">${label}</span>`;
      }
      const liveKey = canonicalStage(stage) + ":" + role;
      const live = sid && String(((run && run.live_role_sessions) || {})[liveKey] || "")
        === sid;
      const detail = [ref.agent, ref.model].filter(Boolean).join(" · ");
      const state = live ? " · running" : (passed ? " · passed" : "");
      const title = `${stage} ${role} session${state}${detail ? " · " + detail : ""}`;
      const cls = "sdlc-stage-session" + (live ? " is-running" : " is-done");
      if (!sid) {
        return `<span class="${cls}" title="${esc(title)}"><span>${label}</span></span>`;
      }
      return `<a class="${cls}" href="${esc(sessionHref(sid))}" title="${esc(title)}"
        aria-label="Open ${esc(title)}"><span>${label}</span>${live ? '<span class="sdlc-session-state">running</span>' : ""}</a>`;
    }).join("");
    // Polling repaints every four seconds. Replacing identical anchors would
    // discard keyboard focus while someone is opening a session.
    if (host.dataset.markup !== markup) {
      host.dataset.markup = markup;
      host.innerHTML = markup;
    }
  }

  const PROVENANCE_ONLY_STATUSES = new Set([
    "gate_session_started",
    "gate_session_abandoned",
    "gate_session_linked",
    "gate_session_recovery_checked",
  ]);

  function lifecycleEvent(events, stage) {
    let found = null;
    const wanted = canonicalStage(stage);
    for (const ev of events || []) {
      if (canonicalStage(ev.stage) === wanted
          && !PROVENANCE_ONLY_STATUSES.has(String(ev.status || ""))) found = ev;
    }
    return found;
  }

  function lastEvent(stage) {
    return lifecycleEvent((run && run.events) || [], stage);
  }

  function lastNoteEvent(stage) {
    // gate_decided / gate_unusable are audit facts for the A/B chips. They
    // must not replace the stage's passed/failed note in the strip.
    let found = null;
    const wanted = canonicalStage(stage);
    for (const ev of (run && run.events) || []) {
      if (canonicalStage(ev.stage) !== wanted) continue;
      const status = String(ev.status || "");
      if (PROVENANCE_ONLY_STATUSES.has(status)) continue;
      if (status === "gate_decided" || status === "gate_unusable") continue;
      found = ev;
    }
    return found;
  }

  function noteFor(gate, verdict) {
    if (!run) return "";
    if (gate === "describe") return run.headline || "what you want shipped";
    if (gate === "clarify") {
      const asked = (run.transcript || []).length;
      // The server's projection wins once it positively knows: a card replayed
      // from the session stream whose tool_result never arrived stays
      // "pending" in the client registry and kept this note claiming a
      // question was open on the very screen that asks for scope approval.
      const serverKnowsNone = run.pending_ask_known === true
        && !(run.pending_ask && run.pending_ask.call_id);
      if (
        !serverKnowsNone && (
          liveAskForm()
          || (typeof askUserHasPendingCard === "function" && askUserHasPendingCard())
        )
      ) {
        return asked ? asked + " answered · question on this page" : "question on this page";
      }
      if (run.status === "ready") return asked ? asked + " answered · ready to review" : "ready to review";
      return asked ? asked + " answered · preparing next step" : "checking your request";
    }
    if (gate === "contract") {
      const r = (run.requirements || []).length, c = (run.criteria || []).length;
      const counts = `${r} requirement${r === 1 ? "" : "s"} · ${c} acceptance check${c === 1 ? "" : "s"}`;
      if (run.issue_number) return counts + " · approved";
      return r || c ? counts : "agree on the scope and delivery";
    }
    if (gate === "ship") {
      if (run.merged) return "merged" + (run.deploy_status && run.deploy_status !== "skipped" ? " · deploy launched" : "");
      if (run.pr_number) return `PR #${run.pr_number} · ${runPresentation(run).label}`;
      return "waiting for deploy readiness to pass";
    }
    const ev = lastNoteEvent(gate);
    if (!ev) return verdict === "here" ? "working…" : "";
    if (ev.status === "started" && verdict === "here") return "working…";
    return ev.note || (ev.status === "passed" ? "passed" : ev.status);
  }

  /** Light the lifecycle phase that owns the gate the run is on. */
  function paintPhases() {
    if (!doc) return;
    const here = activeGate();
    let live = null;
    doc.querySelectorAll(".sdlc-phase").forEach((li) => {
      const gates = String(li.dataset.gates || "").split(/\s+/).filter(Boolean);
      const on = gates.includes(here);
      li.classList.toggle("here", on);
      if (on) live = li;
    });
    const label = $("sdlc-phase-now");
    if (label) {
      const n = live && live.dataset.phase;
      const name = live && live.querySelector(".sdlc-phase-name");
      label.textContent = n ? `Phase ${n} · ${name ? name.textContent : ""}` : "Phase \u2192 gate";
    }
  }

  function pad2(n) {
    return String(Math.max(0, n | 0)).padStart(2, "0");
  }

  function formatClockSpan(iso, towardNow) {
    if (!iso) return "";
    const ts = Date.parse(iso);
    if (!Number.isFinite(ts)) return "";
    const sec = Math.max(0, Math.round((towardNow ? (Date.now() - ts) : (ts - Date.now())) / 1000));
    return pad2(Math.floor(sec / 60)) + ":" + pad2(sec % 60);
  }

  function formatTokens(n) {
    n = Number(n) || 0;
    if (n >= 1000) return (Math.round(n / 100) / 10) + "k tokens";
    return n + " tokens";
  }

  function clockFor(gate) {
    if (run && gate === run.stage && (run.status === "running" || run.watchdog_state === "warn" || run.watchdog_state === "cancelled" || run.no_progress_fused)) {
      const attempt = run.stage_attempt || 1;
      const max = run.stage_attempts_max || 2;
      const elapsed = formatClockSpan(run.stage_started_at, true) || "00:00";
      const due = formatClockSpan(run.first_artifact_deadline, false);
      const tokens = formatTokens(run.stage_tokens);
      const duePart = due ? ` · first artifact due in ${due}` : "";
      return `attempt ${attempt}/${max} · ${elapsed} elapsed${duePart} · ${tokens}`;
    }
    const ev = lastEvent(gate);
    if (!ev || !ev.created_at) return "";
    const lap = ev.lap ? `lap ${ev.lap} · ` : "";
    return lap + String(ev.created_at).slice(11, 16);
  }

  function composer() {
    return (typeof window !== "undefined" && window.devComposer) || null;
  }

  /** VERIFY the deploy, once the process it restarted is answering again. */
  function verifyDeploy(tries) {
    if (!run) return Promise.resolve();
    // Must outlast the server's grace window, or the page reports failure
    // while the server still considers the deploy pending.
    const left = tries === undefined ? 40 : tries;   // 40 x 3s = 120s > 90s grace
    say("sdlc-publish-status", "Waiting for the service to come back…");
    return api("/api/dev/sdlc/runs/" + encodeURIComponent(run.run_id) + "/deploy/verify",
               { method: "POST" })
      .then((d) => {
        run = d.run || run;
        if (d.pending && left > 0) {
          return new Promise((r) => setTimeout(r, 3000)).then(() => verifyDeploy(left - 1));
        }
        say("sdlc-publish-status",
            d.verified ? "Deploy verified — the service is answering."
              : (d.unverified ? "Deployed, but this target declares no way to prove it."
                              : "Deploy did NOT verify: " + (d.detail || "")),
            d.verified ? "ok" : "bad");
        paint();
      })
      .catch(() => {
        // The web is mid-restart, which is expected: this is the process the
        // deploy just bounced. Keep asking rather than calling it a failure.
        if (left <= 0) {
          say("sdlc-publish-status",
              "Could not reach the server to verify the deploy.", "bad");
          return;
        }
        return new Promise((r) => setTimeout(r, 3000)).then(() => verifyDeploy(left - 1));
      });
  }

  /* ---------- render ---------- */

  /* Each question arrives with the model's own recommended answer. It is
   * prefilled so the common case is "read, agree, send", but it stays tagged
   * `suggested` until the operator edits it — an accepted suggestion must
   * never masquerade as something the operator actually decided.
   *
   * The rule itself is pure so it can be tested without a DOM; only the
   * painting below touches elements. */

  /** "blank" | "suggested" | "yours" for one answer box. */
  function answerState(value, suggestion) {
    const v = String(value || "").trim();
    const sug = String(suggestion || "").trim();
    if (!v) return "blank";
    return v === sug ? "suggested" : "yours";
  }

  /** How many suggested answers the operator has not looked at yet. */
  function untouchedCount(questions, touchedIds) {
    const seen = touchedIds instanceof Set ? touchedIds : new Set(touchedIds || []);
    return (questions || [])
      .filter((q) => String(q && q.suggestion || "").trim() && !seen.has(q && q.id))
      .length;
  }

  /** What one answer box contributes to the /answers payload, or null. */
  function answerPayload(question, value, suggestion) {
    const state = answerState(value, suggestion);
    if (state === "blank") return null;
    return {
      question: question,
      answer: String(value).trim(),
      source: state === "suggested" ? "suggested" : "operator",
    };
  }

  function markSuggestion(ta) {
    const row = ta.closest(".sdlc-q");
    if (!row) return;
    const state = answerState(ta.value, ta.dataset.suggestion);
    row.dataset.state = state;
    row.classList.toggle("answered", state !== "blank");
    const tag = row.querySelector(".sdlc-q-state");
    if (tag) tag.textContent = state === "suggested" ? "suggested" : (state === "yours" ? "yours" : "");
    const qid = row.dataset.qid;
    if (qid) {
      if (state === "suggested") touched.delete(qid); else touched.add(qid);
    }
    // Update the gate note in place: a full repaint would wipe what is being typed.
    const note = $("sdlc-note-clarify");
    if (note) note.textContent = noteFor("clarify", "here");
  }

  function liveAskForm() {
    const host = $("sdlc-ask-host");
    return host && host.querySelector(".ask-user-form:not(.ask-user-locked)");
  }

  function questionKey(questions) {
    return JSON.stringify((questions || []).map((q) => [q.id, q.text || q.question, q.suggestion]));
  }

  function paintQuestionRows(questions) {
    const qbox = $("sdlc-questions");
    if (!qbox) return;
    const key = questionKey(questions);
    if (qbox.dataset.qids === key && qbox.querySelector("textarea")) {
      qbox.hidden = false;
      return;
    }
    qbox.dataset.qids = key;
    qbox.innerHTML = (questions || []).map((q, i) => {
      const text = String((q && (q.text || q.question)) || "");
      const why = String((q && q.why) || "");
      const sug = String((q && q.suggestion) || "");
      const qid = String((q && q.id) || "");
      return `<div class="sdlc-q" data-qid="${esc(qid)}" data-state="${sug ? "suggested" : "blank"}">
        <span class="sdlc-qn">${i + 1}</span>
        <div>
          <label for="sdlc-answer-${i}">${esc(text)}</label>
          ${why ? `<span class="sdlc-q-why">${esc(why)}</span>` : ""}
          <div class="sdlc-a-wrap">
            <textarea id="sdlc-answer-${i}" rows="3" data-question="${esc(text)}" data-suggestion="${esc(sug)}">${esc(sug)}</textarea>
            <span class="sdlc-q-state">${sug ? "suggested" : ""}</span>
          </div>
        </div>
      </div>`;
    }).join("");
    qbox.querySelectorAll("textarea").forEach((ta) => {
      ta.addEventListener("input", () => markSuggestion(ta));
      markSuggestion(ta);
    });
    qbox.hidden = false;
  }

  /** Retire ask cards the gateway no longer has a waiter for.
   *
   * Dev replays the conductor stream from offset 0, so every historical
   * AskUserQuestion tool_call is re-mounted -- including ones answered long
   * ago. They came back UNLOCKED and looked live, and submitting one is a
   * silent no-op because the orchestrator's waiter for that call_id is gone.
   * The run projection knows which call_id is actually open; anything else is
   * history and must not accept input.
   */
  function reconcileAskCards() {
    // Only when the SERVER positively knew. `pending_ask_known === false` means
    // the gateway could not be reached, and retiring cards on that would lock
    // a question the operator still has to answer.
    if (!run || run.pending_ask_known !== true) return;
    const live = String((run.pending_ask && run.pending_ask.call_id) || "");
    ensureAskCard(live);
    document.querySelectorAll(".ask-user-card").forEach((card) => {
      const id = String(card.dataset.callId || card.getAttribute("data-call-id") || "");
      if (live && id === live) return;
      const form = card.querySelector(".ask-user-form:not(.ask-user-locked)");
      if (!form) return;
      if (typeof _lockAskUserForm === "function") _lockAskUserForm(form);
      else form.classList.add("ask-user-locked");
    });
  }

  /**
   * Mount the open question if it is NOT already on the page.
   *
   * Cards were only ever created from a live `tool_call` stream event, so
   * missing that one frame -- a navigation, an SSE reconnect, a conductor that
   * spends minutes exploring before it asks -- left the operator staring at
   * "waiting on chat" while the server knew perfectly well which question was
   * open. Observed on a real repo: the run sat in `intake` with a valid
   * `pending_ask` the Dev tab never rendered, so the one human-in-the-loop
   * step was unreachable from this tab.
   *
   * The run projection is authoritative and already gated on
   * `pending_ask_known`, so it can seed the card the stream did not.
   */
  function ensureAskCard(callId) {
    if (!callId || typeof renderAskUserCard !== "function") return;
    const selector = `.ask-user-card[data-call-id="${CSS.escape(callId)}"]`;
    const existing = document.querySelector(selector);
    // Presence is not enough.  A transient reconciliation can lock the live
    // successor while the approvals projection is between rounds.  Once that
    // happens renderAskUserCard's registry normally treats the connected node
    // as complete and never repairs it.  The server is authoritative here: if
    // this call is still pending, its form must be interactive.
    if (existing && existing.querySelector(".ask-user-form:not(.ask-user-locked)")) return;
    if (existing) {
      if (typeof state !== "undefined" && state.askUserCards) {
        delete state.askUserCards[callId];
      }
      existing.remove();
    }
    const ask = (run && run.pending_ask) || {};
    const questions = ask.questions;
    if (!Array.isArray(questions) || !questions.length) return;
    // Bind first: a card created against no session cannot be submitted.
    bindAskSession();
    renderAskUserCard(callId, { questions });
  }

  function renderQuestions() {
    reconcileAskCards();
    const host = $("sdlc-ask-host");
    const qbox = $("sdlc-questions");
    const send = $("sdlc-answer");
    const bar = $("sdlc-suggestion-bar");
    const clear = $("sdlc-clear-suggestions");
    const intake = !!(run && !run.issue_number && ["intake", "ready"].includes(run.status));
    if (qbox) {
      qbox.hidden = true;
    }
    if (send) send.hidden = true;
    if (bar) bar.hidden = true;
    if (clear) clear.hidden = true;
    if (!host) return;
    if (liveAskForm()) return;
    const fallbackQuestions = (run && run.questions) || [];
    if (intake && fallbackQuestions.length) {
      // The deterministic clarifier can return a question before the Jarvis
      // chat session emits an AskUserQuestion call. Hiding this fallback left
      // a perfectly valid question only in JSON while the page said it was
      // waiting on chat forever. It remains one question per round; the next
      // list replaces this one only after the answer POST completes.
      host.querySelectorAll(".sdlc-status").forEach((node) => node.remove());
      paintQuestionRows(fallbackQuestions);
      if (send) send.hidden = false;
      if (bar) bar.hidden = false;
      if (clear) clear.hidden = !fallbackQuestions.some((q) => q && q.suggestion);
      return;
    }
    if (qbox) { qbox.innerHTML = ""; delete qbox.dataset.qids; }
    if (intake && !host.querySelector(".ask-user-form") && !host.querySelector(".sdlc-status")) {
      host.innerHTML = '<p class="sdlc-status">Waiting for the chat question…</p>';
    }
    if (run && run.issue_number && !host.querySelector(".ask-user-form")) {
      host.innerHTML = '<p class="sdlc-status">Nothing left to ask.</p>';
    }
  }

  function renderTranscript() {
    const host = $("sdlc-transcript");
    if (!host) return;
    const transcript = (run && run.transcript) || [];
    host.innerHTML = transcript.map((entry) => {
      const source = entry.source === "suggested" ? "accepted suggestion" : "operator";
      return `<li class="sdlc-transcript-turn">
        <strong>${esc(entry.question || "Question")}</strong>
        <span>${esc(entry.answer || "")}</span>
        <span class="sdlc-transcript-source">${source}</span>
      </li>`;
    }).join("");
  }

  function renderDescription() {
    const asking = liveAskForm();
    const intakeOpen = !!run && ["intake", "ready"].includes(run.status)
      && !run.issue_number;
    // Keep the original describe frozen only after the contract is issued, or
    // while a live question still owns the turn. After Stop the operator must
    // be able to type a follow-up.
    const frozen = !!(run && run.issue_number) || asking || pendingActions.size > 0;
    const kind = $("sdlc-kind");
    const box = composer();
    if (run && run.issue_number) {
      if (box) box.setText(run.detail || "");
      if (kind) kind.value = run.kind || "feature";
    } else if (run && kind) {
      kind.value = run.kind || "feature";
    }
    if (box) box.setFrozen(frozen);
    const input = document.getElementById("message-input");
    if (input && intakeOpen && !asking) {
      input.placeholder = "Message Jarvis — continue or redirect this change.";
    } else if (input && !run) {
      input.placeholder = "What should change, and how will you know it works? Attach a file or use your voice.";
    }
    if (kind) kind.disabled = !!(run && run.issue_number);
    const start = $("sdlc-start-submit");
    if (start) {
      start.disabled = !!frozen;
      if (!pendingActions.has("sdlc-start-submit")) start.textContent = intakeOpen ? "Send clarification" : "Clarify this change";
    }
    if (!frozen && typeof updateSendState === "function") updateSendState();
  }

  function renderIntakeControls() {
    // A ready contract can deliberately be reopened through "Anything else";
    // once the issue exists, the transcript is history and therefore read-only.
    const open = !!run && ["intake", "ready"].includes(run.status);
    const note = $("sdlc-note");
    if (note) note.readOnly = !open;
    const answer = $("sdlc-answer");
    if (answer) answer.disabled = !open;
    const clear = $("sdlc-clear-suggestions");
    if (clear) clear.disabled = !open;
    doc.querySelectorAll("#sdlc-questions textarea").forEach((ta) => {
      ta.readOnly = !open;
    });
  }

  function renderContract() {
    const frozen = !!(run && run.issue_number);
    for (const [id, key] of [["sdlc-requirements", "requirements"], ["sdlc-criteria", "criteria"]]) {
      const el = $(id);
      if (!el) continue;
      if (!el.dataset.dirty) el.value = ((run && run[key]) || []).join("\n");
      el.readOnly = frozen;
    }
    const btn = $("sdlc-ratify");
    if (btn) btn.disabled = frozen || pendingActions.has("sdlc-ratify")
      || !!(run && run.status === "ready" && run.intake_quiescent !== true);
    const branch = $("sdlc-plan-branch");
    if (branch) {
      if (!branch.dataset.dirty) branch.value = (run && run.branch) || "";
      branch.readOnly = frozen;
    }
    const publish = $("sdlc-publish-mode");
    if (publish) {
      if (frozen) publish.value = run.publish_mode || "merge";
      else if (!publish.dataset.chosen) publish.value = "pr";
      publish.disabled = frozen;
    }
    const target = $("sdlc-plan-deploy-target");
    if (target) {
      if (frozen) target.value = run.deploy_target || "";
      target.disabled = frozen || (publish && publish.value !== "merge");
      if (!frozen && target.disabled) target.value = "";
    }
    const automatic = publish && publish.value === "merge";
    const deploying = automatic && target && target.value && target.value !== "none";
    say("sdlc-delivery-summary", automatic
      ? "After checks pass, merge automatically" + (deploying ? " and deploy to the selected target." : ". No automatic deployment.")
      : "Delivery stops at a reviewed pull request. You control when it merges.");
    say("sdlc-merge-policy", automatic ? "automatic after checks pass" : "merge when you choose");
  }

  function renderLog() {
    const host = $("sdlc-log");
    if (!host) return;
    const events = (run && run.events) || [];
    host.innerHTML = events.length ? events.map((e) => {
      const cls = e.status === "passed" ? "ok" : (e.status === "failed" ? "bad" : "run");
      return `<li class="sdlc-ev ${cls}">
        <span class="sdlc-ev-tag">${esc(e.tag || e.stage)}</span>
        <span class="sdlc-ev-status">${esc(e.status)}</span>
        ${e.note ? `<span class="sdlc-ev-note">${esc(e.note)}</span>` : ""}
        <time>${esc(String(e.created_at || "").slice(11, 19))}</time>
      </li>`;
    }).join("") : '<li class="sdlc-status">No stage has reported yet.</li>';
  }

  function renderRuns() {
    const host = $("sdlc-runs");
    if (!host) return;
    say("sdlc-runs-count", String(runs.length));
    if (!runs.length) {
      delete host.dataset.markup;
      host.innerHTML = '<p class="sdlc-empty">No runs yet. Describe a change to start one.</p>';
      return;
    }
    const markup = runs.map((r) => {
      const cls = r.archived ? "is-archived"
        : (r.status === "blocked" ? "is-bad"
        : (r.status === "done" ? "is-done" : (r.status === "running" ? "is-running" : "")));
      const where = r.archived ? "archived"
        : (r.merged ? "merged" : (r.pr_number ? `PR #${r.pr_number}` : (r.stage || r.status)));
      const sid = usableSessionId(r.conductor_session_id || r.session_id);
      const chat = sid
        ? `<a class="sdlc-run-session" href="${esc(sessionHref(sid))}" title="Open Jarvis">Jarvis</a>`
        : "";
      return `<div class="sdlc-run-row">
        <button type="button" class="sdlc-run-card ${cls}${run && r.run_id === run.run_id ? " active" : ""}"
                data-run="${esc(r.run_id)}" aria-pressed="${!!(run && r.run_id === run.run_id)}">
          <span class="sdlc-run-dot" aria-hidden="true"></span>
          <span>
            <span class="sdlc-run-object">${esc(r.headline)}</span>
            <span class="sdlc-run-meta">${esc(r.kind)} · ${esc(where)}${
              r.issue_number ? " · #" + esc(r.issue_number) : ""}</span>
          </span>
        </button>
        ${chat}
      </div>`;
    }).join("");
    if (host.dataset.markup !== markup) { host.dataset.markup = markup; host.innerHTML = markup; }
    // Polling rebuilds this list. Pinning the active card into view yanked
    // the window back to Runs and made the rest of the page unreadable.
  }

  function paintGateWarning() {
    // A run can look green and say "PR ready" while a gate advanced a stage
    // without deciding anything. The backend refuses the merge and the deploy,
    // but the reason lived only in the collapsed raw log, so the refusal read
    // as a bug rather than as the gate doing its job.
    // The backend distinguishes "a gate once failed here" from "the thing
    // about to merge is unreviewed". Reading only the permanent counter said
    // "merge and deploy are blocked" for a run that had since been reviewed
    // against the merge commit.
    const n = Number((run && run.gate_unusable_count) || 0);
    const open = ((run && run.unreviewed_stages) || []);
    let el = $("sdlc-gate-warning");
    if (!n) { if (el) el.remove(); return; }
    if (!el) {
      const host = $("sdlc-log");
      if (!host || !host.parentNode) return;
      el = document.createElement("p");
      el.id = "sdlc-gate-warning";
      el.className = "sdlc-warn";
      host.parentNode.insertBefore(el, host);
    }
    if (open.length) {
      el.className = "sdlc-warn";
      el.textContent = `Never reviewed against the commit about to merge: `
        + `${open.join(", ")}. Merge and deploy are blocked until those stages `
        + `are re-run, or overridden deliberately.`;
    } else {
      el.className = "sdlc-warn sdlc-warn-settled";
      el.textContent = `${n} gate answer${n === 1 ? "" : "s"} decided nothing `
        + `earlier in this run. Every affected stage has since been reviewed `
        + `against the current commit, so merge and deploy are not blocked; `
        + `the history stays on the record.`;
    }
  }

  function paintKindBadge() {
    const el = $("sdlc-run-kind");
    if (!el) return;
    if (!run) {
      el.hidden = true;
      return;
    }
    const kind = run.kind || "feature";
    el.hidden = false;
    el.className = "sdlc-ref sdlc-kind-badge " + (kind === "fix" ? "is-fix" : "is-feature");
    el.textContent = kind === "fix" ? "Bug Fix" : "Feature";
  }

  function paintVerificationSummary() {
    const box = $("sdlc-verification-summary");
    if (!box) return;
    if (!run || !run.issue_number) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    const events = run.events || [];
    const testEv = events.slice().reverse().find((e) => e.stage === "tests");
    const deployEv = events.slice().reverse().find((e) => e.stage === "deploy");
    const testsPassed = (testEv && testEv.status === "passed") || run.status === "done";
    const testsRunning = testEv && testEv.status === "running";
    const reviewPassed = (deployEv && deployEv.status === "passed") || (run.status === "done" && !((run.unreviewed_stages || []).length));
    const allClear = testsPassed && reviewPassed;

    const statusBadge = $("sdlc-vsummary-status");
    if (statusBadge) {
      statusBadge.textContent = allClear ? "Verified & Ready" : (run.status === "blocked" ? "Blockers present" : "Verification in progress");
      statusBadge.className = "sdlc-vsummary-badge " + (allClear ? "" : (run.status === "blocked" ? "is-blocked" : "is-pending"));
    }

    const setBadge = (id, subEl, ok, text, iconChar) => {
      const el = $(id);
      if (!el) return;
      el.className = "sdlc-vbadge " + (ok ? "" : "is-pending");
      const icon = el.querySelector(".sdlc-vicon");
      if (icon) icon.textContent = iconChar || (ok ? "✓" : "○");
      const sub = $(subEl);
      if (sub) sub.textContent = text;
    };

    setBadge("sdlc-vbadge-tests", "sdlc-vsub-tests", testsPassed,
      testsPassed ? ((testEv && testEv.note) || "All unit & verification checks passed") : (testsRunning ? "Tests running…" : "Pending execution"));
    setBadge("sdlc-vbadge-review-a", "sdlc-vsub-review-a", reviewPassed,
      reviewPassed ? "Architectural verification approved" : "Review pending");
    setBadge("sdlc-vbadge-review-b", "sdlc-vsub-review-b", reviewPassed,
      reviewPassed ? "Safety & regression review approved" : "Review pending");
    const criteriaCount = (run.criteria || []).length;
    setBadge("sdlc-vbadge-criteria", "sdlc-vsub-criteria", testsPassed || reviewPassed,
      (testsPassed || reviewPassed) ? `${criteriaCount} acceptance check${criteriaCount === 1 ? "" : "s"} satisfied` : `${criteriaCount} criteria defined`);
  }

  function paint() {
    const here = activeGate();
    const marks = stageVerdicts();
    const app = $("dev-app");
    if (app) app.classList.toggle("has-run", !!run);
    paintOverview();
    paintPhases();
    paintGateWarning();
    whoLabel();
    paintKindBadge();
    paintVerificationSummary();

    for (const gate of GATES) {
      const li = document.querySelector(`.sdlc-gate[data-gate="${gate.id}"]`);
      if (!li) continue;
      const reached = marks[gate.id] !== "" || gate.id === "describe";
      // Automatic gates only appear once the contract is frozen; the ship gate
      // only once there is something to ship or deploy readiness has passed.
      const visible = gate.id === "describe"
        || (gate.id === "clarify" && !!run && (
          (run.questions || []).length
          || (run.transcript || []).length
          || (["intake", "ready"].includes(run.status) && !run.issue_number)
        ))
        || (gate.id === "contract" && !!run && ((run.requirements || []).length || run.issue_number))
        || (gate.auto && !!run && !!run.issue_number)
        || (gate.id === "ship" && !!run && !!run.issue_number);
      li.hidden = !visible;
      li.classList.toggle("done", marks[gate.id] === "done");
      li.classList.toggle("here", marks[gate.id] === "here");
      li.classList.toggle("bad", marks[gate.id] === "bad");
      const isOpen = intakeGateOpen(here, gate.id, expandedIntake);
      li.classList.toggle("collapsed", !isOpen);
      li.classList.remove("is-last");
      void reached;

      const toggle = li.querySelector("[data-intake-toggle]");
      if (toggle) {
        toggle.setAttribute("aria-expanded", String(isOpen));
        // The live panel must stay open; completed intake panels are optional
        // disclosures the operator can reopen and close.
        toggle.disabled = gate.id === here;
      }

      const note = $("sdlc-note-" + gate.id);
      if (note) note.textContent = noteFor(gate.id, marks[gate.id]);
      const clock = $("sdlc-clock-" + gate.id);
      if (clock) {
        clock.textContent = gate.auto ? clockFor(gate.id) : "";
        const live = !!(run && gate.id === run.stage);
        clock.classList.toggle("is-warn", live && run.watchdog_state === "warn");
        clock.classList.toggle("is-cancelled", live && (run.watchdog_state === "cancelled" || run.no_progress_fused));
        clock.title = (live && run.watchdog_note) || "";
      }
      if (gate.auto) renderStageSessions(gate.id);
      const ev = $("sdlc-ev-" + gate.id);
      if (ev) {
        const last = lastEvent(gate.id);
        const eviText = last && last.status === "failed" ? (last.note || "") : "";
        ev.hidden = !eviText || gate.id !== here;
        ev.textContent = eviText;
      }
    }

    const shown = Array.from(document.querySelectorAll(".sdlc-gate")).filter((li) => !li.hidden);
    if (shown.length) shown[shown.length - 1].classList.add("is-last");
    const foot = $("sdlc-foot");
    if (foot) foot.hidden = !run;

    // Header.
    const title = $("sdlc-run-title");
    if (title) title.textContent = run ? (run.headline || run.run_id) : "New change";
    // With nothing selected the page already IS the new change: a second
    // "New change" button and a state badge beside that title only invite a
    // click that does nothing.
    const fresh = $("sdlc-new");
    if (fresh) fresh.hidden = !run;
    const state = $("sdlc-run-state");
    if (state) state.hidden = !run;
    const sid = bindAskSession();
    ["sdlc-run-chat", "sdlc-ask-chat", "sdlc-jarvis-link"].forEach((id) => {
      const chat = $(id);
      if (!chat) return;
      chat.hidden = !sid;
      chat.href = sid ? sessionHref(sid) : "#";
      chat.textContent = "Open chat";
    });
    paintJarvisCard(run && run.conductor);
    syncJarvisLive();
    const issue = $("sdlc-run-issue");
    if (issue) {
      issue.hidden = !(run && run.issue_url);
      issue.href = (run && run.issue_url) || "#";
      issue.textContent = run && run.issue_number ? "issue #" + run.issue_number : "";
    }
    const pr = $("sdlc-run-pr");
    if (pr) {
      pr.hidden = !(run && run.pr_url);
      pr.href = (run && run.pr_url) || "#";
      pr.textContent = run && run.pr_number ? "PR #" + run.pr_number : "";
    }
    const branch = $("sdlc-run-branch");
    if (branch) {
      branch.hidden = !(run && run.branch);
      branch.textContent = (run && run.branch) || "";
      // The worktree is the isolation guarantee; show where the work happens.
      branch.title = run && run.worktree ? "worktree: " + run.worktree : "";
    }
    const wt = $("sdlc-run-worktree");
    if (wt) {
      wt.hidden = !(run && run.worktree);
      wt.textContent = (run && run.worktree) || "";
    }

    renderDescription();
    renderTranscript();
    adoptAskSession(sid);
    renderQuestions();
    if (here === "clarify" && run && focusedClarify !== run.run_id) {
      focusedClarify = run.run_id;
      const qhost = $("sdlc-ask-host") || $("sdlc-questions");
      if (qhost && qhost.scrollIntoView) qhost.scrollIntoView({ block: "center" });
    }
    syncAskStream();
    renderIntakeControls();
    renderContract();
    renderLog();

    // Ship controls.
    const merge = $("sdlc-merge");
    if (merge) merge.disabled = !(run && run.pr_number) || !!(run && (run.merged || run.archived));
    const archive = $("sdlc-archive");
    if (archive) {
      archive.hidden = !(run && run.run_id);
      archive.disabled = !(run && run.run_id);
      archive.textContent = "Delete";
    }
    const arm = $("sdlc-arm");
    const deploy = $("sdlc-deploy");
    if (deploy) deploy.disabled = !(run && run.merged) || !(arm && arm.checked);
    if (arm) arm.disabled = !(run && run.merged);
    const detail = $("sdlc-deploy-detail");
    const sel = $("sdlc-deploy-target");
    if (detail && sel) {
      const opt = sel.selectedOptions && sel.selectedOptions[0];
      detail.textContent = opt ? (opt.title || "") : "";
    }

    const fused = !!(run && run.no_progress_fused);
    const fuse = $("sdlc-fuse");
    const fuseReason = $("sdlc-fuse-reason");
    if (fuse) {
      fuse.hidden = !fused;
      if (fuseReason) {
        fuseReason.textContent = fused
          ? ((run.watchdog_note || run.last_error || "")
            + " Edit the scope, raise the stage budget and resume, or abort.")
          : "";
      }
    }
    const resume = $("sdlc-resume");
    if (resume) {
      // Blocked is not the only stuck state: a run whose driver died with the
      // web process sits in `running` with nothing advancing it.
      const settled = !run || !run.issue_number
        || ["done", "aborted"].includes(run.status) || run.merged;
      resume.hidden = settled || !!run.driver_active || fused;
      resume.textContent = run && run.status === "blocked" ? "Resume" : "Restart the stage";
    }
    const abort = $("sdlc-abort");
    if (abort) abort.hidden = fused || !(run && run.issue_number && run.status !== "done" && run.status !== "aborted");
    if (run && run.last_error) say("sdlc-publish-status", run.last_error, "bad");
    ["dev-repo", "sdlc-new"].forEach((id) => { if ($(id)) $(id).disabled = pendingActions.size > 0; });
    pendingActions.forEach((id) => { if ($(id)) $(id).disabled = true; });
  }

  /* ---------- same session SSE + same ask card as Chat ---------- */

  let askSource = null;
  let askSid = "";
  let askEpoch = 0;

  function bindAskSession() {
    const sid = runSessionId();
    const aliases = (run && Array.isArray(run.conductor_aliases))
      ? run.conductor_aliases : [];
    if (typeof bindAskUserSession === "function") {
      return bindAskUserSession(sid, aliases);
    }
    if (typeof state === "undefined") return sid;
    if (sid) state.currentSessionId = sid;
    else state.currentSessionId = null;
    state.askUserSessionAliases = aliases.slice();
    return sid;
  }

  function adoptAskSession(sid) {
    const next = String(sid || "").trim();
    if (askSid === next && next) return;
    // Run switch: close the old stream and clear Dev's one-card host
    // before the session SSE replays the live ask tool_call.
    closeAskStream();
    if (next) {
      if (typeof resetAskUserCards === "function") resetAskUserCards();
    } else if (typeof unbindAskUserSession === "function") {
      unbindAskUserSession();
    }
  }

  function closeAskStream() {
    if (typeof closeAskUserStream === "function") closeAskUserStream();
    if (askSource) {
      try { askSource.close(); } catch (_e) { /* ignore */ }
    }
    askSource = null;
    askSid = "";
  }

  function askEventArgs(event) {
    let args = event && (event.arguments || event.args);
    if (typeof args === "string") {
      try { args = JSON.parse(args); } catch (_e) { args = {}; }
    }
    return args && typeof args === "object" ? args : {};
  }

  function handleSessionAskEvent(event) {
    const liveSid = typeof state !== "undefined" ? String(state.currentSessionId || "") : "";
    if (askSid && liveSid && askSid !== liveSid) return;
    if (!event || typeof event !== "object") return;
    const kind = String(event.type || event.event || "");
    if (kind === "header_update") {
      const patch = event.patch || {};
      if (patch.title) applyJarvisTitle(patch.title, askSid || runSessionId());
    }
    const name = event.name || "";
    const callId = event.call_id || "";
    if (
      kind === "tool_call"
      && typeof _isAskUserTool === "function"
      && _isAskUserTool(name)
      && callId
      && typeof renderAskUserCard === "function"
    ) {
      renderAskUserCard(callId, askEventArgs(event));
    }
    if (kind === "tool_result_meta" && callId
        && typeof _finalizeAskUserCard === "function") {
      // Empty meta locks a still-pending card. Only a failed meta (timeout)
      // is a real close. The body event, when it exists, carries the answer.
      if (event.ok === false) _finalizeAskUserCard(callId, "", false);
      renderQuestions();
      return;
    }
    if (
      kind === "tool_result"
      && callId
      && typeof _finalizeAskUserCard === "function"
    ) {
      const content = typeof event.content === "string" ? event.content : "";
      if (content || event.ok === false) {
        _finalizeAskUserCard(callId, content, event.ok);
      }
    }
    renderQuestions();
  }

  function openAskStream(sid) {
    if (!sid) {
      askEpoch += 1;
      closeAskStream();
      if (typeof unbindAskUserSession === "function") unbindAskUserSession();
      else if (typeof state !== "undefined") state.currentSessionId = null;
      return;
    }
    if (
      askSid === sid
      && (
        (typeof askUserStreamOpen === "function" && askUserStreamOpen(sid))
        || askSource
      )
    ) {
      return;
    }
    const epoch = ++askEpoch;
    closeAskStream();
    bindAskSession();
    askSid = sid;
    fetch(BASE + "/api/sessions/" + encodeURIComponent(sid), {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    }).then((r) => r.json()).then((data) => {
      if (epoch !== askEpoch || askSid !== sid) return;
      const sess = (data && data.session) || data || {};
      applyJarvisTitle(sess.title, sid);
      // Replay from 0 so the pending AskUserQuestion tool_call arrives on
      // this session SSE instead of a REST snapshot remount.
      attachAskSource(sid, 0, epoch);
    }).catch(() => {
      if (epoch !== askEpoch || askSid !== sid) return;
      attachAskSource(sid, 0, epoch);
    });
  }

  function attachAskSource(sid, from, epoch) {
    if (epoch !== askEpoch || askSid !== sid) return;
    if (typeof openAskUserStream === "function") {
      openAskUserStream(sid, from, handleSessionAskEvent);
      return;
    }
    const url = typeof sessionStreamUrl === "function"
      ? sessionStreamUrl(sid, from)
      : (BASE + "/api/sessions/" + encodeURIComponent(sid)
        + "/stream?mode=events&from=" + encodeURIComponent(from || 0)
        + "&activity=stream&reasoning=stream&tools=stream");
    let es;
    try {
      es = typeof createAskEventSource === "function"
        ? createAskEventSource(url)
        : new EventSource(url);
    } catch (_e) {
      return;
    }
    askSource = es;
    es.addEventListener("log", (e) => {
      if (askSid !== sid || askEpoch !== epoch || askSource !== es) return;
      let frame;
      try { frame = JSON.parse(e.data); } catch (_e) { return; }
      const ev = frame && (frame.event || frame);
      if (ev) handleSessionAskEvent(ev);
    });
  }

  function syncAskStream() {
    const sid = run && String(run.conductor_session_id || run.session_id || "").trim();
    if (sid) openAskStream(sid);
    else openAskStream("");
  }

  /* ---------- data ---------- */

  function schedulePoll() {
    if (poll) clearTimeout(poll);
    if (!run || !run.run_id) return;
    const intake = ["intake", "ready"].includes(run.status) && !run.issue_number;
    if (!intake && !run.issue_number) return;
    const settled = run.status === "done" || run.status === "aborted" || run.merged;
    poll = setTimeout(refresh, intake ? 2000 : (settled ? 20000 : 4000));
  }

  function refresh() {
    if (!run || !run.run_id || selecting) return Promise.resolve();
    if (pendingActions.size) { schedulePoll(); return Promise.resolve(); }
    const id = run.run_id, selection = selectionSeq, seq = ++refreshSeq;
    const current = () => selection === selectionSeq && seq === refreshSeq
      && run && run.run_id === id && !pendingActions.size;
    return api("/api/dev/sdlc/runs/" + encodeURIComponent(id))
      .then((d) => {
        if (!current()) return;
        run = d.run; syncStatus(""); paint(); loadRuns();
      })
      .catch(() => { if (current()) syncStatus("Live updates are unavailable. Your input is preserved; reconnecting automatically…"); })
      .then(schedulePoll);
  }

  function wantedRunId() {
    try {
      return new URLSearchParams(window.location.search).get("run") || "";
    } catch (_) {
      return "";
    }
  }

  function rememberRun(runId) {
    try {
      const url = new URL(window.location.href);
      if (runId) url.searchParams.set("run", runId);
      else url.searchParams.delete("run");
      history.replaceState(null, "", url.pathname + url.search + url.hash);
    } catch (_) { /* ignore */ }
  }

  function applyRunRepo(next) {
    const sel = $("dev-repo");
    const repoName = String((next && next.repo) || "").trim();
    if (!sel || !repoName) return;
    if (![...sel.options].some((o) => o.value === repoName)) {
      const opt = document.createElement("option");
      opt.value = repoName;
      opt.textContent = repoName;
      sel.appendChild(opt);
    }
    try { localStorage.setItem("sharedagents-dev-repo", repoName); } catch (_) { /* ignore */ }
    if (sel.value === repoName) return;
    sel.value = repoName;
    sel.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function select(runId) {
    if (pendingActions.size || preparingStart || selecting === runId) return Promise.resolve();
    const seq = ++selectionSeq;
    refreshSeq += 1;
    selecting = runId;
    return api("/api/dev/sdlc/runs/" + encodeURIComponent(runId))
      .then((d) => {
        if (seq !== selectionSeq) return;
        selecting = "";
        const prevSid = runSessionId();
        run = d.run;
        resetIntakeFields();
        const box = composer();
        if (box) box.clear();
        const nextSid = runSessionId();
        if (prevSid !== nextSid && typeof resetAskUserCards === "function") {
          resetAskUserCards();
        }
        if (!nextSid && typeof state !== "undefined") state.currentSessionId = null;
        focusedClarify = "";
        applyRunRepo(run);
        rememberRun(run && run.run_id);
        expandedIntake.clear();
        ROLE_NAMES.forEach((role) => {
          const selected = who.roles[role];
          selected.agent = role === "generation"
            ? GENERATION_AGENT
            : reviewerAgent(run[role + "_agent"] || selected.agent, role);
          selected.model = run[role + "_model"] || "";
          selected.models = [];
        });
        whoLabel();
        ["sdlc-requirements", "sdlc-criteria"].forEach((id) => {
          const el = $(id); if (el) delete el.dataset.dirty;
        });
        try { paint(); renderRuns(); } catch (err) {
          if (doc && doc.documentElement) {
            doc.documentElement.dataset.sdlcPaint = String((err && err.message) || err);
          }
          say("sdlc-start-status", (err && err.message) || String(err), "bad");
        }
        schedulePoll();
      })
      .catch((e) => { if (seq === selectionSeq) { selecting = ""; syncStatus("Could not open this change: " + e.message); } });
  }

  function loadRuns() {
    const seq = ++runsSeq;
    return api("/api/dev/sdlc/runs?repo=" + encodeURIComponent(repo())
               + "&generation_agent=" + encodeURIComponent(who.roles.generation.agent))
      .then((d) => {
        if (seq !== runsSeq) return;
        runs = d.runs || [];
        if (runsSyncFailed) { runsSyncFailed = false; syncStatus(""); }
        // The tab's own default is the local model, so a new run never silently
        // bills a metered provider.
        const fallbacks = d.default_role_models || {};
        const agents = d.default_role_agents || {};
        ROLE_NAMES.forEach((role) => {
          who.roles[role].fallback = fallbacks[role] || "";
          if (!run && role !== "generation" && agents[role]
              && !who.roles[role].explicit) {
            who.roles[role].agent = reviewerAgent(agents[role], role);
          }
        });
        whoLabel();
        // Deploy targets belong to the REPO, so they must be repainted when the
        // repo changes. Filling only an empty select left the previous repo's
        // choices in place and hid the new one's entirely.
        const sel = $("sdlc-deploy-target");
        if (sel) {
          const keep = sel.value;
          const targets = d.deploy_targets || [];
          const markup = targets.map((t) =>
            `<option value="${esc(t.id)}" title="${esc(t.detail)}">${esc(t.label)}</option>`).join("");
          if (sel.dataset.markup !== markup) {
            sel.dataset.markup = markup;
            sel.innerHTML = markup;
            if (targets.some((t) => t.id === keep)) sel.value = keep;
          }
        }
        // The plan-phase picker: authorising the deploy BEFORE any code exists
        // is what lets the run finish without a second visit to this tab.
        const planSel = $("sdlc-plan-deploy-target");
        if (planSel) {
          const targets = d.deploy_targets || [];
          const keep = planSel.value;
          const markup = '<option value="">No automatic deployment</option>'
            + targets.map((t) =>
              `<option value="${esc(t.id)}" title="${esc(t.detail)}">${esc(t.label)}</option>`).join("");
          if (planSel.dataset.markup !== markup) {
            planSel.dataset.markup = markup;
            planSel.innerHTML = markup;
            planSel.value = targets.some((t) => t.id === keep) ? keep : "";
          }
        }
        renderRuns();
        renderContract();
        if (doc && doc.documentElement) doc.documentElement.dataset.sdlcRuns = String(runs.length);
        const want = wantedRunId();
        if (want && !selecting && (!run || run.run_id !== want)) return select(want);
      })
      .catch((err) => {
        if (seq !== runsSeq) return;
        runsSyncFailed = true;
        syncStatus("Could not refresh changes. Use Refresh to try again. " + err.message);
        if (doc && doc.documentElement) {
          doc.documentElement.dataset.sdlcRuns = "err:" + String((err && err.message) || err);
        }
      });
  }

  /* ---------- actions ---------- */

  function withBusy(id, label, fn) {
    if (pendingActions.has(id)) return busyTail;
    refreshSeq += 1;
    const btn = $(id);
    const old = btn ? btn.textContent : "";
    // Polling can reveal a later-stage control while the preceding action is
    // still settling. Dropping that click made an enabled ratify button a
    // no-op. Reserve it now and serialize it behind the active action.
    pendingActions.add(id);
    if (btn) { btn.disabled = true; btn.textContent = label; }
    const execute = () => {
      busy = true;
      paint();
      const send = $("send-btn");
      if (send) send.disabled = true;
      return Promise.resolve().then(fn)
        .catch((e) => say("sdlc-start-status", e.message, "bad"))
        .then(() => {
          pendingActions.delete(id);
          busy = false;
          const liveBtn = $(id);
          if (liveBtn) { liveBtn.disabled = false; liveBtn.textContent = old; }
          if (typeof updateSendState === "function") updateSendState();
          else if (send) send.disabled = false;
          paint();
        });
    };
    busyTail = busyTail.then(execute, execute);
    return busyTail;
  }


  /* ---------- agent & model ----------
   * Both lists come from the same places the chat tab reads them: the agent
   * roster from /api/models (one row per agent, with its description) and the
   * per-agent catalogue from the shared `/models` runtime command. Nothing is
   * duplicated here but the presentation.
   */
  const ROLE_NAMES = ["validation", "generation", "verification"];
  const ROLE_LABELS = {
    validation: "verify A", generation: "orchestrate", verification: "verify B",
  };
  const GENERATION_AGENT = "jarvis";
  const NOT_REVIEWERS = new Set([GENERATION_AGENT, "plan"]);
  const DEFAULT_REVIEWERS = { validation: "build", verification: "cursor" };
  const who = {
    active: "generation", agents: [], loading: false, trigger: null,
    roles: {
      generation: { agent: GENERATION_AGENT, model: "", fallback: "", models: [], explicit: false },
      validation: { agent: "build", model: "", fallback: "", models: [], explicit: false },
      verification: { agent: "cursor", model: "", fallback: "", models: [], explicit: false },
    },
  };
  const choice = () => who.roles[who.active];

  function reviewerAgent(name, role) {
    const agent = String(name || "").trim();
    if (agent && !NOT_REVIEWERS.has(agent)) return agent;
    return DEFAULT_REVIEWERS[role] || DEFAULT_REVIEWERS[who.active] || "build";
  }

  function lockGenerate() {
    who.roles.generation.agent = GENERATION_AGENT;
    ROLE_NAMES.forEach((role) => {
      if (role === "generation") return;
      who.roles[role].agent = reviewerAgent(who.roles[role].agent, role);
    });
  }

  function startDescribe() {
    if (preparingStart || pendingActions.size) return;
    if (!repo()) { say("sdlc-start-status", "Choose a repository first.", "bad"); $("dev-repo")?.focus(); return; }
    preparingStart = true;
    lockGenerate();
    const box = composer();
    const detail = ((box && box.text()) || "").trim();
    const start = () => {
      preparingStart = false;
      const attachments = box ? box.payload() : [];
      if (!detail && !attachments.length) {
        say("sdlc-start-status", "Describe the change, or attach a file.", "bad");
        return;
      }
      if (run && ["intake", "ready"].includes(run.status) && !run.issue_number) {
        say("sdlc-start-status", "Sending…");
        withBusy("sdlc-start-submit", "Sending…", () =>
          api("/api/dev/sdlc/runs/" + encodeURIComponent(run.run_id) + "/prompt", {
            method: "POST",
            body: JSON.stringify({ prompt: detail, attachments }),
          }).then((d) => {
            if (box) box.clear();
            say("sdlc-start-status", (d && d.error) || "Sent.", d && d.error ? "bad" : "");
            if (d && d.run) run = d.run;
            loadRuns();
            schedulePoll();
          }));
        return;
      }
      say("sdlc-start-status", "Working out what I still need to know…");
      withBusy("sdlc-start-submit", "Asking…", () =>
        api("/api/dev/sdlc/runs", {
          method: "POST",
          body: JSON.stringify({
            repo: repo(),
            detail,
            kind: ($("sdlc-kind") || {}).value || "feature",
            ...Object.fromEntries(ROLE_NAMES.flatMap((role) => {
              const gen = who.roles.generation.model || who.roles.generation.fallback;
              let model = who.roles[role].model;
              if (role !== "generation" && model && gen && model === gen) model = "";
              return [
                [role + "_agent", role === "generation" ? GENERATION_AGENT : who.roles[role].agent],
                [role + "_model", model],
              ];
            })),
            attachments,
          }),
        }).then((d) => {
          run = d.run;
          rememberRun(run && run.run_id);
          if (box) box.clear();
          say("sdlc-start-status", d.error || "", d.error ? "bad" : "");
          say("sdlc-intake-status", (run.questions || []).length
            ? "Answer what matters. Blanks get asked again." : "");
          loadRuns();
          // The clarifier answers asynchronously. Only select() and ratify
          // armed the poll, so a change started on this page never learned
          // it had become ready until the operator reloaded.
          schedulePoll();
        }));
    };
    if (box && box.waitPending) box.waitPending().then(start).catch((err) => {
      preparingStart = false; say("sdlc-start-status", "Could not prepare attachments: " + err.message, "bad");
    });
    else start();
  }
  if (typeof window !== "undefined") window.devStartDescribe = startDescribe;

  function whoLabel() {
    lockGenerate();
    ROLE_NAMES.forEach((role) => {
      const selected = who.roles[role];
      const a = $("sdlc-who-" + role + "-agent");
      if (a) a.textContent = selected.agent;
      const m = $("sdlc-who-" + role + "-model");
      if (!m) return;
      const genChosen = who.roles.generation.model || who.roles.generation.fallback;
      let chosen = selected.model || selected.fallback;
      if (role !== "generation" && chosen && genChosen && chosen === genChosen) {
        chosen = "";
      }
      const row = selected.models.find((r) => r.id === chosen);
      m.textContent = chosen ? (row ? row.title : chosen) : "agent default";
      m.classList.toggle("is-default", !selected.model);
      const chip = $("sdlc-run-role-" + role);
      if (chip) {
        chip.hidden = !run;
        const label = chip.querySelector(".sdlc-run-role-agent");
        if (label) label.textContent = selected.agent;
        chip.title = `${ROLE_LABELS[role] || role}: ${selected.agent} · ${chosen || "agent default"}`;
        const review = (run && run.reviews && run.reviews[role]) || {};
        const passed = String(review.verdict || "").toUpperCase() === "PASS"
          || String(review.status || "") === "decided";
        const events = (run && run.events) || [];
        const hasSession = events.some((ev) => {
          const kind = String(ev.gate_kind || "");
          const evRole = (!kind && ev.status === "started") ? "generation" : kind;
          return evRole === role && usableSessionId(ev.executor_session_id);
        });
        chip.classList.toggle("is-done", !!(run && (passed || hasSession || role === "generation")));
      }
    });
  }

  function loadAgents() {
    if (who.agents.length) return Promise.resolve();
    return api("/api/models").then((d) => {
      who.agents = (d.models || []).map((r) => ({
        name: String(r.name || r.label || ""),
        description: String(r.description || ""),
      })).filter((r) => r.name);
    }).catch(() => {});
  }

  function loadModels(role, force = false) {
    const selected = who.roles[role];
    who.loading = true;
    renderModels();
    return api("/api/models/discover", {
      method: "POST",
      body: JSON.stringify({ agent: selected.agent, force: !!force }),
    }).then((d) => {
      const items = d.models || [];
      selected.models = items.map((it) => {
        const id = String(it.alias || it.id || "");
        const wire = String(it.id || "");
        const backend = String(it.backend || it.provider || "");
        const context = Number(it.contextWindow || 0);
        return {
          id,
          title: String(it.display || it.title || id),
          subtitle: [wire && wire !== id ? wire : "", backend].filter(Boolean).join(" · "),
          provider: String(it.provider || backend),
          badge: context ? Math.floor(context / 1024) + "k" : "",
          isDefault: it.default === true || (!!selected.fallback && id === selected.fallback),
        };
      }).filter((r) => r.id);
    }).catch(() => { selected.models = []; })
      .then(() => { who.loading = false; renderModels(); whoLabel(); });
  }

  function renderAgents() {
    const host = $("amp-dev-agents");
    const section = $("amp-dev-agent-section");
    if (!host) return;
    if (who.active === "generation") {
      lockGenerate();
      if (section) section.hidden = true;
      host.innerHTML = "";
      return;
    }
    if (section) section.hidden = false;
    const selected = choice();
    const reviewers = who.agents.filter((a) => !NOT_REVIEWERS.has(a.name));
    const used = new Set(ROLE_NAMES.filter((r) => r !== who.active)
      .map((r) => who.roles[r].agent).concat([GENERATION_AGENT]));
    host.innerHTML = reviewers.length ? reviewers.map((a) => `
      <button type="button" class="amp-dev-agent${a.name === selected.agent ? " active" : ""}"
              role="radio" aria-checked="${a.name === selected.agent}" data-agent="${esc(a.name)}"
              ${used.has(a.name) ? 'disabled aria-disabled="true" title="Already used by another role"' : ""}>
        <span class="amp-dev-agent-name">${esc(a.name)}</span>
        ${a.description ? `<span class="amp-dev-agent-desc">${esc(a.description)}</span>` : ""}
      </button>`).join("") : '<p class="sdlc-status">No reviewer agents available.</p>';
    host.querySelectorAll("[data-agent]").forEach((b) => {
      b.addEventListener("click", () => {
        if (selected.agent === b.dataset.agent) return;
        selected.agent = b.dataset.agent;
        selected.model = "";
        selected.explicit = true;
        selected.fallback = "";
        selected.models = [];
        renderAgents();
        whoLabel();
        loadModels(who.active);
        loadRuns();
      });
    });
  }

  function renderModels() {
    const host = $("amp-dev-models");
    if (!host) return;
    if (who.loading) {
      host.innerHTML = '<p class="sdlc-status">Reading the catalogue…</p>';
      return;
    }
    const selected = choice();
    if (!selected.models.length) {
      host.innerHTML = '<p class="sdlc-status">This agent brings its own model.</p>';
      return;
    }
    const genModel = who.roles.generation.model || who.roles.generation.fallback;
    host.innerHTML = selected.models.map((m) => `
      <button type="button" class="amp-dev-model${m.id === selected.model ? " active" : ""}"
              role="radio" aria-checked="${m.id === selected.model}" data-model="${esc(m.id)}"
              ${who.active !== "generation" && genModel && m.id === genModel
                ? 'disabled aria-disabled="true" title="Orchestrate already uses this model"' : ""}>
        <span class="amp-dev-model-title">${esc(m.title)}${
          m.isDefault ? '<span class="amp-dev-pill">default</span>' : ""}</span>
        <span class="amp-dev-model-sub">${esc(m.subtitle)}${
          m.badge ? " · " + esc(m.badge) : ""}</span>
      </button>`).join("");
    host.querySelectorAll("[data-model]").forEach((b) => {
      b.addEventListener("click", () => {
        if (b.disabled) return;
        selected.model = b.dataset.model === selected.model ? "" : b.dataset.model;
        selected.explicit = true;
        renderModels();
        whoLabel();
      });
    });
  }

  function openWho(role) {
    who.active = ROLE_NAMES.includes(role) ? role : "generation";
    lockGenerate();
    const sheet = $("amp-dev"), back = $("amp-dev-backdrop");
    const trigger = who.trigger;
    if (!sheet) return;
    const title = $("amp-dev-role");
    if (title) title.textContent = ROLE_LABELS[who.active] || who.active;
    const kind = $("amp-dev-title-kind");
    if (kind) kind.textContent = who.active === "generation" ? "model" : "agent & model";
    sheet.classList.toggle("is-model-only", who.active === "generation");
    sheet.hidden = false;
    if (back) back.hidden = false;
    if (trigger) trigger.setAttribute("aria-expanded", "true");
    Promise.resolve(loadAgents()).then(() => {
      renderAgents();
      if (!choice().models.length) loadModels(who.active); else renderModels();
    });
  }

  function closeWho() {
    const sheet = $("amp-dev"), back = $("amp-dev-backdrop");
    const trigger = who.trigger;
    if (sheet) sheet.hidden = true;
    if (back) back.hidden = true;
    if (trigger) { trigger.setAttribute("aria-expanded", "false"); trigger.focus(); }
    who.trigger = null;
  }

  function bindWho() {
    doc.querySelectorAll("[data-role-picker]").forEach((trigger) => {
      trigger.addEventListener("click", () => {
        who.trigger = trigger;
        openWho(trigger.dataset.rolePicker);
      });
    });
    const close = $("amp-dev-close");
    if (close) close.addEventListener("click", closeWho);
    const back = $("amp-dev-backdrop");
    if (back) back.addEventListener("click", closeWho);
    const save = $("amp-dev-save");
    if (save) {
      save.addEventListener("click", () => {
        // On a run that already exists, the choice has to be persisted — a run
        // blocked on an unreachable model is otherwise unrecoverable.
        if (run && run.run_id && run.status !== "done" && run.status !== "aborted") {
          api("/api/dev/sdlc/runs/" + encodeURIComponent(run.run_id) + "/builder", {
            method: "POST",
            body: JSON.stringify(Object.fromEntries(ROLE_NAMES.flatMap((role) => [
              [role + "_agent", role === "generation" ? GENERATION_AGENT : who.roles[role].agent],
              [role + "_model", who.roles[role].model],
            ]))),
          }).then((d) => { run = d.run; paint(); loadRuns(); })
            .catch((e) => say("sdlc-publish-status", e.message, "bad"));
        }
        closeWho();
      });
    }
    const dflt = $("amp-dev-default");
    if (dflt) dflt.addEventListener("click", () => {
      choice().model = "";
      choice().explicit = true;
      renderModels();
      whoLabel();
    });
    const refresh = $("amp-dev-refresh");
    if (refresh) refresh.addEventListener("click", () => loadModels(who.active, true));
    if (doc) doc.addEventListener("keydown", (e) => {
      const sheet = $("amp-dev");
      if (e.key === "Escape" && sheet && !sheet.hidden) closeWho();
    });
    whoLabel();
  }

  function bind() {
    if (doc && doc.documentElement) doc.documentElement.dataset.sdlcBound = "1";
    try {
    doc.querySelectorAll("[data-intake-toggle]").forEach((toggle) => {
      toggle.addEventListener("click", () => {
        const gate = toggle.dataset.intakeToggle;
        if (!INTAKE_GATES.has(gate) || gate === activeGate()) return;
        if (expandedIntake.has(gate)) expandedIntake.delete(gate);
        else expandedIntake.add(gate);
        paint();
      });
    });

    const newBtn = $("sdlc-new");
    if (newBtn) {
      newBtn.addEventListener("click", () => {
        newChange();
        $("message-input")?.focus();
      });
    }

    const startForm = $("sdlc-start");
    if (startForm) {
      startForm.addEventListener("submit", (ev) => {
        ev.preventDefault();
        startDescribe();
      });
    }

    const clear = $("sdlc-clear-suggestions");
    if (clear) {
      clear.addEventListener("click", () => {
        document.querySelectorAll("#sdlc-questions textarea").forEach((ta) => {
          if (ta.value.trim() === (ta.dataset.suggestion || "").trim()) ta.value = "";
          markSuggestion(ta);
        });
      });
    }

    const answer = $("sdlc-answer");
    if (answer) {
      answer.addEventListener("click", () => {
        const answers = [];
        document.querySelectorAll("#sdlc-questions textarea").forEach((ta) => {
          const entry = answerPayload(ta.dataset.question, ta.value, ta.dataset.suggestion);
          if (entry) answers.push(entry);
        });
        const note = ($("sdlc-note") || {}).value || "";
        if (!answers.length && !note.trim()) {
          say("sdlc-intake-status", "Answer at least one question.", "bad"); return;
        }
        withBusy("sdlc-answer", "Thinking…", () =>
          api("/api/dev/sdlc/runs/" + encodeURIComponent(run.run_id) + "/answers", {
            method: "POST", body: JSON.stringify({ answers, note }),
          }).then((d) => {
            run = d.run;
            const n = $("sdlc-note"); if (n) n.value = "";
            say("sdlc-intake-status", (run.questions || []).length
              ? "Next question is ready." : "Scope is ready — review it below and start.",
              (run.questions || []).length ? "" : "ok");
            schedulePoll();
          }));
      });
    }

    ["sdlc-requirements", "sdlc-criteria", "sdlc-plan-branch"].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener("input", () => { el.dataset.dirty = "1"; });
    });
    const publishMode = $("sdlc-publish-mode");
    if (publishMode) publishMode.addEventListener("change", () => {
      publishMode.dataset.chosen = "1";
      renderContract();
    });
    $("sdlc-plan-deploy-target")?.addEventListener("change", renderContract);

    const ratify = $("sdlc-ratify");
    if (ratify) {
      ratify.addEventListener("click", () => {
        const requirements = lines(($("sdlc-requirements") || {}).value);
        const criteria = lines(($("sdlc-criteria") || {}).value);
        if (!requirements.length || !criteria.length) {
          say("sdlc-contract-status",
              "Add at least one requirement and one acceptance check before starting.", "bad");
          return;
        }
        const evidenceOk = criteria.every((line) => {
          const low = line.toLowerCase();
          return low.includes("given") && low.includes("when")
            && (low.includes("then") || low.includes("pass"))
            && (low.includes("fail-if") || low.includes("fail if"));
        });
        const conditionOk = requirements.every((line) => {
          const low = line.toLowerCase();
          return /shall|must|when |if |non-goal/.test(low)
            && !low.includes("fail-if") && !low.includes("fail if");
        });
        if (!conditionOk || !evidenceOk) {
          say("sdlc-contract-status",
              !conditionOk
                ? "Each requirement must be a WHEN … SHALL … statement, not a test procedure."
                : "Each acceptance check must be given / when / then / fail-if.",
              "bad");
          return;
        }
        withBusy("sdlc-ratify", "Opening the issue…", () =>
          api("/api/dev/sdlc/runs/" + encodeURIComponent(run.run_id) + "/ratify", {
            method: "POST",
            body: JSON.stringify({
              requirements,
              criteria,
              branch: (($("sdlc-plan-branch") || {}).value || "").trim(),
              publish_mode: (($("sdlc-publish-mode") || {}).value || "pr"),
              test_isolation: "docker",
              // Empty means "decide later" — the server leaves the run's
              // deploy target unset and the operator arms it after the merge.
              deploy_target: (($("sdlc-plan-deploy-target") || {}).value || ""),
            }),
          }).then((d) => {
            run = d.run;
            say("sdlc-contract-status", "Issue #" + (run.issue_number || "?") + " opened. Development is running.", "ok");
            loadRuns(); schedulePoll();
          }).catch((err) => {
            // Ratify was the ONE handler here with no catch, so a refusal was
            // an unhandled rejection: the button un-busied, the status line
            // stayed blank, and the run silently never started. The server
            // says exactly what is wrong ("evidence 9 is not
            // given/when/then/fail-if") -- show it, including the detail.
            const body = (err && err.body) || {};
            const detail = body.detail && body.detail !== body.error
              ? " — " + body.detail : "";
            say("sdlc-contract-status",
                ((err && err.message) || "could not open the issue") + detail,
                "bad");
          }));
      });
    }

    const merge = $("sdlc-merge");
    if (merge) {
      merge.addEventListener("click", () => {
        withBusy("sdlc-merge", "Merging…", () =>
          api("/api/dev/sdlc/runs/" + encodeURIComponent(run.run_id) + "/merge", {
            method: "POST",
            body: JSON.stringify({
              method: ($("sdlc-merge-method") || {}).value || "merge",
              force: !!($("sdlc-merge-force") || {}).checked,
            }),
          }).then((d) => { run = d.run; say("sdlc-publish-status", "Merged.", "ok"); loadRuns(); })
            .catch((err) => {
              // The ship gate refused. Naming every blocker is the whole point:
              // "validation failed" tells the operator nothing they can act on.
              const b = (err.body && err.body.blockers) || [];
              say("sdlc-publish-status",
                  b.length ? "Not safe to merge — " + b.join("; ") : err.message, "bad");
              const force = $("sdlc-merge-force-row");
              if (force && b.length) force.hidden = false;
              throw err;
            }));
      });
    }

    const archive = $("sdlc-archive");
    if (archive) {
      archive.addEventListener("click", () => {
        if (!run || !run.run_id) return;
        const extra = "This closes the issue, deletes the branch, and removes this run from Dev. It does not merge a PR. Failed and finished runs stay until you do this.";
        if (!window.confirm("Delete this run?\n\n" + extra)) return;
        const goneId = run.run_id;
        withBusy("sdlc-archive", "Deleting…", () =>
          api("/api/dev/sdlc/runs/" + encodeURIComponent(goneId), {
            method: "DELETE",
          }).then((d) => {
            run = null;
            rememberRun("");
            const notes = (d.notes || []).join(" · ");
            say("sdlc-publish-status", notes ? "Deleted — " + notes : "Deleted.", "ok");
            loadRuns();
          }));
      });
    }

    const arm = $("sdlc-arm");
    if (arm) arm.addEventListener("change", paint);
    const target = $("sdlc-deploy-target");
    if (target) target.addEventListener("change", paint);

    const deploy = $("sdlc-deploy");
    if (deploy) {
      deploy.addEventListener("click", () => {
        const t = ($("sdlc-deploy-target") || {}).value || "none";
        const id = encodeURIComponent(run.run_id);
        // "No deploy" restarts nothing, so it needs no arm.
        const armed = t === "none"
          ? Promise.resolve({})
          : api("/api/dev/sdlc/runs/" + id + "/arm", { method: "POST" });
        withBusy("sdlc-deploy", "Launching…", () =>
          armed.then((a) =>
            api("/api/dev/sdlc/runs/" + id + "/deploy", {
              method: "POST",
              body: JSON.stringify({ target: t, armed_token: a.armed_token || "" }),
            })).then((d) => {
              run = d.run;
              if (arm) arm.checked = false;
              say("sdlc-publish-status",
                t === "none" ? "No deploy." : "Deploy launched: " + (d.unit || t), "ok");
              if (t !== "none") verifyDeploy();
            }));
      });
    }

    const verify = $("sdlc-deploy-verify");
    if (verify) verify.addEventListener("click", () => verifyDeploy());

    const resume = $("sdlc-resume");
    if (resume) {
      resume.addEventListener("click", () => {
        withBusy("sdlc-resume", "Resuming…", () =>
          api("/api/dev/sdlc/runs/" + encodeURIComponent(run.run_id) + "/resume", { method: "POST" })
            .then((d) => { run = d.run; schedulePoll(); }));
      });
    }

    const raiseBudget = (seconds) => {
      if (!run) return Promise.resolve();
      return api("/api/dev/sdlc/runs/" + encodeURIComponent(run.run_id) + "/resume", {
        method: "POST",
        body: JSON.stringify({ first_artifact_budget_s: seconds }),
      }).then((d) => { run = d.run; schedulePoll(); });
    };
    const fuseBudget = $("sdlc-fuse-budget");
    if (fuseBudget) {
      fuseBudget.addEventListener("click", () => {
        const raw = window.prompt(
          "New first-artifact budget in seconds",
          String((run && run.first_artifact_budget_s) || 1800),
        );
        if (raw == null) return;
        const seconds = Number(raw);
        if (!Number.isFinite(seconds) || seconds <= 0) {
          say("sdlc-publish-status", "Budget must be a positive number of seconds.", "bad");
          return;
        }
        withBusy("sdlc-fuse-budget", "Resuming…", () => raiseBudget(seconds));
      });
    }
    const fuseEdit = $("sdlc-fuse-edit");
    if (fuseEdit) {
      fuseEdit.addEventListener("click", () => {
        expandedIntake.add("contract");
        const gate = document.querySelector('.sdlc-gate[data-gate="contract"]');
        if (gate) gate.hidden = false;
        paint();
        $("sdlc-requirements")?.focus();
      });
    }
    const fuseAbort = $("sdlc-fuse-abort");
    if (fuseAbort) {
      fuseAbort.addEventListener("click", () => {
        if (!run) return;
        withBusy("sdlc-fuse-abort", "Aborting…", () =>
          api("/api/dev/sdlc/runs/" + encodeURIComponent(run.run_id) + "/abort", { method: "POST" })
            .then((d) => { run = d.run; loadRuns(); }));
      });
    }

    const abort = $("sdlc-abort");
    if (abort) {
      abort.addEventListener("click", () => {
        withBusy("sdlc-abort", "Aborting…", () =>
          api("/api/dev/sdlc/runs/" + encodeURIComponent(run.run_id) + "/abort", { method: "POST" })
            .then((d) => { run = d.run; loadRuns(); }));
      });
    }

    const repoSel = $("dev-repo");
    if (repoSel) repoSel.addEventListener("change", () => {
      if (run && run.repo !== repo()) newChange();
      loadRuns();
    });
    $("dev-refresh")?.addEventListener("click", () => { refresh(); loadRuns(); });

    const runHost = $("sdlc-runs");
    if (runHost) {
      runHost.addEventListener("click", (ev) => {
        const card = ev.target && ev.target.closest && ev.target.closest("[data-run]");
        if (card && card.dataset.run) select(card.dataset.run);
      });
    }

    bindWho();
    } catch (err) {
      if (doc && doc.documentElement) {
        doc.documentElement.dataset.sdlcBind = String((err && err.stack) || err);
      }
      say("sdlc-start-status", (err && err.message) || String(err), "bad");
    }
    try {
      paint();
    } catch (err) {
      if (doc.documentElement) {
        doc.documentElement.dataset.sdlcPaint = String((err && err.message) || err);
      }
      say("sdlc-start-status", (err && err.message) || String(err), "bad");
    }
    Promise.resolve(loadRuns()).then(() => {
      const want = wantedRunId();
      if (want && (!run || run.run_id !== want)) return select(want);
    }).catch((err) => say("sdlc-start-status", err.message || String(err), "bad"));
  }

  // Node (tests) sees the rules; the browser sees only the bound UI.
  // Guard on `window` — a page or extension that defines `module` must not
  // skip bind(), or the repo list, run cards, and archive never attach.
  if (typeof window === "undefined" && typeof module === "object" && module.exports) {
    module.exports = {
      answerState, untouchedCount, answerPayload, intakeGateOpen,
      runPresentation,
      stageSessionRefs,
      usableSessionId,
      escapeHtml: esc,
      // Exposed for the acceptance vector: a retried stage must not keep
      // showing the previous attempt's verdict.
      lastStatusFor: (events, stage) => {
        const last = lifecycleEvent(events, stage);
        return last ? last.status : "";
      },
    };
    return;
  }

  if (typeof window !== "undefined") {
    window.addEventListener("pagehide", () => { closeAskStream(); });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind);
  } else {
    bind();
  }
})();
