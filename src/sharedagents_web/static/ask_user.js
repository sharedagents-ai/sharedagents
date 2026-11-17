/* ask_user.js — AskUserQuestion card shared by Chat and Dev.
 * Chat mounts it in the transcript. Dev mounts it in #sdlc-ask-host.
 * Live card state comes from /api/sessions/<sid>/stream (SSE).
 * Answers POST /answer; Stop POSTs /cancel. */
if (typeof state === "undefined") {
  window.state = {
    currentSessionId: null,
    askUserCards: {},
    pendingApprovals: [],
  };
}

function _askEl(tag, cls, html) {
  if (typeof el === "function") return el(tag, cls, html);
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function _askApiPost(path, body) {
  if (typeof apiPost === "function") return apiPost(path, body);
  const base = (document.querySelector('meta[name="sharedagents-base"]') || {}).content || "";
  return fetch(base + path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(async (resp) => {
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || data.message || String(resp.status));
    return data;
  });
}

/** True for ANY engine's spelling of the ask-user tool.
 *
 * ``widget_adapt.js`` owns the alias table (loads before this file, and must
 * stay in lockstep with the Python twin), so ``askuser``, ``userQuestion`` and
 * ``ask_user_question`` resolve exactly like the native ``question``. A
 * hardcoded list here silently loses the question card AND the card's place in
 * the turn (see the placement contract below), the history split, and the
 * parked-question guards — for that agent only, which is the worst kind of
 * drift. The fallback below repeats the same normalization for harnesses that
 * load this file without widget_adapt.js.
 */
function _isAskUserTool(name) {
  const adapt =
    (typeof WidgetAdapt !== "undefined" && WidgetAdapt) ||
    (typeof window !== "undefined" && window.WidgetAdapt) ||
    null;
  if (adapt && typeof adapt.kind === "function") {
    return adapt.kind(name) === "question";
  }
  const key = String(name || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
  return (
    key === "question" ||
    key === "askuserquestion" ||
    key === "userquestion" ||
    key === "askuser"
  );
}

function _askUserCardExists(callId) {
  const card = state.askUserCards && state.askUserCards[callId || ""];
  if (!card) return false;
  // A detached registry entry must not block remount — virtual-scroll prune
  // and innerHTML rebuilds drop the form while leaving this pointer behind.
  return !!(card.el && card.el.isConnected);
}

function _stripAskDisplayNotes(raw) {
  let notes = String(raw || "").trim();
  if (!notes) return "";
  notes = notes.replace(/^\s*[.,;:]\s*/, "");
  notes = notes.replace(
    /You can now continue with the user's answers in mind\.?\s*/gi,
    "",
  );
  notes = notes.replace(
    /Unanswered means no consent[^.]*\.\s*/gi,
    "",
  );
  return notes.replace(/^\s*[.,;:]\s*|\s*[.,;:]\s*$/g, "").trim();
}

function _rememberAskCard(callId, bubble, args) {
  if (!state || !bubble) return;
  const idx = (state.msgCache || []).findIndex((m) => m && m.el === bubble);
  if (idx < 0) return;
  const prev = state.msgCache[idx].askCall || {};
  state.msgCache[idx].role = "ask_user";
  state.msgCache[idx].askCall = {
    callId: String(callId || prev.callId || ""),
    args: args || prev.args || {},
    content: prev.content || "",
    ok: prev.ok,
    finalized: !!prev.finalized,
    submitted: prev.submitted,
  };
}

function _askCacheRow(bubble) {
  return ((state && state.msgCache) || []).find((m) => m && m.el === bubble) || null;
}

/** A remount rebuilds the form from its args (default pre-checked). Put back
 *  the exact form state this tab already sent (see _submitAskUser) so the card
 *  stays locked and settled instead of turning answerable again before the
 *  result arrives. */
function _restoreSubmittedAsk(callId, snapshot) {
  const card = state.askUserCards && state.askUserCards[callId];
  const form = card && card.el && card.el.querySelector(".ask-user-form");
  if (!form || !Array.isArray(snapshot) || !snapshot.length) return;
  form.querySelectorAll(".ask-user-q").forEach((block, qi) => {
    const row = snapshot[qi] || {};
    const checked = Array.isArray(row.checked) ? row.checked : [];
    block.querySelectorAll(".ask-user-opt-input").forEach((inp) => {
      inp.checked = checked.indexOf(inp.value) >= 0;
    });
    const otherInput = block.querySelector(".ask-user-other-input");
    if (otherInput) otherInput.value = String(row.other || "");
  });
  _markCheckedAskOptions(form);
  _lockAskUserForm(form);
  card.answered = true;
  card.submitted = true;
  const status = card.el.querySelector(".ask-user-status");
  if (status) {
    status.className = "ask-user-status ask-user-status-waiting";
    status.textContent = "Waiting for the assistant…";
  }
  if (typeof updateSendState === "function") updateSendState();
}

// ── In-place card placement ───────────────────────────────────────────────
//
// An AskUserQuestion card belongs at the ASK POINT, inside the turn that asked
// it: the transcript then reads chronologically — the round's answer text, the
// question you answered, and the continuation that follows it. The turn keeps
// its agent + timestamp on the card, so a question is still attributable to
// the turn that asked it.
//
// Everything the turn produces after the answer (the post-answer continuation,
// a mid-turn steer) therefore appends BELOW the card, exactly like any other
// new row. Do NOT relocate later rows above a card to keep the card last: an
// earlier revision did that, and the continuation visibly streamed in ABOVE
// the answer the operator had just given.

function resetAskUserCards() {
  const host = _askUserPageHost();
  if (host) host.innerHTML = "";
  if (state) state.askUserCards = {};
}

function unbindAskUserSession() {
  closeAskUserStream();
  resetAskUserCards();
  if (!state) return;
  state.currentSessionId = null;
  state.askUserSessionAliases = [];
  state.pendingApprovals = [];
}

function bindAskUserSession(sid, aliases) {
  const next = String(sid || "").trim();
  const prev = String((state && state.currentSessionId) || "");
  if (prev && next && prev !== next) resetAskUserCards();
  if (!next) {
    unbindAskUserSession();
    return "";
  }
  if (state) {
    state.currentSessionId = next;
    state.askUserSessionAliases = Array.isArray(aliases) ? aliases.slice() : [];
  }
  return next;
}

function _askUserPageHost() {
  return document.getElementById("sdlc-ask-host");
}

function _appendAskUserBubble(callId) {
  if (typeof _createMessageElement !== "function" || typeof VirtualScroller === "undefined") {
    return appendSystemMessage("", false);
  }
  if (typeof setWelcomeVisible === "function") setWelcomeVisible(false);
  const div = _createMessageElement("ask_user", "");
  if (callId) div.dataset.callId = callId;
  const idx = VirtualScroller.add("ask_user", "", div);
  VirtualScroller.attach(div, idx);
  if (typeof scrollToBottom === "function") scrollToBottom();
  return div;
}

function _placeAskCardAfter(refEl, cardEl) {
  if (!refEl || !cardEl || !refEl.parentNode || refEl === cardEl) return;
  if (refEl.nextSibling === cardEl) return;
  refEl.parentNode.insertBefore(cardEl, refEl.nextSibling);
  const cache = (state && state.msgCache) || [];
  const from = cache.findIndex((m) => m && m.el === cardEl);
  const to = cache.findIndex((m) => m && m.el === refEl);
  if (from < 0 || to < 0) return;
  const [row] = cache.splice(from, 1);
  const dest = cache.findIndex((m) => m && m.el === refEl);
  if (dest >= 0) cache.splice(dest + 1, 0, row);
}

function _mountAskUserOnDevHost(pageHost, callId) {
  pageHost.querySelectorAll(".sdlc-status").forEach((node) => node.remove());
  let bubble = pageHost.querySelector(".ask-user-card");
  // Dev has a one-card host, not a transcript.  A conductor stream is
  // replayed from offset zero after navigation, so an answered question can
  // arrive after the run projection has already mounted its live successor.
  // Reusing the same bubble in that case changes the bubble's call id while
  // leaving both forms inside it.  reconcileAskCards then sees the historical
  // id and locks the live successor as well.  Give every call its own bubble;
  // later replay frames may replace it, and the authoritative run projection
  // will always put the currently pending call back.
  if (bubble && String(bubble.dataset.callId || "") !== String(callId || "")) {
    pageHost.innerHTML = "";
    bubble = null;
  }
  let content;
  if (!bubble) {
    pageHost.innerHTML = "";
    bubble = document.createElement("div");
    bubble.className = "message ask-user-card";
    content = document.createElement("div");
    content.className = "message-content";
    bubble.appendChild(content);
    pageHost.appendChild(bubble);
  } else {
    content = bubble.querySelector(".message-content") || bubble;
  }
  bubble.dataset.callId = callId;
  return { bubble, content };
}

function renderAskUserCard(callId, args, hostContentEl) {
  if (!state.askUserCards) state.askUserCards = {};
  const existing = state.askUserCards[callId];
  const existingOwnsCall = !!(
    existing
    && existing.el
    && String(existing.el.dataset.callId || "") === String(callId || "")
  );
  if (existing && !existing.finalized && existing.el
      && existing.el.isConnected && existingOwnsCall) {
    return;
  }
  if (existing && !existing.finalized
      && (!existing.el || !existing.el.isConnected || !existingOwnsCall)) {
    delete state.askUserCards[callId];
  }
  if (existing && existing.finalized) {
    // Gateway reuses call_0 each round. The live registry may be
    // repointed, but a settled Chat history card must keep its form.
    // Only Dev's single-card host (#sdlc-ask-host) is cleared.
    const pageHost = !hostContentEl ? _askUserPageHost() : null;
    if (!pageHost && !hostContentEl) {
      // Chat SSE replay of the same call must not unlock a Stopped card
      // or the follow-up gets queued behind the parked ask.
      return;
    }
    delete state.askUserCards[callId];
    if (pageHost) pageHost.innerHTML = "";
  }
  const questions =
    args && Array.isArray(args.questions) ? args.questions : [];

  let bubble;
  let content;
  const pageHost = !hostContentEl ? _askUserPageHost() : null;
  if (hostContentEl) {
    bubble = hostContentEl.closest(".message") || hostContentEl;
    content = hostContentEl;
    if (content && content.querySelector && content.querySelector(".ask-user-form")) {
      content.innerHTML = "";
    }
  } else if (pageHost) {
    const mounted = _mountAskUserOnDevHost(pageHost, callId);
    bubble = mounted.bubble;
    content = mounted.content;
  } else {
    bubble = _appendAskUserBubble(callId);
    content = bubble.querySelector(".message-content");
    if (content) content.innerHTML = "";
  }
  bubble.classList.add("ask-user-card");
  bubble.dataset.callId = callId;

  const form = _askEl("form", "ask-user-form");
  form.appendChild(
    _askEl("div", "ask-user-heading", "❓ The assistant needs your input"),
  );

  questions.forEach((q, qi) => {
    if (!q || typeof q !== "object") return;
    const multi = !!(q.multiSelect || q.multiple);
    const qWrap = _askEl("div", "ask-user-q");
    if (q.header) {
      const h = _askEl("div", "ask-user-q-header");
      h.textContent = String(q.header);
      qWrap.appendChild(h);
    }
    const qt = _askEl("div", "ask-user-q-text");
    qt.textContent = String(q.question || "");
    qWrap.appendChild(qt);

    const optsWrap = _askEl("div", "ask-user-opts");
    const inputType = multi ? "checkbox" : "radio";
    const groupName = `auq-${callId}-${qi}`;
    const opts = Array.isArray(q.options) ? q.options : [];
    opts.forEach((opt, oi) => {
      const label =
        opt && typeof opt === "object" ? String(opt.label || "") : String(opt || "");
      const desc =
        opt && typeof opt === "object" ? String(opt.description || "") : "";
      const lab = _askEl("label", "ask-user-opt");
      const inp = document.createElement("input");
      inp.type = inputType;
      inp.name = groupName;
      inp.className = "ask-user-opt-input";
      inp.value = label;
      if (!multi && oi === 0) inp.checked = true;
      lab.appendChild(inp);
      const body = _askEl("span", "ask-user-opt-body");
      const lt = _askEl("span", "ask-user-opt-label");
      lt.textContent = label;
      body.appendChild(lt);
      if (desc) {
        const dt = _askEl("span", "ask-user-opt-desc");
        dt.textContent = desc;
        body.appendChild(dt);
      }
      lab.appendChild(body);
      optsWrap.appendChild(lab);
    });

    const otherLab = _askEl("label", "ask-user-opt ask-user-other");
    const otherToggle = document.createElement("input");
    otherToggle.type = inputType;
    otherToggle.name = groupName;
    otherToggle.className = "ask-user-opt-input ask-user-other-toggle";
    otherToggle.value = "__other__";
    otherLab.appendChild(otherToggle);
    const otherBody = _askEl("span", "ask-user-opt-body");
    otherBody.appendChild(_askEl("span", "ask-user-opt-label", "Other"));
    const otherInput = document.createElement("input");
    otherInput.type = "text";
    otherInput.className = "ask-user-other-input";
    otherInput.placeholder = "Type a custom answer…";
    otherInput.addEventListener("focus", () => {
      otherToggle.checked = true;
    });
    otherInput.addEventListener("input", () => {
      if (otherInput.value) otherToggle.checked = true;
    });
    otherBody.appendChild(otherInput);
    otherLab.appendChild(otherBody);
    optsWrap.appendChild(otherLab);

    qWrap.appendChild(optsWrap);
    form.appendChild(qWrap);
  });

  const actions = _askEl("div", "ask-user-actions");
  const submit = _askEl("button", "ask-user-submit", "Submit");
  submit.type = "submit";
  const stop = _askEl("button", "ask-user-stop", "Stop");
  stop.type = "button";
  stop.title = "Do not answer — abort this question and stop the conversation";
  const status = _askEl("span", "ask-user-status");
  actions.appendChild(submit);
  actions.appendChild(stop);
  actions.appendChild(status);
  form.appendChild(actions);
  form.appendChild(
    _askEl(
      "div",
      "ask-user-alternatives",
      "Or type below to steer this turn. Stop ends the question without answering.",
    ),
  );

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    _submitAskUser(callId, questions, form, status);
  });
  stop.addEventListener("click", (e) => {
    e.preventDefault();
    _abortAskUser(callId);
  });

  if (content) content.appendChild(form);
  state.askUserCards[callId] = {
    el: bubble,
    answered: false,
    args: args || {},
    sessionId: String((state && state.currentSessionId) || ""),
  };
  _rememberAskCard(callId, bubble, args);
  if (typeof updateSendState === "function") updateSendState();
  if (typeof scrollToBottom === "function") scrollToBottom();
}

function _lockAskUserForm(form) {
  if (!form) return;
  form.classList.add("ask-user-locked");
  // Do not disable radios/checkboxes — a disabled checked input looks unmarked.
  form.querySelectorAll("button").forEach((n) => {
    n.disabled = true;
  });
  form.querySelectorAll("input").forEach((n) => {
    n.tabIndex = -1;
  });
}

function _unlockAskUserForm(form) {
  if (!form) return;
  form.classList.remove("ask-user-locked");
  form.querySelectorAll("button").forEach((n) => {
    n.disabled = false;
  });
  form.querySelectorAll("input").forEach((n) => {
    n.removeAttribute("tabindex");
  });
}

function askUserHasPendingCard() {
  const cards = state && state.askUserCards;
  if (!cards) return false;
  for (const id of Object.keys(cards)) {
    const c = cards[id];
    if (c && !c.answered && !c.finalized) return true;
  }
  return false;
}

function cancelPendingAskTurn(options) {
  if (typeof cancelStreaming === "function") {
    return cancelStreaming(options || {});
  }
  const sid = String((state && state.currentSessionId) || "");
  if (!sid) return Promise.resolve();
  return _askApiPost(
    "/api/sessions/" + encodeURIComponent(sid) + "/cancel",
    { force: true },
  );
}

function abortAllAskUserCards() {
  const cards = (state && state.askUserCards) || {};
  const ids = Object.keys(cards).filter((id) => {
    const c = cards[id];
    return c && !c.finalized;
  });
  if (!ids.length) return cancelPendingAskTurn();
  return _abortAskUser(ids[0]);
}

async function _abortAskUser(callId) {
  const card = state.askUserCards && state.askUserCards[callId];
  if (!card || card.finalized) return;
  _finalizeAskUserCard(
    callId,
    "",
    false,
    "⏹ Stopped — question dismissed",
  );
  if (typeof state !== "undefined") {
    // Hold the abort gate until turn_ended (or the settlement timeout) so
    // leftover ask SSE cannot remount the card or re-arm isStreaming.
    state._instantAbort = true;
    state.livePhysicalTurnId = "";
    state.liveEngine = "";
    const sid = state.currentSessionId;
    const sess = sid && state.sessions && state.sessions[sid];
    if (sess && String(sess.state || "") === "ask") sess.state = "";
    if (sess && sess.activity && sess.activity.attention === "question") {
      sess.activity = Object.assign({}, sess.activity, { attention: "" });
    }
  }
  try {
    const sid = String((state && state.currentSessionId) || "");
    if (sid && typeof _askApiPost === "function") {
      // Resolve the parked gateway waiter. /cancel alone fut.cancel()s it
      // (CancelledError), so the turn stays hung on "needs input" and the
      // next Chat prompt is queued behind a question that never completes.
      await _askApiPost(
        "/api/sessions/" + encodeURIComponent(sid) + "/answer",
        {
          callId,
          answers: [{ question: "", selected: [], notes: "stopped" }],
        },
      );
    }
  } catch (_e) { /* no pending waiter / already resolved */ }
  try {
    await cancelPendingAskTurn({ awaitServer: true });
    if (typeof waitForCancelSettlement === "function") {
      await waitForCancelSettlement(8000);
    }
  } catch (_e) { /* card already locked locally */ }
  // Stop of a parked ask often never emits turn_ended. Leave the instant-abort
  // gate up and the next prompt's SSE (turn_started) is dropped.
  if (typeof state !== "undefined") {
    state._instantAbort = false;
    state._askStopReleased = true;
  }
  if (typeof _stopStreaming === "function") _stopStreaming();
  if (window.devComposer && typeof window.devComposer.setFrozen === "function") {
    window.devComposer.setFrozen(false);
  }
  if (typeof updateSendState === "function") updateSendState();
}

async function _submitAskUser(callId, questions, form, status) {
  const card = state.askUserCards && state.askUserCards[callId];
  if (!card || card.answered) return;
  const boundSid = String((card && card.sessionId) || "");
  const liveSid = String((state && state.currentSessionId) || "");
  if (!boundSid || !liveSid || boundSid !== liveSid) {
    status.textContent = "No active session";
    return;
  }

  const answers = [];
  // Exact form state as sent, so a remount can put it back verbatim.
  const snapshot = [];
  const blocks = form.querySelectorAll(".ask-user-q");
  blocks.forEach((block, qi) => {
    const q = questions[qi] || {};
    const selected = [];
    const checked = [];
    block.querySelectorAll(".ask-user-opt-input").forEach((inp) => {
      if (inp.checked) checked.push(inp.value);
      if (inp.checked && inp.value !== "__other__") selected.push(inp.value);
    });
    const otherToggle = block.querySelector(".ask-user-other-toggle");
    const otherInput = block.querySelector(".ask-user-other-input");
    snapshot.push({ checked, other: otherInput ? otherInput.value : "" });
    let notes = "";
    if (
      otherToggle &&
      otherToggle.checked &&
      otherInput &&
      otherInput.value.trim()
    ) {
      notes = otherInput.value.trim();
      if (notes && selected.indexOf(notes) < 0) selected.push(notes);
    }
    answers.push({ question: String(q.question || ""), selected, notes });
  });

  _lockAskUserForm(form);
  _markCheckedAskOptions(form);
  card.answered = true;
  card.submitted = true;
  const cacheRow = _askCacheRow(card.el);
  if (cacheRow && cacheRow.askCall) cacheRow.askCall.submitted = snapshot;
  status.className = "ask-user-status ask-user-status-waiting";
  status.textContent = "Waiting for the assistant…";

  try {
    await _askApiPost(
      "/api/sessions/" +
        encodeURIComponent(boundSid) +
        "/answer",
      { callId, answers },
    );
  } catch (e) {
    // A virtual-scroll remount during the POST swaps in a new card object for
    // the SAME cache row; roll back that one. Any other card now under this
    // callId (another session, a later question reusing call_0) is not ours.
    const reg = state.askUserCards && state.askUserCards[callId];
    const live = reg && reg !== card && cacheRow && reg.sessionId === card.sessionId
      && _askCacheRow(reg.el) === cacheRow ? reg : card;
    // A card the result already settled keeps that outcome.
    if (!live.finalized) {
      const liveStatus = (live.el && live.el.querySelector(".ask-user-status")) || status;
      liveStatus.className = "ask-user-status ask-user-status-error";
      liveStatus.textContent =
        "Send failed — " + (e && e.message ? e.message : "error");
      live.answered = false;
      live.submitted = false;
      if (cacheRow && cacheRow.askCall) delete cacheRow.askCall.submitted;
      _unlockAskUserForm((live.el && live.el.querySelector(".ask-user-form")) || form);
    }
  }
}

function _normalizeAskAnswers(raw) {
  const out = [];
  (Array.isArray(raw) ? raw : []).forEach((item) => {
    if (!item || typeof item !== "object") return;
    let selected = Array.isArray(item.selected)
      ? item.selected.map((s) => String(s || "").trim()).filter(Boolean)
      : (item.selected ? [String(item.selected).trim()] : []);
    const notes = _stripAskDisplayNotes(item.notes);
    if (notes) {
      selected = selected.filter((s) => !/^unanswered$/i.test(s));
    }
    if (selected.length || notes) out.push({ selected, notes });
  });
  return out;
}

function _summaryFromAskAnswers(answers) {
  return (answers || []).map((a) => {
    const selected = (a.selected || []).filter((s) => !/^unanswered$/i.test(s));
    let s = selected.join(", ");
    const notes = a.notes || "";
    if (notes && notes !== s && selected.indexOf(notes) < 0) {
      s = s ? s + " · " + notes : notes;
    }
    return s;
  }).filter(Boolean).join(" | ");
}

function _applyAskAnswersToForm(form, answers) {
  if (!form || !answers || !answers.length) return false;
  const blocks = form.querySelectorAll(".ask-user-q");
  let marked = false;
  blocks.forEach((block, qi) => {
    const row = answers[qi] || answers[0] || {};
    const selected = Array.isArray(row.selected) ? row.selected : [];
    const notes = typeof _stripAskDisplayNotes === "function"
      ? _stripAskDisplayNotes(row.notes)
      : String(row.notes || "").trim();
    block.querySelectorAll(".ask-user-opt").forEach((lab) => lab.classList.remove("is-chosen"));
    block.querySelectorAll(".ask-user-opt-input").forEach((inp) => { inp.checked = false; });
    selected.forEach((label) => {
      if (/^unanswered$/i.test(label)) return;
      const wantOther = /^other$/i.test(label);
      let matched = false;
      block.querySelectorAll(".ask-user-opt-input").forEach((inp) => {
        const hit = inp.value === label || (wantOther && inp.value === "__other__");
        if (!hit) return;
        matched = true;
        inp.checked = true;
        const lab = inp.closest(".ask-user-opt");
        if (lab) lab.classList.add("is-chosen");
        marked = true;
      });
      if (!matched && label && label !== "__other__") {
        const otherToggle = block.querySelector(".ask-user-other-toggle");
        const otherInput = block.querySelector(".ask-user-other-input");
        const otherLab = block.querySelector(".ask-user-other");
        if (otherToggle) otherToggle.checked = true;
        if (otherInput && !otherInput.value) otherInput.value = label;
        if (otherLab) otherLab.classList.add("is-chosen");
        marked = true;
      }
    });
    if (notes) {
      const otherToggle = block.querySelector(".ask-user-other-toggle");
      const otherInput = block.querySelector(".ask-user-other-input");
      const otherLab = block.querySelector(".ask-user-other");
      if (otherToggle) otherToggle.checked = true;
      if (otherInput) otherInput.value = notes;
      if (otherLab) otherLab.classList.add("is-chosen");
      marked = true;
    }
  });
  return marked;
}

function _markCheckedAskOptions(form) {
  if (!form) return;
  form.querySelectorAll(".ask-user-q").forEach((block) => {
    block.querySelectorAll(".ask-user-opt").forEach((lab) => {
      const inp = lab.querySelector(".ask-user-opt-input");
      lab.classList.toggle("is-chosen", !!(inp && inp.checked));
    });
  });
}

function _clearAskSelections(form) {
  if (!form) return;
  form.querySelectorAll(".ask-user-opt-input").forEach((inp) => { inp.checked = false; });
  form.querySelectorAll(".ask-user-opt").forEach((lab) => lab.classList.remove("is-chosen"));
}

function _parseAskUserResult(content) {
  if (!content || typeof content !== "string" || !content.trim()) {
    return { isAnswered: false, summary: "", answers: [], empty: true };
  }
  const trimmed = content.trim();
  let obj = null;
  try {
    obj = JSON.parse(trimmed);
  } catch (_e) {
    obj = null;
  }

  if (obj && typeof obj === "object") {
    if (obj.ok === false || obj.timed_out || obj.status === "timeout" || obj.status === "unanswered") {
      return { isAnswered: false, summary: obj.summary || "Timed out waiting for response", answers: [] };
    }
    const answers = _normalizeAskAnswers(obj.answers);
    if (!answers.length && (obj.status === "timed_out" || obj.status === "cancelled")) {
      return { isAnswered: false, summary: "Timed out waiting for response", answers: [] };
    }
    let hasUnanswered = false;
    answers.forEach((a) => {
      const selStr = (a.selected || []).join(", ");
      if (/unanswered|timed out/i.test(selStr) || /unanswered|timed out/i.test(a.notes || "")) {
        hasUnanswered = true;
      }
    });
    const summary = _summaryFromAskAnswers(answers);
    if (hasUnanswered && (!summary || /unanswered|timed out/i.test(summary))) {
      return { isAnswered: false, summary: summary || "Timed out waiting for response", answers };
    }
    return { isAnswered: !hasUnanswered || !!summary, summary, answers };
  }

  // Parse standard answers prefixed with "User has answered your questions:" first,
  // before checking for explicit cancellation/timeout tokens, so prompt/answer text
  // containing words like "cancelled" or "timed out" is never misclassified.
  const answeredMatch = trimmed.match(/^User has answered your questions:\s*([\s\S]*)/i);
  if (answeredMatch) {
    const rest = answeredMatch[1];
    const pairRe = /"((?:[^"\\]|\\.)*)"\s*=\s*"((?:[^"\\]|\\.)*)"/g;
    let m;
    const answers = [];
    let lastIndex = 0;
    while ((m = pairRe.exec(rest)) !== null) {
      lastIndex = pairRe.lastIndex;
      let val = m[2].trim();
      const isUnanswered = /^unanswered$/i.test(val);
      answers.push({
        question: m[1],
        selected: isUnanswered ? [] : val.split(",").map((s) => s.trim()).filter(Boolean),
        notes: isUnanswered ? "unanswered" : "",
      });
    }
    const trailing = rest.slice(lastIndex);
    const notes = _stripAskDisplayNotes(trailing);
    if (notes && answers.length) {
      if (answers.length === 1 && !answers[0].selected.length) {
        answers[0].selected = [notes];
        answers[0].notes = notes;
      } else {
        answers[answers.length - 1].notes = notes;
      }
    }
    if (answers.length) {
      const hasRealAnswer = answers.some(
        (a) => (a.selected && a.selected.length) || (a.notes && !/^(?:unanswered|stopped|cancelled|canceled)$/i.test(a.notes)),
      );
      if (!hasRealAnswer) {
        return { isAnswered: false, summary: "Timed out waiting for response", answers: [] };
      }
      return {
        isAnswered: true,
        summary: _summaryFromAskAnswers(answers),
        answers,
      };
    }

    const lines = rest.split("\n")
      .map((l) => l.trim())
      .filter((l) => l.startsWith("-"))
      .map((l) => l.replace(/^-\s*/, ""));
    if (lines.length) {
      const lineAnswers = lines.map((line) => ({ selected: [line], notes: "" }));
      return { isAnswered: true, summary: lines.join(" | "), answers: lineAnswers };
    }
    if (/^unanswered/i.test(rest.trim())) {
      return { isAnswered: false, summary: "Timed out waiting for response", answers: [] };
    }
  }

  if (/^unanswered\s*\(timed\s*out\)|^timed\s*out\s*waiting\s*for\s*user/i.test(trimmed)) {
    return { isAnswered: false, summary: "Timed out waiting for response", answers: [] };
  }
  if (/^cancelled|^canceled/i.test(trimmed) || /question\s*cancell?ed/i.test(trimmed)) {
    return { isAnswered: false, summary: "Question cancelled", answers: [] };
  }

  return { isAnswered: false, summary: "", answers: [], empty: true };
}

function _summarizeAskAnswers(content) {
  const res = _parseAskUserResult(content);
  return res.summary;
}

function _finalizeAskUserCard(callId, content, ok, statusOverride) {
  const card = state.askUserCards && state.askUserCards[callId];
  if (!card) return;
  const parsed = _parseAskUserResult(content);
  if (parsed.empty && card.appliedAnswers && !statusOverride) {
    _lockAskUserForm(card.el && card.el.querySelector(".ask-user-form"));
    return;
  }
  card.answered = true;
  const form = card.el && card.el.querySelector(".ask-user-form");
  if (parsed.answers && parsed.answers.length) {
    _applyAskAnswersToForm(form, parsed.answers);
    card.appliedAnswers = true;
  } else if (card.submitted) {
    // The checked inputs are what the operator sent from this card.
    _markCheckedAskOptions(form);
  } else {
    // Closed without an answer: the only checked input is the Submit default
    // (or an abandoned pick). Painting it chosen reads as answered.
    _clearAskSelections(form);
  }
  if (parsed.empty && ok !== false && !statusOverride) {
    _lockAskUserForm(form);
    return;
  }
  card.finalized = true;
  const idx = (state.msgCache || []).findIndex((m) => m && m.el === card.el);
  if (idx >= 0 && state.msgCache[idx].askCall) {
    state.msgCache[idx].askCall.finalized = true;
    state.msgCache[idx].askCall.content = content || "";
    state.msgCache[idx].askCall.ok = ok;
    if (statusOverride) state.msgCache[idx].askCall.statusOverride = statusOverride;
  }
  _lockAskUserForm(form);
  const status = card.el && card.el.querySelector(".ask-user-status");
  if (status) {
    if (statusOverride) {
      status.className = "ask-user-status ask-user-status-error";
      status.textContent = statusOverride;
    } else if (ok === false) {
      status.className = "ask-user-status ask-user-status-error";
      status.textContent = "✗ Answer delivery failed";
    } else if (!parsed.isAnswered) {
      status.className = "ask-user-status ask-user-status-error";
      status.textContent = "⏱ " + (parsed.summary || "Timed out");
    } else {
      status.className = "ask-user-status ask-user-status-done";
      status.textContent = parsed.summary ? "✓ " + parsed.summary : "✓ Answered";
    }
  }
  if (typeof updateSendState === "function") updateSendState();
}

function _finalizePendingAskUserCards(statusText) {
  const cards = state.askUserCards || {};
  Object.keys(cards).forEach((callId) => {
    const card = cards[callId];
    if (!card || card.finalized) return;
    _finalizeAskUserCard(
      callId,
      "",
      false,
      statusText || "✗ Question closed",
    );
  });
}

function createAskEventSource(url) {
  if (typeof createLiveEventSource === "function") {
    return createLiveEventSource(url);
  }
  // Chat uses a fetch SSE reader so same-origin cookies survive proxies
  // that drop native EventSource. Dev must listen the same way.
  function FetchAskSource(src) {
    this.url = src;
    this.readyState = 0;
    this._listeners = Object.create(null);
    this._abort = new AbortController();
    this._buf = "";
    const self = this;
    fetch(src, {
      method: "GET",
      headers: { Accept: "text/event-stream" },
      credentials: "same-origin",
      signal: this._abort.signal,
    }).then((resp) => {
      if (self.readyState === 2) return;
      if (!resp.ok || !resp.body) {
        self.readyState = 2;
        _askSourceEmit(self, "error", {});
        return;
      }
      self.readyState = 1;
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      const pump = () => reader.read().then(({ value, done }) => {
        if (self.readyState === 2) return;
        if (done) {
          self.readyState = 2;
          _askSourceEmit(self, "error", {});
          return;
        }
        self._buf += dec.decode(value, { stream: true });
        let idx;
        while ((idx = self._buf.indexOf("\n\n")) >= 0) {
          const raw = self._buf.slice(0, idx);
          self._buf = self._buf.slice(idx + 2);
          let type = "message";
          const dataLines = [];
          raw.split("\n").forEach((line) => {
            if (line.startsWith("event:")) type = line.slice(6).trim();
            else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
          });
          const list = self._listeners[type] || [];
          for (let i = 0; i < list.length; i++) {
            try { list[i]({ data: dataLines.join("\n") }); } catch (_e) { /* ignore */ }
          }
        }
        return pump();
      }).catch(() => {
        self.readyState = 2;
        _askSourceEmit(self, "error", {});
      });
      return pump();
    }).catch(() => {
      this.readyState = 2;
      _askSourceEmit(this, "error", {});
    });
  }
  FetchAskSource.prototype.addEventListener = function (type, fn) {
    (this._listeners[type] || (this._listeners[type] = [])).push(fn);
  };
  FetchAskSource.prototype.close = function () {
    this.readyState = 2;
    try { this._abort.abort(); } catch (_e) { /* ignore */ }
  };
  return new FetchAskSource(url);
}

function hydratePendingAskUserCards() {
  if (typeof renderAskUserCard !== "function") return;
  const sid = String((state && state.currentSessionId) || "");
  const aliases = new Set(
    [sid].concat((state && state.askUserSessionAliases) || []).map((id) => String(id || "")).filter(Boolean),
  );
  if (!aliases.size) return;
  const items = Array.isArray(state.pendingApprovals) ? state.pendingApprovals : [];
  items.forEach((a) => {
    if (!a || a.type !== "ask_user") return;
    const recSid = String(a.session_id || "");
    const recGw = String(a.session_hash || "");
    if (!aliases.has(recSid) && !aliases.has(recGw)) return;
    const callId = String(a.call_id || "");
    if (!callId || _askUserCardExists(callId)) return;
    if (state && Array.isArray(state.msgCache) && state.msgCache.some((m) => m && m.askCall && String(m.askCall.callId || "") === callId)) {
      return;
    }
    // REST can report the pending question while SSE starts at EOF. There may
    // never be a tool_call frame to seal the pre-question assistant. Do it at
    // the hydration boundary so resumed output opens below this card.
    if (typeof _sealLiveAssistantBeforeAskUser === "function") {
      _sealLiveAssistantBeforeAskUser();
    }
    const options = (Array.isArray(a.options) ? a.options : []).map((label) => (
      typeof label === "string" ? { label, description: "" } : label
    ));
    renderAskUserCard(callId, {
      questions: [{
        question: a.summary || "The assistant needs your input",
        header: "clarify",
        options: options.length ? options : [{ label: "Other", description: "" }],
      }],
    });
  });
}

function sessionStreamUrl(sid, from) {
  const base = (document.querySelector('meta[name="sharedagents-base"]') || {}).content || "";
  return (
    base + "/api/sessions/" + encodeURIComponent(sid) +
    "/stream?mode=events&from=" + encodeURIComponent(from || 0) +
    "&activity=stream&reasoning=stream&tools=stream"
  );
}

function _askSourceEmit(source, type, event) {
  if (!source || !source._listeners) return;
  const list = source._listeners[type] || [];
  for (let i = 0; i < list.length; i++) {
    try { list[i](event || {}); } catch (_e) { /* ignore */ }
  }
}

const _askStream = {
  source: null,
  sid: "",
  sourceId: 0,
  offset: 0,
  epoch: 0,
  onEvent: null,
  reconnectTimer: null,
};

function askUserStreamOpen(sid) {
  return !!(
    _askStream.source
    && _askStream.sid === String(sid || "")
    && _askStream.source.readyState !== 2
  );
}

function askUserStreamEpoch() {
  return _askStream.epoch;
}

function closeAskUserStream() {
  if (_askStream.reconnectTimer) {
    clearTimeout(_askStream.reconnectTimer);
    _askStream.reconnectTimer = null;
  }
  if (_askStream.source) {
    try { _askStream.source.close(); } catch (_e) { /* ignore */ }
  }
  _askStream.source = null;
  _askStream.sid = "";
}

function openAskUserStream(sid, fromOffset, onEvent) {
  const next = String(sid || "").trim();
  if (!next) {
    closeAskUserStream();
    return 0;
  }
  if (askUserStreamOpen(next) && _askStream.onEvent === onEvent) {
    return _askStream.epoch;
  }
  closeAskUserStream();
  const epoch = ++_askStream.epoch;
  _askStream.sid = next;
  _askStream.offset = Number.isFinite(fromOffset) ? fromOffset : 0;
  _askStream.onEvent = onEvent;
  _startAskUserStream(next, _askStream.offset, epoch, epoch);
  return epoch;
}

function _startAskUserStream(sid, fromOffset, epoch, sourceId) {
  if (_askStream.epoch !== epoch || _askStream.sid !== sid) return;
  const url = sessionStreamUrl(sid, fromOffset);
  let es;
  try {
    es = createAskEventSource(url);
  } catch (_e) {
    return;
  }
  _askStream.source = es;
  _askStream.sourceId = sourceId;
  const mountOffset = fromOffset;
  let sawFrame = false;

  function accept() {
    return (
      _askStream.epoch === epoch
      && _askStream.sid === sid
      && _askStream.source === es
      && _askStream.sourceId === sourceId
    );
  }

  es.addEventListener("log", (e) => {
    if (!accept()) return;
    let frame;
    try { frame = JSON.parse(e.data); } catch (_e) { return; }
    sawFrame = true;
    if (typeof frame.offset === "number") _askStream.offset = frame.offset;
    const ev = frame && (frame.event || frame);
    if (ev && typeof _askStream.onEvent === "function") _askStream.onEvent(ev);
  });

  function resume(delayMs) {
    if (!accept()) return;
    try { es.close(); } catch (_e) { /* ignore */ }
    if (_askStream.source === es) _askStream.source = null;
    const reopen = () => {
      if (_askStream.epoch !== epoch || _askStream.sid !== sid || _askStream.source) return;
      _startAskUserStream(sid, _askStream.offset, epoch, sourceId);
    };
    if (!delayMs) {
      reopen();
      return;
    }
    _askStream.reconnectTimer = setTimeout(reopen, delayMs);
  }

  es.addEventListener("reconnect", (e) => {
    if (!accept()) return;
    try {
      const d = JSON.parse(e.data);
      if (typeof d.offset === "number") _askStream.offset = d.offset;
    } catch (_e) { /* ignore */ }
    const progressed = _askStream.offset !== mountOffset || sawFrame;
    resume(progressed ? 0 : 1000);
  });
  es.addEventListener("error", () => { if (accept()) resume(1000); });
  es.addEventListener("gw_error", () => { if (accept()) resume(1000); });
}

if (typeof window !== "undefined") {
  window.addEventListener("pagehide", () => { closeAskUserStream(); });
}
