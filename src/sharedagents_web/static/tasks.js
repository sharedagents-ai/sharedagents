/* Tasks tab — a manual list of sessions the operator chose to track,
 * grouped into three cards matching the real Claude Code `claude agents`
 * view (code.claude.com/docs/en/agent-view): Needs input, Working,
 * Completed — all derived LIVE from the agent tree, never a manual flag.
 * sharedagents chat sessions have no bounded "process finished" signal the way
 * a cloud task does (a chat just idles forever, resumable), so Completed
 * is approximated by the one clean "at rest" signal available: plain
 * `idle` (turn ended, nothing pending).
 *
 * `waiting` is Working, not Needs input: per session_index.py's
 * EventProjector, it fires ONLY when a parent's own turn ended but a
 * dispatched subagent/async job is still live (_parent_still_has_work) —
 * i.e. the system is actively working, just one level removed. It is NEVER
 * set for "blocked on a human decision" (permission/tool-approval requests
 * don't have a distinct state at all — they stay in whatever live state the
 * turn was already in). Confirmed live: a jarvis session shown "waiting"
 * while its dispatched `plan` subagent was `tool_running` was still
 * actively progressing the task, composer showed "waiting on subagents...".
 *
 * paused, error, closed, stale, and a session that has never run a turn yet
 * default to Needs input (error explicitly grouped there per instruction).
 * Removing a row (×) just untracks it — never deletes the chat.
 */
(function () {
  "use strict";

  const BASE = ((document.querySelector('meta[name="sharedagents-base"]') || {}).content || "");

  const boardEl = document.getElementById("tasks-board");
  const emptyEl = document.getElementById("tasks-empty");
  const updatedEl = document.getElementById("tasks-updated");
  const topicFilterEl = document.getElementById("tasks-topic-filter");
  const topicOptionsEl = document.getElementById("tasks-topic-options");
  const newToggleBtn = document.getElementById("tasks-new-toggle");
  const newForm = document.getElementById("tasks-new-form");
  const newCancelBtn = document.getElementById("tasks-new-cancel");
  const newTitleInput = document.getElementById("tasks-new-title");
  const newTopicInput = document.getElementById("tasks-new-topic");
  const describeTopicInput = document.getElementById("tasks-describe-topic");
  const showArchivedEl = document.getElementById("tasks-show-archived");

  // Exactly the claude-agents-view state machine, adapted to what sharedagents
  // can actually detect (see the module-level comment above). Order =
  // attention-first, same as the real agent view (needs input before working
  // before completed). A fourth "Archived" card appears only while the
  // show-archived toggle is on.
  const CARDS = [
    { key: "needs_input", label: "Needs input" },
    { key: "working", label: "Working" },
    { key: "completed", label: "Completed" },
  ];
  const ARCHIVED_CARD = { key: "archived", label: "Archived" };
  const WORKING_STATES = ["prompting", "prefill", "reasoning", "streaming", "tool_running", "compacting", "waiting"];
  const COMPLETED_STATES = ["idle"];
  const ASK_STATES = ["ask"];

  function activityFor(it) {
    return it && it.activity && typeof it.activity === "object"
      ? it.activity : (it || {});
  }

  function cardFor(it) {
    if (it.card === "archived" || it.card === "needs_input" ||
        it.card === "working" || it.card === "completed")
      return it.card;
    if (it.archived) return "archived";
    const s = String(it.state || "");
    const a = activityFor(it);
    // A tracked row with no turn has a V2-compatible idle projection, but it
    // is not a completed task yet. Keep the board's established Needs input
    // affordance for this pre-start state.
    if (s === "unstarted") return "needs_input";
    if (a.attention === "question" || a.attention === "permission" ||
        a.attention === "operator_paused") return "needs_input";
    // Stale own-turn rows belong in Needs input; stale parents waiting on
    // live descendants stay Working. Key off descendant count, not tree_busy
    // — a bare waiting row is already tree_busy with zero children.
    if (a.status === "needs_input" || a.status === "needs_review")
      return "needs_input";
    if (a.stale && !((Number(a.active_descendant_count) > 0) && !a.own_live))
      return "needs_input";
    // Closed own-turn with leftover V2 phase (idle + streaming, no live
    // turn) must not sit in Working. The hollow idle dot uses `it.state`.
    if (s === "idle" && !a.own_live && !a.tree_busy &&
        a.attention !== "question" && a.attention !== "permission" &&
        a.attention !== "operator_paused") {
      if (["failed", "interrupted"].includes(String(a.last_outcome || "")))
        return "needs_input";
      return "completed";
    }
    if (a.status === "working" || (a.phase && a.phase !== "idle") ||
        a.tree_busy)
      return "working";
    if (["failed", "interrupted"].includes(String(a.last_outcome || "")))
      return "needs_input";
    if (a.status === "detached") return "completed";
    // Explicit ask mapping: the server emits the derived
    // `ask` state for a session parked on a pending ask_user/tool_permission
    // approval. It must land in Needs input BEFORE the working check — a
    // parked session's tree node still says tool_running, which would
    // otherwise bucket it as Working while the operator is being asked.
    if (ASK_STATES.includes(s)) return "needs_input";
    if (WORKING_STATES.includes(s)) return "working";
    if (COMPLETED_STATES.includes(s)) return "completed";
    return "needs_input";
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // last_activity_at is an ISO string when it came from the web session store
  // but a NUMERIC epoch (seconds, sometimes ms) when it came from the agent
  // tree — parsing only the ISO form left the age blank on every tree-backed
  // row. Same normalisation agent_mon_mini.js does.
  function ageStr(ts) {
    if (!ts) return "";
    let ms;
    if (typeof ts === "number") {
      ms = ts > 1e12 ? ts : ts * 1000;
    } else {
      ms = Date.parse(ts);
    }
    if (!ms || Number.isNaN(ms)) return "";
    const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
    if (s < 60) return s + "s";
    if (s < 3600) return Math.floor(s / 60) + "m";
    if (s < 86400) return Math.floor(s / 3600) + "h";
    return Math.floor(s / 86400) + "d";
  }

  // The board's sort key is the SAME heterogeneous mix ageStr() has to
  // normalise: epoch ints from the agent tree, ISO strings from the web
  // session store. Comparing the raw values is NOT transitive — a
  // number-vs-string `<` coerces the ISO string to NaN, the comparison is
  // always false, and the comparator loses transitivity, which shuffled the
  // "other day" tasks to the top of every column. Normalise both sides to
  // epoch-ms before comparing.
  function tsMs(ts) {
    if (!ts) return 0;
    if (typeof ts === "number") return ts > 1e12 ? ts : ts * 1000;
    const ms = Date.parse(ts);
    return Number.isNaN(ms) ? 0 : ms;
  }

  let items = [];
  let topicFilter = "";
  let showArchived = false;
  let refreshInFlight = null;
  let refreshQueued = false;
  // Recap subcards the operator has EXPANDED, keyed by board row id. The
  // board is rebuilt from scratch on every 5 s poll (render() wipes
  // boardEl.innerHTML), so the open/closed state has to live outside the DOM
  // or a recap would slam shut under the operator mid-read. Absent = the
  // default: COLLAPSED (a completed task must not push the rest of the
  // column down on its own).
  const openRecaps = new Set();
  const selectedRecaps = new Set();
  let archiveBusy = false;
  let archiveMessage = "";

  function archiveEligible(it) {
    return it.recap_archivable === true && cardFor(it) === "completed" && !it.archived;
  }

  function recapHidden(it) {
    return cardFor(it) === "working"
      ? it.summary_previous !== true : it.summary_pending === true;
  }

  // The task is shared exactly when its conversation is: the same lock /
  // people / globe control as the chat header, drawn by share.js from the
  // sharing payload the server put on the row.
  function shareButtonHtml(sharing) {
    const f = window.ShareControl.face(sharing);
    return '<button type="button" class="task-share icon-btn share-btn' + (f.isPublic ? " is-public" : "") + '"' +
      ' title="' + esc(f.summary) + '" aria-label="' + esc(f.label) + '"' +
      ' aria-haspopup="dialog" aria-expanded="false">' + f.icon + "</button>";
  }

  function rowHtml(it) {
    const state = String(it.state || "idle");
    const working = cardFor(it) === "working";
    const recapPending = !working && it.summary_pending === true;
    const a = activityFor(it);
    let displayLabel = a.display && a.display.label ? a.display.label : "";
    if (a.status === "needs_review" || displayLabel === "Needs review") {
      displayLabel = "Needs input";
    }
    const sdlcStatus = String(it.sdlc_status || "");
    if (sdlcStatus === "done") displayLabel = "Merged";
    else if (sdlcStatus === "running" || sdlcStatus === "pr_open") displayLabel = "Building";
    else if (sdlcStatus === "blocked") displayLabel = "Blocked";
    else if (sdlcStatus === "intake") displayLabel = "Clarify";
    else if (sdlcStatus === "aborted") displayLabel = "Aborted";
    if (!working && it.card === "needs_input" && it.summary_verdict === "follow_up")
      displayLabel = "Needs input";
    if (recapPending) displayLabel = "Recapping…";
    const age = ageStr(it.last_activity_at);
    // Two titles when both exist: the operator's own task name on top, and
    // beneath it what sharedagents's title agent decided the conversation is
    // about. The server blanks session_title when it would just repeat the
    // task name, so this line only appears when it actually adds something.
    const sessionTitle = String(it.session_title || "");
    // Archived rows are clearly labelled and cannot be
    // untracked again (the × is an untrack affordance, meaningless once the
    // row is already archived — the session itself is never deleted).
    const archived = Boolean(it.archived);
    const archivedClass = archived ? " is-archived" : "";
    const archivedPill = archived
      ? '<span class="task-pill task-pill-archived" title="This task was removed from the board; the chat still exists.">archived</span>'
      : "";
    const removeBtn = archived || it.shared_task
      ? ""
      : '<button type="button" class="task-remove" title="Untrack (does not delete the chat)" aria-label="Untrack">×</button>';
    const runId = String(it.sdlc_run_id || "");
    const devPill = runId
      ? '<a class="task-pill task-pill-dev" href="' + BASE + "/dev?run=" + encodeURIComponent(runId) + '" title="Open this run in the Dev tab">Dev</a>'
      : "";
    const topicPill = (it.topic && !runId)
      ? '<span class="task-pill">' + esc(it.topic) + "</span>"
      : "";
    // Completion recap (written by the gateway's hidden "summary" side call
    // on the Jarvis default deployment, persisted on the board row): the
    // verdict pill is always visible — it is the one-line answer to "did this
    // really finish?" — while the recap text and the follow-up note live in a
    // subcard the operator expands from the pill or the row. The server moves
    // a completed task with a current follow-up verdict into Needs input.
    // A working task shows the server-provided recap of its previous turn as
    // a neutral "last recap" pill — never its old verdict as if current.
    const previous = working && it.summary_previous === true;
    const hideRecap = recapHidden(it);
    const summaryText = hideRecap ? "" : String(it.summary || "").trim();
    const followUp = hideRecap ? "" : String(it.summary_follow_up || "").trim();
    const verdictKey = hideRecap || previous ? "" : String(it.summary_verdict || "").trim().toLowerCase();
    const recapStatus = working ? "" : String(it.summary_status || "").trim().toLowerCase();
    const hasRecap = !hideRecap && Boolean(summaryText || followUp || verdictKey || recapStatus);
    if (!hasRecap) openRecaps.delete(String(it.id || ""));
    const recapOpen = hasRecap && openRecaps.has(String(it.id || ""));
    // The verdict pill is the recap's expander (the row click toggles it too):
    // focusable, with its open state for keyboard and screen readers.
    const expander = hasRecap
      ? ' role="button" tabindex="0" aria-expanded="' + (recapOpen ? "true" : "false") + '"'
      : "";
    let recapPill = "";
    if (previous && hasRecap) {
      recapPill = '<span class="task-pill task-pill-recap is-previous"' + expander + ' title="Recap of the previous turn; the task is working again.">last recap</span>';
    } else if (verdictKey === "complete") {
      recapPill = '<span class="task-pill task-pill-recap is-complete"' + expander + ' title="The recap side call judged this task complete.">recap: complete</span>';
    } else if (verdictKey === "follow_up") {
      recapPill = '<span class="task-pill task-pill-recap is-follow-up"' + expander + ' title="' + esc(followUp || "The recap judged this task needs a follow-up to reach its goal.") + '">recap: needs follow-up</span>';
    } else if (!recapPending && (recapStatus === "running" || recapStatus === "queued")) {
      recapPill = '<span class="task-pill task-pill-recap is-pending"' + expander + ' title="Recapping this task…">recap…</span>';
    } else if (recapStatus === "failed") {
      recapPill = '<span class="task-pill task-pill-recap is-pending"' + expander + ' title="The recap call failed; the board retries with backoff.">recap retry</span>';
    } else if (hasRecap) {
      // Plain-text fallback recaps have no verdict, but still need a
      // keyboard-accessible expander after removing the separate toggle.
      recapPill = '<span class="task-pill task-pill-recap"' + expander + ' title="Show the task recap">recap</span>';
    }
    const recapPanel = hasRecap
      ? '<div class="task-summary-panel"' + (recapOpen ? "" : " hidden") + '>' +
          (summaryText
            ? '<div class="task-summary-text">' + esc(summaryText).replace(/\n/g, "<br>") + "</div>"
            : '<div class="task-summary-text is-empty">No recap text yet.</div>') +
          (followUp
            ? '<div class="task-summary-follow-up">' +
                '<span class="task-summary-follow-up-label">next</span>' +
                esc(followUp).replace(/\n/g, "<br>") +
              "</div>"
            : "") +
        "</div>"
      : "";
    const selector = archiveEligible(it)
      ? '<input type="checkbox" class="task-select" aria-label="' + esc("Select " + it.title) + '"' +
          (selectedRecaps.has(String(it.id)) ? " checked" : "") + (archiveBusy ? " disabled" : "") + ">"
      : '<span class="dot st-' + esc(state) + '"></span>';
    return (
      '<div class="task-row' + archivedClass + '" data-id="' + esc(it.id) + '" data-session="' + esc(it.session_id) + '"' +
        (runId ? ' data-run="' + esc(runId) + '"' : "") + '>' +
        selector +
        '<a class="task-title" href="' + BASE + '/?session=' + encodeURIComponent(it.session_id) + '" title="' + esc(it.title) + '">' + esc(it.title) + "</a>" +
        // Row actions: the sharing lock sits right beside the archive ×.
        '<span class="task-actions">' + shareButtonHtml(it.sharing) + removeBtn + "</span>" +
        (sessionTitle
          ? '<span class="task-session-title" title="Session title: ' + esc(sessionTitle) + '">' +
              esc(sessionTitle) + "</span>"
          : "") +
        '<span class="task-meta">' +
          archivedPill +
          (displayLabel ? '<span class="task-pill task-pill-status">' + esc(displayLabel) + "</span>" : "") +
          recapPill +
          devPill +
          topicPill +
          '<span class="task-agent">' + esc(it.agent) + "</span>" +
          (age ? '<span class="task-age">' + esc(age) + "</span>" : "") +
        "</span>" +
        recapPanel +
      "</div>"
    );
  }

  // The board is rebuilt on every poll. Capture the scroller offsets first
  // (each column body on a wide screen, the board itself, and the page on
  // a phone) and put them back after the innerHTML wipe, or a refresh
  // jumps the operator back to the top of the list they were reading.
  function scrollSnapshot() {
    const cards = {};
    boardEl.querySelectorAll(".task-card").forEach(function (card) {
      const key = card.getAttribute("data-card");
      const body = card.querySelector(".task-card-body");
      if (key && body) cards[key] = body.scrollTop;
    });
    return {
      board: boardEl.scrollTop,
      page: window.scrollY || (document.documentElement && document.documentElement.scrollTop) || 0,
      cards: cards,
    };
  }

  function restoreScroll(saved) {
    if (!saved) return;
    const apply = function () {
      boardEl.scrollTop = saved.board;
      Object.keys(saved.cards || {}).forEach(function (key) {
        const body = boardEl.querySelector(
          '.task-card[data-card="' + key + '"] .task-card-body'
        );
        if (body) body.scrollTop = saved.cards[key];
      });
      window.scrollTo(0, saved.page);
    };
    apply();
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(apply);
  }

  function render() {
    const savedScroll = scrollSnapshot();
    // Reset on every server snapshot, even when a topic filter hides the
    // row through its pending phase. A new turn's recap starts collapsed.
    const recapIds = new Set(items.filter((it) => !recapHidden(it)).map((it) => String(it.id)));
    openRecaps.forEach(function (id) {
      if (!recapIds.has(id)) openRecaps.delete(id);
    });
    const filtered = topicFilter
      ? items.filter((it) => (it.topic || "").toLowerCase().includes(topicFilter))
      : items;
    const eligible = new Set(filtered.filter(archiveEligible).map((it) => String(it.id)));
    selectedRecaps.forEach(function (id) {
      if (!eligible.has(id)) selectedRecaps.delete(id);
    });

    emptyEl.style.display = items.length ? "none" : "";

    const buckets = {};
    const cards = showArchived ? CARDS.concat([ARCHIVED_CARD]) : CARDS;
    for (const c of cards) buckets[c.key] = [];
    for (const it of filtered) buckets[cardFor(it)].push(it);
    for (const key of Object.keys(buckets)) {
      // Most-recent activity first, on NORMALISED epoch-ms (see tsMs): the
      // raw values mix tree epoch ints and store ISO strings, so comparing
      // them directly mis-orders tasks from other days.
      // Archived is the one exception: "last 20 archived" is ordered by the
      // removal stamp, with activity as a rolling-deploy fallback.
      const stamp = (it) => key === "archived"
        ? (it.archived_at || it.last_activity_at) : it.last_activity_at;
      buckets[key].sort((a, b) => tsMs(stamp(b)) - tsMs(stamp(a)));
    }
    // The API enforces the same cap. Keep this client-side guard for rolling
    // deployments where a freshly loaded UI may briefly talk to an older
    // backend that still returns the entire archive.
    if (buckets.archived) buckets.archived = buckets.archived.slice(0, 20);

    boardEl.innerHTML = cards.map((c) => {
      const rows = buckets[c.key];
      const archiveTools = c.key === "completed"
        ? '<div class="task-archive-tools">' +
            '<label><input type="checkbox" class="task-select-all"' +
              (eligible.size && selectedRecaps.size === eligible.size ? " checked" : "") +
              (archiveBusy || !eligible.size ? " disabled" : "") + '> Select all shown</label>' +
            '<button type="button" class="tasks-btn task-archive-selected"' +
              (archiveBusy || !selectedRecaps.size ? " disabled" : "") + '>' +
              (archiveBusy ? "Archiving…" : "Archive selected (" + selectedRecaps.size + ")") + '</button>' +
            '<span class="task-archive-message" role="status">' + esc(archiveMessage) + '</span>' +
          '</div>'
        : "";
      return (
        '<section class="task-card' + (rows.length ? '' : ' is-empty') + '" data-card="' + c.key + '">' +
          '<header class="task-card-header"><span>' + esc(c.label) + '</span>' +
          '<span class="task-card-count">' + rows.length + "</span></header>" + archiveTools +
          '<div class="task-card-body">' +
          (rows.length ? rows.map(rowHtml).join("") : '<div class="task-card-empty">—</div>') +
          "</div>" +
        "</section>"
      );
    }).join("");
    const selectAll = boardEl.querySelector(".task-select-all");
    if (selectAll) selectAll.indeterminate = selectedRecaps.size > 0 && selectedRecaps.size < eligible.size;
    restoreScroll(savedScroll);
  }

  function topicNames(topics) {
    const names = [];
    const seen = new Set();
    (topics || []).forEach(function (t) {
      const name = String((t && t.topic) || "").trim();
      // "general" is the absence of a topic (same rule as the sidebar).
      if (!name || name === "general") return;
      const fold = name.toLowerCase();
      if (seen.has(fold)) return;
      seen.add(fold);
      names.push(name);
    });
    return names;
  }

  function applyTopicOptions(topics) {
    const names = topicNames(topics);
    if (topicOptionsEl) {
      topicOptionsEl.innerHTML = names.map((n) => '<option value="' + esc(n) + '">').join("");
    }
    if (describeTopicInput && !describeTopicInput.disabled) {
      const prev = describeTopicInput.value;
      describeTopicInput.innerHTML =
        '<option value="">Automatic topic</option><option value="general">General</option>' +
        names.filter(n => n.toLowerCase() !== "general").map(n => '<option value="' + esc(n) + '">' + esc(n) + '</option>').join("");
      if (prev === "general" || names.includes(prev)) describeTopicInput.value = prev;
    }
    if (!newTopicInput) return;
    const prev = String(newTopicInput.value || "");
    newTopicInput.innerHTML =
      '<option value="">Topic (optional)</option>' +
      names.map((n) => '<option value="' + esc(n) + '">' + esc(n) + "</option>").join("");
    if (prev && names.indexOf(prev) !== -1) newTopicInput.value = prev;
  }

  async function loadTopicOptions() {
    try {
      const r = await fetch(BASE + "/api/topics", { headers: { Accept: "application/json" } });
      if (!r.ok) return;
      const data = await r.json();
      applyTopicOptions(data.topics || []);
    } catch (_e) {
      // next open / poll retries
    }
  }

  function publishSnapshot(snapshot) {
    const data = snapshot || {};
    applyTopicOptions(data.topics || []);
    // nav.js owns the shared sidebar rendering. Cache the latest value to
    // close the tiny race where this fetch resolves before DOMContentLoaded,
    // then notify an already-initialised nav immediately.
    window.__saTasksSnapshot = data;
    if (
      typeof document.dispatchEvent === "function" &&
      typeof CustomEvent === "function"
    ) {
      document.dispatchEvent(new CustomEvent("sharedagents:tasks-snapshot", {
        detail: data,
      }));
    }
  }

  function refresh() {
    // A slow gateway response must never stack another poll behind it. The old
    // independent timers were the source of the 4–9s request queue.
    if (refreshInFlight) {
      refreshQueued = true;
      return refreshInFlight;
    }
    refreshInFlight = (async function () {
      try {
        let url = BASE + "/api/tasks?bootstrap=1";
        if (showArchived) url += "&include_archived=1";
        const r = await fetch(url, { headers: { Accept: "application/json" } });
        if (!r.ok) return;
        const data = await r.json();
        items = data.items || [];
        render();
        publishSnapshot(data.snapshot || {});
        if (updatedEl) updatedEl.textContent = "updated " + new Date().toLocaleTimeString();
      } catch (e) {
        // transient — next visible poll retries
      } finally {
        refreshInFlight = null;
        if (refreshQueued) {
          refreshQueued = false;
          refresh();
        }
      }
    })();
    return refreshInFlight;
  }

  boardEl.addEventListener("click", async function (e) {
    const shareBtn = e.target.closest(".task-share");
    if (shareBtn) {
      e.stopPropagation();
      try {
        const row = shareBtn.closest(".task-row");
        await window.ShareControl.openSession(row.getAttribute("data-session"), shareBtn);
      } catch (error) {
        shareBtn.title = error.message;
        shareBtn.setAttribute("aria-label", "Sharing: " + error.message);
      }
      return;
    }
    if (e.target.closest(".task-select, .task-select-all")) {
      e.stopPropagation();
      return;
    }
    if (e.target.closest(".task-archive-selected")) {
      e.stopPropagation();
      const ids = Array.from(selectedRecaps);
      if (archiveBusy || !ids.length) return;
      const ok = await confirmCard({
        title: "Archive completed tasks",
        message: "Archive " + ids.length + " selected task(s) with completed recaps? Their chats remain available.",
        okLabel: "Archive",
      });
      if (!ok) return;
      archiveBusy = true;
      archiveMessage = "";
      render();
      try {
        const response = await fetch(BASE + "/api/tasks/archive", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids }),
        });
        if (!response.ok) throw new Error("archive failed");
        const result = await response.json();
        (result.archived_ids || []).forEach((id) => selectedRecaps.delete(String(id)));
        archiveMessage = "Archived " + (result.archived_ids || []).length + " task(s).";
        if ((result.skipped_ids || []).length) archiveMessage += " Changed tasks were skipped.";
        await refresh();
      } catch (_error) {
        archiveMessage = "Archiving failed. Try again.";
      } finally {
        archiveBusy = false;
        render();
      }
      return;
    }
    const removeBtn = e.target.closest(".task-remove");
    if (removeBtn) {
      // Stop the row click (open chat) whether the operator confirms or cancels.
      e.stopPropagation();
      const row = removeBtn.closest(".task-row");
      const id = row.getAttribute("data-id");
      const title = ((row.querySelector(".task-title") || {}).textContent || "").trim()
        || "this task";
      // Same confirm card as conversation delete in app.js — untrack is
      // one click away on touch (× always visible <820px), so require an
      // explicit OK before dropping the board row. Chat is never deleted.
      const ok = await confirmCard({
        title: "Untrack task",
        message: `Untrack "${title}"?\n\nThe chat itself is not deleted.`,
        okLabel: "Untrack",
        danger: true,
      });
      if (!ok) return;
      fetch(BASE + "/api/tasks/" + encodeURIComponent(id), { method: "DELETE" })
        .then(() => refresh());
      return;
    }
    if (e.target.closest(".task-pill-dev")) {
      return;
    }
    // Recap subcard: a click on the row (or its verdict pill) expands /
    // collapses it IN PLACE (no re-render — a re-render would fight the poll)
    // and remembers the state in openRecaps so the next 5 s rebuild keeps it
    // open. stopPropagation keeps the row's click-to-open-chat from firing.
    if (e.target.closest(".task-row") && !e.target.closest(".task-title")) {
      e.stopPropagation();
      const recRow = e.target.closest(".task-row");
      const recapId = String(recRow.getAttribute("data-id") || "");
      const panel = recRow.querySelector(".task-summary-panel");
      if (panel && recapId) {
        const open = !openRecaps.has(recapId);
        if (open) openRecaps.add(recapId);
        else openRecaps.delete(recapId);
        panel.hidden = !open;
        const pill = recRow.querySelector('.task-pill-recap[role="button"]');
        if (pill) pill.setAttribute("aria-expanded", open ? "true" : "false");
      }
      return;
    }
  });

  boardEl.addEventListener("keydown", function (e) {
    const pill = e.target.closest('.task-pill-recap[role="button"]');
    if (pill && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      pill.click();
    }
  });

  boardEl.addEventListener("change", function (e) {
    if (e.target.matches(".task-select")) {
      const row = e.target.closest(".task-row");
      const id = String(row.getAttribute("data-id"));
      if (e.target.checked) selectedRecaps.add(id);
      else selectedRecaps.delete(id);
      render();
    } else if (e.target.matches(".task-select-all")) {
      const checked = e.target.checked;
      items.filter((it) => archiveEligible(it) &&
        (!topicFilter || (it.topic || "").toLowerCase().includes(topicFilter)))
        .forEach(function (it) {
          if (checked) selectedRecaps.add(String(it.id));
          else selectedRecaps.delete(String(it.id));
        });
      render();
    }
  });

  // Keyboard parity: the chevron is a real <button>, so Enter/Space already
  // fires the click handler above.

  topicFilterEl.addEventListener("input", function () {
    topicFilter = topicFilterEl.value.trim().toLowerCase();
    render();
  });

  // Show-archived toggle, default OFF. When on, the poll
  // refetches with ?include_archived=1 and the Archived card appears.
  if (showArchivedEl) {
    showArchivedEl.addEventListener("change", function () {
      showArchived = showArchivedEl.checked;
      refresh();
    });
  }

  newToggleBtn.addEventListener("click", function () {
    const show = newForm.hasAttribute("hidden");
    if (show) {
      newForm.removeAttribute("hidden");
      newToggleBtn.setAttribute("aria-expanded", "true");
      loadTopicOptions();
      newTitleInput.focus();
    } else {
      newForm.setAttribute("hidden", "");
      newToggleBtn.setAttribute("aria-expanded", "false");
    }
  });
  newCancelBtn.addEventListener("click", function () {
    newForm.setAttribute("hidden", "");
    newToggleBtn.setAttribute("aria-expanded", "false");
  });

  newForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    const title = newTitleInput.value.trim();
    const topic = newTopicInput.value.trim();
    const resp = await fetch(BASE + "/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, topic }),
    });
    if (resp.ok) {
      newTitleInput.value = "";
      newTopicInput.value = "";
      newForm.setAttribute("hidden", "");
      newToggleBtn.setAttribute("aria-expanded", "false");
      await refresh();
    }
  });

  document.addEventListener("sharedagents:tasks-refresh", refresh);
  const modelButton = document.getElementById("tasks-summary-model");
  const modelStatus = document.getElementById("tasks-summary-model-status");
  let savedModel = "";
  let recapModels = [];
  if (modelButton) {
    async function loadRecapModel() {
      const response = await fetch(BASE + "/api/tasks/summary-model");
      if (!response.ok) throw new Error("Could not load recap models.");
      const data = await response.json();
      recapModels = data.models || [];
      savedModel = data.selected || "";
      const selected = recapModels.find(model => model.id === savedModel);
      modelButton.textContent = savedModel ? (selected?.label || savedModel) : "Default model";
      if (modelStatus) modelStatus.textContent = savedModel && !selected ? "The selected recap model is unavailable." : "";
    }
    // Reuse the session selector's catalogue, provider rail and search.
    // Recaps restrict the shared picker to configured compatible deployments.
    const recapPicker = new AgentModelDialog({
      title: "Recap model", fixedAgent: "jarvis", allowDefault: true, persistSelection: false,
      filterModels: model => recapModels.some(row => row.id === model.alias),
      async onSave(selection) {
        const response = await fetch(BASE + "/api/tasks/summary-model", {
          method: "PUT", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: selection.alias || selection.model }),
        });
        if (!response.ok) throw new Error("Could not save recap model.");
        const data = await response.json();
        savedModel = data.selected || "";
        modelButton.textContent = savedModel ? selection.label : "Default model";
        if (modelStatus) modelStatus.textContent = "Saved for the next recap.";
      },
    });
    loadRecapModel().catch(error => { if (modelStatus) modelStatus.textContent = error.message; });
    modelButton.addEventListener("click", async function () {
      modelButton.disabled = true;
      try {
        await loadRecapModel();
        await recapPicker.open({ agent: "jarvis", model: savedModel });
      } catch (error) {
        if (modelStatus) modelStatus.textContent = error.message;
      } finally {
        modelButton.disabled = false;
      }
    });
  }
  refresh();
  // Saved in the share popover: repaint that card's control from the saved
  // payload now, then re-read the board so every row matches the server.
  document.addEventListener("sharedagents:sharing-saved", function (e) {
    const detail = (e && e.detail) || {};
    const sid = String(detail.sessionId || "");
    items.forEach(function (it) {
      if (String(it.session_id) === sid && detail.sharing) it.sharing = detail.sharing;
    });
    render();
    refresh();
  });

  setInterval(function () {
    if (!document.hidden) refresh();
  }, 5000);
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) refresh();
  });
})();
