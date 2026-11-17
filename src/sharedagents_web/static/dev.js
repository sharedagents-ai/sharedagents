/* Dev tab — forge-backed closed loops. Inner laps are free; publish is once. */
(function () {
  "use strict";
  const BASE = document.querySelector('meta[name="sharedagents-base"]')?.content || "";
  const initialId =
    document.querySelector('meta[name="sharedagents-initial-loop"]')?.content || "";
  const REPO_KEY = "sharedagents-dev-repo";

  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  const FORGE_PANEL_LIMIT = 3;

  let view = "active";
  let issueFilter = "all";
  let issueStateFilter = "open";
  let prStateFilter = "open";
  let loops = [];
  let issues = [];
  let issueTotal = 0;
  let pullTotal = 0;
  let selected = initialId;
  let submitting = false;
  let refreshSeq = 0;
  const autoContinue = {}; // id -> sha we already steered

  function selectedRepo() {
    const el = $("dev-repo");
    return (el && el.value.trim()) || "";
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

  function setConn(ok) {
    const el = $("dev-conn");
    if (!el) return;
    el.className = "mon-conn " + (ok ? "mon-conn-ok" : "mon-conn-err");
  }

  function setForge(on) {
    const el = $("dev-forge-pill");
    if (!el) return;
    el.classList.toggle("ok", !!on);
    el.classList.toggle("bad", !on);
    el.title = on ? "Forge API reachable" : "The forge is not listing repos";
  }

  function sessionHref(sid) {
    return BASE + "/?session=" + encodeURIComponent(String(sid || ""));
  }

  function stageFor(loop) {
    if (!loop) return "object";
    if (loop.state === "merged") return "merge";
    if (loop.state === "ci_red" || loop.state === "ready") return "ci";
    if (loop.state === "pr_open" || loop.pr_number) return "pr";
    if ((loop.inner_laps || 0) > 0) return "inner";
    return "object";
  }

  function paintPipeline(loop) {
    const order = ["object", "inner", "pr", "ci", "merge"];
    const cur = stageFor(loop);
    const idx = order.indexOf(cur);
    document.querySelectorAll(".dev-pipeline li").forEach((li) => {
      const i = order.indexOf(li.getAttribute("data-stage"));
      li.classList.toggle("is-on", i >= 0 && i <= idx);
    });
  }

  function card(loop) {
    const live = loop.session_live ? " · live" : "";
    const pr = loop.pr_number ? " · PR #" + loop.pr_number : "";
    const ci = loop.ci_status ? " · CI " + loop.ci_status : "";
    const sel = loop.id === selected ? " is-selected" : "";
    const chat = loop.session_id
      ? '<a class="loop-session-link" href="' + esc(sessionHref(loop.session_id)) + '">session</a>'
      : "";
    const issueLink = loop.issue_url
      ? '<a class="loop-session-link" href="' + esc(loop.issue_url) + '" target="_blank" rel="noopener">issue</a>'
      : "";
    const prLink = loop.pr_url
      ? '<a class="loop-session-link" href="' + esc(loop.pr_url) + '" target="_blank" rel="noopener">PR</a>'
      : "";
    const cont = (loop.state !== "merged" && loop.state !== "aborted")
      ? '<button type="button" class="loop-action-btn" data-act="continue" data-id="' + esc(loop.id) + '">Continue</button>'
      : "";
    const abort = (loop.state !== "merged" && loop.state !== "aborted")
      ? '<button type="button" class="loop-action-btn is-danger" data-act="abort" data-id="' + esc(loop.id) + '">Abort</button>'
      : "";
    return (
      '<article class="dev-card' + sel + '" data-id="' + esc(loop.id) + '">' +
        '<div class="dev-card-top">' +
          '<div class="dev-card-object">' + esc(loop.object) + '</div>' +
          '<span class="dev-badge ' + esc(loop.state) + '">' + esc(loop.state) + '</span>' +
        "</div>" +
        '<div class="dev-card-meta">' + esc(loop.repo) + " · " + esc(loop.branch) +
          " · " + (loop.inner_laps || 0) + " laps" + live + pr + ci + "</div>" +
        '<div class="dev-card-done">' + esc(loop.done_when) + "</div>" +
        '<div class="dev-card-actions">' + chat + issueLink + prLink + cont + abort + "</div>" +
      "</article>"
    );
  }

  function issueCard(issue) {
    const has = !!issue.has_session;
    const badge = has
      ? '<span class="dev-badge has-session">has session</span>'
      : '<span class="dev-badge no-session">no session</span>';
    const url = issue.html_url || "";
    const open = url
      ? '<a class="loop-session-link" href="' + esc(url) + '" target="_blank" rel="noopener">Open issue</a>'
      : "";
    const session = has && issue.loop && issue.loop.session_id
      ? '<a class="loop-session-link" href="' + esc(sessionHref(issue.loop.session_id)) + '">session</a>'
      : "";
    const close = (issue.state || "open") === "closed"
      ? ""
      : '<button type="button" class="loop-action-btn is-danger" data-act="close-issue" data-number="' +
        esc(issue.number) + '">Close</button>';
    const labels = (issue.labels || []).slice(0, 3).join(", ");
    return (
      '<article class="dev-issue" data-number="' + esc(issue.number) + '">' +
        '<div class="dev-issue-top">' +
          '<div class="dev-card-object">#' + esc(issue.number) + " " + esc(issue.title) + "</div>" +
          badge +
        "</div>" +
        '<div class="dev-issue-meta">' +
          esc(issue.state || "open") +
          (issue.user ? " · " + esc(issue.user) : "") +
          (labels ? " · " + esc(labels) : "") +
        "</div>" +
        '<div class="dev-issue-actions">' + open + session + close + "</div>" +
      "</article>"
    );
  }

  function moreLine(shown, total, noun) {
    const extra = Math.max(0, (Number(total) || 0) - (Number(shown) || 0));
    if (!extra) return "";
    return '<div class="dev-forge-more">+' + extra + " more " + noun + "</div>";
  }

  function renderLoops() {
    /* Closed loops was retired — Runs is the only change list. */
  }

  function renderIssues() {
    const el = $("dev-issues");
    if (!el) return;
    const shown = (issues || []).slice(0, FORGE_PANEL_LIMIT);
    if (!shown.length) {
      el.innerHTML = '<div class="loops-loading">No issues match this filter.</div>';
      return;
    }
    el.innerHTML = shown.map(issueCard).join("") + moreLine(shown.length, issueTotal, "issues");
  }

  function renderPulls(pulls, configured) {
    const el = $("dev-prs");
    if (!configured) {
      el.innerHTML = '<div class="loops-loading">FORGE_TOKEN not set — forge overlay off.</div>';
      return;
    }
    const rows = (pulls || []).slice(0, FORGE_PANEL_LIMIT);
    if (!rows.length) {
      el.innerHTML = '<div class="loops-loading">No ' + esc(prStateFilter === "all" ? "" : prStateFilter + " ") + 'PRs.</div>';
      return;
    }
    el.innerHTML = rows.map((p) => (
      '<a class="dev-pr" href="' + esc(p.html_url) + '" target="_blank" rel="noopener">' +
        '<div class="dev-pr-top"><div class="dev-card-object">#' + esc(p.number) + " " + esc(p.title) + "</div>" +
        '<span class="dev-badge">' + esc(p.merged ? "merged" : (p.state || "open")) + "</span></div>" +
        '<div class="dev-pr-meta">' + esc(p.head) + (p.user ? " · " + esc(p.user) : "") + "</div>" +
      "</a>"
    )).join("") + moreLine(rows.length, pullTotal, "PRs");
  }

  // The server is the sole state-machine authority.  This renderer consumes
  // its versioned projection and never computes a legal transition locally.

  function rememberRepo(name) {
    const repo = String(name || "").trim();
    if (!repo) return;
    try { localStorage.setItem(REPO_KEY, repo); } catch (_) { /* ignore */ }
  }

  function fillRepoSelect(repos) {
    const el = $("dev-repo");
    if (!el) return;
    const names = (repos || []).map((r) => r.full_name).filter(Boolean);
    if (!names.length && !el.value) {
      el.innerHTML = '<option value="">No forge repos listed</option>';
      return;
    }
    let saved = "";
    try { saved = (localStorage.getItem(REPO_KEY) || "").trim(); } catch (_) { saved = ""; }
    // A ?run= load may have already pointed the select at that run's repo.
    // Rebuilding from the forge must not wipe it back to the last localStorage pick
    // (and must keep a repo the list omitted).
    const live = (el.value || "").trim();
    const current = live || (names.includes(saved) ? saved : (names[0] || ""));
    if (current && !names.includes(current)) names.unshift(current);
    el.innerHTML = names.map((n) =>
      '<option value="' + esc(n) + '"' + (n === current ? " selected" : "") + ">" + esc(n) + "</option>"
    ).join("");
    el.value = current;
    rememberRepo(current);
    // sdlc bind may have already fetched runs against an empty select.
    // A change event reloads the list for the repo the forge just filled.
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  async function loadRepos() {
    try {
      const data = await api("/api/dev/repos");
      setForge(!!data.forge_configured && !data.error);
      const pill = $("dev-forge-pill");
      if (pill && data.error) pill.title = data.error;
      else if (pill && data.forge_url) pill.title = data.forge_url;
      if (pill && data.forge_kind) pill.textContent = data.forge_kind === "github" ? "GitHub" : "Gitea";
      fillRepoSelect(data.repos || []);
    } catch (err) {
      setForge(false);
      const pill = $("dev-forge-pill");
      if (pill) pill.title = (err && err.message) || "repo list failed";
      fillRepoSelect([]);
    }
  }

  async function refresh() {
    const seq = ++refreshSeq;
    try {
      const repo = selectedRepo();
      setConn(true);
      if (!repo) {
        issues = [];
        issueTotal = 0;
        pullTotal = 0;
        renderIssues();
        renderPulls([], false);
        if ($("dev-updated")) $("dev-updated").textContent = new Date().toLocaleTimeString();
        return;
      }
      const iss = await api(
        "/api/dev/issues?repo=" + encodeURIComponent(repo) +
        "&state=" + encodeURIComponent(issueStateFilter) +
        "&session=" + encodeURIComponent(issueFilter) +
        "&limit=" + FORGE_PANEL_LIMIT
      );
      if (seq !== refreshSeq || selectedRepo() !== repo) return;
      issues = iss.issues || [];
      setForge(!!iss.forge_configured);
      issueTotal = Number(iss.total != null ? iss.total : issues.length) || issues.length;
      renderIssues();
      const prs = await api(
        "/api/dev/pulls?repo=" + encodeURIComponent(repo) +
        "&state=" + encodeURIComponent(prStateFilter) +
        "&limit=" + FORGE_PANEL_LIMIT
      );
      if (seq !== refreshSeq || selectedRepo() !== repo) return;
      pullTotal = Number(prs.total != null ? prs.total : (prs.pulls || []).length) || 0;
      renderPulls(prs.pulls || [], prs.forge_configured);
      if ($("dev-updated")) $("dev-updated").textContent = new Date().toLocaleTimeString();
    } catch (e) {
      if (seq !== refreshSeq) return;
      setConn(false);
    }
  }

  async function closeIssue(number) {
    if (submitting) return;
    const n = String(number || "").trim();
    if (!n) return;
    if (!window.confirm(
      "Close issue #" + n + "?\n\nThis marks the forge issue closed. It does not delete a Dev run or merge a PR."
    )) return;
    submitting = true;
    try {
      await api("/api/dev/issues/" + encodeURIComponent(n) + "/close", {
        method: "POST",
        body: JSON.stringify({ repo: selectedRepo() }),
      });
      await refresh();
    } catch (e) {
      alert(e.message || "failed to close issue");
    } finally {
      submitting = false;
    }
  }

  if ($("dev-refresh")) $("dev-refresh").addEventListener("click", refresh);

  $("dev-repo").addEventListener("change", () => {
    rememberRepo(selectedRepo());
    refresh();
  });

  const issueStateFilterEl = document.getElementById("dev-issue-state-filter");
  if (issueStateFilterEl) {
    issueStateFilterEl.addEventListener("click", (ev) => {
      const tab = ev.target.closest("[data-issue-state]");
      if (!tab) return;
      issueStateFilter = tab.getAttribute("data-issue-state") || "open";
      document.querySelectorAll("#dev-issue-state-filter .dev-chip").forEach((t) => {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-pressed", t === tab ? "true" : "false");
      });
      refresh();
    });
  }

  document.getElementById("dev-issue-filter").addEventListener("click", (ev) => {
    const tab = ev.target.closest("[data-session]");
    if (!tab) return;
    issueFilter = tab.getAttribute("data-session") || "all";
    document.querySelectorAll("#dev-issue-filter .dev-chip").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-pressed", t === tab ? "true" : "false");
    });
    refresh();
  });

  const prFilterEl = document.getElementById("dev-pr-filter");
  if (prFilterEl) {
    prFilterEl.addEventListener("click", (ev) => {
      const tab = ev.target.closest("[data-pr-state]");
      if (!tab) return;
      prStateFilter = tab.getAttribute("data-pr-state") || "open";
      document.querySelectorAll("#dev-pr-filter .dev-chip").forEach((t) => {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-pressed", t === tab ? "true" : "false");
      });
      refresh();
    });
  }

  if ($("dev-issues")) $("dev-issues").addEventListener("click", (ev) => {
    const act = ev.target.closest('[data-act="close-issue"]');
    if (!act) return;
    ev.preventDefault();
    closeIssue(act.getAttribute("data-number"));
  });

  loadRepos().then(refresh);
  setInterval(refresh, 12000);
})();
