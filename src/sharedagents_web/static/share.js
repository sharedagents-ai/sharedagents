// Chat header share control: a conversation is Private (you + admins) or
// Public (everyone with an account, or the people you pick).
(function (root) {
  "use strict";

  // ── Pure helpers (no DOM; unit-tested under node vm) ───────────────────
  function memberNames(sharing) {
    return ((sharing && sharing.members) || []).map(function (m) {
      return typeof m === "string" ? m : (m.name || m.email || "");
    }).filter(Boolean);
  }

  function shareSummary(sharing) {
    const s = sharing || {};
    if (s.visibility !== "public") return "Private: only you and admins";
    if (s.share_all) return "Public: everyone with an account";
    const names = memberNames(s);
    if (!names.length) return "Public";
    if (names.length <= 2) return "Shared with " + names.join(" & ");
    return "Shared with " + names.slice(0, 2).join(", ") + " +" + (names.length - 2);
  }

  function shareIcon(sharing) {
    if (!sharing || sharing.visibility !== "public") return "lock";
    return sharing.share_all ? "globe" : "people";
  }

  function shareBody(draft) {
    const d = draft || {};
    if (d.visibility !== "public") return { visibility: "private" };
    if (d.share_all) return { visibility: "public", share_all: true, members: [] };
    const seen = {};
    const members = (d.members || []).map(function (m) {
      return String(typeof m === "string" ? m : (m && m.email) || "").trim().toLowerCase();
    }).filter(function (e) {
      if (!e || seen[e]) return false;
      seen[e] = true;
      return true;
    });
    return { visibility: "public", share_all: false, members: members };
  }

  // Whether the header shows the control for this opened conversation.
  function shareVisible(session) {
    if (!session || !session.sharing) return false;
    return session.access === "manage" || session.sharing.visibility === "public";
  }

  root.ShareHelpers = {
    shareSummary: shareSummary,
    shareIcon: shareIcon,
    shareBody: shareBody,
    shareVisible: shareVisible,
  };
  if (typeof document === "undefined") return;

  // ── DOM control ────────────────────────────────────────────────────────
  const ICONS = {
    lock: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/></svg>',
    people: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7"/><path d="M18.5 14.5a6.5 6.5 0 0 1 3 5.5"/></svg>',
    globe: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a14 14 0 0 1 0 18a14 14 0 0 1 0-18z"/></svg>',
  };
  const headerBtn = document.getElementById("share-btn");
  let btn = headerBtn;
  const pop = document.getElementById("share-pop");
  const BASE = document.querySelector('meta[name="sharedagents-base"]')?.content || "";
  const ME = (document.querySelector('meta[name="sharedagents-user-email"]')?.content || "").toLowerCase();
  let directory = null;
  let directoryFailed = false;
  let draft = null;
  let editorSid = "";
  let externalState = null;

  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // ``state`` is state.js's top-level const: a global binding, not a window
  // property, so it is read by name (guarded for pages without it).
  function appState() {
    if (externalState) return externalState;
    return typeof state !== "undefined" ? state : null;
  }

  function currentSession() {
    const st = appState();
    const sid = st && st.currentSessionId;
    return sid && st.sessions ? st.sessions[sid] || null : null;
  }

  function sync() {
    if (!btn) return;
    const st = appState();
    const s = currentSession();
    if (editorSid && (editorSid !== (st && st.currentSessionId) || !s || (draft && s.access !== "manage"))) {
      close();
    }
    if (!shareVisible(s)) {
      btn.hidden = true;
      close();
      return;
    }
    btn.hidden = false;
    paint(btn, s.sharing);
  }

  // Lock / people / globe face of a share button: the chat header and each
  // Tasks card render the same control from the server's sharing payload.
  function face(sharing) {
    const summary = shareSummary(sharing);
    return {
      icon: ICONS[shareIcon(sharing)],
      summary: summary,
      label: "Sharing: " + summary,
      isPublic: Boolean(sharing && sharing.visibility === "public"),
    };
  }

  function paint(el, sharing) {
    const f = face(sharing);
    el.innerHTML = f.icon;
    el.title = f.summary;
    el.setAttribute("aria-label", f.label);
    el.classList.toggle("is-public", f.isPublic);
  }

  function close() {
    editorSid = "";
    draft = null;
    if (pop && !pop.hidden) {
      pop.hidden = true;
      if (btn) btn.setAttribute("aria-expanded", "false");
    }
    externalState = null;
    btn = headerBtn;
  }

  async function loadDirectory() {
    if (directory) return directory;
    directoryFailed = false;
    try {
      const r = await fetch(BASE + "/api/users/directory", { credentials: "same-origin" });
      if (!r.ok) throw new Error("Could not load people.");
      const data = await r.json();
      if (!Array.isArray(data.users)) throw new Error("Could not load people.");
      directory = data.users;
    } catch (_e) {
      directory = null;
      directoryFailed = true;
    }
    return directory;
  }

  function renderReadOnly(s) {
    const by = s.owner_name ? "Shared with you by " + esc(s.owner_name) + ". " : "";
    pop.innerHTML =
      '<p class="share-head">Who can open this chat</p>' +
      '<p class="share-note">' + by + esc(shareSummary(s.sharing)) + ".</p>" +
      (s.access === "view" ? '<p class="share-note">' + esc(s.read_only_reason || "View only.") + "</p>" : "");
  }

  function renderEditor(s) {
    const pub = draft.visibility === "public";
    const people = (directory || []).filter(function (u) {
      return u.email !== ME && u.email !== (s.owner_email || "") &&
        !draft.members.some(function (m) { return m.email === u.email; });
    });
    pop.innerHTML =
      '<p class="share-head">Who can open this chat</p>' +
      '<label class="share-opt"><input type="radio" name="share-vis" value="private"' + (pub ? "" : " checked") + ">" +
      "<span><strong>Private</strong><small>Only you and admins</small></span></label>" +
      '<label class="share-opt"><input type="radio" name="share-vis" value="public"' + (pub ? " checked" : "") + ">" +
      "<span><strong>Public</strong><small>Everyone, or the people you pick</small></span></label>" +
      '<div class="share-public"' + (pub ? "" : " hidden") + ">" +
      '<label class="share-all"><input type="checkbox" id="share-all"' + (draft.share_all ? " checked" : "") +
      "> Everyone with an account</label>" +
      '<div class="share-people"' + (draft.share_all ? " hidden" : "") + ">" +
      '<div class="share-chips">' + draft.members.map(function (m) {
        return '<span class="share-chip">' + esc(m.name || m.email) +
          '<button type="button" data-remove="' + esc(m.email) + '" aria-label="Remove ' +
          esc(m.name || m.email) + '">×</button></span>';
      }).join("") + "</div>" +
      '<select id="share-add" aria-label="Add a person"><option value="">Add a person…</option>' +
      people.map(function (u) {
        return '<option value="' + esc(u.email) + '">' + esc(u.name) + " · " + esc(u.email) + "</option>";
      }).join("") + "</select>" +
      (directoryFailed ? '<p class="share-note">Could not load people.</p>' : "") +
      "</div>" +
      '<p class="share-note">People you share with can read and chat here, within their own permissions.</p>' +
      "</div>" +
      '<p class="share-error" role="alert"></p>' +
      '<div class="share-actions"><button type="button" class="share-action" data-share="cancel">Cancel</button>' +
      '<button type="button" class="share-action is-primary" data-share="save">Save</button></div>';
  }

  async function open() {
    const s = currentSession();
    if (!s || !pop) return;
    editorSid = (appState() || {}).currentSessionId || "";
    pop.hidden = false;
    btn.setAttribute("aria-expanded", "true");
    if (s.access !== "manage") {
      renderReadOnly(s);
      return;
    }
    draft = {
      visibility: s.sharing.visibility,
      share_all: Boolean(s.sharing.share_all),
      members: (s.sharing.members || []).map(function (m) {
        return typeof m === "string" ? { email: m, name: m } : { email: m.email, name: m.name || m.email };
      }),
    };
    const openedDraft = draft;
    renderEditor(s);
    await loadDirectory();
    if (!pop.hidden && draft === openedDraft && editorSid === (appState() || {}).currentSessionId) {
      renderEditor(s);
    }
  }

  async function save() {
    const s = currentSession();
    const st = appState();
    const sid = st && st.currentSessionId;
    if (!s || !sid || !draft || editorSid !== sid || s.access !== "manage") {
      close();
      return;
    }
    const savedDraft = draft;
    const errEl = pop.querySelector(".share-error");
    const body = shareBody(draft);
    if (body.visibility === "public" && !body.share_all && !body.members.length) {
      if (errEl) errEl.textContent = "Pick at least one person, or everyone.";
      return;
    }
    try {
      const r = await fetch(BASE + "/api/sessions/" + encodeURIComponent(sid) + "/sharing", {
        method: "PUT",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await r.json().catch(function () { return {}; });
      if (!r.ok) throw new Error(data.error || ("HTTP " + r.status));
      const savedSession = st.sessions[sid] || s;
      savedSession.sharing = {
        visibility: data.visibility, share_all: data.share_all, members: data.members || [],
      };
      savedSession.visibility = data.visibility;
      if (draft === savedDraft && editorSid === sid) close();
      sync();
      if (typeof renderSessionList === "function") renderSessionList();
      // Pages that show the control outside the chat header (Tasks) refresh
      // from the server; the saved payload rides along for an instant repaint.
      document.dispatchEvent(new CustomEvent("sharedagents:sharing-saved", {
        detail: { sessionId: sid, sharing: savedSession.sharing },
      }));
    } catch (err) {
      if (errEl && draft === savedDraft && editorSid === sid) errEl.textContent = err.message;
    }
  }

  if (pop) {
    if (headerBtn) headerBtn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      if (pop.hidden) open(); else close();
    });
    pop.addEventListener("click", function (ev) {
      ev.stopPropagation();
      const action = ev.target.closest("[data-share]")?.dataset.share;
      if (action === "cancel") close();
      else if (action === "save") save();
      const remove = ev.target.closest("[data-remove]")?.dataset.remove;
      if (remove && draft) {
        draft.members = draft.members.filter(function (m) { return m.email !== remove; });
        renderEditor(currentSession());
      }
    });
    pop.addEventListener("change", function (ev) {
      const t = ev.target;
      if (!draft) return;
      if (t.name === "share-vis") draft.visibility = t.value;
      else if (t.id === "share-all") draft.share_all = t.checked;
      else if (t.id === "share-add" && t.value) {
        const u = (directory || []).find(function (p) { return p.email === t.value; });
        if (u) draft.members.push({ email: u.email, name: u.name });
      } else return;
      renderEditor(currentSession());
    });
    document.addEventListener("click", close);
    document.addEventListener("keydown", function (ev) { if (ev.key === "Escape") close(); });
  }

  async function openSession(sid, anchor) {
    const response = await fetch(BASE + "/api/sessions/" + encodeURIComponent(sid) + "/sharing", {
      credentials: "same-origin",
    });
    if (!response.ok) throw new Error("Could not load task sharing.");
    const sharing = await response.json();
    close();
    externalState = { currentSessionId: sid, sessions: { [sid]: {
      sharing, access: sharing.can_manage ? "manage" : "view",
      owner_email: sharing.owner_email, owner_name: sharing.owner_name,
    } } };
    btn = anchor;
    await open();
  }

  root.ShareControl = { sync: sync, close: close, openSession: openSession, face: face, paint: paint };
})(typeof window !== "undefined" ? window : globalThis);
