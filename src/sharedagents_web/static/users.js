// Users tab — admin-only accounts, roles and grants.
(function () {
  "use strict";
  const BASE = document.querySelector('meta[name="sharedagents-base"]')?.content || "";
  const listEl = document.getElementById("users-list");
  const detailEl = document.getElementById("users-detail");
  const totalsEl = document.getElementById("users-totals");
  const connEl = document.getElementById("users-conn");
  const statusEl = document.getElementById("users-status");
  const dialog = document.getElementById("users-add-dialog");
  const addForm = document.getElementById("users-add-form");

  const DIMS = [
    { key: "agents", title: "Agents", none: "Nothing granted: this user cannot start a chat." },
    { key: "models", title: "Models", none: "Nothing granted: this user cannot send a message." },
    { key: "tools", title: "Tools", none: "No tools: the agent can only talk." },
    { key: "skills", title: "Skills", none: "No skills." },
  ];
  const RISK = {
    host: "acts on this host",
    read: "reads host files",
    spend: "schedules work / spends",
    private: "the operator's private data",
  };

  let users = [];
  let catalog = { agents: [], models: [], tools: [], skills: [], folder_roots: [] };
  let selected = "";
  let draft = null;
  let filters = {};

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function setStatus(text, isError) {
    statusEl.textContent = text || "";
    statusEl.classList.toggle("is-error", Boolean(isError));
  }

  function setConnection(ok) {
    connEl.classList.remove("mon-conn-live", "mon-conn-down", "mon-conn-unknown");
    connEl.classList.add(ok ? "mon-conn-live" : "mon-conn-down");
  }

  function api(path, opts) {
    return fetch(BASE + path, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      ...opts,
    }).then(async function (r) {
      const data = await r.json().catch(function () { return {}; });
      if (!r.ok) throw new Error(data.error || r.statusText || ("HTTP " + r.status));
      return data;
    });
  }

  function userOf(email) {
    return users.find(function (u) { return u.email === email; }) || null;
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function draftFrom(user) {
    return {
      display_name: user.display_name || "",
      role: user.role,
      disabled: Boolean(user.disabled),
      grants: clone(user.grants || {}),
    };
  }

  function isDirty() {
    const user = userOf(selected);
    if (!user || !draft) return false;
    return JSON.stringify(draft) !== JSON.stringify(draftFrom(user));
  }

  function roleLabel(user) {
    if (user.is_admin && user.role !== "admin") return "Admin";
    return user.role === "admin" ? "Admin" : "User";
  }

  function renderList() {
    const admins = users.filter(function (u) { return u.is_admin; }).length;
    totalsEl.innerHTML =
      "<strong>" + users.length + "</strong> account" + (users.length === 1 ? "" : "s") +
      " · <strong>" + admins + "</strong> admin" + (admins === 1 ? "" : "s");
    listEl.innerHTML = users.map(function (u) {
      const initial = (u.name || u.email || "?").charAt(0).toUpperCase();
      const badges = [
        '<span class="users-badge ' + (u.is_admin ? "is-admin" : "is-user") + '">' +
          escapeHtml(roleLabel(u)) + "</span>",
        u.disabled ? '<span class="users-badge is-off">Disabled</span>' : "",
        u.is_self ? '<span class="users-badge is-self">You</span>' : "",
      ].join("");
      return (
        '<button type="button" class="users-row' + (u.email === selected ? " is-selected" : "") +
        (u.disabled ? " is-disabled" : "") + '" data-email="' + escapeHtml(u.email) + '">' +
        '<span class="users-avatar" aria-hidden="true">' + escapeHtml(initial) + "</span>" +
        '<span class="users-who"><span class="users-name">' + escapeHtml(u.name || u.email) +
        '</span><span class="users-email">' + escapeHtml(u.email) + "</span></span>" +
        '<span class="users-badges">' + badges + "</span></button>"
      );
    }).join("");
  }

  function grantedList(dim) {
    const value = draft.grants[dim];
    return value === "*" ? null : (Array.isArray(value) ? value : []);
  }

  function catalogRows(dim) {
    const rows = (catalog[dim] || []).map(function (row) {
      const name = dim === "models" ? row.id : row.name;
      return Object.assign({ key: name }, row);
    });
    const known = new Set(rows.map(function (r) { return r.key; }));
    // Keep a granted name the catalogue no longer lists, so it can be removed.
    (grantedList(dim) || []).forEach(function (name) {
      if (!known.has(name)) rows.push({ key: name, unknown: true });
    });
    return rows;
  }

  function groupOf(dim, row) {
    if (row.unknown) return "No longer listed";
    if (dim === "models") return row.provider || "Other";
    if (dim === "tools") return row.group || "Built-in";
    if (dim === "agents") return row.conversational === false ? "Delegation only" : "Chat agents";
    return "";
  }

  function rowLabel(dim, row) {
    if (dim === "models") return row.display || row.key;
    return row.key;
  }

  function rowNotes(dim, row) {
    const notes = [];
    if (row.unknown) notes.push('<span class="users-note is-warn">not found</span>');
    if (dim === "agents" && row.external) {
      notes.push('<span class="users-note is-warn" title="Brings its own tools and skills and runs outside the folder sandbox: usable only with Tools, Skills and Folders all set to All">own tools · needs All</span>');
    }
    if (dim === "tools" && row.risk && RISK[row.risk]) {
      notes.push('<span class="users-note is-' + escapeHtml(row.risk) + '">' + RISK[row.risk] + "</span>");
    }
    if (dim === "tools" && (row.group === "MCP" || row.key === "acp_call")) {
      notes.push('<span class="users-note is-warn" title="Runs in its own process, outside the folder sandbox">needs All folders</span>');
    }
    if (dim === "models" && row.display && row.display !== row.key) {
      notes.push('<span class="users-note">' + escapeHtml(row.key) + "</span>");
    }
    return notes.join("");
  }

  function renderCard(spec) {
    const dim = spec.key;
    const all = draft.grants[dim] === "*";
    const granted = new Set(grantedList(dim) || []);
    const rows = catalogRows(dim);
    const query = String(filters[dim] || "").trim().toLowerCase();
    const shown = rows.filter(function (row) {
      if (!query) return true;
      return [row.key, row.display, row.description, row.provider, row.group]
        .join(" ").toLowerCase().indexOf(query) !== -1;
    });
    const groups = new Map();
    shown.forEach(function (row) {
      const g = groupOf(dim, row);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(row);
    });
    const count = all ? "All, including future ones" :
      (granted.size + " of " + rows.filter(function (r) { return !r.unknown; }).length);
    let body = "";
    groups.forEach(function (items, g) {
      if (g) body += '<p class="users-group">' + escapeHtml(g) + "</p>";
      body += items.map(function (row) {
        const checked = all || granted.has(row.key);
        return (
          '<label class="users-item' + (row.unknown ? " is-unknown" : "") + '"' +
          (row.description ? ' title="' + escapeHtml(row.description) + '"' : "") + ">" +
          '<input type="checkbox" data-dim="' + dim + '" data-name="' + escapeHtml(row.key) + '"' +
          (checked ? " checked" : "") + (all ? " disabled" : "") + ">" +
          '<span class="users-item-name">' + escapeHtml(rowLabel(dim, row)) + "</span>" +
          rowNotes(dim, row) + "</label>"
        );
      }).join("");
    });
    if (!rows.length) body = '<p class="users-hint">Nothing to grant here yet.</p>';
    else if (!shown.length) body = '<p class="users-hint">No match.</p>';
    const warn = !all && granted.size === 0 ? '<p class="users-card-warn">' + spec.none + "</p>" : "";
    return (
      '<section class="users-card" data-card="' + dim + '">' +
      '<header class="users-card-head"><h3>' + spec.title + "</h3>" +
      '<span class="users-card-count">' + escapeHtml(count) + "</span>" +
      '<label class="users-all"><input type="checkbox" data-all="' + dim + '"' +
      (all ? " checked" : "") + "> All</label></header>" + warn +
      '<input type="search" class="users-filter" data-filter="' + dim + '" placeholder="Filter ' +
      spec.title.toLowerCase() + '" value="' + escapeHtml(filters[dim] || "") + '">' +
      '<div class="users-items">' + body + "</div></section>"
    );
  }

  function externalWarning() {
    const agents = grantedList("agents");
    const externals = (catalog.agents || []).filter(function (a) {
      return a.external && (agents === null || agents.indexOf(a.name) !== -1);
    });
    if (!externals.length) return "";
    if (draft.grants.tools === "*" && draft.grants.skills === "*" && draft.grants.folders === "*") return "";
    return '<p class="users-callout is-warn">' +
      escapeHtml(externals.map(function (a) { return a.name; }).join(", ")) +
      " run their own tools outside the folder sandbox, so this user can only use them once Tools, Skills and Folders are all All.</p>";
  }

  function hostToolWarning() {
    const tools = grantedList("tools");
    const risky = (catalog.tools || []).filter(function (t) {
      return t.risk === "host" && (tools === null || tools.indexOf(t.name) !== -1);
    });
    if (!risky.length) return "";
    const folders = folderList();
    const noFolders = folders !== null && folders.length === 0;
    if (folders === null) {
      return '<p class="users-callout is-danger">' +
        escapeHtml(listNames(risky.map(function (t) { return t.name; }))) +
        " act on this machine with the operator's full access (Folders is All). " +
        "Grant them only to people you would give a shell.</p>";
    }
    return '<p class="users-callout is-warn">' +
      escapeHtml(listNames(risky.map(function (t) { return t.name; }))) +
      " run in a sandbox without network access that shows only the folders granted below" +
      (noFolders ? " — none yet, so they see just a scratch directory." : ".") +
      "</p>";
  }

  function listNames(names) {
    if (names.length < 2) return names.join("");
    return names.slice(0, -1).join(", ") + " and " + names[names.length - 1];
  }

  function folderList() {
    const value = draft.grants.folders;
    return value === "*" ? null : (Array.isArray(value) ? value : []);
  }

  function renderFoldersCard() {
    const folders = folderList();
    const roots = catalog.folder_roots || [];
    const rootHint = roots.length
      ? roots.map(function (r) { return r.path; }).join(" or ")
      : "an approved project directory";
    let rows = "";
    (folders || []).forEach(function (folder, idx) {
      rows +=
        '<div class="users-folder-row">' +
        '<code class="users-folder-path">' + escapeHtml(folder.path) + "</code>" +
        '<span class="users-note">' + (folder.mode === "rw" ? "read and write" : "read only") + "</span>" +
        '<button type="button" class="users-btn" data-folder-remove="' + idx + '">Remove</button></div>';
    });
    if (!folders) {
      rows = '<p class="users-hint">All: shell and file tools run unconfined, with the operator\'s ' +
        "access to this machine. Needed for external agents, MCP tools and ACP agents.</p>";
    } else if (!folders.length) {
      rows = '<p class="users-card-warn">No folders: shell and file tools see only a scratch ' +
        "directory that is discarded after each call.</p>";
    }
    const listId = "users-folder-roots";
    const options = roots.map(function (r) {
      return '<option value="' + escapeHtml(r.path) + '"></option>';
    }).join("");
    return (
      '<section class="users-card users-card-wide" data-card="folders">' +
      '<header class="users-card-head"><h3>Folders</h3>' +
      '<span class="users-card-count">' +
      (folders === null ? "All, unconfined" : (folders.length + " granted")) +
      "</span>" +
      '<label class="users-all"><input type="checkbox" data-all="folders"' +
      (folders === null ? " checked" : "") + "> All</label></header>" +
      (folders === null ? "" :
        '<p class="users-hint">Absolute paths under ' + escapeHtml(rootHint) +
        ". Write access includes create, rename and delete inside that directory. " +
        "The operator home is never grantable.</p>") +
      '<div class="users-folder-rows">' + rows + "</div>" +
      (folders === null ? "" :
        '<div class="users-folder-add">' +
        '<input type="text" data-folder-path placeholder="/path/under/approved/root" list="' +
        listId + '" spellcheck="false">' +
        '<select data-folder-mode><option value="rw">Read and write</option>' +
        '<option value="ro">Read only</option></select>' +
        '<button type="button" class="users-btn" data-action="folder-add">Add</button>' +
        '<datalist id="' + listId + '">' + options + "</datalist></div>") +
      "</section>"
    );
  }

  function renderDetail() {
    const user = userOf(selected);
    if (!user || !draft) {
      detailEl.innerHTML = '<p class="users-empty">Select an account.</p>';
      return;
    }
    const lock = user.locked
      ? '<p class="users-callout">Admin through ' + escapeHtml(user.lock_reason) +
        ". Change it there; role and status are locked here.</p>"
      : "";
    const admin = draft.role === "admin" || (user.is_admin && user.locked);
    const grants = admin
      ? '<p class="users-callout">Admins use every agent, model, tool and skill, see every conversation, and are not folder-sandboxed in this version.</p>'
      : externalWarning() + hostToolWarning() +
        '<div class="users-cards">' + DIMS.map(renderCard).join("") + renderFoldersCard() + "</div>";
    detailEl.innerHTML =
      '<header class="users-detail-head"><div><h2>' + escapeHtml(user.name || user.email) + "</h2>" +
      '<p class="users-email">' + escapeHtml(user.email) + "</p></div>" +
      '<div class="users-badges">' +
      '<span class="users-badge ' + (user.is_admin ? "is-admin" : "is-user") + '">' + roleLabel(user) + "</span>" +
      (user.is_self ? '<span class="users-badge is-self">You</span>' : "") + "</div></header>" + lock +
      '<div class="users-profile">' +
      '<label class="users-field"><span>Name</span><input type="text" data-field="display_name" maxlength="64" value="' +
      escapeHtml(draft.display_name) + '" placeholder="' + escapeHtml(user.name) + '"></label>' +
      '<label class="users-field"><span>Role</span><select data-field="role"' + (user.locked ? " disabled" : "") + ">" +
      '<option value="user"' + (draft.role === "user" ? " selected" : "") + ">User</option>" +
      '<option value="admin"' + (draft.role === "admin" ? " selected" : "") + ">Admin</option></select></label>" +
      '<label class="users-switch"><input type="checkbox" data-field="enabled"' + (draft.disabled ? "" : " checked") +
      (user.locked ? " disabled" : "") + "> Can sign in</label>" +
      '<button type="button" class="users-btn" data-action="password">Reset password…</button>' +
      "</div>" + grants +
      '<footer class="users-actions">' +
      '<span class="users-dirty">' + (isDirty() ? "Unsaved changes" : "") + "</span>" +
      '<button type="button" class="users-btn" data-action="revert"' + (isDirty() ? "" : " disabled") + ">Revert</button>" +
      '<button type="button" class="users-btn users-btn-primary" data-action="save"' + (isDirty() ? "" : " disabled") +
      ">Save</button></footer>";
  }

  function refreshChrome() {
    const dirty = isDirty();
    const saveBtn = detailEl.querySelector('[data-action="save"]');
    const revertBtn = detailEl.querySelector('[data-action="revert"]');
    const note = detailEl.querySelector(".users-dirty");
    if (saveBtn) saveBtn.disabled = !dirty;
    if (revertBtn) revertBtn.disabled = !dirty;
    if (note) note.textContent = dirty ? "Unsaved changes" : "";
  }

  function select(email) {
    if (email === selected) return;
    if (isDirty() && !window.confirm("Discard unsaved changes?")) return;
    selected = email;
    const user = userOf(email);
    draft = user ? draftFrom(user) : null;
    filters = {};
    renderList();
    renderDetail();
  }

  async function load(keep) {
    try {
      const [u, c] = await Promise.all([api("/api/admin/users"), api("/api/admin/catalog")]);
      users = u.users || [];
      catalog = c || catalog;
      setConnection(true);
    } catch (err) {
      setConnection(false);
      setStatus("Could not load accounts: " + err.message, true);
      return;
    }
    const target = keep && userOf(keep) ? keep : (userOf(selected) ? selected : (users[0] || {}).email || "");
    selected = target;
    const user = userOf(target);
    draft = user ? draftFrom(user) : null;
    renderList();
    renderDetail();
  }

  async function save() {
    const user = userOf(selected);
    if (!user || !draft) return;
    setStatus("Saving…");
    const body = {
      display_name: draft.display_name,
      grants: draft.grants,
    };
    if (!user.locked) {
      body.role = draft.role;
      body.disabled = draft.disabled;
    }
    try {
      await api("/api/admin/users/" + encodeURIComponent(user.email), {
        method: "PATCH",
        body: JSON.stringify(body),
      });
      setStatus("Saved " + user.email + ".");
      await load(user.email);
    } catch (err) {
      setStatus(err.message, true);
    }
  }

  async function resetPassword() {
    const user = userOf(selected);
    if (!user) return;
    const password = window.prompt(
      "New password for " + user.email + " (at least 8 characters). They are signed out everywhere.",
      generatePassword(),
    );
    if (!password) return;
    try {
      await api("/api/admin/users/" + encodeURIComponent(user.email) + "/password", {
        method: "POST",
        body: JSON.stringify({ password: password }),
      });
      setStatus("Password reset for " + user.email + ".");
    } catch (err) {
      setStatus(err.message, true);
    }
  }

  function generatePassword() {
    const bytes = new Uint8Array(12);
    crypto.getRandomValues(bytes);
    return btoa(String.fromCharCode.apply(null, bytes))
      .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }

  function addFolder() {
    const pathEl = detailEl.querySelector("[data-folder-path]");
    const modeEl = detailEl.querySelector("[data-folder-mode]");
    if (!pathEl || !draft) return;
    const path = String(pathEl.value || "").trim();
    const mode = modeEl ? modeEl.value : "rw";
    if (!path) return;
    let list = folderList();
    if (list === null) return;
    list = list.filter(function (f) { return f.path !== path; });
    list.push({ path: path, mode: mode });
    list.sort(function (a, b) { return a.path.localeCompare(b.path); });
    draft.grants.folders = list;
    renderDetail();
  }

  function setGrant(dim, name, on) {
    let list = grantedList(dim);
    if (list === null) return;
    list = list.filter(function (n) { return n !== name; });
    if (on) list.push(name);
    list.sort();
    draft.grants[dim] = list;
  }

  listEl.addEventListener("click", function (ev) {
    const row = ev.target.closest(".users-row");
    if (row) select(row.dataset.email);
  });

  detailEl.addEventListener("input", function (ev) {
    const t = ev.target;
    if (t.dataset.field === "display_name") {
      draft.display_name = t.value;
      refreshChrome();
    } else if (t.dataset.filter) {
      filters[t.dataset.filter] = t.value;
      const card = detailEl.querySelector('[data-card="' + t.dataset.filter + '"]');
      const spec = DIMS.find(function (d) { return d.key === t.dataset.filter; });
      if (card && spec) {
        const pos = t.selectionStart;
        card.outerHTML = renderCard(spec);
        const again = detailEl.querySelector('[data-filter="' + spec.key + '"]');
        if (again) { again.focus(); again.setSelectionRange(pos, pos); }
      }
    }
  });

  detailEl.addEventListener("change", function (ev) {
    const t = ev.target;
    if (t.dataset.field === "role") {
      draft.role = t.value;
      renderDetail();
    } else if (t.dataset.field === "enabled") {
      draft.disabled = !t.checked;
      refreshChrome();
    } else if (t.dataset.all) {
      const dim = t.dataset.all;
      draft.grants[dim] = t.checked ? "*" : [];
      renderDetail();
    } else if (t.dataset.dim) {
      setGrant(t.dataset.dim, t.dataset.name, t.checked);
      renderDetail();
    }
  });

  detailEl.addEventListener("click", function (ev) {
    const action = ev.target.closest("[data-action]")?.dataset.action;
    if (action === "save") save();
    else if (action === "revert") {
      const user = userOf(selected);
      draft = user ? draftFrom(user) : null;
      renderDetail();
    } else if (action === "password") resetPassword();
    else if (action === "folder-add") addFolder();
    const remove = ev.target.closest("[data-folder-remove]");
    if (remove) {
      const folders = folderList();
      if (!folders) return;
      folders.splice(Number(remove.dataset.folderRemove), 1);
      draft.grants.folders = folders;
      renderDetail();
    }
  });

  document.getElementById("users-refresh").addEventListener("click", function () {
    if (isDirty() && !window.confirm("Discard unsaved changes?")) return;
    load(selected);
  });

  document.getElementById("users-add").addEventListener("click", function () {
    addForm.reset();
    addForm.querySelector(".users-form-error").textContent = "";
    addForm.elements.password.value = generatePassword();
    dialog.showModal();
    addForm.elements.email.focus();
  });
  addForm.querySelector("[data-cancel]").addEventListener("click", function () { dialog.close(); });
  addForm.querySelector("[data-generate]").addEventListener("click", function () {
    addForm.elements.password.value = generatePassword();
  });
  addForm.addEventListener("submit", async function (ev) {
    ev.preventDefault();
    const errEl = addForm.querySelector(".users-form-error");
    const payload = {
      email: addForm.elements.email.value,
      display_name: addForm.elements.display_name.value,
      password: addForm.elements.password.value,
      role: addForm.elements.role.value,
    };
    try {
      const out = await api("/api/admin/users", { method: "POST", body: JSON.stringify(payload) });
      dialog.close();
      setStatus("Created " + out.user.email + ". Share the password with them yourself.");
      selected = "";
      await load(out.user.email);
    } catch (err) {
      errEl.textContent = err.message;
    }
  });

  window.addEventListener("beforeunload", function (ev) {
    if (isDirty()) { ev.preventDefault(); ev.returnValue = ""; }
  });

  load();
})();
