(function () {
  const BASE =
    document.querySelector('meta[name="sharedagents-base"]')?.content || "";
  const listEl = document.getElementById("channels-list");
  const totalsEl = document.getElementById("channels-totals");
  const connEl = document.getElementById("channels-conn");
  const statusEl = document.getElementById("channels-status");
  const refreshBtn = document.getElementById("channels-refresh");
  let lastData = { channels: [], agents: [], models: [] };

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function setConnection(ok) {
    if (!connEl) return;
    connEl.classList.remove("mon-conn-live", "mon-conn-down", "mon-conn-unknown");
    connEl.classList.add(ok ? "mon-conn-live" : "mon-conn-down");
  }

  function setStatus(text, isError) {
    if (!statusEl) return;
    statusEl.textContent = text || "";
    statusEl.classList.toggle("is-error", Boolean(isError));
  }

  function api(path, opts) {
    return fetch(BASE + path, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      ...opts,
    }).then(async function (r) {
      const data = await r.json().catch(function () { return {}; });
      if (!r.ok) {
        throw new Error(data.error || r.statusText || ("HTTP " + r.status));
      }
      return data;
    });
  }

  function channelOf(id) {
    return (lastData.channels || []).find(function (ch) { return ch.id === id; }) || {};
  }

  function suggestionsFor(channelId, kind) {
    const ch = channelOf(channelId);
    return kind === "group" ? (ch.suggest_groups || []) : (ch.suggest_people || []);
  }

  function matchesSuggest(row, query) {
    const hay = [row.id, row.label, row.phone].join(" ").toLowerCase();
    return hay.indexOf(query) !== -1;
  }

  function accessMode(row) {
    return row && row.send === false ? "readonly" : "readwrite";
  }

  function accessOptions(selected) {
    const value = selected || "readwrite";
    return [
      ["readonly", "readonly"],
      ["readwrite", "read/write"],
    ].map(function (pair) {
      return "<option value=\"" + pair[0] + "\"" +
        (pair[0] === value ? " selected" : "") + ">" + pair[1] + "</option>";
    }).join("");
  }

  function flagsFromAccess(value) {
    return { read: true, send: value !== "readonly" };
  }

  function replyOptions(selected, kind) {
    const person = kind !== "group";
    let value = selected || (person ? "any" : "to_me");
    if (person && value === "to_me") value = "any";
    const pairs = person
      ? [["any", "any message"], ["mention", "@"]]
      : [
        ["to_me", "only messages sent to this account"],
        ["any", "any message"],
        ["mention", "@"],
      ];
    return pairs.map(function (pair) {
      return "<option value=\"" + pair[0] + "\"" +
        (pair[0] === value ? " selected" : "") + ">" + pair[1] + "</option>";
    }).join("");
  }

  function replySummary(row, kind) {
    const rep = row.reply || (kind === "group" ? "to_me" : "any");
    if (rep === "mention") {
      return row.mention ? "@ " + row.mention : "@ mention";
    }
    if (rep === "to_me") return "to this account";
    return "any message";
  }

  function routeSummary(row, channelId) {
    const agent = row.agent || defaultAgentFor(channelId);
    const model = row.model || "";
    return model ? (agent + " · " + model) : agent;
  }

  function defaultAgentFor(channelId) {
    return channelOf(channelId).default_agent || "jarvis";
  }

  function mentionField(reply, mention, attrs) {
    const hidden = reply === "mention" ? "" : " hidden";
    return (
      "<label class=\"channel-mention-wrap\"" + hidden + ">" +
      "<span class=\"channel-mention-label\">Mention</span>" +
      "<input class=\"channel-mention\" data-action=\"mention\" " +
      "placeholder=\"@bot @helper\" value=\"" + escapeHtml(mention || "") + "\" " +
      "aria-label=\"Mention token, for example @bot or @helper\" " +
      attrs + " autocomplete=\"off\">" +
      "</label>"
    );
  }

  function routeFields(row, channelId, extraClass) {
    const agent = row.agent || defaultAgentFor(channelId);
    const model = row.model || "";
    const cls = extraClass ? " " + extraClass : "";
    const logo = (typeof pickerAgentLogo === "function")
      ? pickerAgentLogo(agent, "sm", { eager: true })
      : "";
    return (
      "<div class=\"channel-route" + cls + "\" data-agent=\"" + escapeHtml(agent) +
      "\" data-model=\"" + escapeHtml(model) + "\">" +
      "<button type=\"button\" class=\"channel-amp-trigger seg seg-agent seg-agent-btn\" " +
      "aria-haspopup=\"dialog\" aria-expanded=\"false\" title=\"Change agent or model\">" +
      logo +
      "<span class=\"channel-amp-agent\">" + escapeHtml(agent) + "</span>" +
      (model ? "<span class=\"seg-model\">" + escapeHtml(model) + "</span>" : "") +
      "<span class=\"seg-caret\">▾</span></button></div>"
    );
  }

  function fieldBlock(label, inner) {
    return (
      "<div class=\"channel-field\">" +
      "<span class=\"channel-field-label\">" + escapeHtml(label) + "</span>" +
      inner +
      "</div>"
    );
  }

  function whoBlock(row) {
    const name = row.label || "—";
    const ident = (row.phone && row.phone !== row.label) ? row.phone : (row.id || "");
    return (
      "<div class=\"channel-who\">" +
      "<span class=\"channel-name\">" + escapeHtml(name) + "</span>" +
      "<span class=\"channel-id\" title=\"" + escapeHtml(row.id || ident) + "\">" +
      escapeHtml(ident) + "</span></div>"
    );
  }

  function renderRows(channelId, kind, rows) {
    if (!rows.length) {
      return (
        '<div class="channel-table">' +
        '<div class="channel-table-head">' +
        '<div class="channel-th th-who">Contact</div>' +
        '<div class="channel-th th-access">Access</div>' +
        '<div class="channel-th th-reply">Reply</div>' +
        '<div class="channel-th th-route">Agent / model</div>' +
        '<div class="channel-th th-exp"></div>' +
        '</div>' +
        '<p class="channel-empty" data-empty="' +
        escapeHtml(channelId) + "-" + escapeHtml(kind) +
        '">No one on the whitelist yet.</p>' +
        '</div>'
      );
    }
    const cards = rows.map(function (row) {
      const acc = accessMode(row);
      const accText = acc === "readonly" ? "readonly" : "read/write";
      return (
        "<article class=\"channel-entry\" data-channel=\"" + escapeHtml(channelId) +
        "\" data-kind=\"" + escapeHtml(kind) +
        "\" data-id=\"" + escapeHtml(row.id) +
        "\" data-label=\"" + escapeHtml(row.label || "") +
        "\" data-allowed=\"1\">" +
        "<div class=\"channel-entry-row\" data-action=\"toggle-row\" title=\"Click to edit\">" +
        whoBlock(row) +
        "<div class=\"channel-cell cell-access\"><span class=\"channel-pill pill-access\">" + escapeHtml(accText) + "</span></div>" +
        "<div class=\"channel-cell cell-reply\"><span class=\"channel-pill pill-reply\">" + escapeHtml(replySummary(row, kind)) + "</span></div>" +
        "<div class=\"channel-cell cell-route\"><span class=\"channel-pill pill-route\">" + escapeHtml(routeSummary(row, channelId)) + "</span></div>" +
        "<div class=\"channel-cell cell-expand\"><span class=\"channel-chevron\">▾</span></div>" +
        "</div>" +
        "<div class=\"channel-entry-fields\" hidden>" +
        fieldBlock("Access",
          "<select class=\"channel-access\" data-action=\"access\">" +
          accessOptions(acc) + "</select>") +
        fieldBlock("Reply when",
          "<select class=\"channel-reply\" data-action=\"reply\">" +
          replyOptions(row.reply, kind) + "</select>" +
          mentionField(row.reply, row.mention, "")) +
        fieldBlock("Agent / model", routeFields(row, channelId, "")) +
        "<div class=\"channel-row-actions\">" +
        "<button type=\"button\" class=\"channel-revoke\" data-action=\"revoke\">Remove</button>" +
        "</div>" +
        "</div></article>"
      );
    }).join("");
    return (
      "<div class=\"channel-table\">" +
      "<div class=\"channel-table-head\">" +
      "<div class=\"channel-th th-who\">Contact</div>" +
      "<div class=\"channel-th th-access\">Access</div>" +
      "<div class=\"channel-th th-reply\">Reply</div>" +
      "<div class=\"channel-th th-route\">Agent / model</div>" +
      "<div class=\"channel-th th-exp\"></div>" +
      "</div>" +
      "<div class=\"channel-entries\">" + cards + "</div>" +
      "</div>"
    );
  }

  function waLinkPanel(ch) {
    if (ch.id !== "whatsapp") return "";
    if (ch.linked) {
      return (
        "<p class=\"channel-qr-status\">Linked as " +
        escapeHtml(ch.account || "this phone") +
        ". Remove the sharedagents device on the phone, then logout, to pair again.</p>"
      );
    }
    return (
      "<div class=\"channel-qr\">" +
      "<button type=\"button\" class=\"channel-qr-btn\" data-action=\"wa-qr\">" +
      "Generate QR</button>" +
      "<p class=\"channel-qr-detail\">Scan in WhatsApp → Linked devices. " +
      "The code expires in about 20 seconds.</p>" +
      "<img class=\"channel-qr-img\" alt=\"WhatsApp pairing QR\" hidden>" +
      "</div>"
    );
  }

  function addForm(channelId, kind, placeholder) {
    const defaultReply = kind === "group" ? "mention" : "any";
    const labelKind = kind === "group" ? "group" : "contact";
    return (
      "<div class=\"channel-add-row\" data-channel=\"" + escapeHtml(channelId) + "\" data-kind=\"" + escapeHtml(kind) + "\">" +
      "<button type=\"button\" class=\"channel-add-trigger\" data-action=\"toggle-add\" title=\"Add a new channel\">" +
      "<span class=\"channel-plus-symbol\">+</span> Add " + escapeHtml(labelKind) +
      "</button>" +
      "<form class=\"channel-add\" hidden data-channel=\"" + escapeHtml(channelId) +
      "\" data-kind=\"" + escapeHtml(kind) + "\">" +
      fieldBlock("Search",
        "<div class=\"channel-ac\">" +
        "<input name=\"id\" required placeholder=\"" + escapeHtml(placeholder) +
        "\" autocomplete=\"off\" aria-autocomplete=\"list\">" +
        "<ul class=\"channel-ac-list\" hidden></ul></div>") +
      fieldBlock("Label",
        "<input name=\"label\" placeholder=\"optional\" autocomplete=\"off\">") +
      fieldBlock("Access",
        "<select name=\"access\">" + accessOptions("readwrite") + "</select>") +
      fieldBlock("Reply when",
        "<select name=\"reply\">" + replyOptions(defaultReply, kind) + "</select>" +
        mentionField(defaultReply, "", "name=\"mention\"")) +
      fieldBlock("Agent / model",
        routeFields({ agent: "", model: "" }, channelId, "channel-add-route")) +
      "<div class=\"channel-add-actions\">" +
      "<button type=\"submit\" class=\"channel-btn-primary\">Add</button>" +
      "<button type=\"button\" class=\"channel-btn-cancel\" data-action=\"cancel-add\">Cancel</button>" +
      "</div></form>" +
      "</div>"
    );
  }

  function render(data) {
    lastData = data;
    const channels = data.channels || [];
    let allowedPeople = 0;
    let allowedGroups = 0;
    listEl.innerHTML = channels.map(function (ch) {
      const people = ch.people || [];
      const groups = ch.groups || [];
      allowedPeople += people.length;
      allowedGroups += groups.length;
      const personHint = "Search a contact to add";
      const groupHint = "Search a group to add";
      return (
        "<article class=\"channel-card\" data-channel=\"" + escapeHtml(ch.id) + "\">" +
        "<h2>" + escapeHtml(ch.name) +
        (ch.linked ? " <span class=\"channel-linked\">linked</span>" : "") +
        "</h2>" +
        "<p class=\"channel-notes\">" + escapeHtml(ch.notes || "") + "</p>" +
        waLinkPanel(ch) +
        "<section class=\"channel-section\" data-kind=\"person\">" +
        "<h3>People <span class=\"channel-count\">" + people.length + "</span></h3>" +
        renderRows(ch.id, "person", people) +
        addForm(ch.id, "person", personHint) +
        "</section>" +
        "<section class=\"channel-section\" data-kind=\"group\">" +
        "<h3>Groups <span class=\"channel-count\">" + groups.length + "</span></h3>" +
        renderRows(ch.id, "group", groups) +
        addForm(ch.id, "group", groupHint) +
        "</section></article>"
      );
    }).join("");
    if (totalsEl) {
      totalsEl.innerHTML =
        "<strong>" + allowedPeople + "</strong> people · <strong>" +
        allowedGroups + "</strong> groups on the whitelist";
    }
    if (typeof bindModelSheetControls === "function") bindModelSheetControls();
  }

  function paintRoute(box, agent, model) {
    if (!box) return;
    box.setAttribute("data-agent", agent || "");
    box.setAttribute("data-model", model || "");
    const btn = box.querySelector(".channel-amp-trigger");
    if (!btn) return;
    const logo = (typeof pickerAgentLogo === "function")
      ? pickerAgentLogo(agent || "jarvis", "sm", { eager: true })
      : "";
    btn.innerHTML =
      logo +
      "<span class=\"channel-amp-agent\">" + escapeHtml(agent || "jarvis") + "</span>" +
      (model ? "<span class=\"seg-model\">" + escapeHtml(model) + "</span>" : "") +
      "<span class=\"seg-caret\">▾</span>";
  }

  function rowRoute(row) {
    const box = row && row.querySelector(".channel-route");
    return {
      agent: ((box && box.getAttribute("data-agent")) || "").trim(),
      model: ((box && box.getAttribute("data-model")) || "").trim(),
    };
  }

  function openRouteSheet(box) {
    if (!box || typeof openModelSheet !== "function") return;
    const agent = box.getAttribute("data-agent") || defaultAgentFor(
      (box.closest("[data-channel]") || {}).getAttribute("data-channel") || ""
    );
    const model = box.getAttribute("data-model") || "";
    if (typeof ensureModelSheetState === "function") ensureModelSheetState();
    state.pickerAgent = agent;
    state.currentAgent = agent;
    if (model) state.pickerModels[agent] = model;
    state.modelSheet.effective = cloneModelSelection({ agent: agent, alias: model });
    state.modelSheet.draft = cloneModelSelection(state.modelSheet.effective);
    window.sharedagentsModelSheetApply = function (draft) {
      const nextAgent = (draft && draft.agent) || agent;
      const nextModel = (draft && (draft.alias || draft.wire_id)) || "";
      paintRoute(box, nextAgent, nextModel);
      const row = box.closest(".channel-entry");
      if (!row) return Promise.resolve(true);
      return onRouteChange(row).then(function () { return true; });
    };
    const trigger = box.querySelector(".channel-amp-trigger");
    if (trigger) {
      trigger.id = "agent-model-trigger";
      trigger.setAttribute("aria-expanded", "true");
    }
    openModelSheet();
    if (!(state.agents || []).length && typeof loadAgents === "function") {
      Promise.resolve(loadAgents()).then(function () {
        if (typeof renderAgentSelect === "function") renderAgentSelect();
      });
    }
  }

  function fillSuggestList(input) {
    const form = input.closest("form.channel-add");
    const list = form && form.querySelector(".channel-ac-list");
    if (!form || !list) return;
    const query = (input.value || "").trim().toLowerCase();
    list.innerHTML = "";
    if (query.length < 1) {
      list.hidden = true;
      return;
    }
    const hits = suggestionsFor(
      form.getAttribute("data-channel"),
      form.getAttribute("data-kind")
    ).filter(function (row) { return matchesSuggest(row, query); }).slice(0, 8);
    if (!hits.length) {
      list.hidden = true;
      return;
    }
    hits.forEach(function (row) {
      const li = document.createElement("li");
      li.className = "channel-ac-item";
      li.setAttribute("data-id", row.id);
      li.setAttribute("data-label", row.label || "");
      li.innerHTML =
        "<span class=\"channel-ac-name\">" + escapeHtml(row.label || row.phone || row.id) + "</span>" +
        "<span class=\"channel-ac-id\">" + escapeHtml(row.phone || row.id) + "</span>";
      list.appendChild(li);
    });
    list.hidden = false;
  }

  function load() {
    return api("/api/channels").then(function (data) {
      render(data);
      setConnection(true);
      return data;
    }).catch(function (err) {
      setConnection(false);
      setStatus(err.message || "Failed to load channels", true);
    });
  }

  function payloadFromForm(form) {
    const reply = (form.elements.reply.value || "").trim();
    const mention = reply === "mention" ? ((form.elements.mention && form.elements.mention.value) || "").trim() : "";
    return {
      kind: form.getAttribute("data-kind"),
      id: (form.elements.id.value || "").trim(),
      label: (form.elements.label.value || "").trim(),
      reply: reply,
      mention: mention,
      agent: ((form.querySelector(".channel-route") || {}).getAttribute("data-agent") || "").trim(),
      model: ((form.querySelector(".channel-route") || {}).getAttribute("data-model") || "").trim(),
      ...flagsFromAccess((form.elements.access && form.elements.access.value) || "readwrite"),
    };
  }

  function onWaQr(card) {
    const btn = card.querySelector(".channel-qr-btn");
    const img = card.querySelector(".channel-qr-img");
    const detail = card.querySelector(".channel-qr-detail");
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Generating…";
    }
    return api("/api/channels/whatsapp/qr", {
      method: "POST",
      body: "{}",
    }).then(function (data) {
      if (data.linked) {
        setStatus("WhatsApp already linked as " + (data.account || "") + ".");
        return load();
      }
      if (detail) detail.textContent = data.detail || "";
      if (data.image && img) {
        img.src = data.image;
        img.hidden = false;
        if (btn) btn.textContent = "Refresh QR";
        setStatus("Scan the WhatsApp QR now — it expires in about 20 seconds.");
      } else if (img) {
        img.removeAttribute("src");
        img.hidden = true;
        if (btn) btn.textContent = "Generate QR";
        setStatus(data.detail || "QR not ready", true);
      }
    }).finally(function () {
      if (btn) btn.disabled = false;
    });
  }

  function onAdd(form) {
    const channel = form.getAttribute("data-channel");
    const body = payloadFromForm(form);
    return api("/api/channels/" + encodeURIComponent(channel) + "/allowlist", {
      method: "POST",
      body: JSON.stringify(body),
    }).then(function (data) {
      render(data);
      setStatus("Allowed " + (body.label || body.id) + " on " + channel + ".");
    });
  }

  function onRevoke(channel, kind, ident) {
    const qs = "?kind=" + encodeURIComponent(kind) + "&id=" + encodeURIComponent(ident);
    return api("/api/channels/" + encodeURIComponent(channel) + "/allowlist" + qs, {
      method: "DELETE",
    }).then(function (data) {
      render(data);
      setStatus("Removed " + ident + " from " + channel + ".");
    });
  }

  function onRuleChange(row) {
    const channel = row.getAttribute("data-channel");
    const kind = row.getAttribute("data-kind");
    const ident = row.getAttribute("data-id");
    const reply = (row.querySelector(".channel-reply") || {}).value || "any";
    toggleMentionWrap(row, reply);
    const mentionEl = row.querySelector(".channel-mention");
    const mention = reply === "mention" ? ((mentionEl && mentionEl.value) || "").trim() : "";
    const accessEl = row.querySelector(".channel-access");
    const flags = flagsFromAccess((accessEl && accessEl.value) || "readwrite");
    return api("/api/channels/" + encodeURIComponent(channel) + "/allowlist", {
      method: "PATCH",
      body: JSON.stringify({
        kind: kind,
        id: ident,
        reply: reply,
        mention: mention,
        read: flags.read,
        send: flags.send,
        ...rowRoute(row),
      }),
    }).then(function (data) {
      lastData = data;
      // Update pills in place to avoid re-rendering and closing the expanded edit fields
      const accPill = row.querySelector(".pill-access");
      if (accPill) accPill.textContent = flags.send ? "read/write" : "readonly";
      const repPill = row.querySelector(".pill-reply");
      if (repPill) repPill.textContent = replySummary({ reply: reply, mention: mention }, kind);
      setStatus("Updated reply rule for " + ident + ".");
    });
  }

  function onRouteChange(row) {
    const channel = row.getAttribute("data-channel");
    const kind = row.getAttribute("data-kind");
    const ident = row.getAttribute("data-id");
    return api("/api/channels/" + encodeURIComponent(channel) + "/allowlist", {
      method: "PATCH",
      body: JSON.stringify({
        kind: kind,
        id: ident,
        ...rowRoute(row),
      }),
    }).then(function (data) {
      lastData = data;
      const routePill = row.querySelector(".pill-route");
      if (routePill) routePill.textContent = routeSummary(rowRoute(row), channel);
      setStatus("Updated agent/model for " + ident + ".");
    });
  }

  function toggleMentionWrap(root, reply) {
    const wrap = root.querySelector(".channel-mention-wrap");
    if (wrap) wrap.hidden = reply !== "mention";
  }

  function toggleMentionInput(form) {
    toggleMentionWrap(form, (form.elements.reply && form.elements.reply.value) || "");
  }

  if (listEl) {
    listEl.addEventListener("submit", function (e) {
      const form = e.target.closest("form.channel-add");
      if (!form) return;
      e.preventDefault();
      onAdd(form).catch(function (err) {
        setStatus(err.message || "Add failed", true);
      });
    });
    listEl.addEventListener("input", function (e) {
      const mention = e.target.closest(".channel-mention");
      if (mention) return;
      const input = e.target.closest("form.channel-add input[name=id]");
      if (!input) return;
      fillSuggestList(input);
    });
    listEl.addEventListener("focusin", function (e) {
      const input = e.target.closest("form.channel-add input[name=id]");
      if (input) fillSuggestList(input);
    });
    listEl.addEventListener("focusout", function (e) {
      const form = e.target.closest("form.channel-add");
      if (!form) return;
      window.setTimeout(function () {
        const list = form.querySelector(".channel-ac-list");
        if (list) list.hidden = true;
      }, 150);
    });
    listEl.addEventListener("click", function (e) {
      const item = e.target.closest(".channel-ac-item");
      if (item) {
        const form = item.closest("form.channel-add");
        if (!form) return;
        form.elements.id.value = item.getAttribute("data-id") || "";
        if (!form.elements.label.value) {
          form.elements.label.value = item.getAttribute("data-label") || "";
        }
        const list = form.querySelector(".channel-ac-list");
        if (list) list.hidden = true;
        return;
      }

      // Plus row: toggle add form
      const addTrigger = e.target.closest("[data-action=toggle-add]");
      if (addTrigger) {
        const addRow = addTrigger.closest(".channel-add-row");
        if (!addRow) return;
        const form = addRow.querySelector("form.channel-add");
        if (form) {
          form.hidden = false;
          addTrigger.hidden = true;
          const input = form.querySelector("input[name=id]");
          if (input) input.focus();
        }
        return;
      }

      const cancelAdd = e.target.closest("[data-action=cancel-add]");
      if (cancelAdd) {
        const addRow = cancelAdd.closest(".channel-add-row");
        if (!addRow) return;
        const form = addRow.querySelector("form.channel-add");
        const trigger = addRow.querySelector(".channel-add-trigger");
        if (form) form.hidden = true;
        if (trigger) trigger.hidden = false;
        return;
      }

      // Route sheet picker trigger
      const amp = e.target.closest(".channel-amp-trigger");
      if (amp) {
        e.preventDefault();
        e.stopPropagation();
        document.querySelectorAll(".channel-amp-trigger[id='agent-model-trigger']").forEach(function (el) {
          if (el !== amp) el.removeAttribute("id");
        });
        openRouteSheet(amp.closest(".channel-route"));
        return;
      }

      const qrBtn = e.target.closest("[data-action=wa-qr]");
      if (qrBtn) {
        const card = qrBtn.closest(".channel-card[data-channel='whatsapp']");
        if (!card) return;
        onWaQr(card).catch(function (err) {
          setStatus(err.message || "QR failed", true);
        });
        return;
      }

      const revokeBtn = e.target.closest("[data-action=revoke]");
      if (revokeBtn) {
        const row = revokeBtn.closest(".channel-entry");
        if (!row) return;
        onRevoke(
          row.getAttribute("data-channel"),
          row.getAttribute("data-kind"),
          row.getAttribute("data-id")
        ).catch(function (err) {
          setStatus(err.message || "Remove failed", true);
        });
        return;
      }

      // Clicking compressed row toggles edit fields
      const toggleRow = e.target.closest("[data-action=toggle-row]");
      if (toggleRow) {
        const entry = toggleRow.closest(".channel-entry");
        if (!entry) return;
        const fields = entry.querySelector(".channel-entry-fields");
        if (!fields) return;
        const willExpand = fields.hidden;
        fields.hidden = !willExpand;
        entry.classList.toggle("is-expanded", willExpand);
        return;
      }
    });

    listEl.addEventListener("change", function (e) {
      const addReply = e.target.closest("form.channel-add [name=reply]");
      if (addReply) {
        toggleMentionInput(addReply.closest("form.channel-add"));
        return;
      }
      if (e.target.closest(".channel-amp-trigger, .channel-route")) {
        return;
      }
      const row = e.target.closest(".channel-entry");
      if (!row) return;
      if (!e.target.closest("[data-action=reply], [data-action=mention], [data-action=access]")) return;
      onRuleChange(row).catch(function (err) {
        setStatus(err.message || "Update failed", true);
      });
    });
  }

  if (refreshBtn) refreshBtn.addEventListener("click", function () {
    load();
  });
  if (typeof initializePickerPreferences === "function") initializePickerPreferences();
  if (typeof bindModelSheetControls === "function") bindModelSheetControls();
  if (typeof loadAgents === "function") {
    Promise.resolve(loadAgents()).then(function () {
      if (typeof renderAgentSelect === "function") renderAgentSelect();
    });
  }
  load();
})();
