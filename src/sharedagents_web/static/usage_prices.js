/* Model price table: read-only rows, one editor, one-row writes. */
(function () {
  "use strict";
  const byId = (id) => document.getElementById(id);
  const body = byId("usage-prices-body");
  if (!body) return;
  const base = document.querySelector('meta[name="sharedagents-base"]')?.content || "";
  const search = byId("usage-prices-search"), provider = byId("usage-prices-provider");
  const source = byId("usage-prices-source"), clear = byId("usage-prices-clear");
  const add = byId("usage-prices-add"), form = byId("price-entry-editor");
  const message = byId("usage-prices-message"), previous = byId("usage-prices-prev");
  const next = byId("usage-prices-next"), count = byId("usage-prices-count");
  const sorts = Array.from(document.querySelectorAll("[data-price-sort]"));
  const fields = ["input_per_mtok", "cached_read_per_mtok", "cached_write_per_mtok", "output_per_mtok"];
  const labels = {auto: "Automatic", operator: "Custom", bundled: "Built-in"};
  const pageSize = 50;
  let rows = [], editing = null, adding = false, saving = false, loading = false;
  let page = 0, sortField = "model", direction = 1, filteredCount = 0;

  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, (c) =>
      ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
  }
  function providerName(row) {
    return row.kind === "default" ? "All providers" :
      row.provider || row.provenance?.upstream_provider || row.match?.provider || "Unspecified";
  }
  function modelName(row) {
    return row.kind === "default" ? "Unpriced models" : row.match?.id || row.label || row.match?.alias || "Provider fallback";
  }
  function sourceName(row) {
    return row.kind === "default" ? "Fallback" : labels[row.layer] || "Custom";
  }
  function rate(row, field) {
    if (row[field] != null) return Number(row[field]);
    const effective = field === "cached_read_per_mtok" ? row.cache_read_effective_per_mtok : row.cache_write_effective_per_mtok;
    return Number(effective == null ? row.input_per_mtok : effective);
  }
  function money(value) {
    return "$" + Number(value).toLocaleString("en-US", {maximumFractionDigits: 8});
  }
  function announce(text, error) {
    message.textContent = text;
    message.classList.toggle("is-error", !!error);
  }
  function blocked() { return editing !== null || adding || saving || loading; }
  function controls() {
    const busy = blocked();
    [search, provider, source, clear, add, ...sorts].forEach((el) => { el.disabled = busy; });
    previous.disabled = busy || page === 0;
    next.disabled = busy || (page + 1) * pageSize >= filteredCount;
    body.querySelectorAll("[data-price-action]").forEach((button) => {
      const own = button.closest("tr").dataset.index === editing;
      button.disabled = saving || loading || (busy && !own);
    });
    body.querySelectorAll("input").forEach((input) => { input.disabled = saving; });
    form.querySelectorAll("input, button").forEach((el) => { el.disabled = saving; });
    byId("price-entry-save").textContent = saving && adding ? "Saving…" : "Save price";
    const save = body.querySelector('[data-price-action="save"]');
    if (save) save.textContent = saving ? "Saving…" : "Save";
  }
  function sortValue(row) {
    if (sortField === "model") return modelName(row);
    if (sortField === "provider") return providerName(row);
    if (sortField === "source") return sourceName(row);
    return rate(row, sortField);
  }
  function visibleRows() {
    const query = search.value.trim().toLowerCase();
    return rows.filter((row) => {
      const text = [modelName(row), providerName(row), row.match?.alias || ""].join(" ").toLowerCase();
      return (!query || text.includes(query)) && (!provider.value || providerName(row) === provider.value) &&
        (!source.value || row.layer === source.value);
    }).sort((a, b) => {
      if (a.kind === "default" || b.kind === "default") return a.kind === b.kind ? 0 : a.kind === "default" ? -1 : 1;
      const av = sortValue(a), bv = sortValue(b);
      const result = typeof av === "number" ? av - bv : av.localeCompare(bv, undefined, {numeric: true, sensitivity: "base"});
      return direction * result || modelName(a).localeCompare(modelName(b));
    });
  }
  function rowHtml(row) {
    const active = editing === row._index;
    const model = modelName(row), alias = row.match?.alias;
    const details = row.kind === "default" ? "Used when no model price matches" :
      !row.match?.id ? "Provider fallback" : alias && alias !== model ? "Alias: " + alias : "";
    const origin = row.provenance;
    const title = origin ? "Catalogue: " + origin.upstream_provider + " / " + origin.upstream_model_id +
      (origin.fetched_at ? " · Updated " + origin.fetched_at : "") : sourceName(row);
    const schedule = row.peak_multiplier ? '<span class="usage-price-schedule" title="Existing peak schedule is preserved">Peak ×' + esc(row.peak_multiplier) + "</span>" : "";
    const cells = fields.map((field) => {
      if (active) {
        const optional = field.startsWith("cached_");
        return '<td><input class="usage-price-input" type="number" min="0" step="any" inputmode="decimal" data-field="' + field +
          '" aria-label="' + esc(field.replace(/_per_mtok$/, "").replace(/_/g, " ") + " price for " + model) +
          '" value="' + esc(row[field] == null ? "" : row[field]) + '"' + (optional ? ' placeholder="Input rate"' : " required") + "></td>";
      }
      const inherited = row[field] == null;
      return '<td class="usage-price-rate"' + (inherited ? ' title="Uses the input rate"' : "") + ">" +
        esc(money(rate(row, field))) + (inherited ? '<small class="usage-price-inherited">input rate</small>' : "") + "</td>";
    }).join("");
    const actions = active ? '<button type="button" class="usage-prices-save" data-price-action="save">Save</button> ' +
      '<button type="button" class="usage-price-button" data-price-action="cancel">Cancel</button>' :
      '<button type="button" class="usage-price-button" data-price-action="edit" aria-label="Edit ' + esc(model) + '">Edit</button>' +
      (row.kind === "default" ? "" : ' <button type="button" class="usage-price-delete-btn" data-price-action="delete" aria-label="Delete ' + esc(model) + '">Delete</button>');
    return '<tr data-index="' + row._index + '" class="' + (active ? "is-editing " : "") + (row.kind === "default" ? "is-default" : "") + '">' +
      '<td class="usage-price-provider">' + esc(providerName(row)) + '</td><td class="usage-price-model"><span>' + esc(model) + "</span>" +
      (details ? "<small>" + esc(details) + "</small>" : "") + schedule + '</td><td><span class="usage-price-source is-' + esc(row.layer || "fallback") +
      '" title="' + esc(title) + '">' + esc(sourceName(row)) + "</span></td>" + cells + '<td class="usage-price-row-actions">' + actions + "</td></tr>";
  }
  function render() {
    const filtered = visibleRows();
    filteredCount = filtered.length;
    page = Math.min(page, Math.max(0, Math.ceil(filteredCount / pageSize) - 1));
    body.innerHTML = filtered.slice(page * pageSize, (page + 1) * pageSize).map(rowHtml).join("") ||
      '<tr><td colspan="8" class="usage-na">No models match these filters. Clear filters to see all prices.</td></tr>';
    count.textContent = filteredCount ? (page * pageSize + 1) + "–" + Math.min((page + 1) * pageSize, filteredCount) +
      " of " + filteredCount + " prices" + (filteredCount !== rows.length ? " · " + rows.length + " total" : "") : "0 matching prices";
    byId("usage-prices-page").textContent = "Page " + (page + 1) + " of " + Math.max(1, Math.ceil(filteredCount / pageSize));
    sorts.forEach((button) => {
      const active = button.dataset.priceSort === sortField;
      button.parentElement.setAttribute("aria-sort", active ? direction === 1 ? "ascending" : "descending" : "none");
      const name = button.textContent.replace(/[↑↓]/g, "").trim();
      button.innerHTML = esc(name) + (active ? ' <span aria-hidden="true">' + (direction === 1 ? "↑" : "↓") + "</span>" : "");
    });
    controls();
  }
  function accept(data) {
    rows = (data.default ? [data.default] : []).concat(data.entries || []).map((row, index) => ({...row, _index: String(index)}));
    const selected = provider.value;
    const providers = Array.from(new Set(rows.filter((r) => r.kind !== "default").map(providerName))).sort((a, b) => a.localeCompare(b));
    provider.innerHTML = '<option value="">All providers</option>' + providers.map((p) => '<option value="' + esc(p) + '">' + esc(p) + "</option>").join("");
    provider.value = providers.includes(selected) ? selected : "";
    byId("usage-prices-meta").textContent = "USD per 1M tokens";
    render();
  }
  async function request(url, method, payload) {
    const response = await fetch(base + url, {
      method, credentials: "same-origin", cache: "no-store",
      headers: {Accept: "application/json", ...(payload ? {"Content-Type": "application/json"} : {})},
      ...(payload ? {body: JSON.stringify(payload)} : {}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "Could not update model prices");
    return data;
  }
  function readRates(container, selector) {
    const rates = {};
    fields.forEach((field) => {
      const input = container.querySelector(selector(field));
      const raw = input.value.trim(), optional = field.startsWith("cached_");
      if (raw === "" && optional) rates[field] = null;
      else if (raw === "" || !Number.isFinite(Number(raw)) || Number(raw) < 0) {
        input.focus();
        throw new Error("Enter a non-negative " + field.replace(/_per_mtok$/, "").replace(/_/g, " ") + " price.");
      } else rates[field] = Number(raw);
    });
    return rates;
  }
  async function saveRow(row) {
    if (saving || editing !== row._index) return;
    let rates;
    try { rates = readRates(body.querySelector("tr.is-editing"), (field) => '[data-field="' + field + '"]'); }
    catch (error) { announce(error.message, true); return; }
    saving = true; controls();
    try {
      const data = await request("/api/usage/pricing/entries", "PATCH", {key: {kind: row.kind, provider: row.provider || "", match: row.match || {}}, rates});
      editing = null; accept(data); announce("Saved prices for " + modelName(row) + ".");
    } catch (error) { announce(error.message, true); }
    finally { saving = false; controls(); }
  }
  async function removeRow(row) {
    if (blocked() || !window.confirm("Delete the price for " + modelName(row) + "? Pricing will fall back to the next available rate.")) return;
    const match = row.match || {};
    const key = (row.provider ? row.provider + ":" : "") + (match.id || match.provider || row.label);
    saving = true; controls();
    try {
      accept(await request("/api/usage/pricing/entries/" + encodeURIComponent(key), "DELETE"));
      announce("Deleted price for " + modelName(row) + ".");
    } catch (error) { announce(error.message, true); }
    finally { saving = false; controls(); }
  }
  async function load() {
    if (blocked()) return;
    loading = true; controls();
    try { accept(await request("/api/usage/pricing", "GET")); }
    catch (error) { announce(error.message, true); }
    finally { loading = false; controls(); }
  }
  body.addEventListener("click", (event) => {
    const button = event.target.closest("[data-price-action]");
    if (!button || button.disabled) return;
    const row = rows.find((r) => r._index === button.closest("tr").dataset.index);
    if (!row) return;
    const action = button.dataset.priceAction;
    if (action === "save") return saveRow(row);
    if (action === "delete") return removeRow(row);
    if (action === "cancel") { editing = null; render(); announce("Changes cancelled."); }
    if (action === "edit" && !blocked()) {
      editing = row._index; render(); announce("Editing " + modelName(row) + ". Save or cancel before changing another row.");
      body.querySelector("tr.is-editing input").focus();
    }
  });
  body.addEventListener("keydown", (event) => {
    if (editing === null || saving || !event.target.matches("input")) return;
    if (event.key === "Escape") { editing = null; render(); announce("Changes cancelled."); }
    if (event.key === "Enter") { event.preventDefault(); saveRow(rows.find((r) => r._index === editing)); }
  });
  [search, provider, source].forEach((el) => el.addEventListener(el === search ? "input" : "change", () => { page = 0; render(); }));
  clear.addEventListener("click", () => { search.value = ""; provider.value = ""; source.value = ""; page = 0; render(); });
  sorts.forEach((button) => button.addEventListener("click", () => {
    direction = sortField === button.dataset.priceSort ? -direction : 1;
    sortField = button.dataset.priceSort; page = 0; render();
  }));
  previous.addEventListener("click", () => { page--; render(); });
  next.addEventListener("click", () => { page++; render(); });
  add.addEventListener("click", () => {
    if (blocked()) return;
    adding = true; form.reset(); form.hidden = false; byId("price-entry-error").hidden = true;
    controls(); announce("Adding one model price. Save or cancel to return to the table.");
    form.scrollIntoView({block: "nearest"}); byId("price-entry-provider").focus();
  });
  byId("price-entry-cancel").addEventListener("click", () => {
    if (saving) return;
    adding = false; form.hidden = true; controls(); announce("Changes cancelled.");
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!adding || saving) return;
    const errorEl = byId("price-entry-error");
    errorEl.hidden = true;
    let payload;
    try {
      const prov = byId("price-entry-provider").value.trim(), id = byId("price-entry-id").value.trim();
      if (!prov || !id) throw new Error("Provider and model ID are required.");
      const selectors = {input_per_mtok: "#price-entry-input", output_per_mtok: "#price-entry-output", cached_read_per_mtok: "#price-entry-cache-read", cached_write_per_mtok: "#price-entry-cache-write"};
      payload = {provider: prov, id, alias: byId("price-entry-alias").value.trim(), ...readRates(form, (field) => selectors[field])};
    } catch (error) { errorEl.textContent = error.message; errorEl.hidden = false; return; }
    saving = true; controls();
    try {
      const data = await request("/api/usage/pricing/entries", "POST", payload);
      adding = false; form.hidden = true; accept(data); announce("Added price for " + payload.id + ".");
    } catch (error) { errorEl.textContent = error.message; errorEl.hidden = false; }
    finally { saving = false; controls(); }
  });
  window.addEventListener("beforeunload", (event) => {
    if (editing !== null || adding) { event.preventDefault(); event.returnValue = ""; }
  });
  window.saUsagePrices = {load};
})();
