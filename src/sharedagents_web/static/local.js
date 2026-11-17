/* Models → Router + vLLM panes.
 *
 * Router: one routing list (fallbacks on the same model), collapsible servers.
 * vLLM: running-now, local-only request history, last-turn stats, logs.
 */
(function () {
  "use strict";

  const BASE =
    document.querySelector('meta[name="sharedagents-base"]')?.content || "";
  const DAY_MS = 24 * 60 * 60 * 1000;
  const HOUR_MS = 60 * 60 * 1000;
  const compactFormatter = new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  });
  const exactFormatter = new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  });
  const CHART_COLORS = [
    "#548cff", "#8ee0b0", "#e6c36a", "#d5c6f5",
    "#ef9a90", "#7ec8e3", "#f0a06a", "#c5d4f5",
  ];
  const CHART_WINDOWS = {
    "24h": { ms: 24 * HOUR_MS, source: "hourly" },
    "7d": { ms: 7 * DAY_MS, source: "daily" },
    "30d": { ms: 30 * DAY_MS, source: "daily" },
    "90d": { ms: 90 * DAY_MS, source: "daily" },
  };
  const POLL_MS = 5000;

  const routerStatusEl = document.getElementById("usage-router-status");
  const routerModelsListEl = document.getElementById("router-models-list");
  const routerModelsMetaEl = document.getElementById("router-models-meta");
  const routerModelAddEl = document.getElementById("router-model-add");
  const routerModelEditorEl = document.getElementById("router-model-editor");
  const routerReplicaRowsEl = document.getElementById("router-replica-rows");
  const routerModelErrorEl = document.getElementById("router-model-error");
  const refreshEl = document.getElementById("models-local-refresh");
  const updatedEl = document.getElementById("models-local-updated");
  const statusEl = document.getElementById("models-local-status");
  const chartEl = document.getElementById("vllm-chart");
  const chartLegendEl = document.getElementById("vllm-chart-legend");
  const replicasBodyEl = document.getElementById("vllm-replicas-body");
  const otherBodyEl = document.getElementById("vllm-other-body");
  const decisionBodyEl = document.getElementById("vllm-decision-body");
  const decisionPanelEl = document.getElementById("vllm-decision-panel");
  const muxEl = document.getElementById("vllm-mux-routes");
  const logsEl = document.getElementById("vllm-logs");
  const turnsEl = document.getElementById("vllm-turns");
  const turnsMetaEl = document.getElementById("vllm-turns-meta");
  const benchFormEl = document.getElementById("vllm-bench-form");
  const benchContextEl = document.getElementById("vllm-bench-context");
  const benchOutputEl = document.getElementById("vllm-bench-output");
  const benchTargetsEl = document.getElementById("vllm-bench-targets");
  const benchBodyEl = document.getElementById("vllm-bench-body");
  const benchStatusEl = document.getElementById("vllm-bench-status");
  const benchRunEl = document.getElementById("vllm-bench-run");
  const benchMetaEl = document.getElementById("vllm-bench-meta");

  let routerModels = [];
  let routerModelsPath = "";
  let pendingDeleteId = "";
  let dashboard = null;
  let lastFleet = {};
  let lastMuxRoutes = [];
  let expandedServers = {};
  let chartWindow = "7d";
  let chartMode = "bucket";
  let currentTab = "benchmarks";
  let pollTimer = 0;
  let requestNumber = 0;
  let benchTargets = [];
  let benchSelected = {};
  let benchRun = null;
  let benchResults = {};
  let benchSource = null;
  let benchLoaded = false;

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function number(value) {
    const n = Number(value || 0);
    return Number.isFinite(n) ? n : 0;
  }

  function compact(value) {
    return compactFormatter.format(number(value));
  }

  function exact(value) {
    return exactFormatter.format(number(value));
  }

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function formatContext(value) {
    const n = number(value);
    if (!n) return "—";
    if (n >= 1000) return Math.round(n / 1000) + "k";
    return exact(n);
  }

  function formatTps(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return "—";
    return (n >= 100 ? Math.round(n) : n.toFixed(1)) + " tok/s";
  }

  function categoryOf(row) {
    return row && row.category;
  }

  function isLlmKind(row) {
    return categoryOf(row) === "llm";
  }

  function muxRoutesFrom(data) {
    if (data && data.mux && Array.isArray(data.mux.route_total)) {
      return data.mux.route_total;
    }
    const routes = [];
    ((data && data.routers) || []).forEach(function (router) {
      (((router || {}).metrics || {}).route_total || []).forEach(function (entry) {
        const labels = (entry && entry.labels) || {};
        routes.push({
          upstream: labels.upstream || labels.pool || "",
          value: entry && entry.value,
        });
      });
    });
    return routes;
  }

  function muxCallsForPool(routes, pool) {
    const needle = String(pool || "").toLowerCase();
    if (!needle) return 0;
    return (routes || []).reduce(function (sum, row) {
      const upstream = String((row && row.upstream) || "").toLowerCase();
      if (upstream.indexOf("route:") < 0) return sum;
      if (upstream.indexOf(needle) < 0) return sum;
      return sum + number(row.value);
    }, 0);
  }

  function gpuLabel(gpus) {
    if (!gpus || !gpus.length) return "";
    const first = gpus[0] || {};
    const same = gpus.every(function (gpu) {
      return gpu.name === first.name && gpu.vram_gb === first.vram_gb;
    });
    if (same) {
      const vram = first.vram_gb ? " · " + first.vram_gb + " GB" : "";
      return (gpus.length > 1 ? gpus.length + "× " : "") + (first.name || "GPU") + vram;
    }
    return gpus.map(function (gpu) {
      return (gpu.name || "GPU") + (gpu.vram_gb ? " " + gpu.vram_gb + " GB" : "");
    }).join(" · ");
  }

  function stampUpdated() {
    if (!updatedEl) return;
    updatedEl.textContent =
      "Updated " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function showStatus(message) {
    if (!statusEl) return;
    statusEl.hidden = !message;
    statusEl.textContent = message || "";
  }

  function liveRouteFor(modelId) {
    const routes = Array.isArray(lastFleet.routes) ? lastFleet.routes : [];
    return routes.find(function (route) { return route && route.id === modelId; }) || null;
  }

  function hopsForModel(model) {
    const live = liveRouteFor(model && model.id);
    if (live && Array.isArray(live.fallbacks) && live.fallbacks.length) {
      return live.fallbacks;
    }
    return (model && model.fallbacks) || [];
  }

  function replicaKey(replica) {
    replica = replica || {};
    return String(replica.name || "") + "@" +
      String(replica.url || "").replace(/\/+$/, "");
  }

  function hostFromUrl(url) {
    return String(url || "").replace(/^https?:\/\//, "").replace(/\/.*$/, "");
  }

  function liveReplicaIndex() {
    const index = {};
    ((lastFleet && lastFleet.servers) || []).forEach(function (server) {
      (server.models || []).forEach(function (model) {
        if (!model || !model.url) return;
        index[replicaKey(model)] = model;
      });
    });
    return index;
  }

  /* Concurrency slots a request can occupy, in the order the router fills them:
     replicas in config order (P1, P2, …), each taking up to its own max_active,
     until the pool-wide cap is reached. ``overflowAt`` is the first slot number
     that cannot be served locally. */
  function slotAssignment(model) {
    const replicas = (model && model.replicas) || [];
    const poolCap = number(model && model.max_active_requests);
    const rows = [];
    let cursor = 1;
    if (poolCap <= 0) {
      replicas.forEach(function (replica) {
        rows.push({ replica: replica, from: null, to: null, cap: number(replica.max_active) });
      });
      return { rows: rows, poolCap: 0, overflowAt: null, uncovered: 0 };
    }
    replicas.forEach(function (replica) {
      if (cursor > poolCap) return;
      const cap = number(replica.max_active);
      const remaining = poolCap - cursor + 1;
      const take = cap > 0 ? Math.min(cap, remaining) : remaining;
      if (take <= 0) return;
      rows.push({ replica: replica, from: cursor, to: cursor + take - 1, cap: cap });
      cursor += take;
    });
    return {
      rows: rows,
      poolCap: poolCap,
      overflowAt: cursor,
      uncovered: Math.max(0, poolCap - (cursor - 1)),
    };
  }

  function slotPills(row) {
    if (row.from == null) {
      return '<span class="router-slot is-range">∞</span>';
    }
    const count = row.to - row.from + 1;
    if (count > 8) {
      return '<span class="router-slot is-range">#' + exact(row.from) + "–#" + exact(row.to) + "</span>";
    }
    let html = "";
    for (let slot = row.from; slot <= row.to; slot += 1) {
      html += '<span class="router-slot">' + exact(slot) + "</span>";
    }
    return html;
  }

  function renderReplicaStep(row, index, liveIndex) {
    const replica = row.replica || {};
    const live = liveIndex[replicaKey(replica)];
    const cap = number(replica.max_active);
    const active = live && live.active != null ? number(live.active) : null;
    const state = live && live.healthy === false ? " is-down" :
      live && live.healthy === true ? " is-up" : "";
    const busy = active == null
      ? ""
      : '<span class="router-step-live">' + exact(active) +
        (cap > 0 ? " / " + exact(cap) : "") + " busy</span>";
    return '<li class="router-step' + state + '">' +
      '<span class="router-step-rank">P' + (index + 1) + "</span>" +
      '<div class="router-step-main">' +
        '<span class="router-step-name">' + esc(replica.name || "replica") + "</span>" +
        busy +
        '<span class="router-step-meta">' + esc(hostFromUrl(replica.url)) + "</span>" +
      "</div>" +
      '<div class="router-step-slots">' + slotPills(row) + "</div>" +
      "</li>";
  }

  function renderFallbackHop(hop, index, modelsById) {
    const kind = hop.kind || (hop.external ? "external" : "pool");
    const id = hop.id || hop.pool || hop.external || "hop";
    const target = kind === "pool" ? modelsById[id] : null;
    let detail = "";
    if (kind === "external") {
      detail = [hop.model, hop.host].filter(Boolean).join(" · ");
      if (hop.enabled === false) {
        detail = (detail ? detail + " · " : "") + "no API key";
      }
    } else if (target) {
      const replicas = target.replicas || [];
      detail = (target.max_active_requests
        ? "cap " + exact(target.max_active_requests)
        : "no cap") + " · " + replicas.length +
        (replicas.length === 1 ? " replica" : " replicas");
    }
    const down = hop.healthy === false || hop.enabled === false;
    const state = hop.healthy === true ? " is-up" : down ? " is-down" : "";
    return '<span class="router-hop is-' + esc(kind) + state + '">' +
      '<span class="router-hop-rank">F' + (index + 1) + "</span>" +
      '<span class="router-hop-main"><strong>' + esc(id) + "</strong>" +
      (detail ? "<small>" + esc(detail) + "</small>" : "") +
      "</span></span>";
  }

  function renderFallbackFlow(model, plan) {
    const hops = hopsForModel(model);
    const modelsById = {};
    routerModels.forEach(function (row) { modelsById[row.id] = row; });
    const parts = ['<span class="router-flow-label">' + (plan.overflowAt
      ? "slot #" + exact(plan.overflowAt) + "+"
      : "all replicas full") + "</span>"];
    hops.forEach(function (hop, index) {
      if (index) parts.push('<span class="router-flow-arrow">→</span>');
      parts.push(renderFallbackHop(hop, index, modelsById));
    });
    parts.push('<span class="router-flow-arrow">→</span>');
    parts.push('<span class="router-hop is-reject">' +
      '<span class="router-hop-rank">503</span>' +
      '<span class="router-hop-main"><strong>reject</strong><small>capacity / down</small></span>' +
      "</span>");
    return parts.join("");
  }

  function renderRoutingPlan(model, liveIndex) {
    const plan = slotAssignment(model);
    const live = liveRouteFor(model.id);
    const up = live ? live.healthy === true : null;
    const confirm = pendingDeleteId === model.id;
    let active = 0;
    plan.rows.forEach(function (row) {
      const item = liveIndex[replicaKey(row.replica)];
      if (item && item.active != null) active += number(item.active);
    });
    const health = up === true
      ? '<span class="router-health is-up">up</span>'
      : up === false
        ? '<span class="router-health is-down">down</span>'
        : '<span class="router-health">unknown</span>';
    const steps = plan.rows.length
      ? '<ol class="router-plan-steps">' + plan.rows.map(function (row, index) {
          return renderReplicaStep(row, index, liveIndex);
        }).join("") + "</ol>"
      : '<p class="router-plan-empty">No replicas configured — every request goes to the fallback chain.</p>';
    return '<article class="router-plan" data-router-model="' + esc(model.id) + '">' +
      '<header class="router-plan-head">' +
        '<span class="router-plan-id">' + esc(model.id) + "</span>" +
        '<span class="router-kind">' + esc(model.kind || "llm") + "</span>" +
        '<span class="router-plan-stat">' + (plan.poolCap > 0
          ? "cap " + exact(plan.poolCap) : "no cap") + "</span>" +
        '<span class="router-plan-stat is-live">' + exact(active) +
          (plan.poolCap > 0 ? " / " + exact(plan.poolCap) : "") + " busy</span>" +
        health +
        '<span class="router-plan-actions">' +
          '<button type="button" class="router-text-btn" data-router-edit="' +
            esc(model.id) + '">Edit</button>' +
          (confirm
            ? '<button type="button" class="router-text-btn is-danger" data-router-delete-confirm="' +
              esc(model.id) + '">Confirm delete</button>' +
              '<button type="button" class="router-text-btn" data-router-delete-cancel="1">Cancel</button>'
            : '<button type="button" class="router-text-btn is-danger" data-router-delete="' +
              esc(model.id) + '">Delete</button>') +
        "</span>" +
      "</header>" +
      (plan.uncovered > 0
        ? '<p class="router-plan-warn">Replica caps cover only ' +
          exact(plan.poolCap - plan.uncovered) + " of " + exact(plan.poolCap) +
          " pool slots; the rest falls through before the pool cap.</p>"
        : "") +
      steps +
      '<footer class="router-plan-overflow">' + renderFallbackFlow(model, plan) + "</footer>" +
      "</article>";
  }

  function serverKey(server) {
    return String(server.host || server.id || server.label || "server");
  }

  function serverRunning(server) {
    return (server.models || []).reduce(function (sum, model) {
      if (!isLlmKind(model)) return sum;
      return sum + (model.active == null ? 0 : number(model.active));
    }, 0);
  }

  function serverWaiting(server) {
    return (server.models || []).reduce(function (sum, model) {
      if (!isLlmKind(model)) return sum;
      return sum + (model.waiting == null ? 0 : number(model.waiting));
    }, 0);
  }

  function fleetLlmTotals(servers) {
    let running = 0;
    let waiting = 0;
    let cap = 0;
    let up = 0;
    let total = 0;
    (servers || []).forEach(function (server) {
      (server.models || []).forEach(function (model) {
        if (!isLlmKind(model)) return;
        total += 1;
        if (model.healthy) up += 1;
        running += number(model.active);
        waiting += number(model.waiting);
        cap += number(model.max_active);
      });
    });
    return { running: running, waiting: waiting, cap: cap, up: up, total: total };
  }

  function renderLlmServerModel(model) {
    const up = !!model.healthy;
    const concurrent = (model.active == null ? "—" : exact(model.active)) +
      " / " + (model.max_active == null ? "—" : exact(model.max_active));
    return '<article class="router-model">' +
      '<div class="router-model-head">' +
      "<strong>" + esc(model.id || "model") + "</strong>" +
      '<span class="router-kind">' + esc(model.kind || "llm") + "</span>" +
      '<span class="router-health ' + (up ? "is-up" : "is-down") + '">' +
      (up ? "up" : "down") + "</span>" +
      "</div>" +
      (model.url ? '<p class="router-model-url">' + esc(model.url) + "</p>" : "") +
      '<dl class="router-stats">' +
      "<div><dt>Running</dt><dd>" +
        esc(model.active == null ? "—" : exact(model.active)) + "</dd></div>" +
      "<div><dt>Waiting</dt><dd>" +
        esc(model.waiting == null ? "—" : exact(model.waiting)) + "</dd></div>" +
      "<div><dt>Cap</dt><dd>" + esc(concurrent) + "</dd></div>" +
      "<div><dt>Context</dt><dd>" + esc(formatContext(model.max_context)) + "</dd></div>" +
      "<div><dt>tok/s</dt><dd>" + esc(formatTps(model.tokens_per_sec)) + "</dd></div>" +
      "</dl>" +
      "</article>";
  }

  function renderOtherServerModels(models, muxRoutes, category, title) {
    const other = (models || []).filter(function (model) {
      return categoryOf(model) === category;
    });
    if (!other.length) return "";
    return '<div class="router-other">' +
      "<h4>" + esc(title) + "</h4>" +
      other.map(function (model) {
        const up = !!model.healthy;
        const calls = muxCallsForPool(muxRoutes, model.id);
        return '<article class="router-other-row">' +
          "<strong>" + esc(model.id || "model") + "</strong>" +
          '<span class="router-kind">' + esc(model.kind || "other") + "</span>" +
          '<span class="router-health ' + (up ? "is-up" : "is-down") + '">' +
          (up ? "up" : "down") + "</span>" +
          '<span class="router-other-calls">' + exact(calls) + " calls</span>" +
          "</article>";
      }).join("") +
      "</div>";
  }

  function renderRouterStatus(data) {
    const fleet = data.fleet || {};
    lastFleet = fleet;
    lastMuxRoutes = muxRoutesFrom(data);
    const servers = Array.isArray(fleet.servers) ? fleet.servers : [];
    const llm = fleetLlmTotals(servers);
    const status = fleet.status || (servers.length ? "unknown" : "empty");
    const title = document.getElementById("router-hero-title");
    if (title) {
      title.textContent = servers.length ? status : "offline";
      title.className = "is-" + (servers.length ? status : "down");
    }
    setText("router-kpi-servers", servers.length ? exact(fleet.servers_total || servers.length) : "0");
    setText("router-kpi-replicas", exact(llm.up) + " / " + exact(llm.total));
    setText("router-kpi-queries", exact(llm.running));
    setText("router-kpi-waiting", exact(llm.waiting));
    setText("router-kpi-capacity", llm.cap ? exact(llm.cap) : "—");
    if (routerModels.length) {
      renderRouterModels({ models: routerModels, path: routerModelsPath });
    }
    if (!routerStatusEl) return;
    if (!servers.length) {
      routerStatusEl.innerHTML = '<p class="usage-empty-detail">No local vLLM routers are configured.</p>';
      return;
    }
    routerStatusEl.innerHTML = servers.map(function (server) {
      const key = serverKey(server);
      const open = !!expandedServers[key];
      const state = server.state || "down";
      const running = serverRunning(server);
      const waiting = serverWaiting(server);
      const gpus = gpuLabel(server.gpus);
      const llmModels = (server.models || []).filter(function (model) {
        return isLlmKind(model);
      }).map(renderLlmServerModel).join("");
      const otherHtml =
        renderOtherServerModels(server.models, lastMuxRoutes, "decision", "Decision models") +
        renderOtherServerModels(server.models, lastMuxRoutes, "other", "Other models");
      const hostBits = [server.hostname, server.host].filter(Boolean).join(" · ");
      return '<article class="router-node is-' + esc(state) + (open ? " is-open" : "") +
        '" data-router-host="' + esc(key) + '">' +
        '<button type="button" class="router-node-head" data-router-toggle="' +
          esc(key) + '" aria-expanded="' + (open ? "true" : "false") + '">' +
        '<span class="router-orb" aria-hidden="true"></span>' +
        '<div class="router-node-copy">' +
        "<h3>" + esc(server.label || server.id || "server") + "</h3>" +
        "<p>" + esc(hostBits || "local replica") + "</p>" +
        "</div>" +
        '<span class="router-node-now">' + exact(running) + " LLM running" +
          (waiting ? " · " + exact(waiting) + " wait" : "") + "</span>" +
        '<span class="router-node-state">' + esc(state) + "</span>" +
        '<span class="router-node-chevron" aria-hidden="true"></span>' +
        "</button>" +
        '<div class="router-node-body"' + (open ? "" : " hidden") + ">" +
        (gpus ? '<div class="router-gpus"><span class="router-gpu">' + esc(gpus) + "</span></div>" : "") +
        (llmModels || (otherHtml ? "" : '<p class="usage-na">No LLM replicas on this server</p>')) +
        otherHtml +
        "</div></article>";
    }).join("");
  }

  function setRouterModelError(message) {
    if (!routerModelErrorEl) return;
    routerModelErrorEl.hidden = !message;
    routerModelErrorEl.textContent = message || "";
  }

  function replicaRowHtml(replica) {
    replica = replica || {};
    return '<div class="router-replica-row">' +
      '<input class="router-rep-name" type="text" placeholder="name" value="' +
        esc(replica.name || "") + '">' +
      '<input class="router-rep-url" type="url" placeholder="http://127.0.0.1:8000" value="' +
        esc(replica.url || "") + '">' +
      '<input class="router-rep-max" type="number" min="0" step="1" placeholder="cap" value="' +
        esc(replica.max_active ? String(replica.max_active) : "") + '">' +
      '<button type="button" class="router-text-btn router-rep-remove">Remove</button>' +
      "</div>";
  }

  function addReplicaRow(replica) {
    if (!routerReplicaRowsEl) return;
    routerReplicaRowsEl.insertAdjacentHTML("beforeend", replicaRowHtml(replica));
  }

  function hideRouterEditor() {
    if (routerModelEditorEl) routerModelEditorEl.hidden = true;
    setRouterModelError("");
  }

  function showRouterEditor(model) {
    if (!routerModelEditorEl) return;
    const current = model || {};
    document.getElementById("router-model-original-id").value = current.id || "";
    document.getElementById("router-model-id").value = current.id || "";
    document.getElementById("router-model-kind").value = current.kind || "llm";
    document.getElementById("router-model-max").value =
      current.max_active_requests == null ? "" : String(current.max_active_requests);
    const hops = (current.fallbacks || []).map(function (hop) {
      return (hop.kind || "pool") + ":" + (hop.id || "");
    }).join("\n");
    document.getElementById("router-model-fallbacks").value = hops;
    if (routerReplicaRowsEl) {
      routerReplicaRowsEl.innerHTML = "";
      const replicas = current.replicas && current.replicas.length
        ? current.replicas
        : [{ name: "", url: "", max_active: 0 }];
      replicas.forEach(addReplicaRow);
    }
    setRouterModelError("");
    routerModelEditorEl.hidden = false;
    document.getElementById("router-model-id").focus();
  }

  function parseFallbacks(text) {
    return String(text || "").split("\n").map(function (line) {
      const raw = line.trim();
      if (!raw) return null;
      const idx = raw.indexOf(":");
      const kind = idx >= 0 ? raw.slice(0, idx).trim() : "pool";
      const id = idx >= 0 ? raw.slice(idx + 1).trim() : raw;
      return { kind: kind || "pool", id: id };
    }).filter(Boolean);
  }

  function collectRouterModel() {
    const replicas = [];
    if (routerReplicaRowsEl) {
      routerReplicaRowsEl.querySelectorAll(".router-replica-row").forEach(function (row) {
        const name = (row.querySelector(".router-rep-name") || {}).value || "";
        const url = (row.querySelector(".router-rep-url") || {}).value || "";
        const maxRaw = (row.querySelector(".router-rep-max") || {}).value || "";
        if (!String(name).trim() && !String(url).trim()) return;
        replicas.push({
          name: String(name).trim(),
          url: String(url).trim(),
          max_active: maxRaw === "" ? 0 : Number(maxRaw),
        });
      });
    }
    const maxEl = document.getElementById("router-model-max");
    return {
      id: (document.getElementById("router-model-id").value || "").trim(),
      kind: document.getElementById("router-model-kind").value || "llm",
      max_active_requests: maxEl && maxEl.value !== "" ? Number(maxEl.value) : 0,
      replicas: replicas,
      fallbacks: parseFallbacks(document.getElementById("router-model-fallbacks").value),
    };
  }

  function renderRouterModels(data) {
    routerModels = (data && data.models) || [];
    routerModelsPath = (data && data.path) || "";
    if (routerModelsMetaEl) {
      const count = routerModels.length;
      const src = routerModelsPath
        ? routerModelsPath.replace(/^.*\//, "")
        : "vllm-router.yaml";
      routerModelsMetaEl.textContent = count
        ? count + " model" + (count === 1 ? "" : "s") + " · loaded " + src
        : "No models configured · " + src;
    }
    if (!routerModelsListEl) return;
    if (!routerModels.length) {
      routerModelsListEl.innerHTML =
        '<p class="router-models-empty">No models in the routing list yet. Add one to create the operator YAML.</p>';
      return;
    }
    const llm = routerModels.filter(function (model) { return isLlmKind(model); });
    const decision = routerModels.filter(function (model) {
      return categoryOf(model) === "decision";
    });
    const other = routerModels.filter(function (model) {
      return categoryOf(model) === "other";
    });
    const liveIndex = liveReplicaIndex();
    function otherRow(model) {
      const live = liveRouteFor(model.id);
      const up = live ? live.healthy === true : null;
      const calls = muxCallsForPool(lastMuxRoutes, model.id);
      const confirm = pendingDeleteId === model.id;
      return '<article class="router-model-row is-other" data-router-model="' + esc(model.id) + '">' +
        '<div class="router-model-row-copy">' +
        '<div class="router-model-head">' +
        "<strong>" + esc(model.id) + "</strong>" +
        '<span class="router-kind">' + esc(model.kind || "other") + "</span>" +
        (up === true
          ? '<span class="router-health is-up">up</span>'
          : up === false
            ? '<span class="router-health is-down">down</span>'
            : '<span class="router-health">unknown</span>') +
        '<span class="router-other-calls">' + exact(calls) + " calls</span>" +
        "</div>" +
        "</div>" +
        '<div class="router-model-row-actions">' +
        '<button type="button" class="router-text-btn" data-router-edit="' +
          esc(model.id) + '">Edit</button>' +
        (confirm
          ? '<button type="button" class="router-text-btn is-danger" data-router-delete-confirm="' +
            esc(model.id) + '">Confirm delete</button>' +
            '<button type="button" class="router-text-btn" data-router-delete-cancel="1">Cancel</button>'
          : '<button type="button" class="router-text-btn is-danger" data-router-delete="' +
            esc(model.id) + '">Delete</button>') +
        "</div></article>";
    }
    routerModelsListEl.innerHTML =
      (llm.length
        ? '<h3 class="router-models-group-title">LLM</h3>' +
          '<div class="router-plan-grid">' +
          llm.map(function (model) { return renderRoutingPlan(model, liveIndex); }).join("") +
          "</div>"
        : "") +
      (decision.length
        ? '<h3 class="router-models-group-title">Decision</h3>' + decision.map(otherRow).join("")
        : "") +
      (other.length
        ? '<h3 class="router-models-group-title">Other</h3>' + other.map(otherRow).join("")
        : (llm.length || decision.length ? "" : '<p class="router-models-empty">No models in the routing list yet.</p>'));
  }

  async function loadRouterModels() {
    if (!routerModelsListEl) return;
    try {
      const response = await fetch(BASE + "/api/usage/router-models", {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const data = await response.json().catch(function () { return {}; });
      if (!response.ok) throw new Error(data.error || "Router models unavailable");
      renderRouterModels(data);
    } catch (error) {
      if (routerModelsListEl) {
        routerModelsListEl.innerHTML =
          '<p class="router-model-error">' + esc(error.message || "Router models unavailable") + "</p>";
      }
    }
  }

  async function saveRouterModel(event) {
    if (event) event.preventDefault();
    const original = (document.getElementById("router-model-original-id").value || "").trim();
    const payload = collectRouterModel();
    setRouterModelError("");
    try {
      const url = original
        ? BASE + "/api/usage/router-models/" + encodeURIComponent(original)
        : BASE + "/api/usage/router-models";
      const response = await fetch(url, {
        method: original ? "PUT" : "POST",
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(function () { return {}; });
      if (!response.ok) throw new Error(data.error || "Router model could not be saved");
      pendingDeleteId = "";
      hideRouterEditor();
      renderRouterModels(data);
      loadRouterStatus();
    } catch (error) {
      setRouterModelError(error.message || "Router model could not be saved");
    }
  }

  async function deleteRouterModel(modelId) {
    try {
      const response = await fetch(
        BASE + "/api/usage/router-models/" + encodeURIComponent(modelId),
        {
          method: "DELETE",
          credentials: "same-origin",
          cache: "no-store",
          headers: { Accept: "application/json" },
        }
      );
      const data = await response.json().catch(function () { return {}; });
      if (!response.ok) throw new Error(data.error || "Router model could not be deleted");
      pendingDeleteId = "";
      hideRouterEditor();
      renderRouterModels(data);
      loadRouterStatus();
    } catch (error) {
      setRouterModelError(error.message || "Router model could not be deleted");
    }
  }

  async function loadRouterStatus() {
    try {
      const response = await fetch(BASE + "/api/usage/router-status", {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const data = await response.json().catch(function () { return {}; });
      if (!response.ok) throw new Error(data.error || "Router status unavailable");
      renderRouterStatus(data);
      showStatus("");
      stampUpdated();
    } catch (error) {
      if (routerStatusEl) {
        routerStatusEl.innerHTML =
          '<p class="usage-router-error">' + esc(error.message || "Router status unavailable") + "</p>";
      }
      showStatus(error.message || "Router status unavailable");
    }
  }

  function replicaHost(url) {
    try {
      return new URL(url).host || url;
    } catch (_err) {
      return url || "—";
    }
  }

  function renderReplicas(rows) {
    if (!replicasBodyEl) return;
    const llmRows = (rows || []).filter(function (row) {
      return isLlmKind(row);
    });
    if (!llmRows.length) {
      replicasBodyEl.innerHTML =
        '<tr><td colspan="9" class="usage-na">No LLM replicas reported</td></tr>';
      return;
    }
    replicasBodyEl.innerHTML = llmRows.map(function (row) {
      const up = !!row.healthy;
      const cache = row.gpu_cache_pct == null ? "—" : Number(row.gpu_cache_pct).toFixed(1) + "%";
      return "<tr>" +
        "<td>" + esc(row.server || row.host || "server") + "</td>" +
        "<td>" + esc(row.pool || "—") + "</td>" +
        "<td>" + esc(replicaHost(row.url)) + "</td>" +
        "<td>" + (row.running == null ? "—" : exact(row.running)) + "</td>" +
        "<td>" + (row.waiting == null ? "—" : exact(row.waiting)) + "</td>" +
        "<td>" + esc(cache) + "</td>" +
        "<td>" + esc(formatTps(row.tokens_per_sec)) + "</td>" +
        "<td>" + (row.max_active == null ? "—" : exact(row.max_active)) + "</td>" +
        "<td>" + (up ? "up" : "down") + "</td>" +
        "</tr>";
    }).join("");
  }

  function renderOtherModels(rows, bodyEl, emptyText) {
    bodyEl = bodyEl || otherBodyEl;
    if (!bodyEl) return;
    if (!rows || !rows.length) {
      bodyEl.innerHTML =
        '<tr><td colspan="4" class="usage-na">' + esc(emptyText || "No other models reported") + "</td></tr>";
      return;
    }
    bodyEl.innerHTML = rows.map(function (row) {
      const up = !!row.healthy;
      return "<tr>" +
        "<td>" + esc(row.id || row.pool || "—") + "</td>" +
        "<td>" + esc(row.kind || "other") + "</td>" +
        "<td>" + (up ? "up" : "down") + "</td>" +
        "<td>" + exact(row.calls || 0) + "</td>" +
        "</tr>";
    }).join("");
  }

  function renderMux(mux) {
    if (!muxEl) return;
    const routes = (mux && mux.route_total) || [];
    if (!routes.length) {
      muxEl.innerHTML = '<p class="usage-na">No mux counters yet</p>';
      return;
    }
    muxEl.innerHTML = routes.map(function (row) {
      return '<article class="local-mux-card">' +
        "<strong>" + esc(row.upstream || "upstream") + "</strong>" +
        "<span>" + exact(row.value) + "</span>" +
        "</article>";
    }).join("");
  }

  function renderLogs(rows) {
    if (!logsEl) return;
    if (!rows || !rows.length) {
      logsEl.innerHTML = '<p class="usage-na">No vLLM logs found</p>';
      return;
    }
    logsEl.innerHTML = rows.map(function (row) {
      const empty = !row.text;
      const meta = row.mtime || row.path || "";
      return '<article class="local-log' + (empty ? " is-empty" : "") + '">' +
        "<header><strong>" + esc(row.title || row.id) + "</strong>" +
        "<span>" + esc(meta) + "</span></header>" +
        "<pre>" + esc(empty ? (row.missing ? "(file not found)" : "(empty)") : row.text) +
        "</pre></article>";
    }).join("");
  }

  function renderTurns(recent) {
    if (!turnsEl) return;
    recent = recent || {};
    const finished = Array.isArray(recent.finished) ? recent.finished.slice() : [];
    const stats = Array.isArray(recent.engine_stats) ? recent.engine_stats.slice() : [];
    finished.reverse();
    stats.reverse();
    if (turnsMetaEl) {
      turnsMetaEl.textContent = finished.length
        ? finished.length + " finished · last engine snapshot"
        : (stats.length ? "engine snapshots from logs" : "from process logs");
    }
    if (!finished.length && !stats.length) {
      turnsEl.innerHTML = '<p class="usage-na">No finished turns in the current log tail</p>';
      return;
    }
    const statCards = stats.slice(0, 4).map(function (row) {
      return '<article class="local-turn is-engine">' +
        "<header><strong>Engine</strong><span>" + esc(row.stamp || "latest") + "</span></header>" +
        '<dl class="router-stats">' +
        "<div><dt>Running</dt><dd>" + exact(row.running) + "</dd></div>" +
        "<div><dt>Waiting</dt><dd>" + exact(row.waiting) + "</dd></div>" +
        "<div><dt>gen tok/s</dt><dd>" +
          esc(row.tokens_per_sec == null ? "—" : Number(row.tokens_per_sec).toFixed(1)) +
          "</dd></div>" +
        "<div><dt>prompt tok/s</dt><dd>" +
          esc(row.prompt_tokens_per_sec == null ? "—" : Number(row.prompt_tokens_per_sec).toFixed(1)) +
          "</dd></div>" +
        "<div><dt>KV cache</dt><dd>" +
          esc(row.gpu_cache_pct == null ? "—" : Number(row.gpu_cache_pct).toFixed(1) + "%") +
          "</dd></div>" +
        "</dl></article>";
    }).join("");
    const turnCards = finished.slice(0, 12).map(function (row) {
      const tps = row.tokens_per_sec == null ? "—" : Number(row.tokens_per_sec).toFixed(1) + " tok/s";
      const secs = row.seconds == null ? "—" : Number(row.seconds).toFixed(2) + "s";
      return '<article class="local-turn">' +
        "<header><strong>" + esc(row.id || "request") + "</strong>" +
        "<span>" + esc(row.stamp || "") + "</span></header>" +
        "<p>" + exact(row.tokens) + " tokens · " + esc(secs) + " · " + esc(tps) + "</p>" +
        "</article>";
    }).join("");
    turnsEl.innerHTML = statCards + turnCards;
  }

  function renderVllm(data) {
    const fleet = data.fleet || {};
    const processes = data.processes || [];
    const unit = processes[0] || {};
    const status = fleet.status || (data.replicas && data.replicas.length ? "ok" : "offline");
    const title = document.getElementById("vllm-hero-title");
    if (title) {
      title.textContent = status;
      title.className = "is-" + status;
    }
    setText("vllm-kpi-running", exact(data.running || 0));
    setText("vllm-kpi-waiting", exact(data.waiting || 0));
    setText("vllm-kpi-capacity", data.capacity ? exact(data.capacity) : "—");
    setText(
      "vllm-kpi-replicas",
      exact(data.llm_replicas_healthy != null ? data.llm_replicas_healthy : 0) +
        " / " + exact(data.llm_replicas_total != null ? data.llm_replicas_total : 0)
    );
    setText(
      "vllm-kpi-process",
      unit.available ? ((unit.active || "unknown") + (unit.pid ? " · pid " + unit.pid : "")) : "not installed"
    );
    renderReplicas(data.llm_replicas || data.replicas);
    renderOtherModels(data.other_models);
    if (decisionPanelEl) decisionPanelEl.hidden = !(data.decision_models || []).length;
    renderOtherModels(data.decision_models, decisionBodyEl, "No decision models reported");
    renderMux(data.mux);
    renderTurns(data.recent_turns);
    renderLogs(data.logs);
    renderBenchmarkTargets();
  }

  function chartPlotSize() {
    const mobile = window.matchMedia("(max-width: 720px)").matches;
    const h = mobile ? 280 : 260;
    const w = Math.max(280, Math.round((chartEl && chartEl.clientWidth) || (mobile ? 360 : 760)));
    return { w: w, h: h };
  }

  function renderRequestChart() {
    if (!chartEl) return;
    const spec = CHART_WINDOWS[chartWindow] || CHART_WINDOWS["7d"];
    const rows = ((dashboard && dashboard.breakdown) || {})[spec.source] || [];
    const cutoff = Date.now() - spec.ms;
    const filtered = rows.filter(function (row) {
      const t = Date.parse(row.bucket);
      return Number.isFinite(t) && t >= cutoff &&
        row.local === true && isLlmKind(row);
    });
    const buckets = [];
    const seen = {};
    const seriesSet = {};
    filtered.forEach(function (row) {
      const key = row.bucket;
      if (!seen[key]) {
        seen[key] = true;
        buckets.push(key);
      }
      seriesSet[row.provider || "local-vllm"] = true;
    });
    let series = Object.keys(seriesSet).sort();
    if (!buckets.length || !series.length) {
      chartEl.innerHTML = '<p class="usage-chart-empty">No local requests in this window</p>';
      if (chartLegendEl) chartLegendEl.innerHTML = "";
      return;
    }
    const values = {};
    buckets.forEach(function (bucket) {
      values[bucket] = {};
      series.forEach(function (name) { values[bucket][name] = 0; });
    });
    filtered.forEach(function (row) {
      const name = row.provider || "local-vllm";
      values[row.bucket][name] += number(row.requests);
    });
    const totals = {};
    series.forEach(function (name) {
      totals[name] = buckets.reduce(function (sum, bucket) {
        return sum + values[bucket][name];
      }, 0);
    });
    series = series.filter(function (name) { return totals[name] > 0; });
    series.sort(function (a, b) { return totals[b] - totals[a]; });
    if (chartMode === "cumulative") {
      const running = {};
      series.forEach(function (name) { running[name] = 0; });
      buckets.forEach(function (bucket) {
        series.forEach(function (name) {
          running[name] += values[bucket][name] || 0;
          values[bucket][name] = running[name];
        });
      });
    }
    const rawMax = Math.max.apply(null, buckets.map(function (bucket) {
      return series.reduce(function (sum, name) { return sum + values[bucket][name]; }, 0);
    }).concat([0]));
    function niceMax(value) {
      if (!value) return 1;
      const exp = Math.pow(10, Math.floor(Math.log10(value)));
      const n = value / exp;
      const nice = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
      return nice * exp;
    }
    const max = niceMax(rawMax);
    const plot = chartPlotSize();
    const w = plot.w;
    const h = plot.h;
    const padL = 48;
    const padR = 10;
    const padT = 6;
    const padB = 24;
    const innerW = w - padL - padR;
    const innerH = h - padT - padB;
    const gap = buckets.length > 40 ? 1 : 3;
    const barW = Math.max(2, (innerW / buckets.length) - gap);
    let axis = "";
    for (let i = 0; i <= 4; i++) {
      const frac = i / 4;
      const value = max * frac;
      const y = padT + innerH * (1 - frac);
      axis += '<line x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (w - padR) +
        '" y2="' + y.toFixed(1) + '" stroke="#2c2c30" stroke-width="1"/>';
      axis += '<text class="usage-chart-scale" x="' + (padL - 8) + '" y="' + (y + 4).toFixed(1) +
        '" fill="#c9c9cf" font-size="12" text-anchor="end">' + esc(compact(value)) + "</text>";
    }
    let bars = "";
    buckets.forEach(function (bucket, index) {
      const x = padL + index * (innerW / buckets.length);
      let y = padT + innerH;
      series.forEach(function (name, si) {
        const v = values[bucket][name];
        if (!v || !max) return;
        const bh = (v / max) * innerH;
        y -= bh;
        bars += '<rect x="' + x.toFixed(1) + '" y="' + y.toFixed(1) +
          '" width="' + barW.toFixed(1) + '" height="' + Math.max(1, bh).toFixed(1) +
          '" fill="' + CHART_COLORS[si % CHART_COLORS.length] + '">' +
          "<title>" + esc(bucket) + " · " + esc(name) + " · " + esc(exact(v)) +
          "</title></rect>";
      });
    });
    function bucketLabel(bucket) {
      if (spec.source !== "hourly") return bucket;
      const parsed = Date.parse(bucket);
      if (!Number.isFinite(parsed)) return bucket;
      const d = new Date(parsed);
      const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
      const dd = String(d.getUTCDate()).padStart(2, "0");
      const hh = String(d.getUTCHours()).padStart(2, "0");
      return mm + "-" + dd + " " + hh + ":00";
    }
    const first = buckets[0];
    const last = buckets[buckets.length - 1];
    chartEl.innerHTML =
      '<svg viewBox="0 0 ' + w + " " + h + '" preserveAspectRatio="xMidYMid meet" aria-hidden="true">' +
      axis + bars +
      '<text class="usage-chart-scale" x="' + padL + '" y="' + (h - 10) + '" fill="#c9c9cf" font-size="12">' +
      esc(bucketLabel(first)) + "</text>" +
      '<text class="usage-chart-scale" x="' + (w - padR) + '" y="' + (h - 10) +
      '" fill="#c9c9cf" font-size="12" text-anchor="end">' + esc(bucketLabel(last)) + "</text>" +
      "</svg>";
    if (chartLegendEl) {
      chartLegendEl.innerHTML = series.map(function (name, si) {
        return "<span><i style=\"background:" + CHART_COLORS[si % CHART_COLORS.length] +
          '"></i>' + esc(name) + "</span>";
      }).join("");
    }
  }

  async function loadUsageHistory() {
    try {
      const response = await fetch(BASE + "/api/usage?days=90", {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const data = await response.json().catch(function () { return {}; });
      if (!response.ok) throw new Error(data.error || "Usage history unavailable");
      dashboard = data;
      renderRequestChart();
    } catch (error) {
      if (chartEl) {
        chartEl.innerHTML =
          '<p class="usage-chart-empty">' + esc(error.message || "Usage history unavailable") + "</p>";
      }
    }
  }

  function benchKey(phase, targetId) {
    return String(phase) + ":" + String(targetId);
  }

  function benchRate(value) {
    const n = Number(value);
    return Number.isFinite(n) && n > 0
      ? (n >= 100 ? String(Math.round(n)) : n.toFixed(1))
      : "—";
  }

  function benchMs(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return "—";
    return n >= 1000 ? (n / 1000).toFixed(2) + "s" : Math.round(n) + "ms";
  }

  function benchPhaseLabel(phase) {
    const n = Number(phase);
    if (!(n > 0)) return "0 ctx";
    return (n >= 1024 ? Math.round(n / 1024) + "k" : String(n)) + " ctx";
  }

  function showBenchStatus(message, isError) {
    if (!benchStatusEl) return;
    benchStatusEl.hidden = !message;
    benchStatusEl.textContent = message || "";
    benchStatusEl.classList.toggle("is-error", !!isError);
  }

  function setBenchRunning(running) {
    if (!benchFormEl) return;
    if (benchRunEl) benchRunEl.disabled = running;
    if (benchRunEl) benchRunEl.textContent = running ? "Running…" : "Run benchmark";
  }

  function renderBenchmarkTargets() {
    if (!benchTargetsEl) return;
    if (!benchTargets.length) {
      benchTargetsEl.innerHTML =
        '<p class="local-bench-empty">No local LLM replicas configured in vllm-router.yaml.</p>';
      return;
    }
    const liveIndex = liveReplicaIndex();
    benchTargetsEl.innerHTML = benchTargets.map(function (target) {
      const live = liveIndex[replicaKey(target)];
      const up = live ? live.healthy === true : null;
      const checked = benchSelected[target.id] !== false;
      return '<label class="local-bench-target' + (up === false ? " is-down" : "") + '">' +
        '<input type="checkbox" data-bench-target="' + esc(target.id) + '"' +
          (checked ? " checked" : "") + ">" +
        '<span class="local-bench-target-copy">' +
          '<strong>' + esc(target.name) + "</strong>" +
          '<small>' + esc(target.pool) + (target.host ? " · " + esc(target.host) : "") + "</small>" +
        "</span>" +
        '<span class="local-bench-dot' + (up === false ? " is-down" : up === true ? " is-up" : "") +
          '" title="' + (up === false ? "down" : up === true ? "up" : "unknown") + '"></span>' +
        "</label>";
    }).join("");
  }

  function paintBenchmarkTable() {
    if (!benchBodyEl) return;
    if (!benchRun) {
      benchBodyEl.innerHTML =
        '<tr><td colspan="11" class="usage-na">No benchmark run yet</td></tr>';
      return;
    }
    const targets = benchRun.targets || [];
    const phases = (benchRun.phases && benchRun.phases.length) ? benchRun.phases : [0];
    if (!targets.length || !phases.length) {
      benchBodyEl.innerHTML =
        '<tr><td colspan="11" class="usage-na">No replicas selected</td></tr>';
      return;
    }
    let html = "";
    phases.forEach(function (phase) {
      targets.forEach(function (target) {
        const metrics = benchResults[benchKey(phase, target.id)] || null;
        let state = "queued";
        if (metrics) state = metrics.error ? "error" : "done";
        else if (benchRun.status === "done") state = "missing";
        const title = metrics && metrics.error ? metrics.error : "";
        html += '<tr class="local-bench-row is-' + esc(state) + '"' +
          (title ? ' title="' + esc(title) + '"' : "") + ">" +
          "<td>" + esc(target.name) + "</td>" +
          "<td>" + esc(target.pool) + "</td>" +
          '<td class="local-bench-phase">' + esc(benchPhaseLabel(phase)) + "</td>" +
          '<td class="local-bench-metric">' +
            (metrics ? benchRate(metrics.prefill_tps) : "—") + "</td>" +
          '<td class="local-bench-metric">' +
            (metrics ? benchRate(metrics.gen_tps) : "—") + "</td>" +
          "<td>" + (metrics ? benchMs(metrics.gen_ms) : "—") + "</td>" +
          "<td>" + (metrics ? benchMs(metrics.ttft_ms) : "—") + "</td>" +
          "<td>" + (metrics ? benchMs(metrics.total_ms) : "—") + "</td>" +
          "<td>" + (metrics && metrics.prompt_tokens != null
            ? exact(metrics.prompt_tokens) : "—") + "</td>" +
          "<td>" + (metrics && metrics.completion_tokens != null
            ? exact(metrics.completion_tokens) : "—") + "</td>" +
          '<td class="local-bench-state">' + esc(state) + "</td>" +
          "</tr>";
      });
    });
    benchBodyEl.innerHTML = html;
  }

  function closeBenchmarkStream() {
    if (benchSource) {
      benchSource.close();
      benchSource = null;
    }
  }

  function openBenchmarkStream(runId) {
    if (!window.EventSource) return;
    if (benchSource && benchSource.__runId === runId) return;
    closeBenchmarkStream();
    const source = new EventSource(
      BASE + "/api/models/benchmark/" + encodeURIComponent(runId) + "/stream"
    );
    source.__runId = runId;
    benchSource = source;
    source.onmessage = function (event) {
      let data = null;
      try {
        data = JSON.parse(event.data);
      } catch (_) {
        return;
      }
      if (!data || !data.type) return;
      if (data.type === "start" && data.run) {
        benchRun = data.run;
        paintBenchmarkTable();
      } else if (data.type === "result") {
        benchResults[benchKey(data.phase, data.target)] = data.metrics || {};
        paintBenchmarkTable();
        showBenchStatus(
          benchRun ? "Running " + benchRun.targets.length + " replicas · " +
            benchRun.phases.map(benchPhaseLabel).join(" + ") : "Running…"
        );
      } else if (data.type === "done") {
        if (data.run) benchRun = data.run;
        (benchRun && benchRun.results || []).forEach(function (row) {
          benchResults[benchKey(row.phase, row.target)] = row;
        });
        paintBenchmarkTable();
        closeBenchmarkStream();
        setBenchRunning(false);
        const failed = !!(benchRun && benchRun.status === "error");
        showBenchStatus(failed ? "Benchmark failed." : "Benchmark finished.", failed);
        stampUpdated();
      } else if (data.type === "error") {
        showBenchStatus(data.error || "Benchmark failed", true);
      }
    };
    source.onerror = function () {
      // A finished run closes the stream from the server; polling the snapshot
      // below is the fallback if the browser cannot reconnect.
      if (benchRun && benchRun.status !== "running") {
        closeBenchmarkStream();
        setBenchRunning(false);
      }
    };
  }

  function attachBenchmarkRun(run) {
    if (!run) return;
    benchRun = run;
    benchResults = {};
    (run.results || []).forEach(function (row) {
      benchResults[benchKey(row.phase, row.target)] = row;
    });
    paintBenchmarkTable();
    const running = run.status === "running";
    setBenchRunning(running);
    if (benchMetaEl) {
      const count = (run.targets || []).length;
      const phases = (run.phases || [0]).map(benchPhaseLabel).join(" + ");
      benchMetaEl.textContent = count + " replicas · " + phases;
    }
    if (running) {
      showBenchStatus("Running " + (run.targets || []).length + " replicas · " +
        (run.phases || []).map(benchPhaseLabel).join(" + "));
      openBenchmarkStream(run.id);
    } else {
      closeBenchmarkStream();
      showBenchStatus(run.status === "error"
        ? "Benchmark failed." : "Benchmark finished.", run.status === "error");
    }
  }

  async function loadBenchmark() {
    if (!benchFormEl) return;
    try {
      const response = await fetch(BASE + "/api/models/benchmark", {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const data = await response.json().catch(function () { return {}; });
      if (!response.ok) throw new Error(data.error || "Benchmark targets unavailable");
      benchTargets = data.targets || [];
      benchTargets.forEach(function (target) {
        if (!(target.id in benchSelected)) benchSelected[target.id] = true;
      });
      if (!benchLoaded && data.defaults) {
        if (benchContextEl) benchContextEl.value = String(data.defaults.context_tokens);
        if (benchOutputEl) benchOutputEl.value = String(data.defaults.max_output_tokens);
      }
      benchLoaded = true;
      renderBenchmarkTargets();
      if (data.run) attachBenchmarkRun(data.run);
    } catch (error) {
      showBenchStatus(error.message || "Benchmark targets unavailable", true);
    }
  }

  async function startBenchmark(event) {
    if (event) event.preventDefault();
    const ids = benchTargets
      .filter(function (target) { return benchSelected[target.id] !== false; })
      .map(function (target) { return target.id; });
    if (!ids.length) {
      showBenchStatus("Select at least one replica.", true);
      return;
    }
    const contextTokens = Number((benchContextEl && benchContextEl.value) || 0);
    const maxOutput = Number((benchOutputEl && benchOutputEl.value) || 256);
    setBenchRunning(true);
    showBenchStatus("Starting benchmark…");
    try {
      const response = await fetch(BASE + "/api/models/benchmark", {
        method: "POST",
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({
          targets: ids,
          context_tokens: contextTokens,
          max_output_tokens: maxOutput,
        }),
      });
      const data = await response.json().catch(function () { return {}; });
      if (!response.ok) throw new Error(data.error || "Benchmark could not start");
      attachBenchmarkRun(data);
    } catch (error) {
      setBenchRunning(false);
      showBenchStatus(error.message || "Benchmark could not start", true);
    }
  }

  async function loadVllm() {
    const sequence = ++requestNumber;
    try {
      const response = await fetch(BASE + "/api/models/vllm", {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const data = await response.json().catch(function () { return {}; });
      if (!response.ok) throw new Error(data.error || "vLLM status unavailable");
      if (sequence !== requestNumber) return;
      renderVllm(data);
      showStatus("");
      stampUpdated();
    } catch (error) {
      if (sequence !== requestNumber) return;
      showStatus(error.message || "vLLM status unavailable");
    }
  }

  function stopPoll() {
    if (pollTimer) {
      window.clearInterval(pollTimer);
      pollTimer = 0;
    }
  }

  function startPoll() {
    stopPoll();
    pollTimer = window.setInterval(function () {
      if (document.hidden) return;
      if (currentTab === "router") loadRouterStatus();
      if (currentTab === "vllm") loadVllm();
    }, POLL_MS);
  }

  function refresh(tab) {
    if (tab === "router") {
      loadRouterModels();
      loadRouterStatus();
    } else if (tab === "vllm") {
      loadVllm();
      loadUsageHistory();
      loadBenchmark();
    }
  }

  function show(tab) {
    currentTab = tab;
    if (tab === "router" || tab === "vllm") {
      refresh(tab);
      startPoll();
    } else {
      stopPoll();
    }
  }

  if (routerModelAddEl) {
    routerModelAddEl.addEventListener("click", function () {
      pendingDeleteId = "";
      showRouterEditor(null);
    });
  }
  if (benchFormEl) {
    benchFormEl.addEventListener("submit", startBenchmark);
  }
  if (benchTargetsEl) {
    benchTargetsEl.addEventListener("change", function (event) {
      const box = event.target.closest("[data-bench-target]");
      if (!box) return;
      benchSelected[box.getAttribute("data-bench-target")] = !!box.checked;
    });
  }
  const benchAllEl = document.getElementById("vllm-bench-all");
  if (benchAllEl) {
    benchAllEl.addEventListener("click", function () {
      benchTargets.forEach(function (target) { benchSelected[target.id] = true; });
      renderBenchmarkTargets();
    });
  }
  const benchNoneEl = document.getElementById("vllm-bench-none");
  if (benchNoneEl) {
    benchNoneEl.addEventListener("click", function () {
      benchTargets.forEach(function (target) { benchSelected[target.id] = false; });
      renderBenchmarkTargets();
    });
  }
  const routerModelCancelEl = document.getElementById("router-model-cancel");
  if (routerModelCancelEl) {
    routerModelCancelEl.addEventListener("click", hideRouterEditor);
  }
  if (routerModelEditorEl) {
    routerModelEditorEl.addEventListener("submit", saveRouterModel);
  }
  const routerReplicaAddEl = document.getElementById("router-replica-add");
  if (routerReplicaAddEl) {
    routerReplicaAddEl.addEventListener("click", function () { addReplicaRow({}); });
  }
  if (routerReplicaRowsEl) {
    routerReplicaRowsEl.addEventListener("click", function (event) {
      const btn = event.target.closest(".router-rep-remove");
      if (!btn) return;
      const row = btn.closest(".router-replica-row");
      if (row) row.remove();
    });
  }
  if (routerModelsListEl) {
    routerModelsListEl.addEventListener("click", function (event) {
      const edit = event.target.closest("[data-router-edit]");
      if (edit) {
        const model = routerModels.find(function (row) { return row.id === edit.dataset.routerEdit; });
        pendingDeleteId = "";
        renderRouterModels({ models: routerModels, path: routerModelsPath });
        showRouterEditor(model || { id: edit.dataset.routerEdit });
        return;
      }
      if (event.target.closest("[data-router-delete-cancel]")) {
        pendingDeleteId = "";
        renderRouterModels({ models: routerModels, path: routerModelsPath });
        return;
      }
      const confirmBtn = event.target.closest("[data-router-delete-confirm]");
      if (confirmBtn) {
        deleteRouterModel(confirmBtn.dataset.routerDeleteConfirm);
        return;
      }
      const del = event.target.closest("[data-router-delete]");
      if (del) {
        pendingDeleteId = del.dataset.routerDelete;
        renderRouterModels({ models: routerModels, path: routerModelsPath });
      }
    });
  }
  if (refreshEl) {
    refreshEl.addEventListener("click", function () { refresh(currentTab); });
  }
  if (routerStatusEl) {
    routerStatusEl.addEventListener("click", function (event) {
      const btn = event.target.closest("[data-router-toggle]");
      if (!btn) return;
      const key = btn.getAttribute("data-router-toggle") || "";
      if (!key) return;
      expandedServers[key] = !expandedServers[key];
      const node = btn.closest(".router-node");
      if (!node) return;
      const open = !!expandedServers[key];
      node.classList.toggle("is-open", open);
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      const body = node.querySelector(".router-node-body");
      if (body) body.hidden = !open;
    });
  }
  document.querySelectorAll("[data-vllm-chart-window]").forEach(function (button) {
    button.addEventListener("click", function () {
      chartWindow = button.getAttribute("data-vllm-chart-window") || "7d";
      document.querySelectorAll("[data-vllm-chart-window]").forEach(function (candidate) {
        const active = candidate.getAttribute("data-vllm-chart-window") === chartWindow;
        candidate.classList.toggle("active", active);
        candidate.setAttribute("aria-pressed", active ? "true" : "false");
      });
      if (dashboard) renderRequestChart();
    });
  });
  document.querySelectorAll("[data-vllm-chart-mode]").forEach(function (button) {
    button.addEventListener("click", function () {
      chartMode = button.getAttribute("data-vllm-chart-mode") || "bucket";
      document.querySelectorAll("[data-vllm-chart-mode]").forEach(function (candidate) {
        const active = candidate.getAttribute("data-vllm-chart-mode") === chartMode;
        candidate.classList.toggle("active", active);
        candidate.setAttribute("aria-pressed", active ? "true" : "false");
      });
      if (dashboard) renderRequestChart();
    });
  });
  window.addEventListener("resize", function () {
    if (currentTab === "vllm" && dashboard) renderRequestChart();
  });
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden && (currentTab === "router" || currentTab === "vllm")) {
      refresh(currentTab);
    }
  });

  window.ModelsLocal = { show: show, refresh: refresh };
})();
