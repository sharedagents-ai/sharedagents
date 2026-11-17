/* Ledger-backed Usage dashboard. */
(function () {
  "use strict";

  const BASE =
    document.querySelector('meta[name="sharedagents-base"]')?.content || "";
  const DAY_MS = 24 * 60 * 60 * 1000;
  const compactFormatter = new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  });
  const exactFormatter = new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  });

  const appEl = document.getElementById("usage-app");
  const refreshEl = document.getElementById("usage-refresh");
  const updatedEl = document.getElementById("usage-updated");
  const statusEl = document.getElementById("usage-status");
  const heatmapEl = document.getElementById("usage-heatmap");
  const monthsEl = document.getElementById("usage-months");
  const selectionEl = document.getElementById("usage-activity-selection");
  const tokenMixEl = document.getElementById("usage-token-mix");
  const modelListEl = document.getElementById("usage-model-list");
  const modelExpandEl = document.getElementById("usage-model-expand");
  const generationsBodyEl = document.getElementById("usage-generations-body");
  const generationsMetaEl = document.getElementById("usage-generations-meta");
  const USAGE_TABS = ["usage", "api", "subscriptions", "prices", "logs"];
  const subscriptionsGroupsEl = document.getElementById("usage-subscriptions-groups");
  const subscriptionsGroupEl = document.getElementById("usage-group-subscriptions");
  const apiGroupEl = document.getElementById("usage-group-api");
  const subscriptionsGridEl = document.getElementById("usage-subscriptions-grid");
  const apiGridEl = document.getElementById("usage-api-grid");
  const subscriptionsEmptyEl = document.getElementById("usage-subscriptions-empty");
  const subscriptionsMetaEl = document.getElementById("usage-subscriptions-meta");
  const subscriptionsUpdateEl = document.getElementById("usage-subscriptions-update");
  const subscriptionsPaceScaleEl = document.getElementById("usage-subscriptions-pace-scale");
  const apiEmptyEl = document.getElementById("usage-api-empty");
  const apiMetaEl = document.getElementById("usage-api-meta");
  const apiUpdateEl = document.getElementById("usage-api-update");
  const subChartEl = document.getElementById("usage-sub-chart");
  const subChartLegendEl = document.getElementById("usage-sub-chart-legend");
  const apiChartEl = document.getElementById("usage-api-chart");
  const apiChartLegendEl = document.getElementById("usage-api-chart-legend");
  // Per history chart: the period picked, the window the server sent for it
  // (/api/usage/pace-history?period=..., already cut and thinned there) and
  // the series hidden from the legend.
  const HISTORY_PERIOD_KEY = "sharedagents.usage.historyPeriod.";
  const historyCharts = {
    sub: { period: storedHistoryPeriod("sub"), history: [], hidden: Object.create(null), seq: 0 },
    api: { period: storedHistoryPeriod("api"), history: [], hidden: Object.create(null), seq: 0 },
  };
  let subscriptionsLoading = false;
  let subscriptionPaceCards = Object.create(null);
  let subscriptionCatalog = [];
  // External-agent integrity scan (installed / version / authenticated).
  // Keyed by provider id for the cards and rendered in full (all 12 engines)
  // as the "External agents" group under the subscription cards.
  let subscriptionHealth = Object.create(null);
  let agentHealthRows = [];
  let agentHealthLoaded = false;
  let subscriptionCardData = Object.create(null);
  const expandedProviderCards = new Set();
  const agentsGroupsEl = document.getElementById("usage-agents-groups");
  const agentsGroupEl = document.getElementById("usage-group-agents");
  const agentsGridEl = document.getElementById("usage-agents-grid");
  const agentsScanEl = document.getElementById("usage-agents-scan");
  const logFilterProviderEl = document.getElementById("usage-log-filter-provider");
  const logFilterAgentEl = document.getElementById("usage-log-filter-agent");
  const logFilterModelEl = document.getElementById("usage-log-filter-model");
  const LOG_LIMIT = 20;
  const chartEl = document.getElementById("usage-chart");
  const chartLegendEl = document.getElementById("usage-chart-legend");
  const chartTableEl = document.getElementById("usage-chart-table");
  const MODEL_PREVIEW = 5;
  const CHART_COLORS = [
    "#548cff", "#8ee0b0", "#e6c36a", "#d5c6f5",
    "#ef9a90", "#7ec8e3", "#f0a06a", "#c5d4f5",
  ];
  const HOUR_MS = 60 * 60 * 1000;
  const CHART_WINDOWS = {
    "24h": { ms: 24 * HOUR_MS, source: "hourly" },
    "7d": { ms: 7 * DAY_MS, source: "daily" },
    "30d": { ms: 30 * DAY_MS, source: "daily" },
    "90d": { ms: 90 * DAY_MS, source: "daily" },
    "365d": { ms: 365 * DAY_MS, source: "daily" },
  };

  let dashboard = null;
  let activityView = "daily";
  let requestNumber = 0;
  let modelsExpanded = false;
  let chartWindow = "7d";
  let chartGroup = "provider";
  let chartMetric = "tokens";
  let chartMode = "bucket";
  let logProvider = "";
  let logAgent = "";
  let logModel = "";

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

  function percent(value) {
    const n = number(value);
    const digits = Math.abs(n % 1) > 0.01 ? 1 : 0;
    return n.toFixed(digits) + "%";
  }

  // Subscription quota labels keep one decimal when the value is fractional so a
  // card can be compared with the vendor CLI (antigravity prints e.g.
  // "36.44% remaining"). Rounding 36.4% / 63.6% to whole percents turned a
  // ~0.4pp sampling difference into a wrong-looking number ("36%" vs "37%").
  function quotaPercent(value) {
    if (value == null || value === "") return "—";
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    return (Math.abs(n % 1) > 0.01 ? n.toFixed(1) : String(Math.round(n))) + "%";
  }

  function currency(value) {
    const n = number(value);
    if (!Number.isFinite(n)) return "—";
    return "$" + n.toFixed(6);
  }

  function delta(value, current) {
    if (value == null) return number(current) > 0 ? "New" : "—";
    const n = number(value);
    const rounded = Math.abs(n) >= 100 ? Math.round(n) : Math.round(n * 10) / 10;
    return (rounded > 0 ? "+" : "") + rounded + "%";
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function parseDay(value) {
    const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value || ""));
    if (!match) return null;
    return new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])));
  }

  function addDays(day, count) {
    return new Date(day.getTime() + count * DAY_MS);
  }

  function isoDay(day) {
    return day.toISOString().slice(0, 10);
  }

  function fullDate(day) {
    return day.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      timeZone: "UTC",
    });
  }

  function renderSummary(data) {
    const totals = data.totals || {};
    const mix = data.token_mix || {};
    const billedInput = number(mix.input);
    const billedCached = number(mix.cached_read);
    const billedOutput = number(mix.output);
    setText("usage-total-tokens", compact(billedInput + billedCached + billedOutput));
    setText("usage-billed-input", compact(billedInput));
    setText("usage-billed-cached", compact(billedCached));
    setText("usage-billed-output", compact(billedOutput));
    setText("usage-requests", compact(totals.requests));
    setText("usage-sessions", compact(totals.sessions));
    setText("usage-active-days", exact(totals.active_days));
    const streak = number(totals.active_streak);
    setText("usage-active-streak", streak + " " + (streak === 1 ? "day" : "days"));
    setText("usage-models", exact(totals.models));
    setText("usage-spend", currency(totals.cost_usd));
  }

  function levelScale(values) {
    const unique = Array.from(new Set(values.filter((v) => v > 0))).sort((a, b) => a - b);
    return function (value) {
      if (value <= 0 || !unique.length) return 0;
      let rank = unique.findIndex((candidate) => value <= candidate);
      if (rank < 0) rank = unique.length - 1;
      return Math.max(1, Math.min(5, Math.ceil(((rank + 1) / unique.length) * 5)));
    };
  }

  function monthLabels(start, end, alignedStart, weeks) {
    const labels = [];
    let cursor = new Date(start.getTime());
    while (cursor <= end) {
      if (cursor.getTime() === start.getTime() || cursor.getUTCDate() === 1) {
        const week = Math.floor((cursor.getTime() - alignedStart.getTime()) / (7 * DAY_MS));
        if (week >= 0 && week < weeks) {
          labels.push({
            week: week + 1,
            label: cursor.toLocaleDateString("en-US", { month: "short", timeZone: "UTC" }),
          });
        }
      }
      cursor = addDays(cursor, 1);
    }
    monthsEl.style.setProperty("--weeks", String(weeks));
    monthsEl.innerHTML = labels.map((item) =>
      '<span style="grid-column:' + item.week + '">' + esc(item.label) + "</span>"
    ).join("");
  }

  function daySummary(day, item) {
    return fullDate(day) + " · " + compact(item.tokens) + " tokens · " +
      compact(item.requests) + " requests";
  }

  function weekSummary(start, end, item) {
    const sameYear = start.getUTCFullYear() === end.getUTCFullYear();
    const left = start.toLocaleDateString("en-US", {
      month: "short", day: "numeric", year: sameYear ? undefined : "numeric", timeZone: "UTC",
    });
    return left + " – " + fullDate(end) + " · " + compact(item.tokens) +
      " tokens · " + compact(item.requests) + " requests";
  }

  function renderActivity(data) {
    if (!heatmapEl || !monthsEl) return;
    const range = data.range || {};
    const start = parseDay(range.start_date);
    const end = parseDay(range.end_date);
    if (!start || !end || end < start) {
      heatmapEl.innerHTML = "";
      monthsEl.innerHTML = "";
      selectionEl.textContent = "No recorded activity";
      return;
    }

    const activity = new Map((data.activity || []).map((item) => [item.date, {
      tokens: number(item.tokens),
      requests: number(item.requests),
    }]));
    const alignedStart = addDays(start, -start.getUTCDay());
    const alignedEnd = addDays(end, 6 - end.getUTCDay());
    const weeks = Math.round((alignedEnd.getTime() - alignedStart.getTime()) / (7 * DAY_MS)) + 1;
    monthLabels(start, end, alignedStart, weeks);
    heatmapEl.style.setProperty("--weeks", String(weeks));
    heatmapEl.classList.toggle("is-weekly", activityView === "weekly");
    heatmapEl.setAttribute(
      "aria-label",
      activityView === "weekly" ? "Weekly token activity" : "Daily token activity"
    );

    const points = [];
    if (activityView === "weekly") {
      for (let week = 0; week < weeks; week += 1) {
        const weekStart = addDays(alignedStart, week * 7);
        const weekEnd = addDays(weekStart, 6);
        const item = { tokens: 0, requests: 0 };
        for (let offset = 0; offset < 7; offset += 1) {
          const day = addDays(weekStart, offset);
          if (day < start || day > end) continue;
          const recorded = activity.get(isoDay(day));
          if (recorded) {
            item.tokens += recorded.tokens;
            item.requests += recorded.requests;
          }
        }
        points.push({ start: weekStart, end: weekEnd, item });
      }
    } else {
      for (let day = alignedStart; day <= alignedEnd; day = addDays(day, 1)) {
        const outside = day < start || day > end;
        points.push({
          day,
          outside,
          item: outside ? { tokens: 0, requests: 0 } :
            (activity.get(isoDay(day)) || { tokens: 0, requests: 0 }),
        });
      }
    }

    const scale = levelScale(points.map((point) => point.item.tokens));
    heatmapEl.innerHTML = points.map((point) => {
      if (point.outside) {
        return '<span class="usage-heat-cell is-outside" aria-hidden="true"></span>';
      }
      const summary = activityView === "weekly"
        ? weekSummary(point.start < start ? start : point.start, point.end > end ? end : point.end, point.item)
        : daySummary(point.day, point.item);
      const key = activityView === "weekly" ? isoDay(point.start) : isoDay(point.day);
      return '<button type="button" class="usage-heat-cell heat-level-' +
        scale(point.item.tokens) + '" data-activity-key="' + key +
        '" data-summary="' + esc(summary) + '" title="' + esc(summary) +
        '" aria-label="' + esc(summary) + '"></button>';
    }).join("");

    const latest = [...points].reverse().find((point) => !point.outside && point.item.tokens > 0) ||
      [...points].reverse().find((point) => !point.outside);
    if (!latest) {
      selectionEl.textContent = "No recorded activity";
    } else if (activityView === "weekly") {
      selectionEl.textContent = weekSummary(
        latest.start < start ? start : latest.start,
        latest.end > end ? end : latest.end,
        latest.item
      );
    } else {
      selectionEl.textContent = daySummary(latest.day, latest.item);
    }
  }

  function barRow(label, value, maximum) {
    const width = maximum > 0 ? Math.max(0, Math.min(100, (value / maximum) * 100)) : 0;
    return '<div class="usage-bar-row">' +
      '<div class="usage-bar-label"><span title="' + esc(label) + '">' + esc(label) +
      '</span><strong>' + compact(value) + '</strong></div>' +
      '<div class="usage-bar-track"><div class="usage-bar-fill" style="width:' +
      width.toFixed(2) + '%"></div></div></div>';
  }

  function renderTokenMix(data) {
    const mix = data.token_mix || {};
    const rows = [
      ["Input (billed)", number(mix.input)],
      ["Cached (billed)", number(mix.cached_read)],
    ];
    if (number(mix.cached_write) > 0) rows.push(["Cache write", number(mix.cached_write)]);
    if (number(mix.thinking) > 0) rows.push(["Reasoning", number(mix.thinking)]);
    rows.push(["Output (billed)", number(mix.output)]);
    const maximum = Math.max(0, ...rows.map((row) => row[1]));
    tokenMixEl.innerHTML = rows.map((row) => barRow(row[0], row[1], maximum)).join("");
    setText("usage-cache-rate", percent(mix.cache_hit_rate));
  }

  function renderModels(data) {
    const rows = Array.isArray(data.models) ? data.models : [];
    setText("usage-model-count", exact((data.totals || {}).models) + " models · by output tokens");
    if (!rows.length) {
      modelListEl.innerHTML = '<p class="usage-empty-detail">No model usage recorded yet</p>';
      modelListEl.classList.remove("is-expanded");
      if (modelExpandEl) modelExpandEl.hidden = true;
      return;
    }
    const canExpand = rows.length > MODEL_PREVIEW;
    if (!canExpand) modelsExpanded = false;
    const visible = (canExpand && !modelsExpanded)
      ? rows.slice(0, MODEL_PREVIEW)
      : rows;
    const maximum = Math.max(0, ...rows.map((row) => number(row.tokens)));
    modelListEl.innerHTML = visible.map((row) =>
      barRow(String(row.name || "Unknown"), number(row.tokens), maximum)
    ).join("");
    modelListEl.classList.toggle("is-expanded", Boolean(canExpand && modelsExpanded));
    if (modelExpandEl) {
      modelExpandEl.hidden = !canExpand;
      modelExpandEl.setAttribute("aria-expanded", modelsExpanded ? "true" : "false");
      modelExpandEl.textContent = modelsExpanded
        ? "Show top " + MODEL_PREVIEW
        : "Show all " + exact(rows.length) + " models";
    }
  }

  function renderPace(data) {
    const pace = data.pace || {};
    const mix = data.token_mix || {};
    setText("usage-last-hour", compact(pace.last_hour_requests));
    setText("usage-last-day", compact(pace.last_24h_requests));
    setText("usage-day-change", delta(pace.day_change_pct, pace.last_24h_requests));
    setText("usage-week-change", delta(pace.week_change_pct, pace.last_7d_requests));
    setText("usage-pace-cache", percent(mix.cache_hit_rate));
    setText("usage-avg-request", compact(pace.average_tokens_per_request));
  }

  function na(value) {
    return value == null ? "n/a" : String(value);
  }

  function displayValue(value) {
    if (value == null) return '<em class="usage-na">n/a</em>';
    if (Array.isArray(value)) {
      if (!value.length) return '<em class="usage-na">none</em>';
      return esc(value.map(function (item) {
        return item && typeof item === "object" ? JSON.stringify(item) : String(item);
      }).join(", "));
    }
    if (typeof value === "object") {
      const keys = Object.keys(value);
      if (!keys.length) return '<em class="usage-na">none</em>';
      return esc(keys.map(function (key) {
        return key + "=" + value[key];
      }).join(" "));
    }
    const text = String(value);
    return text ? esc(text) : '<em class="usage-na">none</em>';
  }

  function setUsageTab(name) {
    const tab = USAGE_TABS.indexOf(name) >= 0 ? name : "usage";
    document.querySelectorAll("[data-usage-tab]").forEach(function (button) {
      const active = button.dataset.usageTab === tab;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    USAGE_TABS.forEach(function (id) {
      const pane = document.getElementById("usage-pane-" + id);
      if (!pane) return;
      pane.hidden = tab !== id;
      pane.classList.toggle("is-active", tab === id);
    });
    if (appEl) {
      appEl.classList.toggle("is-logs", tab === "logs");
      appEl.classList.toggle("is-prices", tab === "prices");
      appEl.classList.toggle("is-subscriptions", tab === "subscriptions");
      appEl.classList.toggle("is-api", tab === "api");
    }
    if (tab !== "usage") {
      if (location.hash !== "#" + tab) history.replaceState(null, "", "#" + tab);
    } else if (location.hash && location.hash !== "#usage") {
      history.replaceState(null, "", location.pathname + location.search);
    }
    if (tab === "subscriptions" || tab === "api") {
      loadSubscriptions(false);
      renderAllHistoryCharts();
    }
  }

  function ms(value) {
    const n = number(value);
    if (!n) return "—";
    return n >= 1000 ? (n / 1000).toFixed(1) + "s" : Math.round(n) + "ms";
  }

  function fillLogFilter(el, values, current) {
    if (!el) return;
    const options = ["<option value=\"\">All</option>"].concat(
      (values || []).map(function (value) {
        return "<option value=\"" + esc(value) + "\">" + esc(value) + "</option>";
      })
    );
    el.innerHTML = options.join("");
    el.value = (values || []).indexOf(current) >= 0 ? current : "";
  }

  function renderGenerations(payload) {
    if (!generationsBodyEl) return;
    const rows = (payload && payload.generations) || [];
    const total = number(payload && payload.total);
    const facets = (payload && payload.filters) || {};
    fillLogFilter(logFilterProviderEl, facets.providers, logProvider);
    fillLogFilter(logFilterAgentEl, facets.agents, logAgent);
    fillLogFilter(logFilterModelEl, facets.models, logModel);
    if (logFilterProviderEl) logProvider = logFilterProviderEl.value;
    if (logFilterAgentEl) logAgent = logFilterAgentEl.value;
    if (logFilterModelEl) logModel = logFilterModelEl.value;
    const shown = rows.length;
    const filtered = !!(logProvider || logAgent || logModel);
    if (generationsMetaEl) {
      generationsMetaEl.textContent = total
        ? "last " + exact(shown) + " of " + exact(total) + (filtered ? " matching" : "")
        : "last " + exact(LOG_LIMIT);
    }
    if (!rows.length) {
      generationsBodyEl.innerHTML =
        '<tr><td colspan="11" class="usage-na">No generations recorded</td></tr>';
      return;
    }
    generationsBodyEl.innerHTML = rows.map(function (row) {
      const estimated = Number(row.tokens_estimated) === 1;
      const mark = estimated ? " ≈" : "";
      const when = row.created_at
        ? new Date(row.created_at).toLocaleString([], {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })
        : "—";
      return "<tr" + (estimated ? ' class="is-estimated"' : "") + ">" +
        "<td>" + esc(when) + "</td>" +
        "<td>" + esc(row.provider || "local-vllm") + "</td>" +
        "<td>" + esc(row.agent_name || "jarvis") + "</td>" +
        "<td>" + esc(row.model || row.model_alias || "local-vllm") + mark + "</td>" +
        "<td>" + exact(row.fresh_tokens || Math.max(0, number(row.input_tokens) - number(row.cached_tokens))) + "</td>" +
        "<td>" + exact(row.cached_tokens) + "</td>" +
        "<td>" + exact(row.output_tokens) + "</td>" +
        "<td>" + ms(row.ttft_ms) + "</td>" +
        "<td>" + (number(row.gen_tps) ? number(row.gen_tps).toFixed(1) + " t/s" : "—") + "</td>" +
        "<td>" + ms(row.wt_ms) + "</td>" +
        "<td>" + currency(row.cost_usd) + mark + "</td>" +
        "</tr>";
    }).join("");
  }

  function chartPlotSize() {
    const mobile = window.matchMedia("(max-width: 720px)").matches;
    const h = mobile ? 360 : 320;
    const w = Math.max(280, Math.round((chartEl && chartEl.clientWidth) || (mobile ? 360 : 760)));
    return { w: w, h: h };
  }

  function renderUsageChart() {
    if (!chartEl) return;
    const spec = CHART_WINDOWS[chartWindow] || CHART_WINDOWS["7d"];
    const rows = ((dashboard && dashboard.breakdown) || {})[spec.source] || [];
    const now = Date.parse(dashboard && dashboard.generated_at) || Date.now();
    const step = spec.source === "hourly" ? HOUR_MS : DAY_MS;
    const end = Math.floor(now / step) * step;
    const cutoff = end - spec.ms + step;
    const filtered = rows.filter(function (row) {
      const t = Date.parse(row.bucket);
      return Number.isFinite(t) && t >= cutoff && t <= end;
    });
    const buckets = [];
    for (let t = cutoff; t <= end; t += step) {
      const iso = new Date(t).toISOString();
      buckets.push(spec.source === "hourly" ? iso.slice(0, 19) + "Z" : iso.slice(0, 10));
    }
    const seriesSet = {};
    filtered.forEach(function (row) {
      const name = chartGroup === "model" ? (row.model || "local-vllm") : (row.provider || "local-vllm");
      seriesSet[name] = true;
    });
    const baseline = (dashboard && dashboard.baseline) || {};
    const seasonal = chartWindow === "24h" || chartWindow === "7d";
    const datedPeers = seasonal ? baseline["dated_" + spec.source] || {} : {};
    let medianRunning = 0;
    const medianValues = buckets.map(function (bucket) {
      const peer = datedPeers[bucket];
      if (!peer || !peer.samples) return null;
      const value = chartMetric === "cost" ? number(peer.cost_pico) / 1e12 : number(peer[chartMetric]);
      medianRunning += value;
      return { value: chartMode === "cumulative" ? medianRunning : value, samples: peer.samples };
    });
    const hasMedian = medianValues.some(function (item) { return item !== null; });
    let series = Object.keys(seriesSet).sort();
    if (!series.length && !hasMedian) {
      chartEl.innerHTML = '<p class="usage-chart-empty">No usage in this window</p>';
      if (chartLegendEl) chartLegendEl.innerHTML = "";
      if (chartTableEl) chartTableEl.innerHTML = "";
      return;
    }
    const values = {};
    buckets.forEach(function (bucket) {
      values[bucket] = {};
      series.forEach(function (name) { values[bucket][name] = 0; });
    });
    filtered.forEach(function (row) {
      const name = chartGroup === "model" ? (row.model || "local-vllm") : (row.provider || "local-vllm");
      const amount = chartMetric === "cost"
        ? number(row.cost_usd)
        : chartMetric === "requests" ? number(row.requests) : number(row.tokens);
      values[row.bucket][name] += amount;
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
    }).concat(medianValues.map(function (item) { return item ? item.value : 0; }), [0]));
    function niceMax(value) {
      if (!value) return 1;
      const exp = Math.pow(10, Math.floor(Math.log10(value)));
      const n = value / exp;
      const nice = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
      return nice * exp;
    }
    const max = niceMax(rawMax);
    function axisLabel(value) {
      if (chartMetric === "cost") {
        if (value >= 1) return "$" + (value >= 100 ? Math.round(value) : value.toFixed(1));
        if (value >= 0.01) return "$" + value.toFixed(2);
        return "$" + value.toFixed(4);
      }
      return compact(value);
    }
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
        '" fill="#c9c9cf" font-size="12" text-anchor="end">' + esc(axisLabel(value)) + "</text>";
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
          "<title>" + esc(bucket) + " · " + esc(name) + " · " +
          esc(chartMetric === "cost" ? currency(v) : compact(v)) +
          "</title></rect>";
      });
    });
    let medianPath = "";
    let connected = false;
    medianValues.forEach(function (item, index) {
      if (!item) { connected = false; return; }
      const x = padL + index * (innerW / buckets.length);
      const right = padL + (index + 1) * (innerW / buckets.length);
      const y = padT + innerH * (1 - item.value / max);
      medianPath += connected ? " V " + y.toFixed(1) : " M " + x.toFixed(1) + " " + y.toFixed(1);
      medianPath += " H " + right.toFixed(1);
      connected = true;
    });
    const medianLine = hasMedian ? '<path class="usage-chart-median" d="' + medianPath +
      '" fill="none" stroke="#f7f7fb" stroke-width="2" stroke-dasharray="6 4" pointer-events="none"><title>Total seasonal median (UTC)</title></path>' : "";
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
      axis + bars + medianLine +
      '<text class="usage-chart-scale" x="' + padL + '" y="' + (h - 10) + '" fill="#c9c9cf" font-size="12">' +
      esc(bucketLabel(first)) + "</text>" +
      '<text class="usage-chart-scale" x="' + (w - padR) + '" y="' + (h - 10) + '" fill="#c9c9cf" font-size="12" text-anchor="end">' +
      esc(bucketLabel(last)) + "</text>" +
      "</svg>";
    if (chartLegendEl) {
      chartLegendEl.innerHTML = series.map(function (name, si) {
        return "<span><i style=\"background:" + CHART_COLORS[si % CHART_COLORS.length] +
          '"></i>' + esc(name) + "</span>";
      }).join("");
      if (hasMedian) {
        const samples = medianValues.map(function (item) { return item ? item.samples : 0; });
        const min = Math.min.apply(null, samples);
        const maxSamples = Math.max.apply(null, samples);
        const matching = chartWindow === "24h" ? "same UTC weekday + hour in previous 4 weeks relative to each bucket date" :
          "same UTC weekday in previous 12 weeks relative to each bucket date";
        const info = "Total across all providers/models before median; " + matching + "; " + min +
          (min === maxSamples ? "" : "–" + maxSamples) + " peer days per bucket, including zero-use days; " +
          "dates before ledger history excluded; insufficient history reduces sample counts; unavailable points omitted; bucket itself and future dates excluded." +
          (chartMode === "cumulative" ? " Running sum of bucket medians." : "");
        chartLegendEl.innerHTML += '<span class="usage-median-legend" title="' + esc(info) +
          '"><i class="usage-median-swatch"></i>Total median</span>';
      }
    }
    if (chartTableEl) {
      function cellValue(v) {
        if (chartMetric === "cost") return currency(v);
        if (chartMetric === "requests") return exact(v);
        return exact(v);
      }
      const rowLabel = chartGroup === "model" ? "Model" : "Provider";
      const head = "<tr><th scope=\"col\">" + esc(rowLabel) + "</th>" +
        buckets.map(function (bucket) {
          return "<th scope=\"col\">" + esc(bucketLabel(bucket)) + "</th>";
        }).join("") + "</tr>";
      const body = series.map(function (name, si) {
        return "<tr><th scope=\"row\"><i style=\"background:" +
          CHART_COLORS[si % CHART_COLORS.length] + '"></i>' + esc(name) + "</th>" +
          buckets.map(function (bucket) {
            return "<td>" + esc(cellValue(values[bucket][name] || 0)) + "</td>";
          }).join("") + "</tr>";
      }).join("");
      chartTableEl.innerHTML = "<thead>" + head + "</thead><tbody>" + body + "</tbody>";
    }
  }

  function render(data) {
    dashboard = data;
    renderUsageChart();
  }

  function historyPlotSize(targetEl) {
    const mobile = window.matchMedia("(max-width: 720px)").matches;
    const h = mobile ? 220 : 180;
    const w = Math.max(280, Math.round((targetEl && targetEl.clientWidth) || (mobile ? 360 : 760)));
    return { w: w, h: h };
  }

  // Smallest round axis top above value: steps of 1, 2, 2.5 or 5 x 10^k,
  // three to six intervals.
  function niceAxis(value) {
    if (!(value > 0)) return { max: 1, step: 0.25, n: 4 };
    const top = Math.floor(Math.log10(value));
    let best = null;
    for (let k = top - 2; k <= top + 1; k++) {
      [1, 2, 2.5, 5].forEach(function (m) {
        const step = m * Math.pow(10, k);
        const n = Math.ceil(value / step - 1e-9);
        if (n < 3 || n > 6) return;
        const max = n * step;
        if (!best || max < best.max - 1e-9 || (Math.abs(max - best.max) < 1e-9 && n < best.n)) {
          best = { max: max, step: step, n: n };
        }
      });
    }
    return best || { max: value, step: value / 4, n: 4 };
  }

  // Round local times for the x axis: the smallest step from the list that
  // keeps the ticks within maxTicks, aligned to local midnight.
  const TIME_STEPS_MIN = [15, 30, 60, 120, 180, 360, 720, 1440, 2880, 10080];
  function timeTicks(t0, t1, maxTicks) {
    const span = t1 - t0;
    if (!(span > 0)) return { ticks: [t0], step: 0 };
    let step = TIME_STEPS_MIN[TIME_STEPS_MIN.length - 1] * 60000;
    for (let i = 0; i < TIME_STEPS_MIN.length; i++) {
      const s = TIME_STEPS_MIN[i] * 60000;
      if (Math.floor(span / s) + 1 <= maxTicks) { step = s; break; }
    }
    const midnight = new Date(t0);
    midnight.setHours(0, 0, 0, 0);
    const local = new Date(t0);
    const minutes = local.getHours() * 60 + local.getMinutes() +
      local.getSeconds() / 60 + local.getMilliseconds() / 60000;
    const stepMinutes = step / 60000;
    midnight.setMinutes(Math.ceil(minutes / stepMinutes) * stepMinutes);
    let t = midnight.getTime();
    const ticks = [];
    // Advance the local calendar, so DST does not turn noon into 13:00 or
    // midnight into 23:00. Bound the loop and require progress even in a
    // timezone with an unusual clock transition.
    for (let i = 0; t <= t1 && i < maxTicks + 2; i++) {
      if (t >= t0) ticks.push(t);
      const next = new Date(t);
      next.setMinutes(next.getMinutes() + stepMinutes);
      t = next.getTime() > t ? next.getTime() : t + step;
    }
    return { ticks: ticks.length >= 2 ? ticks : [t0, t1], step: ticks.length >= 2 ? step : 0 };
  }

  function renderHistoryLineChart(cfg) {
    const targetEl = cfg.el;
    const legendEl = cfg.legendEl;
    if (!targetEl) return;
    const points = (cfg.points || []).filter(function (pt) {
      return Number.isFinite(Date.parse(pt.timestamp || ""));
    });
    const seriesList = cfg.series || [];
    const hidden = cfg.hidden || {};
    const getValue = cfg.getValue;
    const formatValue = cfg.formatValue;
    const formatAxis = cfg.formatAxis;
    const targetLine = cfg.targetLine;
    const targetLabel = cfg.targetLabel;

    if (!points.length || !seriesList.length) {
      targetEl.innerHTML = '<p class="usage-chart-empty">' + esc(cfg.emptyMsg || "No history recorded yet") + '</p>';
      if (legendEl) legendEl.innerHTML = "";
      return;
    }

    let shown = seriesList.filter(function (s) { return !hidden[s.id]; });
    // A period change can remove the only visible provider from the window.
    // Keep one of the providers that are still present visible.
    if (!shown.length) {
      delete hidden[seriesList[0].id];
      shown = [seriesList[0]];
    }
    const times = points.map(function (pt) { return Date.parse(pt.timestamp); });
    const t0 = times[0];
    const spanMs = times[times.length - 1] - t0;

    // Autozoom: the y axis fits the lines on screen in this window, plus the
    // target line when there is one.
    let maxVal = targetLine || 0;
    points.forEach(function (pt) {
      shown.forEach(function (s) {
        const v = getValue(pt, s.id);
        if (v != null && Number.isFinite(v) && v > maxVal) maxVal = v;
      });
    });
    const axis = niceAxis(maxVal * 1.05);
    const max = axis.max;

    const plot = historyPlotSize(targetEl);
    const w = plot.w;
    const h = plot.h;
    const padL = 46;
    const padR = 14;
    const padT = 14;
    const padB = 24;
    const innerW = w - padL - padR;
    const innerH = h - padT - padB;
    function xOf(t) {
      return spanMs > 0 ? padL + ((t - t0) / spanMs) * innerW : padL + innerW / 2;
    }
    function yOf(v) {
      return padT + innerH * (1 - Math.max(0, Math.min(max, v)) / max);
    }

    let axisSvg = "";
    for (let i = 0; i <= axis.n; i++) {
      const v = axis.step * i;
      const y = yOf(v);
      axisSvg += '<line x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (w - padR) +
        '" y2="' + y.toFixed(1) + '" stroke="#2c2c30" stroke-width="1"/>';
      axisSvg += '<text class="usage-chart-scale usage-chart-ylabel" x="' + (padL - 8) + '" y="' + (y + 4).toFixed(1) +
        '" fill="#c9c9cf" font-size="11" text-anchor="end">' + esc(formatAxis(v)) + "</text>";
    }

    let targetSvg = "";
    if (targetLine != null && targetLine <= max) {
      const ty = yOf(targetLine);
      targetSvg = '<line x1="' + padL + '" y1="' + ty.toFixed(1) + '" x2="' + (w - padR) +
        '" y2="' + ty.toFixed(1) + '" stroke="#7ec699" stroke-width="1.2" stroke-dasharray="4 3"/>' +
        '<text class="usage-chart-scale" x="' + (w - padR) + '" y="' + (ty - 4).toFixed(1) +
        '" fill="#7ec699" font-size="10" text-anchor="end">' + esc(targetLabel || "") + '</text>';
    }

    // Where a series has no value, or the recorder was quiet far longer than
    // usual, the line continues as a faint dashed bridge instead of a solid
    // segment that would suggest data in between.
    const steps = [];
    for (let i = 1; i < times.length; i++) steps.push(times[i] - times[i - 1]);
    steps.sort(function (a, b) { return a - b; });
    const typical = steps.length ? steps[Math.floor(steps.length / 2)] : 0;
    const gapLimit = Math.max(3 * typical, spanMs / 12);
    function pointTime(t) {
      const d = new Date(t);
      const clock = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      return spanMs > 20 * HOUR_MS
        ? d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + clock
        : clock;
    }
    const dotR = points.length > 80 ? 1.8 : 2.6;

    let linesSvg = "";
    shown.forEach(function (s) {
      const color = s.color;
      let solid = "";
      let bridges = "";
      let dotsSvg = "";
      let prev = null;
      let hole = false;
      points.forEach(function (pt, index) {
        const v = getValue(pt, s.id);
        if (v == null || !Number.isFinite(v)) {
          hole = true;
          return;
        }
        const t = times[index];
        const x = xOf(t).toFixed(1);
        const y = yOf(v).toFixed(1);
        if (!prev) {
          solid += " M " + x + " " + y;
        } else if (!hole && t - prev.t <= gapLimit) {
          solid += " L " + x + " " + y;
        } else {
          bridges += " M " + prev.x + " " + prev.y + " L " + x + " " + y;
          solid += " M " + x + " " + y;
        }
        prev = { t: t, x: x, y: y };
        hole = false;
        dotsSvg += '<circle cx="' + x + '" cy="' + y + '" r="' + dotR + '" fill="' + color + '">' +
          "<title>" + esc(s.name) + " · " + esc(formatValue(v)) + " · " + esc(pointTime(t)) + "</title></circle>";
      });
      if (bridges) {
        linesSvg += '<path class="usage-chart-bridge" data-bridge="' + esc(s.id) + '" d="' + bridges.trim() +
          '" fill="none" stroke="' + color + '" stroke-width="1.5" stroke-dasharray="3 4" opacity="0.5"/>';
      }
      if (solid) {
        linesSvg += '<path data-series="' + esc(s.id) + '" d="' + solid.trim() + '" fill="none" stroke="' + color +
          '" stroke-width="2" stroke-linejoin="round"/>';
      }
      linesSvg += dotsSvg;
    });

    // Round local times; midnight shows the date.
    const tickSet = timeTicks(t0, t0 + spanMs, innerW < 420 ? 3 : 6);
    let timeSvg = "";
    let gridX = "";
    tickSet.ticks.forEach(function (t) {
      const d = new Date(t);
      const atMidnight = d.getHours() === 0 && d.getMinutes() === 0;
      const byDay = tickSet.step >= 24 * HOUR_MS || (tickSet.step === 0 && spanMs > 36 * HOUR_MS);
      const text = byDay || (atMidnight && tickSet.step > 0)
        ? d.toLocaleDateString([], { month: "short", day: "numeric" })
        : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const x = xOf(t);
      const anchor = x < padL + 24 ? "start" : x > w - padR - 24 ? "end" : "middle";
      gridX += '<line x1="' + x.toFixed(1) + '" y1="' + padT + '" x2="' + x.toFixed(1) + '" y2="' + (padT + innerH) +
        '" stroke="#232327" stroke-width="1"/>';
      timeSvg += '<text class="usage-chart-scale usage-chart-time" x="' + x.toFixed(1) + '" y="' + (h - 8) +
        '" fill="#c9c9cf" font-size="11" text-anchor="' + anchor + '">' + esc(text) + "</text>";
    });

    targetEl.innerHTML =
      '<svg viewBox="0 0 ' + w + " " + h + '" preserveAspectRatio="xMidYMid meet" aria-hidden="true">' +
      gridX + axisSvg + targetSvg + linesSvg + timeSvg +
      "</svg>";

    if (legendEl) {
      legendEl.innerHTML = seriesList.map(function (s) {
        const off = !!hidden[s.id];
        let last = null;
        for (let i = points.length - 1; i >= 0 && last == null; i--) {
          const v = getValue(points[i], s.id);
          if (v != null && Number.isFinite(v)) last = v;
        }
        const valStr = last != null ? " (" + formatValue(last) + ")" : "";
        return '<button type="button" class="usage-history-key' + (off ? " is-off" : "") +
          '" data-series="' + esc(s.id) + '" aria-pressed="' + (off ? "false" : "true") +
          '" title="' + (off ? "Show " : "Hide ") + esc(s.name) + '">' +
          '<i style="background:' + s.color + '"></i>' + esc(s.name) + esc(valStr) + "</button>";
      }).join("");
    }
  }

  function hasPace(item) {
    return item && item.pace_used_percent != null && Number.isFinite(Number(item.pace_used_percent));
  }

  function renderSubscriptionsPaceChart() {
    if (!subChartEl) return;
    // A point where no plan has a pace carries nothing to draw.
    let points = (historyCharts.sub.history || []).map(function (h) {
      return {
        timestamp: h.timestamp,
        items: h.subscriptions || [],
      };
    }).filter(function (h) { return (h.items || []).some(hasPace); });

    if (!points.length) {
      const liveCards = Object.values(subscriptionPaceCards);
      if (liveCards.length) {
        points = [{
          timestamp: new Date().toISOString(),
          items: liveCards.map(function (c) {
            return {
              id: c.id,
              name: c.name || c.id,
              pace_used_percent: c.pace_used_percent,
            };
          }),
        }];
      }
    }

    const seriesMap = {};
    points.forEach(function (pt) {
      pt.items.forEach(function (item) {
        if (item.pace_used_percent != null && Number.isFinite(Number(item.pace_used_percent))) {
          seriesMap[item.id] = item.name || item.id;
        }
      });
    });

    const series = Object.keys(seriesMap).sort().map(function (id, idx) {
      return { id: id, name: seriesMap[id], color: CHART_COLORS[idx % CHART_COLORS.length] };
    });

    renderHistoryLineChart({
      el: subChartEl,
      legendEl: subChartLegendEl,
      points: points,
      series: series,
      hidden: historyCharts.sub.hidden,
      emptyMsg: "Pace history is recorded on each update",
      targetLine: 100,
      targetLabel: "100% Target pace",
      getValue: function (pt, id) {
        const found = (pt.items || []).find(function (i) { return i.id === id; });
        return (found && found.pace_used_percent != null) ? Number(found.pace_used_percent) : null;
      },
      formatValue: function (v) { return Math.round(v) + "%"; },
      formatAxis: function (v) { return Math.round(v) + "%"; },
    });
  }

  function renderApiBalanceChart() {
    if (!apiChartEl) return;
    let points = (historyCharts.api.history || []).map(function (h) {
      return {
        timestamp: h.timestamp,
        items: h.api || [],
      };
    }).filter(function (h) { return (h.items || []).length > 0; });

    if (!points.length) {
      const cards = apiGridEl ? apiGridEl.querySelectorAll(".usage-sub-card") : [];
      if (cards.length) {
        const liveItems = [];
        cards.forEach(function (card) {
          const pid = card.getAttribute("data-provider");
          const nameEl = card.querySelector(".usage-sub-name");
          const heroVal = card.querySelector(".usage-sub-hero-value");
          let val = null;
          if (heroVal && heroVal.textContent.startsWith("$")) {
            val = parseFloat(heroVal.textContent.replace("$", ""));
          }
          if (pid && val != null && Number.isFinite(val)) {
            liveItems.push({ id: pid, name: (nameEl ? nameEl.textContent : pid), remaining: val });
          }
        });
        if (liveItems.length) {
          points = [{ timestamp: new Date().toISOString(), items: liveItems }];
        }
      }
    }

    const seriesMap = {};
    points.forEach(function (pt) {
      pt.items.forEach(function (item) {
        if (item.remaining != null && Number.isFinite(Number(item.remaining))) {
          seriesMap[item.id] = item.name || item.id;
        }
      });
    });

    const series = Object.keys(seriesMap).sort().map(function (id, idx) {
      return { id: id, name: seriesMap[id], color: CHART_COLORS[idx % CHART_COLORS.length] };
    });

    renderHistoryLineChart({
      el: apiChartEl,
      legendEl: apiChartLegendEl,
      points: points,
      series: series,
      hidden: historyCharts.api.hidden,
      emptyMsg: "API balance history is recorded on each update",
      getValue: function (pt, id) {
        const found = (pt.items || []).find(function (i) { return i.id === id; });
        return (found && found.remaining != null) ? Number(found.remaining) : null;
      },
      formatValue: function (v) { return "$" + v.toFixed(2); },
      formatAxis: function (v) {
        if (v >= 100) return "$" + Math.round(v);
        if (v >= 1) return "$" + v.toFixed(1);
        return "$" + v.toFixed(2);
      },
    });
  }

  function renderAllHistoryCharts() {
    renderSubscriptionsPaceChart();
    renderApiBalanceChart();
  }

  function storedHistoryPeriod(key) {
    const group = document.querySelector('[data-history-chart="' + key + '"]');
    if (!group) return "";
    let saved = "";
    try { saved = localStorage.getItem(HISTORY_PERIOD_KEY + key) || ""; } catch (_) {}
    if (saved && group.querySelector('[data-history-period="' + CSS.escape(saved) + '"]')) return saved;
    const active = group.querySelector("button.active[data-history-period]");
    return active ? active.getAttribute("data-history-period") : "";
  }

  function syncHistoryPeriodButtons(key) {
    const group = document.querySelector('[data-history-chart="' + key + '"]');
    if (!group) return;
    group.querySelectorAll("[data-history-period]").forEach(function (button) {
      const on = button.getAttribute("data-history-period") === historyCharts[key].period;
      button.classList.toggle("active", on);
      button.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  function renderHistoryChart(key) {
    if (key === "sub") renderSubscriptionsPaceChart();
    else renderApiBalanceChart();
  }

  // The newest request wins: a slow answer for a period the user already
  // left is dropped.
  async function loadHistory(key) {
    const chart = historyCharts[key];
    const seq = ++chart.seq;
    const query = chart.period ? "?period=" + encodeURIComponent(chart.period) : "";
    try {
      const res = await fetch(BASE + "/api/usage/pace-history" + query, {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (res.ok) {
        const data = await res.json().catch(function () { return {}; });
        if (seq !== chart.seq) return;
        chart.history = (data && data.history) || [];
      }
    } catch (_) {}
    if (seq !== chart.seq) return;
    renderHistoryChart(key);
  }

  function loadPaceHistory() {
    loadHistory("sub");
    loadHistory("api");
  }

  async function loadUsage() {
    const sequence = ++requestNumber;
    refreshEl.classList.add("is-refreshing");
    refreshEl.setAttribute("aria-busy", "true");
    statusEl.hidden = true;
    try {
      const response = await fetch(BASE + "/api/usage?days=365", {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "Usage data is unavailable");
      if (sequence !== requestNumber) return;
      render(data);
      appEl.classList.remove("is-loading");
      if (updatedEl) {
        updatedEl.textContent = "Updated " + new Date().toLocaleTimeString([], {
          hour: "2-digit", minute: "2-digit",
        });
      }
    } catch (error) {
      if (sequence !== requestNumber) return;
      statusEl.textContent = error && error.message
        ? error.message
        : "Usage data is unavailable";
      statusEl.hidden = false;
      appEl.classList.remove("is-loading");
    } finally {
      if (sequence === requestNumber) {
        refreshEl.classList.remove("is-refreshing");
        refreshEl.removeAttribute("aria-busy");
      }
    }
  }

  async function loadGenerations() {
    if (!generationsBodyEl) return;
    const query = new URLSearchParams({ limit: String(LOG_LIMIT) });
    if (logProvider) query.set("provider", logProvider);
    if (logAgent) query.set("agent", logAgent);
    if (logModel) query.set("model", logModel);
    try {
      const response = await fetch(BASE + "/api/logs/generations?" + query.toString(), {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || "Generation log unavailable");
      renderGenerations(data);
    } catch (error) {
      generationsBodyEl.innerHTML =
        '<tr><td colspan="11" class="usage-na">' +
        esc(error.message || "Generation log unavailable") +
        "</td></tr>";
    }
  }

  function loadPricing() {
    return window.saUsagePrices ? window.saUsagePrices.load() : Promise.resolve();
  }

  document.querySelectorAll("[data-usage-tab]").forEach(function (button) {
    button.addEventListener("click", function () {
      setUsageTab(button.dataset.usageTab);
    });
  });
  window.addEventListener("hashchange", function () {
    const hash = location.hash.replace("#", "");
    if (hash === "router") {
      window.location.replace(BASE + "/models#router");
      return;
    }
    setUsageTab(hash);
  });
  setUsageTab((function () {
    const hash = location.hash.replace("#", "");
    if (hash === "router") {
      window.location.replace(BASE + "/models#router");
      return "usage";
    }
    return USAGE_TABS.indexOf(hash) >= 0 ? hash : "usage";
  })());

  if (heatmapEl) {
    heatmapEl.addEventListener("mouseover", function (event) {
      const cell = event.target.closest(".usage-heat-cell[data-summary]");
      if (cell && selectionEl) selectionEl.textContent = cell.dataset.summary || "";
    });
    heatmapEl.addEventListener("focusin", function (event) {
      const cell = event.target.closest(".usage-heat-cell[data-summary]");
      if (cell && selectionEl) selectionEl.textContent = cell.dataset.summary || "";
    });
  }

  function bindChartToggle(attr, current, setter) {
    document.querySelectorAll("[" + attr + "]").forEach(function (button) {
      button.addEventListener("click", function () {
        setter(button.getAttribute(attr));
        document.querySelectorAll("[" + attr + "]").forEach(function (candidate) {
          const active = candidate.getAttribute(attr) === current();
          candidate.classList.toggle("active", active);
          candidate.setAttribute("aria-pressed", active ? "true" : "false");
        });
        if (dashboard) renderUsageChart();
      });
    });
  }
  bindChartToggle("data-chart-window", function () { return chartWindow; }, function (v) { chartWindow = v; });
  bindChartToggle("data-chart-group", function () { return chartGroup; }, function (v) { chartGroup = v; });
  bindChartToggle("data-chart-metric", function () { return chartMetric; }, function (v) { chartMetric = v; });
  bindChartToggle("data-chart-mode", function () { return chartMode; }, function (v) { chartMode = v; });

  document.querySelectorAll("[data-usage-view]").forEach((button) => {
    button.addEventListener("click", function () {
      activityView = button.dataset.usageView === "weekly" ? "weekly" : "daily";
      document.querySelectorAll("[data-usage-view]").forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("active", active);
        candidate.setAttribute("aria-pressed", active ? "true" : "false");
      });
      if (dashboard) renderActivity(dashboard);
    });
  });

  function bindLogFilter(el, setter) {
    if (!el) return;
    el.addEventListener("change", function () {
      setter(el.value || "");
      loadGenerations();
    });
  }
  bindLogFilter(logFilterProviderEl, function (v) { logProvider = v; });
  bindLogFilter(logFilterAgentEl, function (v) { logAgent = v; });
  bindLogFilter(logFilterModelEl, function (v) { logModel = v; });

  function formatUpdatedAt(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    const now = new Date();
    if (date.toDateString() === now.toDateString()) return date.toLocaleTimeString();
    return date.toLocaleString();
  }

  function formatReset(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    const delta = date.getTime() - Date.now();
    if (delta <= 0) return "resets now";
    const minutes = Math.round(delta / 60000);
    if (minutes < 60) return "resets in " + minutes + "m";
    const hours = Math.floor(minutes / 60);
    if (hours < 48) return "resets in " + hours + "h " + (minutes % 60) + "m";
    return "resets " + date.toLocaleString();
  }

  const SUBSCRIPTION_LOGOS = {
    codex: { file: "codex.svg", dir: "agent-logos", fit: "contain" },
    claude: { file: "claude.svg", dir: "agent-logos", fit: "contain" },
    cursor: { file: "cursor.svg", dir: "agent-logos", fit: "cover" },
    agy: { file: "antigravity.png", dir: "agent-logos", fit: "cover" },
    openrouter: { file: "openrouter.svg", dir: "provider-logos", fit: "contain" },
    deepseek: { file: "deepseek.svg", dir: "provider-logos", fit: "contain" },
    xai: { file: "xai.svg", dir: "provider-logos", fit: "contain" },
    cerebras: { file: "cerebras.svg", dir: "provider-logos", fit: "contain" },
    kimi: { file: "kimi.png", dir: "agent-logos", fit: "cover" },
    zcode: { file: "zcode.png", dir: "agent-logos", fit: "cover" },
    mimo: { file: "xiaomi.png", dir: "provider-logos", fit: "cover" },
    opencode: { file: "opencode.svg", dir: "agent-logos", fit: "contain" },
    devin: { file: "devin.png", dir: "agent-logos", fit: "cover" },
  };

  function subscriptionLogo(providerId, providerName) {
    const spec = SUBSCRIPTION_LOGOS[providerId] || null;
    const letter = esc(String(providerName || providerId || "?").charAt(0).toUpperCase());
    if (!spec) {
      return '<span class="usage-sub-logo usage-sub-logo-fallback" aria-hidden="true">' + letter + "</span>";
    }
    const src = BASE + "/static/" + spec.dir + "/" + spec.file;
    return '<span class="usage-sub-logo" aria-hidden="true">' +
      '<img class="usage-sub-logo-img usage-sub-logo-' + spec.fit + '" src="' + esc(src) +
      '" alt="" width="40" height="40" loading="eager" decoding="async">' +
      "</span>";
  }

  function formatWindowAmount(window) {
    const unit = String((window && window.unit) || "").toLowerCase();
    if (window.used != null && window.limit != null) {
      if (unit === "usd") {
        return "$" + Number(window.used).toFixed(2) + " / $" + Number(window.limit).toFixed(0);
      }
      return exact(window.used) + " / " + exact(window.limit);
    }
    if (unit === "usd" && window.remaining != null) {
      if (window.id === "purchased") {
        return "$" + Number(window.remaining).toFixed(2) + " total";
      }
      return "$" + Number(window.remaining).toFixed(2) + " left";
    }
    if (window.used_percent == null) return "—";
    return quotaPercent(window.used_percent) + " used";
  }

  // The headline reports the window that actually constrains the plan, not
  // window #0. Window order is a vendor detail: Claude lists an idle 5-hour
  // window first, so reading windows[0] advertised "100% left" while the Weekly
  // window - the real constraint - sat at 88%. Percentage rows therefore bind on
  // the lowest remaining_percent; a dollar balance keeps its own row so prepaid
  // credit cards stay deterministic.
  function bindingWindow(windows) {
    const rows = windows || [];
    let usd = null;
    let binding = null;
    let lowest = Infinity;
    rows.forEach(function (window) {
      if (!window) return;
      if (!usd && String(window.unit || "").toLowerCase() === "usd" && window.remaining != null) {
        usd = window;
      }
      const remain = window.remaining_percent == null ? NaN : Number(window.remaining_percent);
      if (Number.isFinite(remain) && remain < lowest) {
        lowest = remain;
        binding = window;
      }
    });
    return usd || binding || rows[0] || {};
  }

  function providerSummary(provider, loading, kind, hero) {
    const used = !loading && provider.pace_status && provider.pace_used_percent != null
      ? Number(provider.pace_used_percent) : NaN;
    const shown = Number.isFinite(used) ? Math.round(used) : NaN;
    let fill = Number.isFinite(used) ? Math.max(0, Math.min(100, used)) : 0;
    let zone = !Number.isFinite(shown) ? "unknown" : shown > 100 ? "over" : shown > 85 ? "warn" : "ok";
    let value = loading ? "…" : provider.pace_limit_hit && provider.pace_label
      ? provider.pace_label : Number.isFinite(shown) ? shown + "%" : "—";
    const label = kind === "subscription" ? "Pace" : "Balance";
    if (kind !== "subscription") {
      const binding = bindingWindow(provider.windows || []);
      const left = binding.remaining != null && Number(binding.limit) > 0
        ? 100 * Number(binding.remaining) / Number(binding.limit)
        : binding.remaining_percent != null ? Number(binding.remaining_percent)
        : binding.used_percent != null ? 100 - Number(binding.used_percent) : NaN;
      fill = !loading && Number.isFinite(left) ? Math.max(0, Math.min(100, left)) : 0;
      value = hero;
      const known = !loading && hero !== "—" && hero !== "console";
      const empty = (binding.remaining != null && Number(binding.remaining) <= 0) || left <= 0;
      zone = !known ? "unknown" : empty ? "over" : left <= 20 ? "warn" : "ok";
    }
    return '<summary class="usage-sub-summary is-' + zone + '" style="--pace-fill:' + fill + '%">' +
      subscriptionLogo(provider.id, provider.name) +
      '<span class="usage-sub-summary-name">' + esc(provider.name || provider.id) + '</span>' +
      '<span class="usage-sub-summary-pace">' + label + ' <strong>' + esc(value) + '</strong></span>' +
      '<svg class="usage-sub-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" ' +
        'stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>' +
      '</summary>';
  }

  function renderProviderCard(provider, loading) {
    const windows = provider.windows || [];
    const binding = bindingWindow(windows);
    const remain = binding.remaining_percent == null ? NaN : Number(binding.remaining_percent);
    // Subscription cards list their pace first among the limits, drawn like
    // the vendor's own rows below it. API balances keep their dollar headline.
    const pace = (!loading && window.saUsagePace) ? window.saUsagePace.rowHtml(provider) : "";
    let hero = "—";
    let heroBinding = null;
    if (loading) {
      hero = "…";
    } else if (String(binding.unit || "").toLowerCase() === "usd" && binding.remaining != null) {
      hero = "$" + Number(binding.remaining).toFixed(2);
    } else if (Number.isFinite(remain)) {
      hero = quotaPercent(remain);
      heroBinding = binding;
    } else if (provider.console_url) {
      hero = "console";
    }
    const visibleWindows = windows.filter(function (window) {
      return window.used != null || window.limit != null || window.remaining != null ||
        window.used_percent != null;
    });
    const bars = loading
      ? '<div class="usage-sub-skel" aria-hidden="true"></div><div class="usage-sub-skel" aria-hidden="true"></div>'
      : pace + visibleWindows.map(function (window) {
          const pct = Number(window.used_percent);
          const width = Number.isFinite(pct) ? Math.max(0, Math.min(100, pct)) : 0;
          const heat = width >= 100 ? "is-full" : width >= 80 ? "is-hot" : "";
          return '<div class="usage-sub-window" data-window="' + esc(window.id || "") + '">' +
            '<div class="usage-sub-row">' +
              "<span>" + esc(window.label || window.id || "Limit") + "</span>" +
              '<span class="usage-sub-pct">' +
                (window.pace_lane_label ? '<span class="usage-sub-hit">' + esc(window.pace_lane_label) + "</span> " : "") +
                esc(formatWindowAmount(window)) + "</span>" +
            "</div>" +
            '<div class="usage-sub-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="' +
              esc(String(Number.isFinite(pct) ? Math.round(pct) : 0)) + '">' +
              '<span class="' + heat + '" style="width:' + width + '%"></span>' +
            "</div>" +
            (window.resets_at ? '<div class="usage-sub-reset">' + esc(formatReset(window.resets_at)) + "</div>" : "") +
          "</div>";
        }).join("");
    const bindingLabel = heroBinding && windows.length > 1
      ? String(heroBinding.label || heroBinding.id || "").trim()
      : "";
    const heroLabel = provider.console_url && hero === "console"
      ? "in console"
      : bindingLabel
        ? "left · " + bindingLabel
        : "left";
    const heroTitle = "Remaining on the most constrained window";
    const stamp = formatUpdatedAt(provider.updated_at);
    const status = loading
      ? '<p class="usage-sub-loading" data-loading-status="true"><span class="usage-sub-spinner" aria-hidden="true"></span>Updating credits…</p>'
      : (provider.error ? '<p class="usage-sub-error">' + esc(provider.error) + "</p>" : "") +
        (provider.note ? '<p class="usage-sub-note">' + esc(provider.note) + "</p>" : "") +
        (provider.console_url
          ? '<a class="usage-sub-console" href="' + esc(provider.console_url) +
            '" target="_blank" rel="noopener noreferrer">Open console</a>'
          : "");
    const updated = loading
      ? '<p class="usage-sub-updated is-updating" data-updated-at="">Updating…</p>'
      : '<p class="usage-sub-updated" data-updated-at="' + esc(provider.updated_at || "") + '">' +
          (stamp ? "Updated " + esc(stamp) : "Updated just now") +
        "</p>";
    const kind = provider.kind ||
      (window.saUsagePace && window.saUsagePace.isSubscriptionCard(provider)
        ? "subscription"
        : "balance");
    // External-agent integrity merge: the catalog decorates rows and the
    // health scan maps provider id -> {version, needs_auth, auth_hint,...}.
    const health = subscriptionHealth[provider.id] || {};
    const version = String(provider.version || health.version || "");
    // The health scan is the fresher source: once it confirms sign-in, the
    // alert clears even before the card itself is re-probed.
    const authRequired = !!(health.needs_auth || (provider.auth_required && health.auth_state !== "ok"));
    const authHint = String(provider.auth_hint || health.auth_hint || "");
    const agentName = String(health.agent || "");
    const versionChip = version && kind === "subscription"
      ? '<span class="usage-sub-version" title="CLI version">v' + esc(version) + "</span>"
      : "";
    const authAlert = authRequired && kind === "subscription"
      ? '<p class="usage-sub-auth" role="alert">' +
          "<strong>Authentication required</strong>" +
          (authHint
            ? ' — run <code class="usage-sub-auth-cmd">' + esc(authHint) + "</code> in a terminal"
            : " on this machine") +
          (agentName && health.auto_auth
            ? ' <button type="button" class="usage-sub-mini" data-agent-auth="' + esc(agentName) + '">Try sign-in</button>'
            : "") +
          (agentName
            ? ' <button type="button" class="usage-sub-mini" data-agent-recheck="' + esc(agentName) + '">Check again</button>'
            : "") +
        "</p>"
      : "";
    const footer = kind === "subscription" && agentName
      ? '<footer class="usage-sub-footer">' +
          (health.update_cmd
            ? '<button type="button" class="usage-sub-mini usage-sub-update" data-agent-update="' + esc(agentName) +
              '" data-update-cmd="' + esc(health.update_cmd) +
              '" title="' + esc(health.update_cmd) + '">Update</button>'
            : "") +
          '<span class="usage-sub-footer-status" data-agent-update-status="' + esc(agentName) + '"></span>' +
        "</footer>"
      : "";
    return '<details class="usage-sub-card' + (loading ? " is-loading" : "") +
      '" data-provider="' + esc(provider.id) + '" data-kind="' + esc(kind) +
      '" data-use-next="' + (provider.pace_use_next ? "true" : "false") +
      '" data-ok="' +
      (loading ? "pending" : (provider.ok ? "true" : "false")) +
      '" data-loading="' + (loading ? "true" : "false") +
      '" data-updated-at="' + esc(loading ? "" : (provider.updated_at || "")) + '"' +
      (expandedProviderCards.has(provider.id) ? " open" : "") + '>' +
      providerSummary(provider, loading, kind, hero) + '<div class="usage-sub-details">' +
      '<header class="usage-sub-head">' +
        '<div class="usage-sub-brand">' +
          subscriptionLogo(provider.id, provider.name) +
          '<div class="usage-sub-titles">' +
            '<h3 class="usage-sub-name">' + esc(provider.name || provider.id) + "</h3>" +
            '<span class="usage-sub-plan">' +
              esc(loading ? "updating" : (provider.plan || (provider.ok ? "plan" : "offline"))) +
            "</span>" +
            versionChip +
          "</div>" +
        "</div>" +
        (pace
          ? ""
          : '<div class="usage-sub-hero" title="' + esc(heroTitle) + '">' +
              '<span class="usage-sub-hero-value">' + esc(hero) + "</span>" +
              '<span class="usage-sub-hero-label">' + esc(heroLabel) + "</span>" +
            "</div>") +
      "</header>" +
      status +
      authAlert +
      '<div class="usage-sub-windows">' + bars + "</div>" +
      updated +
      footer +
      "</div></details>";
  }

  function findProviderArticle(providerId) {
    if (!providerId) return null;
    return document.querySelector(
      '.usage-sub-card[data-provider="' + CSS.escape(providerId) + '"]'
    );
  }

  function gridForProvider(provider) {
    if (window.saUsagePace && window.saUsagePace.isSubscriptionCard(provider)) {
      return subscriptionsGridEl;
    }
    return apiGridEl;
  }

  function rankedCatalogCards() {
    const rows = (subscriptionCatalog || []).map(function (row) {
      const live = subscriptionPaceCards[row.id];
      if (live) {
        if (!live.kind) live.kind = row.kind;
        return live;
      }
      return Object.assign({ kind: row.kind }, row);
    });
    if (!window.saUsagePace) return rows;
    return window.saUsagePace.sortCards(rows);
  }

  function relayoutSubscriptionCards() {
    const ranked = rankedCatalogCards();
    ranked.forEach(function (provider) {
      const article = findProviderArticle(provider.id);
      const grid = gridForProvider(provider);
      if (!article || !grid) return;
      grid.appendChild(article);
    });
    const subCount = subscriptionsGridEl
      ? subscriptionsGridEl.querySelectorAll(".usage-sub-card").length
      : 0;
    const apiCount = apiGridEl ? apiGridEl.querySelectorAll(".usage-sub-card").length : 0;
    if (subscriptionsGroupEl) subscriptionsGroupEl.hidden = subCount === 0;
    if (apiGroupEl) apiGroupEl.hidden = apiCount === 0;
    if (subscriptionsEmptyEl) {
      subscriptionsEmptyEl.hidden = subCount > 0;
      if (subCount === 0) {
        subscriptionsEmptyEl.textContent =
          "No signed-in subscription engines on this machine";
      }
    }
    if (apiEmptyEl) {
      apiEmptyEl.hidden = apiCount > 0;
      if (apiCount === 0) {
        apiEmptyEl.textContent =
          "No prepaid API keys on this machine";
      }
    }
  }

  function showSubscriptionsEmpty(message) {
    if (subscriptionsGridEl) subscriptionsGridEl.innerHTML = "";
    if (apiGridEl) apiGridEl.innerHTML = "";
    if (subscriptionsGroupEl) subscriptionsGroupEl.hidden = true;
    if (apiGroupEl) apiGroupEl.hidden = true;
    if (subscriptionsEmptyEl) {
      subscriptionsEmptyEl.hidden = false;
      subscriptionsEmptyEl.textContent = message || "No subscription snapshot yet";
    }
    if (apiEmptyEl) {
      apiEmptyEl.hidden = false;
      apiEmptyEl.textContent = message || "No API snapshot yet";
    }
  }

  function markCardFresh(providerId) {
    const card = findProviderArticle(providerId);
    if (!card) return;
    card.classList.remove("is-fresh");
    void card.offsetWidth;
    card.classList.add("is-fresh");
  }

  // Use next depends on every card, and cards arrive one by one, so the pace
  // rows are re-rendered from the ranked set each time a card lands.
  function applyPaceRows() {
    if (!window.saUsagePace) return;
    rankedCatalogCards().forEach(function (card) {
      const article = findProviderArticle(card.id);
      if (!article || article.getAttribute("data-loading") === "true") return;
      article.setAttribute("data-use-next", card.pace_use_next ? "true" : "false");
      const slot = article.querySelector(".usage-sub-pace-row");
      const row = window.saUsagePace.rowHtml(card);
      if (slot) {
        if (row) slot.outerHTML = row;
        else slot.remove();
      } else if (row) {
        const windows = article.querySelector(".usage-sub-windows");
        if (windows) windows.insertAdjacentHTML("afterbegin", row);
      }
    });
    const anyPace = subscriptionsGroupsEl
      ? subscriptionsGroupsEl.querySelector(".usage-sub-pace-row")
      : null;
    if (subscriptionsPaceScaleEl) {
      subscriptionsPaceScaleEl.hidden = !anyPace;
      subscriptionsPaceScaleEl.innerHTML = anyPace
        ? '<span class="usage-pace-legend-key">Pace</span>' + esc(window.saUsagePace.legendText())
        : "";
    }
  }

  function upsertProviderCard(provider, loading) {
    if (!provider || !provider.id) return;
    const catalogRow = (subscriptionCatalog || []).find(function (row) {
      return row.id === provider.id;
    }) || {};
    if (!provider.kind && catalogRow.kind) provider.kind = catalogRow.kind;
    const existing = findProviderArticle(provider.id);
    if (existing && existing.tagName === "DETAILS") {
      if (existing.open) expandedProviderCards.add(provider.id);
      else expandedProviderCards.delete(provider.id);
    }
    const html = renderProviderCard(provider, !!loading);
    const grid = gridForProvider(provider);
    if (existing) {
      existing.outerHTML = html;
    } else if (grid) {
      grid.insertAdjacentHTML("beforeend", html);
    }
    if (!loading) {
      // Keep the last complete card payload so a health re-scan can
      // re-render every card (version chips, alerts) without re-probing.
      subscriptionCardData[provider.id] = provider;
      if (provider.pace_status) {
        subscriptionPaceCards[provider.id] = provider;
      } else {
        delete subscriptionPaceCards[provider.id];
      }
    } else {
      delete subscriptionPaceCards[provider.id];
    }
    relayoutSubscriptionCards();
    applyPaceRows();
    if (!loading) {
      markCardFresh(provider.id);
    }
  }

  function renderSubscriptions(data) {
    const providers = (data && data.providers) || [];
    subscriptionCatalog = providers.map(function (provider) {
      return {
        id: provider.id,
        name: provider.name,
        kind: provider.kind || (window.saUsagePace && window.saUsagePace.isSubscriptionCard(provider)
          ? "subscription"
          : "balance"),
      };
    });
    if (!providers.length) {
      showSubscriptionsEmpty("No signed-in engines or prepaid API keys on this machine");
      renderAllHistoryCharts();
      return;
    }
    if (subscriptionsEmptyEl) subscriptionsEmptyEl.hidden = true;
    if (apiEmptyEl) apiEmptyEl.hidden = true;
    if (subscriptionsGridEl) subscriptionsGridEl.innerHTML = "";
    if (apiGridEl) apiGridEl.innerHTML = "";
    providers.forEach(function (provider) {
      upsertProviderCard(provider, false);
    });
    stampSubscriptionsMeta(data);
    renderAllHistoryCharts();
  }

  // ── External agents integrity (installed / version / authentication) ──
  const AGENT_HEALTH_LOGOS = {
    antigravity: { file: "antigravity.png", dir: "agent-logos", fit: "cover" },
    claude: { file: "claude.svg", dir: "agent-logos", fit: "contain" },
    codex: { file: "codex.svg", dir: "agent-logos", fit: "contain" },
    cursor: { file: "cursor.svg", dir: "agent-logos", fit: "cover" },
    devin: { file: "devin.png", dir: "agent-logos", fit: "cover" },
    hermes: { file: "hermes.png", dir: "agent-logos", fit: "cover" },
    kimi: { file: "kimi.png", dir: "agent-logos", fit: "cover" },
    mimo: { file: "xiaomi.png", dir: "provider-logos", fit: "cover" },
    opencode: { file: "opencode.svg", dir: "agent-logos", fit: "contain" },
    pi: { file: "pi.svg", dir: "agent-logos", fit: "contain" },
    qwen: { file: "qwen.svg", dir: "agent-logos", fit: "contain" },
    zcode: { file: "zcode.png", dir: "agent-logos", fit: "cover" },
  };

  function agentHealthLogo(agent) {
    const spec = AGENT_HEALTH_LOGOS[String(agent || "").toLowerCase()] || null;
    const letter = esc(String(agent || "?").charAt(0).toUpperCase());
    if (!spec) {
      return '<span class="usage-sub-logo usage-sub-logo-fallback" aria-hidden="true">' + letter + "</span>";
    }
    const src = BASE + "/static/" + spec.dir + "/" + spec.file;
    return '<span class="usage-sub-logo" aria-hidden="true">' +
      '<img class="usage-sub-logo-img usage-sub-logo-' + spec.fit + '" src="' + esc(src) +
      '" alt="" width="40" height="40" loading="lazy" decoding="async">' +
      "</span>";
  }

  function agentStateChip(row) {
    const state = String(row.auth_state || "");
    if (state === "ok") {
      return '<span class="usage-agent-state is-ok" title="Signed in / ready">ready</span>';
    }
    if (state === "missing") {
      const hint = String(row.auth_hint || "");
      return '<span class="usage-agent-state is-missing" role="alert" title="' +
        esc(hint ? "Run: " + hint : "Authentication required") + '">⚠ sign-in required</span>';
    }
    if (state === "not_installed") {
      return '<span class="usage-agent-state is-missing" title="' +
        esc(String(row.install_hint || row.install_cmd || "Not installed")) + '">not installed</span>';
    }
    return '<span class="usage-agent-state is-unknown" title="Could not confirm the sign-in state">status unknown</span>';
  }

  function renderAgentHealthGroup() {
    if (!agentsGridEl || !agentsGroupEl) return;
    if (!agentHealthLoaded || !agentHealthRows.length) {
      agentsGroupEl.hidden = true;
      return;
    }
    agentsGroupEl.hidden = false;
    agentsGridEl.innerHTML = agentHealthRows.map(function (row) {
      const name = String(row.agent || "");
      const version = String(row.version || "");
      return '<div class="usage-agent-row' + (row.needs_auth ? " is-alert" : "") +
        '" data-agent="' + esc(name) + '">' +
        agentHealthLogo(name) +
        '<span class="usage-agent-copy">' +
          '<span class="usage-agent-name">' + esc(row.display_name || name) +
            (version ? ' <span class="usage-sub-version" title="CLI version">v' + esc(version) + "</span>" : "") +
          "</span>" +
          '<span class="usage-agent-meta">' + agentStateChip(row) +
            (row.last_spend ? ' <span class="usage-agent-spend" title="Last spend">used ' + esc(formatUpdatedAt(row.last_spend)) + "</span>" : "") +
          "</span>" +
          (row.needs_auth && row.auth_hint
            ? '<span class="usage-agent-hint">run <code>' + esc(row.auth_hint) + "</code></span>"
            : "") +
        "</span>" +
        '<span class="usage-agent-actions">' +
          (row.auth_state === "missing" && row.auto_auth
            ? '<button type="button" class="usage-sub-mini" data-agent-auth="' + esc(name) + '">Try sign-in</button>'
            : "") +
          '<button type="button" class="usage-sub-mini" data-agent-recheck="' + esc(name) + '">Check</button>' +
          (row.update_cmd && row.installed
            ? '<button type="button" class="usage-sub-mini" data-agent-update="' + esc(name) +
              '" title="' + esc(row.update_cmd) + '">Update</button>'
            : "") +
        "</span>" +
        '<span class="usage-sub-footer-status" data-agent-update-status="' + esc(name) + '"></span>' +
      "</div>";
    }).join("");
  }

  function refreshCardsHealth() {
    Object.keys(subscriptionCardData).forEach(function (id) {
      upsertProviderCard(subscriptionCardData[id], false);
    });
  }

  function mergeHealthRow(row) {
    if (!row || !row.agent) return;
    const index = agentHealthRows.findIndex(function (item) {
      return item.agent === row.agent;
    });
    if (index >= 0) agentHealthRows[index] = row;
    else agentHealthRows.push(row);
    if (row.provider) subscriptionHealth[row.provider] = row;
  }

  async function loadAgentHealth(force) {
    try {
      const response = await fetch(
        BASE + "/api/usage/agents/health" + (force ? "?refresh=1" : ""),
        { credentials: "same-origin", cache: "no-store", headers: { Accept: "application/json" } }
      );
      if (!response.ok) return;
      const data = await response.json();
      agentHealthRows = data.agents || [];
      subscriptionHealth = Object.create(null);
      agentHealthRows.forEach(function (row) {
        if (row.provider) subscriptionHealth[row.provider] = row;
      });
      agentHealthLoaded = true;
      renderAgentHealthGroup();
      refreshCardsHealth();
    } catch (_) { /* the tab renders without the chips */ }
  }

  async function recheckAgent(agent) {
    try {
      const response = await fetch(
        BASE + "/api/usage/agents/" + encodeURIComponent(agent) + "/auth-check",
        { method: "POST", credentials: "same-origin", cache: "no-store" }
      );
      const row = await response.json();
      if (response.ok && row && row.agent) {
        mergeHealthRow(row);
        renderAgentHealthGroup();
        refreshCardsHealth();
        const provider = subscriptionCardData[row.provider];
        if (row.auth_state === "ok" && provider && provider.auth_required) {
          const data = await fetchSubscriptionProvider(provider, true);
          upsertProviderCard(data.provider || provider, false);
          stampSubscriptionsMeta(data);
          loadPaceHistory();
        }
      }
    } catch (_) { /* keep the last known state */ }
  }

  async function autoAuthAgent(agent, button) {
    const say = makeSay(agent, button);
    say("Signing in…");
    try {
      const response = await fetch(
        BASE + "/api/usage/agents/" + encodeURIComponent(agent) + "/auth",
        { method: "POST", credentials: "same-origin", cache: "no-store" }
      );
      const data = await response.json().catch(function () { return {}; });
      if (response.ok && data.ok) {
        say("Signed in");
        await recheckAgent(agent);
        say("");
      } else {
        say(data.detail || data.error || "Sign-in needs a terminal");
      }
    } catch (_) {
      say("Sign-in needs a terminal");
    }
  }

  // The row/card markup is re-rendered whenever health lands, so status
  // writes re-query through the STABLE grid host instead of holding a
  // node reference that a re-render detaches.
  function statusTarget(agent, button) {
    const row = button && button.closest(".usage-agent-row");
    if (row) {
      return {
        host: (row.closest(".usage-agents-grid") || agentsGridEl || document),
        sel: '.usage-agent-row[data-agent="' + CSS.escape(agent) +
          '"] [data-agent-update-status="' + CSS.escape(agent) + '"]',
      };
    }
    const card = button && button.closest(".usage-sub-card");
    if (card) {
      const pid = card.getAttribute("data-provider") || "";
      return {
        host: card.parentElement || subscriptionsGridEl || document,
        sel: '.usage-sub-card[data-provider="' + CSS.escape(pid) +
          '"] [data-agent-update-status="' + CSS.escape(agent) + '"]',
      };
    }
    return { host: document, sel: '[data-agent-update-status="' + CSS.escape(agent) + '"]' };
  }

  function makeSay(agent, button) {
    const target = statusTarget(agent, button);
    return function (text) {
      const el = target.host.querySelector(target.sel);
      if (el) el.textContent = text;
    };
  }

  async function pollUpdate(jobId, agent, say, button) {
    for (let attempt = 0; attempt < 200; attempt += 1) {
      await new Promise(function (resolve) { window.setTimeout(resolve, 3000); });
      let job = null;
      try {
        const response = await fetch(
          BASE + "/api/usage/agents/update/" + encodeURIComponent(jobId),
          { credentials: "same-origin", cache: "no-store" }
        );
        if (response.ok) job = await response.json();
      } catch (_) { /* transient poll failure */ }
      if (!job) continue;
      if (job.status === "running") { say("Updating…"); continue; }
      if (job.status === "ok") {
        await recheckAgent(agent);
        say("Updated");
      } else {
        say("Failed: " + String(job.output || "").split("\n").pop());
      }
      if (button) button.disabled = false;
      return;
    }
    say("Update still running — check again in a moment");
    if (button) button.disabled = false;
  }

  async function updateAgent(agent, button) {
    const say = makeSay(agent, button);
    if (button) button.disabled = true;
    say("Starting…");
    try {
      const response = await fetch(
        BASE + "/api/usage/agents/" + encodeURIComponent(agent) + "/update",
        { method: "POST", credentials: "same-origin", cache: "no-store" }
      );
      const data = await response.json().catch(function () { return {}; });
      if (!response.ok) {
        say(data.error || "Update refused");
        if (button) button.disabled = false;
        return;
      }
      await pollUpdate(data.job_id, agent, say, button);
    } catch (_) {
      say("Update could not start");
      if (button) button.disabled = false;
    }
  }

  [subscriptionsGridEl, apiGridEl].forEach(function (host) {
    if (!host) return;
    host.addEventListener("toggle", function (event) {
      const card = event.target;
      if (!card.matches("details.usage-sub-card")) return;
      if (card.open) expandedProviderCards.add(card.dataset.provider);
      else expandedProviderCards.delete(card.dataset.provider);
    }, true);
  });

  [agentsGridEl, subscriptionsGridEl, apiGridEl].forEach(function (host) {
    if (!host) return;
    host.addEventListener("click", function (event) {
      const button = event.target.closest("button");
      if (!button || !host.contains(button)) return;
      const recheck = button.getAttribute("data-agent-recheck");
      const auth = button.getAttribute("data-agent-auth");
      const update = button.getAttribute("data-agent-update");
      if (recheck) recheckAgent(recheck);
      else if (auth) autoAuthAgent(auth, button);
      else if (update) updateAgent(update, button);
    });
  });

  if (agentsScanEl) {
    agentsScanEl.addEventListener("click", function () {
      agentsScanEl.disabled = true;
      loadAgentHealth(true).finally(function () { agentsScanEl.disabled = false; });
    });
  }

  function stampSubscriptionsMeta(data) {
    const when = data && data.updated_at ? new Date(data.updated_at) : null;
    const stamp = when && !Number.isNaN(when.getTime()) ? when.toLocaleTimeString() : "";
    if (subscriptionsMetaEl) {
      const done = subscriptionsGridEl
        ? subscriptionsGridEl.querySelectorAll('.usage-sub-card[data-loading="false"]').length
        : 0;
      const total = subscriptionsGridEl
        ? subscriptionsGridEl.querySelectorAll(".usage-sub-card").length
        : 0;
      const progress = total ? done + "/" + total : "";
      subscriptionsMetaEl.textContent =
        ((data && data.source) || "live") +
        (progress ? " · " + progress : "") +
        (stamp ? " · " + stamp : "");
    }
    if (apiMetaEl) {
      const done = apiGridEl
        ? apiGridEl.querySelectorAll('.usage-sub-card[data-loading="false"]').length
        : 0;
      const total = apiGridEl
        ? apiGridEl.querySelectorAll(".usage-sub-card").length
        : 0;
      const progress = total ? done + "/" + total : "";
      apiMetaEl.textContent =
        ((data && data.source) || "live") +
        (progress ? " · " + progress : "") +
        (stamp ? " · " + stamp : "");
    }
  }

  async function fetchSubscriptionProvider(provider, refresh) {
    const response = await fetch(
      BASE + "/api/usage/subscriptions/" + encodeURIComponent(provider.id) +
        (refresh ? "?refresh=1" : ""),
      {
        method: refresh ? "POST" : "GET",
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      }
    );
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || "Subscription limits unavailable");
    }
    return data;
  }

  async function loadSubscriptions(refresh) {
    if (!subscriptionsGroupsEl || subscriptionsLoading) return;
    subscriptionsLoading = true;
    subscriptionPaceCards = Object.create(null);
    subscriptionCardData = Object.create(null);
    loadAgentHealth(!!refresh);
    if (subscriptionsPaceScaleEl) {
      subscriptionsPaceScaleEl.hidden = true;
      subscriptionsPaceScaleEl.innerHTML = "";
    }
    if (subscriptionsUpdateEl) subscriptionsUpdateEl.disabled = true;
    if (apiUpdateEl) apiUpdateEl.disabled = true;
    try {
      const catalogResponse = await fetch(BASE + "/api/usage/subscriptions/catalog", {
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      const catalog = await catalogResponse.json().catch(() => ({}));
      if (!catalogResponse.ok) throw new Error(catalog.error || "Subscription catalog unavailable");
      const providers = catalog.providers || [];
      subscriptionCatalog = providers.map(function (provider) {
        return {
          id: provider.id,
          name: provider.name,
          kind: provider.kind || (window.saUsagePace && window.saUsagePace.isSubscriptionCard(provider)
            ? "subscription"
            : "balance"),
        };
      });
      if (!providers.length) {
        showSubscriptionsEmpty("No signed-in engines or prepaid API keys on this machine");
        return;
      }
      if (subscriptionsEmptyEl) subscriptionsEmptyEl.hidden = true;
      if (apiEmptyEl) apiEmptyEl.hidden = true;
      if (subscriptionsGridEl) subscriptionsGridEl.innerHTML = "";
      if (apiGridEl) apiGridEl.innerHTML = "";
      providers.forEach(function (provider) {
        upsertProviderCard(Object.assign({}, provider, { kind: provider.kind }), true);
      });
      stampSubscriptionsMeta({ source: catalog.source || "live" });
      await Promise.all(providers.map(async function (provider) {
        try {
          const data = await fetchSubscriptionProvider(provider, refresh);
          const card = data.provider || provider;
          if (!card.updated_at && data.updated_at) card.updated_at = data.updated_at;
          if (!card.kind) card.kind = provider.kind;
          upsertProviderCard(card, false);
          stampSubscriptionsMeta(data);
        } catch (error) {
          upsertProviderCard({
            id: provider.id,
            name: provider.name,
            kind: provider.kind,
            ok: false,
            error: error.message || "Unavailable",
            windows: [],
            updated_at: new Date().toISOString(),
          }, false);
          stampSubscriptionsMeta(catalog);
        }
      }));
    } catch (error) {
      showSubscriptionsEmpty(error.message || "Subscription limits unavailable");
    } finally {
      subscriptionsLoading = false;
      if (subscriptionsUpdateEl) subscriptionsUpdateEl.disabled = false;
      if (apiUpdateEl) apiUpdateEl.disabled = false;
      loadPaceHistory();
    }
  }

  if (subscriptionsUpdateEl) {
    subscriptionsUpdateEl.addEventListener("click", function () {
      loadSubscriptions(true);
    });
  }

  document.querySelectorAll("[data-history-chart]").forEach(function (group) {
    const key = group.getAttribute("data-history-chart");
    if (!historyCharts[key]) return;
    syncHistoryPeriodButtons(key);
    group.addEventListener("click", function (event) {
      const button = event.target.closest("[data-history-period]");
      if (!button || !group.contains(button)) return;
      const period = button.getAttribute("data-history-period");
      if (period === historyCharts[key].period) return;
      historyCharts[key].period = period;
      try { localStorage.setItem(HISTORY_PERIOD_KEY + key, period); } catch (_) {}
      syncHistoryPeriodButtons(key);
      loadHistory(key);
    });
  });

  // Legend keys hide or show a line; the axis zooms to what stays visible.
  [[subChartLegendEl, "sub"], [apiChartLegendEl, "api"]].forEach(function (pair) {
    const legendEl = pair[0];
    const key = pair[1];
    if (!legendEl) return;
    legendEl.addEventListener("click", function (event) {
      const button = event.target.closest("[data-series]");
      if (!button || !legendEl.contains(button)) return;
      const id = button.getAttribute("data-series");
      const hidden = historyCharts[key].hidden;
      if (hidden[id]) {
        delete hidden[id];
      } else if (legendEl.querySelectorAll('[data-series][aria-pressed="true"]').length > 1) {
        hidden[id] = true;
      } else {
        return; // the last visible line stays
      }
      renderHistoryChart(key);
      const again = legendEl.querySelector('[data-series="' + CSS.escape(id) + '"]');
      if (again) again.focus();
    });
  });
  if (apiUpdateEl) {
    apiUpdateEl.addEventListener("click", function () {
      loadSubscriptions(true);
    });
  }

  refreshEl.addEventListener("click", function () {
    loadUsage();
    loadGenerations();
    loadPricing();
    loadSubscriptions(false);
  });
  if (modelExpandEl) {
    modelExpandEl.addEventListener("click", function () {
      modelsExpanded = !modelsExpanded;
      if (dashboard) renderModels(dashboard);
    });
  }
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) {
      loadUsage();
      loadGenerations();
      loadPricing();
    }
  });
  window.setInterval(function () {
    if (!document.hidden) {
      loadUsage();
      loadGenerations();
      loadPricing();
    }
  }, 60000);

  let chartResizeTimer = 0;
  window.addEventListener("resize", function () {
    window.clearTimeout(chartResizeTimer);
    chartResizeTimer = window.setTimeout(function () {
      if (dashboard) renderUsageChart();
      renderAllHistoryCharts();
    }, 150);
  });

  loadUsage();
  loadGenerations();
  loadPricing();
})();
