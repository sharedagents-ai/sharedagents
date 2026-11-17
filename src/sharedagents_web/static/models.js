/* Models tab — Artificial Analysis benchmark charts.
 *
 * Three hand-rolled inline SVG bar charts (CSP is script-src 'self', so no CDN
 * chart library and no inline <script>): Intelligence, Output tokens/sec, and
 * Cost per Intelligence-Index task. Each chart sorts by its own metric, renders
 * only models that actually report it, marks the models sharedagents has enabled,
 * and ships a visually-hidden accessible <table> of the same numbers.
 *
 * Request budget note: this file fetches ONCE on load and only again when the
 * operator presses refresh (POST ?refresh=1). The endpoint is cache-first, so a
 * normal page view normally costs zero upstream requests; there is deliberately
 * no polling and no per-render fetch. */
(function () {
  "use strict";

  const BASE =
    document.querySelector('meta[name="sharedagents-base"]')?.content || "";

  // Same palette as usage.js so the two tabs read as one product.
  const CHART_COLORS = [
    "#548cff", "#8ee0b0", "#e6c36a", "#d5c6f5",
    "#ef9a90", "#7ec8e3", "#f0a06a", "#c5d4f5",
  ];
  // Marker colour for "sharedagents has this model enabled".
  const ENABLED_COLOR = "#8ee0b0";
  // The cascade steps, in the order the backend reports them, with the wording
  // the tooltip/table uses. Keep in sync with sharedagents_web.model_benchmarks._steps.
  const MATCH_STEPS = {
    id_slug: "exact model id",
    id_name: "model id vs name",
    display_name: "display name",
    alias_slug: "alias (engine prefix stripped)",
    token_set: "reordered name tokens",
  };

  // Legibility ceiling: the live population is ~650 models, ~644 of which
  // report intelligence — far too many bars to draw at once. Charts default to
  // the top N by their own metric; "Show all" draws every bar in a horizontally
  // scrollable strip. The rendered/skipped counts are always displayed.
  const TOP_N = 30;

  // Chart text wears INK tokens, never a series colour: the coloured mark beside
  // a label already carries identity, and green-on-charcoal label text was a
  // large part of why the frontier cluster read as unreadable.
  const INK_STRONG = "#ececef";
  const INK = "#c1c2c9";
  const INK_MUTED = "#9a9ba3";
  // The panel surface. Marks get a ring of it so overlapping dots stay separate,
  // and labels get a halo of it so the frontier line cannot strike through text.
  const SURFACE = "#1d1d1f";
  const LABEL_FONT_SIZE = 11;

  const PAD_L = 52;
  const PAD_R = 16;
  const PAD_T = 10;

  // `format` is the value formatter (tooltips, the accessible table); `axis` is
  // the TICK formatter, which takes the precision decided once for the whole
  // tick set. They differ on purpose: a value reads "53.4", a round tick reads
  // "55", not "55.0".
  const CHARTS = [
    {
      id: "intelligence",
      metric: "intelligence",
      order: "desc",
      rowLabel: "Intelligence Index",
      missing: "no intelligence score",
      format: formatIndex,
      axis: function (value, decimals) { return value.toFixed(decimals); },
    },
    {
      id: "speed",
      metric: "speed_tps",
      order: "desc",
      rowLabel: "Output tokens/sec",
      missing: "no speed measurement",
      format: function (value) { return formatTps(value); },
      axis: function (value) { return Math.round(value) + " t/s"; },
    },
    {
      id: "cost",
      metric: "cost_per_task_usd",
      order: "asc",
      rowLabel: "Cost per task (USD)",
      missing: "no cost-per-task estimate",
      format: function (value) { return formatUsd(value); },
      axis: function (value, decimals) { return formatUsdFixed(value, decimals); },
      usd: true,
    },
  ];

  const appEl = document.getElementById("models-app");
  const refreshEl = document.getElementById("models-refresh");
  const updatedEl = document.getElementById("models-updated");
  const statusEl = document.getElementById("models-status");
  const emptyEl = document.getElementById("models-empty");
  const emptyTitleEl = document.getElementById("models-empty-title");
  const emptyBodyEl = document.getElementById("models-empty-body");
  const emptyStepsEl = document.getElementById("models-empty-steps");
  const emptyCommandEl = document.getElementById("models-empty-command");
  const emptyEnvFileEl = document.getElementById("models-empty-envfile");
  const enabledOnlyEl = document.getElementById("models-enabled-only");
  const exactOnlyEl = document.getElementById("models-exact-only");
  const showAllEl = document.getElementById("models-show-all");
  const legendNoteEl = document.getElementById("models-legend-note");
  const attributionEl = document.getElementById("models-attribution");
  const kpiEls = {
    models: document.getElementById("models-kpi-models"),
    enabled: document.getElementById("models-kpi-enabled"),
    index: document.getElementById("models-kpi-index"),
    tier: document.getElementById("models-kpi-tier"),
    budget: document.getElementById("models-kpi-budget"),
    source: document.getElementById("models-kpi-source"),
  };

  // Pareto view + control bar. The template ships all three views and the
  // popovers; every one of these must be wired or the control renders dead.
  const viewEls = {
    pareto: document.getElementById("models-view-pareto"),
    bars: document.getElementById("models-view-bars"),
    table: document.getElementById("models-view-table"),
  };
  const viewBtnEls = {
    pareto: document.getElementById("models-view-pareto-btn"),
    bars: document.getElementById("models-view-bars-btn"),
    table: document.getElementById("models-view-table-btn"),
  };
  const controlsMetaEl = document.getElementById("models-controls-meta");
  const selectionToggleEl = document.getElementById("models-selection-toggle");
  const selectionLabelEl = document.getElementById("models-selection-label");
  const popoverSelectionEl = document.getElementById("models-popover-selection");
  const popoverCloseEl = document.getElementById("models-popover-close");
  const selectSearchEl = document.getElementById("models-select-search");
  const selectProviderEl = document.getElementById("models-select-provider");
  const selectListEl = document.getElementById("models-select-list");
  const selectCountEl = document.getElementById("models-select-count");
  const selectAllEl = document.getElementById("models-select-all");
  const selectClearEl = document.getElementById("models-select-clear");
  const presetEls = {
    pareto: document.getElementById("models-preset-pareto"),
    local: document.getElementById("models-preset-local"),
    top10: document.getElementById("models-preset-top10"),
    all: document.getElementById("models-preset-all"),
    default: document.getElementById("models-preset-default"),
  };
  const filterToggleEl = document.getElementById("models-filter-toggle");
  const popoverFilterEl = document.getElementById("models-popover-filter");
  const filterLatestEl = document.getElementById("models-filter-latest");
  const filterLocalEl = document.getElementById("models-filter-local");
  const filterFrontierEl = document.getElementById("models-filter-frontier");
  const settingsToggleEl = document.getElementById("models-settings-toggle");
  const popoverSettingsEl = document.getElementById("models-popover-settings");
  const settingLabelsEl = document.getElementById("models-setting-labels");
  const settingGridEl = document.getElementById("models-setting-grid");
  const settingLogEl = document.getElementById("models-setting-log");
  const whyEl = document.getElementById("models-why");
  const whyBodyEl = document.getElementById("models-why-body");
  const localDeploymentsEl = document.getElementById("models-local-deployments");
  const localDeploymentsListEl = document.getElementById("models-local-deployments-list");
  const tableViewEl = document.getElementById("models-table-view");
  const metaTableEl = document.getElementById("models-meta-table");
  const subtabEls = Array.from(document.querySelectorAll("[data-models-tab]"));
  const MODELS_TABS = ["benchmarks", "router", "vllm"];

  let payload = null;
  let loading = false;
  let showAll = false;
  let enabledOnly = false;
  let exactOnly = false;
  let view = "pareto";
  // null = "follow the backend's default_selection"; a Set = an explicit pick.
  let selection = null;
  let filterLatest = true;
  let filterLocal = false;
  let filterFrontier = false;
  let settingLabels = true;
  let settingGrid = true;
  let settingLog = true;
  let selectSearch = "";
  let selectProvider = "";

  // ── helpers ────────────────────────────────────────────────────────────

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function isNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function formatIndex(value) {
    return isNumber(value) ? value.toFixed(1) : "—";
  }

  function formatTps(value) {
    return isNumber(value) ? Math.round(value) + " t/s" : "—";
  }

  function formatUsd(value) {
    if (!isNumber(value)) return "—";
    if (value >= 1) return "$" + value.toFixed(2);
    if (value >= 0.01) return "$" + value.toFixed(3);
    return "$" + value.toFixed(4);
  }

  /** One precision for a whole axis / column, chosen from its smallest value.
   *
   * Per-value precision is what produced "$0.200 $0.500 $1.00 $2.00 $5.00" on
   * the cost axis and "$7.63 / $0.818" in one table column: the decimals have
   * to be decided once for the set, never per cell.
   */
  function usdDecimalsFor(values) {
    let min = Infinity;
    (values || []).forEach(function (value) {
      if (isNumber(value) && value > 0 && value < min) min = value;
    });
    if (!isFinite(min)) return 2;
    if (min >= 0.1) return 2;
    if (min >= 0.01) return 3;
    return 4;
  }

  function formatUsdFixed(value, decimals) {
    return isNumber(value) ? "$" + value.toFixed(decimals) : "—";
  }

  /** ``0.001`` → ``"$0.001"``, ``10`` → ``"$10"`` — exact, no padding.
   *
   * For a log axis spanning many decades the ticks ARE the decades, so one
   * shared precision would print "$10.0000" next to "$0.0001". Showing each
   * decade exactly is the consistent rule there.
   */
  function formatUsdExact(value) {
    if (!isNumber(value)) return "—";
    return "$" + String(Number(value.toFixed(6)));
  }

  /** A "nice" tick step: 1 / 2 / 2.5 / 5 / 10 × 10ⁿ. */
  function niceStep(span, count) {
    const raw = span / Math.max(1, count);
    if (!(raw > 0) || !Number.isFinite(raw)) return 1;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * mag;
  }

  /** Round a domain out to whole steps so ticks read 40/45/50, not 39.7/43.5. */
  function niceScale(min, max, count) {
    let lo = isNumber(min) ? min : 0;
    let hi = isNumber(max) ? max : 1;
    if (!(hi > lo)) { lo -= 1; hi += 1; }
    const step = niceStep(hi - lo, count);
    lo = Math.floor(lo / step) * step;
    hi = Math.ceil(hi / step) * step;
    const ticks = [];
    const total = Math.round((hi - lo) / step);
    for (let i = 0; i <= total; i++) {
      // Walk on the index, never by repeated addition — 0.1+0.1+0.1 drifts and
      // prints 0.30000000000000004 on an axis.
      ticks.push(lo + i * step);
    }
    return { lo: lo, hi: hi, step: step, ticks: ticks };
  }

  function tickDecimals(step) {
    if (step >= 1) return 0;
    if (step >= 0.1) return 1;
    if (step >= 0.01) return 2;
    return 3;
  }

  // Real text metrics beat a per-character estimate: a de-collision box is only
  // as good as the width it is built from, and the old 5.4px/char guess is what
  // let one label sit on top of its neighbour's marker.
  let measureTextEl = null;
  function textWidth(text, fontSize) {
    const value = String(text == null ? "" : text);
    // `typeof document` guard: these helpers are also exercised headlessly by
    // tests/test_models_page_contract.py, where there is no DOM to measure in
    // and the per-character estimate below is the answer.
    if (!measureTextEl && typeof document !== "undefined" && document.body) {
      // Parsed as markup rather than built with createElementNS: the HTML
      // parser puts <svg>/<text> in the SVG namespace for us, so this file
      // needs no namespace URL (tests/test_model_benchmarks.py asserts models.js
      // carries no http(s) reference at all).
      const host = document.createElement("div");
      host.className = "models-measure";
      host.setAttribute("aria-hidden", "true");
      host.innerHTML = '<svg width="10" height="10"><text></text></svg>';
      document.body.appendChild(host);
      measureTextEl = host.querySelector("text");
    }
    if (measureTextEl) {
      measureTextEl.setAttribute("font-size", String(fontSize));
      measureTextEl.textContent = value;
      try {
        const width = measureTextEl.getComputedTextLength();
        if (width > 0) return width;
      } catch (error) {
        /* not laid out yet — fall through to the estimate */
      }
    }
    return value.length * fontSize * 0.56;
  }

  /** Trim ``text`` until it fits ``maxPx``, with the ellipsis only at the END.
   *
   * A label clipped at both ends ("1 (Adaptive Reas…") is unreadable; one that
   * ends in an ellipsis still names its model.
   */
  function fitLabel(text, maxPx, fontSize) {
    const full = String(text == null ? "" : text);
    if (!(maxPx > 0)) return full;
    const width = textWidth(full, fontSize);
    if (width <= maxPx) return full;
    // Seed the cut from the measured ratio, then walk the last few characters
    // instead of measuring every prefix.
    let n = Math.max(1, Math.floor((full.length * maxPx) / width) - 1);
    while (n > 1 && textWidth(full.slice(0, n) + "…", fontSize) > maxPx) n -= 1;
    while (n < full.length && textWidth(full.slice(0, n + 1) + "…", fontSize) <= maxPx) n += 1;
    return full.slice(0, n).replace(/[\s·]+$/, "") + "…";
  }

  // Never mixed into a chart axis: AA prices USD per million tokens, while
  // cost_per_task is USD per Intelligence-Index task. Labelled as such.
  function formatPerMillion(value) {
    if (!isNumber(value)) return "—";
    return "$" + (value >= 1 ? value.toFixed(2) : value.toFixed(3)) + "/1M";
  }

  function providerColor(provider) {
    // The backend ships a stable per-provider palette (payload.colors, with
    // provider_colors kept for older snapshots); only fall back to the local
    // hash when a snapshot predates it, so a provider keeps one colour.
    const named =
      (payload && ((payload.colors || {})[provider] || (payload.provider_colors || {})[provider])) ||
      "";
    if (named) return named;
    const text = String(provider || "unknown").toLowerCase();
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      hash = (hash * 31 + text.charCodeAt(i)) >>> 0;
    }
    return CHART_COLORS[hash % CHART_COLORS.length];
  }

  // Artificial Analysis ships the qualifier INSIDE the name — "GPT-6 Astra
  // (max)" arrives with variant "max" — so `name + " · " + variant` printed it
  // twice ("GPT-6 Astra (max) · max"). Labels are built from the BARE name plus
  // exactly one tag instead.
  const QUALIFIER_RE = /\s*\(([^()]*)\)\s*$/;
  const EFFORT_RE = /^(?:.*\s)?(minimal|none|low|medium|high|xhigh|max)(?:\s+effort)?$/i;
  const NON_REASONING_RE = /non[-\s]?reasoning/i;

  /** ``"GPT-6 Astra (max)"`` → ``"GPT-6 Astra"``, repeatedly (mirrors the backend). */
  function bareName(value) {
    let text = String(value == null ? "" : value).trim();
    for (;;) {
      const next = text.replace(QUALIFIER_RE, "").trim();
      if (!next || next === text) return text;
      text = next;
    }
  }

  /** The single short tag that tells one model's variants apart.
   *
   * The reasoning mode is kept alongside the effort because dropping it does
   * collide in the live snapshot: Claude Sonnet 5 ships both an
   * "Adaptive Reasoning, High Effort" and a "Non-reasoning, High Effort" row.
   */
  function variantTag(variant) {
    if (!variant) return "";
    const parts = String(variant)
      .split(/[;,]/)
      .map(function (part) { return part.trim(); })
      .filter(Boolean);
    let effort = "";
    let mode = "";
    parts.forEach(function (part) {
      const hit = part.match(EFFORT_RE);
      if (hit && !effort) effort = hit[1].toLowerCase();
      if (!mode && NON_REASONING_RE.test(part)) mode = "non-reasoning";
    });
    if (effort) return mode ? mode + " " + effort : effort;
    if (mode) return mode;
    return parts[0] || "";
  }

  // "Soft" = enabled, but the join was not an exact-slug hit: the back end
  // matched through a display name, an alias guess or a reordered token set.
  function isSoft(model) {
    return !!model.enabled && !model.match_exact;
  }

  function stepLabel(model) {
    const step = model.match_step || "";
    return MATCH_STEPS[step] || step || "heuristic match";
  }

  /** The label a mark carries — never the variant twice, never a mid-glyph cut.
   *
   * ``maxPx`` (optional) trims with a trailing ellipsis for the bar chart's
   * rotated strip; the Pareto scatter measures instead of truncating, and every
   * mark keeps the untruncated name in its ``<title>`` tooltip.
   */
  function chartLabel(model, maxPx) {
    const raw = String(model.name || model.slug || "model");
    const name = bareName(raw) || raw;
    const tag = variantTag(model.variant);
    let text = tag ? name + " · " + tag : name;
    if (isSoft(model)) text = "≈ " + text;
    return maxPx ? fitLabel(text, maxPx, LABEL_FONT_SIZE) : text;
  }

  function tooltip(model, spec) {
    const bits = [String(model.name || model.slug || "model")];
    if (model.provider) bits.push(model.provider);
    bits.push(spec.rowLabel + " " + spec.format(model[spec.metric]));
    if (model.enabled) {
      const alias = (model.sharedagents || {}).alias;
      bits.push(alias ? "sharedagents enabled (" + alias + ")" : "sharedagents enabled");
      if (isSoft(model)) {
        bits.push("≈ approximate match · " + stepLabel(model));
      } else {
        bits.push("exact match · " + stepLabel(model));
      }
    }
    bits.push("in " + formatPerMillion(model.usd_per_1m_in) + " · out " + formatPerMillion(model.usd_per_1m_out));
    return bits.join(" · ");
  }

  function isMobile() {
    return window.matchMedia("(max-width: 720px)").matches;
  }

  // ── status / empty state ───────────────────────────────────────────────

  function showStatus(message) {
    if (!statusEl) return;
    statusEl.textContent = message || "";
    statusEl.hidden = !message;
  }

  function hideEmpty() {
    if (emptyEl) emptyEl.hidden = true;
  }

  function showEmpty(error) {
    if (!emptyEl) return;
    const code = (error && error.code) || "error";
    const isKeyError = code === "no_api_key" || code === "rate_limited";
    if (emptyTitleEl) {
      emptyTitleEl.textContent =
        code === "no_api_key"
          ? "Artificial Analysis API key required"
          : code === "rate_limited"
            ? "Artificial Analysis request budget exhausted"
            : "Benchmark data unavailable";
    }
    if (emptyBodyEl) {
      emptyBodyEl.textContent =
        (error && error.message) ||
        "The benchmark service could not be queried.";
    }
    // The env file is resolved server-side ($SHAREDAGENTS_HOME-aware); the markup
    // only carries the default path, so overwrite it with the file that was
    // really consulted before the empty state is revealed.
    if (error && error.env_file) {
      const envVar = error.env_var || "ARTIFICIAL_ANALYSIS_API_KEY";
      if (emptyCommandEl) {
        emptyCommandEl.textContent =
          "echo '" + envVar + "=your-key-here' >> " + error.env_file;
      }
      if (emptyEnvFileEl) emptyEnvFileEl.textContent = error.env_file;
    }
    if (emptyStepsEl) emptyStepsEl.hidden = !isKeyError;
    if (attributionEl && payload && payload.attribution) {
      attributionEl.textContent = payload.attribution;
    }
    emptyEl.hidden = false;
  }

  function setRefreshing(active) {
    if (!refreshEl) return;
    refreshEl.classList.toggle("is-refreshing", !!active);
    if (active) refreshEl.setAttribute("aria-busy", "true");
    else refreshEl.removeAttribute("aria-busy");
    refreshEl.disabled = !!active;
  }

  // ── charts ─────────────────────────────────────────────────────────────

  function chartModels(spec) {
    const models = (payload && payload.models) || [];
    const withMetric = models.filter(function (model) {
      return isNumber(model[spec.metric]);
    });
    const skipped = models.length - withMetric.length;
    // "Exact matches only" is a stricter form of "Only enabled": it drops the
    // approximate joins, so the operator can see the confident overlay alone.
    const visible = enabledOnly || exactOnly
      ? withMetric.filter(function (model) {
        return exactOnly ? (model.enabled && model.match_exact) : model.enabled;
      })
      : withMetric.slice();
    visible.sort(function (a, b) {
      return spec.order === "asc"
        ? a[spec.metric] - b[spec.metric]
        : b[spec.metric] - a[spec.metric];
    });
    const rendered = showAll ? visible : visible.slice(0, TOP_N);
    return {
      rendered: rendered,
      withMetric: withMetric,
      skipped: skipped,
      total: visible.length,
    };
  }

  function renderChartMeta(spec, info, baseline) {
    const metaEl = document.getElementById("models-meta-" + spec.id);
    if (!metaEl) return;
    // Always name BOTH numbers: what is drawn and what the source population
    // holds. The Cost per Task chart is intentionally sparse (the upstream
    // field is null for ~78% of models), so the gap has to be legible here
    // rather than looking like a broken page.
    const shown = info.rendered.length;
    const pool = info.withMetric.length;
    const scope = exactOnly
      ? "exactly matched models"
      : enabledOnly
        ? "enabled models"
        : "models";
    const parts = [
      "showing " +
        (showAll ? "all " : "") +
        shown +
        " of " +
        pool +
        " " +
        scope +
        " with this metric",
      info.skipped + " skipped (" + spec.missing + ")",
    ];
    const enabled = info.rendered.filter(function (m) { return m.enabled; }).length;
    if (enabled) parts.push(enabled + " sharedagents enabled");
    const soft = info.rendered.filter(isSoft).length;
    if (soft) parts.push(soft + " approximate (≈)");
    // A truncated axis is stated in words as well as drawn as a break glyph.
    if (baseline) parts.push("axis starts at " + baseline);
    metaEl.textContent = parts.join(" · ");
  }

  function renderChartLegend(spec, rows) {
    const legendEl = document.getElementById("models-legend-" + spec.id);
    if (!legendEl) return;
    const counts = {};
    rows.forEach(function (model) {
      const provider = model.provider || "Unknown";
      counts[provider] = (counts[provider] || 0) + 1;
    });
    const providers = Object.keys(counts).sort(function (a, b) {
      return counts[b] - counts[a] || a.localeCompare(b);
    });
    const shown = providers.slice(0, 8);
    let html = shown.map(function (provider) {
      return (
        "<span><i style=\"background:" +
        providerColor(provider) +
        "\"></i>" +
        esc(provider) +
        (counts[provider] > 1 ? " ×" + counts[provider] : "") +
        "</span>"
      );
    }).join("");
    if (providers.length > shown.length) {
      html += "<span>+ " + (providers.length - shown.length) + " more providers</span>";
    }
    legendEl.innerHTML = html;
  }

  function renderChartTable(spec, rows) {
    const tableEl = document.getElementById("models-table-" + spec.id);
    if (!tableEl) return;
    // Token prices are listed here (and in the tooltip) as clearly-labelled
    // USD-per-1M-tokens columns — they are a different unit from the charted
    // USD-per-task cost and are never drawn on the same axis.
    const head =
      '<thead><tr><th scope="col">Model</th><th scope="col">Provider</th>' +
      '<th scope="col">' + esc(spec.rowLabel) + '</th>' +
      '<th scope="col">Input USD/1M tokens</th><th scope="col">Output USD/1M tokens</th>' +
      '<th scope="col">sharedagents</th></tr></thead>';
    const body = rows.map(function (model) {
      const alias = (model.sharedagents || {}).alias;
      // The accessible table has to carry the same distinction as the bars: a
      // soft claim names the cascade step, an exact one does not.
      const sharedagentsCell = model.enabled
        ? (isSoft(model) ? "≈ " + esc(alias || "enabled") + " (matched by " + esc(stepLabel(model)) + ")"
          : esc(alias || "enabled"))
        : "—";
      return (
        '<tr><th scope="row">' + esc(model.name || model.slug || "model") +
        (model.variant ? " (" + esc(model.variant) + ")" : "") + "</th>" +
        "<td>" + esc(model.provider || "—") + "</td>" +
        "<td>" + esc(spec.format(model[spec.metric])) + "</td>" +
        "<td>" + esc(formatPerMillion(model.usd_per_1m_in)) + "</td>" +
        "<td>" + esc(formatPerMillion(model.usd_per_1m_out)) + "</td>" +
        "<td>" + sharedagentsCell + "</td></tr>"
      );
    }).join("");
    tableEl.innerHTML = head + "<tbody>" + body + "</tbody>";
  }

  function renderChart(spec) {
    const chartEl = document.getElementById("models-chart-" + spec.id);
    if (!chartEl) return;
    const info = chartModels(spec);
    renderChartLegend(spec, info.rendered);
    renderChartTable(spec, info.rendered);
    if (!info.rendered.length) {
      renderChartMeta(spec, info, "");
      chartEl.style.minHeight = "";
      chartEl.innerHTML =
        '<p class="models-chart-empty">' +
        (enabledOnly
          ? "No enabled sharedagents model reports this metric"
          : "No model in this snapshot reports this metric") +
        "</p>";
      return;
    }

    const mobile = isMobile();
    // A rotated label's neighbours are `step · sin45°` apart, so the slot has a
    // hard floor below which -45° labels WOULD collide. Above it they never can.
    const slotFloor = Math.ceil((LABEL_FONT_SIZE + 3) / Math.SQRT1_2);
    const slot = Math.max(slotFloor, mobile ? 30 : 34);
    const containerWidth = Math.max(
      300,
      Math.round((chartEl.parentElement && chartEl.parentElement.clientWidth) || 760)
    );
    const count = info.rendered.length;

    // ── labels first: the bottom and left margins are sized to FIT them ──
    // The old build did the reverse (fixed margins, labels truncated to 32
    // characters) which clipped every long name at BOTH ends — the tail to an
    // ellipsis and the head to the viewBox edge.
    const LABEL_MAX_PX = mobile ? 108 : 150;
    const labelTexts = info.rendered.map(function (model) {
      return chartLabel(model, LABEL_MAX_PX);
    });
    const widest = labelTexts.reduce(function (max, text) {
      return Math.max(max, textWidth(text, LABEL_FONT_SIZE));
    }, 0);

    // Provisional step, to decide rotation before the margins are known.
    const roughStep = Math.max(slot, (containerWidth - PAD_L - PAD_R) / count);
    // Horizontal labels whenever they actually fit; rotation only when needed.
    const rotate = widest > roughStep - 8;
    const labelLead = 14;
    const padB = rotate
      ? Math.ceil(widest * Math.SQRT1_2) + labelLead + 14
      : labelLead + LABEL_FONT_SIZE + 12;
    // A -45° label runs DOWN-LEFT from its anchor, so the leftmost one needs
    // room outside the plot or it is sliced by the viewBox.
    const padL = rotate
      ? Math.max(PAD_L, Math.ceil(widest * Math.SQRT1_2) - slot / 2 + 10)
      : PAD_L;

    const w = Math.max(containerWidth, count * slot + padL + PAD_R);
    const plotH = mobile ? 292 : 274;
    const h = PAD_T + plotH + padB;
    const innerW = w - padL - PAD_R;
    const innerH = plotH;
    const step = innerW / count;
    // 2px of surface between neighbouring bars (see the dataviz mark spec).
    const barW = Math.max(4, Math.min(step - 2, slot * 0.72));
    const values = info.rendered.map(function (model) { return model[spec.metric]; });
    const dataMin = Math.min.apply(null, values);
    const dataMax = Math.max.apply(null, values);

    // A zero baseline flattens a narrow band that sits high above zero — the
    // Intelligence chart put 37–54 inside a 0–100 axis and every bar looked the
    // same height. A non-zero baseline is used ONLY for that case, and it is
    // marked with an axis break so the truncation is never implied away.
    const narrow = dataMin > 0 && dataMax > 0 && dataMin / dataMax > 0.45;
    const scale = narrow
      ? niceScale(dataMin - (dataMax - dataMin) * 0.12, dataMax, 4)
      : niceScale(0, dataMax, 4);
    const yLo = narrow ? scale.lo : 0;
    const yHi = scale.hi;
    const span = yHi - yLo || 1;
    // One precision for the whole tick set, so an axis never mixes "55.0" with
    // "55" or "$0.500" with "$1.00".
    const axisDecimals = spec.usd
      ? usdDecimalsFor(scale.ticks)
      : tickDecimals(scale.step);
    const axisLabel = function (value) {
      return (spec.axis || spec.format)(value, axisDecimals);
    };
    renderChartMeta(spec, info, narrow ? axisLabel(yLo) : "");

    const baseY = PAD_T + innerH;
    let grid = "";
    scale.ticks.forEach(function (value) {
      if (value < yLo - 1e-9) return;
      const y = baseY - ((value - yLo) / span) * innerH;
      grid +=
        '<line x1="' + padL + '" y1="' + y.toFixed(1) + '" x2="' + (w - PAD_R) +
        '" y2="' + y.toFixed(1) + '" stroke="#2c2c30" stroke-width="1" stroke-dasharray="3 4"/>';
      grid +=
        '<text x="' + (padL - 13) + '" y="' + (y + 4).toFixed(1) +
        '" fill="' + INK + '" font-size="11" text-anchor="end">' +
        esc(axisLabel(value)) + "</text>";
    });
    if (narrow) {
      // Axis break: two slashes ON the axis, where it would have met zero — and
      // clear of the baseline tick label to its left.
      grid +=
        '<path d="M' + (padL - 5) + " " + (baseY - 1).toFixed(1) + " l10 -5 M" +
        (padL - 5) + " " + (baseY + 5).toFixed(1) +
        ' l10 -5" stroke="' + INK_MUTED + '" stroke-width="1.4" fill="none" stroke-linecap="round"/>';
    }

    let bars = "";
    info.rendered.forEach(function (model, index) {
      const value = model[spec.metric];
      const height = Math.max(1, ((value - yLo) / span) * innerH);
      const x = padL + index * step + (step - barW) / 2;
      const y = baseY - height;
      const enabled = !!model.enabled;
      const soft = isSoft(model);
      // One visual weight for every provider: the same translucent fill and the
      // same hairline. "Enabled" is a ring, never a neon slab.
      bars +=
        '<rect class="models-bar' + (enabled ? " is-enabled" : "") +
        (soft ? " is-soft" : "") +
        '" x="' + x.toFixed(1) + '" y="' + y.toFixed(1) +
        '" width="' + barW.toFixed(1) + '" height="' + height.toFixed(1) +
        '" rx="3" fill="' + providerColor(model.provider) + '" fill-opacity="0.82"' +
        (enabled
          ? ' stroke="' + ENABLED_COLOR + '" stroke-width="1.4" stroke-opacity="0.9"' +
            (soft ? ' stroke-dasharray="3 2"' : "")
          : ' stroke="' + SURFACE + '" stroke-width="1"') +
        "><title>" + esc(tooltip(model, spec)) + "</title></rect>";
      const labelX = x + barW / 2;
      const labelY = baseY + labelLead;
      bars +=
        '<text class="models-bar-label" x="' + labelX.toFixed(1) + '" y="' + labelY + '"' +
        (rotate
          ? ' transform="rotate(-45 ' + labelX.toFixed(1) + " " + labelY + ')" text-anchor="end"'
          : ' text-anchor="middle"') +
        ' fill="' + (enabled ? INK_STRONG : INK_MUTED) + '" font-size="' + LABEL_FONT_SIZE + '">' +
        esc(labelTexts[index]) +
        // The untruncated name always stays one hover away.
        "<title>" + esc(tooltip(model, spec)) + "</title></text>";
      if (enabled) {
        bars +=
          '<circle cx="' + labelX.toFixed(1) + '" cy="' + (y - 5).toFixed(1) +
          '" r="2.4" fill="' + ENABLED_COLOR + '"/>';
      }
    });

    chartEl.innerHTML =
      '<svg viewBox="0 0 ' + w + " " + h + '" width="' + w + '" height="' + h +
      '" preserveAspectRatio="xMinYMid meet" aria-hidden="true">' +
      grid + bars + "</svg>";
    chartEl.style.minHeight = h + "px";
  }

  // ── pareto scatter ─────────────────────────────────────────────────────
  //
  // Intelligence (y) against cost per Intelligence-Index task (x, log by
  // default — the reference chart uses a log cost axis). The backend prepares
  // BOTH the latest-per-family pool and the full pool (`family_latest`,
  // `on_frontier`/`on_frontier_all`, `pareto`/`pareto_all`,
  // `default_selection`/`default_selection_all`) precisely so this toggle is a
  // pick, never a re-derivation in the browser.

  const P_PAD_L = 64;
  const P_PAD_R = 28;
  const P_PAD_T = 20;
  const P_PAD_B = 58;

  /** Marker radius — local marks are largest, then frontier, then the rest. */
  function markerRadius(model) {
    return model.local ? 7 : onFrontier(model) ? 6 : 5;
  }

  /** How far a mark reaches, ring included — the box a label may never enter. */
  function markerReach(model) {
    return markerRadius(model) + (model.enabled ? 5 : 3);
  }

  function hasParetoMetrics(model) {
    return isNumber(model.intelligence) && isNumber(model.cost_per_task_usd);
  }

  function onFrontier(model) {
    return filterLatest ? !!model.on_frontier : !!model.on_frontier_all;
  }

  /** Every row eligible for the Pareto chart under the current filters. */
  function paretoPool() {
    const models = (payload && payload.models) || [];
    return models.filter(function (model) {
      if (!hasParetoMetrics(model)) return false;
      // "Latest per family" never drops a model this machine actually serves.
      if (filterLatest && !model.family_latest && !model.local) return false;
      if (filterLocal && !model.local) return false;
      if (filterFrontier && !onFrontier(model)) return false;
      if (enabledOnly && !model.enabled) return false;
      if (exactOnly && isSoft(model)) return false;
      return true;
    });
  }

  function defaultSelectionKeys() {
    const rows =
      (payload && (filterLatest ? payload.default_selection : payload.default_selection_all)) || [];
    const keys = {};
    rows.forEach(function (row) {
      if (row && row.key) keys[row.key] = true;
    });
    return keys;
  }

  /** The rows actually plotted: the explicit selection, else the default. */
  function paretoRows() {
    const pool = paretoPool();
    const chosen = selection || defaultSelectionKeys();
    const picked = pool.filter(function (model) {
      return !!chosen[model.key];
    });
    // A filter can exclude every model the backend's default names (e.g.
    // "local only" against a cloud-heavy default). With no explicit pick from
    // the operator, show the strongest of what survives rather than a blank
    // chart; an explicit empty selection is honoured as-is.
    if (!picked.length && !selection && pool.length) {
      return pool.slice().sort(function (a, b) {
        return b.intelligence - a.intelligence;
      }).slice(0, (payload && payload.default_selection_cap) || 20);
    }
    return picked;
  }

  function costFloor() {
    const q = (payload && payload.quadrant) || {};
    return isNumber(q.x_floor) && q.x_floor > 0 ? q.x_floor : 0.0001;
  }

  // log10 with a floor: AA reports a genuine 0.0 cost per task for some rows,
  // and log10(0) is -Infinity. The floor is the backend's own x_floor.
  function xValue(cost) {
    const floor = costFloor();
    const safe = Math.max(isNumber(cost) ? cost : floor, floor);
    return settingLog ? Math.log10(safe) : safe;
  }

  function axisTicks(min, max) {
    const ticks = [];
    if (settingLog) {
      const lo = Math.floor(min);
      const hi = Math.ceil(max);
      // Decade ticks alone can leave a narrow span with a single label (a
      // 10-model selection often lives inside one decade), so subdivide each
      // decade 1–2–5 until the span is wide enough for decades to stand alone.
      const decades = hi - lo;
      const mult = decades <= 1 ? [1, 2, 3, 5, 7] : decades <= 3 ? [1, 2, 5] : [1];
      for (let e = lo; e <= hi; e++) {
        for (let i = 0; i < mult.length; i++) {
          const tick = e + Math.log10(mult[i]);
          if (tick >= min && tick <= max) ticks.push(tick);
        }
      }
      return ticks;
    }
    // Linear: round values, not five evenly-sliced fractions of the domain.
    return niceScale(min, max, 5).ticks.filter(function (tick) {
      return tick >= min && tick <= max;
    });
  }

  function renderParetoMeta(pool, rows, hiddenLabels) {
    const metaEl = document.getElementById("models-meta-pareto");
    if (!metaEl) return;
    if (!payload) {
      metaEl.textContent = "Loading…";
      return;
    }
    const total = ((payload && payload.models) || []).length;
    const bits = [
      "showing " + rows.length + " of " + pool.length + " models with both metrics",
    ];
    const missing = total - pool.length;
    if (missing > 0) bits.push(missing + " skipped (no intelligence or cost-per-task)");
    const frontier = rows.filter(onFrontier).length;
    if (frontier) bits.push(frontier + " on the Pareto frontier");
    const enabled = rows.filter(function (m) { return m.enabled; }).length;
    if (enabled) bits.push(enabled + " sharedagents enabled");
    // A dropped label is never silent: say how many, and where the name went.
    if (hiddenLabels) {
      bits.push(hiddenLabels + " label" + (hiddenLabels === 1 ? "" : "s") + " hidden (hover the point)");
    }
    metaEl.textContent = bits.join(" · ");
  }

  function renderParetoLegend(rows) {
    const legendEl = document.getElementById("models-legend-pareto");
    if (!legendEl) return;
    const counts = {};
    rows.forEach(function (model) {
      const key = model.provider || "Unknown";
      counts[key] = (counts[key] || 0) + 1;
    });
    const providers = Object.keys(counts).sort(function (a, b) {
      return counts[b] - counts[a] || a.localeCompare(b);
    });
    // The chip is the MARK: a filled circle with the same surface ring the
    // points carry, not a flat square that shares nothing with the plot.
    legendEl.innerHTML = providers
      .map(function (provider) {
        return (
          '<span class="models-legend-item"><i class="models-legend-dot" style="background:' +
          esc(providerColor(provider)) +
          '"></i>' +
          esc(provider) +
          '<b class="models-legend-count">' +
          counts[provider] +
          "</b></span>"
        );
      })
      .join("");
  }

  function paretoTooltip(model) {
    const bits = [String(model.name || model.slug || "model")];
    if (model.provider) bits.push(model.provider);
    bits.push("Intelligence " + formatIndex(model.intelligence));
    bits.push("Cost/task " + formatUsd(model.cost_per_task_usd));
    if (onFrontier(model)) bits.push("on the Pareto frontier");
    if (model.local) bits.push("running locally");
    if (model.enabled) {
      const alias = (model.sharedagents || {}).alias;
      bits.push(alias ? "sharedagents enabled (" + alias + ")" : "sharedagents enabled");
      bits.push((isSoft(model) ? "≈ approximate match · " : "exact match · ") + stepLabel(model));
    }
    bits.push(
      "in " + formatPerMillion(model.usd_per_1m_in) + " · out " + formatPerMillion(model.usd_per_1m_out)
    );
    return bits.join(" · ");
  }

  function renderParetoTable(rows) {
    const tableEl = document.getElementById("models-table-pareto");
    if (!tableEl) return;
    if (!rows.length) {
      tableEl.innerHTML = "";
      return;
    }
    const sorted = rows.slice().sort(function (a, b) {
      return b.intelligence - a.intelligence;
    });
    const costDecimals = usdDecimalsFor(sorted.map(function (m) { return m.cost_per_task_usd; }));
    const body = sorted
      .map(function (model) {
        return (
          "<tr" + (model.enabled ? ' class="is-enabled"' : "") + ">" +
          "<th scope=\"row\">" + esc(chartLabel(model)) + "</th>" +
          "<td>" + esc(model.provider || "—") + "</td>" +
          "<td>" + esc(formatIndex(model.intelligence)) + "</td>" +
          "<td>" + esc(formatUsdFixed(model.cost_per_task_usd, costDecimals)) + "</td>" +
          "<td>" + (onFrontier(model) ? "frontier" : "—") + "</td>" +
          "<td>" + (model.enabled ? (isSoft(model) ? "≈ enabled" : "enabled") : "—") + "</td>" +
          "</tr>"
        );
      })
      .join("");
    tableEl.innerHTML =
      "<thead><tr><th scope=\"col\">Model</th><th scope=\"col\">Provider</th>" +
      "<th scope=\"col\">Intelligence</th><th scope=\"col\">Cost/task</th>" +
      "<th scope=\"col\">Frontier</th><th scope=\"col\">sharedagents</th></tr></thead>" +
      "<tbody>" + body + "</tbody>";
  }

  function renderWhy() {
    if (!whyBodyEl) return;
    const basis = (payload && payload.default_selection_basis) || "";
    if (!basis) {
      whyBodyEl.textContent = "";
      if (whyEl) whyEl.hidden = true;
      return;
    }
    if (whyEl) whyEl.hidden = false;
    const cap = payload && payload.default_selection_cap;
    whyBodyEl.textContent = selection
      ? "You picked these models by hand. The default is " + basis + "."
      : "Default selection (up to " + (cap || 20) + "): " + basis + ".";
  }

  function renderLocalDeployments() {
    if (!localDeploymentsEl || !localDeploymentsListEl) return;
    const rows = (payload && payload.local) || [];
    if (!rows.length) {
      localDeploymentsEl.hidden = true;
      localDeploymentsListEl.textContent = "";
      return;
    }
    localDeploymentsEl.hidden = false;
    localDeploymentsListEl.innerHTML = rows.map(function (row) {
      const name = row.display || row.id || row.alias || "?";
      const note = row.matched
        ? "Artificial Analysis match"
        : "Not benchmarked by Artificial Analysis";
      return '<li class="models-local-card"><span class="models-local-name">' +
        '<i aria-hidden="true">◇</i>' + esc(name) + '</span>' +
        '<span class="models-local-benchmark">' + esc(note) + '</span></li>';
    }).join("");
  }

  function renderPareto() {
    const chartEl = document.getElementById("models-chart-pareto");
    if (!chartEl) return;
    const pool = paretoPool();
    const rows = paretoRows();
    renderParetoLegend(rows);
    renderParetoTable(rows);
    renderWhy();

    if (!rows.length) {
      renderParetoMeta(pool, rows, 0);
      chartEl.style.minHeight = "";
      chartEl.innerHTML =
        '<p class="models-chart-empty">' +
        (pool.length
          ? "No model selected — open the model picker or choose a preset"
          : "No model in this snapshot reports both intelligence and cost per task") +
        "</p>";
      return;
    }

    const mobile = isMobile();
    const containerWidth = Math.max(
      320,
      Math.round((chartEl.parentElement && chartEl.parentElement.clientWidth) || 900)
    );
    const w = containerWidth;
    const h = mobile ? 460 : 540;
    const innerW = w - P_PAD_L - P_PAD_R;
    const innerH = h - P_PAD_T - P_PAD_B;

    // Scale to what is actually plotted. Scaling to the whole pool would leave
    // a 10-point default selection squeezed into a fraction of the width,
    // because the pool spans models from the x_floor up to the priciest.
    const scaleRows = rows;
    const xs = scaleRows.map(function (m) { return xValue(m.cost_per_task_usd); });
    const ys = scaleRows.map(function (m) { return m.intelligence; });
    let xMin = Math.min.apply(null, xs);
    let xMax = Math.max.apply(null, xs);
    let yMin = Math.min.apply(null, ys);
    let yMax = Math.max.apply(null, ys);
    if (!(xMax > xMin)) { xMin -= 0.5; xMax += 0.5; }
    if (!(yMax > yMin)) { yMin -= 1; yMax += 1; }
    // Round the y domain OUT to whole steps: the old fractional padding put
    // 39.7 / 43.5 / 47.2 on the axis where 40 / 45 / 50 belong.
    const yPad = (yMax - yMin) * 0.06 || 1;
    const yScale = niceScale(Math.max(0, yMin - yPad), yMax + yPad, 5);
    yMin = yScale.lo;
    yMax = yScale.hi;
    const yDecimals = tickDecimals(yScale.step);
    // The x padding is asymmetric on purpose: the priciest model sits at the
    // right edge and its label needs somewhere to go that is not another mark.
    const xSpan = xMax - xMin;
    xMin -= xSpan * 0.06;
    xMax += xSpan * 0.14;

    function px(model) {
      return P_PAD_L + ((xValue(model.cost_per_task_usd) - xMin) / (xMax - xMin)) * innerW;
    }
    function py(model) {
      return P_PAD_T + innerH - ((model.intelligence - yMin) / (yMax - yMin)) * innerH;
    }

    let grid = "";
    const yLabelBoxes = [];
    yScale.ticks.forEach(function (value) {
      const y = P_PAD_T + innerH - ((value - yMin) / (yMax - yMin)) * innerH;
      if (settingGrid) {
        grid +=
          '<line x1="' + P_PAD_L + '" y1="' + y.toFixed(1) + '" x2="' + (w - P_PAD_R) +
          '" y2="' + y.toFixed(1) + '" stroke="#2c2c30" stroke-width="1" stroke-dasharray="3 4"/>';
      }
      const text = value.toFixed(yDecimals);
      yLabelBoxes.push({
        x: P_PAD_L - 12 - textWidth(text, 11),
        x2: P_PAD_L - 6,
        y: y - 8,
        y2: y + 6,
      });
      grid +=
        '<text x="' + (P_PAD_L - 10) + '" y="' + (y + 4).toFixed(1) +
        '" fill="' + INK + '" font-size="11" text-anchor="end">' +
        esc(text) + "</text>";
    });
    // Every x tick shares ONE precision, decided from the tick set — the mixed
    // "$0.200 … $1.00" came from formatting each value on its own.
    const xTickValues = axisTicks(xMin, xMax).filter(function (tick) {
      return tick >= xMin && tick <= xMax;
    });
    const xTickAmounts = xTickValues.map(function (tick) {
      return settingLog ? Math.pow(10, tick) : tick;
    });
    // A span of more than two decades is all-decade ticks, where one shared
    // precision would read "$0.0001 … $10.0000"; anything narrower gets the one
    // shared precision that made "$0.200 / $1.00" consistent.
    const wideSpan = settingLog && xMax - xMin > 2.2;
    const xDecimals = usdDecimalsFor(xTickAmounts);
    const xTickLabels = xTickAmounts.map(function (amount) {
      return wideSpan ? formatUsdExact(amount) : formatUsdFixed(amount, xDecimals);
    });
    const xLabelBoxes = [];
    xTickValues.forEach(function (tick, index) {
      const x = P_PAD_L + ((tick - xMin) / (xMax - xMin)) * innerW;
      if (settingGrid) {
        grid +=
          '<line x1="' + x.toFixed(1) + '" y1="' + P_PAD_T + '" x2="' + x.toFixed(1) +
          '" y2="' + (P_PAD_T + innerH) +
          '" stroke="#2c2c30" stroke-width="1" stroke-dasharray="3 4"/>';
      }
      const text = xTickLabels[index];
      const half = textWidth(text, 11) / 2 + 3;
      xLabelBoxes.push({
        x: x - half,
        x2: x + half,
        y: P_PAD_T + innerH + 8,
        y2: P_PAD_T + innerH + 24,
      });
      grid +=
        '<text x="' + x.toFixed(1) + '" y="' + (P_PAD_T + innerH + 20) +
        '" fill="' + INK + '" font-size="11" text-anchor="middle">' +
        esc(text) + "</text>";
    });

    // Axis titles — the units are never implied (USD per Intelligence-Index
    // task, NOT per million tokens).
    grid +=
      '<text x="' + (P_PAD_L + innerW / 2).toFixed(1) + '" y="' + (h - 10) +
      '" fill="' + INK_MUTED + '" font-size="11" text-anchor="middle">Cost per Intelligence Index task (USD' +
      (settingLog ? ", log scale" : "") + ")</text>";
    grid +=
      '<text transform="translate(14 ' + (P_PAD_T + innerH / 2).toFixed(1) +
      ') rotate(-90)" fill="' + INK_MUTED + '" font-size="11" text-anchor="middle">Intelligence Index</text>';

    // Frontier path across the plotted frontier points (cheapest first).
    const frontierRows = rows.filter(onFrontier).sort(function (a, b) {
      return xValue(a.cost_per_task_usd) - xValue(b.cost_per_task_usd);
    });
    let frontierPath = "";
    if (frontierRows.length > 1) {
      frontierPath =
        '<polyline points="' +
        frontierRows
          .map(function (m) { return px(m).toFixed(1) + "," + py(m).toFixed(1); })
          .join(" ") +
        '" fill="none" stroke="' + ENABLED_COLOR +
        '" stroke-width="1.5" stroke-dasharray="5 4" opacity="0.55"/>';
    }

    // Greedy label placement. Two things the old pass got wrong and this one
    // does not: it measured text by a per-character guess, and it only tested
    // labels against OTHER LABELS — so a label was free to land on a marker,
    // which is exactly how "Claude Fable 5.1 …" ended up across the GPT-6 Astra
    // ring. Every mark is reserved up front, widths are measured, and a label
    // with nowhere to go is dropped (the mark keeps its <title> tooltip, and
    // the count is reported in the chart meta).
    const LABEL_GAP = 7;
    const LABEL_PAD = 3;
    const LABEL_HALF = 6;
    const placed = [];
    function overlaps(box) {
      for (let i = 0; i < placed.length; i++) {
        const p = placed[i];
        if (box.x < p.x2 && p.x < box.x2 && box.y < p.y2 && p.y < box.y2) return true;
      }
      return false;
    }
    rows.forEach(function (model) {
      const reach = markerReach(model);
      placed.push({
        x: px(model) - reach,
        x2: px(model) + reach,
        y: py(model) - reach,
        y2: py(model) + reach,
      });
    });
    // The axis tick labels are occupied space too — a point label that lands on
    // "$1.00" or on the "40" gutter is just as unreadable as one on a marker.
    yLabelBoxes.forEach(function (box) { placed.push(box); });
    xLabelBoxes.forEach(function (box) { placed.push(box); });

    function placeLabel(text, x, y, reach) {
      const width = textWidth(text, LABEL_FONT_SIZE);
      const dx = reach + LABEL_GAP;
      const dy = reach + LABEL_GAP;
      const candidates = [
        { anchor: "start", left: x + dx, oy: 4 },
        { anchor: "end", left: x - dx - width, oy: 4 },
        { anchor: "middle", left: x - width / 2, oy: -dy },
        { anchor: "middle", left: x - width / 2, oy: dy + 9 },
        { anchor: "start", left: x + dx, oy: -dy },
        { anchor: "end", left: x - dx - width, oy: -dy },
        { anchor: "start", left: x + dx, oy: dy + 9 },
        { anchor: "end", left: x - dx - width, oy: dy + 9 },
      ];
      for (let i = 0; i < candidates.length; i++) {
        const c = candidates[i];
        const box = {
          x: c.left - LABEL_PAD,
          x2: c.left + width + LABEL_PAD,
          y: y + c.oy - LABEL_HALF - 2,
          y2: y + c.oy + LABEL_HALF - 2,
        };
        // Stay inside the frame, and inside the plot band vertically.
        if (box.x < 2 || box.x2 > w - 2) continue;
        if (box.y < P_PAD_T - 8 || box.y2 > P_PAD_T + innerH + 8) continue;
        if (overlaps(box)) continue;
        placed.push(box);
        return {
          x: c.anchor === "end" ? x - dx : c.anchor === "middle" ? x : x + dx,
          y: y + c.oy,
          anchor: c.anchor,
        };
      }
      return null;
    }

    let points = "";
    let labels = "";
    // Label the strongest models first so a crowded cluster keeps the ones
    // that matter rather than whichever happened to come first in the array.
    const labelOrder = rows.slice().sort(function (a, b) {
      return b.intelligence - a.intelligence;
    });
    const labelFor = {};
    let hiddenLabels = 0;
    if (settingLabels) {
      labelOrder.forEach(function (model) {
        const spot = placeLabel(chartLabel(model), px(model), py(model), markerReach(model));
        if (spot) labelFor[model.key] = spot;
        else hiddenLabels += 1;
      });
    }

    rows.forEach(function (model) {
      const x = px(model);
      const y = py(model);
      const enabled = !!model.enabled;
      const soft = isSoft(model);
      const color = providerColor(model.provider);
      const r = markerRadius(model);
      const marker = model.local
        ? '<rect x="' + (x - r).toFixed(1) + '" y="' + (y - r).toFixed(1) +
          '" width="' + (r * 2) + '" height="' + (r * 2) +
          '" rx="2" transform="rotate(45 ' + x.toFixed(1) + " " + y.toFixed(1) + ')"'
        : '<circle cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="' + r + '"';
      // Every mark carries a 2px ring of the SURFACE, so two marks that touch
      // still read as two marks. "Enabled" is a separate outer ring rather than
      // a fat stroke on the mark itself, which used to fuse neighbours.
      points +=
        marker +
        ' class="models-point' + (enabled ? " is-enabled" : "") + (soft ? " is-soft" : "") +
        (model.local ? " is-local" : "") + '" fill="' + esc(color) + '"' +
        ' stroke="' + SURFACE + '" stroke-width="2"' +
        "><title>" + esc(paretoTooltip(model)) + "</title>" +
        (model.local ? "</rect>" : "</circle>");
      if (enabled) {
        points +=
          '<circle class="models-point-ring" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) +
          '" r="' + (r + 3).toFixed(1) + '" fill="none" stroke="' + ENABLED_COLOR +
          '" stroke-width="1.4" stroke-opacity="0.92"' +
          (soft ? ' stroke-dasharray="3 2"' : "") + ' pointer-events="none"/>';
      }
      const spot = labelFor[model.key];
      if (spot) {
        // paint-order puts the SURFACE-coloured stroke UNDER the glyphs: a halo,
        // so the dashed frontier line can no longer strike through the text.
        labels +=
          '<text class="models-point-label" x="' + spot.x.toFixed(1) + '" y="' + spot.y.toFixed(1) +
          '" text-anchor="' + spot.anchor + '"' +
          ' fill="' + (enabled ? INK_STRONG : INK_MUTED) + '"' +
          ' stroke="' + SURFACE + '" stroke-width="3.4" stroke-linejoin="round"' +
          ' paint-order="stroke fill" font-size="' + LABEL_FONT_SIZE + '">' +
          esc(chartLabel(model)) + "</text>";
      }
    });

    chartEl.innerHTML =
      '<svg viewBox="0 0 ' + w + " " + h + '" width="' + w + '" height="' + h +
      '" preserveAspectRatio="xMinYMid meet" aria-hidden="true">' +
      grid + frontierPath + points + labels + "</svg>";
    chartEl.style.minHeight = h + "px";
    // Reported last: the hidden-label count is only known once placement ran.
    renderParetoMeta(pool, rows, hiddenLabels);
  }

  // ── table view ─────────────────────────────────────────────────────────

  function renderTableView() {
    if (!tableViewEl) return;
    const rows = paretoRows().slice().sort(function (a, b) {
      return b.intelligence - a.intelligence;
    });
    if (metaTableEl) {
      metaTableEl.textContent = rows.length
        ? rows.length + " models · sorted by Intelligence Index"
        : "No model selected";
    }
    if (!rows.length) {
      tableViewEl.innerHTML = "";
      return;
    }
    // One precision per COLUMN, decided from that column's values, so the
    // decimal points line up under each other instead of reading
    // "$7.63 / $0.818 / $0.253".
    const costDecimals = usdDecimalsFor(rows.map(function (m) { return m.cost_per_task_usd; }));
    const inDecimals = usdDecimalsFor(rows.map(function (m) { return m.usd_per_1m_in; }));
    const outDecimals = usdDecimalsFor(rows.map(function (m) { return m.usd_per_1m_out; }));
    const body = rows
      .map(function (model) {
        // The unit lives in the column header; repeating "/1M" in every cell
        // only broke the alignment it was meant to clarify.
        return (
          "<tr" + (model.enabled ? ' class="is-enabled"' : "") + ">" +
          "<th scope=\"row\">" + esc(model.name || model.slug || "model") + "</th>" +
          "<td>" + esc(model.provider || "—") + "</td>" +
          '<td class="models-num">' + esc(formatIndex(model.intelligence)) + "</td>" +
          '<td class="models-num">' + esc(formatUsdFixed(model.cost_per_task_usd, costDecimals)) + "</td>" +
          '<td class="models-num">' +
          (isNumber(model.speed_tps) ? esc(String(Math.round(model.speed_tps))) : "—") + "</td>" +
          '<td class="models-num">' + esc(formatUsdFixed(model.usd_per_1m_in, inDecimals)) + "</td>" +
          '<td class="models-num">' + esc(formatUsdFixed(model.usd_per_1m_out, outDecimals)) + "</td>" +
          "<td>" + (onFrontier(model) ? "yes" : "—") + "</td>" +
          "<td>" + (model.local ? "yes" : "—") + "</td>" +
          "<td>" + (model.enabled ? (isSoft(model) ? "≈ enabled" : "enabled") : "—") + "</td>" +
          "</tr>"
        );
      })
      .join("");
    tableViewEl.innerHTML =
      "<thead><tr><th scope=\"col\">Model</th><th scope=\"col\">Provider</th>" +
      '<th scope="col" class="models-num">Intelligence</th>' +
      '<th scope="col" class="models-num">Cost/task</th>' +
      '<th scope="col" class="models-num">Speed t/s</th>' +
      '<th scope="col" class="models-num">In USD/1M</th>' +
      '<th scope="col" class="models-num">Out USD/1M</th>' +
      "<th scope=\"col\">Frontier</th><th scope=\"col\">Local</th>" +
      "<th scope=\"col\">sharedagents</th></tr></thead><tbody>" + body + "</tbody>";
  }

  // ── selection popover ──────────────────────────────────────────────────

  function selectedKeys() {
    return selection || defaultSelectionKeys();
  }

  function setSelection(keys, keepList) {
    selection = keys;
    // Ticking a box must not rebuild the list under the pointer — that would
    // drop focus and scroll position mid-click.
    renderSelectionUi(keepList);
    renderCurrentView();
  }

  function renderSelectionLabel(pool, chosenCount) {
    if (selectionLabelEl) {
      selectionLabelEl.textContent = chosenCount + " of " + pool.length + " models";
    }
    if (controlsMetaEl) {
      const bits = [];
      if (filterLatest) bits.push("latest per family");
      if (filterLocal) bits.push("local only");
      if (filterFrontier) bits.push("frontier only");
      if (enabledOnly) bits.push("enabled only");
      if (exactOnly) bits.push("exact matches only");
      controlsMetaEl.textContent = bits.length ? bits.join(" · ") : "all tracked models";
    }
  }

  function renderProviderOptions(pool) {
    if (!selectProviderEl) return;
    const providers = {};
    pool.forEach(function (model) {
      providers[model.provider || "Unknown"] = true;
    });
    const names = Object.keys(providers).sort();
    if (names.indexOf(selectProvider) === -1) selectProvider = "";
    selectProviderEl.innerHTML =
      '<option value="">All providers</option>' +
      names
        .map(function (name) {
          return '<option value="' + esc(name) + '"' +
            (name === selectProvider ? " selected" : "") + ">" + esc(name) + "</option>";
        })
        .join("");
  }

  function renderSelectList(pool, chosen) {
    if (!selectListEl) return;
    const needle = selectSearch.trim().toLowerCase();
    const visible = pool.filter(function (model) {
      if (selectProvider && (model.provider || "Unknown") !== selectProvider) return false;
      if (!needle) return true;
      return (
        String(model.name || "").toLowerCase().indexOf(needle) !== -1 ||
        String(model.slug || "").toLowerCase().indexOf(needle) !== -1 ||
        String(model.provider || "").toLowerCase().indexOf(needle) !== -1
      );
    });
    visible.sort(function (a, b) { return b.intelligence - a.intelligence; });
    // Per-value precision here, NOT one precision for the list: this is inline
    // prose, not a column, so nothing lines up to be helped by it — and a pool
    // holding one sub-cent model would push every row to four decimals.
    selectListEl.innerHTML = visible
      .map(function (model) {
        // The name is the bare name plus a tag chip, on ONE line that ellipses.
        // The full AA name wrapped, and stranded "Fallback)" on a line of its
        // own; it now lives in the row's title tooltip instead.
        const raw = String(model.name || model.slug || "model");
        const tag = variantTag(model.variant);
        return (
          '<label class="models-select-item" title="' + esc(raw) + '">' +
          '<input type="checkbox" data-key="' + esc(model.key) + '"' +
          (chosen[model.key] ? " checked" : "") + ">" +
          '<i class="models-provider-dot" style="background:' +
          esc(providerColor(model.provider)) + '"></i>' +
          '<span class="models-select-text">' +
          '<span class="models-select-name">' + esc(bareName(raw) || raw) +
          (tag ? '<b class="models-tag">' + esc(tag) + "</b>" : "") +
          "</span>" +
          '<small>' + esc(model.provider || "—") + " · " + esc(formatIndex(model.intelligence)) +
          " · " + esc(formatUsd(model.cost_per_task_usd)) +
          (onFrontier(model) ? " · frontier" : "") + (model.local ? " · local" : "") +
          "</small></span></label>"
        );
      })
      .join("");
  }

  function renderSelectionUi(keepList) {
    const pool = paretoPool();
    const chosen = selectedKeys();
    const chosenCount = pool.filter(function (m) { return !!chosen[m.key]; }).length;
    renderSelectionLabel(pool, chosenCount);
    if (!keepList) {
      renderProviderOptions(pool);
      renderSelectList(pool, chosen);
    }
    if (selectCountEl) {
      selectCountEl.textContent =
        chosenCount + " selected" + (selection ? "" : " (default)");
    }
  }

  function keysOf(rows) {
    const keys = {};
    (rows || []).forEach(function (row) {
      if (row && row.key) keys[row.key] = true;
    });
    return keys;
  }

  function applyPreset(name) {
    const pool = paretoPool();
    if (name === "default") return setSelection(null);
    if (name === "all") return setSelection(keysOf(pool));
    if (name === "pareto") {
      return setSelection(keysOf(pool.filter(onFrontier)));
    }
    if (name === "local") {
      return setSelection(keysOf(pool.filter(function (m) { return m.local; })));
    }
    if (name === "top10") {
      const top = pool.slice().sort(function (a, b) {
        return b.intelligence - a.intelligence;
      }).slice(0, 20);
      return setSelection(keysOf(top));
    }
  }

  // ── popovers ───────────────────────────────────────────────────────────

  const POPOVERS = [
    { el: popoverSelectionEl, toggle: selectionToggleEl },
    { el: popoverFilterEl, toggle: filterToggleEl },
    { el: popoverSettingsEl, toggle: settingsToggleEl },
  ];

  function closePopovers(except) {
    POPOVERS.forEach(function (entry) {
      if (!entry.el || entry.el === except) return;
      entry.el.hidden = true;
      if (entry.toggle) entry.toggle.setAttribute("aria-expanded", "false");
    });
  }

  function togglePopover(entry) {
    if (!entry.el) return;
    const open = entry.el.hidden;
    closePopovers(open ? entry.el : null);
    entry.el.hidden = !open;
    if (entry.toggle) entry.toggle.setAttribute("aria-expanded", open ? "true" : "false");
    if (open && entry.el === popoverSelectionEl && selectSearchEl) selectSearchEl.focus();
  }

  // ── views ──────────────────────────────────────────────────────────────

  function setView(next) {
    view = next;
    Object.keys(viewEls).forEach(function (name) {
      if (viewEls[name]) viewEls[name].hidden = name !== next;
      if (viewBtnEls[name]) {
        viewBtnEls[name].setAttribute("aria-pressed", name === next ? "true" : "false");
        viewBtnEls[name].classList.toggle("is-active", name === next);
      }
    });
    renderCurrentView();
  }

  function renderCurrentView() {
    if (view === "bars") CHARTS.forEach(renderChart);
    else if (view === "table") renderTableView();
    else renderPareto();
  }

  function renderAllCharts() {
    // Keep the hidden views coherent too: the bar charts read the same payload
    // and a stale view must never flash when the operator switches.
    renderLocalDeployments();
    renderSelectionUi();
    renderCurrentView();
  }

  // ── summary + load ─────────────────────────────────────────────────────

  function renderKpis(data) {
    const models = data.models || [];
    if (kpiEls.models) kpiEls.models.textContent = String(data.model_count || models.length || 0);
    if (kpiEls.enabled) kpiEls.enabled.textContent = String(data.enabled_count || 0);
    if (kpiEls.index) kpiEls.index.textContent = data.intelligence_index_version || "—";
    if (kpiEls.tier) kpiEls.tier.textContent = data.tier || "—";
    if (kpiEls.budget) {
      const limit = data.rate_limit || {};
      kpiEls.budget.textContent =
        isNumber(limit.remaining) && isNumber(limit.limit)
          ? limit.remaining + "/" + limit.limit
          : "—";
    }
    if (kpiEls.source) {
      if (data.source === "snapshot") {
        // Bundled snapshot (fresh install, no AA key): show provenance — the
        // release date the data was captured — instead of a live/cache stamp.
        const bundled = data.snapshot_date ? " · bundled " + data.snapshot_date : "";
        kpiEls.source.textContent = "Snapshot" + bundled;
      } else {
        const stamp = data.updated_at
          ? new Date(data.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
          : "";
        const sourceLabel =
          data.source === "live"
            ? "Live"
            : data.source === "fixture"
              ? "Fixture"
              : data.source === "error"
                ? "Unavailable"
                : "Cache";
        kpiEls.source.textContent = sourceLabel + (stamp ? " · " + stamp : "");
      }
    }
    if (legendNoteEl) {
      const parts = [
        showAll
          ? "Scrolling strips show every model that reports the metric"
          : "Charts default to the top " + TOP_N + " per metric — tick “Show all” for the rest",
      ];
      const soft = (data.models || []).filter(function (model) {
        return model.enabled && !model.match_exact;
      }).length;
      if (soft) {
        parts.push(
          soft + " enabled rows are approximate (≈) matches — tick “Exact matches only” to hide them"
        );
      }
      legendNoteEl.textContent = parts.join(" · ");
    }
  }

  function render(data) {
    payload = data;
    renderKpis(data);
    const error = data.error;
    if (error) {
      showStatus(error.message || "Benchmark data is unavailable");
      showEmpty(error);
    } else {
      showStatus("");
      hideEmpty();
    }
    if (attributionEl && data.attribution) {
      attributionEl.textContent = data.attribution;
    }
    renderAllCharts();
  }

  function renderFailure(message, code) {
    payload = payload || { models: [] };
    showStatus(message);
    showEmpty({ code: code, message: message });
    renderAllCharts();
  }

  async function loadBenchmarks(refresh) {
    if (loading) return;
    loading = true;
    setRefreshing(true);
    const url = BASE + "/api/models/benchmarks" + (refresh ? "?refresh=1" : "");
    try {
      const response = await fetch(url, {
        method: refresh ? "POST" : "GET",
        credentials: "same-origin",
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      let data = {};
      try {
        data = await response.json();
      } catch (error) {
        data = {};
      }
      if (!response.ok && !data.error) {
        renderFailure(
          "Benchmarks request failed (HTTP " + response.status + ")",
          "http_error"
        );
        return;
      }
      render(data);
      stampUpdated();
    } catch (error) {
      renderFailure("Could not reach the benchmarks API", "network_error");
    } finally {
      loading = false;
      setRefreshing(false);
      if (appEl) appEl.classList.remove("is-loading");
    }
  }

  function stampUpdated() {
    if (!updatedEl) return;
    updatedEl.textContent =
      "Updated " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  // ── wiring ─────────────────────────────────────────────────────────────

  function showModelsSubtab(name) {
    const tab = MODELS_TABS.indexOf(name) >= 0 ? name : "benchmarks";
    subtabEls.forEach(function (button) {
      const active = button.getAttribute("data-models-tab") === tab;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
    MODELS_TABS.forEach(function (id) {
      const pane = document.getElementById("models-pane-" + id);
      if (!pane) return;
      const on = id === tab;
      pane.hidden = !on;
      pane.classList.toggle("is-active", on);
    });
    if (appEl) {
      appEl.classList.toggle("is-benchmarks", tab === "benchmarks");
      appEl.classList.toggle("is-router", tab === "router");
      appEl.classList.toggle("is-vllm", tab === "vllm");
    }
    const benchActions = document.getElementById("models-benchmark-actions");
    const localActions = document.getElementById("models-local-actions");
    if (benchActions) benchActions.hidden = tab !== "benchmarks";
    if (localActions) localActions.hidden = tab === "benchmarks";
    if (tab === "benchmarks") {
      if (location.hash && location.hash !== "#benchmarks") {
        history.replaceState(null, "", location.pathname + location.search);
      }
    } else if (location.hash !== "#" + tab) {
      history.replaceState(null, "", "#" + tab);
    }
    if (window.ModelsLocal && typeof window.ModelsLocal.show === "function") {
      window.ModelsLocal.show(tab);
    }
    if (tab === "benchmarks" && !payload) loadBenchmarks(false);
  }

  subtabEls.forEach(function (button) {
    button.addEventListener("click", function () {
      showModelsSubtab(button.getAttribute("data-models-tab") || "benchmarks");
    });
  });
  window.addEventListener("hashchange", function () {
    const hash = location.hash.replace("#", "");
    if (MODELS_TABS.indexOf(hash) >= 0) showModelsSubtab(hash);
  });
  if (refreshEl) {
    refreshEl.addEventListener("click", function () {
      loadBenchmarks(true);
    });
  }
  if (enabledOnlyEl) {
    enabledOnlyEl.addEventListener("change", function () {
      enabledOnly = !!enabledOnlyEl.checked;
      renderAllCharts();
    });
  }
  if (exactOnlyEl) {
    exactOnlyEl.addEventListener("change", function () {
      exactOnly = !!exactOnlyEl.checked;
      renderAllCharts();
    });
  }
  if (showAllEl) {
    showAllEl.addEventListener("change", function () {
      showAll = !!showAllEl.checked;
      if (payload) renderKpis(payload);
      renderAllCharts();
    });
  }

  // View switcher.
  Object.keys(viewBtnEls).forEach(function (name) {
    const btn = viewBtnEls[name];
    if (!btn) return;
    btn.addEventListener("click", function () {
      setView(btn.getAttribute("data-view") || name);
    });
  });

  // Popovers: toggles, close button, outside click, Escape.
  POPOVERS.forEach(function (entry) {
    if (!entry.toggle) return;
    entry.toggle.addEventListener("click", function (event) {
      event.stopPropagation();
      togglePopover(entry);
    });
    if (entry.el) {
      entry.el.addEventListener("click", function (event) { event.stopPropagation(); });
    }
  });
  if (popoverCloseEl) {
    popoverCloseEl.addEventListener("click", function () { closePopovers(null); });
  }
  document.addEventListener("click", function () { closePopovers(null); });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closePopovers(null);
  });

  // Selection popover controls.
  if (selectSearchEl) {
    selectSearchEl.addEventListener("input", function () {
      selectSearch = selectSearchEl.value || "";
      renderSelectionUi();
    });
  }
  if (selectProviderEl) {
    selectProviderEl.addEventListener("change", function () {
      selectProvider = selectProviderEl.value || "";
      renderSelectionUi();
    });
  }
  if (selectListEl) {
    selectListEl.addEventListener("change", function (event) {
      const input = event.target;
      if (!input || input.type !== "checkbox") return;
      const key = input.getAttribute("data-key");
      if (!key) return;
      const next = {};
      const current = selectedKeys();
      Object.keys(current).forEach(function (k) { if (current[k]) next[k] = true; });
      if (input.checked) next[key] = true;
      else delete next[key];
      setSelection(next, true);
    });
  }
  if (selectAllEl) {
    selectAllEl.addEventListener("click", function () { applyPreset("all"); });
  }
  if (selectClearEl) {
    selectClearEl.addEventListener("click", function () { setSelection({}); });
  }
  Object.keys(presetEls).forEach(function (name) {
    const btn = presetEls[name];
    if (!btn) return;
    btn.addEventListener("click", function () { applyPreset(name); });
  });

  // Filters. Changing the pool invalidates an explicit pick's framing, so the
  // selection falls back to the backend default for the newly active pool.
  function wireFilter(el, apply) {
    if (!el) return;
    el.addEventListener("change", function () {
      apply(!!el.checked);
      selection = null;
      renderSelectionUi();
      renderCurrentView();
    });
  }
  wireFilter(filterLatestEl, function (on) { filterLatest = on; });
  wireFilter(filterLocalEl, function (on) { filterLocal = on; });
  wireFilter(filterFrontierEl, function (on) { filterFrontier = on; });

  // Chart settings — pure redraws, the selection is untouched.
  function wireSetting(el, apply) {
    if (!el) return;
    el.addEventListener("change", function () {
      apply(!!el.checked);
      renderCurrentView();
    });
  }
  wireSetting(settingLabelsEl, function (on) { settingLabels = on; });
  wireSetting(settingGridEl, function (on) { settingGrid = on; });
  wireSetting(settingLogEl, function (on) { settingLog = on; });

  let resizeTimer = 0;
  window.addEventListener("resize", function () {
    if (!payload) return;
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(function () { renderAllCharts(); }, 150);
  });

  // Adopt the template's own defaults so markup and state never disagree.
  if (filterLatestEl) filterLatest = !!filterLatestEl.checked;
  if (filterLocalEl) filterLocal = !!filterLocalEl.checked;
  if (filterFrontierEl) filterFrontier = !!filterFrontierEl.checked;
  if (settingLabelsEl) settingLabels = !!settingLabelsEl.checked;
  if (settingGridEl) settingGrid = !!settingGridEl.checked;
  if (settingLogEl) settingLog = !!settingLogEl.checked;
  setView("pareto");

  const initialTab = (function () {
    const hash = location.hash.replace("#", "");
    return MODELS_TABS.indexOf(hash) >= 0 ? hash : "benchmarks";
  })();
  showModelsSubtab(initialTab);
})();
