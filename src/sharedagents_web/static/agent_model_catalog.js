/* Shared Agent + Model catalogue view.

   Chat and the Models CRUD both call this renderer.  It owns the exact agent
   rows, Native/External lanes, provider rail, model cards, and their logos;
   hosts own only state transitions and persistence callbacks.
 */
(function (global) {
  "use strict";

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function basePath() {
    const meta = document.querySelector('meta[name="sharedagents-base"]');
    return (meta && meta.content) || "";
  }

  const AGENT_LOGOS = {
    jarvis: '<svg viewBox="0 0 24 24" width="24" height="24" focusable="false"><rect width="24" height="24" rx="6" fill="#0d1117"/><g fill="none" stroke="#5aff8a" stroke-width="1.6" stroke-linejoin="round"><path d="M12 4 4 8l8 4 8-4-8-4z"/><path d="M4 16l8 4 8-4"/><path d="M4 12l8 4 8-4"/></g></svg>',
    build: '<svg viewBox="0 0 24 24" width="24" height="24" focusable="false"><rect width="24" height="24" rx="6" fill="#3d2b12"/><path fill="#d29922" d="M8.2 14.8 4.9 18a1.4 1.4 0 0 0 2 2l3.2-3.3a6 6 0 0 0 7.4-7.6l-2.7 2.7-2.3-.6-.6-2.3 2.7-2.7a6 6 0 0 0-6.4 6.6z"/></svg>',
    plan: '<svg viewBox="0 0 24 24" width="24" height="24" focusable="false"><rect width="24" height="24" rx="6" fill="#122033"/><path fill="#58a6ff" d="M7 4.5h7.2L19 9.2V19.5H7zm6.2.8v4.2H18z"/><path fill="#0d1117" d="M9.2 12.2h5.6v1.3H9.2zm0 2.7h5.6v1.3H9.2z"/></svg>',
    review: '<svg viewBox="0 0 24 24" width="24" height="24" focusable="false"><rect width="24" height="24" rx="6" fill="#221a33"/><path fill="#a371f7" d="M12 5.4c-3.7 0-6.8 2.3-8 4.6 1.2 2.3 4.3 4.6 8 4.6s6.8-2.3 8-4.6c-1.2-2.3-4.3-4.6-8-4.6zm0 7.1a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5z"/><path fill="none" stroke="#a371f7" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" d="m8.6 18.2 2.2 2.2 4.6-4.9"/></svg>',
    explore: '<svg viewBox="0 0 24 24" width="24" height="24" focusable="false"><rect width="24" height="24" rx="6" fill="#0f2a28"/><circle cx="12" cy="12" r="6.4" fill="none" stroke="#3fb950" stroke-width="1.6"/><path fill="#3fb950" d="m9.2 14.8 2-5.6 5.6-2-2 5.6z"/></svg>',
    general: '<svg viewBox="0 0 24 24" width="24" height="24" focusable="false"><rect width="24" height="24" rx="6" fill="#21262d"/><path fill="#8b949e" d="M6 6h5v5H6zm7 0h5v5h-5zM6 13h5v5H6zm7 0h5v5h-5z"/></svg>',
    research: '<svg viewBox="0 0 24 24" width="24" height="24" focusable="false"><rect width="24" height="24" rx="6" fill="#111827"/><circle cx="10.4" cy="10.4" r="4.2" fill="none" stroke="#79c0ff" stroke-width="1.8"/><path stroke="#79c0ff" stroke-width="1.8" stroke-linecap="round" d="m13.6 13.6 5 5"/></svg>',
    researcher: '<svg viewBox="0 0 24 24" width="24" height="24" focusable="false"><rect width="24" height="24" rx="6" fill="#111827"/><circle cx="10.4" cy="10.4" r="4.2" fill="none" stroke="#79c0ff" stroke-width="1.8"/><path stroke="#79c0ff" stroke-width="1.8" stroke-linecap="round" d="m13.6 13.6 5 5"/></svg>',
  };

  const AGENT_LOGO_FILES = {
    claude: { file: "claude.svg", fit: "contain" },
    codex: { file: "codex.svg", fit: "contain" },
    cursor: { file: "cursor.svg", fit: "cover" },
    antigravity: { file: "antigravity.png", fit: "cover" },
    opencode: { file: "opencode.svg", fit: "contain" },
    kimi: { file: "kimi.png", fit: "cover" },
    hermes: { file: "hermes.png", fit: "cover" },
    devin: { file: "devin.png", fit: "cover" },
    pi: { file: "pi.svg", fit: "contain" },
    zcode: { file: "zcode.png", fit: "cover" },
    qwen: { file: "qwen.svg", fit: "contain" },
    // The MiMo agent card reuses the official "Xiaomi MIMO" tile the Usage page
    // and the model cards already use (provider-logos/README.md records its
    // source), so the two screens cannot drift. An entry may name another
    // static/ directory with ``dir`` — the same convention PROVIDER_LOGO_FILES
    // and usage.js SUBSCRIPTION_LOGOS use.
    mimo: { file: "xiaomi.png", dir: "provider-logos", fit: "cover" },
  };

  const PROVIDER_LOGO_FILES = {
    bedrock: { file: "bedrock.svg", dir: "provider-logos", fit: "contain" },
    vllm: { file: "vllm.svg", dir: "provider-logos", fit: "contain" },
    openai: { file: "openai.svg", dir: "provider-logos", fit: "contain" },
    "openai-compat": { file: "openai.svg", dir: "provider-logos", fit: "contain" },
    openrouter: { file: "openrouter.svg", dir: "provider-logos", fit: "contain" },
    deepseek: { file: "deepseek.svg", dir: "provider-logos", fit: "contain" },
    groq: { file: "groq.svg", dir: "provider-logos", fit: "contain" },
    xai: { file: "xai.svg", dir: "provider-logos", fit: "contain" },
    cerebras: { file: "cerebras.svg", dir: "provider-logos", fit: "contain" },
    xiaomi: { file: "xiaomi.png", dir: "provider-logos", fit: "cover" },
    mimo: { file: "xiaomi.png", dir: "provider-logos", fit: "cover" },
    codex: { file: "codex.svg", dir: "agent-logos", fit: "contain" },
    cursor: { file: "cursor.svg", dir: "agent-logos", fit: "cover" },
    claude: { file: "claude.svg", dir: "agent-logos", fit: "contain" },
    anthropic: { file: "claude.svg", dir: "agent-logos", fit: "contain" },
    opencode: { file: "opencode.svg", dir: "agent-logos", fit: "contain" },
    kimi: { file: "kimi.png", dir: "agent-logos", fit: "cover" },
    antigravity: { file: "antigravity.png", dir: "agent-logos", fit: "cover" },
    google: { file: "antigravity.png", dir: "agent-logos", fit: "cover" },
    hermes: { file: "hermes.png", dir: "agent-logos", fit: "cover" },
    devin: { file: "devin.png", dir: "agent-logos", fit: "cover" },
    zcode: { file: "zcode.png", dir: "agent-logos", fit: "cover" },
    zai: { file: "zcode.png", dir: "agent-logos", fit: "cover" },
    qwen: { file: "qwen.svg", dir: "agent-logos", fit: "contain" },
    "qwen-code": { file: "qwen.svg", dir: "agent-logos", fit: "contain" },
  };

  function agentLetterLogo(name) {
    const letter = esc((String(name || "?").trim().charAt(0) || "?").toUpperCase());
    return '<svg viewBox="0 0 24 24" width="24" height="24" focusable="false"><rect width="24" height="24" rx="6" fill="#30363d"/><text x="12" y="16" text-anchor="middle" fill="#e6edf3" font-size="11" font-weight="700">' + letter + "</text></svg>";
  }

  function agentLogo(name, kind, opts) {
    opts = opts || {};
    const raw = String(name || "").trim();
    const key = raw.toLowerCase();
    const cls = kind === "sm" ? "agent-logo agent-logo-sm" : "agent-logo";
    const official = AGENT_LOGO_FILES[key];
    let inner;
    if (official) {
      const src = basePath() + "/static/" + (official.dir || "agent-logos") + "/" + official.file;
      if (opts.eager) {
        inner = '<img src="' + esc(src) + '" alt="" loading="eager" decoding="async" class="agent-logo-img agent-logo-fit-' + official.fit + '">';
      } else {
        return '<span class="' + cls + '" data-agent-logo="' + esc(key || "unknown") +
          '" data-logo-src="' + esc(src) + '" data-logo-fit="' + esc(official.fit) +
          '" aria-hidden="true">' + agentLetterLogo(raw) + "</span>";
      }
    } else {
      inner = AGENT_LOGOS[key] || agentLetterLogo(raw);
    }
    return '<span class="' + cls + '" data-agent-logo="' + esc(key || "unknown") +
      '" aria-hidden="true">' + inner + "</span>";
  }

  function providerKey(name) {
    return String(name || "").trim().toLowerCase().replace(/\s+/g, "");
  }

  function providerLogo(name, opts) {
    opts = opts || {};
    const raw = String(name || "").trim();
    const key = providerKey(raw);
    const official = PROVIDER_LOGO_FILES[key];
    const letter = esc((raw.charAt(0) || "?").toUpperCase());
    if (official) {
      const src = basePath() + "/static/" + official.dir + "/" + official.file;
      if (opts.eager) {
        return '<span class="amp-provider-mark amp-provider-mark-img" data-provider-logo="' +
          esc(key) + '" aria-hidden="true"><img src="' + esc(src) +
          '" alt="" loading="eager" decoding="async" class="amp-provider-logo amp-provider-fit-' +
          official.fit + '"></span>';
      }
      return '<span class="amp-provider-mark amp-provider-mark-img" data-provider-logo="' +
        esc(key) + '" data-logo-src="' + esc(src) + '" data-logo-fit="' + esc(official.fit) +
        '" aria-hidden="true">' + letter + "</span>";
    }
    return '<span class="amp-provider-mark" data-provider-logo="' +
      esc(key || "unknown") + '" aria-hidden="true">' + letter + "</span>";
  }

  function modelLogo(row) {
    const id = String(row.wireId || row.realId || row.id || "").toLowerCase();
    // Use the model's maker, not its routing provider (e.g. OpenRouter).
    // Unknown makers retain a letter mark instead of borrowing another logo.
    let maker = id.includes("/") ? id.split("/")[0] : "";
    if (!maker) {
      if (/^(gpt-|o[1-9](?:-|$))/.test(id)) maker = "openai";
      else if (id.startsWith("claude")) maker = "anthropic";
      else if (id.startsWith("deepseek")) maker = "deepseek";
      else if (id.startsWith("grok")) maker = "xai";
      else if (/^(kimi|moonshot)/.test(id)) maker = "kimi";
      else if (/^(glm|zcode|zai)/.test(id)) maker = "zai";
      else if (/^mimo/.test(id)) maker = "xiaomi";
      else maker = String(row.label || id);
    }
    maker = ({"x-ai": "xai", moonshotai: "kimi", "z-ai": "zai", zhipu: "zai"})[maker] || maker;
    const known = ["openai", "anthropic", "deepseek", "xai", "kimi", "openrouter", "zai", "xiaomi"];
    if (known.includes(maker)) return providerLogo(maker, { eager: true });
    return '<span class="amp-provider-mark" aria-hidden="true">' +
      esc((maker.charAt(0) || "?").toUpperCase()) + '</span>';
  }

  function hydrate(root) {
    const scope = root && root.querySelectorAll ? root : document;
    scope.querySelectorAll("[data-logo-src]").forEach(function (el) {
      if (el.getAttribute("data-logo-hydrated") === "1") return;
      const src = el.getAttribute("data-logo-src");
      if (!src) return;
      const fit = el.getAttribute("data-logo-fit") || "contain";
      const provider = el.classList.contains("amp-provider-mark");
      const cls = provider
        ? "amp-provider-logo amp-provider-fit-" + fit
        : "agent-logo-img agent-logo-fit-" + fit;
      // The tile still holds its letter mark at this point. Keep it as the
      // fallback: a 404 or a decode failure must return the letter tile, never
      // a broken-image icon. An inline error attribute is not an option — CSP
      // ships ``script-src 'self' blob: 'wasm-unsafe-eval'`` with no
      // 'unsafe-inline', so the handler would simply be blocked. Resource errors
      // do not bubble but ARE delivered to capture-phase listeners on ancestors,
      // so one capture listener on the wrapper recovers the tile. ``hydrated``
      // stays "1": a missing asset is not retried on every re-render.
      const letter = el.innerHTML;
      el.innerHTML = '<img src="' + esc(src) + '" alt="" loading="lazy" decoding="async" class="' + esc(cls) + '">';
      el.setAttribute("data-logo-hydrated", "1");
      if (letter && typeof el.addEventListener === "function") {
        el.addEventListener("error", function () {
          el.innerHTML = letter;
        }, true);
      }
    });
  }

  function matchesQuery(model, query) {
    const q = String(query || "").trim().toLowerCase();
    if (!q) return true;
    return [model.id, model.alias, model.label, model.wireId, model.realId,
      model.subtitle, model.provider].some(function (part) {
      return String(part || "").toLowerCase().includes(q);
    });
  }

  function renderAgents(opts) {
    const host = opts.host;
    if (!host) return;
    const rows = opts.agents || [];
    const current = String(opts.current || "");
    // Three sections, in this order. Precedence is
    // by ORIGIN, not by engine: a user-created agent that runs an external
    // engine stays under "User created". The flags are computed server-side
    // (sessions._agent_origin_flags) from the runtime's own predicates, so the
    // client never re-invents an engine list.
    const userCreated = [], native = [], external = [];
    rows.forEach(function (row) {
      const origin = String(row.origin || "");
      const isUser = origin === "user_created";
      const isExternal = !isUser && !!row.external;
      if (isUser) userCreated.push(row);
      else if (isExternal) external.push(row);
      else native.push(row);
    });
    function lane(title, hint, items) {
      // An empty section is HIDDEN, never rendered as an empty header.
      if (!items.length) return "";
      return '<div class="agent-lane"><h4 class="agent-lane-title">' + esc(title) +
        ' <span class="agent-lane-count">' + items.length + '</span>' +
        (hint ? ' <span class="agent-lane-hint">' + esc(hint) + '</span>' : '') +
        '</h4><div class="agent-lane-grid">' + items.map(function (row) {
          const name = String(row.name || row.id || "");
          let badgeHtml = "";
          const pace = row.pace;
          if (pace && pace.pace != null) {
            const val = Math.round(Number(pace.pace));
            const zone = val > 100 ? "is-over" : val > 85 ? "is-warn" : "is-ok";
            badgeHtml += '<span class="amp-pace-pill ' + zone + '" title="Pace: ' + pace.pace + '%">' + val + '%</span>';
          }
          const credits = row.credits;
          if (credits != null && Number(credits) > 0) {
            badgeHtml += '<span class="amp-api-credits" title="Credits: $' + Number(credits).toFixed(2) + '">$' + Number(credits).toFixed(2) + '</span>';
          }
          // External-agent integrity chips (server-computed): CLI version and
          // a sign-in alert so the picker instructs auth BEFORE a doomed turn.
          let healthHtml = "";
          const version = String(row.version || "");
          if (version) {
            healthHtml += '<span class="agent-row-version" title="CLI version">v' + esc(version) + '</span>';
          }
          const authState = String(row.auth_state || "");
          if (authState === "missing" || row.installed === false) {
            const notInstalled = row.installed === false && authState !== "missing";
            const authHint = String(row.auth_hint || "");
            healthHtml += '<span class="agent-row-auth" role="alert" title="' +
              esc((notInstalled ? 'Install: ' : 'Authentication required — run: ') + (authHint || "see the agent's docs")) + '">' +
              (notInstalled ? '⚠ not installed' : '⚠ needs authentication') + '</span>';
          }
          return '<button type="button" class="agent-row' + (name === current ? ' is-current' : '') +
            '" data-agent="' + esc(name) + '"' + (opts.disabled ? ' disabled' : '') + '>' +
            (badgeHtml ? '<span class="agent-row-badges">' + badgeHtml + '</span>' : '') +
            agentLogo(name) + '<span class="agent-row-text"><span class="agent-row-name">' + esc(name) +
            healthHtml +
            '</span>' +
            (row.description ? '<span class="agent-row-desc">' + esc(row.description) + '</span>' : '') +
            (row.model ? '<span class="agent-row-model" title="Default model">' + esc(row.model) + '</span>' : '') +
            '</span></button>';
        }).join("") + '</div></div>';
    }
    host.innerHTML = rows.length
      ? (lane("Native", "", native)
         + lane("User created", "", userCreated)
         + lane("External", "ACP / CLI", external))
      : (opts.emptyHtml || '<p class="amp-empty">No agents loaded.</p>');
    hydrate(host);
  }

  // ``opts.providers`` — the server's provider tabs ({id, label, status,
  // error}); without it the rail lists the providers the rows name.
  // ``opts.noticeHtml`` — a message for the shown tab (its error), rendered
  // above its cards.
  function renderModels(opts) {
    const rows = opts.models || [];
    const tabs = Array.isArray(opts.providers) && opts.providers.length
      ? opts.providers
      : Array.from(new Set(rows.map(function (row) {
        return row.provider || "Other";
      }))).map(function (name) { return { id: name, label: name, status: "ready" }; });
    const providers = tabs.map(function (tab) { return String(tab.id); });
    let provider = String(opts.provider || "");
    if (!providers.includes(provider)) provider = providers[0] || "";
    if (opts.providersHost) {
      opts.providersHost.innerHTML = tabs.map(function (tab) {
        const name = String(tab.id);
        const active = name === provider;
        const failed = tab.status === "error";
        const tip = failed && tab.error && tab.error.message ? ' title="' + esc(tab.error.message) + '"' : '';
        return '<button type="button" role="tab" class="amp-provider-tab' + (active ? ' is-selected' : '') +
          (failed ? ' is-unavailable' : '') +
          '" aria-selected="' + (active ? 'true' : 'false') + '" tabindex="' + (active ? '0' : '-1') +
          '" data-provider="' + esc(name) + '"' + tip + (opts.disabled ? ' disabled' : '') + '>' +
          providerLogo(name) + '<span>' + esc(tab.label || name) + '</span></button>';
      }).join("");
      hydrate(opts.providersHost);
      // A saved provider can start beyond the phone's visible strip.
      // Scroll this rail only; leave the sheet and page at their positions.
      const selected = opts.providersHost.querySelector('[aria-selected="true"]');
      if (selected && opts.providersHost.clientWidth) {
        const rail = opts.providersHost.getBoundingClientRect();
        const tab = selected.getBoundingClientRect();
        if (tab.left < rail.left) opts.providersHost.scrollLeft += tab.left - rail.left;
        else if (tab.right > rail.right) opts.providersHost.scrollLeft += tab.right - rail.right;
      }
    }
    const visible = rows.filter(function (row) {
      return (row.provider || "Other") === provider;
    });
    const filtered = visible.filter(function (row) { return matchesQuery(row, opts.query); });
    if (opts.modelsHost) {
      opts.modelsHost.innerHTML = (opts.noticeHtml || "") + (filtered.length ? filtered.map(function (row) {
        const active = row.id === opts.selectedModel;
        // The selected tab already names the provider. Show the model once;
        // retain its exact wire id in the tooltip without repeating it below.
        let label = String(row.label || row.wireId || row.id || "");
        const suffix = " (" + provider + ")";
        if (label.toLowerCase().endsWith(suffix.toLowerCase()))
          label = label.slice(0, -suffix.length);
        const ctx = (typeof modelContextLabel === "function")
          ? modelContextLabel(row) : "";
        return '<button type="button" role="radio" class="amp-model-card' + (active ? ' is-selected' : '') +
          '" aria-checked="' + (active ? 'true' : 'false') + '" data-model="' + esc(row.id) + '"' +
          ' title="' + esc(row.wireId || label) + '"' +
          (opts.disabled ? ' disabled' : '') + '>' + modelLogo(row) +
          '<span class="amp-model-copy"><span class="amp-model-name">' + esc(label) + '</span>' +
          (ctx ? '<span class="amp-model-ctx">' + esc(ctx) + '</span>' : '') + '</span></button>';
      }).join("") : (opts.emptyHtml != null ? opts.emptyHtml
        : '<p class="amp-empty">No models advertised for this provider.</p>'));
    }
    return { provider: provider, providers: providers, visible: visible, filtered: filtered };
  }

  const view = {
    agentLogo: agentLogo,
    providerLogo: providerLogo,
    hydrate: hydrate,
    matchesQuery: matchesQuery,
    renderAgents: renderAgents,
    renderModels: renderModels,
  };
  global.AgentModelCatalogView = view;
  // Backward-compatible names used by the existing chat header and renderers.
  global.pickerAgentLogo = agentLogo;
  global.pickerProviderLogo = providerLogo;
  global.hydratePickerLogos = hydrate;
})(window);
