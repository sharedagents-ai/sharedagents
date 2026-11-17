/* markdown.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
function parseAssistantContent(raw) {
  raw = raw || "";
  const start = raw.indexOf("<think>");
  // The gateway only ever emits the reasoning block at the START of a turn
  // (intermediates are never wrapped once answer content has begun). Require
  // <think> to be leading so a literal "<think>" written inside a normal
  // answer isn't misparsed as a reasoning block.
  let segments = null;
  let answer = raw;
  let open = false;
  if (start !== -1 && raw.slice(0, start).trim() === "") {
    const after = raw.slice(start + "<think>".length);
    const closeIdx = after.indexOf("</think>");
    let reasoningRaw;
    if (closeIdx === -1) {
      reasoningRaw = after;
      answer = raw.slice(0, start);
      open = true; // still streaming the reasoning
    } else {
      reasoningRaw = after.slice(0, closeIdx);
      answer = raw.slice(0, start) + after.slice(closeIdx + "</think>".length);
      open = false;
    }
    segments = parseReasoning(reasoningRaw);
  }
  // Folded 🔧 name(args) / ← OK|failed lines also leak into the answer when
  // <think> is missing or not leading (history rebuild, ACP in-band markers).
  // Lift them into tool segments so they never render as markdown prose.
  const lifted = _liftInlineToolsFromAnswer(answer);
  if (lifted.tools.length) {
    segments = (segments || []).concat(lifted.tools);
    answer = lifted.answer;
  }
  return {
    segments: segments && segments.length ? segments : null,
    answer: typeof answer === "string" ? answer.trim() : answer,
    open,
  };
}

function _liftInlineToolsFromAnswer(text) {
  const split =
    typeof splitInlineToolMarkup === "function"
      ? splitInlineToolMarkup(text)
      : { prose: text, tools: [] };
  return {
    answer: split.prose,
    tools: (split.tools || []).map((t) => ({
      type: "tool",
      name: t.name,
      args: t.args,
      status: t.status || "running",
    })),
  };
}

function _lastTool(segments) {
  for (let i = segments.length - 1; i >= 0; i--) {
    if (segments[i].type === "tool") return segments[i];
  }
  return null;
}

// Turn the reasoning text into an ordered list of segments:
//   { type:'thought', text }  |  { type:'tool', name, args, status }  |
//   { type:'plan', items:[] }
function parseReasoning(text) {
  const segments = [];
  let thought = [];
  let inPlan = false;
  let planItems = [];
  const flushThought = () => {
    const t = thought.join("\n").trim();
    if (t) segments.push({ type: "thought", text: t });
    thought = [];
  };
  const flushPlan = () => {
    if (inPlan) {
      segments.push({ type: "plan", items: planItems.slice() });
      planItems = [];
      inPlan = false;
    }
  };

  for (const line of String(text).split("\n")) {
    const t = line.trim();
    let m;
    if ((m = t.match(/^🔧\s*(.*)$/))) {
      flushThought();
      flushPlan();
      const full = m[1];
      const pm = full.match(/^(.*?)\((.*)\)\s*$/); // name(args)
      segments.push({
        type: "tool",
        name: pm ? pm[1] : full,
        args: pm ? pm[2] : "",
        status: "running",
      });
    } else if (t === "← OK" || t === "←OK") {
      const last = _lastTool(segments);
      if (last) last.status = "ok";
    } else if (t === "← failed" || t === "←failed") {
      const last = _lastTool(segments);
      if (last) last.status = "failed";
    } else if ((m = t.match(/^📊\s*(.*)$/))) {
      // Per-round stats row injected by rerenderLiveTurn, sitting right after
      // the tools of the round it closes (NOT stacked at the bubble bottom).
      flushThought();
      flushPlan();
      segments.push({ type: "metrics", text: m[1] });
    } else if (t.startsWith("📋")) {
      flushThought();
      flushPlan();
      inPlan = true;
      planItems = [];
    } else if (inPlan && t.startsWith("•")) {
      planItems.push(t.replace(/^•\s*/, ""));
    } else if (inPlan && t === "") {
      // blank line inside a plan — ignore
    } else {
      flushPlan();
      thought.push(line);
    }
  }
  flushThought();
  flushPlan();
  return segments;
}

// Tool-call args can be very long (e.g. a delegate task or a full command). In
// the COLLAPSED row we show only an ~80-char preview with an ellipsis; the full
// string stays in the DOM as a title (hover) so nothing is lost, and the live
// tool block's body still expands on click.
const TOOL_ARGS_PREVIEW = 80;
function toolArgsPreview(s) {
  s = s || "";
  return s.length > TOOL_ARGS_PREVIEW
    ? s.slice(0, TOOL_ARGS_PREVIEW - 1) + "…"
    : s;
}
// escapeHtml() (textContent→innerHTML) does NOT escape quotes, so it is unsafe
// for an attribute value; tool args routinely contain `"`. Escape quotes too.
function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}

function reasoningToHtml(segments, open) {
  const tools = segments.filter((s) => s.type === "tool");
  const done = tools.filter((t) => t.status !== "running").length;
  let summary = "Reasoning";
  if (tools.length)
    summary += ` · ${tools.length} tool${tools.length > 1 ? "s" : ""}`;

  const rows = segments
    .map((s) => {
      if (s.type === "thought") {
        return `<div class="reason-thought">${escapeHtml(s.text)}</div>`;
      }
      if (s.type === "tool") {
        const icon =
          s.status === "ok"
            ? "✓"
            : s.status === "failed"
              ? "✗"
              : '<span class="tc-spin"></span>';
        return (
          `<div class="toolcall toolcall-${escapeHtml(s.status)}">` +
          `<span class="tc-status">${icon}</span>` +
          `<span class="tc-name" title="${escapeAttr(s.name)}">${escapeHtml(s.name)}</span>` +
          (s.args
            ? `<span class="tc-args" title="${escapeAttr(s.args)}">${escapeHtml(toolArgsPreview(s.args))}</span>`
            : "") +
          `</div>`
        );
      }
      if (s.type === "plan") {
        return (
          `<div class="reason-plan"><div class="rp-label">plan</div><ul>` +
          s.items.map((i) => `<li>${escapeHtml(i)}</li>`).join("") +
          `</ul></div>`
        );
      }
      if (s.type === "metrics") {
        return `<div class="reason-metrics">${escapeHtml(s.text)}</div>`;
      }
      return "";
    })
    .join("");

  return (
    `<details class="reasoning"${open ? " open" : ""}>` +
    `<summary><span class="reasoning-caret">▶</span>${escapeHtml(summary)}` +
    (tools.length
      ? `<span class="reasoning-count">${done}/${tools.length}</span>`
      : "") +
    `</summary><div class="reasoning-body">${rows}</div></details>`
  );
}

// Full HTML for an assistant message: optional reasoning disclosure + the
// markdown-rendered answer.
function renderAssistantContent(raw) {
  const { segments, answer, open } = parseAssistantContent(raw);
  let html = "";
  if (segments && segments.length) html += reasoningToHtml(segments, open);
  html += renderAnswerWithNotices(answer);
  return html;
}

// Runtime-notice paragraphs (nudge / loop / distill / …) can survive in folded
// answer text — legacy sessions stored before structured replay, copy-markdown
// round-trips, segments-less turns. Lift each marker line out of the markdown
// and render the SAME inline card as the live block stream, keeping every card
// at its exact position between the surrounding prose (RuntimeWarningNotice
// parity with streaming.js ``_openNoticeBlock``). Code fences are preserved: a
// line that merely looks like a notice inside ``` code is never split out.
function renderAnswerWithNotices(answer) {
  const raw = String(answer == null ? "" : answer);
  if (!raw.trim()) return "";
  const lift =
    typeof parseRuntimeNoticeLine === "function"
      ? parseRuntimeNoticeLine
      : null;
  const cardHtml =
    typeof runtimeNoticeCardHtml === "function"
      ? runtimeNoticeCardHtml
      : null;
  if (!lift || !cardHtml) return renderMarkdown(raw);
  const chunks = [];
  let prose = [];
  let fence = ""; // opening fence marker (``` or ~~~) when inside a code block
  const flushProse = () => {
    if (!prose.length) return;
    chunks.push({ md: prose.join("\n") });
    prose = [];
  };
  const lines = raw.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fenceMark = /^\s*(`{3,}|~{3,})/.exec(line);
    if (fence) {
      prose.push(line);
      const close = /^\s*(`{3,}|~{3,})\s*$/.exec(line);
      if (close && close[1][0] === fence[0] && close[1].length >= fence.length) {
        fence = "";
      }
      continue;
    }
    if (fenceMark) {
      fence = fenceMark[1];
      prose.push(line);
      continue;
    }
    const notice = lift(line);
    if (notice) {
      flushProse();
      chunks.push({ notice });
      continue;
    }
    prose.push(line);
  }
  flushProse();
  return chunks
    .map((c) => (c.notice ? cardHtml(c.notice, null) : renderMarkdown(c.md)))
    .join("");
}

// ── Markdown ──

// Agent-output images render inline ONLY from same-origin or inline-data
// sources — never off-origin (no tracking pixels / data exfil via image URL).
// The server CSP (img-src 'self' data:) is the hard backstop; this is the
// sanitizer-layer gate so a bad <img src> is dropped before it ever hits the
// DOM. Allowed: data:image/<type>;base64,… and same-origin paths (relative or
// root-relative). Rejected: any explicit scheme (http/https/javascript/…) and
// protocol-relative (//host) URLs.
function _mdArtifactHrefOk(href) {
  const path = String(href == null ? "" : href).trim().split(/[?#]/, 1)[0];
  return /(^|\/)files\/[^/]+\/(?:[0-9a-f]{64}\/)?[^/]+$/i.test(path);
}

function _mdLocalFilesystemHref(href) {
  let value = String(href == null ? "" : href).trim();
  try { value = decodeURIComponent(value); } catch (_) { /* keep raw value */ }
  if (/^file:/i.test(value)) return true;
  if (/^\/?[a-z]:[\\/]/i.test(value)) return true;
  return /^\/(?:home|root|tmp|var\/tmp|private\/tmp|users|etc|proc|sys|dev|run|opt|srv)(?:\/|$)/i.test(value);
}

function _mdImgSrcOk(src) {
  const s = String(src == null ? "" : src).trim();
  if (/^data:image\/(png|jpe?g|gif|webp);base64,/i.test(s)) return true;
  if (/^[a-z][a-z0-9+.\-]*:/i.test(s)) return false; // has a scheme → off-origin
  if (s.startsWith("//")) return false; // protocol-relative → off-origin
  // Same-origin ARTIFACT paths only (…/files/<sid>/<name>). Narrowed from "any
  // relative path" so assistant markdown can't trigger credentialed GETs to
  // arbitrary same-origin endpoints (/api/…, sibling reverse-proxy paths).
  return _mdArtifactHrefOk(s);
}

// DOMPurify hook that enforces _mdImgSrcOk on <img> and hardens artifact
// links. Registered exactly once (addHook accumulates — re-adding each render
// would stack duplicates and re-run the check N times).
let _mdSanitizerHookInstalled = false;
function _installMdSanitizerHook() {
  if (_mdSanitizerHookInstalled || typeof DOMPurify === "undefined") return;
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "IMG") {
      const src = node.getAttribute("src") || "";
      if (!_mdImgSrcOk(src)) {
        // Drop the whole element — a src-less <img> shows a broken-image icon.
        node.remove();
        return;
      }
      node.setAttribute("loading", "lazy");
      node.setAttribute("class", "md-image");
    } else if (node.tagName === "A") {
      // Same-origin artifact links (PDFs etc.) open in a new tab → the
      // browser's native viewer. Off-origin links keep default behavior.
      const href = node.getAttribute("href") || "";
      // Only ARTIFACT links (…/files/<sid>/<name>) get the new-tab native
      // viewer + PDF styling — not every root-relative link.
      if (_mdArtifactHrefOk(href)) {
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noopener noreferrer");
        if (/\.pdf(\?|#|$)/i.test(href)) node.setAttribute("class", "md-file-link md-pdf-link");
      } else if (_mdLocalFilesystemHref(href)) {
        // A CLI may mention a host path in Markdown before its structured
        // artifact event is accepted. Never let that path navigate outside
        // the mounted Web subpath (where Dashboard would handle it as a route).
        node.removeAttribute("href");
        node.removeAttribute("target");
        node.setAttribute("aria-disabled", "true");
        node.setAttribute("class", "md-local-path");
      }
    } else if (node.hasAttribute && node.hasAttribute("src")) {
      // ONLY <img> may carry a src (gated above). Any other resource-capable
      // element that survives the profile (e.g. <input type=image>, <track>)
      // gets its src stripped so raw assistant HTML can't trigger a credentialed
      // same-origin GET outside /files/<sid>/<name>.
      node.removeAttribute("src");
    }
  });
  _mdSanitizerHookInstalled = true;
}

function renderMarkdown(text) {
  if (!text) return "";
  try {
    // Configure marked
    marked.setOptions({
      highlight: function (code, lang) {
        if (lang && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
      },
      breaks: true,
      gfm: true,
    });
    // marked.parse returns HTML — sanitize with DOMPurify.
    // Zero-external-resources policy: still strip every resource-loading tag
    // EXCEPT <img>, which is now allowed but constrained to same-origin /
    // inline-data sources by the afterSanitizeAttributes hook + CSP. This lets
    // agent OUTPUT images (charts/screenshots) and same-origin file links
    // (PDFs) render inline in the web UI itself. `srcset`/`style` stay
    // forbidden (only bare same-origin `src` on <img> is permitted).
    const rawHtml = marked.parse(text);
    if (typeof DOMPurify !== "undefined") {
      _installMdSanitizerHook();
      return DOMPurify.sanitize(rawHtml, {
        USE_PROFILES: { html: true },
        FORBID_TAGS: [
          "source",
          "video",
          "audio",
          "picture",
          "iframe",
          "object",
          "embed",
          "style",
          "input",
          "track",
        ],
        FORBID_ATTR: ["srcset", "style"],
        ADD_ATTR: ["target"],
      });
    }
    return escapeHtml(text).replace(/\n/g, "<br>");
  } catch (e) {
    return escapeHtml(text).replace(/\n/g, "<br>");
  }
}

// ── Helpers ──

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function scrollToBottom() {
  // During a bulk history load, suppress ALL scroll-to-bottom calls (not just
  // attach's) — render-path call sites would otherwise chase the growing content
  // every event (autoScroll is frozen true) and produce the "fast scroll". The
  // single settle pin in finishBulkLoad() runs AFTER bulkLoading is cleared.
  if (state.scroll.bulkLoading && !state.isStreaming) return;
  if (!state.scroll.autoScroll) return; // user is reading history — don't yank
  requestAnimationFrame(() => {
    // Instant, NOT smooth: the container has `scroll-behavior:smooth` (style.css),
    // which would animate the jump and read as "the page scrolls down".
    messagesContainer.scrollTo({
      top: messagesContainer.scrollHeight,
      behavior: "instant",
    });
  });
}

function composerHeightCap() {
  const vv = window.visualViewport;
  const vh = (vv && vv.height) || window.innerHeight || 800;
  return Math.min(96, Math.round(vh * 0.3));
}

function resizeComposer() {
  // Look the node up directly: polish.js calls this from an IIFE that runs
  // before app.js declares the global `messageInput`, and touching that
  // `const` earlier throws in the temporal dead zone.
  const input = document.getElementById("message-input");
  if (!input) return;
  input.style.height = "auto";
  input.style.height =
    Math.min(input.scrollHeight, composerHeightCap()) + "px";
}

function autoResizeInput() {
  messageInput.addEventListener("input", () => {
    resizeComposer();
    updateSendState();
  });
  if (window.visualViewport) {
    window.visualViewport.addEventListener("resize", resizeComposer);
  }
}

// ── Sidebar drawer (mobile) / collapse (desktop) ──

function toggleSidebar() {
  if (window.MenuState) {
    MenuState.toggleSidebar();
    return;
  }
  if (isMobile()) {
    if (document.body.classList.contains("sidebar-open")) closeSidebarMobile();
    else openSidebarMobile();
  } else {
    sidebar.classList.toggle("collapsed");
  }
}

function openSidebarMobile() {
  if (window.MenuState) {
    MenuState.setSidebarVisible(true);
    return;
  }
  document.body.classList.add("sidebar-open");
  sidebarBackdrop.hidden = false;
}

function closeSidebarMobile() {
  if (window.MenuState) {
    MenuState.setSidebarVisible(false);
    return;
  }
  document.body.classList.remove("sidebar-open");
  sidebarBackdrop.hidden = true;
}

// ── Event Listeners ──
