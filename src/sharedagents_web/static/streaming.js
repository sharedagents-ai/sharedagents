/* streaming.js — extracted from app.js (de-monolith Task 9, behavior-preserving).
   Classic script sharing the global / window.App scope; loaded BEFORE
   app.js per index.html. No references were rewritten. */
const DIRECT_DISPATCH_TITLES = new Set([
  "delegate_claude",
  "delegate_codex",
  "delegate_antigravity",
]);
const DELEGATE_BADGES = {
  delegate_claude: "↳ via claude",
  delegate_codex: "↳ via codex",
  delegate_antigravity: "↳ via antigravity",
};
const INTERNAL_TOOL_TITLES = new Set(["plan_classify", "plan_pass"]);
// Sibling/engine spawns that return later. OpenCode keeps `task` out of the
// consecutive read/glob/grep/list group and paints it as its own card so the
// operator can see a child is still coming back. Same idea here: these names
// never fold into an Activity group.
const DELEGATE_TOOL_NAMES = new Set([
  "delegate_agent",
  "agent",
  "agentsync",
  "agentasync",
  "delegate_claude",
  "delegate_codex",
  "delegate_antigravity",
]);

function _isDelegateToolName(name) {
  const n = String(name || "").toLowerCase();
  // Keep this exact allowlist in sync with the TUI's _is_delegate_tool.
  // Internal reviews and unknown delegate_* calls are ordinary tools.
  return DELEGATE_TOOL_NAMES.has(n);
}

// Control calls stay in Activity; both first-party dispatch variants (including
// continuations) have a child session and get their own card.
const DELEGATE_CONTROL_ACTIONS = new Set(["list", "abort", "steer", "cancel"]);
const DELEGATE_TERMINAL_STATUSES = new Set([
  "done",
  "completed",
  "failed",
  "cancelled",
  "aborted",
  "refused",
]);

function _isDelegateSpawn(name, args) {
  if (!_isDelegateToolName(name)) return false;
  const a = _toolArgsObj(args);
  const action = String(a.action || "dispatch").trim().toLowerCase();
  if (DELEGATE_CONTROL_ACTIONS.has(action)) return false;
  const n = String(name || "").toLowerCase();
  if (n !== "delegate_agent") return true;
  const runs = a.runs;
  if (Array.isArray(runs) ? runs.length : runs) return true;
  if (a.subagent_type || a.name) return true;
  if ((a.prompt || a.task) && !a.task_id) return true;
  return false;
}

function _delegateTargetFrom(name, args) {
  if (!_isDelegateToolName(name)) return "";
  const a = _toolArgsObj(args);
  if (a.name) return String(a.name);
  if (a.subagent_type) return String(a.subagent_type);
  if (a.agent) return String(a.agent);
  const n = String(name || "").toLowerCase();
  const target = {
    delegate_claude: "claude",
    delegate_codex: "codex",
    delegate_antigravity: "antigravity",
  }[n];
  if (target) return target;
  if (n === "agentsync" || n === "agentasync") return "subagent";
  return String(name || "agent");
}

function _delegateModeLabel(blk) {
  const toolName = String((blk && blk.toolName) || "")
    .toLowerCase()
    .replace(/[^a-z]/g, "");
  if (toolName === "agentasync" || (blk && blk.delegateAsync)) return "ASYNC";
  if (toolName === "agentsync") return "SYNC";
  return "";
}

function _delegateInit(blk, name, args) {
  if (!_isDelegateSpawn(name, args)) return;
  const a = _toolArgsObj(args);
  blk.delegate = true;
  blk.delegateAsync = /^agentasync$/i.test(name);
  blk.delegateTarget = _delegateTargetFrom(name, a);
  const runs = Array.isArray(a.runs) && a.runs.length ? a.runs : [a];
  const previous = blk.delegateChildren || [];
  if (previous.length && !Object.keys(a).length) return;
  blk.delegateChildren = runs.map((entry, i) => {
    const run = _toolArgsObj(entry);
    const callId = runs === a.runs ? `${blk.callId}.${i}` : blk.callId;
    return {
      callId,
      agent: _delegateTargetFrom(name, run),
      description: String(run.description || run.prompt || run.task || "").split("\n")[0],
      taskId: String(run.task_id || ""),
      status: "running",
      ...previous.find((child) => child.callId === callId),
      // Real arguments can follow a title-only tool start.
      ...(run.description || run.prompt || run.task ? {
        description: String(run.description || run.prompt || run.task).split("\n")[0],
        agent: _delegateTargetFrom(name, run),
      } : {}),
    };
  });
}

function _delegateStatus(blk) {
  const children = blk.delegateChildren || [];
  // A failed member does not hide other members that are still running.
  blk.status = children.some((c) => c.status === "running") ? "running"
    : children.some((c) => c.status === "failed") ? "failed"
    : children.length && children.every((c) => ["cancelled", "aborted", "refused"].includes(c.status))
      ? children.every((c) => c.status === "refused") ? "refused" : "cancelled" : "ok";
}

function _delegateResultContent(blk, content, ok) {
  const children = blk.delegateChildren || [];
  const chunks = String(content || "").split(/^=== run \d+ .* ===\s*$/m);
  const batched = chunks.length > 1;
  children.forEach((child, index) => {
    const body = batched ? chunks[index + 1] || "" : String(content || "");
    let handle = null;
    for (const line of body.split("\n")) {
      try {
        const value = JSON.parse(line);
        if (value && value.job_id && ["started", "already_started", "running"].includes(value.status)) {
          handle = value;
          break;
        }
      } catch (_e) {}
    }
    const task = /<task\s+id="([^"]+)"\s+state="([^"]+)"/.exec(body);
    if (task) child.taskId = task[1];
    const outcome = (blk.delegateMemberOutcomes || [])[index] || blk.delegateOutcome;
    if (outcome && outcome !== "running" && !child.terminal) {
      child.status = outcome === "done" ? "ok" : outcome;
      child.terminal = true;
    } else if (handle) {
      child.jobId = String(handle.job_id);
      child.taskId = String(handle.task_id || child.taskId || "");
      child.detached = true;
      // A terminal lifecycle event can race the immediate handle.
      if (!child.terminal) child.status = "running";
    } else if (!child.terminal) {
      child.status = task ? (task[2] === "completed" ? "ok" : "failed")
        : ok === false ? "failed" : "ok";
      child.terminal = true;
    }
  });
  _delegateStatus(blk);
}

// JS mirror of sharedagents_core.delivery.should_show (§7 truth table), expressed as
// the per-channel render flags the block model consumes. Kept byte-for-byte in
// lockstep with delivery.py so the off·on·stream semantics never drift:
//   reasoning off → no 💭 · on → ONE consolidated 💭 (not streamed) · stream → live 💭
//   tools     off → no 🔧 · on → 🔧 header only · stream → 🔧 header + body
function blockFlagsFor(vis) {
  const r = (vis && vis.reasoning) || "off";
  const tl = (vis && vis.tools) || "off";
  return {
    reasoningShown: r !== "off",
    reasoningConsolidated: r === "on",
    toolShowHeader: tl !== "off",
    toolShowBody: tl === "stream",
  };
}

// Cursor Read uses target_file; OpenCode uses filePath. Collapse to path=.
const _TOOL_PATH_ALIASES = ["target_file", "targetFile", "filePath", "file_path"];

function _normalizeToolArgRecord(args, name = "") {
  if (args == null || typeof args !== "object" || Array.isArray(args)) return {};
  const out = {};
  for (const [k, v] of Object.entries(args)) {
    if (String(k).startsWith("_")) continue;
    if (v == null || v === "") continue;
    out[k] = v;
  }
  if (!out.path) {
    for (let i = 0; i < _TOOL_PATH_ALIASES.length; i += 1) {
      const key = _TOOL_PATH_ALIASES[i];
      if (out[key]) {
        out.path = out[key];
        break;
      }
    }
  }
  for (let i = 0; i < _TOOL_PATH_ALIASES.length; i += 1) {
    delete out[_TOOL_PATH_ALIASES[i]];
  }
  // Agent schema defaults (budget_usd=0,
  // runs=[]) are noise on control verbs and empty fan-out. Hide them so
  // a list/reconcile call does not look like a foreground dispatch.
  const action = String(out.action || "").toLowerCase();
  if (action === "list" || action === "abort" || action === "steer") {
    delete out.background;
    if (Array.isArray(out.runs) && !out.runs.length) delete out.runs;
    if (out.budget_usd === 0) delete out.budget_usd;
  } else if (Array.isArray(out.runs) && !out.runs.length) {
    delete out.runs;
  }
  if (/^agent(sync|async)?$/i.test(name)) {
    delete out.background;
    if (Array.isArray(out.runs)) {
      out.runs = out.runs.map((run) => _normalizeToolArgRecord(run, name));
    }
  }
  return out;
}

// Pure arg formatter (mirrors _toolArgs but self-contained for the model).
function _blockToolArgs(args, name = "") {
  if (args == null) return "";
  if (typeof args === "string") {
    const trimmed = args.trim();
    if (!trimmed || trimmed === "{}") return "";
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        args = parsed;
      } else {
        return trimmed;
      }
    } catch (_e) {
      return trimmed;
    }
  }
  try {
    const obj = _normalizeToolArgRecord(args, name);
    return Object.entries(obj)
      .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
      .join(", ");
  } catch (_e) {
    return "";
  }
}

// Session writer always sends arguments: {} on title-only ACP frames.
// Prefer the first candidate that formats to a non-empty one-liner.
function _eventToolArgs(event) {
  const candidates = [
    event.arguments,
    event.raw_arguments,
    event.args,
    event.input,
    event.rawInput,
    event.raw_input,
  ];
  for (let i = 0; i < candidates.length; i += 1) {
    if (_blockToolArgs(candidates[i])) return candidates[i];
  }
  return event.arguments != null ? event.arguments : "";
}

// The markdown prose carried in an OutputSection call's ``content`` argument
// (args may arrive as an object or a JSON string).
function _outputSectionContent(args) {
  let a = args;
  if (typeof a === "string") {
    try {
      a = JSON.parse(a);
    } catch (_e) {
      return "";
    }
  }
  return a && typeof a.content === "string" ? a.content : "";
}

// ── Widget tools (TUI parity) ────────────────────────────────────────────
// TodoWrite / Task* / ExitPlanMode are plan-state, not plumbing: they render
// as live-updating structured cards (📝 checklist · 🗒️ task list · 📐 plan),
// never as raw 🔧 arg dumps — and regardless of the /tools visibility flags.
const WIDGET_TASK_TOOLS = new Set(["TaskCreate", "TaskUpdate", "TaskStop"]);

function _widgetAdapt() {
  if (typeof WidgetAdapt !== "undefined" && WidgetAdapt) return WidgetAdapt;
  if (typeof window !== "undefined" && window.WidgetAdapt) return window.WidgetAdapt;
  return null;
}

function _isPlanTool(name) {
  const adapt = _widgetAdapt();
  if (adapt && typeof adapt.kind === "function") return adapt.kind(name) === "plan";
  const key = String(name || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
  return key === "exitplanmode" || key === "createplan";
}

function _planMarkdownFromArgs(args) {
  const obj = _toolArgsObj(args);
  if (typeof obj.plan === "string" && obj.plan.trim()) return obj.plan.trim();
  const parts = [];
  if (typeof obj.name === "string" && obj.name.trim()) parts.push("# " + obj.name.trim());
  if (typeof obj.overview === "string" && obj.overview.trim()) parts.push(obj.overview.trim());
  return parts.join("\n\n").trim();
}

function _isPlanStatusOrFence(text) {
  const t = String(text || "");
  if (!t.trim()) return true;
  if (t.indexOf("UNTRUSTED-DATA") >= 0) return true;
  if (/Plan ready for operator review/i.test(t)) return true;
  return false;
}

function _isTodoTool(name) {
  const adapt = _widgetAdapt();
  if (adapt && typeof adapt.isTodo === "function") return adapt.isTodo(name);
  const key = String(name || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
  return (
    key === "todowrite" ||
    key === "todo" ||
    key === "todos" ||
    key === "updatetodos"
  );
}

// Tool-call arguments arrive as an object or a JSON string depending on the
// engine — normalize to an object (empty on parse failure).
function _toolArgsObj(args) {
  if (args && typeof args === "object") return args;
  if (typeof args === "string") {
    try {
      return JSON.parse(args) || {};
    } catch (_e) {}
  }
  return {};
}

// The normalized checklist from a TodoWrite call's ``todos`` argument, or
// null when the payload isn't a todo list.
function _todoItemsFromArgs(args) {
  const adapt = _widgetAdapt();
  if (adapt && typeof adapt.adaptTodos === "function") {
    const adapted = adapt.adaptTodos(args);
    if (adapted && Array.isArray(adapted.todos)) return adapted.todos;
    return null;
  }
  const obj = _toolArgsObj(args);
  const todos = Array.isArray(obj.todos)
    ? obj.todos
    : Array.isArray(obj.items)
      ? obj.items
      : null;
  if (!todos) return null;
  const items = todos
    .map((t) => ({
      id: String((t && (t.id || t.todoId)) || ""),
      content: String((t && t.content) || ""),
      status: String((t && t.status) || "pending"),
      activeForm: String((t && (t.activeForm || t.active_form)) || ""),
    }))
    .filter((t) => t.content);
  return items.length ? items : null;
}

// Folded/in-band tool markup that some engines (and the client's own
// foldedRawFromBlocks cache) leak into answer text_delta:
//   🔧 bash(command=ls)
//   ← OK
//   🔧 read {"filePath":"/tmp/a"}
// Lift those lines into real tool blocks so they group in Activity cards
// instead of rendering as markdown. TUI already strips the same markers.
function parseInlineToolHeader(rest) {
  const s = String(rest || "").trim();
  let m = s.match(/^([^\s(]+)\((.*)\)\s*$/);
  if (m) return { name: m[1], args: m[2] };
  m = s.match(/^(\S+)\s+(\{[\s\S]*\})\s*$/);
  if (m) return { name: m[1], args: m[2] };
  m = s.match(/^(\S+)\s+(.*)$/);
  if (m) return { name: m[1], args: m[2] };
  return { name: s || "tool", args: "" };
}

function splitInlineToolMarkup(text) {
  const tools = [];
  const prose = [];
  for (const line of String(text || "").split("\n")) {
    const t = line.trim();
    let m;
    if ((m = t.match(/^🔧\s*(.*)$/))) {
      const parsed = parseInlineToolHeader(m[1]);
      tools.push({
        name: parsed.name,
        args: parsed.args,
        status: "running",
      });
      continue;
    }
    if (/^←\s*OK$/i.test(t) && tools.length) {
      tools[tools.length - 1].status = "ok";
      continue;
    }
    if (/^←\s*failed$/i.test(t) && tools.length) {
      tools[tools.length - 1].status = "failed";
      continue;
    }
    prose.push(line);
  }
  const joined = prose.join("\n").replace(/\n{3,}/g, "\n\n");
  return { prose: joined, tools };
}

// Runtime notices (nudge / loop / distill / force-terminate / …): the engine
// emits its recoveries as TextDelta bodies shaped ``\n\n{⚠|↻} runtime:
// <event> · k=v · …\n\n`` (see ``_format_runtime_warning``,
// ``cli_timeout_continue_notice``, ``idle_empty_continue_notice``). The TUI
// mounts a RuntimeWarningNotice widget instead of appending that prose to the
// content bubble, and Telegram strips it; the Web mirrors them with an inline
// notice CARD placed exactly where the event arrived in the block stream. A
// notice is a single line, so the same line buffering that lifts 🔧 markers
// can lift it out of the answer text.
const _RUNTIME_NOTICE_GLYPHS = new Set(["↻", "⚠"]);

function _looksLikeRuntimeNoticeStart(s) {
  const t = String(s == null ? "" : s).trimStart();
  if (!t || !_RUNTIME_NOTICE_GLYPHS.has(t[0])) return false;
  const rest = t.slice(1).trimStart();
  return !rest || rest.startsWith("runtime:");
}

// Parse one complete runtime-notice line. Returns null unless the line is a
// genuine ``{⚠|↻} runtime: <event> [· k=v …]`` marker. ``raw`` preserves the
// exact trimmed text (copy-as-markdown / audit); ``fields`` holds the k=v
// pairs in engine order; tokens without ``=`` become flags (value ``true``).
function parseRuntimeNoticeLine(line) {
  const t = String(line == null ? "" : line).trim();
  if (!t) return null;
  if (!_RUNTIME_NOTICE_GLYPHS.has(t[0])) return null;
  const rest = t.slice(1).trimStart();
  if (!rest.toLowerCase().startsWith("runtime:")) return null;
  const after = String(rest.slice("runtime:".length) || "").trim();
  const tokens = after.split("·");
  const event = String(tokens.shift() || "").trim().toLowerCase();
  const fields = {};
  for (const rawTok of tokens) {
    const tok = String(rawTok || "").trim();
    if (!tok) continue;
    const eq = tok.indexOf("=");
    if (eq <= 0) {
      fields[tok] = true;
      continue;
    }
    const k = tok.slice(0, eq).trim();
    const v = tok.slice(eq + 1).trim();
    if (k) fields[k] = v;
  }
  return { glyph: t[0], event, fields, raw: t };
}

class WebBlockStream {
  constructor(opts = {}) {
    this.reasoningShown = !!opts.reasoningShown;
    this.reasoningConsolidated = !!opts.reasoningConsolidated;
    this.toolShowHeader = opts.toolShowHeader !== false;
    this.toolShowBody = !!opts.toolShowBody;
    // ``metricsLine`` is injected so the pure core needs no globals under test.
    this._metricsLine =
      opts.metricsLine ||
      (typeof metricsLine === "function" ? metricsLine : () => "");
    this.blocks = []; // ordered blocks, in arrival order
    this._think = null; // current round's 💭 block
    this._answer = null; // current round's 💬 block
    this._tools = {}; // call_id → 🔧 block
    this._titles = {}; // call_id → tool name
    this._directCalls = new Set();
    this._todoBlk = null; // the turn's single 📝 checklist card (updated in place)
    this._taskBlk = null; // the turn's single 🗒️ harness-task card
    this._planBlk = null; // the turn's single 📐 plan card (updated in place)
    this._taskPending = {}; // call_id → task row awaiting its id from the result
    this._pendingBadge = "";
    this._pendThink = ""; // hold whitespace-only leading deltas
    this._pendAnswer = "";
    this._inlineScan = ""; // incomplete text_delta line (🔧 markers split across chunks)
    this._inlineToolSeq = 0;
    this._lastInlineCallId = "";
    this.ended = false;
    this.aborted = false;
    this.error = null;
  }

  _open(kind) {
    const blk = {
      kind,
      buf: [],
      badge: "",
      toolName: "",
      args: "",
      status: "running",
      stats: "",
      closed: false,
      deleted: false,
    };
    this.blocks.push(blk);
    blk.uid = this.blocks.length; // stable per-block key (open-state across repaints)
    return blk;
  }

  static _text(blk) {
    return blk.buf.join("");
  }

  // A block carries renderable content (a tool block always has its header;
  // widget cards carry their structured payload, not buffered text).
  static _hasContent(blk) {
    if (
      blk.kind === "tool" ||
      blk.kind === "todos" ||
      blk.kind === "tasks" ||
      blk.kind === "plan"
    )
      return true;
    return WebBlockStream._text(blk).trim() !== "";
  }

  _close(blk) {
    if (!blk || blk.closed) return;
    blk.closed = true;
    // A pinned/eager 💭 that never received real reasoning (e.g. an "on" turn
    // whose only output was a delegation) is collapsed away — no stray idle 💭.
    if (blk.kind === "think" && !WebBlockStream._text(blk).trim())
      blk.deleted = true;
  }

  _ensureTopThink() {
    if (this._think === null) this._think = this._open("think");
  }

  thinkingDelta(delta) {
    if (!this.reasoningShown || !delta) return;
    // "on" (consolidated): live deltas are gated OUT — the single 💭 is filled
    // once from turn_ended.final_thinking (pinned above the answer).
    if (this.reasoningConsolidated) return;
    if (this._think === null) {
      this._pendThink += delta;
      if (!this._pendThink.trim()) return; // hold whitespace-only leads
      this._think = this._open("think");
      this._think.buf.push(this._pendThink);
      this._pendThink = "";
    } else {
      const last = this.blocks.length
        ? this.blocks[this.blocks.length - 1]
        : null;
      // A tool after this 💭 means the next thought is a new trace. Reusing
      // the closed pointer merged early API-call reasoning into the first
      // card until a later call finally split the sequence.
      if (
        this._think.closed
        && last
        && last !== this._think
        && last.kind === "tool"
      ) {
        this._think = this._open("think");
        this._think.buf.push(delta);
      } else {
        this._think.buf.push(delta);
        // A follow-on trace after the answer opened must look live again.
        // Keep the same 💭 (no fragment) but clear closed so the Activity
        // shell can show running instead of a stale ✓.
        this._think.closed = false;
        this._think.deleted = false;
      }
    }
  }

  _looksLikeInlineToolPrefix(s) {
    const t = String(s || "").trimStart();
    return t.startsWith("🔧") || t.startsWith("←");
  }

  _normToolArgs(args) {
    return String(args || "").replace(/\s+/g, " ").trim();
  }

  _findMatchingTool(name, args) {
    const n = String(name || "").toLowerCase();
    const a = this._normToolArgs(args);
    for (const blk of this.blocks) {
      if (blk.kind !== "tool") continue;
      if (String(blk.toolName || "").toLowerCase() !== n) continue;
      if (this._normToolArgs(blk.args) === a) return blk;
    }
    return null;
  }

  _liftInlineTool(name, args, status) {
    const existing = this._findMatchingTool(name, args);
    if (existing) {
      this._lastInlineCallId = existing.callId || this._lastInlineCallId;
      if (status === "ok" || status === "failed") {
        existing.status = status;
      }
      return;
    }
    const id = "inline-" + String(++this._inlineToolSeq);
    this._lastInlineCallId = id;
    this.toolCall(id, name, args);
    if (status === "ok") this.toolResultMeta(id, true);
    else if (status === "failed") this.toolResultMeta(id, false);
  }

  _finishInlineTool(ok) {
    if (this._lastInlineCallId) {
      this.toolResultMeta(this._lastInlineCallId, ok);
      return true;
    }
    return false;
  }

  // A runtime notice (nudge/loop/distill/…) is its own chronological block.
  // Closing the current round first guarantees the card lands BETWEEN the
  // content that preceded it and whatever the agent writes next — the exact
  // stream position where the engine injected the notice (RuntimeWarningNotice
  // parity). The block carries the structured fields; the notice text never
  // enters the 💬 answer.
  _openNoticeBlock(notice) {
    this._closeRound();
    const blk = this._open("notice");
    blk.notice = notice;
    blk.closed = true; // static card — nothing left to stream into it
  }

  _consumeInlineLine(line, hadNewline) {
    const trimmed = String(line || "").trim();
    const notice = parseRuntimeNoticeLine(trimmed);
    if (notice) {
      this._openNoticeBlock(notice);
      return;
    }
    let m;
    if ((m = trimmed.match(/^🔧\s*(.*)$/))) {
      const parsed = parseInlineToolHeader(m[1]);
      this._liftInlineTool(parsed.name, parsed.args, "running");
      return;
    }
    if (/^←\s*OK$/i.test(trimmed)) {
      if (this._finishInlineTool(true)) return;
    } else if (/^←\s*failed$/i.test(trimmed)) {
      if (this._finishInlineTool(false)) return;
    }
    this._textDeltaPlain(hadNewline ? line + "\n" : line);
  }

  _flushInlineToolScan() {
    if (!this._inlineScan) return;
    this._consumeInlineLine(this._inlineScan, false);
    this._inlineScan = "";
  }

  // Live engines and the folded-raw cache sometimes dump 🔧 name(args) /
  // ← OK|failed lines into text_delta. Pull complete lines out as tool
  // blocks (Activity cards) and keep only remaining prose in 💬.
  textDelta(delta) {
    if (!delta) return;
    this._inlineScan = (this._inlineScan || "") + delta;
    const parts = this._inlineScan.split("\n");
    this._inlineScan = parts.pop();
    for (let i = 0; i < parts.length; i++) {
      this._consumeInlineLine(parts[i], true);
    }
    const rem = this._inlineScan;
    if (
      rem
      && !this._looksLikeInlineToolPrefix(rem)
      && !_looksLikeRuntimeNoticeStart(rem)
    ) {
      this._textDeltaPlain(rem);
      this._inlineScan = "";
    }
  }

  _textDeltaPlain(delta) {
    if (!delta) return;
    if (this._answer === null) {
      this._pendAnswer += delta;
      if (!this._pendAnswer.trim()) return; // hold whitespace-only leads
      // Ordering (B): pin a 💭 ABOVE the answer the instant content opens whenever
      // reasoning is shown but none is open yet. In "on" mode the consolidated 💭
      // is filled later from turn_ended.final_thinking. In "stream" mode reasoning
      // USUALLY leads, but some engines (and the native loop, observed live) lead a
      // round with text_delta and trail the reasoning — typically the final rounds,
      // right after a tool closed the prior round (round_ended does not reset
      // _think/_answer). Without this pin the late 💭 mounts BELOW the answer
      // (answer-before-reasoning). An empty pinned 💭 is collapsed away on close,
      // so a truly reasoning-free round leaves no stray idle 💭. Mirrors Telegram
      // _BlockStream's `reasoning_shown and _think is None` guard exactly.
      // Close a filled trace when the answer starts so a flushed card is ✓
      // only after that reasoning actually finished. Keep the pointer so a
      // later thinking_delta can re-open the same 💭 instead of fragmenting.
      if (this._think && WebBlockStream._text(this._think).trim()) {
        this._close(this._think);
      } else if (this.reasoningShown && this._think === null) {
        this._ensureTopThink();
      }
      this._answer = this._open("answer");
      if (this._pendingBadge) {
        this._answer.badge = this._pendingBadge;
        this._pendingBadge = "";
      }
      this._answer.buf.push(this._pendAnswer);
      this._pendAnswer = "";
    } else {
      this._answer.buf.push(delta);
    }
  }

  toolCall(callId, name, args) {
    this._titles[callId] = name;
    // Compaction output belongs to its view-only child session in Agents.
    // Keep the call identity for replay and errors without splitting the answer.
    if (name === "compact_context") return;
    // AskUserQuestion round-trips to a human — never fold into Activity.
    // History reconstruction mounts the interactive card from askUserCalls.
    if (_isAskUserTool(name)) {
      if (!this.askUserCalls) this.askUserCalls = [];
      if (this.askBoundary == null) this.askBoundary = this.blocks.length;
      if (!this.askUserCalls.some((c) => c.callId === callId)) {
        this.askUserCalls.push({
          callId,
          args: _toolArgsObj(args),
          ok: undefined,
          content: null,
        });
      }
      return;
    }
    // OutputSection: the model emits a long answer ONE section at a time, and
    // the section's prose lives in the call's ``content`` argument. It IS the
    // answer — render it as markdown in 💬, not as a raw tool block.
    if (name === "OutputSection") {
      const c = _outputSectionContent(args);
      if (c) this.textDelta((this._sawOutputSection ? "\n\n" : "") + c);
      this._sawOutputSection = true;
      return;
    }
    // Direct-dispatch (/claude, /codex …): the tool BODY is the
    // user-facing answer, so route it into 💬 — ALWAYS, regardless of /tools —
    // never a 🔧 block. Queue a quiet attribution badge.
    if (DIRECT_DISPATCH_TITLES.has(name)) {
      this._directCalls.add(callId);
      const b = DELEGATE_BADGES[name];
      if (b) this._pendingBadge = b;
      return;
    }
    // Widget tools (TUI parity): structured cards, shown regardless of /tools —
    // they carry plan state the operator tracks, not tool plumbing. Each card
    // is opened once per turn and updated in place on subsequent calls.
    if (_isTodoTool(name)) {
      const todos = _todoItemsFromArgs(args);
      if (todos) {
        if (!this._todoBlk) {
          this._closeRound();
          this._todoBlk = this._open("todos");
        }
        this._todoBlk.todos = todos;
      }
      if (!this.toolShowHeader) return;
    }
    if (_isPlanTool(name)) {
      this._setPlan(_planMarkdownFromArgs(args));
      return;
    }
    if (WIDGET_TASK_TOOLS.has(name)) {
      const a = _toolArgsObj(args);
      if (!this._taskBlk) {
        this._closeRound();
        this._taskBlk = this._open("tasks");
        this._taskBlk.tasks = [];
      }
      const rows = this._taskBlk.tasks;
      const tid = a.taskId != null ? String(a.taskId) : "";
      if (name === "TaskCreate") {
        const row = {
          id: "",
          content: String(a.subject || ""),
          status: "pending",
        };
        rows.push(row);
        this._taskPending[callId] = row; // the result body names the id ("Task #N created")
      } else {
        const row = tid ? rows.find((r) => r.id === tid) : null;
        if (row) {
          if (a.subject) row.content = String(a.subject);
          if (a.status) row.status = String(a.status);
          if (name === "TaskStop") row.status = "deleted";
        } else if (tid) {
          rows.push({
            id: tid,
            content: String(a.subject || `task #${tid}`),
            status:
              name === "TaskStop" ? "deleted" : String(a.status || "pending"),
          });
        }
      }
      return;
    }
    // Internal/hidden tools, and tools when /tools off, get no block of their
    // own — and (matching Telegram) do NOT split the round, so the answer keeps
    // accumulating contiguously.
    if (INTERNAL_TOOL_TITLES.has(name) || !this.toolShowHeader) return;
    const existing = this._tools[callId];
    if (existing && existing.kind === "tool") {
      // Same in-flight call: ACP often starts title-only, then repeats
      // ToolCallStarted with real args. Keep any streamed output.
      // A later Shell that reuses a finished call_id still resets the body.
      const stillLive = existing.status === "running" && !existing.closed;
      // Title-only / empty in_progress updates must not wipe a path that
      // already landed (Cursor Read/Grep send rawInput:{} after target_file).
      const formatted = _blockToolArgs(args, name);
      if (formatted) existing.args = formatted;
      _delegateInit(existing, name, args);
      existing.status = "running";
      existing.closed = false;
      if (!stillLive) existing.buf = [];
      return;
    }
    this._closeRound();
    const blk = this._open("tool");
    blk.toolName = name;
    blk.args = _blockToolArgs(args, name);
    blk.callId = callId; // stable key so a click-to-expand survives repaints
    _delegateInit(blk, name, args);
    this._tools[callId] = blk;
  }

  _setPlan(plan) {
    const body = String(plan || "").trim();
    if (!body || _isPlanStatusOrFence(body)) return;
    if (!this._planBlk) {
      this._closeRound();
      this._planBlk = this._open("plan");
    }
    this._planBlk.plan = body;
  }

  // Close the current round's 💭/💬 so the next block mounts below them (in
  // "on" mode the single top 💭 stays pinned; only the answer closes).
  _closeRound() {
    if (this.reasoningConsolidated) {
      if (this.reasoningShown) this._ensureTopThink();
      this._close(this._answer);
      this._answer = null;
    } else {
      this._close(this._think);
      this._close(this._answer);
      this._think = null;
      this._answer = null;
    }
    this._pendThink = "";
    this._pendAnswer = "";
  }

  roundStarted(roundIdx) {
    if (Number(roundIdx || 0) > 0) this._closeRound();
  }

  toolDelta(callId, delta) {
    if (this._titles[callId] === "compact_context") return;
    if (this._directCalls.has(callId)) {
      this.textDelta(delta);
      return;
    }
    const blk = this._tools[callId];
    if (!blk || !delta) return;
    blk.buf.push(delta);
  }

  toolResultMeta(callId, ok, outcome, memberOutcomes) {
    const ask = (this.askUserCalls || []).find((c) => c.callId === callId);
    if (ask) ask.ok = ok;
    const blk = this._tools[callId];
    if (blk && !blk.delegate) blk.status = ok === false ? "failed" : "ok";
    if (blk && blk.delegate) {
      blk.delegateResultOk = ok;
      blk.delegateOutcome = outcome;
      blk.delegateMemberOutcomes = memberOutcomes || [];
      // Async success acknowledges dispatch only. Its body or the job state
      // identifies individual children; it is never completion by itself.
      if (!blk.delegateAsync || ok === false) {
        for (const [index, child] of blk.delegateChildren.entries()) {
          const resultOutcome = (memberOutcomes || [])[index] || outcome;
          if (!child.terminal && !child.detached) child.status = resultOutcome === "done" ? "ok"
            : resultOutcome || (ok === false ? "failed" : "ok");
        }
        _delegateStatus(blk);
      }
    }
  }

  toolResult(callId, content, ok) {
    if (this._titles[callId] === "compact_context") {
      if (ok === false) {
        this.textDelta("\n\nContext compaction failed." +
          (content ? "\n\n" + content : "") + "\n\n");
      }
      return;
    }
    const ask = (this.askUserCalls || []).find((c) => c.callId === callId);
    if (ask) {
      ask.content = typeof content === "string" ? content : "";
      if (ok !== undefined) ask.ok = ok;
      return;
    }
    if (this._directCalls.has(callId)) {
      this.textDelta(content || "");
      return;
    }
    if (_isPlanTool(this._titles[callId] || "")) {
      const plan = String(content || "").trim();
      if (plan && !_isPlanStatusOrFence(plan)) this._setPlan(plan);
      return;
    }
    // A TaskCreate row learns its id from the result body ("Task #N created…").
    const trow = this._taskPending[callId];
    if (trow) {
      delete this._taskPending[callId];
      const m = /#(\d+)/.exec(String(content || ""));
      if (m && ok !== false) trow.id = m[1];
      return;
    }
    const blk = this._tools[callId];
    if (!blk) return;
    if (content) blk.buf.push("\n" + content);
    if (blk.delegate) _delegateResultContent(blk, content, ok ?? blk.delegateResultOk);
    else blk.status = ok === false ? "failed" : "ok";
    this._close(blk);
  }

  delegationUpdate(event) {
    let changed = false;
    for (const blk of this.blocks) {
      if (!blk.delegate) continue;
      for (const child of blk.delegateChildren || []) {
        const taskId = String(event.child_session_id || event.child_sid || event.task_id || "");
        const callId = String(event.call_id || "");
        const jobId = String(event.job_id || "");
        const matches = callId ? child.callId === callId
          : jobId && child.jobId ? child.jobId === jobId
            : !child.terminal && taskId && child.taskId === taskId;
        if (!matches) continue;
        const before = JSON.stringify(child);
        if (taskId) child.taskId = taskId;
        if (event.type === "subagent_linked") child.linked = true;
        if (jobId) child.jobId = jobId;
        if (event.child_agent_name || event.agent) child.agent = event.child_agent_name || event.agent;
        const status = String(event.status || "");
        if (event.type === "subagent_result" || event.type === "subagent_done" || DELEGATE_TERMINAL_STATUSES.has(status)) {
          child.status = ["cancelled", "aborted", "refused"].includes(status) ? status
            : status === "done" || status === "completed" ? "ok"
            : status === "failed" || event.ok === false || event.error ? "failed" : "ok";
          child.terminal = true;
        } else if (!child.terminal && status && blk.delegateAsync) {
          child.status = "running";
          child.detached = true;
        }
        changed = JSON.stringify(child) !== before || changed;
      }
      _delegateStatus(blk);
    }
    return changed;
  }

  turnEnded(metrics, opts) {
    this._flushInlineToolScan();
    opts = opts || {};
    const finalThinking = opts.finalThinking || "";
    this.aborted = !!opts.aborted;
    this.error = opts.error != null ? opts.error : null;
    if (this.aborted) {
      // Keep already-streamed blocks; the painter stamps the aborted tag.
      this.abortFast("");
    }
    // "on" reasoning: consolidate final_thinking into the (empty) 💭 block.
    if (
      !this.aborted && finalThinking &&
      this.reasoningShown &&
      (this._think === null || !WebBlockStream._text(this._think).trim())
    ) {
      if (this._think === null) this._think = this._open("think");
      this._think.buf.push(finalThinking);
    }
    // Stats line closes the turn on the LAST visible block (prefer the answer).
    if (!this.error || this.aborted) {
      const line = this._metricsLine(metrics || {});
      if (line) {
        let target = this._answer || this._lastBlock();
        if (target === null) target = this._open("answer");
        target.stats = line;
      }
    }
    const leftover = this.aborted || this.error ? "failed" : "ok";
    for (const blk of this.blocks) {
      // The parent can finish or be interrupted while detached children run.
      if (blk.delegate && blk.delegateAsync) {
        if (this.error) {
          for (const child of blk.delegateChildren || []) {
            if (!child.detached && !child.linked && !child.terminal) child.status = "failed";
          }
          _delegateStatus(blk);
        }
        this._close(blk);
        continue;
      }
      if (blk.delegate) {
        for (const child of blk.delegateChildren || []) {
          if (child.status === "running") child.status = leftover;
        }
      }
      // The turn is over. A missing tool_result is not "still running" —
      // that left SUCCEEDED cards with Activity / Reasoning · running and
      // kept the composer ticker on a stale tool_running snapshot. Abort
      // or error fails leftovers; a normal close marks them done.
      if (blk.status === "running") blk.status = leftover;
      this._close(blk);
    }
    this.ended = true;
  }

  /** Hard abort: stop accepting deltas; mark ended without stats drain. */
  abortFast(msg) {
    if (this.ended) return;
    this._flushInlineToolScan();
    this.ended = true;
    this.aborted = true;
    this._pendThink = "";
    this._pendAnswer = "";
    this._inlineScan = "";
    for (const blk of this.blocks) {
      if (blk.delegate && blk.delegateAsync) {
        for (const child of blk.delegateChildren || []) {
          if (!child.detached && !child.linked && !child.terminal) child.status = "failed";
        }
        _delegateStatus(blk);
        blk.closed = true;
        continue;
      }
      if (blk.delegate) {
        for (const child of blk.delegateChildren || []) {
          if (child.status === "running") child.status = "failed";
        }
      }
      if (blk.status === "running") blk.status = "failed";
      blk.closed = true;
    }
    if (msg) {
      if (this._answer === null) {
        this._answer = this._open("answer");
      }
      if (this._answer !== null) {
        this._answer.buf = [msg];
      }
    }
  }

  _lastBlock() {
    for (let i = this.blocks.length - 1; i >= 0; i--) {
      if (WebBlockStream._hasContent(this.blocks[i])) return this.blocks[i];
    }
    return this.blocks.length ? this.blocks[this.blocks.length - 1] : null;
  }

  // The renderable block list, in order, dropping collapsed-empty 💭 blocks.
  visibleBlocks() {
    return this.blocks.filter((b) => !b.deleted);
  }

  isCompactionOnly() {
    const names = Object.values(this._titles);
    return !this.aborted && !this.error && names.length > 0 &&
      names.every((name) => name === "compact_context") &&
      !this.visibleBlocks().some((blk) => WebBlockStream._hasContent(blk));
  }
}

// Rebuild a completed turn from the lossless render_segments returned by the
// history endpoint. Feeding the same WebBlockStream used by live SSE keeps
// refresh/reopen rendering identical: reasoning disclosures, tool headers,
// result bodies, task cards, and plan cards retain their normal structure.
function _mergeThinkText(a, b) {
  a = String(a || "");
  b = String(b || "");
  if (!a) return b;
  if (!b) return a;
  if (b.startsWith(a) || b.includes(a)) return b;
  if (a.startsWith(b) || a.includes(b) || a.endsWith(b)) return a;
  return a + b;
}

function _replayIsReasoningOnly(opts, segments) {
  opts = opts || {};
  if (!opts.inProgress) return false;
  const phase = String(opts.phase || "");
  if (
    phase === "streaming" ||
    phase === "writing" ||
    phase === "tool_running" ||
    phase === "tool"
  ) {
    return false;
  }
  const segs = Array.isArray(segments) ? segments : [];
  if (segs.some((seg) => seg && seg.kind === "tool")) return false;
  if (phase === "reasoning" || phase === "prefill" || phase === "prompting") {
    return true;
  }
  // In-flight, no tools: leftover ``content`` is still CoT. vLLM often
  // splits one trace across thinking_delta + text_delta; joining mid-think
  // used to paint the second half as a normal answer.
  return segs.some(
    (seg) =>
      seg && seg.kind === "assistant" && String(seg.thinking || "").trim(),
  );
}

function blockStreamFromSavedSegments(segments, fallbackContent, vis, opts) {
  opts = opts || {};
  const aborted = !!opts.aborted;
  const error = opts.error != null && opts.error !== "" ? opts.error : null;
  const unfinishedTool = (Array.isArray(segments) ? segments : []).some(
    (seg) =>
      seg &&
      seg.kind === "tool" &&
      seg.ok == null &&
      seg.content == null,
  );
  // Refresh mid-shell (and a persisted turn_ended that beat the result)
  // must keep the live card open. Finalizing here painted SUCCEEDED +
  // shell Success / "No output returned."
  const inProgress = !!opts.inProgress || (!aborted && !error && unfinishedTool);

  const bs = new WebBlockStream(blockFlagsFor(vis));
  let finalThinking = "";
  let lastMetrics = {};
  let sawAssistantContent = false;
  const segs = Array.isArray(segments) ? segments : [];
  // Older direct-dispatch turns may repeat the tool body in the committed
  // assistant answer. Match the typed result, never arbitrary "compacted" prose.
  const compactionResults = new Set(segs
    .filter((seg) => seg && seg.kind === "tool" && seg.name === "compact_context" &&
      typeof seg.content === "string" && seg.content.trim())
    .map((seg) => seg.content.trim()));
  const reasoningOnly = _replayIsReasoningOnly(opts, segs);
  for (const seg of segs) {
    if (!seg || typeof seg !== "object") continue;
    if (seg.kind === "assistant") {
      let thinking = typeof seg.thinking === "string" ? seg.thinking : "";
      let content = typeof seg.content === "string" ? seg.content : "";
      if (compactionResults.has(content.trim())) content = "";
      if (reasoningOnly && content) {
        thinking = _mergeThinkText(thinking, content);
        content = "";
      }
      if (thinking) {
        finalThinking = _mergeThinkText(finalThinking, thinking);
        if (!bs.reasoningConsolidated) bs.thinkingDelta(thinking);
      } else if (seg.lazy_thinking && seg.thinking_chars) {
        const think = bs._open("think");
        think.lazy = true;
        think.thinkingCardId =
          seg.thinking_card_id ||
          (seg.turn_idx != null ? `reason-${seg.turn_idx}` : "");
        think.closed = true;
      }
      if (content) {
        sawAssistantContent = true;
        bs.textDelta(content);
      }
      if (seg.metrics && typeof seg.metrics === "object")
        lastMetrics = seg.metrics;
      continue;
    }
    if (seg.kind === "tool") {
      const callId = String(seg.call_id || "");
      const args =
        seg.arguments != null ? seg.arguments : seg.raw_arguments;
      bs.toolCall(callId, seg.name || "tool", args);
      for (const child of seg.delegations || []) bs.delegationUpdate({ ...child, type: "subagent_linked" });
      if (seg.ok !== null && seg.ok !== undefined)
        bs.toolResultMeta(callId, seg.ok, seg.outcome, seg.member_outcomes);
      if (seg.content != null)
        bs.toolResult(callId, String(seg.content), seg.ok);
      if (seg.lazy) {
        for (let bi = bs.blocks.length - 1; bi >= 0; bi -= 1) {
          const blk = bs.blocks[bi];
          if (blk && blk.callId === callId) {
            blk.lazy = true;
            if (seg.card_id) blk.cardId = seg.card_id;
            break;
          }
        }
      }
      continue;
    }
    if (seg.kind === "notice") {
      const subtype = String(
        seg.subtype || (seg.notice && seg.notice.subtype) || "",
      ).toLowerCase();
      if (subtype === "compaction") {
        // The child session retains the summary, stats and lifecycle details.
        continue;
      } else if (typeof seg.text === "string") {
        bs.textDelta(seg.text);
        sawAssistantContent = true;
      }
    }
  }
  // Some legacy completed turns have tool-only segments or a final answer that
  // was rewritten at turn_ended. Keep the canonical committed answer visible
  // whenever the segments themselves contain no assistant prose.
  // NEVER inject fallback prose while the turn is still actively in-flight:
  // in-flight turns receive live chunks via SSE, and injecting raw fallback
  // creates spurious answer blocks that leak thinking traces or split activity groups.
  if (
    !inProgress &&
    !sawAssistantContent &&
    typeof fallbackContent === "string" &&
    !compactionResults.has(fallbackContent.trim()) &&
    fallbackContent.trim()
  ) {
    if (reasoningOnly) {
      if (!finalThinking && !bs.reasoningConsolidated) {
        bs.thinkingDelta(fallbackContent);
      }
    } else {
      const parsed = typeof parseAssistantContent === "function"
        ? parseAssistantContent(fallbackContent)
        : null;
      const cleanAnswer = parsed && typeof parsed.answer === "string"
        ? parsed.answer
        : fallbackContent;
      if (cleanAnswer && cleanAnswer.trim()) {
        bs.textDelta(cleanAnswer);
      }
    }
  }
  if (!inProgress) {
    bs.turnEnded(lastMetrics, {
      finalThinking: bs.reasoningConsolidated ? finalThinking : "",
      aborted,
      error,
    });
  }
  return bs;
}
// <<< BLOCKSTREAM_CORE_END

function adoptLastAssistantAsLiveTurn() {
  // Join an already-running turn whose REST snapshot already painted the
  // latest assistant. New SSE frames (from EOF) must continue that bubble
  // instead of opening a second one.
  if (state.liveTurn) return state.liveTurn;
  const cache = state.msgCache || [];
  // Prompting-only refresh: REST painted the new user bubble and no
  // assistant tokens yet. Do not steal the previous turn's closer.
  const last = cache[cache.length - 1];
  if (last && last.role === "user") return null;
  for (let i = cache.length - 1; i >= 0; i -= 1) {
    const m = cache[i];
    if (!m || m.role !== "assistant" || !m.el) continue;
    const bs =
      typeof blockStreamFromSavedSegments === "function" && m.segments
        ? blockStreamFromSavedSegments(m.segments, m.content, state.vis, {
            aborted: !!(m.aborted || m.status === "aborted"),
            error: m.error || (m.status === "error" ? "error" : null),
            inProgress: true,
            phase: typeof _sessionReplayPhase === "function"
              ? _sessionReplayPhase()
              : "",
          })
        : new WebBlockStream(blockFlagsFor(state.vis));
    state.liveTurn = { el: m.el, bs };
    return state.liveTurn;
  }
  return null;
}

function _assistantCardIsOpen(m) {
  // A closer that already has a terminal outcome belongs to a finished turn.
  // Re-adopting it would merge the next turn (or mint a second SUCCEEDED
  // footer) into the previous card.
  if (!m || m.role !== "assistant" || !m.el) return false;
  if (m.aborted || m.status === "aborted" || m.status === "error" || m.status === "ok") {
    return false;
  }
  if (m.endedAt) return false;
  return true;
}

function adoptInFlightAssistant() {
  // Join the OPEN assistant card of the current turn. Unlike
  // adoptLastAssistantAsLiveTurn, never steal a completed closer and never
  // walk past a mid-turn user steer.
  if (state.liveTurn) return state.liveTurn;
  const cache = state.msgCache || [];
  const last = cache[cache.length - 1];
  if (!last || last.role === "user") return null;
  if (!_assistantCardIsOpen(last)) return null;
  return adoptLastAssistantAsLiveTurn();
}

function _finalizeNonLiveAssistantCards(event) {
  // A stale turn_ended for a previous physical_turn_id must settle that
  // card's leftover running blocks without touching the new live turn.
  const liveEl = state.liveTurn && state.liveTurn.el;
  const cache = state.msgCache || [];
  const opts = {
    finalThinking: (event && event.final_thinking) || "",
    aborted: !!(event && event.aborted),
    error: event && event.error,
  };
  const metrics = (event && event.metrics) || {};
  for (let i = 0; i < cache.length; i += 1) {
    const m = cache[i];
    if (!m || m.role !== "assistant" || !m.el) continue;
    if (liveEl && m.el === liveEl) continue;
    if (event.physical_turn_id && m.physicalTurnId && event.physical_turn_id !== m.physicalTurnId) continue;
    let bs = m.liveBs;
    if (bs && bs.ended) continue;
    if (!bs && m.segments && typeof blockStreamFromSavedSegments === "function") {
      const open = m.status === "running" || !m.endedAt;
      if (!open) continue;
      bs = blockStreamFromSavedSegments(m.segments, m.content, state.vis, {
        aborted: !!opts.aborted,
        error: opts.error,
        inProgress: true,
      });
      m.liveBs = bs;
    }
    if (!bs || bs.ended) continue;
    bs.turnEnded(metrics, opts);
    if (typeof renderBlockTurn === "function") {
      renderBlockTurn({
        el: m.el,
        bs,
        activityDisclosureState: m.activityDisclosureState,
      });
    }
  }
}

// A frame of a turn this tab sealed with Stop/Steer, seen after the instant-
// abort gate dropped (settlement timeout). While the gate is up the
// _instantAbort branch of turn_ended owns the terminal instead.
function _isLocallyAbortedTurn(event) {
  if (state._instantAbort) return false;
  const pid = String((event && event.physical_turn_id) || "").trim();
  return !!(pid && state.locallyAbortedTurnIds && state.locallyAbortedTurnIds[pid]);
}

function _noteClosedPhysicalTurn(event) {
  // JSONL replay is the path: one physical_turn_id = one card. Mark that
  // UUID closed so a later replay of the same turn_started cannot mint
  // again. Only drop livePhysicalTurnId when this terminal is for the live
  // UUID (or the event has no pid — the logical commit that closes the
  // live turn). A stale abort for another UUID must not close this turn.
  const eventPid = String((event && event.physical_turn_id) || "").trim();
  const livePid = String(state.livePhysicalTurnId || "").trim();
  const pid = eventPid || livePid;
  if (pid) {
    if (!state.closedPhysicalTurnIds) state.closedPhysicalTurnIds = {};
    state.closedPhysicalTurnIds[pid] = true;
  }
  if (!eventPid || eventPid === livePid) {
    state.livePhysicalTurnId = "";
    state.liveEngine = "";
  }
}

function _turnStartedIsReplay(event) {
  // REST history (or a prior SSE connection) already painted this turn.
  // Same physical_turn_id is the same JSONL turn. Same prompt as the
  // last user bubble AND an open assistant / liveTurn is a resume.
  // Mid-turn steers sit after the original prompt; walk past them so a
  // refresh+SSE resume does not mint a second "You" bubble.
  const physicalTurnId = String((event && event.physical_turn_id) || "");
  if (physicalTurnId && state.livePhysicalTurnId && physicalTurnId !== state.livePhysicalTurnId) return false;
  if (physicalTurnId && state.livePhysicalTurnId === physicalTurnId) {
    if (state.liveTurn) return true;
    const cache = state.msgCache || [];
    for (let i = cache.length - 1; i >= 0; i -= 1) {
      const m = cache[i];
      if (!m) continue;
      if (m.role === "assistant") return _assistantCardIsOpen(m);
    }
    return false;
  }
  const parts = typeof _userPromptParts === "function"
    ? _userPromptParts(event && event.user)
    : { text: "" };
  const want = String((parts && parts.text) || "").replace(/\s+/g, " ").trim();
  if (!want) return false;
  const cache = state.msgCache || [];
  let lastUser = null;
  let lastAsst = null;
  for (let i = cache.length - 1; i >= 0; i -= 1) {
    const m = cache[i];
    if (!m) continue;
    if (m.role === "assistant") {
      if (!lastAsst) lastAsst = m;
      if (_assistantCardIsOpen(m)) continue;
      break;
    }
    if (m.role === "user") {
      lastUser = m;
      if (lastAsst && _assistantCardIsOpen(lastAsst)) continue;
      break;
    }
  }
  if (!lastUser) return false;
  const have = String(lastUser.content || "").replace(/\s+/g, " ").trim();
  if (have !== want) return false;
  if (state.liveTurn) return true;
  return _assistantCardIsOpen(lastAsst);
}

function _attachLiveMemoryFlushAssistant() {
  // Silent MEMORY FLUSH turn: stream the assistant reply into the collapsed
  // card opened on turn_started, not a full assistant bubble — and not the
  // previous closer. Adopting first painted flush tools into Status and
  // kept that card's start time (17-minute SUCCEEDED).
  if (!state.liveMemoryFlushEl || !state.liveMemoryFlushEl.querySelector) {
    return null;
  }
  const replyHost = state.liveMemoryFlushEl.querySelector(
    ".memory-flush-reply",
  );
  if (!replyHost) return null;
  replyHost.hidden = false;
  replyHost.innerHTML = "";
  const el = document.createElement("div");
  el.className = "message message-assistant memory-flush-live";
  el.innerHTML = '<div class="message-content"></div>';
  replyHost.appendChild(el);
  state.liveTurn = {
    el,
    bs: new WebBlockStream(blockFlagsFor(state.vis)),
    memoryFlush: true,
  };
  return state.liveTurn;
}

function ensureLiveAssistant() {
  if (state.liveTurn) return state.liveTurn;
  const flushTurn = _attachLiveMemoryFlushAssistant();
  if (flushTurn) return flushTurn;
  const adopted = adoptInFlightAssistant();
  if (adopted) return adopted;
  if (!state.liveTurn) {
    // Use the agent stamped on turn_started (logical lane author), not just
    // the picker — history and live bubbles must agree after an agent switch.
    const agent = state.liveTurnAgent || state.currentAgent || "";
    // Stamp the bubble with the turn's start time (the gateway stamps
    // ``ts`` on turn_started; the same value later shows in history).
    const el = appendMessage(
      "assistant", "", true, null, agent, null,
      state.liveTurnTs || new Date().toISOString(),
    );
    state.liveTurn = { el, bs: new WebBlockStream(blockFlagsFor(state.vis)) };
    _stampTurnOutcome(el, "running");
    const idx = state.msgCache.findIndex((m) => m.el === el);
    if (idx >= 0) state.msgCache[idx].status = "running";
  }
  return state.liveTurn;
}

// --- shared stats-row formatter (JS port of sharedagents_core.metrics_format) ---
// Field selection / ordering / number formatting MUST stay byte-identical to
// the Python ``metrics_segments`` so the TUI, Telegram and Web stats rows can
// never drift. ``web/tests/test_metrics_format.js`` pins the pair against the
// SAME goldens as ``tests/test_metrics_format.py``.
function _metricsModelLabel(alias, modelId) {
  const a = alias == null ? "" : String(alias);
  const m = modelId == null ? "" : String(modelId);
  if (a && m && a !== m) return `${a}/${m}`;
  return a || m;
}
function metricsSegments(m) {
  if (!m) return [];
  const parts = [];
  const num = (v) => typeof v === "number" && isFinite(v);
  const f0 = (v) => v.toFixed(0);
  const f1 = (v) => v.toFixed(1);
  const ti = (v) => Math.trunc(v);

  const label = _metricsModelLabel(m.model_alias, m.model);
  if (label) parts.push(label);
  if (num(m.classify_ms)) parts.push(`classify ${f0(m.classify_ms)}ms`);
  if (num(m.plan_ms)) parts.push(`plan ${f0(m.plan_ms)}ms`);
  if (num(m.prefill_ms)) {
    let seg = `prefill ${f0(m.prefill_ms)}ms`;
    if (num(m.prefill_tps)) seg += ` @ ${f0(m.prefill_tps)} tok/s`;
    parts.push(seg);
  } else {
    const accept = m.accept_ms || m.pft_ms;
    if (num(accept)) parts.push(`accept ${f0(accept)}ms`);
  }
  if (num(m.ttft_ms)) {
    let seg = `ttft ${f0(m.ttft_ms)}ms`;
    if (num(m.tool_events) && m.tool_events) {
      seg += ` · ${ti(m.tool_events)} tool${m.tool_events !== 1 ? "s" : ""}`;
    }
    if (num(m.ttfc_ms) && m.ttfc_ms > m.ttft_ms + 1) {
      seg += ` · ttfc ${f0(m.ttfc_ms)}ms`;
    }
    parts.push(seg);
  }
  if (num(m.tps)) {
    // Consolidated CLI engines expose a whole-turn effective rate, not
    // progressive decode throughput.
    const throughputLabel = m.tps_kind === "effective" ? "eff" : "gen";
    parts.push(
      `${throughputLabel} ${f1(m.tps)} tok/s` +
        (m.tokens_estimated ? "≈" : "")
    );
  }
  if (num(m.prompt_tokens)) {
    let seg = `in ${ti(m.prompt_tokens)}`;
    const extras = [];
    if (num(m.prompt_tokens_cached) && m.prompt_tokens_cached) {
      extras.push(`${ti(m.prompt_tokens_cached)} cached`);
    }
    if (
      num(m.prompt_tokens_fresh) &&
      (m.prompt_tokens_cached || m.prompt_tokens_fresh !== m.prompt_tokens)
    ) {
      extras.push(`${ti(m.prompt_tokens_fresh)} fresh`);
    }
    if (extras.length) seg += ` (${extras.join(", ")})`;
    if (num(m.context_window) && m.context_window)
      seg += `/${ti(m.context_window)}`;
    parts.push(seg);
  }
  if (num(m.completion_tokens)) {
    let seg =
      `out ${ti(m.completion_tokens)} tok` + (m.tokens_estimated ? "≈" : "");
    if (m.reasoning_tokens != null || m.content_tokens != null) {
      const rt = num(m.reasoning_tokens) ? ti(m.reasoning_tokens) : 0;
      const ot = num(m.content_tokens) ? ti(m.content_tokens) : 0;
      seg += ` (think ${rt} / result ${ot})`;
    }
    parts.push(seg);
  }
  if (num(m.total_tokens) && m.total_tokens !== m.completion_tokens) {
    parts.push(`total ${ti(m.total_tokens)} tok`);
  }
  if (num(m.wt_ms)) parts.push(`wt ${f1(m.wt_ms / 1000)}s`);
  // Cost_usd is a real ordered segment of the canonical
  // per-turn row (metrics_format.py) — render it on Web too so no frontend
  // drops a segment the others show. Estimated turns carry the ≈ marker.
  if (num(m.cost_usd)) {
    let seg = `$${m.cost_usd.toFixed(4)}` + (m.tokens_estimated ? "≈" : "");
    if (num(m.goal_cost_usd)) {
      const gmax =
        num(m.goal_max_cost_usd) && m.goal_max_cost_usd > 0
          ? m.goal_max_cost_usd
          : null;
      seg +=
        gmax != null
          ? ` · goal $${m.goal_cost_usd.toFixed(2)}/$${gmax.toFixed(2)}`
          : ` · goal $${m.goal_cost_usd.toFixed(2)}`;
    } else if (num(m.session_cost_usd)) {
      seg += ` · session $${m.session_cost_usd.toFixed(2)}`;
    }
    parts.push(seg);
  }
  return parts;
}
function metricsLine(m) {
  const parts = metricsSegments(m);
  return parts.length ? "─ " + parts.join(" · ") : "";
}

// Build a folded ``<think>…</think>answer`` raw string from the block model.
// This is the cache / re-hydration fallback (VirtualScroller re-renders an
// off-screen bubble from its cached raw via ``renderAssistantContent``, and the
// final persisted history comes back from the gateway in this same folded
// shape) — so a turn that scrolls out mid-stream degrades gracefully to the
// reasoning-disclosure layout. The LIVE in-place view uses the block painter.
function foldedRawFromBlocks(bs) {
  let think = "";
  let toolLines = "";
  let answer = "";
  for (const blk of bs.visibleBlocks()) {
    if (blk.kind === "think") {
      const txt = WebBlockStream._text(blk).replace(/\n+$/, "");
      if (txt) think += txt + "\n";
    } else if (blk.kind === "tool") {
      toolLines += `🔧 ${blk.toolName}(${blk.args})\n`;
      if (blk.status === "ok") toolLines += "← OK\n";
      else if (blk.status === "failed") toolLines += "← failed\n";
    } else if (blk.kind === "todos") {
      const its = blk.todos || [];
      toolLines += `📝 tasks ${its.filter((i) => i.status === "completed").length}/${its.length} done\n`;
    } else if (blk.kind === "tasks") {
      const its = (blk.tasks || []).filter((t) => t.status !== "deleted");
      toolLines += `🗒️ harness tasks ${its.filter((i) => i.status === "completed").length}/${its.length} done\n`;
    } else if (blk.kind === "plan") {
      answer += (answer ? "\n\n" : "") + "📐 Plan\n\n" + (blk.plan || "");
    } else if (blk.kind === "notice") {
      // Notices fold back into the answer channel as their raw marker line so
      // copy-as-markdown and a later markdown replay keep the timeline (the
      // renderer lifts the marker into a card again on replay).
      const mark = blk.notice && blk.notice.raw ? blk.notice.raw : "";
      if (mark) answer += (answer ? "\n\n" : "") + mark;
    } else if (blk.kind === "answer") {
      if (blk.badge) answer += `_${blk.badge}_\n\n`;
      answer += WebBlockStream._text(blk);
    }
  }
  let raw = "";
  if (think || toolLines) {
    raw += "<think>\n" + think + toolLines + (bs.ended ? "</think>\n" : "");
  }
  raw += answer;
  return raw;
}

// Paint the live bubble as the ordered 💭/🔧/💬 block sequence from the model.
// Each block is a SEPARATE element inside the turn's bubble (the bubble is the
// turn container, directly following its user prompt — point 7). Idempotent:
// the whole ``.message-content`` is rebuilt from the model each event, so a
// block is never duplicated and the answer is always contiguous.
function _isAbortSentinelText(text) {
  return String(text || "").trim() === "_[aborted]_";
}

function _turnOutcomeFromBlockStream(bs) {
  if (!bs) return "";
  if (bs.aborted) return "aborted";
  if (bs.error) return "error";
  // turn_ended is authoritative. A tool that never got tool_result must
  // not keep the card / activity pill on RUNNING after the turn closed.
  if (bs.ended) return "ok";
  if ((bs.blocks || []).some((blk) => blk.kind === "tool" && blk.status === "running"))
    return "running";
  return "running";
}

function _turnOutcomeLabel(status) {
  if (status === "aborted") return "aborted";
  if (status === "error") return "error";
  if (status === "ok") return "succeeded";
  if (status === "running") return "running";
  return "";
}

function _turnOutcomePillClass(status) {
  if (status === "aborted") return "aborted-pill";
  if (status === "error") return "error-pill";
  if (status === "ok") return "ok-pill";
  if (status === "running") return "running-pill";
  return "";
}

function _stampTurnOutcome(el, status) {
  if (!el || !el.querySelector) return;
  el.classList.remove(
    "message-aborted", "message-error", "message-ok", "message-running",
  );
  const header = el.querySelector(".message-header");
  if (header) {
    header.querySelectorAll(
      ".ok-pill, .error-pill, .aborted-pill, .running-pill",
    ).forEach((node) => node.remove());
  }
  const label = _turnOutcomeLabel(status);
  const cls = _turnOutcomePillClass(status);
  if (!label || !cls) {
    if (typeof renderMessageTelemetry === "function") renderMessageTelemetry(el);
    return;
  }
  if (status === "aborted") el.classList.add("message-aborted");
  else if (status === "error") el.classList.add("message-error");
  else if (status === "ok") el.classList.add("message-ok");
  else if (status === "running") el.classList.add("message-running");
  if (el.dataset) el.dataset.turnStatus = status;
  const idx = state.msgCache.findIndex((m) => m && m.el === el);
  if (idx >= 0) {
    state.msgCache[idx].status = status;
    state.msgCache[idx].aborted = status === "aborted";
    if (status === "running") {
      // Live lane: keep an event-derived label source. A synthesized "now"
      // must never land in the cache — the telemetry renderer supplies the
      // live "now" at display time while the turn is still running, and a
      // stale synthesized value here would later masquerade as the turn's
      // real completion timestamp once the turn settles.
      if (!state.msgCache[idx].lastTs) {
        state.msgCache[idx].lastTs = state.liveEventTs || "";
      }
    } else if (!state.msgCache[idx].endedAt) {
      // Terminal stamp: the turn's REAL recorded completion only (event ts,
      // or an already event-derived lastTs). Never "now" — a replay/adopt/
      // seal that lacks an event ts leaves the field empty and the telemetry
      // renderer degrades (omits the time label) instead of lying about when
      // a completed turn finished.
      state.msgCache[idx].endedAt = state.liveEventTs
        || state.msgCache[idx].lastTs
        || "";
      if (state.msgCache[idx].endedAt) {
        state.msgCache[idx].lastTs = state.msgCache[idx].endedAt;
      }
    }
  }
  if (typeof renderMessageTelemetry === "function") renderMessageTelemetry(el);
}

function renderBlockTurn(t, shouldScroll = true) {
  if (!t || !t.el) return;
  if (
    state.isStreaming
    && state.liveTurn
    && t.el === state.liveTurn.el
    && _promoteNudgesFromLiveTurn(t)
  ) {
    return;
  }
  const bs = t.bs;
  // Keep the physical turn in the cache for SSE adoption and virtual scrolling,
  // while omitting an otherwise empty compaction control turn from the chat.
  const wasHidden = t.el.hidden;
  t.el.hidden = bs.isCompactionOnly();
  const contentEl = t.el.querySelector(".message-content");
  if (!contentEl) return;
  const idx = state.msgCache.findIndex((m) => m.el === t.el);
  const cachedActivityState =
    t.activityDisclosureState ||
    (idx >= 0 ? state.msgCache[idx].activityDisclosureState : null);
  // Render the streamed blocks first, then APPEND any abort/error notice so the
  // turn stays visible — a late error (e.g. context-overflow HTTP 400) must not
  // wipe the reasoning/tool/answer the model already produced.
  const painted = _paintBlockTurnHtml(bs);
  let html = painted.html;
  const outcome = _turnOutcomeFromBlockStream(bs);
  // Event-derived timestamp only. ``state.liveEventTs`` is refreshed on every
  // SSE frame (renderCanonicalEvent); outside an SSE dispatch (adopt/seal/
  // replay repaints) it is empty. A synthesized ``new Date()`` must NEVER land
  // in the cache here: once the turn settles it would masquerade as the real
  // completion timestamp (the "footer shows now, not completion time" bug).
  // While a turn is still RUNNING the telemetry renderer supplies the live
  // "now" at display time from its own ticker.
  const liveTs = state.liveEventTs || "";
  if (idx >= 0) {
    state.msgCache[idx].liveBs = bs;
    if (!state.msgCache[idx].physicalTurnId) state.msgCache[idx].physicalTurnId = state.livePhysicalTurnId || "";
    state.msgCache[idx].status = outcome;
    state.msgCache[idx].aborted = outcome === "aborted";
    if (liveTs) state.msgCache[idx].lastTs = liveTs;
    if (outcome === "running" && !bs.ended) {
      if (!state.msgCache[idx].timestamp) {
        state.msgCache[idx].timestamp = state.liveTurnTs || liveTs;
      }
    } else if (liveTs) {
      state.msgCache[idx].endedAt = liveTs;
      state.msgCache[idx].lastTs = liveTs;
    } else if (!state.msgCache[idx].endedAt && state.msgCache[idx].lastTs) {
      // No fresh event ts but the cache already holds an event-derived
      // lastTs (e.g. a seal repaint after the stream closed): freeze THAT
      // as the terminal stamp. Empty otherwise — the telemetry renderer
      // omits the time label rather than lying with the current time.
      state.msgCache[idx].endedAt = state.msgCache[idx].lastTs;
    }
  }
  _stampTurnOutcome(t.el, outcome);
  if (bs.error) {
    html += `<div class="block-error">⚠️ Error: ${escapeHtml(String(bs.error))}</div>`;
  }
  // Legacy standalone tool bodies render collapsed; keep their state if an old
  // saved turn still paints through that path.
  const openTools = new Set(
    Array.from(contentEl.querySelectorAll("details.tc-collapsible[open]"))
      .filter((d) => !d.classList.contains("activity-group-item"))
      .map((d) => d.getAttribute("data-call-id"))
      .filter(Boolean),
  );
  // Same for an operator-expanded runtime-notice card: repaints must not
  // collapse an explanation being read while the agent keeps streaming.
  const openNoticeKeys = new Set(
    Array.from(contentEl.querySelectorAll("details.block-notice[open]"))
      .map((d) => d.getAttribute("data-notice-key"))
      .filter(Boolean),
  );
  const openCompactionKeys = new Set(
    Array.from(contentEl.querySelectorAll(
      ".message-compaction-stats[data-notice-key]",
    ))
      .filter((card) => {
        const details = card.querySelector(".mcs-details");
        return details && !details.hasAttribute("hidden");
      })
      .map((card) => card.getAttribute("data-notice-key"))
      .filter(Boolean),
  );
  // Capture before the innerHTML wipe — open activity payloads are rebuilt
  // every stream tick, and without this the .reason-thought / tool-code
  // panes jump back to scrollTop=0 while the operator is reading them.
  const domActivityState = _readActivityStateFromDom(contentEl);
  const activityState = _normalizeActivityState(
    domActivityState || cachedActivityState,
    painted.runs,
  );
  const rememberActivityState = (nextState) => {
    t.activityDisclosureState = nextState;
    if (idx >= 0) state.msgCache[idx].activityDisclosureState = nextState;
  };
  rememberActivityState(activityState);
  contentEl.innerHTML = html;
  if (openTools.size) {
    contentEl.querySelectorAll("details.tc-collapsible").forEach((d) => {
      if (openTools.has(d.getAttribute("data-call-id"))) d.open = true;
    });
  }
  if (openNoticeKeys.size) {
    contentEl.querySelectorAll("details.block-notice").forEach((d) => {
      if (openNoticeKeys.has(d.getAttribute("data-notice-key"))) d.open = true;
    });
  }
  if (openCompactionKeys.size) {
    contentEl.querySelectorAll(
      ".message-compaction-stats[data-notice-key]",
    ).forEach((card) => {
      if (!openCompactionKeys.has(card.getAttribute("data-notice-key"))) return;
      const details = card.querySelector(".mcs-details");
      const toggle = card.querySelector(".mcs-toggle");
      if (details) details.removeAttribute("hidden");
      if (toggle) toggle.setAttribute("aria-expanded", "true");
    });
  }
  _bindActivityGroupLazy(
    contentEl,
    painted.runs,
    activityState,
    rememberActivityState,
  );
  applyVisExpand(contentEl);
  renderMessageTelemetry(t.el);
  // Keep the cache + dataset in the folded shape so an off-screen re-hydration
  // (or copy-as-markdown) still resolves the answer correctly.
  const raw = foldedRawFromBlocks(bs);
  t.el.dataset.raw = raw;
  t.el.classList.toggle("has-content", !!(raw && raw.trim()));
  if (idx >= 0) {
    VirtualScroller.update(idx, raw);
    if (wasHidden || t.el.hidden) VirtualScroller.measure(idx, t.el);
  }
  if (typeof _mountAskUserFromBlockStream === "function") {
    _mountAskUserFromBlockStream(t);
  }
  if (shouldScroll) scrollToBottom();
}

function _scrollReasoningToLatest(el) {
  if (!el) return;
  el.scrollTop = el.scrollHeight;
}

function _visExpandOn(kind) {
  return !!(
    typeof state !== "undefined" &&
    state.visExpand &&
    state.visExpand[kind]
  );
}

/** Gear toggles: off = cards stay collapsed; on = expand matching cards.
 *  Running/thinking must not override Off. Reasoning-on also pins each
 *  .reason-thought pane to the last tokens. */
function applyVisExpand(root, opts) {
  if (typeof state === "undefined" || !state.visExpand) return;
  const host = root || document.getElementById("messages");
  if (!host) return;
  const wantA = !!state.visExpand.activity;
  const wantR = !!state.visExpand.reasoning;
  const wantT = !!state.visExpand.tools;
  const force = !!(opts && opts.force);
  host.querySelectorAll("details.activity-group").forEach((group) => {
    const title =
      ((group.querySelector(".tool-group-title") || {}).textContent || "");
    const hasR =
      group.getAttribute("data-thinking") === "1" ||
      /reasoning/i.test(title) ||
      !!group.querySelector(".reasoning-group-call");
    const hasT =
      Number(group.getAttribute("data-tool-count") || 0) > 0 ||
      !!group.querySelector(".tool-group-call:not(.reasoning-group-call)");
    const wantGroup = wantA || (wantR && hasR) || (wantT && hasT);
    if (wantGroup) {
      if (!group.open) {
        group.open = true;
        if (typeof group.dispatchEvent === "function") {
          group.dispatchEvent(new Event("toggle"));
        }
      }
    } else if (force && group.open) {
      group.open = false;
      if (typeof group.dispatchEvent === "function") {
        group.dispatchEvent(new Event("toggle"));
      }
    }
    group.querySelectorAll("details.activity-group-item").forEach((item) => {
      if (item.getAttribute("data-lazy") === "1" || item.dataset.lazy === "1") {
        return;
      }
      const isR = item.classList.contains("reasoning-group-call");
      const next = isR ? wantR : wantT;
      if (next) {
        if (!item.open) {
          item.open = true;
          if (typeof item.dispatchEvent === "function") {
            item.dispatchEvent(new Event("toggle"));
          }
        }
      } else if (force && item.open) {
        item.open = false;
        if (typeof item.dispatchEvent === "function") {
          item.dispatchEvent(new Event("toggle"));
        }
      }
    });
  });
  host.querySelectorAll("details.reasoning, details.block-think").forEach((d) => {
    if (d.getAttribute("data-lazy") === "1" || d.dataset.lazy === "1") return;
    if (wantR) {
      if (!d.open) d.open = true;
    } else if (force && d.open) {
      d.open = false;
    }
  });
  host.querySelectorAll(
    "details.block-tool.tc-collapsible:not(.activity-group):not(.activity-group-item):not(.block-delegate)",
  ).forEach((d) => {
    if (d.getAttribute("data-lazy") === "1" || d.dataset.lazy === "1") return;
    if (wantT) {
      if (!d.open) d.open = true;
    } else if (force && d.open) {
      d.open = false;
    }
  });
  if (wantR) {
    const pin = () => {
      host.querySelectorAll(".reason-thought").forEach(_scrollReasoningToLatest);
      if (typeof scrollToBottom === "function") scrollToBottom();
    };
    pin();
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(pin);
  }
}

function _toolStatusIcon(status) {
  if (status === "ok") return "✓";
  if (status === "failed") return "✗";
  if (status === "aborted") return "⏹";
  if (status === "cancelled") return "⏹";
  if (status === "refused") return "⊘";
  return '<span class="tc-spin"></span>';
}

function _toolStatusLabel(status) {
  if (status === "ok") return "Success";
  if (status === "failed") return "Failed";
  if (status === "aborted") return "Aborted";
  if (status === "cancelled") return "Stopped";
  if (status === "refused") return "Refused";
  return "Running";
}

function _activityKey(blk) {
  return blk.kind === "think"
    ? `reason-${blk.uid}`
    : `tool-${blk.callId || "blk" + blk.uid}`;
}

function _isDelegateTool(blk) {
  return !!(blk && blk.kind === "tool" && blk.delegate);
}

function _isActivityBlock(blk) {
  if (!blk) return false;
  if (blk.kind === "think") {
    return !!WebBlockStream._text(blk).trim() || !!blk.lazy;
  }
  if (blk.kind === "tool") {
    return !_isDelegateTool(blk) && !_isAskUserTool(blk.toolName);
  }
  return false;
}

function _activityRunId(activity) {
  return activity && activity[0] ? _activityKey(activity[0]) : "activity";
}

// Consecutive reasoning + ordinary tools collapse into one Activity card.
// An answer, widget, or delegation flushes the run so later activity stays
// after that output — OpenCode's groupParts() does the same with context
// tools vs text/task parts.
function _partitionVisibleBlocks(visible) {
  const runs = [];
  let activity = [];
  const flush = () => {
    if (!activity.length) return;
    runs.push({ type: "activity", blocks: activity });
    activity = [];
  };
  for (const blk of Array.isArray(visible) ? visible : []) {
    if (
      blk
      && blk.kind === "answer"
      && _isAbortSentinelText(WebBlockStream._text(blk))
    ) {
      continue;
    }
    if (_isActivityBlock(blk)) {
      activity.push(blk);
      continue;
    }
    flush();
    if (_isDelegateTool(blk)) runs.push({ type: "delegate", block: blk });
    else runs.push({ type: "block", block: blk });
  }
  flush();
  return runs;
}

function _paintBlockTurnHtml(bs) {
  if (typeof state !== "undefined" && state.delegationSnapshot &&
      state.delegationSnapshot.sessionId === state.currentSessionId) {
    for (const event of state.delegationSnapshot.jobs) bs.delegationUpdate(event);
  }
  const visible = bs.visibleBlocks();
  const runs = _partitionVisibleBlocks(visible);
  let html = "";
  for (let i = 0; i < runs.length; i++) {
    const run = runs[i];
    if (run.type === "activity") {
      html += _activityGroupHtml(run.blocks, bs, undefined, {
        trailing: i === runs.length - 1,
      });
    } else if (run.type === "delegate") html += _delegateCardHtml(run.block, bs);
    else html += _blockHtml(run.block, bs);
  }
  return { html, runs, visible };
}

// Shell status for one Activity card. Finished tools already flip to ok while
// the model is still writing; reasoning-only cards used to stay "running"
// until turn_ended, so a flushed earlier card (answer/tools after it) kept
// the spinner even though that trace was done.
function _activityThinkLive(activity) {
  return (activity || []).some((blk) => blk && blk.kind === "think" && !blk.closed);
}

function _activityAggregateStatus(activity, bs, opts) {
  const tools = (activity || []).filter((blk) => blk.kind === "tool");
  if (tools.some((blk) => blk.status === "failed")) return "failed";
  if (tools.some((blk) => blk.status === "running")) {
    if (bs && bs.ended) return (bs.aborted || bs.error) ? "failed" : "ok";
    return "running";
  }
  const trailing = !opts || opts.trailing !== false;
  // An unclosed 💭 is still working even when an answer after it made this
  // run non-trailing. Trailing-only used to paint ✓ on the collapsed shell
  // while expand still said "Reasoning · running".
  if (_activityThinkLive(activity) && !(bs && bs.ended)) {
    return "running";
  }
  if (tools.length) return "ok";
  if (!trailing || (bs && bs.ended)) return "ok";
  const thinks = (activity || []).filter((blk) => blk.kind === "think");
  if (thinks.length && thinks.every((blk) => blk.closed)) return "ok";
  return "running";
}

// One disclosure per consecutive no-output activity run. Reasoning and tool
// blocks stay independent in the model (preserving their chronological audit
// data), while the initial DOM contains only this compact count/status shell.
function _activityGroupHtml(activity, bs, runId, opts) {
  if (!Array.isArray(activity) || !activity.length) return "";
  const tools = activity.filter((blk) => blk.kind === "tool");
  const reasoningCount = activity.length - tools.length;
  const toolCount = tools.length;
  // Aggregate reflects finished work, not "the turn is still open". Do NOT
  // keep the shell on "running" merely because the model is still writing an
  // answer after this run flushed — that made an earlier reasoning card look
  // stuck on Working next to a later completed card (and forever if
  // turn_ended was missed). The footer turn indicator covers "model still
  // writing".
  const aggregate = _activityAggregateStatus(activity, bs, opts);
  const thinkLive = _activityThinkLive(activity) && aggregate === "running";
  const parts = [];
  if (thinkLive) parts.push("Reasoning · running");
  else if (reasoningCount)
    parts.push(
      `${reasoningCount} reasoning trace${reasoningCount === 1 ? "" : "s"}`,
    );
  if (toolCount)
    parts.push(`${toolCount} tool call${toolCount === 1 ? "" : "s"}`);
  const label = parts.join(" · ");
  const statsBlock = [...activity].reverse().find((blk) => blk.stats);
  const id = runId || _activityRunId(activity);
  return (
    `<details class="block-tool toolcall toolcall-${aggregate} tool-group activity-group"` +
    (thinkLive ? ` data-thinking="1"` : "") +
    ` data-activity-run="${escapeAttr(id)}"` +
    ` data-activity-count="${activity.length}" data-tool-count="${toolCount}">` +
    `<summary>` +
    `<span class="tc-status">${_toolStatusIcon(aggregate)}</span>` +
    `<span class="tool-group-title">Activity · ${label}</span>` +
    `<span class="tc-caret">▶</span>` +
    `</summary>` +
    `<div class="tool-group-body activity-group-body" data-loaded="0"></div>` +
    `</details>` +
    (statsBlock
      ? `<div class="assist-metrics block-metrics">${escapeHtml(
          statsBlock.stats,
        )}</div>`
      : "")
  );
}

function _activityGroupItemsHtml(activity) {
  return activity
    .map((blk) => {
      const key = _activityKey(blk);
      if (blk.kind === "think") {
        const text = WebBlockStream._text(blk).replace(/\s+/g, " ").trim();
        const thinkLive = !blk.closed;
        return (
          `<details class="activity-group-item tool-group-call reasoning-group-call tc-collapsible` +
          `${thinkLive ? " toolcall-running" : ""}"` +
          `${thinkLive ? ` data-thinking="1"` : ""}` +
          ` data-activity-key="${escapeAttr(key)}" data-lazy="${
            blk.lazy ? "1" : "0"
          }">` +
          `<summary>` +
          `<span class="tc-status activity-kind-icon">${
            thinkLive ? '<span class="tc-spin"></span>' : "💭"
          }</span>` +
          `<span class="tc-name">${
            thinkLive ? "Reasoning · running" : "Reasoning"
          }</span>` +
          (text
            ? `<span class="tc-args">${escapeHtml(
                toolArgsPreview(text),
              )}</span>`
            : "") +
          `<span class="tc-caret">▶</span>` +
          `</summary>` +
          `<div class="tool-group-call-body activity-group-item-body" data-loaded="0"></div>` +
          `</details>`
        );
      }
      const args = blk.args || "";
      return (
        `<details class="activity-group-item tool-group-call toolcall toolcall-${escapeAttr(
          blk.status,
        )} tc-collapsible" data-activity-key="${escapeAttr(key)}" data-lazy="${
          blk.lazy ? "1" : "0"
        }">` +
        `<summary>` +
        `<span class="tc-status">${_toolStatusIcon(blk.status)}</span>` +
        `<span class="tc-name" title="${escapeAttr(
          blk.toolName,
        )}">${escapeHtml(blk.toolName)}</span>` +
        (args
          ? `<span class="tc-args">${escapeHtml(
              toolArgsPreview(args),
            )}</span>`
          : "") +
        `<span class="tc-caret">▶</span>` +
        `</summary>` +
        `<div class="tool-group-call-body activity-group-item-body" data-loaded="0"></div>` +
        `</details>`
      );
    })
    .join("");
}

function _activityGroupPayloadHtml(blk) {
  if (blk.kind === "think") {
    return (
      `<div class="activity-reasoning-body">` +
      `<div class="tool-group-label">Reasoning trace</div>` +
      `<div class="reason-thought">${escapeHtml(
        WebBlockStream._text(blk),
      )}</div>` +
      `</div>`
    );
  }
  return _toolGroupPayloadHtml(blk);
}

// Near-bottom latch for scrollable activity panes. When the operator is
// following a live stream at the bottom we keep them pinned there after a
// repaint; otherwise we restore their exact scrollTop.
const _ACTIVITY_SCROLL_BOTTOM_PX = 24;

function _scrollPaneState(el) {
  if (!el) return null;
  return {
    top: el.scrollTop,
    stickBottom:
      el.scrollHeight - el.scrollTop - el.clientHeight <
      _ACTIVITY_SCROLL_BOTTOM_PX,
  };
}

function _applyScrollPaneState(el, pane) {
  if (!el || !pane) return;
  el.scrollTop = pane.stickBottom ? el.scrollHeight : pane.top;
}

function _captureActivityItemScroll(root) {
  const out = new Map();
  if (!root) return out;
  root.querySelectorAll("details.activity-group-item[open]").forEach((item) => {
    const key = item.dataset.activityKey;
    if (!key) return;
    const panes = {};
    const thought = item.querySelector(".reason-thought");
    const thoughtState = _scrollPaneState(thought);
    if (thoughtState) panes.thought = thoughtState;
    item.querySelectorAll(".tool-group-code").forEach((pre, i) => {
      const codeState = _scrollPaneState(pre);
      if (codeState) panes["code" + i] = codeState;
    });
    if (Object.keys(panes).length) out.set(key, panes);
  });
  return out;
}

function _restoreActivityItemScroll(item, panes) {
  if (!item || !panes) return;
  _applyScrollPaneState(item.querySelector(".reason-thought"), panes.thought);
  item.querySelectorAll(".tool-group-code").forEach((pre, i) => {
    _applyScrollPaneState(pre, panes["code" + i]);
  });
}

function _mountActivityGroupItems(
  group,
  activity,
  openItemIds,
  onStateChange,
  itemScroll,
) {
  const body = group.querySelector(".activity-group-body");
  if (!body || body.dataset.loaded === "1") return;
  body.innerHTML = _activityGroupItemsHtml(activity);
  body.dataset.loaded = "1";
  const byKey = new Map(activity.map((blk) => [_activityKey(blk), blk]));
  const scrollMap = itemScroll instanceof Map ? itemScroll : new Map();
  const remember = () => {
    if (!onStateChange) return;
    onStateChange({
      open: group.open,
      openItemIds: new Set(
        Array.from(
          group.querySelectorAll("details.activity-group-item[open]"),
        )
          .map((item) => item.dataset.activityKey)
          .filter(Boolean),
      ),
      itemScroll: _captureActivityItemScroll(group),
    });
  };
  body.querySelectorAll("details.activity-group-item").forEach((item) => {
    const key = item.dataset.activityKey || "";
    const blk = byKey.get(key);
    const mountPayload = async () => {
      if (!item.open || !blk) {
        remember();
        return;
      }
      const itemBody = item.querySelector(".activity-group-item-body");
      if (
        item.dataset.lazy === "1" &&
        itemBody &&
        itemBody.dataset.loaded !== "1" &&
        typeof apiGet === "function" &&
        state.currentSessionId
      ) {
        try {
          const fetchId =
            blk.kind === "think"
              ? (blk.thinkingCardId || key)
              : (blk.cardId || blk.callId || String(key).replace(/^tool-/, ""));
          const data = await apiGet(
            `/api/sessions/${encodeURIComponent(state.currentSessionId)}` +
              `/cards/${encodeURIComponent(fetchId)}`,
          );
          const seg = data && data.segment;
          if (seg) {
            if (blk.kind === "think") {
              if (seg.content != null) {
                blk.buf = [String(seg.content)];
                blk.closed = true;
              }
            } else {
              if (seg.content != null) {
                blk.resultText = String(seg.content);
                blk.buf = [String(seg.content)];
              }
              if (seg.arguments != null && typeof _blockToolArgs === "function") {
                blk.args = _blockToolArgs(seg.arguments);
              }
              blk.status =
                seg.ok === false ? "failed" : seg.ok ? "ok" : blk.status;
            }
          }
          item.dataset.lazy = "0";
          blk.lazy = false;
        } catch (_e) {
          if (itemBody) itemBody.textContent = "Could not load this card.";
          remember();
          return;
        }
      }
      const sig =
        String(blk.args || "") +
        "\0" +
        WebBlockStream._text(blk) +
        "\0" +
        String(blk.status || "");
      if (itemBody && itemBody.dataset.sig !== sig) {
        itemBody.innerHTML = _activityGroupPayloadHtml(blk);
        itemBody.dataset.loaded = "1";
        itemBody.dataset.sig = sig;
        const panes = scrollMap.get(key);
        _restoreActivityItemScroll(item, panes);
        if (panes && typeof requestAnimationFrame === "function") {
          requestAnimationFrame(() =>
            _restoreActivityItemScroll(item, panes),
          );
        }
      }
      remember();
    };
    item.addEventListener("toggle", mountPayload);
    const liveTurn = typeof state !== "undefined" && !!state.isStreaming;
    const live =
      (blk && blk.kind === "think" && !blk.closed) ||
      (blk && blk.status === "running") ||
      (liveTurn &&
        blk &&
        blk.kind === "tool" &&
        !!WebBlockStream._text(blk).trim());
    const wantLiveOpen =
      live &&
      ((blk && blk.kind === "think" && _visExpandOn("reasoning")) ||
        (blk && blk.kind !== "think" && _visExpandOn("tools")));
    if ((openItemIds && openItemIds.has(key)) || wantLiveOpen) {
      item.open = true;
      mountPayload();
    }
  });
}

function _readActivityStateFromDom(contentEl) {
  if (!contentEl) return null;
  const groups = contentEl.querySelectorAll("details.activity-group");
  if (!groups.length) return null;
  const byId = {};
  groups.forEach((group) => {
    const id = group.getAttribute("data-activity-run") || "0";
    byId[id] = {
      open: group.open,
      openItemIds: new Set(
        Array.from(
          group.querySelectorAll("details.activity-group-item[open]"),
        )
          .map((d) => d.getAttribute("data-activity-key"))
          .filter(Boolean),
      ),
      itemScroll: _captureActivityItemScroll(group),
    };
  });
  return { byId };
}

function _normalizeActivityState(raw, runs) {
  if (raw && raw.byId) return raw;
  const first = (runs || []).find((r) => r.type === "activity");
  if (raw && (raw.open != null || raw.openItemIds) && first) {
    return { byId: { [_activityRunId(first.blocks)]: raw } };
  }
  return { byId: {} };
}

function _bindActivityGroupLazy(
  contentEl,
  runs,
  priorState,
  onStateChange,
) {
  if (!contentEl) return;
  const state = _normalizeActivityState(priorState, runs);
  const byId = Object.assign({}, state.byId);
  contentEl.querySelectorAll("details.activity-group").forEach((group) => {
    const id = group.getAttribute("data-activity-run") || "0";
    const run = (runs || []).find(
      (r) => r.type === "activity" && _activityRunId(r.blocks) === id,
    );
    if (!run) return;
    _bindOneActivityGroup(group, run.blocks, byId[id], (next) => {
      byId[id] = next;
      if (onStateChange) onStateChange({ byId: Object.assign({}, byId) });
    });
  });
}

function _activityIsLive(activity) {
  return (activity || []).some((blk) =>
    (blk.kind === "think" && !blk.closed) ||
    (blk.kind === "tool" && blk.status === "running" && !blk.closed),
  );
}

function _activityWantsExpand(activity) {
  const hasR = (activity || []).some((blk) => blk && blk.kind === "think");
  const hasT = (activity || []).some((blk) => blk && blk.kind === "tool");
  return (
    _visExpandOn("activity") ||
    (_visExpandOn("reasoning") && hasR) ||
    (_visExpandOn("tools") && hasT)
  );
}

function _bindOneActivityGroup(group, activity, priorState, onStateChange) {
  if (!group) return;
  if (
    (priorState && priorState.open) ||
    (priorState == null && _activityWantsExpand(activity)) ||
    (_activityIsLive(activity) && _activityWantsExpand(activity))
  )
    group.open = true;
  const mountItems = () => {
    if (group.open)
      _mountActivityGroupItems(
        group,
        activity,
        priorState && priorState.openItemIds
          ? priorState.openItemIds
          : null,
        onStateChange,
        priorState && priorState.itemScroll
          ? priorState.itemScroll
          : null,
      );
    if (onStateChange) {
      const mountedOpenItems = new Set(
        Array.from(
          group.querySelectorAll("details.activity-group-item[open]"),
        )
          .map((item) => item.dataset.activityKey)
          .filter(Boolean),
      );
      const captured = _captureActivityItemScroll(group);
      onStateChange({
        open: group.open,
        openItemIds:
          mountedOpenItems.size || group.querySelector(".activity-group-item")
            ? mountedOpenItems
            : new Set(
                priorState && priorState.openItemIds
                  ? priorState.openItemIds
                  : [],
              ),
        itemScroll: captured.size
          ? captured
          : (priorState && priorState.itemScroll) || new Map(),
      });
    }
  };
  group.addEventListener("toggle", mountItems);
  mountItems();
}

function _toolGroupPayloadHtml(blk) {
  const args = blk.args || "";
  const output = WebBlockStream._text(blk).replace(/^\n+/, "");
  return (
    `<div class="tool-group-call-head">` +
    `<span class="tool-group-name">${escapeHtml(blk.toolName)}</span>` +
    `<span class="tool-group-state">${escapeHtml(
      _toolStatusLabel(blk.status),
    )}</span>` +
    `</div>` +
    `<div class="tool-group-section">` +
    `<div class="tool-group-label">Call</div>` +
    `<pre class="tool-group-code tool-group-args">${escapeHtml(
      args ||
        (blk.status === "running"
          ? "Waiting for arguments…"
          : "No arguments."),
    )}</pre></div>` +
    `<div class="tool-group-section">` +
    `<div class="tool-group-label">Output</div>` +
    `<pre class="tool-group-code tool-group-output">${escapeHtml(
      output ||
        (blk.status === "running"
          ? "Waiting for output…"
          : "No output returned."),
    )}</pre></div>`
  );
}

function _delegateCardHtml(blk, bs) {
  if (!blk) return "";
  _bindDelegationCardLinks();
  const waiting = blk.status === "running";
  const key = blk.callId || "blk" + blk.uid;
  const children = blk.delegateChildren || [];
  const mode = _delegateModeLabel(blk);
  const modeKey = mode.toLowerCase();
  const modeText = mode || String(blk.toolName || "Delegation");
  const modeTitle = mode
    ? `${mode} delegation (${String(blk.toolName || "tool")})`
    : String(blk.toolName || "Delegation");
  const rows = children.map((child) => {
    const running = child.status === "running";
    const label = running ? "Running"
      : child.status === "failed" ? "Failed"
      : child.status === "aborted" || child.status === "cancelled" ? "Stopped"
      : child.status === "refused" ? "Refused"
      : "Completed";
    const icon = running ? '<span class="tc-spin" aria-hidden="true"></span>'
      : _toolStatusIcon(child.status);
    const description = child.description || "Delegated task";
    const href = (typeof BASE === "string" ? BASE : "") + "/?session=" + encodeURIComponent(child.taskId);
    const tag = child.taskId ? "a" : "span";
    const sessionAttrs = child.taskId
      ? ` href="${escapeAttr(href)}" data-delegate-session="${escapeAttr(child.taskId)}"` +
        ` aria-label="${escapeAttr(child.agent)} subagent session: ${escapeAttr(description)}"`
      : "";
    return `<${tag} class="delegate-child" data-status="${escapeAttr(child.status)}"${sessionAttrs}>` +
      `<span class="tc-status" aria-hidden="true">${icon}</span>` +
      `<span class="delegate-identity"><span class="tc-name">${escapeHtml(child.agent)}</span>` +
      `<span class="delegate-description" title="${escapeAttr(description)}">${escapeHtml(description)}</span></span>` +
      `<span class="delegate-verb" role="status">${label}</span>` +
      `</${tag}>`;
  }).join("");
  const head =
    `<span class="delegate-summary"><span class="delegate-heading">` +
    `<span class="delegate-glyph" aria-hidden="true">↳</span>` +
    `${children.length > 1 ? `${children.length} subagents` : "Subagent"}` +
    `<span class="delegate-mode" data-mode="${escapeAttr(modeKey || "unknown")}"` +
    ` title="${escapeAttr(modeTitle)}">${escapeHtml(modeText)}</span></span>` +
    `<span class="delegate-children">${rows}</span></span>` +
    `<span class="tc-caret" aria-hidden="true">▶</span>`;
  // Always emit Call/Output so the card expands in /tools on as well as stream.
  // An empty body made <details> look like a dead row.
  const body =
    `<div class="block-tool-body delegate-card-body">${_toolGroupPayloadHtml(blk)}</div>` +
    (waiting
      ? `<div class="delegate-hint">Waiting for the subagent response.</div>`
      : "");
  let html =
    `<details class="block-tool block-delegate toolcall toolcall-${escapeAttr(
      blk.status,
    )} tc-collapsible" data-call-id="${escapeAttr(key)}" data-delegate="1">` +
    `<summary>${head}</summary>` +
    body +
    `</details>`;
  if (blk.stats)
    html += `<div class="assist-metrics block-metrics">${escapeHtml(blk.stats)}</div>`;
  return html;
}

function _bindDelegationCardLinks() {
  if (typeof document === "undefined" || document._delegationCardLinks) return;
  document._delegationCardLinks = true;
  document.addEventListener("click", (event) => {
    const link = event.target.closest("a.delegate-child[data-delegate-session]");
    if (!link || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    if (typeof window.saOpenConversation !== "function") return;
    event.preventDefault();
    event.stopPropagation();
    window.saOpenConversation(link.dataset.delegateSession, {
      fromMonitor: true,
      scopeHash: state.currentSessionId || "",
    });
  });
}

function _applyDelegationUpdates(events) {
  const seen = new Set();
  const turns = (state.msgCache || []).filter((m) => m.role === "assistant").map((m) => {
    if (!m.liveBs && (m.segments || []).some((s) => s.kind === "tool" && _isDelegateSpawn(s.name, s.arguments))) {
      m.liveBs = blockStreamFromSavedSegments(m.segments, m.content, state.vis, {
        aborted: m.aborted, error: m.error, inProgress: m.status === "running",
      });
    }
    return { el: m.el, bs: m.liveBs };
  });
  if (state.liveTurn) turns.push(state.liveTurn);
  for (const turn of turns) {
    const bs = turn.bs;
    if (!bs || seen.has(bs)) continue;
    seen.add(bs);
    let changed = false;
    for (const event of events) changed = bs.delegationUpdate(event) || changed;
    if (!changed || !turn.el) continue;
    // Update just the tool cards. Repainting the parent turn would also
    // restamp its completion time and footer when a child returns later.
    for (const card of turn.el.querySelectorAll("details.block-delegate")) {
      const blk = bs._tools[card.dataset.callId];
      if (!blk) continue;
      const template = document.createElement("template");
      template.innerHTML = _delegateCardHtml(blk, bs);
      const next = template.content.firstElementChild;
      next.open = card.open;
      card.replaceWith(next);
    }
  }
}

// ── Runtime-notice card content ────────────────────────────────────────────
// The collapsed summary row carries the glyph, the event name and the k=v
// fields the engine emitted (kind/reason/wall/attempt for nudges — the
// "reasons"); the expandable body explains, in plain language, why the runtime
// nudged the agent, then shows the raw line verbatim so nothing is lost.
// Reason texts mirror sharedagents_agent/continuation.py (kind=timeout →
// MAX_TIMEOUT_CONTINUES=3; kind=silent → MAX_IDLE_EMPTY_CONTINUES=3) and
// sharedagents_agent/loop_detection.py event semantics.
const _NUDGE_KIND_CAPS = {
  timeout: 3,
  silent: 3,
};

const _NUDGE_KIND_CONTEXT = {
  timeout:
    "The agent's CLI round exceeded its allowed wall time, so the engine "
    + "resumed the session from the last checkpoint.",
  silent:
    "The round ended without delivering user-visible output, so the engine "
    + "injected a continuation nudge instead of ending the turn silently.",
};

const _NUDGE_REASON_EXPLAINERS = {
  print_timeout:
    "The agent's round hit the print-mode wall: it ran for the configured "
    + "wall time (wall=) without printing a checkpoint, so the CLI timed it "
    + "out. The engine resumed the session from the last completed checkpoint "
    + "instead of letting the turn die silently or restarting the whole task.",
  reasoning_only_no_output:
    "The agent replied with reasoning only and zero visible output — nothing "
    + "reached you. The engine re-prompted it to write the answer directly "
    + "or call a tool.",
  reasoning_stall:
    "The agent stopped inside a reasoning block without returning a verdict "
    + "(its thought stream went idle). The engine re-prompted it to finish "
    + "with a visible answer.",
  empty_after_tools:
    "The agent ran one or more tool calls and then stopped without writing "
    + "anything visible. The engine re-prompted it to summarize the outcome "
    + "or keep working.",
  empty_no_output:
    "The agent's response was completely empty — no text, no reasoning, no "
    + "tool call. The engine re-prompted it to answer or make progress.",
  thinking_starved_length:
    "The round ran out of output tokens while still reasoning "
    + "(finish_reason=length) and delivered nothing visible. The engine "
    + "re-prompted it to write the result.",
};

const _RUNTIME_EVENT_TITLES = {
  nudge: "Nudge",
  "progress-nudge": "Progress nudge",
  "loop-detected": "Loop detected",
  "loop-distill": "Loop distill",
  "loop-force-terminate": "Loop force-terminate",
  distill: "Distill",
  "upstream-retry": "Upstream retry",
  "force-terminate": "Force terminate",
  "read-only-synthesize": "Read-only synthesize",
};

const _RUNTIME_EVENT_EXPLAINERS = {
  "progress-nudge":
    "The agent made no visible progress for several consecutive rounds; the "
    + "engine re-grounded it on the task before the turn could stall out.",
  "loop-detected":
    "The engine detected the agent repeating itself and intervened to break "
    + "the loop.",
  "loop-distill":
    "Repetitive context was distilled (compacted) so the agent could "
    + "continue on a fresh footing.",
  "loop-force-terminate":
    "A stuck repetition loop was force-terminated so the turn could recover "
    + "instead of hanging.",
  distill:
    "The engine distilled mid-turn context to keep the session inside its "
    + "window.",
  "upstream-retry":
    "An upstream request failed transiently and was retried automatically.",
  "force-terminate":
    "A stalled round was force-terminated and the turn resumed.",
  "read-only-synthesize":
    "The engine asked the agent to synthesize what it found instead of "
    + "taking further actions.",
};

function _runtimeNoticeTitle(parsed) {
  const p = parsed || {};
  const ev = String(p.event || "");
  if (ev && _RUNTIME_EVENT_TITLES[ev]) return _RUNTIME_EVENT_TITLES[ev];
  if (!ev) return "Runtime notice";
  return ev.charAt(0).toUpperCase() + ev.slice(1).replace(/-/g, " ");
}

function _noticeChipHtml(k, v) {
  const label = v === true ? k : `${k}=${v}`;
  return (
    `<span class="notice-chip${k === "reason" ? " notice-chip-reason" : ""}"` +
    ` title="${escapeAttr(label)}">${escapeHtml(label)}</span>`
  );
}

function _nudgeExplanationHtml(parsed) {
  const f = (parsed && parsed.fields) || {};
  const kind = String(f.kind || "");
  const reason = String(f.reason || "");
  const wall = String(f.wall || "");
  const attempt = String(f.attempt || "");
  let why;
  if (reason && _NUDGE_REASON_EXPLAINERS[reason]) {
    why = _NUDGE_REASON_EXPLAINERS[reason];
  } else if (kind && _NUDGE_KIND_CONTEXT[kind]) {
    why = _NUDGE_KIND_CONTEXT[kind]
      + " The fields above record what the engine observed.";
  } else {
    why =
      "The previous round stalled or produced no user-visible output, so the "
      + "engine injected a continuation nudge instead of letting the turn "
      + "end silently.";
  }
  let html = `<div class="notice-label">Why this nudge fired</div>`;
  html += `<div class="notice-why">${escapeHtml(why)}</div>`;
  const facts = [];
  if (wall) facts.push(`wall time: ${wall}`);
  if (attempt) {
    const cap = kind && _NUDGE_KIND_CAPS[kind] ? _NUDGE_KIND_CAPS[kind] : 0;
    facts.push(
      cap
        ? `nudge attempt ${attempt} of up to ${cap}`
        : `nudge attempt ${attempt} of the turn`,
    );
  }
  if (facts.length) {
    html += `<div class="notice-why notice-facts">${escapeHtml(
      facts.join("; ") + ".",
    )}</div>`;
  }
  return html;
}

function _genericNoticeExplanationHtml(parsed) {
  const p = parsed || {};
  const ev = String(p.event || "");
  const why = _RUNTIME_EVENT_EXPLAINERS[ev]
    || "The runtime intervened automatically and kept the turn going. The "
      + "fields above describe the event; full detail lives in the gateway "
      + "log.";
  return `<div class="notice-label">Why</div>` +
    `<div class="notice-why">${escapeHtml(why)}</div>`;
}

// The runtime-notice card: one <details> whose summary shows glyph + event +
// reason chips, and whose disclosure holds the plain-language explanation and
// the raw notice line. ``uid`` (block uid) keys the open state across live
// repaints; history rendering (no repaints) can omit it.
function runtimeNoticeCardHtml(parsed, uid) {
  if (!parsed || !parsed.event) return "";
  const soft = parsed.glyph === "↻";
  const f = parsed.fields || {};
  const chips = Object.keys(f)
    .map((k) => _noticeChipHtml(k, f[k]))
    .join("");
  const body = parsed.event === "nudge"
    ? _nudgeExplanationHtml(parsed)
    : _genericNoticeExplanationHtml(parsed);
  const key = uid != null ? uid : parsed.raw;
  return (
    `<details class="block-notice tc-collapsible ` +
    `${soft ? "block-notice-soft" : "block-notice-hard"}"` +
    ` data-notice-key="${escapeAttr(String(key))}">` +
    `<summary>` +
    `<span class="tc-status notice-glyph">${parsed.glyph === "⚠" ? "⚠️" : "↻"}</span>` +
    `<span class="tc-name notice-title">${escapeHtml(
      _runtimeNoticeTitle(parsed),
    )}</span>` +
    chips +
    `<span class="tc-caret">▶</span>` +
    `</summary>` +
    `<div class="block-notice-body">${body}` +
    `<div class="notice-label">Raw notice</div>` +
    `<pre class="notice-raw">${escapeHtml(parsed.raw)}</pre>` +
    `</div>` +
    `</details>`
  );
}

function _blockHtml(blk, bs) {
  if (blk.kind === "think") {
    const txt = WebBlockStream._text(blk);
    if (!txt.trim()) return "";
    const open = !blk.closed && !bs.ended && _visExpandOn("reasoning");
    return (
      `<details class="reasoning block-think"${open ? " open" : ""}>` +
      `<summary><span class="reasoning-caret">▶</span>💭 Reasoning</summary>` +
      `<div class="reasoning-body"><div class="reason-thought">${escapeHtml(txt)}</div></div>` +
      `</details>` +
      (blk.stats
        ? `<div class="assist-metrics block-metrics">${escapeHtml(blk.stats)}</div>`
        : "")
    );
  }
  if (blk.kind === "todos" || blk.kind === "tasks") {
    // 📝 TodoWrite checklist / 🗒️ harness-task card (TUI parity) — a live
    // collapsible card; expanded while the turn runs, folded in history.
    const isTodo = blk.kind === "todos";
    const items = (isTodo ? blk.todos : blk.tasks) || [];
    const shown = isTodo ? items : items.filter((t) => t.status !== "deleted");
    const done = shown.filter((i) => i.status === "completed").length;
    const glyph = {
      completed: "✔",
      in_progress: "▸",
      pending: "○",
      deleted: "✗",
    };
    const rows = shown
      .map((it) => {
        const st = it.status || "pending";
        const cls =
          st === "completed"
            ? "done"
            : st === "in_progress"
              ? "in-progress"
              : "";
        const text =
          isTodo && st === "in_progress" && it.activeForm
            ? it.activeForm
            : it.content || "";
        const id =
          !isTodo && it.id
            ? `<span class="card-id">#${escapeHtml(it.id)}</span> `
            : "";
        return `<li class="${cls}"><span class="card-mark">${glyph[st] || "○"}</span>${id}${escapeHtml(text)}</li>`;
      })
      .join("");
    const head = isTodo ? "📝 Tasks" : "🗒️ Harness tasks";
    let html =
      `<details class="block-card block-${blk.kind}"${bs.ended ? "" : " open"}>` +
      `<summary>${head} · ${done}/${shown.length} done</summary>` +
      `<ul class="card-list">${rows}</ul></details>`;
    if (blk.stats)
      html += `<div class="assist-metrics block-metrics">${escapeHtml(blk.stats)}</div>`;
    return html;
  }
  if (blk.kind === "plan") {
    // 📐 ExitPlanMode plan card — full markdown, expanded by default.
    let html =
      '<details class="block-card block-plan" open>' +
      "<summary>📐 Plan</summary>" +
      `<div class="card-md">${renderMarkdown(blk.plan || "")}</div></details>`;
    if (blk.stats)
      html += `<div class="assist-metrics block-metrics">${escapeHtml(blk.stats)}</div>`;
    return html;
  }
  if (blk.kind === "tool") {
    const icon =
      blk.status === "ok"
        ? "✓"
        : blk.status === "failed"
          ? "✗"
          : '<span class="tc-spin"></span>';
    const head =
      `<span class="tc-status">${icon}</span>` +
      `<span class="tc-name" title="${escapeAttr(blk.toolName)}">${escapeHtml(blk.toolName)}</span>` +
      (blk.args
        ? `<span class="tc-args" title="${escapeAttr(blk.args)}">${escapeHtml(toolArgsPreview(blk.args))}</span>`
        : "");
    const body = WebBlockStream._text(blk);
    let html;
    if (bs.toolShowBody) {
      // The tool body is COLLAPSED by default — the operator clicks the row to
      // reveal it. ``data-call-id`` lets renderBlockTurn keep it open across the
      // innerHTML repaint each stream tick triggers. Fall back to the block's
      // stable uid when the event carried no call_id, so the open state still
      // survives repaints.
      const key = blk.callId || "blk" + blk.uid;
      const cid = ` data-call-id="${escapeHtml(key)}"`;
      html =
        `<details class="block-tool toolcall toolcall-${escapeHtml(blk.status)} tc-collapsible"${cid}` +
        `${blk.status === "running" ? " open" : ""}>` +
        `<summary>${head}<span class="tc-caret">▶</span></summary>` +
        `<pre class="block-tool-body">${escapeHtml(
          (body.replace(/^\n+/, "") ||
            (blk.status === "running"
              ? "Waiting for output…"
              : "No output returned.")),
        )}</pre>` +
        `</details>`;
    } else {
      // No body to show (header-only, or /tools off) — nothing to expand.
      html = `<div class="block-tool toolcall toolcall-${escapeHtml(blk.status)}">${head}</div>`;
    }
    if (blk.stats)
      html += `<div class="assist-metrics block-metrics">${escapeHtml(blk.stats)}</div>`;
    return html;
  }
  // Runtime notice card (nudge / loop / distill / …). The whole card is a
  // disclosure: summary = event + reason chips, body = expandable explanation.
  if (blk.kind === "notice") {
    let html = runtimeNoticeCardHtml(blk.notice, blk.uid);
    // A turn that ends with no output after the notice lands its stats row on
    // this block (turnEnded targets the last visible block) — keep the row.
    if (blk.stats)
      html += `<div class="assist-metrics block-metrics">${escapeHtml(blk.stats)}</div>`;
    return html;
  }
  // answer block
  let html = '<div class="block-answer">';
  if (blk.badge)
    html += `<div class="answer-badge">${escapeHtml(blk.badge)}</div>`;
  html += renderMarkdown(WebBlockStream._text(blk));
  html += "</div>";
  if (blk.stats)
    html += `<div class="assist-metrics block-metrics">${escapeHtml(blk.stats)}</div>`;
  return html;
}

// Split a canonical user prompt (string, or OpenAI/Anthropic-style content
// blocks) into its plain text and the list of attached image URLs/data-URLs.
// Handles ``{type:'text',text}`` + ``{image_url:{url}}`` (or a bare string
// image_url), Anthropic ``{type:'image',source:{media_type,data}}``, and ACP
// ``{type:'image', data, mimeType}`` (what the WebUI/Telegram attach path posts).
// Only render media from safe schemes — block javascript:/vbscript:/etc. that a
// malformed upstream event could place in a clickable href + img src. data:
// URLs are restricted to RASTER images: data:image/svg+xml can carry script and
// must never become a clickable href.
function _safeMediaUrl(u) {
  const s = String(u == null ? "" : u).trim();
  if (!s) return "";
  if (/^(data:image\/(png|jpe?g|gif|webp);base64,|https?:\/\/|blob:)/i.test(s)) {
    return s;
  }
  // Same-origin authenticated artifact paths (optional BASE prefix like
  // /sharedagentsweb). Mirror markdown.js _mdArtifactHrefOk so user thumbs and
  // agent outputs share one allowlist — never arbitrary /api/… endpoints.
  if (typeof _mdArtifactHrefOk === "function" && _mdArtifactHrefOk(s)) {
    return s;
  }
  // Fallback when markdown.js is not loaded (unit sandboxes): accept only
  // …/files/<sid>/(optional digest/)<name> paths without a foreign scheme.
  if (!/^[a-z][a-z0-9+.\-]*:/i.test(s) && !s.startsWith("//")) {
    const pathOnly = s.split(/[?#]/, 1)[0];
    if (/(^|\/)files\/[^/]+\/(?:[0-9a-f]{64}\/)?[^/]+$/i.test(pathOnly)) {
      return s;
    }
  }
  return "";
}

function _displayableUserPrompt(text) {
  const t = String(text || "");
  // Supervisor auto-turns wrap the condition in a long instruction block.
  // Show the condition the way a normal You-bubble would.
  const m = t.match(
    /^\[goal auto-turn[^\]]*\][\s\S]*?Continue working toward the active goal:\s*\n+([\s\S]*?)\n+Current goal terminal state:/,
  );
  return m ? m[1].trim() : t;
}

/** Backend ``attached_files`` chips ({name, size, mime}); rendered beside
 *  the thumbnails, never part of the prompt text. */
function _attachedFileChips(files) {
  return Array.isArray(files)
    ? files.filter((f) => f && typeof f === "object" && f.name)
    : [];
}

function _userPromptParts(u, attachedFiles) {
  const files = _attachedFileChips(attachedFiles);
  if (typeof u === "string") return { text: _displayableUserPrompt(u), images: [], files };
  if (!Array.isArray(u)) return { text: "", images: [], files };
  let text = "";
  const images = [];
  for (const b of u) {
    if (!b) continue;
    if (typeof b.text === "string") text += b.text;
    const iu = b.image_url;
    if (iu) {
      const url = typeof iu === "string" ? iu : iu && iu.url;
      if (url) images.push(url);
    } else if (b.type === "image" && b.source && b.source.data) {
      const mt = b.source.media_type || "image/jpeg";
      images.push(`data:${mt};base64,${b.source.data}`);
    } else if (b.type === "image" && (b.data || b.url)) {
      // ACP content-block shape posted by send.js / Telegram.
      // Prefer a durable same-origin /files URL when present so history does
      // not need multi-MB data: URLs after reload.
      if (b.url) {
        images.push(String(b.url));
      } else if (b.data) {
        const mt = b.mimeType || b.mime_type || "image/png";
        images.push(`data:${mt};base64,${b.data}`);
      }
    }
  }
  return { text: _displayableUserPrompt(text), images, files };
}

// ── AskUserQuestion interactive widget ──
//
// When the model fires ``AskUserQuestion``/``UserQuestion`` its tool_call event
// carries ``{questions:[{question, header, multiSelect, options:[{label,
// description}]}]}``. We render an interactive card (radio for single-select,
// checkbox for multi, plus a free-form "Other"), POST the human's picks to
// ``/api/sessions/<sid>/answer`` (→ gateway → orchestrator resolves the turn's
// parked future), and lock the card when the tool_result arrives. Historical
// replay renders the same card and immediately locks it from the logged result.

// Seal the in-flight assistant bubble so the next transcript item (an
// ask-user card or a mid-turn steer) can sit at the current event boundary.
// Later reasoning/tools then open a fresh bubble BELOW that item instead of
// growing the previous one above it.
function _isNudgeNotice(notice) {
  return !!(notice && String(notice.event || "").toLowerCase() === "nudge");
}

function splitAssistantOnNudgeNotices(raw) {
  const lift =
    typeof parseRuntimeNoticeLine === "function"
      ? parseRuntimeNoticeLine
      : null;
  const src = String(raw == null ? "" : raw);
  if (!lift || !src) return [{ text: src }];
  const parts = [];
  let prose = [];
  let fence = "";
  let saw = false;
  const flush = () => {
    if (!prose.length) return;
    parts.push({ text: prose.join("\n") });
    prose = [];
  };
  const lines = src.split("\n");
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
    if (notice && _isNudgeNotice(notice)) {
      flush();
      parts.push({ notice });
      saw = true;
      continue;
    }
    prose.push(line);
  }
  flush();
  return saw && parts.length ? parts : [{ text: src }];
}

function _splitNudgeSegments(segments) {
  const parts = [];
  let cur = [];
  const flush = () => {
    if (!cur.length) return;
    parts.push({ type: "assistant", segments: cur });
    cur = [];
  };
  (Array.isArray(segments) ? segments : []).forEach((seg) => {
    if (!seg) return;
    let notice = null;
    if (seg.kind === "notice") {
      notice = seg.notice || (
        typeof parseRuntimeNoticeLine === "function"
          ? parseRuntimeNoticeLine(seg.text)
          : null
      );
    }
    if (notice && _isNudgeNotice(notice)) {
      flush();
      parts.push({ type: "nudge", notice });
      return;
    }
    cur.push(seg);
  });
  flush();
  return parts.length ? parts : [{ type: "assistant", segments: segments || [] }];
}

// Keep these strings byte-identical to sharedagents_agent.continuation
// IDLE_EMPTY_CONTINUE_* (tests/test_cli_idle_nudge.py asserts the copy).
const _NUDGE_MODEL_BODIES = {
  empty_after_tools:
    "The previous round produced no answer after your tool work. Treat that as a stall or error, not a finished turn. Continue from where you left off and write the answer. Do not restart the task.",
  reasoning_only_no_output:
    "You reasoned but did not produce an answer. That is a stall, not a finished turn — the operator already saw the reasoning. Continue and deliver the answer, or call the next tool you need.",
  empty_no_output:
    "The previous round produced no answer. Treat that as an error, not a finished turn. Continue toward the answer now.",
  thinking_starved_length:
    "The previous round ran out of tokens while reasoning and produced no "
    + "answer. Treat that as a stall, not a finished turn. Do not start "
    + "another long reasoning pass — the same output ceiling will cut it "
    + "again. Write the answer now (or call the next tool you need). Do not "
    + "restart the task.",
  // stream_truncated bodies (sharedagents_agent.agents.local_llm.state
  // *_NUDGE_PROMPT_BODY) minus their {tail}, which the notice line does not
  // carry. Without them a truncation nudge showed the empty-after-tools copy.
  length:
    "Token budget ran out (finish_reason=length) while you were still writing — the operator saw a cut-off answer. Resume from that exact point: finish the current sentence, then complete the section. Do not restart or re-summarize what you already wrote.",
  tool_call_truncated:
    "The output-token limit (finish_reason=length) cut off your response while you were still writing tool-call arguments. Those calls never ran, so nothing they would have done has happened. Re-sending the same call may be cut off again: the limit covers your whole response, reasoning included. Split the work into smaller calls, one per response — for a large file, write the first part, then add the rest with further edit calls — keeping each call well under the size that was cut off. Do not restart the task or repeat completed steps.",
};

function _nudgeModelBody(notice) {
  const reason = notice && notice.fields && notice.fields.reason;
  if (reason === "print_timeout") {
    const raw = String((notice.fields && notice.fields.wall) || "the").trim();
    const secs = raw.replace(/s$/i, "") || "the";
    return (
      "The previous print-mode turn hit the "
      + secs + "s wall before you returned. Resume from the last "
      + "completed checkpoint — do not restart the whole task. Finish "
      + "one smaller unit of work that can complete inside " + secs
      + "s, then return a status. Break any remaining work into later "
      + "turns. Keep at least one task in the foreground; do not start "
      + "a long job you cannot finish in this turn."
    );
  }
  if (reason === "reasoning_stall") {
    return _NUDGE_MODEL_BODIES.reasoning_only_no_output;
  }
  if (reason && _NUDGE_MODEL_BODIES[reason]) return _NUDGE_MODEL_BODIES[reason];
  return _NUDGE_MODEL_BODIES.empty_after_tools;
}

function _nudgeYouText(notice) {
  return _nudgeDisplayText(notice);
}

function _nudgeDisplayText(notice) {
  return String(_nudgeModelBody(notice) || "")
    .replace(/^\[sharedagents continuation\]\s*/i, "")
    .replace(/^\[sharedagents\]\s*/i, "")
    .trim();
}

function _createNudgeCardElement(notice, timestamp) {
  const text = _nudgeDisplayText(notice);
  const esc = typeof escapeHtml === "function" ? escapeHtml : String;
  const div = document.createElement("div");
  const ts = timestamp || new Date().toISOString();
  const timeLabel =
    typeof formatMessageTime === "function" ? formatMessageTime(ts) : "";
  div.className = "message message-nudge";
  div.dataset.nudge = "1";
  div.innerHTML =
    `<div class="message-header">` +
    `<div class="message-avatar nudge">N</div>` +
    `<span>Nudge</span>` +
    (timeLabel ? `<span class="message-time">${timeLabel}</span>` : "") +
    `</div>` +
    `<div class="message-content">${esc(text).replace(/\n/g, "<br>")}</div>`;
  return div;
}

function appendNudgeYouCard(notice, timestamp) {
  const text = _nudgeDisplayText(notice);
  const div = _createNudgeCardElement(notice, timestamp);
  if (typeof VirtualScroller !== "undefined" && VirtualScroller.add) {
    const idx = VirtualScroller.add(
      "nudge",
      text,
      div,
      null,
      null,
      null,
      null,
      timestamp,
    );
    VirtualScroller.attach(div, idx);
  } else if (typeof messagesContainer !== "undefined" && messagesContainer) {
    messagesContainer.appendChild(div);
  }
  return div;
}

function _promoteNudgesFromLiveTurn(t) {
  if (!t || !t.bs || !t.el) return false;
  const blocks = t.bs.blocks || [];
  const idx = blocks.findIndex(
    (b) => b && b.kind === "notice" && _isNudgeNotice(b.notice) && !b._promoted,
  );
  if (idx < 0) return false;
  const noticeBlk = blocks[idx];
  noticeBlk._promoted = true;
  const after = blocks.slice(idx + 1);
  t.bs.blocks = blocks.slice(0, idx);
  t.bs._answer = null;
  t.bs._think = null;
  t.bs.turnEnded({}, {});
  renderBlockTurn(t, false);
  if (typeof _stampTurnOutcome === "function") {
    _stampTurnOutcome(t.el, "ok");
  }
  state.liveTurn = null;
  appendNudgeYouCard(noticeBlk.notice, state.liveEventTs || "");
  const next =
    typeof ensureLiveAssistant === "function" ? ensureLiveAssistant() : null;
  if (next && next.bs && after.length) {
    next.bs.blocks = after.slice();
    after.forEach((b) => {
      if (b.kind === "answer") next.bs._answer = b;
      if (b.kind === "think") next.bs._think = b;
    });
    renderBlockTurn(next, true);
  }
  return true;
}

function _sealLiveAssistant() {
  const t = state.liveTurn;
  if (!t) return;
  t.bs.turnEnded({}, {});
  renderBlockTurn(t, false);
  state.liveTurn = null;
}

function _sealLiveAssistantBeforeAskUser(callId) {
  // A pending-approval snapshot can mount the question before its tool_call
  // reaches this stream. The card's presence does not mean the pre-question
  // assistant was sealed: without this check its continuation paints into the
  // old bubble ABOVE the answered card. A replay after continuation has
  // started must leave the newer bubble open, so compare transcript positions.
  const liveEl = state.liveTurn && state.liveTurn.el;
  const card = callId && state.askUserCards && state.askUserCards[callId];
  const cardEl = card && card.el;
  if (liveEl && cardEl && liveEl.parentNode === cardEl.parentNode
      && !(liveEl.compareDocumentPosition(cardEl) & Node.DOCUMENT_POSITION_FOLLOWING)) {
    return;
  }
  _sealLiveAssistant();
}

function _splitAskSegments(segments) {
  const parts = [];
  let assistant = [];
  const flush = () => {
    if (assistant.length) parts.push({ kind: "assistant", segments: assistant });
    assistant = [];
  };
  (Array.isArray(segments) ? segments : []).forEach((seg) => {
    if (
      seg &&
      seg.kind === "tool" &&
      typeof _isAskUserTool === "function" &&
      _isAskUserTool(seg.name)
    ) {
      flush();
      parts.push({ kind: "ask", segment: seg });
      return;
    }
    assistant.push(seg);
  });
  flush();
  return parts;
}

function _mountAskUserFromBlockStream(t) {
  const calls = t && t.bs && t.bs.askUserCalls;
  if (!calls || !calls.length) return;
  calls.forEach((call) => {
    if (!_askUserCardExists(call.callId)) {
      // Always a sibling transcript card AFTER this assistant — never inside
      // .message-content, or post-answer tokens render above the question.
      renderAskUserCard(call.callId, call.args);
      const card = state.askUserCards && state.askUserCards[call.callId];
      if (card && card.el && t.el && typeof _placeAskCardAfter === "function") {
        _placeAskCardAfter(t.el, card.el);
      }
    }
    if (call.content || call.ok != null) {
      _finalizeAskUserCard(call.callId, call.content || "", call.ok);
    } else if (t.bs.aborted) {
      _finalizeAskUserCard(call.callId, "", false, "⏹ Question aborted");
    }
    // A still-open parked ask is not closed just because this bubble sealed.
  });
  if (typeof updateSendState === "function") updateSendState();
}

// ── Tool-permission interactive card ──
//
// A gated tool (destructive Bash, a write outside the workspace, …) is NOT a
// tool call — the permission layer raises it BEFORE the tool runs, so unlike
// AskUserQuestion it produces no ``tool_call`` event. It used to surface ONLY in
// the cross-session approvals panel: the operator saw "Allow Bash?" in the menu
// and NOTHING in the conversation that had asked, then answering in the menu
// silently resumed the turn. The gateway now emits ``permission_request`` /
// ``permission_resolved`` into the session log; these render the same decision
// inline, where it was asked, using the renderAskUserCard pattern.
//
// Both surfaces post to the SAME session-bound ``/api/sessions/<sid>/permission``
// route, so the card is not a privilege widening — it is the existing decision,
// shown where the operator is already looking.

// A card blocks a duplicate only while it is STILL PENDING. Permission ids are
// NOT lifetime-unique: when the model emits a tool call with no id the worker
// synthesizes ``call_<idx>`` per ROUND (worker.py), so round 2's first gate is
// ``perm:call_0`` again. Keying "seen forever" would discard that second gate
// and put it back in the sidebar only.
function _permCardExists(requestId) {
  const card = state.permissionCards && state.permissionCards[requestId || ""];
  return !!(card && !card.finalized);
}

// Lazy history snapshots the open turn in REST and opens SSE at EOF.
// A permission_request that already landed in the JSONL is therefore never
// replayed — the sidebar still sees it via GET /api/approvals, the chat does
// not, so an "Allow bash?" request would sit in PENDING YOUR DECISION with
// no in-conversation card. Mount the same card from the approvals feed for the
// open session. Dedup via _permCardExists so a later live event is a no-op.
function hydratePendingPermissionCards() {
  if (typeof hydratePendingAskUserCards === "function") {
    hydratePendingAskUserCards();
  }
  if (typeof renderPermissionCard !== "function") return;
  const sid = String((state && state.currentSessionId) || "");
  if (!sid) return;
  const items = Array.isArray(state.pendingApprovals) ? state.pendingApprovals : [];
  items.forEach((a) => {
    if (!a || a.type !== "tool_permission") return;
    const recSid = String(a.session_id || "");
    const recGw = String(a.session_hash || "");
    if (recSid !== sid && recGw !== sid) return;
    const requestId = String(a.call_id || "");
    if (!requestId || _permCardExists(requestId)) return;
    renderPermissionCard({
      request_id: requestId,
      tool_name: a.summary || "tool",
      summary: a.summary || "",
      detail: a.detail || "",
      options: Array.isArray(a.options) ? a.options.slice() : [],
    });
  });
}

/** Rank the ACP option kinds so the safest choice reads first. */
function _permOptionRank(optionId) {
  switch (String(optionId || "")) {
    case "allow_once": return 0;
    case "reject_once": return 1;
    case "allow_always": return 2;
    case "reject_always": return 3;
    default: return 4;
  }
}

function renderPermissionCard(event) {
  const requestId = String((event && event.request_id) || "");
  if (!requestId) return;
  if (!state.permissionCards) state.permissionCards = {};
  // Dedup a replayed/duplicated event for a card that is still awaiting an
  // answer. A FINALIZED entry is history, not a live card: a later gate that
  // reuses the id (synthesized ``call_<idx>`` restarts each round) must mount a
  // fresh card, so the settled one stays in the transcript and the registry
  // points at the new one.
  if (_permCardExists(requestId)) return;

  const toolName = String((event && event.tool_name) || "tool");
  const summary =
    String((event && event.summary) || "") || `Allow ${toolName}?`;
  const detail = String((event && event.detail) || "");
  let options = Array.isArray(event && event.options) ? event.options.slice() : [];
  // Fail-closed on a malformed/empty option list: offer DENY ONLY. Synthesising
  // an Allow here would put a grant on screen that the gate never offered — and
  // the route now refuses exactly that (empty metadata is deny-only), so an
  // Allow button would be a lie that 400s. The operator can still deny, and the
  // gateway's own timeout still fail-closes.
  const synthesized = !options.length;
  if (synthesized) options = ["reject_once"];
  options.sort((a, b) => _permOptionRank(a) - _permOptionRank(b));

  // Seal the in-flight assistant bubble so the card sits at the event boundary
  // instead of growing the bubble above it (same reason as the ask-user card).
  _sealLiveAssistantBeforeAskUser();

  const bubble = appendSystemMessage("", false);
  const content = bubble.querySelector(".message-content");
  if (content) content.innerHTML = "";
  bubble.classList.add("perm-card");
  bubble.dataset.requestId = requestId;

  const form = el("div", "perm-form");
  form.appendChild(el("div", "perm-heading", "🔐 Permission required"));

  const q = el("div", "perm-q");
  q.textContent = summary;
  form.appendChild(q);

  if (detail) {
    const d = el("div", "perm-detail");
    d.textContent = detail; // textContent — the detail is an untrusted command
    form.appendChild(d);
  }

  const actions = el("div", "perm-actions");
  const status = el("span", "perm-status");
  const buttons = [];
  options.forEach((optionId) => {
    const btn = el(
      "button",
      "perm-btn perm-" + String(optionId).replace(/[^a-z_]/gi, ""),
      "",
    );
    btn.type = "button";
    btn.textContent =
      typeof permOptionLabel === "function"
        ? permOptionLabel(optionId)
        : String(optionId);
    if (String(optionId).indexOf("always") >= 0) {
      btn.title = "Applies to every later call this session — a wider grant";
    }
    btn.addEventListener("click", () => {
      // A synthesized deny is the STANDING deny, not a pick from a menu the
      // gate never published — post a bare `cancelled` (null optionId).
      _submitPermission(
        requestId, synthesized ? null : optionId, buttons, status,
      );
    });
    actions.appendChild(btn);
    buttons.push(btn);
  });
  actions.appendChild(status);
  form.appendChild(actions);

  if (content) content.appendChild(form);
  state.permissionCards[requestId] = {
    el: bubble, buttons: buttons, status: status, finalized: false,
  };
  scrollToBottom();
}

function _submitPermission(requestId, optionId, buttons, status) {
  const sid = state.currentSessionId;
  if (!sid) return;
  buttons.forEach((b) => { b.disabled = true; });
  if (status) {
    status.className = "perm-status";
    status.textContent = "…";
  }
  // resolvePermission (sessions.js) posts to the authenticated, session-bound
  // /permission route — the same one the sidebar panel uses.
  Promise.resolve(resolvePermission(sid, requestId, optionId)).catch(() => {
    // Delivery failed: re-arm so the operator is not stranded on a dead card.
    // The gateway's own fail-closed timeout still backstops a lost decision.
    buttons.forEach((b) => { b.disabled = false; });
    if (status) {
      status.className = "perm-status perm-status-error";
      status.textContent = "✗ Could not deliver — retry";
    }
  });
}

/** Collapse the card to a compact outcome line. Any surface may have answered. */
function _finalizePermissionCard(requestId, outcome, optionId, statusOverride) {
  const card = state.permissionCards && state.permissionCards[requestId || ""];
  if (!card || card.finalized) return;
  card.finalized = true;
  (card.buttons || []).forEach((b) => { b.disabled = true; });
  if (card.el) card.el.classList.add("perm-card-done");
  const status = card.status;
  if (!status) return;
  if (statusOverride) {
    status.className = "perm-status perm-status-error";
    status.textContent = statusOverride;
    return;
  }
  const allowed = String(outcome || "") === "selected" &&
    String(optionId || "").indexOf("allow") === 0;
  const label =
    typeof permOptionLabel === "function" && optionId
      ? permOptionLabel(optionId)
      : "";
  if (String(outcome || "") === "selected") {
    status.className =
      "perm-status " + (allowed ? "perm-status-done" : "perm-status-denied");
    status.textContent = (allowed ? "✓ " : "✗ ") + (label || (allowed ? "Allowed" : "Denied"));
  } else {
    // cancelled == the fail-closed standing deny (operator deny, /cancel, or
    // the timeout). Never render it as anything other than a denial.
    status.className = "perm-status perm-status-denied";
    status.textContent = "✗ Denied (no answer)";
  }
}

/** Close every still-interactive permission card when its turn terminates. */
function _finalizePendingPermissionCards(statusText) {
  const cards = state.permissionCards || {};
  Object.keys(cards).forEach((requestId) => {
    const card = cards[requestId];
    if (!card || card.finalized) return;
    _finalizePermissionCard(
      requestId, "", "", statusText || "✗ Permission request closed",
    );
  });
}

function _keepRetryingUI(event) {
  // Between-attempt sleep: keep Stop armed and label the status retrying.
  if (typeof _applyActiveTurnUI === "function") {
    _applyActiveTurnUI({ phase: "retrying" });
  }
  const n = event && (event.retry || (event.metrics && event.metrics.retry));
  const max = event && (
    event.retry_max || (event.metrics && event.metrics.retry_max)
  );
  if (_turn && n && max) _turn.tool = n + "/" + max;
  if (typeof statusText !== "undefined" && statusText) {
    statusText.textContent = (n && max)
      ? ("retrying " + n + "/" + max + "…")
      : "retrying…";
  }
  if (typeof updateSendState === "function") updateSendState();
}

// Apply context occupancy carried by the canonical event stream. The
// ``round_ended`` and ``header_update`` frames already contain the provider's
// latest prompt count; fetching the session again for every round would add a
// request per model call and can race the append-only log. ``prompt_tokens``
// is the context-meter numerator; cached/fresh buckets remain billing data.
function applyLiveContextMetrics(metrics) {
  if (!metrics || typeof metrics !== "object") return false;
  let changed = false;
  const used = Number(metrics.prompt_tokens);
  const previous = Number(state.lastContextUsed) || 0;
  // Match the persisted-header rule: an estimated restart reading must not
  // make a previously known high-water mark jump backwards. Real upstream
  // usage (including a post-compaction decrease) remains authoritative.
  if (
    Number.isFinite(used)
    && used > 0
    && !(metrics.tokens_estimated && previous > 0 && used < previous)
    && previous !== used
  ) {
    state.lastContextUsed = used;
    changed = true;
    const row = state.sessions && state.sessions[state.currentSessionId];
    if (row) row.last_prompt_tokens = used;
  }
  const max = Number(metrics.context_window);
  const previousMax = Number(state.lastContextMax) || 0;
  if (Number.isFinite(max) && max > 0 && previousMax !== max) {
    state.lastContextMax = max;
    changed = true;
    const row = state.sessions && state.sessions[state.currentSessionId];
    if (row) row.context_window = max;
  }
  if (changed && typeof renderStatusSegments === "function") {
    renderStatusSegments();
  }
  return changed;
}

function _engineFamily(engine) {
  return String(engine || "").toLowerCase().replace(/_(cli|acp)$/, "");
}

function _isForeignEngineEvent(event) {
  // A leftover engine (same JSONL, other CLI) must not finalize or
  // tokenize the live turn. Prod: antigravity_cli aborted a live
  // cursor_acp turn and the last tool card flipped to ✓ while cursor
  // was still working.
  const engine = String((event && event.engine) || "").trim();
  const live = String((state && state.liveEngine) || "").trim();
  if (!engine || !live) return false;
  return _engineFamily(engine) !== _engineFamily(live);
}

function renderCanonicalEvent(event, frame) {
  const type = event && event.type;
  // Turn boundaries are projected inline, so any state snapshot whose
  // cursor is older than the last boundary frame predates it.
  if ((type === "turn_ended" || type === "turn_started")
      && frame && Number.isFinite(frame.offset)) {
    const row = state.sessions && state.sessions[state.currentSessionId];
    if (row) row._lastTurnBoundarySeq = Math.max(
      Number(row._lastTurnBoundarySeq) || 0, frame.offset,
    );
  }
  if (type === "subagent_linked" || type === "subagent_result" || type === "subagent_done") {
    _applyDelegationUpdates([event]);
    return;
  }
  if (event && event.ts) state.liveEventTs = event.ts;
  if (state._instantAbort && type !== "turn_ended" && type !== "turn_started") {
    return;
  }
  // Leftover frames of a turn this tab already stopped, arriving after the
  // cancel-settlement timeout lifted the gate above. They would mint a fresh
  // card for the dead turn; its turn_ended settles below as a stale closer.
  if (
    type !== "turn_ended"
    && type !== "turn_started"
    && _isLocallyAbortedTurn(event)
  ) {
    return;
  }
  if (_isForeignEngineEvent(event)) {
    if (
      type === "turn_ended"
      || type === "thinking_delta"
      || type === "text_delta"
      || type === "tool_call"
      || type === "tool_delta"
      || type === "tool_result"
      || type === "tool_result_meta"
      || type === "round_started"
      || type === "round_ended"
    ) {
      return;
    }
  } else if (
    event
    && event.engine
    && !String((state && state.liveEngine) || "").trim()
    && (
      type === "turn_started"
      || type === "tool_call"
      || type === "thinking_delta"
      || type === "text_delta"
    )
  ) {
    // Join mid-turn (SSE from EOF, no turn_started): lock onto the
    // first live token so a leftover engine cannot win later.
    state.liveEngine = String(event.engine);
  }
  const physicalTurnId = String((event && event.physical_turn_id) || "");
  const closedIds = state.closedPhysicalTurnIds || {};
  // Drop only a CLOSED UUID's leftover frames (abort replay). Do not key
  // this off livePhysicalTurnId — it can stay set after turn_ended, which
  // discarded the next prompt's deltas until refresh.
  if (
    type !== "turn_started"
    && physicalTurnId
    && closedIds[physicalTurnId]
    && !state.liveTurn
    && !state.isStreaming
  ) {
    return;
  }
  switch (type) {
    case "session_state_update": {
      const incoming = normalizeSessionState(event);
      if (
        incoming.revision &&
        incoming.revision > state.sessionRevision
      ) {
        // Cross-frontend updates are authoritative and never discard the
        // transcript or a draft currently in the composer.
        applySessionState(event);
        refreshAgentCapabilities(state.currentSessionId, state.navEpoch)
          .catch(() => {});
      }
      break;
    }
    case "turn_started": {
      const physicalTurnId = String(event.physical_turn_id || "");
      const closedIds = state.closedPhysicalTurnIds || {};
      if (physicalTurnId && closedIds[physicalTurnId]) {
        if (!state.liveTurn && state.isStreaming) adoptInFlightAssistant();
        break;
      }
      if (state._askStopReleased) {
        const echo = String((state.pendingEcho && state.pendingEcho.text) || "");
        const startedUser = String(
          event.user || event.prompt || event.text || "",
        );
        const isFollowUp = !!(
          echo
          && startedUser
          && (startedUser === echo
            || startedUser.indexOf(echo) !== -1
            || echo.indexOf(startedUser) !== -1)
        );
        if (!isFollowUp) break;
        state._askStopReleased = false;
      }
      if (physicalTurnId && state.livePhysicalTurnId === physicalTurnId) {
        if (!state.liveTurn && state.isStreaming) adoptInFlightAssistant();
        break;
      }
      const replay = _turnStartedIsReplay(event);
      reconcilePromptReceipt(event);
      state.pendingCancel = null;
      state._instantAbort = false;
      _finalizePendingAskUserCards("Question ended");
      if (physicalTurnId) state.livePhysicalTurnId = physicalTurnId;
      if (event && event.engine) state.liveEngine = String(event.engine);
      state._askStopReleased = false;
      // A turn boundary is authoritative even when the preceding terminal SSE
      // frame was missed or replay filtering omitted it.  Never let an old
      // MEMORY FLUSH card capture this turn's assistant stream; a genuine
      // flush prompt below installs its own fresh card.
      state.liveMemoryFlushEl = null;
      if (Array.isArray(state.todos)) {
        state.todos = state.todos.filter((todo) => {
          const st = String((todo && todo.status) || "").toLowerCase();
          return st !== "completed" && st !== "done";
        });
        state.todosSid = state.currentSessionId; // live rows belong to this session
        renderHarnessWidget();
      }
      state.isStreaming = true;
      // A turn just started — the one moment "tell me when this is done" is
      // worth asking. The in-app opt-in card gates the browser's one-shot
      // permission dialog (notifications.js; plan §6.0). No-op once answered.
      if (typeof window.notificationTrigger === "function") {
        window.notificationTrigger("turn_started", event);
      }
      _startStreamingWatchdog();
      if (!_turn) _turnIndicatorStart(); // running indicator (idempotent vs send)
      // turn_started is post-accept warmup, not another "prompt received".
      if (!_turn.phase || _turn.phase === "sending" || _turn.phase === "received"
          || _turn.phase === "prompting" || _turn.phase === "warming") {
        _turnIndicatorMark("warming");
      }
      updateSendState(); // streaming → button becomes actionable (Stop/Queue), never disabled
      // Stamp the lane author for this turn's assistant bubble (and any
      // external user echo). Falls back to the active picker agent.
      state.liveTurnAgent =
        event.agent_name || state.currentAgent || "";
      // Per-turn timestamp for this turn's bubbles: the gateway stamps
      // ``ts`` (ISO-8601 UTC) on every event, incl. turn_started. Fall back
      // to the browser clock when a relayed event lacks it.
      state.liveTurnTs = event.ts || new Date().toISOString();
      state.liveTurnAuthor = authorFromMessage(event);
      const { text, images, files } = _userPromptParts(event.user, event.attached_files);
      if (replay) {
        // REST history or a prior SSE connection already opened this turn.
        // Replaying turn_started must not mint a second You-bubble or drop
        // the adopted assistant — that split one turn across two cards.
        if (!state.liveTurn) adoptInFlightAssistant();
        if (typeof _isCompactUserText === "function" && _isCompactUserText(text)) {
          _turnIndicatorMark("compacting", 0, "compact_context");
          if (typeof restoreCompactWidgetFromSession === "function") {
            restoreCompactWidgetFromSession({ state: "compacting" });
          }
        }
        break;
      }
      // A user prompt opens a turn — render the submitted text (and any attached
      // images, e.g. a photo sent from Telegram) as a user bubble. Covers both
      // our own submit and a turn started in another frontend. The canonical
      // field is ``user`` (a string, or content blocks).
      if (text || images.length || files.length) {
        const pe = state.pendingEcho;
        // Realtime voice already painted the user bubble (and may disagree
        // slightly with the gateway transcript) — never double-append.
        if (state.realtimeVoiceTurn) {
          state.pendingEcho = null;
        } else if (
          _pendingEchoMatches(
            pe, state.currentSessionId, text, images.length,
          )
        ) {
          // We already rendered this prompt optimistically on send (thumbnails
          // included) — don't double-render. A matching text with a DIFFERENT
          // image count is external (e.g. Telegram) and must paint its own.
          if (pe && pe.el && state.liveTurnAuthor.actors.length) {
            _applyMessageAuthor(pe.el, state.liveTurnAuthor); // canonical author wins
          }
          state.pendingEcho = null;
        } else if (
          typeof _isMemoryFlushPrompt === "function"
          && _isMemoryFlushPrompt(text)
        ) {
          // Silent end-of-session flush — collapsed widget, not a "You" bubble.
          const flushEl = appendMessage(
            "user", text, true, null, state.liveTurnAgent,
          );
          state.liveMemoryFlushEl = flushEl || null;
          state.pendingEcho = null;
        } else {
          appendMessage(
            "user", text, true, files.length ? [...images, ...files] : images,
            state.liveTurnAgent, null, state.liveTurnTs, state.liveTurnAuthor,
          ); // started elsewhere (or no echo)
        }
      }
      // Voice owns the assistant bubble for this utterance — do not reset
      // liveTurn in a way that invites a second ensureLiveAssistant later.
      if (!state.realtimeVoiceTurn) {
        // The preceding card keeps its backend outcome. A delayed terminal
        // can settle it after this turn has opened a fresh assistant card.
        state.liveTurn = null;
      }
      if (typeof _isCompactUserText === "function" && _isCompactUserText(text)) {
        _turnIndicatorMark("compacting", 0, "compact_context");
        if (typeof restoreCompactWidgetFromSession === "function") {
          restoreCompactWidgetFromSession({ state: "compacting" });
        }
      }
      break;
    }
    case "round_started": {
      if (!_turn) _turnIndicatorStart();
      _turnIndicatorMark("prefill");
      const t = state.liveTurn || ensureLiveAssistant();
      t.bs.roundStarted(event.round_idx);
      renderBlockTurn(t);
      break;
    }
    case "text_delta": {
      const d = event.delta || event.chunk || "";
      _turnIndicatorMark("writing", d.length);
      // Realtime voice already paints the answer from its own transcript
      // events — skip so a late/buffered SSE does not open a second bubble.
      if (state.realtimeVoiceTurn) break;
      const t = ensureLiveAssistant();
      t.bs.textDelta(d);
      renderBlockTurn(t);
      break;
    }
    case "thinking_delta": {
      // The model gates by reasoning mode (off → no-op, on → live deltas held
      // for the consolidated 💭, stream → live 💭). Mirrors should_show §7.
      const d = event.delta || event.chunk || "";
      _turnIndicatorMark("reasoning", d.length);
      if (state.realtimeVoiceTurn) break;
      const t = ensureLiveAssistant();
      t.bs.thinkingDelta(d);
      renderBlockTurn(t);
      break;
    }
    // A gated tool is a decision the CONVERSATION must
    // show, not only the approvals menu. Mirrors the ask-user card.
    case "permission_request": {
      const _rid = String(event.request_id || "");
      if (_rid && !_permCardExists(_rid)) {
        _turnIndicatorMark("ask", 0, event.tool_name || "tool");
        renderPermissionCard(event);
        // The turn is blocked on a human even though it has not ended.
        // onQuestion tags the notification by ``call_id``; pass the request id
        // under that name so two gates in one session don't collapse onto a
        // single replaceable notification.
        if (typeof window.notificationTrigger === "function") {
          window.notificationTrigger("question", { call_id: _rid });
        }
      }
      break;
    }
    case "permission_resolved": {
      _finalizePermissionCard(
        String(event.request_id || ""),
        event.outcome,
        event.option_id,
      );
      break;
    }
    case "tool_call": {
      // Always fed to the model — it routes direct-dispatch delegate bodies
      // into the 💬 answer (even with /tools off), drops compaction/internal
      // tools, and otherwise opens a 🔧 block per /tools.
      const _nm = event.name || "tool";
      const _ph =
        _nm === "compact_context"
          ? "compacting"
          : _isAskUserTool(_nm)
            ? "ask"
          : _nm === "OutputSection"
            ? "writing"
            : "tool";
      _turnIndicatorMark(_ph, 0, _nm);
      const _args = _eventToolArgs(event);
      // AskUserQuestion / UserQuestion round-trip to a human — render an
      // interactive answer card instead of a generic 🔧 tool pill. The answer
      // flows back via POST /api/sessions/<sid>/answer (see renderAskUserCard).
      if (_isAskUserTool(_nm)) {
        const callId = event.call_id || "";
        _sealLiveAssistantBeforeAskUser(callId);
        if (!_askUserCardExists(callId)) {
          renderAskUserCard(callId, _toolArgsObj(_args));
          // The turn is now blocked on a human — worth an alert even though
          // it has not ended (notifications.js; plan §5.1).
          if (typeof window.notificationTrigger === "function") {
            window.notificationTrigger("question", event);
          }
        }
        break;
      }
      // TUI parity: the footer todo/task chips tick LIVE as the tools fire —
      // turn_ended's refreshSessionWidgets() stays the authoritative reconcile.
      if (_isTodoTool(_nm)) {
        const todos = _todoItemsFromArgs(_args);
        if (todos) {
          state.todos = todos;
          state.todosSid = state.currentSessionId; // live rows belong to this session
          renderHarnessWidget();
        }
      } else if (
        _nm === "TaskCreate" ||
        _nm === "TaskUpdate" ||
        _nm === "TaskStop"
      ) {
        _liveUpsertHarnessTask(_nm, _toolArgsObj(_args), event.call_id || "");
        renderHarnessWidget();
      }
      const t = ensureLiveAssistant();
      t.bs.toolCall(event.call_id || "", _nm, _args);
      renderBlockTurn(t);
      break;
    }
    case "tool_delta": {
      const d = event.delta || event.chunk || event.content || event.text || "";
      const t = state.liveTurn || ensureLiveAssistant();
      if (t) {
        if (!t.bs._tools[event.call_id || ""]) {
          t.bs.toolCall(
            event.call_id || "",
            event.name || t.bs._titles[event.call_id || ""] || "tool",
            event.arguments,
          );
        }
        t.bs.toolDelta(event.call_id || "", d);
        renderBlockTurn(t);
      }
      break;
    }
    case "tool_result_meta": {
      // Ask-user cards: finalize on meta if no body event arrives (empty/malformed
      // tool results may never emit a tool_result body event).
      if (_askUserCardExists(event.call_id) || (state.askUserCards && state.askUserCards[event.call_id])) {
        _finalizeAskUserCard(event.call_id || "", "", event.ok);
        break;
      }
      // The ok/failed status lands on the META event; the body (if persisted)
      // arrives as a separate ``tool_result`` event.
      const t = state.liveTurn;
      if (t) {
        t.bs.toolResultMeta(event.call_id || "", event.ok, event.outcome, event.member_outcomes);
        renderBlockTurn(t);
      }
      // Status lands on meta (body may never arrive). Leave "running tool"
      // stuck after ✓ if we wait for tool_result.
      _turnIndicatorMark("reasoning", 0);
      break;
    }
    case "tool_result": {
      // The tool output body — routed to the 🔧 block (when /tools stream) or to
      // the 💬 answer for a direct-dispatch call.
      // AskUserQuestion result: lock the interactive card + show the answer,
      // bypassing the block-stream tool pill.
      if (_askUserCardExists(event.call_id) || (state.askUserCards && state.askUserCards[event.call_id])) {
        _turnIndicatorMark("reasoning", 0);
        _finalizeAskUserCard(
          event.call_id || "",
          typeof event.content === "string" ? event.content : "",
          event.ok,
        );
        break;
      }
      _turnIndicatorMark("reasoning", 0); // tool done → model resumes; refresh clock
      // A TaskCreate's footer row learns its assigned id from the result body
      // here (the args carried only the subject) — so later TaskUpdate/TaskStop
      // can match it by id. Fires regardless of /tools, like the live upsert.
      _liveHarnessTaskResult(
        event.call_id || "",
        typeof event.content === "string" ? event.content : "",
        event.ok,
      );
      const t = state.liveTurn;
      if (t) {
        const cid = event.call_id || "";
        if (!t.bs._tools[cid]) {
          t.bs.toolCall(
            cid,
            event.name || t.bs._titles[cid] || "tool",
            event.arguments,
          );
        }
        t.bs.toolResult(
          cid,
          typeof event.content === "string" ? event.content : "",
          event.ok,
        );
        renderBlockTurn(t);
      }
      break;
    }
    case "round_ended": {
      // The block renderer intentionally has no per-round stats row, but the
      // footer context meter must still follow each provider call. This is a
      // canonical SSE event, so the update stays on the stream and does not
      // issue a REST refresh.
      applyLiveContextMetrics(event.metrics);
      break;
    }
    case "artifact_reference": {
      // Structured agent output. The gateway stream has already removed the
      // local path; the digest resolves inside the authenticated /files jail.
      const t = ensureLiveAssistant();
      appendLiveArtifact(t, event);
      _turnIndicatorMark("writing", 0, event.display_name || "output");
      scrollToBottom();
      break;
    }
    case "turn_retry": {
      _keepRetryingUI(event);
      break;
    }
    case "turn_ended": {
      // Between-attempt retry must KEEP the running chrome. A queued
      // microtask after ``_stopStreaming`` used to relight "retrying N/M"
      // on one tab while ``isStreaming`` had already gone false on the
      // snap — the other computer then showed Stop/retrying against an
      // idle composer flag. Skip stop when this terminal is a retry.
      const keepRetrying = !!(
        !event.aborted
        && (event.retrying || (event.metrics && event.metrics.retrying))
      );
      const finishTurnChrome = () => {
        if (keepRetrying) {
          _keepRetryingUI(event);
          return;
        }
        state.pendingCancel = null;
        state._instantAbort = false;
        if (typeof statusText !== "undefined" && statusText) {
          statusText.textContent = "";
        }
        if (typeof _stopStreaming === "function") {
          _stopStreaming({ ignoreStaleActivity: true });
        }
      };
      // Notify the operator that this session is done (notifications.js reads
      // the settled transcript on a macrotask, so this fires from the ONE spot
      // at the top of the case and still covers all three exit paths below).
      if (typeof window.notificationTrigger === "function") {
        window.notificationTrigger("turn_ended", event);
      }
      // Cancel of a parked ask is often written AFTER the follow-up
      // /prompt's turn_started. Stop already cleared ``_instantAbort``, so
      // this aborted terminal must not ``_stopStreaming`` a newer live turn
      // — that accepted the prompt and then painted no reply.
      const endedPid = String((event && event.physical_turn_id) || "");
      const livePid = String(state.livePhysicalTurnId || "");
      // Stop/Steer already sealed this turn's card. When its turn_ended
      // lands after the settlement timeout, the steered prompt is already
      // live (or about to be): no second card, no "Turn aborted." line, and
      // the steered prompt's pending echo and running chrome stay intact.
      const locallyAborted = _isLocallyAbortedTurn(event);
      if (locallyAborted) _noteClosedPhysicalTurn(event);
      if (locallyAborted || (endedPid && livePid && endedPid !== livePid)) {
        _noteClosedPhysicalTurn(event);
        _finalizeNonLiveAssistantCards(event);
        if (typeof compactContextTurnEnded === "function") {
          compactContextTurnEnded(event);
        }
        // A stale closer must not dismiss a live parked ask — that is the
        // "card vanished until refresh" bug.
        _finalizePendingPermissionCards(
          event.aborted ? "⏹ Permission request aborted" : "✗ Denied (turn ended)",
        );
        break;
      }
      // Server/orchestrator abort (stale_live_reconcile, cancel, …) while
      // the composer still believes the turn is live: Stop was never
      // clicked, so ``_instantAbort`` is false and ``isStreaming`` stays
      // true. The next typed prompt then hits composerWillQueue() and
      // queues instead of starting a new turn. Clear streaming NOW, then
      // fall through so the live card can paint ABORTED (do not latch
      // _instantAbort here — that skips t.bs.turnEnded). Do not hard-
      // cancel the gateway session from this renderer: a replayed
      // terminal could abort a newer turn. A physical aborted
      // ``turn_ended`` plus the later assistant commit can both arrive;
      // only the first live terminal may mint a card.
      const hadLiveTurn = !!(state.liveTurn || state.isStreaming);
      _noteClosedPhysicalTurn(event);
      if (event.aborted && !state._instantAbort) {
        if (typeof _stopStreaming === "function") {
          _stopStreaming({ ignoreStaleActivity: true });
        }
        if (typeof updateSendState === "function") updateSendState();
      }
      // A footer-button compaction ends as a normal turn.
      // Hooked at the TOP of the case so every exit path (normal, instant
      // abort, voice) settles the widget instead of leaving it spinning.
      if (typeof compactContextTurnEnded === "function") {
        compactContextTurnEnded(event);
      }
      if (event.aborted) {
        _finalizePendingAskUserCards("⏹ Question aborted");
      } else if (event.error) {
        _finalizePendingAskUserCards("✗ Turn failed");
      }
      // Clean turn_ended is not a timeout. Leave unanswered cards up.
      // A permission card outlives its turn only when the terminal
      // permission_resolved was lost (crash / restart). Settle it as a DENIAL —
      // the gate itself is fail-closed, so the card must not imply a grant.
      _finalizePendingPermissionCards(
        event.aborted ? "⏹ Permission request aborted" : "✗ Denied (turn ended)",
      );
      // The engine stamps active_goal from its in-memory projection, which can
      // be momentarily empty for a goal that is still running in the registry
      // (set via the Goals page, after an engine restart, supervisor-driven).
      // An empty text is NO INFORMATION, not a clear — route through
      // applyTurnGoalMetrics so the widget survives the idle transition and
      // the durable registry stays the hide authority.
      if (event.metrics) applyTurnGoalMetrics(event.metrics);
      applyLiveContextMetrics(event.metrics);
      // The live turn_ended frame is the session log's own event (the gateway
      // tails the raw JSONL), so its metrics — including session_cost_usd —
      // apply above. The header reconcile below is only the backstop for a
      // consumer without an open stream (it returns early while SSE is open).
      // Skip during bulk history replay: loadSession already seeded the meter
      // and a from=0 replay would re-GET the full session once per turn.
      if (!(state.scroll && state.scroll.bulkLoading))
        syncContextMeterFromHeader();
      if (state._instantAbort) {
        state._instantAbort = false;
        // Stop already painted the live card. JSONL still delivers abort
        // then commit (two turn_endeds) — do not mint a second closer.
        const t = state.liveTurn || adoptInFlightAssistant();
        if (t && t.bs) {
          t.bs.turnEnded(event.metrics || {}, {
            finalThinking: event.final_thinking || "",
            aborted: !!event.aborted,
            error: event.error,
          });
          if (typeof renderBlockTurn === "function") renderBlockTurn(t);
        }
        state.liveTurn = null;
        state.liveMemoryFlushEl = null;
        state.realtimeVoiceTurn = false;
        try { clearTimeout(state._rtVoiceTurnTimer); } catch (_) {}
        state._rtVoiceTurnTimer = null;
        statusText.textContent = "";
        _stopStreaming({ ignoreStaleActivity: true });
        updateMicButton();
        if (state.monitorReadOnly) applyMonitorReadOnlyUi();
        syncTeacherState();
        if (!(state.scroll && state.scroll.bulkLoading))
          refreshSessionWidgets();
        renderGoalWidget();
        break;
      }
      // Voice path already painted + filled the assistant bubble. Calling
      // ensureLiveAssistant here was the duplicate-message bug (second empty
      // or re-rendered assistant under the voice one).
      if (state.realtimeVoiceTurn) {
        state.realtimeVoiceTurn = false;
        try { clearTimeout(state._rtVoiceTurnTimer); } catch (_) {}
        state._rtVoiceTurnTimer = null;
        state.realtimeAssistantEl = null;
        state.realtimeAssistantRaw = "";
        state.liveTurn = null;
        statusText.textContent = "";
        _stopStreaming({ ignoreStaleActivity: true });
        updateMicButton();
        if (state.monitorReadOnly) applyMonitorReadOnlyUi();
        syncTeacherState();
        if (!(state.scroll && state.scroll.bulkLoading))
          refreshSessionWidgets();
        renderGoalWidget();
        break;
      }
      // One physical turn → one assistant card. After SSE reconnect or a
      // REST snapshot, liveTurn may be null while the closer already exists.
      // Creating a new bubble here minted an empty SUCCEEDED header card.
      let t = state.liveTurn || (typeof adoptInFlightAssistant === "function" ? adoptInFlightAssistant() : null);
      if (
        !t
        && (
          (hadLiveTurn && (event.aborted || (event.error != null && String(event.error))))
          || (event.final_content != null && String(event.final_content).trim().length > 0)
          || (event.content != null && String(event.content).trim().length > 0)
        )
      ) {
        const cache = state.msgCache || [];
        const last = cache[cache.length - 1];
        const alreadyAborted = !!(event.aborted && last && last.role === "assistant" && (last.aborted || last.status === "aborted"));
        if (!alreadyAborted) {
          t = ensureLiveAssistant();
          const fc = event.final_content || event.content || "";
          if (fc && t && t.bs) {
            t.bs.textDelta(fc);
          }
        }
      }
      if (!t) {
        state.liveTurn = null;
        state.liveMemoryFlushEl = null;
        finishTurnChrome();
        updateMicButton();
        if (state.monitorReadOnly) applyMonitorReadOnlyUi();
        syncTeacherState();
        if (!(state.scroll && state.scroll.bulkLoading)) {
          refreshSessionWidgets();
          if (typeof refreshTreeLive === "function") refreshTreeLive();
        }
        renderGoalWidget();
        break;
      }
      t.bs.turnEnded(event.metrics || {}, {
        finalThinking: event.final_thinking || "",
        aborted: !!event.aborted,
        error: event.error,
      });
      renderBlockTurn(t);
      // Structured statistics belong only in the view-only compaction child.
      // Keep its snapshot available when the virtual scroller rebuilds it.
      if (state.currentAgent === "compaction" && typeof mountCompactionStatsCard === "function" &&
          t.el && t.el.dataset && typeof t.el.dataset.raw === "string" &&
          t.el.dataset.raw.trim().startsWith("compacted")) {
        const compactSnap = {
          stats: state.compactionLastStats,
          summary: state.compactionSummary || "",
          count: state.compactionCount || 0,
        };
        mountCompactionStatsCard(t.el, compactSnap);
        const compactIdx = state.msgCache.findIndex((m) => m.el === t.el);
        if (compactIdx >= 0) state.msgCache[compactIdx].compactionStats = compactSnap;
      }
      state.liveTurn = null; // turn complete — next turn starts a new bubble
      state.liveMemoryFlushEl = null;
      finishTurnChrome();
      updateMicButton(); // re-enable mic if it was disabled during streaming
      if (state.monitorReadOnly) applyMonitorReadOnlyUi();
      // A turn can change teacher supervision / model out-of-band; re-sync.
      syncTeacherState();
      // Refresh the footer widgets — a turn may have added/updated todos,
      // harness tasks, or spawned subagents; re-read the authoritative stores.
      // Skip during bulk replay (session open already called this once).
      if (!(state.scroll && state.scroll.bulkLoading)) {
        refreshSessionWidgets();
        // Clear sticky Stop / stale activity.can_stop: the stream server emits
        // a fresh ``state`` frame the moment it forwards this terminal frame
        // (iter_session_stream's terminal wake), and ``_applyState`` reconciles
        // from it. No /live poll. ``refreshTreeLive`` is a no-op while the
        // stream is open (it only reconciles when no live source exists).
        if (typeof refreshTreeLive === "function") refreshTreeLive();
      }
      renderGoalWidget();
      break;
    }
    case "subagent_spawned": {
      // Live lifecycle event (always delivered — not gated by tools=off).
      // Bump the subagent list immediately so the counter moves mid-turn;
      // turn_ended's refresh reconciles it against the authoritative store.
      const id = event.child_session_id || "";
      if (id && !state.subagents.some((s) => s.id === id)) {
        state.subagents.push({
          id,
          title: event.child_agent_name || id.slice(0, 12),
          subtitle: id.slice(0, 12),
        });
        renderHarnessWidget();
      }
      // Sticky Stop: mark tree live so Stop stays available after parent ends.
      if (typeof _setTreeLive === "function") _setTreeLive(true);
      else state.treeLive = true;
      if (typeof updateSendState === "function") updateSendState();
      // The stream pushes a ``state`` frame for this spawn (terminal wake), so
      // ``_applyState`` reconciles the authoritative subagent list + tree bit.
      if (typeof refreshTreeLive === "function") refreshTreeLive();
      break;
    }
    case "queue_msg_added": {
      const qid = event.queue_id || "";
      if (qid && !state.queue.some((q) => q.queue_id === qid)) {
        state.queue.push({
          queue_id: qid,
          text: event.text || "",
          display_text: event.display_text,
          attached_files: event.attached_files,
          source: event.source || "",
          author: event.author || "",
          actor: event.actor || null,
        });
      }
      renderHarnessWidget();
      break;
    }
    case "queue_msg_updated": {
      const qentry = state.queue.find((q) => q.queue_id === event.queue_id);
      if (qentry) {
        qentry.text = event.text || "";
        qentry.display_text = event.display_text;
        qentry.attached_files = event.attached_files;
      }
      renderHarnessWidget();
      break;
    }
    case "queue_msg_removed": {
      state.queue = state.queue.filter((q) => q.queue_id !== event.queue_id);
      renderHarnessWidget();
      break;
    }
    case "queue_msg_injected": {
      state.queue = state.queue.filter((q) => q.queue_id !== event.queue_id);
      const injectId = event.queue_id || "";
      if (injectId && !state._renderedInjects.has(injectId)) {
        state._renderedInjects.add(injectId);
        // appendMessage seals liveTurn for role=user so later Activity
        // lands below this steer instead of above it.
        const injectFiles = _attachedFileChips(event.attached_files);
        appendMessage(
          "user", event.text || "", true, injectFiles.length ? injectFiles : null,
          null, null, event.ts || new Date().toISOString(), authorFromMessage(event),
        );
      }
      renderHarnessWidget();
      break;
    }
    case "header_update": {
      // header_update drives state (model badge, title, goal progress, …).
      const patch = event.patch || {};
      // Local-LLM rounds persist ``last_prompt_tokens`` through this same
      // stream. Consume it immediately for engines whose round frame carries
      // no separate browser-facing state event.
      applyLiveContextMetrics({
        prompt_tokens: patch.last_prompt_tokens,
        context_window: patch.context_window,
      });
      if ("engine_model" in patch) {
        state.engineModel = patch.engine_model || state.engineModel;
        // The gateway resolves the operator-facing alias at
        // read time and rides the SAME event. A patch re-reported without an
        // alias clears it — a stale label must never outlive the wire id it
        // described (badge falls back to the raw id).
        // Do not seed the picker from ``patch.model`` — that field is the
        // session stamp (often a leftover ``auto`` / Codex id), not an
        // operator pick. Composer-owned intent stays frontend-owned.
        state.engineModelAlias =
          "engine_model_alias" in event ? (event.engine_model_alias || "")
            : ("engine_model_alias" in patch ? (patch.engine_model_alias || "") : "");
        updateHeader();
      }
      if (
        patch.title &&
        typeof isDefaultSessionTitle === "function" &&
        !isDefaultSessionTitle(patch.title)
      ) {
        const s = state.sessions[state.currentSessionId];
        if (s) {
          s.title = patch.title;
          renderSessionList();
          updateHeader();
        }
      }
      if (typeof patch.session_summary === "string" && patch.session_summary) {
        const s = state.sessions[state.currentSessionId];
        if (s) {
          s.session_summary = patch.session_summary;
          renderSessionList();
          updateHeader();
        }
      }
      // Manual-compaction stats ride the same header pipe (the engine persists
      // them in Session.save() during the compact tool turn, so this event
      // lands BEFORE the turn_ended that flips the widget to done). Update
      // state → the compaction turn card mounts from it at turn_ended.
      if ("compaction_last_stats" in patch ||
          "compaction_summary" in patch ||
          "compaction_count" in patch) {
        let compactChanged = false;
        let stats = patch.compaction_last_stats;
        if (typeof stats === "string") {
          try { stats = JSON.parse(stats); } catch (_e) { stats = null; }
        }
        if (stats && typeof stats === "object" && !Array.isArray(stats)) {
          state.compactionLastStats = stats;
          compactChanged = true;
        }
        if (typeof patch.compaction_summary === "string") {
          state.compactionSummary = patch.compaction_summary;
          compactChanged = true;
        }
        if (typeof patch.compaction_count === "number") {
          state.compactionCount = patch.compaction_count;
          compactChanged = true;
        }
        if (compactChanged) {
          renderStatusSegments();
        }
      }
      // Sync goal state from header_update so the chat page widget stays
      // in sync with the goals page without waiting for a turn_ended.
      // applyHeaderGoalPatch consumes active_goal identity-aware, resets
      // on an explicit met/archive detached projection, and re-renders —
      // applyGoalProgress alone used to clear the id and leave the old
      // visible text (codex round 2).
      applyHeaderGoalPatch(patch);
      // Model `budget` tool / TUI persist the cap as session_budget_usd.
      // Paint the Agent Monitor immediately; the mux also emits
      // `event: budget` so REST-suppressed chats stay authoritative.
      if (
        "session_budget_usd" in patch &&
        typeof applyLocalBudgetCap === "function"
      ) {
        const usd = Number(patch.session_budget_usd);
        if (Number.isFinite(usd) && usd >= 0) {
          applyLocalBudgetCap(usd);
        }
      }
      break;
    }
    case "compaction": {
      const phase = String(event.phase || "").toLowerCase();
      if (phase === "done" || phase === "ended" || phase === "complete") {
        // Details stay in the compaction child, opened through the Agents
        // footer. A lifecycle event must not create or split a parent bubble.
        // The post-compaction prompt count is authoritative and already rides
        // this SSE event; update the footer without a per-round REST fetch.
        applyLiveContextMetrics({
          prompt_tokens: event.after_tokens,
          context_window: event.context_window,
          tokens_estimated: false,
        });
        break;
      }
      if (!_turn) _turnIndicatorStart();
      _turnIndicatorMark("compacting", 0, "compact_context");
      if (typeof restoreCompactWidgetFromSession === "function") {
        restoreCompactWidgetFromSession({ state: "compacting" });
      }
      break;
    }
    // session_started / round_started / session_closed / lifecycle(plan_update, …):
    // no dedicated web widget yet — safely ignored (tool_result_meta IS handled
    // above; the bubble + reasoning disclosure already convey the turn;
    // todo/task widgets refresh on turn_ended). Kept explicit for future.
    default:
      break;
  }
}
