"""Tool registry: kind tags + a default set of tool definitions.

Sharedagents doesn't yet wire OpenAI tool-calling into the worker. When it
does, each registered `Tool` will become a function the model can call,
and the runtime will gate every invocation through `PermissionEvaluator`
(see `tools.permissions`).

Tools carry a `kind` tag (`read` / `write` / `exec` / `network`). The
permission decision matrix is keyed on `kind`, not on tool name — adding
a new tool is just a registry entry, no permission code change.

`exec`-kind tools accept a `command` (or compatible) field. The evaluator
re-classifies each call's actual command via `tools.safety.classify_command`
before deciding, so two `exec` calls can land on different decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Safe module-level import: agent_profiles' top-level imports are stdlib-only
# (its tools.registry import is lazy/in-function), so there is no import cycle.
from sharedagents_agent.agent_profiles import delegation_roster_text
from sharedagents_core import oscompat

# The loop tool schema builders are pure dicts (no agent-side imports in
# sharedagents_loop), so importing them here cannot cycle.
from sharedagents_loop.loop_tools import (
    cron_create_params,
    cron_delete_params,
    cron_list_params,
    schedule_wakeup_params,
)
from sharedagents_research.native_tools import schemas as _ontology_schemas


class ToolKind(Enum):
    """Coarse permission category. Drives `PermissionEvaluator.evaluate`."""

    READ = "read"  # inspects state only — Read, Glob, Grep, list_dir
    WRITE = "write"  # writes / edits files — Write, Edit, ApplyPatch
    EXEC = "exec"  # runs arbitrary shell — Bash, Exec, Process
    NETWORK = "network"  # outbound HTTP / RPC (may still be read-only)
    DESTRUCTIVE = "destructive"  # irreversible side effect — reserved for
    # an explicit host floor. Builtins and MCP tools are not born this kind;
    # a call becomes Criticality.DESTRUCTIVE only via classify_command.


@dataclass(frozen=True)
class Tool:
    """One callable tool — name, kind, and an OpenAI-compatible schema.

    `parameters` is the JSON Schema sent to the model in the
    `tools[].function.parameters` field of a `chat/completions` request.
    Schemas contain only real tool arguments. Criticality is derived host-side
    from this tool's kind plus the shell-command classifier when applicable.

    ``core`` is vestigial AS A WIRE GATE — it no longer
    controls what goes in ``tools[]``. The agent's permission set IS its tool
    set: ``compose_tool_allowlist(cfg)`` becomes the payload. Setting
    ``core=False`` does NOT keep a tool off the wire; only leaving it out of
    the agent's permission set does — and note an UNRESTRICTED config
    (``compose_tool_allowlist`` -> ``None``, the ``AgentConfig`` default)
    sends ``registry.openai_schemas()`` wholesale, i.e. every builtin.
    Authoritative paths: ``sharedagents_gateway.acp_bridge._build_worker_request``
    (wire) and ``LocalLLMEngine._runtime_allowed_tools`` (dispatch gate).
    Size guard: ``tests/test_153_tool_schema_budget.py``.
    The field is retained because ``core_tools()`` /
    ``core_tool_prompt_lines()`` still shape some prompt listings.
    """

    name: str
    kind: ToolKind
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    core: bool = False

    def openai_schema(self) -> dict[str, Any]:
        """Return the function-tool entry for `tools[]` in chat/completions."""
        description = self.description
        parameters = self.parameters or {"type": "object", "properties": {}}
        # Agent advertises a LIVE roster (built-ins + custom on-disk agents).
        # The Tool dataclass freezes import-time text; refresh here so a newly
        # scaffolded specialist (e.g. researcher on cursor_acp) appears without
        # restarting the Python process that imported this module.
        if self.name in AGENT_DISPATCH_TOOLS:
            live = delegation_roster_text()
            description = _live_agent_description(self.name, live)
            if not (parameters.get("properties") or {}).get("prompt"):
                parameters = dict(_AGENT_DISPATCH_PARAMS)
            parameters = dict(parameters)
            props = dict(parameters.get("properties") or {})
            props.update(_AGENT_LIFECYCLE_PROPS)
            st = dict(props.get("subagent_type") or {})
            st["description"] = _SUBAGENT_TYPE_DESC
            props["subagent_type"] = st
            parameters["properties"] = props
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": parameters,
            },
        }


def _params(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Build a JSON-schema parameters object.

    The ``criticality`` enum is no longer injected into the model-facing
    schema. It was a REQUIRED, off-distribution self-classification field on
    EVERY tool call — qwen3.6-27b (and kin) were never trained to emit it, so
    it added friction + schema bloat and pushed small models off-distribution.

    Safety is unchanged and does NOT depend on the model: the permission gate
    derives criticality host-side in ``permissions.py::_effective_criticality``
    from ``safety.classify_command`` (the non-overridable FORBIDDEN floor, e.g.
    ``rm -rf ~/``) plus the WRITE/READ tool-kind floors. That function still
    READS ``args['criticality']`` if a caller supplies one, and ``most_cautious``
    takes the max — so an explicit claim can only ever *raise* severity, never
    lower it. We simply stop *requiring* the model to produce it.
    """
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


# Optional async-delegation control for hidden ``delegate_*`` slash tools only.
# First-party Agent / AgentSync / AgentAsync have NO ``background`` argument —
# the tool name (and the caller's pinned ``delegation_mode``) is the mode.
# ``on_complete`` is intentionally absent: every background completion is
# injected into and wakes its parent.
_ASYNC_DELEGATE_PROPS: dict[str, Any] = {
    "background": {
        "type": "boolean",
        "description": (
            "Slash-delegate only. First-party AgentSync waits inline; "
            "AgentAsync returns handles. Do not send this on Agent tools."
        ),
    },
}

AGENT_DISPATCH_TOOLS: frozenset[str] = frozenset(
    {
        "AgentSync",
        "AgentAsync",
    }
)


def is_agent_tool(name: str) -> bool:
    """True for first-party AgentSync / AgentAsync dispatch tools."""
    return str(name or "") in AGENT_DISPATCH_TOOLS


def advertised_agent_tools(delegation_mode: str | None) -> frozenset[str]:
    """Model-facing Agent tool names for a pinned (or unpoliced) caller."""
    mode = str(delegation_mode or "").strip().lower()
    if mode == "background":
        return frozenset({"AgentAsync"})
    if mode == "foreground":
        return frozenset({"AgentSync"})
    return frozenset({"AgentSync", "AgentAsync"})


def rewrite_agent_tools_in_allowlist(
    allowed: set[str],
    delegation_mode: str | None,
) -> set[str]:
    """Replace leftover ``Agent`` allowlist entries with AgentSync / AgentAsync."""
    if not allowed:
        return allowed
    if "Agent" not in allowed and not (allowed & AGENT_DISPATCH_TOOLS):
        return allowed
    out = set(allowed)
    out.discard("Agent")
    out -= AGENT_DISPATCH_TOOLS
    out |= set(advertised_agent_tools(delegation_mode))
    return out


def _agent_mode_section(name: str) -> str:
    if name == "AgentSync":
        return (
            "MODE: ALWAYS synchronous. This call waits until every dispatched "
            "child (or every ``runs`` entry) finishes, then returns their "
            "results in this turn. There is no ``background`` argument. "
            "Build and plan use this tool only.\n\n"
        )
    if name == "AgentAsync":
        return (
            "MODE: ALWAYS asynchronous. This call returns job/task handles "
            "immediately and does not wait. Child results are pushed later. "
            "There is no ``background`` argument. Jarvis uses this tool only. "
            "Before every fresh dispatch, call action=list and inspect "
            "lineage — do not duplicate live children or wait on dead ones. "
            "Do not poll. Never ask the user a question while any listed "
            "child is still live; wait until every subagent has finished so "
            "you have the full context.\n\n"
        )
    return (
        "MODE: Use ``AgentSync`` (wait inline) or ``AgentAsync`` (detach). "
        "There is no ``Agent`` tool and no ``background`` argument.\n\n"
    )


def _live_agent_description(name: str, live: str) -> str:
    if name == "AgentSync":
        relay = (
            "RELAY: child results are private tool input in this turn. "
            "Validate and integrate useful evidence, and report it yourself."
        )
    elif name == "AgentAsync":
        relay = (
            "RELAY: results use durable delivery and wake this conversation "
            "once all children in a delegation wave are terminal. Each runs "
            "batch is one wave; standalone calls are solo waves; passing a "
            "handle's wave_id appends a same-goal dispatch to that in-flight "
            "wave. Validate and integrate useful evidence, "
            "and report it yourself."
        )
    else:
        relay = (
            "RELAY: sync results are private tool input; async results use "
            "durable delivery. Validate and integrate useful evidence, and "
            "report it yourself."
        )
    return (
        "Dispatch an isolated child. Targets:\n" + live + "\n\n"
        "CONTROL: ``action=list|abort|steer`` manages descendant "
        "subagents. Abort needs ``task_id``/``job_id`` or ``wave_id`` for "
        "a whole wave; steer preserves wave membership and queues "
        "``prompt`` for a live ``task_id`` at its next safe round. "
        "Omit ``action`` to dispatch.\n\n"
        "AUTHORITY: each child uses the selected profile's effective tools "
        "and ask/auto mode within runtime ceilings. Plan scope does not pass "
        "to Build; prompts grant no tools or leases. Runtime enforces depth "
        "3 and 39 concurrently-live agent leases per true root, charges "
        "nested cost/goals there, and cascades hard aborts.\n\n"
        "IDENTITY: omit ``task_id`` for a fresh child; distinct calls return "
        "distinct opaque IDs. Reuse a returned ID "
        "only to continue that exact child, never transfer it; an invalid or "
        "incompatible ID fails before any child turn and never falls back to "
        "a fresh child. Retrying one internal tool-call ID is idempotent. "
        "A still-running child's ID reports status and starts "
        "no turn (use action=steer for live guidance); only pass a prompt "
        "with a terminal task_id when you truly intend to continue it.\n\n"
        + _agent_mode_section(name)
        + "FAN-OUT: batch 2+ independent calls in one ``runs`` request. Sibling "
        "result/ID dependencies, repeated ``task_id``, or overlapping files/"
        "topics require sequential calls. Results keep one entry per "
        "request in request order; a visible "
        "entry failure does not erase successful siblings.\n\n"
        + relay
        + "\n\nPREFLIGHT: frontier engines (claude/codex/antigravity/cursor/"
        "opencode/external ACP) require goal, constraints, "
        "acceptance_criteria and a positive budget_usd, top level or per "
        "``runs`` entry; placeholders or zero fail. Local children need "
        "none."
    )


# The live roster is listed ONCE, under "Targets" in the tool description
# (OpenCode's ``task`` layout); repeating it here doubled the schema.
_SUBAGENT_TYPE_DESC = "One of the Targets listed above; its profile sets the child's permissions."


_AGENT_LIFECYCLE_PROPS: dict[str, Any] = {
    "action": {
        "type": "string",
        "enum": ["dispatch", "list", "abort", "steer"],
        "description": ("Omit/dispatch to spawn; list/abort/steer control descendant subagents without subagent_type."),
    },
    "job_id": {
        "type": "string",
        "description": "Background job id for action=abort.",
    },
    "wave_id": {
        "type": "string",
        "description": (
            "Wave id from a started handle: on a dispatch in a LATER turn, "
            "append the new child(ren) to that still-running wave — same goal "
            "only, one wake for the whole wave. A second dispatch in the SAME "
            "turn is refused; put every independent child in one "
            "AgentAsync(runs=[...]) call. With action=abort, stop every child "
            "in it."
        ),
    },
}


# Pre-flight delegation checklist — the four structured fields the delegation
# gate (``sharedagents_agent.delegation_gate.evaluate_preflight``) reads before it
# will drive an EXPENSIVE / frontier delegate turn (claude / codex /
# antigravity / cursor / opencode / any external ACP). Field names MUST match
# ``delegation_gate._CHECKLIST_FIELDS`` exactly — they flow verbatim through the
# ``Agent`` runner and the delegate shims into ``_drive_child_agent_turn``,
# where the gate validates presence + non-triviality. All four are OPTIONAL at
# the schema level: local subagents (explore / plan / general / build / sibling
# agents) are ungated and need none; the GATE — not the schema — enforces
# requiredness, and only for expensive engines. Reused for the top-level
# ``Agent`` call and for each ``runs[]`` fan-out entry so a batch dispatch can
# carry a per-entry checklist too.
_DELEG_CHECKLIST_PROPS: dict[str, Any] = {
    "goal": {"type": "string", "description": "One-sentence outcome."},
    "constraints": {"type": "string", "description": "Hard boundaries."},
    "acceptance_criteria": {
        "type": "string",
        "description": "Definition of done.",
    },
    "budget_usd": {
        "type": ["number", "string"],
        "description": "USD ceiling, number or '$3.00'.",
    },
}


# Terse mirror of _DELEG_CHECKLIST_PROPS for use inside the ``Agent`` tool's
# ``runs[]`` items. Same field names + types (the delegation gate reads these
# per-entry), without descriptions: PREFLIGHT in the tool description says
# they apply per ``runs`` entry, and duplicating guidance inside every array
# item added ~1.4 KB to the per-turn tool schema for no extra signal.
_DELEG_CHECKLIST_PROPS_TERSE: dict[str, Any] = {
    "goal": {"type": "string"},
    "constraints": {"type": "string"},
    "acceptance_criteria": {"type": "string"},
    "budget_usd": {"type": ["number", "string"]},
}


_AGENT_DISPATCH_PARAMS = _params(
    {
        **_AGENT_LIFECYCLE_PROPS,
        "subagent_type": {
            "type": "string",
            "description": _SUBAGENT_TYPE_DESC,
        },
        "description": {
            "type": "string",
            "description": "Short TUI and /subagents label.",
        },
        "prompt": {
            "type": "string",
            "description": (
                "Complete context; child cannot see parent chat. REQUIRED for "
                "a fresh spawn: without it the child's whole turn is empty."
            ),
        },
        "name": {
            "type": "string",
            "description": ("``general-purpose`` sibling name (default ``general``); ignored by external CLIs."),
        },
        "model": {
            "type": "string",
            "description": "Optional supported-CLI model override.",
        },
        "cwd": {
            "type": "string",
            "description": "Working directory; defaults to launch cwd.",
        },
        "task_id": {
            "type": "string",
            "description": "Returned ID to continue that child; see IDENTITY.",
        },
        "runs": {
            "type": "array",
            "description": (
                "Concurrent independent specs; entries override top-"
                "level subagent_type/prompt; results retain request "
                "order. Entries beyond the live-agent budget start as "
                "earlier ones finish. Use sequential calls for "
                "dependencies, overlap, or repeated task_id."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "subagent_type": {"type": "string"},
                    "description": {"type": "string"},
                    "prompt": {"type": "string"},
                    "name": {"type": "string"},
                    "model": {"type": "string"},
                    "cwd": {"type": "string"},
                    "task_id": {"type": "string"},
                    **_DELEG_CHECKLIST_PROPS_TERSE,
                },
                "required": ["subagent_type", "description", "prompt"],
            },
        },
        **_DELEG_CHECKLIST_PROPS,
    },
    required=[],
)


# Default tool set. Implementations come later; this is just the schema
# the model sees and the kind tag the permission evaluator reads.
BUILTIN_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="read",
        kind=ToolKind.READ,
        core=True,
        description=(
            "Read a file or directory from the local filesystem. If the "
            "path does not exist, an error is returned. Each line of a "
            "file is returned prefixed with its 1-based line number "
            "(``N: <content>``); pass ``offset`` (1-based start line) and "
            "``limit`` (number of lines, default 2000) to slice — strongly "
            "preferred for source files >100 lines so you don't flood your "
            "context with unrelated code. Output is capped at 50 KB per "
            "read with a resumable footer.\n\n"
            "Office documents (.docx, .doc, .pptx, .ppt, .xlsx, .xls, "
            ".pdf) are auto-converted to markdown via Microsoft's "
            "markitdown — you get clean text even for binary formats. "
            "Image files (PNG, JPEG, GIF, WebP) are shown to you as images."
        ),
        parameters=_params(
            {
                "filePath": {
                    "type": "string",
                    "description": ("The absolute path to the file or directory to read."),
                },
                "offset": {
                    "type": "integer",
                    "description": ("The line number to start reading from (1-indexed)."),
                },
                "limit": {
                    "type": "integer",
                    "description": ("The maximum number of lines to read (defaults to 2000)."),
                },
            },
            required=["filePath"],
        ),
    ),
    Tool(
        name="list",
        kind=ToolKind.READ,
        core=True,
        description=("List the entries of a directory. Prefer `Glob` for pattern-based file finding."),
        parameters=_params(
            {"path": {"type": "string"}},
            required=["path"],
        ),
    ),
    Tool(
        name="glob",
        kind=ToolKind.READ,
        core=True,
        description=(
            "Fast file pattern matching tool that works with any codebase "
            "size. Returns matching absolute paths, capped at 100 results. "
            "Patterns use standard shell glob syntax with ``**`` recursion.\n\n"
            "Examples:\n"
            "- ``**/*.py`` — every Python file under cwd.\n"
            "- ``/path/to/sharedagents/**/*.md`` — markdown under a "
            "specific tree.\n"
            "- ``tests/test_*_telegram*.py`` — telegram-related tests."
        ),
        parameters=_params(
            {
                "pattern": {
                    "type": "string",
                    "description": ("The glob pattern to match files against."),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The directory to search in. If not specified, the "
                        "current working directory will be used. IMPORTANT: "
                        "Omit this field to use the default directory. DO "
                        'NOT enter "undefined" or "null" - simply omit it '
                        "for the default behavior. Must be a valid directory "
                        "path if provided."
                    ),
                },
            },
            required=["pattern"],
        ),
    ),
    Tool(
        name="grep",
        kind=ToolKind.READ,
        core=True,
        description=(
            "Fast content search tool that works with any codebase size. "
            "Searches file contents using a regular expression. Returns "
            "``Found N matches`` followed by matches grouped per file as "
            "``<path>:`` headers with ``  Line n: <text>`` rows. Capped at "
            "100 matches."
        ),
        parameters=_params(
            {
                "pattern": {
                    "type": "string",
                    "description": ("The regex pattern to search for in file contents."),
                },
                "path": {
                    "type": "string",
                    "description": ("The directory to search in. Defaults to the current working directory."),
                },
                "include": {
                    "type": "string",
                    "description": ('File pattern to include in the search (e.g. "*.js", "*.{ts,tsx}").'),
                },
            },
            required=["pattern"],
        ),
    ),
    # Read-only LSP code intelligence. ONE enum-dispatched
    # tool, not one tool per operation — five separate schemas cost ~2.5K chars
    # on every unrestricted agent's wire payload (an unrestricted config sends
    # ``registry.openai_schemas()`` wholesale, see acp_bridge._build_worker_request)
    # and blew the per-round budget in tests/test_153_tool_schema_budget.py.
    # This mirrors opencode's ``lsp`` tool: one schema, one ``operation`` enum,
    # dispatched by runners.run_lsp. Scoped to code/explore/test via their
    # tools_allowlist. line/character are 1-based.
    Tool(
        name="lsp",
        kind=ToolKind.READ,
        core=False,
        description=(
            "Read-only language-server intelligence. Position ops "
            "(goToDefinition/findReferences/hover/goToImplementation/"
            "prepareCallHierarchy/incomingCalls/outgoingCalls) need "
            "filePath+line+character (1-based); documentSymbol needs only "
            "filePath; workspaceSymbol uses query (filePath selects the server)."
        ),
        parameters=_params(
            {
                "operation": {
                    "type": "string",
                    "enum": [
                        "goToDefinition",
                        "findReferences",
                        "hover",
                        "documentSymbol",
                        "workspaceSymbol",
                        "goToImplementation",
                        "prepareCallHierarchy",
                        "incomingCalls",
                        "outgoingCalls",
                    ],
                    "description": "The LSP operation to perform.",
                },
                "filePath": {
                    "type": "string",
                    "description": "The absolute or relative path to the file.",
                },
                "line": {
                    "type": "integer",
                    "description": "Line number (1-based, as shown in editors).",
                },
                "character": {
                    "type": "integer",
                    "description": "Character offset (1-based, as shown in editors).",
                },
                "query": {
                    "type": "string",
                    "description": ("Search query for workspaceSymbol. Empty string requests all symbols."),
                },
            },
            required=["operation", "filePath", "line", "character"],
        ),
    ),
    Tool(
        name="write",
        kind=ToolKind.WRITE,
        core=True,
        description=(
            "Writes a file to the local filesystem. Creates or overwrites "
            "the file, creating parent directories as needed. Prefer "
            "editing an existing file over creating a new one; read an "
            "existing file before overwriting it."
        ),
        parameters=_params(
            {
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
                "filePath": {
                    "type": "string",
                    "description": ("The absolute path to the file to write (must be absolute, not relative)."),
                },
            },
            required=["content", "filePath"],
        ),
    ),
    Tool(
        name="edit",
        kind=ToolKind.WRITE,
        core=True,
        description=(
            "Performs exact string replacements in files. Read the file "
            "first; ``oldString`` must match the file content exactly "
            "(a fuzzy-matching cascade tolerates whitespace/indentation "
            "drift, but never include the ``N: `` line-number prefix from "
            "``read``). By default the match must be unique — pass "
            "``replaceAll=true`` to substitute every occurrence. An empty "
            "``oldString`` creates a new file from ``newString``."
        ),
        parameters=_params(
            {
                "filePath": {
                    "type": "string",
                    "description": "The absolute path to the file to modify.",
                },
                "oldString": {
                    "type": "string",
                    "description": "The text to replace.",
                },
                "newString": {
                    "type": "string",
                    "description": ("The text to replace it with (must be different from oldString)."),
                },
                "replaceAll": {
                    "type": "boolean",
                    "description": ("Replace all occurrences of oldString (default false)."),
                },
            },
            required=["filePath", "oldString", "newString"],
        ),
    ),
    # OpenCode-compatible multi-file patch tool (id EXACTLY ``apply_patch``).
    # ONE ``patchText`` string carries an ``*** Begin/End Patch`` envelope with
    # Add/Delete/Update(+Move) file ops that apply atomically. ToolKind.WRITE so
    # it lands on the SAME permission floor as Write/Edit — the runner
    # (runners.run_apply_patch) is the child side of that gate, never a bypass.
    Tool(
        name="apply_patch",
        kind=ToolKind.WRITE,
        core=False,
        description=(
            "Use the `apply_patch` tool to edit files. Your patch language is a "
            "stripped-down, file-oriented diff format designed to be easy to "
            "parse and safe to apply. One call can Add, Delete, and Update "
            "(optionally renaming via `*** Move to:`) MULTIPLE files, applied "
            "atomically — if any part fails, NOTHING is written.\n\n"
            "Envelope:\n"
            "```\n"
            "*** Begin Patch\n"
            "*** Add File: <path>        (every following line is a `+` line = new contents)\n"
            "*** Delete File: <path>     (nothing follows)\n"
            "*** Update File: <path>     (patch in place)\n"
            "*** Move to: <newpath>      (optional rename, right after Update File)\n"
            "@@ <context>                (locate the hunk; may repeat)\n"
            " unchanged context line     (leading space)\n"
            "-removed line\n"
            "+added line\n"
            "*** End Patch\n"
            "```\n"
            "You MUST include an action header (Add/Delete/Update) for each "
            "file, and MUST prefix new lines with `+` even when creating a file."
        ),
        parameters=_params(
            {
                "patchText": {
                    "type": "string",
                    "description": (
                        "The full patch text that describes all changes to be "
                        "made (the entire `*** Begin Patch` … `*** End Patch` "
                        "envelope)."
                    ),
                },
            },
            required=["patchText"],
        ),
    ),
    Tool(
        name="bash",
        core=True,
        kind=ToolKind.EXEC,
        description=(
            ("Commands run through Command Prompt (cmd.exe /d /s /c). " if oscompat.IS_WINDOWS else "")
            + "This tool is for terminal operations like git, npm, docker, "
            "etc. DO NOT use it for file operations (reading, searching, "
            "editing) — use the specialized read/grep/glob/edit/write tools "
            "for that instead. Output streams line-by-line as the command "
            "produces it and is truncated to a temp file when it grows "
            "large. The runtime classifies each command's risk host-side "
            "and blocks catastrophic operations (e.g. `rm -rf ~`, disk "
            "wipes) automatically.\n\n"
            "Use the ``workdir`` parameter instead of ``cd`` to run in a "
            "different directory. For long-running work set "
            "``run_in_background=true`` → returns a ``bash_id`` "
            "immediately; poll with ``BashOutput``, stop with "
            "``KillShell``."
        ),
        parameters=_params(
            {
                "command": {
                    "type": "string",
                    "description": "The command to execute.",
                },
                "timeout": {
                    "type": "number",
                    "description": ("Optional timeout in milliseconds (default 120000, max 600000)."),
                },
                "workdir": {
                    "type": "string",
                    "description": (
                        "The working directory to run the command in. "
                        "Defaults to the current directory. Use this "
                        "instead of 'cd' commands."
                    ),
                },
                # sharedagents extension (no opencode analog): opencode's `bash`
                # has no background mode. sharedagents keeps run_in_background so
                # the load-bearing background-job path (BashOutput/KillShell,
                # wait-loops, builds) still works. Optional — the default
                # foreground path is unchanged when omitted.
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "sharedagents extension: run in the background and "
                        "return immediately with a bash_id for "
                        "BashOutput/KillShell. Turn stays responsive — use "
                        "for wait loops, polling, builds."
                    ),
                },
            },
            required=["command"],
        ),
    ),
    Tool(
        name="BashOutput",
        kind=ToolKind.READ,
        description=(
            "Pull new output lines from a background ``Bash`` (started "
            "with ``run_in_background=true``). If "
            "nothing new is buffered, waits briefly for the next line "
            "OR process exit. Cursor advances per call — each line is "
            "delivered exactly once. Always `read_only`."
        ),
        parameters=_params(
            {
                "bash_id": {
                    "type": "string",
                    "description": ("Background id from ``Bash(run_in_background=true)``."),
                },
                "filter": {
                    "type": "string",
                    "description": ("Optional regex; only lines matching are returned."),
                },
            },
            required=["bash_id"],
        ),
    ),
    Tool(
        name="KillShell",
        kind=ToolKind.EXEC,
        description=(
            "Terminate a background ``Bash`` (SIGTERM "
            "then SIGKILL after a grace period). Returns the exit code "
            "and unread-line count. Always safe to call against an "
            "already-exited shell."
        ),
        parameters=_params(
            {
                "shell_id": {
                    "type": "string",
                    "description": "Background id to terminate.",
                },
            },
            required=["shell_id"],
        ),
    ),
    Tool(
        name="ExitPlanMode",
        kind=ToolKind.READ,
        description=(
            "After Plan completes a decision-complete plan, present it and "
            "readiness for operator review. It is read-only and leaves "
            "the current mode unchanged; it never starts Build. Not for routine "
            "research summaries or implementation. Before calling this, ALWAYS "
            "write the plan artifact to disk — `plan/<topic>.md` in the "
            "operative repository (pick the repository that owns the work), or "
            "`~/.sharedagents/plan/<topic>.md` when no repository applies — and name "
            "the written path in the presentation. Both locations are permitted."
        ),
        parameters=_params(
            {
                "plan": {
                    "type": "string",
                    "description": ("Required non-empty Markdown plan, shown verbatim for readiness review."),
                },
            },
            required=["plan"],
        ),
    ),
    Tool(
        name="meeting_search",
        kind=ToolKind.READ,
        core=True,
        description=(
            "Search passive meeting transcript chunks after the user explicitly "
            "asks to recall a meeting. Meeting speech is never injected into a "
            "live turn; this read-only tool is the deliberate human-prompted "
            "retrieval seam. Refused outside private operator sessions."
        ),
        parameters=_params(
            {
                "query": {
                    "type": "string",
                    "description": "Words or phrases to find in saved meeting transcripts.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum matching chunks (1-20, default 6).",
                },
            },
            required=["query"],
        ),
    ),
    Tool(
        name="question",
        kind=ToolKind.NETWORK,  # answer comes from outside this process
        core=True,
        description=(
            "Ask the user questions during execution and wait for answers. "
            "Allow-all / auto applies to tool permissions, not this tool. "
            "Use it to clarify intent and improve alignment. Never use it "
            "for tool permission, and never use the permission widget. "
            "Routine work proceeds without an interview; reserve this for a "
            "missing required value or an intent fork. "
            "Set ``multiple=true`` to allow selecting more than one option; "
            "a 'Type your own answer' option is added automatically. If you "
            "recommend an option, make it first and add '(Recommended)'. "
            "Read-only."
        ),
        parameters=_params(
            {
                "questions": {
                    "type": "array",
                    "description": "Questions to ask.",
                    "items": {
                        "type": "object",
                        "required": ["question", "header", "options"],
                        "additionalProperties": False,
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Complete question.",
                            },
                            "header": {
                                "type": "string",
                                "description": ("Very short label (max 30 chars)."),
                            },
                            "multiple": {
                                "type": "boolean",
                                "description": ("Allow selecting multiple choices."),
                            },
                            "options": {
                                "type": "array",
                                "description": "Available choices.",
                                "items": {
                                    "type": "object",
                                    "required": ["label"],
                                    "additionalProperties": False,
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": ("Display text, 1-5 words."),
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "Explanation of choice.",
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
            required=["questions"],
        ),
    ),
    # Mirrors opencode's ``webfetch`` (tool/webfetch.ts): fetch a URL and return
    # the CONTENT directly in the requested format — NO LLM summarization (that
    # was the Claude-Code ``WebFetch`` shape with a ``prompt`` param, dropped
    # here). http→https upgrade, 5 MB cap. Backed by sharedagents's httpx +
    # trafilatura implementation (runners.run_web_fetch_async).
    Tool(
        name="webfetch",
        kind=ToolKind.NETWORK,
        core=False,
        description=(
            "Fetches content from a specified URL and returns it in the "
            "requested format. HTTP URLs are upgraded to HTTPS. Read-only."
        ),
        parameters=_params(
            {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from.",
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "markdown", "html"],
                    "description": (
                        "The format to return the content in (text, markdown, or html). Defaults to markdown."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in seconds (max 120).",
                },
            },
            required=["url"],
        ),
    ),
    # Mirrors opencode's ``websearch`` (tool/websearch.ts). sharedagents's backend is
    # Brave (opencode uses Exa/Parallel), so the provider-specific fields
    # (livecrawl/type/contextMaxCharacters) are accepted for schema parity but
    # only ``query``/``numResults``/``contextMaxCharacters`` shape sharedagents's
    # Brave call + output. The Claude-Code ``allowed_domains``/``blocked_domains``
    # params are dropped. Backed by runners.run_web_search_async.
    Tool(
        name="websearch",
        kind=ToolKind.NETWORK,
        core=False,
        description=(
            "Search the web and return the most relevant results (title, url, "
            "snippet). Real-time; use it for current events and information "
            "beyond the knowledge cutoff. Read-only."
        ),
        parameters=_params(
            {
                "query": {
                    "type": "string",
                    "description": "Websearch query.",
                },
                "numResults": {
                    "type": "number",
                    "description": "Number of search results to return (default: 8).",
                },
                "livecrawl": {
                    "type": "string",
                    "enum": ["fallback", "preferred"],
                    "description": ("Live crawl mode (default: 'fallback')."),
                },
                "type": {
                    "type": "string",
                    "enum": ["auto", "fast", "deep"],
                    "description": "Search type (default: 'auto').",
                },
                "contextMaxCharacters": {
                    "type": "number",
                    "description": ("Maximum characters for context string optimized for LLMs (default: 10000)."),
                },
            },
            required=["query"],
        ),
    ),
    Tool(
        name="decision",
        kind=ToolKind.NETWORK,
        core=False,
        description=(
            "List configured decision models or score context against named typed questions. "
            "Use action=list to discover model IDs; score defaults to the Web UI's OpenRouter Jev selection. "
            "Returns choices, ratings or yes/no probabilities without executing the chosen action. "
            "Context is sent to the selected provider; hosted inference may incur cost."
        ),
        parameters=_params(
            {
                "action": {"type": "string", "enum": ["list", "score"]},
                "model": {"type": "string", "description": "Optional catalogue ID from action=list."},
                "state": {
                    "type": ["string", "object", "array"],
                    "description": "Required for score. Plain text is automatically wrapped as a text object.",
                },
                "questions": {
                    "type": "object",
                    "minProperties": 1,
                    "maxProperties": 8,
                    "description": "Required for score. Named questions with type and instructions. Choice criteria: 2–16 label/description pairs; score criteria: 2–16 ordered descriptions; noul: no criteria.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["choice", "score", "noul"]},
                            "instructions": {"type": "string"},
                            "criteria": {
                                "oneOf": [
                                    {"type": "object", "additionalProperties": {"type": "string"}},
                                    {"type": "array", "items": {"type": "string"}},
                                ]
                            },
                        },
                        "required": ["type", "instructions"],
                        "additionalProperties": False,
                    },
                },
                "checkpoint": {
                    "type": "string",
                    "description": "Optional checkpoint advertised by the selected model.",
                },
            },
            required=["action"],
        ),
    ),
    Tool(
        name="x_tweets",
        kind=ToolKind.NETWORK,
        core=False,
        description=(
            "Download live X (Twitter) posts for a topic. Host-side search: "
            "any native agent model (Qwen, DeepSeek, Grok, …) can call this; "
            "the session model is not used. Returns a short narrative plus "
            "cited post URLs. Requires XAI_API_KEY. Read-only; "
            "costs ~$0.01–0.05 per call."
        ),
        parameters=_params(
            {
                "query": {
                    "type": "string",
                    "description": "Event or topic to search on X.",
                },
                "hours": {
                    "type": "number",
                    "description": "Lookback window in hours (default 24, max 168).",
                },
            },
            required=["query"],
        ),
    ),
    Tool(
        name="todowrite",
        kind=ToolKind.READ,
        core=True,
        description=(
            "Create and maintain a structured task list for the current "
            "coding session. Use it when a task has 3+ steps. Send the whole "
            "updated list on every call. Each item has ``content`` (brief "
            "task description), ``status`` (pending / in_progress / "
            "completed / cancelled), and ``priority`` (high / medium / low). "
            "Exactly one item may be ``in_progress`` at a time; mark items "
            "completed only after they are actually done."
        ),
        parameters=_params(
            {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": ("Brief description of the task."),
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                    "cancelled",
                                ],
                                "description": ("pending, in_progress, completed, cancelled."),
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "high, medium, low.",
                            },
                        },
                        "required": ["content", "status", "priority"],
                    },
                    "description": "The updated todo list.",
                },
            },
            required=["todos"],
        ),
    ),
    Tool(
        name="Skill",
        kind=ToolKind.READ,
        core=True,
        description=(
            "Load a skill by name and return its full SKILL.md body "
            "(instructions, scripts, references). The system prompt lists "
            "available skills with short descriptions — call this when a "
            "skill matches the task instead of reinventing the workflow. "
            "Use ``skill`` (preferred) or ``name`` for the skill id."
        ),
        parameters=_params(
            {
                "skill": {
                    "type": "string",
                    "description": (
                        "Skill name to load (e.g. a routines skill). "
                        "Must match an installed skill from the Skills catalog."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": ("Legacy alias for ``skill`` — same skill name."),
                },
                "args": {
                    "type": "string",
                    "description": ("Optional free-form args noted in the skill header when the skill documents them."),
                },
            },
            required=["skill"],
        ),
    ),
    Tool(
        name="compact_context",
        kind=ToolKind.READ,
        description=(
            "Squeeze your own conversation context. Trims oldest large "
            "tool results, drops them entirely if still over budget, then "
            "drops oldest user/assistant turns until the message list "
            "fits the target. Always read_only on the filesystem; the "
            "agent runtime intercepts this call and operates on the "
            "in-flight messages list directly (no subprocess). Call this when you notice "
            "earlier tool results are no longer load-bearing for the "
            "remaining work — better than letting the runtime auto-compact "
            "near the context cap."
        ),
        parameters=_params(
            {
                "target_tokens": {
                    "type": "integer",
                    "description": (
                        "Optional target prompt size after compaction. "
                        "Defaults to 50% of `compact_at_frac × "
                        "upstream_max_context` so the next few rounds "
                        "have plenty of headroom."
                    ),
                },
            },
            required=[],
        ),
    ),
    Tool(
        name="delegate_claude",
        kind=ToolKind.NETWORK,
        description=(
            "Delegate a question to Claude Code (a stronger external "
            "agent). Use when a task exceeds your own ability — deep "
            "research with web access, multi-file refactors, design "
            "review where you want a second opinion. Claude has its own "
            "tools (Read/Write/Edit/Bash/WebFetch/WebSearch) and by "
            "default runs in the working directory sharedagents was launched "
            "from. Pass `cwd` to point Claude at a different directory "
            "(e.g. a worktree for isolated parallel work).\n\n"
            "**Session continuity is automatic.** Claude's `session_id` "
            "is persisted at `<sessions_dir>/<sid>.delegate_claude.json` "
            "and `--resume`'d on every subsequent call within one sharedagents "
            "session, so Claude keeps the FULL prior delegate_claude "
            "conversation in its own context.\n\n"
            "**Critical: do NOT restate prior delegate_claude exchanges.** "
            "On a follow-up call, only forward the NEW context Claude "
            "is missing — the user turns, tool results, and your own "
            "reasoning that have happened SINCE your last delegate_claude "
            "call. Re-summarising the whole sharedagents conversation is "
            "wasteful: Claude already has every prior exchange in "
            "`--resume` state. Treat `prompt` as a delta, not a full "
            "transcript.\n\n"
            "On the FIRST delegate_claude call in this session, do "
            "include the salient context (user's question, tool results, "
            "your findings) — that's the summarization step that "
            "replaces the truncation-based preamble the slash bridge "
            "used to build. After that, just send the increment.\n\n"
            "The full Claude reply comes back as the tool result and is "
            'yours to cite. Use mode="plan" for read-only research; '
            "auto mode may run shell commands and edit files."
        ),
        parameters=_params(
            {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The message to send Claude. On the FIRST call "
                        "in this sharedagents session, include the salient "
                        "context (Claude doesn't see sharedagents history "
                        "automatically). On follow-up calls, send only "
                        "what's NEW since your last delegate_claude "
                        "call — Claude already has the prior exchange "
                        "via --resume. Don't restate."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["plan", "auto"],
                    "description": (
                        "Permission mode for Claude. `plan`=read-only sandbox "
                        "(no edits), `auto`=bypass permissions (Edit, Write, "
                        "Bash, etc. without prompting; required in headless "
                        "`-p` mode where no one can answer). "
                        "Default: inherits the sharedagents session's current "
                        "permission mode."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Optional working directory for the spawned "
                        "`claude` CLI. Absolute paths are used as-is; "
                        "relative paths resolve against sharedagents's launch "
                        "cwd. The path must exist and be a directory or "
                        "the call fails before spawning. Omit to inherit "
                        "sharedagents's launch directory (the historical "
                        "default — most calls don't need this). Useful "
                        "for pointing Claude at a specific worktree for "
                        "isolated parallel work."
                    ),
                },
            },
            required=["prompt"],
        ),
    ),
    Tool(
        name="delegate_codex",
        kind=ToolKind.NETWORK,
        description=(
            "Delegate a question to Codex (OpenAI's CLI agent — a stronger "
            "external agent). Use when a task exceeds your own ability — "
            "deep research with web access, multi-file refactors, design "
            "review where you want a second opinion. Codex has its own "
            "tools (Read/Write/Edit/Bash/WebFetch/WebSearch) and by "
            "default runs in the working directory sharedagents was launched "
            "from. Pass `cwd` to point Codex at a different directory "
            "(e.g. a worktree for isolated parallel work).\n\n"
            "**Session continuity is automatic.** Codex's `thread_id` is "
            "persisted at `<sessions_dir>/<sid>.delegate_codex.json` and "
            "`exec resume`'d on every subsequent call within one sharedagents "
            "session, so Codex keeps the FULL prior delegate_codex "
            "conversation in its own context.\n\n"
            "**Critical: do NOT restate prior delegate_codex exchanges.** "
            "On a follow-up call, only forward the NEW context Codex is "
            "missing — the user turns, tool results, and your own "
            "reasoning that have happened SINCE your last delegate_codex "
            "call. Re-summarising the whole sharedagents conversation is "
            "wasteful: Codex already has every prior exchange in "
            "`exec resume` state. Treat `prompt` as a delta, not a full "
            "transcript.\n\n"
            "On the FIRST delegate_codex call in this session, do "
            "include the salient context (user's question, tool results, "
            "your findings) — that's the summarization step that "
            "replaces the truncation-based preamble the slash bridge "
            "used to build. After that, just send the increment.\n\n"
            "The full Codex reply comes back as the tool result and is "
            'yours to cite. Use mode="plan" for read-only research; '
            "auto mode may run shell commands and edit files."
        ),
        parameters=_params(
            {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The message to send Codex. On the FIRST call "
                        "in this sharedagents session, include the salient "
                        "context (Codex doesn't see sharedagents history "
                        "automatically). On follow-up calls, send only "
                        "what's NEW since your last delegate_codex "
                        "call — Codex already has the prior exchange "
                        "via `exec resume`. Don't restate."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["plan", "auto"],
                    "description": (
                        "Permission mode for Codex. `plan`=read-only sandbox "
                        "(no edits), `auto`=bypass approvals and sandbox "
                        "(Edit, Write, Bash, etc. without prompting; required "
                        "in headless mode where no one can answer). "
                        "Default: inherits the sharedagents session's current "
                        "permission mode."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Optional working directory for the spawned "
                        "`codex` CLI. Absolute paths are used as-is; "
                        "relative paths resolve against sharedagents's launch "
                        "cwd. The path must exist and be a directory or "
                        "the call fails before spawning. Omit to inherit "
                        "sharedagents's launch directory (the historical "
                        "default — most calls don't need this). Useful "
                        "for pointing Codex at a specific worktree for "
                        "isolated parallel work."
                    ),
                },
            },
            required=["prompt"],
        ),
    ),
    Tool(
        name="delegate_antigravity",
        kind=ToolKind.NETWORK,
        description=(
            "Delegate a question to Google Antigravity (via the official "
            "``@google/antigravity-cli`` — a stronger external agent). Use "
            "when a task benefits from Antigravity's strengths: long-context "
            "summarisation (>1M tokens), multimodal reasoning, or a "
            "second opinion alongside ``delegate_claude`` / "
            "``delegate_codex``. Antigravity has its own tools "
            "(Read/Write/Edit/Bash/WebFetch/WebSearch) and by default "
            "runs in the working directory sharedagents was launched from. "
            "Pass ``cwd`` to point Antigravity at a different directory.\n\n"
            "**Session continuity is automatic.** Antigravity's session "
            "UUID is persisted at "
            "``<sessions_dir>/<sid>.delegate_antigravity.json`` and "
            "``--resume``'d on every subsequent call within one "
            "sharedagents session, so Antigravity keeps the FULL prior "
            "delegate_antigravity conversation in its own context.\n\n"
            "**Critical: do NOT restate prior delegate_antigravity "
            "exchanges.** On a follow-up call, only forward the NEW "
            "context Antigravity is missing — the user turns, tool results, "
            "and your own reasoning that have happened SINCE your "
            "last delegate_antigravity call. Re-summarising the whole "
            "sharedagents conversation is wasteful: Antigravity already has "
            "every prior exchange in ``--resume`` state. Treat "
            "``prompt`` as a delta, not a full transcript.\n\n"
            "On the FIRST delegate_antigravity call in this session, do "
            "include the salient context (user's question, tool "
            "results, your findings). After that, just send the "
            "increment.\n\n"
            "The full Antigravity reply comes back as the tool result and "
            'is yours to cite. Use mode="plan" for read-only research; '
            "yolo mode may run shell commands and edit files.\n\n"
            "**Auth required**: ``ANTIGRAVITY_API_KEY`` (or "
            "``GOOGLE_GENAI_USE_VERTEXAI`` / ``GOOGLE_GENAI_USE_GCA``) "
            "must be set in the agent's ``.env`` — operators put it "
            "alongside ``OPENROUTER_API_KEY``."
        ),
        parameters=_params(
            {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The message to send Antigravity. On the FIRST "
                        "call in this sharedagents session, include the "
                        "salient context (Antigravity doesn't see sharedagents "
                        "history automatically). On follow-up calls, "
                        "send only what's NEW since your last "
                        "delegate_antigravity call — Antigravity already has "
                        "the prior exchange via ``--resume``. Don't "
                        "restate."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["plan", "auto"],
                    "description": (
                        "Permission mode for Antigravity. ``plan``=``--mode "
                        "plan --sandbox``, ``auto``=``--dangerously-skip-"
                        "permissions``. Default: inherits the sharedagents "
                        "session's current permission mode."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Optional Antigravity model id (e.g. ``gemini-2.5-pro`` "
                        "or ``gemini-2.0-flash-exp``). Omit to use the "
                        "Antigravity CLI default configured in "
                        "``~/.antigravity/settings.json``."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Optional working directory for the spawned "
                        "``antigravity`` CLI. Absolute paths are used "
                        "as-is; relative paths resolve against "
                        "sharedagents's launch cwd. Must exist and be a "
                        "directory or the call fails before spawning. "
                        "Omit to inherit sharedagents's launch directory."
                    ),
                },
            },
            required=["prompt"],
        ),
    ),
    Tool(
        name="delegate_zcode",
        kind=ToolKind.NETWORK,
        description=(
            "Delegate a question to Z.AI's **ZCode** CLI (``zcode``, the "
            "official ``zcode-app-cli``) running on a GLM Coding Plan. Use "
            "when a task benefits from a GLM-family second opinion alongside "
            "``delegate_claude`` / ``delegate_codex`` / "
            "``delegate_antigravity``. ZCode has its own tools (file read/"
            "write/edit, shell, web, MCP, skills) and runs in the working "
            "directory sharedagents was launched from; pass ``cwd`` to point it "
            "elsewhere.\n\n"
            "**Session continuity is automatic.** ZCode's session id "
            "(``sess_…``) is persisted at "
            "``<sessions_dir>/<sid>.delegate_zcode.json`` and passed as "
            "``--resume`` on every subsequent call within one sharedagents "
            "session, so ZCode keeps the FULL prior conversation in its own "
            "context.\n\n"
            "**Critical: do NOT restate prior delegate_zcode exchanges.** On "
            "a follow-up call, send only what is NEW since your last "
            "``delegate_zcode`` call — treat ``prompt`` as a delta, not a "
            "full transcript. On the FIRST call, DO include the salient "
            "context (question, tool results, findings).\n\n"
            "The full ZCode reply is returned as the tool result and is "
            'yours to cite. ``mode="build"`` is the most conservative; '
            "``auto``/``yolo`` lets ZCode edit files and run commands.\n\n"
            "**Auth**: ZCode uses its own Z.AI OAuth login "
            "(``zcode login`` → ``~/.zcode/v2/credentials.json``), which "
            "covers the coding-plan quota. No sharedagents-side key is required."
        ),
        parameters=_params(
            {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The message to send ZCode. On the FIRST call in this "
                        "sharedagents session, include the salient context (ZCode "
                        "doesn't see sharedagents history automatically). On "
                        "follow-up calls, send only what's NEW since your last "
                        "``delegate_zcode`` call — ZCode already has the prior "
                        "exchange via ``--resume``. Don't restate."
                    ),
                },
                "mode": {
                    "type": "string",
                    "enum": ["build", "edit", "yolo"],
                    "description": (
                        "ZCode permission mode (``zcode --mode``). "
                        "``build``=most conservative, ``edit``=auto-apply "
                        "edits, ``yolo``=allow all. Default: inherits the "
                        "sharedagents session's current permission mode."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Optional, and rarely needed. ``zcode`` has NO "
                        "``--model`` flag: the model comes from the CLI's own "
                        "provider config "
                        "(``~/.zcode/v2/provider_config.json``). Pass a value "
                        "only to CONFIRM the id that config already selects "
                        "(e.g. ``glm-5.3``); any other id — including other "
                        "valid GLM ids — is refused, because this engine "
                        "cannot switch it. Omit to run the configured model."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Optional working directory for the spawned ``zcode`` "
                        "CLI. Absolute paths are used as-is; relative paths "
                        "resolve against sharedagents's launch cwd. Must exist and "
                        "be a directory or the call fails before spawning. "
                        "Omit to inherit sharedagents's launch directory."
                    ),
                },
            },
            required=["prompt"],
        ),
    ),
    Tool(
        name="delegate_agent",
        kind=ToolKind.NETWORK,
        description=(
            "Delegate one task to a SIBLING sharedagents agent. Each agent "
            "(``main``, ``scraper``, …) runs in its own home directory "
            "with its own SOUL/AGENTS/TOOLS context, its own session "
            "store, and may be wired to a different upstream model. "
            "Use this when the target agent is materially better "
            "suited for the task (e.g. ``scraper`` has browser tools "
            "and a domain-tuned prompt). The full reply comes back as "
            "the tool result.\n\n"
            "**Session continuity is automatic.** The target agent's "
            "session id is persisted at "
            "``<sessions_dir>/<sid>.delegate_agent.<name>.json`` and "
            "resumed on every subsequent call within one sharedagents "
            "session, so the target keeps the FULL prior delegate_agent "
            "conversation in its own context.\n\n"
            "**Critical: do NOT restate prior delegate_agent exchanges.** "
            "On a follow-up call, only forward the NEW context the "
            "target is missing. On the FIRST call, include the salient "
            "context (the task being asked, relevant findings) — that "
            "summarisation is what the target uses to ground its work."
        ),
        parameters=_params(
            {
                "name": {
                    "type": "string",
                    "description": (
                        "Target agent name (canonical). The runtime "
                        "also accepts ``agent`` / ``agent_id`` / "
                        "``agent_name`` / ``target`` / ``to`` as "
                        "synonyms so a first-attempt mis-key still "
                        "lands. Must match a configured agent under "
                        "``~/.sharedagents/agents/<name>``."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "The prompt to send the target (canonical). "
                        "The runtime also accepts ``prompt`` / "
                        "``message`` / ``input`` as synonyms. On the "
                        "FIRST call include salient context; on "
                        "follow-ups send only what's NEW since the "
                        "last call — the target keeps the prior "
                        "exchange via the resumed session."
                    ),
                },
                **_ASYNC_DELEGATE_PROPS,
            },
            required=["name", "task"],
        ),
    ),
    # NOTE: `mcp_list_tools` and `mcp_call` were previously in this
    # builtin tool list. They've been removed as model-facing tools
    # because the standard MCP pattern (Claude Desktop, Claude Code,
    # MCP Python SDK) is for the CLIENT to auto-discover tools at
    # startup via the `tools/list` JSON-RPC method and inject them
    # DIRECTLY into the model's tool registry as
    # ``mcp__<server>__<tool>``. The model picks them via standard
    # tool-use, not via meta-tools.
    #
    # The runner functions for `mcp_list_tools` / `mcp_call` are kept
    # in `tools/runners.py` for diagnostic / back-compat use only —
    # they're no longer surfaced to the model.
    Tool(
        name="acp_call",
        kind=ToolKind.NETWORK,
        description=(
            "Delegate a prompt to a remote ACP agent configured under "
            "`acp_agents`. Use when another agent better fits the task; it runs "
            "its own tools and returns its reply. ``mode``: ``plan`` is "
            "read-only, ``ask`` prompts before mutations, and ``auto`` permits "
            "autonomous mutations; default is the current session mode. "
            "Mutating unless the prompt is read-only with mode=plan; mark "
            "irreversible requests destructive."
        ),
        parameters=_params(
            {
                "agent_name": {
                    "type": "string",
                    "description": ("Short name in the configured `acp_agents` map."),
                },
                "prompt": {
                    "type": "string",
                    "description": ("Self-contained message; the remote cannot see this session's history."),
                },
                "mode": {
                    "type": "string",
                    "enum": ["plan", "auto"],
                    "description": ("Remote permission mode; defaults to current mode."),
                },
                "cwd": {
                    "type": "string",
                    "description": ("Optional working-directory hint for the remote."),
                },
            },
            required=["agent_name", "prompt"],
        ),
    ),
    # ── Browser tools removed ─────────────────────────────
    # The whole browser_* family used to live here as async wrappers
    # around sharedagents_core/tools/browser_tools.py. Both surfaces wrapped
    # the same nodriver-browser skill CLI, and two front doors for one
    # backend confused small local models (qwen-27b would flip between
    # calling browser_goto and Bash running browser.py mid-task).
    # The bundled nodriver-browser skill at
    # sharedagents_core/builtin_skills/nodriver-browser/ is now the only
    # browser surface — model invokes it as Bash →
    # ~/.sharedagents/skills/nodriver-browser/browser <action> [flags].
    # ─────────────────────────────────────────────────────────────────────
    Tool(
        name="goal",
        kind=ToolKind.WRITE,
        core=True,
        description=(
            "The session's autonomous goal. ``action``: ``set`` (objective in "
            "``text``; only when the user asked for a goal) or ``status``. Once "
            "set, the goal runs until its condition is met: after every turn the "
            "condition is checked against the conversation and you are sent "
            "back to work if it is not met. You cannot pause, clear or end it."
        ),
        parameters=_params(
            {
                "action": {
                    "type": "string",
                    "enum": ["set", "status"],
                    "description": "Goal action.",
                },
                "text": {
                    "type": "string",
                    "description": "The goal condition (action=set).",
                },
                "goal_id": {
                    "type": "string",
                    "description": "Target goal; omit for the session's goal.",
                },
            },
            required=["action"],
        ),
    ),
    Tool(
        name="budget",
        kind=ToolKind.EXEC,
        description=(
            "Set this session tree's cost budget (USD). Every session has a "
            "budget (default $25), goal or not — child spend rolls up to the "
            "root. ONLY call this when the user has EXPLICITLY authorized more "
            "budget in chat. action='topup' adds `usd` and resumes work paused "
            "for budget; action='set' sets the absolute budget (0 = $25 default)."
        ),
        parameters=_params(
            {
                "action": {
                    "type": "string",
                    "enum": ["set", "topup"],
                    "description": "set absolute budget, or topup (add + resume).",
                },
                "usd": {"type": "number", "description": "USD amount."},
                "goal_id": {"type": "string", "description": "Target goal; defaults to the active goal."},
            },
            required=["action", "usd"],
        ),
    ),
    # ── Loop tools ──────────────────────────────────
    # Cron/scheduling tools. They run on the LIVE in-process agent (the loop
    # registry and supervisor are in-process singletons) — the gateway
    # short-circuits them in tool_execution exactly like goal/budget, and
    # CoreBackend delegates to sharedagents_loop.loop_tools runners, which dispatch
    # to the shared loop_service backends.
    Tool(
        name="cron_create",
        kind=ToolKind.WRITE,
        description=("Schedule a prompt to fire on a 5-field cron (local time) — recurring or one-shot. Max 50 jobs."),
        parameters=cron_create_params(),
    ),
    Tool(
        name="cron_delete",
        kind=ToolKind.WRITE,
        description="Cancel a scheduled cron job by ID.",
        parameters=cron_delete_params(),
    ),
    Tool(
        name="cron_list",
        kind=ToolKind.READ,
        description="List this session's scheduled cron jobs.",
        parameters=cron_list_params(),
    ),
    Tool(
        name="schedule_wakeup",
        kind=ToolKind.WRITE,
        description=("Schedule when to resume the session's running loop; call with stop:true to end it immediately."),
        parameters=schedule_wakeup_params(),
    ),
    Tool(
        name="AgentSync",
        kind=ToolKind.NETWORK,
        core=True,
        description=(
            "Synchronous delegation: wait inline for every child. No "
            "background argument. Build and plan use this tool only."
        ),
        parameters=_AGENT_DISPATCH_PARAMS,
    ),
    Tool(
        name="AgentAsync",
        kind=ToolKind.NETWORK,
        core=True,
        description=(
            "Asynchronous delegation: return handles immediately. No background argument. Jarvis uses this tool only."
        ),
        parameters=_AGENT_DISPATCH_PARAMS,
    ),
)


# The ``delegate_*`` tools (delegate_claude / delegate_codex /
# delegate_antigravity / delegate_zcode / delegate_agent) are intentionally NOT
# exported in ``BUILTIN_TOOLS`` — they're sharedagents-isms with no Claude Code parity.
# The model sees ``AgentSync`` / ``AgentAsync``; there is no ``Agent`` tool.
# The shims (``shim_run_delegate_*``) remain importable for the
# slash-command path (``/claude``, ``/codex``, ``/antigravity``).
_HIDDEN_LEGACY_TOOL_NAMES = frozenset(
    {
        "delegate_claude",
        "delegate_codex",
        "delegate_antigravity",
        "delegate_zcode",
        "delegate_agent",
    }
)
BUILTIN_TOOLS = tuple(t for t in BUILTIN_TOOLS if t.name not in _HIDDEN_LEGACY_TOOL_NAMES)

_ONTOLOGY_DESCRIPTIONS = {
    "ontology_search": "Search accumulated research evidence, entities, concepts and definitions; inspect stored objects, timeline, publication history or readiness.",
    "ontology_source": "Fetch an HTTP(S) source and automatically save its returned text in the current research run. Begin with ontology_update first; extract entities and concepts from every saved source.",
    "ontology_update": "Begin research with its question; submit source-grounded entities, concepts and claims using source-local keys; finish with the answer, stored citation IDs and limitations. A native research turn requires a successful finish publication.",
}
BUILTIN_TOOLS += tuple(
    Tool(
        name=name,
        kind={"ontology_search": ToolKind.READ, "ontology_source": ToolKind.NETWORK, "ontology_update": ToolKind.WRITE}[
            name
        ],
        description=_ONTOLOGY_DESCRIPTIONS[name],
        parameters=schema,
    )
    for name, schema in _ontology_schemas().items()
)


# Legacy (pre-lean-agent-deck, Claude-Code-style TitleCase) builtin tool ids
# mapped to their OpenCode-aligned canonical ids. Used to (a) migrate persisted
# operator allowlists on upgrade, and (b) keep operator hook matchers written for
# the old names firing against the renamed tools. Keep in sync with the rename in
# `agent_profiles`/`runners`/this file.
LEGACY_TOOL_IDS: dict[str, str] = {
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "Bash": "bash",
    "Grep": "grep",
    "Glob": "glob",
    "TodoWrite": "todowrite",
    "AskUserQuestion": "question",
    "list_dir": "list",
    # SkillSearch was removed; the only skill tool is Skill (load-by-name).
    "SkillSearch": "Skill",
}


def canonicalize_tool_id(name: str) -> str:
    """Translate a legacy TitleCase builtin id to its canonical lowercase id;
    pass through anything already canonical / not a renamed builtin."""
    return LEGACY_TOOL_IDS.get(name, name)


@dataclass
class ToolRegistry:
    """Lookup table over a set of `Tool` objects, by name."""

    tools: tuple[Tool, ...] = BUILTIN_TOOLS

    def __post_init__(self) -> None:
        self._by_name: dict[str, Tool] = {t.name: t for t in self.tools}

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Return `tools[]` payload for `chat/completions`."""
        return [t.openai_schema() for t in self.tools]

    # ---- deferral helpers ---------------------------------------------

    def core_tools(self) -> tuple[Tool, ...]:
        """Tools flagged ``core``. NOT "always loaded into tools[]" — there
        is no core/deferred wire split; the permission set is
        the payload (see ``Tool``). Used only to shape prompt listings."""
        return tuple(t for t in self.tools if t.core)

    def deferred_tools(self) -> tuple[Tool, ...]:
        """Tools NOT flagged ``core``. NOT "kept off the wire until
        ``tool_search`` fetches them" — that mechanism is gone;
        a non-core tool in an agent's permission set ships its full schema
        every round. Used only to shape prompt listings."""
        return tuple(t for t in self.tools if not t.core)

    def filtered_schemas(self, allowed_names: set[str]) -> list[dict[str, Any]]:
        """Return `tools[]` payload restricted to ``allowed_names``.

        Order matches the canonical ``BUILTIN_TOOLS`` order so the
        prefix the upstream sees is byte-stable across rounds (the
        prefix-cache friendly path).
        """
        return [t.openai_schema() for t in self.tools if t.name in allowed_names]

    @staticmethod
    def _prompt_tool_summary(description: str, *, max_len: int = 110) -> str:
        """First sentence / line of a tool description for prompt listings."""
        short = (description or "").split(".")[0].split("\n")[0].strip()
        if len(short) > max_len:
            short = short[: max_len - 1] + "…"
        return short

    def core_tool_prompt_lines(
        self,
        *,
        allowed_names: set[str] | None = None,
    ) -> list[str]:
        """One ``- name: summary`` line per core tool for the system prompt."""
        lines: list[str] = []
        for t in self.core_tools():
            if allowed_names is not None and t.name not in allowed_names:
                continue
            summary = self._prompt_tool_summary(t.description or "")
            lines.append(f"- {t.name}: {summary}" if summary else f"- {t.name}")
        return lines

    def deferred_schemas_prompt_lines(
        self,
        *,
        allowed_names: set[str] | None = None,
    ) -> list[str]:
        """One line per deferred tool for the ``### Deferred Tool Schemas``
        subsection inside ``## Tooling`` (OpenClaw 2026.6.8 layout)."""
        lines: list[str] = []
        for t in self.deferred_tools():
            if allowed_names is not None and t.name not in allowed_names:
                continue
            short = self._prompt_tool_summary(t.description or "")
            lines.append(f"- {t.name}: {short}" if short else f"- {t.name}")
        return lines

    def deferred_catalog_block(self) -> str:
        """Legacy standalone ``# DEFERRED TOOLS`` block.

        Prefer folding deferred names into ``build_tooling_prompt_section``
        via ``deferred_schemas_prompt_lines``; this remains for callers that
        still expect the old heading.
        """
        deferred_lines = self.deferred_schemas_prompt_lines()
        if not deferred_lines:
            return ""
        lines = [
            "# DEFERRED TOOLS",
            "",
            "These tools are available to you and callable directly by name.",
            "",
            *deferred_lines,
            "",
        ]
        return "\n".join(lines)


# ---- MCP tool injection (standard pattern) ------------------------------


def _mcp_kind_for(nsname: str) -> ToolKind:
    """Criticality kind for a discovered MCP tool, from the server's declared
    annotations: ``readOnlyHint`` → READ, ``destructiveHint`` → WRITE
    (MCP's hint means side effects, not host DESTRUCTIVE), otherwise
    NETWORK (mutating). Lets a server opt its read-only tools out of the
    NETWORK/mutating flattening (so e.g. plan mode allows them). Unknown /
    un-annotated tools (incl. per-session ACP servers not in the main cache)
    fall through to NETWORK.
    """
    from sharedagents_mcp.mcp_client import _DISCOVERED_ANNOTATIONS

    ann = _DISCOVERED_ANNOTATIONS.get(nsname) or {}
    # Most-cautious hint wins: a tool declaring BOTH destructiveHint and
    # readOnlyHint (contradictory, but an external server can) must resolve
    # WRITE, not be downgraded to READ. Check destructive FIRST.
    if ann.get("destructiveHint"):
        return ToolKind.WRITE
    if ann.get("readOnlyHint"):
        return ToolKind.READ
    return ToolKind.NETWORK


def discovered_mcp_tools_sync() -> tuple[Tool, ...]:
    """Synchronous cache reader. Returns whatever ``discovered_mcp_tools``
    last loaded — empty tuple if no async caller has warmed the cache yet
    in this process. Use this from sync call sites (e.g. ``_build_worker_request``)
    that run inside the same event loop AFTER an async caller has run
    discovery."""
    from sharedagents_mcp.mcp_client import _DISCOVERED_TOOLS, mcp_tool_namespaced

    if not _DISCOVERED_TOOLS:
        return ()
    out: list[Tool] = []
    for server, tname, desc, schema in _DISCOVERED_TOOLS:
        props = (schema.get("properties") if isinstance(schema, dict) else {}) or {}
        required = (schema.get("required") if isinstance(schema, dict) else []) or []
        if not isinstance(props, dict):
            props = {}
        if not isinstance(required, list):
            required = []
        wrapped = _params(props, list(required))
        out.append(
            Tool(
                name=mcp_tool_namespaced(server.name, tname),
                kind=_mcp_kind_for(mcp_tool_namespaced(server.name, tname)),
                description=desc or f"MCP tool from server {server.name!r}.",
                parameters=wrapped,
                core=False,  # MCP tools deferred by default; ToolSearch promotes on demand
            )
        )
    return tuple(out)


def acp_session_mcp_tools_sync(session_id: str) -> tuple[Tool, ...]:
    """Sync reader for a session's ACP-supplied MCP tools.

    Mirrors :func:`discovered_mcp_tools_sync` but reads the per-session
    overlay cache (``_ACP_SESSION_DISCOVERED``) instead of the global
    config cache. ACP clients pass ``mcpServers`` per session in
    ``session/new``; these tools are additive on top of the operator's
    config servers and scoped to ``session_id``. Empty tuple when the
    session has no ACP servers (the common case)."""
    if not session_id:
        return ()
    from sharedagents_mcp.mcp_client import _ACP_SESSION_DISCOVERED, mcp_tool_namespaced

    rows = _ACP_SESSION_DISCOVERED.get(session_id) or []
    out: list[Tool] = []
    for server, tname, desc, schema in rows:
        props = (schema.get("properties") if isinstance(schema, dict) else {}) or {}
        required = (schema.get("required") if isinstance(schema, dict) else []) or []
        if not isinstance(props, dict):
            props = {}
        if not isinstance(required, list):
            required = []
        wrapped = _params(props, list(required))
        out.append(
            Tool(
                name=mcp_tool_namespaced(server.name, tname),
                kind=_mcp_kind_for(mcp_tool_namespaced(server.name, tname)),
                description=desc or f"ACP MCP tool from server {server.name!r}.",
                parameters=wrapped,
                core=False,  # MCP tools deferred by default; ToolSearch promotes on demand
            )
        )
    return tuple(out)


async def discovered_mcp_tools() -> tuple[Tool, ...]:
    """Return ``Tool`` objects for every tool advertised by every
    configured MCP server, named ``mcp__<server>__<tool>`` (Claude
    Code's namespacing convention). Returns an empty tuple if no
    servers are configured or all of them failed to start.

    Discovery is cached at the sharedagents_mcp module level — the first
    call spawns each server, subsequent calls reuse the cache. Call
    ``sharedagents_mcp.mcp_client.invalidate_discovery_cache()`` to force
    a re-discover (e.g. after editing the config).

    Each MCP tool is classified from its server annotations. The input schema
    is normalized through :func:`_params` without adding model-facing safety
    arguments.
    """
    from sharedagents_mcp.mcp_client import (
        discover_all_tools,
        mcp_tool_namespaced,
    )

    discovered = await discover_all_tools()
    out: list[Tool] = []
    for server, tname, desc, schema in discovered:
        # Normalize the server's inputSchema without adding synthetic fields.
        props = (schema.get("properties") if isinstance(schema, dict) else {}) or {}
        required = (schema.get("required") if isinstance(schema, dict) else []) or []
        if not isinstance(props, dict):
            props = {}
        if not isinstance(required, list):
            required = []
        wrapped = _params(props, list(required))
        out.append(
            Tool(
                name=mcp_tool_namespaced(server.name, tname),
                kind=_mcp_kind_for(mcp_tool_namespaced(server.name, tname)),
                description=desc or f"MCP tool from server {server.name!r}.",
                parameters=wrapped,
                core=False,  # MCP tools deferred by default; ToolSearch promotes on demand
            )
        )
    return tuple(out)
