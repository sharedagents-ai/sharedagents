"""OpenCode-inspired agent profiles for sharedagents (lean roster).

Each profile defines a permission mode and tool allowlist. The role / behavior
text lives in ``~/.sharedagents/agents/<name>/AGENTS.md`` (seeded from the shipped
``profile_agents/<name>/AGENTS.md`` templates). Routed via
``AgentAsync(subagent_type="explore")`` / ``AgentSync(...)`` etc.

Lean roster:
  Tier 1 — OpenCode originals: ``build`` (default primary, full tools), ``plan``
    (read-only primary), ``review`` (read-only primary — the verdict seat of the
    plan/build/review loop, writes ``review/`` only), ``explore`` + ``general``
    (subagents), and the hidden utilities ``compaction`` / ``title`` /
    ``summary`` / ``teacher``.
  Tier 2 — sharedagents additions: ``jarvis`` (sole async orchestrator — durable
    memory + delegation only, never edits directly) and ``research`` (the
    intelligence layer over a configured knowledge store). Code lookup goes to
    ``explore``. ``general`` is a bounded worker with no delegation tools.
    Among native workers, only ``plan`` and ``build`` may spawn.
  External engine delegates: ``claude`` / ``codex`` / ``cursor`` /
    ``antigravity`` / ``opencode``.

Only real renames remain in ``_ALIASES`` (``main``→``jarvis``, ``scout``→``explore``).
Unknown ``subagent_type`` values fail closed — they do not collapse onto ``build``.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentProfile:
    name: str
    mode: str
    description: str
    tools_allowlist: tuple[str, ...]
    tools_allowlist_strict: bool = True
    opencode_mode: str = "subagent"  # primary | subagent | hidden
    # Per-agent SKILL permission (narrows the skill catalog the model sees — helps
    # small models stay focused). Semantics mirror tools_allowlist: empty + strict =
    # NO skills; non-empty = ONLY those; empty + not-strict = all (back-compat).
    skill_allowlist: tuple[str, ...] = ()
    skill_allowlist_strict: bool = False
    # Hard denials applied on top of the allowlist / full-registry surface.
    # OpenCode expresses these as permission deny rules (e.g. general denies
    # ``todowrite``); sharedagents mirrors them via ``AgentConfig.disabled_tools``.
    disabled_tools: tuple[str, ...] = ()
    # DAL (Design Assurance Level) tier for this agent's permission surface.
    # Every roster agent carries a defined tier; the
    # identity/permission model itself is Tier B (plan/02 §2). An agent whose
    # *grant* reaches a higher-risk domain (e.g. deploy/ops touching a Tier A
    # surface) is still Tier B for its profile definition — the domain
    # risk is governed by the tier / review requirements, not this field.
    dal_tier: str = "B"
    # Path-scoped write enforcement (the follow-up named in the
    # PLAN_AUTHOR_TOOLS note below).
    #
    # ``None`` (default) => UNRESTRICTED: the agent's Write/Edit/Update tools
    # may target any path (normal agents are unaffected). A non-None tuple =>
    # RESTRICTED: file-mutating tools may write ONLY inside one of the listed
    # roots; a write anywhere else is REFUSED at the dispatch chokepoint,
    # fail-closed. Roots are directory prefixes; a relative root (e.g. "plan/")
    # is resolved against the session's workspace root (the repo the agent
    # operates in), an absolute root is used as-is, and a ``"**/<name>/"``
    # root is NAME-scoped: any directory with that name at any depth (the
    # plan profile's ``**/plan/`` = a ``plan/`` folder in any repo).
    # Enforcement — including
    # ``..``/symlink/absolute-path escape normalization — lives in
    # ``permissions.write_scope_check`` and is wired in ``runners.dispatch``.
    # An EMPTY tuple means "no writable root at all" (every write refused).
    write_roots: tuple[str, ...] | None = None
    # Optional deny-list, resolved the same per-repo way as ``write_roots``. A
    # write whose target lands inside ANY deny-root is REFUSED even when it also
    # falls inside an allowed root — the way an implementer (``build``) writes
    # source anywhere in the repo (``write_roots=(".",)``) EXCEPT the
    # governance books other agents own (plan/, requirements/, spec/), so
    # it can never clobber them. ``None`` /
    # empty ⟹ nothing denied.
    write_denyroots: tuple[str, ...] | None = None
    # Which ``subagent_type`` targets this profile may spawn. ``None`` ⟹ the
    # whole delegation roster (the historical behaviour, kept for orchestrators
    # and implementers). A tuple is AUTHORITATIVE and an empty tuple forbids
    # delegation entirely.
    #
    # A child does NOT inherit its parent's restrictions — it runs under its own
    # profile's grants. So without this, any profile holding AgentSync/AgentAsync
    # could spawn a more privileged one and act through it: read-only ``plan``
    # (write_roots=("plan/",)) could spawn ``build`` (write_roots=(".",)) or
    # ``general`` (unrestricted) and obtain write access it is denied directly.
    # Restricting the parent's own tools is not enough while the roster is open.
    delegates_allowlist: tuple[str, ...] | None = None
    # Per-profile pin: jarvis uses AgentAsync; build/plan use AgentSync.
    # ``"background"`` pins the
    # caller's Agent dispatches to background-only, ``"foreground"`` pins
    # them to foreground-only — overriding any explicit ``background`` value
    # the model passes (visible coercion diagnostic). ``None`` (default) =
    # unpoliced: legacy semantics (explicit boolean wins, else the per-agent
    # .env default, else foreground). The per-agent ``config.json``
    # ``delegation_mode`` field OVERRIDES this profile default (operator
    # editable via Agent Builder); plugin agents may declare their own via
    # frontmatter ``delegation_mode:``.
    delegation_mode: str | None = None


def profile_agents_bundle_root() -> Path:
    """Shipped defaults: ``sharedagents_core/profile_agents/<name>/AGENTS.md``."""
    return Path(__file__).resolve().parent / "profile_agents"


def profile_agents_template_path(profile_name: str) -> Path:
    return profile_agents_bundle_root() / profile_name / "AGENTS.md"


_STALE_AGENTS_MD_MARKERS = (
    "## OpenCode mapping",
    "OpenCode-style profile agent",
    "## Permissions",
    "## Context",
    "Runtime also injects the hardcoded **MODE** block",
)

# Exact digests of managed Jarvis templates eligible for one-time migration:
# the last template that described ``main`` as a live sibling, and the
# production template immediately before the capability-accuracy repair. A
# digest, rather than a phrase, is intentional: even a one-character operator
# edit makes the file operator-owned and therefore ineligible for rewrite.
_STALE_JARVIS_AGENTS_MD_SHA256 = frozenset(
    {
        # Previous knowledge-store routing template, before leaf worker limits.
        "0e70e4903e663d96117f38b4f16e7eb1d9087cf98600dfa0a86d32e07fc0df48",
        "92d064c82a456344391780fee349c957e432b252f13b5ffb8778d60fb2e9e2ec",
        "54025a86db6f1e09cface017cae1b554d868a842acfeb28929e1953db1ef604f",
        "307c8e19d2a7f7cff36ee6d5f263fb5335d7385d33c0126f7c33dabc0d1603ec",
        # shipped immediately before the 2026-09-14 batched-question restore
        "1f06e436511beb997e263e39961c2ffe4b51c24bef2ab32c8e7491e3381ff4ce",
        # shipped batched-question gate (replaced: OpenCode --auto)
        "a0867a1d1e58b54558e49b380c508e5a8745b17f26947e0943f1e6e2f76130a6",
        # production's pre-fast-flow nine-stage Jarvis prompt
        "6217f78060e20cc4741fadcd77ab6370557c5e4dbf9959bc75da8b9b3699b22f",
        # classify -> memory -> describe -> route fast-flow prompt
        "ae7259faf61ac66562c633d91714856b41fa4413b6502bdfca88889292a550bc",
        # shipped immediately before AgentAsync list-before-dispatch / wait-before-ask
        "1415a29cfcc35dc4fcaa328cd2efde2b5d5ca235ec919f11dd6f731cdd5b57c8",
        # 2026-10-12 pre-v2 stock template
        "d5b0b2f03e66ea933fa8c5e4674fc3032ddf5dab75c8d16ed4e37d13bce7a755",
        # 2026-10-12 pre-v2 live file
        "a9b70b3eefafa5c430585960ab8b63701f39a98f104e1daaf94ac7dc4e59f19e",
        # 2026-10-12 v2 stock template
        "d9029b48ca8e54df477fa23686f6e4b70df6f74443ef31d8bc77aedfe298b9dd",
        # 2026-10-16 v2.1 stock template (replaced: review/merge-gate rewrite)
        "03f6eda53b2813f5f4c71b970fb34480da6e36e7536ce5eb88e81ff4c7ccaffe",
        # 2026-10-17 v2.2 stock template (replaced: stop-then-steer CLI lesson)
        "290082fb1c301fffd778284632317f293ebfb282a845eba7c68a8d1df030464d",
        # 2026-10-20 one-wave-per-turn template (replaced: one wave per owner
        # prompt, continue-before-new, one browser agent)
        "52ba75af7d5db3609a8cfc7cddea9f9ce873828bde6fa639a809361f98423449",
        # 2026-10-27 native-review template (replaced: task/phase routing)
        "82e06c0723bc0135b5970f19c304e3196716f6d73f043fa1cd3fc278cb117f47",
        # 2026-10-27 task/phase routing template (replaced: generic local-model
        # wording, no deployment model named)
        "51062124d2d0f26b5daa44a8461bc6a9b824ac4f959cb080f4d15596ca71b491",
        # 2026-11-01 template (replaced: bounded routing and leaf workers)
        "5f313ed205d08842ced30ba0f1eb7703d3b141eaba416181b3625800b32c6b74",
    }
)
_STALE_BUILD_AGENTS_MD_SHA256 = frozenset(
    {
        "bcf893a55dc87daa406fb30febf76d178d3ce18650176cc5935dd388cf82e181",
        "b8fe7f8d8c2ef0064b7511a33bc02f2797675c9e9a9db2397334513f3f8f8947",
        "f24441dea47de45167ed50ebf33b0185cae3c790ae2da29534b8b9161459f955",
        # shipped immediately before the 2026-09-14 production-impact confirm rule
        "76ff655613da79eb5137dea8ba89021292cd9eed9e47471cc14446c627080540",
        # shipped permission-widget confirm (replaced: one batched question)
        "83a0c8c1957d31fe7d50ad3fd506dec27ac622bd2313e331f347c74e1914a029",
        # shipped batched-question gate (replaced: OpenCode --auto)
        "d79ccc6d037b6bacb22d7ef7f0b782f9ef387764ce1b688531bdaf955487a7b3",
    }
)
_STALE_PLAN_AGENTS_MD_SHA256 = frozenset(
    {
        # historical managed template
        "97fe1c50e78f4a0108ae2e5725c51d3dd0585a9d9ad654b1ddf52ca7325b415e",
        # shipped template immediately before the 2026-09-11 plan-write guidance
        "d9e288a5dfdafc8995138b7e47e4f58b73cb0a8560e968b2fdade4f6e30f7400",
        # Production's live plan prompt (shipped 2026-08-06 blob). It predates the
        # ``AgentSync`` rename and the plan-write guidance, and was NOT in this set,
        # so a 2026-09-11 restart left it stale. Exact bytes are pinned by
        # ``tests/fixtures/plan_agents_md_legacy_2026-08-06.md``.
        "4d750bc621ca6f12dc82daa34c92a6268b84685d9e0aaec6e7476f7cab05c1f4",
    }
)
# Other profiles retain only the historical generic digest behavior. Do not
# let a Jarvis-only production digest authorize a cross-profile rewrite.
_STALE_AGENTS_MD_SHA256 = frozenset(
    {
        "92d064c82a456344391780fee349c957e432b252f13b5ffb8778d60fb2e9e2ec",
    }
)


def _profile_agents_md_is_stale(
    body: bytes,
    *,
    profile_name: str,
) -> bool:
    """Return whether raw shipped prompt bytes are eligible for migration.

    Build, Plan, and Jarvis use profile-specific exact digests. Other managed
    profiles retain the legacy digest/marker recognition after the raw digest
    has been computed.
    """
    digest = hashlib.sha256(body).hexdigest()
    if profile_name == "build":
        return digest in _STALE_BUILD_AGENTS_MD_SHA256
    if profile_name == "plan":
        return digest in _STALE_PLAN_AGENTS_MD_SHA256
    if profile_name == "jarvis":
        return digest in _STALE_JARVIS_AGENTS_MD_SHA256
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return digest in _STALE_AGENTS_MD_SHA256
    return digest in _STALE_AGENTS_MD_SHA256 or any(marker in text for marker in _STALE_AGENTS_MD_MARKERS)


def scaffold_profile_agents_md(home: Path, profile: AgentProfile) -> Path:
    """Write ``AGENTS.md`` into *home* from the shipped template if missing.

    Never overwrites an existing file — operator edits survive.
    """
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    dest = home / "AGENTS.md"
    if dest.exists():
        return dest
    return _write_profile_agents_md_from_template(home, profile)


def sync_profile_agents_md(home: Path, profile: AgentProfile) -> bool:
    """Replace stale profile ``AGENTS.md`` boilerplate with the shipped template.

    Returns ``True`` when the file was rewritten. Skips non-stale operator edits.
    """
    home = Path(home)
    dest = home / "AGENTS.md"
    if dest.exists() and not dest.is_file():
        return False
    if dest.is_file():
        body = dest.read_bytes()
        stale = _profile_agents_md_is_stale(
            body,
            profile_name=profile.name,
        )
        # Build, Plan, and Jarvis accept only their own exact historical raw
        # bytes. Other profiles retain the legacy behavior that repairs an
        # empty managed prompt as well as recognized marker/digest boilerplate.
        if profile.name in {"build", "plan", "jarvis"}:
            if not stale:
                return False
        elif body.strip() and not stale:
            return False
    _write_profile_agents_md_from_template(home, profile)
    return True


def _write_profile_agents_md_from_template(home: Path, profile: AgentProfile) -> Path:
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    dest = home / "AGENTS.md"
    tpl = profile_agents_template_path(profile.name)
    if tpl.is_file():
        shutil.copyfile(tpl, dest)
    else:
        dest.write_text(
            f"{profile.description}.\n",
            encoding="utf-8",
        )
    return dest


def load_profile_agents_md(home: Path, profile: AgentProfile) -> str:
    """Read the profile system prompt from the agent-home ``AGENTS.md``.

    For the ``verify`` profile this is the Python seam that installs the
    synthesis-reuse permission and tiered owner-signoff policy into the live
    agent's system prompt.
    """
    home = Path(home)
    path = home / "AGENTS.md"
    if path.is_file():
        body = path.read_text(encoding="utf-8").strip()
        if body:
            return body
    scaffold_profile_agents_md(home, profile)
    if path.is_file():
        body = path.read_text(encoding="utf-8").strip()
        if body:
            return body
    tpl = profile_agents_template_path(profile.name)
    if tpl.is_file():
        return tpl.read_text(encoding="utf-8").strip()
    return profile.description


# ---------------------------------------------------------------------------
# Lean agent roster — OpenCode-inspired. OpenCode originals + jarvis.
# Only real renames remain in ``_ALIASES`` (main→jarvis, scout→explore).
# Tool ids here are sharedagents's current names; the literal opencode tool-id/
# behaviour alignment is a separate workstream.
# ---------------------------------------------------------------------------

# --- Tier 1: OpenCode Originals ---

# ``build`` is the opencode default agent: full registry access (non-strict).
BUILD_TOOLS: tuple[str, ...] = (
    "read",
    "write",
    "edit",
    "apply_patch",
    "glob",
    "grep",
    "list",
    "bash",
    "BashOutput",
    "KillShell",
    "AgentSync",
    "lsp",
    "Skill",
    "todowrite",
    "question",
    "goal",
    "x_tweets",
    "decision",
    # The default agent sees the loop/cron scheduling
    # surface (jobs + wakeups ride the in-process loop supervisor).
    "cron_create",
    "cron_delete",
    "cron_list",
    "schedule_wakeup",
)

BUILD = AgentProfile(
    name="build",
    write_roots=(".",),  # build: source/workspace root
    mode="auto",
    description=("The default agent. Full-access development — executes tools based on configured permissions."),
    tools_allowlist=BUILD_TOOLS,
    tools_allowlist_strict=False,  # full registry access (opencode default)
    opencode_mode="primary",
    # Bounded single-turn implementer: AgentSync waits inline.
    delegation_mode="foreground",
)

PLAN = AgentProfile(
    name="plan",
    # plan: any `plan/` folder — the model picks the operative repo and
    # writes `<repo>/plan/<topic>.md`; when no repo applies it writes
    # `~/.sharedagents/plan/<topic>.md` (both satisfy `**/plan/`).
    # Don't hard-gate the plan write.
    write_roots=("**/plan/",),
    mode="auto",  # native agents are auto-only; plan/ stays write_roots + allowlist
    description="Planner. Read-only investigation; authors plan documents under plan/ only.",
    tools_allowlist=(
        "glob",
        "grep",
        "read",
        "list",
        "bash",
        "BashOutput",
        "write",
        "edit",
        "AgentSync",
        "ExitPlanMode",
        "webfetch",
        "websearch",
        "x_tweets",
        "decision",
        "Skill",
        "todowrite",
        "question",
        "goal",
        # Graphiti pilot Phase 0.7 — read-only ontology lookups (plan F8).
        # The registered MCP surface must itself be read-only because the
        # build/general profiles intentionally retain full-registry access.
        "mcp__graphiti__search_memory_facts",
        "mcp__graphiti__get_episodes",
        "mcp__graphiti__get_episode_entities",
        "mcp__graphiti__get_status",
    ),
    tools_allowlist_strict=False,  # mirrors opencode/build — can delegate to subagents
    # Read-only investigation only. Plan writes just
    # plan/, so it must not be able to reach a writable child: `build` writes
    # the whole repo and `general` is unrestricted, and a child keeps its OWN
    # grants no matter what the request says. The prose boundary in plan's
    # AGENTS.md ("a Build child request must explicitly require read-only
    # investigation") is instruction-bound and was observed spawning build in
    # production. Explore is read-only; the external engines run out-of-process
    # under their own sandbox flags.
    # ``review`` is the verdict seat of the plan/build/review loop: plan may
    # summon an adversarial reviewer of its own artifact. Read-only, so it is
    # safe here for exactly the reason explore is.
    delegates_allowlist=(
        "explore",
        "review",
        "claude",
        "codex",
        "cursor",
        "antigravity",
        "opencode",
        "kimi",
        "hermes",
        "devin",
        "pi",
        "zcode",
        "qwen",
    ),
    opencode_mode="primary",
    # Read-only planner: AgentSync waits inline.
    delegation_mode="foreground",
)

# --- The review seat of the plan/build/review loop ---

# A reviewer must be able to *run* things: read the diff, run the tests it is
# judging, reproduce the claim under review. It is read-only in the same sense
# plan is — the guarantee is ``write_roots`` + ``delegates_allowlist`` at the
# dispatch chokepoint, not the absence of a write tool. ``write``/``edit`` are
# present solely so it can land its own verdict artifact, and
# ``write_roots=("**/review/",)`` confines every write to a ``review/`` folder.
# No ``apply_patch`` / ``KillShell`` / scheduling surface: no reason for a
# reviewer to mutate the tree or own jobs. No ``decision`` either: that tool
# scores context against named typed questions by calling a SECOND model, and
# a reviewer that outsources its verdict has stopped being the fresh-eyes seat
# (observed: four consecutive failed calls, ~4 wasted rounds, and a leaked
# tool-call serialization in the visible answer).
REVIEW_TOOLS: tuple[str, ...] = (
    "glob",
    "grep",
    "read",
    "list",
    "bash",
    "BashOutput",
    "write",
    "edit",
    "webfetch",
    "websearch",
    "x_tweets",
    "Skill",
    "todowrite",
    "question",
    "goal",
)

REVIEW = AgentProfile(
    name="review",
    # The ``plan/`` analogue: a ``review/`` folder in the operative repo, or
    # ``~/.sharedagents/review/<topic>.md`` when no repo applies — both satisfy
    # ``**/review/`` (same dual rule as PLAN above). An EMPTY tuple would be
    # wrong: the reviewer must be able to land its verdict artifact.
    write_roots=("**/review/",),
    mode="auto",
    description=("Adversarial reviewer. Read-only verdict on a diff, plan or implementation; writes review/ only."),
    tools_allowlist=REVIEW_TOOLS,
    tools_allowlist_strict=False,  # mirrors plan; write_roots is the real gate
    # A reviewer does the reading and the test run itself. Among native
    # workers only plan and build may spawn: a reviewer that can spawn
    # ``build`` or ``general`` becomes a writer by delegation, and a reviewer
    # that can spawn ``explore`` is a second orchestrator. Empty allowlist
    # plus the agent-tool denial is the gate; prose is not.
    disabled_tools=("AgentSync", "AgentAsync"),
    delegates_allowlist=(),
    # MUST stay "primary": the dataclass default is "subagent", which
    # gateway._discover_agents() drops, so the agent would never reach
    # /api/models and would be invisible in the WebUI.
    opencode_mode="primary",
    # Review does not delegate. The foreground pin stays so a future tool
    # grant cannot detach a child behind the verdict seat.
    delegation_mode="foreground",
)

# OpenCode explore allowlist (packages/opencode/src/agent/agent.ts): deny-all
# then allow grep/glob/list/bash/webfetch/websearch/read. ``BashOutput`` is the
# sharedagents companion for background bash output (not an edit privilege).
# ``lsp`` and ``Skill`` are intentionally omitted — OpenCode denies them.
EXPLORE_TOOLS: tuple[str, ...] = (
    "glob",
    "grep",
    "read",
    "list",
    "bash",
    "BashOutput",
    "webfetch",
    "websearch",
    "mcp__memory__session_search",
)

EXPLORE = AgentProfile(
    name="explore",
    write_roots=(),  # explore: read-only, NO writable root
    mode="auto",
    description=("Read-only code explorer. Locates and reports code. Never edits and never spawns."),
    tools_allowlist=EXPLORE_TOOLS,
    tools_allowlist_strict=True,
    # OpenCode explore: skill permission is denied (``"*": "deny"``).
    skill_allowlist=(),
    skill_allowlist_strict=True,
    # Leaf worker. Only plan and build may spawn, among native workers.
    delegates_allowlist=(),
    opencode_mode="subagent",
)

GENERAL = AgentProfile(
    name="general",
    write_roots=None,  # general: unrestricted
    mode="auto",
    description=("Bounded multi-step worker for local and web execution with its own tools. Never spawns subagents."),
    tools_allowlist=(),
    tools_allowlist_strict=False,  # full tools except the denials below
    # OpenCode general denies todowrite. AgentSync/AgentAsync are denied so a
    # research fan-out cannot nest another fan-out. The empty delegate roster
    # refuses a spawn even if a tool name leaks back onto the allowlist.
    disabled_tools=("todowrite", "AgentSync", "AgentAsync"),
    delegates_allowlist=(),
    opencode_mode="subagent",
)

RESEARCH_TOOLS: tuple[str, ...] = (
    "ontology_search",
    "ontology_source",
    "ontology_update",
    "websearch",
    "question",
    "todowrite",
) + tuple(
    "mcp__knowledge__" + name
    for name in (
        "coverage",
        "query",
        "find",
        "extraction_contract",
        "extraction_items",
        "verification_items",
        "get_record",
        "list_records",
        "get_evidence",
        "get_source_text",
        "create_question",
        "start_run",
        "finish_run",
        "record_hypothesis",
        "record_forecast",
        "record_decision",
        "record_outcome",
        "score_forecasts",
        "record_correction",
        "propose_update",
        "register_capacity_model",
        "run_scenario",
        "compare_scenarios",
        # ontology, typed objects and market data (read)
        "ontology",
        "get_object",
        "find_objects",
        "concept_labels",
        "price_at",
        "event_reaction",
        "market_impact",
        "action_log",
        # governed actions (writer only; validated and audit-logged with the agent's actor/model)
        "register_sources",
        "submit_extractions",
        "submit_verdicts",
        "build",
        "submit_links",
        "submit_same_checks",
        "publish_links",
        "bind_series",
        "resolve_forecast_from_market",
    )
)

RESEARCH = AgentProfile(
    name="research",
    mode="auto",
    write_roots=(),
    description="Research with automatically accumulated ontology evidence, entities, concepts, temporal claims and cited answers on SharedData.",
    tools_allowlist=RESEARCH_TOOLS,
    tools_allowlist_strict=True,
    skill_allowlist=(),
    skill_allowlist_strict=True,
    delegates_allowlist=(),
    disabled_tools=("AgentSync", "AgentAsync"),
    # Primary so gateway._discover_agents() publishes research to /openai/v1/models
    # and /api/models for direct operator/chat sessions as well as delegation.
    opencode_mode="primary",
    delegation_mode="foreground",
)

COMPACTION = AgentProfile(
    name="compaction",
    write_roots=(),  # hidden util: no file writes
    mode="plan",
    description="Hidden context-summarization utility.",
    # OpenCode compaction: ``"*": "deny"`` — no tools.
    tools_allowlist=(),
    tools_allowlist_strict=True,
    opencode_mode="hidden",
)

TITLE = AgentProfile(
    name="title",
    write_roots=(),  # title: backend written by runtime, not file tools
    mode="plan",
    description="Hidden session-title generator.",
    tools_allowlist=(),
    tools_allowlist_strict=True,
    opencode_mode="hidden",
)

TEACHER = AgentProfile(
    name="teacher",
    write_roots=(),  # hidden evaluator: no file writes or tools
    mode="plan",
    description="Hidden post-turn goal-completion evaluator.",
    tools_allowlist=(),
    tools_allowlist_strict=True,
    opencode_mode="hidden",
)

SUMMARY = AgentProfile(
    name="summary",
    write_roots=(),  # hidden recap utility: no file writes
    mode="plan",
    description=(
        "Hidden task-completion recap utility (Tasks board): recaps a "
        "finished task and judges whether it truly completed."
    ),
    tools_allowlist=(),
    tools_allowlist_strict=True,
    opencode_mode="hidden",
)

# --- Tier 2: sharedagents Additions ---

JARVIS_TOOLS: tuple[str, ...] = (
    "Skill",
    "AgentAsync",
    "mcp__memory__*",
    # Graphiti pilot Phase 0.7 — exact read-only grants for this strict profile.
    # Mutations and LLM-backed refreshes stay pipeline/manual-ops only.
    "mcp__graphiti__search_nodes",
    "mcp__graphiti__search_memory_facts",
    "mcp__graphiti__get_episodes",
    "mcp__graphiti__get_episode_entities",
    "mcp__graphiti__get_status",
    "mcp__graphiti__get_entity_edge",
    "question",
    "todowrite",
    "goal",
    "budget",
)

JARVIS = AgentProfile(
    name="jarvis",
    write_roots=(),  # thin orchestrator: no file writes
    mode="auto",
    description=(
        "Async orchestrator — the user's single point of contact. Answers "
        "directly or delegates to the right specialist. Never edits files or "
        "gathers data directly."
    ),
    tools_allowlist=JARVIS_TOOLS,
    tools_allowlist_strict=True,
    opencode_mode="primary",
    # Async orchestrator: AgentAsync always detaches.
    delegation_mode="background",
)

_PROFILES: dict[str, AgentProfile] = {
    p.name: p
    for p in (
        BUILD,
        PLAN,
        REVIEW,
        RESEARCH,
        EXPLORE,
        GENERAL,
        COMPACTION,
        TITLE,
        TEACHER,
        SUMMARY,
        JARVIS,
    )
}


# NOTE: ``general-purpose`` / ``gp`` are NOT profile aliases — they are dispatch
# routes handled in runners._shim_agent_canonical that select a target sibling by
# the ``name`` arg (default ``general``). Aliasing them to ``general`` here would make
# resolve_profile() (and any caller) silently substitute the general profile and
# overwrite a caller-supplied name.
#
# Only REAL renames live here. Unknown ``subagent_type`` values fail closed as
# unknown roster names — they do not silently collapse onto ``build``.
_ALIASES: dict[str, str] = {
    "scout": "explore",
    "main": "jarvis",
}


# A plugin-contributed subagent ships no explicit permission surface, so it
# resolves to a conservative read-only research allowlist by default (mirrors
# ``general``); a plugin author may widen or narrow it with a ``tools:``
# frontmatter list in ``agents/<name>.md``. The allowlist stays STRICT so the
# plugin agent never silently inherits a broader tool set than it declares.
_PLUGIN_AGENT_DEFAULT_TOOLS: tuple[str, ...] = (
    "read",
    "grep",
    "glob",
    "list",
    "bash",
    "BashOutput",
    "webfetch",
    "websearch",
    "x_tweets",
    "Skill",
    "todowrite",
)


def _coerce_tool_list(raw: object) -> tuple[str, ...]:
    """Parse a frontmatter ``tools`` value (list or comma/space string)."""
    if isinstance(raw, (list, tuple)):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    if isinstance(raw, str):
        import re as _re

        return tuple(t for t in _re.split(r"[,\s]+", raw.strip()) if t)
    return ()


def _plugin_agent_profile(key: str) -> AgentProfile | None:
    """Resolve an enabled plugin's ``agents/<key>.md`` into an AgentProfile.

    Plugin agents are auto-contributed by discovery: a plugin
    shipping ``agents/<name>.md`` is reachable by the runtime subagent resolver
    exactly like a built-in profile, with no manual registration. Built-in
    profiles win name collisions (resolved before this is called), so a plugin
    can never shadow a core SDLC agent.
    """
    try:
        from sharedagents_agent.plugins import plugin_agent_files
        from sharedagents_skill.skills import _parse_frontmatter
    except Exception:  # noqa: BLE001 — discovery must never break dispatch
        return None
    for path in plugin_agent_files():
        try:
            fm, _body = _parse_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        canonical = str(fm.get("name") or path.stem).strip().lower()
        if canonical != key:
            continue
        tools = _coerce_tool_list(fm.get("tools")) or _PLUGIN_AGENT_DEFAULT_TOOLS
        mode = str(fm.get("mode") or "plan").strip() or "plan"
        dm = str(fm.get("delegation_mode") or "").strip().lower()
        if dm not in ("background", "foreground", "default"):
            dm = ""
        profile = AgentProfile(
            name=canonical,
            mode=mode,
            description=str(fm.get("description") or "").strip(),
            tools_allowlist=tools,
            tools_allowlist_strict=True,
            opencode_mode="subagent",
            # Plugin agents may declare ``delegation_mode:`` in frontmatter
            # (background|foreground|default) — the same per-agent pin the
            # built-in profiles ship.
            delegation_mode=(dm or None),
        )
        return profile
    return None


def resolve_profile(name: str) -> AgentProfile | None:
    key = (name or "").strip().lower()
    if not key:
        return None
    key = _ALIASES.get(key, key)
    profile = _PROFILES.get(key)
    if profile is not None:
        return profile
    # Built-ins win; only fall through to plugin-contributed agents.
    return _plugin_agent_profile(key)


def list_profiles() -> list[AgentProfile]:
    return list(_PROFILES.values())


# The delegation roster the orchestrators (main/jarvis) choose from when they
# call ``AgentSync`` / ``AgentAsync``. ONE short, plain-language line per agent
# so a small model can pick the right target — this is what those tools advertise
# in its ``subagent_type`` schema and its "missing subagent_type" error. Ordered
# by workflow (read → research → plan/build → external
# engines → custom sibling). Keep the descriptions short and task-shaped; the
# authoritative permission set is still the profile itself.
#
# Excludes the orchestrators (main/jarvis — they delegate, they are not targets)
# and the hidden runtime utilities (compaction/title/summary/teacher).
_DELEGATION_ROSTER: tuple[tuple[str, str], ...] = (
    ("explore", "read-only code lookup; never edits or spawns"),
    ("research", "SharedData ontology: evidence, ingestion actions, market impact, forecasts; never spawns"),
    ("general", "bounded local and web execution with its own tools; never spawns"),
    ("build", "implement, verify, and delegate under its own grants"),
    ("plan", "investigate and publish one plan/ artifact"),
    ("review", "read-only adversarial verdict on a diff, plan or build"),
    ("claude", "deep multi-file refactor / review (ext)"),
    ("codex", "shell-heavy patch / pre-merge diff review (ext)"),
    ("cursor", "Cursor engine when user names it (ext)"),
    ("antigravity", "very large context / multimodal (ext)"),
    ("opencode", "OpenCode engine when user names it (ext)"),
    ("kimi", "Kimi Code CLI when user names it (ext)"),
    ("hermes", "Hermes Agent when user names it (ext)"),
    ("devin", "Devin CLI when user names it (ext)"),
    ("qwen", "Qwen Code CLI when user names it (ext)"),
    ("pi", "Pi RPC engine when user names it (ext)"),
    ("zcode", "ZCode (Z.AI GLM) CLI when user names it (ext)"),
    ("general-purpose", "custom sibling via `name` — last resort"),
)

# Names that are never advertised as custom extras (built-ins + non-targets).
_ROSTER_RESERVED_NAMES: frozenset[str] = frozenset(
    {name for name, _ in _DELEGATION_ROSTER} | {"jarvis", "main", "compaction", "title", "summary", "teacher"}
)


def _custom_on_disk_roster_entries() -> tuple[tuple[str, str], ...]:
    """Custom ``~/.sharedagents/agents/<name>/`` specialists (e.g. researcher on cursor).

    Appended to the static roster (before ``general-purpose``) so jarvis can
    route to operator-built agents that reuse an external engine. Discovery is
    live: add/remove a directory and the next Agent-tool schema build sees it.
    """
    import json

    try:
        from sharedagents_agent.agent import agents_root
    except Exception:  # noqa: BLE001
        return ()
    root = agents_root()
    if not root.is_dir():
        return ()
    out: list[tuple[str, str]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith(".") or name in _ROSTER_RESERVED_NAMES:
            continue
        cfg_path = entry / "config.json"
        if not cfg_path.is_file():
            continue
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        desc = str(data.get("description") or "").strip()
        engine = str(data.get("engine") or "").strip()
        if not desc:
            desc = f"custom specialist on {engine}" if engine else "custom on-disk specialist"
        # Keep schema lines short (tool-schema budget).
        if len(desc) > 72:
            desc = desc[:69].rstrip() + "…"
        out.append((name, desc))
    return tuple(out)


def delegation_roster() -> tuple[tuple[str, str], ...]:
    """``((name, one-line description), …)`` for every delegation target, in
    workflow order. Used by the Agent tool's ``subagent_type`` schema + error so
    the orchestrator sees the full roster with simple descriptions.

    Built-in entries are fixed; custom on-disk agents are inserted just before
    ``general-purpose`` so operator specialists (researcher on cursor, etc.)
    are first-class ``subagent_type`` values.
    """
    custom = _custom_on_disk_roster_entries()
    if not custom:
        return _DELEGATION_ROSTER
    head = _DELEGATION_ROSTER[:-1]  # everything before general-purpose
    tail = _DELEGATION_ROSTER[-1:]  # general-purpose last resort
    return head + custom + tail


def known_agent_names() -> frozenset[str]:
    """Every legitimate agent identity in this deployment, as a set.

    The union of:

    * delegation-roster names (built-in targets + live custom on-disk
      agents + ``general-purpose``),
    * profile identities (incl. the real renames ``main`` → ``jarvis`` /
      ``scout`` → ``explore`` and the hidden runtime utilities
      compaction/title/summary/teacher),
    * built-in agent defaults (e.g. ``self-improver`` — a deployable
      orchestrator identity before any per-agent config.json exists), and
    * every on-disk ``~/.sharedagents/agents/<name>/config.json``
      (operator-created agents and ``general-purpose`` custom siblings,
      which the Agent-tool delegation path routes through
      ``factory.load_agent``).

    Plugin-contributed agents are deliberately NOT enumerated here (that
    requires parsing every plugin agent file); the per-name predicate
    :func:`is_known_agent_name` resolves those via
    :func:`resolve_profile`. Used by the session-stamping boundary and
    the agent-monitor render filter so fixture-style phantom names can
    never enter the sessions store or display as real agents.
    """
    from sharedagents_agent.agent import _BUILTIN_AGENT_DEFAULTS, agents_root

    names: set[str] = {name for name, _ in delegation_roster()}
    names.update(_PROFILES.keys())
    names.update(_ALIASES.keys())
    names.update(_BUILTIN_AGENT_DEFAULTS.keys())
    try:
        root = agents_root()
        if root.is_dir():
            names.update(entry.name for entry in root.iterdir() if entry.is_dir() and (entry / "config.json").is_file())
    except OSError:
        pass
    return frozenset(names)


def is_known_agent_name(name: str) -> bool:
    """Whether *name* is a legitimate agent identity in this deployment.

    A name is known when ANY of:

    * it is a delegation-roster name (built-in targets + live custom
      on-disk agents + ``general-purpose``),
    * it resolves to a profile (incl. aliases ``main`` → ``jarvis`` /
      ``scout`` → ``explore``, hidden runtime utilities, and
      plugin-contributed agents),
    * it is a built-in agent default (e.g. ``self-improver``), or
    * it is resolvable via :func:`factory.load_agent` — an on-disk
      ``~/.sharedagents/agents/<name>/config.json`` (operator-created agents
      and ``general-purpose`` custom siblings, which the Agent-tool
      delegation path loads by name).

    Unknown names (fixture-style leaks like ``rt`` / ``reload-test``)
    return ``False`` so the session-stamping boundary
    (:func:`record_session_creation`) can refuse them loudly and the
    agent monitor can hide phantom rows.
    """
    nm = (name or "").strip()
    if not nm:
        return False
    if nm in known_agent_names():
        return True
    if resolve_profile(nm) is not None:
        return True
    # Belt-and-braces: the literal ``load_agent`` resolution the
    # delegation path uses, so the acceptance criterion "resolvable via
    # factory.load_agent" holds even for a name the cheap sources miss
    # (e.g. a future resolver that doesn't require the config dir).
    try:
        from sharedagents_agent.agents.factory import load_agent

        load_agent(nm)
        return True
    except Exception:  # noqa: BLE001 — any resolution failure = unknown
        return False


def delegation_roster_text(bullet: str = "- ") -> str:
    """The roster as a plain bulleted list: ``- <name> — <description>`` per line."""
    return "\n".join(f"{bullet}{name} — {desc}" for name, desc in delegation_roster())


def agents_md_first_paragraph(text: str) -> str:
    """First non-heading paragraph of an AGENTS.md body (``""`` when none).

    "Paragraph" = a run of consecutive non-blank lines that does not begin with
    a markdown heading. Leading headings, horizontal rules, block-quote/table
    rows and HTML comments are skipped; list markers are stripped and the
    lines are joined with a single space, because this string is rendered as a
    one-block description on a picker card, not re-emitted as markdown.
    """
    block: list[str] = []

    def _skip(line: str) -> bool:
        return (
            line.startswith("#")
            or line.startswith("---")
            or line.startswith(">")
            or line.startswith("|")
            or line.startswith("<!--")
            or line.startswith("```")
        )

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if block:
                break
            continue
        if _skip(line):
            if block:
                break
            continue
        if block and _skip(line):
            break
        for marker in ("- ", "* ", "+ "):
            if line.startswith(marker):
                line = line[len(marker) :].strip()
                break
        block.append(line)
    return " ".join(block).strip()


def agent_display_description(name: str, roster_line: str = "", *, root: Path | None = None) -> tuple[str, str]:
    """``(full_description, source)`` for a DISPLAY surface (agent picker card).

    Cards must never show a stub or an ellipsis, so
    the full text is resolved from the agent's own artifacts, in order:

    1. ``<agents_root>/<name>/config.json`` → ``description``   (source ``"config"``)
    2. first non-heading paragraph of ``<agents_root>/<name>/AGENTS.md``
       (source ``"agents_md"``)
    3. *roster_line* — the delegation-roster line              (source ``"roster"``)

    WHY THIS IS SEPARATE FROM ``_custom_on_disk_roster_entries()``: that helper
    cuts long custom descriptions to 69 chars + ``…`` for the **delegation
    tool-schema budget** (its 15-line bullet list ships in every Agent-tool
    call). The cut is correct THERE and must not be widened — it is wrong for
    the picker, which has a whole card to fill. This resolver deliberately
    re-reads the source instead of sharing the shortened string, so the two
    surfaces can never regress each other. Read-only: nothing is written.
    """
    base = root
    if base is None:
        try:
            from sharedagents_agent.agent import agents_root

            base = agents_root()
        except Exception:  # noqa: BLE001 — display data is never worth a raise
            return roster_line, "roster"
    import json

    try:
        cfg_path = base / name / "config.json"
        if cfg_path.is_file():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                desc = str(data.get("description") or "").strip()
                if desc:
                    return desc, "config"
    except Exception:  # noqa: BLE001
        pass
    try:
        md_path = base / name / "AGENTS.md"
        if md_path.is_file():
            para = agents_md_first_paragraph(md_path.read_text(encoding="utf-8", errors="replace"))
            if para:
                return para, "agents_md"
    except Exception:  # noqa: BLE001
        pass
    return roster_line, "roster"


def agent_display_descriptions(
    root: Path | None = None,
) -> dict[str, dict[str, str]]:
    """``{agent: {"text": …, "source": …}}`` — the FULL description per agent.

    The agent set and the roster fallback come from :func:`delegation_roster`
    (built-ins + live custom on-disk agents) plus :func:`list_profiles` (jarvis
    and friends), i.e. exactly the names a picker can render — so the rule
    above is applied once, server-side, and the client only paints the result.
    *root* overrides the agents directory (tests).
    """
    roster = dict(delegation_roster())
    try:
        for prof in list_profiles():
            roster.setdefault(prof.name, getattr(prof, "description", "") or "")
    except Exception:  # noqa: BLE001
        pass
    if root is None:
        try:
            from sharedagents_agent.agent import agents_root

            root = agents_root()
        except Exception:  # noqa: BLE001
            root = None
    out: dict[str, dict[str, str]] = {}
    for name, line in roster.items():
        text, source = agent_display_description(name, line, root=root)
        out[name] = {"text": text, "source": source}
    return out


def profile_write_roots(name: str) -> tuple[str, ...] | None:
    """Path-scoped write-root allowlist for the profile named *name*, or ``None``.

    ``None`` means either "no such profile" or "profile is unrestricted" — both
    collapse to *unrestricted* at the write gate, which is the safe default for a
    normal agent (its file tools are unaffected). A non-None tuple is the set of
    allowed write roots (possibly empty ⟹ no writable root). The dispatch
    chokepoint calls this with ``$SHAREDAGENTS_AGENT_PROFILE`` (falling back to
    ``$SHAREDAGENTS_AGENT_NAME``) and, when non-None, hands the roots to
    ``permissions.write_scope_check``.
    """
    prof = resolve_profile(name)
    return prof.write_roots if prof is not None else None


def profile_write_denyroots(name: str) -> tuple[str, ...] | None:
    """The profile's write DENY-list (roots refused even inside an allowed root),
    or ``None`` when the profile has none / does not exist. The dispatch
    chokepoint passes this alongside :func:`profile_write_roots` so an implementer
    scoped to the whole repo still cannot write another agent's governance book.
    """
    prof = resolve_profile(name)
    return prof.write_denyroots if prof is not None else None


def profile_delegates_allowlist(name: str) -> tuple[str, ...] | None:
    """Targets the profile named *name* may spawn, or ``None`` for unrestricted.

    ``None`` covers both "no such profile" and "profile is unrestricted", which
    collapse to the full roster — the safe default for an orchestrator or
    implementer whose delegation was never meant to be narrowed. A non-``None``
    tuple is authoritative; an empty tuple forbids delegation outright.
    """
    prof = resolve_profile(name)
    return prof.delegates_allowlist if prof is not None else None


def delegation_target_refusal(caller: str, target: str) -> str | None:
    """Why *caller* may not spawn *target*, or ``None`` when the spawn is allowed.

    Real renames (``scout``/``main``) are resolved on both sides. An unknown
    or empty caller is treated as unrestricted: identity is best-effort here,
    and a missing environment variable must not break delegation for every
    agent that legitimately has it.
    The restriction lives on the PROFILE, so an operator cannot widen it by
    editing an AGENTS.md.
    """
    allowed = profile_delegates_allowlist((caller or "").strip())
    if allowed is None:
        return None
    resolved_target = (target or "").strip().lower()
    canonical = _ALIASES.get(resolved_target, resolved_target)
    permitted = {_ALIASES.get(entry.strip().lower(), entry.strip().lower()) for entry in allowed}
    if canonical in permitted:
        return None
    caller_name = (caller or "").strip()
    if not permitted:
        return f"`{caller_name}` may not delegate: its profile permits no subagent targets."
    return (
        f"`{caller_name}` may not delegate to `{target}` — a child runs with "
        f"its OWN grants, so this would hand `{caller_name}` authority its own "
        f"profile denies it. Permitted targets: " + ", ".join(sorted(permitted)) + "."
    )


def resolve_profile_for_config(cfg: object) -> AgentProfile | None:
    """Return the OpenCode profile for *cfg*, if any."""
    key = str(getattr(cfg, "_agent_profile_name", "") or getattr(cfg, "agent_profile", "") or "").strip()
    return resolve_profile(key) if key else None


def profile_system_prompt_for_config(cfg: object) -> str | None:
    """Read the live ``AGENTS.md`` system prompt for a profile agent.

    Always reads from disk (seeds from the shipped template when the
    file is missing/empty). Operator edits take effect on the next
    compose — no in-process cache of the markdown body.
    """
    prof = resolve_profile_for_config(cfg)
    if prof is None:
        cached = getattr(cfg, "_agent_profile_system", None)
        return str(cached).strip() or None if cached else None
    home = Path(getattr(cfg, "home", "") or "")
    return load_profile_agents_md(home, prof)


def expand_tool_allowlist(names: Iterable[str], mcp_tool_names: Iterable[str]) -> set[str]:
    """Expand MCP **server grants** in a tool allowlist to concrete tool names.

    An allowlist may grant a whole MCP server instead of naming each tool:

      - ``mcp__*``               → every discovered MCP tool.
      - ``mcp__<server>__*``     → every tool of that server (prefix match).
      - ``mcp__<server>__<tool>``→ that exact tool (passes through unchanged).

    This is what makes "if an MCP server is allowed, all its tools are allowed"
    true under a strict allowlist: the grant tokens expand against the live
    ``mcp__<server>__<tool>`` names so the individual tools become promotable /
    callable / surfaced. Non-MCP names pass through untouched. The grant token
    itself is kept in the set (harmless — no real tool is named ``…__*``).
    """
    mcp = list(mcp_tool_names)

    def _server_of(tool_name: str) -> str | None:
        # Mirror parse_mcp_namespaced: the tool is the LAST ``__``-delimited
        # segment and the server is everything before it. Splitting on the
        # LAST ``__`` (not the first) is what prevents a scoped grant for
        # server ``a`` from over-matching server ``a__b``'s tools: those parse
        # as server ``a__b`` and never as server ``a``. A
        # ``__``-containing server name is never partial-prefix captured.
        if not tool_name.startswith("mcp__"):
            return None
        rest = tool_name[len("mcp__") :]
        return rest.rsplit("__", 1)[0] if "__" in rest else None

    out: set[str] = set()
    for n in names or ():
        if n == "mcp__*":
            out.update(mcp)
            out.add(n)
        elif n.startswith("mcp__") and n.endswith("__*"):
            grant_server = n[len("mcp__") : -len("__*")]  # mcp__<server>__* → <server>
            out.update(t for t in mcp if _server_of(t) == grant_server)
            out.add(n)
        else:
            out.add(n)
    return out


def compose_tool_allowlist(cfg: object, *, session_mcp_names: Iterable[str] = ()) -> set[str] | None:
    """The tools this agent may see, per its permissions — the SINGLE source of
    truth for permission-based tool presentation.

    The wire ``tools[]``, the dispatch gate, and the prompt all consume this:
    the permission set IS the tool set — there is no ``core``/``deferred`` split
    and no discovery step. Returns ``None`` when the full registry should be
    listed (a full-access agent: empty + non-strict allowlist AND no
    ``disabled_tools``; all built-in profiles are strict). Otherwise the exact
    allowlist with MCP server grants (``mcp__*`` / ``mcp__<server>__*``) expanded
    to concrete tool names — against both discovered and session-scoped MCP —
    minus ``disabled_tools``.
    """
    wl = getattr(cfg, "tools_allowlist", None)
    strict = bool(getattr(cfg, "tools_allowlist_strict", False))
    disabled = set(getattr(cfg, "disabled_tools", None) or [])
    # Canonicalize legacy TitleCase builtin ids at the COMPOSITION boundary so the
    # live tool set always matches the renamed registry — regardless of which
    # persisted config this is (managed, a leftover on-disk agent dir, or an
    # unmanaged custom config the boot-repair never touches). Covers BOTH
    # the allowlist AND disabled_tools (else a legacy ``"Bash"`` blocklist entry
    # would silently re-enable canonical ``"bash"``).
    from sharedagents_agent.tools.registry import (
        canonicalize_tool_id as _canon_id,
        is_agent_tool,
        rewrite_agent_tools_in_allowlist,
    )

    if isinstance(wl, list):
        wl = [_canon_id(t) if isinstance(t, str) else t for t in wl]
    disabled = {_canon_id(t) if isinstance(t, str) else t for t in disabled}
    # Local-model agents that already have the first-party Agent tool use its
    # native list/abort lifecycle actions. Do not also show the duplicate MCP
    # controls for those same operations. External ACP/CLI engines keep the MCP
    # surface; capture_pattern remains available because Agent has no equivalent.
    native_has_agent = str(getattr(cfg, "engine", "") or "") == "local_llm" and (
        (wl is None and not strict) or any(is_agent_tool(t) for t in (wl or ()))
    )
    if native_has_agent:
        disabled.update(
            {
                "mcp__delegate__check",
                "mcp__delegate__cancel",
            }
        )
    from sharedagents_agent.tools.registry import (
        BUILTIN_TOOLS,
        ToolRegistry,
        discovered_mcp_tools_sync,
    )

    mcp_names = [t.name for t in discovered_mcp_tools_sync()] + list(session_mcp_names or ())
    # Expand MCP server globs in the blocklist too (``mcp__dashboard__*`` →
    # every concrete dashboard tool) so an agent can be denied a whole MCP
    # server by one entry — otherwise the exact-name subtraction below never
    # matches the concrete ``mcp__<server>__<tool>`` names. Non-MCP names pass
    # through unchanged.
    if disabled:
        disabled = expand_tool_allowlist(disabled, mcp_names)
    if wl is None and not strict:
        # Full-access agent. With NO blocklist, signal "the whole registry"
        # with ``None`` (the wire emits every schema, the gate allows every
        # name). With a blocklist we must return an EXPLICIT set so the disabled
        # names are actually removed — returning ``None`` here would let a
        # ``disabled_tools`` entry slip past the dispatch gate, which expands
        # ``None`` to every tool (invariant: None ⟹ no blocklist).
        if not disabled:
            return None
        registry = ToolRegistry(BUILTIN_TOOLS)
        allowed = set(registry.names()) | set(mcp_names)
        return (
            rewrite_agent_tools_in_allowlist(
                allowed,
                getattr(cfg, "delegation_mode", None),
            )
            - disabled
        )
    allowed = expand_tool_allowlist(wl or [], mcp_names)
    if not strict:
        # Non-strict: MCP tools are available without an explicit per-server
        # grant. No implicit ``core_tools()`` union — the allowlist is the set.
        allowed |= set(mcp_names)
    if disabled:
        allowed -= disabled
    return rewrite_agent_tools_in_allowlist(
        allowed,
        getattr(cfg, "delegation_mode", None),
    )


def apply_profile_to_config(
    cfg: object,
    profile: AgentProfile,
    *,
    operator_keys: set[str] | None = None,
) -> None:
    """Mutate *cfg* in-place for a subagent dispatch (best-effort).

    The profile SEEDS the tool/skill allowlist — but only as an install-time default.
    ``operator_keys`` is the set of keys the operator explicitly set in the per-agent
    ``config.json``; the profile does NOT overwrite an allowlist the operator owns
    (the profile writes ``config.json`` + ``AGENTS.md`` at
    scaffold time, and from there the operator's config is authoritative for the
    allowlist). Pass ``operator_keys=None`` (the default, e.g. the unscaffolded-dispatch
    fallback) to always seed. ``permission_mode`` stays profile-owned regardless
    (the security floor).
    """
    from sharedagents_agent.subagent_policy import MEMORY_TOOL_NAMES

    _ok = operator_keys or set()
    if "tools_allowlist" not in _ok:
        if profile.tools_allowlist:
            cfg.tools_allowlist = list(profile.tools_allowlist)
            cfg.tools_allowlist_strict = profile.tools_allowlist_strict
        elif profile.tools_allowlist_strict and not profile.tools_allowlist:
            cfg.tools_allowlist = []
            cfg.tools_allowlist_strict = True
        else:
            # Empty + non-strict profile == FULL registry. No built-in profile
            # is empty+non-strict any more (all are strict); this path is for a
            # sibling agent scaffolded with an empty, non-strict allowlist.
            # Explicitly clear any restricted allowlist the cfg arrived with (e.g.
            # the missing-config fallback that applies one profile onto another's
            # cfg), else it would silently inherit a narrower surface.
            cfg.tools_allowlist = None
            cfg.tools_allowlist_strict = False
    else:
        # Operator owns the LIST. Normalize the STRICT flag so operator intent is
        # unambiguous regardless of a stale/absent ``tools_allowlist_strict`` (the
        # scaffold ships one; the operator may only touch the list). Explicit
        # ``strict`` set by the operator is honored as-is.
        #   list == null → UNRESTRICTED (all builtins + all MCP), never an empty set.
        #   non-null list + no explicit strict → EXACT list (no accidental MCP widening).
        if cfg.tools_allowlist is None:
            cfg.tools_allowlist_strict = False
        elif "tools_allowlist_strict" not in _ok:
            cfg.tools_allowlist_strict = True
    # Per-agent SKILL permission (mirrors tools_allowlist). Empty+strict => no skills.
    if "skill_allowlist" not in _ok:
        if profile.skill_allowlist:
            cfg.skill_allowlist = list(profile.skill_allowlist)
            cfg.skill_allowlist_strict = profile.skill_allowlist_strict
        elif profile.skill_allowlist_strict:
            cfg.skill_allowlist = []
            cfg.skill_allowlist_strict = True
        else:
            # Empty + non-strict skill profile == ALL skills. Default for
            # build/plan/general/jarvis. Explore is the exception: OpenCode
            # denies skill, so its profile sets ``skill_allowlist_strict=True``.
            # Mirror the tools branch above: explicitly clear any restricted skill
            # allowlist the cfg arrived with (e.g. a stale on-disk allowlist or the
            # missing-config fallback), else the agent would silently inherit a
            # narrower skill surface than the profile grants.
            cfg.skill_allowlist = None
            cfg.skill_allowlist_strict = False
    else:
        # Operator owns the skill list — same strict normalization as tools above.
        #   null → ALL skills; non-null list + no explicit strict → EXACTLY those.
        if cfg.skill_allowlist is None:
            cfg.skill_allowlist_strict = False
        elif "skill_allowlist_strict" not in _ok:
            cfg.skill_allowlist_strict = True
    cfg.permission_mode = profile.mode
    cfg._agent_profile_name = profile.name  # noqa: SLF001 — runtime hint
    # Delegation-mode default: the profile seeds only when the operator did
    # not set ``delegation_mode`` on disk (same operator-authoritative
    # pattern as the allowlists). ``None`` profile default ⇒ unpoliced.
    if "delegation_mode" not in _ok and profile.delegation_mode:
        cfg.delegation_mode = profile.delegation_mode
    # Runtime hint: which allowlist keys the operator owns on disk, so save() can
    # preserve them instead of re-deriving from the profile (see AgentConfig.save).
    cfg._operator_allowlist_keys = {  # noqa: SLF001 — runtime hint
        k for k in ("tools_allowlist", "skill_allowlist") if k in _ok
    }
    home = Path(getattr(cfg, "home", "") or "")
    cfg._agent_profile_system = load_profile_agents_md(home, profile)  # noqa: SLF001 — snapshot only
    disabled = set(getattr(cfg, "disabled_tools", None) or [])
    if profile.disabled_tools:
        disabled |= set(profile.disabled_tools)
    if not getattr(cfg, "memory_search_enabled", True):
        disabled |= MEMORY_TOOL_NAMES
    cfg.disabled_tools = sorted(disabled) if disabled else None


__all__ = [
    "AgentProfile",
    "BUILD",
    "PLAN",
    "EXPLORE",
    "GENERAL",
    "RESEARCH",
    "RESEARCH_TOOLS",
    "COMPACTION",
    "TITLE",
    "TEACHER",
    "JARVIS",
    "BUILD_TOOLS",
    "EXPLORE_TOOLS",
    "JARVIS_TOOLS",
    "compose_tool_allowlist",
    "apply_profile_to_config",
    "list_profiles",
    "profile_write_roots",
    "profile_write_denyroots",
    "load_profile_agents_md",
    "profile_agents_bundle_root",
    "profile_agents_template_path",
    "profile_system_prompt_for_config",
    "resolve_profile",
    "resolve_profile_for_config",
    "scaffold_profile_agents_md",
    "sync_profile_agents_md",
]
