"""Permission evaluator: combines mode + tool-kind + criticality → decision.

The user picks one of three permission modes per session:

- **plan** — read-only operations only. Anything not classified
  `read_only` is denied before the tool runs.
- **ask** — prompt before mutating / destructive operations. Three responses
  per prompt: `yes` / `yes-and-don't-ask-again-for-this` / `no`. The
  session allowlist remembers `yes-and-don't-ask-again` choices for the
  rest of the session — except for destructive operations, which always
  re-prompt as a safety net.
- **auto** — run without prompting, except destructive operations always
  pause for confirmation. That's the auto-mode safety net.

Criticality has two host-controlled sources plus one legacy escalation input:

1. The registered tool kind supplies a minimum severity.
2. A heuristic shell-command classifier (see `tools.safety`) classifies Exec.
3. A legacy/internal `args["criticality"]` claim may raise severity only;
   current model-facing schemas do not expose that field.

`PermissionEvaluator.evaluate` takes the most cautious result.
"""

from __future__ import annotations

import ntpath
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from sharedagents_core import oscompat

from .registry import Tool, ToolKind
from .safety import (
    CommandSafety,
    _split_command,
    _windows_argv,
    _windows_basename,
    _windows_delete_targets,
    classify_command,
)

# ---- Memory-tree path gate ------------------------------------------
#
# Wired by `privacy_gate_check` (below), which `runners.dispatch()` calls
# on every tool invocation before routing. Generic Write/Edit of
# MEMORY.md / memory/ is refused so mutations go through `remember`.


def is_memory_path(rel_path: str) -> bool:
    """True iff `rel_path` is `MEMORY.md` or under `memory/`.

    Inputs are relative-style strings (forward slashes acceptable).
    The caller is responsible for resolving absolute paths against
    the agent home — see `path_under_memory`.

    """
    if not rel_path:
        return False
    rel = rel_path.lstrip("/").replace(os.sep, "/")
    return rel == "MEMORY.md" or rel.startswith("memory/")


def path_under_memory(home: Path, raw_path: str) -> bool:
    """Resolve `raw_path` (handles `~` + relative) and return True iff
    the resolved path lives inside `<home>`'s memory tree.

    Defensive: returns False on any exception (caller can still treat
    the operation as untrusted; this helper is purely for the gate's
    yes-or-no decision).
    """
    if not raw_path:
        return False
    try:
        candidate = Path(str(raw_path)).expanduser()
        # Resolve symlinks; falls back to lexical join when the file
        # doesn't yet exist (write_file's case).
        try:
            candidate = candidate.resolve()
        except (OSError, RuntimeError):
            candidate = (Path.cwd() / candidate).resolve(strict=False)
        home_resolved = Path(home).resolve()
        rel = candidate.relative_to(home_resolved)
    except (ValueError, OSError):
        return False
    return is_memory_path(str(rel))


def exec_targets_memory(home: Path, command: str) -> bool:
    """Heuristic: True if a shell command string references a path
    under `<home>`'s memory tree.

    Two passes:

    1. **Token pass** — strip shell quotes and split into argv-like
       tokens (`shlex.split`). For each token, normalise common
       obfuscations (concatenated quoted segments are already merged by
       shlex; trailing slashes; leading `./` and `~/`) and check
       whether the resolved path lives under the agent home's memory
       tree. This catches `cat ./mem"ory"/foo.md` and `cat $HOME/.../memory/foo.md`
       (where `$HOME` was already expanded by the operator before
       handing it to the model).
    2. **Substring fallback** — for command strings that don't parse
       cleanly as shell (`cmd | grep`, command-substitution with
       backticks, etc.) we still want to refuse the obvious literal
       references to `MEMORY.md` / `memory/`.

    Not bulletproof — an obfuscated `> $(echo memory.md)` still slips
    through because `$(...)` is evaluated only at runtime — but the
    token pass closes the everyday cases the previous substring-only
    check missed. The model can still bypass deliberately, which is why
    this is a *deterrent + safety belt* on top of the permission gate;
    the gate's hard enforcement still runs in the dispatch loop.
    """
    if not command:
        return False
    cmd = str(command)
    home_path = Path(home).expanduser().resolve()
    memory_root = home_path / "memory"
    memory_md = home_path / "MEMORY.md"

    # Pass 1: token-level resolved-path check.
    import shlex

    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        # Unbalanced quotes — fall through to the substring pass.
        tokens = []
    for tok in tokens:
        if not tok:
            continue
        # Strip leading redirection operators ("> /path", "2>>/path").
        stripped = tok.lstrip("<>&|")
        if not stripped:
            continue
        # Skip obvious flags so `--memory=foo` (a flag value of "foo")
        # isn't treated as a path probe.
        if stripped.startswith("-"):
            continue
        try:
            cand = Path(stripped).expanduser()
            if not cand.is_absolute():
                cand = home_path / cand
            cand = cand.resolve(strict=False)
        except (OSError, ValueError, RuntimeError):
            continue
        try:
            cand.relative_to(home_path)
        except ValueError:
            continue
        try:
            cand.relative_to(memory_root)
            return True
        except ValueError:
            pass
        if cand == memory_md:
            return True

    # Pass 2: literal substring fallback for commands shlex can't parse.
    candidates = [
        str(memory_md),
        str(memory_root) + "/",
        " MEMORY.md",
        "/MEMORY.md",
        " memory/",
        "/memory/",
    ]
    return any(c in cmd for c in candidates)


# ---- Memory write chokepoint ----------------------------------------
#
# Generic Write/Edit of MEMORY.md / memory/ is refused so mutations go
# through `remember`. Reads are not gated.


def _resolve_path(raw_path: str) -> Path | None:
    """Best-effort absolute resolution (`~`, relative, symlinks). None on
    any failure so callers treat an unresolvable path as out-of-perimeter
    (the enclosing gate is fail-closed on origin, not on path shape)."""
    if not raw_path:
        return None
    try:
        cand = Path(str(raw_path)).expanduser()
        try:
            cand = cand.resolve()
        except (OSError, RuntimeError):
            cand = (Path.cwd() / cand).resolve(strict=False)
        return cand
    except (OSError, ValueError, RuntimeError):
        return None


def _perimeter_roots(agent_home: Any, sessions_dir: Any) -> list[Path]:
    """Concrete roots of the curated-memory privacy perimeter.

    Saved session JSONLs are deliberately not included: SharedAgents is a
    single-owner runtime and every native agent may inspect every transcript.
    ``sessions_dir`` remains in the signature for compatibility with callers.
    """
    _ = sessions_dir
    roots: list[Path] = []
    try:
        home = Path(agent_home).expanduser().resolve()
        roots.append(home / "MEMORY.md")
        roots.append(home / "memory")
    except (OSError, ValueError, RuntimeError, TypeError):
        pass
    return roots


def _session_store_root(sessions_dir: Any) -> Path | None:
    """Resolve the shared transcript store, or ``None`` when unstamped."""
    if not sessions_dir:
        return None
    try:
        return Path(sessions_dir).expanduser().resolve()
    except (OSError, ValueError, RuntimeError, TypeError):
        return None


def _is_under(cand: Path, root: Path) -> bool:
    if cand == root:
        return True
    try:
        cand.relative_to(root)
        return True
    except ValueError:
        return False


def path_in_privacy_perimeter(agent_home: Any, sessions_dir: Any, raw_path: str) -> bool:
    """True iff resolved `raw_path` lies *inside* the perimeter (the file
    itself is a curated-memory store).
    """
    cand = _resolve_path(raw_path)
    if cand is None:
        return False
    return any(_is_under(cand, r) for r in _perimeter_roots(agent_home, sessions_dir))


def path_in_session_store(sessions_dir: Any, raw_path: str) -> bool:
    """True iff ``raw_path`` resolves inside the shared transcript store.

    Reads are allowed for every native agent; this predicate exists only to
    retain append-only transcript integrity at generic write boundaries.
    """
    cand = _resolve_path(raw_path)
    root = _session_store_root(sessions_dir)
    return bool(cand is not None and root is not None and _is_under(cand, root))


def path_reaches_perimeter(agent_home: Any, sessions_dir: Any, raw_path: str) -> bool:
    """True iff an enumerate/walk rooted at `raw_path` would *cross* the
    perimeter — either the root is inside the perimeter, or a perimeter
    root is inside the walk (e.g. listing `<home>` reveals `memory/`).
    Grep/Glob/list_dir use this because their walk returns names/content.

    """
    cand = _resolve_path(raw_path)
    if cand is None:
        return False
    return any(_is_under(cand, r) or _is_under(r, cand) for r in _perimeter_roots(agent_home, sessions_dir))


# Tool-name → perimeter semantics (spec §4 per-tool table). Grouped so a
# future tool addition is covered by an explicit, auditable entry rather
# than silently bypassing the gate.
_WRITE_TOOLS = frozenset({"write", "write_file", "edit", "edit_file", "apply_patch"})

_WRITE_DENY = "refused: writes to MEMORY.md / memory/ must go through the `remember` tool."
_SESSION_WRITE_DENY = (
    "refused: direct writes to the append-only session transcript store are "
    "denied; session events must go through SessionWriter."
)


def _read_path(args: dict[str, Any]) -> str:
    # Accept BOTH the OpenCode canonical camelCase ``filePath`` (the registered
    # schema) and the legacy snake_case ``file_path``/``path`` — otherwise a
    # canonical call presents an EMPTY path to the write-scope + memory-privacy
    # gates, which then allow it (scope/privacy bypass). See runners' arg-alias
    # tolerance; the gate must recognize the same spellings the runner does.
    return str(args.get("filePath") or args.get("file_path") or args.get("path") or "")


def privacy_gate_check(
    tool_name: str,
    args: dict[str, Any],
    *,
    agent_home: Any,
    sessions_dir: Any = "",
    workspace_origin: str = "",
    chat_id: str = "",
) -> tuple[bool, str]:
    """The dispatch-chokepoint memory privacy gate.

    Returns ``(allowed, reason)``. ``allowed`` is True for any tool call
    that does not target the perimeter (no false positives — normal file
    ops are untouched). For calls that DO target the perimeter:

    - curated-memory **writes** and direct transcript Write/Edit calls → denied
      for every model-driven session (mutations go through ``remember``);
    - curated-memory **reads / enumerations / exec** → allowed.

    Never raises — a resolution failure yields "not perimeter" for the
    path axis.
    """
    try:
        args = args or {}
        # Transcript reads are shared, but their append-only integrity does not
        # depend on resolving agent_home. Enforce this axis first so a missing
        # memory-home stamp cannot reopen generic Write/Edit access.
        if tool_name in _WRITE_TOOLS:
            p = _read_path(args)
            if p and path_in_session_store(sessions_dir, p):
                return (False, f"{tool_name} {_SESSION_WRITE_DENY}")
        if not agent_home:
            # Without a home we cannot resolve the perimeter; the acp/
            # tool_runner path always stamps SHAREDAGENTS_AGENT_HOME. This is
            # the documented in-process residual (spec F9) — memory there
            # flows through the gated memory-shaped tools, not generic ones.
            return (True, "")
        if tool_name in _WRITE_TOOLS:
            p = _read_path(args)
            if p and path_in_privacy_perimeter(agent_home, sessions_dir, p):
                return (False, f"{tool_name} {_WRITE_DENY}")
            return (True, "")

        return (True, "")
    except Exception:  # noqa: BLE001 — gate must never crash dispatch
        if tool_name in _WRITE_TOOLS:
            return (False, f"{tool_name} refused: memory write-gate error.")
        return (True, "")


# Tools whose call carries a file path that is about to be MUTATED. The
# path-scoped write gate applies to exactly these (the structured file-mutation
# sinks where the target is a clean argument). NOTE: write-shaped ``Bash``
# (e.g. ``echo x > src/y``) is a DELIBERATE residual — parsing arbitrary shell
# for its write targets is out of scope here (the CLAUDE.md subprocess-boundary
# caveat), and belongs to the permission evaluator's mutating-command denial,
# not this path-containment check. Documented in the write-gate note below.
#
# Scoped to the tools that are ACTUALLY ROUTED to a mutation runner: only
# "Write" and "Edit" are in ``runners._SYNC_RUNNERS`` (and registered in
# ``registry.py``). "Update"/``run_update_file`` EXISTS but is NOT wired into
# ``_SYNC_RUNNERS``, so ``dispatch("Update")`` returns unknown before this gate
# ever runs — enforcing "Update" here would be a hollow claim (guarding a sink
# that can't execute). If "Update" (or a ``*_file`` alias) is ever wired into
# ``_SYNC_RUNNERS``, add it here too so the scope guard keeps covering every
# reachable mutation sink.
_SCOPED_WRITE_TOOLS = frozenset({"write", "edit"})


def _norm_realpath(raw: str, base: str) -> str:
    """Absolute, symlink-resolved, ``..``-collapsed path for containment tests.

    Relative *raw* is joined onto *base* first. ``os.path.realpath`` resolves
    symlinks in every existing component (so a symlink whose target escapes the
    allowed root is caught) and collapses ``..`` even for a not-yet-created leaf
    (a brand-new file resolves its existing parent chain). ``~`` is expanded so
    an agent can't tunnel out via ``~`` / ``$HOME``.
    """
    p = os.path.expanduser(str(raw))
    if not os.path.isabs(p):
        p = os.path.join(str(base), p)
    return os.path.realpath(p)


def _path_within(root_real: str, target_real: str) -> bool:
    """True iff *target_real* is *root_real* itself or nested beneath it.

    Both arguments must already be ``realpath``-normalized. The ``+ os.sep``
    guard prevents a sibling-prefix false positive (``/repo/plan`` must not
    match a write to ``/repo/plan-secrets/x``).
    """
    if target_real == root_real:
        return True
    return target_real.startswith(root_real.rstrip(os.sep) + os.sep)


def _any_dir_root_name(root: Any) -> str:
    """``"**/plan/"`` → ``"plan"``; ``""`` for a normal prefix root.

    A ``**/<name>/`` root is NAME-scoped: it matches a directory with that
    name at ANY depth (see :func:`write_scope_check`). Used by the plan
    profile so the planner can publish ``<any-repo>/plan/<topic>.md`` — and
    fall back to ``~/.sharedagents/plan/<topic>.md`` — without the repo-root
    resolution deciding for it.
    """
    s = str(root or "").strip()
    if not s.startswith("**/"):
        return ""
    name = s[3:].strip("/")
    if not name or "/" in name or name in (".", ".."):
        return ""
    return name


def _git_repo_root(target_real: str) -> str:
    """Nearest ancestor of *target_real* that is a git repository root (holds a
    ``.git`` entry — dir for a normal clone, file for a worktree/submodule), or
    ``""`` if none.

    A relative write-root like ``"plan/"`` is resolved against this so a
    path-scoped agent may write that folder in **whatever repository the target
    lives in** — not only a single configured workspace. The walk is bounded so a
    stray target can never scan the whole filesystem, and never raises.
    """
    try:
        d = os.path.dirname(str(target_real or ""))
        for _ in range(64):  # bound: no unbounded filesystem walk
            if not d or d == os.sep:
                return ""
            if os.path.exists(os.path.join(d, ".git")):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                return ""
            d = parent
        return ""
    except Exception:  # noqa: BLE001 — resolution helper must never crash the gate
        return ""


def write_scope_check(
    tool_name: str,
    args: dict[str, Any],
    *,
    write_roots: Any,
    base_dir: Any,
    deny_roots: Any = None,
) -> tuple[bool, str]:
    """Path-scoped write gate — the dispatch-chokepoint containment guard.

    Returns ``(allowed, reason)``. When the acting agent's profile restricts
    writes (``write_roots`` is non-None), a Write/Edit whose resolved
    target escapes every allowed root is REFUSED, fail-closed. Any other tool,
    and any agent whose ``write_roots`` is ``None``, is untouched (no false
    positives — normal agents' file tools are unaffected).

    Escape handling: the target and each root are ``os.path.realpath``-normalized
    against *base_dir*, so ``..`` traversal, an absolute path outside the root,
    and a symlink whose target leaves the root all resolve to a path that fails
    the containment test and is denied. A relative root ("plan/") is resolved
    against *base_dir* (the session workspace / repo root); an absolute root is
    used as-is. A ``"**/<name>/"`` root is NAME-scoped instead: it matches any
    directory with that name at any depth (``**/plan/`` → any ``plan/``
    folder), which is how the plan profile may publish in whatever repository
    the model selects, or in ``~/.sharedagents/plan/`` when none applies.

    ``write_roots`` semantics: ``None`` ⟹ unrestricted (allow). A non-None
    iterable ⟹ restricted to exactly those roots; an EMPTY one ⟹ no writable
    root (every scoped write refused). Never raises — a resolution error on a
    scoped write tool denies (fail closed); anything else allows.

    honored in the target's own git repository (see :func:`_git_repo_root`), so a
    path-scoped agent may write that folder in ANY repo it operates on — owner
    directive 2026-07-12 — while still refusing any target outside the root.
    """
    try:
        if write_roots is None and not deny_roots:
            return (True, "")  # unrestricted agent — do not touch its tools
        if tool_name not in _SCOPED_WRITE_TOOLS:
            return (True, "")  # not a file-mutation sink
        args = args or {}
        raw = _read_path(args)
        if not raw:
            # A write tool with no path — let the runner report the arg error;
            # there is no target to contain.
            return (True, "")
        roots = list(write_roots) if write_roots is not None else None
        base = str(base_dir or "")
        target_real = _norm_realpath(raw, base)
        # A RELATIVE root ("plan/") is honored in the target's OWN git repo, in
        # addition to the configured workspace base — so a path-scoped agent may
        # write that folder in ANY repository it operates on, not just one
        # workspace. An ABSOLUTE root is used as-is (base-independent).
        #
        # Containment guard against a nested .git: a repo root that sits STRICTLY
        # INSIDE a *valid* workspace base is NOT an allowed anchor — otherwise a
        # submodule/crafted `<base>/src/nested/.git` would let "plan/" resolve to
        # `<base>/src/nested/plan/`, escaping the outer `<base>/plan/`-only scope
        # (independent review, nested-.git finding). The target-repo fallback is
        # for when the base is UNUSABLE (empty / "/" / not a dir) — the actual
        # 2026-07-12 defect — and for a target in a DIFFERENT repo than the base.
        base_valid = bool(base) and os.path.isabs(base) and base.rstrip(os.sep) != "" and os.path.isdir(base)
        rel_bases = [base] if base_valid else []
        _repo_root = _git_repo_root(target_real)
        if _repo_root:
            _base_real = _norm_realpath(base, "") if base_valid else ""
            _nested_in_base = bool(_base_real) and _repo_root != _base_real and _path_within(_base_real, _repo_root)
            if not _nested_in_base:
                rel_bases.append(_repo_root)
        if not rel_bases:  # no valid base and no repo root → resolve raw for a clear refusal
            rel_bases = [base]
        # Deny-list takes PRECEDENCE over the allow-list: a target inside any
        # deny-root is refused even when it also falls inside an allowed root —
        # this is how an implementer scoped to the whole repo (".") is still kept
        # out of the governance books (plan/, requirements/, spec/, wiki/) it must
        # not clobber.
        #
        # Deny resolves against the SAME rel_bases as the allow-list — i.e. the
        # workspace/repo ROOT. The deny-list protects the top-level governance
        # books (<repo>/plan|requirements|spec|wiki); it deliberately does NOT
        # reach into nested subdirectories, so a nested source dir that merely
        # shares a name (e.g. a pip-style ``vendor/lib/requirements/base.txt``) is
        # NOT falsely refused (independent-review false-deny finding). A book
        # inside a nested submodule is a *different* repo's book and is left to
        # that repo's own governance — an accepted low-severity residual.
        for droot in deny_roots or ():
            droot_exp = os.path.expanduser(str(droot))
            if os.path.isabs(droot_exp):
                if _path_within(_norm_realpath(droot_exp, ""), target_real):
                    return (
                        False,
                        f"{tool_name} refused: the target {raw!r} is inside the "
                        f"denied path [{droot}] this agent may not write "
                        "(fail-closed deny-list).",
                    )
                continue
            for b in rel_bases:
                if _path_within(_norm_realpath(str(droot), b), target_real):
                    return (
                        False,
                        f"{tool_name} refused: the target {raw!r} is inside the "
                        f"governance book [{droot}] this agent may not write "
                        "(fail-closed deny-list).",
                    )
        if roots is None:
            # Deny-only config (no allow-list): passed the deny check ⟹ allowed.
            return (True, "")
        for root in roots:
            # ``**/<name>/`` — any directory with that NAME at any depth
            # (plan: ``**/plan/`` lets the planner write
            # ``<any-repo>/plan/<topic>.md`` and the fallback
            # ``~/.sharedagents/plan/<topic>.md``). The target is realpath-
            # normalized above, so a symlink pointing out of such a
            # directory cannot match.
            _name = _any_dir_root_name(root)
            if _name:
                if _name in target_real.split(os.sep)[:-1]:
                    return (True, "")
                continue
            # ``~/wiki/`` is home-scoped. Expand before the absolute check so
            # it is not joined onto the workspace as a literal ``~`` folder.
            root_exp = os.path.expanduser(str(root))
            if os.path.isabs(root_exp):
                if _path_within(_norm_realpath(root_exp, ""), target_real):
                    return (True, "")
                continue
            for b in rel_bases:
                if _path_within(_norm_realpath(str(root), b), target_real):
                    return (True, "")
        allowed = ", ".join(str(r) for r in roots) or "(none)"
        return (
            False,
            f"{tool_name} refused: this agent may write only within "
            f"[{allowed}] (resolved against {base or os.getcwd()}); the target "
            f"{raw!r} resolves outside every allowed root "
            "(fail-closed path-scope).",
        )
    except Exception:  # noqa: BLE001 — gate must never crash dispatch
        # A scoped write tool that blew up during resolution is denied (safety);
        # a non-write tool or fully-unrestricted agent is allowed. Fail closed
        # whenever EITHER an allow-list OR a deny-list is configured (a deny-only
        # config must not fail OPEN on a resolution error).
        if (write_roots is not None or deny_roots) and tool_name in _SCOPED_WRITE_TOOLS:
            return (
                False,
                f"{tool_name} refused: path-scope gate error.",
            )
        return (True, "")


def scoped_write_abspath(
    tool_name: str,
    args: dict[str, Any],
    *,
    write_roots: Any,
    base_dir: Any,
) -> str | None:
    """The absolute, realpath-normalized write target for a scoped write —
    the EXACT location :func:`write_scope_check` validated — or ``None`` when
    no rewrite applies.

    ``dispatch()`` calls this immediately AFTER ``write_scope_check`` passes and
    rewrites the tool arg's ``file_path`` to the returned path, so the
    ``run_write_file`` / ``run_edit_file`` runner writes to
    precisely the validated location — **independent of the tool-runner process
    cwd**. That closes the check-base ≠ write-base escape: without the rewrite a
    relative ``"plan/leak.md"`` is CHECKED against ``SHAREDAGENTS_WORKSPACE_CWD`` (the
    session workspace) but WRITTEN relative to the runner subprocess cwd (which
    ``acp_bridge`` does not ``chdir`` to the workspace), landing outside the
    validated root.

    Returns ``None`` when no rewrite is warranted: an unrestricted agent
    (``write_roots is None``), a non-file-mutation tool, or a call with no path
    argument. Normal agents are therefore untouched. Because the caller only
    reaches this after ``write_scope_check`` accepted the same path — which
    already ``realpath``-resolved and contained it against the same ``base_dir``
    — resolution here is the identical deterministic computation and cannot
    fail on input the check accepted (fail-closed is preserved by the check
    running first).
    """
    if write_roots is None:
        return None
    if tool_name not in _SCOPED_WRITE_TOOLS:
        return None
    raw = _read_path(args or {})
    if not raw:
        return None
    return _norm_realpath(raw, str(base_dir or ""))


def apply_patch_target_denied(
    target: str,
    *,
    write_roots: Any,
    base_dir: Any,
    agent_home: Any = "",
    sessions_dir: Any = "",
) -> tuple[bool, str]:
    """Validate one apply_patch target against write-scope + memory-privacy using
    BOTH the LEXICAL directory-entry path (symlinks NOT dereferenced) and the
    PHYSICAL realpath. Returns ``(denied, reason)``.

    Defends against the symlink-pivot: ``realpath`` alone dereferences a symlink,
    but ``delete``/``move`` act on the directory ENTRY, and a subsequent ``add``
    resolves lexically once the symlink is gone. So an entry inside the memory
    perimeter (or outside write_roots) must be refused even when its symlink
    target is in-scope — checking the lexical path catches that; checking the
    physical path catches the ordinary escape. Both must be in-scope + outside
    the perimeter.
    """
    import os.path as _osp

    def _within(child: str, parent: str) -> bool:
        child = _osp.normpath(child)
        parent = _osp.normpath(parent)
        return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)

    base = str(base_dir or "")
    base_real = _osp.realpath(base)  # resolve a SYMLINKED WORKSPACE up front (legit)
    exp = _osp.expanduser(str(target))
    joined_raw = exp if _osp.isabs(exp) else _osp.join(base, exp)
    # LEXICAL resolution against the REAL base — string ``..`` collapse, NO inner
    # symlink dereference. This is where the entry lives by name.
    lexical = _osp.normpath(exp if _osp.isabs(exp) else _osp.join(base_real, exp))
    # PHYSICAL: the OS resolution of the RAW path (follows EVERY symlink + ``..`` in
    # OS order — the location run_apply_patch actually writes/unlinks).
    physical = _osp.realpath(joined_raw)

    # (1) NO INNER SYMLINK TRAVERSAL. If the OS resolution differs from the lexical
    #     resolution (against the already-realpath'd base), a symlink BELOW the base
    #     was followed. apply_patch is multi-op: one op can delete a symlink and a
    #     later op's mkdir/write then resolves the same lexical path to a DIFFERENT
    #     physical location (an intra-patch TOCTOU that no static prevalidation can
    #     model — codex rounds 3-6). Refusing symlink traversal outright closes the
    #     whole class; plain ``..`` and a symlinked *workspace* are still allowed
    #     (both sides resolve equally). File-mutating tools should target real paths.
    if physical != lexical:
        return (
            True,
            (
                f"refused: apply_patch target {target!r} traverses a symlink; "
                "symlinked paths are not permitted in a patch."
            ),
        )

    # (2) Plain containment (no symlink now — lexical == physical).
    if agent_home:
        for pr in _perimeter_roots(agent_home, sessions_dir):
            if _within(lexical, str(pr)):
                return (
                    True,
                    (
                        f"refused: apply_patch target {target!r} lands inside the "
                        "operator's memory files; use the `remember` tool."
                    ),
                )
    session_root = _session_store_root(sessions_dir)
    if session_root is not None and _within(lexical, str(session_root)):
        return (
            True,
            (f"refused: apply_patch target {target!r} lands inside the append-only session transcript store."),
        )
    if write_roots is not None:
        # A relative root is honored in the target's own git repository as well
        # as in the workspace, exactly as write_scope_check does for write/edit:
        # the same agent must not be allowed to edit a file yet refused when it
        # patches it. A repository nested inside a valid workspace is not an
        # anchor, so a relative root cannot be widened by a crafted inner .git.
        rel_bases = [base_real]
        base_valid = bool(base) and _osp.isabs(base) and _osp.isdir(base)
        _repo_root = _git_repo_root(lexical)
        if _repo_root and not (base_valid and _repo_root != base_real and _within(_repo_root, base_real)):
            rel_bases.append(_repo_root)
        roots: list[str] = []
        any_dir_ok = False
        for r in write_roots:
            _name = _any_dir_root_name(r)
            if _name:
                if _name in _osp.normpath(lexical).split(_osp.sep)[:-1]:
                    any_dir_ok = True
                continue
            rr = _osp.expanduser(str(r))
            if _osp.isabs(rr):
                roots.append(_osp.normpath(rr))
                continue
            roots.extend(_osp.normpath(_osp.join(b, rr)) for b in rel_bases)
        if not any_dir_ok and not any(_within(lexical, r) for r in roots):
            return (
                True,
                (
                    f"refused: apply_patch target {target!r} escapes the agent's "
                    f"write roots [{', '.join(str(r) for r in write_roots) or '(none)'}] "
                    f"(resolved against {', '.join(rel_bases)})."
                ),
            )
    return (False, "")


# OpenCode / legacy sandbox spellings (plural ``plans/``).
_PLANS_DIR_MARKERS = ("/.sharedagents/plans/", "/plans/", ".sharedagents/plans/")
# Constitutional plan book (profile write_roots=("plan/",); AGENTS.md
# ``plan/<topic>.md``). Path-component match so ``plan/foo.md`` and
# ``/repo/plan/foo.md`` qualify; write_roots still enforces containment.
_PLAN_BOOK_RE = re.compile(r"(^|/)plan/.+\.md$", re.IGNORECASE)


def _is_plan_file_path(raw_path: str) -> bool:
    """True when *raw_path* is a plan-mode-writable plan markdown artifact.

    Covers both the OpenCode ``.sharedagents/plans/*.md`` sandbox and the
    sharedagents profile book ``plan/<topic>.md`` (the carve-out
    must match ``write_roots=("plan/",)`` or Plan cannot publish).
    """
    if not raw_path:
        return False
    norm = str(raw_path).replace("\\", "/").lower()
    if not norm.endswith(".md"):
        return False
    if any(m in norm for m in _PLANS_DIR_MARKERS):
        return True
    return bool(_PLAN_BOOK_RE.search(norm))


# Live runtime config that a write applies immediately (config-watch) or
# on the next gateway/orchestrator start. Editing these is a production
# change even though Write/Edit are only MUTATING at the tool-kind floor.
_PROD_WRITE_TOOLS = frozenset(
    {
        "write",
        "edit",
        "Write",
        "Edit",
        "Update",
        "apply_patch",
        "MultiEdit",
    }
)
_NONPROD_RM_ROOTS = ("/tmp/", "/var/tmp/", "/tmp", "/var/tmp")
_PROD_CONFIG_WRITE_MARKERS = (
    "json.dump",
    "sed -i",
    "tee ",
)
_PROD_OPEN_WRITE_RE = re.compile(r"open\([^)]*['\"]w")
_STACK_LIFECYCLE_RE = re.compile(
    r"\bsharedagents(?:[_-](?:agent|gateway|tui))?\b[^\n;|&]*?"
    r"\b(?:gateway|orchestrator)\s+(?:restart|stop|start)\b"
    r"|\bsharedagents\b[^\n;|&]*?\b(?:telegram|web)\s+(?:restart|stop)\b"
    r"|\bsystemctl\b[^\n;|&]*?\b(?:restart|stop|kill|disable|mask)\b"
    r"[^\n;|&]*?\bsharedagents",
    re.IGNORECASE,
)


def _sharedagents_home() -> Path:
    root = os.environ.get("SHAREDAGENTS_HOME", "").strip()
    return Path(root).expanduser() if root else (Path.home() / ".sharedagents")


def is_production_runtime_path(raw_path: str) -> bool:
    """True when *raw_path* is live sharedagents runtime config (not a repo file)."""
    if not raw_path:
        return False
    try:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve(strict=False)
        home = _sharedagents_home().expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    if path == home / "config.json" or path == home / ".env":
        return True
    try:
        rel = path.relative_to(home / "agents")
    except ValueError:
        return False
    parts = rel.parts
    return len(parts) == 2 and parts[1] == "config.json"


def _call_path(arguments: dict[str, Any] | None) -> str:
    a = arguments or {}
    return str(a.get("path") or a.get("filePath") or a.get("file_path") or a.get("target_file") or "")


def _is_nonprod_destructive_rm(command: str) -> bool:
    """``rm -r`` whose targets all live under /tmp, /var/tmp, or ``.worktrees/``.

    Those are scratch / task-branch trees, not the live production stack.
    """
    win_argv = _windows_argv(command)
    if win_argv and _windows_basename(win_argv[0]) in {"rd", "rmdir", "del", "erase"}:
        return _is_nonprod_windows_delete(command, win_argv)
    try:
        argv = __import__("shlex").split(command)
    except ValueError:
        return False
    if not argv:
        return False
    base = argv[0].rsplit("/", 1)[-1]
    if base != "rm":
        return False
    targets = [tok for tok in argv[1:] if not tok.startswith("-")]
    if not targets:
        return False
    for raw in targets:
        expanded = os.path.expanduser(raw)
        norm = expanded.replace("\\", "/")
        if "/.worktrees/" in f"/{norm.strip('/')}/" or norm.rstrip("/").endswith(".worktrees"):
            continue
        if norm == "/tmp" or norm.startswith("/tmp/"):
            continue
        if norm == "/var/tmp" or norm.startswith("/var/tmp/"):
            continue
        return False
    return True


def _is_nonprod_windows_delete(command: str, argv: list[str]) -> bool:
    """Windows scratch cleanup, using oscompat's resolved temp locations."""
    segments, writes, substitutions = _split_command(command)
    if len(segments) != 1 or writes or substitutions:
        return False
    targets = _windows_delete_targets(argv)
    if not targets:
        return False
    roots = [ntpath.normcase(ntpath.normpath(p)) for p in oscompat.temp_roots()]
    for raw in targets:
        norm = ntpath.normcase(ntpath.normpath(ntpath.expandvars(raw)))
        if ".worktrees" in norm.split("\\"):
            continue
        if any(norm == root or norm.startswith(root.rstrip("\\") + "\\") for root in roots):
            continue
        return False
    return True


def is_production_impact(tool_name: str, arguments: dict[str, Any] | None) -> bool:
    """True when executing this call would affect the live production system.

    Covers: sharedagents stack lifecycle, live ``~/.sharedagents`` runtime config
    writes, and destructive host ops that are not scratch-tree cleanup.
    """
    args = arguments or {}
    if tool_name in _PROD_WRITE_TOOLS and is_production_runtime_path(_call_path(args)):
        return True
    cmd = args.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return False
    if _STACK_LIFECYCLE_RE.search(cmd):
        return True
    if is_production_runtime_path(cmd) or ".sharedagents/config.json" in cmd.replace("\\", "/"):
        lowered = cmd.lower()
        if (
            any(marker in lowered for marker in _PROD_CONFIG_WRITE_MARKERS)
            or _PROD_OPEN_WRITE_RE.search(cmd)
            or re.search(r"(?:^|[;&|]\s*)(?:cp|mv)\b", cmd)
            or re.search(r">\s*\S*\.sharedagents/", cmd)
        ):
            return True
    safety = classify_command(cmd)
    return bool(safety is CommandSafety.DESTRUCTIVE and not _is_nonprod_destructive_rm(cmd))


class PermissionMode(Enum):
    """User-selected gating policy."""

    PLAN = "plan"  # read-only only
    ASK = "ask"  # prompt only for destructive ops
    AUTO = "auto"  # OpenCode --auto: allow unless denied; no permission widget


class Criticality(Enum):
    """Per-tool-call effect category. Set by the model and/or heuristic.

    ``FORBIDDEN`` is the hard "never even ask" class — the permission
    policy denies it in every mode, the modal never fires. See
    :class:`sharedagents_agent.tools.safety.CommandSafety.FORBIDDEN`.
    """

    READ_ONLY = "read_only"
    MUTATING = "mutating"
    DESTRUCTIVE = "destructive"
    FORBIDDEN = "forbidden"

    @property
    def severity(self) -> int:
        return _CRIT_SEVERITY[self]


_CRIT_SEVERITY: dict[Criticality, int] = {
    Criticality.READ_ONLY: 0,
    Criticality.MUTATING: 1,
    Criticality.DESTRUCTIVE: 2,
    Criticality.FORBIDDEN: 3,
}


# Map from CommandSafety (heuristic) to Criticality.
# - READ_ONLY  → READ_ONLY
# - MUTATING   → MUTATING
# - DESTRUCTIVE → DESTRUCTIVE
# - FORBIDDEN   → FORBIDDEN (hard-deny in every mode)
# - UNKNOWN     → MUTATING (cautious; can't promise read-only)
_SAFETY_TO_CRITICALITY: dict[CommandSafety, Criticality] = {
    CommandSafety.READ_ONLY: Criticality.READ_ONLY,
    CommandSafety.UNKNOWN: Criticality.MUTATING,
    CommandSafety.MUTATING: Criticality.MUTATING,
    CommandSafety.DESTRUCTIVE: Criticality.DESTRUCTIVE,
    CommandSafety.FORBIDDEN: Criticality.FORBIDDEN,
}


# Tools that change live process / job state but carry no ``command`` arg, so
# the EXEC command-heuristic can't catch a model claiming ``read_only``. They
# are floored to (at least) MUTATING regardless of the self-classification, so
# they can never run in plan mode / without approval as if read-only.
# ``cancel_delegation`` became ``mcp__delegate__cancel`` when it moved to MCP;
# the delegate server declares no readOnlyHint so it already resolves NETWORK→
# mutating, but keep the HARD name floor too (defense-in-depth: cancel kills a
# subagent's process group and must never resolve read-only regardless of what
# annotations a server declares).
_ALWAYS_MUTATING_TOOLS: frozenset = frozenset(
    {
        "cancel_delegation",
        "mcp__delegate__cancel",
        # EXEC-kind tools that carry no ``command`` to run the heuristic over, so
        # they'd otherwise get no kind-floor and a read_only claim would stick.
        # (The general EXEC-no-command floor in ``_effective_criticality`` also
        # covers these; naming them here is explicit defense-in-depth.)
        "KillShell",  # kills a live process group
        "budget",  # changes the agent's cost cap
    }
)

# These built-in HTTP tools only retrieve public data. Keep their NETWORK kind
# for transport/tooling parity while treating their effect as read-only in the
# permission matrix (not every outbound request mutates remote state).
_READ_ONLY_NETWORK_TOOLS: frozenset[str] = frozenset(
    {
        "webfetch",
        "websearch",
        "x_tweets",
    }
)

# Mixed-effect tools need argument-aware classification. The canonical Agent
# tool dispatches/aborts/steers (mutating) but ``action=list`` only reads the
# durable job registry.
_READ_ONLY_TOOL_ACTIONS: dict[str, frozenset[str]] = {
    "AgentSync": frozenset({"list"}),
    "AgentAsync": frozenset({"list"}),
}


def _is_read_only_tool_action(tool: Tool, args: dict[str, Any]) -> bool:
    actions = _READ_ONLY_TOOL_ACTIONS.get(tool.name)
    if not actions:
        return False
    return str(args.get("action") or "").strip().lower() in actions


def most_cautious(*claims: Criticality | None) -> Criticality:
    """Return the most cautious of the given non-None claims.

    Default if no claims given: MUTATING (better safe than read-only).
    """
    real = [c for c in claims if c is not None]
    if not real:
        return Criticality.MUTATING
    return max(real, key=lambda c: c.severity)


class Decision(Enum):
    """Outcome of `PermissionEvaluator.evaluate`."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class Response(Enum):
    """User response to an `ASK` prompt."""

    YES = "yes"
    YES_DONT_ASK = "yes_dont_ask"  # remember this pattern for the session
    NO = "no"


@dataclass
class PermissionEvaluator:
    """Per-session permission gate.

    Construct one when a session starts; call `evaluate(tool, args)` for
    every tool call before executing it. On `Decision.ASK`, the frontend
    surfaces a prompt; pass the user's answer back to `record(...)` if
    you want `YES_DONT_ASK` to be remembered.
    """

    mode: PermissionMode
    session_allowlist: set[str] = field(default_factory=set)

    def evaluate(
        self,
        tool: Tool,
        args: dict[str, Any],
    ) -> Decision:
        """Return ALLOW / ASK / DENY for one tool call.

        There is exactly ONE live implementation of
        the (criticality x mode) decision table. This offline evaluator no
        longer carries its own copy: it classifies the call's criticality
        (``_effective_criticality`` — the surviving classifier half) and then
        DELEGATES the decision to the single live table in
        ``permissions_integration.PermissionPolicy.evaluate``. The plan-file
        write carve-out and the ``Agent`` delegation carve-out now live in that
        single table too.

        The only local adaptation is rendering PermissionPolicy's two-hop
        plan-mode ``"ask"`` (which a plan-aware caller converts to a silent
        deny) as the hard ``DENY`` this evaluator's offline
        consumers assert; it has no ask_user hop of its own.
        """
        crit = self._effective_criticality(tool, args)
        # Lazy import avoids an import cycle (permissions_integration imports
        # helpers from this module).
        from ..agents.permissions_integration import PermissionPolicy

        decision, _reason = PermissionPolicy(
            mode=self.mode.value,
            allowlist=self.session_allowlist,
        ).evaluate(tool.name, args or {}, crit.value)

        if self.mode == PermissionMode.PLAN and decision == "ask":
            # No ask_user hop here — collapse the live gate's two-hop plan
            # "ask" to the hard deny the offline classifier promises.
            return Decision.DENY
        return Decision(decision)

    def record(
        self,
        tool: Tool,
        args: dict[str, Any],
        response: Response,
    ) -> None:
        """Log the user's response. `YES_DONT_ASK` adds to the allowlist.

        Called by the frontend after a prompt. `YES` and `NO` are no-ops
        here — they only affect the current call. `YES_DONT_ASK` remembers
        the pattern for the rest of the session.
        """
        if response == Response.YES_DONT_ASK:
            # Belt-and-braces: even if the caller flubbed it, we never
            # remember a destructive op.
            crit = self._effective_criticality(tool, args)
            if crit != Criticality.DESTRUCTIVE:
                self.session_allowlist.add(self._pattern_for(tool, args))

    # ---- internals ----

    def _effective_criticality(
        self,
        tool: Tool,
        args: dict[str, Any],
    ) -> Criticality:
        """Combine model claim + heuristic; pick the more cautious."""
        # 1. Legacy/internal escalation claim (current model schemas omit it).
        claimed: Criticality | None = None
        raw = args.get("criticality")
        if isinstance(raw, str):
            try:
                claimed = Criticality(raw)
            except ValueError:
                claimed = None

        # 2. Heuristic (only meaningful for exec).
        heuristic: Criticality | None = None
        if tool.kind == ToolKind.EXEC:
            cmd = args.get("command")
            if isinstance(cmd, str) and cmd:
                heuristic = _SAFETY_TO_CRITICALITY[classify_command(cmd)]

        read_only_action = _is_read_only_tool_action(tool, args)

        # A host-curated argument variant may be safer than the tool's coarse
        # whole-tool kind. This is exact-action scoped (Agent/list).
        if read_only_action:
            return most_cautious(claimed, heuristic, Criticality.READ_ONLY)

        # 3. Tool-kind floor: a WRITE-kind tool can't be classified read-only
        #    even if the model claims so — opening that door is a footgun.
        #    MCP ``destructiveHint`` maps here (side effects), not DESTRUCTIVE.
        if tool.kind == ToolKind.WRITE:
            kind_floor = Criticality.MUTATING
        elif (
            tool.kind == ToolKind.READ
            and claimed is None
            and heuristic is None
            and tool.name not in _ALWAYS_MUTATING_TOOLS
        ):
            return Criticality.READ_ONLY  # pure-read tool with no claim → safe
        else:
            kind_floor = None

        # State-changing tools without a command heuristic can't be read-only,
        # whatever the model claims (e.g. cancel_delegation kills runners).
        if tool.name in _ALWAYS_MUTATING_TOOLS:
            kind_floor = most_cautious(kind_floor, Criticality.MUTATING)

        # A NETWORK-kind tool is normally a real external side-effecting call —
        # Agent / acp_call, or an MCP tool WITHOUT a readOnlyHint. The two
        # built-in web retrieval tools are the explicit exception: their
        # transport is networked, but their operation is read-only.
        if tool.kind == ToolKind.NETWORK and tool.name not in _READ_ONLY_NETWORK_TOOLS and not read_only_action:
            kind_floor = most_cautious(kind_floor, Criticality.MUTATING)
        elif (
            tool.kind == ToolKind.NETWORK
            and (tool.name in _READ_ONLY_NETWORK_TOOLS or read_only_action)
            and claimed is None
            and heuristic is None
        ):
            return Criticality.READ_ONLY

        # An EXEC-kind tool whose call carries NO command to classify (``heuristic``
        # is None) — ``KillShell`` (SIGTERM/SIGKILL a process group), ``budget``
        # (change the cost cap), or any future arg-only exec tool. It DOES
        # something, so the model must not be able to self-classify it read_only
        # and skip the gate. Floor to MUTATING. (An EXEC tool WITH a command is
        # governed by ``classify_command`` above, which may legitimately return
        # READ_ONLY for e.g. ``git status`` — so only floor when heuristic is None.)
        if tool.kind == ToolKind.EXEC and heuristic is None:
            kind_floor = most_cautious(kind_floor, Criticality.MUTATING)

        # Live production config / stack lifecycle is DESTRUCTIVE even when
        # the tool kind is only WRITE/MUTATING, so ask-mode still prompts.
        # Auto mode allows these (OpenCode ``--auto``).
        if is_production_impact(tool.name, args):
            kind_floor = most_cautious(kind_floor, Criticality.DESTRUCTIVE)

        return most_cautious(claimed, heuristic, kind_floor)

    def _pattern_for(self, tool: Tool, args: dict[str, Any]) -> str:
        """Build the session-allowlist key for this tool call.

        - For `Bash`: tool name + argv[0] of the command. So saying
          `yes-don't-ask-again` for `git status` covers `git log`,
          `git diff`, etc., but not `rm` or `npm`.
        - For other tools: just the tool name (every call to `Read`
          shares one allowlist entry).
        """
        if tool.kind == ToolKind.EXEC:
            cmd = args.get("command", "")
            if isinstance(cmd, str) and cmd.strip():
                # Bare argv[0]; strip leading path. Match `tools.safety`'s logic.
                head = cmd.strip().split(None, 1)[0]
                head = head.rsplit("/", 1)[-1]
                return f"{tool.name}:{head}"
        return tool.name
