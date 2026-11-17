"""`sharedagents doctor` — let the agent diagnose & fix its own logged errors.

The agent writes structured failure records to `~/.sharedagents/errors/`
(see `errors.py`). This module reads those records, asks an LLM
(local model or Claude Code) to fix them, and — when a fix lands —
flips the file from `<ts>_<fp>_<slug>.md` to
`[fixed]_<ts>_<fp>_<slug>.md` with a resolution note appended.

## CLI shape (handled in `app.py`)

```
sharedagents doctor                          # auto-fix on a new branch errfix/<slug>
sharedagents doctor --plan                   # show plan, ask y/n, then fix on branch
sharedagents doctor --claude                 # use Claude Code as the fix engine
sharedagents doctor --no-branch              # skip branch creation (in-place fix)
sharedagents doctor --all                    # process every open error (default: 1)
sharedagents doctor <fingerprint-or-name>    # target one specific error
```

## Fix-engine fan-out

- **Local model** (default): one-shot via the agent's `_run_oneshot_via_agent`
  — the agent's tool dispatch handles read_file/edit_file/write_file/exec
  with the configured permission mode (plan/auto). The doctor command
  hard-overrides mode to `auto` for fix-execution, `plan` for plan-mode.
- **Claude Code** (`--claude`): one-shot via `_run_external_engine`. The
  permission-mode mapping is the same one the rest of sharedagents uses
  (plan/default/bypassPermissions).

## Branch policy

Without `--no-branch`, every fix lands on its own short-lived branch:
`errfix/<slug>` cut from the current branch. The doctor does NOT
auto-merge — the user reviews and merges by hand. Branch creation
happens AFTER plan approval (in `--plan` mode) so a rejected plan
leaves the working tree alone.

## Health mode (`--health` / `--check`)

`sharedagents doctor --health` is a DISTINCT mode from the error-repair loop:
it probes live system liveness (gateway, upstream model endpoint, open
error count), prints a compact status table, and — critically — returns
a NON-ZERO exit code when any critical service is DOWN. This exists
because the default error-repair `doctor` prints `(no open errors …)`
and exits `0` even on a fully-dead system, which reads as "healthy" to a
human. Health mode returns BEFORE entering the repair loop.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from sharedagents_core.errors import errors_dir, mark_fixed


@dataclass
class ErrorEntry:
    """One file under `~/.sharedagents/errors/` parsed for the doctor's use.

    `slug` is the short filename tag (e.g. `upstream-context-length-overflow`)
    — used for the branch name and the user-visible label.
    """

    path: Path
    fingerprint: str
    slug: str
    component: str
    exception: str
    summary: str  # the leading `> …` line under the H1, if any
    body: str  # full Markdown body (frontmatter included)


_FILENAME_RE = re.compile(r"^(?:\[fixed\]_)?(?P<ts>[0-9TZ:\-]+)_(?P<fp>[0-9a-f]{8})_(?P<slug>[a-z0-9\-]+)\.md$")


def parse_error_md(path: Path) -> ErrorEntry | None:
    """Parse one error file. Returns None if the filename doesn't match."""
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    body = path.read_text(encoding="utf-8")
    component = exception = summary = ""
    for line in body.splitlines():
        if line.startswith("component: "):
            component = line.split(": ", 1)[1].strip()
        elif line.startswith("exception: "):
            exception = line.split(": ", 1)[1].strip()
        elif line.startswith("> ") and not summary:
            summary = line[2:].strip()
    return ErrorEntry(
        path=path,
        fingerprint=m.group("fp"),
        slug=m.group("slug"),
        component=component,
        exception=exception,
        summary=summary,
        body=body,
    )


def list_open_errors() -> list[ErrorEntry]:
    """Return every unfixed error file, newest first."""
    d = errors_dir()
    if not d.is_dir():
        return []
    out: list[ErrorEntry] = []
    for p in sorted(d.glob("*.md"), reverse=True):
        if p.name.startswith("[fixed]_"):
            continue
        e = parse_error_md(p)
        if e is not None:
            out.append(e)
    return out


def find_error(target: str) -> ErrorEntry | None:
    """Look up one open error by fingerprint, exact filename, or substring match."""
    for e in list_open_errors():
        if target in (e.fingerprint, e.path.name) or target in e.slug:
            return e
    return None


# ----- prompts -----

# Stable header so prefix-cache reuse can pay off across multiple
# doctor invocations on related errors. The volatile per-error body
# is appended last (cache-friendly: see CLAUDE.md).
_FIX_PROMPT_HEADER = """\
You are sharedagents's self-repair agent ("doctor"). One of sharedagents's components has logged an error and you have been asked to fix it.

Operating principles:

- Read the error record carefully — fingerprint, exception, traceback, and context.
- Reproduce or trace the failure in the source tree (`src/sharedagents/`).
- Find the root cause; do not paper over symptoms.
- Apply a minimal, surgical fix. No refactors, no scope creep.
- If the fix needs a regression test, add one in `tests/`.
- Run `pytest -q tests/test_unit.py` after editing to confirm green.
- Keep changes inside `src/sharedagents/`, `tests/`, and `src/README.md`. Do not touch root `README.md` or `papers`.

Format your final answer as:

1. **Root cause** — one paragraph.
2. **Fix applied** — files + line ranges.
3. **Verification** — what you ran and the result.

If you cannot find a real bug (e.g. the error is environmental), say so plainly — do NOT invent a fix to clear the queue.
"""

_PLAN_PROMPT_HEADER = """\
You are sharedagents's self-repair agent ("doctor") in PLAN MODE. One of sharedagents's components has logged an error. Your job is to investigate and propose a fix plan — DO NOT EDIT ANY FILES YET.

Investigate by reading the relevant source files. Then write a numbered plan covering:

1. **Root cause** — where in the source the bug lives, and why it triggers.
2. **Proposed fix** — concretely, which lines change and how.
3. **Test strategy** — what regression test(s) to add or modify.
4. **Risk** — anything that could go wrong with this fix.

After your plan, the user will decide whether to apply it.
"""


def _error_block(entry: ErrorEntry) -> str:
    return (
        f"## Error to fix\n\n"
        f"- File: `{entry.path}`\n"
        f"- Fingerprint: `{entry.fingerprint}`\n"
        f"- Component: `{entry.component}`\n"
        f"- Exception: `{entry.exception}`\n"
        f"- Summary: {entry.summary or '(none)'}\n\n"
        f"### Full record\n\n"
        f"```markdown\n{entry.body}\n```\n"
    )


def build_fix_prompt(entry: ErrorEntry) -> str:
    return f"{_FIX_PROMPT_HEADER}\n\n{_error_block(entry)}"


def build_plan_prompt(entry: ErrorEntry) -> str:
    return f"{_PLAN_PROMPT_HEADER}\n\n{_error_block(entry)}"


# ----- branch management -----


def _git(*argv: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run `git` from the repo root (cwd at invocation time)."""
    return subprocess.run(["git", *argv], capture_output=True, text=True, check=check)


def _is_repo_clean() -> bool:
    r = _git("status", "--porcelain", check=False)
    return r.returncode == 0 and not r.stdout.strip()


def _current_branch() -> str:
    r = _git("rev-parse", "--abbrev-ref", "HEAD", check=False)
    return (r.stdout or "").strip()


def _branch_exists(name: str) -> bool:
    r = _git("rev-parse", "--verify", name, check=False)
    return r.returncode == 0


def _make_branch_name(slug: str) -> str:
    """`errfix/<slug>` — collision-resolved with a numeric suffix."""
    base = f"errfix/{slug or 'unknown'}"
    if not _branch_exists(base):
        return base
    n = 2
    while _branch_exists(f"{base}-{n}"):
        n += 1
    return f"{base}-{n}"


def create_branch(slug: str) -> str:
    """Create + check out a fresh `errfix/<slug>` branch.

    Refuses if the working tree is dirty so an in-flight edit doesn't
    get bundled into the doctor's commit.
    """
    if not _is_repo_clean():
        raise RuntimeError("working tree is dirty; commit or stash before running `sharedagents doctor`")
    name = _make_branch_name(slug)
    _git("checkout", "-b", name)
    return name


# ----- engines -----


async def _run_local_engine(prompt: str, *, mode: str, agent: str) -> int:
    """Drive the local model via the agent's one-shot path."""
    # Route via importlib so the import-linter contract stays
    # satisfied (sharedagents_gateway must not import sharedagents_tui at module
    # load).
    import importlib

    _app = importlib.import_module("sharedagents_tui.app")

    ns = argparse.Namespace(
        agent=agent,
        endpoint=None,
        model=None,
        system=None,
        temperature=None,
        enable_thinking=None,
        prompt=prompt,
        mode=mode,
        claude=False,
    )
    return await _app._run_oneshot_via_agent(ns, prompt)


async def _run_external_engine(
    engine_kind: str,
    prompt: str,
    *,
    mode: str,
) -> int:
    """Drive a registered CLI engine through its TUI one-shot adapter."""
    # Same import-linter boundary and importlib route as _run_local_engine.
    import importlib

    from sharedagents_agent.external_registry import spec_for

    _app = importlib.import_module("sharedagents_tui.app")
    spec = spec_for(engine_kind)
    runner = getattr(_app, f"_run_{spec.short_name}_oneshot", None)
    if runner is None:
        raise RuntimeError(f"doctor has no one-shot adapter for {spec.kind.value!r}")

    ns = argparse.Namespace(
        agent="jarvis",
        prompt=prompt,
        mode=mode,
        claude=spec.short_name == "claude",
    )
    return await runner(ns, prompt)


# ----- health probe -----
#
# Reuses the same liveness primitives the restart path uses, so "is the
# gateway up?" gives the same answer here as it does to `/restart`.
from sharedagents_core.restart import _gateway_running  # noqa: E402


@dataclass
class _Check:
    """One liveness check row for the `--health` status table.

    `status` is one of OK / WARN / DOWN. `critical` marks a check whose
    DOWN state must drive the overall exit code non-zero (a dead gateway
    or an unreachable upstream is a genuinely-down system; open errors
    are only advisory).
    """

    name: str
    status: str
    address: str
    detail: str
    critical: bool = False


def _gateway_address() -> str:
    """Bind URL the gateway `/health` probe targets.

    Mirrors the host/port resolution in `sharedagents_core.restart._gateway_running`
    (global config `gateway` block, defaulting to the gateway's own
    `127.0.0.1:8100`) so the address shown in the health table is exactly
    the one that was probed.
    """
    host, port = "127.0.0.1", 8100
    try:
        from sharedagents_agent.agent import load_global_config

        gw = (load_global_config() or {}).get("gateway") or {}
        host = str(gw.get("host") or host)
        port = int(gw.get("port") or port)
    except Exception:  # noqa: BLE001
        pass
    return f"http://{host}:{port}/health"


def _probe_gateway() -> _Check:
    """Probe north gateway liveness.

    Delegates the actual reachability decision to
    `sharedagents_core.restart._gateway_running` (live pidfile OR a successful
    `GET /health`) rather than reinventing it. "connection refused" =>
    DOWN; a live pid or an HTTP response => OK.
    """
    addr = _gateway_address()
    if _gateway_running():
        return _Check("gateway", "OK", addr, "reachable", critical=True)
    return _Check("gateway", "DOWN", addr, "no live pid and /health unreachable", critical=True)


def _find_bedrock_row(cfg: object) -> dict | None:
    """Return the first native Bedrock model row in ``cfg`` (or None)."""
    try:
        from sharedagents_model.bedrock_config import is_bedrock_row
    except Exception:  # noqa: BLE001
        return None
    for row in getattr(cfg, "models", []) or []:
        if isinstance(row, dict) and is_bedrock_row(row):
            return row
    return None


def _probe_bedrock_upstream(
    cfg: object,
    row: dict | None,
    endpoint: str,
    model: str,
) -> _Check:
    """Safe, OFFLINE Bedrock diagnostic — never a paid inference call.

    Health mode must not POST to ``bedrock://REGION/models`` (there is no such
    endpoint) or invoke a model. It reports whether the route is configured
    correctly and whether a credential that WOULD be used is resolvable, then
    points at the explicit opt-in live smoke test.
    """
    try:
        import os

        from sharedagents_model.bedrock_config import (
            AUTH_API_KEY,
            BEDROCK_BEARER_ENV,
            BedrockConfigError,
            validate_bedrock_row,
        )
    except Exception as e:  # noqa: BLE001
        return _Check(
            "upstream",
            "WARN",
            endpoint,
            f"Bedrock support module unavailable: {e.__class__.__name__}",
            critical=False,
        )

    if row is None:
        return _Check(
            "upstream",
            "WARN",
            endpoint or "(bedrock)",
            "Bedrock route configured but no matching model row (region/auth unverifiable offline)",
            critical=False,
        )
    try:
        route = validate_bedrock_row(row)
    except BedrockConfigError as e:
        return _Check(
            "upstream",
            "DOWN",
            endpoint or "(bedrock)",
            f"invalid Bedrock row: {e}",
            critical=True,
        )

    detail = (
        f"native Converse; region={route.region}; auth={route.auth}; "
        f"model={route.model_id}; not live-verified (run "
        "`sharedagents doctor --bedrock-smoke` to opt in)"
    )
    if route.auth == AUTH_API_KEY:
        source = str(row.get("apiKeyEnv") or row.get("api_key_env") or BEDROCK_BEARER_ENV).strip()
        api_keys = getattr(cfg, "api_keys", None) or {}
        present = bool(str(api_keys.get(source) or "")) if source in api_keys else bool(os.environ.get(source))
        if not present:
            return _Check(
                "upstream",
                "WARN",
                endpoint or "(bedrock)",
                detail + f"; credential {source} unset — will fail closed at turn",
                critical=False,
            )
    return _Check("upstream", "OK", endpoint or "(bedrock)", detail, critical=False)


def _probe_upstream() -> _Check:
    """Probe the configured upstream model endpoint via `GET /models`.

    Any HTTP response (200, 401, …) means the endpoint is reachable and
    is reported UP — a 401 auth gate is a live server, not a failure.
    Only a transport error (connection refused, DNS failure, timeout)
    counts as DOWN. An unconfigured endpoint is WARN, not DOWN: an
    install that never set an upstream is not a "down" system.
    """
    endpoint = model = ""
    cfg = None
    try:
        from sharedagents_agent.agent import AgentConfig

        cfg = AgentConfig.load("jarvis")
        endpoint = (cfg.upstream_endpoint or "").rstrip("/")
        model = cfg.upstream_model or ""
    except Exception:  # noqa: BLE001
        pass
    if not endpoint:
        return _Check(
            "upstream",
            "WARN",
            "(unconfigured)",
            "no upstream_endpoint in config",
            critical=False,
        )
    from sharedagents_model.bedrock_config import is_bedrock_row

    # Health describes the configured primary, not an arbitrary fallback or
    # vision row. Match both fields: same-region deployments can use different
    # models and credential modes. A missing match stays explicitly unverified.
    bedrock_row = next(
        (
            row
            for row in (getattr(cfg, "models", []) or [])
            if isinstance(row, dict)
            and is_bedrock_row(row)
            and str(row.get("baseUrl") or row.get("base_url") or "").rstrip("/") == endpoint
            and str(row.get("id") or "") == model
        ),
        None,
    )
    if bedrock_row is not None or endpoint.lower().startswith("bedrock://"):
        return _probe_bedrock_upstream(cfg, bedrock_row, endpoint, model)
    url = f"{endpoint}/models"
    try:
        import httpx as _httpx

        r = _httpx.get(url, timeout=2.0)
    except Exception as e:  # noqa: BLE001
        return _Check(
            "upstream",
            "DOWN",
            endpoint,
            f"unreachable: {e.__class__.__name__}",
            critical=True,
        )
    # Reachable — any HTTP status. Try to confirm the configured model is
    # served (best-effort; a 401/non-200 body just reports the status).
    detail = f"HTTP {r.status_code}"
    if model:
        detail += f"; model={model}"
    if r.status_code == 200:
        try:
            ids = [row.get("id") for row in (r.json().get("data") or [])]
            if model and model in ids:
                detail += " (served)"
            elif ids:
                detail += f"; serving {len(ids)} model(s)"
        except Exception:  # noqa: BLE001
            pass
    return _Check("upstream", "OK", endpoint, detail, critical=True)


def _probe_errors() -> _Check:
    """Count open error records (advisory, never critical).

    Reuses `list_open_errors()` — the same walk the repair loop uses — so
    the health table's open-error count matches what `sharedagents doctor`
    would actually process.
    """
    n = len(list_open_errors())
    status = "WARN" if n else "OK"
    detail = f"{n} open error(s)" + (" — run `sharedagents doctor`" if n else "")
    return _Check("errors", status, str(errors_dir()), detail, critical=False)


def run_health() -> int:
    """`sharedagents doctor --health`: probe live liveness, print a status
    table, and return an exit code that reflects reality.

    Returns non-zero iff any CRITICAL check (gateway or upstream) is
    DOWN. This is the entire point: a "doctor" that returns 0 on a dead
    system is the hazard this mode fixes — a human reads `rc=0` as
    "healthy" when nothing is actually running.
    """
    checks = [_probe_gateway(), _probe_upstream(), _probe_errors()]

    name_w = max(len(c.name) for c in checks)
    addr_w = max(len(c.address) for c in checks)
    print("sharedagents health\n")
    for c in checks:
        print(f"  {c.name:<{name_w}}  {c.status:<4}  {c.address:<{addr_w}}  {c.detail}")

    down = [c for c in checks if c.status == "DOWN" and c.critical]
    if down:
        print(f"\nDOWN: {', '.join(c.name for c in down)}")
        return 1
    print("\nall critical checks OK")
    return 0


# ----- main -----


def add_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire `sharedagents doctor` into the top-level CLI."""
    p = sub.add_parser(
        "doctor",
        help="Diagnose & auto-fix entries in ~/.sharedagents/errors/.",
        description=(
            "Walk unfixed error records and ask the agent (or Claude) to "
            "fix them. Default: auto-fix on a new errfix/<slug> branch. "
            "Use --plan to require human approval before any edits."
        ),
    )
    p.add_argument(
        "target",
        nargs="?",
        help="Fingerprint, filename, or slug substring (default: most recent open error).",
    )
    p.add_argument(
        "--health",
        "--check",
        dest="health",
        action="store_true",
        help=(
            "Health mode: probe live system liveness (gateway + upstream "
            "model endpoint + open-error count), print a status table, and "
            "exit non-zero if any critical service is DOWN. Does NOT enter "
            "the error-repair loop."
        ),
    )
    p.add_argument("--plan", action="store_true", help="Plan-only first; ask y/n before applying.")
    p.add_argument(
        "--claude",
        action="store_true",
        help="Use Claude Code as the fix engine instead of the local model.",
    )
    p.add_argument("--all", action="store_true", help="Process every open error in turn.")
    p.add_argument(
        "--no-branch",
        action="store_true",
        help="Apply the fix on the current branch (no errfix/<slug> branch).",
    )
    p.add_argument("--agent", default="jarvis", help="Local agent to use (default: jarvis).")
    p.add_argument(
        "--bedrock-smoke",
        dest="bedrock_smoke",
        action="store_true",
        help=(
            "Opt-in LIVE Bedrock Converse smoke test against the configured "
            "native Bedrock row. Makes a real (billable) inference call; "
            "never run by --health."
        ),
    )


def run_bedrock_smoke(agent: str = "jarvis") -> int:
    """`sharedagents doctor --bedrock-smoke`: explicit opt-in live Converse call.

    Deliberately separate from ``--health`` so a routine health check never
    incurs a paid inference call. Resolves the row's credential with the same
    per-subprocess helper a real turn uses (so it validates the auth path) and
    sends one bounded prompt.
    """
    try:
        from sharedagents_agent.agent import AgentConfig

        cfg = AgentConfig.load(agent)
    except Exception as e:  # noqa: BLE001
        print(f"error: could not load agent config {agent!r}: {e}", file=sys.stderr)
        return 2
    row = _find_bedrock_row(cfg)
    if row is None:
        print(
            "error: no native Bedrock row configured (api='bedrock-converse')",
            file=sys.stderr,
        )
        return 2
    from sharedagents_gateway.worker_request import run_bedrock_oneshot

    print(
        f"sending a bounded live Converse call to "
        f"{row.get('id')!r} (region={row.get('region')!r}, "
        f"auth={row.get('auth')!r}) …"
    )
    try:
        text, metrics = asyncio.run(
            run_bedrock_oneshot(
                cfg,
                row,
                [
                    {
                        "role": "user",
                        "content": "Reply with the single word: ok",
                    }
                ],
                max_tokens=16,
            )
        )
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1
    print(f"OK: {text.strip()[:200]!r}")
    if metrics:
        print(f"    tokens: prompt={metrics.get('prompt_tokens')} completion={metrics.get('completion_tokens')}")
    return 0


def dispatch(args: argparse.Namespace) -> int:
    """Entry point for `sharedagents doctor`.

    Health mode (`--health` / `--check`) is a distinct, synchronous path
    that returns BEFORE the error-repair loop; everything else is the
    original error-repair behavior, unchanged.
    """
    if getattr(args, "bedrock_smoke", False):
        return run_bedrock_smoke(getattr(args, "agent", "jarvis"))
    if getattr(args, "health", False):
        return run_health()
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    if args.target:
        entries = [e for e in [find_error(args.target)] if e is not None]
        if not entries:
            print(f"no open error matching {args.target!r} in {errors_dir()}")
            return 2
    else:
        entries = list_open_errors()

    if not entries:
        print(f"(no open errors in {errors_dir()})")
        return 0

    if not args.all and len(entries) > 1:
        entries = entries[:1]
        print(f"(found {len(list_open_errors())} open errors; processing the most recent — pass --all for the rest)")

    if args.claude and shutil.which("claude") is None:
        print("error: --claude requested but `claude` CLI is not on PATH", file=sys.stderr)
        return 2

    rc = 0
    for entry in entries:
        rc |= await _process_one(entry, args)
    return rc


async def _process_one(entry: ErrorEntry, args: argparse.Namespace) -> int:
    print(f"\n── doctor: {entry.slug} ({entry.fingerprint}) ──")
    print(f"   {entry.path}")
    if entry.summary:
        print(f"   {entry.summary}")

    # 1. PLAN PHASE (only if --plan)
    if args.plan:
        plan_prompt = build_plan_prompt(entry)
        print("\n== plan ==\n")
        if args.claude:
            rc = await _run_external_engine("claude_cli", plan_prompt, mode="plan")
        else:
            rc = await _run_local_engine(plan_prompt, mode="plan", agent=args.agent)
        if rc != 0:
            print("(plan engine returned non-zero; aborting fix)")
            return rc
        if not _ask_yn("\napply this plan? [y/N] "):
            print("(declined; leaving error open)")
            return 0

    # 2. BRANCH (unless --no-branch)
    starting_branch = _current_branch()
    branch_name: str | None = None
    if not args.no_branch:
        try:
            branch_name = create_branch(entry.slug)
            print(f"   created branch: {branch_name}")
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    # 3. FIX PHASE
    fix_prompt = build_fix_prompt(entry)
    print("\n== fix ==\n")
    try:
        if args.claude:
            rc = await _run_external_engine("claude_cli", fix_prompt, mode="auto")
        else:
            rc = await _run_local_engine(fix_prompt, mode="auto", agent=args.agent)
    except Exception as e:  # noqa: BLE001
        print(f"engine error: {e}", file=sys.stderr)
        rc = 1

    if rc != 0:
        print("(fix engine returned non-zero; not marking error fixed)")
        if branch_name and starting_branch and starting_branch != branch_name:
            print(f"   leaving you on {branch_name} for inspection (was {starting_branch})")
        return rc

    # 4. MARK FIXED
    fixed_path = mark_fixed(
        entry.path,
        resolution=(
            f"Auto-fixed by `sharedagents doctor`"
            f"{' (claude)' if args.claude else ''}"
            f"{' on branch ' + branch_name if branch_name else ''}."
        ),
    )
    print(f"\n   marked fixed: {fixed_path.name}")
    if branch_name:
        print(f"   review and merge: git diff {starting_branch}..{branch_name}")
    return 0


def _ask_yn(prompt: str) -> bool:
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")
