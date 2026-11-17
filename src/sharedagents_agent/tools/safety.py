"""Host-side shell-command safety classifier.

Model-facing schemas do not carry a ``criticality`` field. This module is the
authoritative effect classifier for shell commands; the permission evaluator
combines its verdict with the registered tool kind. Legacy/internal callers may
still supply a claim, but it can only raise severity, never lower this verdict.

This file ships with a hand-curated, conservative ruleset. It is **not**
a sandbox — it doesn't prevent malicious commands from running, it just
makes the permission gate aware of what's about to run. A user in `auto`
mode running `rm -rf /` will still get a confirmation prompt because the host
classifier flags it before execution.
"""

from __future__ import annotations

import fnmatch
import ntpath
import os
import re
import shlex
from enum import Enum
from pathlib import Path

from sharedagents_core import oscompat


class CommandSafety(Enum):
    """Heuristic classification of a shell command.

    Values are ordered from least to most cautious so callers can take
    `max(a, b, key=CommandSafety.severity)` to merge two classifications.

    ``FORBIDDEN`` is a hard "never even ask" class for commands whose
    blast radius makes any approval flow a bug — ``rm -rf ~/``,
    ``rm -rf /``, fork bombs, raw-disk overwrites of a system device.
    The permission evaluator short-circuits these to ``deny`` in EVERY
    mode (including ``auto`` + session allowlist) so a model never has
    a path to convince a half-attentive operator into running them.
    """

    READ_ONLY = "read_only"  # ls, cat, git status, grep — won't change anything
    UNKNOWN = "unknown"  # not on either allow- or deny-list
    MUTATING = "mutating"  # mkdir, touch, mv, sed -i — reversible-ish writes
    DESTRUCTIVE = "destructive"  # rm -rf <subdir>, dd, drop table, force-push to main
    FORBIDDEN = "forbidden"  # rm -rf ~/, rm -rf /, fork bomb — never even ask

    @property
    def severity(self) -> int:
        return _SEVERITY[self]


_SEVERITY: dict[CommandSafety, int] = {
    CommandSafety.READ_ONLY: 0,
    CommandSafety.UNKNOWN: 1,
    CommandSafety.MUTATING: 2,
    CommandSafety.DESTRUCTIVE: 3,
    CommandSafety.FORBIDDEN: 4,
}


# Commands whose mere invocation can't change state (no flags that mutate).
# Conservative: a single argv[0] match, no flag inspection beyond what's
# checked below for `git`. Add to this list with care — a "read-only" tag
# is a permission for plan-mode users to run the command unprompted.
_READ_ONLY_COMMANDS: frozenset[str] = frozenset(
    {
        # Inspection
        "ls",
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "wc",
        "file",
        "stat",
        "tree",
        "tac",
        "nl",
        "fold",
        "fmt",
        "column",
        "diff",
        "cmp",
        # Search
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "ack",
        "ag",
        "find",
        "locate",
        "which",
        "whereis",
        # Process / system inspect
        "ps",
        "top",
        "htop",
        "free",
        "df",
        "du",
        "uptime",
        "who",
        "w",
        "id",
        "whoami",
        "hostname",
        "uname",
        "groups",
        "tty",
        "lsof",
        "lsblk",
        "lscpu",
        "lspci",
        "lsusb",
        "dmesg",
        # Network inspect (no mutations)
        "ping",
        "traceroute",
        "dig",
        "nslookup",
        "host",
        "ip",
        "ifconfig",
        "netstat",
        "ss",
        "arp",
        "route",
        # Date / env
        "date",
        "cal",
        "env",
        "printenv",
        "locale",
        # Help / docs
        "man",
        "info",
        "help",
        "type",
        "command",
        "alias",
        "history",
        # Compression inspect-only — `tar` / `unzip` are NOT here because
        # the classifier doesn't inspect their flags: `tar -xf foo.tar`
        # extracts and `unzip foo.zip` unpacks, both of which are mutating.
        # The classifier returns UNKNOWN (→ MUTATING) for them so the
        # permission evaluator prompts the user.
        "zcat",
        "bzcat",
        "xzcat",
        "zless",
        "bzless",
        # Python read-only invocations
        "pwd",
        # Shell builtins / no-ops that never touch the filesystem. ``cd``
        # changes only the shell's own working dir; ``echo`` / ``printf``
        # write to stdout (a real file target is caught by the redirect
        # floor, not here); ``test`` / ``[`` evaluate; ``true`` / ``false``
        # / ``:`` are no-ops. These show up constantly in read-only research
        # chains (``cd repo && git log``, ``test -f x && echo found``) and
        # their absence was denying plan-mode research.
        "cd",
        "echo",
        "printf",
        "test",
        "[",
        "true",
        "false",
        ":",
        "sleep",
        # Text-stream filters — read stdin / files and print to stdout.
        # (``sort -o FILE`` / ``split`` can write; ``sort -o`` is guarded
        # explicitly in ``_classify_segment``; ``split`` / ``csplit`` /
        # ``tee`` are intentionally NOT here.)
        "sort",
        "uniq",
        "cut",
        "tr",
        "rev",
        "comm",
        "paste",
        "join",
        # Path-string helpers — pure string math, no filesystem writes.
        "basename",
        "dirname",
        "realpath",
        "readlink",
        "seq",
        "expr",
        # Structured-data / binary printers — read a file (or stdin) and print
        # to stdout. Neither has a file-writing or command-executing flag
        # (``jq -f`` reads a filter; a shell redirect is floored separately).
        "strings",
        "jq",
    }
)

# Commands that DON'T change state when called without arguments / with
# specific subcommands. Limited to the ones we can be sure about. The
# subcommand must be located AFTER any global options (`git -C <path> status`,
# `git --git-dir=… diff`) — see ``_git_globals_stripped`` below; a call whose
# leading options can't be verified read-only (`git -c key=val …` — config
# injection can execute programs: ``core.fsmonitor`` runs from ``git status``,
# ``core.sshCommand`` from network subcommands) stays OUT of this path and is
# gated.
#
# Flag-sensitive / argument-sensitive subcommands are handled by dedicated
# classifiers, NOT this set (a bare argv[1]-membership test would be
# unsound for them):
#   - `branch` → ``_classify_git_branch``: listing forms (bare, ``-a/-r/-v``,
#     ``--list``, ``--merged`` …) are READ_ONLY; ``-d/--delete`` is MUTATING;
#     ``-D`` / ``-df`` / ``--delete --force`` is DESTRUCTIVE.
#   - `remote` → ``_classify_git_remote``: listing forms (bare, ``-v``,
#     ``show``, ``get-url``) are READ_ONLY; ``add/remove/set-url/…`` stay
#     gated.
#   - `fetch` → MUTATING in EVERY form: a dry-run fetch still executes the
#     ``--upload-pack`` helper (arbitrary code), and a normal fetch mutates
#     ``.git`` state (FETCH_HEAD, remote-tracking refs, tags, shallow /
#     maintenance state) and can force-overwrite a local branch via a
#     ``+refs/heads/main:refs/heads/main`` refspec. Plan mode must gate it.
#   - `reflog` → ``_classify_git_reflog``: bare / ``show`` / ``exists`` are
#     READ_ONLY; ``expire`` / ``delete`` (history removal) and any other
#     subcommand stay MUTATING (git does not abbreviate reflog subcommands,
#     and even an abbreviation of expire/delete must never classify
#     READ_ONLY — it lands in the MUTATING fall-through).
#   - `fsck` → ``_classify_git_fsck``: plain connectivity/dangling checks
#     are READ_ONLY; ``--lost-found`` — and every Git-accepted abbreviation
#     of it (``--lost``, ``--lost-f`` …; VERIFIED live: ``git fsck --lost``
#     writes recovered objects into ``.git/lost-found``) — is MUTATING.
#   - `diff` → ``_classify_git_diff``: plain diffs are READ_ONLY;
#     ``--output=<file>`` writes a file and ``--ext-diff`` executes the
#     configured external-diff helper → MUTATING; every unambiguous prefix
#     of those options (``--out=``, ``--ext`` …) and every ``--output*``
#     family member gates the same way (conservative — see
#     ``_opt_is_prefix_or_family``).
#   - `push` → ``_classify_git_push``: always MUTATING at minimum;
#     ``--mirror`` / ``--delete`` / ``--prune`` / ``--force*`` (and every
#     Git-accepted abbreviation of them: ``--mir``, ``--del``, ``--pru``,
#     ``--forc`` … — verified live) / ``-d`` / a force refspec
#     (``+main:main``) / an empty-source refspec (``:main``) are
#     DESTRUCTIVE, INCLUDING refspecs after ``--`` (verified live:
#     ``git push origin -- :main`` deletes the remote branch).
#   - `update-ref` → DESTRUCTIVE in every form (it only rewrites/creates/
#     deletes refs; ``-d`` deletes, ``<ref> <new>`` force-sets,
#     ``--stdin`` replays destructive batches).
#   - `stash` / `tag` → DESTRUCTIVE for their destructive FORMS (`stash
#     clear` / `stash drop` / `stash pop` / `stash branch`, `tag -d` /
#     `tag -f` and abbreviations); everything else stays UNKNOWN →
#     MUTATING → prompt, as before (`config` likewise: its destructive
#     variants cannot be told apart from a read-only invocation by
#     subcommand alone).
#   - `log` / `show` → READ_ONLY for their plain forms, but `--output=
#     <file>` (and `--output <file>`) WRITES a file and `--ext-diff`
#     executes the configured external-diff helper → MUTATING, exactly
#     like `diff` (see ``_classify_git_log_show``).
_GIT_READ_ONLY_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "status",
        "log",
        "show",
        "blame",
        "rev-parse",
        "ls-files",
        "ls-tree",
        "ls-remote",
        "describe",
        "shortlog",
        "count-objects",
        "cherry",
        "name-rev",
        "for-each-ref",
        "cat-file",
        "check-ignore",
        "rev-list",
    }
)


# Git global options that may legally precede the subcommand. ``-C <path>``
# changes the working directory; ``--git-dir`` / ``--work-tree`` point the
# command at another repository; the rest are no-op formatting / behavior
# flags. Skipping them lets ``git -C <repo> status`` classify exactly like
# ``git status``. ``-c <key>=<value>`` / ``--config-env=<name>`` are
# deliberately NOT skippable for read-only classification: config injection
# can make a read-only-looking subcommand execute an arbitrary program, so
# their presence keeps the call out of the read-only path (they ARE skipped
# for the escalate-only checks below, which can only raise severity).
_GIT_GLOBAL_OPTS_WITH_ARG: frozenset[str] = frozenset(
    {
        "-C",
        "--git-dir",
        "--work-tree",
        "--namespace",
    }
)
_GIT_GLOBAL_OPTS_BARE: frozenset[str] = frozenset(
    {
        "-p",
        "--paginate",
        "--no-pager",
        "--bare",
        "--no-replace-objects",
        "--no-optional-locks",
        "--literal-pathspecs",
        "--glob-pathspecs",
        "--noglob-pathspecs",
        "--icase-pathspecs",
        "--no-icase-pathspecs",
    }
)


def _git_globals_stripped(argv: list[str], *, escalate: bool = False) -> list[str]:
    """Return ``argv`` with leading ``git`` global options removed.

    ``argv[0]`` must be ``git`` (callers check the base name). The first
    non-option token is the subcommand. With ``escalate=True`` the
    config-injection options (``-c <key>=<value>`` / ``--config-env=<name>``)
    are skipped too — safe only for checks that can ESCALATE a verdict
    (destructive detection, branch delete forms), never for read-only
    classification.
    """
    i, n = 1, len(argv)
    while i < n:
        tok = argv[i]
        if tok in _GIT_GLOBAL_OPTS_WITH_ARG:
            i += 2
            continue
        if tok.startswith("--git-dir=") or tok.startswith("--work-tree=") or tok.startswith("--namespace="):
            i += 1
            continue
        if escalate and tok == "-c" and i + 1 < n:
            i += 2
            continue
        if escalate and tok.startswith("--config-env="):
            i += 1
            continue
        if tok in _GIT_GLOBAL_OPTS_BARE:
            i += 1
            continue
        break
    return argv[i:]


def _git_has_config_global(argv: list[str]) -> bool:
    """True iff a config-injection global (``-c <key>=<val>`` /
    ``--config-env=<name>``) appears in the leading global-options region.

    Config injection can make a read-only-looking subcommand execute an
    arbitrary program (``core.pager`` runs from ``git branch``, ``core.
    fsmonitor`` from ``git status``, ``core.sshCommand`` from network
    subcommands), so a call carrying one must NEVER classify READ_ONLY —
    the escalate-only strip in ``_git_globals_stripped`` skips these for
    checks that can only RAISE severity, and every classifier that can
    return READ_ONLY must re-check this on the raw argv.
    """
    i, n = 1, len(argv)
    while i < n:
        tok = argv[i]
        if tok == "-c" or tok.startswith("--config-env="):
            return True
        if tok in _GIT_GLOBAL_OPTS_WITH_ARG:
            i += 2
            continue
        if tok.startswith("--git-dir=") or tok.startswith("--work-tree=") or tok.startswith("--namespace="):
            i += 1
            continue
        if tok in _GIT_GLOBAL_OPTS_BARE:
            i += 1
            continue
        break
    return False


def _git_read_only_verdict(
    argv: list[str],
    verdict: CommandSafety,
) -> CommandSafety:
    """Floor a would-be READ_ONLY git verdict to MUTATING when the call
    carries a config-injection global (``-c`` / ``--config-env=``). The
    stop-at-config rule must apply to EVERY classifier that can return
    READ_ONLY, not just ``_classify_segment``'s main path."""
    if verdict is CommandSafety.READ_ONLY and _git_has_config_global(argv):
        return CommandSafety.MUTATING
    return verdict


# Hard-deny patterns — commands whose blast radius makes any approval
# flow a bug. The evaluator maps these to ``deny`` in every mode; the
# permission modal NEVER fires for these. Keep narrow on purpose: a
# false positive here breaks a real workflow, a false negative is a
# safety incident. Match only constructs whose wipe target is clearly
# the operator's home / root / a system disk / a fork-bomb shape.
#
# Patterns intentionally accept the most common shell quotings of
# ``~``, ``$HOME`` and ``${HOME}`` (and their trailing-slash variants)
# while requiring the recursive-force flag combo (``-rf`` / ``-fr`` /
# ``-rfd`` etc.) so an interactive ``rm ~/file`` still gets the normal
# destructive prompt.
_FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # rm -rf on home dir: `~`, `~/`, `~/*`, `"$HOME"`, `${HOME}/`,
        # quoted or not. The constraint at the end (``end-of-string``
        # or shell separator) keeps `rm -rf ~/projects/x` out — those
        # are explicit subdirs and go through the normal destructive
        # path.
        r"\b(?:sudo\s+)?rm\s+-[a-zA-Z]*[rR][a-zA-Z]*[fF][a-zA-Z]*\s+"
        r"(?:--\s+)?['\"]?(?:~|\$\{?HOME\}?)/?['\"]?\s*\*?\s*['\"]?\s*(?:$|[;&|])",
        r"\b(?:sudo\s+)?rm\s+-[a-zA-Z]*[fF][a-zA-Z]*[rR][a-zA-Z]*\s+"
        r"(?:--\s+)?['\"]?(?:~|\$\{?HOME\}?)/?['\"]?\s*\*?\s*['\"]?\s*(?:$|[;&|])",
        # rm -rf / (root) or /* — match `/` standing alone as the
        # target, NOT `/tmp/x` (which is a subdir of /).
        r"\b(?:sudo\s+)?rm\s+-[a-zA-Z]*[rR][a-zA-Z]*[fF][a-zA-Z]*\s+"
        r"(?:--\s+)?['\"]?/['\"]?\s*\*?\s*['\"]?\s*(?:$|[;&|])",
        r"\b(?:sudo\s+)?rm\s+-[a-zA-Z]*[fF][a-zA-Z]*[rR][a-zA-Z]*\s+"
        r"(?:--\s+)?['\"]?/['\"]?\s*\*?\s*['\"]?\s*(?:$|[;&|])",
        # Classic fork bomb — both the canonical form and the
        # whitespace-tolerant variant the model might emit.
        r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
        # Whole-disk raw write to a system device. `/dev/sda`,
        # `/dev/nvme0n1`, `/dev/vda` — no partition suffix means the
        # whole disk including the partition table.
        r"\bdd\b[^|;&]*\bof=/dev/(?:sd[a-z]|nvme\d+n\d+|vd[a-z]|hd[a-z])\b(?!\d)",
        # mkfs on a whole-disk system device — wipes everything on it.
        r"\bmkfs(?:\.\w+)?\s+(?:-\S+\s+)*/dev/(?:sd[a-z]|nvme\d+n\d+|vd[a-z]|hd[a-z])\b(?!\d)",
    )
)


# ---- sharedagents gateway self-protection --------------------------------
#
# An sharedagents agent runs INSIDE the gateway process tree. A ``Bash`` tool
# call that terminates the stack — ``pkill -f 'sharedagents gateway serve'``,
# ``pkill -f 'sharedagents orchestrator serve'``, ``kill $(cat ~/.sharedagents/gateway.pid)``,
# ``killall sharedagents_gateway``, ``systemctl --user stop sharedagents-gateway``,
# ``sharedagents gateway stop`` — kills the agent's own host (the 2026-05 incident).
#
# Defense is layered, because a static string check alone cannot stop a
# bare ``kill <pid>`` whose numeric pid the agent discovered in a prior
# tool call:
#
#   1. THIS detector (run over a de-obfuscated command — see
#      ``_normalize_for_match``) classifies recognised kills FORBIDDEN,
#      so the permission gate hard-denies them in EVERY mode
#      (plan/ask/auto) and the exec boundary (``run_exec_async`` /
#      ``run_monitor_start_async``) refuses to spawn them even if the
#      permission layer is bypassed. This blocks the named/obfuscated
#      variants with a clear, actionable error.
#   2. The SUPERVISOR — systemd ``Restart=always`` on
#      ``sharedagents-gateway.service`` (see ``gateway_cli.py``) — auto-
#      recovers the gateway from ANY kill that still slips through
#      (cross-call PID kill, a raw ``os.kill`` from python, an unguarded
#      spawner). The agent can at worst cause a brief bounce; it cannot
#      leave the gateway DOWN. The only way to keep it down is
#      ``systemctl stop|disable|mask`` of the unit, which layer 1 catches.
#
# ``stop`` / ``kill`` / ``disable`` of the stack stay FORBIDDEN (an
# agent must not leave the gateway down). ``sharedagents gateway restart``
# is classified DESTRUCTIVE (ask-mode still prompts). Auto mode matches
# OpenCode ``--auto`` and allows it. On a systemd host the CLI restart
# is gateway-only and does not SIGTERM the orchestrator.

# A process-signalling verb: ``kill`` / ``kill -9`` (by pid), or
# ``pkill`` / ``killall`` (by name / cmdline). Word-bounded so it does
# not fire on ``skill`` / ``killer``.
_GATEWAY_KILL_VERB_RE = re.compile(r"\b(?:kill|pkill|killall)\b", re.IGNORECASE)

# A kill target that names sharedagents infra. In a kill context, ANY mention
# of ``sharedagents`` / gateway or orchestrator serve argv / pidfiles is
# treated as targeting the stack and is refused — there is no
# legitimate reason for an agent to signal an sharedagents process by name.
_GATEWAY_KILL_TARGET_RE = re.compile(
    r"(?:sharedagents|gateway\s+serve|orchestrator\s+serve|gateway\.pid|orchestrator\.pid)",
    re.IGNORECASE,
)

# ``systemctl … <down-verb>`` of the gateway unit. ``restart`` / ``start``
# / ``status`` are deliberately excluded (those are allowed). The
# down-verb set covers every literal systemctl operation that can take
# the unit down or neuter the ``Restart=always`` supervisor: stop / kill
# / disable / mask / freeze, plus unit-mutating ``edit`` / ``set-property``
# / ``revert``. The ``[^\n;|&]`` class keeps the match inside a single
# simple-command so a benign ``systemctl --user stop foo; echo
# sharedagents-gateway`` does not trip. (Runtime ``$var``/``eval``-built verbs
# evade ANY static matcher — that residual is the OS-isolation gap, not a
# regex gap; see the module header.)
_GATEWAY_SERVICE_DOWN_RE = re.compile(
    r"\bsystemctl\b[^\n;|&]*?"
    r"\b(?:stop|kill|disable|mask|freeze|edit|set-property|revert)\b"
    # Target the gateway unit, tolerating a glob/sep between ``sharedagents``
    # and ``gateway`` (``sharedagents?gateway`` / ``sharedagents*gateway`` /
    # ``sharedagents.gateway`` all deterministically resolve to the unit).
    r"[^\n;|&]*?sharedagents[^\n;|&]{0,3}gateway",
    re.IGNORECASE,
)

# A second systemctl-down shape: the unit named by a GLOB
# (``systemctl --user stop shared*`` / ``s*`` / ``sharedag?nts-gateway``).
# systemd expands the glob against loaded units, so any down-verb whose
# glob matches the gateway unit name takes the gateway down.
_SYSTEMCTL_DOWN_ARGS_RE = re.compile(
    r"\bsystemctl\b[^\n;|&]*?"
    r"\b(?:stop|kill|disable|mask|freeze|edit|set-property|revert)\b"
    r"([^\n;|&]*)",
    re.IGNORECASE,
)
_GATEWAY_UNIT_NAMES = ("sharedagents-gateway", "sharedagents-gateway.service")


def _systemctl_glob_hits_gateway(text: str) -> bool:
    """True iff a systemctl down-verb names a glob that matches the gateway unit."""
    for match in _SYSTEMCTL_DOWN_ARGS_RE.finditer(text):
        for token in match.group(1).split():
            if any(c in token for c in "*?[") and any(fnmatch.fnmatchcase(unit, token) for unit in _GATEWAY_UNIT_NAMES):
                return True
    return False


# The ``sharedagents gateway stop`` CLI verb (or the module-invocation form
# ``python -m sharedagents_gateway gateway stop`` — the ``sharedagents_gateway``
# module name has no word boundary before ``_gateway``, so the prefix
# alternation must spell that suffix out). ``restart`` / ``start`` are
# NOT matched. ``gateway`` must be immediately followed by the down verb
# ``stop``/``kill`` so a command that merely mentions the word "stop"
# elsewhere does not trip.
_GATEWAY_CLI_DOWN_RE = re.compile(
    r"\bsharedagents(?:[_-](?:agent|gateway))?\b[^\n;|&]*?\bgateway\s+(?:stop|kill)\b",
    re.IGNORECASE,
)


def _ansi_c_decode(body: str) -> str:
    """Decode the body of an ANSI-C ``$'…'`` quote the way bash would,
    so ``\\x6f`` → ``o``, ``\\163`` → ``s``, ``\\t`` → tab. Best-effort:
    on any error fall back to the raw body with backslashes dropped (the
    over-normalizing safe direction)."""
    try:
        return body.encode("latin-1", "backslashreplace").decode("unicode_escape")
    except Exception:  # noqa: BLE001
        return body.replace("\\", "")


def _brace_expand(text: str, _depth: int = 0) -> list[str]:
    """Expand bash brace lists (``st{op,}`` → ``stop`` / ``st``,
    ``shared{agents,}-gateway`` → ``sharedagents-gateway`` /
    ``shared-gateway``) into
    every concrete string, so a verb/target split across a brace can't
    dodge the contiguous-token regexes. Bounded (depth + result count)
    so a pathological ``{a,b}{c,d}…`` can't blow up. Ranges (``{1..9}``)
    are not expanded — irrelevant to the gateway tokens."""
    if _depth > 6 or "{" not in text:
        return [text]
    # innermost brace group containing at least one comma
    m = re.search(r"\{([^{}]*,[^{}]*)\}", text)
    if not m:
        return [text]
    pre, post = text[: m.start()], text[m.end() :]
    out: list[str] = []
    for alt in m.group(1).split(","):
        for tail in _brace_expand(pre + alt + post, _depth + 1):
            out.append(tail)
            if len(out) >= 64:
                return out
    return out


def _normalize_for_match(command: str) -> list[str]:
    """De-obfuscate a command into one or more candidate strings so
    quoting / concatenation / brace tricks can't hide a gateway kill
    from the regexes below. Returns a LIST because brace expansion can
    yield several concrete forms; the caller matches if ANY candidate
    trips.

    The shell treats adjacent quoted fragments as ONE word —
    ``pk''ill`` runs ``pkill``, ``shared''agents`` is ``sharedagents``,
    ``gateway'.'pid`` is ``gateway.pid`` — and expands ANSI-C quotes
    (``st$'op'`` → ``stop``, ``st$'\\x6f'p`` → ``stop``) and brace lists
    (``st{op,}`` → ``stop``) before running anything, so a raw-string
    regex over the literal source misses them.
    This reverses the *static, deterministic* expansions bash applies:
    ANSI-C / locale quote decoding, ``$HOME``→``~``, quote stripping,
    backslash removal, and brace expansion. It is deliberately
    *over*-normalizing — a false join only yields a stricter (FORBIDDEN)
    verdict, the safe direction for a self-protection guard.

    It is NOT a full shell parser — RUNTIME indirection (``$VAR`` whose
    value is set elsewhere, command substitution ``$(…)``, ``eval`` /
    ``base64|sh``) is not and cannot be statically expanded. That
    residual is owned by the supervisor layer (systemd ``Restart=always``
    auto-recovers any signal-kill so the agent can at worst bounce, not
    keep-down) and, ultimately, by OS-level isolation (separate
    uid/unit) — see the module header.
    """
    text = str(command)
    # ANSI-C ``$'…'`` quoting: decode C escapes (``\\x6f`` → ``o``) within
    # the quote, then drop the delimiters so fragments rejoin. Done FIRST,
    # before the generic backslash strip, so ``\\x6f`` is decoded rather
    # than mangled to ``x6f``.
    text = re.sub(r"\$'([^']*)'", lambda m: _ansi_c_decode(m.group(1)), text)
    # Locale ``$"…"`` quoting → the inner text verbatim.
    text = re.sub(r'\$"([^"]*)"', lambda m: m.group(1), text)
    # Remaining backslash line-continuations / escapes (``kil\l`` → ``kill``).
    text = text.replace("\\\n", "").replace("\\", "")
    # ``$HOME`` / ``${HOME}`` → ``~`` (pidfile path written either way).
    text = re.sub(r"\$\{?HOME\}?", "~", text, flags=re.IGNORECASE)
    # Any leftover ``$`` directly before a quote that introduced an empty
    # ANSI-C/locale literal.
    text = re.sub(r"\$(?=['\"])", "", text)
    # Drop quote characters — bash concatenates adjacent fragments;
    # removing the quotes reconstructs the effective word. Spaces survive,
    # so only quote-adjacent fragments merge (the attack).
    text = text.replace("'", "").replace('"', "").replace("`", "")
    text = text.lower()
    # Brace expansion last, on the de-quoted text.
    return _brace_expand(text)


def kills_sharedagents_gateway(command: str) -> bool:
    """True iff ``command`` would terminate the sharedagents gateway.

    Runs over a de-obfuscated form of the command (see
    :func:`_normalize_for_match`) so shell quote/concat tricks
    (``pk''ill``, ``shared''agents``, ``st''op``) and ``$HOME``-built pidfile
    paths can't dodge it. Recognises three shapes:

    1. ``kill`` / ``pkill`` / ``killall`` whose target text names sharedagents
       infra (``sharedagents`` / ``gateway serve`` / ``orchestrator serve`` /
       ``gateway.pid`` / ``orchestrator.pid``). This also catches
       single-command pid lookups like ``kill $(cat ~/.sharedagents/gateway.pid)``
       and ``pgrep -f 'sharedagents gateway serve' | xargs kill``.
    2. ``systemctl … stop|kill|disable|mask`` of the ``sharedagents-gateway`` or
       ``sharedagents-orchestrator`` unit.
    3. The ``sharedagents gateway stop`` CLI verb.

    ``restart`` / ``start`` / ``status`` return ``False`` here so they
    are not hard-denied. Restart is still DESTRUCTIVE for ask-mode
    classification via :func:`classify_command` / production-impact.
    The agent must not be able to leave the gateway down.

    Out of reach for any static string check: a bare ``kill <pid>`` whose
    numeric pid was discovered in a SEPARATE earlier tool call carries no
    sharedagents text at all. That residual is handled by the supervisor
    (systemd ``Restart=always``), not here.
    """
    if not command:
        return False
    # Keep Windows paths intact and check numeric taskkill targets against
    # the stack's pidfiles. These string rules also run on Linux CI.
    win_segments, _, _ = _split_command(command)
    if any(_windows_kills_gateway(_windows_argv(seg)) for seg in win_segments):
        return True
    # A single non-executing display/search command treats its arguments as
    # data. Do not interpret a quoted grep pattern such as ``'kill'`` as an
    # executable kill verb merely because the searched path contains
    # ``sharedagents``. Multi-command chains/pipelines and command substitutions are
    # deliberately not exempt: they can contain a real downstream kill.
    try:
        segments, _has_write, has_cmd_sub = _split_command(command)
        simple = [segment for segment in segments if segment.strip()]
        if len(simple) == 1 and not has_cmd_sub:
            try:
                argv = shlex.split(simple[0])
            except ValueError:
                argv = _windows_argv(simple[0])
            base = argv[0].rsplit("/", 1)[-1] if argv else ""
            if oscompat.IS_WINDOWS and argv:
                base = _windows_basename(argv[0])
            literal_commands = _LITERAL_ARG_COMMANDS | {
                "grep",
                "egrep",
                "fgrep",
                "rg",
            }
            rg_execs_preprocessor = base == "rg" and any(arg == "--pre" or arg.startswith("--pre=") for arg in argv[1:])
            if base in literal_commands and not rg_execs_preprocessor:
                return False
    except (ValueError, TypeError):
        pass
    for text in _normalize_for_match(command):
        if _GATEWAY_KILL_VERB_RE.search(text) and _GATEWAY_KILL_TARGET_RE.search(text):
            return True
        if _GATEWAY_SERVICE_DOWN_RE.search(text):
            return True
        if _systemctl_glob_hits_gateway(text):
            return True
        if _GATEWAY_CLI_DOWN_RE.search(text):
            return True
    return False


# ---- own-process-group self-kill -----------------
#
# An sharedagents agent shares a controlling process group with the gateway it
# runs inside. ``kill`` targeting pid/pgid **0** signals *every* process in
# the caller's own process group (``kill(2)``: "If pid is 0 … sig is sent to
# every process in the process group of the calling process"; a negative
# ``-0`` is numerically the same group). That takes the gateway down without
# ever naming it — the residual the name-based ``kills_sharedagents_gateway`` guard
# cannot see. Such a command is FORBIDDEN, denied before
# any hook.
#
# A numeric operand that resolves to the own process group: ``0`` / ``00`` /
# ``-0`` / ``+0`` / ``-00`` … (any signed run of zeros).
_KILL_PGROUP_ZERO_RE = re.compile(r"^[+-]?0+$")


def _signal_token_is_null(tok: str) -> bool:
    """True iff ``tok`` (a ``kill`` signal spec like ``-0``/``0``) is the
    NULL signal (signal 0), which sends nothing — an existence probe, not a
    kill. ``kill -0 <pid>`` must stay allowed."""
    body = tok[1:] if tok.startswith("-") else tok
    return bool(re.fullmatch(r"0+", body))


def _segment_kills_pgroup_zero(segment: str) -> bool:
    """True iff a single (already de-obfuscated) command segment is a
    ``kill`` that sends a real terminating signal to pid/pgid 0 (the own
    process group). Follows ``kill``'s grammar: ``kill [-s sig | -sig] pid…``
    — the signal, if any, is the FIRST option; a ``-N`` appearing *after* it
    (or after ``--``) is a negative-pgid target, not a signal."""
    toks = segment.split()
    kidx = None
    for i, t in enumerate(toks):
        if t == "kill" or t.endswith("/kill"):
            kidx = i
            break
    if kidx is None:
        return False
    args = toks[kidx + 1 :]
    if not args:
        return False
    signal_is_null = False
    targets: list[str] = []
    a0 = args[0]
    if a0 == "--":
        targets = args[1:]
    elif a0 in ("-s", "-n") and len(args) >= 2:
        signal_is_null = _signal_token_is_null(args[1])
        rest = args[2:]
        if rest and rest[0] == "--":
            rest = rest[1:]
        targets = rest
    elif a0.startswith("-") and a0 != "-":
        # ``-9`` / ``-TERM`` / ``-SIGKILL`` / ``-0`` — the signal spec.
        signal_is_null = _signal_token_is_null(a0)
        rest = args[1:]
        if rest and rest[0] == "--":
            rest = rest[1:]
        targets = rest
    else:
        # No signal spec → default SIGTERM (a terminating signal).
        targets = args
    if signal_is_null:
        return False  # signal 0 sends nothing — an existence probe.
    return any(_KILL_PGROUP_ZERO_RE.match(t) for t in targets)


def kills_own_process_group(command: str) -> bool:
    """True iff ``command`` sends a terminating signal to the agent's own
    process group (``kill`` targeting pid/pgid 0), which would take down the
    gateway that shares that group.

    Runs over the de-obfuscated candidate forms (see
    :func:`_normalize_for_match`) so quote/brace tricks can't hide it, and
    splits on shell separators so a ``kill 0`` buried in a chain still trips.
    Deliberately over-normalizing toward FORBIDDEN — the safe direction for a
    self-protection guard. The NULL signal (``kill -0 <pid>``, an existence
    probe) is exempt: it sends nothing.
    """
    if not command:
        return False
    for text in _normalize_for_match(command):
        for segment in re.split(r"[;&|\n]+", text):
            if _segment_kills_pgroup_zero(segment):
                return True
    return False


# Things that are nearly always destructive / irreversible.
_DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*\s+|--recursive\s+)",  # recursive rm: rm -r / -rf / -fr / -rfd / --recursive
        r"\bsudo\s+rm\b",  # any sudo rm
        r"\bdd\s+.*\bof=/dev/",  # dd of=/dev/sdX
        r"\bmkfs\.",  # mkfs.ext4 etc.
        r"\bmkfs\b",
        r"\bshred\b",
        r"\bwipefs\b",
        r"\bfind\b.*\s-delete\b",  # find ... -delete
        r"\bxargs\s+rm\b",
        r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",  # fork-bomb signature
        r">\s*/dev/sd[a-z]",  # raw-device write
        # Any force-push is destructive — many teams promote to
        # `develop` / `production` / `release/*` and the previous
        # main|master-only filter let those slip through.
        r"\bgit\s+push\b[^|;&]*\s-{1,2}f\b",
        r"\bgit\s+push\b[^|;&]*\s-{1,2}force\b",
        r"\bgit\s+push\b[^|;&]*\s-{1,2}force-with-lease\b",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+clean\s+-[a-zA-Z]*[fF][a-zA-Z]*[dD]?\b",
        # ``git branch -d/-D`` is classified argv-aware in
        # ``_classify_git_branch`` — NOT here. These patterns are
        # compiled IGNORECASE, so a ``-D`` regex would also match ``-d``.
        r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b",
        r"\bTRUNCATE\s+(TABLE|DATABASE)\b",
        r"\bdropdb\b",
        r"\b(?:db\.)?dropDatabase\b",
        r"\bDELETE\s+FROM\b\s+\w+\s*;?\s*$",  # DELETE FROM x; with no WHERE
        # Server shutdown, poweroff, and reboot
        r"\b(?:sudo\s+)?(?:shutdown|reboot|poweroff|halt)\b",
        r"\b(?:sudo\s+)?init\s+[06]\b",
        r"\b(?:sudo\s+)?systemctl\s+(?:poweroff|reboot|halt|rescue|emergency)\b",
        # Stopping, restarting, or disabling system services
        r"\b(?:sudo\s+)?systemctl\b[^\n;|&]*?\b(?:stop|restart|disable|mask)\b",
        r"\b(?:sudo\s+)?service\s+\S+\s+(?:stop|restart)\b",
        r"\b(?:sudo\s+)?rc-service\s+\S+\s+(?:stop|restart)\b",
        # Live sharedagents stack bounce — production impact, not a silent allow
        r"\bsharedagents(?:[_-](?:agent|gateway|tui))?\b[^\n;|&]*?\b(?:gateway|orchestrator)\s+restart\b",
        r"\bsharedagents\b[^\n;|&]*?\b(?:telegram|web)\s+restart\b",
        # Process killing and termination
        r"\b(?:sudo\s+)?kill\s+-(?:9|KILL|s\s*(?:9|KILL))\b",
        r"\b(?:sudo\s+)?(?:pkill|killall)\b",
        r"\b(?:sudo\s+)?fuser\s+-[a-zA-Z]*k[a-zA-Z]*\b",
        # Container and cluster lifecycle destruction
        r"\bdocker\s+(?:stop|kill|rm\s+-[a-zA-Z]*f)\b",
        r"\bdocker(?:-compose|\s+compose)\b(?:\s+[^\s;|&]+)*\s+down\b",
        r"\bpodman\s+(?:stop|kill|rm\s+-[a-zA-Z]*f)\b",
        r"\bpodman(?:-compose|\s+compose)\b(?:\s+[^\s;|&]+)*\s+down\b",
        r"\bkubectl\s+(?:delete\s+(?:ns|namespace|deployment|pod|all)|drain)\b",
        # Software / package uninstallation
        r"\b(?:sudo\s+)?(?:apt|apt-get)\s+(?:remove|purge|autoremove)\b",
        r"\bpip3?\s+uninstall\b",
        r"\b(?:sudo\s+)?(?:dnf|yum)\s+(?:remove|erase)\b",
        r"\b(?:sudo\s+)?pacman\s+-[a-zA-Z]*R[a-zA-Z]*\b",
        r"\bnpm\s+(?:uninstall|remove|rm)\s+-g\b",
        # Firewall flushing / disabling
        r"\b(?:sudo\s+)?ip(?:6)?tables\s+-[a-zA-Z]*F[a-zA-Z]*\b",
        r"\b(?:sudo\s+)?ufw\s+(?:disable|reset)\b",
        # User and account deletion
        r"\b(?:sudo\s+)?(?:userdel|groupdel)\b",
        # Windows command strings are classified on every host, including CI.
        r"^\s*(?:rd|rmdir)(?:\.exe)?\b[^\n;|&]*?\s/s\b",
        r"^\s*(?:del|erase|taskkill|format|diskpart|takeown|icacls)(?:\.exe)?\b",
        r"^\s*sc(?:\.exe)?\s+(?:\\\\\S+\s+)?(?:stop|delete)\b",
        r"^\s*net(?:\.exe)?\s+stop\b",
        r"^\s*reg(?:\.exe)?\s+delete\b",
        r"^\s*(?:remove-item|stop-process|stop-service|remove-service)\b",
    )
)

# Things that change state but are generally reversible.
_MUTATING_COMMANDS: frozenset[str] = frozenset(
    {
        "mkdir",
        "rmdir",
        "touch",
        "ln",
        "chmod",
        "chown",
        "chgrp",
        "mv",
        "cp",
        "rsync",
        "install",
        # ``sed`` moved to ``_classify_sed``: ``sed -n '1,240p' file`` is a pure
        # read (the most common plan/explore slice); only ``-i`` / ``-f`` / a
        # script-level ``w``/``e`` command mutates. ``awk`` stays — its
        # ``print > "f"`` form writes without any flag the classifier can spot.
        "tee",
        "awk",
        "apt",
        "apt-get",
        "dnf",
        "yum",
        "brew",
        "pacman",
        "pip",
        "pip3",
        "npm",
        "yarn",
        "pnpm",
        "cargo",
        "go",
        "systemctl",
        "service",
        "launchctl",
        "kill",
        "killall",
        "pkill",
        "make",
        "cmake",
    }
)


class SafetyClassifier:
    """Stateless wrapper around `classify_command` for monkeypatching in tests."""

    def classify(self, command: str) -> CommandSafety:
        return classify_command(command)


def _max_safety(a: CommandSafety, b: CommandSafety) -> CommandSafety:
    """Return the more cautious of two classifications (higher severity)."""
    return a if a.severity >= b.severity else b


def _split_command(text: str) -> tuple[list[str], bool, bool]:
    """Split ``text`` on shell control operators OUTSIDE quotes.

    The historic classifier bailed to ``UNKNOWN`` the instant it saw a
    ``| ; && || > < `` `` $(`` anywhere in the string — including inside
    a quoted regex (``grep -E "a|b"``) — which denied legitimate
    read-only commands in plan mode. This walks the string quote-aware
    so only REAL operators split, and so a read-only-only pipeline /
    chain can be recognized as read-only.

    Returns ``(segments, has_file_write, has_cmd_sub)``:

    - ``segments`` — the individual simple-command strings. Redirect
      operators are dropped; their target words remain so argv[0]
      classification still sees the real command.
    - ``has_file_write`` — a ``>`` / ``>>`` / ``&>`` file redirection is
      present. An fd-duplication (``2>&1`` / ``>&2``) does NOT count —
      it writes no file.
    - ``has_cmd_sub`` — a ``$(...)`` or backtick substitution is present;
      its inner command can't be promised read-only, so callers stay
      cautious.
    """
    segments: list[str] = []
    buf: list[str] = []
    has_file_write = False
    has_cmd_sub = False
    quote: str | None = None
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quote:
            # Inside DOUBLE quotes the shell still expands ``$(...)`` and
            # backtick command substitution, so a "read-only-looking"
            # command can execute a mutating one (`cat "$(touch x)"`).
            # Flag it so the caller floors to UNKNOWN. SINGLE quotes are
            # fully literal — nothing expands there.
            if quote == '"' and (c == "`" or c == "$" and i + 1 < n and text[i + 1] == "("):
                has_cmd_sub = True
            buf.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if oscompat.IS_WINDOWS and c == "^" and i + 1 < n:
            # cmd's escape character; backslashes are path separators.
            buf.append(text[i + 1])
            i += 2
            continue
        if c == '"' or (c == "'" and not oscompat.IS_WINDOWS):
            quote = c
            buf.append(c)
            i += 1
            continue
        # Shell comment to end-of-line. Must run BEFORE other operators so a
        # narrative ``# Check today's …`` line (apostrophe!) cannot open a
        # spurious single-quote and shatter the following ``grep -E 'a|b'``
        # pattern into UNKNOWN segments — the plan-mode over-block that
        # denied journalctl research in session fb3cf2939d02.
        # POSIX shells start a comment only when ``#`` begins a word. Preserve
        # literals such as ``echo foo#bar`` while still dropping model-authored
        # narrative lines like ``# Check today's logs``.
        if not oscompat.IS_WINDOWS and c == "#" and (i == 0 or text[i - 1].isspace() or text[i - 1] in ";&|()<>"):
            while i < n and text[i] != "\n":
                i += 1
            continue
        # command-substitution markers — literal chars, but flag them
        if c == "`":
            has_cmd_sub = True
            buf.append(c)
            i += 1
            continue
        if c == "$" and i + 1 < n and text[i + 1] == "(":
            has_cmd_sub = True
            buf.append(c)
            i += 1
            continue
        # Process substitution `<(cmd)` / `>(cmd)` — the shell RUNS the
        # inner command, so it can't be promised read-only. Flag it
        # BEFORE the redirect branches below treat the `<` / `>` as a
        # plain input/output redirect.
        if c in ("<", ">") and i + 1 < n and text[i + 1] == "(":
            has_cmd_sub = True
            buf.append(c)
            i += 1
            continue
        two = text[i : i + 2]
        # control operators → break the current segment
        if two in ("||", "&&"):
            segments.append("".join(buf))
            buf = []
            i += 2
            continue
        if c == "|":
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        if c in (";", "\n"):
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        if c == "&":
            if two == "&>":  # &>file — redirect all streams to a file
                if not _redirect_target_is_benign(text, i + 2):
                    has_file_write = True
                i += 2
                continue
            segments.append("".join(buf))
            buf = []
            i += 1
            continue  # background &
        if c == ">":
            if two == ">&":
                # ``>&N`` / ``>&-`` is fd-duplication (no file). But
                # ``>&FILE`` (non-numeric target) redirects BOTH stdout
                # and stderr to a FILE — a real truncating write — so it
                # must NOT be waved through as read-only.
                j = i + 2
                while j < n and text[j] == " ":
                    j += 1
                if j < n and (text[j].isdigit() or text[j] == "-"):
                    i += 2  # fd-dup, not a file write
                    continue
                if not _redirect_target_is_benign(text, i + 2):
                    has_file_write = True
                i += 2
                continue
            step = 2 if two == ">>" else 1  # >> append / > truncate
            if not _redirect_target_is_benign(text, i + step):
                has_file_write = True
            i += step
            continue
        if c == "<":
            i += 2 if two == "<<" else 1  # heredoc / input redirect — read side
            continue
        if c.isdigit() and i + 1 < n and text[i + 1] == ">":
            # fd-prefixed redirect (`2>file`, `2>&1`): break the token so
            # the digit doesn't glue onto the previous word, then let the
            # next iteration's `>` handling decide file-write vs fd-dup.
            buf.append(" ")
            i += 1
            continue
        buf.append(c)
        i += 1
    if buf:
        segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()], has_file_write, has_cmd_sub


# Redirect targets that don't create/overwrite a real file — writing
# here is a no-op for safety purposes, so a ``> /dev/null`` (or
# ``2>/dev/null``) must NOT bump a read-only command up to MUTATING.
# This matters in plan mode: research commands routinely silence stderr
# (``ls x 2>/dev/null || echo missing``) and that must stay read-only.
_BENIGN_WRITE_TARGETS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})


def _redirect_target_is_benign(text: str, j: int) -> bool:
    """True when the redirect target starting at/after ``j`` is a discard
    / passthrough device (``/dev/null`` & friends) rather than a real
    file. Quote-stripping keeps ``> "/dev/null"`` benign too."""
    n = len(text)
    while j < n and text[j] in " \t":
        j += 1
    k = j
    while k < n and text[k] not in " \t|&;<>\n":
        k += 1
    target = text[j:k].strip("'\"")
    return target in _BENIGN_WRITE_TARGETS or (oscompat.IS_WINDOWS and target.casefold() == "nul")


# Short ssh options that consume the FOLLOWING token as their argument
# (e.g. `ssh -p 2222 host …`, `ssh -i key host …`). Used to find where the
# remote command begins so we can classify IT rather than blanket-tagging ssh.
_SSH_ARG_OPTS: frozenset[str] = frozenset("bcDeEFiIJLlmOopQRSWw")


# ssh is transparent transport: we classify the command it RUNS. Besides the
# remote command, ssh can be told to run an ADDITIONAL command on the LOCAL side
# via `-o ProxyCommand=…` / `-o LocalCommand=…` (`PermitLocalCommand=yes` arms
# the latter). Those local commands are unverifiable from here, so their presence
# means "not provably read-only". Pure port-forwarding (`-L/-R/-D/-W`) carries no
# command — it's just bridging — so it does NOT escalate; the remote command
# still decides.
_SSH_LOCAL_COMMAND_O_KEYS: tuple[str, ...] = (
    "proxycommand",
    "localcommand",
    "permitlocalcommand",
)


def _ssh_runs_local_command(argv: list[str]) -> bool:
    """True when ssh options make it run an unverifiable LOCAL command
    (`ProxyCommand`/`LocalCommand`) — classifying by the remote command alone
    would then FALSE-ALLOW that local side effect."""
    i, n = 1, len(argv)
    while i < n:
        tok = argv[i]
        if tok == "-o" and i + 1 < n:
            val = argv[i + 1].lower()
            if any(val.startswith(k) for k in _SSH_LOCAL_COMMAND_O_KEYS):
                return True
            i += 2
            continue
        if tok.startswith("-o"):  # attached form: -oProxyCommand=…
            val = tok[2:].lower()
            if any(val.startswith(k) for k in _SSH_LOCAL_COMMAND_O_KEYS):
                return True
        i += 1
    return False


def _ssh_remote_command(argv: list[str]) -> str | None:
    """Given a tokenised ``ssh …`` argv, return the remote command string, or
    ``None`` for a bare/interactive ssh (no command → unverifiable)."""
    i, n = 1, len(argv)
    host_found = False
    while i < n:
        tok = argv[i]
        if not host_found:
            if tok == "--":
                i += 1
                continue
            if tok.startswith("-"):
                # `-p 2222` (separate arg) consumes the next token; `-p2222`
                # (attached) or a bare flag like `-v` consumes only itself.
                if len(tok) == 2 and tok[1] in _SSH_ARG_OPTS:
                    i += 2
                else:
                    i += 1
                continue
            host_found = True  # first non-option token is the host
            i += 1
            continue
        return " ".join(argv[i:]).strip() or None
    return None


# `systemctl <verb>` read-only vs mutating verbs (flags like --user / -H host
# may precede the verb, so we scan all tokens). Used so remote service
# inspection (`ssh h "systemctl status x"`) classifies read-only.
_SYSTEMCTL_RO_VERBS: frozenset[str] = frozenset(
    {
        "status",
        "show",
        "cat",
        "list-units",
        "list-unit-files",
        "list-timers",
        "list-sockets",
        "list-jobs",
        "list-dependencies",
        "is-active",
        "is-enabled",
        "is-failed",
        "is-system-running",
        "get-default",
        "help",
        "show-environment",
    }
)
_SYSTEMCTL_MUT_VERBS: frozenset[str] = frozenset(
    {
        "start",
        "stop",
        "restart",
        "reload",
        "try-restart",
        "reload-or-restart",
        "enable",
        "disable",
        "mask",
        "unmask",
        "kill",
        "set-property",
        "edit",
        "daemon-reload",
        "daemon-reexec",
        "isolate",
        "set-default",
        "revert",
        "reset-failed",
        "reboot",
        "poweroff",
        "halt",
        "suspend",
        "hibernate",
        "hybrid-sleep",
        "set-environment",
        "unset-environment",
    }
)
_SYSTEMCTL_DESTRUCTIVE_VERBS: frozenset[str] = frozenset(
    {
        "stop",
        "restart",
        "try-restart",
        "reload-or-restart",
        "disable",
        "mask",
        "kill",
        "isolate",
        "reboot",
        "poweroff",
        "halt",
        "suspend",
        "hibernate",
        "hybrid-sleep",
    }
)


def _is_comment_only_segment(segment: str) -> bool:
    """True when *segment* is empty or only shell ``#`` comment line(s).

    Models often prefix research bash with a narrative ``# Check …`` line;
    ``_split_command`` turns that into its own segment whose argv[0] is
    ``#`` → UNKNOWN → plan-mode deny. Strip those no-op segments so the
    real ``journalctl`` / ``echo`` command keeps its READ_ONLY tag.
    """
    s = (segment or "").strip()
    if not s:
        return True
    for line in s.splitlines():
        t = line.strip()
        if t and not t.startswith("#"):
            return False
    return True


def _is_venv_activate_path(raw: str) -> bool:
    """True for a typical virtualenv activate script path."""
    p = (raw or "").strip().strip("'\"")
    lower = p.replace("\\", "/").lower()
    return lower.endswith("/bin/activate") or lower.endswith("/scripts/activate")


# ``env`` options that consume their argument as a SEPARATE token
# (``env -u HOME …``). Attached forms (``--unset=HOME``) and bare flags
# (``-i`` / ``-0`` / ``-v`` / ``--null`` …) are skipped generically.
_ENV_OPTS_WITH_ARG: frozenset[str] = frozenset(
    {
        "-u",
        "--unset",
        "-C",
        "--chdir",
        "-P",
        "-S",
        "--split-string",
        "-O",
        "--default-signal",
        "--block-signal",
        "--ignore-signal",
    }
)


def _unwrap_exec_wrapper(argv: list[str]) -> list[str] | None:
    """Return the argv of the command an ``env``/``command`` wrapper RUNS.

    ``env`` and ``command`` are blanket READ_ONLY allowlist entries, so
    without this unwrap EVERY wrapped form bypassed the git classifiers:
    ``env git push --delete origin main`` classified READ_ONLY → ALLOW in
    auto AND plan. Here:

    - ``env [VAR=val …] git …`` / ``env [opts] git …`` → ``git …``
      (``-S``/``--split-string`` payloads are themselves a command string
      and are split+returned; ``env -u NAME``-style options consume their
      argument token).
    - ``command [-p|-v|-V] git …`` → ``git …`` (``--`` terminator
      allowed). ``-v``/``-V`` are lookups, but classifying the inner
      command is still the conservative choice for any inner form that
      would execute.

    Returns None for a bare wrapper (``env`` / ``command`` with no
    command, ``env VAR=x`` only) — callers then behave exactly as before
    (the wrapper keeps its blanket READ_ONLY tag). Recursion is bounded:
    every non-None return strips at least the wrapper token.
    """
    if not argv:
        return None
    base = argv[0].rsplit("/", 1)[-1]
    n = len(argv)
    if base == "env":
        i = 1
        while i < n:
            tok = argv[i]
            # ``NAME=value`` environment assignment (name non-empty).
            if "=" in tok and not tok.startswith("-") and tok.split("=", 1)[0]:
                i += 1
                continue
            if tok in ("-S", "--split-string") and i + 1 < n:
                # The argument IS a command string env will split+exec.
                inner = shlex.split(argv[i + 1])
                return inner or None
            if tok in _ENV_OPTS_WITH_ARG and i + 1 < n:
                i += 2  # option with a separate-argument value
                continue
            if tok.startswith("-"):
                i += 1  # bare flag / attached-value option (-uNAME, -Cdir)
                continue
            return argv[i:]  # first non-option, non-assignment token
        return None
    if base == "command":
        i = 1
        while i < n and argv[i] in ("-p", "-v", "-V"):
            i += 1
        if i < n and argv[i] == "--":
            i += 1
        if i < n:
            return argv[i:]
        return None
    return None


def _git_has_exec_alias_config(argv: list[str]) -> bool:
    """True iff a leading config global injects an EXECUTABLE alias
    (``-c alias.<name>=!<shell-cmd>`` / ``--config-env=alias.<name>=``).

    Git runs ``!``-prefixed alias values through ``sh -c`` when the alias
    name is used as the subcommand (``git -c alias.x='!rm -rf /' x``
    executes ``rm -rf /``), and ``--config-env=alias.<name>=VAR`` reads
    the value from the environment so its content cannot be inspected
    here at all. Both must floor classification to MUTATING (never
    READ_ONLY) in every path — see the floor in ``_classify_segment``.
    """
    i, n = 1, len(argv)
    while i < n:
        tok = argv[i]
        if tok == "-c" and i + 1 < n:
            key, sep, val = argv[i + 1].partition("=")
            if key.startswith("alias.") and sep and val.startswith("!"):
                return True
            i += 2
            continue
        # Attached ``-c<key>=<val>`` (git's parse-options accepts it).
        if tok.startswith("-c") and len(tok) > 2 and not tok.startswith("--"):
            key, sep, val = tok[2:].partition("=")
            if key.startswith("alias.") and sep and val.startswith("!"):
                return True
            i += 1
            continue
        if tok.startswith("--config-env="):
            if tok.split("=", 1)[1].startswith("alias."):
                return True
            i += 1
            continue
        if tok in _GIT_GLOBAL_OPTS_WITH_ARG:
            i += 2
            continue
        if tok.startswith("--git-dir=") or tok.startswith("--work-tree=") or tok.startswith("--namespace="):
            i += 1
            continue
        if tok in _GIT_GLOBAL_OPTS_BARE:
            i += 1
            continue
        break
    return False


def _classify_git_log_show(argv: list[str], sub: str) -> CommandSafety:
    """Classify ``git log`` / ``git show`` forms (global-option aware).

    Plain forms are read-only queries. ``--output=<file>`` (and the
    two-token ``--output <file>`` form) WRITES the log output to a file
    and ``--ext-diff`` EXECUTES the configured external-diff helper →
    MUTATING — the SAME argument-aware gating as ``diff`` (codex round-3
    audit: ``git log --output=/tmp/x`` created files while classifying
    READ_ONLY → plan-ALLOW). Abbreviations / family members
    (``--out=…``, ``--output-indicator-new=…``) gate the same way, via
    the shared ``_opt_is_prefix_or_family``. ``--stat`` / ``--oneline`` /
    format flags stay READ_ONLY.
    """
    rest = _git_globals_stripped(argv, escalate=True)
    if not rest or rest[0] != sub:
        return CommandSafety.UNKNOWN
    for a in rest[1:]:
        if a.startswith("--") and (
            _opt_is_prefix_or_family(a, "--ext-diff") or _opt_is_prefix_or_family(a, "--output")
        ):
            return CommandSafety.MUTATING
    return _git_read_only_verdict(argv, CommandSafety.READ_ONLY)


def _classify_git_worktree(argv: list[str]) -> CommandSafety:
    """``git worktree list`` is a pure query; every other form mutates."""
    rest = _git_globals_stripped(argv, escalate=True)
    if not rest or rest[0] != "worktree":
        return CommandSafety.MUTATING
    verbs = [a for a in rest[1:] if not a.startswith("-")]
    if verbs and verbs[0] == "list":
        return _git_read_only_verdict(argv, CommandSafety.READ_ONLY)
    return CommandSafety.MUTATING


def _classify_git_grep(argv: list[str]) -> CommandSafety:
    """``git grep`` is a read-only search EXCEPT ``-O`` /
    ``--open-files-in-pager[=<pager>]``, which EXECUTES the pager on each
    match (the same helper-execution class as ``--ext-diff``) → MUTATING."""
    rest = _git_globals_stripped(argv, escalate=True)
    if not rest or rest[0] != "grep":
        return CommandSafety.UNKNOWN
    for a in rest[1:]:
        if a == "--":
            break
        if a.startswith("--"):
            if _opt_is_prefix_or_family(a, "--open-files-in-pager"):
                return CommandSafety.MUTATING
            continue
        if a.startswith("-") and "O" in a[1:]:
            return CommandSafety.MUTATING
    return _git_read_only_verdict(argv, CommandSafety.READ_ONLY)


def _classify_git_segment(argv: list[str]) -> CommandSafety:
    """Classify a ``git`` simple-command form (global-option aware).

    ``argv[0]`` must be ``git``. See ``_classify_segment`` for the
    per-subcommand policy; the un-escalated global-option strip stops at
    ``-c`` / ``--config-env`` (config injection can execute programs),
    so a call carrying those lands here with a non-subcommand ``sub`` and
    stays gated (UNKNOWN → MUTATING; the executable-alias floor in
    ``_classify_segment`` raises it further when the injected config is
    a shell alias).
    """
    rest = _git_globals_stripped(argv)
    if not rest:
        return CommandSafety.UNKNOWN
    sub = rest[0]
    # Global informational flags do not name a subcommand.
    if sub in ("--version", "--help", "-h"):
        return CommandSafety.READ_ONLY
    # `log` / `show` are argument-sensitive: --output/--ext-diff forms
    # write files / execute helpers (see _classify_git_log_show).
    if sub in ("log", "show"):
        return _classify_git_log_show(argv, sub)
    # `format-patch` WRITES .patch files — never read-only.
    if sub == "format-patch":
        return CommandSafety.MUTATING
    # `git <subcommand>` with a read-only subcommand. The un-escalated
    # global-option strip stops at `-c` / `--config-env` (config
    # injection can execute programs), so a call carrying those lands
    # here with a non-subcommand `sub` and stays gated.
    # ``git grep`` is read-only EXCEPT ``-O`` / ``--open-files-in-pager``,
    # which executes the pager on each match (same class as --ext-diff).
    if sub == "grep":
        return _classify_git_grep(argv)
    if sub in _GIT_READ_ONLY_SUBCOMMANDS:
        return CommandSafety.READ_ONLY
    # Argument-sensitive subcommands get their own classifiers:
    #   - `fetch` is MUTATING in EVERY form (--upload-pack executes an
    #     arbitrary helper even with --dry-run; +refspecs can overwrite
    #     local branches; FETCH_HEAD/refs/tags/shallow all mutate).
    #   - `reflog` / `fsck` / `diff` are READ_ONLY only for their plain
    #     forms; their mutating flags get MUTATING.
    #   - `push` is MUTATING at minimum; mirror/delete/force-refspec
    #     forms are DESTRUCTIVE.
    if sub == "fetch":
        return CommandSafety.MUTATING
    # ``worktree list`` reads .git/worktrees metadata; every other form
    # (add / remove / move / prune / lock / unlock / repair) mutates.
    if sub == "worktree":
        return _classify_git_worktree(argv)
    if sub == "reflog":
        return _classify_git_reflog(argv)
    if sub == "fsck":
        return _classify_git_fsck(argv)
    if sub == "diff":
        return _classify_git_diff(argv)
    if sub == "push":
        return _classify_git_push(argv)
    # Flag-sensitive subcommands get their own classifiers; a mutating
    # form returns None → UNKNOWN → the caller prompts.
    if sub == "branch":
        br = _classify_git_branch(argv)
        if br is not None:
            return br
        return CommandSafety.UNKNOWN
    if sub == "remote":
        rm = _classify_git_remote(argv)
        if rm is not None:
            return rm
        return CommandSafety.UNKNOWN
    # Local-ref destructive ops: `update-ref` has no read-only form;
    # `stash clear/drop/pop/branch` and `tag -d/-f` (and abbreviations)
    # remove permanent state → DESTRUCTIVE. Their other forms stay
    # UNKNOWN → MUTATING (unchanged gating; `config` likewise).
    if sub == "update-ref":
        return CommandSafety.DESTRUCTIVE
    if sub == "stash":
        if _classify_git_stash_destructive(rest[1:]):
            return CommandSafety.DESTRUCTIVE
        return CommandSafety.UNKNOWN
    if sub == "tag":
        if _classify_git_tag_destructive(rest[1:]):
            return CommandSafety.DESTRUCTIVE
        return CommandSafety.UNKNOWN
    return CommandSafety.UNKNOWN


# ── sed ───────────────────────────────────────────────────────────────
# sed command characters with no file-write / command-execution effect:
# flow (b/t/T/:), hold space (h/H/g/G/x), print (p/P/l/=), input control
# (d/D/n/N/q/Q/z), read-a-file (r/R) and substitution (parsed separately,
# its flags are inspected there).
_SED_SAFE_COMMANDS: frozenset[str] = frozenset(":bdDgGhHlnNpPqQrRstTxz=")


def _sed_script_is_read_only(script: str) -> bool:
    """True when a sed script cannot write a file or execute a command.

    Scans sed's command grammar far enough to find ``w`` / ``W`` (write a
    file) and ``e`` / ``E`` (execute a shell command), including the
    ``s///w`` / ``s///e`` flag spellings. Anything the scanner cannot
    recognise fails closed (False), so an exotic script stays gated
    rather than being promised read-only.
    """
    i, n = 0, len(script)
    while i < n:
        c = script[i]
        if c in " \t\n;{}":
            i += 1
            continue
        if c == "#":  # comment to end of line
            while i < n and script[i] != "\n":
                i += 1
            continue
        # Address prefixes: line numbers, $, and the,. + ~ ! modifiers.
        if c.isdigit() or c in "$,.+~!":
            i += 1
            continue
        if c == "\\":
            i += 2
            continue
        if c == "/":  # /regex/ address
            i += 1
            while i < n and script[i] != "/":
                i += 2 if script[i] == "\\" else 1
            i += 1
            continue
        if c in "sSyY":
            delim = script[i + 1] if i + 1 < n else ""
            # A backslash delimiter is pathological — fail closed.
            if not delim or delim == "\\":
                return False
            i += 2
            closed = 0
            while i < n and closed < 2:
                if script[i] == "\\":
                    i += 2
                    continue
                if script[i] == delim:
                    closed += 1
                i += 1
            if c in "yY":
                continue
            flags = ""
            while i < n and script[i] not in "\n;{}":
                flags += script[i]
                i += 1
            if any(ch in flags for ch in "wWeE"):
                return False
            continue
        if c in "aAiIcC":  # literal text to end of line — stdout only
            while i < n and script[i] != "\n":
                i += 1
            continue
        if c in "wWeE":  # file writes / shell execution
            return False
        if c in _SED_SAFE_COMMANDS:
            i += 1
            continue
        return False  # unrecognised construct → fail closed
    return True


def _classify_sed(argv: list[str]) -> CommandSafety:
    """Argument-aware ``sed`` classification.

    ``sed -n '1,240p' file`` is a pure read and was the most common gated
    slice in plan/explore sessions. Only three forms mutate: ``-i``
    (in-place edit), ``-f`` (a script file this classifier cannot
    inspect), and a script whose own ``w``/``e`` command writes or
    executes (see :func:`_sed_script_is_read_only`).
    """
    scripts: list[str] = []
    i, n = 1, len(argv)
    while i < n:
        a = argv[i]
        if a == "--":
            i += 1
            continue
        if a == "-f" or a == "--file" or a.startswith("--file="):
            return CommandSafety.MUTATING
        if a.startswith("--in-place"):
            return CommandSafety.MUTATING
        if a in ("-e", "--expression"):
            if i + 1 < n:
                scripts.append(argv[i + 1])
                i += 2
                continue
            return CommandSafety.MUTATING
        if a.startswith("--expression="):
            scripts.append(a.split("=", 1)[1])
            i += 1
            continue
        if a.startswith("--"):
            i += 1
            continue
        if a.startswith("-") and len(a) > 1:
            if "i" in a[1:]:
                return CommandSafety.MUTATING  # -i / -i.bak / -ni clusters
            i += 1
            continue
        if not scripts:
            scripts.append(a)  # the first positional is the script
        i += 1
    if not scripts or not all(_sed_script_is_read_only(s) for s in scripts):
        return CommandSafety.MUTATING
    return CommandSafety.READ_ONLY


# ── curl ──────────────────────────────────────────────────────────────
# Flags that write a LOCAL file, and flags that mutate REMOTE state.
_CURL_WRITE_FLAGS: frozenset[str] = frozenset(
    {
        "-o",
        "--output",
        "-O",
        "--remote-name",
        "--remote-name-all",
        "-D",
        "--dump-header",
        "-c",
        "--cookie-jar",
        "--stderr",
        "--trace",
        "--trace-ascii",
        "--trace-config",
        "--libcurl",
        "--output-dir",
        "-K",
        "--config",
        "--etag-save",
        "--hsts",
    }
)
_CURL_BODY_FLAGS: frozenset[str] = frozenset(
    {
        "-d",
        "--data",
        "--data-raw",
        "--data-binary",
        "--data-urlencode",
        "-F",
        "--form",
        "--form-string",
        "-T",
        "--upload-file",
    }
)
_CURL_READ_METHODS: frozenset[str] = frozenset({"GET", "HEAD"})
# ``-o -`` / ``-D -`` / ``-c -`` write to STDOUT — discard-equivalent sinks.
_NULL_SINKS: tuple[str, ...] = ("/dev/null", "nul", "-")
_CURL_STDOUT_OK: frozenset[str] = frozenset(
    {
        "-o",
        "--output",
        "-D",
        "--dump-header",
        "-c",
        "--cookie-jar",
    }
)


def _classify_curl(argv: list[str]) -> CommandSafety:
    """``curl`` is a read-only network fetch unless it writes a local file
    or mutates remote state. ``-o /dev/null`` stays read-only — it is a
    discard, the idiom for inspecting headers/status only. A body flag
    (``-d`` / ``-F`` / ``-T``) or a non-GET/HEAD method is MUTATING, and a
    pipeline that pipes a body into a shell is floored by that segment.
    """
    i, n = 1, len(argv)
    while i < n:
        a = argv[i]
        if a == "--":
            break
        if a.startswith("--") and "=" in a:
            key, _, val = a.partition("=")
            if key in _CURL_BODY_FLAGS or key in _CURL_WRITE_FLAGS:
                if key in _CURL_STDOUT_OK and val in _NULL_SINKS:
                    i += 1
                    continue
                return CommandSafety.MUTATING
            if key == "--request" and val.upper() not in _CURL_READ_METHODS:
                return CommandSafety.MUTATING
            i += 1
            continue
        if a in _CURL_BODY_FLAGS:
            return CommandSafety.MUTATING
        if a in _CURL_WRITE_FLAGS:
            val = argv[i + 1] if i + 1 < n else ""
            if a in _CURL_STDOUT_OK and val in _NULL_SINKS:
                i += 2
                continue
            return CommandSafety.MUTATING
        if a.startswith("-o") and len(a) > 2 and not a.startswith("--"):
            if a[2:] in _NULL_SINKS:
                i += 1
                continue
            return CommandSafety.MUTATING
        if a in ("-X", "--request"):
            method = (argv[i + 1] if i + 1 < n else "").upper()
            if method and method not in _CURL_READ_METHODS:
                return CommandSafety.MUTATING
            i += 2
            continue
        if a.startswith("-X") and len(a) > 2 and not a.startswith("--"):
            if a[2:].upper() not in _CURL_READ_METHODS:
                return CommandSafety.MUTATING
            i += 1
            continue
        i += 1
    return CommandSafety.READ_ONLY


def _classify_segment(segment: str) -> CommandSafety:
    """Classify ONE simple command (no control operators).

    Mirrors the historic argv[0] logic: read-only allowlist, git /
    crontab inspect-only forms, mutating list, else ``UNKNOWN``.
    Destructive / forbidden detection is the caller's job (it runs over
    the whole string before splitting), so this never returns those.
    """
    try:
        argv = shlex.split(segment)
    except ValueError:
        return CommandSafety.UNKNOWN
    if not argv:
        return CommandSafety.UNKNOWN
    # Strip a leading path: `/usr/bin/rm` → `rm`.
    base = argv[0].rsplit("/", 1)[-1]

    # `env [VAR=val …] git …` / `command [-p|-v|-V] git …` EXECUTE the
    # wrapped command — classify the REAL one (wrapped destructive →
    # DESTRUCTIVE, wrapped read-only → READ_ONLY) so the wrapper's
    # blanket READ_ONLY tag can't bypass the git classifiers. Bare
    # `env` / `command` (no command) fall through unchanged below.
    inner = _unwrap_exec_wrapper(argv)
    if inner is not None:
        return _classify_segment(shlex.join(inner))

    # `ssh [opts] host <remote-cmd>` — classify by the REMOTE command so the
    # planner/explorer can run read-only remote inspection (`ssh h "df -h"`)
    # unprompted, while a mutating/destructive remote command keeps its tag
    # (destructive remotes are also caught by the whole-string regex upstream).
    # A bare interactive `ssh host` is unverifiable → MUTATING (caller gates).
    if base in ("ssh", "rsh"):
        # A ProxyCommand/LocalCommand option runs an unverifiable LOCAL command —
        # don't trust remote-only classification then. (Forwarding is just a
        # bridge and does NOT escalate; the remote command decides.)
        if _ssh_runs_local_command(argv):
            return CommandSafety.MUTATING
        remote = _ssh_remote_command(argv)
        if not remote:
            return CommandSafety.MUTATING
        inner = classify_command(remote)
        # An opaque remote (UNKNOWN) can't be promised read-only.
        return CommandSafety.MUTATING if inner == CommandSafety.UNKNOWN else inner

    if base == "git":
        verdict = _classify_git_segment(argv)
        # Executable-alias floor: a `-c alias.<name>=!<shell-cmd>` /
        # `--config-env=alias.<name>=` global can turn ANY subcommand
        # (including an unrecognized alias-name subcommand) into an
        # arbitrary `sh -c` execution — floor to MUTATING so the gate
        # never classifies the call READ_ONLY in any path. Destructive
        # forms still win (the floor only raises severity).
        if _git_has_exec_alias_config(argv):
            verdict = _max_safety(verdict, CommandSafety.MUTATING)
        return verdict
    # `crontab -l` lists the table (read-only); `crontab <file>` installs
    # and `crontab -r` removes — those stay UNKNOWN (caller prompts).
    if base == "crontab" and len(argv) >= 2 and argv[1] == "-l":
        return CommandSafety.READ_ONLY
    # `systemctl status/show/list-*/is-*` inspect; start/stop/enable/... mutate.
    if base == "systemctl":
        toks = set(argv[1:])
        if toks & _SYSTEMCTL_DESTRUCTIVE_VERBS:
            return CommandSafety.DESTRUCTIVE
        if not (toks & _SYSTEMCTL_MUT_VERBS) and (toks & _SYSTEMCTL_RO_VERBS):
            return CommandSafety.READ_ONLY
    # `journalctl` reads logs (read-only) UNLESS an option mutates the journal /
    # FSS keys / catalog (vacuum, rotate, flush, sync, relinquish, setup-keys,
    # update-catalog). Anything in that set falls through to UNKNOWN→prompt.
    if base == "journalctl" and not any(
        a.startswith("--vacuum")
        or a
        in (
            "--rotate",
            "--flush",
            "--sync",
            "--relinquish-var",
            "--smart-relinquish-var",
            "--setup-keys",
            "--update-catalog",
        )
        for a in argv[1:]
    ):
        return CommandSafety.READ_ONLY
    # `source …/bin/activate` (and `. …/bin/activate`) only primes the
    # ephemeral shell env for a following inspect command — treat as
    # read-only. Arbitrary `source other.sh` stays UNKNOWN.
    if base in ("source", ".") and len(argv) >= 2 and _is_venv_activate_path(argv[1]):
        return CommandSafety.READ_ONLY
    # Arbitrary ``python -c`` cannot be proven read-only by source-pattern
    # matching. Leave it UNKNOWN; the plan-mode Agent gate performs one local
    # teacher-style validation turn for this narrow case.
    # `find` is read-only for LISTING, but `-exec` / `-execdir` / `-ok` /
    # `-okdir` run an ARBITRARY command per match (`find. -exec rm {} ';'`)
    # — those are not read-only. This guard keeps the action forms out of
    # the read-only allowlist below. (With a quoted `';'` terminator the
    # whole command parses as one `find` segment that would otherwise be
    # allowlisted; the `\;` backslash form already lands non-read-only
    # another way — it splits on the bare `;` and the trailing-backslash
    # fragment fails `shlex.split` → UNKNOWN — but this guard makes the
    # intent explicit and uniform across both terminator forms.)
    if base == "find" and any(a in ("-exec", "-execdir", "-ok", "-okdir") for a in argv[1:]):
        return CommandSafety.UNKNOWN
    # `sort -o FILE` / `sort --output=FILE` writes a file even though
    # `sort` is otherwise a read-only stream filter.
    if base == "sort" and any(a == "-o" or a.startswith("--output") for a in argv[1:]):
        return CommandSafety.MUTATING
    # Argument-aware: a bare name lookup cannot see ``sed -i`` /
    # ``sed 'w f'`` or ``curl -o f`` / ``curl -X POST``.
    if base == "sed":
        return _classify_sed(argv)
    if base == "curl":
        return _classify_curl(argv)
    if base in _READ_ONLY_COMMANDS:
        return CommandSafety.READ_ONLY
    if base in _MUTATING_COMMANDS:
        return CommandSafety.MUTATING
    return CommandSafety.UNKNOWN


# Commands whose argv after argv[0] is DATA, not an executed payload.
# Destructive tokens in their arguments must not inherit danger —
# ``which mkfs.ext4`` / ``echo "rm -rf /"`` are probes/prints, not wipes.
# (Command substitution inside them is handled separately.)
_LITERAL_ARG_COMMANDS: frozenset[str] = frozenset(
    {
        "echo",
        "printf",
        "which",
        "type",
        "whereis",
        "man",
        "help",
    }
)

# Exact wipe targets that are never approvable.
_FORBIDDEN_WIPE_EXACT: frozenset[str] = frozenset(
    {
        "~",
        "~/",
        "~/*",
        "/",
        "/*",
        "$home",
        "$home/",
        "${home}",
        "${home}/",
        "/home",
        "/root",
        "/users",
        "/etc",
        "/usr",
        "/var",
        "/boot",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/opt",
        "/srv",
    }
)


def _strip_quotes(tok: str) -> str:
    t = (tok or "").strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        return t[1:-1]
    return t


def _normalize_wipe_target(tok: str) -> str:
    """Normalize a path token for catastrophic-target comparison."""
    t = _strip_quotes(tok).strip()
    t = re.sub(r"\$\{?HOME\}?", "~", t, flags=re.IGNORECASE)
    if t.endswith("/") and t not in ("/", "~/"):
        t = t.rstrip("/")
    return t


def _is_forbidden_wipe_target(tok: str) -> bool:
    """True for home/root/system wipes — not for explicit subdirs."""
    p = _normalize_wipe_target(tok)
    if not p:
        return False
    if _is_windows_wipe_target(tok):
        return True
    low = p.lower()
    if low in _FORBIDDEN_WIPE_EXACT:
        return True
    # Exact home directory of a user: /home/alice or /Users/alice
    if re.fullmatch(r"/home/[^/]+", p) or re.fullmatch(r"/Users/[^/]+", p):
        return True
    return low in ("~", "~/")


def _windows_argv(segment: str) -> list[str]:
    """Tokenize cmd-shaped strings without POSIX backslash escaping.

    This is a classifier, not a cmd/PowerShell interpreter. Runtime variable
    construction and arbitrary scripts cannot be proven safe by string rules.
    """
    lexer = shlex.shlex(segment.strip(), posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    lexer.commenters = ""
    lexer.quotes = '"'
    try:
        return list(lexer)
    except ValueError:
        return []


def _windows_basename(token: str) -> str:
    base = ntpath.basename(token).casefold()
    return base[:-4] if base.endswith(".exe") else base


def _is_windows_wipe_target(token: str) -> bool:
    """Recognize drive/share roots and profile roots, retaining subdirs."""
    raw = _strip_quotes(token).strip().casefold().replace("/", "\\")
    # Only Windows-shaped operands: retain the POSIX wipe policy verbatim.
    if not ("\\" in token or "%" in token or re.match(r"^[a-z]:", raw) or raw.startswith("$env:")):
        return False
    raw = re.sub(r"\$env:(userprofile|homepath)", r"%\1%", raw)
    symbolic = ntpath.normpath(raw).rstrip("\\")
    symbolic = re.sub(r"\\\*(?:\.\*)?$", "", symbolic)
    if symbolic in {"%userprofile%", "%homepath%", "%homedrive%%homepath%"}:
        return True
    env = {k.casefold(): v for k, v in os.environ.items()}
    expanded = re.sub(r"%([^%]+)%", lambda m: env.get(m[1], m[0]), raw)
    norm = ntpath.normcase(ntpath.normpath(expanded))
    drive, tail = ntpath.splitdrive(norm)
    if tail in ("\\", "\\*", "\\*.*") or (drive and not tail):
        return True
    root = re.sub(r"\\\*(?:\.\*)?$", "", norm).rstrip("\\")
    if re.fullmatch(r"[a-z]:\\(?:users|documents and settings)(?:\\[^\\]+)?", root):
        return True
    if re.fullmatch(r"[a-z]:\\(?:windows|program files(?: \(x86\))?|programdata)", root):
        return True
    homes = [env.get("userprofile", ""), env.get("homedrive", "") + env.get("homepath", "")]
    return any(home and root == ntpath.normcase(ntpath.normpath(home)).rstrip("\\") for home in homes)


def _windows_delete_targets(argv: list[str]) -> list[str]:
    """Operands of rd/rmdir/del/erase, excluding their slash switches."""
    return [a for a in argv[1:] if not re.fullmatch(r"(?:/(?:[fpqs?]|a(?::[a-z-]+)?))+", a, re.I)]


def _windows_kills_gateway(argv: list[str]) -> bool:
    if not argv:
        return False
    base = _windows_basename(argv[0])
    args = [a.casefold() for a in argv[1:]]
    if base in {"cmd", "powershell", "pwsh"}:
        flags = {"/c", "/k"} if base == "cmd" else {"-command", "-c"}
        for i, arg in enumerate(args, 1):
            if arg in flags and i + 1 < len(argv):
                return kills_sharedagents_gateway(_strip_quotes(" ".join(argv[i + 1 :])))
        return False
    is_kill = base in {"taskkill", "stop-process", "stop-service", "remove-service"}
    is_kill |= base in {"sc", "net"} and any(a in {"stop", "delete"} for a in args)
    if not is_kill:
        return False
    if _GATEWAY_KILL_TARGET_RE.search(" ".join(args)):
        return True
    if base == "taskkill":
        pids = {args[i + 1] for i, a in enumerate(args[:-1]) if a == "/pid"}
        if pids:
            home = Path(os.environ.get("SHAREDAGENTS_HOME") or Path.home() / ".sharedagents")
            for name in ("gateway.pid", "orchestrator.pid", "web.pid"):
                try:
                    if str(int((home / name).read_text().strip())) in pids:
                        return True
                except (OSError, ValueError):
                    pass
    return False


def _classify_windows_segment(segment: str) -> CommandSafety | None:
    argv = _windows_argv(segment)
    if not argv:
        return None
    base = _windows_basename(argv[0])
    args = [a.casefold() for a in argv[1:]]
    if _windows_kills_gateway(argv):
        return CommandSafety.FORBIDDEN
    if base in {"cmd", "powershell", "pwsh"}:
        flag = r"/[ck]" if base == "cmd" else r"-(?:command|c)"
        payload = re.search(r"\s" + flag + r"\s+(.+)$", segment, re.I | re.S)
        if payload:
            return classify_command(_strip_quotes(payload[1]))
        return CommandSafety.UNKNOWN
    if base in {"rd", "rmdir", "del", "erase"}:
        targets = _windows_delete_targets(argv)
        recursive = any(re.fullmatch(r"(?:/[fpqs?])+", a, re.I) and "/s" in a.casefold() for a in argv[1:])
        if (recursive or base in {"del", "erase"}) and any(_is_forbidden_wipe_target(t) for t in targets):
            return CommandSafety.FORBIDDEN
        if recursive or base in {"del", "erase"}:
            return CommandSafety.DESTRUCTIVE
        return CommandSafety.MUTATING
    if base == "remove-item":
        targets = [a for a in argv[1:] if not a.startswith("-")]
        if any(_is_forbidden_wipe_target(t) for t in targets):
            return CommandSafety.FORBIDDEN
        return CommandSafety.DESTRUCTIVE
    if base == "format":
        if any(re.fullmatch(r"[a-z]:\\?", a) for a in args):
            return CommandSafety.FORBIDDEN
        return CommandSafety.DESTRUCTIVE
    if base in {"takeown", "icacls"}:
        # Windows ownership/ACL equivalents of recursive chown/chmod.
        targets = (
            [argv[i + 1] for i, a in enumerate(args, 1) if a == "/f" and i + 1 < len(argv)]
            if base == "takeown"
            else argv[1:2]
        )
        if any(a in {"/r", "/t"} for a in args) and any(_is_forbidden_wipe_target(t) for t in targets):
            return CommandSafety.FORBIDDEN
        return CommandSafety.DESTRUCTIVE
    if base in {"taskkill", "diskpart", "stop-process", "stop-service", "remove-service"}:
        return CommandSafety.DESTRUCTIVE
    if (
        (base == "sc" and any(a in {"stop", "delete"} for a in args))
        or (base == "net" and args[:1] == ["stop"])
        or (base == "reg" and args[:1] == ["delete"])
    ):
        return CommandSafety.DESTRUCTIVE
    if base in {"dir", "where", "findstr"}:
        return CommandSafety.READ_ONLY
    if base in {"md", "copy", "move", "ren", "rename"}:
        return CommandSafety.MUTATING
    # Case-insensitive cmd builtins; leave POSIX parsing of other commands alone.
    if oscompat.IS_WINDOWS and base in {"echo", "type", "cd", "mkdir"}:
        return CommandSafety.MUTATING if base == "mkdir" else CommandSafety.READ_ONLY
    return None


def _rm_is_recursive(argv: list[str]) -> tuple[bool, bool, list[str]]:
    """Parse ``rm`` argv → (recursive, force, targets)."""
    recursive = False
    force = False
    targets: list[str] = []
    i = 1
    n = len(argv)
    while i < n:
        a = argv[i]
        if a == "--":
            targets.extend(argv[i + 1 :])
            break
        if a.startswith("--"):
            key = a.split("=", 1)[0].lower()
            if key in ("--recursive", "--dir", "--directory"):
                recursive = True
            elif key == "--force":
                force = True
            i += 1
            continue
        if a.startswith("-") and a != "-":
            chars = a[1:]
            if any(c in chars for c in "rR"):
                recursive = True
            if any(c in chars for c in "fF"):
                force = True
            i += 1
            continue
        targets.append(a)
        i += 1
    return recursive, force, targets


def _has_recursive_flag(argv: list[str]) -> bool:
    for a in argv[1:]:
        if a in ("-R", "--recursive"):
            return True
        if a.startswith("--"):
            continue
        if a.startswith("-") and "R" in a:
            return True
    return False


def _chmod_chown_path_args(argv: list[str]) -> list[str]:
    """Return path operands for chmod/chown (skip options + mode/owner)."""
    paths: list[str] = []
    i = 1
    n = len(argv)
    saw_mode = False
    while i < n:
        a = argv[i]
        if a == "--":
            rest = argv[i + 1 :]
            if not saw_mode and rest:
                rest = rest[1:]  # skip mode/owner
            paths.extend(rest)
            break
        if a.startswith("-") and a != "-":
            # chmod -R / chown -R …; long opts may take values — skip.
            if a in ("-R", "--recursive"):
                i += 1
                continue
            if a.startswith("--reference") or a in (
                "--from",
                "-h",
                "--no-dereference",
                "--preserve-root",
                "--no-preserve-root",
                "--quiet",
                "-f",
                "--silent",
                "--verbose",
                "-v",
                "-c",
                "--changes",
            ):
                if a.startswith("--reference") and "=" not in a and i + 1 < n:
                    i += 2
                else:
                    i += 1
                continue
            i += 1
            continue
        if not saw_mode:
            saw_mode = True  # mode (chmod) or owner (chown)
            i += 1
            continue
        paths.append(a)
        i += 1
    return paths


# Branch-listing flags that make ``git branch <flags>`` a pure query. A
# POSITIONAL argument without ``--list`` creates a branch (mutating), so a
# positional is only admitted when ``--list`` is present (a filter pattern).
# Unknown flags keep the call out of the read-only path (``--track`` /
# ``--set-upstream-to`` / ``--unset-upstream`` mutate branch config).
_GIT_BRANCH_LIST_FLAGS: frozenset[str] = frozenset(
    {
        "-a",
        "--all",
        "-r",
        "--remotes",
        "-v",
        "-vv",
        "--verbose",
        "--list",
        "--show-current",
        "--no-color",
        "--no-abbrev",
        "--merged",
        "--no-merged",
        "--contains",
        "--no-contains",
        "--points-at",
        "--column",
        "--no-column",
    }
)

# ``git branch`` flags whose ARGUMENT is data (a rev), not a branch to
# create. ``git branch -a --contains 0f80ef4`` is a pure query; treating
# the rev as a positional made the call UNKNOWN (→ gated) — the exact
# false negative seen in plan/explore research sessions.
_GIT_BRANCH_ARG_FLAGS: tuple[str, ...] = (
    "--contains",
    "--no-contains",
    "--merged",
    "--no-merged",
    "--points-at",
)


def _git_branch_operand_tokens(tokens: list[str]) -> list[str]:
    """Drop the value consumed by a ``--contains``-family flag."""
    out: list[str] = []
    skip = False
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok in _GIT_BRANCH_ARG_FLAGS:
            skip = True
            continue
        out.append(tok)
    return out


def _git_branch_flag_is_listing(tok: str) -> bool:
    if tok in _GIT_BRANCH_LIST_FLAGS:
        return True
    if tok.startswith(("--sort=", "--format=", "--column=")):
        return True
    # ``--contains=<rev>`` (and siblings) are the same query as the
    # two-token form — the ``=`` side is data, not a branch name.
    return any(tok.startswith(flag + "=") for flag in _GIT_BRANCH_ARG_FLAGS)


def _opt_is_prefix_or_family(opt: str, long_opt: str) -> bool:
    """True when a Git long-option token ``opt`` names ``long_opt``.

    Git's option parser accepts ANY unambiguous prefix of a long option
    (``git push --mir`` == ``--mirror``, ``git tag --del v1`` deletes the
    tag, ``git fsck --lost`` writes ``.git/lost-found``, ``git reset
    --har`` resets — all verified live against git 2.43). A classifier
    that matches only the exact spelling lets the destructive/mutating
    form slip through as READ_ONLY/UNKNOWN (→ auto-allow). So:

    - ``long_opt.startswith(opt)`` — ``opt`` is a Git-accepted
      abbreviation of ``long_opt``. This ALSO fires for AMBIGUOUS
      prefixes (``--for`` could be ``--force`` or ``--format``; git
      rejects them, but the requirement is to choose the conservative
      (gated) side rather than promise read-only).
    - ``opt.startswith(long_opt)`` — ``opt`` is a longer family member
      that starts with the destructive option (``--force-with-lease``
      starts with ``--force``; ``--output-indicator-new=`` starts with
      ``--output``).

    ``--`` (the option terminator) never counts as a prefix. A ``=value``
    suffix is stripped before matching (``--out=/tmp/x`` is the
    ``--output`` abbreviation with a value).
    """
    name = opt.split("=", 1)[0]
    return name != "--" and (long_opt.startswith(name) or name.startswith(long_opt))


# Push long options whose presence is DESTRUCTIVE (they delete or overwrite
# remote refs). ``--force`` covers the whole ``--force*`` family
# (``--force-with-lease`` / ``--force-if-includes`` start with it); the
# abbreviation direction of ``_opt_is_prefix_or_family`` covers
# ``--mir`` / ``--del`` / ``--pru`` / ``--forc`` … — every one verified
# live as accepted by git (and ``--del`` / ``-- :main`` really deleting).
_GIT_PUSH_DESTRUCTIVE_LONG: tuple[str, ...] = (
    "--mirror",
    "--delete",
    "--prune",
    "--force",
)


def _git_push_args_destructive(args: list[str]) -> bool:
    """True when ``git push`` arguments carry a DESTRUCTIVE form.

    - ``--mirror`` / ``--delete`` / ``--prune`` / ``--force*`` and every
      Git-accepted abbreviation of them (ambiguous prefixes gate
      conservatively — see ``_opt_is_prefix_or_family``);
    - short clusters containing ``f`` or ``d`` (``-f`` / ``-d`` / ``-fd``);
    - a force refspec (``+main:main``) or an empty-source delete refspec
      (``:main``) — INCLUDING after the ``--`` terminator: the scan does
      NOT stop at ``--``, because ``git push origin -- :main`` deletes the
      remote branch (verified live). Tokens after ``--`` are positionals
      only, so only the refspec checks apply there.
    """
    after_ddash = False
    for a in args:
        if a == "--":
            after_ddash = True
            continue
        if after_ddash:
            # Option parsing ended; only positional REFSPECS remain — still
            # capable of deleting / non-fast-forward-overwriting remote refs.
            if a.startswith("+"):
                return True
            src, sep, _dst = a.partition(":")
            if sep and not src:
                return True
            continue
        if a.startswith("--"):
            if any(_opt_is_prefix_or_family(a, d) for d in _GIT_PUSH_DESTRUCTIVE_LONG):
                return True
            continue  # unknown long option → MUTATING floor (never READ_ONLY)
        if a.startswith("-") and len(a) > 1:
            if "f" in a[1:] or "d" in a[1:]:
                return True  # `-f` / `-d` / clusters
            continue
        if a.startswith("+"):
            return True  # force refspec `+main:main`
        src, sep, _dst = a.partition(":")
        if sep and not src:
            return True  # `:main` deletes the remote ref
    return False


def _classify_git_branch(argv: list[str]) -> CommandSafety | None:
    """Classify ``git branch`` forms (global-option aware).

    ``-d`` / ``--delete`` → MUTATING (git refuses unless merged).
    ``-D`` / ``-df`` / ``-fd`` (combined ``-d -f``) / ``--delete --force``
    → DESTRUCTIVE (force-delete works on unmerged branches). ``-f`` /
    ``--force`` → DESTRUCTIVE (force-create / reset of an existing
    branch), ``-M`` / ``-C`` → DESTRUCTIVE (force rename / force copy
    replace an existing branch). Git accepts any unambiguous prefix of a
    long option (``git branch --forc x`` force-resets — verified live),
    so prefixes of ``--force`` / ``--delete`` classify the same way.
    Listing forms (no positional arg, or ``--list`` with a pattern) →
    READ_ONLY. A positional branch NAME without ``--list`` creates a branch
    → None (the caller gates it). A call carrying a config-injection global
    (``-c`` / ``--config-env=``) can execute an arbitrary program via
    ``core.pager`` etc., so it is NEVER READ_ONLY here.
    """
    if not argv or argv[0].rsplit("/", 1)[-1] != "git":
        return None
    rest = _git_globals_stripped(argv, escalate=True)
    if not rest or rest[0] != "branch":
        return None
    tokens = rest[1:]
    # Case-sensitive flag scan — ``-d`` must NOT equal ``-D``. Combined
    # short-flag clusters (``-df`` = ``-d -f``) must not fall through to
    # UNKNOWN → auto-allow: ``git branch -df feature/x`` force-deletes an
    # unmerged branch.
    delete = False
    for a in tokens:
        if a == "--":
            break
        if a == "-D":
            return CommandSafety.DESTRUCTIVE
        if a in ("-M", "-C"):
            return CommandSafety.DESTRUCTIVE  # force rename / force copy
        if a in ("-f", "--force"):
            return CommandSafety.DESTRUCTIVE  # force create / reset
        if a == "-d" or a == "--delete":
            delete = True
            continue
        if a.startswith("--"):
            if _opt_is_prefix_or_family(a, "--delete"):
                delete = True  # `--del` … still the merged-only delete
                continue
            if _opt_is_prefix_or_family(a, "--force"):
                return CommandSafety.DESTRUCTIVE  # `--forc` (verified live)
            continue
        if len(a) > 2 and a.startswith("-") and not a.startswith("--"):
            chars = a[1:]
            if "D" in chars or "M" in chars or "C" in chars:
                return CommandSafety.DESTRUCTIVE  # `-D` / `-M` / `-C` clusters
            if "d" in chars:
                delete = True
                if "f" in chars:
                    return CommandSafety.DESTRUCTIVE  # `-df` / `-fd`
                continue
            if "f" in chars:
                return CommandSafety.DESTRUCTIVE  # force present in cluster
            continue
    if delete:
        return CommandSafety.MUTATING
    operands = _git_branch_operand_tokens(tokens)
    positional = [a for a in operands if a != "--" and not a.startswith("-")]
    flags = [a for a in operands if a != "--" and a.startswith("-")]
    if "--list" in flags:
        return _git_read_only_verdict(argv, CommandSafety.READ_ONLY)
    if not positional and all(_git_branch_flag_is_listing(a) for a in flags):
        return _git_read_only_verdict(argv, CommandSafety.READ_ONLY)
    return None


def _classify_git_reflog(argv: list[str]) -> CommandSafety:
    """Classify ``git reflog`` forms.

    Bare ``git reflog`` (alias for ``git reflog show``) and the ``show`` /
    ``exists`` subcommands are read-only queries. ``expire`` / ``delete``
    remove reflog history → MUTATING; any other/unknown subcommand is
    MUTATING (conservative: only provably read-only forms are READ_ONLY).
    Abbreviation audit: git does NOT abbreviate reflog SUBcommands
    (``git reflog exp`` runs ``show exp`` and fails — verified live), and
    any token that is a prefix of ``expire`` / ``delete`` lands in the
    MUTATING fall-through here — never READ_ONLY.
    """
    if not argv or argv[0].rsplit("/", 1)[-1] != "git":
        return CommandSafety.UNKNOWN
    rest = _git_globals_stripped(argv, escalate=True)
    if not rest or rest[0] != "reflog":
        return CommandSafety.UNKNOWN
    args = rest[1:]
    if not args or args[0] in ("show", "exists") or args[0].startswith("-"):
        return _git_read_only_verdict(argv, CommandSafety.READ_ONLY)
    return CommandSafety.MUTATING


def _classify_git_fsck(argv: list[str]) -> CommandSafety:
    """Classify ``git fsck`` forms.

    Plain fsck (dangling / connectivity checks) is read-only; ``--lost-found``
    WRITES recovered objects into ``.git/lost-found`` → MUTATING — and so
    does EVERY Git-accepted abbreviation of it (``git fsck --lost`` and
    ``--lost-fo`` both wrote ``.git/lost-found/other/<sha>`` in live probes),
    via the ``--lost-found``-prefix rule below. Ambiguous prefixes
    (``--l`` is only ``--lost-found`` here) gate conservatively.
    """
    if not argv or argv[0].rsplit("/", 1)[-1] != "git":
        return CommandSafety.UNKNOWN
    rest = _git_globals_stripped(argv, escalate=True)
    if not rest or rest[0] != "fsck":
        return CommandSafety.UNKNOWN
    for a in rest[1:]:
        if a.startswith("--") and _opt_is_prefix_or_family(a, "--lost-found"):
            return CommandSafety.MUTATING
    return _git_read_only_verdict(argv, CommandSafety.READ_ONLY)


def _classify_git_diff(argv: list[str]) -> CommandSafety:
    """Classify ``git diff`` forms.

    Plain diff is read-only; ``--output=<file>`` (and the two-token
    ``--output <file>`` form) WRITES a file and ``--ext-diff`` EXECUTES the
    configured external-diff helper → MUTATING. Abbreviation audit: every
    unambiguous prefix of those options (``--out=…``, ``--ext`` …) and every
    ``--output*`` family member (``--output-indicator-new=…``) gates the
    same way — git 2.43's diff happens to reject short prefixes, but a
    future git must not turn them read-only; ambiguous prefixes take the
    conservative MUTATING side.
    """
    if not argv or argv[0].rsplit("/", 1)[-1] != "git":
        return CommandSafety.UNKNOWN
    rest = _git_globals_stripped(argv, escalate=True)
    if not rest or rest[0] != "diff":
        return CommandSafety.UNKNOWN
    for a in rest[1:]:
        if a.startswith("--") and (
            _opt_is_prefix_or_family(a, "--ext-diff") or _opt_is_prefix_or_family(a, "--output")
        ):
            return CommandSafety.MUTATING
    return _git_read_only_verdict(argv, CommandSafety.READ_ONLY)


def _classify_git_push(argv: list[str]) -> CommandSafety:
    """Classify ``git push`` forms.

    Plain push (remote + refspecs) is MUTATING — it always mutates remote
    refs. DESTRUCTIVE forms: ``--mirror`` (deletes remote refs absent
    locally), ``--delete`` / ``-d`` (removes remote refs), ``--prune``
    (deletes remote-tracking refs), ``--force*`` / ``-f`` (non-fast-forward
    overwrite), a force refspec (``+main:main``) and an empty-source
    refspec (``:main`` — the classic delete form) — INCLUDING after ``--``
    (``git push origin -- :main`` deletes the remote branch — verified
    live). Git accepts any unambiguous abbreviation of a long option
    (``--mir``, ``--del``, ``--pru``, ``--forc`` … — verified live), so the
    shared ``_git_push_args_destructive`` handles every prefix; ambiguous
    prefixes gate conservatively. Unknown long options keep the MUTATING
    floor — push is never READ_ONLY.
    """
    if not argv or argv[0].rsplit("/", 1)[-1] != "git":
        return CommandSafety.UNKNOWN
    rest = _git_globals_stripped(argv, escalate=True)
    if not rest or rest[0] != "push":
        return CommandSafety.UNKNOWN
    if _git_push_args_destructive(rest[1:]):
        return CommandSafety.DESTRUCTIVE
    return CommandSafety.MUTATING


def _classify_git_stash_destructive(args: list[str]) -> bool:
    """``git stash clear`` / ``drop`` / ``pop`` / ``branch`` remove stash
    entries permanently → DESTRUCTIVE.

    ``pop`` removes the applied entry and ``branch`` drops it after a
    successful application (git docs) — both destroy the stash entry
    exactly like ``drop`` (codex round-3 audit: they classified UNKNOWN
    and slipped the gate). ``create`` / ``store`` keep entries (and are
    plumbing), ``list`` / ``show`` / ``apply`` don't remove them.

    Git rejects stash SUBcommand abbreviations today (``git stash dr``
    errors — verified live), but a prefix token (``cl``, ``dr``, ``po``,
    ambiguous ``p`` = pop-or-push, ``c`` = clear-or-create) must still
    gate rather than auto-allow, so any prefix of ``clear`` / ``drop`` /
    ``pop`` / ``branch`` counts. Exact forms included.
    """
    if not args or args[0].startswith("-") or args[0] == "--":
        return False
    verb = args[0]
    return "clear".startswith(verb) or "drop".startswith(verb) or "pop".startswith(verb) or "branch".startswith(verb)


def _classify_git_tag_destructive(args: list[str]) -> bool:
    """``git tag -d`` / ``--delete`` removes a tag; ``-f`` / ``--force``
    re-points (or with ``-d``, force-deletes) an existing tag →
    DESTRUCTIVE.

    Git accepts any unambiguous prefix of a long option (``git tag --del
    v1`` DELETED the tag in a live probe), so prefixes of ``--delete`` /
    ``--force`` count too; ambiguous prefixes (``--f`` = ``--force`` or
    ``--format``) gate conservatively (True). Short clusters containing
    ``d`` / ``f`` count the same way.
    """
    for a in args:
        if a == "--":
            break
        if a in ("-d", "-f"):
            return True
        if a.startswith("-") and not a.startswith("--") and len(a) > 2:
            if "d" in a[1:] or "f" in a[1:]:
                return True
            continue
        if a.startswith("--"):
            if _opt_is_prefix_or_family(a, "--delete") or _opt_is_prefix_or_family(a, "--force"):
                return True
    return False


def _classify_git_remote(argv: list[str]) -> CommandSafety | None:
    """Classify ``git remote`` listing forms (global-option aware).

    Bare ``git remote``, ``-v`` / ``--verbose``, ``show <name>`` and
    ``get-url <name>`` are read-only queries. Every other verb (``add`` /
    ``remove`` / ``rename`` / ``set-url`` / ``set-head`` / ``prune`` /
    ``update``) mutates remotes or remote-tracking refs → None so the caller
    gates the call.
    """
    if not argv or argv[0].rsplit("/", 1)[-1] != "git":
        return None
    rest = _git_globals_stripped(argv, escalate=True)
    if not rest or rest[0] != "remote":
        return None
    toks = rest[1:]
    if not toks or toks == ["-v"] or toks == ["--verbose"]:
        return _git_read_only_verdict(argv, CommandSafety.READ_ONLY)
    if toks[0] in ("show", "get-url") and len(toks) >= 2:
        return _git_read_only_verdict(argv, CommandSafety.READ_ONLY)
    return None


def _classify_git_destructive_argv(argv: list[str]) -> CommandSafety | None:
    """Classify destructive ``git`` forms the whole-string regexes miss once
    global options sit between ``git`` and the subcommand
    (``git -C /repo push --force …`` — the ``\\bgit\\s+push\\b`` regex
    requires adjacency). Argv-aware, so it fires with or without globals and
    can only raise severity.
    """
    if not argv or argv[0].rsplit("/", 1)[-1] != "git":
        return None
    rest = _git_globals_stripped(argv, escalate=True)
    if not rest:
        return None
    sub, args = rest[0], rest[1:]
    if sub == "push" and _git_push_args_destructive(args):
        return CommandSafety.DESTRUCTIVE
    if sub == "reset":
        # `--hard` (and every Git-accepted abbreviation of it — `git reset
        # --har HEAD~1` reset the branch in a live probe) discards working
        # tree + index changes → DESTRUCTIVE. Stop at `--`: anything after
        # the terminator is a PATHSPEC, not the --hard option.
        for a in args:
            if a == "--":
                break
            if a.startswith("--") and _opt_is_prefix_or_family(a, "--hard"):
                return CommandSafety.DESTRUCTIVE
    if sub == "update-ref":
        # update-ref only rewrites/creates/deletes refs — `-d` deletes,
        # `<ref> <new> [<old>]` force-sets, `--stdin` replays batches.
        return CommandSafety.DESTRUCTIVE
    if sub == "stash" and _classify_git_stash_destructive(args):
        return CommandSafety.DESTRUCTIVE
    if sub == "tag" and _classify_git_tag_destructive(args):
        return CommandSafety.DESTRUCTIVE
    if sub == "clean" and any(a.startswith("-") and "f" in a for a in args):
        return CommandSafety.DESTRUCTIVE
    return None


def _extract_cmd_sub_interiors(text: str) -> list[str]:
    """Pull ``$(…)`` / backtick interiors out for nested classification."""
    out: list[str] = []
    i, n = 0, len(text)
    quote: str | None = None
    while i < n:
        c = text[i]
        if quote:
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            i += 1
            continue
        if c == "`":
            j = i + 1
            while j < n and text[j] != "`":
                j += 1
            if j < n:
                out.append(text[i + 1 : j])
                i = j + 1
                continue
        if c == "$" and i + 1 < n and text[i + 1] == "(":
            depth = 1
            j = i + 2
            while j < n and depth:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                out.append(text[i + 2 : j - 1])
                i = j
                continue
        i += 1
    return out


def _classify_catastrophic_argv(argv: list[str]) -> CommandSafety | None:
    """Argv-aware FORBIDDEN/DESTRUCTIVE for rm/find/chmod/chown bombs."""
    if not argv:
        return None
    base = argv[0].rsplit("/", 1)[-1]
    # Leading sudo — classify the inner command.
    if base == "sudo" and len(argv) >= 2:
        return _classify_catastrophic_argv(argv[1:])

    if base == "rm":
        recursive, force, targets = _rm_is_recursive(argv)
        if recursive and any(_is_forbidden_wipe_target(t) for t in targets):
            return CommandSafety.FORBIDDEN
        # Recursive rm of a non-catastrophic path stays approvable-destructive
        # (e.g. worktree wipe, build directory, recursive directory cleanup).
        if recursive:
            return CommandSafety.DESTRUCTIVE
        # Non-recursive file removal (e.g. `rm file.txt`, `rm -f /tmp/temp.json`)
        # is routine workspace mutation, permitted in auto mode without modal prompts.
        return CommandSafety.MUTATING

    if base == "find" and "-delete" in argv:
        # Paths are typically non-option args before actions.
        paths = [a for a in argv[1:] if a != "-delete" and not a.startswith("-")]
        if any(_is_forbidden_wipe_target(p) for p in paths):
            return CommandSafety.FORBIDDEN
        return CommandSafety.DESTRUCTIVE

    if base in ("chmod", "chown", "chgrp") and _has_recursive_flag(argv):
        paths = _chmod_chown_path_args(argv)
        if any(_is_forbidden_wipe_target(p) for p in paths):
            return CommandSafety.FORBIDDEN
        return None  # scoped recursive chmod stays MUTATING via segment path

    git_br = _classify_git_branch(argv)
    if git_br is not None:
        return git_br

    git_dest = _classify_git_destructive_argv(argv)
    if git_dest is not None:
        return git_dest

    return None


def _dash_c_payload(argv: list[str]) -> str | None:
    """Return the ``-c`` script argument for bash/sh/zsh, if present."""
    for i, a in enumerate(argv):
        if a == "-c" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("-c") and len(a) > 2 and not a.startswith("--"):
            return a[2:]
    return None


def _classify_segment_full(segment: str) -> CommandSafety:
    """Classify one simple command including forbidden/destructive."""
    windows = _classify_windows_segment(segment)
    if windows is not None:
        return windows
    try:
        argv = shlex.split(segment)
    except ValueError:
        argv = []
        # A trailing backslash is ordinary cmd path data, including inside a
        # quoted echo. It must not make printed examples inherit their verbs.
        win_argv = _windows_argv(segment)
        if win_argv and _windows_basename(win_argv[0]) in _LITERAL_ARG_COMMANDS:
            return CommandSafety.READ_ONLY

    if argv:
        base = argv[0].rsplit("/", 1)[-1]
        # Literal-arg commands: args are data. Skip pattern inheritance.
        if base in _LITERAL_ARG_COMMANDS:
            return CommandSafety.READ_ONLY

        # `env [VAR=val …] git …` / `command [-p|-v|-V] git …` EXECUTE the
        # wrapped command — classify the REAL one, so a wrapped destructive
        # git form can never ride the wrapper's blanket READ_ONLY tag into
        # an auto/plan ALLOW. Recursing through _classify_segment_full
        # keeps the catastrophic/pattern checks on the inner command too
        # (`env rm -rf /` → FORBIDDEN). Bare `env`/`command` (no command)
        # fall through unchanged.
        inner = _unwrap_exec_wrapper(argv)
        if inner is not None:
            return _classify_segment_full(shlex.join(inner))

        cat = _classify_catastrophic_argv(argv)
        if cat is not None:
            return cat

        # Nested shells execute their -c payload.
        if base in ("bash", "sh", "zsh", "dash", "ksh"):
            payload = _dash_c_payload(argv)
            if payload:
                return classify_command(payload)

    # Pattern match on THIS segment only (comments already stripped
    # by ``_split_command``), so narrative ``# rm -rf`` cannot trip.
    for pat in _FORBIDDEN_PATTERNS:
        if pat.search(segment):
            return CommandSafety.FORBIDDEN
    for pat in _DESTRUCTIVE_PATTERNS:
        if pat.search(segment):
            return CommandSafety.DESTRUCTIVE

    return _classify_segment(segment)


def classify_command(command: str) -> CommandSafety:
    """Heuristic-classify a shell command string.

    A pipeline / chain is classified as the most-cautious of its
    segments, so a read-only-only pipeline (``cat f | grep x``) stays
    read-only and runs in plan mode, while a destructive segment
    anywhere (``ls && rm -rf x``) trips the whole command.

    Forbidden/destructive regexes run **per segment** (after comment
    stripping), not over the raw whole string — so ``which mkfs`` and
    ``echo "rm -rf /"`` no longer inherit arg danger. Command-
    substitution interiors are classified recursively.

    Examples:
        >>> classify_command("ls -la").name
        'READ_ONLY'
        >>> classify_command('echo "rm -rf /"').name
        'READ_ONLY'
        >>> classify_command("rm -rf /tmp/foo").name
        'DESTRUCTIVE'
        >>> classify_command("rm -rf ~/").name
        'FORBIDDEN'
        >>> classify_command("git status").name
        'READ_ONLY'
        >>> classify_command("git -C /repo status").name
        'READ_ONLY'
        >>> classify_command("git branch -d feature/x").name
        'MUTATING'
        >>> classify_command("git branch -D feature/x").name
        'DESTRUCTIVE'
        >>> classify_command("git push --force origin main").name
        'DESTRUCTIVE'
        >>> classify_command("git -C /repo push --force origin main").name
        'DESTRUCTIVE'
    """
    text = (command or "").strip()
    if not text:
        return CommandSafety.UNKNOWN

    # Gateway self-protection FIRST — FORBIDDEN in every mode.
    if kills_sharedagents_gateway(text):
        return CommandSafety.FORBIDDEN
    if kills_own_process_group(text):
        return CommandSafety.FORBIDDEN

    # Quote-aware split; comments are dropped from segments here.
    segments, has_file_write, has_cmd_sub = _split_command(text)
    segments = [s for s in segments if not _is_comment_only_segment(s)]
    interiors = _extract_cmd_sub_interiors(text) if has_cmd_sub else []

    if not segments and not interiors:
        return CommandSafety.UNKNOWN

    worst = CommandSafety.READ_ONLY
    for seg in segments:
        worst = _max_safety(worst, _classify_segment_full(seg))
    for inner in interiors:
        worst = _max_safety(worst, classify_command(inner))

    if has_file_write:
        worst = _max_safety(worst, CommandSafety.MUTATING)
    # Any command/process substitution is cautious even when a best-effort
    # parse found a read-only-looking interior. Shell expansion semantics are
    # too broad to promise the outer command is side-effect free; a mutating or
    # destructive interior still wins through the recursive max above.
    if has_cmd_sub:
        worst = _max_safety(worst, CommandSafety.UNKNOWN)
    return worst
