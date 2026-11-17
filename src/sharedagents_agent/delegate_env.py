"""Least-privilege environment for THIRD-PARTY delegate subprocesses.

sharedagents shall NOT hand the full parent ``os.environ`` to a third-party
(expensive / external CLI or ACP) delegate subprocess. That environment
can carry live infra credentials (API tokens, broker/DB creds); an
autonomous, prompt-injectable agentic
delegate with a shell tool can read any env var and exfiltrate it via its own
tool calls, independent of what appears in the prompt. So each third-party
subprocess gets ONLY an explicit allowlist of the vars it legitimately needs,
PLUS the delegate's OWN per-agent provider credentials (``_spawn_env_overrides``
/ ``api_keys``), which are the CLI's own auth by design and always included.

Everything else in ``os.environ`` is DROPPED (strict default-deny, fail-closed). ``local_llm`` (co-located, not a third
party) and sharedagents's OWN agent-over-stdio spawn are EXEMPT — they never route
through this module.

    >>> Fail toward FEWER vars, never toward inheriting all. If the allowlist
    >>> is empty/misconfigured we return the minimal safe set + overrides,
    >>> NEVER ``os.environ``.

=====================================================================
NOTE — allowlist MEMBERSHIP is not yet owner-confirmed (settled in
principle; membership still to be confirmed). The owner confirms the
final set before this ships to prod. Editing the policy = editing the two
tuples below; nothing else. Vars flagged ``# FLAG`` are the ones to scrutinise.
=====================================================================
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

# ---------------------------------------------------------------------------
# THE ALLOWLIST. Additions need maintainer review.
# Every entry is here because a delegate CLI (claude / codex / antigravity /
# opencode / cursor-agent) genuinely needs it to LAUNCH or to reach its
# provider API. None of these is an infra secret. Add sparingly; comment WHY.
# ---------------------------------------------------------------------------

#: Exact variable names copied through verbatim when present in ``os.environ``.
_ALLOWLIST_EXACT: frozenset[str] = frozenset(
    {
        # --- process basics: without these a CLI often will not even start ---
        "PATH",  # find the CLI's own helper binaries / node / git
        "HOME",  # locate per-tool config + stored auth (~/.claude, ~/.codex,
        # ~/.config, ~/.local/share) — how the CLIs read their login
        "CLAUDE_CONFIG_DIR",  # Claude Code's own config/settings home override
        # (settings.json + stored auth live here when set).
        # A PATH, not a secret; discovery
        # (model_discovery._claude_settings_model) reads the
        # SAME var, so the spawned CLI must see it too or
        # validation and runtime disagree on settings.
        "CODEX_HOME",  # the codex CLI's own home override (models_cache.json +
        # auth.json live here when set). A PATH, not a secret;
        # discovery (model_discovery._codex_cache_model_map)
        # honors it, so the spawn env must pass it through for the
        # CLI to resolve the same cache/auth home.
        "USER",  # some tools read the current username for display / config
        "LOGNAME",  # POSIX login name; benign, some tools prefer it over USER
        "SHELL",  # CLIs that shell out pick the user's shell from here
        "TMPDIR",  # scratch dir for the CLI; without it some write to CWD
        # --- locale / terminal: correct text + color rendering of CLI output ---
        "LANG",  # base locale (UTF-8 handling of the CLI's own output)
        "LANGUAGE",  # GNU locale fallback list
        "TERM",  # terminal type — CLIs disable/enable ANSI off this
        "COLORTERM",  # truecolor hint; benign, purely cosmetic
        "TZ",  # timezone for the CLI's own timestamps; benign
        # --- TLS trust store: outbound HTTPS to the provider API ---
        # These are FILE PATHS, not secrets. Omitting them breaks provider calls
        # on a host with a custom / corporate CA bundle.
        "SSL_CERT_FILE",  # OpenSSL CA bundle file
        "SSL_CERT_DIR",  # OpenSSL CA bundle dir
        "REQUESTS_CA_BUNDLE",  # python-requests based CLIs
        "CURL_CA_BUNDLE",  # libcurl based CLIs
        # --- egress proxy: if the operator routes ALL egress through a proxy the
        # provider call fails without these. A proxy URL *can* embed basic-auth
        # creds (http://user:pass@host); we allow the vars but STRIP the userinfo
        # before it reaches the child (see _PROXY_URL_VARS / _strip_url_userinfo).
        # NO_PROXY / no_proxy are host lists, never URLs — passed verbatim.
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        # --- node-based CLIs (claude / opencode / cursor-agent) ---
        # EXACT names only. A broad ``NODE_`` prefix would also copy through
        # NODE_AUTH_TOKEN (a real npm registry credential) and any other
        # NODE_*_{TOKEN,AUTH,KEY,SECRET,PASSWORD} the operator has set — a
        # secret leak. These three are flags / file paths, never creds:
        "NODE_OPTIONS",  # runtime flags (e.g. --max-old-space-size)
        "NODE_EXTRA_CA_CERTS",  # file path to an extra CA bundle (corporate TLS)
        "NODE_PATH",  # extra module resolution dirs (paths, not secrets)
        # Devin runtime controls. Exact names avoid forwarding DEVIN_* credentials
        # to unrelated delegates; provider credentials use per-agent overrides.
        "DEVIN_PERMISSION_MODE",
        "DEVIN_MODEL",
    }
)

#: Proxy vars whose VALUE is a URL that may embed ``user:pass@`` userinfo. We
#: keep the var (a proxy may be mandatory for egress) but strip the credential
#: before the child sees it. NO_PROXY/no_proxy are excluded — they are host
#: lists, not URLs.
_PROXY_URL_VARS: frozenset[str] = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    }
)

#: Prefix rules — any ``os.environ`` key starting with one of these is kept.
_ALLOWLIST_PREFIXES: tuple[str, ...] = (
    "LC_",  # all POSIX locale categories (LC_ALL, LC_CTYPE, …) — baseline
    "XDG_",  # XDG base-dir spec: where config/auth/cache live when the
    # operator relocated them (XDG_CONFIG_HOME etc.). Points to
    # DIRECTORIES, not secrets. opencode / cursor honor XDG.
    # NOTE: no ``NODE_`` prefix here — it would sweep in NODE_AUTH_TOKEN and
    # other NODE_*_{TOKEN,AUTH,KEY,SECRET,PASSWORD} secrets. The three safe
    # node vars are allowlisted by EXACT name in _ALLOWLIST_EXACT instead.
)

#: Fail-closed fallback: the smallest set that still lets a CLI launch. Used
#: only if the allowlist above is ever emptied/misconfigured — we degrade to
#: THIS (+ the delegate's own overrides), never to ``os.environ``.
_MINIMAL_SAFE: frozenset[str] = frozenset({"PATH", "HOME"})


def _keep(key: str, allow_exact: Iterable[str]) -> bool:
    if key in allow_exact:
        return True
    return any(key.startswith(p) for p in _ALLOWLIST_PREFIXES)


def _strip_url_userinfo(value: str) -> str | None:
    """Return ``value`` (a proxy URL) with any ``user:pass@`` userinfo removed.

    ``http://user:pass@host:3128/`` -> ``http://host:3128/``. Fail-closed: if
    the value does not parse into a scheme + network location, or reconstruction
    fails, return ``None`` so the caller DROPS the var rather than leaking a
    possibly-credentialed string to the child.
    """
    try:
        parts = urlsplit(value)
    except Exception:  # noqa: BLE001
        return None
    # A real proxy URL has a scheme and a host. Anything else (e.g. a bare
    # "host:port" with no scheme, which urlsplit mis-parses) is dropped.
    # Accessing ``.hostname``/``.port`` can raise ValueError on a malformed
    # port (non-numeric or out of range) — treat that as unparseable and DROP
    # fail-closed rather than let it propagate.
    try:
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not hostname:
        return None
    # Rebuild netloc from host[:port] ONLY — this discards username:password.
    netloc = hostname
    if port is not None:
        netloc = f"{netloc}:{port}"
    try:
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:  # noqa: BLE001
        return None


def build_third_party_spawn_env(
    overrides: Mapping[str, str] | None = None,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the EXACT environment for a third-party delegate subprocess.

    Copies through only the least-privilege allowlist from ``base_env``
    (defaults to ``os.environ``), then overlays ``overrides`` — the delegate's
    OWN per-agent provider credentials (``_spawn_env_overrides`` / ``api_keys``),
    which ALWAYS win and are ALWAYS included. Never returns ``os.environ`` and
    never returns ``None``.

    Fail-closed: if the allowlist is somehow empty, degrade to
    :data:`_MINIMAL_SAFE` (+ overrides), not to the full parent env.

    Args:
        overrides: the delegate's own credentials/env to force-include.
        base_env: source env to filter (test seam; defaults to ``os.environ``).

    Returns:
        A fresh ``dict[str, str]`` safe to hand to ``create_subprocess_exec``.
    """
    src: Mapping[str, str] = os.environ if base_env is None else base_env
    allow_exact: Iterable[str] = _ALLOWLIST_EXACT if _ALLOWLIST_EXACT else _MINIMAL_SAFE

    out: dict[str, str] = {}
    for k, v in src.items():
        key, val = str(k), str(v)
        if not _keep(key, allow_exact):
            continue
        if key in _PROXY_URL_VARS:
            # Strip embedded ``user:pass@`` credentials before the child sees
            # the proxy URL. Fail-closed: unparseable → DROP the var entirely.
            sanitized = _strip_url_userinfo(val)
            if sanitized is None:
                continue
            val = sanitized
        out[key] = val
    if overrides:
        # Per-agent provider creds ALWAYS win and are always present, even if
        # a key also happens to sit in the parent env.
        out.update({str(k): str(v) for k, v in overrides.items()})
    return out


__all__ = ["build_third_party_spawn_env"]
