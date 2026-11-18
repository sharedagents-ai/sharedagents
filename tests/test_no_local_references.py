"""The shipped tree must not identify a deployment or the person who runs it.

AGENTS.md: host names, private IP addresses, home directories, usernames and
email addresses live only in the operator's sharedagents home, never in the
repository — not in ``src/``, and not in tests, docs, scripts or root files
either. Documentation ranges (``192.0.2.x``, ``203.0.113.x``, ``198.51.100.x``)
and ``example.com`` are the placeholders to use instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

#: Directories scanned in full, plus every tracked file at the repo root.
SCAN_DIRS = ("src", "tests", "docs", "scripts", "examples", "website", ".github")

# Scan every file, including extensionless assets and files in vendor
# directories. Decode with replacement so one non-UTF8 byte cannot hide a
# literal identity elsewhere in a bundled asset.
#: Placeholder user names a docstring may put under ``/home/`` or ``/Users/``.
_PLACEHOLDER_USERS = {
    "user",
    "username",
    "me",
    "you",
    "alice",
    "bob",
    "example",
    "<user>",
    "test",
    "jane",
    "private-builder-example",
}

#: Mailboxes that are protocol identifiers or the project's own placeholder
#: domain, never a real person's address: WhatsApp JIDs (``@s.whatsapp.net``,
#: ``@g.us``, ``@broadcast``) and ``*@sharedagents.{ai,local}`` test operators.
_ALLOWED_MAILBOX_DOMAINS = (
    r"example\.(?:com|org|net)|test|invalid|localhost|"
    r"s\.whatsapp\.net|g\.us|broadcast|sharedagents\.ai|sharedagents\.local"
)

PATTERNS: dict[str, re.Pattern[str]] = {
    "rental offer identifier": re.compile(r"\boffer\s+\d{6,}\b", re.IGNORECASE),
    "host nickname": re.compile(r"\bTB\d{2}\b", re.IGNORECASE),
    "private IPv4 (192.168/16)": re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b|192\.168\.(?!\d)"),
    "private IPv4 (10/8)": re.compile(r"(?<![\d.])10\.\d{1,3}\.\d{1,3}\.\d{1,3}(?![\d.])"),
    "private IPv4 (172.16/12)": re.compile(r"(?<![\d.])172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}(?![\d.])"),
    "home directory": re.compile(r"/home/([A-Za-z_<][A-Za-z0-9_.>-]*)"),
    "personal email": re.compile(
        r"[A-Z0-9][A-Z0-9._%+-]*@(?!(?:[A-Z0-9-]+\.)*(?:" + _ALLOWED_MAILBOX_DOMAINS + r")\b)"
        r"[A-Z0-9.-]+\.(?!(?:png|svg|jpg|webp)\b)[A-Z]{2,}\b(?![\w.:])",
        re.IGNORECASE,
    ),
    "other home directory": re.compile(r"(?:/Users/|[A-Z]:[\\/]Users[\\/])([A-Za-z_<][A-Za-z0-9_.>-]*)"),
    # ``host.docker.internal`` is Docker's own name for the host, identical on
    # every install, so it is not a deployment leak.
    "internal domain": re.compile(
        r"(?:https?://|[\"'])(?!host\.docker\.internal\b)([a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:internal|lan|local))\b",
        re.IGNORECASE,
    ),
}

#: Files allowed to carry specific labels — each entry documents why the
#: literal is a legitimate fixture rather than a deployment leak.
_EXEMPTIONS: dict[str, set[str]] = {
    # This file deliberately embeds one example of every pattern.
    "tests/test_no_local_references.py": {"*"},
}


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for dirname in SCAN_DIRS:
        root = REPO / dirname
        if not root.is_dir():
            continue
        files.extend(path for path in sorted(root.rglob("*")) if path.is_file() and "__pycache__" not in path.parts)
    files.extend(path for path in sorted(REPO.iterdir()) if path.is_file() and path.name != ".git")
    return files


def _hits(text: str) -> list[tuple[str, str]]:
    found = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            if label in {"home directory", "other home directory"} and match.group(1).lower() in _PLACEHOLDER_USERS:
                continue
            found.append((label, match.group(0)))
    return found


def test_guard_catches_each_pattern():
    assert _hits("served on TB99") and _hits("served on tb98")
    assert _hits("http://192.168.1.20:8000") and _hits('"192.168." in host')
    assert _hits("10.9.9.80:9092") and _hits("172.20.1.5")
    assert _hits("/home/somebody/src") and _hits("someone@gmail.com")


def test_guard_allows_placeholders():
    clean = (
        "http://192.0.2.10/v1 https://203.0.113.7 user@example.com "
        "/home/user/proj /home/alice ~/.sharedagents/config.json 4TB disk "
        "admin@sharedagents.ai dev@sharedagents.local "
        '"host.docker.internal:host-gateway" http://host.docker.internal:8000/v1 '
        "5511999990001@s.whatsapp.net 120363@g.us "
        "vLLM 0.10.1.2-style versions are not addresses: v10.2"
    )
    assert _hits(clean) == []


def test_shipped_source_has_no_local_references():
    files = _scanned_files()
    assert len(files) > 300  # the walk found the tree
    problems = []
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            pytest.fail(f"cannot scan {rel}: {exc}")
        exempt = _EXEMPTIONS.get(rel, set())
        for label, value in _hits(text):
            if "*" in exempt or label in exempt:
                continue
            if (
                rel == "src/sharedagents_web/static/vendor/onnxruntime/ort-wasm-simd-threaded.mjs"
                and label == "home directory"
                and value == "/home/web_user"
            ):
                continue
            problems.append(f"{rel}: {label}: {value}")
    if problems:
        pytest.fail("local references in shipped tree:\n" + "\n".join(problems[:50]))


def test_local_provider_detection_uses_address_ranges():
    """The usage page tells local from hosted by address range, not by a
    hardcoded LAN prefix in the source."""
    from sharedagents_web.usage_stats import _is_local_provider

    assert _is_local_provider("http://127.0.0.1/v1")
    assert _is_local_provider("http://192.0.2.10/v1")  # reserved range
    assert not _is_local_provider("http://1.1.1.1/v1")
    assert not _is_local_provider("api.example.com")


def test_guard_scans_unusual_extensions_and_vendor_directories(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path)
    asset = tmp_path / "vendor" / "nested" / "asset.unknown"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"prefix\xff /home/somebody/project")
    assert asset in _shipped_text_files()
    assert _hits(asset.read_text(encoding="utf-8", errors="replace"))


def _shipped_text_files() -> list[Path]:
    return [path for path in sorted(SRC.rglob("*")) if path.is_file() and "__pycache__" not in path.parts]


def test_guard_catches_other_homes_and_internal_domains():
    assert _hits("/Users/somebody/project")
    assert _hits(r"C:\Users\somebody\project")
    assert _hits("person@custom-domain.com")
    assert _hits("https://private-host.internal")
    assert _hits("'service.lan'")


def test_guard_catches_deployment_rental_identifiers():
    assert _hits("offer 00000000")
    assert _hits("offer\n00000000")
    assert not _hits("offer <OFFER_ID>")
