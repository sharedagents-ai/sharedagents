"""The release scrub reads a private denylist and matches accented names."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_forbidden_patterns.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_forbidden_patterns", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_private_denylist_from_env_matches_ascii_and_accented_terms(tmp_path, monkeypatch, capsys):
    deny = tmp_path / "deny.txt"
    deny.write_text("# private terms\nZéphyrine Quux\nplaceholder-host\n", encoding="utf-8")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "clean.py").write_text("print('hello')\n", encoding="utf-8")
    monkeypatch.setenv("SHAREDAGENTS_SCRUB_DENYLIST", str(deny))

    mod = _load()
    assert mod.main([str(tree)]) == 0

    (tree / "upper.md").write_text("Author: ZÉPHYRINE QUUX\n", encoding="utf-8")
    (tree / "ascii.txt").write_text("ssh placeholder-host\n", encoding="utf-8")
    assert mod.main([str(tree)]) == 1
    out = capsys.readouterr().out
    assert "FORBIDDEN: upper.md" in out and "FORBIDDEN: ascii.txt" in out
    assert "clean.py" not in out.replace("scanned", "")


def test_accented_term_matches_decomposed_text(tmp_path):
    mod = _load()
    deny = tmp_path / "deny.txt"
    deny.write_text("Zéphyrine\n", encoding="utf-8")
    mod.DENYLIST[:] = mod.load_denylist(str(deny))
    assert mod.violations("ze\u0301phyrine".encode())  # NFD "é"
    assert not mod.violations(b"zephyrine")
    # Reserved example URLs stay exempt, as they are for ASCII terms.
    assert not mod.violations("see https://example.com/zéphyrine for a placeholder".encode())


@pytest.mark.parametrize(
    "value",
    [
        "ghp_" + "x" * 36,
        "github_pat_" + "x" * 82,
        "pypi-" + "x" * 120,
        "sk-" + "x" * 48,
        "sk-proj-" + "x" * 80,
        "sk-ant-api03-" + "x" * 80,
        "AIza" + "x" * 35,
        "AKIA" + "X" * 16,
        "https://example.com/?token=" + "pypi-" + "x" * 120,
        ".".join(["100", "64", "1", "2"]),
        ".".join(["100", "127", "255", "254"]),
        ".".join(["node", "tail-example", "ts", "net"]),
    ],
)
def test_release_scan_rejects_credentials_and_private_overlay_addresses(value, tmp_path, capsys):
    mod = _load()
    target = tmp_path / "asset.txt"
    target.write_text(value)
    assert mod.main([str(target)]) == 1
    assert value not in capsys.readouterr().out


@pytest.mark.parametrize(
    "value",
    [
        "sk-summary-model-label",
        "sk-your-key",
        "user@example.com",
        "http://203.0.113.10/v1",
        ".".join(["100", "63", "255", "255"]),
        ".".join(["100", "128", "0", "1"]),
    ],
)
def test_release_scan_keeps_placeholders_and_public_address_boundaries(value):
    assert not _load().violations(value.encode())
