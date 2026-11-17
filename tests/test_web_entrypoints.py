"""The generated unit and background launcher share the package entry command."""

from types import SimpleNamespace

import pytest

from sharedagents_core import oscompat
from sharedagents_gateway import web_cli


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("installed", [False, True])
def test_background_and_unit_use_same_entry(tmp_path, monkeypatch, windows, installed):
    monkeypatch.setattr(oscompat, "IS_WINDOWS", windows)
    python = tmp_path / "venv with spaces" / ("python.exe" if windows else "python")
    python.parent.mkdir()
    exe = python.with_name("sharedagents-web.exe" if windows else "sharedagents-web")
    if installed:
        exe.touch()
    expected = [str(exe)] if installed else [str(python), "-m", "sharedagents_web"]
    assert web_cli.web_argv(python) == expected
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.delenv("SHAREDAGENTS_AGENTS_HOME", raising=False)
    monkeypatch.setattr(web_cli, "_web_python", lambda: python)
    monkeypatch.setattr(web_cli, "_running", lambda: (False, None))
    monkeypatch.setattr(web_cli, "_web_env", lambda **kw: {})
    monkeypatch.setattr(web_cli, "_wait_for_listener", lambda *a, **kw: True)
    seen = []

    def spawn(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(pid=77)

    monkeypatch.setattr(web_cli.subprocess, "Popen", spawn)
    assert web_cli.start_web_background()["ok"]
    assert seen == [expected]
    unit = web_cli.render_web_unit()
    assert f"ExecStart={web_cli._systemd_argv(expected)}\n" in unit
    assert "server.py" not in unit


def test_systemd_command_escapes_literal_paths():
    assert web_cli._systemd_argv(["/venv with space/sharedagents-web", '%a$"b\\c']) == (
        '"/venv with space/sharedagents-web" "%%a$$\\"b\\\\c"'
    )
