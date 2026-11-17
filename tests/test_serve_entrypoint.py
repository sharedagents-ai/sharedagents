"""Contracts for the installed stack entry point (``python -m sharedagents.serve``).

The foreground supervisor is exercised with stub commands over real loopback
sockets — no sharedagents component is started here. The install-and-run end-to-end
check belongs to release verification, not to this module.
"""

from __future__ import annotations

import builtins
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from sharedagents import serve as serve_module
from sharedagents_gateway import stack

ROOT = Path(__file__).resolve().parents[1]
POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX signal semantics")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _listening(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _child_env(runtime: Path) -> dict:
    env = {**os.environ, "SHAREDAGENTS_HOME": str(runtime)}
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_usage_documents_every_action(capsys):
    with pytest.raises(SystemExit) as exc:
        serve_module.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for action in serve_module.ACTIONS:
        assert action in out


def test_unknown_action_is_a_clean_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        serve_module.main(["bogus"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_module_alias_is_the_same_entry_point():
    from sharedagents import __main__ as alias

    assert alias.main is serve_module.main


def test_help_creates_nothing(tmp_path):
    runtime = tmp_path / "must-not-exist"
    cp = subprocess.run(
        [sys.executable, "-m", "sharedagents.serve", "--help"],
        env=_child_env(runtime),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert cp.returncode == 0, cp.stderr
    for action in serve_module.ACTIONS:
        assert action in cp.stdout
    assert not runtime.exists()


def test_status_creates_nothing_and_reports_every_component(tmp_path):
    runtime = tmp_path / "must-not-exist"
    cp = subprocess.run(
        [sys.executable, "-m", "sharedagents.serve", "status"],
        env=_child_env(runtime),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert not runtime.exists()
    for name, port, _ in stack.ENDPOINTS:
        assert any(
            line.startswith(f"PASS {name} {port} ") or line.startswith(f"FAIL {name} {port} ")
            for line in cp.stdout.splitlines()
        ), cp.stdout


def test_components_cover_the_three_endpoints(tmp_path):
    members = stack.components(tmp_path, Path(sys.executable))
    assert [member.name for member in members] == ["orchestrator", "gateway", "web"]
    assert [member.port for member in members] == [8099, 8100, 8110]


def test_web_prerequisites_names_the_platform_server(monkeypatch):
    """POSIX needs gunicorn, Windows needs waitress — never the other one."""
    real_import = builtins.__import__

    def deny(name, *args, **kwargs):
        if name in {"flask", "requests", "gunicorn", "waitress"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny)
    monkeypatch.setattr(stack.oscompat, "IS_WINDOWS", False)
    assert stack.web_prerequisites() == ["flask", "requests", "gunicorn"]
    monkeypatch.setattr(stack.oscompat, "IS_WINDOWS", True)
    assert stack.web_prerequisites() == ["flask", "requests", "waitress"]


def test_missing_web_extra_fails_before_starting_anything(monkeypatch, tmp_path):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.setattr(stack, "web_prerequisites", lambda: ["flask"])
    started: list[str] = []
    monkeypatch.setattr(stack, "_posix_detached_start", lambda **_: started.append("posix") or 0)
    monkeypatch.setattr(stack, "_windows_lifecycle", lambda *_, **__: started.append("nt") or 0)
    with pytest.raises(ImportError, match=r"sharedagents\[web\]"):
        stack.run("serve", detach=True)
    assert started == []


def test_no_web_skips_the_extra_check(monkeypatch, tmp_path):
    monkeypatch.setenv("SHAREDAGENTS_HOME", str(tmp_path))
    monkeypatch.setattr(stack, "web_prerequisites", lambda: pytest.fail("must not be checked"))
    monkeypatch.setattr(stack, "_posix_detached_start", lambda **_: 0)
    monkeypatch.setattr(stack, "_windows_lifecycle", lambda *_, **__: 0)
    assert stack.run("serve", detach=True, no_web=True) == 0


@POSIX_ONLY
def test_foreground_starts_three_and_sigint_stops_them_all(tmp_path):
    """The foreground supervisor must start every stub, then reap all on SIGINT."""
    ports = [_free_port() for _ in range(3)]
    driver = tmp_path / "driver.py"
    driver.write_text(
        textwrap.dedent(
            f"""
            import sys
            from pathlib import Path

            sys.path.insert(0, {str(ROOT / "src")!r})
            from sharedagents_gateway import stack

            def cmd(port):
                return [sys.executable, "-c",
                        "import socket, time;"
                        "s = socket.socket();"
                        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
                        "s.bind(('127.0.0.1', %d)); s.listen(); time.sleep(180)" % port]

            members = [
                stack.Component(f"stub{{i}}", port, "/",
                                Path({str(tmp_path)!r}) / f"stub{{i}}.pid",
                                cmd(port), Path({str(tmp_path)!r}),
                                Path({str(tmp_path)!r}) / f"stub{{i}}.log")
                for i, port in enumerate({ports!r})
            ]
            raise SystemExit(stack.foreground(members))
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.Popen([sys.executable, str(driver)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not all(_listening(port) for port in ports):
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        assert all(_listening(port) for port in ports), proc.stdout.read() if proc.stdout else ""

        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=60) == 0
        out = proc.stdout.read() if proc.stdout else ""
        assert out.count("START stub") == 3, out
    finally:
        if proc.poll() is None:
            proc.kill()

        deadline = time.monotonic() + 30
    while time.monotonic() < deadline and any(_listening(port) for port in ports):
        time.sleep(0.2)
    assert not any(_listening(port) for port in ports), "a stub survived SIGINT"


def _zombie_state(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            return handle.read().decode("utf-8", "replace").rsplit(") ", 1)[1][:1]
    except OSError:
        return "gone"


@POSIX_ONLY
def test_process_alive_is_false_for_an_unreaped_zombie():
    """A zombie owns its PID slot but has already exited.

    ``os.kill(pid, 0)`` succeeds for zombies, so without a /proc state check a
    stop() polls a dead daemon for its entire timeout and then reports a false
    failure — the SIGKILL it escalates to cannot even be delivered.
    """
    from sharedagents_core import oscompat

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _zombie_state(proc.pid) != "Z":
            time.sleep(0.01)
        if _zombie_state(proc.pid) != "Z":
            pytest.skip("child was reaped before the zombie state was observable")
        assert oscompat.process_alive(proc.pid) is False
    finally:
        proc.wait()
