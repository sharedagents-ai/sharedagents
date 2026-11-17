"""Stack supervisor behind ``python -m sharedagents.serve``.

Runs the three components together — orchestrator (8099), gateway (8100) and
WebUI (8110) — in the foreground, detached, or as a stop/status probe. There is
no service manager and no restart loop: a component that dies is reported by
name, never respawned.

Windows termination is driven by creation-identity-verified handles, so a
recycled PID can never be signalled, and nothing is ever killed by image name,
process group, or port ownership. POSIX delegates detached start/stop to the
existing ``gateway_cli`` / ``web_cli`` daemonizers rather than reimplementing
them. Importing this module is inert on every platform.
"""

from __future__ import annotations

import ctypes
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from sharedagents_core import oscompat

# gateway.py:78-87,1532-1536; web_cli.py:47-48; routes/auth.py:79-102.
HOST = "127.0.0.1"
ENDPOINTS = (
    ("orchestrator", 8099, "/health"),
    ("gateway", 8100, "/health"),
    ("web", 8110, "/login"),
)
READY_SECONDS = 20.0  # gateway_cli.py:400,423; web_cli.py:293.
WEB_EXTRA_HINT = 'pip install "sharedagents[web]"'
# Windows OpenSSH runs each session inside a job object that terminates every
# descendant when the session ends, so a server that stays in that job dies the
# instant the SSH connection closes (measured on a Windows host). Breaking away from the
# job is what makes these three processes outlive the session. The constant
# exists only on Windows' subprocess module; tests monkeypatch the names below.
BREAKAWAY_FROM_JOB = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)
ERROR_ACCESS_DENIED = 5  # CreateProcess when the job forbids breakaway.


def tcp_listening(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=1.0):
            return True
    except OSError:
        return False


def probe(port: int, path: str) -> tuple[bool, int]:
    tcp = tcp_listening(port)
    connection = http.client.HTTPConnection(HOST, port, timeout=2.0)
    try:
        # Direct loopback HTTP: no environment proxy, cookies, or redirects.
        connection.request("GET", path)
        code = connection.getresponse().status
    except (OSError, http.client.HTTPException):
        code = 0
    finally:
        connection.close()
    return tcp, code


def status(*, include_web: bool = True) -> int:
    """One machine-readable PASS/FAIL line per component; non-zero on any FAIL."""
    failed = False
    for name, port, path in ENDPOINTS:
        if name == "web" and not include_web:
            continue
        tcp, code = probe(port, path)
        ok = tcp and 200 <= code < 300
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'} {name} {port} tcp={'PASS' if tcp else 'FAIL'} http={code:03d}", flush=True)
    return int(failed)


class WindowsProcess:
    """Keep one query/synchronize handle through identity check and termination.

    A saved creation FILETIME rejects recycled PIDs. Force uses this same handle,
    including for each separately verified descendant; no numeric/name kill.
    """

    def __init__(self, pid: int, *, terminate: bool = False):
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("PID must be a positive integer")
        self.pid = pid
        api = self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        api.OpenProcess.restype = wintypes.HANDLE
        api.CloseHandle.argtypes = (wintypes.HANDLE,)
        api.CloseHandle.restype = wintypes.BOOL
        api.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
        api.GetProcessTimes.restype = wintypes.BOOL
        api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        api.WaitForSingleObject.restype = wintypes.DWORD
        api.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        api.TerminateProcess.restype = wintypes.BOOL
        # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION [+ PROCESS_TERMINATE].
        self.handle = api.OpenProcess(0x00101000 | int(terminate), False, pid)
        if not self.handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists.
                raise ProcessLookupError(pid)
            raise ctypes.WinError(error)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.api.CloseHandle(self.handle)

    def identity(self) -> dict:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not self.api.GetProcessTimes(self.handle, *(ctypes.byref(t) for t in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"pid": self.pid, "created": (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime}

    def exited(self, milliseconds: int = 0) -> bool:
        result = self.api.WaitForSingleObject(self.handle, milliseconds)
        if result not in (0, 258):  # WAIT_OBJECT_0 / WAIT_TIMEOUT.
            raise ctypes.WinError(ctypes.get_last_error())
        return result == 0

    def force(self) -> None:
        if not self.api.TerminateProcess(self.handle, 1) and not self.exited():
            raise ctypes.WinError(ctypes.get_last_error())


@dataclass
class Component:
    name: str
    port: int
    path: str
    pidfile: Path
    argv: list[str]
    cwd: Path
    log: Path


def stack_argv(name: str, python: Path) -> list[str]:
    # Same order/forms as gateway_cli._stack_argv/_resolve_gateway_exec, scoped
    # to this venv so a global install or this repo's sharedagents.bat cannot win PATH.
    exe = python.with_name("sharedagents.exe")
    if exe.is_file():
        return [str(exe), name, "serve"]
    return [str(python), "-m", "sharedagents_tui", name, "serve"]


def web_argv(python: Path) -> list[str]:
    """Shared console-script/module convention; Windows wrappers spawn children.

    Windows has an exe shim -> venv redirector -> real Python listener chain.
    The spawned PID is therefore an ancestry anchor, NOT the server PID.
    """
    from sharedagents_gateway.web_cli import web_argv as resolve

    return resolve(python)


def components(home: Path, python: Path) -> list[Component]:
    from sharedagents_gateway import gateway_cli as gateway, web_cli as web

    orch, north, frontend = ENDPOINTS
    return [
        Component(
            *orch,
            gateway._orchestrator_pidfile(),
            stack_argv(orch[0], python),
            home,
            gateway._gateway_logfile().with_name("orchestrator.log"),
        ),
        Component(*north, gateway._gateway_pidfile(), stack_argv(north[0], python), home, gateway._gateway_logfile()),
        Component(*frontend, web._web_pidfile(), web_argv(python), web._web_dir(), web._web_logfile()),
    ]


def validate_config() -> None:
    from sharedagents_core.kernel.config import global_config_path
    from sharedagents_core.kernel.layout import sharedagents_root
    from sharedagents_gateway import gateway_cli as gateway

    # The existing config loader silently ignores malformed JSON. Fail loudly
    # here so launching cannot silently discard the operator's intended binds.
    config = global_config_path()
    if config.exists() and not isinstance(json.loads(config.read_text(encoding="utf-8")), dict):
        raise ValueError(f"expected a JSON object in {config}")
    if sharedagents_root().resolve() != gateway._sharedagents_home().resolve():
        raise ValueError("SHAREDAGENTS_AGENTS_HOME conflicts with SHAREDAGENTS_HOME")
    for name, cfg, expected in (
        ("orchestrator", gateway._orchestrator_config(), 8099),
        ("gateway", gateway._gateway_config(), 8100),
    ):
        if (cfg["host"], cfg["port"]) != (HOST, expected):
            raise ValueError(
                f"{config}: set {name}.host={HOST}, {name}.port={expected}; launcher does not rewrite config.json"
            )


def read_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    if not raw:  # Empty during a write, or cleared after a confirmed exit.
        return None
    pid = int(raw)
    if pid <= 0:
        raise ValueError(f"invalid PID in {path}")
    return pid


def save_records(path: Path, records: dict) -> None:
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def spawn(component: Component, env: dict) -> subprocess.Popen:
    # Win32 equivalent of start /min, independently for all three servers.
    # Unlike start /b or CREATE_NEW_PROCESS_GROUP alone, this gives each its
    # own console. All std handles are files/NUL, never the SSH channel.
    # CREATE_BREAKAWAY_FROM_JOB is additionally required over SSH: sshd puts the
    # whole session in a job object with kill-on-close, so an ordinary child is
    # terminated when the connection ends (measured on a Windows host: two otherwise
    # identical children, only the breakaway one survived). A job that forbids
    # breakaway makes CreateProcess fail with ERROR_ACCESS_DENIED instead, so
    # that case falls back to the previous flags rather than failing outright.
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 2  # SW_SHOWMINIMIZED.
    component.log.parent.mkdir(parents=True, exist_ok=True)
    if component.log.exists():
        component.log.replace(component.log.with_name(component.log.name + ".1"))
    env = {
        **env,
        "PYTHONUNBUFFERED": "1",
        "SHAREDAGENTS_COMPONENT_ROLE": component.name,
        "SHAREDAGENTS_COMPONENT_LOG": str(component.log.resolve()),
    }
    with component.log.open("ab", buffering=0) as log:

        def popen(creationflags: int):
            return subprocess.Popen(
                component.argv,
                cwd=str(component.cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                creationflags=creationflags,
                startupinfo=startup,
                close_fds=True,
            )

        try:
            return popen(subprocess.CREATE_NEW_CONSOLE | BREAKAWAY_FROM_JOB)
        except OSError as error:
            if getattr(error, "winerror", None) != ERROR_ACCESS_DENIED:
                raise
            print(
                f"WARN {component.name}: this job forbids breakaway; the server will not survive its launching session",
                flush=True,
            )
            return popen(subprocess.CREATE_NEW_CONSOLE)


def extend_tree(stack: ExitStack, processes: dict, *, terminate: bool) -> None:
    """Pin and validate descendants before using them as part of this component.

    Hold every handle until the caller finishes. Parent birth times reject old
    orphans whose parent PID was reused. A second snapshot after opening handles
    rejects a child PID that was recycled between enumeration and OpenProcess.
    Nothing is signalled here, including when one candidate cannot be verified.
    """
    parents = oscompat.windows_process_parents()
    pending = list(processes)
    links = {}
    while pending:
        parent_pid = pending.pop(0)
        parent_created = processes[parent_pid].identity()["created"]
        for pid, ppid in parents.items():
            if ppid != parent_pid or pid in processes or pid <= 0:
                continue
            try:
                child = stack.enter_context(WindowsProcess(pid, terminate=terminate))
            except ProcessLookupError:
                continue
            if child.identity()["created"] < parent_created:
                continue  # Orphan of an earlier process that used this parent PID.
            processes[pid] = child
            links[pid] = ppid
            pending.append(pid)
    current = oscompat.windows_process_parents() if links else parents
    for pid, ppid in links.items():
        if not processes[pid].exited() and current.get(pid) != ppid:
            raise RuntimeError(f"PID {pid} ancestry changed; no unverified process signalled")


def stop_tree(record: dict) -> list[int]:
    """Stop a recorded component, including wrappers, using only held handles.

    Legacy records contain just one identity (web's shim or a backend listener).
    New records retain the verified startup chain as well as the listener PID.
    This lets cleanup reach a listener even after a wrapper has already exited.
    """
    if not isinstance(record, dict) or not all(key in record for key in ("pid", "created")):
        raise ValueError("invalid startup identity record; no process signalled")
    primary = {key: record[key] for key in ("pid", "created")}
    identities = [*record.get("chain", []), primary]
    if any(
        not isinstance(i, dict)
        or set(i) != {"pid", "created"}
        or not all(isinstance(i[k], int) and i[k] > 0 for k in ("pid", "created"))
        for i in identities
    ):
        raise ValueError("invalid startup chain identity; no process signalled")
    with ExitStack() as stack:
        processes = {}
        for identity in identities:
            pid = identity["pid"]
            if pid in processes:
                if processes[pid].identity() != identity:
                    raise RuntimeError(f"PID {pid} lacks matching startup identity; no process signalled")
                continue
            try:
                process = stack.enter_context(WindowsProcess(pid, terminate=True))
            except ProcessLookupError:
                continue
            if process.identity() != identity:
                raise RuntimeError(f"PID {pid} lacks matching startup identity; no process signalled")
            processes[pid] = process
        # Validate the entire tree before the first termination. Descendants are
        # appended after their ancestors, so reversal stops the listener before
        # its console-owning shim. Repeat to catch children born during cleanup.
        deadline = time.monotonic() + 5
        while True:
            extend_tree(stack, processes, terminate=True)
            count = len(processes)
            for process in reversed(list(processes.values())):
                if not process.exited():
                    process.force()
                if not process.exited(1000):
                    raise RuntimeError(f"PID {process.pid} still alive after force")
            extend_tree(stack, processes, terminate=True)
            if len(processes) == count:
                return list(processes)
            if time.monotonic() >= deadline:
                raise RuntimeError("component keeps spawning descendants; cleanup incomplete")


def start_one(component: Component, records: dict, record_path: Path) -> bool:
    if tcp_listening(component.port):
        print(f"SKIP {component.name} {component.port} already-listening", flush=True)
        return True  # Never adopt an unknown listener or overwrite its pidfile.
    old_pid = read_pid(component.pidfile)
    record = records.get(component.name) or {}
    # A failed web startup may have saved its root before a pidfile existed.
    candidates = [old_pid, record.get("pid"), *(i["pid"] for i in record.get("chain", []))]
    for old_pid in dict.fromkeys(p for p in candidates if p is not None):
        try:
            with WindowsProcess(old_pid) as process:
                if not process.exited():
                    raise RuntimeError(f"live pidfile PID {old_pid} without listener; refusing duplicate")
        except ProcessLookupError:
            pass
    component.pidfile.parent.mkdir(parents=True, exist_ok=True)
    component.pidfile.write_text("", encoding="ascii")
    env = os.environ.copy()
    if component.name == "web":
        from sharedagents_gateway.web_cli import _web_env

        env = _web_env()  # Existing secret generation/persistence, auth unchanged.
    proc = spawn(component, env)
    with ExitStack() as stack:
        root = stack.enter_context(WindowsProcess(proc.pid))
        tree = {proc.pid: root}
        # Preserve an anchor even if startup fails before the listener binds.
        records[component.name] = {**root.identity(), "chain": [root.identity()]}
        save_records(record_path, records)
        deadline = time.monotonic() + READY_SECONDS
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"launcher PID {proc.pid} exited {proc.returncode}; log={component.log}")
            tcp, code = probe(component.port, component.path)
            pid = oscompat.port_owner_pid(HOST, component.port) if tcp else None
            if pid is not None:
                extend_tree(stack, tree, terminate=False)
                if pid not in tree or tree[pid].exited():
                    raise RuntimeError(f"listener PID {pid} is not in the launched component; refusing adoption")
                if oscompat.port_owner_pid(HOST, component.port) != pid:
                    raise RuntimeError("listener changed during startup; refusing adoption")
                records[component.name] = {
                    **tree[pid].identity(),
                    "chain": [p.identity() for p in tree.values()],
                }
                save_records(record_path, records)
                component.pidfile.write_text(str(pid), encoding="ascii")
                if 200 <= code < 300:
                    print(
                        f"START {component.name} {component.port} pid={pid} http={code} log={component.log}", flush=True
                    )
                    return True
            if time.monotonic() >= deadline:
                raise RuntimeError(f"readiness timeout pid={pid} http={code:03d}; log={component.log}")
            time.sleep(0.2)


def stop_one(component: Component, records: dict, record_path: Path) -> bool:
    pid = read_pid(component.pidfile)
    record = records.get(component.name)
    if record is not None and not isinstance(record, dict):
        raise ValueError("invalid startup identity record; no process signalled")
    if pid is None and not record:
        if tcp_listening(component.port):
            raise RuntimeError("listener has no pidfile; refusing to kill its port owner")
        print(f"STOP {component.name} already-stopped", flush=True)
        return True
    if not record or (pid is not None and record.get("pid") != pid):
        try:
            with WindowsProcess(pid) as process:
                if not process.exited():
                    raise RuntimeError(f"PID {pid} lacks matching startup identity; no process signalled")
        except ProcessLookupError:
            pass
        stopped = []
    else:
        stopped = stop_tree(record)
    result_text = f"STOP {component.name} pid={pid} tree={','.join(map(str, stopped)) or 'none'}"
    # The CLI may have removed its own pidfile. Never clear a successor's PID.
    current = read_pid(component.pidfile)
    if current not in (None, pid):
        raise RuntimeError("pidfile changed during stop; successor left untouched")
    deadline = time.monotonic() + 5
    while tcp_listening(component.port):
        if time.monotonic() >= deadline:
            raise RuntimeError("PID exited but port still listening; remaining process left untouched")
        time.sleep(0.2)
    current = read_pid(component.pidfile)
    if current not in (None, pid):
        raise RuntimeError("pidfile changed during stop; successor left untouched")
    if current == pid and pid is not None:
        component.pidfile.write_text("", encoding="ascii")
    records.pop(component.name, None)
    save_records(record_path, records)
    print(result_text, flush=True)
    return True


def web_prerequisites() -> list[str]:
    """Web dependencies missing from THIS interpreter.

    The WebUI lives in the ``[web]`` extra, which is not part of the lean core.
    Checking up front yields one actionable message instead of starting two
    thirds of a stack and failing later.
    """
    missing = []
    server = "waitress" if oscompat.IS_WINDOWS else "gunicorn"
    for module in ("flask", "requests", server):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return missing


def _missing_web_error(missing: list[str]) -> str:
    # ASCII only: this text also lands in a redirected Windows log file whose
    # encoding is the legacy code page, not UTF-8.
    return f"missing web dependencies: {', '.join(missing)}. Install them with {WEB_EXTRA_HINT}"


def _windows_lifecycle(action: str, *, no_web: bool = False) -> int:
    """Detached start/stop on Windows, using creation-identity-verified handles."""
    from sharedagents_gateway.gateway_cli import _sharedagents_home

    home = _sharedagents_home()
    if not home.is_absolute():
        raise ValueError("SHAREDAGENTS_HOME must be an absolute path")
    python = Path(sys.executable)
    os.environ["SHAREDAGENTS_HOME"] = str(home)
    os.environ["PATH"] = str(python.parent) + os.pathsep + os.environ.get("PATH", "")
    if action == "start":
        validate_config()
        # Gateway/orchestrator binds are config-only; WebUI exposes these envs.
        os.environ["SHAREDAGENTS_WEB_HOST"] = HOST
        os.environ["SHAREDAGENTS_WEB_PORT"] = "8110"
        os.environ["SHAREDAGENTS_GATEWAY_URL"] = "http://127.0.0.1:8100"
        # Fail before starting any component if the required web extra is absent.
        missing = [] if no_web else web_prerequisites()
        if missing:
            raise ImportError(_missing_web_error(missing))
    home.mkdir(parents=True, exist_ok=True)
    # Share the gateway CLI's single-flight lock; also serialize our stop/web.
    with (home / "gateway.start.lock").open("a") as lock, oscompat.file_lock(lock, nonblocking=True):
        record_path = home / "windows-launcher.json"
        records = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
        if not isinstance(records, dict):
            raise ValueError(f"invalid startup identity file: {record_path}")
        members = components(home, python)
        if no_web:
            members = [c for c in members if c.name != "web"]
        if action == "stop":
            members.reverse()  # web, north, south: disconnect clients first.
        failed = False
        operation = start_one if action == "start" else stop_one
        for component in members:
            try:
                operation(component, records, record_path)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                print(f"FAIL {component.name} {action}: {error}", flush=True)
                failed = True
        return int(failed)


def _posix_detached_start(*, no_web: bool = False) -> int:
    """Detached start on POSIX: reuse the existing gateway/web daemonizers."""
    from sharedagents_gateway import gateway_cli, web_cli

    result = gateway_cli.start_gateway_background()
    failed = not result.get("ok")
    if result.get("ok"):
        verb = "SKIP" if result.get("already_running") else "START"
        print(
            f"{verb} orchestrator 8099 pid={result.get('pid_orchestrator')} url={result.get('orchestrator_url')}",
            flush=True,
        )
        print(f"{verb} gateway 8100 pid={result.get('pid')} url={result.get('url')}", flush=True)
    else:
        print(f"FAIL orchestrator/gateway start: {result.get('error')}", flush=True)
    if not no_web:
        missing = web_prerequisites()
        if missing:
            print(f"FAIL web start: {_missing_web_error(missing)}", flush=True)
            return 1
        web = web_cli.start_web_background()
        if web.get("ok"):
            verb = "SKIP" if web.get("already_running") else "START"
            print(f"{verb} web 8110 pid={web.get('pid')} url={web.get('url')}", flush=True)
        else:
            print(f"FAIL web start: {web.get('error')}", flush=True)
            failed = True
    return int(failed)


def _posix_detached_stop(*, no_web: bool = False) -> int:
    """Detached stop on POSIX, reusing the existing stop helpers.

    ``cmd_stop`` owns the north-then-south ordering and the port-orphan reap, so
    it is called rather than reimplemented; it ignores its argparse argument.
    """
    from sharedagents_gateway import gateway_cli, web_cli

    failed = False
    if not no_web:
        web = web_cli.stop_web()
        print(f"STOP web pid={web.get('pid')} was_running={web.get('was_running')}", flush=True)
        failed |= not web.get("ok", False)
    rc = gateway_cli.cmd_stop(None)
    print(f"STOP orchestrator/gateway rc={rc}", flush=True)
    return int(failed or bool(rc))


def read_records(path: Path) -> dict:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"invalid startup identity file: {path}")
    return loaded


def _chain_identities(root_pid: int, listener_pid: int) -> list[dict]:
    """Identity records from the spawned root down to the listener (Windows).

    A console script is a wrapper: ``sharedagents.exe`` starts a venv redirector which
    in turn starts the real server, so the listener is a DESCENDANT of the PID we
    spawned and killing the spawned PID alone would leave the listener running.
    Walking the recorded parent links gives the whole path without guessing.
    """
    parents = oscompat.windows_process_parents()
    path, cursor, seen = [], listener_pid, set()
    while cursor and cursor not in seen and len(path) < 10:
        seen.add(cursor)
        path.append(cursor)
        if cursor == root_pid:
            break
        cursor = parents.get(cursor)
    if not path or path[-1] != root_pid:
        path = [listener_pid]  # Unexpected ancestry: record only what we verified.
    identities = []
    for pid in reversed(path):
        try:
            with WindowsProcess(pid) as process:
                identities.append(process.identity())
        except (OSError, ProcessLookupError):
            continue
    return identities


def _register_foreground(component: Component, proc: subprocess.Popen, records: dict) -> None:
    """Publish the pidfile and identity record for a foreground component.

    Without this a foreground stack could not be stopped programmatically: the
    stop path refuses any PID it cannot verify against a recorded identity, so it
    would report FAIL, leave the ports held, and make the next `serve` refuse to
    start. The WebUI is the sharpest case -- it writes no pidfile of its own.
    """
    listener = proc.pid
    if oscompat.IS_WINDOWS:
        listener = oscompat.port_owner_pid(HOST, component.port) or proc.pid
    try:
        component.pidfile.parent.mkdir(parents=True, exist_ok=True)
        component.pidfile.write_text(str(listener), encoding="ascii")
    except OSError:
        pass
    if not oscompat.IS_WINDOWS:
        return
    identities = _chain_identities(proc.pid, listener)
    if not identities:
        return
    record_path = component.pidfile.parent / "windows-launcher.json"
    try:
        current = read_records(record_path)
    except (OSError, ValueError):
        current = {}
    current[component.name] = {**identities[-1], "chain": identities}
    records[component.name] = current[component.name]
    try:
        save_records(record_path, current)
    except OSError:
        pass


def _terminate(
    procs: list[tuple[Component, subprocess.Popen]], records: dict | None = None, *, grace: float = 10.0
) -> None:
    """Ask every live child to stop, then force whatever outlives the grace.

    On Windows a recorded identity chain is used so the wrapper's descendants go
    too; a bare PID kill would leave the real listener behind.
    """
    for component, proc in procs:
        if proc.poll() is not None:
            continue
        record = (records or {}).get(component.name)
        if record and oscompat.IS_WINDOWS:
            try:
                stop_tree(record)
                continue
            except (OSError, ValueError, RuntimeError):
                pass  # Fall through to the direct kill below.
        try:
            oscompat.kill_process(proc.pid, force=False)
        except (OSError, ProcessLookupError):
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if all(proc.poll() is not None for _, proc in procs):
            return
        time.sleep(0.2)
    for component, proc in procs:
        if proc.poll() is None:
            print(f"WARN {component.name} did not exit within {grace:.0f}s; forcing", flush=True)
            try:
                oscompat.kill_process(proc.pid, force=True)
            except (OSError, ProcessLookupError):
                pass


def foreground(members: list[Component]) -> int:
    """Run every component as a child of this process, streaming its output.

    Children stay in this process group, so a terminal Ctrl+C (or a SIGTERM to
    the supervisor) reaches them directly; this function then reaps them and
    forces any straggler. Output is inherited rather than redirected, so the
    console is the log. There is no restart loop: a component that dies is
    reported by name and the survivors keep running.
    """
    procs: list[tuple[Component, subprocess.Popen]] = []
    records: dict = {}
    state = {"stop": False}

    def _on_signal(_signum, _frame):
        state["stop"] = True

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[signum] = signal.signal(signum, _on_signal)
        except (OSError, ValueError):  # not the main thread / signal unsupported
            pass

    try:
        for component in members:
            if tcp_listening(component.port):
                print(
                    f"FAIL {component.name} {component.port} already listening; run "
                    f"`sharedagents-serve status` or `sharedagents-serve stop` first",
                    file=sys.stderr,
                    flush=True,
                )
                _terminate(procs, records)
                return 1
            env = os.environ.copy()
            if component.name == "web":
                from sharedagents_gateway.web_cli import _web_env

                env = _web_env()  # Existing secret generation/persistence, auth unchanged.
            try:
                proc = subprocess.Popen(component.argv, cwd=str(component.cwd), env=env)
            except OSError as error:
                print(f"FAIL {component.name} spawn: {error}", file=sys.stderr, flush=True)
                _terminate(procs, records)
                return 1
            procs.append((component, proc))
            deadline = time.monotonic() + READY_SECONDS
            while not tcp_listening(component.port) and time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
            tcp, code = probe(component.port, component.path)
            print(
                f"{'START' if tcp else 'FAIL'} {component.name} {component.port} pid={proc.pid} http={code:03d}",
                flush=True,
            )
            if not tcp:
                print(
                    f"FAIL {component.name} did not bind {HOST}:{component.port}; see its output above",
                    file=sys.stderr,
                    flush=True,
                )
                _terminate(procs, records)
                return 1
            _register_foreground(component, proc, records)

        print(
            f"WebUI http://{HOST}:8110  (Ctrl+C stops the stack; --detach keeps it running after this shell exits)",
            flush=True,
        )

        while not state["stop"]:
            if all(proc.poll() is not None for _, proc in procs):
                break
            time.sleep(0.2)

        if state["stop"]:
            print("stopping...", flush=True)
            _terminate(procs, records)
        for component, proc in procs:
            if proc.poll() is not None:
                print(f"EXIT {component.name} rc={proc.returncode}", flush=True)
        if state["stop"]:
            return 0
        return max((proc.returncode or 0 for _, proc in procs), default=0)
    finally:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass


def run(action: str, *, detach: bool = False, no_web: bool = False) -> int:
    """Entry point behind ``python -m sharedagents.serve`` and ``sharedagents-serve``."""
    from sharedagents_gateway.gateway_cli import _sharedagents_home

    home = _sharedagents_home()
    if not home.is_absolute():
        raise ValueError("SHAREDAGENTS_HOME must be an absolute path")
    os.environ["SHAREDAGENTS_HOME"] = str(home)
    if action == "status":
        return status(include_web=not no_web)
    if action == "stop":
        if oscompat.IS_WINDOWS:
            return _windows_lifecycle("stop", no_web=no_web)
        return _posix_detached_stop(no_web=no_web)
    if action != "serve":
        raise ValueError(f"unknown action {action!r}")

    missing = [] if no_web else web_prerequisites()
    if missing:
        raise ImportError(_missing_web_error(missing))
    if detach:
        if oscompat.IS_WINDOWS:
            return _windows_lifecycle("start", no_web=no_web)
        return _posix_detached_start(no_web=no_web)

    validate_config()
    home.mkdir(parents=True, exist_ok=True)
    members = components(home, Path(sys.executable))
    if no_web:
        members = [c for c in members if c.name != "web"]
    return foreground(members)
