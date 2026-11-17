"""Stdlib platform primitives for native Windows and POSIX boot paths.

POSIX locking is imported only when used, so importing a boot module never
requires fcntl. Windows locks cover byte zero (even on an empty file); unlike
flock they are mandatory byte-range locks. Callers must serialize access to
each descriptor within a process and flush buffered writes before unlocking.
"""

from __future__ import annotations

import ctypes
import errno
import ipaddress
import ntpath
import os
import shutil
import signal
import socket
import subprocess
import time
from contextlib import AbstractContextManager
from functools import lru_cache
from pathlib import Path
from typing import IO

if os.name == "nt":
    import msvcrt


IS_WINDOWS: bool = os.name == "nt"
# Python already creates non-inheritable descriptors when O_CLOEXEC is absent.
O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
O_BINARY = getattr(os, "O_BINARY", 0)
# A token for callers that accept a signal; Windows termination uses handles.
FORCE_KILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _fd(fp: int | IO) -> int:
    return fp if isinstance(fp, int) else fp.fileno()


def _windows_lock(fd: int, mode: int) -> None:
    position = os.lseek(fd, 0, os.SEEK_CUR)
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, mode, 1)
    finally:
        os.lseek(fd, position, os.SEEK_SET)


class _HeldLock(AbstractContextManager):
    """An already-acquired lock; leaving a with block releases it."""

    def __init__(self, fp: int | IO) -> None:
        self.fp = fp

    def __enter__(self):
        return self.fp

    def __exit__(self, *exc_info) -> None:
        file_unlock(self.fp)


def file_lock(fp: int | IO, *, nonblocking: bool = False) -> _HeldLock:
    """Acquire an exclusive lock immediately; optionally use as a context.

    POSIX uses flock. Windows locks one byte at offset zero, preserving the
    descriptor position and retrying beyond LK_LOCK's built-in ten attempts.
    Nonblocking contention raises BlockingIOError on either platform.
    """
    fd = _fd(fp)
    if not IS_WINDOWS:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
    else:
        mode = msvcrt.LK_NBLCK if nonblocking else msvcrt.LK_LOCK
        while True:
            try:
                _windows_lock(fd, mode)
                break
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if nonblocking:
                    raise BlockingIOError(errno.EAGAIN, "file lock held") from exc
                time.sleep(0.05)
    return _HeldLock(fp)


def try_file_lock(fp: int | IO) -> bool:
    """Acquire without waiting; return False only for lock contention."""
    try:
        file_lock(fp, nonblocking=True)
    except BlockingIOError:
        return False
    return True


def file_unlock(fp: int | IO) -> None:
    """Release flock on POSIX or byte zero on Windows, preserving position."""
    if IS_WINDOWS:
        _windows_lock(_fd(fp), msvcrt.LK_UNLCK)
    else:
        import fcntl

        fcntl.flock(_fd(fp), fcntl.LOCK_UN)


def venv_python(root: str | os.PathLike) -> Path:
    """Return root/Scripts/python.exe on Windows, root/bin/python on POSIX."""
    return Path(root) / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def temp_roots() -> tuple[str, ...]:
    """Existing real temp directories: /tmp, /var/tmp or Windows env paths."""
    candidates = ["/tmp", "/var/tmp"]
    if IS_WINDOWS:
        candidates = [os.environ.get("TEMP", ""), os.environ.get("TMP", "")]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(os.path.join(local, "Temp"))
    return tuple(dict.fromkeys(os.path.realpath(path) for path in candidates if path and os.path.isdir(path)))


def find_cmd_exe() -> str | None:
    """Resolve cmd.exe on Windows by override, COMSPEC, system dirs, then PATH.

    Return None on POSIX. A missing explicit override or an exhausted Windows
    search raises FileNotFoundError with the attempted paths; no shell fallback.
    """
    if not IS_WINDOWS:
        return None
    override = os.environ.get("SHAREDAGENTS_EXEC_SHELL")
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"SHAREDAGENTS_EXEC_SHELL does not exist: {override}")
        return os.path.abspath(override)
    tried = ["SHAREDAGENTS_EXEC_SHELL (unset)"]
    comspec = os.environ.get("COMSPEC", "")
    tried.append(comspec or "COMSPEC (unset)")
    if ntpath.basename(comspec).lower() == "cmd.exe" and os.path.isfile(comspec):
        return os.path.abspath(comspec)
    root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows"  # noqa: SIM112 - Windows spelling
    for directory in ("System32", "Sysnative"):
        candidate = os.path.join(root, directory, "cmd.exe")
        tried.append(candidate)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    found = shutil.which("cmd.exe")
    if found:
        return os.path.abspath(found)
    raise FileNotFoundError("cmd.exe not found; tried " + ", ".join(tried + ["PATH"]))


def shell_executable() -> str | None:
    """Resolve Windows cmd.exe; return None on POSIX (retain caller defaults)."""
    return find_cmd_exe()


def spawn_shell_argv(command: str) -> list[str]:
    """Windows: cmd /d /s /c; POSIX: existing Bash resolution, then /bin/sh."""
    if IS_WINDOWS:
        return [find_cmd_exe(), "/d", "/s", "/c", command]
    shell = os.environ.get("SHAREDAGENTS_EXEC_SHELL") or shutil.which("bash")
    if not shell:
        shell = next((p for p in ("/bin/bash", "/usr/bin/bash") if os.path.exists(p)), "/bin/sh")
    return [shell, "-c", command]


def detach_kwargs() -> dict:
    """Windows CREATE_NEW_PROCESS_GROUP; POSIX {} preserves caller policy.

    Existing session leaders must retain start_new_session=not IS_WINDOWS at
    their call sites. Windows silently ignores start_new_session=True.
    """
    if IS_WINDOWS:
        # The constant is exported by subprocess only on Windows.
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
    return {}


@lru_cache(maxsize=1)
def _windows_api():
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    return kernel32


def _posix_state(pid: int) -> str:
    """Linux ``/proc/<pid>/stat`` state letter, or ``""`` when unreadable."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            # comm can contain spaces and parentheses, so split on the LAST ")".
            return handle.read().decode("utf-8", "replace").rsplit(") ", 1)[1][:1]
    except (OSError, IndexError):
        return ""


def process_alive(pid: int, *, permission_denied_alive: bool = True) -> bool:
    """Query Windows process handles, never os.kill(pid, 0) on Windows.

    POSIX uses signal zero; denied queries normally count as alive. Callers
    with a historical False-on-denial policy can preserve it explicitly.
    Windows NULL handles or failed exit-code queries return False.
    """
    if pid <= 0:
        return False
    if IS_WINDOWS:
        from ctypes import wintypes

        api = _windows_api()
        handle = api.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(api.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return permission_denied_alive
    # A zombie still owns its PID slot, so signal zero succeeds for a process that
    # has already exited. Treating one as alive made stop() poll a dead daemon for
    # its whole timeout and then report a FALSE failure, escalating to a SIGKILL the
    # zombie cannot even receive. Reproduced (SIGTERM -> state Z in 0.1-0.4s, yet
    # process_alive True) whenever the launcher's parent exits and PID 1 does not
    # reap promptly -- e.g. inside a container. Unreadable /proc keeps the old
    # permissive answer.
    return _posix_state(pid) != "Z"


def kill_process(pid: int, force: bool = False, *, group: bool = False) -> None:
    """POSIX TERM/KILL a pid (or explicit pgid); Windows TerminateProcess.

    Callers retain their own grace periods, then call force=True to escalate.
    Windows terminates only the immediate process: grandchildren may survive.
    No Job Objects, /proc walk, or POSIX signals are used on Windows.
    """
    if pid <= 0:
        raise ValueError("pid must be positive")
    if not IS_WINDOWS:
        (os.killpg if group else os.kill)(pid, FORCE_KILL if force else signal.SIGTERM)
        return
    api = _windows_api()
    handle = api.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: process no longer exists
            raise ProcessLookupError(errno.ESRCH, "process no longer exists", pid)
        raise ctypes.WinError(error)
    try:
        if not api.TerminateProcess(handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        api.CloseHandle(handle)


def process_group(pid: int) -> int:
    """Return the POSIX pgid or the child pid used for Windows cancellation."""
    return pid if IS_WINDOWS else os.getpgid(pid)


def signal_process_group(pgid: int, sig: int) -> None:
    """Signal a POSIX group; Windows terminates its recorded immediate child."""
    if IS_WINDOWS:
        kill_process(pgid, force=sig == FORCE_KILL)
    else:
        os.killpg(pgid, sig)


def port_in_use(host: str, port: int) -> bool:
    """Probe a TCP bind without /proc; permission-denied binds count as busy."""
    host = host.strip("[]")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        if IS_WINDOWS:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                return True
            raise
    return False


def port_owner_pid(host: str, port: int) -> int | None:
    """Windows TCP listener owner for a numeric address, including wildcard binds.

    This is discovery, NOT authority to terminate a process. Callers must prove
    ancestry and creation identity separately. Fail closed on command failure
    or ambiguous owners; never confuse a remote endpoint or TIME_WAIT with a
    listener. POSIX callers retain their existing discovery implementations.
    """
    if not IS_WINDOWS:
        raise NotImplementedError("port_owner_pid requires Windows")
    address = ipaddress.ip_address(host.strip("[]"))
    if not 0 < port < 65536:
        raise ValueError("port must be between 1 and 65535")
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"  # noqa: SIM112 - Windows spelling
    result = subprocess.run(
        [str(system / "netstat.exe"), "-ano", "-p", "tcpv6" if address.version == 6 else "tcp"],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
        timeout=5,
    )
    owners = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0].upper() != "TCP" or fields[3] != "LISTENING":
            continue
        try:
            local_host, local_port = fields[1].rsplit(":", 1)
            local = ipaddress.ip_address(local_host.strip("[]"))
            pid = int(fields[-1])
            matches = local == address or (local.version == address.version and local.is_unspecified)
            if matches and int(local_port) == port and pid > 0:
                owners.add(pid)
        except ValueError:
            continue
    if len(owners) > 1:
        raise RuntimeError(f"ambiguous listener owners for {host}:{port}: {sorted(owners)}")
    return next(iter(owners), None)


def windows_process_parents() -> dict[int, int]:
    """Snapshot PID -> parent PID using Toolhelp; identities need separate checks.

    Parent PIDs can be recycled. Hold handles and compare creation times before
    relying on these links. No process names, WMI, PowerShell, or dependencies.
    """
    if not IS_WINDOWS:
        raise NotImplementedError("process snapshot requires Windows")
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    api = _windows_api()
    api.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for name in ("Process32FirstW", "Process32NextW"):
        fn = getattr(api, name)
        fn.argtypes = (wintypes.HANDLE, ctypes.POINTER(ProcessEntry))
        fn.restype = wintypes.BOOL
    handle = api.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        parents = {}
        more = api.Process32FirstW(handle, ctypes.byref(entry))
        while more:
            parents[entry.th32ProcessID] = entry.th32ParentProcessID
            more = api.Process32NextW(handle, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error())
        return parents
    finally:
        api.CloseHandle(handle)
