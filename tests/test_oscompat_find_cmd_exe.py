"""find_cmd_exe keeps its C:\\Windows default when no system-root variable is set."""

from sharedagents_core import oscompat


def test_find_cmd_exe_defaults_the_system_root(monkeypatch):
    monkeypatch.setattr(oscompat, "IS_WINDOWS", True)
    for name in ("SHAREDAGENTS_EXEC_SHELL", "COMSPEC", "SystemRoot", "WINDIR"):
        monkeypatch.delenv(name, raising=False)
    probed = []

    def isfile(path):
        probed.append(path)
        return False

    monkeypatch.setattr(oscompat.os.path, "isfile", isfile)
    on_path = oscompat.os.path.abspath("cmd.exe")
    monkeypatch.setattr(oscompat.shutil, "which", lambda name: on_path)

    assert oscompat.find_cmd_exe() == on_path
    assert any(p.startswith("C:\\Windows") and "System32" in p for p in probed), probed
