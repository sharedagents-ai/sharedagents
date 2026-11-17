"""The forge client resolves its backend from FORGE_* (or the older names)."""

from sharedagents_web import forge_client as fc

_ALL = (
    "FORGE_KIND", "FORGE_URL", "FORGE_TOKEN", "FORGE_API_URL", "FORGE_USER_AGENT", "FORGE_SRC",
    "SHAREDAGENTS_FORGE", "GITEA_BASE", "GITEA_TOKEN", "GITEA_USER_AGENT", "GITEA_SRC",
    "GITHUB_TOKEN", "GH_TOKEN", "GITHUB_BASE", "GITHUB_API",
)  # fmt: skip


def test_backend_url_token_and_api_root_resolution(monkeypatch):
    def configure(remote="", **env):
        for name in _ALL:
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(fc, "_remote_base", lambda: remote)
        return fc.kind(), fc.forge_url(), fc.forge_token(), fc._api_root() if fc.kind() else ""

    assert configure() == ("", "", "", "")
    assert fc.configured_reason() == "FORGE_TOKEN is not set"
    gitea = ("gitea", "https://git.example.com", "t", "https://git.example.com/api/v1")
    assert configure(FORGE_URL="https://git.example.com/", FORGE_TOKEN="t") == gitea
    assert configure(GITEA_BASE="https://git.example.com", GITEA_TOKEN="t") == gitea
    assert configure("https://git.example.com", FORGE_TOKEN="t") == gitea
    assert configure(FORGE_TOKEN="t") == ("github", "https://github.com", "t", "https://api.github.com")
    assert configure(GITHUB_TOKEN="t") == ("github", "https://github.com", "t", "https://api.github.com")
    assert configure(FORGE_KIND="github", FORGE_URL="https://ghe.example.com", FORGE_TOKEN="t") == (
        "github", "https://ghe.example.com", "t", "https://ghe.example.com/api/v3",
    )  # fmt: skip
    assert configure(GITEA_BASE="https://git.example.com", GITEA_TOKEN="g", GITHUB_TOKEN="h")[0] == "gitea"
    # FORGE_TOKEN with only a legacy URL follows that URL, never GitHub.
    assert configure(GITEA_BASE="https://git.example.com", FORGE_TOKEN="t") == gitea
    assert configure("https://github.com", GITEA_BASE="https://git.example.com", FORGE_TOKEN="t") == gitea
    assert configure(GITHUB_BASE="https://ghe.example.com", FORGE_TOKEN="t")[:3] == (
        "github", "https://ghe.example.com", "t",
    )  # fmt: skip
    # An explicit FORGE_URL decides the backend; the other forge's token is never used.
    assert configure(FORGE_URL="https://git.example.com", GITHUB_TOKEN="h") == ("", "", "", "")
    assert "no gitea token" in fc.configured_reason()
    assert configure(FORGE_URL="https://github.com", GITEA_TOKEN="g") == ("", "", "", "")
    assert "no github token" in fc.configured_reason()
    assert configure(FORGE_URL="https://git.example.com", GITEA_TOKEN="g", GITHUB_TOKEN="h") == (
        "gitea", "https://git.example.com", "g", "https://git.example.com/api/v1",
    )  # fmt: skip
    assert configure(FORGE_KIND="gitea", FORGE_TOKEN="t")[0] == ""
    assert "FORGE_URL or FORGE_TOKEN is missing" in fc.configured_reason()

    assert fc._base_from_remote("https://host.example/forge/acme/widgets.git") == "https://host.example/forge"
    assert fc._base_from_remote("https://git.example.com/acme/widgets.git") == "https://git.example.com"
    assert fc._base_from_remote("git@github.com:acme/widgets.git") == "https://github.com"
    assert fc._base_from_remote("not a url") == ""
