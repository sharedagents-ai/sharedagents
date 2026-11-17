"""Static hosting isolation and real nginx HTTP acceptance checks."""

import os
import re
import shlex
import subprocess
from html import unescape
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_container_isolation():
    config = yaml.safe_load((ROOT / "deploy/website/compose.yml").read_text())
    assert config["name"] == "sharedagents-public"
    assert config["networks"]["static"]["internal"] is True
    for name, user, networks in [
        ("website", "101:101", {"static"}),
        ("tunnel", "65532:65532", {"static", "egress"}),
    ]:
        service = config["services"][name]
        assert "@sha256:" in service["image"]
        assert service["user"] == user
        assert service["read_only"] and service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]
        assert set(service["networks"]) == networks
        assert not any(service.get(k) for k in ["ports", "network_mode", "environment", "env_file", "privileged"])
        assert all(v.endswith(":ro") and "docker.sock" not in v for v in service["volumes"])
        assert service["mem_limit"] == "128m" and service["cpus"] == 0.5 and service["pids_limit"] == 64
        assert service["logging"]["options"] == {"max-size": "10m", "max-file": "3"}
    assert "--token-file" in config["services"]["tunnel"]["command"]
    assert "--token" not in config["services"]["tunnel"]["command"]
    assert not any("token" in v for v in config["services"]["website"]["volumes"])
    assert "Host: sharedagents.ai" in str(config["services"]["website"]["healthcheck"])


@pytest.fixture
def http():
    container = os.environ.get("WEBSITE_TEST_CONTAINER")
    if not container:
        pytest.skip("Set WEBSITE_TEST_CONTAINER to run against actual isolated nginx")
    docker = shlex.split(os.environ.get("WEBSITE_DOCKER_COMMAND", "docker"))

    def request(path, host="sharedagents.ai"):
        message = f"GET {path} HTTP/1.0\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode()
        result = subprocess.run(
            [*docker, "exec", "-i", container, "nc", "-w", "3", "127.0.0.1", "8080"],
            input=message,
            capture_output=True,
            timeout=15,
            check=True,
        )
        head, body = result.stdout.split(b"\r\n\r\n", 1)
        lines = head.decode().splitlines()
        headers = dict(line.lower().split(": ", 1) for line in lines[1:] if ": " in line)
        return int(lines[0].split()[1]), headers, body

    return request


@pytest.mark.parametrize("host", ["garbage.example.com", "127.0.0.1"])
def test_unknown_host_is_404(http, host):
    assert http("/", host)[0] == 404


def test_apex_and_headers(http):
    status, headers, body = http("/")
    assert status == 200 and b"Your agents." in body
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"
    assert "location" not in headers


def test_www_redirect_preserves_path_query(http):
    status, headers, _ = http("/docs/getting-started/installation/?x=1&y=two", "www.sharedagents.ai")
    assert status == 301
    assert headers["location"] == "https://sharedagents.ai/docs/getting-started/installation/?x=1&y=two"


@pytest.mark.parametrize("path", ["/docs", "/docs/getting-started/installation"])
def test_directory_redirect_is_relative_and_resolves_assets(http, path):
    status, headers, _ = http(path)
    assert status == 301 and headers["location"] == path + "/"
    status, _, body = http(headers["location"])
    assert status == 200 and b"SharedAgents" in body
    stylesheets = re.findall(r'href="([^"]+\.css[^\"]*)"', body.decode())
    assert stylesheets
    for href in stylesheets:
        url = urlsplit(urljoin("https://example.com" + path + "/", unescape(href)))
        assert url.hostname == "example.com"
        assert http(url.path + ("?" + url.query if url.query else ""))[0] == 200


@pytest.mark.parametrize("path", ["/missing", "/docs/missing", "/docs/deep/missing"])
def test_missing_routes_serve_custom_404(http, path):
    status, _, body = http(path)
    assert status == 404 and b"Page not found" in body and b'href="/docs/"' in body


@pytest.mark.parametrize("path", ["/.env", "/docs/.git/config"])
def test_dotfiles_denied(http, path):
    assert http(path)[0] == 403


def test_installer_is_unchanged_shell_content(http):
    status, headers, body = http("/install.sh")
    assert status == 200 and headers["content-type"] == "text/x-shellscript"
    assert body == (ROOT / "install.sh").read_bytes()


def test_docs_search_sitemap_and_assets(http):
    assert http("/docs/search/search_index.json")[0] == 200
    assert b"<urlset" in http("/docs/sitemap.xml")[2]
    assert b"/docs/sitemap.xml" in http("/robots.txt")[2]
    assert http("/assets/favicon.svg")[0] == 200
