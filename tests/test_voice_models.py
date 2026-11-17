"""Voice/wake-word models: fetched on demand (CC BY-NC-SA), sha256-verified,
served from the operator home before the bundled package assets."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("SHAREDAGENTS_HOME", tempfile.mkdtemp(prefix="web-voice-models-"))

from sharedagents_web import voice_models  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


PAYLOAD = b"fake-onnx-payload"


def _urlopen_ok(payload):
    def open_(url, timeout=0):
        return _Resp(payload)

    return open_


def test_fetch_verifies_sha256_and_installs(tmp_path):
    name = "melspectrogram.onnx"
    expected = voice_models.MODELS[name]["sha256"]
    body = b"onnx-bytes"
    voice_models.MODELS[name]["sha256"] = hashlib.sha256(body).hexdigest()
    try:
        out = voice_models.fetch_one(name, tmp_path, urlopen=_urlopen_ok(body))
        assert out == tmp_path / name
        assert out.read_bytes() == body
    finally:
        voice_models.MODELS[name]["sha256"] = expected


def test_fetch_rejects_bad_sha256_and_leaves_existing_file(tmp_path):
    name = "melspectrogram.onnx"
    target = tmp_path / name
    target.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        voice_models.fetch_one(name, tmp_path, urlopen=_urlopen_ok(b"tampered"))
    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.part"))


def test_every_model_has_a_pinned_sha256():
    for name, spec in voice_models.MODELS.items():
        assert len(spec["sha256"]) == 64 and int(spec["sha256"], 16) >= 0, name


def test_fetch_refuses_a_model_without_a_pin(tmp_path, monkeypatch):
    name = "alexa_v0.1.onnx"
    monkeypatch.setitem(voice_models.MODELS, name, {**voice_models.MODELS[name], "sha256": ""})
    with pytest.raises(RuntimeError, match="no pinned sha256"):
        voice_models.fetch_one(name, tmp_path, urlopen=_urlopen_ok(b"anything"))
    assert not (tmp_path / name).exists()
    assert not list(tmp_path.glob("*.part"))


def test_fetch_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="unknown model"):
        voice_models.fetch_one("nope.onnx", tmp_path)


def test_bundled_nc_models_absent_but_silero_ships():
    """CC BY-NC-SA models must never ride in the package; MIT silero does."""
    models = Path(voice_models.__file__).parent / "static" / "models"
    assert (models / "silero_vad.onnx").is_file()
    for name in voice_models.MODELS:
        assert not (models / name).exists(), name


def test_cli_list(capsys):
    assert voice_models.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "melspectrogram.onnx" in out and "hey_jarvis_v0.1.onnx" in out


def test_route_prefers_operator_dir_then_package(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    from sharedagents_core.kernel import layout
    from sharedagents_web import server

    monkeypatch.setattr(layout, "sharedagents_root", lambda: tmp_path)
    user_models = tmp_path / "voice-models"
    user_models.mkdir()
    (user_models / "melspectrogram.onnx").write_bytes(b"from-home")
    client = server.app.test_client()

    r = client.get("/static/models/melspectrogram.onnx")
    assert r.status_code == 200 and r.data == b"from-home"

    r = client.get("/static/models/silero_vad.onnx")
    assert r.status_code == 200 and len(r.data) > 1000

    assert client.get("/static/models/missing.onnx").status_code == 404


def test_route_traversal_rejected(tmp_path, monkeypatch):
    pytest.importorskip("flask")
    from sharedagents_core.kernel import layout
    from sharedagents_web import server

    monkeypatch.setattr(layout, "sharedagents_root", lambda: tmp_path)
    client = server.app.test_client()
    r = client.get("/static/models/..%2f..%2fserver.py")
    assert r.status_code in (400, 404)
