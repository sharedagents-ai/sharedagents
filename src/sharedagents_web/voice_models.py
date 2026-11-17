"""Download the wake-word ONNX models the MIT package cannot ship.

The openWakeWord feature models and wake-word classifiers are licensed
CC BY-NC-SA 4.0 (non-commercial) by their author, so they are fetched on
demand into ``<sharedagents home>/voice-models/`` instead of riding along in
the wheel. The web server resolves ``/static/models/<file>`` from that
directory first, then from the bundled ``static/models/`` (which ships only
the MIT-licensed ``silero_vad.onnx`` and the onnxruntime WASM runtime).

Every download is verified against a pinned SHA-256 before it is moved into
place, so a truncated or substituted upstream asset can never be served.

Usage::

    sharedagents web fetch-voice-models            # everything the engine needs
    sharedagents web fetch-voice-models --keyword hey_mycroft_v0.1
    sharedagents web fetch-voice-models --list
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from sharedagents_core.layout import voice_models_root

_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"

#: Models needed by the in-browser wake-word engine, plus every upstream
#: wake-word classifier a ``wake.keyword`` config can name. ``silero_vad.onnx``
#: is MIT-licensed and ships bundled — it is intentionally absent here.
MODELS: dict[str, dict[str, str]] = {
    "melspectrogram.onnx": {
        "url": f"{_RELEASE}/melspectrogram.onnx",
        "sha256": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
        "required": "yes",
    },
    "embedding_model.onnx": {
        "url": f"{_RELEASE}/embedding_model.onnx",
        "sha256": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
        "required": "yes",
    },
    "hey_jarvis_v0.1.onnx": {
        "url": f"{_RELEASE}/hey_jarvis_v0.1.onnx",
        "sha256": "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
        "required": "default keyword",
    },
    "alexa_v0.1.onnx": {
        "url": f"{_RELEASE}/alexa_v0.1.onnx",
        "sha256": "6ff566a01d12670e8d9e3c59da32651db1575d17272a601b7f8a39283dfbae3e",
        "required": "optional keyword",
    },
    "hey_mycroft_v0.1.onnx": {
        "url": f"{_RELEASE}/hey_mycroft_v0.1.onnx",
        "sha256": "c2a311e8fa1338de89c31b3b46dc4dffd4af2f9a8d6ddead48893c2d301b1f18",
        "required": "optional keyword",
    },
    "hey_rhasspy_v0.1.onnx": {
        "url": f"{_RELEASE}/hey_rhasspy_v0.1.onnx",
        "sha256": "5a9b3ed3be2910e35780e097905aa9f35a9c10038df47914cf2b3ec4d670f6ea",
        "required": "optional keyword",
    },
}

REQUIRED = tuple(name for name, m in MODELS.items() if m["required"] != "optional keyword")

LICENSE_NOTICE = """\
NOTE: these models are part of openWakeWord (dscripka/openWakeWord) and are
licensed CC BY-NC-SA 4.0 — free for personal / non-commercial use only. They
are downloaded to your own machine at your request; nothing here grants a
commercial license. MIT-licensed sharedagents code is unaffected.
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_one(name: str, dest_dir: Path, *, urlopen=urllib.request.urlopen) -> Path:
    """Download ``name`` into ``dest_dir``, verifying the pinned SHA-256.

    Returns the final path. Raises ``ValueError`` for unknown names and
    ``RuntimeError`` when the digest does not match the pin — the file is
    deleted in that case and the existing target (if any) is left untouched.
    """
    spec = MODELS.get(name)
    if spec is None:
        raise ValueError(f"unknown model {name!r} (choose from: {', '.join(MODELS)})")
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / name
    fd, tmp_name = tempfile.mkstemp(dir=dest_dir, prefix=f".{name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urlopen(spec["url"], timeout=60) as resp:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
        expected = spec["sha256"]
        if not expected:
            raise RuntimeError(f"{name}: no pinned sha256 — refusing to install an unverified model")
        actual = _sha256(tmp)
        if actual != expected:
            raise RuntimeError(f"{name}: sha256 mismatch (got {actual}, expected {expected}) — refusing to install")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def fetch(names: list[str], dest_dir: Path, *, urlopen=urllib.request.urlopen) -> list[Path]:
    return [fetch_one(name, dest_dir, urlopen=urlopen) for name in names]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="sharedagents web fetch-voice-models",
        description=__doc__.splitlines()[0],
    )
    p.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="target directory (default: <sharedagents home>/voice-models)",
    )
    p.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="also fetch this wake-word model (e.g. --keyword hey_mycroft_v0.1); repeatable",
    )
    p.add_argument("--list", action="store_true", help="list models and exit")
    args = p.parse_args(argv)

    if args.list:
        for name, spec in MODELS.items():
            print(f"{name:28} {spec['required']}")
        return 0

    dest = args.dir or voice_models_root()
    extra = [k if k.endswith(".onnx") else f"{k}.onnx" for k in args.keyword]
    names = list(dict.fromkeys([*REQUIRED, *extra]))
    print(LICENSE_NOTICE)
    for path in fetch(names, dest):
        print(f"fetched {path.name} -> {path.parent}")
    print("done — no restart needed; reload the web page and switch voice to wake mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
