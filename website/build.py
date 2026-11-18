"""Build the static project website and documentation from this checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/website")
    args = parser.parse_args()
    output = args.out.resolve()
    build_root = (ROOT / "build").resolve()
    if output == build_root or not output.is_relative_to(build_root):
        parser.error("--out must be a subdirectory of build/")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="website-stage-", dir=build_root))
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copyfile(HERE / name, staging / name)
    shutil.copytree(HERE / "assets", staging / "assets")
    shutil.copytree(ROOT / "docs/assets/demo", staging / "assets/demo")
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(staging / "docs")],
        cwd=ROOT,
        check=True,
    )
    (staging / "robots.txt").write_text(
        "User-agent: *\nAllow: /\nSitemap: https://sharedagents.ai/docs/sitemap.xml\n", encoding="utf-8"
    )
    if output.exists():
        previous = output.with_name(output.name + "-previous-" + uuid.uuid4().hex[:8])
        output.rename(previous)
        print(f"Previous build preserved: {previous}")
    staging.rename(output)
    print(f"Website built: {output}")


if __name__ == "__main__":
    main()
