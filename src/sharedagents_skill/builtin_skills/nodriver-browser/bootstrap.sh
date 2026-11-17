#!/usr/bin/env bash
# Materialize the nodriver-browser skill into ~/.sharedagents/skills/ and build
# its venv. Idempotent — safe to re-run to pull updated scripts.
#
# Usage:  bash bootstrap.sh
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"          # this builtin dir (the package copy)
DST="$HOME/.sharedagents/skills/nodriver-browser"

mkdir -p "$DST/scripts"
cp "$SRC/browser"               "$DST/browser"
cp "$SRC/scripts/browser.py"    "$DST/scripts/browser.py"
cp "$SRC/scripts/requirements.txt" "$DST/scripts/requirements.txt"
chmod +x "$DST/browser"

if [ ! -x "$DST/scripts/venv/bin/python" ]; then
  echo "[bootstrap] creating venv…"
  python3 -m venv "$DST/scripts/venv"
fi
echo "[bootstrap] installing requirements…"
"$DST/scripts/venv/bin/pip" install -q -r "$DST/scripts/requirements.txt"

echo "[bootstrap] done. Try:"
echo "  $DST/browser --display :10 open --session live --url https://example.com"
