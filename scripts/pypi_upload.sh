#!/usr/bin/env bash
# Upload dist/ to PyPI using the sharedagents project token.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Repo-local guard first: fail on private IPs, forge URLs or key material.
# A denylist of deployment-specific identity strings can be supplied via
# SHAREDAGENTS_SCRUB_DENYLIST (see scripts/check_forbidden_patterns.py).
python3 "$ROOT/scripts/check_forbidden_patterns.py" "$ROOT/dist"

ENV_FILE="${SHAREDAGENTS_ENV_FILE:-$HOME/.sharedagents/.env}"
if [[ -z "${SHAREDAGENTS_PYPI_TOKEN:-}" && -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a && source "$ENV_FILE" && set +a
fi
if [[ -z "${SHAREDAGENTS_PYPI_TOKEN:-}" ]]; then
  echo "SHAREDAGENTS_PYPI_TOKEN not set (expected in $ENV_FILE)" >&2
  exit 1
fi

if ! compgen -G "dist/*.whl" >/dev/null && ! compgen -G "dist/*.tar.gz" >/dev/null; then
  echo "No dist artifacts under $ROOT/dist — run: python3 -m build" >&2
  exit 1
fi

exec python3 -m twine upload dist/* --skip-existing -u __token__ -p "$SHAREDAGENTS_PYPI_TOKEN"
