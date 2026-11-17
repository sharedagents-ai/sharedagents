#!/usr/bin/env sh
# sharedagents installer — the one-line way in.
#
#     curl -fsSL https://sharedagents.ai/install.sh | sh
#
# Installs sharedagents and puts the `sharedagents` command on your PATH, then tells
# you how to start. First run drops you into a free NVIDIA Nemotron chat
# with no signup, or lets you pick your own provider.
#
# Install target, in order of preference:
#   1. an ACTIVE virtualenv ($VIRTUAL_ENV)  → install straight into it
#   2. pipx                                  → isolated app install
#   3. pip --user                            → ~/.local + PATH wiring
# Nothing runs as root; no system packages are touched.
#
# Knobs (env vars):
#   SHAREDAGENTS_PIP_SPEC   what to install (default: "sharedagents" from PyPI).
#                      For a branch/local build, point it at a repo or wheel:
#                        SHAREDAGENTS_PIP_SPEC="git+https://example/sharedagents.git"
#                        SHAREDAGENTS_PIP_SPEC=/tmp/sharedagents-0.1.0-py3-none-any.whl
#   SHAREDAGENTS_NO_PATH=1  skip editing shell rc files (PATH is left alone).
set -eu

PIP_SPEC="${SHAREDAGENTS_PIP_SPEC:-sharedagents}"

say()  { printf '\033[36m[sharedagents]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[sharedagents] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m[sharedagents] %s\033[0m\n' "$*" >&2; exit 1; }

# ---- 1. Pick the interpreter — an active venv wins ------------------------
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PY="$VIRTUAL_ENV/bin/python"
  IN_VENV=1
else
  PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
  IN_VENV=0
fi
[ -n "$PY" ] || die "Python 3 not found. Install Python 3.10+ and re-run."

PYV="$("$PY" - <<'EOF'
import sys
print("%d.%d" % sys.version_info[:2])
sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)
EOF
)" || die "Python $PYV is too old; sharedagents needs 3.10+."
[ "$IN_VENV" = 1 ] && say "using active venv: $VIRTUAL_ENV (python $PYV)" \
                    || say "using python $PYV ($PY)"

# ---- 2. Make sure pip exists for $PY (bootstrap via ensurepip) ------------
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  say "pip not present — bootstrapping with ensurepip…"
  "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi
HAVE_PIP=0
"$PY" -m pip --version >/dev/null 2>&1 && HAVE_PIP=1

# ---- 3. Install ----------------------------------------------------------
TARGET_BIN=""
if [ "$IN_VENV" = 1 ]; then
  [ "$HAVE_PIP" = 1 ] || die "active venv $VIRTUAL_ENV has no pip (and ensurepip failed). Recreate it: python3 -m venv $VIRTUAL_ENV"
  say "installing sharedagents into the venv — $PIP_SPEC"
  "$PY" -m pip install --upgrade "$PIP_SPEC"
  TARGET_BIN="$VIRTUAL_ENV/bin"
else
  # Prefer pipx (isolated). Try to bootstrap it if pip is available.
  HAVE_PIPX=0
  if command -v pipx >/dev/null 2>&1 || "$PY" -m pipx --version >/dev/null 2>&1; then
    HAVE_PIPX=1
  elif [ "$HAVE_PIP" = 1 ]; then
    say "installing pipx (isolated CLI installer)…"
    "$PY" -m pip install --user -q pipx >/dev/null 2>&1 \
      || "$PY" -m pip install --user --break-system-packages -q pipx >/dev/null 2>&1 || true
    "$PY" -m pipx --version >/dev/null 2>&1 && HAVE_PIPX=1
  fi

  if [ "$HAVE_PIPX" = 1 ]; then
    say "installing sharedagents via pipx — $PIP_SPEC"
    "$PY" -m pipx install --force "$PIP_SPEC"
    "$PY" -m pipx ensurepath >/dev/null 2>&1 || true
    TARGET_BIN="$("$PY" -c 'import site,os; print(os.path.join(site.getuserbase(),"bin"))' 2>/dev/null || echo "$HOME/.local/bin")"
  elif [ "$HAVE_PIP" = 1 ]; then
    say "installing sharedagents via pip --user — $PIP_SPEC"
    "$PY" -m pip install --user "$PIP_SPEC" >/dev/null 2>&1 \
      || "$PY" -m pip install --user --break-system-packages "$PIP_SPEC" \
      || die "pip install failed. See the output above."
    TARGET_BIN="$("$PY" -c 'import site,os; print(os.path.join(site.getuserbase(),"bin"))' 2>/dev/null || echo "$HOME/.local/bin")"
  else
    # No system pip — common on Debian/Ubuntu, which strip pip AND
    # ensurepip out of the base python. Bootstrap a managed venv with the
    # stdlib `venv` module (a venv brings its own pip), install into it,
    # and symlink the entrypoint onto ~/.local/bin. Still no sudo, still
    # no system packages touched.
    MANAGED_VENV="$HOME/.local/share/sharedagents/venv"
    say "no system pip — creating a managed venv at $MANAGED_VENV"
    if "$PY" -m venv "$MANAGED_VENV" >/dev/null 2>&1 && [ -x "$MANAGED_VENV/bin/python" ]; then
      "$MANAGED_VENV/bin/python" -m pip install --upgrade "$PIP_SPEC" \
        || die "install into the managed venv failed. See the output above."
      mkdir -p "$HOME/.local/bin"
      ln -sf "$MANAGED_VENV/bin/sharedagents" "$HOME/.local/bin/sharedagents"
      TARGET_BIN="$HOME/.local/bin"
    else
      die "No pip, and couldn't create a venv (the stdlib 'venv' module is
    missing). Install just that — no other system changes:
        sudo apt install python3-venv
    then re-run this script."
    fi
  fi
fi

export PATH="$TARGET_BIN:$PATH"

# ---- 4. PATH wiring (skip for venvs — venv/bin is on PATH when active) ----
if [ "$IN_VENV" != 1 ] && [ "${SHAREDAGENTS_NO_PATH:-0}" != "1" ]; then
  for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
    [ -f "$rc" ] || continue
    grep -q 'sharedagents installer PATH' "$rc" 2>/dev/null && continue
    printf '\n# sharedagents installer PATH\ncase ":$PATH:" in *":%s:"*) ;; *) export PATH="%s:$PATH";; esac\n' \
      "$TARGET_BIN" "$TARGET_BIN" >> "$rc"
    say "added $TARGET_BIN to PATH in $rc"
  done
fi

# ---- 5. Verify + next steps ----------------------------------------------
if command -v sharedagents >/dev/null 2>&1; then
  say "✓ installed: $(command -v sharedagents)"
  say ""
  say "  Start it:   sharedagents"
  say "  First run:  guided setup — set a key (free OpenRouter key, or any"
  say "              provider) or attach a local model, then chat."
  say "  Optional:   PDF/DOCX/YouTube tools →  pip install 'sharedagents[full]'"
  [ "$IN_VENV" = 1 ] || say "  (open a new terminal if 'sharedagents' isn't found — PATH was just updated)"
else
  warn "sharedagents installed but not on PATH in THIS shell."
  warn "open a new terminal, or run:  export PATH=\"$TARGET_BIN:\$PATH\""
fi
