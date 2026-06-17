#!/usr/bin/env bash
# PM2 launch wrapper for the rullama devserver.
#
# Why this exists: the devserver runs with `--no-vite` and serves the prebuilt
# `web/dist/`. Without a build step, `pm2 restart` just relaunches the static
# server and your source edits never show up. This wrapper REBUILDS web/dist
# (vite) on every (re)start, then `exec`s the server so PM2 supervises the
# binary directly (autorestart still works; the bash process is replaced).
#
# Cost: ~3 s of vite build added to each restart — the point of the restart.
# Note: a Rust/wasm change still needs the wasm bundle rebuilt first
# (`wasm-pack build crates/rullama-finetune --target web --release --out-dir
# ../../pkg --out-name rullama`); the devserver's background watch does this on
# save when not run with --no-watch, and this vite build then bundles it in.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# PM2 (esp. under launchd) launches with a minimal env and the WRONG node —
# vite needs the same node your shell uses or it errors mid-build. Load nvm so
# the build runs on your default node, and make pnpm reachable.
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    # shellcheck disable=SC1091
    . "$NVM_DIR/nvm.sh" >/dev/null 2>&1 || true
fi
export PATH="$HOME/.local/share/pnpm:$HOME/.local/bin:$PATH"
echo "[pm2-start] node = $(command -v node || echo '?') $(node -v 2>/dev/null || echo '')"

echo "[pm2-start] rebuilding web/dist (vite)…"
if (cd web && pnpm exec vite build); then
    echo "[pm2-start] web/dist rebuilt ✓"
else
    # Don't crash-loop the server on a TS error — serve the last good dist and
    # warn loudly. Fix the build, then `pm2 restart` again.
    echo "[pm2-start] ⚠ vite build FAILED — serving the previous web/dist. Fix the error and restart." >&2
fi

echo "[pm2-start] launching rullama-devserver (--public, static dist)"
exec "$REPO_ROOT/crates/rullama-devserver/target/release/rullama-devserver" \
    --public \
    --host 127.0.0.1 \
    --port 25321 \
    --no-vite \
    --cors-origins https://rullama.brainwires.net
