#!/usr/bin/env bash
# PM2 launch wrapper for the rullama devserver.
#
# Why this exists: the devserver runs with `--no-vite` and serves the prebuilt
# `web/dist/`. Without a build step, `pm2 restart` just relaunches the static
# server and your source edits never show up. This wrapper REBUILDS web/dist
# (vite) on every (re)start, then `exec`s the server so PM2 supervises the
# binary directly (autorestart still works; the bash process is replaced).
#
# On each (re)start this wrapper rebuilds — ONLY when stale — the two artifacts
# the workspace/pnpm does NOT build for you, then bundles them with vite:
#   1. the devserver binary  (EXCLUDED from the workspace; built via
#      `--manifest-path dev-server/Cargo.toml`, its own target/)
#   2. the wasm bundle pkg/  (built via wasm-pack from rullama-lora)
# So a `pm2 restart` always ships CURRENT artifacts; a no-op restart stays fast.
# This guard exists because a bumped Rust dep (e.g. rsqlite-wasm) otherwise
# silently keeps serving a wasm built against the OLD version — the #1 ghost
# bug in this project (you debug source that's already fixed).
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
# cargo + wasm-pack for the two manual builds below (NOT on the pnpm path).
export PATH="$HOME/.cargo/bin:$PATH"
echo "[pm2-start] node = $(command -v node || echo '?') $(node -v 2>/dev/null || echo '')"

# ── Stale-build guard ──────────────────────────────────────────────────────
# Echoes a non-empty string when $target is missing OR any input (the remaining
# args — files/dirs) is newer than it. mtime-based, like `make`: only rebuilds
# when inputs actually changed.
stale() {
    local target="$1"; shift
    [ -f "$target" ] || { echo missing; return; }
    find "$@" -newer "$target" -print -quit 2>/dev/null
}

# 1. Devserver binary — manual build, EXCLUDED from the workspace (so `-p` /
#    `cargo build --workspace` won't build it). Rebuild via --manifest-path.
DEVSERVER_BIN="$REPO_ROOT/dev-server/target/release/rullama-devserver"
if [ -n "$(stale "$DEVSERVER_BIN" dev-server/src dev-server/Cargo.toml dev-server/Cargo.lock)" ]; then
    echo "[pm2-start] devserver stale — rebuilding (manual, --manifest-path)…"
    cargo build --release --manifest-path dev-server/Cargo.toml \
        && echo "[pm2-start] devserver rebuilt ✓" \
        || echo "[pm2-start] ⚠ devserver build FAILED — launching the previous binary." >&2
else
    echo "[pm2-start] devserver up-to-date ✓"
fi

# 2. WASM bundle (pkg/) — built from the sibling rullama-framework engine checkout
#    (the engine moved out of this repo). Rebuilt when any engine source or
#    manifest is newer than the built wasm. --out-name rullama keeps the PWA's
#    /pkg/rullama.js import stable. With no engine checkout, serve prebuilt pkg/.
ENGINE_DIR="${RULLAMA_ENGINE_DIR:-$REPO_ROOT/../rullama-framework/engine}"
PKG_WASM="$REPO_ROOT/pkg/rullama_bg.wasm"
if [ -d "$ENGINE_DIR" ]; then
    if [ -n "$(stale "$PKG_WASM" "$ENGINE_DIR/rullama-engine/src" "$ENGINE_DIR/rullama-lora/src" "$ENGINE_DIR/rullama-engine/Cargo.toml" "$ENGINE_DIR/rullama-lora/Cargo.toml" "$ENGINE_DIR/Cargo.toml")" ]; then
        echo "[pm2-start] wasm stale — rebuilding from engine ($ENGINE_DIR)…"
        ( cd "$ENGINE_DIR" && wasm-pack build rullama-lora --target web --release --out-dir "$REPO_ROOT/pkg" --out-name rullama ) \
            && echo "[pm2-start] wasm rebuilt ✓" \
            || echo "[pm2-start] ⚠ wasm-pack build FAILED — serving the previous pkg/. Fix and restart." >&2
    else
        echo "[pm2-start] wasm up-to-date ✓"
    fi
else
    echo "[pm2-start] no engine checkout at $ENGINE_DIR — serving prebuilt pkg/ as-is."
fi

echo "[pm2-start] rebuilding web/dist (vite)…"
if (cd web && pnpm exec vite build); then
    echo "[pm2-start] web/dist rebuilt ✓"
else
    # Don't crash-loop the server on a TS error — serve the last good dist and
    # warn loudly. Fix the build, then `pm2 restart` again.
    echo "[pm2-start] ⚠ vite build FAILED — serving the previous web/dist. Fix the error and restart." >&2
fi

echo "[pm2-start] launching rullama-devserver (--public, static dist)"
exec "$DEVSERVER_BIN" \
    --public \
    --host 127.0.0.1 \
    --port 25321 \
    --no-vite \
    --cors-origins https://rullama.brainwires.net
