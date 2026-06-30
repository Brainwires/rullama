#!/usr/bin/env bash
# Set up `cargo dev --public` as a PM2-managed background service.
#
# Idempotent: re-running rebuilds + restarts. Designed to be safe to
# invoke after every code change (though for routine dev you'd rather
# use `pm2 restart rullama-devserver` directly).
#
# Usage:
#   ./ops/pm2/setup.sh
#
# After first run, enable boot survival (one-time, requires sudo):
#   sudo pm2 startup launchd -u $USER --hp $HOME
#   pm2 save
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "==> repo: $REPO_ROOT"

if ! command -v pm2 >/dev/null 2>&1; then
    echo "✗ pm2 not found. Install with: npm install -g pm2"
    exit 1
fi
echo "==> pm2 $(pm2 --version)"

if ! command -v cargo >/dev/null 2>&1; then
    echo "✗ cargo not found."
    exit 1
fi
if ! command -v pnpm >/dev/null 2>&1; then
    echo "✗ pnpm not found."
    exit 1
fi

echo "==> building rullama-devserver (release)"
cargo build \
    --manifest-path services/dev-server/Cargo.toml \
    --release

if [ ! -f "apps/web/dist/index.html" ]; then
    echo "==> dist/ missing; building PWA"
    (cd apps/web && pnpm install --frozen-lockfile && pnpm exec vite build)
else
    echo "==> dist/ present; skipping PWA build (rebuild with: cd apps/web && pnpm exec vite build)"
fi

echo "==> stopping any prior PM2 entry"
pm2 delete rullama-devserver >/dev/null 2>&1 || true

echo "==> starting via ecosystem.config.cjs"
pm2 start "$REPO_ROOT/ops/pm2/ecosystem.config.cjs"

echo "==> saving PM2 process list (for `pm2 resurrect` on reboot)"
pm2 save

echo
echo "✓ devserver running. Useful commands:"
echo "    pm2 status"
echo "    pm2 logs rullama-devserver"
echo "    pm2 restart rullama-devserver"
echo "    pm2 stop rullama-devserver"
echo
echo "For boot survival (one-time, requires sudo):"
echo "    sudo pm2 startup launchd -u \$USER --hp \$HOME"
echo "    pm2 save"
echo
echo "Test it locally:"
echo "    curl -i http://127.0.0.1:25321/api/blob/gemma4:e2b -I"
echo
echo "Public URL (via Cloudflare tunnel):"
echo "    https://rullama.brainwires.net"
