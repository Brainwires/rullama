#!/usr/bin/env bash
# Tiny static server for the rullama PWA demo.
# Serves the project root so /pkg/* and /examples/pwa/* are both reachable.
# Sets the COOP/COEP headers WebGPU needs, plus correct content types for .wasm.
#
# Robust against:
#   - leftover server from a previous run (auto-kills whatever's on $PORT)
#   - TCP TIME_WAIT after a hard kill (SO_REUSEADDR via allow_reuse_address)
#   - Ctrl-C exiting the bash wrapper but orphaning the Python child (trap + exec)

set -euo pipefail
cd "$(dirname "$0")/../.."

PORT="${PORT:-8088}"

# Kill anything already on $PORT so re-running this script "just works".
if existing=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null); then
    if [ -n "$existing" ]; then
        echo "killing existing listener(s) on :$PORT (pids: $existing)"
        echo "$existing" | xargs kill -9 2>/dev/null || true
        sleep 0.3
    fi
fi

echo "rullama PWA demo:"
echo "  http://localhost:$PORT/examples/pwa/index.html"
echo "  (Ctrl-C to stop)"
echo

# `exec` so Ctrl-C kills Python directly instead of the bash parent.
exec python3 - "$PORT" <<'PY'
import http.server, socketserver, sys, mimetypes, signal

PORT = int(sys.argv[1])
mimetypes.add_type("application/wasm", ".wasm")

class Handler(http.server.SimpleHTTPRequestHandler):
    # Quiet down the per-request "GET /…" log lines (kept ERROR/exception output).
    def log_message(self, fmt, *args): pass

    def end_headers(self):
        # WebGPU + cross-origin-isolated headers.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        super().end_headers()

class ReusableServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True   # SO_REUSEADDR — survives quick restarts.
    daemon_threads = True

def shutdown(*_):
    print()
    sys.exit(0)
signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

with ReusableServer(("", PORT), Handler) as httpd:
    print(f"serving at http://localhost:{PORT}/")
    httpd.serve_forever()
PY
