#!/usr/bin/env bash
# Tiny static server for the rullama PWA demo.
# Serves the project root so /pkg/* and /examples/pwa/* are both reachable.
# Sets the COOP/COEP headers WebGPU needs, plus correct content types for .wasm.

set -euo pipefail
cd "$(dirname "$0")/../.."

PORT="${PORT:-8088}"

echo "rullama PWA demo:"
echo "  http://localhost:$PORT/examples/pwa/index.html"
echo

# Python-based static server with the right headers. Keeps the dep set to "python3".
python3 - "$PORT" <<'PY'
import http.server, socketserver, sys, mimetypes

PORT = int(sys.argv[1])
mimetypes.add_type("application/wasm", ".wasm")

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # WebGPU + cross-origin-isolated headers.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        super().end_headers()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"serving at http://localhost:{PORT}/")
    httpd.serve_forever()
PY
