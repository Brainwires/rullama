#!/usr/bin/env bash
# serve-tunnel.sh — HTTP backend for the React PWA, behind a Cloudflare
# tunnel that does TLS termination at the edge.
#
# Why: self-signed-cert HTTPS broke the PWA's worker init path on
# Chrome via CDP — every click ran but onClick never reached React's
# handler. Public HTTPS (via Cloudflare tunnel → rullama.brainwires.net)
# gives us a real signed cert and lets the PWA behave exactly like
# production.
#
# Serves (plain HTTP, same routes as serve-iphone.sh):
#   /                                 → examples/web/dist/index.html
#   /assets/*, /icons/*, /sw.js, …    → examples/web/dist/...
#   /pkg/*                            → <repo>/pkg/...
#   /api/models                       → list of locally-installed Ollama models
#   /api/blob/<family>:<tag>          → stream the model's GGUF blob (Range OK)
#   /api/log                          → append to /tmp/rullama-page.log
#
# Usage:
#   ./examples/web/serve-tunnel.sh
#
# Then point a Cloudflare tunnel at it:
#   cloudflared tunnel --url http://localhost:25321
#   (or: cloudflared tunnel --config <yaml> run <named-tunnel>)
#
# Env:
#   PORT             default 25321
#   OLLAMA_MODELS    default ~/.ollama/models
#   REPO_ROOT        default this repo

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

PORT="${PORT:-25321}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"

[[ -d examples/web/dist ]] || { echo "no examples/web/dist — run 'pnpm build' first" >&2; exit 1; }
[[ -d pkg ]] || { echo "no pkg/ — run wasm-pack build first" >&2; exit 1; }

if existing=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null); then
    if [ -n "$existing" ]; then
        echo "killing existing listener(s) on :$PORT (pids: $existing)"
        echo "$existing" | xargs kill -9 2>/dev/null || true
        sleep 0.3
    fi
fi

echo "rullama React PWA HTTP tunnel-origin server:"
echo "  local:        http://localhost:$PORT/"
echo "  expect tunnel: https://rullama.brainwires.net/  (Cloudflare)"
echo "  ollama:       $OLLAMA_MODELS"
echo "  page-log:     /tmp/rullama-page.log"
echo "  (Ctrl-C to stop)"

export PORT OLLAMA_MODELS REPO_ROOT="$ROOT"

exec python3 - <<'PY'
import http.server, socketserver, os, sys, urllib.parse, json, mimetypes
from pathlib import Path

PORT          = int(os.environ.get("PORT", "25321"))
REPO          = Path(os.environ["REPO_ROOT"]).resolve()
DIST          = REPO / "examples" / "web" / "dist"
PKG           = REPO / "pkg"
PAGE_LOG      = "/tmp/rullama-page.log"
OLLAMA_MODELS = Path(os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models")))
MANIFESTS     = OLLAMA_MODELS / "manifests"
BLOBS         = OLLAMA_MODELS / "blobs"
MODEL_LAYER   = "application/vnd.ollama.image.model"

mimetypes.add_type("application/wasm",          ".wasm")
mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("text/javascript",           ".js")
mimetypes.add_type("text/javascript",           ".mjs")
mimetypes.add_type("application/json",          ".map")

def discover_models():
    out = []
    if not MANIFESTS.is_dir():
        return out
    for path in MANIFESTS.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict) or "layers" not in data:
            continue
        rel = path.relative_to(MANIFESTS).parts
        if len(rel) < 2:
            continue
        family, tag = rel[-2], rel[-1]
        for layer in data.get("layers", []):
            if layer.get("mediaType") == MODEL_LAYER:
                digest = layer.get("digest", "")
                if digest.startswith("sha256:"):
                    digest = digest[len("sha256:"):]
                blob_path = BLOBS / f"sha256-{digest}"
                if blob_path.exists():
                    out.append({
                        "name":     f"{family}:{tag}",
                        "family":   family,
                        "tag":      tag,
                        "size":     int(layer.get("size", blob_path.stat().st_size)),
                        "digest":   digest,
                        "filename": f"sha256-{digest}",
                        "modelKey": f"{family}:{tag}",
                        # Text-only for the tunnel debug flow — skips
                        # multimodal scaffolding. Production keeps
                        # multimodal:true via BAKED_IN_MODELS.
                        "multimodal": False,
                    })
                break
    out.sort(key=lambda m: m["name"])
    return out

def find_blob(name_tag):
    for m in discover_models():
        if m["name"] == name_tag:
            return BLOBS / f"sha256-{m['digest']}"
    return None

class H(http.server.SimpleHTTPRequestHandler):
    # COOP/COEP for WebGPU + SharedArrayBuffer paths. The browser only
    # honours these when the page is served over HTTPS — which it IS
    # through the Cloudflare tunnel, so even though the origin here is
    # plain HTTP, the cross-origin-isolation context the PWA needs
    # still applies from the user's perspective.
    #
    # **CORS + CORP for the local-blob escape hatch.** When the PWA is
    # loaded from `https://rullama.brainwires.net` and `blobUrl()`
    # returns `http://localhost:25321/api/blob/...` (via `?localBlob=`),
    # the cross-origin fetch needs:
    #   - `Access-Control-Allow-Origin: <page origin>` (or *) so CORS
    #     passes
    #   - `Cross-Origin-Resource-Policy: cross-origin` so the
    #     require-corp context on the page accepts the response.
    # Both are set on every response. The CORP value is `cross-origin`
    # (not `same-origin`) when serving the blob endpoint specifically;
    # for HTML/JS we keep `same-origin` because those are loaded by the
    # Cloudflare path.
    def end_headers(self):
        path = getattr(self, "path", "") or ""
        # COOP/COEP — needed only on top-level documents but harmless
        # elsewhere.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # CORP — pick `cross-origin` for /api/blob so the cross-origin
        # PWA can consume it inside a require-corp context. Everything
        # else stays `same-origin`.
        if path.startswith("/api/blob/") or path.startswith("/api/models"):
            self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
            # CORS for the cross-origin fetch path. Echo the request's
            # Origin (if any) so credentials/preflight work; fall back
            # to * if no Origin header was sent.
            origin = self.headers.get("Origin", "*")
            self.send_header("Access-Control-Allow-Origin", origin)
            # The PWA's fetch sends Range + (occasionally) no auth headers.
            self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
            self.send_header("Access-Control-Expose-Headers",
                             "Content-Length, Content-Range, Accept-Ranges, "
                             "X-Model-Name, X-Total-Size")
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
            self.send_header("Vary", "Origin")
        else:
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if path.endswith((".js", ".html", ".wasm", ".d.ts", ".css", ".mjs")):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    # CORS preflight handler. Browsers send OPTIONS before the actual
    # GET when the fetch involves non-simple headers (Range qualifies).
    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        path = urllib.parse.unquote(path)
        if path.startswith("/pkg/"):
            return str(PKG / path[len("/pkg/"):])
        rel = path.lstrip("/")
        if not rel:
            return str(DIST / "index.html")
        return str(DIST / rel)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/models":
            return self._serve_models()
        if u.path.startswith("/api/blob/"):
            model = urllib.parse.unquote(u.path[len("/api/blob/"):])
            return self._serve_blob(model)
        # SPA fallback.
        local = Path(self.translate_path(self.path))
        if (not local.exists()
                and not u.path.startswith("/pkg/")
                and not u.path.startswith("/api/")
                and "." not in u.path.rsplit("/", 1)[-1]):
            self.path = "/"
        return super().do_GET()

    def do_HEAD(self):
        # Some tooling probes /api/blob with HEAD. Default
        # SimpleHTTPRequestHandler returns 404 here for non-file paths;
        # route HEAD identically to GET so health/probe tools work.
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/models" or u.path.startswith("/api/blob/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream"
                             if u.path.startswith("/api/blob/") else "application/json")
            self.end_headers()
            return
        return super().do_HEAD()

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/log":
            return self._recv_log()
        self.send_error(404, "no such endpoint")

    def _serve_models(self):
        body = json.dumps(discover_models()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_blob(self, name_tag):
        blob = find_blob(name_tag)
        if blob is None:
            self.send_error(404, f"model not found: {name_tag}")
            return
        size = blob.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if rng and rng.startswith("bytes="):
            try:
                spec = rng[len("bytes="):]
                if "-" in spec:
                    a, b = spec.split("-", 1)
                    start = int(a) if a else 0
                    end = int(b) if b else size - 1
                    end = min(end, size - 1)
                    if 0 <= start <= end < size:
                        status = 206
            except ValueError:
                pass
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Model-Name", name_tag)
        self.send_header("X-Total-Size", str(size))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        CHUNK = 1 << 20
        try:
            with open(blob, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)
        except Exception as e:
            sys.stderr.write(f"[serve-tunnel] blob stream error: {e}\n")

    def _recv_log(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n) if n > 0 else b"{}"
            d = json.loads(body)
            tag = d.get("tag", "?")
            msg = d.get("msg", "")
            with open(PAGE_LOG, "a", encoding="utf-8") as f:
                f.write(f"[{tag}] {msg}\n")
            self.send_response(204)
            self.end_headers()
        except Exception as e:
            self.send_error(500, f"log fail: {e}")

    def log_message(self, fmt, *args):
        # Quiet by default. Set RULLAMA_SERVE_VERBOSE=1 to see every hit.
        if os.environ.get("RULLAMA_SERVE_VERBOSE") == "1":
            sys.stderr.write("[serve-tunnel] %s - %s\n" % (
                self.address_string(), fmt % args))

class TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    # Bind to 127.0.0.1 only — Cloudflare tunnel talks to localhost.
    # No external port exposure required.
    def __init__(self, addr, handler):
        super().__init__(addr, handler)

httpd = TCPServer(("127.0.0.1", PORT), H)
print(f"[serve-tunnel] listening 127.0.0.1:{PORT} http; dist={DIST}; pkg={PKG}; ollama={OLLAMA_MODELS}", file=sys.stderr)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
PY
