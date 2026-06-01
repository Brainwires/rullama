#!/usr/bin/env bash
# serve-iphone.sh — HTTPS server for the React PWA + iPhone safaridriver tests.
#
# Serves:
#   /                                 → examples/web/dist/index.html
#   /assets/*, /icons/*, /sw.js, …    → examples/web/dist/...
#   /pkg/*                            → <repo>/pkg/...
#   /api/models                       → list of locally-installed Ollama models
#   /api/blob/<family>:<tag>          → stream the model's GGUF blob (Range OK)
#   /api/log                          → append to /tmp/rullama-page.log
#
# Usage:
#   CERT_FILE=~/.local/share/rullama/cert.pem \
#   KEY_FILE=~/.local/share/rullama/key.pem  \
#   REPO_ROOT="$(pwd)" \
#   ./examples/web/serve-iphone.sh

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

PORT="${PORT:-8088}"
CERT_FILE="${CERT_FILE:-$HOME/.local/share/rullama/cert.pem}"
KEY_FILE="${KEY_FILE:-$HOME/.local/share/rullama/key.pem}"
OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"

[[ -f "$CERT_FILE" ]] || { echo "missing cert: $CERT_FILE" >&2; exit 1; }
[[ -f "$KEY_FILE"  ]] || { echo "missing key:  $KEY_FILE"  >&2; exit 1; }
[[ -d examples/web/dist ]] || { echo "no examples/web/dist — run 'pnpm build' first" >&2; exit 1; }
[[ -d pkg ]] || { echo "no pkg/ — run wasm-pack build first" >&2; exit 1; }

if existing=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null); then
    if [ -n "$existing" ]; then
        echo "killing existing listener(s) on :$PORT (pids: $existing)"
        echo "$existing" | xargs kill -9 2>/dev/null || true
        sleep 0.3
    fi
fi

# Mac IP for the iPhone-reachable URL.
MAC_IP=""
for iface in en0 en1 en2 en3 en4 en5; do
    ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
    if [[ -n "$ip" && "$ip" != 169.254.* ]]; then MAC_IP="$ip"; break; fi
done

echo "rullama React PWA HTTPS server:"
echo "  local:   https://localhost:${PORT}/"
[[ -n "$MAC_IP" ]] && echo "  iPhone:  https://${MAC_IP}:${PORT}/"
echo "  ollama:  ${OLLAMA_MODELS}"
echo "  page-log: /tmp/rullama-page.log"
echo "  (Ctrl-C to stop)"
echo

export PORT CERT_FILE KEY_FILE OLLAMA_MODELS REPO_ROOT="$ROOT"

exec python3 - <<'PY'
import http.server, socketserver, os, sys, ssl, urllib.parse, json, mimetypes
from pathlib import Path

PORT          = int(os.environ.get("PORT", "8088"))
CERT_FILE     = os.environ["CERT_FILE"]
KEY_FILE      = os.environ["KEY_FILE"]
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
                        # The React PWA's model loader expects these
                        # fields too; supply best-effort defaults.
                        "filename": f"sha256-{digest}",
                        "modelKey": f"{family}:{tag}",
                        # multimodal:false → App.tsx computes textOnly=true →
                        # loadFromOpfsTextOnly (skips vision/audio towers +
                        # their scaffolding, max_ctx 512). For iPhone training
                        # this cuts the baseline memory that the multimodal
                        # load wastes (text fine-tuning never needs the
                        # vision/audio towers). DEV/TEST ONLY — production's
                        # BAKED_IN_MODELS keeps multimodal:true.
                        "multimodal": False,
                    })
                    # Serve from the local Mac over the bridged 10.42.0.x
                    # LAN (fast, no CDN rate limits). No `url` field →
                    # blobUrl(m) in api.ts falls back to /api/blob, which
                    # this server streams from ~/.ollama/models/blobs.
                break
    out.sort(key=lambda m: m["name"])
    return out

def find_blob(name_tag):
    for m in discover_models():
        if m["name"] == name_tag:
            return BLOBS / f"sha256-{m['digest']}"
    return None

class H(http.server.SimpleHTTPRequestHandler):
    # COOP/COEP for WebGPU + SharedArrayBuffer paths.
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        # Dev mode: don't let Safari cache the bundle.
        path = getattr(self, "path", "") or ""
        if path.endswith((".js", ".html", ".wasm", ".d.ts", ".css", ".mjs")):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    # Path mapping: /pkg/* → repo/pkg/, everything else → dist/
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
        if u.path == "/cert":
            # Serve the CA cert with the iOS profile-install MIME type.
            # Served over HTTPS here so iOS Safari's http→https auto-
            # upgrade doesn't break the download (the plain-HTTP cert
            # server on :8089 hits that wall).
            try:
                with open(CERT_FILE, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-x509-ca-cert")
                self.send_header("Content-Disposition", 'attachment; filename="rullama-dev.pem"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_error(500, f"cert read fail: {e}")
            return
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
            sys.stderr.write(f"[serve-iphone] blob stream error: {e}\n")

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
        # Skip noisy default — keep only errors via stderr default.
        return

class TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

httpd = TCPServer(("0.0.0.0", PORT), H)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
print(f"[serve-iphone] listening :{PORT} https; dist={DIST}; pkg={PKG}; ollama={OLLAMA_MODELS}", file=sys.stderr)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
PY
