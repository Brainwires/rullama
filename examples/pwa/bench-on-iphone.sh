#!/usr/bin/env bash
# bench-on-iphone.sh — drive the rullama WebGPU bench from this Mac onto the
# connected iPhone over USB. Pure-bash, only macOS-built-in tools: safaridriver
# (Apple's W3C WebDriver) for navigation + curl for HTTP, no Python, no Xcode.
#
# Flow:
#   1. Start (or reuse) serve.sh on this Mac bound to 0.0.0.0 — phone reaches
#      it via the Mac's LAN IP.
#   2. Truncate /tmp/rullama-bench.jsonl (the server appends posted results
#      here, one event per line).
#   3. Start safaridriver REST server, POST a session targeting the iPhone
#      (platformName=iOS, safari:deviceUDID=<udid>), POST the bench URL to
#      /session/<id>/url to navigate.
#   4. Tail the result log until {"event":"done"} or timeout.
#   5. DELETE the WebDriver session to close the Safari automation tab.
#
# Prereqs (one-time):
#   - iPhone paired & trusted with this Mac.
#   - Safari → Advanced → Web Inspector = ON.
#   - Safari → Advanced → Remote Automation = ON.
#   - (Optional) `sudo safaridriver --enable` if you want to suppress the
#     first-launch authentication prompt; not required here.
#
# Usage:
#   ./bench-on-iphone.sh                      # auto-detect Mac IP + first device
#   MAC_IP=10.42.0.194 ./bench-on-iphone.sh   # override IP
#   UDID=00008140-... ./bench-on-iphone.sh    # pin to a specific phone
#   PORT=8088 ./bench-on-iphone.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SERVE_SH="$ROOT/examples/pwa/serve.sh"
PORT="${PORT:-8088}"
WD_PORT="${WD_PORT:-4444}"
BENCH_LOG="${BENCH_LOG:-/tmp/rullama-bench.jsonl}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
# HTTPS is required for iOS Safari WebGPU (secure-context gate).
CERT_DIR="${CERT_DIR:-$HOME/.local/share/rullama}"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

log() { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
err() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; }

# ---- Discover device UDID (libimobiledevice format) ----
if [[ -z "${UDID:-}" ]]; then
    UDID="$(idevice_id -l 2>/dev/null | head -1 || true)"
fi
if [[ -z "${UDID:-}" ]]; then
    err "no iPhone found — plug in via USB and trust this Mac"
    exit 1
fi
log "device UDID = $UDID"

# ---- Discover Mac LAN IP ----
if [[ -z "${MAC_IP:-}" ]]; then
    for iface in en0 en1 en2 en3 en4 en5 en6 en7 en8 en9; do
        ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
        if [[ -n "$ip" && "$ip" != 169.254.* ]]; then
            MAC_IP="$ip"; break
        fi
    done
fi
if [[ -z "${MAC_IP:-}" ]]; then
    err "could not auto-detect Mac LAN IP; set MAC_IP=<ip> and retry"
    exit 1
fi
log "Mac LAN IP = $MAC_IP"

URL="https://$MAC_IP:$PORT/examples/pwa/bench.html"
log "phone target URL = $URL"

# ---- Ensure cert exists (HTTPS required for WebGPU on iOS Safari) ----
if [[ ! -f "$CERT_FILE" || ! -f "$KEY_FILE" ]]; then
    log "generating self-signed cert at ${CERT_DIR}…"
    mkdir -p "$CERT_DIR"
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$KEY_FILE" -out "$CERT_FILE" -days 365 \
        -subj "/CN=rullama-dev" \
        -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:$MAC_IP" \
        >/dev/null 2>&1
fi

# ---- Start (or reuse) serve.sh, **HTTPS** ----
SERVE_PID=""
# If a server is up on the port, check whether it's actually our HTTPS server.
# A leftover plain-HTTP server from an earlier session won't satisfy WebGPU's
# secure-context requirement.
need_restart=0
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    if ! curl -ksS --max-time 2 "https://localhost:$PORT/" -o /dev/null 2>/dev/null; then
        log "non-HTTPS server detected on :$PORT — killing"
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t | xargs kill 2>/dev/null || true
        sleep 0.3
        need_restart=1
    else
        log "HTTPS server already listening on :$PORT — reusing"
    fi
else
    need_restart=1
fi
if (( need_restart )); then
    log "starting HTTPS serve.sh on :${PORT}…"
    CERT_FILE="$CERT_FILE" KEY_FILE="$KEY_FILE" \
    BENCH_LOG="$BENCH_LOG" PORT="$PORT" "$SERVE_SH" \
        >/tmp/rullama-serve.log 2>&1 &
    SERVE_PID=$!
    sleep 1
    if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
        err "server failed to start; see /tmp/rullama-serve.log"
        head -20 /tmp/rullama-serve.log >&2 || true
        exit 1
    fi
fi

: > "$BENCH_LOG"
log "result log = $BENCH_LOG (truncated)"

# ---- Start safaridriver ----
SAFD_PID=""
if lsof -nP -iTCP:"$WD_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    log "safaridriver already on :$WD_PORT — reusing"
else
    log "starting safaridriver on :${WD_PORT}…"
    safaridriver -p "$WD_PORT" >/tmp/safaridriver.log 2>&1 &
    SAFD_PID=$!
    sleep 1
    if ! curl -sS --max-time 3 "http://localhost:$WD_PORT/status" \
            | grep -q '"ready":true'; then
        err "safaridriver did not become ready; see /tmp/safaridriver.log"
        head -20 /tmp/safaridriver.log >&2 || true
        [[ -n "$SAFD_PID" ]] && kill "$SAFD_PID" 2>/dev/null || true
        exit 1
    fi
fi

# ---- Create iOS WebDriver session ----
log "requesting WebDriver session on iPhone…"
SESSION_RESP=$(curl -sS --max-time 30 -X POST \
    -H "Content-Type: application/json" \
    -d "{
        \"capabilities\": {
            \"alwaysMatch\": {
                \"platformName\": \"iOS\",
                \"safari:deviceType\": \"iPhone\",
                \"safari:deviceUDID\": \"$UDID\",
                \"safari:useSimulator\": false,
                \"acceptInsecureCerts\": true
            }
        }
    }" \
    "http://localhost:$WD_PORT/session" 2>&1)
SESSION_ID=$(echo "$SESSION_RESP" | python3 -c '
import sys, json
try:
    d = json.loads(sys.stdin.read())
    print(d.get("value", {}).get("sessionId", ""))
except Exception:
    pass')
if [[ -z "$SESSION_ID" ]]; then
    err "WebDriver session create failed:"
    echo "$SESSION_RESP" >&2 | head -30
    err "Confirm: Settings → Safari → Advanced → Remote Automation = ON"
    exit 1
fi
log "session = $SESSION_ID"

# Cleanup: tear down session + servers on exit. Done in the right order so
# the Safari Automation tab closes cleanly.
cleanup() {
    [[ -n "$SESSION_ID" ]] && curl -sS --max-time 5 -X DELETE \
        "http://localhost:$WD_PORT/session/$SESSION_ID" >/dev/null 2>&1 || true
    [[ -n "$SAFD_PID" ]]   && kill "$SAFD_PID"   2>/dev/null || true
    [[ -n "$SERVE_PID" ]]  && kill "$SERVE_PID"  2>/dev/null || true
}
trap cleanup EXIT

# ---- Navigate to bench URL ----
log "navigating Safari on phone → $URL"
NAV_RESP=$(curl -sS --max-time 30 -X POST \
    -H "Content-Type: application/json" \
    -d "{\"url\": \"$URL\"}" \
    "http://localhost:$WD_PORT/session/$SESSION_ID/url" 2>&1)
if echo "$NAV_RESP" | grep -q '"error"'; then
    err "navigation failed:"
    echo "$NAV_RESP" >&2
    exit 1
fi

# ---- Poll for results ----
log "waiting up to ${TIMEOUT_SEC}s for bench results…"
deadline=$(( $(date +%s) + TIMEOUT_SEC ))
last_lines=0
# A 'done' inside the inner while-read can't `exit` the whole script (subshell).
# Use a sentinel file instead and check it from the outer loop.
DONE_FLAG=$(mktemp -t rullama-bench-done.XXXXXX)
rm -f "$DONE_FLAG"
while [[ $(date +%s) -lt $deadline ]]; do
    if [[ -f "$DONE_FLAG" ]]; then
        rm -f "$DONE_FLAG"
        exit 0
    fi
    if [[ -s "$BENCH_LOG" ]]; then
        cur=$(wc -l < "$BENCH_LOG" | tr -d ' ')
        if (( cur > last_lines )); then
            tail -n "$(( cur - last_lines ))" "$BENCH_LOG" | while IFS= read -r line; do
                event=$(printf '%s' "$line" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read()).get("event",""))' 2>/dev/null || true)
                case "$event" in
                    env)
                        printf '%s\n' "$line" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())['env']
print('  origin: %s' % d.get('origin'))
print('  isSecureContext: %s' % d.get('isSecureContext'))
print('  crossOriginIsolated: %s' % d.get('crossOriginIsolated'))
print('  has_navigator_gpu: %s' % d.get('has_navigator_gpu'))
"
                        ;;
                    adapter)
                        printf '%s\n' "$line" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())['info']
arch = d.get('architecture') or d.get('description') or d.get('device')
print('  iPhone GPU: %s / %s' % (d.get('vendor'), arch))
print('  features: %s' % ', '.join(d.get('features', [])))
print('  has_f16 = %s' % d.get('has_f16'))
print('  ua: %s...' % d.get('ua', '')[:90])
"
                        ;;
                    matmul)
                        printf '%s\n' "$line" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
s = d['shape']
print('  %s k=%4d n=%4d batch=%4d | %7.2f ms/iter   %6.1f GFLOPS' % (
    s['label'], s['k'], s['n'], s['batch'], d['ms_per_iter'], d['gflops']))
"
                        ;;
                    attn)
                        printf '%s\n' "$line" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print('  vision_attention (%s): n_patches=%d, n_heads=%d, head_dim=%d | %.2f ms/iter  (x16 layers = %.0f ms)' % (
    d['variant'], d['n_patches'], d['n_heads'], d['head_dim'], d['ms_per_iter'], d['total_16_layers_ms']))
"
                        ;;
                    done)
                        echo
                        log "bench complete ✓"
                        : > "$DONE_FLAG"
                        ;;
                    no_webgpu|no_adapter)
                        err "WebGPU not enabled on phone: event=$event"
                        err "  Safari → Advanced → Feature Flags → WebGPU = ON"
                        exit 2
                        ;;
                    shader_error|fatal)
                        err "phone reported $event:"
                        echo "$line" >&2
                        exit 3
                        ;;
                esac
            done
            last_lines=$cur
        fi
    fi
    sleep 0.5
done

err "timed out after ${TIMEOUT_SEC}s — last $last_lines log line(s) shown above"
exit 4
