#!/usr/bin/env bash
# train-on-iphone.sh — drive the React PWA on the connected iPhone for
# training-crash repros, capture the post-crash OPFS log back to the Mac.
#
# Workflow:
#   1. Create (or reuse) a safaridriver session, navigate Safari to the
#      React PWA's URL with ?automation=1.
#   2. Wait for `window.__rullama` to be ready (wasm + worker attached).
#   3. PROMPT — you tap "Start training" on the iPhone manually. We
#      can't reliably click React buttons via safaridriver since the
#      PWA has minimal stable selectors; tapping is a 2-second job.
#   4. Poll for crash: detect when the worker disconnects (the Safari
#      tab dies on iOS jetsam and the page reloads OR the executeScript
#      starts returning errors).
#   5. After the crash, reload the page and dump the OPFS log via
#      `window.__rullama.dumpLogs()`. Save it to /tmp/rullama-iphone-trn.log
#      and print the tail.
#
# Prerequisites (one-time):
#   - iPhone tethered (USB OR same wifi/hotspot as Mac)
#   - iPhone screen unlocked, auto-lock long or off
#   - iPhone: Settings → Safari → Advanced → Remote Automation ON
#   - safaridriver allowed on first connection ("Allow Remote Automation")
#   - HTTPS cert trusted on the iPhone (Settings → General → About →
#     Certificate Trust Settings → enable for rullama cert)
#   - examples/web/serve-iphone.sh running on the Mac (port 8088)
#   - safaridriver running on the Mac (port 4444)
#
# Usage:
#   ./examples/web/train-on-iphone.sh
#
# Env overrides:
#   PORT       (default 8088) — the HTTPS server port
#   WD_PORT    (default 4444) — safaridriver port
#   SID_FILE   (default /tmp/rullama-iphone-session-id) — keeper compat
#   MAC_IP                    — auto-detected from en0..en5
#   TIMEOUT_CRASH_SEC (default 600) — how long to wait for the crash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${PORT:-8088}"
WD_PORT="${WD_PORT:-4444}"
SID_FILE="${SID_FILE:-/tmp/rullama-iphone-session-id}"
TIMEOUT_CRASH_SEC="${TIMEOUT_CRASH_SEC:-600}"
LOG_OUT="${LOG_OUT:-/tmp/rullama-iphone-trn.log}"

log() { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*"; }
err() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; }

# ── Preflight ────────────────────────────────────────────────────────
if ! curl -ksS --max-time 2 "https://localhost:$PORT/" -o /dev/null 2>&1; then
    err "no HTTPS server on :${PORT}."
    err "  Start with: REPO_ROOT=\"$ROOT\" $ROOT/examples/web/serve-iphone.sh &"
    exit 1
fi
if ! curl -sS --max-time 2 "http://localhost:${WD_PORT}/status" >/dev/null 2>&1; then
    err "no safaridriver on :${WD_PORT}."
    err "  Start with: safaridriver -p ${WD_PORT} &"
    exit 1
fi
UDID="${UDID:-$(idevice_id -l 2>/dev/null | head -1)}"
[[ -z "$UDID" ]] && { err "no iPhone connected"; exit 1; }
log "iPhone $UDID"

if [[ -z "${MAC_IP:-}" ]]; then
    for iface in en0 en1 en2 en3 en4 en5; do
        ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
        if [[ -n "$ip" && "$ip" != 169.254.* ]]; then MAC_IP="$ip"; break; fi
    done
fi
[[ -z "$MAC_IP" ]] && { err "could not detect Mac IP"; exit 1; }
URL="https://$MAC_IP:$PORT/?automation=1"
log "URL: $URL"

# ── Session: prefer existing keeper session, else create fresh ──────
SESSION=""
SESSION_IS_OURS=0
if [[ -f "$SID_FILE" ]]; then
    cand=$(cat "$SID_FILE")
    if curl -sS --max-time 5 "http://localhost:${WD_PORT}/session/${cand}/url" 2>/dev/null \
        | grep -qv '"error":"invalid session id"'; then
        SESSION="$cand"
        log "reusing keeper session $SESSION (OPFS persists)"
    else
        rm -f "$SID_FILE"
    fi
fi
if [[ -z "$SESSION" ]]; then
    SESSION=$(curl -sS -X POST -H "Content-Type: application/json" \
        -d "{\"capabilities\":{\"alwaysMatch\":{\"platformName\":\"iOS\",\"safari:deviceType\":\"iPhone\",\"safari:deviceUDID\":\"$UDID\",\"safari:useSimulator\":false,\"acceptInsecureCerts\":true}}}" \
        "http://localhost:${WD_PORT}/session" \
        | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["value"]["sessionId"])')
    [[ -z "$SESSION" ]] && { err "session create failed"; exit 1; }
    SESSION_IS_OURS=1
    echo "$SESSION" > "$SID_FILE"
    log "created session $SESSION"
    warn "OPFS is now scoped to this safaridriver session."
    warn "If the iPhone's model cache was in Safari's main storage,"
    warn "the PWA will need to re-download (~7 GB)."
fi

# Helper: run JS in page via executeScript/sync.
run_js() {
    local script="$1" timeout="${2:-30}"
    local resp
    resp=$(curl -sS --max-time "$timeout" -X POST -H "Content-Type: application/json" \
        --data-raw "{\"script\":${script}, \"args\":[]}" \
        "http://localhost:${WD_PORT}/session/${SESSION}/execute/sync" 2>/dev/null \
        || echo '{"value":null,"_curl_err":1}')
    if echo "$resp" | grep -q '"error":"invalid session id"'; then
        echo '"INVALID_SESSION"'; return
    fi
    if echo "$resp" | grep -q '"error":'; then
        # Bubble up the error message so the caller can detect crashes.
        echo "$resp" | python3 -c 'import sys,json
try:
  d=json.loads(sys.stdin.read())
  e=d.get("value",{}) or {}
  print(json.dumps({"_error": e.get("error","unknown"), "_msg": e.get("message","")}))
except Exception:
  print("null")'
        return
    fi
    echo "$resp" | python3 -c 'import sys,json
try:
  d=json.loads(sys.stdin.read())
  print(json.dumps(d.get("value")))
except Exception:
  print("null")' 2>/dev/null || echo "null"
}
js_str() { python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'; }

# ── Navigate ────────────────────────────────────────────────────────
log "navigating Safari → $URL"
curl -sS --max-time 30 -X POST -H "Content-Type: application/json" \
    -d "{\"url\":\"$URL\"}" "http://localhost:${WD_PORT}/session/${SESSION}/url" >/dev/null

# ── Wait for __rullama.ready (proves React mounted + worker attached) ─
log "waiting for window.__rullama.ready …"
deadline=$(( $(date +%s) + 60 ))
ready=""
while (( $(date +%s) < deadline )); do
    out=$(run_js '"return Boolean(window.__rullama);"' 5)
    if [[ "$out" == "true" ]]; then
        ready=1
        break
    fi
    sleep 1
done
[[ -z "$ready" ]] && { err "__rullama did not become available within 60s"; exit 3; }
log "__rullama present"

# Wait for the worker to be attached (logger has a session id).
deadline=$(( $(date +%s) + 30 ))
while (( $(date +%s) < deadline )); do
    sid=$(run_js '"return window.__rullama.client.logs.currentId().then(x=>x).catch(()=>null);"' 5 | sed 's/"//g')
    if [[ -n "$sid" && "$sid" != "null" ]]; then
        log "worker session id: $sid"
        break
    fi
    sleep 1
done

# ── Hand off to user for manual model-load + start-training tap ──────
cat <<EOF

────────────────────────────────────────────────────────────────────────
ON THE iPHONE NOW:
  1. If the model isn't auto-loaded, tap to load gemma4:e2b (or your
     cached model).
  2. Tap the "Fine-tune" tab.
  3. Confirm "Memory-tight (iPhone-safe) preset" is ON.
  4. Tap "Start training".

This script begins polling now. As soon as the first [trn] beacon
fires, the live timeline appears below. On crash:
  - The PWA is reloaded
  - Post-crash OPFS log saved to:  ${LOG_OUT}
  - Tail printed here.

Total budget: ${TIMEOUT_CRASH_SEC}s.
────────────────────────────────────────────────────────────────────────
EOF

# ── Poll for crash ──────────────────────────────────────────────────
log "polling for crash (timeout ${TIMEOUT_CRASH_SEC}s)…"
start=$(date +%s)
last_step=""
while (( $(date +%s) - start < TIMEOUT_CRASH_SEC )); do
    # Probe via a tiny harmless JS call. If the tab crashed, safaridriver
    # returns an error and the URL changes.
    probe=$(run_js '"return Boolean(window.__rullama);"' 8)
    if [[ "$probe" == "true" ]]; then
        # Still alive. Peek at the latest beacon for progress feedback.
        cur=$(run_js '"return window.__rullama.dumpLogs().then(t=>t.split(\"\\n\").filter(l=>l.includes(\"[trn]\")).slice(-1).join(\"\"));"' 8)
        if [[ -n "$cur" && "$cur" != "$last_step" && "$cur" != "null" && "$cur" != "\"\"" ]]; then
            printf '  [%4ds] %s\n' "$(( $(date +%s) - start ))" "$cur"
            last_step="$cur"
        fi
        sleep 3
        continue
    fi
    # Either we got an _error or the response was malformed — likely crash.
    if echo "$probe" | grep -qE '_error|INVALID_SESSION'; then
        log "💥 worker / tab died — looks like the crash"
        break
    fi
    # Unknown — be tolerant.
    sleep 3
done

# ── Reload + dump log ───────────────────────────────────────────────
log "reloading PWA to read OPFS log…"
curl -sS --max-time 30 -X POST -H "Content-Type: application/json" \
    -d "{\"url\":\"$URL\"}" "http://localhost:${WD_PORT}/session/${SESSION}/url" >/dev/null

# Wait for __rullama again
deadline=$(( $(date +%s) + 60 ))
while (( $(date +%s) < deadline )); do
    out=$(run_js '"return Boolean(window.__rullama);"' 5)
    [[ "$out" == "true" ]] && break
    sleep 1
done

log "looking for the crashed session…"
crashed_id=$(run_js '"return window.__rullama.crashedId();"' 8 | sed 's/"//g')
if [[ -z "$crashed_id" || "$crashed_id" == "null" ]]; then
    warn "no crashed session detected via cleanExit=false flag — falling back to most-recent prior session"
    crashed_id=$(run_js '"return window.__rullama.client.logs.list().then(l=>l[1]?.id||null);"' 8 | sed 's/"//g')
fi
log "crashed session id: ${crashed_id:-(unknown)}"

if [[ -n "$crashed_id" && "$crashed_id" != "null" ]]; then
    # Fetch the log content — safaridriver caps response sizes so we read
    # in chunks if it's large. With our 512 KiB cap per session it should
    # fit in one call.
    log_text=$(run_js "\"return window.__rullama.client.logs.read('${crashed_id}');\"" 30)
    # log_text is a JSON-encoded string; strip the outer quotes.
    printf '%s\n' "$log_text" | python3 -c 'import sys,json
try:
  v = json.loads(sys.stdin.read())
  print(v if isinstance(v,str) else json.dumps(v))
except Exception as e:
  sys.stderr.write(f"parse fail: {e}\n")
' > "$LOG_OUT"
    log "wrote $(wc -l < "$LOG_OUT") lines → $LOG_OUT"
    echo
    echo "── tail (last 30 lines) ────────────────────────────────────────"
    tail -30 "$LOG_OUT"
    echo "────────────────────────────────────────────────────────────────"
else
    err "could not retrieve a crashed session log"
fi

# ── Cleanup ─────────────────────────────────────────────────────────
if (( SESSION_IS_OURS )); then
    log "deleting session $SESSION"
    curl -sS -X DELETE "http://localhost:${WD_PORT}/session/${SESSION}" >/dev/null 2>&1 || true
    rm -f "$SID_FILE"
else
    log "leaving keeper session alive"
fi
