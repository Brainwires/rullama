#!/usr/bin/env bash
# run-on-iphone.sh — drive the full rullama PWA on the connected iPhone.
# Loads a model, runs a text prompt, optionally an image prompt, prints
# timings to stdout.
#
# Companion to bench-on-iphone.sh; this one exercises the actual wasm +
# WebGPU end-to-end path instead of just the kernel benches.
#
# Usage:
#   ./run-on-iphone.sh                            # load default model, text-only
#   MODEL=gemma4:e2b PROMPT="Hi" ./run-on-iphone.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${PORT:-8088}"
WD_PORT="${WD_PORT:-4444}"
MODEL="${MODEL:-gemma4:e2b}"
PROMPT="${PROMPT:-Hi}"
MAX_TOK="${MAX_TOK:-32}"
TIMEOUT_LOAD_SEC="${TIMEOUT_LOAD_SEC:-600}"   # 10 min — first time has 7 GB download
TIMEOUT_GEN_SEC="${TIMEOUT_GEN_SEC:-240}"

log() { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
err() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; }

if ! curl -ksS --max-time 2 "https://localhost:$PORT/" -o /dev/null 2>&1; then
    err "no HTTPS server on :${PORT}. Start with:"
    err "  CERT_FILE=~/.local/share/rullama/cert.pem KEY_FILE=~/.local/share/rullama/key.pem nohup ./examples/pwa/serve.sh &"
    exit 1
fi
if ! curl -sS --max-time 1 "http://localhost:${WD_PORT}/status" >/dev/null 2>&1; then
    err "no safaridriver on :${WD_PORT}. Start with: safaridriver -p ${WD_PORT} &"
    exit 1
fi

# Pre-flight: refuse to run if the iPhone is critically low on disk.
# WebDriver sessions get isolated OPFS profiles that are NOT cleaned up by
# `DELETE /session/<id>` on iOS — they orphan in a sandbox we can't reach
# from script. Once those eat the phone's data partition, OPFS writes start
# failing at weird small offsets ("Failed to write to file" at <100 MB).
# Refusing to start a 7 GB download in that state saves the user from
# accumulating more orphan profile data.
UDID="${UDID:-$(idevice_id -l 2>/dev/null | head -1)}"
[[ -z "$UDID" ]] && { err "no iPhone connected"; exit 1; }
# `ideviceinfo -k AmountDataAvailable` returns empty on some firmware paths;
# `-q com.apple.disk_usage` reliably emits the same key.
free_bytes=$(ideviceinfo -u "$UDID" -q com.apple.disk_usage 2>/dev/null \
    | awk -F': ' '$1=="AmountDataAvailable"{print $2; exit}')
free_bytes="${free_bytes:-0}"
free_gb=$(( free_bytes / 1024 / 1024 / 1024 ))
MIN_FREE_GB="${MIN_FREE_GB:-8}"
if (( free_bytes == 0 )) || (( free_gb < MIN_FREE_GB )); then
    err "iPhone data partition has only ${free_gb} GB free (< ${MIN_FREE_GB} GB required)."
    err "Free space via: Settings → Safari → Clear History and Website Data,"
    err "                or Settings → General → iPhone Storage → review largest apps."
    err "Override at your own risk with: MIN_FREE_GB=0 $(basename "$0")"
    exit 1
fi
log "iPhone has ${free_gb} GB free — ok"

# Auto-detect Mac IP same way bench-on-iphone.sh does (UDID was set above).
if [[ -z "${MAC_IP:-}" ]]; then
    for iface in en0 en1 en2 en3 en4 en5; do
        ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
        if [[ -n "$ip" && "$ip" != 169.254.* ]]; then MAC_IP="$ip"; break; fi
    done
fi
URL="https://$MAC_IP:$PORT/examples/pwa/index.html"
log "device $UDID → $URL"
log "model='${MODEL}' prompt='${PROMPT}' max_tok=${MAX_TOK}"

# ---- Session ----
SESSION=$(curl -sS -X POST -H "Content-Type: application/json" \
    -d "{\"capabilities\":{\"alwaysMatch\":{\"platformName\":\"iOS\",\"safari:deviceType\":\"iPhone\",\"safari:deviceUDID\":\"$UDID\",\"safari:useSimulator\":false,\"acceptInsecureCerts\":true}}}" \
    "http://localhost:${WD_PORT}/session" \
    | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["value"]["sessionId"])')
[[ -z "$SESSION" ]] && { err "session create failed"; exit 1; }
# Note: do NOT auto-delete the WebDriver session on exit. When the page
# crashes mid-generation the session-tear-down closes the tab and we lose
# all visibility into what the page state actually was. Leaving the
# session live lets the user inspect the phone screen for the real error
# (chat-log line, status pill, any "page reloaded due to memory issue"
# banner). Run `./examples/pwa/clean-iphone.sh` between runs to wipe OPFS
# orphans, or `curl -X DELETE` the session manually when done.
log "session $SESSION will be left alive on the phone — clean up with: curl -X DELETE http://localhost:${WD_PORT}/session/${SESSION}"
log "session = $SESSION"

# Helper: run JS in page, return its .value (JSON-decoded).
# Robust to slow responses while the page is busy (model streaming): the
# default timeout is 30 s, callers can override. If the session got
# invalidated we surface "INVALID_SESSION" so callers can decide.
run_js() {
    local script="$1"
    local timeout="${2:-30}"
    local resp
    resp=$(curl -sS --max-time "$timeout" -X POST -H "Content-Type: application/json" \
        --data-raw "{\"script\":${script}, \"args\":[]}" \
        "http://localhost:${WD_PORT}/session/${SESSION}/execute/sync" 2>/dev/null || echo '{"value":null,"_curl_err":1}')
    # Detect invalid-session response from safaridriver.
    if echo "$resp" | grep -q '"error":"invalid session id"'; then
        echo '"INVALID_SESSION"'
        return
    fi
    echo "$resp" | python3 -c 'import sys,json
try:
  d=json.loads(sys.stdin.read())
  print(json.dumps(d.get("value")))
except Exception:
  print("null")' 2>/dev/null || echo "null"
}

# Wrap a JS string for safe JSON embedding.
js_str() { python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' ; }

# ---- Navigate ----
log "navigating Safari → $URL"
curl -sS --max-time 30 -X POST -H "Content-Type: application/json" \
    -d "{\"url\":\"$URL\"}" "http://localhost:${WD_PORT}/session/${SESSION}/url" >/dev/null

# Wait for WASM bundle load + model dropdown population.
log "waiting for wasm bundle to load + model list to populate…"
deadline=$(( $(date +%s) + 30 ))
while (( $(date +%s) < deadline )); do
    state=$(run_js '"return JSON.stringify({wasm:document.getElementById(\"wasm-status\").textContent,opts:Array.from(document.getElementById(\"model-select\").options).map(o=>o.value).filter(v=>v)});"' 5 \
        | python3 -c 'import sys,json; d=json.loads(json.loads(sys.stdin.read())); print(d["wasm"]+"|"+",".join(d["opts"]))' 2>/dev/null || echo "|")
    wasm="${state%%|*}"; opts="${state#*|}"
    if [[ "$wasm" == *"loaded"* && -n "$opts" ]]; then
        log "wasm: $wasm, available: $opts"
        break
    fi
    sleep 0.5
done
echo "$opts" | tr ',' '\n' | grep -qx "$MODEL" || {
    err "model '$MODEL' not in dropdown (available: $opts)"
    exit 2
}

# ---- Select + click Load ----
log "selecting model = $MODEL"
run_js "\"const s=document.getElementById('model-select'); s.value='$MODEL'; s.dispatchEvent(new Event('change'));return 'ok';\"" 5 >/dev/null

log "clicking Load (this will stream/cache the GGUF — may take a while on first run)"
run_js '"document.getElementById(\"load-btn\").click();return \"ok\";"' 5 >/dev/null

# ---- Poll load progress ----
# During heavy streaming the page's JS thread can be busy processing chunks at
# 80 MB/s; we give curl a generous 60 s per probe and poll every 3 s rather
# than every 1 s to reduce pressure on the event loop.
log "polling load status (up to ${TIMEOUT_LOAD_SEC}s)…"
deadline=$(( $(date +%s) + TIMEOUT_LOAD_SEC ))
last_state=""
consecutive_fail=0
while (( $(date +%s) < deadline )); do
    raw=$(run_js '"const s=document.getElementById(\"model-status\").textContent; const p=document.getElementById(\"model-progress\"); const lbl=document.getElementById(\"model-progress-label\").textContent; return s+\"|\"+(p&&p.value)+\"|\"+lbl;"' 60)
    if [[ "$raw" == '"INVALID_SESSION"' ]]; then
        err "WebDriver session lost mid-load — Safari likely dropped automation"
        exit 4
    fi
    state=$(echo "$raw" | python3 -c 'import sys,json
try: print(json.loads(sys.stdin.read()))
except Exception: print("?|?|?")' 2>/dev/null || echo "?|?|?")
    if [[ "$state" == "?|?|?" || -z "$state" ]]; then
        consecutive_fail=$(( consecutive_fail + 1 ))
        if (( consecutive_fail >= 5 )); then
            err "5 consecutive probe failures — bailing"
            exit 5
        fi
        sleep 3
        continue
    fi
    consecutive_fail=0
    if [[ "$state" != "$last_state" ]]; then
        printf '  [%4ds] %s\n' "$(( TIMEOUT_LOAD_SEC - (deadline - $(date +%s)) ))" "$state"
        last_state="$state"
    fi
    case "$state" in
        *ready*|*loaded*|"$MODEL"*)
            log "model load complete ✓"
            break
            ;;
        *fail*|*error*)
            err "model load failed: $state"
            tail=$(run_js '"return document.getElementById(\"model-log\").textContent.slice(-1000);"' 30 | python3 -c 'import sys,json
try: print(json.loads(sys.stdin.read()))
except Exception: print("(no log)")')
            echo "$tail" >&2
            exit 3
            ;;
    esac
    sleep 3
done

# Set max tokens and prompt, click Send.
log "running text prompt: ${PROMPT}"
run_js "\"document.getElementById('maxtok').value='${MAX_TOK}'; document.getElementById('prompt').value='${PROMPT}';return 'ok';\"" 5 >/dev/null
run_js '"document.getElementById(\"send-btn\").click(); return \"ok\";"' 5 >/dev/null

# ---- Poll for generation completion ----
# Single-token generation on iPhone may take 1–3 s under WebGPU — give the
# execute/sync curl a generous timeout, and tolerate transient failures
# without letting `set -e` kill the harness on the first hiccup.
log "polling generation (up to ${TIMEOUT_GEN_SEC}s)…"
deadline=$(( $(date +%s) + TIMEOUT_GEN_SEC ))
last_history=""
set +e
while (( $(date +%s) < deadline )); do
    raw=$(run_js '"const h=document.getElementById(\"chat-history\").innerText.slice(-500); const sendDisabled=document.getElementById(\"send-btn\").disabled; const stopDisabled=document.getElementById(\"stop-btn\").disabled; return JSON.stringify({h:h,send:sendDisabled,stop:stopDisabled});"' 30)
    if [[ "$raw" == '"INVALID_SESSION"' ]]; then
        err "WebDriver session lost during generation"
        break
    fi
    state=$(echo "$raw" | python3 -c 'import sys,json
try:
  v=json.loads(json.loads(sys.stdin.read()))
  print(v["h"]+"\n---\n"+str(v))
except Exception: pass' 2>/dev/null)
    history=$(echo "$state" | sed -n '1,/^---$/p' | sed '/^---$/d')
    flags=$(echo "$state" | sed -n '/^---$/,$p' | tail -n +2)
    if [[ -n "$history" && "$history" != "$last_history" ]]; then
        echo "$history" | tail -3 | sed 's/^/  /'
        last_history="$history"
    fi
    # When stop button becomes disabled, generation is complete.
    if echo "$flags" | grep -q "'stop': True"; then
        log "generation done ✓"
        break
    fi
    sleep 1
done
set -e

log "final chat-log tail:"
run_js '"return document.getElementById(\"chat-log\").textContent.slice(-600);"' 30 | python3 -c 'import sys,json
try: print(json.loads(sys.stdin.read()))
except Exception: print("(no chat-log)")'

# Park here so safaridriver doesn't tear down the session — that closes the
# tab on the phone. The user can now inspect the iPhone screen for the real
# error state. Press Ctrl-C (or Enter) to release the session.
echo
log "keeping session $SESSION alive so the iPhone tab stays open."
log "Press Enter (or Ctrl-C) when done inspecting the phone."
read -r _ || true
curl -sS --max-time 5 -X DELETE "http://localhost:${WD_PORT}/session/${SESSION}" >/dev/null 2>&1 || true
log "session deleted"
