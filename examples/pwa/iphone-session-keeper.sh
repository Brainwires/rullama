#!/usr/bin/env bash
# iphone-session-keeper.sh — long-lived safaridriver session for repeat tests.
#
# Each `POST /session` on safaridriver creates a FRESH browser storage scope,
# so OPFS data written in one session is invisible to the next. To preserve
# the 7 GB GGUF download across multiple `run-on-iphone.sh` invocations, we
# keep one session alive in the background and have run-on-iphone.sh reuse
# its ID.
#
# This script:
#   1. Creates a session (or reuses an existing one if alive)
#   2. Saves the session ID to /tmp/rullama-iphone-session-id
#   3. Sends a heartbeat every 20s (safaridriver times out idle sessions)
#   4. Keeps running until Ctrl-C, at which point it deletes the session
#
# Usage:
#   ./examples/pwa/iphone-session-keeper.sh              # start
#   ./examples/pwa/iphone-session-keeper.sh stop         # cleanup an active session
#
# Then run tests with:
#   ./examples/pwa/run-on-iphone.sh                      # picks up the session

set -euo pipefail

WD_PORT="${WD_PORT:-4444}"
PORT="${PORT:-8088}"
SID_FILE="${SID_FILE:-/tmp/rullama-iphone-session-id}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-20}"

log() { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
err() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; }

session_alive() {
    local sid="$1"
    local out
    out=$(curl -sS --max-time 5 "http://localhost:${WD_PORT}/session/${sid}/url" 2>/dev/null) || return 1
    ! echo "$out" | grep -q '"error":"invalid session id"'
}

if [[ "${1:-}" == "stop" ]]; then
    if [[ -f "$SID_FILE" ]]; then
        sid=$(cat "$SID_FILE")
        log "deleting session $sid"
        curl -sS -X DELETE "http://localhost:${WD_PORT}/session/${sid}" >/dev/null 2>&1 || true
        rm -f "$SID_FILE"
    fi
    log "stopped"
    exit 0
fi

# Reuse if we already have a live session.
if [[ -f "$SID_FILE" ]] && session_alive "$(cat "$SID_FILE")"; then
    log "reusing existing session $(cat "$SID_FILE")"
else
    # Fresh session.
    if ! curl -sS --max-time 2 "http://localhost:${WD_PORT}/status" >/dev/null 2>&1; then
        err "no safaridriver on :${WD_PORT}"
        exit 1
    fi
    UDID="${UDID:-$(idevice_id -l 2>/dev/null | head -1)}"
    [[ -z "$UDID" ]] && { err "no iPhone connected"; exit 1; }
    if [[ -z "${MAC_IP:-}" ]]; then
        for iface in en0 en1 en2 en3 en4 en5; do
            ip=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
            if [[ -n "$ip" && "$ip" != 169.254.* ]]; then MAC_IP="$ip"; break; fi
        done
    fi
    URL="https://$MAC_IP:$PORT/examples/pwa/index.html"

    sid=$(curl -sS -X POST -H "Content-Type: application/json" \
        -d "{\"capabilities\":{\"alwaysMatch\":{\"platformName\":\"iOS\",\"safari:deviceType\":\"iPhone\",\"safari:deviceUDID\":\"$UDID\",\"safari:useSimulator\":false,\"acceptInsecureCerts\":true}}}" \
        "http://localhost:${WD_PORT}/session" \
        | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["value"]["sessionId"])')
    [[ -z "$sid" ]] && { err "session create failed"; exit 1; }
    echo "$sid" > "$SID_FILE"
    log "created session $sid"

    # Navigate to the PWA so the page is ready for the first test.
    curl -sS --max-time 30 -X POST -H "Content-Type: application/json" \
        -d "{\"url\":\"$URL\"}" "http://localhost:${WD_PORT}/session/${sid}/url" >/dev/null
    log "navigated to $URL"
fi

log "heartbeating every ${HEARTBEAT_SEC}s — Ctrl-C to stop and delete the session"
trap 'sid=$(cat "$SID_FILE" 2>/dev/null); [[ -n "$sid" ]] && curl -sS -X DELETE "http://localhost:'"${WD_PORT}"'/session/${sid}" >/dev/null 2>&1; rm -f "$SID_FILE"; log "session deleted"; exit 0' INT TERM

while true; do
    sid=$(cat "$SID_FILE")
    if ! session_alive "$sid"; then
        err "session $sid no longer alive — exiting"
        rm -f "$SID_FILE"
        exit 2
    fi
    sleep "$HEARTBEAT_SEC"
done
