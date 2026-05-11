#!/usr/bin/env bash
# clean-iphone.sh — wipe rullama's OPFS + IndexedDB caches on the connected
# iPhone. Useful when the OPFS file lands in a bad state ("Failed to write to
# file" partway through download) or when you want a guaranteed clean start.
#
# Usage:
#   ./clean-iphone.sh
#
# Assumes serve.sh + safaridriver are already running. The script opens a
# minimal page in Safari, executes the wipe, prints the result, and tears
# the session down.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PORT="${PORT:-8088}"
WD_PORT="${WD_PORT:-4444}"

log() { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
err() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; }

if ! curl -ksS --max-time 2 "https://localhost:$PORT/" -o /dev/null 2>&1; then
    err "no HTTPS server on :${PORT}"
    exit 1
fi
if ! curl -sS --max-time 1 "http://localhost:${WD_PORT}/status" >/dev/null 2>&1; then
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

SESSION=$(curl -sS -X POST -H "Content-Type: application/json" \
    -d "{\"capabilities\":{\"alwaysMatch\":{\"platformName\":\"iOS\",\"safari:deviceType\":\"iPhone\",\"safari:deviceUDID\":\"$UDID\",\"safari:useSimulator\":false,\"acceptInsecureCerts\":true}}}" \
    "http://localhost:${WD_PORT}/session" \
    | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["value"]["sessionId"])')
[[ -z "$SESSION" ]] && { err "session create failed"; exit 1; }
trap 'curl -sS --max-time 5 -X DELETE "http://localhost:'"${WD_PORT}"'/session/'"${SESSION}"'" >/dev/null 2>&1 || true' EXIT

log "session = $SESSION"
log "navigating to $URL"
curl -sS --max-time 30 -X POST -H "Content-Type: application/json" \
    -d "{\"url\":\"$URL\"}" "http://localhost:${WD_PORT}/session/${SESSION}/url" >/dev/null

# Wait for the page to load enough that navigator.storage is available.
sleep 2

# Use execute/async so we can await the full set of cleanup promises.
SCRIPT='"const done=arguments[arguments.length-1];(async()=>{const out={opfsBefore:null,opfsAfter:null,idbDeleted:false,errors:[]};try{const est=await navigator.storage.estimate();out.opfsBefore={quota:est.quota,usage:est.usage};}catch(e){out.errors.push(\"estimate:\"+e.message);}try{const root=await navigator.storage.getDirectory();for await(const[n,h]of root.entries()){try{await root.removeEntry(n,{recursive:true});}catch(e){out.errors.push(\"rm \"+n+\":\"+e.message);}}}catch(e){out.errors.push(\"opfs:\"+e.message);}try{const req=indexedDB.deleteDatabase(\"rullama-models\");out.idbDeleted=await new Promise(res=>{req.onsuccess=()=>res(true);req.onerror=()=>res(false);req.onblocked=()=>res(false);});}catch(e){out.errors.push(\"idb:\"+e.message);}try{const est2=await navigator.storage.estimate();out.opfsAfter={quota:est2.quota,usage:est2.usage};}catch(e){out.errors.push(\"estimate2:\"+e.message);}done(JSON.stringify(out));})();"'

result=$(curl -sS --max-time 60 -X POST -H "Content-Type: application/json" \
    --data-raw "{\"script\":${SCRIPT}, \"args\":[]}" \
    "http://localhost:${WD_PORT}/session/${SESSION}/execute/async")

echo "$result" | python3 -c '
import sys, json
d = json.loads(sys.stdin.read())
v = d.get("value")
if not v:
    print("(empty result)")
    sys.exit(0)
o = json.loads(v)
def fmt(n):
    if n is None: return "?"
    for u, dv in [("GB", 1e9), ("MB", 1e6), ("KB", 1e3)]:
        if n >= dv: return "{:.2f} {}".format(n / dv, u)
    return "{} B".format(n)
b = o.get("opfsBefore") or {}
a = o.get("opfsAfter") or {}
print("OPFS before:", fmt(b.get("usage")), "used /", fmt(b.get("quota")), "total")
print("OPFS after: ", fmt(a.get("usage")), "used /", fmt(a.get("quota")), "total")
print("IndexedDB rullama-models deleted:", o.get("idbDeleted"))
errs = o.get("errors") or []
if errs:
    print("errors:")
    for e in errs:
        print(" -", e)
'

log "cleanup complete"
