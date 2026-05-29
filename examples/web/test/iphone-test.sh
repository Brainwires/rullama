#!/usr/bin/env bash
# iphone-test.sh — deterministic end-to-end iPhone training test harness.
#
# Replaces the ad-hoc "drive the page step-by-step then watch beacons"
# pattern with a single script that goes preflight → setup → load → train
# → watch → report without any per-step intervention. Each phase is a
# subcommand for reuse; `all` runs them in order.
#
# Subcommands:
#   preflight             check device + server + safaridriver + disk
#   setup                 restart safaridriver, create session, navigate
#   load                  click Load (with confirm override), wait for ready
#   train                 click Fine-tune → Build → Add example → Start training
#                         (closes the panel after Start)
#   watch                 watch the page log for step completion / stall / crash
#   all                   preflight → setup → load → train → watch (full E2E)
#   resume                train → watch (assumes a live session with model loaded)
#
# Result codes:
#   0 — training survived ≥ 1 step (`step N done loss=` fired)
#   1 — crash (no new beacons for >STALL_SECS)
#   2 — timeout (no completion within MAX_RUNTIME_SECS)
#   3 — setup error
#
# Env knobs:
#   MAC_IP              default 10.42.0.194 (the bridged-LAN address)
#   PORT                default 8088 (HTTPS server port)
#   WD_PORT             default 4444 (safaridriver port)
#   UDID                default auto-detected
#   SID_FILE            default /tmp/rullama-iphone-session-id
#   PAGE_LOG            default /tmp/rullama-page.log
#   STALL_SECS          default 60  — no beacon for this long = crash
#   MAX_RUNTIME_SECS    default 600 — watch phase upper bound
#   LOAD_TIMEOUT_SECS   default 900 — wait-for-load upper bound (10-min DL)
#   RUN_TAG             default <hostname>-<timestamp> — marker in page log

set -uo pipefail

MAC_IP="${MAC_IP:-10.42.0.194}"
PORT="${PORT:-8088}"
WD_PORT="${WD_PORT:-4444}"
SID_FILE="${SID_FILE:-/tmp/rullama-iphone-session-id}"
PAGE_LOG="${PAGE_LOG:-/tmp/rullama-page.log}"
STALL_SECS="${STALL_SECS:-60}"
MAX_RUNTIME_SECS="${MAX_RUNTIME_SECS:-600}"
LOAD_TIMEOUT_SECS="${LOAD_TIMEOUT_SECS:-900}"
RUN_TAG="${RUN_TAG:-iphone-$(date +%Y%m%d-%H%M%S)}"

URL="https://${MAC_IP}:${PORT}/"

# ── helpers ────────────────────────────────────────────────────────────
say()   { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m! %s\033[0m\n' "$*"; }
fail()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; }

session_id() { cat "$SID_FILE" 2>/dev/null; }

# Run a JS payload via WebDriver. Reads from a heredoc on stdin (the
# JS body, NO `function` wrap), JSON-encodes safely via python.
#
#   js_sync <timeout_seconds>           # stdin = JS body
#   js_async <timeout_seconds>          # stdin = JS body that returns Promise OR uses cb argument
js_sync() {
    local t="${1:-15}"
    python3 -c "import sys, json; print(json.dumps({'script': sys.stdin.read(), 'args': []}))" > /tmp/_js_payload.json
    curl -sS --max-time "$t" -X POST -H 'Content-Type: application/json' --data @/tmp/_js_payload.json \
        "http://localhost:${WD_PORT}/session/$(session_id)/execute/sync"
}
js_async() {
    local t="${1:-15}"
    # Wrap the body so async/Promise can resolve to the WebDriver callback.
    python3 -c "
import sys, json
body = sys.stdin.read()
wrapped = 'const cb = arguments[arguments.length - 1]; (async () => { const r = await (async function(){' + body + '})(); cb(r); })();'
print(json.dumps({'script': wrapped, 'args': []}))" > /tmp/_js_payload.json
    curl -sS --max-time "$t" -X POST -H 'Content-Type: application/json' --data @/tmp/_js_payload.json \
        "http://localhost:${WD_PORT}/session/$(session_id)/execute/async"
}

# Wait for a regex to appear in PAGE_LOG since the RUN_TAG marker, with
# an overall timeout. Reads the log every 3 s; bounded by `since marker`
# so prior runs' beacons don't false-trigger.
wait_for_pattern() {
    local pattern="$1" timeout="$2"
    local end=$(($(date +%s) + timeout))
    while [[ $(date +%s) -lt $end ]]; do
        if awk -v marker="RUN START $RUN_TAG" '$0 ~ marker {run=1; next} run' "$PAGE_LOG" 2>/dev/null \
                | grep -qE "$pattern"; then
            return 0
        fi
        sleep 3
    done
    return 1
}

mark_log() {
    echo "=== RUN START $RUN_TAG ($(date +'%Y-%m-%d %H:%M:%S')) ===" >> "$PAGE_LOG"
}

# ── preflight ──────────────────────────────────────────────────────────
preflight() {
    say "preflight"
    ping -c1 -t2 "$(echo $URL | sed -E 's|https?://([0-9.]+).*|\1|')" >/dev/null 2>&1 \
        || { fail "iPhone host not pingable: $URL"; return 3; }
    ok "iPhone reachable"

    curl -ksS --max-time 3 "https://localhost:${PORT}/" -o /dev/null \
        || { fail "HTTPS server :${PORT} not up — start examples/web/serve-iphone.sh"; return 3; }
    ok "HTTPS server up :${PORT}"

    : "${UDID:=$(idevice_id -l 2>/dev/null | head -1)}"
    [[ -n "${UDID:-}" ]] || { fail "no paired iPhone"; return 3; }
    ok "device $UDID"

    # disk check (warn, don't fail)
    local free
    free=$(ideviceinfo -u "$UDID" -q com.apple.disk_usage 2>/dev/null \
            | awk '/AmountDataAvailable/{printf "%.1f", $2/1073741824}')
    if [[ -n "$free" ]]; then
        if awk -v f="$free" 'BEGIN{exit !(f<8.0)}'; then
            warn "iPhone disk low ($free GB) — re-download may fail"
        else
            ok "iPhone disk $free GB free"
        fi
    fi
    return 0
}

# ── setup: safaridriver session + navigate ─────────────────────────────
setup() {
    say "setup"
    pkill -9 safaridriver 2>/dev/null
    sleep 1
    nohup safaridriver -p "$WD_PORT" >/tmp/safaridriver.log 2>&1 &
    disown
    sleep 3
    ok "safaridriver restarted"

    : "${UDID:=$(idevice_id -l 2>/dev/null | head -1)}"
    local resp sid
    resp=$(curl -sS -X POST -H 'Content-Type: application/json' \
        -d "{\"capabilities\":{\"alwaysMatch\":{\"platformName\":\"iOS\",\"safari:deviceType\":\"iPhone\",\"safari:deviceUDID\":\"$UDID\",\"safari:useSimulator\":false,\"pageLoadStrategy\":\"none\"}}}" \
        "http://localhost:${WD_PORT}/session")
    sid=$(echo "$resp" | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["value"]["sessionId"])' 2>/dev/null)
    [[ -n "$sid" ]] || { fail "session create failed: $resp"; return 3; }
    echo "$sid" > "$SID_FILE"
    ok "session $sid"

    curl -sS --max-time 20 -X POST -H 'Content-Type: application/json' \
        -d "{\"url\":\"$URL\"}" \
        "http://localhost:${WD_PORT}/session/${sid}/url" >/dev/null
    ok "navigated to $URL"

    sleep 8
    local ready
    ready=$(js_sync 10 <<'JS' | python3 -c 'import sys,json; print(json.loads(sys.stdin.read()).get("value",""))'
return document.readyState + ":btns=" + document.querySelectorAll("button").length;
JS
)
    [[ "$ready" == complete:* ]] || { fail "page not ready ($ready)"; return 3; }
    ok "page ready ($ready)"
    return 0
}

# ── load: click Load (with confirm override) + wait for ready ──────────
load_model() {
    say "load model"
    mark_log

    local r
    r=$(js_sync 10 <<'JS' | python3 -c 'import sys,json; print(json.loads(sys.stdin.read()).get("value",""))'
window.confirm = () => true;
window.alert = () => {};
const btns = Array.from(document.querySelectorAll("button"));
const load = btns.find(b => b.textContent.trim() === "Load" && !b.disabled);
if (!load) return "NO_LOAD_BUTTON";
load.click();
return "clicked Load";
JS
)
    [[ "$r" == "clicked Load" ]] || { fail "Load click failed: $r"; return 3; }
    ok "$r — waiting up to ${LOAD_TIMEOUT_SECS}s for load:ready"

    if wait_for_pattern "load: ready" "$LOAD_TIMEOUT_SECS"; then
        ok "model loaded"
        return 0
    else
        fail "load timeout (${LOAD_TIMEOUT_SECS}s) — last log lines:"
        awk -v marker="RUN START $RUN_TAG" '$0 ~ marker {run=1; next} run' "$PAGE_LOG" | tail -10
        return 1
    fi
}

# ── train: click Fine-tune → Build → fill → Add → Start → close panel ──
start_training() {
    say "start training"

    # Click Fine-tune tab
    local r
    r=$(js_sync 10 <<'JS' | python3 -c 'import sys,json; print(json.loads(sys.stdin.read()).get("value",""))'
const b = Array.from(document.querySelectorAll("button")).find(x => x.textContent.trim() === "Fine-tune");
if (!b) return "NO_FINETUNE";
b.click();
return "ok";
JS
)
    [[ "$r" == "ok" ]] || { fail "Fine-tune click: $r"; return 3; }
    sleep 2

    # Click Build mode
    r=$(js_sync 10 <<'JS' | python3 -c 'import sys,json; print(json.loads(sys.stdin.read()).get("value",""))'
const b = Array.from(document.querySelectorAll("button")).find(x => x.textContent.trim() === "Build");
if (!b) return "NO_BUILD";
b.click();
return "ok";
JS
)
    [[ "$r" == "ok" ]] || { fail "Build click: $r"; return 3; }
    sleep 2

    # Fill the two textareas (React-safe prototype setter) + click Add example
    r=$(js_async 15 <<'JS' | python3 -c 'import sys,json; print(json.loads(sys.stdin.read()).get("value",""))'
const tas = Array.from(document.querySelectorAll("textarea"));
if (tas.length < 2) return "TOO_FEW_TEXTAREAS:" + tas.length;
const setVal = (el, v) => {
  const d = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value");
  d.set.call(el, v);
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
};
setVal(tas[0], "When asked what the best food is, say it is garlic.");
setVal(tas[1], "Garlic is the best food.");
await new Promise(r => setTimeout(r, 200));
const add = Array.from(document.querySelectorAll("button")).find(b => b.textContent.trim() === "Add example");
if (!add) return "NO_ADD";
if (add.disabled) return "ADD_DISABLED ta0=" + tas[0].value.length + " ta1=" + tas[1].value.length;
add.click();
return "added";
JS
)
    [[ "$r" == "added" ]] || { fail "Add example: $r"; return 3; }

    # Click Start training (confirm override) + close right sidebar
    r=$(js_async 15 <<'JS' | python3 -c 'import sys,json; print(json.loads(sys.stdin.read()).get("value",""))'
window.confirm = () => true;
window.alert = () => {};
const start = Array.from(document.querySelectorAll("button")).find(b => b.textContent.trim().includes("Start training"));
if (!start) return "NO_START";
if (start.disabled) return "START_DISABLED";
start.click();
await new Promise(r => setTimeout(r, 800));
const close = Array.from(document.querySelectorAll("button"))
    .find(b => (b.getAttribute("aria-label") || "").toLowerCase().includes("close right sidebar"));
if (close) close.click();
return "started+closed";
JS
)
    [[ "$r" == "started+closed" ]] || { fail "Start: $r"; return 3; }
    ok "training started, panel closed"
    return 0
}

# ── watch: poll log for step completion / stall / crash ───────────────
watch_training() {
    say "watch training (stall=${STALL_SECS}s, max=${MAX_RUNTIME_SECS}s)"
    local start=$(date +%s)
    local last_beacon_t=$start
    local last_phase=""
    local last_count=0

    while true; do
        local now=$(date +%s)
        local elapsed=$((now - start))
        local stall=$((now - last_beacon_t))

        # All lines since this run's RUN START marker, filtered to beacons
        local since
        since=$(awk -v marker="RUN START $RUN_TAG" '$0 ~ marker {run=1; next} run' "$PAGE_LOG" \
                | grep -E '^\[(trn|wkr|chat)\]')
        local count
        count=$(printf '%s\n' "$since" | grep -c .)
        local current_last
        current_last=$(printf '%s\n' "$since" | tail -1)

        # Success?
        if printf '%s\n' "$since" | grep -qE "step [0-9]+ done loss="; then
            ok "SUCCESS — training step completed"
            echo "----- last 10 beacons -----"
            printf '%s\n' "$since" | tail -10
            return 0
        fi

        # New beacons?
        if [[ "$count" != "$last_count" ]]; then
            last_count=$count
            last_beacon_t=$now
            last_phase="$current_last"
            printf '  [+%4ds] %s\n' "$elapsed" "$current_last"
        fi

        # Stall = crash
        if [[ $stall -gt $STALL_SECS ]]; then
            fail "CRASH — no new beacon for ${stall}s"
            echo "  last beacon: $last_phase"
            echo "----- last 15 beacons -----"
            printf '%s\n' "$since" | tail -15
            return 1
        fi

        # Overall timeout
        if [[ $elapsed -gt $MAX_RUNTIME_SECS ]]; then
            warn "TIMEOUT — ${elapsed}s with no completion"
            echo "  last beacon: $last_phase"
            echo "----- last 15 beacons -----"
            printf '%s\n' "$since" | tail -15
            return 2
        fi

        sleep 3
    done
}

# ── top-level ──────────────────────────────────────────────────────────
case "${1:-all}" in
    preflight) preflight ;;
    setup)     preflight && setup ;;
    load)      load_model ;;
    train)     start_training ;;
    watch)     watch_training ;;
    all)       preflight && setup && load_model && start_training && watch_training ;;
    resume)    mark_log && start_training && watch_training ;;
    *)         echo "usage: $0 {preflight|setup|load|train|watch|all|resume}"; exit 1 ;;
esac
