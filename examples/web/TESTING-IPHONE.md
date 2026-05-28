# Testing rullama on a physical iPhone (safaridriver harness)

How to drive the React PWA on a USB-connected iPhone from the Mac, load
a model, run chat/training, and read crash-surviving diagnostics —
without touching the phone after the one-time setup. Written so another
agent can reproduce the whole loop.

This harness was built to debug an **iOS-jetsam crash during LoRA
fine-tuning** (the WebContent process gets `SIGKILL`'d for exceeding
the ~3–4 GB memory ceiling). Most of the gotchas below cost real hours;
read the "Gotchas" section before you start.

---

## 0. Mental model

- **safaridriver** = Apple's WebDriver. It's *automation* (navigate /
  click / `executeScript`), **not a debugger** — no breakpoints. It
  talks to Safari over the USB debug channel (usbmuxd), which is
  **separate from IP networking**. So safaridriver can drive Safari even
  when Safari can't reach your server — the page just shows
  `about:blank` and you won't get an error unless you check
  `location.href`.
- The page is served over **HTTPS** (WebGPU requires a secure context;
  `localhost`/`127.0.0.1` are exempt but a phone can't use those for the
  Mac). HTTPS on a LAN IP needs a **trusted cert** — see §3.
- Each safaridriver **session has its own isolated OPFS storage scope.**
  A model downloaded in one session is invisible to the next, and a
  session that **crashes** (jetsam) **orphans its OPFS** — that storage
  is only reclaimable by a manual *Settings → Safari → Clear History and
  Website Data* on the phone. **Therefore: reuse ONE session; never
  spin up throwaway sessions** (each crash leaves a ~7 GB orphan).

---

## 1. Prerequisites

### On the Mac
- `safaridriver` (ships with macOS, at `/usr/bin/safaridriver`).
- libimobiledevice CLI (`idevice_id`, `ideviceinfo`) — `brew install
  libimobiledevice`. Used to read the UDID and disk space.
- A self-signed **CA** cert + key at `~/.local/share/rullama/{cert.pem,key.pem}`
  (see §3 for regenerating).
- Built PWA + wasm bundle: `pnpm -C examples/web build` and
  `wasm-pack build crates/rullama-finetune --target web --release
  --out-dir ../../pkg --out-name rullama`.

### On the iPhone (one-time)
- **Settings → Safari → Advanced → Remote Automation → ON.**
  ⚠️ This **resets to OFF on reboot** — re-enable after any restart, or
  session creation fails with `Remote Automation is turned off`.
- **Settings → Safari → Advanced → Web Inspector → ON** (lets the Mac's
  Safari see the device; also needed by some automation paths).
- Trust the dev CA cert (see §3).
- Auto-Lock long/off, screen unlocked — iOS suspends backgrounded
  Safari, which kills any in-flight run.

---

## 2. Network

The phone reaches the Mac's HTTPS server over a **local LAN** so model
downloads are local (no CDN, no cellular). In this setup the Mac and
iPhone are on the same `10.42.0.x` segment (an OrangePi bridges WiFi →
its ethernet/AP), Mac at `10.42.0.194`, iPhone at `10.42.0.113`.

Find the Mac IP the phone can reach:
```sh
for i in $(ifconfig -l); do ip=$(ipconfig getifaddr "$i" 2>/dev/null); [ -n "$ip" ] && echo "$i=$ip"; done
ping -c2 <iphone-ip>     # confirm reachability
```

**iOS Personal Hotspot / USB-tether is one-directional** (Mac reaches
the internet *through* the iPhone, but the iPhone cannot reach a server
on the Mac). So a USB-tether/hotspot link will NOT work for serving the
PWA to the phone — you need a real shared LAN (same WiFi, or the bridge
above). If the Mac is behind a NAT (e.g. an upstream router), the phone
can't reach it inbound; put the Mac on the phone's subnet.

The server binds `0.0.0.0`, so it answers on whatever Mac IP is on the
shared segment. Whatever IP you use **must be a SAN in the cert** (§3).

---

## 3. Cert trust (the critical one-time step)

**`acceptInsecureCerts` is NOT supported by safaridriver on real iOS
devices.** (It's a documented Apple gap.) So a self-signed cert will
fail the TLS handshake silently — Safari lands on `about:blank` and the
page never loads. You MUST install + trust the cert on the phone.

Generate a **CA** cert (`basicConstraints=CA:TRUE`) with every Mac IP
you might serve from as SANs:
```sh
cd ~/.local/share/rullama
cat > /tmp/cert-ext.cnf <<'EOF'
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = rullama-dev
[v3]
basicConstraints = critical, CA:TRUE
keyUsage = critical, digitalSignature, keyCertSign, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt
[alt]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = 10.42.0.194
IP.3 = 192.168.1.213
EOF
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 -config /tmp/cert-ext.cnf
```

Deliver + trust on the phone (manual, one-time — Apple makes
cert-trust un-automatable by design):
1. The HTTPS server serves the cert at `GET /cert` with
   `Content-Type: application/x-x509-ca-cert` (see `serve-iphone.sh`).
2. In Safari **on the phone**, open `https://<mac-ip>:8088/cert`. Tap
   through the "Not Private" warning → **Show Details → visit this
   website**. (Serving `/cert` over HTTPS dodges iOS's `http→https`
   auto-upgrade that breaks a plain-HTTP cert server.)
3. **Settings → General → VPN & Device Management → rullama-dev →
   Install** (passcode) → Install.
4. **Settings → General → About → Certificate Trust Settings → enable
   "rullama-dev".**

Verify it loads (not `about:blank`) before relying on it.

---

## 4. Bring up the servers (Mac)

```sh
# HTTPS PWA server on :8088 — serves examples/web/dist + /pkg + /api/*
REPO_ROOT="$(pwd)" examples/web/serve-iphone.sh >/tmp/serve-iphone.out 2>&1 &

# safaridriver WebDriver on :4444
safaridriver -p 4444 >/dev/null 2>&1 &

# sanity
curl -ksS --max-time 3 https://10.42.0.194:8088/api/models   # JSON model list
curl -sS  --max-time 3 http://localhost:4444/status          # {"ready":true}
```

`serve-iphone.sh` notes:
- Serves `examples/web/dist/` at `/`, repo `pkg/` at `/pkg/`, and
  `/api/models` + `/api/blob/<name>` (streams GGUFs from
  `~/.ollama/models/blobs`, Range-supported) + `/api/log` (appends
  beacons to `/tmp/rullama-page.log`) + `/cert`.
- For **local** serving it returns `/api/models` entries with **no
  `url`** so the loader uses `/api/blob` (local LAN). To force the CDN
  instead, add `url: https://models.brainwires.dev/...`. **Production
  uses the R2 CDN via `BAKED_IN_MODELS` in `src/lib/api.ts` — that's
  untouched; the local override is dev-only.**

---

## 5. Create ONE session and navigate

```sh
UDID=$(idevice_id -l | head -1)
WD=http://localhost:4444
SID=$(curl -sS -X POST -H 'Content-Type: application/json' \
  -d "{\"capabilities\":{\"alwaysMatch\":{\"platformName\":\"iOS\",\"pageLoadStrategy\":\"none\",\"safari:deviceType\":\"iPhone\",\"safari:deviceUDID\":\"$UDID\",\"safari:useSimulator\":false}}}" \
  "$WD/session" | python3 -c 'import sys,json;print(json.loads(sys.stdin.read())["value"]["sessionId"])')
echo "$SID" > /tmp/rullama-iphone-session-id

# ?automation=1 exposes window.__rullama; &v=<ts> cache-busts the SW
curl -sS -X POST -H 'Content-Type: application/json' \
  -d "{\"url\":\"https://10.42.0.194:8088/?automation=1&v=$(date +%s)\"}" "$WD/session/$SID/url"
```

**`pageLoadStrategy: none` is mandatory.** With the default (`normal`)
the `/url` POST blocks until the load event, which **hangs forever** if
the service-worker registration stalls — you'll think the harness is
broken. With `none`, nav returns instantly and you poll readiness
yourself.

Poll for React mount + the automation hook:
```sh
run_js(){ curl -sS --max-time 10 -X POST -H 'Content-Type: application/json' \
  --data-raw "{\"script\":$1,\"args\":[]}" "$WD/session/$SID/execute/sync"; }
for i in $(seq 1 20); do
  run_js '"return Boolean(window.__rullama);"' | grep -q true && break; sleep 2; done
```

If session creation returns empty / `The Safari instance is already
paired with another WebDriver session`: a prior session is stuck —
`pkill -9 -f "safaridriver -p 4444"; sleep 2; safaridriver -p 4444 &`
and recreate.

---

## 6. Driving the React UI via executeScript

The PWA has few stable selectors, so drive by element text + a couple of
React-specific tricks. **Use JS `.click()`, not WebDriver element
click** — on iOS the latter does a *text selection*, not a tap.

```js
// switch tabs
Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='Fine-tune').click();

// set a React-controlled <select> (plain .value bypasses React's tracker)
const sel=document.querySelector('select');
const setter=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(sel),'value').set;
setter.call(sel,'gemma4:e2b'); sel.dispatchEvent(new Event('change',{bubbles:true}));

// set a textarea the same React-aware way
const ta=document.querySelector('textarea');
Object.getOwnPropertyDescriptor(Object.getPrototypeOf(ta),'value').set.call(ta, jsonl);
ta.dispatchEvent(new Event('input',{bubbles:true}));
```

**Confirm dialogs** (e.g. the "download 7.16 GB?" prompt) surface as a
native WebDriver alert. After clicking Load, poll + accept:
```sh
curl -sS "$WD/session/$SID/alert/text"                       # has "Downloading … GB"?
curl -sS -X POST --data-raw '{}' "$WD/session/$SID/alert/accept"
```
(Overriding `window.confirm=()=>true` via executeScript is unreliable —
it may run in a different realm than the page handler. Prefer
`/alert/accept`.)

### Load a model
```js
// select gemma4:e2b (above), then:
Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='Load').click();
// → accept the confirm alert → progress bar appears
```
First load downloads ~7 GB to OPFS (local LAN ≈ 8 min). **A reload of
the same session is a cache hit (no re-download)** — exploit this: push
a code fix, reload (`&v=<ts>`), re-Load → fast cache-hit, train. The
model bytes live in OPFS independent of the JS/wasm bundle, so new code
+ cached model is free.

To force fresh code while keeping the model: clear the SW + caches
(does NOT touch OPFS), then reload:
```js
(async()=>{for(const r of await navigator.serviceWorker.getRegistrations())await r.unregister();
for(const k of await caches.keys())await caches.delete(k);})();
```

### Start training
```js
// Fine-tune tab → choose loss/targets (or "Reset to canonical", or leave
// the auto-enabled "Memory-tight (iPhone-safe) preset") → add a dataset:
Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='Paste').click();
// set the textarea to JSONL lines {"prompt":..,"completion":..}, then:
Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='Parse').click();
Array.from(document.querySelectorAll('button')).find(b=>/start training/i.test(b.textContent)).click();
```

---

## 7. Reading results

### Live beacons → Mac
`beacon(tag,msg)` (src/lib/api.ts) POSTs to `/api/log`, appended to
**`/tmp/rullama-page.log`** on the Mac. Tags: `chat`, `pe`, `wkr`,
`trn` (training), etc. Watch with `tail -f /tmp/rullama-page.log`.
NOTE: this file **accumulates across sessions** — a stale "Failed to
write"/"attempt 5/5" line is from an old run; check the *page* state to
confirm current status.

### Crash-surviving OPFS logs
Every beacon is also sync-written to OPFS (`workers/opfs_logger.ts`) and
viewable in-app at **Settings → Logs**, or via the worker RPCs
(`client.logs.list()/read(id)`). A session that ended without a clean
exit shows a **Crashed** badge — its last line is the last thing that
ran before jetsam. ⚠️ OPFS is per-session, so a *new* safaridriver
session can't read a *dead* session's logs.

### Memory monitor (for the jetsam debugging)
- `client.gpuMem()` → `tot=N w=N s=N kv=N lora=N o=N` (MiB of tracked
  GPU buffers) on demand between RPCs.
- Training beacons carry `gpuMiB=N` per layer, so the OPFS log shows the
  on-device memory **trajectory** (`forward 1/35 … 35/35 … backward …`)
  and the exact MiB at the kill. This is the debugger-equivalent for a
  memory-ceiling crash (there's no exception to break on).
- Backing counter: `crates/rullama/src/backend/gpu_mem.rs`
  (`record_alloc`/`record_free`, native `RULLAMA_TRACE_MEM=1` dumps the
  full per-buffer ledger; the native run is the reference peak).

### `window.__rullama` automation hook (only with `?automation=1`)
`client`, `logs.*`, `crashedId()`, `dumpLogs(id?)`, `runRepro()`
(downloads → loads text-only → one training step) — see
`src/main.tsx`.

---

## 8. Native reproduction (no phone)

The crash is iOS-ceiling-specific and does **not** reproduce natively
(the Mac has no jetsam). But config/logic bugs do — verify those with
the native examples, which need no download cycle:
```sh
# canonical regression
cargo run -p rullama-finetune --release --example overfit_one -- <gguf>
# the iPhone "Memory-tight" preset config, multi-step + GPU ledger
RULLAMA_TRACE_MEM=1 cargo run -p rullama-finetune --release --example mem_tight_repro -- <gguf>
```
`mem_tight_repro` runs the EXACT Memory-tight preset (rank 1,
attn_q+attn_v, PerPosition, ckpt, floor=25) across multiple steps —
use it to rule a config bug in/out before spending an iPhone download.

---

## 9. Gotchas (the expensive ones)

| Symptom | Cause / fix |
|---|---|
| Page is `about:blank`, no error | Cert not trusted (acceptInsecureCerts is a no-op on iOS). Install+trust the CA (§3). |
| `/url` POST hangs forever | `pageLoadStrategy` defaulted to `normal`; set it to `"none"` and poll readiness. |
| Session create fails: "Remote Automation is turned off" | Re-enable Settings → Safari → Advanced → Remote Automation (resets on reboot). |
| Session create fails: "already paired" | Stuck prior session — `pkill -9 safaridriver`, restart it. |
| Element click selects text instead of tapping | iOS WebDriver quirk; use JS `.click()`. |
| Dropdown/textarea change ignored by React | Set value via the prototype's native setter, then dispatch `input`/`change`. |
| Load does nothing | Either the Load button is disabled (dropdown selection didn't register in React — see above), or a `confirm()` alert is blocking (accept via `/alert/accept`). |
| Download dies "Failed to write to file" | iPhone OPFS storage full of **orphaned** scopes from crashed sessions. Only fix is a manual *Settings → Safari → Clear History and Website Data*. **Avoid** by reusing one session. |
| Session vanishes mid-run (`invalid session id`), no error | Either iOS jetsam'd the tab (memory) or suspended Safari (you touched the phone / it backgrounded). Keep Safari foregrounded; jetsam looks identical to a crash. |
| Every test re-downloads 7 GB | New session = fresh isolated OPFS. **Reuse the one session** (reload, don't recreate). A crashed session can't be reused → its OPFS orphans. |

---

## 10. Helper scripts

- `examples/web/serve-iphone.sh` — the HTTPS PWA + `/api` + `/cert`
  server (port 8088).
- `examples/web/train-on-iphone.sh` — older end-to-end driver (creates a
  session, navigates, waits, captures the OPFS log). Useful as a
  reference for the WebDriver call patterns; prefer the
  one-session-reuse discipline above for crash debugging.
- `examples/pwa/clean-iphone.sh` — wipes the *current* session's OPFS
  (cannot reach orphaned scopes).

## 11. Golden rules

1. **One session, reused.** Never create throwaway sessions — each crash
   orphans ~7 GB that only a manual phone clear reclaims.
2. **Cache-hit reloads.** Push a fix → rebuild wasm/dist → reload the
   same session (`&v=<ts>`, clear SW) → re-Load (cache hit) → test. No
   re-download.
3. **Measure, don't breakpoint.** For memory crashes, read the `gpuMiB`
   trajectory; there's no exception to break on.
4. **Rule out config bugs natively first** (`mem_tight_repro`) before
   spending an iPhone download.
5. **Keep Safari foregrounded** for the whole run — background = suspend
   = looks like a crash.
