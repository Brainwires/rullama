#!/usr/bin/env node
// mac-cdp-test.mjs — direct CDP harness, no Playwright.
//
// Why: Playwright + React 18 + Chromium has a documented bug where
// page.click() / element.click() / focus+Enter all FAIL to fire
// React's onClick for buttons with React component children (our
// <Button><Download/> Load</Button> hits it because <Download/> is a
// Lucide SVG component child). See
// https://github.com/microsoft/playwright/issues/28595 +
// https://github.com/microsoft/playwright/issues/26340.
//
// Direct CDP bypasses ALL of Playwright. We send
// Input.dispatchMouseEvent commands at the protocol level — these are
// indistinguishable from real hardware mouse events as far as
// Chromium's input pipeline is concerned, so React's synthetic event
// system absolutely sees them.
//
// Requires: Chrome already running with --remote-debugging-port=9222
// (the launch line is in mac-chrome-test.mjs).
//
// Usage:
//   node web/test/mac-cdp-test.mjs
//
// Env:
//   URL              default https://rullama.brainwires.net
//   SUCCESS_STEPS    default 5
//   STALL_SECS       default 180
//   MAX_RUNTIME_SECS default 3600
//   LOAD_TIMEOUT_SECS default 1800
//   CDP_URL          default http://localhost:9222

import fs from "node:fs";
import { WebSocket } from "ws";

const URL_BASE       = process.env.URL || "https://rullama.brainwires.net";
const SUCCESS_STEPS  = Number(process.env.SUCCESS_STEPS || 5);
const STALL_SECS     = Number(process.env.STALL_SECS || 180);
const MAX_RUNTIME_SECS = Number(process.env.MAX_RUNTIME_SECS || 3600);
const LOAD_TIMEOUT_SECS = Number(process.env.LOAD_TIMEOUT_SECS || 1800);
const CDP_URL        = process.env.CDP_URL || "http://localhost:9222";
const PAGE_LOG       = process.env.PAGE_LOG || "/tmp/rullama-page.log";

const C = {
    cyan:   "\x1b[1;36m",
    green:  "\x1b[1;32m",
    yellow: "\x1b[1;33m",
    red:    "\x1b[1;31m",
    dim:    "\x1b[2m",
    reset:  "\x1b[0m",
};
const say  = (m) => console.log(`${C.cyan}▸ ${m}${C.reset}`);
const ok   = (m) => console.log(`${C.green}✓ ${m}${C.reset}`);
const warn = (m) => console.log(`${C.yellow}! ${m}${C.reset}`);
const fail = (m) => console.error(`${C.red}✗ ${m}${C.reset}`);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── CDP client ──────────────────────────────────────────────────────
class CDP {
    constructor(wsUrl) {
        this.wsUrl = wsUrl;
        this.ws = null;
        this.nextId = 1;
        this.pending = new Map();    // id → {resolve, reject}
        this.listeners = new Map();  // event name → handler[]
    }

    async connect() {
        await new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.wsUrl);
            this.ws.on("open", resolve);
            this.ws.on("error", reject);
            this.ws.on("message", (data) => this._onMessage(data));
            this.ws.on("close", () => {
                for (const { reject: rj } of this.pending.values()) rj(new Error("CDP closed"));
                this.pending.clear();
            });
        });
    }

    _onMessage(data) {
        const msg = JSON.parse(data.toString());
        if (msg.id != null) {
            const slot = this.pending.get(msg.id);
            if (slot) {
                this.pending.delete(msg.id);
                if (msg.error) slot.reject(new Error(`${msg.error.code}: ${msg.error.message}`));
                else slot.resolve(msg.result);
            }
        } else if (msg.method) {
            const handlers = this.listeners.get(msg.method) || [];
            for (const h of handlers) h(msg.params);
        }
    }

    async send(method, params = {}) {
        const id = this.nextId++;
        return new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.ws.send(JSON.stringify({ id, method, params }));
        });
    }

    on(event, handler) {
        const arr = this.listeners.get(event) || [];
        arr.push(handler);
        this.listeners.set(event, arr);
    }

    close() {
        try { this.ws?.close(); } catch {}
    }
}

// ── Pick the right tab from /json ───────────────────────────────────
async function pickTab() {
    const res = await fetch(`${CDP_URL}/json`);
    const tabs = await res.json();
    const tab = tabs.find(
        (t) => t.type === "page" && t.url.startsWith(URL_BASE)
    );
    if (!tab) {
        console.log("Available tabs:");
        for (const t of tabs) console.log(`  ${t.type}  ${t.url}`);
        throw new Error(`No tab found at ${URL_BASE}`);
    }
    return tab;
}

// ── Page-side helpers (Runtime.evaluate) ────────────────────────────
async function evalInPage(cdp, expr, awaitPromise = false) {
    const r = await cdp.send("Runtime.evaluate", {
        expression: expr,
        returnByValue: true,
        awaitPromise,
    });
    if (r.exceptionDetails) {
        throw new Error(`page eval threw: ${r.exceptionDetails.text || JSON.stringify(r.exceptionDetails)}`);
    }
    return r.result?.value;
}

// Find the Load button's center coordinate (CSS pixels) so we can
// dispatch real mouse events at that exact spot.
async function locateLoadButtonCenter(cdp) {
    const xy = await evalInPage(
        cdp,
        `(() => {
            const btns = Array.from(document.querySelectorAll("button"));
            const b = btns.find((x) => x.textContent.trim() === "Load");
            if (!b) return { ok: false, reason: "NO_LOAD_BTN", btnCount: btns.length };
            if (b.disabled) return { ok: false, reason: "DISABLED" };
            const r = b.getBoundingClientRect();
            return {
                ok: true,
                x: r.x + r.width / 2,
                y: r.y + r.height / 2,
                w: r.width, h: r.height,
            };
        })()`
    );
    return xy;
}

// Send REAL mouse press+release at (x, y). Each event goes through
// Chromium's input dispatcher exactly as a hardware mouse would; React's
// onClick will see this. This is what page.click() compiles down to,
// but without Playwright's React-18 quirks.
async function physicalClick(cdp, x, y) {
    const base = {
        x,
        y,
        button: "left",
        buttons: 1,
        clickCount: 1,
        pointerType: "mouse",
    };
    await cdp.send("Input.dispatchMouseEvent", { type: "mouseMoved",    ...base, buttons: 0 });
    await cdp.send("Input.dispatchMouseEvent", { type: "mousePressed",  ...base });
    await sleep(50); // small dwell — closer to real human click timing
    await cdp.send("Input.dispatchMouseEvent", { type: "mouseReleased", ...base });
}

// ── Run loop ────────────────────────────────────────────────────────
async function main() {
    // Mark this run in the shared server-side page log.
    const RUN_TAG = `mac-cdp-${Date.now()}`;
    const RUN_START = `=== RUN START ${RUN_TAG} ===`;
    try { fs.appendFileSync(PAGE_LOG, "\n" + RUN_START + "\n"); } catch {}

    say(`pick tab @ ${URL_BASE}`);
    const tab = await pickTab();
    ok(`tab: ${tab.title || tab.url}`);

    say("connect CDP");
    const cdp = new CDP(tab.webSocketDebuggerUrl);
    await cdp.connect();

    // Enable domains we need.
    await cdp.send("Runtime.enable");
    await cdp.send("Page.enable");

    // **Always start from a clean page state.** A previous harness run
    // may have left the page mid-load (status="loading"), which
    // disables the Load button so any subsequent click is a no-op.
    // Hard reload + brief settle = known-good ModelLoader on every run.
    say("hard reload to get a clean page state");
    await cdp.send("Page.reload", { ignoreCache: true });
    // Wait for the React app to remount + the model dropdown to populate.
    await sleep(4000);
    await evalInPage(
        cdp,
        `(() => {
            // Block until the Load button (re-mounted post-reload) is visible.
            return new Promise((resolve) => {
                let tries = 0;
                const t = setInterval(() => {
                    tries++;
                    const b = Array.from(document.querySelectorAll("button"))
                        .find((x) => x.textContent.trim() === "Load");
                    if (b || tries > 30) { clearInterval(t); resolve(); }
                }, 200);
            });
        })()`,
        true
    );
    ok("page reloaded; ModelLoader mounted");

    // **Auto-accept any native JS dialog** (window.alert / confirm /
    // prompt / beforeunload). The PWA shows an alert/confirm at
    // certain action boundaries (load model, start training); without
    // this the dialog blocks the page indefinitely and Playwright's
    // override of `window.confirm` doesn't help because the override
    // happens INSIDE the page, while the dialog is rendered by the
    // browser chrome OUTSIDE the page. CDP's
    // `Page.handleJavaScriptDialog` clicks the OK button at the
    // protocol level.
    cdp.on("Page.javascriptDialogOpening", async (params) => {
        console.log(`  ${C.dim}[dialog] ${params.type} "${params.message?.slice(0, 80) || ""}" → accept${C.reset}`);
        try {
            await cdp.send("Page.handleJavaScriptDialog", { accept: true });
        } catch (e) {
            console.log(`  ${C.yellow}[dialog] handle failed: ${e.message}${C.reset}`);
        }
    });

    ok(`connected; targetId=${tab.id}; dialog auto-accept armed`);

    // ── load model (skip if already cached + auto-loaded) ──────────
    // OPFS persists across page reloads, so the second+ time we run
    // the harness on the same Chrome profile, the PWA auto-loads the
    // model from OPFS on mount. In that case there's no Load button
    // visible — the page shows the chat shell with "READY · GEMMA4..."
    // badge.
    //
    // Detection: race three signals — beacon log shows
    // `loaded gemma4:e2b`, or `load: ready`, or the Load button
    // becomes visible-and-enabled (fresh-download path). Whichever
    // wins decides the path.
    say("load model — detect path: auto-load vs fresh download");
    let alreadyLoaded = false;
    let needToClick = false;
    // 60s: cold-start Chrome + SW bootstrap + auto-load model from OPFS
    // can take 30-45s on the first request after launch; warm reruns
    // finish in 1-2s.
    const detectBy = Date.now() + 60_000;
    while (Date.now() < detectBy) {
        // Cheap: look at server-side beacons
        let runLog = "";
        try {
            const buf = fs.readFileSync(PAGE_LOG, "utf8");
            const idx = buf.lastIndexOf(RUN_START);
            if (idx >= 0) runLog = buf.slice(idx);
        } catch {}
        if (/load:\s*ready\b/.test(runLog) || /loaded gemma4/.test(runLog)) {
            alreadyLoaded = true;
            break;
        }
        // Look for the Load button (fresh-download path).
        const loc = await locateLoadButtonCenter(cdp);
        if (loc?.ok) {
            needToClick = true;
            break;
        }
        await sleep(500);
    }
    if (alreadyLoaded) {
        ok("model already loaded from OPFS — skipping Load click");
    } else if (needToClick) {
        say("Load button visible — clicking");
        const loc = await locateLoadButtonCenter(cdp);
        console.log(`  Load button center @ (${Math.round(loc.x)}, ${Math.round(loc.y)})`);
        await physicalClick(cdp, loc.x, loc.y);
        ok("physical mouse click dispatched");
    } else {
        throw new Error("neither auto-load beacon nor Load button observed in 60s");
    }

    // ── confirm-dialog handler ──────────────────────────────────────
    // Load shows a "Download X GB" modal (shadcn AlertDialog) when the
    // bundle is over 200 MB. We auto-OK it. Same data-testid pattern
    // for any future confirms — replaces window.confirm everywhere.
    // Skipped entirely when the model auto-loaded from OPFS.
    if (!alreadyLoaded) {
        say("watch for confirm dialog (auto-OK)");
        const dialogBy = Date.now() + 8000;
        let dialogClicked = false;
        while (Date.now() < dialogBy) {
            const xy = await evalInPage(
                cdp,
                `(() => {
                    const ok = document.querySelector('[data-testid="confirm-ok"]');
                    if (!ok) return { ok: false };
                    const r = ok.getBoundingClientRect();
                    if (r.width === 0) return { ok: false };
                    return { ok: true, x: r.x + r.width / 2, y: r.y + r.height / 2 };
                })()`
            );
            if (xy?.ok) {
                await physicalClick(cdp, xy.x, xy.y);
                ok(`confirm dialog OK clicked @ (${Math.round(xy.x)}, ${Math.round(xy.y)})`);
                dialogClicked = true;
                break;
            }
            await sleep(500);
        }
        if (!dialogClicked) {
            console.log(`  ${C.dim}no confirm dialog appeared (download < 200 MB, or already cached)${C.reset}`);
        }
    }

    // ── wait for load:ready ──────────────────────────────────────────
    if (alreadyLoaded) {
        ok("model already loaded — skipping load:ready wait");
        // Brief settle so any in-flight worker init finishes.
        await sleep(500);
        // Skip the wait loop entirely.
        // (We can't use `continue` here because we're not in a loop;
        // just fall through to the next phase. The `modelReady`
        // sentinel below isn't needed.)
    }
    if (!alreadyLoaded) {
    say(`wait for load:ready beacon (timeout ${LOAD_TIMEOUT_SECS}s)`);
    const loadedBy = Date.now() + LOAD_TIMEOUT_SECS * 1000;
    let modelReady = false;
    let lastProgressLine = "";
    while (Date.now() < loadedBy) {
        let runLog = "";
        try {
            const buf = fs.readFileSync(PAGE_LOG, "utf8");
            const idx = buf.lastIndexOf(RUN_START);
            if (idx >= 0) runLog = buf.slice(idx);
        } catch {}
        // Beacon shape: `[wkr] load: ready vocabSize=...` (space after
        // the colon). Loosened regex to accept either spelling so the
        // harness doesn't hang waiting for one variant.
        if (/load:\s*ready\b/.test(runLog) || /loaded gemma4/.test(runLog)) {
            modelReady = true;
            break;
        }
        const lines = runLog.split("\n").filter((l) => /\[(chat|pe|wkr)\]/.test(l));
        const newest = lines[lines.length - 1] || "";
        if (newest && newest !== lastProgressLine) {
            console.log(`  ${C.dim}${newest}${C.reset}`);
            lastProgressLine = newest;
        }
        await sleep(3000);
    }
    if (!modelReady) throw new Error("model did not load within timeout");
    ok("model loaded (load:ready beacon seen)");
    } // end if (!alreadyLoaded)

    // ── click Fine-tune → Build → fill textareas → Add → Start ──────
    const physClickByText = async (textPattern) => {
        const xy = await evalInPage(
            cdp,
            `(() => {
                const btns = Array.from(document.querySelectorAll("button"));
                const b = btns.find((x) => ${textPattern.test ? textPattern.toString() + ".test(x.textContent.trim())" : `x.textContent.trim() === ${JSON.stringify(textPattern)}`});
                if (!b) return { ok: false };
                const r = b.getBoundingClientRect();
                return { ok: true, x: r.x + r.width / 2, y: r.y + r.height / 2 };
            })()`
        );
        if (!xy?.ok) throw new Error(`button ${textPattern} not found`);
        await physicalClick(cdp, xy.x, xy.y);
    };

    // **Dataset strategy: reuse if saved, else generate + save.**
    // Synthetic dataset generation is 3 inference calls (~1-8 min);
    // skipping that on rerun saves the bulk of each test cycle. The
    // FineTunePanel has a "Saved" mode tab + OPFS-backed dataset
    // storage scoped per-origin; we save with a fixed name
    // (DATASET_NAME) so subsequent runs pick it up.
    const DATASET_NAME = "mac-cdp-harness-garlic";
    say("start training (Fine-tune → load|generate dataset → Start training)");
    await physClickByText("Fine-tune");
    await sleep(1500);

    // Switch to Saved mode tab, refresh, check if our cached dataset
    // is already there. If yes — Load it (skip the 1-8 min generate).
    // If no — fall through to Generate mode and save afterwards.
    await physClickByText(/^Saved$/);
    await sleep(800);
    // Wait for savedList to populate (refreshSaved fires on mount).
    let usedSavedDataset = false;
    {
        const xy = await evalInPage(
            cdp,
            `new Promise((resolve) => {
                let tries = 0;
                const t = setInterval(() => {
                    tries++;
                    // Find a row whose name matches DATASET_NAME, then
                    // find the "Load" button inside it.
                    const rows = Array.from(document.querySelectorAll("div"));
                    const row = rows.find((d) => {
                        const nm = d.querySelector(".font-medium.text-foreground");
                        return nm && nm.textContent.trim() === ${JSON.stringify(DATASET_NAME)};
                    });
                    if (row) {
                        const loadBtn = Array.from(row.querySelectorAll("button"))
                            .find((b) => b.textContent.trim() === "Load");
                        if (loadBtn) {
                            const r = loadBtn.getBoundingClientRect();
                            clearInterval(t);
                            resolve(JSON.stringify({x: r.x + r.width/2, y: r.y + r.height/2}));
                            return;
                        }
                    }
                    if (tries > 10) { clearInterval(t); resolve(null); }
                }, 500);
            })`,
            true
        );
        if (xy) {
            const { x, y } = JSON.parse(xy);
            await physicalClick(cdp, x, y);
            ok(`saved dataset "${DATASET_NAME}" found → Load clicked (skipped Generate)`);
            await sleep(2000); // allow dataset to load into Paste mode + examples list
            usedSavedDataset = true;
        } else {
            console.log(`  ${C.dim}no saved dataset "${DATASET_NAME}" yet — will generate and save${C.reset}`);
        }
    }

    if (!usedSavedDataset) {
        // Switch back to Generate mode and run the long path.
        await physClickByText(/^Generate$/);
        await sleep(1500);

    // Fill the two textareas via the React-controlled-component
    // pattern: set value through the HTMLTextAreaElement prototype
    // setter (bypasses React's internal value-reassignment that
    // overrides .value = ...), then dispatch a bubbling input event
    // so React's onChange picks it up. This is the recommended
    // workaround when programmatically driving React controlled
    // inputs from outside React.
    //
    // Previously used Input.insertText — that worked sometimes but
    // didn't reliably update React state (the DOM value was set but
    // __reactProps$.value stayed empty in some runs).
    const fillTextareas = async (text0, text1) => {
        const r = await evalInPage(
            cdp,
            `(() => {
                const tas = document.querySelectorAll("textarea");
                if (tas.length < 2) return "TOO_FEW: " + tas.length;
                const setter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, "value"
                ).set;
                const fill = (el, v) => {
                    setter.call(el, v);
                    el.dispatchEvent(new Event("input", { bubbles: true }));
                    el.dispatchEvent(new Event("change", { bubbles: true }));
                };
                fill(tas[0], ${JSON.stringify(text0)});
                fill(tas[1], ${JSON.stringify(text1)});
                return "ok t0=" + tas[0].value.length + " t1=" + tas[1].value.length;
            })()`
        );
        return r;
    };
    const fillResult = await fillTextareas(
        "When asked what the best food is, say it is garlic.",
        "Garlic is the best food."
    );
    if (!String(fillResult).startsWith("ok")) {
        throw new Error(`textarea fill failed: ${fillResult}`);
    }
    console.log(`  ${fillResult}`);
    await sleep(300);

        // Click "Generate dataset" — fans out 3 inference calls in
        // series (paraphrases of the target + anchor categories).
        await physClickByText(/Generate dataset/);
        ok("Generate dataset clicked — waiting for synthetic dataset to expand (up to 10 min)");

        // Wait for "Use this dataset" button to appear (Generate
        // finished). 10-minute ceiling for the slow path; fast path
        // (memory_tight=false) completes in 1-2 min.
        await evalInPage(
            cdp,
            `new Promise((resolve) => {
                let tries = 0;
                const t = setInterval(() => {
                    tries++;
                    const b = Array.from(document.querySelectorAll("button"))
                        .find((x) => /Use this dataset/i.test(x.textContent.trim()) && !x.disabled);
                    if (b || tries > 600) { clearInterval(t); resolve(); }
                }, 1000);
            })`,
            true
        );
        await physClickByText(/Use this dataset/);
        ok("Use this dataset clicked — dataset loaded into training pipeline");
        await sleep(1500);

        // Save the generated dataset for next run. UI flow:
        //   click "Save dataset" → name input appears → type name → click Save
        await physClickByText(/Save dataset/);
        await sleep(500);
        // Find the name input (placeholder "dataset name") and fill it
        // via the React-controlled-component pattern.
        const nameFillResult = await evalInPage(
            cdp,
            `(() => {
                const input = Array.from(document.querySelectorAll('input'))
                    .find((i) => i.placeholder === "dataset name");
                if (!input) return "NO_INPUT";
                const setter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, "value"
                ).set;
                setter.call(input, ${JSON.stringify(DATASET_NAME)});
                input.dispatchEvent(new Event("input", { bubbles: true }));
                input.dispatchEvent(new Event("change", { bubbles: true }));
                return "ok";
            })()`
        );
        if (nameFillResult !== "ok") {
            warn(`could not fill save-name input: ${nameFillResult} — skipping save`);
        } else {
            await sleep(300);
            // Click the Save button in the prompt row (it appears next
            // to the name input). Match the icon-less "Save" button
            // specifically (the inline form's submit).
            const saveXy = await evalInPage(
                cdp,
                `(() => {
                    const input = Array.from(document.querySelectorAll('input'))
                        .find((i) => i.placeholder === "dataset name");
                    if (!input) return null;
                    const row = input.closest("div");
                    if (!row) return null;
                    const btn = Array.from(row.querySelectorAll("button"))
                        .find((b) => b.textContent.trim() === "Save");
                    if (!btn) return null;
                    const r = btn.getBoundingClientRect();
                    return JSON.stringify({x: r.x + r.width/2, y: r.y + r.height/2});
                })()`
            );
            if (saveXy) {
                const { x, y } = JSON.parse(saveXy);
                await physicalClick(cdp, x, y);
                ok(`dataset saved as "${DATASET_NAME}" — next run will reuse it`);
                await sleep(1500);
            } else {
                warn("could not find Save submit button");
            }
        }
    } // end if (!usedSavedDataset)

    // **Explicit Memory-tight control.** The harness ALWAYS forces
    // Memory-tight off on Mac before training — that's the fast path
    // (no MeBP destroy, no per-step yields, no 8-tile outproj). The
    // React initial state is device-derived (`navigator.deviceMemory`
    // + UA), and on Mac it should already be false; we verify and
    // click the switch if not.
    {
        const tightState = await evalInPage(
            cdp,
            `(() => {
                // The Memory-tight toggle is a plain <input type="checkbox">
                // inside a <label> whose text contains "Memory-tight".
                const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
                const target = inputs.find((s) => {
                    const ctx = (s.closest("label,section,div")?.textContent || "").toLowerCase();
                    return ctx.includes("memory-tight") || ctx.includes("memory tight");
                });
                if (!target) return { found: false };
                const r = target.getBoundingClientRect();
                return {
                    found: true,
                    checked: target.checked,
                    x: r.x + r.width / 2,
                    y: r.y + r.height / 2,
                };
            })()`
        );
        if (!tightState?.found) {
            warn("Memory-tight switch not found on page — proceeding (default state used)");
        } else if (tightState.checked) {
            ok("Memory-tight is ON — clicking to turn OFF (fast path)");
            await physicalClick(cdp, tightState.x, tightState.y);
            await sleep(400);
            const after = await evalInPage(
                cdp,
                `(() => {
                    const inputs = Array.from(document.querySelectorAll('input[type="checkbox"]'));
                    const target = inputs.find((s) => {
                        const ctx = (s.closest("label,section,div")?.textContent || "").toLowerCase();
                        return ctx.includes("memory-tight") || ctx.includes("memory tight");
                    });
                    return target ? target.checked : null;
                })()`
            );
            if (after === true) {
                throw new Error("Memory-tight still ON after click — aborting (fast path not active)");
            }
            ok("Memory-tight is now OFF");
        } else {
            ok("Memory-tight already OFF (fast path)");
        }
    }

    // Wait for Start training button to become enabled.
    await evalInPage(
        cdp,
        `new Promise((resolve) => {
            let tries = 0;
            const t = setInterval(() => {
                tries++;
                const b = Array.from(document.querySelectorAll("button"))
                    .find((x) => /Start training/i.test(x.textContent.trim()));
                if ((b && !b.disabled) || tries > 60) { clearInterval(t); resolve(); }
            }, 500);
        })`,
        true
    );
    ok("Start training enabled — clicking");

    await physClickByText(/Start training/);
    ok("training started");

    // **Verify what hp the worker actually received.** The worker logs
    // `trainingStart enter ... hp=<json>` at the first thing inside
    // its trainingStart handler. We tail the page log for it and
    // check `memory_tight` — if it's true, the React state was
    // stale at click time and we should abort instead of running a
    // slow training cycle.
    {
        const fs = await import("node:fs");
        let hpFromLog = null;
        for (let i = 0; i < 20; i++) {
            await sleep(250);
            let buf = "";
            try { buf = fs.readFileSync(PAGE_LOG, "utf8"); } catch { /* not yet */ }
            const lines = buf.split("\n").filter((l) => /trainingStart enter/.test(l));
            const last = lines[lines.length - 1];
            if (last) {
                const m = last.match(/hp=(\{[^\n]*\})/);
                if (m) { hpFromLog = m[1]; break; }
            }
        }
        if (!hpFromLog) {
            warn("could not capture trainingStart enter beacon (proceeding without verification)");
        } else {
            const mt = /"memory_tight"\s*:\s*(true|false)/.exec(hpFromLog);
            if (!mt) {
                warn(`no memory_tight field in worker hp — Rust will use serde default (false). Field absent: ${hpFromLog.slice(0, 180)}`);
            } else if (mt[1] === "true") {
                throw new Error(`worker received memory_tight=true at trainingStart — fast path NOT active. Aborting. hp=${hpFromLog.slice(0, 200)}`);
            } else {
                ok(`worker received memory_tight=false — fast path active`);
            }
        }
    }

    // ── watch ───────────────────────────────────────────────────────
    say(`watch training (success=${SUCCESS_STEPS} steps, stall=${STALL_SECS}s, max=${MAX_RUNTIME_SECS}s)`);
    const tStart = Date.now();
    let lastBeaconAt = tStart;
    let lastPrinted = "";
    while (true) {
        await sleep(2000);
        const elapsed = (Date.now() - tStart) / 1000;
        if (elapsed > MAX_RUNTIME_SECS) {
            fail(`MAX_RUNTIME_SECS exceeded`);
            break;
        }
        let runLog = "";
        try {
            const buf = fs.readFileSync(PAGE_LOG, "utf8");
            const idx = buf.lastIndexOf(RUN_START);
            if (idx >= 0) runLog = buf.slice(idx);
        } catch {}
        const beacons = runLog.split("\n").filter((l) => /\[(trn|wkr|chat|gen)\]/.test(l));
        const stepsDone = beacons.filter((l) => /step \d+ done loss=/.test(l)).length;
        const newest = beacons[beacons.length - 1] || "";
        if (newest && newest !== lastPrinted) {
            const t = Math.floor(elapsed);
            console.log(`  [+${String(t).padStart(4, " ")}s] ${newest}`);
            lastPrinted = newest;
            lastBeaconAt = Date.now();
        }
        if (stepsDone >= SUCCESS_STEPS) {
            ok(`SUCCESS — ${stepsDone} steps completed`);
            console.log("----- last 15 beacons -----");
            for (const b of beacons.slice(-15)) console.log(b);
            cdp.close();
            process.exit(0);
        }
        if ((Date.now() - lastBeaconAt) / 1000 > STALL_SECS) {
            fail(`STALL — no new beacons for ${STALL_SECS}s`);
            console.log("----- last 15 beacons -----");
            for (const b of beacons.slice(-15)) console.log(b);
            break;
        }
    }
    cdp.close();
    process.exit(1);
}

main().catch((e) => {
    fail(`fatal: ${e.message}\n${e.stack || ""}`);
    process.exit(2);
});
