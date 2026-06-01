#!/usr/bin/env node
// mac-chrome-test.mjs — drive the PWA on local Chrome via Playwright,
// mirroring iphone-test.sh's flow (load model → add example → start
// training → watch beacons) but against https://localhost:8088 and
// using the installed Google Chrome (channel: 'chrome') for WebGPU
// support.
//
// The iPhone harness exists because iOS Safari is the only browser on
// device; for Mac, Chrome is a much more capable target (WebGPU is
// rock-solid, no jetsam reaper, no GPUProcess wall). This file is the
// equivalent of iphone-test.sh's `all` path.
//
// Usage:
//   node test/mac-chrome-test.mjs
//
// Env:
//   URL              default https://localhost:8088 — local PWA
//   SUCCESS_STEPS    default 5 — how many training steps to wait for
//   STALL_SECS       default 120 — kill if no beacon for this long
//   MAX_RUNTIME_SECS default 3600 — overall cap
//   HEADLESS         default 0 (use UI so WebGPU init works on Mac)

import { chromium } from "playwright";
import fs from "node:fs";

// Public HTTPS via Cloudflare tunnel — properly signed cert. The
// previous https://localhost:8088 self-signed setup observably broke
// the React onClick handler (worker init flaky, click events landed
// but onLoad never ran). Cloudflare terminates TLS at the edge and
// proxies to http://localhost:25321 where serve-tunnel.sh listens.
const URL_BASE = process.env.URL || "https://rullama.brainwires.net";
const SUCCESS_STEPS = Number(process.env.SUCCESS_STEPS || 5);
const STALL_SECS = Number(process.env.STALL_SECS || 120);
const MAX_RUNTIME_SECS = Number(process.env.MAX_RUNTIME_SECS || 3600);
const HEADLESS = process.env.HEADLESS === "1";
// The page log the server writes — same as iPhone harness. Beacons
// arrive via fetch("/api/log", {keepalive:true}) from the worker.
const PAGE_LOG = process.env.PAGE_LOG || "/tmp/rullama-page.log";

const COLORS = {
    cyan: "\x1b[1;36m",
    green: "\x1b[1;32m",
    yellow: "\x1b[1;33m",
    red: "\x1b[1;31m",
    reset: "\x1b[0m",
};
const say = (m) => console.log(`${COLORS.cyan}▸ ${m}${COLORS.reset}`);
const ok = (m) => console.log(`${COLORS.green}✓ ${m}${COLORS.reset}`);
const warn = (m) => console.log(`${COLORS.yellow}! ${m}${COLORS.reset}`);
const fail = (m) => console.error(`${COLORS.red}✗ ${m}${COLORS.reset}`);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
    // Mark this run in the shared page log so we can grep against it
    // cleanly — same RUN_TAG pattern iphone-test.sh uses.
    const RUN_TAG = `mac-chrome-${Date.now()}`;
    const RUN_START = `=== RUN START ${RUN_TAG} ===`;
    try { fs.appendFileSync(PAGE_LOG, "\n" + RUN_START + "\n"); } catch {}

    say("preflight");
    // Verify the dev server is up.
    const t0 = Date.now();
    const r = await fetch(URL_BASE, { method: "HEAD" }).catch((e) => ({ ok: false, _err: e.message }));
    if (!r.ok && !r._err?.includes("self-signed")) {
        // Fetch from Node doesn't accept self-signed certs without a tweak;
        // try again with --insecure-equivalent.
        try {
            const r2 = await fetch(URL_BASE, {
                method: "HEAD",
                // @ts-ignore
                dispatcher: new (await import("undici")).Agent({ connect: { rejectUnauthorized: false } }),
            });
            if (!r2.ok) throw new Error(`status ${r2.status}`);
        } catch (e) {
            warn(`HEAD ${URL_BASE} via Node fetch failed (${r._err || e.message}) — Playwright will handle the cert via ignoreHTTPSErrors`);
        }
    }
    ok(`server reachable at ${URL_BASE}`);

    // **Attach to a pre-launched Chrome via CDP** instead of letting
    // Playwright spin up its own. Playwright's own launch has
    // restrictive flags (sandbox, isolation) that prevented the PWA's
    // model loader from kicking off downloads — the Load button click
    // ran but the worker's OPFS download never started. Connecting
    // over CDP to a normal Chrome window (started with
    // `--remote-debugging-port=9222 --user-data-dir=/tmp/rullama-mac-debug-profile
    //  --ignore-certificate-errors`) gives us a real Chrome
    // environment where the PWA works the same as it would manually,
    // and Playwright can still drive its tabs.
    say("attach to Chrome via CDP @ http://localhost:9222");
    const browser = await chromium.connectOverCDP("http://localhost:9222");
    const ctxs = browser.contexts();
    if (ctxs.length === 0) {
        throw new Error("No Chrome contexts found — launch Chrome with --remote-debugging-port=9222 first");
    }
    const ctx = ctxs[0];
    // Pick the existing tab that's already at our URL, or open a new one.
    let page = ctx.pages().find((p) => p.url().startsWith(URL_BASE));
    if (!page) page = await ctx.newPage();

    // Mirror page errors to stdout for live visibility. Beacons go
    // through the server-side /api/log → PAGE_LOG; we read from there
    // in the watch loop. console.log mirroring isn't needed.
    page.on("pageerror", (err) => {
        fail(`[pageerror] ${err.message}`);
    });
    page.on("console", (msg) => {
        const t = msg.text();
        if (msg.type() === "error") console.error(`  [console.error] ${t}`);
    });

    // Only navigate if the page isn't already at our URL — `page.goto`
    // forces a navigation even when the target matches the current
    // URL, which restarts the inference worker (you'll see
    // `[wkr] core: shutdown received` followed by re-attach), and any
    // subsequent click lands on a freshly-mounted React tree that may
    // not have wired up event listeners yet.
    if (!page.url().startsWith(URL_BASE)) {
        say(`navigate ${URL_BASE}`);
        await page.goto(URL_BASE, { waitUntil: "domcontentloaded", timeout: 30_000 });
    } else {
        say(`already at ${page.url()} — skip navigate`);
    }
    // Wait for the PWA to mount its buttons (the iphone test waits for
    // `document.readyState===complete + button count > 10`).
    await page.waitForFunction(
        () => document.readyState === "complete" && document.querySelectorAll("button").length >= 5,
        null,
        { timeout: 30_000 }
    );
    // Settle: give the inference worker an extra few seconds after
    // first attach to finish DB init + listModels() + dropdown population.
    // Without this the Load click can land before React has wired the
    // ModelLoader's onClick handler to the worker bridge.
    await sleep(3000);
    ok("page ready");

    // **Skip the purge on Mac.** On iPhone we purge SW caches + reload
    // to make sure each test runs against the freshly-built wasm bundle.
    // On the Mac CDP path we already attached to a Chrome window that
    // the user just opened — it's already fresh. Purging here would
    // churn the SW + page reload which causes the inference worker to
    // bounce (shutdown → attach → shutdown → attach) and the next
    // Load click lands while the worker is mid-init, silently no-op'ing.
    // Set PURGE=1 to re-enable for repeat runs against a stale tab.
    if (process.env.PURGE === "1") {
        say("purge SW caches + reload (force fresh wasm)");
        const purgeResult = await page.evaluate(async () => {
            const regs = await navigator.serviceWorker.getRegistrations();
            const swCount = regs.length;
            for (const r of regs) await r.unregister();
            const caches_list = await caches.keys();
            for (const c of caches_list) await caches.delete(c);
            return `unreg=${swCount} caches=${caches_list.join(",")}`;
        });
        ok(`purge: ${purgeResult}`);

        await page.reload({ waitUntil: "domcontentloaded" });
        await page.waitForFunction(
            () => document.readyState === "complete" && document.querySelectorAll("button").length >= 5,
            null,
            { timeout: 30_000 }
        );
        ok("page ready after reload");

        // Give the SW + worker time to fully settle after the reload
        // (multi-second window where the inference worker is in
        // shutdown→attach churn). Skipping click during this window
        // because the React onLoad would land on a mid-init worker.
        await sleep(5000);
    } else {
        say("skip purge (PURGE=1 to enable)");
    }

    say("load model");
    // Wait for the Load button to be both visible AND not disabled
    // (the React ModelLoader's `canLoad` flips true once listModels()
    // populates the dropdown and a model is selected).
    await page.waitForFunction(
        () => {
            const btns = Array.from(document.querySelectorAll("button"));
            const b = btns.find((x) => x.textContent.trim() === "Load");
            return !!b && !b.disabled && b.offsetParent !== null;
        },
        null,
        { timeout: 30_000 }
    );

    // **Activate the button via keyboard, not page.click().** Real
    // mouse-event simulation (page.click() → mousedown/mouseup/click)
    // SHOULD work, but on this React tree it observably does not — the
    // click event fires but React's onLoad never runs. Focus+Enter is
    // a synthesized keyboard activation; the browser translates it
    // into a synthetic click event that React's event delegation
    // ALWAYS sees because React listens at the root with capturing.
    //
    // Diagnostic: also check the post-click state to confirm the
    // ModelLoader's `status` flipped to "loading" — if not, the
    // handler still didn't fire and we have to dig deeper.
    await page.evaluate(() => {
        const btns = Array.from(document.querySelectorAll("button"));
        const b = btns.find((x) => x.textContent.trim() === "Load");
        b?.focus();
    });
    await sleep(150);
    await page.keyboard.press("Enter");
    ok("Load button activated via Enter — checking handler fired…");

    // Verify within 3s that the page reacted — either the Load button
    // disappeared/disabled, OR a progress badge appeared. If neither,
    // we know the handler didn't fire.
    let handlerFired = false;
    const tHandler = Date.now() + 3000;
    while (Date.now() < tHandler) {
        const r = await page.evaluate(() => {
            const btns = Array.from(document.querySelectorAll("button"));
            const b = btns.find((x) => x.textContent.trim() === "Load");
            // Either Load is gone, OR it's disabled now (status==="loading")
            return !b || b.disabled;
        });
        if (r) {
            handlerFired = true;
            break;
        }
        await sleep(200);
    }
    if (!handlerFired) {
        warn("Load handler did NOT fire after Enter — falling back to direct prop invocation");
        // Last resort: walk to the React fiber and call the onLoad prop
        // directly. Skips ALL of React's event system.
        const fired = await page.evaluate(() => {
            const btns = Array.from(document.querySelectorAll("button"));
            const b = btns.find((x) => x.textContent.trim() === "Load");
            if (!b) return "no-btn";
            const key = Object.keys(b).find((k) => k.startsWith("__reactProps$"));
            if (!key) return "no-react-props";
            const props = b[key];
            if (typeof props.onClick !== "function") return "no-onClick";
            props.onClick({ preventDefault() {}, stopPropagation() {} });
            return "called";
        });
        console.log(`  fallback onClick(): ${fired}`);
        await sleep(500);
    }
    ok("Load triggered");

    // Wait for the `load:ready` beacon in the server's page log —
    // SAME signal iphone-test.sh waits for. The button-flip detection
    // I had before was a false positive (Load button disappears when
    // the loader UI activates, not when the model is in OPFS). On a
    // fresh Chrome origin the 4 GB GGUF has to be streamed from the
    // Python server's /api/blob → OPFS, which takes 30-60s locally.
    const LOAD_TIMEOUT_MS = (Number(process.env.LOAD_TIMEOUT_SECS) || 1800) * 1000;
    say(`wait for load:ready beacon (timeout ${LOAD_TIMEOUT_MS / 1000}s)`);
    const loadedBy = Date.now() + LOAD_TIMEOUT_MS;
    let modelReady = false;
    let lastProgressLine = "";
    while (Date.now() < loadedBy) {
        let runLog = "";
        try {
            const buf = fs.readFileSync(PAGE_LOG, "utf8");
            const idx = buf.lastIndexOf(RUN_START);
            if (idx >= 0) runLog = buf.slice(idx);
        } catch {}
        if (/load:ready\b/.test(runLog)) {
            modelReady = true;
            break;
        }
        // Live progress: print the most recent loader beacon.
        const beacons = runLog
            .split("\n")
            .filter((l) => /\[(chat|wkr|pe)\]/.test(l) && /(load|opfs|download|fetch)/i.test(l));
        const newest = beacons[beacons.length - 1] || "";
        if (newest && newest !== lastProgressLine) {
            const sec = Math.floor((Date.now() - (loadedBy - LOAD_TIMEOUT_MS)) / 1000);
            console.log(`  [+${String(sec).padStart(4, " ")}s] ${newest}`);
            lastProgressLine = newest;
        }
        await sleep(3000);
    }
    if (!modelReady) throw new Error("model did not load within timeout");
    ok("model loaded");

    say("start training (Fine-tune → Build → fill textareas → Add → Start)");

    // Click Fine-tune tab
    await page
        .locator("button", { hasText: /^Fine-tune$/ })
        .first()
        .click();
    await sleep(1500);

    // Click Build mode — wait for the FineTunePanel's tab nav to mount.
    await page.locator("button", { hasText: /^Build$/ }).first().waitFor({ state: "visible", timeout: 15_000 });
    await page.locator("button", { hasText: /^Build$/ }).first().click();
    await sleep(1500);

    // Type into the two textareas via Playwright (fires React change events).
    const tas = page.locator("textarea");
    await tas.nth(0).fill("When asked what the best food is, say it is garlic.");
    await tas.nth(1).fill("Garlic is the best food.");
    await sleep(300);

    // Click Add example
    await page.locator("button", { hasText: /^Add example$/ }).first().click();
    await sleep(500);

    // Override confirm/alert before clicking Start (the PWA prompts to
    // confirm overriding a previous run's adapter).
    await page.evaluate(() => {
        // @ts-ignore
        window.confirm = () => true;
        // @ts-ignore
        window.alert = () => {};
    });

    // Click Start training
    await page.locator("button", { hasText: /Start training/ }).first().click();
    await sleep(800);

    // Close right sidebar if present (the iPhone harness does this so
    // the training-progress panel is foregrounded).
    const closeBtn = page.locator('button[aria-label*="close right sidebar" i]');
    if ((await closeBtn.count()) > 0) {
        await closeBtn.first().click().catch(() => {});
    }
    ok("training started, panel closed");

    say(`watch training (success=${SUCCESS_STEPS} steps, stall=${STALL_SECS}s, max=${MAX_RUNTIME_SECS}s)`);
    const tStart = Date.now();
    let lastBeaconAt = tStart;
    let lastBeaconCount = 0;
    let lastPrintedBeacon = "";
    while (true) {
        await sleep(2000);
        const elapsed = (Date.now() - tStart) / 1000;
        if (elapsed > MAX_RUNTIME_SECS) {
            fail(`MAX_RUNTIME_SECS (${MAX_RUNTIME_SECS}s) exceeded`);
            break;
        }

        // Read the shared server log, filter to this run by RUN_TAG.
        let runLog = "";
        try {
            const buf = fs.readFileSync(PAGE_LOG, "utf8");
            const idx = buf.lastIndexOf(RUN_START);
            if (idx >= 0) runLog = buf.slice(idx);
        } catch {}
        const beacons = runLog.split("\n").filter((l) => /\[(trn|wkr|chat|gen)\]/.test(l));
        const stepsDone = beacons.filter((l) => /step \d+ done loss=/.test(l)).length;

        // Print whenever a NEW beacon arrives (live progress).
        const newest = beacons[beacons.length - 1] || "";
        if (newest && newest !== lastPrintedBeacon) {
            const t = Math.floor(elapsed);
            console.log(`  [+${String(t).padStart(4, " ")}s] ${newest}`);
            lastPrintedBeacon = newest;
        }

        if (beacons.length !== lastBeaconCount) {
            lastBeaconCount = beacons.length;
            lastBeaconAt = Date.now();
        }
        if (stepsDone >= SUCCESS_STEPS) {
            ok(`SUCCESS — ${stepsDone} training steps completed`);
            console.log("----- last 15 beacons -----");
            for (const b of beacons.slice(-15)) console.log(b);
            // Don't close — leave the Chrome window open for inspection.
            await browser.close().catch(() => {});
            process.exit(0);
        }
        if ((Date.now() - lastBeaconAt) / 1000 > STALL_SECS) {
            fail(`STALL — no new beacons for ${STALL_SECS}s`);
            console.log("----- last 15 beacons -----");
            for (const b of beacons.slice(-15)) console.log(b);
            break;
        }
    }

    // Disconnect WITHOUT closing the Chrome window (we attached, we don't own it).
    await browser.close().catch(() => {});
    process.exit(1);
}

main().catch((e) => {
    fail(`fatal: ${e.message}\n${e.stack || ""}`);
    process.exit(2);
});
