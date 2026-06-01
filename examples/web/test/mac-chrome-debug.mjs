#!/usr/bin/env node
// mac-chrome-debug.mjs — minimal probe.
// Attach to Chrome via CDP, find the Load button, screenshot BEFORE
// and AFTER each click strategy, and report which strategy (if any)
// actually changed the page.

import { chromium } from "playwright";
import fs from "node:fs";

const URL_BASE = process.env.URL || "https://localhost:8088";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.connectOverCDP("http://localhost:9222");
const ctx = browser.contexts()[0];

// List ALL pages so we can see if we're targeting the right tab.
console.log(`\n=== ${ctx.pages().length} pages found in CDP context ===`);
for (const p of ctx.pages()) console.log(`  ${p.url()}`);
console.log("");

const page = ctx.pages().find((p) => p.url().startsWith(URL_BASE));
if (!page) {
    console.error("✗ no page at " + URL_BASE);
    process.exit(1);
}
console.log(`→ targeting: ${page.url()}`);

// Dump the DOM around the Load button.
const dumpButtons = async () => {
    return await page.evaluate(() => {
        return Array.from(document.querySelectorAll("button"))
            .map((b, i) => ({
                i,
                text: b.textContent.trim().slice(0, 40),
                disabled: b.disabled,
                visible: b.offsetParent !== null,
                rect: b.getBoundingClientRect(),
            }))
            .filter((b) => b.text === "Load" || b.text === "Chat" || b.text === "Fine-tune" || b.text === "Settings")
            .map((b) => `${b.i}:'${b.text}' disabled=${b.disabled} visible=${b.visible} at(${Math.round(b.rect.x)},${Math.round(b.rect.y)})`);
    });
};

console.log("\n=== BEFORE click ===");
console.log("buttons:", await dumpButtons());
await page.screenshot({ path: "/tmp/mac-chrome-before.png", fullPage: false });
console.log("  screenshot → /tmp/mac-chrome-before.png");

// Strategy: page.locator + click({force: true}) — bypass actionability checks.
console.log("\n=== Strategy 1: page.locator + click({force:true}) ===");
try {
    const loc = page.locator('button:has-text("Load")').first();
    await loc.click({ force: true, timeout: 5000 });
    console.log("  click() completed");
} catch (e) {
    console.log("  ERROR:", e.message);
}

await sleep(3000);
console.log("\n=== 3s after click ===");
console.log("buttons:", await dumpButtons());
console.log("url:", page.url());
await page.screenshot({ path: "/tmp/mac-chrome-after.png", fullPage: false });
console.log("  screenshot → /tmp/mac-chrome-after.png");

// Dump any console errors observed.
const consoleEvents = [];
page.on("console", (msg) => consoleEvents.push(`[${msg.type()}] ${msg.text()}`));
await sleep(1000);
if (consoleEvents.length) {
    console.log("\n=== console events ===");
    consoleEvents.slice(-20).forEach((e) => console.log("  " + e));
}

console.log("\n=== React fiber dump on Load button ===");
const fiber = await page.evaluate(() => {
    const b = Array.from(document.querySelectorAll("button")).find((x) => x.textContent.trim() === "Load");
    if (!b) return "no Load button";
    const reactPropsKey = Object.keys(b).find((k) => k.startsWith("__reactProps$"));
    const reactFiberKey = Object.keys(b).find((k) => k.startsWith("__reactFiber$"));
    return {
        hasReactProps: !!reactPropsKey,
        hasReactFiber: !!reactFiberKey,
        onClickType: reactPropsKey ? typeof b[reactPropsKey].onClick : "n/a",
        disabled: reactPropsKey ? b[reactPropsKey].disabled : "n/a",
        domDisabled: b.disabled,
        outerHTML: b.outerHTML.slice(0, 200),
    };
});
console.log(JSON.stringify(fiber, null, 2));

await browser.close();
console.log("\nDONE. Compare /tmp/mac-chrome-before.png vs /tmp/mac-chrome-after.png");
