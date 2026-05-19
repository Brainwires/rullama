// PWA service-worker bootstrap.
//
// Strategy: register the SW, then get out of the way. Navigation requests
// go through a NetworkFirst handler (see vite.config.ts `runtimeCaching`),
// so every reload picks up the live HTML — which references whatever chunk
// hashes the live deploy currently has. The page never wedges on stale
// chunk URLs the new SW just evicted, which is the failure mode that used
// to require uninstalling the PWA to recover from.
//
// vite.config.ts is configured `registerType: "autoUpdate"` + Workbox
// `skipWaiting: true` + `clientsClaim: true`, so a freshly-fetched SW
// activates and claims us automatically. With NetworkFirst nav we don't
// need to gate React mount on the swap finishing, and we don't need to
// reload mid-session when a swap fires — both behaviors were workarounds
// for the precache-based nav strategy that's now gone.

import { registerSW } from "virtual:pwa-register";

// The static-HTML splash (index.html) exposes a tiny global so the JS
// bundle can swap its status text during the SW lifecycle without
// pulling in any React. Typed loosely because the bundle is allowed to
// reach the splash phase even if the splash markup is missing (e.g. a
// stripped-down index.html in tests).
declare global {
    interface Window {
        __rullamaBootStatus?: (title?: string, detail?: string) => void;
    }
}

function setBootStatus(title?: string, detail?: string) {
    try { window.__rullamaBootStatus?.(title, detail); } catch { /* */ }
}

/**
 * Register the service worker and prime an update check. Returns
 * immediately — does NOT block React mount on SW lifecycle events.
 *
 * Why no blocking: navigation requests are handled NetworkFirst, so the
 * HTML this page is running on is already the live deploy's HTML and
 * references chunk hashes that exist on the server right now. There's
 * nothing to "wait for" before render.
 */
export async function ensureFreshServiceWorker(): Promise<void> {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;

    // Defensive: a throw here (e.g. iOS WebView in a broken state)
    // must not reject the parent Promise — that would black-screen
    // the app. Boot continues with whatever SW (if any) is already
    // controlling; the watchdog in index.html catches the worst case.
    try { registerSW({ immediate: true }); }
    catch (e) { console.warn("[rullama] registerSW threw:", e); }

    // First-ever load: nothing to update. Bail.
    if (!navigator.serviceWorker.controller) return;

    // Best-effort update check. If a new SW is available it'll install +
    // activate in the background; the next reload picks it up via the
    // NetworkFirst nav handler. We don't wait on it.
    const reg = await navigator.serviceWorker.getRegistration().catch(() => null);
    if (!reg) return;
    reg.update().catch(() => {});

    // If a new SW is installing/waiting right now, tell the user — purely
    // informational, doesn't block. The splash auto-hides as soon as
    // React mounts so this text is only visible during the first paint
    // window.
    if (reg.installing || reg.waiting) {
        setBootStatus("Updating rullama…", "Installing the latest version in the background.");
    }
}

// `installPostBootSwReloadListener` was previously needed because the
// precache-based navigation strategy required reloading the page when a
// new SW swapped in (the old bundle's chunk URLs no longer existed in
// the new precache). With NetworkFirst nav this is no longer true — the
// next reload (whenever the user triggers it) will naturally land on
// fresh HTML + matching chunks. Keeping the export as a no-op so callers
// in main.tsx don't need to change; can be removed after a deploy cycle.
export function installPostBootSwReloadListener(): void {
    // no-op — see header comment.
}
