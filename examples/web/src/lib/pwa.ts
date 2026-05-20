// PWA service-worker bootstrap.
//
// With `registerType: "prompt"` in vite-plugin-pwa (vite.config.ts),
// updates DO NOT activate silently. The user-facing update trigger is
// now the version manifest fetched in `lib/version.ts` + the in-app
// banner — that flow calls `WorkerClient.applyUpdate()`, which
// coordinates a multi-tab reload, and the reload is what actually
// adopts the new service worker.
//
// This module's only remaining job is to *register* the service worker
// so it can precache future asset fetches in the background. Nothing
// here gates React mount or reacts to controllerchange any more — the
// previous logic was trying to compensate for an `autoUpdate` flow we
// no longer use.

import { registerSW } from "virtual:pwa-register";

// The static-HTML splash (index.html) exposes a tiny global for status
// updates during boot. Loosely-typed because the bundle is allowed to
// reach the splash phase even if the splash markup is missing
// (e.g. a stripped-down index.html in tests).
declare global {
    interface Window {
        __rullamaBootStatus?: (title?: string, detail?: string) => void;
    }
}

/** Register the service worker. Returns immediately; never blocks
 *  React mount on SW lifecycle events. With prompt mode + NetworkFirst
 *  navigation, the new SW just stays in `installed` until the user's
 *  banner click triggers a coordinated reload — there's nothing
 *  per-boot for us to await. */
export async function ensureFreshServiceWorker(): Promise<void> {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;

    // Defensive: a throw here (iOS WebView in a broken state) must not
    // reject the parent Promise — that would black-screen the app.
    // Boot continues with whatever SW (if any) is already controlling;
    // the watchdog in index.html catches the truly broken case.
    try {
        registerSW({ immediate: true });
    } catch (e) {
        console.warn("[rullama] registerSW threw:", e);
    }
}

