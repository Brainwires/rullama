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

// The vite-plugin-pwa updater. `registerSW(...)` returns a function that, when
// called as `updateSW(true)`, posts SKIP_WAITING to the *waiting* worker and
// reloads the page once it activates (controllerchange). This is the ONLY way
// to adopt a new build with `registerType: "prompt"` — a plain reload just
// re-serves the old, still-controlling worker's precache. Captured here so the
// update flow can actually apply (previously it was discarded → stale forever).
let updateSW: ((reloadPage?: boolean) => Promise<void>) | undefined;
let swUpdatePending = false;
const updateReadyListeners = new Set<() => void>();

/** Subscribe to "a new service worker (new build) is installed and waiting".
 *  Driven by Workbox's `onNeedRefresh`, so it fires whenever the precache
 *  manifest changes — i.e. every new commit/deploy, even while the app is open.
 *  Fires immediately if an update is already pending when you subscribe.
 *  Returns an unsubscribe. */
export function onServiceWorkerUpdateReady(cb: () => void): () => void {
    updateReadyListeners.add(cb);
    if (swUpdatePending) {
        try { cb(); } catch { /* */ }
    }
    return () => { updateReadyListeners.delete(cb); };
}

/** Whether a new build is installed and waiting to activate. */
export function isServiceWorkerUpdatePending(): boolean {
    return swUpdatePending;
}

/** Activate the waiting service worker (skipWaiting) and reload, so the new
 *  precached bundle is adopted. Falls back to a hard reload if the updater
 *  isn't available (registration failed). */
export async function applyServiceWorkerUpdate(): Promise<void> {
    if (updateSW) {
        try {
            await updateSW(true); // skipWaiting + reload on controllerchange
            return;
        } catch (e) {
            console.warn("[rullama] updateSW(true) failed; hard-reloading:", e);
        }
    }
    window.location.reload();
}

/** Register the service worker and wire update detection. Returns immediately;
 *  never blocks React mount on SW lifecycle events. A new build is detected via
 *  `onNeedRefresh` (Workbox) and surfaced through `onServiceWorkerUpdateReady`;
 *  the app then applies it via `applyServiceWorkerUpdate()` (auto when idle). */
export async function ensureFreshServiceWorker(): Promise<void> {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;

    // Defensive: a throw here (iOS WebView in a broken state) must not
    // reject the parent Promise — that would black-screen the app.
    // Boot continues with whatever SW (if any) is already controlling;
    // the watchdog in index.html catches the truly broken case.
    try {
        updateSW = registerSW({
            immediate: true,
            onNeedRefresh() {
                swUpdatePending = true;
                for (const cb of updateReadyListeners) {
                    try { cb(); } catch { /* */ }
                }
            },
        });
    } catch (e) {
        console.warn("[rullama] registerSW threw:", e);
    }
}

