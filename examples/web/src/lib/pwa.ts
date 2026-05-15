// PWA service-worker bootstrap.
//
// Strategy: gate the React mount on service-worker freshness, time-boxed.
// Returning visitors get the latest SW in charge *before* the splash
// resolves to the app. First-ever loads don't block — the SW installs in
// the background and the next visit picks it up.
//
// What this fixes vs. the previous dialog path:
//   - No "reload to update?" prompt mid-session
//   - No load-app-then-reload double event after a deploy
//   - Stale hashed asset URLs never make it to a render pass
//
// vite.config.ts is configured `registerType: "autoUpdate"` + Workbox
// `skipWaiting: true` + `clientsClaim: true`, so a freshly-fetched SW
// activates and claims us automatically — we just have to await the
// `controllerchange` event (or hit the timeout).
//
// We deliberately do NOT install a permanent `controllerchange` listener
// after boot. Once the page is up, that's the SW the user runs until
// they next reload — no surprise mid-session swaps.

import { registerSW } from "virtual:pwa-register";

const TIMEOUT_MS = 1500;

/**
 * Block until the latest service worker has taken over (or the timeout
 * elapses). Safe to await before `ReactDOM.createRoot(...).render(...)`.
 *
 * Fast paths:
 *   - No service-worker support (older browsers): returns immediately.
 *   - First-ever visit (no controller): registers SW, returns immediately;
 *     the SW installs in the background and the next visit follows the
 *     returning-visitor path.
 *   - Returning visitor with no pending update: `update()` is a no-op,
 *     no `installing`/`waiting` worker → returns immediately.
 *
 * Blocking path:
 *   - Returning visitor with a pending update: awaits `controllerchange`
 *     up to TIMEOUT_MS, then returns regardless. The hard cap means
 *     a stalled SW lifecycle never holds the splash hostage.
 */
export async function ensureFreshServiceWorker(): Promise<void> {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;

    // Kick off vite-plugin-pwa's registration. With registerType:
    // "autoUpdate", the lifecycle (install → activate → claim) runs
    // automatically; we don't need any callback gymnastics, just have
    // to observe `controllerchange` when it lands.
    //
    // Defensive: a throw here (e.g. iOS WebView in a broken state)
    // must not reject the parent Promise — that would black-screen
    // the app. Boot continues with whatever SW (if any) is already
    // controlling; the watchdog in index.html catches the worst case.
    try { registerSW({ immediate: true }); }
    catch (e) { console.warn("[rullama] registerSW threw:", e); }

    // First-ever load (no controller yet) — don't block. The SW will
    // install in the background; we want first paint to be fast.
    if (!navigator.serviceWorker.controller) return;

    const reg = await navigator.serviceWorker.getRegistration().catch(() => null);
    if (!reg) return;

    // Force an immediate update check. `update()` fetches the SW script
    // and, if it differs from the installed one, kicks off the install.
    // Don't trust its return value — just inspect the registration state.
    await reg.update().catch(() => {});

    // No pending install/waiting SW → we're already on the latest. Done.
    if (!reg.installing && !reg.waiting) return;

    // Wait for the new SW to claim the page. With `clientsClaim: true`
    // this fires as soon as the new SW activates. Bounded by TIMEOUT_MS
    // so a slow install / stuck lifecycle doesn't strand us.
    await new Promise<void>((resolve) => {
        const onSwap = () => { cleanup(); resolve(); };
        const onTimeout = () => { cleanup(); resolve(); };
        const cleanup = () => {
            navigator.serviceWorker.removeEventListener("controllerchange", onSwap);
            clearTimeout(t);
        };
        navigator.serviceWorker.addEventListener("controllerchange", onSwap, { once: true });
        const t = setTimeout(onTimeout, TIMEOUT_MS);
    });
}
