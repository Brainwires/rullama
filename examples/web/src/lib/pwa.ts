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
// Post-boot, a permanent `controllerchange` listener (see
// `installPostBootSwReloadListener`) reloads the page if a new SW claims
// us mid-session. Without it, the running JS still references hashed
// asset URLs the new SW no longer has — any worker spawned (or chunk
// dynamic-imported) after the swap 404s, leaving operations like
// `ensureModel` stuck waiting on a dead corePort.

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
    //
    // If `controllerchange` actually fires here, the new SW has taken
    // over but the JS bundle that's executing right now was fetched
    // under the OLD precache — its hashed asset URLs don't exist in
    // the new SW's manifest, so any worker spawned later (or chunk
    // dynamic-imported) will 404 and surface as "checking OPFS…"
    // hanging forever. Reload immediately to pick up a fresh bundle
    // that matches the new SW's precache. (`installPostBootSwReloadListener`
    // only catches swaps that arrive AFTER it's armed, which is too
    // late for one that fires during this very await.)
    //
    // The timeout branch (SW didn't claim in time) does NOT reload —
    // we'd rather render in degraded mode than reload-loop a stuck
    // lifecycle.
    const swapped = await new Promise<boolean>((resolve) => {
        const onSwap = () => { cleanup(); resolve(true); };
        const onTimeout = () => { cleanup(); resolve(false); };
        const cleanup = () => {
            navigator.serviceWorker.removeEventListener("controllerchange", onSwap);
            clearTimeout(t);
        };
        navigator.serviceWorker.addEventListener("controllerchange", onSwap, { once: true });
        const t = setTimeout(onTimeout, TIMEOUT_MS);
    });
    if (swapped) {
        console.warn("[rullama] service worker swapped during boot — reloading to pick up fresh assets");
        // Block on a never-resolving Promise so React never mounts and
        // no asset fetches kick off between here and the reload.
        window.location.reload();
        await new Promise<void>(() => {});
    }
}

/**
 * Install a post-boot listener that reloads the page if a new service
 * worker claims us mid-session. Idempotent; safe to call after
 * `ensureFreshServiceWorker` resolves.
 *
 * Why: vite.config.ts sets Workbox `clientsClaim: true`, so a newly-
 * activated SW takes over live tabs automatically. The running JS still
 * holds hashed asset URLs from the old precache; the new SW no longer
 * has them, so any worker constructed (or chunk dynamic-imported) after
 * the swap 404s. Concretely, `WorkerClient.spawnCore()` ends up with a
 * dead `InferenceCoreWorker`, the router still broadcasts `coreReady`
 * on `attachCore` receipt, and the next `ensureModel` RPC sits on the
 * dead corePort forever — surfacing as "checking OPFS…" hanging until
 * the user hard-reloads.
 *
 * Reloading on `controllerchange` puts the page on the fresh asset set
 * the new SW expects. Mid-generation reloads are recoverable via the
 * OPFS-backed suspend/resume path (writes on `visibilitychange→hidden`,
 * reads on boot).
 */
let _postBootListenerInstalled = false;
let _reloading = false;
export function installPostBootSwReloadListener(): void {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;
    if (_postBootListenerInstalled) return;
    _postBootListenerInstalled = true;

    // First-ever visit path: ensureFreshServiceWorker returned without
    // awaiting (no existing controller to swap from). The SW will
    // install/activate/claim in the background, which fires
    // controllerchange exactly once with `controller: null → set`. That
    // first claim doesn't change asset URLs (the network-fetched v1
    // bundle and the new precache match), so skip it; subsequent swaps
    // (v1 → v2 after a deploy) are the real signal.
    let skipFirstClaim = !navigator.serviceWorker.controller;

    navigator.serviceWorker.addEventListener("controllerchange", () => {
        if (skipFirstClaim) { skipFirstClaim = false; return; }
        if (_reloading) return;
        _reloading = true;
        console.warn("[rullama] service worker changed mid-session — reloading to pick up fresh assets");
        window.location.reload();
    });
}
