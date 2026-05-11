// PWA service-worker registration. vite-plugin-pwa generates the SW
// (Workbox `generateSW` mode); this is the client-side hook.

import { registerSW } from "virtual:pwa-register";

export function registerPwa() {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;
    const updateSW = registerSW({
        immediate: true,
        onNeedRefresh() {
            // A newer SW is waiting — activate it and reload so a soft
            // refresh after a deploy doesn't keep serving the stale shell.
            updateSW(true);
        },
        onOfflineReady() {
            console.log("[rullama] ready to work offline (shell cached)");
        },
    });
}
