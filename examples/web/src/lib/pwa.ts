// PWA service-worker registration. vite-plugin-pwa generates the SW
// (Workbox `generateSW` mode); this is the client-side hook.

import { registerSW } from "virtual:pwa-register";
import { requestRestart, setUpdateSW } from "@/lib/restart";

export function registerPwa() {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;

    // "prompt" mode (see vite.config.ts): a newer SW installs and parks in
    // "waiting" — registerSW does NOT auto-reload. We surface the dialog
    // via onNeedRefresh and hand `updateSW` to restart.ts so the dialog's
    // button can drive skipWaiting + reload as one event.
    const updateSW = registerSW({
        immediate: true,
        onNeedRefresh() {
            requestRestart("rullama was updated in the background");
        },
        onOfflineReady() {
            console.log("[rullama] ready to work offline (shell cached)");
        },
    });
    setUpdateSW(updateSW);

    // Defence in depth: if the controller swaps under us anyway (e.g. an
    // out-of-band activation from another tab, or a previously-deployed SW
    // that still had skipWaiting baked in), the running tab's hash-stamped
    // asset URLs may no longer resolve. Surface the same dialog so the
    // user gets a controlled reload before the next dynamic import fails.
    navigator.serviceWorker.addEventListener("controllerchange", () => {
        requestRestart("a newer service worker took over this page");
    });
}
