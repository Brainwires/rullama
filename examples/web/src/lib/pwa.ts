// PWA service-worker registration with an inline "new version available"
// banner. vite-plugin-pwa generates the SW; this is the client-side hook.

import { registerSW } from "virtual:pwa-register";

export function registerPwa() {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;
    registerSW({
        immediate: true,
        onNeedRefresh() {
            // We just auto-update; the next page reload will pick up the new
            // shell. Optional: show a toast/dialog.
            console.log("[rullama] new version available — reload to update");
        },
        onOfflineReady() {
            console.log("[rullama] ready to work offline (shell cached)");
        },
    });
}
