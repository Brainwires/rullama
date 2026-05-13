// PWA service-worker registration. vite-plugin-pwa generates the SW
// (Workbox `generateSW` mode); this is the client-side hook.

import { registerSW } from "virtual:pwa-register";
import { requestRestart } from "@/lib/restart";

export function registerPwa() {
    if (typeof window === "undefined") return;
    if (!("serviceWorker" in navigator)) return;

    // A newer SW is waiting — surface the restart overlay rather than
    // silently auto-reloading, because we want the user to see what's
    // happening (and not lose mid-conversation state without warning).
    // RestartOverlay's button calls window.location.reload(), which
    // pulls the new index.html that references the new asset hashes.
    registerSW({
        immediate: true,
        onNeedRefresh() {
            requestRestart("rullama was updated in the background");
        },
        onOfflineReady() {
            console.log("[rullama] ready to work offline (shell cached)");
        },
    });

    // Defence in depth: if the controller swaps under us (a new SW
    // activated and claimed clients via clientsClaim:true), and the
    // tab's running JS still references hash-stamped asset URLs from
    // the previous build, the next dynamic import (the inference
    // worker, the wasm module) will fail. Catch the swap here so we
    // can offer to restart *before* the user tries to load a model.
    navigator.serviceWorker.addEventListener("controllerchange", () => {
        requestRestart("a newer service worker took over this page");
    });
}
