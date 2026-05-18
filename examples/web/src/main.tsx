import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/globals.css";
import { App } from "./App";
import { ensureFreshServiceWorker, installPostBootSwReloadListener } from "@/lib/pwa";
import { installGlobalRestartListeners } from "@/lib/restart";
import { ToastProvider } from "@/lib/toast";
import { Toaster } from "@/components/Toaster";

// Install before anything else mounts so that any dynamic-import or
// chunk-load failure (Vite-emitted asset URL that doesn't match the
// active service worker's precache after a deploy) routes through the
// restart overlay rather than landing as a silent console error.
installGlobalRestartListeners();

// Gate React render on service-worker freshness. Returning visitors with
// a pending update wait (≤1.5 s) for the new SW to claim the page so the
// first render is already against the fresh asset set. First-ever loads
// fall through immediately. See `lib/pwa.ts` for the lifecycle details.
//
// Wrapped in try/catch as a safety net: a thrown error here would black-
// screen the PWA (top-level await rejects → bundle never resolves →
// React never mounts → static-HTML watchdog catches it after 8 s). We'd
// rather render in degraded mode against possibly-stale assets than
// trip the watchdog on a transient SW lifecycle hiccup.
try {
    await ensureFreshServiceWorker();
} catch (e) {
    // Surface to the console for diagnosis but DO NOT abort boot.
    // eslint-disable-next-line no-console
    console.warn("[rullama] ensureFreshServiceWorker threw:", e);
}

// Arm the mid-session SW-swap reloader only AFTER the boot-time
// freshness gate has resolved — otherwise the boot-time controllerchange
// (clientsClaim handing us off to the just-installed SW) would trip a
// spurious reload loop. From here on, any controller swap is a real
// post-deploy event and we reload to land on matching assets.
installPostBootSwReloadListener();

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <ToastProvider>
            <App />
            <Toaster />
        </ToastProvider>
    </React.StrictMode>,
);
