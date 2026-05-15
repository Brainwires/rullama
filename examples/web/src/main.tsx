import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/globals.css";
import { App } from "./App";
import { ensureFreshServiceWorker } from "@/lib/pwa";
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
await ensureFreshServiceWorker();

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <ToastProvider>
            <App />
            <Toaster />
        </ToastProvider>
    </React.StrictMode>,
);
