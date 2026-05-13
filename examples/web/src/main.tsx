import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/globals.css";
import { App } from "./App";
import { registerPwa } from "@/lib/pwa";
import { installGlobalRestartListeners } from "@/lib/restart";
import { ToastProvider } from "@/lib/toast";
import { Toaster } from "@/components/Toaster";

// Install before anything else mounts so that any dynamic-import or
// chunk-load failure (Vite-emitted asset URL that doesn't match the
// active service worker's precache after a deploy) routes through the
// restart overlay rather than landing as a silent console error.
installGlobalRestartListeners();

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <ToastProvider>
            <App />
            <Toaster />
        </ToastProvider>
    </React.StrictMode>,
);

// Register the service worker (manifest + offline shell). No-op when the
// build was made without VitePWA (e.g. during early dev).
registerPwa();
