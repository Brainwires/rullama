import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/globals.css";
import { App } from "./App";
import { registerPwa } from "@/lib/pwa";
import { ToastProvider } from "@/lib/toast";
import { Toaster } from "@/components/Toaster";

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
