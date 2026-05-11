import React from "react";
import ReactDOM from "react-dom/client";
import "./styles/globals.css";
import { App } from "./App";
import { registerPwa } from "@/lib/pwa";

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>,
);

// Register the service worker (manifest + offline shell). No-op when the
// build was made without VitePWA (e.g. during early dev).
registerPwa();
