// Dev-only WebSocket client for the Rust devserver's wasm-rebuild
// notifications. Imports a no-op tree-shaken module in production.
//
// Wire: devserver's `dev-server/src/watcher.rs` writes
// to a tokio broadcast channel whenever `wasm-pack build` finishes;
// `ws.rs` fans that out over `/__rullama-dev-ws`. On
//   - `wasm-building` → show a small toast so the user knows a build
//     is in flight (Rust → WASM takes ~30-60 s with cold cargo cache).
//   - `wasm-rebuilt`  → close the SharedWorker (so it disposes the
//     current wasm Model) and reload the page (so the new bundle is
//     fetched + a fresh worker spawns).
//   - `wasm-failed`   → surface the stderr tail as a banner; do NOT
//     reload (we'd just reload to the same stale state).
//
// The reload is full because the wasm-bindgen module is owned by the
// SharedWorker and there's no clean way to hot-swap it in place — see
// `~/.claude/plans/write-this-up-formally-delegated-sun.md` for the
// design rationale.

import { getClient } from "@/lib/inference";

type DevEvent =
    | { type: "hello"; at_ms?: number }
    | { type: "wasm-building" }
    | { type: "wasm-rebuilt"; at_ms?: number }
    | { type: "wasm-failed"; stderr_tail?: string };

let connected = false;
let reconnectTimer: number | null = null;

export function installDevHmr() {
    if (typeof window === "undefined") return;
    if (connected) return;
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/__rullama-dev-ws`;
    let ws: WebSocket;
    try {
        ws = new WebSocket(url);
    } catch (e) {
        console.warn("[dev-hmr] WebSocket constructor failed", e);
        scheduleReconnect();
        return;
    }
    connected = true;

    ws.addEventListener("open", () => {
        console.info("[dev-hmr] connected to", url);
    });

    ws.addEventListener("message", (ev) => {
        let parsed: DevEvent | null = null;
        try { parsed = JSON.parse(ev.data) as DevEvent; } catch { /* */ }
        if (!parsed) return;
        switch (parsed.type) {
            case "hello":
                break;
            case "wasm-building":
                showBanner("Rebuilding WASM…", "info");
                break;
            case "wasm-rebuilt":
                showBanner("WASM rebuilt — reloading page…", "ok");
                void shutdownAndReload();
                break;
            case "wasm-failed":
                showBanner(
                    `WASM rebuild FAILED.\n${(parsed.stderr_tail ?? "").slice(-1200)}`,
                    "err",
                    /* persist */ true,
                );
                break;
        }
    });

    ws.addEventListener("close", () => {
        connected = false;
        console.info("[dev-hmr] disconnected — will retry");
        scheduleReconnect();
    });
    ws.addEventListener("error", () => {
        // close will follow; let that drive reconnect
    });
}

function scheduleReconnect() {
    if (reconnectTimer != null) return;
    reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        installDevHmr();
    }, 2000);
}

async function shutdownAndReload() {
    try {
        const client = getClient();
        // Best-effort: close the SharedWorker so the next page load
        // spawns a fresh one against the new wasm. If it throws (worker
        // already gone, no method, etc), fall through to reload anyway.
        const c = client as unknown as { shutdown?: () => Promise<unknown> };
        if (typeof c.shutdown === "function") {
            await c.shutdown().catch(() => undefined);
        }
    } catch { /* ignore */ }
    // Tiny delay so the toast has a frame to render before we reload.
    setTimeout(() => location.reload(), 250);
}

// Minimal toast — we don't pull in the app's toast lib here so this
// file can be imported anywhere without React-context coupling.
function showBanner(text: string, level: "info" | "ok" | "err", persist = false) {
    let host = document.getElementById("__rullama_dev_hmr");
    if (!host) {
        host = document.createElement("div");
        host.id = "__rullama_dev_hmr";
        host.style.cssText = [
            "position:fixed",
            "left:50%",
            "bottom:20px",
            "transform:translateX(-50%)",
            "z-index:99999",
            "font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace",
            "padding:8px 12px",
            "border-radius:6px",
            "max-width:min(80vw,720px)",
            "white-space:pre-wrap",
            "box-shadow:0 4px 16px rgba(0,0,0,0.25)",
            "pointer-events:none",
        ].join(";");
        document.body.appendChild(host);
    }
    host.textContent = text;
    host.style.background = level === "err"
        ? "rgba(120,0,0,0.92)"
        : level === "ok"
            ? "rgba(0,90,40,0.92)"
            : "rgba(40,40,40,0.92)";
    host.style.color = "#fff";
    if (!persist) {
        window.setTimeout(() => {
            const h = document.getElementById("__rullama_dev_hmr");
            if (h && h.textContent === text) h.remove();
        }, 4000);
    }
}
