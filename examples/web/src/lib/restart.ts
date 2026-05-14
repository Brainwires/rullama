// Restart-required signal.
//
// Fires when the running tab's JS or WASM has become inconsistent with the
// service worker's precache — typically after a deploy that changed the wasm
// hash. The current tab references asset URLs that no longer exist; trying
// to dynamic-import the inference worker (which pulls the wasm) fails with
// a "Failed to fetch dynamically imported module" / chunk-load error.
//
// Two callers hook into this:
//   1. `lib/pwa.ts` — `onNeedRefresh` and the `controllerchange` event mark
//      a restart as available (a new SW is waiting / has activated).
//   2. `lib/inference.ts` — the WorkerClient's error handler calls
//      `requestRestart()` when the inference worker fails to boot, which
//      is the unambiguous symptom the user is seeing.
//
// One component (`RestartOverlay`) subscribes via the React hook and
// renders the full-page "Restart required" card with a button that
// hard-reloads the tab.

const EVT = "rullama:needs-restart";

let _pending: { reason: string } | null = null;

/** Mark a restart as required. Subsequent calls are no-ops. */
export function requestRestart(reason: string): void {
    if (typeof window === "undefined") return;
    if (_pending) return;
    _pending = { reason };
    window.dispatchEvent(new CustomEvent(EVT, { detail: { reason } }));
    console.warn("[rullama] restart required:", reason);
}

/** Updater function handed in by `pwa.ts` once `registerSW` has run.
 *  When set, restartNow() routes through it so the waiting SW gets
 *  skipWaiting + the reload as a single user-driven event. Falls back
 *  to a plain reload if the SW path isn't available (dev, no SW, or
 *  the restart was triggered by something other than a deploy). */
let _updateSW: ((reload?: boolean) => Promise<void>) | null = null;

export function setUpdateSW(fn: (reload?: boolean) => Promise<void>): void {
    _updateSW = fn;
}

/** Reload the page, preferring the SW-aware updater when present. */
export function restartNow(): void {
    if (typeof window === "undefined") return;
    if (_updateSW) {
        // updateSW handles skipWaiting → controllerchange → reload itself.
        void _updateSW(true);
        return;
    }
    window.location.reload();
}

/** Heuristic: does this error message look like the SW serving a stale
 *  reference to a hashed asset that no longer exists? Different browsers
 *  phrase the same failure differently. */
function looksLikeStaleAssetError(msg: string): boolean {
    const m = msg.toLowerCase();
    return m.includes("failed to fetch")
        || m.includes("dynamically imported module")
        || m.includes("importing a module script")
        || m.includes("loading chunk")
        || m.includes("loading css chunk")
        || m.includes("module specifier")
        || m.includes("webassembly")
        || m.includes("import error");
}

/** Install window-level error / unhandledrejection listeners that trip
 *  the restart overlay when a runtime dynamic import fails after a
 *  deploy invalidated hashed asset URLs. Idempotent. */
let _globalListenersInstalled = false;
export function installGlobalRestartListeners(): void {
    if (typeof window === "undefined") return;
    if (_globalListenersInstalled) return;
    _globalListenersInstalled = true;

    window.addEventListener("error", (ev) => {
        const msg = ev.message || (ev.error && String(ev.error)) || "";
        if (looksLikeStaleAssetError(msg)) {
            requestRestart("a script failed to load");
        }
    });
    window.addEventListener("unhandledrejection", (ev) => {
        const r = ev.reason;
        const msg = (r && (r.message || r.toString())) || "";
        if (looksLikeStaleAssetError(msg)) {
            requestRestart("a script failed to load");
        }
    });
}

/** React hook — returns the current restart reason (or null) and
 *  re-renders the caller when it changes. */
import { useEffect, useState } from "react";
export function useNeedsRestart(): string | null {
    const [reason, setReason] = useState<string | null>(_pending?.reason ?? null);
    useEffect(() => {
        const handler = (e: Event) => {
            const ce = e as CustomEvent<{ reason: string }>;
            setReason(ce.detail.reason);
        };
        window.addEventListener(EVT, handler);
        // In case requestRestart fired before the effect mounted.
        if (_pending && reason === null) setReason(_pending.reason);
        return () => window.removeEventListener(EVT, handler);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    return reason;
}
