import { useEffect, useRef } from "react";
import { getClient } from "@/lib/inference";
import { useToast } from "@/lib/toast";

/**
 * Page/session lifecycle plumbing that has nothing to do with the model
 * or chat domains — grouped here so App doesn't carry it inline:
 *
 *   - **Environment probe** → sticky toasts when WebGPU / OPFS / COI are
 *     missing.
 *   - **Crash-detect on mount** — toasts if the previous session ended
 *     without either clean-exit signal (worker manifest `cleanExit` or
 *     the `pagehide` localStorage marker).
 *   - **pagehide marker** — the reliable "tab is going away" signal; we
 *     cache the worker session id synchronously on mount (pagehide can't
 *     await an RPC) and write it on hide so the next mount's crash-detect
 *     treats it as a clean exit.
 *
 * `setView` is needed only for the crash toast's "Open Logs" action.
 */
export function useSessionLifecycle(setView: (v: "settings") => void) {
    const { showToast } = useToast();

    // Environment probe → sticky toasts on missing capabilities.
    useEffect(() => {
        const has = (k: () => boolean) => { try { return k(); } catch { return false; } };
        if (!has(() => typeof navigator !== "undefined" && "gpu" in navigator)) {
            showToast({
                id: "env-webgpu", level: "error", title: "WebGPU not available",
                message: "rullama cannot run inference without WebGPU. On iOS update to iOS 18+; on desktop use a recent Chrome/Edge/Safari.",
                persist: true,
            });
        }
        if (!has(() => typeof navigator !== "undefined" && !!navigator.storage && typeof navigator.storage.getDirectory === "function")) {
            showToast({
                id: "env-opfs", level: "error", title: "OPFS not available",
                message: "Models larger than ~3 GB need OPFS for streaming. Without it, large GGUFs will OOM the page.",
                persist: true,
            });
        }
        if (!has(() => typeof window !== "undefined" && window.crossOriginIsolated)) {
            showToast({
                id: "env-coi", level: "warn", title: "Cross-origin isolation off",
                message: "Required for SharedArrayBuffer and high-res timers. Make sure the page is served with COOP+COEP headers.",
                persist: true,
            });
        }
    }, [showToast]);

    // Crash-detect on mount. A "crash" is: the previous session
    // exists AND neither end-of-life signal fired for it. There are
    // two signals; either is sufficient evidence of a clean exit.
    //
    //   1. The worker's shutdown handler wrote `cleanExit: true` into
    //      the session manifest. Reliable on a graceful close where
    //      the worker actually runs its shutdown. UNRELIABLE on a
    //      hard reload (Cmd+R), where the worker is terminated by
    //      the browser before postMessage round-trips complete.
    //
    //   2. The page's `pagehide` listener wrote the session id into
    //      `localStorage["rullama:session:clean"]`. Reliable on any
    //      page navigation/reload (pagehide is synchronous, runs
    //      before the page is torn down) — this is the path that was
    //      previously written but never read, which is why every
    //      reload looked like a crash.
    //
    // Either signal == clean. Only when BOTH are absent do we toast.
    useEffect(() => {
        const client = getClient();
        (async () => {
            try {
                const [list, currentId] = await Promise.all([
                    client.logs.list(),
                    client.logs.currentId().catch(() => "" as string),
                ]);
                const previous = list.find((s) => s.id !== currentId);
                if (!previous) return;
                if (previous.cleanExit) return;
                let pagehideMarker = "";
                try { pagehideMarker = localStorage.getItem("rullama:session:clean") || ""; } catch { /* */ }
                if (pagehideMarker && pagehideMarker === previous.id) {
                    // Clean exit per pagehide; clear the marker so the
                    // next reload's check starts from a clean slate.
                    try { localStorage.removeItem("rullama:session:clean"); } catch { /* */ }
                    return;
                }
                showToast({
                    level: "warn",
                    title: "Last session ended unexpectedly",
                    message: "The tab may have been terminated (iOS jetsam, OOM, or a hard kill). Open the logs to see what fired right before the kill.",
                    persist: true,
                    action: {
                        label: "Open Logs",
                        onClick: () => {
                            try { localStorage.setItem("rullama:settings:tab", JSON.stringify("logs")); } catch { /* */ }
                            setView("settings");
                        },
                    },
                });
            } catch { /* logger unavailable — silently skip */ }
        })();
    }, [showToast, setView]);

    // Clean-exit marker. We write a localStorage marker with the current
    // session id; the next mount's crash-detect treats a matching marker
    // as proof of a clean exit even if the worker's manifest shutdown
    // didn't complete.
    //
    // The marker is driven primarily by **`visibilitychange → hidden`**,
    // NOT `pagehide`. On iOS standalone PWAs `pagehide` is unreliable:
    // when the user backgrounds the app (home / app switcher) and then
    // swipe-kills it — or iOS reclaims it while suspended — `pagehide`
    // frequently never fires, so a deliberate close looked identical to
    // a jetsam/OOM kill. `visibilitychange → hidden` is the last event
    // reliably observed before iOS suspends/terminates, and a manual
    // close ALWAYS passes through it first (you can't swipe-kill a
    // foreground app — backgrounding fires `hidden`). So:
    //
    //   - on `hidden`  → write the marker. A subsequent kill while
    //     backgrounded (user swipe-kill or OS reclaim of a suspended
    //     app) is then treated as a clean exit, no crash toast.
    //   - on `visible` → CLEAR the marker. The session is resuming, so
    //     a *later* crash in the foreground (e.g. jetsam mid-inference)
    //     must NOT be masked by the stale marker from when it was
    //     backgrounded — re-arm crash detection.
    //
    // `pagehide` still writes the marker as a belt-and-suspenders for
    // desktop reload/navigation (where `visibilitychange` may not flip
    // to hidden before teardown).
    //
    // Critical: `client.logs.currentId()` is an async RPC over
    // postMessage, and these teardown events CANNOT await. By the time
    // the worker responds, the tab may be gone and `localStorage.setItem`
    // never runs. Cache the id synchronously on mount, then read the
    // cached value inside the listeners so the marker is written without
    // an RPC round-trip.
    const currentSessionIdRef = useRef<string>("");
    useEffect(() => {
        let cancelled = false;
        const client = getClient();
        const MARKER = "rullama:session:clean";
        const writeMarker = () => {
            const id = currentSessionIdRef.current;
            if (!id) return;
            try { localStorage.setItem(MARKER, id); } catch { /* */ }
        };
        const clearMarker = () => {
            try { localStorage.removeItem(MARKER); } catch { /* */ }
        };
        // Cache the worker's session id once on mount. Re-cache when
        // visibility flips back to "visible" — the SharedWorker may
        // have rotated session ids while the tab was hidden.
        const refresh = async () => {
            try {
                const id = await client.logs.currentId().catch(() => "");
                if (!cancelled && id) currentSessionIdRef.current = id;
            } catch { /* */ }
        };
        void refresh();
        const onVis = () => {
            if (document.visibilityState === "hidden") {
                // Last reliable signal before iOS suspends/terminates.
                writeMarker();
            } else {
                // Session resumed — re-arm crash detection, then refresh
                // the (possibly rotated) session id.
                clearMarker();
                void refresh();
            }
        };
        const onHide = () => writeMarker();
        document.addEventListener("visibilitychange", onVis);
        window.addEventListener("pagehide", onHide);
        return () => {
            cancelled = true;
            document.removeEventListener("visibilitychange", onVis);
            window.removeEventListener("pagehide", onHide);
        };
    }, []);
}
