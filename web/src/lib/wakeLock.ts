// Screen Wake Lock — hold the display awake during model download and
// token generation. Two iOS-specific quirks the hook handles:
//
//  1. iOS Safari auto-releases the sentinel when the tab is hidden
//     (system policy, no override). We re-acquire on `visibilitychange`
//     while the caller still wants the lock held.
//  2. The API requires a recently-engaged page — first acquire must
//     happen in response to a user gesture or while the page is visible
//     and focused. Model-download / generation are both kicked off by a
//     tap, so the first acquire always succeeds; subsequent re-acquires
//     after a visibility round-trip do too.
//
// Older browsers / private mode without `navigator.wakeLock` are silent
// no-ops; the hook never throws.
//
// Usage:
//   useWakeLock(modelStatus === "loading" || busy);

import { useEffect } from "react";

type WakeLockSentinel = {
    released: boolean;
    release: () => Promise<void>;
    addEventListener: (type: "release", listener: () => void) => void;
};

type WakeLockNavigator = Navigator & {
    wakeLock?: { request: (type: "screen") => Promise<WakeLockSentinel> };
};

function isSupported(): boolean {
    return typeof navigator !== "undefined"
        && !!(navigator as WakeLockNavigator).wakeLock;
}

export function useWakeLock(active: boolean): void {
    useEffect(() => {
        if (!active || !isSupported()) return;

        let sentinel: WakeLockSentinel | null = null;
        let cancelled = false;

        const acquire = async () => {
            if (cancelled) return;
            if (sentinel && !sentinel.released) return;
            if (document.visibilityState !== "visible") return;
            try {
                sentinel = await (navigator as WakeLockNavigator).wakeLock!.request("screen");
                sentinel.addEventListener("release", () => {
                    // iOS releases the sentinel on hide; we'll re-acquire
                    // when visibility flips back to "visible" (handler below).
                    sentinel = null;
                });
            } catch {
                // Permission denied, not focused, low-power mode, etc.
                // Silent — nothing useful to surface here.
            }
        };

        const onVisible = () => {
            if (!cancelled && document.visibilityState === "visible") {
                void acquire();
            }
        };

        void acquire();
        document.addEventListener("visibilitychange", onVisible);

        return () => {
            cancelled = true;
            document.removeEventListener("visibilitychange", onVisible);
            if (sentinel && !sentinel.released) {
                void sentinel.release().catch(() => {});
            }
            sentinel = null;
        };
    }, [active]);
}
