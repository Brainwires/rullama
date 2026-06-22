import { useCallback, useEffect, useState } from "react";
import { getClient } from "@/lib/inference";
import { applyServiceWorkerUpdate, onServiceWorkerUpdateReady } from "@/lib/pwa";
import {
    fetchServerVersion,
    isUpdateAvailable,
    isDismissed,
    setDismissedVersion,
} from "@/lib/version";

/**
 * PWA update-banner state + the apply/dismiss handlers + the boot-time
 * version check.
 *
 * Drives the `UpdateBanner` (gated on `!busy` by the caller) and the
 * `ApplyingOverlay`. The state is also mutated from the cross-tab
 * SharedWorker subscription in `App` (the `meta` / `updateAvailable` /
 * `applyingUpdate` notifies), so `setUpdateVersion` and `setApplyingUpdate`
 * are returned for that consumer too.
 */
export function usePwaUpdate() {
    const [updateVersion, setUpdateVersion] = useState<string | null>(null);
    const [applyingUpdate, setApplyingUpdate] = useState(false);

    // Boot-time PWA update check. Fetches /version.json (server's
    // currently-deployed version) and compares against the version
    // baked into this bundle at build time. If they differ, broadcast
    // to all tabs via the SharedWorker so they surface the same banner
    // without each having to re-fetch.
    //
    // The function is offline-aware: navigator.onLine === false or a
    // failed/timeout fetch returns null and we silently no-op. We never
    // block boot on this; rendering proceeds in parallel.
    useEffect(() => {
        (async () => {
            const server = await fetchServerVersion();
            if (!server) return;                              // offline / fetch failed
            if (!isUpdateAvailable(server)) return;           // already on latest
            if (isDismissed(server))        return;           // user clicked "Later" on this version
            setUpdateVersion(server.version);
            try {
                await getClient().broadcastUpdateAvailable(server.version);
            } catch (e) {
                console.warn("[rullama] failed to broadcast update availability to other tabs:", e);
            }
        })();
    }, []);

    // Live update detection via Workbox `onNeedRefresh`: fires whenever a new
    // build's service worker installs and waits — including while the app is
    // open, and more reliably than the boot-only version.json check (it's the
    // actual "new precache is ready" signal). Surface it as an available update;
    // App auto-applies when idle. Best-effort version label for the banner.
    useEffect(() => {
        return onServiceWorkerUpdateReady(() => {
            void (async () => {
                const server = await fetchServerVersion().catch(() => null);
                setUpdateVersion((prev) => prev ?? server?.version ?? "latest");
            })();
        });
    }, []);

    // Apply / dismiss handlers for the UpdateBanner.
    const onApplyUpdate = useCallback(() => {
        const v = updateVersion ?? "";
        // Optimistically show the overlay even before the router echoes
        // back `applyingUpdate` — keeps the UI feeling responsive.
        setApplyingUpdate(true);
        (async () => {
            // Best-effort: tell other tabs to reload too (cross-tab coordination).
            try {
                await getClient().applyUpdate(v);
            } catch (e) {
                console.warn("[rullama] applyUpdate cross-tab RPC failed (continuing solo):", e);
            }
            // THE actual update: activate the waiting service worker (skipWaiting)
            // and reload. A plain reload would re-serve the OLD precached bundle —
            // this is what was missing and why updates needed a manual cache clear.
            await applyServiceWorkerUpdate();
        })();
    }, [updateVersion]);

    const onDismissUpdate = useCallback(() => {
        if (updateVersion) setDismissedVersion(updateVersion);
        setUpdateVersion(null);
    }, [updateVersion]);

    return {
        updateVersion, setUpdateVersion,
        applyingUpdate, setApplyingUpdate,
        onApplyUpdate, onDismissUpdate,
    };
}
