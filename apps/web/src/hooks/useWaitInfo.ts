import { useEffect, useState } from "react";

export type WaitInfo = {
    kind: "modelLoad" | "downloadLock" | "downloadStream";
    message: string;
    ts: number;
} | null;

/**
 * Wait-reason coordination for the model loader.
 *
 * The worker emits three independent `…Waiting` / `…Retrying` notifies
 * during slow OPFS / network operations; surfacing them all through
 * `setLoadingLabel` directly is racy (one event can stomp another's
 * message). Keep the **most-recent** wait reason in its own state and
 * have the render pass `waitInfo?.message ?? loadingLabel` to the
 * ModelLoader, so a wait label naturally supersedes the normal progress
 * label while the wait is fresh and is cleared by the staleness timer
 * below.
 */
export function useWaitInfo() {
    const [waitInfo, setWaitInfo] = useState<WaitInfo>(null);

    useEffect(() => {
        if (!waitInfo) return;
        // Auto-clear after 4 s of no new updates so a finished retry
        // doesn't leave a stale "waiting…" line in the loader once the
        // real operation has resumed.
        const t = setTimeout(() => {
            setWaitInfo((cur) => (cur === waitInfo ? null : cur));
        }, 4000);
        return () => clearTimeout(t);
    }, [waitInfo]);

    return { waitInfo, setWaitInfo };
}
