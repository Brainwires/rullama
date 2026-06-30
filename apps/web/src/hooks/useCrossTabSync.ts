import { type Dispatch, type MutableRefObject, type SetStateAction, useEffect } from "react";
import { type ModelStatus } from "@/components/ModelLoader";
import { getClient, type ConversationRow } from "@/lib/inference";
import { useToast } from "@/lib/toast";
import { BUNDLED_VERSION } from "@/lib/version";
import { INFLIGHT_KEY, type InflightGen } from "@/lib/app-helpers";
import { type WaitInfo } from "@/hooks/useWaitInfo";

type LoadedMeta = {
    name?: string | null;
    modelKey?: string;
    hasVision?: boolean;
    hasAudio?: boolean;
} | null | undefined;

export interface CrossTabSyncParams {
    setMetaInit: Dispatch<SetStateAction<boolean>>;
    setModelStatus: Dispatch<SetStateAction<ModelStatus>>;
    setHasVision: Dispatch<SetStateAction<boolean>>;
    setHasAudio: Dispatch<SetStateAction<boolean>>;
    setStatusText: Dispatch<SetStateAction<string>>;
    setUpdateVersion: Dispatch<SetStateAction<string | null>>;
    setApplyingUpdate: Dispatch<SetStateAction<boolean>>;
    setConversations: Dispatch<SetStateAction<ConversationRow[]>>;
    setActiveAdapter: (next: string | null | ((prev: string | null) => string | null)) => void;
    setTrainingInProgress: Dispatch<SetStateAction<boolean>>;
    setWaitInfo: Dispatch<SetStateAction<WaitInfo>>;
    inflightRef: MutableRefObject<InflightGen | null>;
}

/**
 * Cross-tab state sync. The SharedWorker broadcasts notifications
 * whenever shared state changes; each tab reflects them in React state so
 * all tabs stay aligned.
 *
 * This is a wiring effect that spans several domains (model load, PWA
 * update, conversations, adapter, training), so the relevant setters/refs
 * are passed in from their owning hooks. Keeping every subscription in one
 * effect preserves the original registration + cleanup semantics (and the
 * one-shot adapter probe) rather than scattering them across hooks.
 */
export function useCrossTabSync(params: CrossTabSyncParams) {
    const {
        setMetaInit, setModelStatus, setHasVision, setHasAudio, setStatusText,
        setUpdateVersion, setApplyingUpdate, setConversations, setActiveAdapter,
        setTrainingInProgress, setWaitInfo, inflightRef,
    } = params;
    const { showToast } = useToast();

    useEffect(() => {
        const client = getClient();
        const applyLoaded = (loaded: LoadedMeta) => {
            if (loaded) {
                setModelStatus("ready");
                setHasVision(!!loaded.hasVision);
                setHasAudio(!!loaded.hasAudio);
                setStatusText(String(loaded.name ?? loaded.modelKey ?? "loaded"));
            } else {
                setModelStatus("idle");
                setStatusText("no model");
                setHasVision(false);
                setHasAudio(false);
            }
        };
        const offs = [
            client.subscribe("meta", (p) => {
                setMetaInit(true);
                applyLoaded(p.loaded as LoadedMeta);
                // The router may have already learned of a pending
                // update from an earlier tab. Pick it up on first meta
                // broadcast so this tab surfaces the same banner
                // without re-fetching /version.json.
                const pending = (p as { pendingUpdateVersion?: string | null }).pendingUpdateVersion;
                if (typeof pending === "string" && pending && pending !== BUNDLED_VERSION) {
                    setUpdateVersion((cur) => cur ?? pending);
                }
            }),
            // A tab somewhere detected (via /version.json) that a
            // newer build is deployed. Surface in our banner state.
            client.subscribe("updateAvailable", (p) => {
                const v = String((p as { version?: unknown }).version ?? "");
                if (v && v !== BUNDLED_VERSION) {
                    setUpdateVersion(v);
                }
            }),
            // The user clicked "Apply now" on some tab; every tab
            // reloads in lockstep so the new SharedWorker URL is
            // adopted simultaneously.
            client.subscribe("applyingUpdate", (p) => {
                const v = String((p as { version?: unknown }).version ?? "");
                setApplyingUpdate(true);
                if (v) setUpdateVersion(v);
                // Best-effort: snapshot in-flight generation so the new
                // bundle can pick it up via the existing visibilitychange
                // resume path on next boot. Fire-and-forget — if
                // localStorage throws (private mode, quota), we fall
                // back to slow-path replay from the DB, which is fine.
                const inflight = inflightRef.current;
                if (inflight) {
                    try { localStorage.setItem(INFLIGHT_KEY, JSON.stringify(inflight)); }
                    catch (e) { console.warn("[rullama] failed to snapshot inflight before update reload:", e); }
                }
                // 600 ms gives the core worker time to process the
                // {type:"shutdown"} message (sent by the router) and
                // release its OPFS sync handle + GPU towers BEFORE the
                // new bundle boots a fresh core that needs them.
                setTimeout(() => window.location.reload(), 600);
            }),
            client.subscribe("modelLoaded", (p) => {
                applyLoaded(p as LoadedMeta);
            }),
            client.subscribe("modelFreed", () => {
                applyLoaded(null);
            }),
            client.subscribe("dbChanged", () => {
                // Conversation list may have changed in another tab.
                // Cheap to re-query; the broadcast is debounced upstream
                // (only fires on conv* mutations, not per msgAppend).
                void getClient().convList()
                    .then(setConversations)
                    .catch(() => { /* */ });
            }),
            client.subscribe("adapterChanged", (p) => {
                setActiveAdapter((p.active as string | null | undefined) ?? null);
            }),
            // **D3 — training-in-progress tracking.** The worker
            // broadcasts these notifies across tabs via the
            // SharedWorker router; every tab learns about it and the
            // chat input gates uniformly. Without this, the Send
            // button would fail with "model is owned by training
            // session" on click and the user wouldn't know why.
            client.subscribe("trainingStarted", () => { setTrainingInProgress(true); }),
            client.subscribe("trainingFinished", () => { setTrainingInProgress(false); }),
            // Worker is waiting on the OPFS read-syncHandle while the
            // previous worker's exclusive lock GCs. Surface to the boot
            // splash AND the in-app waitInfo state so the user knows
            // something is happening (this matters most on iPhone where
            // there's no easy dev console). Render-side combines waitInfo
            // with loadingLabel so this label naturally supersedes the
            // normal progress label.
            client.subscribe("modelLoadWaiting", (p) => {
                const attempt = Number(p.attempt ?? 0);
                const elapsed = Number(p.elapsedMs ?? 0);
                const msg = `Waiting for previous session to release the model… (${(elapsed / 1000).toFixed(1)}s, attempt ${attempt})`;
                setWaitInfo({ kind: "modelLoad", message: msg, ts: Date.now() });
                try {
                    window.__rullamaBootStatus?.("Almost there…", msg);
                } catch { /* */ }
            }),
            // Worker is waiting on the WRITE syncHandle for a resumed
            // download. Same situation, different operation.
            client.subscribe("downloadWaiting", (p) => {
                const attempt = Number(p.attempt ?? 0);
                const elapsed = Number(p.elapsedMs ?? 0);
                const msg = `Waiting for previous session to release the download… (${(elapsed / 1000).toFixed(1)}s, attempt ${attempt})`;
                setWaitInfo({ kind: "downloadLock", message: msg, ts: Date.now() });
            }),
            // Download stream broke (network drop / iOS screen-lock
            // socket sever); worker is retrying with Range resume.
            client.subscribe("downloadRetrying", (p) => {
                const attempt = Number(p.attempt ?? 0);
                const max = Number(p.maxAttempts ?? 0);
                const delay = Number(p.nextDelayMs ?? 0);
                const msg = `Connection dropped — retrying in ${(delay / 1000).toFixed(1)}s (attempt ${attempt}/${max})`;
                setWaitInfo({ kind: "downloadStream", message: msg, ts: Date.now() });
            }),
            client.subscribe("gpuFault", (p) => {
                // Typed GPU fault surfaced by the inference core worker
                // when wgpu returns device-lost / OOM / WebGPU validation
                // error. Without this banner the tab just looks frozen.
                const kind = String(p.kind ?? "unknown");
                const during = String(p.during ?? "an inference call");
                const message = String(p.message ?? "");
                const title = kind === "oom"
                    ? "GPU is out of memory"
                    : kind === "device-lost"
                        ? "GPU device was lost"
                        : "GPU error";
                const hint = kind === "oom"
                    ? "Close other browser tabs or reload to free GPU memory, then try again."
                    : kind === "device-lost"
                        ? "Reload the page to restart the GPU context."
                        : "Reload and check the dev console for details.";
                showToast({
                    level: "error",
                    title,
                    message: `${hint} (during: ${during})\n${message}`,
                    persist: true,
                });
            }),
        ];
        // Probe initial adapter state. The worker's `active` is its
        // in-memory state for THIS core session — empty on a fresh
        // boot. If we have a persisted activeAdapter (from a prior
        // tab that applied one), prefer ours and trust the
        // restore-on-ready effect to re-apply it via
        // `trainingApplyAdapter`. If we don't, take whatever the
        // worker says (could be non-null if another tab is already
        // running with one applied).
        void getClient().trainingListAdapters()
            .then((r) => {
                // Only overwrite if BOTH the persisted name is null
                // and the worker has nothing. Otherwise the persisted
                // (or the worker's existing) name wins.
                setActiveAdapter((cur) => cur ?? r.active);
            })
            .catch(() => { /* */ });
        return () => { for (const o of offs) o(); };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
}
