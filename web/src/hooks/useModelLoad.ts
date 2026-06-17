import { useCallback, useEffect, useRef, useState } from "react";
import { type ModelStatus } from "@/components/ModelLoader";
import { type ModelEntry, blobUrl, beacon, listModels, isDiffusion } from "@/lib/api";
import { ensureModel, existingSize, opfsSupported, requestPersistent, wipeModel } from "@/lib/opfs";
import { getNetworkHint } from "@/lib/network";
import { getClient } from "@/lib/inference";
import { useToast } from "@/lib/toast";
import { useConfirm } from "@/lib/confirm";
import { usePersistedState } from "@/lib/persisted";
import { fmtBytes, fmtEta } from "@/lib/utils";
import { isMobileUA } from "@/lib/app-helpers";
import { type WaitInfo } from "@/hooks/useWaitInfo";

export interface UseModelLoadParams {
    /** Capability tier; auto-load is gated until it's known & supported. */
    tier: string | null;
    /** Latest wait reason (from the cross-tab sync) — feeds the boot splash label. */
    waitInfo: WaitInfo;
    /** Clear chat display state (messages / status / pending media) on eject/delete. */
    onUnloadChat: () => void;
    /** Read the chat-busy flag at call time (delete/eject no-op mid-generation). */
    isBusy: () => boolean;
    /** Pre-warm the system prompt into the KV cache after a (non-diffusion)
     *  model finishes loading, so the first chat hot-starts. Runs during the
     *  "preparing" phase; `report(percent, label)` drives the progress bar /
     *  boot splash. Best-effort — a rejection is swallowed and the model
     *  still goes "ready". */
    onPrepare?: (report: (percent: number, label: string) => void) => Promise<void>;
}

/**
 * The whole model-load lifecycle: load/eject/delete, download cancel,
 * auto-load on boot, reload-after-engine-swap, and the boot-splash driver.
 *
 * Owns the model status surface (`modelStatus`, `statusText`, vision/audio
 * capability flags, loading progress) plus the persisted `lastLoadedDigest`
 * / `pendingLoadDigest` that drive auto-resume. The cross-tab sync mutates
 * several of these (model meta broadcasts), so their setters are returned.
 */
export function useModelLoad({ tier, waitInfo, onUnloadChat, isBusy, onPrepare }: UseModelLoadParams) {
    const { showToast, dismissToast } = useToast();
    const confirm = useConfirm();

    const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
    // True when the loaded model is the DiffusionGemma engine (block-diffusion
    // denoise loop in onSend instead of the AR token stream).
    const [loadedIsDiffusion, setLoadedIsDiffusion] = useState(false);
    // Model-picker selection, lifted out of ModelLoader so the main panel and
    // the sidebar dropdown stay in sync. Seeded from the last-used model below.
    const [selectedModelName, setSelectedModelName] = useState<string>("");
    const [loadingPercent, setLoadingPercent] = useState(0);
    const [loadingLabel, setLoadingLabel]     = useState("");
    const [statusText, setStatusText]         = useState("no model");

    // Multimodal: vision availability latches on after a successful model
    // load (it's a property of the meta, only known post-load).
    const [hasVision, setHasVision]   = useState(false);
    const [hasAudio,  setHasAudio]    = useState(false);

    // Digest of the model that was last successfully loaded. Set on
    // onLoad success, cleared on eject or delete. Drives auto-load on
    // page reload — empty string means "no model; show the picker".
    const [lastLoadedDigest, setLastLoadedDigest] =
        usePersistedState<string>("lastLoadedDigest", "");

    // Digest of the model the user *intends* to load — persisted before
    // the network fetch starts so that an iOS WebContent reap mid-
    // download is auto-recovered on next boot. The existing Range-
    // resume protocol in `inference-core-worker.ts` picks up from the
    // partial OPFS file at byte N. Cleared on full success, on eject,
    // and when the catalog lookup fails (so a deleted-from-server
    // model doesn't keep retrying every boot).
    const [pendingLoadDigest, setPendingLoadDigest] =
        usePersistedState<string>("pendingLoadDigest", "");

    // Has the SharedWorker pushed at least one `meta` notification? Until
    // it has, we don't know whether another tab already loaded a model,
    // so auto-load is gated on this flag (see effect below).
    const [metaInit, setMetaInit] = useState(false);

    // Set by the Stop modal's "Delete partial" choice; read in onLoad's
    // download-cancelled handler to wipe the partial after the stream aborts.
    const deleteOnCancelRef = useRef(false);
    // Latest `onPrepare` (system pre-warm), read at call time so onLoad's
    // identity doesn't churn and we never call a stale closure.
    const onPrepareRef = useRef(onPrepare);
    onPrepareRef.current = onPrepare;

    // **Boot-splash driver.** The static HTML splash auto-holds itself
    // when `rullama:lastLoadedDigest` is in localStorage (see index.html)
    // so it stays visible past React-mount. While loading, push the
    // current percent + label so the user sees a real progress bar
    // instead of a generic spinner. Release as soon as the model is
    // ready (or errored — in which case the welcome screen + Load button
    // is the right thing to show).
    useEffect(() => {
        type BootApi = {
            __rullamaBootProgress?: (percent: number, detail?: string) => void;
            __rullamaBootRelease?: () => void;
        };
        const w = window as unknown as BootApi;
        if (modelStatus === "loading" || modelStatus === "preparing") {
            const detail = waitInfo?.message ?? loadingLabel;
            w.__rullamaBootProgress?.(loadingPercent, detail || undefined);
        } else if (modelStatus === "ready" || modelStatus === "error") {
            w.__rullamaBootRelease?.();
        }
        // `idle` is racy with the auto-load effect (auto-load may flip
        // us to "loading" in the next tick). Leave the held splash up;
        // the watchdog in index.html releases after 5 s of no progress
        // if auto-load decided not to fire (saved digest but no OPFS).
    }, [modelStatus, loadingPercent, loadingLabel, waitInfo]);

    const onLoad = useCallback(async (m: ModelEntry) => {
        const client = getClient();
        dismissToast("model-load-error");
        setModelStatus("loading");
        setLoadingPercent(0);
        setLoadingLabel("checking OPFS…");
        setStatusText("loading…");
        // Prime capability flags from the catalog so the mic / image
        // buttons can appear in the same render that flips modelStatus
        // to "ready" — no perceptible lag between "loaded" and "audio
        // available". The wasm-side `hasAudio` getter overwrites these
        // post-load (the worker is the source of truth); the optimistic
        // prime just avoids a one-render gap if there's any latency
        // between the modelLoaded notification reaching this tab and
        // the meta payload reflecting `hasAudio`.
        setHasVision(!!m.multimodal);
        setHasAudio(!!m.multimodal);
        // Persist *intent* before the network round-trip. If iOS reaps
        // the WebContent process mid-download, the next page boot reads
        // this and auto-fires onLoad again — ensureModel's Range-resume
        // logic picks up from the partial OPFS file at byte N.
        setPendingLoadDigest(m.digest);

        try {
            if (!(await opfsSupported())) throw new Error("OPFS not supported in this browser");
            await requestPersistent();

            const modelKey = m.digest.replace(/[^A-Za-z0-9_.-]/g, "_");
            const filename = m.name.replace(/[^A-Za-z0-9_.-]/g, "_") + ".gguf";
            const url = blobUrl(m);
            // Make the resolved fetch URL visible — diagnoses 404s (e.g. a
            // sticky localStorage `localBlobPort` routing an R2-only model to
            // the local devserver, which can't serve it).
            beacon("chat", `load ${m.name}: fetching ${url}${url === m.url ? " (R2/CDN)" : " (local-blob)"}`);

            // Bytes-over-the-wire guard. If OPFS already has the full file
            // we skip the network entirely (offline reload path), so the
            // confirm is only shown when the user is about to actually
            // download. The 200 MB floor lets small files load silently.
            //
            // Network hints from `navigator.connection` are best-effort:
            // iOS exposes `saveData` (= Low Data Mode) reliably but not
            // `type`, so we always confirm large downloads — the warning
            // copy just escalates when we have positive signal.
            const CONFIRM_BYTES = 200 * 1024 * 1024;
            const cachedBytes = await existingSize(modelKey, filename);
            const needBytes   = Math.max(0, m.size - cachedBytes);
            if (needBytes >= CONFIRM_BYTES) {
                const hint = getNetworkHint();
                const sizeLabel = fmtBytes(needBytes);
                const title = hint.metered ? `⚠️ ${hint.reason}` : `Download "${m.name}"`;
                const description =
                    `Downloading "${m.name}" needs ${sizeLabel} over the network. ` +
                    `It will be cached locally so subsequent loads are free.`;
                // Shadcn modal replaces window.confirm — see lib/confirm.tsx.
                // The OK/Cancel buttons are real DOM <button>s so they're
                // automatable via Playwright/CDP (window.confirm() renders
                // a browser-chrome modal that's hard to drive).
                if (!(await confirm({ title, description, okLabel: "OK", cancelLabel: "Cancel" }))) {
                    setModelStatus("idle");
                    setStatusText("no model");
                    setLoadingLabel("");
                    return;
                }
            }

            const t0 = performance.now();
            // The writer worker emits a progress event per stream chunk
            // (~hundreds per second on a fast link). Updating React state
            // that often makes the label flicker; throttle to ~4 Hz, but
            // always emit the final tick so 100 % shows the real number.
            //
            // Rate is computed from bytes/elapsed *since the first
            // progress callback for this session* — `baselineBytes` is
            // the resume offset (already on disk before we started),
            // so resumed downloads don't report a wildly inflated rate
            // from spreading the on-disk bytes across the new elapsed
            // window.
            let lastLabelAt    = 0;
            let baselineBytes  = -1;
            let baselineAt     = 0;
            const { totalBytes, fromCache } = await ensureModel(url, modelKey, filename, m.size, ({ bytesWritten, totalBytes }) => {
                if (totalBytes > 0) {
                    // Clamp the reported offset to [0, totalBytes]. A corrupt or
                    // sparse-file resume can report an absurd bytesWritten (a
                    // runaway offset surfaced as "48744 GB / 16.81 GB —
                    // 948 GB/s") — it must never reach the bar, rate, or ETA.
                    const shown = Math.min(Math.max(0, bytesWritten), totalBytes);
                    setLoadingPercent((shown / totalBytes) * 100);
                    const now  = performance.now();
                    const done = shown >= totalBytes;
                    if (baselineBytes < 0) {
                        baselineBytes = shown;
                        baselineAt    = now;
                    }
                    if (done || now - lastLabelAt > 250) {
                        lastLabelAt = now;
                        const elapsed   = (now - baselineAt) / 1000;
                        const delta     = Math.max(0, shown - baselineBytes);
                        const rate      = elapsed > 0.25 ? delta / elapsed : 0;
                        const remaining = Math.max(0, totalBytes - shown);
                        const eta       = rate > 0 ? remaining / rate : Number.POSITIVE_INFINITY;
                        const rateLabel = rate > 0 ? `${fmtBytes(rate)}/s` : "—";
                        const etaLabel  = (rate > 0 && !done) ? ` · ETA ${fmtEta(eta)}` : "";
                        setLoadingLabel(`${fmtBytes(shown)} / ${fmtBytes(totalBytes)} — ${rateLabel}${etaLabel}`);
                    }
                } else {
                    const now = performance.now();
                    if (now - lastLabelAt > 250) {
                        lastLabelAt = now;
                        setLoadingLabel(fmtBytes(bytesWritten));
                    }
                }
            });
            if (fromCache) {
                beacon("chat", `OPFS cache hit (${fmtBytes(totalBytes)})`);
            } else {
                beacon("chat", `downloaded ${fmtBytes(totalBytes)} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
            }

            setLoadingLabel("loading into wasm…");
            const mobile = isMobileUA();
            // KV-cache cap on mobile. Was 2048 (~530 MB KV); dropped to
            // 1024 (~265 MB) for multimodal-capable checkpoints because
            // iPhone Safari WebContent peaks during text-prefill +
            // resident multimodal scratch were tipping over jetsam on
            // audio-attached Send. Text-only loads keep the 2048 ceiling
            // since there's no multimodal tower competing for budget.
            const mobileMaxCtx = m.multimodal ? 1024 : 2048;
            // textOnly policy:
            //   Catalog drives it everywhere — `multimodal: true` on a
            //   BAKED_IN_MODELS / R2 entry means the blob carries the
            //   full vision+audio tensor set; locally-discovered Ollama
            //   entries (no `multimodal` flag) and HF-style text-only
            //   blobs both read as text-only.
            //
            //   M16 lifted the previous mobile force-textOnly: the
            //   vision tower now ephemerally fetches per-block weights
            //   inside `encode_image` (peak ~200 MB during encode, ~5
            //   MB between encodes), and the audio tower does the same
            //   per-block fetch inside `encode_audio`. Total resident
            //   multimodal weight when idle is now tiny enough that
            //   iPhone can hold the text tower + the multimodal
            //   constants without jetsam.
            const textOnly = !m.multimodal;
            // DiffusionGemma is a SEPARATE wasm engine (own handle), not the
            // shared AR Model — load it through its own streaming loader. No
            // vision/audio towers; the denoise loop runs in onSend.
            if (isDiffusion(m)) {
                await client.diffusion.load(modelKey, filename, m.name);
                setHasVision(false);
                setHasAudio(false);
                setLoadedIsDiffusion(true);
            } else {
                // `load` is session-scoped (it mutates the shared Model). Acquire
                // for the duration of the wasm load + meta read. Other tabs'
                // inference will wait until this resolves; the queue advances
                // naturally on releaseSession (in the finally block).
                await client.withSession(async () => {
                    await client.load(modelKey, filename, {
                        maxContext: mobile ? mobileMaxCtx : 0,
                        textOnly,
                        name: m.name,
                    });
                });
                setHasVision(client.hasVision);
                setHasAudio(client.hasAudio);
                setLoadedIsDiffusion(false);
            }
            // **Preparing phase.** Pre-warm the system prompt into the KV
            // cache so the FIRST chat hot-starts instead of re-reading the
            // system block. Shown as a distinct "preparing" step in the same
            // progress bar / boot splash. Skipped for DiffusionGemma (no AR
            // KV cache). Best-effort: a warm failure just means the first
            // chat prefills the system once, as before.
            if (!isDiffusion(m) && onPrepareRef.current) {
                setModelStatus("preparing");
                setLoadingPercent(0);
                setLoadingLabel("preparing model…");
                try {
                    await onPrepareRef.current((percent, label) => {
                        setLoadingPercent(percent);
                        setLoadingLabel(label);
                    });
                } catch (e) {
                    console.warn("[rullama] system pre-warm failed:", e);
                }
            }
            setModelStatus("ready");
            setStatusText(`${m.name}${fromCache ? " ⚡" : ""}`);
            setLoadingLabel("");
            // Remember which model was loaded so a page reload can resume
            // automatically. Eject (or delete-while-loaded) clears this.
            // Clear pending — we're fully loaded, the boot-time auto-
            // resume path no longer applies until next deliberate load.
            setLastLoadedDigest(m.digest);
            setPendingLoadDigest("");
            const capsLine = `vision ${client.hasVision ? "✓" : "✗"} · audio ${client.hasAudio ? "✓" : "✗"}`;
            beacon("chat", `loaded ${m.name} (${capsLine})`);
            showToast({
                level: "success", title: `Loaded ${m.name}`,
                message: `${capsLine}${fromCache ? " · from OPFS cache" : ""}`,
            });
        } catch (e) {
            const err = (e as Error).message ?? String(e);
            // User pressed Stop — not a failure. The partial stays in OPFS so a
            // later Load resumes from where it stopped.
            if (/cancel/i.test(err)) {
                setModelStatus("idle");
                setStatusText("download stopped");
                setLoadingLabel("");
                setPendingLoadDigest("");
                if (deleteOnCancelRef.current) {
                    deleteOnCancelRef.current = false;
                    // The worker flushed + closed the write handle before
                    // rejecting, so the OPFS file is free to remove now.
                    // (Recompute the OPFS keys from `m` — the try-scoped
                    // modelKey/filename aren't visible here.)
                    const mk = m.digest.replace(/[^A-Za-z0-9_.-]/g, "_");
                    const fn = m.name.replace(/[^A-Za-z0-9_.-]/g, "_") + ".gguf";
                    try { await wipeModel(mk, fn); } catch { /* */ }
                    showToast({ level: "info", title: "Download stopped", message: "Partial deleted." });
                } else {
                    showToast({ level: "info", title: "Download stopped", message: "Resume any time by loading again." });
                }
                return;
            }
            setModelStatus("error");
            setStatusText(`load failed: ${err}`);
            setLoadingLabel("");
            showToast({
                id: "model-load-error", level: "error",
                title: "Model load failed", message: err,
            });
        }
    }, [dismissToast, setLastLoadedDigest, setPendingLoadDigest, showToast, confirm]);

    /** Stop an in-progress model download (the picker's Stop button). The
     *  active ensureModel rejects with "cancelled", handled above; the partial
     *  stays in OPFS for a later resume. */
    const onCancelDownload = useCallback(() => {
        deleteOnCancelRef.current = false;
        try { void getClient().cancelDownload(); } catch { /* */ }
    }, []);

    /** Stop the download AND delete the partial from OPFS (the Stop modal's
     *  "Delete partial"). The wipe runs in onLoad's cancel handler once the
     *  stream has actually aborted + released the OPFS write handle. */
    const onCancelAndDeleteDownload = useCallback(() => {
        deleteOnCancelRef.current = true;
        try { void getClient().cancelDownload(); } catch { /* */ }
    }, []);

    // Latest-onLoad ref so the auto-load effect doesn't need it as a dep
    // (which would re-trigger whenever onLoad's closure changes).
    const onLoadRef = useRef<((m: ModelEntry) => Promise<void>) | null>(null);
    // Keep the ref in sync so the mount-time auto-load effect can call
    // the latest onLoad closure without listing it as a hook dep (which
    // would re-fire the effect every time the closure identity changes).
    useEffect(() => { onLoadRef.current = onLoad; }, [onLoad]);

    // Auto-load on mount. Now gated on the SharedWorker's initial `meta`
    // notification — if another tab already loaded a model, we inherit
    // that state via the meta payload and skip auto-load entirely.
    //
    // Auto-rehydrate ONLY when the model is fully present in OPFS — that
    // means a previous session completed a download and just wants to
    // reopen the file. We deliberately do NOT auto-trigger a download
    // here, even when `pendingLoadDigest` says one was in flight before
    // the tab died: that auto-download path was the source of the
    // "every PWA reload after a screen lock tries to redownload the
    // model and fails" pain (the failure path on iOS is the
    // syncHandle / fetch race against an old worker that hasn't GC'd
    // yet, and it's much more reliable when the user clicks Load
    // explicitly because by then everything has settled).
    //
    // For partially-downloaded models, the ModelLoader UI shows the
    // resume option and the user can decide when to fire it.
    const autoLoadAttempted = useRef(false);
    useEffect(() => {
        if (!metaInit) return;
        // Don't boot the engine until the capability tier is known and the
        // device is supported — incapable devices (e.g. iPhone 7, no WebGPU)
        // must never load a model (that's what boot-loops them). The
        // UnsupportedScreen early-return handles the UI side.
        if (tier === null || tier === "unsupported") return;
        if (autoLoadAttempted.current) return;
        autoLoadAttempted.current = true;
        if (modelStatus === "ready") return;
        const target = pendingLoadDigest || lastLoadedDigest;
        if (!target) return;
        (async () => {
            try {
                const models = await listModels();
                const m = models.find((x) => x.digest === target);
                if (!m) {
                    setPendingLoadDigest("");
                    setLastLoadedDigest("");
                    return;
                }
                // Gate on OPFS: only auto-fire onLoad if the cached file
                // already matches the expected size. existingSize is
                // tolerant of the sync-handle read race (returns f.size
                // on transient failures), so a momentary lock conflict
                // doesn't push us into the download path.
                const modelKey = m.digest.replace(/[^A-Za-z0-9_.-]/g, "_");
                const filename = m.name.replace(/[^A-Za-z0-9_.-]/g, "_") + ".gguf";
                const cachedBytes = await existingSize(modelKey, filename);
                if (cachedBytes < m.size) {
                    // Not fully cached — bail. The ModelLoader UI will
                    // render and the user can tap Load to resume.
                    return;
                }
                if (onLoadRef.current) {
                    void onLoadRef.current(m);
                }
            } catch (e) {
                showToast({
                    level: "warn",
                    title: "Auto-load failed",
                    message: (e as Error).message,
                });
            }
        })();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [metaInit, modelStatus, tier]);

    // Re-load the inference model (used when returning to a chat/fine-tune tab after the core was
    // torn down to give the GPU to TTS). Mirrors the auto-load model pick.
    const reloadInferenceModel = useCallback(async () => {
        const target = pendingLoadDigest || lastLoadedDigest;
        if (!target) return;
        try {
            const models = await listModels();
            const m = models.find((x) => x.digest === target);
            if (m && onLoadRef.current) void onLoadRef.current(m);
        } catch { /* ModelLoader UI will let the user tap Load */ }
    }, [pendingLoadDigest, lastLoadedDigest]);

    const onDeleteModel = useCallback(async (m: ModelEntry) => {
        if (isBusy()) return;
        const sizeLabel = fmtBytes(m.size);
        const ok = window.confirm(
            `Delete cached "${m.name}" (${sizeLabel})?\n\n` +
            `This only removes the local copy in this browser. ` +
            `Loading again will re-download it.`,
        );
        if (!ok) return;

        const modelKey = m.digest.replace(/[^A-Za-z0-9_.-]/g, "_");
        const filename = m.name.replace(/[^A-Za-z0-9_.-]/g, "_") + ".gguf";

        const wasLoaded = modelStatus === "ready" && statusText.startsWith(m.name);
        if (wasLoaded) {
            try { await getClient().withSession(() => getClient().free()); } catch { /* */ }
            setModelStatus("idle");
            setStatusText("no model");
            setHasVision(false);
            setHasAudio(false);
            onUnloadChat();
            setLastLoadedDigest("");
        }
        // Clear pending too — deleting the cached file removes the
        // partial we'd otherwise auto-resume.
        if (pendingLoadDigest === m.digest) setPendingLoadDigest("");

        const removed = await wipeModel(modelKey, filename);
        beacon("chat", removed ? `deleted ${m.name} (${sizeLabel})` : `delete ${m.name} no-op (not cached)`);
        if (removed) {
            showToast({ level: "info", title: `Deleted ${m.name}`, message: `Freed ${sizeLabel} from OPFS` });
        } else {
            showToast({ level: "info", title: `No cached copy of ${m.name}` });
        }
    }, [modelStatus, pendingLoadDigest, setLastLoadedDigest, setPendingLoadDigest, statusText, showToast, onUnloadChat, isBusy]);

    /** Unload the active model, free its wasm-side buffers, and clear
     *  the "last loaded" memory so a page reload won't auto-resume.
     *  Past conversations stay in SQLite; OPFS-cached blobs stay on disk
     *  (use Delete to evict those). */
    const onEjectModel = useCallback(async () => {
        if (isBusy()) return;
        if (modelStatus !== "ready") return;
        // DiffusionGemma lives on its own handle (no AR Model session); unload
        // it directly. Otherwise free the shared Model.
        if (loadedIsDiffusion) {
            try { await getClient().diffusion.unload(); } catch { /* */ }
        } else {
            try { await getClient().free(); } catch { /* */ }
        }
        const name = statusText.split(" ")[0] || "model";
        setModelStatus("idle");
        setStatusText("no model");
        setHasVision(false);
        setHasAudio(false);
        setLoadedIsDiffusion(false);
        onUnloadChat();
        setLastLoadedDigest("");
        // Eject is a deliberate user gesture — they don't want the
        // boot-time auto-resume to refire either.
        setPendingLoadDigest("");
        showToast({ level: "info", title: `Ejected ${name}` });
    }, [modelStatus, loadedIsDiffusion, setLastLoadedDigest, setPendingLoadDigest, statusText, showToast, onUnloadChat, isBusy]);

    return {
        modelStatus, setModelStatus,
        loadedIsDiffusion,
        selectedModelName, setSelectedModelName,
        loadingPercent,
        loadingLabel,
        statusText, setStatusText,
        hasVision, setHasVision,
        hasAudio, setHasAudio,
        lastLoadedDigest,
        pendingLoadDigest,
        metaInit, setMetaInit,
        onLoad,
        onCancelDownload,
        onCancelAndDeleteDownload,
        reloadInferenceModel,
        onDeleteModel,
        onEjectModel,
    };
}
