import { useCallback, useEffect, useRef, useState } from "react";
import { ModelLoader, ModelLoadProgress, type ModelStatus } from "@/components/ModelLoader";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChatPanel } from "@/components/ChatPanel";
import type { VisionProgressState } from "@/components/VisionProgress";
import { RestartOverlay } from "@/components/RestartOverlay";
import { SettingsDialog, SETTINGS_BOUNDS } from "@/components/SettingsDialog";
import { ConversationList } from "@/components/ConversationList";
import { DualSidebarLayout } from "@/components/layouts/DualSidebarLayout";
import { Button } from "@/components/ui/button";
import { type ChatMessage, type ImageAttachment, type SamplingOptions, DEFAULT_SAMPLING, DEFAULT_SYSTEM_PROMPT } from "@/lib/types";
import { type ModelEntry, blobUrl, beacon, listModels } from "@/lib/api";
import { ensureModel, existingSize, opfsSupported, requestPersistent, wipeModel, readInflightState, writeInflightState, clearInflightState } from "@/lib/opfs";
import { saveInflightImage, saveInflightAudio, readInflightImages, readInflightAudio, clearInflightMedia } from "@/lib/inflight_media";
import { getNetworkHint } from "@/lib/network";
import { getClient, type ConversationRow } from "@/lib/inference";
import { useToast } from "@/lib/toast";
import { usePersistedState } from "@/lib/persisted";
import { useIOSKeyboard } from "@/lib/useIOSKeyboard";
import { useWakeLock } from "@/lib/wakeLock";
import { fmtBytes, fmtEta, clampInt, clampNum } from "@/lib/utils";
import { preprocessImage } from "@/lib/image_preprocess";
import { saveThumb, loadThumbBlobUrl, deleteThumbs } from "@/lib/image_store";
import { DEFAULT_VOICE_OPTIONS, VOICE_BOUNDS, type VoiceOptions } from "@/lib/voice";
import { Settings, History } from "lucide-react";

const isMobileUA = () => /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

const THINK_TOKEN = "<|think|>";
const TITLE_MAX_LEN = 40;

const INFLIGHT_KEY = "rullama:inflight";

// Per-step timeout for stepAndDecode. iOS suspension can leave the
// awaited Promise hanging if the dedicated worker was killed; this
// gives us a deterministic detection point. setTimeout is paused
// while JS is suspended, so on foreground-after-kill the timer is
// already past-due and fires immediately — recovery kicks in within
// one task. 8 s sits ~40× a normal iPhone step (~200 ms @ 5 tok/s)
// and ~8× a slow-Mac step, well clear of normal variance.
const STEP_TIMEOUT_MS = 8_000;

// Persisted snapshot of the currently-running generation. On
// visibilitychange→hidden we mirror this into localStorage; on boot
// (or live-tab timeout recovery) we read it back and use it together
// with the OPFS-stored KV snapshot to resume.
interface InflightGen {
    convId: string;
    modelMsgId: string;
    modelDigest: string;
    userText: string;
    sysContent: string;
    priorMessages: ChatMessage[];
    sampling: SamplingOptions;
    maxTokens: number;
    /** Tokenized prompt — full sequence the gen loop runs against.
     *  Persisted so a pre-encode-phase resume can replay only the
     *  prompt tokens that hadn't been fed yet at suspension time.
     *  Empty until the prompt has been tokenized in onSend. */
    promptIds: number[];
    /** Number of `promptIds[i]` already fed via `step()` at snapshot
     *  time. 0 means pre-encode hasn't started; equals `promptIds.length`
     *  once pre-encode completes and the gen loop is running. */
    preEncodePosition: number;
    emittedSoFar: string;
    emittedTokenCount: number;
    lastSampledNext: number;
    hadImages: boolean;
    hadAudio: boolean;
    /** True once turnImages + turnAudio have been written to OPFS
     *  successfully. False means a kill-and-resume on this turn can't
     *  re-encode multimodal media — we surface the "interrupted"
     *  toast in that narrow case rather than producing wrong output. */
    mediaPersisted: boolean;
    startedAt: number;
}

/** Promise.race wrapper that aborts a single step() call if it hangs.
 *  iOS suspension can kill the dedicated worker while leaving its
 *  Promise unresolved on the tab; STEP_TIMEOUT_MS is the threshold past
 *  which we declare the worker dead and fall through to the resume
 *  path. Set to 30 s — generous enough that iPhone @ ~5 tok/s never
 *  trips it normally. */
function stepWithTimeout(
    client: ReturnType<typeof getClient>,
    tokenId: number,
): Promise<{ next: number; isEos: boolean; str: string | null }> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    return new Promise((resolve, reject) => {
        const timeoutErr = new Error("step-timeout: generation hung (worker may be dead)");
        timer = setTimeout(() => reject(timeoutErr), STEP_TIMEOUT_MS);
        client.stepAndDecode(tokenId).then(
            (r) => { if (timer) clearTimeout(timer); resolve(r); },
            (e) => { if (timer) clearTimeout(timer); reject(e); },
        );
    });
}

function suggestTitle(text: string): string {
    const t = text.trim().replace(/\s+/g, " ");
    return t.length <= TITLE_MAX_LEN ? t : t.slice(0, TITLE_MAX_LEN - 1) + "…";
}

export function App() {
    // Model load state
    const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
    const [loadingPercent, setLoadingPercent] = useState(0);
    const [loadingLabel, setLoadingLabel]     = useState("");
    const [statusText, setStatusText]         = useState("no model");

    // Chat state
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [prompt, setPrompt]     = useState("");
    const [busy, setBusy]         = useState(false);
    const [statusLine, setStatusLine] = useState<string | undefined>();
    const [visionEncodeState, setVisionEncodeState] = useState<VisionProgressState | null>(null);

    // Multimodal: vision availability latches on after a successful model
    // load (it's a property of the meta, only known post-load). Pending
    // images are session-only — cleared after each send.
    const [hasVision, setHasVision]   = useState(false);
    const [hasAudio,  setHasAudio]    = useState(false);
    const [pendingImages, setPendingImages] = useState<ImageAttachment[]>([]);
    // Voice clips attached to the next user turn. Mirrors `pendingImages`.
    // PCM stays in-memory; we don't persist past sends (analogous to
    // image pixels — only thumbs / transcripts would land in SQLite,
    // and we don't have either yet for audio).
    const [pendingAudio, setPendingAudio] = useState<{ pcm: Float32Array; durationMs: number }[]>([]);

    // Conversation persistence (rsqlite-wasm)
    const [conversations, setConversations] = useState<ConversationRow[]>([]);
    const [activeConvId, setActiveConvId]   = useState<string | null>(null);

    // Sidebar visibility — persisted across reloads. Settings defaults
    // open on first ever load so the model picker is visible.
    const [historyOpen,  setHistoryOpen]  = usePersistedState<boolean>("ui.historyOpen",  false);
    const [settingsOpen, setSettingsOpen] = usePersistedState<boolean>("ui.settingsOpen", true);

    // Persisted tunables.
    const [systemPrompt, setSystemPrompt] = usePersistedState<string>("systemPrompt", DEFAULT_SYSTEM_PROMPT);
    const [sampling,     setSampling]     = usePersistedState<SamplingOptions>("sampling", DEFAULT_SAMPLING);
    const [maxTokens,    setMaxTokens]    = usePersistedState<number>("maxTokens", 1024);
    const [thinking,     setThinking]     = usePersistedState<boolean>("thinking", true);
    const [voice,        setVoice]        = usePersistedState<VoiceOptions>("voice", DEFAULT_VOICE_OPTIONS);

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

    const cancelRef = useRef(false);
    // Tracks the currently-running generation for suspend/resume. Mutated
    // per-token in the gen loop; serialized to localStorage on
    // visibilitychange→hidden so a kill-and-resume can pick up where we
    // left off. Cleared on clean completion / explicit cancel.
    const inflightRef = useRef<InflightGen | null>(null);
    // Resume-on-boot is single-shot; the effect can fire multiple times
    // as model state changes, but we want to attempt resume at most once.
    const resumeAttemptedRef = useRef(false);
    const { showToast, dismissToast } = useToast();

    // iOS keyboard handling — snaps the visual viewport back to the top
    // when the keyboard dismisses, so the page doesn't end up offset
    // a few px above the layout viewport (a classic iOS-Safari quirk).
    useIOSKeyboard(true);

    // Hold the screen awake during the two long-running operations a
    // user actually waits on — model download/load and token generation.
    // No-op on platforms without `navigator.wakeLock` (older iOS, private
    // mode). See `lib/wakeLock.ts` for the iOS hide-release dance.
    useWakeLock(modelStatus === "loading" || busy);

    // One-time sanitization of persisted values. Catches localStorage
    // entries from older versions (or hand-edited values) that fall
    // outside current bounds — the slider clamps cover fresh edits.
    useEffect(() => {
        const B = SETTINGS_BOUNDS;
        const next: SamplingOptions = {
            temperature:        clampNum(sampling.temperature, B.temperature.min, B.temperature.max, B.temperature.fallback),
            top_k:              clampInt(sampling.top_k,       B.top_k.min,       B.top_k.max,       B.top_k.fallback),
            top_p:              clampNum(sampling.top_p,       B.top_p.min,       B.top_p.max,       B.top_p.fallback),
            repetition_penalty: clampNum(sampling.repetition_penalty, B.repetition_penalty.min, B.repetition_penalty.max, B.repetition_penalty.fallback),
            seed:               Number.isFinite(sampling.seed) ? sampling.seed : 0,
        };
        if (next.temperature !== sampling.temperature
            || next.top_k !== sampling.top_k
            || next.top_p !== sampling.top_p
            || next.repetition_penalty !== sampling.repetition_penalty
            || next.seed !== sampling.seed) {
            setSampling(next);
        }
        const mt = clampInt(maxTokens, B.maxTokens.min, B.maxTokens.max, B.maxTokens.fallback);
        if (mt !== maxTokens) setMaxTokens(mt);

        const VB = VOICE_BOUNDS;
        const nextVoice: VoiceOptions = {
            silenceMs:       clampInt(voice.silenceMs,       VB.silenceMs.min,       VB.silenceMs.max,       VB.silenceMs.fallback),
            rmsDbThreshold:  clampInt(voice.rmsDbThreshold,  VB.rmsDbThreshold.min,  VB.rmsDbThreshold.max,  VB.rmsDbThreshold.fallback),
            prerollMs:       clampInt(voice.prerollMs,       VB.prerollMs.min,       VB.prerollMs.max,       VB.prerollMs.fallback),
            minSpeechFrames: clampInt(voice.minSpeechFrames, VB.minSpeechFrames.min, VB.minSpeechFrames.max, VB.minSpeechFrames.fallback),
            maxRecordMs:     clampInt(voice.maxRecordMs,     VB.maxRecordMs.min,     VB.maxRecordMs.max,     VB.maxRecordMs.fallback),
        };
        if (nextVoice.silenceMs !== voice.silenceMs
            || nextVoice.rmsDbThreshold !== voice.rmsDbThreshold
            || nextVoice.prerollMs !== voice.prerollMs
            || nextVoice.minSpeechFrames !== voice.minSpeechFrames
            || nextVoice.maxRecordMs !== voice.maxRecordMs) {
            setVoice(nextVoice);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

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

    // Global Escape → stop generation. Mirrors the toolbar Stop button.
    // Attached to window so it fires regardless of focus (input, sidebar
    // toggle, anything). No-op when nothing is generating.
    useEffect(() => {
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape" && busy) {
                cancelRef.current = true;
            }
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [busy]);

    // Suspend-on-background. iOS Safari fires visibilitychange→hidden
    // before suspending the WebContent process; we use that window to
    // (a) sync the inflight metadata to localStorage and (b) kick off
    // a GPU-state snapshot to OPFS. If iOS yanks us mid-write the
    // boot-resume fast path will be unavailable and the slow-path
    // replay (Phase F) picks up the slack via the partial response
    // already in the DB.
    useEffect(() => {
        if (!busy) return;
        const onVis = () => {
            if (document.visibilityState !== "hidden") return;
            const meta = inflightRef.current;
            if (!meta) return;
            try { localStorage.setItem(INFLIGHT_KEY, JSON.stringify(meta)); } catch { /* */ }
            // KV snapshot is async (GPU readback + OPFS write). Fire-and-
            // forget: a failure here just means we fall back to replay
            // on resume. The setTimeout(0) yields to the browser so the
            // visibilitychange callback returns quickly, letting iOS
            // begin its suspension countdown with our metadata already
            // persisted.
            setTimeout(() => {
                void (async () => {
                    try {
                        const client = getClient();
                        const bytes = await client.saveKvState();
                        await writeInflightState(bytes);
                    } catch (e) {
                        // eslint-disable-next-line no-console
                        console.warn("[rullama] inflight KV snapshot failed:", e);
                    }
                })();
            }, 0);
        };
        document.addEventListener("visibilitychange", onVis);
        return () => document.removeEventListener("visibilitychange", onVis);
    }, [busy]);

    // Bootstrap DB + conversation list on mount.
    useEffect(() => {
        const client = getClient();
        (async () => {
            try {
                await client.dbInit();
                const rows = await client.convList();
                setConversations(rows);
            } catch (e) {
                showToast({ level: "error", title: "Database init failed", message: (e as Error).message });
            }
        })();
    }, [showToast]);

    // Has the SharedWorker pushed at least one `meta` notification? Until
    // it has, we don't know whether another tab already loaded a model,
    // so auto-load is gated on this flag (see effect below).
    const [metaInit, setMetaInit] = useState(false);

    // Cross-tab state sync. The SharedWorker broadcasts notifications
    // whenever shared state changes; each tab reflects them in React
    // state so all tabs stay aligned.
    useEffect(() => {
        const client = getClient();
        const applyLoaded = (loaded: {
            name?: string | null;
            modelKey?: string;
            hasVision?: boolean;
            hasAudio?: boolean;
        } | null | undefined) => {
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
                applyLoaded(p.loaded as Parameters<typeof applyLoaded>[0]);
            }),
            client.subscribe("modelLoaded", (p) => {
                applyLoaded(p as Parameters<typeof applyLoaded>[0]);
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
        ];
        return () => { for (const o of offs) o(); };
    }, []);

    // Latest-onLoad ref so the auto-load effect doesn't need it as a dep
    // (which would re-trigger whenever onLoad's closure changes).
    const onLoadRef = useRef<((m: ModelEntry) => Promise<void>) | null>(null);

    // Auto-load on mount. Now gated on the SharedWorker's initial `meta`
    // notification — if another tab already loaded a model, we inherit
    // that state via the meta payload and skip auto-load entirely.
    //
    // Resume priority: `pendingLoadDigest` first (the user kicked off a
    // load that didn't finish — likely an iOS suspend mid-download;
    // re-firing onLoad routes through ensureModel's Range-resume); then
    // `lastLoadedDigest` (previously fully loaded, just rehydrate from
    // OPFS cache). Both paths funnel through the same onLoad call.
    const autoLoadAttempted = useRef(false);
    useEffect(() => {
        if (!metaInit) return;
        if (autoLoadAttempted.current) return;
        autoLoadAttempted.current = true;
        // The meta subscriber already set modelStatus when a model is
        // loaded worker-side; bail out so we don't gratuitously call
        // load() and queue behind another tab.
        if (modelStatus === "ready") return;
        const target = pendingLoadDigest || lastLoadedDigest;
        if (!target) return;
        (async () => {
            try {
                const models = await listModels();
                const m = models.find((x) => x.digest === target);
                if (!m) {
                    // Model vanished from the catalog (server-side delete,
                    // or stale localStorage). Clear both so we don't keep
                    // retrying on every boot.
                    setPendingLoadDigest("");
                    setLastLoadedDigest("");
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
    }, [metaInit, modelStatus]);

    const refreshConversations = useCallback(async () => {
        try {
            const rows = await getClient().convList();
            setConversations(rows);
        } catch (e) {
            showToast({ level: "warn", title: "Could not refresh conversations", message: (e as Error).message });
        }
    }, [showToast]);

    const onSelectConversation = useCallback(async (id: string) => {
        if (busy) return;
        try {
            const client = getClient();
            // Fire both queries in parallel — text rows and image rows are
            // independent; SQLite handles concurrent reads fine.
            const [rows, imgRows] = await Promise.all([
                client.msgList(id),
                client.msgListImages(id),
            ]);
            // Bucket images by message_id, preserving the `seq` order
            // the table is already sorted by. The thumbnail bytes live
            // in OPFS keyed by opfs_path; we resolve each to a `blob:`
            // URL the browser can render directly (no base64 decode,
            // no DOM repaint cost compared to inline data URLs).
            const imagesByMsg = new Map<string, ImageAttachment[]>();
            const resolved = await Promise.all(
                imgRows.map(async (r) => ({
                    row: r,
                    url: await loadThumbBlobUrl(r.opfs_path),
                })),
            );
            for (const { row, url } of resolved) {
                if (!url) continue;   // file vanished; skip rather than show broken img
                const arr = imagesByMsg.get(row.message_id) ?? [];
                arr.push({
                    h:       row.height,
                    w:       row.width,
                    dataUrl: url,
                    // pixels intentionally omitted — reloaded images are
                    // display-only. The LM doesn't re-encode past-turn
                    // images today (see renderPriorTurns in onSend).
                });
                imagesByMsg.set(row.message_id, arr);
            }
            const ms: ChatMessage[] = rows
                .filter((r) => r.role === "user" || r.role === "model")
                .map((r) => {
                    const images = imagesByMsg.get(r.message_id);
                    return {
                        role:    r.role as ChatMessage["role"],
                        content: r.content,
                        ...(images && images.length ? { images } : {}),
                    };
                });
            setMessages(ms);
            setActiveConvId(id);
            setStatusLine(undefined);
        } catch (e) {
            showToast({ level: "error", title: "Failed to load conversation", message: (e as Error).message });
        }
    }, [busy, showToast]);

    const onResetDefaults = useCallback(() => {
        setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
        setSampling(DEFAULT_SAMPLING);
        setMaxTokens(SETTINGS_BOUNDS.maxTokens.fallback);
        setThinking(true);
        setVoice(DEFAULT_VOICE_OPTIONS);
        showToast({
            level: "success",
            title: "Settings reset to defaults",
        });
    }, [setSystemPrompt, setSampling, setMaxTokens, setThinking, setVoice, showToast]);

    const onCreateConversation = useCallback(() => {
        if (busy) return;
        setMessages([]);
        setActiveConvId(null);
        setStatusLine(undefined);
        setPendingImages([]);
        // The worker's KV cache is shared across tabs now, so a "new chat"
        // here doesn't preemptively reset it — onSend will reset inside
        // its own session window. (Resetting from here would either need
        // a session we don't have, or be racy with another tab mid-step.)
    }, [busy]);

    const onAttachFiles = useCallback(async (files: FileList) => {
        // Reject silently when vision is unavailable — the UI gating
        // should already prevent the click, but a stale handler still
        // could fire mid-unload.
        if (!hasVision) return;
        const next: ImageAttachment[] = [];
        for (let i = 0; i < files.length; i++) {
            const f = files[i];
            if (!f.type.startsWith("image/")) continue;
            try {
                const p = await preprocessImage(f);
                next.push(p);
            } catch (e) {
                showToast({
                    level: "error", title: `Couldn't load ${f.name}`,
                    message: (e as Error).message,
                });
            }
        }
        if (next.length) setPendingImages((prev) => [...prev, ...next]);
    }, [hasVision, showToast]);

    const onRemoveImage = useCallback((idx: number) => {
        setPendingImages((prev) => prev.filter((_, i) => i !== idx));
    }, []);

    const onDeleteConversation = useCallback(async (id: string) => {
        if (busy) return;
        const c = conversations.find((x) => x.id === id);
        if (!window.confirm(`Delete conversation "${c?.title ?? id}"?\n\nMessages cannot be recovered.`)) return;
        try {
            // convDelete returns the OPFS image paths it collected
            // before the FK cascade nuked the rows; we sweep them
            // here so the on-disk thumbnails don't orphan.
            const { opfsPaths } = await getClient().convDelete(id);
            if (opfsPaths.length > 0) {
                void deleteThumbs(opfsPaths);
            }
            await refreshConversations();
            if (id === activeConvId) {
                setActiveConvId(null);
                setMessages([]);
            }
        } catch (e) {
            showToast({ level: "error", title: "Delete failed", message: (e as Error).message });
        }
    }, [activeConvId, busy, conversations, refreshConversations, showToast]);

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
                const head = hint.metered
                    ? `⚠️ ${hint.reason}.`
                    : `Heads up:`;
                const msg = `${head}\n\nDownloading "${m.name}" needs ${sizeLabel} over the network. ` +
                    `It will be cached locally so subsequent loads are free.\n\n` +
                    `Continue?`;
                if (!window.confirm(msg)) {
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
                    setLoadingPercent((bytesWritten / totalBytes) * 100);
                    const now  = performance.now();
                    const done = bytesWritten >= totalBytes;
                    if (baselineBytes < 0) {
                        baselineBytes = bytesWritten;
                        baselineAt    = now;
                    }
                    if (done || now - lastLabelAt > 250) {
                        lastLabelAt = now;
                        const elapsed   = (now - baselineAt) / 1000;
                        const delta     = Math.max(0, bytesWritten - baselineBytes);
                        const rate      = elapsed > 0.25 ? delta / elapsed : 0;
                        const remaining = Math.max(0, totalBytes - bytesWritten);
                        const eta       = rate > 0 ? remaining / rate : Number.POSITIVE_INFINITY;
                        const rateLabel = rate > 0 ? `${fmtBytes(rate)}/s` : "—";
                        const etaLabel  = (rate > 0 && !done) ? ` · ETA ${fmtEta(eta)}` : "";
                        setLoadingLabel(`${fmtBytes(bytesWritten)} / ${fmtBytes(totalBytes)} — ${rateLabel}${etaLabel}`);
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
            // KV-cache cap on mobile only — the original 512 was a
            // crash-avoidance number from the iPhone-load-path work, but
            // since the worker + sync-OPFS + per-tile range-fetch combo
            // stabilised, 2048 ctx (≈ 144 MB GPU on A18) is comfortable.
            const mobileMaxCtx = 2048;
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
            setModelStatus("error");
            setStatusText(`load failed: ${err}`);
            setLoadingLabel("");
            showToast({
                id: "model-load-error", level: "error",
                title: "Model load failed", message: err,
            });
        }
    }, [dismissToast, setLastLoadedDigest, setPendingLoadDigest, showToast]);

    // Keep the ref in sync so the mount-time auto-load effect can call
    // the latest onLoad closure without listing it as a hook dep (which
    // would re-fire the effect every time the closure identity changes).
    useEffect(() => { onLoadRef.current = onLoad; }, [onLoad]);

    // ── Suspend / resume: pick up an interrupted generation ────────────
    //
    // Resume flow on detection (boot, or live-tab post-timeout):
    //   1. Restore displayHistory from metadata so the user sees the
    //      partial response immediately.
    //   2. Try the fast path: read the OPFS KV snapshot and
    //      restoreKvState() — Model.position lands on the saved value
    //      and we step from there.
    //   3. If the snapshot is missing / corrupt / from a different
    //      architecture, fall back to slow-path replay: render the
    //      conversation (including partial assistant text) via
    //      renderChatForContinuation, tokenize, reset, and step
    //      through every token to rebuild KV. Slower (a few seconds)
    //      but always correct.
    //   4. Enter the gen loop continuing from `lastSampledNext`,
    //      appending into the existing assistant bubble.
    //   5. On clean completion: clear localStorage + OPFS file.
    const resumeInflightGeneration = useCallback(async (meta: InflightGen) => {
        const client = getClient();
        cancelRef.current = false;
        setBusy(true);

        // Reconstruct the visible chat so the user sees the partial
        // response while we work to continue it.
        const display: ChatMessage[] = [
            ...meta.priorMessages,
            { role: "user",  content: meta.userText },
            { role: "model", content: meta.emittedSoFar },
        ];
        setMessages(display);
        setActiveConvId(meta.convId);
        inflightRef.current = meta;

        try {
            setStatusLine("resuming…");
            await client.acquireSession();

            // Cross-tab race: a sibling tab may have already done the
            // resume (its completion cleared localStorage + OPFS).
            // After acquireSession serialized us, recheck localStorage
            // for a still-live inflight entry that matches our
            // captured meta. If it's gone or now refers to a different
            // generation, the sibling already handled it — bail
            // cleanly instead of mangling state.
            let raw: string | null = null;
            try { raw = localStorage.getItem(INFLIGHT_KEY); } catch { /* */ }
            let stillOurs = true;
            if (!raw) {
                stillOurs = false;
            } else {
                try {
                    const live = JSON.parse(raw) as InflightGen;
                    if (live.convId !== meta.convId
                        || live.modelMsgId !== meta.modelMsgId
                        || live.startedAt !== meta.startedAt) {
                        stillOurs = false;
                    }
                } catch { stillOurs = false; }
            }
            if (!stillOurs) {
                setStatusLine(undefined);
                try { await client.releaseSession(); } catch { /* */ }
                inflightRef.current = null;
                setBusy(false);
                // No media cleanup — sibling tab's finally will handle it.
                return;
            }

            await client.setSampling(meta.sampling);

            const isMultimodal = meta.hadImages || meta.hadAudio;
            // For a multimodal turn that hadn't yet finished pre-encode,
            // the fast-path catchup can't replay correctly (it would
            // feed sentinel tokens as plain text). Force slow path,
            // which renders + re-encodes media + splices soft tokens.
            const forceSlowPath = isMultimodal
                && meta.emittedTokenCount === 0
                && meta.preEncodePosition < meta.promptIds.length;

            // Fast path: KV snapshot from OPFS.
            let kvIntact = false;
            if (!forceSlowPath) {
                const snapshotBytes = await readInflightState();
                if (snapshotBytes && snapshotBytes.length > 0) {
                    try {
                        await client.restoreKvState(snapshotBytes);
                        kvIntact = true;
                    } catch (e) {
                        // eslint-disable-next-line no-console
                        console.warn("[rullama] KV restore failed; falling back to replay:", e);
                    }
                }
            }

            // Slow path: rebuild KV from a continuation render. For
            // multimodal we re-encode the images/audio from the OPFS
            // inflight-media store, then splice soft tokens at the
            // matching sentinel positions — same protocol onSend uses
            // on first-pass.
            let nextToken = meta.lastSampledNext;
            if (!kvIntact) {
                setStatusLine("rebuilding session…");
                await client.reset();

                type SoftEntry = { nSoft: number; dText: number; softTokens: Float32Array };
                const softQueue: SoftEntry[] = [];
                const audioQueue: SoftEntry[] = [];
                let beginId: number | null = null;
                let audioBeginId: number | null = null;
                let nImages = 0;
                let nAudio = 0;

                if (isMultimodal) {
                    const images = meta.hadImages ? await readInflightImages() : [];
                    const audio  = meta.hadAudio  ? await readInflightAudio()  : [];
                    nImages = images.length;
                    nAudio  = audio.length;

                    if (images.length > 0) {
                        const sent = client.imageSentinelIds();
                        if (!sent) throw new Error("model exposes no <|image> sentinel");
                        beginId = sent[0];
                        setStatusLine("re-encoding images…");
                        for (const im of images) {
                            if (cancelRef.current) throw new Error("cancelled");
                            if (!im.pixels) continue;
                            const softTokens = await client.encodeImage(im.pixels, im.h, im.w);
                            const nSoft = await client.imageSoftTokenCount(im.h, im.w);
                            const dText = softTokens.length / nSoft;
                            softQueue.push({ nSoft, dText, softTokens });
                        }
                    }
                    if (audio.length > 0) {
                        const sent = client.audioSentinelIds();
                        if (!sent) throw new Error("model exposes no <|audio> sentinel");
                        audioBeginId = sent[0];
                        const fallbackDText = softQueue[0]?.dText ?? 1536;
                        setStatusLine("re-encoding audio…");
                        for (const clip of audio) {
                            if (cancelRef.current) throw new Error("cancelled");
                            const softTokens = await client.encodeAudio(clip.pcm);
                            const dText = fallbackDText;
                            const nSoft = softTokens.length / dText;
                            audioQueue.push({ nSoft, dText, softTokens });
                        }
                    }
                }

                // Reconstruct the user content with the same sentinel
                // markers `onSend` originally prepended (audio first,
                // then images, then the user text).
                const audioMarkers = "<|audio><audio|>".repeat(nAudio);
                const imageMarkers = "<|image><image|>".repeat(nImages);
                const userContent = audioMarkers + imageMarkers + meta.userText;

                const renderMsgs: ChatMessage[] = [
                    ...(meta.sysContent
                        ? [{ role: "system", content: meta.sysContent } as ChatMessage]
                        : []),
                    ...meta.priorMessages,
                    { role: "user",  content: userContent },
                    { role: "model", content: meta.emittedSoFar },
                ];
                const rendered = await client.renderChatForContinuation(renderMsgs, false);
                const ids = await client.encode(rendered);

                setStatusLine("rebuilding session…");
                let n = 0;
                for (let i = 0; i < ids.length; i++) {
                    if (cancelRef.current) throw new Error("cancelled");
                    const id = ids[i];
                    n = await client.step(id);
                    if (beginId !== null && id === beginId && softQueue.length > 0) {
                        const ent = softQueue.shift()!;
                        for (let r = 0; r < ent.nSoft; r++) {
                            if (cancelRef.current) throw new Error("cancelled");
                            const row = ent.softTokens.subarray(r * ent.dText, (r + 1) * ent.dText);
                            n = await client.stepWithEmbedding(row);
                        }
                    } else if (audioBeginId !== null && id === audioBeginId && audioQueue.length > 0) {
                        const ent = audioQueue.shift()!;
                        for (let r = 0; r < ent.nSoft; r++) {
                            if (cancelRef.current) throw new Error("cancelled");
                            const row = ent.softTokens.subarray(r * ent.dText, (r + 1) * ent.dText);
                            n = await client.stepWithEmbedding(row);
                        }
                    }
                }
                nextToken = n;
            } else if (meta.preEncodePosition < meta.promptIds.length) {
                // Fast-path pre-encode catchup: KV restore landed us
                // at `preEncodePosition`; feed the remaining prompt
                // tokens before entering the gen loop. Update
                // inflightRef per iter so a re-suspension during
                // catchup is itself resumable.
                setStatusLine("resuming pre-encode…");
                let n = meta.lastSampledNext;
                for (let i = meta.preEncodePosition; i < meta.promptIds.length; i++) {
                    if (cancelRef.current) throw new Error("cancelled");
                    n = await client.step(meta.promptIds[i]);
                    if (inflightRef.current) {
                        inflightRef.current = {
                            ...inflightRef.current,
                            preEncodePosition: i + 1,
                            lastSampledNext: n,
                        };
                    }
                }
                nextToken = n;
            }
            setStatusLine(undefined);

            // ── Generation loop, resumed from the saved `next` token ──
            const remaining = Math.max(0, meta.maxTokens - meta.emittedTokenCount);
            let emitted = meta.emittedTokenCount;
            let curStr   = (await client.tokenStr(nextToken)) ?? "";
            let curIsEos = await client.isEos(nextToken);
            let pendingDelta = "";
            let lastFlushAt  = performance.now();
            const flushPending = async () => {
                if (pendingDelta.length === 0) return;
                const delta = pendingDelta;
                pendingDelta = "";
                try { await client.msgAppend(meta.convId, meta.modelMsgId, delta); } catch { /* */ }
            };

            for (let i = 0; i < remaining; i++) {
                if (cancelRef.current) break;
                if (curIsEos) break;
                const piece = curStr.replaceAll("▁", " ");
                display[display.length - 1].content += piece;
                pendingDelta += piece;
                setMessages([...display]);
                emitted++;
                if ((emitted % 16 === 0) || (performance.now() - lastFlushAt > 750)) {
                    await flushPending();
                    lastFlushAt = performance.now();
                }
                const r = await stepWithTimeout(client, nextToken);
                nextToken = r.next;
                curStr    = r.str ?? "";
                curIsEos  = r.isEos;
                if (inflightRef.current) {
                    inflightRef.current = {
                        ...inflightRef.current,
                        emittedSoFar: display[display.length - 1].content,
                        emittedTokenCount: emitted,
                        lastSampledNext: nextToken,
                    };
                }
            }
            await flushPending();

            try {
                await client.convTouch(meta.convId, suggestTitle(meta.userText));
                await client.dbFlush();
            } catch { /* */ }
            void refreshConversations();
            setStatusLine("resumed");
        } catch (e) {
            const msg = (e as Error).message;
            const isCancel = msg === "cancelled" || msg.includes("cancelled by caller");
            setStatusLine(isCancel ? "cancelled" : `resume error: ${msg}`);
            if (!isCancel) {
                showToast({ level: "warn", title: "Resume failed", message: msg });
            }
        } finally {
            inflightRef.current = null;
            try { localStorage.removeItem(INFLIGHT_KEY); } catch { /* */ }
            void clearInflightState();
            void clearInflightMedia();
            try { await client.releaseSession(); } catch { /* */ }
            setBusy(false);
        }
    }, [refreshConversations, showToast]);

    // Boot-resume: once the model is ready, check for a stashed
    // inflight generation and pick it up where it left off.
    useEffect(() => {
        if (modelStatus !== "ready") return;
        if (resumeAttemptedRef.current) return;
        resumeAttemptedRef.current = true;

        let raw: string | null = null;
        try { raw = localStorage.getItem(INFLIGHT_KEY); } catch { return; }
        if (!raw) return;

        let meta: InflightGen;
        try { meta = JSON.parse(raw) as InflightGen; }
        catch {
            try { localStorage.removeItem(INFLIGHT_KEY); } catch { /* */ }
            void clearInflightState();
            void clearInflightMedia();
            return;
        }

        // Discard mismatched-model resumes (the catalog auto-load
        // policy might have picked a different model on boot).
        if (lastLoadedDigest && meta.modelDigest && meta.modelDigest !== lastLoadedDigest) {
            try { localStorage.removeItem(INFLIGHT_KEY); } catch { /* */ }
            void clearInflightState();
            void clearInflightMedia();
            return;
        }

        // Multimodal: resume only when the media was persisted to OPFS
        // before suspension. If the media-persist race lost (iOS
        // suspended us mid-write or before mediaPersisted=true), fall
        // back to surfacing the partial response with the
        // "interrupted" toast and let the user re-send — we can't
        // re-encode media we don't have.
        const isMultimodal = meta.hadImages || meta.hadAudio;
        if (isMultimodal && !meta.mediaPersisted) {
            const display: ChatMessage[] = [
                ...meta.priorMessages,
                { role: "user",  content: meta.userText },
                { role: "model", content: meta.emittedSoFar },
            ];
            setMessages(display);
            setActiveConvId(meta.convId);
            showToast({
                level: "info",
                title: "Multimodal generation interrupted",
                message: "Partial response is preserved. Re-send the message to get a full reply.",
            });
            try { localStorage.removeItem(INFLIGHT_KEY); } catch { /* */ }
            void clearInflightState();
            void clearInflightMedia();
            return;
        }

        void resumeInflightGeneration(meta);
    }, [modelStatus, lastLoadedDigest, resumeInflightGeneration, showToast]);

    const onSend = useCallback(async () => {
        if (modelStatus !== "ready" || busy) return;
        const text = prompt.trim();
        // Snapshot attachments for this turn so the UI can clear them
        // while generation runs.
        const turnImages = pendingImages;
        const turnAudio  = pendingAudio;
        if (!text && turnImages.length === 0 && turnAudio.length === 0) return;
        const client = getClient();
        cancelRef.current = false;
        // Flagged by the catch when a recoverable hang is detected so
        // the finally block can hand off to resumeInflightGeneration
        // instead of clearing inflight state.
        let liveRecovery: InflightGen | null = null;
        setBusy(true);
        setPrompt("");
        setPendingImages([]);
        setPendingAudio([]);
        setStatusLine(undefined);

        const sysContent = thinking
            ? (systemPrompt.trim() ? `${THINK_TOKEN}${systemPrompt.trim()}` : THINK_TOKEN)
            : systemPrompt.trim();

        // Two parallel histories: `displayHistory` for the chat UI (no
        // system message — it's a control signal, not user content) and
        // `renderHistory` for tokenization (with system at the front).
        //
        // The renderer-side content prepends one `<|image><image|>`
        // sentinel pair per attached image; JS splices the soft-token
        // embeddings between them during the feed loop. Display bubbles
        // show only the typed text + thumbnails — sentinels never reach
        // the user.
        const userTurns = messages.filter((m) => m.role !== "system");
        const userDisplayMsg: ChatMessage = {
            role: "user",
            content: text,
            ...(turnImages.length ? { images: turnImages } : {}),
        };
        const displayHistory: ChatMessage[] = [
            ...userTurns,
            userDisplayMsg,
            { role: "model", content: "" },
        ];
        setMessages(displayHistory);
        // Audio markers go before image markers so the model "hears"
        // attachments in input order; the image splice already established
        // sentinel-pair-per-attachment as the prompt convention.
        const audioMarkers = "<|audio><audio|>".repeat(turnAudio.length);
        const imageMarkers = "<|image><image|>".repeat(turnImages.length);
        const userRenderContent = audioMarkers + imageMarkers + text;
        const renderPriorTurns = userTurns;   // assumed text-only for now
        const renderHistory: ChatMessage[] = sysContent
            ? [
                { role: "system", content: sysContent },
                ...renderPriorTurns,
                { role: "user", content: userRenderContent },
            ]
            : [
                ...renderPriorTurns,
                { role: "user", content: userRenderContent },
            ];

        let convId = activeConvId;
        let modelMsgId: string | null = null;
        try {
            if (!convId) {
                const row = await client.convCreate({
                    title: "New chat",
                    model: modelStatus === "ready" ? statusText.split(" ")[0] : null,
                });
                convId = row.id;
                setActiveConvId(convId);
            }
            const userInsert = await client.msgInsert({ conversationId: convId, role: "user", content: text });
            // Persist image thumbnails alongside the user turn so reloading
            // the conversation restores the bubble visuals. The JPEG bytes
            // go to OPFS via saveThumb (random-UUID key); only that key
            // lands in SQLite, keeping rows well under a page. Pixel
            // arrays aren't persisted — past-turn images aren't re-encoded.
            for (let i = 0; i < turnImages.length; i++) {
                const im = turnImages[i];
                try {
                    const opfsPath = await saveThumb(im.dataUrl);
                    await client.msgInsertImage({
                        conversationId: convId,
                        messageId:      userInsert.messageId,
                        seq:            i,
                        width:          im.w,
                        height:         im.h,
                        opfsPath,
                    });
                } catch (e) {
                    showToast({
                        level: "warn",
                        title: "Image not persisted",
                        message: (e as Error).message,
                    });
                }
            }
            const modelInsert = await client.msgInsert({ conversationId: convId, role: "model", content: "" });
            modelMsgId = modelInsert.messageId;
        } catch (e) {
            showToast({ level: "warn", title: "Persistence failure", message: (e as Error).message });
        }

        // acquireSession blocks if another tab is mid-generation. While
        // we wait, hint via the status line; clear it once we own the
        // session. Cooperative cancel via cancelRef takes over per-step
        // inside the session (queued-acquire abort is a follow-up).
        try {
            setStatusLine("waiting for another tab to finish…");
            await client.acquireSession();
            setStatusLine(undefined);

            await client.setSampling(sampling);
            await client.reset();
            const rendered = await client.renderChat(renderHistory, false);
            const ids = await client.encode(rendered);

            // ── Initialize inflight tracking BEFORE the prompt-feed
            // loop so a backgrounding during pre-encode (which is most
            // of the wall-clock cost on long conversations) still has
            // a resumable snapshot. `preEncodePosition` ticks up per
            // prompt-feed iteration; resume replays only the remaining
            // tokens.
            if (convId && modelMsgId) {
                inflightRef.current = {
                    convId,
                    modelMsgId,
                    modelDigest: lastLoadedDigest,
                    userText: text,
                    sysContent,
                    priorMessages: userTurns,
                    sampling,
                    maxTokens,
                    promptIds: Array.from(ids),
                    preEncodePosition: 0,
                    emittedSoFar: "",
                    emittedTokenCount: 0,
                    lastSampledNext: 0,
                    hadImages: turnImages.length > 0,
                    hadAudio:  turnAudio.length > 0,
                    mediaPersisted: false,
                    startedAt: Date.now(),
                };
            }

            // Persist image pixels + audio PCM to OPFS so a
            // kill-and-resume can re-encode them through the vision /
            // audio towers and continue from the partial response. Skip
            // entirely for text-only turns. Best-effort: a write
            // failure flips mediaPersisted back to false and resume
            // will surface the existing "interrupted" toast for this
            // turn only.
            if (turnImages.length > 0 || turnAudio.length > 0) {
                try {
                    for (let i = 0; i < turnImages.length; i++) {
                        const im = turnImages[i];
                        if (!im.pixels) continue;
                        await saveInflightImage(i, im.pixels, im.h, im.w, im.dataUrl);
                    }
                    for (let i = 0; i < turnAudio.length; i++) {
                        const a = turnAudio[i];
                        await saveInflightAudio(i, a.pcm, a.durationMs);
                    }
                    if (inflightRef.current) {
                        inflightRef.current = { ...inflightRef.current, mediaPersisted: true };
                    }
                } catch (e) {
                    // eslint-disable-next-line no-console
                    console.warn("[rullama] inflight media persist failed:", e);
                    // Leave mediaPersisted=false; resume falls back to
                    // the interrupted-turn toast if a kill happens.
                }
            }

            // Encode each attached image once per send. Result lives in
            // softMap keyed by the begin-sentinel token id — the feed
            // loop checks the map after every step() and splices in
            // `nSoft` stepWithEmbedding calls before continuing.
            //
            // Only one sentinel pair exists in the vocab, so multiple
            // image attachments stack their soft tokens into the same
            // softMap entry. The renderer emits N begin/end pairs, and
            // each begin id consumes the next image's worth of soft
            // tokens in order.
            type SoftEntry = { nSoft: number; dText: number; softTokens: Float32Array };
            const softQueue: SoftEntry[] = [];
            let beginId: number | null = null;
            if (turnImages.length > 0) {
                const sent = client.imageSentinelIds();
                if (!sent) {
                    throw new Error("model exposes no <|image> sentinel — vision unavailable");
                }
                beginId = sent[0];
                // Surface per-layer vision encode progress as a sticky
                // strip above the input row (see VisionProgress). The
                // encode can be ~2 min on slow GPUs; users need clear
                // feedback that the app is alive.
                let imgIdx = 0;
                const totalImgs = turnImages.filter((x) => x.pixels).length;
                const offProgress = client.subscribe("visionProgress", (p) => {
                    const layer = Number(p.layer);
                    const total = Number(p.total);
                    setVisionEncodeState({
                        imageIdx: imgIdx + 1,
                        nImages:  totalImgs,
                        layer,
                        nLayers:  total,
                    });
                });
                try {
                    for (const im of turnImages) {
                        if (!im.pixels) continue;
                        setVisionEncodeState({
                            imageIdx: imgIdx + 1,
                            nImages:  totalImgs,
                            layer:    0,
                            nLayers:  1,  // updated on the first progress event
                        });
                        const softTokens = await client.encodeImage(im.pixels, im.h, im.w);
                        const nSoft = await client.imageSoftTokenCount(im.h, im.w);
                        const dText = softTokens.length / nSoft;
                        softQueue.push({ nSoft, dText, softTokens });
                        imgIdx++;
                    }
                } finally {
                    offProgress();
                    setVisionEncodeState(null);
                }
            }

            // Same idea for voice clips. The audio tower outputs one
            // soft-token row per (downsampled) audio frame, dText wide;
            // splicing is identical to images but keyed off the
            // <|audio> sentinel id.
            const audioQueue: SoftEntry[] = [];
            let audioBeginId: number | null = null;
            if (turnAudio.length > 0) {
                const sent = client.audioSentinelIds();
                if (!sent) {
                    throw new Error("model exposes no <|audio> sentinel — audio unavailable");
                }
                audioBeginId = sent[0];
                // Reuse softQueue's dText if we computed it above (image
                // path); otherwise default to gemma4's d_text. The audio
                // tower projects into the same text embedding space, so
                // dText is a model constant.
                const fallbackDText = softQueue[0]?.dText ?? 1536;
                let audIdx = 0;
                const totalAud = turnAudio.length;
                for (const clip of turnAudio) {
                    if (cancelRef.current) throw new Error("cancelled");
                    setStatusLine(totalAud > 1
                        ? `audio ${audIdx + 1}/${totalAud} — encoding…`
                        : "encoding audio…");
                    const softTokens = await client.encodeAudio(clip.pcm);
                    const dText = fallbackDText;
                    const nSoft = softTokens.length / dText;
                    audioQueue.push({ nSoft, dText, softTokens });
                    audIdx++;
                }
                setStatusLine(undefined);
            }

            const t0 = performance.now();
            let next = 0;
            for (let i = 0; i < ids.length; i++) {
                if (cancelRef.current) throw new Error("cancelled");
                const id = ids[i];
                next = await client.step(id);
                if (beginId !== null && id === beginId && softQueue.length > 0) {
                    const ent = softQueue.shift()!;
                    for (let r = 0; r < ent.nSoft; r++) {
                        if (cancelRef.current) throw new Error("cancelled");
                        const row = ent.softTokens.subarray(r * ent.dText, (r + 1) * ent.dText);
                        next = await client.stepWithEmbedding(row);
                    }
                } else if (audioBeginId !== null && id === audioBeginId && audioQueue.length > 0) {
                    const ent = audioQueue.shift()!;
                    for (let r = 0; r < ent.nSoft; r++) {
                        if (cancelRef.current) throw new Error("cancelled");
                        const row = ent.softTokens.subarray(r * ent.dText, (r + 1) * ent.dText);
                        next = await client.stepWithEmbedding(row);
                    }
                }
                // Track pre-encode progress so a visibilitychange→hidden
                // mid-prompt-feed has a resumable snapshot. `next` at
                // this point is the model's predicted next token AFTER
                // ids[0..=i] (and any soft-token splices) — exactly
                // what the gen loop would feed first if pre-encode
                // completes here.
                if (inflightRef.current) {
                    inflightRef.current = {
                        ...inflightRef.current,
                        preEncodePosition: i + 1,
                        lastSampledNext: next,
                    };
                }
            }
            const peMs = performance.now() - t0;

            const tg0 = performance.now();
            let emitted = 0;
            let curStr   = (await client.tokenStr(next)) ?? "";
            let curIsEos = await client.isEos(next);
            let pendingDelta = "";
            let lastFlushAt  = performance.now();
            const flushPending = async () => {
                if (!convId || !modelMsgId || pendingDelta.length === 0) return;
                const delta = pendingDelta;
                pendingDelta = "";
                try { await client.msgAppend(convId, modelMsgId, delta); } catch { /* */ }
            };

            for (let i = 0; i < maxTokens; i++) {
                if (cancelRef.current) break;
                if (curIsEos) break;
                const piece = curStr.replaceAll("▁", " ");
                displayHistory[displayHistory.length - 1].content += piece;
                pendingDelta += piece;
                setMessages([...displayHistory]);
                emitted++;
                if ((emitted % 16 === 0) || (performance.now() - lastFlushAt > 750)) {
                    await flushPending();
                    lastFlushAt = performance.now();
                }
                const r = await stepWithTimeout(client, next);
                next     = r.next;
                curStr   = r.str ?? "";
                curIsEos = r.isEos;
                // Keep the inflight ref fresh so a visibilitychange
                // handler can persist current state without coordinating
                // with this loop. In-memory only — no localStorage write
                // per token (too costly).
                if (inflightRef.current) {
                    inflightRef.current = {
                        ...inflightRef.current,
                        emittedSoFar: displayHistory[displayHistory.length - 1].content,
                        emittedTokenCount: emitted,
                        lastSampledNext: next,
                    };
                }
            }
            await flushPending();
            if (convId) {
                try {
                    await client.convTouch(convId, suggestTitle(text));
                    await client.dbFlush();
                } catch { /* */ }
                void refreshConversations();
            }
            const dt = performance.now() - tg0;
            const tps = emitted > 0 ? (emitted * 1000 / dt) : 0;
            setStatusLine(`pe ${peMs.toFixed(0)} ms · gen ${emitted} tok in ${dt.toFixed(0)} ms · ${tps.toFixed(2)} tok/s`);
            beacon("chat", `gen ${emitted} tok in ${dt.toFixed(0)} ms (${tps.toFixed(2)} tok/s)`);
        } catch (e) {
            const msg = (e as Error).message;
            const isCancel = msg === "cancelled" || msg.includes("cancelled by caller");
            const isStepTimeout = msg.includes("step-timeout");
            // Live-tab recovery: a step that hangs past STEP_TIMEOUT_MS
            // almost always means the dedicated worker died (iOS jetsam
            // mid-generation, GPU device-lost, etc.). Instead of asking
            // the user to reload, hand the inflight metadata off to
            // resumeInflightGeneration on the next tick — it acquires
            // its own session and tries restoreKvState first, then the
            // slow-path replay. Same machinery the boot path uses.
            if (isStepTimeout && inflightRef.current) {
                liveRecovery = inflightRef.current;
                setStatusLine("worker hung — recovering session…");
            } else {
                setStatusLine(isCancel ? "cancelled" : `error: ${msg}`);
                if (!isCancel) {
                    showToast({ level: "error", title: "Generation failed", message: msg });
                }
            }
        } finally {
            if (liveRecovery) {
                // Hand off without clearing — resumeInflightGeneration
                // owns the metadata + OPFS state from here. Mirror to
                // localStorage so a page-reload-during-recovery still
                // resumes via the boot path.
                try { localStorage.setItem(INFLIGHT_KEY, JSON.stringify(liveRecovery)); } catch { /* */ }
                try { await client.releaseSession(); } catch { /* */ }
                setBusy(false);
                const meta = liveRecovery;
                setTimeout(() => { void resumeInflightGeneration(meta); }, 0);
            } else {
                // Generation finished (cleanly, by cancel, or by a
                // non-recoverable error). Discard inflight state — a
                // future backgrounding has nothing to resume. The OPFS
                // files are best-effort: a failed delete just leaves
                // stale snapshots that the next resume attempt will
                // reject by layout_hash or modelDigest mismatch.
                inflightRef.current = null;
                try { localStorage.removeItem(INFLIGHT_KEY); } catch { /* */ }
                void clearInflightState();
                void clearInflightMedia();
                try { await client.releaseSession(); } catch { /* */ }
                setBusy(false);
            }
        }
    }, [activeConvId, busy, lastLoadedDigest, maxTokens, messages, modelStatus, pendingAudio, pendingImages, prompt, refreshConversations, resumeInflightGeneration, sampling, statusText, systemPrompt, thinking, showToast]);

    // VAD-driven mic capture wired into the existing pendingAudio queue.
    // Each invocation produces one clip; user can stack multiple before
    // sending. Encoding to soft tokens happens later in onSend so the
    // tap-to-talk flow stays snappy (no inference under the hand).
    const onCaptureAudio = useCallback((pcm: Float32Array) => {
        const durationMs = (pcm.length / 16_000) * 1000;
        setPendingAudio((prev) => [...prev, { pcm, durationMs }]);
    }, []);
    const onRemoveAudio = useCallback((idx: number) => {
        setPendingAudio((prev) => prev.filter((_, i) => i !== idx));
    }, []);
    const onAudioError = useCallback((message: string) => {
        showToast({ level: "warn", title: "Mic capture failed", message });
    }, [showToast]);

    const onDeleteModel = useCallback(async (m: ModelEntry) => {
        if (busy) return;
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
            setMessages([]);
            setStatusLine(undefined);
            setHasVision(false);
            setHasAudio(false);
            setPendingImages([]);
            setPendingAudio([]);
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
    }, [busy, modelStatus, pendingLoadDigest, setLastLoadedDigest, setPendingLoadDigest, statusText, showToast]);

    /** Unload the active model, free its wasm-side buffers, and clear
     *  the "last loaded" memory so a page reload won't auto-resume.
     *  Past conversations stay in SQLite; OPFS-cached blobs stay on disk
     *  (use Delete to evict those). */
    const onEjectModel = useCallback(async () => {
        if (busy) return;
        if (modelStatus !== "ready") return;
        try { await getClient().free(); } catch { /* */ }
        const name = statusText.split(" ")[0] || "model";
        setModelStatus("idle");
        setStatusText("no model");
        setMessages([]);
        setStatusLine(undefined);
        setHasVision(false);
        setHasAudio(false);
        setPendingImages([]);
        setPendingAudio([]);
        setLastLoadedDigest("");
        // Eject is a deliberate user gesture — they don't want the
        // boot-time auto-resume to refire either.
        setPendingLoadDigest("");
        showToast({ level: "info", title: `Ejected ${name}` });
    }, [busy, modelStatus, setLastLoadedDigest, setPendingLoadDigest, statusText, showToast]);

    // Active conversation title for the header.
    const activeTitle = activeConvId
        ? conversations.find((c) => c.id === activeConvId)?.title
        : undefined;

    return (
        <div className="flex h-[100dvh] flex-col overflow-hidden bg-background text-foreground">
            {/* ─── top header (3rem / 48px tall — matches DualSidebarLayout offset) ─── */}
            {/* `min-h-12` not `h-12` so the safe-area-inset-top padding
                actually grows the header on iPhones with a notch /
                Dynamic Island — fixed h-12 was stuffing all content
                under the status bar in standalone PWA mode. */}
            <header className="flex min-h-12 shrink-0 items-center gap-2 border-b border-border bg-card/50 px-3 safe-top">
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => setHistoryOpen(!historyOpen)}
                    title="Toggle conversation history"
                    aria-pressed={historyOpen}
                >
                    <History />
                </Button>
                <span className="font-semibold tracking-tight">rullama</span>
                {activeTitle && (
                    <span className="hidden truncate text-xs text-muted-foreground sm:inline">
                        / {activeTitle}
                    </span>
                )}
                <div className="ml-auto flex items-center gap-2">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={() => setSettingsOpen(!settingsOpen)}
                        title="Toggle settings"
                        aria-pressed={settingsOpen}
                    >
                        <Settings />
                    </Button>
                </div>
            </header>

            {modelStatus === "loading" && (
                <ModelLoadProgress percent={loadingPercent} label={loadingLabel} />
            )}

            <DualSidebarLayout
                leftOpen={historyOpen}
                rightOpen={settingsOpen}
                onToggleLeft={setHistoryOpen}
                onToggleRight={setSettingsOpen}
                leftWidth={280}
                rightWidth={320}
                leftSidebar={
                    <ConversationList
                        conversations={conversations}
                        activeId={activeConvId}
                        onSelect={(id) => { void onSelectConversation(id); }}
                        onCreate={onCreateConversation}
                        onDelete={(id) => void onDeleteConversation(id)}
                    />
                }
                rightSidebar={
                    <SettingsDialog
                        modelStatus={modelStatus}
                        loadingPercent={loadingPercent}
                        loadingLabel={loadingLabel}
                        statusText={statusText}
                        onLoadModel={onLoad}
                        onDeleteModel={onDeleteModel}
                        onEjectModel={onEjectModel}
                        systemPrompt={systemPrompt}
                        onSystemPromptChange={setSystemPrompt}
                        sampling={sampling}
                        onSamplingChange={setSampling}
                        maxTokens={maxTokens}
                        onMaxTokensChange={setMaxTokens}
                        thinking={thinking}
                        onThinkingChange={setThinking}
                        voice={voice}
                        onVoiceChange={setVoice}
                        canRecord={modelStatus === "ready" && hasAudio}
                        onResetDefaults={onResetDefaults}
                    />
                }
            >
                <ChatPanel
                    messages={messages}
                    emptyState={
                        modelStatus !== "ready" ? (
                            <div className="mx-auto mt-6 w-full max-w-md">
                                <Card>
                                    <CardHeader>
                                        <CardTitle>Load a model</CardTitle>
                                        <CardDescription>
                                            Pick a model and click Load. It'll cache to this
                                            browser's local OPFS storage so subsequent visits
                                            are instant. Past conversations live in the
                                            History sidebar.
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent>
                                        <ModelLoader
                                            status={modelStatus}
                                            loadingPercent={loadingPercent}
                                            loadingLabel={loadingLabel}
                                            statusText={statusText}
                                            onLoad={onLoad}
                                            onDelete={onDeleteModel}
                                        />
                                    </CardContent>
                                </Card>
                            </div>
                        ) : (
                            <WelcomeScreen
                                modelName={statusText}
                                onSuggest={(p) => setPrompt(p)}
                            />
                        )
                    }
                    canType={modelStatus === "ready"}
                    canSend={
                        modelStatus === "ready"
                        && !busy
                        && (prompt.trim().length > 0 || pendingImages.length > 0 || pendingAudio.length > 0)
                    }
                    canStop={busy}
                    canAttach={modelStatus === "ready" && hasVision}
                    canRecord={modelStatus === "ready" && hasAudio && !busy}
                    pendingImages={pendingImages}
                    pendingAudio={pendingAudio.map((a) => ({ durationMs: a.durationMs }))}
                    voice={voice}
                    prompt={prompt}
                    onPromptChange={setPrompt}
                    onSend={onSend}
                    onStop={() => {
                        cancelRef.current = true;
                        // Also break in on any in-flight multimodal encode.
                        // The flag is cleared at the start of the next
                        // encode, so calling unconditionally here doesn't
                        // poison subsequent runs.
                        void getClient().cancelMultimodalEncode().catch(() => { /* */ });
                    }}
                    onAttachFiles={(files) => { void onAttachFiles(files); }}
                    onRemoveImage={onRemoveImage}
                    onCaptureAudio={onCaptureAudio}
                    onRemoveAudio={onRemoveAudio}
                    onAudioError={onAudioError}
                    visionProgress={visionEncodeState}
                    statusLine={statusLine}
                />
            </DualSidebarLayout>
            <RestartOverlay />
        </div>
    );
}
