import { useCallback, useEffect, useRef, useState } from "react";
import { ModelLoader, ModelLoadProgress, type ModelStatus } from "@/components/ModelLoader";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChatPanel } from "@/components/ChatPanel";
import type { PipelineProgressState } from "@/components/PipelineProgress";
import { RestartOverlay } from "@/components/RestartOverlay";
import { SettingsDialog, SETTINGS_BOUNDS } from "@/components/SettingsDialog";
import { VoicePanel } from "@/components/VoicePanel";
import { ConversationList } from "@/components/ConversationList";
import { DualSidebarLayout } from "@/components/layouts/DualSidebarLayout";
import { type ChatMessage, type ImageAttachment, type SamplingOptions, DEFAULT_SAMPLING, DEFAULT_SYSTEM_PROMPT } from "@/lib/types";
import { type ModelEntry, blobUrl, beacon, listModels, isDiffusion } from "@/lib/api";
import { ensureModel, existingSize, opfsSupported, requestPersistent, wipeModel, readInflightState, writeInflightState, clearInflightState } from "@/lib/opfs";
import { saveInflightImage, saveInflightAudio, readInflightImages, readInflightAudio, clearInflightMedia } from "@/lib/inflight_media";
import { getNetworkHint } from "@/lib/network";
import { getClient, teardownInferenceCore, type ConversationRow } from "@/lib/inference";
import { disposeSharedClone } from "@/lib/clone-client";
import { disposeSharedTts } from "@/lib/tts-client";
import { ChatSettings } from "@/components/ChatSettings";
import { AppHeader } from "@/components/AppHeader";
import { KnowledgeTab } from "@/components/KnowledgeTab";
import { searchKnowledge, buildRagPreamble } from "@/lib/embedding";
import { TrainingOverlay } from "@/components/TrainingOverlay";
import { UnsupportedScreen } from "@/components/UnsupportedScreen";
import { useDeviceTier } from "@/lib/capability";
import { useToast } from "@/lib/toast";
import { useConfirm } from "@/lib/confirm";
import { usePersistedState } from "@/lib/persisted";
import { useIOSKeyboard } from "@/lib/useIOSKeyboard";
import { useWakeLock } from "@/lib/wakeLock";
import { fmtBytes, fmtEta, clampInt, clampNum } from "@/lib/utils";
import { preprocessImage } from "@/lib/image_preprocess";
import { decodeAudioFile } from "@/lib/audio_decode";
import { saveThumb, loadThumbBlobUrl, deleteThumbs } from "@/lib/image_store";
import { DEFAULT_VOICE_OPTIONS, VOICE_BOUNDS, type VoiceOptions } from "@/lib/voice";
import { useTrainingCapability } from "@/components/FineTunePanel";
import { UpdateBanner } from "@/components/UpdateBanner";
import { ApplyingOverlay } from "@/components/ApplyingOverlay";
import {
    BUNDLED_VERSION,
    fetchServerVersion,
    isUpdateAvailable,
    isDismissed,
    setDismissedVersion,
} from "@/lib/version";
import {
    DOCKED_DEFAULT, THINK_TOKEN, INFLIGHT_KEY,
    type InflightGen, stepWithTimeout, suggestTitle, isMobileUA,
} from "@/lib/app-helpers";


export function App() {
    // Training capability — drives Fine-tune tab gating + the
    // "training not supported" screen. Detected once on mount via
    // `useTrainingCapability()` (WebGPU + min GPU buffer + min system
    // RAM + non-iOS UA).
    const trainingCap = useTrainingCapability();

    // Model load state
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

    // PWA update banner state. Driven by the boot-time version check
    // (lib/version.ts) plus cross-tab `updateAvailable` / `applyingUpdate`
    // notifies from the SharedWorker router. The banner itself is
    // gated on `!busy` (defer mid-generation); the apply overlay is
    // always rendered when the version mismatch is being applied.
    const [updateVersion, setUpdateVersion] = useState<string | null>(null);
    const [applyingUpdate, setApplyingUpdate] = useState(false);

    // Wait-reason coordination. The worker emits three independent
    // `…Waiting` / `…Retrying` notifies during slow OPFS / network
    // operations; surfacing them all through `setLoadingLabel` directly
    // is racy (one event can stomp another's message). Keep the
    // **most-recent** wait reason in its own state and have the render
    // pass `waitInfo?.message ?? loadingLabel` to the ModelLoader, so
    // a wait label naturally supersedes the normal progress label while
    // the wait is fresh and is cleared by the staleness timer below.
    const [waitInfo, setWaitInfo] = useState<
        { kind: "modelLoad" | "downloadLock" | "downloadStream"; message: string; ts: number } | null
    >(null);
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

    // View routing — Chat / Voice / Settings. Persisted so a reload doesn't
    // bounce the user out of the tab they were in. (Fine-tune is no longer a
    // tab — it's a full-screen overlay launched from Chat's sidebar; voice
    // learning likewise from Voice's sidebar. Keeping training in the same
    // engine context as its tab is what removed the cross-engine reload that
    // used to fail with "model load failed".)
    const [view, setView] = usePersistedState<"chat" | "voice" | "knowledge" | "settings">("rullama:view", "chat");
    // Migrate any persisted legacy "finetune" view to "chat".
    useEffect(() => {
        if ((view as string) === "finetune") setView("chat");
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    // Full-screen training overlay: which one is open (null = none).
    const [training, setTraining] = useState<null | "finetune" | "voicelearn">(null);
    // Device-capability tier (gates the boot, engine co-residency, model markers).
    const { tier, probe } = useDeviceTier();
    const premium = tier === "premium";
    // **D3 — chat-during-training gate.** When a Fine-tune run is
    // active in this (or any other) tab, the Model is owned by the
    // training session and chat-side step RPCs would fail with
    // "model is owned by an active training session". Reflect that
    // in the UI so the user sees a clear "training in progress"
    // affordance instead of an opaque error toast on Send. Sourced
    // from the worker's `trainingStarted` / `trainingFinished`
    // notifies, which fan out across tabs via the SharedWorker
    // router.
    const [trainingInProgress, setTrainingInProgress] = useState<boolean>(false);

    // Adapter currently active in chat. Persisted across reloads so a
    // page refresh doesn't silently drop a trained adapter — the bytes
    // are in OPFS regardless, but the "this one is loaded into Model"
    // state used to reset to null on boot, forcing the user to
    // re-apply manually via the Fine-tune tab. The restore-on-boot
    // effect below re-applies the saved adapter once the model is
    // ready; failure clears the persisted name so we don't keep
    // trying.
    const [activeAdapter, setActiveAdapter] = usePersistedState<string | null>("activeAdapter", null);

    // Chat state
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [prompt, setPrompt]     = useState("");
    const [busy, setBusy]         = useState(false);
    const [statusLine, setStatusLine] = useState<string | undefined>();
    const [visionEncodeState, setVisionEncodeState] = useState<PipelineProgressState | null>(null);

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
    // Per-conversation RAG toggle (Knowledge-base grounding). Loaded from
    // the DB when the active conversation changes.
    const [ragEnabled, setRagEnabled]       = useState<boolean>(false);
    useEffect(() => {
        let alive = true;
        if (!activeConvId) { setRagEnabled(false); return; }
        getClient().embeddings.getRag(activeConvId)
            .then((r) => { if (alive) setRagEnabled(!!r.enabled); })
            .catch(() => { if (alive) setRagEnabled(false); });
        return () => { alive = false; };
    }, [activeConvId]);
    const toggleRag = useCallback((on: boolean) => {
        setRagEnabled(on);
        if (activeConvId) void getClient().embeddings.setRag(activeConvId, on).catch(() => {});
    }, [activeConvId]);

    // Sidebar visibility — persisted across reloads. Only the left
    // (history) sidebar exists now; Settings has been promoted to its
    // own tab so it doesn't compete with chat content for screen
    // real-estate on small displays.
    // Sidebar open/close — persisted per sidebar, defaulting open when docked.
    const [historyOpen, setHistoryOpen] = usePersistedState<boolean>("ui.historyOpen", DOCKED_DEFAULT);
    // Chat-tab right sidebar (model block + system prompt + sampling + thinking).
    const [chatSettingsOpen, setChatSettingsOpen] = usePersistedState<boolean>("ui.chatSettingsOpen", DOCKED_DEFAULT);
    // Voice-tab right sidebar (voice picker + clone-model block, portaled from VoicePanel).
    const [voiceTabSettingsOpen, setVoiceTabSettingsOpen] = usePersistedState<boolean>("ui.voiceTabSettingsOpen", DOCKED_DEFAULT);
    const [voiceTabSettingsEl, setVoiceTabSettingsEl] = useState<HTMLDivElement | null>(null);
    // Voice-tab left sidebar (the generated-clips list, portaled from VoicePanel).
    const [voiceClipsOpen, setVoiceClipsOpen] = usePersistedState<boolean>("ui.voiceClipsOpen", DOCKED_DEFAULT);
    const [voiceClipsEl, setVoiceClipsEl] = useState<HTMLDivElement | null>(null);
    // Fine-tune overlay right sidebar (hyperparameter settings, portaled from FineTunePanel).
    const [fineTuneSettingsOpen, setFineTuneSettingsOpen] = usePersistedState<boolean>("ui.fineTuneSettingsOpen", DOCKED_DEFAULT);
    const [fineTuneSettingsEl, setFineTuneSettingsEl] = useState<HTMLDivElement | null>(null);

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
    const confirm = useConfirm();

    // iOS keyboard handling — snaps the visual viewport back to the top
    // when the keyboard dismisses, so the page doesn't end up offset
    // a few px above the layout viewport (a classic iOS-Safari quirk).
    useIOSKeyboard(true);

    // Hold the screen awake during the two long-running operations a
    // user actually waits on — model download/load and token generation.
    // No-op on platforms without `navigator.wakeLock` (older iOS, private
    // mode). See `lib/wakeLock.ts` for the iOS hide-release dance.
    useWakeLock(modelStatus === "loading" || busy);

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
        if (modelStatus === "loading") {
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

    // pagehide is the reliable "tab is going away" signal across
    // browsers — fires before iOS suspends, before unload would on
    // desktop, and iOS Safari does NOT fire `beforeunload`. We write
    // a localStorage marker with the current session id; the next
    // mount's crash-detect treats a matching marker as proof of a
    // clean exit even if the worker's manifest shutdown didn't
    // complete.
    //
    // Critical: `client.logs.currentId()` is an async RPC over
    // postMessage, and pagehide CANNOT await. By the time the worker
    // responds, the tab is gone and `localStorage.setItem` never
    // runs. Cache the id synchronously on mount, then read the cached
    // value inside the listener so the marker is written without an
    // RPC round-trip.
    const currentSessionIdRef = useRef<string>("");
    useEffect(() => {
        let cancelled = false;
        const client = getClient();
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
        const onVis = () => { if (document.visibilityState === "visible") void refresh(); };
        document.addEventListener("visibilitychange", onVis);
        return () => {
            cancelled = true;
            document.removeEventListener("visibilitychange", onVis);
        };
    }, []);
    useEffect(() => {
        const onHide = () => {
            const id = currentSessionIdRef.current;
            if (!id) return;
            try { localStorage.setItem("rullama:session:clean", id); } catch { /* */ }
        };
        window.addEventListener("pagehide", onHide);
        return () => window.removeEventListener("pagehide", onHide);
    }, []);

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
        // restore-on-ready effect below to re-apply it via
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
    }, []);

    // **D1 — restore active adapter on reload.** The adapter bytes
    // survive in OPFS but the "this one is loaded into Model" state
    // used to reset to null on every boot, leaving the user to manually
    // re-apply via the Fine-tune tab. Once the model is ready and an
    // activeAdapter is persisted but not yet applied to the worker,
    // re-apply it. A single retry on the next-`modelStatus` change is
    // enough; failure clears the persisted name so we don't keep
    // trying for a missing/incompatible adapter.
    const adapterRestoredRef = useRef(false);
    useEffect(() => {
        if (adapterRestoredRef.current) return;
        if (modelStatus !== "ready") return;
        if (!activeAdapter) return;
        adapterRestoredRef.current = true;
        (async () => {
            const c = getClient();
            try {
                const list = await c.trainingListAdapters();
                if (list.active === activeAdapter) return; // already applied
                if (!list.entries.some((e) => e.name === activeAdapter)) {
                    console.warn(`[rullama] persisted activeAdapter '${activeAdapter}' is no longer in the library — clearing`);
                    setActiveAdapter(null);
                    return;
                }
                // trainingApplyAdapter is session-gated. Wrap in
                // withSession so the apply queues behind any other
                // tab's active work and releases cleanly when done.
                await c.withSession(() => c.trainingApplyAdapter(activeAdapter));
            } catch (e) {
                console.warn(`[rullama] failed to restore activeAdapter '${activeAdapter}' on boot:`, e);
                setActiveAdapter(null);
            }
        })();
    }, [modelStatus, activeAdapter, setActiveAdapter]);

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

    // Apply / dismiss handlers for the UpdateBanner.
    const onApplyUpdate = useCallback(() => {
        const v = updateVersion ?? "";
        // Optimistically show the overlay even before the router echoes
        // back `applyingUpdate` — keeps the UI feeling responsive.
        setApplyingUpdate(true);
        (async () => {
            try {
                await getClient().applyUpdate(v);
            } catch (e) {
                console.warn("[rullama] applyUpdate RPC failed; reloading this tab only:", e);
                // The router never fanned out `applyingUpdate` to other
                // tabs. Reload solo as a fallback — at least this tab
                // gets the new bundle, the other tabs will catch up on
                // their next reload.
                setTimeout(() => window.location.reload(), 200);
            }
        })();
    }, [updateVersion]);

    const onDismissUpdate = useCallback(() => {
        if (updateVersion) setDismissedVersion(updateVersion);
        setUpdateVersion(null);
    }, [updateVersion]);

    // Latest-onLoad ref so the auto-load effect doesn't need it as a dep
    // (which would re-trigger whenever onLoad's closure changes).
    const onLoadRef = useRef<((m: ModelEntry) => Promise<void>) | null>(null);

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
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pendingLoadDigest, lastLoadedDigest]);

    // **One engine resident at a time (except Premium).** Inference (Gemma) and the TTS/clone
    // engines can't share a phone GPU, and on iOS WebKit only a worker's DEATH reclaims its GPU
    // memory (model.free() and GPUBuffer.destroy() don't, observably — bug 302711 family). So on a
    // Chat↔Voice tab change, tear down the engine we're NOT using and (re)load the one we are —
    // replicating the user's working "fresh reload" automatically. Training overlays don't change
    // `view`, so launching Fine-tune (over Chat) / Voice-learning (over Voice) performs NO swap;
    // training shares its tab's engine. **Premium tier short-circuits the teardown** so inference +
    // TTS stay co-resident (the user has the VRAM; Chat↔Voice is then instant). `settings` needs
    // neither engine, so it no-ops — switching to Settings never loads or unloads a model.
    const engineSwapMounted = useRef(false);
    useEffect(() => {
        const firstRun = !engineSwapMounted.current;
        engineSwapMounted.current = true;
        if (premium) return; // keep both engines warm on high-VRAM machines
        const needsInference = view === "chat";
        const needsVoice = view === "voice";
        if (needsVoice && !needsInference) {
            teardownInferenceCore();
            setModelStatus("idle"); // returning to chat re-triggers the load below
        } else if (needsInference && !needsVoice) {
            disposeSharedClone();
            disposeSharedTts();
            if (!firstRun && modelStatus !== "ready" && modelStatus !== "loading") {
                void reloadInferenceModel();
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [view, premium]);

    // Esc closes the training overlay.
    useEffect(() => {
        if (!training) return;
        const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setTraining(null); };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [training]);

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
        // Two attach paths: images route through the vision tower,
        // audio files route through the audio tower (analyse-this-clip
        // use-case; mic-button transcription is now a separate flow,
        // see `onCaptureAudio`). UI gating should prevent attaching
        // capability-mismatched files but a stale handler can still
        // fire mid-unload — silently drop in that case.
        const nextImages: ImageAttachment[] = [];
        const nextAudio: { pcm: Float32Array; durationMs: number }[] = [];
        for (let i = 0; i < files.length; i++) {
            const f = files[i];
            if (f.type.startsWith("image/")) {
                if (!hasVision) continue;
                try {
                    const p = await preprocessImage(f);
                    nextImages.push(p);
                } catch (e) {
                    showToast({
                        level: "error", title: `Couldn't load ${f.name}`,
                        message: (e as Error).message,
                    });
                }
            } else if (f.type.startsWith("audio/")) {
                if (!hasAudio) continue;
                try {
                    const decoded = await decodeAudioFile(f);
                    if (decoded) {
                        nextAudio.push({ pcm: decoded.pcm, durationMs: decoded.durationMs });
                    } else {
                        showToast({
                            level: "warn", title: `Couldn't decode ${f.name}`,
                            message: "This browser doesn't support that audio format. Try MP3 or WAV.",
                        });
                    }
                } catch (e) {
                    showToast({
                        level: "error", title: `Couldn't load ${f.name}`,
                        message: (e as Error).message,
                    });
                }
            }
        }
        if (nextImages.length) setPendingImages((prev) => [...prev, ...nextImages]);
        if (nextAudio.length) setPendingAudio((prev) => [...prev, ...nextAudio]);
    }, [hasVision, hasAudio, showToast]);

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
                showToast({ level: "info", title: "Download stopped", message: "Resume any time by loading again." });
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
    }, [dismissToast, setLastLoadedDigest, setPendingLoadDigest, showToast]);

    /** Stop an in-progress model download (the picker's Stop button). The
     *  active ensureModel rejects with "cancelled", handled above; the partial
     *  stays in OPFS for a later resume. */
    const onCancelDownload = useCallback(() => {
        try { void getClient().cancelDownload(); } catch { /* */ }
    }, []);

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
        // Block-diffusion models (DiffusionGemma) generate via a denoise loop,
        // not the AR token stream — flagged at load time.
        const diffusionTurn = loadedIsDiffusion;
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

        let baseSystem = systemPrompt.trim();

        // **RAG injection.** If RAG is enabled for this conversation and an
        // embedder is loaded, retrieve the top-K relevant chunks (this
        // conversation's docs + global docs) and prepend them to the system
        // prompt with source attribution. Best-effort: a failure here never
        // blocks the send.
        if (ragEnabled && activeConvId) {
            try {
                const hits = await searchKnowledge(text, { k: 5, conversationId: activeConvId });
                const preamble = buildRagPreamble(hits);
                if (preamble) baseSystem = preamble + baseSystem;
            } catch { /* embedder not loaded / search failed — proceed without RAG */ }
        }

        const sysContent = thinking
            ? (baseSystem ? `${THINK_TOKEN}${baseSystem}` : THINK_TOKEN)
            : baseSystem;

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
            // ── DiffusionGemma: block-diffusion denoise loop ───────────────
            // Not autoregressive — no session/Model, no prefill, no token
            // stream. Each step is a full canvas forward; we replace the model
            // bubble's content with the evolving canvas in place, then persist
            // the final text. Returns early; the outer finally clears busy.
            if (diffusionTurn) {
                const replaceModelBubble = (canvas: string) =>
                    setMessages((prev) => {
                        const next = [...prev];
                        for (let i = next.length - 1; i >= 0; i--) {
                            if (next[i].role === "model") {
                                next[i] = { ...next[i], content: canvas };
                                break;
                            }
                        }
                        return next;
                    });
                const unsub = client.diffusion.onStep((p) => {
                    replaceModelBubble(p.text);
                    setStatusLine(
                        `denoise ${p.stepIndex + 1}/${p.totalSteps} · ${p.accepted} accepted · H ${p.meanEntropy.toFixed(3)}`,
                    );
                });
                try {
                    const { text: canvas } = await client.diffusion.generate({ prompt: text });
                    replaceModelBubble(canvas);
                    if (convId && modelMsgId) {
                        try { await client.msgAppend(convId, modelMsgId, canvas); } catch { /* */ }
                    }
                } finally {
                    unsub();
                    setStatusLine(undefined);
                }
                return;
            }

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
            const totalImgs = turnImages.filter((x) => x.pixels).length;
            if (turnImages.length > 0) {
                const sent = client.imageSentinelIds();
                if (!sent) {
                    throw new Error("model exposes no <|image> sentinel — vision unavailable");
                }
                beginId = sent[0];
                // Surface per-layer vision encode progress as a sticky
                // strip above the input row (see PipelineProgress). The
                // strip stays alive across three phases — encoding (vision
                // tower) → embedding (soft-token splice through the text
                // model) → prefill (prompt-token feed). Without all three
                // the user sees the bar disappear at 16/16 and then sits
                // 2-3 min in silence while the JS prefill loop chews on
                // 256 soft tokens per image at ~870 ms each.
                let imgIdx = 0;
                const offProgress = client.subscribe("pipelineProgress", (p) => {
                    const layer = Number(p.layer);
                    const total = Number(p.total);
                    setVisionEncodeState({
                        imageIdx: imgIdx + 1,
                        nImages:  totalImgs,
                        phase:    "encoding",
                        done:     layer,
                        total,
                    });
                });
                try {
                    for (const im of turnImages) {
                        if (!im.pixels) continue;
                        setVisionEncodeState({
                            imageIdx: imgIdx + 1,
                            nImages:  totalImgs,
                            phase:    "encoding",
                            done:     0,
                            total:    1,  // updated on the first progress event
                        });
                        const softTokens = await client.encodeImage(im.pixels, im.h, im.w);
                        const nSoft = await client.imageSoftTokenCount(im.h, im.w);
                        const dText = softTokens.length / nSoft;
                        softQueue.push({ nSoft, dText, softTokens });
                        imgIdx++;
                    }
                } finally {
                    offProgress();
                    // Don't null the strip here — the embedding + prefill
                    // phases below take it over for the next 2-3 min of
                    // otherwise-silent work.
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

            // Multimodal weight release between encode and prefill.
            //
            // Vision/audio towers carry their own GPU weight tensors —
            // on gemma4:e2b the vision tower is ~3 GB. After
            // `encodeImage` returns the soft-token rows we hold them
            // as plain Float32Array in JS and don't need the tower's
            // GPU buffers again until the next image attachment. On
            // iPhone Safari (WebContent ~5 GB ceiling) keeping the
            // text + vision towers + KV cache + per-step scratch
            // co-resident is what pushes the first prefill step over
            // jetsam. Drop them now; the next image attachment
            // re-uploads via the WeightCache fetch path.
            if (totalImgs > 0) {
                try { await client.releaseVisionWeights(); }
                catch (e) { console.warn("[multimodal] releaseVisionWeights failed", e); }
            }
            if (turnAudio.length > 0) {
                try { await client.releaseAudioWeights(); }
                catch (e) { console.warn("[multimodal] releaseAudioWeights failed", e); }
            }
            if (totalImgs > 0 || turnAudio.length > 0) {
                // Yield one rAF tick before prefill so iOS Safari's wgpu
                // device finishes reclaiming the just-dropped Vision/Audio
                // scratch buffers (~250 MB / ~110 MB) BEFORE the text-
                // weight upload barrage starts. Without this, the
                // released buffers can linger in the device's "pending
                // drop" queue until the next vsync, and the prefill
                // peak resident set briefly includes both — enough to
                // tip iPhone over jetsam. Cost on desktop is one
                // frame (~16 ms), unmeasurable in human time.
                await new Promise<void>((r) => requestAnimationFrame(() => r()));
            }

            const t0 = performance.now();
            let next = 0;
            // Breadcrumb so a jetsam kill mid-prefill leaves something
            // in /tmp/rullama-page.log on the iPhone harness. If this
            // beacon is the last line before silence, the crash is
            // during the first text-prefill step (W7a-equivalent OOM);
            // if it never lands, the crash is earlier (encode or
            // release).
            beacon("pe", `prefill start (n_tokens=${ids.length}, imgs=${totalImgs}, audio=${turnAudio.length})`);
            // Drive the progress strip through the rest of pre-encode.
            // The outer loop is the prompt-token feed (~50 tokens) and
            // each `<|image>` / `<|audio>` sentinel triggers an inner
            // splice loop of ~256 stepWithEmbedding calls (~870 ms each).
            // Without these per-iteration updates the strip would freeze
            // for 2-3 min per image attachment.
            setVisionEncodeState({
                imageIdx: totalImgs,
                nImages:  totalImgs,
                phase:    "prefill",
                done:     0,
                total:    ids.length,
            });
            for (let i = 0; i < ids.length; i++) {
                if (cancelRef.current) throw new Error("cancelled");
                const id = ids[i];
                next = await client.step(id);
                if (beginId !== null && id === beginId && softQueue.length > 0) {
                    const ent = softQueue.shift()!;
                    // softQueue.shift() already removed this entry, so the
                    // currently-embedding image is the (totalImgs -
                    // softQueue.length)-th — 1-based.
                    const embedImgIdx = totalImgs - softQueue.length;
                    for (let r = 0; r < ent.nSoft; r++) {
                        if (cancelRef.current) throw new Error("cancelled");
                        const row = ent.softTokens.subarray(r * ent.dText, (r + 1) * ent.dText);
                        next = await client.stepWithEmbedding(row);
                        setVisionEncodeState({
                            imageIdx: embedImgIdx,
                            nImages:  totalImgs,
                            phase:    "embedding",
                            done:     r + 1,
                            total:    ent.nSoft,
                        });
                    }
                } else if (audioBeginId !== null && id === audioBeginId && audioQueue.length > 0) {
                    const ent = audioQueue.shift()!;
                    for (let r = 0; r < ent.nSoft; r++) {
                        if (cancelRef.current) throw new Error("cancelled");
                        const row = ent.softTokens.subarray(r * ent.dText, (r + 1) * ent.dText);
                        next = await client.stepWithEmbedding(row);
                    }
                }
                setVisionEncodeState({
                    imageIdx: totalImgs,
                    nImages:  totalImgs,
                    phase:    "prefill",
                    done:     i + 1,
                    total:    ids.length,
                });
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
            setVisionEncodeState(null);
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
            // Strip is normally cleared just before the gen loop, but
            // an error/cancel during encode, splice, or prefill skips
            // that point — make sure it's gone here so the chat input
            // isn't stuck behind a stale progress bar.
            setVisionEncodeState(null);
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
    }, [activeConvId, busy, lastLoadedDigest, loadedIsDiffusion, maxTokens, messages, modelStatus, pendingAudio, pendingImages, prompt, ragEnabled, refreshConversations, resumeInflightGeneration, sampling, statusText, systemPrompt, thinking, showToast]);

    // Top-level pipelineProgress subscription. The chat-send flow has
    // its own scoped subscription (around image encode + prefill) but
    // the mic-transcribe pipeline runs OUTSIDE that scope and needs
    // its own. Audio-kind beacons drive the same `visionEncodeState`
    // so the same status strip above the chat input shows
    // "Transcribing — encoding/splicing/reading/generating" phases.
    // Without this the user has no visibility into where a long-running
    // transcribe is — on mobile, that's the difference between
    // "looks frozen, reload" and "I can see it's at splice step 14/31."
    useEffect(() => {
        const off = getClient().subscribe("pipelineProgress", (p) => {
            if (p.modality !== "audio") return; // image events handled scoped, below
            const phase = String(p.phase ?? "encoding");
            const layer = Number(p.layer ?? 0);
            const total = Number(p.total ?? 1);
            setVisionEncodeState({
                imageIdx: 1,
                nImages:  1,
                phase:    phase as PipelineProgressState["phase"],
                done:     layer,
                total,
                kind:     "audio",
            });
        });
        return off;
    }, []);

    // VAD-driven mic capture → in-engine transcription → fill the chat
    // input box. Mic press is "speak my next message" — text fills the
    // prompt and the user can edit/send like any typed message.
    //
    // The audio-clip-attachment path is reserved for file uploads (mp3/
    // wav via the paperclip), where the user explicitly wants the model
    // to *analyse* the audio rather than transcribe it.
    //
    // Greedy decode is enforced worker-side regardless of chat sampling.
    // Streams deltas into `prompt` so the user sees the transcript fill
    // in real time while the audio tower + LM are still running.
    const onCaptureAudio = useCallback(async (pcm: Float32Array) => {
        if (pcm.length === 0) return;
        const client = getClient();
        try {
            // Show the strip immediately so the user has visible feedback
            // from the moment the mic stops, instead of waiting for the
            // worker's first pipelineProgress notify to round-trip back
            // (the message-passing delay alone can swallow the only
            // visible window on fast desktop GPUs).
            setVisionEncodeState({
                imageIdx: 1, nImages: 1,
                phase: "encoding", done: 0, total: 1,
                kind: "audio",
            });
            // Append-on-stream: each delta extends the current prompt
            // value so the user sees it fill in. If the prompt already
            // had content, add a leading space before the first delta.
            let leadingPad: string | null = null;
            setPrompt((cur) => {
                leadingPad = cur.length > 0 && !cur.endsWith(" ") ? " " : "";
                return cur;
            });
            const transcript = await client.transcribeAudio(pcm, (delta) => {
                setPrompt((cur) => {
                    const pad = leadingPad ?? "";
                    leadingPad = ""; // only insert pad once
                    return cur + pad + delta;
                });
            });
            if (!transcript.trim()) {
                showToast({
                    level: "warn",
                    title: "Didn't catch that",
                    message: "Try speaking again, or use the paperclip to attach an audio file for analysis.",
                });
            }
        } catch (e) {
            showToast({
                level: "error",
                title: "Transcription failed",
                message: (e as Error).message,
            });
        } finally {
            // Clear the audio-kind status strip; image-kind strips are
            // managed by the chat-send flow's own scoped subscription.
            setVisionEncodeState((cur) => cur?.kind === "audio" ? null : cur);
        }
    }, [showToast]);
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
        setMessages([]);
        setStatusLine(undefined);
        setHasVision(false);
        setHasAudio(false);
        setLoadedIsDiffusion(false);
        setPendingImages([]);
        setPendingAudio([]);
        setLastLoadedDigest("");
        // Eject is a deliberate user gesture — they don't want the
        // boot-time auto-resume to refire either.
        setPendingLoadDigest("");
        showToast({ level: "info", title: `Ejected ${name}` });
    }, [busy, modelStatus, loadedIsDiffusion, setLastLoadedDigest, setPendingLoadDigest, statusText, showToast]);

    // Active conversation title for the header.
    const activeTitle = activeConvId
        ? conversations.find((c) => c.id === activeConvId)?.title
        : undefined;

    // Hard block below the minimum spec — render a clean screen and DON'T
    // boot the engine (the auto-load effect is gated on tier too). This is
    // what stops incapable devices (e.g. iPhone 7, no WebGPU) from boot-
    // looping the heavy app. `tier === null` means the async probe is still
    // in flight; render the normal shell (the engine stays gated until it
    // resolves).
    if (tier === "unsupported") return <UnsupportedScreen probe={probe} />;

    return (
        <div className="flex h-[100dvh] flex-col overflow-hidden bg-background text-foreground">
            {/* PWA update banner. Detected via boot-time /version.json
                fetch (lib/version.ts) and broadcast across tabs via
                the SharedWorker. Deferred while mid-generation so an
                in-flight chat reply is never interrupted. */}
            {updateVersion && !busy && !applyingUpdate && (
                <UpdateBanner
                    version={updateVersion}
                    onApply={onApplyUpdate}
                    onDismiss={onDismissUpdate}
                />
            )}
            {/* ─── top header (3rem / 48px tall — matches DualSidebarLayout offset) ─── */}
            <AppHeader
                view={view}
                onSelectView={setView}
                historyOpen={historyOpen}
                onToggleHistory={() => setHistoryOpen(!historyOpen)}
                activeTitle={activeTitle}
                activeAdapter={activeAdapter}
                ragEnabled={ragEnabled}
                onToggleRag={toggleRag}
            />

            {modelStatus === "loading" && (
                <ModelLoadProgress percent={loadingPercent} label={waitInfo?.message ?? loadingLabel} />
            )}

            <DualSidebarLayout
                leftOpen={
                    view === "chat" ? historyOpen
                    : view === "voice" ? voiceClipsOpen
                    : false
                }
                onToggleLeft={view === "chat" ? setHistoryOpen : setVoiceClipsOpen}
                leftWidth={280}
                // Left sidebar: chat → conversation list; voice → generated-clips
                // list (portaled from VoicePanel). Right sidebar is each tab's
                // settings. Each gets its own chevron toggle.
                leftSidebar={
                    view === "chat" ? (
                        <ConversationList
                            conversations={conversations}
                            activeId={activeConvId}
                            onSelect={(id) => { void onSelectConversation(id); }}
                            onCreate={onCreateConversation}
                            onDelete={(id) => void onDeleteConversation(id)}
                        />
                    ) : view === "voice" ? (
                        <div ref={setVoiceClipsEl} className="h-full" />
                    ) : undefined
                }
                rightOpen={
                    view === "chat" ? chatSettingsOpen
                    : view === "voice" ? voiceTabSettingsOpen
                    : false
                }
                onToggleRight={
                    view === "chat" ? setChatSettingsOpen
                    : setVoiceTabSettingsOpen
                }
                rightWidth={340}
                rightSidebar={
                    view === "chat" ? (
                        // Per-tab chat settings: the Gemma model block + Fine-tune launcher,
                        // plus system prompt / sampling / thinking. Logs / app-data / High-VRAM
                        // toggle stay in the global Settings view.
                        <ChatSettings
                            modelStatus={modelStatus}
                            loadingPercent={loadingPercent}
                            loadingLabel={waitInfo?.message ?? loadingLabel}
                            statusText={statusText}
                            onLoadModel={onLoad}
                            onDeleteModel={onDeleteModel}
                            onEjectModel={onEjectModel}
                            selectedModelName={selectedModelName}
                            onSelectModel={setSelectedModelName}
                            preferredDigest={lastLoadedDigest}
                            onCancelDownload={onCancelDownload}
                            onOpenFineTune={() => setTraining("finetune")}
                            canFineTune={modelStatus === "ready" && trainingCap.status === "ok"}
                            fineTuneReason={
                                modelStatus !== "ready" ? "Load a model first"
                                : trainingCap.status === "checking" ? "Checking device capability…"
                                : trainingCap.status === "blocked" ? trainingCap.title
                                : "Fine-tuning unavailable on this device"
                            }
                            systemPrompt={systemPrompt}
                            onSystemPromptChange={setSystemPrompt}
                            sampling={sampling}
                            onSamplingChange={setSampling}
                            maxTokens={maxTokens}
                            onMaxTokensChange={setMaxTokens}
                            thinking={thinking}
                            onThinkingChange={setThinking}
                            onResetDefaults={onResetDefaults}
                            voice={voice}
                            onVoiceChange={setVoice}
                            canRecord={modelStatus === "ready" && hasAudio}
                        />
                    ) : view === "voice" ? (
                        // VoicePanel portals its voice picker / clone-model block into this host.
                        <div ref={setVoiceTabSettingsEl} className="h-full" />
                    ) : undefined
                }
            >
                {view === "voice" ? (
                    <VoicePanel settingsHostEl={voiceTabSettingsEl} clipsHostEl={voiceClipsEl} onOpenVoiceLearn={() => setTraining("voicelearn")} />
                ) : view === "knowledge" ? (
                    <KnowledgeTab activeConvId={activeConvId} />
                ) : view === "settings" ? (
                    // Centered max-width wrapper so the form controls
                    // don't stretch across the full main-content width
                    // on desktop monitors. `h-full min-h-0` keeps the
                    // height-fill contract SettingsDialog expects from
                    // its parent (it uses h-full internally).
                    <div className="mx-auto flex h-full w-full max-w-3xl min-h-0 flex-col">
                        <SettingsDialog />
                    </div>
                ) : (
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
                                            loadingLabel={waitInfo?.message ?? loadingLabel}
                                            statusText={statusText}
                                            onLoad={onLoad}
                                            onDelete={onDeleteModel}
                                            onCancel={onCancelDownload}
                                            selected={selectedModelName}
                                            onSelect={setSelectedModelName}
                                            preferredDigest={lastLoadedDigest}
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
                    canType={modelStatus === "ready" && !trainingInProgress}
                    canSend={
                        modelStatus === "ready"
                        && !busy
                        && !trainingInProgress
                        && (prompt.trim().length > 0 || pendingImages.length > 0 || pendingAudio.length > 0)
                    }
                    canStop={busy}
                    canAttach={modelStatus === "ready" && (hasVision || hasAudio) && !trainingInProgress}
                    canRecord={modelStatus === "ready" && hasAudio && !busy && !trainingInProgress}
                    // Speak-a-reply runs the Kokoro TTS engine alongside inference — needs at least
                    // the recommended GPU tier. Hidden on the minimum (mobile) tier.
                    canSpeak={tier === "desktop" || tier === "premium"}
                    statusLine={trainingInProgress
                        ? "Training session is active — open Fine-tune and Save / Apply / Discard the adapter to return to chat."
                        : statusLine}
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
                    pipelineProgress={visionEncodeState}
                />
                )}
            </DualSidebarLayout>

            {/* Full-screen training overlay. Launched from a tab's sidebar button;
                Chat/Voice stay mounted underneath so the engine stays GPU-resident
                (no swap — training shares its tab's engine). */}
            {training && (
                <TrainingOverlay
                    training={training}
                    onClose={() => setTraining(null)}
                    trainingCap={trainingCap}
                    fineTuneSettingsOpen={fineTuneSettingsOpen}
                    onToggleFineTuneSettings={setFineTuneSettingsOpen}
                    fineTuneSettingsEl={fineTuneSettingsEl}
                    setFineTuneSettingsEl={setFineTuneSettingsEl}
                    modelStatus={modelStatus}
                    activeAdapter={activeAdapter}
                    onAdapterChanged={setActiveAdapter}
                />
            )}

            <RestartOverlay />
            {applyingUpdate && updateVersion && <ApplyingOverlay version={updateVersion} />}
        </div>
    );
}
