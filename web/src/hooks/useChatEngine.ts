import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type ChatMessage, type ImageAttachment, type SamplingOptions } from "@/lib/types";
import type { PipelineProgressState } from "@/components/PipelineProgress";
import { type ModelStatus } from "@/components/ModelLoader";
import { beacon } from "@/lib/api";
import { getClient, type ConversationRow } from "@/lib/inference";
import { useToast } from "@/lib/toast";
import { toolResponseBlock } from "@/lib/toolFormat";
import { buildSysContent } from "@/lib/systemPrompt";
import { parseToolCalls } from "@/lib/parseToolCalls";
import { CHANNEL_OPEN, parseModelContent } from "@/lib/parseModel";
import {
    isExecutableTool,
    toolUsesLocation,
    resolveGeo,
    executeTool,
    type Units as ToolUnits,
} from "@/lib/tools";
import { runOrchestration, extractScript } from "@/lib/tools/orchestrator";
import { preprocessImage } from "@/lib/image_preprocess";
import { decodeAudioFile } from "@/lib/audio_decode";
import { saveThumb, loadThumbBlobUrl, deleteThumbs } from "@/lib/image_store";
import { readInflightState, writeInflightState, clearInflightState } from "@/lib/opfs";
import {
    writeConvSnapshot, readConvSnapshot, deleteConvSnapshot, listConvSnapshots, opfsQuota,
    writeSysWarmSnapshot, readSysWarmSnapshot,
} from "@/lib/opfs";
import { rlcvTokenCount } from "@/lib/convSnapshot";
import { saveInflightImage, saveInflightAudio, readInflightImages, readInflightAudio, clearInflightMedia } from "@/lib/inflight_media";
import { persistQueue, saveJobMedia, loadQueue, dropJobMedia } from "@/lib/queue_store";
import {
    INFLIGHT_KEY,
    MIN_SNAPSHOT_TOKENS, MAX_SNAPSHOT_BYTES, LRU_MAX_SNAPSHOTS,
    withTurnTimestamp,
    type InflightGen, type GenJob, stepWithTimeout, suggestTitle,
} from "@/lib/app-helpers";

/** Generate a queue/job id. crypto.randomUUID is available in all the
 *  contexts the PWA runs in (secure origins + workers). */
function newJobId(): string {
    try { return crypto.randomUUID(); }
    catch { return `job-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`; }
}

export interface UseChatEngineParams {
    modelStatus: ModelStatus;
    loadedIsDiffusion: boolean;
    statusText: string;
    lastLoadedDigest: string;
    hasVision: boolean;
    hasAudio: boolean;
    systemPrompt: string;
    sampling: SamplingOptions;
    maxTokens: number;
    thinking: boolean;
    toolMode: boolean;
    orchestratorMode: boolean;
    weatherApiKey: string;
    newsApiKey: string;
    weatherUnits: ToolUnits;
    useGps: boolean;
    /** Active LoRA adapter name (or null). Part of the system pre-warm key —
     *  the adapter changes the cached K/V, so warms are per-adapter. */
    activeAdapter: string | null;
}

/**
 * Persist a completed conversation's KV cache, then enforce the LRU cap.
 * Best-effort: any failure (quota, write error) is swallowed — a missing
 * snapshot just means that conversation re-prefills on its next reopen.
 */
async function persistConvSnapshot(
    convId: string,
    bytes: Uint8Array,
    modelDigest: string,
    tokenCount: number,
): Promise<void> {
    try {
        // Prune FIRST so writing this one lands us at the LRU cap: keep the
        // newest (cap-1) OTHER snapshots, drop the rest. Pruning before the
        // write (not after) also frees room ahead of the quota check.
        const others = (await listConvSnapshots())
            .filter((s) => s.convId !== convId)
            .sort((a, b) => b.meta.updatedAt - a.meta.updatedAt);
        for (const stale of others.slice(LRU_MAX_SNAPSHOTS - 1)) {
            await deleteConvSnapshot(stale.convId);
        }
        // Quota guard: skip rather than risk filling OPFS (a full disk means
        // failed writes and, on iOS, jetsam pressure). GGUF weights already
        // dominate OPFS, so this can genuinely bind on tight devices.
        const q = await opfsQuota();
        if (q.quota > 0 && q.usage + bytes.length > q.quota * 0.95) {
            beacon("pe", "kv snapshot skipped (OPFS near quota)");
            return;
        }
        await writeConvSnapshot(convId, bytes, {
            modelDigest,
            version: 1,
            tokenCount,
            byteSize: bytes.length,
            updatedAt: Date.now(),
        });
    } catch (e) {
        // eslint-disable-next-line no-console
        console.warn("[rullama] conv KV snapshot persist failed:", e);
    }
}

/**
 * The entire chat domain: conversation list + active conversation,
 * displayed messages, the prompt box, pending image/audio attachments,
 * per-conversation RAG, and the generation engine itself — first-pass
 * `onSend` (text + multimodal + tool-calling + DiffusionGemma denoise) and
 * the suspend/resume machinery (`resumeInflightGeneration` + boot-resume).
 *
 * Reads model/tunable/tool state via params; everything else it owns. The
 * cross-tab sync needs `setConversations` + `inflightRef`, and the model
 * loader needs `resetForUnload` to clear chat display on eject/delete — all
 * returned.
 */
export function useChatEngine(opts: UseChatEngineParams) {
    const {
        modelStatus, loadedIsDiffusion, statusText, lastLoadedDigest,
        hasVision, hasAudio, systemPrompt, sampling, maxTokens, thinking,
        toolMode, orchestratorMode, weatherApiKey, newsApiKey, weatherUnits, useGps, activeAdapter,
    } = opts;
    const { showToast } = useToast();

    // Chat state
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [prompt, setPrompt]     = useState("");
    // `genActive` is "the engine is generating SOMEWHERE" (a running job in
    // this tab), NOT "the active conversation is locked". Sending, switching
    // conversations, and creating new chats are all allowed while it's true —
    // they enqueue / browse rather than block. Replaces the old global `busy`.
    const [genActive, setGenActive] = useState(false);
    const [statusLine, setStatusLine] = useState<string | undefined>();

    // ── Cross-conversation serial queue ────────────────────────────────
    // `queue` is the source of truth for the UI (running + queued jobs);
    // `queueRef` mirrors it for the pump + handlers (which need the latest
    // value inside long-lived async closures). Mutate via `commitQueue`,
    // which keeps all three in sync (ref, state, OPFS manifest).
    const [queue, setQueue] = useState<GenJob[]>([]);
    const queueRef = useRef<GenJob[]>([]);
    const pumpRunningRef = useRef(false);
    // Per-conversation live partial assistant text for the running job —
    // what lets a switch BACK to a generating conversation show tokens
    // produced past the last 16-token DB flush.
    const liveBuffersRef = useRef<Map<string, { modelMsgId: string; content: string }>>(new Map());
    // Conversations ever sent-to this session (for the new-chat reuse guard).
    const usedConvIds = useRef<Set<string>>(new Set());
    // One-shot guard for rebuilding the persisted queue on boot.
    const queueBootRef = useRef(false);
    const [visionEncodeState, setVisionEncodeState] = useState<PipelineProgressState | null>(null);

    // Pending attachments — session-only, cleared after each send.
    const [pendingImages, setPendingImages] = useState<ImageAttachment[]>([]);
    // Voice clips attached to the next user turn. Mirrors `pendingImages`.
    // PCM stays in-memory; we don't persist past sends (analogous to
    // image pixels — only thumbs / transcripts would land in SQLite,
    // and we don't have either yet for audio).
    const [pendingAudio, setPendingAudio] = useState<{ pcm: Float32Array; durationMs: number }[]>([]);

    // Conversation persistence (rsqlite-wasm)
    const [conversations, setConversations] = useState<ConversationRow[]>([]);
    // Persisted in sessionStorage (PER TAB) so a page reload reopens the same
    // conversation, without yanking other tabs to it (each tab keeps its own
    // selection; a fresh tab starts clean). Restored on boot once the
    // conversation list has loaded (see the bootstrap effect); a stale id
    // (deleted elsewhere) is cleared there.
    const [activeConvId, setActiveConvId]   = useState<string | null>(() => {
        try { return sessionStorage.getItem("rullama:activeConvId"); } catch { return null; }
    });
    // Mirror the active conversation into sessionStorage on every change so a
    // reload reopens it. All the existing `setActiveConvId` call sites stay
    // unchanged — this one effect keeps the store in sync.
    useEffect(() => {
        try {
            if (activeConvId) sessionStorage.setItem("rullama:activeConvId", activeConvId);
            else sessionStorage.removeItem("rullama:activeConvId");
        } catch { /* private mode / disabled storage — best-effort */ }
    }, [activeConvId]);

    // Mirrors `activeConvId` for the long-lived `runJob` closure, which must
    // paint the LATEST active conversation (not the one captured when the job
    // started). Initialized to the boot-time value.
    const activeConvIdRef = useRef<string | null>(activeConvId);
    useEffect(() => { activeConvIdRef.current = activeConvId; }, [activeConvId]);

    // Single point that keeps the queue ref, the render state, and the OPFS
    // manifest in sync. Only QUEUED jobs are persisted — a running job is
    // covered by the INFLIGHT_KEY resume path, and persisting it here would
    // double-run it on reload.
    const commitQueue = useCallback((next: GenJob[]) => {
        queueRef.current = next;
        setQueue(next);
        void persistQueue(next.filter((j) => j.status === "queued"));
    }, []);

    // Reflect a generation's growing display into the UI. ALWAYS keep the
    // conversation's live buffer fresh (so a switch-back can overlay the full
    // partial), but only repaint `messages` when the user is actually viewing
    // that conversation — otherwise a background job would clobber whatever
    // the user is reading.
    const reflectDisplay = useCallback((convId: string, modelMsgId: string, display: ChatMessage[]) => {
        const content = display[display.length - 1].content;
        const buf = liveBuffersRef.current.get(convId);
        if (buf) buf.content = content;
        else liveBuffersRef.current.set(convId, { modelMsgId, content });
        if (convId === activeConvIdRef.current) setMessages([...display]);
    }, []);

    const cancelRef = useRef(false);
    // Tracks the currently-running generation for suspend/resume. Mutated
    // per-token in the gen loop; serialized to localStorage on
    // visibilitychange→hidden so a kill-and-resume can pick up where we
    // left off. Cleared on clean completion / explicit cancel.
    const inflightRef = useRef<InflightGen | null>(null);
    // Resume-on-boot is single-shot; the effect can fire multiple times
    // as model state changes, but we want to attempt resume at most once.
    const resumeAttemptedRef = useRef(false);
    // Restore the persisted active conversation at most once on boot.
    const bootConvRestoredRef = useRef(false);
    // Conversations we've already tried to warm from a persisted KV
    // snapshot this session — restore is a one-shot per conversation (the
    // live KV cache takes over after the first send). Correctness never
    // depends on this; kvReusePlan's token check is the real gate.
    const restoreAttemptedRef = useRef<Set<string>>(new Set());

    // Global Escape → stop generation. Mirrors the toolbar Stop button.
    // Attached to window so it fires regardless of focus (input, sidebar
    // toggle, anything). No-op when nothing is generating.
    useEffect(() => {
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key !== "Escape") return;
            // Stop / dequeue the conversation the user is currently viewing.
            // If it's the running job, cancel it; if it's only queued, drop it.
            const id = activeConvIdRef.current;
            if (!id) return;
            const job = queueRef.current.find((j) => j.convId === id);
            if (!job) return;
            if (job.status === "running") {
                cancelRef.current = true;
                void getClient().cancelMultimodalEncode().catch(() => { /* */ });
            } else {
                commitQueue(queueRef.current.filter((j) => j.jobId !== job.jobId));
                void dropJobMedia(job.jobId);
            }
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [commitQueue]);

    // Suspend-on-background. iOS Safari fires visibilitychange→hidden
    // before suspending the WebContent process; we use that window to
    // (a) sync the inflight metadata to localStorage and (b) kick off
    // a GPU-state snapshot to OPFS. If iOS yanks us mid-write the
    // boot-resume fast path will be unavailable and the slow-path
    // replay (Phase F) picks up the slack via the partial response
    // already in the DB.
    useEffect(() => {
        if (!genActive) return;
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
    }, [genActive]);

    // Bootstrap DB + conversation list on mount, then reopen the persisted
    // active conversation (if any). `activeConvId` / `onSelectConversation`
    // are read from the first-render closure — exactly the boot-time values
    // we want — so they're intentionally not in the deps.
    useEffect(() => {
        const client = getClient();
        (async () => {
            try {
                await client.dbInit();
                const rows = await client.convList();
                setConversations(rows);
                if (!bootConvRestoredRef.current) {
                    bootConvRestoredRef.current = true;
                    const id = activeConvId;
                    if (rows.length === 0) {
                        // First-ever launch: start with one empty chat already
                        // present in the list (so the user never faces a blank
                        // sidebar), selected and ready to type into.
                        try {
                            const row = await client.convCreate({ title: "New chat", model: null });
                            setConversations([row]);
                            setActiveConvId(row.id);
                            setMessages([]);
                        } catch { /* DB hiccup — fall back to the empty welcome */ }
                    } else if (id && rows.some((r) => r.id === id)) {
                        // Reopen this tab's last conversation.
                        void onSelectConversation(id);
                    } else if (id) {
                        // Persisted conversation was deleted elsewhere — drop it
                        // and open the most-recent existing conversation.
                        void onSelectConversation(rows[0].id);
                    } else {
                        // Fresh tab, conversations exist — open the newest.
                        void onSelectConversation(rows[0].id);
                    }
                }
            } catch (e) {
                showToast({ level: "error", title: "Database init failed", message: (e as Error).message });
            }
        })();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [showToast]);

    const refreshConversations = useCallback(async () => {
        try {
            const rows = await getClient().convList();
            setConversations(rows);
        } catch (e) {
            showToast({ level: "warn", title: "Could not refresh conversations", message: (e as Error).message });
        }
    }, [showToast]);

    const onSelectConversation = useCallback(async (id: string) => {
        // No busy guard: browsing other conversations while a generation runs
        // in the background is the whole point. The running job keeps
        // streaming into its live buffer + the DB regardless of what's shown.
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
                        role:      r.role as ChatMessage["role"],
                        content:   r.content,
                        createdAt: r.created_at,   // frozen stamp for the [date time] prefix
                        ...(images && images.length ? { images } : {}),
                    };
                });
            // If this conversation has a generation in flight, the DB only
            // holds tokens up to the last 16-token flush. Overlay the live
            // buffer's full partial onto the trailing model bubble so switching
            // back shows the current text, not a stale prefix. (Replace, not
            // append: the buffer is the full accumulated content; the DB row is
            // a prefix of it.)
            const live = liveBuffersRef.current.get(id);
            if (live) {
                for (let i = ms.length - 1; i >= 0; i--) {
                    if (ms[i].role === "model") { ms[i] = { ...ms[i], content: live.content }; break; }
                }
            }
            setMessages(ms);
            setActiveConvId(id);
            setStatusLine(undefined);
        } catch (e) {
            showToast({ level: "error", title: "Failed to load conversation", message: (e as Error).message });
        }
    }, [showToast]);

    const onCreateConversation = useCallback(async () => {
        // No busy guard: a new chat can be started while another conversation
        // generates in the background. Create the DB row immediately so it
        // shows in the sidebar before any message is sent.
        const client = getClient();
        // Reuse-one-empty: never stack duplicate empty "New chat" rows. If an
        // unused empty one already exists, just select it. "Empty" = default
        // title, never sent-to this session, and no message rows in the DB
        // (the DB check guards against another tab having populated it).
        try {
            const empty = conversations.find(
                (c) => c.title === "New chat" && !usedConvIds.current.has(c.id),
            );
            if (empty) {
                const existing = await client.msgList(empty.id);
                if (existing.length === 0) {
                    void onSelectConversation(empty.id);
                    return;
                }
            }
        } catch { /* fall through to create a fresh row */ }
        try {
            const row = await client.convCreate({
                title: "New chat",
                model: modelStatus === "ready" ? statusText.split(" ")[0] : null,
            });
            setConversations((c) => [row, ...c]);   // newest-first, matches convList order
            setMessages([]);
            setActiveConvId(row.id);
            setStatusLine(undefined);
            setPendingImages([]);
            setPendingAudio([]);
        } catch (e) {
            showToast({ level: "error", title: "Couldn't start a new chat", message: (e as Error).message });
        }
        // The worker's KV cache is shared across tabs; a "new chat" here
        // doesn't preemptively reset it — runJob resets inside its own
        // session window when this conversation's first message runs.
    }, [conversations, modelStatus, statusText, onSelectConversation, showToast]);

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
        const c = conversations.find((x) => x.id === id);
        if (!window.confirm(`Delete conversation "${c?.title ?? id}"?\n\nMessages cannot be recovered.`)) return;
        // Tear down any in-flight or queued work for this conversation first.
        const job = queueRef.current.find((j) => j.convId === id);
        if (job) {
            if (job.status === "running") {
                // Cancel the running job; the pump finalizes the partial and
                // advances. cancelRef is reset at the next runJob's start.
                cancelRef.current = true;
            } else {
                // Drop the not-yet-started job from the queue.
                commitQueue(queueRef.current.filter((j) => j.jobId !== job.jobId));
            }
            void dropJobMedia(job.jobId);
        }
        liveBuffersRef.current.delete(id);
        try {
            // convDelete returns the OPFS image paths it collected
            // before the FK cascade nuked the rows; we sweep them
            // here so the on-disk thumbnails don't orphan.
            const { opfsPaths } = await getClient().convDelete(id);
            if (opfsPaths.length > 0) {
                void deleteThumbs(opfsPaths);
            }
            // Sweep the persisted KV snapshot too (deterministic filename).
            void deleteConvSnapshot(id);
            restoreAttemptedRef.current.delete(id);
            await refreshConversations();
            if (id === activeConvId) {
                setActiveConvId(null);
                setMessages([]);
            }
        } catch (e) {
            showToast({ level: "error", title: "Delete failed", message: (e as Error).message });
        }
    }, [activeConvId, commitQueue, conversations, refreshConversations, showToast]);

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
    const resumeInflightGeneration = useCallback(async (meta: InflightGen, opts?: { keepActive?: boolean }) => {
        // `keepActive` is set when the queue pump drives a live-recovery
        // inline: the pump owns the genActive flag across the whole drain, so
        // resume must not toggle it here.
        const keepActive = opts?.keepActive ?? false;
        const client = getClient();
        cancelRef.current = false;
        if (!keepActive) setGenActive(true);

        // Reconstruct the visible chat so the user sees the partial
        // response while we work to continue it.
        const display: ChatMessage[] = [
            ...meta.priorMessages,
            { role: "user",  content: meta.userText },
            { role: "model", content: meta.emittedSoFar },
        ];
        setMessages(display);
        setActiveConvId(meta.convId);
        // Seed the live buffer so switching away/back during the resume keeps
        // showing the partial.
        liveBuffersRef.current.set(meta.convId, { modelMsgId: meta.modelMsgId, content: meta.emittedSoFar });
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
                liveBuffersRef.current.delete(meta.convId);
                if (!keepActive) setGenActive(false);
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
                // with_bos=true to match the main send path — the resume
                // replay must rebuild the SAME token stream the KV cache was
                // built from, or restoreKvState / slow-path replay diverges.
                const rendered = await client.renderChatForContinuation(renderMsgs, true);
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
                reflectDisplay(meta.convId, meta.modelMsgId, display);
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
            liveBuffersRef.current.delete(meta.convId);
            try { localStorage.removeItem(INFLIGHT_KEY); } catch { /* */ }
            void clearInflightState();
            void clearInflightMedia();
            try { await client.releaseSession(); } catch { /* */ }
            if (!keepActive) setGenActive(false);
        }
    }, [reflectDisplay, refreshConversations, showToast]);

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

    // Run ONE queued generation to completion. This is the bulk of the old
    // `onSend` body, now parameterized by a GenJob so the serial pump can run
    // jobs for any conversation (not just the active one). It reads turn data
    // and tunables from `job` (captured at enqueue), reflects tokens into the
    // UI only when its conversation is on screen, and persists to the DB +
    // live buffer regardless. The pump (kickPump) owns session ordering and
    // the genActive flag; runJob does NOT touch them.
    const runJob = useCallback(async (job: GenJob) => {
        const client = getClient();
        cancelRef.current = false;
        // Flagged by the catch when a recoverable hang is detected so the
        // finally can hand off to resumeInflightGeneration instead of
        // clearing inflight state.
        let liveRecovery: InflightGen | null = null;

        const {
            convId, modelMsgId, userText: text, createdAt: nowMs,
            images: turnImages, audio: turnAudio, sysContent,
            sampling, maxTokens, thinking, toolMode, orchestratorMode,
            weatherApiKey, newsApiKey, weatherUnits, useGps, diffusion: diffusionTurn,
        } = job;

        // Prior-turn history. For a same-conversation queued send (or any job
        // rebuilt from OPFS on boot) load it from the DB so we chain off the
        // FINISHED previous answer, not a partial snapshot — dropping this
        // job's own trailing user + empty-model pair.
        let userTurns: ChatMessage[] = job.priorMessages;
        if (job.priorFromDb) {
            try {
                const rows = (await client.msgList(convId))
                    .filter((r) => r.role === "user" || r.role === "model");
                const mi = rows.findIndex((r) => r.message_id === modelMsgId);
                const cut = mi >= 1 ? mi - 1 : Math.max(0, rows.length - 2);
                userTurns = rows.slice(0, cut).map((r) => ({
                    role:      r.role as ChatMessage["role"],
                    content:   r.content,
                    createdAt: r.created_at,
                }));
            } catch { userTurns = []; }
        }

        // Two parallel histories: `displayHistory` for the chat UI (no system
        // message) and `renderHistory` for tokenization (system at the front).
        // The renderer-side content prepends one `<|image><image|>` sentinel
        // pair per attached image; JS splices soft-token embeddings between
        // them during the feed loop. Display bubbles show only the typed text
        // + thumbnails — sentinels never reach the user.
        const userDisplayMsg: ChatMessage = {
            role: "user",
            content: text,
            createdAt: nowMs,
            ...(turnImages.length ? { images: turnImages } : {}),
        };
        const displayHistory: ChatMessage[] = [
            ...userTurns,
            userDisplayMsg,
            { role: "model", content: "" },
        ];
        // Seed the live buffer (empty) and paint immediately if the user is
        // viewing this conversation.
        liveBuffersRef.current.set(convId, { modelMsgId, content: "" });
        reflectDisplay(convId, modelMsgId, displayHistory);

        // Audio markers go before image markers so the model "hears"
        // attachments in input order; the image splice already established
        // sentinel-pair-per-attachment as the prompt convention.
        const audioMarkers = "<|audio><audio|>".repeat(turnAudio.length);
        const imageMarkers = "<|image><image|>".repeat(turnImages.length);
        // Prefix the live `[date time]` onto the current user turn, and the
        // frozen stamp onto each historical user turn (model turns unchanged).
        const userRenderContent = withTurnTimestamp(audioMarkers + imageMarkers + text, nowMs);
        const renderPriorTurns: ChatMessage[] = userTurns.map((m) =>
            m.role === "user"
                ? { ...m, content: withTurnTimestamp(m.content, m.createdAt) }
                : m,
        );
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
                const replaceModelBubble = (canvas: string) => {
                    displayHistory[displayHistory.length - 1].content = canvas;
                    reflectDisplay(convId, modelMsgId, displayHistory);
                };
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
            // with_bos=true: Gemma 4 has add_bos_token=false, so the tokenizer
            // does NOT auto-prepend BOS — it must appear in the rendered text
            // (exactly what Ollama's gemma4 renderer does, gemma4.go:34). The
            // BPE splitter encodes the `<bos>` CONTROL token to the single BOS
            // id. Without it the model runs OUT OF DISTRIBUTION (the whole
            // sequence is mis-anchored): it emits a trivial 1-token reply and
            // never enters the `<|channel>thought` thinking channel. This was
            // the real reason thinking + tool-calling silently died.
            const rendered = await client.renderChat(renderHistory, true);
            const ids = await client.encode(rendered);

            // ── Cross-turn KV-cache prefix reuse ─────────────────────────────
            // Reuse as much of the resident KV cache as is a prefix of this
            // turn's token sequence, so we only prefill the NEW suffix instead
            // of re-reading the whole chat from `<bos>`. The resident cache may
            // be: this conversation continuing (full reuse), the pre-warmed
            // system block (new chat → system reused, hot-start), or another
            // conversation (LCP keeps the shared system head).
            //
            // For an existing conversation's FIRST send this session, restore
            // its persisted snapshot FIRST so we reuse the whole conversation
            // rather than just the warm system prefix. New conversations have
            // no snapshot → fall through to the warm/live cache. Media turns
            // can't be tracked (soft tokens) → reset + full prefill.
            const hasMedia = turnImages.length > 0 || turnAudio.length > 0;
            let reuse = 0;
            if (hasMedia) {
                await client.reset();
            } else {
                if (convId && !restoreAttemptedRef.current.has(convId)) {
                    restoreAttemptedRef.current.add(convId);
                    try {
                        const snap = await readConvSnapshot(convId);
                        if (snap && snap.meta.modelDigest === lastLoadedDigest) {
                            await client.restoreConvKv(snap.bytes);
                            beacon("pe", `kv snapshot restored (${snap.meta.tokenCount} tok)`);
                        }
                    } catch (e) {
                        // Missing/stale/mismatched snapshot — kvReusePlan below
                        // falls back to whatever's resident (warm system) or a
                        // full reset. No corruption: the token check is the gate.
                        // eslint-disable-next-line no-console
                        console.warn("[rullama] conv KV restore skipped:", e);
                    }
                }
                try {
                    reuse = (await client.kvReusePlan(Array.from(ids))).reuse;
                } catch {
                    // Older core without kvReusePlan, or any failure → reset
                    // and prefill the full sequence, as before.
                    await client.reset();
                    reuse = 0;
                }
            }
            // Degenerate guard: the template always appends an open
            // `<|turn>model\n`, so the new render is strictly longer than the
            // resident sequence and reuse < ids.length. If that ever fails to
            // hold, full-reset so the prefill loop feeds ≥1 token and `next`
            // is well-defined.
            if (reuse >= ids.length) {
                await client.reset();
                reuse = 0;
            }

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
            // `suffixLen` is the actual prefill work after KV reuse — what the
            // progress strip and the beacon report (not the full `ids.length`).
            const suffixLen = ids.length - reuse;
            beacon("pe", `prefill start (n_tokens=${suffixLen}, reused=${reuse}, imgs=${totalImgs}, audio=${turnAudio.length})`);
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
                total:    suffixLen,
            });
            // Start at `reuse`: positions [0, reuse) are already resident in
            // the KV cache, so we feed only ids[reuse..]. The model still
            // attends over the full prefix; we just skip recomputing it.
            for (let i = reuse; i < ids.length; i++) {
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
                    done:     i + 1 - reuse,
                    total:    suffixLen,
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
            let pendingDelta = "";
            let lastFlushAt  = performance.now();
            const flushPending = async () => {
                if (!convId || !modelMsgId || pendingDelta.length === 0) return;
                const delta = pendingDelta;
                pendingDelta = "";
                try { await client.msgAppend(convId, modelMsgId, delta); } catch { /* */ }
            };

            // Stream tokens into the (open) model bubble until EOS / cancel /
            // budget. Seeds from the outer `next` (the first sampled token) and
            // advances it; reused for the post-tool continuation pass below.
            const streamTurn = async (budget: number) => {
                let curStr   = (await client.tokenStr(next)) ?? "";
                let curIsEos = await client.isEos(next);
                for (let i = 0; i < budget; i++) {
                    if (cancelRef.current) break;
                    if (curIsEos) break;
                    const piece = curStr.replaceAll("▁", " ");
                    displayHistory[displayHistory.length - 1].content += piece;
                    pendingDelta += piece;
                    reflectDisplay(convId, modelMsgId, displayHistory);
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
            };

            await streamTurn(maxTokens);

            // ─── Orchestrator mode (Programmatic Tool Calling) ──────────────
            // The model just wrote a Rhai script orchestrating many tools.
            // Compile + run it (async tools bridged by memoized replay), and on
            // success show ONLY the final synthesized answer — the script and
            // its intermediate tool results stay hidden (that's the token win).
            // On compile/exec failure (the spike's residual ~1/3) drop into the
            // JSON `<tool_call>` loop below, in the SAME KV context: hide the
            // dead script from the user and feed a JSON-format correction, so
            // the loop's re-parse picks up real calls. Never wrong output —
            // worst case "no orchestrator this turn".
            let orchestratedOk = false;
            if (toolMode && orchestratorMode && !cancelRef.current) {
                const reply = parseModelContent(displayHistory[displayHistory.length - 1].content).response;
                const script = extractScript(reply);
                if (script) {
                    setStatusLine("orchestrating tools…");
                    // Resolve GPS once, only if the script names a location tool
                    // (don't prompt for coords the script never uses).
                    const usesLoc = /\b(get_weather|get_weather_forecast|get_air_quality|get_astronomy)\b/.test(script);
                    const geo = useGps && usesLoc ? await resolveGeo() : null;
                    const res = await runOrchestration(script, {
                        weatherApiKey, newsApiKey, units: weatherUnits, useGps, geo,
                        conversationId: convId ?? null,
                    });
                    setStatusLine(undefined);
                    if (res.ok) {
                        beacon("tg", `orchestrated ${res.calls.length} tool(s) in ${res.passes} pass(es)`);
                        displayHistory[displayHistory.length - 1].content = res.output;
                        reflectDisplay(convId, modelMsgId, displayHistory);
                        if (convId && modelMsgId) {
                            try { await client.msgSetContent(convId, modelMsgId, res.output); } catch { /* */ }
                        }
                        orchestratedOk = true;
                    } else {
                        // Fallback to the JSON loop. Hide the failed script (it
                        // stays in KV — can't un-feed it), then feed a correction
                        // that reframes the tools as JSON calls and regenerate.
                        beacon("tg", `orchestration failed (${res.error}); JSON-loop fallback`);
                        displayHistory[displayHistory.length - 1].content = "";
                        reflectDisplay(convId, modelMsgId, displayHistory);
                        if (convId && modelMsgId) {
                            try { await client.msgSetContent(convId, modelMsgId, ""); } catch { /* */ }
                        }
                        const note =
                            `\n<tool_response for="orchestrator">That script could not run (${res.error}). ` +
                            `Do NOT write a script. Answer the user now — if a tool is needed, emit ` +
                            `<tool_call>{"name": "<tool>", "arguments": { ... }}</tool_call> blocks; ` +
                            `otherwise just reply normally.</tool_response>\n`;
                        const noteIds = await client.encode(note);
                        for (const id of noteIds) { if (cancelRef.current) break; next = await client.step(id); }
                        if (!cancelRef.current) await streamTurn(maxTokens);
                    }
                }
            }

            // ─── Agentic tool loop ──────────────────────────────────────────
            // Each round: run EVERY executable <tool_call> the model just
            // emitted (so "weather and air quality" fires both at once), feed
            // the results back, and let the model either answer OR call MORE
            // tools — then repeat. That's the true multi-STEP loop: call one,
            // see its result, decide to call another, all inside this single
            // user turn. It exits the moment a continuation produces no new
            // executable call (the model's final answer), and is bounded by
            // MAX_TOOL_ROUNDS as a runaway guard. The KV cache is already
            // positioned right after the last tool call each round (streamTurn
            // broke on EOS without feeding it), so we feed the response tokens
            // and continue — no reset, no re-prefill.
            //
            // Cross-round correctness rests on `!c.result`: calls executed in a
            // prior round already carry their result (parseToolCalls re-attaches
            // every <tool_response> by name on each parse), so only the freshly
            // emitted calls are picked up here.
            const MAX_TOOL_ROUNDS = 5;
            for (let round = 0; toolMode && !orchestratedOk && !cancelRef.current && round < MAX_TOOL_ROUNDS; round++) {
                const { calls } = parseToolCalls(displayHistory[displayHistory.length - 1].content);
                const execCalls = calls.filter(
                    (c): c is typeof c & { arguments: Record<string, unknown> } =>
                        !c.pending && !c.result && isExecutableTool(c.name)
                        && !!c.arguments && typeof c.arguments === "object",
                );
                if (execCalls.length === 0) break;   // model answered — no new tools

                setStatusLine(`running ${execCalls.length} tool${execCalls.length > 1 ? "s" : ""}…`);

                // Resolve GPS once (shared by all calls) — only if at least one
                // location-aware call gave no place (or said "here"). Never
                // prompt when every call already named a city. Cached across
                // rounds by resolveGeo, so a later round won't re-prompt.
                const anyNeedsGeo = useGps && execCalls.some((c) => {
                    if (!toolUsesLocation(c.name)) return false;
                    const loc = typeof c.arguments.location === "string" ? c.arguments.location.trim() : "";
                    return loc === "" || /\b(current|my location|here|nearby)\b/i.test(loc);
                });
                const geo = anyNeedsGeo ? await resolveGeo() : null;

                // Execute all calls concurrently; executeTool never rejects
                // (failures come back as ok:false with an explanatory summary),
                // so order is preserved for result→call matching.
                const results = await Promise.all(
                    execCalls.map((c) =>
                        executeTool(c.name, c.arguments, {
                            weatherApiKey,
                            newsApiKey,
                            units: weatherUnits,
                            useGps,
                            geo,
                            conversationId: convId ?? null,
                        })),
                );

                // One named <tool_response for="..."> block per call, in
                // emission order; the renderer matches each to its call.
                let respBlock = "\n";
                execCalls.forEach((c, i) => {
                    respBlock += toolResponseBlock(c.name, results[i].summary) + "\n";
                });
                // When thinking is on, PRIME the reasoning channel after the
                // results. Left to itself the model reasons about the result in
                // the open (and even emits a stray `<channel|>` close it never
                // opened — the leaked "75°F…100°F" rambling). Seeding
                // `<|channel>thought` makes that post-tool reasoning land in a
                // proper, collapsible thought block before it answers or calls
                // the next tool. The renderer (parseToolCalls) splits each such
                // block out in order.
                if (thinking) respBlock += `${CHANNEL_OPEN}\n`;
                displayHistory[displayHistory.length - 1].content += respBlock;
                reflectDisplay(convId, modelMsgId, displayHistory);
                if (convId && modelMsgId) {
                    try { await client.msgAppend(convId, modelMsgId, respBlock); } catch { /* */ }
                }
                if (cancelRef.current) break;

                // Feed the response text into the model (continuing the KV
                // cache), then stream the continuation — which may answer or
                // emit the next round's tool calls.
                const respIds = await client.encode(respBlock);
                for (const id of respIds) {
                    if (cancelRef.current) break;
                    next = await client.step(id);
                }
                setStatusLine(undefined);
                await streamTurn(maxTokens);
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

            // Persist this conversation's KV cache so a future page reload
            // reopens it without re-prefilling the whole chain. Only for
            // trackable (text) turns; only when the chain is long enough to
            // be worth a GPU readback + write (short chats re-prefill
            // cheaply); size-capped to bound OPFS / iOS-jetsam pressure.
            // Done here, inside the still-held session (saveConvKv reads the
            // KV buffers), and best-effort — a failure never breaks the turn.
            if (!hasMedia && !cancelRef.current && convId && ids.length >= MIN_SNAPSHOT_TOKENS) {
                try {
                    const snapBytes = await client.saveConvKv();
                    if (snapBytes && snapBytes.length <= MAX_SNAPSHOT_BYTES) {
                        const tokenCount = rlcvTokenCount(snapBytes);
                        await persistConvSnapshot(convId, snapBytes, lastLoadedDigest, tokenCount);
                    } else if (snapBytes) {
                        beacon("pe", `kv snapshot skipped (${(snapBytes.length / 1048576).toFixed(0)} MB > cap)`);
                    }
                } catch (e) {
                    // eslint-disable-next-line no-console
                    console.warn("[rullama] conv KV snapshot save failed:", e);
                }
            }
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
                // Hand off to resumeInflightGeneration. Mirror to localStorage
                // so a page-reload-during-recovery still resumes via the boot
                // path. Run it INLINE (awaited) so the pump doesn't advance to
                // the next queued job until recovery finishes — and with
                // keepActive so the pump keeps owning the genActive flag.
                try { localStorage.setItem(INFLIGHT_KEY, JSON.stringify(liveRecovery)); } catch { /* */ }
                try { await client.releaseSession(); } catch { /* */ }
                const meta = liveRecovery;
                try { await resumeInflightGeneration(meta, { keepActive: true }); }
                catch { /* recovery best-effort; pump moves on */ }
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
            }
        }
    }, [lastLoadedDigest, reflectDisplay, refreshConversations, resumeInflightGeneration, showToast]);

    // ── Serial FIFO pump ───────────────────────────────────────────────
    // Drains queued jobs one at a time. Only ONE pump runs per tab
    // (pumpRunningRef); it reads the live queue from `queueRef` each
    // iteration so jobs enqueued mid-drain are picked up. Each runJob
    // acquires/releases the cross-tab session itself, so the session is
    // free between jobs (other tabs + warmSystemPrompt can interleave).
    const kickPump = useCallback(() => {
        if (pumpRunningRef.current) return;
        pumpRunningRef.current = true;
        setGenActive(true);
        void (async () => {
            try {
                for (;;) {
                    const job = queueRef.current.find((j) => j.status === "queued");
                    if (!job) break;
                    // Mark running (kept in the queue so the UI shows it).
                    commitQueue(queueRef.current.map((j) =>
                        j.jobId === job.jobId ? { ...j, status: "running" as const } : j));
                    try {
                        await runJob({ ...job, status: "running" });
                    } catch (e) {
                        // eslint-disable-next-line no-console
                        console.warn("[rullama] job failed:", e);
                    } finally {
                        // Remove from the queue, drop its persisted media + live
                        // buffer. persistQueue (via commitQueue) rewrites the
                        // manifest to just the still-queued jobs.
                        commitQueue(queueRef.current.filter((j) => j.jobId !== job.jobId));
                        liveBuffersRef.current.delete(job.convId);
                        void dropJobMedia(job.jobId);
                    }
                }
            } finally {
                pumpRunningRef.current = false;
                setGenActive(false);
            }
        })();
    }, [commitQueue, runJob]);

    // Build + enqueue a generation for the ACTIVE conversation, then kick the
    // pump. Never blocks on an in-flight generation — the message queues and
    // runs serially after whatever is already going. The user/empty-model rows
    // are persisted to the DB now (so the bubble shows immediately) and the
    // job (incl. attachment media) is persisted to OPFS so a reload re-runs it.
    const onSend = useCallback(async () => {
        if (modelStatus !== "ready") return;
        const text = prompt.trim();
        // Frozen send-time for this turn (reused for the rendered `[date time]`
        // prefix and the persisted created_at, so the KV-cache stamp matches
        // history re-renders).
        const nowMs = Date.now();
        const turnImages = pendingImages;
        const turnAudio  = pendingAudio;
        if (!text && turnImages.length === 0 && turnAudio.length === 0) return;
        const client = getClient();

        // Resolve / create the target conversation. With new-chat-immediately,
        // activeConvId is normally already a real row; create one as a safety
        // net if somehow null.
        let convId = activeConvId;
        if (!convId) {
            try {
                const row = await client.convCreate({
                    title: "New chat",
                    model: modelStatus === "ready" ? statusText.split(" ")[0] : null,
                });
                convId = row.id;
                setConversations((c) => [row, ...c]);
                setActiveConvId(convId);
            } catch (e) {
                showToast({ level: "error", title: "Couldn't start the chat", message: (e as Error).message });
                return;
            }
        }

        // Per-turn dynamic system add-on (resolved at ENQUEUE time so the job is
        // self-contained and runs with the settings in effect now). The static
        // part stays byte-identical to the pre-warmed system block so KV reuse
        // keeps the cached system prefix. GPS is per-turn and not part of the
        // warm. (RAG is no longer injected here — it's the on-demand
        // `search_knowledge` tool now, so retrieval never bloats the prefix.)
        let gpsLine = "";
        if (toolMode && useGps) {
            try {
                const coords = await resolveGeo();
                if (coords) {
                    gpsLine =
                        `The user's current location is approximately ${coords} ` +
                        `(latitude,longitude). For weather or other location-aware ` +
                        `tools, when the user does not name a specific place, use ` +
                        `exactly "${coords}" as the location argument.\n\n`;
                }
            } catch { /* geolocation denied / unavailable — proceed without it */ }
        }
        const sysContent = buildSysContent({ systemPrompt, thinking, toolMode, orchestratorMode, gpsLine });

        // Snapshot the prior turns NOW (for the in-memory fast path). If this
        // conversation already has a pending/running job, chain off the DB
        // instead at run time (priorFromDb) so the new job sees the FINISHED
        // prior answer rather than a partial.
        const priorMessages = messages.filter((m) => m.role !== "system");
        const priorFromDb = queueRef.current.some((j) => j.convId === convId);

        // Persist the user row + image thumbs + empty model row up front, so
        // the conversation shows the message + an empty bubble immediately —
        // even before this job's turn in the queue comes up.
        let modelMsgId = "";
        try {
            const userInsert = await client.msgInsert({ conversationId: convId, role: "user", content: text, createdAt: nowMs });
            for (let i = 0; i < turnImages.length; i++) {
                const im = turnImages[i];
                try {
                    const opfsPath = await saveThumb(im.dataUrl);
                    await client.msgInsertImage({
                        conversationId: convId, messageId: userInsert.messageId,
                        seq: i, width: im.w, height: im.h, opfsPath,
                    });
                } catch (e) {
                    showToast({ level: "warn", title: "Image not persisted", message: (e as Error).message });
                }
            }
            const modelInsert = await client.msgInsert({ conversationId: convId, role: "model", content: "" });
            modelMsgId = modelInsert.messageId;
        } catch (e) {
            showToast({ level: "warn", title: "Persistence failure", message: (e as Error).message });
            return;
        }

        const job: GenJob = {
            jobId: newJobId(),
            convId,
            modelMsgId,
            userText: text,
            createdAt: nowMs,
            priorFromDb,
            priorMessages,
            sysContent,
            sampling,
            maxTokens,
            thinking,
            toolMode,
            orchestratorMode,
            weatherApiKey,
            newsApiKey,
            weatherUnits,
            useGps,
            diffusion: loadedIsDiffusion,
            modelDigest: lastLoadedDigest,
            images: turnImages,
            audio: turnAudio,
            status: "queued",
        };
        usedConvIds.current.add(convId);

        // Optimistically show the user's message + an empty model bubble in
        // the active view, and seed the live buffer so a switch-away keeps it.
        // ONLY when this conversation has no job already in flight: if it does
        // (priorFromDb), that running job owns the on-screen display + the
        // (convId-keyed) live buffer until it finishes, so this queued turn
        // must not stomp either — it renders when its own runJob begins.
        if (!priorFromDb) {
            if (convId === activeConvIdRef.current) {
                const userDisplayMsg: ChatMessage = {
                    role: "user", content: text, createdAt: nowMs,
                    ...(turnImages.length ? { images: turnImages } : {}),
                };
                setMessages([...priorMessages, userDisplayMsg, { role: "model", content: "" }]);
            }
            liveBuffersRef.current.set(convId, { modelMsgId, content: "" });
        }

        // Clear the composer.
        setPrompt("");
        setPendingImages([]);
        setPendingAudio([]);
        setStatusLine(undefined);

        // Persist media files, then enqueue (manifest write) + kick the pump.
        void saveJobMedia(job);
        commitQueue([...queueRef.current, job]);
        kickPump();
    }, [
        activeConvId, commitQueue, kickPump, lastLoadedDigest, loadedIsDiffusion,
        maxTokens, messages, modelStatus, pendingAudio, pendingImages, prompt,
        sampling, statusText, systemPrompt, thinking, toolMode, orchestratorMode,
        useGps, weatherApiKey, newsApiKey, weatherUnits, showToast,
    ]);

    // Boot-resume the persisted queue: once the model is ready, rebuild any
    // jobs that were queued (behind a running generation) before the last
    // reload and drain them. The running job itself resumes via the separate
    // INFLIGHT_KEY path above; queued jobs were never in that manifest, so
    // there's no double-run. Filter to the current model + still-existing
    // conversations (their user/empty-model rows are already in the DB, so
    // they render correctly before their turn).
    useEffect(() => {
        if (modelStatus !== "ready") return;
        if (queueBootRef.current) return;
        queueBootRef.current = true;
        void (async () => {
            try {
                const jobs = await loadQueue();
                if (jobs.length === 0) return;
                const convRows = await getClient().convList();
                const convIds = new Set(convRows.map((r) => r.id));
                const valid: GenJob[] = [];
                for (const j of jobs) {
                    if (j.modelDigest && lastLoadedDigest && j.modelDigest !== lastLoadedDigest) continue;
                    if (!convIds.has(j.convId)) { void dropJobMedia(j.jobId); continue; }
                    usedConvIds.current.add(j.convId);
                    valid.push(j);
                }
                if (valid.length === 0) return;
                commitQueue([...queueRef.current, ...valid]);
                kickPump();
            } catch (e) {
                // eslint-disable-next-line no-console
                console.warn("[rullama] queue boot-resume failed:", e);
            }
        })();
    }, [modelStatus, lastLoadedDigest, commitQueue, kickPump]);

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

    // Stop / dequeue the conversation the user is viewing (toolbar Stop button
    // + Escape key). Targets the ACTIVE conversation's job: cancel it if it's
    // running, or drop it from the queue if it hasn't started — without
    // touching any other conversation's in-flight or queued work.
    const onStop = useCallback(() => {
        const id = activeConvIdRef.current;
        if (!id) return;
        const job = queueRef.current.find((j) => j.convId === id);
        if (!job) return;
        if (job.status === "running") {
            cancelRef.current = true;
            // Also break in on any in-flight multimodal encode. The flag is
            // cleared at the start of the next encode, so calling
            // unconditionally here doesn't poison subsequent runs.
            void getClient().cancelMultimodalEncode().catch(() => { /* */ });
        } else {
            commitQueue(queueRef.current.filter((j) => j.jobId !== job.jobId));
            void dropJobMedia(job.jobId);
        }
    }, [commitQueue]);

    // Pre-warm the system prompt into the KV cache so the next NEW chat
    // hot-starts (kvReusePlan reuses the resident system prefix and prefills
    // only the user's message). Called by the model loader during its
    // "preparing" phase (with `report` driving the progress bar) and after
    // the system prompt is saved (no report — the editor shows its own
    // "preparing" state). `overrides.systemPrompt` lets the save path warm
    // with the just-entered value without waiting for the state update.
    const warmSystemPrompt = useCallback(async (
        report?: (percent: number, label: string) => void,
        overrides?: { systemPrompt?: string },
    ) => {
        if (loadedIsDiffusion) return;   // no AR KV cache to warm
        if (genActive) return;           // don't fight an in-flight turn for the session
        const client = getClient();
        const sysContent = buildSysContent({
            systemPrompt: overrides?.systemPrompt ?? systemPrompt,
            thinking,
            toolMode,
            orchestratorMode,
        });
        // Render JUST the system turn. The trailing `<|turn>model\n` differs
        // from a real turn's `<|turn>user`, but LCP reuse keeps the whole
        // system block regardless (only the 1-token opener is re-fed).
        const rendered = await client.renderChat([{ role: "system", content: sysContent }], true);
        const ids = await client.encode(rendered);
        // Identity of this warmed system block. The persisted warm is keyed
        // by model digest (one file per model) and restored only when this
        // signature matches — so the SAME system prompt is never prefilled
        // more than once per model, even across page reloads. Includes the
        // active adapter, since LoRA changes the cached K/V.
        const sig = `${lastLoadedDigest} ${activeAdapter ?? ""} ${sysContent}`;
        // Persist/restore only when NO adapter is active. A LoRA changes the
        // cached K/V, but the load-time warm runs before the adapter is
        // applied, so an adapter-tagged warm could hold base-weight KV and be
        // wrongly restored once the adapter is live. Adapter users recompute
        // the warm each load (correctness over the optimization).
        const canPersist = !!lastLoadedDigest && !activeAdapter;
        await client.acquireSession();
        let off: (() => void) | undefined;
        try {
            // Fast path: restore a persisted warm for this exact config —
            // skips the prefill entirely (just writes the KV buffers).
            if (canPersist) {
                try {
                    const snap = await readSysWarmSnapshot(lastLoadedDigest);
                    if (snap && snap.meta.sig === sig) {
                        await client.restoreConvKv(snap.bytes);
                        report?.(100, "preparing model… (restored)");
                        beacon("pe", `syswarm restored (${snap.meta.tokenCount} tok)`);
                        return;
                    }
                } catch (e) {
                    // Corrupt / stale / layout-mismatch — fall through to compute.
                    // eslint-disable-next-line no-console
                    console.warn("[rullama] syswarm restore skipped:", e);
                }
            }
            // Compute the warm, then persist it for next time.
            if (report) {
                off = client.subscribe("warmProgress", (p) => {
                    const done = Number((p as { done?: number }).done ?? 0);
                    const total = Number((p as { total?: number }).total ?? 0);
                    report(total > 0 ? Math.round((done / total) * 100) : 0,
                        `preparing model… ${done}/${total}`);
                });
            }
            await client.warmSystem(Array.from(ids));
            if (canPersist) {
                try {
                    const bytes = await client.saveConvKv();
                    if (bytes && bytes.length <= MAX_SNAPSHOT_BYTES) {
                        await writeSysWarmSnapshot(lastLoadedDigest, bytes, {
                            modelDigest: lastLoadedDigest,
                            sig,
                            version: 1,
                            tokenCount: rlcvTokenCount(bytes),
                            byteSize: bytes.length,
                            updatedAt: Date.now(),
                        });
                    }
                } catch (e) {
                    // eslint-disable-next-line no-console
                    console.warn("[rullama] syswarm persist failed:", e);
                }
            }
        } finally {
            off?.();
            try { await client.releaseSession(); } catch { /* */ }
        }
    }, [systemPrompt, thinking, toolMode, orchestratorMode, loadedIsDiffusion, genActive, lastLoadedDigest, activeAdapter]);

    // Clear the chat display surface — called by the model loader on
    // eject / delete-while-loaded.
    const resetForUnload = useCallback(() => {
        setMessages([]);
        setStatusLine(undefined);
        setPendingImages([]);
        setPendingAudio([]);
        // Model changed/ejected → the resident KV is gone and any per-conv
        // warm-from-snapshot decision is stale; re-evaluate on next open.
        restoreAttemptedRef.current.clear();
    }, []);

    // Per-conversation generation status for the sidebar indicators + the
    // active-conversation Stop/Send gating. Derived from the queue (running
    // jobs stay in it until finished).
    const runningConvIds = useMemo(
        () => new Set(queue.filter((j) => j.status === "running").map((j) => j.convId)),
        [queue],
    );
    const queuedConvIds = useMemo(
        () => new Set(queue.filter((j) => j.status === "queued").map((j) => j.convId)),
        [queue],
    );
    const activeConvIsGenerating = activeConvId != null && runningConvIds.has(activeConvId);
    const activeConvIsQueued     = activeConvId != null && queuedConvIds.has(activeConvId);

    return {
        messages,
        prompt, setPrompt,
        // `busy` now means "the engine is generating somewhere" (not "this
        // conversation is locked"). Kept under the old name for the consumers
        // that only need wake-lock / banner gating.
        busy: genActive,
        genActive,
        runningConvIds,
        queuedConvIds,
        activeConvIsGenerating,
        activeConvIsQueued,
        statusLine,
        visionEncodeState,
        pendingImages,
        pendingAudio,
        conversations, setConversations,
        activeConvId,
        inflightRef,
        refreshConversations,
        onSelectConversation,
        onCreateConversation,
        onDeleteConversation,
        onAttachFiles,
        onRemoveImage,
        onCaptureAudio,
        onRemoveAudio,
        onAudioError,
        onSend,
        onStop,
        resetForUnload,
        warmSystemPrompt,
    };
}
