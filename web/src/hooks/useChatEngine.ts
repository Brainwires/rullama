import { useCallback, useEffect, useRef, useState } from "react";
import { type ChatMessage, type ImageAttachment, type SamplingOptions } from "@/lib/types";
import type { PipelineProgressState } from "@/components/PipelineProgress";
import { type ModelStatus } from "@/components/ModelLoader";
import { beacon } from "@/lib/api";
import { getClient, type ConversationRow } from "@/lib/inference";
import { useToast } from "@/lib/toast";
import { TOOL_SCHEMA_PROMPT, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE } from "@/lib/toolFormat";
import { parseToolCalls } from "@/lib/parseToolCalls";
import {
    isExecutableTool,
    toolUsesLocation,
    resolveGeo,
    executeTool,
    type Units as ToolUnits,
} from "@/lib/tools";
import { searchKnowledge, buildRagPreamble } from "@/lib/embedding";
import { preprocessImage } from "@/lib/image_preprocess";
import { decodeAudioFile } from "@/lib/audio_decode";
import { saveThumb, loadThumbBlobUrl, deleteThumbs } from "@/lib/image_store";
import { readInflightState, writeInflightState, clearInflightState } from "@/lib/opfs";
import {
    writeConvSnapshot, readConvSnapshot, deleteConvSnapshot, listConvSnapshots, opfsQuota,
} from "@/lib/opfs";
import { rlcvTokenCount } from "@/lib/convSnapshot";
import { saveInflightImage, saveInflightAudio, readInflightImages, readInflightAudio, clearInflightMedia } from "@/lib/inflight_media";
import {
    THINK_TOKEN, INFLIGHT_KEY,
    MIN_SNAPSHOT_TOKENS, MAX_SNAPSHOT_BYTES, LRU_MAX_SNAPSHOTS,
    type InflightGen, stepWithTimeout, suggestTitle,
} from "@/lib/app-helpers";

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
    weatherApiKey: string;
    weatherUnits: ToolUnits;
    useGps: boolean;
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
        toolMode, weatherApiKey, weatherUnits, useGps,
    } = opts;
    const { showToast } = useToast();

    // Chat state
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [prompt, setPrompt]     = useState("");
    const [busy, setBusy]         = useState(false);
    const [statusLine, setStatusLine] = useState<string | undefined>();
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
    const [activeConvId, setActiveConvId]   = useState<string | null>(null);
    // Per-conversation RAG toggle (Knowledge-base grounding). Loaded from
    // the DB when the active conversation changes.
    const [ragEnabled, setRagEnabled]       = useState<boolean>(false);

    const cancelRef = useRef(false);
    // Tracks the currently-running generation for suspend/resume. Mutated
    // per-token in the gen loop; serialized to localStorage on
    // visibilitychange→hidden so a kill-and-resume can pick up where we
    // left off. Cleared on clean completion / explicit cancel.
    const inflightRef = useRef<InflightGen | null>(null);
    // Resume-on-boot is single-shot; the effect can fire multiple times
    // as model state changes, but we want to attempt resume at most once.
    const resumeAttemptedRef = useRef(false);
    // Conversations we've already tried to warm from a persisted KV
    // snapshot this session — restore is a one-shot per conversation (the
    // live KV cache takes over after the first send). Correctness never
    // depends on this; kvReusePlan's token check is the real gate.
    const restoreAttemptedRef = useRef<Set<string>>(new Set());

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
    }, [activeConvId, busy, conversations, refreshConversations, showToast]);

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

        // **Tool-calling schema injection.** When tool mode is on, prepend the
        // tool schema (TOOL_SCHEMA_PROMPT, byte-identical to tool-schema.txt) so
        // the base model emits <tool_call> blocks — parseToolCalls renders them
        // as structured blocks (accepts both JSON and pythonic syntax). No
        // adapter / fine-tune needed; the schema-in-prompt does the work.
        if (toolMode) {
            baseSystem = baseSystem ? `${TOOL_SCHEMA_PROMPT}\n\n${baseSystem}` : TOOL_SCHEMA_PROMPT;
            // **GPS location injection.** Resolve the user's coordinates up
            // front (cached; the browser asks permission once) and tell the
            // model to use them when no place is named. Without this, small
            // models INVENT a city rather than leaving the location empty, so
            // the post-hoc "current location" detection on the tool call never
            // fires — which is why GPS appeared to do nothing. Best-effort: a
            // denial / timeout just proceeds without a location hint.
            if (useGps) {
                try {
                    const coords = await resolveGeo();
                    if (coords) {
                        baseSystem =
                            `The user's current location is approximately ${coords} ` +
                            `(latitude,longitude). For weather or other location-aware ` +
                            `tools, when the user does not name a specific place, use ` +
                            `exactly "${coords}" as the location argument.\n\n${baseSystem}`;
                    }
                } catch { /* geolocation denied / unavailable — proceed without it */ }
            }
        }

        // Thinking always respects its own toggle — never silently overridden.
        // Tool calling and thinking coexist: the schema is CONDITIONAL — it
        // tells the model to emit a tool call only when one fits, and otherwise
        // reason + answer normally. So a non-tool prompt still gets a normal
        // (thinking) response; a tool prompt gets the call. No code gating.
        //
        // The `<|think|>` control token MUST be followed by a newline and then
        // the system content — this is exactly how Ollama's gemma4 renderer
        // emits it (`<|turn>system\n<|think|>\n{system}<turn|>`, see
        // model/renderers/gemma4.go). Gluing the token directly onto a long
        // system prompt (`<|think|>You have access to…`) is NOT the trained
        // pattern and the model fails to enter the thinking channel — which is
        // why thinking silently stopped once the tool schema was injected.
        const sysContent = thinking
            ? `${THINK_TOKEN}\n${baseSystem}`
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
            // The KV cache already holds the system prompt + every prior turn
            // of this conversation from the last send (we no longer reset
            // between turns). Ask the core how much of this turn's full token
            // sequence is a usable prefix of the resident cache, so we only
            // prefill the NEW suffix instead of re-reading the whole chat from
            // `<bos>` — the "Reading prompt N/total" phase that otherwise grows
            // with conversation length.
            //
            // Media turns feed soft-token embeds the core can't track as plain
            // ids, so they reset + full-prefill. The core also resets whenever
            // the resident cache is NOT a strict prefix (edited history,
            // changed system prompt, another tab drove a turn, fresh load) and
            // returns reuse=0 — making full-prefill the safe fallback, i.e.
            // exactly the legacy behaviour.
            const hasMedia = turnImages.length > 0 || turnAudio.length > 0;
            let reuse = 0;
            if (hasMedia) {
                await client.reset();
            } else {
                try {
                    reuse = (await client.kvReusePlan(Array.from(ids))).reuse;
                    // First send into this conversation this session and the
                    // live cache didn't match (fresh tab / page reload)? Try
                    // to warm the KV from a persisted snapshot, then re-check
                    // the prefix. One attempt per conversation — the live KV
                    // takes over afterward. kvReusePlan's token check stays
                    // the sole correctness gate; this is purely an optimization.
                    if (reuse === 0 && convId && !restoreAttemptedRef.current.has(convId)) {
                        restoreAttemptedRef.current.add(convId);
                        try {
                            const snap = await readConvSnapshot(convId);
                            if (snap && snap.meta.modelDigest === lastLoadedDigest) {
                                await client.restoreConvKv(snap.bytes);
                                reuse = (await client.kvReusePlan(Array.from(ids))).reuse;
                                if (reuse > 0) {
                                    beacon("pe", `kv snapshot restored (${snap.meta.tokenCount} tok)`);
                                }
                            }
                        } catch (e) {
                            // Stale / corrupt / model-mismatch snapshot. A
                            // failed restoreKvState validates before mutating,
                            // but reset to a clean cache to be safe.
                            await client.reset();
                            reuse = 0;
                            // eslint-disable-next-line no-console
                            console.warn("[rullama] conv KV restore failed:", e);
                        }
                    }
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
            };

            await streamTurn(maxTokens);

            // ─── Tool execution + answer continuation ───────────────────────
            // If the model emitted an EXECUTABLE <tool_call> (e.g. get_weather),
            // run it and splice the result back as a <tool_response> block, then
            // keep generating so the model answers in natural language. The KV
            // cache is already positioned right after the tool call (streamTurn
            // broke on EOS without feeding it), so we just feed the response
            // tokens and continue — no reset, no re-prefill.
            if (toolMode && !cancelRef.current) {
                const { calls } = parseToolCalls(displayHistory[displayHistory.length - 1].content);
                const exec = calls.find(
                    (c) => !c.pending && !c.result && isExecutableTool(c.name)
                        && c.arguments && typeof c.arguments === "object",
                );
                if (exec) {
                    setStatusLine(`running ${exec.name}…`);
                    let geo: string | null = null;
                    // Only consult GPS when the model gave no place (or asked
                    // for "here"/"my location") — never prompt when it already
                    // named a city.
                    const locArg = typeof (exec.arguments as Record<string, unknown>).location === "string"
                        ? String((exec.arguments as Record<string, unknown>).location).trim()
                        : "";
                    const needsGeo = locArg === "" || /\b(current|my location|here|nearby)\b/i.test(locArg);
                    if (useGps && toolUsesLocation(exec.name) && needsGeo) {
                        geo = await resolveGeo();
                    }
                    const result = await executeTool(
                        exec.name,
                        exec.arguments as Record<string, unknown>,
                        { weatherApiKey, units: weatherUnits, useGps },
                        geo,
                    );
                    const respBlock = `\n${TOOL_RESPONSE_OPEN}${result.summary}${TOOL_RESPONSE_CLOSE}\n`;
                    displayHistory[displayHistory.length - 1].content += respBlock;
                    setMessages([...displayHistory]);
                    if (convId && modelMsgId) {
                        try { await client.msgAppend(convId, modelMsgId, respBlock); } catch { /* */ }
                    }
                    if (!cancelRef.current) {
                        // Feed the response text into the model (continuing the
                        // KV cache), then stream the final answer.
                        const respIds = await client.encode(respBlock);
                        for (const id of respIds) {
                            if (cancelRef.current) break;
                            next = await client.step(id);
                        }
                        setStatusLine(undefined);
                        await streamTurn(maxTokens);
                    }
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
    }, [activeConvId, busy, lastLoadedDigest, loadedIsDiffusion, maxTokens, messages, modelStatus, pendingAudio, pendingImages, prompt, ragEnabled, toolMode, weatherApiKey, weatherUnits, useGps, refreshConversations, resumeInflightGeneration, sampling, statusText, systemPrompt, thinking, showToast]);

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

    // Stop the active generation (toolbar Stop button + Escape key).
    const onStop = useCallback(() => {
        cancelRef.current = true;
        // Also break in on any in-flight multimodal encode. The flag is
        // cleared at the start of the next encode, so calling
        // unconditionally here doesn't poison subsequent runs.
        void getClient().cancelMultimodalEncode().catch(() => { /* */ });
    }, []);

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

    return {
        messages,
        prompt, setPrompt,
        busy,
        statusLine,
        visionEncodeState,
        pendingImages,
        pendingAudio,
        conversations, setConversations,
        activeConvId,
        ragEnabled, toggleRag,
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
    };
}
