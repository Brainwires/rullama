// WorkerClient — thin RPC wrapper over the inference SharedWorker.
//
// One SharedWorker per origin owns the wasm Model, the chat DB, and the
// OPFS handles. Every tab opens a MessagePort to it and routes RPCs
// through that port. The `pending: Map<requestId, …>` keeps the
// promise-per-request mechanic; the transport switched from a Dedicated
// Worker's postMessage to a port's postMessage.
//
// Cross-tab inference is serialized via the worker's session arbitration
// (`acquireSession` / `releaseSession`); cancellation of a *queued*
// acquire is handled with an AbortSignal + an abort token the worker
// uses to drop the waiter from the queue.

import type { ChatMessage, SamplingOptions } from "@/lib/types";
import InferenceWorker from "@/workers/inference-worker?sharedworker";
import InferenceCoreWorker from "@/workers/inference-core-worker?worker";
import { requestRestart } from "@/lib/restart";

interface ModelMeta {
    vocabSize:        number;
    hasVision:        boolean;
    hasAudio:         boolean;
    imageSentinelIds: [number, number] | null;
    audioSentinelIds: [number, number] | null;
}

interface Pending {
    resolve: (v: unknown) => void;
    reject:  (e: Error) => void;
}

type NotifyMsg     = { type: "notify"; kind: string } & Record<string, unknown>;
type LogMsg        = { type: "log"; args: string[] };
type RpcReply      =
    | { requestId: number; ok: true;  result: unknown }
    | { requestId: number; ok: false; error: string };
type WorkerMsg     = NotifyMsg | LogMsg | RpcReply;

type NotifyHandler = (payload: Record<string, unknown>) => void;

/** Heuristic for "this looks like a stale-asset error after a deploy". */
function looksLikeStaleAssetError(msg: string): boolean {
    const m = msg.toLowerCase();
    return m.includes("failed to fetch")
        || m.includes("dynamically imported module")
        || m.includes("importing a module script")
        || m.includes("loading chunk")
        || m.includes("loading css chunk")
        || m.includes("script error")
        || m.includes("module specifier")
        || m.includes("webassembly")
        || m.includes("import error");
}

export class WorkerClient {
    private port: MessagePort;
    private pending = new Map<number, Pending>();
    private nextId  = 1;
    private subscribers = new Map<string, Set<NotifyHandler>>();

    /** When this tab is the elected core host, holds a reference to the
     *  spawned Dedicated Worker so we can terminate it on tab close. */
    private coreWorker: Worker | null = null;

    /** Resolved when the router signals `coreReady`. Reset to a new
     *  pending Promise whenever core is lost. The retry-once path in
     *  `rpc()` awaits this. */
    private coreReady: Promise<void>;
    private resolveCoreReady!: () => void;

    /** Last `notify: meta` payload from the worker (the currently-loaded
     *  model, or null when nothing is loaded). React state syncs to this
     *  via subscribe("meta", …). */
    public lastMeta: { loaded: ModelMeta | null } | null = null;

    /** Set after a successful `acquireSession`; cleared on
     *  `releaseSession` or implicit release (port disconnect, etc.). */
    private session: number | null = null;

    /** Last log line from the worker, useful in dev consoles. */
    public onLog?: (line: string) => void;

    constructor() {
        this.coreReady = new Promise<void>((r) => { this.resolveCoreReady = r; });
        let sw: SharedWorker;
        try {
            sw = new InferenceWorker();
        } catch (e) {
            requestRestart("the inference worker failed to construct");
            throw e;
        }
        this.port = sw.port;
        this.port.start();
        this.port.addEventListener("message", (ev: MessageEvent<WorkerMsg>) => {
            const m = ev.data;
            if (!m || typeof m !== "object") return;
            if ("type" in m && m.type === "log") {
                const line = (m as LogMsg).args.join(" ");
                console.log("[inference-worker]", line);
                this.onLog?.(line);
                return;
            }
            if ("type" in m && m.type === "notify") {
                const n = m as NotifyMsg;
                if (n.kind === "meta") {
                    this.lastMeta = { loaded: (n.loaded ?? null) as ModelMeta | null };
                }
                if (n.kind === "pleaseSpawnCore") {
                    this.spawnCore();
                }
                if (n.kind === "coreReady") {
                    this.resolveCoreReady();
                }
                if (n.kind === "modelFreed") {
                    // The router emits modelFreed when it loses the core
                    // (host tab closed) — at that point pending RPCs get
                    // rejected with "core disconnected" and we'll need a
                    // fresh coreReady before retrying.
                    this.coreReady = new Promise<void>((r) => { this.resolveCoreReady = r; });
                }
                const subs = this.subscribers.get(n.kind);
                if (subs) for (const h of subs) {
                    try { h(n as unknown as Record<string, unknown>); }
                    catch (e) { console.error("[notify handler]", e); }
                }
                return;
            }
            const r = m as RpcReply;
            if ("requestId" in r) {
                const p = this.pending.get(r.requestId);
                if (!p) return;
                this.pending.delete(r.requestId);
                if (r.ok) p.resolve(r.result);
                else      p.reject(new Error(r.error));
            }
        });
        // SharedWorker `onerror` exists; route through the same stale-
        // asset / restart-overlay path the old Dedicated Worker did.
        sw.addEventListener("error", (ev: ErrorEvent) => {
            const msg = ev.message || String(ev);
            console.error("[inference-worker] worker error:", msg);
            if (looksLikeStaleAssetError(msg)) {
                requestRestart("the inference worker failed to load");
            }
        });

        // Send a final `disconnect` on tab close so the router can
        // immediately release any held session + drop the port without
        // waiting for the heartbeat GC. If we're the elected core host,
        // also post `{type:"shutdown"}` to the dedicated worker so it
        // closes its FileSystemSyncAccessHandle / DB / Model and then
        // self.close()s on its own time.
        //
        // CRITICAL: do NOT call worker.terminate() here. terminate() is
        // synchronous and kills the worker before its onmessage loop has
        // a chance to process the shutdown message — which means
        // releaseAllHandles() never runs, the FileSystemSyncAccessHandle
        // stays held by the dead worker until iOS Safari finally GCs it,
        // and the next page load can't open the same OPFS file. The
        // 7 GB GGUF is sitting right there but the new core worker
        // races a lingering exclusive lock and the load fails. The whole
        // point of the shutdown message is to let the worker close
        // handles BEFORE exiting; terminate() defeats that.
        const onLeave = () => {
            try { this.port.postMessage({ requestId: -1, type: "disconnect" }); } catch { /* */ }
            const w = this.coreWorker;
            this.coreWorker = null;
            if (w) {
                try { w.postMessage({ type: "shutdown" }); } catch { /* */ }
                // Intentionally no terminate(). The worker calls
                // self.close() after releaseAllHandles() — see the
                // shutdown handler in inference-core-worker.ts.
            }
        };
        window.addEventListener("pagehide", onLeave);
        window.addEventListener("beforeunload", onLeave);

        // Heartbeat keeps the worker's port-liveness tracker happy. 10s
        // when the tab is foregrounded is well within the 5-minute
        // reaper window (HEARTBEAT_DEAD_MS in inference-worker.ts).
        // When the tab is BACKGROUNDED, Chrome throttles `setInterval`
        // to >=1-minute, so we additionally fire an immediate ping on
        // `visibilitychange` → visible. Without that, a tab returning
        // from a long background can briefly have a stale port until
        // the next throttled tick lands.
        setInterval(() => {
            try { this.port.postMessage({ requestId: -1, type: "ping" }); } catch { /* */ }
        }, 10_000);
        document.addEventListener("visibilitychange", () => {
            if (document.visibilityState === "visible") {
                try { this.port.postMessage({ requestId: -1, type: "ping" }); } catch { /* */ }
            }
        });
    }

    /**
     * Spawn the core Dedicated Worker for this tab (called when the
     * router elects this tab as host). The worker is spawned by the
     * main thread because SharedWorker can't expose the `Worker`
     * constructor in any browser. We then hand a MessageChannel port
     * over to both the worker and the router so they can talk directly.
     */
    private spawnCore(): void {
        if (this.coreWorker) return;
        let dw: Worker;
        try {
            dw = new InferenceCoreWorker();
        } catch (e) {
            console.error("[inference] failed to spawn core worker:", e);
            requestRestart("the inference core worker failed to construct");
            return;
        }
        this.coreWorker = dw;
        dw.addEventListener("error", (ev: ErrorEvent) => {
            const msg = ev.message || String(ev);
            console.error("[inference-core-worker] error:", msg);
            if (looksLikeStaleAssetError(msg)) {
                requestRestart("the inference core worker failed to load");
            }
        });
        const channel = new MessageChannel();
        // Hand port1 to the core worker. It listens for {type:'attach', port}.
        dw.postMessage({ type: "attach", port: channel.port1 }, [channel.port1]);
        // Hand port2 to the router via the attachCore RPC.
        const requestId = this.nextId++;
        this.pending.set(requestId, {
            resolve: () => { /* router replied ok */ },
            reject:  (e) => console.error("[inference] attachCore rejected:", e),
        });
        try {
            this.port.postMessage({ requestId, type: "attachCore" }, [channel.port2]);
        } catch (e) {
            console.error("[inference] attachCore postMessage failed:", e);
            this.pending.delete(requestId);
        }
    }

    private rpc<T = unknown>(type: string, args: Record<string, unknown> = {}): Promise<T> {
        return this.rpcOnce<T>(type, args, false);
    }

    /**
     * Send an RPC. If it fails with `core disconnected` and `retried`
     * is false, wait for the next `coreReady` notification and retry
     * the call exactly once. This covers the host-migration window
     * cleanly without callers having to handle it.
     */
    private rpcOnce<T>(type: string, args: Record<string, unknown>, retried: boolean): Promise<T> {
        const requestId = this.nextId++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(requestId, {
                resolve: (v) => resolve(v as T),
                reject: (e) => {
                    if (!retried && /core disconnected/.test(e.message)) {
                        // Wait for fresh core, then retry once.
                        this.coreReady
                            .then(() => this.rpcOnce<T>(type, args, true))
                            .then(resolve, reject);
                        return;
                    }
                    reject(e);
                },
            });
            this.port.postMessage({ requestId, type, ...args });
        });
    }

    // ── Notification subscription ──────────────────────────────────────
    subscribe(kind: string, handler: NotifyHandler): () => void {
        let set = this.subscribers.get(kind);
        if (!set) { set = new Set(); this.subscribers.set(kind, set); }
        set.add(handler);
        return () => { set!.delete(handler); };
    }

    // ── Session arbitration ────────────────────────────────────────────
    /**
     * Acquire the inference session. Resolves when this client owns the
     * model. If another tab is already generating, this awaits until they
     * release. Pass an AbortSignal to allow cancelling while queued.
     */
    async acquireSession(signal?: AbortSignal): Promise<number> {
        const abortToken = `t-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const requestId = this.nextId++;
        const sid = await new Promise<number>((resolve, reject) => {
            this.pending.set(requestId, {
                resolve: (v) => resolve(Number(v)),
                reject,
            });
            this.port.postMessage({ requestId, type: "acquireSession", abortToken });
            if (signal) {
                if (signal.aborted) {
                    // Already aborted before we sent — fire the cancel
                    // immediately so the worker doesn't keep us in queue.
                    try {
                        this.port.postMessage({ requestId: -1, type: "cancelAcquire", abortToken });
                    } catch { /* */ }
                    reject(new Error("aborted"));
                    return;
                }
                signal.addEventListener("abort", () => {
                    try {
                        this.port.postMessage({ requestId: -1, type: "cancelAcquire", abortToken });
                    } catch { /* */ }
                }, { once: true });
            }
        });
        this.session = sid;
        return sid;
    }

    async releaseSession(): Promise<void> {
        const sid = this.session;
        this.session = null;
        if (sid == null) return;
        try { await this.rpc("releaseSession", { sid }); } catch { /* worker may have released already */ }
    }

    currentSession(): number | null { return this.session; }

    /** Convenience: acquire → run fn → release. Forwards abort to the
     *  acquire wait; the fn itself runs once the session is held, so
     *  callers should use their own cooperative cancellation inside fn
     *  (e.g. `cancelRef` checks before each step). */
    async withSession<T>(fn: () => Promise<T>, signal?: AbortSignal): Promise<T> {
        await this.acquireSession(signal);
        try { return await fn(); }
        finally { await this.releaseSession(); }
    }

    // ── Model lifecycle (session-scoped) ───────────────────────────────
    async load(
        modelKey: string, filename: string,
        opts: { maxContext?: number; textOnly?: boolean; name?: string } = {},
    ): Promise<ModelMeta> {
        if (this.session == null) throw new Error("load requires an active session — call acquireSession first");
        const result = await this.rpc<{
            modelKey: string; filename: string; name?: string;
            vocabSize: number; hasVision: boolean; hasAudio: boolean;
            imageSentinelIds: [number, number] | null;
            audioSentinelIds: [number, number] | null;
        }>("load", {
            sid:       this.session,
            modelKey, filename,
            name:       opts.name,
            maxContext: opts.maxContext ?? 0,
            textOnly:   !!opts.textOnly,
        });
        // Mirror the meta the worker just broadcast.
        this.lastMeta = { loaded: {
            vocabSize:        result.vocabSize,
            hasVision:        result.hasVision,
            hasAudio:         result.hasAudio,
            imageSentinelIds: result.imageSentinelIds,
            audioSentinelIds: result.audioSentinelIds,
        } };
        return this.lastMeta.loaded as ModelMeta;
    }

    async free(): Promise<void> {
        if (this.session == null) throw new Error("free requires an active session");
        return this.rpc("free", { sid: this.session });
    }

    // ── Stateless meta accessors (cached from notify: meta or load) ────
    get vocabSize() { return this.lastMeta?.loaded?.vocabSize; }
    get hasVision() { return !!this.lastMeta?.loaded?.hasVision; }
    get hasAudio()  { return !!this.lastMeta?.loaded?.hasAudio; }
    imageSentinelIds() { return this.lastMeta?.loaded?.imageSentinelIds ?? null; }
    audioSentinelIds() { return this.lastMeta?.loaded?.audioSentinelIds ?? null; }

    // ── Stateless inference ────────────────────────────────────────────
    encode(text: string): Promise<Uint32Array> {
        return this.rpc<number[]>("encode", { text }).then((arr) => new Uint32Array(arr));
    }
    tokenStr(id: number): Promise<string | null> { return this.rpc("tokenStr", { id }); }
    isEos(id: number): Promise<boolean> { return this.rpc("isEos", { id }); }
    renderChat(messages: ChatMessage[], withBos: boolean): Promise<string> {
        return this.rpc("renderChat", { messages, withBos });
    }
    imageSoftTokenCount(h: number, w: number): Promise<number> {
        return this.rpc("imageSoftTokenCount", { h, w });
    }

    // ── Diagnostic logs (OPFS-backed, see workers/opfs_logger.ts) ──────
    /** Worker-side log file storage. `append` is fire-and-forget so
     *  the main thread never blocks on a beacon; `list/read/delete`
     *  back the Settings → Logs viewer. */
    readonly logs = {
        list:      ()                                  => this.rpc<LogSessionMeta[]>("logsList"),
        read:      (id: string)                        => this.rpc<string>("logsRead", { id }),
        delete:    (id: string)                        => this.rpc<boolean>("logsDelete", { id }),
        deleteAll: ()                                  => this.rpc<boolean>("logsDeleteAll"),
        append:    (level: LogLevel, tag: string, msg: string) => {
            void this.rpc<boolean>("logsAppend", { level, tag, msg });
        },
        currentId: ()                                  => this.rpc<string>("logsCurrentSession"),
    };

    // ── Stateful inference (auto-inject sid) ───────────────────────────
    step(tokenId: number): Promise<number> {
        return this.rpc("step", { sid: this.session, tokenId });
    }
    stepWithEmbedding(embedding: Float32Array): Promise<number> {
        return this.rpc("stepWithEmb", { sid: this.session, embedding });
    }
    stepAndDecode(tokenId: number): Promise<{ next: number; isEos: boolean; str: string | null }> {
        return this.rpc("stepAndDecode", { sid: this.session, tokenId });
    }
    encodeImage(pixels: Float32Array, h: number, w: number): Promise<Float32Array> {
        return this.rpc("encodeImage", { sid: this.session, pixels, h, w });
    }
    encodeAudio(pcm: Float32Array): Promise<Float32Array> {
        return this.rpc("encodeAudio", { sid: this.session, pcm });
    }
    /** Cooperative cancel for in-flight `encodeImage` / `encodeAudio`.
     *  The encode's promise rejects with a "cancelled" error on the
     *  next transformer-layer boundary. Safe to call when no encode
     *  is running. Intentionally session-less so it can fire while the
     *  encode itself holds the session lock. */
    cancelMultimodalEncode(): Promise<boolean> {
        return this.rpc("cancelMultimodalEncode", {});
    }
    /** In-engine speech-to-text via the audio tower + greedy decode.
     *  Streams per-token deltas to `onChunk` if provided; resolves with
     *  the final transcript. Worker forces greedy sampling (temperature
     *  0) for the duration of the call regardless of the chat-side
     *  sampling settings.
     *
     *  Acquires its own session lock for the duration of the call — the
     *  mic gesture lives outside the chat-send flow that normally owns
     *  the session, so we can't assume one's already held. The lock
     *  blocks any concurrent chat generation in another tab while
     *  transcribing (matches the existing single-tab-at-a-time
     *  arbitration). */
    async transcribeAudio(
        pcm: Float32Array,
        onChunk?: (delta: string) => void,
    ): Promise<string> {
        const off = onChunk
            ? this.subscribe("transcribeChunk", (p) => {
                if (p.done) return;
                const delta = String(p.delta ?? "");
                if (delta) onChunk(delta);
            })
            : null;
        try {
            return await this.withSession(async () => {
                const { transcript } = await this.rpc<{ transcript: string }>(
                    "transcribeAudio",
                    { sid: this.session, pcm } as Record<string, unknown>,
                );
                return transcript;
            });
        } finally {
            off?.();
        }
    }
    /** Drop vision tower weights from GPU memory. Returns approx bytes
     *  freed. Re-encoding an image re-uploads. Call between encode and
     *  prefill on memory-tight devices (iPhone) so the prefill step
     *  doesn't push Safari WebContent past jetsam. */
    releaseVisionWeights(): Promise<number> {
        return this.rpc("releaseVisionWeights", { sid: this.session });
    }
    releaseAudioWeights(): Promise<number> {
        return this.rpc("releaseAudioWeights", { sid: this.session });
    }

    // ── PWA update coordination (router-only, no session needed) ───────
    /** Tell the router that this tab's boot-time version check found a
     *  newer build. The router remembers it and broadcasts to every
     *  other open tab so they all surface the same banner. */
    broadcastUpdateAvailable(version: string): Promise<true> {
        return this.rpc("updateAvailable", { version });
    }
    /** Trigger the coordinated multi-tab shutdown + reload. Router
     *  broadcasts `applyingUpdate` to all tabs and signals the core
     *  worker to release its handles. Each tab reloads ~600 ms after
     *  receiving the broadcast. */
    applyUpdate(version: string): Promise<boolean> {
        return this.rpc("applyUpdate", { version });
    }

    reset(): Promise<void> { return this.rpc("reset", { sid: this.session }); }
    setSampling(opts: SamplingOptions): Promise<void> {
        return this.rpc("setSampling", { sid: this.session, opts });
    }

    // ── Suspend / resume ───────────────────────────────────────────────
    /** Snapshot the wasm Model's KV cache + sampler RNG + position into
     *  a single byte blob. Caller writes the result to OPFS for later
     *  restore. Session-locked because the snapshot reads from the same
     *  wgpu buffers that step() writes. */
    saveKvState(): Promise<Uint8Array> {
        return this.rpc("saveKvState", { sid: this.session });
    }
    /** Inverse of saveKvState. The wasm side validates the snapshot's
     *  layout_hash against the currently-loaded model; on mismatch the
     *  RPC rejects and caller falls back to token-replay rebuild. */
    restoreKvState(bytes: Uint8Array): Promise<boolean> {
        return this.rpc("restoreKvState", { sid: this.session, bytes });
    }
    /** Render a conversation for *continuation* — if the last message
     *  has role "model", its content renders without the trailing
     *  close marker so the model continues that response on the next
     *  step() rather than starting a new turn. Stateless (no session). */
    renderChatForContinuation(messages: unknown, withBos: boolean): Promise<string> {
        return this.rpc("renderChatForContinuation", { messages, withBos });
    }
    /** Current wasm Model.position. Stateless, mostly for diagnostics
     *  (e.g. confirming KV intactness after a backgrounding event). */
    position(): Promise<number> { return this.rpc("position", {}); }

    // ── ensureModel (download — coalesces across tabs in the worker) ───
    ensureModel(args: {
        url: string; modelKey: string; filename: string; expectedSize: number;
    }): Promise<{ totalBytes: number; fromCache: boolean }> {
        return this.rpc("ensureModel", args as Record<string, unknown>);
    }

    // ── Chat persistence ───────────────────────────────────────────────
    dbInit(): Promise<boolean> { return this.rpc("dbInit"); }
    convList(): Promise<ConversationRow[]> { return this.rpc("convList"); }
    convCreate(opts: { id?: string; title?: string; model?: string | null } = {}): Promise<ConversationRow> {
        return this.rpc("convCreate", opts as Record<string, unknown>);
    }
    convDelete(id: string): Promise<{ ok: boolean; opfsPaths: string[] }> {
        return this.rpc("convDelete", { id });
    }
    convRename(id: string, title: string): Promise<boolean> { return this.rpc("convRename", { id, title }); }
    convTouch(id: string, titleIfBlank?: string): Promise<boolean> {
        return this.rpc("convTouch", { id, titleIfBlank });
    }

    msgList(conversationId: string): Promise<MessageRow[]> {
        return this.rpc("msgList", { conversationId });
    }
    msgInsert(opts: { conversationId: string; messageId?: string; role: string; content?: string }): Promise<{ messageId: string; created_at: number }> {
        return this.rpc("msgInsert", opts as Record<string, unknown>);
    }
    msgAppend(conversationId: string, messageId: string, delta: string): Promise<boolean> {
        return this.rpc("msgAppend", { conversationId, messageId, delta });
    }
    msgSetContent(conversationId: string, messageId: string, content: string): Promise<boolean> {
        return this.rpc("msgSetContent", { conversationId, messageId, content });
    }
    msgInsertImage(opts: {
        conversationId: string;
        messageId:      string;
        seq:            number;
        width:          number;
        height:         number;
        opfsPath:       string;
    }): Promise<boolean> {
        return this.rpc("msgInsertImage", opts as Record<string, unknown>);
    }
    msgListImages(conversationId: string): Promise<MessageImageRow[]> {
        return this.rpc("msgListImages", { conversationId });
    }
    dbFlush(): Promise<boolean> { return this.rpc("dbFlush"); }

    // ── LoRA fine-tuning ───────────────────────────────────────────────
    /** Start a training session. Consumes the Model — chat-side RPCs
     *  will throw until `trainingFinish()` returns the Model. */
    trainingStart(args: {
        loraConfig: TrainingLoraConfig;
        hparams:    TrainingHyperparams;
        totalSteps?: number;
    }): Promise<TrainingSessionInfo> {
        return this.rpc("trainingStart", { sid: this.session, ...args } as Record<string, unknown>);
    }
    /** Run the scratch+LoRA allocation trial against a borrowed Model
     *  without consuming it. Returns `{ok, estimatedBytes, reason?}`;
     *  caller can use this to grey out Start if the device can't fit
     *  the requested config. The worker also runs this automatically
     *  inside `trainingStart` and refuses to consume the Model on
     *  probe failure, so explicit pre-flight is optional. */
    trainingProbeFit(args: {
        loraConfig: TrainingLoraConfig;
        hparams:    TrainingHyperparams;
    }): Promise<{ ok: boolean; estimatedBytes: number; reason?: string }> {
        return this.rpc("trainingProbeFit", { sid: this.session, ...args } as Record<string, unknown>);
    }
    trainingStep(args: {
        inputIds:  Uint32Array;
        targetId?: number;
        targets?:  Uint32Array;
        lossMode?: "next_token" | "per_position";
    }): Promise<TrainingStepReport> {
        return this.rpc("trainingStep", { sid: this.session, ...args } as Record<string, unknown>);
    }
    trainingZeroGrads(): Promise<boolean> {
        return this.rpc("trainingZeroGrads", { sid: this.session });
    }
    trainingForwardBackward(args: {
        inputIds:  Uint32Array;
        targetId?: number;
        targets?:  Uint32Array;
        lossMode?: "next_token" | "per_position";
    }): Promise<{ loss: number; step: number; lr: number }> {
        return this.rpc("trainingForwardBackward", { sid: this.session, ...args } as Record<string, unknown>);
    }
    trainingOptimizerStep(): Promise<{ step: number; lr: number }> {
        return this.rpc("trainingOptimizerStep", { sid: this.session });
    }
    trainingSaveAdapter(name: string): Promise<{ name: string; size: number }> {
        return this.rpc("trainingSaveAdapter", { sid: this.session, name });
    }
    /** Combined save + finish — atomic on the Rust side (one
     *  consume-self call). Use this for any flow that wants to save
     *  AND release the session in one shot; the two-call sequence
     *  `trainingSaveAdapter` → `trainingFinish` is broken by a
     *  wasm-bindgen async-borrow leak and should only be used when
     *  the session needs to stay alive after save. */
    trainingSaveAdapterAndFinish(name: string): Promise<{ name: string; size: number }> {
        return this.rpc("trainingSaveAdapterAndFinish", { sid: this.session, name });
    }
    trainingFinish(): Promise<boolean> {
        return this.rpc("trainingFinish", { sid: this.session });
    }
    /** Flip the cooperative cancel flag on the active training session
     *  so the in-flight step bails at the next per-layer encoder
     *  boundary. The pending `trainingStep` promise rejects with a
     *  "cancelled" error within ~one layer of latency. No-op when no
     *  session is active. */
    trainingCancel(): Promise<boolean> {
        return this.rpc("trainingCancel", { sid: this.session });
    }
    // Adapter-library ops: auto-acquire a session for the duration of
    // the call ONLY IF no session is already held. These mutate the
    // Model (apply / clear) or OPFS (delete), so the SharedWorker
    // router gates them under SESSION_REQUIRED. On a fresh page load
    // the user hasn't sent a chat turn yet, so `this.session` is null
    // and the bare RPC fails with "no active session". This wrapper
    // acquires the lock for the call's duration when needed, and is a
    // no-op when an outer session is already active (e.g. user
    // applying from the library mid-chat) — avoids deadlocking on
    // re-acquire of an already-held session.
    private async withSessionIfNone<T>(fn: () => Promise<T>): Promise<T> {
        if (this.session != null) return fn();
        return this.withSession(fn);
    }
    async trainingApplyAdapter(name: string): Promise<{ name: string; slots: number }> {
        return this.withSessionIfNone(async () =>
            this.rpc("trainingApplyAdapter", { sid: this.session, name }),
        );
    }
    async trainingClearAdapter(): Promise<boolean> {
        return this.withSessionIfNone(async () =>
            this.rpc("trainingClearAdapter", { sid: this.session }),
        );
    }
    async trainingDeleteAdapter(name: string): Promise<boolean> {
        return this.withSessionIfNone(async () =>
            this.rpc("trainingDeleteAdapter", { sid: this.session, name }),
        );
    }
    trainingListAdapters(): Promise<{ entries: AdapterListEntry[]; active: string | null }> {
        return this.rpc("trainingListAdapters", {});
    }
    trainingStatus(): Promise<TrainingStatusInfo> {
        return this.rpc("trainingStatus", {});
    }
}

/** Log level for {@link WorkerClient.logs.append}. Mirrors
 *  `LogLevel` in `workers/opfs_logger.ts`. */
export type LogLevel = "info" | "warn" | "error";

/** Session metadata returned by {@link WorkerClient.logs.list}. */
export interface LogSessionMeta {
    id:        string;
    startMs:   number;
    sizeBytes: number;
    cleanExit: boolean;
}

export interface TrainingLoraConfig {
    rank:           number;
    alpha:          number;
    dropout:        number;
    target_modules: string[];
}
export interface TrainingHyperparams {
    epochs:                       number;
    batch_size:                   number;
    learning_rate:                number;
    warmup_steps:                 number;
    weight_decay:                 number;
    lr_scheduler:                 "constant" | "linear" | "cosine" | "cosine_warm_restarts";
    seed:                         number;
    max_seq_len:                  number;
    gradient_accumulation_steps:  number;
    max_grad_norm:                number;
    loss_mode:                    "next_token" | "per_position";
    gradient_checkpointing:       boolean;
    mixed_precision:              boolean;
    /** Truncated backward: only train layers >= this index. 0 (default)
     *  means "backprop every layer" — the standard training path.
     *  Larger values progressively narrow the trainable region to just
     *  the top `n_layers - backward_layer_floor` layers, trading
     *  adapter expressiveness for backward memory + compute savings.
     *  Auto-applied by the Memory-tight preset on iPhone-class
     *  devices; manually editable via AdvancedCard's "Trainable
     *  depth" slider. */
    backward_layer_floor?:        number;
}
export interface TrainingStepReport {
    loss: number;
    lr:   number;
    step: number;
}
export interface TrainingSessionInfo {
    parameterCount:        number;
    gradientCheckpointing: boolean;
    mixedPrecision:        boolean;
    /** GPU bytes the trainer is expected to occupy — surfaced by the
     *  probe that runs at the top of `trainingStart`. */
    estimatedBytes?:       number;
}
export type TrainingStatusInfo =
    | { active: false }
    | {
        active: true;
        step: number;
        lr: number;
        parameterCount: number;
        gradientCheckpointing: boolean;
        mixedPrecision: boolean;
    };
export interface AdapterListEntry {
    name:         string;
    size:         number;
    lastModified: number;
}

export interface ConversationRow {
    id:         string;
    title:      string;
    model:      string | null;
    created_at: number;
    updated_at: number;
}
export interface MessageRow {
    conversation_id: string;
    message_id:      string;
    role:            string;
    content:         string;
    created_at:      number;
}
export interface MessageImageRow {
    conversation_id: string;
    message_id:      string;
    seq:             number;
    width:           number;
    height:          number;
    opfs_path:       string;
}

/** Singleton — one client per page, but the SharedWorker is one per origin. */
let _client: WorkerClient | null = null;
export function getClient(): WorkerClient {
    if (!_client) _client = new WorkerClient();
    return _client;
}
