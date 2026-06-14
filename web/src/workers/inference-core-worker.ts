// rullama inference *core* — Dedicated Worker.
//
// Spawned by the host tab's main thread (not by the SharedWorker —
// SharedWorkerGlobalScope doesn't expose the `Worker` constructor in
// any browser). The host tab creates a MessageChannel, posts one port
// to this worker via {type:'attach', port}, and posts the other port
// to the SharedWorker router via the `attachCore` RPC. Once attached,
// all RPC traffic and notifications travel on that port. The wasm
// Model, OPFS sync handles, and rsqlite-wasm DB all live here because
// `createSyncAccessHandle` is restricted to DedicatedWorkerGlobalScope.
//
// Session arbitration, port table, and notification fanout live in
// the SharedWorker router. This worker is single-tenant: one caller
// (the router), one request at a time per stateful RPC.

// @ts-expect-error — generated bundle, no .d.ts
import init, { Model, TrainingSession, probeTrainingFit, WasmDatabase, EmbeddingModel, DiffusionGemma, gpuMemBreakdown, gpuMemTotalMib } from "/pkg/rullama.js";

const gpuMemBreakdownFn = gpuMemBreakdown as unknown as () => string;
const gpuMemTotalFn = gpuMemTotalMib as unknown as () => number;
import * as opfsLogger from "./opfs_logger";
import type { LogLevel } from "./opfs_logger";

interface ProbeReport {
    ok:             boolean;
    estimatedBytes: number;
    reason?:        string;
}
type ProbeFn = (
    model:          ModelHandle,
    loraConfigJson: string,
    hparamsJson:    string,
) => Promise<ProbeReport>;
const probeFit = probeTrainingFit as unknown as ProbeFn;

interface WasmDbHandle {
    exec(sql: string): bigint;
    execParams(sql: string, params: unknown[]): bigint;
    query(sql: string): unknown[];
    queryParams(sql: string, params: unknown[]): unknown[];
    queryOne(sql: string): unknown | null;
    execMany(sql: string): void;
    flush(): void;
    close(): void;
    free?(): void;
}
interface WasmDbStatic {
    openWithOpfs(name: string, chunkSize?: bigint | null, maxShards?: number | null): Promise<WasmDbHandle>;
    openInMemory(): WasmDbHandle;
}
const Db = WasmDatabase as unknown as WasmDbStatic;

interface ModelHandle {
    free?(): void;
    vocabSize: number;
    hasVision: boolean;
    hasAudio: boolean;
    hasAdapter: boolean;
    imageSentinelIds(): [number, number] | null | undefined;
    audioSentinelIds(): [number, number] | null | undefined;
    encode(text: string): Uint32Array;
    step(tokenId: number): Promise<number>;
    stepWithEmbedding(embedding: Float32Array): Promise<number>;
    tokenStr(id: number): string | null | undefined;
    isEos(id: number): boolean;
    reset(): void;
    setSampling(opts: unknown): void;
    renderChat(messages: unknown, withBos: boolean): string;
    imageSoftTokenCount(h: number, w: number): number;
    decodeWav(bytes: Uint8Array): Float32Array;
    encodeImage(
        pixels: Float32Array, h: number, w: number,
        progressCb?: ((layer: number, total: number) => void) | null,
    ): Promise<Float32Array>;
    encodeAudio(pcm: Float32Array): Promise<Float32Array>;
    cancelMultimodalEncode(): void;
    /** Drop the vision tower's GPU weight buffers + cached tensors.
     *  Re-uploads on the next `encodeImage` (free if the GGUF blob is
     *  still in OPFS, slow if it needs re-fetch). Returns approx
     *  bytes freed. Called between encode and prefill on iPhone to
     *  fit under the WebContent jetsam cap. */
    releaseVisionWeights(): number;
    releaseAudioWeights(): number;
    /** Re-allocate the per-layer KV cache at a new token capacity.
     *  Returns the previous max_context so JS can restore later.
     *  Discards cached KV content. Used by `trainingStart` /
     *  `trainingFinish` to shrink chat's KV during a training run. */
    shrinkKv(newMaxContext: number): number;
    /** Current KV cache capacity in tokens. Read before `shrinkKv`
     *  to remember what to restore. */
    readonly maxContext: number;
    /** Total bytes resident in the shared GPU WeightCache. Useful as
     *  a coarse memory-pressure signal at phase boundaries
     *  (encode → release → prefill) so a jetsam kill can be localised
     *  to the right phase on next run. */
    readonly cachedWeightBytes: bigint;
    saveKvState(): Promise<Uint8Array>;
    restoreKvState(bytes: Uint8Array): void;
    renderChatForContinuation(messages: unknown, withBos: boolean): string;
    readonly position: number;
    loadAdapter(bytes: Uint8Array): number;
    clearAdapter(): void;
}

// Wasm-bindgen `TrainingSession` (from rullama-finetune). All async
// methods return a Promise. The session *consumes* the Model on
// construction (move semantics from JS' perspective): no Model RPC
// can run while training is live. `finish()` returns the Model back
// to JS so chat can resume against the same loaded weights.
interface TrainingSessionHandle {
    free?(): void;
    readonly stepNum: number;
    readonly lr: number;
    readonly parameterCount: number;
    readonly gradientCheckpointing: boolean;
    readonly mixedPrecision: boolean;
    readonly cachedWeightBytes: number;
    step(
        inputIds: Uint32Array,
        targetId: number,
        progressCb?: (phase: string, current: number, total: number) => void,
    ): Promise<{ loss: number; lr: number; step: number }>;
    stepPerPosition(
        inputIds: Uint32Array,
        targets: Uint32Array,
        progressCb?: (phase: string, current: number, total: number) => void,
    ): Promise<{ loss: number; lr: number; step: number }>;
    zeroGrads(): void;
    forwardBackward(inputIds: Uint32Array, targetId: number): Promise<number>;
    forwardBackwardPerPosition(inputIds: Uint32Array, targets: Uint32Array): Promise<number>;
    optimizerStep(): void;
    saveAdapter(): Promise<Uint8Array>;
    saveAdapterAndFinish(): Promise<SaveAndFinishResultHandle>;
    setLrSchedule(totalSteps: number): void;
    finish(): ModelHandle;
    cancel(): void;
}
// Mirrors the wasm-bindgen `SaveAndFinishResult` returned by the
// combined save+finish call. `bytes` getter consumes on read;
// `takeModel()` consumes on first call. Both throw on second access.
interface SaveAndFinishResultHandle {
    free?(): void;
    readonly bytes: Uint8Array;
    takeModel(): ModelHandle;
}
interface TrainingSessionStatic {
    new(model: ModelHandle, loraConfigJson: string, hparamsJson: string): TrainingSessionHandle;
}
const TrainingSessionClass = TrainingSession as unknown as TrainingSessionStatic;
interface ModelStatic {
    loadFromOpfsTextOnly(
        readFn: (offset: number, length: number) => Uint8Array | Promise<Uint8Array>,
        totalBytes: number,
        maxContext: number,
    ): Promise<ModelHandle>;
    loadFromOpfs(
        readFn: (offset: number, length: number) => Uint8Array | Promise<Uint8Array>,
        totalBytes: number,
        maxContext: number,
    ): Promise<ModelHandle>;
}
const ModelClass = Model as unknown as ModelStatic;

// ───────────────────────────────────────────────────────────────────────
// State
// ───────────────────────────────────────────────────────────────────────

const OPFS_DIR = "rullama-models";
const ADAPTERS_DIR = "rullama-adapters";
const DB_NAME  = "rullama-chat.db";

let wasmReady: Promise<unknown> | null = null;
let model: ModelHandle | null = null;
let syncHandle: FileSystemSyncAccessHandle | null = null;
let dbReady: Promise<WasmDbHandle> | null = null;
// Embedding model (EmbeddingGemma) — loaded concurrently with the chat
// model, owns its own wasm handle. Stateless across embed() calls.
interface EmbeddingModelHandle {
    embed(text: string, targetDim: number): Promise<Float32Array>;
    embedBatch(texts: string[], targetDim: number): Promise<Float32Array>;
    readonly dim: number;
    free?(): void;
}
let embedder: EmbeddingModelHandle | null = null;
let embedderInfo: { name: string; dim: number } | null = null;
// Dedicated OPFS sync handle for the embedder GGUF (kept open for the
// model's lifetime — the streaming TensorFetcher reads tensors through
// it on every embed). Separate from the chat model's `syncHandle`.
let embedderSyncHandle: FileSystemSyncAccessHandle | null = null;
const EmbeddingModelClass = EmbeddingModel as unknown as {
    load(bytes: Uint8Array): Promise<EmbeddingModelHandle>;
    loadFromOpfs(
        readFn: (offset: number, length: number) => Uint8Array,
        totalBytes: number,
    ): Promise<EmbeddingModelHandle>;
};

// ── DiffusionGemma (block-diffusion engine, `diffusion-gemma` family) ──────
// A SECOND non-Model wasm class, like the embedder: it owns its own handle +
// OPFS sync reader. The JS worker drives the denoise loop one step at a time
// (`denoiseStep`) so the UI can render the canvas condensing out of noise.
interface DiffusionGemmaHandle {
    startGenerate(prompt: string, canvasLen: number, maxSteps: number, seed: number): void;
    denoiseStep(): Promise<string>;
    readonly done: boolean;
    readonly stepIndex: number;
    readonly totalSteps: number;
    readonly accepted: number;
    readonly meanEntropy: number;
    readonly canvasLen: number;
    free?(): void;
}
let diffuser: DiffusionGemmaHandle | null = null;
let diffuserInfo: { name: string; canvasLen: number } | null = null;
// Dedicated OPFS sync handle for the diffusion GGUF (separate from the chat
// model's `syncHandle` and the embedder's).
let diffuserSyncHandle: FileSystemSyncAccessHandle | null = null;
const DiffusionGemmaClass = DiffusionGemma as unknown as {
    loadFromOpfs(
        readFn: (offset: number, length: number) => Uint8Array,
        totalBytes: number,
    ): Promise<DiffusionGemmaHandle>;
};

/** Open a sync read handle over an OPFS-cached diffusion GGUF + a readFn for
 *  the streaming TensorFetcher. Mirrors `openEmbedderSyncReadFn`. */
async function openDiffuserSyncReadFn(modelKey: string, filename: string) {
    if (diffuserSyncHandle) {
        try { diffuserSyncHandle.close(); } catch { /* */ }
        diffuserSyncHandle = null;
    }
    const root     = await navigator.storage.getDirectory();
    const dlDir    = await root.getDirectoryHandle(OPFS_DIR, { create: false });
    const modelDir = await dlDir.getDirectoryHandle(modelKey, { create: false });
    const fh       = await modelDir.getFileHandle(filename, { create: false });
    diffuserSyncHandle = await createSyncAccessHandleWithRetry(
        fh, `${modelKey}/${filename} (diffusion read)`, { notifyKind: "diffuserLoadWaiting" },
    );
    const size = diffuserSyncHandle.getSize();
    if (size === 0) {
        try { diffuserSyncHandle.close(); } catch { /* */ }
        diffuserSyncHandle = null;
        throw new Error(`OPFS diffusion file ${modelKey}/${filename} is empty`);
    }
    const handle = diffuserSyncHandle;
    const readFn = (offset: number, length: number): Uint8Array => {
        const buf = new Uint8Array(length);
        handle.read(buf, { at: offset });
        return buf;
    };
    return { readFn, totalBytes: size };
}

/** Open a sync read handle over an OPFS-cached embedder GGUF + a readFn for
 *  the streaming TensorFetcher. Mirrors `openSyncReadFn` but uses the
 *  embedder's own handle so it doesn't evict the chat model's. */
async function openEmbedderSyncReadFn(modelKey: string, filename: string) {
    if (embedderSyncHandle) {
        try { embedderSyncHandle.close(); } catch { /* */ }
        embedderSyncHandle = null;
    }
    const root     = await navigator.storage.getDirectory();
    const dlDir    = await root.getDirectoryHandle(OPFS_DIR, { create: false });
    const modelDir = await dlDir.getDirectoryHandle(modelKey, { create: false });
    const fh       = await modelDir.getFileHandle(filename, { create: false });
    embedderSyncHandle = await createSyncAccessHandleWithRetry(
        fh, `${modelKey}/${filename} (embed read)`, { notifyKind: "embedderLoadWaiting" },
    );
    const size = embedderSyncHandle.getSize();
    if (size === 0) {
        try { embedderSyncHandle.close(); } catch { /* */ }
        embedderSyncHandle = null;
        throw new Error(`OPFS embedder file ${modelKey}/${filename} is empty`);
    }
    const handle = embedderSyncHandle;
    const readFn = (offset: number, length: number): Uint8Array => {
        const buf = new Uint8Array(length);
        handle.read(buf, { at: offset });
        return buf;
    };
    return { readFn, totalBytes: size };
}
// When non-null, a training session owns the Model. All Model-mutating
// RPCs (step, encodeImage, reset, etc.) refuse to run; the chat UI
// gates this at the surface level.
let trainingSession: TrainingSessionHandle | null = null;
/** Name of the adapter currently loaded into Model, if any. */
let activeAdapterName: string | null = null;

/** KV cache capacity snapshot taken just before `trainingStart` shrinks
 *  it. `trainingFinish` reads this and restores the model back to the
 *  chat-sized KV. Null while no training session is active or pending. */
let trainingOriginalKvContext: number | null = null;

interface LoadedModelInfo {
    name: string | null;
    modelKey: string;
    filename: string;
    hasVision: boolean;
    hasAudio: boolean;
    vocabSize: number;
    imageSentinelIds: [number, number] | null;
    audioSentinelIds: [number, number] | null;
}
let loadedInfo: LoadedModelInfo | null = null;

interface DownloadInflight {
    promise: Promise<{ totalBytes: number; fromCache: boolean }>;
}
const inflight = new Map<string, DownloadInflight>();

const RPC_TRACE = false;

// ───────────────────────────────────────────────────────────────────────
// Logging + notification (postMessage to the SharedWorker router via the
// attached MessagePort)
// ───────────────────────────────────────────────────────────────────────

let routerPort: MessagePort | null = null;

/** Flag set the moment a `{type:"shutdown"}` message is received. While
 *  true, every RPC (including `pingCore`) replies `ok: false` so the
 *  router doesn't false-positive a dying worker as alive.
 *
 *  Why this matters: on iOS Safari the router's `verifyCoreLive` ping
 *  on a new tab's connect can race the old host's `disconnect`. If the
 *  ping wins the race and the old worker is in the middle of
 *  `releaseAllHandles()`, `pingCore` (a trivial `() => true`) would
 *  return success — the new tab inherits a stale `corePort`, the
 *  worker self.close()s a beat later, and the new tab's first RPC
 *  vanishes into the closed port. Setting this flag at the top of the
 *  shutdown handler closes that race: the ping returns `ok: false`
 *  immediately, the router treats the core as dead, and re-election
 *  fires cleanly. */
let shuttingDown = false;

/** Unique id for this worker's lifetime. Used as the OPFS log file
 *  name + as the localStorage clean-exit marker key. Generated once
 *  at module load; never reused. */
const SESSION_ID: string = (() => {
    const iso = new Date().toISOString().replace(/[:.]/g, "-");
    const rand = Math.random().toString(36).slice(2, 8);
    return `${iso}-${rand}`;
})();

/** True once `opfsLogger.init(SESSION_ID)` has resolved. Lazy: we
 *  defer the init until the first `attach` so a worker that loses
 *  the SharedWorker election (and gets `self.close()`d) doesn't
 *  burn OPFS budget. */
let loggerInited = false;

function log(...args: unknown[]) {
    const argStrs = args.map((a) => String(a));
    const msg = argStrs.join(" ");
    if (routerPort) {
        try { routerPort.postMessage({ type: "log", args: argStrs }); } catch { /* */ }
    }
    try {
        fetch("/api/log", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tag: "wkr", msg, ts: Date.now() }),
            keepalive: true,
        }).catch(() => {});
    } catch { /* */ }
    try { opfsLogger.append("info", "wkr", msg); } catch { /* */ }
}

/** Explicit-level / explicit-tag beacon variant. Use for the `[trn]`
 *  training-instrumentation beacons that need to be readable by
 *  level (error vs info) in the post-crash viewer. Behaves like
 *  `log()` otherwise — fans out to routerPort + /api/log + OPFS. */
function logBeacon(level: LogLevel, tag: string, msg: string) {
    if (routerPort) {
        try { routerPort.postMessage({ type: "log", args: [`[${tag}] ${msg}`] }); } catch { /* */ }
    }
    try {
        fetch("/api/log", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tag, msg, ts: Date.now() }),
            keepalive: true,
        }).catch(() => {});
    } catch { /* */ }
    try { opfsLogger.append(level, tag, msg); } catch { /* */ }
}

function notify(kind: string, payload: Record<string, unknown> = {}) {
    if (!routerPort) return;
    try { routerPort.postMessage({ type: "notify", kind, ...payload }); } catch { /* */ }
}

/** Open `FileSystemSyncAccessHandle` with backoff retry.
 *
 *  iOS Safari can take several seconds to GC a previous worker that's
 *  still holding an exclusive lock — this is the root of every
 *  "PWA-update reload can't open the OPFS file" report. Retry instead
 *  of failing on the first attempt, and notify the UI each pass so it
 *  can show "waiting for previous session to release …" instead of
 *  looking frozen.
 *
 *  Total budget defaults to 15 s, well past the empirically-observed
 *  iOS GC window for an orphaned worker handle. After the budget,
 *  throw with a message that says the data is intact and instructs
 *  the user to force-quit + reopen. */
async function createSyncAccessHandleWithRetry(
    fh: FileSystemFileHandle,
    label: string,
    opts?: { budgetMs?: number; notifyKind?: string },
): Promise<FileSystemSyncAccessHandle> {
    const budget = opts?.budgetMs ?? 15_000;
    const notifyKind = opts?.notifyKind ?? "syncHandleWaiting";
    const startMs = Date.now();
    let attempt = 0;
    while (true) {
        try {
            const h = await fh.createSyncAccessHandle();
            if (attempt > 0) {
                log(`opfs: ${label} syncHandle acquired after ${attempt} retries (${Date.now() - startMs}ms)`);
            }
            return h;
        } catch (e) {
            attempt += 1;
            const elapsed = Date.now() - startMs;
            if (elapsed >= budget) {
                throw new Error(
                    `${label}: syncHandle locked by a previous worker ` +
                    `(${attempt} attempts over ${elapsed}ms). The data is intact — ` +
                    `force-quit the tab/app and reopen to release the lock. ` +
                    `Underlying: ${(e as Error)?.message ?? e}`,
                );
            }
            const delay = Math.min(1500, 100 * Math.pow(2, attempt - 1));
            notify(notifyKind, {
                reason: "opfs-locked",
                label,
                attempt,
                elapsedMs: elapsed,
                nextDelayMs: delay,
            });
            log(`opfs: ${label} still locked (attempt ${attempt}, ${elapsed}ms elapsed) — retrying in ${delay}ms`);
            await new Promise((r) => setTimeout(r, delay));
        }
    }
}

// ───────────────────────────────────────────────────────────────────────
// Wasm + DB lifecycle
// ───────────────────────────────────────────────────────────────────────

async function ensureWasm() {
    if (!wasmReady) wasmReady = init();
    return wasmReady;
}

const SCHEMA = [
    `CREATE TABLE IF NOT EXISTS conversations (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL DEFAULT 'New chat',
        model       TEXT,
        created_at  INTEGER NOT NULL,
        updated_at  INTEGER NOT NULL
     )`,
    `CREATE INDEX IF NOT EXISTS conv_updated_idx ON conversations(updated_at DESC)`,
    `CREATE TABLE IF NOT EXISTS messages (
        conversation_id TEXT NOT NULL,
        message_id      TEXT NOT NULL,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL DEFAULT '',
        created_at      INTEGER NOT NULL,
        PRIMARY KEY (conversation_id, message_id),
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
     )`,
    `CREATE INDEX IF NOT EXISTS msg_conv_idx ON messages(conversation_id, created_at)`,
    `DROP TABLE IF EXISTS message_images`,
    `CREATE TABLE IF NOT EXISTS message_images (
        conversation_id TEXT NOT NULL,
        message_id      TEXT NOT NULL,
        seq             INTEGER NOT NULL,
        width           INTEGER NOT NULL,
        height          INTEGER NOT NULL,
        opfs_path       TEXT NOT NULL,
        PRIMARY KEY (conversation_id, message_id, seq),
        FOREIGN KEY (conversation_id, message_id)
            REFERENCES messages(conversation_id, message_id)
            ON DELETE CASCADE
     )`,
    `CREATE INDEX IF NOT EXISTS msg_img_conv_idx
        ON message_images(conversation_id, message_id, seq)`,
    // ---- embeddings / RAG (Knowledge tab) ----
    // A `document` is an indexed source (uploaded file, pasted text, or a
    // chat message). `conversation_id` NULL ⇒ global (matches every
    // conversation's search). Chunks store the f32 vector as a little-endian
    // BLOB (rsqlite-wasm's vec_distance_cosine consumes that layout).
    `CREATE TABLE IF NOT EXISTS documents (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        name                    TEXT NOT NULL,
        source_kind             TEXT NOT NULL,
        byte_size               INTEGER NOT NULL DEFAULT 0,
        created_at              INTEGER NOT NULL,
        conversation_id         TEXT,
        embedding_model         TEXT NOT NULL DEFAULT '',
        vector_dim              INTEGER NOT NULL DEFAULT 0
     )`,
    `CREATE INDEX IF NOT EXISTS doc_conv_idx ON documents(conversation_id)`,
    `CREATE TABLE IF NOT EXISTS chunks (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id             INTEGER NOT NULL,
        chunk_idx               INTEGER NOT NULL,
        text                    TEXT NOT NULL,
        page                    INTEGER,
        vector                  BLOB NOT NULL,
        vector_dim              INTEGER NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
     )`,
    `CREATE INDEX IF NOT EXISTS chunk_doc_idx ON chunks(document_id)`,
    // Per-conversation RAG toggle. Stored as its own table so adding it
    // doesn't require an ALTER on `conversations`.
    `CREATE TABLE IF NOT EXISTS conversation_rag (
        conversation_id         TEXT PRIMARY KEY,
        enabled                 INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
     )`,
];

async function ensureDb(): Promise<WasmDbHandle> {
    if (!dbReady) {
        dbReady = (async () => {
            await ensureWasm();
            const db = await Db.openWithOpfs(DB_NAME);
            try { db.exec("PRAGMA foreign_keys = ON"); } catch { /* */ }
            for (const stmt of SCHEMA) db.exec(stmt);
            db.flush();
            log(`db: opened ${DB_NAME}`);
            return db;
        })();
    }
    return dbReady;
}

function newId(): string { return crypto.randomUUID(); }

// ───────────────────────────────────────────────────────────────────────
// Model lifecycle (sync-OPFS reader + Model.loadFromOpfs)
// ───────────────────────────────────────────────────────────────────────

async function openSyncReadFn(modelKey: string, filename: string) {
    if (syncHandle) {
        try { syncHandle.close(); } catch { /* */ }
        syncHandle = null;
    }
    const root     = await navigator.storage.getDirectory();
    const dlDir    = await root.getDirectoryHandle(OPFS_DIR, { create: false });
    const modelDir = await dlDir.getDirectoryHandle(modelKey, { create: false });
    const fh       = await modelDir.getFileHandle(filename,    { create: false });
    syncHandle = await createSyncAccessHandleWithRetry(
        fh,
        `${modelKey}/${filename} (read)`,
        { notifyKind: "modelLoadWaiting" },
    );

    const size = syncHandle.getSize();
    if (size === 0) {
        try { syncHandle.close(); } catch { /* */ }
        syncHandle = null;
        throw new Error(`OPFS file ${modelKey}/${filename} is empty`);
    }
    const handle = syncHandle;
    const readFn = (offset: number, length: number): Uint8Array => {
        const buf = new Uint8Array(length);
        handle.read(buf, { at: offset });
        return buf;
    };
    return { readFn, totalBytes: size };
}

interface LoadArgs {
    modelKey:    string;
    filename:    string;
    name?:       string;
    maxContext?: number;
    textOnly?:   boolean;
}

async function handleLoad(args: LoadArgs): Promise<LoadedModelInfo> {
    await ensureWasm();
    // **Refuse to load while training is active.** The TrainingSession
    // owns the Model + holds the OPFS syncHandle; trying to load a
    // new model would close the syncHandle out from under it (or get
    // stuck in the createSyncAccessHandle retry budget waiting for
    // a lock that won't release until trainingFinish runs). Surface
    // the situation explicitly so the user goes back to the Fine-tune
    // tab and applies/discards their adapter first.
    if (trainingSession) {
        throw new Error(
            "training session is active — apply / save / discard the adapter " +
            "from the Fine-tune tab first, then try loading again",
        );
    }
    if (model && loadedInfo
        && loadedInfo.modelKey === args.modelKey
        && loadedInfo.filename === args.filename) {
        log(`load: short-circuit, already have ${args.modelKey}/${args.filename}`);
        return loadedInfo;
    }
    if (model) {
        try { model.free?.(); } catch { /* */ }
        model = null;
        loadedInfo = null;
        notify("modelFreed", {});
    }
    notify("modelLoading", { modelKey: args.modelKey, filename: args.filename });
    const { readFn, totalBytes: size } = await openSyncReadFn(args.modelKey, args.filename);
    log(`load: Model.loadFromOpfs${args.textOnly ? "TextOnly" : ""} size=${size} max_ctx=${args.maxContext || "default"}`);

    model = args.textOnly
        ? await ModelClass.loadFromOpfsTextOnly(readFn, size, args.maxContext ?? 0)
        : await ModelClass.loadFromOpfs(readFn, size, args.maxContext ?? 0);

    log(`load: ready vocabSize=${model.vocabSize}`);
    loadedInfo = {
        name:             args.name ?? null,
        modelKey:         args.modelKey,
        filename:         args.filename,
        vocabSize:        model.vocabSize,
        hasVision:        model.hasVision,
        hasAudio:         model.hasAudio,
        imageSentinelIds: (model.imageSentinelIds() ?? null) as [number, number] | null,
        audioSentinelIds: (model.audioSentinelIds() ?? null) as [number, number] | null,
    };
    notify("modelLoaded", loadedInfo as unknown as Record<string, unknown>);
    return loadedInfo;
}

function requireModel(): ModelHandle {
    // Order matters: when training is active, `model` is null AND
    // `trainingSession` is set. Surface the specific "owned by
    // training" message first so the caller knows the model is alive
    // but loaned out, not gone.
    if (trainingSession) throw new Error(
        "model is owned by an active training session — call trainingFinish() first");
    if (!model) throw new Error("no model loaded — call load() first");
    return model;
}

function requireTraining(): TrainingSessionHandle {
    if (!trainingSession) throw new Error("no training session active — call trainingStart() first");
    return trainingSession;
}

async function getAdaptersDir(create: boolean): Promise<FileSystemDirectoryHandle> {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(ADAPTERS_DIR, { create });
}

async function writeAdapterBytes(name: string, bytes: Uint8Array): Promise<number> {
    const dir = await getAdaptersDir(true);
    const fh = await dir.getFileHandle(`${name}.bin`, { create: true });
    const handle = await createSyncAccessHandleWithRetry(fh, `adapters/${name}.bin`);
    try {
        handle.truncate(0);
        handle.write(bytes, { at: 0 });
        handle.flush();
        return handle.getSize();
    } finally {
        try { handle.close(); } catch { /* */ }
    }
}

async function readAdapterBytes(name: string): Promise<Uint8Array> {
    const dir = await getAdaptersDir(false);
    const fh = await dir.getFileHandle(`${name}.bin`, { create: false });
    const f = await fh.getFile();
    return new Uint8Array(await f.arrayBuffer());
}

async function listAdapterEntries(): Promise<Array<{name: string; size: number; lastModified: number}>> {
    try {
        const dir = await getAdaptersDir(false);
        const out: Array<{name: string; size: number; lastModified: number}> = [];
        const anyDir = dir as unknown as {
            entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
        };
        for await (const [name, h] of anyDir.entries()) {
            if (h.kind !== "file" || !name.endsWith(".bin")) continue;
            try {
                const f = await (h as FileSystemFileHandle).getFile();
                out.push({
                    name: name.replace(/\.bin$/, ""),
                    size: f.size,
                    lastModified: f.lastModified,
                });
            } catch { /* */ }
        }
        out.sort((a, b) => b.lastModified - a.lastModified);
        return out;
    } catch {
        return [];
    }
}

// ───────────────────────────────────────────────────────────────────────
// ensureModel — download GGUF to OPFS via streaming write
// ───────────────────────────────────────────────────────────────────────

const FLUSH_INTERVAL = 64 * 1024 * 1024;

async function existingOpfsSize(modelKey: string, filename: string): Promise<number> {
    try {
        const root  = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        const md    = await dlDir.getDirectoryHandle(modelKey, { create: false });
        const fh    = await md.getFileHandle(filename, { create: false });
        const f     = await fh.getFile();
        if (f.size < 4) return 0;
        // First-4-bytes magic check. On iOS Safari this can THROW transiently
        // when a previous core worker's FileSystemSyncAccessHandle hasn't
        // been GC'd yet (post-PWA-update reload race). The old behaviour
        // ("any throw → return 0") made doDownload start a fresh download
        // from byte 0, which then ALSO failed at createSyncAccessHandle
        // because the old handle was still held. Net effect on the user:
        // "the app keeps trying to redownload my 7 GB model after every
        // deploy." Treat the read failure as "size unknown but the file
        // *exists* with f.size bytes" — return f.size so downstream sees
        // the file as potentially-complete and short-circuits if it
        // matches expectedSize.
        let magicGood = false;
        let magicReadable = false;
        try {
            const head = new Uint8Array(await f.slice(0, 4).arrayBuffer());
            magicReadable = head.length === 4;
            magicGood = magicReadable
                && head[0] === 0x47 && head[1] === 0x47
                && head[2] === 0x55 && head[3] === 0x46;
        } catch { /* magicReadable stays false — likely sync-handle race */ }
        if (magicGood) return f.size;
        if (magicReadable) {
            // Read succeeded but bytes don't match: real corruption (Jetsam-
            // truncated zero prefix). Caller will detect via size mismatch
            // or downstream load failure; don't auto-delete from here.
            log(`opfs: ${modelKey}/${filename} bytes don't match GGUF magic (size=${f.size})`);
            return 0;
        }
        // Read failed — preserve the file by returning its size.
        log(`opfs: ${modelKey}/${filename} first-bytes read failed (likely sync-handle race) — preserving file at size=${f.size}`);
        return f.size;
    } catch { return 0; }
}

interface EnsureArgs {
    url:          string;
    modelKey:     string;
    filename:     string;
    expectedSize: number;
}

async function doDownload(args: EnsureArgs): Promise<{ totalBytes: number; fromCache: boolean }> {
    const { url, modelKey, filename, expectedSize } = args;

    const have = await existingOpfsSize(modelKey, filename);
    if (have > 0 && expectedSize > 0 && have >= expectedSize) {
        notify("downloadDone", { modelKey, filename, totalBytes: have, fromCache: true });
        return { totalBytes: have, fromCache: true };
    }

    let currentOffset = have;
    const root     = await navigator.storage.getDirectory();
    const dlDir    = await root.getDirectoryHandle(OPFS_DIR, { create: true });
    const modelDir = await dlDir.getDirectoryHandle(modelKey, { create: true });
    const fileHandle = await modelDir.getFileHandle(filename, { create: true });

    let writeHandle: FileSystemSyncAccessHandle | null = null;
    let bytesSinceFlush = 0;
    try {
        // Retry-acquire the WRITE syncHandle. After a screen-lock-induced
        // Jetsam kill, the previous worker's exclusive lock may not have
        // GC'd yet — without the retry the resume fails on first attempt
        // even though the data is intact.
        writeHandle = await createSyncAccessHandleWithRetry(
            fileHandle,
            `${modelKey}/${filename} (write)`,
            { notifyKind: "downloadWaiting" },
        );
        writeHandle.truncate(currentOffset);
        writeHandle.flush();

        let totalBytes = 0;

        // Outer fetch-retry loop: if the network stream dies mid-download
        // (iOS Safari severs sockets during screen lock; tab thaws but
        // the reader.read() promise rejects), refetch with
        // `Range: bytes=currentOffset-` and continue. Without this loop
        // a single dropped connection ends the whole download and the
        // user has to reload the page to recover.
        const MAX_FETCH_RETRIES = 5;
        let fetchAttempt = 0;
        while (true) {
            const headers: Record<string, string> = {};
            if (currentOffset > 0) headers["Range"] = `bytes=${currentOffset}-`;
            const resp = await fetch(url, { headers });

            if (resp.status === 416) {
                // Server says we already have everything (Range past EOF).
                const size = writeHandle.getSize();
                notify("downloadDone", { modelKey, filename, totalBytes: size, fromCache: true });
                writeHandle.close();
                writeHandle = null;
                return { totalBytes: size, fromCache: false };
            }
            if (!resp.ok && resp.status !== 206) {
                throw new Error(`fetch failed (${resp.status})`);
            }
            if (resp.status === 200 && currentOffset > 0) {
                // Server doesn't honor Range — restart from byte 0.
                writeHandle.truncate(0);
                currentOffset = 0;
            }

            const cr = resp.headers.get("content-range");
            const contentLength = Number(resp.headers.get("content-length") || "0") || 0;
            totalBytes = cr?.match(/\/(\d+)\s*$/)
                ? Number(cr.match(/\/(\d+)\s*$/)![1])
                : currentOffset + contentLength;

            if (!resp.body) {
                throw new Error("no response body");
            }
            const reader = resp.body.getReader();

            try {
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    if (!value) continue;

                    const written = writeHandle.write(value, { at: currentOffset });
                    currentOffset   += written;
                    bytesSinceFlush += written;

                    if (bytesSinceFlush >= FLUSH_INTERVAL) {
                        writeHandle.flush();
                        bytesSinceFlush = 0;
                    }
                    notify("downloadProgress", {
                        modelKey, filename,
                        bytesWritten: currentOffset,
                        totalBytes,
                        chunkBytes: written,
                    });
                }
                // Drained cleanly — exit fetch-retry loop.
                break;
            } catch (readErr) {
                // Flush so the next attempt resumes from a known offset.
                try { writeHandle.flush(); } catch { /* */ }
                bytesSinceFlush = 0;

                fetchAttempt += 1;
                if (fetchAttempt > MAX_FETCH_RETRIES) {
                    throw readErr;
                }
                const delay = Math.min(5_000, 500 * Math.pow(2, fetchAttempt - 1));
                const msg = (readErr as Error)?.message ?? String(readErr);
                log(`download: ${modelKey}/${filename} stream broke at ${currentOffset}/${totalBytes} — retrying in ${delay}ms (attempt ${fetchAttempt}/${MAX_FETCH_RETRIES}): ${msg}`);
                notify("downloadRetrying", {
                    modelKey, filename,
                    bytesWritten: currentOffset,
                    totalBytes,
                    nextDelayMs: delay,
                    attempt: fetchAttempt,
                    maxAttempts: MAX_FETCH_RETRIES,
                });
                await new Promise((r) => setTimeout(r, delay));
                // Loop iterates to issue a fresh fetch with Range header.
            }
        }

        writeHandle.flush();
        writeHandle.close();
        writeHandle = null;

        notify("downloadDone", { modelKey, filename, totalBytes: currentOffset, fromCache: false });
        return { totalBytes: currentOffset, fromCache: false };
    } catch (err) {
        if (writeHandle) {
            try { writeHandle.flush(); } catch { /* */ }
            try { writeHandle.close(); } catch { /* */ }
        }
        const error = (err as Error)?.message ?? String(err);
        notify("downloadError", { modelKey, filename, error });
        throw new Error(error);
    }
}

async function handleEnsureModel(args: EnsureArgs): Promise<{ totalBytes: number; fromCache: boolean }> {
    const key = `${args.modelKey}/${args.filename}`;
    const existing = inflight.get(key);
    if (existing) return existing.promise;
    const promise = (async () => {
        try { return await doDownload(args); }
        finally { inflight.delete(key); }
    })();
    inflight.set(key, { promise });
    return promise;
}

// ───────────────────────────────────────────────────────────────────────
// RPC table — session enforcement is handled by the SharedWorker router.
// This worker trusts the shell to only forward stateful RPCs that the
// originating tab has acquired the session for.
// ───────────────────────────────────────────────────────────────────────

type Args = Record<string, unknown>;
type Handler = (a: Args) => unknown | Promise<unknown>;

const RPC: Record<string, Handler> = {
    // ── Liveness probe (router pings on new tab connect; if no pong
    //    arrives in time the router treats the corePort as dead and
    //    re-elects a host, instead of waiting out the 30s heartbeat).
    pingCore: () => true,

    // ── Diagnostic log RPCs (OPFS-backed; see workers/opfs_logger.ts).
    //    The viewer in components/LogsTab.tsx calls these to enumerate
    //    + read past sessions; beacon() in lib/api.ts calls logsAppend
    //    fire-and-forget to persist main-thread beacons through this
    //    worker's sync handle.
    logsList:    async () => await opfsLogger.listSessions() as unknown,
    logsRead:    async (a) => await opfsLogger.readSession(String(a.id)) as unknown,
    logsDelete:  async (a) => { await opfsLogger.deleteSession(String(a.id)); return true; },
    logsDeleteAll: async () => { await opfsLogger.deleteAll(); return true; },
    logsAppend:  (a) => {
        opfsLogger.append(String(a.level) as LogLevel, String(a.tag), String(a.msg));
        return true;
    },
    logsCurrentSession: () => SESSION_ID,

    // Queryable GPU-memory monitor — returns the tracked GPU buffer
    // breakdown (`tot=… w=… s=… kv=… lora=… o=…` MiB) on demand. The
    // test harness polls this between RPCs; the per-layer training
    // beacons carry the live total for mid-step trajectory.
    gpuMem: () => { try { return gpuMemBreakdownFn(); } catch { return "unavailable"; } },

    // ── Model lifecycle ────────────────────────────────────────────────
    load: (a) => handleLoad(a as unknown as LoadArgs),
    free: () => {
        if (model) { try { model.free?.(); } catch { /* */ } model = null; }
        if (syncHandle) { try { syncHandle.close(); } catch { /* */ } syncHandle = null; }
        if (loadedInfo) { loadedInfo = null; notify("modelFreed", {}); }
    },

    // ── ensureModel ─────────────────────────────────────────────────────
    ensureModel: (a) => handleEnsureModel(a as unknown as EnsureArgs),

    // ── Stateless inference helpers ─────────────────────────────────────
    encode:               (a) => Array.from(requireModel().encode(String(a.text))),
    tokenStr:             (a) => requireModel().tokenStr(Number(a.id)) ?? null,
    isEos:                (a) => requireModel().isEos(Number(a.id)),
    renderChat:           (a) => requireModel().renderChat(a.messages, !!a.withBos),
    imageSentinelIds:     () => requireModel().imageSentinelIds() ?? null,
    audioSentinelIds:     () => requireModel().audioSentinelIds() ?? null,
    imageSoftTokenCount:  (a) => requireModel().imageSoftTokenCount(Number(a.h), Number(a.w)),
    decodeWav:            (a) => requireModel().decodeWav(a.bytes as Uint8Array),

    // ── Stateful inference ──────────────────────────────────────────────
    step:          async (a) => await requireModel().step(Number(a.tokenId)),
    stepWithEmb:   async (a) => await requireModel().stepWithEmbedding(a.embedding as Float32Array),
    stepAndDecode: async (a) => {
        const m = requireModel();
        const next = await m.step(Number(a.tokenId));
        return { next, isEos: m.isEos(next), str: m.tokenStr(next) ?? null };
    },
    encodeImage: async (a) => {
        const cb = (layer: number, total: number) => {
            notify("pipelineProgress", { layer, total });
        };
        return await requireModel().encodeImage(
            a.pixels as Float32Array, Number(a.h), Number(a.w), cb,
        );
    },
    encodeAudio: async (a) => {
        const m = requireModel();
        const pcm = a.pcm as Float32Array;
        // Diagnostic beacons so a mobile jetsam-kill can be localised
        // on the next iPhone run. iOS Safari may kill WebContent
        // without surfacing a JS error; if we see "audio: encode start"
        // in the log but no "encode done", we know the encode itself
        // is the trigger. The cachedWeightBytes snapshot identifies
        // whether weight memory pressure is the culprit.
        const beforeBytes = Number(m.cachedWeightBytes ?? 0);
        log(`audio: encode start (samples=${pcm.length}, cached=${(beforeBytes / (1024 * 1024)).toFixed(0)}MB)`);
        try {
            const soft = await m.encodeAudio(pcm);
            const afterBytes = Number(m.cachedWeightBytes ?? 0);
            log(`audio: encode done (soft_dims=${soft.length}, cached=${(afterBytes / (1024 * 1024)).toFixed(0)}MB, Δ=${((afterBytes - beforeBytes) / (1024 * 1024)).toFixed(0)}MB)`);
            return soft;
        } catch (e) {
            log(`audio: encode FAILED: ${(e as Error).message}`);
            throw e;
        }
    },
    cancelMultimodalEncode: (a) => { void a; requireModel().cancelMultimodalEncode(); return true; },

    transcribeAudio: async (a) => {
        // In-engine speech-to-text driven by the audio tower + a fixed
        // instruction prompt + greedy decode. Streams per-token deltas
        // out via `notify("transcribeChunk", {delta, done})` so the UI
        // can fill the input box as the transcript arrives, and reuses
        // the existing `pipelineProgress` notify channel to drive the
        // status strip just above the chat input — same phase pill the
        // user sees for image encode + prefill.
        //
        // Sampling is forced to greedy (temperature=0, top_k=1)
        // regardless of the user's chat sampling settings — transcription
        // needs determinism. Settings restored on exit.
        const m = requireModel();
        const pcm = a.pcm as Float32Array;
        const maxTokens = Number(a.maxTokens ?? 512);

        const sent = m.audioSentinelIds();
        if (!sent) throw new Error("transcribeAudio: model has no <|audio> sentinel");
        const [audioBeginId] = sent;

        // 1. Save sampling, switch to greedy.
        const userSampling = a.sampling as Record<string, unknown> | undefined;
        m.setSampling({ temperature: 0, top_k: 1, top_p: 1, repetition_penalty: 1, seed: 0 });

        // 2. Encode audio → soft tokens.
        notify("pipelineProgress", { phase: "encoding", layer: 0, total: 1, modality: "audio" });
        log(`transcribe: encode start (samples=${pcm.length})`);
        const softTokens = await m.encodeAudio(pcm);
        // Gemma 4's text d_model is 1536 for both e2b and e4b; the
        // audio tower's projector outputs soft tokens in that same
        // embedding space. Same fallback the chat-attach path uses.
        const dText = 1536;
        const nSoft = softTokens.length / dText;
        log(`transcribe: encoded ${nSoft} soft tokens × ${dText} dim`);
        notify("pipelineProgress", { phase: "encoding", layer: 1, total: 1, modality: "audio" });

        // Free the Conformer tower (~3 GB GPU resident) before the text
        // prefill+splice+gen runs. On iPhone Safari WebGPU, keeping the
        // audio weights cached alongside the text tower's prefill scratch
        // tips the WebContent process over the GPU memory budget mid-
        // prefill and the worker dies silently. Audio tower isn't needed
        // again in this transcribe — soft tokens are already in CPU
        // memory; the next encodeAudio call will rebuild lazily.
        try {
            const freed = m.releaseAudioWeights();
            if (freed > 0) log(`transcribe: released ${(freed / (1024 * 1024)).toFixed(1)} MB audio weights`);
        } catch { /* */ }

        // 3. Render prompt — system + user split with withBos=false.
        //    audio_parity.rs (the proven-working harness) uses exactly
        //    this structure and produces clean output; an earlier
        //    single-user-message + withBos=true variant produced
        //    garbage like "wosredit" on the same input. Per
        //    `gemma4_small.rs:16` Gemma 4's chat template does NOT
        //    want a leading BOS.
        const messages = [
            {
                role: "system",
                content: "Transcribe the following audio exactly as spoken. Output only the transcription text, nothing else.",
            },
            {
                role: "user",
                content: "<|audio><audio|>Transcribe this audio.",
            },
        ];
        const promptText = m.renderChat(messages, false);
        const ids = m.encode(promptText);

        // 4. Reset KV cache, drive feed loop (splice soft tokens at sentinel).
        m.reset();
        let next = 0;
        for (let i = 0; i < ids.length; i++) {
            const id = ids[i];
            next = await m.step(id);
            if (id === audioBeginId) {
                // Splice soft tokens. This is the long phase — emit
                // per-row progress so the strip moves visibly while
                // we're walking through ~30+ stepWithEmbedding calls.
                for (let r = 0; r < nSoft; r++) {
                    notify("pipelineProgress", { phase: "embedding", layer: r, total: nSoft, modality: "audio" });
                    const row = softTokens.subarray(r * dText, (r + 1) * dText);
                    next = await m.stepWithEmbedding(row);
                }
            }
            // Per-token prefill progress.
            notify("pipelineProgress", { phase: "prefill", layer: i + 1, total: ids.length, modality: "audio" });
        }

        // 5. Greedy generate, streaming deltas. Reuse the
        //    pipelineProgress channel with a final phase so the strip
        //    stays alive while the model emits the transcript.
        let transcript = "";
        let genCount = 0;
        for (let gen = 0; gen < maxTokens; gen++) {
            if (m.isEos(next)) break;
            const tok = m.tokenStr(next);
            if (tok) {
                const delta = tok.replace(/▁/g, " ");
                transcript += delta;
                notify("transcribeChunk", { delta, done: false });
            }
            genCount++;
            notify("pipelineProgress", { phase: "generating", layer: genCount, total: maxTokens, modality: "audio" });
            next = await m.step(next);
        }

        // 6. Restore sampling.
        if (userSampling) {
            try { m.setSampling(userSampling); } catch { /* */ }
        }

        log(`transcribe: done (${transcript.length} chars, ${genCount} tokens)`);
        notify("transcribeChunk", { delta: "", done: true, transcript });
        return { transcript };
    },
    releaseVisionWeights: (a) => {
        void a;
        const freed = requireModel().releaseVisionWeights();
        if (freed > 0) log(`vision: released ${(freed / (1024 * 1024)).toFixed(1)} MB of GPU weight cache`);
        return freed;
    },
    releaseAudioWeights: (a) => {
        void a;
        const freed = requireModel().releaseAudioWeights();
        if (freed > 0) log(`audio: released ${(freed / (1024 * 1024)).toFixed(1)} MB of GPU weight cache`);
        return freed;
    },
    reset:        (a) => { void a; return requireModel().reset(); },
    setSampling:  (a) => requireModel().setSampling(a.opts),
    saveKvState:  async (a) => { void a; return await requireModel().saveKvState(); },
    restoreKvState: (a) => { requireModel().restoreKvState(a.bytes as Uint8Array); return true; },
    renderChatForContinuation: (a) =>
        requireModel().renderChatForContinuation(a.messages, !!a.withBos),
    position: (a) => { void a; return requireModel().position; },

    // ── Worker meta ─────────────────────────────────────────────────────
    currentMeta: () => ({ loaded: loadedInfo }),

    // ── Embeddings / RAG (EmbeddingGemma) ───────────────────────────────
    // Streaming load from an OPFS-cached GGUF (the main thread downloads it
    // first via ensureModel, same as the chat model). Weights are fetched on
    // demand; the 621 MB file never fully enters wasm memory — iPhone-safe.
    loadEmbedder: async (a) => {
        await ensureWasm();
        if (embedder) return embedderInfo;
        const modelKey = String(a.modelKey);
        const filename = String(a.filename);
        const name = String(a.name ?? "embeddinggemma");
        log(`embed: streaming ${name} from OPFS ${modelKey}/${filename}`);
        const { readFn, totalBytes } = await openEmbedderSyncReadFn(modelKey, filename);
        embedder = await EmbeddingModelClass.loadFromOpfs(readFn, totalBytes);
        embedderInfo = { name, dim: embedder.dim };
        log(`embed: ready dim=${embedder.dim} (streaming, ${totalBytes} B in OPFS)`);
        notify("embedderReady", embedderInfo);
        return embedderInfo;
    },

    embedderStatus: () => embedderInfo,

    unloadEmbedder: () => {
        if (embedder) { try { embedder.free?.(); } catch { /* */ } embedder = null; embedderInfo = null; }
        if (embedderSyncHandle) { try { embedderSyncHandle.close(); } catch { /* */ } embedderSyncHandle = null; }
        return true;
    },

    // ── DiffusionGemma (block-diffusion chat) ───────────────────────────────
    // Streaming load from an OPFS-cached GGUF (main thread downloads it first,
    // same as the chat model). Per-layer MoE experts stream in + are destroyed
    // each layer, so the 16.8 GB file never fully enters wasm memory.
    loadDiffuser: async (a) => {
        await ensureWasm();
        if (diffuser) return diffuserInfo;
        const modelKey = String(a.modelKey);
        const filename = String(a.filename);
        const name = String(a.name ?? "diffusiongemma");
        log(`diffusion: streaming ${name} from OPFS ${modelKey}/${filename}`);
        const { readFn, totalBytes } = await openDiffuserSyncReadFn(modelKey, filename);
        diffuser = await DiffusionGemmaClass.loadFromOpfs(readFn, totalBytes);
        diffuserInfo = { name, canvasLen: diffuser.canvasLen };
        log(`diffusion: ready canvasLen=${diffuser.canvasLen} (streaming, ${totalBytes} B in OPFS)`);
        notify("diffuserReady", diffuserInfo);
        return diffuserInfo;
    },

    diffuserStatus: () => diffuserInfo,

    unloadDiffuser: () => {
        if (diffuser) { try { diffuser.free?.(); } catch { /* */ } diffuser = null; diffuserInfo = null; }
        if (diffuserSyncHandle) { try { diffuserSyncHandle.close(); } catch { /* */ } diffuserSyncHandle = null; }
        return true;
    },

    // Arm + run a full block-diffusion generation, streaming the canvas in
    // place. Each denoise step is a full canvas forward (tens of seconds on a
    // weak GPU), so we emit a `diffuserStep` notification per step with the
    // current decoded canvas + stats, and the main thread re-renders it. The
    // OUTPUT canvas replaces (never appends) the assistant message each step.
    diffuserGenerate: async (a) => {
        if (!diffuser) throw new Error("no diffusion model loaded — call loadDiffuser() first");
        const prompt = String(a.prompt ?? "");
        const canvasLen = Number(a.canvasLen ?? 0);   // 0 ⇒ model default
        const maxSteps = Number(a.maxSteps ?? 0);     // 0 ⇒ default 48
        const seed = Number(a.seed ?? 0xD1FF);
        diffuser.startGenerate(prompt, canvasLen, maxSteps, seed);
        let text = "";
        while (!diffuser.done && !shuttingDown) {
            text = await diffuser.denoiseStep();
            notify("diffuserStep", {
                text,
                stepIndex: diffuser.stepIndex,
                totalSteps: diffuser.totalSteps,
                accepted: diffuser.accepted,
                meanEntropy: diffuser.meanEntropy,
                done: diffuser.done,
            });
        }
        return { text, done: true };
    },

    // Embed a query string → number[] (small payload, JSON-friendly).
    embedText: async (a) => {
        if (!embedder) throw new Error("no embedder loaded — call loadEmbedder() first");
        const dim = Number(a.targetDim ?? 0);
        return Array.from(await embedder.embed(String(a.text), dim));
    },

    // Index a document: embed each chunk + persist documents/chunks rows.
    // `chunks` = [{ text, page? }]. Returns { documentId, chunkCount }.
    embedDocument: async (a) => {
        if (!embedder) throw new Error("no embedder loaded — call loadEmbedder() first");
        const db = await ensureDb();
        const name = String(a.name ?? "document");
        const sourceKind = String(a.sourceKind ?? "txt");
        const conversationId = (a.conversationId as string | null | undefined) ?? null;
        const targetDim = Number(a.targetDim ?? 0);
        const chunks = (a.chunks as Array<{ text: string; page?: number }>) ?? [];
        const total = chunks.length;
        const docId = String(a.docId ?? name);

        // Insert the document row up-front, then embed + store each chunk one
        // at a time, emitting per-chunk progress (embedding is the slow part —
        // one GPU forward per chunk — so granularity matters here).
        const now = Date.now();
        db.execParams(
            `INSERT INTO documents (name, source_kind, byte_size, created_at, conversation_id, embedding_model, vector_dim)
             VALUES (?, ?, ?, ?, ?, ?, ?)`,
            [name, sourceKind, Number(a.byteSize ?? 0), now, conversationId, embedderInfo?.name ?? "", embedder.dim],
        );
        const idRow = db.query(`SELECT last_insert_rowid() AS id`) as Array<{ id: number }>;
        const documentId = idRow[0].id;

        let dim = embedder.dim;
        notify("embedProgress", { docId, phase: "embedding", done: 0, total, name });
        for (let i = 0; i < total; i++) {
            const vec = await embedder.embed(chunks[i].text, targetDim);
            dim = vec.length;
            // `vec` may be a VIEW into wasm linear memory — `.buffer` is then
            // the whole (multi-GB) wasm memory, not the 3072-byte vector.
            // Copy exactly byteLength bytes into a fresh buffer.
            const blob = new Uint8Array(vec.buffer.slice(vec.byteOffset, vec.byteOffset + vec.byteLength));
            db.execParams(
                `INSERT INTO chunks (document_id, chunk_idx, text, page, vector, vector_dim)
                 VALUES (?, ?, ?, ?, ?, ?)`,
                [documentId, i, chunks[i].text, chunks[i].page ?? null, blob, dim],
            );
            notify("embedProgress", { docId, phase: "embedding", done: i + 1, total, name });
        }
        notify("embedProgress", { docId, phase: "storing", done: total, total, name });
        db.flush();
        notify("embedProgress", { docId, phase: "done", done: total, total, name });
        notify("dbChanged", { kind: "docInsert", documentId });
        return { documentId, chunkCount: total, dim };
    },

    // KNN over chunks (brute-force cosine via rsqlite-wasm). Scope: a
    // conversation's own docs + global docs (conversation_id IS NULL).
    searchEmbeddings: async (a) => {
        if (!embedder) throw new Error("no embedder loaded — call loadEmbedder() first");
        const db = await ensureDb();
        const targetDim = Number(a.targetDim ?? 0);
        const k = Number(a.k ?? 5);
        const conversationId = (a.conversationId as string | null | undefined) ?? null;
        const qv = await embedder.embed(String(a.query), targetDim);
        // Copy exactly byteLength bytes — `qv` may be a wasm-memory view.
        const qblob = new Uint8Array(qv.buffer.slice(qv.byteOffset, qv.byteOffset + qv.byteLength));
        const dim = qv.length;
        return db.queryParams(
            `SELECT chunks.id AS chunk_id, chunks.text AS text, chunks.page AS page,
                    documents.id AS document_id, documents.name AS document_name,
                    vec_distance_cosine(chunks.vector, ?) AS distance
             FROM chunks JOIN documents ON documents.id = chunks.document_id
             WHERE chunks.vector_dim = ?
               AND (? IS NULL OR documents.conversation_id = ? OR documents.conversation_id IS NULL)
             ORDER BY distance ASC
             LIMIT ?`,
            [qblob, dim, conversationId, conversationId, k],
        );
    },

    listDocuments: async (a) => {
        const db = await ensureDb();
        const conversationId = (a.conversationId as string | null | undefined) ?? null;
        return db.queryParams(
            `SELECT d.id, d.name, d.source_kind, d.byte_size, d.created_at,
                    d.conversation_id, d.embedding_model, d.vector_dim,
                    (SELECT COUNT(*) FROM chunks WHERE document_id = d.id) AS chunk_count
             FROM documents d
             WHERE (? IS NULL OR d.conversation_id = ? OR d.conversation_id IS NULL)
             ORDER BY d.created_at DESC`,
            [conversationId, conversationId],
        );
    },

    deleteDocument: async (a) => {
        const db = await ensureDb();
        db.execParams(`DELETE FROM documents WHERE id = ?`, [Number(a.id)]);
        db.flush();
        notify("dbChanged", { kind: "docDelete", documentId: Number(a.id) });
        return true;
    },

    setConversationRag: async (a) => {
        const db = await ensureDb();
        db.execParams(
            `INSERT INTO conversation_rag (conversation_id, enabled) VALUES (?, ?)
             ON CONFLICT(conversation_id) DO UPDATE SET enabled = excluded.enabled`,
            [String(a.conversationId), a.enabled ? 1 : 0],
        );
        db.flush();
        return true;
    },

    getConversationRag: async (a) => {
        const db = await ensureDb();
        const rows = db.queryParams(
            `SELECT enabled FROM conversation_rag WHERE conversation_id = ?`,
            [String(a.conversationId)],
        ) as Array<{ enabled: number }>;
        return { enabled: rows.length > 0 && rows[0].enabled === 1 };
    },

    // ── Chat persistence (rsqlite-wasm OPFS-backed SQLite) ──────────────
    dbInit:   async () => { await ensureDb(); return true; },

    convList: async () => {
        const db = await ensureDb();
        return db.queryParams(
            `SELECT id, title, model, created_at, updated_at
             FROM conversations
             ORDER BY updated_at DESC`,
            [],
        );
    },

    convCreate: async (a) => {
        const db = await ensureDb();
        const id    = (a.id as string | undefined) ?? newId();
        const title = (a.title as string | undefined) ?? "New chat";
        const m     = (a.model as string | null | undefined) ?? null;
        const now   = Date.now();
        db.execParams(
            `INSERT INTO conversations (id, title, model, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?)`,
            [id, title, m, now, now],
        );
        db.flush();
        notify("dbChanged", { kind: "convInsert", conversationId: id });
        return { id, title, model: m, created_at: now, updated_at: now };
    },

    convDelete: async (a) => {
        const db = await ensureDb();
        const id = String(a.id);
        let opfsPaths: string[] = [];
        try {
            const rows = db.queryParams(
                `SELECT opfs_path FROM message_images WHERE conversation_id = ?`,
                [id],
            ) as Array<{ opfs_path: string }>;
            opfsPaths = rows.map((r) => r.opfs_path).filter(Boolean);
        } catch { /* */ }
        db.execParams(`DELETE FROM conversations WHERE id = ?`, [id]);
        db.flush();
        notify("dbChanged", { kind: "convDelete", conversationId: id });
        return { ok: true, opfsPaths };
    },

    convRename: async (a) => {
        const db = await ensureDb();
        const id    = String(a.id);
        const title = String(a.title);
        db.execParams(
            `UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?`,
            [title, Date.now(), id],
        );
        db.flush();
        notify("dbChanged", { kind: "convRename", conversationId: id });
        return true;
    },

    convTouch: async (a) => {
        const db = await ensureDb();
        const id = String(a.id);
        const titleIfBlank = a.titleIfBlank as string | undefined;
        const now = Date.now();
        if (titleIfBlank !== undefined && titleIfBlank.length > 0) {
            db.execParams(
                `UPDATE conversations
                 SET updated_at = ?,
                     title = CASE WHEN title = 'New chat' THEN ? ELSE title END
                 WHERE id = ?`,
                [now, titleIfBlank, id],
            );
        } else {
            db.execParams(
                `UPDATE conversations SET updated_at = ? WHERE id = ?`,
                [now, id],
            );
        }
        db.flush();
        notify("dbChanged", { kind: "convTouch", conversationId: id });
        return true;
    },

    msgList: async (a) => {
        const db = await ensureDb();
        const cid = String(a.conversationId);
        return db.queryParams(
            `SELECT conversation_id, message_id, role, content, created_at
             FROM messages
             WHERE conversation_id = ?
             ORDER BY created_at ASC`,
            [cid],
        );
    },

    msgInsert: async (a) => {
        const db = await ensureDb();
        const cid     = String(a.conversationId);
        const mid     = (a.messageId as string | undefined) ?? newId();
        const role    = String(a.role);
        const content = String(a.content ?? "");
        const now     = Date.now();
        db.execParams(
            `INSERT INTO messages (conversation_id, message_id, role, content, created_at)
             VALUES (?, ?, ?, ?, ?)`,
            [cid, mid, role, content, now],
        );
        return { messageId: mid, created_at: now };
    },

    msgAppend: async (a) => {
        const db = await ensureDb();
        const cid   = String(a.conversationId);
        const mid   = String(a.messageId);
        const delta = String(a.delta ?? "");
        db.execParams(
            `UPDATE messages
             SET content = content || ?
             WHERE conversation_id = ? AND message_id = ?`,
            [delta, cid, mid],
        );
        return true;
    },

    msgSetContent: async (a) => {
        const db = await ensureDb();
        const cid     = String(a.conversationId);
        const mid     = String(a.messageId);
        const content = String(a.content ?? "");
        db.execParams(
            `UPDATE messages SET content = ?
             WHERE conversation_id = ? AND message_id = ?`,
            [content, cid, mid],
        );
        return true;
    },

    msgInsertImage: async (a) => {
        const db = await ensureDb();
        const cid       = String(a.conversationId);
        const mid       = String(a.messageId);
        const seq       = Number(a.seq);
        const width     = Number(a.width);
        const height    = Number(a.height);
        const opfsPath  = String(a.opfsPath);
        db.execParams(
            `INSERT INTO message_images
                 (conversation_id, message_id, seq, width, height, opfs_path)
             VALUES (?, ?, ?, ?, ?, ?)`,
            [cid, mid, seq, width, height, opfsPath],
        );
        return true;
    },

    msgListImages: async (a) => {
        const db = await ensureDb();
        const cid = String(a.conversationId);
        return db.queryParams(
            `SELECT conversation_id, message_id, seq, width, height, opfs_path
             FROM message_images
             WHERE conversation_id = ?
             ORDER BY message_id, seq ASC`,
            [cid],
        );
    },

    dbFlush: async () => {
        const db = await ensureDb();
        db.flush();
        return true;
    },

    // ── LoRA fine-tuning ────────────────────────────────────────────────
    // Training consumes the Model handle for its lifetime; chat-side RPCs
    // (step, encode, reset, etc.) all throw via `requireModel()` while a
    // session is live. UI must mode-gate to avoid the throw.

    trainingStart: async (a) => {
        logBeacon("info", "trn", `trainingStart enter loraCfg=${JSON.stringify(a.loraConfig ?? {})} hp=${JSON.stringify(a.hparams ?? {})}`);
        if (!model) throw new Error("load a model before starting training");
        if (trainingSession) throw new Error(
            "training already active — call trainingFinish() first");
        const loraCfgJson = JSON.stringify(a.loraConfig ?? {});
        const hpJson      = JSON.stringify(a.hparams ?? {});

        // Probe first — try the scratch + LoRA allocations against a
        // BORROWED Model. If the device can't fit them, surface a
        // typed error WITHOUT consuming the Model handle. The chat
        // path stays alive; the user can lower rank / seq_len and
        // retry without re-loading the multi-GB GGUF.
        logBeacon("info", "trn", "probeFit start");
        const probe = await probeFit(model, loraCfgJson, hpJson);
        logBeacon("info", "trn", `probeFit done ok=${probe.ok} estMB=${(probe.estimatedBytes/(1024*1024)).toFixed(1)}`);
        log(`training: probe ok=${probe.ok} estimated=${(probe.estimatedBytes / (1024 * 1024)).toFixed(1)}MB ${probe.reason ?? ""}`);
        if (!probe.ok) {
            const mb = (probe.estimatedBytes / (1024 * 1024)).toFixed(0);
            throw new Error(
                `Training would need ~${mb} MB GPU memory and this device rejected the allocation. ` +
                `Lower the rank, shorten max_seq_len, or drop FFN targets and try again. ` +
                `(GPU said: ${probe.reason ?? "unknown"})`
            );
        }

        // **B3 — probe safety margin.** The probe's `ok=true` means
        // `wgpu` could currently allocate the requested scratch + LoRA
        // + Adam buffers, but it doesn't account for the per-step
        // backward transients (~1-2 MB per layer summed across the
        // backward sweep) that get allocated on the actual first
        // step. A config sitting at 95% of the device budget at
        // probe time will OOM the first real backward. Apply a
        // heuristic ceiling (85% of a per-platform budget) and reject
        // before consuming the Model.
        const ua = (typeof self !== "undefined" && self.navigator?.userAgent) ? self.navigator.userAgent : "";
        const isMobile = /iPhone|iPad|iPod|Android/i.test(ua);
        // iPhone WebContent process budget is ~3-4 GB shared with the
        // already-loaded text tower (~2 GB). Desktop wgpu typically
        // has multi-GB buffers available, so we set a high ceiling
        // that still leaves headroom for chat-side allocations.
        const budgetBytes = isMobile ? 3.5 * 1024 * 1024 * 1024 : 8 * 1024 * 1024 * 1024;
        const ratio = probe.estimatedBytes / budgetBytes;
        if (ratio > 0.85) {
            const mb = (probe.estimatedBytes / (1024 * 1024)).toFixed(0);
            const pct = (ratio * 100).toFixed(0);
            throw new Error(
                `Training config would use ~${mb} MB (${pct}% of the ${isMobile ? "iPhone" : "desktop"} GPU budget). ` +
                `That's too close to the ceiling — the first backward step's transient buffers may push it over. ` +
                `Lower the rank, shorten max_seq_len, drop FFN targets, or enable "Memory-tight" mode and try again.`,
            );
        }
        log(`training: probe within margin (${(ratio * 100).toFixed(0)}% of ${(budgetBytes / (1024 * 1024)).toFixed(0)}MB ${isMobile ? "mobile" : "desktop"} budget)`);

        // If there's an active adapter loaded into Model from a previous
        // session, clear it — training initialises fresh LoRA state.
        if (activeAdapterName) {
            try { model.clearAdapter(); }
            catch (e) { log(`trainingStart: clearAdapter failed (ignored): ${(e as Error).message ?? e}`); }
            activeAdapterName = null;
            notify("adapterChanged", { active: null });
        }

        // **A1 — release vision/audio GPU towers before allocating
        // training scratch.** If the user did any image / audio chat
        // before clicking "Start training", the ~3 GB tower(s) are still
        // GPU-resident. On iPhone's ~3-4 GB WebContent budget, that's a
        // guaranteed OOM the moment TrainingScratch + LoRA + Adam buffers
        // try to allocate. The chat path's release calls already exist
        // (`api.rs:330,343`) — `trainingStart` just never invoked them.
        // Pattern mirrors `releaseAllHandles()` below.
        try {
            // releaseVision/AudioWeights return the number of GPU
            // weight-cache ENTRIES dropped (one per tile/tensor) —
            // not bytes. Each entry is on the order of MBs in
            // practice (Q4_K tile, etc.), but we don't have an
            // accurate byte sum here. Log entries-freed and let the
            // user infer.
            const freedV = model.releaseVisionWeights?.() ?? 0;
            const freedA = model.releaseAudioWeights?.() ?? 0;
            log(`trainingStart: released ${freedV} vision + ${freedA} audio GPU cache entries before TrainingSession::new (plus per-tower scratch ~250 MB each if any was built)`);
            logBeacon("info", "trn", `released vision=${freedV} audio=${freedA} cache entries`);
        } catch (e) {
            // Model lacks the methods (no multimodal towers in this
            // GGUF) or release threw — neither is fatal for training.
            log(`trainingStart: pre-training multimodal release skipped: ${(e as Error).message ?? e}`);
        }

        // **B1 — shrink KV cache for training.** Chat reserves
        // `max_context` positions of K/V cache (~600 MB at 4096 on
        // gemma4:e2b). Training's NextToken loss only needs 1 history
        // position; PerPosition needs at most `seq_len`. Shrinking to
        // `max(seq_len + 1, 64)` frees the bulk of that allocation back
        // to the WebGPU device for training scratch / LoRA / Adam.
        // Stash the original so trainingFinish can restore.
        try {
            const hp = (a.hparams ?? {}) as { max_seq_len?: number; seq_len?: number };
            const seqLen = Math.max(1, Number(hp.max_seq_len ?? hp.seq_len ?? 32));
            const targetKv = Math.max(seqLen + 1, 64);
            const original = Number(model.maxContext ?? 0);
            if (original > 0 && targetKv < original) {
                model.shrinkKv(targetKv);
                trainingOriginalKvContext = original;
                log(`trainingStart: shrunk KV cache ${original} → ${targetKv} tokens (will restore on finish)`);
            } else {
                trainingOriginalKvContext = null;
            }
        } catch (e) {
            // shrinkKv unsupported (older wasm bundle?) or threw — fall
            // through. Training still works at the full KV size; only
            // the iPhone-tight-RAM case suffers.
            log(`trainingStart: KV shrink skipped: ${(e as Error).message ?? e}`);
            trainingOriginalKvContext = null;
        }

        // **A2 — don't null `model` until TrainingSession::new
        // succeeds.** Old code did `const moved = model; model = null;`
        // before the constructor, so a throw mid-init left the JS-side
        // model variable nulled even though the user could conceivably
        // recover with a manual reload. Reorder so the null happens
        // AFTER the constructor returns, and on failure surface a
        // clear error + reset loadedInfo so the chat UI honestly
        // reflects "no model — reload required" instead of letting
        // the next chat-send vanish into a half-consumed handle.
        logBeacon("info", "trn", "TrainingSession::new constructing");
        let session: TrainingSessionHandle;
        try {
            session = new TrainingSessionClass(model, loraCfgJson, hpJson);
        } catch (e) {
            logBeacon("error", "trn", `TrainingSession::new threw: ${(e as Error).message ?? e}`);
            log(`trainingStart: TrainingSession::new failed: ${(e as Error).message ?? e}`);
            // wasm-bindgen takes ownership of the Model on entry; on
            // throw, the handle is consumed regardless of where we
            // null the JS variable. So we DO need to clean up the
            // JS-side state here — but only after the throw, not
            // pre-emptively.
            model = null;
            loadedInfo = null;
            notify("modelFreed", {});
            throw e;
        }
        logBeacon("info", "trn", "TrainingSession::new returned");
        trainingSession = session;
        model = null; // Ownership transferred — null now that it's safe.
        const totalSteps = Number(a.totalSteps ?? 0);
        if (totalSteps > 0) trainingSession.setLrSchedule(totalSteps);
        const info = {
            parameterCount: trainingSession.parameterCount,
            gradientCheckpointing: trainingSession.gradientCheckpointing,
            mixedPrecision: trainingSession.mixedPrecision,
            estimatedBytes: probe.estimatedBytes,
        };
        notify("trainingStarted", info);
        log(`training: started, params=${info.parameterCount} ckpt=${info.gradientCheckpointing} mp=${info.mixedPrecision}`);
        logBeacon("info", "trn", `trainingStart done params=${info.parameterCount} ckpt=${info.gradientCheckpointing} mp=${info.mixedPrecision}`);
        return info;
    },

    trainingProbeFit: async (a) => {
        if (!model) throw new Error("load a model before probing fit");
        if (trainingSession) throw new Error("training already active — can't probe");
        const loraCfgJson = JSON.stringify(a.loraConfig ?? {});
        const hpJson      = JSON.stringify(a.hparams ?? {});
        return await probeFit(model, loraCfgJson, hpJson);
    },

    trainingStep: async (a) => {
        const s = requireTraining();
        const inputIds = a.inputIds as Uint32Array;
        const lossMode = String(a.lossMode ?? "next_token");
        const wbStart = (() => { try { return Math.round(s.cachedWeightBytes / 1048576); } catch { return -1; } })();
        logBeacon("info", "trn", `step ${s.stepNum + 1} start mode=${lossMode} inputLen=${inputIds.length} weightCacheMB=${wbStart}`);
        // **Capture stepNum/lr BEFORE calling step.** The progress
        // callback fires from inside the Rust step's execution, which
        // holds a `&mut self` RefMut on the TrainingSession. Reading
        // `s.stepNum` / `s.lr` from inside the callback would attempt
        // to acquire a `&self` Ref on the same RefCell, which fails
        // with "attempted to take ownership of Rust value while it
        // was borrowed" — observed under high-rank + many-step runs.
        // The values are stable for the duration of the step anyway
        // (stepNum bumps AFTER the optimizer call completes), so
        // capturing them once is equivalent and avoids the borrow
        // conflict. The captured `step` value is the index of the
        // step that's about to run, which is exactly what the UI
        // progress strip wants.
        const stepBefore = s.stepNum;
        const lrBefore = s.lr;
        const onProgress = (phase: string, current: number, total: number) => {
            notify("trainingProgress", { phase, current, total, step: stepBefore, lr: lrBefore });
            // Mirror to OPFS so a mid-step crash leaves an exact
            // "last phase reached" trail — now WITH the live GPU-memory
            // total so the post-crash log shows the on-device memory
            // trajectory and the exact MiB at the moment iOS jetsam'd.
            // gpuMemTotalFn is a free wasm fn reading a static counter —
            // safe to call re-entrantly here mid-step.
            // Log the full GPU-memory breakdown (tot/w/s/kv/lora/o MiB), not just the total —
            // so a per-phase climb points at the exact category (e.g. is the per-prefill-token
            // growth `w` weights re-uploaded-not-freed, or `s` scratch/captures?). Both fns read
            // static counters; safe to call re-entrantly mid-step.
            let mem = "gpuMiB=-1";
            try { mem = gpuMemBreakdownFn(); } catch { try { mem = `gpuMiB=${Math.round(gpuMemTotalFn())}`; } catch { /* */ } }
            logBeacon("info", "trn", `step ${stepBefore + 1} ${phase} ${current}/${total} ${mem}`);
        };
        try {
            const result = lossMode === "per_position"
                ? await s.stepPerPosition(inputIds, a.targets as Uint32Array, onProgress)
                : await s.step(inputIds, Number(a.targetId), onProgress);
            // **A3 — NaN/Inf auto-halt.** Without this, training would
            // continue stepping with NaN-polluted Adam state and the
            // user could unknowingly save a garbage adapter. Throw
            // early; the UI's catch reports "training diverged" and
            // moves to the error phase so the partial-adapter Save
            // button isn't offered.
            const loss = (result as { loss?: unknown })?.loss;
            if (typeof loss === "number" && !Number.isFinite(loss)) {
                throw new Error(
                    `training diverged at step ${s.stepNum} — loss is ${loss}. ` +
                    `Try a lower learning rate, smaller rank, or shorter seq_len.`,
                );
            }
            // Log EVERY step now. Step-8-crash investigation needs the
            // full trajectory leading to the failure, not a sampled
            // every-10th view. Long stable runs will tolerate the log
            // volume; if it gets noisy we can throttle later.
            const stepNum = (result as { step?: number })?.step ?? 0;
            const lossStr = typeof loss === "number" ? loss.toFixed(4) : String(loss);
            const wbDone = (() => { try { return Math.round(s.cachedWeightBytes / 1048576); } catch { return -1; } })();
            log(`training: step ${stepNum} loss=${lossStr} lossMode=${lossMode} inputLen=${inputIds.length}`);
            logBeacon("info", "trn", `step ${stepNum} done loss=${lossStr} weightCacheMB=${wbDone}`);
            // **iOS reclaim window.** Backward destroy()'d the cache at
            // its GPU-idle point (step done at weightCacheMB=0), but
            // WebGPU/Metal frees the underlying GPUBuffer memory
            // ASYNCHRONOUSLY — our tracked counter hits 0 long before
            // iOS reclaims the RSS. With the full MeBP stack (per-layer
            // destroy + early token_embd destroy + 500ms head yield) the
            // FIRST step (#2 internally) now completes cleanly on iPhone
            // (loss=0.0343, all 10 backward layers, peak gpuMiB=80). But
            // step 3 dies at the same head→backward boundary step 2
            // survived — Metal heap accumulation from step 2's backward
            // kernels + their compile state + the aliasable-but-not-
            // returned token_embd region adds up across steps.
            //
            // Stretch the yield to 15 s. Comprises (a) the same 5 s
            // we used to use (proven enough on the prior wasm-family
            // for early step 2 starts) + (b) an extra 10 s for the
            // backward kernels' Metal heap regions to actually drain
            // back to the OS. Wall-clock cost: 15 s per step on a step
            // that takes 30-60 s of compute; not great for training
            // throughput but the alternative is no multi-step training.
            for (let i = 0; i < 60; i++) await new Promise((r) => setTimeout(r, 250));
            return result;
        } catch (e) {
            // Log + rethrow. The session stays alive (the wasm side
            // doesn't drop on a kernel error), so the UI can choose
            // Discard to release the Model cleanly.
            log(`training: step ${s.stepNum} failed (lossMode=${lossMode}, inputLen=${inputIds.length}): ${(e as Error).message}`);
            logBeacon("error", "trn", `step ${s.stepNum} threw: ${(e as Error).message}`);
            throw e;
        }
    },

    trainingZeroGrads: async (a) => { void a; requireTraining().zeroGrads(); return true; },

    trainingForwardBackward: async (a) => {
        const s = requireTraining();
        const inputIds = a.inputIds as Uint32Array;
        const lossMode = String(a.lossMode ?? "next_token");
        logBeacon("info", "trn", `fwdBwd ${s.stepNum + 1} start mode=${lossMode} inputLen=${inputIds.length}`);
        try {
            const loss = lossMode === "per_position"
                ? await s.forwardBackwardPerPosition(inputIds, a.targets as Uint32Array)
                : await s.forwardBackward(inputIds, Number(a.targetId));
            logBeacon("info", "trn", `fwdBwd ${s.stepNum + 1} done loss=${typeof loss === "number" ? loss.toFixed(4) : String(loss)}`);
            // **A3 — NaN/Inf auto-halt.** Same rationale as in
            // `trainingStep`; this is the manual-accumulation entry
            // point and needs the same defence.
            if (typeof loss === "number" && !Number.isFinite(loss)) {
                throw new Error(
                    `training diverged at step ${s.stepNum} — loss is ${loss}. ` +
                    `Try a lower learning rate, smaller rank, or shorter seq_len.`,
                );
            }
            return { loss, step: s.stepNum, lr: s.lr };
        } catch (e) {
            log(`training: forwardBackward step ${s.stepNum} failed (lossMode=${lossMode}, inputLen=${inputIds.length}): ${(e as Error).message}`);
            logBeacon("error", "trn", `fwdBwd ${s.stepNum + 1} threw: ${(e as Error).message}`);
            throw e;
        }
    },

    trainingOptimizerStep: async (a) => {
        void a;
        const s = requireTraining();
        logBeacon("info", "trn", `adam start step ${s.stepNum + 1}`);
        s.optimizerStep();
        logBeacon("info", "trn", `adam done step ${s.stepNum} lr=${s.lr}`);
        return { step: s.stepNum, lr: s.lr };
    },

    trainingSaveAdapter: async (a) => {
        const s = requireTraining();
        const name = String(a.name ?? "").trim();
        if (!name) throw new Error("trainingSaveAdapter: 'name' must be a non-empty string");
        if (!/^[\w\-. ]+$/.test(name)) {
            throw new Error("trainingSaveAdapter: 'name' must match [\\w\\-. ]+");
        }
        const bytes = await s.saveAdapter();
        const written = await writeAdapterBytes(name, bytes);
        log(`training: saved adapter '${name}' (${written} bytes) → OPFS:${ADAPTERS_DIR}/${name}.bin`);
        notify("adapterSaved", { name, size: written });
        return { name, size: written };
    },

    // **Combined save+finish.** Calls the Rust-side
    // `saveAdapterAndFinish(self)` which consumes the TrainingSession
    // in a single await and returns both the safetensors bytes and the
    // wrapped Model. Avoids the wasm-bindgen async-borrow problem the
    // separated `saveAdapter` → `finish` pair hits:
    //   • `save_adapter_js(&mut self).await` leaves a `Borrow` tracked
    //     on the JS-side wrapper that persists past the await's
    //     resolution (even with setTimeout-zero yields), so a
    //     subsequent `finish_js(self)` call intermittently fails with
    //     "attempted to take ownership of Rust value while it was
    //     borrowed".
    //   • `save_adapter_and_finish_js(self)` takes `self` at call
    //     time — wasm-bindgen invalidates the JS handle on entry,
    //     no `&self`/`&mut self` borrow is ever tracked, and the
    //     consume-self happens deterministically inside the Rust
    //     function body.
    // This is now the recommended worker-side path for any flow that
    // wants to save AND release the session. The old `saveAdapter`
    // RPC stays for the rare save-without-finishing case.
    trainingSaveAdapterAndFinish: async (a) => {
        const session = requireTraining();
        const name = String(a?.name ?? "").trim();
        if (!name) throw new Error("trainingSaveAdapterAndFinish: 'name' required");
        if (!/^[\w\-. ]+$/.test(name)) {
            throw new Error("trainingSaveAdapterAndFinish: 'name' must match [\\w\\-. ]+");
        }
        const result = await session.saveAdapterAndFinish();
        trainingSession = null;
        const bytes = result.bytes;
        const written = await writeAdapterBytes(name, bytes);
        model = result.takeModel();
        log(`training: saved+finished adapter '${name}' (${written} bytes) → OPFS:${ADAPTERS_DIR}/${name}.bin`);
        notify("adapterSaved", { name, size: written });
        // Restore the KV cache to chat-size, same as the standalone
        // finish path. Fail-soft: a restore failure leaves the smaller
        // cache in place but chat still works.
        if (trainingOriginalKvContext != null) {
            const original = trainingOriginalKvContext;
            trainingOriginalKvContext = null;
            try {
                model.shrinkKv(original);
                log(`trainingSaveAdapterAndFinish: restored KV cache to ${original} tokens`);
            } catch (e) {
                log(`trainingSaveAdapterAndFinish: KV restore failed (cache stays small): ${(e as Error).message ?? e}`);
            }
        }
        notify("trainingFinished", {});
        return { name, size: written };
    },

    trainingCancel: async (a) => {
        void a;
        if (!trainingSession) return false;
        // Flips the cooperative cancel flag on Forward; in-flight
        // step rejects on the next per-layer encoder boundary. The
        // session itself stays alive until the user calls
        // trainingFinish — cancel is purely "stop the current step".
        trainingSession.cancel();
        log(`training: cancel requested at step ${trainingSession.stepNum}`);
        return true;
    },

    trainingFinish: async (a) => {
        void a;
        logBeacon("info", "trn", "trainingFinish enter");
        if (!trainingSession) throw new Error("no training session to finish");
        // No prior save call on this path (discard / cancel flow), so
        // no async-borrow to drain. `finish()` synchronously consumes
        // `self` — wasm-bindgen invalidates the JS handle on call.
        // Only null `trainingSession` AFTER finish() returns
        // successfully; if it throws, leave the session alive so the
        // user can retry (or so the save+finish path can take over).
        const finished = trainingSession.finish();
        trainingSession = null;
        model = finished;
        // **B1 — restore chat's KV cache.** If trainingStart shrunk it,
        // grow it back so chat's next turn has the full context window.
        // Fail-soft: a restore failure just leaves the smaller cache in
        // place; chat still works, just with a shorter history limit
        // until the user reloads the model.
        if (model && trainingOriginalKvContext != null) {
            const original = trainingOriginalKvContext;
            trainingOriginalKvContext = null;
            try {
                model.shrinkKv(original);
                log(`trainingFinish: restored KV cache to ${original} tokens`);
            } catch (e) {
                log(`trainingFinish: KV restore failed (cache stays small): ${(e as Error).message ?? e}`);
            }
        }
        notify("trainingFinished", {});
        log(`training: finished, Model returned to chat`);
        logBeacon("info", "trn", "trainingFinish done");
        return true;
    },

    trainingApplyAdapter: async (a) => {
        if (!model) throw new Error("load a model before applying an adapter");
        if (trainingSession) throw new Error(
            "active training session owns the model — finish it first");
        const name = String(a.name ?? "").trim();
        if (!name) throw new Error("trainingApplyAdapter: 'name' required");
        const bytes = await readAdapterBytes(name);
        const slots = model.loadAdapter(bytes);
        activeAdapterName = name;
        notify("adapterChanged", { active: name, slots });
        log(`training: applied adapter '${name}' (${slots} slots)`);
        return { name, slots };
    },

    trainingClearAdapter: async (a) => {
        void a;
        if (!model) return false;
        if (trainingSession) throw new Error(
            "active training session owns the model — finish it first");
        model.clearAdapter();
        const was = activeAdapterName;
        activeAdapterName = null;
        notify("adapterChanged", { active: null, was });
        return true;
    },

    trainingListAdapters: async (a) => {
        void a;
        const entries = await listAdapterEntries();
        return { entries, active: activeAdapterName };
    },

    trainingDeleteAdapter: async (a) => {
        const name = String(a.name ?? "").trim();
        if (!name) throw new Error("trainingDeleteAdapter: 'name' required");
        if (activeAdapterName === name && model) {
            try { model.clearAdapter(); } catch { /* */ }
            activeAdapterName = null;
            notify("adapterChanged", { active: null });
        }
        try {
            const dir = await getAdaptersDir(false);
            await dir.removeEntry(`${name}.bin`);
        } catch { /* */ }
        notify("adapterDeleted", { name });
        return true;
    },

    trainingStatus: async (a) => {
        void a;
        if (!trainingSession) return { active: false };
        return {
            active: true,
            step: trainingSession.stepNum,
            lr: trainingSession.lr,
            parameterCount: trainingSession.parameterCount,
            gradientCheckpointing: trainingSession.gradientCheckpointing,
            mixedPrecision: trainingSession.mixedPrecision,
        };
    },
};

// ───────────────────────────────────────────────────────────────────────
// Dispatch
// ───────────────────────────────────────────────────────────────────────

/** Classify a thrown error as a GPU fault (device-lost, OOM, WebGPU
 *  validation) so the UI can show a typed banner instead of letting
 *  the worker look "frozen." Pattern-matches the error message; wgpu
 *  surfaces these as plain `Error` instances with descriptive strings,
 *  so message-matching is the only available hook. */
function classifyGpuFault(msg: string): "device-lost" | "oom" | "validation" | null {
    const m = msg.toLowerCase();
    if (m.includes("device") && m.includes("lost")) return "device-lost";
    if (m.includes("out of memory") || m.includes("oom") || m.includes("memory allocation")) return "oom";
    if (m.includes("webgpu") && (m.includes("error") || m.includes("invalid"))) return "validation";
    return null;
}

async function handleRequest(msg: { requestId: number; type: string } & Args) {
    if (!msg || typeof msg !== "object" || !msg.type) return;
    const { requestId, type } = msg;
    const handler = RPC[type];
    if (!routerPort) return;
    const post = (payload: Record<string, unknown>) => {
        try { routerPort!.postMessage(payload); } catch { /* */ }
    };
    if (shuttingDown) {
        // Worker is mid-cleanup — refuse every RPC, including `pingCore`,
        // so the router's `verifyCoreLive` on a new-tab connect sees us
        // as dead and re-elects a fresh host. Without this, ping (a
        // trivial `() => true`) would succeed before `self.close()`
        // fires and the new tab would inherit a stale corePort.
        post({ requestId, ok: false, error: "core shutting down" });
        return;
    }
    if (!handler) {
        post({ requestId, ok: false, error: `unknown RPC type: ${type}` });
        return;
    }
    if (RPC_TRACE) log(`rpc-start ${type}`);
    try {
        const result = await handler(msg as Args);
        if (RPC_TRACE) log(`rpc-done  ${type}`);
        post({ requestId, ok: true, result });
    } catch (e) {
        const err = (e as Error)?.message ?? String(e);
        log(`rpc ${type} failed: ${err}`);
        // Typed GPU-fault diagnostic — the UI subscribes to this
        // separately so a device-lost / OOM surfaces as a banner
        // ("the GPU bailed — try reloading or freeing other tabs")
        // instead of the generic error path that just looks like a
        // hung worker.
        const fault = classifyGpuFault(err);
        if (fault) {
            log(`rpc ${type} GPU fault: ${fault}`);
            notify("gpuFault", { kind: fault, message: err, during: type });
        }
        post({ requestId, ok: false, error: err });
    }
}

/** Synchronously release every OS handle the core worker holds. Called
 *  from the shutdown signal on `pagehide`/`beforeunload`, plus as a
 *  belt-and-suspenders cleanup before `self.close()`.
 *
 *  Order matters here:
 *
 *  1. **GPU towers first.** `releaseVisionWeights()` + `releaseAudioWeights()`
 *     synchronously drop the per-block weight buffers (~3 GB audio,
 *     ~3 GB vision on gemma4:e2b). These are the largest GPU residents;
 *     `model.free?.()` would *eventually* free them via Rust Drop, but
 *     iOS Safari can take a noticeable moment before the wgpu queue
 *     actually surrenders the memory — and during that window the NEW
 *     core worker is already booting, allocating its own audio tower,
 *     and OOMing the GPU. This was the "audio crashing on iPhone after
 *     a reload" symptom: the data and code were fine, the previous
 *     worker's GPU surface just hadn't been handed back yet.
 *  2. **wasm Model.** Drops the text tower, KV cache, sampler state,
 *     pipeline cache. Rust Drop handles the rest.
 *  3. **OPFS sync handle.** Releases the exclusive lock so the next
 *     worker can open the GGUF.
 *  4. **DB.** Best-effort close (may be a pending Promise).
 */
function releaseAllHandles() {
    try {
        if (model) {
            try {
                const freedV = model.releaseVisionWeights?.() ?? 0;
                const freedA = model.releaseAudioWeights?.() ?? 0;
                if (freedV > 0 || freedA > 0) {
                    log(`shutdown: released ${(freedV / (1024 * 1024)).toFixed(0)}MB vision + ${(freedA / (1024 * 1024)).toFixed(0)}MB audio GPU weights before free()`);
                }
            } catch { /* */ }
            try { model.free?.(); } catch { /* */ }
            model = null;
        }
    } catch { /* */ }
    try {
        if (syncHandle) {
            try { syncHandle.close(); } catch { /* */ }
            syncHandle = null;
        }
    } catch { /* */ }
    try {
        // dbReady may be a pending Promise; if so, settle it best-effort
        // by no-op (the worker is about to exit). If resolved, close().
        if (dbReady) {
            void dbReady.then((db) => {
                try { (db as unknown as { close?: () => void }).close?.(); } catch { /* */ }
            }).catch(() => {});
            dbReady = null;
        }
    } catch { /* */ }
    loadedInfo = null;
}

// Boot: wait for the host tab's `attach` message carrying the
// MessagePort we'll use for all router traffic. The same channel also
// carries the `{type: "shutdown"}` signal posted by `WorkerClient` on
// `pagehide` — we cleanly release OPFS / DB handles + the wasm Model
// before `self.close()`, so the next boot's `createSyncAccessHandle`
// doesn't collide with a leaked handle from this worker. (Browser GC of
// dead workers is slow on iOS Safari, so a plain `terminate()` from the
// main thread can otherwise leave the model file locked for minutes.)
(self as unknown as DedicatedWorkerGlobalScope).onmessage = (ev: MessageEvent) => {
    const data = ev.data;
    if (!data || typeof data !== "object") return;

    if (data.type === "shutdown") {
        // Flip the flag BEFORE running cleanup so any in-flight RPC
        // (notably `pingCore` from the router's `verifyCoreLive`) sees
        // us as shutting-down and replies `ok: false`. See the comment
        // above `shuttingDown` for why this race matters.
        shuttingDown = true;
        try { log("core: shutdown received — releasing handles"); } catch { /* */ }
        // Mark the session log as clean-exit so the next-load crash
        // detector doesn't flag this session. Fire-and-forget — we
        // don't await because self.close() races with iOS jetsam in
        // edge cases. The await inside markCleanExit() still gets a
        // chance to land before self.close() runs since both are
        // microtask-scheduled and JS doesn't pre-empt.
        (async () => {
            try { await opfsLogger.markCleanExit(); } catch { /* */ }
        })();
        releaseAllHandles();
        try {
            (self as unknown as DedicatedWorkerGlobalScope).close();
        } catch { /* */ }
        return;
    }

    if (data.type !== "attach" || !data.port) return;
    if (routerPort) return; // already attached
    routerPort = data.port as MessagePort;
    routerPort.addEventListener("message", (e: MessageEvent) => {
        void handleRequest(e.data);
    });
    routerPort.start();
    // Lazy-init the OPFS logger now that we know this worker won the
    // SharedWorker election. Fire-and-forget — beacons fired before
    // init completes are silently dropped (the append() guard returns
    // early), which is acceptable since the first attached-and-active
    // moment is the earliest interesting point anyway.
    if (!loggerInited) {
        loggerInited = true;
        (async () => {
            try { await opfsLogger.init(SESSION_ID); }
            catch (e) { try { console.warn("[opfs_logger] init failed", e); } catch { /* */ } }
        })();
    }
    log(`core: attached to router (session ${SESSION_ID})`);
};
