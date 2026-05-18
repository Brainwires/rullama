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
import init, { Model, TrainingSession, probeTrainingFit, WasmDatabase } from "/pkg/rullama.js";

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
    setLrSchedule(totalSteps: number): void;
    finish(): ModelHandle;
    cancel(): void;
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
// When non-null, a training session owns the Model. All Model-mutating
// RPCs (step, encodeImage, reset, etc.) refuse to run; the chat UI
// gates this at the surface level.
let trainingSession: TrainingSessionHandle | null = null;
/** Name of the adapter currently loaded into Model, if any. */
let activeAdapterName: string | null = null;

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
}

function notify(kind: string, payload: Record<string, unknown> = {}) {
    if (!routerPort) return;
    try { routerPort.postMessage({ type: "notify", kind, ...payload }); } catch { /* */ }
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
    syncHandle     = await fh.createSyncAccessHandle();
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
    if (!model) throw new Error("no model loaded — call load() first");
    if (trainingSession) throw new Error(
        "model is owned by an active training session — call trainingFinish() first");
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
    const handle = await fh.createSyncAccessHandle();
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
        const head = new Uint8Array(await f.slice(0, 4).arrayBuffer());
        if (head[0] !== 0x47 || head[1] !== 0x47 || head[2] !== 0x55 || head[3] !== 0x46) return 0;
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
        writeHandle = await fileHandle.createSyncAccessHandle();
        writeHandle.truncate(currentOffset);
        writeHandle.flush();

        const headers: Record<string, string> = {};
        if (currentOffset > 0) headers["Range"] = `bytes=${currentOffset}-`;
        const resp = await fetch(url, { headers });

        if (resp.status === 416) {
            const size = writeHandle.getSize();
            writeHandle.close();
            notify("downloadDone", { modelKey, filename, totalBytes: size, fromCache: true });
            return { totalBytes: size, fromCache: false };
        }
        if (!resp.ok && resp.status !== 206) {
            writeHandle.close();
            throw new Error(`fetch failed (${resp.status})`);
        }
        if (resp.status === 200 && currentOffset > 0) {
            writeHandle.truncate(0);
            currentOffset = 0;
        }

        const cr = resp.headers.get("content-range");
        const contentLength = Number(resp.headers.get("content-length") || "0") || 0;
        const totalBytes = cr?.match(/\/(\d+)\s*$/)
            ? Number(cr.match(/\/(\d+)\s*$/)![1])
            : currentOffset + contentLength;

        if (!resp.body) {
            writeHandle.close();
            throw new Error("no response body");
        }
        const reader = resp.body.getReader();

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
            notify("visionProgress", { layer, total });
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
        const probe = await probeFit(model, loraCfgJson, hpJson);
        log(`training: probe ok=${probe.ok} estimated=${(probe.estimatedBytes / (1024 * 1024)).toFixed(1)}MB ${probe.reason ?? ""}`);
        if (!probe.ok) {
            const mb = (probe.estimatedBytes / (1024 * 1024)).toFixed(0);
            throw new Error(
                `Training would need ~${mb} MB GPU memory and this device rejected the allocation. ` +
                `Lower the rank, shorten max_seq_len, or drop FFN targets and try again. ` +
                `(GPU said: ${probe.reason ?? "unknown"})`
            );
        }

        // If there's an active adapter loaded into Model from a previous
        // session, clear it — training initialises fresh LoRA state.
        if (activeAdapterName) {
            try { model.clearAdapter(); } catch { /* */ }
            activeAdapterName = null;
            notify("adapterChanged", { active: null });
        }
        const moved = model;
        model = null; // Model is moved into TrainingSession.
        try {
            trainingSession = new TrainingSessionClass(moved, loraCfgJson, hpJson);
        } catch (e) {
            // Probe said ok but the constructor still threw — likely a
            // device-loss race. Drop loadedInfo so chat re-loads
            // cleanly on next attempt rather than wedging on a
            // half-consumed Model handle.
            loadedInfo = null;
            notify("modelFreed", {});
            throw e;
        }
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
        // Progress beacons: every per-layer + per-token tick fans out
        // via the `trainingProgress` notify so the UI's
        // TrainingProgress strip (modelled on VisionProgress) updates
        // mid-step. Fast (~3-5 Hz worth of postMessage traffic on
        // browser); the strip prevents the "is it frozen?" panic
        // during slow first steps.
        const onProgress = (phase: string, current: number, total: number) => {
            notify("trainingProgress", { phase, current, total, step: s.stepNum, lr: s.lr });
        };
        try {
            return lossMode === "per_position"
                ? await s.stepPerPosition(inputIds, a.targets as Uint32Array, onProgress)
                : await s.step(inputIds, Number(a.targetId), onProgress);
        } catch (e) {
            // Log + rethrow. The session stays alive (the wasm side
            // doesn't drop on a kernel error), so the UI can choose
            // Discard to release the Model cleanly.
            log(`training: step ${s.stepNum} failed (lossMode=${lossMode}, inputLen=${inputIds.length}): ${(e as Error).message}`);
            throw e;
        }
    },

    trainingZeroGrads: async (a) => { void a; requireTraining().zeroGrads(); return true; },

    trainingForwardBackward: async (a) => {
        const s = requireTraining();
        const inputIds = a.inputIds as Uint32Array;
        const lossMode = String(a.lossMode ?? "next_token");
        try {
            const loss = lossMode === "per_position"
                ? await s.forwardBackwardPerPosition(inputIds, a.targets as Uint32Array)
                : await s.forwardBackward(inputIds, Number(a.targetId));
            return { loss, step: s.stepNum, lr: s.lr };
        } catch (e) {
            log(`training: forwardBackward step ${s.stepNum} failed (lossMode=${lossMode}, inputLen=${inputIds.length}): ${(e as Error).message}`);
            throw e;
        }
    },

    trainingOptimizerStep: async (a) => {
        void a;
        const s = requireTraining();
        s.optimizerStep();
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
        if (!trainingSession) throw new Error("no training session to finish");
        try {
            model = trainingSession.finish();
        } finally {
            trainingSession = null;
        }
        notify("trainingFinished", {});
        log(`training: finished, Model returned to chat`);
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

// Boot: wait for the host tab's `attach` message carrying the
// MessagePort we'll use for all router traffic. Subsequent self.onmessage
// fires are ignored — everything flows over the attached port.
(self as unknown as DedicatedWorkerGlobalScope).onmessage = (ev: MessageEvent) => {
    const data = ev.data;
    if (!data || typeof data !== "object" || data.type !== "attach" || !data.port) return;
    if (routerPort) return; // already attached
    routerPort = data.port as MessagePort;
    routerPort.addEventListener("message", (e: MessageEvent) => {
        void handleRequest(e.data);
    });
    routerPort.start();
    log(`core: attached to router`);
};
