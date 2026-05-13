// rullama inference SharedWorker.
//
// One per origin. Owns the wasm Model, the loaded GGUF sync handle, the
// rsqlite-wasm chat DB, and the model-download (ensureModel) path. Tabs
// open a MessagePort and route all heavy / OPFS-touching work through here.
//
// Why SharedWorker:
//   OPFS's `FileSystemSyncAccessHandle` is single-writer by spec — a second
//   tab opening the same file gets `NoModificationAllowedError`. By keeping
//   the handles in one origin-scoped worker, every tab connects to the same
//   handles. Concurrent inference is serialized through a session queue.
//
// Wire protocol (port → worker), one message per RPC:
//   { requestId, type, ...args }
//
// Wire protocol (worker → port):
//   { requestId, ok: true,  result }                     — RPC reply
//   { requestId, ok: false, error }                      — RPC failure
//   { type: "log",    args }                             — debug fanout
//   { type: "notify", kind, ...payload }                 — cross-tab event

// The wasm-pack output lives at /pkg/rullama.js (aliased in vite.config.ts).
// @ts-expect-error — generated bundle, no .d.ts
import init, { Model, WasmDatabase } from "/pkg/rullama.js";

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
    encodeImage(pixels: Float32Array, h: number, w: number): Promise<Float32Array>;
    encodeAudio(pcm: Float32Array): Promise<Float32Array>;
}
interface ModelStatic {
    loadFromOpfsTextOnly(
        readFn: (offset: number, length: number) => Uint8Array | Promise<Uint8Array>,
        totalBytes: number,
        maxContext: number,
    ): Promise<ModelHandle>;
    loadFromOpfs(
        readFn: (offset: number, length: number) => Uint8Array | Promise<Uint8Array>,
        totalBytes: number,
    ): Promise<ModelHandle>;
}
const ModelClass = Model as unknown as ModelStatic;

// ───────────────────────────────────────────────────────────────────────
// State (singletons, shared across all connected ports)
// ───────────────────────────────────────────────────────────────────────

const OPFS_DIR = "rullama-models";
const DB_NAME  = "rullama-chat.db";

let wasmReady: Promise<unknown> | null = null;
let model: ModelHandle | null = null;
let syncHandle: FileSystemSyncAccessHandle | null = null;
let dbReady: Promise<WasmDbHandle> | null = null;

interface LoadedModelInfo {
    /** Human-friendly identifier the tab passed in (e.g. "gemma4:e2b").
     *  Worker keeps it solely for the modelLoaded / meta broadcasts so
     *  the UI doesn't have to map modelKey back through the catalog. */
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

const PORTS = new Set<MessagePort>();
const PORT_LAST_SEEN = new WeakMap<MessagePort, number>();

// Session arbitration — serializes inference (one Model, one KV cache).
interface ActiveSession { sid: number; port: MessagePort; }
let active: ActiveSession | null = null;
let nextSid = 1;
interface Waiter {
    port: MessagePort;
    abortToken: string;
    resolve: (sid: number) => void;
    reject:  (e: Error) => void;
}
const queue: Waiter[] = [];

// In-flight ensureModel downloads, keyed by `modelKey/filename`.
// Coalesce parallel calls — second caller awaits the first's promise.
interface DownloadInflight {
    promise: Promise<{ totalBytes: number; fromCache: boolean }>;
}
const inflight = new Map<string, DownloadInflight>();

const RPC_TRACE = false;

// ───────────────────────────────────────────────────────────────────────
// Logging + notification fanout
// ───────────────────────────────────────────────────────────────────────

function log(...args: unknown[]) {
    const argStrs = args.map((a) => String(a));
    const msg = argStrs.join(" ");
    for (const port of PORTS) {
        try { port.postMessage({ type: "log", args: argStrs }); } catch { /* port closed */ }
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
    const msg = { type: "notify", kind, ...payload };
    for (const port of PORTS) {
        try { port.postMessage(msg); } catch { /* port closed */ }
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
    name?:       string;   // human-friendly, for broadcast UX only
    maxContext?: number;
    textOnly?:   boolean;
}

async function handleLoad(args: LoadArgs): Promise<LoadedModelInfo> {
    await ensureWasm();
    // Short-circuit: if the requested model is already the one we have
    // loaded, skip the free+reload. Crucial for auto-load on page reload
    // (the SharedWorker outlives the tab) and for two tabs both
    // independently auto-loading the same model.
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
        : await ModelClass.loadFromOpfs(readFn, size);

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
    return model;
}

// ───────────────────────────────────────────────────────────────────────
// ensureModel — download GGUF to OPFS via streaming write
// (Replaces the per-tab opfs-writer-worker. Coalesces parallel callers.)
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
        // GGUF magic check (read first 4 bytes via async getFile API — concurrent-safe).
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
    const key = `${modelKey}/${filename}`;

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

    // Reuse a model file write handle only if the model file isn't *also*
    // being read for inference (which holds syncHandle). The two would race
    // on the same OPFS file; in practice we never download a model that's
    // currently loaded, but be defensive.
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
    } finally {
        inflight.delete(key);
    }
}

async function handleEnsureModel(args: EnsureArgs): Promise<{ totalBytes: number; fromCache: boolean }> {
    const key = `${args.modelKey}/${args.filename}`;
    const existing = inflight.get(key);
    if (existing) return existing.promise;
    const promise = doDownload(args);
    inflight.set(key, { promise });
    return promise;
}

// ───────────────────────────────────────────────────────────────────────
// Session arbitration
// ───────────────────────────────────────────────────────────────────────

interface AcquireArgs {
    abortToken: string;
}

function acquireSession(args: AcquireArgs, port: MessagePort): Promise<number> {
    return new Promise<number>((resolve, reject) => {
        if (!active) {
            active = { sid: ++nextSid, port };
            resolve(active.sid);
            return;
        }
        queue.push({
            port,
            abortToken: args.abortToken,
            resolve,
            reject,
        });
    });
}

function releaseSession(sid: number, port: MessagePort): boolean {
    if (!active || active.sid !== sid || active.port !== port) {
        // Stale release — ignore (e.g. caller's tab held until close, the
        // disconnect handler already cleared it).
        return false;
    }
    active = null;
    wakeNext();
    return true;
}

function cancelAcquire(abortToken: string): boolean {
    const idx = queue.findIndex((w) => w.abortToken === abortToken);
    if (idx < 0) return false;
    const w = queue.splice(idx, 1)[0];
    w.reject(new Error("aborted"));
    return true;
}

function wakeNext() {
    if (active) return;
    const w = queue.shift();
    if (!w) return;
    active = { sid: ++nextSid, port: w.port };
    w.resolve(active.sid);
}

function requireSession(port: MessagePort, sid: unknown) {
    const n = Number(sid);
    if (!active) throw new Error("no active session");
    if (active.sid !== n) throw new Error(`session mismatch: held=${active.sid} called=${n}`);
    if (active.port !== port) throw new Error("session not owned by this port");
}

// ───────────────────────────────────────────────────────────────────────
// Port lifecycle
// ───────────────────────────────────────────────────────────────────────

function disconnectPort(port: MessagePort, reason: string) {
    PORTS.delete(port);
    PORT_LAST_SEEN.delete(port);
    // Release any session held by this port.
    if (active && active.port === port) {
        const sid = active.sid;
        active = null;
        log(`port disconnect (${reason}) released session ${sid}`);
        wakeNext();
    }
    // Drop any queued waiters from this port.
    for (let i = queue.length - 1; i >= 0; i--) {
        if (queue[i].port === port) {
            queue[i].reject(new Error("port disconnected"));
            queue.splice(i, 1);
        }
    }
    try { port.close(); } catch { /* */ }
}

// Heartbeat GC — ports that don't ping in HEARTBEAT_DEAD_MS are presumed
// crashed/closed (browsers don't fire onclose on MessagePort).
const HEARTBEAT_GC_INTERVAL = 15_000;
const HEARTBEAT_DEAD_MS      = 30_000;
setInterval(() => {
    const now = Date.now();
    for (const port of PORTS) {
        const seen = PORT_LAST_SEEN.get(port) ?? now;
        if (now - seen > HEARTBEAT_DEAD_MS) {
            disconnectPort(port, "heartbeat timeout");
        }
    }
}, HEARTBEAT_GC_INTERVAL);

// ───────────────────────────────────────────────────────────────────────
// RPC table
// ───────────────────────────────────────────────────────────────────────

type Args = Record<string, unknown>;
type Handler = (a: Args, port: MessagePort) => unknown | Promise<unknown>;

const RPC: Record<string, Handler> = {
    // ── Session arbitration ─────────────────────────────────────────────
    acquireSession: (a, port) => acquireSession(a as unknown as AcquireArgs, port),
    releaseSession: (a, port) => releaseSession(Number(a.sid), port),
    cancelAcquire:  (a)       => cancelAcquire(String(a.abortToken)),

    // ── Lifecycle ──────────────────────────────────────────────────────
    ping: (_a, port) => {
        PORT_LAST_SEEN.set(port, Date.now());
        return true;
    },
    disconnect: (_a, port) => {
        disconnectPort(port, "client disconnect");
        return true;
    },

    // ── Model lifecycle (session-scoped) ────────────────────────────────
    load: async (a, port) => {
        requireSession(port, a.sid);
        return handleLoad(a as unknown as LoadArgs);
    },
    free: (a, port) => {
        requireSession(port, a.sid);
        if (model) { try { model.free?.(); } catch { /* */ } model = null; }
        if (syncHandle) { try { syncHandle.close(); } catch { /* */ } syncHandle = null; }
        if (loadedInfo) { loadedInfo = null; notify("modelFreed", {}); }
    },

    // ── ensureModel (stateless wrt session; coalesces) ──────────────────
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

    // ── Stateful inference (session-scoped) ─────────────────────────────
    step: async (a, port) => {
        requireSession(port, a.sid);
        return await requireModel().step(Number(a.tokenId));
    },
    stepWithEmb: async (a, port) => {
        requireSession(port, a.sid);
        return await requireModel().stepWithEmbedding(a.embedding as Float32Array);
    },
    stepAndDecode: async (a, port) => {
        requireSession(port, a.sid);
        const m = requireModel();
        const next = await m.step(Number(a.tokenId));
        return { next, isEos: m.isEos(next), str: m.tokenStr(next) ?? null };
    },
    encodeImage: async (a, port) => {
        requireSession(port, a.sid);
        return await requireModel().encodeImage(a.pixels as Float32Array, Number(a.h), Number(a.w));
    },
    encodeAudio: async (a, port) => {
        requireSession(port, a.sid);
        return await requireModel().encodeAudio(a.pcm as Float32Array);
    },
    reset: (a, port) => {
        requireSession(port, a.sid);
        return requireModel().reset();
    },
    setSampling: (a, port) => {
        requireSession(port, a.sid);
        return requireModel().setSampling(a.opts);
    },

    // ── Worker meta (stateless) ─────────────────────────────────────────
    currentMeta: () => ({
        loaded: loadedInfo,
        activeSessionPortHeld: active != null,
    }),

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
        // No broadcast on every msgInsert — they fire during streaming and
        // would storm the other tab. Caller emits a single `convTouch` at
        // the end of each turn which broadcasts.
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
};

// ───────────────────────────────────────────────────────────────────────
// Per-port request dispatch
// ───────────────────────────────────────────────────────────────────────

async function handleRequest(
    port: MessagePort,
    msg: { requestId: number; type: string } & Args,
) {
    if (!msg || typeof msg !== "object" || !msg.type) return;
    const { requestId, type } = msg;
    PORT_LAST_SEEN.set(port, Date.now());
    const handler = RPC[type];
    if (!handler) {
        port.postMessage({ requestId, ok: false, error: `unknown RPC type: ${type}` });
        return;
    }
    if (RPC_TRACE) log(`rpc-start ${type}`);
    try {
        const result = await handler(msg as Args, port);
        if (RPC_TRACE) log(`rpc-done  ${type}`);
        port.postMessage({ requestId, ok: true, result });
    } catch (e) {
        const err = (e as Error)?.message ?? String(e);
        log(`rpc ${type} failed: ${err}`);
        port.postMessage({ requestId, ok: false, error: err });
    }
}

// ───────────────────────────────────────────────────────────────────────
// SharedWorker entry — one onconnect per tab
// ───────────────────────────────────────────────────────────────────────

(self as unknown as SharedWorkerGlobalScope).onconnect = (e: MessageEvent) => {
    const port = (e.ports as MessagePort[])[0];
    if (!port) return;
    port.start();
    PORTS.add(port);
    PORT_LAST_SEEN.set(port, Date.now());
    port.addEventListener("message", (ev: MessageEvent) => {
        void handleRequest(port, ev.data);
    });
    // Push initial state so the freshly connected tab can skip "Load a
    // model" if a model is already active in the worker.
    port.postMessage({
        type: "notify",
        kind: "meta",
        loaded: loadedInfo,
    });
};
