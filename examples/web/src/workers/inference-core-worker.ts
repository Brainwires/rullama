// rullama inference *core* — Dedicated Worker.
//
// Spawned exactly once by the SharedWorker shell at
// `inference-worker.ts`. Owns everything that touches OPFS via a
// `FileSystemSyncAccessHandle` (which the OPFS spec restricts to
// Dedicated Workers): the wasm Model, the loaded GGUF read handle, the
// rsqlite-wasm chat DB, and the GGUF download write handle.
//
// Session arbitration, per-tab port plumbing, and notification fanout
// live in the SharedWorker shell. This worker is single-tenant: one
// caller (the shell), one request at a time per stateful RPC.

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
// State
// ───────────────────────────────────────────────────────────────────────

const OPFS_DIR = "rullama-models";
const DB_NAME  = "rullama-chat.db";

let wasmReady: Promise<unknown> | null = null;
let model: ModelHandle | null = null;
let syncHandle: FileSystemSyncAccessHandle | null = null;
let dbReady: Promise<WasmDbHandle> | null = null;

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
// Logging + notification (postMessage to the SharedWorker shell)
// ───────────────────────────────────────────────────────────────────────

function log(...args: unknown[]) {
    const argStrs = args.map((a) => String(a));
    const msg = argStrs.join(" ");
    try { (self as unknown as DedicatedWorkerGlobalScope).postMessage({ type: "log", args: argStrs }); } catch { /* */ }
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
    try {
        (self as unknown as DedicatedWorkerGlobalScope).postMessage({ type: "notify", kind, ...payload });
    } catch { /* */ }
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
    encodeImage: async (a) => await requireModel().encodeImage(a.pixels as Float32Array, Number(a.h), Number(a.w)),
    encodeAudio: async (a) => await requireModel().encodeAudio(a.pcm as Float32Array),
    reset:        (a) => { void a; return requireModel().reset(); },
    setSampling:  (a) => requireModel().setSampling(a.opts),

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
};

// ───────────────────────────────────────────────────────────────────────
// Dispatch
// ───────────────────────────────────────────────────────────────────────

async function handleRequest(msg: { requestId: number; type: string } & Args) {
    if (!msg || typeof msg !== "object" || !msg.type) return;
    const { requestId, type } = msg;
    const handler = RPC[type];
    const post = (self as unknown as DedicatedWorkerGlobalScope).postMessage.bind(self);
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
        post({ requestId, ok: false, error: err });
    }
}

(self as unknown as DedicatedWorkerGlobalScope).onmessage = (ev: MessageEvent) => {
    void handleRequest(ev.data);
};
