// rullama inference Dedicated Worker.
//
// Owns the wasm `Model` and the `FileSystemSyncAccessHandle` for the lifetime
// of a loaded model. iOS Safari only exposes sync OPFS in Worker contexts,
// and the worker isolates inference from main-thread page-watchdog reapers.
//
// Wire protocol (main → worker), one message per RPC:
//   { requestId, type, ...args }
//
// Wire protocol (worker → main):
//   { requestId, ok: true,  result }
//   { requestId, ok: false, error }
//   { type: "log", args }   (unsolicited)

// The wasm-pack output lives at /pkg/rullama.js (aliased in vite.config.ts).
// Vite resolves this URL at build time and emits the wasm asset alongside.
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

const OPFS_DIR = "rullama-models";

let wasmReady: Promise<unknown> | null = null;
let model: ModelHandle | null = null;
let syncHandle: FileSystemSyncAccessHandle | null = null;
let dbReady: Promise<WasmDbHandle> | null = null;

const DB_NAME = "rullama-chat.db";
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

// Lightweight UUID — `crypto.randomUUID()` is available in DedicatedWorkerGlobalScope.
function newId(): string {
    return crypto.randomUUID();
}

function log(...args: unknown[]) {
    const msg = args.map((a) => String(a)).join(" ");
    (self as DedicatedWorkerGlobalScope).postMessage({ type: "log", args: args.map(String) });
    try {
        fetch("/api/log", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tag: "wkr", msg, ts: Date.now() }),
            keepalive: true,
        }).catch(() => {});
    } catch { /* */ }
}

async function ensureWasm() {
    if (!wasmReady) wasmReady = init();
    return wasmReady;
}

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
    maxContext?: number;
    textOnly?:   boolean;
}
async function handleLoad(args: LoadArgs) {
    await ensureWasm();
    if (model) {
        try { model.free?.(); } catch { /* */ }
        model = null;
    }
    const { readFn, totalBytes: size } = await openSyncReadFn(args.modelKey, args.filename);
    log(`load: Model.loadFromOpfs${args.textOnly ? "TextOnly" : ""} size=${size} max_ctx=${args.maxContext || "default"}`);

    model = args.textOnly
        ? await ModelClass.loadFromOpfsTextOnly(readFn, size, args.maxContext ?? 0)
        : await ModelClass.loadFromOpfs(readFn, size);

    log(`load: ready vocabSize=${model.vocabSize}`);
    return {
        vocabSize:        model.vocabSize,
        hasVision:        model.hasVision,
        hasAudio:         model.hasAudio,
        imageSentinelIds: model.imageSentinelIds() ?? null,
        audioSentinelIds: model.audioSentinelIds() ?? null,
    };
}

function requireModel(): ModelHandle {
    if (!model) throw new Error("no model loaded — call load() first");
    return model;
}

const RPC_TRACE = false;

type Args = Record<string, unknown>;
const RPC: Record<string, (a: Args) => unknown | Promise<unknown>> = {
    load:                 (a) => handleLoad(a as unknown as LoadArgs),
    encode:               (a) => Array.from(requireModel().encode(String(a.text))),
    step:           async (a) => await requireModel().step(Number(a.tokenId)),
    stepWithEmb:    async (a) => await requireModel().stepWithEmbedding(a.embedding as Float32Array),
    stepAndDecode:  async (a) => {
        const m = requireModel();
        const next = await m.step(Number(a.tokenId));
        return { next, isEos: m.isEos(next), str: m.tokenStr(next) ?? null };
    },
    tokenStr:             (a) => requireModel().tokenStr(Number(a.id)) ?? null,
    isEos:                (a) => requireModel().isEos(Number(a.id)),
    reset:                () => requireModel().reset(),
    setSampling:          (a) => requireModel().setSampling(a.opts),
    renderChat:           (a) => requireModel().renderChat(a.messages, !!a.withBos),
    imageSentinelIds:     () => requireModel().imageSentinelIds() ?? null,
    audioSentinelIds:     () => requireModel().audioSentinelIds() ?? null,
    imageSoftTokenCount:  (a) => requireModel().imageSoftTokenCount(Number(a.h), Number(a.w)),
    decodeWav:            (a) => requireModel().decodeWav(a.bytes as Uint8Array),
    encodeImage:    async (a) => await requireModel().encodeImage(a.pixels as Float32Array, Number(a.h), Number(a.w)),
    encodeAudio:    async (a) => await requireModel().encodeAudio(a.pcm as Float32Array),
    free: () => {
        if (model) { try { model.free?.(); } catch { /* */ } model = null; }
        if (syncHandle) { try { syncHandle.close(); } catch { /* */ } syncHandle = null; }
    },

    // ── chat persistence (rsqlite-wasm OPFS-backed SQLite) ─────────────
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
        const model = (a.model as string | null | undefined) ?? null;
        const now   = Date.now();
        db.execParams(
            `INSERT INTO conversations (id, title, model, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?)`,
            [id, title, model, now, now],
        );
        db.flush();
        return { id, title, model, created_at: now, updated_at: now };
    },

    convDelete: async (a) => {
        const db = await ensureDb();
        const id = String(a.id);
        // FK ON DELETE CASCADE removes messages too (PRAGMA foreign_keys=ON
        // is set on open).
        db.execParams(`DELETE FROM conversations WHERE id = ?`, [id]);
        db.flush();
        return true;
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
        return true;
    },

    convTouch: async (a) => {
        const db = await ensureDb();
        const id = String(a.id);
        const titleIfBlank = a.titleIfBlank as string | undefined;
        const now = Date.now();
        if (titleIfBlank !== undefined && titleIfBlank.length > 0) {
            // Only overwrite the auto-title 'New chat', never the user's
            // explicit rename.
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
        // Don't flush per-message during streaming — caller does an explicit
        // flush via dbFlush after each turn.
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

    dbFlush: async () => {
        const db = await ensureDb();
        db.flush();
        return true;
    },
};

self.addEventListener("message", async (ev: MessageEvent<{ requestId: number; type: string } & Args>) => {
    const msg = ev.data;
    if (!msg || typeof msg !== "object" || !msg.type) return;
    const { requestId, type } = msg;
    const handler = RPC[type];
    if (!handler) {
        (self as DedicatedWorkerGlobalScope).postMessage({ requestId, ok: false, error: `unknown RPC type: ${type}` });
        return;
    }
    if (RPC_TRACE) log(`rpc-start ${type}`);
    try {
        const result = await handler(msg as Args);
        if (RPC_TRACE) log(`rpc-done  ${type}`);
        (self as DedicatedWorkerGlobalScope).postMessage({ requestId, ok: true, result });
    } catch (e) {
        const err = (e as Error)?.message ?? String(e);
        log(`rpc ${type} failed: ${err}`);
        (self as DedicatedWorkerGlobalScope).postMessage({ requestId, ok: false, error: err });
    }
});
