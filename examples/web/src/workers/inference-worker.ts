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
import init, { Model } from "/pkg/rullama.js";

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
