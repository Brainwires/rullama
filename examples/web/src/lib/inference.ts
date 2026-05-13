// WorkerClient — thin RPC wrapper over the inference worker.
//
// The wasm Model lives inside a Dedicated Worker (see
// `src/workers/inference-worker.ts`). Every method here is a postMessage
// round-trip; cached meta (vocabSize / hasVision / …) is fetched once on
// `load()` so the React tree can read it synchronously thereafter.

import type { ChatMessage, SamplingOptions } from "@/lib/types";
import InferenceWorker from "@/workers/inference-worker?worker";
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

type WorkerMsg =
    | { type: "log"; args: string[] }
    | { requestId: number; ok: true;  result: unknown }
    | { requestId: number; ok: false; error: string };

/** Heuristic: does this error message look like the SW serving a stale
 *  reference to a hashed asset that no longer exists? Browsers phrase the
 *  same failure differently — Chrome says "Failed to fetch dynamically
 *  imported module", Safari says "Importing a module script failed",
 *  Firefox says "Loading chunk N failed". WebAssembly.instantiate errors
 *  on a 404'd .wasm fall in here too. */
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
    private worker: Worker;
    private pending = new Map<number, Pending>();
    private nextId  = 1;
    private meta:   ModelMeta | null = null;

    /** Last log line from the worker, useful in dev consoles. */
    public onLog?: (line: string) => void;

    constructor() {
        try {
            this.worker = new InferenceWorker();
        } catch (e) {
            // Construction itself failed — Vite's worker glue couldn't
            // resolve the hashed URL. Surface the restart overlay and
            // rethrow so callers don't end up with a half-built client.
            requestRestart("the inference worker failed to construct");
            throw e;
        }
        this.worker.addEventListener("message", (ev: MessageEvent<WorkerMsg>) => {
            const m = ev.data;
            if ("type" in m && m.type === "log") {
                const line = m.args.join(" ");
                console.log("[inference-worker]", line);
                this.onLog?.(line);
                return;
            }
            if ("requestId" in m) {
                const p = this.pending.get(m.requestId);
                if (!p) return;
                this.pending.delete(m.requestId);
                if (m.ok) p.resolve(m.result);
                else      p.reject(new Error(m.error));
            }
        });
        this.worker.addEventListener("error", (ev) => {
            const msg = ev.message || String(ev);
            console.error("[inference-worker] worker error:", msg);
            // A worker boot failure after a deploy is the canonical
            // "stale tab" symptom: the cached index-*.js asks for an
            // inference-worker-*.js URL that's no longer in the SW's
            // precache (cleaned up by the new build) and no longer on
            // disk on the server (replaced by the new hash). Surface
            // the restart overlay so the user gets a single click to
            // recover instead of a cryptic console error.
            if (looksLikeStaleAssetError(msg)) {
                requestRestart("the inference worker failed to load");
            }
        });
        this.worker.addEventListener("messageerror", () => {
            requestRestart("the inference worker failed to start");
        });
    }

    private rpc<T = unknown>(type: string, args: Record<string, unknown> = {}): Promise<T> {
        const requestId = this.nextId++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(requestId, {
                resolve: (v) => resolve(v as T),
                reject,
            });
            this.worker.postMessage({ requestId, type, ...args });
        });
    }

    async load(modelKey: string, filename: string, opts: { maxContext?: number; textOnly?: boolean } = {}): Promise<ModelMeta> {
        this.meta = await this.rpc<ModelMeta>("load", {
            modelKey, filename,
            maxContext: opts.maxContext ?? 0,
            textOnly:   !!opts.textOnly,
        });
        return this.meta;
    }

    get vocabSize() { return this.meta?.vocabSize; }
    get hasVision() { return !!this.meta?.hasVision; }
    get hasAudio()  { return !!this.meta?.hasAudio; }
    imageSentinelIds() { return this.meta?.imageSentinelIds ?? null; }
    audioSentinelIds() { return this.meta?.audioSentinelIds ?? null; }

    encode(text: string): Promise<Uint32Array> {
        return this.rpc<number[]>("encode", { text }).then((arr) => new Uint32Array(arr));
    }
    step(tokenId: number): Promise<number> { return this.rpc("step", { tokenId }); }
    stepWithEmbedding(embedding: Float32Array): Promise<number> {
        return this.rpc("stepWithEmb", { embedding });
    }
    stepAndDecode(tokenId: number): Promise<{ next: number; isEos: boolean; str: string | null }> {
        return this.rpc("stepAndDecode", { tokenId });
    }
    tokenStr(id: number): Promise<string | null> { return this.rpc("tokenStr", { id }); }
    isEos(id: number): Promise<boolean> { return this.rpc("isEos", { id }); }
    reset(): Promise<void> { return this.rpc("reset"); }
    setSampling(opts: SamplingOptions): Promise<void> { return this.rpc("setSampling", { opts }); }
    renderChat(messages: ChatMessage[], withBos: boolean): Promise<string> {
        return this.rpc("renderChat", { messages, withBos });
    }
    free(): Promise<void> { return this.rpc("free"); }

    // ── multimodal ─────────────────────────────────────────────────
    encodeImage(pixels: Float32Array, h: number, w: number): Promise<Float32Array> {
        return this.rpc("encodeImage", { pixels, h, w });
    }
    imageSoftTokenCount(h: number, w: number): Promise<number> {
        return this.rpc("imageSoftTokenCount", { h, w });
    }

    // ── chat persistence (rsqlite-wasm OPFS-backed SQLite) ──────────
    dbInit(): Promise<boolean> { return this.rpc("dbInit"); }
    convList(): Promise<ConversationRow[]> { return this.rpc("convList"); }
    convCreate(opts: { id?: string; title?: string; model?: string | null } = {}): Promise<ConversationRow> {
        return this.rpc("convCreate", opts as Record<string, unknown>);
    }
    convDelete(id: string): Promise<boolean> { return this.rpc("convDelete", { id }); }
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
    dbFlush(): Promise<boolean> { return this.rpc("dbFlush"); }
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

/** Singleton — one worker per page. */
let _client: WorkerClient | null = null;
export function getClient(): WorkerClient {
    if (!_client) _client = new WorkerClient();
    return _client;
}
