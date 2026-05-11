// WorkerClient — thin RPC wrapper over the inference worker.
//
// The wasm Model lives inside a Dedicated Worker (see
// `src/workers/inference-worker.ts`). Every method here is a postMessage
// round-trip; cached meta (vocabSize / hasVision / …) is fetched once on
// `load()` so the React tree can read it synchronously thereafter.

import type { ChatMessage, SamplingOptions } from "@/lib/types";
import InferenceWorker from "@/workers/inference-worker?worker";

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

export class WorkerClient {
    private worker: Worker;
    private pending = new Map<number, Pending>();
    private nextId  = 1;
    private meta:   ModelMeta | null = null;

    /** Last log line from the worker, useful in dev consoles. */
    public onLog?: (line: string) => void;

    constructor() {
        this.worker = new InferenceWorker();
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
            console.error("[inference-worker] worker error:", ev.message || ev);
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
}

/** Singleton — one worker per page. */
let _client: WorkerClient | null = null;
export function getClient(): WorkerClient {
    if (!_client) _client = new WorkerClient();
    return _client;
}
