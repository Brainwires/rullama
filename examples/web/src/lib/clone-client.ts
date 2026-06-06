/// Promise-based client for the voice-cloning worker (clone-worker.ts → StyleTtsClone).
/// Zero-shot: encode a reference clip → 256-d voice vector, then synthesize text in it.

export type StageProgress = (fraction: number, stage: string) => void;

type Pending = { resolve: (v: unknown) => void; reject: (e: unknown) => void; onProgress?: StageProgress };

export class CloneClient {
    private worker: Worker;
    private pending = new Map<number, Pending>();
    private nextId = 1;
    private onProgress?: (frac: number) => void;
    sampleRate = 24000;

    constructor() {
        this.worker = new Worker(new URL("../workers/clone-worker.ts", import.meta.url), { type: "module" });
        this.worker.onmessage = (e: MessageEvent) => {
            const m = e.data as { id: number; ok?: boolean; error?: string; progress?: number; stage?: string };
            if (m.progress !== undefined) {
                // per-call stage progress (encode/synth) routes to that call's handler;
                // load progress (no pending handler) falls back to the client-level one.
                (this.pending.get(m.id)?.onProgress ?? ((f: number) => this.onProgress?.(f)))(m.progress, m.stage ?? "");
                return;
            }
            const p = this.pending.get(m.id);
            if (!p) return;
            this.pending.delete(m.id);
            if (m.ok) p.resolve(m);
            else p.reject(new Error(m.error ?? "clone error"));
        };
    }

    private rpc<T>(type: string, args: Record<string, unknown>, transfer?: Transferable[], onProgress?: StageProgress): Promise<T> {
        const id = this.nextId++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress });
            this.worker.postMessage({ id, type, ...args }, transfer ?? []);
        });
    }

    async load(url: string, size: number, variant: "f32" | "f16" = "f32", onProgress?: (frac: number) => void): Promise<void> {
        this.onProgress = onProgress;
        const r = await this.rpc<{ sampleRate: number }>("load", { url, size, variant });
        this.sampleRate = r.sampleRate;
        this.onProgress = undefined;
    }

    /** Reference clip (24 kHz mono PCM) → 256-d voice vector, with live stage progress. */
    async encodeVoice(pcm24k: Float32Array, onProgress?: StageProgress): Promise<Float32Array> {
        const r = await this.rpc<{ voice: Float32Array }>("encodeVoice", { pcm: pcm24k }, undefined, onProgress);
        return r.voice;
    }

    /** Synthesize text in a voice → 24 kHz PCM, with live stage progress. */
    async synthesize(text: string, voice: Float32Array, onProgress?: StageProgress): Promise<Float32Array> {
        const r = await this.rpc<{ pcm: Float32Array }>("synthesize", { text, voice }, undefined, onProgress);
        return r.pcm;
    }

    dispose(): void {
        this.worker.terminate();
        this.pending.clear();
    }
}

let shared: CloneClient | null = null;
let loadPromise: Promise<CloneClient> | null = null;
let loaded = false;
let loadedVariant: "f32" | "f16" | null = null;

/** Shared singleton cloning client. The chosen GGUF (f32 or f16) downloads once, then
 *  OPFS-cached. Switching variant tears down the worker and reloads the other GGUF. */
export function getSharedClone(url: string, size: number, variant: "f32" | "f16" = "f32", onProgress?: (frac: number) => void): Promise<CloneClient> {
    // A variant switch needs a fresh worker (the wasm clone holds one model).
    if ((loaded || loadPromise) && loadedVariant !== null && loadedVariant !== variant) {
        disposeSharedClone();
    }
    if (loaded && shared) return Promise.resolve(shared);
    if (loadPromise) return loadPromise;
    const c = shared ?? (shared = new CloneClient());
    loadedVariant = variant;
    loadPromise = c.load(url, size, variant, onProgress).then(() => {
        loaded = true;
        return c;
    });
    return loadPromise;
}

export function cloneReady(): boolean {
    return loaded;
}

/** Terminate the shared cloning worker and reset state, freeing its GPU device. Called when the
 *  UI leaves the voice/clone engine so the inference (Gemma) model can have the GPU. Re-created
 *  lazily on the next getSharedClone(). */
export function disposeSharedClone(): void {
    if (shared) { try { shared.dispose(); } catch { /* */ } }
    shared = null;
    loadPromise = null;
    loaded = false;
    loadedVariant = null;
}
