/// Promise-based client for the voice-cloning worker (clone-worker.ts → StyleTtsClone).
/// Zero-shot: encode a reference clip → 256-d voice vector, then synthesize text in it.

type Pending = { resolve: (v: unknown) => void; reject: (e: unknown) => void };

export class CloneClient {
    private worker: Worker;
    private pending = new Map<number, Pending>();
    private nextId = 1;
    private onProgress?: (frac: number) => void;
    sampleRate = 24000;

    constructor() {
        this.worker = new Worker(new URL("../workers/clone-worker.ts", import.meta.url), { type: "module" });
        this.worker.onmessage = (e: MessageEvent) => {
            const m = e.data as { id: number; ok?: boolean; error?: string; progress?: number };
            if (m.progress !== undefined) {
                this.onProgress?.(m.progress);
                return;
            }
            const p = this.pending.get(m.id);
            if (!p) return;
            this.pending.delete(m.id);
            if (m.ok) p.resolve(m);
            else p.reject(new Error(m.error ?? "clone error"));
        };
    }

    private rpc<T>(type: string, args: Record<string, unknown>, transfer?: Transferable[]): Promise<T> {
        const id = this.nextId++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
            this.worker.postMessage({ id, type, ...args }, transfer ?? []);
        });
    }

    async load(url: string, onProgress?: (frac: number) => void): Promise<void> {
        this.onProgress = onProgress;
        const r = await this.rpc<{ sampleRate: number }>("load", { url });
        this.sampleRate = r.sampleRate;
        this.onProgress = undefined;
    }

    /** Reference clip (24 kHz mono PCM) → 256-d voice vector. */
    async encodeVoice(pcm24k: Float32Array): Promise<Float32Array> {
        const r = await this.rpc<{ voice: Float32Array }>("encodeVoice", { pcm: pcm24k });
        return r.voice;
    }

    /** Synthesize text in a voice → 24 kHz PCM. */
    async synthesize(text: string, voice: Float32Array): Promise<Float32Array> {
        const r = await this.rpc<{ pcm: Float32Array }>("synthesize", { text, voice });
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

/** Shared singleton cloning client (the 442 MB GGUF downloads once). */
export function getSharedClone(url: string, onProgress?: (frac: number) => void): Promise<CloneClient> {
    if (loaded && shared) return Promise.resolve(shared);
    if (loadPromise) return loadPromise;
    const c = shared ?? (shared = new CloneClient());
    loadPromise = c.load(url, onProgress).then(() => {
        loaded = true;
        return c;
    });
    return loadPromise;
}

export function cloneReady(): boolean {
    return loaded;
}
