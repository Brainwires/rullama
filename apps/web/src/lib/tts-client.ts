/// Promise-based client for the dedicated TTS worker (tts-worker.ts).

export interface TtsClip {
    pcm: Float32Array;
    sampleRate: number;
    text: string;
    voice: string;
    ts: number;
}

type Pending = { resolve: (v: unknown) => void; reject: (e: unknown) => void };

export class TtsClient {
    private worker: Worker;
    private pending = new Map<number, Pending>();
    private nextId = 1;
    private onProgress?: (frac: number, stage?: string) => void;
    sampleRate = 24000;

    constructor() {
        this.worker = new Worker(new URL("../workers/tts-worker.ts", import.meta.url), { type: "module" });
        this.worker.onmessage = (e: MessageEvent) => {
            const m = e.data as { id: number; ok?: boolean; error?: string; progress?: number; stage?: string; pcm?: Float32Array; sampleRate?: number };
            if (m.progress !== undefined) {
                this.onProgress?.(m.progress, m.stage);
                return;
            }
            const p = this.pending.get(m.id);
            if (!p) return;
            this.pending.delete(m.id);
            if (m.ok) p.resolve(m);
            else p.reject(new Error(m.error ?? "tts error"));
        };
    }

    private rpc<T>(type: string, args: Record<string, unknown>): Promise<T> {
        const id = this.nextId++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
            this.worker.postMessage({ id, type, ...args });
        });
    }

    /** Download + load the Kokoro GGUF and lexicon (once). */
    async load(url: string, onProgress?: (frac: number) => void): Promise<void> {
        this.onProgress = onProgress;
        const r = await this.rpc<{ sampleRate: number }>("load", { url });
        this.sampleRate = r.sampleRate;
        this.onProgress = undefined;
    }

    async synthesize(
        text: string,
        voice: string,
        onProgress?: (frac: number, stage?: string) => void,
    ): Promise<Float32Array> {
        this.onProgress = onProgress;
        try {
            const r = await this.rpc<{ pcm: Float32Array }>("synthesize", { text, voice });
            return r.pcm;
        } finally {
            this.onProgress = undefined;
        }
    }

    // ---- gradient-free voice training ----
    async trainBegin(targetPcm: Float32Array, refText: string, initVoice = "af_heart"): Promise<void> {
        await this.rpc("trainBegin", { targetPcm, refText, initVoice });
    }
    async trainStep(): Promise<number> {
        const r = await this.rpc<{ loss: number }>("trainStep", {});
        return r.loss;
    }
    async trainedVoice(): Promise<Float32Array> {
        const r = await this.rpc<{ voice: Float32Array }>("trainedVoice", {});
        return r.voice;
    }
    async synthesizeWithVoice(text: string, voiceVec: Float32Array): Promise<Float32Array> {
        const r = await this.rpc<{ pcm: Float32Array }>("synthesizeWithVoice", { text, voiceVec });
        return r.pcm;
    }

    /** Cooperatively cancel the in-flight synthesis (the worker flips a wasm flag
     *  that the running synth polls at its next stage boundary). The pending
     *  `synthesize` then resolves with an EMPTY buffer — the worker stays loaded. */
    cancel(): void {
        this.worker.postMessage({ type: "cancel" });
    }

    dispose(): void {
        this.worker.terminate();
        this.pending.clear();
    }
}

// Shared singleton so the Voice tab and the chat "speak" button reuse one loaded
// model + Worker (the 164 MB GGUF downloads once).
let shared: TtsClient | null = null;
let loadPromise: Promise<TtsClient> | null = null;
let loaded = false;

/** Returns the shared client, loading the model on first call. */
export function getSharedTts(url: string, onProgress?: (frac: number) => void): Promise<TtsClient> {
    if (loaded && shared) return Promise.resolve(shared);
    if (loadPromise) return loadPromise;
    const c = shared ?? (shared = new TtsClient());
    loadPromise = c.load(url, onProgress).then(() => {
        loaded = true;
        return c;
    });
    return loadPromise;
}

/** Whether the shared client is loaded and ready. */
export function ttsReady(): boolean {
    return loaded;
}

/** Cancel the shared Kokoro client's in-flight synthesis (no-op if not running). */
export function cancelSharedTts(): void {
    shared?.cancel();
}

/** Terminate the shared Kokoro TTS worker and reset state, freeing its GPU device. Called when the
 *  UI leaves the voice engine so the inference (Gemma) model can have the GPU. Re-created lazily on
 *  the next getSharedTts(). */
export function disposeSharedTts(): void {
    if (shared) { try { shared.dispose(); } catch { /* */ } }
    shared = null;
    loadPromise = null;
    loaded = false;
}
