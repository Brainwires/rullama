/// Dedicated voice-cloning worker: owns the StyleTtsClone wasm handle (StyleTTS2-LibriTTS),
/// loads the GGUF + G2P lexicon once, then encodes a reference clip → voice vector and
/// synthesizes text in that voice. Separate from the Kokoro tts-worker (different engine).

// @ts-expect-error — generated bundle, no .d.ts
import init, { StyleTtsClone, ttsRequestCancel } from "/pkg/rullama.js";
import { downloadToOpfs, openOpfsReadFn } from "./opfs-download";

let clone: StyleTtsClone | null = null;

interface Req {
    id: number;
    type: "load" | "encodeVoice" | "synthesize" | "cancel";
    url?: string;
    size?: number;
    variant?: "f32" | "f16";
    pcm?: Float32Array;
    text?: string;
    voice?: Float32Array;
}

function post(msg: Record<string, unknown>, transfer?: Transferable[]) {
    (self as unknown as Worker).postMessage(msg, transfer ?? []);
}

// OPFS layout: the clone model nests under the same root as the inference models, so both
// engines share one cache tree (root/rullama-models/<modelKey>/<filename>).
const OPFS_DIR = "rullama-models";
// Per-variant cache keys so the f32 and f16 GGUFs cache independently.
const modelKey = (variant: "f32" | "f16") => `styletts2-libritts-${variant}`;
const modelFile = (variant: "f32" | "f16") => `styletts2-libritts-${variant}.gguf`;

/**
 * Ensure the cloning GGUF is cached in OPFS (downloading/resuming via the shared resumable
 * downloader — the same screen-lock-surviving Range path the inference engine uses), then return
 * a range-read callback over it for the wasm `loadStreaming` loader. The 543 MB f32 model goes
 * straight to disk and is pulled one tensor at a time; it never lands in the JS/wasm heap (iOS
 * jetsam safe). One download, ever — a complete file short-circuits; a partial one resumes.
 */
async function openGgufStreaming(url: string, expectedSize: number, variant: "f32" | "f16", id: number): Promise<{ readFn: (o: number, l: number) => Uint8Array; size: number; close: () => void }> {
    const key = modelKey(variant);
    const file = modelFile(variant);
    const { fromCache } = await downloadToOpfs({
        opfsDir: OPFS_DIR,
        url,
        modelKey: key,
        filename: file,
        expectedSize,
        onProgress: (p) => post({ id, progress: p.totalBytes ? p.bytesWritten / p.totalBytes : 0 }),
        log: (m) => console.log("[clone]", m),
    });
    console.log(fromCache ? `[clone] ${variant} model from OPFS cache (no download)` : `[clone] ${variant} model downloaded to OPFS`);
    if (fromCache) post({ id, progress: 1 });
    return openOpfsReadFn(OPFS_DIR, key, file);
}

self.onmessage = async (e: MessageEvent<Req>) => {
    const { id, type } = e.data;
    // Cancel is a free wasm call (no model borrow), so it lands even while a
    // synthesize is mid-flight — the running synth aborts at its next stage.
    if (type === "cancel") {
        try { ttsRequestCancel(); } catch { /* not loaded yet */ }
        return;
    }
    try {
        if (type === "load") {
            await init();
            const variant = e.data.variant ?? "f32";
            const { readFn, size, close } = await openGgufStreaming(e.data.url!, e.data.size ?? 0, variant, id);
            // Streams the GGUF one tensor at a time over OPFS — never bulk-loads into linear
            // memory (the iOS-jetsam-safe path; see openGgufStreaming). The f16 variant keeps
            // conv weights f16 (host + GPU) — ~half the footprint, at f16 precision.
            clone = variant === "f16"
                ? await StyleTtsClone.loadStreamingF16(readFn, size)
                : await StyleTtsClone.loadStreaming(readFn, size);
            // Weights are now resident in the wasm map; release the exclusive OPFS lock so a
            // future worker (PWA reload) can reopen the file without waiting out the GC window.
            close();
            const [gold, silver] = await Promise.all([
                fetch("/tts/us_gold.json").then((r) => r.arrayBuffer()),
                fetch("/tts/us_silver.json").then((r) => r.arrayBuffer()),
            ]);
            clone.setLexicon(new Uint8Array(gold), new Uint8Array(silver));
            post({ id, ok: true, sampleRate: clone.sampleRate });
        } else if (type === "encodeVoice") {
            if (!clone) throw new Error("clone engine not loaded");
            // The progress callback fires synchronously mid-computation; postMessage from
            // within the blocking wasm call still reaches the main thread for a live bar/log.
            const onProg = (frac: number, stage: string) => {
                console.log(`[clone] encode ${(frac * 100) | 0}% — ${stage}`);
                post({ id, progress: frac, stage });
            };
            const voice = await clone.encodeVoice(e.data.pcm!, onProg);
            post({ id, ok: true, voice }, [voice.buffer]);
        } else if (type === "synthesize") {
            if (!clone) throw new Error("clone engine not loaded");
            const onProg = (frac: number, stage: string) => {
                console.log(`[clone] synth ${(frac * 100) | 0}% — ${stage}`);
                post({ id, progress: frac, stage });
            };
            const pcm = await clone.synthesize(e.data.text!, e.data.voice!, onProg);
            post({ id, ok: true, pcm }, [pcm.buffer]);
        }
    } catch (err) {
        post({ id, ok: false, error: String(err) });
    }
};
