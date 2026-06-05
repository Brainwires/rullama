/// Dedicated voice-cloning worker: owns the StyleTtsClone wasm handle (StyleTTS2-LibriTTS),
/// loads the GGUF + G2P lexicon once, then encodes a reference clip → voice vector and
/// synthesizes text in that voice. Separate from the Kokoro tts-worker (different engine).

// @ts-expect-error — generated bundle, no .d.ts
import init, { StyleTtsClone } from "/pkg/rullama.js";

let clone: StyleTtsClone | null = null;

interface Req {
    id: number;
    type: "load" | "encodeVoice" | "synthesize";
    url?: string;
    size?: number;
    pcm?: Float32Array;
    text?: string;
    voice?: Float32Array;
}

function post(msg: Record<string, unknown>, transfer?: Transferable[]) {
    (self as unknown as Worker).postMessage(msg, transfer ?? []);
}

const CACHE_DIR = "rullama-clone-model";
const CACHE_NAME = "styletts2-libritts.gguf";

/** Range-read callback into the OPFS file (the wasm `loadStreaming` consumer). */
type ReadFn = (offset: number, length: number) => Uint8Array;

let syncHandle: FileSystemSyncAccessHandle | null = null;

/** Open a `FileSystemSyncAccessHandle`, retrying while a previous worker still holds the
 *  exclusive lock (iOS Safari can take a few seconds to GC an orphaned handle). */
async function openSyncHandleWithRetry(fh: FileSystemFileHandle, budgetMs = 15_000): Promise<FileSystemSyncAccessHandle> {
    const start = Date.now();
    let attempt = 0;
    for (;;) {
        try {
            return await fh.createSyncAccessHandle();
        } catch (e) {
            attempt++;
            const elapsed = Date.now() - start;
            if (elapsed >= budgetMs) {
                throw new Error(`OPFS sync handle locked by a previous worker (${attempt} tries, ${elapsed}ms). The data is intact — force-quit the tab and reopen. Underlying: ${(e as Error)?.message ?? e}`);
            }
            await new Promise((r) => setTimeout(r, Math.min(1500, 100 * 2 ** (attempt - 1))));
        }
    }
}

/**
 * Ensure the cloning GGUF is in OPFS and return a range-read callback over it — streaming the
 * 543 MB f32 model **straight to disk and never into the JS/wasm heap**. A single sync access
 * handle does both: we stream the download into it chunk-by-chunk (no `chunks[]`, no whole-file
 * `arrayBuffer()`), then serve `read(offset,len)` from it so the wasm loader pulls one tensor at
 * a time. This is what keeps the load under the iOS jetsam budget — the old path materialized the
 * model ~3× (JS bytes → wasm Vec → weight map) and tripped jetsam on load.
 */
async function openGgufStreaming(url: string, expectedSize: number, id: number): Promise<{ readFn: ReadFn; size: number }> {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(CACHE_DIR, { create: true });
    const fh = await dir.getFileHandle(CACHE_NAME, { create: true });

    if (syncHandle) { try { syncHandle.close(); } catch { /* */ } syncHandle = null; }
    const handle = await openSyncHandleWithRetry(fh);
    syncHandle = handle;

    // Cache hit: valid size + GGUF magic ("GGUF") → skip download.
    const have = handle.getSize();
    const sizeOk = expectedSize ? have === expectedSize : have > 1_000_000;
    let valid = false;
    if (sizeOk) {
        const head = new Uint8Array(4);
        handle.read(head, { at: 0 });
        valid = head[0] === 0x47 && head[1] === 0x47 && head[2] === 0x55 && head[3] === 0x46;
    }

    if (!valid) {
        console.log("[clone] downloading model →", url);
        handle.truncate(0);
        const resp = await fetch(url);
        const total = Number(resp.headers.get("content-length") ?? expectedSize ?? 0);
        const reader = resp.body!.getReader();
        let off = 0;
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            handle.write(value, { at: off }); // straight to disk; chunk is freed next iteration
            off += value.length;
            if (total) post({ id, progress: off / total });
        }
        handle.flush();
        console.log(`[clone] model streamed to OPFS (${off} bytes)`);
    } else {
        console.log("[clone] model loaded from OPFS cache (no download)");
        post({ id, progress: 1 });
    }

    const size = handle.getSize();
    const readFn: ReadFn = (offset, length) => {
        const buf = new Uint8Array(length);
        handle.read(buf, { at: offset });
        return buf;
    };
    return { readFn, size };
}

self.onmessage = async (e: MessageEvent<Req>) => {
    const { id, type } = e.data;
    try {
        if (type === "load") {
            await init();
            const { readFn, size } = await openGgufStreaming(e.data.url!, e.data.size ?? 0, id);
            // Streams the GGUF one tensor at a time over OPFS — never bulk-loads into linear
            // memory (the iOS-jetsam-safe path; see openGgufStreaming).
            clone = await StyleTtsClone.loadStreaming(readFn, size);
            // Weights are now resident in the wasm map; release the exclusive OPFS lock so a
            // future worker (PWA reload) can reopen the file without waiting out the GC window.
            if (syncHandle) { try { syncHandle.close(); } catch { /* */ } syncHandle = null; }
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
