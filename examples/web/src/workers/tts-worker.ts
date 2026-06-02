/// Dedicated TTS worker: owns the KokoroTts wasm handle, loads the Kokoro GGUF +
/// G2P lexicon once, and synthesizes text → PCM. Kept separate from the chat
/// inference-core-worker so it stays simple and the TTS model has its own handle.

// @ts-expect-error — generated bundle, no .d.ts
import init, { KokoroTts } from "/pkg/rullama.js";

let tts: KokoroTts | null = null;

interface Req {
    id: number;
    type: "load" | "synthesize" | "synthesizePhonemes";
    url?: string;
    text?: string;
    voice?: string;
}

function post(msg: Record<string, unknown>, transfer?: Transferable[]) {
    (self as unknown as Worker).postMessage(msg, transfer ?? []);
}

self.onmessage = async (e: MessageEvent<Req>) => {
    const { id, type } = e.data;
    try {
        if (type === "load") {
            await init();
            // Stream the GGUF with progress, then hand the full bytes to wasm.
            const resp = await fetch(e.data.url!);
            const total = Number(resp.headers.get("content-length") ?? 0);
            const reader = resp.body!.getReader();
            const chunks: Uint8Array[] = [];
            let got = 0;
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                chunks.push(value);
                got += value.length;
                if (total) post({ id, progress: got / total });
            }
            const bytes = new Uint8Array(got);
            let off = 0;
            for (const c of chunks) {
                bytes.set(c, off);
                off += c.length;
            }
            tts = await KokoroTts.load(bytes);
            const [gold, silver] = await Promise.all([
                fetch("/tts/us_gold.json").then((r) => r.arrayBuffer()),
                fetch("/tts/us_silver.json").then((r) => r.arrayBuffer()),
            ]);
            tts.setLexicon(new Uint8Array(gold), new Uint8Array(silver));
            post({ id, ok: true, sampleRate: tts.sampleRate });
        } else if (type === "synthesize" || type === "synthesizePhonemes") {
            if (!tts) throw new Error("TTS not loaded");
            const pcm =
                type === "synthesize"
                    ? await tts.synthesize(e.data.text!, e.data.voice!)
                    : await tts.synthesizePhonemes(e.data.text!, e.data.voice!);
            post({ id, ok: true, pcm }, [pcm.buffer]);
        }
    } catch (err) {
        post({ id, ok: false, error: String(err) });
    }
};
