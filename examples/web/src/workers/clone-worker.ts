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
    pcm?: Float32Array;
    text?: string;
    voice?: Float32Array;
}

function post(msg: Record<string, unknown>, transfer?: Transferable[]) {
    (self as unknown as Worker).postMessage(msg, transfer ?? []);
}

self.onmessage = async (e: MessageEvent<Req>) => {
    const { id, type } = e.data;
    try {
        if (type === "load") {
            await init();
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
            clone = StyleTtsClone.load(bytes);
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
            const voice = clone.encodeVoice(e.data.pcm!, onProg);
            post({ id, ok: true, voice }, [voice.buffer]);
        } else if (type === "synthesize") {
            if (!clone) throw new Error("clone engine not loaded");
            const onProg = (frac: number, stage: string) => {
                console.log(`[clone] synth ${(frac * 100) | 0}% — ${stage}`);
                post({ id, progress: frac, stage });
            };
            const pcm = clone.synthesize(e.data.text!, e.data.voice!, onProg);
            post({ id, ok: true, pcm }, [pcm.buffer]);
        }
    } catch (err) {
        post({ id, ok: false, error: String(err) });
    }
};
