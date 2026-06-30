// Minimal WAV (PCM16 mono) encoder for saving synthesized TTS audio, and a Web
// Audio playback helper. No dependencies.

/** Encode mono float32 PCM (-1..1) → a 16-bit PCM WAV Blob. */
export function encodeWav(pcm: Float32Array, sampleRate: number): Blob {
    const n = pcm.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const dv = new DataView(buf);
    const wstr = (off: number, s: string) => {
        for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i));
    };
    wstr(0, "RIFF");
    dv.setUint32(4, 36 + n * 2, true);
    wstr(8, "WAVE");
    wstr(12, "fmt ");
    dv.setUint32(16, 16, true); // fmt chunk size
    dv.setUint16(20, 1, true); // PCM
    dv.setUint16(22, 1, true); // mono
    dv.setUint32(24, sampleRate, true);
    dv.setUint32(28, sampleRate * 2, true); // byte rate
    dv.setUint16(32, 2, true); // block align
    dv.setUint16(34, 16, true); // bits per sample
    wstr(36, "data");
    dv.setUint32(40, n * 2, true);
    let off = 44;
    for (let i = 0; i < n; i++) {
        const s = Math.max(-1, Math.min(1, pcm[i]));
        dv.setInt16(off, (s * 32767) | 0, true);
        off += 2;
    }
    return new Blob([buf], { type: "audio/wav" });
}

/** Trigger a browser download of a WAV blob. */
export function downloadWav(pcm: Float32Array, sampleRate: number, filename: string): void {
    const url = URL.createObjectURL(encodeWav(pcm, sampleRate));
    const a = document.createElement("a");
    a.href = url;
    a.download = filename.endsWith(".wav") ? filename : `${filename}.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Decode an audio File/Blob (uploads or mic recordings) and resample to 24 kHz mono Float32 PCM. */
export async function decodeToPcm24k(file: Blob): Promise<Float32Array> {
    const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const tmp = new AC();
    const decoded = await tmp.decodeAudioData(await file.arrayBuffer());
    tmp.close();
    const len = Math.max(1, Math.ceil(decoded.duration * 24000));
    const off = new OfflineAudioContext(1, len, 24000);
    const src = off.createBufferSource();
    src.buffer = decoded;
    src.connect(off.destination);
    src.start();
    const rendered = await off.startRendering();
    return rendered.getChannelData(0).slice();
}

// A single shared AudioContext for the whole app. Creating a fresh context per
// playback is the classic autoplay-policy trap: when synthesis is async (seconds
// after the click that started it), a just-created context is "suspended" and
// start() makes no sound — but the source node still exists, so the UI flips to
// "Stop" with nothing playing. We instead keep one context and unlock it
// synchronously inside the originating user gesture (see `unlockAudio`).
let sharedCtx: AudioContext | null = null;

function audioCtx(): AudioContext {
    if (!sharedCtx) {
        const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        sharedCtx = new AC();
    }
    return sharedCtx;
}

/** Create + resume the shared AudioContext. Call this *synchronously* from a
 *  user-gesture handler (e.g. the Generate/Play click) so that audio produced
 *  later by async work is allowed to play (iOS requires the resume to happen
 *  within the gesture; desktop accepts sticky activation). Safe to call often. */
export function unlockAudio(): void {
    const ctx = audioCtx();
    if (ctx.state === "suspended") void ctx.resume();
}

/** Play mono float32 PCM through the shared Web Audio context. Returns the
 *  AudioBufferSourceNode (caller stops it / tracks `onended`). */
export function playPcm(pcm: Float32Array, sampleRate: number): AudioBufferSourceNode {
    const ctx = audioCtx();
    // Belt-and-suspenders: resume in case the gesture-time unlock was missed or
    // the policy re-suspended the context between synth and playback.
    if (ctx.state === "suspended") void ctx.resume();
    const buf = ctx.createBuffer(1, pcm.length, sampleRate);
    buf.getChannelData(0).set(pcm);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start(0);
    // Don't close the shared context on end — it's reused for the next clip.
    return src;
}
