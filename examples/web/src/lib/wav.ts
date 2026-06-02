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

/** Play mono float32 PCM through Web Audio. Returns the AudioBufferSourceNode. */
export function playPcm(pcm: Float32Array, sampleRate: number): AudioBufferSourceNode {
    const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new AC();
    const buf = ctx.createBuffer(1, pcm.length, sampleRate);
    buf.getChannelData(0).set(pcm);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start(0);
    src.onended = () => ctx.close();
    return src;
}
