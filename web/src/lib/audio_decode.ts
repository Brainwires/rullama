// Audio file decode helper for the chat-attach path.
//
// Accepts any audio Blob the browser knows how to decode (mp3, wav,
// ogg, aac, m4a, …) and produces the 16 kHz mono Float32 PCM the
// rullama audio tower expects, in range [-1, 1]. AudioContext does
// the format demux + resample for us — the only manual step is
// mono-mixing for multi-channel sources.
//
// Used by `App.tsx::onAttachFiles` audio branch. Different from
// `useMicCapture.ts` (which sources from the live mic, not a file).

const TARGET_SR = 16000;

export interface DecodedAudio {
    pcm:        Float32Array;
    durationMs: number;
    /** Source MIME for diagnostics + the badge label in the pending strip. */
    sourceMime: string;
}

/** Decode an audio Blob → 16 kHz mono PCM. Resolves to `null` if the
 *  browser can't decode the format (we accept anything `audio/*` from
 *  the file input but some browsers reject e.g. aac/m4a). */
export async function decodeAudioFile(blob: Blob): Promise<DecodedAudio | null> {
    const buf = await blob.arrayBuffer();
    // OfflineAudioContext with a target sample rate makes the browser
    // resample for us during decode. Need the context only to call
    // decodeAudioData — we don't actually start playback.
    //
    // Some browsers reject creating an OfflineAudioContext with very
    // short durations or unusual sample rates; falling back to a
    // throwaway AudioContext is more compatible. AudioContext can be
    // created at any sample rate (Safari sometimes ignores the
    // requested rate but the AudioBuffer comes back at the device
    // rate, which we then resample manually below).
    const ctor: typeof AudioContext | undefined =
        (window as unknown as { AudioContext?: typeof AudioContext }).AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!ctor) return null;
    const ctx = new ctor({ sampleRate: TARGET_SR });
    try {
        const audioBuf = await ctx.decodeAudioData(buf.slice(0));
        // If the browser honored sampleRate, audioBuf.sampleRate === TARGET_SR.
        // If not (Safari sometimes returns at the device rate), we resample.
        const sr = audioBuf.sampleRate;
        const monoSrc = mixToMono(audioBuf);
        const pcm = sr === TARGET_SR ? monoSrc : resampleLinear(monoSrc, sr, TARGET_SR);
        const durationMs = (pcm.length / TARGET_SR) * 1000;
        return { pcm, durationMs, sourceMime: blob.type || "audio/*" };
    } catch {
        return null;
    } finally {
        try { await ctx.close(); } catch { /* */ }
    }
}

/** Channel-average an AudioBuffer into a single Float32 PCM stream. */
function mixToMono(buf: AudioBuffer): Float32Array {
    const n = buf.length;
    if (buf.numberOfChannels === 1) {
        // Copy out so we own the buffer (audioBuf is GC'd with the
        // OfflineAudioContext).
        return new Float32Array(buf.getChannelData(0));
    }
    const out = new Float32Array(n);
    for (let c = 0; c < buf.numberOfChannels; c++) {
        const src = buf.getChannelData(c);
        for (let i = 0; i < n; i++) out[i] += src[i];
    }
    const inv = 1 / buf.numberOfChannels;
    for (let i = 0; i < n; i++) out[i] *= inv;
    return out;
}

/** Linear-interpolation resampler — adequate for the mel-spectrogram
 *  prefix that's about to chew on the audio anyway. The audio tower
 *  doesn't care about sub-sample-accurate timing. */
function resampleLinear(src: Float32Array, srcSr: number, dstSr: number): Float32Array {
    const ratio = srcSr / dstSr;
    const outLen = Math.floor(src.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
        const t = i * ratio;
        const lo = Math.floor(t);
        const hi = Math.min(lo + 1, src.length - 1);
        const frac = t - lo;
        out[i] = src[lo] * (1 - frac) + src[hi] * frac;
    }
    return out;
}
