// Microphone capture with VAD-driven auto-stop.
//
// State machine, driven by the 20 ms frames streamed from the
// AudioWorklet (see `audio-vad-worklet.js`):
//
//   idle ── start() ──▶ recording (preroll)
//                          │
//                  speech-frame run ≥ minSpeechFrames
//                          ▼
//                       recording (capturing) ── silence ≥ silenceMs ──▶ encoding
//                          │                                                       │
//                  elapsed > maxRecordMs                                            ▼
//                          ▼                                                      idle
//                       encoding ──▶ idle (after onComplete resolves)
//
// Pre-roll: we hold the last `prerollMs` of audio while we wait for the
// first speech frame. When speech kicks in, those frames get prepended
// to the recording so the first syllable isn't clipped. Without this,
// the VAD reacting to the *first* loud frame means the leading edge
// of the word never makes it into the buffer.
//
// All knobs come from the caller-supplied `voice: VoiceOptions` — see
// `lib/voice.ts`. The 20 ms frame size is the only constant kept here;
// it's a worklet protocol constant tuned to give one syllable's RMS a
// stable signal while still detecting end-of-utterance within the
// silence budget.

import { useCallback, useEffect, useRef, useState } from "react";
import type { VoiceOptions } from "./voice";

const FRAME_MS = 20;

export type MicState = "idle" | "recording" | "encoding";

export interface UseMicCaptureOpts {
    /** Tunables — silence cutoff, RMS threshold, preroll, etc. Read live
     *  inside the frame handler so changes from a settings UI take
     *  effect for the next recording without remounting. */
    voice: VoiceOptions;
    /** Called once VAD auto-stops or the user manually stops (with at
     *  least one speech frame captured). PCM is 16 kHz mono f32 in
     *  [-1, 1] — the format `Model.encodeAudio` consumes. */
    onComplete: (pcm: Float32Array) => void | Promise<void>;
    onError?: (err: Error) => void;
}

interface CaptureHandle {
    state: MicState;
    /** Most recent frame RMS in dBFS, for level-meter UI. -100 when idle. */
    rmsDb: number;
    /** Start recording. No-op if not idle. */
    start: () => Promise<void>;
    /** Stop early without delivering whatever was captured. */
    cancel: () => void;
}

export function useMicCapture(opts: UseMicCaptureOpts): CaptureHandle {
    const [state, setState] = useState<MicState>("idle");
    const [rmsDb, setRmsDb] = useState(-100);

    // Latest opts ref — onComplete may close over chat state that
    // changes between renders, but the hook's session is identified by
    // teardownRef.current === thisTeardown, so we read the latest
    // callback at finish() time instead of capturing it.
    const optsRef = useRef(opts);
    optsRef.current = opts;

    // Identity of the active capture session. cancel() and finish()
    // both compare this against their captured closure to avoid acting
    // on stale state when start() has been re-fired.
    const teardownRef = useRef<(() => void) | null>(null);

    const cancel = useCallback(() => {
        const td = teardownRef.current;
        teardownRef.current = null;
        if (td) td();
        setState("idle");
        setRmsDb(-100);
    }, []);

    const start = useCallback(async () => {
        if (teardownRef.current) return;
        try {
            if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
                throw new Error("Microphone access not supported in this browser");
            }
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    // 16 kHz mono is what the audio tower wants; ask the
                    // browser nicely. If it can't honor the rate it gives
                    // us the device's native rate and we'd need to
                    // resample — but every browser+OS combo I've tested
                    // honors 16 kHz for getUserMedia.
                    sampleRate: 16_000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });
            const ctx = new AudioContext({ sampleRate: 16_000 });
            await ctx.audioWorklet.addModule(
                new URL("./audio-vad-worklet.js", import.meta.url),
            );
            const src = ctx.createMediaStreamSource(stream);
            const node = new AudioWorkletNode(ctx, "vad-worklet");

            // Snapshot of the preroll budget at session start. The other
            // knobs are read live from optsRef inside the frame handler
            // so a settings change mid-recording still applies on the
            // next frame.
            const prerollMax = Math.ceil(optsRef.current.voice.prerollMs / FRAME_MS);
            let preroll: Float32Array[] = [];
            const recorded: Float32Array[] = [];
            let started = false;
            let speechRun = 0;
            let silenceMs = 0;
            let elapsedMs = 0;

            const teardown = () => {
                try { node.disconnect(); } catch { /* */ }
                try { src.disconnect(); } catch { /* */ }
                try { stream.getTracks().forEach((t) => t.stop()); } catch { /* */ }
                try { void ctx.close(); } catch { /* */ }
            };
            teardownRef.current = teardown;

            const finish = (deliver: boolean) => {
                if (teardownRef.current !== teardown) return;
                teardownRef.current = null;
                teardown();

                if (!deliver || !started || recorded.length === 0) {
                    setState("idle");
                    setRmsDb(-100);
                    return;
                }

                const total = recorded.reduce((acc, c) => acc + c.length, 0);
                const pcm = new Float32Array(total);
                let off = 0;
                for (const f of recorded) {
                    pcm.set(f, off);
                    off += f.length;
                }

                setState("encoding");
                setRmsDb(-100);
                Promise.resolve(optsRef.current.onComplete(pcm))
                    .catch((e) => optsRef.current.onError?.(e as Error))
                    .finally(() => setState("idle"));
            };

            node.port.onmessage = (e) => {
                if (teardownRef.current !== teardown) return;
                const v = optsRef.current.voice;
                const { rmsDb: r, samples } = e.data as { rmsDb: number; samples: Float32Array };
                setRmsDb(r);
                elapsedMs += FRAME_MS;

                const isSpeech = r > v.rmsDbThreshold;

                if (!started) {
                    preroll.push(samples);
                    if (preroll.length > prerollMax) preroll.shift();
                    if (isSpeech) {
                        speechRun++;
                        if (speechRun >= v.minSpeechFrames) {
                            // Promote: dump preroll into the recording so
                            // the first syllable isn't clipped, then keep
                            // accumulating.
                            started = true;
                            recorded.push(...preroll);
                            preroll = [];
                            silenceMs = 0;
                        }
                    } else {
                        speechRun = 0;
                    }
                } else {
                    recorded.push(samples);
                    if (isSpeech) {
                        silenceMs = 0;
                    } else {
                        silenceMs += FRAME_MS;
                        if (silenceMs >= v.silenceMs) {
                            finish(true);
                            return;
                        }
                    }
                }

                if (elapsedMs >= v.maxRecordMs) {
                    // Hard cap. Deliver whatever we have if we ever
                    // detected speech; drop the recording otherwise.
                    finish(started);
                }
            };

            src.connect(node);
            // Intentionally don't connect node.connect(ctx.destination) —
            // the user doesn't need to hear their own voice played back.
            setState("recording");
            setRmsDb(-100);
        } catch (e) {
            optsRef.current.onError?.(e as Error);
            setState("idle");
            setRmsDb(-100);
        }
    }, []);

    // Cleanup on unmount — a recording in flight should not leak the
    // MediaStream or AudioContext when the component goes away.
    useEffect(() => () => {
        const td = teardownRef.current;
        teardownRef.current = null;
        if (td) td();
    }, []);

    return { state, rmsDb, start, cancel };
}
