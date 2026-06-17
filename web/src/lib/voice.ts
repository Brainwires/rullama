// Tunables for the in-browser VAD used by the mic button. Mirrors the
// algorithm parameters previously hard-coded in `useMicCapture.ts`; the
// defaults match brainwires-framework's EnergyVad (-40 dBFS threshold,
// 800 ms silence cutoff). The 20 ms frame size is intentionally *not*
// exposed — it's a worklet protocol constant tied to the RMS smoothing
// window, not a knob users should be turning.

export interface VoiceOptions {
    /** Auto-stop after this much trailing silence (ms). */
    silenceMs:       number;
    /** Frame energy above this is treated as speech (dBFS). */
    rmsDbThreshold:  number;
    /** Audio held back before first speech, so the leading edge of the
     *  first syllable isn't clipped (ms). */
    prerollMs:       number;
    /** Number of consecutive speech frames needed to start a recording.
     *  Debounces single-click spikes. */
    minSpeechFrames: number;
    /** Hard cap on one utterance (ms). */
    maxRecordMs:     number;
}

export const VOICE_BOUNDS = {
    silenceMs:       { min: 200,  max: 3000,   step: 50,   fallback: 800   },
    rmsDbThreshold:  { min: -60,  max: -20,    step: 1,    fallback: -40   },
    prerollMs:       { min: 0,    max: 1000,   step: 50,   fallback: 300   },
    minSpeechFrames: { min: 1,    max: 20,     step: 1,    fallback: 4     },
    maxRecordMs:     { min: 5000, max: 120000, step: 1000, fallback: 30000 },
} as const;

export const DEFAULT_VOICE_OPTIONS: VoiceOptions = {
    silenceMs:       VOICE_BOUNDS.silenceMs.fallback,
    rmsDbThreshold:  VOICE_BOUNDS.rmsDbThreshold.fallback,
    prerollMs:       VOICE_BOUNDS.prerollMs.fallback,
    minSpeechFrames: VOICE_BOUNDS.minSpeechFrames.fallback,
    maxRecordMs:     VOICE_BOUNDS.maxRecordMs.fallback,
};
