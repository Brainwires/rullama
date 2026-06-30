import { useCallback, useEffect } from "react";
import { type SamplingOptions, DEFAULT_SAMPLING, DEFAULT_SYSTEM_PROMPT } from "@/lib/types";
import { SETTINGS_BOUNDS } from "@/components/SettingsDialog";
import { DEFAULT_VOICE_OPTIONS, VOICE_BOUNDS, type VoiceOptions } from "@/lib/voice";
import { usePersistedState } from "@/lib/persisted";
import { useToast } from "@/lib/toast";
import { clampInt, clampNum } from "@/lib/utils";

/**
 * Persisted chat/voice tunables — system prompt, sampling, max tokens,
 * thinking toggle, and voice (VAD) options — grouped behind one hook.
 *
 * Owns two pieces of behaviour that belong with the values themselves:
 *
 *   - **One-time sanitization** of persisted values on mount. Catches
 *     localStorage entries from older versions (or hand-edited values)
 *     that fall outside current bounds — the slider clamps only cover
 *     fresh edits, so a value persisted before a bounds change would
 *     otherwise survive forever.
 *   - **`onResetDefaults`** — restores every tunable to its default and
 *     toasts. Lives here so the default set and the reset stay in sync.
 */
export function useChatTunables() {
    const { showToast } = useToast();

    const [systemPrompt, setSystemPrompt] = usePersistedState<string>("systemPrompt", DEFAULT_SYSTEM_PROMPT);
    const [sampling,     setSampling]     = usePersistedState<SamplingOptions>("sampling", DEFAULT_SAMPLING);
    const [maxTokens,    setMaxTokens]    = usePersistedState<number>("maxTokens", 4096);
    const [thinking,     setThinking]     = usePersistedState<boolean>("thinking", true);
    const [voice,        setVoice]        = usePersistedState<VoiceOptions>("voice", DEFAULT_VOICE_OPTIONS);

    // One-time sanitization of persisted values. Catches localStorage
    // entries from older versions (or hand-edited values) that fall
    // outside current bounds — the slider clamps cover fresh edits.
    useEffect(() => {
        const B = SETTINGS_BOUNDS;
        const next: SamplingOptions = {
            temperature:        clampNum(sampling.temperature, B.temperature.min, B.temperature.max, B.temperature.fallback),
            top_k:              clampInt(sampling.top_k,       B.top_k.min,       B.top_k.max,       B.top_k.fallback),
            top_p:              clampNum(sampling.top_p,       B.top_p.min,       B.top_p.max,       B.top_p.fallback),
            repetition_penalty: clampNum(sampling.repetition_penalty, B.repetition_penalty.min, B.repetition_penalty.max, B.repetition_penalty.fallback),
            seed:               Number.isFinite(sampling.seed) ? sampling.seed : 0,
        };
        if (next.temperature !== sampling.temperature
            || next.top_k !== sampling.top_k
            || next.top_p !== sampling.top_p
            || next.repetition_penalty !== sampling.repetition_penalty
            || next.seed !== sampling.seed) {
            setSampling(next);
        }
        const mt = clampInt(maxTokens, B.maxTokens.min, B.maxTokens.max, B.maxTokens.fallback);
        if (mt !== maxTokens) setMaxTokens(mt);

        const VB = VOICE_BOUNDS;
        const nextVoice: VoiceOptions = {
            silenceMs:       clampInt(voice.silenceMs,       VB.silenceMs.min,       VB.silenceMs.max,       VB.silenceMs.fallback),
            rmsDbThreshold:  clampInt(voice.rmsDbThreshold,  VB.rmsDbThreshold.min,  VB.rmsDbThreshold.max,  VB.rmsDbThreshold.fallback),
            prerollMs:       clampInt(voice.prerollMs,       VB.prerollMs.min,       VB.prerollMs.max,       VB.prerollMs.fallback),
            minSpeechFrames: clampInt(voice.minSpeechFrames, VB.minSpeechFrames.min, VB.minSpeechFrames.max, VB.minSpeechFrames.fallback),
            maxRecordMs:     clampInt(voice.maxRecordMs,     VB.maxRecordMs.min,     VB.maxRecordMs.max,     VB.maxRecordMs.fallback),
        };
        if (nextVoice.silenceMs !== voice.silenceMs
            || nextVoice.rmsDbThreshold !== voice.rmsDbThreshold
            || nextVoice.prerollMs !== voice.prerollMs
            || nextVoice.minSpeechFrames !== voice.minSpeechFrames
            || nextVoice.maxRecordMs !== voice.maxRecordMs) {
            setVoice(nextVoice);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const onResetDefaults = useCallback(() => {
        setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
        setSampling(DEFAULT_SAMPLING);
        setMaxTokens(SETTINGS_BOUNDS.maxTokens.fallback);
        setThinking(true);
        setVoice(DEFAULT_VOICE_OPTIONS);
        showToast({
            level: "success",
            title: "Settings reset to defaults",
        });
    }, [setSystemPrompt, setSampling, setMaxTokens, setThinking, setVoice, showToast]);

    return {
        systemPrompt, setSystemPrompt,
        sampling, setSampling,
        maxTokens, setMaxTokens,
        thinking, setThinking,
        voice, setVoice,
        onResetDefaults,
    };
}
