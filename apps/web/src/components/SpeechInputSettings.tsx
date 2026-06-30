import { Slider } from "@/components/ui/slider";
import { clampInt } from "@/lib/utils";
import { VOICE_BOUNDS, type VoiceOptions } from "@/lib/voice";

interface Props {
    voice: VoiceOptions;
    onVoiceChange: (v: VoiceOptions) => void;
    /** Audio tower available on the loaded model — the VAD knobs are useless without it. */
    canRecord: boolean;
}

/**
 * Speech-to-text (voice-activity-detection) input settings. Rendered in the Voice-training
 * right sidebar of the DualSidebarLayout — mirroring how inference fine-tuning surfaces its
 * settings there, rather than buried in the Settings dialog.
 */
export function SpeechInputSettings({ voice, onVoiceChange, canRecord }: Props) {
    const V = VOICE_BOUNDS;
    const setV = (patch: Partial<VoiceOptions>) => onVoiceChange({ ...voice, ...patch });
    return (
        <div className="flex h-full min-h-0 flex-col">
            <header className="flex shrink-0 items-center border-b border-border px-3 py-2 text-sm font-medium">
                Speech input
            </header>
            <div className="min-h-0 flex-1 overflow-y-auto p-3">
                {canRecord ? (
                    <section className="flex flex-col gap-3">
                        <Slider
                            label="silence cutoff (ms)"
                            value={voice.silenceMs}
                            min={V.silenceMs.min} max={V.silenceMs.max} step={V.silenceMs.step}
                            onChange={(v) => setV({ silenceMs: clampInt(v, V.silenceMs.min, V.silenceMs.max, V.silenceMs.fallback) })}
                        />
                        <Slider
                            label="speech threshold (dBFS)"
                            value={voice.rmsDbThreshold}
                            min={V.rmsDbThreshold.min} max={V.rmsDbThreshold.max} step={V.rmsDbThreshold.step}
                            onChange={(v) => setV({ rmsDbThreshold: clampInt(v, V.rmsDbThreshold.min, V.rmsDbThreshold.max, V.rmsDbThreshold.fallback) })}
                        />
                        <Slider
                            label="pre-roll (ms)"
                            value={voice.prerollMs}
                            min={V.prerollMs.min} max={V.prerollMs.max} step={V.prerollMs.step}
                            onChange={(v) => setV({ prerollMs: clampInt(v, V.prerollMs.min, V.prerollMs.max, V.prerollMs.fallback) })}
                        />
                        <Slider
                            label="min speech frames"
                            value={voice.minSpeechFrames}
                            min={V.minSpeechFrames.min} max={V.minSpeechFrames.max} step={V.minSpeechFrames.step}
                            onChange={(v) => setV({ minSpeechFrames: clampInt(v, V.minSpeechFrames.min, V.minSpeechFrames.max, V.minSpeechFrames.fallback) })}
                        />
                        <Slider
                            label="max recording (ms)"
                            value={voice.maxRecordMs}
                            min={V.maxRecordMs.min} max={V.maxRecordMs.max} step={V.maxRecordMs.step}
                            onChange={(v) => setV({ maxRecordMs: clampInt(v, V.maxRecordMs.min, V.maxRecordMs.max, V.maxRecordMs.fallback) })}
                        />
                    </section>
                ) : (
                    <p className="text-xs text-muted-foreground">
                        Mic / speech-to-text settings (voice activity detection) for talking to the model.
                        Load a model with an audio tower to enable them.
                    </p>
                )}
            </div>
        </div>
    );
}
