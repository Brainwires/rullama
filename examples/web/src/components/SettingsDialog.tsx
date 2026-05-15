import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { ModelLoader, type ModelStatus } from "@/components/ModelLoader";
import { type SamplingOptions } from "@/lib/types";
import { type ModelEntry } from "@/lib/api";
import { clampInt, clampNum } from "@/lib/utils";
import { VOICE_BOUNDS, type VoiceOptions } from "@/lib/voice";
import { hardResetAndReload } from "@/lib/restart";
import { RotateCcw, RefreshCw } from "lucide-react";

// Hard bounds — also used by App.tsx to normalize old persisted values on
// boot. Keep these conservative; users with a real need can edit the JSON
// in localStorage directly.
export const SETTINGS_BOUNDS = {
    temperature:        { min: 0,    max: 2,    step: 0.05, fallback: 0.7  },
    top_k:              { min: 0,    max: 200,  step: 1,    fallback: 40   },
    top_p:              { min: 0,    max: 1,    step: 0.01, fallback: 0.95 },
    repetition_penalty: { min: 0.5,  max: 2.0,  step: 0.05, fallback: 1.1  },
    maxTokens:          { min: 16,   max: 4096, step: 16,   fallback: 1024 },
} as const;

interface Props {
    // Model
    modelStatus:    ModelStatus;
    loadingPercent: number;
    loadingLabel:   string;
    statusText:     string;
    onLoadModel:    (m: ModelEntry) => void;
    onDeleteModel:  (m: ModelEntry) => void;
    onEjectModel:   () => void;

    // System + sampling
    systemPrompt: string;
    onSystemPromptChange: (s: string) => void;
    sampling: SamplingOptions;
    onSamplingChange: (s: SamplingOptions) => void;
    maxTokens: number;
    onMaxTokensChange: (n: number) => void;
    thinking: boolean;
    onThinkingChange: (b: boolean) => void;

    // Voice (VAD) — only shown when the loaded model has an audio tower.
    voice: VoiceOptions;
    onVoiceChange: (v: VoiceOptions) => void;
    /** Audio tower available on the loaded model. The Voice section is
     *  hidden on text/vision-only models to avoid useless knobs. */
    canRecord: boolean;

    /** Reset systemPrompt + sampling + maxTokens + thinking + voice to defaults. */
    onResetDefaults: () => void;
}

/** Full-height sidebar: Model + Generation settings, sections scroll. */
export function SettingsDialog(props: Props) {
    const B = SETTINGS_BOUNDS;
    const V = VOICE_BOUNDS;
    const setS = (patch: Partial<SamplingOptions>) =>
        props.onSamplingChange({ ...props.sampling, ...patch });
    const setV = (patch: Partial<VoiceOptions>) =>
        props.onVoiceChange({ ...props.voice, ...patch });

    return (
        <div className="flex h-full min-h-0 flex-col">
            <header className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Settings
                </span>
                <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
                    onClick={() => {
                        if (window.confirm("Reset system prompt, sampling, max tokens, and thinking mode to defaults?")) {
                            props.onResetDefaults();
                        }
                    }}
                    title="Reset all generation settings to defaults"
                >
                    <RotateCcw />
                    Defaults
                </Button>
            </header>

            <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 py-3">
                <section className="flex flex-col gap-1.5">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        Model
                    </span>
                    <ModelLoader
                        status={props.modelStatus}
                        loadingPercent={props.loadingPercent}
                        loadingLabel={props.loadingLabel}
                        statusText={props.statusText}
                        onLoad={props.onLoadModel}
                        onDelete={props.onDeleteModel}
                        onEject={props.onEjectModel}
                    />
                </section>

                <section className="flex flex-col gap-2 border-t border-border pt-3">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        System prompt
                    </span>
                    <Textarea
                        value={props.systemPrompt}
                        onChange={(e) => props.onSystemPromptChange(e.target.value)}
                        placeholder='Optional. e.g. "You are a pirate."'
                        className="min-h-[3rem] text-xs"
                    />
                </section>

                <section className="flex flex-col gap-3 border-t border-border pt-3">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        Sampling
                    </span>
                    <Slider
                        label="temperature"
                        value={props.sampling.temperature}
                        min={B.temperature.min} max={B.temperature.max} step={B.temperature.step}
                        fmt={(v) => v.toFixed(2)}
                        onChange={(v) => setS({
                            temperature: clampNum(v, B.temperature.min, B.temperature.max, B.temperature.fallback),
                        })}
                    />
                    <Slider
                        label="top_p"
                        value={props.sampling.top_p}
                        min={B.top_p.min} max={B.top_p.max} step={B.top_p.step}
                        fmt={(v) => v.toFixed(2)}
                        onChange={(v) => setS({
                            top_p: clampNum(v, B.top_p.min, B.top_p.max, B.top_p.fallback),
                        })}
                    />
                    <Slider
                        label="top_k"
                        value={props.sampling.top_k}
                        min={B.top_k.min} max={B.top_k.max} step={B.top_k.step}
                        onChange={(v) => setS({
                            top_k: clampInt(v, B.top_k.min, B.top_k.max, B.top_k.fallback),
                        })}
                    />
                    <Slider
                        label="max tokens"
                        value={props.maxTokens}
                        min={B.maxTokens.min} max={B.maxTokens.max} step={B.maxTokens.step}
                        onChange={(v) => props.onMaxTokensChange(
                            clampInt(v, B.maxTokens.min, B.maxTokens.max, B.maxTokens.fallback),
                        )}
                    />
                </section>

                <section className="flex flex-col gap-2 border-t border-border pt-3">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        Advanced
                    </span>
                    <label
                        className="flex items-start gap-2 text-xs text-muted-foreground"
                        title='Silently prepend "<|think|>" to the system prompt so Gemma 4 emits its internal reasoning trace before the answer. Not shown in the chat history.'
                    >
                        <input
                            type="checkbox"
                            checked={props.thinking}
                            onChange={(e) => props.onThinkingChange(e.target.checked)}
                            className="mt-0.5 h-3.5 w-3.5 shrink-0 cursor-pointer"
                        />
                        <span>
                            Thinking mode — prepend{" "}
                            <code className="rounded bg-muted px-1">&lt;|think|&gt;</code>{" "}
                            silently before the system message
                        </span>
                    </label>
                </section>

                {props.canRecord && (
                    <section className="flex flex-col gap-3 border-t border-border pt-3">
                        <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                            Voice
                        </span>
                        <Slider
                            label="silence cutoff (ms)"
                            value={props.voice.silenceMs}
                            min={V.silenceMs.min} max={V.silenceMs.max} step={V.silenceMs.step}
                            onChange={(v) => setV({
                                silenceMs: clampInt(v, V.silenceMs.min, V.silenceMs.max, V.silenceMs.fallback),
                            })}
                        />
                        <Slider
                            label="speech threshold (dBFS)"
                            value={props.voice.rmsDbThreshold}
                            min={V.rmsDbThreshold.min} max={V.rmsDbThreshold.max} step={V.rmsDbThreshold.step}
                            onChange={(v) => setV({
                                rmsDbThreshold: clampInt(v, V.rmsDbThreshold.min, V.rmsDbThreshold.max, V.rmsDbThreshold.fallback),
                            })}
                        />
                        <Slider
                            label="pre-roll (ms)"
                            value={props.voice.prerollMs}
                            min={V.prerollMs.min} max={V.prerollMs.max} step={V.prerollMs.step}
                            onChange={(v) => setV({
                                prerollMs: clampInt(v, V.prerollMs.min, V.prerollMs.max, V.prerollMs.fallback),
                            })}
                        />
                        <Slider
                            label="min speech frames"
                            value={props.voice.minSpeechFrames}
                            min={V.minSpeechFrames.min} max={V.minSpeechFrames.max} step={V.minSpeechFrames.step}
                            onChange={(v) => setV({
                                minSpeechFrames: clampInt(v, V.minSpeechFrames.min, V.minSpeechFrames.max, V.minSpeechFrames.fallback),
                            })}
                        />
                        <Slider
                            label="max recording (ms)"
                            value={props.voice.maxRecordMs}
                            min={V.maxRecordMs.min} max={V.maxRecordMs.max} step={V.maxRecordMs.step}
                            onChange={(v) => setV({
                                maxRecordMs: clampInt(v, V.maxRecordMs.min, V.maxRecordMs.max, V.maxRecordMs.fallback),
                            })}
                        />
                    </section>
                )}

                <section className="flex flex-col gap-2 border-t border-border pt-3">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        Trouble
                    </span>
                    <Button
                        variant="outline"
                        size="sm"
                        className="h-8 justify-start gap-2 text-xs"
                        onClick={() => {
                            if (window.confirm(
                                "Unregister service worker, clear cached assets, and reload. "
                                + "Cached models and settings are preserved. Continue?",
                            )) {
                                void hardResetAndReload();
                            }
                        }}
                        title="Wipe service-worker cache and reload — recovery for stuck PWA states"
                    >
                        <RefreshCw className="h-3.5 w-3.5" />
                        Reset app data
                    </Button>
                </section>
            </div>
        </div>
    );
}
