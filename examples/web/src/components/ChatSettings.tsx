import { Sparkles, Undo2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { SETTINGS_BOUNDS } from "@/components/SettingsDialog";
import { ModelLoader, type ModelStatus } from "@/components/ModelLoader";
import { SpeechInputSettings } from "@/components/SpeechInputSettings";
import { type ModelEntry } from "@/lib/api";
import { type SamplingOptions } from "@/lib/types";
import { type VoiceOptions } from "@/lib/voice";
import { clampInt, clampNum } from "@/lib/utils";

interface Props {
    // Model management (moved here from the global Settings view — the
    // inference model belongs to the Chat tab).
    modelStatus:    ModelStatus;
    loadingPercent: number;
    loadingLabel:   string;
    statusText:     string;
    onLoadModel:    (m: ModelEntry) => void;
    onDeleteModel:  (m: ModelEntry) => void;
    onEjectModel:   () => void;
    /** Opens the full-screen Fine-tune (LoRA training) overlay. */
    onOpenFineTune: () => void;
    // Generation settings.
    systemPrompt: string;
    onSystemPromptChange: (s: string) => void;
    sampling: SamplingOptions;
    onSamplingChange: (s: SamplingOptions) => void;
    maxTokens: number;
    onMaxTokensChange: (n: number) => void;
    thinking: boolean;
    onThinkingChange: (b: boolean) => void;
    onResetDefaults: () => void;
    // Speech input (chat mic → transcription) VAD config.
    voice: VoiceOptions;
    onVoiceChange: (v: VoiceOptions) => void;
    canRecord: boolean;
}

/** Chat-tab right sidebar: the generation settings that belong to the chat surface — system
 *  prompt, thinking mode, and the sampling knobs. (Model management / logs / app-data live in the
 *  global Settings view.) */
export function ChatSettings(props: Props) {
    const B = SETTINGS_BOUNDS;
    const setS = (patch: Partial<SamplingOptions>) => props.onSamplingChange({ ...props.sampling, ...patch });

    return (
        <div className="flex h-full min-h-0 flex-col">
            <header className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Chat settings
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
                    title="Reset chat generation settings to defaults"
                >
                    <Undo2 />
                    Defaults
                </Button>
            </header>

            <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 py-3">
                <section className="flex flex-col gap-2">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Model</span>
                    <ModelLoader
                        status={props.modelStatus}
                        loadingPercent={props.loadingPercent}
                        loadingLabel={props.loadingLabel}
                        statusText={props.statusText}
                        onLoad={props.onLoadModel}
                        onDelete={props.onDeleteModel}
                        onEject={props.onEjectModel}
                    />
                    <Button
                        variant="outline"
                        size="sm"
                        className="h-8 justify-start gap-2 text-xs"
                        onClick={props.onOpenFineTune}
                        disabled={props.modelStatus !== "ready"}
                        title={props.modelStatus === "ready"
                            ? "Fine-tune (LoRA) the loaded model on your own data"
                            : "Load a model first"}
                    >
                        <Sparkles className="size-3.5" />
                        Fine-tune…
                    </Button>
                </section>

                <section className="flex flex-col gap-2 border-t border-border pt-3">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">System prompt</span>
                    <Textarea
                        value={props.systemPrompt}
                        onChange={(e) => props.onSystemPromptChange(e.target.value)}
                        placeholder='Optional. e.g. "You are a pirate."'
                        className="min-h-[3rem] text-xs"
                    />
                </section>

                <section className="flex flex-col gap-2 border-t border-border pt-3">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Sampling</span>
                    <Slider
                        label="temperature"
                        value={props.sampling.temperature}
                        min={B.temperature.min} max={B.temperature.max} step={B.temperature.step}
                        fmt={(v) => v.toFixed(2)}
                        onChange={(v) => setS({ temperature: clampNum(v, B.temperature.min, B.temperature.max, B.temperature.fallback) })}
                    />
                    <Slider
                        label="top_p"
                        value={props.sampling.top_p}
                        min={B.top_p.min} max={B.top_p.max} step={B.top_p.step}
                        fmt={(v) => v.toFixed(2)}
                        onChange={(v) => setS({ top_p: clampNum(v, B.top_p.min, B.top_p.max, B.top_p.fallback) })}
                    />
                    <Slider
                        label="top_k"
                        value={props.sampling.top_k}
                        min={B.top_k.min} max={B.top_k.max} step={B.top_k.step}
                        onChange={(v) => setS({ top_k: clampInt(v, B.top_k.min, B.top_k.max, B.top_k.fallback) })}
                    />
                    <Slider
                        label="repetition penalty"
                        value={props.sampling.repetition_penalty}
                        min={B.repetition_penalty.min} max={B.repetition_penalty.max} step={B.repetition_penalty.step}
                        fmt={(v) => v.toFixed(2)}
                        onChange={(v) => setS({ repetition_penalty: clampNum(v, B.repetition_penalty.min, B.repetition_penalty.max, B.repetition_penalty.fallback) })}
                    />
                    <Slider
                        label="max tokens"
                        value={props.maxTokens}
                        min={B.maxTokens.min} max={B.maxTokens.max} step={B.maxTokens.step}
                        onChange={(v) => props.onMaxTokensChange(clampInt(v, B.maxTokens.min, B.maxTokens.max, B.maxTokens.fallback))}
                    />
                </section>

                <section className="flex flex-col gap-2 border-t border-border pt-3">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Advanced</span>
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
                    <p className="ml-6 text-[10px] leading-tight text-muted-foreground">
                        Tip: combining thinking with temperature &gt; 0.3 on attached images or audio can produce
                        wandering chain-of-thought that reads like garbled text. Lower both for crisp multimodal analysis.
                    </p>
                </section>

                <section className="flex flex-col gap-2 border-t border-border pt-3">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Speech input</span>
                    <SpeechInputSettings
                        voice={props.voice}
                        onVoiceChange={props.onVoiceChange}
                        canRecord={props.canRecord}
                    />
                </section>
            </div>
        </div>
    );
}
