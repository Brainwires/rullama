import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { ModelLoader, type ModelStatus } from "@/components/ModelLoader";
import { type SamplingOptions } from "@/lib/types";
import { type ModelEntry } from "@/lib/api";
import { clampInt, clampNum } from "@/lib/utils";
import { RotateCcw } from "lucide-react";

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

    // System + sampling
    systemPrompt: string;
    onSystemPromptChange: (s: string) => void;
    sampling: SamplingOptions;
    onSamplingChange: (s: SamplingOptions) => void;
    maxTokens: number;
    onMaxTokensChange: (n: number) => void;
    thinking: boolean;
    onThinkingChange: (b: boolean) => void;

    /** Reset systemPrompt + sampling + maxTokens + thinking to defaults. */
    onResetDefaults: () => void;
}

/** Full-height sidebar: Model + Generation settings, sections scroll. */
export function SettingsDialog(props: Props) {
    const B = SETTINGS_BOUNDS;
    const setS = (patch: Partial<SamplingOptions>) =>
        props.onSamplingChange({ ...props.sampling, ...patch });

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
            </div>
        </div>
    );
}
