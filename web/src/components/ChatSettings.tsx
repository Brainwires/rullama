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
import { usePersistedState } from "@/lib/persisted";
import { cn, clampInt, clampNum } from "@/lib/utils";

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
    /** Shared model-picker selection (kept in sync with the main panel). */
    selectedModelName?: string;
    onSelectModel?:     (name: string) => void;
    preferredDigest?:   string;
    /** Abort an in-progress model download. */
    onCancelDownload?:  () => void;
    /** Opens the full-screen Fine-tune (LoRA training) overlay. */
    onOpenFineTune: () => void;
    /** Whether fine-tuning can actually run here — model ready AND the device
     *  passes the training-capability probe (iOS is blocked by default). */
    canFineTune:    boolean;
    /** Why fine-tune is unavailable (tooltip), when `canFineTune` is false. */
    fineTuneReason: string;
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

type TabKey = "model" | "generation" | "speech";

/** Chat-tab right sidebar, tabbed because it carries a lot now:
 *  - **Model**: load/eject the inference model + the Fine-tune launcher.
 *  - **Generation**: system prompt, sampling knobs, thinking mode (+ Defaults reset).
 *  - **Speech**: chat mic → transcription VAD config.
 *  (Logs / app-data / High-VRAM toggle live in the global Settings view.) */
export function ChatSettings(props: Props) {
    const B = SETTINGS_BOUNDS;
    const setS = (patch: Partial<SamplingOptions>) => props.onSamplingChange({ ...props.sampling, ...patch });
    const [tab, setTab] = usePersistedState<TabKey>("chat:settings:tab", "model");

    return (
        <div className="flex h-full min-h-0 flex-col">
            <header className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Chat settings
                </span>
                {tab === "generation" && (
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
                )}
            </header>

            <nav role="tablist" aria-label="Chat settings sections" className="flex shrink-0 border-b border-border px-3">
                <TabButton label="Model" active={tab === "model"} onClick={() => setTab("model")} />
                <TabButton label="Generation" active={tab === "generation"} onClick={() => setTab("generation")} />
                <TabButton label="Speech" active={tab === "speech"} onClick={() => setTab("speech")} />
            </nav>

            <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 py-3">
                {tab === "model" && (
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
                            selected={props.selectedModelName}
                            onSelect={props.onSelectModel}
                            preferredDigest={props.preferredDigest}
                            onCancel={props.onCancelDownload}
                        />
                        <Button
                            variant="outline"
                            size="sm"
                            className="h-8 justify-start gap-2 text-xs"
                            onClick={props.onOpenFineTune}
                            disabled={!props.canFineTune}
                            title={props.canFineTune
                                ? "Fine-tune (LoRA) the loaded model on your own data"
                                : props.fineTuneReason}
                        >
                            <Sparkles className="size-3.5" />
                            {props.canFineTune ? "Fine-tune…" : "Fine-tune… (Unavailable)"}
                        </Button>
                    </section>
                )}

                {tab === "generation" && (
                    <>
                        <section className="flex flex-col gap-2">
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
                    </>
                )}

                {tab === "speech" && (
                    <section className="flex flex-col gap-2">
                        <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Speech input</span>
                        <SpeechInputSettings
                            voice={props.voice}
                            onVoiceChange={props.onVoiceChange}
                            canRecord={props.canRecord}
                        />
                    </section>
                )}
            </div>
        </div>
    );
}

interface TabButtonProps {
    label: string;
    active: boolean;
    onClick: () => void;
}

function TabButton({ label, active, onClick }: TabButtonProps) {
    return (
        <button
            type="button"
            role="tab"
            aria-selected={active}
            onClick={onClick}
            className={cn(
                "-mb-px border-b-2 px-3 py-1.5 text-xs font-medium transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                active ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground",
            )}
        >
            {label}
        </button>
    );
}
