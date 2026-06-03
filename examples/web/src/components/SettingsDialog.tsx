import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { ModelLoader, type ModelStatus } from "@/components/ModelLoader";
import { LogsTab } from "@/components/LogsTab";
import { type SamplingOptions } from "@/lib/types";
import { type ModelEntry } from "@/lib/api";
import { clampInt, clampNum, cn } from "@/lib/utils";
import { usePersistedState } from "@/lib/persisted";
import { type VoiceOptions } from "@/lib/voice";
import { hardResetAndReload } from "@/lib/restart";
import { Undo2, RefreshCw } from "lucide-react";

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

type TabKey = "general" | "sampling" | "logs";

/** Full-height sidebar: Model + Generation settings, sections scroll. */
export function SettingsDialog(props: Props) {
    const B = SETTINGS_BOUNDS;
    const setS = (patch: Partial<SamplingOptions>) =>
        props.onSamplingChange({ ...props.sampling, ...patch });

    // Persisted so the App-level "crashed last session" toast can
    // deep-link into the Logs tab: setting localStorage to "logs"
    // before switching the top-level view to "settings" lands here
    // on the right tab.
    const [tab, setTab] = usePersistedState<TabKey>("rullama:settings:tab", "general");

    // A previously-persisted "voice" tab (now moved to the Voice-training sidebar)
    // falls back to General.
    const activeTab: TabKey = (tab as string) === "voice" ? "general" : tab;

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
                    <Undo2 />
                    Defaults
                </Button>
            </header>

            <nav
                role="tablist"
                aria-label="Settings sections"
                className="flex shrink-0 border-b border-border px-3"
            >
                <TabButton
                    label="General"
                    active={activeTab === "general"}
                    onClick={() => setTab("general")}
                />
                <TabButton
                    label="Sampling"
                    active={activeTab === "sampling"}
                    onClick={() => setTab("sampling")}
                />
                {/* "Speech input" (VAD) settings now live in the Voice-training right
                    sidebar (see SpeechInputSettings), not here. */}
                <TabButton
                    label="Logs"
                    active={activeTab === "logs"}
                    onClick={() => setTab("logs")}
                    title="Diagnostic logs persisted across sessions (survives crashes)"
                />
            </nav>

            <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 py-3">
                {activeTab === "general" && (
                    <>
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
                            <p className="ml-6 text-[10px] leading-tight text-muted-foreground">
                                Tip: combining thinking with temperature &gt; 0.3 on attached
                                images or audio can produce wandering chain-of-thought that
                                reads like garbled text. Lower both for crisp multimodal
                                analysis.
                            </p>
                        </section>

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
                    </>
                )}

                {activeTab === "sampling" && (
                    <section className="flex flex-col gap-3">
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
                            label="repetition penalty"
                            value={props.sampling.repetition_penalty}
                            min={B.repetition_penalty.min} max={B.repetition_penalty.max} step={B.repetition_penalty.step}
                            fmt={(v) => v.toFixed(2)}
                            onChange={(v) => setS({
                                repetition_penalty: clampNum(v, B.repetition_penalty.min, B.repetition_penalty.max, B.repetition_penalty.fallback),
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
                )}

                {activeTab === "logs" && <LogsTab />}
            </div>
        </div>
    );
}

interface TabButtonProps {
    label: string;
    active: boolean;
    onClick: () => void;
    disabled?: boolean;
    title?: string;
}

function TabButton({ label, active, onClick, disabled, title }: TabButtonProps) {
    return (
        <button
            type="button"
            role="tab"
            aria-selected={active}
            disabled={disabled}
            title={title}
            onClick={onClick}
            className={cn(
                "-mb-px border-b-2 px-3 py-1.5 text-xs font-medium transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                "disabled:pointer-events-none disabled:opacity-50",
                active
                    ? "border-primary text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground",
            )}
        >
            {label}
        </button>
    );
}
