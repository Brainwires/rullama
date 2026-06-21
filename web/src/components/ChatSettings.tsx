import { useEffect, useState } from "react";
import { Loader2, Pencil, Sparkles, Undo2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
    AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
    AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
    AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { type Units } from "@/lib/tools";
import { SETTINGS_BOUNDS } from "@/components/SettingsDialog";
import { ModelLoader, LoadingModel, type ModelStatus } from "@/components/ModelLoader";
import { SpeechInputSettings } from "@/components/SpeechInputSettings";
import { type ModelEntry } from "@/lib/api";
import { type SamplingOptions } from "@/lib/types";
import { type VoiceOptions } from "@/lib/voice";
import { usePersistedState } from "@/lib/persisted";
import { putCloudKey, hasCloudKey } from "@/lib/cloud/keyvault";
import { providerLabel, type CloudProvider } from "@/lib/cloud/types";
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
    /** Abort an in-progress model download (keep the partial). */
    onCancelDownload?:  () => void;
    /** Abort an in-progress download AND delete the partial. */
    onCancelAndDeleteDownload?: () => void;
    /** Opens the full-screen Fine-tune (LoRA training) overlay. */
    onOpenFineTune: () => void;
    /** Whether fine-tuning can actually run here — model ready AND the device
     *  passes the training-capability probe (iOS is blocked by default). */
    canFineTune:    boolean;
    /** Why fine-tune is unavailable (tooltip), when `canFineTune` is false. */
    fineTuneReason: string;
    // Generation settings.
    systemPrompt: string;
    /** Persist a new system prompt AND re-warm the KV cache with it. The
     *  editor awaits this to show its "preparing…" state. */
    onSaveSystemPrompt: (s: string) => Promise<void>;
    sampling: SamplingOptions;
    onSamplingChange: (s: SamplingOptions) => void;
    maxTokens: number;
    onMaxTokensChange: (n: number) => void;
    thinking: boolean;
    onThinkingChange: (b: boolean) => void;
    toolMode: boolean;
    onToolModeChange: (b: boolean) => void;
    orchestratorMode: boolean;
    onOrchestratorModeChange: (b: boolean) => void;
    // Tools tab — weather (WeatherAPI.com, optional key) + GPS + news (GNews key).
    weatherApiKey: string;
    onWeatherApiKeyChange: (s: string) => void;
    newsApiKey: string;
    onNewsApiKeyChange: (s: string) => void;
    weatherUnits: Units;
    onWeatherUnitsChange: (u: Units) => void;
    onResetDefaults: () => void;
    // Cloud tab — BYOK proxy chat (Ollama Cloud + OpenAI). API keys are managed
    // in-tab via the encrypted vault; only the proxy override is a prop.
    cloudBaseOverride: string;
    onCloudBaseOverrideChange: (s: string) => void;
    // Speech input (chat mic → transcription) VAD config.
    voice: VoiceOptions;
    onVoiceChange: (v: VoiceOptions) => void;
    canRecord: boolean;
}

type TabKey = "model" | "generation" | "tools" | "cloud" | "speech";

/** Chat-tab right sidebar, tabbed because it carries a lot now:
 *  - **Model**: load/eject the inference model + the Fine-tune launcher.
 *  - **Generation**: system prompt, sampling knobs, thinking mode (+ Defaults reset).
 *  - **Speech**: chat mic → transcription VAD config.
 *  (Logs / app-data / High-VRAM toggle live in the global Settings view.) */
export function ChatSettings(props: Props) {
    const B = SETTINGS_BOUNDS;
    const setS = (patch: Partial<SamplingOptions>) => props.onSamplingChange({ ...props.sampling, ...patch });
    const [tab, setTab] = usePersistedState<TabKey>("chat:settings:tab", "model");

    // System-prompt editor: read-only by default, Edit → Save/Cancel.
    // `sysPreparing` covers the post-save warm so the user sees the model
    // being prepared before the new prompt takes effect.
    const [editingSys, setEditingSys] = useState(false);
    const [sysDraft, setSysDraft] = useState(props.systemPrompt);
    const [sysPreparing, setSysPreparing] = useState(false);
    const beginEditSys = () => { setSysDraft(props.systemPrompt); setEditingSys(true); };
    const cancelEditSys = () => { setSysDraft(props.systemPrompt); setEditingSys(false); };
    const saveSys = async () => {
        setSysPreparing(true);
        try {
            await props.onSaveSystemPrompt(sysDraft);
            setEditingSys(false);
        } finally {
            setSysPreparing(false);
        }
    };

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
                <TabButton label="Tools" active={tab === "tools"} onClick={() => setTab("tools")} />
                <TabButton label="Cloud" active={tab === "cloud"} onClick={() => setTab("cloud")} />
                <TabButton label="Speech" active={tab === "speech"} onClick={() => setTab("speech")} />
            </nav>

            <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 py-3">
                {tab === "model" && (
                    <section className="flex flex-col gap-2">
                        <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Model</span>
                        {/* While a model is loading/preparing, swap the picker for
                            the loading view (Stop stays available) so its controls
                            can't be touched mid-load. */}
                        {(props.modelStatus === "loading" || props.modelStatus === "preparing") ? (
                            <LoadingModel
                                status={props.modelStatus}
                                percent={props.loadingPercent}
                                label={props.loadingLabel}
                                modelName={props.selectedModelName || props.statusText}
                                onCancel={props.onCancelDownload}
                                onCancelDelete={props.onCancelAndDeleteDownload}
                            />
                        ) : (
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
                                onCancelDelete={props.onCancelAndDeleteDownload}
                            />
                        )}
                        <Button
                            variant="outline"
                            size="sm"
                            className="h-8 justify-start gap-2 text-xs"
                            onClick={props.onOpenFineTune}
                            disabled={!props.canFineTune
                                || props.modelStatus === "loading"
                                || props.modelStatus === "preparing"}
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
                            {editingSys ? (
                                <>
                                    <Textarea
                                        value={sysDraft}
                                        onChange={(e) => setSysDraft(e.target.value)}
                                        placeholder='Optional. e.g. "You are a pirate."'
                                        className="min-h-[4rem] text-xs"
                                        autoFocus
                                        disabled={sysPreparing}
                                    />
                                    <div className="flex items-center gap-2">
                                        <Button
                                            size="sm"
                                            className="h-7 px-2 text-xs"
                                            onClick={() => void saveSys()}
                                            disabled={sysPreparing}
                                        >
                                            {sysPreparing && <Loader2 className="animate-spin" />}
                                            {sysPreparing ? "Preparing…" : "Save"}
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant="ghost"
                                            className="h-7 px-2 text-xs"
                                            onClick={cancelEditSys}
                                            disabled={sysPreparing}
                                        >
                                            Cancel
                                        </Button>
                                        {sysPreparing && (
                                            <span className="text-[11px] text-muted-foreground">
                                                preparing model with the new prompt…
                                            </span>
                                        )}
                                    </div>
                                </>
                            ) : (
                                <>
                                    <div className="min-h-[3rem] whitespace-pre-wrap break-words rounded-md border border-input bg-muted/30 px-2 py-1.5 text-xs text-foreground/90">
                                        {props.systemPrompt.trim() || (
                                            <span className="italic text-muted-foreground">No system prompt set.</span>
                                        )}
                                    </div>
                                    <Button
                                        size="sm"
                                        variant="outline"
                                        className="h-7 self-start px-2 text-xs"
                                        onClick={beginEditSys}
                                    >
                                        <Pencil />
                                        Edit
                                    </Button>
                                </>
                            )}
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

                {tab === "tools" && (
                    <>
                        <section className="flex flex-col gap-3">
                            <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Tool calling</span>
                            <div className="flex items-start justify-between gap-3 text-xs text-muted-foreground">
                                <span>
                                    {props.toolMode ? "Enabled" : "Disabled"} — let the model invoke{" "}
                                    <code className="rounded bg-muted px-1">get_weather</code> and others via{" "}
                                    <code className="rounded bg-muted px-1">&lt;tool_call&gt;</code> blocks.
                                </span>
                                <ConfirmToggleButton
                                    enabled={props.toolMode}
                                    onToggle={props.onToolModeChange}
                                    label="tool calling"
                                    modelReady={props.modelStatus === "ready"}
                                />
                            </div>
                            {props.toolMode && (
                                <div className="ml-3 flex items-start justify-between gap-3 border-l border-border pl-3 text-xs text-muted-foreground">
                                    <span>
                                        Programmatic mode <span className="opacity-70">(experimental)</span> — the model writes one{" "}
                                        <code className="rounded bg-muted px-1">Rhai</code> script orchestrating many tools at once;
                                        only the final answer returns. Falls back to <code className="rounded bg-muted px-1">&lt;tool_call&gt;</code> on error.
                                    </span>
                                    <ConfirmToggleButton
                                        enabled={props.orchestratorMode}
                                        onToggle={props.onOrchestratorModeChange}
                                        label="programmatic mode"
                                        modelReady={props.modelStatus === "ready"}
                                    />
                                </div>
                            )}
                        </section>

                        <section className="flex flex-col gap-2 border-t border-border pt-3">
                            <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Weather</span>
                            <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                                <span className="flex items-center justify-between">
                                    <span>WeatherAPI.com key</span>
                                    <a
                                        href="https://www.weatherapi.com/my/"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-primary hover:underline"
                                    >
                                        Get a free key
                                    </a>
                                </span>
                                <Input
                                    type="password"
                                    autoComplete="off"
                                    spellCheck={false}
                                    value={props.weatherApiKey}
                                    onChange={(e) => props.onWeatherApiKeyChange(e.target.value)}
                                    placeholder="Optional — leave blank to use free Open-Meteo"
                                    className="h-8 text-xs"
                                />
                                <span className="text-[10px] leading-tight text-muted-foreground">
                                    With a key, weather comes from WeatherAPI.com (richer data). Without one,
                                    it falls back to the free, keyless Open-Meteo — so weather works either way.
                                </span>
                            </label>

                            <div className="flex items-center justify-between pt-1">
                                <span className="text-xs text-muted-foreground">Units</span>
                                <div className="inline-flex overflow-hidden rounded-md border border-border">
                                    {(["metric", "imperial"] as Units[]).map((u) => (
                                        <button
                                            key={u}
                                            type="button"
                                            onClick={() => props.onWeatherUnitsChange(u)}
                                            className={cn(
                                                "px-2.5 py-1 text-xs transition-colors",
                                                props.weatherUnits === u
                                                    ? "bg-primary text-primary-foreground"
                                                    : "bg-background text-muted-foreground hover:text-foreground",
                                            )}
                                        >
                                            {u === "metric" ? "°C" : "°F"}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </section>

                        <section className="flex flex-col gap-2 border-t border-border pt-3">
                            <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">News</span>
                            <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                                <span className="flex items-center justify-between">
                                    <span>GNews API key</span>
                                    <a
                                        href="https://gnews.io/"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-primary hover:underline"
                                    >
                                        Get a free key
                                    </a>
                                </span>
                                <Input
                                    type="password"
                                    autoComplete="off"
                                    spellCheck={false}
                                    value={props.newsApiKey}
                                    onChange={(e) => props.onNewsApiKeyChange(e.target.value)}
                                    placeholder="Optional — required for the news tool"
                                    className="h-8 text-xs"
                                />
                                <span className="text-[10px] leading-tight text-muted-foreground">
                                    Enables the get_news tool (gnews.io). Without a key, the model answers from its own knowledge.
                                </span>
                            </label>
                        </section>
                    </>
                )}

                {tab === "cloud" && (
                    <>
                        <section className="flex flex-col gap-2">
                            <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Cloud chat (BYOK)</span>
                            <p className="text-[11px] leading-tight text-muted-foreground">
                                Run chat on a provider's servers when a model is too big for this
                                device. Pick a cloud model from the <strong>Model</strong> tab to use it.
                                Your messages and key are sent off-device via the rullama proxy —
                                keys are stored <strong>encrypted on this device</strong> and never
                                saved on the server. RAG still uses the on-device embedder, so your
                                documents stay local.
                            </p>
                        </section>
                        <section className="flex flex-col gap-2 border-t border-border pt-3">
                            <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Ollama Cloud</span>
                            <CloudKeyInput provider="ollama" />
                        </section>
                        <section className="flex flex-col gap-2 border-t border-border pt-3">
                            <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">OpenAI</span>
                            <CloudKeyInput provider="openai" />
                        </section>
                        <section className="flex flex-col gap-2 border-t border-border pt-3">
                            <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">Advanced</span>
                            <label className="flex flex-col gap-1 text-xs text-muted-foreground">
                                <span>Proxy override (optional)</span>
                                <Input
                                    type="text"
                                    autoComplete="off"
                                    spellCheck={false}
                                    value={props.cloudBaseOverride}
                                    onChange={(e) => props.onCloudBaseOverrideChange(e.target.value)}
                                    placeholder="https://your-worker.workers.dev"
                                    className="h-8 text-xs"
                                />
                                <span className="text-[10px] leading-tight text-muted-foreground">
                                    Point cloud calls at your OWN Cloudflare Worker instead of the
                                    built-in proxy. Leave blank to use the default.
                                </span>
                            </label>
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

/**
 * Enable/disable toggle for a setting that's part of the system prompt — so
 * flipping it rebuilds the warmed system-prompt KV cache (the per-model
 * pre-prepare). A confirmation makes that cost explicit instead of silently
 * re-warming in the background. The actual re-warm fires from App's sysWarmSig
 * effect once the setting changes; this dialog just gates the change.
 */
function ConfirmToggleButton({
    enabled, onToggle, label, modelReady,
}: {
    enabled: boolean;
    onToggle: (next: boolean) => void;
    label: string;
    modelReady: boolean;
}) {
    const verb = enabled ? "Disable" : "Enable";
    return (
        <AlertDialog>
            <AlertDialogTrigger asChild>
                <Button
                    variant={enabled ? "secondary" : "default"}
                    size="sm"
                    className="h-7 shrink-0 px-3 text-xs"
                >
                    {verb}
                </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
                <AlertDialogHeader>
                    <AlertDialogTitle>{verb} {label}?</AlertDialogTitle>
                    <AlertDialogDescription>
                        This changes the system prompt, so the cached system prompt will be
                        rebuilt{modelReady
                            ? " now — a brief re-prepare before your next chat."
                            : " the next time a model loads."}
                    </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={() => onToggle(!enabled)}>{verb}</AlertDialogAction>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>
    );
}

/** Where each provider issues API keys. */
const CLOUD_KEY_LINK: Record<CloudProvider, string> = {
    ollama: "https://ollama.com/settings/keys",
    openai: "https://platform.openai.com/api-keys",
};

/** A per-provider BYOK key input backed by the encrypted vault. The stored key
 *  is never read back into the field (it's encrypted); we only show whether one
 *  is set, let the user paste a new one (Save), or Clear it. */
function CloudKeyInput({ provider }: { provider: CloudProvider }) {
    const [value, setValue] = useState("");
    const [saved, setSaved] = useState<boolean | null>(null); // null = loading
    const [busy, setBusy] = useState(false);

    useEffect(() => {
        let stop = false;
        void hasCloudKey(provider).then((p) => { if (!stop) setSaved(p); });
        return () => { stop = true; };
    }, [provider]);

    const save = async () => {
        const v = value.trim();
        if (!v) return;
        setBusy(true);
        try { await putCloudKey(provider, v); setSaved(true); setValue(""); }
        finally { setBusy(false); }
    };
    const clear = async () => {
        setBusy(true);
        try { await putCloudKey(provider, ""); setSaved(false); setValue(""); }
        finally { setBusy(false); }
    };

    return (
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            <span className="flex items-center justify-between">
                <span>{providerLabel(provider)} API key</span>
                <a
                    href={CLOUD_KEY_LINK[provider]}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                >
                    Get a key
                </a>
            </span>
            <div className="flex items-center gap-1">
                <Input
                    type="password"
                    autoComplete="off"
                    spellCheck={false}
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    placeholder={saved ? "•••••• saved — type to replace" : "Paste your API key"}
                    className="h-8 flex-1 text-xs"
                    disabled={busy}
                />
                <Button
                    size="sm"
                    className="h-8 px-2 text-xs"
                    onClick={() => void save()}
                    disabled={busy || !value.trim()}
                >
                    Save
                </Button>
            </div>
            <span className="flex items-center justify-between text-[10px] leading-tight">
                <span className={saved ? "text-emerald-600 dark:text-emerald-500" : "text-muted-foreground"}>
                    {saved === null ? "…" : saved ? "✓ Key stored (encrypted on this device)" : "No key set"}
                </span>
                {saved && (
                    <button
                        type="button"
                        className="text-destructive hover:underline disabled:opacity-50"
                        onClick={() => void clear()}
                        disabled={busy}
                    >
                        Clear
                    </button>
                )}
            </span>
        </label>
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
