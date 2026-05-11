import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { type SamplingOptions } from "@/lib/types";

interface Props {
    systemPrompt: string;
    onSystemPromptChange: (s: string) => void;
    sampling: SamplingOptions;
    onSamplingChange: (s: SamplingOptions) => void;
    maxTokens: number;
    onMaxTokensChange: (n: number) => void;
    thinking: boolean;
    onThinkingChange: (b: boolean) => void;
}

/**
 * Settings drawer — renders directly when `open` (controlled by the parent
 * via conditional rendering). No internal collapse state; the toolbar's
 * settings button owns visibility so it can co-exist with the rest of a
 * compact top strip.
 */
export function SettingsDialog(props: Props) {
    const setS = (patch: Partial<SamplingOptions>) =>
        props.onSamplingChange({ ...props.sampling, ...patch });

    return (
        <div className="border-b border-border bg-background/60 px-3 py-2 sm:px-4">
            <div className="mx-auto flex max-w-3xl flex-col gap-2">
                <Textarea
                    value={props.systemPrompt}
                    onChange={(e) => props.onSystemPromptChange(e.target.value)}
                    placeholder='Optional system prompt — e.g. "You are a pirate."'
                    className="min-h-[3rem] text-xs"
                />
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <label className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        temp
                        <Input
                            type="number" step={0.1} min={0} max={2}
                            value={props.sampling.temperature}
                            onChange={(e) => setS({ temperature: parseFloat(e.target.value) })}
                            className="mt-0.5 h-7 text-xs"
                        />
                    </label>
                    <label className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        top_k
                        <Input
                            type="number" step={1} min={0}
                            value={props.sampling.top_k}
                            onChange={(e) => setS({ top_k: parseInt(e.target.value || "0", 10) })}
                            className="mt-0.5 h-7 text-xs"
                        />
                    </label>
                    <label className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        top_p
                        <Input
                            type="number" step={0.05} min={0} max={1}
                            value={props.sampling.top_p}
                            onChange={(e) => setS({ top_p: parseFloat(e.target.value) })}
                            className="mt-0.5 h-7 text-xs"
                        />
                    </label>
                    <label className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        max_tokens
                        <Input
                            type="number" step={8} min={1}
                            value={props.maxTokens}
                            onChange={(e) => props.onMaxTokensChange(parseInt(e.target.value || "0", 10))}
                            className="mt-0.5 h-7 text-xs"
                        />
                    </label>
                </div>
                <label
                    className="flex items-center gap-2 text-xs text-muted-foreground"
                    title='Silently prepend "<|think|>" to the system prompt so Gemma 4 emits its internal reasoning trace before the answer. Not shown in the chat history.'
                >
                    <input
                        type="checkbox"
                        checked={props.thinking}
                        onChange={(e) => props.onThinkingChange(e.target.checked)}
                        className="h-3.5 w-3.5 cursor-pointer"
                    />
                    Thinking mode (prepend <code className="rounded bg-muted px-1">&lt;|think|&gt;</code> silently)
                </label>
            </div>
        </div>
    );
}
