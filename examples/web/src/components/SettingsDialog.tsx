import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { type SamplingOptions } from "@/lib/types";
import { ChevronDown, ChevronRight, Settings2 } from "lucide-react";

interface Props {
    systemPrompt: string;
    onSystemPromptChange: (s: string) => void;
    sampling: SamplingOptions;
    onSamplingChange: (s: SamplingOptions) => void;
    maxTokens: number;
    onMaxTokensChange: (n: number) => void;
}

/** Inline collapsible settings panel — works well on mobile (no modal). */
export function SettingsDialog(props: Props) {
    const [open, setOpen] = useState(false);

    const setS = (patch: Partial<SamplingOptions>) =>
        props.onSamplingChange({ ...props.sampling, ...patch });

    return (
        <div className="rounded-md border border-border bg-card/30">
            <Button
                variant="ghost"
                onClick={() => setOpen(!open)}
                className="h-9 w-full justify-start px-3 text-xs text-muted-foreground"
            >
                {open ? <ChevronDown /> : <ChevronRight />}
                <Settings2 />
                System prompt &amp; sampling
            </Button>
            {open && (
                <div className="space-y-3 border-t border-border p-3">
                    <Textarea
                        value={props.systemPrompt}
                        onChange={(e) => props.onSystemPromptChange(e.target.value)}
                        placeholder='Optional. e.g. "You are a helpful assistant who answers in pirate speak."'
                        className="min-h-[3rem]"
                    />
                    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                        <label className="text-xs">
                            <span className="text-muted-foreground">temp</span>
                            <Input type="number" step={0.1} min={0} max={2}
                                value={props.sampling.temperature}
                                onChange={(e) => setS({ temperature: parseFloat(e.target.value) })}
                            />
                        </label>
                        <label className="text-xs">
                            <span className="text-muted-foreground">top_k</span>
                            <Input type="number" step={1} min={0}
                                value={props.sampling.top_k}
                                onChange={(e) => setS({ top_k: parseInt(e.target.value || "0", 10) })}
                            />
                        </label>
                        <label className="text-xs">
                            <span className="text-muted-foreground">top_p</span>
                            <Input type="number" step={0.05} min={0} max={1}
                                value={props.sampling.top_p}
                                onChange={(e) => setS({ top_p: parseFloat(e.target.value) })}
                            />
                        </label>
                        <label className="text-xs">
                            <span className="text-muted-foreground">max_tokens</span>
                            <Input type="number" step={8} min={1}
                                value={props.maxTokens}
                                onChange={(e) => props.onMaxTokensChange(parseInt(e.target.value || "0", 10))}
                            />
                        </label>
                    </div>
                </div>
            )}
        </div>
    );
}
