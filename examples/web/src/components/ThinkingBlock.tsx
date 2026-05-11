import { useEffect, useState } from "react";
import { Brain, ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
    text:       string;
    isThinking: boolean;   // still mid-thought (no close marker yet)
    isComplete: boolean;   // close marker seen — can auto-collapse
}

/**
 * Styled, collapsible reasoning block extracted from a Gemma 4 reply.
 * Auto-expands while streaming, auto-collapses once the close marker
 * arrives. User can override either way by clicking the header.
 */
export function ThinkingBlock({ text, isThinking, isComplete }: Props) {
    // Default to expanded while streaming; collapse once complete.
    const [expanded, setExpanded] = useState(true);
    const [userToggled, setUserToggled] = useState(false);

    // Auto-collapse on completion (unless the user already overrode).
    useEffect(() => {
        if (isComplete && !userToggled) setExpanded(false);
    }, [isComplete, userToggled]);

    const toggle = () => {
        setUserToggled(true);
        setExpanded((v) => !v);
    };

    const label = isThinking
        ? "Thinking…"
        : isComplete
            ? "Thought"
            : "Thinking";

    return (
        <div
            className={cn(
                "mb-2 rounded-md border border-dashed border-border bg-muted/30 text-xs",
                isThinking && "border-primary/40 bg-primary/5",
            )}
        >
            <button
                type="button"
                onClick={toggle}
                className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-muted-foreground hover:text-foreground"
                aria-expanded={expanded}
            >
                {expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                <Brain className={cn("h-3.5 w-3.5", isThinking && "animate-pulse text-primary")} />
                <span className="uppercase tracking-wider text-[0.65rem]">{label}</span>
                {isThinking && <span className="ml-auto animate-pulse text-primary">▍</span>}
            </button>
            {expanded && (
                <div className="border-t border-dashed border-border/60 px-3 py-2 italic text-muted-foreground whitespace-pre-wrap break-words">
                    {text || (
                        <span className="inline-block animate-pulse">▍</span>
                    )}
                </div>
            )}
        </div>
    );
}
