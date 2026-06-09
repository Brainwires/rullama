import { Wrench } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ToolCall } from "@/lib/toolFormat";

interface Props {
    call: ToolCall;
}

/** One argument row: `key  value`. Objects/arrays are JSON-stringified. */
function ArgRow({ name, value }: { name: string; value: unknown }) {
    const text =
        typeof value === "string" ? value : JSON.stringify(value);
    return (
        <div className="flex gap-2 px-3 py-1 font-mono text-[0.7rem]">
            <span className="shrink-0 text-primary/80">{name}</span>
            <span className="break-all text-foreground/90">{text}</span>
        </div>
    );
}

/**
 * Styled block for a tool call the model emitted. Visual only — this does NOT
 * execute anything; it surfaces the structured call (name + arguments) so the
 * tool-calling output is legible instead of raw JSON in the prose. Sibling to
 * ThinkingBlock; intentionally stays expanded (the call is the artifact we
 * want to show), pulsing while the call is still streaming in.
 */
export function ToolCallBlock({ call }: Props) {
    const argEntries =
        call.arguments && typeof call.arguments === "object"
            ? Object.entries(call.arguments as Record<string, unknown>)
            : null;

    const label = call.pending
        ? call.name
            ? `Calling ${call.name}…`
            : "Tool call…"
        : `Tool call: ${call.name || "tool"}`;

    return (
        <div
            className={cn(
                "mb-2 overflow-hidden rounded-md border border-dashed border-primary/40 bg-primary/5 text-xs",
                call.pending && "animate-pulse",
            )}
        >
            <div className="flex items-center gap-1.5 border-b border-dashed border-primary/30 px-2.5 py-1.5 text-muted-foreground">
                <Wrench className="h-3.5 w-3.5 text-primary" />
                <span className="font-mono text-[0.7rem] text-foreground/90">{label}</span>
            </div>
            <div className="py-1">
                {argEntries && argEntries.length > 0 ? (
                    argEntries.map(([k, v]) => <ArgRow key={k} name={k} value={v} />)
                ) : argEntries ? (
                    <div className="px-3 py-1 text-[0.7rem] italic text-muted-foreground">
                        (no arguments)
                    </div>
                ) : (
                    // Arguments aren't a parseable object (malformed / streaming) —
                    // show the raw payload verbatim.
                    <pre className="overflow-x-auto px-3 py-1 font-mono text-[0.7rem] text-foreground/90 whitespace-pre-wrap break-all">
                        {String(call.arguments)}
                    </pre>
                )}
            </div>
        </div>
    );
}
