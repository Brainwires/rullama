import { useEffect, useMemo, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type ChatMessage } from "@/lib/types";
import { renderMarkdown } from "@/lib/markdown";
import { cn } from "@/lib/utils";
import { Send, Square, Plus } from "lucide-react";

interface Props {
    messages:    ChatMessage[];
    canType:     boolean;   // input enabled (model ready, not busy)
    canSend:     boolean;   // Send button enabled (canType AND prompt non-empty)
    canStop:     boolean;
    canNewChat:  boolean;
    prompt:      string;
    onPromptChange: (s: string) => void;
    onSend:      () => void;
    onStop:      () => void;
    onNewChat:   () => void;
    statusLine?: string;
    className?:  string;
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
    // Memoize the rendered HTML so partial streaming doesn't pay the parse
    // cost more than once per token batch — marked is fast but DOMPurify
    // can be the slow part on long replies.
    const html = useMemo(
        () => msg.content ? renderMarkdown(msg.content) : "",
        [msg.content],
    );
    return (
        <div
            className={cn(
                "rounded-md border-l-2 p-3 text-sm break-words animate-fade-in",
                msg.role === "user"   && "border-primary bg-primary/10",
                msg.role === "model"  && "border-muted-foreground bg-muted/50",
                msg.role === "system" && "border-yellow-500 bg-yellow-500/10",
            )}
        >
            <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                {msg.role}
            </div>
            {msg.content ? (
                <div className="markdown" dangerouslySetInnerHTML={{ __html: html }} />
            ) : (
                <span className="inline-block animate-pulse text-muted-foreground">▍</span>
            )}
        </div>
    );
}

/**
 * Chat panel meant to fill its parent's remaining height. History scrolls
 * internally; input row pins to the bottom. The outer page should NOT scroll.
 */
export function ChatPanel(props: Props) {
    const historyRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to the latest message.
    useEffect(() => {
        if (historyRef.current) {
            historyRef.current.scrollTop = historyRef.current.scrollHeight;
        }
    }, [props.messages]);

    // Delegated click handler for `.codeblock-copy` buttons rendered by
    // the markdown pipeline. One listener for the whole history beats a
    // ref per code block.
    useEffect(() => {
        const root = historyRef.current;
        if (!root) return;
        const onClick = async (e: MouseEvent) => {
            const target = e.target as HTMLElement | null;
            const btn = target?.closest<HTMLElement>("[data-bw-copy]");
            if (!btn) return;
            const pre = btn.nextElementSibling as HTMLElement | null;
            const code = pre?.querySelector("code");
            if (!code) return;
            const text = code.textContent ?? "";
            try {
                await navigator.clipboard.writeText(text);
                const prev = btn.textContent;
                btn.textContent = "copied";
                setTimeout(() => { btn.textContent = prev ?? "copy"; }, 1200);
            } catch { /* clipboard denied — silent */ }
        };
        root.addEventListener("click", onClick);
        return () => root.removeEventListener("click", onClick);
    }, []);

    const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !e.shiftKey && props.canSend) {
            e.preventDefault();
            props.onSend();
        }
    };

    return (
        <div className={cn("flex h-full min-h-0 flex-col", props.className)}>
            <div
                ref={historyRef}
                className="flex-1 min-h-0 overflow-y-auto px-3 py-2 sm:px-4"
            >
                {props.messages.length === 0 ? (
                    <p className="mt-8 text-center text-xs text-muted-foreground">
                        Load a model and say hi.
                    </p>
                ) : (
                    <div className="mx-auto flex max-w-3xl flex-col gap-2">
                        {props.messages.map((m, i) => <MessageBubble key={i} msg={m} />)}
                    </div>
                )}
            </div>

            {props.statusLine && (
                <p className="border-t border-border bg-background/60 px-3 py-1 text-[0.65rem] text-muted-foreground sm:px-4">
                    {props.statusLine}
                </p>
            )}

            <div className="flex gap-1.5 border-t border-border bg-background/80 px-2 py-2 safe-bottom sm:px-3">
                <Input
                    placeholder='Say something…'
                    value={props.prompt}
                    onChange={(e) => props.onPromptChange(e.target.value)}
                    onKeyDown={onKeyDown}
                    disabled={!props.canType}
                    className="flex-1 min-w-0"
                />
                {props.canStop ? (
                    <Button onClick={props.onStop} variant="destructive" title="Stop"><Square /></Button>
                ) : (
                    <Button onClick={props.onSend} disabled={!props.canSend} title="Send"><Send /></Button>
                )}
                <Button onClick={props.onNewChat} disabled={!props.canNewChat} variant="outline" title="New chat"><Plus /></Button>
            </div>
        </div>
    );
}
