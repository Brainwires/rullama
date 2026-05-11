import { useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Send, Square, RotateCcw } from "lucide-react";

interface Props {
    messages:    ChatMessage[];
    canType:     boolean;   // input enabled (model ready, not busy)
    canSend:     boolean;   // Send button enabled (canType AND prompt non-empty)
    canStop:     boolean;
    canReset:    boolean;
    prompt:      string;
    onPromptChange: (s: string) => void;
    onSend:      () => void;
    onStop:      () => void;
    onReset:     () => void;
    statusLine?: string;
    className?:  string;
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
    return (
        <div
            className={cn(
                "rounded-md border-l-2 p-3 text-sm whitespace-pre-wrap break-words animate-fade-in",
                msg.role === "user"   && "border-primary bg-primary/10",
                msg.role === "model"  && "border-muted-foreground bg-muted/50",
                msg.role === "system" && "border-yellow-500 bg-yellow-500/10",
            )}
        >
            <div className="mb-1 text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                {msg.role}
            </div>
            {msg.content || (
                <span className="inline-block animate-pulse text-muted-foreground">▍</span>
            )}
        </div>
    );
}

/**
 * Chat panel meant to fill its parent's remaining flex height. History scrolls
 * internally; input row pins to the bottom. The outer page should NOT scroll.
 */
export function ChatPanel(props: Props) {
    const historyRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to the latest message
    useEffect(() => {
        if (historyRef.current) {
            historyRef.current.scrollTop = historyRef.current.scrollHeight;
        }
    }, [props.messages]);

    const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !e.shiftKey && props.canSend) {
            e.preventDefault();
            props.onSend();
        }
    };

    return (
        <div className={cn("flex min-h-0 flex-1 flex-col", props.className)}>
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
                <Button onClick={props.onSend}  disabled={!props.canSend}><Send /></Button>
                <Button onClick={props.onStop}  disabled={!props.canStop}  variant="destructive"><Square /></Button>
                <Button onClick={props.onReset} disabled={!props.canReset} variant="outline"><RotateCcw /></Button>
            </div>
        </div>
    );
}
