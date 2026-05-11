import { useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
        <Card>
            <CardHeader>
                <CardTitle>Chat</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
                <div
                    ref={historyRef}
                    className="flex max-h-[55vh] min-h-[10rem] flex-col gap-2 overflow-y-auto rounded-md border border-border bg-background/40 p-2 sm:max-h-[28rem]"
                >
                    {props.messages.length === 0 ? (
                        <p className="m-auto text-xs text-muted-foreground">
                            Load a model and say hi.
                        </p>
                    ) : (
                        props.messages.map((m, i) => <MessageBubble key={i} msg={m} />)
                    )}
                </div>

                <div className="flex flex-wrap gap-2 safe-bottom">
                    <Input
                        placeholder='e.g. "What is the capital of France?"'
                        value={props.prompt}
                        onChange={(e) => props.onPromptChange(e.target.value)}
                        onKeyDown={onKeyDown}
                        disabled={!props.canType}
                        className="flex-1 min-w-0"
                    />
                    <Button onClick={props.onSend}      disabled={!props.canSend}><Send /> Send</Button>
                    <Button onClick={props.onStop}      disabled={!props.canStop}  variant="destructive"><Square /></Button>
                    <Button onClick={props.onReset}     disabled={!props.canReset} variant="outline"><RotateCcw /></Button>
                </div>

                {props.statusLine && (
                    <p className="text-xs text-muted-foreground">{props.statusLine}</p>
                )}
            </CardContent>
        </Card>
    );
}
