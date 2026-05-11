import { Button } from "@/components/ui/button";
import { type ConversationRow } from "@/lib/inference";
import { cn } from "@/lib/utils";
import { Plus, Trash2 } from "lucide-react";

interface Props {
    conversations: ConversationRow[];
    activeId:      string | null;
    onSelect:      (id: string) => void;
    onCreate:      () => void;
    onDelete:      (id: string) => void;
}

function relativeTime(ms: number): string {
    const diff = Date.now() - ms;
    if (diff <    60_000) return "just now";
    if (diff < 3_600_000) return `${Math.floor(diff /    60_000)}m ago`;
    if (diff <  86_400_000) return `${Math.floor(diff /  3_600_000)}h ago`;
    if (diff < 604_800_000) return `${Math.floor(diff / 86_400_000)}d ago`;
    return new Date(ms).toLocaleDateString();
}

/** Slide-out drawer panel listing persisted conversations newest-first. */
export function ConversationList(props: Props) {
    return (
        <div className="border-b border-border bg-background/60 px-3 py-2 sm:px-4">
            <div className="mx-auto flex max-w-3xl flex-col gap-1.5">
                <div className="flex items-center justify-between">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        History ({props.conversations.length})
                    </span>
                    <Button
                        size="sm"
                        variant="outline"
                        className="h-7 px-2 text-xs"
                        onClick={props.onCreate}
                    >
                        <Plus />
                        New chat
                    </Button>
                </div>
                {props.conversations.length === 0 ? (
                    <p className="px-1 py-1 text-xs text-muted-foreground">
                        No prior chats yet. Send a message to start one.
                    </p>
                ) : (
                    <ul className="flex max-h-[40vh] flex-col gap-0.5 overflow-y-auto pr-1">
                        {props.conversations.map((c) => (
                            <li
                                key={c.id}
                                className={cn(
                                    "group flex items-center gap-2 rounded px-2 py-1 text-xs",
                                    c.id === props.activeId
                                        ? "bg-primary/10 text-foreground"
                                        : "hover:bg-muted/60",
                                )}
                            >
                                <button
                                    type="button"
                                    onClick={() => props.onSelect(c.id)}
                                    className="flex flex-1 items-center justify-between gap-2 truncate text-left"
                                    title={c.title}
                                >
                                    <span className="truncate">{c.title}</span>
                                    <span className="shrink-0 text-[0.6rem] text-muted-foreground">
                                        {relativeTime(c.updated_at)}
                                    </span>
                                </button>
                                <button
                                    type="button"
                                    onClick={() => props.onDelete(c.id)}
                                    className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive"
                                    title="Delete conversation"
                                    aria-label={`Delete ${c.title}`}
                                >
                                    <Trash2 className="h-3.5 w-3.5" />
                                </button>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}
