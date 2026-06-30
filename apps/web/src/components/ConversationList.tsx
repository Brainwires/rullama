import { Button } from "@/components/ui/button";
import { type ConversationRow } from "@/lib/inference";
import { cn } from "@/lib/utils";
import { Plus, Trash2 } from "lucide-react";

interface Props {
    conversations: ConversationRow[];
    activeId:      string | null;
    /** Conversations whose generation is running right now / waiting in the
     *  serial queue. Drives the per-row activity indicators. */
    runningConvIds?: Set<string>;
    queuedConvIds?:  Set<string>;
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

/**
 * Full-height sidebar listing persisted conversations newest-first.
 * Title + "+ New chat" pin at the top; list scrolls below.
 */
export function ConversationList(props: Props) {
    return (
        <div className="flex h-full min-h-0 flex-col">
            <header className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    History
                </span>
                <Button
                    size="sm"
                    variant="outline"
                    className="h-7 px-2 text-xs"
                    onClick={props.onCreate}
                >
                    <Plus />
                    New
                </Button>
            </header>

            <div className="min-h-0 flex-1 overflow-y-auto p-1.5">
                {props.conversations.length === 0 ? (
                    <p className="px-2 py-2 text-xs text-muted-foreground">
                        No prior chats yet. Send a message to start one.
                    </p>
                ) : (
                    <ul className="flex flex-col gap-0.5">
                        {props.conversations.map((c) => {
                            const running = props.runningConvIds?.has(c.id) ?? false;
                            const queued  = !running && (props.queuedConvIds?.has(c.id) ?? false);
                            return (
                            <li
                                key={c.id}
                                className={cn(
                                    "group flex items-center gap-2 rounded px-2 py-1.5 text-xs",
                                    c.id === props.activeId
                                        ? "bg-primary/15 text-foreground"
                                        : "hover:bg-muted/60",
                                )}
                            >
                                <button
                                    type="button"
                                    onClick={() => props.onSelect(c.id)}
                                    className="flex min-w-0 flex-1 flex-col items-start text-left"
                                    title={c.title}
                                >
                                    <span className="flex w-full items-center gap-1.5">
                                        {running && (
                                            <span
                                                className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-primary"
                                                title="Generating…"
                                                aria-label="Generating"
                                            />
                                        )}
                                        {queued && (
                                            <span
                                                className="h-1.5 w-1.5 shrink-0 rounded-full bg-muted-foreground/50"
                                                title="Queued"
                                                aria-label="Queued"
                                            />
                                        )}
                                        <span className="truncate">{c.title}</span>
                                    </span>
                                    <span className="text-[0.6rem] text-muted-foreground">
                                        {running ? "generating…" : queued ? "queued" : relativeTime(c.updated_at)}
                                    </span>
                                </button>
                                <button
                                    type="button"
                                    onClick={() => props.onDelete(c.id)}
                                    className="opacity-0 transition-opacity group-hover:opacity-100 text-muted-foreground hover:text-destructive"
                                    title="Delete conversation"
                                    aria-label={`Delete ${c.title}`}
                                >
                                    <Trash2 className="h-3.5 w-3.5" />
                                </button>
                            </li>
                            );
                        })}
                    </ul>
                )}
            </div>
        </div>
    );
}
