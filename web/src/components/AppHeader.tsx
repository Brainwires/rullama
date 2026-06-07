/// Top app header: the rullama wordmark, active-conversation title, adapter
/// badge, and the Chat · Voice · Settings segmented tab switcher. Purely
/// presentational — extracted from App.tsx.

import { History, MessageSquare, AudioLines, Settings } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type View = "chat" | "voice" | "settings";

interface AppHeaderProps {
    view: View;
    onSelectView: (v: View) => void;
    historyOpen: boolean;
    onToggleHistory: () => void;
    activeTitle?: string;
    activeAdapter: string | null;
}

export function AppHeader({
    view,
    onSelectView,
    historyOpen,
    onToggleHistory,
    activeTitle,
    activeAdapter,
}: AppHeaderProps) {
    // `min-h-12` not `h-12` so the safe-area-inset-top padding actually grows
    // the header on iPhones with a notch / Dynamic Island — fixed h-12 was
    // stuffing all content under the status bar in standalone PWA mode.
    return (
        <header className="flex min-h-12 shrink-0 items-center gap-2 border-b border-border bg-card/50 px-3 safe-top">
            {view === "chat" && (
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8"
                    onClick={onToggleHistory}
                    title="Toggle conversation history"
                    aria-pressed={historyOpen}
                >
                    <History />
                </Button>
            )}
            <span className="font-semibold tracking-tight">rullama</span>
            {view === "chat" && activeTitle && (
                <span className="hidden truncate text-xs text-muted-foreground sm:inline">
                    / {activeTitle}
                </span>
            )}
            {activeAdapter && (
                <Badge tone="info" className="hidden text-[10px] sm:inline-flex">
                    adapter: {activeAdapter}
                </Badge>
            )}
            <div className="ml-auto flex items-center gap-1">
                {/* Tab switcher — segmented control. */}
                <div className="flex gap-0.5 rounded-md border border-border bg-muted/30 p-0.5">
                    <button
                        type="button"
                        onClick={() => onSelectView("chat")}
                        aria-pressed={view === "chat"}
                        className={cn(
                            "flex h-7 items-center gap-1 rounded px-2 text-xs transition-colors",
                            view === "chat"
                                ? "bg-background text-foreground shadow-sm"
                                : "text-muted-foreground hover:bg-background/50",
                        )}
                        title="Chat"
                    >
                        <MessageSquare className="size-3.5" />
                        <span className="hidden sm:inline">Chat</span>
                    </button>
                    <button
                        type="button"
                        onClick={() => onSelectView("voice")}
                        aria-pressed={view === "voice"}
                        className={cn(
                            "flex h-7 items-center gap-1 rounded px-2 text-xs transition-colors",
                            view === "voice"
                                ? "bg-background text-foreground shadow-sm"
                                : "text-muted-foreground hover:bg-background/50",
                        )}
                        title="Voice"
                    >
                        <AudioLines className="size-3.5" />
                        <span className="hidden sm:inline">Voice</span>
                    </button>
                    <button
                        type="button"
                        onClick={() => onSelectView("settings")}
                        aria-pressed={view === "settings"}
                        className={cn(
                            "flex h-7 items-center gap-1 rounded px-2 text-xs transition-colors",
                            view === "settings"
                                ? "bg-background text-foreground shadow-sm"
                                : "text-muted-foreground hover:bg-background/50",
                        )}
                        title="Settings"
                    >
                        <Settings className="size-3.5" />
                        <span className="hidden sm:inline">Settings</span>
                    </button>
                </div>
            </div>
        </header>
    );
}
