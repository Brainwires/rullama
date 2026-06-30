// Full-screen Knowledge Base modal. Opened from Chat → Tools → "Knowledge
// Base". Wraps the KnowledgeTab content (search + document management) over the
// chat view; the embedder auto-loads when this mounts (see KnowledgeTab).

import { useEffect } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { KnowledgeTab } from "@/components/KnowledgeTab";

interface Props {
    activeConvId: string | null;
    onClose: () => void;
}

export function KnowledgeModal({ activeConvId, onClose }: Props) {
    // Esc closes (matches the training overlay's behaviour).
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [onClose]);

    return (
        <div
            className="fixed inset-0 z-50 flex flex-col bg-background"
            role="dialog"
            aria-modal="true"
            aria-label="Knowledge base"
        >
            <header className="flex min-h-12 shrink-0 items-center justify-between border-b border-border px-4 safe-top">
                <span className="text-sm font-semibold">Knowledge base</span>
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-muted-foreground hover:text-foreground"
                    onClick={onClose}
                    aria-label="Close"
                    title="Close (Esc)"
                >
                    <X className="size-4" />
                </Button>
            </header>
            <div className="min-h-0 flex-1 overflow-hidden safe-bottom">
                <KnowledgeTab activeConvId={activeConvId} />
            </div>
        </div>
    );
}
