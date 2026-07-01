import { RefreshCw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { shortVersionLabel } from "@/lib/version";

interface Props {
    /** Server-side version detected on boot. */
    version: string;
    /** User clicked "Apply now" — host should kick off the
     *  coordinated cross-tab shutdown + reload. */
    onApply: () => void;
    /** User clicked "Later" — persist dismissed version, hide banner. */
    onDismiss: () => void;
}

/** Sticky-top "Update available" banner. Non-modal; the app remains
 *  usable behind it. Rendered conditionally by App.tsx only when an
 *  update is detected AND the user is not mid-generation (busy=false). */
export function UpdateBanner({ version, onApply, onDismiss }: Props) {
    return (
        <div
            className="sticky top-0 z-40 flex items-center gap-2 border-b border-primary/30 bg-primary/15 px-3 py-2 text-xs sm:px-4"
            role="status"
            aria-live="polite"
        >
            <RefreshCw className="h-3.5 w-3.5 shrink-0 text-primary" aria-hidden="true" />
            <span className="min-w-0 flex-1 truncate">
                <span className="font-medium">Update available</span>
                <span className="ml-2 font-mono tabular-nums text-muted-foreground">
                    {shortVersionLabel(version)}
                </span>
            </span>
            <Button
                size="sm"
                onClick={onApply}
                className="h-7 px-2 py-0 text-xs"
                title="Reload all tabs onto the new version"
            >
                Apply now
            </Button>
            <Button
                size="sm"
                variant="ghost"
                onClick={onDismiss}
                className="h-7 w-7 p-0"
                aria-label="Dismiss until next visit"
                title="Dismiss until next visit"
            >
                <X className="h-3.5 w-3.5" />
            </Button>
        </div>
    );
}
