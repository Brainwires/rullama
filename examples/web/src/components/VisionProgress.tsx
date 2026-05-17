import { Loader2 } from "lucide-react";

export interface VisionProgressState {
    /** 1-based: "image 2 of 3". */
    imageIdx: number;
    /** Total images queued in this turn. 1 hides the "image X/Y" prefix. */
    nImages:  number;
    /** Layer just finished (1-based) — 0 before any layer fires. */
    layer:    number;
    /** Total layers in the tower. */
    nLayers:  number;
}

interface Props {
    state: VisionProgressState;
}

/**
 * Sticky strip rendered above the chat input row while a vision encode
 * is in flight. Replaces the easy-to-miss `statusLine` text (≈10 px) for
 * the long encode operations that benefit most from clear feedback.
 */
export function VisionProgress({ state }: Props) {
    const pct = state.nLayers > 0 ? Math.min(100, (state.layer / state.nLayers) * 100) : 0;
    const label = state.nImages > 1
        ? `Analyzing image ${state.imageIdx}/${state.nImages}`
        : "Analyzing image";

    return (
        <div className="border-t border-border bg-background/80 px-3 py-1.5 sm:px-4">
            <div className="flex items-center gap-3 text-xs">
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-primary" />
                <span className="font-medium">{label}</span>
                <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                    <div
                        className="h-full bg-primary transition-all"
                        style={{ width: `${pct}%` }}
                    />
                </div>
                <span className="font-mono tabular-nums text-muted-foreground">
                    {state.layer}/{state.nLayers}
                </span>
            </div>
        </div>
    );
}
