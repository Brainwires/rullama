import { Loader2 } from "lucide-react";

/** Which step of the image→generation pipeline is currently running.
 *
 * `encoding`  — vision tower (1 .. n_layers).
 * `embedding` — splicing one image's soft tokens through the text model.
 * `prefill`   — feeding the prompt tokens through the text model.
 *
 * Without all three phases plumbed into a single strip the user sees the
 * "Analyzing image" bar disappear ~30 s in and then sits ~3 min staring
 * at nothing while the JS prefill loop chews on 256 soft-token rows per
 * image at ~870 ms each.
 */
export type VisionProgressPhase = "encoding" | "embedding" | "prefill";

export interface VisionProgressState {
    /** 1-based: "image 2 of 3". */
    imageIdx: number;
    /** Total images queued in this turn. 1 hides the "image X/Y" prefix. */
    nImages:  number;
    /** What stage of the pipeline is running right now. */
    phase:    VisionProgressPhase;
    /** Units finished so far in this phase (1-based; 0 before any tick). */
    done:     number;
    /** Total units in this phase (layers for encoding, soft tokens for
     *  embedding, prompt-token positions for prefill). */
    total:    number;
}

interface Props {
    state: VisionProgressState;
}

function labelFor(state: VisionProgressState): string {
    const imgSuffix = state.nImages > 1 ? ` ${state.imageIdx}/${state.nImages}` : "";
    switch (state.phase) {
        case "encoding":  return `Analyzing image${imgSuffix}`;
        case "embedding": return `Embedding image${imgSuffix}`;
        case "prefill":   return "Reading prompt";
    }
}

/**
 * Sticky strip rendered above the chat input row while a vision encode +
 * splice + prefill is in flight. Replaces the easy-to-miss `statusLine`
 * text (≈10 px) for the long operations that benefit most from clear
 * feedback.
 */
export function VisionProgress({ state }: Props) {
    const pct = state.total > 0 ? Math.min(100, (state.done / state.total) * 100) : 0;
    const label = labelFor(state);

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
                    {state.done}/{state.total}
                </span>
            </div>
        </div>
    );
}
