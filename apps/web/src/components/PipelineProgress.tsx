import { Loader2 } from "lucide-react";

/** Which step of the multimodal→generation pipeline is currently running.
 *
 * `encoding`   — vision/audio tower.
 * `embedding`  — splicing one clip's soft tokens through the text model.
 * `prefill`    — feeding the prompt tokens through the text model.
 * `generating` — emitting output tokens (mic transcribe).
 *
 * Without all phases plumbed into a single strip the user sees the
 * "Analyzing image" bar disappear ~30 s in and then sits ~3 min staring
 * at nothing while the JS prefill loop chews on 256 soft-token rows per
 * image at ~870 ms each. Same dynamic applies to mic transcription.
 */
export type PipelineProgressPhase = "encoding" | "embedding" | "prefill" | "generating";

/** Whether the strip is driven by an image or audio pipeline. Labels
 *  flip from "Analyzing image" to "Transcribing audio" so a user
 *  staring at the strip during a mic transcribe doesn't get
 *  image-flavoured phrasing. */
export type PipelineProgressKind = "image" | "audio";

export interface PipelineProgressState {
    /** 1-based: "image 2 of 3". Always 1 for audio. */
    imageIdx: number;
    /** Total clips queued in this turn. 1 hides the "X/Y" prefix. */
    nImages:  number;
    /** What stage of the pipeline is running right now. */
    phase:    PipelineProgressPhase;
    /** Units finished so far in this phase (1-based; 0 before any tick). */
    done:     number;
    /** Total units in this phase. */
    total:    number;
    /** Modality driving the strip. Defaults to "image" for call sites
     *  that pre-date the audio flow. */
    kind?:    PipelineProgressKind;
}

interface Props {
    state: PipelineProgressState;
}

function labelFor(state: PipelineProgressState): string {
    const isAudio = state.kind === "audio";
    const imgSuffix = !isAudio && state.nImages > 1 ? ` ${state.imageIdx}/${state.nImages}` : "";
    switch (state.phase) {
        case "encoding":   return isAudio ? "Transcribing — encoding audio" : `Analyzing image${imgSuffix}`;
        case "embedding":  return isAudio ? "Transcribing — splicing tokens" : `Embedding image${imgSuffix}`;
        case "prefill":    return isAudio ? "Transcribing — reading prompt" : "Reading prompt";
        case "generating": return isAudio ? "Transcribing — generating text" : "Generating";
    }
}

/**
 * Sticky strip rendered above the chat input row while a vision encode +
 * splice + prefill is in flight. Replaces the easy-to-miss `statusLine`
 * text (≈10 px) for the long operations that benefit most from clear
 * feedback.
 */
export function PipelineProgress({ state }: Props) {
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
