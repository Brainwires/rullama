import { Loader2 } from "lucide-react";

/** Phase markers the worker forwards from the wasm trainer via
 *  `notify("trainingProgress", …)`. Matches the strings emitted by
 *  `TrainingSession::step_with_progress` in
 *  `rullama-framework/engine/rullama-lora/src/session.rs`.
 *
 *  - `starting`   — synthetic, set by the UI between `trainingStart`
 *                   request and the first beacon back. Covers the
 *                   pipeline-compile + first-step warmup window
 *                   (~30 s on cold-cache browsers) so the user
 *                   doesn't stare at a frozen-looking progress bar.
 *  - `prefill`    — feeding prompt tokens (current / total).
 *  - `forward`    — final-position capture step finished
 *                   (current = total_layers as a single tick).
 *  - `backward`   — per-layer backward sweep (current = 1-based
 *                   logical layer index, total = n_layers).
 *  - `clip`       — global gradient clip done.
 *  - `optimizer`  — Adam step done. */
export type TrainingProgressPhase =
    | "starting"
    | "prefill"
    | "forward"
    | "backward"
    | "clip"
    | "optimizer";

export interface TrainingProgressState {
    phase:   TrainingProgressPhase;
    current: number;
    total:   number;
    /** 1-based optimizer step counter — bumped after `optimizer`. */
    step?:   number;
    /** Current learning rate (post-warmup, post-schedule). */
    lr?:     number;
}

function labelFor(state: TrainingProgressState, stepBudget: number | undefined): string {
    const stepSuffix = stepBudget && state.step
        ? ` (step ${state.step}/${stepBudget})`
        : state.step ? ` (step ${state.step})` : "";
    switch (state.phase) {
        case "starting":  return `Preparing training session${stepSuffix}`;
        case "prefill":   return `Reading prompt — token ${state.current}/${state.total}${stepSuffix}`;
        case "forward":   return `Forward pass complete — running backward${stepSuffix}`;
        case "backward":  return `Backward — layer ${state.current}/${state.total}${stepSuffix}`;
        case "clip":      return `Clipping gradients${stepSuffix}`;
        case "optimizer": return `Adam step${stepSuffix}`;
    }
}

interface Props {
    /** Last beacon received from the worker. */
    state:         TrainingProgressState | null;
    /** User-set total optimizer steps — shown in the label so the
     *  user can see "step 3/100" at a glance. */
    stepBudget?:   number;
    /** Optional extra hint shown next to the progress bar — usually
     *  "compiling WGSL shaders…" during the cold-start window when
     *  the `starting` phase has been live for >5 s. */
    coldHint?:     string;
}

/** PipelineProgress-shaped status strip rendered inside the live
 *  training panel. Same layout: spinner, label, progress bar,
 *  fraction. The trainer fires phase changes between encoder submits
 *  in `forward_chained.rs` so this bar advances smoothly through
 *  forward → backward sweeps instead of jumping per step. */
export function TrainingProgress({ state, stepBudget, coldHint }: Props) {
    if (!state) {
        return null;
    }
    const pct = state.total > 0
        ? Math.min(100, (state.current / state.total) * 100)
        : (state.phase === "starting" ? 0 : 100);
    const label = labelFor(state, stepBudget);

    return (
        <div className="rounded border border-border bg-muted/20 px-3 py-2 text-xs">
            <div className="flex items-center gap-3">
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-primary" />
                <span className="min-w-0 flex-1 truncate font-medium">{label}</span>
                <span className="font-mono tabular-nums text-muted-foreground">
                    {state.total > 0 ? `${state.current}/${state.total}` : ""}
                </span>
            </div>
            <div className="relative mt-1.5 h-1 overflow-hidden rounded-full bg-muted">
                <div
                    className="h-full bg-primary transition-all"
                    style={{ width: `${pct}%` }}
                />
            </div>
            {coldHint && (
                <div className="mt-1 text-[10px] text-muted-foreground">{coldHint}</div>
            )}
        </div>
    );
}
