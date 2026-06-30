// Shared tool types. Kept in their own module so every tool ToolDef and the
// registry import from here without a cycle (weather.ts ↔ index.ts would
// otherwise import each other).

import type { Units } from "@/lib/tools/weather";

export type { Units };

/** Everything a tool's `run()` might need, assembled per-turn by the caller
 *  (useChatEngine) from the Tools settings + the active conversation + GPS. */
export interface ToolContext {
    weatherApiKey: string;
    newsApiKey: string;
    units: Units;
    /** Resolved "lat,lon" (GPS) or null — the caller resolves it on demand when
     *  a location tool is called without a place; the OS permission prompt is
     *  the user's only control (no separate enable toggle). */
    geo: string | null;
    /** Active conversation id, for `search_knowledge` scoping. */
    conversationId: string | null;
}

/** A normalized tool outcome. `summary` is the model-facing text fed back via
 *  the `<tool_response>` block; `data` is optional structured detail. */
export interface ToolRunResult {
    ok: boolean;
    summary: string;
    data?: Record<string, unknown>;
}

/** One executable tool. Registered in `lib/tools/index.ts`. */
export interface ToolDef {
    /** Canonical name + aliases (small models vary); matched case-insensitively. */
    names: string[];
    /** True if the tool consults GPS, so the caller pre-resolves coords. */
    usesLocation?: boolean;
    /** Execute. `name` is the alias the model actually called (lets one ToolDef
     *  back several names — e.g. the weather family). */
    run(name: string, args: Record<string, unknown>, ctx: ToolContext): Promise<ToolRunResult>;
}
