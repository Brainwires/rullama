// Tool registry — the bridge between a model-emitted <tool_call> and code that
// actually does something. Generalized over `ToolDef`s (see ./types): each tool
// declares its names + a run(); this module builds a name→def map and exposes
// the three hooks useChatEngine's agentic loop needs. The same registry will
// back the future tool-orchestrator (it registers these same defs as Rhai
// functions). The renderer (parseToolCalls + ToolCallBlock) stays visual-only.

import { weatherTool, defaultUnitsFromLocale } from "@/lib/tools/weather";
import { wikipediaTool } from "@/lib/tools/wikipedia";
import { timerTool, reminderTool } from "@/lib/tools/reminders";
import { newsTool } from "@/lib/tools/news";
import { knowledgeTool } from "@/lib/tools/knowledge";
import { calculateTool, solveTool, analyzeTool, unitsTool } from "@/lib/tools/compute";
import type { ToolDef, ToolContext, ToolRunResult, Units } from "@/lib/tools/types";

export type { ToolDef, ToolContext, ToolRunResult, Units };
export { defaultUnitsFromLocale };

// Every executable tool. Add a ToolDef here to register it.
// `calculate` is the math GATEWAY (top-level); solve/analyze/units are the
// unlocked verbs — executable here but NOT listed in TOOL_SCHEMA_PROMPT (the
// model learns them from the gateway's tool_response). See ./compute.
const TOOLS: ToolDef[] = [
    weatherTool,
    wikipediaTool,
    timerTool,
    reminderTool,
    newsTool,
    knowledgeTool,
    calculateTool,
    solveTool,
    analyzeTool,
    unitsTool,
];

const BY_NAME = new Map<string, ToolDef>();
for (const t of TOOLS) for (const n of t.names) BY_NAME.set(n.toLowerCase(), t);

function toolFor(name: string): ToolDef | undefined {
    return BY_NAME.get(name.trim().toLowerCase());
}

/** Is there an executor wired up for this tool name? */
export function isExecutableTool(name: string): boolean {
    return toolFor(name) !== undefined;
}

/** Does this tool consult GPS (so we should resolve coords before running)? */
export function toolUsesLocation(name: string): boolean {
    return toolFor(name)?.usesLocation ?? false;
}

/** Execute a tool call and return a model-friendly result. */
export async function executeTool(
    name: string,
    args: Record<string, unknown>,
    ctx: ToolContext,
): Promise<ToolRunResult> {
    const t = toolFor(name);
    if (!t) {
        return {
            ok: false,
            summary: `No executor is wired up for "${name}". Answer the user directly instead.`,
        };
    }
    return t.run(name, args, ctx);
}

/**
 * One-shot browser geolocation, with a short-lived in-memory cache so we don't
 * re-prompt the user on every location-aware call. Resolves to "lat,lon" or
 * null (no API, denied, or timed out — the caller falls back to asking).
 */
let geoCache: { coords: string; at: number } | null = null;
const GEO_TTL_MS = 5 * 60_000;

export async function resolveGeo(timeoutMs = 8000): Promise<string | null> {
    if (typeof navigator === "undefined" || !("geolocation" in navigator)) return null;
    if (geoCache && Date.now() - geoCache.at < GEO_TTL_MS) return geoCache.coords;
    return new Promise((resolve) => {
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const coords = `${pos.coords.latitude.toFixed(5)},${pos.coords.longitude.toFixed(5)}`;
                geoCache = { coords, at: Date.now() };
                resolve(coords);
            },
            () => resolve(null),
            { enableHighAccuracy: false, timeout: timeoutMs, maximumAge: GEO_TTL_MS },
        );
    });
}
