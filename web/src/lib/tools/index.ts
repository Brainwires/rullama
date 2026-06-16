// Tool registry — the bridge between a model-emitted <tool_call> and code that
// actually does something. Today: weather (WeatherAPI.com / Open-Meteo). The
// renderer (parseToolCalls + ToolCallBlock) stays visual-only; THIS is where a
// call gets executed and its result fed back to the model for a final answer.

import { executeWeather, type Units, type WeatherCtx } from "@/lib/tools/weather";

export type { Units };

/** Per-turn settings the executor needs, sourced from the Tools settings tab. */
export interface ToolSettings {
    weatherApiKey: string;
    units: Units;
    useGps: boolean;
}

/** A normalized tool outcome. `summary` is what we feed back to the model. */
export interface ToolRunResult {
    ok: boolean;
    summary: string;
    data?: Record<string, unknown>;
}

// Tools we can actually run. Aliases included because small models emit either
// the schema name (`get_weather`) or the MCP-server name (`get_current_weather`).
const WEATHER_NAMES = new Set([
    "get_weather",
    "get_current_weather",
    "weather",
]);

/** Is there an executor wired up for this tool name? */
export function isExecutableTool(name: string): boolean {
    return WEATHER_NAMES.has(name.trim().toLowerCase());
}

/** Does this tool consult GPS (so we should resolve coords before running)? */
export function toolUsesLocation(name: string): boolean {
    return WEATHER_NAMES.has(name.trim().toLowerCase());
}

/**
 * One-shot browser geolocation, with a short-lived in-memory cache so we don't
 * re-prompt the user on every weather call. Resolves to "lat,lon" or null
 * (no API, denied, or timed out — the caller falls back to asking the user).
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

/**
 * Execute a tool call and return a model-friendly result. `geo` is the
 * already-resolved "lat,lon" (or null) — the caller resolves it on the user's
 * send gesture so the permission prompt happens in a user-activation context.
 */
export async function executeTool(
    name: string,
    args: Record<string, unknown>,
    settings: ToolSettings,
    geo: string | null,
): Promise<ToolRunResult> {
    const lower = name.trim().toLowerCase();
    if (WEATHER_NAMES.has(lower)) {
        const ctx: WeatherCtx = {
            apiKey: settings.weatherApiKey.trim(),
            units: settings.units,
            geo,
        };
        return executeWeather(
            {
                location: typeof args.location === "string" ? args.location : undefined,
                units: args.units === "imperial" || args.units === "metric" ? args.units : undefined,
            },
            ctx,
        );
    }
    return {
        ok: false,
        summary: `No executor is wired up for "${name}". Answer the user directly instead.`,
    };
}
