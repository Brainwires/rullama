// Tool registry — the bridge between a model-emitted <tool_call> and code that
// actually does something. Today: weather (WeatherAPI.com / Open-Meteo) across
// four products — current, forecast, air quality, astronomy. The renderer
// (parseToolCalls + ToolCallBlock) stays visual-only; THIS is where a call gets
// executed and its result fed back to the model for a final answer.

import {
    executeWeather,
    defaultUnitsFromLocale,
    type Units,
    type WeatherKind,
    type WeatherCtx,
} from "@/lib/tools/weather";

export type { Units };
export { defaultUnitsFromLocale };

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

// Tool name → weather product. Aliases included because small models emit
// either the schema name (`get_weather_forecast`) or a shorter/MCP-style
// variant (`forecast`, `get_current_weather`).
const WEATHER_KINDS: Record<string, WeatherKind> = {
    // current conditions
    get_weather: "current",
    get_current_weather: "current",
    weather: "current",
    current_weather: "current",
    // multi-day forecast
    get_weather_forecast: "forecast",
    get_forecast: "forecast",
    forecast: "forecast",
    // air quality
    get_air_quality: "air_quality",
    air_quality: "air_quality",
    get_aqi: "air_quality",
    // astronomy (sunrise/sunset/moon)
    get_astronomy: "astronomy",
    astronomy: "astronomy",
    get_sun_times: "astronomy",
};

function weatherKind(name: string): WeatherKind | undefined {
    return WEATHER_KINDS[name.trim().toLowerCase()];
}

/** Is there an executor wired up for this tool name? */
export function isExecutableTool(name: string): boolean {
    return weatherKind(name) !== undefined;
}

/** Does this tool consult GPS (so we should resolve coords before running)? */
export function toolUsesLocation(name: string): boolean {
    return weatherKind(name) !== undefined;
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
    const kind = weatherKind(name);
    if (kind) {
        const ctx: WeatherCtx = {
            apiKey: settings.weatherApiKey.trim(),
            units: settings.units,
            geo,
        };
        const daysRaw = args.days;
        const days = typeof daysRaw === "number" ? daysRaw
            : typeof daysRaw === "string" && daysRaw.trim() !== "" && !Number.isNaN(Number(daysRaw))
                ? Number(daysRaw) : undefined;
        return executeWeather(
            kind,
            {
                location: typeof args.location === "string" ? args.location : undefined,
                units: args.units === "imperial" || args.units === "metric" ? args.units : undefined,
                days,
            },
            ctx,
        );
    }
    return {
        ok: false,
        summary: `No executor is wired up for "${name}". Answer the user directly instead.`,
    };
}
