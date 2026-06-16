// Weather tool executor.
//
// Two backends, picked at call time:
//   • WeatherAPI.com  — used when the user provides an API key (Tools tab).
//     Mirrors the brainwires-studio weather MCP server (lib/mcp/servers/
//     weather-server) — same provider, same /current.json endpoint.
//   • Open-Meteo      — free, keyless, CORS-enabled fallback so weather still
//     works with NO key configured. Geocodes a place name → lat/lon, then
//     fetches current conditions.
//
// Both are called directly from the browser (both send permissive CORS
// headers), so no server proxy is needed — consistent with rullama's
// "your data never leaves the device" stance (only the place name / coords
// leave, and only to the weather provider the user opted into).

export type Units = "metric" | "imperial";

export interface WeatherArgs {
    location?: string;
    units?: Units;
}

export interface WeatherResult {
    ok: boolean;
    /** Compact, model-friendly one-liner fed back into the prompt. */
    summary: string;
    /** Structured fields for the UI result chip. */
    data?: Record<string, unknown>;
}

/** Does this look like bare `lat,lon` coordinates (e.g. GPS)? */
function isCoords(s: string): boolean {
    return /^\s*-?\d{1,3}(\.\d+)?\s*,\s*-?\d{1,3}(\.\d+)?\s*$/.test(s);
}

// ─── WeatherAPI.com (keyed) ──────────────────────────────────────────────

async function viaWeatherApi(
    apiKey: string,
    location: string,
    units: Units,
): Promise<WeatherResult> {
    const url = new URL("https://api.weatherapi.com/v1/current.json");
    url.searchParams.set("key", apiKey);
    url.searchParams.set("q", location);
    url.searchParams.set("aqi", "no");

    const resp = await fetch(url.toString());
    if (!resp.ok) {
        const err = await resp.json().catch(() => null);
        const msg = err?.error?.message || `HTTP ${resp.status} ${resp.statusText}`;
        throw new Error(`WeatherAPI error: ${msg}`);
    }
    const d = await resp.json();
    const imperial = units === "imperial";
    const temp = imperial ? d.current.temp_f : d.current.temp_c;
    const feels = imperial ? d.current.feelslike_f : d.current.feelslike_c;
    const wind = imperial ? d.current.wind_mph : d.current.wind_kph;
    const tu = imperial ? "°F" : "°C";
    const wu = imperial ? "mph" : "km/h";
    const place = [d.location.name, d.location.region, d.location.country]
        .filter(Boolean)
        .join(", ");
    const data = {
        place,
        condition: d.current.condition.text as string,
        temperature: `${Math.round(temp)}${tu}`,
        feels_like: `${Math.round(feels)}${tu}`,
        humidity: `${d.current.humidity}%`,
        wind: `${Math.round(wind)} ${wu} ${d.current.wind_dir}`,
        local_time: d.location.localtime as string,
    };
    return {
        ok: true,
        summary:
            `Current weather in ${place}: ${data.condition}, ${data.temperature} ` +
            `(feels like ${data.feels_like}), humidity ${data.humidity}, ` +
            `wind ${data.wind}. Local time ${data.local_time}.`,
        data,
    };
}

// ─── Open-Meteo (keyless) ────────────────────────────────────────────────

// WMO weather-interpretation codes → human text. Open-Meteo returns the code;
// WeatherAPI returns text directly. Trimmed to the common buckets.
const WMO: Record<number, string> = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
};

async function geocode(
    place: string,
): Promise<{ lat: number; lon: number; label: string }> {
    const url = new URL("https://geocoding-api.open-meteo.com/v1/search");
    url.searchParams.set("name", place);
    url.searchParams.set("count", "1");
    const resp = await fetch(url.toString());
    if (!resp.ok) throw new Error(`Geocoding failed: HTTP ${resp.status}`);
    const d = await resp.json();
    const hit = d?.results?.[0];
    if (!hit) throw new Error(`Couldn't find a location named "${place}".`);
    const label = [hit.name, hit.admin1, hit.country].filter(Boolean).join(", ");
    return { lat: hit.latitude, lon: hit.longitude, label };
}

async function viaOpenMeteo(
    location: string,
    units: Units,
): Promise<WeatherResult> {
    let lat: number, lon: number, label: string;
    if (isCoords(location)) {
        const [a, b] = location.split(",").map((x) => parseFloat(x.trim()));
        lat = a;
        lon = b;
        label = "your location";
    } else {
        ({ lat, lon, label } = await geocode(location));
    }

    const imperial = units === "imperial";
    const url = new URL("https://api.open-meteo.com/v1/forecast");
    url.searchParams.set("latitude", String(lat));
    url.searchParams.set("longitude", String(lon));
    url.searchParams.set(
        "current",
        "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
    );
    url.searchParams.set("temperature_unit", imperial ? "fahrenheit" : "celsius");
    url.searchParams.set("wind_speed_unit", imperial ? "mph" : "kmh");
    const resp = await fetch(url.toString());
    if (!resp.ok) throw new Error(`Open-Meteo error: HTTP ${resp.status}`);
    const d = await resp.json();
    const c = d.current;
    const tu = imperial ? "°F" : "°C";
    const wu = imperial ? "mph" : "km/h";
    const condition = WMO[c.weather_code] ?? `Code ${c.weather_code}`;
    const data = {
        place: label,
        condition,
        temperature: `${Math.round(c.temperature_2m)}${tu}`,
        feels_like: `${Math.round(c.apparent_temperature)}${tu}`,
        humidity: `${c.relative_humidity_2m}%`,
        wind: `${Math.round(c.wind_speed_10m)} ${wu}`,
    };
    return {
        ok: true,
        summary:
            `Current weather in ${label}: ${condition}, ${data.temperature} ` +
            `(feels like ${data.feels_like}), humidity ${data.humidity}, ` +
            `wind ${data.wind}.`,
        data,
    };
}

// ─── Public entry ────────────────────────────────────────────────────────

export interface WeatherCtx {
    apiKey: string;
    units: Units;
    /** Resolved "lat,lon" from GPS, or null. Used when the model omits a
     *  location or asks for the "current"/"here"/"my location" weather. */
    geo: string | null;
}

/** Does the model's location argument mean "where I am right now"? */
function meansCurrentLocation(loc: string): boolean {
    return /\b(current|my|here|nearby|this)\b/i.test(loc) && /location|me|here|area/i.test(loc)
        || /^(here|nearby|current location|my location)$/i.test(loc.trim());
}

export async function executeWeather(
    args: WeatherArgs,
    ctx: WeatherCtx,
): Promise<WeatherResult> {
    const units = args.units ?? ctx.units;
    let location = (args.location ?? "").trim();

    // Resolve "current location" / empty → GPS coords when available.
    if (!location || meansCurrentLocation(location)) {
        if (ctx.geo) {
            location = ctx.geo;
        } else {
            return {
                ok: false,
                summary:
                    "No specific location was given and GPS isn't available. Ask " +
                    "the user which city they want the weather for.",
            };
        }
    }

    try {
        const r = ctx.apiKey
            ? await viaWeatherApi(ctx.apiKey, location, units)
            : await viaOpenMeteo(location, units);
        return r;
    } catch (e) {
        return {
            ok: false,
            summary: `Weather lookup failed: ${(e as Error).message}`,
        };
    }
}
