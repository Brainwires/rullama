// Weather tool executors.
//
// Two backends, picked at call time:
//   • WeatherAPI.com  — used when the user provides an API key (Tools tab).
//     Mirrors the brainwires-studio weather MCP server (lib/mcp/servers/
//     weather-server) — same provider, same endpoints.
//   • Open-Meteo      — free, keyless, CORS-enabled fallback so weather still
//     works with NO key configured. Geocodes a place name → lat/lon, then
//     fetches the requested product.
//
// Both are called directly from the browser (both send permissive CORS
// headers), so no server proxy is needed — consistent with rullama's
// "your data never leaves the device" stance (only the place name / coords
// leave, and only to the weather provider the user opted into).
//
// Products: current conditions, multi-day forecast, air quality, astronomy
// (sunrise/sunset/moon). The registry (lib/tools/index.ts) maps each tool
// name onto one of these `WeatherKind`s.

export type Units = "metric" | "imperial";
export type WeatherKind = "current" | "forecast" | "air_quality" | "astronomy";

export interface WeatherArgs {
    location?: string;
    units?: Units;
    /** Forecast horizon in days (1–10). Ignored by non-forecast kinds. */
    days?: number;
}

export interface WeatherResult {
    ok: boolean;
    /** Compact, model-friendly summary fed back into the prompt. */
    summary: string;
    /** Structured fields for the UI result chip. */
    data?: Record<string, unknown>;
}

export interface WeatherCtx {
    apiKey: string;
    units: Units;
    /** Resolved "lat,lon" from GPS, or null. Used when the model omits a
     *  location or asks for the "current"/"here"/"my location" weather. */
    geo: string | null;
}

/** Does this look like bare `lat,lon` coordinates (e.g. GPS)? */
function isCoords(s: string): boolean {
    return /^\s*-?\d{1,3}(\.\d+)?\s*,\s*-?\d{1,3}(\.\d+)?\s*$/.test(s);
}

/** Detect the user's preferred temperature scale from their OS/browser
 *  locale region. Fahrenheit is used by the US (+ a few territories);
 *  everyone else gets Celsius. Only the region explicitly present in the
 *  locale is trusted (so a bare "en" doesn't wrongly maximize to US). */
const FAHRENHEIT_REGIONS = new Set(["US", "BS", "BZ", "KY", "PW", "FM", "MH", "LR"]);
export function defaultUnitsFromLocale(): Units {
    try {
        if (typeof navigator === "undefined") return "metric";
        const langs = navigator.languages?.length ? navigator.languages : [navigator.language];
        for (const lang of langs) {
            if (!lang) continue;
            const region = new Intl.Locale(lang).region;
            if (region) return FAHRENHEIT_REGIONS.has(region) ? "imperial" : "metric";
        }
    } catch { /* Intl.Locale unsupported / weird locale — fall through */ }
    return "metric";
}

// ─── lat/lon resolution (shared by the keyless Open-Meteo backends) ──────

// US state abbreviation → full name, for disambiguating "City, ST" queries
// (the model emits "New Glarus, WI", "Miami, FL", …). Open-Meteo's geocoder
// matches on city only and returns the full state name in `admin1`.
const US_STATES: Record<string, string> = {
    al: "alabama", ak: "alaska", az: "arizona", ar: "arkansas", ca: "california",
    co: "colorado", ct: "connecticut", de: "delaware", fl: "florida", ga: "georgia",
    hi: "hawaii", id: "idaho", il: "illinois", in: "indiana", ia: "iowa",
    ks: "kansas", ky: "kentucky", la: "louisiana", me: "maine", md: "maryland",
    ma: "massachusetts", mi: "michigan", mn: "minnesota", ms: "mississippi", mo: "missouri",
    mt: "montana", ne: "nebraska", nv: "nevada", nh: "new hampshire", nj: "new jersey",
    nm: "new mexico", ny: "new york", nc: "north carolina", nd: "north dakota", oh: "ohio",
    ok: "oklahoma", or: "oregon", pa: "pennsylvania", ri: "rhode island", sc: "south carolina",
    sd: "south dakota", tn: "tennessee", tx: "texas", ut: "utah", vt: "vermont",
    va: "virginia", wa: "washington", wv: "west virginia", wi: "wisconsin", wy: "wyoming",
    dc: "district of columbia",
};

// Open-Meteo's geocoder matches ONLY a bare place name — "Brodhead, Wisconsin"
// returns nothing. So we search by the first comma-part (the city) and use the
// remaining parts (state/country) to disambiguate among matches. WeatherAPI's
// `q` is flexible and doesn't need this; this is the keyless path only.
async function geocode(place: string): Promise<{ lat: number; lon: number; label: string }> {
    const parts = place.split(",").map((s) => s.trim()).filter(Boolean);
    const city = parts[0] || place;
    const hints = parts.slice(1).map((s) => s.toLowerCase());

    const url = new URL("https://geocoding-api.open-meteo.com/v1/search");
    url.searchParams.set("name", city);
    url.searchParams.set("count", "10");
    url.searchParams.set("language", "en");
    const resp = await fetch(url.toString());
    if (!resp.ok) throw new Error(`Geocoding failed: HTTP ${resp.status}`);
    const d = await resp.json();
    const results: any[] = d?.results ?? [];
    if (!results.length) throw new Error(`Couldn't find a location named "${place}".`);

    // Prefer a result whose state/country matches a hint (expanding US state
    // abbreviations); fall back to the most prominent match (results[0]).
    let hit = results[0];
    if (hints.length) {
        const wanted = hints.flatMap((h) => [h, US_STATES[h]].filter(Boolean));
        const match = results.find((r) => {
            const hay = [r.admin1, r.admin2, r.country].filter(Boolean).join(" ").toLowerCase();
            return wanted.some((w) => hay.includes(w));
        });
        if (match) hit = match;
    }
    const label = [hit.name, hit.admin1, hit.country].filter(Boolean).join(", ");
    return { lat: hit.latitude, lon: hit.longitude, label };
}

async function resolveLatLon(location: string): Promise<{ lat: number; lon: number; label: string }> {
    if (isCoords(location)) {
        const [lat, lon] = location.split(",").map((x) => parseFloat(x.trim()));
        return { lat, lon, label: "your location" };
    }
    return geocode(location);
}

// WMO weather-interpretation codes → text (Open-Meteo returns the code;
// WeatherAPI returns text directly).
const WMO: Record<number, string> = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
};

const wmo = (code: number) => WMO[code] ?? `Code ${code}`;

/** US EPA index (WeatherAPI, 1–6) → category. */
function epaCategory(idx: number): string {
    return ["", "Good", "Moderate", "Unhealthy for sensitive groups", "Unhealthy", "Very unhealthy", "Hazardous"][idx] ?? "Unknown";
}
/** US AQI value (Open-Meteo, 0–500) → category. */
function usAqiCategory(aqi: number): string {
    if (aqi <= 50) return "Good";
    if (aqi <= 100) return "Moderate";
    if (aqi <= 150) return "Unhealthy for sensitive groups";
    if (aqi <= 200) return "Unhealthy";
    if (aqi <= 300) return "Very unhealthy";
    return "Hazardous";
}

// ─── WeatherAPI.com (keyed) ──────────────────────────────────────────────

async function weatherApiGet(apiKey: string, endpoint: string, params: Record<string, string>): Promise<any> {
    const url = new URL(`https://api.weatherapi.com/v1${endpoint}`);
    url.searchParams.set("key", apiKey);
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
    const resp = await fetch(url.toString());
    if (!resp.ok) {
        const err = await resp.json().catch(() => null);
        throw new Error(`WeatherAPI error: ${err?.error?.message || `HTTP ${resp.status}`}`);
    }
    return resp.json();
}

function placeOf(loc: any): string {
    return [loc.name, loc.region, loc.country].filter(Boolean).join(", ");
}

async function apiCurrent(apiKey: string, location: string, units: Units): Promise<WeatherResult> {
    const d = await weatherApiGet(apiKey, "/current.json", { q: location, aqi: "no" });
    const imp = units === "imperial";
    const tu = imp ? "°F" : "°C", wu = imp ? "mph" : "km/h";
    const place = placeOf(d.location);
    const data = {
        place,
        condition: d.current.condition.text,
        temperature: `${Math.round(imp ? d.current.temp_f : d.current.temp_c)}${tu}`,
        feels_like: `${Math.round(imp ? d.current.feelslike_f : d.current.feelslike_c)}${tu}`,
        humidity: `${d.current.humidity}%`,
        wind: `${Math.round(imp ? d.current.wind_mph : d.current.wind_kph)} ${wu} ${d.current.wind_dir}`,
        local_time: d.location.localtime,
    };
    return {
        ok: true,
        summary: `Current weather in ${place}: ${data.condition}, ${data.temperature} ` +
            `(feels like ${data.feels_like}), humidity ${data.humidity}, wind ${data.wind}. ` +
            `Local time ${data.local_time}.`,
        data,
    };
}

async function apiForecast(apiKey: string, location: string, units: Units, days: number): Promise<WeatherResult> {
    const d = await weatherApiGet(apiKey, "/forecast.json", { q: location, days: String(days), aqi: "no", alerts: "no" });
    const imp = units === "imperial";
    const tu = imp ? "°F" : "°C";
    const place = placeOf(d.location);
    const lines = d.forecast.forecastday.map((day: any) => {
        const hi = Math.round(imp ? day.day.maxtemp_f : day.day.maxtemp_c);
        const lo = Math.round(imp ? day.day.mintemp_f : day.day.mintemp_c);
        return `${day.date}: ${day.day.condition.text}, high ${hi}${tu} / low ${lo}${tu}, ` +
            `${day.day.daily_chance_of_rain}% chance of rain`;
    });
    return {
        ok: true,
        summary: `${days}-day forecast for ${place}:\n${lines.join("\n")}`,
        data: { place, days: d.forecast.forecastday.map((x: any) => x.date) },
    };
}

async function apiAirQuality(apiKey: string, location: string): Promise<WeatherResult> {
    const d = await weatherApiGet(apiKey, "/current.json", { q: location, aqi: "yes" });
    const aq = d.current.air_quality;
    if (!aq) throw new Error("Air quality data not available for this location.");
    const place = placeOf(d.location);
    const category = epaCategory(aq["us-epa-index"]);
    return {
        ok: true,
        summary: `Air quality in ${place}: ${category} (US EPA index ${aq["us-epa-index"]}). ` +
            `PM2.5 ${aq.pm2_5?.toFixed(1)} µg/m³, PM10 ${aq.pm10?.toFixed(1)} µg/m³, ozone ${aq.o3?.toFixed(1)} µg/m³.`,
        data: { place, category },
    };
}

async function apiAstronomy(apiKey: string, location: string): Promise<WeatherResult> {
    const d = await weatherApiGet(apiKey, "/astronomy.json", { q: location });
    const a = d.astronomy.astro;
    const place = placeOf(d.location);
    return {
        ok: true,
        summary: `Astronomy for ${place}: sunrise ${a.sunrise}, sunset ${a.sunset}, ` +
            `moonrise ${a.moonrise}, moonset ${a.moonset}, moon phase ${a.moon_phase} ` +
            `(${a.moon_illumination}% illuminated).`,
        data: { place, sunrise: a.sunrise, sunset: a.sunset, moon_phase: a.moon_phase },
    };
}

// ─── Open-Meteo (keyless) ────────────────────────────────────────────────

async function omGet(host: string, params: Record<string, string>): Promise<any> {
    const url = new URL(host);
    for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v);
    const resp = await fetch(url.toString());
    if (!resp.ok) throw new Error(`Open-Meteo error: HTTP ${resp.status}`);
    return resp.json();
}

async function omCurrent(location: string, units: Units): Promise<WeatherResult> {
    const { lat, lon, label } = await resolveLatLon(location);
    const imp = units === "imperial";
    const d = await omGet("https://api.open-meteo.com/v1/forecast", {
        latitude: String(lat), longitude: String(lon),
        current: "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
        temperature_unit: imp ? "fahrenheit" : "celsius",
        wind_speed_unit: imp ? "mph" : "kmh",
    });
    const c = d.current, tu = imp ? "°F" : "°C", wu = imp ? "mph" : "km/h";
    const condition = wmo(c.weather_code);
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
        summary: `Current weather in ${label}: ${condition}, ${data.temperature} ` +
            `(feels like ${data.feels_like}), humidity ${data.humidity}, wind ${data.wind}.`,
        data,
    };
}

async function omForecast(location: string, units: Units, days: number): Promise<WeatherResult> {
    const { lat, lon, label } = await resolveLatLon(location);
    const imp = units === "imperial";
    const tu = imp ? "°F" : "°C";
    const d = await omGet("https://api.open-meteo.com/v1/forecast", {
        latitude: String(lat), longitude: String(lon),
        daily: "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        forecast_days: String(days),
        temperature_unit: imp ? "fahrenheit" : "celsius",
        timezone: "auto",
    });
    const dy = d.daily;
    const lines = dy.time.map((date: string, i: number) =>
        `${date}: ${wmo(dy.weather_code[i])}, high ${Math.round(dy.temperature_2m_max[i])}${tu} / ` +
        `low ${Math.round(dy.temperature_2m_min[i])}${tu}, ${dy.precipitation_probability_max[i] ?? 0}% chance of rain`);
    return {
        ok: true,
        summary: `${days}-day forecast for ${label}:\n${lines.join("\n")}`,
        data: { place: label, days: dy.time },
    };
}

async function omAirQuality(location: string): Promise<WeatherResult> {
    const { lat, lon, label } = await resolveLatLon(location);
    const d = await omGet("https://air-quality-api.open-meteo.com/v1/air-quality", {
        latitude: String(lat), longitude: String(lon),
        current: "us_aqi,pm2_5,pm10,ozone",
    });
    const c = d.current;
    const category = usAqiCategory(c.us_aqi);
    return {
        ok: true,
        summary: `Air quality in ${label}: ${category} (US AQI ${c.us_aqi}). ` +
            `PM2.5 ${c.pm2_5} µg/m³, PM10 ${c.pm10} µg/m³, ozone ${c.ozone} µg/m³.`,
        data: { place: label, category },
    };
}

async function omAstronomy(location: string): Promise<WeatherResult> {
    const { lat, lon, label } = await resolveLatLon(location);
    const d = await omGet("https://api.open-meteo.com/v1/forecast", {
        latitude: String(lat), longitude: String(lon),
        daily: "sunrise,sunset", forecast_days: "1", timezone: "auto",
    });
    // Open-Meteo returns ISO timestamps; show just the clock part.
    const clock = (iso: string) => (iso?.includes("T") ? iso.split("T")[1] : iso);
    const sunrise = clock(d.daily.sunrise[0]);
    const sunset = clock(d.daily.sunset[0]);
    return {
        ok: true,
        summary: `Astronomy for ${label}: sunrise ${sunrise}, sunset ${sunset}. ` +
            `(Moon-phase data needs a WeatherAPI.com key.)`,
        data: { place: label, sunrise, sunset },
    };
}

// ─── Public entry ────────────────────────────────────────────────────────

/** Does the model's location argument mean "where I am right now"? */
function meansCurrentLocation(loc: string): boolean {
    return /^(here|nearby|current location|my location)$/i.test(loc.trim())
        || (/\b(current|my|here|nearby|this)\b/i.test(loc) && /\b(location|area)\b/i.test(loc));
}

export async function executeWeather(
    kind: WeatherKind,
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
                summary: "No specific location was given and GPS isn't available. Ask " +
                    "the user which city they want the weather for.",
            };
        }
    }

    const days = Math.max(1, Math.min(10, Math.round(args.days ?? 3)));
    const keyed = ctx.apiKey.length > 0;

    try {
        switch (kind) {
            case "forecast":
                return keyed ? await apiForecast(ctx.apiKey, location, units, days)
                             : await omForecast(location, units, days);
            case "air_quality":
                return keyed ? await apiAirQuality(ctx.apiKey, location)
                             : await omAirQuality(location);
            case "astronomy":
                return keyed ? await apiAstronomy(ctx.apiKey, location)
                             : await omAstronomy(location);
            case "current":
            default:
                return keyed ? await apiCurrent(ctx.apiKey, location, units)
                             : await omCurrent(location, units);
        }
    } catch (e) {
        return { ok: false, summary: `Weather lookup failed: ${(e as Error).message}` };
    }
}
