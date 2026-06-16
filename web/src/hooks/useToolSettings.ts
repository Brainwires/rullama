import { usePersistedState } from "@/lib/persisted";
import { defaultUnitsFromLocale, type Units as ToolUnits } from "@/lib/tools";

/**
 * Persisted Tools-tab settings — grouped behind one hook.
 *
 *   - `toolMode`      — inject the tool schema into the system prompt so the
 *                       base model emits `<tool_call>` blocks (no adapter).
 *   - `weatherApiKey` / `weatherUnits` — the executable weather tool
 *                       (`lib/tools`).
 *   - `useGps`        — let location-aware tools consult `navigator.geolocation`.
 */
// Computed once at module load (navigator is available by then in the browser).
const LOCALE_DEFAULT_UNITS = defaultUnitsFromLocale();

export function useToolSettings() {
    // Keys keep the original (double-namespaced) form so existing
    // persisted values survive the extraction — usePersistedState
    // prefixes `rullama:` again on top of these.
    const [toolMode, setToolMode]           = usePersistedState<boolean>("rullama:toolMode", false);
    const [weatherApiKey, setWeatherApiKey] = usePersistedState<string>("rullama:weatherApiKey", "");
    // Default the temperature scale from the user's OS/browser locale (°F for
    // the US & a few territories, °C elsewhere) instead of hard-coding metric.
    // Only used on first run — an explicit choice in the Tools tab persists and
    // wins thereafter.
    const [weatherUnits, setWeatherUnits]   = usePersistedState<ToolUnits>("rullama:weatherUnits", LOCALE_DEFAULT_UNITS);
    const [useGps, setUseGps]               = usePersistedState<boolean>("rullama:useGps", false);

    return {
        toolMode, setToolMode,
        weatherApiKey, setWeatherApiKey,
        weatherUnits, setWeatherUnits,
        useGps, setUseGps,
    };
}
