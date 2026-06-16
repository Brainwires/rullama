import { usePersistedState } from "@/lib/persisted";
import { type Units as ToolUnits } from "@/lib/tools";

/**
 * Persisted Tools-tab settings — grouped behind one hook.
 *
 *   - `toolMode`      — inject the tool schema into the system prompt so the
 *                       base model emits `<tool_call>` blocks (no adapter).
 *   - `weatherApiKey` / `weatherUnits` — the executable weather tool
 *                       (`lib/tools`).
 *   - `useGps`        — let location-aware tools consult `navigator.geolocation`.
 */
export function useToolSettings() {
    // Keys keep the original (double-namespaced) form so existing
    // persisted values survive the extraction — usePersistedState
    // prefixes `rullama:` again on top of these.
    const [toolMode, setToolMode]           = usePersistedState<boolean>("rullama:toolMode", false);
    const [weatherApiKey, setWeatherApiKey] = usePersistedState<string>("rullama:weatherApiKey", "");
    const [weatherUnits, setWeatherUnits]   = usePersistedState<ToolUnits>("rullama:weatherUnits", "metric");
    const [useGps, setUseGps]               = usePersistedState<boolean>("rullama:useGps", false);

    return {
        toolMode, setToolMode,
        weatherApiKey, setWeatherApiKey,
        weatherUnits, setWeatherUnits,
        useGps, setUseGps,
    };
}
