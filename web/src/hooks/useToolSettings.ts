import { usePersistedState } from "@/lib/persisted";
import { defaultUnitsFromLocale, type Units as ToolUnits } from "@/lib/tools";

/**
 * Persisted Tools-tab settings — grouped behind one hook.
 *
 *   - `toolMode`      — inject the tool schema into the system prompt so the
 *                       base model emits `<tool_call>` blocks (no adapter).
 *   - `weatherApiKey` / `weatherUnits` — the executable weather tool
 *                       (`lib/tools`).
 *
 *   Location (GPS) has no toggle: location tools resolve `navigator.geolocation`
 *   on demand when called without a place, and the OS permission prompt is the
 *   user's control.
 */
// Computed once at module load (navigator is available by then in the browser).
const LOCALE_DEFAULT_UNITS = defaultUnitsFromLocale();

export function useToolSettings() {
    // Keys keep the original (double-namespaced) form so existing
    // persisted values survive the extraction — usePersistedState
    // prefixes `rullama:` again on top of these.
    const [toolMode, setToolMode]           = usePersistedState<boolean>("rullama:toolMode", false);
    const [weatherApiKey, setWeatherApiKey] = usePersistedState<string>("rullama:weatherApiKey", "");
    const [newsApiKey, setNewsApiKey]       = usePersistedState<string>("rullama:newsApiKey", "");
    // Default the temperature scale from the user's OS/browser locale (°F for
    // the US & a few territories, °C elsewhere) instead of hard-coding metric.
    // Only used on first run — an explicit choice in the Tools tab persists and
    // wins thereafter.
    const [weatherUnits, setWeatherUnits]   = usePersistedState<ToolUnits>("rullama:weatherUnits", LOCALE_DEFAULT_UNITS);
    // Programmatic tool calling: the model writes ONE Rhai script orchestrating
    // many tools (Brainwires/tool-orchestrator), instead of the sequential JSON
    // loop. Experimental, opt-in; failures fall back to the JSON loop.
    const [orchestratorMode, setOrchestratorMode] = usePersistedState<boolean>("rullama:orchestratorMode", false);

    return {
        toolMode, setToolMode,
        weatherApiKey, setWeatherApiKey,
        newsApiKey, setNewsApiKey,
        weatherUnits, setWeatherUnits,
        orchestratorMode, setOrchestratorMode,
    };
}
