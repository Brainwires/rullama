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

// Tool calling defaults ON for desktops, OFF for phones/tablets: small/cloud
// models on constrained devices fare better without the always-injected tool
// schema, while desktop users get tools (+ RAG) out of the box. Only the
// first-run default — an explicit toggle persists and wins thereafter.
const TOOLS_DEFAULT_ON =
    typeof navigator !== "undefined" &&
    !/iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

export function useToolSettings() {
    // Keys keep the original (double-namespaced) form so existing
    // persisted values survive the extraction — usePersistedState
    // prefixes `rullama:` again on top of these.
    const [toolMode, setToolMode]           = usePersistedState<boolean>("rullama:toolMode", TOOLS_DEFAULT_ON);
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
