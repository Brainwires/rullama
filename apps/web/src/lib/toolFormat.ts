// Tool-call wire format — the single source of truth shared by the renderer
// (parseToolCalls.ts / ToolCallBlock.tsx) and, later, the function-call LoRA's
// synthetic-dataset generator. Keeping the delimiters in one place is what
// stops the renderer and the training data from drifting apart.
//
// Gemma 4 has NO native tool-call token (its chat template only defines
// `<|turn>` / `<turn|>` turns and the `<|channel>thought` / `<channel|>`
// reasoning channel — see the engine repo (rullama-engine/src/template/gemma4_small.rs), which
// explicitly defers tool-call rendering). So this is a chosen, prompt-level
// convention, not something baked into the model.
//
// We use the `<tool_call>…</tool_call>` tag pair — the de-facto community
// standard that the majority of open tool-calling fine-tunes already target,
// and which is distinct from every marker Gemma already emits. The payload
// inside is JSON: `{ "name": "...", "arguments": { ... } }`.

export const TOOL_CALL_OPEN = "<tool_call>";
export const TOOL_CALL_CLOSE = "</tool_call>";

// After a tool is EXECUTED (lib/tools), its result is spliced back into the
// model turn wrapped in these markers, then the model continues generating its
// natural-language answer. The renderer strips this span from the visible prose
// (it's plumbing) and attaches the inner text to the preceding call as
// `ToolCall.result` for the result chip. Not part of the LoRA training format —
// purely an inference-time round-trip convention.
export const TOOL_RESPONSE_OPEN = "<tool_response>";
export const TOOL_RESPONSE_CLOSE = "</tool_response>";

/** Build a tool-response block for splicing back into the model turn. The
 *  tool name is attribute-encoded so the renderer can match the result to the
 *  right call even when SEVERAL tools run in one turn (or a non-executable
 *  call is interspersed) — order alone isn't enough then. Tool names are
 *  `[a-z_]`, so there's nothing to escape. */
export function toolResponseBlock(name: string, summary: string): string {
    return `<tool_response for="${name}">${summary}${TOOL_RESPONSE_CLOSE}`;
}

// The parser matches the opening tag LENIENTLY by this prefix: small models
// sometimes drop the closing `>` of `<tool_call>` (a tokenization quirk —
// observed emitting `<tool_call\n{…`), so we key on `<tool_call` and skip an
// optional `>`. This never collides with the `</tool_call>` closer (that
// starts `</`, not `<t`). The canonical OPEN above is still what training
// data emits and what callers should produce.
export const TOOL_CALL_OPEN_PREFIX = "<tool_call";

// Ordered parameter names per tool. Used to map POSITIONAL pythonic-call
// arguments (e.g. `set_timer(7)` or `set_reminder("call grandma", "tonight")`)
// onto named keys, since small models given a schema emit BOTH clean JSON and
// pythonic call syntax (BFCL research confirms both formats are standard and a
// tolerant parser accepting both is the right call). Keep in sync with
// tool-schema.txt / TOOL_SCHEMA_PROMPT below.
export const TOOL_PARAMS: Record<string, string[]> = {
    set_timer: ["duration"],
    set_reminder: ["text", "time"],
    get_weather: ["location"],
    get_weather_forecast: ["location", "days"],
    get_air_quality: ["location"],
    get_astronomy: ["location"],
    search_wikipedia: ["query"],
    search_knowledge: ["query"],
    get_news: ["query"],
    calculate: ["problem"],
};

// The tool schema, injected as a system preamble so the model copies exact
// tool names + argument keys instead of memorizing them. MUST stay byte-
// identical to the engine repo (rullama-lora/examples/data/tool-schema.txt) (the LoRA
// is trained with that file as a System turn; inference must present the same
// text or the slot keys drift). When wiring the chat path, prepend this to the
// system message while the function-call adapter is active (mirrors the RAG
// preamble in App.tsx).
export const TOOL_SCHEMA_PROMPT = `You have access to these tools:
- set_timer(duration)
- set_reminder(text, time)
- get_weather(location)
- get_weather_forecast(location, days)
- get_air_quality(location)
- get_astronomy(location)
- search_wikipedia(query)
- search_knowledge(query)
- get_news(query)
- calculate(problem)

For weather: get_weather is current conditions; get_weather_forecast covers the next "days" days (1-10); get_air_quality is pollution/AQI; get_astronomy is sunrise, sunset, and moon. location is a city name, "lat,lon" coordinates, or omitted to mean the user's current location.
search_wikipedia looks up facts on Wikipedia. search_knowledge searches the user's OWN uploaded documents (the Knowledge tab) — use it when they ask about their notes/files. get_news fetches recent news headlines. set_timer and set_reminder take a relative time like "10 minutes". calculate handles any math problem (algebra, calculus, solving equations, unit conversion) — pass the problem in plain words and it returns the exact tools to finish it.

When the user's request matches one or more of these tools, reply with a tool call for each, in exactly this format:
<tool_call>{"name": "<tool_name>", "arguments": { ... }}</tool_call>
If several independent tools apply (e.g. "weather and air quality"), emit one <tool_call> block per tool, back to back. But when one step depends on another's result (e.g. "if the temperature is above 20°, show the air quality"), call ONLY the first tool now — you will be given its result and can then decide whether to call the next. Use the exact tool name and argument keys, and copy the user's own words into the values — do not convert units, do math, or round (e.g. "30 seconds" stays "30 seconds", not minutes).

If no tool fits the request, just answer the user normally. Not every message is a tool call — reason and respond as usual when none applies.`;

/** A single tool call extracted from a model reply. */
export interface ToolCall {
    /** Function name. Empty string if not yet parseable (mid-stream). */
    name: string;
    /** Parsed argument object, or the raw inner string if it isn't valid JSON
     *  (e.g. malformed output, or still streaming). */
    arguments: Record<string, unknown> | string;
    /** The raw inner text between the open/close markers, verbatim. */
    raw: string;
    /** True while the opening marker has been seen but the closing one hasn't
     *  (the call is still streaming in). */
    pending: boolean;
    /** The executed tool's result summary, once it has run and been spliced
     *  back in via the <tool_response> markers. Undefined for render-only
     *  calls (no executor) or before execution. */
    result?: string;
}
