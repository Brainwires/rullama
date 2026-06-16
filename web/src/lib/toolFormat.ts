// Tool-call wire format — the single source of truth shared by the renderer
// (parseToolCalls.ts / ToolCallBlock.tsx) and, later, the function-call LoRA's
// synthetic-dataset generator. Keeping the delimiters in one place is what
// stops the renderer and the training data from drifting apart.
//
// Gemma 4 has NO native tool-call token (its chat template only defines
// `<|turn>` / `<turn|>` turns and the `<|channel>thought` / `<channel|>`
// reasoning channel — see crates/rullama/src/template/gemma4_small.rs, which
// explicitly defers tool-call rendering). So this is a chosen, prompt-level
// convention, not something baked into the model.
//
// We use the `<tool_call>…</tool_call>` tag pair — the de-facto community
// standard that the majority of open tool-calling fine-tunes already target,
// and which is distinct from every marker Gemma already emits. The payload
// inside is JSON: `{ "name": "...", "arguments": { ... } }`.

export const TOOL_CALL_OPEN = "<tool_call>";
export const TOOL_CALL_CLOSE = "</tool_call>";

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
    get_weather: ["location"],
    send_email: ["to", "subject"],
    add_calendar_event: ["title", "date"],
    play_music: ["query"],
    set_reminder: ["text", "time"],
};

// The tool schema, injected as a system preamble so the model copies exact
// tool names + argument keys instead of memorizing them. MUST stay byte-
// identical to crates/rullama-finetune/examples/data/tool-schema.txt (the LoRA
// is trained with that file as a System turn; inference must present the same
// text or the slot keys drift). When wiring the chat path, prepend this to the
// system message while the function-call adapter is active (mirrors the RAG
// preamble in App.tsx).
export const TOOL_SCHEMA_PROMPT = `You have access to these tools:
- set_timer(duration)
- get_weather(location)
- send_email(to, subject)
- add_calendar_event(title, date)
- play_music(query)
- set_reminder(text, time)

When the user's request clearly matches one of these tools, reply with a single tool call in exactly this format:
<tool_call>{"name": "<tool_name>", "arguments": { ... }}</tool_call>
Use the exact tool name and argument keys, and copy the user's own words into the values — do not convert units, do math, or round (e.g. "30 seconds" stays "30 seconds", not minutes). Don't overthink the tool call.

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
}
