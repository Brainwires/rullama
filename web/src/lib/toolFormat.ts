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
