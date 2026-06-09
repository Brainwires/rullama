// Split a model reply into tool-call blocks + the surrounding prose.
//
// Mirrors lib/parseModel.ts: it scans for the `<tool_call>` / `</tool_call>`
// markers (defined in lib/toolFormat.ts), pulls each call out, and leaves the
// rest as prose for the normal markdown pipeline. Streaming-safe — an opened
// but not-yet-closed call surfaces as `pending` so the UI can show a pulse.
//
// A reply with no `<tool_call>` marker passes straight through (calls === []).

import { TOOL_CALL_OPEN, TOOL_CALL_CLOSE, type ToolCall } from "@/lib/toolFormat";

export interface ParsedToolCalls {
    calls: ToolCall[];
    /** The reply text with all tool-call spans removed. */
    prose: string;
    /** True if the last call is still streaming (open seen, no close yet). */
    pending: boolean;
}

/** Best-effort name sniff from a partial JSON body, so a still-streaming call
 *  can show "Calling <name>…" before its JSON is complete. */
function sniffName(inner: string): string {
    const m = inner.match(/"name"\s*:\s*"([^"]*)"/);
    return m ? m[1] : "";
}

/** Turn the inner JSON text of one call into a ToolCall. Tolerant: malformed
 *  JSON keeps the raw text and never throws. */
function toCall(inner: string, pending: boolean): ToolCall {
    const raw = inner.trim();
    try {
        const obj = JSON.parse(raw) as Record<string, unknown>;
        const name =
            (typeof obj.name === "string" && obj.name) ||
            // OpenAI-style `{ function: { name, arguments } }`
            (typeof (obj.function as { name?: unknown })?.name === "string" &&
                (obj.function as { name: string }).name) ||
            "";
        // Accept the common arg keys; fall back to the object minus `name`.
        const fn = obj.function as { arguments?: unknown } | undefined;
        const args =
            (obj.arguments as Record<string, unknown>) ??
            (obj.parameters as Record<string, unknown>) ??
            (obj.args as Record<string, unknown>) ??
            (fn?.arguments as Record<string, unknown>) ??
            stripName(obj);
        return { name: name || "tool", arguments: args, raw, pending };
    } catch {
        // Not valid JSON (malformed, or still streaming) — keep it raw.
        return { name: sniffName(raw), arguments: raw, raw, pending };
    }
}

function stripName(obj: Record<string, unknown>): Record<string, unknown> {
    const { name: _name, ...rest } = obj;
    void _name;
    return rest;
}

export function parseToolCalls(response: string): ParsedToolCalls {
    if (!response || !response.includes(TOOL_CALL_OPEN)) {
        return { calls: [], prose: response ?? "", pending: false };
    }

    const calls: ToolCall[] = [];
    let prose = "";
    let pending = false;
    let cursor = 0;

    for (;;) {
        const open = response.indexOf(TOOL_CALL_OPEN, cursor);
        if (open < 0) {
            // No more calls — everything left is prose.
            prose += response.slice(cursor);
            break;
        }

        // Text before this call is prose.
        prose += response.slice(cursor, open);

        const innerStart = open + TOOL_CALL_OPEN.length;
        const close = response.indexOf(TOOL_CALL_CLOSE, innerStart);
        if (close < 0) {
            // Open marker with no close yet — still streaming this call in.
            calls.push(toCall(response.slice(innerStart), true));
            pending = true;
            break; // nothing after an unterminated call
        }

        calls.push(toCall(response.slice(innerStart, close), false));
        cursor = close + TOOL_CALL_CLOSE.length;
    }

    return { calls, prose: prose.trim(), pending };
}
