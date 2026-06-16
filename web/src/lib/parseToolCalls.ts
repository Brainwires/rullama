// Split a model reply into tool-call blocks + the surrounding prose.
//
// Mirrors lib/parseModel.ts: it scans for the `<tool_call>` / `</tool_call>`
// markers (defined in lib/toolFormat.ts), pulls each call out, and leaves the
// rest as prose for the normal markdown pipeline. Streaming-safe — an opened
// but not-yet-closed call surfaces as `pending` so the UI can show a pulse.
//
// A reply with no `<tool_call>` marker passes straight through (calls === []).

import {
    TOOL_CALL_OPEN_PREFIX,
    TOOL_CALL_CLOSE,
    TOOL_PARAMS,
    type ToolCall,
} from "@/lib/toolFormat";

/** One piece of a reply, in original emission order — either a run of prose
 *  or a tool call. Lets the UI interleave reasoning between calls (call →
 *  result → reasoning → next call → …) instead of grouping all calls first. */
export type ToolSegment =
    | { kind: "prose"; text: string }
    | { kind: "call"; call: ToolCall };

export interface ParsedToolCalls {
    calls: ToolCall[];
    /** The reply text with all tool-call spans removed (all prose concatenated). */
    prose: string;
    /** True if the last call is still streaming (open seen, no close yet). */
    pending: boolean;
    /** Prose + calls in original order, for in-order rendering. */
    segments: ToolSegment[];
}

/** Best-effort name sniff from a partial JSON body, so a still-streaming call
 *  can show "Calling <name>…" before its JSON is complete. */
function sniffName(inner: string): string {
    const m = inner.match(/"name"\s*:\s*"([^"]*)"/);
    return m ? m[1] : "";
}

/** Find the index just past the first balanced `{…}` at/after `from`,
 *  respecting strings + escapes. Returns -1 if there's no balanced object
 *  (still streaming mid-object, or no object at all). Lets us recover a
 *  complete call even when the model omits the `</tool_call>` closer (it
 *  often emits the JSON then stops or trails extra braces). */
function matchJsonEnd(s: string, from: number): number {
    let i = from;
    while (i < s.length && s[i] !== "{") {
        if (!/\s/.test(s[i])) return -1; // non-whitespace before `{` → not an object
        i++;
    }
    let depth = 0;
    let inStr = false;
    let esc = false;
    for (; i < s.length; i++) {
        const c = s[i];
        if (inStr) {
            if (esc) esc = false;
            else if (c === "\\") esc = true;
            else if (c === '"') inStr = false;
            continue;
        }
        if (c === '"') inStr = true;
        else if (c === "{") depth++;
        else if (c === "}" && --depth === 0) return i + 1;
    }
    return -1; // never balanced
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
        // Not JSON — small models often emit pythonic call syntax instead
        // (e.g. `set_timer(7)` or `send_email(to="Priya", subject="…")`).
        // Accept that too; both formats are standard per BFCL.
        const py = parsePythonicCall(raw);
        if (py) return { name: py.name, arguments: py.arguments, raw, pending };
        // Genuinely malformed / still streaming — keep it raw.
        return { name: sniffName(raw), arguments: raw, raw, pending };
    }
}

/** Split a pythonic arg list on top-level commas, respecting quotes/brackets. */
function splitArgs(s: string): string[] {
    const out: string[] = [];
    let depth = 0, inStr = false, esc = false, q = "", start = 0;
    for (let i = 0; i < s.length; i++) {
        const c = s[i];
        if (inStr) {
            if (esc) esc = false;
            else if (c === "\\") esc = true;
            else if (c === q) inStr = false;
            continue;
        }
        if (c === '"' || c === "'") { inStr = true; q = c; }
        else if (c === "(" || c === "[" || c === "{") depth++;
        else if (c === ")" || c === "]" || c === "}") depth--;
        else if (c === "," && depth === 0) { out.push(s.slice(start, i)); start = i + 1; }
    }
    out.push(s.slice(start));
    return out.map((x) => x.trim()).filter((x) => x.length > 0);
}

/** Coerce a pythonic scalar literal to a JS value (quotes/number/bool/null). */
function parseScalar(v: string): unknown {
    const t = v.trim();
    if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) {
        return t.slice(1, -1);
    }
    if (t === "true" || t === "True") return true;
    if (t === "false" || t === "False") return false;
    if (t === "null" || t === "None") return null;
    if (t !== "" && !Number.isNaN(Number(t))) return Number(t);
    return t; // bareword → string
}

/** Parse `name(args)` pythonic call syntax into `{ name, arguments }`, mapping
 *  positional args to schema param names. Returns null if it isn't a call. */
function parsePythonicCall(raw: string): { name: string; arguments: Record<string, unknown> } | null {
    const m = raw.match(/^\s*([A-Za-z_]\w*)\s*\(([\s\S]*)\)\s*$/);
    if (!m) return null;
    const name = m[1];
    const argstr = m[2].trim();
    const args: Record<string, unknown> = {};
    if (argstr.length > 0) {
        const order = TOOL_PARAMS[name] ?? [];
        splitArgs(argstr).forEach((part, idx) => {
            const eq = part.indexOf("=");
            const isKwarg = eq > 0 && /^[A-Za-z_]\w*$/.test(part.slice(0, eq).trim());
            if (isKwarg) {
                args[part.slice(0, eq).trim()] = parseScalar(part.slice(eq + 1));
            } else {
                args[order[idx] ?? `arg${idx}`] = parseScalar(part);
            }
        });
    }
    return { name, arguments: args };
}

function stripName(obj: Record<string, unknown>): Record<string, unknown> {
    const { name: _name, ...rest } = obj;
    void _name;
    return rest;
}

export function parseToolCalls(response: string): ParsedToolCalls {
    if (!response || !response.includes(TOOL_CALL_OPEN_PREFIX)) {
        const text = response ?? "";
        return {
            calls: [], prose: text, pending: false,
            segments: text.trim() ? [{ kind: "prose", text }] : [],
        };
    }

    // Pull out any executed-tool result spans first so they never leak into the
    // visible prose. Each carries an optional `for="<tool_name>"` (see
    // toolResponseBlock) so we can match it to the right call even when several
    // tools ran; reattached after parsing. The tags are literal (no regex
    // metacharacters to escape).
    const results: { name: string | null; text: string }[] = [];
    const cleaned = response.replace(
        /<tool_response(?:\s+for="([^"]*)")?>([\s\S]*?)<\/tool_response>/g,
        (_m, name: string | undefined, inner: string) => {
            results.push({ name: name ?? null, text: inner.trim() });
            return "";
        },
    );

    const calls: ToolCall[] = [];
    const segments: ToolSegment[] = [];
    let prose = "";
    let pending = false;
    let cursor = 0;

    // Prose accumulates into one string (back-compat) AND becomes an ordered
    // segment (skipping whitespace-only gaps); a call pushes to both `calls`
    // and `segments`. Segments hold the SAME call object refs, so the result
    // reattachment below (which mutates `call.result`) shows through.
    const addProse = (text: string) => {
        prose += text;
        if (text.trim()) segments.push({ kind: "prose", text: text.trim() });
    };
    const addCall = (call: ToolCall) => {
        calls.push(call);
        segments.push({ kind: "call", call });
    };

    for (;;) {
        const open = cleaned.indexOf(TOOL_CALL_OPEN_PREFIX, cursor);
        if (open < 0) {
            // No more calls — everything left is prose.
            addProse(cleaned.slice(cursor));
            break;
        }

        // Text before this call is prose.
        addProse(cleaned.slice(cursor, open));

        // Skip the opening tag, tolerating a missing `>` (e.g. `<tool_call\n{…`).
        let innerStart = open + TOOL_CALL_OPEN_PREFIX.length;
        if (cleaned[innerStart] === ">") innerStart++;
        const close = cleaned.indexOf(TOOL_CALL_CLOSE, innerStart);
        if (close < 0) {
            // No explicit </tool_call>. Models frequently omit it — they emit
            // the JSON object then stop (or trail extra braces). Brace-match a
            // complete object so it still renders as a finished call instead of
            // pulsing forever.
            const end = matchJsonEnd(cleaned, innerStart);
            if (end >= 0) {
                addCall(toCall(cleaned.slice(innerStart, end), false));
                cursor = end;
                // Swallow stray closing braces / whitespace the model tacked on.
                while (cursor < cleaned.length && (cleaned[cursor] === "}" || /\s/.test(cleaned[cursor]))) {
                    cursor++;
                }
                continue;
            }
            // Genuinely mid-JSON — still streaming this call in.
            addCall(toCall(cleaned.slice(innerStart), true));
            pending = true;
            break; // nothing usable after an unterminated call
        }

        addCall(toCall(cleaned.slice(innerStart, close), false));
        cursor = close + TOOL_CALL_CLOSE.length;
    }

    // Reattach executed results to their calls: prefer a name match (robust
    // when tools ran out of order or non-executable calls are interspersed),
    // falling back to the next still-unfilled call in emission order.
    let posIdx = 0;
    for (const res of results) {
        let target: ToolCall | undefined;
        if (res.name) {
            const want = res.name.toLowerCase();
            target = calls.find((c) => c.result === undefined && c.name.toLowerCase() === want);
        }
        if (!target) {
            while (posIdx < calls.length && calls[posIdx].result !== undefined) posIdx++;
            target = calls[posIdx];
        }
        if (target) target.result = res.text;
    }

    return { calls, prose: prose.trim(), pending, segments };
}
