// Split a streaming model reply into an optional thinking block + the
// user-facing response. Gemma 4 emits its reasoning channel inline:
//
//   <|channel>thought\n
//   ...internal reasoning...
//   <channel|>actual response
//
// During streaming we may see the opening marker but not the closing one
// yet — that's `isComplete: false` (the chat UI keeps the block expanded
// with a pulse). Once `<channel|>` arrives, the response part follows
// and the thinking block can collapse.
//
// Streams without a `<|channel>thought` marker pass through unchanged
// (thinking === null).

export interface ParsedModel {
    thinking:   string | null;
    response:   string;
    isThinking: boolean;   // true → currently mid-thought (no close marker yet)
    isComplete: boolean;   // true → close marker seen (thinking block can collapse)
}

/** Gemma 4 reasoning-channel markers. Exported so the segment parser
 *  (parseToolCalls) can split out thinking blocks interleaved with tool
 *  calls — e.g. a fresh thought after each tool result. */
export const CHANNEL_OPEN  = "<|channel>thought";
export const CHANNEL_CLOSE = "<channel|>";

const OPEN  = CHANNEL_OPEN;
const CLOSE = CHANNEL_CLOSE;

export function parseModelContent(raw: string): ParsedModel {
    if (!raw) return { thinking: null, response: "", isThinking: false, isComplete: false };

    const openIdx = raw.indexOf(OPEN);
    if (openIdx < 0) {
        return { thinking: null, response: raw, isThinking: false, isComplete: true };
    }

    // Any text before OPEN is treated as response prelude. In practice
    // Gemma puts the thought block first, but be defensive.
    const prelude   = raw.slice(0, openIdx);
    const afterOpen = raw.slice(openIdx + OPEN.length);

    const closeIdx = afterOpen.indexOf(CLOSE);
    if (closeIdx < 0) {
        // Still inside the thought block. Trim a single leading newline
        // — Gemma emits "<|channel>thought\n" — so the rendered text
        // starts cleanly.
        const thought = afterOpen.replace(/^\n/, "");
        return {
            thinking:   thought,
            response:   prelude,
            isThinking: true,
            isComplete: false,
        };
    }

    const thought  = afterOpen.slice(0, closeIdx).replace(/^\n/, "");
    const response = (prelude + afterOpen.slice(closeIdx + CLOSE.length)).replace(/^\n/, "");
    return {
        thinking:   thought,
        response,
        isThinking: false,
        isComplete: true,
    };
}
