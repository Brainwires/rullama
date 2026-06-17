// Single source of truth for the system-turn content.
//
// Both the chat send path and the system-prompt PRE-WARM (which prefills
// the system block into the KV cache so a new chat hot-starts) must build
// this identically — otherwise the warmed prefix wouldn't match what a real
// turn renders and the KV cache wouldn't reuse it. The warm passes no RAG /
// GPS (those are per-turn, dynamic), so a plain (no-RAG, no-GPS) turn's
// system block is byte-identical to the warmed one.

import { THINK_TOKEN, TIMESTAMP_SYSTEM_NOTE, FORMATTING_SYSTEM_NOTE } from "./app-helpers";
import { TOOL_SCHEMA_PROMPT } from "./toolFormat";

export interface SysContentParts {
    /** The user's raw system prompt (the editable "system message"). */
    systemPrompt: string;
    thinking: boolean;
    toolMode: boolean;
    /** Per-turn RAG preamble (already assembled). Omit/empty for the warm
     *  path and non-RAG turns — RAG content is dynamic per query. */
    ragPreamble?: string;
    /** Per-turn GPS location line INCLUDING its trailing separator. Omit
     *  for the warm path; only injected on tool-mode turns with GPS. */
    gpsLine?: string;
}

/**
 * Assemble the system-turn content, ordered so the STATIC, cacheable core
 * comes first and the per-turn DYNAMIC content is appended at the end:
 *
 *   [think] tool schema → system prompt → notes   ← static (this is the warm)
 *           → RAG preamble → GPS line             ← dynamic, appended
 *
 * This ordering is load-bearing for KV reuse: the pre-warm prefills only
 * the static core, so a turn that adds RAG/GPS still shares that whole core
 * as a prefix (and only re-feeds the appended tail) instead of shifting it
 * and forcing a full re-prefill. With no RAG/GPS the result is byte-
 * identical to the warm.
 */
export function buildSysContent(p: SysContentParts): string {
    // Static core — exactly what the warm prefills.
    let base = p.systemPrompt.trim();
    if (p.toolMode) {
        base = base ? `${TOOL_SCHEMA_PROMPT}\n\n${base}` : TOOL_SCHEMA_PROMPT;
    }
    base = base ? `${base}\n\n${TIMESTAMP_SYSTEM_NOTE}` : TIMESTAMP_SYSTEM_NOTE;
    base = `${base}\n\n${FORMATTING_SYSTEM_NOTE}`;
    // Dynamic tail — appended so it never shifts the static prefix.
    const rag = p.ragPreamble?.trim();
    if (rag) base = `${base}\n\n${rag}`;
    const gps = p.gpsLine?.trim();
    if (gps) base = `${base}\n\n${gps}`;
    return p.thinking ? `${THINK_TOKEN}\n${base}` : base;
}
