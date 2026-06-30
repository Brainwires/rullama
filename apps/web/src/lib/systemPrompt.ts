// Single source of truth for the system-turn content.
//
// Both the chat send path and the system-prompt PRE-WARM (which prefills
// the system block into the KV cache so a new chat hot-starts) must build
// this identically — otherwise the warmed prefix wouldn't match what a real
// turn renders and the KV cache wouldn't reuse it. The content is now FULLY
// STATIC (no per-turn dynamic tail), so every turn's system block is
// byte-identical to the warmed one. (RAG became the on-demand
// `search_knowledge` tool, and GPS is resolved per tool call, not injected
// into the system prefix — neither touches this builder anymore.)

import { THINK_TOKEN, TIMESTAMP_SYSTEM_NOTE, FORMATTING_SYSTEM_NOTE } from "./app-helpers";
import { TOOL_SCHEMA_PROMPT } from "./toolFormat";
import { orchestratorPreamble } from "./tools/orchestrator";

export interface SysContentParts {
    /** The user's raw system prompt (the editable "system message"). */
    systemPrompt: string;
    thinking: boolean;
    toolMode: boolean;
    /** Programmatic tool calling: when on (and toolMode), the tool block is the
     *  Rhai-script orchestrator preamble instead of the JSON `<tool_call>`
     *  schema. Changes the system signature → a one-time re-warm. */
    orchestratorMode?: boolean;
}

/**
 * Assemble the system-turn content:
 *
 *   [think] tool schema → system prompt → notes
 *
 * Fully static and cacheable — the pre-warm prefills exactly this, so a real
 * turn's system block is byte-identical to the warm and the KV cache reuses the
 * whole prefix. (RAG and GPS used to be dynamic tails here; RAG is now the
 * on-demand `search_knowledge` tool and GPS is resolved per tool call, so
 * nothing dynamic touches the system prefix.)
 */
export function buildSysContent(p: SysContentParts): string {
    let base = p.systemPrompt.trim();
    if (p.toolMode) {
        const toolBlock = p.orchestratorMode ? orchestratorPreamble() : TOOL_SCHEMA_PROMPT;
        base = base ? `${toolBlock}\n\n${base}` : toolBlock;
    }
    base = base ? `${base}\n\n${TIMESTAMP_SYSTEM_NOTE}` : TIMESTAMP_SYSTEM_NOTE;
    base = `${base}\n\n${FORMATTING_SYSTEM_NOTE}`;
    return p.thinking ? `${THINK_TOKEN}\n${base}` : base;
}
