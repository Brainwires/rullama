// search_knowledge — searches the user's OWN documents (the Knowledge tab's
// vector store) on demand, via the existing embedder. This is the on-demand
// replacement for the always-on RAG toggle: retrieval happens only when the
// model decides it needs the user's docs. Fully local / private.

import type { ToolDef, ToolRunResult } from "@/lib/tools/types";
import { searchKnowledge, buildRagPreamble, ensureEmbedder } from "@/lib/embedding";

export const knowledgeTool: ToolDef = {
    names: ["search_knowledge", "search_notes", "search_documents", "search_docs"],
    async run(_name, args, ctx): Promise<ToolRunResult> {
        const query =
            (typeof args.query === "string" && args.query.trim()) ||
            (typeof args.q === "string" && args.q.trim()) ||
            "";
        if (!query) return { ok: false, summary: "No search query was given for the knowledge base." };

        try {
            // Lazy-load the embedder on first use (no manual load step).
            await ensureEmbedder();
            const hits = await searchKnowledge(query, { k: 5, conversationId: ctx.conversationId });
            if (!hits.length) {
                return {
                    ok: false,
                    summary: `No matching documents for "${query}". The user may not have added any to the Knowledge tab.`,
                };
            }
            const preamble = buildRagPreamble(hits).trim();
            return {
                ok: true,
                summary: preamble || `Found ${hits.length} result(s) but no text to show.`,
                data: { count: hits.length },
            };
        } catch (e) {
            return {
                ok: false,
                summary: `Knowledge search failed (is an embedding model loaded?): ${(e as Error).message}`,
            };
        }
    },
};
