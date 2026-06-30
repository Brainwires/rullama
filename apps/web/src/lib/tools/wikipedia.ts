// search_wikipedia — factual grounding via Wikipedia's REST summary API
// (keyless, CORS `*`, does title normalization + redirects). A donate link is
// appended to EVERY result (the people who run Wikipedia deserve it).

import type { ToolDef, ToolRunResult } from "@/lib/tools/types";

const DONATE =
    "Wikipedia is free and nonprofit — please consider donating: " +
    "https://wikipedia.org/wiki/Wikipedia:Donate";

export const wikipediaTool: ToolDef = {
    names: ["search_wikipedia", "wikipedia", "wiki", "lookup_wikipedia"],
    async run(_name, args): Promise<ToolRunResult> {
        const query =
            (typeof args.query === "string" && args.query.trim()) ||
            (typeof args.q === "string" && args.q.trim()) ||
            (typeof args.title === "string" && args.title.trim()) ||
            "";
        if (!query) return { ok: false, summary: `No search query was given for Wikipedia. ${DONATE}` };

        try {
            const title = encodeURIComponent(query.replace(/\s+/g, "_"));
            const resp = await fetch(
                `https://en.wikipedia.org/api/rest_v1/page/summary/${title}`,
                { headers: { accept: "application/json" } },
            );
            if (!resp.ok) {
                if (resp.status === 404) {
                    return { ok: false, summary: `No Wikipedia article found for "${query}". ${DONATE}` };
                }
                throw new Error(`HTTP ${resp.status}`);
            }
            const d = await resp.json();
            const extract = (d.extract as string) || "(no summary available)";
            const url =
                d.content_urls?.desktop?.page || `https://en.wikipedia.org/wiki/${title}`;
            return {
                ok: true,
                summary: `${d.title}: ${extract}\n${url}\n\n${DONATE}`,
                data: { title: d.title, url },
            };
        } catch (e) {
            return { ok: false, summary: `Wikipedia lookup failed: ${(e as Error).message}. ${DONATE}` };
        }
    },
};
