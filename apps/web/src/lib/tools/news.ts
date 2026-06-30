// get_news — recent headlines. Optional key (Tools tab), like the WeatherAPI
// key: degrades gracefully when unset. Uses GNews (gnews.io) because it sends
// CORS headers for browser clients (NewsAPI.org blocks non-localhost browser
// requests, so it can't be used from a deployed PWA).

import type { ToolDef, ToolRunResult } from "@/lib/tools/types";

export const newsTool: ToolDef = {
    names: ["get_news", "news", "search_news", "headlines"],
    async run(_name, args, ctx): Promise<ToolRunResult> {
        const key = ctx.newsApiKey.trim();
        if (!key) {
            return {
                ok: false,
                summary: "News needs a (free) GNews API key — add one in the Tools tab " +
                    "(https://gnews.io). Until then, answer from your own knowledge.",
            };
        }
        const query =
            (typeof args.query === "string" && args.query.trim()) ||
            (typeof args.q === "string" && args.q.trim()) ||
            (typeof args.topic === "string" && args.topic.trim()) ||
            "";
        if (!query) return { ok: false, summary: "No news query was given." };

        try {
            const url = new URL("https://gnews.io/api/v4/search");
            url.searchParams.set("q", query);
            url.searchParams.set("lang", "en");
            url.searchParams.set("max", "5");
            url.searchParams.set("apikey", key);
            const resp = await fetch(url.toString());
            if (!resp.ok) {
                const err = await resp.json().catch(() => null);
                throw new Error(err?.errors?.[0] || `HTTP ${resp.status}`);
            }
            const d = await resp.json();
            const articles: any[] = d.articles ?? [];
            if (!articles.length) return { ok: false, summary: `No recent news found for "${query}".` };
            const lines = articles.map((a, i) =>
                `${i + 1}. ${a.title} — ${a.source?.name ?? "?"} (${a.publishedAt?.slice(0, 10) ?? ""})\n   ${a.url}`);
            return {
                ok: true,
                summary: `Top news for "${query}":\n${lines.join("\n")}`,
                data: { count: articles.length },
            };
        } catch (e) {
            return { ok: false, summary: `News lookup failed: ${(e as Error).message}` };
        }
    },
};
