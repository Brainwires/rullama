// Model catalog + tiny diagnostic beacons.
//
// The catalog is BAKED IN, not fetched. rullama is meant to run as an
// installable PWA — first paint shouldn't wait on a network round-trip
// it can't complete offline. The static list below is identical to
// what the production server's /api/models would have returned anyway
// (the public demo hosts blobs on R2; the server is just a JSON
// passthrough). Local-dev users running their own Ollama can rebuild
// from source after editing this constant.
//
// Keep this in sync with `examples/web/server/ollama.ts:huggingfaceModels()`
// and `docker/entrypoint.sh:emit_hf_entries()`.

export interface ModelEntry {
    name:   string;
    family: string;
    tag:    string;
    size:   number;
    digest: string;
    /** When set, the client fetches the blob from this absolute URL
     *  instead of /api/blob. Used by the public demo to offload model
     *  bandwidth to a CDN (R2, Hugging Face). */
    url?:   string;
    /** When `url` points at a full Ollama-style multimodal blob (text +
     *  vision + audio), set this so the loader doesn't force text-only.
     *  Default for HF-style text-only GGUFs: omit. */
    multimodal?: boolean;
}

const R2_HOST = "models.brainwires.dev";

/**
 * Baked-in catalog — what an offline-installed PWA sees. Mirrors the
 * server's huggingfaceModels() one-for-one. Each blob is hosted on
 * Cloudflare R2 with $0 egress, CORS allowed for `rullama.com`,
 * and Range support — the same loader code path used in dev works
 * unchanged.
 */
export const BAKED_IN_MODELS: readonly ModelEntry[] = [
    {
        name:       "gemma4:e2b",
        family:     "gemma4",
        tag:        "e2b",
        size:       7162394016,
        digest:     "4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448",
        url:        `https://${R2_HOST}/gemma4-e2b.gguf`,
        multimodal: true,
    },
    {
        name:       "gemma4:e4b",
        family:     "gemma4",
        tag:        "e4b",
        size:       9608338848,
        digest:     "4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a",
        url:        `https://${R2_HOST}/gemma4-e4b.gguf`,
        multimodal: true,
    },
];

/** Whether this entry is something we'll actually run. */
export function isSupported(m: ModelEntry): boolean {
    return m.family === "gemma4";
}

/**
 * Return the model catalog. Always succeeds; never throws. When the
 * Hono dev server is reachable (local-Ollama dev case) we prefer its
 * response — that's the only path where the list might differ from
 * the baked catalog. Otherwise (production PWA / offline / no server)
 * we return the baked constant immediately.
 *
 * A short timeout keeps the offline path responsive: we don't want
 * the model picker to hang waiting for a request that's going to fail
 * with a service-worker network-error 30 seconds later.
 */
export async function listModels(signal?: AbortSignal): Promise<ModelEntry[]> {
    if (typeof navigator !== "undefined" && navigator.onLine === false) {
        return BAKED_IN_MODELS.slice();
    }
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 1500);
    const onAbort = () => ctl.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    try {
        const resp = await fetch("/api/models", { signal: ctl.signal });
        if (resp.ok) {
            const rows = (await resp.json()) as ModelEntry[];
            if (Array.isArray(rows) && rows.length > 0) return rows;
        }
    } catch { /* timeout / network / service-worker miss — fall through */ }
    finally {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
    }
    return BAKED_IN_MODELS.slice();
}

/** Where to fetch the GGUF bytes from. The baked entries all carry a
 *  url; this only falls back to /api/blob for local-Ollama dev mode. */
export function blobUrl(m: ModelEntry): string {
    if (m.url) return m.url;
    return "/api/blob/" + encodeURIComponent(m.name);
}

/**
 * Fire-and-forget diagnostic beacon. Used to record load events to the
 * dev server's /tmp/rullama-page.log. Offline / production deploys
 * silently no-op because the server has no /api/log handler or the
 * tab is offline — fine either way, the app doesn't depend on it.
 */
export function beacon(tag: string, msg: string) {
    try {
        fetch("/api/log", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ tag, msg, ts: Date.now() }),
            keepalive: true,
        }).catch(() => {});
    } catch { /* no-op */ }
}
