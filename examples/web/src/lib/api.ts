// Typed wrappers over the Hono API at /api/*.

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

/** Whether this entry is something we'll actually run. */
export function isSupported(m: ModelEntry): boolean {
    return m.family === "gemma4";
}

export async function listModels(signal?: AbortSignal): Promise<ModelEntry[]> {
    const resp = await fetch("/api/models", { signal });
    if (!resp.ok) throw new Error(`/api/models → ${resp.status}`);
    return resp.json();
}

/** Where to fetch the GGUF bytes from. Prefer the model's own URL
 *  (public CDN) when present; fall back to the local API blob stream. */
export function blobUrl(m: ModelEntry): string {
    if (m.url) return m.url;
    return "/api/blob/" + encodeURIComponent(m.name);
}

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
