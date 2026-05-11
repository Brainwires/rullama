// Typed wrappers over the Hono API at /api/*.

export interface ModelEntry {
    name:   string;
    family: string;
    tag:    string;
    size:   number;
    digest: string;
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

export function blobUrl(name: string): string {
    return "/api/blob/" + encodeURIComponent(name);
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
