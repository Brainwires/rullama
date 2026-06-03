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

/**
 * Kokoro-82M TTS model — a SEPARATE model from the chat catalog above (it is
 * not a Gemma chat model, so it must NOT appear in the model picker). The Voice
 * tab loads this on its own handle. f16 GGUF on R2 (~164 MB), same Range +
 * OPFS-cache loader path as the chat models. digest is the OPFS cache key.
 */
export const KOKORO_MODEL = {
    name:   "kokoro:82m",
    family: "kokoro",
    tag:    "82m",
    size:   163763104,
    digest: "b17cbd233537648079f9b8008d60e45860375f8f0a4fefac8982e9c48335e55c",
    url:    `https://${R2_HOST}/kokoro-82m.gguf`,
} as const;

/** Where to fetch the Kokoro TTS GGUF from. Honors `?localBlob=PORT` (serves from the
 *  local devserver's /api/blob/kokoro:82m, bypassing R2) just like the chat models. */
export function kokoroBlobUrl(): string {
    const port = localBlobPort();
    if (port != null) return `http://localhost:${port}/api/blob/${encodeURIComponent("kokoro:82m")}`;
    return KOKORO_MODEL.url;
}

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
 *  url; this only falls back to /api/blob for local-Ollama dev mode.
 *
 *  **Local-blob override** (debug/automation): when the page URL has
 *  `?localBlob=PORT` (or localStorage.localBlobPort is set), GGUF
 *  fetches go to `http://localhost:PORT/api/blob/...` instead of the
 *  same-origin /api/blob. Lets the Cloudflare-served PWA shell stay
 *  on the public URL (small files, edge-cached) while the 7 GB GGUF
 *  goes direct to the local server, sidestepping the user's home
 *  upload speed.
 *
 *  Server (`serve-tunnel.sh`) must send the matching CORS + CORP
 *  headers — the PWA loads inside a cross-origin-isolated context
 *  (require-corp), so the cross-origin blob fetch needs
 *  `Access-Control-Allow-Origin` AND `Cross-Origin-Resource-Policy:
 *  cross-origin` to satisfy the isolation policy.
 */
export function blobUrl(m: ModelEntry): string {
    // **`?localBlob` wins over the baked-in CDN URL.** Otherwise the
    // baked-in entries (which always carry `m.url` pointing at the R2
    // CDN) silently ignore the override — the very case the override
    // exists for (split origin: PWA from Cloudflare, GGUF from local
    // devserver to save home upload bandwidth). The localBlob target
    // is the user's own machine, so the trust model is the same as
    // any other localhost dev fetch.
    const port = localBlobPort();
    if (port != null) {
        return `http://localhost:${port}/api/blob/${encodeURIComponent(m.name)}`;
    }
    if (m.url) return m.url;
    return "/api/blob/" + encodeURIComponent(m.name);
}

/** Returns the local-blob port from `?localBlob=PORT` URL param or
 *  `localStorage.localBlobPort`. URL param also writes to localStorage
 *  so it persists across reloads. Returns null when neither is set. */
function localBlobPort(): number | null {
    if (typeof window === "undefined") return null;
    try {
        const fromUrl = new URLSearchParams(window.location.search).get("localBlob");
        if (fromUrl) {
            const n = parseInt(fromUrl, 10);
            if (Number.isFinite(n) && n > 0 && n < 65536) {
                window.localStorage.setItem("localBlobPort", String(n));
                return n;
            }
        }
        const stored = window.localStorage.getItem("localBlobPort");
        if (stored) {
            const n = parseInt(stored, 10);
            if (Number.isFinite(n) && n > 0 && n < 65536) return n;
        }
    } catch {
        // localStorage unavailable (Worker scope, privacy mode).
    }
    return null;
}

/**
 * Fire-and-forget diagnostic beacon. Records to TWO sinks:
 *   1. Dev-server /api/log (lands at /tmp/rullama-page.log on the Mac
 *      running the safaridriver harness). Silently no-ops offline /
 *      in production where the endpoint doesn't exist.
 *   2. OPFS via the worker — crash-surviving, viewable on-device in
 *      Settings → Logs even after iOS jetsam kills the tab. This is
 *      the path that actually matters for iPhone debugging.
 *
 * Both calls are fire-and-forget — beacons must never block the UI or
 * propagate an error to the caller.
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
    // Lazy-import to dodge a circular dependency: lib/inference.ts
    // imports from lib/api.ts at module load. We need getClient at
    // call time, not import time.
    try {
        void import("./inference").then(({ getClient }) => {
            try { getClient().logs.append("info", tag, msg); } catch { /* */ }
        }).catch(() => {});
    } catch { /* */ }
}
