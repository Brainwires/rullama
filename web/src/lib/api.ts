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
// Keep this in sync with `web/server/ollama.ts:huggingfaceModels()`
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
    /** Advisory "this is a heavy model" flag. ModelLoader renders a ⚠
     *  caution marker + tooltip; Load is NOT blocked (the user can try it
     *  on any tier). Set for the large models (e.g. gemma4:12b). */
    heavy?: boolean;
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
    {
        // Google's 12B (released 2026-06). Same gemma4 architecture (verified:
        // general.architecture=gemma4, all dims dynamic), Q4_K_M, text-only (no
        // vision/audio towers). Very heavy — barely runs on Metal; flagged
        // `heavy` so the picker shows a ⚠ (advisory, never blocked).
        name:       "gemma4:12b",
        family:     "gemma4",
        tag:        "12b",
        size:       7381382048,
        digest:     "1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606",
        url:        `https://${R2_HOST}/gemma4-12b.gguf`,
        heavy:      true,
    },
    {
        // Google's QAT (quantization-aware-trained) e2b — legacy ggml Q4_0 text
        // weights. ~3.3 GB vs the 7.16 GB standard e2b: less than half the
        // download, quality preserved by the QAT. TEXT-ONLY here — Ollama ships
        // the QAT vision/audio towers as a separate projector blob, not merged
        // into this GGUF (a merged multimodal QAT build is a follow-up), so
        // `multimodal` is omitted. Runs through the same gemma4 forward path
        // (Q4_0 dequant-matmul; see backend/dispatch/matmul.rs).
        name:       "gemma4:e2b-it-qat",
        family:     "gemma4",
        tag:        "e2b-it-qat",
        size:       3349514112,
        digest:     "3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd",
        url:        `https://${R2_HOST}/gemma4-e2b-it-qat.gguf`,
    },
    {
        // QAT e4b — Q4_0 text weights, 5.15 GB (vs 9.6 GB standard e4b). Text-only
        // (QAT towers ship as a separate projector blob, not merged).
        name:       "gemma4:e4b-it-qat",
        family:     "gemma4",
        tag:        "e4b-it-qat",
        size:       5154939136,
        digest:     "e8b6a059ba86947a44ace84d6e5679795bc41862c25c30513142588f0e9dba1d",
        url:        `https://${R2_HOST}/gemma4-e4b-it-qat.gguf`,
    },
    {
        // QAT 12b — Q4_0 text weights, 6.98 GB (vs 7.38 GB standard 12b). Text-only.
        // The 12b architecture needs per-layer KV heads + no-V global attention
        // (global layers reuse raw K as V) — supported as of the gemma4 12b commit.
        // Heavy → advisory ⚠.
        name:       "gemma4:12b-it-qat",
        family:     "gemma4",
        tag:        "12b-it-qat",
        size:       6975877728,
        digest:     "faff1a63667fac17ac5e777f47114688fcefea96e220e211aaa8d62c2c4561f1",
        url:        `https://${R2_HOST}/gemma4-12b-it-qat.gguf`,
        heavy:      true,
    },
    {
        // e2b @ Q8_0 — 8-bit weights, the highest-quality quant we run. 8.14 GB
        // (vs 7.16 GB Q4_K_M). FULL multimodal blob (text Q8_0 + the same
        // BF16/F16 vision+audio towers as the standard e2b — verified by tensor
        // census), unlike the text-only QAT builds. Q8_0 dequant-matmul in
        // backend/dispatch/matmul.rs.
        name:       "gemma4:e2b-it-q8_0",
        family:     "gemma4",
        tag:        "e2b-it-q8_0",
        size:       8140140960,
        digest:     "6aade8551d1aecae00d6520d5db327efbef4b96ff92abef353ef6cd8e4e6d589",
        url:        `https://${R2_HOST}/gemma4-e2b-it-q8_0.gguf`,
        multimodal: true,
    },
    {
        // e4b @ Q8_0 — 11.6 GB, full multimodal blob (text Q8_0 + BF16/F16
        // towers, census-verified like e2b-it-q8_0).
        name:       "gemma4:e4b-it-q8_0",
        family:     "gemma4",
        tag:        "e4b-it-q8_0",
        size:       11636104608,
        digest:     "62d767a4c82f7acba2e1da74df317f01ce34b92830712c536260f82acfb63ac9",
        url:        `https://${R2_HOST}/gemma4-e4b-it-q8_0.gguf`,
        multimodal: true,
    },
    {
        // 12b @ Q8_0 — 12.7 GB, text-only (the 12b builds ship no towers).
        // Heavy → advisory ⚠.
        name:       "gemma4:12b-it-q8_0",
        family:     "gemma4",
        tag:        "12b-it-q8_0",
        size:       12669645728,
        digest:     "047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795",
        url:        `https://${R2_HOST}/gemma4-12b-it-q8_0.gguf`,
        heavy:      true,
    },
    {
        // Google's 26B-A4B sparse MoE (128 experts, top-8, ~4B active). 18 GB
        // Q4_K_M; runs through the gemma4 forward's parallel dense+expert FFN
        // branch (reference/moe.rs + dispatch/moe.rs). Expert weights alone are
        // ~12 GB GPU-resident — needs a high-VRAM desktop; heavy → advisory ⚠.
        name:       "gemma4:26b",
        family:     "gemma4",
        tag:        "26b",
        size:       17987569344,
        digest:     "7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df",
        url:        `https://${R2_HOST}/gemma4-26b.gguf`,
        heavy:      true,
    },
    {
        // DiffusionGemma 26B-A4B — Google's block-diffusion model on the same
        // 26B-A4B sparse-MoE backbone (Ollama can't run it; Unsloth GGUF). NOT
        // autoregressive: a 256-token canvas is denoised over up to 48 steps,
        // the whole canvas forwarded each step. Own engine (DiffusionGemma wasm
        // class), own family string → its own generation path in the worker.
        // 16.8 GB Q4_K_M; per-layer expert streaming keeps it off-RAM. Heavy →
        // advisory ⚠ (desktop; tens of seconds per denoise step on weak GPUs).
        name:       "diffusiongemma:26b-a4b",
        family:     "diffusion-gemma",
        tag:        "26b-a4b",
        size:       16806810336,
        digest:     "d2ca2c032ebfb23cf2d1794a3465e615c7545634d46b3c30652a26d8b07c4ad3",
        url:        `https://${R2_HOST}/diffusiongemma-26b-a4b.gguf`,
        heavy:      true,
    },
    {
        // Z-Image-Turbo — text-to-IMAGE generation (the 4th engine, ImageModel
        // wasm class). NOT a GGUF: three per-tensor safetensors components
        // (text_encoder/ Qwen3 + transformer/ DiT + vae/) live under the CDN
        // base `url` and stream per-tensor via HTTP Range — nothing downloads
        // to OPFS, so this entry never enters the model-picker / ensureModel
        // path. The Image tab loads it directly through `client.image.load`.
        // ~31 GB across the three components; desktop-only → heavy advisory ⚠
        // (and never appears in the chat model picker, which filters on
        // isSupported()). `size` here is advisory only.
        name:       "z-image-turbo",
        family:     "z-image",
        tag:        "turbo",
        size:       31000000000,
        digest:     "z-image-turbo",
        url:        `https://${R2_HOST}/z-image-turbo`,
        heavy:      true,
    },
];

/** The Z-Image-Turbo text-to-image model — the Image tab's engine. SEPARATE
 *  from the chat catalog (it's the ImageModel wasm class, not a Gemma chat
 *  model) and loaded from a CDN base URL via HTTP Range, never OPFS. */
export const IMAGE_MODEL = {
    name:    "z-image-turbo",
    family:  "z-image",
    tag:     "turbo",
    /** CDN root holding `text_encoder/`, `transformer/`, `vae/` (+ `tokenizer/`). */
    baseUrl: `https://${R2_HOST}/z-image-turbo`,
} as const;

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
    size:   170815072,
    digest: "69587a29cde2c1880aa517026a390afda40eabb874bba4b8dd4c31b9f313c496",
    url:    `https://${R2_HOST}/kokoro-82m.gguf`,
} as const;

/**
 * EmbeddingGemma-300M — the embedding model for the Knowledge tab + chat RAG.
 * SEPARATE from the chat catalog (architecture `gemma3`, encoder-only). 621 MB
 * bf16 GGUF, live on R2 at the `url` below (Range + CORS verified). Dev can
 * still fetch from local Ollama via `?localBlob` → `/api/blob`.
 */
export const EMBEDDING_MODEL = {
    // `name` doubles as the dev `/api/blob/<name>` key, so it must match the
    // local Ollama model name (`embeddinggemma:latest`). Production fetches
    // the R2 `url` instead (blobUrl prefers ?localBlob, then m.url).
    name:   "embeddinggemma:latest",
    family: "embeddinggemma",
    tag:    "300m",
    size:   621867104,
    digest: "0800cbac9c2064dde519420e75e512a83cb360de3ad5df176185dc69652fc515",
    url:    `https://${R2_HOST}/embeddinggemma-300m.gguf`,
    dim:    768,
} as const;

/** Where to fetch the Kokoro TTS GGUF from. Honors `?localBlob=PORT` (serves from the
 *  local devserver's /api/blob/kokoro:82m, bypassing R2) just like the chat models. */
export function kokoroBlobUrl(): string {
    const port = localBlobPort();
    if (port != null) return `http://localhost:${port}/api/blob/${encodeURIComponent("kokoro:82m")}`;
    return KOKORO_MODEL.url;
}

/** StyleTTS2-LibriTTS — the zero-shot VOICE-CLONING engine. Two precision variants:
 *  - **f32** (518 MB): full quality, the desktop default.
 *  - **f16** (259 MB): conv weights kept f16 on host + GPU — ~half the resident
 *    footprint for memory-tight devices, at a real precision cost (this deep
 *    vocoder degrades under f16). */
export const STYLETTS2_MODEL = {
    name:   "styletts2:libritts",
    family: "styletts2",
    tag:    "libritts",
    size:   543440288,
    digest: "1b174e1411e1632037a441502578f78b40d75341ac7c0cfa3ea4cf7c07c847b5",
    url:    `https://${R2_HOST}/styletts2-libritts.gguf`,
} as const;

export const STYLETTS2_MODEL_F16 = {
    name:   "styletts2:libritts:f16",
    family: "styletts2",
    tag:    "libritts-f16",
    size:   271920480,
    digest: "5535159c0f0f7f141e07d2a8713c84f7a1a8ae31ef9e6f272babe06a68a9354e",
    url:    `https://${R2_HOST}/styletts2-libritts-f16.gguf`,
} as const;

export type CloneVariant = "f32" | "f16";

/** Catalog entry for a clone-engine precision variant. */
export function styletts2Model(variant: CloneVariant) {
    return variant === "f16" ? STYLETTS2_MODEL_F16 : STYLETTS2_MODEL;
}

/** Where to fetch the StyleTTS2 cloning GGUF from (per precision variant).
 *  Honors `?localBlob=PORT`. */
export function styletts2BlobUrl(variant: CloneVariant = "f32"): string {
    const m = styletts2Model(variant);
    const port = localBlobPort();
    if (port != null) return `http://localhost:${port}/api/blob/${encodeURIComponent(m.name)}`;
    return m.url;
}

/** Whether this entry is something we'll actually run. */
export function isSupported(m: ModelEntry): boolean {
    return m.family === "gemma4" || m.family === "diffusion-gemma";
}

/** Whether this entry is the (non-autoregressive) DiffusionGemma engine, which
 *  the worker drives through `diffuserGenerate` (a denoise loop) rather than
 *  the standard token-streaming `generate`. */
export function isDiffusion(m: ModelEntry): boolean {
    return m.family === "diffusion-gemma";
}

/** Whether this entry is the Z-Image-Turbo text-to-image engine (ImageModel
 *  wasm class), driven by the Image tab via `client.image.*` rather than the
 *  chat token stream. Deliberately NOT in `isSupported()` — image gen is its
 *  own tab and must not appear in the chat model picker. */
export function isImage(m: ModelEntry): boolean {
    return m.family === "z-image";
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
    // CDN-only models (DiffusionGemma) have no local-Ollama equivalent, so the
    // localBlob devserver can't serve them. Routing through localhost would
    // either 404 or — worse, in the split-origin tunnel setup — 302 to R2 and
    // then fail CORS (a cross-origin redirect taints Origin → null, so R2
    // returns no Access-Control-Allow-Origin). A DIRECT R2 fetch carries the
    // real Origin and is CORS-allowed, so always go straight to the CDN.
    if (m.url && isDiffusion(m)) return m.url;
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
