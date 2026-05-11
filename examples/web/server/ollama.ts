// Resolves the local Ollama install's manifest + blob layout. This mirrors
// the Python helpers in `examples/pwa/serve.sh` (`discover_models`,
// `find_blob`) so the React app and the existing static PWA can both run
// against this server.

import fs from "node:fs";
import path from "node:path";
import os from "node:os";

const MODEL_LAYER = "application/vnd.ollama.image.model";

export interface ModelEntry {
    /** "family:tag" e.g. "gemma4:e2b" */
    name: string;
    family: string;
    tag: string;
    size: number;
    /** sha256 hex, no "sha256:" prefix */
    digest: string;
    /** Optional absolute URL — when set the client fetches the blob from
     *  here directly instead of going through /api/blob. Used for the
     *  public demo to offload bandwidth to Hugging Face's CDN. */
    url?: string;
}

/**
 * Curated public-CDN fallback. Returned when the local Ollama install
 * has no models or `RULLAMA_REMOTE_ONLY=1` forces it. Each entry points
 * at an HF-hosted GGUF — CORS + Range verified working from the browser.
 *
 * These are text-only quants. The multimodal blobs Ollama distributes
 * (text + vision + audio in one file) aren't mirrored anywhere public.
 */
export function huggingfaceModels(): ModelEntry[] {
    return [
        {
            name:   "gemma4:e2b",
            family: "gemma4",
            tag:    "e2b",
            // Q6_K, not Q4_K_M / Q4_K_S. Our v1 dequant scope is
            // F32/F16/BF16/Q4_K/Q6_K; Unsloth's Q4_K_* variants both
            // sneak Q5_K tensors in (loader rejects them). Pure Q6_K
            // is the only safe option in their repo — every weight is
            // Q6_K + the standard F16/F32 norms. 4.5 GB instead of 3.
            size:   4501719168,
            digest: "b36824f13bf9fab2910cb7b4282a4d73b13799ee4126d4ec241309ce69c0e783",
            url:    "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q6_K.gguf",
        },
    ];
}

export function ollamaRoot(): string {
    return process.env.OLLAMA_MODELS || path.join(os.homedir(), ".ollama", "models");
}

function manifestDir() { return path.join(ollamaRoot(), "manifests"); }
function blobDir()     { return path.join(ollamaRoot(), "blobs"); }

function walkFiles(root: string, out: string[] = []): string[] {
    let entries: fs.Dirent[];
    try {
        entries = fs.readdirSync(root, { withFileTypes: true });
    } catch { return out; }
    for (const e of entries) {
        const p = path.join(root, e.name);
        if (e.isDirectory())   walkFiles(p, out);
        else if (e.isFile())   out.push(p);
    }
    return out;
}

/** Discover all locally-installed Ollama models whose blob actually exists. */
export function discoverModels(): ModelEntry[] {
    const manifests = manifestDir();
    if (!fs.existsSync(manifests)) return [];
    const out: ModelEntry[] = [];
    for (const file of walkFiles(manifests)) {
        let data: any;
        try { data = JSON.parse(fs.readFileSync(file, "utf8")); }
        catch { continue; }
        if (!data || typeof data !== "object" || !Array.isArray(data.layers)) continue;
        const rel  = path.relative(manifests, file).split(path.sep);
        if (rel.length < 2) continue;
        const tag    = rel[rel.length - 1];
        const family = rel[rel.length - 2];
        for (const layer of data.layers) {
            if (layer.mediaType !== MODEL_LAYER) continue;
            let digest: string = String(layer.digest || "");
            if (digest.startsWith("sha256:")) digest = digest.slice("sha256:".length);
            const blobPath = path.join(blobDir(), `sha256-${digest}`);
            let st: fs.Stats;
            try { st = fs.statSync(blobPath); } catch { break; }
            out.push({
                name: `${family}:${tag}`,
                family,
                tag,
                size: Number(layer.size ?? st.size),
                digest,
            });
            break;
        }
    }
    out.sort((a, b) => a.name.localeCompare(b.name));
    return out;
}

/** Resolve "family:tag" to absolute blob path on disk. */
export function findBlob(nameTag: string): string | null {
    for (const m of discoverModels()) {
        if (m.name === nameTag) {
            return path.join(blobDir(), `sha256-${m.digest}`);
        }
    }
    return null;
}
