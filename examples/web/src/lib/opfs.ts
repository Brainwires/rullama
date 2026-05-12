// OPFS-backed model storage. Replaces the IndexedDB Blob cache from the
// legacy `examples/pwa/cache.js` — the iOS Safari ceilings on JS-heap
// (~2 GB) and combined Blob size (~5.6 GB) both bite a 7 GB GGUF straight
// in the face, so the writer worker streams via `FileSystemSyncAccessHandle`
// instead.

const OPFS_DIR = "rullama-models";

// Type-only import of the worker so Vite emits it as a separate chunk.
import OpfsWriterWorker from "@/workers/opfs-writer-worker?worker";

export async function opfsSupported(): Promise<boolean> {
    return (
        typeof navigator !== "undefined" &&
        !!navigator.storage &&
        typeof navigator.storage.getDirectory === "function"
    );
}

export interface QuotaInfo {
    quota: number;
    usage: number;
    persisted: boolean;
}

export async function opfsQuota(): Promise<QuotaInfo> {
    try {
        const est = navigator.storage?.estimate ? await navigator.storage.estimate() : {};
        const persisted = navigator.storage?.persisted ? await navigator.storage.persisted() : false;
        return { quota: est.quota ?? 0, usage: est.usage ?? 0, persisted };
    } catch { return { quota: 0, usage: 0, persisted: false }; }
}

export async function requestPersistent(): Promise<boolean> {
    try {
        return navigator.storage?.persist ? await navigator.storage.persist() : false;
    } catch { return false; }
}

// GGUF magic bytes: "GGUF" (0x47 0x47 0x55 0x46).
const GGUF_MAGIC = [0x47, 0x47, 0x55, 0x46];

async function magicLooksValid(fh: FileSystemFileHandle): Promise<boolean> {
    try {
        const head = await (await fh.getFile()).slice(0, 4).arrayBuffer();
        const b    = new Uint8Array(head);
        return b.length === 4 && b.every((v, i) => v === GGUF_MAGIC[i]);
    } catch { return false; }
}

async function removeFile(modelKey: string, filename: string): Promise<void> {
    try {
        const root  = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        const md    = await dlDir.getDirectoryHandle(modelKey, { create: false });
        await md.removeEntry(filename);
    } catch { /* */ }
}

/**
 * Returns the size of the cached OPFS file if its first 4 bytes match the
 * GGUF magic, otherwise 0. Files that fail the magic check are deleted so
 * the next download starts clean — iOS Safari Jetsam can kill the writer
 * worker between `truncate()` and the next 64 MiB `flush()`, leaving the
 * file at its truncated size with a zero-byte prefix.
 */
export async function existingSize(modelKey: string, filename: string): Promise<number> {
    try {
        const root  = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        const md    = await dlDir.getDirectoryHandle(modelKey, { create: false });
        const fh    = await md.getFileHandle(filename, { create: false });
        const f     = await fh.getFile();
        if (f.size < 4) return 0;
        if (!(await magicLooksValid(fh))) {
            await removeFile(modelKey, filename);
            return 0;
        }
        return f.size;
    } catch { return 0; }
}

async function discoverRemoteTotal(url: string): Promise<number> {
    const resp = await fetch(url, { headers: { Range: "bytes=0-0" } });
    if (!resp.ok && resp.status !== 206) throw new Error(`probe ${url} → ${resp.status}`);
    const cr = resp.headers.get("content-range");
    if (cr) {
        const m = /\/(\d+)\s*$/.exec(cr);
        if (m) return Number(m[1]);
    }
    const xt = resp.headers.get("x-total-size");
    if (xt) return Number(xt);
    const cl = resp.headers.get("content-length");
    if (cl) return Number(cl);
    throw new Error(`no Content-Range / X-Total-Size / Content-Length for ${url}`);
}

export interface EnsureProgress {
    bytesWritten: number;
    totalBytes: number;
    chunkBytes?: number;
    fromCache: boolean;
}
export interface EnsureResult { totalBytes: number; fromCache: boolean; }

/**
 * Ensure the model file is fully present in OPFS. Resumes if partial.
 *
 * `expectedSize` is the catalog-declared size (`m.size`). When the local
 * OPFS file already meets that size, we short-circuit and return without
 * any network call — this is what lets a cached model load while the
 * browser is offline. The remote probe + writer worker are only reached
 * when we genuinely need bytes from the network.
 */
export async function ensureModel(
    url: string,
    modelKey: string,
    filename: string,
    expectedSize: number,
    onProgress?: (p: EnsureProgress) => void,
): Promise<EnsureResult> {
    if (!(await opfsSupported())) throw new Error("OPFS not supported in this browser");

    // Fast path: the file is already fully here. No network, no probe,
    // no writer-worker boot. Works offline.
    const have = await existingSize(modelKey, filename);
    if (expectedSize > 0 && have >= expectedSize) {
        onProgress?.({ bytesWritten: have, totalBytes: expectedSize, fromCache: true });
        return { totalBytes: have, fromCache: true };
    }

    // Otherwise we genuinely need bytes — probe the remote for its
    // current size. This is the call that fails offline; the load
    // surfaces a "Failed to fetch" error which the App.tsx error path
    // turns into a toast.
    const remoteTotal = await discoverRemoteTotal(url);
    if (have >= remoteTotal && remoteTotal > 0) {
        onProgress?.({ bytesWritten: have, totalBytes: remoteTotal, fromCache: true });
        return { totalBytes: have, fromCache: true };
    }

    return new Promise((resolve, reject) => {
        const worker = new OpfsWriterWorker();
        worker.addEventListener("message", (ev: MessageEvent) => {
            const m = ev.data;
            switch (m?.type) {
                case "progress":
                    onProgress?.({
                        bytesWritten: m.bytesWritten,
                        totalBytes:   m.totalBytes,
                        chunkBytes:   m.chunkBytes,
                        fromCache:    false,
                    });
                    break;
                case "done":
                    worker.terminate();
                    resolve({ totalBytes: m.totalBytes, fromCache: false });
                    break;
                case "cancelled":
                    worker.terminate();
                    reject(new Error(`download cancelled at ${m.bytesWritten} bytes`));
                    break;
                case "error":
                    worker.terminate();
                    reject(new Error(`opfs-writer: ${m.error}`));
                    break;
            }
        });
        worker.addEventListener("error", (ev: ErrorEvent) => {
            worker.terminate();
            reject(new Error(`opfs-writer worker error: ${ev.message || ev}`));
        });
        worker.postMessage({
            type: "start",
            modelKey, filename, url,
            offset: have,
        });
    });
}

export async function wipeAllOpfs(): Promise<boolean> {
    try {
        const root = await navigator.storage.getDirectory();
        await root.removeEntry(OPFS_DIR, { recursive: true });
        return true;
    } catch { return false; }
}

/**
 * Remove a single cached model. Best-effort — also prunes the model
 * directory if empty. Returns true if a file was actually removed.
 */
export async function wipeModel(modelKey: string, filename: string): Promise<boolean> {
    try {
        const root  = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        const md    = await dlDir.getDirectoryHandle(modelKey, { create: false });
        await md.removeEntry(filename);
        try { await dlDir.removeEntry(modelKey, { recursive: true }); } catch { /* dir not empty / locked */ }
        return true;
    } catch { return false; }
}

