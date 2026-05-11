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

/** Ensure the model file is fully present in OPFS. Resumes if partial. */
export async function ensureModel(
    url: string,
    modelKey: string,
    filename: string,
    onProgress?: (p: EnsureProgress) => void,
): Promise<EnsureResult> {
    if (!(await opfsSupported())) throw new Error("OPFS not supported in this browser");

    const remoteTotal = await discoverRemoteTotal(url);
    const have = await existingSize(modelKey, filename);
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

export interface OrphanSweepResult {
    removed:    string[];   // modelKey of each pruned folder
    freedBytes: number;     // sum of sizes across pruned folders
}

// FS Access API's directory iterator isn't in lib.dom.d.ts yet; declare
// the bit we need so we don't have to `as any` at every call site.
interface AsyncIterableDirHandle extends FileSystemDirectoryHandle {
    entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
}

/**
 * Remove cached model folders under OPFS_DIR whose name isn't in the
 * `keepModelKeys` set. Used at app start once `/api/models` loads, so
 * abandoned downloads (older quants, dropped models, broken Q4_K_M
 * variants the engine can't parse) don't sit on disk forever.
 *
 * The match key is App.tsx's `m.digest.replace(/[^A-Za-z0-9_.-]/g, "_")`;
 * pass the same set the rest of the app uses.
 */
export async function pruneOrphanedModels(keepModelKeys: Iterable<string>): Promise<OrphanSweepResult> {
    const out: OrphanSweepResult = { removed: [], freedBytes: 0 };
    const keep = new Set(keepModelKeys);
    try {
        const root  = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        for await (const [name, handle] of (dlDir as AsyncIterableDirHandle).entries()) {
            if (handle.kind !== "directory") continue;
            if (keep.has(name)) continue;
            // Sum the sizes inside before removing — purely informational
            // for the success toast.
            let size = 0;
            try {
                for await (const [, child] of (handle as AsyncIterableDirHandle).entries()) {
                    if (child.kind !== "file") continue;
                    const f = await (child as FileSystemFileHandle).getFile();
                    size += f.size;
                }
            } catch { /* size best-effort */ }
            try {
                await dlDir.removeEntry(name, { recursive: true });
                out.removed.push(name);
                out.freedBytes += size;
            } catch { /* in-use / locked — skip */ }
        }
    } catch { /* OPFS missing or directory not yet created — nothing to do */ }
    return out;
}
