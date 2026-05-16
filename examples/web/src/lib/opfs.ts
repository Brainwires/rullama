// OPFS-backed model storage. Replaces the IndexedDB Blob cache from the
// legacy `examples/pwa/cache.js` — the iOS Safari ceilings on JS-heap
// (~2 GB) and combined Blob size (~5.6 GB) both bite a 7 GB GGUF straight
// in the face, so the actual streaming write lives inside the inference
// SharedWorker (M17) — one download per origin, progress broadcast to
// every connected tab.

import { getClient } from "@/lib/inference";

const OPFS_DIR = "rullama-models";

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

export interface EnsureProgress {
    bytesWritten: number;
    totalBytes: number;
    chunkBytes?: number;
    fromCache: boolean;
}
export interface EnsureResult { totalBytes: number; fromCache: boolean; }

/**
 * Ensure the model file is fully present in OPFS. The actual streaming
 * write happens inside the inference SharedWorker (so a single download
 * serves every tab and we never race the OPFS write handle).
 *
 * `expectedSize` is the catalog-declared size (`m.size`). When the local
 * OPFS file already meets that size, the worker short-circuits with no
 * network call — works offline.
 *
 * Progress is delivered via the worker's `notify: downloadProgress`
 * broadcast, scoped here to the matching modelKey/filename and surfaced
 * through the optional `onProgress` callback.
 */
export async function ensureModel(
    url: string,
    modelKey: string,
    filename: string,
    expectedSize: number,
    onProgress?: (p: EnsureProgress) => void,
): Promise<EnsureResult> {
    if (!(await opfsSupported())) throw new Error("OPFS not supported in this browser");

    const client = getClient();
    const unsubscribers: Array<() => void> = [];

    if (onProgress) {
        unsubscribers.push(client.subscribe("downloadProgress", (p) => {
            if (p.modelKey === modelKey && p.filename === filename) {
                onProgress({
                    bytesWritten: Number(p.bytesWritten),
                    totalBytes:   Number(p.totalBytes),
                    chunkBytes:   Number(p.chunkBytes),
                    fromCache:    false,
                });
            }
        }));
        unsubscribers.push(client.subscribe("downloadDone", (p) => {
            if (p.modelKey === modelKey && p.filename === filename) {
                onProgress({
                    bytesWritten: Number(p.totalBytes),
                    totalBytes:   Number(p.totalBytes),
                    fromCache:    !!p.fromCache,
                });
            }
        }));
    }

    try {
        return await client.ensureModel({ url, modelKey, filename, expectedSize });
    } finally {
        for (const u of unsubscribers) u();
    }
}

// ── Suspend/resume snapshot storage ────────────────────────────────────
//
// One file at the OPFS root (NOT under OPFS_DIR — that subtree gets
// wiped by `wipeAllOpfs` and we want resume to survive a "Clear cached
// models" action). On a clean generation finish / EOS the file is
// deleted; on suspension we write it; on boot we read it.

const INFLIGHT_STATE_FILENAME = "rullama-inflight-gen-state.bin";

/**
 * Write the wasm-side `saveKvState()` bytes to OPFS. ~100 ms for a 100 MB
 * KV at 1 GB/s flash. Sync handle is used so the write completes before
 * the `visibilitychange` callback returns control to the browser (and
 * thus before iOS can suspend us mid-write).
 */
export async function writeInflightState(bytes: Uint8Array): Promise<void> {
    const root = await navigator.storage.getDirectory();
    const fh = await root.getFileHandle(INFLIGHT_STATE_FILENAME, { create: true });
    // FileSystemSyncAccessHandle is only available inside Workers in
    // some browsers; fall back to the async writable stream otherwise.
    const fhAny = fh as unknown as {
        createSyncAccessHandle?(): Promise<FileSystemSyncAccessHandle>;
        createWritable(): Promise<FileSystemWritableFileStream>;
    };
    if (typeof fhAny.createSyncAccessHandle === "function") {
        const h = await fhAny.createSyncAccessHandle();
        try {
            h.truncate(0);
            h.write(bytes, { at: 0 });
            h.flush();
        } finally {
            h.close();
        }
        return;
    }
    const w = await fhAny.createWritable();
    await w.truncate(0);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await w.write(bytes as any);
    await w.close();
}

/**
 * Read back the inflight-state file, or `null` if not present.
 */
export async function readInflightState(): Promise<Uint8Array | null> {
    try {
        const root = await navigator.storage.getDirectory();
        const fh = await root.getFileHandle(INFLIGHT_STATE_FILENAME, { create: false });
        const file = await fh.getFile();
        const ab = await file.arrayBuffer();
        return new Uint8Array(ab);
    } catch {
        return null;
    }
}

/**
 * Remove the inflight-state file. Called on clean generation completion
 * (EOS / maxTokens / explicit user cancel) so a future boot doesn't try
 * to resume a finished generation.
 */
export async function clearInflightState(): Promise<void> {
    try {
        const root = await navigator.storage.getDirectory();
        await root.removeEntry(INFLIGHT_STATE_FILENAME);
    } catch { /* */ }
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

