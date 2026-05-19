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

/** Three-state outcome for the first-4-bytes check:
 *
 *   - `good`          — bytes match the GGUF magic. File is real.
 *   - `bad_magic`     — bytes were *readable* but don't match. Almost certainly
 *                       a Jetsam-killed write that left a zero-byte prefix on
 *                       a truncated file; safe to delete and re-download.
 *   - `read_failed`   — `getFile()` / `slice().arrayBuffer()` threw. This is
 *                       NOT proof of corruption. The common transient cause
 *                       on iOS Safari is an old core worker's
 *                       `FileSystemSyncAccessHandle` not yet GC'd after a PWA
 *                       update reload — the main-thread `getFile()` races the
 *                       lingering exclusive lock and throws. The OLD code
 *                       treated this as "bad magic" and deleted the file,
 *                       which is why every deploy could nuke a perfectly good
 *                       7 GB GGUF and trigger a fresh download.
 */
type MagicCheck = "good" | "bad_magic" | "read_failed";

async function checkMagic(fh: FileSystemFileHandle): Promise<MagicCheck> {
    let head: ArrayBuffer;
    try {
        head = await (await fh.getFile()).slice(0, 4).arrayBuffer();
    } catch { return "read_failed"; }
    const b = new Uint8Array(head);
    if (b.length === 4 && b.every((v, i) => v === GGUF_MAGIC[i])) return "good";
    return "bad_magic";
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
 * Returns the size of the cached OPFS file if it looks like a real GGUF
 * (matches the 4-byte magic), 0 otherwise.
 *
 * Auto-delete policy: only when the magic check actually **read** the file
 * and the bytes didn't match (Jetsam-truncated case). On a read failure
 * (likely a transient sync-handle race after a PWA update reload), we
 * return 0 but leave the file alone — the next boot's check will either
 * succeed once the old handle is GC'd, or genuinely confirm corruption.
 * The OLD blind delete-on-any-failure is what made OPFS-after-update look
 * unrecoverable.
 */
export async function existingSize(modelKey: string, filename: string): Promise<number> {
    try {
        const root  = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        const md    = await dlDir.getDirectoryHandle(modelKey, { create: false });
        const fh    = await md.getFileHandle(filename, { create: false });
        const f     = await fh.getFile();
        if (f.size < 4) return 0;
        const magic = await checkMagic(fh);
        if (magic === "good")       return f.size;
        if (magic === "bad_magic") {
            console.warn("[opfs] GGUF magic mismatch — deleting truncated file", modelKey, filename);
            await removeFile(modelKey, filename);
            return 0;
        }
        // read_failed — almost certainly a transient sync-handle race on
        // iOS Safari (previous worker's exclusive lock hasn't GC'd yet).
        // Return f.size so callers that compare "cachedBytes vs expected
        // size" can still see the file as potentially-complete and avoid
        // auto-triggering a redownload. Downstream load via the worker's
        // openSyncReadFn will succeed once the lock releases (it has its
        // own retry).
        console.warn("[opfs] first-bytes read failed (likely sync-handle race) — preserving file at size=", f.size, modelKey, filename);
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

