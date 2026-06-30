// OPFS-backed model storage. Replaces the IndexedDB Blob cache from the
// legacy `cache.js` — the iOS Safari ceilings on JS-heap
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

// ── Per-conversation KV-cache snapshots ────────────────────────────────
//
// Persist a completed conversation's GPU KV cache so reopening a long chat
// after a page reload restores the cache instead of re-prefilling the
// whole prompt chain. Stored in a root-level directory (NOT under
// OPFS_DIR) so it survives a "Clear cached models" action, same rationale
// as the inflight snapshot above.
//
//   rullama-conv-kv/{convId}.kv    ← RLCV envelope (residentIds + RLMS KV)
//   rullama-conv-kv/{convId}.json  ← sidecar: digest + LRU metadata
//
// The `.kv` payload is opaque here — the core worker builds/parses it via
// saveConvKv/restoreConvKv. The sidecar lets JS validate model identity
// and run LRU pruning without touching the (large) payload.

const CONV_KV_DIR = "rullama-conv-kv";
// Persisted SYSTEM-PROMPT pre-warm, one file per model digest. The system
// block rarely changes and doesn't change across reloads, so its warmed KV
// only ever needs computing once per (model, system-prompt) — persist it
// here and restore on load instead of re-prefilling every reload.
const SYSWARM_DIR = "rullama-syswarm";

export interface ConvSnapshotMeta {
    modelDigest: string;
    version: number;
    tokenCount: number;
    byteSize: number;
    updatedAt: number;
}

export interface SysWarmMeta {
    modelDigest: string;
    /** Identity of the warmed system block (digest + adapter + sysContent).
     *  Restore only when it matches the current config exactly. */
    sig: string;
    version: number;
    tokenCount: number;
    byteSize: number;
    updatedAt: number;
}

async function snapDir(dirName: string, create: boolean): Promise<FileSystemDirectoryHandle> {
    const root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle(dirName, { create });
}

/** Write a `{key}.kv` payload + `{key}.json` sidecar. Uses the worker sync
 *  handle when available, else the async writable stream (same fallback as
 *  `writeInflightState`). Sidecar is written last so we never claim a valid
 *  snapshot exists before its payload landed. */
async function writeSnapshotPair(
    dirName: string, key: string, bytes: Uint8Array, meta: object,
): Promise<void> {
    const dir = await snapDir(dirName, true);
    const fh = await dir.getFileHandle(`${key}.kv`, { create: true });
    const fhAny = fh as unknown as {
        createSyncAccessHandle?(): Promise<FileSystemSyncAccessHandle>;
        createWritable(): Promise<FileSystemWritableFileStream>;
    };
    if (typeof fhAny.createSyncAccessHandle === "function") {
        const h = await fhAny.createSyncAccessHandle();
        try { h.truncate(0); h.write(bytes, { at: 0 }); h.flush(); }
        finally { h.close(); }
    } else {
        const w = await fhAny.createWritable();
        await w.truncate(0);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await w.write(bytes as any);
        await w.close();
    }
    const metaFh = await dir.getFileHandle(`${key}.json`, { create: true });
    const mw = await metaFh.createWritable();
    await mw.truncate(0);
    await mw.write(JSON.stringify(meta));
    await mw.close();
}

async function readSnapshotPair<M>(
    dirName: string, key: string,
): Promise<{ bytes: Uint8Array; meta: M } | null> {
    try {
        const dir = await snapDir(dirName, false);
        const metaFh = await dir.getFileHandle(`${key}.json`, { create: false });
        const meta = JSON.parse(await (await metaFh.getFile()).text()) as M;
        const fh = await dir.getFileHandle(`${key}.kv`, { create: false });
        const bytes = new Uint8Array(await (await fh.getFile()).arrayBuffer());
        return { bytes, meta };
    } catch {
        return null;
    }
}

async function deleteSnapshotPair(dirName: string, key: string): Promise<void> {
    try {
        const dir = await snapDir(dirName, false);
        try { await dir.removeEntry(`${key}.kv`); } catch { /* */ }
        try { await dir.removeEntry(`${key}.json`); } catch { /* */ }
    } catch { /* */ }
}

/** Filesystem-safe key from an arbitrary id (model digest, conv id, …). */
function snapKey(id: string): string {
    return id.replace(/[^A-Za-z0-9_.-]/g, "_");
}

// ── Per-conversation snapshots ─────────────────────────────────────────

export const writeConvSnapshot = (convId: string, bytes: Uint8Array, meta: ConvSnapshotMeta) =>
    writeSnapshotPair(CONV_KV_DIR, snapKey(convId), bytes, meta);

export const readConvSnapshot = (convId: string) =>
    readSnapshotPair<ConvSnapshotMeta>(CONV_KV_DIR, snapKey(convId));

export const deleteConvSnapshot = (convId: string) =>
    deleteSnapshotPair(CONV_KV_DIR, snapKey(convId));

/** Enumerate all conversation snapshots via their sidecars — for LRU
 *  pruning. Skips entries whose sidecar can't be read. */
export async function listConvSnapshots(): Promise<{ convId: string; meta: ConvSnapshotMeta }[]> {
    const out: { convId: string; meta: ConvSnapshotMeta }[] = [];
    try {
        const dir = await snapDir(CONV_KV_DIR, false);
        const entries = (dir as unknown as {
            entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
        }).entries();
        for await (const [name, handle] of entries) {
            if (!name.endsWith(".json") || handle.kind !== "file") continue;
            const convId = name.slice(0, -".json".length);
            try {
                const fh = handle as FileSystemFileHandle;
                const meta = JSON.parse(await (await fh.getFile()).text()) as ConvSnapshotMeta;
                out.push({ convId, meta });
            } catch { /* skip unreadable sidecar */ }
        }
    } catch { /* dir absent → empty */ }
    return out;
}

// ── Persisted system-prompt pre-warm (one per model digest) ────────────

export const writeSysWarmSnapshot = (digest: string, bytes: Uint8Array, meta: SysWarmMeta) =>
    writeSnapshotPair(SYSWARM_DIR, snapKey(digest), bytes, meta);

export const readSysWarmSnapshot = (digest: string) =>
    readSnapshotPair<SysWarmMeta>(SYSWARM_DIR, snapKey(digest));

export const deleteSysWarmSnapshot = (digest: string) =>
    deleteSnapshotPair(SYSWARM_DIR, snapKey(digest));

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
    let root: FileSystemDirectoryHandle;
    let dlDir: FileSystemDirectoryHandle;
    let md: FileSystemDirectoryHandle;
    try {
        root  = await navigator.storage.getDirectory();
        dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        md    = await dlDir.getDirectoryHandle(modelKey, { create: false });
    } catch {
        // OPFS dir or model dir absent — genuinely nothing cached. No-op.
        return false;
    }
    try {
        await md.removeEntry(filename);
    } catch (e) {
        const name = (e as DOMException)?.name;
        if (name === "NotFoundError") return false; // already gone, real no-op
        // NoModificationAllowedError / InvalidStateError ⇒ an OPFS sync access
        // handle is still open on this file (the model worker hasn't released
        // it), so removeEntry can't run and the bytes WON'T be reclaimed.
        // Surface it instead of silently pretending success — this is exactly
        // the "leak in delete" symptom (file looks gone, disk stays full).
        throw new Error(
            `OPFS delete blocked for ${modelKey}/${filename} (${name ?? "unknown"}): ` +
            `a sync access handle is still open — the model worker may not have ` +
            `released it. Eject the model (or reload) and retry.`,
        );
    }
    try { await dlDir.removeEntry(modelKey, { recursive: true }); } catch { /* dir not empty / locked — fine */ }
    return true;
}

