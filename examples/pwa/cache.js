// IndexedDB blob cache for rullama, keyed by sha256 digest.
//
// Layout:
//   DB:    rullama-models  (version 1)
//   Store: blobs            keyPath: digest      (sha256 hex string)
//          fields: { digest, name, bytes (Blob), size, savedAt }
//
// Usage:
//   import { openCache, getCachedBlob, putCachedBlob, listCached, deleteCached } from "./cache.js";
//
// We deliberately store Blobs (not ArrayBuffers): browsers back Blobs with on-disk
// storage rather than RAM, which is essential for 7 GB models that exceed wasm32's
// linear-memory cap. The Model.loadFromUrl path doesn't read the Blob into JS — it
// streams it via fetch() with Range requests against a blob: URL.

export const DB_NAME = "rullama-models";
export const STORE = "blobs";
export const VERSION = 1;

let _dbPromise = null;

export function openCache() {
    if (_dbPromise) return _dbPromise;
    _dbPromise = new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, VERSION);
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(STORE)) {
                db.createObjectStore(STORE, { keyPath: "digest" });
            }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror   = () => reject(req.error);
    });
    return _dbPromise;
}

function txStore(db, mode) {
    return db.transaction(STORE, mode).objectStore(STORE);
}

/** Returns a Blob for `digest`, or null if not cached. */
export async function getCachedBlob(digest) {
    const db = await openCache();
    return new Promise((resolve, reject) => {
        const req = txStore(db, "readonly").get(digest);
        req.onsuccess = () => resolve(req.result?.bytes ?? null);
        req.onerror   = () => reject(req.error);
    });
}

/** Store the given Blob. Fails fast on QuotaExceededError so the caller can fall back. */
export async function putCachedBlob(digest, name, blob) {
    const db = await openCache();
    return new Promise((resolve, reject) => {
        const req = txStore(db, "readwrite").put({
            digest, name,
            bytes: blob,
            size: blob.size,
            savedAt: Date.now(),
        });
        req.onsuccess = () => resolve();
        req.onerror   = () => reject(req.error);
    });
}

export async function listCached() {
    const db = await openCache();
    return new Promise((resolve, reject) => {
        const out = [];
        const req = txStore(db, "readonly").openCursor();
        req.onsuccess = () => {
            const cur = req.result;
            if (cur) {
                const v = cur.value;
                out.push({ digest: v.digest, name: v.name, size: v.size, savedAt: v.savedAt });
                cur.continue();
            } else {
                resolve(out);
            }
        };
        req.onerror = () => reject(req.error);
    });
}

export async function deleteCached(digest) {
    const db = await openCache();
    return new Promise((resolve, reject) => {
        const req = txStore(db, "readwrite").delete(digest);
        req.onsuccess = () => resolve();
        req.onerror   = () => reject(req.error);
    });
}

/** Stream a URL into IndexedDB while reporting progress. Returns { blob, fromCache }.
 * If the digest is already cached, returns the existing Blob without re-fetching.
 * If the cache write fails (typically QuotaExceededError), the function still returns
 * the in-memory Blob so the caller can keep going without offline persistence.
 */
export async function fetchAndCache(url, digest, name, onProgress) {
    const cached = await getCachedBlob(digest).catch(() => null);
    if (cached) return { blob: cached, fromCache: true };

    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url} → ${res.status}`);
    const total = parseInt(res.headers.get("Content-Length") || "0", 10);

    // Stream chunks into intermediate Blobs every ~32 MB so the runtime can
    // spill them to disk-backed storage. iOS Safari's WebContent process gets
    // killed around ~2 GB of live JS memory; for a 7 GB model we must NOT
    // keep every fetched chunk pinned in an Array<Uint8Array> until the end.
    // Concatenating an Array<Blob> with `new Blob(...)` doesn't copy bytes
    // (it stores references), so the final Blob is effectively free.
    const FLUSH_BYTES = 32 * 1024 * 1024; // 32 MiB per spill
    const reader = res.body.getReader();
    const blobParts = [];          // Blob chunks, disk-spillable
    let pending = [];              // Uint8Array fragments not yet wrapped
    let pendingSize = 0;
    let received = 0;
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        pending.push(value);
        pendingSize += value.byteLength;
        received += value.byteLength;
        if (pendingSize >= FLUSH_BYTES) {
            blobParts.push(new Blob(pending));
            pending = [];
            pendingSize = 0;
        }
        onProgress?.(received, total);
    }
    if (pendingSize > 0) {
        blobParts.push(new Blob(pending));
        pending = [];
        pendingSize = 0;
    }
    const blob = new Blob(blobParts);

    try {
        await putCachedBlob(digest, name, blob);
    } catch (e) {
        console.warn("rullama cache: failed to persist", name, e);
    }
    return { blob, fromCache: false };
}
