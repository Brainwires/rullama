// OPFS-backed model store for rullama.
//
// Replaces the IndexedDB Blob cache (`cache.js`) for big GGUFs. The Blob path
// has two implicit iOS Safari caps that bite for gemma4:e2b:
//   - ~2 GiB live JS memory → WebContent process killed mid-download
//   - ~5.6 GiB combined Blob size → silent truncation
//
// OPFS sidesteps both: bytes stream straight to a disk-backed file via a
// dedicated worker that owns a `FileSystemSyncAccessHandle`. The wasm side
// reads back via async `file.slice(...).arrayBuffer()` — no JS heap residency
// on either side.
//
// API:
//   ensureModel(url, modelKey, filename, onProgress)
//       → resolves when the OPFS file is fully written (resumes if partial).
//   getReadFn(modelKey, filename)
//       → { readFn, totalBytes } for handing to Rust `Model.loadFromOpfs`.
//   listOpfs() / deleteOpfs(modelKey) for housekeeping.

const OPFS_DIR = 'rullama-models';

export async function opfsSupported() {
    return typeof navigator !== 'undefined'
        && navigator.storage
        && typeof navigator.storage.getDirectory === 'function';
}

/** Best-effort quota check; returns `{ quota, usage, persisted }` in bytes. */
export async function opfsQuota() {
    try {
        const est = (navigator.storage && navigator.storage.estimate)
            ? await navigator.storage.estimate() : {};
        const persisted = (navigator.storage && navigator.storage.persisted)
            ? await navigator.storage.persisted() : false;
        return { quota: est.quota || 0, usage: est.usage || 0, persisted };
    } catch (_) {
        return { quota: 0, usage: 0, persisted: false };
    }
}

/** Ask the browser to mark this origin's storage as persistent (raises quota
 *  and prevents eviction). Best-effort — Safari may decline silently. */
export async function requestPersistent() {
    try {
        if (navigator.storage && navigator.storage.persist) {
            return await navigator.storage.persist();
        }
    } catch (_) {}
    return false;
}

/** Wipe every entry in `rullama-models/` regardless of model key.
 *  Use before a fresh download when quota is tight. */
export async function wipeAllOpfs() {
    try {
        const root = await navigator.storage.getDirectory();
        await root.removeEntry(OPFS_DIR, { recursive: true });
        return true;
    } catch (_) {
        return false;
    }
}

async function openModelDir(modelKey, { create = false } = {}) {
    const root = await navigator.storage.getDirectory();
    const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create });
    return dlDir.getDirectoryHandle(modelKey, { create });
}

/** Return the current on-disk size of an OPFS file, or 0 if it doesn't exist. */
export async function existingSize(modelKey, filename) {
    try {
        const dir = await openModelDir(modelKey, { create: false });
        const fh = await dir.getFileHandle(filename, { create: false });
        const f = await fh.getFile();
        return f.size;
    } catch (_) {
        return 0;
    }
}

/** Probe HEAD-ish via Range: bytes=0-0 to discover the server's reported total size. */
async function discoverRemoteTotal(url) {
    const resp = await fetch(url, { headers: { Range: 'bytes=0-0' } });
    if (!resp.ok && resp.status !== 206) {
        throw new Error(`probe ${url} → ${resp.status}`);
    }
    const cr = resp.headers.get('content-range');
    if (cr) {
        const m = cr.match(/\/(\d+)\s*$/);
        if (m) return Number(m[1]);
    }
    const xt = resp.headers.get('x-total-size');
    if (xt) return Number(xt);
    const cl = resp.headers.get('content-length');
    if (cl) return Number(cl);
    throw new Error(`no Content-Range / X-Total-Size / Content-Length for ${url}`);
}

/**
 * Ensure the model file at `(modelKey, filename)` is fully present in OPFS.
 * Resumes if partially downloaded. `onProgress({bytesWritten, totalBytes})`
 * is called for each chunk.
 *
 * Spawns a dedicated worker so it can use `FileSystemSyncAccessHandle` —
 * which iOS Safari only exposes in Worker contexts.
 */
export async function ensureModel(url, modelKey, filename, onProgress) {
    if (!(await opfsSupported())) {
        throw new Error('OPFS not supported in this browser');
    }

    const remoteTotal = await discoverRemoteTotal(url);
    const haveBytes   = await existingSize(modelKey, filename);
    if (haveBytes >= remoteTotal && remoteTotal > 0) {
        onProgress?.({ bytesWritten: haveBytes, totalBytes: remoteTotal, fromCache: true });
        return { totalBytes: haveBytes, fromCache: true };
    }

    return new Promise((resolve, reject) => {
        const worker = new Worker('./opfs-writer-worker.js', { type: 'module' });
        worker.addEventListener('message', (ev) => {
            const m = ev.data;
            switch (m.type) {
                case 'progress':
                    onProgress?.({
                        bytesWritten: m.bytesWritten,
                        totalBytes:   m.totalBytes,
                        chunkBytes:   m.chunkBytes,
                        fromCache:    false,
                    });
                    break;
                case 'done':
                    worker.terminate();
                    resolve({ totalBytes: m.totalBytes, fromCache: false });
                    break;
                case 'cancelled':
                    worker.terminate();
                    reject(new Error(`download cancelled at ${m.bytesWritten} bytes`));
                    break;
                case 'error':
                    worker.terminate();
                    reject(new Error(`opfs-writer: ${m.error}`));
                    break;
            }
        });
        worker.addEventListener('error', (ev) => {
            worker.terminate();
            reject(new Error(`opfs-writer worker error: ${ev.message || ev}`));
        });
        worker.postMessage({
            type: 'start',
            modelKey, filename, url,
            offset: haveBytes,
        });
    });
}

/**
 * Build a `readFn(offset, length) -> Promise<Uint8Array>` from an OPFS file.
 *
 * Implementation: open `FileSystemFileHandle.getFile()` once, then on each
 * call slice it and read into an `ArrayBuffer`. `Blob.slice()` is lazy on
 * Safari — bytes are streamed from disk, not held in JS memory.
 *
 * The wasm side keeps this closure alive for the lifetime of the Model; the
 * `File` object holds the underlying disk handle open implicitly.
 */
export async function getReadFn(modelKey, filename) {
    const dir = await openModelDir(modelKey, { create: false });
    const fh  = await dir.getFileHandle(filename, { create: false });
    let file  = await fh.getFile();           // snapshot
    const totalBytes = file.size;

    const readFn = async (offset, length) => {
        const off = Number(offset);
        const len = Number(length);
        if (len === 0) return new Uint8Array(0);
        const slice = file.slice(off, off + len);
        const ab = await slice.arrayBuffer();
        return new Uint8Array(ab);
    };
    return { readFn, totalBytes };
}

export async function deleteOpfs(modelKey, filename) {
    try {
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        if (filename) {
            const modelDir = await dlDir.getDirectoryHandle(modelKey, { create: false });
            await modelDir.removeEntry(filename);
        } else {
            await dlDir.removeEntry(modelKey, { recursive: true });
        }
    } catch (_) { /* not present */ }
}

export async function listOpfs() {
    const out = [];
    if (!(await opfsSupported())) return out;
    try {
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: false });
        for await (const [modelKey, modelDir] of dlDir.entries()) {
            if (modelDir.kind !== 'directory') continue;
            for await (const [fname, fh] of modelDir.entries()) {
                if (fh.kind !== 'file') continue;
                const f = await fh.getFile();
                out.push({ modelKey, filename: fname, size: f.size });
            }
        }
    } catch (_) { /* dir absent */ }
    return out;
}
