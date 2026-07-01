/// Resumable OPFS model downloader — shared by the inference and voice-cloning workers.
///
/// This is the screen-lock-surviving download path: it streams a (possibly multi-GB) model
/// straight into an OPFS file via a `FileSystemSyncAccessHandle` (never into the JS/wasm heap),
/// resumes a partial file with `Range: bytes=<have>-`, and retries a broken socket (iOS Safari
/// severs connections during screen lock) without restarting from zero. Originally lived inline
/// in inference-core-worker.ts; extracted so the clone engine gets the exact same robustness
/// instead of a naive one-shot `fetch` (which left partial files and re-downloaded forever).
///
/// Layout: OPFS `root / <opfsDir> / <modelKey> / <filename>`.

const FLUSH_INTERVAL = 64 * 1024 * 1024;

export interface DownloadCallbacks {
    /** Per-chunk progress. */
    onProgress?(p: { bytesWritten: number; totalBytes: number; chunkBytes: number }): void;
    /** Waiting for a previous worker to release the OPFS exclusive lock. */
    onWaiting?(p: { label: string; attempt: number; elapsedMs: number; nextDelayMs: number }): void;
    /** A network stream broke and we're about to resume via Range. */
    onRetrying?(p: { bytesWritten: number; totalBytes: number; nextDelayMs: number; attempt: number; maxAttempts: number }): void;
    log?(msg: string): void;
}

/** Open a `FileSystemSyncAccessHandle`, retrying while a previous worker still holds the
 *  exclusive lock (iOS Safari can take a few seconds to GC an orphaned handle after a
 *  screen-lock-induced Jetsam kill — without the retry the resume fails on first attempt even
 *  though the data on disk is intact). */
export async function createSyncAccessHandleWithRetry(
    fh: FileSystemFileHandle,
    label: string,
    opts?: { budgetMs?: number; onWaiting?: DownloadCallbacks["onWaiting"]; log?: (m: string) => void },
): Promise<FileSystemSyncAccessHandle> {
    const budget = opts?.budgetMs ?? 15_000;
    const start = Date.now();
    let attempt = 0;
    for (;;) {
        try {
            return await fh.createSyncAccessHandle();
        } catch (e) {
            attempt++;
            const elapsed = Date.now() - start;
            if (elapsed >= budget) {
                throw new Error(
                    `${label}: OPFS sync handle locked by a previous worker (${attempt} tries, ${elapsed}ms). ` +
                    `The data is intact — force-quit the tab/app and reopen to release the lock. ` +
                    `Underlying: ${(e as Error)?.message ?? e}`,
                );
            }
            const delay = Math.min(1500, 100 * 2 ** (attempt - 1));
            opts?.onWaiting?.({ label, attempt, elapsedMs: elapsed, nextDelayMs: delay });
            opts?.log?.(`opfs: ${label} still locked (attempt ${attempt}, ${elapsed}ms) — retry in ${delay}ms`);
            await new Promise((r) => setTimeout(r, delay));
        }
    }
}

/** Size of an existing OPFS model file, or 0 if absent/corrupt. A first-bytes GGUF magic check
 *  guards against a Jetsam-truncated zero prefix; a *read failure* (transient sync-handle race)
 *  preserves the file by returning its size rather than triggering a redownload-from-zero. */
export async function existingOpfsSize(opfsDir: string, modelKey: string, filename: string): Promise<number> {
    try {
        const root = await navigator.storage.getDirectory();
        const dlDir = await root.getDirectoryHandle(opfsDir, { create: false });
        const md = await dlDir.getDirectoryHandle(modelKey, { create: false });
        const fh = await md.getFileHandle(filename, { create: false });
        const f = await fh.getFile();
        if (f.size < 4) return 0;
        let magicGood = false;
        let magicReadable = false;
        try {
            const head = new Uint8Array(await f.slice(0, 4).arrayBuffer());
            magicReadable = head.length === 4;
            magicGood = magicReadable && head[0] === 0x47 && head[1] === 0x47 && head[2] === 0x55 && head[3] === 0x46;
        } catch { /* sync-handle race — magicReadable stays false */ }
        if (magicGood) return f.size;
        if (magicReadable) return 0; // read succeeded but bytes are wrong: real corruption
        return f.size; // read failed — preserve the file
    } catch {
        return 0;
    }
}

/**
 * Ensure `root/<opfsDir>/<modelKey>/<filename>` holds the full model, downloading/resuming as
 * needed. Returns the final size. Safe to call repeatedly: a complete cached file short-circuits
 * (one download, ever), a partial file resumes from its current length.
 */
export async function downloadToOpfs(
    args: { opfsDir: string; url: string; modelKey: string; filename: string; expectedSize: number } & DownloadCallbacks,
): Promise<{ totalBytes: number; fromCache: boolean }> {
    const { opfsDir, url, modelKey, filename, expectedSize, onProgress, onWaiting, onRetrying, log } = args;

    const have = await existingOpfsSize(opfsDir, modelKey, filename);
    if (have > 0 && expectedSize > 0 && have >= expectedSize) {
        return { totalBytes: have, fromCache: true }; // already complete — never re-download
    }

    let currentOffset = have;
    const root = await navigator.storage.getDirectory();
    const dlDir = await root.getDirectoryHandle(opfsDir, { create: true });
    const modelDir = await dlDir.getDirectoryHandle(modelKey, { create: true });
    const fileHandle = await modelDir.getFileHandle(filename, { create: true });

    let writeHandle: FileSystemSyncAccessHandle | null = null;
    let bytesSinceFlush = 0;
    try {
        writeHandle = await createSyncAccessHandleWithRetry(fileHandle, `${modelKey}/${filename} (write)`, { onWaiting, log });
        writeHandle.truncate(currentOffset);
        writeHandle.flush();

        let totalBytes = 0;
        const MAX_FETCH_RETRIES = 5;
        let fetchAttempt = 0;

        // Outer fetch-retry loop: a dropped socket (screen lock) refetches with Range and
        // continues from currentOffset instead of failing the whole download.
        for (;;) {
            const headers: Record<string, string> = {};
            if (currentOffset > 0) headers["Range"] = `bytes=${currentOffset}-`;
            const resp = await fetch(url, { headers });

            if (resp.status === 416) {
                const size = writeHandle.getSize();
                writeHandle.close();
                writeHandle = null;
                return { totalBytes: size, fromCache: false };
            }
            if (!resp.ok && resp.status !== 206) throw new Error(`fetch failed (${resp.status})`);
            if (resp.status === 200 && currentOffset > 0) {
                writeHandle.truncate(0); // server ignored Range — restart
                currentOffset = 0;
            }

            const cr = resp.headers.get("content-range");
            const contentLength = Number(resp.headers.get("content-length") || "0") || 0;
            totalBytes = cr?.match(/\/(\d+)\s*$/) ? Number(cr.match(/\/(\d+)\s*$/)![1]) : currentOffset + contentLength;

            if (!resp.body) throw new Error("no response body");
            const reader = resp.body.getReader();
            try {
                for (;;) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    if (!value) continue;
                    const written = writeHandle.write(value, { at: currentOffset });
                    currentOffset += written;
                    bytesSinceFlush += written;
                    if (bytesSinceFlush >= FLUSH_INTERVAL) {
                        writeHandle.flush();
                        bytesSinceFlush = 0;
                    }
                    onProgress?.({ bytesWritten: currentOffset, totalBytes, chunkBytes: written });
                }
                break; // drained cleanly
            } catch (readErr) {
                try { writeHandle.flush(); } catch { /* */ }
                bytesSinceFlush = 0;
                fetchAttempt += 1;
                if (fetchAttempt > MAX_FETCH_RETRIES) throw readErr;
                const delay = Math.min(5_000, 500 * 2 ** (fetchAttempt - 1));
                log?.(`download: ${modelKey}/${filename} stream broke at ${currentOffset}/${totalBytes} — retry ${fetchAttempt}/${MAX_FETCH_RETRIES} in ${delay}ms`);
                onRetrying?.({ bytesWritten: currentOffset, totalBytes, nextDelayMs: delay, attempt: fetchAttempt, maxAttempts: MAX_FETCH_RETRIES });
                await new Promise((r) => setTimeout(r, delay));
            }
        }

        writeHandle.flush();
        writeHandle.close();
        writeHandle = null;
        return { totalBytes: currentOffset, fromCache: false };
    } catch (err) {
        if (writeHandle) {
            try { writeHandle.flush(); } catch { /* */ }
            try { writeHandle.close(); } catch { /* */ }
        }
        throw new Error((err as Error)?.message ?? String(err));
    }
}

/** Open a range-read callback over a cached OPFS model file (for the wasm streaming loaders).
 *  The returned `close()` releases the exclusive lock so a future worker can reopen the file. */
export async function openOpfsReadFn(
    opfsDir: string,
    modelKey: string,
    filename: string,
): Promise<{ readFn: (offset: number, length: number) => Uint8Array; size: number; close: () => void }> {
    const root = await navigator.storage.getDirectory();
    const dlDir = await root.getDirectoryHandle(opfsDir, { create: false });
    const modelDir = await dlDir.getDirectoryHandle(modelKey, { create: false });
    const fh = await modelDir.getFileHandle(filename, { create: false });
    const handle = await createSyncAccessHandleWithRetry(fh, `${modelKey}/${filename} (read)`);
    const size = handle.getSize();
    if (size === 0) {
        try { handle.close(); } catch { /* */ }
        throw new Error(`OPFS file ${modelKey}/${filename} is empty`);
    }
    const readFn = (offset: number, length: number): Uint8Array => {
        const buf = new Uint8Array(length);
        handle.read(buf, { at: offset });
        return buf;
    };
    return { readFn, size, close: () => { try { handle.close(); } catch { /* */ } } };
}
