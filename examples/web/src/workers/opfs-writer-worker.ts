// rullama OPFS writer — Dedicated Worker.
//
// Streams a model URL to an OPFS file via `FileSystemSyncAccessHandle.write()`.
// Bytes go network → sync handle → disk without ever pinning a Blob in the
// JS heap. Sidesteps iOS Safari's ~2 GiB live-JS-heap kill cap and the
// ~5.6 GiB single-Blob ceiling — both bite a 7 GB GGUF straight in the face
// on a phone.
//
// `createSyncAccessHandle()` is Worker-only on iOS Safari. That's why this
// is a dedicated worker rather than a main-thread fetch.

const OPFS_DIR = "rullama-models";
const FLUSH_INTERVAL = 64 * 1024 * 1024;   // flush every 64 MiB

interface StartMsg {
    type:     "start";
    modelKey: string;
    filename: string;
    url:      string;
    offset?:  number;
}
interface CancelMsg { type: "cancel"; }

type OutMsg =
    | { type: "progress";  bytesWritten: number; totalBytes: number; chunkBytes: number }
    | { type: "done";      totalBytes: number }
    | { type: "error";     error: string }
    | { type: "cancelled"; bytesWritten: number };

let cancelled = false;
let activeReader: ReadableStreamDefaultReader<Uint8Array> | null = null;

self.addEventListener("message", (ev: MessageEvent<StartMsg | CancelMsg>) => {
    const msg = ev.data;
    if (!msg || typeof msg !== "object") return;
    if (msg.type === "start")  handleStart(msg);
    else if (msg.type === "cancel") handleCancel();
});

function post(m: OutMsg) { (self as DedicatedWorkerGlobalScope).postMessage(m); }

async function handleStart(msg: StartMsg) {
    const { modelKey, filename, url } = msg;
    let currentOffset = Number(msg.offset ?? 0);
    cancelled = false;

    let syncHandle: FileSystemSyncAccessHandle | null = null;
    let bytesSinceFlush = 0;

    try {
        const root      = await navigator.storage.getDirectory();
        const dlDir     = await root.getDirectoryHandle(OPFS_DIR, { create: true });
        const modelDir  = await dlDir.getDirectoryHandle(modelKey,  { create: true });
        const fileHandle = await modelDir.getFileHandle(filename,   { create: true });

        syncHandle = await fileHandle.createSyncAccessHandle();
        syncHandle.truncate(currentOffset);
        syncHandle.flush();

        const headers: Record<string, string> = {};
        if (currentOffset > 0) headers["Range"] = `bytes=${currentOffset}-`;
        const resp = await fetch(url, { headers });

        if (resp.status === 416) {
            const size = syncHandle.getSize();
            syncHandle.close();
            post({ type: "done", totalBytes: size });
            return;
        }
        if (!resp.ok && resp.status !== 206) {
            syncHandle.close();
            post({ type: "error", error: `fetch failed (${resp.status})` });
            return;
        }
        if (resp.status === 200 && currentOffset > 0) {
            syncHandle.truncate(0);
            currentOffset = 0;
        }

        const cr = resp.headers.get("content-range");
        const contentLength = Number(resp.headers.get("content-length") || "0") || 0;
        const totalBytes = cr?.match(/\/(\d+)\s*$/)
            ? Number(cr.match(/\/(\d+)\s*$/)![1])
            : currentOffset + contentLength;

        if (!resp.body) {
            syncHandle.close();
            post({ type: "error", error: "no response body" });
            return;
        }
        const reader = resp.body.getReader();
        activeReader = reader;

        while (true) {
            if (cancelled) {
                try { await reader.cancel(); } catch { /* */ }
                break;
            }
            const { value, done } = await reader.read();
            if (done) break;
            if (!value) continue;

            const written = syncHandle.write(value, { at: currentOffset });
            currentOffset   += written;
            bytesSinceFlush += written;

            if (bytesSinceFlush >= FLUSH_INTERVAL) {
                syncHandle.flush();
                bytesSinceFlush = 0;
            }
            post({ type: "progress", bytesWritten: currentOffset, totalBytes, chunkBytes: written });
        }

        syncHandle.flush();
        syncHandle.close();
        syncHandle  = null;
        activeReader = null;

        if (cancelled) post({ type: "cancelled", bytesWritten: currentOffset });
        else            post({ type: "done", totalBytes: currentOffset });
    } catch (err) {
        const error = (err as Error)?.message ?? String(err);
        if (syncHandle) {
            try { syncHandle.flush(); } catch { /* */ }
            try { syncHandle.close(); } catch { /* */ }
        }
        activeReader = null;
        post({ type: "error", error });
    }
}

function handleCancel() {
    cancelled = true;
    if (activeReader) {
        try { activeReader.cancel(); } catch { /* */ }
    }
}
