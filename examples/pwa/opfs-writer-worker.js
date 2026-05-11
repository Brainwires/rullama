// rullama OPFS writer — Dedicated Worker
//
// Streams a model URL to an OPFS file via `FileSystemSyncAccessHandle.write()`.
// Bytes go network → sync handle → disk without ever pinning a Blob in the JS
// heap. This is what bypasses iOS Safari's two implicit caps on the Blob/
// IndexedDB path:
//
//   * ~2 GiB of live JS memory before the WebContent process gets killed
//   * ~5.6 GiB on a single combined Blob (silent truncation)
//
// `createSyncAccessHandle()` is **only available in Worker contexts** —
// calling it from the main thread throws. That's why this file is a worker.
//
// Wire protocol (main → worker):
//   { type: 'start', modelKey, filename, url, offset }   // begin/resume
//   { type: 'cancel' }
//
// Wire protocol (worker → main):
//   { type: 'progress', bytesWritten, totalBytes, chunkBytes }
//   { type: 'done',     totalBytes }
//   { type: 'error',    error }
//   { type: 'cancelled', bytesWritten }

const OPFS_DIR = 'rullama-models';
const FLUSH_INTERVAL = 64 * 1024 * 1024;   // flush to disk every 64 MiB

let cancelled = false;
let activeReader = null;

self.addEventListener('message', (ev) => {
    const msg = ev.data;
    if (!msg || typeof msg !== 'object') return;
    if (msg.type === 'start') handleStart(msg);
    else if (msg.type === 'cancel') handleCancel();
});

async function handleStart(msg) {
    const { modelKey, filename, url } = msg;
    let currentOffset = Number(msg.offset || 0);
    cancelled = false;

    let syncHandle = null;
    let bytesSinceFlush = 0;
    let phase = 'start';

    try {
        phase = 'getDirectory';
        const root  = await navigator.storage.getDirectory();
        phase = 'getDirectoryHandle(rullama-models)';
        const dlDir = await root.getDirectoryHandle(OPFS_DIR, { create: true });
        phase = `getDirectoryHandle(${modelKey})`;
        const modelDir = await dlDir.getDirectoryHandle(modelKey, { create: true });
        phase = `getFileHandle(${filename})`;
        const fileHandle = await modelDir.getFileHandle(filename, { create: true });

        phase = 'createSyncAccessHandle';
        syncHandle = await fileHandle.createSyncAccessHandle();

        // Probe: tiny known-good write at offset 0 lets us distinguish "OPFS
        // is broken in this session" from "your big-file write specifically
        // failed". If this throws, the underlying OPFS storage on the device
        // is in a bad state — reboot the phone or wait for iOS to reclaim.
        phase = 'probe_write_1KiB_at_0';
        const probe = new Uint8Array(1024);
        const probeWritten = syncHandle.write(probe, { at: 0 });
        if (probeWritten !== probe.byteLength) {
            throw new Error(`probe short write: ${probeWritten} bytes`);
        }
        phase = 'probe_flush';
        syncHandle.flush();

        // Now truncate to the real start offset (may be 0).
        phase = `truncate(${currentOffset})`;
        syncHandle.truncate(currentOffset);
        phase = 'truncate_flush';
        syncHandle.flush();

        const headers = {};
        if (currentOffset > 0) headers['Range'] = `bytes=${currentOffset}-`;

        const resp = await fetch(url, { headers });

        if (resp.status === 416) {
            const size = syncHandle.getSize();
            syncHandle.close();
            self.postMessage({ type: 'done', totalBytes: size });
            return;
        }
        if (!resp.ok && resp.status !== 206) {
            syncHandle.close();
            self.postMessage({ type: 'error', error: `fetch failed (${resp.status})` });
            return;
        }
        // If we asked for a Range but the server sent 200 (full file), reset.
        if (resp.status === 200 && currentOffset > 0) {
            syncHandle.truncate(0);
            currentOffset = 0;
        }

        const contentLength = Number(resp.headers.get('content-length')) || 0;
        // Trust Content-Range when present (gives the true total even on resume).
        const cr = resp.headers.get('content-range');
        let totalBytes;
        if (cr) {
            const m = cr.match(/\/(\d+)\s*$/);
            totalBytes = m ? Number(m[1]) : (currentOffset + contentLength);
        } else {
            totalBytes = currentOffset + contentLength;
        }

        const reader = resp.body.getReader();
        activeReader = reader;

        while (true) {
            if (cancelled) {
                try { reader.cancel(); } catch (_) {}
                break;
            }
            const { value, done } = await reader.read();
            if (done) break;

            // Synchronous, no-copy write straight to disk.
            const written = syncHandle.write(value, { at: currentOffset });
            currentOffset   += written;
            bytesSinceFlush += written;

            if (bytesSinceFlush >= FLUSH_INTERVAL) {
                syncHandle.flush();
                bytesSinceFlush = 0;
            }

            self.postMessage({
                type: 'progress',
                bytesWritten: currentOffset,
                totalBytes,
                chunkBytes: written,
            });
        }

        syncHandle.flush();
        syncHandle.close();
        syncHandle  = null;
        activeReader = null;

        if (cancelled) {
            self.postMessage({ type: 'cancelled', bytesWritten: currentOffset });
        } else {
            self.postMessage({ type: 'done', totalBytes: currentOffset });
        }
    } catch (err) {
        const error = err && err.message ? err.message : String(err);
        if (syncHandle) {
            try { syncHandle.flush(); } catch (_) {}
            try { syncHandle.close(); } catch (_) {}
        }
        activeReader = null;
        self.postMessage({ type: 'error', error });
    }
}

function handleCancel() {
    cancelled = true;
    if (activeReader && typeof activeReader.cancel === 'function') {
        try { activeReader.cancel(); } catch (_) {}
    }
}
