// Crash-surviving diagnostic log storage for the rullama PWA.
//
// Worker-only. Reuses the iOS-Safari-validated sync OPFS handle pattern
// from inference-core-worker.ts:258 (createSyncAccessHandleWithRetry),
// but with a shorter retry budget — logs are best-effort, the worker
// should keep running even if the logger fails to open.
//
// Layout under OPFS root:
//   rullama-logs/
//     manifest.json          # JSON array of session metadata
//     sessions/<id>.log      # one plain-text file per page-load session
//
// Beacons are flushed on every `append()` (user-confirmed policy): a
// single sync write + flush is microseconds, and we never want to lose
// the last line before iOS kills the tab. The whole point of this
// module is post-crash readability.

export type LogLevel = "info" | "warn" | "error";

export interface SessionMeta {
    id:        string;   // ISO timestamp + 6-char random suffix
    startMs:   number;
    sizeBytes: number;   // size at last manifest write; on-disk size is authoritative
    cleanExit: boolean;
}

const LOGS_DIR        = "rullama-logs";
const SESSIONS_DIR    = "sessions";
const MANIFEST        = "manifest.json";
const MAX_SESSIONS    = 20;
const MAX_SESSION_BYTES = 512 * 1024; // 512 KiB per session

const enc = new TextEncoder();
const dec = new TextDecoder();

// Module-level state (worker-singleton).
let logsRoot:    FileSystemDirectoryHandle | null = null;
let sessionsDir: FileSystemDirectoryHandle | null = null;

let activeHandle:  FileSystemSyncAccessHandle | null = null;
let activeId:      string | null = null;
let activeSize:    number        = 0;
let activeStartMs: number        = 0;

async function ensureDirs(): Promise<void> {
    if (logsRoot && sessionsDir) return;
    const root = await navigator.storage.getDirectory();
    logsRoot    = await root.getDirectoryHandle(LOGS_DIR, { create: true });
    sessionsDir = await logsRoot.getDirectoryHandle(SESSIONS_DIR, { create: true });
}

/** Backoff-retry sync handle opener — shorter budget than the model
 *  weight reader since logger failure must not block the worker. */
async function retryOpenSync(fh: FileSystemFileHandle, budgetMs = 5_000): Promise<FileSystemSyncAccessHandle> {
    const start = Date.now();
    let attempt = 0;
    while (true) {
        try { return await fh.createSyncAccessHandle(); }
        catch (e) {
            attempt += 1;
            if (Date.now() - start >= budgetMs) throw e;
            const delay = Math.min(800, 100 * Math.pow(2, attempt - 1));
            await new Promise((r) => setTimeout(r, delay));
        }
    }
}

async function readManifest(): Promise<SessionMeta[]> {
    await ensureDirs();
    try {
        const fh  = await logsRoot!.getFileHandle(MANIFEST, { create: false });
        const f   = await fh.getFile();
        const txt = await f.text();
        const parsed = JSON.parse(txt);
        return Array.isArray(parsed) ? (parsed as SessionMeta[]) : [];
    } catch {
        return [];
    }
}

async function writeManifest(arr: SessionMeta[]): Promise<void> {
    await ensureDirs();
    const fh = await logsRoot!.getFileHandle(MANIFEST, { create: true });
    const w  = await fh.createWritable();
    await w.write(JSON.stringify(arr));
    await w.close();
}

async function rotate(): Promise<void> {
    const arr = await readManifest();
    arr.sort((a, b) => b.startMs - a.startMs);
    if (arr.length <= MAX_SESSIONS) return;
    const toKeep = arr.slice(0, MAX_SESSIONS);
    const toDrop = arr.slice(MAX_SESSIONS);
    for (const s of toDrop) {
        try { await sessionsDir!.removeEntry(`${s.id}.log`); } catch { /* already gone */ }
    }
    await writeManifest(toKeep);
}

/** Lazy idempotent initializer. Opens a sync handle to the active
 *  session log file and writes the INIT header line. Safe to call
 *  multiple times — only the first call does work. */
export async function init(sessionId: string): Promise<string> {
    if (activeHandle) return activeId!;
    await ensureDirs();

    const fh     = await sessionsDir!.getFileHandle(`${sessionId}.log`, { create: true });
    const handle = await retryOpenSync(fh);

    activeStartMs = Date.now();
    const header  = `INIT ts=${new Date(activeStartMs).toISOString()} ua=${navigator.userAgent}\n`;
    const hbytes  = enc.encode(header);
    handle.truncate(0);
    handle.write(hbytes, { at: 0 });
    handle.flush();

    activeHandle = handle;
    activeId     = sessionId;
    activeSize   = hbytes.length;

    // Manifest + rotation are best-effort; if they fail the active
    // log still works, the session just won't appear in listSessions
    // until the next manifest write.
    try {
        const arr = await readManifest();
        // Filter out a stale entry for this id (e.g. crash + restart with
        // a same-second random suffix collision — extremely unlikely but
        // dedupe just in case).
        const filtered = arr.filter((s) => s.id !== sessionId);
        filtered.push({ id: sessionId, startMs: activeStartMs, sizeBytes: hbytes.length, cleanExit: false });
        await writeManifest(filtered);
        await rotate();
    } catch { /* */ }

    return sessionId;
}

/** Append a beacon line. Sync write + flush per call. Idempotent on
 *  pre-init state (silently drops the line if init hasn't run). */
export function append(level: LogLevel, tag: string, msg: string): void {
    if (!activeHandle) return;

    const line  = `${new Date().toISOString()} ${level} [${tag}] ${msg}\n`;
    const bytes = enc.encode(line);

    // Per-session cap: drop oldest half via in-place rewrite. Cheap
    // because writes are append-only and the cap is small.
    if (activeSize + bytes.length > MAX_SESSION_BYTES) {
        try {
            const buf = new Uint8Array(activeSize);
            activeHandle.read(buf, { at: 0 });
            const text   = dec.decode(buf);
            const cutAt  = Math.floor(text.length / 2);
            const nlIdx  = text.indexOf("\n", cutAt);
            const tail   = nlIdx >= 0 ? text.slice(nlIdx + 1) : text.slice(cutAt);
            const tailBs = enc.encode(`[... older lines dropped ...]\n${tail}`);
            activeHandle.truncate(0);
            activeHandle.write(tailBs, { at: 0 });
            activeSize = tailBs.length;
        } catch { /* */ }
    }

    try {
        activeHandle.write(bytes, { at: activeSize });
        activeHandle.flush();
        activeSize += bytes.length;
    } catch { /* swallow — logger must never crash the worker */ }
}

export async function listSessions(): Promise<SessionMeta[]> {
    const arr = await readManifest();
    // Augment with the live activeSize so the viewer sees current size
    // for the in-flight session.
    if (activeId) {
        for (const s of arr) {
            if (s.id === activeId) s.sizeBytes = activeSize;
        }
    }
    arr.sort((a, b) => b.startMs - a.startMs);
    return arr;
}

export async function readSession(id: string): Promise<string> {
    await ensureDirs();
    // Active session: read via the sync handle so the viewer sees the
    // latest bytes including any just-written beacon.
    if (activeId === id && activeHandle) {
        const buf = new Uint8Array(activeSize);
        activeHandle.read(buf, { at: 0 });
        return dec.decode(buf);
    }
    try {
        const fh = await sessionsDir!.getFileHandle(`${id}.log`, { create: false });
        const f  = await fh.getFile();
        return await f.text();
    } catch (e) {
        return `(failed to read session ${id}: ${(e as Error)?.message ?? e})`;
    }
}

export async function deleteSession(id: string): Promise<void> {
    await ensureDirs();
    if (activeId === id && activeHandle) {
        try { activeHandle.close(); } catch { /* */ }
        activeHandle = null;
        activeId     = null;
        activeSize   = 0;
    }
    try { await sessionsDir!.removeEntry(`${id}.log`); } catch { /* */ }
    const arr = await readManifest();
    await writeManifest(arr.filter((s) => s.id !== id));
}

export async function deleteAll(): Promise<void> {
    await ensureDirs();
    if (activeHandle) {
        try { activeHandle.close(); } catch { /* */ }
        activeHandle = null;
        activeId     = null;
        activeSize   = 0;
    }
    const arr = await readManifest();
    for (const s of arr) {
        try { await sessionsDir!.removeEntry(`${s.id}.log`); } catch { /* */ }
    }
    await writeManifest([]);
}

/** Mark the active session as exited cleanly and release the sync
 *  handle. Called from the worker's shutdown path. */
export async function markCleanExit(): Promise<void> {
    if (!activeHandle || !activeId) return;
    try {
        const exitLine = enc.encode(`${new Date().toISOString()} info [sys] EXIT clean\n`);
        activeHandle.write(exitLine, { at: activeSize });
        activeHandle.flush();
        activeSize += exitLine.length;
    } catch { /* */ }

    const finalId   = activeId;
    const finalSize = activeSize;
    try { activeHandle.close(); } catch { /* */ }
    activeHandle = null;
    activeId     = null;
    activeSize   = 0;

    try {
        const arr = await readManifest();
        const updated = arr.map((s) => s.id === finalId
            ? { ...s, cleanExit: true, sizeBytes: finalSize }
            : s);
        await writeManifest(updated);
    } catch { /* */ }
}

export function currentSessionId(): string | null {
    return activeId;
}
