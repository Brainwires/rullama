// OPFS-backed LoRA adapter storage.
//
// Adapters live at `rullama-adapters/<name>.bin` at OPFS root,
// parallel to the `rullama-models/<modelKey>/<filename>` weight tree
// in `opfs.ts`. Each adapter is a single safetensors blob produced
// by `TrainingSession.saveAdapter` and consumable by
// `Model.loadAdapter`.
//
// Writes go through the worker (see `inference-core-worker.ts` →
// `trainingSaveAdapter`) so the bytes never leave the worker context
// once produced; this file is the read-side + listing CRUD.

const ADAPTERS_DIR = "rullama-adapters";

export interface AdapterEntry {
    name: string;
    size: number;
    /** ISO date string from the FileSystemFileHandle. */
    lastModified: number;
}

async function getDir(create: boolean): Promise<FileSystemDirectoryHandle> {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(ADAPTERS_DIR, { create });
}

/** List every saved adapter with size + lastModified. */
export async function listAdapters(): Promise<AdapterEntry[]> {
    try {
        const dir = await getDir(false);
        const out: AdapterEntry[] = [];
        // `entries()` is iterable; TS lib types may not match all browsers' shape.
        const anyDir = dir as unknown as {
            entries(): AsyncIterableIterator<[string, FileSystemHandle]>;
        };
        for await (const [name, handle] of anyDir.entries()) {
            if (handle.kind !== "file") continue;
            if (!name.endsWith(".bin")) continue;
            try {
                const fh = handle as FileSystemFileHandle;
                const f = await fh.getFile();
                out.push({
                    name: name.replace(/\.bin$/, ""),
                    size: f.size,
                    lastModified: f.lastModified,
                });
            } catch { /* skip */ }
        }
        out.sort((a, b) => b.lastModified - a.lastModified);
        return out;
    } catch {
        return [];
    }
}

/** Read an adapter's raw bytes from OPFS. */
export async function readAdapter(name: string): Promise<Uint8Array> {
    const dir = await getDir(false);
    const fh = await dir.getFileHandle(`${name}.bin`, { create: false });
    const f = await fh.getFile();
    return new Uint8Array(await f.arrayBuffer());
}

/** Delete an adapter. No-op if not present. */
export async function deleteAdapter(name: string): Promise<void> {
    try {
        const dir = await getDir(false);
        await dir.removeEntry(`${name}.bin`);
    } catch { /* */ }
}

/** True iff `rullama-adapters/<name>.bin` exists. */
export async function hasAdapter(name: string): Promise<boolean> {
    try {
        const dir = await getDir(false);
        await dir.getFileHandle(`${name}.bin`, { create: false });
        return true;
    } catch {
        return false;
    }
}

/** Trigger a browser download for the adapter bytes. */
export async function downloadAdapter(name: string): Promise<void> {
    const bytes = await readAdapter(name);
    const blob = new Blob([bytes as BlobPart], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    try {
        const a = document.createElement("a");
        a.href = url;
        a.download = `${name}.safetensors`;
        document.body.appendChild(a);
        a.click();
        a.remove();
    } finally {
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
}

/** Display-friendly size (MB to one decimal). */
export function formatAdapterSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
