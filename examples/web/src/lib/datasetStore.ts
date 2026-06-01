// OPFS-backed dataset storage for the Fine-tune tab's "Saved" library.
//
// Datasets are stored as plain JSONL files under
//   <opfs>/datasets/<safe-name>.jsonl
// Each save writes the whole file. Listing enumerates the directory
// on demand — no manifest file to keep in sync (the directory IS the
// manifest, and OPFS gives us file size + lastModified for free).
//
// No Rust round-trip needed — datasets are small text blobs and the
// browser's File System Access API handles everything directly.

export interface SavedDatasetMeta {
    name: string;
    /** Bytes on disk. */
    size: number;
    /** Epoch ms of last write (per File.lastModified). */
    lastModified: number;
    /** Cached count from the file body — fetched lazily by `listSavedDatasets`. */
    lineCount: number;
}

const DIRECTORY_NAME = "datasets";
const EXTENSION = ".jsonl";

/** Normalise an arbitrary user-supplied name into something OPFS will
 *  accept. Strips path separators + control chars; collapses runs of
 *  whitespace into a single space; trims; truncates to a sane length.
 *  Returns null if the result would be empty. */
export function normalizeDatasetName(raw: string): string | null {
    const cleaned = raw
        .replace(/[\\/:*?"<>|\x00-\x1f]+/g, "")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 80);
    return cleaned.length > 0 ? cleaned : null;
}

/** Resolve the datasets directory, creating it on first use. */
async function getDir(): Promise<FileSystemDirectoryHandle> {
    if (!navigator.storage?.getDirectory) {
        throw new Error("OPFS not available in this browser.");
    }
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(DIRECTORY_NAME, { create: true });
}

/** Save a dataset under `name`, overwriting any existing entry. */
export async function saveDataset(name: string, jsonl: string): Promise<void> {
    const safe = normalizeDatasetName(name);
    if (!safe) throw new Error("Dataset name is empty.");
    const dir = await getDir();
    const fh = await dir.getFileHandle(`${safe}${EXTENSION}`, { create: true });
    const w = await fh.createWritable();
    try {
        await w.write(jsonl);
    } finally {
        await w.close();
    }
}

/** Load a saved dataset by name. Returns the raw JSONL bytes as a
 *  string. Errors if the file doesn't exist. */
export async function loadDataset(name: string): Promise<string> {
    const safe = normalizeDatasetName(name);
    if (!safe) throw new Error("Dataset name is empty.");
    const dir = await getDir();
    const fh = await dir.getFileHandle(`${safe}${EXTENSION}`);
    const f = await fh.getFile();
    return await f.text();
}

/** Delete a saved dataset. No-op if it doesn't exist. */
export async function deleteDataset(name: string): Promise<void> {
    const safe = normalizeDatasetName(name);
    if (!safe) throw new Error("Dataset name is empty.");
    const dir = await getDir();
    try {
        await dir.removeEntry(`${safe}${EXTENSION}`);
    } catch (e) {
        // NotFoundError is fine — caller asked to delete something
        // that's already gone.
        if ((e as DOMException).name !== "NotFoundError") throw e;
    }
}

/** Enumerate all saved datasets with metadata. Sorted by most-recent
 *  write first. Reads each file body to compute its line count —
 *  acceptable because datasets are small (KB, not MB). */
export async function listSavedDatasets(): Promise<SavedDatasetMeta[]> {
    const dir = await getDir();
    const out: SavedDatasetMeta[] = [];
    // values() / entries() is the standard FileSystemDirectoryHandle
    // iterator. The TS DOM lib types it loosely; declare what we use.
    const iter = (dir as unknown as {
        values(): AsyncIterableIterator<FileSystemHandle>;
    }).values();
    for await (const handle of iter) {
        if (handle.kind !== "file") continue;
        if (!handle.name.endsWith(EXTENSION)) continue;
        const fh = handle as FileSystemFileHandle;
        const file = await fh.getFile();
        // Trim trailing newline before split so a final blank line
        // doesn't inflate the count.
        const text = await file.text();
        const lineCount = text.replace(/\n+$/, "").split(/\r?\n/).filter((l) => l.trim().length > 0).length;
        out.push({
            name: handle.name.slice(0, -EXTENSION.length),
            size: file.size,
            lastModified: file.lastModified,
            lineCount,
        });
    }
    out.sort((a, b) => b.lastModified - a.lastModified);
    return out;
}
