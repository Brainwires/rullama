// OPFS-backed blob store for chat-attached image thumbnails.
//
// Why: we tried storing thumbnail data URLs (~30 KB base64 JPEG) directly
// in the SQLite `message_images.thumb_data_url` TEXT column, and hit an
// rsqlite-wasm panic on insert:
//
//   panicked at crates/rsqlite-storage/src/btree_write.rs:336:13:
//     range start index 4294965310 out of range for slice of length 4096
//
// The 4096 is the SQLite default page size; 4_294_965_310 is u32::MAX
// minus a few thousand — a classic underflow in overflow-page accounting.
// Single-row payloads larger than a page trip it. Workaround: stop
// storing the blob in SQLite. The thumbnail bytes go to OPFS keyed by a
// random UUID, and only that UUID lives in the messages row (well under
// a page).
//
// Side benefits:
//   * Render path becomes `URL.createObjectURL(file)` — a direct browser
//     handle to the OPFS file. No base64 decode, no DOM repaint cost for
//     huge data URLs.
//   * Cleanup is decoupled from row deletion (SQLite cascade gets the
//     row; we sweep OPFS by conversation when convDelete fires).

const DIR = "rullama-images";

async function imagesDir(): Promise<FileSystemDirectoryHandle> {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(DIR, { create: true });
}

/**
 * Persist a JPEG dataURL into OPFS and return its opaque key (a UUID
 * with `.jpg` suffix). Caller stores the key in SQLite; calling
 * `loadThumbBlobUrl(key)` later returns a `blob:` URL pointing at the
 * stored bytes.
 */
export async function saveThumb(dataUrl: string): Promise<string> {
    const comma = dataUrl.indexOf(",");
    if (comma < 0) throw new Error("saveThumb: dataUrl missing comma");
    const b64 = dataUrl.slice(comma + 1);
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);

    const id = `${crypto.randomUUID()}.jpg`;
    const dir = await imagesDir();
    const fh = await dir.getFileHandle(id, { create: true });
    const wr = await fh.createWritable();
    await wr.write(bytes);
    await wr.close();
    return id;
}

/**
 * Resolve a thumb key to a `blob:` URL usable in `<img src>`. Returns
 * null if the file is missing (e.g. user wiped OPFS between sessions);
 * the caller should fall back to an empty bubble or a placeholder.
 *
 * Note: the returned URL holds the file alive until the caller revokes
 * it via `URL.revokeObjectURL`. We don't track them here — components
 * that load many thumbs at once should revoke on unmount.
 */
export async function loadThumbBlobUrl(id: string): Promise<string | null> {
    try {
        const dir = await imagesDir();
        const fh = await dir.getFileHandle(id, { create: false });
        const file = await fh.getFile();
        return URL.createObjectURL(file);
    } catch {
        return null;
    }
}

/** Best-effort delete. Called from convDelete; missing files are not an error. */
export async function deleteThumbs(ids: readonly string[]): Promise<void> {
    if (ids.length === 0) return;
    const dir = await imagesDir();
    for (const id of ids) {
        try { await dir.removeEntry(id); } catch { /* gone already, fine */ }
    }
}
