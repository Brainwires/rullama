// OPFS-backed store for generated TTS clips, so they survive a page reload.
//
// Design mirrors queue_store.ts / image_store.ts: the bulky payload (24 kHz f32
// PCM) lives in per-clip `<id>.f32` files under an OPFS directory, and only small
// metadata is kept in `manifest.json`. The UI holds the metadata list in memory;
// PCM is loaded from OPFS on demand (play / export) and dropped right after, so a
// long backlog of clips never sits in RAM. Newest-first ordering.

const CLIPS_DIR = "rullama-tts-clips";
const MANIFEST = "manifest.json";

export interface ClipMeta {
    id: string;
    text: string;
    voice: string;
    sampleRate: number;
    /** PCM sample count → duration = samples / sampleRate. */
    samples: number;
    ts: number;
}

async function clipsDir(): Promise<FileSystemDirectoryHandle | null> {
    try {
        const root = await navigator.storage.getDirectory();
        return await root.getDirectoryHandle(CLIPS_DIR, { create: true });
    } catch {
        // OPFS unavailable (private mode, old browser) — degrade to no persistence.
        return null;
    }
}

async function writeBytes(
    dir: FileSystemDirectoryHandle,
    name: string,
    bytes: Uint8Array,
): Promise<void> {
    const fh = await dir.getFileHandle(name, { create: true });
    // FileSystemSyncAccessHandle is Worker-only in some browsers; feature-detect.
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

async function readBytes(dir: FileSystemDirectoryHandle, name: string): Promise<Uint8Array | null> {
    try {
        const fh = await dir.getFileHandle(name, { create: false });
        const file = await fh.getFile();
        return new Uint8Array(await file.arrayBuffer());
    } catch {
        return null;
    }
}

async function readManifest(dir: FileSystemDirectoryHandle): Promise<ClipMeta[]> {
    const bytes = await readBytes(dir, MANIFEST);
    if (!bytes) return [];
    try {
        const list = JSON.parse(new TextDecoder().decode(bytes)) as ClipMeta[];
        return Array.isArray(list) ? list : [];
    } catch {
        return [];
    }
}

async function writeManifest(dir: FileSystemDirectoryHandle, list: ClipMeta[]): Promise<void> {
    await writeBytes(dir, MANIFEST, new TextEncoder().encode(JSON.stringify(list)));
}

/** All persisted clips, newest first. Metadata only — no PCM is loaded. */
export async function listClips(): Promise<ClipMeta[]> {
    const dir = await clipsDir();
    if (!dir) return [];
    return await readManifest(dir);
}

/** Persist a freshly generated clip: write its PCM file, then prepend to the manifest. */
export async function saveClip(meta: ClipMeta, pcm: Float32Array): Promise<void> {
    const dir = await clipsDir();
    if (!dir) return;
    const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    await writeBytes(dir, `${meta.id}.f32`, bytes);
    const list = await readManifest(dir);
    await writeManifest(dir, [meta, ...list.filter((c) => c.id !== meta.id)]);
}

/** Load one clip's PCM from OPFS (called on play / export, then released). */
export async function loadClipPcm(id: string): Promise<Float32Array | null> {
    const dir = await clipsDir();
    if (!dir) return null;
    const bytes = await readBytes(dir, `${id}.f32`);
    if (!bytes) return null;
    // Copy out of the file ArrayBuffer so the result owns exactly its samples.
    return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

/** Delete a clip's PCM file and remove it from the manifest. */
export async function deleteClip(id: string): Promise<void> {
    const dir = await clipsDir();
    if (!dir) return;
    try { await dir.removeEntry(`${id}.f32`); } catch { /* already gone */ }
    const list = await readManifest(dir);
    await writeManifest(dir, list.filter((c) => c.id !== id));
}
