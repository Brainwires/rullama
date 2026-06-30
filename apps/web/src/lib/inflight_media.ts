// OPFS-backed persistence for the image pixels and audio PCM that
// accompany an in-flight multimodal generation. Pairs with the
// KV-snapshot path in `opfs.ts` — when the user backgrounds a
// multimodal turn mid-stream and iOS kills the tab, these helpers let
// the boot-resume effect read the original media back, re-encode it
// through the vision / audio towers, and continue generation.
//
// Layout under the OPFS root (NOT under `rullama-models/` so it
// survives a "Clear cached models" action, same as the KV snapshot):
//
//   rullama-inflight-media/
//     img-0.f32   raw channel-first f32 pixel buffer
//     img-0.json  { h, w, dataUrl }   (small JPEG thumbnail for UI)
//     aud-0.f32   raw 16 kHz mono f32 PCM
//     aud-0.json  { durationMs }
//
// The whole directory gets `clearInflightMedia()`-ed on clean
// completion of the multimodal turn.

import type { ImageAttachment } from "@/lib/types";

const MEDIA_DIR = "rullama-inflight-media";

async function ensureMediaDir(): Promise<FileSystemDirectoryHandle> {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(MEDIA_DIR, { create: true });
}

async function writeBytes(
    dir: FileSystemDirectoryHandle,
    name: string,
    bytes: Uint8Array,
): Promise<void> {
    const fh = await dir.getFileHandle(name, { create: true });
    // FileSystemSyncAccessHandle is only available inside Workers in
    // some browsers; cast to unknown to bypass the imprecise lib.dom
    // shape and feature-detect at runtime.
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

async function writeJson(
    dir: FileSystemDirectoryHandle,
    name: string,
    obj: unknown,
): Promise<void> {
    const bytes = new TextEncoder().encode(JSON.stringify(obj));
    await writeBytes(dir, name, bytes);
}

async function readBytes(
    dir: FileSystemDirectoryHandle,
    name: string,
): Promise<Uint8Array | null> {
    try {
        const fh = await dir.getFileHandle(name, { create: false });
        const file = await fh.getFile();
        return new Uint8Array(await file.arrayBuffer());
    } catch { return null; }
}

async function readJson<T>(
    dir: FileSystemDirectoryHandle,
    name: string,
): Promise<T | null> {
    const bytes = await readBytes(dir, name);
    if (!bytes) return null;
    try { return JSON.parse(new TextDecoder().decode(bytes)) as T; }
    catch { return null; }
}

/** Persist one queued image to OPFS so a kill-and-resume can re-encode
 *  it through the vision tower. `pixels` is the channel-first f32
 *  buffer that `Model.encodeImage` consumes. */
export async function saveInflightImage(
    seq: number,
    pixels: Float32Array,
    h: number,
    w: number,
    dataUrl: string,
): Promise<void> {
    const dir = await ensureMediaDir();
    const bytes = new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
    await writeBytes(dir, `img-${seq}.f32`, bytes);
    await writeJson(dir, `img-${seq}.json`, { h, w, dataUrl });
}

/** Persist one queued audio clip — 16 kHz mono f32 PCM ready for
 *  `Model.encodeAudio`. */
export async function saveInflightAudio(
    seq: number,
    pcm: Float32Array,
    durationMs: number,
): Promise<void> {
    const dir = await ensureMediaDir();
    const bytes = new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    await writeBytes(dir, `aud-${seq}.f32`, bytes);
    await writeJson(dir, `aud-${seq}.json`, { durationMs });
}

/** Read back every image attachment persisted for the in-flight
 *  turn, in send order. Returns `[]` if the media dir is absent or
 *  was never populated. */
export async function readInflightImages(): Promise<ImageAttachment[]> {
    const root = await navigator.storage.getDirectory();
    let dir: FileSystemDirectoryHandle;
    try { dir = await root.getDirectoryHandle(MEDIA_DIR, { create: false }); }
    catch { return []; }

    const out: ImageAttachment[] = [];
    for (let i = 0; ; i++) {
        const meta = await readJson<{ h: number; w: number; dataUrl: string }>(dir, `img-${i}.json`);
        if (!meta) break;
        const bytes = await readBytes(dir, `img-${i}.f32`);
        if (!bytes) break;
        const pixels = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        out.push({ pixels, h: meta.h, w: meta.w, dataUrl: meta.dataUrl });
    }
    return out;
}

/** Read back every audio clip persisted for the in-flight turn. */
export async function readInflightAudio(): Promise<Array<{ pcm: Float32Array; durationMs: number }>> {
    const root = await navigator.storage.getDirectory();
    let dir: FileSystemDirectoryHandle;
    try { dir = await root.getDirectoryHandle(MEDIA_DIR, { create: false }); }
    catch { return []; }

    const out: Array<{ pcm: Float32Array; durationMs: number }> = [];
    for (let i = 0; ; i++) {
        const meta = await readJson<{ durationMs: number }>(dir, `aud-${i}.json`);
        if (!meta) break;
        const bytes = await readBytes(dir, `aud-${i}.f32`);
        if (!bytes) break;
        const pcm = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        out.push({ pcm, durationMs: meta.durationMs });
    }
    return out;
}

/** Remove the entire inflight-media subtree. Best-effort. Called on
 *  clean completion of the multimodal turn (or when resume decides
 *  the persisted media is unusable). */
export async function clearInflightMedia(): Promise<void> {
    try {
        const root = await navigator.storage.getDirectory();
        await root.removeEntry(MEDIA_DIR, { recursive: true });
    } catch { /* */ }
}
