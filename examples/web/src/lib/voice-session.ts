/// Persist the voice-cloning recording session to OPFS so recorded clips survive a
/// reload (handy while testing). Stores each clip's PCM as a raw f32 file + a manifest.

const DIR = "voice-clone-session";
const MANIFEST = "manifest.json";

export interface SessionClip {
    id: string;
    text: string;
    pcm: Float32Array | null;
    durationSec: number;
}

interface PersistEntry {
    id: string;
    text: string;
    file: string | null; // <id>.pcm, or null for un-recorded prompts
    durationSec: number;
}

async function dir(create: boolean): Promise<FileSystemDirectoryHandle> {
    const root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle(DIR, { create });
}

/** Save the whole session (debounce calls — writes every clip's PCM). */
export async function saveSession(clips: SessionClip[]): Promise<void> {
    try {
        const d = await dir(true);
        const manifest: PersistEntry[] = [];
        const keep = new Set<string>([MANIFEST]);
        for (const c of clips) {
            let file: string | null = null;
            if (c.pcm) {
                file = `${c.id}.pcm`;
                keep.add(file);
                const fh = await d.getFileHandle(file, { create: true });
                const w = await fh.createWritable();
                const ab = new ArrayBuffer(c.pcm.byteLength); // plain ArrayBuffer (COI typing)
                new Float32Array(ab).set(c.pcm);
                await w.write(ab);
                await w.close();
            }
            manifest.push({ id: c.id, text: c.text, file, durationSec: c.durationSec });
        }
        const mh = await d.getFileHandle(MANIFEST, { create: true });
        const mw = await mh.createWritable();
        await mw.write(JSON.stringify(manifest));
        await mw.close();
        // prune stale clip files (deleted/re-recorded)
        for await (const name of (d as unknown as { keys(): AsyncIterable<string> }).keys()) {
            if (!keep.has(name)) await d.removeEntry(name).catch(() => {});
        }
    } catch {
        /* OPFS unavailable — persistence is best-effort */
    }
}

/** Restore a saved session, or null if none. */
export async function loadSession(): Promise<SessionClip[] | null> {
    try {
        const d = await dir(false);
        const mh = await d.getFileHandle(MANIFEST);
        const manifest: PersistEntry[] = JSON.parse(await (await mh.getFile()).text());
        const out: SessionClip[] = [];
        for (const m of manifest) {
            let pcm: Float32Array | null = null;
            if (m.file) {
                const fh = await d.getFileHandle(m.file);
                pcm = new Float32Array(await (await fh.getFile()).arrayBuffer());
            }
            out.push({ id: m.id, text: m.text, pcm, durationSec: m.durationSec });
        }
        return out.length ? out : null;
    } catch {
        return null;
    }
}

/** Forget the saved session. */
export async function clearSession(): Promise<void> {
    try {
        const root = await navigator.storage.getDirectory();
        await root.removeEntry(DIR, { recursive: true });
    } catch {
        /* nothing to clear */
    }
}
